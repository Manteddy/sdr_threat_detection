# Implementation Plan — Sweep + Signal Processing Pipeline Integration

> Companion to: [sdr_threat_detection_ProjectDefinition.md §8](sdr_threat_detection_ProjectDefinition.md). Sibling plan: [sdr_simulator_plan.md](sdr_simulator_plan.md).

## 1. Context

The repo already has a working **sweep + detect** stack:

- [`SpectrumEngine.step()`](spectrum_engine/engine.py) drives the loop, scheduling coarse / fine measurements via [`Scheduler`](spectrum_engine/scheduler.py) and ingesting IQ through [`SDRBackend.capture()`](spectrum_engine/sdr_backend.py).
- Coarse and fine PSDs come out of [`spectrum_engine/psd.py`](spectrum_engine/psd.py).
- Fine measurements run **CA-CFAR** in [`spectrum_engine/detectors.py`](spectrum_engine/detectors.py) → `DetectedRegion`s → [`TrackManager`](spectrum_engine/tracks.py).
- `SignalTrack.classification` (line 67 of tracks.py) is currently always `"UNKNOWN"`.

The new **Signal Processing Pipeline** (project definition §8 + the user-provided spec) adds:
1. Hamming + averaged PSD pre-processing
2. **OS-CFAR** (Ordered-Statistic) trigger
3. Per-ROI feature extraction (bandwidth, Shannon entropy, sync-pulse 15.625 kHz / 50-60 Hz)
4. Heuristic classifier → `ANALOG_FPV` / `BARRAGE_JAMMER` / `TELEMETRY_LINK` / `UNKNOWN`
5. Structured `ClassificationResult` dataclass output

**Goal of this plan:** wire the new pipeline into the existing sweep loop so every fine scan also yields classification telemetry, *without* destabilising the proven CA-CFAR detection path or the < 10 ms per-buffer budget.

---

## 2. Integration shape (high level)

```
                                                  ┌──────────────────────────────┐
                                                  │  new: signal_pipeline/       │
                                                  │  ───────────────────────────  │
SpectrumEngine.step()                              │  preprocess() → PSD          │
   ├─ Scheduler.choose_next_measurement()         │  os_cfar() → ROIs            │
   ├─ SDRBackend.capture() ──► IQCapture ─────────▶│  extract_features(roi)       │
   ├─ channel_psd() ──► PSD                       │  classify(features)          │
   ├─ compute_fine_measurement() (CA-CFAR)        │  → ClassificationResult[]    │
   │           │                                  └──────────────┬───────────────┘
   │           ├─ DetectedRegion[]                                │
   │           │                                                  │
   │           └─ TrackManager.update_from_regions(...)           │
   │                                ▲                             │
   │                                └── merge classifications by  │
   │                                    frequency overlap ────────┘
   │
   └─► EngineSnapshot (adds: classifications: List[ClassificationResult])
                │
                ▼
       Qt UI + AlertEngine
```

Two boundary properties to preserve:

- **Trigger remains CA-CFAR.** The engine's existing CA-CFAR continues to drive cell occupancy and track creation. OS-CFAR runs in parallel as a *second opinion* feeding the classifier — it is not promoted to primary trigger in v1.
- **Pipeline is pure.** The new package has no thread/IO/Qt deps; it accepts IQ and config in, returns dataclasses out. The Qt worker thread still owns scheduling; the engine still owns state.

---

## 3. Module layout (new package)

```
signal_pipeline/
    __init__.py              # exports SignalPipeline, ClassificationResult, Classification
    pipeline.py              # SignalPipeline.run(iq, fs, center_hz) — orchestrates stages 1-5
    preprocess.py            # split-window-FFT-average → PSD (stage 1)
    os_cfar.py               # OS-CFAR sliding window → ROIs (stage 2)
    features.py              # bandwidth, spectral entropy, sync-pulse detection (stage 3)
    classifier.py            # heuristic decision matrix → label + confidence (stage 4)
    results.py               # @dataclass ClassificationResult, ROIFeatures, Classification enum (stage 5)
```

Public API (single entry-point):

```python
from signal_pipeline import SignalPipeline, ClassificationResult

pipeline = SignalPipeline(cfg=engine_cfg.signal_pipeline)
results: list[ClassificationResult] = pipeline.run(
    iq=capture.omni_iq.ravel(),           # complex64 1-D
    fs=cfg.hardware.sample_rate_hz,
    center_hz=capture.center_hz,
)
```

`ClassificationResult` carries: `frequency_hz`, `bandwidth_hz`, `signal_strength_db`, `classification` (enum), `confidence_score` (0..1), plus an internal `roi_bins` tuple and `features` dict for debugging.

### Reuse, do not rewrite

- **PSD pre-processing** mirrors the math already in [`psd.coarse_psd_db`](spectrum_engine/psd.py:55) but with **Hamming** (per spec) and explicit M segments × N samples shaping. Factor the windowed-FFT-average loop into a small helper if the call signatures differ enough that a shared function is awkward.
- **ROI extraction** can reuse the contiguous-run-finding pattern from [`cfar_detect_regions`](spectrum_engine/detectors.py:177) — copy the structure, swap CA threshold for OS threshold.
- **DC-spike excision** — apply the same median-replacement trick the engine already uses ([engine.py:300-303](spectrum_engine/engine.py:300)) so PlutoSDR's LO leakage does not become a false ROI.

---

## 4. Stage-by-stage implementation

### 4.1 Stage 1 — `preprocess.py`

```python
def integrated_psd(iq: np.ndarray, n: int = 1024, m: int | None = None,
                   window: str = "hamming") -> np.ndarray
```
- Truncate/reshape `iq` into `(M, N)`. If `m is None`, use `len(iq) // n`.
- Apply Hamming window to each row (cache the window like `psd._get_window`).
- FFT each row, `fftshift`, magnitude-squared.
- Mean across the M axis → 1-D PSD (linear power).
- Convert to dB: `10*log10(psd + eps)`.

### 4.2 Stage 2 — `os_cfar.py`

```python
def os_cfar(psd_db: np.ndarray, *, guard: int = 4, reference: int = 16,
            k_percentile: float = 75.0, alpha_db: float = 6.0) -> list[tuple[int, int]]
```
- For each CUT, gather the `reference` cells split evenly around the CUT, skipping the `guard` window.
- Sort and pick the **k-percentile** value (default 75th — spec). Vectorise with `np.partition` over a sliding view (`numpy.lib.stride_tricks.sliding_window_view`) to keep this under the budget.
- Threshold = `P_k + alpha_db`. Flag bins above threshold.
- Cluster contiguous flagged bins into `(start_bin, stop_bin)` tuples.
- Drop ROIs narrower than `min_roi_bins` (config). Merge ROIs with gap < `merge_gap_bins`.
- **Early-exit short-circuit:** if no ROIs, the caller skips stages 3-5. Keep the empty list as the cheap path.

### 4.3 Stage 3 — `features.py`

For each ROI:
- **Bandwidth**: `(stop_bin - start_bin + 1) * fs / n`.
- **Spectral entropy**: convert in-ROI PSD bins to linear power, normalise so they sum to 1, return `-sum(p * log2(p + eps))`.
- **Sync-pulse detection**: take the time-domain magnitude of the IQ within the ROI's frequency band (band-pass via `np.fft.ifft` of zeroed-out-of-band PSD bins, or downconvert+lowpass), compute envelope-power FFT (low resolution, e.g. 4096 pts), and probe for peaks at:
  - `15.625 kHz ± 200 Hz` (analog horizontal sync)
  - `50 Hz / 60 Hz ± 2 Hz` (frame refresh)
- Returns `ROIFeatures(bandwidth_hz, entropy_bits, sync_15625_db, sync_50_db, sync_60_db, peak_db, noise_floor_db)`.

### 4.4 Stage 4 — `classifier.py`

Hardcoded decision matrix from the spec:

| Class | Conditions |
|-------|------------|
| `ANALOG_FPV` | `6e6 ≤ bw ≤ 10e6` AND entropy mid-range (e.g. `3 < H < 6`) AND `sync_15625_db > sync_threshold_db` |
| `BARRAGE_JAMMER` | `bw > 12e6` AND entropy near-max (`H > 6.5`) AND no sync detected |
| `TELEMETRY_LINK` | `bw < 1e6` |
| `UNKNOWN` | otherwise |

Confidence is built from the count of matched conditions divided by the count tested (cap at 1.0). All thresholds live in `signal_pipeline.config.SignalPipelineCfg` so they can be tuned without code changes.

### 4.5 Stage 5 — `results.py`

```python
@dataclass(frozen=True)
class ClassificationResult:
    frequency_hz: float       # ROI centre
    bandwidth_hz: float
    signal_strength_db: float # peak in ROI (dBFS)
    classification: Classification  # enum
    confidence_score: float
    roi_bins: tuple[int, int]
    features: dict            # ROIFeatures as dict for telemetry
```

`Classification` is an `Enum` (`ANALOG_FPV`, `BARRAGE_JAMMER`, `TELEMETRY_LINK`, `UNKNOWN`).

---

## 5. Wiring into the existing engine

### 5.1 Config additions

Extend [`engine_config.yaml`](engine_config.yaml) with a `signal_pipeline:` section and add a matching dataclass in [`spectrum_engine/config.py`](spectrum_engine/config.py):

```yaml
signal_pipeline:
  enabled: true
  fft_size: 1024
  segments: 8
  window: hamming
  os_cfar:
    guard_cells: 4
    reference_cells: 16
    k_percentile: 75.0
    alpha_db: 6.0
    min_roi_bw_hz: 100000
    merge_gap_hz: 1000000
  classifier:
    fpv_bw_hz: [6000000, 10000000]
    jammer_bw_hz_min: 12000000
    telemetry_bw_hz_max: 1000000
    entropy_fpv_range: [3.0, 6.0]
    entropy_jammer_min: 6.5
    sync_threshold_db: 8.0
```

The fallback defaults in `config.py` should match these values so the engine still boots without YAML.

### 5.2 Hook in `SpectrumEngine._process_capture`

In [`engine.py`](spectrum_engine/engine.py) `_process_capture`, right after `compute_fine_measurement(...)` and before `track_mgr.update_from_regions(...)`:

```python
if cmd.is_fine and self._signal_pipeline is not None:
    iq_flat = capture.omni_iq.ravel()
    classifications = self._signal_pipeline.run(
        iq=iq_flat,
        fs=cfg.hardware.sample_rate_hz,
        center_hz=cmd.center_hz,
    )
    self._attach_classifications_to_regions(fine_meas.detected_regions, classifications)
```

`_attach_classifications_to_regions` matches by frequency overlap (use the same overlap test as `map_measurement_to_cells`) and copies `classification` + `confidence_score` into the matching `DetectedRegion`. When the track manager promotes a region to a `SignalTrack`, copy the classification onto `SignalTrack.classification` (already exists at [tracks.py:67](spectrum_engine/tracks.py:67)).

### 5.3 Pipeline lifecycle

Construct lazily in `SpectrumEngine.__init__`:

```python
from signal_pipeline import SignalPipeline
self._signal_pipeline: Optional[SignalPipeline] = (
    SignalPipeline(self.cfg.signal_pipeline) if self.cfg.signal_pipeline.enabled else None
)
```

A future "disable at runtime" toggle can flip `_signal_pipeline = None` from the GUI — same pattern the existing `FEATURES` flags use.

### 5.4 Snapshot surface

Add to [`EngineSnapshot`](spectrum_engine/engine.py:61):

```python
classifications: List[ClassificationResult] = field(default_factory=list)
```

Populate in `_make_snapshot` from `self._last_classifications` (set in `_process_capture`). Importantly: copy the list; do not share references with the worker thread state.

### 5.5 GUI + alert surface

[`drone_detector_enhanced.py`](drone_detector_enhanced.py) already routes `EngineSnapshot` to a slot. Extend that slot (around line 1802) to:
- Render classification labels next to detected tracks (small overlay or table column).
- Pass classification + confidence into [`AlertEngine.update(...)`](proximity_alert/engine.py:152) as additional fields (requires adding a `classification` parameter that defaults to `None` to keep the alert engine usable standalone).
- For `BARRAGE_JAMMER`: suppress the proximity alert promotion to CRITICAL (jamming is not approaching, it's just loud). Add a guard in `AlertEngine._classify_threat`.

### 5.6 Telemetry

Extend [`TelemetryLogger`](spectrum_engine/telemetry.py) with `log_classifications(classifications, ts)` so each fine scan's classification list lands in `spectrum_logs/classifications.log`. JSON-lines format, one row per classification.

---

## 6. Performance budget

Spec target: **< 10 ms per dwell buffer**. With default config (`fine_scan.num_frames=16`, `fine_scan.fft_size=4096`) this is the most expensive scan path.

- Stage 1 (PSD): ~1-2 ms (8 × 1024-point FFTs).
- Stage 2 (OS-CFAR): the hot loop. Avoid Python per-bin loops; use `sliding_window_view` + `np.partition` along the last axis. Aim for ~2-3 ms.
- Stages 3-5: only run if ROIs found; ~1 ms total in the common case (0-2 ROIs).
- Slack for engine overhead: ~3-4 ms.

Add a `time.perf_counter()` wrapper around `pipeline.run` and log mean / p95 to telemetry. If p95 exceeds 8 ms, the implementation lands a Numba kernel for OS-CFAR (parallel to the existing optional Numba accelerator in [drone_detector_enhanced.py:_cfar_threshold_numba](drone_detector_enhanced.py)).

---

## 7. Implementation order

1. **Scaffold `signal_pipeline/`** with all five module files, dataclasses, and empty function signatures. No behaviour yet — just import-clean.
2. **Stage 1 + Stage 2** end-to-end on a vector of pure noise — assert zero ROIs, confirm early-exit path.
3. **Stage 3 + Stage 4** against synthetic IQ from the simulator (see [sdr_simulator_plan.md](sdr_simulator_plan.md)): FPV scene → `ANALOG_FPV`, jammer → `BARRAGE_JAMMER`, telemetry → `TELEMETRY_LINK`.
4. **Config + engine wiring**: add `signal_pipeline:` to `engine_config.yaml`, the dataclass in `config.py`, lazy construct in `SpectrumEngine.__init__`.
5. **Hook in `_process_capture`** — gated by `self._signal_pipeline is not None`, executes only when `cmd.is_fine`.
6. **Snapshot field + GUI surface** — render labels, route to `AlertEngine`.
7. **Telemetry** — JSON-lines log of every classification with timestamp + scan_count.
8. **Performance pass** — measure, optionally JIT OS-CFAR with Numba.

Each step lands as its own commit / PR so review is reasonable.

---

## 8. Verification

- **Unit tests** (new `tests/` folder): per-stage tests that build synthetic PSDs / IQ and assert the expected ROI / feature / class. Use the simulator (sibling plan) as the IQ source — never call out to real hardware in tests.
- **Integration test**: spin up `SpectrumEngine` with the simulated `SDRBackend` (see sibling plan §4) and assert that running N fine scans against a known FPV scene yields ≥ M classifications matching `ANALOG_FPV` with confidence > 0.6.
- **Performance benchmark**: 1000-iteration loop, print mean / p50 / p95 / p99 for `pipeline.run`. Gate the integration on p95 < 8 ms.
- **Manual smoke**: run `drone_detector_enhanced.py` against the simulator, confirm classification labels appear in the GUI and alert engine flags FPV scenes as `DETECTED` and jammer scenes as low-severity.

---

## 9. Out of scope (deferred)

- Adaptive (non-heuristic) classifier — a small CNN over the in-ROI spectrogram is a natural Phase 2 but adds heavy dependencies; the heuristic spec is the deliverable.
- Replacing CA-CFAR with OS-CFAR as the engine's primary trigger. We may revisit once OS-CFAR has been validated in parallel for several weeks of telemetry.
- Direction finding from the dir-antenna channel — `pipeline.run` currently only consumes omni. The dir channel is available on `IQCapture.dir_iq` if/when DOA is added.
- Multi-emitter co-channel separation. The classifier assumes one emitter per ROI.
