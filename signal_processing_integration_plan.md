# Implementation Plan — OS-CFAR + Classifier Processor

> Companion to: [sdr_threat_detection_ProjectDefinition.md §8](sdr_threat_detection_ProjectDefinition.md).
> Builds on: [signal_processing_selector_plan.md](signal_processing_selector_plan.md) —
> the pluggable processor architecture, GUI Proc combo, and registry are
> already in place. This plan only covers the new OSCFARProcessor.
> Sibling plan: [sdr_simulator_plan.md](sdr_simulator_plan.md).

## 1. Context

The plug-in machinery in [signal_pipeline/](signal_pipeline/__init__.py)
landed first because we wanted to **preserve the proven CA-CFAR detector
as a fallback** rather than archive it. [`ClassicProcessor`](signal_pipeline/classic.py)
wraps `compute_fine_measurement` unchanged and is the default.

This plan adds **`OSCFARProcessor`** — the new five-stage pipeline the
user specified (Project Definition §8), shipped as a sibling plug-in. It
shows up in the GUI Proc combo, runs on every fine scan when selected,
and produces both `DetectedRegion`s (consumed by the existing
`TrackManager`) and `ClassificationResult`s (consumed by the GUI label /
AlertEngine routing in a follow-up).

## 2. Where it plugs in

The architecture work is done. New code only touches:

| Change | File |
|--------|------|
| New | `signal_pipeline/oscfar.py` (`OSCFARProcessor`, helper functions) |
| Modified | `signal_pipeline/registry.py` (append `OSCFARProcessor` to `_PROCESSORS`) |
| Modified | `engine_config.yaml` + `spectrum_engine/config.py` (new `signal_pipeline:` section with OS-CFAR knobs) |

The engine, the GUI combo, the runtime-swap mechanism, and
`FineMeasurement.classifications` already exist.

## 3. The five stages

### 3.1 Stage 1 — Preprocessing

The engine already hands the processor a Welch-averaged PSD via
`channel_psd(..., fine=True)`. That **is** the integrated PSD the spec
calls for; we use it directly. Hamming-vs-Blackman window choice is a
config knob (`signal_pipeline.window`) defaulting to `blackman` for
consistency with the rest of the engine, switchable to `hamming` per spec.

Sync-pulse detection in Stage 3 needs the raw time-domain IQ too, so
`process_fine` reads `capture.omni_iq.ravel()` once and keeps it.

### 3.2 Stage 2 — OS-CFAR sliding window

Per spec:
- Guard cells: 4 (2 / side)
- Reference cells: 16 (8 / side)
- Threshold = `P_k + α_db`, where `P_k` is the **k-th percentile** of the
  sorted reference window (default 75th).
- Above-threshold bins clustered into contiguous ROIs.
- Minimum ROI width = `min_roi_bw_hz` (default 100 kHz); pairs separated
  by < `merge_gap_hz` (default 1 MHz) merged.

Implementation: vectorise with `numpy.lib.stride_tricks.sliding_window_view`
+ `np.partition` so the percentile is computed across all CUTs in one
NumPy call. Edge bins use a shorter window (asymmetric) rather than
wrap-around — same convention as the existing `cfar_threshold_1d`.

Early-exit: if no ROIs, skip stages 3-5 and return a `FineMeasurement`
with the (empty) Welch threshold and empty classification list — no
wasted feature work.

### 3.3 Stage 3 — Per-ROI features

For each ROI:

- **Bandwidth** = `(stop_bin - start_bin + 1) * fs / fft_size`.
- **Shannon spectral entropy** = `-Σ p_i log₂ p_i` where `p_i` is the
  in-ROI PSD bin normalised to sum to 1 (converted to linear power first).
- **Sync-pulse detection**:
  1. Take a copy of the original PSD spectrum, zero out everything
     outside the ROI, IFFT → time-domain band-limited signal `x_roi(t)`.
  2. Compute envelope power `|x_roi(t)|²`.
  3. Low-resolution FFT of the envelope (4096 bins).
  4. Probe for peaks at **15.625 kHz ± 200 Hz** (horizontal sync),
     **50 Hz ± 2 Hz** and **60 Hz ± 2 Hz** (frame refresh).
  5. Each detection reports a SNR-like "peak above local noise" in dB.

The fine-scan buffer at default config (`num_frames=16`, `fft_size=4096`
→ 65 536 samples / 1.6 ms at 40 MS/s) covers ~25 horizontal sync periods,
so the 15.625 kHz line is resolvable. The 50/60 Hz vertical line requires
≥ one full frame — at 1.6 ms we get ~0.1 frames, so vertical sync needs a
longer fine scan to register. Document the limitation; classifier falls
back to horizontal-sync-only confidence.

### 3.4 Stage 4 — Heuristic classifier

Hardcoded decision matrix:

| Class | Conditions |
|-------|------------|
| `ANALOG_FPV` | `fpv_bw_min ≤ bw ≤ fpv_bw_max` AND `entropy_fpv_min < H < entropy_fpv_max` AND `sync_15625_db > sync_threshold_db` |
| `BARRAGE_JAMMER` | `bw > jammer_bw_min` AND `H > entropy_jammer_min` AND no sync detected |
| `TELEMETRY_LINK` | `bw < telemetry_bw_max` |
| `UNKNOWN` | otherwise |

Confidence = matched-conditions / tested-conditions, capped at 1.0. All
thresholds come from the `signal_pipeline:` config block.

### 3.5 Stage 5 — Output

Each ROI's classification is a `ClassificationResult` (already defined in
[signal_pipeline/base.py](signal_pipeline/base.py)). The processor
populates `FineMeasurement.classifications` (already a field after the
selector work) with one entry per ROI.

## 4. Config

Append to [engine_config.yaml](engine_config.yaml):

```yaml
signal_pipeline:
  window: blackman
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

Add a matching `SignalPipelineCfg` dataclass in
[spectrum_engine/config.py](spectrum_engine/config.py) with the same
default values, so the engine still boots without YAML / PyYAML and a
processor can read the cfg via the existing `EngineConfig.signal_pipeline`
attribute.

## 5. Performance budget

Spec target: **< 10 ms per dwell buffer** for the fine path (the new code
budget — the engine's own PSD + cell-update work is separate).

- Stage 2 (OS-CFAR) is the hot loop. Vectorised partition should keep it
  ~1-2 ms at `fft_size = 4096`.
- Stage 3 envelope FFT and sync probe: ~1 ms per ROI; capped by the
  early-exit when no ROIs.
- Stages 4-5: trivial.

Add a `time.perf_counter()` instrument around `process_fine` and emit
mean / p95 to telemetry. If p95 exceeds the budget, drop in a Numba
kernel for OS-CFAR — same shape as the existing
`_cfar_threshold_numba` accelerator in `drone_detector_enhanced.py`.

## 6. Implementation order

1. **Scaffold** `signal_pipeline/oscfar.py` with class skeleton and
   helper signatures, no behaviour.
2. **Stage 2 (OS-CFAR)** with a noise-only PSD test — assert zero ROIs.
3. **Stage 3 features** against synthetic FPV IQ produced by the
   simulator (`scenarios.fpv_at(5.8e9)`) — assert 15.625 kHz peak is
   detected, entropy in the moderate range.
4. **Stage 4 classifier** — assert FPV scene → `ANALOG_FPV`, jammer
   scene → `BARRAGE_JAMMER`.
5. **Config plumbing** — new dataclass, YAML extension, defaults.
6. **Register** in `signal_pipeline/registry.py` so the GUI Proc combo
   shows it.
7. **Telemetry** — append per-scan classifications to
   `spectrum_logs/classifications.log`.
8. **Performance pass** — measure, optionally JIT Stage 2.

Each step a separate commit.

## 7. Verification

- **Headless test, FPV scene** (`scenarios.fpv_at(5.8e9, power_dbfs=-30)`):
  switch engine processor to `OSCFARProcessor`, run ≥ 200 fine scans,
  assert at least one `ClassificationResult` with `classification ==
  "ANALOG_FPV"` and `confidence_score > 0.5`.
- **Jammer scene** (`scenarios.jammer_at(2.45e9)`): switch processor,
  assert `BARRAGE_JAMMER` classification — this is the jammer-blind-spot
  that motivated the new processor in the first place; passing this is
  the headline win.
- **Default-path regression**: leave engine processor as Classic, repeat
  the standard FPV @ 5.8 GHz cell-occupancy assertion. Must be unchanged.
- **GUI**: run the enhanced detector against the simulator, flip the
  Proc combo Classic ↔ OS-CFAR mid-run — no crash, status label updates.
- **Performance**: p95 of `OSCFARProcessor.process_fine` < 8 ms over
  1000 iterations at default fine-scan config.

## 8. Out of scope (deferred)

- **Routing classifications into `AlertEngine`** — needs a new field on
  `ProximityAlert` and threat-rule changes (e.g. don't escalate
  `BARRAGE_JAMMER` to `CRITICAL`). Lands as a separate plan after the
  processor itself is proven against live signals.
- **Snapshot surfacing** — `EngineSnapshot.classifications` field plus a
  small GUI overlay. Out of scope until the AlertEngine routing decision
  is made; until then, `_dump_detection_log` in
  [`SpectrumEngine`](spectrum_engine/engine.py) can be extended to log
  classifications for debugging.
- **OS-CFAR as primary trigger replacing CA-CFAR** — explicitly *not*
  done. Both run side-by-side; the operator picks via the Proc combo.
  Promotion is a separate decision once telemetry shows OS-CFAR is at
  least as good across the regression scenarios.
- **Multi-emitter co-channel separation** — the classifier assumes one
  emitter per ROI. Multi-emitter rejection / splitting is a Phase 2
  refinement.
