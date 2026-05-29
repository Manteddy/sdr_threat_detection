# sdr_threat_detection — Project Definition

Project-level guide for engineers and AI agents (Claude / Cursor / etc.) working in this repo. Read this before touching code or generating large patches.

---

## 1. What this project is

`sdr_threat_detection` (a.k.a. `pluto_drone_detector`) is an SDR-based anti-drone detection system. It uses a PlutoSDR / AD9361 to listen for the wideband analog video transmissions used by FPV strike and reconnaissance drones in the 5.8 GHz ISM band (and other priority bands at 1.2 / 2.4 GHz). Output is a live spectrum + waterfall UI plus a separate proximity alert window that can also be exposed over WebSocket.

It implements **steps 3-4** of the broader pipeline documented in [product_pipeline.md](product_pipeline.md) (SDR acquisition + signal processing). Upstream RF capture (steps 1-2) and downstream Starlink relay / mission-control overlay (steps 5-10) are out of scope for this repo.

---

## 2. High-level architecture

```
PlutoSDR  ──►  SDRBackend.capture()  ──►  PSD (coarse FFT / fine Welch)
   │                  │
   │             SpectrumEngine.step()  ◄── Scheduler picks MeasurementCommand
   │                  │
   │           cells / groups / tracks                          (state)
   │                  │
   │           EngineSnapshot  ──►  Qt UI thread  ──►  spectrum + waterfall
   │                                       │
   │                                       └──►  AlertEngine  ──►  ProximityAlertPanel / WS
   │
   └─►  (legacy) SweepWorker fixed-step loop ──►  spectrum_omni/_dir buffers
```

Two coexisting processing paths:

1. **Adaptive engine path** (preferred, `spectrum_engine/`) — coarse-to-fine, CA-CFAR, scheduled revisits, track manager.
2. **Classic sweep path** (legacy, embedded in `drone_detector*.py`) — fixed-step LO sweep, single FFT, hybrid CFAR-or-fixed detection, used as backward-compatible fallback when the engine cannot attach.

The enhanced GUI in [drone_detector_enhanced.py](drone_detector_enhanced.py) drives whichever path is active and emits results to the same Qt widgets and to `proximity_alert`.

---

## 3. Codebase map

### Top-level
| Path | Role |
|------|------|
| [drone_detector.py](drone_detector.py) | Minimal version: single FFT + fixed threshold + fast sweep. Reference implementation, no engine. |
| [drone_detector_enhanced.py](drone_detector_enhanced.py) | Production GUI. Hosts `SweepWorker` (Qt thread) which runs either the adaptive engine loop or a classic sweep, plus `DistanceEstimator`, `FastlockManager`, hybrid detection. |
| [engine_config.yaml](engine_config.yaml) | Runtime knobs for `SpectrumEngine`: frequency range, hardware bandwidth, coarse/fine FFT, CFAR, scheduler, state thresholds, priority bands. |
| [requirements.txt](requirements.txt) | `pyadi-iio`, `numpy`, `pyqtgraph`, `PyQt5`. Optional: `numba`, `websockets`, `pyyaml`. |
| [README.md](README.md) | Operator-facing usage and feature summary. |
| [short_overview.md](short_overview.md) | Non-technical pitch. |
| [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) | Concept guide for Welch PSD / CA-CFAR / Kalman distance / hysteresis. |
| [signal_pipeline.md](signal_pipeline.md) | Long-form (~49 KB) walk-through of the **classic** RF → display pipeline. |
| [product_pipeline.md](product_pipeline.md) | End-to-end air → Starlink → goggles diagram. |

### `spectrum_engine/`
Adaptive Hierarchical Multiband Spectrum Sensing Engine. Phases 1-3 are implemented.

| Module | Role |
|--------|------|
| [engine.py](spectrum_engine/engine.py) | `SpectrumEngine.step()` orchestrates one measurement cycle. Owns cells, nodes, scheduler, track manager, telemetry. Returns `EngineSnapshot`. |
| [sdr_backend.py](spectrum_engine/sdr_backend.py) | `SDRBackend.capture(center_hz, bandwidth_hz, dwell_s, num_frames) -> IQCapture`. Single `rx()` per scan, paranoid about pyadi-iio buffer ordering. |
| [psd.py](spectrum_engine/psd.py) | `coarse_psd_db`, `fine_welch_db`, `channel_psd`, `freq_axis_hz`. Cached Blackman/Hann/Hamming windows. |
| [detectors.py](spectrum_engine/detectors.py) | `CoarseMeasurement`, `FineMeasurement`, `DetectedRegion`, **CA-CFAR** (`cfar_threshold_1d`), region merging, confidence scoring. |
| [spectrum_grid.py](spectrum_engine/spectrum_grid.py) | `SpectrumCell` dataclass + `CellState` enum (UNKNOWN/QUIET/SUSPECT/ACTIVE/TRACKED/STALE/UNSUPPORTED), `build_grid`, `map_measurement_to_cells`. |
| [scheduler.py](spectrum_engine/scheduler.py) | `Scheduler.choose_next_measurement` interleaves coverage scans (round-robin by staleness) with priority work (track revisits, group fine scans, priority bands). |
| [tracks.py](spectrum_engine/tracks.py) | `SignalTrack` lifecycle CANDIDATE → CONFIRMED → TRACKING → LOST → EXPIRED via `TrackManager`. |
| [occupancy.py](spectrum_engine/occupancy.py) | Per-cell occupancy probability (log-odds), state transitions, activity grouping. |
| [hierarchy.py](spectrum_engine/hierarchy.py) | Multi-resolution spectrum nodes for zoom-in decisions. |
| [baseline.py](spectrum_engine/baseline.py) | Slow EWMA noise-floor baseline, energy-delta tracking, uncertainty decay. |
| [telemetry.py](spectrum_engine/telemetry.py) | Append-only logs in `spectrum_logs/` (gitignored). |
| [config.py](spectrum_engine/config.py) | YAML loader → `EngineConfig` dataclasses with embedded defaults (works without PyYAML). |

### `proximity_alert/`
Embeddable alert package — no SDR dependency, safe to import standalone.

| Module | Role |
|--------|------|
| [engine.py](proximity_alert/engine.py) | `AlertEngine.update(freq_ghz, signal_dbfs, distance_m, confidence) -> ProximityAlert`. Trend detection, threat classification (NONE / DETECTED / APPROACHING / CRITICAL), boundary hysteresis. |
| [widget.py](proximity_alert/widget.py) | `ProximityAlertPanel` PyQt5 widget for embedding in any layout. |
| [ws_server.py](proximity_alert/ws_server.py) | Optional WebSocket bridge for web dashboards. Requires `websockets`. |
| [demo.py](proximity_alert/demo.py) | Standalone demo (`python -m proximity_alert`) with simulated signals. |

---

## 4. Running

```bash
pip install -r requirements.txt          # base
pip install numba websockets pyyaml      # all optional accelerators / features

python drone_detector.py                 # simple version
python drone_detector_enhanced.py        # production GUI (adaptive engine if available)
python -m proximity_alert                # alert demo, no hardware
```

The enhanced GUI auto-attaches `SpectrumEngine` when `spectrum_engine` imports cleanly; otherwise it falls back to the classic sweep loop and prints a status string in the engine status label. Engine telemetry lands in `spectrum_logs/`.

---

## 5. Hardware notes

- Targets PlutoSDR (ADALM-PLUTO) and AD9361-based clones. Stock Pluto only tunes 325 MHz - 3.8 GHz; **5.8 GHz needs Pluto+ (AD9363) or a modified unit**.
- Sample rate / RF BW driven from `engine_config.yaml.hardware`. Defaults are 40 MS/s, 40 MHz BW; coarse FFT 512 bins, fine FFT 4096 bins.
- `SDRBackend` is paranoid about pyadi-iio buffer ordering (destroy-then-retune-then-read). Don't reorder those calls.
- Dual-channel mode (omni + directional antenna) requires Pluto 2r2t firmware. Single-RX hardware mirrors omni into dir.

---

## 6. Conventions

- **Python 3.8+**. Code uses `from __future__ import annotations` and `dataclasses`.
- **NumPy first.** All hot paths are vectorised; SciPy is used sparingly (Welch / windowing rolled by hand in [psd.py](spectrum_engine/psd.py) to control the window-sum normalisation explicitly).
- **dBFS** is the default power unit (ratios to ADC full scale 2048). The classic detectors and the engine both feed dBFS through detection logic — keep new code in dBFS unless you have a reason to leave.
- **Frequencies in Hz** everywhere internal; convert to GHz only at the UI / alert boundary.
- **Immutable snapshots** cross thread boundaries (`EngineSnapshot` between worker thread and Qt main thread). Arrays are copies.
- **Telemetry / debug logs** go under `spectrum_logs/` (already gitignored). Never write logs to repo-tracked paths.
- **No test suite yet** — verify by running the GUI or `python -m proximity_alert`. The new SDR simulator (planned, see §8) is the first piece that will enable headless tests.

---

## 7. Existing CA-CFAR (legacy + engine)

For reference when extending detection — both implementations follow the same shape:

- Vectorised mean of a training window on each side of a Cell Under Test, **plus a guard region** to keep signal energy out of the noise estimate.
- Threshold = mean + `threshold_offset_db` (≈ 6 dB default). Above-threshold contiguous bins become regions; regions < `min_region_bw_hz` are dropped; pairs separated by < 1 MHz are merged.
- The classic detector in [drone_detector_enhanced.py](drone_detector_enhanced.py) additionally OR-s CA-CFAR with a fixed dBFS threshold as a safety fallback.

This is **mean-based CA-CFAR**. The new Signal Processing Pipeline (§8) layers in an **OS-CFAR** (Ordered-Statistic) variant that uses a percentile of the sorted reference window — more robust against interferers inside the training cells. The two should coexist; the existing CA-CFAR path is the engine's primary trigger and stays as-is.

---

## 8. Signal Processing Pipeline (new module, planned)

A deterministic feed-forward filter chain for **edge-side analog FPV classification**. It consumes the same IQ buffers produced by `SDRBackend.capture()` and emits structured detection telemetry. Target execution time per dwell buffer: **< 10 ms**.

### Stages

| # | Stage | Output |
|---|-------|--------|
| 1 | **Pre-processing** — split IQ into M segments of N (default 1024), Hamming window each, FFT, magnitude-squared, average → integrated PSD. | 1-D PSD array |
| 2 | **OS-CFAR trigger** — sliding window over the PSD: guard cells = 4 (2/side), reference cells = 16 (8/side), threshold = α × P_k where P_k is the 75th-percentile reference cell. Flag bins > threshold, cluster contiguous bins into ROIs. Early-exit if zero ROIs. | list of ROI bin ranges |
| 3 | **Feature extraction** per ROI — bandwidth (Δbin × fs/N), Shannon spectral entropy over normalised in-ROI power, sync-pulse detection by FFT of in-ROI envelope power looking for cyclic spikes at **15.625 kHz** (analog horizontal video sync) and **50 / 60 Hz** (frame refresh). | feature dict per ROI |
| 4 | **Classification** — hardcoded heuristic decision matrix: `ANALOG_FPV` (BW 6-10 MHz, moderate entropy, 15.625 kHz sync present), `BARRAGE_JAMMER` (BW > 12 MHz, very high entropy, no sync), `TELEMETRY_LINK` (BW < 1 MHz). Else `UNKNOWN`. | classification + confidence 0..1 |
| 5 | **Output** — dataclass: `frequency_hz`, `bandwidth_hz`, `signal_strength_db`, `classification`, `confidence_score`. | structured detection telemetry |

### Architectural priorities

- **Early-discard pattern.** Stage 2 returning zero ROIs must exit before stages 3-5 run.
- **Vectorised math.** Stays on NumPy / SciPy so it can be JIT'd (Numba) or moved to hardware-accelerated SIMD later without changing the call graph.
- **Decoupled state.** Pipeline takes one immutable IQ buffer in, returns one immutable telemetry struct out. Knows nothing about the sweep scheduler or SDR wrapper. This is what makes it droppable into [SpectrumEngine.step()](spectrum_engine/engine.py) without rewiring threading.

### Where it will live (planned)

A new package, tentatively `signal_pipeline/`, separate from `spectrum_engine/detectors.py`. The existing CA-CFAR is **not** being replaced — the new module is a parallel **classification** pass that consumes the same IQ buffers and complements the engine's track manager with semantic labels.

Integration plan: [signal_processing_integration_plan.md](signal_processing_integration_plan.md).
Offline test harness plan: [sdr_simulator_plan.md](sdr_simulator_plan.md).

---

## 9. Things to avoid

- Don't change the destroy-buffer-then-retune-then-read order in `SDRBackend._read_natural` — pyadi-iio breaks silently if you reorder them (errno-0 stalls, zero scans).
- Don't change `rx_buffer_size` inside `SDRBackend.capture()` — see the comment in [sdr_backend.py:131](spectrum_engine/sdr_backend.py:131); larger buffers change the sample layout and broke captures entirely in the past.
- Don't drop the DC-spike excision in `SpectrumEngine._process_capture` — PlutoSDR leaks LO into the centre FFT bin every capture, and removing the excision lights up every cell with a false centre signal.
- Don't widen the scheduler's priority cadence beyond `_PRIORITY_EVERY = 4` — small priority-band revisit intervals (0.5 s) used to monopolise the scheduler and starve coverage.
- Don't print to stdout from worker threads — use `error_occurred` signal or telemetry log.
