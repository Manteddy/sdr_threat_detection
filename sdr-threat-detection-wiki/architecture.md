# Architecture

## Stack

- **Python 3.8+** — `from __future__ import annotations`, `dataclasses` throughout
- **NumPy** — all hot paths vectorised; SciPy avoided (windowing rolled by hand in `psd.py`)
- **PyQt5 + pyqtgraph** — GUI and live plots
- **pyadi-iio** — hardware path only (PlutoSDR / AD9361); not required for simulator or replay
- **pyyaml** — runtime config (`engine_config.yaml`)
- **numba** (optional) — JIT acceleration for CFAR

## Directory Structure

```
sdr_threat_detection/
├── drone_detector.py              # Minimal reference: single FFT + fixed threshold (no engine)
├── drone_detector_enhanced.py     # Production GUI — SweepWorker, all controls, panels
├── engine_config.yaml             # Runtime knobs (freq range, FFT sizes, CFAR, scheduler, states, experiments)
├── requirements.txt               # Core deps; pyadi-iio hardware-only; numba optional
├── run.sh                         # Launcher — calls .venv/bin/python, no activate needed
│
├── spectrum_engine/               # Adaptive Hierarchical Multiband Spectrum Sensing Engine
│   ├── engine.py                  # SpectrumEngine.step() — orchestrates one measurement cycle
│   ├── iq_source.py               # IQSource ABC + IQCapture / HardwareLimits dataclasses
│   ├── signal_reader.py           # SignalReader.capture() — single capture fn for hw + sim + replay
│   ├── psd.py                     # coarse_psd_db, fine_welch_db, channel_psd, freq_axis_hz
│   ├── detectors.py               # CoarseMeasurement, FineMeasurement, cfar_threshold_1d, regions
│   ├── spectrum_grid.py           # SpectrumCell / CellState, build_grid, map_measurement_to_cells
│   ├── scheduler.py               # Scheduler.choose_next_measurement — coverage + priority interleave
│   ├── tracks.py                  # SignalTrack lifecycle + TrackManager
│   ├── occupancy.py               # Per-cell log-odds occupancy, state transitions, activity groups
│   ├── hierarchy.py               # Multi-resolution spectrum nodes for zoom-in decisions
│   ├── baseline.py                # Slow EWMA noise-floor baseline
│   ├── telemetry.py               # Append-only logs → spectrum_logs/ (gitignored)
│   ├── config.py                  # YAML loader → EngineConfig dataclasses
│   ├── sources/
│   │   ├── pyadi.py               # PyAdiIQSource — PlutoSDR via pyadi-iio
│   │   └── replay.py              # ReplayIQSource — time-locked playback of recorded experiments
│   └── sim/                       # Synthetic IQ source + scene preview (no pyadi-iio)
│       ├── iq_source.py           # SimulatedIQSource — tune() stores LO, acquire() builds IQ from Scene
│       ├── scene.py               # Scene, Emitter, EmitterClass dataclasses
│       ├── synth.py               # gen_awgn, gen_analog_fpv, gen_barrage_jammer
│       ├── scenarios.py           # Preset factories + GUI_PRESETS list
│       └── preview.py             # scene_psd_db() — analytical full-range PSD for Sim Preview panel
│
├── signal_pipeline/               # Pluggable signal processors (runtime-swappable via GUI Proc combo)
│   ├── base.py                    # SignalProcessor base class + ClassificationResult dataclass
│   ├── classic.py                 # ClassicProcessor — CA-CFAR (default, stable fallback)
│   ├── oscfar.py                  # OSCFARProcessor — OS-CFAR + features + heuristic classifier
│   └── registry.py                # list_processors(), get_processor(name), default_processor()
│
├── experiments/                   # Recording and replay package
│   ├── __init__.py                # Exports ExperimentRecorder, RecordingOptions
│   └── recorder.py                # ExperimentRecorder — background I/O, int16 IQ, sweep log, heatmaps
│
├── proximity_alert/               # Embeddable alert package — no SDR dependency
│   ├── engine.py                  # AlertEngine.update() → ProximityAlert (NONE/DETECTED/APPROACHING/CRITICAL)
│   ├── widget.py                  # ProximityAlertPanel PyQt5 widget
│   ├── ws_server.py               # Optional WebSocket bridge (requires websockets)
│   └── demo.py                    # Standalone demo: python -m proximity_alert
│
└── spectrum_logs/                 # Runtime telemetry (gitignored)
```

## Data Flow

```
Hardware (PlutoSDR) ──► PyAdiIQSource ───┐
                                         ├──► SignalReader.capture()
Simulator     ──► SimulatedIQSource ─────┤    (tune → settle → acquire → normalise →
                                         │     anti-alias → frame → mirror)
Recording     ──► ReplayIQSource ────────┘    (time-locked: scheduler overridden by
                                               recorded MeasurementCommands)
                                         │
                                         ▼
                               SpectrumEngine.step()
                               ├── Scheduler picks MeasurementCommand (or ReplayIQSource vends it)
                               ├── RECEIVE: channel_psd() + display buffer
                               ├── PROCESS: cells + groups + occupancy
                               └── CLASSIFY: processor.process_fine() → FineMeasurement
                                            ├── DetectedRegion[]
                                            └── ClassificationResult[]
                                         │
                                         ▼
                               Qt main thread (drone_detector_enhanced.py)
                               ├── RX1 spectrum + waterfall (live 20 MHz window)
                               ├── Sim Preview panel (analytical 1–6.5 GHz ground truth)
                               ├── Experiment browser dock (past recordings)
                               └── AlertEngine → ProximityAlertPanel / WebSocket
```

## Key Entry Points

| File | Role |
|------|------|
| `run.sh` | Operator launch point |
| `drone_detector_enhanced.py` | Production GUI — Qt `QApplication`, `SweepWorker` thread |
| `spectrum_engine/engine.py` | `SpectrumEngine.step()` — one measurement cycle |
| `spectrum_engine/signal_reader.py` | `SignalReader.capture()` — the single capture contract |
| `signal_pipeline/registry.py` | Processor plug-in entry point |
| `experiments/recorder.py` | `ExperimentRecorder` — recording entry point |
| `spectrum_engine/sources/replay.py` | `ReplayIQSource` — replay entry point |

## Plan Docs

| Plan file | Scope |
|-----------|-------|
| `sdr_simulator_plan.md` | Synthetic SDR backend design |
| `signal_processing_selector_plan.md` | Plug-in processor registry |
| `signal_processing_integration_plan.md` | OS-CFAR + classifier processor |
| `signal_reader_refactor_plan.md` | One-`capture()` architecture |
| `gui_stage_ladder_plan.md` | 4-stage operating ladder |
| `experiment_recording_plan.md` | Experiment recording and replay |

## Engine Stage Contract

`EngineStage` is the single knob that controls how deep `step()` goes. Each stage is a strict superset of those below it:

| Stage | What the engine does |
|-------|---------------------|
| `IDLE` | Nothing (worker stopped) |
| `RECEIVE` | Scheduler + tune + acquire → drop IQ (confirms data flowing) |
| `PROCESS` | + PSD + display buffer + cell/occupancy state (spectrum + waterfall live) |
| `CLASSIFY` | + `processor.process_fine()` + TrackManager (full detection) |

Transitioning **down** a stage triggers cleanup: Classify→Process drops tracks; Process→Receive resets cells.

## Three Orthogonal Runtime Axes (GUI)

| Axis | Choices | How switched |
|------|---------|-------------|
| IQ source | Hardware / Simulator / Replay | `Src:` combo — snaps to Idle on change |
| Detection algorithm | Classic ↔ OS-CFAR | `Proc:` combo → `SpectrumEngine.set_processor()` |
| Pipeline depth | Idle / Receive / Process / Classify | 4-segment stage ladder |

## Adding a New Signal Processor

1. Write a class in `signal_pipeline/` inheriting `SignalProcessor`; set `name` and `label`; implement `process_fine(capture, psd_db, freq_axis_hz, cfg) -> FineMeasurement`.
2. Append it to `_PROCESSORS` in `signal_pipeline/registry.py`.
3. Done — the GUI Proc combo picks it up automatically.

The engine, scheduler, track manager, and GUI need no changes. Existing processors are preserved as fallbacks.

**Performance budget:** < 10 ms per fine-scan buffer. Measured: `OSCFARProcessor.process_fine` p95 ≈ 2.5 ms on a 2026-era MacBook (4096-bin fine PSD); Classic is sub-ms.

## Empirical Signal-Processing Results

- All 7 simulator presets classify correctly with `OSCFARProcessor`.
- OS-CFAR's 16-cell reference window (≈ 78 kHz) sits inside an 8 MHz FPV emitter, so the raw OS-CFAR threshold is too high for wideband signals. The global noise-floor fallback (30th-percentile + α) catches what the local estimator cannot.
- Shannon entropy is not a useful discriminator at this resolution — FM video and band-limited noise both spread energy uniformly. v1 classifier uses bandwidth + sync-pulse presence only; entropy is kept for telemetry/tuning.
- FPV transmitters are constant-envelope (pure FM), but residual AM at the sync tip produces the 15.625 kHz envelope-power line the classifier detects. `gen_analog_fpv` in `synth.py` reproduces this intentionally.

## Key Constraints

See [footguns.md](footguns.md) for the full load-bearing list. Short summary:

- `IQSource` implementations do **raw IQ only** — no normalisation, framing, or anti-alias inside a source.
- `SignalProcessor.process_fine()` must be **stateless across calls**.
- `signal_pipeline` must **not** import from `spectrum_engine.engine` at module top.
- `EngineSnapshot` objects cross worker→Qt thread boundary as **immutable copies**.
