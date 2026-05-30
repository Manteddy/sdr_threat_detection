# sdr_threat_detection — Project Definition

Project-level guide for engineers and AI agents (Claude / Cursor / etc.) working in this repo. Read this before touching code or generating large patches. See [§11](#11-how-this-document-stays-current) for the rules that keep this doc in sync with the codebase.

---

## 1. What this project is

`sdr_threat_detection` (a.k.a. `pluto_drone_detector`) is an SDR-based anti-drone detection system. It uses a PlutoSDR / AD9361 to listen for the wideband analog video transmissions used by FPV strike and reconnaissance drones across **1.0 - 6.5 GHz**, with priority on the 1.2 / 2.4 / 5.8 GHz bands. Output is a live spectrum + waterfall UI, a ground-truth simulator preview, and a separate proximity alert window that can also be exposed over WebSocket.

It implements **steps 3-4** of the broader pipeline documented in [product_pipeline.md](product_pipeline.md) (SDR acquisition + signal processing). Upstream RF capture (steps 1-2) and downstream Starlink relay / mission-control overlay (steps 5-10) are out of scope.

Operationally there are **two data sources** (hardware or simulator) and **two detection algorithms** (Classic CA-CFAR or OS-CFAR + classifier), all switchable at runtime from the GUI — see §9.

---

## 2. High-level architecture

```
Hardware ─► PyAdiIQSource ─┐                       ┌─────────────────────────┐
 (Pluto)                   │                       │  signal_pipeline/       │
                           ├─► SignalReader ──IQ───┤  (plug-in processors)   │
Simulator ─► Simulated     │   (one capture fn)    │  ──────────────────────  │
 (Mac/CI)    IQSource  ────┘                       │  base.SignalProcessor   │
                                                   │  classic.ClassicProc    │
                                                   │  oscfar.OSCFARProc      │
                                                   │  registry.get/list      │
                                                   └────────┬────────────────┘
            ┌───────────────────────────────────────────────┘
            │
            │   SpectrumEngine.step()                  (stage-gated)
            │   ├─ Scheduler picks MeasurementCommand
            │   ├─ reader.capture() ─► IQCapture                  ┐
            │   │     (tune → settle → acquire → normalise →     │ RECEIVE
            │   │      anti-alias → frame → single-antenna mirror)│ and up
            │   ├─ channel_psd() + cells + groups + display buffer ┐ PROCESS
            │   │                                                  │ and up
            │   ├─ processor.process_fine() ─► FineMeasurement     ┐ CLASSIFY
            │   │     ├─ DetectedRegion[]                          │ only
            │   │     └─ ClassificationResult[]                    │
            │   └─ emit EngineSnapshot
            │
            ▼
   Qt main thread (drone_detector_enhanced.py)
      ├─ Spectrum + waterfall (RX1) — live receiver view
      ├─ Sim Preview panel — analytical ground-truth scene PSD (1-6.5 GHz)
      ├─ AlertEngine ─► ProximityAlertPanel / WebSocket
      └─ Top-bar controls: Src / Scene / Proc / Connect / Detect
```

`SignalReader` is the **single** capture function the engine sees. It is
the same code path for hardware and simulator — `IQSource`s do raw IQ
acquisition only; everything else (normalisation to dBFS, anti-alias,
framing, dual-mirror, dwell budget) lives in the reader. That is the
contract that makes "signal is signal" hold.

`EngineStage` is the **single knob** that controls how deep `step()` goes:

| Stage | Engine does | Use |
|------|-------------|-----|
| `IDLE` | nothing (worker stopped) | Engine off. |
| `RECEIVE` | scheduler + tune + acquire → drop IQ | Diagnostic: confirm data is flowing. |
| `PROCESS` | + PSD + display buffer + cell state | Spectrum + waterfall live, no classification. |
| `CLASSIFY` | + processor.process_fine + TrackManager | Full detection pipeline (today's default). |

Strict superset: each stage runs everything from the stages below it.

Three orthogonal axes the GUI exposes:

| Axis | Choices | Mechanism |
|------|---------|-----------|
| Source of IQ | Hardware ↔ Simulator | `SimulatedSDRBackend` is duck-typed to `SDRBackend` |
| Detection algorithm | Classic CA-CFAR ↔ OS-CFAR + Classifier | `signal_pipeline` registry + `SpectrumEngine.set_processor` |
| Detection on/off | Detect OFF ↔ ON | `SpectrumEngine.detect_enabled` flag |

The receiver path is unchanged across simulator vs hardware. The only thing that differs is which `*Backend` is wired in.

---

## 3. Codebase map

### Top-level
| Path | Role |
|------|------|
| [drone_detector.py](drone_detector.py) | Minimal version: single FFT + fixed threshold + fast sweep. Reference implementation, no engine. |
| [drone_detector_enhanced.py](drone_detector_enhanced.py) | Production GUI. Hosts `SweepWorker` (Qt thread), the Src/Scene/Proc/Detect controls, the RX1 spectrum + waterfall, and the new Sim Preview panel. |
| [engine_config.yaml](engine_config.yaml) | Runtime knobs for `SpectrumEngine`: frequency range, hardware bandwidth, coarse/fine FFT, CFAR, scheduler, state thresholds, priority bands. |
| [requirements.txt](requirements.txt) | `pyadi-iio` (hardware-only), `numpy`, `pyqtgraph`, `PyQt5`. Optional: `numba`, `websockets`, `pyyaml`. |
| [README.md](README.md) | Operator-facing usage and feature summary. |
| [short_overview.md](short_overview.md) | Non-technical pitch. |
| [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) | Concept guide for Welch PSD / CA-CFAR / Kalman distance / hysteresis. |
| [signal_pipeline.md](signal_pipeline.md) | Long-form (~49 KB) walk-through of the **classic** RF → display pipeline. |
| [product_pipeline.md](product_pipeline.md) | End-to-end air → Starlink → goggles diagram. |
| [signal_processing_selector_plan.md](signal_processing_selector_plan.md) | Plug-in architecture for swappable signal processors. |
| [signal_processing_integration_plan.md](signal_processing_integration_plan.md) | OS-CFAR + classifier processor on top of the plug-in foundation. |
| [sdr_simulator_plan.md](sdr_simulator_plan.md) | Synthetic SDR backend design (now implemented; doc still useful for context). |
| [signal_reader_refactor_plan.md](signal_reader_refactor_plan.md) | One-`capture()` architecture across hardware + simulator. |
| [gui_stage_ladder_plan.md](gui_stage_ladder_plan.md) | 4-stage operating ladder (Idle / Receive / Process / Classify) + segmented control replacing Connect + Detect. |

### `spectrum_engine/`
Adaptive Hierarchical Multiband Spectrum Sensing Engine.

| Module | Role |
|--------|------|
| [engine.py](spectrum_engine/engine.py) | `SpectrumEngine.step()` orchestrates one measurement cycle. Owns cells, scheduler, tracks, telemetry, the active `SignalProcessor`, and the `detect_enabled` flag. Public hooks: `attach_backend(SignalReader)`, `set_processor`, `set_detection_enabled`, `reset_detection_state`. Returns `EngineSnapshot`. |
| [iq_source.py](spectrum_engine/iq_source.py) | `IQSource` ABC + canonical `IQCapture` / `HardwareLimits` dataclasses. Sources only do raw IQ acquisition; framing / normalisation / etc. is the reader's job. |
| [signal_reader.py](spectrum_engine/signal_reader.py) | `SignalReader.capture(...)` — the single capture function for both hardware and simulator. Owns tune-settle, dBFS normalisation, anti-alias filtering, framing, single-antenna mirror (`dir_iq = omni_iq.copy()`), and the realtime dwell budget. |
| [sources/pyadi.py](spectrum_engine/sources/pyadi.py) | `PyAdiIQSource` — `tune()` via `rx_lo` (destroy-buffer-first), `acquire()` via `rx()`. `rx_buffer_size` is sized once at init to `fft_size_base × max(fine_frames, coarse_frames)` to lift the historic 2048-sample cap. Single-channel. |
| [psd.py](spectrum_engine/psd.py) | `coarse_psd_db`, `fine_welch_db`, `channel_psd`, `freq_axis_hz`. Cached Blackman/Hann/Hamming windows. |
| [detectors.py](spectrum_engine/detectors.py) | `CoarseMeasurement`, `FineMeasurement` (now carries an optional `classifications` list), `DetectedRegion`, mean-based CA-CFAR (`cfar_threshold_1d`), region merging. |
| [spectrum_grid.py](spectrum_engine/spectrum_grid.py) | `SpectrumCell` dataclass + `CellState` enum, `build_grid`, `map_measurement_to_cells`. |
| [scheduler.py](spectrum_engine/scheduler.py) | `Scheduler.choose_next_measurement` interleaves coverage scans with priority work (track revisits, group fine scans, priority bands). |
| [tracks.py](spectrum_engine/tracks.py) | `SignalTrack` lifecycle CANDIDATE → CONFIRMED → TRACKING → LOST → EXPIRED via `TrackManager`. `clear_all()` drops every track. |
| [occupancy.py](spectrum_engine/occupancy.py) | Per-cell occupancy probability (log-odds), state transitions, activity grouping. |
| [hierarchy.py](spectrum_engine/hierarchy.py) | Multi-resolution spectrum nodes for zoom-in decisions. |
| [baseline.py](spectrum_engine/baseline.py) | Slow EWMA noise-floor baseline, energy-delta, uncertainty. |
| [telemetry.py](spectrum_engine/telemetry.py) | Append-only logs in `spectrum_logs/` (gitignored). |
| [config.py](spectrum_engine/config.py) | YAML loader → `EngineConfig` dataclasses with embedded defaults. |

### `spectrum_engine/sim/` — synthetic IQ source + scene preview
Synthetic `IQSource` consumed by `SignalReader`, plus the ground-truth preview synthesizer. **No pyadi-iio dependency** so the whole stack runs headless on macOS.

| Module | Role |
|--------|------|
| [sim/iq_source.py](spectrum_engine/sim/iq_source.py) | `SimulatedIQSource` — `tune()` stores LO, `acquire()` builds IQ from the active `Scene` at raw ADC scale so `SignalReader` normalises both sources identically. Thread-safe `set_scene` for live GUI swaps. |
| [sim/scene.py](spectrum_engine/sim/scene.py) | `Scene`, `Emitter`, `EmitterClass` dataclasses. |
| [sim/synth.py](spectrum_engine/sim/synth.py) | `gen_awgn`, `gen_analog_fpv` (FM video with sync-tip AM dip so 15.625 kHz survives in the envelope-power FFT), `gen_barrage_jammer` (band-limited noise). |
| [sim/scenarios.py](spectrum_engine/sim/scenarios.py) | Preset factories: `empty_band`, `fpv_at(hz)`, `jammer_at(hz)`, plus `GUI_PRESETS` driving the Scene combo. |
| [sim/preview.py](spectrum_engine/sim/preview.py) | `scene_psd_db(scene, start, stop, n_bins)` — analytical full-range PSD. Used by the GUI Sim Preview panel; no IQ involved. |

### `signal_pipeline/` — pluggable signal processors
Multiple fine-scan algorithms live side-by-side and are selectable at runtime through the GUI Proc combo.

| Module | Role |
|--------|------|
| [base.py](signal_pipeline/base.py) | `SignalProcessor` base class (`name`, `label`, `process_fine`). `ClassificationResult` dataclass. |
| [classic.py](signal_pipeline/classic.py) | `ClassicProcessor` wraps the existing `compute_fine_measurement` unchanged — current default, guaranteed-stable fallback. |
| [oscfar.py](signal_pipeline/oscfar.py) | `OSCFARProcessor`: integrated PSD → OS-CFAR (75th-percentile sliding window + global noise-floor fallback for wideband emitters) → per-ROI bandwidth/entropy/sync features → heuristic `ANALOG_FPV / BARRAGE_JAMMER / TELEMETRY_LINK / UNKNOWN` classifier. |
| [registry.py](signal_pipeline/registry.py) | `list_processors()`, `get_processor(name)`, `default_processor()`. Hand-maintained list; first entry is the default. |

### `proximity_alert/`
Embeddable alert package — no SDR dependency, safe to import standalone.

| Module | Role |
|--------|------|
| [engine.py](proximity_alert/engine.py) | `AlertEngine.update(...)` returns `ProximityAlert` (NONE / DETECTED / APPROACHING / CRITICAL + trend + boundary hysteresis). |
| [widget.py](proximity_alert/widget.py) | `ProximityAlertPanel` PyQt5 widget. |
| [ws_server.py](proximity_alert/ws_server.py) | Optional WebSocket bridge. Requires `websockets`. |
| [demo.py](proximity_alert/demo.py) | Standalone demo (`python -m proximity_alert`). |

---

## 4. Running

### Mac — simulator only (no hardware required)

```bash
# one-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyqtgraph PyQt5 pyyaml
deactivate            # optional — run.sh doesn't need an active venv

# every run
./run.sh
```

`pyadi-iio` is intentionally skipped — `libiio` is awkward on macOS and the simulator does not need it. In the GUI: **Src** → `Simulator`, pick a **Scene**, click up the stage ladder: **Receive** → **Process** → **Classify**.

### Linux — real Pluto (full hardware path)

```bash
# one-time setup
sudo apt install -y libiio0 libiio-utils python3-libiio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pyyaml             # YAML config support
pip install numba              # optional CFAR JIT
deactivate

# every run
./run.sh
```

In the GUI: **Src** → `Hardware`, click **Receive** (or any stage up to Classify). (See troubleshooting in [README.md](README.md) §Requirements for udev rules etc.)

### About `run.sh`

The launcher (`run.sh` at the repo root) calls `.venv/bin/python` directly so you don't need to `source .venv/bin/activate` every shell session. It also `cd`s into the repo so `engine_config.yaml` and `spectrum_logs/` resolve correctly regardless of where you invoke it from. If `.venv/bin/python` is missing, the script prints the one-time setup steps and exits.

### Headless / scripted (no Qt)

```python
import time
from spectrum_engine import SpectrumEngine
from spectrum_engine.sim import SimulatedSDRBackend, scenarios
from signal_pipeline import OSCFARProcessor

eng = SpectrumEngine(telemetry_enabled=False)
eng.set_processor(OSCFARProcessor())
eng.attach_backend(SimulatedSDRBackend(eng.cfg, scenarios.fpv_at(5.8e9)))
eng.set_active_range(eng.cfg.frequency_range.start_hz, eng.cfg.frequency_range.stop_hz)
for _ in range(800):
    snap = eng.step(now=time.monotonic())
```

This is the same path the integration tests will take.

---

## 5. Hardware notes

- Targets PlutoSDR (ADALM-PLUTO) and AD9361-based clones. Stock Pluto only tunes 325 MHz - 3.8 GHz; **5.8 GHz needs Pluto+ (AD9363) or a modified unit**.
- Sample rate / RF BW driven from `engine_config.yaml.hardware`. Defaults are 40 MS/s, 40 MHz BW; coarse FFT 512 bins, fine FFT 4096 bins.
- `SDRBackend` is paranoid about pyadi-iio buffer ordering (destroy-then-retune-then-read). Don't reorder those calls.
- Dual-channel mode (omni + directional antenna) requires Pluto 2r2t firmware. Single-RX hardware mirrors omni into dir.
- macOS Docker Desktop **cannot** map a Pluto over USB. Real-hardware deployment is Linux-only; everything else (sim, dev, CI) runs anywhere.

---

## 6. Conventions

- **Python 3.8+**. Code uses `from __future__ import annotations` and `dataclasses`.
- **NumPy first.** All hot paths are vectorised; SciPy is used sparingly (Welch / windowing rolled by hand in [psd.py](spectrum_engine/psd.py) to control the window-sum normalisation explicitly).
- **dBFS** is the default power unit (ratios to ADC full scale 2048). Keep new code in dBFS unless you have a reason to leave.
- **Frequencies in Hz** everywhere internal; convert to GHz only at the UI / alert boundary.
- **Immutable snapshots** cross thread boundaries (`EngineSnapshot` between worker thread and Qt main thread). Arrays are copies.
- **Stateless processors.** `SignalProcessor` instances may have internal caches (FFT windows etc.), but `process_fine(...)` must depend only on its arguments — that is what makes the GUI Proc combo's runtime swap safe.
- **Telemetry / debug logs** go under `spectrum_logs/` (already gitignored). Never write logs to repo-tracked paths.
- **No formal test suite yet.** Verification is via the headless script in §4, the offscreen Qt smoke tests in `drone_detector_enhanced` (run with `QT_QPA_PLATFORM=offscreen`), or hand-run of the GUI. A proper `tests/` directory is a tracked follow-up.

---

## 7. Detection algorithms (plug-ins)

Every fine-scan detection runs through the **active `SignalProcessor`**, selected via the GUI Proc combo and held in `SpectrumEngine._processor`. The engine itself never picks an algorithm; it dispatches whatever is registered.

### Available processors

| `name` | `label` | What it does |
|--------|---------|--------------|
| `classic` | Classic (CA-CFAR) | Wraps the existing `compute_fine_measurement` — mean-based CA-CFAR with guard cells, threshold = mean + `threshold_offset_db`, contiguous-bin region clustering, gap merging. Same behaviour as the engine had before the plug-in refactor. **This is the default.** |
| `oscfar` | OS-CFAR + Classifier | 5-stage pipeline: integrated PSD → 75th-percentile OS-CFAR with global noise-floor fallback for wideband emitters → per-ROI bandwidth, Shannon entropy, sync-pulse detection at 15.625 kHz → heuristic classifier → `ClassificationResult`. Spec-driven implementation that catches the wideband jammers Classic is blind to. |

### Adding a new processor

1. Write a class in `signal_pipeline/` inheriting `SignalProcessor`, set `name` and `label`, implement `process_fine(capture, psd_db, freq_axis_hz, cfg) -> FineMeasurement`.
2. Append it to `_PROCESSORS` in [registry.py](signal_pipeline/registry.py).
3. Done — the GUI Proc combo picks it up automatically.

The engine, the scheduler, the track manager, and the GUI need no changes. Existing processors (especially Classic) are preserved as fallbacks; nothing is archived.

### Performance budget

The spec target for new processors is **< 10 ms per fine-scan buffer**. Measured `OSCFARProcessor.process_fine` p95 is ~2.5 ms on a 2026-era MacBook against the default 4096-bin fine PSD; Classic is sub-ms.

---

## 8. Signal Processing Pipeline — implemented

The five-stage pipeline (Hamming-style integrated PSD → OS-CFAR → features → classifier → structured output) called out in the original user spec is now shipped as `OSCFARProcessor`. The standing reference for tuning and known limitations is [signal_processing_integration_plan.md](signal_processing_integration_plan.md). For the plug-in architecture this processor lives inside, see [signal_processing_selector_plan.md](signal_processing_selector_plan.md).

Key empirical results from the implementation:

- All 7 simulator presets (Empty band + FPV @ 1.3 / 2.4 / 3.7 / 5.8 / 6.2 GHz + Jammer @ 2.45 GHz) classify correctly.
- OS-CFAR's small reference window (16 cells ≈ 78 kHz) sits inside an 8 MHz FPV emitter, so its raw output misses wideband signals. The implementation combines OS-CFAR with a **global noise-floor fallback** (30th-percentile of the PSD + α) and takes the minimum — narrowband sensitivity intact, wideband coverage restored.
- Empirically, Shannon entropy is **not** a useful discriminator at this resolution: FM video and band-limited noise both occupy many bins relatively uniformly. The v1 classifier keys on bandwidth + horizontal sync detection only; entropy is kept in the feature snapshot for telemetry / future tuning.
- Real FPV transmitters are pure FM (constant envelope) on paper, but residual AM at the sync tip produces the 15.625 kHz envelope-power line the classifier looks for. The simulator's `gen_analog_fpv` reproduces this on purpose.

---

## 9. GUI operator controls

Top bar (left → right) in [drone_detector_enhanced.py](drone_detector_enhanced.py):

| Control | What it does |
|---------|--------------|
| `Src:` Hardware / Simulator | Picks the IQ source. Hardware uses `PyAdiIQSource` over pyadi-iio; Simulator uses `SimulatedIQSource`. Both feed `SignalReader`. Changing source while a session is running snaps the stage back to Idle. |
| `Scene:` preset combo | Active only in Simulator mode. Presets: Empty band, FPV @ 1.3 / 2.4 / 3.7 / 5.8 / 6.2 GHz, Jammer @ 2.45 GHz. Live-swap supported via `SimulatedIQSource.set_scene`. |
| `Proc:` Classic / OS-CFAR | Picks the `SignalProcessor`. Runtime swap via `SpectrumEngine.set_processor`. |
| **Stage ladder** `[Idle] [Receive] [Process] [Classify]` | Four-segment exclusive control. The selected stage + every stage to its left is filled with its colour (gray / cyan / green / orange). One click advances or retreats. Default on launch: **Idle**. See [§7 Stage ladder](#7-detection-algorithms-plug-ins) and [gui_stage_ladder_plan.md](gui_stage_ladder_plan.md). |

The stage ladder is owned by `EngineStage` (see §2). The same segmented control replaces the previous Connect button (Idle ↔ active transitions own connection) and the Detect toggle (Process ↔ Classify owns detection). One control instead of two; no hidden state.

Main display:

- **RX1 (Omni) — Spectrum + Waterfall**: live receiver view inside whatever 20 MHz window the engine is currently tuned to.
- **Simulated Signal (1.0–6.5 GHz) — Spectrum + Waterfall** (formerly RX2 in the layout): **analytical** ground-truth PSD of the active simulator scene, refreshed at 5 Hz on its own timer, independent of the engine loop. Blank in Hardware mode.

The Sim Preview panel is purely visual — no IQ, no detection, no algorithm runs against it. It's the "you-are-here" reference for what scene you've selected.

---

## 10. Things to avoid

- Don't change the destroy-buffer-then-set-`rx_lo` order in `PyAdiIQSource.tune` — pyadi-iio breaks silently if you reorder them (errno-0 stalls, zero scans). The reverse order has caused incidents in the past.
- Don't change `rx_buffer_size` **at runtime**. It's set once at `PyAdiIQSource.__init__` based on cfg; per-capture resizing has broken libiio "interleaved sample layout" in the past. If the lifted default (`fft_size_base × max(fine_frames, coarse_frames)`) regresses on real hardware, pass `rx_buffer_size_override=2048` and document it.
- Don't add normalisation, framing, or anti-alias logic inside an `IQSource`. That work belongs in `SignalReader.capture` so both sources share one rule book. Sources are raw-IQ-only.
- Don't bypass `EngineStage` to "just do detection". Walk the ladder: cells must already exist (PROCESS) before tracks (CLASSIFY) make sense. Going down a stage triggers the right cleanup (drop tracks at Classify→Process; reset cells at Process→Receive) — preserve that contract.
- Don't reintroduce a `detect_enabled` bool. The 4-stage ladder replaces it. `set_detection_enabled` survives as a thin alias mapping `{True: CLASSIFY, False: PROCESS}` for back-compat smoke scripts; new code uses `set_stage` directly.
- Don't drop the DC-spike excision in `SpectrumEngine._process_capture` — PlutoSDR leaks LO into the centre FFT bin every capture, and removing the excision lights up every cell with a false centre signal. The simulator deliberately reproduces this spike.
- Don't widen the scheduler's priority cadence beyond `_PRIORITY_EVERY = 4` — small priority-band revisit intervals (0.5 s) used to monopolise the scheduler and starve coverage.
- Don't print to stdout from worker threads — use `error_occurred` signal or telemetry log.
- Don't make `SignalProcessor` instances stateful across calls. The GUI Proc swap and the engine loop both rely on `process_fine` being a function of its arguments. Caches (windows, lookup tables) are fine; observation state is not.
- Don't let `signal_pipeline` import anything from `spectrum_engine.engine` at module top — only `spectrum_engine.detectors` / `config` / `sdr_backend` (the data dataclasses). The engine imports the registry, so the reverse direction would cycle. `classic.py` already uses a lazy `compute_fine_measurement` import for exactly this reason.
- Don't write data to `spec_dir` / `wf_dir` from the main display loop — those widgets are now the Sim Preview panel, driven exclusively by `_refresh_sim_preview`.

---

## 11. How this document stays current

This file is the **first place** anyone (human or AI) should look to understand the project, and it ages badly if no one is responsible for keeping it fresh. The contract:

> **Whenever a change lands that affects any of the items below, the same commit (or the immediately following commit) must update this file.**

| If you change... | Update §... |
|------------------|-------------|
| The data flow between source / engine / processor / GUI | §2 diagram |
| File/module layout in `spectrum_engine/`, `signal_pipeline/`, `proximity_alert/`, top-level | §3 codebase map |
| Install steps, deps, supported platforms | §4 Running |
| Hardware tuning ranges, libiio behaviour | §5 Hardware notes |
| Coding conventions, threading rules, default units | §6 Conventions |
| Set of available processors, processor budgets, registry mechanics | §7 |
| Signal-processing spec or its implementation | §8 (and the corresponding plan doc) |
| GUI controls — combos, buttons, panels | §9 |
| Newly discovered footguns / hard-learned rules | §10 |

Rules for AI agents touching code in this repo:

1. **Read this file at the start of any session before touching code.**
2. **Update this file at the end of any session that changes architecture, files, controls, conventions, or installs.** Mention which sections changed in the commit message.
3. **Don't create a new planning doc when an existing one applies.** If a change relates to the simulator, update [sdr_simulator_plan.md](sdr_simulator_plan.md). Algorithm selector → [signal_processing_selector_plan.md](signal_processing_selector_plan.md). New processor → [signal_processing_integration_plan.md](signal_processing_integration_plan.md). Only create new plan docs for genuinely new scope.
4. **Prefer linking over duplicating.** This doc is a map. Detail belongs in the plans, READMEs, or the code comments — link to them from here, don't copy them in.
5. **If a section in this doc disagrees with the code, fix the code or fix the doc — never leave them inconsistent.** Note the resolution in your commit message.
6. **Treat the "Things to avoid" list (§10) as load-bearing.** Each entry is a real incident or principle. Don't remove an entry without explaining why in the commit.
