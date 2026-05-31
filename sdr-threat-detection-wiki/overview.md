# Project Overview

## What This Is

`sdr_threat_detection` (also called `pluto_drone_detector`) is a real-time SDR-based anti-drone detection system. It uses a PlutoSDR / AD9361 to monitor the wideband analog video transmissions used by FPV strike and reconnaissance drones across **1.0 – 6.5 GHz**, with priority on 1.2 / 2.4 / 5.8 GHz bands.

Output is a live spectrum + waterfall UI, a ground-truth simulator preview panel, and a proximity alert window that can also be exposed over WebSocket.

It implements **steps 3–4** of a broader pipeline: SDR acquisition + signal processing. Upstream RF capture (steps 1–2) and downstream Starlink relay / mission-control overlay (steps 5–10) are out of scope.

## Main Features

- **Three data sources switchable at runtime**: PlutoSDR hardware (Linux only), fully synthetic simulator (Mac/CI), or recorded experiment replay.
- **Two detection algorithms switchable at runtime**: Classic CA-CFAR and OS-CFAR + Classifier.
- **4-stage operating ladder** (Idle → Receive → Process → Classify): one segmented control drives connection, spectrum display, and detection on/off — no hidden state.
- **Live spectrum + waterfall** (20 MHz receive window, auto-tuning scheduler).
- **Sim Preview panel**: analytical ground-truth PSD of the active simulator scene (1–6.5 GHz), independent of the engine loop.
- **Experiment recording + replay**: hardware sessions recorded as self-contained folders (raw IQ + sweep log + heatmap); any recording can be replayed through any processor. See [experiments.md](experiments.md).
- **ProximityAlertPanel + WebSocket bridge**: classifies threat proximity (NONE / DETECTED / APPROACHING / CRITICAL) and broadcasts over WebSocket.
- **Adaptive hierarchical scheduler**: interleaves coverage scans with priority revisits on 1.2 / 2.4 / 5.8 GHz bands, track-follow dwells, and group fine scans.
- **Pluggable processor registry**: adding a new detection algorithm requires only writing one class and appending it to the registry.

## Simulator Scenes (Presets)

| Scene | Description |
|-------|-------------|
| Empty band | Noise floor only |
| FPV @ 1.3 GHz | Analog FPV video emitter |
| FPV @ 2.4 GHz | Analog FPV video emitter |
| FPV @ 3.7 GHz | Analog FPV video emitter |
| FPV @ 5.8 GHz | Analog FPV video emitter |
| FPV @ 6.2 GHz | Analog FPV video emitter |
| Jammer @ 2.45 GHz | Barrage noise jammer |

All 7 presets classify correctly with `OSCFARProcessor`.

## Running the Project

**Every run (Mac or Linux):**
```bash
./run.sh
```

`run.sh` calls `.venv/bin/python` directly — no `source .venv/bin/activate` needed. It also `cd`s into the repo so `engine_config.yaml` and `spectrum_logs/` resolve correctly.

**First-time setup — Mac (simulator only, no hardware required):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyqtgraph PyQt5 pyyaml
```
In the GUI: **Src → Simulator**, pick a Scene, advance the stage ladder: Receive → Process → Classify.

**First-time setup — Linux (real Pluto):**
```bash
sudo apt install -y libiio0 libiio-utils python3-libiio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install pyyaml
```
In the GUI: **Src → Hardware**, advance to Receive (or Classify).

**Headless / scripted (no Qt):**
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

## Planned Work

1. **Sweep algorithm edge-case improvements** — Harden the adaptive scheduler and `SignalReader` for edge cases (e.g. band boundaries, rapid scene changes, stale cells after source swap).

2. **Classifier upgrade: decision tree → ML** — Evolve `OSCFARProcessor`'s heuristic classifier into a proper decision tree as a baseline, then improve toward a trained ML model. Features (bandwidth, entropy, sync-pulse presence) are already extracted per ROI and available as the training surface.

3. **Classifier UI improvements** — Richer display of per-detection classification results (label, confidence, features) in the main window; currently classifications surface only in telemetry logs and the proximity alert panel.

4. **BetaFlight simulator integration** — Connect the detection output to a BetaFlight simulator instance, enabling end-to-end testing of detection → control-link disruption scenarios without physical hardware.

5. **Formal `tests/` directory** — currently verified via headless script and Qt offscreen smoke tests.

### Also tracked

- Bayesian and ML inference hooks are stubbed in `engine_config.yaml` (`bayesian.enabled`, `inference.enabled`) — not yet active.
- Dual-RX antenna path (omni + directional) — hardware-ready but pending Pluto 2r2t firmware; currently mirrors omni into dir.

## Out of Scope

- Upstream RF capture / hardware prior to PlutoSDR (steps 1–2 of product pipeline).
- Downstream Starlink relay, mission-control overlay, video goggle feed (steps 5–10).
- macOS + real hardware (Docker Desktop cannot map a Pluto over USB; hardware deployment is Linux-only).
