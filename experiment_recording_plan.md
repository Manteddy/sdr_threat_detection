# Experiment Recording & Replay — Plan

Status: **implemented** (2026-05-31).
Related: [sdr_threat_detection_ProjectDefinition.md](sdr_threat_detection_ProjectDefinition.md),
[signal_reader_refactor_plan.md](signal_reader_refactor_plan.md),
[signal_processing_selector_plan.md](signal_processing_selector_plan.md).

## 1. Goal

Let an operator capture a hardware session as a self-contained "experiment"
folder on disk, browse past experiments in-app, and replay one through the
same engine + processor pipeline as if it were live — so any current or
future `SignalProcessor` can be evaluated against bit-identical real-world
RF without needing the drone in the air again.

## 2. Decisions (locked with user)

| # | Decision |
|---|----------|
| 1 | Record **raw IQ + metadata**. Lossless capture is the whole point; lossy PSD-only recordings would defeat the "test new processor against old data" use case. |
| 2 | IQ on disk is **`int16` interleaved I/Q** — matches Pluto's native 12-bit ADC, half the size of `complex64`, no fidelity loss. `SignalReader` redoes its dBFS normalization at replay time. |
| 3 | Replay is a **new `ReplayIQSource`** that plugs into `SignalReader` exactly like `PyAdiIQSource` / `SimulatedIQSource`. Adds a third entry to the `Src:` combo. |
| 4 | Replay mode is **time-locked**: captures are fed back in original chronological order; the live scheduler is overridden during replay. Lookup-by-frequency replay is a future extension. |
| 5 | Recording is **hardware-only**. The `REC` button is disabled in Simulator and Replay modes (sim is already deterministic from scene presets; replay would just copy bits). |
| 6 | Experiment browser is **in-app**, as a right-side dockable panel listing past experiments. The record-modal handles new captures. |
| 7 | Captured artifacts per experiment: description + name + timestamps, hardware/SDR config, sweeping logs, probability heatmap snapshots, plus the IQ. |

## 3. On-disk layout

```
experiments/                                    # configurable root (engine_config.yaml)
  2026-05-31_142233_fpv-5g8-rooftop/
    experiment.json     # name, comments, start_ts, stop_ts, processor name,
                        #   engine_config snapshot, hardware config
                        #   (device_id, sample_rate, rf_bw, gain, antenna_mode, iq_dtype)
    sweep_log.jsonl     # one MeasurementCommand per line: ts, center_hz, span_hz, dwell_s, kind
    snapshots.jsonl     # one EngineSnapshot summary per line: ts, detections, tracks, cell deltas
    heatmap/
      000001.npy        # periodic occupancy/probability grid dumps (rate from config)
      000002.npy
      ...
    iq/
      000001.npz        # {samples: int16[2N] interleaved I,Q, ts, center_hz,
      000002.npz        #  sample_rate, gain, rf_bw}
      ...
    INDEX.json          # written on graceful stop: counts + sha256 of each file
```

- Per-capture `.npz` keeps recording crash-safe (no monolithic file to truncate)
  and gives `ReplayIQSource` trivial seeking.
- `INDEX.json` is the "experiment is complete" marker. Folders without it are
  shown as "interrupted" in the browser.

## 4. Components

### 4.1 `experiments/recorder.py` — `ExperimentRecorder`

- Owns the folder, opened on `start(name, comment, opts)`.
- `on_capture(capture: IQCapture, cmd: MeasurementCommand)` — writes one
  `iq/NNNNNN.npz` and appends one `sweep_log.jsonl` line.
- `on_snapshot(snap: EngineSnapshot)` — appends `snapshots.jsonl` and, on the
  configured cadence, dumps a `heatmap/NNNNNN.npy`.
- `stop()` — finalizes `experiment.json` (adds stop_ts, totals) and writes
  `INDEX.json`.
- Streams to disk via a background thread + bounded queue so the engine thread
  never blocks on I/O. Drops with a warning if the queue saturates (not silently).

### 4.2 `spectrum_engine/sources/replay.py` — `ReplayIQSource(IQSource)`

- `__init__(path)` reads `experiment.json` + scans `iq/`.
- `tune(freq_hz)` is a no-op for the freq value; advances the internal index.
- `acquire(n)` returns the next recorded buffer's `int16` samples cast back to
  `complex64`, length-matched to `n` by truncation or zero-pad (rare — sizes
  should match because the engine config snapshot is replayed too).
- Exposes `HardwareLimits` reconstructed from the recorded metadata.
- Sentinel "end of recording" stops the engine cleanly (snaps to Idle).

### 4.3 Engine hook

`SpectrumEngine` gains optional `self._recorder: ExperimentRecorder | None`
with `attach_recorder` / `detach_recorder`. `step()` calls
`_recorder.on_capture(...)` right after `reader.capture()` returns and
`_recorder.on_snapshot(...)` after the snapshot is built. No processor
changes; recording is orthogonal to detection.

### 4.4 GUI

- **Top bar**: small `REC` button right of the stage ladder.
  - Grey/disabled when `Src` is Simulator or Replay.
  - Grey/enabled when `Src: Hardware` and stage ≥ Receive.
  - Click → modal (`ExperimentDialog`) with: Name (required), Comment,
    Max duration (default 5 min), Heatmap snapshot rate, IQ dtype shown
    read-only as `int16 I/Q`.
  - During recording: button turns red, shows `● mm:ss / NN.N MB`. Click →
    confirm → finalize.
- **`Src:` combo**: adds `Replay` entry. Selecting it opens the experiment
  browser dock (or focuses it if already open).
- **Right-side dock — Experiment Browser** (`QDockWidget`):
  - Table of folders under `experiments/`: name, date, duration, processor,
    size, status (complete / interrupted).
  - Buttons: Load (sets `Src: Replay` against this folder), Open in Finder,
    Delete (with confirm).
  - Refreshed on focus + on recorder `stop()`.

### 4.5 `engine_config.yaml` additions

```yaml
experiments:
  root: experiments/          # relative to repo root
  iq_dtype: int16             # int16 only for now; reserved for future
  heatmap_snapshot_hz: 1.0
  max_queue_frames: 256       # back-pressure for recorder I/O thread
```

## 5. Touch list

- New: `experiments/__init__.py`, `experiments/recorder.py`,
  `experiments/dialog.py`, `experiments/browser.py`,
  `spectrum_engine/sources/replay.py`,
  `experiment_recording_plan.md` (this file).
- Edited: `spectrum_engine/engine.py` (recorder hooks),
  `spectrum_engine/config.py` (new `experiments` block),
  `engine_config.yaml`, `drone_detector_enhanced.py` (REC button, Src combo
  `Replay` entry, dock), `sdr_threat_detection_ProjectDefinition.md`
  (§2 diagram, §3 codebase map, §9 GUI controls, §10 don't-record-in-sim,
  new §12 for experiments).

## 6. Risks / things to avoid

- **Don't write IQ from the engine thread.** Recorder I/O must be on its own
  thread or the dwell budget collapses. Bounded queue + drop-with-warning,
  never silent drop.
- **Don't change `IQSource` semantics for replay.** Replay sits behind the
  same `tune/acquire` contract; the time-locked ordering lives inside
  `ReplayIQSource`, not in the engine.
- **Don't allow recording in Simulator / Replay modes.** UI must enforce it
  and the recorder must assert it.
- **Don't store derived PSDs.** They're recomputable from IQ; storing them
  would invite drift between recorded PSD and replay-recomputed PSD.
- **Stale `INDEX.json` after a crash** — recorder writes it last, browser
  treats absence as "interrupted" and offers cleanup.

## 7. Out of scope (v1)

- Lookup-by-frequency replay (try a different sweep strategy against the
  same recording). Add when we actually want algorithm A/B.
- Exporting recordings to SigMF or other interchange formats.
- Cloud / network storage of experiments.
- Editing or annotating snapshots after capture.
