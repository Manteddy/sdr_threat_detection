# Experiment Recording and Replay

Full design rationale: [experiment_recording_plan.md](../experiment_recording_plan.md).

## What it is

Hardware sessions can be recorded to disk as self-contained experiment folders and later replayed through any `SignalProcessor` without needing the drone in the air again. Recording is hardware-only (REC button is disabled in Simulator and Replay modes).

## On-disk layout

```
experiments/
  YYYY-MM-DD_HHMMSS_<name>/
    experiment.json     # name, comment, start/stop ts, processor, engine config, hw config
    sweep_log.jsonl     # one MeasurementCommand per line: ts, center_hz, bw, dwell, is_fine
    snapshots.jsonl     # one EngineSnapshot summary per line: ts, scan_count, tracks, groups
    heatmap/
      000001.npy        # periodic float32 occupancy_probs array dumps
      ...
    iq/
      000001.npz        # int16 interleaved I/Q + metadata per capture
      ...
    INDEX.json          # written on clean stop; absence = interrupted recording
```

`INDEX.json` is the "experiment is complete" marker. The browser dock shows folders without it as "interrupted".

## IQ encoding

| Property | Value |
|---|---|
| dtype | `int16` interleaved I/Q (4 bytes/sample) |
| Resolution | Lossless at Pluto's native 12-bit ADC |
| Encoding | `recorder` reverses `SignalReader`'s normalisation: `round(normalised_iq × adc_full_scale)` → `int16` |
| Decoding | `ReplayIQSource.acquire()` reconstructs `complex64` in raw ADC units; `SignalReader` normalises again → identical result |

The round-trip is lossless to within ±0.5 LSB (negligible relative to 12-bit noise floor).

## Time-locked replay

Replay is **time-locked**: the engine's live scheduler is overridden by `ReplayIQSource.get_next_command()`, which vends recorded `MeasurementCommand` objects from `sweep_log.jsonl` in sequence. Cells are updated at the original recorded center frequencies, so detection state builds up identically to the original session.

`SpectrumEngine.step()` checks `self._replay_source` first; if set and not exhausted, it uses the recorded command instead of calling `Scheduler.choose_next_measurement()`. When all frames are consumed, the engine snaps to Idle cleanly.

## GUI controls

| Element | Behaviour |
|---|---|
| `Src: Replay` | Switches source; opens the experiment browser dock |
| **REC** button | Amber when Hardware + stage ≥ Receive. Click → `ExperimentDialog` (name, comment, max duration, heatmap rate). During recording: red, shows `● mm:ss / NMB`. Click again → stop + finalise |
| **Experiment browser dock** | Right-side QDockWidget. Columns: Name, Date, Duration, Frames, Status. Load / Open Folder / Delete. Refreshed on dock show and on recorder stop |

## Key code locations

| Role | File |
|---|---|
| Recorder | `experiments/recorder.py` — `ExperimentRecorder` |
| Replay source | `spectrum_engine/sources/replay.py` — `ReplayIQSource` |
| Engine hooks | `spectrum_engine/engine.py` — `attach_recorder`, `attach_replay_source` |
| Config | `engine_config.yaml experiments:` block, `spectrum_engine/config.py ExperimentsCfg` |
| GUI | `drone_detector_enhanced.py` — `ExperimentDialog`, `ExperimentBrowserDock`, `_connect_replay`, `_start_recording` |
