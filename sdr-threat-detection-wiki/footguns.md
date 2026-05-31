# Footguns — Things to Avoid

**This page is load-bearing.** Each entry records a real incident, a subtle invariant, or a hard-learned principle. Do not remove an entry without explaining why in the commit message.

---

## Hardware / pyadi-iio

**Don't reorder `rx_destroy_buffer` → set `rx_lo` in `PyAdiIQSource.tune`.**
pyadi-iio breaks silently if you set `rx_lo` before destroying the buffer — errno-0 stalls, zero scans returned. This has caused incidents. The order is: destroy buffer first, then set LO.

**Don't change `rx_buffer_size` at runtime.**
It is sized once at `PyAdiIQSource.__init__` based on cfg. Per-capture resizing has broken libiio's "interleaved sample layout" in the past. If the lifted default (`fft_size_base × max(fine_frames, coarse_frames)`) regresses on real hardware, pass `rx_buffer_size_override=2048` and document it.

---

## IQ sources and `SignalReader`

**Don't add normalisation, framing, or anti-alias logic inside an `IQSource`.**
That work belongs in `SignalReader.capture()` so hardware, simulator, and replay sources all share one rule book. Sources are raw-IQ-only.

**Don't break the `IQSource` contract in `ReplayIQSource`.**
`acquire()` must return raw ADC-scale complex64 (NOT normalised), because `SignalReader` normalises it afterwards. The recorder scales back up (×`adc_full_scale`) before int16 conversion so the round-trip is transparent.

**Don't change `anti_alias=False` for `ReplayIQSource`.**
Recorded data is already bandlimited from the original capture. Applying the LPF again produces a double-filter with measurable frequency-domain distortion.

---

## Engine stage ladder

**Don't bypass `EngineStage` to "just do detection".**
Walk the ladder: cells must already exist (PROCESS) before tracks (CLASSIFY) make sense. Going down a stage triggers the right cleanup (drop tracks at Classify→Process; reset cells at Process→Receive). Skipping this leaves stale state.

**Don't reintroduce a `detect_enabled` bool.**
The 4-stage ladder replaced it. `set_detection_enabled` survives as a thin alias mapping `{True: CLASSIFY, False: PROCESS}` for back-compat smoke scripts only. New code uses `set_stage` directly.

---

## Signal processing

**Don't drop the DC-spike excision in `SpectrumEngine._process_capture`.**
PlutoSDR leaks LO into the centre FFT bin every capture. Removing the excision lights up every cell with a false centre signal. The simulator deliberately reproduces this spike to keep the code path exercised.

**Don't widen the scheduler's priority cadence beyond `_PRIORITY_EVERY = 4`.**
Small priority-band revisit intervals (0.5 s) used to monopolise the scheduler and starve coverage sweeps, leaving large parts of 1–6.5 GHz unscanned.

**Don't make `SignalProcessor` instances stateful across calls.**
The GUI Proc swap and the engine loop both rely on `process_fine` being a function of its arguments. Caches (windows, lookup tables) are fine; observation state is not.

---

## Import graph

**Don't let `signal_pipeline` import from `spectrum_engine.engine` at module top.**
The engine imports the registry (`signal_pipeline`), so the reverse direction cycles. Use lazy imports where needed. `classic.py` already uses a lazy `compute_fine_measurement` import for exactly this reason.

---

## GUI display loop

**Don't write data to `spec_dir` / `wf_dir` from the main display loop.**
Those widgets are now the Sim Preview panel, driven exclusively by `_refresh_sim_preview`. Writing from the display loop would corrupt the sim preview output.

**Don't print to stdout from worker threads.**
Use the `error_occurred` signal or the telemetry log in `spectrum_logs/`.

---

## Experiment recording

**Don't record in Replay mode.**
Replay would just copy already-recorded bits. Simulator recording is allowed (useful for testing the recorder without real hardware); only Replay is blocked.

**Don't write IQ from the engine thread.**
`ExperimentRecorder.on_capture()` queues to a background I/O thread. The engine thread must never block on disk I/O. The queue is bounded; overflow logs a warning and drops — never silently.

**Don't store derived PSDs in experiment folders.**
PSDs are recomputable from IQ. Storing them creates a risk of drift between the saved PSD and the replay-recomputed PSD. IQ is the ground truth.
