# Technical Decisions

A record of significant design decisions, the rationale behind them, and any known trade-offs. Append new entries at the bottom; don't edit past entries.

---

## [2026-05-31] Single capture function (`SignalReader`)

**Decision:** Both hardware (`PyAdiIQSource`) and simulator (`SimulatedIQSource`) sources feed the same `SignalReader.capture()` function. Sources do raw IQ acquisition only; all normalisation, anti-alias filtering, framing, dwell budgeting, and the single-antenna mirror live in the reader.

**Rationale:** "Signal is signal" — the processing pipeline should not know or care which source produced the IQ. Keeping source-specific logic out of the reader prevents divergent code paths that would make simulator results unreliable as a proxy for hardware behaviour. Documented in `signal_reader_refactor_plan.md`.

**Trade-off:** The `IQSource` ABC is strictly raw-IQ-only; any future source must resist the temptation to do even light processing internally.

---

## [2026-05-31] 4-stage EngineStage ladder replaces Connect + Detect toggles

**Decision:** One 4-segment exclusive control (Idle / Receive / Process / Classify) replaces the previous `Connect` button and `detect_enabled` toggle. `set_detection_enabled(bool)` survives as a thin alias for back-compat smoke scripts only; new code uses `set_stage` directly.

**Rationale:** Two orthogonal controls (connected + detect on/off) created 4 possible states, two of which (disconnected + detect on; detecting but no cells yet) were incoherent. The ladder enforces the correct invariant: each stage is a superset of those below it, and downward transitions trigger the right cleanup. Documented in `gui_stage_ladder_plan.md`.

**Trade-off:** Operators used to a separate Connect button need to learn the new model; mitigated by colour-coding each segment.

---

## [2026-05-31] Plug-in processor registry (`signal_pipeline/`)

**Decision:** Detection algorithms are implemented as `SignalProcessor` subclasses in `signal_pipeline/`, registered in `registry.py`, and swappable at runtime via the GUI Proc combo. The engine dispatches to whatever is registered; it never picks an algorithm itself.

**Rationale:** Classic CA-CFAR and OS-CFAR + Classifier have very different performance profiles and use cases. Locking to one algorithm at the engine level would force a restart to switch. The registry pattern lets both coexist and lets new algorithms be added without touching the engine, scheduler, track manager, or GUI. Documented in `signal_processing_selector_plan.md`.

**Trade-off:** Processors must be stateless across calls (caches OK; observation state not), which is a non-obvious constraint for future implementors.

---

## [2026-05-31] OS-CFAR global noise-floor fallback for wideband emitters

**Decision:** `OSCFARProcessor` combines OS-CFAR (75th-percentile, 16-cell reference window ≈ 78 kHz) with a global noise-floor fallback (30th-percentile of the PSD + α) and takes the minimum threshold.

**Rationale:** A 16-cell OS-CFAR reference window sits entirely inside an 8 MHz FPV emitter, making the local "noise" estimate far too high and causing the OS-CFAR alone to miss wideband signals. The global fallback catches what the local estimator cannot, while keeping the local estimator's narrowband sensitivity intact. Documented in `signal_processing_integration_plan.md`.

**Trade-off:** The global fallback raises false-alarm rate slightly in very crowded spectrum; α is a tunable offset in the config.

---

## [2026-05-31] dBFS as default internal power unit

**Decision:** All internal power values are in dBFS (dB relative to ADC full scale = 2048). Convert to GHz only at UI / alert boundary; convert to other power units only when absolutely required.

**Rationale:** Keeps the entire processing chain dimensionally consistent and avoids implicit conversions that have historically caused calibration bugs. The ADC full scale (2048) is fixed in `engine_config.yaml:hardware.adc_full_scale`.

---

## [2026-05-31] DC-spike excision preserved in `SpectrumEngine._process_capture`

**Decision:** The centre-bin DC spike excision is kept permanently and non-optional.

**Rationale:** PlutoSDR leaks its LO into the centre FFT bin every capture. Removing the excision causes every cell to register a false centre-bin signal. The simulator deliberately reproduces this spike to keep the code path exercised. Removing the excision is listed as a "Things to avoid" footgun in the ProjectDefinition.

---

## [2026-05-31] `rx_buffer_size` set once at `PyAdiIQSource.__init__`

**Decision:** Buffer size is `fft_size_base × max(fine_frames, coarse_frames)`, sized once at init and never changed at runtime.

**Rationale:** Per-capture resizing has broken libiio's "interleaved sample layout" in the past. The fixed size also lifts the historic 2048-sample cap that truncated fine FFTs. If this regresses on real hardware, the escape hatch is `rx_buffer_size_override=2048` — document it if used.

---

## [2026-05-31] Scheduler priority cadence capped at `_PRIORITY_EVERY = 4`

**Decision:** Priority-band revisit interleaving is limited to one priority scan every 4 coarse sweeps.

**Rationale:** Small priority revisit intervals (0.5 s) used to monopolise the scheduler and starve coverage sweeps, leaving large parts of the 1–6.5 GHz range unscanned. Documented as a "Things to avoid" entry.

---

## [2026-05-31] Analytical Sim Preview panel (no IQ, no detection)

**Decision:** The Sim Preview panel (`spectrum_engine/sim/preview.py`) renders an analytical full-range PSD from the active scene geometry, driven by a 5 Hz Qt timer independent of the engine loop. No IQ is generated and no algorithm runs against it.

**Rationale:** The preview is a "you-are-here" reference for what scene is selected — it should be fast, always accurate, and not interact with the detection pipeline. Running the engine just for a visual reference would burn CPU and couple the display to engine state.
