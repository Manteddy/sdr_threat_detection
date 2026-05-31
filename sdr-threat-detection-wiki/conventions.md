# Coding Conventions

Project-wide rules that apply to all new code. Violations introduce subtle bugs or break the simulator ↔ hardware equivalence guarantee.

## Language and style

- **Python 3.8+.** All modules use `from __future__ import annotations` and `dataclasses`.
- **NumPy first.** All hot paths are vectorised. SciPy is used sparingly — windowing is rolled by hand in `spectrum_engine/psd.py` to control window-sum normalisation explicitly.

## Units

- **dBFS** is the default internal power unit — ratios to ADC full scale (2048). Keep new code in dBFS unless there is a reason to leave.
- **Frequencies in Hz** everywhere internal. Convert to GHz only at the UI / alert boundary (display labels, WebSocket output). Never store or compare frequencies in GHz internally.

## Thread safety

- **Immutable snapshots cross thread boundaries.** `EngineSnapshot` is the only object passed from the `SweepWorker` thread to the Qt main thread. All arrays inside it are copies — the worker must not hold a reference after emitting.
- **Never print to stdout from worker threads.** Use the `error_occurred` signal or write to the telemetry log.

## Processors

- **`SignalProcessor.process_fine()` must be stateless across calls.** The function may use internal caches (FFT windows, lookup tables) but must produce output that depends only on its arguments. This invariant is what makes the GUI Proc combo's runtime swap safe.

## Logging and telemetry

- Telemetry and debug logs go under `spectrum_logs/` (already gitignored). Never write logs to repo-tracked paths.

## Testing

- No formal test suite yet. Verification is via the headless script (see [overview.md](overview.md)) or `QT_QPA_PLATFORM=offscreen` Qt smoke tests in `drone_detector_enhanced`. A proper `tests/` directory is a tracked follow-up (see [overview.md](overview.md) Planned Work).

## Import ordering

- `signal_pipeline` must **not** import from `spectrum_engine.engine` at module top — the engine imports the registry, so the reverse direction creates a cycle. Use lazy imports where needed (`classic.py` already does this for `compute_fine_measurement`).
