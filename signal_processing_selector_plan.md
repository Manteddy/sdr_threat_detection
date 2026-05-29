# Implementation Plan — Pluggable Signal Processing Algorithms

> Companion to: [sdr_threat_detection_ProjectDefinition.md §7-§8](sdr_threat_detection_ProjectDefinition.md).
> Supersedes the "hook directly into `_process_capture`" approach in the
> first revision of [signal_processing_integration_plan.md](signal_processing_integration_plan.md);
> that doc has been rewritten to match.

## 1. Context

The current engine has one detection algorithm hard-wired into
[`SpectrumEngine._process_capture`](spectrum_engine/engine.py): mean-based
**CA-CFAR** via [`compute_fine_measurement`](spectrum_engine/detectors.py:261).
It's proven, but it has known blind spots — most obviously, broadband
jammers raise the percentile noise-floor estimate uniformly across a cell,
so they read as "no signal above noise" instead of as a wideband emitter
(empirically reproduced against the `Jammer @ 2.45 GHz` simulator preset).

The user-supplied Signal Processing Pipeline spec (Project Definition §8)
introduces a **second** algorithm built around OS-CFAR + a heuristic
classifier specifically designed for that case. We want both algorithms
available in production so:

- the existing CA-CFAR cannot regress when a new processor is added;
- new processors can be evaluated against live signals without rebuilding
  or branching the codebase;
- the operator can switch algorithms at runtime — exactly like the
  Hardware↔Simulator toggle that already exists.

This plan defines the plug-in architecture and the GUI affordance.

## 2. Architecture

```
                         ┌───────────────────────────┐
                         │  signal_pipeline/         │
                         │  ──────────────────────   │
                         │  base.SignalProcessor     │  <── Protocol
                         │  registry.{name, label}   │
                         │  classic.ClassicProcessor │  <── wraps existing CA-CFAR
                         │  oscfar.OSCFARProcessor   │  <── new (next plan)
                         └───────────────┬───────────┘
                                         │ get() / list()
                                         ▼
SpectrumEngine.set_processor(processor)
   └─► _process_capture(cmd, capture, ...)
         ├─ channel_psd(...)
         ├─ self._processor.process_fine(capture, psd, f_ax, cfg) ──► FineMeasurement
         │     (classifications attached when the processor supports it)
         └─ track_mgr.update_from_regions(fine_meas.detected_regions, ...)
                                         ▲
                                         │  set_processor() at runtime
GUI: Proc: [Classic ▾] / [OS-CFAR + Classifier ▾]
```

Three invariants:

1. **Classic is the default.** A fresh `SpectrumEngine()` ships with
   `ClassicProcessor` already set, so behavior is byte-identical to today
   without any caller changes.
2. **Processors are stateless across calls.** `process_fine()` is a pure
   function of its inputs (plus optional internal warm caches like the FFT
   window cache). That keeps runtime swap trivial — just point
   `self._processor` at the new instance.
3. **No engine surgery per new processor.** Adding a third or fourth
   algorithm later is "write a class, register it, ship it." The engine
   stays untouched.

## 3. The `SignalProcessor` Protocol

`signal_pipeline/base.py`:

```python
class SignalProcessor(Protocol):
    name: ClassVar[str]   # short id, e.g. "classic"
    label: ClassVar[str]  # human label for GUI, e.g. "Classic (CA-CFAR)"

    def process_fine(
        self,
        capture: IQCapture,
        psd_db: np.ndarray,
        freq_axis_hz: np.ndarray,
        cfg: EngineConfig,
    ) -> FineMeasurement: ...
```

`FineMeasurement` (already in `spectrum_engine/detectors.py`) gains one
optional, default-empty field so processors that classify can return
something the engine and GUI know how to surface:

```python
classifications: List[ClassificationResult] = field(default_factory=list)
```

Adding a field with a default is backward compatible with every existing
construction site.

A new top-level dataclass `ClassificationResult` lives in
`signal_pipeline/base.py` (frequency_hz, bandwidth_hz, signal_strength_db,
classification: str, confidence_score: float, features: dict). Defined
once, used by every classifying processor.

## 4. The registry

`signal_pipeline/registry.py`:

```python
_PROCESSORS: list[type[SignalProcessor]] = [
    ClassicProcessor,
    # OSCFARProcessor — added by the next plan
]

def list_processors() -> list[tuple[str, str]]:
    """[(name, label), ...] in display order. Drives the GUI combo."""

def get_processor(name: str) -> SignalProcessor:
    """Instantiate a processor by name."""
```

The registry is hand-maintained (not import-magic). Three processors fit
in a list; if it ever grows past ~8 the GUI gets a menu instead.

## 5. Engine integration

In [`SpectrumEngine.__init__`](spectrum_engine/engine.py:113), instantiate
the default processor:

```python
from signal_pipeline import ClassicProcessor
self._processor: SignalProcessor = ClassicProcessor()
```

Add a setter:

```python
def set_processor(self, processor: SignalProcessor) -> None:
    self._processor = processor
```

Replace the existing `compute_fine_measurement(...)` call inside
`_process_capture` with:

```python
fine_meas = self._processor.process_fine(capture, psd_primary, f_ax, cfg)
```

`ClassicProcessor.process_fine` delegates to the unchanged
`compute_fine_measurement` so behavior under the default path is bit-for-bit
identical.

## 6. GUI changes

[`drone_detector_enhanced.py`](drone_detector_enhanced.py):

1. Import the registry alongside the simulator import block, gated by
   `HAS_PIPELINE`.
2. Add a `Proc:` combo to the top bar, immediately after the
   Source/Scene combos. Populate it from `list_processors()`.
3. Default selection: `Classic (CA-CFAR)` (current behavior).
4. `currentIndexChanged` → call `self._engine.set_processor(get_processor(name))`
   if the engine is attached; otherwise just stash the name and apply at
   connect time.
5. Both `_connect_hardware()` and `_connect_simulator()` apply the
   currently-selected processor after attaching the engine.

Runtime swap is supported because processors are stateless across calls.
The GUI never reaches into `_process_capture` directly.

## 7. Why this is the right shape

- Mirrors the **Src** / **Scene** pattern that already exists, so the UX
  and the architecture rhyme.
- Doesn't require changing any of the production telemetry / track /
  scheduler code — they all consume `FineMeasurement`, which the
  abstraction preserves.
- Falls back to Classic on import or instantiation error of any
  alternative processor — no startup risk from work in progress.
- Aligns with the simulator design: a contract + concrete implementations
  + a registry. We're using the same pattern that turned the SDR boundary
  from "one hard-coded class" into a place we can extend.

## 8. Files added / modified

| Change | File |
|--------|------|
| New | `signal_pipeline/__init__.py` |
| New | `signal_pipeline/base.py` (Protocol, `ClassificationResult`) |
| New | `signal_pipeline/classic.py` (`ClassicProcessor`) |
| New | `signal_pipeline/registry.py` |
| Modified | `spectrum_engine/detectors.py` (add optional `classifications` field) |
| Modified | `spectrum_engine/engine.py` (default `_processor`, `set_processor`, call site) |
| Modified | `drone_detector_enhanced.py` (Proc combo + slot) |

`OSCFARProcessor` and its own files land separately in
[signal_processing_integration_plan.md](signal_processing_integration_plan.md).

## 9. Verification

1. **Default-path regression**: with no GUI changes selected, run the
   simulator (`Src: Simulator`, `Scene: FPV @ 5.8 GHz`) for ≥600 engine
   steps; assert the same cells become ACTIVE as before (5.79 / 5.81 GHz
   at occupancy ≥ 0.99) — no observable difference.
2. **Runtime swap**: with the engine running, change the Proc combo;
   confirm no exception, no thread crash, and that the cell state on a
   stable FPV scene does not degrade.
3. **Headless test**: instantiate `SpectrumEngine`, call
   `set_processor(ClassicProcessor())`, run a few steps; instantiate
   again with the registry default; assert outputs match.

## 10. Out of scope (handled by later plans)

- `OSCFARProcessor` itself — handled by the updated
  [signal_processing_integration_plan.md](signal_processing_integration_plan.md).
- Routing classifications into the `AlertEngine` and the proximity panel
  — a follow-up after the new processor is live.
- Configuration block in `engine_config.yaml` to override processor
  defaults per processor — added when the second processor lands.
- Telemetry logging of classifications — added with the new processor's
  ship.
