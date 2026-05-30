# Implementation Plan — 4-stage GUI ladder (Idle / Receive / Process / Classify)

> Replaces today's `Connect` + `Detect` button pair with a single
> segmented control that walks through four strictly-ascending stages.
> Each stage is a strict superset of the previous, so the operator can
> tell at a glance how deep into the pipeline the engine currently is.

## 1. Context

Today the GUI has two independent toggles:

* `Simulate` / `Stop simulation` (or `Receiver On` / `Receiver Off`) — controls connection + worker thread.
* `Detect: OFF` / `Detect: ON` — controls cell-state updates + processor work.

Two problems:

1. The boundary between "no detection" and "with detection" is fuzzy — `Detect: OFF` already computes PSD, updates the display buffer, and updates per-cell occupancy. The label suggests no detection is running, but per-cell state updates *are* detection in any meaningful sense.
2. There is no way to **just sweep + receive raw IQ** as a diagnostic — useful for confirming data is flowing before any interpretation.

The 4-stage model splits this into honest layers and replaces both buttons with one segmented control.

## 2. Stage model

| Stage | What `engine.step()` runs | What's visible | Engine flags |
|------|----------------------------|----------------|---------------|
| **Idle** | Worker thread stopped. `step()` never called. | Nothing moves. Last frame frozen. | (worker not running) |
| **Receive** | Scheduler picks → `source.tune()` → `source.acquire()` → raw IQ dropped. No PSD, no display, no cells, no processor. | Activity / coverage map ticks. Confirms IQ is flowing. | `stage == RECEIVE` |
| **Process** | Receive + `channel_psd()` + display buffer + per-cell baseline / occupancy / state machine + activity grouping. | Spectrum + waterfall live. Cells colour-coded on activity map. No tracks, no classifications. | `stage == PROCESS` |
| **Classify** | Process + `processor.process_fine()` on fine scans + `TrackManager` updates. | Today's `Detect: ON` behaviour. Tracks appear, classifications shown. | `stage == CLASSIFY` |

Strict superset: each stage runs everything from the stages below it.

## 3. Engine changes

`spectrum_engine/engine.py`:

```python
class EngineStage(IntEnum):
    IDLE     = 0
    RECEIVE  = 1
    PROCESS  = 2
    CLASSIFY = 3

class SpectrumEngine:
    stage: EngineStage = EngineStage.CLASSIFY    # back-compat default

    def set_stage(self, stage: EngineStage) -> None:
        self.stage = stage

    def _process_capture(self, cmd, capture, ...):
        # Stage RECEIVE: discard IQ entirely. Tune + acquire already
        # happened upstream; we don't even compute PSD.
        if self.stage <= EngineStage.RECEIVE:
            self._scan_count += 1
            return self._make_snapshot(capture_time, groups)

        # ... existing PSD compute + DC excise + cell updates +
        # group rebuild + display buffer ...

        # Stage PROCESS: skip processor + tracks; everything else runs.
        if self.stage == EngineStage.PROCESS:
            self._scan_count += 1
            return self._make_snapshot(capture_time, groups)

        # Stage CLASSIFY: full pipeline (current behaviour).
        # processor.process_fine + track_mgr updates...
```

`detect_enabled` retires. Nothing outside the GUI used it; the GUI is being rewritten.

## 4. GUI segmented control

`drone_detector_enhanced.py` top bar:

```
   [● Idle]   [○ Receive]   [○ Process]   [○ Classify]
```

* `QButtonGroup` with `setExclusive(True)`, four `QPushButton(checkable=True)` children, each `~110px` wide.
* Click handler: `_on_stage_clicked(stage)` → `_set_stage(stage)`.
* Active stage **and every stage to its left** are filled with their stage colour. Stages to the right stay dim. Makes the "strict superset" visual immediately.
* Colour scheme:
  * Idle — `#444` background, `#888` text. Off / disconnected look.
  * Receive — `#0a4a4a` filled, cyan text. Raw IQ flowing.
  * Process — `#0a4a0a` filled, green text. PSD + display + cells.
  * Classify — `#5a3000` filled, orange text. Full detection.
* The currently-selected button is checked; that gives PyQt its built-in pressed-look in addition to the colour.

Source / Scene / Proc combos are untouched — they configure the pipeline, they don't drive the stage.

Removed entirely: `connect_btn`, `detect_btn`, `try_connect`, `_refresh_primary_button_label`, `_refresh_detect_button_label`, `_on_detect_toggled`, `_apply_detection_state`, `_detect_enabled` flag.

## 5. Stage transition rules

`_set_stage(target_stage)` dispatches based on current and target:

| Transition | What happens |
|------------|--------------|
| Idle → Receive (or higher) | Build the source per Src combo (`PyAdiIQSource` or `SimulatedIQSource`), wrap in `SignalReader`, attach to engine, set `engine.stage = target`, start `SweepWorker`. |
| Active → Active (same connection) | `engine.set_stage(target)`. Worker keeps running. |
| Going down: Classify → Process | `track_mgr.clear_all()` + `engine.set_stage(PROCESS)`. Tracks were a Classify-stage artefact. |
| Going down: Process → Receive | `reset_detection_state()` (cells back to UNKNOWN, tracks cleared) + `engine.set_stage(RECEIVE)`. Displayed spectrum stays at last frame. |
| Receive → Idle (or any active → Idle) | Stop `SweepWorker`, detach reader, close source, refresh buttons. |

Same "reset on step-down" rule that today's Detect-OFF transition already follows.

## 6. Implementation order (commits)

| Commit | What lands |
|--------|-----------|
| **A** — Engine | `EngineStage` enum, `set_stage`, branch in `_process_capture`. Default stage `CLASSIFY`. Verify headless: at each stage the right amount of work runs. |
| **B** — GUI segmented control | Build the four-button group, styling, default selection (Idle). Hook click → `_set_stage` stub that only updates `self._stage`. No engine connection yet. |
| **C** — Connection lifecycle | Refactor `_connect_simulator` + `_connect_hardware` to take a `target_stage` parameter. Implement full `_set_stage` with the transition rules above. Retire `connect_btn`, `detect_btn`, `try_connect`. |
| **D** — Verify + docs | Offscreen GUI lifecycle: Idle → Receive → Process → Classify → Process → Receive → Idle. Verify stage transitions reset state correctly. Update `sdr_threat_detection_ProjectDefinition.md` §2 / §9 / §10 / §11 + `CLAUDE.md` quick index. |

## 7. Verification per phase

**A**: Headless — instantiate engine, set stage to each value, call step() N times, assert:
* `RECEIVE`: scan_count climbs, all cells stay `UNKNOWN`, no tracks.
* `PROCESS`: scan_count climbs, FPV @ 5.8 GHz cell becomes `ACTIVE`, no tracks created.
* `CLASSIFY`: scan_count climbs, FPV cell `ACTIVE`, OSCFAR classifications appear.

**B**: Offscreen Qt — instantiate window, confirm 4 buttons present, default checked is `Idle`, colour styles applied.

**C**: Offscreen Qt — drive the full stage ladder, observe:
* Idle → Receive starts worker, attaches engine, stage=`RECEIVE`.
* Receive → Process flips engine stage, no worker disturbance.
* Process → Classify flips engine stage, processor swap honoured.
* Classify → Process drops tracks but keeps cells.
* Process → Receive resets cells, drops tracks.
* Receive → Idle stops worker.
* Switching Src mid-cycle resets to Idle.

**D**: Repeat C with both Hardware and Simulator sources. Run OSCFAR on FPV scene under CLASSIFY — confirm classifications still match the 7-preset table.

## 8. Out of scope

* Engine-stage telemetry / logging. The state is observable through the GUI; persistent logs of stage transitions can land later.
* Per-stage performance budgets. The 4-stage model doesn't change per-step cost meaningfully — RECEIVE is faster than today's `Detect: OFF` (no PSD compute), PROCESS matches `Detect: OFF` exactly, CLASSIFY matches `Detect: ON`.
* Sim source aliasing fix from the SignalReader refactor follow-ups. Independent.
* `gen_analog_fpv` visual realism fixes. Independent.
