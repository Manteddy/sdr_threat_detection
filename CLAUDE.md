# Project context for AI agents in this repo

## Start here

Read `sdr-threat-detection-wiki/index.md` before touching code.
The wiki is the single source of truth for architecture, decisions, conventions, and footguns.

## House rules

1. **Read the wiki first.** Start with `index.md`, then the pages relevant to your task.

2. **Update the wiki whenever you change something meaningful.** Use this table:

   | Changed… | Update wiki page |
   |---|---|
   | Data flow, file layout, entry points, stage contract | `architecture.md` |
   | A significant design decision (why, not just what) | `decisions.md` |
   | Hardware constraints, tuning ranges, libiio behaviour | `hardware.md` |
   | Coding conventions, threading rules, default units | `conventions.md` |
   | Footgun discovered or confirmed | `footguns.md` |
   | Experiment recording format, IQ encoding, replay | `experiments.md` |

3. **`footguns.md` is load-bearing.** Each entry is a real incident or principle. Don't remove one without explaining why in the commit.

4. **Don't create a new planning doc when an existing one applies.** Update the matching plan:
   - simulator → `sdr_simulator_plan.md`
   - algorithm selector → `signal_processing_selector_plan.md`
   - new processor / pipeline change → `signal_processing_integration_plan.md`
   - experiment recording → `experiment_recording_plan.md`

5. **Prefer linking over duplicating.** The wiki is a map; detail lives in the plan docs and code comments.

6. **If the wiki disagrees with the code, fix one or the other in the same change — never leave them inconsistent.**

7. **Append an entry to `sdr-threat-detection-wiki/log.md`** when you make a meaningful wiki update.

8. **At the end of any implementation task, include this command** so the user can test immediately:
   ```
   ./run.sh
   ```

## Where things live (quick index)

- Single capture function: `spectrum_engine/signal_reader.py`
- IQSource ABC + dataclasses: `spectrum_engine/iq_source.py`
- Hardware source: `spectrum_engine/sources/pyadi.py`
- Replay source: `spectrum_engine/sources/replay.py`
- Simulator source + scene preview: `spectrum_engine/sim/`
- Processors: `signal_pipeline/` (registry + Classic + OSCFAR)
- Engine: `spectrum_engine/engine.py`
- GUI: `drone_detector_enhanced.py`
- Experiment recorder: `experiments/recorder.py`
- Plans: `signal_processing_selector_plan.md`, `signal_processing_integration_plan.md`,
  `sdr_simulator_plan.md`, `signal_reader_refactor_plan.md`, `gui_stage_ladder_plan.md`,
  `experiment_recording_plan.md`
