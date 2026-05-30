# Project context for AI agents in this repo

The full project definition lives in another file (kept this filename
free so the doc can be shared / linked / read by non-Claude tools):

@sdr_threat_detection_ProjectDefinition.md

## House rules (load-bearing — also restated in §11 of the imported file)

1. **Read `sdr_threat_detection_ProjectDefinition.md` before touching code.** It is the map.
2. **Update it whenever architecture, files, controls, conventions, or installs change.** Same commit or immediately following. Mention which sections changed in the commit message.
3. **Don't create a new planning doc when an existing one applies.** Update the matching plan:
   - simulator → `sdr_simulator_plan.md`
   - algorithm selector → `signal_processing_selector_plan.md`
   - new processor / pipeline change → `signal_processing_integration_plan.md`
4. **Prefer linking over duplicating.** The project definition is a map; detail lives in the plans / code comments.
5. **Treat the "Things to avoid" section as load-bearing.** Each entry is a real incident or principle. Don't remove an entry without explaining why.
6. **If the doc disagrees with the code, fix one or the other in the same change — never leave them inconsistent.**

## Where things live (quick index)

- Sources: `spectrum_engine/sdr_backend.py` (hardware), `spectrum_engine/sim/` (simulator)
- Processors: `signal_pipeline/` (registry + Classic + OSCFAR)
- Engine: `spectrum_engine/engine.py` (owns processor, detect_enabled, reset_detection_state)
- GUI: `drone_detector_enhanced.py` (Src / Scene / Proc / Detect controls + Sim Preview panel)
- Plans: `signal_processing_selector_plan.md`, `signal_processing_integration_plan.md`, `sdr_simulator_plan.md`
