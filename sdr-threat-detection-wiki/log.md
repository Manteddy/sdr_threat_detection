# Wiki Log

Append-only chronological record of wiki changes.

---

## [2026-05-31] init | Wiki initialized

Context sources scanned:
- `CLAUDE.md` + `sdr_threat_detection_ProjectDefinition.md` (primary — comprehensive project map)
- `engine_config.yaml` (runtime knobs, freq ranges, CFAR/scheduler/state params)
- `requirements.txt` (dependency list)
- Plan docs: `gui_stage_ladder_plan.md`, `sdr_simulator_plan.md`, `signal_processing_integration_plan.md`, `signal_processing_selector_plan.md`, `signal_reader_refactor_plan.md`
- Source tree structure: `spectrum_engine/`, `signal_pipeline/`, `proximity_alert/`

## [2026-05-31] update | Planned work populated in overview.md

Added 5 planned work items to `overview.md`: sweep edge-case improvements, decision-tree→ML classifier upgrade, signal recording button (with band-stitching note), classifier UI improvements, BetaFlight simulator integration.

## [2026-05-31] feature | Experiment recording and replay implemented

New modules: `experiments/recorder.py` (ExperimentRecorder, background I/O, int16 IQ storage),
`spectrum_engine/sources/replay.py` (ReplayIQSource, time-locked replay, get_next_command).
Engine hooks: `attach_recorder` / `detach_recorder` / `attach_replay_source` in `SpectrumEngine`.
Config: `ExperimentsCfg` block added to `EngineConfig` + `engine_config.yaml`.
GUI: REC button (top bar, hardware-only), `ExperimentDialog` modal, `ExperimentBrowserDock`
right-side QDockWidget, `Replay` added to Src combo. Plan doc: `experiment_recording_plan.md`.

---

Pages created:
- `index.md` — master index + agent quick-start
- `overview.md` — project purpose, features, simulator presets, planned work, out-of-scope
- `architecture.md` — directory tree, data flow, key entry points, stage contract, runtime axes, key constraints
- `dependencies.md` — core / hardware-only / optional packages with install commands
- `decisions.md` — 8 architectural decisions extracted from ProjectDefinition and plan docs

## [2026-05-31] consolidation | ProjectDefinition migrated into wiki; CLAUDE.md rewritten

`sdr_threat_detection_ProjectDefinition.md` deleted. All content migrated:
- §5 Hardware notes → new `hardware.md`
- §6 Conventions → new `conventions.md`
- §10 full footguns list (16 entries) → new `footguns.md`
- §12 Experiment recording → new `experiments.md`
- §7 processor recipe + performance budget → `architecture.md`
- §8 empirical results → `architecture.md`
- §4 run commands + headless script → `overview.md`
- Plan docs table → `architecture.md`
- Directory tree updated with `experiments/` and `sources/replay.py`
- Data flow diagram updated with `ReplayIQSource`

`CLAUDE.md` rewritten: removed `@import` of ProjectDefinition, replaced House rules with
wiki-centric update table (what changed → which wiki page), added `./run.sh` instruction,
updated quick-index with replay + experiments entries.
