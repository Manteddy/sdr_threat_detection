# SDR Threat Detection — Wiki Index

> Last updated: 2026-05-31

## Pages

| Page | Description |
|------|-------------|
| [overview.md](overview.md) | Project purpose, features, scope, run commands, and current status |
| [architecture.md](architecture.md) | File structure, data-flow diagram, major components, processor recipe |
| [hardware.md](hardware.md) | PlutoSDR constraints, tuning ranges, buffer ordering, ADC scale |
| [conventions.md](conventions.md) | Coding conventions: units, threading, imports, test approach |
| [footguns.md](footguns.md) | Load-bearing "don't do this" list — real incidents and invariants |
| [experiments.md](experiments.md) | Experiment recording, on-disk format, IQ encoding, replay mode |
| [decisions.md](decisions.md) | Significant technical and design decisions with rationale |
| [dependencies.md](dependencies.md) | External libraries, optional packages, and version notes |
| [log.md](log.md) | Chronological record of wiki changes |

## Quick-start for agents

1. Read this index first.
2. For task context check **overview.md** (what/why/run) and **architecture.md** (where/how).
3. For constraints check **footguns.md** before writing any hardware or engine code.
4. For "why was X done this way" check **decisions.md**.
5. For hardware-specific limits check **hardware.md**.
6. After any session that changes code or docs, append an entry to **log.md**.
