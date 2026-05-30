# Activity Probability Mapping

This document describes how the **Activity Probability** strip in the GUI is computed, what causes a band’s score to rise or fall, and how that value is turned into color on screen.

The probability map answers one question per 20 MHz cell:

> *Given everything the engine has measured so far, how likely is it that this band currently contains RF activity?*

It is **not** a raw power meter. It is a **persistent, baseline-relative belief** that accumulates evidence across scans and decays when repeated measurements look like noise.

---

## End-to-end pipeline

```
IQ capture → PSD (dB) → per-cell metrics → log-odds update → cell.occupancy_prob
                                                              ↓
EngineSnapshot.occupancy_probs → GUI gamma + colormap → Activity Probability strip
```

| Stage | Location | What happens |
|-------|----------|--------------|
| Grid | `spectrum_engine/spectrum_grid.py` | Band split into 20 MHz cells (default 240 cells for 1.2–6.0 GHz) |
| Capture + PSD | `spectrum_engine/engine.py`, `spectrum_engine/psd.py` | Coarse or fine FFT; omni channel used as primary |
| DC excision | `engine.py` | Center 3 FFT bins replaced with local median (Pluto LO leakage) |
| Per-cell metrics | `engine.py::_process_capture` | Noise floor, baseline, `occ_frac`, `strong_bins` |
| Probability update | `spectrum_engine/occupancy.py` | Log-odds evidence → `occupancy_prob` |
| State machine | `occupancy.py::update_cell_state` | UNKNOWN → QUIET / SUSPECT → ACTIVE → TRACKED |
| Snapshot | `engine.py::_make_snapshot` | Full `occupancy_probs[]` array copied for UI |
| Display | `drone_detector_enhanced.py::_update_activity_maps` | Gamma lift + colormap → `ImageItem` strip |

Occupancy is only updated when the engine stage is **PROCESS** or **CLASSIFY** (`EngineStage >= PROCESS`). At **RECEIVE**, raw PSD is shown but cells are not updated.

---

## Frequency grid (what one “band” means)

Configured in `engine_config.yaml`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `frequency_range.start_hz` | 1.2 GHz | Grid start |
| `frequency_range.stop_hz` | 6.0 GHz | Grid end |
| `frequency_range.base_cell_bw_hz` | 20 MHz | Width of one probability cell |
| `hardware.default_bandwidth_hz` | 40 MHz | One coarse capture covers **two** adjacent cells |

Each cell stores:

- `occupancy_prob` — the value shown on the map (0.01 … 0.999)
- `occupancy_logodds` — internal accumulator (converted to prob after each update)
- `baseline_db` — slow EWMA of the noise floor
- `last_occ_frac`, `last_strong_bins` — last measurement’s detection inputs (logged for debug)

---

## Step 1: From PSD to detection metrics

After each capture, the engine finds all cells overlapping the measured window (`map_measurement_to_cells`) and, for each affected cell:

### Noise floor and peak

For the PSD bins falling inside the cell’s `[start_hz, stop_hz)`:

```python
noise_floor = percentile(cell_psd, 30)
peak        = max(cell_psd)
```

### Baseline (reference for “is there a signal?”)

`update_baseline()` maintains a **slow EWMA** of the noise floor:

- Normal learning rate: α = 0.02
- When suspicious (noise floor > baseline + 10 dB, or `coarse_suspicious`): α = 0.002

This prevents a persistent carrier from quickly being absorbed into the baseline, so the signal continues to register as excess energy on later scans.

### Two complementary occupancy metrics

Both are measured **relative to the tracked baseline**, not the current capture’s own percentile:

```python
excess       = cell_psd - baseline
occ_frac     = mean(excess > +12 dB)    # wideband occupancy fraction
strong_bins  = count(excess > +18 dB)   # narrowband strong-bin count
```

Constants in `engine.py`:

| Constant | Value | Role |
|----------|-------|------|
| `_OCC_MARGIN_DB` | 12 dB | “Occupied” threshold for wideband detection |
| `_STRONG_MARGIN_DB` | 18 dB | “Strong bin” threshold for narrowband carriers |

**Why two metrics?** A single FFT of pure noise still has ~10% of bins slightly above a low margin and one bin ~15 dB above the floor. Measuring against a generous baseline-relative margin keeps `occ_frac ≈ 0` on quiet bands, while counting **several** strong bins distinguishes a real narrowband carrier from a lone noise spike.

---

## Step 2: Log-odds evidence (what drives probability **up**)

Implemented in `update_occupancy_probability()` (`occupancy.py`).

Each time a cell is scanned, an **evidence** score is computed and added to `occupancy_logodds`. Probability is then:

```python
occupancy_prob = 1 / (1 + exp(-occupancy_logodds))
```

Starting value: `occupancy_prob = 0.01` (slightly skeptical prior).

### Positive evidence (probability rises)

Evidence is **additive within one scan** — both wideband and narrowband terms can fire on the same measurement.

#### Wideband signal (`occupied_bin_fraction`)

| Condition | Evidence | Typical cause |
|-----------|----------|---------------|
| `occ_frac > 0.30` | **+1.5** | Wide signal filling much of the 20 MHz cell |
| `occ_frac > 0.15` (config threshold) | **+0.7** | Moderate wideband occupancy |
| `occ_frac > 0.05` | **+0.15** | Weak wideband hint |

Config key: `states.occupied_bin_fraction_threshold` (default **0.15**).

#### Narrowband carrier (`strong_bin_count`)

| Condition | Evidence | Typical cause |
|-----------|----------|---------------|
| `strong_bins >= 6` | **+1.5** | Clear narrowband carrier (several adjacent strong bins) |
| `strong_bins >= 3` | **+0.7** | Probable narrowband carrier |

A lone noise spike typically produces **one** strong bin → no narrowband bonus.

### Negative evidence (probability falls)

| Condition | Evidence |
|-----------|----------|
| `occ_frac <= 0.05` **and** `strong_bins < 3` | **−0.35** |

There is no separate time-based decay loop in the running engine (`decay_occupancy()` exists but is not called). Quiet bands fade because **each scan that looks like noise applies −0.35 log-odds**.

### Accumulation behavior

Because updates use log-odds:

- **Repeated detections** on a bursty signal (e.g. drone hopping, or revisited every few hundred ms) push probability toward **0.999** within a handful of hits.
- **Intermittent activity** stays “warm” between hits: a quiet scan only pulls −0.35, so a band that was recently active remains visibly elevated until several consecutive quiet measurements.
- **Never-scanned cells** keep the prior **0.01** until first measurement.

Rough intuition (not exact, because log-odds is non-linear):

| Scenario | Scans needed (order of magnitude) |
|----------|-----------------------------------|
| Strong wideband hit (+1.5) | 1–2 scans → high red |
| Moderate hit (+0.7) | 3–5 scans → orange/yellow |
| Quiet scan (−0.35) | ~3–4 consecutive quiet scans to pull a warm band back toward dark |

---

## Step 3: Cell state machine (related but separate from color)

`update_cell_state()` uses `occupancy_prob` plus thresholds from `engine_config.yaml`:

| Threshold | Default | Transition |
|-----------|---------|------------|
| `quiet_prob_threshold` | 0.15 | → QUIET |
| `suspect_prob_threshold` | 0.35 | → SUSPECT |
| `active_prob_threshold` | 0.65 | → ACTIVE |

The **map color uses raw `occupancy_prob`**, not the discrete state label. A cell can appear cyan/yellow on the map while still classified SUSPECT, or red while ACTIVE.

State **does** feed back into scheduling (see below) and into activity-group formation.

---

## Step 4: Downstream uses of occupancy probability

### Scheduler scan score

`compute_cell_scan_score()` adds **`+2.0 × occupancy_prob`** to each cell’s scheduling score, among other terms (state, priority band weight, energy delta, staleness). Higher probability → more likely to be picked for **priority revisits** (1 in 4 scheduler cycles), but coverage sweeps still visit all cells.

### Activity groups

Adjacent cells with `occupancy_prob > 0.35` and state not QUIET/UNKNOWN are merged into `ActivityGroup` objects. Group probability uses independent-cell OR:

```
P(group active) = 1 − ∏(1 − p_i)
```

Groups can trigger **fine scans** when their combined probability is high enough and cooldown has elapsed.

---

## Step 5: GUI display mapping

Implemented in `_update_activity_maps()` (`drone_detector_enhanced.py`).

### Input

```python
occ = snapshot.occupancy_probs   # float32 array, one value per cell
```

### Display transform (does not affect detection)

```python
prob      = clip(occ, 0, 1)
prob_disp = prob ** 0.5          # PROB_DISPLAY_GAMMA = 0.5
```

Gamma **0.5** (square root) lifts low-but-nonzero probabilities so emerging activity is visible before it reaches red. Example: `prob = 0.09` → `prob_disp ≈ 0.30` (visible blue/cyan rather than near-black).

### Colormap

`_probability_colormap()` maps `prob_disp ∈ [0, 1]` through 7 stops:

| Display value | Color | Meaning |
|---------------|-------|---------|
| 0.0 | Dark navy `(10, 10, 35)` | Quiet / prior |
| ~0.15 | Deep blue | Very low belief |
| ~0.35 | Cyan | Emerging activity |
| ~0.55 | Green-yellow | Moderate |
| ~0.75 | Orange | High |
| 1.0 | Red | Very high (≈ confirmed active) |

Legend on the panel: **dark = quiet → red = likely active**.

### Spatial layout

- One column per cell along the frequency (X) axis.
- Strip tiled to 8 rows (`N_MAP_ROWS`) for reliable pyqtgraph rendering.
- `setRect()` is applied **after** `setImage()` so cell indices align with GHz axis.
- Semi-transparent overlays mark configured priority bands (1.2G, 2.4G, 5.8G).

### Debug outputs

When `MAP_DEBUG_OVERLAY = True`:

- On-panel text: min/median/max of raw and display values, top cell details.
- `spectrum_logs/map_colors.log` — periodic RGB dumps.
- `spectrum_logs/detection.log` — data-level `occ_frac`, `strong_bins`, `prob` per top cell.
- `spectrum_logs/live_prob.png` / `data_prob.png` — visual ground truth.

---

## What makes a band light up on the map?

Summary checklist — **all** require the cell to actually be **scanned** (only affected cells update each cycle):

1. **RF energy above baseline** in that 20 MHz slice when measured.
2. Either:
   - **Wideband**: >15% of bins more than 12 dB above baseline (`occ_frac`), or
   - **Narrowband**: ≥3 bins more than 18 dB above baseline (`strong_bins`).
3. **Repeated positive evidence** across revisits (log-odds accumulation).
4. **Baseline not corrupted** — suspicious captures slow baseline learning so carriers stay visible.
5. **No DC spike** — center bins excised so LO leakage does not false-light every tuned window.

What does **not** directly raise probability:

- Being in a priority band (1.2G / 2.4G / 5.8G) — only affects **how often** the cell is scanned, not the evidence math.
- Raw peak power alone — peak feeds `energy_delta_db` for scheduling, but occupancy uses baseline-relative bin fractions.
- CFAR / classifier / track confidence — those run at CLASSIFY stage and affect tracks, not the coarse occupancy update.

---

## Interaction with the Scan Coverage Tracker

The two strips are **independent**:

| Strip | Data source | Meaning |
|-------|-------------|---------|
| **Activity Probability** | `occupancy_prob` (persistent belief) | “How suspicious is this band?” |
| **Scan Coverage Tracker** | `exp(−age / 2.5 s)` from `last_scan_time` | “Was this band scanned recently?” |

A band can be **bright red on probability** but **dark on scan coverage** if it has not been revisited lately (belief persists from earlier hits). Conversely, a band can **flash bright on scan coverage** during a full sweep while staying **dark blue on probability** if the measurement looked like noise.

After the scheduler fix (tier-based round-robin), multiple suspicious bands (e.g. ~1.8 GHz and ~2.2 GHz) should all show elevated probability **and** receive frequent priority revisits, not just the single highest-scoring cell.

---

## Key files

| File | Responsibility |
|------|----------------|
| `spectrum_engine/engine.py` | PSD, DC excision, per-cell metrics, snapshot assembly |
| `spectrum_engine/occupancy.py` | Log-odds update, state machine, activity groups |
| `spectrum_engine/baseline.py` | Noise-floor EWMA baseline |
| `spectrum_engine/scheduler.py` | Uses `occupancy_prob` in scan scoring and priority selection |
| `spectrum_engine/spectrum_grid.py` | Cell dataclass and grid builder |
| `engine_config.yaml` | Thresholds, margins, priority bands |
| `drone_detector_enhanced.py` | Probability strip rendering (gamma, LUT, layout) |

---

## Tuning knobs

| Knob | File | Effect on map |
|------|------|---------------|
| `_OCC_MARGIN_DB` / `_STRONG_MARGIN_DB` | `engine.py` | Sensitivity vs false alarms |
| Evidence table (+1.5 / +0.7 / −0.35) | `occupancy.py` | Speed of rise and fall |
| `occupied_bin_fraction_threshold` | `engine_config.yaml` | Wideband moderate-evidence gate |
| `PROB_DISPLAY_GAMMA` | `drone_detector_enhanced.py` | Visual lift of low probabilities |
| `_probability_colormap()` stops | `drone_detector_enhanced.py` | Color appearance only |
| Scheduler `_PRIORITY_EVERY` | `scheduler.py` | How often high-prob cells get extra revisits vs full sweep |

---

## ASCII flow (single cell, one scan)

```
capture at f_c
    │
    ▼
PSD bins in [cell.start, cell.stop)
    │
    ├── update_baseline(noise_floor)
    │
    ├── occ_frac  = fraction(bins > baseline + 12 dB)
    ├── strong    = count(bins > baseline + 18 dB)
    │
    ▼
evidence = f(occ_frac, strong)     ← +1.5 / +0.7 / +0.15 / −0.35
    │
    ▼
logodds += evidence  →  clamp  →  occupancy_prob
    │
    ├── update_cell_state()        ← SUSPECT / ACTIVE / …
    │
    ▼
EngineSnapshot.occupancy_probs[i]
    │
    ▼
prob_disp = sqrt(prob)  →  colormap  →  strip pixel color
```
