# Advanced Feature Guide (Enhanced Detector)

This document explains how the advanced processing features work in two ways:

1. **General concept**: how the method works in SDR (Software-Defined Radio) and signal processing.
2. **This program**: how it is implemented in `drone_detector_enhanced.py`.

The four key features are:

- **Welch PSD** (Welch Power Spectral Density)
- **CA-CFAR detection** (Cell-Averaging Constant False Alarm Rate)
- **Kalman-filtered distance estimation**
- **Hysteresis** (boundary and warning stability)

---

## 1) Welch PSD (Power Spectral Density)

### General concept

A single **FFT (Fast Fourier Transform)** from one short **IQ (In-phase and Quadrature)** block can be noisy and jumpy. IQ data is the raw complex signal from the SDR: the real part (I) and imaginary part (Q) together represent amplitude and phase of the received radio waves.

**Welch PSD** improves stability by:

1. Splitting IQ data into overlapping segments.
2. Applying a **window** (e.g. Blackman) to each segment—a smooth taper that reduces spectral leakage (false energy spreading into neighboring frequency bins).
3. Computing FFT power for each segment.
4. Averaging segment powers. This reduces **variance** (random fluctuation from one measurement to the next) of the spectrum estimate. The tradeoff is slight smoothing/latency and more CPU work.

### In this program

- Implemented in `_welch_psd_db(...)`.
- Configured by:
  - `WELCH_OVERLAP_FRAC = 0.5` — half of each segment overlaps the next.
  - `FFT_SIZE = 512` — each segment is 512 samples.
- Feature flag: `FEATURES["use_welch_psd"]` (UI checkbox: **Welch (alerts)**).

Important design choice:

- **Display stays single-FFT** for consistent visual behavior.
- Welch is used for **alerts and distance logic only**:
  - `spec_o = self.spectrum_omni_welch if use_welch else self.spectrum_omni`
  - `spec_d = self.spectrum_dir_welch if use_welch else self.spectrum_dir`

So if Welch is enabled, detection decisions and distance updates use a smoother PSD, but the plotted main spectrum remains unchanged.

---

## 2) CA-CFAR Detection

### General concept

**CFAR (Constant False Alarm Rate)** means the detector adjusts its threshold so that the rate of false alarms stays roughly constant even when noise levels change. Instead of one fixed threshold for all **bins** (frequency slots in the spectrum), CFAR uses a **local** threshold that adapts to nearby noise.

**CA (Cell-Averaging)** is the simplest CFAR variant: it averages neighboring cells to estimate noise.

For each **CUT (Cell Under Test)**—the bin we are checking for a signal:

1. **Guard cells**: ignore bins immediately next to the CUT so that signal energy does not leak into the noise estimate.
2. **Training cells**: average the bins on both sides (outside the guard region) to estimate local noise.
3. **Threshold** = local noise estimate + scale (an offset in **dB**, decibels—a logarithmic unit for power ratios).
4. **Detection**: declare a signal if the CUT exceeds this threshold.

Why useful: in **RF (radio frequency)** scans, noise and interference are not uniform across frequency. CFAR handles changing noise floors better than a global fixed threshold.

### In this program

Core constants:

- `CFAR_GUARD_CELLS = 4` — 4 bins on each side of the CUT are excluded from the noise estimate.
- `CFAR_TRAINING_CELLS = 16` — 16 bins on each side are averaged for noise.
- `CFAR_SCALE_DB = 6.0` — threshold is 6 dB above the estimated noise.
- `CFAR_MONITOR_SEARCH_HALF = 25` — when checking a monitor frequency, we search ±25 bins for the peak.

Functions:

- `_cfar_threshold_np(...)`: vectorized NumPy threshold array (uses convolution for speed).
- `_cfar_threshold_numba(...)`: optional faster Numba (Just-In-Time compiled) version.
- `cfar_threshold(...)`: dispatches to Numba or NumPy.
- `cfar_detect_at_bin(...)`: per-monitor-frequency local detection around the target bin.

Runtime behavior:

- If enabled (`use_cfar_detection`), CFAR thresholds are computed each sweep.
- For each monitored frequency, the code checks a local region around that bin and extracts:
  - `detected` — whether a signal was found
  - `peak_db` — peak power in dB
  - `noise_est_db` — estimated noise level in dB
  - `threshold_db` — adaptive threshold in dB

Safety fallback:

- Final detection uses **hybrid logic**:
  - `detected = cfar_detected or above_fixed`
- This means fixed-threshold detection remains active as backup if CFAR misses something.

---

## 3) Kalman Filter Distance Estimation

### General concept

**RSSI (Received Signal Strength Indicator)** or power-to-distance conversion is noisy and strongly environment-dependent. Walls, reflections, and antenna orientation all affect the reading. A simple direct conversion can flicker badly.

This detector uses two stages:

1. **Measurement model (log-distance path loss)**:
  - A formula that converts received power (in **dBFS**, decibels relative to Full Scale—i.e. relative to the ADC’s maximum) into a rough distance estimate.
  - Path loss describes how radio power decreases with distance; the log-distance model assumes power falls off roughly as 1/d^n in linear terms, or as a straight line in dB vs. log(distance).
2. **1D Kalman filter**:
  - A recursive algorithm that smooths the rough estimate over time and tracks uncertainty.
  - It combines a **prediction** (where we expect the distance to be) with a **measurement** (what the RSSI formula says) using a **gain** that depends on how much we trust each.

Kalman steps (scalar form):

- Predict uncertainty: `P_pred = P + Q`
- Gain: `K = P_pred / (P_pred + R)`
- Update estimate: `x = x + K * (z - x)`
- Update uncertainty: `P = (1 - K) * P_pred`

Where:

- `z` = current raw distance measurement from the path-loss formula
- `x` = filtered distance estimate (our best guess)
- `Q` = **process noise** — how much we expect the true distance to change between updates (e.g. drone moving)
- `R` = **measurement noise** — how unreliable we think each RSSI-based measurement is

### In this program

Class: `DistanceEstimator`

Main constants:

- `DIST_REF_POWER_DBFS = -30.0` — reference power at a known distance (calibration point).
- `DIST_REF_DISTANCE_M = 10.0` — that known distance in meters.
- `DIST_PATH_LOSS_EXP = 2.2` — path loss exponent (typically 2 for free space, higher with obstacles).
- `DIST_KALMAN_Q = 25.0` — process noise for the Kalman filter.
- `DIST_KALMAN_R = 400.0` — measurement noise for the Kalman filter.

Distance conversion:

- `power_to_distance(...)` uses the log-distance formula:
  - `dist = d0 * 10^((A0 - power_dbfs)/(10*n))`
  - where `d0` is reference distance, `A0` is reference power, `n` is path loss exponent.
- Then clamps the result to `[1, 10000]` meters.

Per-sweep update:

- In `_on_sweep_done(...)`:
  - `best_power = max(peak_o, peak_d)` — use the stronger of omni or directional antenna.
  - `self.distance_estimators[freq].update(best_power)`

Outputs kept per monitor frequency:

- `x` — filtered distance in meters
- `confidence` — derived from filter uncertainty `P` (higher when estimate is stable)
- `inside_boundary` — hysteresis state (see next section)

Notes:

- Distances are approximate unless calibrated for your RF chain, antennas, and environment.
- Confidence rises as filter uncertainty drops.

---

## 4) Hysteresis

### General concept

**Hysteresis** means using different thresholds for “turn on” and “turn off” to avoid rapid toggling near a boundary. The idea comes from physics (e.g. magnets): the state depends on history, not just the current value.

Without hysteresis:

- If the measured value oscillates near a single threshold, alarms flicker on and off repeatedly.

With hysteresis:

- **Enter** alert only when crossing a stricter threshold (e.g. distance drops below 95 m).
- **Exit** alert only after moving past a looser threshold (e.g. distance rises above 120 m).

The gap between enter and exit thresholds is the **hysteresis band**—a dead zone that prevents chatter.

### In this program

There are two hysteresis-like stabilizers:

#### A) Distance boundary hysteresis

Inside `DistanceEstimator.update(...)`:

- Enter boundary if `x < enter_m` (e.g. 95 m).
- Exit only if `x > exit_m` (e.g. 120 m).

Defaults:

- `DIST_ENTER_M = 95.0`
- `DIST_EXIT_M = 120.0`

So once marked “inside” near 100 m, the estimate must move farther away before the warning clears.

#### B) Detection persistence counter

In `_on_sweep_done(...)`:

- If detected: counter increases by `+2` (capped at a maximum).
- If not detected: counter decreases by `-1` (floor at 0).
- Alert triggers only when counter reaches `WARNING_PERSIST_COUNT`.

This acts as **temporal hysteresis** or **debouncing**—requiring several consecutive detections before raising an alert, and allowing a few misses before clearing it. It smooths out noisy on/off decisions over time.

---

## Feature Interaction (Program Flow)

For each sweep:

1. Acquire IQ and compute spectrum.
2. (Optional) Compute Welch PSD for decision paths.
3. (Optional) Compute CFAR thresholds and per-monitor detection.
4. Apply hybrid detect rule (CFAR OR fixed threshold).
5. Update warning persistence counters.
6. (Optional) Update per-frequency Kalman distance estimator.
7. UI warning logic:
  - Proximity warning if `inside_boundary` and confidence is sufficient.
  - Otherwise signal warning if persistence counter is high.

This layered design is why alerts are more stable than a plain “single FFT + single threshold” detector.

---

## Practical Tuning Tips

- **Too many false alerts**:
  - Increase `CFAR_SCALE_DB`
  - Raise fixed threshold slider
  - Increase persistence requirement
- **Missed weak signals**:
  - Lower `CFAR_SCALE_DB`
  - Lower fixed threshold
  - Enable Welch for smoother decision inputs
- **Distance too jumpy**:
  - Increase `DIST_KALMAN_R`
  - Increase `DIST_EXIT_M` relative to `DIST_ENTER_M`
- **Distance biased (too near/far)**:
  - Recalibrate `DIST_REF_POWER_DBFS`, `DIST_REF_DISTANCE_M`, `DIST_PATH_LOSS_EXP`

