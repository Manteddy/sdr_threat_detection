# FPV Drone Detector for PlutoSDR

Python application that uses a PlutoSDR (or AD9361-based SDR) to detect FPV drones by monitoring the 5.8 GHz band. Displays dual-antenna signals (omni + directional) with SDR++ style spectrum and waterfall, plus strength-over-time for predefined monitor frequencies.

---

## Two Versions

| File | Description |
|------|-------------|
| `drone_detector.py` | **Simple** ? single FFT, fixed threshold, fast sweep |
| `drone_detector_enhanced.py` | **Enhanced** ? adds CFAR, adaptive baseline, distance estimation |

Both share identical acquisition code (single `rx()`, single FFT, same buffer/timing) so spectrum and waterfall look the same.

---

## Enhanced Features

- **CA-CFAR detection** ? adaptive threshold that tracks local noise per frequency bin; clearly separates real spikes from ground noise
- **Adaptive noise baseline** ? slow-tracking floor estimate that prevents signal contamination of noise reference
- **Distance estimation** ? log-distance path loss model + Kalman filter converts signal power to estimated range with confidence score
- **Proximity hysteresis** ? enter/exit thresholds (default 95 m / 120 m) prevent warning chatter near boundary
- **Optional Numba JIT** ? accelerates CFAR loops when `numba` is installed
- **AD9361 fastlock** ? opt-in profile-based retuning (off by default; can cause artifacts on some hardware)

All enhanced features have runtime toggles (checkboxes in the UI) and fall back gracefully to fixed-threshold detection.

---

## Workflow

### 1. Startup

- Creates main window with dark theme
- Initializes sweep range (5.6?6.0 GHz), monitor frequencies (5.800, 5.900, 5.920 GHz), threshold (-40 dBFS)
- Sets up spectrum + waterfall plots for RX1 (omni) and RX2 (directional)
- Adds interactive crosshairs for frequency/dB readout on hover

### 2. Connection

- Tries `ad9361` (IP) first, then `Pluto` (USB)
- Configures bandwidth (20 MHz), sample rate (20 MHz), gain, buffer size (2048)
- Detects dual-channel support (4+ IIO RX channels)
- Starts `SweepWorker` thread and display update timer (~80 ms)

### 3. Sweep Loop (Background Thread)

For each frequency step:
1. Set LO + wait 0.8 ms for PLL lock
2. `rx()` ? IQ samples (2048 complex samples)
3. Single FFT with Blackman window ? dBFS spectrum (512 bins)
4. Fill `spectrum_omni` / `spectrum_dir`
5. After full sweep ? downsample to 800 cols ? emit `sweep_done`

### 4. Sweep Results (Main Thread, Enhanced Only)

```
_on_sweep_done() ?
  - Append waterfall lines
  - Update adaptive noise baseline (slow EWMA, signal-resistant)
  - Compute CFAR threshold array for full spectrum
  - For each monitor freq:
      - CFAR detection around �25 bins
      - Hybrid decision: CFAR OR fixed threshold (safety fallback)
      - Update distance estimator (Kalman filter)
      - Update warning counters with hysteresis
```

### 5. Display Update (Timer, ~80 ms)

- Current LO frequency label
- Warning label (proximity or signal alert)
- Spectrum curves + CFAR threshold overlay (yellow dash-dot)
- Waterfall images (normalized to noise floor)
- Monitor mini-plots (strength vs sweep #)
- Distance estimate + confidence per monitor frequency

---

## User Controls

| Control | Effect |
|---------|--------|
| **Sweep range** | Start/end GHz; Apply recomputes steps and restarts sweep |
| **Threshold** | dBFS level for fixed-threshold fallback (-100 to -10) |
| **Gain** | RX gain 0?73 dB |
| **WF Depth** | Waterfall history length (50?1000 lines) |
| **CFAR Scale** | dB above local noise for CFAR detection (1?20 dB) |
| **CFAR** | Toggle adaptive CFAR detection on/off |
| **Distance Est.** | Toggle distance estimation on/off |
| **Monitor frequencies** | Comma-separated GHz values to track |

---

## Data Flow

```
SDR (IQ) ? single FFT ? dB spectrum ? [spectrum plot, waterfall]
                                           ?
                                     CFAR threshold ? detection decision
                                           ?
                                     Kalman filter ? distance estimate
                                           ?
                                     Hysteresis ? Warning display
```

---

## Main Components

| Component | Role |
|-----------|------|
| **SweepWorker** | Background thread: SDR sweep, single FFT per step |
| **DroneDetector** | Main window: UI, CFAR, baseline, distance model, display |
| **cfar_threshold()** | Vectorized CA-CFAR with optional Numba dispatch |
| **cfar_detect_at_bin()** | Per-monitor-frequency CFAR detection |
| **DistanceEstimator** | Log-distance path loss + Kalman filter + hysteresis |
| **FastlockManager** | AD9361 fastlock profile setup and recall (opt-in) |

---

## Requirements

- PlutoSDR (ADALM-PLUTO) or AD9361-based SDR connected via USB or IP
- PlutoSDR USB drivers installed
- Python 3.8+

**Note:** Stock PlutoSDR is 325 MHz?3.8 GHz. For 5.8 GHz, use Pluto+ (AD9363) or a modified unit.

---

## Installation

```bash
pip install -r requirements.txt
```

Optional acceleration (recommended):

```bash
pip install numba
```

---

## Usage

```bash
python drone_detector.py           # simple version
python drone_detector_enhanced.py  # enhanced version
```

1. Click **Connect**
2. Adjust sweep range, threshold, and gain if needed
3. Toggle **CFAR** and **Distance Est.** features as desired
4. Tune **CFAR Scale** (higher = fewer false alarms, lower = more sensitive)
5. Add monitor frequencies (e.g. `5.800, 5.900, 5.920`)
6. Watch spectrum and waterfall; yellow curve shows CFAR threshold; red warning for signals; dark red for proximity

---

## Distance Calibration

The distance model uses default parameters (reference power -30 dBFS at 10 m, path loss exponent 2.2). For accurate 100 m boundary warnings:

1. Place a known FPV transmitter at a measured distance (e.g. 10 m)
2. Note the peak dBFS reading at that distance
3. Update `DIST_REF_POWER_DBFS` and `DIST_REF_DISTANCE_M` constants in `drone_detector_enhanced.py`
4. Repeat at 50 m and 100 m; adjust `DIST_PATH_LOSS_EXP` to fit

Without calibration, distance estimates are approximate and environment-dependent.

---

## Dual Antenna Setup

For true dual-antenna (omni + directional), the PlutoSDR must be in 2r2t mode:

1. SSH into Pluto: `ssh root@192.168.2.1`
2. Set mode: `fw_setenv attr_name compatible` and ensure 2r2t is enabled
3. Or use firmware 0.33+ with dual RX enabled

With a single RX Pluto, both displays show the same antenna.
