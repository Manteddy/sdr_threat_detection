"""
FPV Drone Detector for PlutoSDR (Enhanced)
SDR++ style spectrum + waterfall.
Monitors predefined frequencies with strength-over-time.

Enhancements over the simple detector:
  - CA-CFAR adaptive detection (works alongside fixed threshold)
  - Adaptive noise-floor baseline tracking
  - Distance estimation via log-distance path loss + Kalman filter
  - Optional Numba JIT acceleration for CFAR
  - AD9361 fastlock profile support (opt-in, off by default)

Note: Acquisition and display are identical to the simple detector to
ensure stable waterfall and no ghost frequencies.
"""

import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
np.seterr(all="ignore")
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QCheckBox,
    QScrollArea, QFrame, QSplitter, QLineEdit, QDoubleSpinBox, QSpinBox,
    QComboBox,
)
from PyQt5.QtCore import QTimer, Qt, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg

from proximity_alert.engine import AlertEngine
from proximity_alert.widget import ProximityAlertPanel

# ---- Adaptive Spectrum Sensing Engine ----
try:
    from spectrum_engine import SpectrumEngine, EngineSnapshot, SignalReader
    from spectrum_engine.sources.pyadi import PyAdiIQSource
    from spectrum_engine.config import load_config as _load_engine_config
    from spectrum_engine.tracks import TrackState
    HAS_ENGINE = True
except Exception as _eng_err:
    HAS_ENGINE = False
    _eng_err_msg = str(_eng_err)

# ---- SDR Simulator (no libiio / pyadi-iio dependency) ----
HAS_SIM = False
try:
    from spectrum_engine.sim import SimulatedIQSource
    from spectrum_engine.sim import scenarios as sim_scenarios
    HAS_SIM = True
except Exception:
    pass

# ---- Pluggable signal-processing algorithms ----
HAS_PIPELINE = False
try:
    from signal_pipeline import list_processors, get_processor
    HAS_PIPELINE = True
except Exception:
    pass

# ---- Optional acceleration ----
HAS_NUMBA = False
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    pass

# ---- Default Configuration (matches simple detector) ----
DEFAULT_START_GHZ = 1.2
DEFAULT_END_GHZ   = 6.0
BANDWIDTH_HZ   = 20_000_000
SAMPLE_RATE    = 20_000_000
FFT_SIZE       = 512
SWEEP_STEP_HZ  = 20_000_000
BINS_PER_STEP  = FFT_SIZE
PLL_SETTLE_S   = 0.0008

DEFAULT_WF_LINES = 200
WATERFALL_COLS   = 800

MONITOR_HISTORY_LEN = 300
DEFAULT_MONITORS = [5.650, 5.900, 5.920]
DEFAULT_THRESHOLD_DBFS = -40
WARNING_PERSIST_COUNT = 2

# ---- Welch (for alerts/distance only; display stays single-FFT) ----
WELCH_OVERLAP_FRAC = 0.5

# ---- CFAR ----
CFAR_GUARD_CELLS = 4
CFAR_TRAINING_CELLS = 16
CFAR_SCALE_DB = 6.0
CFAR_MONITOR_SEARCH_HALF = 25

# ---- Adaptive baseline ----
BASELINE_ALPHA = 0.02
BASELINE_MARGIN_DB = 10.0

# ---- Distance model ----
DIST_REF_POWER_DBFS = -30.0
DIST_REF_DISTANCE_M = 10.0
DIST_PATH_LOSS_EXP = 2.2
DIST_ENTER_M = 95.0
DIST_EXIT_M = 120.0
DIST_KALMAN_Q = 25.0
DIST_KALMAN_R = 400.0

# ---- dBFS calibration (matches simple detector) ----
ADC_FULL_SCALE = 2048.0
WINDOW = np.blackman(FFT_SIZE).astype(np.float32)
WINDOW_SUM = float(np.sum(WINDOW))

# ---- SDR++ Style ----
BG_COLOR       = "#1a1a2e"
GRID_ALPHA     = 0.15
SPECTRUM_PEN   = (0, 255, 200, 220)
SPECTRUM_FILL  = (0, 255, 200, 25)
DB_MIN         = -140.0
DB_MAX         = 0.0
DYNAMIC_RANGE  = 60.0

BTN_STYLE = (
    "QPushButton{background:#2a2a4a;color:#0ff;border:1px solid #444;padding:3px;}"
    "QPushButton:hover{background:#3a3a5a;}"
)
INPUT_STYLE = "color:#ccc;background:#222;border:1px solid #444;padding:3px;"

# ---- Feature flags ----
FEATURES = {
    "use_welch_psd": True,
    "use_adaptive_baseline": True,
    "use_cfar_detection": True,
    "use_numba": HAS_NUMBA,
    "use_distance_model": True,
    "use_fastlock": False,
}

# ---- UI panels ----
# Monitor-frequency panel is superseded by the Spectrum Activity Maps when the
# adaptive engine is in use. Set to True to restore the old per-frequency panel.
SHOW_MONITOR_PANEL = False

# Scan-coverage "fire up" glow: wall-clock decay time-constant (seconds).
# A freshly scanned cell flares to 1.0 and fades to 1/e after this many seconds.
# Kept short so frequently-revisited bands (e.g. an active 2.4 GHz, hit every
# ~0.3 s) stay bright while the once-per-sweep coverage trail (~5 s) fades to
# dark -> drastic contrast between often-revisited and rarely-visited bands.
SCAN_HEAT_TAU_S = 2.5

# Perceptual gamma applied to the scan-coverage heat for display only. gamma < 1
# keeps partially-faded (recently-but-not-just scanned) cells clearly visible.
SCAN_DISPLAY_GAMMA = 0.5

# Perceptual gamma applied to the probability map for display only (does NOT
# change detection). gamma < 1 lifts low-but-nonzero probabilities so a band
# that is starting to show activity is clearly visible instead of dark blue.
PROB_DISPLAY_GAMMA = 0.5


# ---------------------------------------------------------------- Utilities
def _welch_psd_db(iq, fft_size=FFT_SIZE, overlap_frac=WELCH_OVERLAP_FRAC):
    """Welch PSD in dBFS from IQ. Used for alerts/distance only."""
    iq_norm = iq / ADC_FULL_SCALE
    n = len(iq_norm)
    if n < fft_size * 2:
        seg = iq_norm[:fft_size]
        X = np.fft.fftshift(np.fft.fft(seg * WINDOW))
        db = 20.0 * np.log10(np.abs(X) / WINDOW_SUM + 1e-20)
        return np.clip(db, DB_MIN, DB_MAX).astype(np.float32)
    hop = int(fft_size * (1.0 - overlap_frac))
    n_seg = max(1, (n - fft_size) // hop + 1)
    powers = np.zeros((n_seg, fft_size), dtype=np.float64)
    for i in range(n_seg):
        start = i * hop
        seg = iq_norm[start:start + fft_size]
        if len(seg) < fft_size:
            break
        X = np.fft.fftshift(np.fft.fft(seg * WINDOW))
        powers[i] = (np.abs(X) / WINDOW_SUM) ** 2
    avg_power = np.mean(powers[:n_seg], axis=0) if n_seg > 1 else powers[0]
    db = 10.0 * np.log10(avg_power + 1e-20)
    return np.clip(db, DB_MIN, DB_MAX).astype(np.float32)


def _clamp(val, lo, hi, default=None):
    v = float(val)
    if not np.isfinite(v):
        return default if default is not None else lo
    return max(lo, min(hi, v))


def _sdr_colormap():
    colors = [
        (0, 0, 20), (0, 0, 100), (0, 50, 180), (0, 160, 220),
        (0, 200, 80), (200, 220, 0), (255, 80, 0), (255, 0, 0), (255, 200, 200),
    ]
    return pg.ColorMap(np.linspace(0, 1, len(colors)), colors).getLookupTable(0, 1, 256)


def _probability_colormap():
    """LUT for the activity-probability map: dark-blue -> cyan -> yellow -> red."""
    colors = [
        (10, 10, 35), (0, 60, 140), (0, 160, 200),
        (40, 210, 120), (220, 220, 0), (255, 110, 0), (255, 0, 0),
    ]
    return pg.ColorMap(np.linspace(0, 1, len(colors)), colors).getLookupTable(0, 1, 256)


def _scan_colormap():
    """LUT for the scan-coverage tracker: black -> orange -> white (cold -> just scanned)."""
    colors = [
        (8, 8, 16), (60, 30, 0), (160, 80, 0),
        (255, 150, 0), (255, 220, 120), (255, 255, 255),
    ]
    return pg.ColorMap(np.linspace(0, 1, len(colors)), colors).getLookupTable(0, 1, 256)


# ----------------------------------------------------------- CFAR Detection
def _cfar_threshold_np(spectrum_db, guard, train, scale):
    """Vectorised CA-CFAR threshold via convolution (NumPy)."""
    n = len(spectrum_db)
    half = guard + train
    if n < 2 * half + 1:
        return np.full(n, DB_MAX, dtype=np.float32)
    kernel = np.zeros(2 * half + 1, dtype=np.float64)
    n_train = 2 * train
    kernel[:train] = 1.0 / n_train
    kernel[-train:] = 1.0 / n_train
    padded = np.pad(spectrum_db.astype(np.float64), half, mode="edge")
    noise_est = np.convolve(padded, kernel, mode="valid")[:n]
    return (noise_est + scale).astype(np.float32)


if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _cfar_threshold_numba(spectrum_db, guard, train, scale):
        n = len(spectrum_db)
        threshold = np.empty(n, dtype=np.float32)
        half = guard + train
        for i in range(n):
            if i < half or i >= n - half:
                threshold[i] = spectrum_db[i] + scale
                continue
            total = 0.0
            count = 0
            for j in range(i - half, i - guard):
                total += spectrum_db[j]
                count += 1
            for j in range(i + guard + 1, i + half + 1):
                total += spectrum_db[j]
                count += 1
            noise = total / count if count > 0 else -140.0
            threshold[i] = noise + scale
        return threshold


def cfar_threshold(spectrum_db, guard=CFAR_GUARD_CELLS,
                   train=CFAR_TRAINING_CELLS, scale=CFAR_SCALE_DB):
    """Compute CA-CFAR adaptive threshold array, dispatching to fastest backend."""
    if FEATURES["use_numba"] and HAS_NUMBA:
        return _cfar_threshold_numba(
            spectrum_db.astype(np.float32), guard, train, scale)
    return _cfar_threshold_np(spectrum_db, guard, train, scale)


def cfar_detect_at_bin(spectrum_db, center_bin, search_half=CFAR_MONITOR_SEARCH_HALF,
                       guard=CFAR_GUARD_CELLS, train=CFAR_TRAINING_CELLS,
                       scale=CFAR_SCALE_DB):
    """
    CFAR detection around a single monitor frequency bin.
    Returns (detected, peak_db, noise_est_db, threshold_db).
    """
    n = len(spectrum_db)
    lo = max(0, center_bin - search_half)
    hi = min(n, center_bin + search_half + 1)
    local = spectrum_db[lo:hi]
    n_local = len(local)
    if n_local == 0:
        return False, DB_MIN, DB_MIN, DB_MAX
    peak_idx = int(np.argmax(local))
    peak_db = float(local[peak_idx])
    half = guard + train
    if peak_idx >= half and peak_idx < n_local - half:
        left = local[peak_idx - half:peak_idx - guard]
        right = local[peak_idx + guard + 1:peak_idx + half + 1]
        noise_est = float(np.mean(np.concatenate([left, right])))
    else:
        mask = np.ones(n_local, dtype=bool)
        exc_lo = max(0, peak_idx - guard)
        exc_hi = min(n_local, peak_idx + guard + 1)
        mask[exc_lo:exc_hi] = False
        noise_est = float(np.median(local[mask])) if np.any(mask) else float(np.median(local))
    threshold_db = noise_est + scale
    detected = peak_db > threshold_db
    return detected, peak_db, noise_est, threshold_db


# ------------------------------------------------------ Distance Estimation
class DistanceEstimator:
    """Log-distance path loss model + 1-D Kalman filter for RSSI → distance."""

    def __init__(self, ref_power=DIST_REF_POWER_DBFS, ref_dist=DIST_REF_DISTANCE_M,
                 path_loss_n=DIST_PATH_LOSS_EXP,
                 enter_m=DIST_ENTER_M, exit_m=DIST_EXIT_M,
                 kalman_q=DIST_KALMAN_Q, kalman_r=DIST_KALMAN_R):
        self.A0 = ref_power
        self.d0 = ref_dist
        self.n = path_loss_n
        self.enter_m = enter_m
        self.exit_m = exit_m
        self.x = 1000.0
        self.P = 10000.0
        self.Q = kalman_q
        self.R = kalman_r
        self.inside_boundary = False
        self.confidence = 0.0

    def power_to_distance(self, power_dbfs):
        delta = self.A0 - power_dbfs
        if delta <= 0:
            return max(1.0, self.d0)
        dist = self.d0 * (10.0 ** (delta / (10.0 * self.n)))
        return _clamp(dist, 1.0, 10000.0, 1000.0)

    def update(self, power_dbfs):
        z = self.power_to_distance(power_dbfs)
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * P_pred
        self.confidence = _clamp(1.0 - (self.P / (self.P + 100.0)), 0.0, 1.0, 0.0)
        if not self.inside_boundary and self.x < self.enter_m:
            self.inside_boundary = True
        elif self.inside_boundary and self.x > self.exit_m:
            self.inside_boundary = False
        return self.x, self.confidence, self.inside_boundary

    def reset(self):
        self.x = 1000.0
        self.P = 10000.0
        self.inside_boundary = False
        self.confidence = 0.0


# ------------------------------------------------------ Fastlock Management
class FastlockManager:
    """Manages AD9361 fastlock profiles for faster frequency hopping."""

    def __init__(self):
        self.available = False
        self.profiles = {}
        self.max_profiles = 8

    def try_setup(self, sdr, frequencies_hz):
        """Attempt to setup fastlock profiles. Returns True if successful."""
        try:
            ctrl = getattr(sdr, "_ctrl", None)
            if ctrl is None:
                return False
            phy = None
            for dev in ctrl.devices:
                if hasattr(dev, "name") and dev.name and "ad9361" in dev.name:
                    phy = dev
                    break
            if phy is None:
                return False
            rx_lo_ch = None
            for ch in phy.channels:
                if hasattr(ch, "id") and "altvoltage0" in str(getattr(ch, "id", "")):
                    rx_lo_ch = ch
                    break
            if rx_lo_ch is None:
                return False
            has_fl = any("fastlock" in str(a) for a in getattr(rx_lo_ch, "attrs", {}))
            if not has_fl:
                return False
            freqs = list(frequencies_hz)[:self.max_profiles]
            for i, freq_hz in enumerate(freqs):
                try:
                    sdr.rx_lo = int(freq_hz)
                    time.sleep(0.002)
                    rx_lo_ch.attrs["fastlock_store"].value = str(i)
                    self.profiles[int(freq_hz)] = i
                except Exception:
                    continue
            self.available = len(self.profiles) > 0
            return self.available
        except Exception:
            self.available = False
            return False

    def recall(self, sdr, freq_hz):
        idx = self.profiles.get(int(freq_hz))
        if idx is None:
            return False
        try:
            ctrl = sdr._ctrl
            for dev in ctrl.devices:
                if hasattr(dev, "name") and dev.name and "ad9361" in dev.name:
                    for ch in dev.channels:
                        cid = str(getattr(ch, "id", ""))
                        if "altvoltage0" in cid:
                            ch.attrs["fastlock_recall"].value = str(idx)
                            return True
        except Exception:
            pass
        return False


# ------------------------------------------------------------ Sweep Thread
class SweepWorker(QThread):
    """
    Runs the SDR acquisition loop.

    When a SpectrumEngine is attached (via configure_engine), it runs the
    adaptive engine.step() loop and emits engine_update snapshots.
    Otherwise it falls back to the classic fixed-step sweep for backward
    compatibility and emits sweep_done as before.
    """
    sweep_done = pyqtSignal(object, object, float)
    engine_update = pyqtSignal(object)   # carries an EngineSnapshot
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.sdr = None
        self.sweep_start_hz = 0
        self.num_steps = 0
        self.dual_channel = False
        self.spectrum_omni = None
        self.spectrum_dir = None
        self.spectrum_omni_welch = None
        self.spectrum_dir_welch = None
        self.current_freq = 0
        self.sweep_col = 0
        self.features = dict(FEATURES)
        self.fastlock = FastlockManager()

        # Engine-mode state
        self._engine = None
        self._backend = None

    def configure(self, sdr, start_hz, num_steps, dual, spec_omni, spec_dir,
                  spec_omni_welch=None, spec_dir_welch=None,
                  features=None, fastlock=None):
        self.sdr = sdr
        self.sweep_start_hz = start_hz
        self.num_steps = num_steps
        self.dual_channel = dual
        self.spectrum_omni = spec_omni
        self.spectrum_dir = spec_dir
        self.spectrum_omni_welch = spec_omni_welch
        self.spectrum_dir_welch = spec_dir_welch
        if features is not None:
            self.features = dict(features)
        if fastlock is not None:
            self.fastlock = fastlock

    def configure_engine(self, engine, backend):
        """Attach a SpectrumEngine + SignalReader to use the adaptive scan loop."""
        self._engine = engine
        self._backend = backend

    def run(self):
        self._running = True

        if self._engine is not None and self._backend is not None:
            self._run_engine_loop()
        else:
            self._run_classic_loop()

    def _run_engine_loop(self):
        """Adaptive engine-driven scan loop (Phases 1-3)."""
        engine = self._engine
        last_emit = 0.0
        # Cap the snapshot emit rate so the (heavy) GUI slot never falls behind
        # the engine and renders stale state. The snapshot is cumulative, so
        # skipped steps lose nothing: each cell's scan age is read from the
        # engine's actual last_scan_time, which persists across steps.
        emit_interval = 1.0 / 20.0
        while self._running:
            try:
                now = time.monotonic()
                snapshot = engine.step(now=now)

                # Update current_freq for the freq_label
                if snapshot.last_command is not None:
                    self.current_freq = int(snapshot.last_command.center_hz)

                # Also keep the classic spectrum buffers updated from the
                # engine's rolling display PSD so the waterfall still works.
                self._fill_classic_buffers_from_snapshot(snapshot)

                if (now - last_emit) >= emit_interval:
                    self.engine_update.emit(snapshot)
                    last_emit = now

            except Exception as e:
                self.error_occurred.emit(str(e)[:80])
                time.sleep(0.1)

    def _fill_classic_buffers_from_snapshot(self, snapshot):
        """
        Copy the engine's full-spectrum display PSD into the shared spectrum_omni
        buffer so the existing waterfall rendering path keeps working.
        """
        if self.spectrum_omni is None:
            return
        try:
            psd = snapshot.last_psd_db
            freq = snapshot.last_psd_freq_hz
            if len(psd) == 0 or len(freq) == 0:
                return

            lo_hz = freq[0]
            hi_hz = freq[-1]
            total = len(self.spectrum_omni)

            # Compute the slice of the classic buffer that this PSD covers
            span_start = self.sweep_start_hz
            step_hz = SWEEP_STEP_HZ
            # Map measured freq range to buffer indices
            lo_idx = max(0, int((lo_hz - span_start) / step_hz * BINS_PER_STEP))
            hi_idx = min(total, int((hi_hz - span_start) / step_hz * BINS_PER_STEP) + 1)
            if hi_idx <= lo_idx:
                return

            dest_slice = hi_idx - lo_idx
            interp_psd = np.interp(
                np.linspace(lo_hz, hi_hz, dest_slice),
                freq, psd,
            ).astype(np.float32)
            self.spectrum_omni[lo_idx:hi_idx] = interp_psd
            if self.spectrum_dir is not None:
                self.spectrum_dir[lo_idx:hi_idx] = interp_psd
        except Exception:
            pass

    def _run_classic_loop(self):
        """Original fixed-step sweep loop (fallback / backward compat)."""
        while self._running and self.sdr:
            self.current_freq = self.sweep_start_hz
            self.sweep_col = 0
            try:
                self.sdr.rx_destroy_buffer()
            except Exception:
                pass

            ref_nf = -80.0
            ok = True
            for col in range(self.num_steps):
                if not self._running:
                    return
                try:
                    nf = self._do_one_step(col)
                    ref_nf = 0.1 * nf + 0.9 * ref_nf
                except Exception as e:
                    self.error_occurred.emit(str(e)[:60])
                    time.sleep(0.1)
                    ok = False
                    break
                self.sweep_col = col + 1
                self.current_freq += SWEEP_STEP_HZ

            if ok and self._running:
                total = self.num_steps * BINS_PER_STEP
                x_full = np.arange(total, dtype=np.float32)
                x_ds = np.linspace(0, total - 1, WATERFALL_COLS)
                wf_o = np.interp(x_ds, x_full, self.spectrum_omni[:total]).astype(np.float32)
                wf_d = np.interp(x_ds, x_full, self.spectrum_dir[:total]).astype(np.float32)
                ref_nf = _clamp(ref_nf, -140, -20, -80)
                self.sweep_done.emit(wf_o, wf_d, ref_nf)

    def _do_one_step(self, col):
        """Single step — identical to simple detector."""
        self.sdr.rx_destroy_buffer()
        self.sdr.rx_lo = int(self.current_freq)
        time.sleep(PLL_SETTLE_S)
        raw = self.sdr.rx()

        if self.dual_channel:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                omni_iq = np.asarray(raw[0], dtype=np.complex64).ravel()
                dir_iq = np.asarray(raw[1], dtype=np.complex64).ravel()
            elif isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[0] >= 2:
                omni_iq = raw[0].astype(np.complex64).ravel()
                dir_iq = raw[1].astype(np.complex64).ravel()
            else:
                omni_iq = np.asarray(raw, dtype=np.complex64).ravel()
                dir_iq = omni_iq.copy()
        else:
            omni_iq = np.asarray(raw, dtype=np.complex64).ravel()
            dir_iq = omni_iq.copy()

        n = len(omni_iq)
        if n < FFT_SIZE:
            omni_iq = np.pad(omni_iq, (0, FFT_SIZE - n))
            dir_iq = np.pad(dir_iq, (0, FFT_SIZE - n))

        omni_norm = omni_iq[:FFT_SIZE] / ADC_FULL_SCALE
        dir_norm = dir_iq[:FFT_SIZE] / ADC_FULL_SCALE

        omni_fft = np.fft.fftshift(np.fft.fft(omni_norm * WINDOW))
        dir_fft = np.fft.fftshift(np.fft.fft(dir_norm * WINDOW))

        omni_db = 20.0 * np.log10(np.abs(omni_fft) / WINDOW_SUM + 1e-20)
        dir_db = 20.0 * np.log10(np.abs(dir_fft) / WINDOW_SUM + 1e-20)
        omni_db = np.clip(omni_db, DB_MIN, DB_MAX).astype(np.float32)
        dir_db = np.clip(dir_db, DB_MIN, DB_MAX).astype(np.float32)

        s = col * BINS_PER_STEP
        e = s + BINS_PER_STEP
        self.spectrum_omni[s:e] = omni_db
        self.spectrum_dir[s:e] = dir_db

        if self.features.get("use_welch_psd") and self.spectrum_omni_welch is not None:
            omni_welch = _welch_psd_db(omni_iq)
            dir_welch = _welch_psd_db(dir_iq)
            self.spectrum_omni_welch[s:e] = omni_welch
            self.spectrum_dir_welch[s:e] = dir_welch

        return float(np.median(omni_db))

    def stop(self):
        self._running = False
        self.wait(3000)


# ------------------------------------------------------------ Main Window
class DroneDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPV Drone Detector - PlutoSDR")
        self.setMinimumSize(1200, 800)
        self.resize(1500, 1000)
        self.setStyleSheet("background-color:#0e0e1a;color:#ccc;")

        self.sdr = None
        self.connected = False
        self.dual_channel = False
        self.features = dict(FEATURES)

        self.sweep_start_hz = int(DEFAULT_START_GHZ * 1e9)
        self.sweep_end_hz = int(DEFAULT_END_GHZ * 1e9)
        self.reference_noise_floor = -80.0
        self.wf_max_lines = DEFAULT_WF_LINES

        self.monitor_freqs_ghz = list(DEFAULT_MONITORS)
        self.monitor_history = {}
        self.monitor_widgets = {}
        self.warning_counters = {}
        self.distance_estimators = {}
        self.cfar_scale_db = CFAR_SCALE_DB

        # Engine state
        self._engine = None
        self._engine_backend = None
        self._engine_scan_count = 0

        # Scan-coverage "fire up" heat buffer (lazily sized on first snapshot)
        self._scan_heat = None
        self._last_map_time = None

        self._recalc_sweep()
        self._init_monitor_history()

        self.fastlock = FastlockManager()

        # --- Sim preview panel state ---------------------------------
        # The right-hand column (formerly RX2 directional antenna view)
        # is repurposed as a continuous ground-truth view of whatever
        # simulator scene is currently selected. Driven by a separate
        # timer; the main display update loop does NOT write to it.
        self._sim_preview_n_bins = 2048
        self._sim_preview_start_ghz = 1.0
        self._sim_preview_stop_ghz = 6.5
        self._sim_preview_freq_ghz = np.linspace(
            self._sim_preview_start_ghz, self._sim_preview_stop_ghz,
            self._sim_preview_n_bins, dtype=np.float64,
        )
        self._sim_preview_psd_db = np.full(
            self._sim_preview_n_bins, DB_MIN, dtype=np.float32,
        )
        self._sim_preview_wf = deque(maxlen=DEFAULT_WF_LINES)
        self._sim_preview_rng = np.random.default_rng(42)
        self._sim_preview_active = False
        # Detect flag mirrors engine.detect_enabled — default OFF
        # so the operator can inspect the raw signal first and opt in.
        self._detect_enabled = False
        # ------------------------------------------------------------

        self._setup_ui()
        self._setup_plots()
        self._setup_sim_preview_panel()
        self._setup_crosshairs()
        self._setup_alert_window()

        self.sweep_worker = SweepWorker()
        self.sweep_worker.sweep_done.connect(self._on_sweep_done)
        self.sweep_worker.engine_update.connect(self._on_engine_update)
        self.sweep_worker.error_occurred.connect(self._on_sweep_error)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_display)
        self._sim_preview_timer = QTimer(self)
        self._sim_preview_timer.timeout.connect(self._refresh_sim_preview)

    # -------------------------------------------------------- Sweep math
    def _recalc_sweep(self):
        span = self.sweep_end_hz - self.sweep_start_hz
        self.num_steps = max(1, int(span / SWEEP_STEP_HZ))
        self.total_bins = self.num_steps * BINS_PER_STEP

        self.spectrum_omni = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.spectrum_dir = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.spectrum_omni_welch = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.spectrum_dir_welch = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.waterfall_omni = deque(maxlen=self.wf_max_lines)
        self.waterfall_dir = deque(maxlen=self.wf_max_lines)

        self.baseline_omni = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.baseline_dir = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self._baseline_init = False

        self.cfar_thresh_omni = np.full(self.total_bins, DB_MAX, dtype=np.float32)
        self.cfar_thresh_dir = np.full(self.total_bins, DB_MAX, dtype=np.float32)

        self.freq_axis_ghz = np.zeros(self.total_bins, dtype=np.float64)
        for i in range(self.num_steps):
            center_hz = self.sweep_start_hz + i * SWEEP_STEP_HZ
            lo = center_hz - SWEEP_STEP_HZ / 2
            hi = center_hz + SWEEP_STEP_HZ / 2
            s = i * BINS_PER_STEP
            self.freq_axis_ghz[s:s + BINS_PER_STEP] = \
                np.linspace(lo, hi, BINS_PER_STEP, endpoint=False) / 1e9

    def _init_monitor_history(self):
        new = {}
        for f in self.monitor_freqs_ghz:
            if f in self.monitor_history:
                new[f] = self.monitor_history[f]
            else:
                new[f] = {
                    "omni": deque(maxlen=MONITOR_HISTORY_LEN),
                    "dir": deque(maxlen=MONITOR_HISTORY_LEN),
                }
        self.monitor_history = new
        self.warning_counters = {f: 0 for f in self.monitor_freqs_ghz}
        new_de = {}
        for f in self.monitor_freqs_ghz:
            if f in self.distance_estimators:
                new_de[f] = self.distance_estimators[f]
            else:
                new_de[f] = DistanceEstimator()
        self.distance_estimators = new_de

    def _freq_to_bin(self, freq_ghz):
        if self.total_bins == 0:
            return 0
        idx = int(np.searchsorted(self.freq_axis_ghz, freq_ghz))
        return max(0, min(self.total_bins - 1, idx))

    # --------------------------------------------------------------- UI
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # ---- Top bar ----
        top = QHBoxLayout()
        self.status_label = QLabel("Disconnected")
        self.status_label.setFont(QFont("Consolas", 9))
        self.status_label.setStyleSheet("color:#888;padding:2px;")
        top.addWidget(self.status_label)

        self.warning_label = QLabel("")
        self.warning_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.warning_label.setAlignment(Qt.AlignCenter)
        self.warning_label.setMinimumWidth(300)
        self.warning_label.setStyleSheet("color:transparent;padding:4px;")
        top.addWidget(self.warning_label, 1)

        self.freq_label = QLabel("-- GHz")
        self.freq_label.setFont(QFont("Consolas", 14, QFont.Bold))
        self.freq_label.setStyleSheet("color:#0ff;padding:2px;")
        top.addWidget(self.freq_label)

        # ---- Source selector (Hardware vs Simulator) ----
        top.addWidget(self._lbl("Src:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Hardware")
        if HAS_SIM and HAS_ENGINE:
            self.source_combo.addItem("Simulator")
        self.source_combo.setFixedWidth(100)
        self.source_combo.setStyleSheet(INPUT_STYLE)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self.source_combo)

        # ---- Simulator scene preset (runtime-switchable) ----
        if HAS_SIM and HAS_ENGINE:
            top.addWidget(self._lbl(" Scene:"))
            self.scene_combo = QComboBox()
            for label, _ in sim_scenarios.GUI_PRESETS:
                self.scene_combo.addItem(label)
            default_idx = next(
                (i for i, (lbl, _) in enumerate(sim_scenarios.GUI_PRESETS)
                 if "5.8 GHz" in lbl),
                0,
            )
            self.scene_combo.setCurrentIndex(default_idx)
            self.scene_combo.setFixedWidth(210)
            self.scene_combo.setStyleSheet(INPUT_STYLE)
            self.scene_combo.setEnabled(False)  # only meaningful in Simulator mode
            self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)
            top.addWidget(self.scene_combo)
        else:
            self.scene_combo = None

        # ---- Signal-processing algorithm selector (runtime-switchable) ----
        if HAS_PIPELINE and HAS_ENGINE:
            top.addWidget(self._lbl(" Proc:"))
            self.proc_combo = QComboBox()
            for name, label in list_processors():
                self.proc_combo.addItem(label, userData=name)
            self.proc_combo.setCurrentIndex(0)  # default = Classic
            self.proc_combo.setFixedWidth(180)
            self.proc_combo.setStyleSheet(INPUT_STYLE)
            self.proc_combo.currentIndexChanged.connect(self._on_proc_changed)
            top.addWidget(self.proc_combo)
        else:
            self.proc_combo = None

        self.connect_btn = QPushButton("Receiver On")
        self.connect_btn.setFixedWidth(140)
        self.connect_btn.clicked.connect(self.try_connect)
        top.addWidget(self.connect_btn)

        # Detect toggle (default OFF — operator opts in to detection)
        self.detect_btn = QPushButton("Detect: OFF")
        self.detect_btn.setFixedWidth(120)
        self.detect_btn.setCheckable(True)
        self.detect_btn.setChecked(False)
        self.detect_btn.toggled.connect(self._on_detect_toggled)
        top.addWidget(self.detect_btn)

        self._refresh_primary_button_label()
        self._refresh_detect_button_label()
        root.addLayout(top)

        # ---- Controls bar row 1 ----
        ctrl = QHBoxLayout()

        ctrl.addWidget(self._lbl("Sweep:"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.325, 6.0)
        self.start_spin.setDecimals(3)
        self.start_spin.setSuffix(" GHz")
        self.start_spin.setValue(DEFAULT_START_GHZ)
        self.start_spin.setSingleStep(0.1)
        self.start_spin.setFixedWidth(120)
        self.start_spin.setStyleSheet(INPUT_STYLE)
        ctrl.addWidget(self.start_spin)

        ctrl.addWidget(self._lbl(" \u2192 "))

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.325, 6.0)
        self.end_spin.setDecimals(3)
        self.end_spin.setSuffix(" GHz")
        self.end_spin.setValue(DEFAULT_END_GHZ)
        self.end_spin.setSingleStep(0.1)
        self.end_spin.setFixedWidth(120)
        self.end_spin.setStyleSheet(INPUT_STYLE)
        ctrl.addWidget(self.end_spin)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(55)
        apply_btn.setStyleSheet(BTN_STYLE)
        apply_btn.clicked.connect(self._apply_sweep_range)
        ctrl.addWidget(apply_btn)

        ctrl.addSpacing(15)
        ctrl.addWidget(self._lbl("Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(-100, -10)
        self.threshold_slider.setValue(DEFAULT_THRESHOLD_DBFS)
        self.threshold_slider.setMaximumWidth(150)
        self.threshold_slider.valueChanged.connect(self._on_threshold)
        ctrl.addWidget(self.threshold_slider)
        self.threshold_label = QLabel(f"{DEFAULT_THRESHOLD_DBFS} dBFS")
        self.threshold_label.setFixedWidth(60)
        self.threshold_label.setStyleSheet("color:#aaa;")
        ctrl.addWidget(self.threshold_label)

        ctrl.addSpacing(15)
        ctrl.addWidget(self._lbl("Gain:"))
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 73)
        self.gain_slider.setValue(40)
        self.gain_slider.setMaximumWidth(120)
        self.gain_slider.valueChanged.connect(self._on_gain)
        ctrl.addWidget(self.gain_slider)
        self.gain_label = QLabel("40 dB")
        self.gain_label.setFixedWidth(45)
        self.gain_label.setStyleSheet("color:#aaa;")
        ctrl.addWidget(self.gain_label)

        ctrl.addSpacing(15)
        ctrl.addWidget(self._lbl("WF Depth:"))
        self.wf_depth_spin = QSpinBox()
        self.wf_depth_spin.setRange(50, 1000)
        self.wf_depth_spin.setValue(DEFAULT_WF_LINES)
        self.wf_depth_spin.setSingleStep(50)
        self.wf_depth_spin.setSuffix(" lines")
        self.wf_depth_spin.setFixedWidth(100)
        self.wf_depth_spin.setStyleSheet(INPUT_STYLE)
        self.wf_depth_spin.valueChanged.connect(self._on_wf_depth)
        ctrl.addWidget(self.wf_depth_spin)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- Controls bar row 2 (CFAR + feature toggles) ----
        ctrl2 = QHBoxLayout()

        ctrl2.addWidget(self._lbl("CFAR Scale:"))
        self.cfar_spin = QDoubleSpinBox()
        self.cfar_spin.setRange(1.0, 20.0)
        self.cfar_spin.setDecimals(1)
        self.cfar_spin.setSuffix(" dB")
        self.cfar_spin.setValue(CFAR_SCALE_DB)
        self.cfar_spin.setSingleStep(0.5)
        self.cfar_spin.setFixedWidth(90)
        self.cfar_spin.setStyleSheet(INPUT_STYLE)
        self.cfar_spin.valueChanged.connect(self._on_cfar_scale)
        ctrl2.addWidget(self.cfar_spin)

        ctrl2.addSpacing(15)
        self.chk_welch = QCheckBox("Welch (alerts)")
        self.chk_welch.setChecked(self.features["use_welch_psd"])
        self.chk_welch.setStyleSheet("color:#aaa;")
        self.chk_welch.setToolTip("Use Welch PSD for alerts and distance (display unchanged)")
        self.chk_welch.toggled.connect(lambda v: self._set_feature("use_welch_psd", v))
        ctrl2.addWidget(self.chk_welch)

        self.chk_cfar = QCheckBox("CFAR")
        self.chk_cfar.setChecked(self.features["use_cfar_detection"])
        self.chk_cfar.setStyleSheet("color:#aaa;")
        self.chk_cfar.toggled.connect(lambda v: self._set_feature("use_cfar_detection", v))
        ctrl2.addWidget(self.chk_cfar)

        self.chk_dist = QCheckBox("Distance Est.")
        self.chk_dist.setChecked(self.features["use_distance_model"])
        self.chk_dist.setStyleSheet("color:#aaa;")
        self.chk_dist.toggled.connect(lambda v: self._set_feature("use_distance_model", v))
        ctrl2.addWidget(self.chk_dist)

        ctrl2.addSpacing(10)
        accel_str = "Numba" if HAS_NUMBA else "no accel"
        self.accel_label = QLabel(f"[{accel_str}]")
        self.accel_label.setStyleSheet("color:#666;")
        ctrl2.addWidget(self.accel_label)

        ctrl2.addStretch()
        root.addLayout(ctrl2)

        # ---- Engine status bar ----
        if HAS_ENGINE:
            eng_bar = QHBoxLayout()
            eng_bar.addWidget(self._lbl("Engine:"))
            self.engine_status_label = QLabel("Idle")
            self.engine_status_label.setFont(QFont("Consolas", 8))
            self.engine_status_label.setStyleSheet("color:#666;padding:1px 4px;")
            eng_bar.addWidget(self.engine_status_label)

            self.engine_coverage_label = QLabel("")
            self.engine_coverage_label.setFont(QFont("Consolas", 8))
            self.engine_coverage_label.setStyleSheet("color:#666;padding:1px 4px;")
            eng_bar.addWidget(self.engine_coverage_label)

            self.engine_tracks_label = QLabel("")
            self.engine_tracks_label.setFont(QFont("Consolas", 8))
            self.engine_tracks_label.setStyleSheet("color:#0ff;padding:1px 4px;")
            eng_bar.addWidget(self.engine_tracks_label)

            eng_bar.addStretch()
            root.addLayout(eng_bar)
        else:
            self.engine_status_label = None
            self.engine_coverage_label = None
            self.engine_tracks_label = None

        # ---- Main vertical splitter: spectrum/waterfall vs monitor panel ----
        main_vsplit = QSplitter(Qt.Vertical)

        hsplit = QSplitter(Qt.Horizontal)

        self._ch_titles = ["RX1 (Omni)", "RX2 (Directional)"]
        for title, attr_spec, attr_wf in [
            (self._ch_titles[0], "spec_omni", "wf_omni"),
            (self._ch_titles[1], "spec_dir", "wf_dir"),
        ]:
            col_split = QSplitter(Qt.Vertical)
            spec = pg.PlotWidget()
            spec.setTitle(title, color="w", size="10pt")
            self._style_spectrum(spec)
            col_split.addWidget(spec)
            setattr(self, attr_spec, spec)

            wf = pg.PlotWidget()
            self._style_waterfall(wf)
            col_split.addWidget(wf)
            setattr(self, attr_wf, wf)

            col_split.setSizes([300, 400])
            hsplit.addWidget(col_split)

        main_vsplit.addWidget(hsplit)

        # Monitor panel (legacy; hidden when SHOW_MONITOR_PANEL is False)
        if SHOW_MONITOR_PANEL:
            mon_group = QGroupBox("Monitor Frequencies")
            mon_group.setStyleSheet(
                "QGroupBox{color:#aaa;border:1px solid #333;margin-top:4px;}"
                "QGroupBox::title{padding:0 4px;}"
            )
            mon_outer = QVBoxLayout()

            mon_ctrl = QHBoxLayout()
            mon_ctrl.addWidget(self._lbl("Frequencies (GHz):"))
            self.monitor_input = QLineEdit(
                ", ".join(f"{f:.3f}" for f in self.monitor_freqs_ghz)
            )
            self.monitor_input.setPlaceholderText("5.650, 5.900, 5.920")
            self.monitor_input.setStyleSheet(INPUT_STYLE)
            self.monitor_input.setFixedWidth(300)
            mon_ctrl.addWidget(self.monitor_input)
            mon_apply = QPushButton("Apply")
            mon_apply.setFixedWidth(55)
            mon_apply.setStyleSheet(BTN_STYLE)
            mon_apply.clicked.connect(self._apply_monitors)
            mon_ctrl.addWidget(mon_apply)
            mon_ctrl.addStretch()
            mon_outer.addLayout(mon_ctrl)

            self.monitor_scroll = QScrollArea()
            self.monitor_scroll.setWidgetResizable(True)
            self.monitor_scroll_widget = QWidget()
            self.monitor_layout = QVBoxLayout(self.monitor_scroll_widget)
            self.monitor_scroll.setWidget(self.monitor_scroll_widget)
            mon_outer.addWidget(self.monitor_scroll)

            mon_group.setLayout(mon_outer)
            main_vsplit.addWidget(mon_group)
        else:
            # Spectrum Activity Maps replace the monitor panel
            self._setup_activity_maps(main_vsplit)

        main_vsplit.setSizes([450, 350])
        root.addWidget(main_vsplit)

        if SHOW_MONITOR_PANEL:
            self._rebuild_monitor_widgets()

    def _lbl(self, text):
        l = QLabel(text)
        l.setStyleSheet("color:#aaa;")
        return l

    # ----------------------------------------- Spectrum Activity Maps
    def _engine_map_range_ghz(self):
        """Return (start_ghz, stop_ghz) of the engine's monitored range."""
        if HAS_ENGINE:
            try:
                cfg = _load_engine_config()
                return (cfg.frequency_range.start_hz / 1e9,
                        cfg.frequency_range.stop_hz / 1e9)
            except Exception:
                pass
        return (1.2, 6.0)

    def _update_activity_map_ranges(self):
        """Zoom the activity-map strips to the current active sweep window."""
        if not hasattr(self, "_prob_plot") or self._prob_plot is None:
            return
        s = self.sweep_start_hz / 1e9
        e = self.sweep_end_hz / 1e9
        # X-linked, so setting one updates both strips
        self._prob_plot.setXRange(s, e, padding=0)

    def _setup_activity_maps(self, parent_splitter):
        """
        Build the two frequency-aligned heatmap strips that replace the
        monitor-frequency panel:
          - Activity Probability map (per-cell occupancy probability)
          - Scan Coverage Tracker (cells flare when scanned, then fade)
        """
        maps_group = QGroupBox("Spectrum Activity Maps")
        maps_group.setStyleSheet(
            "QGroupBox{color:#aaa;border:1px solid #333;margin-top:4px;}"
            "QGroupBox::title{padding:0 4px;}"
        )
        maps_outer = QVBoxLayout()
        maps_outer.setContentsMargins(4, 4, 4, 4)
        maps_outer.setSpacing(2)

        # The maps span the engine's full monitored range (1.2-6.0 GHz by
        # default), independent of the legacy sweep range. The exact rect is
        # refined from cell centers once the first snapshot arrives.
        start_ghz, stop_ghz = self._engine_map_range_ghz()
        span_ghz = stop_ghz - start_ghz

        title_font = QFont("Segoe UI", 11, QFont.Bold)
        axis_font = QFont("Consolas", 9)

        def _style_map_plot(pw, title_html):
            pw.setBackground(BG_COLOR)
            pw.setXRange(start_ghz, stop_ghz, padding=0)
            pw.setYRange(0, 1)
            pw.hideAxis("left")
            pw.setTitle(title_html, size="11pt")
            pw.setLabel("bottom", "Frequency (GHz)", color="#bbb",
                        **{"font-size": "10pt"})
            ax = pw.getAxis("bottom")
            ax.setPen(pg.mkPen((70, 70, 110)))
            ax.setTextPen("#ddd")
            ax.setTickFont(axis_font)
            pw.setMouseEnabled(x=False, y=False)
            pw.setMenuEnabled(False)

        # ---- Probability strip ----
        self._prob_plot = pg.PlotWidget()
        _style_map_plot(
            self._prob_plot,
            "<span style='color:#dddddd'>Activity Probability</span>"
            "&nbsp;&nbsp;<span style='color:#888888;font-size:9pt'>"
            "dark = quiet &#8594; red = likely active</span>",
        )
        self._prob_img = pg.ImageItem(
            np.zeros((1, 1), dtype=np.float32), autoLevels=False, levels=(0, 1)
        )
        self._prob_img.setLookupTable(_probability_colormap())
        self._prob_img.setRect(QRectF(start_ghz, 0, span_ghz, 1))
        self._prob_plot.addItem(self._prob_img)
        self._add_priority_band_markers(self._prob_plot)
        maps_outer.addWidget(self._prob_plot)

        # ---- Scan coverage strip ----
        self._scan_plot = pg.PlotWidget()
        _style_map_plot(
            self._scan_plot,
            "<span style='color:#dddddd'>Scan Coverage Tracker</span>"
            "&nbsp;&nbsp;<span style='color:#888888;font-size:9pt'>"
            "bright = just scanned &#8594; dark = not scanned recently</span>",
        )
        self._scan_plot.setXLink(self._prob_plot)
        self._scan_img = pg.ImageItem(
            np.zeros((1, 1), dtype=np.float32), autoLevels=False, levels=(0, 1)
        )
        self._scan_img.setLookupTable(_scan_colormap())
        self._scan_img.setRect(QRectF(start_ghz, 0, span_ghz, 1))
        self._scan_plot.addItem(self._scan_img)
        maps_outer.addWidget(self._scan_plot)

        maps_group.setLayout(maps_outer)
        parent_splitter.addWidget(maps_group)

    def _add_priority_band_markers(self, plot):
        """Shade the configured priority bands on a strip for orientation."""
        bands = [
            (1.200, 1.350, "1.2G"),
            (2.300, 2.500, "2.4G"),
            (5.650, 5.950, "5.8G"),
        ]
        for lo, hi, name in bands:
            region = pg.LinearRegionItem(
                values=(lo, hi), movable=False,
                brush=pg.mkBrush(255, 255, 255, 18),
                pen=pg.mkPen(255, 255, 255, 40),
            )
            region.setZValue(10)
            plot.addItem(region)

    def _set_feature(self, key, value):
        self.features[key] = value

    def _on_cfar_scale(self, val):
        self.cfar_scale_db = val

    def _style_spectrum(self, pw):
        pw.setBackground(BG_COLOR)
        pw.showGrid(x=True, y=True, alpha=GRID_ALPHA)
        pw.setXRange(self.sweep_start_hz / 1e9, self.sweep_end_hz / 1e9, padding=0)
        pw.setYRange(-120, 0)
        pw.setLabel("left", "dBFS")
        pw.setLabel("bottom", "GHz")
        for a in ("bottom", "left"):
            pw.getAxis(a).setPen(pg.mkPen((40, 40, 80)))
            pw.getAxis(a).setTextPen("w")

    def _style_waterfall(self, pw):
        pw.setBackground(BG_COLOR)
        pw.setXRange(self.sweep_start_hz / 1e9, self.sweep_end_hz / 1e9, padding=0)
        pw.setYRange(0, self.wf_max_lines)
        pw.setLabel("bottom", "GHz")
        pw.hideAxis("left")
        pw.getAxis("bottom").setPen(pg.mkPen((40, 40, 80)))
        pw.getAxis("bottom").setTextPen("w")

    def _setup_plots(self):
        pen = pg.mkPen(color=SPECTRUM_PEN, width=1)
        fill = pg.mkBrush(*SPECTRUM_FILL)

        self.spec_omni.clear()
        self.spec_dir.clear()
        self.spec_omni_curve = self.spec_omni.plot(
            self.freq_axis_ghz, self.spectrum_omni,
            pen=pen, fillLevel=DB_MIN, fillBrush=fill
        )
        self.spec_dir_curve = self.spec_dir.plot(
            self.freq_axis_ghz, self.spectrum_dir,
            pen=pen, fillLevel=DB_MIN, fillBrush=fill
        )

        thresh_val = self.threshold_slider.value()
        self.thresh_line_omni = pg.InfiniteLine(
            pos=thresh_val, angle=0, pen=pg.mkPen("#f00", width=1, style=Qt.DashLine)
        )
        self.thresh_line_dir = pg.InfiniteLine(
            pos=thresh_val, angle=0, pen=pg.mkPen("#f00", width=1, style=Qt.DashLine)
        )
        self.spec_omni.addItem(self.thresh_line_omni)
        self.spec_dir.addItem(self.thresh_line_dir)

        cfar_pen = pg.mkPen("#ff0", width=1, style=Qt.DashDotLine)
        self.cfar_omni_curve = self.spec_omni.plot(
            self.freq_axis_ghz, self.cfar_thresh_omni, pen=cfar_pen)
        self.cfar_dir_curve = self.spec_dir.plot(
            self.freq_axis_ghz, self.cfar_thresh_dir, pen=cfar_pen)

        # Engine scan-target indicator lines (show where the engine is currently looking)
        scan_pen = pg.mkPen("#ff6600", width=1, style=Qt.SolidLine)
        self._scan_target_line_omni = pg.InfiniteLine(
            pos=self.sweep_start_hz / 1e9, angle=90, pen=scan_pen,
            label="↓ scan", labelOpts={"color": "#ff6600", "position": 0.95}
        )
        self._scan_target_line_dir = pg.InfiniteLine(
            pos=self.sweep_start_hz / 1e9, angle=90, pen=scan_pen,
        )
        self.spec_omni.addItem(self._scan_target_line_omni)
        self.spec_dir.addItem(self._scan_target_line_dir)
        self._scan_target_line_omni.setVisible(False)
        self._scan_target_line_dir.setVisible(False)

        cmap = _sdr_colormap()
        freq_span = (self.sweep_end_hz - self.sweep_start_hz) / 1e9
        empty = np.zeros((WATERFALL_COLS, 1), dtype=np.float32)

        for attr_wf, attr_img in [("wf_omni", "wf_omni_img"), ("wf_dir", "wf_dir_img")]:
            wf_widget = getattr(self, attr_wf)
            wf_widget.clear()
            img = pg.ImageItem(empty, autoLevels=False, levels=(0, 1))
            img.setLookupTable(cmap)
            img.setRect(QRectF(
                self.sweep_start_hz / 1e9, 0, freq_span, self.wf_max_lines
            ))
            wf_widget.addItem(img)
            setattr(self, attr_img, img)

    def _setup_sim_preview_panel(self):
        """Take over the RX2 column as the simulator ground-truth preview.

        The widgets keep their old variable names (spec_dir, wf_dir,
        spec_dir_curve, wf_dir_img) — only the title, X range, data, and
        overlays change. The main display update loop no longer writes to
        them; they are driven by `_refresh_sim_preview` exclusively.
        """
        self.spec_dir.setTitle(
            "Simulated Signal (1.0–6.5 GHz)", color="w", size="10pt",
        )
        self.spec_dir.setXRange(
            self._sim_preview_start_ghz, self._sim_preview_stop_ghz, padding=0,
        )
        # Re-seed the curve with the sim-preview frequency axis + empty data.
        self.spec_dir_curve.setData(
            self._sim_preview_freq_ghz, self._sim_preview_psd_db,
        )
        # Detection overlays belonged to the RX2 antenna view; hide them.
        if hasattr(self, "cfar_dir_curve"):
            self.cfar_dir_curve.setVisible(False)
        if hasattr(self, "thresh_line_dir"):
            self.thresh_line_dir.setVisible(False)
        if hasattr(self, "_scan_target_line_dir"):
            self._scan_target_line_dir.setVisible(False)
        # Waterfall rect: 1.0–6.5 GHz wide, full waterfall depth tall.
        span = self._sim_preview_stop_ghz - self._sim_preview_start_ghz
        self.wf_dir_img.setRect(QRectF(
            self._sim_preview_start_ghz, 0, span, self.wf_max_lines,
        ))

    def _refresh_sim_preview(self):
        """Timer slot — render the active simulator scene's analytical PSD.

        Runs while Source == Simulator and the scene_combo is populated.
        Does nothing in Hardware mode. The detection state of the engine
        is irrelevant here — this view is ground truth of the air.
        """
        if not (self._sim_preview_active and HAS_SIM):
            return
        if self.scene_combo is None:
            return
        try:
            scene = sim_scenarios.preset_by_label(self.scene_combo.currentText())
        except KeyError:
            return
        from spectrum_engine.sim import scene_psd_db
        _, psd = scene_psd_db(
            scene,
            self._sim_preview_start_ghz * 1e9,
            self._sim_preview_stop_ghz * 1e9,
            self._sim_preview_n_bins,
            rng=self._sim_preview_rng,
        )
        self._sim_preview_psd_db = psd
        self.spec_dir_curve.setData(self._sim_preview_freq_ghz, psd)
        # Waterfall — same normalization conventions as the omni waterfall.
        self._sim_preview_wf.append(psd.copy())
        if len(self._sim_preview_wf) >= 2:
            wf = np.array(list(self._sim_preview_wf), dtype=np.float32)
            nf = _clamp(self.reference_noise_floor, -140, -20, -80)
            wf_n = np.clip((wf - nf) / DYNAMIC_RANGE, 0, 1)
            self.wf_dir_img.setImage(wf_n.T, autoLevels=False, levels=(0, 1))
            span = self._sim_preview_stop_ghz - self._sim_preview_start_ghz
            self.wf_dir_img.setRect(QRectF(
                self._sim_preview_start_ghz, 0, span, len(self._sim_preview_wf),
            ))
            self.wf_dir.setYRange(0, len(self._sim_preview_wf))

    def _activate_sim_preview(self):
        """Show + start refreshing the sim preview (called when Src=Simulator)."""
        self._sim_preview_active = True
        self._sim_preview_wf.clear()
        # 5 Hz refresh — light, plenty for static scenes; visible motion when
        # the scene gets time-varying (drone_approach style scenarios).
        if not self._sim_preview_timer.isActive():
            self._sim_preview_timer.start(200)
        # Run one refresh immediately so the user sees the picked scene
        # without waiting up to 200 ms.
        self._refresh_sim_preview()

    def _deactivate_sim_preview(self):
        """Stop the timer and blank the panel (Hardware mode / startup)."""
        self._sim_preview_active = False
        if self._sim_preview_timer.isActive():
            self._sim_preview_timer.stop()
        self._sim_preview_psd_db[:] = DB_MIN
        self.spec_dir_curve.setData(
            self._sim_preview_freq_ghz, self._sim_preview_psd_db,
        )
        self._sim_preview_wf.clear()

    def _setup_crosshairs(self):
        self._crosshair_proxies = []
        self._crosshair_items = {}

        for key, spec_pw, wf_pw in [
            ("omni", self.spec_omni, self.wf_omni),
            ("dir", self.spec_dir, self.wf_dir),
        ]:
            cross_pen = pg.mkPen("#ff8800", width=1, style=Qt.DashLine)
            vline_s = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
            vline_w = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
            freq_text = pg.TextItem(color=(255, 136, 0), anchor=(0, 0))
            freq_text.setFont(QFont("Consolas", 9))
            db_text = pg.TextItem(color=(255, 136, 0), anchor=(0, 1))
            db_text.setFont(QFont("Consolas", 9))

            spec_pw.addItem(vline_s)
            wf_pw.addItem(vline_w)
            wf_pw.addItem(freq_text)
            spec_pw.addItem(db_text)

            for item in (vline_s, vline_w, freq_text, db_text):
                item.setVisible(False)

            items = {
                "vline_spec": vline_s, "vline_wf": vline_w,
                "freq_text": freq_text, "db_text": db_text,
                "spec_pw": spec_pw, "wf_pw": wf_pw,
            }
            self._crosshair_items[key] = items

            def make_wf_handler(k):
                def on_move(evt):
                    ci = self._crosshair_items[k]
                    pos = evt[0]
                    wf = ci["wf_pw"]
                    if not wf.sceneBoundingRect().contains(pos):
                        return
                    pt = wf.plotItem.vb.mapSceneToView(pos)
                    f = pt.x()
                    ci["vline_spec"].setPos(f)
                    ci["vline_wf"].setPos(f)
                    ci["vline_spec"].setVisible(True)
                    ci["vline_wf"].setVisible(True)
                    ci["freq_text"].setText(f"{f:.4f} GHz")
                    ci["freq_text"].setPos(f, pt.y())
                    ci["freq_text"].setVisible(True)
                    bidx = self._freq_to_bin(f)
                    db_val = self.spectrum_omni[bidx] if k == "omni" else self.spectrum_dir[bidx]
                    ci["db_text"].setText(f"{db_val:.1f} dBFS")
                    ci["db_text"].setPos(f, db_val)
                    ci["db_text"].setVisible(True)
                return on_move

            def make_spec_handler(k):
                def on_move(evt):
                    ci = self._crosshair_items[k]
                    pos = evt[0]
                    sp = ci["spec_pw"]
                    if not sp.sceneBoundingRect().contains(pos):
                        return
                    pt = sp.plotItem.vb.mapSceneToView(pos)
                    f = pt.x()
                    ci["vline_spec"].setPos(f)
                    ci["vline_wf"].setPos(f)
                    ci["vline_spec"].setVisible(True)
                    ci["vline_wf"].setVisible(True)
                    ci["freq_text"].setText(f"{f:.4f} GHz")
                    ci["freq_text"].setPos(f, 0)
                    ci["freq_text"].setVisible(True)
                    bidx = self._freq_to_bin(f)
                    db_val = self.spectrum_omni[bidx] if k == "omni" else self.spectrum_dir[bidx]
                    ci["db_text"].setText(f"{db_val:.1f} dBFS")
                    ci["db_text"].setPos(f, db_val)
                    ci["db_text"].setVisible(True)
                return on_move

            pw = pg.SignalProxy(
                wf_pw.scene().sigMouseMoved, rateLimit=60, slot=make_wf_handler(key)
            )
            ps = pg.SignalProxy(
                spec_pw.scene().sigMouseMoved, rateLimit=60, slot=make_spec_handler(key)
            )
            self._crosshair_proxies.extend([pw, ps])

    # ----------------------------------------------- Separate alert window
    def _setup_alert_window(self):
        self.alert_engine = AlertEngine(
            enter_m=DIST_ENTER_M,
            exit_m=DIST_EXIT_M,
            detection_threshold_dbfs=float(DEFAULT_THRESHOLD_DBFS),
            critical_m=100.0,
            approaching_m=250.0,
        )
        self._alert_window = QMainWindow()
        self._alert_window.setWindowTitle("Proximity Alert")
        self._alert_window.setMinimumSize(520, 260)
        self._alert_window.setStyleSheet("background:#0e0e1a;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("PROXIMITY ALERT")
        header.setFont(QFont("Segoe UI", 10))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color:#666;padding:4px;")
        layout.addWidget(header)

        self.alert_panel = ProximityAlertPanel(self.alert_engine)
        layout.addWidget(self.alert_panel)

        self._alert_window.setCentralWidget(container)
        self._alert_window.show()

    # -------------------------------------------------------- Connection
    def try_connect(self):
        """Dispatch to hardware or simulator path based on the Source combo."""
        source = self.source_combo.currentText() if hasattr(self, "source_combo") else "Hardware"
        if source == "Simulator":
            self._connect_simulator()
        else:
            self._connect_hardware()

    def _connect_hardware(self):
        try:
            import adi
        except (ImportError, TypeError, OSError):
            self.status_label.setText("Error: pip install pyadi-iio (needs libiio)")
            self.status_label.setStyleSheet("color:#f44;")
            return

        try:
            if self.sdr:
                del self.sdr
                self.sdr = None

            # Prefer the ad9361/IP transport: it exposes both RX channels and is
            # reliable. The USB Pluto fallback only maps a single channel and, on
            # this setup, fails to allocate RX buffers (errno 0) and drops off the
            # bus -> 0 scans. The IP (USB-ethernet) interface can take 10-20 s to
            # come up after the device enumerates, so retry it before falling back.
            self.sdr = None
            for _attempt in range(8):
                try:
                    self.sdr = adi.ad9361(uri="ip:192.168.2.1")
                    break
                except Exception:
                    try:
                        self.sdr = adi.ad9361(uri="ip:pluto.local")
                        break
                    except Exception:
                        time.sleep(1.5)
            if self.sdr is None:
                # Last resort: single-channel USB Pluto (least reliable here).
                self.sdr = adi.Pluto(uri="usb:")

            self.sdr.rx_rf_bandwidth = BANDWIDTH_HZ
            self.sdr.sample_rate = SAMPLE_RATE
            # Use same buffer size as simple detector (FFT_SIZE*4) for reliable dual-channel
            # layout; larger buffers can cause pyadi-iio to return interleaved format.
            self.sdr.rx_buffer_size = FFT_SIZE * 4
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.gain_slider.value()

            n_rx_ch = 0
            try:
                rxadc = self.sdr._rxadc
                if rxadc:
                    n_rx_ch = sum(
                        1 for ch in rxadc.channels
                        if hasattr(ch, 'scan_element') and ch.scan_element
                    )
            except Exception:
                pass

            self.dual_channel = False
            self._dual_ch_err = ""
            self._ch_verify = ""
            if n_rx_ch >= 4:
                try:
                    self.sdr.rx_enabled_channels = [0, 1]
                    self.sdr.gain_control_mode_chan1 = "manual"
                    self.sdr.rx_hardwaregain_chan1 = self.gain_slider.value()

                    n_identical = 0
                    n_tests = 5
                    for _ in range(n_tests):
                        self.sdr.rx_destroy_buffer()
                        td = self.sdr.rx()
                        if isinstance(td, (list, tuple)) and len(td) >= 2:
                            ch0 = np.asarray(td[0]).ravel()
                            ch1 = np.asarray(td[1]).ravel()
                        elif isinstance(td, np.ndarray) and td.ndim == 2:
                            ch0 = td[0].ravel()
                            ch1 = td[1].ravel()
                        else:
                            n_identical = n_tests
                            break
                        if np.array_equal(ch0, ch1):
                            n_identical += 1

                    rms0 = float(np.sqrt(np.mean(np.abs(ch0.astype(np.float64))**2)))
                    rms1 = float(np.sqrt(np.mean(np.abs(ch1.astype(np.float64))**2)))
                    same_obj = (ch0 is ch1)
                    corr = float(np.abs(np.corrcoef(
                        np.abs(ch0[:min(512, len(ch0))].astype(np.float64)),
                        np.abs(ch1[:min(512, len(ch1))].astype(np.float64)),
                    )[0, 1]))

                    self._ch_verify = (
                        f"identical={n_identical}/{n_tests}, same_ref={same_obj}, "
                        f"RMS ch0={rms0:.0f} ch1={rms1:.0f}, corr={corr:.3f}"
                    )

                    if n_identical < n_tests:
                        self.dual_channel = True
                    else:
                        self._dual_ch_err = f"ch0==ch1 in {n_identical}/{n_tests} reads"
                        self.sdr.rx_enabled_channels = [0]

                except Exception as ex:
                    self._dual_ch_err = f"{ex} ({n_rx_ch} IIO ch)"
                    try:
                        self.sdr.rx_enabled_channels = [0]
                    except Exception:
                        pass
            else:
                self._dual_ch_err = f"only {n_rx_ch} IIO scan channels (need 4 for MIMO)"

            # Attempt fastlock setup
            fl_status = ""
            if self.features["use_fastlock"]:
                step_freqs = [
                    self.sweep_start_hz + i * SWEEP_STEP_HZ
                    for i in range(self.num_steps)
                ]
                if self.fastlock.try_setup(self.sdr, step_freqs):
                    fl_status = f" | fastlock={len(self.fastlock.profiles)} profiles"

            adi_class = type(self.sdr).__name__
            self.connected = True
            if self.dual_channel:
                ch_info = f"DUAL RX via {adi_class} | {self._ch_verify}"
                self.spec_dir.setTitle("RX2 (Directional) - LIVE", color="#0f0", size="10pt")
            else:
                ch_info = f"single RX ({adi_class})"
                hint = self._dual_ch_err[:50] if self._dual_ch_err else "unknown"
                self.spec_dir.setTitle(
                    f"RX2 (mirrored) - {hint}",
                    color="#ff8800", size="9pt",
                )
            self.status_label.setText(
                f"Connected ({ch_info}{fl_status}) - {self.sweep_start_hz/1e9:.3f}-"
                f"{self.sweep_end_hz/1e9:.3f} GHz ({self.num_steps} steps)"
            )
            self.status_label.setStyleSheet("color:#0f0;")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self._disconnect)
            self._refresh_primary_button_label()

            self._reset_buffers()
            self.sweep_worker.configure(
                self.sdr, self.sweep_start_hz, self.num_steps,
                self.dual_channel, self.spectrum_omni, self.spectrum_dir,
                spec_omni_welch=self.spectrum_omni_welch,
                spec_dir_welch=self.spectrum_dir_welch,
                features=self.features, fastlock=self.fastlock,
            )

            # Attach adaptive sensing engine if available
            if HAS_ENGINE:
                try:
                    eng_cfg = _load_engine_config()
                    self._engine = SpectrumEngine(cfg=eng_cfg, telemetry_enabled=True)
                    # SignalReader is the single capture function shared
                    # with the simulator path. PyAdiIQSource handles only
                    # raw IQ acquisition from the Pluto.
                    self._engine_backend = SignalReader(
                        PyAdiIQSource(self.sdr, eng_cfg),
                        eng_cfg,
                    )
                    self._engine.attach_backend(self._engine_backend)
                    # Apply the GUI's currently-selected signal processor.
                    self._apply_selected_processor()
                    # Push current Detect button state onto the engine.
                    self._apply_detection_state()
                    # Honour the currently-selected sweep window
                    self._engine.set_active_range(self.sweep_start_hz, self.sweep_end_hz)
                    self.sweep_worker.configure_engine(self._engine, self._engine_backend)
                    if self.engine_status_label:
                        n_cells = len(self._engine.cells)
                        proc_label = self._engine.get_processor().label
                        self.engine_status_label.setText(
                            f"Active — {n_cells} cells over "
                            f"{eng_cfg.frequency_range.start_hz/1e9:.1f}–"
                            f"{eng_cfg.frequency_range.stop_hz/1e9:.1f} GHz "
                            f"[{proc_label}]"
                        )
                        self.engine_status_label.setStyleSheet("color:#0f0;padding:1px 4px;")
                except Exception as eng_ex:
                    self._engine = None
                    self._engine_backend = None
                    if self.engine_status_label:
                        self.engine_status_label.setText(f"Engine error: {str(eng_ex)[:60]}")
                        self.engine_status_label.setStyleSheet("color:#f44;padding:1px 4px;")

            self.sweep_worker.start()
            self.update_timer.start(80)

        except Exception as e:
            self.connected = False
            self.status_label.setText(f"Connection failed: {str(e)[:60]}")
            self.status_label.setStyleSheet("color:#f44;")

    def _disconnect(self):
        self.sweep_worker.stop()
        self.update_timer.stop()

        # Shut down engine telemetry
        if self._engine is not None:
            try:
                self._engine.telemetry.close()
            except Exception:
                pass
            self._engine = None
            self._engine_backend = None

        if self.sdr:
            try:
                del self.sdr
            except Exception:
                pass
            self.sdr = None
        self.connected = False
        self.connect_btn.clicked.disconnect()
        self.connect_btn.clicked.connect(self.try_connect)
        self._refresh_primary_button_label()
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color:#888;")
        self.warning_label.setText("")
        self.warning_label.setStyleSheet("color:transparent;")
        if self.engine_status_label:
            self.engine_status_label.setText("Idle")
            self.engine_status_label.setStyleSheet("color:#666;padding:1px 4px;")
        if self.engine_coverage_label:
            self.engine_coverage_label.setText("")
        if self.engine_tracks_label:
            self.engine_tracks_label.setText("")
        self.alert_panel.clear()

    # ------------------------------------------------------------ Simulator
    def _connect_simulator(self):
        """Spin up the engine against a simulated IQ source instead of pyadi-iio."""
        if not (HAS_SIM and HAS_ENGINE):
            self.status_label.setText("Simulator unavailable (needs spectrum_engine + sim)")
            self.status_label.setStyleSheet("color:#f44;")
            return

        try:
            eng_cfg = _load_engine_config()
            scene_label = self.scene_combo.currentText() if self.scene_combo else "Empty band (noise only)"
            scene = sim_scenarios.preset_by_label(scene_label)

            # No physical SDR — but other code paths inspect self.sdr defensively
            self.sdr = None
            self.dual_channel = False
            self.connected = True

            self.spec_dir.setTitle("RX2 (sim mirror)", color="#ff8800", size="9pt")
            self.status_label.setText(
                f"Connected (SIMULATOR: {scene_label}) - "
                f"{self.sweep_start_hz/1e9:.3f}-{self.sweep_end_hz/1e9:.3f} GHz "
                f"({self.num_steps} steps)"
            )
            self.status_label.setStyleSheet("color:#0ff;")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self._disconnect)
            self._refresh_primary_button_label()

            self._reset_buffers()
            # Worker still needs configure() so its display buffers / features
            # are wired up; sdr=None is fine because the simulator path uses
            # the engine loop, never the classic loop.
            self.sweep_worker.configure(
                None, self.sweep_start_hz, self.num_steps,
                False, self.spectrum_omni, self.spectrum_dir,
                spec_omni_welch=self.spectrum_omni_welch,
                spec_dir_welch=self.spectrum_dir_welch,
                features=self.features, fastlock=self.fastlock,
            )

            self._engine = SpectrumEngine(cfg=eng_cfg, telemetry_enabled=True)
            # Same SignalReader pipeline as the hardware path; only the
            # source differs.
            self._engine_backend = SignalReader(
                SimulatedIQSource(eng_cfg, scene, seed=42, add_dc_spike=True),
                eng_cfg,
                realtime=True,
            )
            self._engine.attach_backend(self._engine_backend)
            self._apply_selected_processor()
            self._apply_detection_state()
            self._engine.set_active_range(self.sweep_start_hz, self.sweep_end_hz)
            self.sweep_worker.configure_engine(self._engine, self._engine_backend)

            if self.engine_status_label:
                proc_label = self._engine.get_processor().label
                self.engine_status_label.setText(
                    f"Sim: {scene_label} — {len(self._engine.cells)} cells [{proc_label}]"
                )
                self.engine_status_label.setStyleSheet("color:#0ff;padding:1px 4px;")

            self.sweep_worker.start()
            self.update_timer.start(80)

        except Exception as e:
            self.connected = False
            self.status_label.setText(f"Sim connect failed: {str(e)[:60]}")
            self.status_label.setStyleSheet("color:#f44;")

    def _on_source_changed(self, _idx):
        """Enable scene combo + start/stop the sim preview based on source."""
        if self.scene_combo is None:
            return
        is_sim = self.source_combo.currentText() == "Simulator"
        self.scene_combo.setEnabled(is_sim)
        if is_sim:
            self._activate_sim_preview()
        else:
            self._deactivate_sim_preview()
        self._refresh_primary_button_label()

    def _on_scene_changed(self, _idx):
        """Live-swap the simulator's scene and refresh the preview panel."""
        if self.scene_combo is None:
            return
        # Always refresh the preview, even when not connected — the user
        # picks a scene to *see* it before hitting Simulate.
        if self._sim_preview_active:
            # Clear waterfall so the new scene's pattern is unmistakable.
            self._sim_preview_wf.clear()
            self._refresh_sim_preview()
        backend = self._engine_backend
        # The simulator path wraps a SimulatedIQSource inside a SignalReader.
        source = getattr(backend, "source", None)
        if not isinstance(source, SimulatedIQSource):
            return  # not in simulator mode or not connected
        label = self.scene_combo.currentText()
        try:
            new_scene = sim_scenarios.preset_by_label(label)
        except KeyError:
            return
        source.set_scene(new_scene)
        if self.engine_status_label:
            self.engine_status_label.setText(
                f"Sim: {label} — {len(self._engine.cells)} cells"
            )

    def _on_proc_changed(self, _idx):
        """Live-swap the engine's signal processor (Classic / OS-CFAR / ...)."""
        if not (HAS_PIPELINE and self.proc_combo is not None):
            return
        if self._engine is None:
            return  # not connected; selection is applied at connect time
        self._apply_selected_processor()

    def _apply_selected_processor(self):
        """Push the Proc combo's current selection onto the active engine."""
        if not (HAS_PIPELINE and self.proc_combo is not None and self._engine is not None):
            return
        name = self.proc_combo.currentData()  # short id stored via userData
        if not name:
            return
        try:
            self._engine.set_processor(get_processor(name))
        except KeyError:
            return  # unknown name → leave the previous processor in place

    # ------------------------------------------------------------ Detect / Connect button helpers
    def _refresh_primary_button_label(self):
        """Set Connect-button text + style based on Source + connected state."""
        if not hasattr(self, "connect_btn"):
            return
        src = self.source_combo.currentText() if hasattr(self, "source_combo") else "Hardware"
        start_style = (
            "QPushButton{background:#1e4d2b;color:#9fff9f;border:1px solid #2f7a44;"
            "padding:5px;font-weight:bold;}"
            "QPushButton:hover{background:#2a6b3b;}"
        )
        stop_style = (
            "QPushButton{background:#4d1e1e;color:#ff9f9f;border:1px solid #7a2f2f;"
            "padding:5px;font-weight:bold;}"
            "QPushButton:hover{background:#6b2a2a;}"
        )
        if self.connected:
            label = "Stop simulation" if src == "Simulator" else "Receiver Off"
            self.connect_btn.setStyleSheet(stop_style)
        else:
            label = "Simulate" if src == "Simulator" else "Receiver On"
            self.connect_btn.setStyleSheet(start_style)
        self.connect_btn.setText(label)

    def _refresh_detect_button_label(self):
        """Set Detect toggle text + style based on current state."""
        if not hasattr(self, "detect_btn"):
            return
        if self._detect_enabled:
            self.detect_btn.setText("Detect: ON")
            self.detect_btn.setStyleSheet(
                "QPushButton{background:#5a3b00;color:#ffcc66;border:1px solid #885a00;"
                "padding:5px;font-weight:bold;}"
                "QPushButton:hover{background:#7a4f00;}"
            )
        else:
            self.detect_btn.setText("Detect: OFF")
            self.detect_btn.setStyleSheet(
                "QPushButton{background:#2a2a4a;color:#888;border:1px solid #444;"
                "padding:5px;}"
                "QPushButton:hover{background:#3a3a5a;color:#aaa;}"
            )

    def _on_detect_toggled(self, checked):
        """Toggle detection. Going OFF resets all cells + drops tracks."""
        was_on = self._detect_enabled
        self._detect_enabled = bool(checked)
        if self._engine is not None:
            self._engine.set_detection_enabled(self._detect_enabled)
            if was_on and not self._detect_enabled:
                self._engine.reset_detection_state()
        self._refresh_detect_button_label()

    def _apply_detection_state(self):
        """Push current detect flag onto the engine (called at connect time)."""
        if self._engine is None:
            return
        self._engine.set_detection_enabled(self._detect_enabled)

    def _reset_buffers(self):
        self.spectrum_omni[:] = DB_MIN
        self.spectrum_dir[:] = DB_MIN
        self.spectrum_omni_welch[:] = DB_MIN
        self.spectrum_dir_welch[:] = DB_MIN
        self.waterfall_omni.clear()
        self.waterfall_dir.clear()
        self.baseline_omni[:] = DB_MIN
        self.baseline_dir[:] = DB_MIN
        self._baseline_init = False
        self.cfar_thresh_omni[:] = DB_MAX
        self.cfar_thresh_dir[:] = DB_MAX
        self.warning_counters = {f: 0 for f in self.monitor_freqs_ghz}
        for de in self.distance_estimators.values():
            de.reset()

    # ------------------------------------------------------------ Events
    def _on_gain(self, val):
        self.gain_label.setText(f"{val} dB")
        if self.sdr and self.connected:
            try:
                self.sdr.rx_hardwaregain_chan0 = val
                if self.dual_channel:
                    self.sdr.rx_hardwaregain_chan1 = val
            except Exception:
                pass

    def _on_threshold(self, val):
        self.threshold_label.setText(f"{val} dBFS")
        if hasattr(self, "thresh_line_omni"):
            self.thresh_line_omni.setValue(val)
            self.thresh_line_dir.setValue(val)

    def _on_wf_depth(self, val):
        self.wf_max_lines = val
        old_omni = list(self.waterfall_omni)
        old_dir = list(self.waterfall_dir)
        self.waterfall_omni = deque(old_omni[-val:], maxlen=val)
        self.waterfall_dir = deque(old_dir[-val:], maxlen=val)

    def _apply_sweep_range(self):
        start = self.start_spin.value()
        end = self.end_spin.value()
        if end - start < 0.02:
            return

        was_running = self.sweep_worker.isRunning()
        if was_running:
            self.sweep_worker.stop()
            self.update_timer.stop()

        self.sweep_start_hz = int(start * 1e9)
        self.sweep_end_hz = int(end * 1e9)
        self._recalc_sweep()
        self._update_plot_ranges()
        self._setup_plots()
        self._setup_crosshairs()
        self._reset_buffers()

        # Restrict the adaptive engine to the new window so it only sweeps the
        # selected band (cells outside become UNSUPPORTED and are skipped).
        if self._engine is not None:
            self._engine.set_active_range(self.sweep_start_hz, self.sweep_end_hz)
            self._scan_heat = None  # re-size/realign the coverage map
            self._update_activity_map_ranges()

        if was_running and self.connected:
            if self._engine is not None and self._engine_backend is not None:
                self.sweep_worker.configure_engine(self._engine, self._engine_backend)
            else:
                self.sweep_worker.configure(
                    self.sdr, self.sweep_start_hz, self.num_steps,
                    self.dual_channel, self.spectrum_omni, self.spectrum_dir,
                    spec_omni_welch=self.spectrum_omni_welch,
                    spec_dir_welch=self.spectrum_dir_welch,
                    features=self.features, fastlock=self.fastlock,
                )
            self.sweep_worker.start()
            self.update_timer.start(80)

        self.status_label.setText(
            f"Sweep range: {start:.3f}-{end:.3f} GHz ({self.num_steps} steps)"
        )

    def _update_plot_ranges(self):
        s, e = self.sweep_start_hz / 1e9, self.sweep_end_hz / 1e9
        for pw in (self.spec_omni, self.spec_dir):
            pw.setXRange(s, e, padding=0)
        for pw in (self.wf_omni, self.wf_dir):
            pw.setXRange(s, e, padding=0)

    def _apply_monitors(self):
        text = self.monitor_input.text().strip()
        if not text:
            return
        try:
            freqs = sorted(set(
                round(float(f.strip()), 3)
                for f in text.split(",") if f.strip()
            ))
            freqs = [f for f in freqs if 0.325 <= f <= 6.0]
        except ValueError:
            return
        if not freqs:
            return

        self.monitor_freqs_ghz = freqs
        self._init_monitor_history()
        self._rebuild_monitor_widgets()
        self.monitor_input.setText(", ".join(f"{f:.3f}" for f in freqs))

    def _rebuild_monitor_widgets(self):
        while self.monitor_layout.count():
            item = self.monitor_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.monitor_widgets = {}

        for freq in self.monitor_freqs_ghz:
            frame = QFrame()
            frame.setStyleSheet("background:#1a1a2e;border:1px solid #333;")
            row = QHBoxLayout(frame)
            row.setContentsMargins(4, 4, 4, 4)

            lbl = QLabel(f"  {freq:.3f} GHz")
            lbl.setStyleSheet("color:#0ff;font-weight:bold;border:none;")
            lbl.setFixedWidth(110)
            row.addWidget(lbl)

            pw_o = pg.PlotWidget()
            pw_o.setBackground(BG_COLOR)
            pw_o.setMinimumHeight(80)
            pw_o.showGrid(x=True, y=True, alpha=0.1)
            pw_o.setLabel("left", "dBFS")
            pw_o.setLabel("bottom", "Sweep #")
            pw_o.setTitle("RX1", color="c", size="8pt")
            for a in ("bottom", "left"):
                pw_o.getAxis(a).setPen(pg.mkPen((40, 40, 80)))
                pw_o.getAxis(a).setTextPen("#888")
            row.addWidget(pw_o, 1)

            pw_d = pg.PlotWidget()
            pw_d.setBackground(BG_COLOR)
            pw_d.setMinimumHeight(80)
            pw_d.showGrid(x=True, y=True, alpha=0.1)
            pw_d.setLabel("left", "dBFS")
            pw_d.setLabel("bottom", "Sweep #")
            pw_d.setTitle("RX2", color="m", size="8pt")
            for a in ("bottom", "left"):
                pw_d.getAxis(a).setPen(pg.mkPen((40, 40, 80)))
                pw_d.getAxis(a).setTextPen("#888")
            row.addWidget(pw_d, 1)

            info_col = QVBoxLayout()
            info_col.setSpacing(2)

            pwr_lbl = QLabel("-- dBFS")
            pwr_lbl.setStyleSheet("color:#ff0;border:none;font-weight:bold;")
            pwr_lbl.setFixedWidth(100)
            pwr_lbl.setAlignment(Qt.AlignCenter)
            info_col.addWidget(pwr_lbl)

            dist_lbl = QLabel("-- m")
            dist_lbl.setStyleSheet("color:#888;border:none;")
            dist_lbl.setFixedWidth(100)
            dist_lbl.setAlignment(Qt.AlignCenter)
            info_col.addWidget(dist_lbl)

            cfar_lbl = QLabel("")
            cfar_lbl.setStyleSheet("color:#666;border:none;font-size:9px;")
            cfar_lbl.setFixedWidth(100)
            cfar_lbl.setAlignment(Qt.AlignCenter)
            info_col.addWidget(cfar_lbl)

            row.addLayout(info_col)

            self.monitor_widgets[freq] = {
                "omni_plot": pw_o, "dir_plot": pw_d,
                "pwr_label": pwr_lbl, "dist_label": dist_lbl,
                "cfar_label": cfar_lbl,
                "omni_curve": None, "dir_curve": None,
            }
            self.monitor_layout.addWidget(frame)

    # ---------------------------------------- Adaptive baseline update
    def _update_baseline(self):
        if not self.features.get("use_adaptive_baseline"):
            return
        for spec, bl in [(self.spectrum_omni, self.baseline_omni),
                         (self.spectrum_dir, self.baseline_dir)]:
            if not self._baseline_init:
                bl[:] = spec
            else:
                quiet = spec < (bl + BASELINE_MARGIN_DB)
                bl[quiet] += BASELINE_ALPHA * (spec[quiet] - bl[quiet])
                loud = ~quiet
                bl[loud] += (BASELINE_ALPHA * 0.1) * (spec[loud] - bl[loud])
        self._baseline_init = True

    # ------------------------------------------- Sweep results (main thread)
    def _on_sweep_done(self, wf_omni_line, wf_dir_line, ref_nf):
        self.waterfall_omni.append(wf_omni_line)
        self.waterfall_dir.append(wf_dir_line)
        self.reference_noise_floor = ref_nf

        self._update_baseline()

        use_cfar = self.features.get("use_cfar_detection", False)
        use_dist = self.features.get("use_distance_model", False)
        use_welch = self.features.get("use_welch_psd", False)
        fixed_threshold = float(self.threshold_slider.value())
        scale = self.cfar_scale_db

        spec_o = self.spectrum_omni_welch if use_welch else self.spectrum_omni
        spec_d = self.spectrum_dir_welch if use_welch else self.spectrum_dir

        if use_cfar:
            self.cfar_thresh_omni = cfar_threshold(
                spec_o, CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, scale)
            self.cfar_thresh_dir = cfar_threshold(
                spec_d, CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, scale)

        for freq in self.monitor_freqs_ghz:
            if freq not in self.monitor_history:
                continue
            idx = self._freq_to_bin(freq)

            if use_cfar:
                cfar_det_o, peak_o, noise_o, thr_o = cfar_detect_at_bin(
                    spec_o, idx, CFAR_MONITOR_SEARCH_HALF,
                    CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, scale)
                cfar_det_d, peak_d, noise_d, thr_d = cfar_detect_at_bin(
                    spec_d, idx, CFAR_MONITOR_SEARCH_HALF,
                    CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, scale)
            else:
                lo_b = max(0, idx - CFAR_MONITOR_SEARCH_HALF)
                hi_b = min(self.total_bins, idx + CFAR_MONITOR_SEARCH_HALF + 1)
                peak_o = float(np.max(spec_o[lo_b:hi_b]))
                peak_d = float(np.max(spec_d[lo_b:hi_b]))
                cfar_det_o = peak_o > fixed_threshold
                cfar_det_d = peak_d > fixed_threshold
                noise_o = noise_d = ref_nf

            omni_val = _clamp(peak_o, DB_MIN, DB_MAX, DB_MIN)
            dir_val = _clamp(peak_d, DB_MIN, DB_MAX, DB_MIN)
            self.monitor_history[freq]["omni"].append(omni_val)
            self.monitor_history[freq]["dir"].append(dir_val)

            # Hybrid detection: CFAR OR fixed threshold (safety fallback)
            above_fixed = peak_o > fixed_threshold or peak_d > fixed_threshold
            cfar_detected = cfar_det_o or cfar_det_d
            detected = cfar_detected or above_fixed

            counter = self.warning_counters.get(freq, 0)
            if detected:
                self.warning_counters[freq] = min(counter + 2, WARNING_PERSIST_COUNT + 8)
            else:
                self.warning_counters[freq] = max(counter - 1, 0)

            if use_dist and freq in self.distance_estimators:
                best_power = max(peak_o, peak_d)
                self.distance_estimators[freq].update(best_power)

            de = self.distance_estimators.get(freq)
            if detected:
                self.alert_panel.push(
                    freq_ghz=freq,
                    signal_dbfs=max(peak_o, peak_d),
                    distance_m=de.x if (use_dist and de) else None,
                    confidence=de.confidence if (use_dist and de) else 0.0,
                )
            else:
                self.alert_panel.push(
                    freq_ghz=freq,
                    signal_dbfs=DB_MIN,
                    distance_m=None,
                    confidence=0.0,
                )

    def _on_engine_update(self, snapshot):
        """
        Handle a SpectrumEngine snapshot delivered from the worker thread.

        Updates:
          - Waterfall buffer from the engine's rolling display PSD
          - Engine status bar (scan count, coverage health, tracks)
          - Alert panel and proximity alerts from confirmed tracks
        """
        if snapshot is None:
            return

        self._engine_scan_count = snapshot.scan_count

        # --- Waterfall: build a downsampled line from the engine's full PSD ---
        psd = snapshot.last_psd_db
        freq = snapshot.last_psd_freq_hz
        if len(psd) > 0 and len(freq) > 0:
            nf = float(np.percentile(psd, 30))
            nf = _clamp(nf, -140, -20, -80)
            self.reference_noise_floor = nf

            x_src = np.linspace(0, len(psd) - 1, len(psd))
            x_ds = np.linspace(0, len(psd) - 1, WATERFALL_COLS)
            wf_line = np.interp(x_ds, x_src, psd).astype(np.float32)
            self.waterfall_omni.append(wf_line)
            self.waterfall_dir.append(wf_line)

            # Also update the spectrum display buffers used by _update_display
            # Map engine PSD onto the sweep buffer freq axis
            if self.total_bins > 0 and len(self.freq_axis_ghz) > 0:
                freq_ghz = freq / 1e9
                lo_g = freq_ghz[0]
                hi_g = freq_ghz[-1]
                mask = (self.freq_axis_ghz >= lo_g) & (self.freq_axis_ghz <= hi_g)
                n_dest = int(np.sum(mask))
                if n_dest > 0:
                    dest_f = self.freq_axis_ghz[mask] * 1e9
                    interp = np.interp(dest_f, freq, psd).astype(np.float32)
                    self.spectrum_omni[mask] = interp
                    self.spectrum_dir[mask] = interp

        # --- Scan-target indicator line ---
        if snapshot.last_command is not None and hasattr(self, '_scan_target_line_omni'):
            scan_ghz = snapshot.last_command.center_hz / 1e9
            self._scan_target_line_omni.setPos(scan_ghz)
            self._scan_target_line_dir.setPos(scan_ghz)
            self._scan_target_line_omni.setVisible(True)
            self._scan_target_line_dir.setVisible(True)

        # --- Spectrum Activity Maps ---
        self._update_activity_maps(snapshot)

        # --- Engine status bar ---
        if self.engine_status_label is not None:
            age_str = f"max_age={snapshot.max_cell_age_s:.1f}s"
            overdue_str = f"overdue={snapshot.overdue_cell_count}"
            self.engine_coverage_label.setText(f"scans={snapshot.scan_count} | {age_str} | {overdue_str}")

        # --- Track-based alerts ---
        active_tracks = [t for t in snapshot.tracks if t.state not in ("EXPIRED", "LOST")]
        confirmed = [t for t in active_tracks if t.state in ("CONFIRMED", "TRACKING")]

        if self.engine_tracks_label is not None:
            if confirmed:
                track_summary = " | ".join(
                    f"{t.center_hz/1e9:.3f}GHz({t.state[0]})" for t in confirmed[:4]
                )
                self.engine_tracks_label.setText(f"Tracks: {track_summary}")
                self.engine_tracks_label.setStyleSheet("color:#0ff;padding:1px 4px;font-weight:bold;")
            elif active_tracks:
                self.engine_tracks_label.setText(f"Candidates: {len(active_tracks)}")
                self.engine_tracks_label.setStyleSheet("color:#fa0;padding:1px 4px;")
            else:
                self.engine_tracks_label.setText("")
                self.engine_tracks_label.setStyleSheet("color:#0ff;padding:1px 4px;")

        # --- Push confirmed tracks to the proximity alert panel ---
        use_dist = self.features.get("use_distance_model", False)
        track_freqs_pushed = set()
        for track in confirmed:
            freq_ghz = track.center_hz / 1e9
            de = None
            if use_dist:
                # Reuse or create a distance estimator keyed to this track
                key = round(freq_ghz, 3)
                if key not in self.distance_estimators:
                    self.distance_estimators[key] = DistanceEstimator()
                de = self.distance_estimators[key]
                de.update(track.peak_db)
            self.alert_panel.push(
                freq_ghz=freq_ghz,
                signal_dbfs=track.peak_db,
                distance_m=de.x if (use_dist and de) else None,
                confidence=track.confidence,
            )
            track_freqs_pushed.add(round(freq_ghz, 3))

        # Push "no signal" for any monitor freq not covered by a track
        for freq in self.monitor_freqs_ghz:
            if round(freq, 3) not in track_freqs_pushed:
                self.alert_panel.push(
                    freq_ghz=freq,
                    signal_dbfs=DB_MIN,
                    distance_m=None,
                    confidence=0.0,
                )

        # Update warning banner based on confirmed tracks
        if confirmed:
            freqs_str = ", ".join(f"{t.center_hz/1e9:.3f}" for t in confirmed[:3])
            self.warning_label.setText(f"\u26a0 SIGNAL @ {freqs_str} GHz")
            self.warning_label.setStyleSheet(
                "color:white;background:#c00;padding:4px;border-radius:3px;font-weight:bold;"
            )
        else:
            self.warning_label.setText("")
            self.warning_label.setStyleSheet("color:transparent;")

    def _update_activity_maps(self, snapshot):
        """
        Render the activity-probability and scan-coverage strips from a snapshot.

        The probability strip maps each cell's occupancy probability to color.
        The scan strip uses a decaying heat buffer: a cell flares to 1.0 when
        scanned and fades by SCAN_HEAT_DECAY each update, so even rarely visited
        bands visibly blink when they finally get swept.
        """
        if not hasattr(self, "_prob_img") or self._prob_img is None:
            return

        occ = snapshot.occupancy_probs
        n_cells = len(occ)
        if n_cells == 0:
            return

        # Lazily size the heat buffer once we know the cell count
        if self._scan_heat is None or len(self._scan_heat) != n_cells:
            self._scan_heat = np.zeros(n_cells, dtype=np.float32)
            # Align the image rect to the actual cell frequency span
            centers = snapshot.cell_freq_centers_hz
            if len(centers) >= 2:
                width = float(centers[1] - centers[0])
                start_ghz = (float(centers[0]) - width / 2.0) / 1e9
                stop_ghz = (float(centers[-1]) + width / 2.0) / 1e9
                span_ghz = stop_ghz - start_ghz
                self._prob_img.setRect(QRectF(start_ghz, 0, span_ghz, 1))
                self._scan_img.setRect(QRectF(start_ghz, 0, span_ghz, 1))
                # Zoom to the active sweep window (may be narrower than the grid)
                self._update_activity_map_ranges()

        # --- Probability strip (gamma-lifted for visibility) ---
        prob = np.clip(occ.astype(np.float32), 0.0, 1.0)
        prob_disp = np.power(prob, PROB_DISPLAY_GAMMA)
        self._prob_img.setImage(prob_disp.reshape(n_cells, 1),
                                autoLevels=False, levels=(0, 1))

        # --- Scan coverage strip: heat derived from per-cell scan age ---
        # heat = exp(-age / tau): a cell scanned just now is ~1.0 and fades to
        # 1/e after SCAN_HEAT_TAU_S seconds. This is robust to command type
        # (coarse, fine, or track revisit) and update rate, since it reads the
        # engine's actual last_scan_time per cell.
        age = snapshot.cell_scan_age_s
        if age is not None and len(age) == n_cells:
            heat = np.exp(-age.astype(np.float32) / SCAN_HEAT_TAU_S)
            self._scan_heat = heat
        else:
            # Fallback: keep previous behaviour if the field is unavailable
            heat = self._scan_heat

        heat_disp = np.power(np.clip(heat, 0.0, 1.0), SCAN_DISPLAY_GAMMA)
        self._scan_img.setImage(heat_disp.reshape(n_cells, 1),
                                autoLevels=False, levels=(0, 1))

        self._log_map_colors(occ, prob_disp, heat, heat_disp)

    def _log_map_colors(self, prob, prob_disp, heat, heat_disp):
        """Periodically log render-level values (raw value, gamma-lifted value,
        and resulting RGB) for the brightest cells, so a 'uniform map' complaint
        can be debugged at the color level alongside the engine's data log."""
        import time as _t
        now = _t.time()
        if now - getattr(self, "_last_color_log", 0.0) < 3.0:
            return
        self._last_color_log = now
        try:
            import os
            prob_lut = _probability_colormap()
            scan_lut = _scan_colormap()

            def rgb(lut, v):
                idx = int(np.clip(v, 0, 1) * (len(lut) - 1))
                r, g, b = lut[idx][:3]
                return f"({r:3d},{g:3d},{b:3d})"

            order = np.argsort(prob)[::-1][:6]
            os.makedirs("spectrum_logs", exist_ok=True)
            with open(os.path.join("spectrum_logs", "map_colors.log"), "a") as fh:
                fh.write(
                    f"\n[colors] PROB raw(min/med/max)={prob.min():.3f}/"
                    f"{np.median(prob):.3f}/{prob.max():.3f}  "
                    f"HEAT raw(min/med/max)={heat.min():.3f}/"
                    f"{np.median(heat):.3f}/{heat.max():.3f}\n"
                )
                for i in order:
                    fh.write(
                        f"  cell#{int(i):3d}: prob={prob[i]:.3f} disp={prob_disp[i]:.3f} "
                        f"rgb{rgb(prob_lut, prob_disp[i])} | "
                        f"heat={heat[i]:.3f} disp={heat_disp[i]:.3f} "
                        f"rgb{rgb(scan_lut, heat_disp[i])}\n"
                    )
        except Exception:
            pass

    def _on_sweep_error(self, msg):
        self.status_label.setText(f"Sweep error: {msg}")
        self.status_label.setStyleSheet("color:#f44;")

    # -------------------------------------------------- Display update
    def _update_display(self):
        cur = self.sweep_worker.current_freq
        if cur > 0:
            # When engine is running, also show scan count
            if self._engine is not None:
                self.freq_label.setText(
                    "%.3f GHz  [#%d]" % (cur / 1e9, self._engine_scan_count)
                )
            else:
                self.freq_label.setText("%.3f GHz" % (cur / 1e9))

        alerts = []
        proximity_alerts = []
        use_dist = self.features.get("use_distance_model", False)
        for freq in self.monitor_freqs_ghz:
            if self.warning_counters.get(freq, 0) >= WARNING_PERSIST_COUNT:
                alerts.append(f"{freq:.3f}")
            if use_dist:
                de = self.distance_estimators.get(freq)
                if de and de.inside_boundary and de.confidence > 0.3:
                    proximity_alerts.append(f"{freq:.3f}")

        if proximity_alerts:
            self.warning_label.setText(
                f"\u26a0 DRONE <{int(DIST_ENTER_M)}m @ "
                f"{', '.join(proximity_alerts)} GHz"
            )
            self.warning_label.setStyleSheet(
                "color:white;background:#a00;padding:4px;border-radius:3px;font-weight:bold;"
            )
        elif alerts:
            self.warning_label.setText(
                f"\u26a0 SIGNAL @ {', '.join(alerts)} GHz"
            )
            self.warning_label.setStyleSheet(
                "color:white;background:#c00;padding:4px;border-radius:3px;font-weight:bold;"
            )
        else:
            self.warning_label.setText("")
            self.warning_label.setStyleSheet("color:transparent;")

        self.spec_omni_curve.setData(self.freq_axis_ghz, self.spectrum_omni)
        # spec_dir is the Simulated Signal panel — driven by
        # _refresh_sim_preview, never overwritten here.

        if self.features.get("use_cfar_detection"):
            self.cfar_omni_curve.setData(self.freq_axis_ghz, self.cfar_thresh_omni)
            self.cfar_omni_curve.setVisible(True)
        else:
            self.cfar_omni_curve.setVisible(False)
        # cfar_dir_curve stays hidden — it was an RX2-antenna overlay.

        if self.waterfall_omni:
            nf = _clamp(self.reference_noise_floor, -140, -20, -80)
            wf_o = np.array(list(self.waterfall_omni), dtype=np.float32)
            wf_o_n = np.clip((wf_o - nf) / DYNAMIC_RANGE, 0, 1)

            self.wf_omni_img.setImage(wf_o_n.T, autoLevels=False, levels=(0, 1))

            h = len(wf_o)
            span = (self.sweep_end_hz - self.sweep_start_hz) / 1e9
            self.wf_omni_img.setRect(QRectF(self.sweep_start_hz / 1e9, 0, span, h))
            self.wf_omni.setYRange(0, h)

        if SHOW_MONITOR_PANEL:
            self._update_monitor_plots()

    def _update_monitor_plots(self):
        use_dist = self.features.get("use_distance_model", False)
        for freq, widgets in self.monitor_widgets.items():
            hist = self.monitor_history.get(freq)
            if not hist:
                continue

            for key, color, plot_key, curve_key in [
                ("omni", "c", "omni_plot", "omni_curve"),
                ("dir", "m", "dir_plot", "dir_curve"),
            ]:
                data = hist[key]
                pw = widgets[plot_key]
                if len(data) == 0:
                    continue

                vals = np.array(data, dtype=np.float32)

                if widgets[curve_key] is None:
                    widgets[curve_key] = pw.plot(
                        vals, pen=pg.mkPen(color, width=1)
                    )
                else:
                    widgets[curve_key].setData(vals)

                pw.setYRange(-100, 0)

            if hist["omni"]:
                last_db = hist["omni"][-1]
                widgets["pwr_label"].setText(f"{last_db:.0f} dBFS")
                is_alert = self.warning_counters.get(freq, 0) >= WARNING_PERSIST_COUNT
                if is_alert:
                    widgets["pwr_label"].setStyleSheet(
                        "color:#f00;border:none;font-weight:bold;"
                    )
                else:
                    widgets["pwr_label"].setStyleSheet(
                        "color:#ff0;border:none;font-weight:bold;"
                    )

            de = self.distance_estimators.get(freq)
            if use_dist and de:
                dist_m = de.x
                conf = de.confidence
                if dist_m < 10000 and conf > 0.1:
                    dist_str = f"~{dist_m:.0f}m ({conf*100:.0f}%)"
                    if de.inside_boundary:
                        widgets["dist_label"].setStyleSheet(
                            "color:#f44;border:none;font-weight:bold;")
                    elif dist_m < 200:
                        widgets["dist_label"].setStyleSheet(
                            "color:#fa0;border:none;")
                    else:
                        widgets["dist_label"].setStyleSheet(
                            "color:#888;border:none;")
                else:
                    dist_str = "-- m"
                    widgets["dist_label"].setStyleSheet("color:#666;border:none;")
                widgets["dist_label"].setText(dist_str)
            else:
                widgets["dist_label"].setText("")

            if self.features.get("use_cfar_detection"):
                idx = self._freq_to_bin(freq)
                thr_val = self.cfar_thresh_omni[idx] if idx < len(self.cfar_thresh_omni) else 0
                widgets["cfar_label"].setText(f"CFAR thr: {thr_val:.0f}")
                widgets["cfar_label"].setStyleSheet("color:#ff0;border:none;font-size:9px;")
            else:
                widgets["cfar_label"].setText("")


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DroneDetector()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
