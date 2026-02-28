"""
FPV Drone Detector for PlutoSDR
SDR++ style spectrum + waterfall.
Monitors predefined frequencies with strength-over-time.
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
    QLabel, QPushButton, QSlider, QGroupBox,
    QScrollArea, QFrame, QSplitter, QLineEdit, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import QTimer, Qt, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ---- Default Configuration ----
DEFAULT_START_GHZ = 5.6
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
DEFAULT_MONITORS = [5.800, 5.900, 5.920]
DEFAULT_THRESHOLD_DBFS = -40
WARNING_PERSIST_COUNT = 2

# ---- dBFS calibration ----
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


# ------------------------------------------------------------ Sweep Thread
class SweepWorker(QThread):
    """Runs the SDR sweep loop in a background thread so the GUI stays responsive."""
    sweep_done = pyqtSignal(object, object, float)
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
        self.current_freq = 0
        self.sweep_col = 0

    def configure(self, sdr, start_hz, num_steps, dual, spec_omni, spec_dir):
        self.sdr = sdr
        self.sweep_start_hz = start_hz
        self.num_steps = num_steps
        self.dual_channel = dual
        self.spectrum_omni = spec_omni
        self.spectrum_dir = spec_dir

    def run(self):
        self._running = True
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
        self.sdr.rx_destroy_buffer()
        self.sdr.rx_lo = int(self.current_freq)
        time.sleep(PLL_SETTLE_S)
        raw = self.sdr.rx()

        if self.dual_channel:
            if isinstance(raw, (list, tuple)):
                omni_iq = np.asarray(raw[0], dtype=np.complex64).ravel()
                dir_iq = np.asarray(raw[1], dtype=np.complex64).ravel()
            elif isinstance(raw, np.ndarray) and raw.ndim == 2:
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

        self.sweep_start_hz = int(DEFAULT_START_GHZ * 1e9)
        self.sweep_end_hz = int(DEFAULT_END_GHZ * 1e9)
        self.reference_noise_floor = -80.0
        self.wf_max_lines = DEFAULT_WF_LINES

        self.monitor_freqs_ghz = list(DEFAULT_MONITORS)
        self.monitor_history = {}
        self.monitor_widgets = {}
        self.warning_counters = {}

        self._recalc_sweep()
        self._init_monitor_history()

        self._setup_ui()
        self._setup_plots()
        self._setup_crosshairs()

        self.sweep_worker = SweepWorker()
        self.sweep_worker.sweep_done.connect(self._on_sweep_done)
        self.sweep_worker.error_occurred.connect(self._on_sweep_error)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_display)

    # -------------------------------------------------------- Sweep math
    def _recalc_sweep(self):
        span = self.sweep_end_hz - self.sweep_start_hz
        self.num_steps = max(1, int(span / SWEEP_STEP_HZ))
        self.total_bins = self.num_steps * BINS_PER_STEP

        self.spectrum_omni = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.spectrum_dir = np.full(self.total_bins, DB_MIN, dtype=np.float32)
        self.waterfall_omni = deque(maxlen=self.wf_max_lines)
        self.waterfall_dir = deque(maxlen=self.wf_max_lines)

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

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedWidth(100)
        self.connect_btn.setStyleSheet(
            "QPushButton{background:#2a2a4a;color:#0f0;border:1px solid #444;padding:4px;}"
            "QPushButton:hover{background:#3a3a5a;}"
        )
        self.connect_btn.clicked.connect(self.try_connect)
        top.addWidget(self.connect_btn)
        root.addLayout(top)

        # ---- Controls bar ----
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

        # Monitor panel
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
        self.monitor_input.setPlaceholderText("5.800, 5.900, 5.920")
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

        main_vsplit.setSizes([450, 350])
        root.addWidget(main_vsplit)

        self._rebuild_monitor_widgets()

    def _lbl(self, text):
        l = QLabel(text)
        l.setStyleSheet("color:#aaa;")
        return l

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

    # -------------------------------------------------------- Connection
    def try_connect(self):
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

            # Try ad9361 via IP (full MIMO), fall back to Pluto via USB
            try:
                self.sdr = adi.ad9361(uri="ip:192.168.2.1")
            except Exception:
                try:
                    self.sdr = adi.ad9361(uri="ip:pluto.local")
                except Exception:
                    self.sdr = adi.Pluto(uri="usb:")

            self.sdr.rx_rf_bandwidth = BANDWIDTH_HZ
            self.sdr.sample_rate = SAMPLE_RATE
            self.sdr.rx_buffer_size = FFT_SIZE * 4
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = 40

            # Detect how many RX channels the IIO context exposes
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
                    self.sdr.rx_hardwaregain_chan1 = 40

                    # Verify with multiple reads
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
                f"Connected ({ch_info}) - {self.sweep_start_hz/1e9:.3f}-"
                f"{self.sweep_end_hz/1e9:.3f} GHz ({self.num_steps} steps)"
            )
            self.status_label.setStyleSheet("color:#0f0;")
            self.connect_btn.setText("Disconnect")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self._disconnect)

            self._reset_buffers()
            self.sweep_worker.configure(
                self.sdr, self.sweep_start_hz, self.num_steps,
                self.dual_channel, self.spectrum_omni, self.spectrum_dir,
            )
            self.sweep_worker.start()
            self.update_timer.start(80)

        except Exception as e:
            self.connected = False
            self.status_label.setText(f"Connection failed: {str(e)[:60]}")
            self.status_label.setStyleSheet("color:#f44;")

    def _disconnect(self):
        self.sweep_worker.stop()
        self.update_timer.stop()
        if self.sdr:
            try:
                del self.sdr
            except Exception:
                pass
            self.sdr = None
        self.connected = False
        self.connect_btn.setText("Connect")
        self.connect_btn.clicked.disconnect()
        self.connect_btn.clicked.connect(self.try_connect)
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color:#888;")
        self.warning_label.setText("")
        self.warning_label.setStyleSheet("color:transparent;")

    def _reset_buffers(self):
        self.spectrum_omni[:] = DB_MIN
        self.spectrum_dir[:] = DB_MIN
        self.waterfall_omni.clear()
        self.waterfall_dir.clear()
        self.warning_counters = {f: 0 for f in self.monitor_freqs_ghz}

    # ------------------------------------------------------------ Events
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

        if was_running and self.connected:
            self.sweep_worker.configure(
                self.sdr, self.sweep_start_hz, self.num_steps,
                self.dual_channel, self.spectrum_omni, self.spectrum_dir,
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

            pwr_lbl = QLabel("-- dBFS")
            pwr_lbl.setStyleSheet("color:#ff0;border:none;font-weight:bold;")
            pwr_lbl.setFixedWidth(80)
            pwr_lbl.setAlignment(Qt.AlignCenter)
            row.addWidget(pwr_lbl)

            self.monitor_widgets[freq] = {
                "omni_plot": pw_o, "dir_plot": pw_d,
                "pwr_label": pwr_lbl,
                "omni_curve": None, "dir_curve": None,
            }
            self.monitor_layout.addWidget(frame)

    # ------------------------------------------- Sweep results (main thread)
    def _on_sweep_done(self, wf_omni_line, wf_dir_line, ref_nf):
        self.waterfall_omni.append(wf_omni_line)
        self.waterfall_dir.append(wf_dir_line)
        self.reference_noise_floor = ref_nf

        threshold = float(self.threshold_slider.value())
        for freq in self.monitor_freqs_ghz:
            if freq not in self.monitor_history:
                continue
            idx = self._freq_to_bin(freq)
            lo_b = max(0, idx - 25)
            hi_b = min(self.total_bins, idx + 26)
            omni_val = float(np.max(self.spectrum_omni[lo_b:hi_b]))
            dir_val = float(np.max(self.spectrum_dir[lo_b:hi_b]))
            self.monitor_history[freq]["omni"].append(_clamp(omni_val, DB_MIN, DB_MAX, DB_MIN))
            self.monitor_history[freq]["dir"].append(_clamp(dir_val, DB_MIN, DB_MAX, DB_MIN))

            counter = self.warning_counters.get(freq, 0)
            above = omni_val > threshold or dir_val > threshold
            if above:
                self.warning_counters[freq] = min(counter + 2, WARNING_PERSIST_COUNT + 8)
            else:
                self.warning_counters[freq] = max(counter - 1, 0)

    def _on_sweep_error(self, msg):
        self.status_label.setText(f"Sweep error: {msg}")
        self.status_label.setStyleSheet("color:#f44;")


    # -------------------------------------------------- Display update
    def _update_display(self):
        cur = self.sweep_worker.current_freq
        if cur > 0:
            self.freq_label.setText("%.3f GHz" % (cur / 1e9))

        alerts = []
        for freq in self.monitor_freqs_ghz:
            if self.warning_counters.get(freq, 0) >= WARNING_PERSIST_COUNT:
                alerts.append(f"{freq:.3f}")

        if alerts:
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
        self.spec_dir_curve.setData(self.freq_axis_ghz, self.spectrum_dir)

        if self.waterfall_omni:
            nf = _clamp(self.reference_noise_floor, -140, -20, -80)
            wf_o = np.array(list(self.waterfall_omni), dtype=np.float32)
            wf_d = np.array(list(self.waterfall_dir), dtype=np.float32)
            wf_o_n = np.clip((wf_o - nf) / DYNAMIC_RANGE, 0, 1)
            wf_d_n = np.clip((wf_d - nf) / DYNAMIC_RANGE, 0, 1)

            self.wf_omni_img.setImage(wf_o_n.T, autoLevels=False, levels=(0, 1))
            self.wf_dir_img.setImage(wf_d_n.T, autoLevels=False, levels=(0, 1))

            h = len(wf_o)
            span = (self.sweep_end_hz - self.sweep_start_hz) / 1e9
            for img, wf_pw in [(self.wf_omni_img, self.wf_omni),
                                (self.wf_dir_img, self.wf_dir)]:
                img.setRect(QRectF(self.sweep_start_hz / 1e9, 0, span, h))
                wf_pw.setYRange(0, h)

        self._update_monitor_plots()

    def _update_monitor_plots(self):
        threshold = float(self.threshold_slider.value())
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


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DroneDetector()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
