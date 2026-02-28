"""
FPV Drone Detector for PlutoSDR
Displays signals from omni and directional antennas, sweeps 3-6 GHz,
and alerts when strong signals detected in 3.5 GHz or 5.8 GHz bands.
"""

import sys
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QFrame
)
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg

# PlutoSDR import is deferred to try_connect() - libiio may not be available until Pluto is connected
PLUTO_AVAILABLE = True  # Assume available; actual check happens on connect


# Configuration
# Stock PlutoSDR: 325MHz-3.8GHz. Set SWEEP_END to 3.8e9 for stock units.
# Pluto+ (AD9363) or modified: 70MHz-6GHz.
SWEEP_START_HZ = 3_000_000_000   # 3 GHz
SWEEP_END_HZ = 6_000_000_000     # 6 GHz (use 3_800_000_000 for stock Pluto)
SWEEP_STEP_HZ = 40_000_000       # 40 MHz steps (faster sweep, fewer steps)
BANDWIDTH_HZ = 20_000_000        # 20 MHz per capture
SAMPLE_RATE = 20_000_000         # 20 MSPS (Pluto USB limit ~6-8 MSPS sustained)
FFT_SIZE = 1024                  # Smaller FFT = faster capture
WATERFALL_LINES = 128
WATERFALL_COLS = 75              # One per 40 MHz from 3-6 GHz
SWEEP_TIMER_MS = 25              # ms between steps (must allow rx() to complete)
WARNING_BANDS = [
    (3_400_000_000, 3_600_000_000),   # 3.5 GHz band
    (5_700_000_000, 5_900_000_000),   # 5.8 GHz band (FPV common)
]
WARNING_THRESHOLD_DB = 15         # dB above noise floor

# Note: Stock PlutoSDR is 325MHz-3.8GHz. Pluto+ (AD9363) or modified units support 3-6 GHz.


class DroneDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPV Drone Detector - PlutoSDR")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # PlutoSDR state
        self.sdr = None
        self.connected = False
        self.dual_channel = False
        self.current_freq = SWEEP_START_HZ
        self.sweep_direction = 1
        self.waterfall_omni = deque(maxlen=WATERFALL_LINES)
        self.waterfall_dir = deque(maxlen=WATERFALL_LINES)
        self.sweep_buffer_omni = []   # Accumulates one full 3-6 GHz sweep
        self.sweep_buffer_dir = []
        self.warning_active = False
        self.noise_floor = -80
        self.reference_noise_floor = -80   # From non-warning-band sweeps (stable baseline)
        self.last_peak_db = -100
        self.last_spectrum_freq = SWEEP_START_HZ
        self.last_spectrum_omni = None
        self.last_spectrum_dir = None

        # Initialize UI
        self.setup_ui()
        self.setup_plots()

        # Connection timer
        self.connect_timer = QTimer(self)
        self.connect_timer.timeout.connect(self.try_connect)
        self.sweep_timer = QTimer(self)
        self.sweep_timer.timeout.connect(self.sweep_step)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Disconnected - Connect PlutoSDR via USB")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #888; padding: 5px;")
        status_layout.addWidget(self.status_label)

        self.warning_label = QLabel("")
        self.warning_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.warning_label.setStyleSheet(
            "color: #333; background: #333; padding: 10px; border-radius: 5px;"
        )
        self.warning_label.setMinimumWidth(400)
        self.warning_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.warning_label, 1)

        self.connect_btn = QPushButton("Connect PlutoSDR")
        self.connect_btn.clicked.connect(self.try_connect)
        status_layout.addWidget(self.connect_btn)
        layout.addLayout(status_layout)

        # Controls
        ctrl_layout = QHBoxLayout()
        ctrl_group = QGroupBox("Controls")
        ctrl_grid = QGridLayout()

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(5, 40)
        self.threshold_slider.setValue(WARNING_THRESHOLD_DB)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        self.threshold_label = QLabel(f"Threshold: {WARNING_THRESHOLD_DB} dB")
        ctrl_grid.addWidget(QLabel("Warning threshold:"), 0, 0)
        ctrl_grid.addWidget(self.threshold_slider, 0, 1)
        ctrl_grid.addWidget(self.threshold_label, 0, 2)

        self.freq_label = QLabel("Current: -- GHz")
        self.freq_label.setFont(QFont("Consolas", 10))
        ctrl_grid.addWidget(QLabel("Sweep frequency:"), 1, 0)
        ctrl_grid.addWidget(self.freq_label, 1, 1, 1, 2)


        ctrl_group.setLayout(ctrl_grid)
        ctrl_layout.addWidget(ctrl_group)
        layout.addLayout(ctrl_layout)

        # Plots area
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)

        # Spectrum plots
        spec_group = QGroupBox("Live Spectrum (Omni vs Directional)")
        spec_layout = QHBoxLayout()
        self.spectrum_omni = pg.PlotWidget(title="Omni Antenna")
        self.spectrum_omni.setLabel("left", "Power", units="dB")
        self.spectrum_omni.setLabel("bottom", "Frequency", units="MHz")
        self.spectrum_omni.showGrid(x=True, y=True)
        self.spectrum_omni.setYRange(-100, -20)
        spec_layout.addWidget(self.spectrum_omni, 1)

        self.spectrum_dir = pg.PlotWidget(title="Directional Antenna")
        self.spectrum_dir.setLabel("left", "Power", units="dB")
        self.spectrum_dir.setLabel("bottom", "Frequency", units="MHz")
        self.spectrum_dir.showGrid(x=True, y=True)
        self.spectrum_dir.setYRange(-100, -20)
        spec_layout.addWidget(self.spectrum_dir, 1)
        spec_group.setLayout(spec_layout)
        plots_layout.addWidget(spec_group)

        # Waterfall plots
        wf_group = QGroupBox("Signal Strength Waterfall (Stronger = Approaching)")
        wf_layout = QHBoxLayout()
        self.waterfall_omni_plot = pg.PlotWidget(title="Omni - Waterfall")
        self.waterfall_omni_plot.setLabel("bottom", "Frequency", units="MHz")
        self.waterfall_omni_plot.setLabel("left", "Time (newest top)")
        wf_layout.addWidget(self.waterfall_omni_plot, 1)

        self.waterfall_dir_plot = pg.PlotWidget(title="Directional - Waterfall")
        self.waterfall_dir_plot.setLabel("bottom", "Frequency", units="MHz")
        self.waterfall_dir_plot.setLabel("left", "Time (newest top)")
        wf_layout.addWidget(self.waterfall_dir_plot, 1)
        wf_group.setLayout(wf_layout)
        plots_layout.addWidget(wf_group)

        layout.addWidget(plots_widget)

        # Waterfall image items (created in setup_plots)
        self.wf_omni_img = None
        self.wf_dir_img = None

    def setup_plots(self):
        # Initialize waterfall images: rows=time, cols=3-6 GHz
        placeholder = np.zeros((WATERFALL_LINES, WATERFALL_COLS))
        self.wf_omni_img = pg.ImageItem(placeholder, autoLevels=False)
        self.wf_omni_img.setLookupTable(self._waterfall_colormap())
        self.wf_omni_img.setRect(QRectF(3, 0, 3, WATERFALL_LINES))  # 3-6 GHz
        self.waterfall_omni_plot.addItem(self.wf_omni_img)

        self.wf_dir_img = pg.ImageItem(placeholder, autoLevels=False)
        self.wf_dir_img.setLookupTable(self._waterfall_colormap())
        self.wf_dir_img.setRect(QRectF(3, 0, 3, WATERFALL_LINES))
        self.waterfall_dir_plot.addItem(self.wf_dir_img)

    def _waterfall_colormap(self):
        """Blue (weak) -> Green -> Yellow -> Red (strong)"""
        colors = [
            (0, 0, 0),
            (0, 0, 128),
            (0, 0, 255),
            (0, 128, 255),
            (0, 255, 255),
            (0, 255, 128),
            (0, 255, 0),
            (128, 255, 0),
            (255, 255, 0),
            (255, 128, 0),
            (255, 0, 0),
        ]
        positions = np.linspace(0, 1, len(colors))
        return pg.ColorMap(positions, colors).getLookupTable(0, 1, 256)

    def try_connect(self):
        try:
            import adi
        except (ImportError, TypeError, OSError) as e:
            self.status_label.setText("Error: Install pyadi-iio + libiio. Connect PlutoSDR via USB first.")
            self.status_label.setStyleSheet("color: #c00;")
            return

        try:
            if self.sdr:
                del self.sdr
                self.sdr = None

            self.sdr = adi.Pluto(uri="usb:")
            self.sdr.rx_rf_bandwidth = BANDWIDTH_HZ
            self.sdr.sample_rate = SAMPLE_RATE
            self.sdr.rx_buffer_size = FFT_SIZE * 4
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = 40

            # Try dual channel (requires Pluto 2r2t firmware)
            try:
                self.sdr.rx_enabled_channels = [0, 1]
                self.dual_channel = True
            except Exception:
                self.dual_channel = False

            self.connected = True
            self.status_label.setText("Connected - Sweeping 3-6 GHz")
            self.status_label.setStyleSheet("color: #0a0;")
            self.connect_btn.setText("Disconnect")
            self.connect_btn.clicked.disconnect()
            self.connect_btn.clicked.connect(self.disconnect)

            self.current_freq = SWEEP_START_HZ
            self.sweep_timer.start(SWEEP_TIMER_MS)
            self.update_timer.start(100)

        except Exception as e:
            self.connected = False
            self.status_label.setText(f"Connection failed: {e}")
            self.status_label.setStyleSheet("color: #c00;")

    def disconnect(self):
        self.sweep_timer.stop()
        self.update_timer.stop()
        if self.sdr:
            try:
                del self.sdr
            except Exception:
                pass
            self.sdr = None
        self.connected = False
        self.connect_btn.setText("Connect PlutoSDR")
        self.connect_btn.clicked.disconnect()
        self.connect_btn.clicked.connect(self.try_connect)
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: #888;")
        self.warning_label.setText("")
        self.warning_label.setStyleSheet("color: #333; background: #333;")

    def on_threshold_change(self, val):
        self.threshold_label.setText(f"Threshold: {val} dB")

    def sweep_step(self):
        if not self.connected or not self.sdr:
            return

        try:
            self.sdr.rx_lo = int(self.current_freq)
            data = self.sdr.rx()

            if self.dual_channel and data.ndim == 2:
                omni_data = data[:, 0]
                dir_data = data[:, 1]
            else:
                omni_data = np.asarray(data).flatten()
                dir_data = omni_data.copy()  # Same data if single channel

            # Compute FFT and power spectrum
            omni_fft = np.fft.fftshift(np.fft.fft(omni_data, FFT_SIZE))
            dir_fft = np.fft.fftshift(np.fft.fft(dir_data, FFT_SIZE))

            omni_power = 20 * np.log10(np.abs(omni_fft) + 1e-12)
            dir_power = 20 * np.log10(np.abs(dir_fft) + 1e-12)

            # Update noise floor (median = more stable than percentile)
            self.noise_floor = float(np.percentile(omni_power, 50))

            # Store for spectrum display
            self.last_spectrum_freq = self.current_freq
            self.last_spectrum_omni = omni_power.copy()
            self.last_spectrum_dir = dir_power.copy()

            # Accumulate sweep for full 3-6 GHz waterfall (one row per full sweep)
            col = min(WATERFALL_COLS - 1, int((self.current_freq - SWEEP_START_HZ) / SWEEP_STEP_HZ))
            if len(self.sweep_buffer_omni) < WATERFALL_COLS:
                self.sweep_buffer_omni = [-100.0] * WATERFALL_COLS
                self.sweep_buffer_dir = [-100.0] * WATERFALL_COLS
            self.sweep_buffer_omni[col] = max(self.sweep_buffer_omni[col], float(np.max(omni_power)))
            self.sweep_buffer_dir[col] = max(self.sweep_buffer_dir[col], float(np.max(dir_power)))

            # Check warning bands - use REFERENCE noise floor (from quiet freqs), not current
            freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / SAMPLE_RATE))
            freq_hz = self.current_freq + freqs

            peak_db = max(np.max(omni_power), np.max(dir_power))
            self.last_peak_db = peak_db

            # Update reference noise floor when we're NOT in a warning band (quiet baseline)
            in_any_warning_band = any(
                (self.current_freq + BANDWIDTH_HZ/2 >= band_start) and
                (self.current_freq - BANDWIDTH_HZ/2 <= band_end)
                for band_start, band_end in WARNING_BANDS
            )
            if not in_any_warning_band:
                # Exponential moving average for stable baseline
                alpha = 0.1
                self.reference_noise_floor = alpha * self.noise_floor + (1 - alpha) * self.reference_noise_floor

            # Only warn when signal in band exceeds reference + threshold
            in_warning_band = False
            threshold_db = self.reference_noise_floor + self.threshold_slider.value()
            for band_start, band_end in WARNING_BANDS:
                mask = (freq_hz >= band_start) & (freq_hz <= band_end)
                if np.any(mask):
                    band_peak = max(np.max(omni_power[mask]), np.max(dir_power[mask]))
                    if band_peak > threshold_db:
                        in_warning_band = True
                        break

            self.warning_active = in_warning_band

            # Advance sweep - when we complete a full cycle, add to waterfall
            self.current_freq += SWEEP_STEP_HZ * self.sweep_direction
            if self.current_freq >= SWEEP_END_HZ:
                self.waterfall_omni.append(np.array(self.sweep_buffer_omni))
                self.waterfall_dir.append(np.array(self.sweep_buffer_dir))
                self.sweep_buffer_omni = [-100.0] * WATERFALL_COLS
                self.sweep_buffer_dir = [-100.0] * WATERFALL_COLS
                self.sweep_direction = -1
            elif self.current_freq <= SWEEP_START_HZ:
                self.waterfall_omni.append(np.array(self.sweep_buffer_omni))
                self.waterfall_dir.append(np.array(self.sweep_buffer_dir))
                self.sweep_buffer_omni = [-100.0] * WATERFALL_COLS
                self.sweep_buffer_dir = [-100.0] * WATERFALL_COLS
                self.sweep_direction = 1

        except Exception as e:
            print(f"Sweep error: {e}")
            self.connected = False

    def update_display(self):
        self.freq_label.setText(f"Current: {self.current_freq / 1e9:.3f} GHz")

        # Update warning
        if self.warning_active:
            self.warning_label.setText("⚠ DRONE DETECTED - Strong signal in 3.5/5.8 GHz band!")
            self.warning_label.setStyleSheet(
                "color: white; background: #c00; padding: 10px; "
                "border-radius: 5px; font-weight: bold;"
            )
        else:
            self.warning_label.setText("")
            self.warning_label.setStyleSheet(
                "color: #333; background: #333; padding: 10px; border-radius: 5px;"
            )

        # Update spectrum curves (current 20 MHz window)
        if self.last_spectrum_omni is not None:
            freqs_hz = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / SAMPLE_RATE))
            freqs_mhz = self.last_spectrum_freq / 1e9 + freqs_hz / 1e9
            self.spectrum_omni.clear()
            self.spectrum_omni.plot(freqs_mhz, self.last_spectrum_omni, pen=pg.mkPen("c", width=2))
            self.spectrum_dir.clear()
            self.spectrum_dir.plot(freqs_mhz, self.last_spectrum_dir, pen=pg.mkPen("m", width=2))

        # Update waterfall images (3-6 GHz, newest at top)
        if len(self.waterfall_omni) > 0:
            wf_data = np.array(list(self.waterfall_omni))
            wf_norm = np.clip((wf_data - self.noise_floor) / 50, 0, 1)
            self.wf_omni_img.setImage(np.flipud(wf_norm), autoLevels=False)

        if len(self.waterfall_dir) > 0:
            wf_data = np.array(list(self.waterfall_dir))
            wf_norm = np.clip((wf_data - self.noise_floor) / 50, 0, 1)
            self.wf_dir_img.setImage(np.flipud(wf_norm), autoLevels=False)


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DroneDetector()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
