"""
Demo simulator — run with:

    python -m proximity_alert.demo

Shows the ProximityAlertPanel with simulated signal data:
a drone approaches from 500 m, gets close, then retreats.
No SDR hardware needed.
"""

from __future__ import annotations

import math
import sys
import time

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from .engine import AlertEngine
from .widget import ProximityAlertPanel

DEMO_FREQS = [5.800, 5.900]
SIM_INTERVAL_MS = 200


class _SimState:
    """Simulate a drone flying an approach-and-retreat pattern."""

    def __init__(self):
        self.t = 0.0
        self.phase = "approach"

    def tick(self, dt: float):
        self.t += dt

    def distance(self) -> float:
        cycle = 60.0
        half = cycle / 2
        pos = self.t % cycle
        if pos < half:
            return 500.0 - (500.0 - 40.0) * (pos / half)
        return 40.0 + (500.0 - 40.0) * ((pos - half) / half)

    def signal_dbfs(self, base_ref: float = -30.0, ref_dist: float = 10.0, n: float = 2.2) -> float:
        d = max(self.distance(), 1.0)
        noise = 2.0 * math.sin(self.t * 3.7) + 1.5 * math.cos(self.t * 7.1)
        return base_ref - 10.0 * n * math.log10(d / ref_dist) + noise

    def confidence(self) -> float:
        return min(1.0, self.t / 10.0)


class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proximity Alert — Demo")
        self.setMinimumSize(600, 340)
        self.setStyleSheet("background:#0e0e1a;color:#ccc;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("PROXIMITY ALERT PANEL — SIMULATED DATA")
        header.setFont(QFont("Segoe UI", 10))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color:#666;padding:4px;")
        layout.addWidget(header)

        self.panel = ProximityAlertPanel(AlertEngine(
            enter_m=95.0,
            exit_m=120.0,
            detection_threshold_dbfs=-55.0,
            critical_m=100.0,
            approaching_m=250.0,
        ))
        layout.addWidget(self.panel)

        self._sims = {f: _SimState() for f in DEMO_FREQS}
        self._sims[DEMO_FREQS[1]].t = 15.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(SIM_INTERVAL_MS)

    def _tick(self):
        dt = SIM_INTERVAL_MS / 1000.0
        for freq, sim in self._sims.items():
            sim.tick(dt)
            self.panel.push(
                freq_ghz=freq,
                signal_dbfs=sim.signal_dbfs(),
                distance_m=sim.distance(),
                confidence=sim.confidence(),
            )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = DemoWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
