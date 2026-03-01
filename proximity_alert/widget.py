"""
Embeddable PyQt5 widget — drop into any PyQt5 / pyqtgraph dashboard.

    from proximity_alert.widget import ProximityAlertPanel
    panel = ProximityAlertPanel()
    layout.addWidget(panel)
    ...
    panel.push(freq_ghz=5.8, signal_dbfs=-38, distance_m=87, confidence=0.6)
"""

from __future__ import annotations

from collections import deque
from typing import Dict

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .engine import AlertEngine, ProximityAlert, ThreatLevel, TrendDirection


_PANEL_BG = "#111118"
_ROW_BG = "#1a1a2e"
_BORDER = "#333"

_THREAT_BG = {
    ThreatLevel.NONE:        "transparent",
    ThreatLevel.DETECTED:    "#3a3500",
    ThreatLevel.APPROACHING: "#3a2200",
    ThreatLevel.CRITICAL:    "#500000",
}

_FLASH_INTERVAL_MS = 400


class _FreqRow(QFrame):
    """One row per monitored frequency."""

    def __init__(self, freq_ghz: float, parent=None):
        super().__init__(parent)
        self.freq_ghz = freq_ghz
        self.setStyleSheet(f"background:{_ROW_BG};border:1px solid {_BORDER};")
        self._flash_state = False

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)

        self.freq_label = QLabel(f"{freq_ghz:.3f} GHz")
        self.freq_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.freq_label.setStyleSheet("color:#0ff;border:none;")
        self.freq_label.setFixedWidth(120)
        row.addWidget(self.freq_label)

        self.signal_label = QLabel("-- dBFS")
        self.signal_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setFixedWidth(100)
        row.addWidget(self.signal_label)

        self.dist_label = QLabel("-- m")
        self.dist_label.setFont(QFont("Consolas", 11))
        self.dist_label.setAlignment(Qt.AlignCenter)
        self.dist_label.setFixedWidth(90)
        row.addWidget(self.dist_label)

        self.trend_label = QLabel("")
        self.trend_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.trend_label.setAlignment(Qt.AlignCenter)
        self.trend_label.setFixedWidth(40)
        row.addWidget(self.trend_label)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        row.addWidget(self.status_label, 1)

    def apply_alert(self, alert: ProximityAlert):
        self.signal_label.setText(f"{alert.signal_dbfs:.0f} dBFS")
        self.signal_label.setStyleSheet(
            f"color:{alert.color};border:none;font-weight:bold;"
        )

        if alert.distance_m is not None:
            self.dist_label.setText(f"~{alert.distance_m:.0f}m")
            self.dist_label.setStyleSheet(f"color:{alert.color};border:none;")
        else:
            self.dist_label.setText("-- m")
            self.dist_label.setStyleSheet("color:#666;border:none;")

        self.trend_label.setText(alert.trend_arrow)
        trend_color = {
            TrendDirection.RISING: "#ff3b30",
            TrendDirection.STABLE: "#ffd60a",
            TrendDirection.FALLING: "#34c759",
            TrendDirection.UNKNOWN: "#8e8e93",
        }[alert.trend]
        self.trend_label.setStyleSheet(f"color:{trend_color};border:none;")

        self.status_label.setText(alert.threat.name)
        self.status_label.setStyleSheet(f"color:{alert.color};border:none;")

        bg = _THREAT_BG[alert.threat]
        self.setStyleSheet(
            f"background:{bg};border:1px solid {_BORDER};"
        )

    def flash_toggle(self, critical: bool):
        if not critical:
            self._flash_state = False
            return
        self._flash_state = not self._flash_state
        border = "#ff3b30" if self._flash_state else _BORDER
        self.setStyleSheet(
            f"background:{_THREAT_BG[ThreatLevel.CRITICAL]};"
            f"border:2px solid {border};"
        )


class ProximityAlertPanel(QWidget):
    """
    Self-contained proximity alert panel.

    Embed in any PyQt5 layout, then call push() each sweep.
    """

    def __init__(self, engine: AlertEngine | None = None, parent=None):
        super().__init__(parent)
        self.engine = engine or AlertEngine()
        self._rows: Dict[float, _FreqRow] = {}
        self._last_alerts: Dict[float, ProximityAlert] = {}

        self.setStyleSheet(f"background:{_PANEL_BG};")
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(6, 6, 6, 6)
        self._root.setSpacing(4)

        self._banner = QLabel("")
        self._banner.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self._banner.setAlignment(Qt.AlignCenter)
        self._banner.setMinimumHeight(36)
        self._banner.setStyleSheet("color:transparent;padding:4px;border:none;")
        self._root.addWidget(self._banner)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(3)
        self._root.addLayout(self._rows_layout)
        self._root.addStretch()

        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._flash_tick)
        self._flash_timer.start(_FLASH_INTERVAL_MS)

    def push(
        self,
        freq_ghz: float,
        signal_dbfs: float,
        distance_m: float | None = None,
        confidence: float = 0.0,
    ) -> ProximityAlert:
        alert = self.engine.update(freq_ghz, signal_dbfs, distance_m, confidence)
        key = round(freq_ghz, 3)

        if key not in self._rows:
            row = _FreqRow(freq_ghz, self)
            self._rows[key] = row
            self._rows_layout.addWidget(row)

        self._rows[key].apply_alert(alert)
        self._last_alerts[key] = alert
        self._update_banner()
        return alert

    def _update_banner(self):
        criticals = [
            a for a in self._last_alerts.values()
            if a.threat == ThreatLevel.CRITICAL
        ]
        approaching = [
            a for a in self._last_alerts.values()
            if a.threat == ThreatLevel.APPROACHING
        ]

        if criticals:
            worst = min(criticals, key=lambda a: a.distance_m or 9999)
            self._banner.setText(worst.summary_line())
            self._banner.setStyleSheet(
                "color:white;background:#b00;padding:4px;"
                "border-radius:3px;font-weight:bold;border:none;"
            )
        elif approaching:
            worst = min(approaching, key=lambda a: a.distance_m or 9999)
            self._banner.setText(worst.summary_line())
            self._banner.setStyleSheet(
                "color:white;background:#a05000;padding:4px;"
                "border-radius:3px;font-weight:bold;border:none;"
            )
        else:
            self._banner.setText("")
            self._banner.setStyleSheet("color:transparent;padding:4px;border:none;")

    def _flash_tick(self):
        for key, alert in self._last_alerts.items():
            row = self._rows.get(key)
            if row:
                row.flash_toggle(alert.threat == ThreatLevel.CRITICAL)

    def clear(self):
        self.engine.reset()
        self._last_alerts.clear()
        for row in self._rows.values():
            row.deleteLater()
        self._rows.clear()
        self._banner.setText("")
        self._banner.setStyleSheet("color:transparent;padding:4px;border:none;")
