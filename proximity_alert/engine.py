"""
Core alert engine.

Accepts signal updates (freq, dBFS, distance, confidence) and produces
ProximityAlert objects with threat level, trend direction, and color.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class TrendDirection(Enum):
    RISING = auto()
    STABLE = auto()
    FALLING = auto()
    UNKNOWN = auto()


class ThreatLevel(Enum):
    NONE = auto()
    DETECTED = auto()
    APPROACHING = auto()
    CRITICAL = auto()


THREAT_COLORS = {
    ThreatLevel.NONE:        "#8e8e93",
    ThreatLevel.DETECTED:    "#ffd60a",
    ThreatLevel.APPROACHING: "#ff9500",
    ThreatLevel.CRITICAL:    "#ff3b30",
}

TREND_ARROWS = {
    TrendDirection.RISING:  "\u25b2",
    TrendDirection.STABLE:  "\u25ac",
    TrendDirection.FALLING: "\u25bc",
    TrendDirection.UNKNOWN: "?",
}


@dataclass
class ProximityAlert:
    freq_ghz: float
    signal_dbfs: float
    distance_m: Optional[float]
    confidence: float
    threat: ThreatLevel
    trend: TrendDirection
    color: str
    trend_arrow: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "freq_ghz": round(self.freq_ghz, 3),
            "signal_dbfs": round(self.signal_dbfs, 1),
            "distance_m": round(self.distance_m, 1) if self.distance_m is not None else None,
            "confidence": round(self.confidence, 2),
            "threat": self.threat.name,
            "trend": self.trend.name,
            "color": self.color,
            "trend_arrow": self.trend_arrow,
            "timestamp": round(self.timestamp, 3),
        }

    def summary_line(self) -> str:
        dist_str = f"~{self.distance_m:.0f}m" if self.distance_m is not None else "??m"
        return (
            f"\u26a0 {self.threat.name} | {self.freq_ghz:.3f} GHz | "
            f"{self.signal_dbfs:.0f} dBFS | {dist_str} | "
            f"{self.trend_arrow} {self.trend.name.lower()}"
        )


class _FreqTracker:
    """Per-frequency state: signal history, trend, boundary hysteresis."""

    def __init__(self, enter_m: float, exit_m: float, history_len: int):
        self.enter_m = enter_m
        self.exit_m = exit_m
        self.history: deque[float] = deque(maxlen=history_len)
        self.inside_boundary = False

    def push(self, signal_dbfs: float):
        self.history.append(signal_dbfs)

    def trend(self, window: int = 6) -> TrendDirection:
        if len(self.history) < window:
            return TrendDirection.UNKNOWN
        recent = list(self.history)[-window:]
        first_half = sum(recent[: window // 2]) / (window // 2)
        second_half = sum(recent[window // 2 :]) / (window - window // 2)
        diff = second_half - first_half
        if diff > 1.5:
            return TrendDirection.RISING
        if diff < -1.5:
            return TrendDirection.FALLING
        return TrendDirection.STABLE

    def update_boundary(self, distance_m: Optional[float]):
        if distance_m is None:
            return
        if not self.inside_boundary and distance_m < self.enter_m:
            self.inside_boundary = True
        elif self.inside_boundary and distance_m > self.exit_m:
            self.inside_boundary = False


class AlertEngine:
    """
    Feed signal updates, get back ProximityAlert objects.

    Usage:
        engine = AlertEngine()
        alert = engine.update(freq_ghz=5.8, signal_dbfs=-38, distance_m=87, confidence=0.6)
        if alert.threat != ThreatLevel.NONE:
            print(alert.summary_line())
    """

    def __init__(
        self,
        enter_m: float = 95.0,
        exit_m: float = 120.0,
        detection_threshold_dbfs: float = -50.0,
        critical_m: float = 100.0,
        approaching_m: float = 250.0,
        history_len: int = 60,
        trend_window: int = 6,
    ):
        self.enter_m = enter_m
        self.exit_m = exit_m
        self.detection_threshold = detection_threshold_dbfs
        self.critical_m = critical_m
        self.approaching_m = approaching_m
        self.history_len = history_len
        self.trend_window = trend_window
        self._trackers: Dict[float, _FreqTracker] = {}

    def _get_tracker(self, freq_ghz: float) -> _FreqTracker:
        key = round(freq_ghz, 3)
        if key not in self._trackers:
            self._trackers[key] = _FreqTracker(
                self.enter_m, self.exit_m, self.history_len
            )
        return self._trackers[key]

    def update(
        self,
        freq_ghz: float,
        signal_dbfs: float,
        distance_m: Optional[float] = None,
        confidence: float = 0.0,
    ) -> ProximityAlert:
        tk = self._get_tracker(freq_ghz)
        tk.push(signal_dbfs)
        tk.update_boundary(distance_m)

        trend = tk.trend(self.trend_window)

        if signal_dbfs < self.detection_threshold:
            threat = ThreatLevel.NONE
        elif tk.inside_boundary or (distance_m is not None and distance_m < self.critical_m):
            threat = ThreatLevel.CRITICAL
        elif distance_m is not None and distance_m < self.approaching_m:
            threat = ThreatLevel.APPROACHING
        else:
            threat = ThreatLevel.DETECTED

        if threat != ThreatLevel.NONE and trend == TrendDirection.RISING:
            if threat == ThreatLevel.APPROACHING:
                threat = ThreatLevel.CRITICAL

        return ProximityAlert(
            freq_ghz=freq_ghz,
            signal_dbfs=signal_dbfs,
            distance_m=distance_m,
            confidence=confidence,
            threat=threat,
            trend=trend,
            color=THREAT_COLORS[threat],
            trend_arrow=TREND_ARROWS[trend],
        )

    def get_all_active(self) -> List[ProximityAlert]:
        """Return last-known alerts for all tracked frequencies."""
        results = []
        for freq, tk in self._trackers.items():
            if tk.history:
                results.append(
                    self.update(freq, tk.history[-1])
                )
        return results

    def reset(self, freq_ghz: Optional[float] = None):
        if freq_ghz is not None:
            self._trackers.pop(round(freq_ghz, 3), None)
        else:
            self._trackers.clear()
