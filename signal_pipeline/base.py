"""
SignalProcessor base class + shared output dataclasses.

A SignalProcessor consumes one fine-measurement worth of PSD (already
computed by the engine via Welch averaging) plus the underlying IQ
capture, and returns a `FineMeasurement` populated with detected regions
and (optionally) per-region classifications.

The base class is concrete — subclasses override `process_fine`. Using a
plain base class instead of typing.Protocol keeps isinstance checks and
default attributes simple, and matches the existing codebase style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spectrum_engine.config import EngineConfig
    from spectrum_engine.detectors import FineMeasurement
    from spectrum_engine.iq_source import IQCapture


@dataclass(frozen=True)
class ClassificationResult:
    """
    One classifier verdict for a single detected region.

    Defined here so it can be referenced by FineMeasurement (in
    spectrum_engine.detectors) without dragging a dependency on any
    specific processor into the engine.
    """
    frequency_hz: float
    bandwidth_hz: float
    signal_strength_db: float
    classification: str          # "ANALOG_FPV" | "BARRAGE_JAMMER" | "TELEMETRY_LINK" | "UNKNOWN"
    confidence_score: float      # 0.0 - 1.0
    features: Dict[str, Any] = field(default_factory=dict)


class SignalProcessor:
    """
    Base class for every pluggable fine-scan detector.

    Subclasses set `name` (short slug) and `label` (human label for GUI)
    as class attributes, and override `process_fine`.

    Implementations must be stateless across calls — the engine reuses a
    single instance for every fine scan, and the GUI swaps instances on
    runtime selection changes. Internal caches (window arrays, etc.) are
    fine; mutable per-call state is not.
    """

    name: str = "base"
    label: str = "Base SignalProcessor"

    def process_fine(
        self,
        capture: "IQCapture",
        psd_db: np.ndarray,
        freq_axis_hz: np.ndarray,
        cfg: "EngineConfig",
    ) -> "FineMeasurement":
        raise NotImplementedError("SignalProcessor subclasses must implement process_fine()")
