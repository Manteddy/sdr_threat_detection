"""
ClassicProcessor — wraps the existing CA-CFAR fine detector.

Delegates to `spectrum_engine.detectors.compute_fine_measurement`
unchanged, so behaviour under this processor is byte-identical to the
pre-pluggable engine. This is the default, and the safety net for any
new processor that turns out worse than CA-CFAR on real signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import SignalProcessor

if TYPE_CHECKING:
    from spectrum_engine.config import EngineConfig
    from spectrum_engine.detectors import FineMeasurement
    from spectrum_engine.iq_source import IQCapture


class ClassicProcessor(SignalProcessor):
    name = "classic"
    label = "Classic (CA-CFAR)"

    def process_fine(
        self,
        capture: "IQCapture",
        psd_db: np.ndarray,
        freq_axis_hz: np.ndarray,
        cfg: "EngineConfig",
    ) -> "FineMeasurement":
        # Imported lazily — keeps this module import-safe even if
        # spectrum_engine.engine is mid-load when the package is first
        # touched (engine.py imports signal_pipeline at module top).
        from spectrum_engine.detectors import compute_fine_measurement

        return compute_fine_measurement(
            psd_db=psd_db,
            freq_axis=freq_axis_hz,
            timestamp=capture.timestamp,
            center_hz=float(capture.center_hz),
            bandwidth_hz=float(capture.bandwidth_hz),
            cfg=cfg,
        )
