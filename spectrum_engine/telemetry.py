"""
Telemetry and structured logging for the Spectrum Sensing Engine (spec section 17).

Writes newline-delimited JSON (JSONL) to a log file. Two record types:
  - MEASUREMENT  : one record per capture/detection cycle
  - SCORE_DUMP   : periodic snapshot of all cell scores
  - TRACK        : when a track changes state

All logging is best-effort; errors are silently swallowed so a logging
failure never crashes the scan loop.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .scheduler import MeasurementCommand
    from .detectors import CoarseMeasurement, FineMeasurement, DetectedRegion
    from .spectrum_grid import SpectrumCell
    from .tracks import SignalTrack


class TelemetryLogger:
    """
    JSONL telemetry writer.

    Parameters
    ----------
    log_dir :
        Directory where log files are written. Created if absent.
    score_dump_interval_s :
        Minimum seconds between periodic cell-score dumps.
    enabled :
        Set to False to disable all I/O (useful in tests).
    """

    def __init__(
        self,
        log_dir: str = "spectrum_logs",
        score_dump_interval_s: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._score_dump_interval = score_dump_interval_s
        self._last_score_dump = 0.0
        self._fh = None

        if not enabled:
            return

        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(log_dir, f"engine_{ts}.jsonl")
            self._fh = open(path, "w", encoding="utf-8", buffering=8192)
        except Exception:
            self._fh = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_measurement(
        self,
        cmd: "MeasurementCommand",
        coarse: Optional["CoarseMeasurement"],
        fine: Optional["FineMeasurement"],
        affected_cells: List["SpectrumCell"],
        tracks: List["SignalTrack"],
        now: float,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            record = {
                "type": "MEASUREMENT",
                "timestamp": now,
                "center_hz": cmd.center_hz,
                "bandwidth_hz": cmd.bandwidth_hz,
                "target_type": str(cmd.target_type),
                "reason": cmd.reason,
                "dwell_s": cmd.dwell_s,
                "fft_size": cmd.fft_size,
                "num_frames": cmd.num_frames,
                "is_fine": cmd.is_fine,
            }

            if coarse is not None:
                record.update({
                    "noise_floor_db": coarse.noise_floor_db,
                    "peak_db": coarse.peak_db,
                    "band_power_db": coarse.band_power_db,
                    "occupied_bw_hz": coarse.occupied_bw_hz,
                    "occupied_bin_fraction": coarse.occupied_bin_fraction,
                    "energy_delta_db": coarse.energy_delta_db,
                    "coarse_suspicious": coarse.coarse_suspicious,
                })

            if fine is not None:
                record["fine_confidence"] = fine.confidence
                record["fine_occupied_bw_hz"] = fine.occupied_bw_hz
                record["fine_regions"] = [
                    {
                        "center_hz": r.center_hz,
                        "bandwidth_hz": r.bandwidth_hz,
                        "peak_db": r.peak_db,
                        "snr_like_db": r.snr_like_db,
                        "confidence": r.confidence,
                    }
                    for r in fine.detected_regions
                ]

            record["affected_cells"] = [c.cell_id for c in affected_cells]

            record["tracks"] = [
                {
                    "track_id": t.track_id,
                    "center_hz": t.center_hz,
                    "state": t.state,
                    "confidence": t.confidence,
                    "persistence_count": t.persistence_count,
                    "missed_count": t.missed_count,
                }
                for t in tracks
                if t.state != "EXPIRED"
            ]

            self._write(record)
        except Exception:
            pass

    def log_score_dump_if_due(
        self,
        cells: List["SpectrumCell"],
        now: float,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        if now - self._last_score_dump < self._score_dump_interval:
            return
        self._last_score_dump = now
        try:
            record = {
                "type": "SCORE_DUMP",
                "timestamp": now,
                "cells": [
                    {
                        "cell_id": c.cell_id,
                        "center_hz": c.center_hz,
                        "state": c.state,
                        "scan_score": round(c.scan_score, 3),
                        "occupancy_prob": round(c.occupancy_prob, 4),
                        "uncertainty": round(c.uncertainty, 4),
                        "energy_delta_db": round(c.energy_delta_db, 2),
                        "last_scan_age_s": round(now - c.last_scan_time, 3) if c.last_scan_time > 0 else -1,
                        "priority_weight": c.priority_weight,
                    }
                    for c in cells
                    if c.state != "UNSUPPORTED"
                ],
            }
            self._write(record)
        except Exception:
            pass

    def log_track_event(
        self,
        track: "SignalTrack",
        event: str,
        now: float,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        try:
            record = {
                "type": "TRACK_EVENT",
                "timestamp": now,
                "event": event,
                "track_id": track.track_id,
                "center_hz": track.center_hz,
                "bandwidth_hz": track.bandwidth_hz,
                "state": track.state,
                "confidence": track.confidence,
                "persistence_count": track.persistence_count,
                "missed_count": track.missed_count,
                "classification": track.classification,
            }
            self._write(record)
        except Exception:
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        try:
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
