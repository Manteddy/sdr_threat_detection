"""
Signal track manager (spec section 10).

Tracks represent persistent RF emitters confirmed across multiple measurements.
State machine: CANDIDATE → CONFIRMED → TRACKING → LOST → EXPIRED.

The track manager:
  - Creates CANDIDATE tracks from fine-measurement DetectedRegion objects.
  - Promotes to CONFIRMED after persistent observations.
  - Updates frequency/bandwidth with exponential smoothing (alpha=0.3).
  - Marks missed detections and decays confidence.
  - Expires tracks that have not been seen for too long.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .detectors import DetectedRegion
    from .config import EngineConfig
    from .spectrum_grid import SpectrumCell


# ---------------------------------------------------------------------------
# Track states
# ---------------------------------------------------------------------------

class TrackState:
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    TRACKING = "TRACKING"
    LOST = "LOST"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# SignalTrack
# ---------------------------------------------------------------------------

@dataclass
class SignalTrack:
    track_id: int

    center_hz: float
    bandwidth_hz: float
    start_hz: float
    stop_hz: float

    first_seen_time: float
    last_seen_time: float
    last_revisit_time: float

    peak_db: float = -140.0
    noise_floor_db: float = -100.0
    snr_like_db: float = 0.0

    confidence: float = 0.0
    persistence_count: int = 0
    missed_count: int = 0

    state: str = TrackState.CANDIDATE
    history: Deque = field(default_factory=lambda: deque(maxlen=100))
    classification: str = "UNKNOWN"  # UNKNOWN, NARROWBAND, WIDEBAND, OFDM_LIKE

    @property
    def revisit_interval(self) -> float:
        """Recommended revisit interval in seconds based on track state."""
        if self.state == TrackState.CONFIRMED:
            return 0.10
        if self.state == TrackState.TRACKING:
            return 0.05
        if self.state == TrackState.CANDIDATE:
            return 0.10
        if self.state == TrackState.LOST:
            return 0.50
        return 1.0

    def overlaps(self, region: "DetectedRegion", overlap_fraction: float = 0.3) -> bool:
        """Return True if a detected region significantly overlaps this track."""
        lo = max(self.start_hz, region.start_hz)
        hi = min(self.stop_hz, region.stop_hz)
        if hi <= lo:
            return False
        overlap_bw = hi - lo
        track_bw = max(self.stop_hz - self.start_hz, 1.0)
        region_bw = max(region.stop_hz - region.start_hz, 1.0)
        return overlap_bw / min(track_bw, region_bw) >= overlap_fraction


# ---------------------------------------------------------------------------
# Track manager
# ---------------------------------------------------------------------------

class TrackManager:
    """
    Manages the lifecycle of all signal tracks.
    """

    _ALPHA: float = 0.3          # EMA weight for frequency/BW updates
    _PROMOTE_MIN_HITS: int = 2   # hits in last N checks to promote
    _PROMOTE_WINDOW: int = 3     # window for promotion check
    _PROMOTE_CONF_THRESHOLD: float = 0.60
    _CANDIDATE_CONF_THRESHOLD: float = 0.40
    _LOST_MISSED_THRESHOLD: int = 3
    _EXPIRED_MISSED_THRESHOLD: int = 10
    _EXPIRED_CONF_THRESHOLD: float = 0.10
    _MAX_AGE_S: float = 30.0     # expire any track unseen for this long

    def __init__(self, cfg: "EngineConfig") -> None:
        self._cfg = cfg
        self._tracks: Dict[int, SignalTrack] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> List[SignalTrack]:
        return list(self._tracks.values())

    @property
    def active_tracks(self) -> List[SignalTrack]:
        return [t for t in self._tracks.values()
                if t.state not in (TrackState.EXPIRED,)]

    def update_from_regions(
        self,
        regions: List["DetectedRegion"],
        now: float,
    ) -> None:
        """
        Match detected regions to existing tracks, create candidates for
        unmatched regions, and record misses on unmatched tracks.
        """
        matched_track_ids = set()
        matched_region_indices = set()

        # --- Match regions to existing tracks ---
        for i, region in enumerate(regions):
            best_id = None
            best_overlap = 0.0
            for tid, track in self._tracks.items():
                if track.state == TrackState.EXPIRED:
                    continue
                if track.overlaps(region):
                    lo = max(track.start_hz, region.start_hz)
                    hi = min(track.stop_hz, region.stop_hz)
                    overlap = (hi - lo) / max(track.stop_hz - track.start_hz, 1.0)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_id = tid

            if best_id is not None:
                self._update_track(self._tracks[best_id], region, now)
                matched_track_ids.add(best_id)
                matched_region_indices.add(i)

        # --- Create candidates for unmatched regions ---
        for i, region in enumerate(regions):
            if i in matched_region_indices:
                continue
            if region.confidence >= self._CANDIDATE_CONF_THRESHOLD or region.snr_like_db >= 10.0:
                self._create_track(region, now)

        # --- Record misses for unmatched tracks ---
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids and track.state != TrackState.EXPIRED:
                self._record_miss(track)

    def update_track_states(self, now: float) -> None:
        """
        Run state-machine transitions and expire old tracks.
        Called once per engine step after update_from_regions.
        """
        for track in list(self._tracks.values()):
            if track.state == TrackState.EXPIRED:
                continue
            self._update_state(track, now)

    def update_cell_states_from_tracks(self, cells: List["SpectrumCell"]) -> None:
        """
        Mark cells as TRACKED when a confirmed track covers them.
        """
        from .spectrum_grid import CellState
        # Reset TRACKED cells first
        for cell in cells:
            if cell.state == CellState.TRACKED:
                cell.state = CellState.ACTIVE  # revert; will be re-TRACKED below

        for track in self._tracks.values():
            if track.state in (TrackState.CONFIRMED, TrackState.TRACKING):
                for cell in cells:
                    if cell.state == CellState.UNSUPPORTED:
                        continue
                    if cell.stop_hz > track.start_hz and cell.start_hz < track.stop_hz:
                        cell.state = CellState.TRACKED
                        cell.active_track_id = track.track_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, region: "DetectedRegion", now: float) -> SignalTrack:
        tid = self._next_id
        self._next_id += 1
        t = SignalTrack(
            track_id=tid,
            center_hz=region.center_hz,
            bandwidth_hz=region.bandwidth_hz,
            start_hz=region.start_hz,
            stop_hz=region.stop_hz,
            first_seen_time=now,
            last_seen_time=now,
            last_revisit_time=now,
            peak_db=region.peak_db,
            noise_floor_db=region.noise_floor_db,
            snr_like_db=region.snr_like_db,
            confidence=region.confidence,
            persistence_count=1,
        )
        self._tracks[tid] = t
        return t

    def _update_track(
        self,
        track: SignalTrack,
        region: "DetectedRegion",
        now: float,
    ) -> None:
        a = self._ALPHA
        track.center_hz = a * region.center_hz + (1 - a) * track.center_hz
        track.bandwidth_hz = a * region.bandwidth_hz + (1 - a) * track.bandwidth_hz
        track.start_hz = track.center_hz - track.bandwidth_hz / 2.0
        track.stop_hz = track.center_hz + track.bandwidth_hz / 2.0
        track.peak_db = region.peak_db
        track.noise_floor_db = region.noise_floor_db
        track.snr_like_db = region.snr_like_db
        track.confidence = min(
            a * region.confidence + (1 - a) * track.confidence + 0.05,
            1.0,
        )
        track.last_seen_time = now
        track.last_revisit_time = now
        track.missed_count = 0
        track.persistence_count += 1

        # Record history sample
        track.history.append({
            "t": now,
            "center_hz": track.center_hz,
            "peak_db": track.peak_db,
            "confidence": track.confidence,
        })

        # Classify bandwidth
        if track.bandwidth_hz < 2e6:
            track.classification = "NARROWBAND"
        elif track.bandwidth_hz > 20e6:
            track.classification = "WIDEBAND"
        else:
            track.classification = "UNKNOWN"

    def _record_miss(self, track: SignalTrack) -> None:
        track.missed_count += 1
        track.confidence *= 0.80

    def _update_state(self, track: SignalTrack, now: float) -> None:
        age = now - track.last_seen_time

        # Expire by age
        if age > self._MAX_AGE_S:
            track.state = TrackState.EXPIRED
            return

        # Expire by missed count or confidence
        if (track.missed_count >= self._EXPIRED_MISSED_THRESHOLD
                or track.confidence < self._EXPIRED_CONF_THRESHOLD):
            track.state = TrackState.EXPIRED
            return

        if track.state == TrackState.CANDIDATE:
            # Promote if seen enough times and confidence high enough
            if (track.persistence_count >= self._PROMOTE_MIN_HITS
                    and track.confidence >= self._PROMOTE_CONF_THRESHOLD):
                track.state = TrackState.CONFIRMED
            elif track.missed_count >= self._LOST_MISSED_THRESHOLD:
                track.state = TrackState.LOST
            return

        if track.state in (TrackState.CONFIRMED, TrackState.TRACKING):
            if track.missed_count >= self._LOST_MISSED_THRESHOLD:
                track.state = TrackState.LOST
            return

        if track.state == TrackState.LOST:
            if track.missed_count == 0:
                # Re-seen
                track.state = TrackState.TRACKING
            return
