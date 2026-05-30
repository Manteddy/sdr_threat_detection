"""
Adaptive measurement scheduler for the Spectrum Sensing Engine.

Implements:
  - Scan-score calculation per cell (spec section 8.2)
  - Track-score calculation (spec section 8.3)
  - MeasurementCommand selection with the section-8.4 priority ordering:
      1. Overdue active tracks
      2. Overdue max-revisit cells
      3. Highest-score suspect/active groups
      4. High-priority band cells
      5. Background refresh cells
  - Hard max-revisit override (100-point score boost) so no cell is
    ever silently forgotten (spec section 8.2 / section 6.3).
  - Adaptive dwell/FFT selection per cell state (spec section 6.2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .spectrum_grid import SpectrumCell
    from .tracks import SignalTrack, TrackState
    from .occupancy import ActivityGroup
    from .config import EngineConfig


# ---------------------------------------------------------------------------
# Target types
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    TRACK_REVISIT = "TRACK_REVISIT"
    ACTIVE_GROUP_FINE_SCAN = "ACTIVE_GROUP_FINE_SCAN"
    SUSPECT_CELL_FINE_SCAN = "SUSPECT_CELL_FINE_SCAN"
    PRIORITY_CELL_COARSE_SCAN = "PRIORITY_CELL_COARSE_SCAN"
    STALE_CELL_REFRESH = "STALE_CELL_REFRESH"
    BACKGROUND_CELL_COARSE_SCAN = "BACKGROUND_CELL_COARSE_SCAN"


# ---------------------------------------------------------------------------
# MeasurementCommand
# ---------------------------------------------------------------------------

@dataclass
class MeasurementCommand:
    target_type: TargetType
    center_hz: float
    bandwidth_hz: float
    fft_size: int
    num_frames: int
    dwell_s: float
    reason: str
    target_cell_ids: List[int] = field(default_factory=list)
    target_track_id: Optional[int] = None

    @property
    def is_fine(self) -> bool:
        return self.target_type in (
            TargetType.TRACK_REVISIT,
            TargetType.ACTIVE_GROUP_FINE_SCAN,
            TargetType.SUSPECT_CELL_FINE_SCAN,
        )


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _normalize_delta(energy_delta_db: float, ref: float = 20.0) -> float:
    """Map energy_delta_db to [0,1] with ref dB as saturation point."""
    return min(max(energy_delta_db / ref, 0.0), 1.0)


def _recently_quiet_penalty(cell: "SpectrumCell", now: float) -> float:
    """Mild penalty if a cell was confirmed quiet very recently."""
    from .spectrum_grid import CellState
    if cell.state == CellState.QUIET:
        age = now - cell.last_scan_time
        if age < 1.0:
            return 0.5 * (1.0 - age)
    return 0.0


def compute_cell_scan_score(cell: "SpectrumCell", now: float) -> float:
    """
    Compute and store cell.scan_score.

    Implements the scoring formula from spec section 8.2, including the
    +100 hard override when the cell is past its max_revisit_interval.
    """
    from .spectrum_grid import CellState

    if cell.state == CellState.UNSUPPORTED:
        cell.scan_score = -1000.0
        return cell.scan_score

    age_s = max(0.0, now - cell.last_scan_time)
    if cell.max_revisit_interval_s > 0:
        age_norm = min(age_s / cell.max_revisit_interval_s, 2.0)
    else:
        age_norm = 1.0

    is_tracked = 1.0 if cell.state == CellState.TRACKED else 0.0
    is_active = 1.0 if cell.state == CellState.ACTIVE else 0.0
    is_suspect = 1.0 if cell.state == CellState.SUSPECT else 0.0

    score = (
        3.0 * is_tracked
        + 2.5 * is_active
        + 2.0 * is_suspect
        + 1.5 * cell.priority_weight
        + 2.0 * cell.occupancy_prob
        + 1.5 * _normalize_delta(cell.energy_delta_db)
        + 1.2 * cell.uncertainty
        + 1.0 * age_norm
        - 1.0 * _recently_quiet_penalty(cell, now)
    )

    # Hard override: never let a cell be forgotten
    if age_s > cell.max_revisit_interval_s:
        score += 100.0

    cell.scan_score = score
    return score


def compute_track_score(track: "SignalTrack", now: float) -> float:
    """
    Compute a priority score for a track (spec section 8.3).

    Higher = should be revisited sooner.
    """
    age_revisit = now - track.last_revisit_time
    max_revisit = track.revisit_interval
    age_norm = min(age_revisit / max(max_revisit, 0.01), 2.0)

    # Rising signal = peak increasing trend (approximated by persistence count vs missed)
    rising_score = min(track.persistence_count / max(track.persistence_count + track.missed_count, 1), 1.0)

    lost_penalty = 1.0 if track.state == "LOST" else 0.0

    return (
        5.0 * track.confidence
        + 3.0 * 1.0          # threat_relevance placeholder (always 1 for now)
        + 2.0 * rising_score
        + 2.0 * age_norm
        - 2.0 * lost_penalty
    )


# ---------------------------------------------------------------------------
# Dwell / revisit policy
# ---------------------------------------------------------------------------

def choose_dwell(cell: "SpectrumCell", cfg: "EngineConfig") -> float:
    """Return the recommended dwell time for a cell based on its state."""
    from .spectrum_grid import CellState
    sc = cfg.scheduler
    state = cell.state
    if state == CellState.TRACKED:
        return sc.track_dwell_s
    if state == CellState.ACTIVE:
        return sc.active_dwell_s
    if state == CellState.SUSPECT:
        return sc.fine_dwell_s
    if state in (CellState.STALE, CellState.UNKNOWN):
        return sc.coarse_dwell_s
    return sc.short_coarse_dwell_s  # QUIET


def _fft_and_frames_for_dwell(dwell_s: float, cfg: "EngineConfig", fine: bool):
    """Compute (fft_size, num_frames) that best fits the requested dwell."""
    hw = cfg.hardware
    if fine:
        fft_size = cfg.fine_scan.fft_size
        num_frames = cfg.fine_scan.num_frames
    else:
        fft_size = cfg.coarse_scan.fft_size
        num_frames = cfg.coarse_scan.num_frames
    return fft_size, num_frames


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def _group_fine_scan_age(group: "ActivityGroup", now: float) -> float:
    """Seconds since the most recent fine scan of any cell in this group."""
    times = [c.last_fine_scan_time for c in group.cells if c.last_fine_scan_time > 0]
    if not times:
        return 9999.0
    return now - max(times)


def _group_all_tracked(group: "ActivityGroup") -> bool:
    """True if every cell in this group is already TRACKED (handled via step 1)."""
    from .spectrum_grid import CellState
    return all(c.state == CellState.TRACKED for c in group.cells)


class Scheduler:
    """
    Measurement scheduler.  Call choose_next_measurement() each cycle.

    Fine-scan budget
    ----------------
    Without throttling, step 3 (active groups) fires every iteration when a
    strong persistent signal exists, starving all background coarse scans.
    A consecutive-fine-scan counter limits how many fine scans can happen
    in a row before a coarse background scan is forced.
    """

    # Interleave ratio: 1 in PRIORITY_EVERY cycles is reserved for "interesting"
    # work (track revisits, active-group fine scans, frequent priority-band
    # revisits). The other cycles guarantee full-spectrum coverage. This keeps
    # coverage clearly dominant so the whole band is always being re-swept, while
    # still revisiting active/priority regions far more often than cold ones.
    _PRIORITY_EVERY: int = 4

    def __init__(self, cfg: "EngineConfig") -> None:
        self._cfg = cfg
        self._cycle: int = 0

    def update_scores(self, cells: List["SpectrumCell"], now: float) -> None:
        """Refresh scan_score for every non-unsupported cell."""
        for cell in cells:
            compute_cell_scan_score(cell, now)

    def choose_next_measurement(
        self,
        cells: List["SpectrumCell"],
        tracks: List["SignalTrack"],
        groups: List["ActivityGroup"],
        now: float,
    ) -> MeasurementCommand:
        """
        Select the next measurement command.

        The selection interleaves two concerns so neither starves the other:

          * Coverage  : round-robin by *absolute* staleness (now - last_scan_time)
                        so every cell in the band is re-swept, and never-scanned
                        cells (staleness = now) are visited first.
          * Priority  : on 1-in-_PRIORITY_EVERY cycles, do "interesting" work —
                        overdue track revisits, active-group fine scans (with a
                        cooldown), and frequent coarse revisits of active /
                        priority-band cells.

        This fixes the pathology where short priority revisit intervals (e.g.
        0.5 s on the 2.4 GHz band, unreachable given a ~20 s full sweep) made
        those cells perpetually "overdue" and monopolised the scheduler.
        """
        self._cycle += 1

        # Reserve some cycles for interesting work; the rest guarantee coverage.
        if self._cycle % self._PRIORITY_EVERY == 0:
            cmd = self._try_priority(cells, tracks, groups, now)
            if cmd is not None:
                return cmd

        cmd = self._coverage_scan(cells, now)
        if cmd is not None:
            return cmd

        # Nothing overdue right now → still service the most interesting cell so
        # we never idle, then fall back to a dummy command.
        cmd = self._try_priority(cells, tracks, groups, now)
        if cmd is not None:
            return cmd
        return self._coverage_scan(cells, now, force=True) or self._dummy_cmd(self._cfg)

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _coverage_scan(self, cells: List["SpectrumCell"], now: float,
                       force: bool = False) -> Optional[MeasurementCommand]:
        """
        Pick the cell that has gone longest without a scan (absolute staleness).

        Never-scanned cells (last_scan_time == 0) have staleness == now, so they
        are always serviced before anything else. Round-robin emerges naturally:
        a just-scanned cell resets to 0 staleness and goes to the back of the
        queue, so the whole band is swept before any cell repeats.

        With force=False only cells past their own max_revisit_interval are
        considered (the normal coverage guarantee). With force=True any cell is
        eligible, used as a last resort so the engine never idles.
        """
        from .spectrum_grid import CellState

        best = None
        best_stale = -1.0
        for c in cells:
            if c.state == CellState.UNSUPPORTED:
                continue
            if c.last_scan_time == 0.0:
                stale = now  # never scanned: maximal urgency
            else:
                stale = now - c.last_scan_time
                if not force and stale <= c.max_revisit_interval_s:
                    continue
            if stale > best_stale:
                best_stale = stale
                best = c

        if best is None:
            return None

        if best.last_scan_time == 0.0:
            tt = TargetType.BACKGROUND_CELL_COARSE_SCAN
            reason = "never scanned"
        elif best.priority_weight > 1.0:
            tt = TargetType.PRIORITY_CELL_COARSE_SCAN
            reason = f"coverage {best_stale:.1f}s"
        else:
            tt = TargetType.STALE_CELL_REFRESH
            reason = f"coverage {best_stale:.1f}s"
        return self._make_coarse_cmd(best, self._cfg, tt, reason=reason)

    def _try_priority(
        self,
        cells: List["SpectrumCell"],
        tracks: List["SignalTrack"],
        groups: List["ActivityGroup"],
        now: float,
    ) -> Optional[MeasurementCommand]:
        """
        Interesting work, in order:
          1. Overdue track revisits.
          2. Active/suspect group fine scans (with a per-group cooldown).
          3. Frequent coarse revisit of the highest-score active/priority cell.
        Returns None if there is nothing interesting to do.
        """
        from .spectrum_grid import CellState
        cfg = self._cfg

        # 1. Overdue tracks
        if tracks:
            overdue = [t for t in tracks
                       if now - t.last_revisit_time > t.revisit_interval
                       and t.state not in ("EXPIRED",)]
            if overdue:
                best_track = max(overdue, key=lambda t: compute_track_score(t, now))
                return self._make_track_cmd(best_track, cfg, cells)

        # 2. Active / suspect groups → fine scan (cooldown-limited)
        suspect_cooldown = cfg.scheduler.suspect_revisit_s
        active_groups = [
            g for g in groups
            if g.group_probability > cfg.states.suspect_prob_threshold
            and _group_fine_scan_age(g, now) > suspect_cooldown
            and not _group_all_tracked(g)
        ]
        if active_groups:
            best_group = max(active_groups, key=lambda g: g.group_probability)
            return self._make_group_fine_cmd(best_group, cfg)

        # 3. Tier-based round-robin among "interesting" cells.
        #
        # Previously this picked max(scan_score), which locks the priority slot
        # to whichever active cell has the highest score (e.g. 1.8 GHz) and
        # starves other equally suspicious cells (2.0/2.2 GHz). Instead we use
        # tiers + staleness:
        #
        #   Tier 3 (signal-confirmed):  TRACKED / ACTIVE
        #   Tier 2 (suspected):         SUSPECT
        #   Tier 1 (priority-band):     no signal yet, but configured as a
        #                               drone-relevant band (e.g. 2.4G, 5.8G)
        #
        # We always serve the highest non-empty tier and round-robin within
        # it by absolute staleness (now - last_scan_time). A just-revisited
        # cell resets to staleness=0 and goes to the back of the queue, so
        # if two active cells exist they alternate naturally; if five active
        # cells exist they cycle through all five before any repeat. This is
        # exactly what the user asked for: "frequently revisit ALL frequencies
        # that are suspicious enough".
        #
        # Coverage (3 of 4 cycles) still guarantees every cell gets re-swept,
        # including non-signal priority-band cells, so configured drone bands
        # are never silently forgotten.
        def _tier(c) -> int:
            if c.state in (CellState.TRACKED, CellState.ACTIVE):
                return 3
            if c.state == CellState.SUSPECT:
                return 2
            if c.priority_weight > 1.0:
                return 1
            return 0

        interesting = [c for c in cells if _tier(c) > 0]
        if interesting:
            top_tier = max(_tier(c) for c in interesting)
            tier_cells = [c for c in interesting if _tier(c) == top_tier]

            def _stale_of(c):
                return (now - c.last_scan_time) if c.last_scan_time > 0 else now

            # Pick the cell in the top tier that has waited the longest.
            best = max(tier_cells, key=lambda c: (_stale_of(c), c.scan_score))
            stale = _stale_of(best)
            tt = (TargetType.PRIORITY_CELL_COARSE_SCAN
                  if best.priority_weight > 1.0 else TargetType.BACKGROUND_CELL_COARSE_SCAN)
            return self._make_coarse_cmd(
                best, cfg, tt,
                reason=f"priority t{top_tier} stale={stale:.2f}s score={best.scan_score:.2f}",
            )

        return None

    # ------------------------------------------------------------------
    # Command constructors
    # ------------------------------------------------------------------

    def _make_track_cmd(self, track: "SignalTrack", cfg: "EngineConfig",
                        cells: Optional[List["SpectrumCell"]] = None) -> MeasurementCommand:
        fft_size, num_frames = _fft_and_frames_for_dwell(cfg.scheduler.track_dwell_s, cfg, fine=True)
        # Identify cells overlapping the track's frequency span for coverage tracking
        target_ids: List[int] = []
        if cells is not None:
            for c in cells:
                if c.stop_hz > track.start_hz and c.start_hz < track.stop_hz:
                    target_ids.append(c.cell_id)
        return MeasurementCommand(
            target_type=TargetType.TRACK_REVISIT,
            center_hz=track.center_hz,
            bandwidth_hz=cfg.hardware.default_bandwidth_hz,
            fft_size=fft_size,
            num_frames=num_frames,
            dwell_s=cfg.scheduler.track_dwell_s,
            reason=f"track {track.track_id} revisit",
            target_cell_ids=target_ids,
            target_track_id=track.track_id,
        )

    def _make_coarse_cmd(
        self,
        cell: "SpectrumCell",
        cfg: "EngineConfig",
        target_type: TargetType = TargetType.BACKGROUND_CELL_COARSE_SCAN,
        reason: str = "",
    ) -> MeasurementCommand:
        dwell = choose_dwell(cell, cfg)
        fft_size, num_frames = _fft_and_frames_for_dwell(dwell, cfg, fine=False)
        return MeasurementCommand(
            target_type=target_type,
            center_hz=cell.center_hz,
            bandwidth_hz=cfg.hardware.default_bandwidth_hz,
            fft_size=fft_size,
            num_frames=num_frames,
            dwell_s=dwell,
            reason=reason or f"cell {cell.cell_id} coarse",
            target_cell_ids=[cell.cell_id],
        )

    def _make_group_fine_cmd(
        self,
        group: "ActivityGroup",
        cfg: "EngineConfig",
    ) -> MeasurementCommand:
        from .spectrum_grid import CellState
        fft_size, num_frames = _fft_and_frames_for_dwell(cfg.scheduler.fine_dwell_s, cfg, fine=True)
        tt = TargetType.ACTIVE_GROUP_FINE_SCAN
        if any(c.state in (CellState.SUSPECT,) for c in group.cells):
            tt = TargetType.SUSPECT_CELL_FINE_SCAN
        return MeasurementCommand(
            target_type=tt,
            center_hz=group.center_hz,
            bandwidth_hz=group.bandwidth_hz or cfg.hardware.default_bandwidth_hz,
            fft_size=fft_size,
            num_frames=num_frames,
            dwell_s=cfg.scheduler.fine_dwell_s,
            reason=f"group p={group.group_probability:.2f}",
            target_cell_ids=[c.cell_id for c in group.cells],
        )

    def _dummy_cmd(self, cfg: "EngineConfig") -> MeasurementCommand:
        return MeasurementCommand(
            target_type=TargetType.BACKGROUND_CELL_COARSE_SCAN,
            center_hz=cfg.frequency_range.start_hz,
            bandwidth_hz=cfg.hardware.default_bandwidth_hz,
            fft_size=cfg.coarse_scan.fft_size,
            num_frames=cfg.coarse_scan.num_frames,
            dwell_s=cfg.coarse_scan.dwell_s,
            reason="no eligible cells",
        )
