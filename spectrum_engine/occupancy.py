"""
Occupancy probability model and activity-group detection.

Implements:
  - Log-odds occupancy update per cell (spec section 5.2)
  - Cell-state transitions (spec section 6.1)
  - Adjacent-cell grouping into ActivityGroup objects (spec section 5.3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .spectrum_grid import SpectrumCell
    from .config import EngineConfig

_LOG_ODDS_MIN: float = math.log(0.001 / 0.999)
_LOG_ODDS_MAX: float = math.log(0.999 / 0.001)


# ---------------------------------------------------------------------------
# Log-odds helpers
# ---------------------------------------------------------------------------

def _logodds(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def _from_logodds(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))


def _clamp_logodds(lo: float) -> float:
    return max(_LOG_ODDS_MIN, min(_LOG_ODDS_MAX, lo))


# ---------------------------------------------------------------------------
# Occupancy update
# ---------------------------------------------------------------------------

def update_occupancy_probability(
    cell: "SpectrumCell",
    occupied_bin_fraction: float,
    strong_bin_count: int,
    occupied_bin_threshold: float = 0.10,
) -> None:
    """
    Update cell.occupancy_prob via log-odds from baseline-relative evidence.

    Both inputs are measured against the cell's *tracked baseline* (a slow,
    signal-resistant noise-floor EWMA), NOT the current capture's own
    percentile. This is what lets quiet bands stay dark instead of saturating.

    Why two metrics:
      * A single FFT of pure noise still has ~10% of bins a bit above the noise
        floor and one bin ~15 dB above it, so naive "fraction over +8 dB" or
        "peak SNR" tests fire on every cell. We therefore measure occupancy with
        a generous margin (so noise fraction ~0) and treat narrowband signals
        via a *count of strong, well-above-noise bins* — noise produces lone
        spikes, real carriers light several adjacent bins.

    Parameters
    ----------
    occupied_bin_fraction : Fraction of bins exceeding (baseline + wide margin).
        ~0 for noise, large for a wideband signal filling the cell.
    strong_bin_count : Number of bins exceeding (baseline + strong margin).
        ~0 for noise (and ~1 for an isolated noise spike); >=3 indicates a real
        narrowband carrier.
    occupied_bin_threshold : Fraction giving moderate positive evidence.
    """
    evidence = 0.0

    # Wideband occupancy. Rise fast so an obvious signal lights the band on the
    # first hit or two (a single capture is a snapshot of a bursty signal).
    if occupied_bin_fraction > 0.30:
        evidence += 1.5
    elif occupied_bin_fraction > occupied_bin_threshold:
        evidence += 0.7
    elif occupied_bin_fraction > 0.05:
        evidence += 0.15

    # Narrowband carrier: several strong bins (not a lone noise spike).
    if strong_bin_count >= 6:
        evidence += 1.5
    elif strong_bin_count >= 3:
        evidence += 0.7

    # Decay toward "quiet" gently when nothing looks like a signal, so a band
    # with intermittent bursts stays visibly warm between hits (it IS active),
    # while a truly dead band still fades to dark over a number of revisits.
    if occupied_bin_fraction <= 0.05 and strong_bin_count < 3:
        evidence -= 0.35

    cell.occupancy_logodds = _clamp_logodds(cell.occupancy_logodds + evidence)
    cell.occupancy_prob = _from_logodds(cell.occupancy_logodds)


def decay_occupancy(cell: "SpectrumCell", decay_per_second: float, elapsed_s: float) -> None:
    """
    Gently decay occupancy probability towards 0.01 when the cell has not
    been scanned recently. Called once per scheduler cycle for stale cells.
    """
    total_decay = decay_per_second * elapsed_s
    cell.occupancy_logodds = _clamp_logodds(cell.occupancy_logodds - total_decay)
    cell.occupancy_prob = _from_logodds(cell.occupancy_logodds)


# ---------------------------------------------------------------------------
# Cell-state machine
# ---------------------------------------------------------------------------

def update_cell_state(cell: "SpectrumCell", cfg: "EngineConfig", now: float) -> None:
    """
    Apply state transition rules based on updated occupancy_prob and age.
    Does NOT touch cells that are UNSUPPORTED.
    """
    from .spectrum_grid import CellState

    if cell.state == CellState.UNSUPPORTED:
        return

    p = cell.occupancy_prob
    st = cfg.states

    age_s = now - cell.last_scan_time if cell.last_scan_time > 0 else 9999.0

    # STALE: overdue for scan (supersedes other transitions)
    if age_s > cell.max_revisit_interval_s and cell.state not in (CellState.STALE, CellState.UNKNOWN):
        cell.state = CellState.STALE
        return

    if cell.state == CellState.STALE:
        # After being scanned, fall through to normal logic below
        # (last_scan_time will have been updated)
        if age_s <= cell.max_revisit_interval_s:
            cell.state = CellState.UNKNOWN  # re-evaluate on next tick

    if cell.state in (CellState.UNKNOWN, CellState.STALE):
        if p > st.suspect_prob_threshold:
            cell.state = CellState.SUSPECT
        elif p <= st.quiet_prob_threshold:
            cell.state = CellState.QUIET
        else:
            cell.state = CellState.QUIET
        return

    if cell.state == CellState.QUIET:
        if p > st.suspect_prob_threshold:
            cell.state = CellState.SUSPECT
        return

    if cell.state == CellState.SUSPECT:
        if p > st.active_prob_threshold:
            cell.state = CellState.ACTIVE
        elif p < st.quiet_prob_threshold:
            cell.state = CellState.QUIET
        return

    if cell.state == CellState.ACTIVE:
        if p < st.quiet_prob_threshold:
            cell.state = CellState.QUIET
        # TRACKED transition is managed by the track manager
        return

    if cell.state == CellState.TRACKED:
        # Track manager controls return to ACTIVE; we only push back to QUIET
        if p < st.quiet_prob_threshold:
            cell.state = CellState.QUIET
        return


# ---------------------------------------------------------------------------
# ActivityGroup
# ---------------------------------------------------------------------------

@dataclass
class ActivityGroup:
    """Represents a contiguous run of adjacent active/suspect cells."""
    start_hz: float
    stop_hz: float
    center_hz: float
    bandwidth_hz: float
    cells: List["SpectrumCell"] = field(default_factory=list)
    group_probability: float = 0.0
    peak_db: float = -140.0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0

    def update_aggregate(self) -> None:
        """Recompute aggregate statistics from member cells."""
        if not self.cells:
            return
        self.start_hz = self.cells[0].start_hz
        self.stop_hz = self.cells[-1].stop_hz
        self.center_hz = (self.start_hz + self.stop_hz) / 2.0
        self.bandwidth_hz = self.stop_hz - self.start_hz

        # Joint probability: 1 - product(1 - p_i)
        joint_inactive = 1.0
        for c in self.cells:
            joint_inactive *= (1.0 - c.occupancy_prob)
        self.group_probability = 1.0 - joint_inactive

        self.peak_db = max(c.peak_db for c in self.cells)


def build_activity_groups(
    cells: List["SpectrumCell"],
    prob_threshold: float = 0.35,
    now: float = 0.0,
) -> List[ActivityGroup]:
    """
    Merge adjacent cells above prob_threshold into ActivityGroup objects.

    Cells are assumed ordered by frequency (which the grid builder guarantees).
    Returns groups sorted by group_probability descending.
    """
    from .spectrum_grid import CellState

    active_cells = [
        c for c in cells
        if c.occupancy_prob > prob_threshold
        and c.state not in (CellState.UNSUPPORTED, CellState.QUIET, CellState.UNKNOWN)
    ]

    if not active_cells:
        return []

    groups: List[ActivityGroup] = []
    current_run: List["SpectrumCell"] = [active_cells[0]]

    for cell in active_cells[1:]:
        prev = current_run[-1]
        # Adjacent if cells touch (stop_hz of prev == start_hz of current, ±1%)
        gap = cell.start_hz - prev.stop_hz
        if gap < 0.01 * prev.width_hz:
            current_run.append(cell)
        else:
            g = ActivityGroup(
                start_hz=current_run[0].start_hz,
                stop_hz=current_run[-1].stop_hz,
                center_hz=0.0,
                bandwidth_hz=0.0,
                cells=list(current_run),
                first_seen_time=now,
                last_seen_time=now,
            )
            g.update_aggregate()
            groups.append(g)
            current_run = [cell]

    if current_run:
        g = ActivityGroup(
            start_hz=current_run[0].start_hz,
            stop_hz=current_run[-1].stop_hz,
            center_hz=0.0,
            bandwidth_hz=0.0,
            cells=list(current_run),
            first_seen_time=now,
            last_seen_time=now,
        )
        g.update_aggregate()
        groups.append(g)

    groups.sort(key=lambda g: g.group_probability, reverse=True)
    return groups
