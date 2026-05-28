"""
Frequency cell grid for the Adaptive Spectrum Sensing Engine.

Divides the monitored band (default 1.2-6.0 GHz) into fixed-width cells.
Each cell is a persistent stateful object updated after every measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EngineConfig


# ---------------------------------------------------------------------------
# Cell states
# ---------------------------------------------------------------------------

class CellState:
    UNKNOWN = "UNKNOWN"
    QUIET = "QUIET"
    SUSPECT = "SUSPECT"
    ACTIVE = "ACTIVE"
    TRACKED = "TRACKED"
    STALE = "STALE"
    UNSUPPORTED = "UNSUPPORTED"


# ---------------------------------------------------------------------------
# SpectrumCell
# ---------------------------------------------------------------------------

@dataclass
class SpectrumCell:
    cell_id: int

    start_hz: float
    stop_hz: float
    center_hz: float
    width_hz: float

    # Measurement state
    last_scan_time: float = 0.0
    last_coarse_scan_time: float = 0.0
    last_fine_scan_time: float = 0.0

    # Signal statistics (filled by detectors + baseline)
    noise_floor_db: float = -100.0
    noise_var_db: float = 5.0
    peak_db: float = -140.0
    band_power_db: float = -140.0
    occupied_bw_hz: float = 0.0

    # Baseline / change detection
    baseline_db: float = -100.0
    baseline_var_db: float = 5.0
    energy_delta_db: float = 0.0

    # Detection probabilities / confidence
    occupancy_prob: float = 0.01
    occupancy_logodds: float = math.log(0.01 / 0.99)  # log-odds form
    uncertainty: float = 1.0

    # Last-measurement detection internals (for debugging / logging)
    last_occ_frac: float = 0.0
    last_strong_bins: int = 0
    last_peak_excess_db: float = 0.0

    # Scheduling
    priority_weight: float = 1.0
    priority_band_name: str = ""
    scan_score: float = 0.0
    min_revisit_interval_s: float = 0.1
    max_revisit_interval_s: float = 5.0
    preferred_dwell_s: float = 0.002

    # State labels
    state: str = CellState.UNKNOWN
    active_track_id: Optional[int] = None

    # History for coarse-suspicious flag
    coarse_suspicious: bool = False


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(cfg: "EngineConfig") -> List[SpectrumCell]:
    """
    Create the full array of SpectrumCell objects spanning start_hz..stop_hz.

    Cell width equals base_cell_bw_hz. Priority band weights and max-revisit
    intervals are applied, and any cell whose center is above hardware_max_hz
    is marked UNSUPPORTED.
    """
    fr = cfg.frequency_range
    sc = cfg.scheduler

    start = fr.start_hz
    stop = fr.stop_hz
    width = fr.base_cell_bw_hz

    n_cells = max(1, int(round((stop - start) / width)))
    cells: List[SpectrumCell] = []

    for i in range(n_cells):
        cell_start = start + i * width
        cell_stop = cell_start + width
        center = cell_start + width / 2.0

        cell = SpectrumCell(
            cell_id=i,
            start_hz=cell_start,
            stop_hz=cell_stop,
            center_hz=center,
            width_hz=width,
            max_revisit_interval_s=sc.default_max_revisit_s,
            preferred_dwell_s=sc.coarse_dwell_s,
        )

        # Apply priority band settings (first matching band wins)
        for pb in cfg.priority_bands:
            if pb.start_hz <= center <= pb.stop_hz:
                cell.priority_weight = pb.priority_weight
                cell.priority_band_name = pb.name
                cell.max_revisit_interval_s = pb.max_revisit_s
                break

        cells.append(cell)

    return cells


def mark_unsupported_cells(cells: List[SpectrumCell], hw_max_hz: float) -> None:
    """Mark cells whose center exceeds the hardware's max tuning frequency."""
    for cell in cells:
        if cell.center_hz > hw_max_hz:
            cell.state = CellState.UNSUPPORTED


# ---------------------------------------------------------------------------
# Measurement → cell mapping
# ---------------------------------------------------------------------------

def map_measurement_to_cells(
    center_hz: float,
    bandwidth_hz: float,
    cells: List[SpectrumCell],
) -> List[SpectrumCell]:
    """
    Return cells whose frequency range overlaps the measured window.

    The measured window is [center_hz - bandwidth_hz/2, center_hz + bandwidth_hz/2].
    Cells that are UNSUPPORTED are excluded.
    """
    lo = center_hz - bandwidth_hz / 2.0
    hi = center_hz + bandwidth_hz / 2.0

    result = []
    for cell in cells:
        if cell.state == CellState.UNSUPPORTED:
            continue
        if cell.stop_hz > lo and cell.start_hz < hi:
            result.append(cell)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cells_to_occupancy_array(cells: List[SpectrumCell]):
    """Return a numpy array of occupancy probabilities aligned to cell order."""
    import numpy as np
    return np.array([c.occupancy_prob for c in cells], dtype=np.float32)


def cells_to_freq_centers(cells: List[SpectrumCell]):
    """Return a numpy array of cell center frequencies in Hz."""
    import numpy as np
    return np.array([c.center_hz for c in cells], dtype=np.float64)
