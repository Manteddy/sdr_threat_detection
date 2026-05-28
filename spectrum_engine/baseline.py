"""
Per-cell adaptive baseline and uncertainty tracking.

Each SpectrumCell maintains:
  - baseline_db         : slow EWMA of the noise floor
  - baseline_var_db     : slow EWMA of squared error (variance proxy)
  - uncertainty         : combined metric from age and baseline variance
  - energy_delta_db     : peak_db - baseline_db (how far above baseline)

The EWMA alpha is reduced during suspicious activity so a real signal
does not inflate the baseline too quickly (spec section 11).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spectrum_grid import SpectrumCell

# EWMA alpha values
_ALPHA_NORMAL: float = 0.02
_ALPHA_SUSPICIOUS: float = 0.002

# Margin above baseline considered "suspicious" for baseline update slowdown
_SUSPICIOUS_MARGIN_DB: float = 10.0

# Baseline initialisation flag uses a sentinel: baseline_db starts at -100,
# but we detect first-update by checking whether last_scan_time == 0.


def update_baseline(cell: "SpectrumCell", noise_floor_db: float) -> None:
    """
    Update cell.baseline_db and cell.baseline_var_db with one new sample.

    Uses a slow EWMA. When the cell is in a suspicious/active state the
    alpha is an order of magnitude smaller so persistent signals do not
    corrupt the baseline estimate.
    """
    current = noise_floor_db

    # Initialise on first measurement
    if cell.last_scan_time == 0.0:
        cell.baseline_db = current
        cell.baseline_var_db = 5.0
        return

    suspicious = (current > cell.baseline_db + _SUSPICIOUS_MARGIN_DB) or cell.coarse_suspicious
    alpha = _ALPHA_SUSPICIOUS if suspicious else _ALPHA_NORMAL

    err = current - cell.baseline_db
    cell.baseline_db = (1.0 - alpha) * cell.baseline_db + alpha * current
    cell.baseline_var_db = (1.0 - alpha) * cell.baseline_var_db + alpha * (err * err)


def update_energy_delta(cell: "SpectrumCell", peak_db: float) -> None:
    """Recompute energy_delta_db from the current peak and baseline."""
    cell.energy_delta_db = peak_db - cell.baseline_db


def update_uncertainty(cell: "SpectrumCell", now: float) -> None:
    """
    Recompute cell.uncertainty from scan age and baseline variance.

    uncertainty ∈ [0, 1]:
      0 = very well characterised and recently scanned
      1 = old data or high variance
    """
    if cell.max_revisit_interval_s <= 0:
        age_norm = 1.0
    else:
        age_s = now - cell.last_scan_time
        age_norm = min(age_s / cell.max_revisit_interval_s, 2.0)

    # Normalise variance: 5 dB² is the "typical" variance at start-up
    var_norm = min(cell.baseline_var_db / 25.0, 1.0)

    cell.uncertainty = min(0.5 * age_norm + 0.5 * var_norm, 1.0)
