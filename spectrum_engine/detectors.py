"""
Coarse energy detector and fine CA-CFAR region detector.

Implements:
  - CoarseMeasurement: fast noise/peak/occupied-bin statistics
  - FineMeasurement: full CFAR region detection with confidence scoring
  - DetectedRegion: per-detected-signal descriptor
  - Region merging and persistence filtering

Spec sections 4, 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .config import EngineConfig


# ---------------------------------------------------------------------------
# Measurement result containers (spec section 4)
# ---------------------------------------------------------------------------

@dataclass
class CoarseMeasurement:
    center_hz: float
    bandwidth_hz: float
    timestamp: float
    psd_db: np.ndarray          # shape (fft_size,)
    freq_axis_hz: np.ndarray    # shape (fft_size,)
    noise_floor_db: float
    peak_db: float
    band_power_db: float
    occupied_bins: int
    occupied_bw_hz: float
    occupied_bin_fraction: float
    spectral_flatness: float
    energy_delta_db: float = 0.0  # filled by caller from cell baseline
    coarse_suspicious: bool = False


@dataclass
class DetectedRegion:
    start_hz: float
    stop_hz: float
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    noise_floor_db: float
    snr_like_db: float
    confidence: float
    persistence_count: int = 0


@dataclass
class FineMeasurement:
    center_hz: float
    bandwidth_hz: float
    timestamp: float
    psd_db: np.ndarray
    freq_axis_hz: np.ndarray
    cfar_threshold_db: np.ndarray
    detected_regions: List[DetectedRegion] = field(default_factory=list)
    occupied_bw_hz: float = 0.0
    confidence: float = 0.0
    noise_floor_db: float = -100.0
    # Populated by classifying processors (e.g. OSCFARProcessor). Typed
    # `List[Any]` to avoid a circular import on signal_pipeline; the
    # consumer (GUI / AlertEngine) treats entries via duck typing on the
    # ClassificationResult shape defined in signal_pipeline.base.
    classifications: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coarse detector (spec section 9.1)
# ---------------------------------------------------------------------------

def compute_coarse_measurement(
    psd_db: np.ndarray,
    freq_axis: np.ndarray,
    timestamp: float,
    center_hz: float,
    bandwidth_hz: float,
    baseline_db: float,
    cfg: EngineConfig,
) -> CoarseMeasurement:
    """
    Derive fast statistics from a coarse PSD (spec section 9.1).

    The PSD is already in dBFS; we compute percentile noise floor, peak,
    occupied bins, and band power.
    """
    n = len(psd_db)
    bin_width_hz = bandwidth_hz / max(n, 1)

    noise_floor_db = float(np.percentile(psd_db, 30))
    peak_db = float(np.max(psd_db))

    # Band power in dB
    lin = 10.0 ** (psd_db / 10.0)
    band_power_db = float(10.0 * np.log10(np.mean(lin) + 1e-30))

    # Occupied bins: > noise_floor + 6 dB
    threshold_6db = noise_floor_db + 6.0
    occ_mask = psd_db > threshold_6db
    occupied_bins = int(np.sum(occ_mask))
    occupied_bw_hz = occupied_bins * bin_width_hz
    occupied_bin_fraction = occupied_bins / max(n, 1)

    # Spectral flatness (geometric / arithmetic mean ratio in linear)
    geo_mean = float(np.exp(np.mean(np.log(lin + 1e-30))))
    arith_mean = float(np.mean(lin) + 1e-30)
    spectral_flatness = geo_mean / arith_mean

    energy_delta_db = peak_db - baseline_db

    # Suspicion rule (spec section 9.1)
    occ_thresh = cfg.states.occupied_bin_fraction_threshold
    suspicious = (
        energy_delta_db > 8.0
        or (peak_db - noise_floor_db) > 10.0
        or occupied_bin_fraction > occ_thresh
    )

    return CoarseMeasurement(
        center_hz=center_hz,
        bandwidth_hz=bandwidth_hz,
        timestamp=timestamp,
        psd_db=psd_db,
        freq_axis_hz=freq_axis,
        noise_floor_db=noise_floor_db,
        peak_db=peak_db,
        band_power_db=band_power_db,
        occupied_bins=occupied_bins,
        occupied_bw_hz=occupied_bw_hz,
        occupied_bin_fraction=occupied_bin_fraction,
        spectral_flatness=spectral_flatness,
        energy_delta_db=energy_delta_db,
        coarse_suspicious=suspicious,
    )


# ---------------------------------------------------------------------------
# CA-CFAR (spec section 9.2)
# ---------------------------------------------------------------------------

def cfar_threshold_1d(
    psd_db: np.ndarray,
    guard_cells: int,
    training_cells: int,
    threshold_offset_db: float,
) -> np.ndarray:
    """
    Cell-Averaging CFAR — return threshold array same shape as psd_db.

    Each bin's threshold = mean(training window) + threshold_offset_db.
    Bins near the edges use available training cells (asymmetric window).
    """
    n = len(psd_db)
    thresholds = np.full(n, psd_db.max() + 20.0, dtype=np.float32)

    for i in range(n):
        lo_guard = max(0, i - guard_cells)
        hi_guard = min(n, i + guard_cells + 1)
        lo_train = max(0, i - guard_cells - training_cells)
        hi_train = min(n, i + guard_cells + training_cells + 1)

        train_indices = list(range(lo_train, lo_guard)) + list(range(hi_guard, hi_train))
        if not train_indices:
            continue

        mean_train = float(np.mean(psd_db[train_indices]))
        thresholds[i] = mean_train + threshold_offset_db

    return thresholds


def cfar_detect_regions(
    psd_db: np.ndarray,
    cfar_thresh: np.ndarray,
    freq_axis: np.ndarray,
    min_region_bw_hz: float,
) -> List[Tuple[int, int]]:
    """
    Find contiguous runs of bins where psd_db > cfar_thresh.

    Returns list of (start_bin, stop_bin) index pairs (inclusive).
    """
    above = psd_db > cfar_thresh
    regions = []
    in_region = False
    start_bin = 0
    for i, val in enumerate(above):
        if val and not in_region:
            in_region = True
            start_bin = i
        elif not val and in_region:
            in_region = False
            regions.append((start_bin, i - 1))
    if in_region:
        regions.append((start_bin, len(above) - 1))

    bin_width = (freq_axis[-1] - freq_axis[0]) / max(len(freq_axis) - 1, 1)
    min_bins = max(1, int(min_region_bw_hz / bin_width))
    return [(s, e) for s, e in regions if (e - s + 1) >= min_bins]


def _merge_close_regions(
    regions: List[Tuple[int, int]],
    freq_axis: np.ndarray,
    gap_hz: float = 1e6,
) -> List[Tuple[int, int]]:
    """Merge region pairs whose gap is smaller than gap_hz."""
    if not regions:
        return []

    bin_width = (freq_axis[-1] - freq_axis[0]) / max(len(freq_axis) - 1, 1)
    gap_bins = max(1, int(gap_hz / bin_width))

    merged = [regions[0]]
    for s, e in regions[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_bins:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _region_confidence(
    psd_db: np.ndarray,
    cfar_thresh: np.ndarray,
    start_bin: int,
    stop_bin: int,
    noise_floor_db: float,
) -> Tuple[float, float, float]:
    """
    Return (peak_db, snr_like_db, confidence) for a detected region.

    Confidence formula (spec section 9.3 simplified initial version).
    """
    region = psd_db[start_bin: stop_bin + 1]
    peak_db = float(np.max(region))
    snr_like_db = peak_db - noise_floor_db

    cfar_passed = 1.0 if peak_db > float(np.max(cfar_thresh[start_bin: stop_bin + 1])) else 0.0

    confidence = min(
        0.04 * snr_like_db
        + 0.20 * cfar_passed
        + 0.10,   # base for being detected at all
        1.0,
    )
    confidence = max(confidence, 0.0)
    return peak_db, snr_like_db, confidence


# ---------------------------------------------------------------------------
# Fine detector (spec section 9.2)
# ---------------------------------------------------------------------------

def compute_fine_measurement(
    psd_db: np.ndarray,
    freq_axis: np.ndarray,
    timestamp: float,
    center_hz: float,
    bandwidth_hz: float,
    cfg: EngineConfig,
) -> FineMeasurement:
    """
    Run Welch PSD through CFAR and produce a FineMeasurement with detected regions.
    """
    cfar_cfg = cfg.cfar

    noise_floor_db = float(np.percentile(psd_db, 30))

    cfar_thresh = cfar_threshold_1d(
        psd_db,
        guard_cells=cfar_cfg.guard_cells,
        training_cells=cfar_cfg.training_cells,
        threshold_offset_db=cfar_cfg.threshold_offset_db,
    )

    raw_regions = cfar_detect_regions(
        psd_db, cfar_thresh, freq_axis, cfar_cfg.min_region_bw_hz,
    )
    raw_regions = _merge_close_regions(raw_regions, freq_axis, gap_hz=1e6)

    detected: List[DetectedRegion] = []
    bin_width = (freq_axis[-1] - freq_axis[0]) / max(len(freq_axis) - 1, 1)

    for s_bin, e_bin in raw_regions:
        peak_db, snr_like_db, confidence = _region_confidence(
            psd_db, cfar_thresh, s_bin, e_bin, noise_floor_db,
        )
        r_start = float(freq_axis[s_bin])
        r_stop = float(freq_axis[e_bin])
        r_center = (r_start + r_stop) / 2.0
        r_bw = r_stop - r_start + bin_width

        detected.append(DetectedRegion(
            start_hz=r_start,
            stop_hz=r_stop,
            center_hz=r_center,
            bandwidth_hz=r_bw,
            peak_db=peak_db,
            noise_floor_db=noise_floor_db,
            snr_like_db=snr_like_db,
            confidence=confidence,
        ))

    total_occ_bw = sum(r.bandwidth_hz for r in detected)
    overall_conf = max((r.confidence for r in detected), default=0.0)

    return FineMeasurement(
        center_hz=center_hz,
        bandwidth_hz=bandwidth_hz,
        timestamp=timestamp,
        psd_db=psd_db,
        freq_axis_hz=freq_axis,
        cfar_threshold_db=cfar_thresh,
        detected_regions=detected,
        occupied_bw_hz=total_occ_bw,
        confidence=overall_conf,
        noise_floor_db=noise_floor_db,
    )
