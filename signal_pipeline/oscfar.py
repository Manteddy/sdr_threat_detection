"""
OSCFARProcessor — Ordered-Statistic CFAR + heuristic classifier.

Implements the five-stage Signal Processing Pipeline from the Project
Definition (§8) as a plug-in next to ClassicProcessor:

    Stage 1: Hamming-windowed integrated PSD  (uses the engine's Welch
             PSD directly — equivalent integrated estimate)
    Stage 2: OS-CFAR sliding window           (75th percentile + alpha)
    Stage 3: Per-ROI features                 (bandwidth, Shannon entropy,
                                              sync detection at 15.625 kHz)
    Stage 4: Heuristic classifier             (ANALOG_FPV / BARRAGE_JAMMER
                                              / TELEMETRY_LINK / UNKNOWN)
    Stage 5: ClassificationResult per ROI

Each stage is a pure function so the pieces can be unit-tested
independently (in addition to the end-to-end smoke against the simulator).

Constants live in the module for now; the matching `signal_pipeline:`
config block in engine_config.yaml is a follow-up — see the integration
plan §4. Override via constructor kwargs for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from .base import ClassificationResult, SignalProcessor

if TYPE_CHECKING:
    from spectrum_engine.config import EngineConfig
    from spectrum_engine.detectors import FineMeasurement
    from spectrum_engine.iq_source import IQCapture


# ---------------------------------------------------------------------------
# Defaults — match the spec; overridable per instance.
# ---------------------------------------------------------------------------

_DEFAULT_GUARD_CELLS: int = 4
_DEFAULT_REFERENCE_CELLS: int = 16   # 8 / side
_DEFAULT_K_PERCENTILE: float = 75.0
_DEFAULT_ALPHA_DB: float = 6.0
_DEFAULT_MIN_ROI_BW_HZ: float = 30_000.0
_DEFAULT_MERGE_GAP_HZ: float = 1_000_000.0

# Global noise-floor fallback. Textbook OS-CFAR with a narrow reference
# window (16 cells = ~78 kHz at fine config) sits entirely *inside* a
# wideband emitter (8 MHz FPV), so its threshold tracks the signal level
# and never triggers in the middle of the band. Combining with a global
# percentile noise floor catches those wideband cases without breaking
# OS-CFAR's narrowband sensitivity: the effective threshold is the
# minimum of the two estimates.
_GLOBAL_NOISE_PERCENTILE: float = 30.0
_GLOBAL_ALPHA_DB: float = 6.0

# Sync probing
_SYNC_H_HZ: float = 15625.0
_SYNC_H_TOL_HZ: float = 2_000.0
# Threshold for "horizontal sync present". Random Gaussian noise has
# envelope-power peaks reaching ~10 dB above the median by chance alone
# in a narrow band of frequencies, so 8 dB triggers on every jammer.
# 20 dB is comfortably above that variance and well below the real FPV
# synth's 40-50 dB peak.
_SYNC_PEAK_THRESHOLD_DB: float = 20.0

# Classifier thresholds
_FPV_BW_HZ_MIN: float = 6_000_000.0
_FPV_BW_HZ_MAX: float = 10_000_000.0
_JAMMER_BW_HZ_MIN: float = 12_000_000.0
_TELEMETRY_BW_HZ_MAX: float = 1_000_000.0
_ENTROPY_FPV_MIN_BITS: float = 3.0
_ENTROPY_FPV_MAX_BITS: float = 6.0
_ENTROPY_JAMMER_MIN_BITS: float = 6.5


# ---------------------------------------------------------------------------
# Stage helpers — pure functions
# ---------------------------------------------------------------------------

def os_cfar_threshold(
    psd_db: np.ndarray,
    *,
    guard_cells: int,
    reference_cells: int,
    k_percentile: float,
    alpha_db: float,
) -> np.ndarray:
    """
    Vectorised Ordered-Statistic CFAR threshold per PSD bin.

    Window layout per CUT (size 2*guard + reference + 1):
        [ref_left ... guard_left  CUT  guard_right ... ref_right]
    The k-th percentile is taken over the reference cells only; guards
    are ignored to prevent target leakage into the noise estimate.

    Edge bins where a full window does not fit get a very high threshold
    so they never trigger — same convention as the engine's CA-CFAR.
    """
    n = len(psd_db)
    ref = int(reference_cells)
    guard = int(guard_cells)
    half_ref = ref // 2
    window = 2 * guard + ref + 1

    out = np.full(n, psd_db.max() + 20.0, dtype=np.float32)

    if n < window:
        return out  # too short to run CFAR at all

    # All sliding windows: shape (n - window + 1, window)
    sw = np.lib.stride_tricks.sliding_window_view(psd_db, window_shape=window)

    # Reference cell indices within each window:
    #   left ref:  [0, half_ref)
    #   right ref: [half_ref + 2*guard + 1, window)
    left_idx = np.arange(0, half_ref)
    right_idx = np.arange(half_ref + 2 * guard + 1, window)
    ref_idx = np.concatenate([left_idx, right_idx])

    ref_vals = sw[:, ref_idx]                              # (n_cuts, ref)
    # k-th percentile across the reference window. np.partition gives the
    # k-th order statistic in O(N) per row.
    k_idx = int(np.clip(round((k_percentile / 100.0) * (ref - 1)), 0, ref - 1))
    partitioned = np.partition(ref_vals, k_idx, axis=-1)
    p_k = partitioned[:, k_idx]

    # CUT index inside each window is half_ref + guard
    cut_pos = half_ref + guard
    out_start = cut_pos
    out_stop = cut_pos + sw.shape[0]
    out[out_start:out_stop] = (p_k + float(alpha_db)).astype(np.float32)
    return out


def cluster_rois(
    psd_db: np.ndarray,
    threshold_db: np.ndarray,
    freq_axis_hz: np.ndarray,
    *,
    min_roi_bw_hz: float,
    merge_gap_hz: float,
) -> List[Tuple[int, int]]:
    """Find contiguous above-threshold runs → ROIs; drop tiny ones; merge near ones."""
    above = psd_db > threshold_db
    runs: List[Tuple[int, int]] = []
    in_run = False
    s = 0
    for i, v in enumerate(above):
        if v and not in_run:
            in_run = True
            s = i
        elif not v and in_run:
            in_run = False
            runs.append((s, i - 1))
    if in_run:
        runs.append((s, len(above) - 1))

    if not runs:
        return runs

    bin_width = (freq_axis_hz[-1] - freq_axis_hz[0]) / max(len(freq_axis_hz) - 1, 1)
    min_bins = max(1, int(min_roi_bw_hz / bin_width))
    runs = [(s, e) for s, e in runs if (e - s + 1) >= min_bins]
    if not runs:
        return runs

    gap_bins = max(1, int(merge_gap_hz / bin_width))
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_bins:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def shannon_entropy_bits(psd_db: np.ndarray) -> float:
    """Shannon entropy of a PSD slice converted to normalised linear power."""
    lin = np.power(10.0, psd_db.astype(np.float64) / 10.0)
    total = float(lin.sum())
    if total <= 0.0:
        return 0.0
    p = lin / total
    p = np.clip(p, 1e-30, 1.0)
    return float(-np.sum(p * np.log2(p)))


def band_limit_iq(
    iq: np.ndarray,
    fs: float,
    center_offset_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Zero out FFT bins outside [center - bw/2, center + bw/2] → band-limited IQ."""
    n = len(iq)
    spec = np.fft.fft(iq)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    lo = center_offset_hz - bandwidth_hz / 2.0
    hi = center_offset_hz + bandwidth_hz / 2.0
    mask = (freqs >= lo) & (freqs <= hi)
    spec_out = np.zeros_like(spec)
    spec_out[mask] = spec[mask]
    return np.fft.ifft(spec_out).astype(np.complex64)


def envelope_power_peak_db(
    iq_band: np.ndarray,
    fs: float,
    target_hz: float,
    tolerance_hz: float,
) -> float:
    """
    Find the strongest spectral line in |iq_band|² near `target_hz`,
    return its level in dB relative to the median of the envelope spectrum.

    Returns -inf when the buffer is too short to resolve `target_hz`.
    """
    env = np.abs(iq_band).astype(np.float64) ** 2
    env = env - env.mean()    # remove DC so the bin-0 spike doesn't dominate
    n = len(env)
    if n < 16:
        return float("-inf")
    spec = np.abs(np.fft.fft(env))[: n // 2]
    freqs = np.fft.fftfreq(n, d=1.0 / fs)[: n // 2]
    in_band = (freqs >= target_hz - tolerance_hz) & (freqs <= target_hz + tolerance_hz)
    if not np.any(in_band):
        return float("-inf")
    peak = float(spec[in_band].max())
    median = float(np.median(spec)) + 1e-30
    return 20.0 * np.log10(peak / median)


# ---------------------------------------------------------------------------
# Feature container + classifier
# ---------------------------------------------------------------------------

@dataclass
class _ROIFeatures:
    start_bin: int
    stop_bin: int
    start_hz: float
    stop_hz: float
    center_hz: float
    bandwidth_hz: float
    peak_db: float
    noise_floor_db: float
    snr_like_db: float
    entropy_bits: float
    sync_h_peak_db: float


def _classify(features: _ROIFeatures) -> Tuple[str, float, dict]:
    """Heuristic decision matrix → (label, confidence, feature snapshot).

    Sync-pulse detection is the strongest single discriminator empirically
    — FM video has a clean cyclic envelope feature at 15.625 kHz that
    band-limited noise (jammer) cannot produce. Entropy is kept in the
    feature snapshot for telemetry / tuning but not used in the v1
    decision matrix: real FM video occupies many PSD bins relatively
    uniformly, giving it near-jammer entropy even though the signals are
    obviously distinguishable on the sync feature.
    """
    bw = features.bandwidth_hz
    H = features.entropy_bits
    sync = features.sync_h_peak_db

    snapshot = {
        "bandwidth_hz": bw,
        "entropy_bits": H,
        "sync_h_peak_db": sync,
        "peak_db": features.peak_db,
        "snr_like_db": features.snr_like_db,
    }

    sync_present = sync > _SYNC_PEAK_THRESHOLD_DB

    # TELEMETRY_LINK first — narrowest band wins outright.
    if bw < _TELEMETRY_BW_HZ_MAX:
        return "TELEMETRY_LINK", 0.85, snapshot

    # ANALOG_FPV: typical analog FM video band AND horizontal sync present.
    fpv_bw_ok = _FPV_BW_HZ_MIN <= bw <= _FPV_BW_HZ_MAX
    if fpv_bw_ok and sync_present:
        return "ANALOG_FPV", 0.9, snapshot

    # BARRAGE_JAMMER: very wide AND no sync.
    if bw > _JAMMER_BW_HZ_MIN and not sync_present:
        return "BARRAGE_JAMMER", 0.85, snapshot

    # Soft cases — emit best partial guess so the GUI sees *something*.
    if fpv_bw_ok and not sync_present:
        return "UNKNOWN", 0.4, snapshot
    if bw > _JAMMER_BW_HZ_MIN and sync_present:
        # Wide AND syncing — overlap of FPV + something; lean FPV.
        return "ANALOG_FPV", 0.5, snapshot
    return "UNKNOWN", 0.3, snapshot


# ---------------------------------------------------------------------------
# OSCFARProcessor
# ---------------------------------------------------------------------------

class OSCFARProcessor(SignalProcessor):
    name = "oscfar"
    label = "OS-CFAR + Classifier"

    def __init__(
        self,
        *,
        guard_cells: int = _DEFAULT_GUARD_CELLS,
        reference_cells: int = _DEFAULT_REFERENCE_CELLS,
        k_percentile: float = _DEFAULT_K_PERCENTILE,
        alpha_db: float = _DEFAULT_ALPHA_DB,
        min_roi_bw_hz: float = _DEFAULT_MIN_ROI_BW_HZ,
        merge_gap_hz: float = _DEFAULT_MERGE_GAP_HZ,
    ) -> None:
        self.guard_cells = guard_cells
        self.reference_cells = reference_cells
        self.k_percentile = k_percentile
        self.alpha_db = alpha_db
        self.min_roi_bw_hz = min_roi_bw_hz
        self.merge_gap_hz = merge_gap_hz

    def process_fine(
        self,
        capture: "IQCapture",
        psd_db: np.ndarray,
        freq_axis_hz: np.ndarray,
        cfg: "EngineConfig",
    ) -> "FineMeasurement":
        from spectrum_engine.detectors import DetectedRegion, FineMeasurement  # lazy

        psd_db = np.asarray(psd_db, dtype=np.float32)
        f_ax = np.asarray(freq_axis_hz, dtype=np.float64)

        # ---- Stage 2: OS-CFAR + global wideband fallback ----
        thresh = os_cfar_threshold(
            psd_db,
            guard_cells=self.guard_cells,
            reference_cells=self.reference_cells,
            k_percentile=self.k_percentile,
            alpha_db=self.alpha_db,
        )
        noise_floor_db = float(np.percentile(psd_db, _GLOBAL_NOISE_PERCENTILE))
        global_thresh = np.float32(noise_floor_db + _GLOBAL_ALPHA_DB)
        # Effective threshold = min(per-bin OS-CFAR, global noise floor).
        # OS-CFAR keeps narrowband sensitivity; global catches wideband
        # emitters whose own energy contaminates the OS-CFAR reference.
        thresh = np.minimum(thresh, global_thresh)

        rois = cluster_rois(
            psd_db, thresh, f_ax,
            min_roi_bw_hz=self.min_roi_bw_hz,
            merge_gap_hz=self.merge_gap_hz,
        )

        # Empty result — Stage 2 early-exit path.
        if not rois:
            return FineMeasurement(
                center_hz=float(capture.center_hz),
                bandwidth_hz=float(capture.bandwidth_hz),
                timestamp=capture.timestamp,
                psd_db=psd_db,
                freq_axis_hz=f_ax,
                cfar_threshold_db=thresh,
                detected_regions=[],
                occupied_bw_hz=0.0,
                confidence=0.0,
                noise_floor_db=noise_floor_db,
                classifications=[],
            )

        # IQ time series for sync detection (concatenated frames)
        iq_flat = np.asarray(capture.omni_iq, dtype=np.complex64).ravel()
        fs = float(cfg.hardware.sample_rate_hz)
        capture_center_hz = float(capture.center_hz)

        bin_width = (f_ax[-1] - f_ax[0]) / max(len(f_ax) - 1, 1)

        detected_regions: List[DetectedRegion] = []
        classifications: List[ClassificationResult] = []

        for s_bin, e_bin in rois:
            r_start = float(f_ax[s_bin])
            r_stop = float(f_ax[e_bin])
            r_center = (r_start + r_stop) / 2.0
            r_bw = r_stop - r_start + bin_width
            roi_slice = psd_db[s_bin: e_bin + 1]
            peak = float(np.max(roi_slice))
            snr_like = peak - noise_floor_db

            # ---- Stage 3: features ----
            entropy = shannon_entropy_bits(roi_slice)
            # Band-limit IQ to the ROI for envelope-power probing
            offset_hz = r_center - capture_center_hz
            try:
                iq_band = band_limit_iq(iq_flat, fs, offset_hz, r_bw)
                sync_h = envelope_power_peak_db(
                    iq_band, fs, _SYNC_H_HZ, _SYNC_H_TOL_HZ,
                )
            except Exception:
                sync_h = float("-inf")

            features = _ROIFeatures(
                start_bin=s_bin,
                stop_bin=e_bin,
                start_hz=r_start,
                stop_hz=r_stop,
                center_hz=r_center,
                bandwidth_hz=r_bw,
                peak_db=peak,
                noise_floor_db=noise_floor_db,
                snr_like_db=snr_like,
                entropy_bits=entropy,
                sync_h_peak_db=sync_h,
            )

            # ---- Stage 4: classify ----
            label, conf, snapshot = _classify(features)

            # DetectedRegion for the engine track manager
            detected_regions.append(DetectedRegion(
                start_hz=r_start,
                stop_hz=r_stop,
                center_hz=r_center,
                bandwidth_hz=r_bw,
                peak_db=peak,
                noise_floor_db=noise_floor_db,
                snr_like_db=snr_like,
                confidence=conf,
            ))

            # ---- Stage 5: ClassificationResult ----
            classifications.append(ClassificationResult(
                frequency_hz=r_center,
                bandwidth_hz=r_bw,
                signal_strength_db=peak,
                classification=label,
                confidence_score=conf,
                features=snapshot,
            ))

        return FineMeasurement(
            center_hz=float(capture.center_hz),
            bandwidth_hz=float(capture.bandwidth_hz),
            timestamp=capture.timestamp,
            psd_db=psd_db,
            freq_axis_hz=f_ax,
            cfar_threshold_db=thresh,
            detected_regions=detected_regions,
            occupied_bw_hz=sum(r.bandwidth_hz for r in detected_regions),
            confidence=max((r.confidence for r in detected_regions), default=0.0),
            noise_floor_db=noise_floor_db,
            classifications=classifications,
        )
