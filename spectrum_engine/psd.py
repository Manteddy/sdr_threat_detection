"""
PSD computation module for the Adaptive Spectrum Sensing Engine.

Provides:
  - coarse_psd_db   — single windowed FFT per frame, averaged across frames
  - fine_welch_db   — Welch overlapped PSD for fine measurements
  - freq_axis_hz    — frequency axis for a given center/bandwidth/bins

Both coarse and fine paths accept (num_frames, fft_size) IQ arrays and
return a 1-D dBFS spectrum aligned to the hardware bandwidth.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftshift

# Pre-built window cache: key → (window_array, window_sum)
_WINDOW_CACHE: dict = {}


def _get_window(size: int, name: str = "blackman"):
    key = (size, name)
    if key not in _WINDOW_CACHE:
        if name == "blackman":
            w = np.blackman(size).astype(np.float32)
        elif name == "hann":
            w = np.hanning(size).astype(np.float32)
        elif name == "hamming":
            w = np.hamming(size).astype(np.float32)
        else:
            w = np.blackman(size).astype(np.float32)
        _WINDOW_CACHE[key] = (w, float(np.sum(w)))
    return _WINDOW_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def freq_axis_hz(
    center_hz: float,
    bandwidth_hz: float,
    n_bins: int,
) -> np.ndarray:
    """
    Return the frequency axis in Hz for a baseband FFT centred at center_hz.
    The axis covers [center - bw/2, center + bw/2).
    """
    lo = center_hz - bandwidth_hz / 2.0
    hi = center_hz + bandwidth_hz / 2.0
    return np.linspace(lo, hi, n_bins, endpoint=False, dtype=np.float64)


def coarse_psd_db(
    iq_frames: np.ndarray,
    fft_size: int | None = None,
    window_name: str = "blackman",
) -> np.ndarray:
    """
    Compute coarse PSD in dBFS by FFT-averaging across frames.

    Parameters
    ----------
    iq_frames : complex64 array, shape (num_frames, frame_len) or (frame_len,)
    fft_size  : FFT size; defaults to the frame length.
    window_name : Window function name.

    Returns
    -------
    psd_db : float32 array, shape (fft_size,) — averaged magnitude spectrum in dBFS.
    """
    if iq_frames.ndim == 1:
        iq_frames = iq_frames[np.newaxis, :]

    num_frames, frame_len = iq_frames.shape
    if fft_size is None:
        fft_size = frame_len

    window, win_sum = _get_window(fft_size, window_name)

    acc = np.zeros(fft_size, dtype=np.float64)
    for i in range(num_frames):
        frame = iq_frames[i, :fft_size].astype(np.complex64)
        if len(frame) < fft_size:
            frame = np.pad(frame, (0, fft_size - len(frame)))
        spec = fftshift(fft(frame * window))
        acc += np.abs(spec) ** 2

    acc /= num_frames
    acc = np.sqrt(acc)  # back to amplitude-like for log scaling

    psd = 20.0 * np.log10(acc / win_sum + 1e-20)
    return psd.astype(np.float32)


def fine_welch_db(
    iq_frames: np.ndarray,
    fft_size: int = 4096,
    overlap: float = 0.5,
    window_name: str = "blackman",
) -> np.ndarray:
    """
    Compute fine PSD via Welch's method (overlapped averaging).

    Concatenates all frames into one long segment, then divides into
    overlapping blocks of length fft_size and averages the periodograms.

    Parameters
    ----------
    iq_frames : complex64 array, shape (num_frames, frame_len) or (frame_len,)
    fft_size  : FFT size for each Welch block.
    overlap   : Fractional overlap between blocks (0..1).
    window_name : Window function name.

    Returns
    -------
    psd_db : float32 array, shape (fft_size,) in dBFS.
    """
    if iq_frames.ndim == 1:
        iq_frames = iq_frames[np.newaxis, :]

    # Flatten all frames into a single time series
    flat = iq_frames.ravel().astype(np.complex64)
    window, win_sum = _get_window(fft_size, window_name)

    step = max(1, int(fft_size * (1.0 - overlap)))
    n_total = len(flat)

    if n_total < fft_size:
        flat = np.pad(flat, (0, fft_size - n_total))
        n_total = fft_size

    starts = range(0, n_total - fft_size + 1, step)
    n_blocks = len(starts)

    if n_blocks == 0:
        n_blocks = 1
        starts = [0]

    acc = np.zeros(fft_size, dtype=np.float64)
    for s in starts:
        block = flat[s: s + fft_size]
        if len(block) < fft_size:
            block = np.pad(block, (0, fft_size - len(block)))
        spec = fftshift(fft(block * window))
        acc += np.abs(spec) ** 2

    acc /= n_blocks
    acc = np.sqrt(acc)

    psd = 20.0 * np.log10(acc / win_sum + 1e-20)
    return psd.astype(np.float32)


def channel_psd(
    capture,        # IQCapture
    fine: bool,
    fine_fft_size: int = 4096,
    fine_overlap: float = 0.5,
    coarse_fft_size: int = 512,
    window_name: str = "blackman",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: compute PSD for omni and dir channels of an IQCapture.

    Returns (psd_omni_db, psd_dir_db, freq_axis_hz_array).
    """
    if fine:
        psd_o = fine_welch_db(
            capture.omni_iq, fft_size=fine_fft_size,
            overlap=fine_overlap, window_name=window_name,
        )
        psd_d = fine_welch_db(
            capture.dir_iq, fft_size=fine_fft_size,
            overlap=fine_overlap, window_name=window_name,
        )
        n_bins = fine_fft_size
    else:
        psd_o = coarse_psd_db(capture.omni_iq, fft_size=coarse_fft_size, window_name=window_name)
        psd_d = coarse_psd_db(capture.dir_iq, fft_size=coarse_fft_size, window_name=window_name)
        n_bins = coarse_fft_size

    f_ax = freq_axis_hz(capture.center_hz, capture.bandwidth_hz, n_bins)
    return psd_o, psd_d, f_ax
