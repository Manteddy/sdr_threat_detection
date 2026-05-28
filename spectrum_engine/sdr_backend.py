"""
SDR Backend abstraction for the Adaptive Spectrum Sensing Engine.

Wraps a pyadi-iio object (AD9361 / PlutoSDR) and exposes a clean
capture() interface used by both coarse and fine measurement paths.

The tune/capture/IQ-extraction logic is refactored from
SweepWorker._do_one_step() in drone_detector_enhanced.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import EngineConfig, HardwareCfg


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HardwareLimits:
    """Reported capabilities of the connected SDR."""
    min_hz: float
    max_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    dual_channel: bool


@dataclass
class IQCapture:
    """Raw IQ sample block returned by a single capture call."""
    center_hz: float
    bandwidth_hz: float
    timestamp: float
    # Shape: (num_frames, fft_size) or (fft_size,) when num_frames==1
    omni_iq: np.ndarray    # complex64
    dir_iq: np.ndarray     # complex64 — mirrors omni when single-channel


# ---------------------------------------------------------------------------
# SDRBackend
# ---------------------------------------------------------------------------

class SDRBackend:
    """
    Thin wrapper around a live pyadi-iio SDR object.

    Parameters
    ----------
    sdr_obj :
        A connected pyadi-iio object (adi.ad9361 or adi.Pluto).
    cfg :
        Engine configuration (hardware section used for defaults).
    dual_channel :
        True when two independent RX channels are available and verified.
    """

    # PlutoSDR / AD9361 practical tuning range
    _HW_MIN_HZ: float = 70e6
    _HW_MAX_HZ: float = 6000e6

    def __init__(
        self,
        sdr_obj,
        cfg: EngineConfig,
        dual_channel: bool = False,
    ) -> None:
        self._sdr = sdr_obj
        self._cfg: HardwareCfg = cfg.hardware
        self.dual_channel: bool = dual_channel

        # Derive limits from hardware config
        self._limits = HardwareLimits(
            min_hz=self._HW_MIN_HZ,
            max_hz=self._HW_MAX_HZ,
            bandwidth_hz=self._cfg.default_bandwidth_hz,
            sample_rate_hz=self._cfg.sample_rate_hz,
            dual_channel=dual_channel,
        )

        self._adc_scale: float = self._cfg.adc_full_scale
        self._pll_settle_s: float = self._cfg.pll_settle_s
        self._base_fft_size: int = self._cfg.fft_size_base

        # Drive the radio at the engine's configured sample rate / bandwidth.
        # At 40 MHz each capture spans two 20 MHz cells, so a full sweep needs
        # ~half the retunes (the per-rx() USB latency dominates, not dwell).
        try:
            self._sdr.sample_rate = int(self._cfg.sample_rate_hz)
            self._sdr.rx_rf_bandwidth = int(self._cfg.default_bandwidth_hz)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_limits(self) -> HardwareLimits:
        return self._limits

    def capture(
        self,
        center_hz: float,
        bandwidth_hz: float,
        dwell_s: float,
        num_frames: int,
    ) -> IQCapture:
        """
        Tune the SDR to center_hz and capture num_frames of IQ data.

        dwell_s is used only to compute an optional extra sleep when
        num_frames × frame_time < dwell_s (keeps measured dwell accurate
        for energy calculations even with short captures).

        Returns an IQCapture with omni_iq / dir_iq shaped (num_frames, fft_size).

        Performance & robustness: a SINGLE rx() call reads whatever the SDR's
        existing hardware buffer holds (the GUI configures it to FFT_SIZE*4 =
        2048 samples). We deliberately DO NOT change rx_buffer_size here —
        larger buffers make pyadi-iio return a different/interleaved sample
        layout, which broke capture entirely (0 scans). Reading the existing
        2048-sample buffer once and reshaping keeps the proven-working format
        while still doing only ONE rx() per scan (vs. the old per-frame loop).
        """
        fft_size = self._base_fft_size
        num_frames = max(1, int(num_frames))

        # IMPORTANT: match the proven-reliable order used by the simple detector
        # and the classic sweep loop EXACTLY — destroy the old buffer FIRST, then
        # retune, settle, and read. Destroying *after* a retune (the previous
        # order) made libiio fail to recreate the RX buffer (OSError errno 0),
        # which silently zeroed out every scan.
        try:
            self._sdr.rx_destroy_buffer()
        except Exception:
            pass

        self._sdr.rx_lo = int(center_hz)
        time.sleep(self._pll_settle_s)

        timestamp = time.monotonic()

        # One rx() returns the full hardware buffer at its natural length;
        # reshape into as many fft_size frames as actually fit (coarse uses 1,
        # fine uses whatever the buffer provides, capped at num_frames).
        omni_flat, dir_flat = self._read_natural(fft_size)
        avail_frames = max(1, min(num_frames, len(omni_flat) // fft_size))
        need = avail_frames * fft_size
        omni_arr = omni_flat[:need].reshape(avail_frames, fft_size)
        dir_arr = dir_flat[:need].reshape(avail_frames, fft_size)

        return IQCapture(
            center_hz=center_hz,
            bandwidth_hz=bandwidth_hz,
            timestamp=timestamp,
            omni_iq=omni_arr,
            dir_iq=dir_arr,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_natural(self, min_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call sdr.rx() once and return (omni, dir) normalised flat arrays at the
        buffer's natural length (no forced target).         Pads only up to min_samples
        so at least one fft_size frame is always available.
        """
        try:
            raw = self._sdr.rx()
        except Exception:
            # Buffer allocation/read can transiently fail right after a retune;
            # drop the buffer and try once more before giving up.
            try:
                self._sdr.rx_destroy_buffer()
            except Exception:
                pass
            time.sleep(0.002)
            raw = self._sdr.rx()

        if self.dual_channel:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                omni_raw = np.asarray(raw[0], dtype=np.complex64).ravel()
                dir_raw = np.asarray(raw[1], dtype=np.complex64).ravel()
            elif isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[0] >= 2:
                omni_raw = raw[0].astype(np.complex64).ravel()
                dir_raw = raw[1].astype(np.complex64).ravel()
            else:
                omni_raw = np.asarray(raw, dtype=np.complex64).ravel()
                dir_raw = omni_raw.copy()
        else:
            omni_raw = np.asarray(raw, dtype=np.complex64).ravel()
            dir_raw = omni_raw.copy()

        n = len(omni_raw)
        if n < min_samples:
            omni_raw = np.pad(omni_raw, (0, min_samples - n))
            dir_raw = np.pad(dir_raw, (0, min_samples - n))

        omni_norm = (omni_raw / self._adc_scale).astype(np.complex64)
        dir_norm = (dir_raw / self._adc_scale).astype(np.complex64)
        return omni_norm, dir_norm

    def _read_block(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call sdr.rx() once and return (omni, dir) flat arrays of length
        n_samples each (normalised, padded if short).

        Handles single-channel, list/tuple dual-channel, and 2D array formats.
        """
        raw = self._sdr.rx()

        if self.dual_channel:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                omni_raw = np.asarray(raw[0], dtype=np.complex64).ravel()
                dir_raw = np.asarray(raw[1], dtype=np.complex64).ravel()
            elif isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[0] >= 2:
                omni_raw = raw[0].astype(np.complex64).ravel()
                dir_raw = raw[1].astype(np.complex64).ravel()
            else:
                omni_raw = np.asarray(raw, dtype=np.complex64).ravel()
                dir_raw = omni_raw.copy()
        else:
            omni_raw = np.asarray(raw, dtype=np.complex64).ravel()
            dir_raw = omni_raw.copy()

        # Pad or truncate to exactly n_samples
        n = len(omni_raw)
        if n < n_samples:
            omni_raw = np.pad(omni_raw, (0, n_samples - n))
            dir_raw = np.pad(dir_raw, (0, n_samples - n))

        omni_norm = (omni_raw[:n_samples] / self._adc_scale).astype(np.complex64)
        dir_norm = (dir_raw[:n_samples] / self._adc_scale).astype(np.complex64)

        return omni_norm, dir_norm
