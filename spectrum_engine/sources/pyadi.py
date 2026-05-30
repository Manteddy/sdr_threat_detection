"""
PyAdiIQSource — raw IQ acquisition from a pyadi-iio AD9361 / Pluto.

Stripped to just the hardware-specific concerns:
  * tune the LO (with the libiio destroy-buffer-first ordering)
  * read one rx() buffer
  * extract a single channel and return complex64 in raw ADC units

Framing, normalisation, anti-alias filtering, and dwell budgeting all
live in `SignalReader` — they are identical across sources, so they
don't belong here.

Notable changes vs. the deprecated `SDRBackend`:

* **No dual-channel detection.** The whole project is single-antenna;
  `SignalReader` mirrors `omni_iq → dir_iq` in the IQCapture.
* **`rx_buffer_size` is set once at `__init__`** to
  `fft_size_base × max(fine_num_frames, coarse_num_frames)` so the
  hardware actually delivers the number of frames the engine asks
  for. The historical 2048-sample cap (which made fine-scan PSD
  use 4 frames instead of the requested 16, breaking OS-CFAR sync
  detection on real hardware) is lifted by default. Pass
  `rx_buffer_size_override=2048` if you hit the libiio "interleaved
  sample layout" bug the old comment in sdr_backend.py warned about.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..iq_source import HardwareLimits, IQSource

if TYPE_CHECKING:
    from ..config import EngineConfig


class PyAdiIQSource(IQSource):
    """`IQSource` over a pyadi-iio object (AD9361 or Pluto)."""

    # Practical AD9361 tuning range.
    _HW_MIN_HZ: float = 70e6
    _HW_MAX_HZ: float = 6000e6

    def __init__(
        self,
        sdr_obj,
        cfg: "EngineConfig",
        *,
        rx_buffer_size_override: Optional[int] = None,
    ) -> None:
        self._sdr = sdr_obj
        cfg_hw = cfg.hardware

        # Pick a buffer size large enough for fine scans by default.
        default_buf = int(cfg_hw.fft_size_base) * max(
            int(cfg.fine_scan.num_frames),
            int(cfg.coarse_scan.num_frames),
            1,
        )
        self._rx_buffer_size = int(rx_buffer_size_override or default_buf)

        # Drive the radio at the engine's configured sample rate / RF BW
        # and pin the buffer size so subsequent rx() calls return
        # `_rx_buffer_size` samples each.
        try:
            self._sdr.sample_rate = int(cfg_hw.sample_rate_hz)
            self._sdr.rx_rf_bandwidth = int(cfg_hw.default_bandwidth_hz)
            self._sdr.rx_buffer_size = self._rx_buffer_size
        except Exception:
            # Some pyadi-iio versions / firmwares reject one of these; the
            # capture loop's error handling will surface real failures.
            pass

        self._limits = HardwareLimits(
            min_hz=self._HW_MIN_HZ,
            max_hz=self._HW_MAX_HZ,
            bandwidth_hz=float(cfg_hw.default_bandwidth_hz),
            sample_rate_hz=float(cfg_hw.sample_rate_hz),
            dual_channel=False,  # explicitly single-antenna
        )

    # ------------------------------------------------------------------
    # IQSource contract
    # ------------------------------------------------------------------

    def get_limits(self) -> HardwareLimits:
        return self._limits

    def tune(self, center_hz: float) -> None:
        """Retune the LO.

        Order matters: destroy the buffer *first*, then set rx_lo. The
        reverse order has caused libiio to fail to recreate the RX
        buffer (errno-0 stalls, zero scans). See the historic comment in
        the deprecated sdr_backend.py for the incident notes.
        """
        try:
            self._sdr.rx_destroy_buffer()
        except Exception:
            pass
        self._sdr.rx_lo = int(center_hz)

    def acquire(self, n_samples: int) -> np.ndarray:
        """One `rx()` call. Returns raw ADC-scale complex64 samples.

        `SignalReader` divides by `adc_full_scale` to normalise to dBFS.
        Length matches `rx_buffer_size` set at init; the reader will
        truncate or pad if it differs from `n_samples`.
        """
        try:
            raw = self._sdr.rx()
        except Exception:
            # Buffer allocation can transiently fail right after a retune
            # on some hardware. One retry, same as the old code did.
            try:
                self._sdr.rx_destroy_buffer()
            except Exception:
                pass
            time.sleep(0.002)
            raw = self._sdr.rx()

        # Pluto in 2r2t mode returns a list/tuple of two channels even
        # when only one is enabled. Take channel 0 unconditionally.
        if isinstance(raw, (list, tuple)) and len(raw) >= 1:
            iq = np.asarray(raw[0], dtype=np.complex64).ravel()
        elif isinstance(raw, np.ndarray) and raw.ndim == 2:
            iq = raw[0].astype(np.complex64).ravel()
        else:
            iq = np.asarray(raw, dtype=np.complex64).ravel()

        return iq
