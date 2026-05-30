"""
IQ acquisition abstractions for the Adaptive Spectrum Sensing Engine.

The engine ingests IQ exclusively through a `SignalReader` (see
`signal_reader.py`) which delegates raw sample acquisition to an
`IQSource` — pyadi-iio for real hardware, a scene synthesiser for the
simulator. This file owns:

* `IQCapture` — the immutable record handed to `SpectrumEngine.step()`.
* `HardwareLimits` — what an SDR (or simulator) reports about itself.
* `IQSource` — the abstract base class every source implements.

Sources are deliberately single-channel. The dual-channel (omni + dir)
shape that crosses thread boundaries on `IQCapture` is satisfied by the
`SignalReader` mirroring the omni sample on the dir slot — we are not
using two antennas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses (canonical home; `sdr_backend.py` re-exports for compat)
# ---------------------------------------------------------------------------

@dataclass
class HardwareLimits:
    """Capabilities reported by a source."""
    min_hz: float
    max_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    dual_channel: bool


@dataclass
class IQCapture:
    """One IQ measurement crossing the source → engine boundary.

    `omni_iq` and `dir_iq` are shaped `(num_frames, fft_size)` complex64.
    Since the project is single-antenna, `dir_iq` is currently always a
    copy of `omni_iq` (set by `SignalReader`).
    """
    center_hz: float
    bandwidth_hz: float
    timestamp: float
    omni_iq: np.ndarray
    dir_iq: np.ndarray


# ---------------------------------------------------------------------------
# IQSource ABC
# ---------------------------------------------------------------------------

class IQSource(ABC):
    """Abstract raw-IQ producer.

    `SignalReader` calls `tune()` then `acquire()` once per capture. The
    source returns complex64 samples in *raw ADC units* — i.e. on the
    same scale a real Pluto's `rx()` returns (roughly ±`adc_full_scale`).
    `SignalReader` divides by `adc_full_scale` to land in dBFS units, so
    both sources go through identical normalisation downstream.

    Sources are single-channel. `dir_iq` is the reader's concern.
    """

    @abstractmethod
    def get_limits(self) -> HardwareLimits:
        """Return the SDR's (or simulated SDR's) declared limits."""

    @abstractmethod
    def tune(self, center_hz: float) -> None:
        """Set the LO to `center_hz`. May trigger a hardware retune.

        Implementations must complete any state setup the subsequent
        `acquire()` call relies on. The `SignalReader` will sleep
        `pll_settle_s` after this returns before reading samples.
        """

    @abstractmethod
    def acquire(self, n_samples: int) -> np.ndarray:
        """Return complex64 IQ samples at the configured sample rate.

        The returned length should be ≥ `n_samples` whenever possible;
        the `SignalReader` will truncate or zero-pad to the framed size
        if it isn't.

        Returned values are in raw ADC units (≈ ±`adc_full_scale` peak).
        """

    # Optional lifecycle: default no-op. Sources may override to release
    # OS resources or stop background generators.
    def close(self) -> None:
        return None
