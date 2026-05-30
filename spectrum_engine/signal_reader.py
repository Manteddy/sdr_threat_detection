"""
Single-source-of-truth capture pipeline for the engine.

`SignalReader` is the only thing `SpectrumEngine.attach_backend` accepts.
It does all the work that is independent of where the IQ comes from:

1. Retune the source.
2. Honour the configured PLL settle time.
3. Pull raw IQ via `source.acquire`.
4. Normalise to dBFS (divide by `adc_full_scale`).
5. Apply an anti-alias filter at `bandwidth_hz`.
6. Reshape into `(num_frames, fft_size)`.
7. Mirror to `dir_iq` (we are single-antenna).
8. Honour an optional realtime dwell budget so the engine cadence
   matches what hardware would actually deliver.

The point of routing everything through one function is to guarantee
that an `IQCapture` produced by the simulator is shape-, scale-, and
spectrum-equivalent to one produced by real hardware. The engine cannot
tell the difference.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .iq_source import HardwareLimits, IQCapture, IQSource

if TYPE_CHECKING:
    from .config import EngineConfig


def _bandlimit(iq: np.ndarray, fs: float, bandwidth_hz: float) -> np.ndarray:
    """Zero out FFT bins outside `[-bw/2, +bw/2]` to suppress aliased
    energy from emitters near the edge of the tuned window.

    On real hardware the analog LPF already does this, so the filter is
    a no-op in spectral terms. On the simulator it eliminates the
    edge-of-window aliasing called out in the architecture review.
    """
    n = iq.size
    if n == 0 or bandwidth_hz <= 0:
        return iq
    spec = np.fft.fft(iq)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    mask = np.abs(freqs) > (bandwidth_hz / 2.0)
    spec[mask] = 0.0
    return np.fft.ifft(spec).astype(np.complex64)


class SignalReader:
    """The single capture function shared by hardware and simulator paths.

    Parameters
    ----------
    source :
        An `IQSource` implementation — `PyAdiIQSource` for hardware,
        `SimulatedIQSource` for the simulator.
    cfg :
        Engine configuration. The `hardware` section drives sample rate,
        ADC scale, PLL settle, and base FFT size.
    realtime :
        When True (default), `capture()` sleeps the residual dwell
        budget so the engine loop runs at roughly hardware cadence.
        Tests set this False for fastest iteration.
    anti_alias :
        When True (default), apply the `bandwidth_hz` LPF. Disable in
        narrow benchmarks where you want raw source output.
    """

    def __init__(
        self,
        source: IQSource,
        cfg: "EngineConfig",
        *,
        realtime: bool = True,
        anti_alias: bool = True,
    ) -> None:
        self._source = source
        self._cfg = cfg
        self._realtime = realtime
        self._anti_alias = anti_alias

    # ------------------------------------------------------------------
    # Engine-facing API (duck-typed with the deprecated SDRBackend)
    # ------------------------------------------------------------------

    @property
    def source(self) -> IQSource:
        """The underlying source. Used by the GUI to swap simulator scenes."""
        return self._source

    def get_limits(self) -> HardwareLimits:
        return self._source.get_limits()

    def close(self) -> None:
        self._source.close()

    def capture(
        self,
        center_hz: float,
        bandwidth_hz: float,
        dwell_s: float,
        num_frames: int,
    ) -> IQCapture:
        cfg_hw = self._cfg.hardware
        fft_size = cfg_hw.fft_size_base
        num_frames = max(1, int(num_frames))
        n_request = fft_size * num_frames
        fs = float(cfg_hw.sample_rate_hz)

        # --- 1. retune + 2. PLL settle ---
        self._source.tune(center_hz)
        time.sleep(cfg_hw.pll_settle_s)

        # --- 3. acquire raw ---
        timestamp = time.monotonic()
        raw = self._source.acquire(n_request)
        if raw is None:
            raw = np.zeros(n_request, dtype=np.complex64)
        raw = np.asarray(raw, dtype=np.complex64).ravel()

        # --- 4. normalise to dBFS unit ---
        scale = float(cfg_hw.adc_full_scale)
        if scale > 0:
            raw = (raw / scale).astype(np.complex64)

        # --- 5. anti-alias filter to bandwidth_hz ---
        if self._anti_alias:
            raw = _bandlimit(raw, fs, bandwidth_hz)

        # --- 6. frame & pad ---
        avail_frames = max(1, min(num_frames, len(raw) // fft_size))
        need = avail_frames * fft_size
        if len(raw) < need:
            raw = np.pad(raw, (0, need - len(raw)))
        omni = raw[:need].reshape(avail_frames, fft_size)

        # --- 7. single-antenna: dir mirrors omni ---
        dir_iq = omni.copy()

        # --- 8. realtime dwell budget ---
        if self._realtime:
            target = max(0.0, dwell_s - (n_request / fs))
            if target > 0:
                time.sleep(target)

        return IQCapture(
            center_hz=float(center_hz),
            bandwidth_hz=float(bandwidth_hz),
            timestamp=timestamp,
            omni_iq=omni,
            dir_iq=dir_iq,
        )
