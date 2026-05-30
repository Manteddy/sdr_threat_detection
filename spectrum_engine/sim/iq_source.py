"""
SimulatedIQSource — scene-driven raw IQ producer.

Replaces the framing-and-normalisation-heavy `SimulatedSDRBackend`. This
class only knows how to acquire raw samples for the active `Scene` at
the currently-tuned LO. Everything else — normalisation to dBFS, anti-
alias filter, framing, single-antenna mirror, dwell budget — lives in
`SignalReader`, shared with the hardware path.

The IQ returned by `acquire()` is in **raw ADC units** (≈ ±
`adc_full_scale` peak), the same scale `PyAdiIQSource` returns from
`rx()`. `SignalReader` divides by `adc_full_scale` for both. That is
the structural fix that makes hardware and simulator produce IQ which
is statistically the same from the engine's point of view.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..iq_source import HardwareLimits, IQSource
from . import synth
from .scene import Emitter, EmitterClass, Scene

if TYPE_CHECKING:
    from ..config import EngineConfig


# Tunable range the simulator advertises. Wider than real AD9361 so the
# user-requested 1-6.5 GHz monitoring band is fully covered.
_SIM_MIN_HZ: float = 1.0e9
_SIM_MAX_HZ: float = 6.5e9


class SimulatedIQSource(IQSource):
    """Synthetic `IQSource` driven by a mutable `Scene`."""

    def __init__(
        self,
        cfg: "EngineConfig",
        scene: Scene,
        *,
        seed: Optional[int] = 42,
        add_dc_spike: bool = True,
    ) -> None:
        cfg_hw = cfg.hardware
        self._scene_lock = threading.Lock()
        self._scene = scene
        self._add_dc_spike = add_dc_spike
        self._rng = np.random.default_rng(seed)
        self._t0 = time.monotonic()

        self._adc_scale = float(cfg_hw.adc_full_scale)
        self._fs = float(cfg_hw.sample_rate_hz)
        self._bandwidth_hz = float(cfg_hw.default_bandwidth_hz)
        self._current_center_hz: float = 0.0

        self._limits = HardwareLimits(
            min_hz=_SIM_MIN_HZ,
            max_hz=_SIM_MAX_HZ,
            bandwidth_hz=self._bandwidth_hz,
            sample_rate_hz=self._fs,
            dual_channel=False,  # single-antenna project-wide
        )

    # ------------------------------------------------------------------
    # Scene management (thread-safe; GUI may call set_scene from main
    # thread while worker thread is inside acquire()).
    # ------------------------------------------------------------------

    def set_scene(self, scene: Scene) -> None:
        with self._scene_lock:
            self._scene = scene

    def get_scene(self) -> Scene:
        with self._scene_lock:
            return self._scene

    # ------------------------------------------------------------------
    # IQSource contract
    # ------------------------------------------------------------------

    def get_limits(self) -> HardwareLimits:
        return self._limits

    def tune(self, center_hz: float) -> None:
        self._current_center_hz = float(center_hz)

    def acquire(self, n_samples: int) -> np.ndarray:
        """Synthesise `n_samples` of complex64 IQ for the active scene.

        Returned values are scaled to raw ADC units so the
        `SignalReader.capture()` divide-by-`adc_full_scale` step lands
        them in dBFS the same way real Pluto data does.
        """
        # Snapshot scene + (optionally) advance time-varying state.
        with self._scene_lock:
            scene = self._scene
            if scene.update is not None:
                scene.update(scene, time.monotonic() - self._t0)
            emitters = tuple(scene.emitters)
            noise_floor_dbfs = scene.noise_floor_dbfs

        n = max(1, int(n_samples))
        fs = self._fs
        center_hz = self._current_center_hz
        bandwidth_hz = self._bandwidth_hz

        # Ambient noise floor (always present).
        iq = synth.gen_awgn(n, noise_floor_dbfs, self._rng).astype(np.complex64)

        # In-window emitters, frequency-shifted to their offset from
        # the current LO.
        half_bw = bandwidth_hz / 2.0
        win_lo = center_hz - half_bw
        win_hi = center_hz + half_bw
        # Time axis hoisted out of the per-emitter loop.
        t: Optional[np.ndarray] = None
        for em in emitters:
            em_lo = em.center_hz - em.bandwidth_hz / 2.0
            em_hi = em.center_hz + em.bandwidth_hz / 2.0
            if em_hi < win_lo or em_lo > win_hi:
                continue
            iq_em = self._synth_emitter(em, n, fs)
            delta = em.center_hz - center_hz
            if delta != 0.0:
                if t is None:
                    t = np.arange(n, dtype=np.float64) / fs
                shift = np.exp(2j * np.pi * delta * t).astype(np.complex64)
                iq_em = iq_em * shift
            iq = iq + iq_em

        # DC spike — analogue of Pluto's LO leakage so SpectrumEngine's
        # DC excision is exercised against simulated data too.
        if self._add_dc_spike:
            iq = iq + np.complex64(0.04 + 0.0j)

        # Scale into raw ADC units. SignalReader divides this back out.
        return (iq * self._adc_scale).astype(np.complex64)

    # ------------------------------------------------------------------
    # Per-emitter synthesis dispatch
    # ------------------------------------------------------------------

    def _synth_emitter(self, em: Emitter, n: int, fs: float) -> np.ndarray:
        if em.cls is EmitterClass.ANALOG_FPV:
            return synth.gen_analog_fpv(
                n, fs,
                bandwidth_hz=em.bandwidth_hz,
                power_dbfs=em.power_dbfs,
                rng=self._rng,
                sync_h_hz=float(em.extra.get("sync_h_hz", 15625.0)),
                sync_v_hz=float(em.extra.get("sync_v_hz", 50.0)),
            )
        if em.cls is EmitterClass.BARRAGE_JAMMER:
            return synth.gen_barrage_jammer(
                n, fs,
                bandwidth_hz=em.bandwidth_hz,
                power_dbfs=em.power_dbfs,
                rng=self._rng,
            )
        # NOISE_BURST / TELEMETRY_LINK fall back to band-limited noise
        # so callers see *something* rather than silently nothing.
        return synth.gen_barrage_jammer(
            n, fs,
            bandwidth_hz=em.bandwidth_hz,
            power_dbfs=em.power_dbfs,
            rng=self._rng,
        )
