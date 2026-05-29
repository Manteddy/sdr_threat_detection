"""
SimulatedSDRBackend — duck-typed drop-in for `SDRBackend`.

Exposes the same `get_limits()` + `capture()` surface so it can be passed
to `SpectrumEngine.attach_backend(...)` unchanged. Pure NumPy; no
pyadi-iio / libiio dependency, so the whole stack runs on macOS.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ..config import EngineConfig
from ..sdr_backend import HardwareLimits, IQCapture
from .scene import Emitter, EmitterClass, Scene
from . import synth


# Tunable range reported by the simulator. Wider than the real AD9361
# (70 MHz - 6 GHz) so the user-requested 1-6.5 GHz monitoring band is
# fully covered.
_SIM_MIN_HZ: float = 1.0e9
_SIM_MAX_HZ: float = 6.5e9


class SimulatedSDRBackend:
    """
    Synthetic IQ source matching the SDRBackend contract.

    Parameters
    ----------
    cfg :
        Engine configuration (hardware section consumed).
    scene :
        Initial scene. May be swapped at runtime via `set_scene`.
    seed :
        RNG seed for reproducibility. None → non-deterministic.
    dual_channel :
        If True, dir_iq gets an uncorrelated noise realization;
        otherwise dir_iq mirrors omni_iq (matches single-RX hardware).
    add_dc_spike :
        Inject a small DC offset to mimic PlutoSDR LO leakage so the
        engine's DC excision is exercised.
    realtime :
        Sleep for the requested dwell time. Tests set this False for
        fastest iteration.
    """

    def __init__(
        self,
        cfg: EngineConfig,
        scene: Scene,
        *,
        seed: Optional[int] = 42,
        dual_channel: bool = False,
        add_dc_spike: bool = True,
        realtime: bool = True,
    ) -> None:
        self._cfg = cfg
        self._hw = cfg.hardware
        self._scene_lock = threading.Lock()
        self._scene = scene
        self._dual_channel = dual_channel
        self._add_dc_spike = add_dc_spike
        self._realtime = realtime
        self._rng = np.random.default_rng(seed)
        self._t0 = time.monotonic()

        self._limits = HardwareLimits(
            min_hz=_SIM_MIN_HZ,
            max_hz=_SIM_MAX_HZ,
            bandwidth_hz=self._hw.default_bandwidth_hz,
            sample_rate_hz=self._hw.sample_rate_hz,
            dual_channel=dual_channel,
        )

    # ------------------------------------------------------------------
    # Scene management (thread-safe; GUI may call set_scene from main thread
    # while worker thread is in capture()).
    # ------------------------------------------------------------------

    def set_scene(self, scene: Scene) -> None:
        with self._scene_lock:
            self._scene = scene

    def get_scene(self) -> Scene:
        with self._scene_lock:
            return self._scene

    @property
    def dual_channel(self) -> bool:
        return self._dual_channel

    # ------------------------------------------------------------------
    # SDRBackend contract
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
        fft_size = self._hw.fft_size_base
        num_frames = max(1, int(num_frames))
        fs = float(self._hw.sample_rate_hz)
        n = fft_size * num_frames

        # Resolve scene state (snapshot under lock so a concurrent
        # set_scene cannot mutate emitters mid-capture).
        with self._scene_lock:
            scene = self._scene
            if scene.update is not None:
                scene.update(scene, time.monotonic() - self._t0)
            emitters = tuple(scene.emitters)
            noise_floor_dbfs = scene.noise_floor_dbfs

        # Ambient noise floor (always present).
        omni = synth.gen_awgn(n, noise_floor_dbfs, self._rng).astype(np.complex64)

        # Each in-range emitter contributes its own synth, frequency-shifted
        # so its carrier sits at delta = emitter.center_hz - center_hz
        # within the baseband window.
        half_bw = bandwidth_hz / 2.0
        win_lo = center_hz - half_bw
        win_hi = center_hz + half_bw

        for em in emitters:
            em_lo = em.center_hz - em.bandwidth_hz / 2.0
            em_hi = em.center_hz + em.bandwidth_hz / 2.0
            if em_hi < win_lo or em_lo > win_hi:
                continue  # outside the tuned window

            iq_em = self._synth_emitter(em, n, fs)
            delta = em.center_hz - center_hz
            if delta != 0.0:
                t = np.arange(n, dtype=np.float64) / fs
                shift = np.exp(2j * np.pi * delta * t).astype(np.complex64)
                iq_em = iq_em * shift
            omni = omni + iq_em

        # Optional DC spike — mirror the PlutoSDR LO-leakage pathology
        # so engine.DC excision (engine.py:300) is exercised.
        if self._add_dc_spike:
            omni = omni + np.complex64(0.04 + 0.0j)

        # Dual-channel: independent noise realization for `dir`.
        if self._dual_channel:
            # Re-synthesise the noise floor and emitter content with a
            # different RNG seed branch so the two channels are
            # statistically independent (matches independent antennas).
            dir_iq = synth.gen_awgn(n, noise_floor_dbfs, self._rng).astype(np.complex64)
            for em in emitters:
                em_lo = em.center_hz - em.bandwidth_hz / 2.0
                em_hi = em.center_hz + em.bandwidth_hz / 2.0
                if em_hi < win_lo or em_lo > win_hi:
                    continue
                iq_em = self._synth_emitter(em, n, fs)
                delta = em.center_hz - center_hz
                if delta != 0.0:
                    t = np.arange(n, dtype=np.float64) / fs
                    shift = np.exp(2j * np.pi * delta * t).astype(np.complex64)
                    iq_em = iq_em * shift
                dir_iq = dir_iq + iq_em
            if self._add_dc_spike:
                dir_iq = dir_iq + np.complex64(0.04 + 0.0j)
        else:
            dir_iq = omni.copy()

        omni_arr = omni.reshape(num_frames, fft_size).astype(np.complex64)
        dir_arr = dir_iq.reshape(num_frames, fft_size).astype(np.complex64)

        timestamp = time.monotonic()

        # Approximate real timing so the engine loop runs at hardware-ish
        # cadence. Skippable for tests via `realtime=False`.
        if self._realtime:
            target = self._hw.pll_settle_s + max(0.0, dwell_s - n / fs)
            if target > 0:
                time.sleep(target)

        return IQCapture(
            center_hz=center_hz,
            bandwidth_hz=bandwidth_hz,
            timestamp=timestamp,
            omni_iq=omni_arr,
            dir_iq=dir_arr,
        )

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
        # NOISE_BURST / TELEMETRY_LINK not yet implemented — fall back to
        # band-limited noise at the emitter bandwidth so callers still see
        # something rather than silently nothing.
        return synth.gen_barrage_jammer(
            n, fs,
            bandwidth_hz=em.bandwidth_hz,
            power_dbfs=em.power_dbfs,
            rng=self._rng,
        )
