"""
Baseband IQ synthesis primitives.

All generators return complex64 arrays at the engine's sample rate, centred
at 0 Hz. The backend handles frequency-shifting them to each emitter's
carrier, summing, and scaling to the requested power_dbfs.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bandlimit_complex_noise(n: int, fs: float, bandwidth_hz: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Generate complex Gaussian noise then zero out spectral content outside
    `[-bw/2, +bw/2]` so the resulting signal has a flat in-band spectrum
    and ~zero out-of-band content.
    """
    raw = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    spec = np.fft.fft(raw)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    mask = np.abs(freqs) > (bandwidth_hz / 2.0)
    spec[mask] = 0.0
    out = np.fft.ifft(spec)
    # Re-normalise to unit RMS so downstream power scaling is exact.
    rms = float(np.sqrt(np.mean(np.abs(out) ** 2)) + 1e-30)
    return (out / rms).astype(np.complex64)


def _dbfs_to_amplitude(power_dbfs: float) -> float:
    """Convert dBFS power to linear *amplitude* such that a unit-RMS signal
    scaled by this value has the desired power dBFS."""
    return float(10.0 ** (power_dbfs / 20.0))


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def gen_awgn(n: int, power_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """Complex white Gaussian noise at the configured power level."""
    real = rng.standard_normal(n).astype(np.float32)
    imag = rng.standard_normal(n).astype(np.float32)
    iq = (real + 1j * imag) / np.sqrt(2.0)
    return (_dbfs_to_amplitude(power_dbfs) * iq).astype(np.complex64)


def gen_barrage_jammer(n: int, fs: float, bandwidth_hz: float,
                       power_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Wideband AWGN-style jammer: flat noise spectrum across `bandwidth_hz`,
    zero outside. Spectral flatness is intentionally high (no cyclic
    structure) so the classifier reliably labels it BARRAGE_JAMMER.
    """
    iq_unit = _bandlimit_complex_noise(n, fs, bandwidth_hz, rng)
    return (_dbfs_to_amplitude(power_dbfs) * iq_unit).astype(np.complex64)


def gen_analog_fpv(n: int, fs: float, bandwidth_hz: float, power_dbfs: float,
                   rng: np.random.Generator, *,
                   sync_h_hz: float = 15625.0,
                   sync_v_hz: float = 50.0,
                   sync_am_depth: float = 0.18,
                   sync_duty: float = 0.08) -> np.ndarray:
    """
    Synthesise a baseband analog FPV video signal.

    Real analog FPV transmitters are FM-modulated composite video. The IF
    envelope is *nearly* constant (FM) but exhibits small periodic dips
    aligned to the horizontal sync tip — that residual AM is what makes
    the 15.625 kHz cyclic feature detectable in the envelope-power FFT.

    What we synthesise:
      1. Build a video "message" m(t) = horizontal sawtooth + sync tip pulse
         + slowly varying vertical envelope × random luma noise. This drives
         the FM modulator and produces the wideband FM spectrum.
      2. FM-modulate: iq_fm(t) = exp(j 2π Kf ∫m(t)dt). Kf is chosen so the
         resulting spectrum covers ~bandwidth_hz (Carson's rule, rough).
      3. Multiply by a small AM envelope that dips during the sync interval,
         producing the residual cyclic 15.625 kHz feature in |iq|².
    """
    t = np.arange(n, dtype=np.float64) / fs

    # ---- Video message ----
    h_phase = (t * sync_h_hz) % 1.0
    h_sawtooth = h_phase.astype(np.float32)
    in_sync = h_phase < sync_duty
    h_sync_pulse = np.where(in_sync, -0.5, 0.0).astype(np.float32)

    v_env = (0.5 + 0.5 * np.cos(2.0 * np.pi * sync_v_hz * t)).astype(np.float32)
    luma = rng.standard_normal(n).astype(np.float32) * 0.2
    message = h_sawtooth + h_sync_pulse + 0.3 * v_env * luma

    # ---- FM modulation ----
    # Carson's rule: BW ≈ 2(Δf + fm). Targeting `bandwidth_hz`, with the
    # message peak ~1.3 and message BW ~5 MHz worst case, Kf ≈ bw/4 lands
    # the spectrum in the right neighbourhood for the engine's PSD path.
    kf = float(bandwidth_hz) / 4.0
    phase = 2.0 * np.pi * kf * np.cumsum(message) / fs
    iq_fm = np.exp(1j * phase).astype(np.complex64)

    # ---- Residual AM aligned to sync ----
    am_envelope = (1.0 - sync_am_depth * in_sync.astype(np.float32))
    iq_unit = iq_fm * am_envelope

    # Normalise to unit RMS, then scale to requested power
    rms = float(np.sqrt(np.mean(np.abs(iq_unit) ** 2)) + 1e-30)
    iq_unit = iq_unit / rms
    return (_dbfs_to_amplitude(power_dbfs) * iq_unit).astype(np.complex64)
