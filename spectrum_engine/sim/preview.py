"""
Analytical ground-truth PSD for the simulator preview panel.

The detector's RX1 spectrum shows what the receiver currently sees inside
one 20 MHz tuning window. The preview panel instead shows what is
*actually in the air* across the full 1 - 6.5 GHz monitoring range, all
the time, regardless of where the receiver is tuned. That makes it easy
to verify scenes and presets at a glance.

We do not synthesise IQ here — we draw the emitters analytically: noise
floor everywhere, plus a per-emitter spectral shape laid into the right
bin range. Cheap and deterministic, refreshed at GUI rate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .scene import Emitter, EmitterClass, Scene


def scene_psd_db(
    scene: Scene,
    start_hz: float,
    stop_hz: float,
    n_bins: int,
    *,
    noise_jitter_db: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return `(freqs_hz, psd_db)` for the scene over `[start_hz, stop_hz]`.

    Each emitter is laid into its expected bandwidth at its `power_dbfs`
    level. ANALOG_FPV gets gentle edge rolloff; BARRAGE_JAMMER stays
    flat; TELEMETRY_LINK is a sharp line. A small random jitter is added
    everywhere so the line keeps looking alive on refresh.
    """
    if rng is None:
        rng = np.random.default_rng()

    freqs = np.linspace(start_hz, stop_hz, n_bins, dtype=np.float64)
    # Linear power baseline = noise floor + jitter.
    noise_floor_dbfs = float(scene.noise_floor_dbfs)
    jitter = rng.standard_normal(n_bins).astype(np.float32) * noise_jitter_db
    psd_db = np.full(n_bins, noise_floor_dbfs, dtype=np.float32) + jitter

    for em in scene.emitters:
        _stamp_emitter(psd_db, freqs, em, rng)

    return freqs, psd_db


def _stamp_emitter(
    psd_db: np.ndarray,
    freqs: np.ndarray,
    em: Emitter,
    rng: np.random.Generator,
) -> None:
    """Imprint one emitter's PSD shape into `psd_db` in-place."""
    lo = em.center_hz - em.bandwidth_hz / 2.0
    hi = em.center_hz + em.bandwidth_hz / 2.0
    in_band = (freqs >= lo) & (freqs <= hi)
    if not np.any(in_band):
        return

    n_in = int(in_band.sum())
    level = float(em.power_dbfs)

    if em.cls is EmitterClass.ANALOG_FPV:
        # Gentle rolloff at the band edges + small FM ripple.
        x = np.linspace(-1.0, 1.0, n_in, dtype=np.float32)
        shape = 1.0 - 0.4 * np.abs(x) ** 2     # ~4 dB lower at edges
        ripple = rng.standard_normal(n_in).astype(np.float32) * 1.5
        contrib = level + 10.0 * np.log10(shape + 1e-3) + ripple
    elif em.cls is EmitterClass.BARRAGE_JAMMER:
        # Flat plateau with small noise jitter.
        contrib = level + rng.standard_normal(n_in).astype(np.float32) * 0.8
    elif em.cls is EmitterClass.TELEMETRY_LINK:
        # Narrow signal — make a small peak in the centre, rest at noise floor.
        contrib = np.full(n_in, level, dtype=np.float32)
    else:
        # NOISE_BURST or unknown — broadband contribution.
        contrib = level + rng.standard_normal(n_in).astype(np.float32) * 0.5

    # Combine: take max in dB (a strong emitter dominates the noise floor
    # in its band — same as what a real SDR sees).
    existing = psd_db[in_band]
    psd_db[in_band] = np.maximum(existing, contrib).astype(np.float32)
