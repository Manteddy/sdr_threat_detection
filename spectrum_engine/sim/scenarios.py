"""
Preset Scene factories exposed in the GUI source/scene selector.

Frequencies span the 1 - 6.5 GHz monitoring range:
  * 1.3 GHz, 2.4 GHz, 3.7 GHz, 5.8 GHz, 6.2 GHz — single ANALOG_FPV emitter
  * 2.45 GHz BARRAGE_JAMMER — sanity check for the wideband-jammer path
  * Empty band — noise floor only
"""

from __future__ import annotations

from typing import List, Tuple

from .scene import Emitter, EmitterClass, Scene


_DEFAULT_FPV_BW_HZ: float = 8e6
_DEFAULT_FPV_POWER_DBFS: float = -38.0
_DEFAULT_NOISE_FLOOR_DBFS: float = -78.0


# ---------------------------------------------------------------------------
# Single-emitter factories
# ---------------------------------------------------------------------------

def empty_band(noise_floor_dbfs: float = _DEFAULT_NOISE_FLOOR_DBFS) -> Scene:
    """No emitters — just ambient noise."""
    return Scene(name="empty_band", emitters=[], noise_floor_dbfs=noise_floor_dbfs)


def fpv_at(
    center_hz: float,
    *,
    bandwidth_hz: float = _DEFAULT_FPV_BW_HZ,
    power_dbfs: float = _DEFAULT_FPV_POWER_DBFS,
    name: str = "",
) -> Scene:
    """Single ANALOG_FPV emitter at the given carrier."""
    emitter = Emitter(
        center_hz=center_hz,
        bandwidth_hz=bandwidth_hz,
        power_dbfs=power_dbfs,
        cls=EmitterClass.ANALOG_FPV,
    )
    label = name or f"fpv_{int(center_hz / 1e6)}_mhz"
    return Scene(name=label, emitters=[emitter], noise_floor_dbfs=_DEFAULT_NOISE_FLOOR_DBFS)


def jammer_at(
    center_hz: float,
    *,
    bandwidth_hz: float = 12e6,
    power_dbfs: float = -32.0,
    name: str = "",
) -> Scene:
    """
    Single BARRAGE_JAMMER (band-limited noise) at the given carrier.

    Default bandwidth (12 MHz) is intentionally narrower than the default
    capture bandwidth (20 MHz) so the engine sees out-of-jammer bins on
    each side and percentile-based noise floor estimation still works.
    A jammer that fills the entire capture is detectable only by
    cell-aggregate techniques — see `signal_processing_integration_plan`.
    """
    emitter = Emitter(
        center_hz=center_hz,
        bandwidth_hz=bandwidth_hz,
        power_dbfs=power_dbfs,
        cls=EmitterClass.BARRAGE_JAMMER,
    )
    label = name or f"jammer_{int(center_hz / 1e6)}_mhz"
    return Scene(name=label, emitters=[emitter], noise_floor_dbfs=_DEFAULT_NOISE_FLOOR_DBFS)


# ---------------------------------------------------------------------------
# Named GUI presets — (label, factory)
# Order matters: this is the order shown in the combo box.
# ---------------------------------------------------------------------------

GUI_PRESETS: List[Tuple[str, "callable"]] = [
    ("Empty band (noise only)",  lambda: empty_band()),
    ("FPV @ 1.3 GHz",            lambda: fpv_at(1.300e9, name="fpv_1300_mhz")),
    ("FPV @ 2.4 GHz",            lambda: fpv_at(2.400e9, name="fpv_2400_mhz")),
    ("FPV @ 3.7 GHz",            lambda: fpv_at(3.700e9, name="fpv_3700_mhz")),
    ("FPV @ 5.8 GHz",            lambda: fpv_at(5.800e9, name="fpv_5800_mhz")),
    ("FPV @ 6.2 GHz",            lambda: fpv_at(6.200e9, name="fpv_6200_mhz")),
    ("Jammer @ 2.45 GHz",        lambda: jammer_at(2.450e9, name="jammer_2450_mhz")),
]


def preset_by_label(label: str) -> Scene:
    """Return a freshly-constructed Scene matching one of GUI_PRESETS."""
    for lbl, factory in GUI_PRESETS:
        if lbl == label:
            return factory()
    raise KeyError(f"Unknown preset: {label!r}")
