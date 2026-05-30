"""
Scene + Emitter dataclasses for the SDR simulator.

A Scene is an immutable description of what RF energy exists in the air at
a given moment. The SimulatedIQSource uses it to synthesise IQ in
response to each SignalReader capture call.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class EmitterClass(str, enum.Enum):
    ANALOG_FPV = "ANALOG_FPV"
    BARRAGE_JAMMER = "BARRAGE_JAMMER"
    TELEMETRY_LINK = "TELEMETRY_LINK"
    NOISE_BURST = "NOISE_BURST"


@dataclass(frozen=True)
class Emitter:
    """One RF source contributing to a Scene."""
    center_hz: float
    bandwidth_hz: float
    power_dbfs: float            # peak power at the simulated RX (relative to ADC full-scale)
    cls: EmitterClass
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    """
    Mutable container for a set of emitters + ambient noise floor.

    `update` is optional; if set it is called with `(scene, t_seconds)` at
    the top of every capture so emitter power / position can walk over
    wall-clock time (e.g. drone_approach scenario).
    """
    name: str = "scene"
    emitters: List[Emitter] = field(default_factory=list)
    noise_floor_dbfs: float = -78.0
    update: Optional[Callable[["Scene", float], None]] = None
