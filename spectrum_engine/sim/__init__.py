"""
SDR Simulator — synthetic IQ source for the unified SignalReader.

Public exports cover the IQ source, the scene/emitter dataclasses,
and the preset scenarios surfaced in the GUI source/scene selectors.
"""

from .iq_source import SimulatedIQSource  # noqa: F401
from .preview import scene_psd_db  # noqa: F401
from .scene import Emitter, EmitterClass, Scene  # noqa: F401
from . import scenarios  # noqa: F401
