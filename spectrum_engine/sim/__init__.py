"""
SDR Simulator — drop-in replacement for the live SDRBackend.

Public exports cover the backend, the scene/emitter dataclasses, and the
preset scenarios surfaced in the GUI source/scene selectors.
"""

from .backend import SimulatedSDRBackend  # noqa: F401
from .preview import scene_psd_db  # noqa: F401
from .scene import Emitter, EmitterClass, Scene  # noqa: F401
from . import scenarios  # noqa: F401
