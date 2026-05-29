"""
Pluggable signal-processing algorithms for the spectrum engine.

The engine calls a `SignalProcessor` for every fine-scan measurement.
Several implementations live side-by-side here and are selectable at
runtime through the GUI — mirroring how the Hardware/Simulator source
toggle replaced one hard-coded SDR backend.

See `signal_processing_selector_plan.md` for the architecture, and
`signal_processing_integration_plan.md` for the new OS-CFAR + classifier.
"""

from .base import ClassificationResult, SignalProcessor  # noqa: F401
from .classic import ClassicProcessor  # noqa: F401
from .oscfar import OSCFARProcessor  # noqa: F401
from .registry import (  # noqa: F401
    default_processor,
    get_processor,
    list_processors,
)
