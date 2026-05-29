"""
Processor registry — single source of truth for what algorithms exist.

Hand-maintained list (not import-magic). Order matters: it drives the
order of the GUI Proc combo. The first entry is the default.
"""

from __future__ import annotations

from typing import List, Tuple, Type

from .base import SignalProcessor
from .classic import ClassicProcessor
from .oscfar import OSCFARProcessor

# Ordered list of available processors. First entry is the default.
_PROCESSORS: List[Type[SignalProcessor]] = [
    ClassicProcessor,
    OSCFARProcessor,
]


def list_processors() -> List[Tuple[str, str]]:
    """Return `[(name, label), ...]` in display order. Drives the GUI combo."""
    return [(p.name, p.label) for p in _PROCESSORS]


def get_processor(name: str) -> SignalProcessor:
    """Instantiate the processor whose `name` matches."""
    for p in _PROCESSORS:
        if p.name == name:
            return p()
    raise KeyError(f"Unknown signal processor: {name!r}")


def default_processor() -> SignalProcessor:
    """Instantiate the default (first-registered) processor."""
    return _PROCESSORS[0]()
