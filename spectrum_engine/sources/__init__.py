"""
Concrete `IQSource` implementations.

Each module here owns the bits that are genuinely hardware-specific
(libiio quirks, scene synthesis, future USRP/HackRF backends).
Everything that's source-independent — normalisation, framing, anti-
alias, dwell budget, single-antenna mirror — lives in
`spectrum_engine/signal_reader.py`.
"""

from .pyadi import PyAdiIQSource  # noqa: F401
