"""
proximity_alert - Drone proximity warning package.

Provides alert engine, embeddable PyQt5 widget, and WebSocket bridge
for integration with mission-control / GCS dashboards.
"""

from .engine import AlertEngine, ProximityAlert, TrendDirection  # noqa: F401

__version__ = "0.1.0"
