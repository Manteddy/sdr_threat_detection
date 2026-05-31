"""
Experiment recording and replay package.

ExperimentRecorder streams hardware captures to disk as self-contained
experiment folders. ReplayIQSource feeds those recordings back through
the engine pipeline.
"""

from .recorder import ExperimentRecorder, RecordingOptions

__all__ = ["ExperimentRecorder", "RecordingOptions"]
