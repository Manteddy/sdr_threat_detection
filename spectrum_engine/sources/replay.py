"""
ReplayIQSource — feeds a recorded experiment back through the engine.

Implements the IQSource contract so SignalReader sees it identically to
PyAdiIQSource. acquire() returns IQ in raw ADC units (not normalised);
SignalReader's normalisation step cancels with the scale-back-up done at
recording time.

Time-locked replay: captures are consumed in the order they were
recorded. get_next_command() vends the corresponding MeasurementCommand
from the sweep log so the engine can use the original cell-update
coordinates instead of the live scheduler's choice.

Usage (inside SpectrumEngine with replay mode)::

    source = ReplayIQSource("experiments/2026-05-31_142233_fpv/")
    reader = SignalReader(source, eng_cfg, realtime=False, anti_alias=False)
    engine.attach_backend(reader)
    engine.attach_replay_source(source)   # overrides scheduler
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

from ..iq_source import HardwareLimits, IQSource


class ReplayIQSource(IQSource):
    """IQSource backed by a recorded experiment folder."""

    def __init__(self, folder: str) -> None:
        self._folder = folder
        self._iq_dir = os.path.join(folder, "iq")

        # Load experiment metadata
        exp_path = os.path.join(folder, "experiment.json")
        if not os.path.isfile(exp_path):
            raise FileNotFoundError(f"experiment.json not found in {folder}")
        with open(exp_path, "r", encoding="utf-8") as fh:
            self._meta: dict = json.load(fh)

        # Build an ordered list of IQ file paths
        try:
            iq_files = sorted(
                f for f in os.listdir(self._iq_dir) if f.endswith(".npz")
            )
        except FileNotFoundError:
            iq_files = []
        self._iq_paths: List[str] = [
            os.path.join(self._iq_dir, f) for f in iq_files
        ]
        self._index: int = 0

        # Load sweep log as a list of command dicts
        sweep_path = os.path.join(folder, "sweep_log.jsonl")
        self._commands: List[dict] = []
        if os.path.isfile(sweep_path):
            with open(sweep_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            self._commands.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        self._cmd_index: int = 0

        # Reconstruct HardwareLimits from the recorded hardware config
        hw = self._meta.get("hardware", {})
        self._adc_full_scale: float = float(
            hw.get("adc_full_scale",
                   self._meta.get("engine_config", {})
                              .get("hardware", {})
                              .get("adc_full_scale", 2048.0))
        )
        self._limits = HardwareLimits(
            min_hz=float(hw.get("min_hz", 70e6)),
            max_hz=float(hw.get("max_hz", 6000e6)),
            bandwidth_hz=float(hw.get("bandwidth_hz",
                                      hw.get("rf_bw", 40e6))),
            sample_rate_hz=float(hw.get("sample_rate_hz", 40e6)),
            dual_channel=False,
        )

    # ------------------------------------------------------------------
    # IQSource contract
    # ------------------------------------------------------------------

    def get_limits(self) -> HardwareLimits:
        return self._limits

    def tune(self, center_hz: float) -> None:
        """No-op — replay source ignores live tune requests.

        The engine uses get_next_command() to drive cell updates at the
        original recorded frequencies, so the tune argument is irrelevant.
        """

    def acquire(self, n_samples: int) -> np.ndarray:
        """Return the next recorded IQ buffer in raw ADC units.

        Raises StopIteration when all frames are exhausted — the engine
        catches this to snap back to Idle.
        """
        if self._index >= len(self._iq_paths):
            raise StopIteration("Replay exhausted")

        path = self._iq_paths[self._index]
        self._index += 1

        data = np.load(path)
        samples_i16: np.ndarray = data["samples"]

        # Reconstruct complex64 in raw ADC units
        i_part = samples_i16[0::2].astype(np.float32)
        q_part = samples_i16[1::2].astype(np.float32)
        raw_iq = (i_part + 1j * q_part).astype(np.complex64)
        # SignalReader will divide by adc_full_scale → produces normalized data
        return raw_iq

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Replay-specific API (used by SpectrumEngine in replay mode)
    # ------------------------------------------------------------------

    def get_next_command(self):
        """Return the next recorded MeasurementCommand.

        Returns None when the sweep log is exhausted (engine should stop).
        Imported lazily to avoid import cycles.
        """
        from spectrum_engine.scheduler import MeasurementCommand, TargetType

        if self._cmd_index >= len(self._commands):
            return None
        raw = self._commands[self._cmd_index]
        self._cmd_index += 1

        try:
            tt = TargetType(raw.get("target_type",
                                    TargetType.BACKGROUND_CELL_COARSE_SCAN))
        except ValueError:
            tt = TargetType.BACKGROUND_CELL_COARSE_SCAN

        return MeasurementCommand(
            target_type=tt,
            center_hz=float(raw["center_hz"]),
            bandwidth_hz=float(raw["bandwidth_hz"]),
            fft_size=int(raw.get("fft_size", 512)),
            num_frames=int(raw.get("num_frames", 1)),
            dwell_s=float(raw.get("dwell_s", 0.003)),
            reason="replay",
        )

    @property
    def is_exhausted(self) -> bool:
        return self._index >= len(self._iq_paths)

    @property
    def total_frames(self) -> int:
        return len(self._iq_paths)

    @property
    def current_frame(self) -> int:
        return self._index

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def adc_full_scale(self) -> float:
        return self._adc_full_scale
