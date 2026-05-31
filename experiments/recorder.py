"""
ExperimentRecorder — streams a hardware session to disk as a self-contained
experiment folder.

Folder layout (written under `root_dir`):
    YYYY-MM-DD_HHMMSS_<safe-name>/
        experiment.json     name, comments, start_ts, stop_ts, processor,
                            engine_config snapshot, hardware config
        sweep_log.jsonl     one MeasurementCommand per line
        snapshots.jsonl     one EngineSnapshot summary per line
        heatmap/NNN.npy     periodic occupancy probability grid dumps
        iq/NNN.npz          per-capture int16 interleaved I/Q
        INDEX.json          written on clean stop; absence means interrupted

All disk I/O happens on a background thread. The engine thread hands data
to a bounded queue; if the queue is full the frame is dropped with a warning
(never silent loss).
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spectrum_engine.iq_source import IQCapture
    from spectrum_engine.engine import EngineSnapshot
    from spectrum_engine.scheduler import MeasurementCommand


# ---------------------------------------------------------------------------
# RecordingOptions
# ---------------------------------------------------------------------------

@dataclass
class RecordingOptions:
    """User-facing options set in the ExperimentDialog."""
    name: str
    comment: str = ""
    max_duration_s: float = 300.0       # 0 = no limit
    heatmap_snapshot_hz: float = 1.0    # how often to dump the occupancy grid
    max_queue_frames: int = 256         # back-pressure limit


# ---------------------------------------------------------------------------
# ExperimentRecorder
# ---------------------------------------------------------------------------

_SENTINEL = object()   # signals the I/O thread to flush and exit


def _safe_name(name: str) -> str:
    """Convert a free-form experiment name to a folder-safe slug."""
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:60] or "experiment"


class ExperimentRecorder:
    """
    Attach to a SpectrumEngine to record a hardware session.

    Usage::

        opts = RecordingOptions(name="fpv-rooftop", comment="5.8 GHz test")
        rec = ExperimentRecorder("experiments/", adc_full_scale=2048.0)
        rec.start(opts, processor_name="classic", engine_cfg_dict={...},
                  hw_cfg_dict={...})
        engine.attach_recorder(rec)
        ...
        rec.stop()      # finalises INDEX.json

    The engine calls on_capture() and on_snapshot() on the engine thread;
    all disk I/O is serialised through a background queue thread.
    """

    def __init__(
        self,
        root_dir: str = "experiments",
        adc_full_scale: float = 2048.0,
    ) -> None:
        self._root_dir = root_dir
        self._adc_full_scale = float(adc_full_scale)

        self._folder: Optional[str] = None
        self._iq_dir: Optional[str] = None
        self._heatmap_dir: Optional[str] = None

        self._opts: Optional[RecordingOptions] = None
        self._start_ts: float = 0.0
        self._iq_index: int = 0
        self._heatmap_index: int = 0
        self._snapshot_index: int = 0
        self._last_heatmap_ts: float = 0.0

        self._sweep_log_fh = None
        self._snapshot_log_fh = None

        self._q: queue.Queue = queue.Queue()
        self._io_thread: Optional[threading.Thread] = None
        self._running: bool = False

        self._dropped: int = 0
        self._written_bytes: int = 0

        # For INDEX.json
        self._processor_name: str = ""
        self._engine_cfg: Dict[str, Any] = {}
        self._hw_cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        opts: RecordingOptions,
        processor_name: str = "",
        engine_cfg_dict: Optional[Dict[str, Any]] = None,
        hw_cfg_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create the experiment folder and start the I/O thread.

        Returns the absolute path of the experiment folder.
        """
        if self._running:
            raise RuntimeError("Recorder already running; call stop() first.")

        self._opts = opts
        self._processor_name = processor_name
        self._engine_cfg = engine_cfg_dict or {}
        self._hw_cfg = hw_cfg_dict or {}
        self._start_ts = time.time()
        self._iq_index = 0
        self._heatmap_index = 0
        self._snapshot_index = 0
        self._last_heatmap_ts = self._start_ts
        self._dropped = 0
        self._written_bytes = 0

        # Build folder path
        ts_str = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(self._start_ts))
        slug = _safe_name(opts.name)
        folder_name = f"{ts_str}_{slug}"
        self._folder = os.path.join(self._root_dir, folder_name)
        self._iq_dir = os.path.join(self._folder, "iq")
        self._heatmap_dir = os.path.join(self._folder, "heatmap")
        os.makedirs(self._iq_dir, exist_ok=True)
        os.makedirs(self._heatmap_dir, exist_ok=True)

        self._sweep_log_fh = open(
            os.path.join(self._folder, "sweep_log.jsonl"), "w", encoding="utf-8"
        )
        self._snapshot_log_fh = open(
            os.path.join(self._folder, "snapshots.jsonl"), "w", encoding="utf-8"
        )

        self._running = True
        self._io_thread = threading.Thread(
            target=self._io_worker, daemon=True, name="exp-recorder-io"
        )
        self._io_thread.start()

        return self._folder

    def on_capture(
        self,
        capture: "IQCapture",
        cmd: "MeasurementCommand",
        adc_full_scale: Optional[float] = None,
    ) -> None:
        """Called from the engine thread after each IQCapture.

        Drops the frame (with a counter increment) if the queue is full.
        """
        if not self._running:
            return

        # Check max_duration
        if (self._opts and self._opts.max_duration_s > 0
                and (time.time() - self._start_ts) > self._opts.max_duration_s):
            return

        scale = adc_full_scale or self._adc_full_scale
        try:
            self._q.put_nowait(("capture", capture, cmd, scale))
        except queue.Full:
            self._dropped += 1
            if self._dropped % 10 == 1:
                print(
                    f"[ExperimentRecorder] WARNING: queue full, "
                    f"dropped {self._dropped} frames so far"
                )

    def on_snapshot(self, snap: "EngineSnapshot") -> None:
        """Called from the engine thread after each EngineSnapshot."""
        if not self._running:
            return
        try:
            self._q.put_nowait(("snapshot", snap))
        except queue.Full:
            pass  # snapshot drops are silent — they're optional summaries

    def stop(self) -> Optional[str]:
        """Flush the queue, write INDEX.json, close file handles.

        Returns the experiment folder path, or None if not running.
        """
        if not self._running:
            return None
        self._running = False
        self._q.put(_SENTINEL)
        if self._io_thread:
            self._io_thread.join(timeout=30)
        return self._folder

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def folder(self) -> Optional[str]:
        return self._folder

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def elapsed_s(self) -> float:
        return time.time() - self._start_ts if self._running else 0.0

    @property
    def written_bytes(self) -> int:
        return self._written_bytes

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    # ------------------------------------------------------------------
    # I/O worker (background thread)
    # ------------------------------------------------------------------

    def _io_worker(self) -> None:
        """Drain the queue and write to disk until the sentinel arrives."""
        try:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    break
                kind = item[0]
                if kind == "capture":
                    _, capture, cmd, scale = item
                    self._write_capture(capture, cmd, scale)
                elif kind == "snapshot":
                    _, snap = item
                    self._write_snapshot(snap)
        except Exception as exc:
            print(f"[ExperimentRecorder] I/O worker error: {exc}")
        finally:
            self._finalise()

    def _write_capture(
        self,
        capture: "IQCapture",
        cmd: "MeasurementCommand",
        scale: float,
    ) -> None:
        # --- IQ as int16 interleaved I/Q ---
        flat = capture.omni_iq.ravel().astype(np.complex64)
        i_vals = np.clip(flat.real * scale, -32768, 32767).astype(np.int16)
        q_vals = np.clip(flat.imag * scale, -32768, 32767).astype(np.int16)
        samples_i16 = np.empty(len(flat) * 2, dtype=np.int16)
        samples_i16[0::2] = i_vals
        samples_i16[1::2] = q_vals

        idx = self._iq_index
        self._iq_index += 1
        iq_path = os.path.join(self._iq_dir, f"{idx:06d}.npz")
        np.savez_compressed(
            iq_path,
            samples=samples_i16,
            frames=np.array(capture.omni_iq.shape[0], dtype=np.int32),
            fft_size=np.array(capture.omni_iq.shape[1], dtype=np.int32),
            center_hz=np.array(capture.center_hz, dtype=np.float64),
            bandwidth_hz=np.array(capture.bandwidth_hz, dtype=np.float64),
            timestamp=np.array(capture.timestamp, dtype=np.float64),
            adc_full_scale=np.array(scale, dtype=np.float64),
        )
        self._written_bytes += os.path.getsize(iq_path)

        # --- Sweep log ---
        log_entry = {
            "ts": capture.timestamp,
            "center_hz": cmd.center_hz,
            "bandwidth_hz": cmd.bandwidth_hz,
            "fft_size": cmd.fft_size,
            "num_frames": cmd.num_frames,
            "dwell_s": cmd.dwell_s,
            "target_type": str(cmd.target_type),
            "is_fine": cmd.is_fine,
        }
        self._sweep_log_fh.write(json.dumps(log_entry) + "\n")
        self._sweep_log_fh.flush()

    def _write_snapshot(self, snap: "EngineSnapshot") -> None:
        now = time.time()

        # Periodic heatmap dump
        hm_interval = (1.0 / self._opts.heatmap_snapshot_hz
                       if self._opts and self._opts.heatmap_snapshot_hz > 0
                       else 1.0)
        if (now - self._last_heatmap_ts) >= hm_interval:
            hm_path = os.path.join(
                self._heatmap_dir, f"{self._heatmap_index:06d}.npy"
            )
            np.save(hm_path, snap.occupancy_probs)
            self._written_bytes += os.path.getsize(hm_path)
            self._heatmap_index += 1
            self._last_heatmap_ts = now

        # Snapshot summary line
        detection_list = []
        for grp in snap.groups:
            detection_list.append({
                "start_hz": float(grp.start_hz),
                "stop_hz": float(grp.stop_hz),
            })
        summary = {
            "ts": snap.timestamp,
            "scan_count": snap.scan_count,
            "track_count": len(snap.tracks),
            "groups": detection_list,
        }
        self._snapshot_log_fh.write(json.dumps(summary) + "\n")
        self._snapshot_index += 1

        if self._snapshot_index % 50 == 0:
            self._snapshot_log_fh.flush()

    def _finalise(self) -> None:
        stop_ts = time.time()

        # Close log file handles
        for fh in (self._sweep_log_fh, self._snapshot_log_fh):
            try:
                if fh:
                    fh.flush()
                    fh.close()
            except Exception:
                pass

        if not self._folder:
            return

        # Write experiment.json
        exp_meta = {
            "name": self._opts.name if self._opts else "",
            "comment": self._opts.comment if self._opts else "",
            "start_ts": self._start_ts,
            "stop_ts": stop_ts,
            "duration_s": stop_ts - self._start_ts,
            "processor": self._processor_name,
            "iq_dtype": "int16",
            "iq_frames": self._iq_index,
            "heatmap_frames": self._heatmap_index,
            "dropped_frames": self._dropped,
            "engine_config": self._engine_cfg,
            "hardware": self._hw_cfg,
        }
        exp_path = os.path.join(self._folder, "experiment.json")
        with open(exp_path, "w", encoding="utf-8") as fh:
            json.dump(exp_meta, fh, indent=2)

        # Write INDEX.json (presence = clean stop)
        index = {
            "iq_frames": self._iq_index,
            "heatmap_frames": self._heatmap_index,
            "snapshot_frames": self._snapshot_index,
            "written_bytes": self._written_bytes,
            "dropped_frames": self._dropped,
            "complete": True,
        }
        idx_path = os.path.join(self._folder, "INDEX.json")
        with open(idx_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2)
