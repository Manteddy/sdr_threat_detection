"""
Adaptive Hierarchical Multiband Spectrum Sensing Engine — main orchestrator.

SpectrumEngine owns all state (cells, nodes, tracks, scheduler, telemetry)
and exposes a single step() method.  The step() method implements the full
processing loop from spec section 15.

EngineSnapshot is a lightweight read-only view of the engine's current state
that is safe to pass across thread boundaries (the Qt main thread reads it
while the worker thread continues scanning).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING

import numpy as np


class EngineStage(IntEnum):
    """Four-stage operating ladder.

    Strict superset: each stage runs everything from the stages below.
    The GUI drives this via `SpectrumEngine.set_stage`.

      * IDLE     — worker thread stopped; `step()` is never called.
      * RECEIVE  — scheduler picks + tune + acquire raw IQ, then drop.
                   No PSD, no display update, no cell-state work, no
                   processor. Useful as a diagnostic that data is flowing.
      * PROCESS  — RECEIVE + PSD compute + display buffer + per-cell
                   baseline / occupancy / state machine + activity groups.
                   Spectrum + waterfall come alive; no tracks, no
                   classifications.
      * CLASSIFY — PROCESS + `processor.process_fine` on fine scans +
                   TrackManager updates. Full detection pipeline.
    """
    IDLE = 0
    RECEIVE = 1
    PROCESS = 2
    CLASSIFY = 3

from .config import EngineConfig, load_config
from .spectrum_grid import (
    SpectrumCell, build_grid, map_measurement_to_cells,
    mark_unsupported_cells, cells_to_occupancy_array, cells_to_freq_centers,
    CellState,
)
from .iq_source import IQCapture
from .signal_reader import SignalReader
from .psd import coarse_psd_db, fine_welch_db, freq_axis_hz, channel_psd
from .detectors import (
    compute_coarse_measurement, compute_fine_measurement,
    CoarseMeasurement, FineMeasurement,
)
from .baseline import update_baseline, update_energy_delta, update_uncertainty

# Pluggable signal processor — defaults to ClassicProcessor which wraps
# the existing compute_fine_measurement so behaviour is unchanged unless
# the caller (typically the GUI Proc combo) swaps it out.
from signal_pipeline import default_processor, SignalProcessor  # noqa: E402
from .occupancy import (
    update_occupancy_probability, update_cell_state, build_activity_groups,
    ActivityGroup,
)
from .hierarchy import (
    SpectrumNode, build_hierarchy, update_hierarchy_nodes,
)
from .tracks import SignalTrack, TrackManager, TrackState
from .scheduler import Scheduler, MeasurementCommand, TargetType
from .telemetry import TelemetryLogger


# Occupancy is measured against the tracked baseline (a slow noise-floor EWMA).
# A single FFT of pure noise has ~10% of bins >baseline+8 dB and a lone bin
# ~15 dB above it, so the margins must clear those statistics:
#   _OCC_MARGIN_DB    : "occupied" (wideband) margin -> noise fraction ~0.3%.
#   _STRONG_MARGIN_DB : "strong bin" margin for narrowband carriers; noise
#                       essentially never reaches it, so a count >=3 means signal.
_OCC_MARGIN_DB: float = 12.0
_STRONG_MARGIN_DB: float = 18.0


# ---------------------------------------------------------------------------
# EngineSnapshot — passed to the Qt UI thread
# ---------------------------------------------------------------------------

@dataclass
class EngineSnapshot:
    """
    Immutable view of engine state at one point in time.
    Arrays are copies; lists are shallow copies of dataclass objects.
    """
    timestamp: float

    # Full-spectrum occupancy (shape: num_cells)
    cell_freq_centers_hz: np.ndarray  # float64
    occupancy_probs: np.ndarray       # float32
    cell_states: List[str]            # one entry per cell

    # Last measured PSD (omni channel)
    last_psd_db: np.ndarray           # float32
    last_psd_freq_hz: np.ndarray      # float64
    last_center_hz: float
    last_bandwidth_hz: float

    # Activity groups and tracks
    groups: List[ActivityGroup]
    tracks: List[SignalTrack]

    # Scheduler metadata
    last_command: Optional[MeasurementCommand]
    scan_count: int

    # Coverage health
    max_cell_age_s: float             # worst-case unscanned cell age
    overdue_cell_count: int           # cells past max_revisit_interval

    # Per-cell seconds since last scan (1e9 if never scanned). Drives the
    # scan-coverage map robustly regardless of command/target type.
    cell_scan_age_s: np.ndarray = None  # float32, shape (num_cells,)


# ---------------------------------------------------------------------------
# SpectrumEngine
# ---------------------------------------------------------------------------

class SpectrumEngine:
    """
    Full stateful spectrum sensing engine (Phases 1-3).

    Usage (from SweepWorker thread):

        engine = SpectrumEngine(cfg)
        engine.attach_backend(SignalReader(PyAdiIQSource(sdr_obj, cfg), cfg))
        while running:
            snapshot = engine.step(now=time.monotonic())
            # emit snapshot to Qt
    """

    def __init__(
        self,
        cfg: Optional[EngineConfig] = None,
        telemetry_enabled: bool = True,
        telemetry_dir: str = "spectrum_logs",
    ) -> None:
        self.cfg: EngineConfig = cfg or load_config()

        # Build cell grid
        self.cells: List[SpectrumCell] = build_grid(self.cfg)

        # Build hierarchy
        self.nodes: List[SpectrumNode] = build_hierarchy(self.cells, self.cfg)

        # Track manager
        self.track_mgr = TrackManager(self.cfg)

        # Scheduler
        self.scheduler = Scheduler(self.cfg)

        # Telemetry
        self.telemetry = TelemetryLogger(
            log_dir=telemetry_dir,
            enabled=telemetry_enabled,
        )

        # Runtime state
        self._backend: Optional[SignalReader] = None
        self._processor: SignalProcessor = default_processor()
        # Operating stage (see EngineStage docstring). Default CLASSIFY
        # keeps existing callers (headless smoke scripts) running the full
        # pipeline without opt-in. The GUI sets IDLE at construction and
        # advances explicitly through the stage ladder.
        self.stage: EngineStage = EngineStage.CLASSIFY
        self._scan_count: int = 0
        self._last_snapshot: Optional[EngineSnapshot] = None
        self._last_cmd: Optional[MeasurementCommand] = None
        self._hw_max_hz: float = float("inf")
        self._last_det_log: float = 0.0  # throttle for detection-value dumps
        # Active scan window (defaults to the full configured range)
        self._active_start_hz: float = self.cfg.frequency_range.start_hz
        self._active_stop_hz: float = self.cfg.frequency_range.stop_hz

        # Rolling display buffer (maps last full-spectrum PSD for waterfall)
        n = len(self.cells)
        self._display_psd = np.full(n * self.cfg.coarse_scan.fft_size,
                                    -140.0, dtype=np.float32)
        self._display_freq = np.zeros(n * self.cfg.coarse_scan.fft_size, dtype=np.float64)
        self._display_initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _debug_log(self, where: str, exc: Exception, cmd=None) -> None:
        """Append a capture/processing error with traceback to a debug file.

        Limited to the first ~20 entries so a persistent failure does not grow
        the file without bound. Used to diagnose why scans may stall.
        """
        try:
            count = getattr(self, "_debug_count", 0)
            if count >= 20:
                return
            self._debug_count = count + 1
            import os, traceback
            os.makedirs("spectrum_logs", exist_ok=True)
            with open(os.path.join("spectrum_logs", "engine_debug.log"), "a") as fh:
                fh.write(f"[{where}] {type(exc).__name__}: {exc}\n")
                if cmd is not None:
                    fh.write(f"  cmd: type={getattr(cmd,'target_type',None)} "
                             f"center={getattr(cmd,'center_hz',None)} "
                             f"frames={getattr(cmd,'num_frames',None)}\n")
                fh.write(traceback.format_exc() + "\n")
        except Exception:
            pass

    def attach_backend(self, backend: SignalReader) -> None:
        """Attach a `SignalReader` and mark cells outside hardware range.

        Argument named `backend` for backward compatibility; the engine
        treats whatever it receives as the single capture function it
        calls every step.
        """
        self._backend = backend
        limits = backend.get_limits()
        self._hw_max_hz = limits.max_hz
        mark_unsupported_cells(self.cells, limits.max_hz)

    def set_processor(self, processor: SignalProcessor) -> None:
        """Swap the fine-scan signal processor at runtime.

        Safe to call while the engine loop is running — processors are
        stateless across calls, so a swap just changes which one handles
        the next fine measurement.
        """
        self._processor = processor

    def get_processor(self) -> SignalProcessor:
        return self._processor

    def set_stage(self, stage: EngineStage) -> None:
        """Switch operating stage. Safe to call live from the GUI thread.

        Each `step()` re-reads `self.stage`, so the change takes effect
        at the next scan. The GUI is responsible for any state cleanup
        that should happen on step-down transitions (drop tracks when
        leaving CLASSIFY; reset cells when leaving PROCESS).
        """
        self.stage = EngineStage(int(stage))

    # Back-compat alias used during the migration. New callers should
    # call `set_stage(EngineStage.CLASSIFY | PROCESS)` directly.
    def set_detection_enabled(self, enabled: bool) -> None:
        self.stage = EngineStage.CLASSIFY if enabled else EngineStage.PROCESS

    def reset_detection_state(self) -> None:
        """Reset all cells to UNKNOWN, drop every track, clear groups.

        Called by the GUI when Detect toggles OFF (per the answer in the
        feature plan). Leaves baselines and noise floor estimates in
        place — those are observations about the channel, not detections.
        """
        for cell in self.cells:
            if cell.state == CellState.UNSUPPORTED:
                continue
            cell.state = CellState.UNKNOWN
            cell.occupancy_prob = 0.01
            # log-odds form must follow the prob field
            cell.occupancy_logodds = float(np.log(0.01 / 0.99))
            cell.energy_delta_db = 0.0
            cell.uncertainty = 1.0
            cell.coarse_suspicious = False
            cell.last_strong_bins = 0
            cell.last_occ_frac = 0.0
            cell.last_peak_excess_db = 0.0
        self.track_mgr.clear_all()

    def set_active_range(self, start_hz: float, stop_hz: float) -> None:
        """
        Restrict active scanning to [start_hz, stop_hz].

        Cells whose center falls outside the window (or above the hardware
        maximum) are marked UNSUPPORTED so the scheduler skips them entirely.
        Cells that re-enter the window are reset to UNKNOWN and flagged as
        never-scanned, so the engine immediately re-sweeps the new range.
        """
        from .spectrum_grid import CellState

        self._active_start_hz = float(start_hz)
        self._active_stop_hz = float(stop_hz)

        for cell in self.cells:
            in_window = (cell.center_hz >= start_hz) and (cell.center_hz <= stop_hz)
            hw_ok = cell.center_hz <= self._hw_max_hz
            if in_window and hw_ok:
                if cell.state == CellState.UNSUPPORTED:
                    # Re-enabled: force a fresh sweep of this cell
                    cell.state = CellState.UNKNOWN
                    cell.last_scan_time = 0.0
            else:
                cell.state = CellState.UNSUPPORTED

    # ------------------------------------------------------------------
    # Main step — runs one complete measurement iteration
    # ------------------------------------------------------------------

    def step(self, now: float) -> EngineSnapshot:
        """
        Execute one measurement cycle (spec section 15):
          1. Update scores / uncertainties
          2. Pick next measurement
          3. Capture IQ
          4. Compute PSD
          5. Update cells
          6. Group cells
          7. Run fine detection if command is fine
          8. Update tracks
          9. Publish snapshot
        """
        cfg = self.cfg

        # --- 1. Update scores and uncertainties ---
        for cell in self.cells:
            update_uncertainty(cell, now)
        update_hierarchy_nodes(self.nodes, now)
        groups = build_activity_groups(
            self.cells,
            prob_threshold=cfg.states.suspect_prob_threshold,
            now=now,
        )
        self.scheduler.update_scores(self.cells, now)

        # --- 2. Pick next measurement ---
        cmd = self.scheduler.choose_next_measurement(
            self.cells, self.track_mgr.active_tracks, groups, now,
        )
        self._last_cmd = cmd

        # --- 3. Capture IQ ---
        if self._backend is None:
            return self._make_snapshot(now, groups)

        try:
            capture: IQCapture = self._backend.capture(
                center_hz=cmd.center_hz,
                bandwidth_hz=cmd.bandwidth_hz,
                dwell_s=cmd.dwell_s,
                num_frames=cmd.num_frames,
            )
        except Exception as exc:
            self._debug_log("capture", exc, cmd)
            return self._make_snapshot(now, groups)

        capture_time = time.monotonic()

        try:
            return self._process_capture(cmd, capture, capture_time, groups, cfg)
        except Exception as exc:
            self._debug_log("process", exc, cmd)
            return self._make_snapshot(now, groups)

    def _process_capture(self, cmd, capture, capture_time, groups, cfg):
        """PSD → cell update → grouping → fine detect → tracks → snapshot."""
        # --- 4. Compute PSD ---
        coarse_meas: Optional[CoarseMeasurement] = None
        fine_meas: Optional[FineMeasurement] = None

        if cmd.is_fine:
            psd_o, psd_d, f_ax = channel_psd(
                capture, fine=True,
                fine_fft_size=cfg.fine_scan.fft_size,
                fine_overlap=cfg.fine_scan.welch_overlap,
            )
        else:
            psd_o, psd_d, f_ax = channel_psd(
                capture, fine=False,
                coarse_fft_size=cfg.coarse_scan.fft_size,
            )

        # Use omni channel as primary
        psd_primary = psd_o

        # Excise the DC spike: PlutoSDR leaks the LO into the center FFT bin
        # on every capture. The Blackman window spreads the spike into the
        # two adjacent bins as sidelobes (~-20 dB, still -48 dBFS on a
        # -28 dBFS spike vs a noise floor at -105 dBFS). The original 3-bin
        # window (mid-1 to mid+1) left those sidelobes in, inflating peak_db
        # by ~57 dB and permanently saturating energy_delta across the whole
        # band. Widened to 5 bins (±2) to cover the first sidelobe pair.
        if psd_primary.size > 8:
            mid = psd_primary.size // 2
            med = float(np.median(psd_primary))
            psd_primary[max(0, mid - 2):mid + 3] = med

        # Stage RECEIVE: show the raw single-FFT spectrum without any
        # processing — no DC excision, no Welch averaging, no cell state,
        # no baseline, no processor. The display is intentionally "dirty":
        # the LO-leakage DC spike is visible, noise is unsuppressed.
        # This is a diagnostic view — "what the ADC actually handed us."
        #
        # We recompute a fresh coarse PSD here from the un-excised IQ so
        # the display reflects the true raw capture. The Welch/excised
        # psd_primary computed above (used at PROCESS+) is not touched.
        if self.stage <= EngineStage.RECEIVE:
            raw_psd = coarse_psd_db(
                capture.omni_iq,
                fft_size=cfg.coarse_scan.fft_size,
            )
            raw_f_ax = freq_axis_hz(
                capture.center_hz,
                capture.bandwidth_hz,
                cfg.coarse_scan.fft_size,
            )
            # Excise the DC spike here too. RECEIVE used to show the raw
            # PSD un-touched, but with the scheduler sweeping the band the
            # operator ended up seeing the LO-leakage spike at the centre
            # of every cell (~240 identical -28 dBFS peaks every 20 MHz).
            # That visual forest of artifact peaks drowned out any real
            # signal. The DC spike is a known hardware/sim artifact, not
            # data -- excising it preserves the "raw" intent of RECEIVE
            # (no Welch averaging, no baseline, no detection) while
            # removing the misleading repeated artifact.
            if raw_psd.size > 8:
                mid = raw_psd.size // 2
                med = float(np.median(raw_psd))
                raw_psd[max(0, mid - 2):mid + 3] = med
            self._update_display_buffer(raw_psd, raw_f_ax)
            # Tell the scheduler this window was visited so it advances
            # to the next cell instead of re-scanning the same one forever.
            # Full occupancy / state updates are intentionally skipped.
            #
            # We DO pre-initialise baseline_db for cells seeing their very
            # first scan. Without this, RECEIVE stamps last_scan_time > 0
            # which breaks the "if last_scan_time == 0" guard in
            # update_baseline, causing PROCESS to start the EWMA from the
            # -100 dBFS sentinel instead of the real noise floor. That
            # inflates energy_delta for every cell during the warm-up
            # window and degrades scheduler prioritisation.
            raw_noise = float(np.percentile(raw_psd, 30))
            for cell in map_measurement_to_cells(
                cmd.center_hz, cmd.bandwidth_hz, self.cells
            ):
                if cell.last_scan_time == 0.0:
                    # First visit — snap baseline to real noise floor.
                    cell.baseline_db = raw_noise
                    cell.baseline_var_db = 5.0
                cell.last_scan_time = capture_time
            self._scan_count += 1
            snapshot = self._make_snapshot(capture_time, groups)
            self._last_snapshot = snapshot
            return snapshot

        # --- 5. Update cells touched by this measurement ---
        affected = map_measurement_to_cells(
            cmd.center_hz, cmd.bandwidth_hz, self.cells,
        )

        for cell in affected:
            # Compute a per-cell noise floor from the portion of PSD in its range
            cell_lo = cell.start_hz
            cell_hi = cell.stop_hz
            mask = (f_ax >= cell_lo) & (f_ax < cell_hi)
            if np.any(mask):
                cell_psd = psd_primary[mask]
            else:
                cell_psd = psd_primary
            noise_floor = float(np.percentile(cell_psd, 30))
            peak = float(np.max(cell_psd))

            # Update the slow, signal-resistant baseline FIRST, then measure
            # occupancy *relative to that baseline* (not the current capture's
            # own percentile). This keeps quiet cells dark instead of every cell
            # saturating on the natural ~12 dB FFT-noise peak.
            update_baseline(cell, noise_floor)
            baseline = cell.baseline_db
            excess = cell_psd - baseline
            occ_frac = float(np.mean(excess > _OCC_MARGIN_DB))
            strong_bins = int(np.count_nonzero(excess > _STRONG_MARGIN_DB))

            cell.noise_floor_db = noise_floor
            cell.peak_db = peak
            cell.last_occ_frac = occ_frac
            cell.last_strong_bins = strong_bins
            cell.last_peak_excess_db = float(peak - baseline)
            update_energy_delta(cell, peak)
            update_occupancy_probability(
                cell, occ_frac, strong_bins,
                cfg.states.occupied_bin_fraction_threshold,
            )
            cell.last_scan_time = capture_time
            if cmd.is_fine:
                cell.last_fine_scan_time = capture_time
            else:
                cell.last_coarse_scan_time = capture_time

            update_cell_state(cell, cfg, capture_time)

        # --- 4b. Build coarse measurement for telemetry ---
        if not cmd.is_fine and affected:
            ref_cell = affected[0]
            coarse_meas = compute_coarse_measurement(
                psd_primary, f_ax, capture.timestamp,
                cmd.center_hz, cmd.bandwidth_hz,
                ref_cell.baseline_db, cfg,
            )
            for cell in affected:
                cell.coarse_suspicious = coarse_meas.coarse_suspicious

        # --- 6. Rebuild groups after cell update ---
        groups = build_activity_groups(
            self.cells,
            prob_threshold=cfg.states.suspect_prob_threshold,
            now=capture_time,
        )

        # Stage PROCESS: PSD + display + cell-state machine ran above;
        # we stop short of the processor and TrackManager.
        if self.stage <= EngineStage.PROCESS:
            self._update_display_buffer(psd_primary, f_ax)
            self._scan_count += 1
            snapshot = self._make_snapshot(capture_time, groups)
            self._last_snapshot = snapshot
            return snapshot

        # Stage CLASSIFY: full pipeline.
        # --- 7. Fine detection → region → track update ---
        if cmd.is_fine:
            # Dispatch through the pluggable processor. The default
            # (ClassicProcessor) just calls compute_fine_measurement,
            # preserving previous behaviour byte-for-byte.
            fine_meas = self._processor.process_fine(
                capture=capture,
                psd_db=psd_primary,
                freq_axis_hz=f_ax,
                cfg=cfg,
            )
            self.track_mgr.update_from_regions(fine_meas.detected_regions, capture_time)

        # --- 8. Track maintenance ---
        self.track_mgr.update_track_states(capture_time)
        self.track_mgr.update_cell_states_from_tracks(self.cells)

        # Update display buffer for the measured window
        self._update_display_buffer(psd_primary, f_ax)

        self._scan_count += 1

        # --- 9. Telemetry ---
        self.telemetry.log_measurement(
            cmd, coarse_meas, fine_meas, affected,
            self.track_mgr.active_tracks, capture_time,
        )
        self.telemetry.log_score_dump_if_due(self.cells, capture_time)

        snapshot = self._make_snapshot(capture_time, groups)
        self._last_snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _dump_detection_log(self, now: float) -> None:
        """Periodically append data-level detection values to a log file so a
        persistent "uniform map" issue can be debugged from real numbers."""
        if now - self._last_det_log < 3.0:
            return
        self._last_det_log = now
        try:
            import os
            from .spectrum_grid import CellState
            in_range = [c for c in self.cells if c.state != CellState.UNSUPPORTED]
            if not in_range:
                return
            probs = np.array([c.occupancy_prob for c in in_range], dtype=np.float32)
            top = sorted(in_range, key=lambda c: c.occupancy_prob, reverse=True)[:8]
            os.makedirs("spectrum_logs", exist_ok=True)
            with open(os.path.join("spectrum_logs", "detection.log"), "a") as fh:
                fh.write(
                    f"\n[t={now:8.1f} scans={self._scan_count}] "
                    f"prob min={probs.min():.3f} med={np.median(probs):.3f} "
                    f"max={probs.max():.3f}  active(>0.5)={int((probs>0.5).sum())}/{len(probs)}\n"
                )
                fh.write(f"  {'GHz':>6} {'noise':>7} {'base':>7} {'peak':>7} "
                         f"{'pk-base':>7} {'occf':>6} {'strong':>6} {'prob':>6} {'state':>9}\n")
                for c in top:
                    fh.write(
                        f"  {c.center_hz/1e9:6.2f} {c.noise_floor_db:7.1f} "
                        f"{c.baseline_db:7.1f} {c.peak_db:7.1f} {c.last_peak_excess_db:7.1f} "
                        f"{c.last_occ_frac:6.3f} {c.last_strong_bins:6d} "
                        f"{c.occupancy_prob:6.3f} {str(c.state):>9}\n"
                    )
        except Exception:
            pass

    def _make_snapshot(self, now: float, groups: List[ActivityGroup]) -> EngineSnapshot:
        self._dump_detection_log(now)
        occ = cells_to_occupancy_array(self.cells)
        freq_c = cells_to_freq_centers(self.cells)
        states = [c.state for c in self.cells]

        # Coverage health
        ages = [
            now - c.last_scan_time
            for c in self.cells
            if c.state != CellState.UNSUPPORTED and c.last_scan_time > 0
        ]
        max_age = max(ages, default=0.0)
        overdue = sum(
            1 for c in self.cells
            if c.state != CellState.UNSUPPORTED
            and c.last_scan_time > 0
            and (now - c.last_scan_time) > c.max_revisit_interval_s
        )

        # Per-cell scan age (1e9 = never scanned) for the coverage map
        scan_age = np.array(
            [(now - c.last_scan_time) if c.last_scan_time > 0 else 1e9
             for c in self.cells],
            dtype=np.float32,
        )

        return EngineSnapshot(
            timestamp=now,
            cell_freq_centers_hz=freq_c,
            occupancy_probs=occ,
            cell_states=states,
            last_psd_db=self._display_psd.copy(),
            last_psd_freq_hz=self._display_freq.copy(),
            last_center_hz=self._last_cmd.center_hz if self._last_cmd else 0.0,
            last_bandwidth_hz=self._last_cmd.bandwidth_hz if self._last_cmd else 0.0,
            groups=list(groups),
            tracks=list(self.track_mgr.active_tracks),
            last_command=self._last_cmd,
            scan_count=self._scan_count,
            max_cell_age_s=max_age,
            overdue_cell_count=overdue,
            cell_scan_age_s=scan_age,
        )

    def _update_display_buffer(
        self, psd_db: np.ndarray, f_ax: np.ndarray,
    ) -> None:
        """
        Insert the measured PSD into the rolling full-spectrum display buffer.
        Bins are mapped by frequency so the buffer always covers all cells.
        """
        if not self._display_initialized:
            fft_size = self.cfg.coarse_scan.fft_size
            n_cells = len(self.cells)
            total = n_cells * fft_size
            start_hz = self.cfg.frequency_range.start_hz
            stop_hz = self.cfg.frequency_range.stop_hz
            self._display_freq = np.linspace(start_hz, stop_hz, total, dtype=np.float64)
            self._display_psd = np.full(total, -140.0, dtype=np.float32)
            self._display_initialized = True

        # Find indices in display buffer matching the measured frequency window
        lo = float(f_ax[0])
        hi = float(f_ax[-1])
        mask = (self._display_freq >= lo) & (self._display_freq <= hi)
        n_dest = int(np.sum(mask))
        if n_dest == 0:
            return

        # Interpolate measured PSD onto display grid
        dest_freqs = self._display_freq[mask]
        interp_psd = np.interp(dest_freqs, f_ax, psd_db).astype(np.float32)
        self._display_psd[mask] = interp_psd
