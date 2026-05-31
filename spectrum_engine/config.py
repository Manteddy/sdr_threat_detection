"""
Configuration loader for the Adaptive Spectrum Sensing Engine.

Loads engine_config.yaml from the project root (next to this package).
Falls back to embedded defaults when PyYAML is not installed or the file
is missing, so the engine always starts cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FrequencyRangeCfg:
    start_hz: float = 1.2e9
    stop_hz: float = 6.0e9
    base_cell_bw_hz: float = 20e6


@dataclass
class HardwareCfg:
    default_bandwidth_hz: float = 20e6
    sample_rate_hz: float = 20e6
    fft_size_base: int = 512
    adc_full_scale: float = 2048.0
    pll_settle_s: float = 0.0008
    overlap_fraction: float = 0.0


@dataclass
class CoarseScanCfg:
    fft_size: int = 512
    num_frames: int = 2
    dwell_s: float = 0.003


@dataclass
class FineScanCfg:
    fft_size: int = 4096
    num_frames: int = 16
    dwell_s: float = 0.030
    welch_overlap: float = 0.5


@dataclass
class CfarCfg:
    guard_cells: int = 4
    training_cells: int = 24
    threshold_offset_db: float = 6.0
    min_region_bw_hz: float = 100_000.0


@dataclass
class SchedulerCfg:
    tracked_revisit_s: float = 0.05
    active_revisit_s: float = 0.10
    suspect_revisit_s: float = 0.20
    default_max_revisit_s: float = 5.0
    priority_max_revisit_s: float = 0.5
    short_coarse_dwell_s: float = 0.001
    coarse_dwell_s: float = 0.003
    fine_dwell_s: float = 0.030
    active_dwell_s: float = 0.020
    track_dwell_s: float = 0.010


@dataclass
class StatesCfg:
    suspect_prob_threshold: float = 0.35
    active_prob_threshold: float = 0.65
    confirmed_track_prob_threshold: float = 0.70
    quiet_prob_threshold: float = 0.15
    node_zoom_prob_threshold: float = 0.35
    node_zoom_delta_threshold_db: float = 8.0
    node_uncertainty_threshold: float = 0.7
    occupied_bin_fraction_threshold: float = 0.15


@dataclass
class PriorityBandCfg:
    name: str = ""
    start_hz: float = 0.0
    stop_hz: float = 0.0
    priority_weight: float = 1.0
    max_revisit_s: float = 5.0


@dataclass
class BayesianCfg:
    enabled: bool = False
    weight: float = 1.0


@dataclass
class InferenceCfg:
    enabled: bool = False
    weight: float = 1.0


@dataclass
class ExperimentsCfg:
    root: str = "experiments"
    heatmap_snapshot_hz: float = 1.0
    max_queue_frames: int = 256


@dataclass
class EngineConfig:
    frequency_range: FrequencyRangeCfg = field(default_factory=FrequencyRangeCfg)
    hardware: HardwareCfg = field(default_factory=HardwareCfg)
    coarse_scan: CoarseScanCfg = field(default_factory=CoarseScanCfg)
    fine_scan: FineScanCfg = field(default_factory=FineScanCfg)
    cfar: CfarCfg = field(default_factory=CfarCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    states: StatesCfg = field(default_factory=StatesCfg)
    priority_bands: List[PriorityBandCfg] = field(default_factory=list)
    bayesian: BayesianCfg = field(default_factory=BayesianCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    experiments: ExperimentsCfg = field(default_factory=ExperimentsCfg)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _nested_get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def load_config(yaml_path: str | None = None) -> EngineConfig:
    """
    Load EngineConfig from a YAML file.

    Search order for the file path:
      1. Explicit `yaml_path` argument.
      2. engine_config.yaml next to the spectrum_engine package.
      3. engine_config.yaml in the current working directory.

    Returns embedded defaults if the file cannot be read or PyYAML is absent.
    """
    raw: dict = {}

    if yaml_path is None:
        # Look one directory up from this file (the project root)
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(os.path.dirname(here), "engine_config.yaml"),
            os.path.join(os.getcwd(), "engine_config.yaml"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                yaml_path = c
                break

    if yaml_path and os.path.isfile(yaml_path):
        try:
            import yaml  # type: ignore
            with open(yaml_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except ImportError:
            pass
        except Exception:
            pass

    cfg = EngineConfig()

    fr = raw.get("frequency_range", {})
    cfg.frequency_range = FrequencyRangeCfg(
        start_hz=float(fr.get("start_hz", cfg.frequency_range.start_hz)),
        stop_hz=float(fr.get("stop_hz", cfg.frequency_range.stop_hz)),
        base_cell_bw_hz=float(fr.get("base_cell_bw_hz", cfg.frequency_range.base_cell_bw_hz)),
    )

    hw = raw.get("hardware", {})
    cfg.hardware = HardwareCfg(
        default_bandwidth_hz=float(hw.get("default_bandwidth_hz", cfg.hardware.default_bandwidth_hz)),
        sample_rate_hz=float(hw.get("sample_rate_hz", cfg.hardware.sample_rate_hz)),
        fft_size_base=int(hw.get("fft_size_base", cfg.hardware.fft_size_base)),
        adc_full_scale=float(hw.get("adc_full_scale", cfg.hardware.adc_full_scale)),
        pll_settle_s=float(hw.get("pll_settle_s", cfg.hardware.pll_settle_s)),
        overlap_fraction=float(hw.get("overlap_fraction", cfg.hardware.overlap_fraction)),
    )

    cs = raw.get("coarse_scan", {})
    cfg.coarse_scan = CoarseScanCfg(
        fft_size=int(cs.get("fft_size", cfg.coarse_scan.fft_size)),
        num_frames=int(cs.get("num_frames", cfg.coarse_scan.num_frames)),
        dwell_s=float(cs.get("dwell_s", cfg.coarse_scan.dwell_s)),
    )

    fs = raw.get("fine_scan", {})
    cfg.fine_scan = FineScanCfg(
        fft_size=int(fs.get("fft_size", cfg.fine_scan.fft_size)),
        num_frames=int(fs.get("num_frames", cfg.fine_scan.num_frames)),
        dwell_s=float(fs.get("dwell_s", cfg.fine_scan.dwell_s)),
        welch_overlap=float(fs.get("welch_overlap", cfg.fine_scan.welch_overlap)),
    )

    cf = raw.get("cfar", {})
    cfg.cfar = CfarCfg(
        guard_cells=int(cf.get("guard_cells", cfg.cfar.guard_cells)),
        training_cells=int(cf.get("training_cells", cfg.cfar.training_cells)),
        threshold_offset_db=float(cf.get("threshold_offset_db", cfg.cfar.threshold_offset_db)),
        min_region_bw_hz=float(cf.get("min_region_bw_hz", cfg.cfar.min_region_bw_hz)),
    )

    sc = raw.get("scheduler", {})
    cfg.scheduler = SchedulerCfg(
        tracked_revisit_s=float(sc.get("tracked_revisit_s", cfg.scheduler.tracked_revisit_s)),
        active_revisit_s=float(sc.get("active_revisit_s", cfg.scheduler.active_revisit_s)),
        suspect_revisit_s=float(sc.get("suspect_revisit_s", cfg.scheduler.suspect_revisit_s)),
        default_max_revisit_s=float(sc.get("default_max_revisit_s", cfg.scheduler.default_max_revisit_s)),
        priority_max_revisit_s=float(sc.get("priority_max_revisit_s", cfg.scheduler.priority_max_revisit_s)),
        short_coarse_dwell_s=float(sc.get("short_coarse_dwell_s", cfg.scheduler.short_coarse_dwell_s)),
        coarse_dwell_s=float(sc.get("coarse_dwell_s", cfg.scheduler.coarse_dwell_s)),
        fine_dwell_s=float(sc.get("fine_dwell_s", cfg.scheduler.fine_dwell_s)),
        active_dwell_s=float(sc.get("active_dwell_s", cfg.scheduler.active_dwell_s)),
        track_dwell_s=float(sc.get("track_dwell_s", cfg.scheduler.track_dwell_s)),
    )

    st = raw.get("states", {})
    cfg.states = StatesCfg(
        suspect_prob_threshold=float(st.get("suspect_prob_threshold", cfg.states.suspect_prob_threshold)),
        active_prob_threshold=float(st.get("active_prob_threshold", cfg.states.active_prob_threshold)),
        confirmed_track_prob_threshold=float(st.get("confirmed_track_prob_threshold", cfg.states.confirmed_track_prob_threshold)),
        quiet_prob_threshold=float(st.get("quiet_prob_threshold", cfg.states.quiet_prob_threshold)),
        node_zoom_prob_threshold=float(st.get("node_zoom_prob_threshold", cfg.states.node_zoom_prob_threshold)),
        node_zoom_delta_threshold_db=float(st.get("node_zoom_delta_threshold_db", cfg.states.node_zoom_delta_threshold_db)),
        node_uncertainty_threshold=float(st.get("node_uncertainty_threshold", cfg.states.node_uncertainty_threshold)),
        occupied_bin_fraction_threshold=float(st.get("occupied_bin_fraction_threshold", cfg.states.occupied_bin_fraction_threshold)),
    )

    pb_list = raw.get("priority_bands", [])
    cfg.priority_bands = [
        PriorityBandCfg(
            name=pb.get("name", ""),
            start_hz=float(pb.get("start_hz", 0)),
            stop_hz=float(pb.get("stop_hz", 0)),
            priority_weight=float(pb.get("priority_weight", 1.0)),
            max_revisit_s=float(pb.get("max_revisit_s", cfg.scheduler.priority_max_revisit_s)),
        )
        for pb in pb_list
    ]

    # Default priority bands when no file or no bands defined
    if not cfg.priority_bands:
        cfg.priority_bands = [
            PriorityBandCfg("1.2G", 1.2e9, 1.35e9, 1.5, 1.0),
            PriorityBandCfg("2.4G", 2.3e9, 2.5e9, 2.0, 0.5),
            PriorityBandCfg("5.8G", 5.65e9, 5.95e9, 2.0, 0.5),
        ]

    by = raw.get("bayesian", {})
    cfg.bayesian = BayesianCfg(
        enabled=bool(by.get("enabled", False)),
        weight=float(by.get("weight", 1.0)),
    )

    inf = raw.get("inference", {})
    cfg.inference = InferenceCfg(
        enabled=bool(inf.get("enabled", False)),
        weight=float(inf.get("weight", 1.0)),
    )

    ex = raw.get("experiments", {})
    cfg.experiments = ExperimentsCfg(
        root=str(ex.get("root", cfg.experiments.root)),
        heatmap_snapshot_hz=float(ex.get("heatmap_snapshot_hz",
                                         cfg.experiments.heatmap_snapshot_hz)),
        max_queue_frames=int(ex.get("max_queue_frames",
                                    cfg.experiments.max_queue_frames)),
    )

    return cfg
