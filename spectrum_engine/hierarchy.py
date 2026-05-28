"""
Hierarchical frequency tree for coarse-to-fine zoom (spec sections 7, 12).

The hierarchy is a scheduler abstraction, not a physical scan order.
Each SpectrumNode aggregates statistics from its child cells and can
signal that a sub-band needs a zoom measurement.

Tree layout (default):
  Level 0: 1.2-6.0 GHz  (1 root node)
  Level 1: 800 MHz blocks  (6 nodes)
  Level 2: 100 MHz blocks
  Level 3: 20 MHz blocks  (== base cells)

Only the root and level-1 nodes are explicitly stored; finer levels
are the cells themselves.  The tree is rebuilt after grid construction.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .spectrum_grid import SpectrumCell
    from .config import EngineConfig


# ---------------------------------------------------------------------------
# SpectrumNode
# ---------------------------------------------------------------------------

@dataclass
class SpectrumNode:
    node_id: int
    start_hz: float
    stop_hz: float
    level: int
    child_nodes: List["SpectrumNode"] = field(default_factory=list)
    child_cells: List["SpectrumCell"] = field(default_factory=list)

    aggregate_energy_delta_db: float = 0.0
    aggregate_occupancy_prob: float = 0.0
    uncertainty: float = 1.0
    last_scan_time: float = 0.0
    state: str = "QUIET"  # QUIET, SUSPECT, ACTIVE

    @property
    def center_hz(self) -> float:
        return (self.start_hz + self.stop_hz) / 2.0

    @property
    def bandwidth_hz(self) -> float:
        return self.stop_hz - self.start_hz


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def build_hierarchy(cells: List["SpectrumCell"], cfg: "EngineConfig") -> List[SpectrumNode]:
    """
    Build a two-level hierarchy over the cell grid.

    Level-1 nodes each span ~800 MHz (or whatever divides the range into ~6 blocks).
    Each level-1 node owns the cells whose center_hz falls in its range.

    Returns the list of level-1 nodes (roots of subtrees).
    """
    fr = cfg.frequency_range
    total_span = fr.stop_hz - fr.start_hz
    # Aim for ~6 level-1 blocks
    block_width = total_span / 6.0
    # Round to nearest 100 MHz
    block_width = round(block_width / 100e6) * 100e6
    block_width = max(block_width, 100e6)

    nodes: List[SpectrumNode] = []
    node_id = 0
    start = fr.start_hz
    while start < fr.stop_hz:
        stop = min(start + block_width, fr.stop_hz)
        node = SpectrumNode(
            node_id=node_id,
            start_hz=start,
            stop_hz=stop,
            level=1,
        )
        # Assign cells
        node.child_cells = [c for c in cells if node.start_hz <= c.center_hz < node.stop_hz]
        nodes.append(node)
        node_id += 1
        start = stop

    return nodes


# ---------------------------------------------------------------------------
# Node aggregation (spec section 7.2)
# ---------------------------------------------------------------------------

def update_hierarchy_nodes(nodes: List[SpectrumNode], now: float) -> None:
    """Recompute aggregate statistics for every node from its child cells."""
    for node in nodes:
        if not node.child_cells:
            continue

        # Joint probability: 1 - product(1 - p_i)
        joint_inactive = 1.0
        for c in node.child_cells:
            joint_inactive *= (1.0 - c.occupancy_prob)
        node.aggregate_occupancy_prob = 1.0 - joint_inactive

        # 90th percentile energy delta
        deltas = [c.energy_delta_db for c in node.child_cells]
        node.aggregate_energy_delta_db = float(np.percentile(deltas, 90))

        # Mean uncertainty
        node.uncertainty = float(np.mean([c.uncertainty for c in node.child_cells]))

        # Most recent scan time
        node.last_scan_time = max(
            (c.last_scan_time for c in node.child_cells if c.last_scan_time > 0),
            default=0.0,
        )

        # Node state
        node.state = _classify_node_state(node)


def _classify_node_state(node: SpectrumNode) -> str:
    from .config import StatesCfg
    p = node.aggregate_occupancy_prob
    # Use fixed thresholds (not cfg-bound here to avoid circular import)
    if p > 0.65:
        return "ACTIVE"
    if p > 0.35:
        return "SUSPECT"
    return "QUIET"


# ---------------------------------------------------------------------------
# Zoom-target generation (spec section 12.1-12.2)
# ---------------------------------------------------------------------------

def get_zoom_worthy_nodes(
    nodes: List[SpectrumNode],
    cfg: "EngineConfig",
) -> List[SpectrumNode]:
    """Return nodes whose aggregate statistics exceed the zoom thresholds."""
    st = cfg.states
    result = []
    for node in nodes:
        if (
            node.aggregate_occupancy_prob > st.node_zoom_prob_threshold
            or node.aggregate_energy_delta_db > st.node_zoom_delta_threshold_db
            or node.uncertainty > st.node_uncertainty_threshold
        ):
            result.append(node)
    result.sort(key=lambda n: n.aggregate_occupancy_prob, reverse=True)
    return result


def generate_zoom_targets(
    node: SpectrumNode,
    cfg: "EngineConfig",
) -> List[tuple]:
    """
    Divide a suspicious node's band into zoom windows.

    Returns list of (center_hz, bandwidth_hz) tuples for coarse sub-scans.
    The windows are sized based on the node's bandwidth (spec section 12.1).
    """
    bw = node.bandwidth_hz
    hw_bw = cfg.hardware.default_bandwidth_hz

    if bw > 100e6:
        split_width = 20e6
    elif bw > 20e6:
        split_width = 5e6
    else:
        split_width = bw

    split_width = max(split_width, hw_bw)

    targets = []
    start = node.start_hz
    while start < node.stop_hz:
        stop = min(start + split_width, node.stop_hz)
        center = (start + stop) / 2.0
        targets.append((center, split_width))
        start = stop

    return targets
