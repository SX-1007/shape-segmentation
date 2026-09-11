"""Shared data contracts for traffic-sign detection and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class SignShape(IntEnum):
    UNKNOWN = 0
    TRIANGLE = 1
    RECTANGLE = 2
    OCTAGON = 3
    CIRCLE = 4
    DIAMOND = 5


class ProposalSource(IntEnum):
    COLOR = 0
    EDGE = 1
    HOUGH = 2


def _empty_points() -> np.ndarray:
    return np.empty((0, 1, 2), dtype=np.int32)


@dataclass(slots=True)
class ShapeInfo:
    shape: SignShape = SignShape.UNKNOWN
    score: float = 0.0
    color_id: int = -1
    contour: np.ndarray = field(default_factory=_empty_points)
    hull: np.ndarray = field(default_factory=_empty_points)
    poly: np.ndarray = field(default_factory=_empty_points)
    ideal: np.ndarray = field(default_factory=_empty_points)
    circle_center: tuple[float, float] = (0.0, 0.0)
    circle_radius: float = 0.0
    ellipse: tuple[tuple[float, float], tuple[float, float], float] = (
        (0.0, 0.0),
        (0.0, 0.0),
        0.0,
    )
    ellipse_valid: bool = False
    box: tuple[tuple[float, float], tuple[float, float], float] = (
        (0.0, 0.0),
        (0.0, 0.0),
        0.0,
    )
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    area: float = 0.0
    perimeter: float = 0.0
    hull_area: float = 0.0
    hull_perimeter: float = 0.0
    circularity: float = 0.0
    solidity: float = 0.0
    extent: float = 0.0
    aspect: float = 0.0
    rim_ratio: float = 0.0
    rim_rescued: bool = False
    circle_fit: float = 0.0
    rectangle_fit: float = 0.0
    triangle_fit: float = 0.0
    vertices_low: int = 0
    vertices: int = 0
    vertices_very_high: int = 0
    touches_border: bool = False
    proposal_source: ProposalSource = ProposalSource.COLOR
    edge_support: float = 0.0
    color_coverage: float = 0.0
    color_capture: float = 0.0
    contrast: float = 0.0
    center_proximity: float = 0.0
    selection_score: float = 0.0
    reject: str = ""


@dataclass(slots=True)
class ShapeParams:
    th_red: float = 0.14
    th_blue: float = 0.13
    th_yellow: float = 0.11
    adapt_rel: float = 0.55
    th_floor: float = 0.05
    yellow_green_tol: float = 0.20
    dark_sum: float = 60.0
    morph_frac: float = 0.020
    morph_min: int = 3
    fill_holes: bool = True
    multi_scale: bool = True
    weld_frac: float = 0.055
    weld_min: int = 7
    weld_iou: float = 0.30
    split_merged: bool = True
    split_open_frac: float = 0.028
    split_open_min: int = 5
    min_area_ratio: float = 0.006
    max_area_ratio: float = 0.95
    min_area: float = 100.0
    min_solidity: float = 0.60
    max_aspect: float = 3.00
    rim_min_solid: float = 0.12
    rim_low: float = 1.45
    rim_high: float = 2.80
    rim_min_circularity: float = 0.65
    epsilon_low: float = 0.010
    epsilon_high: float = 0.030
    epsilon_factor: float = 0.020
    circle_vertices: int = 10
    vertex_weight: float = 0.050
    circularity_weight: float = 0.050
    octagon_bias: float = 0.020
    rectangle_gate: float = 0.90
    triangle_gate: float = 0.65
    octagon_gate: float = 0.80
    min_score: float = 0.20
    inner_min_area: float = 0.10
    inner_max_area: float = 0.70
    inner_min_score: float = 0.85
    selection_size_weight: float = 0.35
    center_prior_strength: float = 0.22
    edge_proposals: bool = True
    hough_circles: bool = False
    edge_close_frac: float = 0.018
    edge_close_min: int = 3
    edge_tolerance_frac: float = 0.018
    min_edge_support: float = 0.34
    min_color_coverage: float = 0.018
    dedupe_iou: float = 0.72
    refine_with_grabcut: bool = True
    grabcut_iterations: int = 5
    grabcut_outer_frac: float = 0.080
    grabcut_inner_frac: float = 0.160
    refine_min_area_ratio: float = 0.72
    refine_max_area_ratio: float = 1.16
    refine_min_span_ratio: float = 0.80
    refine_min_color_recall: float = 0.90
    octagon_chamfer_proposals: bool = True
    grabcut_early_stop: bool = True
    grabcut_use_roi: bool = True


@dataclass(slots=True)
class RefinementDiagnostics:
    attempted: bool = False
    accepted: bool = False
    color_repaired: bool = False
    prior_area: int = 0
    graphcut_area: int = 0
    final_area: int = 0
    area_ratio: float = 0.0
    core_recall: float = 0.0
    color_recall: float = 0.0
    span_ratio: float = 0.0
    prior_scale: float = 1.0
    decision: str = "not attempted"
    iterations_run: int = 0
    graphcut_pixels: int = 0


@dataclass(slots=True)
class MaskMetrics:
    iou: float = 0.0
    dice: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    boundary_iou: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    color_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    candidate_mask: np.ndarray
    edge_map: np.ndarray
    candidates: list[ShapeInfo]
    rejected: list[ShapeInfo]
    selected_index: int
    final_mask: np.ndarray
    segmented: np.ndarray
    trimap: np.ndarray | None
    refinement: RefinementDiagnostics
    latency_ms: float

    @property
    def selected(self) -> ShapeInfo | None:
        return self.candidates[self.selected_index] if self.selected_index >= 0 else None


@dataclass(slots=True)
class BatchSummary:
    processed: int = 0
    recognized: int = 0
    correct: int = 0
    non_empty_masks: int = 0
    binary_masks: int = 0
    one_component_masks: int = 0
    black_outside_masks: int = 0
    source_identical_masks: int = 0


@dataclass(slots=True)
class EvaluationSummary:
    evaluated: int = 0
    mean_iou: float = 0.0
    mean_dice: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_boundary_iou: float = 0.0
    shape_correct: int = 0
    strong_masks: int = 0
    weak_masks: int = 0
    zero_overlap: int = 0
    mean_latency_ms: float = 0.0
