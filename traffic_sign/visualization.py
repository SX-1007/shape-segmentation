"""OpenCV visualization helpers for interactive and saved diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .detection import _cv_round
from .models import PipelineResult, ProposalSource, ShapeInfo, SignShape


_SHAPE_COLORS: dict[SignShape, tuple[int, int, int]] = {
    SignShape.TRIANGLE: (0, 255, 0),
    SignShape.RECTANGLE: (255, 160, 0),
    SignShape.OCTAGON: (0, 200, 255),
    SignShape.CIRCLE: (0, 0, 255),
    SignShape.DIAMOND: (220, 60, 220),
    SignShape.UNKNOWN: (170, 170, 170),
}


def draw_shape(
    canvas: np.ndarray,
    info: ShapeInfo,
    with_label: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    color = _SHAPE_COLORS[info.shape]
    if info.shape is SignShape.CIRCLE and info.ellipse_valid:
        cv2.ellipse(canvas, info.ellipse, color, thickness, cv2.LINE_AA)
    elif info.shape is SignShape.CIRCLE:
        cv2.circle(
            canvas,
            (_cv_round(info.circle_center[0]), _cv_round(info.circle_center[1])),
            _cv_round(info.circle_radius),
            color,
            thickness,
            cv2.LINE_AA,
        )
    elif len(info.ideal) >= 3:
        cv2.drawContours(canvas, [info.ideal], 0, color, thickness, cv2.LINE_AA)
    for point in info.poly.reshape(-1, 2):
        cv2.circle(canvas, tuple(int(value) for value in point), 2, (255, 255, 255), -1)
    if with_label:
        source = {
            ProposalSource.COLOR: "C",
            ProposalSource.EDGE: "E",
            ProposalSource.HOUGH: "H",
        }[info.proposal_source]
        score = info.selection_score if info.selection_score > 0.0 else info.score
        label = f"{info.shape.name} {score:.2f} {source}"
        x, y, width, height = info.bbox
        baseline_y = y - 4 if y - 4 >= 10 else y + height + 11
        cv2.putText(
            canvas,
            label,
            (x, baseline_y),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x, baseline_y),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _captioned(image: np.ndarray, caption: str, size: tuple[int, int]) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    panel = np.zeros((size[1] + 22, size[0], 3), dtype=np.uint8)
    panel[22:] = resized
    cv2.putText(
        panel,
        caption,
        (5, 15),
        cv2.FONT_HERSHEY_PLAIN,
        0.9,
        (250, 250, 250),
        1,
        cv2.LINE_AA,
    )
    return panel


def _grid(panels: Sequence[np.ndarray], columns: int) -> np.ndarray:
    rows = [cv2.hconcat(panels[index : index + columns]) for index in range(0, len(panels), columns)]
    return cv2.vconcat(rows)


def make_stage_panel(source: np.ndarray, result: PipelineResult) -> np.ndarray:
    candidate_canvas = source.copy()
    for candidate in result.candidates[:3]:
        draw_shape(candidate_canvas, candidate)
    selected_canvas = source.copy()
    if result.selected is not None:
        draw_shape(selected_canvas, result.selected)
    target_size = (source.shape[1], source.shape[0])
    stages = (
        (source, "1 Original"),
        (result.candidate_mask, "2 Colour candidate mask"),
        (candidate_canvas, "3 Fused colour + edge shapes"),
        (selected_canvas, "4 Selected sign"),
        (result.final_mask, "5 Graph-cut refined mask"),
        (result.segmented, "6 Pixel-accurate segmentation"),
    )
    return _grid([_captioned(image, caption, target_size) for image, caption in stages], 3)


def make_result_panel(source: np.ndarray, result: PipelineResult) -> np.ndarray:
    target_size = (source.shape[1], source.shape[0])
    return _grid(
        [
            _captioned(source, "Original", target_size),
            _captioned(result.segmented, "Sign segmented by shape", target_size),
        ],
        2,
    )
