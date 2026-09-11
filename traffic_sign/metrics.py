"""Pixel-level segmentation metrics compatible with the C++ reference."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .models import MaskMetrics


def binary8(mask: np.ndarray) -> np.ndarray:
    """Return an 8-bit mask whose nonzero pixels are 255."""
    array = np.asarray(mask)
    if array.size == 0:
        return np.empty((0, 0), dtype=np.uint8)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("mask must be a single-channel or 3-channel image")
    return cv2.compare(gray, 0, cv2.CMP_GT)


def _safe_ratio(numerator: float, denominator: float, empty_value: float) -> float:
    return numerator / denominator if denominator > 0.0 else empty_value


def inner_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Extract the reference Boundary-IoU inner band, including image edges."""
    binary = binary8(mask)
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(
        padded,
        np.ones((3, 3), dtype=np.uint8),
        iterations=max(1, int(width)),
    )
    cropped = eroded[1 : binary.shape[0] + 1, 1 : binary.shape[1] + 1]
    return cv2.subtract(binary, cropped)


def evaluate_binary_mask(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> MaskMetrics:
    """Calculate region and reference-compatible boundary overlap scores."""
    pred = binary8(prediction)
    truth = binary8(ground_truth)
    if pred.shape != truth.shape:
        raise ValueError("prediction and ground truth must have identical dimensions")
    if pred.size == 0:
        return MaskMetrics()

    intersection = cv2.bitwise_and(pred, truth)
    union = cv2.bitwise_or(pred, truth)
    true_positive = float(cv2.countNonZero(intersection))
    predicted_area = float(cv2.countNonZero(pred))
    truth_area = float(cv2.countNonZero(truth))
    union_area = float(cv2.countNonZero(union))

    diagonal = math.hypot(truth.shape[1], truth.shape[0])
    boundary_width = max(1, math.floor(0.020 * diagonal + 0.5))
    pred_boundary = inner_boundary(pred, boundary_width)
    truth_boundary = inner_boundary(truth, boundary_width)
    boundary_intersection = cv2.countNonZero(
        cv2.bitwise_and(pred_boundary, truth_boundary)
    )
    boundary_union = cv2.countNonZero(cv2.bitwise_or(pred_boundary, truth_boundary))

    return MaskMetrics(
        iou=_safe_ratio(true_positive, union_area, 1.0),
        dice=_safe_ratio(2.0 * true_positive, predicted_area + truth_area, 1.0),
        precision=_safe_ratio(
            true_positive,
            predicted_area,
            1.0 if truth_area == 0.0 else 0.0,
        ),
        recall=_safe_ratio(
            true_positive,
            truth_area,
            1.0 if predicted_area == 0.0 else 0.0,
        ),
        boundary_iou=_safe_ratio(
            float(boundary_intersection),
            float(boundary_union),
            1.0,
        ),
    )
