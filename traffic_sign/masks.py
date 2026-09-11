"""Vectorized colour evidence and binary-mask cleanup."""

from __future__ import annotations

import cv2
import numpy as np

from .models import ShapeParams


def require_bgr8(image: np.ndarray, function_name: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ValueError(f"{function_name} requires an 8-bit 3-channel BGR image")


def require_mask8(
    mask: np.ndarray,
    expected_shape: tuple[int, int] | None,
    function_name: str,
) -> None:
    valid = (
        isinstance(mask, np.ndarray)
        and mask.dtype == np.uint8
        and mask.ndim == 2
        and mask.size > 0
    )
    if expected_shape is not None:
        valid = valid and mask.shape == expected_shape
    if not valid:
        raise ValueError(f"{function_name} requires an 8-bit single-channel mask")


def _high_percentile(values: np.ndarray, fraction: float = 0.995) -> float:
    bins = 512
    if values.size == 0:
        return 0.0
    indices = np.clip((values * bins).astype(np.int32), 0, bins - 1)
    histogram = np.bincount(indices.ravel(), minlength=bins)
    wanted = int(values.size * fraction)
    selected = int(np.searchsorted(np.cumsum(histogram), wanted, side="left"))
    return (selected + 0.5) / bins if selected < bins else 1.0


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    flood = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood[1:-1, 1:-1])
    return cv2.bitwise_or(mask, holes)


def clean_mask(mask: np.ndarray, params: ShapeParams) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return np.empty((0, 0), dtype=np.uint8)
    require_mask8(mask, mask.shape, "clean_mask")
    side = min(mask.shape)
    kernel_size = max(params.morph_min, int(np.rint(side * params.morph_frac)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    large_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, large_kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, small_kernel, iterations=1)
    if params.fill_holes:
        result = _fill_enclosed_holes(result)
    return result


def _threshold_color_evidence(
    bgr: np.ndarray,
    params: ShapeParams,
    *,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_bgr8(bgr, "build_color_masks")
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0) if smooth else bgr
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    red_hue = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 50, 35), (12, 255, 255)),
        cv2.inRange(hsv, (163, 50, 35), (180, 255, 255)),
    )
    blue_hue = cv2.inRange(hsv, (90, 50, 25), (140, 255, 255))
    yellow_hue = cv2.inRange(hsv, (15, 50, 50), (40, 255, 255))

    blue, green, red = cv2.split(blurred.astype(np.float32))
    channel_sum = red + green + blue
    bright = channel_sum >= params.dark_sum
    safe_sum = np.maximum(channel_sum, 1.0)

    red_evidence = np.maximum(0.0, np.minimum(red - green, red - blue) / safe_sum)
    blue_evidence = np.maximum(0.0, np.minimum(blue - red, blue - green) / safe_sum)
    yellow_evidence = np.maximum(
        0.0,
        np.minimum(red - blue, green - blue) / safe_sum,
    )
    yellow_evidence[(red - green) / safe_sum < -params.yellow_green_tol] = 0.0

    red_threshold = min(
        params.th_red,
        max(params.th_floor, params.adapt_rel * _high_percentile(red_evidence)),
    )
    blue_threshold = min(
        params.th_blue,
        max(params.th_floor, params.adapt_rel * _high_percentile(blue_evidence)),
    )
    yellow_threshold = min(
        params.th_yellow,
        max(params.th_floor, params.adapt_rel * _high_percentile(yellow_evidence)),
    )

    red_mask = np.where(
        (red_evidence > red_threshold) & (red_hue != 0) & bright,
        255,
        0,
    ).astype(np.uint8)
    blue_mask = np.where(
        (blue_evidence > blue_threshold) & (blue_hue != 0) & bright,
        255,
        0,
    ).astype(np.uint8)
    yellow_mask = np.where(
        (yellow_evidence > yellow_threshold) & (yellow_hue != 0) & bright,
        255,
        0,
    ).astype(np.uint8)

    return red_mask, blue_mask, yellow_mask


def build_color_masks(
    bgr: np.ndarray,
    params: ShapeParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red_mask, blue_mask, yellow_mask = _threshold_color_evidence(bgr, params)
    return (
        clean_mask(red_mask, params),
        clean_mask(blue_mask, params),
        clean_mask(yellow_mask, params),
    )


def build_color_candidate_mask(
    bgr: np.ndarray,
    params: ShapeParams,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    masks = build_color_masks(bgr, params)
    union = cv2.bitwise_or(cv2.bitwise_or(masks[0], masks[1]), masks[2])
    return union, masks
