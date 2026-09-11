"""Contour geometry, shape classification, and candidate selection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .masks import _threshold_color_evidence, require_bgr8, require_mask8
from .models import ProposalSource, ShapeInfo, ShapeParams, SignShape


_SHAPE_MODELS: tuple[tuple[SignShape, float, float, float, int], ...] = (
    (SignShape.CIRCLE, 1.000, 0.785, 0.605, 0),
    (SignShape.OCTAGON, 0.900, 0.828, 0.630, 8),
    (SignShape.RECTANGLE, 0.637, 1.000, 0.500, 4),
    (SignShape.TRIANGLE, 0.407, 0.500, 1.000, 3),
)


def _cv_round(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _normalize_contour(
    contour: np.ndarray,
    box: tuple[tuple[float, float], tuple[float, float], float],
) -> np.ndarray:
    (center_x, center_y), (width, height), angle = box
    width = max(width, 1.0)
    height = max(height, 1.0)
    radians = math.radians(-angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    delta_x = points[:, 0] - center_x
    delta_y = points[:, 1] - center_y
    rotated_x = delta_x * cosine - delta_y * sine
    rotated_y = delta_x * sine + delta_y * cosine
    normalized = np.column_stack((rotated_x / width * 100.0, rotated_y / height * 100.0))
    return np.ascontiguousarray(normalized.reshape(-1, 1, 2), dtype=np.float32)


def _model_distance(
    info: ShapeInfo,
    model: tuple[SignShape, float, float, float, int],
    params: ShapeParams,
) -> float:
    _, circle_fit, rectangle_fit, triangle_fit, corners = model
    dc = info.circle_fit - circle_fit
    dr = info.rectangle_fit - rectangle_fit
    dt = info.triangle_fit - triangle_fit
    distance = math.sqrt(0.35 * dc * dc + 0.30 * dr * dr + 0.35 * dt * dt)
    if corners == 0:
        missing = params.circle_vertices - info.vertices_low
        if missing > 0:
            distance += params.circularity_weight * min(4, missing)
        if info.vertices_very_high == 4 and info.rectangle_fit > 0.78:
            distance += 0.10
    elif corners == 8:
        distance += params.vertex_weight * min(4, abs(info.vertices - 8))
        extra = info.vertices_low - 8
        if extra > 0:
            distance += params.circularity_weight * min(4, extra)
        distance += params.octagon_bias
    else:
        observed = (
            info.vertices_very_high
            if corners == 4 and info.vertices_very_high == 4
            else info.vertices
        )
        distance += params.vertex_weight * min(4, abs(observed - corners))
    return distance


def _reason_area(label: str, value: float, limit: float) -> str:
    return f"{label} {value:.0f} (limit {limit:.0f})"


def _is_diamond(info: ShapeInfo) -> bool:
    """Recognize a tip-up rhombus in image coordinates, before normalization.

    Normalized rectangle fits discard orientation and cannot separate a square
    from a diamond. Allow modest perspective/roll, but reject elongated rotated
    rectangles and polygons that only become quadrilaterals after heavy smoothing.
    """
    if len(info.poly) != 4 or info.vertices_low > 6:
        return False
    points = info.poly.reshape(4, 2).astype(np.float64)
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if sides.min() < 0.72 * sides.max():
        return False
    if cv2.contourArea(info.poly) < 0.90 * info.hull_area:
        return False
    diagonals = np.abs(points[2:] - points[:2])
    horizontal = int(np.argmax(diagonals[:, 0]))
    vertical = 1 - horizontal
    # tan(20 degrees): diagonals must follow the image axes within 20 degrees.
    slope_limit = math.tan(math.radians(20.0))
    if (diagonals[horizontal, 1] > slope_limit * diagonals[horizontal, 0]
            or diagonals[vertical, 0] > slope_limit * diagonals[vertical, 1]):
        return False
    midpoints = (points[2:] + points[:2]) * 0.5
    return bool(np.linalg.norm(midpoints[0] - midpoints[1])
                <= 0.15 * np.linalg.norm(diagonals, axis=1).min())


def _measure_contour(
    contour: np.ndarray,
    image_size: tuple[int, int],
    params: ShapeParams,
) -> tuple[bool, ShapeInfo]:
    info = ShapeInfo(contour=np.ascontiguousarray(contour, dtype=np.int32))
    if len(info.contour) < 6:
        info.reject = "boundary shorter than 6 points"
        return False, info

    info.area = float(cv2.contourArea(info.contour))
    info.perimeter = float(cv2.arcLength(info.contour, True))
    info.hull = np.ascontiguousarray(cv2.convexHull(info.contour), dtype=np.int32)
    info.hull_area = float(cv2.contourArea(info.hull))
    info.hull_perimeter = float(cv2.arcLength(info.hull, True))
    if info.hull_area < 1.0 or info.hull_perimeter < 1.0 or info.perimeter < 1.0:
        info.reject = "degenerate boundary"
        return False, info

    width, height = image_size
    image_area = float(width * height)
    if info.hull_area < params.min_area:
        info.reject = _reason_area("area", info.hull_area, params.min_area)
        return False, info
    if info.hull_area < params.min_area_ratio * image_area:
        info.reject = _reason_area(
            "area",
            info.hull_area,
            params.min_area_ratio * image_area,
        )
        return False, info
    if info.hull_area > params.max_area_ratio * image_area:
        info.reject = _reason_area(
            "area too large",
            info.hull_area,
            params.max_area_ratio * image_area,
        )
        return False, info

    info.solidity = info.area / info.hull_area
    info.rim_ratio = info.perimeter / info.hull_perimeter
    info.circularity = min(
        1.0,
        4.0 * math.pi * info.hull_area / (info.hull_perimeter * info.hull_perimeter),
    )
    if info.solidity < params.min_solidity:
        looks_like_rim = (
            info.solidity >= params.rim_min_solid
            and params.rim_low <= info.rim_ratio <= params.rim_high
            and info.circularity >= params.rim_min_circularity
        )
        if not looks_like_rim:
            info.reject = (
                f"solidity {info.solidity:.2f} < {params.min_solidity:.2f} "
                f"and not a rim (P/Phull {info.rim_ratio:.2f})"
            )
            return False, info
        info.rim_rescued = True

    info.bbox = tuple(int(value) for value in cv2.boundingRect(info.hull))
    info.box = cv2.minAreaRect(info.hull)
    circle_center, circle_radius = cv2.minEnclosingCircle(info.hull)
    info.circle_center = (float(circle_center[0]), float(circle_center[1]))
    info.circle_radius = float(circle_radius)
    if len(info.hull) >= 5:
        try:
            info.ellipse = cv2.fitEllipseAMS(info.hull)
            info.ellipse_valid = info.ellipse[1][0] > 2.0 and info.ellipse[1][1] > 2.0
        except cv2.error:
            info.ellipse_valid = False

    long_side = max(info.box[1])
    short_side = min(info.box[1])
    if short_side < 4.0:
        info.reject = "thinner than 4 pixels"
        return False, info
    info.aspect = long_side / short_side
    if info.aspect > params.max_aspect:
        info.reject = f"aspect {info.aspect:.2f} > {params.max_aspect:.2f} (a pole or a marking)"
        return False, info

    x, y, box_width, box_height = info.bbox
    info.extent = info.hull_area / float(box_width * box_height)
    dx = (info.box[0][0] - 0.5 * width) / max(1.0, 0.5 * width)
    dy = (info.box[0][1] - 0.5 * height) / max(1.0, 0.5 * height)
    info.center_proximity = float(
        np.clip(1.0 - math.hypot(dx, dy) / math.sqrt(2.0), 0.0, 1.0)
    )

    normalized = _normalize_contour(info.hull, info.box)
    normalized_hull = cv2.convexHull(normalized)
    normalized_area = float(cv2.contourArea(normalized_hull))
    normalized_perimeter = float(cv2.arcLength(normalized_hull, True))
    if normalized_area < 1.0 or normalized_perimeter < 1.0:
        info.reject = "degenerate after aspect normalisation"
        return False, info

    _, normalized_radius = cv2.minEnclosingCircle(normalized_hull)
    try:
        normalized_triangle_area, _ = cv2.minEnclosingTriangle(normalized_hull)
    except cv2.error:
        normalized_triangle_area = 0.0
    normalized_box = cv2.minAreaRect(normalized_hull)
    normalized_rectangle_area = normalized_box[1][0] * normalized_box[1][1]
    info.circle_fit = (
        normalized_area / (math.pi * normalized_radius * normalized_radius)
        if normalized_radius > 1.0
        else 0.0
    )
    info.rectangle_fit = (
        normalized_area / normalized_rectangle_area
        if normalized_rectangle_area > 1.0
        else 0.0
    )
    info.triangle_fit = (
        normalized_area / normalized_triangle_area
        if normalized_triangle_area > 1.0
        else 0.0
    )

    polygon_low = cv2.approxPolyDP(
        normalized_hull,
        params.epsilon_low * normalized_perimeter,
        True,
    )
    polygon_high = cv2.approxPolyDP(
        normalized_hull,
        params.epsilon_high * normalized_perimeter,
        True,
    )
    polygon_very_high = cv2.approxPolyDP(
        normalized_hull,
        0.050 * normalized_perimeter,
        True,
    )
    info.vertices_low = len(polygon_low)
    info.vertices = len(polygon_high)
    info.vertices_very_high = len(polygon_very_high)
    info.poly = np.ascontiguousarray(
        cv2.approxPolyDP(
            info.hull,
            params.epsilon_factor * info.hull_perimeter,
            True,
        ),
        dtype=np.int32,
    )

    best_model: tuple[SignShape, float, float, float, int] | None = None
    best_distance = math.inf
    for model in _SHAPE_MODELS:
        shape = model[0]
        if (
            shape is SignShape.RECTANGLE
            and info.rectangle_fit < params.rectangle_gate
            and not (
                info.vertices_very_high == 4 and info.rectangle_fit >= 0.78
            )
        ):
            continue
        if shape is SignShape.TRIANGLE and info.triangle_fit < params.triangle_gate:
            continue
        if shape is SignShape.OCTAGON and info.circle_fit < params.octagon_gate:
            continue
        distance = _model_distance(info, model, params)
        if distance < best_distance:
            best_distance = distance
            best_model = model
    if best_model is None:
        best_model = _SHAPE_MODELS[0]
        best_distance = _model_distance(info, best_model, params)

    info.shape = best_model[0]
    info.score = float(np.clip(1.0 - best_distance / 0.50, 0.0, 1.0))
    if _is_diamond(info):
        info.shape = SignShape.DIAMOND
        info.score = float(np.clip(cv2.contourArea(info.poly) / info.hull_area, 0.0, 1.0))
    if info.score < params.min_score:
        info.shape = SignShape.UNKNOWN

    if info.shape is SignShape.TRIANGLE:
        if len(info.poly) == 3:
            info.ideal = info.poly.copy()
        else:
            try:
                _, triangle = cv2.minEnclosingTriangle(info.hull)
                info.ideal = np.ascontiguousarray(
                    np.rint(triangle).astype(np.int32).reshape(-1, 1, 2)
                )
            except cv2.error:
                info.ideal = info.hull.copy()
    elif info.shape in (SignShape.RECTANGLE, SignShape.DIAMOND):
        if len(info.poly) == 4:
            info.ideal = info.poly.copy()
        else:
            info.ideal = np.ascontiguousarray(
                np.rint(cv2.boxPoints(info.box)).astype(np.int32).reshape(-1, 1, 2)
            )
    elif info.shape is SignShape.OCTAGON:
        info.ideal = info.poly.copy() if 7 <= len(info.poly) <= 9 else info.hull.copy()
    else:
        info.ideal = info.hull.copy()

    info.touches_border = (
        x <= 1
        or y <= 1
        or x + box_width >= width - 2
        or y + box_height >= height - 2
    )
    return True, info


def classify_contour(
    contour: np.ndarray,
    image_size: tuple[int, int],
    params: ShapeParams,
) -> ShapeInfo | None:
    accepted, info = _measure_contour(contour, image_size, params)
    return info if accepted else None


def _detect_one_mask(
    mask: np.ndarray,
    params: ShapeParams,
    color_id: int,
    rejected: list[ShapeInfo] | None,
) -> list[ShapeInfo]:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return []
    require_mask8(mask, mask.shape, "detect_sign_shapes")
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result: list[ShapeInfo] = []
    image_size = (mask.shape[1], mask.shape[0])
    for contour in contours:
        accepted, info = _measure_contour(contour, image_size, params)
        info.color_id = color_id
        if accepted:
            result.append(info)
        elif rejected is not None:
            rejected.append(info)
    return result


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def detect_sign_shapes(
    masks: np.ndarray | Sequence[np.ndarray],
    params: ShapeParams,
    color_id: int = -1,
    rejected: list[ShapeInfo] | None = None,
) -> list[ShapeInfo]:
    if isinstance(masks, np.ndarray):
        return _detect_one_mask(masks, params, color_id, rejected)

    all_candidates: list[ShapeInfo] = []
    for current_color, mask in enumerate(masks):
        require_mask8(mask, mask.shape, "detect_sign_shapes")
        candidates = _detect_one_mask(mask, params, current_color, rejected)
        side = min(mask.shape)
        if params.multi_scale:
            kernel_size = max(params.weld_min, _cv_round(side * params.weld_frac))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            welded = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            for coarse in _detect_one_mask(welded, params, current_color, None):
                if not any(
                    _bbox_iou(coarse.bbox, fine.bbox) > params.weld_iou
                    for fine in candidates
                ):
                    candidates.append(coarse)
        if params.split_merged:
            for scale in (1.0, 1.75):
                kernel_size = max(
                    params.split_open_min,
                    _cv_round(side * params.split_open_frac * scale),
                )
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size),
                )
                opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                for split in _detect_one_mask(opened, params, current_color, None):
                    if not any(
                        _bbox_iou(split.bbox, existing.bbox) > 0.88
                        for existing in candidates
                    ):
                        candidates.append(split)
        all_candidates.extend(candidates)
    return all_candidates


def _is_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
    padding: int = 3,
) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    return (
        ix >= ox - padding
        and iy >= oy - padding
        and ix + iw <= ox + ow + padding
        and iy + ih <= oy + oh + padding
    )


def select_best_sign(candidates: Sequence[ShapeInfo], params: ShapeParams) -> int:
    maximum_area = max((candidate.hull_area for candidate in candidates), default=0.0)
    if maximum_area < 1.0:
        return -1
    reference_area = max(
        (
            candidate.hull_area
            for candidate in candidates
            if not candidate.touches_border and candidate.selection_score >= 0.40
        ),
        default=maximum_area,
    )
    carriers = [False] * len(candidates)
    nested_boost = [1.0] * len(candidates)
    nested_penalty = [1.0] * len(candidates)
    protected_outer = [False] * len(candidates)

    for inner_index, inner in enumerate(candidates):
        if inner.shape is SignShape.UNKNOWN:
            continue
        for outer_index, outer in enumerate(candidates):
            if (
                inner_index == outer_index
                or outer.shape is SignShape.UNKNOWN
                or inner.hull_area >= outer.hull_area
            ):
                continue
            credible_inner = outer.touches_border or (
                inner.edge_support >= 0.65
                and inner.selection_score >= outer.selection_score + 0.05
            )
            if (
                inner.shape is not outer.shape
                and credible_inner
                and _is_inside(inner.bbox, outer.bbox)
                and inner.hull_area >= params.inner_min_area * outer.hull_area
                and inner.hull_area <= params.inner_max_area * outer.hull_area
                and inner.score >= params.inner_min_score * outer.score
                and inner.color_coverage >= params.min_color_coverage
            ):
                carriers[outer_index] = True

    for inner_index, inner in enumerate(candidates):
        for outer_index, outer in enumerate(candidates):
            if (
                inner_index == outer_index
                or inner.shape is SignShape.UNKNOWN
                or outer.shape is SignShape.UNKNOWN
                or inner.hull_area >= outer.hull_area
                or not _is_inside(inner.bbox, outer.bbox, 5)
            ):
                continue
            ratio = inner.hull_area / max(1.0, outer.hull_area)
            border_shape = inner.shape in {
                SignShape.CIRCLE,
                SignShape.TRIANGLE,
                SignShape.OCTAGON,
            }
            if (
                border_shape
                and inner.shape is outer.shape
                and 0.34 <= ratio <= 0.84
                and outer.edge_support >= 0.50
                and outer.color_capture >= 0.78 * inner.color_capture
                and outer.selection_score >= 0.72 * inner.selection_score
            ):
                nested_penalty[inner_index] = min(nested_penalty[inner_index], 0.88)
                nested_boost[outer_index] = max(nested_boost[outer_index], 1.10)
            if (
                inner.shape is SignShape.CIRCLE
                and outer.shape is SignShape.RECTANGLE
                and params.inner_max_area < ratio < 0.92
                and outer.score >= 0.65
                and outer.rectangle_fit >= 0.86
                and outer.edge_support >= 0.58
                and outer.color_coverage >= 0.50
                and outer.color_capture >= 0.80
                and outer.color_capture >= inner.color_capture + 0.06
                and outer.selection_score >= 0.72 * inner.selection_score
            ):
                nested_penalty[inner_index] = min(nested_penalty[inner_index], 0.62)
                nested_boost[outer_index] = max(nested_boost[outer_index], 1.16)
                protected_outer[outer_index] = True
            if (
                inner.shape is SignShape.TRIANGLE
                and outer.shape is SignShape.RECTANGLE
                and 0.02 <= ratio < params.inner_min_area
                and inner.color_id >= 0
                and inner.color_id == outer.color_id
                and outer.rectangle_fit >= 0.92
                and outer.score >= 0.78
                and outer.edge_support >= 0.58
                and outer.color_coverage >= 0.62
                and outer.color_capture >= 0.86
                and outer.selection_score >= 0.72
            ):
                nested_penalty[inner_index] = min(nested_penalty[inner_index], 0.48)
                nested_boost[outer_index] = max(nested_boost[outer_index], 1.14)
                protected_outer[outer_index] = True

    best_index = -1
    best_rank = 0.0
    size_weight = float(np.clip(params.selection_size_weight, 0.0, 1.0))
    for index, candidate in enumerate(candidates):
        if candidate.shape is SignShape.UNKNOWN or carriers[index]:
            continue
        size_term = min(1.0, math.sqrt(candidate.hull_area / reference_area))
        rank = (
            (1.0 - size_weight) * candidate.selection_score + size_weight * size_term
            if candidate.selection_score > 0.0
            else candidate.score * (0.35 + 0.65 * size_term)
        )
        rank *= nested_boost[index] * nested_penalty[index]
        rank *= (
            1.0 - params.center_prior_strength
            + params.center_prior_strength * candidate.center_proximity
        )
        if candidate.touches_border:
            rank *= (
                0.88
                if protected_outer[index]
                else (0.52 if candidate.hull_area > 0.55 * maximum_area else 0.78)
            )
        if rank > best_rank:
            best_rank = rank
            best_index = index
    return best_index


def build_shape_mask(info: ShapeInfo, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    if info.shape is SignShape.CIRCLE and info.ellipse_valid:
        cv2.ellipse(mask, info.ellipse, 255, cv2.FILLED, cv2.LINE_8)
    elif info.shape is SignShape.CIRCLE and info.circle_radius > 1.0:
        cv2.circle(
            mask,
            (_cv_round(info.circle_center[0]), _cv_round(info.circle_center[1])),
            _cv_round(info.circle_radius),
            255,
            cv2.FILLED,
            cv2.LINE_8,
        )
    elif len(info.ideal) >= 3:
        polygon = cv2.convexHull(np.ascontiguousarray(info.ideal, dtype=np.int32))
        cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_8)
    elif len(info.contour) >= 3:
        cv2.drawContours(mask, [info.contour], 0, 255, cv2.FILLED)
    return mask


def _odd_kernel(wanted: int, minimum: int) -> int:
    kernel_size = max(wanted, minimum)
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size


def _expanded_rect(
    rectangle: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = rectangle
    image_width, image_height = image_size
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return left, top, max(0, right - left), max(0, bottom - top)


def build_adaptive_edge_map(bgr: np.ndarray, params: ShapeParams) -> np.ndarray:
    require_bgr8(bgr, "build_adaptive_edge_map")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    local = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(local, (5, 5), 0.9)
    gradient_x = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3)
    absolute_x = cv2.convertScaleAbs(gradient_x)
    absolute_y = cv2.convertScaleAbs(gradient_y)
    gradient = cv2.addWeighted(absolute_x, 0.5, absolute_y, 0.5, 0.0)
    threshold_value, _ = cv2.threshold(
        gradient,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    high = min(210.0, max(55.0, 1.25 * threshold_value))
    low = max(18.0, 0.42 * high)
    edges = cv2.Canny(smooth, low, high, apertureSize=3, L2gradient=True)
    side = min(bgr.shape[:2])
    kernel_size = _odd_kernel(
        _cv_round(side * params.edge_close_frac),
        params.edge_close_min,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


def _detect_octagon_chamfer_proposals(
    bgr: np.ndarray,
    masks: Sequence[np.ndarray],
    edges: np.ndarray,
    params: ShapeParams,
) -> list[ShapeInfo]:
    """Find regular octagons whose outline follows edges inside sign colour."""
    image_height, image_width = bgr.shape[:2]
    side = min(image_height, image_width)
    edge_distance = cv2.distanceTransform(
        cv2.bitwise_not(edges),
        cv2.DIST_L2,
        3,
    ).astype(np.float32)
    normalized_masks = [
        (mask != 0).astype(np.float32)
        for mask in masks
        if mask.size != 0 and cv2.countNonZero(mask) > 0
    ]
    if not normalized_masks:
        return []

    minimum_size = max(24, _cv_round(0.18 * side))
    maximum_size = min(side - 2, _cv_round(0.68 * side))
    size_step = max(6, _cv_round(0.05 * side))
    proposals: list[ShapeInfo] = []
    for size in range(minimum_size, maximum_size + 1, size_step):
        center = 0.5 * (size - 1)
        radius = 0.5 * (size - 5)
        template_points = np.array(
            [
                [
                    _cv_round(center + radius * math.cos(math.pi / 8 + index * math.pi / 4)),
                    _cv_round(center + radius * math.sin(math.pi / 8 + index * math.pi / 4)),
                ]
                for index in range(8)
            ],
            dtype=np.int32,
        )
        outline = np.zeros((size, size), dtype=np.uint8)
        interior = np.zeros_like(outline)
        cv2.polylines(
            outline,
            [template_points],
            True,
            255,
            max(1, _cv_round(0.015 * size)),
            cv2.LINE_8,
        )
        cv2.fillConvexPoly(interior, template_points, 255, cv2.LINE_8)
        outline_weights = (outline != 0).astype(np.float32)
        interior_weights = (interior != 0).astype(np.float32)
        outline_area = float(outline_weights.sum())
        interior_area = float(interior_weights.sum())
        if outline_area <= 0.0 or interior_area <= 0.0:
            continue
        edge_cost = cv2.matchTemplate(
            edge_distance,
            outline_weights,
            cv2.TM_CCORR,
        ) / outline_area
        color_coverage = np.zeros_like(edge_cost)
        for color_mask in normalized_masks:
            color_coverage = np.maximum(
                color_coverage,
                cv2.matchTemplate(
                    color_mask,
                    interior_weights,
                    cv2.TM_CCORR,
                )
                / interior_area,
            )
        rows, columns = np.indices(edge_cost.shape)
        centers_x = columns + center
        centers_y = rows + center
        center_distance = np.hypot(
            centers_x - 0.5 * (image_width - 1),
            centers_y - 0.5 * (image_height - 1),
        ) / max(1.0, math.hypot(image_width, image_height))
        valid = (color_coverage >= 0.78) & (edge_cost <= 3.0)
        if not np.any(valid):
            continue
        objective = (
            edge_cost
            - 0.85 * color_coverage
            + 0.35 * center_distance.astype(np.float32)
        )
        objective[~valid] = np.inf
        top, left = np.unravel_index(int(np.argmin(objective)), objective.shape)
        absolute = template_points + np.array([left, top], dtype=np.int32)
        contour = np.ascontiguousarray(absolute.reshape(-1, 1, 2))
        info = classify_contour(contour, (image_width, image_height), params)
        if info is not None:
            info.shape = SignShape.OCTAGON
            info.proposal_source = ProposalSource.EDGE
            proposals.append(info)
    return proposals


def _shape_outline_mask(
    info: ShapeInfo,
    image_size: tuple[int, int],
    thickness: int,
) -> np.ndarray:
    width, height = image_size
    outline = np.zeros((height, width), dtype=np.uint8)
    if info.shape is SignShape.CIRCLE and info.ellipse_valid:
        cv2.ellipse(outline, info.ellipse, 255, thickness, cv2.LINE_8)
    elif info.shape is SignShape.CIRCLE and info.circle_radius > 1.0:
        cv2.circle(
            outline,
            (_cv_round(info.circle_center[0]), _cv_round(info.circle_center[1])),
            _cv_round(info.circle_radius),
            255,
            thickness,
            cv2.LINE_8,
        )
    elif len(info.ideal) >= 3:
        cv2.drawContours(outline, [info.ideal], 0, 255, thickness, cv2.LINE_8)
    return outline


def _dominant_color(prior: np.ndarray, masks: Sequence[np.ndarray]) -> int:
    best_index = -1
    best_area = 0
    for index, mask in enumerate(masks):
        if mask.size == 0:
            continue
        area = cv2.countNonZero(cv2.bitwise_and(prior, mask))
        if area > best_area:
            best_area = area
            best_index = index
    return best_index


def _measure_candidate_evidence(
    info: ShapeInfo,
    bgr: np.ndarray,
    masks: Sequence[np.ndarray],
    color_union: np.ndarray,
    edge_distance: np.ndarray,
    lab: np.ndarray,
    params: ShapeParams,
) -> None:
    image_size = (bgr.shape[1], bgr.shape[0])
    prior = build_shape_mask(info, image_size)
    prior_area = cv2.countNonZero(prior)
    if prior_area <= 0:
        return
    inside_color = cv2.bitwise_and(prior, color_union)
    color_inside = cv2.countNonZero(inside_color)
    info.color_coverage = color_inside / prior_area

    nonzero = cv2.findNonZero(prior)
    prior_box = cv2.boundingRect(nonzero) if nonzero is not None else info.bbox
    padding = max(3, _cv_round(max(prior_box[2], prior_box[3]) * 0.18))
    local_x, local_y, local_width, local_height = _expanded_rect(
        prior_box,
        image_size,
        padding,
    )
    local_color = (
        cv2.countNonZero(
            color_union[
                local_y : local_y + local_height,
                local_x : local_x + local_width,
            ]
        )
        if local_width > 0 and local_height > 0
        else 0
    )
    info.color_capture = (
        float(np.clip(color_inside / local_color, 0.0, 1.0))
        if local_color > 0
        else 0.0
    )
    if info.color_id < 0:
        info.color_id = _dominant_color(prior, masks)

    side = min(bgr.shape[:2])
    tolerance = max(1.25, side * params.edge_tolerance_frac)
    outline = _shape_outline_mask(info, image_size, 1)
    outline_pixels = outline != 0
    info.edge_support = (
        float(np.exp(-edge_distance[outline_pixels] / tolerance).mean())
        if np.any(outline_pixels)
        else 0.0
    )

    band_size = _odd_kernel(
        _cv_round(min(info.bbox[2], info.bbox[3]) * 0.07),
        3,
    )
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (band_size, band_size),
    )
    eroded = cv2.erode(prior, band_kernel)
    dilated = cv2.dilate(prior, band_kernel)
    inner_band = cv2.subtract(prior, eroded)
    outer_band = cv2.subtract(dilated, prior)
    if cv2.countNonZero(inner_band) > 0 and cv2.countNonZero(outer_band) > 0:
        inside_mean = cv2.mean(lab, mask=inner_band)[:3]
        outside_mean = cv2.mean(lab, mask=outer_band)[:3]
        difference = math.sqrt(
            sum((inside_mean[index] - outside_mean[index]) ** 2 for index in range(3))
        )
        info.contrast = float(np.clip(difference / 95.0, 0.0, 1.0))

    color_evidence = float(
        np.clip(
            0.50 * min(1.0, info.color_coverage / 0.50)
            + 0.50 * min(1.0, info.color_capture),
            0.0,
            1.0,
        )
    )
    image_area = float(bgr.shape[1] * bgr.shape[0])
    size_evidence = float(
        np.clip(math.sqrt(info.hull_area / max(1.0, 0.18 * image_area)), 0.0, 1.0)
    )
    info.selection_score = float(
        np.clip(
            0.36 * info.score
            + 0.34 * info.edge_support
            + 0.17 * color_evidence
            + 0.08 * info.contrast
            + 0.05 * size_evidence,
            0.0,
            1.0,
        )
    )
    if (
        info.shape is SignShape.TRIANGLE
        and info.edge_support >= 0.90
        and info.color_coverage >= 0.65
        and info.triangle_fit >= 0.75
    ):
        info.selection_score = min(1.0, info.selection_score + 0.055)
    if info.touches_border and info.hull_area > 0.55 * image_area:
        info.selection_score *= 0.48


def _sampled_circle(
    center: tuple[float, float],
    radius: float,
    count: int = 96,
) -> np.ndarray:
    angles = np.arange(count, dtype=np.float64) * (2.0 * math.pi / count)
    points = np.column_stack(
        (
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        )
    )
    rounded = np.where(points >= 0.0, np.floor(points + 0.5), np.ceil(points - 0.5))
    return np.ascontiguousarray(rounded.astype(np.int32).reshape(-1, 1, 2))


def _mask_iou(first: ShapeInfo, second: ShapeInfo, image_size: tuple[int, int]) -> float:
    first_mask = build_shape_mask(first, image_size)
    second_mask = build_shape_mask(second, image_size)
    union = cv2.bitwise_or(first_mask, second_mask)
    union_area = cv2.countNonZero(union)
    if union_area == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(first_mask, second_mask)) / union_area


def _accept_chamfer_fallback(
    ordinary: ShapeInfo,
    chamfer: ShapeInfo,
    overlap: float,
) -> bool:
    return (
        ordinary.selection_score < 0.78
        and overlap < 0.10
        and chamfer.hull_area / max(1.0, ordinary.hull_area) > 2.0
    )


def detect_sign_shapes_advanced(
    bgr: np.ndarray,
    masks: Sequence[np.ndarray],
    params: ShapeParams,
    rejected: list[ShapeInfo] | None = None,
) -> tuple[list[ShapeInfo], np.ndarray]:
    require_bgr8(bgr, "detect_sign_shapes_advanced")
    for mask in masks:
        require_mask8(mask, bgr.shape[:2], "detect_sign_shapes_advanced")
    edges = build_adaptive_edge_map(bgr, params)
    evidence_masks = _threshold_color_evidence(bgr, params)
    candidates = detect_sign_shapes(masks, params, rejected=rejected)
    for candidate in candidates:
        candidate.proposal_source = ProposalSource.COLOR

    if params.edge_proposals:
        edge_contours, _ = cv2.findContours(
            edges.copy(),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        image_size = (bgr.shape[1], bgr.shape[0])
        for contour in edge_contours:
            info = classify_contour(contour, image_size, params)
            if info is not None:
                info.proposal_source = ProposalSource.EDGE
                candidates.append(info)

        if params.hough_circles:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (5, 5), 1.1)
            side = min(bgr.shape[:2])
            circles = cv2.HoughCircles(
                smooth,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=max(8.0, side * 0.18),
                param1=110.0,
                param2=max(11.0, side * 0.075),
                minRadius=max(5, _cv_round(side * 0.075)),
                maxRadius=max(8, _cv_round(side * 0.58)),
            )
            if circles is not None:
                for center_x, center_y, radius in circles[0, :16]:
                    info = classify_contour(
                        _sampled_circle((center_x, center_y), radius),
                        image_size,
                        params,
                    )
                    if info is not None:
                        info.shape = SignShape.CIRCLE
                        info.score = max(info.score, 0.72)
                        info.proposal_source = ProposalSource.HOUGH
                        candidates.append(info)

    color_union = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for evidence_mask in evidence_masks:
        color_union = cv2.bitwise_or(color_union, evidence_mask)
    inverse_edges = cv2.bitwise_not(edges)
    edge_distance = cv2.distanceTransform(inverse_edges, cv2.DIST_L2, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    supported: list[ShapeInfo] = []
    image_area = float(bgr.shape[1] * bgr.shape[0])
    for candidate in candidates:
        _measure_candidate_evidence(
            candidate,
            bgr,
            evidence_masks,
            color_union,
            edge_distance,
            lab,
            params,
        )
        has_color = (
            candidate.color_coverage >= params.min_color_coverage
            or candidate.color_capture >= 0.12
        )
        has_edge = candidate.edge_support >= params.min_edge_support
        area_ratio = candidate.hull_area / max(1.0, image_area)
        plausible_hough = candidate.proposal_source is not ProposalSource.HOUGH or (
            area_ratio >= 0.050 and candidate.center_proximity >= 0.52
        )
        if plausible_hough and (
            candidate.proposal_source is ProposalSource.COLOR or (has_color and has_edge)
        ):
            supported.append(candidate)

    ordinary_candidates = supported.copy()
    ordinary_index = select_best_sign(ordinary_candidates, params)
    should_search_octagon = (
        params.octagon_chamfer_proposals
        and not any(candidate.shape is SignShape.OCTAGON for candidate in candidates)
        and (
            ordinary_index < 0
            or ordinary_candidates[ordinary_index].selection_score < 0.78
        )
    )
    chamfer_candidates: list[ShapeInfo] = []
    if should_search_octagon:
        for candidate in _detect_octagon_chamfer_proposals(bgr, masks, edges, params):
            _measure_candidate_evidence(
                candidate,
                bgr,
                evidence_masks,
                color_union,
                edge_distance,
                lab,
                params,
            )
            has_color = (
                candidate.color_coverage >= params.min_color_coverage
                or candidate.color_capture >= 0.12
            )
            has_edge = candidate.edge_support >= params.min_edge_support
            if has_color and has_edge:
                chamfer_candidates.append(candidate)
        supported.extend(chamfer_candidates)

    if chamfer_candidates and ordinary_candidates:
        chamfer_index = select_best_sign(chamfer_candidates, params)
        if ordinary_index >= 0 and chamfer_index >= 0:
            ordinary = ordinary_candidates[ordinary_index]
            chamfer = chamfer_candidates[chamfer_index]
            overlap = _mask_iou(
                ordinary,
                chamfer,
                (bgr.shape[1], bgr.shape[0]),
            )
            if _accept_chamfer_fallback(ordinary, chamfer, overlap):
                supported = ordinary_candidates + [chamfer]
            else:
                supported = ordinary_candidates

    supported.sort(key=lambda candidate: candidate.selection_score, reverse=True)
    unique: list[ShapeInfo] = []
    image_size = (bgr.shape[1], bgr.shape[0])
    for candidate in supported:
        duplicate = any(
            candidate.shape is existing.shape
            and _bbox_iou(candidate.bbox, existing.bbox) > 0.55
            and _mask_iou(candidate, existing, image_size) >= params.dedupe_iou
            for existing in unique
        )
        if not duplicate:
            unique.append(candidate)
        if len(unique) >= 48:
            break
    return unique, edges
