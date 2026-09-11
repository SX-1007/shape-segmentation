"""Shape-constrained, deterministic mask refinement."""

from __future__ import annotations

import math
from dataclasses import replace
from collections.abc import Sequence

import cv2
import numpy as np

from .detection import (
    _cv_round,
    _expanded_rect,
    _odd_kernel,
    build_adaptive_edge_map,
    build_shape_mask,
)
from .masks import (
    _fill_enclosed_holes,
    _threshold_color_evidence,
    require_bgr8,
    require_mask8,
)
from .models import RefinementDiagnostics, ShapeInfo, ShapeParams, SignShape


def _binary_boundary_support(
    mask: np.ndarray,
    edge_distance: np.ndarray,
    tolerance: float,
) -> float:
    eroded = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    outline = cv2.subtract(mask, eroded)
    pixels = outline != 0
    return (
        float(np.exp(-edge_distance[pixels] / tolerance).mean())
        if np.any(pixels)
        else 0.0
    )


def _scaled_binary_mask(
    source: np.ndarray,
    center: tuple[float, float],
    scale: float,
) -> np.ndarray:
    transform = cv2.getRotationMatrix2D(center, 0.0, scale)
    return cv2.warpAffine(
        source,
        transform,
        (source.shape[1], source.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _snap_prior_to_outer_boundary(
    bgr: np.ndarray,
    info: ShapeInfo,
    params: ShapeParams,
    base: np.ndarray,
) -> tuple[np.ndarray, float]:
    if (
        info.touches_border
        or info.shape is not SignShape.TRIANGLE
        or cv2.countNonZero(base) == 0
        or info.color_coverage < 0.88
        or info.color_capture < 0.45
    ):
        return base, 1.0
    edges = build_adaptive_edge_map(bgr, params)
    distance = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)
    tolerance = max(1.20, min(bgr.shape[:2]) * params.edge_tolerance_frac)
    scales = [1.0 + 0.05 * step for step in range(8)]
    hypotheses = [
        base if step == 0 else _scaled_binary_mask(base, info.box[0], scale)
        for step, scale in enumerate(scales)
    ]
    support = [
        _binary_boundary_support(candidate, distance, tolerance)
        for candidate in hypotheses
    ]
    global_best = max(support)
    selected = 0
    for index in range(1, len(support)):
        left = support[index - 1]
        right = support[index + 1] if index + 1 < len(support) else -1.0
        if (
            support[index] >= left
            and support[index] >= right
            and support[index] >= 0.58
            and support[index] >= 0.90 * global_best
        ):
            selected = index
    if selected > 0:
        base_radius = 0.5 * min(info.bbox[2], info.bbox[3])
        if (scales[selected] - 1.0) * base_radius >= 2.5:
            return hypotheses[selected], scales[selected]
    return _scaled_binary_mask(base, info.box[0], 1.28), 1.28


def _recover_circular_color_rim(
    info: ShapeInfo,
    raw_masks: Sequence[np.ndarray],
    base: np.ndarray,
) -> tuple[np.ndarray, float]:
    if (
        info.shape is not SignShape.CIRCLE
        or info.touches_border
        or info.color_id < 0
        or info.color_id >= len(raw_masks)
        or raw_masks[info.color_id].size == 0
        or cv2.countNonZero(base) == 0
        or info.color_coverage > 0.16
        or info.color_capture > 0.35
    ):
        return base, 1.0

    base_area = cv2.countNonZero(base)
    short_side = max(5, min(info.bbox[2], info.bbox[3]))
    search_radius = max(3, _cv_round(0.30 * short_side))
    search = cv2.dilate(
        base,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * search_radius + 1, 2 * search_radius + 1),
        ),
    )
    near_color = cv2.bitwise_and(raw_masks[info.color_id], search)
    if cv2.countNonZero(near_color) < max(10, _cv_round(0.015 * base_area)):
        return base, 1.0
    original_near = near_color.copy()
    outside_base = cv2.bitwise_and(near_color, cv2.compare(base, 0, cv2.CMP_EQ))
    if cv2.countNonZero(outside_base) < max(8, _cv_round(0.012 * base_area)):
        return base, 1.0

    close_size = _odd_kernel(_cv_round(0.08 * short_side), 3)
    near_color = cv2.morphologyEx(
        near_color,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    contours, _ = cv2.findContours(
        near_color,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    best = base
    chosen_scale = 1.0
    best_quality = -1.0
    for contour in contours:
        if len(contour) < 5 or cv2.contourArea(contour) < 0.20 * base_area:
            continue
        try:
            fitted = cv2.fitEllipseAMS(contour)
        except cv2.error:
            continue
        small_axis = min(fitted[1])
        large_axis = max(fitted[1])
        if small_axis < 4.0 or large_axis / small_axis > 1.75:
            continue
        if math.hypot(
            fitted[0][0] - info.box[0][0],
            fitted[0][1] - info.box[0][1],
        ) > 0.18 * short_side:
            continue
        candidate = np.zeros_like(base)
        cv2.ellipse(candidate, fitted, 255, cv2.FILLED, cv2.LINE_8)
        candidate_area = cv2.countNonZero(candidate)
        area_ratio = candidate_area / base_area
        if not 1.08 <= area_ratio <= 1.85:
            continue
        base_recall = cv2.countNonZero(cv2.bitwise_and(candidate, base)) / base_area
        original_area = cv2.countNonZero(original_near)
        color_capture = cv2.countNonZero(
            cv2.bitwise_and(candidate, original_near)
        ) / max(1, original_area)
        if base_recall < 0.88 or color_capture < 0.72:
            continue
        quality = (
            0.55 * color_capture
            + 0.35 * base_recall
            + 0.10 * min(1.0, area_ratio / 1.35)
        )
        if quality > best_quality:
            best_quality = quality
            best = candidate
            chosen_scale = math.sqrt(area_ratio)
    return best, chosen_scale


def _fallback(
    prior: np.ndarray,
    diagnostics: RefinementDiagnostics,
    reason: str,
) -> tuple[np.ndarray, RefinementDiagnostics]:
    diagnostics.accepted = False
    diagnostics.final_area = cv2.countNonZero(prior)
    diagnostics.decision = reason
    return prior, diagnostics


def refine_shape_mask(
    bgr: np.ndarray,
    info: ShapeInfo,
    color_masks: Sequence[np.ndarray],
    params: ShapeParams,
    include_trimap: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, RefinementDiagnostics]:
    diagnostics = RefinementDiagnostics()
    if not isinstance(bgr, np.ndarray) or bgr.size == 0:
        diagnostics.decision = "empty input image"
        return np.empty((0, 0), dtype=np.uint8), None, diagnostics
    require_bgr8(bgr, "refine_shape_mask")
    for mask in color_masks:
        require_mask8(mask, bgr.shape[:2], "refine_shape_mask")

    image_size = (bgr.shape[1], bgr.shape[0])
    prior = build_shape_mask(info, image_size)
    # Trust pixel-sharp evidence only when the two color measurements agree
    # inside the plate. Noisy/fragmented paint needs the established smoothed
    # evidence and all background samples; local models can otherwise drift.
    smooth_masks = _threshold_color_evidence(bgr, params)
    sharp_masks = _threshold_color_evidence(bgr, params, smooth=False)
    interior = cv2.erode(prior, np.ones((7, 7), np.uint8)) != 0
    color_ids = [info.color_id] if 0 <= info.color_id < len(sharp_masks) else range(3)
    sharp_color = np.zeros_like(prior)
    smooth_color = np.zeros_like(prior)
    for color_id in color_ids:
        sharp_color = cv2.bitwise_or(sharp_color, sharp_masks[color_id])
        smooth_color = cv2.bitwise_or(smooth_color, smooth_masks[color_id])
    evidence_pixels = interior & ((sharp_color != 0) | (smooth_color != 0))
    evidence_count = np.count_nonzero(evidence_pixels)
    disagreement = np.count_nonzero(evidence_pixels & (sharp_color != smooth_color))
    sharp_evidence = evidence_count >= 8 and disagreement <= 0.02 * evidence_count
    raw_masks = sharp_masks if sharp_evidence else smooth_masks
    prior, prior_scale = _snap_prior_to_outer_boundary(bgr, info, params, prior)
    if prior_scale == 1.0:
        prior, prior_scale = _recover_circular_color_rim(info, raw_masks, prior)
    prior_area = cv2.countNonZero(prior)
    diagnostics.prior_area = prior_area
    diagnostics.prior_scale = prior_scale

    if not params.refine_with_grabcut:
        result, diagnostics = _fallback(
            prior,
            diagnostics,
            "shape prior (refinement disabled)",
        )
        return result, None, diagnostics
    if prior_area == 0:
        result, diagnostics = _fallback(
            prior,
            diagnostics,
            "empty image or shape prior",
        )
        return result, None, diagnostics
    diagnostics.attempted = True

    short_side = max(3, min(info.bbox[2], info.bbox[3]))
    outer_size = _odd_kernel(
        _cv_round(short_side * params.grabcut_outer_frac) * 2 + 1,
        3,
    )
    inner_size = _odd_kernel(
        _cv_round(short_side * params.grabcut_inner_frac) * 2 + 1,
        3,
    )
    outer = cv2.dilate(
        prior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_size, outer_size)),
    )
    inner = cv2.erode(
        prior,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_size, inner_size)),
    )
    if cv2.countNonZero(inner) < 8:
        inner = cv2.erode(
            prior,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
    if cv2.countNonZero(inner) < 4:
        result, diagnostics = _fallback(
            prior,
            diagnostics,
            "shape prior (no stable foreground core)",
        )
        return result, None, diagnostics

    outward_rim_evidence = False
    if (
        info.shape is SignShape.CIRCLE
        and 0 <= info.color_id < len(raw_masks)
        and info.color_coverage <= 0.16
        and info.color_capture <= 0.35
    ):
        near_raw = cv2.bitwise_and(raw_masks[info.color_id], outer)
        outside_prior = cv2.bitwise_and(
            near_raw,
            cv2.compare(prior, 0, cv2.CMP_EQ),
        )
        outward_rim_evidence = cv2.countNonZero(outside_prior) >= max(
            8,
            _cv_round(0.012 * prior_area),
        )

    grabcut_mask = np.full(bgr.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[outer != 0] = cv2.GC_PR_BGD
    grabcut_mask[prior != 0] = cv2.GC_PR_FGD
    grabcut_mask[inner != 0] = cv2.GC_FGD
    color_union = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for color_mask in (sharp_masks if sharp_evidence else color_masks):
        color_union = cv2.bitwise_or(color_union, color_mask)
    color_inside = cv2.bitwise_and(color_union, prior)
    sure_color = cv2.erode(
        color_inside,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    grabcut_mask[sure_color != 0] = cv2.GC_FGD
    trimap = (grabcut_mask * 85).astype(np.uint8) if include_trimap else None

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    # Retain a definite-background context band around the full allowed mask.
    # Cropping the graph keeps native pixel resolution and avoids spending time
    # on distant background that cannot belong to this selected sign.
    region = (slice(None), slice(None))
    if params.grabcut_use_roi and sharp_evidence:
        x, y, width, height = _expanded_rect(
            cv2.boundingRect(outer), image_size, max(8, _cv_round(0.10 * short_side)),
        )
        region = (slice(y, y + height), slice(x, x + width))
    graph_image = np.ascontiguousarray(bgr[region])
    graph_mask = grabcut_mask[region].copy()
    diagnostics.graphcut_pixels = graph_mask.size
    try:
        if not params.grabcut_early_stop or not sharp_evidence:
            cv2.grabCut(
                graph_image, graph_mask, None, background_model, foreground_model,
                params.grabcut_iterations, cv2.GC_INIT_WITH_MASK,
            )
            diagnostics.iterations_run = max(0, params.grabcut_iterations)
        else:
            stable_steps = 0
            for iteration in range(max(0, params.grabcut_iterations)):
                previous = graph_mask.copy()
                cv2.grabCut(
                    graph_image, graph_mask, None, background_model, foreground_model,
                    1, cv2.GC_INIT_WITH_MASK if iteration == 0 else cv2.GC_EVAL,
                )
                diagnostics.iterations_run += 1
                # Require two successive unchanged updates after initialization.
                # This is a bounded stopping heuristic, not a proof that the
                # evolving mixture models have reached their global optimum.
                stable_steps = stable_steps + 1 if iteration > 0 and np.array_equal(
                    previous, graph_mask
                ) else 0
                if stable_steps >= 2:
                    break
    except cv2.error:
        result, diagnostics = _fallback(
            prior,
            diagnostics,
            "shape prior (GrabCut exception)",
        )
        return result, trimap, diagnostics

    grabcut_mask[region] = graph_mask
    foreground = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground = cv2.bitwise_and(foreground, outer)
    diagnostics.graphcut_area = cv2.countNonZero(foreground)

    protected_color = np.zeros(bgr.shape[:2], dtype=np.uint8)
    if 0 <= info.color_id < len(raw_masks):
        protected_color = raw_masks[info.color_id].copy()
    else:
        for raw_mask in raw_masks:
            protected_color = cv2.bitwise_or(protected_color, raw_mask)
    protected_color = cv2.bitwise_and(
        protected_color,
        outer if outward_rim_evidence else prior,
    )
    protected_area = cv2.countNonZero(protected_color)
    missing_protected = cv2.bitwise_and(
        protected_color,
        cv2.compare(foreground, 0, cv2.CMP_EQ),
    )
    diagnostics.color_repaired = cv2.countNonZero(missing_protected) > 0
    foreground = cv2.bitwise_or(foreground, protected_color)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8,
    )
    best_label = -1
    best_score = -1.0
    for label in range(1, component_count):
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        core = cv2.countNonZero(cv2.bitwise_and(component, inner))
        color_hit = cv2.countNonZero(cv2.bitwise_and(component, protected_color))
        area = int(stats[label, cv2.CC_STAT_AREA])
        score = core + 1.25 * color_hit + 0.015 * area
        if score > best_score:
            best_score = score
            best_label = label
    if best_label < 0:
        result, diagnostics = _fallback(
            prior,
            diagnostics,
            "shape prior (no connected foreground)",
        )
        return result, trimap, diagnostics

    refined = np.where(labels == best_label, 255, 0).astype(np.uint8)
    refined = _fill_enclosed_holes(refined)
    contours, _ = cv2.findContours(
        refined.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        area = cv2.contourArea(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 1.0 and area / hull_area < 0.88:
            refined.fill(0)
            cv2.fillConvexPoly(refined, hull, 255, cv2.LINE_8)

    refined_area = cv2.countNonZero(refined)
    area_ratio = refined_area / prior_area
    core_recall = cv2.countNonZero(cv2.bitwise_and(refined, inner)) / max(
        1,
        cv2.countNonZero(inner),
    )
    color_recall = (
        cv2.countNonZero(cv2.bitwise_and(refined, protected_color)) / protected_area
        if protected_area > 0
        else 1.0
    )
    prior_points = cv2.findNonZero(prior)
    refined_points = cv2.findNonZero(refined)
    span_ratio = 0.0
    if prior_points is not None and refined_points is not None:
        prior_box = cv2.boundingRect(prior_points)
        refined_box = cv2.boundingRect(refined_points)
        span_ratio = min(
            refined_box[2] / max(1, prior_box[2]),
            refined_box[3] / max(1, prior_box[3]),
        )
    diagnostics.area_ratio = area_ratio
    diagnostics.core_recall = core_recall
    diagnostics.color_recall = color_recall
    diagnostics.span_ratio = span_ratio

    fallback_reason: str | None = None
    if area_ratio < params.refine_min_area_ratio:
        fallback_reason = "shape prior (partial GrabCut area)"
    maximum_area_ratio = (
        max(params.refine_max_area_ratio, 1.45)
        if outward_rim_evidence
        else params.refine_max_area_ratio
    )
    if fallback_reason is None and area_ratio > maximum_area_ratio:
        fallback_reason = "shape prior (GrabCut leaked outside)"
    if fallback_reason is None and core_recall < 0.72:
        fallback_reason = "shape prior (foreground core lost)"
    if fallback_reason is None and span_ratio < params.refine_min_span_ratio:
        fallback_reason = "shape prior (one-sided/half mask)"
    if fallback_reason is None and color_recall < params.refine_min_color_recall:
        fallback_reason = "shape prior (sign colour or rim lost)"
    if fallback_reason is not None:
        if params.grabcut_use_roi and graph_mask.size < prior.size:
            # A local background model can be misleading. Retry once with all
            # background evidence before accepting a coarse geometric fallback.
            retry_mask, retry_trimap, retry_diagnostics = refine_shape_mask(
                bgr, info, color_masks, replace(params, grabcut_use_roi=False),
                include_trimap=include_trimap,
            )
            retry_diagnostics.iterations_run += diagnostics.iterations_run
            retry_diagnostics.graphcut_pixels += diagnostics.graphcut_pixels
            retry_diagnostics.decision = 'full-image retry: ' + retry_diagnostics.decision
            return retry_mask, retry_trimap, retry_diagnostics
        result, diagnostics = _fallback(prior, diagnostics, fallback_reason)
        return result, trimap, diagnostics

    diagnostics.accepted = True
    diagnostics.final_area = refined_area
    diagnostics.decision = (
        "accepted after colour/rim repair"
        if diagnostics.color_repaired
        else "accepted GrabCut"
    )
    return refined, trimap, diagnostics
