"""Fast deterministic checks matching the C++ self-test command."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .detection import build_shape_mask, classify_contour, detect_sign_shapes, select_best_sign
from .masks import build_color_masks, clean_mask
from .metrics import evaluate_binary_mask
from .models import ShapeInfo, ShapeParams, SignShape


def run_self_tests() -> int:
    passed = 0
    failed = 0

    def check(condition: bool, name: str) -> None:
        nonlocal passed, failed
        print(f"  {'[PASS]' if condition else '[FAIL]'} {name}")
        if condition:
            passed += 1
        else:
            failed += 1

    print("\nDeterministic shape/segmentation self-tests")
    empty = np.zeros((64, 64), np.uint8)
    empty_metrics = evaluate_binary_mask(empty, empty)
    check(
        empty_metrics.iou == 1.0 and empty_metrics.boundary_iou == 1.0,
        "two empty masks score as an exact match",
    )
    border_object = np.zeros((64, 64), np.uint8)
    cv2.rectangle(border_object, (0, 8, 31, 40), 255, cv2.FILLED)
    exact = evaluate_binary_mask(border_object, border_object)
    check(
        exact.iou == 1.0 and exact.boundary_iou == 1.0,
        "Boundary-IoU handles objects truncated by an image border",
    )
    disjoint = np.zeros((64, 64), np.uint8)
    cv2.rectangle(disjoint, (40, 8, 20, 40), 255, cv2.FILLED)
    separated = evaluate_binary_mask(border_object, disjoint)
    check(
        separated.iou == 0.0 and separated.boundary_iou == 0.0,
        "disjoint masks have zero region and boundary overlap",
    )
    params = ShapeParams()
    check(
        detect_sign_shapes(np.empty((0, 0), np.uint8), params) == [],
        "an empty candidate mask is handled without an OpenCV assertion",
    )
    unknown = ShapeInfo(shape=SignShape.UNKNOWN, hull_area=1200.0)
    check(
        select_best_sign([unknown], params) == -1,
        "an UNKNOWN-only proposal set abstains instead of selecting a blob",
    )
    rejected_wrong_type = False
    try:
        build_color_masks(np.zeros((32, 32), np.uint8), params)
    except ValueError:
        rejected_wrong_type = True
    check(rejected_wrong_type, "a non-BGR input fails with an explicit format error")
    ring = np.zeros((96, 96), np.uint8)
    cv2.circle(ring, (48, 48), 30, 255, 5, cv2.LINE_8)
    check(
        clean_mask(ring, params)[48, 48] == 255,
        "morphological cleanup fills enclosed pictogram/rim holes",
    )

    for shape in (
        SignShape.CIRCLE,
        SignShape.TRIANGLE,
        SignShape.RECTANGLE,
        SignShape.OCTAGON,
    ):
        mask = np.zeros((160, 160), np.uint8)
        if shape is SignShape.CIRCLE:
            cv2.circle(mask, (80, 80), 52, 255, cv2.FILLED, cv2.LINE_8)
        elif shape is SignShape.TRIANGLE:
            cv2.fillConvexPoly(mask, np.array([[80, 18], [20, 140], [140, 140]], np.int32), 255, cv2.LINE_8)
        elif shape is SignShape.RECTANGLE:
            cv2.rectangle(mask, (28, 42, 104, 76), 255, cv2.FILLED, cv2.LINE_8)
        else:
            points = np.array(
                [
                    [
                        round(80 + 56 * math.cos(math.pi / 8.0 + index * math.pi / 4.0)),
                        round(80 + 56 * math.sin(math.pi / 8.0 + index * math.pi / 4.0)),
                    ]
                    for index in range(8)
                ],
                np.int32,
            )
            cv2.fillConvexPoly(mask, points, 255, cv2.LINE_8)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        info = classify_contour(contours[0], (160, 160), params) if contours else None
        check(
            info is not None and info.shape is shape,
            f"ideal {shape.name} is classified correctly",
        )
        rebuilt = build_shape_mask(info, (160, 160)) if info is not None else np.empty((0, 0), np.uint8)
        check(
            rebuilt.dtype == np.uint8 and cv2.countNonZero(rebuilt) > 0,
            f"ideal {shape.name} rebuilds a non-empty 8-bit mask",
        )
    print(f"Self-test result: {passed} passed, {failed} failed.")
    return 0 if failed == 0 else 1
