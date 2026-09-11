"""Custom-dataset segmentation evaluation and compatible reporting."""

from __future__ import annotations

import csv
import math
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .detection import _cv_round, classify_contour, detect_sign_shapes_advanced, select_best_sign
from .masks import build_color_masks
from .metrics import evaluate_binary_mask
from .models import EvaluationSummary, MaskMetrics, ProposalSource, ShapeParams, SignShape
from .pipeline import process_image
from .refinement import refine_shape_mask


_EVALUATION_HEADER = (
    "file",
    "expected_shape",
    "detected_shape",
    "proposal_count",
    "selected_source",
    "selection_score",
    "selected_area_ratio",
    "latency_ms",
    "iou",
    "dice",
    "precision",
    "recall",
    "boundary_iou",
)


def _read_shape_labels(path: Path) -> dict[str, SignShape]:
    labels: dict[str, SignShape] = {}
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in SignShape.__members__:
            labels[parts[0]] = SignShape[parts[1]]
    return labels


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = float(np.clip(quantile, 0.0, 1.0)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _source_mark(source: ProposalSource) -> str:
    return {
        ProposalSource.COLOR: "C",
        ProposalSource.EDGE: "E",
        ProposalSource.HOUGH: "H",
    }[source]


def evaluate_segmentation_dataset(
    root: Path | str,
    params: ShapeParams,
    save_predictions: bool = True,
) -> EvaluationSummary:
    dataset_root = Path(root)
    files = sorted((dataset_root / "images").glob("*.png"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f'no PNG evaluation images found under "{dataset_root / "images"}"')
    labels = _read_shape_labels(dataset_root / "labels.txt")
    predictions = dataset_root / "predictions"
    if save_predictions:
        predictions.mkdir(parents=True, exist_ok=True)

    totals = MaskMetrics()
    by_shape: dict[SignShape, list[MaskMetrics]] = defaultdict(list)
    iou_values: list[float] = []
    boundary_values: list[float] = []
    summary = EvaluationSummary()
    shape_scored = 0
    csv_path = dataset_root / "evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_EVALUATION_HEADER)
        for image_path in files:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            truth = cv2.imread(str(dataset_root / "masks" / image_path.name), cv2.IMREAD_GRAYSCALE)
            if image is None or truth is None or image.shape[:2] != truth.shape:
                continue
            result = process_image(image, params)
            selected = result.selected
            found = selected.shape if selected is not None else SignShape.UNKNOWN
            selected_source = _source_mark(selected.proposal_source) if selected is not None else "-"
            selected_score = selected.selection_score if selected is not None else 0.0
            selected_area_ratio = (
                selected.hull_area / float(image.shape[0] * image.shape[1])
                if selected is not None
                else 0.0
            )
            metrics = evaluate_binary_mask(result.final_mask, truth)
            totals.iou += metrics.iou
            totals.dice += metrics.dice
            totals.precision += metrics.precision
            totals.recall += metrics.recall
            totals.boundary_iou += metrics.boundary_iou
            summary.evaluated += 1
            summary.mean_latency_ms += result.latency_ms
            iou_values.append(metrics.iou)
            boundary_values.append(metrics.boundary_iou)
            summary.strong_masks += metrics.iou >= 0.80 and metrics.boundary_iou >= 0.55
            summary.weak_masks += metrics.iou < 0.50
            summary.zero_overlap += metrics.iou == 0.0
            expected = labels.get(image_path.name, SignShape.UNKNOWN)
            if expected is not SignShape.UNKNOWN:
                shape_scored += 1
                summary.shape_correct += found is expected
                by_shape[expected].append(metrics)
            writer.writerow(
                (
                    image_path.name,
                    expected.name,
                    found.name,
                    len(result.candidates),
                    selected_source,
                    f"{selected_score:.5f}",
                    f"{selected_area_ratio:.5f}",
                    f"{result.latency_ms:.5f}",
                    f"{metrics.iou:.5f}",
                    f"{metrics.dice:.5f}",
                    f"{metrics.precision:.5f}",
                    f"{metrics.recall:.5f}",
                    f"{metrics.boundary_iou:.5f}",
                )
            )
            if save_predictions:
                prediction_path = predictions / image_path.name
                if not cv2.imwrite(str(prediction_path), result.final_mask):
                    raise OSError(f'could not write image "{prediction_path}"')
    if summary.evaluated == 0:
        raise ValueError("no compatible image/mask pairs were evaluated")

    count = float(summary.evaluated)
    summary.mean_iou = totals.iou / count
    summary.mean_dice = totals.dice / count
    summary.mean_precision = totals.precision / count
    summary.mean_recall = totals.recall / count
    summary.mean_boundary_iou = totals.boundary_iou / count
    summary.mean_latency_ms /= count
    median_iou = statistics.median(iou_values)
    percentile_iou = _percentile(iou_values, 0.10)
    median_boundary = statistics.median(boundary_values)
    throughput = 1000.0 / summary.mean_latency_ms if summary.mean_latency_ms > 0.0 else 0.0
    report_lines = [
        "================ pixel-mask summary =====================",
        f"images evaluated       : {summary.evaluated}",
        f"mean IoU               : {summary.mean_iou:.3f}",
        f"mean Dice              : {summary.mean_dice:.3f}",
        f"mean precision         : {summary.mean_precision:.3f}",
        f"mean recall            : {summary.mean_recall:.3f}",
        f"mean Boundary-IoU      : {summary.mean_boundary_iou:.3f}",
        f"median IoU             : {median_iou:.3f}",
        f"10th-percentile IoU    : {percentile_iou:.3f}",
        f"median Boundary-IoU    : {median_boundary:.3f}",
        f"strong masks           : {summary.strong_masks}/{summary.evaluated}  (IoU >= .80 and Boundary-IoU >= .55)",
        f"weak masks             : {summary.weak_masks}/{summary.evaluated}  (IoU < .50)",
        f"zero-overlap masks     : {summary.zero_overlap}/{summary.evaluated}",
        f"mean pipeline latency  : {summary.mean_latency_ms:.2f} ms/image",
        f"measured throughput    : {throughput:.1f} images/s",
    ]
    if shape_scored:
        report_lines.append(
            f"shape accuracy          : {summary.shape_correct}/{shape_scored}  ({100.0 * summary.shape_correct / shape_scored:.1f}%)"
        )
    report_lines.extend(("", "per-shape segmentation (mean IoU / Boundary-IoU)"))
    for shape in (SignShape.CIRCLE, SignShape.TRIANGLE, SignShape.RECTANGLE, SignShape.OCTAGON, SignShape.DIAMOND):
        values = by_shape.get(shape, [])
        if values:
            report_lines.append(
                f"  {shape.name:<9} : {sum(value.iou for value in values) / len(values):.3f} / "
                f"{sum(value.boundary_iou for value in values) / len(values):.3f}  (n={len(values)})"
            )
        else:
            report_lines.append(f"  {shape.name:<9} : n/a")
    report_lines.append("=========================================================")
    (dataset_root / "evaluation_summary.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return summary


_CAMVID_HEADER = (
    "sample",
    "source_image",
    "component",
    "original_x",
    "original_y",
    "original_width",
    "original_height",
    "original_pixels",
    "detected_shape",
    "proposal_count",
    "selected_source",
    "selection_score",
    "latency_ms",
    "iou",
    "dice",
    "precision",
    "recall",
    "boundary_iou",
    "size_bin",
    "legacy_eligible",
    "native_iou",
    "native_dice",
    "native_boundary_iou",
)


_SOURCE_SIZE_BINS = ("<6", "6-9", "10-15", "16-31", ">=32")


def _source_size_bin(width: int, height: int) -> str:
    short_side = min(width, height)
    for limit, name in zip((6, 10, 16, 32), _SOURCE_SIZE_BINS):
        if short_side < limit:
            return name
    return _SOURCE_SIZE_BINS[-1]


def _write_camvid_size_reports(
    output: Path,
    inventory: list[dict],
    measurements: list[tuple[str, MaskMetrics, MaskMetrics]],
) -> None:
    """Offline reporting only; empty strata are missing data, never successes."""
    with (output / "component_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "sample", "source_image", "original_width", "original_height",
            "original_pixels", "size_bin", "legacy_eligible", "evaluated",
        ), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    measured = [
        dict(size_bin=name, iou=metric.iou, dice=metric.dice,
             boundary_iou=metric.boundary_iou, native_iou=native.iou,
             native_dice=native.dice, native_boundary_iou=native.boundary_iou)
        for name, metric, native in measurements
    ]
    metrics = ("iou", "dice", "boundary_iou", "native_iou", "native_dice", "native_boundary_iou")
    with (output / "size_strata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "size_bin", "components", "evaluated", "excluded", "weak", "zero_overlap",
            *(f"mean_{metric}" for metric in metrics),
        ), lineterminator="\n")
        writer.writeheader()
        for name in _SOURCE_SIZE_BINS:
            count = sum(item["size_bin"] == name for item in inventory)
            group = [row for row in measured if row["size_bin"] == name]
            writer.writerow({
                "size_bin": name, "components": count, "evaluated": len(group),
                "excluded": count - len(group),
                "weak": sum(float(row["iou"]) < 0.5 for row in group),
                "zero_overlap": sum(float(row["iou"]) == 0.0 for row in group),
                **{f"mean_{metric}": (
                    f"{statistics.mean(float(row[metric]) for row in group):.5f}" if group else ""
                ) for metric in metrics},
            })


def _centred_square_crop(
    image: np.ndarray,
    mask: np.ndarray,
    object_box: tuple[int, int, int, int],
    context_scale: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x, y, width, height = object_box
    side = max(max(width, height), _cv_round(max(width, height) * context_scale))
    centre_x = x + width // 2
    centre_y = y + height // 2
    crop_x = centre_x - side // 2
    crop_y = centre_y - side // 2
    left = max(0, -crop_x)
    top = max(0, -crop_y)
    right = max(0, crop_x + side - image.shape[1])
    bottom = max(0, crop_y + side - image.shape[0])
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    padded_x = crop_x + left
    padded_y = crop_y + top
    image_crop = padded_image[padded_y : padded_y + side, padded_x : padded_x + side].copy()
    mask_crop = padded_mask[padded_y : padded_y + side, padded_x : padded_x + side].copy()
    object_in_crop = (x + left - padded_x, y + top - padded_y, width, height)
    return image_crop, mask_crop, object_in_crop


def _sampled_rectangle(box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    x1, y1 = x + width - 1, y + height - 1
    xm, ym = (x + x1) // 2, (y + y1) // 2
    return np.asarray(
        ((x, y), (xm, y), (x1, y), (x1, ym), (x1, y1), (xm, y1), (x, y1), (x, ym)),
        dtype=np.int32,
    ).reshape((-1, 1, 2))


def _stable_instance_seed(stem: str, component: int) -> int:
    """Return the stable seed accepted by Python's 32-bit OpenCV binding.

    C++ assigns the complete FNV-1a value directly to ``cv::RNG::state``.
    Python exposes only ``setRNGSeed(int)``, so CamVid GrabCut is deterministic
    within this port but cannot be bit-identical to that 64-bit C++ state.
    """
    value = 1469598103934665603
    for byte in stem.encode("utf-8"):
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ component) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    low = value & 0xFFFFFFFF
    return low if low < 0x80000000 else low - 0x100000000


def _camvid_diagnostic(
    image: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    metrics: MaskMetrics,
) -> np.ndarray:
    truth_binary = np.where(truth != 0, 255, 0).astype(np.uint8)
    prediction_binary = np.where(prediction != 0, 255, 0).astype(np.uint8)
    truth_view = image.copy()
    prediction_view = image.copy()
    error_view = image.copy()
    truth_tint = np.full_like(image, (35, 210, 35))
    prediction_tint = np.full_like(image, (35, 35, 225))
    cv2.copyTo(cv2.addWeighted(image, 0.48, truth_tint, 0.52, 0.0), truth_binary, truth_view)
    cv2.copyTo(
        cv2.addWeighted(image, 0.48, prediction_tint, 0.52, 0.0),
        prediction_binary,
        prediction_view,
    )
    overlap = cv2.bitwise_and(truth_binary, prediction_binary)
    false_negative = cv2.bitwise_and(truth_binary, cv2.bitwise_not(prediction_binary))
    false_positive = cv2.bitwise_and(prediction_binary, cv2.bitwise_not(truth_binary))
    error_view[overlap != 0] = (0, 210, 255)
    error_view[false_negative != 0] = (0, 190, 0)
    error_view[false_positive != 0] = (0, 0, 230)

    panels = (
        (image.copy(), "Original real-background crop"),
        (truth_view, "CamVid SignSymbol ground truth"),
        (prediction_view, "Pipeline prediction"),
        (error_view, f"Error: IoU {metrics.iou:.3f}  B-IoU {metrics.boundary_iou:.3f}"),
    )
    rendered: list[np.ndarray] = []
    for panel, caption in panels:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (0, 0, 0), cv2.FILLED)
        cv2.putText(panel, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(panel)
    return cv2.vconcat((cv2.hconcat(rendered[:2]), cv2.hconcat(rendered[2:])))


def _path_matches_split(path: Path, requested_split: str, annotation: bool) -> bool:
    if requested_split == "all":
        return True
    expected = f"{requested_split}annot" if annotation else requested_split
    return expected in {part.lower() for part in path.parts}


def evaluate_camvid_dataset(
    dataset_root: Path | str,
    output_root: Path | str,
    params: ShapeParams,
    save_predictions: bool = True,
    split: str = "all",
    context_scale: float = 3.0,
    original_label_root: Path | str | None = None,
    use_box_prompt: bool = False,
    include_tiny: bool = False,
) -> EvaluationSummary:
    """Evaluate physical sign instances using the corrected CamVid protocol.

    ``use_box_prompt`` is segmentation given a ground-truth location; it is not
    an end-to-end sign-detection score.
    ``include_tiny`` evaluates previously excluded components in a separate,
    expanded track. Source-size reports always account for excluded components.
    """
    requested_split = split.lower()
    if requested_split not in {"all", "train", "val", "test"}:
        raise ValueError('split must be one of "all", "train", "val", or "test"')
    if not 1.05 <= context_scale <= 5.0:
        raise ValueError("CamVid context scale must be in [1.05, 5.0]")
    dataset = Path(dataset_root)
    png_files = sorted(dataset.rglob("*.png"), key=lambda path: str(path).lower())
    if not png_files:
        raise FileNotFoundError(f'no PNG files found below CamVid root "{dataset}"')

    original_files: list[Path] = []
    if original_label_root is not None:
        original_root = Path(original_label_root)
        original_files = sorted(original_root.rglob("*.png"), key=lambda path: str(path).lower())
        if not original_files:
            raise FileNotFoundError(f'no original CamVid label PNGs found below "{original_root}"')
    use_original_labels = bool(original_files)
    source_images: dict[str, Path] = {}
    label_files: list[Path] = []
    for path in png_files:
        lower_parts = {part.lower() for part in path.parts}
        annotation = any("annot" in part for part in lower_parts)
        if not _path_matches_split(path, requested_split, annotation):
            continue
        lower_name = path.name.lower()
        if not use_original_labels and annotation and not lower_name.endswith(("_c.png", "_c_c.png")):
            label_files.append(path)
        elif not annotation:
            source_images[path.stem] = path
    if use_original_labels:
        label_files = [path for path in original_files if path.name.endswith("_L.png")]
    if not label_files or not source_images:
        raise ValueError("CamVid images and matching annotation masks were not both found")

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    truth_dir = output / "truth"
    diagnostic_dir = output / "diagnostics"
    if save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        truth_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    external = replace(
        params,
        min_area=40.0,
        min_area_ratio=0.002,
        max_aspect=4.5,
        min_color_coverage=0.0,
        min_edge_support=0.24,
        selection_size_weight=0.60,
        center_prior_strength=0.55,
    )
    totals = MaskMetrics()
    summary = EvaluationSummary()
    iou_values: list[float] = []
    boundary_values: list[float] = []
    paired_frames = incompatible_frames = labelled_frames = 0
    connected_regions = skipped_tiny = detected = 0
    rows: list[tuple[object, ...]] = []
    inventory: list[dict] = []
    size_measurements: list[tuple[str, MaskMetrics, MaskMetrics]] = []

    for label_path in label_files:
        label_stem = label_path.stem[:-2] if label_path.stem.endswith("_L") else label_path.stem
        source_path = source_images.get(label_stem)
        if source_path is None:
            continue
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        label_flag = cv2.IMREAD_COLOR if use_original_labels else cv2.IMREAD_GRAYSCALE
        label = cv2.imread(str(label_path), label_flag)
        if image is None or label is None:
            incompatible_frames += 1
            continue
        if label.shape[:2] != image.shape[:2]:
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        paired_frames += 1
        sign_mask = (
            cv2.inRange(label, np.asarray((128, 128, 192)), np.asarray((128, 128, 192)))
            if use_original_labels
            else cv2.compare(label, 6, cv2.CMP_EQ)
        )
        if cv2.countNonZero(sign_mask) == 0:
            continue
        labelled_frames += 1
        component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            sign_mask, connectivity=8, ltype=cv2.CV_32S
        )
        for component in range(1, component_count):
            connected_regions += 1
            x, y, width, height, original_pixels = (int(value) for value in stats[component])
            legacy_eligible = width >= 10 and height >= 10 and original_pixels >= 30
            sample_name = f"{label_stem}_sign{component:02d}"
            size_bin = _source_size_bin(width, height)
            inventory.append(dict(
                sample=sample_name, source_image=source_path.name,
                original_width=width, original_height=height, original_pixels=original_pixels,
                size_bin=size_bin, legacy_eligible=int(legacy_eligible),
                evaluated=int(legacy_eligible or include_tiny),
            ))
            if not legacy_eligible and not include_tiny:
                skipped_tiny += 1
                continue
            instance_mask = cv2.compare(component_labels, component, cv2.CMP_EQ)
            source_crop, source_truth, object_in_crop = _centred_square_crop(
                image, instance_mask, (x, y, width, height), context_scale
            )
            crop = cv2.resize(source_crop, (256, 256), interpolation=cv2.INTER_AREA)
            truth = cv2.resize(source_truth, (256, 256), interpolation=cv2.INTER_NEAREST)
            prompt_scale = 256.0 / source_crop.shape[1]
            px, py, pw, ph = object_in_crop
            prompt_box = (
                _cv_round(px * prompt_scale),
                _cv_round(py * prompt_scale),
                max(4, _cv_round(pw * prompt_scale)),
                max(4, _cv_round(ph * prompt_scale)),
            )
            bx, by, bw, bh = prompt_box
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, 256 - bx), min(bh, 256 - by)
            prompt_box = (bx, by, max(0, bw), max(0, bh))
            if use_box_prompt and bw >= 12 and bh >= 12:
                calibrated_width = max(4, _cv_round(0.95 * bw))
                calibrated_height = max(4, _cv_round(0.95 * bh))
                prompt_box = (
                    bx + (bw - calibrated_width) // 2,
                    by + (bh - calibrated_height) // 2,
                    calibrated_width,
                    calibrated_height,
                )

            cv2.setRNGSeed(_stable_instance_seed(label_stem, component))
            started = time.perf_counter()
            color_masks = build_color_masks(crop, external)
            candidates, _ = detect_sign_shapes_advanced(crop, color_masks, external)
            best = select_best_sign(candidates, external)
            selected = candidates[best] if best >= 0 else None
            selected_source = _source_mark(selected.proposal_source) if selected is not None else "-"
            selected_score = selected.selection_score if selected is not None else 0.0
            if use_box_prompt and prompt_box[2] > 0 and prompt_box[3] > 0:
                box_info = classify_contour(_sampled_rectangle(prompt_box), (256, 256), external)
                if box_info is not None:
                    detected_outer = False
                    if selected is not None and selected.shape is not SignShape.RECTANGLE:
                        sx = selected.bbox[2] / max(1, prompt_box[2])
                        sy = selected.bbox[3] / max(1, prompt_box[3])
                        prompt_center = (
                            prompt_box[0] + 0.5 * prompt_box[2],
                            prompt_box[1] + 0.5 * prompt_box[3],
                        )
                        dx = abs(selected.box[0][0] - prompt_center[0]) / max(1, prompt_box[2])
                        dy = abs(selected.box[0][1] - prompt_center[1]) / max(1, prompt_box[3])
                        detected_outer = sx >= 0.78 and sy >= 0.78 and dx <= 0.16 and dy <= 0.16
                    if not detected_outer:
                        selected = box_info
                        selected_source = "B"
                        selected_score = 1.0
            prediction = np.zeros((256, 256), np.uint8)
            found = SignShape.UNKNOWN
            if selected is not None:
                found = selected.shape
                prediction, _, _ = refine_shape_mask(crop, selected, color_masks, external)
                detected += 1
            latency_ms = (time.perf_counter() - started) * 1000.0
            metrics = evaluate_binary_mask(prediction, truth)
            # Downsample the prediction back to the untouched source annotation
            # grid. Upscaled metrics alone can disguise small-object errors.
            native_prediction = cv2.resize(
                prediction, (source_truth.shape[1], source_truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            native_metrics = evaluate_binary_mask(native_prediction, source_truth)
            size_measurements.append((size_bin, metrics, native_metrics))
            for field in ("iou", "dice", "precision", "recall", "boundary_iou"):
                setattr(totals, field, getattr(totals, field) + getattr(metrics, field))
            summary.evaluated += 1
            summary.mean_latency_ms += latency_ms
            summary.strong_masks += metrics.iou >= 0.80 and metrics.boundary_iou >= 0.55
            summary.weak_masks += metrics.iou < 0.50
            summary.zero_overlap += metrics.iou == 0.0
            iou_values.append(metrics.iou)
            boundary_values.append(metrics.boundary_iou)
            rows.append(
                (
                    sample_name, source_path.name, component, x, y, width, height, original_pixels,
                    found.name, len(candidates), selected_source, f"{selected_score:.5f}",
                    f"{latency_ms:.5f}", f"{metrics.iou:.5f}", f"{metrics.dice:.5f}",
                    f"{metrics.precision:.5f}", f"{metrics.recall:.5f}", f"{metrics.boundary_iou:.5f}",
                    size_bin, int(legacy_eligible), f"{native_metrics.iou:.5f}",
                    f"{native_metrics.dice:.5f}", f"{native_metrics.boundary_iou:.5f}",
                )
            )
            if save_predictions:
                artifacts = (
                    (prediction_dir / f"{sample_name}.png", prediction),
                    (truth_dir / f"{sample_name}.png", truth),
                    (diagnostic_dir / f"{sample_name}.png", _camvid_diagnostic(crop, truth, prediction, metrics)),
                )
                for artifact_path, artifact in artifacts:
                    if not cv2.imwrite(str(artifact_path), artifact):
                        raise OSError(f'could not write image "{artifact_path}"')

    with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_CAMVID_HEADER)
        writer.writerows(rows)
    _write_camvid_size_reports(output, inventory, size_measurements)
    if summary.evaluated == 0:
        raise ValueError("no eligible CamVid SignSymbol regions were found")
    count = float(summary.evaluated)
    summary.mean_iou = totals.iou / count
    summary.mean_dice = totals.dice / count
    summary.mean_precision = totals.precision / count
    summary.mean_recall = totals.recall / count
    summary.mean_boundary_iou = totals.boundary_iou / count
    summary.mean_latency_ms /= count
    report = "\n".join(
        (
            "================ CamVid exact-mask summary ==============",
            "dataset type           : real dashboard street scenes",
            f"evaluated split        : {requested_split}",
            f"context scale          : {context_scale:.2f}x target max side",
            f"box prompt             : {'yes' if use_box_prompt else 'no'}",
            f"size track             : {'all components (includes tiny)' if include_tiny else 'legacy eligible components'}",
            "metric grids           : primary 256x256; native source-crop metrics in CSV",
            "localization contract  : ground-truth-centred crops, not autonomous full frames",
            "rng protocol           : deterministic Python OpenCV 32-bit seed (not C++ 64-bit bit parity)",
            "ground truth class     : " + (
                "CamVid32 SignSymbol RGB(192,128,128), TrafficLight excluded"
                if use_original_labels else "CamVid11 merged SignSymbol class ID 6"
            ),
            f"paired labelled files : {paired_frames}",
            f"files containing signs: {labelled_frames}",
            f"connected sign regions: {connected_regions}",
            f"tiny regions excluded : {skipped_tiny}  (<10 px side or <30 source pixels)",
            f"regions evaluated      : {summary.evaluated}",
            f"non-empty predictions : {detected}/{summary.evaluated}",
            f"mean IoU               : {summary.mean_iou:.3f}",
            f"mean Dice              : {summary.mean_dice:.3f}",
            f"mean precision         : {summary.mean_precision:.3f}",
            f"mean recall            : {summary.mean_recall:.3f}",
            f"mean Boundary-IoU      : {summary.mean_boundary_iou:.3f}",
            f"median IoU             : {_percentile(iou_values, 0.50):.3f}",
            f"10th-percentile IoU    : {_percentile(iou_values, 0.10):.3f}",
            f"median Boundary-IoU    : {_percentile(boundary_values, 0.50):.3f}",
            f"strong masks           : {summary.strong_masks}/{summary.evaluated}  (IoU >= .80 and Boundary-IoU >= .55)",
            f"weak masks             : {summary.weak_masks}/{summary.evaluated}  (IoU < .50)",
            f"zero-overlap masks     : {summary.zero_overlap}/{summary.evaluated}",
            f"mean pipeline latency  : {summary.mean_latency_ms:.2f} ms/crop",
            f"measured throughput    : {1000.0 / summary.mean_latency_ms if summary.mean_latency_ms else 0.0:.1f} crops/s",
            f"incompatible pairs     : {incompatible_frames}",
            "=========================================================",
        )
    ) + "\n"
    (output / "evaluation_summary.txt").write_text(report, encoding="utf-8")
    return summary
