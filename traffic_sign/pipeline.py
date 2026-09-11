"""End-to-end image processing and batch orchestration."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .detection import detect_sign_shapes_advanced, select_best_sign
from .masks import build_color_candidate_mask, require_bgr8
from .models import (
    BatchSummary,
    PipelineResult,
    ProposalSource,
    RefinementDiagnostics,
    ShapeParams,
    SignShape,
)
from .refinement import refine_shape_mask
from .visualization import make_result_panel, make_stage_panel


_DIAGNOSTIC_HEADER = (
    "file",
    "shape",
    "source",
    "prior_area",
    "graphcut_area",
    "final_area",
    "area_ratio",
    "core_recall",
    "color_recall",
    "span_ratio",
    "prior_scale",
    "accepted",
    "decision",
)


def process_image(
    bgr: np.ndarray,
    params: ShapeParams | None = None,
    include_trimap: bool = False,
) -> PipelineResult:
    require_bgr8(bgr, "process_image")
    settings = params if params is not None else ShapeParams()
    start = time.perf_counter()
    candidate_mask, color_masks = build_color_candidate_mask(bgr, settings)
    rejected = []
    candidates, edge_map = detect_sign_shapes_advanced(
        bgr,
        color_masks,
        settings,
        rejected,
    )
    selected_index = select_best_sign(candidates, settings)
    if selected_index >= 0:
        final_mask, trimap, diagnostics = refine_shape_mask(
            bgr,
            candidates[selected_index],
            color_masks,
            settings,
            include_trimap=include_trimap,
        )
    else:
        final_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        trimap = None
        diagnostics = RefinementDiagnostics(decision="no sign selected")
    segmented = np.zeros_like(bgr)
    cv2.copyTo(bgr, final_mask, segmented)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return PipelineResult(
        color_masks=color_masks,
        candidate_mask=candidate_mask,
        edge_map=edge_map,
        candidates=candidates,
        rejected=rejected,
        selected_index=selected_index,
        final_mask=final_mask,
        segmented=segmented,
        trimap=trimap,
        refinement=diagnostics,
        latency_ms=latency_ms,
    )


def _load_labels(path: Path | None) -> dict[str, SignShape]:
    labels: dict[str, SignShape] = {}
    if path is None or not path.is_file():
        return labels
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] in SignShape.__members__:
            shape = SignShape[parts[1]]
            if shape is not SignShape.UNKNOWN:
                labels[parts[0]] = shape
    return labels


def _image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f'input directory does not exist: "{root}"')
    supported = {".png", ".jpg", ".jpeg"}
    return sorted(
        (path for path in root.rglob("*") if path.suffix.lower() in supported),
        key=lambda path: str(path).lower(),
    )


def _source_mark(source: ProposalSource) -> str:
    return {
        ProposalSource.COLOR: "C",
        ProposalSource.EDGE: "E",
        ProposalSource.HOUGH: "H",
    }[source]


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f'could not write image "{path}"')


def _print_batch_report(image_path: Path, result: PipelineResult, verbose: bool) -> None:
    print(f"\n{image_path.name}")
    display_limit = len(result.candidates) if verbose else min(3, len(result.candidates))
    for index, candidate in enumerate(result.candidates[:display_limit]):
        print(
            f"    candidate {index}: {candidate.shape.name} "
            f"source {_source_mark(candidate.proposal_source)} "
            f"model {candidate.score:.3f} selection {candidate.selection_score:.3f} "
            f"edge {candidate.edge_support:.3f} colour {candidate.color_coverage:.3f}"
        )
    if len(result.candidates) > display_limit:
        print(f"    ... {len(result.candidates) - display_limit} lower-ranked proposal(s) suppressed")
    if not result.candidates:
        print("    no candidate region survived the proposal gates")
        for index, rejected in enumerate(result.rejected):
            print(f"    rejected {index}: {rejected.reject or 'unknown reason'}")
    if result.selected is None:
        print("    ==> no sign selected")
        return
    selected = result.selected
    diagnostics = result.refinement
    print(
        f"    ==> selected region {result.selected_index}: {selected.shape.name} "
        f"(score {selected.score:.2f})"
    )
    print(
        f"        refinement: {diagnostics.decision} | prior scale {diagnostics.prior_scale:.2f} "
        f"| area {diagnostics.final_area}/{diagnostics.prior_area} "
        f"| core {diagnostics.core_recall:.2f} | colour {diagnostics.color_recall:.2f} "
        f"| span {diagnostics.span_ratio:.2f}"
    )


def run_batch(
    input_root: Path | str,
    output_root: Path | str,
    labels_path: Path | str | None = None,
    verbose: bool = False,
    params: ShapeParams | None = None,
) -> BatchSummary:
    input_path = Path(input_root)
    output_path = Path(output_root)
    label_path = Path(labels_path) if labels_path is not None else None
    images = _image_paths(input_path)
    if not images:
        raise FileNotFoundError(f'no PNG or JPG image found under "{input_path}"')
    labels = _load_labels(label_path)
    settings = params if params is not None else ShapeParams()
    output_path.mkdir(parents=True, exist_ok=True)
    summary = BatchSummary()
    diagnostics_path = output_path / "refinement_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_DIAGNOSTIC_HEADER)
        for image_path in images:
            source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if source is None:
                raise OSError(f'could not decode input image "{image_path}"')
            summary.processed += 1
            result = process_image(source, settings)
            _print_batch_report(image_path, result, verbose)
            selected = result.selected
            detected_shape = selected.shape if selected is not None else SignShape.UNKNOWN
            if selected is not None and detected_shape is not SignShape.UNKNOWN:
                summary.recognized += 1
                diagnostics = result.refinement
                writer.writerow(
                    (
                        image_path.name,
                        detected_shape.name,
                        _source_mark(selected.proposal_source),
                        diagnostics.prior_area,
                        diagnostics.graphcut_area,
                        diagnostics.final_area,
                        diagnostics.area_ratio,
                        diagnostics.core_recall,
                        diagnostics.color_recall,
                        diagnostics.span_ratio,
                        diagnostics.prior_scale,
                        int(diagnostics.accepted),
                        diagnostics.decision,
                    )
                )
            if labels.get(image_path.name) is detected_shape:
                summary.correct += 1
            if cv2.countNonZero(result.final_mask) > 0:
                summary.non_empty_masks += 1
            if set(np.unique(result.final_mask)).issubset({0, 255}):
                summary.binary_masks += 1
            if cv2.connectedComponents(result.final_mask, connectivity=8)[0] == 2:
                summary.one_component_masks += 1
            if np.count_nonzero(result.segmented[result.final_mask == 0]) == 0:
                summary.black_outside_masks += 1
            if np.array_equal(
                result.segmented[result.final_mask != 0],
                source[result.final_mask != 0],
            ):
                summary.source_identical_masks += 1

            prefix = image_path.name
            _write_image(output_path / f"{prefix}_stages.png", make_stage_panel(source, result))
            _write_image(output_path / f"{prefix}_result.png", make_result_panel(source, result))
            _write_image(output_path / f"{prefix}_mask.png", result.final_mask)
            _write_image(output_path / f"{prefix}_segmented.png", result.segmented)
    return summary
