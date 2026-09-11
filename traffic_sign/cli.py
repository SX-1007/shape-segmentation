"""Command-line interface for the pure-Python traffic-sign pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .evaluation import evaluate_camvid_dataset, evaluate_segmentation_dataset
from .models import BatchSummary, EvaluationSummary, ShapeParams
from .pipeline import _image_paths, process_image, run_batch
from .self_test import run_self_tests
from .synthetic import generate_synthetic_background_dataset
from .visualization import make_result_panel, make_stage_panel


def _write_image(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f'could not write image "{path}"')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_sign",
        description="Pure-Python traffic-sign shape detection and segmentation.",
    )
    parser.add_argument("input_root", nargs="?", default="Test_84_Signs", help="input image directory")
    parser.add_argument("-batch", "--batch", action="store_true", help="process all images without windows")
    parser.add_argument("-v", "--verbose", action="store_true", help="print additional batch diagnostics")
    parser.add_argument("--output", default="Outputs_Python", help="batch output directory")
    parser.add_argument("--labels", default="shape_labels.txt", help="optional shape-label file")
    parser.add_argument("--self-test", action="store_true", help="run the 15 deterministic checks")
    parser.add_argument("--synthetic-test", action="store_true", help="generate and evaluate 64 exact-mask scenes")
    parser.add_argument("--synthetic-output", default="Synthetic_Background_Test_Python")
    parser.add_argument("--evaluate-segmentation", metavar="ROOT", help="evaluate ROOT/images against ROOT/masks")
    parser.add_argument("--evaluate-camvid", metavar="ROOT", help="evaluate a CamVid dataset")
    parser.add_argument("--external-output", default="External_Test_Results/CamVid_Python")
    parser.add_argument("--camvid-split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--camvid-context", type=float, default=3.0)
    parser.add_argument("--camvid-label-root")
    parser.add_argument("--camvid-box-prompt", action="store_true")
    parser.add_argument(
        "--camvid-include-tiny", action="store_true",
        help="evaluate all sign components and report accuracy by original size (separate from legacy scores)",
    )
    parser.add_argument("--no-external-predictions", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="disable GrabCut for lower latency at a small mask-quality cost",
    )
    return parser


def _print_batch(summary: BatchSummary) -> None:
    print(
        f"Processed {summary.processed}; recognized {summary.recognized}; "
        f"correct {summary.correct}; valid non-empty masks {summary.non_empty_masks}."
    )


def _print_evaluation(summary: EvaluationSummary) -> None:
    print(
        f"Evaluated {summary.evaluated}; mean IoU {summary.mean_iou:.3f}; "
        f"Dice {summary.mean_dice:.3f}; Boundary-IoU {summary.mean_boundary_iou:.3f}."
    )


def _run_interactive(input_root: Path, output_root: Path, params: ShapeParams) -> int:
    paths = _image_paths(input_root)
    if not paths:
        raise FileNotFoundError(f'no PNG or JPG image found under "{input_root}"')
    print(f'{len(paths)} image(s) found under "{input_root}"')
    for path in paths:
        source = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if source is None:
            raise OSError(f'could not decode input image "{path}"')
        result = process_image(source, params, include_trimap=True)
        panel = make_stage_panel(source, result)
        cv2.imshow("Traffic-sign segmentation", panel)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break
        if key in (ord("s"), ord("S")):
            output_root.mkdir(parents=True, exist_ok=True)
            _write_image(output_root / f"{path.name}_stages.png", panel)
            _write_image(output_root / f"{path.name}_result.png", make_result_panel(source, result))
            _write_image(output_root / f"{path.name}_mask.png", result.final_mask)
            _write_image(output_root / f"{path.name}_segmented.png", result.segmented)
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    params = ShapeParams(refine_with_grabcut=not args.fast)
    try:
        if args.self_test:
            return run_self_tests()
        if args.synthetic_test:
            root = Path(args.synthetic_output)
            if not generate_synthetic_background_dataset(root, 64):
                raise OSError(f'could not generate synthetic dataset below "{root}"')
            _print_evaluation(evaluate_segmentation_dataset(root, params, True))
            return 0
        if args.evaluate_segmentation:
            _print_evaluation(evaluate_segmentation_dataset(Path(args.evaluate_segmentation), params, True))
            return 0
        if args.evaluate_camvid:
            summary = evaluate_camvid_dataset(
                Path(args.evaluate_camvid),
                Path(args.external_output),
                params,
                save_predictions=not args.no_external_predictions,
                split=args.camvid_split,
                context_scale=args.camvid_context,
                original_label_root=Path(args.camvid_label_root) if args.camvid_label_root else None,
                use_box_prompt=args.camvid_box_prompt,
                include_tiny=args.camvid_include_tiny,
            )
            _print_evaluation(summary)
            return 0
        if args.batch:
            summary = run_batch(
                Path(args.input_root),
                Path(args.output),
                Path(args.labels),
                args.verbose,
                params,
            )
            _print_batch(summary)
            return 0
        return _run_interactive(Path(args.input_root), Path(args.output), params)
    except (FileNotFoundError, ValueError, OSError, cv2.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
