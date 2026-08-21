# Python Traffic-Sign Pipeline Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a Python-only runtime with the complete capability, segmentation quality, and practical computational cost of the existing C++ traffic-sign application while retaining the C++ sources as reference files.

**Architecture:** Build a focused `traffic_sign` package that ports the established OpenCV pipeline without a C++ runtime bridge. Keep geometry/proposal work, refinement, evaluation, synthetic data, visualization, and CLI orchestration separate; connect them through typed dataclasses and a single `process_image` result contract. Develop test-first and compare every full-pipeline milestone with the saved C++ baseline.

**Tech Stack:** Python 3.12, OpenCV-Python 4.11, NumPy, pytest, standard-library argparse/csv/dataclasses/pathlib/statistics/time.

**Spec:** `docs/superpowers/specs/2026-08-21-python-port-design.md`

## Global Constraints

- The production runtime must execute entirely through Python source and normal Python packages.
- It must not compile, launch, load, or communicate with `ShapeDetection.exe` or any project-specific C++ extension.
- Preserve the existing C++ sources, scripts, datasets, and reports without deletion or behavioral edits.
- Support Python 3.12 and OpenCV-Python compatible with OpenCV 4.11.
- Do not add a learned model, GPU requirement, network dependency, or heavyweight framework.
- Do not use image-wide Python pixel loops; use vectorized NumPy/OpenCV operations.
- Preserve CLI modes, output names, CSV schemas, deterministic seeds, mask dtypes, and documented semantics.
- Acceptance requires 15/15 self-tests, 84/84 supplied-set classifications and integrity checks, synthetic mean IoU 0.896, Dice 0.934, Boundary-IoU 0.728 at three decimals, at least 59/64 synthetic shape accuracy, and zero zero-overlap masks.
- Profile material slowdowns before optimizing; do not exchange quality for speed.

## File Map

- Create `traffic_sign/__init__.py`: stable public API exports.
- Create `traffic_sign/__main__.py`: `python -m traffic_sign` entry point.
- Create `traffic_sign/models.py`: enums, parameter/result dataclasses, shape names and colours.
- Create `traffic_sign/masks.py`: input validation, colour evidence, morphology, and binary-mask helpers.
- Create `traffic_sign/detection.py`: contour geometry, proposals, evidence fusion, deduplication, and selection.
- Create `traffic_sign/refinement.py`: ideal priors, boundary snapping, rim recovery, GrabCut, and fallback validation.
- Create `traffic_sign/pipeline.py`: one-image orchestration and result contract.
- Create `traffic_sign/metrics.py`: binary-mask metrics and Boundary-IoU.
- Create `traffic_sign/synthetic.py`: deterministic synthetic dataset generation.
- Create `traffic_sign/evaluation.py`: custom-dataset and CamVid evaluators and reports.
- Create `traffic_sign/visualization.py`: overlays, captions, and stage/result panels.
- Create `traffic_sign/cli.py`: argument parsing, interactive/batch runs, and evaluation dispatch.
- Create `tests/`: unit, integration, regression, and optional external-data tests.
- Create `requirements.txt`, `requirements-dev.txt`, and later `run_python.bat`: reproducible Python environment and launcher.
- Modify `.gitignore`: ignore Python coverage/profiling artifacts not already covered.
- Modify `README.md` and `README_ShapeDetection.md`: Python-first setup, commands, results, and C++ reference status.

---

### Task 1: Python Package Contracts and Reproducible Environment

**Files:**
- Create: `traffic_sign/__init__.py`
- Create: `traffic_sign/models.py`
- Create: `tests/test_models.py`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: field names and defaults from `ShapeDetect.h:35-259` and `ShapeDetect.cpp:84-176`.
- Produces: `SignShape`, `ProposalSource`, `ShapeParams`, `ShapeInfo`, `RefinementDiagnostics`, `MaskMetrics`, `PipelineResult`, `BatchSummary`, and `EvaluationSummary` for every later task.

- [ ] **Step 1: Write contract tests before creating the models**

```python
# tests/test_models.py
import numpy as np
from traffic_sign.models import ProposalSource, ShapeInfo, ShapeParams, SignShape


def test_shape_enum_values_match_cpp_contract() -> None:
    assert [member.value for member in SignShape] == [0, 1, 2, 3, 4]
    assert SignShape.CIRCLE.name == "CIRCLE"


def test_parameter_defaults_are_independent_and_match_cpp() -> None:
    first = ShapeParams()
    second = ShapeParams()
    assert first.selection_size_weight == 0.35
    first.th_red = -1.0
    assert second.th_red != -1.0


def test_shape_info_owns_independent_empty_arrays() -> None:
    first = ShapeInfo()
    second = ShapeInfo()
    first.contour = np.array([[[1, 2]]], dtype=np.int32)
    assert second.contour.size == 0
```

- [ ] **Step 2: Run the model tests and verify collection fails**

Run: `python -m pytest tests/test_models.py -v`

Expected: FAIL during import because `traffic_sign.models` does not exist.

- [ ] **Step 3: Add pinned dependencies, launcher, enums, and dataclasses**

Use these dependency files:

```text
# requirements.txt
numpy==2.2.6
opencv-python==4.11.0.86
```

```text
# requirements-dev.txt
-r requirements.txt
pytest==8.4.1
```

Implement `SignShape(IntEnum)` with `UNKNOWN=0`, `TRIANGLE=1`,
`RECTANGLE=2`, `OCTAGON=3`, `CIRCLE=4`, and `ProposalSource(IntEnum)` with
`COLOR=0`, `EDGE=1`, `HOUGH=2`. Translate every `ShapeParams` default from
`ShapeDetect.cpp:84-167` exactly, using snake_case names. Define `ShapeInfo`
with contiguous empty `int32` point arrays from `default_factory`, plus every
measurement and decision field in `ShapeDetect.h:54-119`.

Define these shared result types:

```python
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
```

Export the stable model names from `traffic_sign/__init__.py`. Add `.coverage`,
`htmlcov/`, and `*.prof` to `.gitignore`.

- [ ] **Step 4: Install dependencies and run the contract tests**

Run: `python -m pip install -r requirements-dev.txt`

Run: `python -m pytest tests/test_models.py -v`

Expected: all tests PASS and `python -c "import cv2; print(cv2.__version__)"`
prints a version beginning with `4.11`.

- [ ] **Step 5: Commit the package contracts**

```powershell
git add traffic_sign/__init__.py traffic_sign/models.py tests/test_models.py requirements.txt requirements-dev.txt .gitignore
git commit -m "feat: establish Python traffic-sign contracts"
```

### Task 2: Binary Metrics and Boundary-Safe Self-Test Foundation

**Files:**
- Create: `traffic_sign/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `MaskMetrics` from `traffic_sign.models`.
- Produces: `binary8(mask) -> np.ndarray`, `evaluate_binary_mask(prediction, ground_truth) -> MaskMetrics`, and `inner_boundary(mask, width) -> np.ndarray`.

- [ ] **Step 1: Write metric edge-case tests**

```python
# tests/test_metrics.py
import numpy as np
from traffic_sign.metrics import evaluate_binary_mask


def test_two_empty_masks_are_an_exact_match() -> None:
    empty = np.zeros((32, 32), np.uint8)
    result = evaluate_binary_mask(empty, empty)
    assert result.iou == result.dice == result.precision == result.recall == 1.0
    assert result.boundary_iou == 1.0


def test_disjoint_masks_have_zero_overlap() -> None:
    left = np.zeros((40, 40), np.uint8)
    right = left.copy()
    left[4:14, 4:14] = 255
    right[24:34, 24:34] = 255
    result = evaluate_binary_mask(left, right)
    assert result.iou == result.dice == result.boundary_iou == 0.0


def test_boundary_iou_handles_an_object_touching_the_image_border() -> None:
    truth = np.zeros((64, 64), np.uint8)
    truth[0:30, 8:40] = 255
    prediction = truth.copy()
    assert evaluate_binary_mask(prediction, truth).boundary_iou == 1.0
```

- [ ] **Step 2: Run the metric tests and verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: FAIL because `traffic_sign.metrics` does not exist.

- [ ] **Step 3: Port the exact metric implementation**

Port `SegmentationTest.cpp:19-73`: normalize nonzero values to 255-valued
`uint8`, reject mismatched dimensions, compute intersection/union with OpenCV
bitwise operations, and use the same empty-set values. Implement the reference
Boundary-IoU convention with a boundary width of
`max(1, round(0.02 * hypot(rows, cols)))`, explicit one-pixel zero padding,
erosion, and intersection/union of the two boundary bands.

```python
def evaluate_binary_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> MaskMetrics:
    pred = binary8(prediction)
    truth = binary8(ground_truth)
    if pred.shape != truth.shape:
        raise ValueError("prediction and ground truth must have identical dimensions")
    intersection = cv2.countNonZero(cv2.bitwise_and(pred, truth))
    pred_area = cv2.countNonZero(pred)
    truth_area = cv2.countNonZero(truth)
    union = pred_area + truth_area - intersection
    width = max(1, round(0.02 * math.hypot(*pred.shape)))
    pred_boundary = inner_boundary(pred, width)
    truth_boundary = inner_boundary(truth, width)
    return MaskMetrics(
        iou=safe_ratio(intersection, union, 1.0),
        dice=safe_ratio(2.0 * intersection, pred_area + truth_area, 1.0),
        precision=safe_ratio(intersection, pred_area, 1.0 if truth_area == 0 else 0.0),
        recall=safe_ratio(intersection, truth_area, 1.0),
        boundary_iou=boundary_overlap(pred_boundary, truth_boundary),
    )
```

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest tests/test_metrics.py tests/test_models.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit metric behavior**

```powershell
git add traffic_sign/metrics.py tests/test_metrics.py
git commit -m "feat: port reference mask metrics"
```

### Task 3: Colour Evidence and Morphological Cleanup

**Files:**
- Create: `traffic_sign/masks.py`
- Create: `tests/test_masks.py`

**Interfaces:**
- Consumes: `ShapeParams`.
- Produces: `require_bgr8`, `require_mask8`, `clean_mask`, `build_color_masks`, and `build_color_candidate_mask`.

- [ ] **Step 1: Write validation and cleanup tests**

```python
# tests/test_masks.py
import cv2
import numpy as np
import pytest
from traffic_sign.masks import build_color_masks, clean_mask
from traffic_sign.models import ShapeParams


def test_non_bgr_input_has_an_explicit_error() -> None:
    with pytest.raises(ValueError, match="8-bit 3-channel BGR"):
        build_color_masks(np.zeros((40, 40), np.uint8), ShapeParams())


def test_cleanup_fills_an_enclosed_hole() -> None:
    mask = np.zeros((96, 96), np.uint8)
    cv2.circle(mask, (48, 48), 28, 255, -1)
    cv2.circle(mask, (48, 48), 10, 0, -1)
    cleaned = clean_mask(mask, ShapeParams())
    assert cleaned.dtype == np.uint8
    assert cleaned[48, 48] == 255


def test_primary_colours_remain_separate() -> None:
    image = np.zeros((120, 180, 3), np.uint8)
    image[20:80, 10:50] = (0, 0, 255)
    image[20:80, 70:110] = (255, 0, 0)
    image[20:80, 130:170] = (0, 255, 255)
    red, blue, yellow = build_color_masks(image, ShapeParams())
    assert red[40, 30] and blue[40, 90] and yellow[40, 150]
    assert not blue[40, 30] and not yellow[40, 90] and not red[40, 150]
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run: `python -m pytest tests/test_masks.py -v`

Expected: FAIL importing `traffic_sign.masks`.

- [ ] **Step 3: Port validation, percentile thresholding, and cleanup**

Translate `ShapeDetect.cpp:62-383` exactly. Split BGR channels once as float32,
compute red/blue/yellow enhancement arrays vectorially, calculate the 99.5th
percentile with the same histogram/bin policy, apply the adaptive floor and dark
pixel rules, and keep colour masks independent. Use odd elliptical closing
kernels derived from `morph_frac`/`morph_min`; fill enclosed holes by padded
flood fill. The union helper must use `cv2.bitwise_or` and return `uint8` values
in `{0, 255}`.

- [ ] **Step 4: Run cleanup tests and inspect a representative supplied image**

Run: `python -m pytest tests/test_masks.py tests/test_metrics.py -v`

Run: `python -c "import cv2; from traffic_sign.masks import build_color_masks; from traffic_sign.models import ShapeParams; image=cv2.imread(r'Test_84_Signs/Red Signs/000_1_0002.png'); print([cv2.countNonZero(x) for x in build_color_masks(image, ShapeParams())])"`

Expected: tests PASS and exactly three nonnegative mask-area counts print without
warnings or exceptions.

- [ ] **Step 5: Commit colour segmentation**

```powershell
git add traffic_sign/masks.py tests/test_masks.py
git commit -m "feat: port color masks and cleanup"
```

### Task 4: Contour Classification and Ideal Shape Reconstruction

**Files:**
- Create: `traffic_sign/detection.py`
- Create: `tests/test_classification.py`

**Interfaces:**
- Consumes: `ShapeInfo`, `ShapeParams`, `SignShape`, `ProposalSource`, and mask validation.
- Produces: `classify_contour(contour, image_size, params) -> ShapeInfo | None`, `detect_sign_shapes(masks, params, rejected=None) -> list[ShapeInfo]`, `select_best_sign(candidates, params) -> int`, and `build_shape_mask(info, image_size) -> np.ndarray`.

- [ ] **Step 1: Write ideal-shape and abstention tests**

Create one 256x256 binary mask each for a filled circle, triangle, rectangle,
and regular octagon. Parameterize the expected class and assert that contour
classification and shape-mask reconstruction succeed:

```python
@pytest.mark.parametrize("shape,expected", IDEAL_CASES)
def test_ideal_shapes_classify_and_rebuild(shape: np.ndarray, expected: SignShape) -> None:
    contour = max(cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    info = classify_contour(contour, (shape.shape[1], shape.shape[0]), ShapeParams())
    assert info is not None and info.shape is expected
    rebuilt = build_shape_mask(info, (shape.shape[1], shape.shape[0]))
    assert rebuilt.dtype == np.uint8 and cv2.countNonZero(rebuilt) > 0


def test_empty_and_unknown_only_candidates_abstain() -> None:
    params = ShapeParams()
    assert detect_sign_shapes(np.zeros((128, 128), np.uint8), params) == []
    unknown = ShapeInfo(shape=SignShape.UNKNOWN, selection_score=1.0)
    assert select_best_sign([unknown], params) == -1
```

- [ ] **Step 2: Run classification tests and verify failure**

Run: `python -m pytest tests/test_classification.py -v`

Expected: FAIL because the detection functions do not exist.

- [ ] **Step 3: Port geometric measurement and model scoring**

Translate `ShapeDetect.cpp:385-985` in original operation order. Preserve contour
and hull point shapes as contiguous `int32`; compute area, perimeter, hull area,
hull perimeter, circularity, solidity, extent, aspect, rim ratio, normalized
circle/rectangle/triangle fit, three polygon simplifications, border contact,
and model distances. Use `cv2.fitEllipseAMS` only with at least five points and
fall back exactly as C++ does. Copy the circle/triangle/rectangle/octagon gates,
weights, octagon bias, rejection strings, and ideal-outline construction without
retuning.

`detect_sign_shapes` must accept either one mask or a three-mask sequence, keep
colours separate, run fine/coarse/split proposal passes, suppress overlap using
the same IoU and nesting rules, and populate `rejected` when provided.

- [ ] **Step 4: Run the first eight Python equivalents of C++ self-tests**

Run: `python -m pytest tests/test_classification.py tests/test_masks.py tests/test_metrics.py -v`

Expected: all tests PASS, including four class decisions and four nonempty mask
reconstructions.

- [ ] **Step 5: Commit the geometric classifier**

```powershell
git add traffic_sign/detection.py tests/test_classification.py
git commit -m "feat: port contour shape classification"
```

### Task 5: Advanced Proposals, Evidence Fusion, and Selection

**Files:**
- Modify: `traffic_sign/detection.py`
- Create: `tests/test_detection.py`

**Interfaces:**
- Consumes: Task 4 detector primitives and three colour masks.
- Produces: `build_adaptive_edge_map(bgr, params) -> np.ndarray` and `detect_sign_shapes_advanced(bgr, masks, params, rejected=None) -> tuple[list[ShapeInfo], np.ndarray]`.

- [ ] **Step 1: Write deterministic proposal tests**

Use a synthetic coloured circle plus a larger uncoloured carrier rectangle to
assert that the selected candidate is the circle, and use two overlapping
colour/edge copies to assert deduplication retains one physical proposal. Add a
real regression test for `027_0012.png` asserting `CIRCLE` selection.

```python
def test_roundabout_regression_selects_physical_circle() -> None:
    image = cv2.imread(str(ROOT / "Test_84_Signs/Blue Signs/027_0012.png"))
    masks = build_color_masks(image, ShapeParams())
    candidates, edges = detect_sign_shapes_advanced(image, masks, ShapeParams())
    selected = select_best_sign(candidates, ShapeParams())
    assert edges.dtype == np.uint8
    assert selected >= 0 and candidates[selected].shape is SignShape.CIRCLE
```

- [ ] **Step 2: Run proposal tests and confirm missing advanced functions**

Run: `python -m pytest tests/test_detection.py -v`

Expected: FAIL importing `build_adaptive_edge_map` or
`detect_sign_shapes_advanced`.

- [ ] **Step 3: Port advanced proposal and evidence logic**

Translate `ShapeDetect.cpp:987-1260`: adaptive grayscale median thresholds,
Canny, closing, external edge contours, optional Hough circles, edge-support
distance transforms, dominant colour, colour coverage/capture, Lab boundary
contrast, centre proximity, selection score, and bounding-box/mask IoU
deduplication. Keep Hough disabled by default and preserve all minimum-support,
coverage, weighting, and crop-size thresholds. Cache the Lab image, edge distance
map, and colour union once per input image.

- [ ] **Step 4: Run proposal tests and a label-only smoke subset**

Run: `python -m pytest tests/test_detection.py tests/test_classification.py -v`

Run a temporary test parameterization over the five repaired examples listed in
`results_console.txt`, asserting their labels from `shape_labels.txt`.

Expected: all focused tests PASS without per-pixel Python loops.

- [ ] **Step 5: Commit advanced detection**

```powershell
git add traffic_sign/detection.py tests/test_detection.py
git commit -m "feat: port advanced sign proposal selection"
```

### Task 6: Shape-Constrained Refinement

**Files:**
- Create: `traffic_sign/refinement.py`
- Create: `tests/test_refinement.py`

**Interfaces:**
- Consumes: selected `ShapeInfo`, source BGR image, three colour masks, and `ShapeParams`.
- Produces: `refine_shape_mask(bgr, info, color_masks, params, include_trimap=False) -> tuple[np.ndarray, np.ndarray | None, RefinementDiagnostics]`.

- [ ] **Step 1: Write refinement integrity and repaired-example tests**

Test that every return is contiguous binary `uint8`, one connected foreground
component, and bounded to the image. Add regressions for the thin circular rim
in `001_0004.png` and the complete triangle in `033_1_0001.png`: assert the
final mask contains at least the C++ diagnostic area/span ratios recorded in
`Outputs/refinement_diagnostics.csv` within a one-pixel/OpenCV rounding tolerance.

- [ ] **Step 2: Run refinement tests and verify the module is missing**

Run: `python -m pytest tests/test_refinement.py -v`

Expected: FAIL importing `traffic_sign.refinement`.

- [ ] **Step 3: Port prior snapping, rim recovery, and GrabCut acceptance**

Translate `ShapeDetect.cpp:1262-1637`. Implement the odd-kernel helper,
binary-boundary support, affine scaling about a centre, outward-prior search,
circular colour-rim evidence, and constrained GrabCut trimap. Seed OpenCV RNG
per input where the C++ path does. Preserve foreground/background labels,
iteration count, core recall, colour recall, span ratio, area windows, component
cleanup, repair decision strings, prior scale, and fallback behavior exactly.

Return copies only at API boundaries; reuse working arrays inside the function.
Never scan pixels in Python.

- [ ] **Step 4: Run refinement and all primitive tests**

Run: `python -m pytest tests/test_refinement.py tests/test_detection.py tests/test_classification.py tests/test_masks.py tests/test_metrics.py -v`

Expected: all tests PASS and repeated runs produce identical mask checksums.

- [ ] **Step 5: Commit refinement behavior**

```powershell
git add traffic_sign/refinement.py tests/test_refinement.py
git commit -m "feat: port shape constrained refinement"
```

### Task 7: One-Image Pipeline, Visualization, and 84-Image Batch Mode

**Files:**
- Create: `traffic_sign/pipeline.py`
- Create: `traffic_sign/visualization.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_batch_regression.py`

**Interfaces:**
- Consumes: mask, detector, and refinement APIs from Tasks 3-6.
- Produces: `process_image(bgr, params=None, include_trimap=False) -> PipelineResult`, `draw_shape`, `make_stage_panel`, `make_result_panel`, and `run_batch(input_root, output_root, labels_path, verbose=False) -> BatchSummary`.

- [ ] **Step 1: Write pipeline and generated-artifact tests**

```python
def test_process_image_preserves_source_pixels_inside_and_black_outside() -> None:
    source = cv2.imread(str(ROOT / "Test_84_Signs/Red Signs/000_1_0002.png"))
    result = process_image(source)
    assert result.selected is not None
    assert set(np.unique(result.final_mask)).issubset({0, 255})
    assert np.count_nonzero(result.segmented[result.final_mask == 0]) == 0
    assert np.array_equal(result.segmented[result.final_mask != 0], source[result.final_mask != 0])


def test_batch_writes_compatible_artifacts(tmp_path: Path) -> None:
    summary = run_batch(FIXTURE_ROOT, tmp_path, FIXTURE_LABELS)
    assert summary.processed == 1
    assert (tmp_path / "sample_stages.png").is_file()
    assert (tmp_path / "sample_result.png").is_file()
    assert (tmp_path / "sample_mask.png").is_file()
    assert (tmp_path / "sample_segmented.png").is_file()
    assert (tmp_path / "refinement_diagnostics.csv").is_file()
```

- [ ] **Step 2: Run pipeline tests and verify failure**

Run: `python -m pytest tests/test_pipeline.py tests/test_batch_regression.py -v`

Expected: FAIL because pipeline and visualization modules do not exist.

- [ ] **Step 3: Implement orchestration and compatible panels**

Port the per-image flow and reporting from `Source.cpp:149-405`. Time only colour
masks through final refinement. Construct the candidate union and edge image,
select or abstain, return a zero mask on abstention, and composite using
`cv2.copyTo` semantics (`np.zeros_like` plus `cv2.copyTo`). Port shape colours,
labels, contour drawing, captions, six-stage layout, result layout, label-file
loading, confusion matrix, mask-integrity counters, diagnostic CSV header/order,
recursive JPG/PNG discovery, and deterministic case-insensitive sorting.

- [ ] **Step 4: Run the complete supplied 84-image regression**

Run: `python -m pytest tests/test_pipeline.py -v`

Run: `python -c "from pathlib import Path; from traffic_sign.pipeline import run_batch; print(run_batch(Path('Test_84_Signs'), Path('Outputs_Python'), Path('shape_labels.txt')))"`

Expected: 84/84 recognized, 84/84 shape accuracy, and every five mask-integrity
counter at 84/84. Confirm all 336 image artifacts plus one diagnostics CSV exist.

- [ ] **Step 5: Commit the batch pipeline**

```powershell
git add traffic_sign/pipeline.py traffic_sign/visualization.py tests/test_pipeline.py tests/test_batch_regression.py
git commit -m "feat: add Python batch segmentation pipeline"
```

### Task 8: Deterministic Self-Test CLI and Synthetic Exact-Mask Benchmark

**Files:**
- Create: `traffic_sign/synthetic.py`
- Create: `traffic_sign/self_test.py`
- Create: `tests/test_synthetic.py`
- Create: `traffic_sign/evaluation.py` (initial custom-dataset implementation)

**Interfaces:**
- Consumes: `process_image` and `evaluate_binary_mask`.
- Produces: `run_self_tests() -> int`, `generate_synthetic_background_dataset(root, image_count=64) -> bool`, and `evaluate_segmentation_dataset(root, params, save_predictions=True) -> EvaluationSummary`.

- [ ] **Step 1: Write determinism, schema, and acceptance tests**

Generate a small four-image fixture twice and assert identical SHA-256 hashes for
images, masks, labels, and metadata. Evaluate it and assert the CSV header is:

```text
file,expected_shape,detected_shape,proposal_count,selected_source,selection_score,selected_area_ratio,latency_ms,iou,dice,precision,recall,boundary_iou
```

Add a test that `run_self_tests()` returns zero only after exactly 15 named checks
pass.

- [ ] **Step 2: Run synthetic tests and verify failure**

Run: `python -m pytest tests/test_synthetic.py -v`

Expected: FAIL because generator/evaluator/self-test functions do not exist.

- [ ] **Step 3: Port generator, evaluator, and 15-test reporter**

Translate `SegmentationTest.cpp:75-497`: implement the OpenCV multiply-with-carry
RNG state transition used by `cv::RNG` for scalar drawing decisions, seed OpenCV
array operations explicitly, and preserve scene order, backgrounds, sign templates, homographies,
compression artifacts, labels, metadata, filenames, percentile interpolation,
per-shape aggregates, strong/weak/zero rules, CSV formatting, and summary text.
The evaluator must stream one image/mask pair at a time and write predictions
only when requested.

The self-test reporter must call the Python primitive tests through equivalent
in-process assertions and print the existing `[PASS]`/`[FAIL]` names followed by
`Self-test result: 15 passed, 0 failed.` on success.

- [ ] **Step 4: Run the full 64-image quality gate**

Run: `python -m pytest tests/test_synthetic.py -v`

Run: `python -c "from pathlib import Path; from traffic_sign.synthetic import generate_synthetic_background_dataset; from traffic_sign.evaluation import evaluate_segmentation_dataset; from traffic_sign.models import ShapeParams; root=Path('Synthetic_Background_Test_Python'); generate_synthetic_background_dataset(root, 64); print(evaluate_segmentation_dataset(root, ShapeParams()))"`

Expected: 64 evaluated; rounded mean IoU at least 0.896, Dice at least 0.934,
Boundary-IoU at least 0.728; shape accuracy at least 59/64; zero-overlap masks
equal zero. If any gate fails, compare per-image CSV rows and intermediate masks
against C++ before changing any algorithm threshold.

- [ ] **Step 5: Commit deterministic evaluation**

```powershell
git add traffic_sign/synthetic.py traffic_sign/self_test.py traffic_sign/evaluation.py tests/test_synthetic.py
git commit -m "feat: port deterministic segmentation evaluation"
```

### Task 9: CamVid Held-Out Evaluation Parity

**Files:**
- Modify: `traffic_sign/evaluation.py`
- Create: `tests/test_camvid.py`

**Interfaces:**
- Consumes: `process_image`, shape-supported box prompts, and metric/report helpers.
- Produces: `evaluate_camvid_dataset(dataset_root, output_root, params, save_predictions=True, split="all", context_scale=3.0, original_label_root=None, use_box_prompt=False) -> EvaluationSummary`.

- [ ] **Step 1: Write CamVid protocol tests using a tiny local fixture**

Construct one synthetic frame plus CamVid32 label mask with a border-touching
SignSymbol component. Assert split filtering, centred square context with padding,
256x256 normalization, component count, original-label preference, box-prompt
selection, output paths, and `save_predictions=False` behavior.

- [ ] **Step 2: Run CamVid tests and verify missing implementation**

Run: `python -m pytest tests/test_camvid.py -v`

Expected: FAIL because `evaluate_camvid_dataset` is not implemented.

- [ ] **Step 3: Port the corrected external protocol**

Translate `SegmentationTest.cpp:498-942`: suffix/stem normalization, split-list
loading, source/label matching, original CamVid32 SignSymbol extraction,
connected components, centred context with explicit border padding, sampled box
contour, box-supported candidate injection, per-instance deterministic RNG,
diagnostic panel, prediction/truth outputs, CSV, and aggregate summary. Preserve
the scientific limitation label that box-prompt mode is segmentation given a
location, not end-to-end detection.

- [ ] **Step 4: Run fixture and available held-out data**

Run: `python -m pytest tests/test_camvid.py -v`

If `External_Datasets/CamVid/raw` and `raw32_labels` contain the full test data,
run the documented 181-component held-out command with
`--no-external-predictions`. Expected schema and count: 181 rows. Compare against
saved reference aggregates (IoU 0.782, Dice 0.872, Boundary-IoU 0.311) and
investigate unexplained differences before acceptance.

- [ ] **Step 5: Commit external evaluation**

```powershell
git add traffic_sign/evaluation.py tests/test_camvid.py
git commit -m "feat: port CamVid segmentation evaluation"
```

### Task 10: Complete Python CLI and Documentation

**Files:**
- Create: `traffic_sign/cli.py`
- Create: `traffic_sign/__main__.py`
- Create: `run_python.bat`
- Create: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `README_ShapeDetection.md`

**Interfaces:**
- Consumes: batch, self-test, synthetic, custom, CamVid, and visualization APIs.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv=None) -> int`, and all documented `python -m traffic_sign` commands.

- [ ] **Step 1: Write CLI parsing and dispatch tests**

Test all aliases and missing-value errors without processing full datasets.
Monkeypatch dispatch functions and assert exact argument forwarding for
`--batch`, `--verbose`, `--self-test`, `--synthetic-test`,
`--evaluate-segmentation`, `--evaluate-camvid`, `--external-output`,
`--camvid-split`, `--camvid-context`, `--camvid-label-root`,
`--camvid-box-prompt`, and `--no-external-predictions`. Assert `main()` returns
nonzero with a concise error for unreadable input.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL because the CLI and module entry point do not exist.

- [ ] **Step 3: Implement dispatch and Python-first documentation**

Use argparse while accepting the original positional-folder and flag shapes.
Default to `Test_84_Signs` and interactive display; retain Escape/advance/save
controls. Catch expected `ValueError`, `FileNotFoundError`, and OpenCV I/O errors
at the boundary, print one actionable message to stderr, and return a nonzero
code. Do not catch programming errors broadly. Implement `run_python.bat` as
`@python -m traffic_sign %*` followed by `@exit /b %errorlevel%`.

Update both READMEs with virtual-environment setup, pinned installation,
`run_python.bat` and module commands, output viewing, expected metrics, low-cost
design notes, and troubleshooting. State that C++ is retained as reference and
never used by the Python runtime.

- [ ] **Step 4: Run CLI tests and command smoke tests**

Run: `python -m pytest tests/test_cli.py -v`

Run: `python -m traffic_sign --self-test`

Run: `run_python.bat --self-test`

Expected: both commands report 15 passed, 0 failed and return exit code zero.

- [ ] **Step 5: Commit CLI and docs**

```powershell
git add traffic_sign/cli.py traffic_sign/__main__.py run_python.bat tests/test_cli.py README.md README_ShapeDetection.md
git commit -m "feat: expose complete Python traffic-sign CLI"
```

### Task 11: Full Parity Audit and Computational-Cost Optimization

**Files:**
- Create: `tests/test_quality_gates.py`
- Create: `benchmarks/profile_pipeline.py`
- Create: `PYTHON_PORT_RESULTS.md`
- Modify: Python modules only where profiling identifies measured hot paths.

**Interfaces:**
- Consumes: the complete Python application and existing C++ reports/executable as development baselines.
- Produces: reproducible quality/performance evidence and the accepted Python runtime.

- [ ] **Step 1: Encode non-regression gates before final optimization**

Create slow-marked tests that parse batch and synthetic summaries and assert:

```python
assert batch.processed == batch.recognized == batch.correct == 84
assert batch.non_empty_masks == batch.binary_masks == 84
assert batch.one_component_masks == batch.black_outside_masks == 84
assert batch.source_identical_masks == 84
assert round(synthetic.mean_iou, 3) >= 0.896
assert round(synthetic.mean_dice, 3) >= 0.934
assert round(synthetic.mean_boundary_iou, 3) >= 0.728
assert synthetic.shape_correct >= 59
assert synthetic.zero_overlap == 0
```

- [ ] **Step 2: Run the full suite and both complete benchmarks**

Run: `python -m pytest -v`

Run: `python -m traffic_sign --batch --output Outputs_Python`

Run: `python -m traffic_sign --synthetic-test`

Expected: all tests and quality gates PASS. Save terminal results and compare
Python per-image labels, proposal sources, diagnostics, and masks with C++
outputs. Classify differences as harmless numeric rounding or defects.

- [ ] **Step 3: Profile measured pipeline time without output I/O**

Create a profiler entry point that loads the already-generated synthetic images,
runs `process_image` once per image, and performs no visualization or file writes:

```python
# benchmarks/profile_pipeline.py
from pathlib import Path
import cv2
from traffic_sign.pipeline import process_image

for image_path in sorted((Path("Synthetic_Background_Test_Python") / "images").glob("*.png")):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {image_path}")
    process_image(image)
```

Run: `python -m cProfile -o python_pipeline.prof benchmarks/profile_pipeline.py`

Inspect cumulative time with:

```powershell
python -c "import pstats; pstats.Stats('python_pipeline.prof').sort_stats('cumulative').print_stats(30)"
```

Optimize only demonstrated Python overhead: replace scalar/list image work with
NumPy/OpenCV, hoist invariant kernels/conversions, reuse masks, and remove
unnecessary `.copy()` calls. Do not alter thresholds, proposal gates, GrabCut
iterations, or refinement fallbacks for speed.

- [ ] **Step 4: Re-run every quality gate after optimization and document evidence**

Run: `python -m pytest -v`

Run both 84-image and 64-image commands again. Record dependency versions,
hardware-independent command lines, all accuracy metrics, mean pipeline latency,
throughput, and any CamVid result in `PYTHON_PORT_RESULTS.md`. Explicitly state
that measured Python runtime has no project C++ executable/extension dependency
and distinguish pipeline timing from image generation and disk output.

- [ ] **Step 5: Commit the verified port evidence**

```powershell
git add tests/test_quality_gates.py benchmarks/profile_pipeline.py PYTHON_PORT_RESULTS.md traffic_sign
git commit -m "test: verify Python quality and performance parity"
```

### Task 12: Final Verification and Handoff

**Files:**
- Verify only; modify documentation solely if a command or recorded value is inaccurate.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a clean, reproducible handoff with no unsupported success claim.

- [ ] **Step 1: Verify repository state and Python-only imports**

Run: `git status --short`

Run: `rg -n "ShapeDetection\.exe|subprocess|ctypes|cffi|pybind|\.dll" traffic_sign tests`

Expected: only explicit negative/reference assertions may mention the executable;
no runtime bridge imports or calls exist.

- [ ] **Step 2: Execute the final command matrix from a fresh Python process**

Run self-test, full pytest, 84-image batch, synthetic benchmark, and the available
CamVid/custom fixture commands. Record exit codes and verify expected files open
with OpenCV and have correct shapes/dtypes.

- [ ] **Step 3: Check documentation and generated schemas**

Compare README commands with `python -m traffic_sign --help`. Compare Python CSV
headers and report fields with the C++ schemas. Run `git diff --check` and scan
tracked Python files for placeholder markers.

- [ ] **Step 4: Apply the verification-before-completion skill**

Read and follow `superpowers:verification-before-completion`; report fresh test,
quality, and timing evidence rather than relying on earlier output.

- [ ] **Step 5: Commit any final documentation correction**

If verification required a documentation-only correction:

```powershell
git add README.md README_ShapeDetection.md PYTHON_PORT_RESULTS.md
git commit -m "docs: finalize Python port verification"
```

If no correction was required, do not create an empty commit.
