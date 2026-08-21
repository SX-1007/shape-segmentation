# Python Traffic-Sign Pipeline Port Design

## Objective

Port the complete traffic-sign shape detection and segmentation application from
C++ to Python while preserving the existing C++ sources as an untouched
reference. The production runtime must execute entirely through Python source
and normal Python packages. It must not compile, launch, load, or communicate
with `ShapeDetection.exe` or any project-specific C++ extension.

The port must retain the current shape-classification and segmentation quality,
avoid unnecessarily expensive computation, and preserve all existing user-facing
workflows and generated artifacts.

## Runtime and dependencies

- Support the installed Python 3.12 runtime.
- Use `opencv-python` compatible with OpenCV 4.11 and NumPy for optimized image
  and array operations.
- Use Python standard-library modules for command-line parsing, paths, CSV,
  statistics, timing, and dataclasses where practical.
- Pin compatible dependency versions so benchmark results are reproducible.
- Do not add a machine-learning model, GPU requirement, network dependency, or
  heavyweight framework.
- Do not implement image-wide pixel processing as Python loops. Colour
  transforms, masks, morphology, contours, distance operations, GrabCut, and
  compositing must use vectorized NumPy or OpenCV operations.

Although OpenCV and NumPy contain compiled internals, all project-owned runtime
code and orchestration will be Python. There will be no C++ build or executable
dependency.

## Architecture

The new `traffic_sign` package will contain these focused modules:

- `models.py`: `IntEnum` and `dataclass` equivalents of `SignShape`,
  `ProposalSource`, `ShapeInfo`, `ShapeParams`, and `RefinementDiagnostics`.
- `detection.py`: colour-mask generation and cleanup, adaptive edge extraction,
  contour measurement and classification, colour/edge/Hough proposals,
  evidence fusion, candidate deduplication, and best-sign selection.
- `refinement.py`: ideal shape-mask construction, trimap generation, constrained
  GrabCut, completeness validation, colour repair, and evidence-backed rim
  recovery.
- `evaluation.py`: region metrics, reference-compatible Boundary-IoU, dataset
  discovery, prediction output, CSV generation, and aggregate summaries.
- `synthetic.py`: deterministic generation of the existing 64-image exact-mask
  stress dataset.
- `visualization.py`: shape overlays, captions, processing-stage mosaics, and
  interactive OpenCV windows.
- `cli.py`: argument parsing and orchestration for interactive, batch, self-test,
  synthetic, custom-dataset, and CamVid modes.
- `__main__.py`: entry point for `python -m traffic_sign`.

Tests will live under `tests/`. A dependency file and Windows Python launcher
will provide reproducible setup and commands without changing or removing the
existing C++ build and run scripts.

## Behavioral compatibility

The Python CLI will support the existing capabilities:

```text
python -m traffic_sign
python -m traffic_sign <input-folder>
python -m traffic_sign --batch
python -m traffic_sign <input-folder> --batch
python -m traffic_sign --self-test
python -m traffic_sign --synthetic-test
python -m traffic_sign --evaluate-segmentation <root>
python -m traffic_sign --evaluate-camvid <root> [CamVid options]
```

The port will retain verbose reporting, CamVid split/context/label-root/box-prompt
options, configurable external output, and the option not to save external
predictions. Invalid inputs will produce clear messages and nonzero exit codes
instead of OpenCV assertions or Python tracebacks for expected user errors.

Batch mode will retain the four generated images per input and
`refinement_diagnostics.csv`. Evaluation modes will retain prediction masks,
diagnostics, `evaluation.csv`, and `evaluation_summary.txt`, including compatible
filenames and column ordering.

## Data flow

For each input image, the pipeline will:

1. Validate and load a BGR image.
2. Build independent red, blue, and yellow masks with the existing adaptive
   thresholds and morphological cleanup.
3. Create colour-contour proposals and independent edge/Hough proposals.
4. Measure geometric descriptors and classify circle, triangle, rectangle,
   octagon, or unknown using the current thresholds and scoring formulas.
5. Calculate boundary, colour, contrast, location, and scale evidence; deduplicate
   proposals and select the strongest valid sign.
6. Build an ideal geometric prior and refine it with the current constrained
   GrabCut, completeness, repair, and fallback rules.
7. Produce the binary mask, source-identical segmented image, diagnostics,
   visualizations, and optional evaluation metrics.

Algorithm constants, ordering, OpenCV flags, random seeds, rounding behavior,
and mask dtypes will be ported deliberately. Arrays passed to OpenCV will use
contiguous `uint8`, `int32`, or floating-point representations as required.

## Computational-cost controls

- Read and decode each image only when needed; do not retain an entire dataset
  in memory.
- Reuse computed masks, contours, distance transforms, and proposal measurements
  within an image.
- Avoid redundant colour-space conversions and array copies.
- Restrict GrabCut and Hough work to the same evidence-driven paths as the C++
  implementation.
- Keep diagnostic image generation optional where the current CLI permits it.
- Use deterministic, sequential processing initially. Parallel execution will
  not be added unless measurement proves it beneficial, because OpenCV already
  uses optimized native operations and uncontrolled nested parallelism can add
  overhead.
- Measure pipeline latency separately from dataset generation, file writing, and
  display time, matching the existing benchmark meaning.

## Verification and acceptance criteria

The C++ executable and its saved reports provide the baseline only; they are not
part of the Python runtime. Verification will proceed from small deterministic
checks to full datasets.

### Deterministic tests

Port all 15 existing self-tests. The required result is 15 passed and 0 failed,
covering empty/disjoint masks, border-safe Boundary-IoU, invalid input,
UNKNOWN-only abstention, cleanup and hole filling, all four ideal shapes, and
nonempty 8-bit mask reconstruction.

### Supplied 84-image regression

Required results:

- 84 images processed and recognized;
- 84/84 correct shape classifications;
- 84/84 nonempty binary masks;
- 84/84 masks with one foreground component;
- 84/84 segmented images black outside the mask;
- 84/84 segmented images source-identical inside the mask.

Where deterministic OpenCV behavior permits, Python and C++ labels, selected
proposal sources, and masks will also be compared directly to expose accidental
porting differences.

### Synthetic exact-mask regression

The regenerated deterministic 64-image set must preserve the current rounded
quality results:

- mean IoU: at least 0.896 when rounded to three decimals;
- mean Dice: at least 0.934 when rounded to three decimals;
- mean Boundary-IoU: at least 0.728 when rounded to three decimals;
- shape accuracy: at least 59/64;
- zero-overlap masks: 0/64.

Precision, recall, median and 10th-percentile IoU, strong/weak mask counts, and
per-shape results will be reported and checked for unexplained regressions.

### External evaluation and performance

If the complete local CamVid inputs are present, exercise the fixed held-out test
command and compare its schema, sample count, and aggregate results with the
saved baseline. This remains a box-supported segmentation evaluation rather than
an end-to-end detection claim.

Benchmark Python on the same machine and report mean latency and throughput.
The port must introduce no C++ process-launch or IPC latency. Quality gates take
priority over a misleading speed claim, but any material slowdown will be
profiled and optimized by removing Python-level loops, duplicate computation,
or unnecessary copies before acceptance.

## Documentation and transition

Update the main documentation with Python installation, dependency setup, all
commands, expected benchmark results, and troubleshooting. Clearly identify the
C++ implementation as retained reference code and the Python package as the
runtime implementation. Existing C++ sources, scripts, datasets, and reports
will not be deleted.

## Out of scope

- Replacing the classical pipeline with a learned model.
- Claiming perfect generalization beyond the supplied and external benchmarks.
- Removing or rewriting the existing C++ reference implementation.
- Adding a GUI framework beyond the current OpenCV interactive demonstration.
- Optimizing through a new project-specific native extension.
