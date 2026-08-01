# Technical audit: traffic-sign shape detection for segmentation support

Audit date: 1 August 2026  
Implementation: C++17, OpenCV 4.11  
Scope: one dominant traffic sign in an already cropped BGR image

## 1. Executive verdict

The implementation is now a strong, reproducible **classical vision baseline**
for the scope above. It is not scientifically defensible to call it perfect or
production-ready.

The distinction matters:

- all 84 supplied crops now receive the correct shape label, but those files do
  not have pixel-level ground-truth masks;
- the deterministic 64-image exact-mask stress set reaches 0.896 mean IoU,
  0.934 Dice, and 0.728 Boundary-IoU, but it is synthetic and its generator is
  in the same repository as the method being developed;
- one dense same-colour crossing still reduces a synthetic octagon to almost no
  overlap, demonstrating a real limit of colour/edge geometry;
- the code assumes a cropped, mostly dominant sign and is not a multi-instance
  traffic-sign detector for complete road scenes;
- thresholds encode the supplied colour and sign-design distribution. They have
  not yet been validated across countries, cameras, seasons, weather, severe
  occlusion, retroreflection, motion blur, or domain shift.

The current system is therefore suitable for an explainable academic project,
for generating proposals, and as a deterministic fallback. For a high-reliability
system, the recommended next tier is a hybrid: use this module to produce a
shape/box/mask prompt, use a learned segmenter to resolve semantic ambiguity,
validate the learned mask against the geometric evidence, and abstain or fall
back when the two disagree.

## 2. What “perfect enough” must mean

“Shape detection to support segmentation” contains several separate tasks. A
single accuracy percentage cannot establish success.

1. **Proposal recall**: does at least one proposal cover the physical sign?
2. **Proposal selection**: is the physical sign selected instead of a pictogram,
   lamp, advertisement, or background object?
3. **Shape classification**: is the selected plate labelled circle, triangle,
   rectangle, or octagon correctly?
4. **Mask fidelity**: does the final mask include the complete plate and exclude
   the background, including a thin coloured rim?
5. **Boundary fidelity**: is the outline accurate enough for later recognition?
6. **Failure detection**: does the system know when evidence is insufficient?
7. **Operational performance**: is latency, memory, and stability acceptable on
   the actual deployment hardware?
8. **Generalization**: do the results persist on independently collected,
   human-annotated, never-tuned data?

The current code verifies items 1–5 on deterministic test cases and partially
verifies item 8 on supplied shape labels. It does not yet provide calibrated
abstention, external pixel-mask validation, or a deployment-specific latency
guarantee.

## 3. Reproduced architecture

| Stage | Current implementation | Purpose |
|---|---|---|
| Input guard | Require non-empty `CV_8UC3` BGR images and matching `CV_8UC1` masks | Turn OpenCV assertions into explicit contract failures |
| Colour evidence | HSV plus normalized red/blue/yellow channels | Reduce illumination sensitivity and isolate sign-colour evidence |
| Proposal cleanup | Scale-aware close/fill plus retained raw masks | Produce complete plate geometry without corrupting later evidence measurement |
| Bridge splitting | Two proposal-only morphological openings | Recover a sign contour when a thin same-colour object is attached |
| Edge evidence | CLAHE, median-derived Canny, scale-aware closing | Recover boundaries when colour covers only a rim or interior |
| Geometry | Contours, convex hull, normalized aspect, solidity, polygon approximations, circle/rectangle/triangle fits | Classify four plate shapes in an explainable way |
| Evidence fusion | Edge support, colour coverage/capture, Lab boundary contrast, geometry, crop scale, centre prior | Rank the physical plate instead of the best isolated contour |
| Nested reasoning | Border-pair and carrier/pictogram rules | Resolve outer plate versus inner graphic ambiguity |
| Mask refinement | Shape prior, four-state GrabCut, raw-colour repair, connected-component selection, hole fill, guarded fallback | Preserve the complete plate while allowing a pixel-level boundary |
| Diagnostics | CSV audit trail plus pixel-mask evaluation | Make fallbacks and regressions inspectable |

The architecture is appropriate for the project constraints because every
decision remains reproducible from the image, parameters, and OpenCV version.

## 4. Findings in the pre-audit implementation

### 4.1 High-impact correctness findings

#### A. Selection underweighted the crop contract

The selector assigned only 10% of the final decision to candidate scale. That is
too weak for a module whose documented input is an already-cropped sign. Small
pictograms and sign-coloured clutter could outrank the physical plate despite
the larger candidate having credible geometry and boundary support.

**Correction:** scale evidence is configurable and defaults to 35%, leaving 65%
for fused appearance/geometry evidence. Nested-carrier safeguards prevent this
change from blindly selecting a weak border-touching rectangle.

#### B. Same-colour bridges were irreversible

Closing and hole filling improve fragmented borders, but they can also preserve
or reinforce a narrow bridge between a sign and a distractor. A contour method
then sees one concave merged object; downstream convex-hull fitting cannot infer
which pixels belonged to the sign.

**Correction:** retain the original proposal streams and add two scale-aware
opening streams solely for candidate generation. This is important: the opened
mask does not replace evidence or the final segmentation. It only supplies a
candidate that can still be rejected by the normal evidence gates.

#### C. `colorCapture` was not a valid fraction

The former numerator and denominator were measured over inconsistent support.
Observed values exceeded 1.0 (for example, about 1.42), so the quantity was not a
probability-like coverage score and could distort candidate ranking.

**Correction:** numerator and denominator now use the same local bounding-box
support and the result is clamped to `[0,1]`.

#### D. Boundary-IoU was non-reference

The older test used a 1.4%-diagonal elliptical band. The authors' released
Boundary-IoU convention uses a 2%-of-image-diagonal distance and handles image
edges by padding before erosion. Reporting the old score as if it were the
published metric would overstate comparability.

**Correction:** the evaluator uses a 2% diagonal band, repeated 3×3 square
erosion, and explicit border padding. The older 0.598 and current 0.728 results
are marked non-comparable rather than presented as an improvement on the same
metric.

### 4.2 Reliability and maintainability findings

- Full-frame union-colour masks, Lab conversion, and edge distance transforms
  were being recomputed per proposal. They are now cached once per image.
- Empty or incorrectly typed inputs could reach OpenCV operations and fail with
  low-context assertions. Public stages now check input type, size, and emptiness.
- An empty candidate mask was not explicitly tested. It is now a deterministic
  no-result case.
- Evaluation emphasized means, which can hide catastrophic tails. It now reports
  median and 10th-percentile IoU, weak and exact-zero counts, per-shape results,
  candidate metadata, latency, and throughput.
- There was no fast regression suite. `--self-test` now exercises empty masks,
  border-touching Boundary-IoU, disjoint masks, UNKNOWN-only abstention, invalid
  BGR input, cleanup/hole filling, all four ideal shapes, and reconstruction of
  each shape mask.

### 4.3 Existing strengths retained

- normalized colour channels supplement HSV rather than relying on fixed raw
  BGR thresholds;
- raw evidence is kept separate from morphology-cleaned proposal masks;
- convex-hull geometry makes border rings and pictogram holes less destructive;
- several polygon tolerances reduce sensitivity to one arbitrary
  `approxPolyDP` epsilon;
- Lab inner/outer contrast adds boundary evidence independent of sign colour;
- GrabCut is constrained by a geometric prior instead of being asked to discover
  the object from a loose rectangle;
- final validation checks area, core recall, colour recall, and both spatial
  spans before accepting GrabCut;
- refinement can fall back to the complete prior instead of returning a visibly
  incomplete but numerically plausible component.

## 5. Changes made in this audit

| Change | Expected effect | Observed verification |
|---|---|---|
| Configurable 35% selection-scale weight | Prefer physical plate in cropped inputs | Supplied shape result remains 84/84 |
| Generalized, evidence-gated carrier rule | Preserve credible outer boards even at crop edge | Prevented regressions introduced by stronger scale evidence |
| Correct local `colorCapture` | Keep evidence bounded and comparable | No capture value can exceed 1.0 |
| Two-scale proposal-only opening | Split narrow same-colour attachments | Removed former zero-overlap synthetic selections; corrected five shape labels in aggregate |
| Cached image-level evidence | Avoid repeated `O(pixels × proposals)` transforms | Same decisions with less redundant work |
| Reference-compatible Boundary-IoU | Make boundary claims interpretable | New score is 0.728; old score explicitly non-comparable |
| Input validation, abstention, and empty-candidate handling | Controlled failure rather than OpenCV assertion or largest-blob guess | Included in 15/15 self-tests |
| Expanded CSV and distribution metrics | Expose tail risk and selection causes | One weak case and no exact-zero cases are now visible |

## 6. Verified quantitative results

All values below were reproduced after a clean build on 1 August 2026.

### 6.1 Supplied 84-image set

| Measurement | Result | What it proves |
|---|---:|---|
| Images processed | 84/84 | I/O and pipeline completion |
| Shape classification | 84/84 (100.0%) | Correct class on this supplied labelled sample |
| Non-empty binary mask | 84/84 | Structural validity only |
| One foreground component | 84/84 | Structural validity only |
| Black outside mask | 84/84 | Output compositing integrity |
| Source-identical inside mask | 84/84 | No unintended colour changes |

These 84 results do **not** prove segmentation accuracy because there are no
reference pixel masks. “84 valid masks” means the files obey the representation
contract, not that all 84 outlines are correct.

### 6.2 Deterministic 64-image exact-mask stress set

| Metric | Original baseline | Current |
|---|---:|---:|
| Mean IoU | 0.654 | **0.896** |
| Mean Dice | 0.703 | **0.934** |
| Mean precision | 0.765 | **0.912** |
| Mean recall | 0.670 | **0.968** |
| Mean Boundary-IoU | not comparable | **0.728** |
| Median IoU | not recorded | **0.966** |
| 10th-percentile IoU | not recorded | **0.724** |
| Median Boundary-IoU | not recorded | **0.842** |
| Strong masks (`IoU >= .80` and `Boundary-IoU >= .55`) | not comparable | **47/64** |
| Weak masks (`IoU < .50`) | not recorded | **1/64** |
| Exact zero overlap | not recorded | **0/64** |
| Shape classification | 54/64 | **59/64 (92.2%)** |
| Mean measured pipeline time | not recorded | **201.01 ms/image** |
| Measured throughput | not recorded | **5.0 images/s** |

Per-shape segmentation:

| Expected shape | Mean IoU | Mean Boundary-IoU | Count |
|---|---:|---:|---:|
| Circle | 0.912 | 0.698 | 16 |
| Triangle | 0.908 | 0.759 | 16 |
| Rectangle | 0.912 | 0.772 | 16 |
| Octagon | 0.853 | 0.683 | 16 |

### 6.3 Interpretation

- Recall (0.968) exceeds precision (0.912), which is appropriate when the mask
  will support later sign recognition: omitting the rim or pictogram is often
  more damaging than a narrow background fringe. It also shows that leakage,
  rather than missing foreground, is now the larger average error source.
- Mean IoU (0.896) is lower than median IoU (0.966), proving a left-tail failure
  distribution. The mean alone would hide that risk.
- Boundary-IoU (0.728) remains much lower than region IoU, so boundary quality is
  still the principal improvement opportunity.
- Octagons have the lowest mean IoU. There are only 16 synthetic octagons and one
  supplied octagon, so shape-specific claims have wide uncertainty.
- The measured 5.0 images/s is a single local CPU run, not a portable real-time
  guarantee. Resolution, processor, build flags, proposal count, and OpenCV
  threading all affect it.

The latest run averaged 20.5 proposals per image (maximum 40) and had a measured
95th-percentile pipeline latency of 276.61 ms. The lowest-IoU tail was:

| Image | Expected / detected | Proposals | Source | IoU | Boundary-IoU |
|---|---|---:|:---:|---:|---:|
| `synthetic_031.png` | octagon / rectangle | 30 | colour | 0.00016 | 0.00047 |
| `synthetic_054.png` | rectangle / rectangle | 22 | colour | 0.58653 | 0.20283 |
| `synthetic_037.png` | triangle / triangle | 3 | colour | 0.59054 | 0.10321 |
| `synthetic_021.png` | triangle / triangle | 27 | edge | 0.62011 | 0.18839 |
| `synthetic_063.png` | octagon / circle | 33 | edge | 0.64463 | 0.28334 |

This table prevents “zero-overlap masks: 0” from being misread: one mask has a
tiny but nonzero 0.00016 IoU and is still correctly counted as a catastrophic
weak case.

## 7. Remaining failure modes

### 7.1 Evidence-destroying contact

In the remaining weak synthetic image, wide red clutter crosses and connects to
the red octagonal plate. Colour sees one component, edge closure is broken, and
no closed octagon proposal survives. Morphological opening can split a narrow
bridge but cannot remove a wide crossing without also eroding the sign. This is
not safely solvable by choosing a more aggressive fixed kernel.

Professional remedies are semantic instance segmentation, temporal tracking,
multi-view reconstruction, or an explicit occlusion model. The system should
also be able to abstain when proposal evidence is inconsistent.

### 7.2 Domain assumptions

- Colour support is optimized for red, blue, and yellow signs. White, orange,
  green, brown, fluorescent, faded, heavily overexposed, or monochrome signs
  receive much weaker proposal support.
- The classifier exposes only circle, triangle, rectangle, and octagon. Diamonds,
  pentagons, shields, crosses, and country-specific plates become `UNKNOWN` or
  can be forced into the nearest supported class by imperfect evidence.
- The centre and size priors are valid for crops, not full road frames.
- A bounded triangle expansion encodes a common border design; borderless or
  non-standard triangles need separate validation.
- Severe perspective makes an octagon appear closer to a rounded rectangle or
  circle and reduces reliable corner count.
- Tiny signs, blooming highlights, night-time retroreflection, rain, snow,
  shadow boundaries, motion blur, compression, vegetation, stickers, damage,
  and partial crop truncation remain under-sampled.

### 7.3 Statistical limitations

- The supplied set is small and class-imbalanced: 51 circles, 29 triangles,
  three rectangles, and one octagon.
- Parameters were examined against the same supplied and synthetic samples used
  for reporting; this is development performance, not an untouched test result.
- Synthetic masks are exact, but synthetic backgrounds do not reproduce the
  joint distribution of real cameras, materials, optics, and weather.
- No confidence intervals are reported, and individual rare classes have too few
  observations for a narrow interval.
- There is no negative-only dataset, so false positives per image are not yet
  measured.
- There is no multi-instance evaluation, AP, panoptic metric, video stability
  metric, or calibration measurement.

## 8. Professional techniques reviewed

### 8.1 Classical, self-contained techniques

| Technique | Value for this task | Recommendation |
|---|---|---|
| HSV/Lab/normalized RGB | Handles hue and illumination better than fixed BGR | Already used; add colour calibration tests for each camera |
| Adaptive thresholding and photometric normalization | Helps shadows and local exposure | CLAHE is already used for edges; Retinex/colour constancy should be tested, not assumed beneficial |
| Morphological reconstruction/open-close by scale | Repairs gaps and separates attachments | Already used conservatively; retain originals to avoid destructive preprocessing |
| Contours, hulls, polygon approximation | Fast, explainable geometry | Already used; test epsilon stability and resolution scaling |
| Hough circle/line transforms | Recovers incomplete analytic shapes | Keep optional; unrestricted Hough creates convincing pictogram false positives |
| RANSAC ellipse/polygon fitting | More robust to outlier boundary points | Worth testing for partial occlusion if runtime allows |
| Hu moments/Fourier descriptors/templates | Add shape information beyond vertex counts | Useful as a secondary score, but sensitive to occlusion and domain-specific templates |
| MSER/selective-search/superpixels | Alternative region proposals | Only add if measured proposal recall improves without unacceptable candidate growth |
| Watershed/graph cuts/active contours | Refine image-aligned boundaries | GrabCut is already constrained by geometry; watershed may help touching objects but is marker-sensitive |
| Temporal optical flow/tracking | Stabilizes video and resolves intermittent occlusion | Strong next step if sequential frames exist |

OpenCV's official contour feature documentation covers polygon approximation,
convex hull, rectangle fitting, and related descriptors. The original GrabCut
work combines iterative graph cuts with foreground/background colour models;
the current refinement uses that professional pattern but constrains it with the
detected sign geometry.

### 8.2 Learned techniques

| Family | Strength | Limitation / use here |
|---|---|---|
| Mask R-CNN-style instance segmentation | Joint object detection and per-instance masks | Needs representative labelled masks; default coarse mask heads can blur small boundaries |
| Mask2Former / Mask DINO | Modern transformer-based detection and segmentation | Higher integration/training cost; best choice when enough domain data and GPU resources exist |
| PointRend | Concentrates high-resolution predictions at uncertain boundaries | Excellent boundary head for small signs; still needs a trained base model |
| SegFix-like boundary repair | Corrects uncertain boundary labels using reliable interiors | Principle is reflected in current guarded colour/prior repair; a learned version needs training data |
| SAM 3 concept segmentation | Text/exemplar prompt can locate and segment matching concepts | Current strongest open-vocabulary research path; large runtime/checkpoint and domain validation required |
| SAM 3.1 | Faster joint multi-object video tracking in the SAM 3 family | Particularly relevant if this project expands to video; not a reason to claim perfect still-image masks |
| SAM 2.1 | Promptable box/mask image and video segmentation | Natural drop-in refinement using this module's mask/box as prompt |
| EfficientSAM | Smaller promptable segmentation models, with deployment-oriented paths | Easier edge experiment; must be benchmarked on small traffic signs rather than selected only by model size |

No paper or foundation model can establish perfection on this project's target
distribution without an independent, task-specific evaluation. Model adoption
must be based on measured error slices, not publication-wide averages.

## 9. Recommended target architecture

### Tier A: self-contained academic system

Keep the current implementation. Add confidence/abstention and freeze the
current 64 images as regression tests. This tier is transparent, requires no
external weights, and directly demonstrates shape detection.

### Tier B: practical hybrid system

1. A road-scene detector produces candidate traffic-sign boxes, or the existing
   crop contract is retained.
2. The current module generates shape, hull, box, and a conservative mask.
3. SAM 3/SAM 2.1 or a lightweight promptable segmenter receives box plus mask
   prompts.
4. A validator compares learned output against colour recall, shape fit, edge
   support, area/span growth, and agreement with the classical prior.
5. Accept the learned mask only when validation passes; otherwise use the
   classical prior or return `UNKNOWN/LOW_CONFIDENCE`.
6. If video is available, track the instance and combine evidence across frames.

This architecture is stronger than replacing everything with a foundation
model because it keeps deterministic diagnostics and a safe fallback while
adding semantic reasoning for occlusion and same-colour clutter.

### Tier C: task-specific production system

Train or fine-tune an instance segmenter on representative traffic-sign masks,
using a high-resolution/boundary head and hard-negative mining. Use SAM-family
masks to accelerate annotation, but have humans correct them. Retain the
classical geometry as a consistency check and interpretable fallback.

## 10. Required independent validation plan

### 10.1 Data

Use data that was not used to tune thresholds:

- **Mapillary Vistas Validation for Traffic Signs (MVV, 2025):** particularly
  relevant because it has expert-annotated pixel-level instance masks and
  fine-grained traffic-sign labels.
- **Mapillary Vistas:** diverse, high-resolution street scenes with dense polygon
  annotations and broad geographic/condition coverage.
- **TT100K:** a large Chinese traffic-sign benchmark whose annotations can
  include polygon masks; useful for domain and design shift.
- **GTSDB:** 900 road images and zero-to-six signs per image; useful for detection
  and negative/full-frame behavior, but its primary annotations are detection
  oriented rather than a substitute for mask ground truth.
- **Local deployment captures:** required for the exact camera, lens, mounting,
  compression, routes, weather, and jurisdiction.

Split by route/site/capture session rather than random neighboring frames. Keep
one final test set sealed until every parameter and model choice is frozen.

### 10.2 Annotation protocol

- Define whether the mask includes the physical outer rim, mounting hardware,
  holes, stickers, and occluded-but-inferred regions.
- Double-annotate a representative subset and adjudicate disagreements.
- Measure inter-annotator IoU/Boundary-IoU; a model cannot be meaningfully held
  to an undefined or less-consistent target.
- Store visibility, truncation, damage, blur, illumination, sign size, country,
  material, and weather attributes for slice analysis.

### 10.3 Metrics

Report at least:

- proposal recall at IoU thresholds 0.50, 0.75, and 0.90;
- mask IoU, Dice, precision, recall, and reference Boundary-IoU;
- per-class and macro averages, not only the micro/overall average;
- instance AP/AR if full frames contain multiple signs;
- false positives per negative image;
- failure/abstention coverage curves and risk at each retained coverage;
- 5th, 10th, median, and mean metrics, plus worst-case examples;
- bootstrap 95% confidence intervals;
- latency percentiles, peak memory, and throughput on the target hardware;
- temporal mask stability and identity switches for video.

### 10.4 Robustness tests

Create both natural and controlled slices for:

- sign width bands (for example `<16`, `16–31`, `32–63`, `>=64` pixels);
- perspective/yaw/pitch, partial crop, truncation, and 10–70% occlusion;
- day, night, backlight, glare, retroreflection, fog, rain, and snow;
- defocus, motion blur, sensor noise, rescaling, and JPEG/WebP compression;
- faded paint, dirt, stickers, graffiti, physical damage, and vegetation;
- same-colour adjacent objects and crossings;
- unsupported shapes/colours and frames with no traffic sign;
- adversarial scale changes that exercise every size-dependent kernel.

### 10.5 Software verification

- Run the 15 deterministic tests on every change.
- Treat the 64-image benchmark and supplied 84 labels as frozen regression data.
- Add tests at minimum and maximum accepted image dimensions.
- Run AddressSanitizer/UndefinedBehaviorSanitizer in a compatible Clang/GCC CI
  build, plus MSVC warnings at a strict level.
- Fuzz public image/mask entry points with empty, malformed, huge, and unusual
  stride/channel inputs.
- Verify deterministic results across supported OpenCV builds or explicitly
  document acceptable numerical variation.
- Version parameters, evaluation code, model checkpoints, and dataset hashes.

### 10.6 Suggested acceptance gate

Exact targets must follow downstream recognition needs, but a defensible initial
gate would be:

- no known crash or invalid memory access on the supported input contract;
- at least 0.90 mean and 0.85 10th-percentile mask IoU on the sealed real set;
- at least 0.75 mean Boundary-IoU;
- at least 0.99 proposal recall at IoU 0.50 for signs at or above the minimum
  supported size;
- separately reported performance for every safety-relevant class/slice;
- calibrated abstention so ambiguous cases are not silently forced into a class;
- latency and memory within the actual deployment budget at the 95th percentile.

These are engineering gates, not universal claims. If downstream recognition is
more sensitive to false background than missing rim, the precision/recall trade
must be adjusted and revalidated.

## 11. Reproduction

```bat
build.bat
run.bat --self-test
run.bat -batch
run.bat --synthetic-test
```

Generated evidence:

- `Outputs/refinement_diagnostics.csv`
- `Synthetic_Background_Test/evaluation.csv`
- `Synthetic_Background_Test/evaluation_summary.txt`

## 12. Primary sources consulted

- [OpenCV contour features](https://docs.opencv.org/4.13.0/dd/d49/tutorial_py_contour_features.html)
- [OpenCV GrabCut and watershed APIs](https://docs.opencv.org/master/d3/d47/group__imgproc__segmentation.html)
- [GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf)
- [Boundary IoU, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)
- [Official Boundary-IoU API](https://github.com/bowenc0221/boundary-iou-api)
- [SegFix, ECCV 2020](https://www.microsoft.com/en-us/research/publication/segfix-model-agnostic-boundary-refinement-for-segmentation/)
- [Mask R-CNN, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)
- [PointRend, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)
- [Mask2Former, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)
- [Mask DINO, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.html)
- [SAM 3 concept segmentation, Meta, 2025](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Official SAM 3 repository and SAM 3.1 release](https://github.com/facebookresearch/sam3)
- [SAM 2 image/video segmentation, Meta, 2024](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [EfficientSAM repository](https://github.com/yformer/EfficientSAM)
- [TT100K, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.html)
- [GTSDB official benchmark](https://benchmark.ini.rub.de/gtsdb_dataset.html)
- [Mapillary Vistas, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.html)
- [MVV traffic-sign pixel-mask validation, ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/DataCV/html/Garg_Mapillary_Vistas_Validation_for_Fine-Grained_Traffic_Signs_A_Benchmark_Revealing_ICCVW_2025_paper.html)

## 13. Final assessment

The code is substantially better than the starting point and is now unusually
well-instrumented for a small classical-vision project. The improvement from
0.654 to 0.896 mean synthetic IoU, the 0.966 median, elimination of exact-zero
overlap, 84/84 supplied shape labels, and 15/15 regression tests are meaningful.

They do not prove perfection. The correct scientific conclusion is:

> Strong and reproducible for the stated cropped-sign classroom scope; not yet
> externally validated or semantically powerful enough for safety-critical,
> full-scene, multi-country deployment.

The next unit of effort should go to independent real pixel masks and calibrated
failure handling, followed by a measured hybrid learned refinement. More fixed
thresholds without new independent data would mostly increase overfitting risk.
