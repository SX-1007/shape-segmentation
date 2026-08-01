# Advanced shape-supported traffic-sign segmentation

UCCC2513 mini project - Member 4 module.

This module is deliberately limited to the assigned function: **shape detection
of signs to support segmentation**. It accepts the red, blue, and yellow masks
from the colour members, adds independent edge evidence, identifies the sign
shape, and returns a clean binary mask. It does not implement sign recognition.

The important change from the lecture example is that the final result is no
longer the longest colour contour or a blindly filled enclosing circle. The
system uses a shape prior to initialise a constrained graph-cut refinement, so
the final mask follows the physical plate boundary and excludes background
wedges, nearby posters, foliage, walls, and carrier boards.

## Files

| File | Purpose |
|---|---|
| `ShapeDetect.h`, `ShapeDetect.cpp` | Complete reusable shape/segmentation module. |
| `Source.cpp` | Six-stage demonstration, directory input, batch export, and scoring. |
| `SegmentationTest.h`, `SegmentationTest.cpp` | Exact-mask metrics plus deterministic background-stress dataset generator. |
| `Supp.h`, `supp.cpp` | Lecturer's window-partition helper. |
| `shape_labels.txt` | Visual shape labels for the 84 supplied images; evaluation only. |
| `build.bat`, `run.bat` | Reproducible x64 OpenCV 4.11 build and execution. |
| `Outputs/` | Stage images, raw masks, and final segmented images. |
| `Synthetic_Background_Test/` | 64 generated challenge images, exact masks, predictions, metadata, CSV scores, and summary. |
| `External_Datasets/CamVid/` | Isolated, downloaded real-scene CamVid archive and extracted source data. |
| `External_Test_Results/CamVid32_HeldOut_Test/` | Corrected fixed-test exact-mask results using original CamVid32 labels. |
| `External_Test_Results/CamVid32_Improved/` | All-frame corrected predictions, truth crops, diagnostics, CSV, and summary. |
| `REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md` | Detailed protocol correction, failure taxonomy, ablations, limitations, and reproduction commands. |

## Build and run

```bat
build.bat
run.bat
```

Batch evaluation:

```bat
run.bat -batch
```

Generate the exact-ground-truth background stress set and evaluate it:

```bat
run.bat --synthetic-test
```

Run the fast deterministic regression suite:

```bat
run.bat --self-test
```

Evaluate another dataset laid out as `<root>\images\*.png` and
`<root>\masks\<same-name>.png`:

```bat
run.bat --evaluate-segmentation "D:\my segmentation test"
```

Evaluate the separately downloaded CamVid real-background dataset with the
correct original 32-class target and box-supported segmentation protocol:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw" ^
  --camvid-label-root "External_Datasets\CamVid\raw32_labels" ^
  --camvid-split test --camvid-context 1.5 --camvid-box-prompt ^
  --external-output "External_Test_Results\CamVid32_HeldOut_Test"
```

CamVid is used instead of GTSRB for this test because GTSRB is a cropped-sign
classification benchmark.  CamVid supplies real dashboard scenes and
hand-labelled `SignSymbol` pixels, allowing genuine pixel-mask evaluation.
GTSDB would also provide full backgrounds, but its public ground truth consists
of bounding boxes rather than exact masks.

The batch command recursively reads the supplied directory and writes four files
per input image:

- `<name>_stages.png` - the six processing stages;
- `<name>_result.png` - original and final segmentation side by side;
- `<name>_mask.png` - the raw 0/255 binary mask;
- `<name>_segmented.png` - the isolated sign on a black background.

`Outputs/refinement_diagnostics.csv` records whether GrabCut was accepted or
replaced by the geometric prior, plus area, core, colour, span, and selected
outer-border scale diagnostics for every image. Add `--verbose` to print every
proposal instead of only the three leading proposals.

An alternative input root may be supplied before `-batch`:

```bat
run.bat "D:\my traffic signs" -batch
```

In interactive mode, `Esc` quits, `s` saves the current result, and any other
key advances to the next image.

## Demonstration panels

| Panel | Content |
|---|---|
| 1 | Original image. |
| 2 | Union of adaptive red, blue, and yellow candidate masks. |
| 3 | Three highest-ranked fused colour/edge shape proposals. |
| 4 | Selected shape model and fused confidence. |
| 5 | Shape-constrained GrabCut mask. |
| 6 | Pixel-accurate segmentation produced with `copyTo(mask)`. |

The proposal labels end in `C`, `E`, or `H` for colour contour, edge contour,
or optional Hough source. Hough circles are implemented but disabled by default:
unrestricted voting proposed too many convincing circles on pictograms. This is
an intentional evidence-based setting, not a missing feature.

## Method

### 1. Illumination-normalised colour support

The module combines HSV hue windows with normalised colour enhancement:

```text
fR = max(0, min(R-G, R-B)) / (R+G+B)
fB = max(0, min(B-R, B-G)) / (R+G+B)
fY = max(0, min(R-B, G-B)) / (R+G+B)
```

Each threshold adapts to the 99.5th percentile of the current image. Gaussian
denoising, scale-aware closing, opening, and enclosed-hole filling reconnect
paint that was divided by a pictogram. The three colours remain separate so a
red sign on a blue board cannot be welded into one meaningless contour.

Two forms of this evidence are deliberately retained. The filled masks generate
complete shape proposals; the raw, unfilled masks measure actual paint. This
fixes a subtle previous bug: after hole filling, a white digit or inner circular
pictogram appeared to have 100% traffic-sign colour merely because it lay inside
the filled plate. Raw evidence also identifies the thin red rim that GrabCut
must not remove.

A proposal-only opening stream now complements the fine and coarse closing
passes. It evaluates two opening scales to split a sign from same-colour clutter
connected through a narrow bridge. The original masks are never replaced; an
opened hypothesis must pass the same geometry, colour, edge, contrast, and
non-maximum-suppression gates as every other proposal.

This stage is supporting evidence only. The colour members may pass their own
three masks directly to `detectSignShapesAdvanced()`.

### 2. Independent adaptive edge proposals

Colour-only segmentation fails when paint is faded, illumination is poor, or a
triangle is printed on a larger board. The advanced path therefore creates a
second proposal stream:

1. Convert to grayscale and apply CLAHE local-contrast normalisation.
2. Apply a 5x5 Gaussian blur.
3. Estimate the Canny thresholds from Otsu's threshold on gradient magnitude.
4. Close short edge gaps with an image-scale elliptical kernel.
5. Follow every closed edge contour with `findContours(RETR_LIST)`.

This is what recovers `048_0004`: the old system selected the complete purple
rectangular board; the fused system recovers the inner triangular sign boundary.

### 3. Viewpoint-normalised geometric modelling

Every contour is converted to its convex plate hull, rotated into its
minimum-area-rectangle frame, and normalised to a 100x100 coordinate system.
The following descriptors are then invariant to translation, rotation, scale,
and much of the foreshortening caused by camera viewpoint:

```text
cirFit  = hull area / area(minimum enclosing circle)
rectFit = hull area / area(minimum-area rectangle)
triFit  = hull area / area(minimum enclosing triangle)
```

Polygonal approximation is evaluated at 1%, 3%, and 5% of perimeter. The
multi-scale corner stability separates true polygons from curves more reliably
than a single arbitrary `approxPolyDP` tolerance. The model also uses solidity,
circularity, aspect ratio, rim ratio, area, and border contact.

Round signs seen obliquely are not forced back into a circle. `fitEllipseAMS`
fits the projected ellipse, eliminating the background wedges introduced by a
minimum enclosing circle.

### 4. Evidence fusion and proposal suppression

A region is not selected merely because it is large. Each proposal receives
independent measurements:

- geometric model confidence;
- soft boundary-to-Canny support using a distance transform;
- colour coverage inside the fitted plate;
- local colour capture around the plate;
- Lab colour contrast across its boundary;
- a small scale/area prior.

Because the documented input contract is an already-cropped sign candidate,
scale evidence now contributes 35% of the final pre-hierarchy rank instead of
10%. Tiny, crisp background rectangles no longer win merely because their four
edges are cleaner than the much larger physical plate. The weight is exposed as
`selectionSizeWeight`; reduce it if this module is applied directly to full
street frames.

The fused score prevents a large coloured wall or carrier board from winning on
size alone. Near-duplicate colour and edge models are removed by mask IoU
non-maximum suppression. Nested candidates are handled as a hierarchy:

- an outer circle/triangle/octagon gets a small boost when it is a supported
  second border around the same shape;
- a large circle is treated as a pictogram only when the surrounding rectangle
  has credible rectangular geometry and boundary support;
- a tiny triangular graphic yields to a same-colour outer rectangle only when
  that physical plate has strong fit, edge, coverage, and capture evidence;
- leaked regions touching the image edge cannot set the size normalisation;
- a light centre prior resolves highly cluttered *sign crop* inputs. It is only
  22% of the final rank multiplier, so evidence still dominates.

If every surviving proposal is `UNKNOWN`, selection now returns no sign instead
of falling back to the largest blob. This explicit abstention prevents an
unsupported shape or colour-only false positive from becoming a confident-
looking downstream mask.

The hierarchy fixes both semantic directions without filename-specific code.
For `027_0012`, the high-quality blue circle now defeats the weak, border-
touching rectangle caused by background leakage. For `051_0005_j`, the tiny
printed warning triangle yields to the strongly supported yellow rectangular
physical plate.

### 5. Automatic shape-constrained GrabCut

The ideal model is useful as a prior but is not accepted as the final boundary.
The code automatically creates a four-state GrabCut mask:

- pixels far outside a dilation of the shape are definite background;
- the narrow outer band is probable background;
- the geometric interior is probable foreground;
- an eroded shape core and eroded colour support are definite foreground.

Five GrabCut iterations combine foreground/background colour mixtures with
edge smoothness. The result is restricted to the allowed shape neighbourhood.
The refiner then restores raw sign-colour pixels inside the prior *before*
connected-component selection and hole filling. A low-colour circular interior
receives one additional evidence gate: if raw same-colour pixels form a narrow
surrounding rim, those pixels are protected in the outer GrabCut band and the
area ceiling is relaxed from 1.16 to 1.45 for that case only. This turns a
fragmented red rim back into a closed plate instead of accepting only the pale
centre, without globally permitting larger leaks.

Four independent guards reject unsafe refinement: retained area, core recall,
raw-colour recall, and two-dimensional span. The minimum area ratio was raised
from 0.48 to 0.72, and a cut must retain at least 80% of both the prior width and
height. These checks directly stop the half-mask failure in `033_1_0001`.

Triangular warnings receive one additional boundary hypothesis step. A 1.00 to
1.35 multi-scale family searches for a coherent outer Canny boundary. When a
triangle is almost entirely yellow, indicating that the detected contour is
the yellow/black interface, a bounded 1.28 frame hypothesis recovers the
physical border. Circles and octagons deliberately do not use this fallback,
because curved background edges produced false expansion in testing.

This follows the central idea of the original
[GrabCut paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf):
iterative graph cuts combine colour-region and boundary information. The OpenCV
implementation and its four mask states are documented in
[`cv::grabCut`](https://docs.opencv.org/4.12.0/d7/d1b/group__imgproc__misc.html).
Shape and ellipse operations follow the official
[OpenCV structural-analysis documentation](https://docs.opencv.org/4.12.0/d3/dc0/group__imgproc__shape.html).

## Why not bundle a foundation model?

As of this August 2026 audit, Meta's
[SAM 3 family](https://github.com/facebookresearch/sam3) is the current
open-vocabulary upgrade: it can detect and segment all instances matching a
text concept such as "traffic sign", and also accepts visual prompts. The
[SAM 3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
mainly improves multi-object video tracking efficiency. For a smaller
prompt-only upgrade, [SAM 2.1](https://github.com/facebookresearch/sam2) accepts
boxes and masks, so this module's selected shape prior is a natural prompt.
EfficientSAM also publishes ONNX export support for a lighter deployment path.
It was not silently added here because that would:

- require a large external checkpoint and a second runtime;
- make the submitted C++/OpenCV build non-self-contained;
- obscure the assigned contribution, which is shape detection;
- make an 84-image classroom demonstration harder to reproduce.

The implemented system therefore uses the most capable self-contained approach
that remains explainable line by line in the module: multi-cue proposals,
geometric invariance, evidence fusion, and graph-cut boundary refinement.

The evaluation now follows the authors' reference implementation of
[Boundary IoU (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html),
including its 2%-of-image-diagonal band, repeated 3x3 erosion, and explicit
padding for objects touching an image border. Boundary IoU is more sensitive to
boundary errors than ordinary mask IoU. The
model-agnostic idea of correcting uncertain boundary pixels from reliable
interior evidence is also consistent with
[SegFix (ECCV 2020)](https://www.microsoft.com/en-us/research/publication/segfix-model-agnostic-boundary-refinement-for-segmentation/),
implemented here with deterministic geometry and raw-colour reconstruction
rather than a learned network.

## Verified results on the supplied 84 images

Latest clean batch run:

```text
images processed              84
non-empty signs recognised    84 / 84
shape accuracy                84 / 84 (100.0%)

ground truth -> detected
CIRCLE       51 circle
TRIANGLE     29 triangle
RECTANGLE     3 rectangle
OCTAGON       1 octagon
```

The immediately preceding advanced implementation reported 83/84 (98.8%); its
remaining nested plate/pictogram ambiguity is now resolved. The older
lecture-style implementation reported 78/84 (92.9%).

All 84 generated masks were automatically checked to confirm that they are:

- non-empty;
- binary (only 0 and 255);
- one connected foreground component;
- black outside the mask;
- pixel-identical to the source image inside the mask.

No pixel-level IoU/Dice claim is made for these 84 files because they have shape
labels but no ground-truth masks. The former "84/84 valid masks" statement only
proved file integrity; it could not detect a removed rim. Visual review remains
necessary, and `refinement_diagnostics.csv` now makes the fallback decisions
auditable.

## Independent real-background test (CamVid)

A public CamVid copy was downloaded from the
[Figshare dataset record](https://doi.org/10.6084/m9.figshare.17080916.v1) into
`External_Datasets/CamVid`, with the published MD5 checksum verified. It is
kept completely separate from `Test_84_Signs`; derived files are written only
below `External_Test_Results/`.

The first external run used reduced CamVid11 class ID `6` and produced mean IoU
`0.241`. Visual failure analysis showed that this target included traffic-light
housings and broader sign-like regions, so it was not an exact physical-plate
target. That result is preserved under `External_Test_Results/CamVid` only as a
diagnostic baseline and is not directly comparable to the corrected run.

The corrected evaluator uses the original CamVid32 `SignSymbol` colour RGB
`(192,128,128)`, where `TrafficLight` is a separate class. Each eligible
connected component is evaluated in a centred 1.5x-context crop at 256x256.
Regions below 10 pixels on either side or 30 labelled source pixels are declared
out of scope before evaluation.

Two operating modes must be distinguished:

- without a box, validation mean IoU is `0.157`; the detector commonly selects
  the coloured inner pictogram instead of the neutral physical plate;
- with `--camvid-box-prompt`, the ground-truth component box simulates an
  upstream sign detector and the module performs segmentation inside that
  location. This is segmentation-only, not end-to-end full-frame detection.

| Metric | Validation | Fixed test | All 701 frames |
|---|---:|---:|---:|
| Eligible instances | 35 | 181 | 284 |
| Non-empty predictions | 35/35 | 181/181 | 284/284 |
| Mean IoU | **0.836** | **0.782** | **0.791** |
| Median IoU | 0.853 | 0.827 | 0.841 |
| 10th-percentile IoU | 0.778 | 0.585 | 0.591 |
| Mean Dice | 0.908 | **0.872** | 0.876 |
| Mean precision / recall | 0.889 / 0.937 | **0.834 / 0.935** | 0.846 / 0.928 |
| Mean Boundary-IoU | 0.339 | **0.311** | 0.319 |
| Zero-overlap masks | 0 | **0** | 0 |

The fixed test excludes 355 of 536 connected components because they are below
the declared source-resolution threshold. Also, 173/181 predictions use the box
prior as their primary shape. Therefore the result is a strong improvement for
box-supported segmentation, but it is neither a small-object claim nor an
autonomous detection claim. Boundary-IoU remains far below a “perfect” result.

GrabCut is now seeded deterministically from the frame and component identity.
All 181 fixed-test rows have identical shape and metrics whether evaluated alone
or inside the all-frame run.

See `REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md` for the full label correction,
failure taxonomy, rejected ablations, per-shape/source breakdown, limitations,
and exact reproduction commands.

## Pixel-level background stress test

`--synthetic-test` generates 64 deterministic 256x256 images covering four
shapes and eight background families: foliage, brick, skyline, urban clutter,
same-colour distractors, shadows, low light, blur/noise, and JPEG artefacts. Each
perspective-warped sign has an exact independently generated binary mask. This
is a validation/stress set, not training data, and therefore avoids licensing or
multi-gigabyte download requirements.

| Metric | Original baseline | Current audited implementation |
|---|---:|---:|
| Mean mask IoU | 0.654 | **0.896** |
| Mean Dice | 0.703 | **0.934** |
| Mean precision | 0.765 | **0.912** |
| Mean recall | 0.670 | **0.968** |
| Mean Boundary-IoU | not comparable | **0.728** |
| Median mask IoU | not recorded | **0.966** |
| 10th-percentile IoU | not recorded | **0.724** |
| Strong masks (IoU >= .80 and Boundary-IoU >= .55) | not comparable | **47/64** |
| Weak masks (IoU < .50) | not recorded | **1/64** |
| Exact zero-overlap masks | not recorded | **0/64** |
| Shape accuracy | not recorded for original baseline | **59/64 (92.2%)** |

The older 0.598 Boundary-IoU used a non-reference 1.4%-diagonal elliptical
band, so it is not a like-for-like baseline against the corrected 0.728 score.
The one remaining weak synthetic case deliberately crosses a red STOP plate
with connected red clutter and several foreground lines; no closed octagonal
proposal survives. SAM 3, SAM 2.1 with this module's mask prompt, or a trained
traffic-sign instance segmenter is the appropriate next tier for that
ambiguity. Public future test sources include
[TT100K](https://cg.cs.tsinghua.edu.cn/traffic-sign/tutorial.html), whose
annotations can include polygon masks, and the diverse
[Mapillary Vistas](https://openaccess.thecvf.com/content_iccv_2017/html/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.html),
whose dense polygon annotations include instance-specific street-scene classes.

## Important parameters

All settings are in `ShapeParams::ShapeParams()`.

| Parameter | Default | Meaning |
|---|---:|---|
| `edgeProposals` | `true` | Fuse adaptive Canny contour proposals. |
| `houghCircles` | `false` | Optional Hough circle proposals. |
| `minEdgeSupport` | `0.34` | Minimum soft outline support for edge-only models. |
| `minColorCoverage` | `0.018` | Reject uncoloured digits/arrows as standalone signs. |
| `dedupeIoU` | `0.72` | Duplicate-model suppression threshold. |
| `splitMerged` | `true` | Add bridge-splitting opening proposals without replacing originals. |
| `splitOpenFrac` | `0.028` | Base opening scale; a second 1.75x scale is also evaluated. |
| `selectionSizeWeight` | `0.35` | Scale evidence for the documented cropped-sign input contract. |
| `centerPriorStrength` | `0.22` | Light centre prior for already-cropped inputs. |
| `refineWithGrabCut` | `true` | Enable pixel-boundary refinement. |
| `grabCutIterations` | `5` | Iterated graph-cut passes. |
| `grabCutOuterFrac` | `0.080` | Maximum boundary motion outside the shape prior. |
| `grabCutInnerFrac` | `0.160` | Erosion used for sure foreground. |
| `refineMinAreaRatio` | `0.72` | Smallest plausible refined/prior area ratio. |
| `refineMaxAreaRatio` | `1.16` | Normal maximum; verified circular rims may use `1.45`. |
| `refineMinSpanRatio` | `0.80` | Minimum retained width and height. |
| `refineMinColorRecall` | `0.90` | Minimum retained raw colour/rim evidence. |

Do not report the supplied-set 100% shape result as independent generalisation
performance. Use the exact-mask stress results for segmentation claims, keep its
remaining failure cases visible, and validate on a separately annotated public
set before deployment.
