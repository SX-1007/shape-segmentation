# Detailed segmentation audit and improvement report

## Executive conclusion

The reported 98.8% result in the earlier version did **not** measure segmentation
quality. It measured only whether the selected proposal had the expected shape
name. A mask could lose its red rim, contain half a triangle, or select an inner
pictogram and still count as a correct shape. The binary/connected-component
audit was also insufficient: a wrong half-mask can still be binary, non-empty,
and connected.

The border and partial-mask errors matter even when the current shape label does
not change. Shape classification happens before final refinement, so the current
demo can still print `CIRCLE` after deleting a circular red border. Downstream
recognition is different:

- a red border distinguishes regulatory/prohibitory signs from visually similar
  non-regulatory discs;
- removing the rim changes colour histograms, contour scale, crop size, and
  features used by an OCR/classifier;
- a half-mask removes pictogram strokes and can make recognition fail outright;
- masks used as later training labels would teach a model the wrong boundary.

Therefore, border deletion should not be accepted merely because the shape label
is unchanged.

## Reproduced failures and their causes

| Image | Observed failure | Direct cause in the old code | Fix |
|---|---|---|---|
| `000_1_0002.png` | The red speed-limit rim disappeared; only the pale centre remained. | The outer rim was only probable foreground in the GrabCut trimap. GrabCut preferred the internally homogeneous pale disc. The final plausibility check did not measure rim recall. | Reintroduce raw, unfilled sign-colour pixels before component selection and hole filling; require at least 90% raw-colour recall. |
| `033_1_0001.png` | Only one side/part of the triangular sign survived. | `refineMinAreaRatio=0.48` allowed a result containing roughly half of the prior. There was no width/height completeness test. | Raise the minimum area ratio to 0.72 and require at least 80% of both prior dimensions. Fall back to the complete geometric prior when either fails. |
| `051_0005_j.png` | Only the tiny triangular warning pictogram was returned; most of the yellow physical plate was missing. | The generic carrier rule starts at a 10% inner/outer area ratio. This graphic is only 3.9% of the credible rectangle, while the outer proposal also received an image-border penalty. | Add a tightly gated tiny-graphic rule: matching dominant colour plus strong rectangular fit, edge support, colour coverage, and colour capture protect the complete outer plate. |
| `027_0012.png` | Background/rectangular leakage surrounded the actual circular roundabout sign. | An earlier nested-candidate rule treated the excellent blue circle as an interior pictogram because a larger, weak rectangle captured more blue pixels. That rectangle touched the crop edge, had low contrast, and had only `rectFit=0.789`. | Require credible rectangle geometry and boundary support before suppressing a nested circle. The physical circle now wins and removes the rectangular background leakage. |
| `001_0004.png` | The speed-limit result contained the pale centre but omitted the thin red outside rim. | GrabCut recovered the rim, but its area was 24% larger than the incorrectly selected inner-disc prior and therefore exceeded the normal 1.16 leak limit. | Detect the low-colour-interior/surrounding-colour-ring signature, protect raw rim pixels in the narrow outer band, and allow up to 1.45 area growth only for that evidence-backed case. |

## Code audit by pipeline stage

### 1. Colour thresholding

The normalized red/blue/yellow enhancement is a good choice for illumination
variation, and keeping colours separate correctly prevents unlike-colour signs
and boards from being welded together.

The defect was data reuse. `cleanMask()` fills enclosed holes, which is useful
for proposal geometry but invalid for evidence measurement. After filling, a
white number, black arrow, or blue area enclosed by another colour inherits the
surrounding mask label.

The revised implementation keeps two representations:

1. cleaned, hole-filled masks for complete contour proposals;
2. raw, unfilled masks for colour coverage, colour capture, dominant colour,
   GrabCut repair, and colour-recall validation.

This separation is essential. Proposal generation and evidence scoring have
different requirements and should not share a destructively transformed mask.

### 2. Edge proposals

CLAHE, adaptive Canny thresholds, scale-aware closing, and `RETR_LIST` are
reasonable for the small supplied crops. Edge proposals recover outer sign
boundaries when colour follows only the interior.

Hough circles remain optional and disabled. Tests with unrestricted Hough
voting reduced supplied-set accuracy from 100% to 95.2% because circular
pictograms and traffic-light elements became convincing sign proposals. A new
large/central plausibility gate is present if Hough is deliberately enabled, but
the default remains evidence-based contour proposals.

### 3. Geometric classification

The convex-hull descriptors, multi-tolerance corner counts, aspect
normalization, and ellipse fitting are strong parts of the implementation. They
give explainable invariance without a learned model.

The main semantic weakness was that each contour was originally ranked in
isolation. Real signs create nested contour families: outer plate, coloured
interior, border, pictogram, and carrier. The revised selector uses those
relationships:

- same-shape inner/outer border pairs are softly re-ranked, never hard-deleted;
- the carrier rule has both a minimum and maximum inner/outer area ratio;
- a rounded rectangle can suppress a nested circle only when its rectangle fit
  and boundary support are credible;
- a very small triangular pictogram cannot suppress a strongly supported,
  same-colour rectangular physical plate;
- a border-leaking component cannot determine the scale normalization;
- a 22% centre multiplier helps this crop-based module reject edge clutter.

The centre prior is intentionally light. If this module is later applied to
uncropped full-road frames, set `centerPriorStrength=0` and place a detector or
sliding-window stage before segmentation.

### 4. Shape-constrained refinement

The old acceptance rule checked area and core recall only. Neither detects all
one-sided masks, and the 48% lower area bound explicitly permitted a half-mask.

The revised sequence is:

1. build the geometric prior;
2. for a colour-dominated triangle, evaluate bounded outer-border scale
   hypotheses; for a low-colour circular interior, test for an outer colour rim;
3. initialize the four-state GrabCut trimap;
4. run GrabCut;
5. restore raw sign-colour/rim pixels inside the prior, or in the narrow outer
   band when a surrounding circular rim has been verified;
6. close short gaps;
7. retain the component maximizing core and raw-colour overlap;
8. fill enclosed plate holes;
9. repair only strongly concave cuts;
10. validate area ratio, core recall, colour recall, and two-axis span;
11. use the complete prior if any guard fails.

`RefinementDiagnostics` records all decisions. Batch mode exports the values to
`Outputs/refinement_diagnostics.csv`, so a future regression no longer needs to
be found solely by looking through montage images.

### 5. Evaluation

The original 84 images contain shape labels but no segmentation masks. Their
shape score and mask integrity checks remain useful, but they cannot produce
pixel IoU, Dice, precision, recall, or Boundary-IoU.

`SegmentationTest.cpp` adds a deterministic 64-image stress set with exact
masks. It covers four shapes, perspective warp, foliage, brick, skyline, urban
clutter, sign-coloured distractors, low light, shadow, blur, noise, and JPEG
compression. Ground truth is generated independently from the prediction path.

Boundary-IoU is reported because it exposes the exact failure in question: a
thin missing rim can have a modest effect on region IoU but a severe effect on
the boundary band. The implementation now follows the authors' reference
definition: a 2%-of-image-diagonal band, a square erosion kernel, and padding
before erosion so that objects truncated by an image boundary are scored
correctly.

## Quantitative result

| Test | Old implementation | Revised implementation |
|---|---:|---:|
| Supplied shape labels | 83/84 (98.8%) | **84/84 (100.0%)** |
| Synthetic mean IoU | 0.654 | **0.896** |
| Synthetic mean Dice | 0.703 | **0.934** |
| Synthetic mean precision | 0.765 | **0.912** |
| Synthetic mean recall | 0.670 | **0.968** |
| Synthetic mean Boundary-IoU | not comparable | **0.728** |
| Synthetic median / 10th-percentile IoU | not recorded | **0.966 / 0.724** |
| Synthetic strong masks | not comparable | **47/64** |
| Synthetic weak masks (IoU < 0.50) | not recorded | **1/64** |
| Synthetic exact zero-overlap masks | not recorded | **0/64** |
| Synthetic shape labels | 54/64 (84.4%) | **59/64 (92.2%)** |

The earlier 0.598 Boundary-IoU result used a non-reference 1.4%-diagonal
elliptical band, so it is deliberately not presented as directly comparable.
The proposal-only opening stream now separates signs from narrow same-colour
bridges while retaining the original stream, which removed the ten former
zero-overlap cases and corrected five shape labels. One adversarial case remains
below 0.50 IoU: dense same-colour clutter physically crosses the octagon, so no
closed octagonal boundary survives for the classical proposal generator.

## Current research and upgrade path

The implemented changes borrow the practical principles of boundary-aware
segmentation without adding a non-reproducible model checkpoint:

- [Boundary IoU](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html)
  motivates boundary-sensitive evaluation.
- [SegFix](https://www.microsoft.com/en-us/research/publication/segfix-model-agnostic-boundary-refinement-for-segmentation/)
  shows why reliable interior predictions can repair uncertain boundary pixels.
- [SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
  is the current concept-aware foundation-model option: text or exemplar prompts
  can detect and segment matching signs. The
  [SAM 3.1 update](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)
  primarily improves multi-object video efficiency. For an image-crop integration,
  [SAM 2 / SAM 2.1](https://github.com/facebookresearch/sam2) also supports box
  and mask prompts; pass this module's geometric prior as the prompt, validate
  the returned mask, and retain the deterministic prior as a fallback.

For external generalization, use independently licensed datasets rather than
folding them into parameter tuning. TT100K annotations can contain polygon masks,
while Mapillary provides far broader geography, weather, and lighting. Large
public datasets were not copied into this repository because TT100K requires
roughly 100 GB according to its official tutorial and Mapillary access/licensing
should remain explicit.

## Reproduction

```bat
build.bat
run.bat -batch
run.bat --synthetic-test
run.bat --self-test
```

The authoritative generated reports are:

- `Outputs/refinement_diagnostics.csv`
- `Synthetic_Background_Test/evaluation.csv`
- `Synthetic_Background_Test/evaluation_summary.txt`
- `External_Test_Results/CamVid32_HeldOut_Test/evaluation_summary.txt`
- `REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md`

## Remaining limitations

- This is segmentation of one dominant sign in an already cropped image, not a
  multi-instance full-street detector.
- Same-colour contact between a sign and a clutter object can erase the true
  boundary for both colour and Canny. Proposal-only opening repairs narrow
  bridges, but wide crossings still require a learned semantic prior or a
  temporal/multi-view cue.
- The 1.28 triangle-frame fallback encodes a broad traffic-sign design prior.
  It is bounded and evidence-gated, but unusual borderless triangles should be
  tested separately.
- Supplied-set parameters are not independent validation. The synthetic set is
  deterministic and exact. A corrected public CamVid32 box-supported evaluation
  is now included, but it uses ground-truth-derived boxes and excludes very tiny
  instances. A real detector-box test and a new untouched instance-mask dataset
  remain necessary before deployment.
