# CamVid segmentation outputs

> **Historical diagnostic only.** This folder preserves the first reduced
> CamVid11 run. Visual auditing showed that class ID `6` is broader than the
> original CamVid32 physical `SignSymbol` target and includes traffic-light or
> other sign-like regions unsuitable for this project contract. Do not compare
> its `0.241` IoU directly with the corrected result. Use
> `../CamVid32_HeldOut_Test/` and read
> `../../REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md` for the corrected protocol.

This directory contains derived evaluation artifacts only.  It is separate
from both the read-only external source data in `External_Datasets/CamVid` and
the school-provided `Test_84_Signs` directory.

Run the benchmark from the project root with:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw"
```

The evaluator selects class ID `6` from the archive's reduced indexed CamVid11
masks, splits that broad semantic mask into connected regions,
and excludes source regions smaller than 10 pixels on either side or 30 labelled
pixels.  Each eligible region is centred in a square crop containing 3x natural
background context, then image and truth are resized to 256x256 (nearest-neighbour
for truth).  The segmentation pipeline receives only the image; it never sees
the truth mask.

Generated files:

- `evaluation.csv`: per-region metrics and detector diagnostics;
- `evaluation_summary.txt`: aggregate exact-mask results;
- `predictions/`: pipeline binary masks;
- `truth/`: corresponding hand-labelled CamVid masks after crop/resize;
- `diagnostics/`: original, truth, prediction, and colour-coded error panels.

The connected components are evaluation units derived from CamVid's semantic
labels; they are not claimed to be original instance IDs.  The resize protocol
matches this project's sign-candidate segmentation contract and is not a test
of full-frame traffic-sign detection.

## Completed result

The 2026-08-01 run processed every one of the 701 source/mask pairs.  Of 5,169
connected `SignSymbol` regions, 3,308 were excluded by the declared tiny-region
rule and 1,861 were evaluated.

| Metric | Result |
|---|---:|
| Non-empty prediction | 1,850 / 1,861 |
| Mean IoU | 0.241 |
| Median IoU | 0.055 |
| Mean Dice | 0.306 |
| Mean precision | 0.406 |
| Mean recall | 0.293 |
| Mean Boundary-IoU | 0.127 |
| Any overlap | 1,123 / 1,861 (60.3%) |
| IoU >= 0.50 | 411 / 1,861 |
| Strong mask (IoU >= .80 and Boundary-IoU >= .55) | 81 / 1,861 |
| Zero overlap | 738 / 1,861 |

This is weak out-of-dataset generalisation, not a passing segmentation result.
The best example, `0006R0_f03300_sign01`, reaches IoU 0.961 and Boundary-IoU
0.854, demonstrating that the pipeline can isolate some real sign boards very
accurately.  Many failures select another red/blue/yellow object in the crop,
and the CamVid `SignSymbol` semantic class also contains broader commercial and
rectangular signage beyond the traffic-sign plates for which this module was
tuned.  The aggregate deliberately keeps those cases rather than presenting a
hand-selected traffic-sign subset.

Because the hand label defines each centred context crop, this is a supervised
candidate-segmentation benchmark.  It does not measure full-frame detection.
The label is used only for crop selection and scoring; it is never passed to the
segmentation pipeline.
