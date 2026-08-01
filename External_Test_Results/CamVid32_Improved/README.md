# Corrected CamVid32 real-background results

This folder contains derived evaluation artifacts only. It is separate from the
downloaded data in `External_Datasets/CamVid/` and from the school-provided
`Test_84_Signs/` directory.

## Protocol

- 701 paired CamVid street frames and original CamVid32 label masks
- target: exact `SignSymbol` RGB `(192,128,128)`; `TrafficLight` excluded
- connected components at least 10 source pixels on both sides and 30 pixels in
  area
- centred 1.5x-context crop, resized to 256x256
- ground-truth-derived component box used as an upstream detector prompt
- shape-supported GrabCut segmentation; truth pixels used only for scoring
- deterministic per-instance OpenCV RNG seed

This is a segmentation-only protocol. It is not full-frame sign detection.

## All-frame result

| Metric | Value |
|---|---:|
| Eligible instances | 284 |
| Mean IoU | 0.791 |
| Mean Dice | 0.876 |
| Mean precision / recall | 0.846 / 0.928 |
| Mean Boundary-IoU | 0.319 |
| Median / 10th-percentile IoU | 0.841 / 0.591 |
| Non-empty predictions | 284/284 |
| Zero-overlap predictions | 0 |

Use the separate fixed-test result in `../CamVid32_HeldOut_Test/` for the more
appropriate generalisation estimate: 0.782 IoU and 0.872 Dice over 181 eligible
instances.

## Files

- `evaluation.csv`: one row per evaluated component
- `evaluation_summary.txt`: aggregate metrics and exclusions
- `predictions/`: final 0/255 masks
- `truth/`: matched CamVid32 truth crops
- `diagnostics/`: original, truth, prediction, and error panels

The complete interpretation, failure analysis, and limitations are in
`../../REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md`.
