# CamVid external dataset

This folder is intentionally separate from `Test_84_Signs`, the 84-image
school-provided dataset.  Nothing in the external evaluation writes to or
reorganises that school dataset.

## Provenance

- Dataset: Cambridge-driving Labeled Video Database (CamVid)
- Content: 701 real dashboard street-scene images. The Figshare archive
  includes reduced indexed CamVid11 masks. Those class-ID masks are retained
  only to reproduce the first diagnostic: visual auditing showed that their
  available class `6` target is broader than the physical traffic-sign plate
  required by this project.
- Figshare record: <https://doi.org/10.6084/m9.figshare.17080916.v1>
- Download API record: <https://api.figshare.com/v2/articles/17080916>
- Archive file ID: `31579223`
- Published archive size: `195299699` bytes
- Published MD5: `f9993486061dfe040194dadeb379d306`
- License shown by the Figshare record: CC BY 4.0
- Original CamVid publication: G. J. Brostow, J. Fauqueur, and R. Cipolla,
  "Semantic object classes in video: A high-definition ground truth
  database," Pattern Recognition Letters 30(2), 2009.

The downloaded ZIPs under `archives/` are retained unchanged. `raw/` contains
the extracted Figshare mirror. `raw32_labels/` contains the separately extracted
original 32-class `_L.png` masks. All test-derived crops and results belong
under `../../External_Test_Results/`, never here.

## Verified local copy

- Archive: `archives/Camvid_figshare_17080916.zip`
- Observed size: `195299699` bytes (matches published size)
- Observed MD5: `F9993486061DFE040194DADEB379D306` (matches published MD5)
- Extracted source images: 701
- Extracted indexed semantic masks: 701
- Split: 367 train, 101 validation, 233 test images
- Original label archive: `archives/LabeledApproved_full.zip`
- Original label archive size: `16567585` bytes
- Original label archive SHA-256:
  `0792D4A2EB7150F417CC4745AC5CE51EC3BB4B2AA099C775D212B69C5AFE6526`
- Extracted original CamVid32 masks: 701
- Correct evaluation target: CamVid32 `SignSymbol` RGB `(192,128,128)`;
  `TrafficLight` is excluded.

## Evaluation warning

The corrected result uses the original CamVid32 masks and an optional
ground-truth-derived bounding-box prompt. It is a segmentation-only protocol,
not full-frame sign detection. The old CamVid11 result is preserved for failure
analysis but must not be compared directly because its target class and crop
contract differ. See `../../REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md` for the
complete protocol and limitations.

## Why CamVid instead of GTSRB

GTSRB is designed for classification and contains tightly cropped sign images.
CamVid provides natural road backgrounds and exact, human-labelled pixels, so
it supports genuine segmentation IoU, Dice, precision, recall, and Boundary-IoU
measurements.  GTSDB has full road scenes but only bounding-box annotations;
boxes cannot support honest pixel-mask scores.
