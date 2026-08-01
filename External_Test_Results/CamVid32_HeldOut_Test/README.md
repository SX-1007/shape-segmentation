# Corrected CamVid32 fixed-test result

This folder contains the corrected CamVid `test` split evaluation using original
CamVid32 `SignSymbol` masks and a ground-truth-derived box prompt.

```text
paired frames            233
connected sign regions   536
tiny regions excluded    355
instances evaluated      181
mean IoU                 0.782
mean Dice                0.872
mean precision           0.834
mean recall              0.935
mean Boundary-IoU        0.311
median IoU               0.827
10th-percentile IoU      0.585
non-empty predictions    181/181
zero-overlap masks       0/181
```

The prompt supplies target location, so this measures segmentation given a sign
box, not end-to-end detection. See `../../REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md`
for the full protocol, failure taxonomy, ablations, and limitations.
