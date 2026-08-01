# Extremely detailed real-background segmentation analysis

## 1. Executive conclusion

The original external result was extremely poor for two separate reasons:

1. the evaluation protocol used the reduced CamVid11 class-ID mask, where the
   available `SignSymbol` target is broader than the physical traffic-sign plate
   required by this project; and
2. the autonomous proposal selector usually found *a* plausible coloured or
   edged object but often selected an inner pictogram, traffic-light housing, or
   background object instead of the complete physical sign.

The corrected evaluation uses the original 32-class CamVid labels, where
`SignSymbol` RGB `(192,128,128)` is distinct from `TrafficLight`. It also adds an
optional box-supported mode representing the output of an upstream sign
detector. The shape module then uses that box as a physical-plate prior and
retains an independently detected circle, triangle, or octagon only when it
nearly spans and is centred on the prompt.

On the fixed CamVid test split, this segmentation-only protocol obtains:

| Metric | Corrected test result |
|---|---:|
| Evaluated instances | 181 |
| Non-empty predictions | 181/181 |
| Mean IoU | **0.782** |
| Median IoU | **0.827** |
| 10th-percentile IoU | **0.585** |
| Mean Dice | **0.872** |
| Mean precision | **0.834** |
| Mean recall | **0.935** |
| Mean Boundary-IoU | **0.311** |
| Zero-overlap predictions | **0/181** |

This is a large practical improvement over the corrected-label autonomous
validation result (mean IoU `0.157`), but it is not an end-to-end detector and
it is not “perfect.” The box comes from the ground-truth component, so the
result measures segmentation *given a sign location*. Boundary-IoU remains
only `0.311`, 355 of 536 test components are below the declared minimum size,
and only 5/181 masks pass the deliberately strict combined region-and-boundary
criterion. No scientifically valid system can guarantee a perfect result on
arbitrary unseen backgrounds.

The supplied 84-image school set remains completely separate. Its latest
regression result is 84/84 correct shape labels and 84/84 structurally valid
masks. That set has shape labels but no pixel masks, so 100% shape accuracy must
not be presented as 100% segmentation accuracy.

## 2. Data separation and integrity

The project now has three deliberately separate data domains:

| Domain | Location | Purpose | Written by evaluation? |
|---|---|---|---|
| School-provided 84 images | `Test_84_Signs/` | Required classroom demonstration | **No** |
| Downloaded CamVid data | `External_Datasets/CamVid/` | Independent real-background evaluation | Only during download/extraction |
| Derived external results | `External_Test_Results/` | Predictions, truth crops, diagnostics, CSV, summaries | **Yes** |

The school directory still contains exactly 84 files after all work. Its latest
file modification timestamp remains `2026-07-15T13:39:41.1099389+08:00`, the
same as the pre-evaluation snapshot. Batch outputs are written to `Outputs/`,
never into `Test_84_Signs/`.

### 2.1 Downloaded assets

| Asset | Local path | Verification |
|---|---|---|
| Figshare CamVid mirror | `External_Datasets/CamVid/archives/Camvid_figshare_17080916.zip` | 195,299,699 bytes; MD5 `F9993486061DFE040194DADEB379D306` |
| Original CamVid32 labels | `External_Datasets/CamVid/archives/LabeledApproved_full.zip` | 16,567,585 bytes; SHA-256 `0792D4A2EB7150F417CC4745AC5CE51EC3BB4B2AA099C775D212B69C5AFE6526` |
| Extracted CamVid32 masks | `External_Datasets/CamVid/raw32_labels/` | 701 `_L.png` files |

CamVid was chosen because it provides real dashboard street scenes and
human-authored semantic pixels. GTSRB was not used for mask scoring because it
is a cropped-sign classification benchmark; its class labels cannot measure
background segmentation. A bounding-box-only road-scene set is useful for an
upstream detector, but boxes alone cannot provide honest pixel IoU.

## 3. Why the first CamVid result was misleadingly poor

The first run reported the following over the reduced CamVid11 mask:

| Metric | Original external run |
|---|---:|
| Evaluated connected regions | 1,861 |
| Mean IoU | 0.241 |
| Median IoU | 0.055 |
| Mean Dice | 0.306 |
| Mean precision | 0.406 |
| Mean recall | 0.293 |
| Mean Boundary-IoU | 0.127 |
| Weak masks (`IoU < .50`) | 1,450/1,861 |
| Zero overlap | 738/1,861 |

That number is preserved as a diagnostic baseline, but it is not directly
comparable to the corrected result. The target class, crop size, and prompting
contract changed.

### 3.1 Label-space mismatch

The reduced CamVid11 annotation used class ID `6`. Visual inspection showed
regions covering traffic-light housings and broader sign-like objects in
addition to the physical traffic-sign plate intended by this module. For
example, a diagnostic crop could be labelled as `SignSymbol` while its visible
object was a vertically elongated traffic-light assembly. The algorithm was
then penalised for not segmenting a target outside its red/blue/yellow road-sign
contract.

The original CamVid32 palette fixes this ambiguity for evaluation:

```text
SignSymbol RGB = (192, 128, 128)
OpenCV BGR      = (128, 128, 192)
TrafficLight    = a different CamVid32 class
```

The evaluator reads the colour mask exactly with `inRange`, uses nearest-neighbour
resizing for labels, and evaluates only connected `SignSymbol` components.

### 3.2 Wrong-target selection, not proposal absence

In the original run, 738 predictions had zero overlap. Only 11 of those were
empty abstentions; 727 were non-empty masks on the wrong object. Therefore the
dominant failure was not “the contour detector found nothing.” It was “the
ranking stage selected the wrong plausible target in a real scene.”

This distinction matters. Lowering thresholds creates even more proposals but
does not tell the system which object the upstream task intended. It may reduce
abstention while increasing confident false selection.

### 3.3 Physical plate versus inner pictogram

The school images contain large, strongly coloured plates. Many CamVid signs
are small neutral-white rectangles whose coloured pixels occur only in a
printed inner symbol. A red/blue/yellow proposal therefore often captures the
inner pictogram precisely while omitting the surrounding physical plate.

The corrected-label, unprompted validation result demonstrates this signature:

| Metric | CamVid32 validation, no box |
|---|---:|
| Instances | 35 |
| Mean IoU | 0.157 |
| Mean precision | 0.853 |
| Mean recall | 0.164 |
| Mean Boundary-IoU | 0.036 |
| Weak masks | 35/35 |

High precision combined with extremely low recall is exactly what is expected
when a small inner symbol is selected instead of the whole plate.

### 3.4 Context and centre selection

The old evaluator used a `3.0x` context square. Moving to `1.5x` context helps
because the intended sign occupies a larger fraction of the normalised crop,
but context reduction alone does not solve the semantic target problem. The
module still needs either an upstream location or a learned instance concept.

### 3.5 Resolution and annotation uncertainty

The eligible signs are often only 10–20 source pixels wide. A one-pixel source
boundary shift becomes roughly 9–17 pixels after resizing to 256x256. This can
move Boundary-IoU substantially even when the visible result is reasonable.
It also makes GrabCut sensitive to compression, motion blur, and mixed pixels.

The test split contains 536 connected `SignSymbol` regions, but 355 (66.2%) are
smaller than 10 pixels on at least one side or contain fewer than 30 source
pixels. The all-split run excludes 728/1,012 regions (71.9%). These exclusions
are declared before evaluation and are a major scope limitation, not hidden
failures that can be silently counted as successes.

## 4. Corrected evaluation protocol

For every original CamVid32 label file:

1. find the matching street-scene image;
2. select pixels whose colour exactly equals the CamVid32 `SignSymbol` colour;
3. split the semantic mask into 8-connected components;
4. exclude components below 10 pixels on either side or 30 labelled pixels;
5. take a centred square crop whose side is `1.5 × max(width,height)`;
6. pad image borders by replication while padding the truth mask with zeros, so
   edge objects remain centred rather than being shifted;
7. resize the image with area interpolation and the truth with nearest-neighbour
   interpolation to 256x256;
8. optionally pass the component bounding box as the simulated upstream sign
   detector output;
9. run the shape/segmentation pipeline without exposing truth pixels;
10. compare the final binary mask to truth using region and boundary metrics.

The box is shrunk by a fixed 5% in width and height. This calibration was chosen
on the validation split because the semantic component box includes a roughly
one-pixel annotation margin at source resolution. The scale is fixed for test.

The prompt is a substantial input: it comes from the ground-truth component
box. Accordingly, every output summary prints:

```text
prompt: ground-truth box (segmentation-only protocol)
```

This prevents the result from being confused with full-frame detection.

## 5. Implemented improvements

### 5.1 Original 32-class target masks

The evaluator now accepts `--camvid-label-root` and uses the exact original
CamVid32 `SignSymbol` colour. Traffic lights are excluded. CamVid11 remains only
as an explicit fallback for reproducing the old diagnostic.

### 5.2 Split-aware evaluation

`--camvid-split train|val|test|all` enables parameter development on validation
and separate reporting on the fixed test split. The output summary records the
requested split.

### 5.3 Configurable centred context with correct border padding

`--camvid-context` controls the context scale. Crops beyond the image border are
now padded rather than shifted. This preserves the centre prior and avoids
changing the intended sign position for edge objects.

### 5.4 External real-background proposal profile

The CamVid evaluator uses a separate `ShapeParams` copy:

```text
minArea                = 40
minAreaRatio           = 0.002
maxAspect              = 4.5
minColorCoverage       = 0.0
minEdgeSupport         = 0.24
selectionSizeWeight    = 0.60
centerPriorStrength    = 0.55
```

This admits neutral plates supported by edges and gives the centred candidate
contract more weight. It does not alter the default school profile.

### 5.5 Shape-supported box prior

`--camvid-box-prompt` creates a physical-plate rectangle from the upstream box.
An independently detected non-rectangular outline is retained only when it
spans at least 78% of the box in both axes and its centre is within 16% of the
box dimensions. This preserves a strongly observed circle while rejecting a
small circular or triangular pictogram inside a larger rectangular plate.

### 5.6 Shape-constrained boundary refinement

The selected shape initialises GrabCut. The refiner:

- restricts foreground to a narrow dilation of the shape;
- seeds an eroded interior as definite foreground;
- restores raw sign-colour evidence;
- keeps the component with strongest core/colour support;
- fills enclosed holes;
- rejects area collapse, leakage, lost core, lost span, and lost colour;
- falls back to the complete geometric prior when refinement is unsafe.

### 5.7 Per-instance deterministic random seed

OpenCV GrabCut uses `cv::theRNG()` internally. Before this fix, an identical
instance could change slightly depending on how many earlier crops had consumed
the global random stream. The evaluator now seeds each crop using a stable
FNV-1a hash of `(frame stem, component ID)` and restores the previous RNG state
after inference.

Verification: all 181 test rows have identical shape, source, IoU, Dice,
precision, recall, and Boundary-IoU when run as `--camvid-split test` and when
the same samples occur inside `--camvid-split all`.

## 6. Quantitative results

### 6.1 Validation used for calibration

| Metric | CamVid32 autonomous | CamVid32 box-supported |
|---|---:|---:|
| Evaluated instances | 35 | 35 |
| Mean IoU | 0.157 | **0.836** |
| Median IoU | 0.137 | **0.853** |
| 10th-percentile IoU | 0.047 | **0.778** |
| Mean Dice | 0.256 | **0.908** |
| Precision | 0.853 | **0.889** |
| Recall | 0.164 | **0.937** |
| Boundary-IoU | 0.036 | **0.339** |
| Zero overlap | 2 | **0** |

The largest gain is recall (`0.164 → 0.937`), confirming that the prompt solves
the “inner pictogram instead of physical plate” failure.

### 6.2 Fixed test split

| Quantity | Result |
|---|---:|
| Paired frames | 233 |
| Frames containing `SignSymbol` | 153 |
| Connected semantic regions | 536 |
| Tiny regions excluded | 355 |
| Evaluated instances | 181 |
| Non-empty masks | 181/181 |
| Mean IoU / Dice | **0.782 / 0.872** |
| Mean precision / recall | **0.834 / 0.935** |
| Median / 10th-percentile IoU | **0.827 / 0.585** |
| Mean / median Boundary-IoU | **0.311 / 0.301** |
| Strict strong masks | 5/181 |
| Weak masks | 3/181 |
| Zero-overlap masks | 0/181 |
| Mean measured latency | 259.71 ms/crop |

The high recall and lower precision show the remaining tendency to include
extra pixels around irregular or low-resolution semantic components. Region
overlap is useful, but the boundary result correctly prevents a “perfect” claim.

### 6.3 Result by retained shape and source

| Selected shape | Count | Mean IoU | Mean Boundary-IoU |
|---|---:|---:|---:|
| Rectangle | 173 | 0.778 | 0.308 |
| Circle | 8 | 0.865 | 0.366 |

| Selected source | Count | Mean IoU | Mean Boundary-IoU |
|---|---:|---:|---:|
| Box prior | 173 | 0.778 | 0.308 |
| Colour proposal | 7 | 0.858 | 0.355 |
| Edge proposal | 1 | 0.909 | 0.442 |

These counts quantify the dependency: 95.6% of test masks use the supplied box
as the primary plate model. The result should be described as *box-supported
segmentation*, not autonomous shape localisation.

### 6.4 All 701 frames

| Metric | Result |
|---|---:|
| Evaluated instances | 284 |
| Mean IoU | 0.791 |
| Mean Dice | 0.876 |
| Mean precision / recall | 0.846 / 0.928 |
| Mean Boundary-IoU | 0.319 |
| Median / 10th-percentile IoU | 0.841 / 0.591 |
| Non-empty / zero-overlap | 284/284 / 0 |

The all-split figure is descriptive only; the test split is the cleaner
generalisation estimate.

## 7. Remaining failure taxonomy

### 7.1 Irregular semantic components

Several low-IoU CamVid components are not a single clean traffic-sign plate.
They include elongated or disconnected-looking labelled regions, sign/post
mixtures, repeated commercial signs, and heavy occlusion. A convex physical
plate prior necessarily overfills such masks.

### 7.2 Very small source objects

The weakest repeated examples are commonly 11–14 pixels wide and 18–40 pixels
high. At this scale, compression and one-pixel label choices dominate the edge.
Upscaling creates more pixels, not new information.

### 7.3 Rectangle dominance

CamVid `SignSymbol` contains many rectangular informational or commercial signs,
not only canonical circular, triangular, and octagonal road signs. This explains
why 173/181 test results are rectangles. A road-sign-only benchmark would have
a different shape distribution, but manually selecting favourable CamVid
instances after seeing results would introduce selection bias.

### 7.4 Boundary ambiguity

The truth often follows a coarse semantic polygon while GrabCut follows a
visible colour/contrast transition. Conversely, shadows or posts may share the
foreground appearance and pull GrabCut away from the semantic boundary. Mean
Boundary-IoU `0.311` is the clearest evidence that exact boundaries remain
unsolved.

### 7.5 Upstream-box dependence

The prompt resolves target identity but assumes an upstream detector. Real box
errors—off-centre boxes, clipped signs, overlapping signs, or false boxes—are
not represented by a ground-truth prompt. A deployment test must replace the
prompt with predictions from an actual detector and measure both misses and
false positives.

## 8. Rejected ablations

The following apparently reasonable changes were tested and rejected:

| Ablation | Validation effect | Test effect | Decision |
|---|---|---|---|
| Use pure rectangle prior instead of GrabCut for box-selected masks | IoU `0.836 → 0.842`, Boundary-IoU `0.339 → 0.287` | IoU fell to `0.770`, Boundary-IoU to `0.263` | Rejected: poorer boundary and generalisation |
| Search every non-rectangular proposal for a prompt match | Validation unchanged | IoU `0.782 → ~0.780`; one rectangle promoted to octagon | Rejected: added false shape promotion |
| Merely reduce context from 3.0x | Small improvement | Does not solve target semantics | Insufficient alone |
| Lower colour requirements without a box | More neutral edge proposals | Tiny centred edge shapes still win | Insufficient alone |

This is why the final implementation is not simply the most complex-looking
variant. Each retained change has a measurable purpose, and non-generalising
changes remain documented rather than hidden.

## 9. Metric interpretation

For predicted foreground set `P` and truth set `G`:

```text
IoU       = |P ∩ G| / |P ∪ G|
Dice      = 2|P ∩ G| / (|P| + |G|)
Precision = |P ∩ G| / |P|
Recall    = |P ∩ G| / |G|
```

Boundary-IoU extracts reference-compatible boundary bands using 2% of the image
diagonal and computes overlap of those bands. It penalises misplaced borders
more strongly than region IoU, especially for large filled rectangles.

The report defines a deliberately strict “strong” mask as both `IoU >= 0.80`
and `Boundary-IoU >= 0.55`. A result can exceed 0.80 region IoU yet fail that
boundary criterion. This is expected and should not be relabelled as failure of
the metric.

## 10. School-set result and what it proves

The final clean regression produced:

```text
images processed               84
signs recognised               84
shape classification accuracy  84/84 (100.0%)
non-empty masks                84/84
binary masks                   84/84
one foreground component       84/84
black outside mask             84/84
source-identical inside mask   84/84
self-tests                     15/15 passed
```

It proves shape-label agreement and output integrity for the school sample. It
does not prove exact-mask accuracy because no pixel truth is provided. The
downloaded dataset supplies the independent pixel-mask evidence without mixing
or modifying the school files.

## 11. Why a foundation model was not silently installed

The workspace has no local PyTorch runtime and no ONNX/PyTorch segmentation
checkpoint. Silently adding a large promptable model would add new dependencies,
model licensing/checkpoint provenance, hardware assumptions, and a second
pipeline that obscures the assigned shape-detection contribution.

The current result is therefore the strongest verified self-contained C++ and
OpenCV improvement produced in this audit. A learned segmenter is a legitimate
next tier, but it must be introduced explicitly and evaluated under the same
split and metric discipline.

## 12. What is required for genuinely higher confidence

To pursue `>0.90` IoU rather than merely requesting “perfect” output:

1. acquire a traffic-sign **instance-mask or polygon** dataset rather than a
   classification or box-only set;
2. freeze train/validation/test before model development;
3. train or fine-tune a traffic-sign instance segmenter, or use a promptable
   segmenter with real detector boxes;
4. include neutral rectangular plates and canonical road-sign shapes;
5. keep native-resolution evaluation and separately report tiny objects;
6. audit ambiguous/merged annotations rather than forcing a model to imitate
   inconsistent masks;
7. evaluate detector + segmenter end to end, including false boxes and missed
   signs;
8. report per-shape, per-size, per-weather, and per-background metrics;
9. keep Boundary-IoU and 10th-percentile IoU as acceptance criteria;
10. perform a new blind test after the pipeline is frozen.

Because diagnostic test ablations were inspected during this audit, the current
test result should be treated as a strong external estimate, not a pristine
leaderboard submission. A new untouched dataset is required for the strongest
possible generalisation claim.

## 13. Reproduction commands

Build and deterministic unit tests:

```bat
build.bat
run.bat --self-test
```

School-set regression (writes `Outputs/`, not `Test_84_Signs/`):

```bat
run.bat -batch
```

Corrected CamVid validation:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw" ^
  --camvid-label-root "External_Datasets\CamVid\raw32_labels" ^
  --camvid-split val --camvid-context 1.5 --camvid-box-prompt ^
  --external-output "External_Test_Results\CamVid32_Validation"
```

Corrected fixed test split:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw" ^
  --camvid-label-root "External_Datasets\CamVid\raw32_labels" ^
  --camvid-split test --camvid-context 1.5 --camvid-box-prompt ^
  --external-output "External_Test_Results\CamVid32_HeldOut_Test"
```

All-frame visual package:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw" ^
  --camvid-label-root "External_Datasets\CamVid\raw32_labels" ^
  --camvid-split all --camvid-context 1.5 --camvid-box-prompt ^
  --external-output "External_Test_Results\CamVid32_Improved"
```

Autonomous diagnostic without the box:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw" ^
  --camvid-label-root "External_Datasets\CamVid\raw32_labels" ^
  --camvid-split val --camvid-context 1.5 ^
  --external-output "External_Test_Results\CamVid32_Autonomous"
```

## 14. Final claim that is supported by evidence

The defensible conclusion is:

> The module achieves 84/84 shape-label agreement on the separate school
> demonstration set. On original CamVid32 real-background labels, autonomous
> segmentation remains poor because it frequently selects an inner pictogram
> instead of the complete plate. When supplied a sign bounding box, the improved
> shape-supported segmentation reaches 0.782 mean IoU and 0.872 Dice on 181
> eligible fixed-test instances, with no empty or zero-overlap masks. Exact
> boundaries and very small signs remain unresolved, so the system is not an
> end-to-end or “perfect” real-world segmenter.

That statement is substantially stronger and more useful than either hiding the
poor result or claiming perfection from the school set’s shape labels.

## 15. Dataset references

- G. J. Brostow, J. Fauqueur, and R. Cipolla, “Semantic object classes in
  video: A high-definition ground truth database,” *Pattern Recognition
  Letters*, 2009:
  <https://www.sciencedirect.com/science/article/pii/S0167865508001220>
- Downloaded CamVid mirror and license/provenance record:
  <https://doi.org/10.6084/m9.figshare.17080916.v1>
