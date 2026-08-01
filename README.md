# Traffic-Sign Shape Detection and Segmentation

This project is a C++17/OpenCV application that detects red, blue, and yellow
traffic signs, classifies their geometric shape, and produces a refined binary
segmentation mask. It includes an interactive demonstration, an 84-image batch
test, deterministic regression tests, synthetic exact-mask evaluation, and an
optional CamVid real-background benchmark.

## Quick test

From a Windows Command Prompt in the project directory:

```bat
build.bat
run.bat --self-test
run.bat -batch
```

A healthy run should end with:

```text
Self-test result: 15 passed, 0 failed.
```

The batch test should process all 84 supplied images and write its generated
images and diagnostics under `Outputs\`.

## How to view the test results

Each test mode reports and saves different information:

| Test command | Overall result | Per-image CSV | Images to inspect |
|---|---|---|---|
| `run.bat --self-test` | Printed in the terminal | Not generated | Not generated |
| `run.bat -batch` | Printed in the terminal | `Outputs\refinement_diagnostics.csv` | `Outputs\*.png` |
| `run.bat --synthetic-test` | `Synthetic_Background_Test\evaluation_summary.txt` | `Synthetic_Background_Test\evaluation.csv` | `Synthetic_Background_Test\predictions\` |
| `run.bat --evaluate-segmentation <root>` | `<root>\evaluation_summary.txt` | `<root>\evaluation.csv` | `<root>\predictions\` |
| `run.bat --evaluate-camvid ...` | `<external-output>\evaluation_summary.txt` | `<external-output>\evaluation.csv` | `<external-output>\diagnostics\`, `predictions\`, and `truth\` |

### View the 84-image batch results

Run:

```bat
run.bat -batch
```

The final terminal section reports the number of processed and recognized
images, mask-integrity checks, shape-classification accuracy, and the confusion
matrix. A clean run currently reports `84/84` correct shape classifications.

Open the generated folder in Windows File Explorer:

```bat
explorer Outputs
```

The generated PNG files let you visually compare the source image, processing
stages, binary mask, and isolated sign. Open the diagnostics CSV with its
default application, normally Excel:

```bat
start "" "Outputs\refinement_diagnostics.csv"
```

`refinement_diagnostics.csv` contains one row per image, including the detected
shape, proposal source, mask areas, area ratio, core/color recall, selected
prior scale, whether GrabCut was accepted, and the final refinement decision.

The 84 supplied images have shape labels but do not have hand-drawn pixel masks.
Consequently, this test can calculate shape accuracy and mask-integrity checks,
but it cannot calculate genuine IoU, Dice, precision, recall, or Boundary-IoU.

The batch summary is normally printed only in the terminal. To display it and
save a copy at the same time, use PowerShell:

```powershell
.\run.bat -batch 2>&1 | Tee-Object -FilePath results_console.txt
```

To save it from Command Prompt without displaying it during the run:

```bat
run.bat -batch > results_console.txt 2>&1
```

### View pixel-accuracy scores

For IoU, Dice, precision, recall, and Boundary-IoU, use a dataset that includes
matching ground-truth masks. The quickest built-in option is:

```bat
run.bat --synthetic-test
```

Open the overall human-readable score report:

```bat
notepad "Synthetic_Background_Test\evaluation_summary.txt"
```

Open the per-image scores in Excel or another CSV viewer:

```bat
start "" "Synthetic_Background_Test\evaluation.csv"
```

Open the predicted masks for visual inspection:

```bat
explorer "Synthetic_Background_Test\predictions"
```

The summary includes mean, median, and 10th-percentile scores, the numbers of
strong, weak, and zero-overlap masks, shape accuracy, mean latency, and measured
throughput. The CSV contains one row per image with these columns:

```text
file, expected_shape, detected_shape, proposal_count, selected_source,
selection_score, selected_area_ratio, latency_ms, iou, dice, precision,
recall, boundary_iou
```

All accuracy metrics range from `0.0` to `1.0`, where a value closer to `1.0`
is better:

- **IoU** measures the overlap between the predicted and true foreground masks.
- **Dice** is another overlap score that gives more weight to shared pixels.
- **Precision** measures how much of the prediction is actually part of the sign.
- **Recall** measures how much of the true sign was successfully retained.
- **Boundary-IoU** measures agreement around the sign boundary.
- **Shape accuracy** compares the detected geometric class with its label.

### View custom-dataset results

After running:

```bat
run.bat --evaluate-segmentation "D:\My segmentation test"
```

view:

```text
D:\My segmentation test\evaluation_summary.txt   overall scores
D:\My segmentation test\evaluation.csv           one row per image
D:\My segmentation test\predictions\             predicted masks
```

### View CamVid results

CamVid results are saved in the directory supplied through
`--external-output`. For example, the fixed-test command later in this README
writes:

```text
External_Test_Results\CamVid32_HeldOut_Test\
|-- evaluation_summary.txt   overall scores and evaluated-region counts
|-- evaluation.csv           one row per evaluated sign component
|-- diagnostics\             source/truth/prediction/error comparison panels
|-- predictions\             predicted binary masks
`-- truth\                   corresponding CamVid32 ground-truth masks
```

Open the complete result folder with:

```bat
explorer "External_Test_Results\CamVid32_HeldOut_Test"
```

If `--no-external-predictions` was used, the CSV and summary are still written,
but the per-sample prediction and diagnostic images are omitted.

### View self-test results

`run.bat --self-test` does not create a CSV because it checks software behavior
rather than scoring a dataset. Each check prints `[PASS]` or `[FAIL]` in the
terminal, followed by the total, such as `15 passed, 0 failed`.

## Requirements

- 64-bit Windows 10 or Windows 11
- Visual Studio 2019/2022 or Visual Studio Build Tools
- The **Desktop development with C++** workload, including the x64 MSVC tools
- A compatible 64-bit Windows build of OpenCV 4.x
- Approximately 100 MB of free space for the executable, OpenCV runtime DLLs,
  and batch-test output; external datasets require substantially more space

Python, CMake, and the Visual Studio IDE are not required for the supplied
build script.

## 1. Install OpenCV

OpenCV is deliberately excluded from Git because a local distribution contains
hundreds of megabytes of downloaded sources, libraries, samples, and binaries.
The build script supports either of the following arrangements.

### Option A: place OpenCV inside the project

Extract the Windows OpenCV package so the project has this structure:

```text
SX_mini_project_1\
|-- opencv\
|   `-- build\
|       |-- include\opencv2\opencv.hpp
|       `-- x64\vcXX\
|           |-- bin\opencv_world*.dll
|           `-- lib\opencv_world*.lib
|-- build.bat
|-- run.bat
`-- Source.cpp
```

`vcXX` represents the compiler folder supplied by the OpenCV package, such as
`vc16` or `vc17`. `build.bat` detects the available folder and release
`opencv_world` library automatically.

### Option B: use an OpenCV installation elsewhere

Set `OPENCV_DIR` to the directory that directly contains `build\include` and
`build\x64`, then build the project.

Command Prompt:

```bat
set OPENCV_DIR=D:\Libraries\opencv
build.bat
```

PowerShell:

```powershell
$env:OPENCV_DIR = 'D:\Libraries\opencv'
.\build.bat
```

The selected OpenCV package must contain an x64 MSVC `opencv_world*.lib` file
and its matching runtime DLL.

## 2. Build the application

Command Prompt:

```bat
build.bat
```

PowerShell:

```powershell
.\build.bat
```

The script performs the following work automatically:

1. Finds OpenCV from `OPENCV_DIR` or the local `opencv\` directory.
2. Finds the newest compatible `build\x64\vc*` OpenCV library directory.
3. Locates and initializes the Visual Studio x64 compiler with `vswhere.exe`.
4. Compiles all project sources with C++17, optimization, and strict warnings.
5. Creates `build\ShapeDetection.exe`.
6. Copies the required OpenCV runtime and optional video backend beside it.

A successful build ends with a message similar to:

```text
Build OK : build\ShapeDetection.exe
```

## 3. Run the fast regression suite

This is the recommended first test after every build:

```bat
run.bat --self-test
```

It performs 15 deterministic checks covering:

- empty and disjoint binary masks;
- Boundary-IoU behavior at image borders;
- invalid or empty candidate input;
- cleanup and hole filling;
- UNKNOWN-only abstention;
- ideal circles, triangles, rectangles, and octagons;
- reconstruction of a non-empty 8-bit mask for every supported shape.

`run.bat` returns exit code `0` when all checks pass and a nonzero code when a
build, input, or test failure occurs. In PowerShell, inspect the latest exit
code with:

```powershell
$LASTEXITCODE
```

## 4. Test the supplied 84 images

Run the complete supplied dataset without opening GUI windows:

```bat
run.bat -batch
```

The program recursively reads PNG files below `Test_84_Signs\`, compares the
detected shape with `shape_labels.txt`, and writes four files per input image:

- `<name>_stages.png` — the six processing stages;
- `<name>_result.png` — the source and final segmentation side by side;
- `<name>_mask.png` — the final 0/255 binary mask;
- `<name>_segmented.png` — the isolated sign on a black background.

It also writes `Outputs\refinement_diagnostics.csv`, which records proposal,
selection, geometry, color, and mask-refinement diagnostics for each image.

The currently documented clean-run expectation is:

```text
images processed              84
non-empty signs recognised    84 / 84
shape accuracy                84 / 84 (100.0%)
```

Use verbose mode to print every proposal considered by the detector:

```bat
run.bat -batch --verbose
```

To batch-test another folder of PNG or JPG traffic-sign images:

```bat
run.bat "D:\My traffic signs" -batch
```

## 5. Run the interactive demonstration

```bat
run.bat
```

The default input is `Test_84_Signs\`. During the demonstration:

- press `Esc` to quit;
- press `s` to save the current result;
- press any other key to advance to the next image.

An alternative input folder may be supplied:

```bat
run.bat "D:\My traffic signs"
```

Interactive mode requires a normal desktop session capable of displaying
OpenCV windows. Use `-batch` on a headless or unattended machine.

## 6. Run the deterministic synthetic benchmark

```bat
run.bat --synthetic-test
```

This command generates 64 repeatable 256x256 challenge images containing four
sign shapes across varied backgrounds, along with exact ground-truth masks. It
then evaluates IoU, Dice, precision, recall, Boundary-IoU, shape accuracy, and
runtime performance.

Generated files are written below `Synthetic_Background_Test\`:

```text
Synthetic_Background_Test\
|-- images\
|-- masks\
|-- predictions\
|-- evaluation.csv
|-- evaluation_summary.txt
|-- labels.txt
`-- metadata.csv
```

The entire directory is generated and Git-ignored; rerunning the command safely
recreates or updates its contents.

## 7. Evaluate a custom mask dataset

Prepare a directory with this layout:

```text
D:\My segmentation test\
|-- images\
|   |-- example_001.png
|   `-- example_002.png
|-- masks\
|   |-- example_001.png
|   `-- example_002.png
`-- labels.txt                 optional
```

Requirements:

- input images must be PNG files directly inside `images\`;
- each truth mask must have the same filename and dimensions as its image;
- zero-valued mask pixels are background and any nonzero value is foreground;
- `labels.txt` is optional and enables shape-accuracy reporting;
- each label line uses `<filename> <SHAPE>`, where `SHAPE` is `CIRCLE`,
  `TRIANGLE`, `RECTANGLE`, or `OCTAGON`.

Example `labels.txt`:

```text
example_001.png CIRCLE
example_002.png TRIANGLE
```

Run the evaluation:

```bat
run.bat --evaluate-segmentation "D:\My segmentation test"
```

The evaluator writes `evaluation.csv`, `evaluation_summary.txt`, and a
`predictions\` directory inside the supplied dataset root.

## 8. Run the optional CamVid test

CamVid is not included in Git. This test requires the separately downloaded and
extracted CamVid frames plus the original CamVid32 label masks. With the layout
used by this project, run the corrected fixed-test protocol as follows:

```bat
run.bat --evaluate-camvid "External_Datasets\CamVid\raw" ^
  --camvid-label-root "External_Datasets\CamVid\raw32_labels" ^
  --camvid-split test --camvid-context 1.5 --camvid-box-prompt ^
  --external-output "External_Test_Results\CamVid32_HeldOut_Test"
```

Important: `--camvid-box-prompt` uses the ground-truth component box to simulate
an upstream detector. Therefore, this command evaluates segmentation given a
sign location; it does not evaluate end-to-end full-frame sign detection.

To calculate metrics without saving hundreds of prediction images, add:

```text
--no-external-predictions
```

Available CamVid splits are `train`, `val`, `test`, and `all`.

## Command reference

| Command | Purpose |
|---|---|
| `run.bat` | Interactive demonstration using `Test_84_Signs\` |
| `run.bat <folder>` | Interactive demonstration using another folder |
| `run.bat -batch` | Headless 84-image regression and output generation |
| `run.bat <folder> -batch` | Headless processing of another image folder |
| `run.bat --self-test` | Fast deterministic 15-check regression suite |
| `run.bat --synthetic-test` | Generate and evaluate the 64-image exact-mask stress set |
| `run.bat --evaluate-segmentation <root>` | Evaluate a custom `images\` and `masks\` dataset |
| `run.bat --evaluate-camvid <root>` | Evaluate extracted CamVid frames and labels |
| `--verbose` or `-v` | Print all detector proposals |
| `--camvid-split train\|val\|test\|all` | Select the CamVid dataset split |
| `--camvid-context <scale>` | Set the CamVid crop context multiplier |
| `--camvid-label-root <root>` | Use the original CamVid32 label directory |
| `--camvid-box-prompt` | Supply the target component box to segmentation |
| `--external-output <root>` | Select the CamVid report/output directory |
| `--no-external-predictions` | Calculate CamVid metrics without saving prediction images |

## Generated files and Git

The following directories are expected to be local and are excluded by
`.gitignore`:

| Path | Contents |
|---|---|
| `build\` | Executable, object files, and OpenCV runtime DLLs |
| `opencv\` | Local OpenCV distribution |
| `Outputs\` | Standard batch-test images and diagnostics |
| `Synthetic_Background_Test\` | Generated synthetic benchmark data |
| `External_Datasets\` | Downloaded external datasets, except provenance notes |
| `External_Test_Results\` | Generated external evaluation artifacts, except curated READMEs |

The supplied `Test_84_Signs\` input images and `shape_labels.txt` are retained
in Git because they are required for the standard 84-image test.

## Troubleshooting

### `OpenCV headers not found`

`OPENCV_DIR` is incorrect or the OpenCV package was extracted at an unexpected
level. It must point to the directory containing:

```text
build\include\opencv2\opencv.hpp
```

### `No ...\build\x64\vc##\lib folder found`

Install a Windows x64 OpenCV package built for MSVC. Confirm that
`%OPENCV_DIR%\build\x64\vcXX\lib` exists.

### `Cannot find vcvars64.bat`

Open Visual Studio Installer and add the **Desktop development with C++**
workload. The script searches Visual Studio 2022 and 2019 installations.

### `ShapeDetection.exe not found`

Run `build.bat` first and resolve any compiler or OpenCV error printed above
`BUILD FAILED`.

### A required OpenCV DLL is missing at startup

Re-run `build.bat`. It copies the matching `opencv_world*.dll` beside
`build\ShapeDetection.exe`. Also confirm that the OpenCV `bin\` directory
contains the DLL matching the selected `.lib` file.

### `No PNG evaluation images found`

For `--evaluate-segmentation`, PNG inputs must be directly inside
`<root>\images\`, and matching masks must be directly inside `<root>\masks\`.

## Further documentation

- [`README_ShapeDetection.md`](README_ShapeDetection.md) — algorithm,
  implementation, outputs, and quantitative results
- [`TECHNICAL_AUDIT_2026.md`](TECHNICAL_AUDIT_2026.md) — verification evidence,
  limitations, performance, and regression analysis
- [`SEGMENTATION_ANALYSIS.md`](SEGMENTATION_ANALYSIS.md) — segmentation decisions
  and diagnostic interpretation
- [`REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md`](REAL_BACKGROUND_SEGMENTATION_ANALYSIS.md)
  — corrected CamVid protocol, failure analysis, and reproduction commands
