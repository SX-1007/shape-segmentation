@echo off
REM ===========================================================================
REM  run.bat - start the demo from the project folder, so that the relative
REM            paths "Test_84_Signs", "shape_labels.txt" and "Outputs" resolve.
REM
REM      run.bat                          interactive, default folder
REM      run.bat "Test_84_Signs"          interactive, given folder
REM      run.bat -batch                   no windows, writes Outputs\
REM      run.bat --synthetic-test         generate and score exact-mask stress set
REM      run.bat --evaluate-segmentation "D:\dataset"
REM      run.bat --evaluate-camvid "External_Datasets\CamVid\raw"
REM              [--camvid-split train|val|test|all] [--camvid-context 1.5]
REM              [--camvid-label-root "External_Datasets\CamVid\raw32_labels"]
REM              [--camvid-box-prompt]
REM              [--external-output "External_Test_Results\CamVid32_Improved"]
REM              [--no-external-predictions]
REM      run.bat --self-test              fast deterministic regression checks
REM ===========================================================================
setlocal
pushd "%~dp0"

if not exist "build\ShapeDetection.exe" (
    echo ShapeDetection.exe not found - run build.bat first.
    popd & endlocal & exit /b 1
)

build\ShapeDetection.exe %*
set "RC=%ERRORLEVEL%"

popd
endlocal & exit /b %RC%
