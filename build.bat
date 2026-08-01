@echo off
REM ===========================================================================
REM  build.bat - compile the shape detection module without the Visual Studio
REM              IDE.
REM
REM  Nothing normally needs to be edited:  the script finds OpenCV in the
REM  "opencv" folder that sits next to it, works out the vc## / world library
REM  version by itself, and locates the Visual Studio build tools with vswhere.
REM
REM  To use an OpenCV that lives somewhere else, either edit OPENCV_DIR below
REM  or set it in the environment before calling this script, e.g.
REM      set OPENCV_DIR=D:\libs\opencv
REM      build.bat
REM  It must be the folder that contains  build\include  and  build\x64\vc16 .
REM ===========================================================================
setlocal EnableDelayedExpansion

REM Always work from the folder this script lives in.
pushd "%~dp0"

REM ---------------------------------------------------------------- OpenCV --
if not defined OPENCV_DIR set "OPENCV_DIR=%~dp0opencv"
if "%OPENCV_DIR:~-1%"=="\" set "OPENCV_DIR=%OPENCV_DIR:~0,-1%"

set "OPENCV_INC=%OPENCV_DIR%\build\include"
if not exist "%OPENCV_INC%\opencv2\opencv.hpp" (
    echo [ERROR] OpenCV headers not found at "%OPENCV_INC%".
    echo         Set OPENCV_DIR to the folder holding build\include.
    goto :fail
)

REM Pick the newest vc## folder that actually has a lib directory (vc16, vc17).
set "OPENCV_VC="
for /d %%D in ("%OPENCV_DIR%\build\x64\vc*") do (
    if exist "%%~D\lib" set "OPENCV_VC=%%~D"
)
if not defined OPENCV_VC (
    echo [ERROR] No "%OPENCV_DIR%\build\x64\vc##\lib" folder found.
    goto :fail
)

REM Work out the release world library name, e.g. opencv_world4110.lib
REM (the debug one ends with "d" and is skipped).
set "OCV_LIB="
for %%F in ("%OPENCV_VC%\lib\opencv_world*.lib") do (
    set "N=%%~nF"
    if /i not "!N:~-1!"=="d" set "OCV_LIB=%%~nxF"
)
if not defined OCV_LIB (
    echo [ERROR] No opencv_world*.lib found in "%OPENCV_VC%\lib".
    goto :fail
)
set "OCV_DLL=!OCV_LIB:.lib=.dll!"

echo Using OpenCV : %OPENCV_DIR%
echo Library      : !OCV_LIB!
echo.

REM ------------------------------------------------- Visual Studio compiler --
where cl.exe >nul 2>nul
if not errorlevel 1 goto :have_compiler

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSDIR="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%I"
)
if defined VSDIR set "VCVARS=!VSDIR!\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "!VCVARS!" (
    echo [ERROR] Cannot find vcvars64.bat - install the "Desktop development
    echo         with C++" workload, or set VCVARS in this file by hand.
    goto :fail
)

REM vcvars64.bat itself needs vswhere.exe on the PATH.
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "!VCVARS!" >nul
if errorlevel 1 (
    echo [ERROR] Failed to initialise the Visual Studio x64 environment.
    goto :fail
)

:have_compiler

REM ------------------------------------------------------------- compiling --
if not exist build mkdir build

REM C4864 originates in OpenCV 4.11's mat.inl.hpp under MSVC /W4; suppress
REM that third-party diagnostic while keeping strict warnings for this project.
cl /nologo /EHsc /O2 /std:c++17 /MD /W4 /permissive- /wd4864 ^
   /I "%OPENCV_INC%" ^
   Source.cpp ShapeDetect.cpp SegmentationTest.cpp supp.cpp ^
   /Fe:build\ShapeDetection.exe /Fo:build\ ^
   /link /LIBPATH:"%OPENCV_VC%\lib" !OCV_LIB!
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    goto :fail
)

REM The exe needs the OpenCV runtime next to it.
copy /y "%OPENCV_VC%\bin\!OCV_DLL!" build\ >nul
if errorlevel 1 (
    echo [WARN] Could not copy !OCV_DLL! - the exe will not start without it.
)
REM Optional video backend, present in most OpenCV packages.
for %%F in ("%OPENCV_VC%\bin\opencv_videoio_ffmpeg*_64.dll") do copy /y "%%~F" build\ >nul

echo.
echo Build OK : build\ShapeDetection.exe
echo.
echo Run it from THIS folder so that "Test_84_Signs" can be found:
echo     run.bat                  ^(interactive demo^)
echo     run.bat -batch           ^(no windows, writes Outputs\^)
popd
endlocal
exit /b 0

:fail
popd
endlocal
exit /b 1
