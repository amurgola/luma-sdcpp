# Build sd-server + sd-cli (CUDA 13, Windows on Arm) from work/stable-diffusion.cpp
# natively on an ARM64 host, and package the two zips LumaBrowser's runtime
# installer expects (names mirror the x64 pair from build-windows-cuda.ps1):
#   sd-<Tag>-bin-win-cuda13-arm64.zip   sd-server, sd-cli, ggml + sd DLLs
#   cudart-sd-bin-win-cu13-arm64.zip    CUDA runtime DLLs (redistributed per
#                                       the CUDA EULA redist list)
# Requires: VS 2022 Build Tools with the ARM64 C++ tools (VC.Tools.ARM64),
# cmake + ninja on PATH, and a CUDA 13.x toolkit with Windows ARM64 target
# libraries (lib\arm64, bin\arm64) - NVIDIA's windows-arm64 redist archives.
param(
  [string]$Tag = 'local',
  [string]$CudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4',
  # Every NVIDIA GPU in a Windows on Arm machine so far is the RTX Spark's
  # Blackwell (sm_121), so one native arch keeps the compile short.
  [string]$Archs = '121a-real',
  [int]$Jobs = [Math]::Max(1, [int]([Environment]::ProcessorCount / 2)),
  [switch]$LowPriority
)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path "$PSScriptRoot\.."
$work = Join-Path $root 'work\stable-diffusion.cpp'
$dist = Join-Path $root 'dist'
if (-not (Test-Path $work)) { throw 'work/stable-diffusion.cpp missing - run scripts\checkout.ps1 first.' }
if ($LowPriority) { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsroot = & $vswhere -products * -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath | Select-Object -First 1
if (-not $vsroot) { throw 'No VS 2022 toolset with ARM64 C++ tools found (vswhere).' }
$vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvarsall.bat'
$nvcc = Join-Path $CudaPath 'bin\nvcc.exe'
$cudaLib = (Join-Path $CudaPath 'lib\arm64') -replace '\\', '/'

# Upstream's windows cuda release defines, with two Arm differences:
# - GGML_NATIVE=ON instead of GGML_CPU_ALL_VARIANTS: ggml refuses MSVC on ARM
#   otherwise, and only probes the build host's CPU features. A Neoverse N2
#   runner (ARMv9.0) is a subset of the RTX Spark N1X's ARMv9.2 cores.
# - FindCUDAToolkit picks lib\x64 from the host on Windows; the ARM64 import
#   libraries are pinned explicitly (as ggml-org's arm64 CUDA build does).
$defines = "-DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=ON -DSD_SERVER_BUILD_FRONTEND=OFF -DCMAKE_CUDA_ARCHITECTURES=`"$Archs`" -DCMAKE_CUDA_COMPILER=`"$nvcc`" -DCUDAToolkit_ROOT=`"$CudaPath`" -DCUDA_CUDART=`"$cudaLib/cudart.lib`" -DCUDA_cudart_LIBRARY=`"$cudaLib/cudart.lib`" -DCUDA_cublas_LIBRARY=`"$cudaLib/cublas.lib`" -DCUDA_cublasLt_LIBRARY=`"$cudaLib/cublasLt.lib`" -DCUDA_cuda_driver_LIBRARY=`"$cudaLib/cuda.lib`""
# CMAKE_CXX_FLAGS replaces CMake's MSVC defaults rather than adding to them, so /EHsc is restated: without it the
# examples' common.cpp (sd-server's request handling) is built with no C++ exception unwinding (warning C4530).
cmd /c "call `"$vcvars`" arm64 >nul && cmake -S `"$work`" -B `"$work\build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CXX_FLAGS=`"/DWIN32 /D_WINDOWS /EHsc /bigobj`" $defines && cmake --build `"$work\build`" -- -j $Jobs"
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

New-Item -ItemType Directory -Force $dist | Out-Null
$binZip = Join-Path $dist "sd-$Tag-bin-win-cuda13-arm64.zip"
$cudartZip = Join-Path $dist 'cudart-sd-bin-win-cu13-arm64.zip'
if (Test-Path $binZip) { Remove-Item $binZip -Confirm:$false }
if (Test-Path $cudartZip) { Remove-Item $cudartZip -Confirm:$false }

Compress-Archive -Path "$work\build\bin\*.exe", "$work\build\bin\*.dll" -DestinationPath $binZip
$runtimeDlls = Get-ChildItem "$CudaPath\bin\arm64" | Where-Object { $_.Name -match '^(cudart64|cublas64|cublasLt64)_13.*\.dll$' }
if (-not $runtimeDlls) { throw "no CUDA runtime DLLs found under $CudaPath\bin\arm64" }
Compress-Archive -Path $runtimeDlls.FullName -DestinationPath $cudartZip

Get-Item $binZip, $cudartZip | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
