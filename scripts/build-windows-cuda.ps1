# Build sd-server + sd-cli (CUDA) from work/stable-diffusion.cpp with
# upstream's release flags, and package the two zips LumaBrowser's runtime
# installer expects (names mirror leejet's release assets):
#   sd-<Tag>-bin-win-cuda12-x64.zip    sd-server, sd-cli, ggml + sd DLLs
#   cudart-sd-bin-win-cu12-x64.zip     CUDA runtime DLLs (redistributed per
#                                      the CUDA EULA redist list)
# Requires: VS 2022 Build Tools (nvcc 12.x cannot host the VS2026 compiler),
# cmake + ninja on PATH, a CUDA 12.x toolkit.
param(
  [string]$Tag = 'local',
  [string]$CudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8',
  # Same architecture list as leejet's cuda12 release build.
  [string]$Archs = '61;70;75;80;86;89;90;100;120',
  # Compile jobs. A full-width CUDA compile makes the machine unusable for
  # anything else while it runs; the CI runner can pass its whole core count.
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
$vsroot = & $vswhere -products * -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsroot) { throw 'No VS 2022 toolset with C++ tools found (vswhere).' }
$vcvars = Join-Path $vsroot 'VC\Auxiliary\Build\vcvars64.bat'
$nvcc = Join-Path $CudaPath 'bin\nvcc.exe'

# Upstream's windows cuda12 release defines, plus: no server web frontend
# (needs pnpm; LumaBrowser talks to the HTTP API only).
$defines = "-DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON -DSD_BUILD_SHARED_GGML_LIB=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DSD_SERVER_BUILD_FRONTEND=OFF -DCMAKE_CUDA_ARCHITECTURES=`"$Archs`" -DCMAKE_CUDA_COMPILER=`"$nvcc`""
# CMAKE_CXX_FLAGS replaces CMake's MSVC defaults rather than adding to them, so /EHsc is restated: without it the
# examples' common.cpp (sd-server's request handling) is built with no C++ exception unwinding (warning C4530).
cmd /c "call `"$vcvars`" >nul && cmake -S `"$work`" -B `"$work\build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_CXX_FLAGS=`"/DWIN32 /D_WINDOWS /EHsc /bigobj`" $defines && cmake --build `"$work\build`" -- -j $Jobs"
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }

New-Item -ItemType Directory -Force $dist | Out-Null
$binZip = Join-Path $dist "sd-$Tag-bin-win-cuda12-x64.zip"
$cudartZip = Join-Path $dist 'cudart-sd-bin-win-cu12-x64.zip'
if (Test-Path $binZip) { Remove-Item $binZip -Confirm:$false }
if (Test-Path $cudartZip) { Remove-Item $cudartZip -Confirm:$false }

Compress-Archive -Path "$work\build\bin\*.exe", "$work\build\bin\*.dll" -DestinationPath $binZip
$runtimeDlls = Get-ChildItem "$CudaPath\bin" | Where-Object { $_.Name -match '^(cudart64|cublas64|cublasLt64)_12.*\.dll$' }
if (-not $runtimeDlls) { throw "no CUDA runtime DLLs found under $CudaPath\bin" }
Compress-Archive -Path $runtimeDlls.FullName -DestinationPath $cudartZip

Get-Item $binZip, $cudartZip | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
