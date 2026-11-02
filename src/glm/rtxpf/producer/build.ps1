param([Parameter(Mandatory=$true)][string]$BuildDir,[switch]$Fresh,[ValidateRange(1,16)][int]$Parallel=2,
    [string]$TierstreamSource='',[string]$CutlassDir='',[string]$ExtraCMake='')
$ErrorActionPreference = 'Stop'
$build = [IO.Path]::GetFullPath($BuildDir)
New-Item -ItemType Directory -Force $build,(Join-Path $build 'tmp') | Out-Null
$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$vs = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$freshOption = if ($Fresh) { '--fresh' } else { '' }
# Optional source locations: the tierstream snapshot (else the CMake default) and an
# offline CUTLASS v4.2.1 tree (else CMake fetches the pinned release).
if ($TierstreamSource) { $freshOption += " -DGLM_TIERSTREAM_SRC=`"$([IO.Path]::GetFullPath($TierstreamSource) -replace '\\','/')`"" }
if ($CutlassDir) { $freshOption += " -DGLM_CUTLASS_DIR=`"$([IO.Path]::GetFullPath($CutlassDir) -replace '\\','/')`"" }
$freshOption += " -DGLM_TS_PARALLEL=$Parallel" # the tierstream sub-build follows -Parallel
if ($ExtraCMake) { $freshOption += " $ExtraCMake" } # e.g. diagnostic symbols
$lines = @('@echo off', 'set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"', "set `"TMP=$build\tmp`"", "set `"TEMP=$build\tmp`"", 'set "CCACHE_DISABLE=1"',
    "call `"$vs`" >nul", 'if errorlevel 1 exit /b 1', 'chcp 65001 >nul',
    "cmake $freshOption -S `"$PSScriptRoot`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=`"$($cuda -replace '\\','/')/bin/nvcc.exe`" -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_CUDA_COMPILER_LAUNCHER=",
    'if errorlevel 1 exit /b 1', "cmake --build `"$build`" --parallel $Parallel", 'exit /b %errorlevel%')
$script = Join-Path $build 'producer-build.cmd'
$lines | Set-Content -LiteralPath $script -Encoding ascii
$process = Get-Process -Id $PID
$oldPriority = $process.PriorityClass
try {
    $process.PriorityClass = 'Idle'
    & cmd /d /c $script 2>&1 | Tee-Object (Join-Path $build 'producer-build.log')
    if ($LASTEXITCODE -ne 0) { throw 'Producer build failed' }
    # Compiler artifacts only. Never load a service/benchmark or CUDA runtime.
    & python -B (Join-Path $PSScriptRoot 'inspect_build.py') --build $build --cuobjdump (Join-Path $cuda 'bin/cuobjdump.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Producer/bench SASS check failed' }
} finally { $process.PriorityClass = $oldPriority }
Get-FileHash (Join-Path $build 'glm_native_service.exe'),(Join-Path $build 'glm_native_integrity.exe') -Algorithm SHA256
