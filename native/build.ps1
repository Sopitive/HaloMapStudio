# Builds the native HaloMapStudioDLL and stages it next to the Rust binaries.
#
# Self-contained: everything this needs (source, vendored MinHook under
# packages\, vendored libdeflate under MccMapStudioDLL\libdeflate\) lives in
# this native\ tree. On Linux use build-windows.sh (clang-cl + lld-link) instead.
#
# Usage:   pwsh native\build.ps1            # Release (default)
#          pwsh native\build.ps1 -Config Debug
param(
    [ValidateSet('Release','Debug')]
    [string]$Config = 'Release'
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $here
$sln  = Join-Path $here 'HaloMapStudioNative.sln'

$msbuild = 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'
if (-not (Test-Path $msbuild)) { throw "MSBuild not found at $msbuild" }

& $msbuild $sln /t:MccMapStudioDLL /p:Configuration=$Config /p:Platform=x64 /m /v:m /nologo
if ($LASTEXITCODE -ne 0) { throw "MSBuild failed ($LASTEXITCODE)" }

$dll = Join-Path $here "x64\$Config\HaloMapStudioDLL.dll"
if (-not (Test-Path $dll)) { throw "Built DLL missing: $dll" }

# Stage into the Rust target dir so hms-app picks it up. MCC can hold a lock
# on target\release\HaloMapStudioDLL.dll; if the copy fails, warn (not fatal).
$targetDir = Join-Path $repo "target\$($Config.ToLower())"
if (Test-Path $targetDir) {
    try {
        Copy-Item $dll $targetDir -Force
        Write-Host "Staged HaloMapStudioDLL.dll -> $targetDir"
    } catch {
        Write-Warning "Could not copy DLL to $targetDir (MCC may have it locked): $_"
    }
} else {
    Write-Host "Built $dll (target dir $targetDir not present yet; run cargo build first)"
}
