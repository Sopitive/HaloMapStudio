<#
.SYNOPSIS
  Build Halo Map Studio from source, natively on Windows (no WSL, no cross-compile).

.DESCRIPTION
  1. checks the prerequisites and stops at the first missing one with the command that fixes it,
  2. builds the native map-parser library HaloMapStudioDLL.dll with MSBuild
     (native\HaloMapStudioNative.sln, Release|x64, vendored MinHook / libdeflate / miniz),
  3. builds hms-app.exe with `cargo build --release --locked -p hms-app`,
  4. assembles dist\windows-x64\ (the same file set as the release zip),
  5. with -Package, also writes dist\release\HaloMapStudio-<version>-windows-x64.zip (+ .sha256).

  Nothing is downloaded by this script (cargo fetches crates from crates.io, which is the only
  network use) and nothing outside the repository folder is modified. See BUILDING.md.

.PARAMETER SkipNative
  Reuse an existing native\x64\<Config>\HaloMapStudioDLL.dll instead of running MSBuild.
.PARAMETER Debug
  Debug configuration (cargo dev profile, MSBuild Debug|x64). Staged in target\debug only;
  dist\ is assembled for release builds.
.PARAMETER Clean
  Remove previous build output first (native\x64, native\staging, dist\windows-x64, cargo target\).
.PARAMETER Package
  Also write the release-style zip and its SHA-256 file to dist\release\.
.PARAMETER Verbose
  Show the full MSBuild / cargo command lines and verbose build output.

.EXAMPLE
  .\build-windows.ps1
.EXAMPLE
  .\build-windows.ps1 -Package
#>
param(
    [switch]$SkipNative,
    [switch]$Debug,
    [switch]$Clean,
    [switch]$Package,
    [switch]$Verbose,
    [switch]$Help
)

# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads a BOM-less file as the ANSI code
# page, so any non-ASCII character would be mangled.
$ErrorActionPreference = 'Stop'

if ($Help) { Get-Help $MyInvocation.MyCommand.Path -Detailed; exit 0 }

$Repo   = $PSScriptRoot
$Config = 'Release'
if ($Debug) { $Config = 'Debug' }
$Prof = $Config.ToLower()             # cargo output dir: release | debug

function Step($msg) { Write-Host ''; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    [ok] $msg" -ForegroundColor Green }
function Note($msg) { Write-Host "    $msg" }
function Warn($msg) { Write-Host "    [warn] $msg" -ForegroundColor Yellow }

# Stop at the first problem: what is wrong, then what to do about it.
function Fail($what, $fix) {
    Write-Host ''
    Write-Host "ERROR: $what" -ForegroundColor Red
    if ($fix) {
        Write-Host ''
        Write-Host 'How to fix:' -ForegroundColor Yellow
        foreach ($line in @($fix)) { Write-Host "  $line" }
    }
    Write-Host ''
    Write-Host 'See BUILDING.md (Troubleshooting) for more.'
    exit 1
}

# Run a native command and return its stdout lines; never throws (stderr dropped, exit code ignored).
function Capture([string]$exe, [string[]]$arguments) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $exe @arguments 2>$null
        return @($out)
    } catch {
        return @()
    } finally {
        $ErrorActionPreference = $old
    }
}

# Run a native command with output going to the console; throws on a non-zero exit.
function Run([string]$what, [string]$exe, [string[]]$arguments) {
    if ($Verbose) { Note ("> " + $exe + " " + ($arguments -join ' ')) }
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)." 'Read the build output above for the first error.' }
}

# Only ever delete things that are really inside the repository folder.
function Remove-InRepo([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $full = [System.IO.Path]::GetFullPath($path)
    $root = [System.IO.Path]::GetFullPath($Repo).TrimEnd('\') + '\'
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        Fail "refusing to delete '$full': it is outside the repository." $null
    }
    Remove-Item -LiteralPath $full -Recurse -Force
}

Write-Host "Halo Map Studio - Windows source build ($Config, x64)" -ForegroundColor White
Note "repository: $Repo"

# ---------------------------------------------------------------- 0. platform + checkout sanity
if ([Environment]::OSVersion.Platform -ne 'Win32NT') {
    Fail 'this script builds the Windows version and must run on Windows.' 'On Linux use ./build-all.sh linux (see BUILDING.md).'
}
if (-not [Environment]::Is64BitOperatingSystem) { Fail 'a 64-bit Windows is required.' $null }

$Sln     = Join-Path $Repo 'native\HaloMapStudioNative.sln'
$Vcxproj = Join-Path $Repo 'native\MccMapStudioDLL\MccMapStudioDLL.vcxproj'
$required = @(
    'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml',
    'native\HaloMapStudioNative.sln',
    'native\MccMapStudioDLL\MccMapStudioDLL.vcxproj',
    'native\MccMapStudioDLL\HaloMapStudioDLL.def',
    'native\MccMapStudioDLL\miniz.c',
    'native\MccMapStudioDLL\libdeflate\lib\deflate_decompress.c',
    'native\MccMapStudioDLL\libdeflate\libdeflate.h',
    'native\packages\minhook.1.3.3\lib\native\include\MinHook.h',
    'native\packages\minhook.1.3.3\lib\native\src\hook.c',
    'crates\hms-app\Cargo.toml',
    'dist\README.md', 'LICENSE.md', 'COMMERCIAL-LICENSE.md', 'THIRD-PARTY-NOTICES.md'
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Repo $_)) })
if ($missing.Count -gt 0) {
    Fail ("the checkout is incomplete; missing: " + ($missing -join ', ')) @(
        'Clone the repository with git (do not use a partial download):',
        '  git config --global core.longpaths true',
        '  git clone <repository url>',
        'No submodules are used: libdeflate, MinHook and miniz are vendored under native\.')
}
if ($Repo.Length -gt 90) {
    Warn "the repository path is long ($($Repo.Length) chars). If the build fails with 'path too long', move the folder closer to the drive root (for example C:\src\hms) and run  git config --global core.longpaths true"
}

# ---------------------------------------------------------------- 1. prerequisites
Step 'Checking prerequisites'

# --- Rust (rustup + the MSVC host toolchain)
$cargoBin = $null
if ($env:CARGO_HOME) { $cargoBin = Join-Path $env:CARGO_HOME 'bin' } elseif ($env:USERPROFILE) { $cargoBin = Join-Path $env:USERPROFILE '.cargo\bin' }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue) -and $cargoBin -and (Test-Path -LiteralPath (Join-Path $cargoBin 'cargo.exe'))) {
    $env:PATH = $cargoBin + ';' + $env:PATH      # this process only
    Note "cargo was not on PATH; using $cargoBin for this run"
}
if (-not (Get-Command cargo -ErrorAction SilentlyContinue) -or -not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Fail 'Rust (cargo / rustc) was not found.' @(
        'Install rustup, then open a NEW terminal and rerun this script:',
        '  winget install Rustlang.Rustup',
        '(or download rustup-init.exe from https://rustup.rs and choose the default MSVC host).')
}
Push-Location $Repo
try {
    $rustcInfo = Capture 'rustc' @('-vV')
    $cargoVer  = Capture 'cargo' @('--version')
    $targets   = @()
    if (Get-Command rustup -ErrorAction SilentlyContinue) { $targets = Capture 'rustup' @('target', 'list', '--installed') }
} finally { Pop-Location }
$hostLine = @($rustcInfo | Where-Object { "$_" -like 'host:*' })
if ($hostLine.Count -eq 0) {
    Fail "rustc did not run (rust-toolchain.toml pins the 'stable' channel and may not be installed)." @(
        'Install the pinned toolchain:',
        '  rustup toolchain install stable --target x86_64-pc-windows-msvc',
        'then rerun this script.')
}
$hostTriple = ("$($hostLine[0])").Substring(5).Trim()
if ($hostTriple -ne 'x86_64-pc-windows-msvc') {
    Fail "the active Rust host is '$hostTriple'; this build needs x86_64-pc-windows-msvc (not the GNU/MinGW toolchain)." @(
        'Switch the default host to MSVC:',
        '  rustup set default-host x86_64-pc-windows-msvc',
        '  rustup toolchain install stable-x86_64-pc-windows-msvc',
        '  rustup default stable-x86_64-pc-windows-msvc')
}
if ((Get-Command rustup -ErrorAction SilentlyContinue) -and (@($targets | Where-Object { "$_" -eq 'x86_64-pc-windows-msvc' }).Count -eq 0)) {
    Fail 'the Rust target x86_64-pc-windows-msvc is not installed for the pinned toolchain.' 'rustup target add x86_64-pc-windows-msvc'
}
Ok ("Rust: " + (@($rustcInfo | Where-Object { "$_" -like 'rustc *' }) | Select-Object -First 1) + "  [$hostTriple]  " + ($cargoVer | Select-Object -First 1))

# --- Visual Studio 2022 / Build Tools with the C++ workload (vswhere ships with the VS installer)
$vswhere = $null
$pf86 = ${env:ProgramFiles(x86)}
if ($pf86) {
    $cand = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $cand) { $vswhere = $cand }
}
if (-not $vswhere) {
    $cmd = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($cmd) { $vswhere = $cmd.Source }
}
$vsPath = $null
$installHelp = @(
    'Install the Visual Studio 2022 Build Tools with the C++ workload (about 6 GB, free):',
    '  winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"',
    'or open the Visual Studio Installer, Modify, and tick "Desktop development with C++".',
    'Then rerun this script (a new terminal is not required).')
if ($vswhere) {
    $found = Capture $vswhere @('-latest', '-products', '*', '-version', '[17.0,)', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath')
    if ($found.Count -gt 0 -and "$($found[0])".Trim()) { $vsPath = "$($found[0])".Trim() }
}
if (-not $vsPath) {
    Fail 'Visual Studio 2022 (or Build Tools) with the "Desktop development with C++" workload was not found.' $installHelp
}
Ok "Visual Studio: $vsPath"

# --- MSBuild
$MSBuild = Join-Path $vsPath 'MSBuild\Current\Bin\MSBuild.exe'
if (-not (Test-Path -LiteralPath $MSBuild)) {
    $f = Capture $vswhere @('-latest', '-products', '*', '-requires', 'Microsoft.Component.MSBuild', '-find', 'MSBuild\**\Bin\MSBuild.exe')
    if ($f.Count -gt 0 -and (Test-Path -LiteralPath "$($f[0])")) { $MSBuild = "$($f[0])" }
}
if (-not (Test-Path -LiteralPath $MSBuild)) {
    Fail 'MSBuild.exe was not found in the Visual Studio installation.' $installHelp
}
Ok "MSBuild: $MSBuild"

# --- C++ platform toolset (the project asks for v143 = VS 2022 / MSVC 14.3x)
$toolsetOverride = $null
$toolsetRoot = Join-Path $vsPath 'MSBuild\Microsoft\VC'
$toolsets = @()
if (Test-Path -LiteralPath $toolsetRoot) {
    $toolsets = @(Get-ChildItem -LiteralPath $toolsetRoot -Directory -Filter 'v*' -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'Platforms\x64\PlatformToolsets' } |
        Where-Object { Test-Path -LiteralPath $_ } |
        ForEach-Object { Get-ChildItem -LiteralPath $_ -Directory -ErrorAction SilentlyContinue } |
        ForEach-Object { $_.Name } | Sort-Object -Unique)
}
if ($toolsets -contains 'v143') {
    Ok 'C++ toolset: v143'
} elseif ($toolsets.Count -gt 0) {
    $toolsetOverride = @($toolsets | Where-Object { $_ -match '^v14\d$' } | Sort-Object | Select-Object -Last 1)[0]
    if (-not $toolsetOverride) { Fail "no usable C++ platform toolset found (saw: $($toolsets -join ', '))." $installHelp }
    Warn "toolset v143 is not installed; building with $toolsetOverride instead (untested). Preferred: add the component 'MSVC v143 - VS 2022 C++ x64/x86 build tools'."
} else {
    Fail 'the MSVC C++ x64 toolset is not installed.' $installHelp
}

# --- Windows SDK (headers + libs for both the C++ DLL and Rust's linker)
$sdkRoots = @()
foreach ($k in @('HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots', 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots')) {
    try {
        $p = (Get-ItemProperty -LiteralPath $k -ErrorAction Stop).KitsRoot10
        if ($p) { $sdkRoots += $p }
    } catch { }
}
$sdkVer = $null
foreach ($r in $sdkRoots) {
    $inc = Join-Path $r 'Include'
    if (-not (Test-Path -LiteralPath $inc)) { continue }
    $vers = @(Get-ChildItem -LiteralPath $inc -Directory -ErrorAction SilentlyContinue | Where-Object {
        (Test-Path -LiteralPath (Join-Path $_.FullName 'um\windows.h')) -and
        (Test-Path -LiteralPath (Join-Path $r ("Lib\" + $_.Name + "\um\x64\kernel32.lib")))
    } | Sort-Object { try { [version]$_.Name } catch { [version]'0.0' } })
    if ($vers.Count -gt 0) { $sdkVer = $vers[-1].Name; break }
}
if (-not $sdkVer) {
    Fail 'no Windows 10/11 SDK (with x64 libraries) was found.' @(
        'Add it with the Visual Studio Installer (Individual components: "Windows 11 SDK") or:',
        '  winget install Microsoft.WindowsSDK.10.0.22621',
        'then rerun this script.')
}
Ok "Windows SDK: $sdkVer"

# ---------------------------------------------------------------- 2. clean
$NativeOut = Join-Path $Repo "native\x64\$Config"
$DllPath   = Join-Path $NativeOut 'HaloMapStudioDLL.dll'
$Dist      = Join-Path $Repo 'dist\windows-x64'
$TargetDir = Join-Path $Repo "target\$Prof"

if ($Clean) {
    Step 'Cleaning previous build output'
    Remove-InRepo (Join-Path $Repo 'native\x64')
    Remove-InRepo (Join-Path $Repo 'native\staging')
    Remove-InRepo $Dist
    Push-Location $Repo
    try { Run 'cargo clean' 'cargo' @('clean') } finally { Pop-Location }
    Ok 'clean'
}

# ---------------------------------------------------------------- 3. native DLL
if ($SkipNative) {
    Step 'Native library: skipped (-SkipNative)'
    if (-not (Test-Path -LiteralPath $DllPath)) {
        Fail "-SkipNative was given but $DllPath does not exist." 'Run the script once without -SkipNative.'
    }
    Ok "reusing $DllPath"
} else {
    Step "Building HaloMapStudioDLL.dll (MSBuild, $Config|x64)"
    $verbosity = 'minimal'
    if ($Verbose) { $verbosity = 'normal' }
    $msbArgs = @(
        $Sln,
        '/t:MccMapStudioDLL',
        "/p:Configuration=$Config",
        '/p:Platform=x64',
        '/m', "/v:$verbosity", '/nologo', '/nr:false'
    )
    if ($toolsetOverride) { $msbArgs += "/p:PlatformToolset=$toolsetOverride" }
    Run 'The native (MSBuild) build' $MSBuild $msbArgs
    if (-not (Test-Path -LiteralPath $DllPath)) { Fail "MSBuild finished but $DllPath is missing." $null }
}

# The Rust side resolves these by name (crates/hms-native); make sure the DLL really exports them.
$bytes = [System.IO.File]::ReadAllBytes($DllPath)
$text  = [System.Text.Encoding]::GetEncoding(28591).GetString($bytes)
foreach ($sym in @('ZH_MBP_SetSharedCacheDirs', 'ZH_MBP_OpenCache', 'ZH_MBP_DecodeBitmap')) {
    if ($text.IndexOf($sym, [System.StringComparison]::Ordinal) -lt 0) {
        Fail "$DllPath does not export $sym; it is not the HMS native library." 'Rerun without -SkipNative, or with -Clean.'
    }
}
Ok ("{0} ({1:N1} MB)" -f $DllPath, ($bytes.Length / 1MB))
$bytes = $null; $text = $null

# ---------------------------------------------------------------- 4. Rust app
Step "Building hms-app.exe (cargo, $Prof profile)"
$cargoArgs = @('build', '--locked', '-p', 'hms-app')
if (-not $Debug) { $cargoArgs = @('build', '--release', '--locked', '-p', 'hms-app') }
if ($Verbose) { $cargoArgs += '-v' }
Note 'The first build compiles about 220 crates and takes roughly 5-20 minutes; later builds are incremental.'
Push-Location $Repo
try { Run 'The Rust (cargo) build' 'cargo' $cargoArgs } finally { Pop-Location }
$ExePath = Join-Path $TargetDir 'hms-app.exe'
if (-not (Test-Path -LiteralPath $ExePath)) { Fail "cargo finished but $ExePath is missing." $null }
Ok $ExePath

# Stage the DLL beside the exe in target\<profile> so `cargo run` / the exe there works too.
try {
    Copy-Item -LiteralPath $DllPath -Destination (Join-Path $TargetDir 'HaloMapStudioDLL.dll') -Force
    Ok "copied HaloMapStudioDLL.dll next to $ExePath"
} catch {
    Warn "could not copy the DLL into $TargetDir : $($_.Exception.Message)"
}

# ---------------------------------------------------------------- 5. assemble dist\windows-x64
if ($Debug) {
    Step 'Debug build: dist\ not assembled'
    Note "Run: $ExePath"
    exit 0
}

Step 'Assembling dist\windows-x64\'
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$copies = @(
    @($ExePath,                                              'hms-app.exe'),
    @($DllPath,                                              'HaloMapStudioDLL.dll'),
    @((Join-Path $Repo 'dist\README.md'),                    'README.md'),
    @((Join-Path $Repo 'LICENSE.md'),                        'LICENSE.md'),
    @((Join-Path $Repo 'COMMERCIAL-LICENSE.md'),             'COMMERCIAL-LICENSE.md'),
    @((Join-Path $Repo 'THIRD-PARTY-NOTICES.md'),            'THIRD-PARTY-NOTICES.md')
)
foreach ($c in $copies) {
    try {
        Copy-Item -LiteralPath $c[0] -Destination (Join-Path $Dist $c[1]) -Force
    } catch {
        Fail "could not write $($c[1]) to $Dist : $($_.Exception.Message)" 'Close any running hms-app.exe from that folder (or from MCC) and rerun.'
    }
}
Ok $Dist
Get-ChildItem -LiteralPath $Dist -File | ForEach-Object { Note ("{0,-28} {1,12:N0} bytes" -f $_.Name, $_.Length) }

# ---------------------------------------------------------------- 6. optional zip (same layout as build-all.sh package)
$ZipPath = $null
if ($Package) {
    Step 'Packaging'
    $verLine = Select-String -LiteralPath (Join-Path $Repo 'Cargo.toml') -Pattern '^version = "(.*)"' | Select-Object -First 1
    if (-not $verLine) { Fail 'cannot read the workspace version from Cargo.toml.' $null }
    $version = $verLine.Matches[0].Groups[1].Value
    $name    = "HaloMapStudio-$version-windows-x64"
    $relDir  = Join-Path $Repo 'dist\release'
    New-Item -ItemType Directory -Force -Path $relDir | Out-Null
    $ZipPath = Join-Path $relDir "$name.zip"
    $tmpZip  = "$ZipPath.tmp"
    if (Test-Path -LiteralPath $tmpZip) { Remove-Item -LiteralPath $tmpZip -Force }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($tmpZip, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        # one top-level folder, forward-slash entry names (Compress-Archive on 5.1 writes backslashes)
        foreach ($f in @('hms-app.exe', 'HaloMapStudioDLL.dll', 'README.md', 'LICENSE.md', 'COMMERCIAL-LICENSE.md', 'THIRD-PARTY-NOTICES.md')) {
            [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $zip, (Join-Path $Dist $f), "$name/$f", [System.IO.Compression.CompressionLevel]::Optimal)
        }
    } finally { $zip.Dispose() }
    Move-Item -LiteralPath $tmpZip -Destination $ZipPath -Force
    $hash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLower()
    # sha256sum format: "<hash>  <file>" (ASCII, LF)
    [System.IO.File]::WriteAllText("$ZipPath.sha256", "$hash  $name.zip`n", (New-Object System.Text.ASCIIEncoding))
    Ok ("{0} ({1:N1} MB)" -f $ZipPath, ((Get-Item -LiteralPath $ZipPath).Length / 1MB))
    Note "sha256: $hash"
}

Write-Host ''
Write-Host 'Build finished.' -ForegroundColor Green
Write-Host "  Run:  $(Join-Path $Dist 'hms-app.exe')"
if ($ZipPath) { Write-Host "  Zip:  $ZipPath" }
Write-Host '  HMS reads your own installed Halo files; none are bundled. Point it at your MCC install on first run (see BUILDING.md).'
exit 0
