# Building Halo Map Studio from source

Halo Map Studio (HMS) is a Rust application (`hms-app`) plus a small C++ map-parser library
(`HaloMapStudioDLL.dll` on Windows, `libhalomapstudio.so` on Linux). The two files must sit
next to each other. Everything the build needs is in this repository: libdeflate, MinHook
and miniz are vendored under `native\`, there are no git submodules, and the only network use
is cargo downloading crates from crates.io.

**No Halo content is built, downloaded or bundled.** HMS reads the game files of a copy of
Halo: The Master Chief Collection that you own and have installed. Source builds are licensed
like the rest of the project (PolyForm Noncommercial 1.0.0, see `LICENSE.md`; commercial use
needs a separate licence, see `COMMERCIAL-LICENSE.md`).

## Windows (native, no WSL, no cross-compiling)

### Requirements

| What | Version | Notes |
| --- | --- | --- |
| Windows | 10 or 11, 64-bit | |
| Rust via [rustup](https://rustup.rs) | stable, **MSVC host** (`x86_64-pc-windows-msvc`) | `rust-toolchain.toml` pins the channel and the target |
| Visual Studio 2022 **or** Build Tools 2022 | with the **Desktop development with C++** workload | provides MSVC (toolset v143), MSBuild and the linker |
| Windows 10/11 SDK | any recent one (10.0.19041 or newer) | part of the workload above |
| Disk space | about 10 GB free (Build Tools about 6 GB, `target\` about 3 GB) | |
| Time | first build roughly 5-20 minutes (about 220 crates), later builds are incremental | |
| Network | crates.io only | no other download happens during the build |

You do **not** need CMake, git submodules, WSL, LLVM or cargo-xwin. You do not need to start
from a "Developer" prompt: the script finds Visual Studio itself.

Install what is missing (the script prints these too, at the first thing that is missing):

```powershell
winget install Rustlang.Rustup
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended"
```

Open a new terminal afterwards so `cargo` is on `PATH`.

### Build

Get the sources with git (not a partial download), then from the repository folder:

```powershell
git config --global core.longpaths true      # once; the repo has deep paths
git clone https://github.com/Sopitive/HaloMapStudio.git
cd HaloMapStudio
.\build-windows.ps1
```

`build-windows.cmd` does the same for people who prefer double-clicking or `cmd.exe` (it runs
the script with `-NoProfile -ExecutionPolicy Bypass`). The script works in Windows PowerShell
5.1 and PowerShell 7.

If PowerShell refuses to run the script ("running scripts is disabled"), use
`build-windows.cmd`, or run `powershell -NoProfile -ExecutionPolicy Bypass -File .\build-windows.ps1`.

What it does, in order, stopping at the first problem with the command that fixes it:

1. Checks the checkout, Rust (MSVC host and target), Visual Studio / Build Tools (via
   `vswhere`), MSBuild, the C++ toolset and the Windows SDK.
2. Builds `HaloMapStudioDLL.dll` with MSBuild from `native\HaloMapStudioNative.sln`
   (Release|x64) into `native\x64\Release\`.
3. Runs `cargo build --release --locked -p hms-app` (output `target\release\hms-app.exe`;
   the DLL is copied beside it too).
4. Assembles `dist\windows-x64\` with the same files as the release zip.

| Option | Effect |
| --- | --- |
| `-Package` | also write `dist\release\HaloMapStudio-<version>-windows-x64.zip` and a `.sha256` |
| `-SkipNative` | reuse an existing `native\x64\Release\HaloMapStudioDLL.dll` |
| `-Debug` | debug build (stays in `target\debug\`; `dist\` is not assembled) |
| `-Clean` | delete previous output first (`native\x64`, `native\staging`, `dist\windows-x64`, `target\`) |
| `-Verbose` | print the MSBuild / cargo command lines and verbose output |
| `-Help` | show the built-in help |

The script only writes inside the repository folder and downloads nothing itself.

### What you get

```
dist\windows-x64\
  hms-app.exe              the application
  HaloMapStudioDLL.dll     the native map parser (must stay next to hms-app.exe)
  README.md  LICENSE.md  COMMERCIAL-LICENSE.md  THIRD-PARTY-NOTICES.md
```

This is the same file set as the release archive. The default build is the offline viewer and
editor; the optional `injection` cargo feature (attaching to a running game) is not enabled.

### Run it and point it at your game

```powershell
.\dist\windows-x64\hms-app.exe
```

HMS finds an MCC installed through Steam (including other Steam library drives and Workshop
maps) by itself; Microsoft Store / Xbox installs in the default folders are found when no Steam
copy exists. Install Halo: Reach (and Halo 4 / Halo 3 / Halo 2 Anniversary if you want those)
in MCC first. If your game is somewhere unusual, set `HMS_MCC_DIRS` to the MCC folder (or a
`haloreach` / `maps` folder) in the terminal before starting the app, for example
`$env:HMS_MCC_DIRS = 'D:\Games\Halo The Master Chief Collection'`; the FAQ page describes this
and `hms-app --dump-maps` shows what was found. The app reads those files in place and never
copies or modifies them.

A GPU with DirectX 12 or Vulkan support is required (wgpu picks one automatically).

### Troubleshooting

* **"Visual Studio ... was not found" / `link.exe` not found.** The C++ workload is missing.
  Run the `winget install Microsoft.VisualStudio.2022.BuildTools ...` command above, or open
  the Visual Studio Installer, choose Modify and tick "Desktop development with C++".
  `link.exe: command not found` or `linking with link.exe failed` from cargo means the same
  thing, or that a different `link.exe` (for example from Git for Windows' `usr\bin`) is
  first on `PATH`; run the build from "x64 Native Tools Command Prompt for VS 2022" or the
  "Developer PowerShell for VS 2022" to put the right one first.
* **"cannot open input file 'kernel32.lib'" / missing `windows.h`.** The Windows SDK is not
  installed: add "Windows 11 SDK" (or Windows 10 SDK) in the Visual Studio Installer.
* **The Rust host is `x86_64-pc-windows-gnu`.** Switch to MSVC:
  `rustup set default-host x86_64-pc-windows-msvc` then
  `rustup toolchain install stable-x86_64-pc-windows-msvc`.
* **`error: the lock file ... needs to be updated but --locked was passed`.** You changed a
  `Cargo.toml` or the checkout is damaged. `Cargo.lock` is committed and the build uses it
  as-is for reproducibility.
* **"path too long" / files missing after cloning.** `git config --global core.longpaths true`,
  clone again, and keep the folder near the drive root (for example `C:\src\hms`). Also enable
  Windows long paths if you can.
* **"No such file ... native\..." / the checkout looks incomplete.** Clone with git rather
  than downloading parts; nothing is a submodule.
* **Antivirus deletes or blocks `hms-app.exe` / the DLL or the build is very slow.** Freshly
  linked, unsigned executables are sometimes quarantined. Exclude the repository folder (or
  `target\` and `dist\`) from real-time scanning while building, and restore the file if it was
  quarantined. The binaries are unsigned.
* **`LNK1104: cannot open file ... HaloMapStudioDLL.dll` / copy fails.** A running `hms-app.exe`
  (or MCC, if the DLL was ever injected) holds the file. Close it and rebuild.
* **The app starts but says the native library is unavailable.** `HaloMapStudioDLL.dll` is not
  in the same folder as `hms-app.exe`, or Windows blocks it: right-click, Properties, Unblock,
  if you copied it from a downloaded archive. The dll uses the Visual C++ runtime that the Build
  Tools installed; on another PC install the "Microsoft Visual C++ Redistributable (x64)".
* **No maps listed.** See the first entry of the FAQ page (`docs\faq.html`): MCC must be
  installed with the Halo: Reach content.
* **Black or no window.** Update your GPU driver; DirectX 12 or Vulkan is required.

### Status of this procedure

The script and this page were written on Linux. Every path, project file and MSBuild property
they use was checked against the repository there, and the Rust side was verified with a
clean `cargo xwin check --locked --target x86_64-pc-windows-msvc -p hms-app`; the PowerShell
script itself has not yet been run on a Windows machine. If a step fails on yours, please open
an issue with the script output.

## Linux

Requirements: stable Rust (rustup), a C++20 compiler (g++ or clang++), `libdeflate` development
files for the native parser, a Vulkan-capable GPU.

```sh
./build-all.sh linux     # native/build-linux.sh + cargo build --release -p hms-app
```

Output: `target/release/hms-app` next to `libhalomapstudio.so` (keep them together). After
rebuilding only the native library use `native/install-linux.sh`; it swaps the file atomically
because a running app has it mapped. `./build-all.sh --debug ...` builds the debug profile and
`./build-all.sh package` assembles `dist/linux-x64` and the release archives.

### Cross-building the Windows release from Linux (maintainers)

`./build-all.sh windows` needs [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin)
(`cargo install cargo-xwin`; it downloads the MSVC CRT and Windows SDK into
`~/.cache/cargo-xwin`) plus `clang-cl` and `lld-link` from LLVM for the native DLL
(`native/build-windows.sh`). This is how the official Windows release is produced. Windows
users building for themselves should use `build-windows.ps1` above; the two routes build the
same sources, but the MSBuild route links the C runtime dynamically (`/MD`) while the
cross-build links it statically (`/MT`).

## Tests

```sh
cargo test --release -p hms-app        # or --profile quick while iterating
```
