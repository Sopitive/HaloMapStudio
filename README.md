# Halo Map Studio

A standalone map viewer and Forge editor for **Halo: Reach** and **Halo 4**
(with early, view-only **Halo 2 Anniversary** support) from *Halo: The Master Chief Collection* on PC. It reads the game's
own `.map` cache files, renders them with the engine's lighting, materials and
effects reverse-engineered from the shipped shaders, and opens, edits and saves
Forge map variants (`.mvar`) - no running game required.

Screenshots: [`site/assets/`](site/assets/) (Forge World, Countdown, Zealot,
Boardwalk, Halo 4 Ravine / Haven). The user guide is published from the same
`site/` folder: **https://sopitive.github.io/HaloMapStudio/**

## Requirements

* Halo: The Master Chief Collection (Steam) with Halo: Reach installed. The
  app locates the game folder automatically (Steam and Proton libraries;
  `HMS_MCC_DIRS=<path>` overrides). **No game content is included** - you must
  own the game.
* A GPU with Vulkan (Linux) or DirectX 12 / Vulkan (Windows). A software
  rasteriser is used as a last resort.

## Repository layout

| Path | What |
|---|---|
| `crates/hms-app` | The application: egui/eframe UI, scene, Forge editor, scripting, headless renders |
| `crates/hms-render` | wgpu renderer and WGSL shaders |
| `crates/hms-native` | FFI to the C++ map parser |
| `crates/hms-ipc`, `crates/hms-inject` | Optional Windows-only live-game bridge (`injection` feature; not built by default) |
| `crates/hms-mcp` | Model Context Protocol bridge to the app's script server |
| `native/` | The C++ map parser (`libhalomapstudio.so` / `HaloMapStudioDLL.dll`): BSP, render-model, bitmap, lightmap and scenario tag decoders |
| `site/` | User guide (GitHub Pages) |
| `dist/README.md` | README shipped inside the binary release archives |

## Building

Prerequisites: stable Rust (`rust-toolchain.toml` pins the channel and adds the
`x86_64-pc-windows-msvc` target), a C++20 compiler and `libdeflate` for the
native parser on Linux.

```sh
./build-all.sh linux     # native/build-linux.sh + cargo build --release -p hms-app
./build-all.sh windows   # cross-build the Windows release from Linux (see below)
./build-all.sh           # both
```

Output: `target/release/hms-app` next to `libhalomapstudio.so`, and
`target/x86_64-pc-windows-msvc/release/hms-app.exe` next to
`HaloMapStudioDLL.dll`. `build-all.sh --debug ...` builds the debug profile.

**Windows cross-build from Linux** needs [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin)
(`cargo install cargo-xwin`; it downloads the MSVC CRT / Windows SDK into
`~/.cache/cargo-xwin/xwin`) plus `clang-cl` and `lld-link` from LLVM for the
native DLL (`native/build-windows.sh`).

**Building on Windows:** `cargo build --release -p hms-app` and
`pwsh native\build.ps1` (MSBuild, `native/HaloMapStudioNative.sln`), which
stages `HaloMapStudioDLL.dll` into `target\release\`.

The native library is loaded at runtime from the executable's directory, so
keep the two files together. After rebuilding only the native library on Linux
use `native/install-linux.sh` (it swaps the file atomically; a running app has
it mapped).

## Running

```sh
target/release/hms-app                # interactive
HMS_SHOT=out.png HMS_MAP=<path>.map target/release/hms-app   # headless render
```

The guide covers the editor, keyboard shortcuts, the scripting language, the
headless / batch modes and the environment variables:
https://sopitive.github.io/HaloMapStudio/

Tests: `cargo test --release -p hms-app`.

## License

Halo Map Studio is licensed under the **PolyForm Noncommercial License 1.0.0**
(`LICENSE.md`): free for personal, hobby, educational and other noncommercial
use. **Commercial use of any kind requires a separate commercial license** from
the copyright holder - see `COMMERCIAL-LICENSE.md`.

No Halo game content (maps, tags, textures, shaders) is distributed with this
software; it reads the user's own installed copy of the game. Halo, Halo: Reach
and The Master Chief Collection are trademarks of Microsoft Corporation. This
project is not affiliated with or endorsed by Microsoft or 343 Industries.

## Third-party software

The native parser vendors **MinHook** (BSD 2-Clause), **libdeflate** (MIT) and
**miniz** (MIT). The Rust crates build on **wgpu**, **egui / eframe**, **glam**,
**rayon**, **serde**, **png**, **half**, **bytemuck**, **rfd**, **memmap2**,
**miniz_oxide**, **libloading** and the **windows** crate, among others - all
under MIT / Apache-2.0 / BSD / Zlib-style licenses. Full texts and the complete
list are in `THIRD-PARTY-NOTICES.md`. Tag layouts were cross-checked against
the open-source **Reclaimer** and **Assembly** tag definitions.
