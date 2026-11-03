<img width="1887" height="1165" alt="viewer_last" src="https://github.com/user-attachments/assets/e427bdb1-3a5a-4a35-bd79-cb42bdace61b" />
<img width="1887" height="1165" alt="viewer_last" src="https://github.com/user-attachments/assets/cb9b79ce-bac4-49ad-9cdf-d44ffa922688" />
<img width="1887" height="1165" alt="viewer_last" src="https://github.com/user-attachments/assets/f5365c0a-e5d4-4f52-b7cf-0292fc9f80b8" />


# Halo Map Studio

**Current release: v1.8.0** — [downloads](https://github.com/Sopitive/HaloMapStudio/releases/latest) · [changelog](https://sopitive.github.io/HaloMapStudio/changelog.html). The app can check for new releases itself (Help ▸ Software update…; opt-in, see the [user guide](https://sopitive.github.io/HaloMapStudio/getting-started.html#updating)). Release archives are named `HaloMapStudio-<version>-linux-x64.zip` / `HaloMapStudio-<version>-windows-x64.zip` (see `RELEASING.md`).

A standalone map viewer and Forge editor for **Halo: Reach**, **Halo 3**,
**Halo 4** and **Halo 2 Anniversary** from *Halo: The Master Chief Collection*
on PC. It also plays back **Reach saved films** —
see [Theater](https://sopitive.github.io/HaloMapStudio/theater.html). It reads the game's own `.map` cache files, renders
them with the engine's lighting, materials and effects reverse-engineered from
the shipped shaders, and opens, edits and saves Forge map variants (`.mvar`) —
no running game required.

The user guide — the editor, keyboard shortcuts, the scripting language, the
headless / batch modes and the environment variables — is published at
**https://sopitive.github.io/HaloMapStudio/**. More screenshots are in
[`docs/assets/`](docs/assets/) (Forge World, Countdown, Zealot, Boardwalk,
Halo 4 Ravine / Haven).

## Requirements

* Halo: The Master Chief Collection (Steam) with Halo: Reach installed. The
  app locates the game folder automatically (Steam and Proton libraries;
  `HMS_MCC_DIRS=<path>` overrides). **No game content is included** — you must
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
| `docs/` | User guide (GitHub Pages) |
| `dist/README.md` | README shipped inside the binary release archives |

## Building

Full, ordered instructions for **Windows (native, one command)** and Linux are in
[BUILDING.md](BUILDING.md). In short:

```powershell
.\build-windows.ps1 [-Package]   # Windows: VS Build Tools "Desktop development with C++" + rustup (MSVC)
```

```sh
./build-all.sh linux             # Linux: native/build-linux.sh + cargo build --release -p hms-app
```

Output: `dist\windows-x64\hms-app.exe` next to `HaloMapStudioDLL.dll` (Windows) or
`target/release/hms-app` next to `libhalomapstudio.so` (Linux). The native library is loaded at
runtime from the executable's directory, so keep the two files together. No game content is
built or downloaded; HMS reads your own MCC install.

## Running

```sh
target/release/hms-app                # interactive
HMS_SHOT=out.png HMS_MAP=<path>.map target/release/hms-app   # headless render
```

Tests: `cargo test --release -p hms-app`.

## License

Halo Map Studio is licensed under the **PolyForm Noncommercial License 1.0.0**
(`LICENSE.md`): free for personal, hobby, educational and other noncommercial
use. **Commercial use of any kind requires a separate commercial license** from
the copyright holder — see `COMMERCIAL-LICENSE.md`.

No Halo game content (maps, tags, textures, shaders) is distributed with this
software; it reads the user's own installed copy of the game. Halo, Halo: Reach,
Halo 4, Halo 2: Anniversary and The Master Chief Collection are trademarks of
Microsoft Corporation. This project is not affiliated with or endorsed by
Microsoft or 343 Industries.

Copyright © 2026 Sopitive.

## Third-party software

The native parser vendors **MinHook** (BSD 2-Clause), **libdeflate** (MIT) and
**miniz** (MIT). The Rust crates build on **wgpu**, **egui / eframe**, **glam**,
**rayon**, **serde**, **png**, **half**, **bytemuck**, **rfd**, **memmap2**,
**miniz_oxide**, **libloading** and the **windows** crate, among others — all
under MIT / Apache-2.0 / BSD / Zlib-style licenses. Full texts and the complete
list are in `THIRD-PARTY-NOTICES.md`. Tag layouts were cross-checked against
the open-source **Reclaimer** and **Assembly** tag definitions.
