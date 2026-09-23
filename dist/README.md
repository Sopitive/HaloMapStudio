# Halo Map Studio

A standalone map viewer and Forge editor for Halo: Reach and Halo 4 (with
early view-only Halo 2 Anniversary support) from the Master Chief Collection
on PC. It reads your installed game's map files directly and renders
them with the engine's own lighting, materials and effects, and lets you open
and edit Forge map variants (`.mvar`).

## Requirements

* Halo: The Master Chief Collection installed via Steam (Halo: Reach content).
  The app finds the game folder automatically; Steam Workshop maps are listed
  separately from the built-in ones.
* A GPU with Vulkan (Linux) or DirectX 12 / Vulkan (Windows) support.
* You must own the game. No game content is included in this download.

## Running

* **Windows:** unzip and run `hms-app.exe`. Keep `HaloMapStudioDLL.dll` next
  to it (it is the map parser).
* **Linux:** unzip and run `./hms-app`. Keep `libhalomapstudio.so` next to it.
  Steam/Proton installs of MCC are detected; use `HMS_MCC_DIRS=<path>` if your
  library is somewhere unusual.

Settings and projects are stored in your user profile (Windows
`%LOCALAPPDATA%\HaloMapStudio`, Linux `$XDG_DATA_HOME/HaloMapStudio` or
`~/.local/share/HaloMapStudio`).

## Halo 4

Halo 4 maps (from `halo4\maps` and Halo 4 Workshop items) appear in their own
"Halo 4" group of the map picker: BSP geometry with textures, baked lightmaps
and sun, scenario objects, vehicles with their attachments, and Forge map
variants (`.mvar`) whose objects can be selected, moved, rotated, placed from
the map's palette, edited in the Object panel and **saved**. Saving is confirmed
in the game: the encoder reproduces every shipped file byte-for-byte and an
HMS-saved variant loads in MCC, so **File > Save** writes the open file in place
and **Save As** writes a new one. Team change-colours are applied from the
multiplayer team palette. Some material detail (normal and specular maps) is not
finished, and the map's own scenario objects render but are not yet selectable.

## Halo 2 Anniversary

Halo 2 Anniversary (`groundhog`) maps appear in their own group and render with
their geometry, textures, baked lightmaps and post-processing, through the same
reader as Halo 4. This support is **early and view-only** for now: the map
variants use a format that is not yet decoded, so Forge editing is Halo 4 and
Reach only.

## Documentation

The user guide (getting started, editing, shortcuts, scripting reference, headless
rendering, Halo 4 and Halo 2 Anniversary status) is published
via GitHub Pages: https://sopitive.github.io/HaloMapStudio/

## Source code

The source code is published at https://github.com/Sopitive/HaloMapStudio
under the same license as these binaries (see below).

## License

This software is licensed under the **PolyForm Noncommercial License 1.0.0**
(see `LICENSE.md`): free for personal, hobby, educational and other
noncommercial use. **Any commercial use requires a separate commercial
license** from the copyright holder; see `COMMERCIAL-LICENSE.md` for what
counts as commercial use and how to obtain one. Third-party components bundled
in this software keep their own licenses (`THIRD-PARTY-NOTICES.md`).

Halo, Halo: Reach, Halo 4, Halo 2: Anniversary and The Master Chief Collection
are trademarks of Microsoft Corporation. This project is not affiliated with or endorsed by Microsoft or
343 Industries.

Copyright (c) 2026 Sopitive. All rights reserved.
