# Halo Map Studio 1.4.0

A standalone map viewer and Forge editor for Halo: Reach, Halo 3, Halo 4 and
Halo 2 Anniversary from the Master Chief Collection on PC. It reads your installed game's map files directly and renders
them with the engine's own lighting, materials and effects, and lets you open
and edit Forge map variants (`.mvar`). It also plays back Halo: Reach saved
films, with per-player statistics you can pull from the command line.

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

## Updating

The app checks GitHub Releases for a newer version about once a day (turn it
off under Settings ▸ Updates) and shows a small "Update available" pill next
to the Help menu when there is one — no pop-ups. Help ▸ Software update…
shows the release notes, downloads the archive for your platform, verifies it
and installs it: the app saves your session (map, variant, unsaved variant
edits, camera) to a recovery file, restarts itself and restores the session.
The previous version is kept in `.hms-backup` next to the app until the next
start.

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
reader as Halo 4. Map variants open, edit and save through the same editor as
Halo 4; HMS reproduces 283 of the 289 variants the game ships byte-for-byte.

## Documentation

The user guide (getting started, editing, shortcuts, scripting reference, headless
rendering, Halo 4 and Halo 2 Anniversary status) is published from the `docs/` folder of this repository
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
