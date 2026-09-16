<img width="1887" height="1138" alt="viewer_last" src="https://github.com/user-attachments/assets/4ea79701-e559-43d0-9ad9-4256da8b6f55" />
<img width="1887" height="1138" alt="viewer_last" src="https://github.com/user-attachments/assets/7b94d894-8bf3-4438-ad72-2b25532e503f" />

# Halo Map Studio

A standalone map viewer and Forge editor for Halo: Reach (Master Chief
Collection, PC). It reads your installed game's map files directly and renders
them with the engine's own lighting, materials and effects, lets you open and
edit Forge map variants (`.mvar`) for placing
objects.

## Requirements

* Halo: The Master Chief Collection installed via Steam (Halo: Reach content).
  The app finds the game folder automatically; Steam Workshop maps are listed
  separately from the built-in ones.
* A GPU with Vulkan (Linux) or DirectX 12 / Vulkan (Windows) support.
* You must own the game. No game content is included in this download.

## Running

* **Windows:** unzip and run `hms-app.exe`. Keep `HaloMapStudioDLL.dll` next
  to it (it is the map parser and the game hook).
* **Linux:** unzip and run `./hms-app`. Keep `libhalomapstudio.so` next to it.
  Steam/Proton installs of MCC are detected; use `HMS_MCC_DIRS=<path>` if your
  library is somewhere unusual.

Settings and projects are stored in your user profile (Windows
`%LOCALAPPDATA%`, Linux XDG config directory).

## Source code

**Source code is coming soon.** The project is still under heavy development
and the full source will be published once the renderer and the Forge editing
workflow are more or less finished. Until then only these binary builds are
distributed.

## Documentation
You can find all relevent documentation [here](https://sopitive.github.io/HaloMapStudio)

## License

This software is licensed under the **PolyForm Noncommercial License 1.0.0**
(see `LICENSE.md`): free for personal, hobby, educational and other
noncommercial use. **Any commercial use requires a separate commercial
license** from the copyright holder; see `COMMERCIAL-LICENSE.md` for what
counts as commercial use and how to obtain one. Third-party components bundled
in this software keep their own licenses (`THIRD-PARTY-NOTICES.md`).

Halo, Halo: Reach and The Master Chief Collection are trademarks of Microsoft
Corporation. This project is not affiliated with or endorsed by Microsoft or
343 Industries.

Copyright (c) 2026 Sopitive. All rights reserved.
