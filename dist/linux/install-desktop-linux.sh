#!/usr/bin/env bash
# Install the Halo Map Studio desktop entry + themed icon so Linux (especially Wayland,
# which ignores the window-set icon) shows the real icon in the taskbar / launcher.
# Run it from the extracted release folder; it points the launcher at the hms-app next to it.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bin="$here/hms-app"; [ -x "$bin" ] || bin="$here/../hms-app"; [ -x "$bin" ] || bin="$(command -v hms-app || true)"
[ -x "$bin" ] || { echo "!! hms-app not found next to this script; pass its path: $0 /path/to/hms-app"; [ -n "${1:-}" ] && bin="$1"; }
[ -x "$bin" ] || { echo "!! no hms-app binary"; exit 1; }
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
mkdir -p "$apps" "$icons/scalable/apps"
# scalable SVG icon (best), plus a 256 PNG fallback for themes that ignore SVG
install -Dm644 "$here/halo-map-studio.svg" "$icons/scalable/apps/halo-map-studio.svg"
if command -v magick >/dev/null || command -v convert >/dev/null; then
  conv() { command -v magick >/dev/null && magick "$@" || convert "$@"; }
  for s in 48 64 128 256; do mkdir -p "$icons/${s}x${s}/apps"; conv "$here/halo-map-studio.png" -resize ${s}x${s} "$icons/${s}x${s}/apps/halo-map-studio.png"; done
else
  mkdir -p "$icons/512x512/apps"; install -Dm644 "$here/halo-map-studio.png" "$icons/512x512/apps/halo-map-studio.png"
fi
sed "s|@EXEC@|$bin|" "$here/halo-map-studio.desktop" > "$apps/halo-map-studio.desktop"
chmod +x "$apps/halo-map-studio.desktop" 2>/dev/null || true
command -v update-desktop-database >/dev/null && update-desktop-database "$apps" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -f -t "$icons" 2>/dev/null || true
echo "installed: $apps/halo-map-studio.desktop  ->  $bin"
echo "icon: $icons/scalable/apps/halo-map-studio.svg"
echo "Log out/in (or restart the shell) if the taskbar icon does not refresh immediately."
