#!/usr/bin/env bash
# Build HaloMapStudio's map/mvar parsers as a Linux shared library.
#
# Same sources as the Windows build -- the only difference is this script's
# include path (native/compat, which supplies a <windows.h> shim over POSIX) and
# the translation-unit list below. The live-game injection TUs (Forge*,
# FramePump*, *Snapshot, dllmain) are excluded: they hook a running MCC, which
# is a Windows process, so they only exist in the Windows DLL.
set -euo pipefail
cd "$(dirname "$0")"

SRC=MccMapStudioDLL
OUT=build-linux
mkdir -p "$OUT/obj"

CXXFLAGS="-std=c++20 -fPIC -O2 -w -fvisibility=hidden -Icompat -I$SRC"
CFLAGS="-fPIC -O2 -w"

# Offline parsing / decode only.
TUS=(
  MapCacheCommon MapCacheUnload MapBspParser MapModelParser MapBitmapParser
  LightmapParser Logging
  DecalWalker PreplacedDecalsWalker CollisionModelWalker PhysicsModelWalker
  LightWalker SimpleLightsWalker FogParser PlanarFogWalker
  OverlayWalker SkyWalker ChangeColorWalker ScenarioObjectWalker
  ScenarioExposureWalker ScenarioForgePaletteWalker ScenarioTriggerVolumeWalker
  SoftCeilingWalker
)

echo ">> compiling ${#TUS[@]} parser translation units"
OBJS=()
for t in "${TUS[@]}"; do
  g++ $CXXFLAGS -c "$SRC/$t.cpp" -o "$OUT/obj/$t.o"
  OBJS+=("$OUT/obj/$t.o")
done

echo ">> compiling compat layer"
for c in hms_win_compat hms_linux_stubs; do
  g++ $CXXFLAGS -c "compat/$c.cpp" -o "$OUT/obj/$c.o"
  OBJS+=("$OUT/obj/$c.o")
done

echo ">> compiling miniz"
gcc $CFLAGS -c "$SRC/miniz.c" -o "$OUT/obj/miniz.o"
OBJS+=("$OUT/obj/miniz.o")

echo ">> linking libhalomapstudio.so"
# --no-undefined so a missing symbol fails the build instead of at dlopen time.
g++ -shared -Wl,--no-undefined -o "$OUT/libhalomapstudio.so" "${OBJS[@]}" \
    -ldeflate -lm -ldl -lpthread

echo ">> $(nm -D --defined-only "$OUT/libhalomapstudio.so" | grep -c ' T ZH_') ZH_* exports"
ls -la "$OUT/libhalomapstudio.so"
