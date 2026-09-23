#!/usr/bin/env bash
# Build HaloMapStudioDLL.dll (x64 Windows) on Linux with clang-cl + lld-link against the xwin MSVC/SDK cache
# (~/.cache/cargo-xwin/xwin). Same TU list/defines/def file as the MSVC vcxproj (native/build.ps1 on Windows).
set -euo pipefail
cd "$(dirname "$0")"
X=${XWIN:-$HOME/.cache/cargo-xwin/xwin}
OUT=build-windows; mkdir -p "$OUT/obj"
SRC=MccMapStudioDLL
CXX="clang-cl --target=x86_64-pc-windows-msvc /nologo /c /EHsc /std:c++17 /O2 /MT /DNDEBUG /DMCCMAPSTUDIODLL_EXPORTS /D_WINDOWS /D_USRDLL /DWIN32_LEAN_AND_MEAN -Wno-everything /I$SRC /Ipackages/minhook.1.3.3/lib/native/include /Ipackages/minhook.1.3.3/lib/native/src /imsvc $X/crt/include /imsvc $X/sdk/include/ucrt /imsvc $X/sdk/include/um /imsvc $X/sdk/include/shared /imsvc $X/sdk/include/winrt"
CC="clang-cl --target=x86_64-pc-windows-msvc /nologo /c /O2 /MT /DNDEBUG /D_WINDOWS -Wno-everything /I$SRC /Ipackages/minhook.1.3.3/lib/native/include /Ipackages/minhook.1.3.3/lib/native/src /imsvc $X/crt/include /imsvc $X/sdk/include/ucrt /imsvc $X/sdk/include/um /imsvc $X/sdk/include/shared"
CPP=(pch dllmain Logging HostByteHelper FramePumpHook MapCacheCommon MapCacheUnload MapBitmapParser MapModelParser MapBspParser SkyWalker CollisionModelWalker PhysicsModelWalker DecalWalker PreplacedDecalsWalker FogParser PlanarFogWalker LightWalker SimpleLightsWalker OverlayWalker ScenarioExposureWalker LightmapParser MapInfoSnapshot ForgePaletteSnapshot ObjectTableSnapshot PlayerModeSnapshot PoseSnapshot TransformQueueSnapshot WorldSpawnSnapshot ForgeSpawnAndPose SpawnDefinitionLive ForgeObjectTableSnapshot MegaloObjectSnapshot ForgeGtLabelsMmf ForgeObjectEdit ForgeBudgetBypass ForgeBarrierBypass ForgePhysics ScenarioForgePaletteWalker ScenarioObjectWalker ScenarioTriggerVolumeWalker SoftCeilingWalker ChangeColorWalker)
C=(packages/minhook.1.3.3/lib/native/src/buffer.c packages/minhook.1.3.3/lib/native/src/hook.c packages/minhook.1.3.3/lib/native/src/trampoline.c packages/minhook.1.3.3/lib/native/src/hde/hde64.c $SRC/miniz.c $SRC/libdeflate/lib/deflate_decompress.c $SRC/libdeflate/lib/utils.c $SRC/libdeflate/lib/x86/cpu_features.c)
echo ">> compiling ${#CPP[@]} C++ + ${#C[@]} C TUs (clang-cl)"
printf '%s\n' "${CPP[@]}" | xargs -P "$(nproc)" -I{} sh -c "$CXX $SRC/{}.cpp /Fo$OUT/obj/{}.obj"
for c in "${C[@]}"; do n=$(basename "$c" .c); $CC "$c" /Fo"$OUT/obj/c_$n.obj"; done
echo ">> linking HaloMapStudioDLL.dll"
lld-link /nologo /DLL /MACHINE:X64 /DEF:$SRC/HaloMapStudioDLL.def /OUT:$OUT/HaloMapStudioDLL.dll /LIBPATH:$X/crt/lib/x86_64 /LIBPATH:$X/sdk/lib/um/x86_64 /LIBPATH:$X/sdk/lib/ucrt/x86_64 $OUT/obj/*.obj kernel32.lib user32.lib advapi32.lib shell32.lib ole32.lib psapi.lib
ls -la "$OUT/HaloMapStudioDLL.dll"
