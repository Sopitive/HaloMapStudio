// MapCacheUnload.cpp
// =============================================================================
// Hot-reload support: lets the host call into us right before it
// FreeLibrary's the DLL, so we can:
//   1. Mark a "shutting down" atomic so any running tick path bails fast.
//   2. Drain every parser handle table (cache / model / bsp), unmapping the
//      memory-mapped .map files and freeing per-handle resource buffers.
//   3. Best-effort uninstall the FramePumpHook if it ever managed to install
//      itself in this process. (When loaded into the viewer process via FFI
//      the hook never installs because haloreach.dll isn't present; the call
//      is a cheap no-op in that case. When this DLL is also injected into MCC
//      it would be installed - we still leave that copy alone since it's a
//      separate module instance.)
//
// Returns when it is safe for the caller to FreeLibrary us. Callers must
// then drop every cached function pointer before doing the actual unload.
//
// IMPORTANT: After PrepareUnload, every handle the viewer held (cache /
// model / bsp / bitmap) is invalid. The contract is: hot reload invalidates
// all native state and the viewer must re-open whatever it needs. The viewer
// honours this by re-opening the currently-displayed map after a successful
// reload.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <atomic>

extern "C" void ZH_Logf(const char* fmt, ...);
extern "C" void FramePumpHook_Remove();
extern "C" void ForgeBudgetBypass_Remove();

// Defined in MapModelParser.cpp / MapBspParser.cpp. File-scope draining
// helpers so the per-file anonymous-namespace handle tables can be cleared
// from a different translation unit.
extern "C" void MapModelParser_DrainAllHandles();
extern "C" void MapBspParser_DrainAllHandles();

namespace {
std::atomic<bool> g_ShuttingDown{ false };
}

extern "C" __declspec(dllexport) void __stdcall ZH_MMP_PrepareUnload()
{
    // Idempotent - multiple PrepareUnload calls are a no-op.
    bool prev = g_ShuttingDown.exchange(true, std::memory_order_acq_rel);
    if (prev) {
        ZH_Logf("[HaloMapStudioDLL] ZH_MMP_PrepareUnload: already shutting down\n");
        return;
    }

    ZH_Logf("[HaloMapStudioDLL] ZH_MMP_PrepareUnload: begin\n");

    // 1. Drain every model handle (frees per-model resource buffers).
    __try { MapModelParser_DrainAllHandles(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // 2. Drain every BSP handle (frees per-BSP resource buffers).
    __try { MapBspParser_DrainAllHandles(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // 3. Drain every cache handle. This unmaps the .map files and tears down
    //    any lazily-opened shared sibling caches.
    __try { zh_mcc::DrainAllCacheHandles(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // 4. Best-effort: tear down the frame-pump hook if it managed to install
    //    itself in this process. In the viewer process (the viewer) the install
    //    never succeeds because haloreach.dll isn't loaded, so this is a
    //    no-op there. In MCC.exe (the injected copy) it's the right call - 
    //    but the injected copy is a separate module instance and won't have
    //    this PrepareUnload invoked against it.
    __try { FramePumpHook_Remove(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // 5. Detach the forge budget/cap bypass hooks too, so NO detour of ours is
    //    live in the game when the caller FreeLibrary's us (a dangling detour into
    //    freed DLL memory would crash the game). DETACH also does this, but doing it
    //    here - before FreeLibrary - closes the window where a game thread could be
    //    inside one of these detours mid-unload.
    __try { ForgeBudgetBypass_Remove(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    ZH_Logf("[HaloMapStudioDLL] ZH_MMP_PrepareUnload: done - safe to FreeLibrary\n");
}
