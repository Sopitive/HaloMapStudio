// dllmain.cpp (HaloMapStudioDLL)
// =============================================================================
// DLL entry point.
//
// The DLL has two roles. Loaded by hms-app (Linux .so / Windows .dll) it is
// a pure map parser: the ZH_* exports decode .map files and nothing below
// runs against a game. Injected into MCC-Win64-Shipping.exe (an optional,
// Windows-only workflow behind hms-app's `injection` cargo feature; never
// done by default) it additionally hooks the engine frame pump and serves
// the live-game MMFs.
//
// Tasks on DLL_PROCESS_ATTACH:
//   1. Init logging (Logging.cpp).
//   2. Spawn a brief installer thread that waits for haloreach.dll to load,
//      then calls FramePumpHook_Install() - installs ONE MinHook patch on
//      `haloreach.dll + kRva_FramePumpHook` (a callee of the engine's frame
//      pump, see EngineThreadResolver.h). Loaded outside MCC the wait simply
//      never resolves and the hook is never installed. All snapshot
//      ticks fire from inside that hook, on the engine thread, so the
//      object-table TLS resolver sees a valid descriptor.
//
// Tasks on DLL_PROCESS_DETACH:
//   1. Tear down the frame-pump hook.
//   2. Close the log file.
//
// Why a tiny installer thread instead of inline install at attach:
//   We don't want DllMain to block on a "wait for haloreach.dll" loop - 
//   that risks deadlocking against the host's loader lock. Spinning the
//   wait off to a thread means DllMain returns immediately, and our hook
//   installs as soon as the engine module appears.
//
// The disk parsers (ZH_MBP_*, ZH_MMP_*, ZH_BSP_*) need no setup - they're
// pure exports the viewer calls when it wants to decode a .map file.
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <atomic>

extern "C" void ZH_Logf(const char* fmt, ...);
extern "C" void ZH_Logf_Shutdown();

extern "C" bool FramePumpHook_Install();
extern "C" void FramePumpHook_Remove();

extern "C" bool ForgeBudgetBypass_Install();
extern "C" void ForgeBudgetBypass_Remove();

extern "C" bool ForgeBarrierBypass_Install();

namespace {

HMODULE g_SelfModule = nullptr;
HANDLE  g_InstallThread = nullptr;
std::atomic<bool> g_DetachRequested{ false };

DWORD WINAPI InstallProc(LPVOID)
{
    // Brief settle to let MCC's own loader finish initialising itself - 
    // matches the timing the main DLL waits for its hook installs.
    if (g_DetachRequested.load(std::memory_order_relaxed)) return 0;
    Sleep(500);
    if (g_DetachRequested.load(std::memory_order_relaxed)) return 0;

    bool ok = FramePumpHook_Install();
    if (!ok) {
        ZH_Logf("[HaloMapStudioDLL] FramePumpHook_Install returned false; ticks will not run\n");
    }

    // Forge budget/max-count bypass - independent of the frame pump (no
    // ticks of its own). Best-effort: if haloreach.dll isn't loaded yet
    // the install logs and skips; the frame-pump tick re-checks each
    // frame and reinstalls on byte drift, so a deferred map load still
    // ends up with the bypass active.
    if (!ForgeBudgetBypass_Install()) {
        ZH_Logf("[HaloMapStudioDLL] ForgeBudgetBypass_Install returned false; "
                "caps still enforced this session\n");
    }

    // KILL_VOLUME_RE_V2: install the MinHook detour on
    // haloreach.dll's kill_volume begin-tick reset so the user-toggle
    // bypass actually disables kill volumes for real (the previous
    // TLS-stamp implementation was hitting the wrong address).
    if (!ForgeBarrierBypass_Install()) {
        ZH_Logf("[HaloMapStudioDLL] ForgeBarrierBypass_Install returned false; "
                "kill-volume bypass inactive this session\n");
    }

    return ok ? 0 : 1;
}

} // namespace

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        g_SelfModule = hModule;
        DisableThreadLibraryCalls(hModule);
        ZH_Logf("[HaloMapStudioDLL] DLL_PROCESS_ATTACH\n");
        g_DetachRequested.store(false, std::memory_order_release);
        g_InstallThread = CreateThread(nullptr, 0, InstallProc, nullptr, 0, nullptr);
        break;

    case DLL_PROCESS_DETACH:
        ZH_Logf("[HaloMapStudioDLL] DLL_PROCESS_DETACH\n");
        g_DetachRequested.store(true, std::memory_order_release);
        if (g_InstallThread) {
            // Bounded wait - installer is a fire-and-forget; we don't
            // need it to finish to tear the hook down.
            WaitForSingleObject(g_InstallThread, 200);
            CloseHandle(g_InstallThread);
            g_InstallThread = nullptr;
        }
        ForgeBudgetBypass_Remove();
        FramePumpHook_Remove();
        ZH_Logf_Shutdown();
        g_SelfModule = nullptr;
        break;
    }
    return TRUE;
}
