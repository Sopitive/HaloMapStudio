// FramePumpHook.cpp (HaloMapStudioDLL)
// =============================================================================
// MinHook frame-pump install (Windows, injected-into-MCC role only).
//
// Why the ticks must run inside the engine's frame pump:
//   A "worker thread polls every 33 ms" approach runs the snapshot ticks on
//   a thread the engine never populated TLS on. ObjectTableSnapshot's
//   resolver needs `*(haloreach_tls + 0x10)` - that read returns null on a
//   non-engine thread, as does any field the engine's own object accessors
//   touch via TLS, so every per-tick read silently fails (zero objects /
//   zero markers).
//
// What this file does:
//   * Installs ONE MinHook patch on `haloreach.dll + kRva_FramePumpHook`.
//     This is ReplicationContext_SendStateUpdate (0x2A90F4), a no-arg CALLEE
//     of the engine's per-frame pump - NOT the shared pump entry 0x2AA92C that
//     other trainers hook. Hooking a callee lets trainers coexist AND keeps
//     the ticks inside the pump's descriptor-coherent window. See
//     EngineThreadResolver.h::kRva_FramePumpHook for the full story.
//   * The hook calls the original first, then runs each viewer
//     `_FramePumpTick()` inside an SEH guard. Each tick is now on the
//     engine thread, so ObjectTableSnapshot/PlayerModeSnapshot's TLS-based
//     resolver works correctly.
//   * Bound install - we don't pull in the main DLL's
//     AutonomousHookMonitor or vectored exception handler. If haloreach
//     reloads mid-session and our hook is unhooked, the worst case is the
//     viewer goes stale until the user reloads the DLL - far better than
//     the static "always returns zero" failure we have today.
//
// Public surface:
//   FramePumpHook_Install() - call once from DLL_PROCESS_ATTACH.
//   FramePumpHook_Remove() - call once from DLL_PROCESS_DETACH.
//
// Both are noexcept and idempotent.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"
#include "HaloReachTagHelpers.h"

#include <windows.h>
#include <cstdint>
#include <atomic>

#include "MinHook.h"

extern "C" void ZH_Logf(const char* fmt, ...);

// Tick entry points (defined in the per-snapshot .cpp files).
extern "C" void ObjectTableSnapshot_FramePumpTick();
extern "C" void MapInfoSnapshot_FramePumpTick();
extern "C" void PlayerModeSnapshot_FramePumpTick();
extern "C" void ForgePaletteSnapshot_FramePumpTick();
extern "C" void TransformQueueSnapshot_FramePumpTick();
extern "C" void WorldSpawnSnapshot_FramePumpTick();
extern "C" void ForgeSpawnAndPose_FramePumpTick();
extern "C" void ForgeObjectTableSnapshot_FramePumpTick();
extern "C" void MegaloObjectSnapshot_FramePumpTick();
extern "C" void PoseSnapshot_FramePumpTick();
extern "C" void ForgeObjectEdit_FramePumpTick();

// Forge budget/max-count bypass - declared in ForgeBudgetBypass.cpp. Tick
// here runs an 8-byte fingerprint check against the hooked engine site and
// reinstalls the detour on drift (MCC's map-load anti-tamper restores some
// engine bytes; piggy-backing on the existing pump avoids a second watcher
// thread).
extern "C" void ForgeBudgetBypass_TickCheck();

// Kill volume / soft ceiling bypass - declared in ForgeBarrierBypass.cpp.
// Re-stamps the TLS bitmask every tick when the user has barriers disabled.
extern "C" void HaloMapStudio_ForgeBarrier_Tick();

namespace {

// Hook target: ReplicationContext_SendStateUpdate (kRva_FramePumpHook =
// 0x2A90F4) - a NO-ARG callee of the engine's per-frame pump. We hook a pump
// CALLEE (not the pump itself at 0x2AA92C, which other trainers hook) so
// our ticks run INSIDE the pump's descriptor-coherent window. Signature is
// `int64 __fastcall(void)` - no args, so the detour ABI is trivially correct;
// we forward the return value. See EngineThreadResolver.h::kRva_FramePumpHook.
using FramePump_t = int64_t (__stdcall*)();

FramePump_t       g_OrigPump  = nullptr;
void*             g_HookTarget = nullptr;
std::atomic<bool> g_Installed{ false };

// Bytes we expect to see at the target right after MH_EnableHook succeeds.
// MinHook writes a 5-byte JMP rel32 (E9 xx xx xx xx) on x64-near and a
// 14-byte indirect FF 25 [rel32=0] + qword on x64-far. Either way the very
// first byte is what changes from whatever the original instruction's
// opcode was -> 0xE9 / 0xFF. We snapshot 8 bytes for a fingerprint that's
// unlikely to false-match the original engine code if the engine ever
// happens to overwrite back to a JMP-shaped sequence; 8 bytes covers the
// rel32 + a few opcode bytes after it.
uint8_t g_HookFingerprint[8] = {};

// Map-load can restore the original engine bytes at our target without
// unloading the DLL - bytes-comparison is the only reliable detector for
// this case (module-base check fails because the base is unchanged).
HANDLE        g_WatcherThread = nullptr;
HANDLE        g_WatcherStop   = nullptr;

// Guard against tick re-entry if the engine ever calls the pump from inside
// another pump (it doesn't, but a defensive flag is cheap).
thread_local bool tls_InTick = false;

void RunSnapshotTicks()
{
    if (tls_InTick) return;
    tls_InTick = true;

    // STUTTER FIX: this runs SYNCHRONOUSLY on the engine's
    // frame-pump thread every frame, so the game can't finish a frame until all
    // our ticks return. The heavy READ walks below (ObjectTable / ForgeObject
    // Table / Pose) each iterate thousands of object slots and re-publish EVERY
    // frame - that per-frame full-table walk on the engine thread is the
    // constant frame stutter. Stagger them 1-per-frame (round-robin over 3
    // frames ~= 20 Hz @60fps, imperceptible for a static-scene viewer) so each
    // frame pays for at most ONE heavy walk instead of all three. The cheap
    // edit-applier ticks + the internally-paced palette/mapinfo ticks stay
    // every frame.
    static uint32_t s_pumpFrame = 0;
    const uint32_t s_pumpPhase = (s_pumpFrame++) % 3u;

    // ---- Map-change detection prelude ----
    // Probe ResolveObjectPoolDescriptor BEFORE running any snapshot. The
    // resolver bumps the global map epoch when it observes a stale anchor
    // pointer / module reload / bogus descriptor bounds. Doing this up
    // front guarantees ALL downstream snapshot ticks (Forge palette walk,
    // ObjectTable walk, PlayerMode walk, etc.) see the post-bump epoch
    // and reset their caches BEFORE walking the new map's data.
    //
    // The descriptor pointer itself isn't used here - we just want the
    // side-effect of detection. Errors are silently swallowed; a real
    // descriptor read failure shows up as kErr_BulkScanFailed in the
    // ObjectTable snapshot's per-tick error code, plenty for diagnosis.
    static thread_local uint64_t s_LastObservedEpoch = 0;
    __try { (void)HaloMapStudio::Engine::ResolveObjectPoolDescriptor(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
    {
        uint64_t curEpoch = HaloMapStudio::Engine::GetMapEpoch();
        if (curEpoch != s_LastObservedEpoch) {
            // Drop the TLS HrTag cache (used by ForgePaletteSnapshot,
            // ObjectTableSnapshot, PlayerModeSnapshot via
            // ResolveHlmtAndModeDatums_Cached). Per-snapshot caches keyed
            // by datum (the unordered_maps in PlayerMode etc.) reconcile
            // their own epoch tracker - see the matching block in each
            // snapshot's tick. Doing the TLS flush here also covers
            // ForgePaletteSnapshot which runs first.
            ZeroHour::HrTag::ResolveHlmtAndModeDatums_FlushCache();
            s_LastObservedEpoch = curEpoch;
        }
    }

    // CLIENT_OBJ_DIAG: once every ~180 frames, log the static vs TLS
    // object-descriptor active counts so we can see where a client's live objects
    // are. Cheap + throttled; remove once the client object path is wired.
    if ((s_pumpFrame % 180u) == 1u)
        __try { HaloMapStudio::Engine::DiagObjectDescriptors(); }
        __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { MapInfoSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { ForgePaletteSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // Heavy full-table walk - staggered to phase 0 (~20Hz).
    if (s_pumpPhase == 0)
    __try { ObjectTableSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { PlayerModeSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { TransformQueueSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { WorldSpawnSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { ForgeSpawnAndPose_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // Forge-object table walker - must run before the edit handler so the
    // edit handler has a fresh resolved base. Re-published every frame so
    // the viewer's property panel sees external changes (user editing in
    // Forge mode) immediately.
    // Heavy full-table walk (publishes all kMaxObjects slots) - staggered to
    // phase 1 (~20Hz). The edit handler below still runs every frame on the
    // cached resolved base so external edits stay responsive.
    if (s_pumpPhase == 1)
    __try { ForgeObjectTableSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // Megalo-object debug overlay walker. UNCONDITIONAL (not forge-gated) - 
    // megalo objects exist in MP gametypes, which is the whole point. Heavy
    // full-array walk (512 slots) - staggered to phase 1 like the forge table.
    if (s_pumpPhase == 1)
    __try { MegaloObjectSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    __try { ForgeObjectEdit_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // Live biped poses (haloreach's per-frame pose ring -> MMF). Runs late so
    // it sees the freshest engine pose ring this tick. SEH-guarded to fall
    // back to A-pose silently if the pool isn't populated (e.g. menu screen).
    // Heavy pose-ring walk - staggered to phase 2 (~20Hz). Bipeds in a level
    // viewer are near-static, so 20Hz pose updates are imperceptible.
    if (s_pumpPhase == 2)
    __try { PoseSnapshot_FramePumpTick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // Cheap fingerprint check on the budget-bypass detour. Re-patches if
    // MCC's map-load restored the engine bytes. Runs last so a fault here
    // can't poison earlier ticks.
    __try { ForgeBudgetBypass_TickCheck(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    // Kill volume / soft ceiling bypass. Re-stamps the TLS bitmask every
    // tick when the user has barriers disabled (engine re-zeroes on map
    // load). No-op when barriers are in their default state.
    __try { HaloMapStudio_ForgeBarrier_Tick(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}

    tls_InTick = false;
}

int64_t __stdcall hkFramePump()
{
    // Call the original first, then run our ticks - we're still inside the
    // pump's call stack (this is one of the pump's per-frame callees), so the
    // engine's object-pool anchor + TLS descriptor are coherent. Forward the
    // original's return value so the pump sees the real result.
    int64_t r = 0;
    if (g_OrigPump) {
        __try { r = g_OrigPump(); }
        __except (EXCEPTION_EXECUTE_HANDLER) { r = 0; }
    }

    RunSnapshotTicks();
    return r;
}

} // namespace

// Snapshot the current first 8 bytes at the target into g_HookFingerprint.
// Wrapped in __try because the page is RX (we just made it executable via
// MinHook's VirtualProtect dance, but a concurrent unmap would still AV).
namespace {
void CaptureHookFingerprint(void* target)
{
    if (!target) return;
    __try {
        memcpy(g_HookFingerprint, target, sizeof(g_HookFingerprint));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        memset(g_HookFingerprint, 0, sizeof(g_HookFingerprint));
    }
}

// Returns true iff the bytes at the target still match our captured
// fingerprint - i.e. our hook is still in place. False means the engine
// (or anti-cheat / map-loader) restored the original bytes and our hkFramePump
// will never run again until we re-install.
bool HookBytesIntact(void* target)
{
    if (!target) return false;
    uint8_t cur[8] = {};
    __try {
        memcpy(cur, target, sizeof(cur));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return memcmp(cur, g_HookFingerprint, sizeof(cur)) == 0;
}

// Take an installed-but-now-stale hook back to a working state. Removes
// MinHook's tracking (so MH_CreateHook can be called again), creates the
// hook fresh against the current bytes, enables it, and refreshes the
// fingerprint. Used only by the watcher when it detects the bytes drifted.
//
// Why we VirtualProtect first: MCC's map-load restoration writes the
// original bytes back via VirtualProtect(RW)+memcpy without restoring the
// page to RX, so MinHook's IsExecutableAddress check rejects the target
// with MH_ERROR_NOT_EXECUTABLE (code 7). Flipping the page back to
// PAGE_EXECUTE_READWRITE here is the smallest-blast-radius fix - 
// MH_CreateHook would have to escalate the protection internally anyway
// to write its trampoline, so doing it up front just unblocks the check.
bool ReinstallHookLocked()
{
    if (!g_HookTarget) return false;

    // Make sure the target page is executable before MinHook checks. We
    // restore the previous protection on the way out (the next instruction
    // pump call will fault if we left it RW - it's a code page).
    DWORD oldProt = 0;
    BOOL vpOk = VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READWRITE, &oldProt);
    if (!vpOk) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump REINSTALL FAILED - VirtualProtect=%lu\n",
                (unsigned long)GetLastError());
        return false;
    }

    // Drop the now-meaningless MinHook entry. MH_RemoveHook silently
    // succeeds on a stale entry (it just frees its trampoline).
    MH_DisableHook(g_HookTarget);
    MH_RemoveHook(g_HookTarget);
    g_OrigPump = nullptr;

    MH_STATUS create = MH_CreateHook(g_HookTarget, (void*)&hkFramePump, (void**)&g_OrigPump);
    if (create != MH_OK) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump REINSTALL FAILED - MH_CreateHook=%d\n", (int)create);
        // Re-pin the page back to executable so the engine can keep running.
        DWORD ignored = 0;
        VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READ, &ignored);
        return false;
    }
    MH_STATUS enable = MH_EnableHook(g_HookTarget);
    if (enable != MH_OK) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump REINSTALL FAILED - MH_EnableHook=%d\n", (int)enable);
        MH_RemoveHook(g_HookTarget);
        g_OrigPump = nullptr;
        DWORD ignored = 0;
        VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READ, &ignored);
        return false;
    }

    // Restore page protection - MinHook's hook bytes are in place; the
    // page just needs PAGE_EXECUTE_READ so the engine's instruction
    // fetcher works.
    {
        DWORD ignored = 0;
        VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READ, &ignored);
    }

    CaptureHookFingerprint(g_HookTarget);
    ZH_Logf("[HaloMapStudioDLL] frame-pump hook REINSTALLED after byte-drift detected\n");
    return true;
}

DWORD WINAPI HookWatcherThread(LPVOID)
{
    // Poll every 500ms. Map loads are a 1-3 second event; missing the start
    // of a new map by half a second is acceptable. We also can't run too
    // fast - VirtualProtect contention with the engine during a map load
    // can stall the engine briefly, and a per-frame poll would worsen that.
    DWORD reinstallFailLogBudget = 4; // log first N failures, then go silent
    while (true) {
        DWORD wait = WaitForSingleObject(g_WatcherStop, 500);
        if (wait == WAIT_OBJECT_0) break;

        if (!g_Installed.load(std::memory_order_acquire)) continue;
        if (!g_HookTarget) continue;

        // Skip the watcher entirely if haloreach.dll is currently unloaded.
        // MCC unloads the engine DLL between maps; trying to VirtualProtect
        // an unmapped page produces ERROR_INVALID_ADDRESS (487) which spams
        // the log every 500ms with no recovery possible. When the engine
        // reloads the DLL the install path runs fresh anyway.
        HMODULE hr = GetModuleHandleW(L"haloreach.dll");
        if (!hr) continue;

        // Recompute the expected hook target from the CURRENT module base.
        // After a map switch MCC may reload haloreach.dll at a different
        // base address, which makes the captured g_HookTarget point at
        // unrelated memory. If the expected target differs from the
        // tracked one, the module has been re-loaded and we must drop the
        // stale hook tracking and re-install at the new address.
        void* expectedTarget = (uint8_t*)hr + HaloMapStudio::Engine::kRva_FramePumpHook;
        bool moduleRebased = (expectedTarget != g_HookTarget);

        // Sanity-check that the hook target is still inside the loaded
        // module's address range. If the module was reloaded at a new base
        // the old g_HookTarget address points at unrelated memory.
        MEMORY_BASIC_INFORMATION mbi = {};
        if (VirtualQuery(g_HookTarget, &mbi, sizeof(mbi)) == 0 ||
            (mbi.State & MEM_COMMIT) == 0)
        {
            // Old target memory is gone. If the new module is loaded,
            // retarget; otherwise wait for the next tick.
            if (moduleRebased) {
                ZH_Logf("[HaloMapStudioDLL] frame-pump retargeting old=%p new=%p (haloreach.dll rebase)\n",
                        g_HookTarget, expectedTarget);
                MH_DisableHook(g_HookTarget);
                MH_RemoveHook(g_HookTarget);
                g_HookTarget = expectedTarget;
                g_OrigPump = nullptr;
                __try { ReinstallHookLocked(); } __except (EXCEPTION_EXECUTE_HANDLER) {}
            }
            continue;
        }

        if (moduleRebased) {
            // Module reloaded at a new base (old target may technically be
            // committed if MCC didn't fully unmap, but our hook is gone
            // since the new module's bytes at the new base haven't been
            // patched yet). Retarget + re-install.
            ZH_Logf("[HaloMapStudioDLL] frame-pump retargeting old=%p new=%p (haloreach.dll rebase)\n",
                    g_HookTarget, expectedTarget);
            MH_DisableHook(g_HookTarget);
            MH_RemoveHook(g_HookTarget);
            g_HookTarget = expectedTarget;
            g_OrigPump = nullptr;
            bool ok = false;
            __try { ok = ReinstallHookLocked(); }
            __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }
            if (!ok && reinstallFailLogBudget > 0) {
                --reinstallFailLogBudget;
                if (reinstallFailLogBudget == 0)
                    ZH_Logf("[HaloMapStudioDLL] frame-pump REINSTALL silenced (bursting)\n");
            }
            continue;
        }

        if (!HookBytesIntact(g_HookTarget)) {
            // Engine restored the original bytes (map-load anti-tamper).
            // Re-install before any tick gets dropped.
            bool ok = false;
            __try { ok = ReinstallHookLocked(); }
            __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }
            if (!ok && reinstallFailLogBudget > 0) {
                --reinstallFailLogBudget;
                if (reinstallFailLogBudget == 0)
                    ZH_Logf("[HaloMapStudioDLL] frame-pump REINSTALL silenced (bursting)\n");
            }
        }
    }
    return 0;
}
} // namespace

extern "C" bool FramePumpHook_Install()
{
    if (g_Installed.load(std::memory_order_acquire)) return true;

    // Wait briefly for haloreach.dll to load - it's not present at the
    // exact moment the viewer DLL injects, since MCC loads game DLLs on
    // demand. We sleep up to ~5s in 100ms slices.
    HMODULE hr = nullptr;
    for (int i = 0; i < 50; ++i) {
        hr = GetModuleHandleW(L"haloreach.dll");
        if (hr) break;
        Sleep(100);
    }
    if (!hr) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump install FAILED - haloreach.dll not loaded\n");
        return false;
    }

    void* target = (uint8_t*)hr + HaloMapStudio::Engine::kRva_FramePumpHook;
    g_HookTarget = target;

    MH_STATUS init = MH_Initialize();
    if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump install FAILED - MH_Initialize=%d\n", (int)init);
        return false;
    }

    MH_STATUS create = MH_CreateHook(target, (void*)&hkFramePump, (void**)&g_OrigPump);
    if (create != MH_OK) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump install FAILED - MH_CreateHook=%d\n", (int)create);
        return false;
    }

    MH_STATUS enable = MH_EnableHook(target);
    if (enable != MH_OK) {
        ZH_Logf("[HaloMapStudioDLL] frame-pump install FAILED - MH_EnableHook=%d\n", (int)enable);
        MH_RemoveHook(target);
        return false;
    }

    CaptureHookFingerprint(target);
    g_Installed.store(true, std::memory_order_release);
    ZH_Logf("[HaloMapStudioDLL] frame-pump hook installed\n");

    // Start the watcher thread - handles the case where MCC's map-load
    // restores the original bytes at our target without unloading
    // haloreach.dll. Module-base never changes, so a module-watch wouldn't
    // catch this; we have to compare the actual instruction bytes.
    if (!g_WatcherStop) {
        g_WatcherStop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    }
    if (!g_WatcherThread) {
        g_WatcherThread = CreateThread(nullptr, 0, HookWatcherThread, nullptr, 0, nullptr);
        if (g_WatcherThread) {
            SetThreadPriority(g_WatcherThread, THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
    return true;
}

extern "C" void FramePumpHook_Remove()
{
    // Shut down the watcher first so it doesn't race with us tearing
    // the hook down.
    if (g_WatcherStop) SetEvent(g_WatcherStop);
    if (g_WatcherThread) {
        WaitForSingleObject(g_WatcherThread, 2000);
        CloseHandle(g_WatcherThread);
        g_WatcherThread = nullptr;
    }
    if (g_WatcherStop) {
        CloseHandle(g_WatcherStop);
        g_WatcherStop = nullptr;
    }

    if (!g_Installed.load(std::memory_order_acquire)) return;
    if (g_HookTarget) {
        MH_DisableHook(g_HookTarget);
        MH_RemoveHook(g_HookTarget);
    }
    // Don't MH_Uninitialize - we don't own MinHook's lifecycle if some
    // future module also uses it. (Today we're the only user; tomorrow
    // we might not be.)
    g_HookTarget = nullptr;
    g_OrigPump = nullptr;
    g_Installed.store(false, std::memory_order_release);
}
