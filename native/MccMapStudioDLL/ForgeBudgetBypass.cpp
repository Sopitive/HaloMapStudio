// ForgeBudgetBypass.cpp
// =============================================================================
// Bypass the per-palette-entry budget + max-count gate so the viewer can
// spawn forge objects past the scnr-authored caps.
//
// Target: haloreach.dll + 0x751F4 (Forge_ValidateSpawnRequest). RE notes in
// EngineThreadResolver.h next to kRva_Forge_ValidateSpawnRequest.
//
// Strategy: MinHook detour. The hook calls the original first so its side
// effects (writing animation lookup data into the optional `param_2` output
// buffer) run unchanged, then returns '\x01' unconditionally - overriding
// the engine's verdict on max-count and budget. The only invariant we keep
// is `param_1 != -1` (the engine's own "is this a valid palette index"
// sanity gate); without it the orig would crash dereferencing the index.
//
// Why hook this single function rather than NOP the call sites:
//   * Three callers (Forge_ResolvePalletObject, Session_HandleWeaponSpawnRequest,
//     ForgeDescriptor_CanSpawn). One detour neutralises all three.
//   * In-place NOP of the branch bytes would require identifying the exact
//     `je`/`jne` instructions inside the function - fragile across game
//     updates. Hooking the entry-point is stable as long as the function
//     keeps the same RVA.
//
// Map-load survival: MCC's anti-tamper restores some engine bytes during
// map transitions. We piggy-back on the FramePumpHook watcher pattern by
// re-checking byte integrity on each frame-pump tick. Cheap (one memcmp)
// and avoids spawning yet another watcher thread.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"

#include <windows.h>
#include <cstdint>
#include <atomic>

#include "MinHook.h"

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

// Match the original function's calling convention. x64 has a single ABI
// for all extern "C" functions, but we type the signature precisely so the
// compiler doesn't fuse anything weird.
using ForgeValidate_t = char (*)(int /*param_1*/, void* /*param_2_optional*/);

ForgeValidate_t   g_OrigValidate = nullptr;
void*             g_HookTarget   = nullptr;

using ForgeResolveCap_t = char (*)(void* /*param_1*/, void* /*param_2*/, void* /*param_3*/);
ForgeResolveCap_t g_OrigResolveCap = nullptr;
void*             g_CapHookTarget  = nullptr;

std::atomic<bool> g_Installed{ false };

// 8-byte fingerprint of the bytes at the hook target right after install.
// Same pattern FramePumpHook uses: if MCC restores the engine bytes on
// map-load this will mismatch and CheckAndReinstall() repatches.
uint8_t g_HookFingerprint[8] = {};

char hkForgeValidateSpawnRequest(int param_1, void* param_2)
{
    // Preserve the engine's only safety invariant - param_1 = -1 means
    // "no palette entry selected" and dereferencing it would crash. Let
    // the orig handle that case (it returns '\0' quickly).
    if (param_1 == -1) {
        if (g_OrigValidate) return g_OrigValidate(param_1, param_2);
        return 0;
    }

    // Call orig so its side-effects on param_2 (animation lookup output)
    // run if any caller relies on them. We discard the budget verdict.
    if (g_OrigValidate) {
        __try { (void)g_OrigValidate(param_1, param_2); }
        __except (EXCEPTION_EXECUTE_HANDLER) {
            // Orig faulted (likely on a stale palette pointer). Fall
            // through to the unconditional success - the upstream
            // Forge_ResolvePalletObject already validated the index
            // before reaching us.
        }
    }
    return 1;
}

char hkForgeValidateAndResolveSpawn(void* p1, void* p2, void* p3)
{
    if (g_OrigResolveCap) {
        __try { (void)g_OrigResolveCap(p1, p2, p3); }
        __except (EXCEPTION_EXECUTE_HANDLER) {}
    }
    return 1;
}

void CaptureFingerprint(void* target)
{
    if (!target) return;
    __try {
        memcpy(g_HookFingerprint, target, sizeof(g_HookFingerprint));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        memset(g_HookFingerprint, 0, sizeof(g_HookFingerprint));
    }
}

bool FingerprintIntact()
{
    if (!g_HookTarget) return false;
    uint8_t cur[8] = {};
    __try {
        memcpy(cur, g_HookTarget, sizeof(cur));
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return memcmp(cur, g_HookFingerprint, sizeof(cur)) == 0;
}

bool ReinstallLocked()
{
    if (!g_HookTarget) return false;

    DWORD oldProt = 0;
    BOOL vpOk = VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READWRITE, &oldProt);
    if (!vpOk) {
        ZH_Logf("[ForgeBudgetBypass] REINSTALL FAILED - VirtualProtect=%lu\n",
                (unsigned long)GetLastError());
        return false;
    }

    MH_DisableHook(g_HookTarget);
    MH_RemoveHook(g_HookTarget);
    g_OrigValidate = nullptr;

    MH_STATUS create = MH_CreateHook(g_HookTarget,
                                     (void*)&hkForgeValidateSpawnRequest,
                                     (void**)&g_OrigValidate);
    if (create != MH_OK) {
        ZH_Logf("[ForgeBudgetBypass] REINSTALL FAILED - MH_CreateHook=%d\n", (int)create);
        DWORD ignored = 0;
        VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READ, &ignored);
        return false;
    }
    MH_STATUS enable = MH_EnableHook(g_HookTarget);
    if (enable != MH_OK) {
        ZH_Logf("[ForgeBudgetBypass] REINSTALL FAILED - MH_EnableHook=%d\n", (int)enable);
        MH_RemoveHook(g_HookTarget);
        g_OrigValidate = nullptr;
        DWORD ignored = 0;
        VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READ, &ignored);
        return false;
    }

    DWORD ignored = 0;
    VirtualProtect(g_HookTarget, 32, PAGE_EXECUTE_READ, &ignored);

    CaptureFingerprint(g_HookTarget);
    ZH_Logf("[ForgeBudgetBypass] hook REINSTALLED after byte-drift detected\n");
    return true;
}

} // namespace

extern "C" bool ForgeBudgetBypass_Install()
{
    if (g_Installed.load(std::memory_order_acquire)) return true;

    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) {
        ZH_Logf("[ForgeBudgetBypass] install skipped - haloreach.dll not loaded\n");
        return false;
    }

    void* target = (uint8_t*)hr + HaloMapStudio::Engine::kRva_Forge_ValidateSpawnRequest;
    g_HookTarget = target;

    // MinHook is already initialised by FramePumpHook_Install - we just
    // hop on the existing instance. MH_Initialize is safe to call twice
    // (returns MH_ERROR_ALREADY_INITIALIZED on the second call) so we
    // don't depend on install order here.
    MH_STATUS init = MH_Initialize();
    if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) {
        ZH_Logf("[ForgeBudgetBypass] install FAILED - MH_Initialize=%d\n", (int)init);
        return false;
    }

    MH_STATUS create = MH_CreateHook(target,
                                     (void*)&hkForgeValidateSpawnRequest,
                                     (void**)&g_OrigValidate);
    if (create != MH_OK) {
        ZH_Logf("[ForgeBudgetBypass] install FAILED - MH_CreateHook=%d target=%p\n",
                (int)create, target);
        return false;
    }

    MH_STATUS enable = MH_EnableHook(target);
    if (enable != MH_OK) {
        ZH_Logf("[ForgeBudgetBypass] install FAILED - MH_EnableHook=%d\n", (int)enable);
        MH_RemoveHook(target);
        g_OrigValidate = nullptr;
        return false;
    }

    CaptureFingerprint(target);

    void* capTarget = (uint8_t*)hr + HaloMapStudio::Engine::kRva_Forge_ValidateAndResolveSpawn;
    g_CapHookTarget = capTarget;
    MH_STATUS capCreate = MH_CreateHook(capTarget,
                                        (void*)&hkForgeValidateAndResolveSpawn,
                                        (void**)&g_OrigResolveCap);
    if (capCreate == MH_OK) {
        MH_STATUS capEnable = MH_EnableHook(capTarget);
        if (capEnable == MH_OK)
            ZH_Logf("[ForgeBudgetBypass] session-cap hook installed at %p (32-object cap bypassed)\n", capTarget);
        else
            ZH_Logf("[ForgeBudgetBypass] session-cap hook enable FAILED=%d\n", (int)capEnable);
    } else {
        ZH_Logf("[ForgeBudgetBypass] session-cap hook create FAILED=%d\n", (int)capCreate);
    }

    g_Installed.store(true, std::memory_order_release);
    ZH_Logf("[ForgeBudgetBypass] hook installed at %p (max-count + budget bypassed)\n", target);
    return true;
}

// Called from the frame-pump watcher. Three jobs:
//   1) If we never installed (haloreach.dll wasn't loaded at startup), try
//      to install NOW - this is the common case where MMS launches before
//      the user opens a map, so haloreach.dll loads later. Without this
//      retry the bypass is permanently disabled for the session, even
//      though the rest of the rendering pipeline picks up the new module
//      just fine. User report: "limit is still in effect".
//   2) If we're installed but the engine bytes have drifted (MCC anti-tamper
//      restored them across a map transition), reinstall.
//   3) Otherwise do nothing - cheap path.
extern "C" void ForgeBudgetBypass_TickCheck()
{
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return;  // engine not loaded yet - nothing to hook

    if (!g_Installed.load(std::memory_order_acquire)) {
        // Late install: haloreach.dll is loaded now but wasn't at startup.
        // Best-effort; if install fails (e.g. RVA drift on a game update),
        // we'll keep retrying every tick - harmless because
        // ForgeBudgetBypass_Install short-circuits the second-attempt path.
        __try { (void)ForgeBudgetBypass_Install(); }
        __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }
    if (!g_HookTarget) return;
    if (FingerprintIntact()) return;
    __try { (void)ReinstallLocked(); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
}

extern "C" void ForgeBudgetBypass_Remove()
{
    if (!g_Installed.load(std::memory_order_acquire)) return;
    if (g_HookTarget) {
        MH_DisableHook(g_HookTarget);
        MH_RemoveHook(g_HookTarget);
    }
    if (g_CapHookTarget) {
        MH_DisableHook(g_CapHookTarget);
        MH_RemoveHook(g_CapHookTarget);
    }
    g_HookTarget     = nullptr;
    g_OrigValidate   = nullptr;
    g_CapHookTarget  = nullptr;
    g_OrigResolveCap = nullptr;
    g_Installed.store(false, std::memory_order_release);
}
