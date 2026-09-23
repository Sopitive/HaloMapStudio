// ForgePhysics.cpp
// =============================================================================
// Wake an object's havok rigid body after a manual pose override so gravity
// applies. User report: vehicles dropped via MMS's manual transform override
// stay frozen in mid-air until the engine "touches" them on its own.
//
// CURRENT STATE: stub. The engine wake function's RVA hasn't been RE'd yet
// (kRva_Object_WakePhysics is 0). The helper short-circuits cleanly in that
// state - calling it is a no-op + an optional log line. Once the RVA lands,
// uncomment the wake call below.
//
// Hook integration: caller is `ForgePhysics_WakeAfterPose(objectDatum)` from
// every site in `ForgeObjectEdit.cpp` that writes a pose. The helper looks
// up the engine object struct from the datum, fetches the rigid-body
// pointer, and calls the wake function on it.
//
// See PHYSICS_WAKE_RE.md for the RE attack plan + Ghidra search strategy.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"

#include <windows.h>
#include <cstdint>
#include <cstdlib>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

// Engine object-table pointer. TODO: wire to the datum-to-object resolver
// used by ForgeObjectEdit.cpp.
using ObjectFromDatum_t = void* (*)(uint32_t /*datum*/);
ObjectFromDatum_t g_ObjectFromDatum = nullptr;

// Havok wake function. Signature once RE'd:
//   void object_wake_physics(s_object* obj);
using WakeFn_t = void(*)(void*);
WakeFn_t g_WakeFn = nullptr;

bool LookupResolvers()
{
    if (g_WakeFn != nullptr) return true;
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return false;
    constexpr uintptr_t rva = HaloMapStudio::Engine::kRva_Object_WakePhysics;
    if (rva == 0)
    {
        // Not RE'd yet - sentinel. Log once and bail.
        static bool s_loggedOnce = false;
        if (!s_loggedOnce)
        {
            s_loggedOnce = true;
            ZH_Logf("[ForgePhysics] WakeAfterPose: kRva_Object_WakePhysics not set; gravity-after-move disabled. See PHYSICS_WAKE_RE.md\n");
        }
        return false;
    }
    g_WakeFn = (WakeFn_t)((uint8_t*)hr + rva);
    return true;
}

} // namespace

// Public entry. Safe to call from any forge-edit code path; short-circuits
// when the wake function hasn't been RE'd yet.
extern "C" void ForgePhysics_WakeAfterPose(uint32_t objectDatum)
{
    if (objectDatum == 0u || objectDatum == 0xFFFFFFFFu) return;
    if (!LookupResolvers()) return;
    if (g_ObjectFromDatum == nullptr) return;  // datum resolver not wired yet

    void* obj = nullptr;
    __try { obj = g_ObjectFromDatum(objectDatum); }
    __except (EXCEPTION_EXECUTE_HANDLER) { obj = nullptr; }
    if (obj == nullptr) return;

    __try { g_WakeFn(obj); }
    __except (EXCEPTION_EXECUTE_HANDLER) { /* swallow - bad RVA */ }

    // Optional diagnostic when MMS_PHYSICS_WAKE_LOG=1
    char* envBuf = nullptr; size_t envLen = 0;
    _dupenv_s(&envBuf, &envLen, "MMS_PHYSICS_WAKE_LOG");
    if (envBuf != nullptr)
    {
        if (envBuf[0] == '1')
            ZH_Logf("[ForgePhysics] WAKE called obj=0x%08X\n", (unsigned)objectDatum);
        free(envBuf);
    }
}
