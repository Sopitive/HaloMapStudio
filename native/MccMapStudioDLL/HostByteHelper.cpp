// HostByteHelper.cpp
// =============================================================================
// Stripped Enqueue_UpdateObjectPosAndAiming for HaloMapStudioDLL.
//
// The viewer's TransformQueueSnapshot drains its MMF and forwards each entry
// to Enqueue_UpdateObjectPosAndAiming. A full implementation would do
// host-byte authority swapping + SEH protection + TLS-aware flag handling.
// The viewer doesn't need any of
// that - its only use case is "set object position to x,y,z right now". So
// we resolve haloreach.dll!UpdatePosAndAiming (RVA 0x46CB38) directly and
// invoke it under SEH. If it fails, the entry is dropped silently - the
// next drag tick will publish a fresh attempt.
//
// This file deliberately does NOT export any host-byte plumbing
// (GetHostByteAddress, GetHostByteRefValue, IsHost, etc.) - the rest of the
// viewer DLL never needs them.
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <cstdint>

namespace {

struct vec3 { float x; float y; float z; };

using UpdatePosAndAiming_Fn = void* (__fastcall*)(
    uint32_t objectId,
    vec3*    positionOrNull,
    vec3*    forwardOrNull,
    vec3*    upOrNull,
    uint64_t arg5,
    uint64_t arg6);

constexpr uintptr_t kRva_UpdatePosAndAiming = 0x46CB38;

UpdatePosAndAiming_Fn ResolveUpdatePosAndAiming()
{
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    return reinterpret_cast<UpdatePosAndAiming_Fn>((uint8_t*)mod + kRva_UpdatePosAndAiming);
}

} // namespace

// TransformQueueSnapshot
// passes flags=0x1 (HasPos) and zeroed forward/up/arg5/arg6 in the viewer path,
// so we ignore the orientation flags and just invoke pos-only. Returns 0 on
// success-ish (we don't truly know whether the engine accepted the call); 1
// on resolve failure; 2 if SEH caught a fault inside the engine call.
extern "C" __declspec(dllexport) uint32_t __stdcall Enqueue_UpdateObjectPosAndAiming(
    uint32_t objectId, uint32_t flags,
    float px, float py, float pz,
    float fx, float fy, float fz,
    float ux, float uy, float uz,
    uint64_t arg5, uint64_t arg6)
{
    (void)arg5; (void)arg6;

    UpdatePosAndAiming_Fn fn = ResolveUpdatePosAndAiming();
    if (!fn) return 1;

    vec3 pos { px, py, pz };
    vec3 fwd { fx, fy, fz };
    vec3 up  { ux, uy, uz };

    vec3* posPtr = (flags & 0x1u) ? &pos : nullptr;
    vec3* fwdPtr = (flags & 0x2u) ? &fwd : nullptr;
    vec3* upPtr  = (flags & 0x2u) ? &up  : nullptr;

    __try {
        fn(objectId, posPtr, fwdPtr, upPtr, 0ull, 0ull);
        return 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 2;
    }
}
