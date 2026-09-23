// SpawnDefinitionLive.cpp
// =============================================================================
// Live, networked spawn-by-tag.
//
// The default paths spawn forge objects either through the forge-palette
// variant command (DispatchSpawnAtPosition) or the LOCAL object_new primitive
// (SpawnObjectFromTagInner). Both create the object on our client but do NOT
// register it with the session's replication/authority system, so such a
// spawn isn't a "real" live object.
//
// Spawns an object from a tag definition into a LIVE game. The two engine
// functions involved are the same ones the forge spawn path uses - 
//     Prepare  == object_placement_data_new  == haloreach+0x49CC24
//     Spawn    == SpawnObjectFromDefinition   == haloreach+0x46D6E0
// so the ONLY thing that makes the spawn network is the wrapper:
//   1. Force session-state byte  *(haloreach+0x2C8DFE8) = 0x07  around the spawn
//   2. Force the TLS host byte    (tlsBlock+0x48 -> node+0x11) = 0x05
//   3. Networking handover after: AssignAuthoritySlot + SetOwnerPeer(host=0) +
//      RebuildPhysics.
// Both bytes MUST be set or the engine refuses to network the spawn. The
// previous byte values are restored afterward.
//
// We reuse MMS's own placement-struct offsets (pos@+0x28, fwd@+0x34, scale@+0x58
// - validated by the working CACHE_SPAWN path) and snap the full pose with the
// engine poser after spawn.
//
// GAME THREAD ONLY: call from the frame-pump tick (that's where the host TLS
// block is populated, same as the object-descriptor resolver).
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <cstdint>
#include "EngineThreadResolver.h"   // GetTLSArrayBase, ResolveTlsIndexFromModule, SafeReadPtr, kRva_object_*

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

using namespace HaloMapStudio::Engine;

// Extra RVAs for the SAME haloreach.dll build as kRva_object_placement_data_new /
// kRva_object_new.
constexpr uintptr_t kRva_Force07Byte     = 0x2C8DFE8; // session-state byte; must be 0x07 for a networked spawn
constexpr uintptr_t kRva_AssignAuthority = 0x1284F8;  // Networking_AssignAuthoritySlot(datum)
constexpr uintptr_t kRva_SetOwnerPeer    = 0x17E3C0;  // Networking_SetOwnerPeer(datum, peerIndex)
constexpr uintptr_t kRva_RebuildPhysics  = 0x476F1C;  // Object_RebuildPhysics(datum)

using objpd_new_t  = void     (*)(void* outPd, uint32_t tagIndex, uint32_t owner, void* r9);
using object_new_t = uint32_t (*)(void* placement);
using setpose_t    = uint8_t  (*)(uint32_t datum, float* pos, float* fwd, float* up, int a5, int a6);
using authority_t  = void (__fastcall*)(uint32_t datum);
using setowner_t   = void (__fastcall*)(uint32_t datum, int peer);
using rebuild_t    = void (__fastcall*)(uint32_t datum);

// TLS host byte: tlsBlock+0x48 -> node, node+0x11 (0x04 = client, 0x05 = host).
// tlsBlock is the same haloreach TLS block MMS's descriptor resolver reads at
// +0x10; here we read +0x48 for the networking node.
uint8_t* TryGetHostBytePtr()
{
    static uint32_t s_tlsIndex = 0xFFFFFFFFu;
    if (s_tlsIndex == 0xFFFFFFFFu)
        s_tlsIndex = ResolveTlsIndexFromModule(L"haloreach.dll");
    if (s_tlsIndex == 0xFFFFFFFFu) return nullptr;

    void** tlsArray = GetTLSArrayBase();
    if (!tlsArray) return nullptr;

    void* tlsBlock = nullptr;
    __try { tlsBlock = tlsArray[s_tlsIndex]; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    if (!tlsBlock) return nullptr;

    void* node = nullptr;
    if (!SafeReadPtr((uint8_t*)tlsBlock + 0x48, node) || !node) return nullptr;
    return (uint8_t*)node + 0x11;
}

struct ForcedState {
    uint8_t* b07; uint8_t prev07; bool changed07;
    uint8_t* bHost; uint8_t prevHost; bool changedHost;
};

void BeginForced(ForcedState& st, uint8_t* base)
{
    st = {};
    // 1. Session-state byte @ haloreach+0x2C8DFE8 must be 0x07.
    st.b07 = base + kRva_Force07Byte;
    __try {
        st.prev07 = *st.b07;
        if (st.prev07 != 0x07) { *st.b07 = 0x07; st.changed07 = true; }
    } __except (EXCEPTION_EXECUTE_HANDLER) { st.b07 = nullptr; }

    // 2. TLS host byte must be 0x05 (host).
    uint8_t* hp = TryGetHostBytePtr();
    if (hp) {
        st.bHost = hp;
        __try {
            st.prevHost = *hp;
            if (*hp != 0x05) { *hp = 0x05; st.changedHost = true; }
        } __except (EXCEPTION_EXECUTE_HANDLER) { st.bHost = nullptr; }
    }
}

void EndForced(ForcedState& st)
{
    if (st.b07 && st.changed07)
        __try { *st.b07 = st.prev07; } __except (EXCEPTION_EXECUTE_HANDLER) {}
    if (st.bHost && st.changedHost)
        __try { *st.bHost = st.prevHost; } __except (EXCEPTION_EXECUTE_HANDLER) {}
}

void PerformNetworkingHandover(uint8_t* base, uint32_t datum)
{
    if (!datum || datum == 0xFFFFFFFFu) return;
    auto assignAuthority = reinterpret_cast<authority_t>(base + kRva_AssignAuthority);
    auto setOwnerPeer    = reinterpret_cast<setowner_t >(base + kRva_SetOwnerPeer);
    auto rebuildPhysics  = reinterpret_cast<rebuild_t  >(base + kRva_RebuildPhysics);
    __try {
        if (assignAuthority) assignAuthority(datum);
        if (setOwnerPeer)    setOwnerPeer(datum, 0);   // hand ownership to host peer 0
        if (rebuildPhysics)  rebuildPhysics(datum);
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
}

} // namespace

// -----------------------------------------------------------------------------
// Live networked spawn-from-definition by object-definition tag id.
//
//   returns: new object datum, or 0xFFFFFFFF on failure.
//   tagId  : the obje-derived object DEFINITION tag (MMS PrimaryTagId), the same
//            value SpawnObjectFromTagInner takes.
//   pose   : world position + forward/up unit vectors (Halo space).
//
// GAME THREAD ONLY.
// -----------------------------------------------------------------------------
extern "C" __declspec(dllexport) uint32_t HaloMapStudio_SpawnDefinitionLive(
    uint32_t tagId,
    float x,  float y,  float z,
    float fx, float fy, float fz,
    float ux, float uy, float uz)
{
    if (tagId == 0u || tagId == 0xFFFFFFFFu) return 0xFFFFFFFFu;

    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return 0xFFFFFFFFu;
    uint8_t* base = reinterpret_cast<uint8_t*>(hr);

    auto objpd_new  = reinterpret_cast<objpd_new_t >(base + kRva_object_placement_data_new);
    auto object_new = reinterpret_cast<object_new_t>(base + kRva_object_new);
    auto setpose    = reinterpret_cast<setpose_t   >(base + kRva_object_set_pos_orient);

    ForcedState st{};
    BeginForced(st, base);

    uint32_t datum = 0xFFFFFFFFu;
    __try {
        // 424-byte placement struct, 16-byte aligned for the engine's vector writes.
        alignas(16) uint8_t pd[0x1A8] = {};
        objpd_new(pd, tagId, 0xFFFFFFFFu, nullptr);

        // MMS's validated placement offsets (pos@+0x28, fwd@+0x34). Scale stays
        // at the 1.0 default placement_data_new wrote (+0x58).
        *reinterpret_cast<float*>(pd + 0x28) = x;
        *reinterpret_cast<float*>(pd + 0x2C) = y;
        *reinterpret_cast<float*>(pd + 0x30) = z;
        *reinterpret_cast<float*>(pd + 0x34) = fx;
        *reinterpret_cast<float*>(pd + 0x38) = fy;
        *reinterpret_cast<float*>(pd + 0x3C) = fz;

        datum = object_new(pd);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        datum = 0xFFFFFFFFu;
    }

    if (datum != 0xFFFFFFFFu) {
        // Snap the full pose (placement only encoded fwd, not up).
        float pos[3] = { x, y, z };
        float fwd[3] = { fx, fy, fz };
        float up[3]  = { ux, uy, uz };
        __try { if (setpose) setpose(datum, pos, fwd, up, 0, 0); }
        __except (EXCEPTION_EXECUTE_HANDLER) {}

        // Register with the session so it's a real live/replicated object.
        PerformNetworkingHandover(base, datum);
    }

    EndForced(st);

    ZH_Logf("[SpawnDefLive] tag=0x%X datum=0x%08X pos=(%.2f,%.2f,%.2f) "
            "force07=%d host=%d\n",
            tagId, datum, x, y, z, st.changed07 ? 1 : 0, st.changedHost ? 1 : 0);
    return datum;
}
