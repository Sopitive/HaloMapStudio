// TransformQueueSnapshot.cpp
// =============================================================================
// Cross-process transform queue (viewer translate-gizmo + spawn-here).
//
// (Enqueue_UpdateObjectPosAndAiming is satisfied by HostByteHelper.cpp in
// this DLL - a stripped, host-byte-free version that calls haloreach
// directly.)
//
// The viewer can't enqueue calls into UpdateObjectPosAndAimingCall.cpp's
// in-process ring buffer directly - that lives in MCC's address space. Instead
// we expose a small MMF, ZeroHour_TransformQueue_Shared, that the viewer
// writes pending entries into; this file's frame-pump tick drains them
// and forwards to the existing in-process Enqueue_UpdateObjectPosAndAiming
// (so the calls land at a known-good engine entry point with the host-byte
// authority dance already in place).
//
// Layout (single-producer / single-consumer ring buffer, viewer writes,
// DLL reads):
//   Header (32 B)
//     u32 Magic         'ZHTQ' (0x5154485A)
//     u32 Version       1
//     u32 WriteCounter  bumped by viewer after each entry write
//     u32 ReadCounter   bumped by DLL after each entry consumed
//     u32 Capacity      kQueueCap (16)
//     u32 _pad[3]
//   Entry[kQueueCap] - 32 B each
//     u32 Datum
//     u32 Flags         bit0=hasPos (only Pos is supported in v1)
//     f32 Px, Py, Pz
//     u32 _reservedFwdUp[3]   (ignored in v1; future: forward + up)
//
// Total: 32 + 16*32 = 544 bytes.
//
// Cap of 16 is plenty for translate-gizmo dragging (~30 Hz) - the ring
// is drained every frame-pump tick. If the viewer is dragging hard and
// the DLL is paused (alt-tabbed game), older entries get overwritten
// once WriteCounter laps ReadCounter; we silently accept that. The user
// only cares about the *latest* position when they release the mouse.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"
#include <windows.h>
#include <cstdint>
#include <cstring>

extern "C" void ZH_Logf(const char* fmt, ...);

// In-process enqueue (UpdateObjectPosAndAimingCall.cpp). Same signature.
extern "C" __declspec(dllexport) uint32_t __stdcall Enqueue_UpdateObjectPosAndAiming(
    uint32_t objectId, uint32_t flags,
    float px, float py, float pz,
    float fx, float fy, float fz,
    float ux, float uy, float uz,
    uint64_t arg5, uint64_t arg6);

namespace {
// haloreach.dll RVAs of the DeleteForgeObject body. We replicate the
// body verbatim (skipping the Forge_HasPermissionToEditObject gate) so
// that:
//   * The forge object array entry is cleaned up (FUN_180071194 walks
//     every player's TLS placement table and clears any +0x1758 ref to
//     this datum). Without this, the next time the engine ticks the
//     placement table it sees a stale ref to a deleted object and the
//     monitor for that slot can crash.
//   * The selection-holder for the object (if any monitor was carrying
//     it) gets detached.
//   * The actual delete primitive runs (Object_Delete).
//
// The weapon-specific cleanup (Forge_ClearWeaponEntryAndDeleteInstances /
// UI_NotifyWeaponSlotStateChange) wants a weapon-slot index pulled from
// a TLS chain we can't easily walk from outside haloreach's TLS context.
// For non-weapon objects (blocks, vehicles, scenery) those cleanups are
// no-ops anyway, so we skip them and pass -1. For weapons spawned via
// the viewer this means the weapon-pool UI won't reflect the delete - 
// acceptable trade-off.
constexpr uintptr_t kRva_Object_Delete                       = 0x46F0B0;
constexpr uintptr_t kRva_PlayerSelectionTable_ClearForDatum  = 0x71194;  // FUN_180071194
constexpr uintptr_t kRva_Forge_TryDetachSelectedObject       = 0x761B8;
constexpr uintptr_t kRva_Network_GetCurrentSessionContext    = 0x6BCEC;
constexpr uintptr_t kRva_Forge_ClearWeaponEntryAndDeleteInst = 0x6DEE4;
constexpr uintptr_t kRva_UI_NotifyWeaponSlotStateChange      = 0x3A6054;
// haloreach!DeleteForgeObject - the engine's correct forge-delete entrypoint.
// VariantCommand_ApplyUnitAction op code 1 calls THIS verbatim. Decompile
// (see FORGE_UPDATE_OBJECT_RE.md section "DeleteForgeObject"):
//   permission gate -> read forge-index from object_table[datum]+0x1c ->
//   PlayerSelectionTable_ClearForDatum(datum) ->
//   Forge_TryDetachSelectedObjectForPlayer(playerKey) ->
//   Forge_ClearWeaponEntryAndDeleteInstances(sessCtx, forgeIndex) <-- THIS
//     is the SPAWN-LIST clear; without it the engine respawns the object.
//   Object_Delete(datum) ->
//   UI_NotifyWeaponSlotStateChange(forgeIndex, 4) // state=4 = delete
// playerKey encoding matches UpdateObjectData: 0xEC700000 | (self_idx & 0xF).
constexpr uintptr_t kRva_DeleteForgeObject                   = 0x76A80;
constexpr uintptr_t kRva_SelfPlayerIndex                     = 0x02D596C8u;
using PFN_DeleteForgeObject = void(__fastcall*)(uint64_t playerKey, uint64_t datum);

using PFN_OneU32        = void(*)(uint32_t);
using PFN_GetSessionCtx = void*(*)();
using PFN_ClearWeapons  = void(*)(void* sessionCtx, int32_t weaponSlot);
using PFN_NotifyWeapon  = void(*)(int32_t weaponSlot, int32_t state);

template <typename TFn>
TFn ResolveEngineFn(uintptr_t rva)
{
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return nullptr;
    return reinterpret_cast<TFn>(reinterpret_cast<uint8_t*>(hr) + rva);
}

PFN_OneU32 ResolveObjectDelete()
{
    static PFN_OneU32 cached = nullptr;
    if (!cached) cached = ResolveEngineFn<PFN_OneU32>(kRva_Object_Delete);
    return cached;
}

PFN_DeleteForgeObject ResolveDeleteForgeObject()
{
    static PFN_DeleteForgeObject cached = nullptr;
    if (!cached) cached = ResolveEngineFn<PFN_DeleteForgeObject>(kRva_DeleteForgeObject);
    return cached;
}

// haloreach!Forge_PlayerCanEdit (sub_180070FFC) - the leaf inside
// Forge_HasPermissionToEditObject that returns the monitor/edit-mode verdict.
// It reads the per-player record at +0x1754 and requires == 1 (true only when
// the local player is a Forge MONITOR). DeleteForgeObject wraps its ENTIRE body
// in Forge_HasPermissionToEditObject, so when the player is a Spartan (not a
// monitor) the delete is a silent no-op. (RE: ReverseMe func haloreachnew
// 0x18007753C -> CALL 0x180070FFC; the +0x1754==1 compare is the gate.)
//
// We force it to return true for the *duration of one DeleteForgeObject call*
// by patching its prologue to `mov al,1 ; ret` (B0 01 C3), then restoring the
// original bytes. This is the same scoped-bypass idea as the host-byte flip.
// Safe because Forge edits + this gate are all game-thread-only and our call
// is synchronous - nothing else runs the function in the patch window. The
// gate's OTHER checks (object-valid, forge-index-valid, ownership) still run
// inside Forge_HasPermissionToEditObject; only the monitor flag is forced.
constexpr uintptr_t kRva_Forge_PlayerCanEdit = 0x70FFC;

uint8_t* ApplyCanEditBypass(uint8_t saved[3], DWORD* savedProt)
{
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return nullptr;
    uint8_t* addr = reinterpret_cast<uint8_t*>(hr) + kRva_Forge_PlayerCanEdit;
    __try
    {
        DWORD oldProt = 0;
        if (!VirtualProtect(addr, 3, PAGE_EXECUTE_READWRITE, &oldProt))
            return nullptr;
        *savedProt = oldProt;
        saved[0] = addr[0]; saved[1] = addr[1]; saved[2] = addr[2];
        addr[0] = 0xB0; addr[1] = 0x01; addr[2] = 0xC3; // mov al,1 ; ret
        FlushInstructionCache(GetCurrentProcess(), addr, 3);
        return addr;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
}

void RestoreCanEditBypass(uint8_t* addr, const uint8_t saved[3], DWORD savedProt)
{
    if (!addr) return;
    __try
    {
        addr[0] = saved[0]; addr[1] = saved[1]; addr[2] = saved[2];
        FlushInstructionCache(GetCurrentProcess(), addr, 3);
        DWORD tmp = 0; VirtualProtect(addr, 3, savedProt, &tmp);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
}

// Self-player-index resolver - mirrors ForgeObjectEdit.cpp::ReadSelfPlayerIndex.
// Lower 4 bits of the global u32 at haloreach.dll+kRva_SelfPlayerIndex.
uint8_t ReadSelfPlayerIndexForDelete()
{
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return 0;
    uint32_t v = 0;
    __try { v = *(uint32_t*)((uint8_t*)mod + kRva_SelfPlayerIndex); }
    __except (EXCEPTION_EXECUTE_HANDLER) { v = 0; }
    return (uint8_t)(v & 0x0F);
}

// Host-byte resolver - same TLS chain as ForgeObjectEdit.cpp.
// haloreach.dll TLS[0x48] -> node -> +0x11 = host byte. Must be 0x05 across
// the DeleteForgeObject call for the permission gate to pass.
uint8_t* ResolveHostBytePtrForDelete()
{
    static uint32_t s_tlsIndex = 0xFFFFFFFFu;
    if (s_tlsIndex == 0xFFFFFFFFu)
        s_tlsIndex = HaloMapStudio::Engine::ResolveTlsIndexFromModule(L"haloreach.dll");
    if (s_tlsIndex == 0xFFFFFFFFu) return nullptr;

    void** tlsArray = HaloMapStudio::Engine::GetTLSArrayBase();
    if (!tlsArray) return nullptr;

    void* tlsBlock = nullptr;
    __try { tlsBlock = tlsArray[s_tlsIndex]; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    if (!tlsBlock) return nullptr;

    void* node = nullptr;
    __try { node = *(void**)((uint8_t*)tlsBlock + 0x48); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    if (!node) return nullptr;

    uint8_t* hostByte = (uint8_t*)node + 0x11;
    uint8_t tmp = 0;
    __try { tmp = *hostByte; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    (void)tmp;
    return hostByte;
}

// Forge-delete a single object. STRIPPED to just the Object_Delete
// primitive after a serious bug report: the FULL replica path
// (PlayerSelectionTable_ClearForDatum + Forge_TryDetachSelectedObject
// + Forge_ClearWeaponEntryAndDeleteInstances(-1) + Object_Delete +
// UI_NotifyWeaponSlotStateChange) was deleting EVERYTHING in the
// running scene the first time it actually fired live (the viewer's
// EnqueueDelete had been silently rejected by an EnqueuePose gate that
// required FlagHasPos|FlagHasFwdUp, so this code path had never been
// exercised). Most likely culprit was step 3 - the
// `Forge_ClearWeaponEntryAndDeleteInstances(sessionCtx, -1)` call - 
// where the -1 weapon-slot was assumed to short-circuit but actually
// triggers a mass-clear sweep. Until we have proper Ghidra-validated
// per-step understanding, keep the path minimal.
//
// Just calling Object_Delete on a single datum is sufficient for an
// out-of-band viewer: the object disappears from the engine state and
// the per-tick object-table publish stops carrying it. The cosmetic
// cleanups (player monitor placement-table clear, weapon-slot UI
// sync) are nice-to-haves we can re-add ONE AT A TIME after individual
// validation. Step 1 (PlayerSelectionTable_ClearForDatum) is the only
// one with a documented "without this, the next tick crashes" risk - 
// keep it but ONLY pass the specific datum, never broadcast.
void ReplicateDeleteForgeObject(uint32_t playerSlot, uint32_t datum)
{
    (void)playerSlot;  // unused - playerKey is derived from kRva_SelfPlayerIndex

    // rewritten to call the engine's REAL DeleteForgeObject
    // (haloreach.dll+0x76A80) under a host-byte flip - the same pattern
    // ForgeObjectEdit.cpp::CallEngineCommit uses for UpdateObjectData.
    //
    // The prior stripped path (Object_Delete + PlayerSelectionTable_ClearForDatum)
    // removed the runtime object but LEFT THE FORGE SLOT POPULATED with
    // an idExt pointing at the dead datum. The engine's spawn-respawn
    // logic re-instantiates the slot's authored object every gametype tick.
    // User-visible symptom: object disappears, then immediately respawns
    // a few frames later. Calling the engine's DeleteForgeObject ensures
    // the slot itself is cleared via the chain:
    //   permission gate -> read forge-index from object_table[datum]+0x1C
    //   -> PlayerSelectionTable_ClearForDatum -> Forge_TryDetachSelectedObject
    //   -> Forge_ClearWeaponEntryAndDeleteInstances(sessionCtx, forgeIndex)
    //      <-- THIS is the spawn-list clear we were missing
    //   -> Object_Delete -> UI_NotifyWeaponSlotStateChange(forgeIndex, 4)
    //
    // playerKey = 0xEC700000 | (selfPlayerIdx & 0xF) - the engine's
    // "synthetic action source" marker that lets the permission gate
    // bypass animation-state checks (matches UpdateObjectData).
    //
    // Past concern about "deleting EVERYTHING": that was when we passed
    // -1 directly to Forge_ClearWeaponEntryAndDeleteInstances - a mass
    // sweep. Calling DeleteForgeObject reads the SPECIFIC forge-index
    // from object_table[datum]+0x1C, so the clear is scoped to one slot.
    auto fnDel = ResolveDeleteForgeObject();
    if (!fnDel) {
        // Bridge unavailable - fall back to the prior stripped behaviour
        // so we at least destroy the live instance. Slot will still
        // respawn but the user gets some feedback.
        if (auto fnObjDel = ResolveObjectDelete()) {
            __try { fnObjDel(datum); }
            __except (EXCEPTION_EXECUTE_HANDLER) {
                ZH_Logf("[TransformQueue] DeleteForgeObject unresolvable; Object_Delete fallback faulted\n");
            }
        }
        ZH_Logf("[TransformQueue] DeleteForgeObject unresolvable (haloreach.dll not loaded?)\n");
        return;
    }

    // Flip host byte to 0x05 across the call so the permission gate passes.
    uint8_t* hostByte = ResolveHostBytePtrForDelete();
    uint8_t prev = 0;
    bool havePrev = false, didFlip = false;
    if (hostByte) {
        __try {
            prev = *hostByte;
            havePrev = true;
            if (prev != 0x05) { *hostByte = 0x05; didFlip = true; }
        } __except (EXCEPTION_EXECUTE_HANDLER) { hostByte = nullptr; }
    }

    uint64_t playerKey = 0xEC700000ull | (uint64_t)ReadSelfPlayerIndexForDelete();

    // Force the monitor/edit-mode gate true across the call so deletes work
    // when the local player is NOT a Forge monitor. Restored immediately after.
    uint8_t  ceSaved[3] = {0,0,0};
    DWORD    ceProt     = 0;
    uint8_t* cePatch    = ApplyCanEditBypass(ceSaved, &ceProt);

    __try {
        fnDel(playerKey, (uint64_t)datum);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ZH_Logf("[TransformQueue] DeleteForgeObject(0x%016llX, 0x%08X) faulted\n",
                (unsigned long long)playerKey, datum);
    }

    RestoreCanEditBypass(cePatch, ceSaved, ceProt);

    // Restore the host byte unconditionally - leaving it flipped breaks
    // every subsequent host-gate check in the engine.
    if (hostByte && havePrev && didFlip) {
        __try { *hostByte = prev; } __except (EXCEPTION_EXECUTE_HANDLER) {}
    }

    ZH_Logf("[TransformQueue] forge-delete datum=0x%08X via DeleteForgeObject (spawn-list cleared)\n",
            datum);
}
} // namespace

namespace {

constexpr uint32_t kMagic    = 0x5154485Au;  // 'ZHTQ' little-endian
// v2: TQ_Entry grew from 32B -> 64B to carry forward+up unit vectors so the
// viewer's modal rotate (R) and gizmo can push pose updates, not just
// translation. Flags bit1 = HasFwdUp; older v1 viewers writing 32B entries
// won't be able to even open the MMF - our EnsureShared stamps fresh state
// when Magic/Version mismatch.
constexpr uint32_t kVersion  = 2u;
constexpr uint32_t kQueueCap = 16u;

#pragma pack(push, 1)
struct TQ_Entry {
    uint32_t Datum;
    uint32_t Flags;       // bit0 = HasPos, bit1 = HasFwdUp
    float    Px, Py, Pz;
    float    FwdX, FwdY, FwdZ;
    float    UpX,  UpY,  UpZ;
    uint32_t _reserved[5]; // pad to 64B for clean alignment
};
struct TQ_Shared {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter; // viewer-bumped
    uint32_t ReadCounter;  // DLL-bumped
    uint32_t Capacity;     // kQueueCap, sanity for the viewer
    uint32_t _pad[3];
    TQ_Entry Items[kQueueCap];
};
static_assert(sizeof(TQ_Entry) == 64, "TQ_Entry v2 layout");
static_assert(sizeof(TQ_Shared) == 32 + 64 * kQueueCap, "TQ_Shared v2 layout");
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_TransformQueue_Shared";

HANDLE      g_hMap   = nullptr;
TQ_Shared*  g_Shared = nullptr;

TQ_Shared* EnsureShared()
{
    if (g_Shared) return g_Shared;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(TQ_Shared), kMapName);
    if (!g_hMap) return nullptr;
    g_Shared = (TQ_Shared*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(TQ_Shared));
    if (!g_Shared) {
        CloseHandle(g_hMap);
        g_hMap = nullptr;
        return nullptr;
    }
    __try {
        if (g_Shared->Magic != kMagic || g_Shared->Version != kVersion) {
            ZeroMemory(g_Shared, sizeof(TQ_Shared));
            g_Shared->Magic    = kMagic;
            g_Shared->Version  = kVersion;
            g_Shared->Capacity = kQueueCap;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return g_Shared;
}

} // namespace

extern "C" __declspec(dllexport) void TransformQueueSnapshot_FramePumpTick()
{
    TQ_Shared* sh = EnsureShared();
    if (!sh) return;

    uint32_t writeCounter = 0;
    uint32_t readCounter  = 0;
    __try {
        writeCounter = sh->WriteCounter;
        readCounter  = sh->ReadCounter;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return; }

    if (writeCounter == readCounter) return;

    // How many new entries did the viewer publish? If the viewer outran
    // capacity, clamp to capacity and discard the older ones - this is the
    // correct behaviour for translate-gizmo drag (latest wins).
    uint32_t pending = writeCounter - readCounter;
    if (pending > kQueueCap) {
        readCounter = writeCounter - kQueueCap;
        pending     = kQueueCap;
    }

    for (uint32_t i = 0; i < pending; ++i) {
        uint32_t slot = (readCounter + i) % kQueueCap;
        TQ_Entry e{};
        __try { e = sh->Items[slot]; } __except (EXCEPTION_EXECUTE_HANDLER) { continue; }
        if (e.Datum == 0u || e.Datum == 0xFFFFFFFFu) continue;

        // FlagDelete (0x4) - replicate the body of haloreach!DeleteForgeObject
        // (skipping its permission gate). This removes the object from the
        // forge object array AND from the live engine state - what the user
        // wants when they press Delete. Calling Object_Delete alone leaves
        // a stale +0x1758 ref in the player monitor's TLS placement table
        // and the next monitor tick crashes; the full replica path cleans
        // those up. Player slot 0 is fine for the local-monitor-detach
        // call - if no monitor was holding the object, the detach is a
        // no-op.
        if (e.Flags & 0x4u) {
            ZH_Logf("[TransformQueue] forge-delete request datum=0x%08X\n", e.Datum);
            ReplicateDeleteForgeObject(/*playerSlot*/ 0u, e.Datum);
            continue; // Delete is exclusive - don't also pump pose for it.
        }

        // Pose path: at least one of HasPos / HasFwdUp must be set, else
        // the entry is a no-op and we ignore it.
        if (!(e.Flags & 0x3u)) continue;

        // Forward to the in-process ring with whatever flags the viewer set.
        // The existing pump handles host-byte authority + SEH protection so
        // we don't have to duplicate it here. Pose-only updates (HasFwdUp
        // without HasPos) work too - the engine's UpdatePosAndAiming reads
        // each ptr independently and skips null ones.
        __try {
            Enqueue_UpdateObjectPosAndAiming(
                e.Datum, e.Flags,
                e.Px, e.Py, e.Pz,
                e.FwdX, e.FwdY, e.FwdZ,
                e.UpX,  e.UpY,  e.UpZ,
                0ull, 0ull);
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
    }

    __try { sh->ReadCounter = writeCounter; } __except (EXCEPTION_EXECUTE_HANDLER) {}
}
