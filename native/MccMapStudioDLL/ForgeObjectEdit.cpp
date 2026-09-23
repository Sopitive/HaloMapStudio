// ForgeObjectEdit.cpp
// =============================================================================
// Handles edit requests from `HaloMapStudio_ForgeObjectEdit_Request`.
//
// On each viewer-issued TriggerCounter bump:
//   1. Locate the target forge slot via ForgeObjectTableSnapshot's resolver
//      (gives us the runtime ForgeObject base pointer for slot 0; offset by
//      forgeIdx * 76 to hit the requested slot).
//   2. Stamp the requested fields (per FieldMask) directly into the runtime
//      struct. Most fields take effect on the engine's next round-restart
//      tick; some (position/rotation) re-pose the rigid body when the
//      engine drains its forge command queue.
//   3. If a UpdateObjectData engine call is known (RE'd separately), call
//      it now so the runtime object instance is re-built without waiting
//      for a round restart. Until that RE lands, return kStatus_PartialSuccess
//      (fields stamped, engine not yet notified).
//
// Field-level granularity (FieldMask bits) lets the viewer push a position-
// only update every drag-frame at 30 Hz without rewriting unrelated bytes,
// then follow up with a full-field commit on drag-release.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"

#include <windows.h>
#include <cstdint>
#include <cstring>

extern "C" void ZH_Logf(const char* fmt, ...);
extern "C" uint8_t* HaloMapStudio_ForgeObjectTable_GetBase();

// =============================================================================
// UpdateObjectData engine call.
// Signature: void __fastcall UpdateObjectData(uint64_t playerKey, uint64_t datum,
//                                              uint64_t* payload[8 qwords / 64 B]);
// Located at haloreach.dll + 0x771BC.
// =============================================================================
using UpdateObjectData_Fn = void(__fastcall*)(uint64_t, uint64_t, uint64_t*);
static constexpr uintptr_t kRva_UpdateObjectData = 0x000771BCu;
static constexpr uintptr_t kRva_SelfPlayerIndex  = 0x02D596C8u;  // *u32

// =============================================================================
// Anti-garbage-collection: Object_SetGarbageCollectionEnabled
//
// Halo Reach's "candy monitor" garbage-collects forge objects that have been
// moved/disturbed. It starts an abandonment timer when the object is disturbed
// and deletes it when the timer exceeds the GC time threshold. The engine
// function `Object_SetGarbageCollectionEnabled` (haloreachnew+0x47C5F8)
// toggles bit 0 of s_object_data+0x143:
//
//   if (enable) byte |= 0x01;   // never garbage-collected
//   else        byte &= 0xFE;   // can be garbage-collected
//
// When HaloMapStudio moves an object via MoveObjectRelative or UpdateObjectData,
// the engine's candy monitor sees the object as "disturbed" and starts its
// abandonment countdown (default ~30 seconds). We stamp the NoGC bit after
// every edit to prevent the despawn.
//
// We resolve s_object_data from the datum ourselves rather than calling the
// engine function - the engine function's param_1 is an object_index (the
// datum's low 16 bits), not the full datum, and it does its own
// Object_TryGetByTypeMask lookup. Direct stamping is cheaper and avoids
// re-entering the engine's object system.
// =============================================================================
static constexpr size_t kObjData_NoGcFlags = 0x143;  // s_object_data byte
static constexpr uint8_t kNoGcBit          = 0x01;   // bit 0 = "never garbage"

// Object_SetScale (haloreach.dll+0x46cc34) - engine entrypoint that writes
// the live-object scale field (s_object_data+0x80 current, +0x84 target,
// +0xE8 interp duration) and calls MarkObjectAndAncestors so the new scale
// replicates over the network. THIS is what makes SCALE-label objects
// visibly resize in the running game; without it our viewer scales the
// viewport mesh but the in-game model stays at 1x until the SCALE-label
// gametype script applies it at match start.
//
// Signature (decompiled from haloreachnew):
//   void Object_SetScale(uint32_t datum, float scale, uint32_t durationFloatBits);
//
// arg3 is interpreted as a FLOAT (via `(arg3 ^ DAT_1809da3f0) < 0` - 
// xor-flips the sign bit, so a positive float arg means "smooth interp over
// N ticks"; arg3 = 0 or any non-positive float means "snap immediately").
// We always pass 0 - there's no UX win to interpolation here.
using ObjectSetScale_Fn = void(__fastcall*)(uint32_t, float, uint32_t);
static constexpr uintptr_t kRva_ObjectSetScale = 0x0046CC34u;

// Object_SetTeam (haloreach.dll+0x14C1E8) - touches the runtime instance and
// rebinds the model atlases for the new team colour. UpdateObjectData by
// itself persists the team byte into the forge slot + reapplies transforms +
// pokes the UI weapon-slot counter, but does NOT re-bind materials on the
// live mesh, so the change is invisible until a round restart. See
// FORGE_UPDATE_OBJECT_RE.md section 4 "Team - runtime instance only".
using ObjectSetTeam_Fn = void(__fastcall*)(uint32_t, uint32_t);
static constexpr uintptr_t kRva_ObjectSetTeam = 0x0014C1E8u;

// MoveObjectRelative (haloreach.dll+0x46CB38) - pose the runtime instance
// from the supplied pos/fwd/up. Used for transform-only edits where we
// don't trigger UpdateObjectData (skipped during live drag to avoid network
// churn). Without this, position/rotation writes hit only the slot bytes
// and the engine doesn't re-pose the visible mesh until next round.
using MoveObjectRelative_Fn = void(__fastcall*)(uint64_t, float*, float*, float*, uint64_t, uint8_t);
static constexpr uintptr_t kRva_MoveObjectRelative = 0x0046CB38u;

// ForgeBlock_WriteAndNotify (haloreach.dll+0x6E638) - lower-level function
// called internally by UpdateObjectData. Writes the 0x4C slot buffer back
// to the session table and triggers ForgeBlock_NotifyWeaponSync +
// ForgeBlock_ReapplyRuntimeTransform (re-poses the runtime instance from
// the slot's pos/fwd/up). Unlike UpdateObjectData, this function does NOT
// have a Forge_HasPermissionToEditObject gate - it always executes.
//
// Signature (from Ghidra):
//   void ForgeBlock_WriteAndNotify(void* sessionCtx, uint32_t forgeIndex, void* slotBuf76);
//
// We call this as a FALLBACK after UpdateObjectData to ensure the engine's
// internal notifiers (weapon sync, transform reapply) fire even when the
// permission gate silently blocks UpdateObjectData. The permission gate
// requires the player to be in a valid respawn/monitor state AND own the
// object - conditions that may not hold when HaloMapStudio edits externally.
using ForgeBlockWriteAndNotify_Fn = void(__fastcall*)(void*, uint32_t, void*);
static constexpr uintptr_t kRva_ForgeBlockWriteAndNotify = 0x0006E638u;

// Forge_GetSessionContextBase (haloreach.dll+0x6BD30) - returns the session
// context pointer (the root of the forge slot array). Online uses
// Network_GetCurrentSessionContext; offline uses DAT_184E2FCC0. Needed for
// the ForgeBlock_WriteAndNotify call.
using ForgeGetSessionCtx_Fn = void*(__fastcall*)();
static constexpr uintptr_t kRva_ForgeGetSessionCtx = 0x0006BD30u;

namespace {

// Resolve s_object_data* from a datum handle (salt<<16 | index), using the
// same TLS-based object pool descriptor as EngineThreadResolver.h.
// Returns nullptr if the datum is invalid, the pool can't be read, or the
// entry is inactive.
uint8_t* ResolveObjectDataPtr(uint32_t datum)
{
    if (datum == 0 || datum == 0xFFFFFFFFu) return nullptr;

    using namespace HaloMapStudio::Engine;
    uint8_t* desc = nullptr;
    __try { desc = ResolveObjectPoolDescriptor(); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    if (!desc) return nullptr;

    uint32_t entrySize = 0;
    uint32_t maxCount  = 0;
    void*    rawBase   = nullptr;
    if (!SafeReadT(desc + kDesc_EntrySize, entrySize) || entrySize == 0) return nullptr;
    if (!SafeReadT(desc + kDesc_MaxCount,  maxCount)  || maxCount  == 0) return nullptr;
    if (!SafeReadPtr(desc + kDesc_Entries, rawBase)   || !rawBase)        return nullptr;

    uint32_t idx = datum & 0xFFFFu;
    if (idx >= maxCount) return nullptr;

    uint8_t* entry = (uint8_t*)rawBase + (size_t)idx * (size_t)entrySize;
    uint16_t salt = 0;
    if (!SafeReadT(entry + kEntry_Salt, salt)) return nullptr;
    if (salt != (uint16_t)(datum >> 16)) return nullptr;  // stale datum

    void* obj = nullptr;
    if (!SafeReadPtr(entry + kEntry_ObjPtr, obj) || !obj) return nullptr;
    return (uint8_t*)obj;
}

// Stamp the "never garbage collected" bit on s_object_data+0x143.
// Prevents the candy monitor from despawning the object after its
// abandonment timer fires. Safe to call every edit - the OR is idempotent.
void StampAntiGarbageCollection(uint32_t datum)
{
    uint8_t* objData = ResolveObjectDataPtr(datum);
    if (!objData) return;

    __try {
        uint8_t cur = *(objData + kObjData_NoGcFlags);
        if ((cur & kNoGcBit) == 0) {
            *(objData + kObjData_NoGcFlags) = cur | kNoGcBit;
            ZH_Logf("[ForgeEdit] stamped NoGC bit on datum=0x%08X (obj+0x143: 0x%02X -> 0x%02X)\n",
                    datum, (unsigned)cur, (unsigned)(cur | kNoGcBit));
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        ZH_Logf("[ForgeEdit] NoGC stamp faulted for datum=0x%08X\n", datum);
    }
}

UpdateObjectData_Fn ResolveUpdateObjectData()
{
    static UpdateObjectData_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    cached = reinterpret_cast<UpdateObjectData_Fn>((uint8_t*)mod + kRva_UpdateObjectData);
    return cached;
}

ObjectSetTeam_Fn ResolveObjectSetTeam()
{
    static ObjectSetTeam_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    cached = reinterpret_cast<ObjectSetTeam_Fn>((uint8_t*)mod + kRva_ObjectSetTeam);
    return cached;
}

MoveObjectRelative_Fn ResolveMoveObjectRelative()
{
    static MoveObjectRelative_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    cached = reinterpret_cast<MoveObjectRelative_Fn>((uint8_t*)mod + kRva_MoveObjectRelative);
    return cached;
}

ObjectSetScale_Fn ResolveObjectSetScale()
{
    static ObjectSetScale_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    cached = reinterpret_cast<ObjectSetScale_Fn>((uint8_t*)mod + kRva_ObjectSetScale);
    return cached;
}

ForgeBlockWriteAndNotify_Fn ResolveForgeBlockWriteAndNotify()
{
    static ForgeBlockWriteAndNotify_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    cached = reinterpret_cast<ForgeBlockWriteAndNotify_Fn>((uint8_t*)mod + kRva_ForgeBlockWriteAndNotify);
    return cached;
}

ForgeGetSessionCtx_Fn ResolveForgeGetSessionCtx()
{
    static ForgeGetSessionCtx_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    cached = reinterpret_cast<ForgeGetSessionCtx_Fn>((uint8_t*)mod + kRva_ForgeGetSessionCtx);
    return cached;
}

uint8_t ReadSelfPlayerIndex()
{
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return 0;
    uint32_t v = 0;
    __try { v = *(uint32_t*)((uint8_t*)mod + kRva_SelfPlayerIndex); }
    __except (EXCEPTION_EXECUTE_HANDLER) { v = 0; }
    return (uint8_t)(v & 0x0F);
}

// Host-byte resolver: TLS block @ engine TLS slot, then +0x48 = node,
// then node+0x11 = host byte. Must be 0x05 across the UpdateObjectData call.
uint8_t* ResolveHostBytePtr()
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
    // Probe-read to verify reachability.
    uint8_t tmp = 0;
    __try { tmp = *hostByte; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return nullptr; }
    (void)tmp;
    return hostByte;
}

// Combined commit. Inside a single host-byte-flip block:
//   1. UpdateObjectData - persist trailing 28 bytes of slot, fire
//                               ForgeBlock_NotifyWeaponSync +
//                               ForgeBlock_ReapplyRuntimeTransform.
//   2. Object_SetTeam (opt.) - rebind the live mesh's team colour materials.
//                               Only called when teamForRebind != kNoTeam.
//   3. MoveObjectRelative(opt) - pose the live instance from the supplied
//                               world pos/fwd/up. Only called when
//                               doMove != 0; used for transform-only edits
//                               (where we skip UpdateObjectData to avoid
//                               network churn during a 30Hz drag).
//
// All under one host-byte flip so the gate is open during ALL three calls.
// Restoring once at the end is the right shape - we never want the host
// byte left flipped on exit (anti-cheat tracks it).
constexpr uint32_t kNoTeam = 0xFFFFFFFFu;
constexpr float    kNoScale = -1.0f;       // sentinel: skip Object_SetScale

bool CallEngineCommit(
    uint32_t objectDatum,
    const uint8_t* slot76OrNull,       // null skips UpdateObjectData
    uint32_t teamForRebind,            // kNoTeam skips Object_SetTeam
    const float* posOrNull,            // nullptr skips MoveObjectRelative pos arg
    const float* fwdOrNull,
    const float* upOrNull,
    bool doMove,                       // true to call MoveObjectRelative at all
    float liveScale,                   // kNoScale or <=0 skips Object_SetScale
    uint32_t forgeIdx = 0xFFFFFFFFu)   // forge slot index for WriteAndNotify fallback
{
    UpdateObjectData_Fn   fnUpdate = slot76OrNull ? ResolveUpdateObjectData() : nullptr;
    ObjectSetTeam_Fn      fnTeam   = (teamForRebind != kNoTeam) ? ResolveObjectSetTeam() : nullptr;
    MoveObjectRelative_Fn fnMove   = doMove ? ResolveMoveObjectRelative() : nullptr;
    ObjectSetScale_Fn     fnScale  = (liveScale > 0.0f) ? ResolveObjectSetScale() : nullptr;

    // Nothing to do.
    if (!fnUpdate && !fnTeam && !fnMove && !fnScale) return false;

    // Build 64-byte payload only if UpdateObjectData is in play. Game reads
    // only the first 28 bytes; the rest stays zero.
    uint64_t payloadQ[8] = {};
    if (slot76OrNull) memcpy(payloadQ, slot76OrNull + 0x30, 0x1C);

    uint8_t* hostByte = ResolveHostBytePtr();
    uint8_t prev = 0;
    bool havePrev = false, didFlip = false;
    if (hostByte) {
        __try {
            prev = *hostByte;
            havePrev = true;
            if (prev != 0x05) { *hostByte = 0x05; didFlip = true; }
        } __except (EXCEPTION_EXECUTE_HANDLER) { hostByte = nullptr; }
    }

    // Diagnostic: capture host-byte values + which calls actually fired.
    // Logged so the user can correlate "panel commit -> in-game change" or
    // failures. Without this we can't tell if UpdateObjectData's permission
    // gate is silently rejecting (in which case the slot bytes are stamped
    // but the engine never propagates).
    uint8_t hostByteBefore = hostByte ? prev : 0xFF;
    uint8_t hostByteFlipped = (hostByte && didFlip) ? 0x05 : hostByteBefore;
    uint64_t selfPK = 0xEC700000ull | (uint64_t)ReadSelfPlayerIndex();

    bool ok = false;
    bool firedUpdate = false, firedTeam = false, firedMove = false, firedScale = false;
    bool faultedUpdate = false, faultedTeam = false, faultedMove = false, faultedScale = false;
    __try {
        // 1. UpdateObjectData persists the slot AND re-applies the slot's
        //    pos/fwd/up via ForgeBlock_ReapplyRuntimeTransform internally,
        //    so the explicit MoveObjectRelative below is only needed for
        //    transform-only edits where slot76OrNull is null.
        if (fnUpdate) {
            __try {
                fnUpdate(selfPK, (uint64_t)objectDatum, payloadQ);
                firedUpdate = true;
            } __except (EXCEPTION_EXECUTE_HANDLER) { faultedUpdate = true; }
        }

        // 1b. ForgeBlock_WriteAndNotify fallback - bypass the permission gate.
        //
        // UpdateObjectData internally calls Forge_HasPermissionToEditObject,
        // which silently returns (no-op) when the player isn't in a valid
        // forge-monitor state or doesn't own the object. When that happens
        // the slot bytes are already stamped (we wrote them above), but the
        // engine's notifiers (ForgeBlock_NotifyWeaponSync,
        // ForgeBlock_ReapplyRuntimeTransform) never fire - the live object
        // doesn't visually update and the variant isn't persisted.
        //
        // ForgeBlock_WriteAndNotify is the lower-level function that
        // UpdateObjectData calls AFTER the permission check. It takes the
        // session context + forge index + a 76-byte slot buffer, writes it
        // back, and fires the notifiers unconditionally. We call it here as
        // an unconditional fallback to guarantee the notifiers run.
        //
        // This is safe to call even when UpdateObjectData succeeded - it
        // just re-writes the same bytes and re-fires the same notifiers
        // (idempotent from the engine's perspective, since the slot data
        // hasn't changed between the two calls within this same frame).
        if (slot76OrNull && forgeIdx != 0xFFFFFFFFu && forgeIdx < 650u) {
            auto fnWriteNotify = ResolveForgeBlockWriteAndNotify();
            auto fnGetCtx      = ResolveForgeGetSessionCtx();
            if (fnWriteNotify && fnGetCtx) {
                __try {
                    void* sessCtx = fnGetCtx();
                    if (sessCtx) {
                        fnWriteNotify(sessCtx, forgeIdx, (void*)slot76OrNull);
                        ZH_Logf("[ForgeEdit] WriteAndNotify fallback fired "
                                "sessCtx=%p forgeIdx=%u\n", sessCtx, forgeIdx);
                    }
                } __except (EXCEPTION_EXECUTE_HANDLER) {
                    ZH_Logf("[ForgeEdit] WriteAndNotify fallback faulted "
                            "forgeIdx=%u\n", forgeIdx);
                }
            }
        }

        // 2. Team material rebind. UpdateObjectData wrote the team byte to
        //    the slot already (if Team bit was in the field mask); this
        //    call is what makes the visible mesh's materials swap.
        //    First arg is the runtime datum (RCX), second is the new
        //    team index 0..15 (RDX).
        if (fnTeam) {
            __try {
                fnTeam(objectDatum, teamForRebind & 0xFu);
                firedTeam = true;
            } __except (EXCEPTION_EXECUTE_HANDLER) { faultedTeam = true; }
        }

        // 3. Live pose re-apply for transform-only edits. Pass NULL for any
        //    axis we don't want to change (per FORGE_UPDATE_OBJECT_RE.md
        //    section 4). When UpdateObjectData ran, this is redundant - but harm-
        //    less, and we gate on doMove so the call site can opt out.
        if (fnMove) {
            __try {
                fnMove((uint64_t)objectDatum,
                       (float*)posOrNull, (float*)fwdOrNull, (float*)upOrNull,
                       0, 0);
                firedMove = true;
            } __except (EXCEPTION_EXECUTE_HANDLER) { faultedMove = true; }
        }

        // 4. Live scale apply - pushes the SCALE-gtLabel-overloaded scale
        //    into the runtime s_object_data so the in-game mesh resizes on
        //    the same frame the user changes label/spawnSeq. Without this
        //    only our viewer's mesh resizes; the game model stays at 1x
        //    until the gametype script applies it at match start.
        //    Third arg 0 = immediate snap (no interpolation).
        if (fnScale) {
            __try {
                fnScale(objectDatum, liveScale, 0);
                firedScale = true;
            } __except (EXCEPTION_EXECUTE_HANDLER) { faultedScale = true; }
        }

        // 5. Anti-GC stamp: set the "never garbage collected" bit on the
        //    runtime s_object_data so the candy monitor doesn't despawn the
        //    object after our external edit "disturbs" it.  This is the
        //    equivalent of the megalo `object_set_never_garbage(obj, true)`
        //    script action.  Idempotent - the OR is harmless if already set.
        //    Runs inside the SEH block because it resolves the object pool
        //    descriptor which may fault on map-switch boundaries.
        __try { StampAntiGarbageCollection(objectDatum); }
        __except (EXCEPTION_EXECUTE_HANDLER) {}

        ok = true;
    } __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }

    // Read host-byte AFTER to verify nothing else stepped on it mid-call.
    uint8_t hostByteAfterCalls = hostByte ? *hostByte : 0xFF;

    ZH_Logf("[ForgeEdit] commit datum=0x%08X playerKey=0x%llX host=(pre=%u flipped=%u afterCalls=%u) "
            "fnUpdate=%p fired=%d fault=%d / "
            "fnTeam=%p team=%u fired=%d fault=%d / "
            "fnMove=%p fired=%d fault=%d / "
            "fnScale=%p scale=%.3f fired=%d fault=%d / ok=%d\n",
            objectDatum, (unsigned long long)selfPK,
            (unsigned)hostByteBefore, (unsigned)hostByteFlipped, (unsigned)hostByteAfterCalls,
            (void*)fnUpdate, firedUpdate ? 1 : 0, faultedUpdate ? 1 : 0,
            (void*)fnTeam, teamForRebind & 0xFu, firedTeam ? 1 : 0, faultedTeam ? 1 : 0,
            (void*)fnMove, firedMove ? 1 : 0, faultedMove ? 1 : 0,
            (void*)fnScale, liveScale, firedScale ? 1 : 0, faultedScale ? 1 : 0,
            ok ? 1 : 0);

    // Restore the host byte unconditionally (even on fault) - leaving it
    // flipped breaks every subsequent host-gate check in the engine.
    if (hostByte && havePrev && didFlip) {
        __try { *hostByte = prev; } __except (EXCEPTION_EXECUTE_HANDLER) {}
    }
    return ok;
}

// Back-compat thin wrapper - used by older call sites that only need the
// full Forge edit commit (no team rebind, slot-driven transform reapply).
bool CallUpdateObjectData(uint32_t objectDatum, const uint8_t* slot76)
{
    return CallEngineCommit(objectDatum, slot76, kNoTeam,
                            nullptr, nullptr, nullptr, /*doMove=*/false,
                            kNoScale);
}

} // namespace

namespace {

constexpr uint32_t kMagic        = 0x45474F46u; // 'FOGE'
// v3: a ring of kCapacity entries drained whole each frame pump so a host can
// apply hundreds of edits in ONE frame.
constexpr uint32_t kVersion      = 3u;
constexpr uint32_t kCapacity     = 512u;
constexpr uint32_t kMaxForgeIdx  = 650u;
constexpr size_t   kForgeObjSize = 76u;

// Mirrors the hms-ipc ForgeObjectEditField.
constexpr uint32_t kField_Position    = 1u << 0;
constexpr uint32_t kField_Rotation    = 1u << 1;
constexpr uint32_t kField_Shape       = 1u << 2;
constexpr uint32_t kField_Physics     = 1u << 3;
constexpr uint32_t kField_Symmetry    = 1u << 4;
constexpr uint32_t kField_HideAtStart = 1u << 5;
constexpr uint32_t kField_GameSpecific= 1u << 6;
constexpr uint32_t kField_Team        = 1u << 7;
constexpr uint32_t kField_Color       = 1u << 8;
constexpr uint32_t kField_SpawnSeq    = 1u << 9;
constexpr uint32_t kField_SpawnTime   = 1u << 10;
constexpr uint32_t kField_GtLabel     = 1u << 11;
constexpr uint32_t kField_OtherInfoA  = 1u << 12;
constexpr uint32_t kField_OtherInfoB  = 1u << 13;
constexpr uint32_t kField_IdExt       = 1u << 14;
constexpr uint32_t kField_SpawnRelMap = 1u << 15;
constexpr uint32_t kField_Show        = 1u << 16;
constexpr uint32_t kField_LiveScale   = 1u << 17;

// Mirrors the hms-ipc ForgeObjectEditStatus.
constexpr uint32_t kStatus_Idle          = 0u;
constexpr uint32_t kStatus_Success       = 1u;
constexpr uint32_t kStatus_InvalidSlot   = 2u;
constexpr uint32_t kStatus_NoLiveObject  = 3u;
constexpr uint32_t kStatus_EngineNotReady= 4u;
constexpr uint32_t kStatus_FieldStampFail= 5u;
constexpr uint32_t kStatus_UpdateCallFail= 6u;
constexpr uint32_t kStatus_PartialSuccess= 7u;
constexpr uint32_t kStatus_NotImpl       = 99u;

// Bit masks within the 76-byte struct.
constexpr uint8_t kFlagBits_PhysicsMask  = 0b11000000;
constexpr uint8_t kFlagBits_GameSpecific = 0b00100000;
constexpr uint8_t kFlagBits_SymmetryMask = 0b00001100;
constexpr uint8_t kFlagBits_HideAtStart  = 0b00000010;

#pragma pack(push, 1)
struct Header {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteSeq;    // viewer increments per entry written
    uint32_t ReadSeq;     // we increment per entry processed
    uint32_t Capacity;    // == kCapacity
    uint32_t LastStatus;  // last ForgeObjectEditStatus
    uint32_t _h0;
    uint32_t _h1;
};
struct Entry {
    uint32_t ForgeIdx;
    uint32_t FieldMask;
    uint32_t EngineDatumHint;
    uint32_t _Pad0;

    uint16_t Show;
    uint16_t ItemCategory;
    uint32_t IdExt;
    float    PosX, PosY, PosZ;
    float    FwdX, FwdY, FwdZ;
    float    UpX,  UpY,  UpZ;
    uint16_t SpawnRelativeToMapIndex;
    uint8_t  ItemVariant;
    uint8_t  _Pad1;
    float    Width, Length, Top, Bottom;
    uint8_t  Shape;
    int8_t   SpawnSequence;
    uint8_t  SpawnTime;
    uint8_t  CachedType;
    uint16_t GtLabelIndex;
    uint8_t  Flags;
    uint8_t  Team;
    uint8_t  OtherInfoA;
    uint8_t  OtherInfoB;
    uint8_t  Color;
    uint8_t  _Pad2;
    uint32_t _Pad3;

    // SCALE-gtLabel live-scale value (drives Object_SetScale). Only consumed
    // when (FieldMask & kField_LiveScale). Values <= 0 are "no change".
    float    LiveScale;
    uint32_t _PadLiveScale;
};
#pragma pack(pop)

const wchar_t kMapName[] = L"HaloMapStudio_ForgeObjectEdit_Request";

static const size_t kMmfSize = sizeof(Header) + (size_t)kCapacity * sizeof(Entry);

HANDLE   g_FileMapping = nullptr;
Header*  g_Header      = nullptr;
Entry*   g_Ring        = nullptr;

bool EnsureMmf()
{
    if (g_Header) return true;
    g_FileMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        0, (DWORD)kMmfSize, kMapName);
    if (!g_FileMapping) return false;
    uint8_t* base = (uint8_t*)MapViewOfFile(g_FileMapping, FILE_MAP_ALL_ACCESS, 0, 0, kMmfSize);
    if (!base) { CloseHandle(g_FileMapping); g_FileMapping = nullptr; return false; }
    g_Header = (Header*)base;
    g_Ring   = (Entry*)(base + sizeof(Header));
    if (g_Header->Magic != kMagic || g_Header->Version != kVersion ||
        g_Header->Capacity != kCapacity) {
        memset(base, 0, kMmfSize);
        g_Header->Magic = kMagic;
        g_Header->Version = kVersion;
        g_Header->Capacity = kCapacity;
    }
    return true;
}

// SEH-safe field stamps. Each one writes a tiny region of the runtime
// ForgeObject struct. We never write the WHOLE struct in one shot - that
// risks tearing a 76-byte memcpy against a live engine read.
bool StampU8 (uint8_t* base, size_t off, uint8_t  v) { __try { base[off] = v; return true; } __except(EXCEPTION_EXECUTE_HANDLER){ return false; } }
bool StampU16(uint8_t* base, size_t off, uint16_t v) { __try { *(uint16_t*)(base + off) = v; return true; } __except(EXCEPTION_EXECUTE_HANDLER){ return false; } }
bool StampU32(uint8_t* base, size_t off, uint32_t v) { __try { *(uint32_t*)(base + off) = v; return true; } __except(EXCEPTION_EXECUTE_HANDLER){ return false; } }
bool StampF32(uint8_t* base, size_t off, float    v) { __try { *(float*)(base + off) = v; return true; } __except(EXCEPTION_EXECUTE_HANDLER){ return false; } }

bool StampMaskedU8(uint8_t* base, size_t off, uint8_t mask, uint8_t v)
{
    __try {
        uint8_t cur = base[off];
        base[off] = (uint8_t)((cur & ~mask) | (v & mask));
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

uint32_t HandleOneRequest(const Entry* req)
{
    if (req->ForgeIdx >= kMaxForgeIdx) return kStatus_InvalidSlot;

    uint8_t* tableBase = HaloMapStudio_ForgeObjectTable_GetBase();
    if (!tableBase) return kStatus_EngineNotReady;

    uint8_t* slot = tableBase + (size_t)req->ForgeIdx * kForgeObjSize;
    uint32_t mask = req->FieldMask;
    if (mask == 0) return kStatus_Success;

    bool anyFault = false;

    if (mask & kField_Position) {
        if (!StampF32(slot,  8, req->PosX)) anyFault = true;
        if (!StampF32(slot, 12, req->PosY)) anyFault = true;
        if (!StampF32(slot, 16, req->PosZ)) anyFault = true;
    }
    if (mask & kField_Rotation) {
        if (!StampF32(slot, 20, req->FwdX)) anyFault = true;
        if (!StampF32(slot, 24, req->FwdY)) anyFault = true;
        if (!StampF32(slot, 28, req->FwdZ)) anyFault = true;
        if (!StampF32(slot, 32, req->UpX )) anyFault = true;
        if (!StampF32(slot, 36, req->UpY )) anyFault = true;
        if (!StampF32(slot, 40, req->UpZ )) anyFault = true;
    }
    if (mask & kField_Shape) {
        if (!StampF32(slot, 48, req->Width )) anyFault = true;
        if (!StampF32(slot, 52, req->Length)) anyFault = true;
        if (!StampF32(slot, 56, req->Top   )) anyFault = true;
        if (!StampF32(slot, 60, req->Bottom)) anyFault = true;
        if (!StampU8 (slot, 64, req->Shape )) anyFault = true;
    }
    if (mask & kField_Physics) {
        if (!StampMaskedU8(slot, 70, kFlagBits_PhysicsMask, req->Flags))
            anyFault = true;
    }
    if (mask & kField_Symmetry) {
        if (!StampMaskedU8(slot, 70, kFlagBits_SymmetryMask, req->Flags))
            anyFault = true;
    }
    if (mask & kField_HideAtStart) {
        if (!StampMaskedU8(slot, 70, kFlagBits_HideAtStart, req->Flags))
            anyFault = true;
    }
    if (mask & kField_GameSpecific) {
        if (!StampMaskedU8(slot, 70, kFlagBits_GameSpecific, req->Flags))
            anyFault = true;
    }
    if (mask & kField_Team)      if (!StampU8 (slot, 71, req->Team       )) anyFault = true;
    if (mask & kField_Color)     if (!StampU8 (slot, 74, req->Color      )) anyFault = true;
    if (mask & kField_SpawnSeq)  if (!StampU8 (slot, 65, (uint8_t)req->SpawnSequence)) anyFault = true;
    if (mask & kField_SpawnTime) if (!StampU8 (slot, 66, req->SpawnTime  )) anyFault = true;
    if (mask & kField_GtLabel)   if (!StampU16(slot, 68, req->GtLabelIndex)) anyFault = true;
    if (mask & kField_OtherInfoA)if (!StampU8 (slot, 72, req->OtherInfoA )) anyFault = true;
    if (mask & kField_OtherInfoB)if (!StampU8 (slot, 73, req->OtherInfoB )) anyFault = true;
    if (mask & kField_IdExt)     if (!StampU32(slot,  4, req->IdExt      )) anyFault = true;
    if (mask & kField_SpawnRelMap) if (!StampU16(slot, 44, req->SpawnRelativeToMapIndex)) anyFault = true;
    if (mask & kField_Show)      if (!StampU16(slot,  0, req->Show       )) anyFault = true;

    if (anyFault) return kStatus_FieldStampFail;

    // ----------------------------------------------------------------
    // Engine notify path (per FORGE_UPDATE_OBJECT_RE.md, +0x771BC).
    //
    // UpdateObjectData reads the trailing 28 bytes of the slot
    // (offsets 0x30..0x4C -> width/length/top/bottom/shape/spawnSeq/
    //  spawnTime/cachedType/gtLabel/flags/team/info/info/color) and:
    //   1. Persists those edits into the saved variant (round restart
    //      will replay them).
    //   2. Triggers ForgeBlock_ReapplyRuntimeTransform which re-poses
    //      the runtime instance from the slot's pos/fwd/up fields.
    // So calling it once per commit covers BOTH "behavior" fields (in
    // the 28-byte payload) AND "transform" fields (re-applied from the
    // slot bytes we already stamped above) in a single hop.
    //
    // We skip the call entirely for Position/Rotation-only edits where
    // the user might be live-dragging at 30Hz - repeated UpdateObjectData
    // calls during a drag cause network churn (the engine broadcasts
    // each edit to peers). On drag-release, the viewer's "Commit All"
    // button sends AllFields which DOES include behavior bits and so
    // triggers the notify.
    constexpr uint32_t kBehaviorMask =
        kField_Shape | kField_Physics | kField_Symmetry |
        kField_HideAtStart | kField_GameSpecific |
        kField_Team | kField_Color | kField_SpawnSeq |
        kField_SpawnTime | kField_GtLabel |
        kField_OtherInfoA | kField_OtherInfoB |
        kField_IdExt | kField_SpawnRelMap | kField_Show;
    bool wantsNotify = (mask & kBehaviorMask) != 0;
    bool transformOnly = !wantsNotify && (mask & (kField_Position | kField_Rotation)) != 0;
    // LiveScale is independent - it never touches the slot bytes (it just
    // pushes Object_SetScale into the engine), so it can ride along with
    // either path or stand alone. Pre-extract once.
    float liveScaleArg = ((mask & kField_LiveScale) && req->LiveScale > 0.0f)
                       ? req->LiveScale : kNoScale;
    // A scale-only edit (no behavior, no transform) needs an engine call
    // hop too - treat it like the transform-only path so we still reach
    // CallEngineCommit.
    bool scaleOnly = !wantsNotify && !transformOnly && (liveScaleArg > 0.0f);

    // Resolve the runtime datum once - needed for BOTH the full commit and
    // the transform-only MoveObjectRelative path.
    uint32_t targetDatum = req->EngineDatumHint;
    if (targetDatum == 0 || targetDatum == 0xFFFFFFFFu) {
        __try {
            uint32_t fromSlot = *(uint32_t*)(slot + 4);
            if (fromSlot != 0 && fromSlot != 0xFFFFFFFFu) targetDatum = fromSlot;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
    }
    if (targetDatum == 0 || targetDatum == 0xFFFFFFFFu) {
        // No live runtime instance - slot edits will still apply on next
        // round restart, but we can't notify the engine.
        return kStatus_PartialSuccess;
    }

    // -----------------------------------------------------------------
    // Path A: transform-only edit (no behavior bits in mask).
    //
    // The prior implementation here just returned kStatus_Success on the
    // (incorrect) assumption that the engine would pose-reapply on its
    // own. It does not - the visible mesh stayed at the old pose until
    // a round restart. Route through MoveObjectRelative so the runtime
    // instance pose updates immediately. Slot pos/fwd/up are already
    // stamped above, so map-variant export still captures the new pose.
    // -----------------------------------------------------------------
    if (transformOnly || scaleOnly) {
        float pos[3] = { req->PosX, req->PosY, req->PosZ };
        float fwd[3] = { req->FwdX, req->FwdY, req->FwdZ };
        float up [3] = { req->UpX,  req->UpY,  req->UpZ  };

        bool wantPos = (mask & kField_Position) != 0;
        bool wantRot = (mask & kField_Rotation) != 0;
        bool wantMove = wantPos || wantRot;

        bool ok = false;
        __try {
            ok = CallEngineCommit(
                targetDatum,
                /*slot76OrNull=*/nullptr,            // no UpdateObjectData
                /*teamForRebind=*/kNoTeam,
                wantPos ? pos : nullptr,
                wantRot ? fwd : nullptr,
                wantRot ? up  : nullptr,
                /*doMove=*/wantMove,
                liveScaleArg);
        } __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }
        return ok ? kStatus_Success : kStatus_UpdateCallFail;
    }

    // -----------------------------------------------------------------
    // Path B: behavior edit (full UpdateObjectData commit, optionally
    // with team material rebind).
    //
    // Snapshot the slot bytes so the engine call doesn't read torn data
    // mid-frame (the slot table is written by the engine itself; calling
    // UpdateObjectData with a stale pointer is fine, but reading inside
    // the engine's own write window can return torn fields).
    // -----------------------------------------------------------------
    uint8_t slotCopy[76];
    __try { memcpy(slotCopy, slot, 76); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return kStatus_FieldStampFail; }

    // If Team is in the mask, force the live mesh's materials to rebind
    // by calling Object_SetTeam after the persistence call. Without this
    // the new team byte sits in the slot but the visible mesh keeps the
    // old team colour until round restart.
    uint32_t teamForRebind = kNoTeam;
    if (mask & kField_Team) {
        teamForRebind = (uint32_t)req->Team;
    }

    bool ok = false;
    __try {
        ok = CallEngineCommit(
            targetDatum, slotCopy, teamForRebind,
            nullptr, nullptr, nullptr, /*doMove=*/false,
            liveScaleArg,
            req->ForgeIdx);
    } __except (EXCEPTION_EXECUTE_HANDLER) { ok = false; }

    return ok ? kStatus_Success : kStatus_UpdateCallFail;
}

} // namespace

// =============================================================================
// Per-frame entry - picks up viewer trigger-counter bumps and dispatches.
// =============================================================================
extern "C" void ForgeObjectEdit_FramePumpTick()
{
    if (!EnsureMmf()) return;

    // RING DRAIN - process EVERY queued entry (ReadSeq..WriteSeq) this frame.
    // As the host we can apply hundreds of edits per frame; the per-tick cap
    // is just a runaway guard. Each entry is processed via HandleOneRequest.
    uint32_t writeSeq = g_Header->WriteSeq;
    uint32_t readSeq  = g_Header->ReadSeq;
    if (readSeq == writeSeq) return;  // nothing pending

    // If the viewer overran the ring (more than kCapacity unread), skip the
    // lost entries - readSeq jumps to keep within one ring of writeSeq.
    if ((uint32_t)(writeSeq - readSeq) > kCapacity) {
        readSeq = writeSeq - kCapacity;
    }

    const uint32_t kMaxPerTick = kCapacity;  // drain the whole ring in one frame
    uint32_t processed = 0;
    while (readSeq != writeSeq && processed < kMaxPerTick) {
        const Entry* e = &g_Ring[readSeq % kCapacity];
        uint32_t status = kStatus_NotImpl;
        __try {
            status = HandleOneRequest(e);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            status = kStatus_FieldStampFail;
        }
        g_Header->LastStatus = status;
        if (status != kStatus_PartialSuccess && status != kStatus_Success) {
            ZH_Logf("[HaloMapStudioDLL] forge edit slot=%u mask=0x%X -> status=%u\n",
                    e->ForgeIdx, e->FieldMask, status);
        }
        ++readSeq;
        ++processed;
    }
    g_Header->ReadSeq = readSeq;
}
