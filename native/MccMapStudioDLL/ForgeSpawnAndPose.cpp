// ForgeSpawnAndPose.cpp
// =============================================================================
// Single-shot trigger MMF that spawns a forge palette item AT a viewer-
// supplied world pose (XYZ + forward + up vectors).
//
// Two-phase pipeline (engine has no one-call API for spawn-with-rotation - 
// see HaloMapStudio_RE notes on Forge_SpawnRequest_Execute @ 0x1802CC73C and
// the variant-command opcode table; opcode 0 carries position only, opcode
// 11 carries pose but is built from scnr palette baked data, not user input):
//
//   1. Snapshot the engine's object-datum set                      (pre-spawn)
//   2. Build a ForgeSpawnDescriptor on the stack, call Forge_SpawnRequest_
//      Execute. Object spawns at the player's monitor cursor location with
//      identity rotation.
//   3. Wait N frames for the spawn to land in the object table.
//   4. Diff the object-datum set; the new entry is our spawn.
//   5. Call Enqueue_UpdateObjectPosAndAiming(newDatum, hasPos|hasFwdUp,
//      targetXYZ, fwdXYZ, upXYZ, ...) to snap to the user-requested pose.
//
// Turn 1 (this file): plumb the MMF round-trip end-to-end. Frame-pump tick
// detects a TriggerCounter bump, ACKs with ResultStatus = kStatus_NotImpl
// (99) and bumps ResponseCounter. No engine call yet - that lands in turn 2.
// This proves the viewer <-> DLL transport works without risking a crash.
//
// MMF: HaloMapStudio_ForgeSpawnAndPose_Shared - 80 bytes.
//   Header (16 B):
//     u32 Magic           'MMSF' (0x4653 4D4D LE) - HaloMapStudio Forge
//     u32 Version         1
//     u32 TriggerCounter  viewer bumps to request a spawn
//     u32 ResponseCounter DLL bumps after handling (success or failure)
//   Request body (52 B):
//     u32 PaletteIndex
//     u32 EntryIndex
//     u32 VariantIndex
//     u32 _Pad0           (keeps the float block 8-byte aligned)
//     f32 X, Y, Z         target world position
//     f32 FwdX, FwdY, FwdZ target forward unit vector
//     f32 UpX,  UpY,  UpZ  target up unit vector
//   Response body (12 B):
//     u32 ResultStatus    see kStatus_* below
//     u32 SpawnedDatum    populated on success; 0 otherwise
//     u32 _Pad1           (alignment)
//
// Total: 16 + 52 + 12 = 80 bytes. Verified by static_assert.
// =============================================================================

#include "pch.h"
#include <windows.h>
#include <cstdint>
#include <cstring>
#include "EngineThreadResolver.h"

extern "C" void ZH_Logf(const char* fmt, ...);

// Live networked spawn-by-tag (SpawnDefinitionLive.cpp). Selected via
// the SpawnMethod field on the ForgeSpawnAndPose / WorldSpawn MMFs.
extern "C" uint32_t HaloMapStudio_SpawnDefinitionLive(
    uint32_t tagId, float x, float y, float z,
    float fx, float fy, float fz, float ux, float uy, float uz);

namespace {

// Engine entry-point resolver. Caches the function pointer across calls so we
// don't re-walk the module table per-tick. Returns nullptr if haloreach.dll
// isn't loaded yet (DLL injected before MCC mapped the engine module).
using VariantCommand_HandleExecution_Fn = void(*)(uint32_t* cmd);

VariantCommand_HandleExecution_Fn ResolveVariantCommandHandler()
{
    static VariantCommand_HandleExecution_Fn cached = nullptr;
    if (cached) return cached;
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return nullptr;
    cached = reinterpret_cast<VariantCommand_HandleExecution_Fn>(
        reinterpret_cast<uint8_t*>(hr) +
        HaloMapStudio::Engine::kRva_VariantCommand_HandleExecution);
    return cached;
}

// Build a 64-byte forge-command struct and dispatch it. Layout is documented
// in EngineThreadResolver.h next to kRva_VariantCommand_HandleExecution. This
// path mirrors what Forge_SpawnRequest_Execute @ haloreach+0x2CC73C does
// internally - except we supply the world position from MMF input rather than
// reading it out of the player's monitor TLS.
//
// Returns:
//   1 = command dispatched (spawn will land 1-2 ticks later when the queue
//       drains; we don't currently capture the spawned datum here)
//   2 = engine entry point unresolved (haloreach.dll not loaded)
//   3 = dispatcher faulted (SEH). Likely the gating preconditions weren't
//       met (e.g. user wasn't in forge mode, or scnr palette indices are
//       invalid for the loaded scenario).
// FORGE_MONITOR_GRAB_V2: the engine auto-grabs each forge-spawned
// object into the spawning monitor (writes monitor+0x1758 = newDatum). MMS does
// NOT want its programmatic spawns left grabbed (otherwise the monitor ends
// up holding every MMS-spawned object).
//
// Why V1 failed: V1 just wrote monitor+0x1758 = 0xFFFFFFFF once. But the
// per-tick forge network-state applier (haloreach FUN_180074768, runs every
// frame) reads the network-replicated "should-be-held" datum and RE-APPLIES the
// grab via the grab core FUN_180076B10 whenever the live field disagrees. The
// spawn registers the object as held in network/replication state, so a
// fire-once field poke is reverted on the very next tick. (Verified by
// decompiling FUN_180074768 - it loops slots 0..0xF and calls FUN_180076B10 to
// restore the grab.)
//
// V2 fix: call the engine's OWN release primitive Forge_ReleaseHeldObject
// (FUN_180077068). It verifies heldDatum == monitor[slot]+0x1758, then
// PLACES the object (mode 0, not delete) and tears down the network-side held
// record - exactly how FUN_180074768 releases when the network says "nothing
// held". With the network state cleared, the applier stops re-grabbing, so the
// release is permanent rather than a one-frame flicker.
//
// Timing/robustness: the spawn only QUEUES here and the grab lands 1-2 ticks
// later (sometimes more under load), so a 6-tick fire-once window could expire
// before the grab even appeared. V2 keeps the window OPEN until it actually
// observes a held object, then releases it and closes the window. The window is
// generous (~30 ticks ~= 0.5 s) so queue-drain jitter can't make us miss the
// grab. Because the window is short and only armed by MMS spawns, a genuine
// human monitor grab outside that window is never touched.
volatile long g_PendingMonitorClearTicks = 0;

using Forge_ReleaseHeldObject_Fn = void (*)(uint32_t slot, uint32_t datum,
                                            uint64_t r8, uint64_t r9);

// Returns true once a held object was found AND released (so the caller can
// close the suppression window); false if nothing was held this tick (keep the
// window open and retry next frame).
static bool ReleaseMonitorHeldObject(uint32_t slot)
{
    using namespace HaloMapStudio::Engine;

    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return false;
    uint8_t* base = reinterpret_cast<uint8_t*>(hr);

    uint32_t tlsIndex = 0;
    if (!SafeReadT(base + kRva_MonitorTlsIndex, tlsIndex)) return false;
    if (tlsIndex == 0u || tlsIndex == 0xFFFFFFFFu) return false;

    void** tlsArray = GetTLSArrayBase();
    if (!tlsArray) return false;

    void* tlsBlock = nullptr;
    __try { tlsBlock = tlsArray[tlsIndex]; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    if (!tlsBlock) return false;

    void* ctxBase = nullptr;
    if (!SafeReadPtr(reinterpret_cast<uint8_t*>(tlsBlock) + kMonitorCtxOffset, ctxBase)
        || !ctxBase) return false;

    uint8_t* monitor = reinterpret_cast<uint8_t*>(ctxBase) + (size_t)slot * kMonitorStride;

    uint32_t held = 0xFFFFFFFFu;
    __try { held = *reinterpret_cast<volatile uint32_t*>(monitor + kMonitor_HeldObject); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
    if (held == 0xFFFFFFFFu) return false;  // nothing latched yet - keep waiting

    // Drive the engine's canonical release so the network-state applier won't
    // re-grab on the next tick. Falls back to a direct field clear if the
    // engine call faults (e.g. preconditions not met).
    auto releaseFn = reinterpret_cast<Forge_ReleaseHeldObject_Fn>(
        base + kRva_Forge_ReleaseHeldObject);
    bool engineReleased = false;
    __try {
        releaseFn(slot, held, 0, 0);
        engineReleased = true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        engineReleased = false;
    }

    // Belt-and-suspenders: stamp the sentinel too. If the engine release
    // already cleared it this is a redundant no-op; if the engine call faulted
    // this at least drops the visual for this frame.
    __try {
        *reinterpret_cast<volatile uint32_t*>(monitor + kMonitor_HeldObject) = 0xFFFFFFFFu;
        *reinterpret_cast<volatile uint8_t* >(monitor + kMonitor_GrabState)  = 0u;
    } __except (EXCEPTION_EXECUTE_HANDLER) {}

    ZH_Logf("[ForgeSpawnAndPose] released monitor[%u] auto-grab (held=0x%08X, "
            "engineRelease=%d) -> none\n",
            slot, held, engineReleased ? 1 : 0);
    return true;
}

uint32_t DispatchSpawnAtPosition(
    uint32_t paletteIndex, uint32_t entryIndex, uint32_t variantIndex,
    float x, float y, float z,
    uint32_t monitorSlot)
{
    auto fn = ResolveVariantCommandHandler();
    if (!fn) return 2;

    // 64-byte command struct, zero-initialised so the 0x18..0x38 region (which
    // the engine doesn't read for opcode 0 but might validate later) is clean.
    alignas(8) uint8_t cmd[0x40] = {};
    *reinterpret_cast<uint32_t*>(cmd + 0x00) = paletteIndex;
    *reinterpret_cast<uint32_t*>(cmd + 0x04) = entryIndex;
    *reinterpret_cast<uint32_t*>(cmd + 0x08) = variantIndex;
    *reinterpret_cast<float*   >(cmd + 0x0C) = x;
    *reinterpret_cast<float*   >(cmd + 0x10) = y;
    *reinterpret_cast<float*   >(cmd + 0x14) = z;
    *reinterpret_cast<uint32_t*>(cmd + 0x38) = monitorSlot & 0xFFFFu;
    *reinterpret_cast<uint16_t*>(cmd + 0x3C) = 0u;  // opcode 0 = spawn-at-pos

    __try {
        fn(reinterpret_cast<uint32_t*>(cmd));
        // FORGE_MONITOR_GRAB_V2: arm the post-spawn release WINDOW. The grab
        // lands 1-2 ticks after this queues (more under load); the tick
        // consumer keeps the window open until it actually observes a held
        // object, releases it via the engine primitive, then closes the
        // window. ~30 ticks (~0.5 s) is a generous upper bound for queue drain.
        ::InterlockedExchange(&g_PendingMonitorClearTicks, 30);
        return 1;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 3;
    }
}

// CACHE_SPAWN: generic object-from-tag spawn. Instantiates ANY
// obje-derived cache tag at a world pose via the engine's object_new primitive
// - no forge palette slot required. This is what the Master Palette's
// cache-only entries (InScenarioPalette=false) route to.
//   1. object_placement_data_new(pd, tagIndex, owner=0xFFFFFFFF, 0)
//   2. overwrite position (+0x28) and forward (+0x34) in the placement
//   3. object_new(pd) -> new object datum
//   4. object_set_position_and_orientation(datum, pos, fwd, up, ...) to snap
//      the full requested pose (the placement struct only carries fwd, not up).
// Returns: 1 = spawned, 2 = engine not loaded, 3 = SEH fault / spawn rejected.
using objpd_new_t   = void     (*)(void* outPd, uint32_t tagIndex, uint32_t owner, void* r9);
using object_new_t  = uint32_t (*)(void* placement);
using obj_setpose_t = uint8_t  (*)(uint32_t datum, float* pos, float* fwd, float* up, int a5, int a6);

uint32_t SpawnObjectFromTagInner(uint32_t tagIndex,
                                 float x, float y, float z,
                                 float fx, float fy, float fz,
                                 float ux, float uy, float uz)
{
    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) return 2;
    auto base = reinterpret_cast<uint8_t*>(hr);
    auto objpd_new  = reinterpret_cast<objpd_new_t>(base + HaloMapStudio::Engine::kRva_object_placement_data_new);
    auto object_new = reinterpret_cast<object_new_t>(base + HaloMapStudio::Engine::kRva_object_new);
    auto setpose    = reinterpret_cast<obj_setpose_t>(base + HaloMapStudio::Engine::kRva_object_set_pos_orient);

    __try {
        // 424-byte placement struct, 16-byte aligned for the float vector
        // writes the engine does internally.
        alignas(16) uint8_t pd[0x1A8] = {};
        objpd_new(pd, tagIndex, 0xFFFFFFFFu, nullptr);

        // Overwrite world position (+0x28) and forward (+0x34). Leave the
        // default scale (1.0 at +0x58) the placement_data_new set.
        *reinterpret_cast<float*>(pd + 0x28) = x;
        *reinterpret_cast<float*>(pd + 0x2C) = y;
        *reinterpret_cast<float*>(pd + 0x30) = z;
        *reinterpret_cast<float*>(pd + 0x34) = fx;
        *reinterpret_cast<float*>(pd + 0x38) = fy;
        *reinterpret_cast<float*>(pd + 0x3C) = fz;

        uint32_t datum = object_new(pd);
        if (datum == 0xFFFFFFFFu) return 3;

        // Snap full pose (pos + fwd + up). The placement only encoded fwd;
        // the poser is the universal setter the forge/teleport paths use.
        float pos[3] = { x, y, z };
        float fwd[3] = { fx, fy, fz };
        float up[3]  = { ux, uy, uz };
        __try { setpose(datum, pos, fwd, up, 0, 0); }
        __except (EXCEPTION_EXECUTE_HANDLER) { /* object exists; pose is best-effort */ }

        return 1;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 3;
    }
}

} // namespace

// Public helper so other MMF tickers (WorldSpawnSnapshot, etc.) can share the
// same dispatch path instead of re-implementing it.
extern "C" uint32_t HaloMapStudio_SpawnObjectFromTag(
    uint32_t tagIndex,
    float x, float y, float z,
    float fx, float fy, float fz,
    float ux, float uy, float uz)
{
    return SpawnObjectFromTagInner(tagIndex, x, y, z, fx, fy, fz, ux, uy, uz);
}

extern "C" uint32_t HaloMapStudio_DispatchForgeSpawn(
    uint32_t paletteIndex, uint32_t entryIndex, uint32_t variantIndex,
    float x, float y, float z, uint32_t monitorSlot)
{
    return DispatchSpawnAtPosition(paletteIndex, entryIndex, variantIndex,
                                   x, y, z, monitorSlot);
}

namespace {

// 'MMSF' little-endian = M(0x4D) M(0x4D) S(0x53) F(0x46) -> 0x4653 4D4D.
constexpr uint32_t kMagic   = 0x46534D4Du;
constexpr uint32_t kVersion = 1u;

// Status values written into ResultStatus. Mirrored in crates/hms-ipc/src/lib.rs.
constexpr uint32_t kStatus_Idle       = 0u;  // initial state, never returned
constexpr uint32_t kStatus_Success    = 1u;  // spawn + pose both applied
constexpr uint32_t kStatus_SpawnFail  = 2u;  // Forge_SpawnRequest_Execute didn't produce a datum
constexpr uint32_t kStatus_PoseFail   = 3u;  // datum found but pose call faulted
constexpr uint32_t kStatus_NotReady   = 4u;  // engine not in forge mode / scnr not loaded
constexpr uint32_t kStatus_NotImpl    = 99u; // turn-1 stub return (this file)

#pragma pack(push, 1)
struct FSP_Shared {
    // Header
    uint32_t Magic;
    uint32_t Version;
    uint32_t TriggerCounter;
    uint32_t ResponseCounter;

    // Request body
    uint32_t PaletteIndex;
    uint32_t EntryIndex;
    uint32_t VariantIndex;
    uint32_t SpawnMethod;   // 0 = default (palette / local object_new),
                            // 1 = live networked SpawnObjectFromDefinition by tag.
                            // (was _Pad0; keeps the 80-byte layout unchanged.)
    float    X,    Y,    Z;
    float    FwdX, FwdY, FwdZ;
    float    UpX,  UpY,  UpZ;

    // Response body
    uint32_t ResultStatus;
    uint32_t SpawnedDatum;
    uint32_t _Pad1;
};
static_assert(sizeof(FSP_Shared) == 80, "FSP_Shared layout drifted - keep in sync with the Rust mirror in crates/hms-ipc/src/lib.rs");
#pragma pack(pop)

const wchar_t kMapName[] = L"HaloMapStudio_ForgeSpawnAndPose_Shared";

HANDLE      g_hMap   = nullptr;
FSP_Shared* g_Shared = nullptr;
uint32_t    g_LastTrigger = 0;

bool EnsureOpen()
{
    if (g_Shared) return true;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(FSP_Shared), kMapName);
    if (!g_hMap) return false;
    g_Shared = (FSP_Shared*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0,
                                          sizeof(FSP_Shared));
    if (!g_Shared) {
        CloseHandle(g_hMap);
        g_hMap = nullptr;
        return false;
    }
    __try {
        // First-use init: stamp magic/version, zero everything else. Subsequent
        // EnsureOpen calls (e.g. across re-injections) preserve the existing
        // counters so the viewer's torn-read fence has a stable baseline.
        if (g_Shared->Magic != kMagic || g_Shared->Version != kVersion) {
            ZeroMemory(g_Shared, sizeof(FSP_Shared));
            g_Shared->Magic   = kMagic;
            g_Shared->Version = kVersion;
        }
        // Seed g_LastTrigger from the current value so an in-flight request
        // from a previous DLL session doesn't auto-fire on attach.
        g_LastTrigger = g_Shared->TriggerCounter;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}

} // namespace

// -----------------------------------------------------------------------------
// Public tick. Drained once per frame-pump (see FramePumpHook.cpp).
// -----------------------------------------------------------------------------
extern "C" __declspec(dllexport) void ForgeSpawnAndPose_FramePumpTick()
{
    // FORGE_MONITOR_GRAB_V2: consume the post-spawn auto-grab release window
    // FIRST, every frame, independent of this MMF's own request state (the
    // spawn that armed it may have come from the WorldSpawn ring, not here).
    // The window stays open until we actually OBSERVE and release a held
    // object - so a grab that lands late (slow queue drain) is still caught.
    // Once released, close the window immediately so we don't keep poking the
    // monitor (and can't fight a subsequent genuine human grab).
    if (g_PendingMonitorClearTicks > 0)
    {
        if (ReleaseMonitorHeldObject(0u))
            ::InterlockedExchange(&g_PendingMonitorClearTicks, 0);   // released - done
        else
            ::InterlockedDecrement(&g_PendingMonitorClearTicks);     // not yet - keep waiting
    }

    if (!EnsureOpen()) return;

    uint32_t trig = 0;
    __try { trig = g_Shared->TriggerCounter; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return; }

    if (trig == g_LastTrigger) return;
    g_LastTrigger = trig;

    // Snapshot the request fields under SEH so a torn map-view read can't
    // crash us. Local copies because turn-2 will hand these to the engine
    // call site and we want a stable view across the whole pipeline.
    uint32_t palette = 0, entry = 0, variant = 0, method = 0;
    float x = 0, y = 0, z = 0;
    float fx = 0, fy = 0, fz = 0;
    float ux = 0, uy = 0, uz = 0;
    __try {
        palette = g_Shared->PaletteIndex;
        entry   = g_Shared->EntryIndex;
        variant = g_Shared->VariantIndex;
        method  = g_Shared->SpawnMethod;
        x  = g_Shared->X;     y  = g_Shared->Y;     z  = g_Shared->Z;
        fx = g_Shared->FwdX;  fy = g_Shared->FwdY;  fz = g_Shared->FwdZ;
        ux = g_Shared->UpX;   uy = g_Shared->UpY;   uz = g_Shared->UpZ;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return;
    }

    ZH_Logf("[ForgeSpawnAndPose] trigger=%u P=%u E=%u V=%u "
            "pos=(%.2f,%.2f,%.2f) fwd=(%.2f,%.2f,%.2f) up=(%.2f,%.2f,%.2f)\n",
            trig, palette, entry, variant, x, y, z, fx, fy, fz, ux, uy, uz);

    // Turn 2 (this commit): wire DispatchSpawnAtPosition. The engine queues
    // the spawn via VariantCommand_HandleExecution -> ExecuteOrEnqueue -> the
    // session command pool. The actual object spawn lands 1-2 ticks later.
    // Slot 0 = local-player monitor (single-player forge editor; multi-player
    // forge would need to look up the active monitor's slot id).
    //
    // Forward/up vectors are still ignored - opcode 0 is position-only;
    // applying user rotation requires turn 3 (object-table diff to find the
    // newly-spawned datum, then Enqueue_UpdateObjectPosAndAiming with the
    // user's fwd/up). Until that lands the spawn appears at the requested
    // XYZ with engine-default rotation.
    // CACHE_SPAWN: the viewer's master palette routes cache-only entries (no forge
    // palette slot) here with EntryIndex == 0xFFFFFFFF and PaletteIndex set to
    // the raw object-definition tag id. Detect that sentinel and use the
    // generic object_new primitive instead of the forge palette dispatch.
    uint32_t status;
    uint32_t spawnedDatum = 0u;
    if (method == 1u) {
        // LIVE path: spawn the object-definition tag (carried in the
        // PaletteIndex field) via SpawnObjectFromDefinition with the forced
        // session/host bytes + networking handover, so it lands as a real live
        // networked object. Returns the new datum directly.
        uint32_t d = HaloMapStudio_SpawnDefinitionLive(palette, x, y, z, fx, fy, fz, ux, uy, uz);
        spawnedDatum = (d == 0xFFFFFFFFu) ? 0u : d;
        status = (d == 0xFFFFFFFFu) ? 3u : 1u;
        ZH_Logf("[ForgeSpawnAndPose] LIVE spawn tag=0x%X datum=0x%08X result=%u\n",
                palette, d, status);
    } else if (entry == 0xFFFFFFFFu) {
        status = SpawnObjectFromTagInner(palette, x, y, z, fx, fy, fz, ux, uy, uz);
        ZH_Logf("[ForgeSpawnAndPose] cache-tag spawn tag=0x%X result=%u\n", palette, status);
    } else {
        status = DispatchSpawnAtPosition(palette, entry, variant,
                                         x, y, z, /*monitorSlot*/ 0u);
        ZH_Logf("[ForgeSpawnAndPose] dispatch result=%u\n", status);
    }

    uint32_t mappedStatus;
    switch (status) {
        case 1:  mappedStatus = kStatus_Success;   break;
        case 2:  mappedStatus = kStatus_NotReady;  break; // engine module gone
        case 3:  mappedStatus = kStatus_SpawnFail; break; // SEH inside engine
        default: mappedStatus = kStatus_SpawnFail; break;
    }

    __try {
        g_Shared->SpawnedDatum    = spawnedDatum; // populated on the LIVE path
        g_Shared->ResultStatus    = mappedStatus;
        g_Shared->ResponseCounter = trig;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // Map view became invalid mid-write - give up; viewer's MMF probe
        // will eventually re-open and retry.
    }
}
