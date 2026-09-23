// EngineThreadResolver.h
// =============================================================================
// Resolves haloreach's object-table descriptor from inside the engine's
// frame-pump thread. Two complementary paths:
//
//   1. STATIC-GLOBAL path (cheap, works from any thread): the engine stores
//      the descriptor address as a fixed offset from the session anchor
//      `*(haloreach + 0x24FB710)`. Adding 0xB65EFC gives the descriptor
//      address directly (NOT a pointer to one - this is the descriptor
//      itself living inside a global state block).
//
//   2. TLS path (matches what the engine's own object accessors use): each
//      game thread keeps a TLS block whose +0x10 slot holds a pointer to
//      the live descriptor. Resolved via the PE TLS directory's
//      AddressOfIndex; the per-thread block lives at TEB+0x58 (TLS array)
//      indexed by that index.
//
// We try the static-global path first (it's a single deref) and fall back
// to TLS if it gave us a bogus pointer. The CALLER must be on the engine
// frame-pump thread for path 2 to work - this header is intended for use
// from inside our hkFramePump in FramePumpHook.cpp.
//
// Descriptor layout (matches the engine's GetBulkObjectsToBuffer):
//   +0x20  uint32  entrySize
//   +0x44  uint32  maxCount
//   +0x50  void**  entries
// Per entry:
//   +0x00  uint16  salt          (matches datum's high word; 0 = empty)
//   +0x02  uint8   flags         (active bit & 0x80)
//   +0x04  uint8   objectType
//   +0x10  void*   object        (NOT +0x0C)
// Per object:
//   +0x00  uint32  primaryTag
//   +0x10  uint32  attached-children head datum
//   +0x14  uint32  parent datum
//   +0x54  float   X             (NOT +0x20 - biped world pos lives at +0x54)
//   +0x58  float   Y
//   +0x5C  float   Z
//   +0xE4  float   health
//   +0xE8  float   shield
// =============================================================================

#pragma once

#include <windows.h>
#include <cstdint>
#include <intrin.h>
#include <atomic>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace HaloMapStudio { namespace Engine {

// =============================================================================
// Map-change epoch tracking.
//
// Background: the descriptor lives at `*(haloreach + 0x24FB710) + 0xB65EFC`.
// `*(haloreach + 0x24FB710)` (the "anchor"/global-state pointer) is stable
// across an entire haloreach.dll lifetime EXCEPT when a new scenario is
// loaded - at that point the engine swaps the anchor to a new global-state
// block, which makes the descriptor address change too. Anything we cached
// keyed by datum (mode-tag map, hlmt cache, per-biped lookup, etc.) is
// stale because:
//   * tag table is repacked -> same datum may now point to a different tag.
//   * obj-pool slots are repurposed -> cached "datum -> obj*" memos fault or
//     point at unrelated objects (which is what produced the (0,0,0)
//     symptom - the position fetch was reading garbage as floats).
//
// We track the anchor pointer + the haloreach module base + every
// observed bad-bounds read on the descriptor. ANY of those changing bumps
// the epoch atomically. Caches that opt in compare a per-cache stored epoch
// against this one and clear themselves on mismatch.
//
// Cheap to check (one global+atomic read per tick); keeps every consumer
// in lockstep without each having to re-implement detection.
// =============================================================================
inline std::atomic<uint64_t>  g_MapEpoch{ 1 };  // 0 reserved for "uninitialised"
inline std::atomic<uintptr_t> g_LastAnchor{ 0 };
inline std::atomic<uintptr_t> g_LastHaloBase{ 0 };

// DESCRIPTOR CACHE: the last descriptor pointer we
// resolved successfully, plus the anchor it was valid under. The object-pool
// descriptor pointer is stable for the life of a loaded map (it only moves when
// the anchor changes = map load). But the LIVE read that recovers it - static
// (anchor+0xB65EFC) or TLS (tlsBlock+0x10) - is only coherent at certain points
// in the frame; at other points the static bounds read as garbage and the TLS
// slot is stale. Rather than depend on tick timing, we cache the good pointer
// keyed by anchor and hand it back when a given tick's live read comes up empty.
// Invalidated automatically when the anchor changes (different key). This is
// what lets the viewer keep showing objects even though our per-frame hook can't
// always land inside the engine's descriptor-coherent window.
inline std::atomic<uintptr_t> g_CachedDesc{ 0 };
inline std::atomic<uintptr_t> g_CachedDescAnchor{ 0 };

// Bump the epoch. Logs once per change to dll.log (the user's diagnostic
// channel - we deliberately do NOT spam every tick).
inline void BumpMapEpoch_(const char* reason)
{
    uint64_t newEpoch = g_MapEpoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    ZH_Logf("[HaloMapStudioDLL] descriptor cache invalidated (map change detected) "
            "epoch=%llu reason=%s\n",
            (unsigned long long)newEpoch, reason ? reason : "?");
}

// Read the current epoch. Caches store the value they last reconciled
// against and clear themselves when this differs.
inline uint64_t GetMapEpoch()
{
    return g_MapEpoch.load(std::memory_order_acquire);
}

// Manually trigger an epoch bump (e.g., from a downstream cache that
// detected a stale lookup the resolver itself didn't catch). Rare - 
// the in-resolver detection covers the normal case.
inline void NotifyMapChanged_External(const char* reason)
{
    BumpMapEpoch_(reason);
}

// =============================================================================
// Descriptor field offsets (corrected per ZScriptNative.cpp).
// =============================================================================
inline constexpr size_t kDesc_EntrySize = 0x20;
inline constexpr size_t kDesc_MaxCount  = 0x44;
inline constexpr size_t kDesc_Entries   = 0x50;

inline constexpr size_t kEntry_Salt    = 0x00;
inline constexpr size_t kEntry_Flags   = 0x02;   // u16, not u8
inline constexpr size_t kEntry_ObjPtr  = 0x10;
// corrected from 0x80 (host-replicates) -> 0x01 (alive).
// Bit 0x80 is the host-side network-replication-owner flag; on a
// client (or when MCC isn't the lobby host), held weapons / projectiles
// / equipment have 0x80 == 0 but 0x01 == 1. Filtering by 0x80 dropped
// every held weapon from the publish (attached=0/N never resolving).
// Verified via Ghidra:
//   Object_DetachInternal               inside an iter loop, when the
//                                       signed-byte flags read is < 0
//                                       (bit 0x80 set) it fires
//                                       Network_PropagateOwnershipUpdate
//                                       and `break`s out of that inner
//                                       step (NOT a function return).
//   ObjectEntry_ClearActiveFlags...     clears bits & 0xFFFA (i.e.
//                                       0x0001 + 0x0004) on delete --
//                                       independent evidence that 0x01
//                                       is the alive bit, set at init
//                                       and cleared at destroy.
//   ExecuteAction_SetObjectEnabled      gates on (flags & 1) for the
//                                       liveness check.
// Read as u16 so we can also reject the queued-spawn-suppress bit 0x10
// without false-positives from upper-byte network bits.
inline constexpr uint16_t kEntry_AliveBit    = 0x0001;
inline constexpr uint16_t kEntry_SuppressBit = 0x0010;
// ATTACHED_CHILDREN_FIX. Vehicle turrets / scorpion cannons
// (and other attached-child objects) have 0x0001 == 0 in their entry
// flags but 0x0004 == 1. Ghidra evidence:
//   ObjectEntry_ClearActiveFlags  ANDs with 0xFFFA on delete -- which
//                                 clears BOTH 0x0001 and 0x0004,
//                                 confirming both are alive indicators.
// Diag confirmed in-game: parent=0xE54002D1 (scorpion) referenced child
// 0xE54102D2 (cannon) in AttachedHandles, but the cannon entry was
// silently skipped by the 0x0001-only filter -- attSkipNoTable=15 every
// tick. Widening the mask to (0x0001 | 0x0004) lets attached-secondary
// objects through without affecting primary-object liveness gating.
inline constexpr uint16_t kEntry_AliveWideMask = 0x0001 | 0x0004;
// Legacy alias for any code that still references kEntry_ActiveMask.
// Pointing at kEntry_AliveBit so the rename doesn't require renaming
// every call site, but the comment above documents the change.
inline constexpr uint16_t kEntry_ActiveMask  = kEntry_AliveBit;

inline constexpr size_t kObj_PrimaryTag = 0x00;
inline constexpr size_t kObj_AttachHead = 0x10;
inline constexpr size_t kObj_Parent     = 0x14;
inline constexpr size_t kObj_NextSibling = 0x0C;
inline constexpr size_t kObj_WorldPos   = 0x54;
// Forward + up live at +0x50/+0x5C (NOT +0x60/+0x6C): the former pair stays
// valid AT REST. The +0x60/+0x6C pair appears to be a live physics
// scratchpad that only holds correct values during active manipulation
// (Forge drag); at rest it goes stale and the viewer renders meshes at
// the wrong orientation. Confirmed by user moving a forge platform - 
// rotation tracks correctly while moving, snaps to wrong-but-fixed
// orientation when released.
inline constexpr size_t kObj_Forward    = 0x50;
inline constexpr size_t kObj_Up         = 0x5C;
inline constexpr size_t kObj_TypeSig    = 0x04;
inline constexpr size_t kObj_Health     = 0xE4;
inline constexpr size_t kObj_Shield     = 0xE8;

// =============================================================================
// Module-level constants (haloreach.dll RVAs).
// =============================================================================
inline constexpr uintptr_t kRva_SessionAnchor   = 0x24FB710;
inline constexpr uintptr_t kOff_ObjDescStatic   = 0xB65EFC;  // anchor+offset = descriptor addr
// The per-frame hook site.
//   Other trainers hook 0x2AA92C - the engine's per-frame pump function
//   (clean entry: ret/int3 then `mov [rsp+8],rbx` prologue; takes dt in XMM0).
//   Two DLLs can't MinHook one address (their byte-drift watchers ping-pong
//   reinstalls), so co-injection would kill one pump.
//
//   Hooking Engine_FrameUpdate(0x33E6C) - the pump's CALLER - runs the ticks
//   OUTSIDE the pump's descriptor-coherent window: the object-pool anchor/TLS
//   descriptor is only valid DURING the pump (after GetThreadContextOrFlag
//   establishes the game context). At end-of-frame the static descriptor reads
//   garbage and the TLS slot points at an EMPTY pool (dll.log "descriptor
//   bounds invalid" every frame + PoseSnapshot poolPopulated=0).
//
//   So the hook goes on a CALLEE of the pump, at a distinct clean entry, so the
//   ticks run INSIDE the coherent window. ReplicationContext_SendStateUpdate(0x2A90F4):
//     * verified clean function entry (ret/int3 then `push rbx` prologue)
//     * NO arguments (int64 return) -> trivially-correct detour ABI
//     * SINGLE caller = the pump -> runs exactly once per frame, unconditionally,
//       AFTER GetThreadContextOrFlag (so the descriptor is coherent), and every
//       frame as a CLIENT (clients send their state to the host each frame)
//   Our detour calls the original FIRST, then runs ticks - still inside the
//   pump's call stack, so TLS + anchor are coherent. Signature: int64 fn(void).
inline constexpr uintptr_t kRva_FramePumpHook   = 0x2A90F4;
inline constexpr uintptr_t kRva_PlayerList      = 0x5DF578;  // anchor+offset = player-list base

// Variant-command dispatcher for forge actions. RE'd:
//   * VariantCommand_HandleExecution(uint32_t* cmd[16 dwords / 64 B])
//   * Opcode at cmd[+0x3C] (u16):
//       0 = spawn-at-position    1 = (delete?)         3 = respawn
//       0xA / 0xB / 0xC / 0xE    = (other forge ops)
//   * For opcode 0 (palette spawn at user position):
//       cmd[0x00] = paletteIndex (u32)
//       cmd[0x04] = entryIndex   (u32)
//       cmd[0x08] = variantIndex (u32)
//       cmd[0x0C] = X position   (f32)
//       cmd[0x10] = Y position   (f32)
//       cmd[0x14] = Z position   (f32)
//       cmd[0x18..0x38] = unread for opcode 0 (zero-fill)
//       cmd[0x38] = local-player monitor slot (u32, low 16b)
//       cmd[0x3C] = 0 (opcode)
//   * Routes through VariantCommand_ExecuteOrEnqueue -> bit-stream serialize ->
//     SessionCommandPool entry -> VariantCommand_QueueCommand. The actual spawn
//     happens 1-2 ticks later when the queue drains.
//   * Note: ExecuteOrEnqueue is gated on `IsContextInitializedAndInGameFlagSet
//     == 0`. Forge mode evaluates as 0 here (forge is "pre-game / lobby"
//     state, not "in matchmaking gameplay"), so the body runs as expected.
inline constexpr uintptr_t kRva_VariantCommand_HandleExecution = 0x77B54;

// Forge_ValidateSpawnRequest - the per-palette-entry "can spawn?" predicate.
// RE'd from haloreach.dll +0x751F4. Signature: char(int param_1,
// undefined8* param_2). Three internal gates:
//   1. *(int*)(palette_entry + 0x140) > 0 - entry is enabled
//   2. *(int*)(weapon_def    + 0x10 ) > current_count  (max-count cap)
//   3. current_budget + item_cost <= max_budget        (budget cap)
// All three must pass for the function to return '\x01'; otherwise it returns
// '\0' and the upstream spawn request is silently dropped. Three callers:
// Forge_ResolvePalletObject (the player-monitor spawn path), Session_Handle
// WeaponSpawnRequest, ForgeDescriptor_CanSpawn. Hooking this one function
// bypasses the cap for ALL three paths.
inline constexpr uintptr_t kRva_Forge_ValidateSpawnRequest = 0x751F4;

// PHYSICS_WAKE_V1: havok rigid-body wake on object pose
// override. Placeholder = 0 ("not RE'd yet"); ForgePhysics_WakeAfterPose
// short-circuits the call when this is 0 so the absent-RE state is safe.
// Fill with the haloreach.dll RVA once located; see PHYSICS_WAKE_RE.md
// for the attack plan + Ghidra-static search strategy.
inline constexpr uintptr_t kRva_Object_WakePhysics = 0;

// Forge_ValidateAndResolveSpawn - session context forge object cap.
// RE'd from haloreach.dll +0x76834. Enforces a 32-object (0x20)
// per-session forge context limit at TLS+0x2C14. When 32 forge objects exist,
// further spawns are silently rejected. Hook to raise/remove the cap.
inline constexpr uintptr_t kRva_Forge_ValidateAndResolveSpawn = 0x76834;

// Generic object-from-tag spawn primitive (CACHE_SPAWN, RE'd).
// Lets us instantiate ANY obje-derived cache tag at a world pose without a
// forge palette slot. NOT forge-gated; subject only to the global object
// pool capacity, not the forge sandbox budget.
//   object_placement_data_new(out[0x1A8], tagIndex, ownerDatum=0xFFFFFFFF, 0)
//   -> memsets + defaults a 424-byte placement struct.
//   object_new(placement) -> returns new object datum (0xFFFFFFFF on fail).
//   object_set_position_and_orientation(datum, pos, fwd, up, ...) -> poses it.
// Placement struct: +0x00 tagIndex, +0x28 pos(3f), +0x34 fwd(3f), +0x58 scale.
inline constexpr uintptr_t kRva_object_placement_data_new = 0x49CC24;
inline constexpr uintptr_t kRva_object_new                = 0x46D6E0;
inline constexpr uintptr_t kRva_object_set_pos_orient     = 0x46CB38;

// FORGE_MONITOR_GRAB (RE'd, High/Certain confidence - verified by
// decompile + raw disasm of Forge_SpawnRequest_Execute(+0x2CC73C),
// Session_HandleWeaponSpawnRequest(+0x76608) and the grab writer(+0x76B10),
// which all compute the SAME monitor base). The per-player Forge monitor
// "held object" latch. After a forge spawn the engine auto-grabs the new
// object into the spawning monitor; the next spawn's gate at +0x7665A bails
// when the held field != -1, so MMS's back-to-back spawns get rejected.
// Resolution chain:
//   tlsIndex = *(u32*)(haloreach + kRva_MonitorTlsIndex)   // a DWORD, NOT a ptr
//   tlsBlock = (*(void***)(TEB+0x58))[tlsIndex]            // GetTLSArrayBase()[idx]
//   ctxBase  = *(void**)(tlsBlock + kMonitorCtxOffset)
//   monitor  = ctxBase + slot*kMonitorStride
// Release = restore the engine's own "nothing held" sentinel:
//   monitor + kMonitor_HeldObject = 0xFFFFFFFF ; monitor + kMonitor_GrabState = 0.
inline constexpr uintptr_t kRva_MonitorTlsIndex = 0xC17B18;
inline constexpr size_t    kMonitorCtxOffset    = 0x20;
inline constexpr size_t    kMonitorStride       = 0x12C;
inline constexpr size_t    kMonitor_HeldObject  = 0x1758;  // u32, 0xFFFFFFFF = none
inline constexpr size_t    kMonitor_GrabState   = 0x1849;  // u8 grab-state byte

// FORGE_MONITOR_GRAB_V2: why the V1 "write +0x1758 = -1"
// release didn't stick. The per-tick forge network-state applier
// FUN_180074768 (fn-ptr table entry @ +0x995450, runs every frame) loops
// monitor slots 0..0xF, reads the network-replicated "should-be-held" datum,
// and if the live +0x1758 field disagrees it RE-APPLIES the grab via the grab
// core FUN_180076B10 (which writes +0x1758 = datum, +0x1849 = 0). So a
// fire-once field poke is reverted on the very next tick - the spawn registers
// the object as held in network/replication state, and the applier keeps
// restoring it. The correct release is the engine's own "drop held object"
// primitive FUN_180077068(monitorSlot, heldDatum): it verifies
// heldDatum == monitor[slot]+0x1758, then calls
// Forge_ApplyObjectPlacementOrDeletion(slot, datum, 0) to PLACE (not delete)
// the object and tear down the network-side held record, so the applier stops
// re-grabbing. This is exactly how FUN_180074768 itself releases when the
// network says "nothing held". Signature (4 args, only first 2 meaningful):
//   void Forge_ReleaseHeldObject(uint32_t monitorSlot, uint32_t heldDatum,
//                                uint64_t r8=0, uint64_t r9=0);
inline constexpr uintptr_t kRva_Forge_ReleaseHeldObject = 0x77068;

// =============================================================================
// SEH-safe scalar/ptr reads.
// =============================================================================
template <typename T>
inline bool SafeReadT(const void* addr, T& out)
{
    __try { out = *(const T*)addr; return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { out = T{}; return false; }
}

inline bool SafeReadPtr(const void* addr, void*& out)
{
    out = nullptr;
    __try { out = *(void* const*)addr; return out != nullptr; }
    __except (EXCEPTION_EXECUTE_HANDLER) { out = nullptr; return false; }
}

// =============================================================================
// PE TLS directory walker - gives us haloreach.dll's TLS index without
// having to know which slot the linker chose.
// =============================================================================
inline uint32_t ResolveTlsIndexFromModule(const wchar_t* moduleName)
{
    HMODULE mod = GetModuleHandleW(moduleName);
    if (!mod) return 0xFFFFFFFFu;

    uint8_t* base = (uint8_t*)mod;
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)base;
    if (!dos || dos->e_magic != IMAGE_DOS_SIGNATURE) return 0xFFFFFFFFu;

    IMAGE_NT_HEADERS64* nt = (IMAGE_NT_HEADERS64*)(base + dos->e_lfanew);
    if (!nt || nt->Signature != IMAGE_NT_SIGNATURE) return 0xFFFFFFFFu;

    auto& dd = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS];
    if (dd.VirtualAddress == 0 || dd.Size == 0) return 0xFFFFFFFFu;

    IMAGE_TLS_DIRECTORY64* tls = (IMAGE_TLS_DIRECTORY64*)(base + dd.VirtualAddress);
    if (!tls || tls->AddressOfIndex == 0) return 0xFFFFFFFFu;

    uint32_t idx = 0xFFFFFFFFu;
    __try { idx = *(uint32_t*)(uintptr_t)tls->AddressOfIndex; }
    __except (EXCEPTION_EXECUTE_HANDLER) { idx = 0xFFFFFFFFu; }
    return idx;
}

// =============================================================================
// TEB -> TLS array. Per-thread; only valid on the engine's own frame thread
// (or any thread the engine has populated TLS on, which in practice is just
// the frame thread for object-table access).
// =============================================================================
inline void** GetTLSArrayBase()
{
#if defined(_M_X64)
    uintptr_t teb = (uintptr_t)__readgsqword(0x30);
    if (!teb) return nullptr;
    return *(void***)(teb + 0x58);
#else
    return nullptr;
#endif
}

// =============================================================================
// Resolve the descriptor pointer. Tries static-global first, falls back to
// TLS. Returns nullptr only if both paths give us a null/bogus pointer.
//
// MUST be called from the engine frame-pump thread (or any thread on which
// haloreach has populated TLS) for the TLS fallback to fire - but the
// static path itself works from any thread, so most calls succeed even
// before TLS is set up.
// =============================================================================
inline uint8_t* ResolveObjectPoolDescriptor()
{
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return nullptr;
    uint8_t* base = (uint8_t*)mod;

    // Module-base check - if haloreach.dll was unloaded and reloaded (full
    // game-process restart of the engine, rare but possible), every cache
    // is invalid. This corresponds to ResetPlayerListCache_Internal's
    // "haloreach base changed" path in PlayerListTLS.cpp.
    //
    // Skip the 0 -> first-real-base transition so we don't log a spurious
    // "module base changed" on the very first tick (the map epoch is
    // still updated, but no log line - clean dll.log).
    uintptr_t curBase = (uintptr_t)base;
    uintptr_t prevBase = g_LastHaloBase.load(std::memory_order_acquire);
    if (prevBase != curBase) {
        if (g_LastHaloBase.compare_exchange_strong(
                prevBase, curBase, std::memory_order_acq_rel))
        {
            if (prevBase != 0) {
                BumpMapEpoch_("haloreach module base changed");
            }
            // Reset anchor regardless - different module => different
            // anchor space, even on first observation.
            g_LastAnchor.store(0, std::memory_order_release);
        }
    }

    // ---- Path 1: static-global (anchor + 0xB65EFC) ----
    uint8_t* anchor = nullptr;
    if (SafeReadPtr(base + kRva_SessionAnchor, (void*&)anchor) && anchor) {

        // Anchor-pointer change => new map loaded inside the same haloreach
        // instance. This is the common case after a "load second map"
        // sequence; the descriptor address itself shifts and any cached
        // datum-keyed lookup must be discarded.
        uintptr_t curAnchor = (uintptr_t)anchor;
        uintptr_t prevAnchor = g_LastAnchor.load(std::memory_order_acquire);
        if (prevAnchor != curAnchor) {
            if (g_LastAnchor.compare_exchange_strong(
                    prevAnchor, curAnchor, std::memory_order_acq_rel))
            {
                // Skip the very-first transition (0 -> first real anchor) so
                // we don't log a spurious "map change" on initial population.
                if (prevAnchor != 0) {
                    BumpMapEpoch_("global-state anchor pointer changed");
                }
            }
        }

        uint8_t* desc = anchor + kOff_ObjDescStatic;
        // Sanity-check: read entrySize and maxCount; if both look reasonable
        // we're good.
        uint32_t entrySize = 0;
        uint32_t maxCount  = 0;
        if (SafeReadT(desc + kDesc_EntrySize, entrySize) &&
            SafeReadT(desc + kDesc_MaxCount,  maxCount)  &&
            entrySize > 0 && entrySize <= 0x100 &&
            maxCount  > 0 && maxCount  <= 0x4000)
        {
            // Good live read - refresh the cache (keyed by this anchor).
            g_CachedDesc.store((uintptr_t)desc, std::memory_order_release);
            g_CachedDescAnchor.store(curAnchor, std::memory_order_release);
            return desc;
        }
        // Bounds were bad - the static descriptor slot isn't coherent at this
        // point in the frame (NOT a map change; the anchor above is unchanged).
        // Do NOT bump the epoch - that would flush every downstream cache every
        // frame. Instead, if we have a cached descriptor from a coherent read
        // under THIS SAME anchor, hand it back - the pool pointer is stable for
        // the life of the map. Fall through to the TLS path only if we have no
        // cache yet.
        uintptr_t cached = g_CachedDesc.load(std::memory_order_acquire);
        if (cached && g_CachedDescAnchor.load(std::memory_order_acquire) == curAnchor)
            return (uint8_t*)cached;
    }

    // ---- Path 2: TLS - each engine thread has *(tlsBlock + 0x10) = desc ----
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

    void* desc = nullptr;
    if (SafeReadPtr((uint8_t*)tlsBlock + 0x10, desc) && desc)
    {
        // Validate bounds before trusting/caching the TLS descriptor.
        uint32_t es = 0, mc = 0;
        if (SafeReadT((uint8_t*)desc + kDesc_EntrySize, es) &&
            SafeReadT((uint8_t*)desc + kDesc_MaxCount,  mc) &&
            es > 0 && es <= 0x100 && mc > 0 && mc <= 0x4000)
        {
            uintptr_t a = g_LastAnchor.load(std::memory_order_acquire);
            g_CachedDesc.store((uintptr_t)desc, std::memory_order_release);
            g_CachedDescAnchor.store(a, std::memory_order_release);
            return (uint8_t*)desc;
        }
    }
    // TLS gave nothing coherent this tick - fall back to the last good
    // descriptor cached under the current anchor (stable for the map's life).
    {
        uintptr_t cached = g_CachedDesc.load(std::memory_order_acquire);
        uintptr_t a = g_LastAnchor.load(std::memory_order_acquire);
        if (cached && a && g_CachedDescAnchor.load(std::memory_order_acquire) == a)
            return (uint8_t*)cached;
    }
    return nullptr;
}

// =============================================================================
// Resolve a datum to its live object pointer using the SAME descriptor we
// already cracked. Mirrors ObjectTableCtx::ResolveDatum from PlayerListTLS.cpp.
// Returns nullptr on any read fault or a stale-salt mismatch.
// =============================================================================
inline uint8_t* ResolveDatumToObject(uint8_t* desc, uint32_t datum)
{
    if (!desc || datum == 0u || datum == 0xFFFFFFFFu) return nullptr;

    uint32_t entrySize = 0;
    uint32_t maxCount  = 0;
    void*    rawBase   = nullptr;
    if (!SafeReadT(desc + kDesc_EntrySize, entrySize) || entrySize == 0) return nullptr;
    if (!SafeReadT(desc + kDesc_MaxCount,  maxCount)  || maxCount  == 0) return nullptr;
    if (!SafeReadPtr(desc + kDesc_Entries, rawBase)   || !rawBase) return nullptr;

    uint32_t idx = datum & 0xFFFFu;
    if (idx >= maxCount) return nullptr;

    uint8_t* entry = (uint8_t*)rawBase + (size_t)idx * (size_t)entrySize;

    uint16_t salt = 0;
    if (!SafeReadT(entry + kEntry_Salt, salt) || salt == 0) return nullptr;
    uint16_t expected = (uint16_t)((datum >> 16) & 0xFFFFu);
    if (salt != expected) return nullptr;

    uint16_t flags = 0;
    if (!SafeReadT(entry + kEntry_Flags, flags) || (flags & kEntry_AliveBit) == 0) return nullptr;

    void* obj = nullptr;
    if (!SafeReadPtr(entry + kEntry_ObjPtr, obj) || !obj) return nullptr;
    return (uint8_t*)obj;
}

// =============================================================================
// CLIENT_OBJ_DIAG: as a CLIENT the object pool the DLL
// reads (static anchor+0xB65EFC) resolves with valid bounds but ZERO active
// entries, so no forge objects load - while the same descriptor is populated as
// host. This throttled probe logs, side by side, the STATIC descriptor and the
// TLS descriptor (tlsBlock+0x10) with each one's address, bounds, and ACTIVE
// entry count, so we can see which (if either) holds the client's live objects.
// Call once/second from the frame pump. Remove once the client path is wired.
inline uint32_t CountActiveEntries_(uint8_t* desc)
{
    if (!desc) return 0xFFFFFFFFu;   // unresolved
    uint32_t entrySize = 0, maxCount = 0; void* base = nullptr;
    if (!SafeReadT(desc + kDesc_EntrySize, entrySize) || entrySize == 0 || entrySize > 0x100) return 0xFFFFFFFEu;
    if (!SafeReadT(desc + kDesc_MaxCount,  maxCount)  || maxCount == 0 || maxCount  > 0x4000) return 0xFFFFFFFDu;
    if (!SafeReadPtr(desc + kDesc_Entries, base) || !base) return 0xFFFFFFFCu;
    uint32_t active = 0;
    for (uint32_t i = 0; i < maxCount; i++) {
        uint8_t* ent = (uint8_t*)base + (size_t)i * (size_t)entrySize;
        uint16_t salt = 0;
        if (!SafeReadT(ent + kEntry_Salt, salt) || salt == 0) continue;
        active++;
    }
    return active;
}

inline void DiagObjectDescriptors()
{
    HMODULE mod = GetModuleHandleW(L"haloreach.dll");
    if (!mod) return;
    uint8_t* base = (uint8_t*)mod;

    // Static descriptor.
    uint8_t* staticDesc = nullptr;
    uint8_t* anchor = nullptr;
    if (SafeReadPtr(base + kRva_SessionAnchor, (void*&)anchor) && anchor)
        staticDesc = anchor + kOff_ObjDescStatic;

    // TLS descriptor.
    uint8_t* tlsDesc = nullptr;
    static uint32_t s_idx = 0xFFFFFFFFu;
    if (s_idx == 0xFFFFFFFFu) s_idx = ResolveTlsIndexFromModule(L"haloreach.dll");
    if (s_idx != 0xFFFFFFFFu) {
        void** tlsArray = GetTLSArrayBase();
        if (tlsArray) {
            void* tlsBlock = nullptr;
            __try { tlsBlock = tlsArray[s_idx]; } __except (EXCEPTION_EXECUTE_HANDLER) { tlsBlock = nullptr; }
            if (tlsBlock) { void* d = nullptr; if (SafeReadPtr((uint8_t*)tlsBlock + 0x10, d)) tlsDesc = (uint8_t*)d; }
        }
    }

    uint32_t esS = 0, mcS = 0, esT = 0, mcT = 0;
    if (staticDesc) { SafeReadT(staticDesc + kDesc_EntrySize, esS); SafeReadT(staticDesc + kDesc_MaxCount, mcS); }
    if (tlsDesc)    { SafeReadT(tlsDesc    + kDesc_EntrySize, esT); SafeReadT(tlsDesc    + kDesc_MaxCount, mcT); }

    ZH_Logf("[CLIENT_OBJ_DIAG] anchor=%p | STATIC desc=%p es=%u mc=%u active=%u | "
            "TLS desc=%p es=%u mc=%u active=%u\n",
            (void*)anchor,
            (void*)staticDesc, esS, mcS, CountActiveEntries_(staticDesc),
            (void*)tlsDesc,    esT, mcT, CountActiveEntries_(tlsDesc));
}

}} // namespace HaloMapStudio::Engine
