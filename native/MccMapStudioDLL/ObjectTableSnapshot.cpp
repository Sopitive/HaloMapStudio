// ObjectTableSnapshot.cpp (HaloMapStudioDLL)
// =============================================================================
// Publishes the v4 ZeroHour_ObjectTable_Snapshot MMF that the viewer reads.
//
// Runs from the frame-pump hook (FramePumpHook.cpp), so we're always on the
// engine thread. That means the descriptor resolver in EngineThreadResolver.h
// can take either the static-global path or the TLS-fallback path - both
// give us the live object-pool descriptor at the same place the engine's
// own object accessors do.
//
// Field layout (matches the engine's GetBulkObjectsToBuffer):
//   desc + 0x20  uint32  entrySize
//   desc + 0x44  uint32  maxCount
//   desc + 0x50  void**  entries
//   entry + 0x00 uint16  salt        (0 = empty slot)
//   entry + 0x02 uint8   flags       (& 0x80 = active)
//   entry + 0x10 void*   object
//   obj + 0x54..0x5C    float[3]  world position (NOT +0x20 - that's some
//                                                  other struct's pos field)
//   obj + 0xE4          float     health
//   obj + 0xE8          float     shield
//
// The MMF schema matches v4 byte-for-byte so the hms-ipc client deserialises
// without changes. We don't implement parent-up-the-chain handles - children
// only. Render-model resolution is via HaloReachTagHelpers (header-only).
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"
#include "HaloReachTagHelpers.h"

#include <windows.h>
#include <cstdint>
#include <cstring>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic       = 0x544A424Fu;  // 'OBJT'
// v5 appended PrimaryTagId (4 bytes, 88->92) so the viewer can match an
// object against the active Forge palette tag-id set without re-walking
// the scenario palette tables. The viewer's filter is:
//   render mesh if ModeTagId resolved AND mesh load succeeded
//   ELSE render dot if PrimaryTagId is in palette (forge fallback)
//   ELSE hide entirely
// Refusing v4 prevents the viewer from interpreting a missing PrimaryTagId
// as zero (which would mark every object as "not in palette").
constexpr uint32_t kVersion     = 5u;
constexpr uint32_t kMaxObjects  = 1024u;
constexpr size_t   kMaxAttached = 8;

constexpr uint32_t kErr_Ok                = 0u;
constexpr uint32_t kErr_BulkScanFailed    = 1u;
constexpr uint32_t kErr_BulkReturnedZero  = 2u;
constexpr uint32_t kErr_MapNotReady       = 3u;

#pragma pack(push, 1)
struct SnapshotObject {
    uint32_t Datum;
    uint16_t TypeSig;
    uint8_t  Sig0;
    uint8_t  Sig1;
    float    X, Y, Z;
    float    Health;
    float    Shield;
    uint32_t ModeTagId;
    float    FwdX, FwdY, FwdZ;
    float    UpX,  UpY,  UpZ;
    uint32_t AttachedHandles[8];
    // v5: primary tag datum (the object's class tag id, e.g. the bloc/scen
    // datum). The viewer matches this against the active forge palette to
    // decide whether a missing-mode object should still get a dot marker.
    // Read from obj+0x00 (kObj_PrimaryTag) - the same offset that's already
    // walked locally to drive ResolveHlmtAndModeDatums_Cached.
    uint32_t PrimaryTagId;
};
static_assert(sizeof(SnapshotObject) == 92, "SnapshotObject layout v5");

struct ObjectTable_Shared {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter;
    uint32_t ObjectCount;
    uint32_t SchemaSize;
    uint32_t LastTickMs;
    uint32_t LastError;
    uint32_t Reserved;
    SnapshotObject Objects[kMaxObjects];
};
static_assert(sizeof(ObjectTable_Shared) == 32 + 92 * kMaxObjects, "OBJT v5 size");
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_ObjectTable_Snapshot";

HANDLE              g_hMap   = nullptr;
ObjectTable_Shared* g_Shared = nullptr;

ObjectTable_Shared* EnsureShared()
{
    if (g_Shared) return g_Shared;
    g_hMap = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                (DWORD)sizeof(ObjectTable_Shared), kMapName);
    if (!g_hMap) return nullptr;
    g_Shared = (ObjectTable_Shared*)MapViewOfFile(g_hMap, FILE_MAP_ALL_ACCESS, 0, 0,
                                                  sizeof(ObjectTable_Shared));
    if (!g_Shared) { CloseHandle(g_hMap); g_hMap = nullptr; return nullptr; }

    __try {
        if (g_Shared->Magic != kMagic || g_Shared->Version != kVersion ||
            g_Shared->SchemaSize != (uint32_t)sizeof(SnapshotObject))
        {
            ZeroMemory(g_Shared, sizeof(ObjectTable_Shared));
            g_Shared->Magic      = kMagic;
            g_Shared->Version    = kVersion;
            g_Shared->SchemaSize = (uint32_t)sizeof(SnapshotObject);
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {}
    return g_Shared;
}

template <typename T>
static bool SafeReadAt(const uint8_t* p, size_t off, T& out)
{
    __try { out = *(const T*)(p + off); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

} // namespace

extern "C" __declspec(dllexport) void ObjectTableSnapshot_FramePumpTick()
{
    using namespace HaloMapStudio::Engine;

    ObjectTable_Shared* sh = EnsureShared();
    if (!sh) return;

    // Map-change detection + tag-cache flush already happened in the
    // FramePumpHook dispatcher's RunSnapshotTicks() prelude - by the time
    // we get here the engine epoch is current and ResolveHlmtAndModeDatums
    // _Cached holds only post-swap data. We DO still call the resolver
    // here to fetch the descriptor for this tick.
    uint8_t* desc = ResolveObjectPoolDescriptor();

    if (!desc) {
        __try {
            sh->ObjectCount   = 0;
            sh->LastError     = kErr_MapNotReady;
            sh->LastTickMs    = GetTickCount();
            sh->WriteCounter  = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }

    uint32_t entrySize = 0;
    uint32_t maxCount  = 0;
    uint8_t* entries   = nullptr;
    if (!SafeReadAt(desc, kDesc_EntrySize, entrySize) ||
        !SafeReadAt(desc, kDesc_MaxCount,  maxCount)  ||
        !SafeReadAt(desc, kDesc_Entries,   entries)   ||
        entrySize == 0 || entrySize > 0x100 ||
        maxCount  == 0 || maxCount  > 0x4000 ||
        !entries)
    {
        __try {
            sh->ObjectCount   = 0;
            sh->LastError     = kErr_BulkScanFailed;
            sh->LastTickMs    = GetTickCount();
            sh->WriteCounter  = sh->WriteCounter + 1u;
        } __except (EXCEPTION_EXECUTE_HANDLER) {}
        return;
    }

    // Local datum -> object lookup: same descriptor, same offsets, used for
    // walking attached children without leaving the closure. Cheaper than
    // calling ResolveDatumToObject (which re-reads desc fields).
    auto poolLookup = [&](uint32_t datum) -> uint8_t* {
        if (datum == 0u || datum == 0xFFFFFFFFu) return nullptr;
        uint32_t idx = datum & 0xFFFFu;
        if (idx >= maxCount) return nullptr;
        uint8_t* ent = entries + (size_t)idx * entrySize;
        uint16_t s = 0;
        if (!SafeReadAt(ent, kEntry_Salt, s) || s == 0) return nullptr;
        uint16_t expected = (uint16_t)((datum >> 16) & 0xFFFFu);
        if (s != expected) return nullptr;
        // dropped the flags filter entirely -- non-null
        // obj pointer at +0x10 is the canonical "exists" signal, same
        // logic the main publish loop now uses. Attached-secondary
        // entries (vehicle turrets) have flags with NEITHER 0x01 nor
        // 0x04 set, so any flags-based filter dropped them.
        uint8_t* o = nullptr;
        if (!SafeReadAt(ent, kEntry_ObjPtr, o) || !o) return nullptr;
        return o;
    };

    // Engine-canonical world position fetch. The raw obj+0x54..0x5C region
    // varies per object class - for bipeds it IS world pos, but for forge
    // bloc/scenery it's the FORWARD vector. The engine's Object_GetWorldPosition
    // (haloreach+0x471D38) handles all classes correctly. Signature:
    //   void __fastcall fn(uint32_t datum, float outPos[6])
    // Writes 6 floats to outPos; first 3 are world XYZ. Mirrors what
    // ObjectESP / GetLookedAt / AssassinProxCancel use in the main DLL.
    using GetWorldPosFn = void(__fastcall*)(uint32_t, float*);
    HMODULE hrMod = GetModuleHandleW(L"haloreach.dll");
    GetWorldPosFn fnGetWorldPos = nullptr;
    if (hrMod) {
        fnGetWorldPos = (GetWorldPosFn)((uint8_t*)hrMod + 0x471D38);
    }

    uint32_t outCount = 0;
    __try {
        for (uint32_t i = 0; i < maxCount && outCount < kMaxObjects; ++i) {
            uint8_t* ent = entries + (size_t)i * entrySize;
            uint16_t salt = 0;
            uint16_t flags = 0;
            uint8_t* obj  = nullptr;
            if (!SafeReadAt(ent, kEntry_Salt,  salt)  || salt == 0) continue;
            // was `(flags & kEntry_ActiveMask) == 0` with
            // mask=0x80 (host-replicates). On non-host clients (or even
            // host for non-replicated objects) held weapons / projectiles
            // / equipment had 0x80 == 0 but 0x01 == 1, so they were
            // silently dropped from the publish -- the user-visible
            // symptom was attached=0/N never resolving for weapons.
            // widened from 0x01 to (0x01 | 0x04). In-game
            // diag confirmed scorpion cannon entry (datum 0xE54102D2)
            // had 0x01 cleared but 0x04 set; Ghidra's ClearActiveFlags
            // ANDs with 0xFFFA on delete, confirming both bits are
            // alive indicators.
            //
            // above widening STILL didn't help. attSkipNoTable
            // remains at 31 after relaunch with new DLL. The cannon entries
            // must have NEITHER 0x01 NOR 0x04 set. Dropping the flags
            // filter entirely -- a non-null object pointer at +0x10 IS
            // the canonical "exists in pool" signal. Read the obj ptr
            // first (cheap), gate on null instead. Diag the flags value
            // for the first dropped entry per second so we can identify
            // exactly which bits attached children DO have set, and
            // refine the filter later if needed.
            if (!SafeReadAt(ent, kEntry_ObjPtr, obj)  || !obj) continue;
            // Best-effort read of flags for diag only; failure is
            // tolerable since the obj-ptr gate above already filtered.
            (void)SafeReadAt(ent, kEntry_Flags, flags);

            uint32_t datum = ((uint32_t)salt << 16) | (i & 0xFFFFu);

            uint16_t typeSig = 0;
            float    pos[3]  = {0,0,0};
            float    fwd[3]  = {0,0,0};
            float    up[3]   = {0,0,0};
            float    health  = 0;
            float    shield  = 0;
            uint32_t primaryTag = 0;
            uint32_t childHead  = 0xFFFFFFFFu;

            (void)SafeReadAt(obj, kObj_TypeSig,    typeSig);
            (void)SafeReadAt(obj, kObj_PrimaryTag, primaryTag);
            (void)SafeReadAt(obj, kObj_AttachHead, childHead);
            // Position via engine API - uniform across all object classes.
            if (fnGetWorldPos) {
                float buf[6] = {0};
                __try { fnGetWorldPos(datum, buf); }
                __except (EXCEPTION_EXECUTE_HANDLER) {}
                pos[0] = buf[0]; pos[1] = buf[1]; pos[2] = buf[2];
            }
            // Forward / up vectors still come from raw struct offsets - these
            // are at obj+0x60 / +0x6C per the engine layout (the previous
            // hard-coded +0x54 was the SAME forward vector - pre-fix the
            // viewer was just unknowingly storing it in `pos`).
            __try {
                memcpy(fwd, obj + kObj_Forward,  sizeof(fwd));
                memcpy(up,  obj + kObj_Up,       sizeof(up));
            } __except (EXCEPTION_EXECUTE_HANDLER) {}
            (void)SafeReadAt(obj, kObj_Health, health);
            (void)SafeReadAt(obj, kObj_Shield, shield);

            // primary tag -> hlmt -> mode chain via the cached header-only
            // helper. Cheap once warm; warm-up cost amortises across the
            // first frame on a fresh haloreach load.
            uint32_t hlmt = 0, modeTag = 0;
            ZeroHour::HrTag::ResolveHlmtAndModeDatums_Cached(primaryTag, &hlmt, &modeTag);

            SnapshotObject& dst = sh->Objects[outCount++];
            dst.Datum   = datum;
            dst.TypeSig = typeSig;
            dst.Sig0    = (uint8_t)(typeSig & 0xFFu);
            dst.Sig1    = (uint8_t)((typeSig >> 8) & 0xFFu);
            dst.X       = pos[0];
            dst.Y       = pos[1];
            dst.Z       = pos[2];
            dst.Health  = health;
            dst.Shield  = shield;
            dst.ModeTagId = modeTag;
            dst.FwdX = fwd[0]; dst.FwdY = fwd[1]; dst.FwdZ = fwd[2];
            dst.UpX  = up[0];  dst.UpY  = up[1];  dst.UpZ  = up[2];
            // v5 - primary tag id for forge-palette membership tests on the
            // viewer side. Already read above for the hlmt/mode chain.
            dst.PrimaryTagId = primaryTag;

            // Attached-children walk. Head at obj+0x10, walked via +0x0C
            // next-sibling. Each child resolved through the same pool, so
            // no PlayerListTLS dependency. Cap at kMaxAttached, with a
            // self-loop guard for malformed lists.
            for (size_t k = 0; k < kMaxAttached; ++k) dst.AttachedHandles[k] = 0;
            if (childHead != 0u && childHead != 0xFFFFFFFFu) {
                uint32_t cur  = childHead;
                uint32_t prev = 0xFFFFFFFFu;
                size_t   outIdx = 0;
                for (size_t step = 0; step < 64 && outIdx < kMaxAttached; ++step) {
                    if (cur == 0u || cur == 0xFFFFFFFFu) break;
                    if (cur == prev) break;
                    dst.AttachedHandles[outIdx++] = cur;
                    uint8_t* childObj = poolLookup(cur);
                    if (!childObj) break;
                    uint32_t nextSib = 0xFFFFFFFFu;
                    if (!SafeReadAt(childObj, kObj_NextSibling, nextSib)) break;
                    prev = cur;
                    cur  = nextSib;
                }
            }
        }
        sh->ObjectCount  = outCount;
        sh->LastError    = (outCount == 0) ? kErr_BulkReturnedZero : kErr_Ok;
        sh->LastTickMs   = GetTickCount();
        sh->WriteCounter = sh->WriteCounter + 1u;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        UnmapViewOfFile(sh);
        CloseHandle(g_hMap);
        g_hMap   = nullptr;
        g_Shared = nullptr;
    }
}
