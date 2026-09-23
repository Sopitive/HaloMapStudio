// ForgeObjectTableSnapshot.cpp
// =============================================================================
// Walks haloreach's runtime forge-object table (650 x 76-byte ForgeObject
// structs) each engine frame and publishes the contents to the
// `ZeroHour_ForgeObjectTable_Snapshot` MMF for the viewer to consume.
//
// HOW WE FIND THE TABLE
// ---------------------
// Mjolnir-Forge-Editor (Waffle1434) RE'd the location via an AOB scan in 2022
// - pattern stable across MCC builds because the surrounding code (mov of the
// pointer to a global, sets monitor-slot count to 0x20000, calls
// VariantCommand_HandleExecution) hasn't changed. We port the same scan here:
//
//   AOB: 48 89 05 ?? ?? ?? ?? 89 35 ?? ?? ?? ?? 75 0A C7 05 ?? ?? ?? ?? 02 00
//        00 00 48 8B 05 ?? ?? ?? ?? 4C 8D 4D 20 ...
//   At hit + 28 = rel32 displacement of the second `mov rax, [rip+disp]`.
//   abs = (hit + 28 + 4) + disp
//   forgeBase     = *abs + 0x20000
//   forgeObjects  = forgeBase + 0x10 + 0x19FC
//   gtLabels      = forgeBase + 0x10 + 0x07F4
//
// The +0x10 offset Mjolnir notes ("Something was added inbetween?") is the
// only versioned piece; if a future MCC build moves it, this scanner will
// publish empty slots and log the bad-Show diagnostic, prompting a re-RE.
//
// The 650-entry table is fully walked every tick (650 x 76B = 49,400 B is
// trivial). Each slot's `show` flag (offset 0, u8 - Mjolnir reads as u16 but
// only LSB matters) gates whether the entry is valid; we publish ALL slots
// (so the viewer can show empty-slot stats) and let the viewer's
// `ForgeObjectInfo` filter skip Show==0 entries.
//
// We also try to find each forge slot's live engine datum by:
//   1. Reading the slot's world position.
//   2. Looking up engine objects via the existing ResolveObjectPoolDescriptor
//      from EngineThreadResolver.h.
//   3. Finding the runtime object whose +0x54..+0x5C XYZ matches within an
//      epsilon AND whose +0x00 primary-tag matches the forge entry's tag.
// This is best-effort - if no live object matches, EngineDatum is published
// as 0 and the viewer falls back to position-based association.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <vector>

extern "C" void ZH_Logf(const char* fmt, ...);
extern "C" void ForgeGtLabelsMmf_FramePumpTick(const uint8_t* gtLabelsBase);

namespace {

constexpr uint32_t kMagic         = 0x4F475246u;  // 'FRGO'
constexpr uint32_t kVersion       = 1u;
constexpr uint32_t kMaxObjects    = 650u;
constexpr size_t   kForgeObjSize  = 76u;
constexpr size_t   kForgeBaseHdr  = 0x10u;
constexpr size_t   kOff_Objects   = kForgeBaseHdr + 0x19FCu;
constexpr size_t   kOff_GtLabels  = kForgeBaseHdr + 0x07F4u;

// Error codes published in Header.LastError for the viewer's diagnostic UI.
constexpr uint32_t kErr_None              = 0u;
constexpr uint32_t kErr_HaloreachUnloaded = 1u;
constexpr uint32_t kErr_AobScanFailed     = 2u;
constexpr uint32_t kErr_BasePointerNull   = 3u;
constexpr uint32_t kErr_FirstSlotUnread   = 4u;

#pragma pack(push, 1)
struct Entry {
    uint32_t Slot;
    uint32_t EngineDatum;
    uint16_t Show;
    uint16_t ItemCategory;
    uint32_t IdExt;
    float    PosX, PosY, PosZ;
    float    FwdX, FwdY, FwdZ;
    float    UpX,  UpY,  UpZ;
    uint16_t SpawnRelativeToMapIndex;
    uint8_t  ItemVariant;
    uint8_t  _Pad0;
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
    uint8_t  _Pad1;
    uint32_t _Pad2;
};
static_assert(sizeof(Entry) == 88, "Entry must be 88B (matches the hms-ipc mirror)");

struct Header {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter;
    uint32_t EntryCount;
    uint32_t RequestCounter;
    uint32_t LastError;
    uint32_t RuntimeTableVA;
    uint32_t _Pad;
};

struct Snapshot {
    Header   H;
    Entry    Entries[kMaxObjects];
};
#pragma pack(pop)

const wchar_t kMapName[] = L"ZeroHour_ForgeObjectTable_Snapshot";

// MMF backing.
HANDLE     g_FileMapping  = nullptr;
Snapshot*  g_SharedView   = nullptr;

// Cached pointers - invalidated on map-epoch bump or RequestCounter edge.
uint64_t   g_LastEpoch         = 0;
uint32_t   g_LastSeenRequest   = 0;
uint8_t*   g_ForgeObjectsBase  = nullptr;
uint8_t*   g_GtLabelsBase      = nullptr;
uint32_t   g_RetryCountdown    = 0;  // throttle AOB rescans

constexpr uint32_t kRetryFrames = 60;  // ~1s @ 60fps

// =============================================================================
// AOB scanner - Mjolnir's pattern. `??` bytes are wildcards.
// =============================================================================
struct PatternByte { uint8_t value; bool wild; };

bool ParsePattern(const char* str, std::vector<PatternByte>& out)
{
    out.clear();
    while (*str) {
        if (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r') { ++str; continue; }
        if (str[0] == '?' && str[1] == '?') {
            out.push_back({ 0, true });
            str += 2;
            continue;
        }
        auto hex = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            return -1;
        };
        int hi = hex(str[0]); if (hi < 0) return false;
        int lo = hex(str[1]); if (lo < 0) return false;
        out.push_back({ (uint8_t)((hi << 4) | lo), false });
        str += 2;
    }
    return !out.empty();
}

bool FindPattern(uint8_t* base, size_t size, const std::vector<PatternByte>& pat, size_t& outOffset)
{
    if (pat.empty() || pat.size() > size) return false;
    size_t patLen = pat.size();
    size_t scanEnd = size - patLen;
    __try {
        for (size_t i = 0; i <= scanEnd; ++i) {
            bool ok = true;
            for (size_t j = 0; j < patLen; ++j) {
                if (pat[j].wild) continue;
                if (base[i + j] != pat[j].value) { ok = false; break; }
            }
            if (ok) { outOffset = i; return true; }
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return false;
}

// Resolve haloreach.dll's mapped size by walking PE headers (so we don't scan
// past the module's commit region and AV).
size_t GetModuleSize(HMODULE mod)
{
    if (!mod) return 0;
    uint8_t* base = (uint8_t*)mod;
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)base;
    if (!dos || dos->e_magic != IMAGE_DOS_SIGNATURE) return 0;
    IMAGE_NT_HEADERS64* nt = (IMAGE_NT_HEADERS64*)(base + dos->e_lfanew);
    if (!nt || nt->Signature != IMAGE_NT_SIGNATURE) return 0;
    return nt->OptionalHeader.SizeOfImage;
}

bool ResolveForgeBases(uint8_t*& outObjects, uint8_t*& outGtLabels, uint32_t& outErr)
{
    outObjects = nullptr;
    outGtLabels = nullptr;
    outErr = kErr_None;

    HMODULE hr = GetModuleHandleW(L"haloreach.dll");
    if (!hr) { outErr = kErr_HaloreachUnloaded; return false; }
    uint8_t* base = (uint8_t*)hr;
    size_t size = GetModuleSize(hr);
    if (size == 0) { outErr = kErr_HaloreachUnloaded; return false; }

    // Mjolnir's pattern (forge_ptr_aob).
    static const char* kAob =
        "48 89 05 ?? ?? ?? ?? "
        "89 35 ?? ?? ?? ?? "
        "75 0A "
        "C7 05 ?? ?? ?? ?? 02 00 00 00 "
        "48 8B 05 ?? ?? ?? ?? "
        "4C 8D 4D 20 "
        "89 74 24 28 "
        "4C 8D 45 28 "
        "BA 00 00 02 00 "
        "48 89 45 28 "
        "33 C9 "
        "C7 45 20 00 00 38 01 "
        "C7 44 24 20 08 00 00 00 "
        "E8 ?? ?? ?? ??";

    static std::vector<PatternByte> sPattern;
    if (sPattern.empty()) ParsePattern(kAob, sPattern);

    size_t hit = 0;
    if (!FindPattern(base, size, sPattern, hit)) {
        outErr = kErr_AobScanFailed;
        return false;
    }

    // `mov rax, [rip+disp]` is at hit+0x19; rel32 displacement at +0x1C
    // (= 7-byte mov ptr stored + 7-byte mov idx + 2-byte jne + 10-byte cmov =
    //  hit+25 to +28 is the disp). Match Mjolnir's offsets: address_offset=28,
    //  next_offset = 32.
    int32_t disp = 0;
    __try {
        disp = *(int32_t*)(base + hit + 28);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        outErr = kErr_AobScanFailed;
        return false;
    }
    uint8_t* nextInstr = base + hit + 32;
    uint8_t* absAddr   = nextInstr + disp;

    // Read the pointer-of-pointer.
    void* p = nullptr;
    __try { p = *(void**)absAddr; } __except (EXCEPTION_EXECUTE_HANDLER) { p = nullptr; }
    if (!p) { outErr = kErr_BasePointerNull; return false; }

    uint8_t* forgeBase = (uint8_t*)p + 0x20000;
    outObjects  = forgeBase + kOff_Objects;
    outGtLabels = forgeBase + kOff_GtLabels;
    return true;
}

// SEH-safe read of one ForgeObject record (76 bytes) into a local buffer.
bool SafeReadForgeObject(const uint8_t* src, uint8_t dst[kForgeObjSize])
{
    __try {
        memcpy(dst, src, kForgeObjSize);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Per-runtime-object forge-index back-reference. Each s_object_data has its
// owning forge slot index stored at +0x1C (u16). Authoritative - way more
// reliable than position-match or trusting slot+0x04 (idExt), which engine
// firmware may or may not write back. 0xFFFF = "not a forge object".
// (Provenance: ZeroHourDLL FindForgeObjectByForgeIndex; same offset across
//  Reach builds because it's emitted by the spawn path itself.)
constexpr size_t kObj_ForgeIndex = 0x1C;

// Walk the entire engine object pool once and build a forge-index ->
// engine-datum table. Linear in pool size (~4K entries worst case) - runs
// once per snapshot publish, cheap. Returns false if the descriptor can't
// be read; on success the array is populated with packed datums (salt<<16 |
// index) for every forge object in the pool, indexed by forge-slot index.
// Unused slots are left at 0.
bool BuildForgeIndexToDatumMap(uint8_t* desc, uint32_t outMap[kMaxObjects])
{
    memset(outMap, 0, sizeof(uint32_t) * kMaxObjects);
    if (!desc) return false;

    using namespace HaloMapStudio::Engine;
    uint32_t entrySize = 0;
    uint32_t maxCount  = 0;
    void*    rawBase   = nullptr;
    if (!SafeReadT(desc + kDesc_EntrySize, entrySize) || entrySize == 0) return false;
    if (!SafeReadT(desc + kDesc_MaxCount,  maxCount)  || maxCount  == 0) return false;
    if (!SafeReadPtr(desc + kDesc_Entries, rawBase)   || !rawBase)        return false;
    if (maxCount > 0x4000) return false; // guard

    uint8_t* table = (uint8_t*)rawBase;
    for (uint32_t i = 0; i < maxCount; ++i) {
        uint8_t* entry = table + (size_t)i * (size_t)entrySize;
        uint16_t salt = 0;
        uint8_t  flags = 0;
        void*    obj   = nullptr;
        if (!SafeReadT (entry + kEntry_Salt,   salt))  continue;
        if (salt == 0) continue;
        if (!SafeReadT (entry + kEntry_Flags,  flags)) continue;
        if ((flags & kEntry_ActiveMask) == 0) continue;
        if (!SafeReadPtr(entry + kEntry_ObjPtr, obj) || !obj) continue;
        uint16_t fi = 0xFFFF;
        if (!SafeReadT((uint8_t*)obj + kObj_ForgeIndex, fi)) continue;
        if (fi == 0xFFFF) continue;
        if (fi >= kMaxObjects) continue;
        outMap[fi] = ((uint32_t)salt << 16) | i;
    }
    return true;
}

bool EnsureMmf()
{
    if (g_SharedView) return true;
    size_t totalSize = sizeof(Snapshot);
    g_FileMapping = CreateFileMappingW(
        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
        (DWORD)((uint64_t)totalSize >> 32), (DWORD)(totalSize & 0xFFFFFFFFu),
        kMapName);
    if (!g_FileMapping) return false;
    g_SharedView = (Snapshot*)MapViewOfFile(g_FileMapping, FILE_MAP_ALL_ACCESS, 0, 0, totalSize);
    if (!g_SharedView) {
        CloseHandle(g_FileMapping);
        g_FileMapping = nullptr;
        return false;
    }
    // First-touch stamping.
    g_SharedView->H.Magic   = kMagic;
    g_SharedView->H.Version = kVersion;
    return true;
}

} // namespace

// =============================================================================
// Per-frame entry point - called from FramePumpHook.cpp.
// =============================================================================
extern "C" void ForgeObjectTableSnapshot_FramePumpTick()
{
    if (!EnsureMmf()) return;

    // Edge-detect on the RequestCounter - viewer can force a re-walk by
    // bumping it. We compare to the value we last saw; mismatch invalidates
    // our cached forge-base pointer.
    uint32_t reqCtr = g_SharedView->H.RequestCounter;
    if (reqCtr != g_LastSeenRequest) {
        g_ForgeObjectsBase = nullptr;
        g_GtLabelsBase     = nullptr;
        g_RetryCountdown   = 0;
        g_LastSeenRequest  = reqCtr;
    }

    // Map-epoch detection - bump invalidates all caches.
    uint64_t curEpoch = HaloMapStudio::Engine::GetMapEpoch();
    if (curEpoch != g_LastEpoch) {
        g_ForgeObjectsBase = nullptr;
        g_GtLabelsBase     = nullptr;
        g_RetryCountdown   = 0;
        g_LastEpoch        = curEpoch;
    }

    uint32_t err = kErr_None;
    if (!g_ForgeObjectsBase) {
        if (g_RetryCountdown > 0) {
            --g_RetryCountdown;
            // Publish an empty snapshot with the last error so the viewer
            // can show "still trying" without zero-spam.
            g_SharedView->H.EntryCount = 0;
            ++g_SharedView->H.WriteCounter;
            return;
        }
        uint8_t* objs = nullptr;
        uint8_t* labels = nullptr;
        if (!ResolveForgeBases(objs, labels, err)) {
            g_SharedView->H.LastError = err;
            g_SharedView->H.EntryCount = 0;
            g_RetryCountdown = kRetryFrames;
            ++g_SharedView->H.WriteCounter;
            return;
        }
        g_ForgeObjectsBase = objs;
        g_GtLabelsBase     = labels;
        ZH_Logf("[HaloMapStudioDLL] forge object table resolved @ %p (gtLabels @ %p)\n",
                g_ForgeObjectsBase, g_GtLabelsBase);

        // One-shot diag dump of the first 512B at gtLabels for layout
        // verification. The Mjolnir-Forge-Editor port reads gtLabels as a
        // packed sequence of variable-length NUL-terminated ASCII strings
        // (terminated by an empty string); we dump raw bytes here to confirm
        // the encoding/stride matches before publishing via MMF.
        if (g_GtLabelsBase) {
            __try {
                char hexbuf[3 * 64 + 16];
                char ascbuf[64 + 16];
                ZH_Logf("[HaloMapStudioDLL] gtLabels dump (first 512 B):\n");
                for (int row = 0; row < 8; ++row) {
                    int hpos = 0;
                    int apos = 0;
                    for (int col = 0; col < 64; ++col) {
                        uint8_t b = g_GtLabelsBase[row * 64 + col];
                        // 3 bytes per hex pair ("XX ")
                        const char* hex = "0123456789ABCDEF";
                        hexbuf[hpos++] = hex[b >> 4];
                        hexbuf[hpos++] = hex[b & 0xF];
                        hexbuf[hpos++] = ' ';
                        ascbuf[apos++] = (b >= 0x20 && b < 0x7F) ? (char)b : '.';
                    }
                    hexbuf[hpos] = 0;
                    ascbuf[apos] = 0;
                    ZH_Logf("[HaloMapStudioDLL]   %04X  %s  %s\n", row * 64, hexbuf, ascbuf);
                }
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                ZH_Logf("[HaloMapStudioDLL] gtLabels dump SEH-faulted\n");
            }
        }
    }

    // Stamp diagnostic VA for the viewer's status panel.
    g_SharedView->H.RuntimeTableVA = (uint32_t)((uintptr_t)g_ForgeObjectsBase & 0xFFFFFFFFu);

    // Resolve the object pool once per tick - we'll join forge -> engine
    // datum by position match.
    uint8_t* desc = nullptr;
    __try { desc = HaloMapStudio::Engine::ResolveObjectPoolDescriptor(); }
    __except (EXCEPTION_EXECUTE_HANDLER) { desc = nullptr; }

    // Build the forge-index -> engine-datum map ONCE per tick by walking
    // the object pool and reading each runtime object's +0x1C forge-index
    // back-reference. This is the authoritative mapping (set by the
    // engine's spawn path itself) - replaces the previous position-match
    // heuristic AND the unreliable slot+0x04 idExt fallback.
    static uint32_t forgeIdxToDatum[kMaxObjects];
    __try { BuildForgeIndexToDatumMap(desc, forgeIdxToDatum); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        memset(forgeIdxToDatum, 0, sizeof(forgeIdxToDatum));
    }

    uint8_t obj[kForgeObjSize];
    uint32_t written = 0;
    uint32_t anySlotsRead = 0;

    for (uint32_t i = 0; i < kMaxObjects; ++i) {
        const uint8_t* src = g_ForgeObjectsBase + (size_t)i * kForgeObjSize;
        if (!SafeReadForgeObject(src, obj)) {
            if (i == 0) {
                // Couldn't even read slot 0 - the base pointer drifted (likely
                // a map switch we missed). Drop caches and try again next tick.
                g_ForgeObjectsBase = nullptr;
                g_GtLabelsBase     = nullptr;
                g_SharedView->H.LastError = kErr_FirstSlotUnread;
                g_SharedView->H.EntryCount = 0;
                ++g_SharedView->H.WriteCounter;
                return;
            }
            continue;
        }
        anySlotsRead++;

        // Decode the 76-byte ForgeObject - see Mjolnir's struct.
        Entry& e = g_SharedView->Entries[i];
        e.Slot        = i;
        e.EngineDatum = 0;

        uint16_t show = *(uint16_t*)(obj + 0);   // Mjolnir uses u16; LSB is real
        uint16_t cat  = *(uint16_t*)(obj + 2);
        uint32_t idex = *(uint32_t*)(obj + 4);
        e.Show         = show;
        e.ItemCategory = cat;
        e.IdExt        = idex;

        e.PosX = *(float*)(obj +  8); e.PosY = *(float*)(obj + 12); e.PosZ = *(float*)(obj + 16);
        e.FwdX = *(float*)(obj + 20); e.FwdY = *(float*)(obj + 24); e.FwdZ = *(float*)(obj + 28);
        e.UpX  = *(float*)(obj + 32); e.UpY  = *(float*)(obj + 36); e.UpZ  = *(float*)(obj + 40);

        e.SpawnRelativeToMapIndex = *(uint16_t*)(obj + 44);
        e.ItemVariant = obj[46];

        e.Width  = *(float*)(obj + 48);
        e.Length = *(float*)(obj + 52);
        e.Top    = *(float*)(obj + 56);
        e.Bottom = *(float*)(obj + 60);

        e.Shape         = obj[64];
        e.SpawnSequence = (int8_t)obj[65];
        e.SpawnTime     = obj[66];
        e.CachedType    = obj[67];
        e.GtLabelIndex  = *(uint16_t*)(obj + 68);
        e.Flags         = obj[70];
        e.Team          = obj[71];
        e.OtherInfoA    = obj[72];
        e.OtherInfoB    = obj[73];
        e.Color         = obj[74];
        e._Pad0 = 0; e._Pad1 = 0; e._Pad2 = 0;

        if (show) {
            ++written;
            // Primary join: read the runtime back-reference at obj+0x1C.
            // The map was populated once above; if no runtime object owns
            // this slot (e.g. authoring entry not yet spawned, or the slot
            // is despawned this tick) EngineDatum stays 0.
            e.EngineDatum = forgeIdxToDatum[i];
            // Secondary fallback: trust slot+0x04 idExt if the pool-walk
            // missed the slot. Some engine code paths write the runtime
            // datum back here as a mirror, but we treat it as advisory.
            if (e.EngineDatum == 0 && idex != 0 && idex != 0xFFFFFFFFu) {
                e.EngineDatum = idex;
            }
        }
    }

    g_SharedView->H.LastError  = kErr_None;
    g_SharedView->H.EntryCount = kMaxObjects;  // we publish all slots; consumer filters
    (void)written;  // available for future telemetry
    (void)anySlotsRead;
    ++g_SharedView->H.WriteCounter;

    // Publish the gametype-label table to its own MMF. Internal change-
    // detect skips the write when nothing differs vs the last tick (so the
    // cost is one memcmp per frame on the steady state).
    __try { ForgeGtLabelsMmf_FramePumpTick(g_GtLabelsBase); }
    __except (EXCEPTION_EXECUTE_HANDLER) {}
}

// Public accessor for the edit handler - gives it the resolved runtime base.
// Returns nullptr if not yet resolved (caller should bail with NotReady).
extern "C" uint8_t* HaloMapStudio_ForgeObjectTable_GetBase()
{
    return g_ForgeObjectsBase;
}

// Public accessor for the gt-labels base.
extern "C" uint8_t* HaloMapStudio_ForgeObjectTable_GetGtLabelsBase()
{
    return g_GtLabelsBase;
}
