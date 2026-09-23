// MegaloObjectSnapshot.cpp (HaloMapStudioDLL)
// =============================================================================
// Walks haloreach's runtime *megalo object* datum-array each engine frame and
// publishes its contents to the `HaloMapStudio_MegaloObjects` MMF for the
// viewer's debug overlay. Megalo objects are a DEDICATED engine structure (the
// Reach Megalo gametype VM's own object array) - NOT forge objects. This walker
// runs UNCONDITIONALLY (megalo objects only exist in MP gametypes, which is the
// whole point); it is NOT gated on forge mode / the forge table.
//
// RE recipe (see docs/rendering/megalo_objects_re.md):
//   tlsIndex = *(u32*)(haloreach.dll + 0xC17B18)            // kRva_MonitorTlsIndex
//   tlsBlock = GetTLSArrayBase()[tlsIndex]
//   g_megalo_object_data_allocator = *(void**)(tlsBlock + 0x4F0)   // an s_data_array*
//       -> "megalo_objects" datum array: capacity 0x200 (512), element 0x94 (148B)
//   Walk slots 0..maxCount; skip salt==0 or +0x38 == 0xFFFFFFFF (no bound object).
//
// Per-element s_megalo_object (148 bytes):
//   +0x00 u32   this megalo object's own datum (salt<<16 | index)
//   +0x38 u32   bound object datum (the sandbox/forge object; 0xFFFFFFFF = none)
//   +0x88 f32x2 boundary extents pair (shape geometry)
//   +0x90 u32   category / label / type index (the megalo "what is this" key;
//               no static human-readable name - render the raw index)
//
// World position: resolve the +0x38 object datum through the SAME object-pool
// resolver the other snapshots use (ResolveObjectPoolDescriptor +
// ResolveDatumToObject) and read obj+0x54..0x5C - exactly like
// ObjectTableSnapshot.cpp. No engine call needed.
//
// Team: there is no statically-pinned team field on s_megalo_object. We publish
// Team = 0xFF (unresolved) rather than fabricate one.
//
// All reads are SEH-guarded; the megalo array is only coherent on the engine
// frame-pump thread, so we publish from FramePumpHook.cpp like the others.
//
// MMF WIRE FORMAT  (map name "HaloMapStudio_MegaloObjects")
// ---------------------------------------------------------------------------
//   Header (32 B):
//     u32 Magic          'MGLO' (0x4F4C474D LE)
//     u32 Version        1
//     u32 WriteCounter   bumped AFTER publish (torn-read fence)
//     u32 EntryCount     0..kMaxObjects (count of LIVE rows written this tick)
//     u32 LastError      diag, 0 on success
//     u32 RuntimeAllocVA low 32b of the resolved allocator addr (diag)
//     u32 _Pad0
//     u32 _Pad1
//   Entry[] (36 B each, EntryCount of them):
//     u32 MegaloDatum    elem +0x00
//     u32 ObjectDatum    elem +0x38
//     u32 CategoryIndex  elem +0x90
//     f32 PosX, PosY, PosZ   resolved world position (0,0,0 if datum unresolved)
//     f32 BoundExtA, BoundExtB  elem +0x88
//     u8  Team           0xFF = unresolved (no static field)
//     u8  _Pad0, _Pad1, _Pad2
//   Total: 32 + 512 * 36 = 18,464 bytes.
// =============================================================================

#include "pch.h"
#include "EngineThreadResolver.h"

#include <windows.h>
#include <cstdint>
#include <cstring>

extern "C" void ZH_Logf(const char* fmt, ...);

namespace {

constexpr uint32_t kMagic        = 0x4F4C474Du;  // 'MGLO'
constexpr uint32_t kVersion      = 1u;
constexpr uint32_t kMaxObjects   = 512u;         // 0x200 capacity
constexpr size_t   kMegaloAllocOffset = 0x4F0;   // TLS block -> s_data_array*

// Per-element s_megalo_object offsets (148-byte stride).
constexpr size_t kElem_Datum   = 0x00;  // u32 own datum head (salt high word)
constexpr size_t kElem_ObjDatum= 0x38;  // u32 bound object datum
constexpr size_t kElem_Extents = 0x88;  // f32x2 boundary extents
constexpr size_t kElem_Category= 0x90;  // u32 category/label/type index

constexpr uint32_t kErr_None             = 0u;
constexpr uint32_t kErr_HaloreachUnloaded= 1u;
constexpr uint32_t kErr_TlsUnresolved    = 2u;
constexpr uint32_t kErr_AllocNull        = 3u;
constexpr uint32_t kErr_ArrayBoundsBad   = 4u;

#pragma pack(push, 1)
struct Entry {
    uint32_t MegaloDatum;
    uint32_t ObjectDatum;
    uint32_t CategoryIndex;
    float    PosX, PosY, PosZ;
    float    BoundExtA, BoundExtB;
    uint8_t  Team;
    uint8_t  _Pad0, _Pad1, _Pad2;
};
static_assert(sizeof(Entry) == 36, "Megalo Entry must be 36B (matches the hms-ipc mirror)");

struct Header {
    uint32_t Magic;
    uint32_t Version;
    uint32_t WriteCounter;
    uint32_t EntryCount;
    uint32_t LastError;
    uint32_t RuntimeAllocVA;
    uint32_t _Pad0;
    uint32_t _Pad1;
};

struct Snapshot {
    Header H;
    Entry  Entries[kMaxObjects];
};
#pragma pack(pop)
static_assert(sizeof(Snapshot) == 32 + 36 * kMaxObjects, "MGLO MMF size");

const wchar_t kMapName[] = L"HaloMapStudio_MegaloObjects";

HANDLE     g_FileMapping = nullptr;
Snapshot*  g_SharedView  = nullptr;

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
    g_SharedView->H.Magic   = kMagic;
    g_SharedView->H.Version = kVersion;
    return true;
}

// Resolve the megalo allocator (s_data_array*) from the TLS block.
// Returns nullptr (and sets *err) on any failure.
uint8_t* ResolveMegaloAllocator(uint32_t& err)
{
    err = kErr_None;
    if (!GetModuleHandleW(L"haloreach.dll")) { err = kErr_HaloreachUnloaded; return nullptr; }

    using namespace HaloMapStudio::Engine;
    static uint32_t s_tlsIndex = 0xFFFFFFFFu;
    if (s_tlsIndex == 0xFFFFFFFFu)
        s_tlsIndex = ResolveTlsIndexFromModule(L"haloreach.dll");
    if (s_tlsIndex == 0xFFFFFFFFu) { err = kErr_TlsUnresolved; return nullptr; }

    void** tlsArray = GetTLSArrayBase();
    if (!tlsArray) { err = kErr_TlsUnresolved; return nullptr; }

    void* tlsBlock = nullptr;
    __try { tlsBlock = tlsArray[s_tlsIndex]; }
    __except (EXCEPTION_EXECUTE_HANDLER) { tlsBlock = nullptr; }
    if (!tlsBlock) { err = kErr_TlsUnresolved; return nullptr; }

    void* alloc = nullptr;
    if (!SafeReadPtr((uint8_t*)tlsBlock + kMegaloAllocOffset, alloc) || !alloc) {
        err = kErr_AllocNull; return nullptr;
    }
    return (uint8_t*)alloc;
}

} // namespace

// =============================================================================
// Per-frame entry point - called from FramePumpHook.cpp (engine thread).
// =============================================================================
extern "C" void MegaloObjectSnapshot_FramePumpTick()
{
    if (!EnsureMmf()) return;

    using namespace HaloMapStudio::Engine;

    uint32_t err = kErr_None;
    uint8_t* alloc = ResolveMegaloAllocator(err);
    if (!alloc) {
        g_SharedView->H.LastError  = err;
        g_SharedView->H.EntryCount = 0;
        ++g_SharedView->H.WriteCounter;
        return;
    }
    g_SharedView->H.RuntimeAllocVA = (uint32_t)((uintptr_t)alloc & 0xFFFFFFFFu);

    // Read the s_data_array header (SAME offsets as the object pool descriptor).
    uint32_t entrySize = 0, maxCount = 0;
    void*    entries   = nullptr;
    if (!SafeReadT(alloc + kDesc_EntrySize, entrySize) || entrySize == 0 ||
        !SafeReadT(alloc + kDesc_MaxCount,  maxCount)  || maxCount  == 0 ||
        !SafeReadPtr(alloc + kDesc_Entries, entries)   || !entries)
    {
        g_SharedView->H.LastError  = kErr_ArrayBoundsBad;
        g_SharedView->H.EntryCount = 0;
        ++g_SharedView->H.WriteCounter;
        return;
    }
    // Sanity-bound the walk (expect entrySize 0x94, maxCount 0x200).
    if (entrySize > 0x100 || maxCount > kMaxObjects) {
        // A torn read mid-map-swap, or an unexpected layout. Clamp/bail safely.
        if (maxCount > kMaxObjects) maxCount = kMaxObjects;
        if (entrySize == 0 || entrySize > 0x100) {
            g_SharedView->H.LastError  = kErr_ArrayBoundsBad;
            g_SharedView->H.EntryCount = 0;
            ++g_SharedView->H.WriteCounter;
            return;
        }
    }

    // Resolve the object pool ONCE so we can turn each bound object datum into a
    // live world position (same approach as ObjectTableSnapshot.cpp).
    uint8_t* objDesc = nullptr;
    __try { objDesc = ResolveObjectPoolDescriptor(); }
    __except (EXCEPTION_EXECUTE_HANDLER) { objDesc = nullptr; }

    uint8_t* table = (uint8_t*)entries;
    uint32_t written = 0;

    for (uint32_t i = 0; i < maxCount && written < kMaxObjects; ++i) {
        uint8_t* elem = table + (size_t)i * (size_t)entrySize;

        uint16_t salt = 0;
        if (!SafeReadT(elem + kEntry_Salt, salt) || salt == 0) continue;  // empty slot

        uint32_t objDatum = 0xFFFFFFFFu;
        if (!SafeReadT(elem + kElem_ObjDatum, objDatum)) continue;
        if (objDatum == 0xFFFFFFFFu) continue;  // no bound object - skip

        uint32_t megaloDatum = 0;
        uint32_t category    = 0;
        float    extA = 0.0f, extB = 0.0f;
        SafeReadT(elem + kElem_Datum,    megaloDatum);
        SafeReadT(elem + kElem_Category, category);
        SafeReadT(elem + kElem_Extents,        extA);
        SafeReadT(elem + kElem_Extents + 4,    extB);

        // World position via the bound object datum.
        float px = 0.0f, py = 0.0f, pz = 0.0f;
        if (objDesc) {
            uint8_t* obj = nullptr;
            __try { obj = ResolveDatumToObject(objDesc, objDatum); }
            __except (EXCEPTION_EXECUTE_HANDLER) { obj = nullptr; }
            if (obj) {
                SafeReadT(obj + kObj_WorldPos,     px);
                SafeReadT(obj + kObj_WorldPos + 4, py);
                SafeReadT(obj + kObj_WorldPos + 8, pz);
            }
        }

        Entry& e = g_SharedView->Entries[written++];
        e.MegaloDatum   = megaloDatum;
        e.ObjectDatum   = objDatum;
        e.CategoryIndex = category;
        e.PosX = px; e.PosY = py; e.PosZ = pz;
        e.BoundExtA = extA;
        e.BoundExtB = extB;
        e.Team  = 0xFF;  // no static team field on s_megalo_object - unresolved
        e._Pad0 = e._Pad1 = e._Pad2 = 0;
    }

    g_SharedView->H.LastError  = kErr_None;
    g_SharedView->H.EntryCount = written;
    ++g_SharedView->H.WriteCounter;
}
