// MapCacheCommon.h
// =============================================================================
// Shared cache-file infrastructure for the native MCC HaloReach .map parsers.
// Exposes the open-handle / header-parse / resource-gestalt / resource-layout
// machinery that the bitmap, model, BSP and lightmap parsers share.
//
// All types here are internal; nothing from this header is exported from the
// DLL. The public C exports are in MapBitmapParser.h and MapModelParser.h.
//
// Threading model
//   * One CacheHandle per open .map file. Opening / closing is serialised via
//     a global g_handlesMutex.
//   * Lazy parses (gestalt, layout, per-tag) are guarded by per-cache mutex
//     CacheHandle::parseMutex. The cache header is parsed eagerly on Open.
//   * Decode itself is lock-free once parsing is done - the mmap pointer +
//     all parsed tables are immutable thereafter.
// =============================================================================

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <windows.h>

namespace zh_mcc {

// ----- Build / cache version detection ----------------------------------------

enum class CacheType : int {
    Unknown        = 0,
    MccHaloReach,      // release through update 2 (single CacheHeader at 288)
    MccHaloReachU3,    // U3-U7
    MccHaloReachU8,    // U8 (CacheHeaderU8 layout, build string @ 160)
    MccHaloReachU10,
    MccHaloReachU13,
};

struct PointerExpander {
    int64_t magic;
    int64_t Expand(int32_t pointer) const {
        return ((int64_t)pointer << 2) + magic;
    }
};

PointerExpander MakeExpander(const char* buildString);

// ----- Tag index entries -------------------------------------------------------

struct TagClass {
    uint32_t classId;
    char     classCode[5];   // 4 chars + null
};

struct TagEntry {
    int32_t      classIndex;
    uint32_t     metaPointerRaw;  // 32-bit raw pointer (expand via expander then translate)
    char         classCode[5];
    std::string  tagName;
};

// ----- Resource tables ---------------------------------------------------------

struct ResourceFixup {
    int32_t  unknown;
    int32_t  offset;     // mask 0x0FFFFFFF gives byte offset into the page payload
};

struct ResourceEntry {
    int32_t  resourcePointer;
    int32_t  fixupOffset;
    int32_t  fixupSize;
    int16_t  segmentIndex;
    // Lazily-populated. The resource-fixup block lives at +40 in the entry
    // (count + 32-bit pointer). Loaded on demand by EnsureResourceFixups so
    // the bitmap path doesn't pay for it.
    std::vector<ResourceFixup> fixups;
    bool fixupsLoaded = false;
    uint32_t fixupsBlockPointer = 0;  // raw, expanded -> tag space
    int32_t  fixupsBlockCount   = 0;
};

struct PageBlock {
    int16_t cacheIndex;
    int32_t dataOffset;
    int32_t compressedSize;
    int32_t decompressedSize;
};

struct SegmentBlock {
    int16_t primaryPageIndex;
    int16_t secondaryPageIndex;
    int32_t primaryPageOffset;
    int32_t secondaryPageOffset;
};

// ----- Open-handle state -------------------------------------------------------

struct CacheHandle {
    HANDLE       hFile = INVALID_HANDLE_VALUE;
    HANDLE       hMap  = nullptr;
    const uint8_t* base = nullptr;
    size_t       size = 0;
    std::wstring path;

    CacheType    cacheType = CacheType::Unknown;
    char         buildString[33] = {0};
    PointerExpander expander{ 0x50000000 };

    int64_t      headerMagic = 0;
    int64_t      tagMagic    = 0;

    std::vector<TagClass> classes;
    std::vector<TagEntry> tags;

    bool                       gestaltParsed = false;
    std::vector<ResourceEntry> resourceEntries;

    bool                          layoutParsed = false;
    // LAYOUT_RACE_FIX: ParseLayoutTable was racing - multiple
    // mesh-tag parser threads (the viewer's parallel loader) entered concurrently,
    // both saw layoutParsed=false, both resize()d cache->pages/segments,
    // and the second resize freed the buffer the first thread was still
    // filling -> heap corruption crash at vector::_Resize_reallocate inside
    // RtlFreeHeap. Lock-guard the whole parse to serialize the first-call
    // work; subsequent threads block briefly then early-return on the now-
    // true flag.
    std::mutex                    layoutMutex;
    std::vector<PageBlock>        pages;
    std::vector<SegmentBlock>     segments;
    uint32_t                      dataTableAddress = 0;

    // PERF - decompressed-page LRU cache. shared.map atlas pages
    // hold many bitmaps; without this each bitmap re-inflates the WHOLE page
    // (the p99 1.2-1.4s/bitmap cost on forge_halo/sword_slayer). Keyed by the
    // resolved pageIndex (stable for the lifetime of this handle); bounded by
    // bytes (MMS_PAGE_CACHE_MB, default 384; 0 disables). Has its OWN mutex
    // because the viewer's texture decode - and thus ReadResourceData - now runs
    // concurrently across cores. LRU = monotonic lastUsed tick + linear-scan
    // eviction (live page count is small). Freed with the handle, so a map
    // reload (new handle) starts clean - no cross-map staleness.
    struct DecompPage { std::shared_ptr<const std::vector<uint8_t>> data; /* #lockfree-copy: shared so readers copy OUTSIDE pageCacheMutex */ uint64_t lastUsed = 0; };
    std::mutex                          pageCacheMutex;
    std::unordered_map<int, DecompPage> pageCache;
    uint64_t                            pageCacheTick = 0;
    size_t                              pageCacheBytes = 0;
    // SINGLE-FLIGHT inflate. With 30 decode threads all missing on the same (up to 250 MB)
    // deflate page at once, every thread allocated its own full-page buffer and inflated it
    // (measured: 12.6 GB RSS peak on Forge World with the cache disabled, 10.4 GB with it, of
    // which several GB were N concurrent copies of the SAME page). The first thread to miss a page
    // registers it here; later threads wait on the condvar and take the cached page instead.
    std::condition_variable             pageInflightCv;
    std::unordered_set<int>             pageInflight;
    // Budget cap applied AFTER the app's post-load ZH_MBP_ClearPageCache (0 = none). The
    // big load-time budget (1536 MB) makes the load fast; once the load has settled the only
    // readers are small on-demand decodes (object probes, previews), which refilled the cache
    // to ~400 MB of resident pages that nothing needed. Post-load the cache is capped small.
    std::atomic<size_t>                 pageCacheBudgetCap{ 0 };

    // String table - populated eagerly by ParseCacheHeader. The indices array
    // is parallel to the string blob: indices[i] is the byte offset of
    // string[i] inside the blob (or -1 = null). The blob is plaintext UTF-8
    // (verified empirically - no XOR / RC4 on MccHaloReach builds despite
    // earlier comments referencing an "ILikeSafeStrings" key; that key
    // applies to other Reclaimer cache types). Resolution skips the
    // StringNamespaceTable bitfield translation for now and uses the raw
    // 32-bit StringId as the index - matches Reclaimer's
    // StringIndexBase.GetStringIndex with an identity translator. Good
    // enough to match the small set of usage names we care about
    // ("base_map", "diffuse_map", "diffuse", etc.).
    bool                          stringTableParsed = false;
    int32_t                       stringCount = 0;
    std::vector<int32_t>          stringIndices;   // size == stringCount, -1 = null
    std::vector<uint8_t>          stringBlob;

    // #175 PERF: memoized diffuse-bitmap resolution keyed by shaderTagId. ResolveDiffuse-
    // BitmapTagId walks the full rmsh->rmt2->Usages[]->ShaderMaps[] chain + ResolveStringId
    // per usage - and it is called ONCE PER MESH (2500+ on Forge World), even though a shader
    // is shared by many meshes. Memoizing collapses those to one walk per unique shader (the
    // dominant per-mesh load cost after textures). Freed with the handle -> no cross-map
    // staleness. Own mutex: mesh-tag parsing runs concurrently across cores (the viewer's parallel loader).
    std::mutex                            diffuseResolveMutex;
    std::unordered_map<int32_t, uint32_t> diffuseResolveCache;

    // StringId namespace translator (Reclaimer's StringIdTranslator). For U13:
    // indexBits=19, namespaceBits=8, lengthBits=5. The namespace table lives
    // in the cache header at StringNamespaceTablePointer (header offset 68 for
    // U8+) - N int32s, one per namespace. Each entry's low indexBits give the
    // count of strings in that namespace; entries are accumulated to compute
    // the start index of each namespace. Namespace 0 is special: its Min =
    // nsArray[0] & mask; its Start = sum of all (placed last).
    //
    // Resolution: index = stringId & ((1 << indexBits) - 1)
    //             nsId  = (stringId >> indexBits) & ((1 << namespaceBits) - 1)
    //             ns    = namespaces[nsId] (fall back to nsId-1 ... nsId-K if missing)
    //             out   = (index < ns.Min) ? index : (index - ns.Min + ns.Start)
    bool                          stringIdTranslatorReady = false;
    int                           sidIndexBits = 17;     // U8 default; U13 overrides to 19
    int                           sidNamespaceBits = 8;
    struct SidNamespace { int id; int min; int start; };
    std::vector<SidNamespace>     sidNamespaces;

    // Shared-cache support. Each layout-table SharedCaches[] entry is the
    // bare filename (e.g. "shared.map") that's expected to live in the same
    // directory as the primary map. We open them lazily on first cross-cache
    // read so a fully-self-contained map (no shared refs) pays nothing.
    //
    // sharedCacheNames is populated eagerly from ParseLayoutTable() and is
    // immutable thereafter. sharedCaches[i] is the lazily-opened CacheHandle
    // for sharedCacheNames[i] (or nullptr if open failed/not yet attempted).
    // The sharedTried[i] flag distinguishes "never attempted" from "tried and
    // failed" so we don't retry on every read.
    //
    // Open / lookup is serialised via the parent handle's parseMutex.
    // sharedCaches own their own CacheHandle::parseMutex for their own
    // gestalt/layout parses (which we don't need - we only use the header
    // + dataTableAddress + the mmap base).
    std::vector<std::string>      sharedCacheNames;     // bare names from layout table
    std::vector<CacheHandle*>     sharedCaches;         // lazily opened, may be nullptr
    std::vector<bool>             sharedTried;          // open already attempted?
    bool                          isSharedChild = false; // true = opened as a shared cache, skip recursive shared discovery

    // Bitmap-specific cache (owned by MapBitmapParser).
    void* bitmapCache = nullptr;
    void (*bitmapCacheDeleter)(void*) = nullptr;

    // Model-specific cache (owned by MapModelParser).
    void* modelCache = nullptr;
    void (*modelCacheDeleter)(void*) = nullptr;

    std::mutex   parseMutex;
    // Split lock for shared.map page routing (not `parseMutex`, the lock
    // used for BitmapSubCache::cache + ResourceEntry::fixups). With 8+
    // worker threads, parseMutex contention would serialize page-routing
    // during bitmap decode -> ~500-1000ms wallclock penalty on cold load.
    // Splitting is safe: no code path holds `parseMutex` while entering
    // ReadResourceData's shared-route branch (the lock is explicitly
    // released between phases - see MapBitmapParser.cpp:1077-1079).
    // Canonical lock-acquisition order (if ever needed):
    //   parseMutex -> pageRouteMutex -> pageCacheMutex
    std::mutex   pageRouteMutex;
};

// ----- Diagnostic logging ------------------------------------------------------
//
// Writes a timestamped line to HaloMapStudio_native.log next to the host exe.
// Implementation is in MapCacheCommon.cpp; the function is shared by every
// parser TU so failures can be traced across the cache+bitmap+model pipeline.
void NativeDiag(const char* fmt, ...);

// ----- Open / close handles ----------------------------------------------------
//
// These are NOT exported from the DLL - the public exports live in the
// per-parser headers. They're implemented in MapCacheCommon.cpp and called by
// the public ZH_MBP_OpenCache / ZH_MBP_CloseCache wrappers.
uint64_t      AcquireCacheHandle(const wchar_t* path);
void          ReleaseCacheHandle(uint64_t handle);
CacheHandle*  LookupHandle(uint64_t handle);

// Hot-reload helper: drain every cache handle (called from
// ZH_MMP_PrepareUnload). After this returns the table is empty and every
// previously-issued handle is dangling - callers must not reuse them.
void          DrainAllCacheHandles();

// ----- Parse helpers -----------------------------------------------------------
//
// All return false on any decode failure; on success the corresponding state
// fields on CacheHandle are populated. Idempotent: calling twice is a no-op.

// Eagerly parsed in AcquireCacheHandle. Parses the cache header, tag classes,
// tag entries, and tag names.
bool ParseCacheHeader(CacheHandle* cache);

// Light-weight parse for shared resource-only caches (e.g. shared.map). These
// have a 'daeh' magic + a build string at offset 160 but NO tag index, NO
// virtual base, and NO populated section table. Verifies magic + build,
// detects the build's CacheType, and reads dataTableAddress directly from the
// build-specific fixed offset (mirrors Reclaimer's ResourceIdentifier.ReadData
// approach, which never opens shared caches as full CacheFile objects).
bool ParseSharedCacheHeader(CacheHandle* cache);

// Parses the cache_file_resource_gestalt (zone tag) so resourceEntries[] is
// populated. Each entry's ResourceFixups[] block is recorded but NOT loaded - 
// EnsureResourceFixups loads on demand.
bool ParseGestalt(CacheHandle* cache);

// Loads ResourceEntries[entryIndex].fixups[] from the gestalt (lazy). Must hold
// cache->parseMutex when calling.
bool EnsureResourceFixups(CacheHandle* cache, size_t entryIndex);

// Parses the cache_file_resource_layout_table (play tag) so pages / segments /
// dataTableAddress are populated.
bool ParseLayoutTable(CacheHandle* cache);

// ----- Address translators -----------------------------------------------------

int64_t TagAddrToFileOff(const CacheHandle* c, int64_t tagAddress);
int64_t HdrAddrToFileOff(const CacheHandle* c, int64_t hdrAddress);
int64_t TagMetaFileOff  (CacheHandle* cache, uint32_t metaPointerRaw);

// Walks the tag table and returns the index of the first tag whose ClassCode
// matches the 4-char code, or -1 if not found.
int FindGlobalTag(CacheHandle* cache, const char* classCode4);

// Resolve a 32-bit StringId to a NUL-terminated string in cache->stringBlob.
// Returns nullptr if the id is OOB, the blob entry is null, or the string
// table wasn't parsed. The lifetime of the returned pointer is the cache
// handle's lifetime.
//
// Note: this uses the raw stringId as the index (no StringNamespaceTable
// bitfield translation). Reclaimer's full implementation splits the id into
// (namespace, index) using build-specific bit widths, but the ids we look
// up here come from rmt!.Usages[] which sit in namespace 0 on every Reach
// build tested - the identity mapping matches.
const char* ResolveStringId(CacheHandle* cache, int32_t stringId);

// ----- Resource data reader ----------------------------------------------------
//
// Decompresses (or memcpys) the segment data referenced by the given resource
// id. Returns a malloc'd buffer of *outSize bytes; caller must free() it.
// Pages whose CacheIndex >= 0 reference shared cache files - currently
// unsupported (returns nullptr).
uint8_t* ReadResourceData(CacheHandle* cache, int resourceIdValue,
                          size_t maxLength, size_t* outSize);
// which: 0 auto, 1 primary page only, 2 secondary page only (see .cpp)
uint8_t* ReadResourceDataPage(CacheHandle* cache, int resourceIdValue,
                              size_t maxLength, size_t* outSize, int which);

// ----- Tag-block reader --------------------------------------------------------
//
// BlockCollection<T> headers in MCC HaloReach metadata are 8 bytes: int32 count
// followed by a 32-bit Pointer (expanded via the cache's PointerExpander).
struct TagBlockRef { int32_t count; uint32_t pointer; };
TagBlockRef ReadTagBlock(const uint8_t* p);

// ----- Little-endian readers (tiny inlined helpers) -----------------------------
inline int16_t  R16  (const uint8_t* p) { int16_t  v; memcpy(&v, p, 2); return v; }
inline uint16_t RU16 (const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }
inline int32_t  R32  (const uint8_t* p) { int32_t  v; memcpy(&v, p, 4); return v; }
inline uint32_t RU32 (const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline int64_t  R64  (const uint8_t* p) { int64_t  v; memcpy(&v, p, 8); return v; }
inline uint64_t RU64 (const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

} // namespace zh_mcc
