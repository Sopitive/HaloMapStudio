// SkyWalker.cpp
// =============================================================================
// Native walker for the scnr -> Skies[] -> scenery -> model -> render_model
// chain in Halo MCC HaloReach .map files. Replaces the previous
// MeshAssetLoader.RequestSkyWalk no-op so the viewer can render the sky dome
// instead of leaving the sky cluster's flat cloud-panel BSP geometry visible.
//
// Layout reference (Reclaimer source, cross-checked against the disk cache):
//
//   scenario.cs / scenario.config.cs (HaloReach):
//     - Skies @ scnr+132  (MccHaloReach default)
//     - Skies @ scnr+136  (MccHaloReachU13)
//     - SkyReferenceBlock fixed size = 48 bytes
//     - SkyReference (TagReference) @ +0 inside the block
//     - Reclaimer treats Reach skies as `scenery` tags, not `sky `.
//
//   ObjectTagBase.cs (HaloReach):
//     - Model (TagReference) @ scenery+100  (HaloReachRetail+, MCC)
//     - This points at an `hlmt` (model) tag.
//
//   model.cs (HaloReach):
//     - RenderModel (TagReference) @ hlmt+0
//     - The render_model id we surface here feeds straight into
//       ZH_MMP_OpenModel for geometry decode.
//
// Defensive contract:
//   * Every cross-tag dereference is wrapped in __try / __except so a busted
//     chain (e.g. sky tag with a null model ref) returns 0xFFFFFFFFu rather
//     than tearing down the worker thread.
//   * Index validation against cache->tags.size() guards every TagId before
//     it's used to look up class codes / metaPointerRaw.
//   * Class-code checks confirm the tag at each link is what we expect; any
//     mismatch reports the failure via NativeDiag and returns 0xFFFFFFFFu.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

namespace {

// --- TagReference helpers (mirrors MapBspParser.cpp's ReadTagRefId) ----------
//
// Reclaimer Gen3+ TagReference layout: ClassId @ +0, padding @ +4..11,
// TagId @ +12. Returns the masked low-16 tag-table index, or -1 for the
// 0xFFFFFFFF null sentinel.
int32_t SkyReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// scnr.Skies[] block offset. Reclaimer sets:
//   - 132 for MccHaloReach (release through U10)
//   - 136 for MccHaloReachU13
int PickScnrSkiesOffset(CacheType ct) {
    switch (ct) {
        case CacheType::MccHaloReachU13:
            return 136;
        case CacheType::MccHaloReach:
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
        case CacheType::MccHaloReachU10:
        default:
            return 132;
    }
}

// SkyReferenceBlock is 48 bytes; SkyReference TagReference at +0.
constexpr int SKY_BLOCK_SIZE     = 48;
constexpr int SKY_REF_OFFSET     = 0;

// scenery (ObjectTagBase) Model TagReference offset. HaloReachRetail+ / Mcc
// builds put it at +100; older alpha/beta would be +80. We only target the
// Mcc builds, matching the rest of the parser surface.
constexpr int SCENERY_MODEL_REF_OFFSET = 100;

// model (hlmt) RenderModel TagReference is at +0.
constexpr int HLMT_RENDER_MODEL_REF_OFFSET = 0;

constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_SCEN = "scen";   // scenery tag class
constexpr const char* TC_HLMT = "hlmt";   // model tag class
constexpr const char* TC_MODE = "mode";   // render_model tag class

// -----------------------------------------------------------------------------
// Inner walkers - every step is bounds- and class-checked. Caller wraps in SEH.
// -----------------------------------------------------------------------------

// Resolve scnr meta + Skies[] tagblock pointer/count. Returns false on any
// failure (wrong class, OOB metaOff, OOB block).
bool LocateSkiesBlock(CacheHandle* cache, uint32_t scnrTagId,
                      const uint8_t** outBlockBase, int32_t* outCount)
{
    *outBlockBase = nullptr;
    *outCount = 0;

    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scnrTagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    int skiesOff = PickScnrSkiesOffset(cache->cacheType);
    if ((size_t)metaOff + (size_t)skiesOff + 8 > cache->size) return false;

    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef blk = ReadTagBlock(meta + skiesOff);
    if (blk.count <= 0 || blk.count > 0x10000) return false;

    int64_t skiesArrayOff = TagMetaFileOff(cache, blk.pointer);
    if (skiesArrayOff < 0 ||
        (size_t)skiesArrayOff + (size_t)blk.count * SKY_BLOCK_SIZE > cache->size)
        return false;

    *outBlockBase = cache->base + skiesArrayOff;
    *outCount = blk.count;
    return true;
}

// Read the i-th Skies[] entry's TagReference and return the scenery tag id
// (validated against cache->tags + class == 'scen'). -1 on failure.
int32_t ReadSkySceneryTagId(CacheHandle* cache, const uint8_t* blockBase, int32_t i)
{
    const uint8_t* entry = blockBase + (size_t)i * SKY_BLOCK_SIZE;
    int32_t id = SkyReadTagRefId(entry + SKY_REF_OFFSET);
    if (id < 0) return -1;
    if ((uint32_t)id >= cache->tags.size()) return -1;
    // Reach scnr.Skies[] points at scenery tags (Reclaimer reads
    // TagReference -> ReadMetadata<scenery>). Validate the class so a
    // stale/garbage id can't fool us into chasing arbitrary metadata.
    if (memcmp(cache->tags[id].classCode, TC_SCEN, 4) != 0) return -1;
    return id;
}

// scenery -> Model (hlmt) tag id. Returns -1 on failure.
int32_t ReadSceneryModelTagId(CacheHandle* cache, uint32_t scenTagId)
{
    if (scenTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[scenTagId];
    if (memcmp(te.classCode, TC_SCEN, 4) != 0) return -1;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return -1;
    if ((size_t)metaOff + (size_t)SCENERY_MODEL_REF_OFFSET + 16 > cache->size) return -1;

    int32_t hlmtId = SkyReadTagRefId(cache->base + metaOff + SCENERY_MODEL_REF_OFFSET);
    if (hlmtId < 0) return -1;
    if ((uint32_t)hlmtId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[hlmtId].classCode, TC_HLMT, 4) != 0) return -1;
    return hlmtId;
}

// hlmt (model) -> RenderModel (mode) tag id. Returns -1 on failure.
int32_t ReadModelRenderModelTagId(CacheHandle* cache, uint32_t hlmtTagId)
{
    if (hlmtTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[hlmtTagId];
    if (memcmp(te.classCode, TC_HLMT, 4) != 0) return -1;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return -1;
    if ((size_t)metaOff + (size_t)HLMT_RENDER_MODEL_REF_OFFSET + 16 > cache->size) return -1;

    int32_t modeId = SkyReadTagRefId(cache->base + metaOff + HLMT_RENDER_MODEL_REF_OFFSET);
    if (modeId < 0) return -1;
    if ((uint32_t)modeId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[modeId].classCode, TC_MODE, 4) != 0) return -1;
    return modeId;
}

uint32_t SkyCountInner(CacheHandle* cache, uint32_t scnrTagId)
{
    const uint8_t* base = nullptr;
    int32_t count = 0;
    if (!LocateSkiesBlock(cache, scnrTagId, &base, &count)) return 0;
    return (uint32_t)count;
}

uint32_t SkyTagIdInner(CacheHandle* cache, uint32_t scnrTagId, uint32_t skyIndex)
{
    const uint8_t* base = nullptr;
    int32_t count = 0;
    if (!LocateSkiesBlock(cache, scnrTagId, &base, &count)) return 0xFFFFFFFFu;
    if ((int32_t)skyIndex >= count) return 0xFFFFFFFFu;
    int32_t scenId = ReadSkySceneryTagId(cache, base, (int32_t)skyIndex);
    if (scenId < 0) return 0xFFFFFFFFu;
    return (uint32_t)scenId;
}

uint32_t SkyRenderModelInner(CacheHandle* cache, uint32_t scnrTagId, uint32_t skyIndex)
{
    const uint8_t* base = nullptr;
    int32_t count = 0;
    if (!LocateSkiesBlock(cache, scnrTagId, &base, &count)) return 0xFFFFFFFFu;
    if ((int32_t)skyIndex >= count) return 0xFFFFFFFFu;

    int32_t scenId = ReadSkySceneryTagId(cache, base, (int32_t)skyIndex);
    if (scenId < 0) {
        NativeDiag("SkyWalker: scnr=%u sky[%u] scenery ref unresolved",
                   scnrTagId, skyIndex);
        return 0xFFFFFFFFu;
    }

    int32_t hlmtId = ReadSceneryModelTagId(cache, (uint32_t)scenId);
    if (hlmtId < 0) {
        NativeDiag("SkyWalker: scnr=%u sky[%u] scen=%d hlmt ref unresolved",
                   scnrTagId, skyIndex, scenId);
        return 0xFFFFFFFFu;
    }

    int32_t modeId = ReadModelRenderModelTagId(cache, (uint32_t)hlmtId);
    if (modeId < 0) {
        NativeDiag("SkyWalker: scnr=%u sky[%u] scen=%d hlmt=%d render_model ref unresolved",
                   scnrTagId, skyIndex, scenId, hlmtId);
        return 0xFFFFFFFFu;
    }
    return (uint32_t)modeId;
}

// SEH wrappers - every cross-pointer deref happens through these so a busted
// chain returns the failure sentinel instead of unwinding into the caller.

uint32_t SehSkyCount(CacheHandle* cache, uint32_t scnrTagId)
{
    __try { return SkyCountInner(cache, scnrTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}

uint32_t SehSkyTagId(CacheHandle* cache, uint32_t scnrTagId, uint32_t skyIndex)
{
    __try { return SkyTagIdInner(cache, scnrTagId, skyIndex); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

uint32_t SehSkyRenderModel(CacheHandle* cache, uint32_t scnrTagId, uint32_t skyIndex)
{
    __try { return SkyRenderModelInner(cache, scnrTagId, skyIndex); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SKY_GetSkyCount(
    uint64_t cacheHandle, uint32_t scnrTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    return SehSkyCount(cache, scnrTagId);
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SKY_GetSkyTagId(
    uint64_t cacheHandle, uint32_t scnrTagId, uint32_t skyIndex)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehSkyTagId(cache, scnrTagId, skyIndex);
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SKY_GetRenderModelTagId(
    uint64_t cacheHandle, uint32_t scnrTagId, uint32_t skyIndex)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehSkyRenderModel(cache, scnrTagId, skyIndex);
}
