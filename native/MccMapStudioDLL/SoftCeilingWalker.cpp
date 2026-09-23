// SoftCeilingWalker.cpp
// =============================================================================
// Reads Halo: Reach SOFT CEILINGS -- the invisible planes that bound the playable
// volume (the "map floor" soft-kill under the map, the acceleration ceilings that
// push you back down, slip surfaces) -- so HaloMapStudio can draw them like the
// trigger volumes.
//
// The geometry does NOT live in the scenario: the scnr only carries a small
// metadata block (name / type / flags). The triangles live in the structure
// DESIGN tag (sddt) of each BSP, referenced from scnr.StructureDesigns[].
//
// Layout (RE'd from reach_tag_test.exe field tables, imagebase
// 0x140000000; block sizes are the sizeof() the binary states next to each
// field list):
//
//   scnr + 0x5C   tagblock(structure design[0x20])   (0x58 pre-U13)
//     +0x00  tagRef(16)  -> sddt          (datum @ +0xC)
//   scnr + 0x25C  tagblock(scenario_soft_ceilings_block[0x0C])   -- metadata only
//     +0x00  word flags   { 1 ignore bipeds, 2 ignore vehicles, 4 ignore camera,
//                           8 ignore huge vehicles }
//     +0x02  word runtime flags
//     +0x04  string_id name
//     +0x08  enum16 type
//
//   sddt (structure_design, sizeof 0x160)
//     +0x30  struct global_structure_physics_design_struct (0x40)   [field table 0x141EE1020]
//       +0x00  long   importer version
//       +0x04  block  soft ceiling mopp code
//       +0x10  block  soft ceilings                         => sddt + 0x40
//       +0x1C  block  water mopp code
//       +0x28  block  water groups
//       +0x34  block  water instances                       => sddt + 0x64 (PlanarFogWalker agrees)
//
//   structure_soft_ceiling_block  (sizeof 0x14, max 128)   [struct 0x141EE0390]
//     +0x00  string_id name
//     +0x04  enum16    type  soft_ceiling_type_enum { 0 acceleration, 1 soft kill, 2 slip surface }
//     +0x06  pad 2
//     +0x08  block     soft ceiling triangles
//
//   structure_soft_ceiling_triangle_block (sizeof 0x44, max 0x7FFF) [struct 0x141EE0190]
//     +0x00  real_plane3d  plane (i, j, k, d)
//     +0x10  real_point3d  bounding sphere center
//     +0x1C  real          bounding sphere radius
//     +0x20  real_point3d  vertex0
//     +0x2C  real_point3d  vertex1
//     +0x38  real_point3d  vertex2
//
// Vertices are WORLD SPACE (the design tag is baked per BSP, like the fog
// planes next to it).
//
// Public exports (two passes, no buffer ownership to hand back):
//   uint32_t ZH_SDDT_EnumerateSoftCeilings(cache, scnrTagId, ZH_SoftCeiling* out, max)
//       -> number of ceilings across every structure design (out=null -> count query).
//          Each record carries the [triStart, triStart+triCount) window of the
//          triangle stream returned by the second export.
//   uint32_t ZH_SDDT_EnumerateSoftCeilingTriangles(cache, scnrTagId, float* outXyz, maxTris)
//       -> total triangle count; writes 9 floats (v0 v1 v2) per triangle, in the
//          SAME order the ceilings were enumerated (out=null -> count query).
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

namespace {

constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_SDDT = "sddt";

constexpr int SCNR_SOFT_CEILINGS_META_OFFSET = 0x25C;
constexpr int SCNR_SOFT_CEILING_META_STRIDE  = 0x0C;
constexpr int SCM_FLAGS_OFFSET = 0x00;
constexpr int SCM_NAME_OFFSET  = 0x04;
constexpr int SCM_TYPE_OFFSET  = 0x08;

constexpr int SD_ENTRY_SIZE = 0x20;

constexpr int SDDT_SOFT_CEILINGS_OFFSET = 0x40;
constexpr int SC_STRIDE       = 0x14;
constexpr int SC_NAME_OFFSET  = 0x00;
constexpr int SC_TYPE_OFFSET  = 0x04;
constexpr int SC_TRIS_OFFSET  = 0x08;

constexpr int TRI_STRIDE      = 0x44;
constexpr int TRI_V0_OFFSET   = 0x20;

constexpr int SC_COUNT_SANITY  = 128;     // k_maximum_structure_soft_ceilings_count
constexpr int TRI_COUNT_SANITY = 0x7FFF;  // k_maximum_structure_soft_ceiling_triangles
constexpr int SD_COUNT_SANITY  = 64;
constexpr int META_COUNT_SANITY = 1024;

inline float RF32(const uint8_t* p) { float v; memcpy(&v, p, 4); return v; }

#pragma pack(push, 1)
struct ZH_SoftCeiling {
    uint32_t type;        // 0 acceleration, 1 soft kill, 2 slip surface
    uint32_t flags;       // scnr metadata flags (0 when the scnr has no entry of that name)
    uint32_t triStart;    // first triangle in the triangle stream
    uint32_t triCount;
    uint32_t sddtTagId;   // owning structure_design tag
    uint32_t ceilingIndex;// index within that sddt's soft ceilings block
    char     name[40];
};
#pragma pack(pop)

static_assert(sizeof(ZH_SoftCeiling) == 6 * 4 + 40, "ZH_SoftCeiling layout drift");

void CopyResolvedString(CacheHandle* cache, uint32_t stringId, char* outBuf, size_t outBufLen)
{
    if (outBuf == nullptr || outBufLen == 0) return;
    outBuf[0] = '\0';
    if (stringId == 0u || stringId == 0xFFFFFFFFu) return;
    const char* resolved = ResolveStringId(cache, (int32_t)stringId);
    if (resolved == nullptr) return;
    size_t n = strlen(resolved);
    if (n >= outBufLen) n = outBufLen - 1;
    memcpy(outBuf, resolved, n);
    outBuf[n] = '\0';
}

int StructureDesignsOffset(CacheType ct)
{
    switch (ct) {
        case CacheType::MccHaloReachU13: return 0x5C;
        default:                          return 0x58;
    }
}

// Flags of the scnr soft-ceilings metadata entry with this name (0 when absent).
uint32_t LookupScnrFlags(CacheHandle* cache, const uint8_t* scnrMeta, uint32_t nameId)
{
    if ((size_t)(scnrMeta - cache->base) + SCNR_SOFT_CEILINGS_META_OFFSET + 12 > cache->size) return 0;
    TagBlockRef blk = ReadTagBlock(scnrMeta + SCNR_SOFT_CEILINGS_META_OFFSET);
    if (blk.count <= 0 || blk.count > META_COUNT_SANITY) return 0;
    int64_t off = TagMetaFileOff(cache, blk.pointer);
    if (off < 0) return 0;
    if ((size_t)off + (size_t)blk.count * SCNR_SOFT_CEILING_META_STRIDE > cache->size) return 0;
    const uint8_t* arr = cache->base + off;
    for (int i = 0; i < blk.count; ++i)
    {
        const uint8_t* e = arr + (size_t)i * SCNR_SOFT_CEILING_META_STRIDE;
        if (RU32(e + SCM_NAME_OFFSET) == nameId)
            return (uint32_t)RU16(e + SCM_FLAGS_OFFSET);
    }
    return 0;
}

// Shared walk: visits every soft ceiling of every structure design of the scenario.
// `outCeil` / `outXyz` may be null (count queries).
struct WalkResult { uint32_t ceilings; uint32_t triangles; };

WalkResult Walk(CacheHandle* cache, uint32_t scnrTagId,
                ZH_SoftCeiling* outCeil, uint32_t maxCeil,
                float* outXyz, uint32_t maxTris)
{
    WalkResult r{0, 0};
    if (cache == nullptr || cache->base == nullptr) return r;
    if (scnrTagId >= cache->tags.size()) return r;
    const TagEntry& scnrEntry = cache->tags[scnrTagId];
    if (memcmp(scnrEntry.classCode, TC_SCNR, 4) != 0) return r;

    int64_t scnrMetaOff = TagMetaFileOff(cache, scnrEntry.metaPointerRaw);
    if (scnrMetaOff < 0) return r;
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    int sdOff = StructureDesignsOffset(cache->cacheType);
    if ((size_t)scnrMetaOff + (size_t)sdOff + 12 > cache->size) return r;
    TagBlockRef sdBlk = ReadTagBlock(scnrMeta + sdOff);
    if (sdBlk.count <= 0 || sdBlk.count > SD_COUNT_SANITY)
    {
        int alt = (sdOff == 0x5C) ? 0x58 : 0x5C;
        if ((size_t)scnrMetaOff + (size_t)alt + 12 <= cache->size)
        {
            TagBlockRef altBlk = ReadTagBlock(scnrMeta + alt);
            if (altBlk.count > 0 && altBlk.count <= SD_COUNT_SANITY) sdBlk = altBlk;
        }
    }
    if (sdBlk.count <= 0 || sdBlk.count > SD_COUNT_SANITY) return r;
    int64_t sdArrOff = TagMetaFileOff(cache, sdBlk.pointer);
    if (sdArrOff < 0 || (size_t)sdArrOff + (size_t)sdBlk.count * SD_ENTRY_SIZE > cache->size) return r;

    for (int d = 0; d < sdBlk.count; ++d)
    {
        const uint8_t* sdEntry = cache->base + sdArrOff + (size_t)d * SD_ENTRY_SIZE;
        uint32_t rawId = RU32(sdEntry + 0x0C);
        if (rawId == 0xFFFFFFFFu) continue;
        uint32_t sddtId = rawId & 0xFFFFu;
        if (sddtId >= cache->tags.size()) continue;
        const TagEntry& te = cache->tags[sddtId];
        if (memcmp(te.classCode, TC_SDDT, 4) != 0) continue;
        int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
        if (metaOff < 0) continue;
        if ((size_t)metaOff + SDDT_SOFT_CEILINGS_OFFSET + 12 > cache->size) continue;
        const uint8_t* meta = cache->base + metaOff;

        TagBlockRef scBlk = ReadTagBlock(meta + SDDT_SOFT_CEILINGS_OFFSET);
        if (scBlk.count <= 0 || scBlk.count > SC_COUNT_SANITY) continue;
        int64_t scArrOff = TagMetaFileOff(cache, scBlk.pointer);
        if (scArrOff < 0 || (size_t)scArrOff + (size_t)scBlk.count * SC_STRIDE > cache->size) continue;
        const uint8_t* scArr = cache->base + scArrOff;

        for (int c = 0; c < scBlk.count; ++c)
        {
            const uint8_t* sc = scArr + (size_t)c * SC_STRIDE;
            TagBlockRef triBlk = ReadTagBlock(sc + SC_TRIS_OFFSET);
            int triCount = triBlk.count;
            const uint8_t* triArr = nullptr;
            if (triCount < 0 || triCount > TRI_COUNT_SANITY) triCount = 0;
            if (triCount > 0)
            {
                int64_t triArrOff = TagMetaFileOff(cache, triBlk.pointer);
                if (triArrOff < 0 || (size_t)triArrOff + (size_t)triCount * TRI_STRIDE > cache->size)
                    triCount = 0;
                else
                    triArr = cache->base + triArrOff;
            }

            uint32_t nameId = RU32(sc + SC_NAME_OFFSET);
            if (outCeil != nullptr && r.ceilings < maxCeil)
            {
                ZH_SoftCeiling& dst = outCeil[r.ceilings];
                memset(&dst, 0, sizeof(dst));
                dst.type         = (uint32_t)RU16(sc + SC_TYPE_OFFSET);
                dst.flags        = LookupScnrFlags(cache, scnrMeta, nameId);
                dst.triStart     = r.triangles;
                dst.triCount     = (uint32_t)triCount;
                dst.sddtTagId    = sddtId;
                dst.ceilingIndex = (uint32_t)c;
                CopyResolvedString(cache, nameId, dst.name, sizeof(dst.name));
            }
            ++r.ceilings;

            for (int t = 0; t < triCount; ++t)
            {
                if (outXyz != nullptr && r.triangles < maxTris)
                {
                    const uint8_t* tri = triArr + (size_t)t * TRI_STRIDE + TRI_V0_OFFSET;
                    float* o = outXyz + (size_t)r.triangles * 9;
                    for (int k = 0; k < 9; ++k) o[k] = RF32(tri + k * 4);
                }
                ++r.triangles;
            }
        }
    }
    return r;
}

} // namespace

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SDDT_EnumerateSoftCeilings(
    uint64_t cacheHandle, uint32_t scnrTagId, ZH_SoftCeiling* out, uint32_t maxCount)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (cache == nullptr) return 0;
    return Walk(cache, scnrTagId, out, maxCount, nullptr, 0).ceilings;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_SDDT_EnumerateSoftCeilingTriangles(
    uint64_t cacheHandle, uint32_t scnrTagId, float* outXyz, uint32_t maxTris)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (cache == nullptr) return 0;
    return Walk(cache, scnrTagId, nullptr, 0, outXyz, maxTris).triangles;
}
