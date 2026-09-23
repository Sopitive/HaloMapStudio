// CollisionModelWalker.cpp
// =============================================================================
// Native walker for the collision_model ('coll') tag in Halo MCC HaloReach
// .map files. Many forge objects (the "nut blocker" / collision-blocker
// gameplay items) ship a render_model ('mode') that is effectively invisible
// in-game but carry a meaningful collision hull. HaloMapStudio normally renders
// only the render_model, so the user can't see where the collision sits when
// forging. This walker resolves the object's 'coll' tag and decodes its
// surface geometry into a simple triangle soup the viewer wraps in a mesh
// and renders as a translucent / wireframe overlay.
//
// Resolution chain (mirrors ZH_TAG_ResolveModeTagId in MapBitmapParser.cpp):
//   object tag -> hlmt tag-ref -> hlmt -> coll tag-ref -> coll
//
// Geometry layout (authoritative: Assembly ReachMCC/coll.xml, cross-checked
// against the disk cache). collision_model, baseSize 0x50:
//   Regions[]            @ 0x20  elemSize 0x10
//     Name (stringid)    @ 0x00
//     Permutations[]     @ 0x04  elemSize 0x28
//       BSPs[]           @ 0x04  elemSize 0x70
//         Surfaces[]     @ 0x4C  elemSize 0x0C
//           PlaneIndex u16 @ 0x00, FirstEdge u16 @ 0x02, Material i16 @ 0x04,
//           ... Flags u8 @ 0x0A (bit1 = Invisible)
//         Edges[]        @ 0x58  elemSize 0x0C
//           StartVertex i16 @ 0x00, EndVertex i16 @ 0x02,
//           ForwardEdge i16 @ 0x04, ReverseEdge i16 @ 0x06,
//           LeftSurface i16 @ 0x08, RightSurface i16 @ 0x0A
//         Vertices[]     @ 0x64  elemSize 0x10
//           Point (3xfloat) @ 0x00, FirstEdge i16 @ 0x0C, Sink i16 @ 0x0E
//
// Surface triangulation: collision surfaces are convex polygons stored as a
// half-edge loop. Starting at Surface.FirstEdge, walk the loop: at each edge,
// the vertex on the surface side and the next edge differ depending on whether
// this surface is the edge's LeftSurface or RightSurface (standard Halo
// collision-BSP winding). We collect the loop's ordered vertices and emit a
// triangle fan (v0, v[i], v[i+1]). Degenerate / runaway loops are bounded.
//
// Defensive contract (matches SkyWalker.cpp / ScenarioTriggerVolumeWalker.cpp):
//   * Every cross-tag deref is SEH-fenced so a busted chain returns cleanly.
//   * Block counts are sanity-capped before any pointer math.
//   * All buffer offsets are validated against cache->size.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

using namespace zh_mcc;

namespace {

// ---- coll schema offsets (ReachMCC/coll.xml) --------------------------------
constexpr int OFF_COLL_REGIONS        = 0x20;   // tagblock, elem 0x10
constexpr int REGION_BLOCK_SIZE       = 0x10;
constexpr int OFF_REGION_PERMS        = 0x04;   // tagblock, elem 0x28
constexpr int PERM_BLOCK_SIZE         = 0x28;
constexpr int OFF_PERM_BSPS           = 0x04;   // tagblock, elem 0x70
constexpr int BSP_BLOCK_SIZE          = 0x70;

constexpr int OFF_BSP_SURFACES        = 0x4C;   // tagblock, elem 0x0C
constexpr int SURFACE_BLOCK_SIZE      = 0x0C;
constexpr int OFF_BSP_EDGES           = 0x58;   // tagblock, elem 0x0C
constexpr int EDGE_BLOCK_SIZE         = 0x0C;
constexpr int OFF_BSP_VERTICES        = 0x64;   // tagblock, elem 0x10
constexpr int VERTEX_BLOCK_SIZE       = 0x10;

// Surface flags (Surface+0x0A).
constexpr uint8_t SURF_FLAG_INVISIBLE = 0x02;   // bit 1

// Sanity caps - collision models are small relative to render models.
constexpr int32_t MAX_REGIONS   = 256;
constexpr int32_t MAX_PERMS     = 256;
constexpr int32_t MAX_BSPS      = 256;
constexpr int32_t MAX_SURFACES  = 200000;
constexpr int32_t MAX_EDGES     = 400000;
constexpr int32_t MAX_VERTS     = 400000;
constexpr int      MAX_LOOP     = 256;          // bound a single surface edge loop

struct CollGeom {
    std::vector<float>    verts;   // x,y,z triples (world units, model space)
    std::vector<uint32_t> indices; // triangle list
};

// Translate a (count + raw pointer) tagblock to a validated base pointer.
// Returns nullptr if OOB / bad. On success *outBase points into cache->base
// and the block of `count * elemSize` bytes is in range.
const uint8_t* ResolveBlock(CacheHandle* cache, const TagBlockRef& blk,
                            int elemSize, int32_t maxCount)
{
    if (blk.count <= 0 || blk.count > maxCount) return nullptr;
    int64_t off = TagMetaFileOff(cache, blk.pointer);
    if (off < 0) return nullptr;
    if ((size_t)off + (size_t)blk.count * (size_t)elemSize > cache->size) return nullptr;
    return cache->base + off;
}

// Walk one BSP block's surfaces -> edges -> vertices and append triangles.
void WalkBsp(CacheHandle* cache, const uint8_t* bsp, CollGeom& out)
{
    TagBlockRef surfBlk = ReadTagBlock(bsp + OFF_BSP_SURFACES);
    TagBlockRef edgeBlk = ReadTagBlock(bsp + OFF_BSP_EDGES);
    TagBlockRef vertBlk = ReadTagBlock(bsp + OFF_BSP_VERTICES);

    const uint8_t* surfBase = ResolveBlock(cache, surfBlk, SURFACE_BLOCK_SIZE, MAX_SURFACES);
    const uint8_t* edgeBase = ResolveBlock(cache, edgeBlk, EDGE_BLOCK_SIZE,    MAX_EDGES);
    const uint8_t* vertBase = ResolveBlock(cache, vertBlk, VERTEX_BLOCK_SIZE,  MAX_VERTS);
    if (!surfBase || !edgeBase || !vertBase) return;

    const int32_t surfCount = surfBlk.count;
    const int32_t edgeCount = edgeBlk.count;
    const int32_t vertCount = vertBlk.count;

    // Base index for this BSP's vertices in the merged output stream. We push
    // ALL vertices of this BSP up front so per-surface indices map directly.
    const uint32_t vbase = (uint32_t)(out.verts.size() / 3);
    out.verts.reserve(out.verts.size() + (size_t)vertCount * 3);
    for (int32_t v = 0; v < vertCount; ++v) {
        const uint8_t* vp = vertBase + (size_t)v * VERTEX_BLOCK_SIZE;
        float x, y, z;
        memcpy(&x, vp + 0, 4);
        memcpy(&y, vp + 4, 4);
        memcpy(&z, vp + 8, 4);
        out.verts.push_back(x);
        out.verts.push_back(y);
        out.verts.push_back(z);
    }

    // For each surface, walk its edge loop and fan-triangulate.
    int loopBuf[MAX_LOOP];
    for (int32_t s = 0; s < surfCount; ++s) {
        const uint8_t* sp = surfBase + (size_t)s * SURFACE_BLOCK_SIZE;
        uint16_t firstEdge = RU16(sp + 0x02);
        uint8_t  flags      = *(sp + 0x0A);
        if (flags & SURF_FLAG_INVISIBLE) continue;   // skip invisible collision faces

        // Walk the half-edge loop. At each edge, the surface appears as either
        // the LeftSurface or RightSurface; that decides which vertex to take
        // and which neighbour edge to follow.
        int loopCount = 0;
        int32_t e = (int32_t)firstEdge;
        if (e < 0 || e >= edgeCount) continue;
        int32_t startEdge = e;
        do {
            const uint8_t* ep = edgeBase + (size_t)e * EDGE_BLOCK_SIZE;
            int16_t startV   = R16(ep + 0x00);
            int16_t endV     = R16(ep + 0x02);
            int16_t fwdEdge  = R16(ep + 0x04);
            int16_t revEdge  = R16(ep + 0x06);
            int16_t leftSurf = R16(ep + 0x08);

            int32_t vtx;
            int32_t nextEdge;
            if ((int32_t)leftSurf == s) {
                vtx      = startV;
                nextEdge = fwdEdge;
            } else {
                vtx      = endV;
                nextEdge = revEdge;
            }

            if (vtx < 0 || vtx >= vertCount) break;
            if (loopCount < MAX_LOOP) loopBuf[loopCount++] = vtx;

            if (nextEdge < 0 || nextEdge >= edgeCount) break;
            e = nextEdge;
        } while (e != startEdge && loopCount < MAX_LOOP);

        // Fan-triangulate the polygon (loopBuf[0], loopBuf[i], loopBuf[i+1]).
        for (int i = 1; i + 1 < loopCount; ++i) {
            out.indices.push_back(vbase + (uint32_t)loopBuf[0]);
            out.indices.push_back(vbase + (uint32_t)loopBuf[i]);
            out.indices.push_back(vbase + (uint32_t)loopBuf[i + 1]);
        }
    }
}

// Walk every region -> permutation -> bsp. SEH-guarded by the caller.
void DecodeCollInner(CacheHandle* cache, uint32_t collTagId, CollGeom& out)
{
    if (collTagId >= cache->tags.size()) return;
    const TagEntry& te = cache->tags[collTagId];
    if (te.classIndex < 0) return;
    if (memcmp(te.classCode, "coll", 4) != 0) return;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return;
    if ((size_t)metaOff + 0x50 > cache->size) return;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef regionsBlk = ReadTagBlock(meta + OFF_COLL_REGIONS);
    const uint8_t* regBase = ResolveBlock(cache, regionsBlk, REGION_BLOCK_SIZE, MAX_REGIONS);
    if (!regBase) return;

    for (int32_t r = 0; r < regionsBlk.count; ++r) {
        const uint8_t* rp = regBase + (size_t)r * REGION_BLOCK_SIZE;
        TagBlockRef permsBlk = ReadTagBlock(rp + OFF_REGION_PERMS);
        const uint8_t* permBase = ResolveBlock(cache, permsBlk, PERM_BLOCK_SIZE, MAX_PERMS);
        if (!permBase) continue;

        for (int32_t p = 0; p < permsBlk.count; ++p) {
            const uint8_t* pp = permBase + (size_t)p * PERM_BLOCK_SIZE;
            TagBlockRef bspsBlk = ReadTagBlock(pp + OFF_PERM_BSPS);
            const uint8_t* bspBase = ResolveBlock(cache, bspsBlk, BSP_BLOCK_SIZE, MAX_BSPS);
            if (!bspBase) continue;

            for (int32_t b = 0; b < bspsBlk.count; ++b) {
                const uint8_t* bp = bspBase + (size_t)b * BSP_BLOCK_SIZE;
                WalkBsp(cache, bp, out);
            }
        }
    }
}

bool SehDecodeColl(CacheHandle* cache, uint32_t collTagId, CollGeom& out)
{
    __try { DecodeCollInner(cache, collTagId, out); return true; }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

// ---- coll resolution (mirror of ZH_TAG_ResolveModeTagId for "coll") --------
// Scan the first `maxBytes` of `meta` for a tag-ref whose class fourcc matches
// groupAscii; return its in-cache tag id (0..tags.size()-1) or 0xFFFFFFFFu.
uint32_t ScanTagRefByClassColl(CacheHandle* cache, const uint8_t* meta,
                               size_t maxBytes, const char* groupAscii)
{
    if (maxBytes < 16) return 0xFFFFFFFFu;
    const uint32_t targetU32 =
        ((uint32_t)(uint8_t)groupAscii[0] << 24) |
        ((uint32_t)(uint8_t)groupAscii[1] << 16) |
        ((uint32_t)(uint8_t)groupAscii[2] <<  8) |
        ((uint32_t)(uint8_t)groupAscii[3]      );
    const size_t scanEnd = maxBytes - 16;
    for (size_t off = 0; off <= scanEnd; off += 4) {
        if (RU32(meta + off) != targetU32) continue;
        uint32_t raw = RU32(meta + off + 12);
        if (raw == 0xFFFFFFFFu) continue;
        uint32_t id = raw & 0xFFFFu;
        if (id >= cache->tags.size()) continue;
        if (memcmp(cache->tags[id].classCode, groupAscii, 4) != 0) continue;
        return id;
    }
    return 0xFFFFFFFFu;
}

uint32_t ResolveCollInner(CacheHandle* cache, uint32_t primaryTagId)
{
    if (primaryTagId >= cache->tags.size()) return 0xFFFFFFFFu;
    const TagEntry& objTe = cache->tags[primaryTagId];
    if (objTe.classIndex < 0) return 0xFFFFFFFFu;

    int64_t objMetaOff = TagMetaFileOff(cache, objTe.metaPointerRaw);
    if (objMetaOff < 0) return 0xFFFFFFFFu;

    constexpr size_t kObjScanBytes  = 0x400;
    constexpr size_t kHlmtScanBytes = 0x100;

    size_t objAvail = (size_t)cache->size - (size_t)objMetaOff;
    if (objAvail < 16) return 0xFFFFFFFFu;
    size_t objScan = objAvail < kObjScanBytes ? objAvail : kObjScanBytes;

    uint32_t hlmtId = ScanTagRefByClassColl(cache, cache->base + objMetaOff, objScan, "hlmt");
    if (hlmtId == 0xFFFFFFFFu) return 0xFFFFFFFFu;

    int64_t hlmtMetaOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
    if (hlmtMetaOff < 0) return 0xFFFFFFFFu;
    size_t hlmtAvail = (size_t)cache->size - (size_t)hlmtMetaOff;
    if (hlmtAvail < 16) return 0xFFFFFFFFu;
    size_t hlmtScan = hlmtAvail < kHlmtScanBytes ? hlmtAvail : kHlmtScanBytes;

    return ScanTagRefByClassColl(cache, cache->base + hlmtMetaOff, hlmtScan, "coll");
}

uint32_t SehResolveColl(CacheHandle* cache, uint32_t primaryTagId)
{
    __try { return ResolveCollInner(cache, primaryTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

// Resolve an object tag (bloc/scen/vehi/...) to its collision_model ('coll')
// tag id by walking object -> hlmt -> coll. Returns the coll tag id
// (0..tags.size()-1) on success, or 0xFFFFFFFFu if the object has no coll
// (most lights/sounds/effects), the chain breaks, or the handle is bad.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolveCollTagId(
    uint64_t cacheHandle, uint32_t primaryTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehResolveColl(cache, primaryTagId);
}

// Decode a collision_model's surface geometry into a triangle soup.
//
//   outVerts   -> malloc'd float[3*outVertCount] (x,y,z per vertex, model space)
//   outVertCount -> number of vertices (== outVerts length / 3)
//   outIndices -> malloc'd uint32[outIndexCount] (triangle list, 3 per tri)
//   outIndexCount -> number of indices (multiple of 3)
//
// Returns true on success (outIndexCount may be 0 for an all-invisible /
// empty coll); false on bad handle / bad tag / decode fault. The caller frees
// both buffers via ZH_COLL_FreeBuffer. On failure all out params are zeroed.
extern "C" __declspec(dllexport) bool __stdcall ZH_COLL_DecodeGeometry(
    uint64_t cacheHandle, uint32_t collTagId,
    float**  outVerts,  uint32_t* outVertCount,
    uint32_t** outIndices, uint32_t* outIndexCount)
{
    if (outVerts)      *outVerts = nullptr;
    if (outVertCount)  *outVertCount = 0;
    if (outIndices)    *outIndices = nullptr;
    if (outIndexCount) *outIndexCount = 0;
    if (!outVerts || !outVertCount || !outIndices || !outIndexCount) return false;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    CollGeom geom;
    if (!SehDecodeColl(cache, collTagId, geom)) return false;

    uint32_t vCount = (uint32_t)(geom.verts.size() / 3);
    uint32_t iCount = (uint32_t)geom.indices.size();

    if (vCount > 0) {
        float* vb = (float*)malloc((size_t)vCount * 3 * sizeof(float));
        if (!vb) return false;
        memcpy(vb, geom.verts.data(), (size_t)vCount * 3 * sizeof(float));
        *outVerts = vb;
        *outVertCount = vCount;
    }
    if (iCount > 0) {
        uint32_t* ib = (uint32_t*)malloc((size_t)iCount * sizeof(uint32_t));
        if (!ib) { free(*outVerts); *outVerts = nullptr; *outVertCount = 0; return false; }
        memcpy(ib, geom.indices.data(), (size_t)iCount * sizeof(uint32_t));
        *outIndices = ib;
        *outIndexCount = iCount;
    }

    NativeDiag("Coll[%u]: decoded verts=%u tris=%u", collTagId, vCount, iCount / 3);
    return true;
}

// Free a buffer returned by ZH_COLL_DecodeGeometry (verts OR indices). Safe on
// nullptr.
extern "C" __declspec(dllexport) void __stdcall ZH_COLL_FreeBuffer(void* buf)
{
    if (buf) free(buf);
}
