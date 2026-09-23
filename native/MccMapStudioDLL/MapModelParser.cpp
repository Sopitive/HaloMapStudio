// MapModelParser.cpp
// =============================================================================
// Native render_model (mode tag) decoder for Halo MCC HaloReach .map files.
//
// Mirrors Reclaimer.Blam.HaloReach.render_model + HaloReachCommon.GetMeshes.
// Reuses the cache infrastructure in MapCacheCommon.cpp - the user opens the
// .map once via ZH_MBP_OpenCache, and both the bitmap and model parsers
// share that handle.
//
// Schema layout for MccHaloReach (release through U13). See Reclaimer's
// render_model.cs for the per-build offsets; Reach Retail has Sections@104,
// BoundingBoxes@116, ResourcePointer@248. We default to those (the Reach
// retail / MCC-Reach offsets). Earlier (Beta) builds shift ResourcePointer
// to 224 / 236 and are not supported.
//
// Resource gestalt walk (see HaloReachCommon.GetMeshes):
//   entry = gestalt.ResourceEntries[mode.ResourcePointer.ResourceIndex]
//   At entry.FixupOffset + (entry.FixupSize - 24): vertex_buffer_count
//                                                  (skip 8 bytes)
//                                                  index_buffer_count
//   At entry.FixupOffset:           VertexBufferInfo[vb_count]   (28 bytes each)
//                                   skip 12 * vb_count
//                                   IndexBufferInfo[ib_count]    (28 bytes each)
//                                   skip 12 * ib_count
//                                   skip 4 * 12 (4 trailer structs)
//   For section vb data: entry.ResourceFixups[vbIdx].Offset & 0x0FFFFFFF
//   For section ib data: entry.ResourceFixups[2*vb_count + ibIdx].Offset & 0x0FFFFFFF
//
// Vertex format catalog (MccHaloReach U13, authoritative - see
// Reclaimer.Blam/Resources/MccHaloReachVertexBuffer.xml). The on-disk data
// matches the XML: positions are raw Float32_4 / Float32_3, NOT the legacy
// Xbox-360 UInt16_N4 packed form (decoding as packed produces jumbled
// geometry).
//
//   Format 0x00 / 0x04 (s_world_vertex / s_flat_world)
//       stride 0x24 (36)
//       +0x00 Float32_4 position (only xyz used)
//       +0x10 Float16_2 texcoords  (IEEE 754 binary16 -> binary32)
//       +0x14 Int16_N4  normal     (not decoded; renderer synthesizes)
//       +0x1C Int16_N4  tangent    (not decoded)
//
//   Format 0x01 / 0x05 (s_rigid_vertex / s_flat_rigid)
//       stride 0x24 (36)
//       +0x00 Float32_4 position
//       +0x10 UInt16_N2 texcoords  (quantize via uvMin/uvMax)
//       +0x14 Int16_N4  normal
//       +0x1C Int16_N4  tangent
//
//   Format 0x02 / 0x06 (s_skinned_vertex / s_flat_skinned)
//       stride 0x2C (44) - same first 0x24 as rigid, then:
//       +0x24 UInt8_4   blendindices
//       +0x28 UInt8_N4  blendweight
//       (bone data ignored - biped previews show bind-pose mesh)
//
//   Format 0x0F        (s_decorator_vertex)
//       stride 0x20 (32)
//       +0x00 Float32_3 position
//       +0x0C Float32_2 texcoords  (raw floats, NOT quantized)
//       +0x14 Float32_3 normal
//
//   Format 0x16+       (tessellated / exotic) - return false
//
// posMin/posMax are NOT applied to positions for any of these formats - the
// values on disk are already in world units. uvMin/uvMax still apply for
// UInt16_N2 texcoord decode (rigid/skinned).
// =============================================================================

#include "pch.h"
#include "MapModelParser.h"
#include "MapBspParser.h"      // ZH_ShaderConstants struct (shared)
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>       // SKY-3: getenv-gated stderr DIAG (fprintf)
#include <atomic>
#include <new>
#include <vector>
#include <unordered_map>
#include <mutex>

using namespace zh_mcc;

namespace {

// -----------------------------------------------------------------------------
// In-memory model handle
// -----------------------------------------------------------------------------

struct ModelSection {
    int16_t  vertexBufferIndex;
    int16_t  indexBufferIndex;
    uint16_t flags;
    uint8_t  nodeIndex;
    uint8_t  vertexFormat;
    uint8_t  indexFormat;       // engine value; 5 = TriangleStrip, 3 = TriangleList
    uint32_t vbDataLength;
    uint32_t vertexCount;
    uint32_t ibDataLength;
    uint32_t indexCount;        // post-strip-expansion count, computed lazily
    uint32_t vbResourceOffset;
    uint32_t ibResourceOffset;
    bool     isUnindexed;

    // Convenience: bounds for this section's positions/UVs (from BoundingBoxes[0],
    // shared across all sections in HaloReach since ResourceFixups are
    // section-keyed but bounds are model-wide).
    float    posMin[3];
    float    posMax[3];
    float    uvMin[2];
    float    uvMax[2];

    // Submesh range - submeshes are stored flat in ModelData::submeshes
    uint32_t submeshStart;
    uint32_t submeshCount;
    int32_t  materialIndex;

    // Per-vertex COLOR stream (vb[vertexBufferIndex+1], stride 12 = float3 in
    // [0,1]) present on vertex-gradient sky domes (condemned/prisoner). 0 length = no color stream.
    uint32_t colorResourceOffset;
    uint32_t colorDataLength;
    uint32_t colorStride;
};

struct ModelSubmesh {
    uint32_t sectionIndex;
    int32_t  shaderIndex;
    uint32_t indexStart;
    uint32_t indexLength;
    uint32_t flags;        // SubmeshFlags @ submeshBlock+0x12 (Reclaimer);
                           // bit 0 = IsWaterSurface, bit 8 = IsTransparent.
};

struct ShaderEntry {
    int32_t  shaderTagId;       // -1 if no tag reference
    char     shaderClass[5];    // tag class (e.g. "rmsh"); zeroed if unresolved
};

// One render_model NodeBlock entry - only the bits we actually need at decode
// time. Reach NodeBlock is [FixedSize(96)] with Position@12 (3 floats) and
// Rotation@24 (4 floats, Quaternion x,y,z,w). The remaining fields (parent,
// children, inverseTransform, distanceFromParent) are unused for static-rigid
// vertex baking.
struct NodeEntry {
    float    translation[3];
    float    rotation[4];          // x, y, z, w
    int16_t  parentIndex;          // -1 = root; chain of bones forms the local -> model transform
};

struct ModelData {
    uint64_t       cacheHandle;
    uint32_t       modeTagId;
    int32_t        resourceIndex;            // gestalt entry index

    std::vector<ModelSection>  sections;
    std::vector<ModelSubmesh>  submeshes;
    std::vector<ShaderEntry>   shaders;
    std::vector<NodeEntry>     nodes;

    // Section indices that belong to permutation 0 of every region - used
    // to filter out the fan of optional armor variants on player bipeds
    // (helmet variants, chest variants, etc.) so we render ONE coherent
    // armor set instead of overlapping all of them. Empty = no filter
    // (e.g. simple props with no Regions block); the adapter renders all
    // sections in that case. See `BuildAllowedSectionSet`.
    std::vector<uint8_t>       allowedSection; // size = sections.size(); 1 = render

    // Decoded resource page payload - owned by this model. Allocated in
    // OpenModel via ReadResourceData and freed in CloseModel. Kept alive
    // because section decodes can be called repeatedly.
    uint8_t*       resourceData = nullptr;
    size_t         resourceSize = 0;

    std::mutex     decodeMutex;
};

// Handle table for ZH_ModelHandle.
std::mutex g_modelHandlesMutex;
std::unordered_map<uint64_t, ModelData*> g_modelHandles;
std::atomic<uint64_t> g_nextModelHandle{ 1 };

ModelData* LookupModel(ZH_ModelHandle h) {
    std::lock_guard<std::mutex> lk(g_modelHandlesMutex);
    auto it = g_modelHandles.find(h);
    return it == g_modelHandles.end() ? nullptr : it->second;
}

// -----------------------------------------------------------------------------
// Model schema parse
//
// Defaults to MccHaloReach Retail / U8 / U10 / U13 layout (the user's actual
// build). All these share the same render_model offsets - only the Beta builds
// differ, which we don't support.
// -----------------------------------------------------------------------------

constexpr int OFF_MODE_REGIONS         = 12;
constexpr int OFF_MODE_NODES           = 48;
constexpr int OFF_MODE_SHADERS         = 72;
constexpr int OFF_MODE_SECTIONS        = 104;
constexpr int OFF_MODE_BOUNDING_BOXES  = 116;
constexpr int OFF_MODE_RESOURCE_PTR    = 248;

constexpr int SECTION_BLOCK_SIZE       = 92;
constexpr int BOUNDING_BOX_BLOCK_SIZE  = 52;
constexpr int SUBMESH_BLOCK_SIZE       = 24;
constexpr int SHADER_BLOCK_SIZE        = 44;  // Reclaimer's [FixedSize(44)] on render_model.ShaderBlock - TagReference at +0, 28 bytes of internal padding after
constexpr int NODE_BLOCK_SIZE          = 96;  // Reclaimer's [FixedSize(96)] on render_model.NodeBlock
constexpr int VERTEX_BUFFER_INFO_SIZE  = 28;
constexpr int INDEX_BUFFER_INFO_SIZE   = 28;

bool ParseSubmeshes(CacheHandle* cache, uint32_t submeshesPointer, int32_t submeshesCount,
                    uint32_t sectionIndex, std::vector<ModelSubmesh>& outAll)
{
    if (submeshesCount <= 0) return true;
    if (submeshesCount > 0x10000) return false;
    int64_t off = TagMetaFileOff(cache, submeshesPointer);
    if (off < 0 ||
        (size_t)off + (size_t)submeshesCount * SUBMESH_BLOCK_SIZE > cache->size)
        return false;

    for (int i = 0; i < submeshesCount; ++i) {
        const uint8_t* sm = cache->base + off + i * SUBMESH_BLOCK_SIZE;
        ModelSubmesh m;
        m.sectionIndex = sectionIndex;
        m.shaderIndex  = R16(sm + 0);
        m.indexStart   = (uint32_t)R32(sm + 4);
        m.indexLength  = (uint32_t)R32(sm + 8);
        m.flags        = (uint32_t)(uint16_t)R16(sm + 18); // SubmeshFlags
        outAll.push_back(m);
    }
    return true;
}

bool ParseShaders(CacheHandle* cache, uint32_t shadersPointer, int32_t shadersCount,
                  std::vector<ShaderEntry>& out)
{
    if (shadersCount <= 0) return true;
    if (shadersCount > 0x10000) return false;
    int64_t off = TagMetaFileOff(cache, shadersPointer);
    if (off < 0 ||
        (size_t)off + (size_t)shadersCount * SHADER_BLOCK_SIZE > cache->size)
        return false;

    out.resize(shadersCount);

    // Diagnostic: log the first 4 shaders' resolved class + bitmap so we can
    // confirm the +12 offset is reading real tag references. Capped per session.
    static std::atomic<int> s_shaderParseDiag{ 4 };
    auto wantDiag = [&]() -> bool {
        int v = s_shaderParseDiag.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_shaderParseDiag.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed))
                return true;
        }
        return false;
    };

    for (int i = 0; i < shadersCount; ++i) {
        // ShaderBlock is 44 bytes; ShaderReference (TagReference, 16 bytes) at +0.
        // TagReference layout for Gen3+ MCC: ClassId@0, +4..11 skipped, TagId@12.
        // Reclaimer.Blam.Common.TagReference.TagId getter is:
        //     (short)(tagId & ushort.MaxValue)
        // The raw int at +12 frequently has high bits set (engine identity /
        // generation bits - e.g. 0x85821DC8 for a tagId=0x1DC8). The OLD check
        // `if (tagId < 0)` rejected every such reference because the high bit
        // is set, so we got -1 for every shader. The correct logic mirrors
        // Reclaimer exactly: mask to ushort first, then cast to short, then
        // check for the -1 sentinel (0xFFFF == null).
        const uint8_t* sb = cache->base + off + i * SHADER_BLOCK_SIZE;
        uint32_t rawId = (uint32_t)R32(sb + 12);
        memset(out[i].shaderClass, 0, sizeof(out[i].shaderClass));

        if (rawId == 0xFFFFFFFFu) {
            // Genuine null reference - Reclaimer's TagId would return -1 here.
            out[i].shaderTagId = -1;
        } else {
            int16_t shortId = (int16_t)(rawId & 0xFFFFu);
            if (shortId < 0) {
                out[i].shaderTagId = -1;
            } else {
                out[i].shaderTagId = (int32_t)(uint16_t)shortId;
                if ((size_t)out[i].shaderTagId < cache->tags.size()) {
                    memcpy(out[i].shaderClass,
                           cache->tags[out[i].shaderTagId].classCode, 4);
                }
            }
        }

        if (wantDiag()) {
            uint32_t classId = (uint32_t)R32(sb + 0);
            // The on-disk class id reads as bytes that spell the class
            // backwards (engine stores it as the natural-order ASCII; R32
            // little-endian then reverses the byte order). Decode for log.
            char classChars[5] = {
                (char)((classId >> 24) & 0xFF),
                (char)((classId >> 16) & 0xFF),
                (char)((classId >>  8) & 0xFF),
                (char)((classId >>  0) & 0xFF),
                0
            };
            NativeDiag("ParseShader[%d]: refClass=%s rawId=0x%08X tagId=%d "
                       "resolvedClass=%s",
                i, classChars, rawId, out[i].shaderTagId,
                out[i].shaderClass[0] ? out[i].shaderClass : "<none>");
        }
    }
    return true;
}

bool ParseNodes(CacheHandle* cache, uint32_t nodesPointer, int32_t nodesCount,
                std::vector<NodeEntry>& out)
{
    if (nodesCount <= 0) return true;
    if (nodesCount > 0x10000) return false;
    int64_t off = TagMetaFileOff(cache, nodesPointer);
    if (off < 0 ||
        (size_t)off + (size_t)nodesCount * NODE_BLOCK_SIZE > cache->size)
        return false;

    out.resize(nodesCount);

    // Diagnostic: log the first 4 nodes' translation + quaternion at parse
    // time so the bake can be audited offline. Capped per session.
    static std::atomic<int> s_nodeParseDiag{ 4 };
    auto wantDiag = [&]() -> bool {
        int v = s_nodeParseDiag.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_nodeParseDiag.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed))
                return true;
        }
        return false;
    };

    for (int i = 0; i < nodesCount; ++i) {
        // NodeBlock: name@0 (StringId, 4), parentIndex@4 (i16),
        // firstChildIndex@6 (i16), nextSiblingIndex@8 (i16), pad@10 (2),
        // position@12 (3 floats), rotation@24 (4 floats, x/y/z/w),
        // inverseScale@40 (float), inverseTransform@44 (Matrix3x4 affine,
        // 48 bytes), distanceFromParent@92 (float).
        const uint8_t* nb = cache->base + off + i * NODE_BLOCK_SIZE;
        memcpy(out[i].translation, nb + 12, 12);
        memcpy(out[i].rotation,    nb + 24, 16);
        out[i].parentIndex = (int16_t)R16(nb + 4);

        if (wantDiag()) {
            const float* t = out[i].translation;
            const float* q = out[i].rotation;
            NativeDiag("ParseNode[%d]: parent=%d pos=(%g, %g, %g) rot=(%g, %g, %g, %g)",
                i, (int)out[i].parentIndex,
                t[0], t[1], t[2], q[0], q[1], q[2], q[3]);
        }
    }
    return true;
}

// Reads vertex/index buffer info arrays from the resource entry's fixup
// region (located inside the gestalt's FixupData blob). Returns false on
// any parse failure.
bool ParseFixupRegion(CacheHandle* cache, const ResourceEntry& entry,
                      std::vector<uint32_t>& vertexCounts,
                      std::vector<uint32_t>& vertexDataLengths,
                      std::vector<uint8_t>&  indexFormats,
                      std::vector<uint32_t>& indexDataLengths,
                      uint32_t diagModeTagId = 0xFFFFFFFFu)
{
    // The fixup data (post-fixup blob) lives in the gestalt - in Reclaimer
    // it's accessed as virtualReader from FixupDataPointer. We implement that
    // by reading the gestalt's FixupDataPointer once and using
    // (entry.FixupOffset + ...) inside that blob.
    int zoneIdx = FindGlobalTag(cache, "zone");
    if (zoneIdx < 0) { NativeDiag("FixupRegion: no zone tag"); return false; }
    int64_t metaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + 350 > cache->size) {
        NativeDiag("FixupRegion: bad zone meta off=%lld raw=0x%x",
            (long long)metaOff, cache->tags[zoneIdx].metaPointerRaw);
        return false;
    }
    const uint8_t* meta = cache->base + metaOff;

    // FixupDataSize @ +328, FixupDataPointer @ +340 (Pointer, expanded).
    int32_t fixupSize  = R32(meta + 328);
    uint32_t fixupPtrR = RU32(meta + 340);
    int64_t fixupOff   = TagMetaFileOff(cache, fixupPtrR);
    if (fixupSize < 0 || fixupOff < 0) {
        NativeDiag("FixupRegion: bad fixupData fixupSize=%d ptrRaw=0x%x off=%lld",
            fixupSize, fixupPtrR, (long long)fixupOff);
        return false;
    }
    if ((size_t)fixupOff + (size_t)fixupSize > cache->size) {
        NativeDiag("FixupRegion: fixupData OOB off=%lld size=%d cacheSize=%llu",
            (long long)fixupOff, fixupSize, (unsigned long long)cache->size);
        return false;
    }
    const uint8_t* fixupBase = cache->base + fixupOff;

    // Trailer (last 24 bytes of this entry's slice): VBcount @ +0, skip 8,
    // IBcount @ +12.
    if (entry.fixupSize < 24) {
        NativeDiag("FixupRegion: entry fixupSize<24 entry.fixupOff=%d fixupSize=%d",
            entry.fixupOffset, entry.fixupSize);
        return false;
    }
    int64_t trailerOff = (int64_t)entry.fixupOffset + (int64_t)entry.fixupSize - 24;
    if (trailerOff < 0 || (size_t)trailerOff + 16 > (size_t)fixupSize) {
        NativeDiag("FixupRegion: trailer OOB trailerOff=%lld entry.foff=%d entry.fsz=%d gestaltFsz=%d",
            (long long)trailerOff, entry.fixupOffset, entry.fixupSize, fixupSize);
        return false;
    }
    int32_t vbCount = R32(fixupBase + trailerOff);
    int32_t ibCount = R32(fixupBase + trailerOff + 12);
    if (vbCount < 0 || vbCount > 0x10000) {
        NativeDiag("FixupRegion: bad vbCount=%d entry.foff=%d entry.fsz=%d",
            vbCount, entry.fixupOffset, entry.fixupSize);
        return false;
    }
    if (ibCount < 0 || ibCount > 0x10000) {
        NativeDiag("FixupRegion: bad ibCount=%d entry.foff=%d entry.fsz=%d",
            ibCount, entry.fixupOffset, entry.fixupSize);
        return false;
    }

    // Walk forward from entry.fixupOffset.
    size_t cursor = (size_t)entry.fixupOffset;
    if (cursor + (size_t)vbCount * VERTEX_BUFFER_INFO_SIZE > (size_t)fixupSize) {
        NativeDiag("FixupRegion: vb array OOB cursor=%llu vbCount=%d gestaltFsz=%d",
            (unsigned long long)cursor, vbCount, fixupSize);
        return false;
    }

    vertexCounts.resize(vbCount);
    vertexDataLengths.resize(vbCount);
    for (int i = 0; i < vbCount; ++i) {
        const uint8_t* p = fixupBase + cursor + i * VERTEX_BUFFER_INFO_SIZE;
        vertexCounts[i]      = (uint32_t)R32(p + 0);
        vertexDataLengths[i] = (uint32_t)R32(p + 8);
    }
    cursor += (size_t)vbCount * VERTEX_BUFFER_INFO_SIZE;

    // Dump the per-VB 12-byte stride/format structs so the true
    // vertex-COLOR stream layout (d3dcolor stride-4) can be confirmed on real bytes before the
    // decode path is changed. Gated on HMS_VBDIAG so normal runs are unaffected.
    if (getenv("HMS_VBDIAG")) {
        NativeDiag("VBDIAG mode=0x%x vbCount=%d ibCount=%d", diagModeTagId, vbCount, ibCount);
        for (int i = 0; i < vbCount && i < 16; ++i) {
            const uint8_t* sp = fixupBase + cursor + (size_t)i * 12;
            NativeDiag("VBDIAG vb[%d] count=%u dataLen=%u stride12=%02x %02x %02x %02x  %02x %02x %02x %02x  %02x %02x %02x %02x",
                i, vertexCounts[i], vertexDataLengths[i],
                sp[0],sp[1],sp[2],sp[3], sp[4],sp[5],sp[6],sp[7], sp[8],sp[9],sp[10],sp[11]);
        }
    }

    // Skip 12-byte stride structs per VB.
    cursor += (size_t)vbCount * 12;
    if (cursor + (size_t)ibCount * INDEX_BUFFER_INFO_SIZE > (size_t)fixupSize) {
        NativeDiag("FixupRegion: ib array OOB cursor=%llu ibCount=%d gestaltFsz=%d",
            (unsigned long long)cursor, ibCount, fixupSize);
        return false;
    }

    indexFormats.resize(ibCount);
    indexDataLengths.resize(ibCount);
    for (int i = 0; i < ibCount; ++i) {
        const uint8_t* p = fixupBase + cursor + i * INDEX_BUFFER_INFO_SIZE;
        indexFormats[i]    = (uint8_t)(R32(p + 0) & 0xFF);
        indexDataLengths[i] = (uint32_t)R32(p + 8);
    }
    return true;
}

// Cap the diagnostic spam - only the first N mode tags log their full path,
// the rest only log the failing-step name (no offsets/values). N = 4 gives us
// enough variety to spot intermittent issues without filling the log.
static std::atomic<int> g_modeDiagBudget{ 4 };
static bool ShouldLogModeDiag() {
    int v = g_modeDiagBudget.load(std::memory_order_relaxed);
    while (v > 0) {
        if (g_modeDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed))
            return true;
    }
    return false;
}
// Parse the full mode tag -> ModelData. Caller owns 'data' via std::unique_ptr-like
// semantics; on failure returns false and 'data' is left in indeterminate state.
bool ParseModeTag(CacheHandle* cache, uint32_t modeTagId, ModelData& data) {
    bool logThis = ShouldLogModeDiag();
    if (modeTagId >= cache->tags.size()) {
        if (logThis) NativeDiag("Mode[%u]: tagId OOB tagsCount=%llu",
            modeTagId, (unsigned long long)cache->tags.size());
        return false;
    }
    const TagEntry& te = cache->tags[modeTagId];
    if (te.classIndex < 0) {
        if (logThis) NativeDiag("Mode[%u]: classIndex<0", modeTagId);
        return false;
    }
    if (memcmp(te.classCode, "mode", 4) != 0) {
        // Common - caller passes any tag id; not really a parse failure.
        if (logThis) NativeDiag("Mode[%u]: not mode (class='%c%c%c%c')",
            modeTagId, te.classCode[0], te.classCode[1], te.classCode[2], te.classCode[3]);
        return false;
    }

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_MODE_RESOURCE_PTR + 4) > cache->size) {
        if (logThis) NativeDiag("Mode[%u]: bad meta raw=0x%x off=%lld size=%llu need=%d",
            modeTagId, te.metaPointerRaw, (long long)metaOff,
            (unsigned long long)cache->size, OFF_MODE_RESOURCE_PTR + 4);
        else NativeDiag("Mode[%u]: bad meta", modeTagId);
        return false;
    }
    const uint8_t* meta = cache->base + metaOff;

    // ResourcePointer is a 32-bit identifier. ResourceIndex = value & 0xFFFF.
    int32_t resourceId = R32(meta + OFF_MODE_RESOURCE_PTR);
    int32_t resourceIndex = resourceId & 0xFFFF;
    data.resourceIndex = resourceIndex;

    // Sections + BoundingBoxes + Shaders + Nodes blocks.
    TagBlockRef sectionsBlk = ReadTagBlock(meta + OFF_MODE_SECTIONS);
    TagBlockRef boundsBlk   = ReadTagBlock(meta + OFF_MODE_BOUNDING_BOXES);
    TagBlockRef shadersBlk  = ReadTagBlock(meta + OFF_MODE_SHADERS);
    TagBlockRef nodesBlk    = ReadTagBlock(meta + OFF_MODE_NODES);
    TagBlockRef regionsBlk  = ReadTagBlock(meta + OFF_MODE_REGIONS);

    if (logThis) NativeDiag("Mode[%u]: meta@%lld rid=0x%x rIdx=%d sections=%d/0x%x bounds=%d/0x%x shaders=%d/0x%x nodes=%d/0x%x",
        modeTagId, (long long)metaOff, resourceId, resourceIndex,
        sectionsBlk.count, sectionsBlk.pointer, boundsBlk.count, boundsBlk.pointer,
        shadersBlk.count, shadersBlk.pointer, nodesBlk.count, nodesBlk.pointer);

    if (sectionsBlk.count < 0 || sectionsBlk.count > 0x10000) {
        if (logThis) NativeDiag("Mode[%u]: bad sectionsBlk count=%d", modeTagId, sectionsBlk.count);
        else NativeDiag("Mode[%u]: bad sectionsBlk", modeTagId);
        return false;
    }
    if (boundsBlk.count   < 0 || boundsBlk.count   > 0x1000) {
        if (logThis) NativeDiag("Mode[%u]: bad boundsBlk count=%d", modeTagId, boundsBlk.count);
        else NativeDiag("Mode[%u]: bad boundsBlk", modeTagId);
        return false;
    }

    // Bounds[0] is shared model-wide.
    float posMin[3] = {0,0,0}, posMax[3] = {1,1,1};
    float uvMin[2]  = {0,0},   uvMax[2]  = {1,1};
    if (boundsBlk.count > 0) {
        int64_t boundsOff = TagMetaFileOff(cache, boundsBlk.pointer);
        if (boundsOff < 0 ||
            (size_t)boundsOff + BOUNDING_BOX_BLOCK_SIZE > cache->size) {
            if (logThis) NativeDiag("Mode[%u]: bad bounds off=%lld ptr=0x%x",
                modeTagId, (long long)boundsOff, boundsBlk.pointer);
            else NativeDiag("Mode[%u]: bad bounds", modeTagId);
            return false;
        }
        const uint8_t* bb = cache->base + boundsOff;
        // RealBounds = (float min, float max). Bounds layout per Reclaimer:
        //   +4  XBounds, +12 YBounds, +20 ZBounds, +28 UBounds, +36 VBounds.
        memcpy(&posMin[0], bb + 4,  4); memcpy(&posMax[0], bb + 8,  4);
        memcpy(&posMin[1], bb + 12, 4); memcpy(&posMax[1], bb + 16, 4);
        memcpy(&posMin[2], bb + 20, 4); memcpy(&posMax[2], bb + 24, 4);
        memcpy(&uvMin[0],  bb + 28, 4); memcpy(&uvMax[0],  bb + 32, 4);
        memcpy(&uvMin[1],  bb + 36, 4); memcpy(&uvMax[1],  bb + 40, 4);
    }

    // Parse shaders -> tag id table. Failure here is non-fatal; we just won't
    // resolve diffuse bitmaps.
    ParseShaders(cache, shadersBlk.pointer, shadersBlk.count, data.shaders);

    // Parse nodes -> per-bone translation + rotation. Failure here is non-fatal;
    // sections that reference an unparsable node fall back to identity transform.
    if (nodesBlk.count > 0 && nodesBlk.count <= 0x10000) {
        if (!ParseNodes(cache, nodesBlk.pointer, nodesBlk.count, data.nodes)) {
            data.nodes.clear();
            if (logThis) NativeDiag("Mode[%u]: ParseNodes failed (count=%d ptr=0x%x) - falling back to identity bone transforms",
                modeTagId, nodesBlk.count, nodesBlk.pointer);
        }
    }

    // Bug 2 diag: dump the first node's pos/rot for ff_plat_* / ff_bridge_*
    // tags so we can see whether they have a non-identity baked transform we
    // need to handle. Rate-limited per tag-name-prefix via static budget.
    {
        const std::string& nm = te.tagName;
        bool isForge = (nm.find("ff_plat_") != std::string::npos) ||
                       (nm.find("ff_bridge_") != std::string::npos) ||
                       (nm.find("ff_5x5") != std::string::npos);
        if (isForge && !data.nodes.empty()) {
            static std::atomic<int> s_forgeDiagBudget{ 8 };
            int v = s_forgeDiagBudget.load(std::memory_order_relaxed);
            while (v > 0) {
                if (s_forgeDiagBudget.compare_exchange_weak(v, v - 1,
                    std::memory_order_relaxed)) {
                    const NodeEntry& n0 = data.nodes[0];
                    NativeDiag("ForgeNodeDiag tag='%s' nodeCount=%d node[0] parent=%d "
                               "pos=(%g,%g,%g) rot=(%g,%g,%g,%g)",
                        nm.c_str(), (int)data.nodes.size(), (int)n0.parentIndex,
                        n0.translation[0], n0.translation[1], n0.translation[2],
                        n0.rotation[0], n0.rotation[1], n0.rotation[2], n0.rotation[3]);
                    break;
                }
            }
        }
    }

    // Parse sections.
    if (sectionsBlk.count > 0) {
        int64_t secOff = TagMetaFileOff(cache, sectionsBlk.pointer);
        if (secOff < 0 ||
            (size_t)secOff + (size_t)sectionsBlk.count * SECTION_BLOCK_SIZE > cache->size) {
            if (logThis) NativeDiag("Mode[%u]: bad sections off=%lld count=%d ptr=0x%x",
                modeTagId, (long long)secOff, sectionsBlk.count, sectionsBlk.pointer);
            else NativeDiag("Mode[%u]: bad sections", modeTagId);
            return false;
        }

        data.sections.resize(sectionsBlk.count);
        for (int i = 0; i < sectionsBlk.count; ++i) {
            const uint8_t* s = cache->base + secOff + i * SECTION_BLOCK_SIZE;
            ModelSection& sec = data.sections[i];

            TagBlockRef submeshesBlk = ReadTagBlock(s + 0);   // Submeshes @ +0
            // Subsets @ +12 (we don't currently expose subsets - render-model
            // path uses submeshes; subsets are only needed for instanced BSP).

            sec.vertexBufferIndex = R16(s + 24);
            sec.indexBufferIndex  = R16(s + 40);
            sec.flags             = (uint16_t)R16(s + 44);
            sec.nodeIndex         = s[46];
            sec.vertexFormat      = s[47];
            sec.indexFormat       = s[50];
            sec.isUnindexed       = (sec.indexBufferIndex == -1) ||
                                    ((sec.flags & 0x10) != 0); // MeshIsUnindexed
            sec.vbDataLength      = 0;
            sec.vertexCount       = 0;
            sec.ibDataLength      = 0;
            sec.indexCount        = 0;
            sec.vbResourceOffset  = 0;
            sec.ibResourceOffset  = 0;
            sec.materialIndex     = -1;
            sec.colorResourceOffset = 0;
            sec.colorDataLength   = 0;
            sec.colorStride       = 0;
            memcpy(sec.posMin, posMin, sizeof(posMin));
            memcpy(sec.posMax, posMax, sizeof(posMax));
            memcpy(sec.uvMin,  uvMin,  sizeof(uvMin));
            memcpy(sec.uvMax,  uvMax,  sizeof(uvMax));

            sec.submeshStart = (uint32_t)data.submeshes.size();
            ParseSubmeshes(cache, submeshesBlk.pointer, submeshesBlk.count,
                           (uint32_t)i, data.submeshes);
            sec.submeshCount = (uint32_t)(data.submeshes.size() - sec.submeshStart);

            // Pick the "primary" submesh's shader index as the section's
            // material - the viewer can iterate all submeshes for finer
            // granularity. -1 if no submeshes.
            if (sec.submeshCount > 0)
                sec.materialIndex = data.submeshes[sec.submeshStart].shaderIndex;

            // Per-section layout diagnostics (first ~2 sections of each
            // logged-mode tag). Helps detect SECTION_BLOCK_SIZE / offset drift
            // for U13 forge_halo render_model layout. vbDataLength /
            // vertexCount aren't hooked up yet at this point - they are
            // populated below from the resource fixup region; we log them
            // as 0 here intentionally and again after the hook step would
            // be redundant. The fields directly read from the section
            // block (vertexFormat / buffer indices / flags / submeshCount)
            // are what we actually want to verify against Reclaimer.
            if (logThis && i < 2) {
                NativeDiag("Mode[%u]: sec[%d] vbIdx=%d ibIdx=%d flags=0x%04x "
                           "submeshes=%u vfmt=0x%02x ifmt=0x%02x node=%u "
                           "stride=%d (raw s+47=0x%02x s+50=0x%02x)",
                    modeTagId, i, sec.vertexBufferIndex, sec.indexBufferIndex,
                    sec.flags, sec.submeshCount, sec.vertexFormat,
                    sec.indexFormat, sec.nodeIndex, SECTION_BLOCK_SIZE,
                    s[47], s[50]);
            }
        }
    }

    // Parse the resource entry's fixup region (vb/ib counts + sizes).
    if (!ParseGestalt(cache)) {
        if (logThis) NativeDiag("Mode[%u]: ParseGestalt failed", modeTagId);
        else NativeDiag("Mode[%u]: ParseGestalt failed", modeTagId);
        return false;
    }
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size()) {
        if (logThis) NativeDiag("Mode[%u]: rIdx OOB rIdx=%d count=%llu rid=0x%x",
            modeTagId, resourceIndex, (unsigned long long)cache->resourceEntries.size(), resourceId);
        else NativeDiag("Mode[%u]: rIdx OOB", modeTagId);
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(cache->parseMutex);
        if (!EnsureResourceFixups(cache, (size_t)resourceIndex)) {
            if (logThis) NativeDiag("Mode[%u]: EnsureResourceFixups failed rIdx=%d",
                modeTagId, resourceIndex);
            else NativeDiag("Mode[%u]: EnsureResourceFixups failed", modeTagId);
            return false;
        }
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];
    if (logThis) NativeDiag("Mode[%u]: entry rIdx=%d rPtr=0x%x fOff=%d fSz=%d segIdx=%d fixups=%d",
        modeTagId, resourceIndex, entry.resourcePointer,
        entry.fixupOffset, entry.fixupSize, entry.segmentIndex, (int)entry.fixups.size());

    std::vector<uint32_t> vbCounts, vbLens, ibLens;
    std::vector<uint8_t>  ibFmts;
    if (!ParseFixupRegion(cache, entry, vbCounts, vbLens, ibFmts, ibLens, modeTagId)) {
        if (logThis) NativeDiag("Mode[%u]: ParseFixupRegion failed", modeTagId);
        else NativeDiag("Mode[%u]: ParseFixupRegion failed", modeTagId);
        return false;
    }
    if (logThis) NativeDiag("Mode[%u]: fixupRegion vb=%d ib=%d",
        modeTagId, (int)vbCounts.size(), (int)ibLens.size());

    // Hook each section to its vb/ib resource-page offsets.
    for (auto& sec : data.sections) {
        if (sec.vertexBufferIndex < 0 ||
            sec.vertexBufferIndex >= (int)vbCounts.size())
            continue;

        sec.vertexCount   = vbCounts[sec.vertexBufferIndex];
        sec.vbDataLength  = vbLens[sec.vertexBufferIndex];

        // Resource-page offset for this VB.
        if ((size_t)sec.vertexBufferIndex < entry.fixups.size()) {
            sec.vbResourceOffset = (uint32_t)(entry.fixups[sec.vertexBufferIndex].offset & 0x0FFFFFFF);
        }

        // Detect a per-vertex COLOR stream = the NEXT vertex buffer
        // (vertexBufferIndex+1) when it has the SAME vertex count and a 12-byte (float3) stride.
        // This is the vertex-gradient sky-dome color the engine's sky_dome_simple reads (verified
        // on condemned: vb[1] float3 in [0,1]). Very specific gate -> won't false-positive on normal
        // models (their second stream, if any, differs in count/stride); read only from the sky path.
        {
            int ci = sec.vertexBufferIndex + 1;
            if (ci >= 0 && ci < (int)vbCounts.size() && vbCounts[ci] == sec.vertexCount
                && vbCounts[ci] > 0 && (size_t)ci < entry.fixups.size()) {
                uint32_t cstride = vbLens[ci] / vbCounts[ci];
                if (cstride == 12) {
                    sec.colorStride = cstride;
                    sec.colorDataLength = vbLens[ci];
                    sec.colorResourceOffset = (uint32_t)(entry.fixups[ci].offset & 0x0FFFFFFF);
                }
            }
        }

        if (!sec.isUnindexed &&
            sec.indexBufferIndex >= 0 &&
            sec.indexBufferIndex < (int)ibLens.size())
        {
            sec.ibDataLength = ibLens[sec.indexBufferIndex];
            // Index format from the engine info struct (overrides what's on
            // the section header - Reclaimer trusts the gestalt).
            if ((size_t)sec.indexBufferIndex < ibFmts.size())
                sec.indexFormat = ibFmts[sec.indexBufferIndex];

            // Index count = data length / index size.
            uint32_t indexStride = (sec.vertexCount > 0xFFFF) ? 4u : 2u;
            sec.indexCount = sec.ibDataLength / indexStride;

            size_t ibFixupIdx = vbCounts.size() * 2 + (size_t)sec.indexBufferIndex;
            if (ibFixupIdx < entry.fixups.size()) {
                sec.ibResourceOffset =
                    (uint32_t)(entry.fixups[ibFixupIdx].offset & 0x0FFFFFFF);
            }
        } else if (sec.isUnindexed) {
            // Implied buffer - index list is just [0..vertexCount).
            sec.indexCount = sec.vertexCount;
        }
    }

    // Read + decompress the resource page payload (kept alive on the model).
    // ReadResourceData takes cache->parseMutex internally for the shared-cache
    // open path, so DON'T hold it here - std::mutex is non-recursive and a
    // double-lock raises a system_error that SehParseModeTag silently swallows.
    constexpr size_t kMaxRead = 64 * 1024 * 1024;
    data.resourceData = ReadResourceData(cache, resourceId, kMaxRead, &data.resourceSize);
    if (!data.resourceData) {
        if (logThis) NativeDiag("Mode[%u]: ReadResourceData failed rid=0x%x", modeTagId, resourceId);
        else NativeDiag("Mode[%u]: ReadResourceData failed", modeTagId);
        return false;
    }

    // Dump vb[1] (the secondary per-vertex stream, suspected float3 COLOR) as
    // 3 floats + raw hex so we can confirm content (color values ~[0,1] vs normals ~[-1,1]).
    if (getenv("HMS_VBDIAG") && vbCounts.size() >= 2 && (size_t)1 < entry.fixups.size()) {
        uint32_t off1 = (uint32_t)(entry.fixups[1].offset & 0x0FFFFFFF);
        uint32_t len1 = vbLens[1];
        uint32_t stride1 = vbCounts[1] ? (len1 / vbCounts[1]) : 0;
        NativeDiag("VBDIAG[%u] vb1 off=0x%x len=%u stride=%u", modeTagId, off1, len1, stride1);
        uint32_t n1 = vbCounts[1];
        if (stride1 >= 12 && (size_t)off1 + (size_t)n1 * stride1 <= data.resourceSize) {
            // Scan the WHOLE stream as float3: min/max per channel + nonzero count. Tells us
            // if it's color (values ~[0,1]), normals (~[-1,1]), or something else.
            float mn[3] = {1e9f,1e9f,1e9f}, mx[3] = {-1e9f,-1e9f,-1e9f};
            uint32_t nonzero = 0;
            for (uint32_t v = 0; v < n1; ++v) {
                const uint8_t* p = data.resourceData + off1 + (size_t)v * stride1;
                float f[3]; memcpy(f, p, 12);
                if (f[0] || f[1] || f[2]) nonzero++;
                for (int c = 0; c < 3; ++c) { if (f[c] < mn[c]) mn[c] = f[c]; if (f[c] > mx[c]) mx[c] = f[c]; }
            }
            NativeDiag("VBDIAG[%u] vb1 SCAN n=%u nonzero=%u  R[%.3f..%.3f] G[%.3f..%.3f] B[%.3f..%.3f]",
                modeTagId, n1, nonzero, mn[0],mx[0], mn[1],mx[1], mn[2],mx[2]);
            // Also dump 3 mid-buffer sample verts.
            uint32_t si[3] = { n1/4, n1/2, (n1*3)/4 };
            for (int k = 0; k < 3; ++k) {
                const uint8_t* p = data.resourceData + off1 + (size_t)si[k] * stride1;
                float f[3]; memcpy(f, p, 12);
                NativeDiag("VBDIAG[%u] vb1 v%u f3=(%.4f,%.4f,%.4f)", modeTagId, si[k], f[0],f[1],f[2]);
            }
        }
    }

    // -----------------------------------------------------------------
    // Build allowedSection[] - section indices that belong to perm[0]
    // of every region. Without this filter, a Spartan biped would render
    // every helmet/chest/shoulder variant overlapping each other (Reach
    // packs all customization permutations into one render_model).
    //
    // RegionBlock layout (HaloReach Retail / U13):
    //   +0x00  StringId Name      (4 bytes)
    //   +0x04  TagBlock Permutations  (12 bytes count+ptr+pad)
    //   total  16 bytes
    //
    // PermutationBlock layout:
    //   +0x00  StringId Name           (4 bytes)
    //   +0x04  short    SectionIndex   (2 bytes)
    //   +0x06  short    SectionCount   (2 bytes - high byte usually padding)
    //   ... rest of fields irrelevant here ...
    //   total  16 bytes (Reclaimer's [FixedSize(16)])
    //
    // If Regions block is empty / not present (small static props), leave
    // allowedSection empty - the adapter renders all sections as before.
    // -----------------------------------------------------------------
    constexpr int REGION_BLOCK_SIZE      = 16;
    constexpr int OFF_REGION_PERMS       = 4;   // Permutations TagBlock within RegionBlock
    constexpr int PERMUTATION_BLOCK_SIZE = 16;

    data.allowedSection.assign(data.sections.size(), 0);
    if (regionsBlk.count > 0 && regionsBlk.count <= 0x1000)
    {
        int64_t regOff = TagMetaFileOff(cache, regionsBlk.pointer);
        if (regOff >= 0 &&
            (size_t)regOff + (size_t)regionsBlk.count * REGION_BLOCK_SIZE <= cache->size)
        {
            int allowedCount = 0;
            // #269 DEFAULT-DAMAGE-STATE SELECTION (ground-truthed against the Falcon + Warthog via
            // HMS_SECDIAG). Each 16-byte RenderModel permutation carries TWO section references, not
            // one: A = (sectionIndex@0x04, count@0x06) and B = (sectionIndex@0x0C, count@0x0E), with
            // a name-hash at 0x08. A region's permutations are ordered as damage STATES: the default
            // (undamaged) permutations come first, then - once a permutation named minor/medium/major/
            // destroyed/damaged appears - every following permutation IN THAT REGION belongs to the
            // damage state (its A/B sections are damaged geometry, including the low-LOD damage pieces
            // that live in un-named '?' permutations after the damage perm). So the rule is POSITIONAL,
            // per region: walk perms in order, flip into damageMode at the first state-damage-named
            // perm, and skip everything from there on. 'blur' is the Falcon rotor's motion-blur disc - 
            // a local effect variant, NOT a state boundary - so we exclude just its A section and keep
            // its B section (the STATIC blades, which the engine stores in the following data). This
            // makes the Falcon complete (tail/wings/engine/blades come from B-refs) while the Warthog
            // drops its minor/medium/major hull+fender damage AND their low-LOD counterparts.
            auto toLowerBuf = [](const char* s, char* out, size_t cap) {
                size_t n = 0;
                if (s) for (; s[n] && n < cap - 1; ++n) {
                    char c = s[n];
                    out[n] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
                }
                out[n] = 0;
            };
            // A state-damage boundary: this permutation and every one after it in the region is damage.
            auto hasStateDamageToken = [&](const char* s) -> bool {
                if (!s || !*s) return false;
                char low[160]; toLowerBuf(s, low, sizeof(low));
                static const char* toks[] = { "destroy", "damag", "broken", "crack", "minor", "medium", "major" };
                for (const char* t : toks) if (strstr(low, t)) return true;
                return false;
            };
            auto hasBlurToken = [&](const char* s) -> bool {
                if (!s || !*s) return false;
                char low[160]; toLowerBuf(s, low, sizeof(low));
                return strstr(low, "blur") != nullptr;
            };
            // Section -> shader tag name (materialIndex -> shaders[].shaderTagId -> tags[].tagName).
            auto sectionShaderName = [&](size_t i) -> const char* {
                if (i >= data.sections.size()) return nullptr;
                int mi = data.sections[i].materialIndex;
                if (mi < 0 || (size_t)mi >= data.shaders.size()) return nullptr;
                int32_t st = data.shaders[mi].shaderTagId;
                if (st < 0 || (size_t)st >= cache->tags.size()) return nullptr;
                return cache->tags[st].tagName.c_str();
            };
            std::vector<uint8_t> referenced(data.sections.size(), 0);
            auto markAllow = [&](int idx, int cnt) {
                if (idx < 0 || cnt <= 0) return;
                for (int s = 0; s < cnt; ++s) {
                    int i = idx + s;
                    if (i >= 0 && (size_t)i < data.allowedSection.size()) {
                        referenced[i] = 1;
                        if (data.allowedSection[i] == 0) { data.allowedSection[i] = 1; ++allowedCount; }
                    }
                }
            };
            for (int r = 0; r < regionsBlk.count; ++r)
            {
                const uint8_t* regBase = cache->base + regOff + (size_t)r * REGION_BLOCK_SIZE;
                TagBlockRef permsBlk = ReadTagBlock(regBase + OFF_REGION_PERMS);
                if (permsBlk.count <= 0 || permsBlk.count > 0x100) continue;
                int64_t permOff = TagMetaFileOff(cache, permsBlk.pointer);
                if (permOff < 0 ||
                    (size_t)permOff + (size_t)permsBlk.count * PERMUTATION_BLOCK_SIZE > cache->size)
                    continue;
                bool damageMode = false;
                for (int p = 0; p < permsBlk.count; ++p)
                {
                    const uint8_t* pe = cache->base + permOff + (size_t)p * PERMUTATION_BLOCK_SIZE;
                    const char* pn = ResolveStringId(cache, (int32_t)R32(pe + 0));
                    int aIdx = (int16_t)R16(pe + 4),  aCnt = (int16_t)R16(pe + 6);
                    int bIdx = (int16_t)R16(pe + 12), bCnt = (int16_t)R16(pe + 14);
                    if (hasStateDamageToken(pn)) damageMode = true; // this perm + rest of region = damage
                    if (damageMode) {                               // record refs (for diag) but never allow
                        if (aIdx >= 0 && aCnt > 0) for (int s = 0; s < aCnt; ++s) { int i = aIdx + s; if (i >= 0 && (size_t)i < referenced.size()) referenced[i] = 1; }
                        if (bIdx >= 0 && bCnt > 0) for (int s = 0; s < bCnt; ++s) { int i = bIdx + s; if (i >= 0 && (size_t)i < referenced.size()) referenced[i] = 1; }
                        continue;
                    }
                    if (hasBlurToken(pn)) {
                        // motion-blur disc: drop its A (the blur section), keep B (the static blades).
                        if (aIdx >= 0 && aCnt > 0) for (int s = 0; s < aCnt; ++s) { int i = aIdx + s; if (i >= 0 && (size_t)i < referenced.size()) referenced[i] = 1; }
                        markAllow(bIdx, bCnt);
                    } else {
                        markAllow(aIdx, aCnt);
                        markAllow(bIdx, bCnt);
                    }
                }
            }
            if (logThis)
                NativeDiag("Mode[%u]: regions=%d default-state -> %d/%d sections allowed",
                    modeTagId, regionsBlk.count, allowedCount, (int)data.sections.size());
            // Ground-truth dump of every section's shader name +
            // referenced/allowed flags, and every region/perm's resolved name + section range.
            // Read-only. Reveals which sections are damage and whether the token filter catches them.
            {
                static bool secdiag = []{
                    char* v = nullptr; size_t n = 0;
                    if (_dupenv_s(&v, &n, "HMS_SECDIAG") == 0 && v) { bool on = v[0] && v[0] != '0'; free(v); return on; }
                    return false;
                }();
                if (secdiag) {
                    for (size_t i = 0; i < data.sections.size(); ++i) {
                        const char* sn = sectionShaderName(i);
                        NativeDiag("SECDIAG Mode[%u] SECTION %zu shader='%s' referenced=%d allowed=%d",
                            modeTagId, i, sn ? sn : "<none>", (int)referenced[i], (int)data.allowedSection[i]);
                    }
                    for (int r = 0; r < regionsBlk.count; ++r) {
                        const uint8_t* rb = cache->base + regOff + (size_t)r * REGION_BLOCK_SIZE;
                        const char* rn = ResolveStringId(cache, (int32_t)R32(rb + 0));
                        TagBlockRef pb = ReadTagBlock(rb + OFF_REGION_PERMS);
                        if (pb.count <= 0 || pb.count > 0x100) continue;
                        int64_t pOff = TagMetaFileOff(cache, pb.pointer);
                        if (pOff < 0 || (size_t)pOff + (size_t)pb.count * PERMUTATION_BLOCK_SIZE > cache->size) continue;
                        for (int p = 0; p < pb.count; ++p) {
                            const uint8_t* pe = cache->base + pOff + (size_t)p * PERMUTATION_BLOCK_SIZE;
                            const char* pn = ResolveStringId(cache, (int32_t)R32(pe + 0));
                            NativeDiag("SECDIAG Mode[%u] region '%s' perm[%d] '%s' raw16=[%d %d %d %d %d %d]",
                                modeTagId, rn ? rn : "?", p, pn ? pn : "?",
                                (int)(int16_t)R16(pe + 4), (int)(int16_t)R16(pe + 6),
                                (int)(int16_t)R16(pe + 8), (int)(int16_t)R16(pe + 10),
                                (int)(int16_t)R16(pe + 12), (int)(int16_t)R16(pe + 14));
                        }
                    }
                }
            }
            // Safety: if NO sections matched (malformed data / bad offsets), allow all so we never
            // render an empty mesh.
            if (allowedCount == 0)
                std::fill(data.allowedSection.begin(), data.allowedSection.end(), (uint8_t)1);
        }
        else
        {
            // Couldn't read regions - allow all.
            std::fill(data.allowedSection.begin(), data.allowedSection.end(), (uint8_t)1);
        }
    }
    else
    {
        // No regions block at all -> simple model, allow everything.
        std::fill(data.allowedSection.begin(), data.allowedSection.end(), (uint8_t)1);
    }

    if (logThis) NativeDiag("Mode[%u]: parse OK sections=%d submeshes=%d resSz=%llu",
        modeTagId, (int)data.sections.size(), (int)data.submeshes.size(),
        (unsigned long long)data.resourceSize);
    return true;
}

// -----------------------------------------------------------------------------
// Vertex-format decode (formats 0x00 / 0x01 / 0x02 / 0x04 / 0x05 / 0x06 / 0x0F)
// -----------------------------------------------------------------------------

constexpr uint32_t VFMT_WORLD          = 0x00;
constexpr uint32_t VFMT_RIGID          = 0x01;
constexpr uint32_t VFMT_SKINNED        = 0x02;
constexpr uint32_t VFMT_FLAT_WORLD     = 0x04;
constexpr uint32_t VFMT_FLAT_RIGID     = 0x05;
constexpr uint32_t VFMT_FLAT_SKINNED   = 0x06;
constexpr uint32_t VFMT_DECORATOR      = 0x0F;

// Per-format strides (see XML in MccHaloReachVertexBuffer.xml).
constexpr uint32_t STRIDE_WORLD_RIGID    = 0x24;  // 36 bytes - fmt 0x00/0x01/0x04/0x05
constexpr uint32_t STRIDE_SKINNED        = 0x2C;  // 44 bytes - fmt 0x02/0x06
constexpr uint32_t STRIDE_DECORATOR      = 0x20;  // 32 bytes - fmt 0x0F

bool IsFormatSupported(uint32_t fmt) {
    switch (fmt) {
        case VFMT_WORLD:
        case VFMT_RIGID:
        case VFMT_SKINNED:
        case VFMT_FLAT_WORLD:
        case VFMT_FLAT_RIGID:
        case VFMT_FLAT_SKINNED:
        case VFMT_DECORATOR:
            return true;
    }
    return false;
}

// Stride for a given format. World/rigid (and their flat variants) share the
// 36-byte stride. Skinned formats add 8 bytes of bone data -> 44. Decorator
// is leaner - 32 bytes with raw float3 position + float2 uv + float3 normal.
uint32_t StrideForFormat(uint32_t fmt) {
    switch (fmt) {
        case VFMT_SKINNED:
        case VFMT_FLAT_SKINNED:
            return STRIDE_SKINNED;
        case VFMT_DECORATOR:
            return STRIDE_DECORATOR;
        case VFMT_WORLD:
        case VFMT_RIGID:
        case VFMT_FLAT_WORLD:
        case VFMT_FLAT_RIGID:
        default:
            return STRIDE_WORLD_RIGID;
    }
}

// IEEE 754 binary16 -> binary32 (sign 1 / exp 5 / mantissa 10 -> sign 1 / exp 8 /
// mantissa 23). Inline; used for s_world_vertex texcoords.
inline float HalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;                       // signed zero
        } else {
            // Subnormal: normalize.
            // Find the leading 1, shift mantissa, adjust exponent.
            int e = -1;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            uint32_t fexp = (uint32_t)(127 + (-14 + e));
            bits = (sign << 31) | (fexp << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        // Inf or NaN.
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        // Normalized.
        uint32_t fexp = (uint32_t)((int)exp - 15 + 127);
        bits = (sign << 31) | (fexp << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Position decode. The XML's "Float32_4" / "Float32_3" types are NORMALIZED
// floats in [0, 1] (not raw world coordinates) - they're packed exactly the
// way UInt16_N4 was on Xbox 360, just promoted to 32-bit float precision.
// We dequantize via the model-wide BoundingBox[0]:
//     world_x = posMin[0] + raw_x * (posMax[0] - posMin[0])
// Empirically confirmed: first decoded vertex of every format reads in [0, 1]
// before this scaling. Without it, geometry renders at unit scale and looks
// "stretched" relative to actual world dimensions.
//   fmt 0x00/0x01/0x02/0x04/0x05/0x06: Float32_4 normalized at +0x00 (xyz used)
//   fmt 0x0F:                          Float32_3 normalized at +0x00
void DecodeRigidPositions(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                          const float posMin[3], const float posMax[3],
                          uint8_t* dst /* float3[vertexCount] */)
{
    const float scaleX = posMax[0] - posMin[0];
    const float scaleY = posMax[1] - posMin[1];
    const float scaleZ = posMax[2] - posMin[2];
    float* o = reinterpret_cast<float*>(dst);
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const uint8_t* v = src + (size_t)i * stride;
        float x, y, z;
        memcpy(&x, v + 0, 4);
        memcpy(&y, v + 4, 4);
        memcpy(&z, v + 8, 4);
        o[0] = posMin[0] + x * scaleX;
        o[1] = posMin[1] + y * scaleY;
        o[2] = posMin[2] + z * scaleZ;
        o += 3;
    }
}

// UV decode. Per-format:
//   fmt 0x00/0x04 (world):     Float16_2 @ +0x10 (decode via HalfToFloat)
//   fmt 0x01/0x02/0x05/0x06:   UInt16_N2 @ +0x10 (quantize via uvMin/uvMax)
//   fmt 0x0F (decorator):      Float32_2 @ +0x0C (raw floats, no quant)
void DecodeRigidUVs(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                    uint32_t fmt,
                    const float uvMin[2], const float uvMax[2],
                    float* dst /* float2[vertexCount] */)
{
    if (fmt == VFMT_DECORATOR) {
        for (uint32_t i = 0; i < vertexCount; ++i) {
            const uint8_t* v = src + (size_t)i * stride;
            float u, w;
            memcpy(&u, v + 0x0C, 4);
            memcpy(&w, v + 0x10, 4);
            dst[i * 2 + 0] = u;
            dst[i * 2 + 1] = w;
        }
        return;
    }

    if (fmt == VFMT_WORLD || fmt == VFMT_FLAT_WORLD) {
        for (uint32_t i = 0; i < vertexCount; ++i) {
            const uint8_t* v = src + (size_t)i * stride;
            uint16_t hu = RU16(v + 0x10);
            uint16_t hv = RU16(v + 0x12);
            dst[i * 2 + 0] = HalfToFloat(hu);
            dst[i * 2 + 1] = HalfToFloat(hv);
        }
        return;
    }

    // Rigid / skinned (and their flat variants): UInt16_N2 quantized to
    // [uvMin, uvMax].
    float scale[2] = {
        uvMax[0] - uvMin[0],
        uvMax[1] - uvMin[1],
    };
    constexpr float inv65535 = 1.0f / 65535.0f;
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const uint8_t* v = src + (size_t)i * stride;
        uint16_t ru = RU16(v + 0x10);
        uint16_t rv = RU16(v + 0x12);
        dst[i * 2 + 0] = uvMin[0] + ((float)ru * inv65535) * scale[0];
        dst[i * 2 + 1] = uvMin[1] + ((float)rv * inv65535) * scale[1];
    }
}

// -----------------------------------------------------------------------------
// Index strip -> triangle list expansion
//
// IndexFormat enum (Reclaimer):
//   0 Default, 1 LineList, 2 LineStrip, 3 TriangleList, 4 TriangleFan,
//   5 TriangleStrip
// -----------------------------------------------------------------------------

constexpr uint8_t IF_TRI_LIST   = 3;
constexpr uint8_t IF_TRI_STRIP  = 5;
constexpr uint8_t IF_TRI_FAN    = 4;
constexpr uint8_t IF_DEFAULT    = 0;

// Read an index from the source buffer (either uint16 or uint32).
template <typename SrcIdx>
inline uint32_t ReadIdx(const uint8_t* src, uint32_t i) {
    SrcIdx v;
    memcpy(&v, src + i * sizeof(SrcIdx), sizeof(SrcIdx));
    return (uint32_t)v;
}

// Expand a triangle strip to a triangle list. Output is written as the same
// index width as the input.
//
// Honors the strip RESTART sentinel: 0xFFFF for uint16 strips, 0xFFFFFFFF for
// uint32 strips. When the sentinel appears, the current strip ENDS and a new
// strip BEGINS at the next index (the sentinel itself is not a vertex). The
// triangle parity (winding) resets at each restart.
//
// Without restart handling we'd emit triangles like (last_a, 0xFFFF, first_b)
// which the viewer remaps to vertex 0 (OOB), producing the characteristic
// "spider web" triangles spanning each section back to vertex 0.
template <typename Idx>
size_t StripToList(const uint8_t* src, uint32_t srcCount, uint8_t* dst) {
    if (srcCount < 3) return 0;
    constexpr Idx kRestart = (Idx)~(Idx)0;  // 0xFFFF or 0xFFFFFFFF
    uint32_t outCount = 0;
    uint32_t stripStart = 0;
    for (uint32_t i = 0; i + 2 < srcCount; ++i) {
        Idx a, b, c;
        memcpy(&a, src + (i + 0) * sizeof(Idx), sizeof(Idx));
        memcpy(&b, src + (i + 1) * sizeof(Idx), sizeof(Idx));
        memcpy(&c, src + (i + 2) * sizeof(Idx), sizeof(Idx));
        // If any of the three indices is the restart sentinel, skip this
        // window entirely. The next strip resumes from the index just past
        // the sentinel; reset stripStart so winding parity is correct.
        if (a == kRestart || b == kRestart || c == kRestart) {
            // Advance i so the loop continues past the sentinel.
            // The earliest position we can next form a valid triangle is the
            // index right after the sentinel.
            uint32_t skipTo;
            if (a == kRestart)      skipTo = i + 1;
            else if (b == kRestart) skipTo = i + 2;
            else                    skipTo = i + 3;
            i = skipTo - 1;     // -1 because the for-loop will ++i
            stripStart = skipTo;
            continue;
        }
        if (a == b || b == c || a == c) continue;  // degenerate
        // Alternate winding within the current strip to maintain CCW.
        bool flip = ((i - stripStart) & 1) != 0;
        if (!flip) {
            memcpy(dst + (outCount + 0) * sizeof(Idx), &a, sizeof(Idx));
            memcpy(dst + (outCount + 1) * sizeof(Idx), &b, sizeof(Idx));
            memcpy(dst + (outCount + 2) * sizeof(Idx), &c, sizeof(Idx));
        } else {
            memcpy(dst + (outCount + 0) * sizeof(Idx), &a, sizeof(Idx));
            memcpy(dst + (outCount + 1) * sizeof(Idx), &c, sizeof(Idx));
            memcpy(dst + (outCount + 2) * sizeof(Idx), &b, sizeof(Idx));
        }
        outCount += 3;
    }
    return outCount;
}

// Decode the index buffer into a malloc'd uint16/uint32 triangle list.
// Returns nullptr on failure. *outCount = element count, *outStride = bytes
// per index (2 or 4).
//
// For IF_TRI_STRIP the strip is expanded PER SUBMESH (not whole-section).
// Reclaimer keeps strips raw and expands at draw time using each MeshSegment's
// (IndexStart, IndexLength) range; expanding the entire section's strip in one
// shot creates stitching triangles between submeshes - the "spider web"
// artifact on multi-submesh models. For each submesh we slice the strip at
// its (indexStart, indexLength), run StripToList independently (parity resets
// at slice start, just like a fresh strip), then rewrite the submesh's
// (indexStart, indexLength) to be LIST-RELATIVE so downstream draws no longer
// need to know about strips.
//
// `model` is mutated: each submesh whose sectionIndex == sec's index has its
// indexStart / indexLength rewritten in place. Caller is `DecodeGeometryInner`
// which already holds the model handle.
uint8_t* DecodeIndices(ModelData* model, uint32_t sectionIndex,
                       const uint8_t* indexSrc,
                       uint32_t* outCount, uint32_t* outStride)
{
    *outCount  = 0;
    *outStride = 0;
    const ModelSection& sec = model->sections[sectionIndex];
    uint32_t srcStride = (sec.vertexCount > 0xFFFF) ? 4u : 2u;
    *outStride = srcStride;

    if (sec.isUnindexed) {
        // Synthesize indices [0..vertexCount).
        // Reclaimer uses the section's IndexFormat to decide grouping; for
        // unindexed sections this is usually TriangleList (3) so we just emit
        // 0,1,2,3,4,5,...
        size_t outBytes = (size_t)sec.vertexCount * srcStride;
        uint8_t* buf = (uint8_t*)malloc(outBytes);
        if (!buf) return nullptr;
        if (srcStride == 2) {
            for (uint32_t i = 0; i < sec.vertexCount; ++i) {
                uint16_t v = (uint16_t)i;
                memcpy(buf + i * 2, &v, 2);
            }
        } else {
            for (uint32_t i = 0; i < sec.vertexCount; ++i) {
                memcpy(buf + i * 4, &i, 4);
            }
        }
        *outCount = sec.vertexCount;
        return buf;
    }

    uint32_t srcCount = sec.indexCount;  // pre-strip-expansion count
    if (srcCount == 0 || !indexSrc) return nullptr;

    uint8_t fmt = sec.indexFormat;
    if (fmt == IF_DEFAULT) fmt = IF_TRI_STRIP; // Reach default = strip

    if (fmt == IF_TRI_LIST) {
        // Pass-through copy. Submesh ranges already list-relative - no
        // rewrite needed.
        uint32_t outBytes = srcCount * srcStride;
        uint8_t* buf = (uint8_t*)malloc(outBytes);
        if (!buf) return nullptr;
        memcpy(buf, indexSrc, outBytes);
        *outCount = srcCount;
        return buf;
    }

    if (fmt == IF_TRI_STRIP) {
        if (srcCount < 3) return nullptr;
        // Worst-case output bound: per-slice expansion can never produce more
        // triangles than (sliceLen - 2) * 3, summed over slices that's at most
        // (srcCount - 2 * submeshCount) * 3 <= (srcCount * 3). Allocate the
        // loose bound and shrink later - much simpler than a two-pass count.
        uint32_t maxOut = srcCount * 3;
        uint8_t* buf = (uint8_t*)malloc((size_t)maxOut * srcStride);
        if (!buf) return nullptr;

        uint32_t outIdx = 0;  // running output index count

        // No submeshes: fall back to whole-section expansion (preserves the
        // pre-fix behaviour for the rare degenerate case).
        if (sec.submeshCount == 0) {
            size_t produced;
            if (srcStride == 2)
                produced = StripToList<uint16_t>(indexSrc, srcCount, buf);
            else
                produced = StripToList<uint32_t>(indexSrc, srcCount, buf);
            *outCount = (uint32_t)produced;
            return buf;
        }

        // Per-submesh expansion. We mutate model->submeshes in place so
        // ZH_MMP_GetSubmesh subsequently returns list-relative ranges.
        for (uint32_t si = 0; si < sec.submeshCount; ++si) {
            uint32_t smIndex = sec.submeshStart + si;
            ModelSubmesh& sm = model->submeshes[smIndex];

            uint32_t sliceStart = sm.indexStart;
            uint32_t sliceLen   = sm.indexLength;

            // Bounds-check the strip slice. Out-of-range submeshes get
            // zeroed - better than crashing and lets the rest of the
            // model render.
            if (sliceStart >= srcCount || sliceLen < 3 ||
                sliceStart + sliceLen > srcCount)
            {
                sm.indexStart  = outIdx;
                sm.indexLength = 0;
                continue;
            }

            const uint8_t* sliceSrc = indexSrc + (size_t)sliceStart * srcStride;
            size_t produced;
            if (srcStride == 2)
                produced = StripToList<uint16_t>(sliceSrc, sliceLen,
                                                 buf + (size_t)outIdx * srcStride);
            else
                produced = StripToList<uint32_t>(sliceSrc, sliceLen,
                                                 buf + (size_t)outIdx * srcStride);

            sm.indexStart  = outIdx;
            sm.indexLength = (uint32_t)produced;
            outIdx += (uint32_t)produced;
        }

        *outCount = outIdx;
        return buf;
    }

    // Other formats not supported - let the caller fall back.
    return nullptr;
}

// -----------------------------------------------------------------------------
// Per-section decode helpers
// -----------------------------------------------------------------------------

bool DecodeGeometryInner(ModelData* model, uint32_t sectionIndex,
                         uint8_t** outVertexBytes, uint32_t* outVertexLen,
                         uint8_t** outIndexBytes,  uint32_t* outIndexLen)
{
    if (sectionIndex >= model->sections.size()) return false;
    const ModelSection& sec = model->sections[sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;

    // Validate the resource page offsets land inside the decompressed page.
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > model->resourceSize)
        return false;
    const uint8_t* vbSrc = model->resourceData + sec.vbResourceOffset;

    const uint8_t* ibSrc = nullptr;
    if (!sec.isUnindexed) {
        if ((size_t)sec.ibResourceOffset + (size_t)sec.ibDataLength > model->resourceSize)
            return false;
        ibSrc = model->resourceData + sec.ibResourceOffset;
    }

    // Validate that the per-format stride doesn't run past the buffer end.
    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) {
        // gestalt's data length disagrees with the format's expected stride - 
        // the section probably uses a layout we can't decode safely.
        return false;
    }

    // Decode positions to packed float3.
    size_t vertexBytes = (size_t)sec.vertexCount * 12;
    uint8_t* vBuf = (uint8_t*)malloc(vertexBytes);
    if (!vBuf) return false;
    DecodeRigidPositions(vbSrc, sec.vertexCount, stride, sec.posMin, sec.posMax, vBuf);

    // Bone transforms are NOT baked into vertices. Reclaimer's
    // HaloReachCommon.GetMeshes (line 180) just stores BoneIndex on the mesh
    // and lets the renderer apply per-bone transforms separately. Baking at
    // parse time gives the wrong result whenever the bone has a non-trivial
    // rotation, because the per-instance world transform we apply afterward
    // doesn't compose correctly with a pre-baked rotation. The earlier
    // parent-chain bake was the wrong model and produced visibly tilted forge
    // platforms / bridges.
    //
    // For static rigid forge content most bones are identity anyway, so the
    // visible difference is small except on the few tags whose mesh hangs off
    // a non-root bone with rotation. If those need the bone transform later,
    // we should surface the per-section node index + per-node quat/pos via
    // ZH_MMP_GetSection and apply it in the viewer as a
    // per-mesh Transform3DGroup, not as a vertex bake.

    // One-time-per-format diagnostic so we can verify the new stride/decode
    // is sane (positions in world-units, not 1e30 / NaN).
    {
        static std::atomic<uint32_t> s_loggedFormats{ 0 };
        uint32_t mask = 1u << (sec.vertexFormat & 0x1F);
        uint32_t prev = s_loggedFormats.fetch_or(mask, std::memory_order_relaxed);
        if ((prev & mask) == 0 && sec.vertexCount > 0) {
            const float* f = reinterpret_cast<const float*>(vBuf);
            NativeDiag("DecodeVerts: type=0x%02x stride=%u verts=%u first=(%g,%g,%g)",
                sec.vertexFormat, stride, sec.vertexCount,
                f[0], f[1], f[2]);
        }
    }

    // Decode indices (handles strip expansion). For TriStrip this also
    // mutates model->submeshes for the section so their (indexStart,
    // indexLength) become list-relative.
    uint32_t outIndexCount = 0;
    uint32_t outIndexStride = 0;
    uint8_t* iBuf = DecodeIndices(model, sectionIndex, ibSrc,
                                  &outIndexCount, &outIndexStride);
    if (!iBuf) { free(vBuf); return false; }

    *outVertexBytes = vBuf;
    *outVertexLen   = (uint32_t)vertexBytes;
    *outIndexBytes  = iBuf;
    *outIndexLen    = outIndexCount * outIndexStride;
    return true;
}

bool DecodeUVsInner(ModelData* model, uint32_t sectionIndex,
                    float** outUv, uint32_t* outUvFloatCount)
{
    if (sectionIndex >= model->sections.size()) return false;
    const ModelSection& sec = model->sections[sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > model->resourceSize)
        return false;
    const uint8_t* vbSrc = model->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    size_t bytes = (size_t)sec.vertexCount * 2 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    DecodeRigidUVs(vbSrc, sec.vertexCount, stride, sec.vertexFormat,
                   sec.uvMin, sec.uvMax, buf);

    *outUv = buf;
    *outUvFloatCount = sec.vertexCount * 2;
    return true;
}

// Decode per-vertex normals into a malloc'd float3[vertexCount] array.
// Decorator (fmt 0x0F): Float32_3 at +0x14 (raw floats, already unit-ish, same
// encoding the BSP path's DecodeBspNormals uses). Other formats: Int16_N4 at
// +0x14 dequantized by /32767. Added for the decorator-template path so each
// instanced blade carries the render_model's authored normal (HLSL
// decorators.hlsl_include:280, world_normal = quaternion_transform_point(q, vn)).
bool DecodeNormalsInner(ModelData* model, uint32_t sectionIndex,
                        float** outNrm, uint32_t* outNrmFloatCount)
{
    if (sectionIndex >= model->sections.size()) return false;
    const ModelSection& sec = model->sections[sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > model->resourceSize)
        return false;
    const uint8_t* vbSrc = model->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    size_t bytes = (size_t)sec.vertexCount * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;

    constexpr float kInv32767 = 1.0f / 32767.0f;
    if (sec.vertexFormat == VFMT_DECORATOR) {
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            const uint8_t* v = vbSrc + (size_t)i * stride;
            float nx, ny, nz;
            memcpy(&nx, v + 0x14, 4);
            memcpy(&ny, v + 0x18, 4);
            memcpy(&nz, v + 0x1C, 4);
            buf[i*3+0] = nx; buf[i*3+1] = ny; buf[i*3+2] = nz;
        }
    } else {
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            const uint8_t* v = vbSrc + (size_t)i * stride;
            int16_t nx = (int16_t)RU16(v + 0x14);
            int16_t ny = (int16_t)RU16(v + 0x16);
            int16_t nz = (int16_t)RU16(v + 0x18);
            buf[i*3+0] = (float)nx * kInv32767;
            buf[i*3+1] = (float)ny * kInv32767;
            buf[i*3+2] = (float)nz * kInv32767;
        }
    }
    *outNrm = buf;
    *outNrmFloatCount = sec.vertexCount * 3;
    return true;
}

// Decode the per-vertex COLOR stream (vb[vertexBufferIndex+1], float3 stride 12,
// values in [0,1]) into a malloc'd float3[vertexCount]. Populated only for vertex-gradient sky
// domes (sec.colorDataLength != 0). Returns false when the section has no color stream.
bool DecodeColorsInner(ModelData* model, uint32_t sectionIndex,
                       float** outCol, uint32_t* outColFloatCount)
{
    if (sectionIndex >= model->sections.size()) return false;
    const ModelSection& sec = model->sections[sectionIndex];
    if (sec.colorDataLength == 0 || sec.colorStride < 12) return false;
    if (sec.vertexCount == 0) return false;
    if ((size_t)sec.colorResourceOffset + (size_t)sec.vertexCount * sec.colorStride
        > model->resourceSize) return false;
    const uint8_t* src = model->resourceData + sec.colorResourceOffset;

    size_t bytes = (size_t)sec.vertexCount * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    for (uint32_t i = 0; i < sec.vertexCount; ++i) {
        const uint8_t* v = src + (size_t)i * sec.colorStride;
        memcpy(buf + i * 3, v, 12); // float3 RGB in [0,1]
    }
    *outCol = buf;
    *outColFloatCount = sec.vertexCount * 3;
    return true;
}

// -----------------------------------------------------------------------------
// Diffuse bitmap resolver
//
// Walk: shaders[shaderIndex].shaderTagId  (a shader / rmsh / rmtr / rmcs ... tag)
//   -> shader.ShaderProperties[0].ShaderMaps[*].BitmapReference -> first bitm
//      tag found across the maps array.
//
// Reclaimer's "correct" path uses render_method_template.Usages to map each
// ShaderMap by index to a usage StringId ("base_map" / "bump_map" / ...).
// Picking the first valid bitm map approximates "diffuse" well enough for the
// common rmsh / rmtr / rmcs cases that drive forge tags - diffuse is index 0
// of ShaderMaps[] on those templates. We can layer the StringId-driven
// lookup on top later (it requires a global string-id table parse which is
// not yet wired here).
//
// shader.cs offsets:
//   ShaderProperties[] @ +56 (BlockCollection<ShaderPropertiesBlock 172B>)
//   ShaderMaps[]       @ +16 in props (BlockCollection<ShaderMapBlock 24B>)
//   BitmapReference    @ +0 in map (TagReference 16B; tag id @ +12)
//
// First 4 resolutions per session emit a diagnostic line so the chain can be
// audited at a glance:
//   Shader[N]: tagId=0xXX class=rmsh propsOk=1 maps=K firstBmpId=0xZZ class=bitm
// -----------------------------------------------------------------------------

constexpr int OFF_SHADER_PROPS         = 56;
constexpr int SHADER_PROPS_BLOCK_SIZE  = 172;
constexpr int OFF_SHADER_MAPS_IN_PROPS = 16;
constexpr int SHADER_MAP_BLOCK_SIZE    = 24;

// rmt! (render_method_template) layout - see Reclaimer
// HaloReach/render_method_template.cs:
//   Arguments[]  @ +72 (BlockCollection<StringId>)
//   Usages[]     @ +108 (BlockCollection<StringId>)
constexpr int OFF_RMT_USAGES           = 108;
constexpr int STRINGID_BLOCK_SIZE      = 4;

// Surface-color usage names, ranked by priority. Mirrors Reclaimer's actual
// rendering pick at Reclaimer/Controls/DirectX/TextureLoader.cs:30-31:
//
//   diffuse = TextureMappings.FirstOrDefault(Usage == TextureUsage.Diffuse)
//          ?? TextureMappings.FirstOrDefault(Usage == TextureUsage.ColorChange);
//
// Combined with the UsageLookup at Gen3Constants.cs:25-40 that gives us:
//   base_map, alpha_mask_map, foam_texture       -> TextureUsage.Diffuse  (rank 0)
//   change_color_map                              -> TextureUsage.ColorChange (rank 1, fallback)
//
// "Diffuse" priority: pick the first slot whose usage maps to TextureUsage.Diffuse.
// If none of those exist, fall back to "change_color_map" (rank 1) - typical
// for armor / characters / vehicles whose surface color is the change-color
// base. Everything else (bump_map, specular_map, detail_map, self_illum_map,
// alpha_test_map, environment_map) is a different role and MUST NOT be picked.
//
// Terrain blend channels (rmtr) carry usages like "base_map_m_0..3" - 
// Reclaimer strips the `_m_<n>` suffix via the regex `^(\w+?)(?:_m_(\d))?$`.
// MatchDiffuseUsageRank() does the same.
struct DiffuseUsageEntry {
    const char* name;
    int         rank;   // 0 = TextureUsage.Diffuse, 1 = ColorChange fallback
};
static const DiffuseUsageEntry kDiffuseUsages[] = {
    { "base_map",         0 },
    { "alpha_mask_map",   0 },
    { "foam_texture",     0 },
    { "change_color_map", 1 },
};

// Strip the optional Reclaimer-spec `_m_<digit>` suffix. Returns the
// stripped length; if no suffix is present, returns the original length.
// Caller compares the first `outLen` chars against the diffuse list.
static size_t StripUsageBlendSuffix(const char* s, size_t len) {
    // Suffix form is `_m_<single-digit>` (regex group 2 is `\d`, no `+` or `*`).
    if (len < 4) return len;
    if (s[len - 4] != '_' || s[len - 3] != 'm' || s[len - 2] != '_') return len;
    char d = s[len - 1];
    if (d < '0' || d > '9') return len;
    return len - 4;
}

// Returns Reclaimer's rendering rank: 0 = TextureUsage.Diffuse, 1 = ColorChange
// fallback, -1 = not a surface-color slot (skip). Lower rank wins.
static int MatchDiffuseUsageRank(const char* usageName) {
    if (!usageName) return -1;
    size_t len = strlen(usageName);
    size_t stripped = StripUsageBlendSuffix(usageName, len);
    for (int i = 0; i < (int)(sizeof(kDiffuseUsages) / sizeof(kDiffuseUsages[0])); ++i) {
        const char* candidate = kDiffuseUsages[i].name;
        size_t candLen = strlen(candidate);
        if (candLen != stripped) continue;
        if (memcmp(usageName, candidate, candLen) == 0) return kDiffuseUsages[i].rank;
    }
    return -1;
}

// Substrings that strongly indicate a non-diffuse role. If a candidate
// bitmap's tag name contains any of these we skip it. Substring match is
// case-insensitive.
static const char* const kNonDiffuseSubstrings[] = {
    "_bump",
    "_normal",
    "_norm",
    "_bumpmap",
    "_detail",
    "_spec",
    "_specular",
    "_glow",
    "_self_illum",
    "_blend",
    "_mask",
    "_height",
    "_noise",
    "_alpha",
    "_emissive",
    "_metallic",
};

// Substrings that strongly indicate a diffuse / color role. A candidate
// matching any of these is preferred over the first-valid-bitm fallback,
// even if it's not in slot 0.
static const char* const kDiffuseSubstrings[] = {
    "_diff",
    "_diffuse",
    "_color",
    "_albedo",
    "_base",
};

static bool ContainsLowerSubstring(const std::string& haystack, const char* needle) {
    if (!needle || !*needle) return false;
    size_t hlen = haystack.size();
    size_t nlen = strlen(needle);
    if (nlen == 0 || hlen < nlen) return false;
    for (size_t i = 0; i + nlen <= hlen; ++i) {
        bool match = true;
        for (size_t j = 0; j < nlen; ++j) {
            char hc = haystack[i + j];
            if (hc >= 'A' && hc <= 'Z') hc = (char)(hc - 'A' + 'a');
            char nc = needle[j];
            if (nc >= 'A' && nc <= 'Z') nc = (char)(nc - 'A' + 'a');
            if (hc != nc) { match = false; break; }
        }
        if (match) return true;
    }
    return false;
}

static bool BitmapNameSuggestsNonDiffuse(const std::string& tagName) {
    for (const char* s : kNonDiffuseSubstrings) {
        if (ContainsLowerSubstring(tagName, s)) return true;
    }
    return false;
}

static bool BitmapNameSuggestsDiffuse(const std::string& tagName) {
    for (const char* s : kDiffuseSubstrings) {
        if (ContainsLowerSubstring(tagName, s)) return true;
    }
    return false;
}

static std::atomic<int> g_shaderDiagBudget{ 2048 };  // large enough to log every shader of a vehicle (falcon: 21 materials)
static bool ShouldLogShaderDiag() {
    int v = g_shaderDiagBudget.load(std::memory_order_relaxed);
    while (v > 0) {
        if (g_shaderDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed))
            return true;
    }
    return false;
}

uint32_t ResolveDiffuseBitmapTagId(CacheHandle* cache, int32_t shaderIndex,
                                   int32_t shaderTagId)
{
    bool logThis = ShouldLogShaderDiag();

    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) {
        if (logThis)
            NativeDiag("Shader[%d]: tagId=0x%x OOB tagsCount=%llu",
                shaderIndex, shaderTagId, (unsigned long long)cache->tags.size());
        return 0xFFFFFFFFu;
    }
    const TagEntry& te = cache->tags[shaderTagId];
    char shClass[5] = {0};
    memcpy(shClass, te.classCode, 4);

    if (te.classIndex < 0) {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x classIndex<0",
            shaderIndex, shaderTagId);
        return 0xFFFFFFFFu;
    }
    // Shader class is rmsh / rmtr / rmcs / rmd ... - all share the layout we
    // walk. Only require the class starts with 'rm'.
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x class='%s' not rm**",
            shaderIndex, shaderTagId, shClass);
        return 0xFFFFFFFFu;
    }

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size) {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x class=%s bad meta off",
            shaderIndex, shaderTagId, shClass);
        return 0xFFFFFFFFu;
    }
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x class=%s no props (count=%d)",
            shaderIndex, shaderTagId, shClass, propsBlk.count);
        return 0xFFFFFFFFu;
    }

    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 ||
        (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size)
    {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x class=%s bad props off",
            shaderIndex, shaderTagId, shClass);
        return 0xFFFFFFFFu;
    }
    const uint8_t* props = cache->base + propsOff;  // ShaderProperties[0]

    TagBlockRef mapsBlk = ReadTagBlock(props + OFF_SHADER_MAPS_IN_PROPS);
    if (mapsBlk.count <= 0) {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x class=%s no shaderMaps (count=%d)",
            shaderIndex, shaderTagId, shClass, mapsBlk.count);
        return 0xFFFFFFFFu;
    }

    int64_t mapsOff = TagMetaFileOff(cache, mapsBlk.pointer);
    if (mapsOff < 0 ||
        (size_t)mapsOff + (size_t)mapsBlk.count * SHADER_MAP_BLOCK_SIZE > cache->size)
    {
        if (logThis) NativeDiag("Shader[%d]: tagId=0x%x class=%s bad shaderMaps off",
            shaderIndex, shaderTagId, shClass);
        return 0xFFFFFFFFu;
    }

    // -------------------------------------------------------------------
    // Step 1: try the rmt!-driven StringId path. Resolve the shader's
    // ShaderProperties[0].TemplateReference (rmt!) -> Usages[] of StringIds.
    // For each ShaderMap[i], the matching usage at index i tells us what
    // texture role that slot plays ("base_map", "bump_map", etc.). We pick
    // the slot whose usage is in kDiffuseUsageNames and whose bitmap
    // reference resolves to a valid bitm tag.
    //
    // Falls through to the heuristic (first valid bitm) if any link breaks
    // (no rmt!, OOB Usages, string id unresolvable, etc.).
    // -------------------------------------------------------------------
    int rmtBestSlot = -1;
    // bestRank is the Reclaimer-style rank from MatchDiffuseUsageRank: 0 =
    // Diffuse, 1 = ColorChange fallback. Sentinel = max + 1 so any real match
    // wins. -1 from MatchDiffuseUsageRank means "skip", not "best so far".
    int rmtBestRank = (int)(sizeof(kDiffuseUsages) / sizeof(kDiffuseUsages[0]));
    char rmtBestUsage[32] = {0};
    // Captured for the fallback diag so we can read which usages a shader
    // declared when no name matched (helps grow kDiffuseUsages).
    char rmtUsagesDump[256] = {0};
    int  rmtUsagesCount = 0;
    {
        // ShaderProperties[0].TemplateReference is at +0 (the props block
        // starts with the 16-byte TagReference). TagId @ +12, masked to
        // low 16 bits - same pattern as ShaderMap below.
        int32_t rmtRawId = R32(props + 12);
        int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                           ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
        if (logThis) {
            const char* foundClass = "<oob>";
            char foundClassBuf[5] = {0};
            if (rmtTagId >= 0 && (uint32_t)rmtTagId < cache->tags.size()) {
                memcpy(foundClassBuf, cache->tags[rmtTagId].classCode, 4);
                foundClass = foundClassBuf;
            }
            NativeDiag("Shader[%d]: rmt-probe rawId=0x%x rmtTagId=0x%x class='%s' strParsed=%d",
                shaderIndex, (unsigned)rmtRawId, rmtTagId, foundClass,
                (int)cache->stringTableParsed);
        }
        if (rmtTagId >= 0 && (uint32_t)rmtTagId < cache->tags.size() &&
            cache->stringTableParsed)
        {
            const TagEntry& rmtTe = cache->tags[rmtTagId];
            // Accept any class that starts with "rmt" - older Reach uses "rmt!"
            // but U13 MCC uses "rmt2". Both have the same render_method_template
            // layout (Usages BlockCollection at offset 108).
            if (rmtTe.classIndex >= 0 &&
                rmtTe.classCode[0] == 'r' &&
                rmtTe.classCode[1] == 'm' &&
                rmtTe.classCode[2] == 't')
            {
                int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
                if (rmtOff >= 0 &&
                    (size_t)rmtOff + OFF_RMT_USAGES + 8 <= cache->size)
                {
                    const uint8_t* rmtMeta = cache->base + rmtOff;

                    // DIAG: scan rmt2's first 256 bytes for plausible
                    // BlockCollections (count + pointer pairs). Look for one
                    // whose count matches mapsBlk.count and whose first
                    // StringId resolves to "base_map" - that's the real Usages
                    // offset for U13 rmt2 (which differs from "rmt!"+108).
                    if (logThis) {
                        char scan[512] = {0};
                        size_t used = 0;
                        for (int o = 0; o + 8 <= 256; o += 4) {
                            int32_t cnt = R32(rmtMeta + o);
                            uint32_t ptr = RU32(rmtMeta + o + 4);
                            if (cnt <= 0 || cnt > 64) continue;
                            int64_t off = TagMetaFileOff(cache, ptr);
                            if (off < 0 || (size_t)off + 4 > cache->size) continue;
                            int32_t firstSid = R32(cache->base + off);
                            const char* s = ResolveStringId(cache, firstSid);
                            if (!s || !*s) continue;
                            int n = _snprintf_s(scan + used, sizeof(scan) - used,
                                _TRUNCATE, "[+%d c=%d:'%s'] ", o, cnt, s);
                            if (n > 0) used += (size_t)n;
                            if (used > sizeof(scan) - 64) break;
                        }
                        NativeDiag("Shader[%d]: rmt-scan tagId=0x%x mapsCnt=%d  %s",
                            shaderIndex, rmtTagId, mapsBlk.count, scan);
                    }

                    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
                    if (usagesBlk.count > 0 && usagesBlk.count <= 0x1000) {
                        int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
                        if (usagesOff >= 0 &&
                            (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE
                                <= cache->size)
                        {
                            int slotMax = mapsBlk.count;
                            if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;
                            rmtUsagesCount = slotMax;
                            size_t dumpUsed = 0;
                            for (int i = 0; i < slotMax; ++i) {
                                int32_t sid = R32(cache->base + usagesOff +
                                                  (size_t)i * STRINGID_BLOCK_SIZE);
                                const char* usageName = ResolveStringId(cache, sid);
                                // Capture into the comma-separated dump (truncate-safe).
                                if (logThis && dumpUsed + 32 < sizeof(rmtUsagesDump)) {
                                    int n = _snprintf_s(
                                        rmtUsagesDump + dumpUsed,
                                        sizeof(rmtUsagesDump) - dumpUsed,
                                        _TRUNCATE,
                                        "%s%s",
                                        dumpUsed == 0 ? "" : ",",
                                        (usageName && *usageName) ? usageName : "<?>");
                                    if (n > 0) dumpUsed += (size_t)n;
                                }
                                if (!usageName || !*usageName) continue;
                                // MatchDiffuseUsageRank handles the `_m_<n>`
                                // suffix strip per Reclaimer's UsageRegex. So
                                // `base_map_m_2` correctly matches `base_map`.
                                int rank = MatchDiffuseUsageRank(usageName);
                                if (rank < 0 || rank >= rmtBestRank) continue;
                                // Verify this slot actually points at a valid bitm.
                                const uint8_t* mapEntry = cache->base + mapsOff +
                                                         (size_t)i * SHADER_MAP_BLOCK_SIZE;
                                int32_t rawId = R32(mapEntry + 12);
                                if ((uint32_t)rawId == 0xFFFFFFFFu) continue;
                                uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
                                if (bmpId >= cache->tags.size()) continue;
                                if (memcmp(cache->tags[bmpId].classCode, "bitm", 4) != 0) continue;
                                rmtBestRank = rank;
                                rmtBestSlot = i;
                                strncpy_s(rmtBestUsage, usageName, _TRUNCATE);
                                if (rank == 0) break;  // "base_map" - best possible
                            }
                        }
                    }
                }
            }
        }
    }

    if (rmtBestSlot >= 0) {
        const uint8_t* mapEntry = cache->base + mapsOff +
                                  (size_t)rmtBestSlot * SHADER_MAP_BLOCK_SIZE;
        int32_t rawId = R32(mapEntry + 12);
        uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
        if (logThis) {
            // Show the bitmap's tag name AND a per-slot dump (usage + bitmap
            // name pair) for ALL slots. Lets us confirm at a glance that
            // slot N's bitmap really is the one the usage name suggests
            // (e.g. usage='base_map' -> bitmap ends with '_diff', not '_bump').
            const char* pickedName = (bmpId < cache->tags.size())
                ? cache->tags[bmpId].tagName.c_str() : "?";
            char slotDump[768] = {0};
            size_t sdUsed = 0;
            for (int i = 0; i < rmtUsagesCount; ++i) {
                const uint8_t* mEnt = cache->base + mapsOff + (size_t)i * SHADER_MAP_BLOCK_SIZE;
                int32_t mRaw = R32(mEnt + 12);
                uint32_t mBmp = ((uint32_t)mRaw == 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)mRaw & 0xFFFFu;
                const char* mName = (mBmp < cache->tags.size() &&
                    memcmp(cache->tags[mBmp].classCode, "bitm", 4) == 0)
                    ? cache->tags[mBmp].tagName.c_str() : "<no-bitm>";
                // shorten name to last path component for readability
                const char* shortName = mName;
                for (const char* p = mName; *p; ++p)
                    if (*p == '\\' || *p == '/') shortName = p + 1;
                if (sdUsed + 64 < sizeof(slotDump)) {
                    int n = _snprintf_s(slotDump + sdUsed, sizeof(slotDump) - sdUsed,
                        _TRUNCATE, "%s[%d]=%s", sdUsed == 0 ? "" : " ", i, shortName);
                    if (n > 0) sdUsed += (size_t)n;
                }
            }
            NativeDiag("Shader[%d]: tagId=0x%x class=%s rmt-match slot=%d usage='%s' bmpId=0x%x picked='%s' usages=[%s] slots=%s",
                shaderIndex, shaderTagId, shClass, rmtBestSlot,
                rmtBestUsage, bmpId, pickedName, rmtUsagesDump, slotDump);
        }
        return bmpId;
    }

    // -------------------------------------------------------------------
    // Step 2 (fallback): tightened heuristic picking the best non-diffuse
    // bitmap from ShaderMaps[].
    //
    // Priority order:
    //   (A) any slot whose BITMAP TAG NAME ends with a diffuse-y substring
    //       (_diff, _diffuse, _color, _albedo, _base) - strongest signal.
    //   (B) the FIRST slot whose bitmap name does NOT match a non-diffuse
    //       substring (_bump, _normal, _spec, _glow, _detail, ...).
    //   (C) failing both, the original behaviour: first valid bitm in slot
    //       order - better than nothing for shaders whose maps don't follow
    //       the naming convention.
    //
    // Iterate once, classify each slot, then choose. This avoids the
    // pathological case where slot 0 is a normal map and slot 1+ is the
    // diffuse (common in custom Forge shaders where the rmt!.Usages don't
    // match any of our known names).
    // -------------------------------------------------------------------
    int      diffuseNamedSlot = -1;
    int      neutralSlot      = -1;  // first slot that's not non-diffuse-named
    int      anyValidSlot     = -1;  // first valid bitm slot (legacy fallback)
    uint32_t diffuseNamedBmp  = 0xFFFFFFFFu;
    uint32_t neutralBmp       = 0xFFFFFFFFu;
    uint32_t anyValidBmp      = 0xFFFFFFFFu;
    char     anyValidClass[5] = {0};
    for (int i = 0; i < mapsBlk.count; ++i) {
        const uint8_t* mapEntry = cache->base + mapsOff +
                                  (size_t)i * SHADER_MAP_BLOCK_SIZE;
        // BitmapReference's tagId field at +12 is a 32-bit Reach tag identifier
        // where the high bits are engine identity/generation and the low 16
        // bits are the index. Reclaimer's TagReference.TagId is
        // (short)(tagId & ushort.MaxValue) - null only if the FULL 32-bit value
        // is 0xFFFFFFFF.
        int32_t rawId = R32(mapEntry + 12);
        if ((uint32_t)rawId == 0xFFFFFFFFu) continue;  // genuine null
        uint32_t bitmapTagId = (uint32_t)rawId & 0xFFFFu;
        if (bitmapTagId >= cache->tags.size()) continue;
        if (memcmp(cache->tags[bitmapTagId].classCode, "bitm", 4) != 0)
            continue;
        if (anyValidSlot < 0) {
            anyValidSlot = i;
            anyValidBmp  = bitmapTagId;
            memcpy(anyValidClass, cache->tags[bitmapTagId].classCode, 4);
        }
        const std::string& bmpName = cache->tags[bitmapTagId].tagName;
        bool nonDiffuse = BitmapNameSuggestsNonDiffuse(bmpName);
        bool diffuseHit = !nonDiffuse && BitmapNameSuggestsDiffuse(bmpName);
        if (diffuseHit && diffuseNamedSlot < 0) {
            diffuseNamedSlot = i;
            diffuseNamedBmp  = bitmapTagId;
        }
        if (!nonDiffuse && neutralSlot < 0) {
            neutralSlot = i;
            neutralBmp  = bitmapTagId;
        }
    }

    int      pickedSlot = -1;
    uint32_t firstBmpId = 0xFFFFFFFFu;
    char     firstBmpClass[5] = {0};
    const char* pickedReason = "none";
    if (diffuseNamedSlot >= 0) {
        pickedSlot = diffuseNamedSlot;
        firstBmpId = diffuseNamedBmp;
        pickedReason = "diffuse-name";
    } else if (neutralSlot >= 0) {
        pickedSlot = neutralSlot;
        firstBmpId = neutralBmp;
        pickedReason = "non-bumpish";
    } else if (anyValidSlot >= 0) {
        pickedSlot = anyValidSlot;
        firstBmpId = anyValidBmp;
        pickedReason = "any-valid";
    }
    if (firstBmpId != 0xFFFFFFFFu && firstBmpId < cache->tags.size())
        memcpy(firstBmpClass, cache->tags[firstBmpId].classCode, 4);

    if (logThis) {
        const char* bmpName = (firstBmpId != 0xFFFFFFFFu &&
                               firstBmpId < cache->tags.size())
                              ? cache->tags[firstBmpId].tagName.c_str() : "<none>";
        NativeDiag("Shader[%d]: tagId=0x%x class=%s propsOk=1 maps=%d "
                   "rmt-fallback pickedSlot=%d reason=%s firstBmpId=0x%x bmpClass=%s name='%s' "
                   "usages=[%s] (count=%d)",
            shaderIndex, shaderTagId, shClass, mapsBlk.count,
            pickedSlot, pickedReason, firstBmpId,
            firstBmpClass[0] ? firstBmpClass : "<none>",
            bmpName,
            rmtUsagesDump[0] ? rmtUsagesDump : "<no rmt!>",
            rmtUsagesCount);
    }

    return firstBmpId;
}

// -----------------------------------------------------------------------------
// Per-shader blend mode resolution (rmsh + rmdf option-string lookup).
//
// In Halo Reach the per-material blend mode is NOT a single byte on rmt2 - 
// it's encoded via the rmdf shader-options system. The chain:
//
//   rmsh.RenderMethodDefinitionReference (TagReference @ +0)         -> rmdf
//   rmsh.ShaderOptions[]                 (BlockCollection @ +32, 2B)
//     each = ShaderOptionIndexBlock { short OptionIndex; }
//   rmdf.Categories[]                    (BlockCollection @ +16, 24B)
//     each = ShaderOptionCategoryBlock { StringId Name; BlockCollection<28B> Options; }
//   rmdf.Categories[i].Options[j].Name   (StringId @ +0)
//
// Pick the category whose Name resolves to "blend_mode", index into rmsh's
// ShaderOptions[] at the same ordinal to get the option index, then resolve
// rmdf.Categories[catIdx].Options[optIdx].Name to a string. Map the string to
// our 0..5 enum (see SAPIEN_RENDER_PIPELINE_NOTES.md "blend mode -> register
// value (gold)").
//
// This matches Sapien's runtime path (decal_shader_parse_blend_mode_string @
// sapien.exe+0x64EBD0 reads the same string-table path via FUN_140b27c30 a.k.a.
// rmsh_option_category_equals).
//
// Returns 0..5 on success, 0xFF on any failure (caller treats as opaque).
// -----------------------------------------------------------------------------

constexpr int OFF_RMSH_RMDF_REF        = 0;    // TagReference (16B)
constexpr int OFF_RMSH_SHADER_OPTIONS  = 32;   // BlockCollection<2B>
constexpr int RMSH_SHADER_OPTION_SIZE  = 2;    // short OptionIndex
constexpr int OFF_RMDF_CATEGORIES      = 16;   // BlockCollection<24B>
constexpr int RMDF_CATEGORY_SIZE       = 24;
constexpr int OFF_RMDF_CAT_OPTIONS     = 4;    // BlockCollection<28B> inside a Category
constexpr int RMDF_OPTION_SIZE         = 28;

// 0xFF = unknown / fall back to opaque.
// Indices match the viewer's BlendMode enum.
//   0 = opaque, 1 = additive, 2 = multiply, 3 = double_multiply,
//   4 = alpha_blend, 5 = add_src_times_srcalpha
static uint8_t MapBlendModeStringToIndex(const char* s) {
    if (!s || !*s) return 0xFF;
    // Common rmdf option strings; case-sensitive match (rmdf string table is
    // canonical lowercase per Reclaimer/Sapien).
    if (strcmp(s, "opaque") == 0)                       return 0;
    if (strcmp(s, "additive") == 0)                     return 1;
    if (strcmp(s, "multiply") == 0)                     return 2;
    if (strcmp(s, "double_multiply") == 0)              return 3;
    if (strcmp(s, "alpha_blend") == 0)                  return 4;
    if (strcmp(s, "add_src_times_srcalpha") == 0)       return 5;
    // `add_src_times_dstalpha` is a DISTINCT engine mode (Src=DestAlpha,
    // Dst=One) from add_src_times_srcalpha (Src=SrcAlpha); byte 8 routes it
    // to the viewer's AddSrcTimesDstAlpha state.
    if (strcmp(s, "add_src_times_dstalpha") == 0)       return 8;
    // `pre_multiplied_alpha` is a distinct engine blend mode (Src=ONE,
    // Dst=INV_SRC_ALPHA - no double-applied alpha at silhouette edges);
    // collapsing it to AlphaBlend (Src=SRC_ALPHA) darkens silhouettes ~30%.
    // The viewer carries a PreMultipliedAlpha=6 state; sky panels use it too.
    if (strcmp(s, "pre_multiplied_alpha") == 0)         return 6;
    // `maximum` is a per-channel Max blend (BlendOp.Max), NOT double_multiply
    // (collapsing it to 3 makes overlapping `maximum` glows sum-via-modulate
    // instead of taking the max, over-brightening). Route to BlendMode byte 7; the render core's
    // BlendModeId.Maximum (8) sets BlendOperation.Maximum exactly.
    if (strcmp(s, "maximum") == 0)                      return 7;
    return 0xFF;
}

uint8_t ResolveShaderBlendMode(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0xFF;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFF;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFF;
    if (!cache->stringTableParsed) return 0xFF;

    int64_t rmshOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (rmshOff < 0 || (size_t)rmshOff + OFF_SHADER_PROPS + 8 > cache->size) return 0xFF;
    const uint8_t* rmsh = cache->base + rmshOff;

    // Read rmdf TagReference (TagId @ +12 within the 16-byte ref).
    int32_t rmdfRawId = R32(rmsh + OFF_RMSH_RMDF_REF + 12);
    int32_t rmdfTagId = ((uint32_t)rmdfRawId == 0xFFFFFFFFu)
                        ? -1 : (int32_t)((uint32_t)rmdfRawId & 0xFFFFu);
    if (rmdfTagId < 0 || (uint32_t)rmdfTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& rmdfTe = cache->tags[rmdfTagId];
    if (rmdfTe.classIndex < 0) return 0xFF;
    // class is "rmdf" (terrain shader's rmdf is also "rmdf").
    if (rmdfTe.classCode[0] != 'r' || rmdfTe.classCode[1] != 'm' ||
        rmdfTe.classCode[2] != 'd' || rmdfTe.classCode[3] != 'f') return 0xFF;

    // Read rmsh.ShaderOptions[] block.
    TagBlockRef shaderOptsBlk = ReadTagBlock(rmsh + OFF_RMSH_SHADER_OPTIONS);
    if (shaderOptsBlk.count <= 0 || shaderOptsBlk.count > 0x1000) return 0xFF;
    int64_t shaderOptsOff = TagMetaFileOff(cache, shaderOptsBlk.pointer);
    if (shaderOptsOff < 0 ||
        (size_t)shaderOptsOff + (size_t)shaderOptsBlk.count * RMSH_SHADER_OPTION_SIZE > cache->size)
        return 0xFF;

    // Read rmdf.Categories[] block.
    int64_t rmdfMetaOff = TagMetaFileOff(cache, rmdfTe.metaPointerRaw);
    if (rmdfMetaOff < 0 || (size_t)rmdfMetaOff + OFF_RMDF_CATEGORIES + 8 > cache->size) return 0xFF;
    const uint8_t* rmdfMeta = cache->base + rmdfMetaOff;
    TagBlockRef catsBlk = ReadTagBlock(rmdfMeta + OFF_RMDF_CATEGORIES);
    if (catsBlk.count <= 0 || catsBlk.count > 0x1000) return 0xFF;
    int64_t catsOff = TagMetaFileOff(cache, catsBlk.pointer);
    if (catsOff < 0 ||
        (size_t)catsOff + (size_t)catsBlk.count * RMDF_CATEGORY_SIZE > cache->size)
        return 0xFF;

    // Find the "blend_mode" category by string-id resolved name.
    int blendCatIdx = -1;
    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
    for (int ci = 0; ci < catLimit; ++ci) {
        const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
        int32_t catNameSid = R32(catEntry + 0);
        const char* catName = ResolveStringId(cache, catNameSid);
        if (!catName) continue;
        if (strcmp(catName, "blend_mode") == 0) {
            blendCatIdx = ci;
            break;
        }
    }
    if (blendCatIdx < 0) return 0xFF;

    // rmsh.ShaderOptions[blendCatIdx].OptionIndex (short).
    const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                    (size_t)blendCatIdx * RMSH_SHADER_OPTION_SIZE;
    int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                    ((uint16_t)shaderOptEntry[1] << 8));
    if (optionIndex < 0) return 0xFF;

    // Read the picked category's Options[] block (Options @ +4 in 24-byte cat).
    const uint8_t* catEntry = cache->base + catsOff + (size_t)blendCatIdx * RMDF_CATEGORY_SIZE;
    TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
    if (optsBlk.count <= 0 || optsBlk.count > 0x1000) return 0xFF;
    if (optionIndex >= optsBlk.count) return 0xFF;
    int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
    if (optsOff < 0 ||
        (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE > cache->size)
        return 0xFF;

    const uint8_t* optEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
    int32_t optNameSid = R32(optEntry + 0);
    const char* optName = ResolveStringId(cache, optNameSid);
    return MapBlendModeStringToIndex(optName);
}

// -----------------------------------------------------------------------------
// MAT-1: ResolveShaderMaterialModel - the per-shader material-model selector.
// The engine links exactly one calc_material_<model>_ps per shader variant, chosen
// at compile time by the rmsh `material_model` category option
// (material_models.hlsl_include:12-31). `material_model` is a sibling rmdf Category
// to `blend_mode`, so this reuses the identical rmsh -> rmdf categories ->
// rmsh.ShaderOptions[catIdx] -> option Name walk as ResolveShaderBlendMode.
// Returns 0..9 (the MATERIAL_TYPE_* enum) on success, 0xFF on any failure.
//   0 diffuse_only  1 cook_torrance  2 two_lobe_phong  3 foliage  4 none
//   5 glass  6 organism  7 single_lobe_phong  8 hair  9 custom_specular
static uint8_t MapMaterialModelStringToIndex(const char* s) {
    if (!s || !*s) return 0xFF;
    if (strcmp(s, "diffuse_only") == 0)       return 0;
    if (strcmp(s, "cook_torrance") == 0)      return 1;
    if (strcmp(s, "two_lobe_phong") == 0)     return 2;
    if (strcmp(s, "foliage") == 0)            return 3;
    if (strcmp(s, "none") == 0)               return 4;
    if (strcmp(s, "glass") == 0)              return 5;
    if (strcmp(s, "organism") == 0)           return 6;
    if (strcmp(s, "single_lobe_phong") == 0)  return 7;
    if (strcmp(s, "hair") == 0)               return 8;
    if (strcmp(s, "custom_specular") == 0)    return 9;
    return 0xFF;
}

uint8_t ResolveShaderMaterialModel(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0xFF;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFF;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFF;
    if (!cache->stringTableParsed) return 0xFF;

    int64_t rmshOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (rmshOff < 0 || (size_t)rmshOff + OFF_SHADER_PROPS + 8 > cache->size) return 0xFF;
    const uint8_t* rmsh = cache->base + rmshOff;

    int32_t rmdfRawId = R32(rmsh + OFF_RMSH_RMDF_REF + 12);
    int32_t rmdfTagId = ((uint32_t)rmdfRawId == 0xFFFFFFFFu)
                        ? -1 : (int32_t)((uint32_t)rmdfRawId & 0xFFFFu);
    if (rmdfTagId < 0 || (uint32_t)rmdfTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& rmdfTe = cache->tags[rmdfTagId];
    if (rmdfTe.classIndex < 0) return 0xFF;
    if (rmdfTe.classCode[0] != 'r' || rmdfTe.classCode[1] != 'm' ||
        rmdfTe.classCode[2] != 'd' || rmdfTe.classCode[3] != 'f') return 0xFF;

    TagBlockRef shaderOptsBlk = ReadTagBlock(rmsh + OFF_RMSH_SHADER_OPTIONS);
    if (shaderOptsBlk.count <= 0 || shaderOptsBlk.count > 0x1000) return 0xFF;
    int64_t shaderOptsOff = TagMetaFileOff(cache, shaderOptsBlk.pointer);
    if (shaderOptsOff < 0 ||
        (size_t)shaderOptsOff + (size_t)shaderOptsBlk.count * RMSH_SHADER_OPTION_SIZE > cache->size)
        return 0xFF;

    int64_t rmdfMetaOff = TagMetaFileOff(cache, rmdfTe.metaPointerRaw);
    if (rmdfMetaOff < 0 || (size_t)rmdfMetaOff + OFF_RMDF_CATEGORIES + 8 > cache->size) return 0xFF;
    const uint8_t* rmdfMeta = cache->base + rmdfMetaOff;
    TagBlockRef catsBlk = ReadTagBlock(rmdfMeta + OFF_RMDF_CATEGORIES);
    if (catsBlk.count <= 0 || catsBlk.count > 0x1000) return 0xFF;
    int64_t catsOff = TagMetaFileOff(cache, catsBlk.pointer);
    if (catsOff < 0 ||
        (size_t)catsOff + (size_t)catsBlk.count * RMDF_CATEGORY_SIZE > cache->size)
        return 0xFF;

    // Find the "material_model" category by string-id resolved name.
    int mmCatIdx = -1;
    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
    for (int ci = 0; ci < catLimit; ++ci) {
        const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
        int32_t catNameSid = R32(catEntry + 0);
        const char* catName = ResolveStringId(cache, catNameSid);
        if (!catName) continue;
        if (strcmp(catName, "material_model") == 0) { mmCatIdx = ci; break; }
    }
    if (mmCatIdx < 0) return 0xFF;

    const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                    (size_t)mmCatIdx * RMSH_SHADER_OPTION_SIZE;
    int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                    ((uint16_t)shaderOptEntry[1] << 8));
    if (optionIndex < 0) return 0xFF;

    const uint8_t* catEntry = cache->base + catsOff + (size_t)mmCatIdx * RMDF_CATEGORY_SIZE;
    TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
    if (optsBlk.count <= 0 || optsBlk.count > 0x1000) return 0xFF;
    if (optionIndex >= optsBlk.count) return 0xFF;
    int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
    if (optsOff < 0 ||
        (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE > cache->size)
        return 0xFF;

    const uint8_t* optEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
    int32_t optNameSid = R32(optEntry + 0);
    const char* optName = ResolveStringId(cache, optNameSid);
    return MapMaterialModelStringToIndex(optName);
}

// -----------------------------------------------------------------------------
// LIT-SI-3 (model side): ResolveShaderSelfIllumMode - the render-model mirror of
// MapBspParser.cpp::ResolveShaderSelfIllumMode. Identical rmsh -> rmdf categories
// walk as ResolveShaderMaterialModel above, but matches the `self_illumination`
// category and maps the selected option Name to the same self-illum MODE enum the
// BSP side uses (BspSelfIllumModeStringToIndex). The enum + Rust mapping (0xFF -> 1
// simple) are documented at MapBspParser.cpp:3719. Returns 0..12 on success, 0xFF
// on any failure.
static uint8_t MmpSelfIllumModeStringToIndex(const char* s) {
    if (!s || !*s) return 0xFF;
    if (strcmp(s, "off") == 0)                       return 0;
    if (strcmp(s, "simple") == 0)                    return 1;
    if (strcmp(s, "3_channel_self_illum") == 0)      return 2;
    if (strcmp(s, "plasma") == 0)                    return 3;
    if (strcmp(s, "from_diffuse") == 0)              return 4;  // from_albedo
    if (strcmp(s, "illum_detail") == 0)              return 5;
    if (strcmp(s, "meter") == 0)                     return 6;
    if (strcmp(s, "self_illum_times_diffuse") == 0)  return 7;
    if (strcmp(s, "simple_with_alpha_mask") == 0)    return 8;
    if (strcmp(s, "multilayer_additive") == 0)       return 9;
    if (strcmp(s, "palettized_plasma") == 0)         return 10;
    if (strcmp(s, "change_color") == 0)              return 11;
    if (strcmp(s, "change_color_detail") == 0)       return 12;
    return 0xFF;
}

uint8_t ResolveShaderSelfIllumMode(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0xFF;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFF;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFF;
    if (!cache->stringTableParsed) return 0xFF;

    int64_t rmshOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (rmshOff < 0 || (size_t)rmshOff + OFF_SHADER_PROPS + 8 > cache->size) return 0xFF;
    const uint8_t* rmsh = cache->base + rmshOff;

    int32_t rmdfRawId = R32(rmsh + OFF_RMSH_RMDF_REF + 12);
    int32_t rmdfTagId = ((uint32_t)rmdfRawId == 0xFFFFFFFFu)
                        ? -1 : (int32_t)((uint32_t)rmdfRawId & 0xFFFFu);
    if (rmdfTagId < 0 || (uint32_t)rmdfTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& rmdfTe = cache->tags[rmdfTagId];
    if (rmdfTe.classIndex < 0) return 0xFF;
    if (rmdfTe.classCode[0] != 'r' || rmdfTe.classCode[1] != 'm' ||
        rmdfTe.classCode[2] != 'd' || rmdfTe.classCode[3] != 'f') return 0xFF;

    TagBlockRef shaderOptsBlk = ReadTagBlock(rmsh + OFF_RMSH_SHADER_OPTIONS);
    if (shaderOptsBlk.count <= 0 || shaderOptsBlk.count > 0x1000) return 0xFF;
    int64_t shaderOptsOff = TagMetaFileOff(cache, shaderOptsBlk.pointer);
    if (shaderOptsOff < 0 ||
        (size_t)shaderOptsOff + (size_t)shaderOptsBlk.count * RMSH_SHADER_OPTION_SIZE > cache->size)
        return 0xFF;

    int64_t rmdfMetaOff = TagMetaFileOff(cache, rmdfTe.metaPointerRaw);
    if (rmdfMetaOff < 0 || (size_t)rmdfMetaOff + OFF_RMDF_CATEGORIES + 8 > cache->size) return 0xFF;
    const uint8_t* rmdfMeta = cache->base + rmdfMetaOff;
    TagBlockRef catsBlk = ReadTagBlock(rmdfMeta + OFF_RMDF_CATEGORIES);
    if (catsBlk.count <= 0 || catsBlk.count > 0x1000) return 0xFF;
    int64_t catsOff = TagMetaFileOff(cache, catsBlk.pointer);
    if (catsOff < 0 ||
        (size_t)catsOff + (size_t)catsBlk.count * RMDF_CATEGORY_SIZE > cache->size)
        return 0xFF;

    int siCatIdx = -1;
    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
    for (int ci = 0; ci < catLimit; ++ci) {
        const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
        int32_t catNameSid = R32(catEntry + 0);
        const char* catName = ResolveStringId(cache, catNameSid);
        if (!catName) continue;
        if (strcmp(catName, "self_illumination") == 0) { siCatIdx = ci; break; }
    }
    if (siCatIdx < 0) return 0xFF;

    const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                    (size_t)siCatIdx * RMSH_SHADER_OPTION_SIZE;
    int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                    ((uint16_t)shaderOptEntry[1] << 8));
    if (optionIndex < 0) return 0xFF;

    const uint8_t* catEntry = cache->base + catsOff + (size_t)siCatIdx * RMDF_CATEGORY_SIZE;
    TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
    if (optsBlk.count <= 0 || optsBlk.count > 0x1000) return 0xFF;
    if (optionIndex >= optsBlk.count) return 0xFF;
    int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
    if (optsOff < 0 ||
        (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE > cache->size)
        return 0xFF;

    const uint8_t* optEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
    int32_t optNameSid = R32(optEntry + 0);
    const char* optName = ResolveStringId(cache, optNameSid);
    return MmpSelfIllumModeStringToIndex(optName);
}

// -----------------------------------------------------------------------------
// SKY-3: ResolveShaderSkyClass - detect a `sky_dome_simple` sky section.
// The engine renders a sky section whose shader's @generate-sky base template is
// `sky_dome_simple` as `vColor * g_exposure.r` with NO texture sampler
// (sky_dome_simple.hlsl_include). HMS's sky mesh path samples base+emis (a correct
// superset for OTHER cloud/planet templates that DO sample), so we must detect the
// pure-gradient dome to route it to gradient-only.
//
// The rmt2 permutation NAME is numeric (`shaders\shader_templates\_7_...`), so it
// never contains "sky_dome_simple". The @generate-sky identity IS preserved in the
// render_method_definition (rmdf) the sky rmsh references: the rmdf tag NAME is the
// template family (e.g. `...\sky_dome_simple.render_method_definition`). We walk the
// same rmsh->rmdf chain as ResolveShaderMaterialModel and test:
//   (a) rmdfTe.tagName contains "sky_dome_simple", OR
//   (b) any rmdf category OPTION name == "sky_dome_simple"
// Returns 1 = sky_dome_simple, 0 = resolvable-but-other (textured sky template),
// 0xFF = unresolved (Rust falls back to its is_vgradient_dome heuristic).
//
// HMS_SKY3_DIAG (env, stderr) dumps the shader class, rmdf tagName and every
// category+selected-option name so the detection strings can be verified per map.
static uint8_t ResolveShaderSkyClass(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0xFF;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFF;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFF;
    if (!cache->stringTableParsed) return 0xFF;

    const bool diag = (getenv("HMS_SKY3_DIAG") != nullptr);

    int64_t rmshOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (rmshOff < 0 || (size_t)rmshOff + OFF_SHADER_PROPS + 8 > cache->size) return 0xFF;
    const uint8_t* rmsh = cache->base + rmshOff;

    int32_t rmdfRawId = R32(rmsh + OFF_RMSH_RMDF_REF + 12);
    int32_t rmdfTagId = ((uint32_t)rmdfRawId == 0xFFFFFFFFu)
                        ? -1 : (int32_t)((uint32_t)rmdfRawId & 0xFFFFu);
    if (rmdfTagId < 0 || (uint32_t)rmdfTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& rmdfTe = cache->tags[rmdfTagId];
    if (rmdfTe.classIndex < 0) return 0xFF;
    if (rmdfTe.classCode[0] != 'r' || rmdfTe.classCode[1] != 'm' ||
        rmdfTe.classCode[2] != 'd' || rmdfTe.classCode[3] != 'f') return 0xFF;

    const std::string& rmdfName = rmdfTe.tagName;
    bool nameMatch = (rmdfName.find("sky_dome_simple") != std::string::npos);

    if (diag) {
        fprintf(stderr, "HMS_SKY3_DIAG shaderTag=0x%X class=%c%c%c%c rmdf=0x%X rmdfName='%s'\n",
                (unsigned)shaderTagId, te.classCode[0], te.classCode[1], te.classCode[2], te.classCode[3],
                (unsigned)rmdfTagId, rmdfName.c_str());
    }

    // Scan rmdf categories + selected options (for both the diag and option-name match).
    bool optMatch = false;
    do {
        TagBlockRef shaderOptsBlk = ReadTagBlock(rmsh + OFF_RMSH_SHADER_OPTIONS);
        if (shaderOptsBlk.count <= 0 || shaderOptsBlk.count > 0x1000) break;
        int64_t shaderOptsOff = TagMetaFileOff(cache, shaderOptsBlk.pointer);
        if (shaderOptsOff < 0 ||
            (size_t)shaderOptsOff + (size_t)shaderOptsBlk.count * RMSH_SHADER_OPTION_SIZE > cache->size) break;

        int64_t rmdfMetaOff = TagMetaFileOff(cache, rmdfTe.metaPointerRaw);
        if (rmdfMetaOff < 0 || (size_t)rmdfMetaOff + OFF_RMDF_CATEGORIES + 8 > cache->size) break;
        const uint8_t* rmdfMeta = cache->base + rmdfMetaOff;
        TagBlockRef catsBlk = ReadTagBlock(rmdfMeta + OFF_RMDF_CATEGORIES);
        if (catsBlk.count <= 0 || catsBlk.count > 0x1000) break;
        int64_t catsOff = TagMetaFileOff(cache, catsBlk.pointer);
        if (catsOff < 0 ||
            (size_t)catsOff + (size_t)catsBlk.count * RMDF_CATEGORY_SIZE > cache->size) break;

        int catLimit = catsBlk.count;
        if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
        for (int ci = 0; ci < catLimit; ++ci) {
            const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
            int32_t catNameSid = R32(catEntry + 0);
            const char* catName = ResolveStringId(cache, catNameSid);
            const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                            (size_t)ci * RMSH_SHADER_OPTION_SIZE;
            int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                            ((uint16_t)shaderOptEntry[1] << 8));
            const char* optName = nullptr;
            if (optionIndex >= 0) {
                TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
                if (optsBlk.count > 0 && optsBlk.count <= 0x1000 && optionIndex < optsBlk.count) {
                    int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
                    if (optsOff >= 0 &&
                        (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE <= cache->size) {
                        const uint8_t* optEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
                        optName = ResolveStringId(cache, R32(optEntry + 0));
                    }
                }
            }
            if (optName && strcmp(optName, "sky_dome_simple") == 0) optMatch = true;
            if (diag) {
                fprintf(stderr, "  cat[%d]='%s' opt[%d]='%s'\n",
                        ci, catName ? catName : "<?>", (int)optionIndex, optName ? optName : "<?>");
            }
        }
    } while (0);

    if (nameMatch || optMatch) return 1;
    return 0;
}

// -----------------------------------------------------------------------------
// SEH wrappers for the public exports - keep all C++ object construction
// outside __try, but wrap the actual decode so a malformed map can't bring
// down the host.
// -----------------------------------------------------------------------------

bool SehParseModeTag(CacheHandle* cache, uint32_t modeTagId, ModelData& data) {
    __try { return ParseModeTag(cache, modeTagId, data); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeGeometry(ModelData* model, uint32_t sectionIndex,
                       uint8_t** outV, uint32_t* outVLen,
                       uint8_t** outI, uint32_t* outILen)
{
    __try { return DecodeGeometryInner(model, sectionIndex, outV, outVLen, outI, outILen); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeUVs(ModelData* model, uint32_t sectionIndex,
                  float** outUv, uint32_t* outUvCount)
{
    __try { return DecodeUVsInner(model, sectionIndex, outUv, outUvCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeNormals(ModelData* model, uint32_t sectionIndex,
                      float** outNrm, uint32_t* outNrmCount)
{
    __try { return DecodeNormalsInner(model, sectionIndex, outNrm, outNrmCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeColors(ModelData* model, uint32_t sectionIndex,
                     float** outCol, uint32_t* outColCount)
{
    __try { return DecodeColorsInner(model, sectionIndex, outCol, outColCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

uint32_t SehResolveDiffuse(CacheHandle* cache, int32_t shaderIndex,
                           int32_t shaderTagId)
{
    __try { return ResolveDiffuseBitmapTagId(cache, shaderIndex, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

// walk the same rmsh -> ShaderProperties[0] ->
// ShaderMaps[] chain as the diffuse resolver, but rank usages with a
// self_illum-first table. Only runs the rmt-driven Step 1 path (no
// heuristic fallback) because for emissive bitmaps we only want explicit
// "self_illum_map" / "self_illum_detail_map" matches - anything else
// would be lying about what the shader authored. Returns 0xFFFFFFFFu
// when no emissive usage resolved.
//
// Caller (NativeMeshAdapter) only invokes this for materials where the
// shader constants show albedo_color ~= 0 AND self_illum_color is
// authored - i.e. the 18-of-27 emissive-only sky panels that currently
// sample base_map (a black-bg placeholder) instead of the real
// self_illum_map content.
static uint32_t ResolveEmissiveBitmapTagId(CacheHandle* cache,
                                           int32_t shaderIndex,
                                           int32_t shaderTagId)
{
    static const struct EmissiveUsageEntry {
        const char* name;
        int         rank;
    } kEmissiveUsages[] = {
        { "self_illum_map",        0 },
        { "self_illum_detail_map", 1 },
    };
    constexpr int kEmissiveUsageCount =
        (int)(sizeof(kEmissiveUsages) / sizeof(kEmissiveUsages[0]));

    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size())
        return 0xFFFFFFFFu;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFFFFFFFFu;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFFFFFFFFu;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 ||
        (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size)
        return 0xFFFFFFFFu;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return 0xFFFFFFFFu;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 ||
        (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size)
        return 0xFFFFFFFFu;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef mapsBlk = ReadTagBlock(props + OFF_SHADER_MAPS_IN_PROPS);
    if (mapsBlk.count <= 0) return 0xFFFFFFFFu;
    int64_t mapsOff = TagMetaFileOff(cache, mapsBlk.pointer);
    if (mapsOff < 0 ||
        (size_t)mapsOff + (size_t)mapsBlk.count * SHADER_MAP_BLOCK_SIZE > cache->size)
        return 0xFFFFFFFFu;

    // rmt2 lookup - same code path as the diffuse resolver but switched
    // to the emissive usage table.
    int32_t rmtRawId = R32(props + 12);
    int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                       ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
    if (rmtTagId < 0 || (uint32_t)rmtTagId >= cache->tags.size())
        return 0xFFFFFFFFu;
    if (!cache->stringTableParsed) return 0xFFFFFFFFu;
    const TagEntry& rmtTe = cache->tags[rmtTagId];
    if (rmtTe.classIndex < 0 ||
        rmtTe.classCode[0] != 'r' ||
        rmtTe.classCode[1] != 'm' ||
        rmtTe.classCode[2] != 't')
        return 0xFFFFFFFFu;
    int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
    if (rmtOff < 0 ||
        (size_t)rmtOff + OFF_RMT_USAGES + 8 > cache->size)
        return 0xFFFFFFFFu;
    const uint8_t* rmtMeta = cache->base + rmtOff;

    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
    if (usagesBlk.count <= 0 || usagesBlk.count > 0x1000) return 0xFFFFFFFFu;
    int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
    if (usagesOff < 0 ||
        (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE > cache->size)
        return 0xFFFFFFFFu;

    int slotMax = mapsBlk.count;
    if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;

    int bestSlot = -1;
    int bestRank = kEmissiveUsageCount;
    for (int i = 0; i < slotMax; ++i) {
        int32_t sid = R32(cache->base + usagesOff + (size_t)i * STRINGID_BLOCK_SIZE);
        const char* usageName = ResolveStringId(cache, sid);
        if (!usageName || !*usageName) continue;
        // Match against the emissive table (rank 0 = self_illum_map).
        int rank = -1;
        size_t len = strlen(usageName);
        size_t stripped = StripUsageBlendSuffix(usageName, len);
        for (int j = 0; j < kEmissiveUsageCount; ++j) {
            size_t candLen = strlen(kEmissiveUsages[j].name);
            if (candLen != stripped) continue;
            if (memcmp(usageName, kEmissiveUsages[j].name, candLen) == 0) {
                rank = kEmissiveUsages[j].rank;
                break;
            }
        }
        if (rank < 0 || rank >= bestRank) continue;
        // Verify the slot points at a valid bitm tag.
        const uint8_t* mapEntry = cache->base + mapsOff +
                                  (size_t)i * SHADER_MAP_BLOCK_SIZE;
        int32_t rawId = R32(mapEntry + 12);
        if ((uint32_t)rawId == 0xFFFFFFFFu) continue;
        uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
        if (bmpId >= cache->tags.size()) continue;
        if (memcmp(cache->tags[bmpId].classCode, "bitm", 4) != 0) continue;
        bestSlot = i;
        bestRank = rank;
        if (rank == 0) break;  // self_illum_map - best possible
    }

    if (bestSlot < 0) return 0xFFFFFFFFu;
    const uint8_t* mapEntry = cache->base + mapsOff +
                              (size_t)bestSlot * SHADER_MAP_BLOCK_SIZE;
    int32_t rawId = R32(mapEntry + 12);
    return (uint32_t)rawId & 0xFFFFu;
}

static uint32_t SehResolveEmissive(CacheHandle* cache, int32_t shaderIndex,
                                   int32_t shaderTagId)
{
    __try { return ResolveEmissiveBitmapTagId(cache, shaderIndex, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

// =============================================================================
// OBJECT_DETAIL_MAP_V1: diffuse `detail_map` / `detail_map2`
// resolver for render_model object segments. Direct mirror of the BSP path's
// ResolveDetailMapInfo (MapBspParser.cpp) - the rmsh/rmt2 tag layout is
// identical regardless of whether the shader is referenced by a BSP material
// or a render_model section. 71% of lit object segments and 100% of PBR
// object segments author a `detail_map` constant that the engine composites
// (albedo.hlsl_include calc_albedo_detail_ps: base.rgb * (detail.rgb *
// DETAIL_MULTIPLIER 4.59479)), but the object walker never sampled it - only
// the self_illum detail fuse (sky) was exported. This is the missing diffuse
// detail layer. We export bitmap tag id + UV tile (xform.xy) per segment so
// the viewer materials sample detail at its own engine-authored tile, distinct
// from the base UV (object detail maps tile at their own scale).
//
// rmt2 layout constants (identical to MapBspParser.cpp FWD_OFF_* block):
//   Arguments[]   @ rmt+72   (BlockCollection<StringId>)
//   TilingData[]  @ props+28 (real4 per arg, 16 bytes)
constexpr int OFF_RMT_ARGUMENTS        = 72;
constexpr int OFF_TILING_DATA_IN_PROPS = 28;
constexpr int TILING_DATA_BLOCK_SIZE   = 16;

struct ModelDetailMapInfo {
    uint32_t bitmapTagId;
    float    tileX;
    float    tileY;
};

// usageMatch must be the exact rmt2 usage / arg string ("detail_map" len 10 or
// "detail_map2" len 11). matchLen is its strlen. The base/diffuse usage is NOT
// matched here - caller passes the auxiliary detail slot it wants. Returns
// 0xFFFFFFFFu bitmap on any miss; tiles default to 1.0.
static ModelDetailMapInfo ResolveModelDetailInfo(CacheHandle* cache,
                                                 int32_t shaderTagId,
                                                 const char* usageMatch,
                                                 size_t matchLen)
{
    ModelDetailMapInfo out = { 0xFFFFFFFFu, 1.0f, 1.0f };
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return out;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return out;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return out;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size) return out;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return out;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 ||
        (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size) return out;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef mapsBlk = ReadTagBlock(props + OFF_SHADER_MAPS_IN_PROPS);
    if (mapsBlk.count <= 0) return out;
    int64_t mapsOff = TagMetaFileOff(cache, mapsBlk.pointer);
    if (mapsOff < 0 ||
        (size_t)mapsOff + (size_t)mapsBlk.count * SHADER_MAP_BLOCK_SIZE > cache->size) return out;

    int32_t rmtRawId = R32(props + 12);
    int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                       ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
    if (rmtTagId < 0 || (uint32_t)rmtTagId >= cache->tags.size()) return out;
    if (!cache->stringTableParsed) return out;
    const TagEntry& rmtTe = cache->tags[rmtTagId];
    if (rmtTe.classIndex < 0 ||
        rmtTe.classCode[0] != 'r' || rmtTe.classCode[1] != 'm' ||
        rmtTe.classCode[2] != 't') return out;
    int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
    if (rmtOff < 0 || (size_t)rmtOff + OFF_RMT_USAGES + 8 > cache->size) return out;
    const uint8_t* rmtMeta = cache->base + rmtOff;

    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
    if (usagesBlk.count <= 0 || usagesBlk.count > 0x1000) return out;
    int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
    if (usagesOff < 0 ||
        (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE > cache->size) return out;

    int slotMax = mapsBlk.count;
    if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;

    int matchSlot = -1;
    for (int i = 0; i < slotMax; ++i) {
        int32_t sid = R32(cache->base + usagesOff + (size_t)i * STRINGID_BLOCK_SIZE);
        const char* usageName = ResolveStringId(cache, sid);
        if (!usageName || !*usageName) continue;
        if (strlen(usageName) != matchLen) continue;
        if (memcmp(usageName, usageMatch, matchLen) != 0) continue;

        const uint8_t* mapEntry = cache->base + mapsOff +
                                  (size_t)i * SHADER_MAP_BLOCK_SIZE;
        int32_t rawId = R32(mapEntry + 12);
        if ((uint32_t)rawId == 0xFFFFFFFFu) continue;
        uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
        if (bmpId >= cache->tags.size()) continue;
        if (memcmp(cache->tags[bmpId].classCode, "bitm", 4) != 0) continue;
        out.bitmapTagId = bmpId;
        matchSlot = i;
        break;
    }
    if (matchSlot < 0) return out;

    // Read TilingData[argIdx] where argIdx is the position of usageMatch in
    // rmt2.Arguments[]. Same chain as the BSP detail resolver.
    TagBlockRef argsBlk = ReadTagBlock(rmtMeta + OFF_RMT_ARGUMENTS);
    if (argsBlk.count <= 0 || argsBlk.count > 0x1000) return out;
    int64_t argsOff = TagMetaFileOff(cache, argsBlk.pointer);
    if (argsOff < 0 ||
        (size_t)argsOff + (size_t)argsBlk.count * STRINGID_BLOCK_SIZE > cache->size) return out;

    int argIdx = -1;
    for (int a = 0; a < argsBlk.count; ++a) {
        int32_t sid = R32(cache->base + argsOff + (size_t)a * STRINGID_BLOCK_SIZE);
        const char* argName = ResolveStringId(cache, sid);
        if (!argName) continue;
        if (strlen(argName) != matchLen) continue;
        if (memcmp(argName, usageMatch, matchLen) != 0) continue;
        argIdx = a;
        break;
    }
    if (argIdx < 0) return out;

    TagBlockRef tilingBlk = ReadTagBlock(props + OFF_TILING_DATA_IN_PROPS);
    if (tilingBlk.count <= 0 || (size_t)argIdx >= (size_t)tilingBlk.count) return out;
    int64_t tilingOff = TagMetaFileOff(cache, tilingBlk.pointer);
    if (tilingOff < 0 ||
        (size_t)tilingOff + (size_t)tilingBlk.count * TILING_DATA_BLOCK_SIZE > cache->size) return out;
    const uint8_t* td = cache->base + tilingOff + (size_t)argIdx * TILING_DATA_BLOCK_SIZE;
    float fx = 1.0f, fy = 1.0f;
    memcpy(&fx, td + 0, 4);
    memcpy(&fy, td + 4, 4);
    if (std::isfinite(fx) && fx > 0.0f) out.tileX = fx;
    if (std::isfinite(fy) && fy > 0.0f) out.tileY = fy;
    return out;
}

static ModelDetailMapInfo SehResolveModelDetail(CacheHandle* cache, int32_t shaderTagId,
                                                const char* usageMatch, size_t matchLen)
{
    __try { return ResolveModelDetailInfo(cache, shaderTagId, usageMatch, matchLen); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return ModelDetailMapInfo{ 0xFFFFFFFFu, 1.0f, 1.0f };
    }
}

uint8_t SehResolveBlendMode(CacheHandle* cache, int32_t shaderTagId)
{
    __try { return ResolveShaderBlendMode(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

// MAT-1: SEH-guarded material-model resolve (same contract as SehResolveBlendMode).
uint8_t SehResolveMaterialModel(CacheHandle* cache, int32_t shaderTagId)
{
    __try { return ResolveShaderMaterialModel(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

// LIT-SI-3 (model side): SEH-guarded self-illum-mode resolve (same contract).
uint8_t SehResolveSelfIllumMode(CacheHandle* cache, int32_t shaderTagId)
{
    __try { return ResolveShaderSelfIllumMode(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

// SKY-3: SEH-guarded sky-class resolve (same contract).
uint8_t SehResolveSkyClass(CacheHandle* cache, int32_t shaderTagId)
{
    __try { return ResolveShaderSkyClass(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

// #264 turret/attachment walker. hlmt Variants[]->Objects[] lists each child object + the parent
// render_model marker it rides. We resolve the child's render_model and compose the child's
// MODEL-SPACE transform (parent node chain  o  marker local) so the Rust side only multiplies by the
// placed object's world matrix. Offsets RE-verified against forge_halo.map (MccHaloReachU13):
//   hlmt Variants block @ hlmt+0x84 ; variant stride 0x38 (Name@0, Objects sub-block @+0x20)
//   Objects element stride 0x20 (parentMarkerSid@0, child tag-ref@+0x0C -> datum@+0x18)
//   mode MarkerGroups @ mode+0x3C (stride 0x10: Name@0, markers block @+0x04)
//   marker instance stride 0x30 (nodeIdx u8 @2, T float3 @+0x04, Q xyzw @+0x10, scale @+0x20)
//   mode Nodes @ mode+48, stride 96 (parent i16 @4, pos float3 @12, quat xyzw @24)
#pragma pack(push, 1)
struct ZH_Attachment {
    uint32_t childModeTagId;   // resolved child render_model
    uint32_t childObjTagId;    // child obj short id
    uint32_t markerStringId;   // parent marker (debug)
    uint32_t variantNameSid;   // Rust matches the placement's variant by this
    int32_t  variantIndex;     // hlmt Variants[] index (0 = default loadout)
    float    pos[3];           // child model-space translation
    float    rot[4];           // child model-space quaternion (x,y,z,w)
    float    scale;            // (kept for completeness; caller may ignore)
};
#pragma pack(pop)

extern "C" uint32_t __stdcall ZH_TAG_ResolveModeTagId(uint64_t, uint32_t);

namespace {
struct AQuat { float x, y, z, w; };
struct AVec3 { float x, y, z; };
inline AQuat aq_mul(AQuat a, AQuat b) {
    return { a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
             a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
             a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
             a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z };
}
inline AVec3 aq_rot(AQuat q, AVec3 v) {
    AVec3 u{ q.x, q.y, q.z }; float s = q.w;
    float d = u.x*v.x + u.y*v.y + u.z*v.z;
    float ss = s*s - (u.x*u.x + u.y*u.y + u.z*u.z);
    AVec3 c{ u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x };
    return { 2*d*u.x + ss*v.x + 2*s*c.x, 2*d*u.y + ss*v.y + 2*s*c.y, 2*d*u.z + ss*v.z + 2*s*c.z };
}

static uint32_t ScanClassRef(CacheHandle* cache, const uint8_t* meta, size_t maxBytes, const char* g) {
    if (maxBytes < 16) return 0xFFFFFFFFu;
    const uint32_t t = ((uint32_t)(uint8_t)g[0]<<24)|((uint32_t)(uint8_t)g[1]<<16)|((uint32_t)(uint8_t)g[2]<<8)|((uint32_t)(uint8_t)g[3]);
    const size_t end = maxBytes - 16;
    for (size_t o = 0; o <= end; o += 4) {
        if (RU32(meta+o) != t) continue;
        uint32_t raw = RU32(meta+o+12);
        if (raw == 0xFFFFFFFFu) continue;
        uint32_t id = raw & 0xFFFFu;
        if (id >= cache->tags.size()) continue;
        if (memcmp(cache->tags[id].classCode, g, 4) != 0) continue;
        return id;
    }
    return 0xFFFFFFFFu;
}

// Model-space transform of node `nodeIdx` (walk parent chain to root, compose parent  o  local).
static void NodeWorld(CacheHandle* cache, uint32_t modeTagId, int nodeIdx, AQuat& outRot, AVec3& outPos) {
    outRot = { 0, 0, 0, 1 }; outPos = { 0, 0, 0 };
    if (nodeIdx < 0 || modeTagId >= cache->tags.size()) return;
    int64_t off = TagMetaFileOff(cache, cache->tags[modeTagId].metaPointerRaw);
    if (off < 0 || (size_t)off + 0x40 > cache->size) return;
    const uint8_t* meta = cache->base + off;
    TagBlockRef nodes = ReadTagBlock(meta + 48); // OFF_MODE_NODES
    if (nodes.count <= 0 || nodes.count > 4096) return;
    int64_t nOff = TagMetaFileOff(cache, nodes.pointer);
    if (nOff < 0 || (size_t)nOff + (size_t)nodes.count*96 > cache->size) return;
    const uint8_t* base = cache->base + nOff;
    // Collect chain nodeIdx -> ... -> root (guard against cycles / bad parents).
    int chain[64]; int n = 0; int cur = nodeIdx;
    while (cur >= 0 && cur < nodes.count && n < 64) {
        chain[n++] = cur;
        int16_t parent = R16(base + (size_t)cur*96 + 4);
        if (parent == cur) break;
        cur = parent;
    }
    // Fold from root down: world = parent  o  local.
    for (int i = n - 1; i >= 0; --i) {
        const uint8_t* e = base + (size_t)chain[i]*96;
        AVec3 lp; memcpy(&lp, e + 12, 12);
        AQuat lr; memcpy(&lr, e + 24, 16);
        outPos = { outPos.x + aq_rot(outRot, lp).x, outPos.y + aq_rot(outRot, lp).y, outPos.z + aq_rot(outRot, lp).z };
        outRot = aq_mul(outRot, lr);
    }
}

// Marker `markerSid` in mode `modeTagId`: returns instance-0 node + local T/Q/scale.
static bool MarkerLocal(CacheHandle* cache, uint32_t modeTagId, uint32_t markerSid,
                        int& nodeIdx, AVec3& lp, AQuat& lr, float& scale) {
    nodeIdx = -1; lp = { 0, 0, 0 }; lr = { 0, 0, 0, 1 }; scale = 1;
    if (modeTagId >= cache->tags.size()) return false;
    int64_t off = TagMetaFileOff(cache, cache->tags[modeTagId].metaPointerRaw);
    if (off < 0 || (size_t)off + 0x44 > cache->size) return false;
    const uint8_t* meta = cache->base + off;
    TagBlockRef groups = ReadTagBlock(meta + 0x3C);
    if (groups.count <= 0 || groups.count > 4096) return false;
    int64_t gOff = TagMetaFileOff(cache, groups.pointer);
    if (gOff < 0 || (size_t)gOff + (size_t)groups.count*0x10 > cache->size) return false;
    for (int g = 0; g < groups.count; ++g) {
        const uint8_t* ge = cache->base + gOff + (size_t)g*0x10;
        if (RU32(ge + 0) != markerSid) continue;
        TagBlockRef mk = ReadTagBlock(ge + 4);
        if (mk.count <= 0 || mk.count > 4096) return false;
        int64_t mOff = TagMetaFileOff(cache, mk.pointer);
        if (mOff < 0 || (size_t)mOff + (size_t)mk.count*0x30 > cache->size) return false;
        const uint8_t* m = cache->base + mOff; // instance 0
        uint8_t ni = m[2];
        nodeIdx = (ni == 0xFF) ? -1 : (int)ni;
        memcpy(&lp, m + 0x04, 12);
        memcpy(&lr, m + 0x10, 16);
        memcpy(&scale, m + 0x20, 4);
        return true;
    }
    return false;
}
} // namespace

extern "C" __declspec(dllexport) int32_t __stdcall ZH_MMP_EnumerateAttachments(
    uint64_t cacheHandle, uint32_t objTagId, ZH_Attachment* out, int32_t maxOut)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return -1;
    if (objTagId >= cache->tags.size()) return -1;
    int64_t objOff = TagMetaFileOff(cache, cache->tags[objTagId].metaPointerRaw);
    if (objOff < 0) return -1;
    size_t objAvail = (size_t)cache->size - (size_t)objOff;
    if (objAvail < 16) return 0;
    size_t objScan = objAvail < 0x400 ? objAvail : 0x400;
    uint32_t hlmtId = ScanClassRef(cache, cache->base + objOff, objScan, "hlmt");
    if (hlmtId == 0xFFFFFFFFu) return 0;
    int64_t hlmtOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
    if (hlmtOff < 0 || (size_t)hlmtOff + 0x8C > cache->size) return 0;
    const uint8_t* hlmt = cache->base + hlmtOff;
    uint32_t parentMode = ZH_TAG_ResolveModeTagId(cacheHandle, objTagId);

    TagBlockRef variants = ReadTagBlock(hlmt + 0x84);
    if (variants.count <= 0 || variants.count > 256) return 0;
    int64_t vOff = TagMetaFileOff(cache, variants.pointer);
    if (vOff < 0 || (size_t)vOff + (size_t)variants.count*0x38 > cache->size) return 0;

    int32_t written = 0;
    for (int v = 0; v < variants.count; ++v) {
        const uint8_t* ve = cache->base + vOff + (size_t)v*0x38;
        uint32_t vName = RU32(ve + 0x00);
        TagBlockRef objs = ReadTagBlock(ve + 0x20);
        if (objs.count <= 0 || objs.count > 256) continue;
        int64_t oOff = TagMetaFileOff(cache, objs.pointer);
        if (oOff < 0 || (size_t)oOff + (size_t)objs.count*0x20 > cache->size) continue;
        for (int o = 0; o < objs.count; ++o) {
            const uint8_t* oe = cache->base + oOff + (size_t)o*0x20;
            uint32_t markerSid = RU32(oe + 0x00);
            uint32_t childRaw = RU32(oe + 0x18);
            if (childRaw == 0xFFFFFFFFu) continue;
            uint32_t childShort = childRaw & 0xFFFFu;
            if (childShort >= cache->tags.size()) continue;
            uint32_t childMode = ZH_TAG_ResolveModeTagId(cacheHandle, childShort);
            if (childMode == 0xFFFFFFFFu || childMode == 0) continue;
            int nodeIdx; AVec3 lp; AQuat lr; float mscale;
            MarkerLocal(cache, parentMode, markerSid, nodeIdx, lp, lr, mscale);
            AQuat nRot; AVec3 nPos;
            NodeWorld(cache, parentMode, nodeIdx, nRot, nPos);
            // child model-space = node world  o  marker local
            AVec3 cp = aq_rot(nRot, lp);
            AVec3 pos = { nPos.x + cp.x, nPos.y + cp.y, nPos.z + cp.z };
            AQuat rot = aq_mul(nRot, lr);
            if (out && written < maxOut) {
                ZH_Attachment& a = out[written];
                a.childModeTagId = childMode;
                a.childObjTagId = childShort;
                a.markerStringId = markerSid;
                a.variantNameSid = vName;
                a.variantIndex = v;
                a.pos[0] = pos.x; a.pos[1] = pos.y; a.pos[2] = pos.z;
                a.rot[0] = rot.x; a.rot[1] = rot.y; a.rot[2] = rot.z; a.rot[3] = rot.w;
                a.scale = 1.0f;
            }
            ++written;
        }
    }
    return written;
}

// #269: build a per-section allow mask for a specific model VARIANT of `objTagId`.
// Chain (RE-verified byte-exact against forge_halo.map):
//   obj -> hlmt (ScanClassRef "hlmt") -> Variants@hlmt+0x84 (stride 0x38): match Name@0x00==variantSid
//   variant.Regions@+0x14 (stride 0x18): RenderModelRegionIndex i8@+0x04, Permutations TB@+0x08
//   variant.Perm (stride 0x10): RenderModelPermutationIndex i8@+0x04 (<0 => region hidden this variant)
//   render_model.Regions@mode+12 (stride 16): [rmRegIdx].Permutations@+0x04 [rmPermIdx] (stride 16)
//     -> SectionIndex@+0x04 / SectionCount@+0x06 -> set those bytes in outMask.
// Returns render_model section count written (0 = variant not found / no geometry => caller keeps
// its default perm[0]-of-every-region behaviour, so there is never a regression).
extern "C" __declspec(dllexport) int32_t __stdcall ZH_MMP_GetVariantSectionMask(
    uint64_t cacheHandle, uint32_t objTagId, uint32_t variantSid,
    uint8_t* outMask, int32_t maxMask)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || objTagId >= cache->tags.size()) return 0;
    if (variantSid == 0 || variantSid == 0xFFFFFFFFu) return 0; // default variant -> caller default
    if (!outMask || maxMask <= 0) return 0;

    __try {
        // --- render_model: regions + section count ---
        uint32_t modeId = ZH_TAG_ResolveModeTagId(cacheHandle, objTagId);
        if (modeId == 0 || modeId == 0xFFFFFFFFu || modeId >= cache->tags.size()) return 0;
        int64_t modeOff = TagMetaFileOff(cache, cache->tags[modeId].metaPointerRaw);
        if (modeOff < 0 || (size_t)modeOff + 0x70 > cache->size) return 0;
        const uint8_t* mmeta = cache->base + modeOff;
        TagBlockRef rmSections = ReadTagBlock(mmeta + OFF_MODE_SECTIONS);
        TagBlockRef rmRegions  = ReadTagBlock(mmeta + OFF_MODE_REGIONS);
        int nSec = rmSections.count;
        if (nSec <= 0 || nSec > 0x1000) return 0;
        int64_t rmRegOff = TagMetaFileOff(cache, rmRegions.pointer);
        if (rmRegions.count <= 0 || rmRegions.count > 0x1000 || rmRegOff < 0 ||
            (size_t)rmRegOff + (size_t)rmRegions.count * 16 > cache->size) return 0;

        // --- obj -> hlmt -> Variants ---
        int64_t objOff = TagMetaFileOff(cache, cache->tags[objTagId].metaPointerRaw);
        if (objOff < 0) return 0;
        size_t objAvail = (size_t)cache->size - (size_t)objOff;
        if (objAvail < 16) return 0;
        uint32_t hlmtId = ScanClassRef(cache, cache->base + objOff,
                                       objAvail < 0x400 ? objAvail : 0x400, "hlmt");
        if (hlmtId == 0xFFFFFFFFu) return 0;
        int64_t hlmtOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
        if (hlmtOff < 0 || (size_t)hlmtOff + 0x8C > cache->size) return 0;
        const uint8_t* hlmt = cache->base + hlmtOff;
        TagBlockRef variants = ReadTagBlock(hlmt + 0x84);
        if (variants.count <= 0 || variants.count > 256) return 0;
        int64_t vOff = TagMetaFileOff(cache, variants.pointer);
        if (vOff < 0 || (size_t)vOff + (size_t)variants.count * 0x38 > cache->size) return 0;

        // Find the variant whose Name@0x00 == variantSid.
        const uint8_t* ve = nullptr;
        for (int v = 0; v < variants.count; ++v) {
            const uint8_t* e = cache->base + vOff + (size_t)v * 0x38;
            if (RU32(e + 0x00) == variantSid) { ve = e; break; }
        }
        if (!ve) return 0; // not found -> default

        TagBlockRef vRegions = ReadTagBlock(ve + 0x14);
        if (vRegions.count <= 0 || vRegions.count > 256) return 0;
        int64_t vrOff = TagMetaFileOff(cache, vRegions.pointer);
        if (vrOff < 0 || (size_t)vrOff + (size_t)vRegions.count * 0x18 > cache->size) return 0;

        int fill = nSec < maxMask ? nSec : maxMask;
        memset(outMask, 0, (size_t)fill);
        int marked = 0;

        for (int vr = 0; vr < vRegions.count; ++vr) {
            const uint8_t* vre = cache->base + vrOff + (size_t)vr * 0x18;
            int rmRegIdx = (int)(int8_t)vre[0x04];
            if (rmRegIdx < 0 || rmRegIdx >= rmRegions.count) continue;
            TagBlockRef vPerms = ReadTagBlock(vre + 0x08);
            if (vPerms.count <= 0 || vPerms.count > 256) continue;
            int64_t vpOff = TagMetaFileOff(cache, vPerms.pointer);
            if (vpOff < 0 || (size_t)vpOff + (size_t)vPerms.count * 0x10 > cache->size) continue;

            // render_model region rmRegIdx -> its permutations block.
            const uint8_t* rmReg = cache->base + rmRegOff + (size_t)rmRegIdx * 16;
            TagBlockRef rmPerms = ReadTagBlock(rmReg + 4 /*OFF_REGION_PERMS*/);
            if (rmPerms.count <= 0 || rmPerms.count > 0x1000) continue;
            int64_t rmpOff = TagMetaFileOff(cache, rmPerms.pointer);
            if (rmpOff < 0 || (size_t)rmpOff + (size_t)rmPerms.count * 16 > cache->size) continue;

            for (int vp = 0; vp < vPerms.count; ++vp) {
                const uint8_t* vpe = cache->base + vpOff + (size_t)vp * 0x10;
                int rmPermIdx = (int)(int8_t)vpe[0x04];
                if (rmPermIdx < 0 || rmPermIdx >= rmPerms.count) continue; // hidden this variant
                const uint8_t* rmp = cache->base + rmpOff + (size_t)rmPermIdx * 16;
                int16_t secIdx = (int16_t)R16(rmp + 4 /*OFF_PERM_SECTION_IDX*/);
                int16_t secCnt = (int16_t)R16(rmp + 6 /*OFF_PERM_SECTION_CNT*/);
                if (secIdx < 0 || secCnt <= 0) continue;
                for (int s = 0; s < secCnt; ++s) {
                    int idx = secIdx + s;
                    if (idx >= 0 && idx < fill && outMask[idx] == 0) {
                        outMask[idx] = 1; ++marked;
                    }
                }
            }
        }
        return marked > 0 ? fill : 0;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return 0;
    }
}

// Per-section allow mask for an object's DEFAULT (index-0) hlmt variant - the engine's
// resting appearance. HMS otherwise takes perm[0]-of-every-region, which for vehicles like the
// Falcon picks the WRONG permutation (region 'props' perm0='blur' = the spinning rotor disc) and
// DROPS regions whose perm[0] is empty (tail/wings/lights, SectionIndex=-1). The default variant's
// region -> permutation map selects the static-blade rotor perm + the real tail/wing sections. Returns
// section count written (0 = no hlmt variants / no geometry => caller keeps perm[0] behaviour, so no
// regression for models without a meaningful default variant).
extern "C" __declspec(dllexport) int32_t __stdcall ZH_MMP_GetDefaultVariantSectionMask(
    uint64_t cacheHandle, uint32_t objTagId, uint8_t* outMask, int32_t maxMask)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || objTagId >= cache->tags.size() || !outMask || maxMask <= 0) return 0;
    __try {
        int64_t objOff = TagMetaFileOff(cache, cache->tags[objTagId].metaPointerRaw);
        if (objOff < 0) return 0;
        size_t objAvail = (size_t)cache->size - (size_t)objOff;
        if (objAvail < 16) return 0;
        uint32_t hlmtId = ScanClassRef(cache, cache->base + objOff,
                                       objAvail < 0x400 ? objAvail : 0x400, "hlmt");
        if (hlmtId == 0xFFFFFFFFu) return 0;
        int64_t hlmtOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
        if (hlmtOff < 0 || (size_t)hlmtOff + 0x8C > cache->size) return 0;
        TagBlockRef variants = ReadTagBlock(cache->base + hlmtOff + 0x84);
        if (variants.count <= 0 || variants.count > 256) return 0;
        int64_t vOff = TagMetaFileOff(cache, variants.pointer);
        if (vOff < 0 || (size_t)vOff + 0x38 > cache->size) return 0;
        uint32_t v0sid = RU32(cache->base + vOff + 0x00); // variant[0].Name sid (default loadout)
        if (v0sid == 0 || v0sid == 0xFFFFFFFFu) return 0;
        return ZH_MMP_GetVariantSectionMask(cacheHandle, objTagId, v0sid, outMask, maxMask);
    } __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}

// #301: DETERMINISTIC resting-appearance section mask - the engine-accurate replacement for the
// name-token damage heuristic in SehParseModeTag (which marks A+B of EVERY non-damage perm per
// region, so static + rotating wheels overlap, blur/damage geometry leaks when it isn't name-tagged,
// and legit parts named "minor/major/..." get falsely dropped). The engine picks, per render-model
// region, exactly ONE permutation: the one the object's hlmt DEFAULT variant (variant[0]) maps that
// region to, falling back to render-model permutation 0 for regions the variant doesn't list. We mark
// BOTH the A(idx@0x04/cnt@0x06) and B(idx@0x0C/cnt@0x0E) section refs of that single permutation (the
// B ref carries the static blades / real body geometry per the #269 falcon RE), then OR-in orphan
// sections referenced by NO region at all (small detached props). One perm per region => no overlap:
// static wheels, static rotor, undamaged body - the object's resting in-game look, identical for the
// preview and the placed viewport (both call this with the obje tag). Returns section count written;
// 0 => the obj has no hlmt VARIANTS (a plain prop) => caller keeps the SehParseModeTag heuristic, so
// there is no regression for models that never had a meaningful variant to resolve.
// Body split out of the SEH wrapper: MSVC forbids C++ objects that need unwinding (the marking
// lambda, etc.) in a function that uses __try. The exported ZH_MMP_ResolveDefaultVariantMask below
// wraps this in __try/__except - same structure as SehParseModeTag.
static int ResolveDefaultVariantMaskImpl(
    uint64_t cacheHandle, CacheHandle* cache, uint32_t objTagId, uint8_t* outMask, int32_t maxMask)
{
    static bool secdiag = []{
        char* v = nullptr; size_t n = 0;
        if (_dupenv_s(&v, &n, "HMS_SECDIAG") == 0 && v) { bool on = v[0] && v[0] != '0'; free(v); return on; }
        return false;
    }();
    {
        // --- render_model: sections + regions ---
        uint32_t modeId = ZH_TAG_ResolveModeTagId(cacheHandle, objTagId);
        if (modeId == 0 || modeId == 0xFFFFFFFFu || modeId >= cache->tags.size()) return 0;
        int64_t modeOff = TagMetaFileOff(cache, cache->tags[modeId].metaPointerRaw);
        if (modeOff < 0 || (size_t)modeOff + 0x70 > cache->size) return 0;
        const uint8_t* mmeta = cache->base + modeOff;
        TagBlockRef rmSections = ReadTagBlock(mmeta + OFF_MODE_SECTIONS);
        TagBlockRef rmRegions  = ReadTagBlock(mmeta + OFF_MODE_REGIONS);
        int nSec = rmSections.count;
        if (nSec <= 0 || nSec > 0x1000) return 0;
        if (rmRegions.count <= 0 || rmRegions.count > 0x1000) return 0;
        int64_t rmRegOff = TagMetaFileOff(cache, rmRegions.pointer);
        if (rmRegOff < 0 || (size_t)rmRegOff + (size_t)rmRegions.count * 16 > cache->size) return 0;

        // --- obj -> hlmt -> variant[0] (default resting variant) -> per-region permutation override ---
        // Fixed stack arrays (not std::vector) - MSVC forbids objects needing unwinding under __try.
        int regOverride[0x1000];
        for (int i = 0; i < rmRegions.count; ++i) regOverride[i] = -1;
        uint32_t v0sid = 0;
        {
            int64_t objOff = TagMetaFileOff(cache, cache->tags[objTagId].metaPointerRaw);
            if (objOff >= 0) {
                size_t objAvail = (size_t)cache->size - (size_t)objOff;
                if (objAvail >= 16) {
                    uint32_t hlmtId = ScanClassRef(cache, cache->base + objOff,
                                                   objAvail < 0x400 ? objAvail : 0x400, "hlmt");
                    if (hlmtId != 0xFFFFFFFFu && hlmtId < cache->tags.size()) {
                        int64_t hlmtOff = TagMetaFileOff(cache, cache->tags[hlmtId].metaPointerRaw);
                        if (hlmtOff >= 0 && (size_t)hlmtOff + 0x8C <= cache->size) {
                            TagBlockRef variants = ReadTagBlock(cache->base + hlmtOff + 0x84);
                            int64_t vOff = TagMetaFileOff(cache, variants.pointer);
                            if (variants.count > 0 && variants.count <= 256 && vOff >= 0 &&
                                (size_t)vOff + (size_t)variants.count * 0x38 <= cache->size) {
                                const uint8_t* ve = cache->base + vOff; // variant[0] = default
                                v0sid = RU32(ve + 0x00);
                                TagBlockRef vRegions = ReadTagBlock(ve + 0x14);
                                int64_t vrOff = TagMetaFileOff(cache, vRegions.pointer);
                                if (vRegions.count > 0 && vRegions.count <= 256 && vrOff >= 0 &&
                                    (size_t)vrOff + (size_t)vRegions.count * 0x18 <= cache->size) {
                                    for (int vr = 0; vr < vRegions.count; ++vr) {
                                        const uint8_t* vre = cache->base + vrOff + (size_t)vr * 0x18;
                                        int rmRegIdx = (int)(int8_t)vre[0x04];
                                        if (rmRegIdx < 0 || rmRegIdx >= rmRegions.count) continue;
                                        TagBlockRef vPerms = ReadTagBlock(vre + 0x08);
                                        if (vPerms.count <= 0 || vPerms.count > 256) continue;
                                        int64_t vpOff = TagMetaFileOff(cache, vPerms.pointer);
                                        if (vpOff < 0 || (size_t)vpOff + 0x10 > cache->size) continue;
                                        // First permutation entry = the variant's resting selection.
                                        int rmPermIdx = (int)(int8_t)(cache->base + vpOff)[0x04];
                                        if (rmPermIdx >= 0) regOverride[rmRegIdx] = rmPermIdx;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // No hlmt variants => plain prop => let the caller keep the heuristic (no regression).
        if (v0sid == 0 || v0sid == 0xFFFFFFFFu) return 0;

        int fill = nSec < maxMask ? nSec : maxMask;
        memset(outMask, 0, (size_t)fill);
        uint8_t referenced[0x1000];
        memset(referenced, 0, (size_t)nSec);
        int marked = 0;
        auto markRange = [&](int idx, int cnt, bool allow) {
            if (idx < 0 || cnt <= 0) return;
            for (int s = 0; s < cnt; ++s) {
                int i = idx + s;
                if (i < 0 || i >= nSec) continue;
                referenced[i] = 1;
                if (allow && i < fill && outMask[i] == 0) { outMask[i] = 1; ++marked; }
            }
        };

        for (int r = 0; r < rmRegions.count; ++r) {
            const uint8_t* rmReg = cache->base + rmRegOff + (size_t)r * 16;
            TagBlockRef rmPerms = ReadTagBlock(rmReg + 4 /*OFF_REGION_PERMS*/);
            if (rmPerms.count <= 0 || rmPerms.count > 0x1000) continue;
            int64_t rmpOff = TagMetaFileOff(cache, rmPerms.pointer);
            if (rmpOff < 0 || (size_t)rmpOff + (size_t)rmPerms.count * 16 > cache->size) continue;
            int chosen = regOverride[r] >= 0 ? regOverride[r] : 0;
            if (chosen >= rmPerms.count) chosen = 0;
            // Allow ONLY the chosen permutation (A+B); walk the rest solely to mark referenced[]
            // so genuinely-orphan sections (no region references them) can still be included.
            for (int p = 0; p < rmPerms.count; ++p) {
                const uint8_t* rmp = cache->base + rmpOff + (size_t)p * 16;
                int aIdx = (int16_t)R16(rmp + 4),  aCnt = (int16_t)R16(rmp + 6);
                int bIdx = (int16_t)R16(rmp + 12), bCnt = (int16_t)R16(rmp + 14);
                bool allow = (p == chosen);
                markRange(aIdx, aCnt, allow);
                markRange(bIdx, bCnt, allow);
            }
        }
        // Orphans: sections referenced by NO region/perm at all (detached props) -> allow.
        for (int i = 0; i < fill; ++i)
            if (!referenced[i] && outMask[i] == 0) { outMask[i] = 1; ++marked; }

        if (secdiag)
            NativeDiag("RESOLVEMASK obj[%u] mode[%u] v0sid=0x%08X regions=%d -> %d/%d sections allowed",
                objTagId, modeId, v0sid, rmRegions.count, marked, nSec);
        return marked > 0 ? fill : 0;
    }
}

extern "C" __declspec(dllexport) int32_t __stdcall ZH_MMP_ResolveDefaultVariantMask(
    uint64_t cacheHandle, uint32_t objTagId, uint8_t* outMask, int32_t maxMask)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || objTagId >= cache->tags.size() || !outMask || maxMask <= 0) return 0;
    __try { return ResolveDefaultVariantMaskImpl(cacheHandle, cache, objTagId, outMask, maxMask); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}

extern "C" __declspec(dllexport) ZH_ModelHandle __stdcall ZH_MMP_OpenModel(
    uint64_t cacheHandle, uint32_t modeTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;

    auto* model = new (std::nothrow) ModelData();
    if (!model) return 0;
    model->cacheHandle = cacheHandle;
    model->modeTagId   = modeTagId;
    model->resourceIndex = -1;

    bool ok = SehParseModeTag(cache, modeTagId, *model);
    if (!ok) {
        if (model->resourceData) free(model->resourceData);
        delete model;
        return 0;
    }

    uint64_t handle = g_nextModelHandle.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_modelHandlesMutex);
        g_modelHandles[handle] = model;
    }
    return handle;
}

extern "C" __declspec(dllexport) void __stdcall ZH_MMP_CloseModel(ZH_ModelHandle h)
{
    ModelData* model = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_modelHandlesMutex);
        auto it = g_modelHandles.find(h);
        if (it == g_modelHandles.end()) return;
        model = it->second;
        g_modelHandles.erase(it);
    }
    if (!model) return;
    if (model->resourceData) free(model->resourceData);
    delete model;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_MMP_GetSectionCount(ZH_ModelHandle h)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0;
    return (uint32_t)model->sections.size();
}

// Returns 1 iff the given section index is part of permutation 0 of any
// region - used by the viewer to filter out the fan of armor variants
// on player bipeds. For models with no Regions block (small props), every
// section is allowed (the parser pre-fills allowedSection[] with all 1s
// in that case). Returns 0 if h is invalid or the section is filtered.
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_MMP_IsSectionAllowed(
    ZH_ModelHandle h, uint32_t sectionIndex)
{
    ModelData* model = LookupModel(h);
    if (!model || sectionIndex >= model->allowedSection.size()) return 0;
    return model->allowedSection[sectionIndex];
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_GetSection(
    ZH_ModelHandle h, uint32_t i, ZH_ModelSection* outSec)
{
    if (!outSec) return false;
    memset(outSec, 0, sizeof(*outSec));
    ModelData* model = LookupModel(h);
    if (!model || i >= model->sections.size()) return false;
    const ModelSection& s = model->sections[i];

    outSec->VertexCount  = s.vertexCount;
    outSec->IndexCount   = s.indexCount;
    outSec->VertexStride = 12;
    outSec->IndexStride  = (s.vertexCount > 0xFFFF) ? 4u : 2u;
    outSec->VertexFormat = s.vertexFormat;
    outSec->Flags        = s.flags;
    // NodeIndex == 0xFF in Reclaimer means "no bone" (renders as -1).
    outSec->NodeIndex    = (s.nodeIndex == 0xFF) ? (int16_t)-1 : (int16_t)s.nodeIndex;
    // Format class: skinned (0x02 / 0x06) renders in MODEL frame post-bind-pose
    // bake - the viewer applies its mesh-local 90 deg Z only for rigid
    // sections, which is what we tag with class 0.
    outSec->FormatClass  = (s.vertexFormat == 0x02 || s.vertexFormat == 0x06)
                           ? (int16_t)1 : (int16_t)0;
    memcpy(outSec->BoundsMin, s.posMin, sizeof(s.posMin));
    memcpy(outSec->BoundsMax, s.posMax, sizeof(s.posMax));
    memcpy(outSec->UvMin,     s.uvMin,  sizeof(s.uvMin));
    memcpy(outSec->UvMax,     s.uvMax,  sizeof(s.uvMax));
    outSec->SubmeshCount = s.submeshCount;
    outSec->MaterialIndex = s.materialIndex;

    // Compose the per-section node transform by walking the parent chain from
    // bone[NodeIndex] up to root. Each NodeBlock carries a local translation +
    // rotation in its parent's frame; the composition is:
    //   q_total = q_parent * q_child
    //   t_total = t_parent + q_parent * t_child
    // (i.e. the standard "chain of local transforms" from leaf to root).
    //
    // For sections with no bone (NodeIndex == 0xFF) or when the model has no
    // node hierarchy parsed, fall back to identity translation + identity
    // quaternion. The viewer treats identity as "apply only the mesh-local
    // 90 deg Z", so single-bone forge tags with rotation-free root nodes are
    // unaffected.
    float nodeT[3]   = { 0.0f, 0.0f, 0.0f };
    float nodeQ[4]   = { 0.0f, 0.0f, 0.0f, 1.0f };  // identity
    if (s.nodeIndex != 0xFF && !model->nodes.empty()) {
        // Build the local -> model transform by walking leaf -> root and
        // post-multiplying each parent transform on the LEFT. Iteration cap
        // mirrors Reclaimer's bone limit (skeleton hierarchies in Halo never
        // exceed a few hundred bones; cap at the actual node count to avoid
        // runaway loops on malformed parent indices).
        int idx = (int)s.nodeIndex;
        int hops = 0;
        const int maxHops = (int)model->nodes.size() + 1;
        // Start from the leaf bone's local transform, then keep applying the
        // chain of parents on the LEFT: world = parent * (child).
        if (idx >= 0 && (size_t)idx < model->nodes.size()) {
            nodeT[0] = model->nodes[idx].translation[0];
            nodeT[1] = model->nodes[idx].translation[1];
            nodeT[2] = model->nodes[idx].translation[2];
            nodeQ[0] = model->nodes[idx].rotation[0];
            nodeQ[1] = model->nodes[idx].rotation[1];
            nodeQ[2] = model->nodes[idx].rotation[2];
            nodeQ[3] = model->nodes[idx].rotation[3];

            int parent = model->nodes[idx].parentIndex;
            while (parent >= 0 && (size_t)parent < model->nodes.size() &&
                   ++hops < maxHops)
            {
                const NodeEntry& p = model->nodes[parent];
                // q_total = p.q * nodeQ
                const float px = p.rotation[0], py = p.rotation[1],
                            pz = p.rotation[2], pw = p.rotation[3];
                const float cx = nodeQ[0],      cy = nodeQ[1],
                            cz = nodeQ[2],      cw = nodeQ[3];
                const float qx = pw * cx + px * cw + py * cz - pz * cy;
                const float qy = pw * cy + py * cw + pz * cx - px * cz;
                const float qz = pw * cz + pz * cw + px * cy - py * cx;
                const float qw = pw * cw - px * cx - py * cy - pz * cz;
                // t_total = p.t + p.q * nodeT
                // (rotate nodeT by p.q using v' = v + 2*cross(u, cross(u,v) + w*v))
                const float ux = px, uy = py, uz = pz;
                const float vx = nodeT[0], vy = nodeT[1], vz = nodeT[2];
                const float t1x = 2.0f * (uy * vz - uz * vy);
                const float t1y = 2.0f * (uz * vx - ux * vz);
                const float t1z = 2.0f * (ux * vy - uy * vx);
                const float rx = vx + pw * t1x + (uy * t1z - uz * t1y);
                const float ry = vy + pw * t1y + (uz * t1x - ux * t1z);
                const float rz = vz + pw * t1z + (ux * t1y - uy * t1x);
                nodeT[0] = p.translation[0] + rx;
                nodeT[1] = p.translation[1] + ry;
                nodeT[2] = p.translation[2] + rz;
                nodeQ[0] = qx; nodeQ[1] = qy; nodeQ[2] = qz; nodeQ[3] = qw;

                parent = p.parentIndex;
            }
        }
    }
    outSec->NodeTranslation[0] = nodeT[0];
    outSec->NodeTranslation[1] = nodeT[1];
    outSec->NodeTranslation[2] = nodeT[2];
    outSec->NodeRotation[0]    = nodeQ[0];
    outSec->NodeRotation[1]    = nodeQ[1];
    outSec->NodeRotation[2]    = nodeQ[2];
    outSec->NodeRotation[3]    = nodeQ[3];
    return true;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionGeometry(
    ZH_ModelHandle h, uint32_t sectionIndex,
    uint8_t** outVertexBytes, uint32_t* outVertexBytesLen,
    uint8_t** outIndexBytes,  uint32_t* outIndexBytesLen)
{
    if (!outVertexBytes || !outVertexBytesLen || !outIndexBytes || !outIndexBytesLen)
        return false;
    *outVertexBytes = nullptr;
    *outVertexBytesLen = 0;
    *outIndexBytes = nullptr;
    *outIndexBytesLen = 0;

    ModelData* model = LookupModel(h);
    if (!model) return false;
    bool ok = SehDecodeGeometry(model, sectionIndex,
                                outVertexBytes, outVertexBytesLen,
                                outIndexBytes,  outIndexBytesLen);
    if (!ok) {
        if (*outVertexBytes) { free(*outVertexBytes); *outVertexBytes = nullptr; }
        if (*outIndexBytes)  { free(*outIndexBytes);  *outIndexBytes = nullptr; }
        *outVertexBytesLen = 0;
        *outIndexBytesLen  = 0;
    }
    return ok;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionUVs(
    ZH_ModelHandle h, uint32_t sectionIndex,
    float** outUvFloat2, uint32_t* outUvFloatCount)
{
    if (!outUvFloat2 || !outUvFloatCount) return false;
    *outUvFloat2 = nullptr;
    *outUvFloatCount = 0;

    ModelData* model = LookupModel(h);
    if (!model) return false;
    bool ok = SehDecodeUVs(model, sectionIndex, outUvFloat2, outUvFloatCount);
    if (!ok) {
        if (*outUvFloat2) { free(*outUvFloat2); *outUvFloat2 = nullptr; }
        *outUvFloatCount = 0;
    }
    return ok;
}

// Decode per-vertex normals (float3[vertexCount]). Mirrors DecodeSectionUVs.
// Used by the decorator-template path so each instanced blade carries the
// render_model's authored normal. Caller frees with ZH_MMP_FreeBuffer.
extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionNormals(
    ZH_ModelHandle h, uint32_t sectionIndex,
    float** outNrmFloat3, uint32_t* outNrmFloatCount)
{
    if (!outNrmFloat3 || !outNrmFloatCount) return false;
    *outNrmFloat3 = nullptr;
    *outNrmFloatCount = 0;

    ModelData* model = LookupModel(h);
    if (!model) return false;
    bool ok = SehDecodeNormals(model, sectionIndex, outNrmFloat3, outNrmFloatCount);
    if (!ok) {
        if (*outNrmFloat3) { free(*outNrmFloat3); *outNrmFloat3 = nullptr; }
        *outNrmFloatCount = 0;
    }
    return ok;
}

// Decode the section's per-vertex COLOR stream (float3[vertexCount], RGB in [0,1]).
// Returns false (no alloc) when the section has no color stream. Caller frees with ZH_MMP_FreeBuffer.
extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionColors(
    ZH_ModelHandle h, uint32_t sectionIndex,
    float** outColFloat3, uint32_t* outColFloatCount)
{
    if (!outColFloat3 || !outColFloatCount) return false;
    *outColFloat3 = nullptr;
    *outColFloatCount = 0;

    ModelData* model = LookupModel(h);
    if (!model) return false;
    bool ok = SehDecodeColors(model, sectionIndex, outColFloat3, outColFloatCount);
    if (!ok) {
        if (*outColFloat3) { free(*outColFloat3); *outColFloat3 = nullptr; }
        *outColFloatCount = 0;
    }
    return ok;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_MMP_GetSubmeshCount(ZH_ModelHandle h)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0;
    return (uint32_t)model->submeshes.size();
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_GetSubmesh(
    ZH_ModelHandle h, uint32_t i, ZH_ModelSubmesh* outSm)
{
    if (!outSm) return false;
    memset(outSm, 0, sizeof(*outSm));
    ModelData* model = LookupModel(h);
    if (!model || i >= model->submeshes.size()) return false;
    const ModelSubmesh& m = model->submeshes[i];
    outSm->SectionIndex = m.sectionIndex;
    outSm->ShaderIndex  = m.shaderIndex;
    outSm->IndexStart   = m.indexStart;
    outSm->IndexLength  = m.indexLength;
    outSm->Flags        = m.flags;
    return true;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_MMP_GetShaderDiffuseBitmapTagId(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    // Rate-limited diag at the entry. The earlier shader-diag budget is gated
    // inside ResolveDiffuseBitmapTagId, but we never reach it when shaderTagId
    // is -1 - so all those silent failures look identical to "never called".
    // Log the FIRST 8 calls unconditionally so we can see what the viewer
    // is actually requesting and how the early returns trip.
    static std::atomic<int> s_entryDiagBudget{ 8 };
    auto wantLog = [&]() -> bool {
        int v = s_entryDiagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_entryDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
                return true;
        }
        return false;
    };

    ModelData* model = LookupModel(h);
    if (!model) {
        if (wantLog()) NativeDiag("GetShaderDiffuse: bad handle h=0x%llx", (unsigned long long)h);
        return 0xFFFFFFFFu;
    }
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) {
        if (wantLog()) NativeDiag("GetShaderDiffuse: shaderIdx=%d OOB shadersCount=%llu",
            shaderIndex, (unsigned long long)model->shaders.size());
        return 0xFFFFFFFFu;
    }
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) {
        if (wantLog()) NativeDiag("GetShaderDiffuse: shaderIdx=%d shaderTagId=-1 (parsed as missing)",
            shaderIndex);
        return 0xFFFFFFFFu;
    }
    if (wantLog()) NativeDiag("GetShaderDiffuse: shaderIdx=%d shaderTagId=0x%x - resolving...",
        shaderIndex, shaderTagId);
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehResolveDiffuse(cache, shaderIndex, shaderTagId);
}

// sibling to ZH_MMP_GetShaderDiffuseBitmapTagId
// that resolves the shader's `self_illum_map` (or `self_illum_detail_map`)
// bitmap instead of the diffuse pick. Returns 0xFFFFFFFFu when the shader
// doesn't declare an emissive usage. NativeMeshAdapter calls this in
// place of GetShaderDiffuseBitmapTagId for materials whose shader
// constants show `albedo_color ~= 0` AND `self_illum_color` is authored.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_MMP_GetShaderEmissiveBitmapTagId(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFFFFFFFFu;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size())
        return 0xFFFFFFFFu;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFFFFFFFu;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehResolveEmissive(cache, shaderIndex, shaderTagId);
}

// CUTOUT_V1 - resolve a render_model shader's `alpha_test_map`
// bitmap tag id. This is the DEFINITIVE cutout/foliage signal (the same slot
// the BSP path reads via ZH_BSP_GetMaterialBitmapByUsage): Reach authors
// cutout foliage (tree canopy cards, plants) as opaque-blend + a separate
// alpha_test_map, so the shader blend mode can never signal it and the alpha
// histogram of the base map is only a heuristic that misses graded-alpha
// canopies. Returns the bitmap tag id, or 0xFFFFFFFFu when the shader does
// not author an alpha_test_map (i.e. not a cutout material). Reuses the
// generic usage-matching resolver; tiling is irrelevant here so it's ignored.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_MMP_GetShaderAlphaTestBitmapTagId(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFFFFFFFFu;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size())
        return 0xFFFFFFFFu;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFFFFFFFu;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    ModelDetailMapInfo info = SehResolveModelDetail(cache, shaderTagId, "alpha_test_map", 14);
    return info.bitmapTagId;
}

extern "C" __declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderBlendMode(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFF;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) {
        NativeDiag("BlendMode: shaderIndex=%d OUT OF RANGE (count=%zu)",
            shaderIndex, model ? model->shaders.size() : 0);
        return 0xFF;
    }
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) {
        NativeDiag("BlendMode: shader[%d] tagId<0 - no rmsh", shaderIndex);
        return 0xFF;
    }
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFF;
    uint8_t bm = SehResolveBlendMode(cache, shaderTagId);
    NativeDiag("BlendMode: shader[%d] tagId=0x%X => %u (%s)",
        shaderIndex, (unsigned)shaderTagId, (unsigned)bm,
        bm == 0   ? "opaque" :
        bm == 1   ? "additive" :
        bm == 2   ? "multiply" :
        bm == 3   ? "double_multiply/maximum" :
        bm == 4   ? "alpha_blend" :
        bm == 5   ? "add_src_times_srcalpha" :
                    "UNRESOLVED-default-opaque");
    return bm;
}

// MAT-1: per-shader material_model (0..9) for a MODEL shader index; 0xFF unresolved.
// Mirrors ZH_MMP_GetShaderBlendMode. The Rust side maps 0xFF -> default 1
// (cook_torrance, the current unconditional behaviour) so an unresolved read is a
// no-op.
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderMaterialModel(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFF;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return 0xFF;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveMaterialModel(cache, shaderTagId);
}

// LIT-SI-3: per-shader self_illumination MODE (0..12) for a MODEL shader index;
// 0xFF unresolved. Mirrors ZH_MMP_GetShaderMaterialModel / ZH_BSP_GetSelfIllumMode.
// See MapBspParser.cpp::ResolveShaderSelfIllumMode for the enum. Rust maps 0xFF -> 1
// (simple = the current single-composite self-illum path, so unresolved is a no-op).
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderSelfIllumMode(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFF;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return 0xFF;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveSelfIllumMode(cache, shaderTagId);
}

// SKY-3: per-shader sky class for a MODEL shader index. 1 = sky_dome_simple
// (gradient-only, no sampler), 0 = other/textured sky template, 0xFF unresolved.
// Mirrors ZH_MMP_GetShaderMaterialModel. Rust maps 1 -> gradient-only routing.
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderSkyClass(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFF;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return 0xFF;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveSkyClass(cache, shaderTagId);
}

// Cross-TU bridge implemented in MapBspParser.cpp. Forwards to the shared
// SehResolveShaderConstants helper so both BSP- and model-side exports use
// one offset table + one diag budget.
extern "C" bool MapBspParser_ResolveShaderConstants_ForMmp(
    void* cacheHandlePtr, int32_t shaderTagId, int32_t shaderIndexForDiag,
    void* outConstsPtr);

// Helpers consumed by OverlayWalker.cpp without
// duplicating model-handle plumbing. Both exported (rather than inline-named)
// so they have stable linkage when OverlayWalker's TU compiles separately.
extern "C" __declspec(dllexport) int32_t __stdcall
ZH_MMP_GetShaderTagId(ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return -1;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return -1;
    return model->shaders[shaderIndex].shaderTagId;
}

// The shader's tag CLASS packed as 4 little-endian bytes (e.g. 'r','m','g','l'
// -> 0x6C676D72). 0 when unresolved / out of range. Lets the Rust side detect
// glass shaders (rmgl) directly - the blend-mode resolver reads the rmsh layout
// and returns 0xFF (unresolved) for rmgl, so rmgl glass panes can't be told
// apart from other unresolved shaders by blend mode alone.
extern "C" __declspec(dllexport) uint32_t __stdcall
ZH_MMP_GetShaderClass(ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return 0;
    const char* c = model->shaders[shaderIndex].shaderClass;
    return (uint32_t)(uint8_t)c[0]
         | ((uint32_t)(uint8_t)c[1] << 8)
         | ((uint32_t)(uint8_t)c[2] << 16)
         | ((uint32_t)(uint8_t)c[3] << 24);
}

// #285: resolve a MODEL shader's bitmap for a named render-method usage (e.g. "noise_map_a",
// "palette", "alpha_mask_map") - the data the palettized_plasma forcefield shader needs, which
// the diffuse/emissive resolvers never fetch. Reuses the BSP by-usage resolver (defined in
// MapBspParser.cpp) on the model shader's render_method tag id. 0xFFFFFFFF when absent.
uint32_t SehResolveBitmapByUsage(CacheHandle* cache, int32_t shaderTagId, const char* usageNeedle);
uint32_t SehResolveSamplerByUsage(CacheHandle* cache, int32_t shaderTagId, const char* usageNeedle);
// Sampler address-mode byte (u low nibble, v high; 1 = clamp) of a model shader's
// texture constant by usage; 0xFF when absent.
extern "C" __declspec(dllexport) uint32_t __stdcall
ZH_MMP_GetShaderSamplerByUsage(ZH_ModelHandle h, int32_t shaderIndex, const char* usageName)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFFu;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return 0xFFu;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFu;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFFu;
    return SehResolveSamplerByUsage(cache, shaderTagId, usageName);
}
extern "C" __declspec(dllexport) uint32_t __stdcall
ZH_MMP_GetShaderBitmapByUsage(ZH_ModelHandle h, int32_t shaderIndex, const char* usageName)
{
    ModelData* model = LookupModel(h);
    if (!model) return 0xFFFFFFFFu;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return 0xFFFFFFFFu;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFFFFFFFu;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehResolveBitmapByUsage(cache, shaderTagId, usageName);
}

// #250: the rmt2 (render_method_template) tag id for a model shader - its NAME encodes the
// per-category option indices (e.g. shaders\halogram_templates\_2_8_3_0_1_1_0). The Rust side
// resolves the name and parses the self_illum digit (halogram = 2nd number, opaque = 7th) to tell
// change_color (opaque 11/12, halogram 11) / multilayer_additive (halogram 8) from a fixed colour.
// -1 on any break. Walks shader rmsh -> ShaderProperties[0] -> TemplateReference@+12 (same as the
// diffuse resolver).
extern "C" __declspec(dllexport) int32_t __stdcall
ZH_MMP_GetShaderTemplateTagId(ZH_ModelHandle h, int32_t shaderIndex)
{
    ModelData* model = LookupModel(h);
    if (!model) return -1;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return -1;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return -1;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache || (uint32_t)shaderTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return -1;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + OFF_SHADER_PROPS + 8 > cache->size) return -1;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return -1;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 || (size_t)propsOff + 16 > cache->size) return -1;
    const uint8_t* props = cache->base + propsOff;
    uint32_t rmtRaw = (uint32_t)R32(props + 12);
    if (rmtRaw == 0xFFFFFFFFu) return -1;
    return (int32_t)(rmtRaw & 0xFFFFu);
}

extern "C" CacheHandle* MapModelParser_GetCacheForModel(ZH_ModelHandle h)
{
    ModelData* model = LookupModel(h);
    if (!model) return nullptr;
    return LookupHandle(model->cacheHandle);
}

// Per-shader RealConstants + ScalarConstants for render_models. Walks the
// same rmsh -> ShaderProperties[0] chain as the BSP path. See
// MapBspParser.h::ZH_BSP_GetMaterialShaderConstants for the layout.
extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_GetShaderConstants(
    ZH_ModelHandle h, int32_t shaderIndex, ZH_ShaderConstants* outConsts)
{
    if (!outConsts) return false;
    memset(outConsts, 0, sizeof(*outConsts));

    ModelData* model = LookupModel(h);
    if (!model) return false;
    if (shaderIndex < 0 || (size_t)shaderIndex >= model->shaders.size()) return false;
    int32_t shaderTagId = model->shaders[shaderIndex].shaderTagId;
    if (shaderTagId < 0) return false;
    CacheHandle* cache = LookupHandle(model->cacheHandle);
    if (!cache) return false;

    return MapBspParser_ResolveShaderConstants_ForMmp(
        (void*)cache, shaderTagId, shaderIndex, (void*)outConsts);
}

extern "C" __declspec(dllexport) void __stdcall ZH_MMP_FreeBuffer(uint8_t* buf)
{
    if (buf) free(buf);
}

// Drain all open model handles (called from ZH_MMP_PrepareUnload). Used by the
// hot-reload path so the viewer can FreeLibrary cleanly without leaking the
// per-model resource buffers.
extern "C" void MapModelParser_DrainAllHandles()
{
    std::vector<ModelData*> doomed;
    {
        std::lock_guard<std::mutex> lk(g_modelHandlesMutex);
        doomed.reserve(g_modelHandles.size());
        for (auto& kv : g_modelHandles) doomed.push_back(kv.second);
        g_modelHandles.clear();
    }
    for (auto* model : doomed) {
        if (!model) continue;
        if (model->resourceData) free(model->resourceData);
        delete model;
    }
}
