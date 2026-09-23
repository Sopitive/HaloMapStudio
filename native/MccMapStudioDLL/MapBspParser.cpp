// MapBspParser.cpp
// =============================================================================
// Native scenario_structure_bsp (sbsp) decoder for Halo MCC HaloReach .map
// files.
//
// Reach BSP rendering geometry does NOT live in scenario_structure_bsp
// directly. The scnr -> sbsp -> ltmp(scenario_lightmap) -> Lbsp
// (scenario_lightmap_bsp_data) -> Sections / ResourcePointer chain holds the
// real cluster geometry. See Reclaimer's HaloReach/scenario_structure_bsp.cs
// + scenario_lightmap.cs + scenario_lightmap_bsp_data.cs for the schema
// definitions we're porting.
//
// Output model:
//   * Cluster meshes (one per ClusterBlock that resolves to a valid section)
//     are emitted first at world origin (identity transform).
//   * Instance meshes (one per BspGeometryInstanceBlock) follow, with their
//     per-instance transform pre-baked into the position stream. This
//     matches ReclaimerMeshAdapter's walkPermutations:true behaviour.
//
// Schema offsets are MccHaloReach Retail / U3-U10 by default - falling back
// to Reclaimer's fallback offsets via cache->cacheType when needed (U13 is
// covered separately).
//
// Vertex-format coverage (matches MapModelParser):
//   * Format 0x00 / 0x04 - world / flat-world (BSP cluster geometry)
//   * Format 0x01 / 0x05 - rigid / flat-rigid (instance / decorator caps)
//   * Format 0x02 / 0x06 - skinned / flat-skinned (positions only; bone
//                            influences ignored, base-pose mesh shown)
//   * Format 0x0F - decorator (primary stream only)
//   * Other formats - return false; the entry shows up as a magenta
//                            debug material in the viewer.
//
// Stride / packing:
//   The MCC HaloReach .map vertex stream is FLOAT32_4 normalized + UInt16_N2
//   UVs (NOT the legacy Xbox-360 UInt16_N4 packed layout the original BSP
//   parser comment claimed). Strides:
//     * 0x00/0x01/0x04/0x05  -> 36 bytes (Float32_4 pos + UInt16_N2 uv + ...)
//     * 0x02/0x06            -> 44 bytes (rigid + 8 bytes bone data, ignored)
//     * 0x0F (decorator)     -> 32 bytes (Float32_3 pos + Float32_2 uv + n)
//   Position bytes 0..11 are normalized [0,1] floats - dequantize against the
//   per-section bounding-box from BoundingBoxes[i]. UV bytes 0x10..0x13 are
//   UInt16_N2 quantized against the section's UV bounds. Format 0x00 (world
//   cluster geometry) packs UVs as Float16_2 instead - same offset, decoded
//   via HalfToFloat. This mirrors MapModelParser exactly; the previous
//   stride-20 UInt16_N4 layout produced unit-cube positions and broken UVs,
//   which is why every BSP cluster decoded into a black tangled mess.
// =============================================================================

#include "pch.h"
#include <stdio.h>
#include "MapBspParser.h"
#include "MapModelParser.h"   // decorator blade template: ZH_MMP_* render_model decode
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <cstdio>      // snprintf - DECO_QUAT_SWEEP candidate descriptor formatting
#include <algorithm>   // std::sort - DECO_QUAT_SWEEP candidate ranking
#include <atomic>
#include <new>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>

using namespace zh_mcc;

// Forward-declare the runtime decorator mesh struct at file scope so both
// the anonymous namespace implementation and the extern "C" exports can use it.
struct ZH_RuntimeDecoratorMesh {
    float*    Positions;       // malloc'd float3[VertexCount]
    float*    UVs;             // malloc'd float2[VertexCount]
    float*    Normals;         // malloc'd float3[VertexCount] - geometric normal
                               // decoded from +0x14 of the runtime VFMT_DECORATOR
                               // vertex (stream0 NORMAL0, per HREK
                               // decorators.hlsl_include s_decorator_vertex_input).
                               // Used by the viewer for real sun-dir lighting. NOT the
                               // baked RGBE instance_color (that lives in the
                               // separate per-instance stream this path doesn't read).
    uint16_t* Indices;         // malloc'd uint16_t[IndexCount]
    uint32_t  VertexCount;
    uint32_t  IndexCount;
    uint32_t  BitmapTagId;     // dctr.texture -> bitm tag id (0xFFFFFFFFu if unresolved)
    uint32_t  DctrTagId;       // dctr tag id (0xFFFFFFFFu if unresolved)
    float*    Colors;          // DECO_COLOR_RE: malloc'd float3[VertexCount] LINEAR HDR
                               // per-instance baked ambient (instance_color.rgb),
                               // reconstructed by sampling the BSP airprobe SH grid
                               // at each blade's world position. NULL when the
                               // MMS_DECO_COLOR gate is OFF (default) - the viewer then
                               // keeps the normal-only foliage shading.
    float*    Sway;            // DECO_WIND_RE: malloc'd float3[VertexCount]
                               // per-vertex world-space SWAY BASIS vector. Engine
                               // DECORATOR_WAVY (decorators.hlsl_include:172-175) adds
                               //   wave = motion_scale * saturate(abs(vertex.z)) * sin(phase)
                               // to the TEMPLATE-LOCAL x, THEN rotates by the per-instance
                               // quaternion. Since quaternion_transform_point is linear, the
                               // world displacement = quat*(wave,0,0) = quat*(1,0,0) * wave.
                               // We bake Sway = quat*(1,0,0) * (motion_scale * heightGate)
                               // here (per-instance motion_scale = aux byte B1/256, per-
                               // vertex heightGate = saturate(abs(localZ))); the VS only
                               // multiplies by the animated sin(phase). motion_scale~=0
                               // (static ground-cover flowers) -> zero sway; high (grass
                               // blades) -> full sway; root (localZ~=0) planted, tip bends.
                               // ALWAYS emitted (never NULL on a successful decode).
    uint32_t  DecoType;        // DECO_TYPE_RE: per-set decorator class
                               // for the viewer's per-type alpha-test cutoff (FIX: tree
                               // branches' "green outlines" - branches need a HIGHER
                               // cutoff than grass so their AA fringe clips). Derived
                               // from the authored template local-Z height (byte-
                               // verified: tree z~10, bush ~0.57, ground_cover ~0.45,
                               // rock ~0.30, flowers ~0.115):
                               //   0 = ground scatter (grass/flower/ground_cover/rock,
                               //       height < kDecoTallZ) -> LOW cutoff (keep blades)
                               //   1 = tall woody (tree/branch/bush, height >= kDecoTallZ)
                               //       -> HIGH cutoff (clip branch fringe)
                               // Appended at struct tail (ABI-safe; marshalled by
                               // explicit offset in the Rust mirror).
    uint32_t  InstanceCount;   // DECO_BUDGET_RE: decorator instances baked
                               // into THIS mesh (one mesh == one per-cluster group now).
                               // Lands at struct offset 68 (fills DecoType's tail pad, so
                               // the 72-byte stride the Rust FFI reads is UNCHANGED). The
                               // renderer sorts groups by camera distance and draws
                               // nearest-first up to a total instance budget - the engine's
                               // decorator decimation (haloreach.dll sub_1806F9FAC).
};

namespace {

// -----------------------------------------------------------------------------
// Schema offsets
//
// Defaults are MccHaloReach Retail (release through U10). U13 is handled
// separately. These mirror Reclaimer's per-build offset tables.
// -----------------------------------------------------------------------------

struct ScnrLayout {
    int OFF_STRUCTURE_BSPS;
    int OFF_SCENARIO_LIGHTMAP_REF;
};

struct SbspLayout {
    int OFF_CLUSTERS;
    int OFF_SHADERS;
    int OFF_GEOMETRY_INSTANCES;
    int OFF_SECTIONS;
    int OFF_BOUNDING_BOXES;
    int OFF_INSTANCES_RESOURCE_POINTER;
    int CLUSTER_BLOCK_SIZE;
    int CLUSTER_OFFSET_SECTION_INDEX;
    int GEOMETRY_INSTANCE_BLOCK_SIZE;
};

struct LbspLayout {
    int OFF_SECTIONS;
    int OFF_RESOURCE_POINTER;
};

ScnrLayout PickScnrLayout(CacheType ct) {
    ScnrLayout L;
    switch (ct) {
        case CacheType::MccHaloReachU13:
            L.OFF_STRUCTURE_BSPS = 80;
            L.OFF_SCENARIO_LIGHTMAP_REF = 1800;
            break;
        case CacheType::MccHaloReach:
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
        case CacheType::MccHaloReachU10:
        default:
            L.OFF_STRUCTURE_BSPS = 76;
            L.OFF_SCENARIO_LIGHTMAP_REF = 1856;
            break;
    }
    return L;
}

SbspLayout PickSbspLayout(CacheType ct) {
    SbspLayout L;
    // MccHaloReach (release through U10) common defaults.
    L.OFF_CLUSTERS                    = 312;
    L.OFF_SHADERS                     = 324;
    L.OFF_GEOMETRY_INSTANCES          = 612;
    L.OFF_SECTIONS                    = 1128;
    L.OFF_BOUNDING_BOXES              = 1140;
    L.OFF_INSTANCES_RESOURCE_POINTER  = 1336;
    // ClusterBlock for HaloReachRetail+ has FixedSize 140 (drops to 4 SectionIndex
    // earlier); but Reclaimer's MccHaloReach rebases it back to 140 with
    // SectionIndex @ +64 - same as render_model SectionBlock VertexBufferIndex.
    L.CLUSTER_BLOCK_SIZE              = 140;
    L.CLUSTER_OFFSET_SECTION_INDEX    = 64;
    // BspGeometryInstanceBlock for HaloReachRetail+ FixedSize is 4 (just StringId
    // Name @ +0); the per-instance transform/scale/section are stored OUT-OF-LINE
    // in the resource entry's FixupData blob (see ParseGeometryInstances below).
    L.GEOMETRY_INSTANCE_BLOCK_SIZE    = 4;

    switch (ct) {
        case CacheType::MccHaloReachU13:
            L.OFF_GEOMETRY_INSTANCES         = 600;
            L.OFF_SECTIONS                   = 1104;
            L.OFF_BOUNDING_BOXES             = 1116;
            L.OFF_INSTANCES_RESOURCE_POINTER = 1312;
            break;
        default:
            break;
    }
    return L;
}

LbspLayout PickLbspLayout(CacheType ct) {
    LbspLayout L;
    // Reclaimer: HaloReachRetail+ uses Sections @124, ResourcePointer @268.
    // MccHaloReach inherits HaloReachRetail mins.
    L.OFF_SECTIONS         = 124;
    L.OFF_RESOURCE_POINTER = 268;
    (void)ct;
    return L;
}

constexpr int SECTION_BLOCK_SIZE       = 92;
constexpr int BOUNDING_BOX_BLOCK_SIZE  = 52;
constexpr int SUBMESH_BLOCK_SIZE       = 24;
constexpr int SHADER_BLOCK_SIZE        = 44;
constexpr int VERTEX_BUFFER_INFO_SIZE  = 28;
constexpr int INDEX_BUFFER_INFO_SIZE   = 28;

// Tag class codes
constexpr const char* TC_SBSP = "sbsp";
constexpr const char* TC_SCNR = "scnr";

// Bounded-vector helpers (std::vector<float[N]> isn't allowed because C-arrays
// aren't copy-assignable).
struct V3 { float v[3]; };
struct V2 { float v[2]; };

// -----------------------------------------------------------------------------
// In-memory mesh representation
// -----------------------------------------------------------------------------

struct BspSection {
    int16_t  vertexBufferIndex;            // alias for vertexBufferIndices[0]
    int16_t  vertexBufferIndices[8];       // s_mesh.vertex_buffer_indices[8]
                                           // (slot 0 = primary world VB; slots
                                           // 1..7 = parallel streams keyed off
                                           // the same vertex index - UV2,
                                           // tangent_alt, etc. -1 = unbound).
                                           // See SAPIEN_LIGHTMAP_PHASE3_RE.md section 9.
    int16_t  indexBufferIndex;
    uint16_t flags;
    uint8_t  nodeIndex;
    uint8_t  vertexFormat;
    uint8_t  indexFormat;
    uint32_t vbDataLength;
    uint32_t vertexCount;
    uint32_t ibDataLength;
    uint32_t indexCount;
    uint32_t vbResourceOffset;
    uint32_t ibResourceOffset;
    bool     isUnindexed;

    // Resolved UV2 (lightmap UV) VB info - populated during VB hookup. -1
    // when this section has no UV2 stream (rigid/skinned/decorator meshes
    // typically don't, and even some `world` clusters bind only slot 0).
    // The resolver picks the slot in {1..7} where vbStrides[idx]==4 AND
    // vbCounts[idx]==primary.vertexCount (see section 9.5/9.6 of the spec).
    int16_t  uv2VbIndex;
    uint32_t uv2VbResourceOffset;
    uint32_t uv2VbDataLength;
    uint32_t uv2VbStride;
    uint32_t uv2VertexCount;

    // Bounds for de-quantising positions / UVs. Reach BSPs use one bounding
    // box per section (BoundingBoxes[sectionIndex]), unlike render_model which
    // uses BoundingBoxes[0] for all sections. In practice some BSPs only have
    // one bbox - we fall back to bbox[0] when section index is OOB.
    float    posMin[3];
    float    posMax[3];
    float    uvMin[2];
    float    uvMax[2];

    // Submeshes are stored flat in BspData::submeshes
    uint32_t submeshStart;
    uint32_t submeshCount;
    int32_t  materialIndex;
};

struct BspSubmesh {
    uint32_t sectionIndex;
    int32_t  shaderIndex;
    uint32_t indexStart;
    uint32_t indexLength;
};

struct ShaderEntry {
    int32_t  shaderTagId;
};

// Per-output-mesh metadata. Cluster meshes have isInstance=false and an
// identity transform; instance meshes carry a per-instance transform that's
// already been applied to their decoded positions.
//
// Reach BSP cluster sections often pack multiple materials into ONE section
// - wall + window + trim might share a section but each gets its own
// submesh with its own shaderIndex + index range. We emit ONE BspMesh per
// SUBMESH so the renderer can apply per-submesh materials. submeshIndexInSec
// is the sub's index within the section's submesh list (NOT a global
// index into BspData::submeshes).
struct BspMesh {
    uint32_t sectionIndex;     // index into BspData::sections
    uint32_t submeshIndexInSec; // 0..section.submeshCount-1; controls the slice
    bool     isInstance;
    int32_t  materialIndex;    // submesh's shaderIndex (NOT section.materialIndex anymore)
    float    transform[16];    // applied to positions during decode
    float    uniformScale;
    // Bounds AFTER transform application (instance meshes' AABB is the rotated/scaled section bbox).
    float    posMin[3];
    float    posMax[3];
    float    uvMin[2];
    float    uvMax[2];
    // For instance meshes: the index into the sbsp's GeometryInstance list
    // (which the engine uses to look up the per-instance lightmap UV2 VB
    // in the Lbsp tag at offset 0xF4). 0xFFFFFFFFu for cluster meshes.
    uint32_t instanceOrdinal;
    // Owning SBSP cluster index for cluster meshes (the clusters[] loop's
    // `i`). 0xFFFFFFFF for instances. Lets the viewer's Lbsp lookup index
    // Lbsp.clusters[] by the real cluster index.
    uint32_t lightmapClusterIndex;
};

struct BspData {
    uint64_t       cacheHandle;
    uint32_t       sbspTagId;
    int32_t        resourceIndex;        // sbsp's InstancesResourcePointer (for instance metadata)
    int32_t        lbspResourceIndex;    // gestalt entry holding the lightmap geometry

    std::vector<BspSection>    sections;
    std::vector<BspSubmesh>    submeshes;
    std::vector<ShaderEntry>   shaders;
    std::vector<BspMesh>       meshes;

    // Decoded resource page payload - owned by this bsp.
    uint8_t*       resourceData = nullptr;
    size_t         resourceSize = 0;

    // VB pool parsed from the Lbsp resource page's fixup region. Indexed
    // by VB-index - the same index space the per-section VBs use AND
    // the same space the Per-Instance Lightmap Texcoords block at
    // Lbsp+0xF4 emits. The per-instance PVL fetch reads from this pool:
    // given an instance's `vbIndex` (resolved by ExtractInstanceUV2VbIndex),
    // the bytes for that VB live at
    // `resourceData + lbspVbFixupOffsets[vbIndex]` for
    // `lbspVbLens[vbIndex]` bytes, with `lbspVbStrides[vbIndex]` per vertex
    // and `lbspVbCounts[vbIndex]` total vertices.
    //
    // Previously these were local vectors scoped to BuildInner and
    // dropped on the floor after the section-hookup loop finished. Lifting
    // them to BspData so the per-instance decoder can index into the same
    // pool without re-parsing the fixup region.
    std::vector<uint32_t> lbspVbCounts;
    std::vector<uint32_t> lbspVbLens;
    std::vector<uint32_t> lbspVbStrides;
    std::vector<uint32_t> lbspVbFixupOffsets;

    std::mutex     decodeMutex;
};

// Handle table for ZH_BspHandle.
std::mutex g_bspHandlesMutex;
std::unordered_map<uint64_t, BspData*> g_bspHandles;
std::atomic<uint64_t> g_nextBspHandle{ 1 };

BspData* LookupBsp(ZH_BspHandle h) {
    std::lock_guard<std::mutex> lk(g_bspHandlesMutex);
    auto it = g_bspHandles.find(h);
    return it == g_bspHandles.end() ? nullptr : it->second;
}

// -----------------------------------------------------------------------------
// Tag-reference + scenario walk helpers
// -----------------------------------------------------------------------------

// Read a TagReference's TagId. TagReference (Reclaimer Gen3+) layout:
//   ClassId @ +0, padding @ +4..11, TagId @ +12
// Reach's TagId is a 32-bit identifier where the high bits are engine
// identity/generation salt and the low 16 bits are the index into the tag
// table. Reclaimer's TagReference.TagId is `(short)(tagId & ushort.MaxValue)`
// - null only when the FULL 32-bit value is 0xFFFFFFFF. Don't reject on
// rawId < 0 (perfectly valid for high-bit identities like 0xA60C3056).
// Returns -1 only for the genuine null sentinel.
int32_t ReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// -----------------------------------------------------------------------------
// Section decode helpers - reused from MapModelParser.
// -----------------------------------------------------------------------------

bool ParseSubmeshes(CacheHandle* cache, uint32_t submeshesPointer, int32_t submeshesCount,
                    uint32_t sectionIndex, std::vector<BspSubmesh>& outAll)
{
    if (submeshesCount <= 0) return true;
    if (submeshesCount > 0x10000) return false;
    int64_t off = TagMetaFileOff(cache, submeshesPointer);
    if (off < 0 ||
        (size_t)off + (size_t)submeshesCount * SUBMESH_BLOCK_SIZE > cache->size)
        return false;

    for (int i = 0; i < submeshesCount; ++i) {
        const uint8_t* sm = cache->base + off + i * SUBMESH_BLOCK_SIZE;
        BspSubmesh m;
        m.sectionIndex = sectionIndex;
        m.shaderIndex  = R16(sm + 0);
        m.indexStart   = (uint32_t)R32(sm + 4);
        m.indexLength  = (uint32_t)R32(sm + 8);
        {
            // MMS_PART_DIAG=1: raw part record bytes (print-only debugging aid, #condemned-fx).
            static int s_partDiag = -1;
            if (s_partDiag < 0) { char dv[8]; s_partDiag = GetEnvironmentVariableA("MMS_PART_DIAG", dv, sizeof(dv)) ? 1 : 0; }
            if (s_partDiag == 1) {
                fprintf(stderr, "[PART] sec=%u sub=%d shader=%d bytes=", sectionIndex, i, m.shaderIndex);
                for (int b = 0; b < SUBMESH_BLOCK_SIZE; ++b) fprintf(stderr, "%02x", sm[b]);
                fprintf(stderr, "\n");
            }
        }
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
    for (int i = 0; i < shadersCount; ++i) {
        const uint8_t* sb = cache->base + off + i * SHADER_BLOCK_SIZE;
        out[i].shaderTagId = ReadTagRefId(sb);
    }
    return true;
}

// Reads vertex/index buffer info arrays from the resource entry's fixup
// region (located inside the gestalt's FixupData blob). Same as MapModelParser.
//
// vertexStrides (UV2 multi-stream support): the per-VB
// byte stride. Computed as `vertexDataLengths[i] / vertexCounts[i]` when
// count > 0 (else 0). Used by the UV2 stream resolver - slot K is the
// lightmap UV stream when stride==4 (Float16x2 packed).
//
// TODO: the 28-byte VertexBufferInfo record may carry stride directly
// (e.g. via declaration_type at +4) - see SAPIEN_LIGHTMAP_PHASE3_RE.md section 9.5.
// The dataLength/count derivation is the safe path for now; if a real-cache
// dump shows count==0 entries with non-zero stride we'll need to read the
// 12-byte stride struct that immediately follows the VB info array (see
// MapModelParser.cpp:436-437 "Skip 12-byte stride structs per VB").
bool ParseFixupRegion(CacheHandle* cache, const ResourceEntry& entry,
                      std::vector<uint32_t>& vertexCounts,
                      std::vector<uint32_t>& vertexDataLengths,
                      std::vector<uint32_t>& vertexStrides,
                      std::vector<uint8_t>&  indexFormats,
                      std::vector<uint32_t>& indexDataLengths)
{
    // Per-cache diag budget - log first 4 calls into here.
    static std::atomic<int> s_fixupDiagBudget{ 4 };
    bool logThis = false;
    {
        int v = s_fixupDiagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_fixupDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            { logThis = true; break; }
        }
    }

    int zoneIdx = FindGlobalTag(cache, "zone");
    if (zoneIdx < 0) {
        if (logThis) NativeDiag("BspFixupRegion: no zone tag");
        return false;
    }
    int64_t metaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + 350 > cache->size) {
        if (logThis) NativeDiag("BspFixupRegion: bad zone meta off=%lld",
            (long long)metaOff);
        return false;
    }
    const uint8_t* meta = cache->base + metaOff;

    int32_t fixupSize  = R32(meta + 328);
    uint32_t fixupPtrR = RU32(meta + 340);
    int64_t fixupOff   = TagMetaFileOff(cache, fixupPtrR);
    if (fixupSize < 0 || fixupOff < 0) {
        if (logThis) NativeDiag("BspFixupRegion: bad fixup data fsz=%d ptrR=0x%x off=%lld",
            fixupSize, fixupPtrR, (long long)fixupOff);
        return false;
    }
    if ((size_t)fixupOff + (size_t)fixupSize > cache->size) {
        if (logThis) NativeDiag("BspFixupRegion: fixup OOB off=%lld sz=%d",
            (long long)fixupOff, fixupSize);
        return false;
    }
    const uint8_t* fixupBase = cache->base + fixupOff;

    if (entry.fixupSize < 24) {
        if (logThis) NativeDiag("BspFixupRegion: entry.fixupSize<24 fOff=%d fSz=%d",
            entry.fixupOffset, entry.fixupSize);
        return false;
    }
    int64_t trailerOff = (int64_t)entry.fixupOffset + (int64_t)entry.fixupSize - 24;
    if (trailerOff < 0 || (size_t)trailerOff + 16 > (size_t)fixupSize) {
        if (logThis) NativeDiag("BspFixupRegion: trailer OOB toff=%lld fSz=%d gestaltFsz=%d",
            (long long)trailerOff, entry.fixupSize, fixupSize);
        return false;
    }
    int32_t vbCount = R32(fixupBase + trailerOff);
    int32_t ibCount = R32(fixupBase + trailerOff + 12);
    if (vbCount < 0 || vbCount > 0x10000) {
        if (logThis) NativeDiag("BspFixupRegion: bad vbCount=%d entry.foff=%d fsz=%d",
            vbCount, entry.fixupOffset, entry.fixupSize);
        return false;
    }
    if (ibCount < 0 || ibCount > 0x10000) {
        if (logThis) NativeDiag("BspFixupRegion: bad ibCount=%d entry.foff=%d fsz=%d",
            ibCount, entry.fixupOffset, entry.fixupSize);
        return false;
    }

    size_t cursor = (size_t)entry.fixupOffset;
    if (cursor + (size_t)vbCount * VERTEX_BUFFER_INFO_SIZE > (size_t)fixupSize) {
        if (logThis) NativeDiag("BspFixupRegion: vb arr OOB cursor=%llu vbCount=%d gestaltFsz=%d",
            (unsigned long long)cursor, vbCount, fixupSize);
        return false;
    }

    vertexCounts.resize(vbCount);
    vertexDataLengths.resize(vbCount);
    vertexStrides.assign((size_t)vbCount, 0u);
    for (int i = 0; i < vbCount; ++i) {
        const uint8_t* p = fixupBase + cursor + i * VERTEX_BUFFER_INFO_SIZE;
        vertexCounts[i]      = (uint32_t)R32(p + 0);
        vertexDataLengths[i] = (uint32_t)R32(p + 8);
        // Derive stride from (dataLength / count). Robust: doesn't depend on
        // knowing where declaration_type lives in the 28-byte record. UV2 VB
        // is stride==4 (Float16x2 only). Primary world VB is stride==36.
        if (vertexCounts[i] > 0)
            vertexStrides[i] = vertexDataLengths[i] / vertexCounts[i];
    }
    cursor += (size_t)vbCount * VERTEX_BUFFER_INFO_SIZE;

    cursor += (size_t)vbCount * 12;
    if (cursor + (size_t)ibCount * INDEX_BUFFER_INFO_SIZE > (size_t)fixupSize) {
        if (logThis) NativeDiag("BspFixupRegion: ib arr OOB cursor=%llu ibCount=%d gestaltFsz=%d",
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
    if (logThis) {
        uint32_t vc0 = vbCount > 0 ? vertexCounts[0] : 0;
        uint32_t vd0 = vbCount > 0 ? vertexDataLengths[0] : 0;
        uint32_t vc1 = vbCount > 1 ? vertexCounts[1] : 0;
        uint32_t vd1 = vbCount > 1 ? vertexDataLengths[1] : 0;
        NativeDiag("BspFixupRegion: OK vbCount=%d ibCount=%d trailerOff=%lld "
                   "(vb[0] vc=%u dl=%u; vb[1] vc=%u dl=%u)",
            vbCount, ibCount, (long long)trailerOff, vc0, vd0, vc1, vd1);
    }
    return true;
}

// -----------------------------------------------------------------------------
// SBSP + lightmap chain parse
//
//   sbsp -> ScenarioLightmapReference (TagReference @ ScnrLayout offset, but
//           actually we get the lightmap via a different walk: scnr ->
//           StructureBsps[i] -> sbsp_tag_id; we already know the sbsp tag, so
//           we go scnr -> ScenarioLightmapReference -> scenario_lightmap ->
//           LightmapRefs[bspIndex].LightmapDataReference -> Lbsp).
//
// Step-by-step:
//   1. Find scnr (global tag).
//   2. Read scnr.StructureBsps -> walk entries to find one whose BspReference
//      tag id == sbspTagId. This gives us bspIndex.
//   3. Read scnr.ScenarioLightmapReference -> tag id (scenario_lightmap).
//   4. Read scenario_lightmap.LightmapRefs[bspIndex].LightmapDataReference ->
//      tag id (scenario_lightmap_bsp_data).
//   5. Read scenario_lightmap_bsp_data.Sections + ResourcePointer.
//
// On any failure we fall back to using sbsp's own Sections / Clusters
// (cluster-direct path); this won't match Reach's actual rendering but at
// least produces non-empty output.
// -----------------------------------------------------------------------------

bool ResolveLightmapChain(CacheHandle* cache, uint32_t sbspTagId,
                          int* outBspIndex,
                          int32_t* outScenarioLightmapTagId,
                          int32_t* outLbspTagId)
{
    *outBspIndex = -1;
    *outScenarioLightmapTagId = -1;
    *outLbspTagId = -1;

    // Per-cache diag budget - log up to 8 chain attempts so we can see what's
    // happening without flooding the log on a 137-section map. Each log line
    // prints the failure step + relevant offsets.
    static std::atomic<int> s_chainDiagBudget{ 8 };
    bool logThis = false;
    {
        int v = s_chainDiagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_chainDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            { logThis = true; break; }
        }
    }

    int scnrIdx = FindGlobalTag(cache, TC_SCNR);
    if (scnrIdx < 0) {
        if (logThis) NativeDiag("LbspChain[%u]: no scnr tag", sbspTagId);
        return false;
    }

    int64_t scnrMetaOff = TagMetaFileOff(cache, cache->tags[scnrIdx].metaPointerRaw);
    if (scnrMetaOff < 0) {
        if (logThis) NativeDiag("LbspChain[%u]: bad scnr meta off", sbspTagId);
        return false;
    }

    ScnrLayout SL = PickScnrLayout(cache->cacheType);
    if ((size_t)scnrMetaOff + (size_t)SL.OFF_SCENARIO_LIGHTMAP_REF + 16 > cache->size) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: scnr meta truncated metaOff=%lld lmRef=%d size=%llu",
            sbspTagId, (long long)scnrMetaOff, SL.OFF_SCENARIO_LIGHTMAP_REF,
            (unsigned long long)cache->size);
        return false;
    }
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    // 2. Walk StructureBsps[].BspReference to find bspIndex.
    TagBlockRef bspsBlk = ReadTagBlock(scnrMeta + SL.OFF_STRUCTURE_BSPS);
    if (bspsBlk.count <= 0 || bspsBlk.count > 0x10000) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: bad StructureBsps block count=%d ptr=0x%x off=%d ct=%d",
            sbspTagId, bspsBlk.count, bspsBlk.pointer, SL.OFF_STRUCTURE_BSPS,
            (int)cache->cacheType);
        return false;
    }
    int64_t bspsOff = TagMetaFileOff(cache, bspsBlk.pointer);
    constexpr int STRUCTURE_BSP_BLOCK_SIZE = 172;
    if (bspsOff < 0 ||
        (size_t)bspsOff + (size_t)bspsBlk.count * STRUCTURE_BSP_BLOCK_SIZE > cache->size) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: StructureBsps OOB off=%lld count=%d",
            sbspTagId, (long long)bspsOff, bspsBlk.count);
        return false;
    }

    // Mask the queried id to its low 16 bits so high engine-identity bits
    // can't break a direct equality check. ReadTagRefId already returns the
    // masked low-16, but we mirror the mask here to make the comparison
    // explicit and to support a sweep for the BspReference offset within the
    // 172-byte block in case U13 has shifted it (Reclaimer says +0, but the
    // diag below also probes a few candidate offsets just to be sure).
    uint16_t sbspMasked = (uint16_t)(sbspTagId & 0xFFFFu);
    for (int i = 0; i < bspsBlk.count; ++i) {
        const uint8_t* b = cache->base + bspsOff + (size_t)i * STRUCTURE_BSP_BLOCK_SIZE;

        // BspReference @ +0 is the Reclaimer-spec location; check it first.
        int32_t bspRefId   = ReadTagRefId(b);
        uint16_t maskedRef = (bspRefId < 0) ? 0xFFFFu : (uint16_t)(bspRefId & 0xFFFFu);
        // Per-entry raw dump (logThis-budgeted parent guards flooding).
        if (logThis) {
            uint32_t raw00 = RU32(b + 0);
            uint32_t raw04 = RU32(b + 4);
            uint32_t raw0C = RU32(b + 12);
            const char* foundClass = "??";
            char foundClassBuf[5] = {0};
            if (bspRefId >= 0 && (uint32_t)bspRefId < cache->tags.size()) {
                memcpy(foundClassBuf, cache->tags[bspRefId].classCode, 4);
                foundClass = foundClassBuf;
            }
            NativeDiag(
                "LbspChain[%u]: StructureBsps[%d] raw[+0]=0x%08x raw[+4]=0x%08x "
                "raw[+12]=0x%08x bspRefId=%d (masked=0x%04x) class=%s want=0x%04x",
                sbspTagId, i, raw00, raw04, raw0C,
                bspRefId, maskedRef, foundClass, sbspMasked);
        }

        if (bspRefId == (int32_t)sbspTagId || maskedRef == sbspMasked) {
            *outBspIndex = i;
            break;
        }
    }
    if (*outBspIndex < 0) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: bspRef not in scnr.StructureBsps[%d] (queried masked=0x%04x)",
            sbspTagId, bspsBlk.count, sbspMasked);
        return false;
    }

    // 3. ScenarioLightmapReference.
    int32_t lightmapTagId = ReadTagRefId(scnrMeta + SL.OFF_SCENARIO_LIGHTMAP_REF);
    if (lightmapTagId < 0 || (uint32_t)lightmapTagId >= cache->tags.size()) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: bad lightmapTagId=%d (lmRefOff=%d, tags=%llu)",
            sbspTagId, lightmapTagId, SL.OFF_SCENARIO_LIGHTMAP_REF,
            (unsigned long long)cache->tags.size());
        return false;
    }
    *outScenarioLightmapTagId = lightmapTagId;

    // 4. scenario_lightmap.LightmapRefs[bspIndex].LightmapDataReference.
    int64_t lightmapMetaOff = TagMetaFileOff(cache, cache->tags[lightmapTagId].metaPointerRaw);
    if (lightmapMetaOff < 0 || (size_t)lightmapMetaOff + 16 > cache->size) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: bad lightmap meta off=%lld lmTag=%d",
            sbspTagId, (long long)lightmapMetaOff, lightmapTagId);
        return false;
    }
    const uint8_t* lightmapMeta = cache->base + lightmapMetaOff;

    // scenario_lightmap.LightmapRefs @ +4 (BlockCollection<LightmapDataInfoBlock>)
    constexpr int OFF_LIGHTMAP_REFS = 4;
    constexpr int LIGHTMAP_DATA_INFO_BLOCK_SIZE = 32;
    TagBlockRef lmRefsBlk = ReadTagBlock(lightmapMeta + OFF_LIGHTMAP_REFS);
    if (lmRefsBlk.count <= *outBspIndex) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: LightmapRefs.count=%d <= bspIndex=%d (ptr=0x%x)",
            sbspTagId, lmRefsBlk.count, *outBspIndex, lmRefsBlk.pointer);
        return false;
    }
    int64_t lmRefsOff = TagMetaFileOff(cache, lmRefsBlk.pointer);
    if (lmRefsOff < 0 ||
        (size_t)lmRefsOff + (size_t)lmRefsBlk.count * LIGHTMAP_DATA_INFO_BLOCK_SIZE > cache->size) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: LightmapRefs OOB off=%lld count=%d",
            sbspTagId, (long long)lmRefsOff, lmRefsBlk.count);
        return false;
    }

    const uint8_t* lmInfo = cache->base + lmRefsOff +
                            (size_t)(*outBspIndex) * LIGHTMAP_DATA_INFO_BLOCK_SIZE;
    int32_t lbspTagId = ReadTagRefId(lmInfo);
    if (lbspTagId < 0 || (uint32_t)lbspTagId >= cache->tags.size()) {
        if (logThis) NativeDiag(
            "LbspChain[%u]: bad lbspTagId=%d at bspIndex=%d (lmRefsCount=%d)",
            sbspTagId, lbspTagId, *outBspIndex, lmRefsBlk.count);
        return false;
    }
    *outLbspTagId = lbspTagId;

    if (logThis) NativeDiag(
        "LbspChain[%u]: OK bspIndex=%d lmTag=%d lbspTag=%d",
        sbspTagId, *outBspIndex, lightmapTagId, lbspTagId);
    return true;
}

// -----------------------------------------------------------------------------
// Section parse (shared between Lbsp and sbsp-direct fallback).
// -----------------------------------------------------------------------------

bool ParseSections(CacheHandle* cache, uint32_t sectionsPointer, int32_t sectionsCount,
                   const std::vector<V3>& posMins,    // per-section bounds, may be empty (use defaults)
                   const std::vector<V3>& posMaxs,
                   const std::vector<V2>& uvMins,
                   const std::vector<V2>& uvMaxs,
                   std::vector<BspSection>& outSections,
                   std::vector<BspSubmesh>& outSubmeshes)
{
    if (sectionsCount <= 0) return true;
    if (sectionsCount > 0x10000) return false;
    int64_t off = TagMetaFileOff(cache, sectionsPointer);
    if (off < 0 ||
        (size_t)off + (size_t)sectionsCount * SECTION_BLOCK_SIZE > cache->size)
        return false;

    outSections.resize(sectionsCount);
    for (int i = 0; i < sectionsCount; ++i) {
        const uint8_t* s = cache->base + off + (size_t)i * SECTION_BLOCK_SIZE;
        BspSection& sec = outSections[i];

        TagBlockRef submeshesBlk = ReadTagBlock(s + 0);

        // s_mesh.vertex_buffer_indices is an i16[8] array at +0x18 (16 bytes
        // total). Slot 0 is the primary VB; slots 1..7 are parallel streams
        // (UV2/lightmap, tangent_alt, etc). Reach uses -1 (0xFFFF) as the
        // "unbound" sentinel. See SAPIEN_LIGHTMAP_PHASE3_RE.md section 9.3.
        for (int k = 0; k < 8; ++k) {
            sec.vertexBufferIndices[k] = R16(s + 24 + 2 * k);
        }
        sec.vertexBufferIndex = sec.vertexBufferIndices[0];
        sec.indexBufferIndex  = R16(s + 40);
        sec.flags             = (uint16_t)R16(s + 44);
        sec.nodeIndex         = s[46];
        sec.vertexFormat      = s[47];
        sec.indexFormat       = s[50];
        sec.isUnindexed       = (sec.indexBufferIndex == -1) ||
                                ((sec.flags & 0x10) != 0);
        sec.vbDataLength      = 0;
        sec.vertexCount       = 0;
        sec.ibDataLength      = 0;
        sec.indexCount        = 0;
        sec.vbResourceOffset  = 0;
        sec.ibResourceOffset  = 0;
        sec.materialIndex     = -1;
        // UV2 fields - populated later during VB hookup. -1 = no UV2 stream.
        sec.uv2VbIndex          = -1;
        sec.uv2VbResourceOffset = 0;
        sec.uv2VbDataLength     = 0;
        sec.uv2VbStride         = 0;
        sec.uv2VertexCount      = 0;

        // Per-section bounds - Reach BSPs use one bbox per section, parallel
        // to the BoundingBoxes block. Reclaimer iterates BoundingBoxes.Count
        // and assigns to model.Meshes[i] only for i in [0..BoundingBoxes.Count);
        // sections beyond that get NO PositionBounds, which RealBounds3D treats
        // as IsEmpty=true and CreateExpansionMatrix returns Identity
        // (scenario_structure_bsp.cs:115-125, RealBounds3D.cs:16). We mirror
        // that here with identity (0..1) bounds, NOT a fall-back to bbox[0]
        // which would silently apply the wrong section's compression bbox to
        // unrelated geometry.
        size_t bboxIdx = (size_t)i < posMins.size() ? (size_t)i : (size_t)-1;
        if (bboxIdx == (size_t)-1) {
            sec.posMin[0]=0; sec.posMin[1]=0; sec.posMin[2]=0;
            sec.posMax[0]=1; sec.posMax[1]=1; sec.posMax[2]=1;
            sec.uvMin[0]=0;  sec.uvMin[1]=0;
            sec.uvMax[0]=1;  sec.uvMax[1]=1;
        } else {
            memcpy(sec.posMin, posMins[bboxIdx].v, sizeof(sec.posMin));
            memcpy(sec.posMax, posMaxs[bboxIdx].v, sizeof(sec.posMax));
            memcpy(sec.uvMin,  uvMins[bboxIdx].v,  sizeof(sec.uvMin));
            memcpy(sec.uvMax,  uvMaxs[bboxIdx].v,  sizeof(sec.uvMax));
        }

        sec.submeshStart = (uint32_t)outSubmeshes.size();
        ParseSubmeshes(cache, submeshesBlk.pointer, submeshesBlk.count,
                       (uint32_t)i, outSubmeshes);
        sec.submeshCount = (uint32_t)(outSubmeshes.size() - sec.submeshStart);

        if (sec.submeshCount > 0)
            sec.materialIndex = outSubmeshes[sec.submeshStart].shaderIndex;
    }
    return true;
}

bool ParseBoundingBoxes(CacheHandle* cache, uint32_t bboxesPointer, int32_t bboxesCount,
                       std::vector<V3>& posMins, std::vector<V3>& posMaxs,
                       std::vector<V2>& uvMins,  std::vector<V2>& uvMaxs)
{
    if (bboxesCount <= 0) return true;
    if (bboxesCount > 0x10000) return false;
    int64_t off = TagMetaFileOff(cache, bboxesPointer);
    if (off < 0 ||
        (size_t)off + (size_t)bboxesCount * BOUNDING_BOX_BLOCK_SIZE > cache->size)
        return false;

    posMins.resize(bboxesCount);
    posMaxs.resize(bboxesCount);
    uvMins.resize(bboxesCount);
    uvMaxs.resize(bboxesCount);
    for (int i = 0; i < bboxesCount; ++i) {
        const uint8_t* bb = cache->base + off + (size_t)i * BOUNDING_BOX_BLOCK_SIZE;
        memcpy(&posMins[i].v[0], bb + 4,  4); memcpy(&posMaxs[i].v[0], bb + 8,  4);
        memcpy(&posMins[i].v[1], bb + 12, 4); memcpy(&posMaxs[i].v[1], bb + 16, 4);
        memcpy(&posMins[i].v[2], bb + 20, 4); memcpy(&posMaxs[i].v[2], bb + 24, 4);
        memcpy(&uvMins[i].v[0],  bb + 28, 4); memcpy(&uvMaxs[i].v[0],  bb + 32, 4);
        memcpy(&uvMins[i].v[1],  bb + 36, 4); memcpy(&uvMaxs[i].v[1],  bb + 40, 4);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Cluster + GeometryInstance parse from sbsp metadata
// -----------------------------------------------------------------------------

struct ClusterInfo {
    int16_t sectionIndex;
};

bool ParseClusters(CacheHandle* cache, const SbspLayout& SL,
                   uint32_t sbspMetaPtr, std::vector<ClusterInfo>& out)
{
    int64_t metaOff = TagMetaFileOff(cache, sbspMetaPtr);
    if (metaOff < 0 || (size_t)metaOff + SL.OFF_CLUSTERS + 8 > cache->size) return false;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef blk = ReadTagBlock(meta + SL.OFF_CLUSTERS);
    if (blk.count < 0 || blk.count > 0x10000) return false;
    if (blk.count == 0) return true;

    int64_t clOff = TagMetaFileOff(cache, blk.pointer);
    if (clOff < 0 ||
        (size_t)clOff + (size_t)blk.count * SL.CLUSTER_BLOCK_SIZE > cache->size)
        return false;

    out.resize(blk.count);
    for (int i = 0; i < blk.count; ++i) {
        const uint8_t* c = cache->base + clOff +
                           (size_t)i * SL.CLUSTER_BLOCK_SIZE;
        out[i].sectionIndex = R16(c + SL.CLUSTER_OFFSET_SECTION_INDEX);
    }
    return true;
}

// Parse BspGeometryInstanceBlock metadata.
//
// For HaloReachRetail / MccHaloReach builds, the per-instance transform/scale
// /sectionIndex live in the sbsp's InstancesResourcePointer fixup-data blob,
// NOT in the sbsp metadata itself (Reclaimer's scenario_structure_bsp.cs:65-83
// reads them from gestalt.FixupData with stride 156 starting at
// entry.FixupOffset + entry.ResourceFixups[count - 10].Offset & 0x0FFFFFFF).
//
// Layout per Reclaimer:
//   reader.Seek(address + 156 * i);
//   TransformScale = ReadSingle();   // 4
//   Transform      = ReadObject<Matrix4x4>();   // 64 (16 floats)
//   reader.Seek(6, Current);         // skip 6 bytes
//   SectionIndex   = ReadInt16();    // 2
//
// We don't need the Name (StringId) for rendering.
//
// Returns true on success (out filled with `count` instances). Returns false
// if the lookup chain fails - caller treats as "no instances".

struct GeometryInstance {
    float    transform[16];   // row-major 4x4 (or column? - see comment below)
    float    transformScale;
    int16_t  sectionIndex;
};

bool ParseGeometryInstances(CacheHandle* cache, const SbspLayout& SL,
                            uint32_t sbspMetaPtr, int32_t resourceIdRaw,
                            std::vector<GeometryInstance>& out)
{
    int64_t metaOff = TagMetaFileOff(cache, sbspMetaPtr);
    if (metaOff < 0 || (size_t)metaOff + SL.OFF_GEOMETRY_INSTANCES + 8 > cache->size)
        return false;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef blk = ReadTagBlock(meta + SL.OFF_GEOMETRY_INSTANCES);
    if (blk.count <= 0 || blk.count > 0x100000) return true;  // 0 instances OK

    if (resourceIdRaw == 0 || resourceIdRaw == -1) return false;
    int resourceIndex = resourceIdRaw & 0xFFFF;
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size())
        return false;

    {
        std::lock_guard<std::mutex> lk(cache->parseMutex);
        if (!EnsureResourceFixups(cache, (size_t)resourceIndex)) return false;
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];

    // Find FixupData blob pointer in the gestalt (same as ParseFixupRegion).
    int zoneIdx = FindGlobalTag(cache, "zone");
    if (zoneIdx < 0) return false;
    int64_t zoneMetaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
    if (zoneMetaOff < 0 || (size_t)zoneMetaOff + 350 > cache->size) return false;
    const uint8_t* zoneMeta = cache->base + zoneMetaOff;
    int32_t fixupSize  = R32(zoneMeta + 328);
    uint32_t fixupPtrR = RU32(zoneMeta + 340);
    int64_t fixupOff   = TagMetaFileOff(cache, fixupPtrR);
    if (fixupSize < 0 || fixupOff < 0) return false;
    if ((size_t)fixupOff + (size_t)fixupSize > cache->size) return false;
    const uint8_t* fixupBase = cache->base + fixupOff;

    if (entry.fixups.size() < 10) return false;  // need at least count-10 entry
    size_t fixupArrIdx = entry.fixups.size() - 10;
    int32_t fixupArrOff = entry.fixups[fixupArrIdx].offset & 0x0FFFFFFF;
    int64_t address = (int64_t)entry.fixupOffset + (int64_t)fixupArrOff;
    if (address < 0) return false;

    constexpr int INSTANCE_STRIDE = 156;
    if ((size_t)address + (size_t)blk.count * INSTANCE_STRIDE > (size_t)fixupSize)
        return false;

    out.resize(blk.count);
    for (int i = 0; i < blk.count; ++i) {
        const uint8_t* p = fixupBase + address + (size_t)i * INSTANCE_STRIDE;
        memcpy(&out[i].transformScale, p + 0, 4);
        // The on-disk transform is a 4x3 affine (12 floats = 48 bytes), NOT a
        // System.Numerics.Matrix4x4. Layout (matches Reach's AffineTransform):
        //   floats[0..2]  = rotation row 0 (X-axis after rotation)
        //   floats[3..5]  = rotation row 1 (Y-axis after rotation)
        //   floats[6..8]  = rotation row 2 (Z-axis after rotation)
        //   floats[9..11] = translation (x, y, z)
        // We pack into our float[16] as a row-major 4x4 (with the implicit 4th
        // column = 0,0,0,1) so ApplyTransform's existing math works:
        //   M[0..2]   = rot_row0,    M[3]  = 0
        //   M[4..6]   = rot_row1,    M[7]  = 0
        //   M[8..10]  = rot_row2,    M[11] = 0
        //   M[12..14] = translation, M[15] = 1
        float a[12];
        memcpy(a, p + 4, 48);
        out[i].transform[0]  = a[0];  out[i].transform[1]  = a[1];  out[i].transform[2]  = a[2];  out[i].transform[3]  = 0;
        out[i].transform[4]  = a[3];  out[i].transform[5]  = a[4];  out[i].transform[6]  = a[5];  out[i].transform[7]  = 0;
        out[i].transform[8]  = a[6];  out[i].transform[9]  = a[7];  out[i].transform[10] = a[8];  out[i].transform[11] = 0;
        out[i].transform[12] = a[9];  out[i].transform[13] = a[10]; out[i].transform[14] = a[11]; out[i].transform[15] = 1;
        // SectionIndex sits after the 48-byte matrix + 6 skip-bytes (per
        // Reclaimer's reader sequence). The "skip 6" is actually a 32-bit
        // pad + 16-bit pad slot we don't care about; SectionIndex is int16.
        memcpy(&out[i].sectionIndex, p + 4 + 48 + 6, 2);
    }

    // Diagnostic: dump the first 4 instance transforms so we can see whether
    // the matrix fields are populated and where the translation lives. If all
    // instances have identity matrices + zero translation, the on-disk layout
    // doesn't match Reclaimer's 4-byte-scale + 64-byte-matrix4x4 stride.
    static std::atomic<int> s_instDiagBudget{ 4 };
    int diagN = 0;
    while (diagN < (int)out.size() && diagN < 8) {
        int v = s_instDiagBudget.load(std::memory_order_relaxed);
        if (v <= 0) break;
        if (!s_instDiagBudget.compare_exchange_weak(v, v - 1,
            std::memory_order_relaxed, std::memory_order_relaxed)) continue;
        const float* M = out[diagN].transform;
        NativeDiag(
            "BspInstance[%d]: scale=%g sec=%d M=[%g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g]",
            diagN, out[diagN].transformScale, (int)out[diagN].sectionIndex,
            M[0], M[1], M[2], M[3],
            M[4], M[5], M[6], M[7],
            M[8], M[9], M[10], M[11],
            M[12], M[13], M[14], M[15]);
        ++diagN;
    }

    return true;
}

// -----------------------------------------------------------------------------
// Vertex-format decode (formats 0x00 and 0x01).
//
// Mirrors MapModelParser. Both formats use the same stride-20 UInt16 layout;
// anything else falls back.
// -----------------------------------------------------------------------------

constexpr uint32_t VFMT_WORLD          = 0x00;
constexpr uint32_t VFMT_RIGID          = 0x01;
constexpr uint32_t VFMT_SKINNED        = 0x02;
constexpr uint32_t VFMT_FLAT_WORLD     = 0x04;
constexpr uint32_t VFMT_FLAT_RIGID     = 0x05;
constexpr uint32_t VFMT_FLAT_SKINNED   = 0x06;
constexpr uint32_t VFMT_DECORATOR      = 0x0F;

// Matches MapModelParser STRIDE_* constants exactly.
constexpr uint32_t STRIDE_WORLD_RIGID  = 0x24;  // 36 bytes
constexpr uint32_t STRIDE_SKINNED      = 0x2C;  // 44 bytes
constexpr uint32_t STRIDE_DECORATOR    = 0x20;  // 32 bytes

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

// IEEE 754 binary16 -> binary32. Local copy of MapModelParser::HalfToFloat
// so the BSP TU stays self-contained.
inline float Bsp_HalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -1;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            uint32_t fexp = (uint32_t)(127 + (-14 + e));
            bits = (sign << 31) | (fexp << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        uint32_t fexp = (uint32_t)((int)exp - 15 + 127);
        bits = (sign << 31) | (fexp << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Position decode - Float32_4 normalized at +0x00 (xyz used) for the rigid /
// world / skinned families; Float32_3 normalized at +0x00 for decorator.
// Mirrors MapModelParser::DecodeRigidPositions (the on-disk packing is
// identical; BSP and render_model both produce normalized [0,1] floats that
// must be dequantized via the per-section AABB).
void DecodeBspPositions(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                        uint32_t fmt,
                        const float posMin[3], const float posMax[3],
                        uint8_t* dst /* float3[vertexCount] */)
{
    float* o = reinterpret_cast<float*>(dst);
    // BSP_RE_PASS3 #G1: WORLD-format verts are already world-space
    // on disk - the engine deform_flat_world applies NO position compression.
    // Decode raw Float32 and ignore the section AABB. (Previously the compression
    // multiply ran for ALL formats and only produced correct output because world
    // sections empirically have an identity (0..1) bbox; a world section landing
    // at an index < bboxCount would get a real bbox multiplied into already-world
    // coordinates and collapse. Gating on format removes that fragility.)
    if (fmt == VFMT_WORLD || fmt == VFMT_FLAT_WORLD) {
        for (uint32_t i = 0; i < vertexCount; ++i) {
            const uint8_t* v = src + (size_t)i * stride;
            memcpy(o + 0, v + 0, 4);
            memcpy(o + 1, v + 4, 4);
            memcpy(o + 2, v + 8, 4);
            o += 3;
        }
        return;
    }
    const float scaleX = posMax[0] - posMin[0];
    const float scaleY = posMax[1] - posMin[1];
    const float scaleZ = posMax[2] - posMin[2];
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

// UV decode - per-format. Mirrors MapModelParser::DecodeRigidUVs:
//   fmt 0x00/0x04 (world):     Float16_2 @ +0x10 (decode via HalfToFloat)
//   fmt 0x01/0x02/0x05/0x06:   UInt16_N2 @ +0x10 (quantize via uvMin/uvMax)
//   fmt 0x0F (decorator):      Float32_2 @ +0x0C (raw floats, no quant)
void DecodeBspUVs(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                  uint32_t fmt,
                  const float uvMin[2], const float uvMax[2],
                  float* dst)
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
            dst[i * 2 + 0] = Bsp_HalfToFloat(hu);
            dst[i * 2 + 1] = Bsp_HalfToFloat(hv);
        }
        return;
    }

    // Rigid / skinned (and their flat variants): UInt16_N2 quantized via
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

// Normal decode - Int16_N4 at +0x14 for world/rigid/skinned/flat variants;
// Float32_3 at +0x14 for decorator (0x0F).
//
// Int16_N4 packs 4 signed 16-bit normalized values into 8 bytes. Each
// component decodes to float via `(int16_t)raw / 32767.0f`. The 4th
// component (w) is a binormal sign bit and is discarded.
//
// The output is a malloc'd float3[vertexCount] array. For instance meshes
// the caller rotates normals by the 3x3 part of the instance transform
// AFTER this call returns (same pattern as positions).
void DecodeBspNormals(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                      uint32_t fmt, float* dst)
{
    constexpr float kInv32767 = 1.0f / 32767.0f;
    if (fmt == VFMT_DECORATOR) {
        // Decorator: Float32_3 at +0x14
        for (uint32_t i = 0; i < vertexCount; ++i) {
            const uint8_t* v = src + (size_t)i * stride;
            float nx, ny, nz;
            memcpy(&nx, v + 0x14, 4);
            memcpy(&ny, v + 0x18, 4);
            memcpy(&nz, v + 0x1C, 4);
            dst[i * 3 + 0] = nx;
            dst[i * 3 + 1] = ny;
            dst[i * 3 + 2] = nz;
        }
        return;
    }
    // World / rigid / skinned (and flat variants): Int16_N4 at +0x14
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const uint8_t* v = src + (size_t)i * stride;
        int16_t nx = (int16_t)RU16(v + 0x14);
        int16_t ny = (int16_t)RU16(v + 0x16);
        int16_t nz = (int16_t)RU16(v + 0x18);
        dst[i * 3 + 0] = (float)nx * kInv32767;
        dst[i * 3 + 1] = (float)ny * kInv32767;
        dst[i * 3 + 2] = (float)nz * kInv32767;
    }
}

// Tangent decode - Int16_N4 at +0x1C for world/rigid/skinned/flat variants.
// Decorator format has no tangent data (zero-fills output).
//
// Same encoding as normals: (int16_t)raw / 32767.0f. The 4th component (w)
// carries the binormal sign - we discard it and output float3 only.
void DecodeBspTangents(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                       uint32_t fmt, float* dst)
{
    constexpr float kInv32767 = 1.0f / 32767.0f;
    if (fmt == VFMT_DECORATOR) {
        // Decorator doesn't have tangent data - zero-fill.
        memset(dst, 0, (size_t)vertexCount * 3 * sizeof(float));
        return;
    }
    // World / rigid / skinned (and flat variants): Int16_N4 at +0x1C
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const uint8_t* v = src + (size_t)i * stride;
        int16_t tx = (int16_t)RU16(v + 0x1C);
        int16_t ty = (int16_t)RU16(v + 0x1E);
        int16_t tz = (int16_t)RU16(v + 0x20);
        dst[i * 3 + 0] = (float)tx * kInv32767;
        dst[i * 3 + 1] = (float)ty * kInv32767;
        dst[i * 3 + 2] = (float)tz * kInv32767;
    }
}

// Binormal decode - computed from normal (Int16_N4 @ +0x14) and tangent
// (Int16_N4 @ +0x1C) using cross(N, T) * sign(tangent_w). The tangent's
// 4th component carries the binormal handedness sign (+1 or -1). Without
// the sign, mirrored UVs (common on symmetric BSP geometry) produce
// flipped bump mapping.
//
// Decorator format has no tangent/binormal - zero-fills output.
void DecodeBspBinormals(const uint8_t* src, uint32_t vertexCount, uint32_t stride,
                        uint32_t fmt, float* dst)
{
    if (fmt == VFMT_DECORATOR) {
        memset(dst, 0, (size_t)vertexCount * 3 * sizeof(float));
        return;
    }
    constexpr float kInv32767 = 1.0f / 32767.0f;
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const uint8_t* v = src + (size_t)i * stride;
        // Normal xyz
        float nx = (float)(int16_t)RU16(v + 0x14) * kInv32767;
        float ny = (float)(int16_t)RU16(v + 0x16) * kInv32767;
        float nz = (float)(int16_t)RU16(v + 0x18) * kInv32767;
        // Tangent xyzw (w = binormal sign)
        float tx = (float)(int16_t)RU16(v + 0x1C) * kInv32767;
        float ty = (float)(int16_t)RU16(v + 0x1E) * kInv32767;
        float tz = (float)(int16_t)RU16(v + 0x20) * kInv32767;
        float tw = (float)(int16_t)RU16(v + 0x22) * kInv32767;
        float sign = (tw >= 0.0f) ? 1.0f : -1.0f;
        // cross(N, T) * sign
        float bx = (ny * tz - nz * ty) * sign;
        float by = (nz * tx - nx * tz) * sign;
        float bz = (nx * ty - ny * tx) * sign;
        // Normalize
        float len = sqrtf(bx*bx + by*by + bz*bz);
        if (len > 1e-8f) { bx /= len; by /= len; bz /= len; }
        dst[i * 3 + 0] = bx;
        dst[i * 3 + 1] = by;
        dst[i * 3 + 2] = bz;
    }
}

// Secondary UV (lightmap UV2) decode.
//
// CORRECTED (per SAPIEN_LIGHTMAP_PHASE3_RE.md section 9): UV2 does NOT
// live inside the primary 36-byte world VB at offset +0x20 - those bytes are
// tangent tail / padding. UV2 is in a SEPARATE vertex buffer bound via
// `s_mesh.vertex_buffer_indices[K]` for some K  in  {1..7}. The UV2 stream's
// stride is 4 bytes (Float16x2 packed at offset 0).
//
// The caller passes the resolved UV2 VB pointer (sec.uv2VbResourceOffset
// already validated to fit inside the resource page) and vertex count. The
// stride is implicitly 4 - the resolver in the section hookup loop only
// picks slots whose stride is exactly 4.
//
// Returns true on success, false if uv2Src is null or vertexCount is 0.
bool DecodeBspUVs2(const uint8_t* uv2Src, uint32_t vertexCount, float* dst)
{
    if (!uv2Src || vertexCount == 0) return false;

    // One-shot diag: dump the UV2 VB's first vertex bytes for the first 4
    // calls so we can verify the bind picked the right stream. Pre-fix this
    // dump showed tangent-tail garbage from offset +0x20 of the primary VB;
    // post-fix it should show valid Float16x2 patterns (u,v in [0,1]).
    static std::atomic<int> s_uv2DumpBudget{ 4 };
    {
        int v = s_uv2DumpBudget.load(std::memory_order_relaxed);
        if (v > 0 && s_uv2DumpBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
        {
            const uint8_t* v0 = uv2Src;
            uint16_t hu = RU16(v0 + 0);
            uint16_t hv = RU16(v0 + 2);
            float fuFloat16 = Bsp_HalfToFloat(hu);
            float fvFloat16 = Bsp_HalfToFloat(hv);
            float fuUnorm = (float)hu * (1.0f / 65535.0f);
            float fvUnorm = (float)hv * (1.0f / 65535.0f);
            NativeDiag("[BSP_UV2_DUMP] vc=%u uv2Stream vert0Bytes=%02X %02X %02X %02X "
                "asFloat16=(%.4f, %.4f) asUNorm=(%.4f, %.4f)",
                vertexCount, v0[0], v0[1], v0[2], v0[3],
                fuFloat16, fvFloat16, fuUnorm, fvUnorm);
        }
    }

    // Read from the dedicated UV2 stream: 4 bytes per vertex.
    // ENCODING: UInt16x2 NORMALIZED (not Float16x2 as the original RE
    // assumed). Verified via the BSP_UV2_DUMP diag - bytes
    // `5C 87 40 AB` decoded as Float16 give (-0.0001, -0.0566) which is
    // bogus; decoded as UNORM (uint16/65535) give (0.529, 0.669) which are
    // valid UV coordinates. This is consistent with how Reach stores
    // diffuse UV in rigid/skinned formats (Reclaimer's MapModelParser
    // line ~1042 decodes those as UInt16_N2 with `(float)ru / 65535.0f`).
    constexpr float kInv65535 = 1.0f / 65535.0f;
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const uint8_t* v = uv2Src + (size_t)i * 4;
        uint16_t hu = RU16(v + 0);
        uint16_t hv = RU16(v + 2);
        dst[i * 2 + 0] = (float)hu * kInv65535;
        dst[i * 2 + 1] = (float)hv * kInv65535;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Index buffer expansion (same as MapModelParser).
// -----------------------------------------------------------------------------

constexpr uint8_t IF_TRI_LIST   = 3;
constexpr uint8_t IF_TRI_STRIP  = 5;
constexpr uint8_t IF_DEFAULT    = 0;

// Strip-to-list with restart-sentinel handling. Mirrors
// MapModelParser::StripToList. The Reach engine emits 0xFFFF / 0xFFFFFFFF
// as a strip-restart sentinel; without honouring it we'd produce "spider
// web" triangles spanning every section back to vertex 0.
template <typename Idx>
size_t StripToList(const uint8_t* src, uint32_t srcCount, uint8_t* dst) {
    if (srcCount < 3) return 0;
    constexpr Idx kRestart = (Idx)~(Idx)0;
    uint32_t outCount = 0;
    uint32_t stripStart = 0;
    for (uint32_t i = 0; i + 2 < srcCount; ++i) {
        Idx a, b, c;
        memcpy(&a, src + (i + 0) * sizeof(Idx), sizeof(Idx));
        memcpy(&b, src + (i + 1) * sizeof(Idx), sizeof(Idx));
        memcpy(&c, src + (i + 2) * sizeof(Idx), sizeof(Idx));
        if (a == kRestart || b == kRestart || c == kRestart) {
            uint32_t skipTo;
            if (a == kRestart)      skipTo = i + 1;
            else if (b == kRestart) skipTo = i + 2;
            else                    skipTo = i + 3;
            i = skipTo - 1;
            stripStart = skipTo;
            continue;
        }
        if (a == b || b == c || a == c) continue;
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

// Per-submesh strip expansion. Caller passes the section, the section's
// submesh slice in `bsp->submeshes`, and the raw strip data; we expand each
// submesh independently (so triangles at submesh boundaries don't stitch
// together). Returns the malloc'd index buffer (full section, all submeshes
// concatenated in submesh order) or nullptr.
//
// `outPostStart` / `outPostLen` (optional, sized >= sec.submeshCount when
// not null) receive the POST-EXPANSION list-relative window for each submesh.
// These replaced an earlier design that mutated `bsp->submeshes[smIndex]`
// in-place - that mutation was a DATA RACE under Parallel.For over meshes:
// two threads decoding different submeshes of the same section concurrently
// would overwrite each other's slice metadata, leading to one or two meshes
// per BSP being rendered with the wrong index window (the "blob mesh" bug,
// see BSP_BLOB_MESH_RE.md). The new contract: DecodeIndices does
// not touch `bsp->submeshes`; callers thread the local arrays through to the
// slice step.
uint8_t* DecodeIndices(BspData* bsp, uint32_t sectionIndex,
                       const uint8_t* indexSrc,
                       uint32_t* outCount, uint32_t* outStride,
                       uint32_t* outPostStart = nullptr,
                       uint32_t* outPostLen   = nullptr)
{
    *outCount  = 0;
    *outStride = 0;
    const BspSection& sec = bsp->sections[sectionIndex];
    uint32_t srcStride = (sec.vertexCount > 0xFFFF) ? 4u : 2u;
    *outStride = srcStride;

    if (sec.isUnindexed) {
        size_t outBytes = (size_t)sec.vertexCount * srcStride;
        uint8_t* buf = (uint8_t*)malloc(outBytes);
        if (!buf) return nullptr;
        if (srcStride == 2) {
            for (uint32_t i = 0; i < sec.vertexCount; ++i) {
                uint16_t v = (uint16_t)i;
                memcpy(buf + i * 2, &v, 2);
            }
        } else {
            for (uint32_t i = 0; i < sec.vertexCount; ++i)
                memcpy(buf + i * 4, &i, 4);
        }
        *outCount = sec.vertexCount;
        if (outPostStart && outPostLen) {
            for (uint32_t si = 0; si < sec.submeshCount; ++si) {
                outPostStart[si] = 0;
                outPostLen[si]   = sec.vertexCount;
            }
        }
        return buf;
    }

    uint32_t srcCount = sec.indexCount;
    if (srcCount == 0 || !indexSrc) return nullptr;

    uint8_t fmt = sec.indexFormat;
    if (fmt == IF_DEFAULT) fmt = IF_TRI_STRIP;
    // #bsp-G1 (engine-exact): primitive type mesh+0x32 maps 3->list(0x15), 5->strip(0x16),
    // 6->(0x17). Type 6 was falling through to `return nullptr` at the bottom -> the entire
    // section produced NO indices and vanished from the render. Type 6 is a triangle-strip
    // variant; decode it as a strip (strictly better than dropping the geometry).
    if (fmt == 6) fmt = IF_TRI_STRIP;

    if (fmt == IF_TRI_LIST) {
        uint32_t outBytes = srcCount * srcStride;
        uint8_t* buf = (uint8_t*)malloc(outBytes);
        if (!buf) return nullptr;
        memcpy(buf, indexSrc, outBytes);
        *outCount = srcCount;
        if (outPostStart && outPostLen) {
            for (uint32_t si = 0; si < sec.submeshCount; ++si) {
                uint32_t smIndex = sec.submeshStart + si;
                const BspSubmesh& sm = bsp->submeshes[smIndex];
                // Tri-list: raw window IS the list window. Bounds-check.
                if (sm.indexStart < srcCount &&
                    sm.indexStart + sm.indexLength <= srcCount) {
                    outPostStart[si] = sm.indexStart;
                    outPostLen[si]   = sm.indexLength;
                } else {
                    outPostStart[si] = 0;
                    outPostLen[si]   = 0;
                }
            }
        }
        return buf;
    }

    if (fmt == IF_TRI_STRIP) {
        if (srcCount < 3) return nullptr;
        // Worst-case bound: per-slice expansion never produces more than
        // (sliceLen - 2) * 3 <= sliceLen * 3 triangles. Allocate the loose
        // bound and shrink later.
        uint32_t maxOut = srcCount * 3;
        uint8_t* buf = (uint8_t*)malloc((size_t)maxOut * srcStride);
        if (!buf) return nullptr;

        uint32_t outIdx = 0;

        if (sec.submeshCount == 0) {
            size_t produced;
            if (srcStride == 2)
                produced = StripToList<uint16_t>(indexSrc, srcCount, buf);
            else
                produced = StripToList<uint32_t>(indexSrc, srcCount, buf);
            *outCount = (uint32_t)produced;
            return buf;
        }

        for (uint32_t si = 0; si < sec.submeshCount; ++si) {
            uint32_t smIndex = sec.submeshStart + si;
            // Read-only access to the raw-strip window stored at parse time.
            // Do NOT mutate this struct - it's shared across decode threads.
            const BspSubmesh& sm = bsp->submeshes[smIndex];

            uint32_t sliceStart = sm.indexStart;
            uint32_t sliceLen   = sm.indexLength;

            if (sliceStart >= srcCount || sliceLen < 3 ||
                sliceStart + sliceLen > srcCount)
            {
                if (outPostStart && outPostLen) {
                    outPostStart[si] = outIdx;
                    outPostLen[si]   = 0;
                }
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

            if (outPostStart && outPostLen) {
                outPostStart[si] = outIdx;
                outPostLen[si]   = (uint32_t)produced;
            }
            outIdx += (uint32_t)produced;
        }

        *outCount = outIdx;
        return buf;
    }

    return nullptr;
}

// -----------------------------------------------------------------------------
// Transform a position by a 4x4 matrix + uniformScale.
//
// Reclaimer's Matrix4x4 (System.Numerics) is row-major with translation in
// M41/M42/M43 (i.e. position * matrix). We apply v_out = (v_in * scale) * M.
// -----------------------------------------------------------------------------

void ApplyTransform(const float in[3], const float M[16], float scale, float out[3]) {
    float vx = in[0] * scale;
    float vy = in[1] * scale;
    float vz = in[2] * scale;
    out[0] = vx * M[0] + vy * M[4] + vz * M[8]  + M[12];
    out[1] = vx * M[1] + vy * M[5] + vz * M[9]  + M[13];
    out[2] = vx * M[2] + vy * M[6] + vz * M[10] + M[14];
}

void IdentityMatrix(float M[16]) {
    memset(M, 0, sizeof(float) * 16);
    M[0] = M[5] = M[10] = M[15] = 1.0f;
}

// -----------------------------------------------------------------------------
// Top-level parse: load all metadata, then build the unified mesh list.
// -----------------------------------------------------------------------------

bool ParseSbspTag(CacheHandle* cache, uint32_t sbspTagId, BspData& data) {
    if (sbspTagId >= cache->tags.size()) {
        NativeDiag("Bsp[%u]: tag id OOB (tagsCount=%llu)",
            sbspTagId, (unsigned long long)cache->tags.size());
        return false;
    }
    const TagEntry& te = cache->tags[sbspTagId];
    if (te.classIndex < 0) { NativeDiag("Bsp[%u]: classIndex<0", sbspTagId); return false; }
    if (memcmp(te.classCode, TC_SBSP, 4) != 0) {
        NativeDiag("Bsp[%u]: not sbsp (class=%c%c%c%c)", sbspTagId,
            te.classCode[0], te.classCode[1], te.classCode[2], te.classCode[3]);
        return false;
    }
    NativeDiag("Bsp[%u]: parse begin name='%s' ct=%d",
        sbspTagId, te.tagName.c_str(), (int)cache->cacheType);

    SbspLayout SL = PickSbspLayout(cache->cacheType);
    LbspLayout LL = PickLbspLayout(cache->cacheType);

    // 1. Resolve the lightmap chain (sbsp -> ltmp -> Lbsp).
    int bspIndex = -1;
    int32_t scenarioLightmapTagId = -1;
    int32_t lbspTagId = -1;
    bool chainOk = ResolveLightmapChain(cache, sbspTagId, &bspIndex,
                                        &scenarioLightmapTagId, &lbspTagId);

    // 2. Sbsp meta - for shaders, clusters, geometry instances,
    //    instances resource pointer.
    int64_t sbspMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (sbspMetaOff < 0 ||
        (size_t)sbspMetaOff + (size_t)SL.OFF_INSTANCES_RESOURCE_POINTER + 4 > cache->size)
        return false;
    const uint8_t* sbspMeta = cache->base + sbspMetaOff;

    // Shaders.
    TagBlockRef shadersBlk = ReadTagBlock(sbspMeta + SL.OFF_SHADERS);
    ParseShaders(cache, shadersBlk.pointer, shadersBlk.count, data.shaders);

    // 3. Choose the geometry source. Prefer Lbsp; fall back to sbsp-direct.
    uint32_t geomSectionsPointer = 0;  int32_t geomSectionsCount = 0;
    uint32_t geomBboxesPointer   = 0;  int32_t geomBboxesCount   = 0;
    int32_t  geomResourceIdRaw   = 0;
    bool usingLbsp = false;

    if (chainOk && lbspTagId >= 0) {
        const TagEntry& lbspTe = cache->tags[lbspTagId];
        int64_t lbspMetaOff = TagMetaFileOff(cache, lbspTe.metaPointerRaw);
        if (lbspMetaOff >= 0 &&
            (size_t)lbspMetaOff + (size_t)LL.OFF_RESOURCE_POINTER + 4 <= cache->size)
        {
            const uint8_t* lbspMeta = cache->base + lbspMetaOff;
            TagBlockRef secBlk = ReadTagBlock(lbspMeta + LL.OFF_SECTIONS);
            geomSectionsPointer = secBlk.pointer;
            geomSectionsCount   = secBlk.count;
            geomResourceIdRaw   = R32(lbspMeta + LL.OFF_RESOURCE_POINTER);
            // Lbsp doesn't carry BoundingBoxes - they live on the sbsp.
            // Use sbsp's BoundingBoxes for de-quantisation.
            TagBlockRef sbspBboxBlk = ReadTagBlock(sbspMeta + SL.OFF_BOUNDING_BOXES);
            geomBboxesPointer = sbspBboxBlk.pointer;
            geomBboxesCount   = sbspBboxBlk.count;
            usingLbsp = true;
        }
    }

    if (!usingLbsp) {
        // Fall back to sbsp's own Sections / BoundingBoxes / InstancesResourcePointer.
        // Reach BSPs don't actually render this way (the sbsp-direct sections
        // are usually empty / shells) but at least we don't crash.
        TagBlockRef sbspSecBlk = ReadTagBlock(sbspMeta + SL.OFF_SECTIONS);
        TagBlockRef sbspBboxBlk = ReadTagBlock(sbspMeta + SL.OFF_BOUNDING_BOXES);
        geomSectionsPointer = sbspSecBlk.pointer;
        geomSectionsCount   = sbspSecBlk.count;
        geomBboxesPointer   = sbspBboxBlk.pointer;
        geomBboxesCount     = sbspBboxBlk.count;
        geomResourceIdRaw   = R32(sbspMeta + SL.OFF_INSTANCES_RESOURCE_POINTER);
    }

    if (geomSectionsCount <= 0) return false;
    int32_t resourceIndex = geomResourceIdRaw & 0xFFFF;
    data.lbspResourceIndex = resourceIndex;

    // 4. BoundingBoxes -> per-section bounds (or fallback to bbox[0] inside ParseSections).
    std::vector<V3> posMins, posMaxs;
    std::vector<V2> uvMins,  uvMaxs;
    if (geomBboxesCount > 0) {
        ParseBoundingBoxes(cache, geomBboxesPointer, geomBboxesCount,
                           posMins, posMaxs, uvMins, uvMaxs);
    }
    NativeDiag(
        "Bsp[%u]: bboxBlk count=%d secsCount=%d (if count<<secsCount, every section falls back to bbox[0])",
        sbspTagId, geomBboxesCount, geomSectionsCount);
    for (size_t bi = 0; bi < posMins.size() && bi < 8; ++bi) {
        NativeDiag(
            "Bsp[%u]: bbox[%llu] posMin=(%.1f,%.1f,%.1f) posMax=(%.1f,%.1f,%.1f) uvMin=(%.3f,%.3f) uvMax=(%.3f,%.3f)",
            sbspTagId, (unsigned long long)bi,
            posMins[bi].v[0], posMins[bi].v[1], posMins[bi].v[2],
            posMaxs[bi].v[0], posMaxs[bi].v[1], posMaxs[bi].v[2],
            uvMins[bi].v[0],  uvMins[bi].v[1],
            uvMaxs[bi].v[0],  uvMaxs[bi].v[1]);
    }

    // 5. Sections + submeshes.
    if (!ParseSections(cache, geomSectionsPointer, geomSectionsCount,
                       posMins, posMaxs, uvMins, uvMaxs,
                       data.sections, data.submeshes))
        return false;

    // 6. Resource fixups -> hook each section to its vb/ib data.
    if (!ParseGestalt(cache)) return false;
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size()) return false;

    {
        std::lock_guard<std::mutex> lk(cache->parseMutex);
        if (!EnsureResourceFixups(cache, (size_t)resourceIndex)) return false;
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];

    // One-shot per-bsp diag of the fixup-entry shape - mirrors the model
    // parser's "Mode[%u]: entry rIdx=%d ..." line so we can compare BSP and
    // model entries side-by-side in the log.
    NativeDiag("Bsp[%u]: entry rIdx=%d rPtr=0x%x fOff=%d fSz=%d segIdx=%d fixups=%d "
               "lbsp=%d resRaw=0x%x",
        sbspTagId, resourceIndex, entry.resourcePointer, entry.fixupOffset,
        entry.fixupSize, entry.segmentIndex, (int)entry.fixups.size(),
        usingLbsp ? 1 : 0, (uint32_t)geomResourceIdRaw);

    std::vector<uint32_t> vbCounts, vbLens, vbStrides, ibLens;
    std::vector<uint8_t>  ibFmts;
    if (!ParseFixupRegion(cache, entry, vbCounts, vbLens, vbStrides, ibFmts, ibLens)) {
        NativeDiag("Bsp[%u]: ParseFixupRegion failed (entry foff=%d fsz=%d)",
            sbspTagId, entry.fixupOffset, entry.fixupSize);
        return false;
    }
    NativeDiag("Bsp[%u]: fixupRegion vb=%d ib=%d secs=%d",
        sbspTagId, (int)vbCounts.size(), (int)ibLens.size(), (int)data.sections.size());

    // Cache the VB pool on BspData so the per-instance PVL fetch
    // (ZH_BSP_GetInstancePvlVb) can index into the
    // same fixup-derived (count, len, stride, fixupOffset) tuples that
    // the section-hookup loop below uses for per-section UV2 resolution.
    // Without this, per-instance VB-index resolution at the viewer
    // would have to re-parse the fixup region or duplicate the load.
    data.lbspVbCounts        = vbCounts;
    data.lbspVbLens          = vbLens;
    data.lbspVbStrides       = vbStrides;
    data.lbspVbFixupOffsets.clear();
    data.lbspVbFixupOffsets.reserve(vbCounts.size());
    for (size_t fi = 0; fi < vbCounts.size(); ++fi) {
        // Fixups for VBs come first in entry.fixups[]; indices [vbCount, vbCount+ibCount)
        // are IB fixups. Same masking pattern as the section path
        // (`entry.fixups[sec.vertexBufferIndex].offset & 0x0FFFFFFF`).
        uint32_t foff = 0;
        if (fi < entry.fixups.size())
            foff = (uint32_t)(entry.fixups[fi].offset & 0x0FFFFFFFu);
        data.lbspVbFixupOffsets.push_back(foff);
    }

    int hookedSecs = 0;
    int firstHookedSecIdx = -1;
    // Diag budget for per-section failure logging - capped at 12 lines per BSP
    // so we don't drown the log on a totally broken parse, but get enough
    // detail to tell whether the failure is OOB-vbIdx, vbInfo[idx].vc==0, or
    // some interleaved-section pattern. The "ground missing in most places"
    // symptom on forge_halo points at sections whose vbIdx is in-range but
    // resolves to vbInfo[idx].vc==0 - those should NOT show up here as
    // "unhooked", they'll just have vertexCount=0 and skip in DecodeMesh.
    int hookFailDiagBudget = 12;
    // BSP_HOOKUP_OOB_RE.md: split the legacy "oobCount" into
    // separate buckets so we can tell the difference between
    //   (a) a true walker bug - vbIdx >= vbCounts.size() (overflowCount),
    //       which should always be 0 on a correctly-parsed BSP; and
    //   (b) the expected per-instance / subpart-merged template stub - 
    //       vbIdx == -1 (sentinelCount). These are NOT a bug; they're
    //       `s_mesh` entries kept around so GeometryInstance.sectionIndex
    //       stays stable while the real geometry has been merged into
    //       another section. See BSP_HOOKUP_OOB_RE.md section 2 + LBSP_INSTANCE_-
    //       BUCKETS_RE.md for the full RE.
    int sentinelCount = 0;   // vbIdx == -1 (per-instance / merged template)
    int overflowCount = 0;   // vbIdx >= 0 but >= vbCounts.size() (real bug)
    int reboundCount  = 0;   // sentinel section recovered via slot 1..7 scan
    int zeroVbCount = 0;
    for (size_t si = 0; si < data.sections.size(); ++si) {
        auto& sec = data.sections[si];
        bool isSentinel = (sec.vertexBufferIndex < 0);
        bool isOverflow = (!isSentinel &&
                           sec.vertexBufferIndex >= (int)vbCounts.size());

        // 6a. Sentinel rebind - try slots 1..7. Some merged-template meshes
        // carry their primary VB in a non-zero slot (the merge target rather
        // than the merge source). Accept the first slot whose VB has the
        // canonical world-vertex shape: non-zero `vc` and stride==36
        // (s_world_vertex). UV2 streams use stride==4 and are explicitly
        // rejected here so we don't accidentally rebind to a UV2 slot.
        // If no slot matches, fall through to the original skip path.
        if (isSentinel) {
            for (int k = 1; k < 8; ++k) {
                int16_t alt = sec.vertexBufferIndices[k];
                if (alt < 0) continue;
                if ((size_t)alt >= vbCounts.size()) continue;
                if (vbCounts[alt] == 0) continue;
                if ((size_t)alt >= vbStrides.size()) continue;
                if (vbStrides[alt] != 36u) continue;   // primary-shape only
                sec.vertexBufferIndex = alt;
                sec.vertexBufferIndices[0] = alt;      // promote for downstream
                isSentinel = false;
                ++reboundCount;
                break;
            }
        }

        if (isSentinel || isOverflow) {
            if (isSentinel) ++sentinelCount;
            else            ++overflowCount;
            if (hookFailDiagBudget > 0) {
                --hookFailDiagBudget;
                // Dump all 8 slots so we can audit whether a future map has
                // a primary VB in a slot we don't recognise (e.g. non-36-byte
                // stride world vert variant). Existing `picked=-1` UV2 diag
                // only fired for the first 4 sections; this one is targeted
                // at the suspicious-skip cases.
                int16_t s1 = sec.vertexBufferIndices[1];
                int16_t s2 = sec.vertexBufferIndices[2];
                int16_t s3 = sec.vertexBufferIndices[3];
                int16_t s4 = sec.vertexBufferIndices[4];
                int16_t s5 = sec.vertexBufferIndices[5];
                int16_t s6 = sec.vertexBufferIndices[6];
                int16_t s7 = sec.vertexBufferIndices[7];
                const char* kind = isSentinel ? "sentinel" : "overflow";
                NativeDiag(
                    "Bsp[%u]: BspSec[%llu] %s vbIdx=%d ibIdx=%d "
                    "slots=[s1=%d s2=%d s3=%d s4=%d s5=%d s6=%d s7=%d] "
                    "vbCountsSize=%d flags=0x%04x vfmt=0x%02x submeshCnt=%u "
                    "(merged/per-instance template - no own geometry)",
                    sbspTagId, (unsigned long long)si, kind,
                    (int)sec.vertexBufferIndex, (int)sec.indexBufferIndex,
                    (int)s1, (int)s2, (int)s3, (int)s4, (int)s5, (int)s6, (int)s7,
                    (int)vbCounts.size(), sec.flags, sec.vertexFormat,
                    sec.submeshCount);
            }
            continue;
        }
        if (firstHookedSecIdx < 0) firstHookedSecIdx = (int)si;
        ++hookedSecs;
        sec.vertexCount   = vbCounts[sec.vertexBufferIndex];
        sec.vbDataLength  = vbLens[sec.vertexBufferIndex];
        if ((size_t)sec.vertexBufferIndex < entry.fixups.size())
            sec.vbResourceOffset =
                (uint32_t)(entry.fixups[sec.vertexBufferIndex].offset & 0x0FFFFFFF);

        // Note: in-range vbIdx with vbCounts[idx]==0 is legitimate (matches
        // Reclaimer's `if (vInfo.VertexCount == 0) continue;` skip path). Log
        // it once per BSP so we can correlate ground-missing reports with
        // these "section has a VB slot but the slot is empty" sections.
        if (sec.vertexCount == 0) {
            ++zeroVbCount;
            if (hookFailDiagBudget > 0) {
                --hookFailDiagBudget;
                NativeDiag(
                    "Bsp[%u]: BspSec[%llu] vbIdx=%d ibIdx=%d vbInfo.vc=0 - "
                    "section will produce no geometry (matches Reclaimer skip) "
                    "flags=0x%04x vfmt=0x%02x submeshCnt=%u",
                    sbspTagId, (unsigned long long)si,
                    (int)sec.vertexBufferIndex, (int)sec.indexBufferIndex,
                    sec.flags, sec.vertexFormat, sec.submeshCount);
            }
        }

        if (!sec.isUnindexed &&
            sec.indexBufferIndex >= 0 &&
            sec.indexBufferIndex < (int)ibLens.size())
        {
            sec.ibDataLength = ibLens[sec.indexBufferIndex];
            if ((size_t)sec.indexBufferIndex < ibFmts.size())
                sec.indexFormat = ibFmts[sec.indexBufferIndex];

            uint32_t indexStride = (sec.vertexCount > 0xFFFF) ? 4u : 2u;
            sec.indexCount = sec.ibDataLength / indexStride;

            size_t ibFixupIdx = vbCounts.size() * 2 + (size_t)sec.indexBufferIndex;
            if (ibFixupIdx < entry.fixups.size())
                sec.ibResourceOffset =
                    (uint32_t)(entry.fixups[ibFixupIdx].offset & 0x0FFFFFFF);
        } else if (sec.isUnindexed) {
            sec.indexCount = sec.vertexCount;
        }

        // -- UV2 (lightmap UV) stream resolver --------------------------------
        // Per SAPIEN_LIGHTMAP_PHASE3_RE.md section 9: each `s_mesh` carries an i16[8]
        // vertex_buffer_indices array. Slot 0 is the primary world VB; UV2
        // lives in some slot K in {1..7} where the engine bound the parallel
        // Float16x2 stream. We pick the first slot whose VB has the same
        // vertex count as the primary AND a 4-byte stride (Float16x2 only).
        //
        // If no slot matches (rigid/skinned/decorator meshes; or world meshes
        // that didn't get a UV2 stream baked) the section keeps uv2VbIndex=-1
        // and DecodeMeshUV2sInner returns false -> the viewer falls back to
        // "no lightmap" rendering.
        sec.uv2VbIndex = -1;
        sec.uv2VbResourceOffset = 0;
        sec.uv2VbDataLength = 0;
        sec.uv2VbStride = 0;
        sec.uv2VertexCount = 0;
        if (sec.vertexCount > 0) {
            // A cluster section can be lit by TWO tiers at once.
            // The lightmap-texcoord stream (Float16x2, stride 4) then covers only the
            // per-pixel vertices [0, count) and the per-vertex-lit parts occupy the tail
            // [count, vertexCount) (their lighting VB starts at pvb_offset == count).
            // Countdown sbsp 0x1af4 mesh 440: vertexCount 16624, uv2 stream 16580, PVL
            // window [16580, 16624). The old exact-count match rejected such streams, so
            // the whole cluster lost its atlas. Pass 1 keeps the exact match; pass 2
            // accepts a shorter stream (never a longer one).
            for (int pass = 0; pass < 2 && sec.uv2VbIndex < 0; ++pass) {
                for (int k = 1; k < 8; ++k) {
                    int16_t vbIdx = sec.vertexBufferIndices[k];
                    if (vbIdx < 0) continue;
                    if ((size_t)vbIdx >= vbCounts.size()) continue;
                    if (vbStrides[vbIdx] != 4) continue;       // Float16x2 only
                    uint32_t cnt = vbCounts[vbIdx];
                    if (pass == 0 ? (cnt != sec.vertexCount) : (cnt == 0 || cnt >= sec.vertexCount)) continue;
                    sec.uv2VbIndex      = vbIdx;
                    sec.uv2VbDataLength = vbLens[vbIdx];
                    sec.uv2VbStride     = vbStrides[vbIdx];
                    sec.uv2VertexCount  = cnt;
                    if ((size_t)vbIdx < entry.fixups.size())
                        sec.uv2VbResourceOffset =
                            (uint32_t)(entry.fixups[vbIdx].offset & 0x0FFFFFFF);
                    break;
                }
            }
        }

        // Diagnostic: dump the first 4 sections' VB binding map so we can
        // verify the UV2 slot resolver against a real cache. Format:
        //   BSP_UV2_BIND mesh=<si> primaryVB=<K0> otherVBs=[K1:c=N,s=B; ...] picked=<KP>
        // After the fix, picked should be the slot whose VB has 4-byte stride
        // and primary-matching vertex count. If picked=-1 for a world-format
        // section, that's either (a) a section without a UV2 stream baked or
        // (b) the slot resolver missed it (e.g. stride-derivation got the
        // wrong value because vbCounts[idx]==0 for the UV2 slot).
        // LIGHTMAP_FLAT_GREY_RE: widened 4 -> 96 so the cluster-mesh
        // UV2-binding distribution (how many cluster sections carry a stride-4 UV2
        // VB in vertex_buffer_indices[1..7] vs picked=-1) is fully observable on
        // one launch. picked=-1 cluster sections are why a resolved SDM atlas still
        // falls to flat grey: no per-vertex UV2 -> the viewer drops the lightmap -> textured
        // path. Bounded + cheap (one short log line per section, 96-cap per process).
        static std::atomic<int> s_uv2BindDiagBudget{ 96 };
        // ZH_UV2DIAG=1: print EVERY section's VB slot list to stderr (no budget) so a
        // specific section (by vertex count) can be inspected on any map.
        const bool uv2AllDiag = getenv("ZH_UV2DIAG") != nullptr;
        if (true) {
            int v = s_uv2BindDiagBudget.load(std::memory_order_relaxed);
            if (uv2AllDiag || (v > 0 && s_uv2BindDiagBudget.compare_exchange_weak(v, v - 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)))
            {
                char buf[512];
                int n = 0;
                int rem = (int)sizeof(buf);
                int w = 0;
                w = _snprintf_s(buf + n, rem, _TRUNCATE,
                    "[BSP_UV2_BIND] mesh=%llu primaryVB=%d primaryVc=%u primaryStride=%u otherVBs=[",
                    (unsigned long long)si, (int)sec.vertexBufferIndex,
                    sec.vertexCount,
                    (sec.vertexBufferIndex >= 0 && (size_t)sec.vertexBufferIndex < vbStrides.size())
                        ? vbStrides[sec.vertexBufferIndex] : 0u);
                if (w > 0) { n += w; rem -= w; }
                for (int k = 1; k < 8 && rem > 0; ++k) {
                    int16_t vbIdx = sec.vertexBufferIndices[k];
                    if (vbIdx < 0) continue;
                    uint32_t cnt = (size_t)vbIdx < vbCounts.size() ? vbCounts[vbIdx] : 0u;
                    uint32_t str = (size_t)vbIdx < vbStrides.size() ? vbStrides[vbIdx] : 0u;
                    w = _snprintf_s(buf + n, rem, _TRUNCATE, "S%d=%d:c=%u,s=%u; ",
                        k, (int)vbIdx, cnt, str);
                    if (w > 0) { n += w; rem -= w; }
                }
                _snprintf_s(buf + n, rem, _TRUNCATE, "] picked=%d vfmt=0x%02x",
                    (int)sec.uv2VbIndex, sec.vertexFormat);
                if (uv2AllDiag) fprintf(stderr, "ZH_UV2DIAG sbsp=0x%04x %s\n", (unsigned)sbspTagId, buf);
                NativeDiag("%s", buf);
            }
        }
    }
    // BspHookup tally - shows how many sections actually got VB/IB info hooked
    // up vs. how many were skipped. Buckets (per BSP_HOOKUP_OOB_RE.md):
    //   hooked - section has a valid primary VB; renders geometry.
    //   rebound - section's slot 0 was -1 but slot 1..7 had a primary-shape
    //              VB; we promoted it. Folded into `hooked` for compatibility
    //              but reported separately so we can audit how often it fires.
    //   sentinel - slot 0 == -1 AND no usable secondary slot. This is the
    //              expected stub for per-instance / subpart-merged template
    //              meshes. Not a bug. Same skip semantics as Reclaimer.
    //   overflow - slot 0 >= 0 but >= vbCounts.size(). This WOULD be a real
    //              walker bug. Always 0 on correctly-parsed Reach BSPs.
    //   zeroVb - in-range vbIdx with vbCounts[idx] == 0. Engine-faithful
    //              skip (matches Reclaimer's `if vc==0 continue`).
    if (firstHookedSecIdx >= 0) {
        const BspSection& fs = data.sections[firstHookedSecIdx];
        NativeDiag("Bsp[%u]: BspHookup secs=%d hooked=%d rebound=%d "
                   "sentinel=%d overflow=%d zeroVb=%d "
                   "vbCount=%d firstSec[%d] vbIdx=%d ibIdx=%d vc(post)=%u "
                   "ic(post)=%u vfmt=0x%02x",
            sbspTagId, (int)data.sections.size(), hookedSecs, reboundCount,
            sentinelCount, overflowCount, zeroVbCount,
            (int)vbCounts.size(), firstHookedSecIdx,
            fs.vertexBufferIndex, fs.indexBufferIndex,
            fs.vertexCount, fs.indexCount, fs.vertexFormat);
    } else if (!data.sections.empty()) {
        const BspSection& fs = data.sections[0];
        NativeDiag("Bsp[%u]: BspHookup secs=%d hooked=0 rebound=%d "
                   "sentinel=%d overflow=%d zeroVb=%d "
                   "vbCount=%d sec[0].vbIdx=%d vfmt=0x%02x - no sections hooked!",
            sbspTagId, (int)data.sections.size(), reboundCount,
            sentinelCount, overflowCount, zeroVbCount,
            (int)vbCounts.size(), fs.vertexBufferIndex, fs.vertexFormat);
    }

    // 7. Read + decompress the geometry resource page payload.
    // ReadResourceData takes cache->parseMutex internally for the shared-cache
    // open path; don't hold it here (std::mutex is non-recursive).
    constexpr size_t kMaxRead = 64 * 1024 * 1024;
    data.resourceData = ReadResourceData(cache, geomResourceIdRaw, kMaxRead, &data.resourceSize);
    if (!data.resourceData) return false;

    // 8. Build the unified mesh list:
    //      a) one mesh per cluster (identity transform)
    //      b) one mesh per geometry instance (per-instance transform)

    // 8a. Clusters. One BspMesh per SUBMESH (Reach packs multiple materials
    // into a single section - wall + window + trim share a section but each
    // gets its own submesh with its own shaderIndex + index range).
    std::vector<ClusterInfo> clusters;
    ParseClusters(cache, SL, te.metaPointerRaw, clusters);
    int clustersTotal = (int)clusters.size();
    int clustersBadIdx = 0;
    int clustersZeroVc = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
        int16_t secIdx = clusters[i].sectionIndex;
        if (secIdx < 0 || (size_t)secIdx >= data.sections.size()) {
            ++clustersBadIdx;
            continue;
        }
        const BspSection& sec = data.sections[secIdx];
        if (sec.vertexCount == 0) ++clustersZeroVc;
        // Placeholder per-mesh bounds - the real world-space AABB is computed
        // from the decoded vertices inside DecodeMeshGeometryInner and
        // overwrites these values. We seed with the section's compression
        // bbox so the diagnostic at first-decode time has something to log;
        // anything that calls ZH_BSP_GetMesh BEFORE the first decode will
        // see compression-bbox numbers (which are a closer approximation
        // than the old identity (0..1) placeholder).
        uint32_t subCount = sec.submeshCount > 0 ? sec.submeshCount : 1u;
        for (uint32_t si = 0; si < subCount; ++si) {
            BspMesh m{};
            m.sectionIndex = (uint32_t)secIdx;
            m.submeshIndexInSec = si;
            m.isInstance   = false;
            // BAKED_SHADOW_FIX: this mesh belongs to SBSP cluster i, which is
            // parallel to Lbsp.clusters[i] (the per-cluster lightmap slice).
            m.lightmapClusterIndex = (uint32_t)i;
            // Per-submesh material - pulls from the section's submesh array
            // built earlier (ParseSubmeshes stored shaderIndex into
            // outSubmeshes[sec.submeshStart + si].shaderIndex).
            if (sec.submeshCount > 0) {
                size_t globalSubIdx = (size_t)sec.submeshStart + si;
                if (globalSubIdx < data.submeshes.size())
                    m.materialIndex = data.submeshes[globalSubIdx].shaderIndex;
                else
                    m.materialIndex = sec.materialIndex;
            } else {
                m.materialIndex = sec.materialIndex;
            }
            IdentityMatrix(m.transform);
            m.uniformScale = 1.0f;
            memcpy(m.posMin, sec.posMin, sizeof(sec.posMin));
            memcpy(m.posMax, sec.posMax, sizeof(sec.posMax));
            memcpy(m.uvMin,  sec.uvMin,  sizeof(sec.uvMin));
            memcpy(m.uvMax,  sec.uvMax,  sizeof(sec.uvMax));
            // Cluster meshes have no instance ordinal - use sentinel.
            m.instanceOrdinal = 0xFFFFFFFFu;
            data.meshes.push_back(m);
        }
        NativeDiag(
            "Bsp[%u]: cluster[%llu] secIdx=%d vc=%u submeshes=%u vfmt=0x%02x firstMat=%d",
            sbspTagId, (unsigned long long)i, (int)secIdx, sec.vertexCount,
            sec.submeshCount, sec.vertexFormat, sec.materialIndex);
    }
    // Cluster summary - if clustersZeroVc is high, that's the "ground missing
    // in most places" smoking gun: most clusters point at sections whose
    // vbInfo[idx].vc is 0 (an in-range VB slot that's empty). Reclaimer
    // would also skip these, so the fix isn't on the read side - it's
    // upstream (the clusters are pointing at the wrong sections, or the
    // sections' actual VB lives at a different vbIdx than what they advertise).
    NativeDiag("Bsp[%u]: clusterSummary total=%d emitted=%d badIdx=%d zeroVc=%d",
        sbspTagId, clustersTotal,
        clustersTotal - clustersBadIdx, clustersBadIdx, clustersZeroVc);

    // 8b. Geometry instances. Need the InstancesResourcePointer entry which
    // is sbsp-side (resourceIndex above is the LBSP resource - different).
    int32_t sbspInstResourceIdRaw =
        R32(sbspMeta + SL.OFF_INSTANCES_RESOURCE_POINTER);
    data.resourceIndex = sbspInstResourceIdRaw & 0xFFFF;

    std::vector<GeometryInstance> instances;
    ParseGeometryInstances(cache, SL, te.metaPointerRaw,
                           sbspInstResourceIdRaw, instances);
    for (size_t i = 0; i < instances.size(); ++i) {
        int16_t secIdx = instances[i].sectionIndex;
        if (secIdx < 0 || (size_t)secIdx >= data.sections.size()) continue;
        const BspSection& sec = data.sections[secIdx];

        // Compute instance bounds once - same for every submesh of the
        // instance (they all share the section AABB transformed by the same
        // matrix).
        float corners[8][3] = {
            { sec.posMin[0], sec.posMin[1], sec.posMin[2] },
            { sec.posMax[0], sec.posMin[1], sec.posMin[2] },
            { sec.posMin[0], sec.posMax[1], sec.posMin[2] },
            { sec.posMax[0], sec.posMax[1], sec.posMin[2] },
            { sec.posMin[0], sec.posMin[1], sec.posMax[2] },
            { sec.posMax[0], sec.posMin[1], sec.posMax[2] },
            { sec.posMin[0], sec.posMax[1], sec.posMax[2] },
            { sec.posMax[0], sec.posMax[1], sec.posMax[2] },
        };
        float bMin[3] = {  3.4e38f,  3.4e38f,  3.4e38f };
        float bMax[3] = { -3.4e38f, -3.4e38f, -3.4e38f };
        float uniformScale = instances[i].transformScale;
        if (!std::isfinite(uniformScale) || uniformScale == 0.0f) uniformScale = 1.0f;
        for (int k = 0; k < 8; ++k) {
            float t[3];
            ApplyTransform(corners[k], instances[i].transform, uniformScale, t);
            for (int a = 0; a < 3; ++a) {
                if (t[a] < bMin[a]) bMin[a] = t[a];
                if (t[a] > bMax[a]) bMax[a] = t[a];
            }
        }

        uint32_t subCount = sec.submeshCount > 0 ? sec.submeshCount : 1u;
        for (uint32_t si = 0; si < subCount; ++si) {
            BspMesh m{};
            m.sectionIndex = (uint32_t)secIdx;
            m.submeshIndexInSec = si;
            m.isInstance   = true;
            if (sec.submeshCount > 0) {
                size_t globalSubIdx = (size_t)sec.submeshStart + si;
                if (globalSubIdx < data.submeshes.size())
                    m.materialIndex = data.submeshes[globalSubIdx].shaderIndex;
                else
                    m.materialIndex = sec.materialIndex;
            } else {
                m.materialIndex = sec.materialIndex;
            }
            memcpy(m.transform, instances[i].transform, sizeof(m.transform));
            m.uniformScale = uniformScale;
            memcpy(m.posMin, bMin, sizeof(bMin));
            memcpy(m.posMax, bMax, sizeof(bMax));
            memcpy(m.uvMin,  sec.uvMin, sizeof(sec.uvMin));
            memcpy(m.uvMax,  sec.uvMax, sizeof(sec.uvMax));
            // Stash the instance ordinal so the viewer can fetch this
            // instance's per-instance lighting (ZH_LBSP_GetInstancePvl /
            // ZH_BSP_GetInstancePvlVb).
            m.instanceOrdinal = (uint32_t)i;
            data.meshes.push_back(m);
        }
    }

    // Format histogram - one line summarising vertexFormat byte distribution
    // across every section + per-format mesh-emit counts. Lets us see at a
    // glance which formats exist in the BSP that IsFormatSupported() drops on
    // the floor (water=0x12, ripple=0x13, fog=0x11, implicit=0x14, etc.).
    {
        uint32_t fmtSecCount[256] = {};
        uint32_t fmtMeshCount[256] = {};
        for (const auto& sec : data.sections)
            ++fmtSecCount[sec.vertexFormat & 0xFF];
        for (const auto& m : data.meshes) {
            if (m.sectionIndex < data.sections.size())
                ++fmtMeshCount[data.sections[m.sectionIndex].vertexFormat & 0xFF];
        }
        char buf[768];
        size_t off = 0;
        off += (size_t)snprintf(buf + off, sizeof(buf) - off,
            "Bsp[%u]: vfmt-hist sections=", sbspTagId);
        for (int f = 0; f < 256 && off + 24 < sizeof(buf); ++f) {
            if (fmtSecCount[f] == 0) continue;
            const char* sup = IsFormatSupported((uint32_t)f) ? "ok" : "DROP";
            off += (size_t)snprintf(buf + off, sizeof(buf) - off,
                "[0x%02x %s s=%u m=%u]",
                (unsigned)f, sup, fmtSecCount[f], fmtMeshCount[f]);
        }
        NativeDiag("%s", buf);
    }

    NativeDiag("Bsp[%u]: parse OK sections=%d submeshes=%d shaders=%d meshes=%d "
               "(clusters+inst) lbsp=%d resSz=%llu",
        sbspTagId, (int)data.sections.size(), (int)data.submeshes.size(),
        (int)data.shaders.size(), (int)data.meshes.size(), usingLbsp ? 1 : 0,
        (unsigned long long)data.resourceSize);
    return true;
}

// -----------------------------------------------------------------------------
// Decode one mesh (cluster or instance). Vertex+index buffers go to malloc'd
// outputs; the per-instance transform is baked into the position stream.
// -----------------------------------------------------------------------------

bool DecodeMeshGeometryInner(BspData* bsp, uint32_t meshIndex,
                             uint8_t** outVertexBytes, uint32_t* outVertexLen,
                             uint8_t** outIndexBytes,  uint32_t* outIndexLen)
{
    if (meshIndex >= bsp->meshes.size()) return false;
    // Need a mutable reference because we cache the per-mesh world-space AABB
    // into mesh.posMin/posMax after decode (the constructor only had the
    // section's compression bbox, which is NOT a world AABB).
    BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;

    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > bsp->resourceSize)
        return false;
    const uint8_t* vbSrc = bsp->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    const uint8_t* ibSrc = nullptr;
    if (!sec.isUnindexed) {
        if ((size_t)sec.ibResourceOffset + (size_t)sec.ibDataLength > bsp->resourceSize)
            return false;
        ibSrc = bsp->resourceData + sec.ibResourceOffset;
    }

    // Decode positions - engine-faithful uncompress transform per section.
    //
    // The Reach BSP vertex pipeline ALWAYS applies the per-section
    // Position_Compression_Scale/Offset constants in the vertex shader
    // (see HREK/tags/shaders/templated/deform.hlsl_include::deform_flat_rigid:
    //   position.xyz = vertex.position.xyz * Position_Compression_Scale.xyz
    //                + Position_Compression_Offset.xyz
    // ). The constants come from BoundingBoxes[sectionIndex]:
    //   Position_Compression_Scale  = (posMax - posMin)
    //   Position_Compression_Offset = posMin
    // This applies uniformly to BOTH cluster meshes AND geometry instances - 
    // there is no cluster carve-out in the shader pipeline. (Reclaimer's
    // scenario_structure_bsp.cs assigns PositionBounds per-mesh from the
    // BoundingBoxes block to every model.Meshes[i] and then the renderer
    // applies the ExpansionMatrix uniformly - RealBounds3D.cs:16.)
    //
    // Fallback: if the section's bbox is fully degenerate (Min==Max on EVERY
    // axis), Reclaimer's `RealBounds3D.IsEmpty` is true and CreateExpansionMatrix
    // returns Identity. We mirror that here - without this guard, sections
    // whose bbox is (0,0,0)..(0,0,0) would zero out every vertex. A partial
    // degenerate (e.g. zero-X, non-zero-YZ for a true planar section) IS still
    // applied - that's a legitimate quantization, not an empty bbox.
    const bool bboxIsEmpty = (sec.posMin[0] == sec.posMax[0]) &&
                             (sec.posMin[1] == sec.posMax[1]) &&
                             (sec.posMin[2] == sec.posMax[2]);
    const float identityMin[3] = { 0.0f, 0.0f, 0.0f };
    const float identityMax[3] = { 1.0f, 1.0f, 1.0f };
    const float* decodePosMin = bboxIsEmpty ? identityMin : sec.posMin;
    const float* decodePosMax = bboxIsEmpty ? identityMax : sec.posMax;
    size_t vertexBytes = (size_t)sec.vertexCount * 12;
    uint8_t* vBuf = (uint8_t*)malloc(vertexBytes);
    if (!vBuf) return false;
    DecodeBspPositions(vbSrc, sec.vertexCount, stride, sec.vertexFormat,
                       decodePosMin, decodePosMax, vBuf);

    if (mesh.isInstance) {
        float* o = reinterpret_cast<float*>(vBuf);
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            float p[3] = { o[i*3+0], o[i*3+1], o[i*3+2] };
            float r[3];
            ApplyTransform(p, mesh.transform, mesh.uniformScale, r);
            o[i*3+0] = r[0]; o[i*3+1] = r[1]; o[i*3+2] = r[2];
        }
    }

    // Compute the actual world-space AABB from the decoded (and possibly
    // instance-transformed) vertices, and overwrite the placeholder bounds
    // the mesh-emit step set up. This is what frustum culling, shadow
    // volume bounds, the bounding-sphere transparency Z-sort, and the
    // mouse-pick raycaster all read via ZH_BSP_GetMesh.BoundsMin/Max - 
    // they MUST be real world-space numbers, NOT the (0..1) identity that
    // cluster meshes were emitting and NOT the compression bbox (which on
    // many BSPs is degenerate/normalized and not a world AABB).
    {
        const float* o = reinterpret_cast<const float*>(vBuf);
        float bMin[3] = {  3.4e38f,  3.4e38f,  3.4e38f };
        float bMax[3] = { -3.4e38f, -3.4e38f, -3.4e38f };
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            for (int a = 0; a < 3; ++a) {
                float v = o[i*3 + a];
                if (!std::isfinite(v)) continue;
                if (v < bMin[a]) bMin[a] = v;
                if (v > bMax[a]) bMax[a] = v;
            }
        }
        if (sec.vertexCount > 0 && bMin[0] <= bMax[0]) {
            memcpy(mesh.posMin, bMin, sizeof(bMin));
            memcpy(mesh.posMax, bMax, sizeof(bMax));
        }
    }

    // One-time-per-format diagnostic - verify stride/decode is sane.
    {
        static std::atomic<uint32_t> s_loggedFormats{ 0 };
        uint32_t mask = 1u << (sec.vertexFormat & 0x1F);
        uint32_t prev = s_loggedFormats.fetch_or(mask, std::memory_order_relaxed);
        if ((prev & mask) == 0 && sec.vertexCount > 0) {
            const float* f = reinterpret_cast<const float*>(vBuf);
            NativeDiag("BspDecodeVerts: type=0x%02x stride=%u verts=%u first=(%g,%g,%g) "
                       "inst=%d secBbox=(%g..%g,%g..%g,%g..%g) bboxEmpty=%d "
                       "meshAABB=(%g..%g,%g..%g,%g..%g)",
                sec.vertexFormat, stride, sec.vertexCount,
                f[0], f[1], f[2], mesh.isInstance ? 1 : 0,
                sec.posMin[0], sec.posMax[0], sec.posMin[1], sec.posMax[1],
                sec.posMin[2], sec.posMax[2], bboxIsEmpty ? 1 : 0,
                mesh.posMin[0], mesh.posMax[0], mesh.posMin[1], mesh.posMax[1],
                mesh.posMin[2], mesh.posMax[2]);
        }
    }

    // Compute post-expansion submesh windows in a LOCAL array (not
    // mutating bsp->submeshes - that shared mutation was the data race
    // that caused the "one or two blob meshes per BSP" symptom under
    // Parallel.For decode. See DecodeIndices header comment +
    // BSP_BLOB_MESH_RE.md.
    uint32_t outIdxCount = 0, outIdxStride = 0;
    uint32_t localPostStart[16];
    uint32_t localPostLen[16];
    std::vector<uint32_t> heapPostStart;
    std::vector<uint32_t> heapPostLen;
    uint32_t* postStart = nullptr;
    uint32_t* postLen   = nullptr;
    if (sec.submeshCount > 0) {
        if (sec.submeshCount <= 16) {
            postStart = localPostStart;
            postLen   = localPostLen;
        } else {
            heapPostStart.resize(sec.submeshCount);
            heapPostLen.resize(sec.submeshCount);
            postStart = heapPostStart.data();
            postLen   = heapPostLen.data();
        }
    }
    uint8_t* iBuf = DecodeIndices(bsp, mesh.sectionIndex, ibSrc,
                                  &outIdxCount, &outIdxStride,
                                  postStart, postLen);
    if (!iBuf) { free(vBuf); return false; }

    // Slice indices to the submesh's range. Each submesh references a
    // (start, length) window inside the section's full index buffer; the
    // engine renders that window with its own shader. We replace iBuf
    // with a copy that's just the windowed indices.
    if (sec.submeshCount > 0 && mesh.submeshIndexInSec < sec.submeshCount &&
        postStart && postLen)
    {
        uint32_t subStart = postStart[mesh.submeshIndexInSec];
        uint32_t subLen   = postLen[mesh.submeshIndexInSec];
        // Bounds-check the window against the decoded index count.
        if (subStart < outIdxCount &&
            subStart + subLen <= outIdxCount &&
            subLen > 0)
        {
            size_t sliceBytes = (size_t)subLen * outIdxStride;
            uint8_t* sliced = (uint8_t*)malloc(sliceBytes);
            if (sliced) {
                memcpy(sliced, iBuf + (size_t)subStart * outIdxStride,
                       sliceBytes);
                free(iBuf);
                iBuf = sliced;
                outIdxCount = subLen;
            }
        } else if (subLen == 0) {
            // Submesh's raw-strip window was invalid - engine renders
            // nothing for it; produce zero-length output (DecodeMesh
            // caller treats this as "skip").
            free(iBuf);
            iBuf = (uint8_t*)malloc(outIdxStride);  // 1-element placeholder
            if (!iBuf) { free(vBuf); return false; }
            outIdxCount = 0;
            // Diag (budgeted) so this surfaces in the native log if a real
            // map starts producing zero-window submeshes.
            static std::atomic<int> s_zeroWinDiagBudget{ 8 };
            int bv = s_zeroWinDiagBudget.load(std::memory_order_relaxed);
            if (bv > 0 && s_zeroWinDiagBudget.compare_exchange_weak(
                    bv, bv - 1,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
            {
                NativeDiag(
                    "BspDecodeMesh: mesh=%u sec=%u sub=%u zero-length window "
                    "(raw-strip slice OOB or <3 indices) - no geometry emitted",
                    meshIndex, mesh.sectionIndex,
                    (unsigned)mesh.submeshIndexInSec);
            }
        }
    }

    *outVertexBytes = vBuf;
    *outVertexLen   = (uint32_t)vertexBytes;
    *outIndexBytes  = iBuf;
    *outIndexLen    = outIdxCount * outIdxStride;
    return true;
}

bool DecodeMeshUVsInner(BspData* bsp, uint32_t meshIndex,
                        float** outUv, uint32_t* outUvFloatCount)
{
    if (meshIndex >= bsp->meshes.size()) return false;
    const BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > bsp->resourceSize)
        return false;
    const uint8_t* vbSrc = bsp->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    size_t bytes = (size_t)sec.vertexCount * 2 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    DecodeBspUVs(vbSrc, sec.vertexCount, stride, sec.vertexFormat,
                 sec.uvMin, sec.uvMax, buf);

    *outUv = buf;
    *outUvFloatCount = sec.vertexCount * 2;
    return true;
}

// Mirror of DecodeMeshUVsInner but for the SECONDARY (lightmap UV2) channel.
// Returns false on:
//   * mesh / section OOB
//   * section has no resolved UV2 VB (uv2VbIndex == -1; rigid/skinned/
//     decorator meshes typically lack this stream, and even some `world`
//     clusters bind only slot 0)
//   * UV2 VB extents that don't fit the resource page
// Caller falls back to "no lightmap" rendering on false.
bool DecodeMeshUV2sInner(BspData* bsp, uint32_t meshIndex,
                         float** outUv, uint32_t* outUvFloatCount)
{
    if (meshIndex >= bsp->meshes.size()) return false;
    const BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;

    // No UV2 stream resolved for this section (slots 1..7 all -1, or none
    // matched stride==4 / count==primary). Caller falls back gracefully.
    if (sec.uv2VbIndex < 0) return false;
    if (sec.uv2VbStride != 4) return false;  // Float16x2 stream only
    // The stream may cover only the per-pixel prefix [0, uv2VertexCount).
    if (sec.uv2VertexCount == 0 || sec.uv2VertexCount > sec.vertexCount) return false;
    if ((size_t)sec.uv2VbResourceOffset + (size_t)sec.uv2VbDataLength
            > bsp->resourceSize) return false;
    if ((size_t)sec.uv2VertexCount * 4 > sec.uv2VbDataLength) return false;

    const uint8_t* uv2Src = bsp->resourceData + sec.uv2VbResourceOffset;

    size_t bytes = (size_t)sec.vertexCount * 2 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    if (!DecodeBspUVs2(uv2Src, sec.uv2VertexCount, buf)) {
        free(buf);
        return false;
    }
    // Vertices past the stream (per-vertex-lit tail parts) get NaN so the consumer
    // (which keeps only finite texcoords) leaves them unmapped.
    for (uint32_t v = sec.uv2VertexCount; v < sec.vertexCount; ++v) {
        buf[(size_t)v * 2 + 0] = NAN;
        buf[(size_t)v * 2 + 1] = NAN;
    }

    *outUv = buf;
    *outUvFloatCount = sec.vertexCount * 2;
    return true;
}

// Decode one mesh's per-vertex normals. Output is a malloc'd float3[vc] array.
// For instance meshes, normals are rotated by the instance's 3x3 rotation
// matrix (same approach as position transform, minus translation).
bool DecodeMeshNormalsInner(BspData* bsp, uint32_t meshIndex,
                            float** outNormals, uint32_t* outNormalFloatCount)
{
    if (meshIndex >= bsp->meshes.size()) return false;
    const BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > bsp->resourceSize)
        return false;
    const uint8_t* vbSrc = bsp->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    size_t bytes = (size_t)sec.vertexCount * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    DecodeBspNormals(vbSrc, sec.vertexCount, stride, sec.vertexFormat, buf);

    // Rotate normals by the instance's 3x3 rotation matrix (upper-left 3x3
    // of the 4x4 transform). No translation, no uniform scale - normals are
    // direction vectors. Re-normalize after rotation to handle non-unit-scale
    // rotation matrices.
    if (mesh.isInstance) {
        const float* M = mesh.transform;
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            float nx = buf[i*3+0], ny = buf[i*3+1], nz = buf[i*3+2];
            float rx = M[0]*nx + M[1]*ny + M[2]*nz;
            float ry = M[4]*nx + M[5]*ny + M[6]*nz;
            float rz = M[8]*nx + M[9]*ny + M[10]*nz;
            float len = sqrtf(rx*rx + ry*ry + rz*rz);
            if (len > 1e-8f) { rx /= len; ry /= len; rz /= len; }
            buf[i*3+0] = rx; buf[i*3+1] = ry; buf[i*3+2] = rz;
        }
    }

    *outNormals = buf;
    *outNormalFloatCount = sec.vertexCount * 3;
    return true;
}

// Decode one mesh's per-vertex tangents. Output is a malloc'd float3[vc] array.
// Same instance rotation as normals.
bool DecodeMeshTangentsInner(BspData* bsp, uint32_t meshIndex,
                             float** outTangents, uint32_t* outTangentFloatCount)
{
    if (meshIndex >= bsp->meshes.size()) return false;
    const BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;
    // Decorator has no tangent data.
    if (sec.vertexFormat == VFMT_DECORATOR) return false;
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > bsp->resourceSize)
        return false;
    const uint8_t* vbSrc = bsp->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    size_t bytes = (size_t)sec.vertexCount * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    DecodeBspTangents(vbSrc, sec.vertexCount, stride, sec.vertexFormat, buf);

    // Rotate tangents by the instance's 3x3 rotation matrix, same as normals.
    if (mesh.isInstance) {
        const float* M = mesh.transform;
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            float tx = buf[i*3+0], ty = buf[i*3+1], tz = buf[i*3+2];
            float rx = M[0]*tx + M[1]*ty + M[2]*tz;
            float ry = M[4]*tx + M[5]*ty + M[6]*tz;
            float rz = M[8]*tx + M[9]*ty + M[10]*tz;
            float len = sqrtf(rx*rx + ry*ry + rz*rz);
            if (len > 1e-8f) { rx /= len; ry /= len; rz /= len; }
            buf[i*3+0] = rx; buf[i*3+1] = ry; buf[i*3+2] = rz;
        }
    }

    *outTangents = buf;
    *outTangentFloatCount = sec.vertexCount * 3;
    return true;
}

// Decode one mesh's per-vertex binormals. Computed from normal + tangent with
// the tangent W sign bit. Same instance rotation as normals/tangents.
bool DecodeMeshBinormalsInner(BspData* bsp, uint32_t meshIndex,
                              float** outBinormals, uint32_t* outBinormalFloatCount)
{
    if (meshIndex >= bsp->meshes.size()) return false;
    const BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;
    if (sec.vertexFormat == VFMT_DECORATOR) return false;
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > bsp->resourceSize)
        return false;
    const uint8_t* vbSrc = bsp->resourceData + sec.vbResourceOffset;

    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;

    size_t bytes = (size_t)sec.vertexCount * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);
    if (!buf) return false;
    DecodeBspBinormals(vbSrc, sec.vertexCount, stride, sec.vertexFormat, buf);

    // Rotate binormals by instance 3x3 rotation.
    if (mesh.isInstance) {
        const float* M = mesh.transform;
        for (uint32_t i = 0; i < sec.vertexCount; ++i) {
            float bx = buf[i*3+0], by = buf[i*3+1], bz = buf[i*3+2];
            float rx = M[0]*bx + M[1]*by + M[2]*bz;
            float ry = M[4]*bx + M[5]*by + M[6]*bz;
            float rz = M[8]*bx + M[9]*by + M[10]*bz;
            float len = sqrtf(rx*rx + ry*ry + rz*rz);
            if (len > 1e-8f) { rx /= len; ry /= len; rz /= len; }
            buf[i*3+0] = rx; buf[i*3+1] = ry; buf[i*3+2] = rz;
        }
    }

    *outBinormals = buf;
    *outBinormalFloatCount = sec.vertexCount * 3;
    return true;
}

// -----------------------------------------------------------------------------
// Diffuse bitmap resolver - same walk as MapModelParser.
// -----------------------------------------------------------------------------

constexpr int OFF_SHADER_PROPS         = 56;
constexpr int SHADER_PROPS_BLOCK_SIZE  = 172;
constexpr int OFF_SHADER_MAPS_IN_PROPS = 16;
constexpr int SHADER_MAP_BLOCK_SIZE    = 24;
constexpr int OFF_RMT_USAGES           = 108;
constexpr int STRINGID_BLOCK_SIZE      = 4;

// Surface-color usage names with Reclaimer's TextureLoader fallback rank
// (TextureLoader.cs:30-31). Rank 0 = TextureUsage.Diffuse (base_map / etc).
// Rank 1 = TextureUsage.ColorChange fallback. Lower rank wins.
struct BspDiffuseUsageEntry {
    const char* name;
    int         rank;
};
static const BspDiffuseUsageEntry kBspDiffuseUsages[] = {
    { "base_map",         0 },
    { "alpha_mask_map",   0 },
    { "foam_texture",     0 },
    { "change_color_map", 1 },
    // detail_map is normally a SECONDARY layer multiplied on top of base.
    // We accept it at low priority (rank 2) so that when a shader's base
    // slot has a detail-named bitmap (m50_concrete_b on panopticon - 
    // base_map=`d_concrete_weathered_detail_smudge`, detail_map=`concrete_a_diff`)
    // we can fall through to the detail_map slot whose bitmap is the
    // real diffuse. The detail-named-bitmap penalty (+3) below ensures
    // legitimate detail layers don't override real base diffuses.
    { "detail_map",       2 },
};

// Detail-map heuristic: Bungie names secondary detail bitmaps with `d_` as
// the last-path-segment prefix and/or `_detail` somewhere in the stem
// (e.g. `levels\shared\bitmaps\human\concrete\d_concrete_weathered_detail_smudge`).
// These are meant to multiply on top of a base diffuse, never to be the
// primary surface texture. When the rmt2's `base_map` slot points at one
// (rare but verified - concrete floor on panopticon), we want to fall
// through to a different slot or the next-rank candidate instead of
// installing the gray detail tile as the base. Returns true for "this
// bitmap looks like a detail map, skip it as a primary diffuse".
static bool BspBitmapNameLooksLikeDetail(const char* bmpName) {
    if (!bmpName || !*bmpName) return false;
    // Last-segment scan - the path may have any number of leading dirs.
    const char* last = bmpName;
    for (const char* p = bmpName; *p; ++p) {
        if (*p == '\\' || *p == '/') last = p + 1;
    }
    // Prefix `d_` on the file part (NOT just `d` followed by anything - 
    // common diffuse bitmaps start with `d` too).
    if (last[0] == 'd' && last[1] == '_') return true;
    // `_detail` substring anywhere in the file part. Reach uses both
    // `_detail_` and `_detail` at end.
    for (const char* p = last; *p; ++p) {
        if ((p[0] == '_' || p == last) &&
            p[1] == 'd' && p[2] == 'e' && p[3] == 't' && p[4] == 'a' && p[5] == 'i' && p[6] == 'l' &&
            (p[7] == '_' || p[7] == 0)) return true;
    }
    return false;
}

// #158 (RE: shader_apply_pass_state_for_rmt2 routing): a shader template with no artist
// `base_map` parameter routes its base register to an EXTERN texture (g_extern_descriptions)
// or the rasterizer DEFAULT bitmap - an engine-internal LUT like `rasterizer\direction_lut`
// or `shaders\default_bitmaps\*`. Those are NEVER an artist diffuse; selecting one paints the
// surface with a LUT (rendered white/garbage downstream). Reject any bitmap whose tag path is
// under those internal roots so the resolver returns "no diffuse" (0xFFFFFFFF) and the caller
// falls back to self-illum / neutral, which is the correct result for such procedural templates.
static bool BspBitmapIsInternal(const char* bmpName) {
    if (!bmpName || !*bmpName) return false;
    // Case-insensitive prefix compare on the two internal roots.
    static const char* kRoots[] = { "rasterizer\\", "shaders\\default_bitmaps\\" };
    for (const char* root : kRoots) {
        size_t n = strlen(root);
        bool match = true;
        for (size_t i = 0; i < n; ++i) {
            char a = bmpName[i]; if (a == 0) { match = false; break; }
            char b = root[i];
            if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
            if (a != b) { match = false; break; }
        }
        if (match) return true;
    }
    return false;
}

// Strip Reclaimer's `_m_<digit>` suffix from a usage string and return the
// resulting rank (or -1 if not a surface-color slot).
static int BspMatchDiffuseUsageRank(const char* usageName) {
    if (!usageName) return -1;
    size_t len = strlen(usageName);
    if (len >= 4 && usageName[len - 4] == '_' && usageName[len - 3] == 'm' &&
        usageName[len - 2] == '_' && usageName[len - 1] >= '0' && usageName[len - 1] <= '9') {
        len -= 4;
    }
    for (int i = 0; i < (int)(sizeof(kBspDiffuseUsages) / sizeof(kBspDiffuseUsages[0])); ++i) {
        const char* candidate = kBspDiffuseUsages[i].name;
        size_t candLen = strlen(candidate);
        if (candLen != len) continue;
        if (memcmp(usageName, candidate, candLen) == 0) return kBspDiffuseUsages[i].rank;
    }
    return -1;
}

// Diag budget - first N calls per process print a full per-shader dump of
// the rmt2 usages walk + chosen bitmap + ALL slots (usage + bitmap name).
// Lets us see exactly why a given material picked a detail bitmap.
static std::atomic<int> s_bspResolverDiagBudget{ 600 };

// Resolve a shader's diffuse bitmap tag id. Walks the same chain as
// MapModelParser::ResolveDiffuseBitmapTagId - try the rmt!.Usages[]
// StringId match first, fall back to "first valid bitm in ShaderMaps[]".
uint32_t ResolveDiffuseBitmapTagIdUncached(CacheHandle* cache, int32_t shaderTagId);

// #175 PERF wrapper: memoize the (expensive) rmt2 chain walk per shaderTagId. A shader is
// shared by many meshes; the uncached resolver was re-walking the whole chain once PER MESH.
uint32_t ResolveDiffuseBitmapTagId(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0xFFFFFFFFu;
    {
        std::lock_guard<std::mutex> lk(cache->diffuseResolveMutex);
        auto it = cache->diffuseResolveCache.find(shaderTagId);
        if (it != cache->diffuseResolveCache.end()) return it->second;
    }
    uint32_t result = ResolveDiffuseBitmapTagIdUncached(cache, shaderTagId);
    {
        std::lock_guard<std::mutex> lk(cache->diffuseResolveMutex);
        // #204: do NOT memoize a 0xFFFFFFFF FAILURE. A shader resolved EARLY (during a probe/
        // prewarm, before the bitmap/tag/string state is fully ready) can transiently fail; caching
        // that poisons EVERY mesh sharing the shader for the whole session (Countdown mat 50 =
        // unsc_gantry_track rendered as flat grey "see-through" patches - the same shader resolves
        // fine when re-walked at render time). Only cache real hits; let a failure re-resolve.
        if (result != 0xFFFFFFFFu) {
            cache->diffuseResolveCache[shaderTagId] = result;
        }
    }
    return result;
}

uint32_t ResolveDiffuseBitmapTagIdUncached(CacheHandle* cache, int32_t shaderTagId) {
    bool logThis = false;
    {
        int v = s_bspResolverDiagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_bspResolverDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed)) {
                logThis = true;
                break;
            }
        }
    }
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size())
        return 0xFFFFFFFFu;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFFFFFFFFu;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFFFFFFFFu;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size)
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

    // rmt!.Usages[] StringId match - same logic as MapModelParser. Returns
    // the bitmap of the slot whose usage is in kBspDiffuseUsageNames (the
    // earliest-priority match wins).
    {
        int32_t rmtRawId = R32(props + 12);
        int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                           ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
        if (rmtTagId >= 0 && (uint32_t)rmtTagId < cache->tags.size() &&
            cache->stringTableParsed)
        {
            const TagEntry& rmtTe = cache->tags[rmtTagId];
            // Accept both "rmt!" (older) and "rmt2" (U13) - same Usages layout.
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
                    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
                    if (usagesBlk.count > 0 && usagesBlk.count <= 0x1000) {
                        int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
                        if (usagesOff >= 0 &&
                            (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE
                                <= cache->size)
                        {
                            int slotMax = mapsBlk.count;
                            if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;
                            int bestRank = (int)(sizeof(kBspDiffuseUsages) /
                                                 sizeof(kBspDiffuseUsages[0]));
                            int bestSlot = -1;
                            for (int i = 0; i < slotMax; ++i) {
                                int32_t sid = R32(cache->base + usagesOff +
                                                  (size_t)i * STRINGID_BLOCK_SIZE);
                                const char* usageName = ResolveStringId(cache, sid);
                                if (!usageName || !*usageName) continue;
                                // Match per Reclaimer (TextureLoader.cs:30-31):
                                // rank 0 = base_map/alpha_mask_map/foam_texture,
                                // rank 1 = change_color_map fallback, _m_<n>
                                // suffix stripped per UsageRegex.
                                int rank = BspMatchDiffuseUsageRank(usageName);
                                if (rank < 0 || rank >= bestRank) continue;
                                const uint8_t* mapEntry = cache->base + mapsOff +
                                                         (size_t)i * SHADER_MAP_BLOCK_SIZE;
                                int32_t rawId = R32(mapEntry + 12);
                                if ((uint32_t)rawId == 0xFFFFFFFFu) continue;
                                uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
                                if (bmpId >= cache->tags.size()) continue;
                                if (memcmp(cache->tags[bmpId].classCode, "bitm", 4) != 0) continue;
                                // #158: skip engine-internal LUT/default bitmaps (rasterizer\,
                                // shaders\default_bitmaps\) - a template with no artist base_map
                                // routes its base register to one; it is never a real diffuse.
                                if (BspBitmapIsInternal(cache->tags[bmpId].tagName.c_str())) continue;
                                // Penalty for detail-named bitmaps. Bungie
                                // marks detail textures with `d_` prefix or
                                // `_detail` substring on the file name; when
                                // a shader's `base_map` slot points at one,
                                // it's an authoring quirk and the real
                                // diffuse is usually in a `detail_map` slot
                                // a few entries down (verified on panopticon's
                                // m50_concrete_b - slot 0 base_map=detail-named
                                // smudge, slot 1 detail_map=concrete_a_diff
                                // which IS the real diffuse).
                                //
                                // Effective rank table (lower wins):
                                //   non-detail base_map / alpha_mask / foam       = 0
                                //   non-detail change_color_map                   = 1
                                //   non-detail detail_map                         = 2
                                //   detail-named base_map / alpha_mask / foam     = 3
                                //   detail-named change_color_map                 = 4
                                //   detail-named detail_map (legitimate detail)   = 5
                                //
                                // So a non-detail-named diffuse in a
                                // detail_map slot beats a detail-named
                                // smudge in the base_map slot.
                                int effectiveRank = rank;
                                if (BspBitmapNameLooksLikeDetail(cache->tags[bmpId].tagName.c_str())) {
                                    effectiveRank = rank + 3;
                                }
                                if (effectiveRank >= bestRank) continue;
                                bestRank = effectiveRank;
                                bestSlot = i;
                                if (effectiveRank == 0) break;
                            }
                            if (bestSlot >= 0) {
                                const uint8_t* mapEntry = cache->base + mapsOff +
                                                         (size_t)bestSlot * SHADER_MAP_BLOCK_SIZE;
                                uint32_t bmpId = (uint32_t)R32(mapEntry + 12) & 0xFFFFu;
                                if (logThis) {
                                    const char* shClass = cache->tags[shaderTagId].classCode;
                                    const char* shName  = cache->tags[shaderTagId].tagName.c_str();
                                    const char* bmpName = (bmpId < cache->tags.size())
                                        ? cache->tags[bmpId].tagName.c_str() : "?";
                                    char slotDump[1024] = {0};
                                    size_t sdUsed = 0;
                                    int dumpMax = mapsBlk.count;
                                    if (dumpMax > 16) dumpMax = 16;
                                    for (int s = 0; s < dumpMax; ++s) {
                                        const uint8_t* mEnt = cache->base + mapsOff +
                                                              (size_t)s * SHADER_MAP_BLOCK_SIZE;
                                        int32_t mRaw = R32(mEnt + 12);
                                        uint32_t mBmp = ((uint32_t)mRaw == 0xFFFFFFFFu)
                                            ? 0xFFFFFFFFu : (uint32_t)mRaw & 0xFFFFu;
                                        const char* mName = (mBmp < cache->tags.size() &&
                                            memcmp(cache->tags[mBmp].classCode, "bitm", 4) == 0)
                                            ? cache->tags[mBmp].tagName.c_str() : "<no-bitm>";
                                        const char* shortBmp = mName;
                                        for (const char* p = mName; *p; ++p)
                                            if (*p == '\\' || *p == '/') shortBmp = p + 1;
                                        const char* uname = "?";
                                        if (s < usagesBlk.count) {
                                            int32_t sid2 = R32(cache->base + usagesOff +
                                                              (size_t)s * STRINGID_BLOCK_SIZE);
                                            const char* un = ResolveStringId(cache, sid2);
                                            if (un && *un) uname = un;
                                        }
                                        if (sdUsed + 96 < sizeof(slotDump)) {
                                            int n = _snprintf_s(slotDump + sdUsed,
                                                sizeof(slotDump) - sdUsed, _TRUNCATE,
                                                "%s[%d:%s]=%s",
                                                sdUsed == 0 ? "" : " ", s, uname, shortBmp);
                                            if (n > 0) sdUsed += (size_t)n;
                                        }
                                    }
                                    zh_mcc::NativeDiag(
                                        "BspShader[0x%X] cls=%c%c%c%c tag='%s' "
                                        "rank=%d slot=%d picked='%s' slots=%s",
                                        shaderTagId,
                                        shClass[0], shClass[1], shClass[2], shClass[3],
                                        shName, bestRank, bestSlot, bmpName, slotDump);
                                }
                                return bmpId;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: first valid bitm. Skip detail-named bitmaps in the first
    // pass; if no non-detail candidate exists, do a second pass that
    // accepts them (better to render the wrong texture than no texture).
    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < mapsBlk.count; ++i) {
            const uint8_t* mapEntry = cache->base + mapsOff +
                                      (size_t)i * SHADER_MAP_BLOCK_SIZE;
            int32_t rawId = R32(mapEntry + 12);
            if ((uint32_t)rawId == 0xFFFFFFFFu) continue;
            uint32_t bitmapTagId = (uint32_t)rawId & 0xFFFFu;
            if (bitmapTagId >= cache->tags.size()) continue;
            if (memcmp(cache->tags[bitmapTagId].classCode, "bitm", 4) != 0)
                continue;
            // #158: never fall back to an engine-internal LUT/default bitmap (in EITHER pass)
            // - better to return no-diffuse (caller renders self-illum/neutral) than paint the
            // surface with rasterizer\direction_lut.
            if (BspBitmapIsInternal(cache->tags[bitmapTagId].tagName.c_str()))
                continue;
            if (pass == 0 &&
                BspBitmapNameLooksLikeDetail(cache->tags[bitmapTagId].tagName.c_str()))
                continue;
            return bitmapTagId;
        }
    }
    // #204 FAILURE DIAG: shader is a valid rm** with props/maps but NO slot
    // yielded a usable diffuse (all internal LUT/default bitmaps, or no bitm at
    // all). Dump the shader + every map slot so we can see what the engine binds
    // for these see-through meshes (e.g. Countdown material 50, the 39 gantry/
    // concrete meshes). Gated by the same diag budget as the success path.
    if (logThis) {
        const char* shClass = cache->tags[shaderTagId].classCode;
        const char* shName  = cache->tags[shaderTagId].tagName.c_str();
        char slotDump[1200] = {0};
        size_t sdUsed = 0;
        int dumpMax = mapsBlk.count; if (dumpMax > 20) dumpMax = 20;
        for (int s = 0; s < dumpMax; ++s) {
            const uint8_t* mEnt = cache->base + mapsOff +
                                  (size_t)s * SHADER_MAP_BLOCK_SIZE;
            int32_t mRaw = R32(mEnt + 12);
            uint32_t mBmp = ((uint32_t)mRaw == 0xFFFFFFFFu)
                ? 0xFFFFFFFFu : (uint32_t)mRaw & 0xFFFFu;
            const char* mName = "<none>";
            char flags[8] = {0};
            if (mBmp != 0xFFFFFFFFu && mBmp < cache->tags.size()) {
                if (memcmp(cache->tags[mBmp].classCode, "bitm", 4) == 0) {
                    mName = cache->tags[mBmp].tagName.c_str();
                    int fi = 0;
                    if (BspBitmapIsInternal(mName)) flags[fi++] = 'I';
                    if (BspBitmapNameLooksLikeDetail(mName)) flags[fi++] = 'D';
                } else {
                    mName = "<not-bitm>";
                }
            }
            const char* shortBmp = mName;
            for (const char* p = mName; *p; ++p)
                if (*p == '\\' || *p == '/') shortBmp = p + 1;
            if (sdUsed + 96 < sizeof(slotDump)) {
                int n = _snprintf_s(slotDump + sdUsed, sizeof(slotDump) - sdUsed,
                    _TRUNCATE, "%s[%d]=%s%s%s", sdUsed == 0 ? "" : " ",
                    s, shortBmp, flags[0] ? ":" : "", flags);
                if (n > 0) sdUsed += (size_t)n;
            }
        }
        zh_mcc::NativeDiag(
            "BspShaderFAIL[0x%X] cls=%c%c%c%c tag='%s' mapCount=%d NO-DIFFUSE slots=%s",
            shaderTagId, shClass[0], shClass[1], shClass[2], shClass[3],
            shName, mapsBlk.count, slotDump);
    }
    return 0xFFFFFFFFu;
}

// -----------------------------------------------------------------------------
// Reach `rmtr` (terrain) 4-layer blend resolver.
//
// Walks the same rmsh -> rmt2.Usages[] -> ShaderMaps[] chain as
// ResolveDiffuseBitmapTagId, but instead of picking ONE diffuse it surfaces
// all five surface-texture slots used by terrain shaders:
//
//   base_map_m_0 / _m_1 / _m_2 / _m_3 - four base diffuse layers
//   blend_map - RGBA per-channel weight mask
//
// The shader's tag class must be `rmtr` (Reach terrain). Returns false for
// any other class (rmsh / rmd / rmhg / rmgl / etc.) so the caller falls back
// to the existing single-bitmap path. Diffuse-only - detail / normal /
// specular maps are out of scope for first-cut terrain rendering.
//
// Out fields not resolved are written as 0xFFFFFFFFu. IsTerrainBlend is set
// iff at least the blend_map AND base_map_m_0 resolved (otherwise the caller
// has nothing useful to render with and falls back).
// -----------------------------------------------------------------------------

struct TerrainLayersResolved {
    uint32_t base_m[4];
    uint32_t bump_m[4];
    uint32_t detail_m[4];
    uint32_t detail_bump_m[4];
    uint32_t blend_map;
    bool     is_terrain_blend;
    // Per-layer tile factors. Indexed by layer 0..3, .x and .y components.
    // Defaults to (1.0, 1.0) when the rmt2 didn't carry the corresponding
    // Float Constants entry. Detail/base ratio is what determines how much
    // faster detail tiles relative to base.
    float    base_tile[4][2];
    float    detail_tile[4][2];
    float    bump_tile[4][2];
    float    detail_bump_tile[4][2];
    // global_albedo_tint Vec4 (R, G, B, A) - multiplied over the whole
    // composited output. Defaults to (1,1,1,1) (engine no-op).
    float    global_albedo_tint[4];
    // TERRAIN_BLEND_FIX (Fix 1): blend_map_xform (sx, sy, ox, oy).
    // The engine samples the weight mask at uv*xy + zw; identity (1,1,0,0).
    float    blend_xform[4];
    // TERRAIN_BLEND_FIX (Fix 3): per-layer base/detail/bump/detail-bump
    // UV TRANSLATION (the .zw the engine's transform_texcoord adds). Identity (0,0).
    float    base_offset[4][2];
    float    detail_offset[4][2];
    float    bump_offset[4][2];
    float    detail_bump_offset[4][2];
    // TERRAIN_BLEND_FIX (Fix 2): bake-authored material_N_type active
    // mask (bit n = layer n authored active). 0 = unresolved (caller falls back).
    uint32_t authored_active_mask;
    // MAT-15: distance_blend_base far-distance base->target color lerp.
    //   blend_type: 0 = morph (default; base_blend forced 0 -> no-op),
    //               1 = distance_blend_base (apply the lerp), 0xFFFFFFFF unresolved.
    //   base_blend = saturate(dist*blend_slope + blend_offset);
    //   amount_N   = min(base_blend, blend_max_N);
    //   base       = lerp(base, blend_target_N, amount_N)  [terrain_new:102-129]
    uint32_t blend_type;
    float    blend_slope;
    float    blend_offset;
    float    blend_target[4][4];   // per-layer target COLOUR (rgba)
    float    blend_max[4];         // per-layer max blend amount (scalar)
};

// TERRAIN_BLEND_FIX (Fix 2): defined below (after ResolveShaderBlendMode,
// which it copies). Forward-declared so ResolveTerrainLayers can call it.
static uint8_t ResolveTerrainMaterialActiveMask(CacheHandle* cache, int32_t shaderTagId);
// MAT-15: forward-declared blend_type category resolver (defined after
// ResolveShaderBlendMode; returns 0 morph / 1 distance_blend_base / 0xFF unresolved).
static uint8_t ResolveTerrainBlendType(CacheHandle* cache, int32_t shaderTagId);

static bool ResolveTerrainLayers(CacheHandle* cache, int32_t shaderTagId,
                                 TerrainLayersResolved& out)
{
    out.base_m[0] = out.base_m[1] = out.base_m[2] = out.base_m[3] = 0xFFFFFFFFu;
    out.bump_m[0] = out.bump_m[1] = out.bump_m[2] = out.bump_m[3] = 0xFFFFFFFFu;
    out.detail_m[0] = out.detail_m[1] = out.detail_m[2] = out.detail_m[3] = 0xFFFFFFFFu;
    out.detail_bump_m[0] = out.detail_bump_m[1] = out.detail_bump_m[2] = out.detail_bump_m[3] = 0xFFFFFFFFu;
    for (int n = 0; n < 4; ++n) {
        out.base_tile[n][0] = 1.0f;
        out.base_tile[n][1] = 1.0f;
        out.detail_tile[n][0] = 1.0f;
        out.detail_tile[n][1] = 1.0f;
        out.bump_tile[n][0] = 1.0f;
        out.bump_tile[n][1] = 1.0f;
        out.detail_bump_tile[n][0] = 1.0f;
        out.detail_bump_tile[n][1] = 1.0f;
    }
    out.global_albedo_tint[0] = 1.0f;
    out.global_albedo_tint[1] = 1.0f;
    out.global_albedo_tint[2] = 1.0f;
    out.global_albedo_tint[3] = 1.0f;
    // TERRAIN_BLEND_FIX: engine-identity defaults for the appended
    // xform/offset/active-mask fields. Identity = today's exact behaviour.
    out.blend_xform[0] = 1.0f; out.blend_xform[1] = 1.0f;
    out.blend_xform[2] = 0.0f; out.blend_xform[3] = 0.0f;
    for (int n = 0; n < 4; ++n) {
        out.base_offset[n][0] = 0.0f; out.base_offset[n][1] = 0.0f;
        out.detail_offset[n][0] = 0.0f; out.detail_offset[n][1] = 0.0f;
        out.bump_offset[n][0] = 0.0f; out.bump_offset[n][1] = 0.0f;
        out.detail_bump_offset[n][0] = 0.0f; out.detail_bump_offset[n][1] = 0.0f;
    }
    out.authored_active_mask = 0u;
    // MAT-15: engine-identity defaults. blend_type 0xFFFFFFFF = unresolved -> the
    // Rust side treats it as morph (no-op), so an older/failed read never applies
    // the distance lerp. slope/offset 0 + max 0 => base_blend 0 => lerp no-op too.
    out.blend_type = 0xFFFFFFFFu;
    out.blend_slope = 0.0f;
    out.blend_offset = 0.0f;
    for (int n = 0; n < 4; ++n) {
        out.blend_target[n][0] = 0.0f; out.blend_target[n][1] = 0.0f;
        out.blend_target[n][2] = 0.0f; out.blend_target[n][3] = 0.0f;
        out.blend_max[n] = 0.0f;
    }
    out.blend_map = 0xFFFFFFFFu;
    out.is_terrain_blend = false;

    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size())
        return false;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return false;

    // Reach terrain shader class is `rmtr`. Only handle that - caller falls
    // back to single-diffuse path for `rmsh` and other render-method classes.
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm' ||
        te.classCode[2] != 't' || te.classCode[3] != 'r')
        return false;

    if (!cache->stringTableParsed) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size)
        return false;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return false;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 ||
        (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size)
        return false;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef mapsBlk = ReadTagBlock(props + OFF_SHADER_MAPS_IN_PROPS);
    if (mapsBlk.count <= 0) return false;
    int64_t mapsOff = TagMetaFileOff(cache, mapsBlk.pointer);
    if (mapsOff < 0 ||
        (size_t)mapsOff + (size_t)mapsBlk.count * SHADER_MAP_BLOCK_SIZE > cache->size)
        return false;

    // Walk rmt2.Usages[] for the strings we care about. The string IDs at
    // index `i` map 1:1 to ShaderMaps[i].
    int32_t rmtRawId = R32(props + 12);
    int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                       ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
    if (rmtTagId < 0 || (uint32_t)rmtTagId >= cache->tags.size()) return false;

    const TagEntry& rmtTe = cache->tags[rmtTagId];
    if (rmtTe.classIndex < 0) return false;
    if (rmtTe.classCode[0] != 'r' || rmtTe.classCode[1] != 'm' ||
        rmtTe.classCode[2] != 't') return false;

    int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
    if (rmtOff < 0 ||
        (size_t)rmtOff + OFF_RMT_USAGES + 8 > cache->size)
        return false;
    const uint8_t* rmtMeta = cache->base + rmtOff;
    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
    if (usagesBlk.count <= 0 || usagesBlk.count > 0x1000) return false;
    int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
    if (usagesOff < 0 ||
        (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE
            > cache->size)
        return false;

    int slotMax = mapsBlk.count;
    if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;

    // Helper: read bitm tag id at slot `i`, or 0xFFFFFFFFu if not a bitm.
    auto bitmAt = [&](int i) -> uint32_t {
        const uint8_t* mapEntry = cache->base + mapsOff +
                                  (size_t)i * SHADER_MAP_BLOCK_SIZE;
        int32_t rawId = R32(mapEntry + 12);
        if ((uint32_t)rawId == 0xFFFFFFFFu) return 0xFFFFFFFFu;
        uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
        if (bmpId >= cache->tags.size()) return 0xFFFFFFFFu;
        if (memcmp(cache->tags[bmpId].classCode, "bitm", 4) != 0)
            return 0xFFFFFFFFu;
        return bmpId;
    };

    for (int i = 0; i < slotMax; ++i) {
        int32_t sid = R32(cache->base + usagesOff +
                          (size_t)i * STRINGID_BLOCK_SIZE);
        const char* usageName = ResolveStringId(cache, sid);
        if (!usageName || !*usageName) continue;

        // Match - case-sensitive, exact strings (no _m_<n> stripping; the
        // suffix IS what selects the layer for terrain).
        if (strcmp(usageName, "base_map_m_0") == 0) {
            out.base_m[0] = bitmAt(i);
        } else if (strcmp(usageName, "base_map_m_1") == 0) {
            out.base_m[1] = bitmAt(i);
        } else if (strcmp(usageName, "base_map_m_2") == 0) {
            out.base_m[2] = bitmAt(i);
        } else if (strcmp(usageName, "base_map_m_3") == 0) {
            out.base_m[3] = bitmAt(i);
        } else if (strcmp(usageName, "bump_map_m_0") == 0) {
            out.bump_m[0] = bitmAt(i);
        } else if (strcmp(usageName, "bump_map_m_1") == 0) {
            out.bump_m[1] = bitmAt(i);
        } else if (strcmp(usageName, "bump_map_m_2") == 0) {
            out.bump_m[2] = bitmAt(i);
        } else if (strcmp(usageName, "bump_map_m_3") == 0) {
            out.bump_m[3] = bitmAt(i);
        } else if (strcmp(usageName, "detail_map_m_0") == 0) {
            out.detail_m[0] = bitmAt(i);
        } else if (strcmp(usageName, "detail_map_m_1") == 0) {
            out.detail_m[1] = bitmAt(i);
        } else if (strcmp(usageName, "detail_map_m_2") == 0) {
            out.detail_m[2] = bitmAt(i);
        } else if (strcmp(usageName, "detail_map_m_3") == 0) {
            out.detail_m[3] = bitmAt(i);
        } else if (strcmp(usageName, "detail_bump_m_0") == 0) {
            out.detail_bump_m[0] = bitmAt(i);
        } else if (strcmp(usageName, "detail_bump_m_1") == 0) {
            out.detail_bump_m[1] = bitmAt(i);
        } else if (strcmp(usageName, "detail_bump_m_2") == 0) {
            out.detail_bump_m[2] = bitmAt(i);
        } else if (strcmp(usageName, "detail_bump_m_3") == 0) {
            out.detail_bump_m[3] = bitmAt(i);
        } else if (strcmp(usageName, "blend_map") == 0) {
            out.blend_map = bitmAt(i);
        }
    }

    // Per-layer tile factors + global_albedo_tint live in the rmt2's
    // Arguments[] + the rmsh's ShaderProperties[0].Float Constants block.
    // Each Float Constant entry is a Vec4 (16 bytes); index by argIdx (the
    // position in rmt2.Arguments[] of the matching usage string). Mirrors
    // ResolveDiffuseTiling but for ALL terrain-layer usages at once.
    //
    // Constants are forward-declared locally so we can use them before the
    // canonical definitions further down in the file (same pattern as the
    // existing FWD_* fallback in ResolveDetailMap below).
    constexpr int LCL_OFF_RMT_ARGUMENTS         = 72;
    constexpr int LCL_OFF_TILING_DATA_IN_PROPS  = 28;
    constexpr int LCL_TILING_DATA_BLOCK_SIZE    = 16;

    TagBlockRef argsBlk = ReadTagBlock(rmtMeta + LCL_OFF_RMT_ARGUMENTS);
    TagBlockRef tilingBlk = ReadTagBlock(props + LCL_OFF_TILING_DATA_IN_PROPS);
    int64_t argsOff = (argsBlk.count > 0)
        ? TagMetaFileOff(cache, argsBlk.pointer) : -1;
    int64_t tilingOff = (tilingBlk.count > 0)
        ? TagMetaFileOff(cache, tilingBlk.pointer) : -1;
    bool argsOk = (argsBlk.count > 0 && argsBlk.count <= 0x1000 &&
                   argsOff >= 0 &&
                   (size_t)argsOff + (size_t)argsBlk.count * STRINGID_BLOCK_SIZE
                       <= cache->size);
    bool tilingOk = (tilingBlk.count > 0 && tilingBlk.count <= 0x1000 &&
                     tilingOff >= 0 &&
                     (size_t)tilingOff +
                         (size_t)tilingBlk.count * LCL_TILING_DATA_BLOCK_SIZE
                         <= cache->size);

    // Helper: locate argIdx by usage name. Falls back to string compare
    // when StringId equality fails (cross-pool case). Returns -1 on miss.
    auto findArgIdx = [&](const char* needle) -> int {
        if (!argsOk || !needle || !*needle) return -1;
        // First pass: StringId equality. Compute the StringId from the
        // needle by walking Args once and comparing resolved name strings - 
        // it's the simplest correct approach.
        for (int i = 0; i < argsBlk.count; ++i) {
            int32_t argSid = R32(cache->base + argsOff +
                                 (size_t)i * STRINGID_BLOCK_SIZE);
            const char* argName = ResolveStringId(cache, argSid);
            if (!argName) continue;
            if (strcmp(argName, needle) == 0) return i;
        }
        return -1;
    };

    auto readVec4 = [&](int argIdx, float* outXYZW) -> bool {
        if (!tilingOk) return false;
        if (argIdx < 0 || argIdx >= tilingBlk.count) return false;
        const uint8_t* entry = cache->base + tilingOff +
                               (size_t)argIdx * LCL_TILING_DATA_BLOCK_SIZE;
        memcpy(outXYZW + 0, entry + 0, 4);
        memcpy(outXYZW + 1, entry + 4, 4);
        memcpy(outXYZW + 2, entry + 8, 4);
        memcpy(outXYZW + 3, entry + 12, 4);
        return true;
    };

    auto sanitizeTile = [](float& v) {
        if (!std::isfinite(v) || v <= 0.0f || v > 1024.0f) v = 1.0f;
    };

    // Per-layer base + detail tile factors. Missing entries leave the
    // pre-set defaults (1.0, 1.0).
    static const char* kBaseNames[4] = {
        "base_map_m_0", "base_map_m_1", "base_map_m_2", "base_map_m_3"
    };
    static const char* kDetailNames[4] = {
        "detail_map_m_0", "detail_map_m_1", "detail_map_m_2", "detail_map_m_3"
    };
    static const char* kBumpNames[4] = {
        "bump_map_m_0", "bump_map_m_1", "bump_map_m_2", "bump_map_m_3"
    };
    static const char* kDetailBumpNames[4] = {
        "detail_bump_m_0", "detail_bump_m_1", "detail_bump_m_2", "detail_bump_m_3"
    };
    // TERRAIN_BLEND_FIX (Fix 3): keep the raw .zw OFFSET (v[2],v[3]) of
    // each map's xform - engine transform_texcoord does uv*xy + zw. Only the SCALE
    // (v[0],v[1]) is sanitized; the offset is kept raw (sanitizeTile forces
    // non-positive/large to 1.0, which would corrupt a legitimate 0/negative offset).
    for (int n = 0; n < 4; ++n) {
        int aIdx = findArgIdx(kBaseNames[n]);
        float v[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
        if (readVec4(aIdx, v)) {
            sanitizeTile(v[0]); sanitizeTile(v[1]);
            out.base_tile[n][0] = v[0];
            out.base_tile[n][1] = v[1];
            if (std::isfinite(v[2])) out.base_offset[n][0] = v[2];
            if (std::isfinite(v[3])) out.base_offset[n][1] = v[3];
        }
        aIdx = findArgIdx(kDetailNames[n]);
        v[0] = 1.0f; v[1] = 1.0f; v[2] = 0.0f; v[3] = 0.0f;
        if (readVec4(aIdx, v)) {
            sanitizeTile(v[0]); sanitizeTile(v[1]);
            out.detail_tile[n][0] = v[0];
            out.detail_tile[n][1] = v[1];
            if (std::isfinite(v[2])) out.detail_offset[n][0] = v[2];
            if (std::isfinite(v[3])) out.detail_offset[n][1] = v[3];
        }
        aIdx = findArgIdx(kBumpNames[n]);
        v[0] = 1.0f; v[1] = 1.0f; v[2] = 0.0f; v[3] = 0.0f;
        if (readVec4(aIdx, v)) {
            sanitizeTile(v[0]); sanitizeTile(v[1]);
            out.bump_tile[n][0] = v[0];
            out.bump_tile[n][1] = v[1];
            if (std::isfinite(v[2])) out.bump_offset[n][0] = v[2];
            if (std::isfinite(v[3])) out.bump_offset[n][1] = v[3];
        }
        aIdx = findArgIdx(kDetailBumpNames[n]);
        v[0] = 1.0f; v[1] = 1.0f; v[2] = 0.0f; v[3] = 0.0f;
        if (readVec4(aIdx, v)) {
            sanitizeTile(v[0]); sanitizeTile(v[1]);
            out.detail_bump_tile[n][0] = v[0];
            out.detail_bump_tile[n][1] = v[1];
            if (std::isfinite(v[2])) out.detail_bump_offset[n][0] = v[2];
            if (std::isfinite(v[3])) out.detail_bump_offset[n][1] = v[3];
        }
    }

    // TERRAIN_BLEND_FIX (Fix 1): blend_map_xform. Engine samples the
    // weight mask at transform_texcoord(uv, blend_map_xform) = uv*xy + zw
    // (terrain_new.hlsl_include:162). MMS sampled at bare uv -> every per-pixel
    // layer weight misregisters when the xform is non-identity. readVec4 already
    // reads all 4 floats. Sanitize ONLY the scale; keep the .zw offset raw.
    {
        int bIdx = findArgIdx("blend_map");
        float bx[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
        if (readVec4(bIdx, bx)) {
            sanitizeTile(bx[0]); sanitizeTile(bx[1]);
            out.blend_xform[0] = bx[0];
            out.blend_xform[1] = bx[1];
            if (std::isfinite(bx[2])) out.blend_xform[2] = bx[2];
            if (std::isfinite(bx[3])) out.blend_xform[3] = bx[3];
        }
    }

    // global_albedo_tint Vec4. Defaults to (1,1,1,1) if absent. Sanitize
    // each channel to [0, 4] - Reach tints are typically 0..1 with a few
    // shaders pushing slightly above to brighten; > 4 is bogus.
    {
        int tintIdx = findArgIdx("global_albedo_tint");
        float v[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        if (readVec4(tintIdx, v)) {
            for (int c = 0; c < 4; ++c) {
                if (!std::isfinite(v[c]) || v[c] < 0.0f || v[c] > 4.0f)
                    v[c] = 1.0f;
            }
            out.global_albedo_tint[0] = v[0];
            out.global_albedo_tint[1] = v[1];
            out.global_albedo_tint[2] = v[2];
            out.global_albedo_tint[3] = v[3];
        }
    }

    // TERRAIN_BLEND_FIX (Fix 2): read the bake-authored material_N_type
    // active set. 0 = unresolved -> adapter falls back to the bitmap-derived mask.
    out.authored_active_mask = ResolveTerrainMaterialActiveMask(cache, shaderTagId);

    // MAT-15: resolve blend_type + the distance_blend_base params. blend_type is a
    // render_method category (mirror of material_model); the params live in the same
    // Float Constants block as global_albedo_tint, read by rmt2.Arguments[] name.
    out.blend_type = ResolveTerrainBlendType(cache, shaderTagId);
    {
        int aIdx = findArgIdx("blend_slope");
        float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (readVec4(aIdx, v) && std::isfinite(v[0])) out.blend_slope = v[0];
        aIdx = findArgIdx("blend_offset");
        v[0] = 0.0f;
        if (readVec4(aIdx, v) && std::isfinite(v[0])) out.blend_offset = v[0];
        static const char* kTargetNames[4] = {
            "blend_target_0", "blend_target_1", "blend_target_2", "blend_target_3"
        };
        static const char* kMaxNames[4] = {
            "blend_max_0", "blend_max_1", "blend_max_2", "blend_max_3"
        };
        for (int n = 0; n < 4; ++n) {
            float t[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (readVec4(findArgIdx(kTargetNames[n]), t)) {
                for (int c = 0; c < 4; ++c)
                    out.blend_target[n][c] = std::isfinite(t[c]) ? t[c] : 0.0f;
            }
            float mx[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (readVec4(findArgIdx(kMaxNames[n]), mx) && std::isfinite(mx[0]))
                out.blend_max[n] = mx[0];
        }
        if (getenv("HMS_MAT15_DIAG") != nullptr) {
            fprintf(stderr, "HMS_MAT15_DIAG shaderTag=0x%X blend_type=%u slope=%.4f offset=%.4f "
                    "max=[%.3f,%.3f,%.3f,%.3f] tgt0=[%.3f,%.3f,%.3f]\n",
                    (unsigned)shaderTagId, out.blend_type, out.blend_slope, out.blend_offset,
                    out.blend_max[0], out.blend_max[1], out.blend_max[2], out.blend_max[3],
                    out.blend_target[0][0], out.blend_target[0][1], out.blend_target[0][2]);
        }
    }

    // Terrain blend is "valid" iff we resolved at least the blend mask AND
    // base layer 0. Otherwise the renderer can't compose anything sensible
    // and the caller should fall back to single-diffuse.
    bool ok = (out.blend_map != 0xFFFFFFFFu) &&
              (out.base_m[0] != 0xFFFFFFFFu);
    out.is_terrain_blend = ok;
    return ok;
}

// -----------------------------------------------------------------------------
// Per-shader blend mode resolution. Same chain as MapModelParser's
// ResolveShaderBlendMode: rmsh -> rmdf via rmsh.RenderMethodDefinitionRef,
// match the rmdf "blend_mode" Category to rmsh.ShaderOptions[catIdx], then
// map the option's Name string to our 0..5 enum.
// -----------------------------------------------------------------------------

constexpr int OFF_RMSH_RMDF_REF        = 0;
constexpr int OFF_RMSH_SHADER_OPTIONS  = 32;
constexpr int RMSH_SHADER_OPTION_SIZE  = 2;
constexpr int OFF_RMDF_CATEGORIES      = 16;
constexpr int RMDF_CATEGORY_SIZE       = 24;
constexpr int OFF_RMDF_CAT_OPTIONS     = 4;
constexpr int RMDF_OPTION_SIZE         = 28;

static uint8_t BspMapBlendModeStringToIndex(const char* s) {
    if (!s || !*s) return 0xFF;
    if (strcmp(s, "opaque") == 0)                       return 0;
    if (strcmp(s, "additive") == 0)                     return 1;
    if (strcmp(s, "multiply") == 0)                     return 2;
    if (strcmp(s, "double_multiply") == 0)              return 3;
    if (strcmp(s, "alpha_blend") == 0)                  return 4;
    if (strcmp(s, "add_src_times_srcalpha") == 0)       return 5;
    if (strcmp(s, "add_src_times_dstalpha") == 0)       return 5;
    // `pre_multiplied_alpha` is a distinct engine blend mode (Src=ONE,
    // Dst=INV_SRC_ALPHA); collapsing it to AlphaBlend (4) would double-apply
    // alpha. Route to PreMultipliedAlpha (6) to match MapModelParser and the
    // viewer's BlendModeId enum.
    if (strcmp(s, "pre_multiplied_alpha") == 0)         return 6;
    if (strcmp(s, "maximum") == 0)                      return 3;
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

    int blendCatIdx = -1;
    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
    for (int ci = 0; ci < catLimit; ++ci) {
        const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
        int32_t catNameSid = R32(catEntry + 0);
        const char* catName = ResolveStringId(cache, catNameSid);
        if (!catName) continue;
        if (strcmp(catName, "blend_mode") == 0) { blendCatIdx = ci; break; }
    }
    if (blendCatIdx < 0) return 0xFF;

    const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                    (size_t)blendCatIdx * RMSH_SHADER_OPTION_SIZE;
    int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                    ((uint16_t)shaderOptEntry[1] << 8));
    if (optionIndex < 0) return 0xFF;

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
    return BspMapBlendModeStringToIndex(optName);
}

// -----------------------------------------------------------------------------
// MAT-1: ResolveShaderMaterialModel (BSP-side mirror). Same rmsh -> rmdf categories
// -> rmsh.ShaderOptions[catIdx] -> option Name walk as ResolveShaderBlendMode, but
// matches the `material_model` category and maps the option Name to the 0..9
// MATERIAL_TYPE_* enum (material_models.hlsl_include:12-31). 0xFF on any failure.
// -----------------------------------------------------------------------------
static uint8_t BspMaterialModelStringToIndex(const char* s) {
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
    return BspMaterialModelStringToIndex(optName);
}

// -----------------------------------------------------------------------------
// ALBEDO-VARIANT (protomorph albedo_fx.hlsl): the `albedo` render-method category
// picks the base/detail composite (default vs two-detail vs overlay vs ...). HMS
// only did `default` (base*detail*4.59) for every material -> multi-detail materials
// rendered FLAT/wrong contrast. This resolver (clone of ResolveShaderMaterialModel,
// matching the `albedo` category) surfaces the option so the shader can apply the
// exact per-variant formula. Enum (only the ones HMS composites; others -> 0 default):
//   0 default / 1 two_detail / 2 two_detail_black_point / 3 two_detail_overlay /
//   4 detail_blend / 5 three_detail_blend / 6 color_mask / 7 constant_color.
// 0xFF on failure (Rust maps -> 0 default = the current single-detail path, no regression).
static uint8_t BspAlbedoOptionStringToIndex(const char* s) {
    if (!s || !*s) return 0xFF;
    if (strcmp(s, "default") == 0)                 return 0;
    if (strcmp(s, "two_detail") == 0)              return 1;
    if (strcmp(s, "two_detail_black_point") == 0)  return 2;
    if (strcmp(s, "two_detail_overlay") == 0)      return 3;
    if (strcmp(s, "detail_blend") == 0)            return 4;
    if (strcmp(s, "three_detail_blend") == 0)      return 5;
    if (strcmp(s, "color_mask") == 0)              return 6;
    if (strcmp(s, "constant_color") == 0)          return 7;
    return 0; // any other albedo option (chameleon, waterfall, ...) -> treat as default
}

uint8_t ResolveShaderAlbedoOption(CacheHandle* cache, int32_t shaderTagId) {
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

    int albCatIdx = -1;
    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
    for (int ci = 0; ci < catLimit; ++ci) {
        const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
        int32_t catNameSid = R32(catEntry + 0);
        const char* catName = ResolveStringId(cache, catNameSid);
        if (!catName) continue;
        if (strcmp(catName, "albedo") == 0) { albCatIdx = ci; break; }
    }
    if (albCatIdx < 0) return 0xFF;

    const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                    (size_t)albCatIdx * RMSH_SHADER_OPTION_SIZE;
    int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                    ((uint16_t)shaderOptEntry[1] << 8));
    if (optionIndex < 0) return 0xFF;

    const uint8_t* catEntry = cache->base + catsOff + (size_t)albCatIdx * RMDF_CATEGORY_SIZE;
    TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
    if (optsBlk.count <= 0 || optsBlk.count > 0x1000) return 0xFF;
    if (optionIndex >= optsBlk.count) return 0xFF;
    int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
    if (optsOff < 0 ||
        (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE > cache->size)
        return 0xFF;

    const uint8_t* aoptEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
    int32_t aoptNameSid = R32(aoptEntry + 0);
    const char* aoptName = ResolveStringId(cache, aoptNameSid);
    return BspAlbedoOptionStringToIndex(aoptName);
}

// -----------------------------------------------------------------------------
// LIT-SI-3: ResolveShaderSelfIllumMode (BSP-side). Identical rmsh -> rmdf category
// walk as ResolveShaderMaterialModel, but matches the `self_illumination` category
// and maps the selected option Name to a self-illum MODE enum. The option Name
// strings (and their fixed order in shader.render_method_definition) are:
//   0 off / 1 simple / 2 3_channel_self_illum / 3 plasma / 4 from_diffuse /
//   5 illum_detail / 6 meter / 7 self_illum_times_diffuse / 8 simple_with_alpha_mask /
//   9 multilayer_additive / 10 palettized_plasma / 11 change_color / 12 change_color_detail.
// The Rust enum mirrors this order. 0xFF on any failure (Rust maps -> 1 simple/default,
// which is the current HMS single-composite path, so an unresolved read is a no-op).
// -----------------------------------------------------------------------------
static uint8_t BspSelfIllumModeStringToIndex(const char* s) {
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
    return BspSelfIllumModeStringToIndex(optName);
}

// -----------------------------------------------------------------------------
// TERRAIN_BLEND_FIX (Fix 2): bake-authored terrain active-material set.
// Copies the ResolveShaderBlendMode walk verbatim but loops the four category
// names material_0_type..material_3_type instead of the single "blend_mode".
// For each found category it resolves the selected option's Name string and sets
// bit n when it is "diffuse_only" or "diffuse_plus_specular" (engine
// ACTIVE_MATERIAL == 1, terrain_new.hlsl_include:72-76); "off" / unknown / missing
// leaves the bit clear. Returns the 4-bit mask. A mask of 0 (no category resolved)
// signals "unresolved" so the caller falls back to the bitmap-derived mask.
//
// The terrain shader class is `rmtr` (a render_method subtype) - same rmdf-ref @+0
// and ShaderOptions @+32 header layout as `rmsh`, so the chain is identical.
static uint8_t ResolveTerrainMaterialActiveMask(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0;
    if (!cache->stringTableParsed) return 0;

    int64_t rmshOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (rmshOff < 0 || (size_t)rmshOff + OFF_SHADER_PROPS + 8 > cache->size) return 0;
    const uint8_t* rmsh = cache->base + rmshOff;

    int32_t rmdfRawId = R32(rmsh + OFF_RMSH_RMDF_REF + 12);
    int32_t rmdfTagId = ((uint32_t)rmdfRawId == 0xFFFFFFFFu)
                        ? -1 : (int32_t)((uint32_t)rmdfRawId & 0xFFFFu);
    if (rmdfTagId < 0 || (uint32_t)rmdfTagId >= cache->tags.size()) return 0;
    const TagEntry& rmdfTe = cache->tags[rmdfTagId];
    if (rmdfTe.classIndex < 0) return 0;
    if (rmdfTe.classCode[0] != 'r' || rmdfTe.classCode[1] != 'm' ||
        rmdfTe.classCode[2] != 'd' || rmdfTe.classCode[3] != 'f') return 0;

    TagBlockRef shaderOptsBlk = ReadTagBlock(rmsh + OFF_RMSH_SHADER_OPTIONS);
    if (shaderOptsBlk.count <= 0 || shaderOptsBlk.count > 0x1000) return 0;
    int64_t shaderOptsOff = TagMetaFileOff(cache, shaderOptsBlk.pointer);
    if (shaderOptsOff < 0 ||
        (size_t)shaderOptsOff + (size_t)shaderOptsBlk.count * RMSH_SHADER_OPTION_SIZE > cache->size)
        return 0;

    int64_t rmdfMetaOff = TagMetaFileOff(cache, rmdfTe.metaPointerRaw);
    if (rmdfMetaOff < 0 || (size_t)rmdfMetaOff + OFF_RMDF_CATEGORIES + 8 > cache->size) return 0;
    const uint8_t* rmdfMeta = cache->base + rmdfMetaOff;
    TagBlockRef catsBlk = ReadTagBlock(rmdfMeta + OFF_RMDF_CATEGORIES);
    if (catsBlk.count <= 0 || catsBlk.count > 0x1000) return 0;
    int64_t catsOff = TagMetaFileOff(cache, catsBlk.pointer);
    if (catsOff < 0 ||
        (size_t)catsOff + (size_t)catsBlk.count * RMDF_CATEGORY_SIZE > cache->size)
        return 0;

    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;

    static const char* kMaterialTypeNames[4] = {
        "material_0_type", "material_1_type", "material_2_type", "material_3_type"
    };

    uint8_t mask = 0;
    for (int layer = 0; layer < 4; ++layer) {
        // Locate the rmdf category whose name == material_<layer>_type.
        int catIdx = -1;
        for (int ci = 0; ci < catLimit; ++ci) {
            const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
            int32_t catNameSid = R32(catEntry + 0);
            const char* catName = ResolveStringId(cache, catNameSid);
            if (!catName) continue;
            if (strcmp(catName, kMaterialTypeNames[layer]) == 0) { catIdx = ci; break; }
        }
        if (catIdx < 0) continue;   // category missing -> leave bit clear (fallback)

        const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                        (size_t)catIdx * RMSH_SHADER_OPTION_SIZE;
        int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                        ((uint16_t)shaderOptEntry[1] << 8));
        if (optionIndex < 0) continue;

        const uint8_t* catEntry = cache->base + catsOff + (size_t)catIdx * RMDF_CATEGORY_SIZE;
        TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
        if (optsBlk.count <= 0 || optsBlk.count > 0x1000) continue;
        if (optionIndex >= optsBlk.count) continue;
        int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
        if (optsOff < 0 ||
            (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE > cache->size)
            continue;

        const uint8_t* optEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
        int32_t optNameSid = R32(optEntry + 0);
        const char* optName = ResolveStringId(cache, optNameSid);
        if (!optName) continue;
        if (strcmp(optName, "diffuse_only") == 0 ||
            strcmp(optName, "diffuse_plus_specular") == 0) {
            mask |= (uint8_t)(1u << layer);
        }
        // "off" (or any other option name) leaves the bit clear (inactive).
    }
    return mask;
}

// -----------------------------------------------------------------------------
// MAT-15: ResolveTerrainBlendType. The terrain far-distance base->target lerp is
// gated by the `blend_type` render_method category (terrain_new.hlsl_include:102).
// Same rmsh(rmtr)->rmdf categories -> ShaderOptions[catIdx] -> option Name walk as
// ResolveShaderMaterialModel, matching category "blend_type" and mapping the option:
//   "morph" (or any non-distance option) -> 0 ; "distance_blend_base" -> 1.
// 0xFF on any failure (Rust treats unresolved as morph = no-op). HMS_MAT15_DIAG
// dumps the category + selected option to stderr.
// -----------------------------------------------------------------------------
static uint8_t ResolveTerrainBlendType(CacheHandle* cache, int32_t shaderTagId) {
    if (!cache) return 0xFF;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0xFF;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0xFF;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFF;
    if (!cache->stringTableParsed) return 0xFF;

    const bool diag = (getenv("HMS_MAT15_DIAG") != nullptr);

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

    // The terrain (rmtr) far-distance base-blend mode is selected by the `blending`
    // render_method category (option "morph" | "distance_blend_base"). (Earlier RE
    // doc called it blend_type after the HLSL macro; the tag category is `blending`.)
    int btCatIdx = -1;
    int catLimit = catsBlk.count;
    if (catLimit > shaderOptsBlk.count) catLimit = shaderOptsBlk.count;
    for (int ci = 0; ci < catLimit; ++ci) {
        const uint8_t* catEntry = cache->base + catsOff + (size_t)ci * RMDF_CATEGORY_SIZE;
        int32_t catNameSid = R32(catEntry + 0);
        const char* catName = ResolveStringId(cache, catNameSid);
        if (diag && catName) fprintf(stderr, "  MAT15 cat[%d]='%s'\n", ci, catName);
        if (catName && (strcmp(catName, "blending") == 0 ||
                        strcmp(catName, "blend_type") == 0)) { btCatIdx = ci; if (!diag) break; }
    }
    if (btCatIdx < 0) return 0xFF;

    const uint8_t* shaderOptEntry = cache->base + shaderOptsOff +
                                    (size_t)btCatIdx * RMSH_SHADER_OPTION_SIZE;
    int16_t optionIndex = (int16_t)((uint16_t)shaderOptEntry[0] |
                                    ((uint16_t)shaderOptEntry[1] << 8));
    if (optionIndex < 0) return 0xFF;

    const uint8_t* catEntry = cache->base + catsOff + (size_t)btCatIdx * RMDF_CATEGORY_SIZE;
    TagBlockRef optsBlk = ReadTagBlock(catEntry + OFF_RMDF_CAT_OPTIONS);
    if (optsBlk.count <= 0 || optsBlk.count > 0x1000) return 0xFF;
    if (optionIndex >= optsBlk.count) return 0xFF;
    int64_t optsOff = TagMetaFileOff(cache, optsBlk.pointer);
    if (optsOff < 0 ||
        (size_t)optsOff + (size_t)optsBlk.count * RMDF_OPTION_SIZE > cache->size)
        return 0xFF;

    const uint8_t* optEntry = cache->base + optsOff + (size_t)optionIndex * RMDF_OPTION_SIZE;
    const char* optName = ResolveStringId(cache, R32(optEntry + 0));
    if (diag) fprintf(stderr, "  MAT15 blend_type option='%s'\n", optName ? optName : "<?>");
    if (!optName || !*optName) return 0xFF;
    if (strcmp(optName, "distance_blend_base") == 0) return 1;
    return 0;   // morph or any other option => no distance lerp
}

// -----------------------------------------------------------------------------
// SEH wrappers
// -----------------------------------------------------------------------------

bool SehParseSbspTag(CacheHandle* cache, uint32_t sbspTagId, BspData& data) {
    __try { return ParseSbspTag(cache, sbspTagId, data); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeMeshGeometry(BspData* bsp, uint32_t meshIndex,
                           uint8_t** outV, uint32_t* outVLen,
                           uint8_t** outI, uint32_t* outILen)
{
    __try { return DecodeMeshGeometryInner(bsp, meshIndex, outV, outVLen, outI, outILen); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeMeshUVs(BspData* bsp, uint32_t meshIndex,
                      float** outUv, uint32_t* outUvCount)
{
    __try { return DecodeMeshUVsInner(bsp, meshIndex, outUv, outUvCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeMeshUV2s(BspData* bsp, uint32_t meshIndex,
                       float** outUv, uint32_t* outUvCount)
{
    __try { return DecodeMeshUV2sInner(bsp, meshIndex, outUv, outUvCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeMeshNormals(BspData* bsp, uint32_t meshIndex,
                          float** outN, uint32_t* outNCount)
{
    __try { return DecodeMeshNormalsInner(bsp, meshIndex, outN, outNCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeMeshTangents(BspData* bsp, uint32_t meshIndex,
                           float** outT, uint32_t* outTCount)
{
    __try { return DecodeMeshTangentsInner(bsp, meshIndex, outT, outTCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

bool SehDecodeMeshBinormals(BspData* bsp, uint32_t meshIndex,
                            float** outB, uint32_t* outBCount)
{
    __try { return DecodeMeshBinormalsInner(bsp, meshIndex, outB, outBCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

uint32_t SehResolveDiffuse(CacheHandle* cache, int32_t shaderTagId) {
    __try { return ResolveDiffuseBitmapTagId(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

uint8_t SehResolveBlendMode(CacheHandle* cache, int32_t shaderTagId) {
    __try { return ResolveShaderBlendMode(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

// MAT-1: SEH-guarded material-model resolve (same contract as SehResolveBlendMode).
uint8_t SehResolveMaterialModel(CacheHandle* cache, int32_t shaderTagId) {
    __try { return ResolveShaderMaterialModel(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

// ALBEDO-VARIANT: SEH-guarded albedo-option resolve (same contract).
uint8_t SehResolveAlbedoOption(CacheHandle* cache, int32_t shaderTagId) {
    __try { return ResolveShaderAlbedoOption(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

// LIT-SI-3: SEH-guarded self-illum-mode resolve (same contract).
uint8_t SehResolveSelfIllumMode(CacheHandle* cache, int32_t shaderTagId) {
    __try { return ResolveShaderSelfIllumMode(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFF; }
}

bool SehResolveTerrainLayers(CacheHandle* cache, int32_t shaderTagId,
                              TerrainLayersResolved& out)
{
    __try { return ResolveTerrainLayers(cache, shaderTagId, out); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        out.base_m[0] = out.base_m[1] = out.base_m[2] = out.base_m[3] = 0xFFFFFFFFu;
        out.bump_m[0] = out.bump_m[1] = out.bump_m[2] = out.bump_m[3] = 0xFFFFFFFFu;
        out.detail_m[0] = out.detail_m[1] = out.detail_m[2] = out.detail_m[3] = 0xFFFFFFFFu;
        out.detail_bump_m[0] = out.detail_bump_m[1] = out.detail_bump_m[2] = out.detail_bump_m[3] = 0xFFFFFFFFu;
        for (int n = 0; n < 4; ++n) {
            out.base_tile[n][0] = 1.0f;
            out.base_tile[n][1] = 1.0f;
            out.detail_tile[n][0] = 1.0f;
            out.detail_tile[n][1] = 1.0f;
            out.bump_tile[n][0] = 1.0f;
            out.bump_tile[n][1] = 1.0f;
            out.detail_bump_tile[n][0] = 1.0f;
            out.detail_bump_tile[n][1] = 1.0f;
        }
        out.global_albedo_tint[0] = 1.0f;
        out.global_albedo_tint[1] = 1.0f;
        out.global_albedo_tint[2] = 1.0f;
        out.global_albedo_tint[3] = 1.0f;
        out.blend_type = 0xFFFFFFFFu;   // MAT-15: unresolved => morph (no-op)
        out.blend_slope = 0.0f;
        out.blend_offset = 0.0f;
        for (int n = 0; n < 4; ++n) {
            out.blend_target[n][0] = 0.0f; out.blend_target[n][1] = 0.0f;
            out.blend_target[n][2] = 0.0f; out.blend_target[n][3] = 0.0f;
            out.blend_max[n] = 0.0f;
        }
        out.blend_map = 0xFFFFFFFFu;
        out.is_terrain_blend = false;
        return false;
    }
}

// rmt2 Arguments tagblock offset (the canonical TilingData constants are
// defined further down next to ResolveDiffuseTiling).
constexpr int FWD_OFF_RMT_ARGUMENTS         = 72;

// Detail-map bitmap tag id for a shader, plus its TilingData (the engine
// composites base x detail at sample time).
struct DetailMapInfo {
    uint32_t bitmapTagId;
    float    tileX;
    float    tileY;
};

// -----------------------------------------------------------------------------
// Per-shader diffuse-slot UV tiling resolver.
//
// Reach BSP textures are designed to TILE - each shader's rmt2 carries a
// per-argument tiling table (RealVector4 array; X/Y are TileU/TileV) that
// the engine multiplies by the vertex UVs at sample time. Without this scale
// walls/floors render at native UV space and look stretched / low-res.
//
// Reclaimer reference:
//   * Gen3MaterialHelper.PopulateTextureMappings - picks usage by index in
//     rmt2.Usages, then `tileIndex = arguments.IndexOf(usage)` and reads
//     `tilingData[tileIndex]`. (i.e. the tiling slot is keyed by the usage
//     STRING - `arguments` and `usages` are not the same length.)
//   * render_method_template.cs - Arguments at +72, Usages at +108
//   * shader.cs - ShaderProperties at +56 (size 172),
//                                   ShaderMaps at +16, TilingData at +28.
//   * RealVector4 is 16 bytes - TilingData stride 16, X = bytes [0..3],
//                                Y = bytes [4..7].
//
// Walk:
//   1. rmsh -> ShaderProperties[0] -> ShaderMaps[] / Usages chain (same as
//      ResolveDiffuseBitmapTagId).
//   2. Pick the diffuse slot using the existing rank table - gives both the
//      slot index in Usages[] AND its usage string ("base_map" etc).
//   3. rmsh -> ShaderProperties[0] -> rmt2 -> Arguments[] - find the index
//      of the same usage string. (StringId compare.)
//   4. rmsh -> ShaderProperties[0] -> TilingData[argIdx] -> read first two
//      floats.
//
// Returns true on success with TileX/TileY filled in. Returns false on any
// failure - caller substitutes 1.0 (no scale).
// -----------------------------------------------------------------------------

constexpr int OFF_RMT_ARGUMENTS         = 72;     // render_method_template.Arguments
constexpr int OFF_TILING_DATA_IN_PROPS  = 28;     // ShaderPropertiesBlock.TilingData
constexpr int TILING_DATA_BLOCK_SIZE    = 16;     // RealVector4

bool ResolveDiffuseTiling(CacheHandle* cache, int32_t shaderTagId,
                          int32_t materialIndexForDiag,
                          float& outTileX, float& outTileY)
{
    outTileX = 1.0f;
    outTileY = 1.0f;
    if (!cache) return false;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size())
        return false;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return false;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return false;
    if (!cache->stringTableParsed) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size)
        return false;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return false;

    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 ||
        (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size)
        return false;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef mapsBlk = ReadTagBlock(props + OFF_SHADER_MAPS_IN_PROPS);
    if (mapsBlk.count <= 0) return false;

    // rmt2 lookup (same pattern as ResolveDiffuseBitmapTagId).
    int32_t rmtRawId = R32(props + 12);
    int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                       ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
    if (rmtTagId < 0 || (uint32_t)rmtTagId >= cache->tags.size()) return false;
    const TagEntry& rmtTe = cache->tags[rmtTagId];
    if (rmtTe.classIndex < 0) return false;
    if (rmtTe.classCode[0] != 'r' || rmtTe.classCode[1] != 'm' ||
        rmtTe.classCode[2] != 't') return false;

    int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
    // Need both Arguments (+72) and Usages (+108) to be in-range.
    if (rmtOff < 0 ||
        (size_t)rmtOff + OFF_RMT_USAGES + 8 > cache->size)
        return false;
    const uint8_t* rmtMeta = cache->base + rmtOff;

    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
    if (usagesBlk.count <= 0 || usagesBlk.count > 0x1000) return false;
    int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
    if (usagesOff < 0 ||
        (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE
            > cache->size)
        return false;

    TagBlockRef argsBlk = ReadTagBlock(rmtMeta + OFF_RMT_ARGUMENTS);
    if (argsBlk.count <= 0 || argsBlk.count > 0x1000) return false;
    int64_t argsOff = TagMetaFileOff(cache, argsBlk.pointer);
    if (argsOff < 0 ||
        (size_t)argsOff + (size_t)argsBlk.count * STRINGID_BLOCK_SIZE
            > cache->size)
        return false;

    // Step 1: pick the diffuse usage slot in rmt2.Usages[] using the same
    // rank table as ResolveDiffuseBitmapTagId. We need both:
    //   * the usage string (to look up in Arguments[])
    //   * the StringId (to compare directly without string roundtrip - and
    //     to handle "_m_<n>" suffix variants which strcmp on the resolved
    //     name correctly preserves).
    int slotMax = mapsBlk.count;
    if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;
    int bestRank = (int)(sizeof(kBspDiffuseUsages) /
                         sizeof(kBspDiffuseUsages[0]));
    int diffuseUsageIdx = -1;
    int32_t diffuseUsageSid = 0;
    for (int i = 0; i < slotMax; ++i) {
        int32_t sid = R32(cache->base + usagesOff +
                          (size_t)i * STRINGID_BLOCK_SIZE);
        const char* usageName = ResolveStringId(cache, sid);
        if (!usageName || !*usageName) continue;
        int rank = BspMatchDiffuseUsageRank(usageName);
        if (rank < 0 || rank >= bestRank) continue;
        bestRank = rank;
        diffuseUsageIdx = i;
        diffuseUsageSid = sid;
        if (rank == 0) break;
    }
    if (diffuseUsageIdx < 0) {
        // Nothing matched the diffuse rank table - emit a bounded diag and
        // bail. Caller substitutes 1.0.
        static std::atomic<int> s_noDiffBudget{ 64 };
        int v = s_noDiffBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_noDiffBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                __try {
                    zh_mcc::NativeDiag(
                        "TilingProbe: shader=0x%04x matIdx=%d argIdx not found (no diffuse usage)",
                        (unsigned)(shaderTagId & 0xFFFF), materialIndexForDiag);
                } __except (EXCEPTION_EXECUTE_HANDLER) { }
                break;
            }
        }
        return false;
    }

    // Resolve the usage name string ONCE for the diag + Arguments[] match.
    const char* diffuseUsageName = ResolveStringId(cache, diffuseUsageSid);
    if (!diffuseUsageName || !*diffuseUsageName) return false;

    // Step 2: find the matching string in Arguments[]. StringId equality
    // first (cheap, common case - the engine reuses the same pool), with a
    // strcmp fallback so we still match across pool boundaries.
    int argIdx = -1;
    for (int i = 0; i < argsBlk.count; ++i) {
        int32_t argSid = R32(cache->base + argsOff +
                             (size_t)i * STRINGID_BLOCK_SIZE);
        if (argSid == diffuseUsageSid) { argIdx = i; break; }
    }
    if (argIdx < 0) {
        for (int i = 0; i < argsBlk.count; ++i) {
            int32_t argSid = R32(cache->base + argsOff +
                                 (size_t)i * STRINGID_BLOCK_SIZE);
            const char* argName = ResolveStringId(cache, argSid);
            if (!argName) continue;
            if (strcmp(argName, diffuseUsageName) == 0) { argIdx = i; break; }
        }
    }
    if (argIdx < 0) {
        // Diffuse usage exists but no matching Argument - fall through to 1.0
        // and log it. This is the case Gen3MaterialHelper handles by returning
        // RealVector4(1,1,1,1) - preserve that behavior.
        static std::atomic<int> s_noArgBudget{ 64 };
        int v = s_noArgBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_noArgBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                __try {
                    zh_mcc::NativeDiag(
                        "TilingProbe: shader=0x%04x matIdx=%d usage='%s' argIdx not found (defaulting to 1.0)",
                        (unsigned)(shaderTagId & 0xFFFF), materialIndexForDiag,
                        diffuseUsageName);
                } __except (EXCEPTION_EXECUTE_HANDLER) { }
                break;
            }
        }
        return false;
    }

    // Step 3: read TilingData[argIdx].XY (RealVector4, first two floats).
    TagBlockRef tilingBlk = ReadTagBlock(props + OFF_TILING_DATA_IN_PROPS);
    if (tilingBlk.count <= 0 || tilingBlk.count > 0x1000) return false;
    if (argIdx >= tilingBlk.count) {
        // arg index out of TilingData bounds - log and fall back. Real
        // Reclaimer behavior is to return RealVector4(1,1,1,1) for this case;
        // we mirror that.
        static std::atomic<int> s_oobBudget{ 64 };
        int v = s_oobBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_oobBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                __try {
                    zh_mcc::NativeDiag(
                        "TilingProbe: shader=0x%04x matIdx=%d usage='%s' argIdx=%d OOB (TilingData count=%d, defaulting to 1.0)",
                        (unsigned)(shaderTagId & 0xFFFF), materialIndexForDiag,
                        diffuseUsageName, argIdx, tilingBlk.count);
                } __except (EXCEPTION_EXECUTE_HANDLER) { }
                break;
            }
        }
        return false;
    }
    int64_t tilingOff = TagMetaFileOff(cache, tilingBlk.pointer);
    if (tilingOff < 0 ||
        (size_t)tilingOff + (size_t)tilingBlk.count * TILING_DATA_BLOCK_SIZE
            > cache->size)
        return false;

    const uint8_t* entry = cache->base + tilingOff +
                           (size_t)argIdx * TILING_DATA_BLOCK_SIZE;
    float tx = 1.0f, ty = 1.0f;
    memcpy(&tx, entry + 0, 4);
    memcpy(&ty, entry + 4, 4);

    // Sanitize - bad data (NaN, ridiculous values) substitutes 1.0 instead
    // of corrupting the UVs.
    if (!std::isfinite(tx) || tx <= 0.0f || tx > 1024.0f) tx = 1.0f;
    if (!std::isfinite(ty) || ty <= 0.0f || ty > 1024.0f) ty = 1.0f;

    outTileX = tx;
    outTileY = ty;

    // One bounded diag per material - gives "TilingProbe: shader=0xN matIdx=N
    // usage='base_map' argIdx=N tile=(4.0, 4.0)" which is what the spec asks
    // for.
    static std::atomic<int> s_okBudget{ 256 };
    int v = s_okBudget.load(std::memory_order_relaxed);
    while (v > 0) {
        if (s_okBudget.compare_exchange_weak(v, v - 1,
            std::memory_order_relaxed, std::memory_order_relaxed))
        {
            __try {
                zh_mcc::NativeDiag(
                    "TilingProbe: shader=0x%04x matIdx=%d usage='%s' argIdx=%d tile=(%.3f, %.3f)",
                    (unsigned)(shaderTagId & 0xFFFF), materialIndexForDiag,
                    diffuseUsageName, argIdx, tx, ty);
            } __except (EXCEPTION_EXECUTE_HANDLER) { }
            break;
        }
    }
    return true;
}

bool SehResolveDiffuseTiling(CacheHandle* cache, int32_t shaderTagId,
                             int32_t materialIndexForDiag,
                             float& outTileX, float& outTileY)
{
    __try { return ResolveDiffuseTiling(cache, shaderTagId, materialIndexForDiag,
                                        outTileX, outTileY); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        outTileX = 1.0f;
        outTileY = 1.0f;
        return false;
    }
}

// -----------------------------------------------------------------------------
// Shader-constants reader - RealConstants (vec4 per arg slot) + ScalarConstants
// (float per slot, bit-cast from Integer Constants in Reach). Driven by the
// rmt2's Arguments[] string list which gives each slot a usage name.
//
// Layout (MccHaloReach Retail rmsh, see header for the full breakdown):
//   ShaderProperties[0] +0x1C : Float Constants tagblock, 16B/elem -> Reals
//   ShaderProperties[0] +0x28 : Integer Constants tagblock, 4B/elem -> Scalars
//
// The Float Constants block is the SAME physical block Reclaimer / the existing
// tiling resolver call "TilingData". Reach packs UV-tile data and tint /
// wave-height / specular vec4s into one array. Caller distinguishes by name
// (rmt2.Arguments[] / rmt2.Usages[]).
//
// Confidence: HIGH for RealConstants offset (matches working tiling code).
// MEDIUM for ScalarConstants - the int32 Integer Constants block is the only
// scalar-shaped storage in Reach, but we bit-cast to float in case the engine
// uses the slot for floats.
// -----------------------------------------------------------------------------

constexpr int OFF_REAL_CONSTS_IN_PROPS   = 28;     // 0x1C - same as TilingData
constexpr int REAL_CONSTS_BLOCK_SIZE     = 16;     // RealVector4
constexpr int OFF_SCALAR_CONSTS_IN_PROPS = 40;     // 0x28 - Integer Constants
constexpr int SCALAR_CONSTS_BLOCK_SIZE   = 4;      // int32 / float bit-cast

constexpr uint32_t SHADER_CONSTS_MAX = 64;
// ARG_NAME_WIDEN: per-slot arg-name buffer widened from 32 to 48.
// The old 32-byte slot left only 31 usable chars (strnlen caps at LEN-1), which
// silently truncated the longest authored Reach render_method arg names:
//   environment_map_specular_contribution (37)  -> "...contri"   DROPPED
//   analytical_specular_contribution       (32)  -> "...contributio"
//   material_texture_black_roughness       (32)  -> "...roughnes"
// Those are real cook_torrance / env-map specular params; the truncation forced
// the viewer to carry fragile "match both full + truncated" fallbacks and any
// future >=32-char param would defaulted-to-0 silently. 48 fits the 37-char max
// with headroom and keeps 8-byte struct alignment. NOTE: this changes the
// ZH_ShaderConstants binary layout - MapBspParser.h, MapBspParserInterop.cs and
// the BspMeshDiskCache version constant were updated in lockstep.
constexpr uint32_t SHADER_CONSTS_NAME_LEN = 48;

// Forward-declared in MapBspParser.h. Re-declared locally so the helper can be
// shared with MapModelParser.cpp without forcing a circular include.
struct ZH_ShaderConstantsImpl {
    uint32_t RealCount;
    uint32_t ScalarCount;
    float    RealConstants[SHADER_CONSTS_MAX * 4];
    float    ScalarConstants[SHADER_CONSTS_MAX];
    char     ArgNames[SHADER_CONSTS_MAX * SHADER_CONSTS_NAME_LEN];
    uint32_t Reserved[8];
};
static_assert(sizeof(ZH_ShaderConstantsImpl) ==
              8 + 64 * 16 + 64 * 4 + 64 * 48 + 32,
              "ZH_ShaderConstantsImpl size mismatch");

// Inner walker - caller wraps in SEH. Zero-init the out struct on entry.
bool ResolveShaderConstantsInner(CacheHandle* cache, int32_t shaderTagId,
                                 int32_t materialIndexForDiag,
                                 ZH_ShaderConstantsImpl* outConsts)
{
    if (!outConsts) return false;
    memset(outConsts, 0, sizeof(*outConsts));
    if (!cache) return false;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size())
        return false;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return false;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size)
        return false;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return false;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 ||
        (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size)
        return false;
    const uint8_t* props = cache->base + propsOff;

    // ---- Reals (Float Constants @ 0x1C) ----
    TagBlockRef realsBlk = ReadTagBlock(props + OFF_REAL_CONSTS_IN_PROPS);
    if (realsBlk.count > 0 && realsBlk.count <= 0x1000) {
        int64_t realsOff = TagMetaFileOff(cache, realsBlk.pointer);
        if (realsOff >= 0 &&
            (size_t)realsOff + (size_t)realsBlk.count * REAL_CONSTS_BLOCK_SIZE
                <= cache->size)
        {
            uint32_t cap = (uint32_t)realsBlk.count;
            if (cap > SHADER_CONSTS_MAX) cap = SHADER_CONSTS_MAX;
            for (uint32_t i = 0; i < cap; ++i) {
                const uint8_t* entry = cache->base + realsOff +
                                       (size_t)i * REAL_CONSTS_BLOCK_SIZE;
                memcpy(&outConsts->RealConstants[i * 4], entry, 16);
            }
            outConsts->RealCount = cap;
        }
    }

    // ---- Scalars (Integer Constants @ 0x28, bit-cast to float) ----
    TagBlockRef scalarsBlk = ReadTagBlock(props + OFF_SCALAR_CONSTS_IN_PROPS);
    if (scalarsBlk.count > 0 && scalarsBlk.count <= 0x1000) {
        int64_t scalarsOff = TagMetaFileOff(cache, scalarsBlk.pointer);
        if (scalarsOff >= 0 &&
            (size_t)scalarsOff + (size_t)scalarsBlk.count * SCALAR_CONSTS_BLOCK_SIZE
                <= cache->size)
        {
            uint32_t cap = (uint32_t)scalarsBlk.count;
            if (cap > SHADER_CONSTS_MAX) cap = SHADER_CONSTS_MAX;
            for (uint32_t i = 0; i < cap; ++i) {
                const uint8_t* entry = cache->base + scalarsOff +
                                       (size_t)i * SCALAR_CONSTS_BLOCK_SIZE;
                // Read raw 4 bytes; reinterpret as float (engine consumes the
                // shader register as float in set 1 even when the tag schema
                // names it "Integer Constants"). Caller can re-bit-cast back to
                // int32 if they want the raw integer value.
                memcpy(&outConsts->ScalarConstants[i], entry, 4);
            }
            outConsts->ScalarCount = cap;
        }
    }

    // ---- ArgNames (rmt2.Arguments[]) ----
    int32_t rmtRawId = R32(props + 12);
    int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                       ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
    if (rmtTagId >= 0 && (uint32_t)rmtTagId < cache->tags.size() &&
        cache->stringTableParsed)
    {
        const TagEntry& rmtTe = cache->tags[rmtTagId];
        if (rmtTe.classIndex >= 0 &&
            rmtTe.classCode[0] == 'r' && rmtTe.classCode[1] == 'm' &&
            rmtTe.classCode[2] == 't')
        {
            int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
            if (rmtOff >= 0 &&
                (size_t)rmtOff + OFF_RMT_ARGUMENTS + 8 <= cache->size)
            {
                const uint8_t* rmtMeta = cache->base + rmtOff;
                TagBlockRef argsBlk = ReadTagBlock(rmtMeta + OFF_RMT_ARGUMENTS);
                if (argsBlk.count > 0 && argsBlk.count <= 0x1000) {
                    int64_t argsOff = TagMetaFileOff(cache, argsBlk.pointer);
                    if (argsOff >= 0 &&
                        (size_t)argsOff + (size_t)argsBlk.count * STRINGID_BLOCK_SIZE
                            <= cache->size)
                    {
                        uint32_t cap = (uint32_t)argsBlk.count;
                        if (cap > SHADER_CONSTS_MAX) cap = SHADER_CONSTS_MAX;
                        for (uint32_t i = 0; i < cap; ++i) {
                            int32_t sid = R32(cache->base + argsOff +
                                              (size_t)i * STRINGID_BLOCK_SIZE);
                            const char* name = ResolveStringId(cache, sid);
                            if (!name) continue;
                            char* dst = &outConsts->ArgNames[i * SHADER_CONSTS_NAME_LEN];
                            size_t n = strnlen(name, SHADER_CONSTS_NAME_LEN - 1);
                            memcpy(dst, name, n);
                            // Already zero-init; trailing bytes stay null.
                        }
                    }
                }
            }
        }
    }

    // Bounded diag - first N materials per session emit a one-liner with arg
    // names + values. Helps the user verify the offsets are correct without
    // walking the binary by hand.
    if (outConsts->RealCount > 0 || outConsts->ScalarCount > 0) {
        static std::atomic<int> s_diagBudget{ 64 };
        int v = s_diagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_diagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                __try {
                    char nameBuf[256] = {0}; size_t nu = 0;
                    char realBuf[384] = {0}; size_t ru = 0;
                    char scalarBuf[256] = {0}; size_t su = 0;
                    uint32_t nNames = (outConsts->RealCount > outConsts->ScalarCount)
                                      ? outConsts->RealCount : outConsts->ScalarCount;
                    if (nNames > 8) nNames = 8;
                    for (uint32_t i = 0; i < nNames; ++i) {
                        const char* nm = &outConsts->ArgNames[i * SHADER_CONSTS_NAME_LEN];
                        if (!*nm) nm = "<?>";
                        int n = _snprintf_s(nameBuf + nu, sizeof(nameBuf) - nu,
                            _TRUNCATE, "%s%s", nu == 0 ? "" : ",", nm);
                        if (n > 0) nu += (size_t)n;
                        if (nu > sizeof(nameBuf) - 32) break;
                    }
                    uint32_t nReals = outConsts->RealCount > 4 ? 4 : outConsts->RealCount;
                    for (uint32_t i = 0; i < nReals; ++i) {
                        float x = outConsts->RealConstants[i * 4 + 0];
                        float y = outConsts->RealConstants[i * 4 + 1];
                        float z = outConsts->RealConstants[i * 4 + 2];
                        float w = outConsts->RealConstants[i * 4 + 3];
                        int n = _snprintf_s(realBuf + ru, sizeof(realBuf) - ru,
                            _TRUNCATE, "%s(%.3f,%.3f,%.3f,%.3f)",
                            ru == 0 ? "" : ",", x, y, z, w);
                        if (n > 0) ru += (size_t)n;
                        if (ru > sizeof(realBuf) - 64) break;
                    }
                    uint32_t nSc = outConsts->ScalarCount > 8 ? 8 : outConsts->ScalarCount;
                    for (uint32_t i = 0; i < nSc; ++i) {
                        float f = outConsts->ScalarConstants[i];
                        int32_t asInt; memcpy(&asInt, &f, 4);
                        int n = _snprintf_s(scalarBuf + su, sizeof(scalarBuf) - su,
                            _TRUNCATE, "%s%g(0x%x)",
                            su == 0 ? "" : ",", f, (unsigned)asInt);
                        if (n > 0) su += (size_t)n;
                        if (su > sizeof(scalarBuf) - 32) break;
                    }
                    zh_mcc::NativeDiag(
                        "ShaderConsts: shader=0x%04x matIdx=%d realCount=%u scalarCount=%u args=[%s] reals=[%s] scalars=[%s]",
                        (unsigned)(shaderTagId & 0xFFFF), materialIndexForDiag,
                        outConsts->RealCount, outConsts->ScalarCount,
                        nameBuf, realBuf, scalarBuf);
                } __except (EXCEPTION_EXECUTE_HANDLER) { }
                break;
            }
        }
    }

    return outConsts->RealCount > 0 || outConsts->ScalarCount > 0;
}

bool SehResolveShaderConstants(CacheHandle* cache, int32_t shaderTagId,
                               int32_t materialIndexForDiag,
                               ZH_ShaderConstantsImpl* outConsts)
{
    __try { return ResolveShaderConstantsInner(cache, shaderTagId,
                                                materialIndexForDiag, outConsts); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outConsts) memset(outConsts, 0, sizeof(*outConsts));
        return false;
    }
}

// Cross-TU bridge (consumed by MapModelParser.cpp). MapModelParser's
// ResolveShaderConstantsForMmp() forwards to this so both code paths share a
// single offset table + one diag budget.
extern "C" bool MapBspParser_ResolveShaderConstants_ForMmp(
    void* cacheHandlePtr, int32_t shaderTagId, int32_t shaderIndexForDiag,
    void* outConstsPtr)
{
    CacheHandle* cache = (CacheHandle*)cacheHandlePtr;
    ZH_ShaderConstantsImpl* out = (ZH_ShaderConstantsImpl*)outConstsPtr;
    return SehResolveShaderConstants(cache, shaderTagId, shaderIndexForDiag, out);
}

// Inner walker (raw, no __try). Caller wraps in SEH.
uint32_t EnumerateSbspsInner(CacheHandle* cache, uint32_t scnrTagId,
                             uint32_t* outIds, uint32_t maxCount)
{
    if (scnrTagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[scnrTagId];
    if (te.classIndex < 0) return 0;
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return 0;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return 0;
    ScnrLayout SL = PickScnrLayout(cache->cacheType);
    if ((size_t)metaOff + (size_t)SL.OFF_STRUCTURE_BSPS + 8 > cache->size) return 0;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef blk = ReadTagBlock(meta + SL.OFF_STRUCTURE_BSPS);
    if (blk.count <= 0 || blk.count > 0x10000) return 0;
    int64_t off = TagMetaFileOff(cache, blk.pointer);
    constexpr int STRUCTURE_BSP_BLOCK_SIZE = 172;
    if (off < 0 ||
        (size_t)off + (size_t)blk.count * STRUCTURE_BSP_BLOCK_SIZE > cache->size)
        return 0;

    uint32_t emitted = 0;
    std::unordered_set<uint32_t> seen;
    for (int i = 0; i < blk.count; ++i) {
        const uint8_t* b = cache->base + off + (size_t)i * STRUCTURE_BSP_BLOCK_SIZE;
        int32_t bspRefId = ReadTagRefId(b);
        if (bspRefId < 0) continue;
        uint32_t id = (uint32_t)bspRefId;
        if (id >= cache->tags.size()) continue;
        if (memcmp(cache->tags[id].classCode, TC_SBSP, 4) != 0) continue;
        if (!seen.insert(id).second) continue;
        if (outIds && emitted < maxCount) outIds[emitted] = id;
        emitted++;
    }
    return emitted;
}

uint32_t SehEnumerateSbsps(CacheHandle* cache, uint32_t scnrTagId,
                           uint32_t* outIds, uint32_t maxCount)
{
    __try { return EnumerateSbspsInner(cache, scnrTagId, outIds, maxCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}

// =============================================================================
// Runtime Decorator Geometry - implementation (inside anonymous namespace so
// ParseSections, ParseBoundingBoxes, ParseFixupRegion, ReadResourceData, etc.
// are all accessible). The ZH_RuntimeDecoratorMesh struct is defined outside
// the anonymous namespace (after the close brace) so extern "C" exports can
// reference it; a forward declaration here suffices for the implementation.
// =============================================================================

constexpr int kDecInstBuf_Meshes         = 0x00;
constexpr int kDecInstBuf_BoundingBoxes  = 0x0C;
// DECORATOR_GEOM_FIX (2026-06): inline global_render_geometry_struct embedded
// at sbsp+0x280 has its resource handle at +0x94, NOT +0x90. Diagnosed via
// the instBufHdr dump on forge_halo (sbsp=0x3056): bytes 0x90..0x93 are
// trailing pad-zeros of the previous field, bytes 0x94..0x97 hold the
// 0x854F23EB-style resource id. Walker was bailing with "resourcePtr=0
// (null) @+0x90" on every map because of this off-by-4. With the fix
// applied the VFMT_DECORATOR sections decode and decorator grass renders.
constexpr int kDecInstBuf_ResourcePtr    = 0x94;
constexpr int kSbspRuntimeDecoratorSetsOff   = 0x274;
constexpr int kSbspDecoratorInstanceBufOff   = 0x280;

int32_t ResolveDctrTextureBitmapTagId(CacheHandle* cache, int32_t dctrTagId)
{
    if (dctrTagId < 0 || (uint32_t)dctrTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[dctrTagId];
    if (memcmp(te.classCode, "dctr", 4) != 0) return -1;
    int64_t dctrMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (dctrMetaOff < 0) return -1;
    // dctr.texture (bitm) tag-reference @ +0x50. (SAPIEN_DECORATOR_RE.md said
    // +0x60; the real offset in MccHaloReach cooked dctr meta is +0x50 - pinned
    // from forge_halo dctr 0x2DE2/0x2DE5/... where classDword 'bitm' sits at +0x50.
    // The +0x10/+0x20 slots are the LOD render_models (mode); +0x50 is texture.)
    constexpr int kDctrTextureRefOff = 0x50;
    if ((size_t)dctrMetaOff + kDctrTextureRefOff + 16 > cache->size) return -1;
    const uint8_t* texRef = cache->base + dctrMetaOff + kDctrTextureRefOff;
    uint32_t rawId = RU32(texRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    int32_t bitmId = (int32_t)(rawId & 0xFFFFu);
    if ((uint32_t)bitmId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[bitmId].classCode, "bitm", 4) != 0) return -1;
    return bitmId;
}

// =============================================================================
// DECORATOR BLADE TEMPLATE (render_model) loader.
//
// RE'd from forge_halo.map bytes (deco_probe harness, see DECORATOR_RE notes):
// the dctr (decorator_set) meta carries up to 4 render_model (mode) tag-refs,
// one per LOD, at fixed 16-byte tag-reference slots:
//     +0x00  LOD1 render_model (highest detail)
//     +0x10  LOD2 render_model
//     +0x20  LOD3 render_model
//     +0x30  LOD4 render_model
//     +0x50  texture bitm (already used by ResolveDctrTextureBitmapTagId)
// Null slots read 0xFFFFFFFF at +0xC. Each render_model holds ONE VFMT_DECORATOR
// (fmt 0x0F, stride 0x20) section whose vertices are the absolute-sized blade
// template: Float32_3 position decompressed via the section's posMin/posMax
// (== vertex_compression_offset/scale, decorators.hlsl_include:242), Float32_2
// texcoord @ +0x0C, Float32_3 normal @ +0x14. Verified vert/bbox dumps:
//   flowers lod2     mode 0x2DE7  24 verts  bbox +-0.115 x  z 0..0.020
//   ground_cover lod2 mode 0x2DEB 56 verts  bbox ~+-0.45 x z -0.02..0.27
//   bushes lod3      mode 0x2DF0  70 verts  bbox ~+-0.40 x z -0.06..0.57
//   rocks lod2       mode 0x2DF4  48 verts  bbox ~+-0.30 x z -0.02..0.03
// These are real, world-unit blade/tuft/rock meshes - NOT flat cards.
//
// DECO_LOD_FIX: we pick the HIGHEST-detail available LOD (LOD1,
// the +0x00 slot) per set by default. RATIONALE - MMS is a STATIC, whole-map
// viewer with NO runtime distance-LOD: the camera sits at close range and the
// engine renders near-camera decorators at LOD1 (the +0x00 highest-detail
// render_model). Picking the LOWEST-detail LOD ("fewest verts") as a
// vertex-budget hedge under-details the near foliage (chunky/blocky tufts);
// the 24M cap (DECO_CAP_FIX, 4x headroom) exists specifically so dense maps
// do not decimate. LOD1 is the engine-faithful
// near-view silhouette. The global vertex cap's even-stride subsample
// (instStride, ~line 5829) STILL self-protects: if LOD1's higher per-template
// vert count pushes a pathological map over 24M, instances are evenly strided
// (coverage degrades gracefully) rather than crashing - so raising detail is
// safe. Env MMS_DECO_LOD overrides (1=highest..4=lowest) for detail-vs-budget
// tuning without a rebuild; out-of-range/unset falls back to LOD1. The engine's
// per-vertex scale/orientation is then applied per instance by the existing
// non-unit-quaternion transform (the per-instance scale of
// decorators.hlsl_include:256 rides the SNORM16 quat magnitude, DECO_SCALE_FIX).
//
// We do NOT decompress positions ourselves: ZH_MMP_DecodeSectionGeometry already
// applies posMin/posMax (DecodeRigidPositions), so the floats it returns are the
// absolute template. Likewise ZH_MMP_DecodeSectionUVs / *Normals return the
// authored uv/normal. Indices come back as a triangle list (strips expanded).
// =============================================================================
struct DecoTemplate {
    // DECO_SUBPART: per-section templates. Engine s_decorator_runtime_placement byte 0x7 = subpart_index
    // selects ONE section of the set's render_model per placement; the merged arrays below are kept for the
    // set-level height/alpha classification only.
    std::vector<DecoTemplate> subs;
    uint32_t typeCount = 0;      // DECO_TYPES: dctr+0x4C valid instance count (subpart_index range)
    std::vector<float>    pos;   // float3 * N
    std::vector<float>    uv;    // float2 * N
    std::vector<float>    nrm;   // float3 * N
    std::vector<uint16_t> idx;   // triangle list
    bool valid = false;
    // DECO_TYPE_RE: authored blade height = template local-Z extent.
    // The byte-verified per-class template z-extents (see GPU-INSTANCED DECORATOR
    // DECODE notes) cleanly separate tall woody decorators (tree z up to ~10,
    // bush ~0.57h) from flat ground scatter (flowers +-0.115, ground_cover ~0.45,
    // rocks ~0.30). Used by the viewer to pick a PER-TYPE alpha-test cutoff: branches
    // need a HIGHER cutoff (~0.4) so their anti-aliased fringe clips to transparent
    // (the user's "green outlines"), while thin grass blades need the LOW 0.12 ref
    // to survive. Computed from the LOCAL template verts (before instancing), so
    // it is the true blade height, not a world-space (terrain-slope) spread.
    float heightZ = 0.0f;
    float zmin = 0.0f;   // #227 diag: template local-Z base (>0 => geometry sits ABOVE the origin -> floats when seated)
    float zmax = 0.0f;
};

// Resolve a dctr -> render_model LOD tag id, or -1.
// DECO_LOD_FIX: default = HIGHEST-detail LOD present (LOD1, +0x00)
// - the engine's near-camera render_model, faithful for MMS's close-range static
// view. Env MMS_DECO_LOD picks a preferred slot (1=highest .. 4=lowest); when
// that slot is null we fall back to the NEAREST present slot toward higher detail
// (then toward lower), never returning -1 when ANY LOD exists.
int32_t ResolveDctrTemplateModeTagId(CacheHandle* cache, int32_t dctrTagId, int prefOverride = -1)
{
    if (dctrTagId < 0 || (uint32_t)dctrTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[dctrTagId];
    if (memcmp(te.classCode, "dctr", 4) != 0) return -1;
    int64_t dctrMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (dctrMetaOff < 0) return -1;
    const uint8_t* m = cache->base + dctrMetaOff;
    // LOD render_model refs at +0x00 / +0x10 / +0x20 / +0x30 (LOD1..LOD4).
    static const int kLodOffs[4] = { 0x00, 0x10, 0x20, 0x30 };

    // Resolve each LOD slot to a valid mode tag id (or -1 if null/missing).
    int32_t lodId[4] = { -1, -1, -1, -1 };
    for (int li = 0; li < 4; ++li) {
        int off = kLodOffs[li];
        if ((size_t)dctrMetaOff + off + 16 > cache->size) break;
        uint32_t rawId = RU32(m + off + 12);
        if (rawId == 0xFFFFFFFFu) continue;
        int32_t id = (int32_t)(rawId & 0xFFFFu);
        if ((uint32_t)id >= cache->tags.size()) continue;
        if (memcmp(cache->tags[id].classCode, "mode", 4) != 0) continue;
        lodId[li] = id;
    }

    // Preferred LOD index (0=LOD1 highest .. 3=LOD4 lowest).
    // Trade-off: a whole-map viewer shows decorators at ALL distances, and LOD1
    // turns low-profile ground cover into full 3D plant clumps (ground_cover
    // 624v vs LOD2 56v; rocks 1344v vs 48v) - the engine shows the simpler LODs
    // at distance. Settings>Decorators "Decorator detail" (MMS_DECO_LOD) selects
    // the LOD.
    // DECO_LOD_FIX2: default to HIGHEST detail (slot 0). The lowest LOD
    // (pref=3) renders near plants as flat imposters - panopticon's broadleaf
    // low-LOD is 0x1B1A (28v, a flat billboard) vs the real bush 0x1B19 (904v). A
    // whole-map viewer sees these up close, so the imposter reads as "flat / wrong
    // rotation / wrong size" (confirmed via Sapien A/B). Highest detail matches
    // the engine's near-LOD. (The old "distant ground cover renders a whole plant"
    // concern is a DISTANCE-LOD problem - the real fix is per-group distance LOD via
    // the DECO_BUDGET_RE centroids, not a global lowest-detail default.)
    int pref = 0;
    char buf[8] = {0};
    if (GetEnvironmentVariableA("MMS_DECO_LOD", buf, sizeof(buf)) > 0) {
        int v = atoi(buf);                      // user-facing 1..4
        if (v >= 1 && v <= 4) pref = v - 1;     // -> slot index 0..3
    }
    if (lodId[pref] >= 0) return lodId[pref];
    // Preferred slot absent: from the requested slot, search toward LOWER detail
    // (higher slot index) first, then toward higher detail - so the default lands on
    // the lowest-detail LOD actually present.
    for (int d = pref + 1; d < 4; ++d) if (lodId[d] >= 0) return lodId[d];
    for (int d = pref - 1; d >= 0; --d) if (lodId[d] >= 0) return lodId[d];
    return -1;
}

// Build a DecoTemplate from a dctr's render_model. Returns valid=false if the
// model can't be opened/decoded (caller falls back to the fixed card + logs).
DecoTemplate LoadDecoratorTemplate(CacheHandle* cache, uint64_t cacheHandleId,
                                   int32_t dctrTagId, uint32_t sbspTagId, uint32_t setIdx)
{
    DecoTemplate T;
    int32_t modeId = ResolveDctrTemplateModeTagId(cache, dctrTagId);
    if (modeId < 0) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X set=%u: dctr 0x%X has no render_model LOD ref - fixed-card fallback",
            sbspTagId, setIdx, (uint32_t)dctrTagId);
        return T;
    }
    // ZH_MMP_OpenModel takes the PUBLIC cache-handle id (not the CacheHandle*).
    ZH_ModelHandle h = ZH_MMP_OpenModel(cacheHandleId, (uint32_t)modeId);
    if (!h) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X set=%u: OpenModel(0x%X) failed - fixed-card fallback",
            sbspTagId, setIdx, (uint32_t)modeId);
        return T;
    }
    uint32_t dctrValid = 0, dctrNames = 0;
    {   // DECO_TYPES: dctr +0x40 = "render model instance names" block (count), +0x4C = VALID count. The decorator
        // vertex stream is `valid` EQUAL blocks (one per type; decorators.hlsl_include `vertex_index += type_index *
        // instance_data.x`) and the runtime placement's byte 0x7 selects the block. Byte-verified on Panopticon:
        // 1430 = 5x286, 7644 = 6x1274, 4200 = 15x280, 904 = 4x226 ...
        int64_t dOff = TagMetaFileOff(cache, cache->tags[(uint32_t)dctrTagId & 0xFFFFu].metaPointerRaw);
        if (dOff >= 0 && (size_t)dOff + 0x50 <= cache->size) {
            const uint8_t* d = cache->base + dOff;
            dctrNames = RU32(d + 0x40); dctrValid = RU32(d + 0x4C);
            if (dctrValid > 64) dctrValid = 0;
            { char hx[3*0x90+8]; int n = 0; for (int k = 0; k < 0x90 && n < (int)sizeof(hx) - 4; ++k) n += snprintf(hx + n, sizeof(hx) - n, "%02X%s", d[k], (k % 16 == 15) ? " | " : " ");
              NativeDiag("[DECO_DCTR_HEX] set=%u dctr=0x%X %s", setIdx, (uint32_t)dctrTagId, hx); }
            for (int bo = 0x20; bo <= 0x30; bo += 4) {
                TagBlockRef nb = ReadTagBlock(d + bo);
                if (nb.count <= 0 || nb.count > 32) continue;
                int64_t nOff = TagMetaFileOff(cache, nb.pointer);
                if (nOff < 0 || (size_t)nOff + (size_t)nb.count * 8 > cache->size) continue;
                for (int k = 0; k < nb.count; ++k) {
                    const uint8_t* e = cache->base + nOff + (size_t)k * 8;
                    NativeDiag("[DECO_NAMES] set=%u dctr=0x%X blockOff=0x%X entry[%d] sid=0x%08X valid=%d", setIdx, (uint32_t)dctrTagId, bo, k, RU32(e), (int)RU32(e + 4));
                }
            }
        }
    }
    uint32_t sc = ZH_MMP_GetSectionCount(h);
    NativeDiag("[DECO_TEMPLATE] sbsp=0x%X set=%u dctr=0x%X mode=0x%X sections=%u "
               "(>1 section => ALL baked per instance = leaf+plant stacked bug)",
        sbspTagId, setIdx, (uint32_t)dctrTagId, (uint32_t)modeId, sc);
    for (uint32_t si = 0; si < sc; ++si) {
        ZH_ModelSection sec;
        if (!ZH_MMP_GetSection(h, si, &sec)) continue;
        if (sec.VertexCount == 0 || sec.IndexCount == 0) continue;
        NativeDiag("[DECO_TEMPLATE]   set=%u sec[%u] verts=%u idx=%u",
            setIdx, si, sec.VertexCount, sec.IndexCount);
        uint8_t* vb = nullptr; uint32_t vbLen = 0;
        uint8_t* ib = nullptr; uint32_t ibLen = 0;
        if (!ZH_MMP_DecodeSectionGeometry(h, si, &vb, &vbLen, &ib, &ibLen)) continue;
        uint32_t vc = vbLen / 12;
        float* uv = nullptr; uint32_t uvCount = 0;
        float* nm = nullptr; uint32_t nmCount = 0;
        ZH_MMP_DecodeSectionUVs(h, si, &uv, &uvCount);
        ZH_MMP_DecodeSectionNormals(h, si, &nm, &nmCount);

        uint32_t base = (uint32_t)(T.pos.size() / 3);
        const size_t idxStart = T.idx.size();
        const float* fp = (const float*)vb;
        for (uint32_t v = 0; v < vc; ++v) {
            T.pos.push_back(fp[v*3+0]); T.pos.push_back(fp[v*3+1]); T.pos.push_back(fp[v*3+2]);
            if (uv && (v*2+1) < uvCount) { T.uv.push_back(uv[v*2+0]); T.uv.push_back(uv[v*2+1]); }
            else { T.uv.push_back(0.0f); T.uv.push_back(0.0f); }
            float nx=0,ny=0,nz=1;
            if (nm && (v*3+2) < nmCount) { nx=nm[v*3+0]; ny=nm[v*3+1]; nz=nm[v*3+2]; }
            float nl = nx*nx+ny*ny+nz*nz;
            if (nl > 1e-12f) { float inv=1.0f/sqrtf(nl); nx*=inv; ny*=inv; nz*=inv; }
            else { nx=0; ny=0; nz=1; }
            T.nrm.push_back(nx); T.nrm.push_back(ny); T.nrm.push_back(nz);
        }
        // Indices: model parser returns 2-byte indices for sections < 0xFFFF verts.
        uint32_t idxStride = (vc > 0xFFFF) ? 4u : 2u;
        uint32_t idxCount = ibLen / idxStride;
        std::vector<uint32_t> rawIdx(idxCount);
        for (uint32_t k = 0; k < idxCount; ++k) {
            uint32_t e;
            if (idxStride == 2) { uint16_t t; memcpy(&t, ib + (size_t)k*2, 2); e = t; }
            else                { memcpy(&e, ib + (size_t)k*4, 4); }
            rawIdx[k] = e;
        }
        // DECORATOR STRIP FIX: decorator render_model sections are unindexed
        // (flags&0x10) with IndexFormat = triangle STRIP (ifmt 0x05). The model
        // parser's unindexed path synthesizes a flat [0..N) sequence WITHOUT strip
        // expansion, so the raw indices arrive as 0,1,2,...,N-1 and would render as
        // a sparse flat list (every blade missing 2/3 of its triangles). Detect the
        // synthesized-sequential case and expand it as a triangle strip (alternating
        // winding, skipping degenerate tris) so the real blade silhouette appears.
        bool sequential = (idxCount == vc) && (idxCount >= 3);
        for (uint32_t k = 0; sequential && k < idxCount; ++k)
            if (rawIdx[k] != k) { sequential = false; }
        if (sequential) {
            for (uint32_t k = 0; k + 2 < idxCount; ++k) {
                uint32_t a = k, b = k+1, c = k+2;
                if (a==b||b==c||a==c) continue;
                if ((k & 1) == 0) {
                    T.idx.push_back((uint16_t)(base+a));
                    T.idx.push_back((uint16_t)(base+b));
                    T.idx.push_back((uint16_t)(base+c));
                } else {
                    T.idx.push_back((uint16_t)(base+a));
                    T.idx.push_back((uint16_t)(base+c));
                    T.idx.push_back((uint16_t)(base+b));
                }
            }
        } else {
            for (uint32_t k = 0; k < idxCount; ++k)
                T.idx.push_back((uint16_t)(base + rawIdx[k]));
        }
        {   // DECO_SUBPART: engine placement byte 0x7 (subpart_index) selects a render_model PART (submesh
            // index range) of the set model. Build one compacted sub-template per part of this section; if the
            // model exposes no parts, the whole section is the single subpart.
            const std::vector<uint16_t> secIdx(T.idx.begin() + idxStart, T.idx.end());   // section-relative? no: base-offset
            auto makeSub = [&](uint32_t i0, uint32_t i1) {
                DecoTemplate S;
                std::vector<int32_t> remap((size_t)vc, -1);
                for (uint32_t k = i0; k < i1 && k < (uint32_t)secIdx.size(); ++k) {
                    uint32_t e = (uint32_t)secIdx[k] - base;
                    if (e >= vc) continue;
                    if (remap[e] < 0) {
                        remap[e] = (int32_t)(S.pos.size() / 3);
                        S.pos.push_back(T.pos[(size_t)(base + e) * 3 + 0]); S.pos.push_back(T.pos[(size_t)(base + e) * 3 + 1]); S.pos.push_back(T.pos[(size_t)(base + e) * 3 + 2]);
                        S.uv.push_back(T.uv[(size_t)(base + e) * 2 + 0]);   S.uv.push_back(T.uv[(size_t)(base + e) * 2 + 1]);
                        S.nrm.push_back(T.nrm[(size_t)(base + e) * 3 + 0]); S.nrm.push_back(T.nrm[(size_t)(base + e) * 3 + 1]); S.nrm.push_back(T.nrm[(size_t)(base + e) * 3 + 2]);
                    }
                    S.idx.push_back((uint16_t)remap[e]);
                }
                if (!S.pos.empty() && !S.idx.empty()) {
                    S.valid = true;
                    float szmin = 1e30f, szmax = -1e30f;
                    for (size_t v = 0; v + 2 < S.pos.size(); v += 3) { float z = S.pos[v + 2]; if (z < szmin) szmin = z; if (z > szmax) szmax = z; }
                    S.zmin = szmin; S.zmax = szmax; S.heightZ = szmax - szmin;
                    T.subs.push_back(std::move(S));
                }
            };
            // Strip-ordered sections (identity index stream, `sequential`): a part's IndexStart/IndexLength is a
            // VERTEX range of that stream; rebuild its triangles with the same even/odd strip winding.
            auto makeSubStrip = [&](uint32_t v0, uint32_t v1) {
                DecoTemplate S;
                if (v1 > vc) v1 = vc;
                if (v0 + 2 >= v1) return;
                for (uint32_t v = v0; v < v1; ++v) {
                    S.pos.push_back(T.pos[(size_t)(base + v) * 3 + 0]); S.pos.push_back(T.pos[(size_t)(base + v) * 3 + 1]); S.pos.push_back(T.pos[(size_t)(base + v) * 3 + 2]);
                    S.uv.push_back(T.uv[(size_t)(base + v) * 2 + 0]);   S.uv.push_back(T.uv[(size_t)(base + v) * 2 + 1]);
                    S.nrm.push_back(T.nrm[(size_t)(base + v) * 3 + 0]); S.nrm.push_back(T.nrm[(size_t)(base + v) * 3 + 1]); S.nrm.push_back(T.nrm[(size_t)(base + v) * 3 + 2]);
                }
                const uint32_t n = v1 - v0;
                for (uint32_t k = 0; k + 2 < n; ++k) {
                    uint32_t a = k, b = k + 1, c = k + 2;
                    if ((k & 1) == 0) { S.idx.push_back((uint16_t)a); S.idx.push_back((uint16_t)b); S.idx.push_back((uint16_t)c); }
                    else              { S.idx.push_back((uint16_t)a); S.idx.push_back((uint16_t)c); S.idx.push_back((uint16_t)b); }
                }
                if (!S.pos.empty() && !S.idx.empty()) {
                    S.valid = true;
                    float szmin = 1e30f, szmax = -1e30f;
                    for (size_t v = 0; v + 2 < S.pos.size(); v += 3) { float z = S.pos[v + 2]; if (z < szmin) szmin = z; if (z > szmax) szmax = z; }
                    S.zmin = szmin; S.zmax = szmax; S.heightZ = szmax - szmin;
                    T.subs.push_back(std::move(S));
                }
            };
            uint32_t partsHere = 0;
            const uint32_t smc = ZH_MMP_GetSubmeshCount(h);
            if (sc == 1 && sequential && dctrValid >= 1 && vc % dctrValid == 0) {
                const uint32_t V = vc / dctrValid;
                for (uint32_t k = 0; k < dctrValid; ++k) makeSubStrip(k * V, (k + 1) * V);
                partsHere = dctrValid; T.typeCount = dctrValid;
                NativeDiag("[DECO_TYPES] set=%u dctr=0x%X names=%u valid=%u -> %u type blocks of %u verts", setIdx, (uint32_t)dctrTagId, dctrNames, dctrValid, dctrValid, V);
            } else
            for (uint32_t pi = 0; pi < smc; ++pi) {
                ZH_ModelSubmesh sm;
                if (!ZH_MMP_GetSubmesh(h, pi, &sm)) continue;
                if (sm.SectionIndex != si || sm.IndexLength == 0) continue;
                NativeDiag("[DECO_TEMPLATE]   set=%u sec[%u] part[%u] idx[%u..+%u) shader=%d", setIdx, si, pi, sm.IndexStart, sm.IndexLength, sm.ShaderIndex);
                if (sequential) makeSubStrip(sm.IndexStart, sm.IndexStart + sm.IndexLength);
                else            makeSub(sm.IndexStart, sm.IndexStart + sm.IndexLength);
                ++partsHere;
            }
            if (partsHere == 0) { if (sequential) makeSubStrip(0, vc); else makeSub(0, (uint32_t)secIdx.size()); }
            NativeDiag("[DECO_TEMPLATE]   set=%u sec[%u] parts=%u (submeshes=%u strip=%d) -> subs=%u",
                setIdx, si, partsHere, smc, sequential ? 1 : 0, (unsigned)T.subs.size());
        }
        if (uv) ZH_MMP_FreeBuffer((uint8_t*)uv);
        if (nm) ZH_MMP_FreeBuffer((uint8_t*)nm);
        if (vb) ZH_MMP_FreeBuffer(vb);
        if (ib) ZH_MMP_FreeBuffer(ib);
    }
    ZH_MMP_CloseModel(h);
    if (!T.pos.empty() && !T.idx.empty()) {
        T.valid = true;
        // DECO_TYPE_RE: local-Z extent of the authored blade (max-min over the
        // template verts). Tree/bush templates stand tall (>~0.5), grass/flower/
        // ground-cover/rock are short (<~0.5) - feeds the per-type alpha cutoff.
        float zmin = 1e30f, zmax = -1e30f;
        for (size_t v = 0; v + 2 < T.pos.size(); v += 3) {
            float z = T.pos[v + 2];
            if (z < zmin) zmin = z;
            if (z > zmax) zmax = z;
        }
        T.heightZ = (zmax > zmin) ? (zmax - zmin) : 0.0f;
        T.zmin = zmin; T.zmax = zmax;   // #227 diag
        // #227 FLOATING FIX (RE_decorator_floating): the former #166 "base-seat" shifted every
        // centered template (zmin<0) up by -zmin so its lowest vertex sat at local 0. But the bake
        // then multiplies the shifted vertex by instScale=|q|^2 -> a SPURIOUS lift of |q|^2*|zmin|
        // (~=1.5x, worse after the linear -> |q|^2 change) = the reported foliage/debris FLOATING above
        // the surface. The engine does NOT re-seat: world = |q|^2*R(v_original) + instPos verbatim
        // (decorators.hlsl_include:195/218 + quaternions.hlsl:66-70 raw non-unit sandwich). So the
        // template is used AS-AUTHORED - centered debris sits with its centre at the surface
        // (slightly embedded), which is what retail renders. zmin/zmax kept only for heightZ + the
        // per-type alpha cutoff. (If seating is ever wanted it must be a fixed UNSCALED post-transform
        // offset, never a pre-scale template shift.)
        NativeDiag("RuntimeDecGeom sbsp=0x%X set=%u: template mode=0x%X verts=%llu tris=%llu heightZ=%.3f (real render_model blade)",
            sbspTagId, setIdx, (uint32_t)modeId,
            (unsigned long long)(T.pos.size()/3), (unsigned long long)(T.idx.size()/3), T.heightZ);
    } else {
        NativeDiag("RuntimeDecGeom sbsp=0x%X set=%u: render_model 0x%X decoded empty - fixed-card fallback",
            sbspTagId, setIdx, (uint32_t)modeId);
    }
    return T;
}

// =============================================================================
// DECO_COLOR_RE - per-instance baked decorator color via the BSP
// SH airprobe lighting-point grid (LightmapParser.cpp ZH_LBSP_GetAirprobeGrid).
//
// The engine's `structure_bsp_light_decorators_from_scenario` samples the BSP
// SH/airprobe lighting grid at each decorator placement position to produce the
// RGBE `instance_color`. We mirror that by loading the airprobe grid for this
// sbsp and, for each blade's world position, sampling the NEAREST airprobe's
// pre-evaluated up-facing dual-VMF ambient. The reconstructed LINEAR HDR color
// is baked into the runtime decorator geometry's per-vertex Colors channel.
//
// GATED OFF by default (env MMS_DECO_COLOR=1). When OFF the Colors arrays are
// NULL and the viewer foliage path is byte-identical to the shipping normal-only
// shading - this reconstruction is not runtime-validated, so it must never be
// default-on.
// =============================================================================

// Mirror of LightmapParser.cpp ZH_AirprobePoint (now 44B - OBJECT_PROBE_SH T1-5).
// The decorator path only reads pos/ambient, but the STRIDE must match the real
// 44B record or grid[i] indexing reads misaligned records. The appended
// domDir/mask/dirWeight feed the per-object directional path (LightProbeSampler),
// not decorators. The exported parser is reachable in-module via this forward decl.
struct DecoAirprobePoint { float pos[3]; float ambient[3]; float domDir[3]; float mask; float dirWeight; float domRgb[3]; float fillRgb[3]; float bandwidth; float fillDir[3]; };
static_assert(sizeof(DecoAirprobePoint) == 84, "DecoAirprobePoint must mirror ZH_AirprobePoint");
extern "C" __declspec(dllexport) int __stdcall ZH_LBSP_GetAirprobeGrid(
    uint64_t cacheHandle, uint32_t sbspTagId,
    DecoAirprobePoint** outBuf, uint32_t* outCount);
extern "C" __declspec(dllexport) void __stdcall ZH_LBSP_FreeAirprobeGrid(
    DecoAirprobePoint* buf);

// Before/after [DECO_COLOR] per-blade sample logging budget (so a launch can
// validate the reconstruction doesn't mislight). Bounded so dense maps don't
// flood the native log.
static int g_decoColorLogBudget = 24;

// DECO_WIND_RE FIX: bounded budget for dumping the 4 raw aux bytes
// (B0=rec+0x04 .. B3=rec+0x07) per decorator set so the next Ctrl+R run confirms
// which byte carries the real per-type motion_scale (B2 per HREK `.wzyx` swizzle).
// Lets us validate the byte choice against actual tree vs grass vs flower records
// without guessing.
static int g_decoAuxLogBudget = 48;

static bool DecoColorGateEnabled() {
    // Re-read per decode (NOT cached) so a Settings toggle (which mirrors into
    // MMS_DECO_COLOR) is honored on the next map reload without an app restart.
    char buf[8] = {0};
    DWORD n = GetEnvironmentVariableA("MMS_DECO_COLOR", buf, sizeof(buf));
    return (n > 0 && buf[0] == '1');
}

// =============================================================================
// DECO_QUAT_SWEEP - EXHAUSTIVE per-instance quaternion-orientation
// decode RE harness (offline, env MMS_DECO_QUAT_PROBE=1, no MCC launch).
//
// The prior harness only tried 4 component orders and settled on raw xyzw with a
// MEDIOCRE tall-blade up-clustering (mean up.z 0.65-0.86) - tall grass renders
// "every which way". This sweep is exhaustive: for the SAME instance buffer +
// per-set render_model templates the production path uses, it evaluates EVERY
//   24 component permutations  x  16 sign-flip masks (handedness/conjugate)
// candidate transform of (qi0,qi1,qi2,qi3) -> (x,y,z,w), broken down by SET TYPE
// (TALL woody/grass vs FLAT ground-scatter, by template local long-axis extent),
// and measures how tightly the TALL-set template UP axis clusters near world +Z.
//
// The conjugate (negate x,y,z keep w = inverse rotation) is sign mask 0b0111,
// already inside the 16 sign masks - so the 384-candidate space is complete.
//
// It ALSO tests the template UP-AXIS assumption: it uses each TALL template's
// REAL longest local axis (max bbox extent of +X/+Y/+Z) as "up" - if blades are
// modeled along local +Y or +X, +Z would be the wrong reference and scatter.
//
// The CORRECT decode makes TALL-set up.z cluster TIGHTLY near +1 (mean>0.9,
// high frac(>0.9)) with yaw still varying. The ranked top-10 (by tall mean up.z)
// is printed; pin the order/signs from those numbers.
//
// RESOLVED (HARNESS-PROVEN - REVERTS the engine-source
// wzyx decode, which the harness FALSIFIED): production ORIENTATION = cand18
// (cand3  o  Rx180), DecoDecodePack case 18. tools/deco_quat_probe.exe on
// forge_halo.map (the DECO_QUAT_PACK flat-ground table is authoritative) measured:
//   cand18 (cand3 o Rx180): FLAT up.z = +0.996, f.9=0.99, f.7=1.00 - CORRECT
//     (blades vertical). [cand19 = cand3 o Ry180 = +0.996 too.]
//   cand1 (RAW wzyx = (qi3,qi2,qi1,qi0), qi0=real-w): FLAT up.z = -0.767 - 
//     UPSIDE DOWN.
// The intermediate "engine-faithful wzyx, qi0=real-w" reasoning (read the 8 bytes
// .wzyx with qi0 as the real scalar w, no reconstruction, no flip) was WRONG: the
// decorator render_model template is +Z-up (template z runs 0..height from the
// base), and cand1 sends template +Z to -0.767 (DOWN). The harness is GROUND
// TRUTH here, so production reverts to cand18. The Rx180-vs-Ry180 split is a
// COSMETIC 180 deg yaw only (both flat-ground +0.996, identical verticality) - NOT a
// verticality question; we keep Rx180 (cand18, the prior shipped choice). The
// 24x16 SNORM sweep + DECO_QUAT_PACK table below are the proving harness; the
// production path calls DecoDecodePack(18,...) so it IS definitionally that decode.
// =============================================================================
static bool DecoQuatProbeGateEnabled() {
    char buf[8] = {0};
    DWORD n = GetEnvironmentVariableA("MMS_DECO_QUAT_PROBE", buf, sizeof(buf));
    return (n > 0 && buf[0] == '1');
}

// Apply quaternion q=(x,y,z,w) to v (standard v + 2w(qvxv) + 2(qvx(qvxv))),
// then renormalize (we only care about the DIRECTION of the rotated up axis, so
// the non-unit |q|^2 scale is divided out). Returns the unit rotated vector.
static void QuatRotateUnit(const float q[4], const float v[3], float out[3]) {
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3];
    float tx = 2.0f*(qy*v[2]-qz*v[1]);
    float ty = 2.0f*(qz*v[0]-qx*v[2]);
    float tz = 2.0f*(qx*v[1]-qy*v[0]);
    float ox = v[0] + qw*tx + (qy*tz-qz*ty);
    float oy = v[1] + qw*ty + (qz*tx-qx*tz);
    float oz = v[2] + qw*tz + (qx*ty-qy*tx);
    float l = ox*ox+oy*oy+oz*oz;
    if (l > 1e-20f) { float inv=1.0f/sqrtf(l); ox*=inv; oy*=inv; oz*=inv; }
    else { ox=0; oy=0; oz=1; }
    out[0]=ox; out[1]=oy; out[2]=oz;
}

// The 24 permutations of (0,1,2,3) -> (x,y,z,w) lane assignments.
static const uint8_t kQuatPerms[24][4] = {
    {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
    {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
    {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
    {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0},
};

struct QuatCandStat {
    int perm; int sign;           // sign in [0..15], bit b set => negate lane (x=0,y=1,z=2,w=3)
    // TALL-set metrics (the diagnostic): template true long-axis used as up.
    double tSumZ = 0.0; uint64_t tN = 0; uint64_t tGt09 = 0; uint64_t tGt07 = 0;
    double tSumYawAbs = 0.0;       // |atan2(up.x sideways)| proxy for horizontal facing spread
    // also track +Z-axis-as-up tall metric (in case templates are modeled +Z up)
    double tzSumZ = 0.0; uint64_t tzN = 0; uint64_t tzGt09 = 0;
    // NORMALIZE-Q-FIRST variant (unit quaternion => pure rotation, no shear). A
    // non-unit q in v+2w(qvxv)+2qvx(qvxv) is NOT a similarity transform, so it can
    // bend the up axis away from vertical; normalizing q first removes that. Same
    // long-axis-up metric, but with q normalized before the rotate.
    double nSumZ = 0.0; uint64_t nN = 0; uint64_t nGt09 = 0; uint64_t nGt07 = 0;
    double nzSumZ = 0.0; uint64_t nzN = 0; uint64_t nzGt09 = 0;   // normalized, +Z axis
};

// Per-set cached template "up" axis (true longest local axis) as a unit vector,
// plus a flag for whether the set is TALL. Filled once before the instance sweep.
struct DecoSetProbeInfo { float upAxis[3]; bool tall; bool valid; };

// GLOBAL cross-BSP/cross-map accumulator (process-lifetime). The per-BSP argmax
// over 384 candidates is essentially noise (with that many candidates one always
// "wins" by chance, and the winner DISAGREES across BSPs). A genuine engine decode
// must win GLOBALLY across every TALL instance on every map. This accumulates the
// SAME candidate index across all GetDeco calls; deco_quat_probe.cpp prints the
// global table once at the end via ZH_BSP_DumpDecoQuatSweepGlobal.
static QuatCandStat g_quatGlobal[24 * 16];
static uint64_t     g_quatGlobalTallInst = 0;
static bool         g_quatGlobalInit = false;

// =============================================================================
// DECO_QUAT_PACK_PROBE - FLAT-GROUND format-candidate accumulator.
//
// The 24-perm x 16-sign sweep above only ever interprets the 8 bytes at rec+0x08
// as 4xint16 SNORM and PERMUTES the lanes. It can NOT detect a different on-disk
// PACKING (smallest-three compressed quat, 3xSNORM16 + reconstructed w, etc.). And
// it pools EVERY tall instance regardless of terrain slope, so genuine authored
// terrain-follow tilt dilutes the signal.
//
// This block fixes both: it isolates FLAT-GROUND instances (decorator_group whose
// authored position_bounds_size.z is small => all placements at ~one elevation =>
// blades MUST be vertical) and, on those instances only, evaluates a fixed set of
// CANDIDATE PACKINGS of the same 8 bytes, measuring up.z (template +Z rotated) +
// |q|~=1 after reconstruction. On flat ground the CORRECT packing gives up.z ~0.95;
// the wrong one gives ~0.74 random. Decisive - see the [DECO_QUAT_PACK] table.
//
// Candidates (index order is load-bearing for the dump labels):
//   0  RAW xyzw           : (qi0,qi1,qi2,qi3)/32767 as x,y,z,w (PRODUCTION)
//   1  RAW wzyx           : reversed lane order (HREK line 202 .wzyx swizzle)
//   2  3xSNORM16 + recon-w: x,y,z = qi0,qi1,qi2 /32767; w=sqrt(max(0,1-x^2-y^2-z^2))
//   3  3xSNORM16 + recon-w (wzyx-src): x,y,z = qi3,qi2,qi1; same w recon
//   4  smallest-three     : 2-bit max index in top bits of the 8-byte word, other
//                           three 3xSNORM(~15-bit-ish) reconstruct the largest.
//   5  half-float4        : the 8 bytes are 4xfp16 (IEEE half) x,y,z,w
//   6  RAW xyzw NORMALIZED: production but force-unit (drops the |q|^2 scale)
// For each we report flat-ground mean up.z, frac>.9, frac>.7, and mean |q| (pre-
// normalize) so a "looks like a unit quat" packing is visible.
// =============================================================================
static const int kDecoPackCandN = 21;  // +cand20 (cand1 o Rx180) for the 3-vs-4-lane compare
struct DecoPackStat {
    double sumUpZ = 0.0; uint64_t n = 0; uint64_t gt09 = 0; uint64_t gt07 = 0;
    double sumQlen = 0.0;            // mean |q| BEFORE normalization (unit-ness check)
    double sumYawSpread = 0.0;       // mean horizontal tilt magnitude of up (proxy)
    double sumAbsUpZ = 0.0; uint64_t absGt09 = 0;   // |up.z| (sign-convention-free verticality)
};
static DecoPackStat g_decoPackFlat[kDecoPackCandN];   // flat-ground only
static DecoPackStat g_decoPackAll[kDecoPackCandN];    // all tall (control)
static uint64_t     g_decoFlatInst = 0;
static uint64_t     g_decoSlopeInst = 0;
static bool         g_decoPackInit = false;
// DECO_TYPE_TILT: per-DecoType tilt accumulators, covering ALL sets
// (incl. the short ground-scatter rocks/debris the tall-only sweep skips), so the
// "are solid debris over-tilted vs blades?" question is measurable and cand18 vs
// cand20 (3-lane vs 4-lane) can be compared per type.
static DecoPackStat g_decoPackWoody[kDecoPackCandN];    // DecoType 1 (tall woody / blade)
static DecoPackStat g_decoPackScatter[kDecoPackCandN];  // DecoType 0 (ground scatter / rock / debris)
static uint64_t     g_decoWoodyInst = 0;
static uint64_t     g_decoScatterInst = 0;
static bool         g_decoTypeInit = false;

// =============================================================================
// DECO_SCALE_PROBE - per-instance UNIFORM SCALE source RE.
//
// REGRESSION: the quaternion fix normalized q to a pure unit rotation, which
// REMOVED the per-instance size variation. HREK decorators.hlsl_include proves
// the scale is REAL and lives in the quaternion MAGNITUDE, not a separate lane:
//   line 218  rotated_position = quaternion_transform_point(instance_quaternion,
//                                                            vertex_position);
//   line 220  world_position.xyz = rotated_position + instance_position;
//   (NO explicit *scale on the cooked-map / XENON path)
//   line 180  float instance_scale = dot(instance_quaternion, instance_quaternion);
// and quaternions.hlsl_include:66 quaternion_transform_point is the RAW double
// sandwich q*pt*q' with NO normalize - so for a non-unit q it scales pt by |q|^2.
// i.e. the engine's per-instance uniform scale == |q_raw|^2 (= dot(q,q)), applied
// implicitly by NOT normalizing the rotation. instance_quaternion is
// instance_input.quaternion.wzyx - all FOUR stored SNORM16 lanes, w NOT
// reconstructed. So |q_raw|^2 != 1 carries the scale.
//
// This probe accumulates, over the SAME tall-set instances, candidate scale
// sources + a histogram, so we can confirm which gives a PLAUSIBLE ~0.5..2x
// distribution (not constant, not garbage) before wiring it into the bake:
//   srcA = dot(q4raw, q4raw)            (all 4 SNORM16 lanes, the engine's literal
//                                        dot(quaternion,quaternion))  -> SCALE
//   srcB = |q4raw|  = sqrt(srcA)        (if the dot is the *linear* scale)
//   srcC = dot of the cand18 unit-recon q (CONTROL - must be ~1, i.e. no scale,
//          proving the normalize-recon path is what dropped the size variation)
//   srcD = qi0 as SNORM16 magnitude     (task alt-hypothesis: scale rides qi0 lane)
//   srcE = qi0 as UNORM16 [0,1]         (task alt-hypothesis b)
// Histogram buckets over [0,2.5] (the engine notes "max scale of 2.0", line 193).
// =============================================================================
struct DecoScaleStat {
    double   sum = 0.0, sumSq = 0.0;
    double   mn = 1e30, mx = -1e30;
    uint64_t n = 0;
    uint64_t hist[10] = {0};   // buckets of width 0.25 over [0, 2.5)
};
static DecoScaleStat g_decoScale[5];   // srcA..srcE
static bool          g_decoScaleInit = false;
static void DecoScaleAccum(int src, double v) {
    DecoScaleStat& s = g_decoScale[src];
    s.sum += v; s.sumSq += v*v; s.n++;
    if (v < s.mn) s.mn = v;
    if (v > s.mx) s.mx = v;
    int b = (int)(v / 0.25);
    if (b < 0) b = 0; if (b > 9) b = 9;
    s.hist[b]++;
}

// True-sandwich rotate (matches the SHIPPED quatTransform: pt' = (w^2-|u|^2)pt +
// 2(u*pt)u + 2w(uxpt) = |q|^2*R(pt)), then divide out |q|^2 so we read the pure
// rotated DIRECTION. q need NOT be unit. Returns unit out.
static void QuatSandwichDir(const float q[4], const float v[3], float out[3]) {
    const float ux=q[0], uy=q[1], uz=q[2], w=q[3];
    const float uu = ux*ux+uy*uy+uz*uz;
    const float a  = w*w - uu;
    const float d  = 2.0f*(ux*v[0]+uy*v[1]+uz*v[2]);
    const float cx = uy*v[2]-uz*v[1];
    const float cy = uz*v[0]-ux*v[2];
    const float cz = ux*v[1]-uy*v[0];
    float ox = a*v[0] + d*ux + 2.0f*w*cx;
    float oy = a*v[1] + d*uy + 2.0f*w*cy;
    float oz = a*v[2] + d*uz + 2.0f*w*cz;
    float l = ox*ox+oy*oy+oz*oz;
    if (l > 1e-20f) { float inv=1.0f/sqrtf(l); ox*=inv; oy*=inv; oz*=inv; }
    else { ox=0; oy=0; oz=1; }
    out[0]=ox; out[1]=oy; out[2]=oz;
}

// half (fp16) -> float.
static float DecoHalfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

static inline float t_clamp01(float x){ return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// Decode the 8 quaternion bytes at `qb` under candidate packing `cand` into a
// quaternion q[4] (x,y,z,w). Returns true if usable (non-degenerate).
static bool DecoDecodePack(int cand, const uint8_t* qb, float q[4]) {
    int16_t s[4]; memcpy(s, qb, 8);
    const float inv = 1.0f / 32767.0f;
    switch (cand) {
        case 0: // raw xyzw
            q[0]=s[0]*inv; q[1]=s[1]*inv; q[2]=s[2]*inv; q[3]=s[3]*inv; break;
        case 1: // raw wzyx (reverse)
            q[0]=s[3]*inv; q[1]=s[2]*inv; q[2]=s[1]*inv; q[3]=s[0]*inv; break;
        case 2: { // 3xSNORM16 xyz + reconstruct w
            q[0]=s[0]*inv; q[1]=s[1]*inv; q[2]=s[2]*inv;
            float t = 1.0f - (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3] = t > 0.0f ? sqrtf(t) : 0.0f; break;
        }
        case 3: { // 3xSNORM16 from wzyx source + reconstruct w
            q[0]=s[3]*inv; q[1]=s[2]*inv; q[2]=s[1]*inv;
            float t = 1.0f - (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3] = t > 0.0f ? sqrtf(t) : 0.0f; break;
        }
        case 4: { // smallest-three: 2-bit max index in top of the 64-bit word,
                  // other three as 3xSNORM in remaining bits. Use the classic
                  // 30-bit-3x10 layout in the low 4 bytes + index in bits 30..31.
            uint32_t lo; memcpy(&lo, qb, 4);
            uint32_t idx = (lo >> 30) & 0x3u;
            const float k = 1.41421356f; // range [-1/sqrt2 .. 1/sqrt2]
            float c0 = ((float)(int)((lo      ) & 0x3FFu) - 511.0f) / 511.0f / k;
            float c1 = ((float)(int)((lo >> 10) & 0x3FFu) - 511.0f) / 511.0f / k;
            float c2 = ((float)(int)((lo >> 20) & 0x3FFu) - 511.0f) / 511.0f / k;
            float sum = c0*c0 + c1*c1 + c2*c2;
            float big = sqrtf(t_clamp01(1.0f - sum));
            float qq[4];
            int j = 0;
            for (int a = 0; a < 4; ++a) {
                if ((uint32_t)a == idx) { qq[a] = big; }
                else { qq[a] = (j==0)?c0:(j==1)?c1:c2; ++j; }
            }
            q[0]=qq[0]; q[1]=qq[1]; q[2]=qq[2]; q[3]=qq[3]; break;
        }
        case 5: { // 4xfp16
            uint16_t h[4]; memcpy(h, qb, 8);
            q[0]=DecoHalfToFloat(h[0]); q[1]=DecoHalfToFloat(h[1]);
            q[2]=DecoHalfToFloat(h[2]); q[3]=DecoHalfToFloat(h[3]); break;
        }
        case 6: { // raw xyzw, force-normalized (drop |q|^2 scale)
            float x=s[0]*inv,y=s[1]*inv,z=s[2]*inv,w=s[3]*inv;
            float l=x*x+y*y+z*z+w*w;
            if (l>1e-12f){float r=1.0f/sqrtf(l);q[0]=x*r;q[1]=y*r;q[2]=z*r;q[3]=w*r;}
            else {q[0]=0;q[1]=0;q[2]=0;q[3]=1;} break;
        }
        // --- refined 3-component (xyz packed, w reconstructed) candidates ---
        // cand3 (x=s3,y=s2,z=s1,reconW) hit |up.z|=0.96 NEGATED. These pin the
        // exact lane triple + the sign (CONJUGATE = inverse rotation flips the
        // rotated-axis sign cleanly; w-sign does NOT change rotation).
        case 7: { // CONJUGATE of cand3: xyz negated, reconstructed +w
            q[0]=-s[3]*inv; q[1]=-s[2]*inv; q[2]=-s[1]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3]= t>0.0f? sqrtf(t):0.0f; break;
        }
        case 8: { // cand3 lanes, reconstructed w NEGATED (-w)
            q[0]=s[3]*inv; q[1]=s[2]*inv; q[2]=s[1]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3]= t>0.0f? -sqrtf(t):0.0f; break;
        }
        case 9: { // xyz = s0,s1,s2 reconstruct w  (forward lane triple, +w)
            q[0]=s[0]*inv; q[1]=s[1]*inv; q[2]=s[2]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3]= t>0.0f? sqrtf(t):0.0f; break;
        }
        case 10: { // CONJUGATE of cand9
            q[0]=-s[0]*inv; q[1]=-s[1]*inv; q[2]=-s[2]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3]= t>0.0f? sqrtf(t):0.0f; break;
        }
        case 11: { // xyz = s1,s2,s3 (drop s0), reconstruct w, +w
            q[0]=s[1]*inv; q[1]=s[2]*inv; q[2]=s[3]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3]= t>0.0f? sqrtf(t):0.0f; break;
        }
        case 12: { // CONJUGATE of cand11
            q[0]=-s[1]*inv; q[1]=-s[2]*inv; q[2]=-s[3]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
            q[3]= t>0.0f? sqrtf(t):0.0f; break;
        }
        // cand3 family hit |up.z|=0.96 but NEGATIVE (template +Z -> world -Z) and
        // conjugate/(-w) did NOT flip it => the sign is an intrinsic axis-handedness
        // flip. These flip exactly ONE xyz lane of cand3 (changes the rotation, not
        // just its inverse) to try to land +Z -> +Z.
        case 13: { // cand3 with x negated
            q[0]=-s[3]*inv; q[1]=s[2]*inv; q[2]=s[1]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); q[3]=t>0.0f?sqrtf(t):0.0f; break;
        }
        case 14: { // cand3 with y negated
            q[0]=s[3]*inv; q[1]=-s[2]*inv; q[2]=s[1]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); q[3]=t>0.0f?sqrtf(t):0.0f; break;
        }
        case 15: { // cand3 with z negated
            q[0]=s[3]*inv; q[1]=s[2]*inv; q[2]=-s[1]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); q[3]=t>0.0f?sqrtf(t):0.0f; break;
        }
        case 16: { // cand3 lane order x=s1,y=s2,z=s3 (forward-from-s1) +reconW
            q[0]=s[1]*inv; q[1]=s[2]*inv; q[2]=s[3]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); q[3]=t>0.0f?sqrtf(t):0.0f; break;
        }
        case 17: { // x=s1,y=s2,z=s3 with z negated
            q[0]=s[1]*inv; q[1]=s[2]*inv; q[2]=-s[3]*inv;
            float t=1.0f-(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]); q[3]=t>0.0f?sqrtf(t):0.0f; break;
        }
        // cand3 (x=s3,y=s2,z=s1,+reconW) maps template +Z -> world -Z perfectly
        // (|up.z|=0.996): it is the CORRECT rotation but the template's modeled UP is
        // -Z (Sapien decorator blades grow along local -Z; the seat note's "+Z up to
        // ~10" is the OTHER set/LOD convention). The production fix rotates template
        // -Z (so blade up = -Z), OR equivalently composes cand3 with a 180 deg-about-X.
        // cand18 = cand3  o  Rx(180 deg) : Q' = Q * (x=1,y=0,z=0,w=0). For Q=(a,b,c,d):
        //   Q*(1,0,0,0) = (d, c, -b, -a)  (Hamilton, x,y,z,w order).
        case 18: {
            float a,b,c,d; { float qq[4]; DecoDecodePack(3, qb, qq); a=qq[0];b=qq[1];c=qq[2];d=qq[3]; }
            q[0]= d; q[1]= c; q[2]= -b; q[3]= -a; break;
        }
        // cand19 = cand3  o  Ry(180 deg): Q*(0,1,0,0) = (-c, d, a, -b)
        case 19: {
            float a,b,c,d; { float qq[4]; DecoDecodePack(3, qb, qq); a=qq[0];b=qq[1];c=qq[2];d=qq[3]; }
            q[0]= -c; q[1]= d; q[2]= a; q[3]= -b; break;
        }
        // cand20 = cand1  o  Rx(180 deg): same up-axis fix as cand18 but the 4-LANE cand1
        // (qi0 = real w, NO w-reconstruction) instead of the 3-lane cand3. Review #08
        // hypothesis: the disk carries a real 4th w lane, and cand3 loses w's sign for
        // rotations >180 deg - exactly the tilted/solid-debris case. Q*(1,0,0,0) for
        // Q=(a,b,c,d) = (d, c, -b, -a) (Hamilton, x,y,z,w). 
        case 20: {
            float a,b,c,d; { float qq[4]; DecoDecodePack(1, qb, qq); a=qq[0];b=qq[1];c=qq[2];d=qq[3]; }
            q[0]= d; q[1]= c; q[2]= -b; q[3]= -a; break;
        }
        default: q[0]=0;q[1]=0;q[2]=0;q[3]=1; break;
    }
    float l = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
    return l > 1e-8f;
}


// Nearest-airprobe sampler. Linear scan (airprobe counts are small - tens to a
// few hundred per BSP, confirmed via airprobe_probe on 50_panopticon=64,
// 70_boneyard=183). Returns false (white) when the grid is empty so the caller
// can decide to skip color emission. Granularity: PER-INSTANCE (nearest point),
// matching the engine's per-placement sample - recorded in the DONE diag.
static bool SampleNearestAirprobe(const DecoAirprobePoint* grid, uint32_t gridCount,
                                  const float wpos[3], float outRgb[3], int* outIdx)
{
    if (!grid || gridCount == 0) { outRgb[0]=outRgb[1]=outRgb[2]=1.0f; if(outIdx)*outIdx=-1; return false; }
    float best = 1e30f; int bestI = 0;
    for (uint32_t k = 0; k < gridCount; ++k) {
        float dx = grid[k].pos[0]-wpos[0], dy = grid[k].pos[1]-wpos[1], dz = grid[k].pos[2]-wpos[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best) { best = d2; bestI = (int)k; }
    }
    outRgb[0] = grid[bestI].ambient[0];
    outRgb[1] = grid[bestI].ambient[1];
    outRgb[2] = grid[bestI].ambient[2];
    if (outIdx) *outIdx = bestI;
    return true;
}

bool GetRuntimeDecoratorGeometryInner(
    CacheHandle* cache, uint64_t cacheHandleId, uint32_t sbspTagId,
    ZH_RuntimeDecoratorMesh** outMeshes, uint32_t* outMeshCount)
{
    *outMeshes = nullptr;
    *outMeshCount = 0;

    if (sbspTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[sbspTagId];
    if (memcmp(te.classCode, TC_SBSP, 4) != 0) return false;

    int64_t sbspMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (sbspMetaOff < 0) return false;
    if ((size_t)sbspMetaOff + kSbspDecoratorInstanceBufOff + 168 > cache->size) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: meta too small for instBuf", sbspTagId);
        return false;
    }
    const uint8_t* sbsp = cache->base + sbspMetaOff;

    // Read decorator_sets[] for dctr tag-refs
    TagBlockRef setsBlk = ReadTagBlock(sbsp + kSbspRuntimeDecoratorSetsOff);
    std::vector<int32_t> dctrTagIds;
    std::vector<int32_t> dctrBitmapIds;
    if (setsBlk.count > 0 && setsBlk.count <= 64) {
        int64_t setsOff = TagMetaFileOff(cache, setsBlk.pointer);
        if (setsOff >= 0 &&
            (size_t)setsOff + (size_t)setsBlk.count * 16 <= cache->size)
        {
            dctrTagIds.resize(setsBlk.count);
            dctrBitmapIds.resize(setsBlk.count);
            for (int i = 0; i < setsBlk.count; ++i) {
                const uint8_t* s = cache->base + setsOff + (size_t)i * 16;
                uint32_t rawId = RU32(s + 12);
                int32_t dctrId = (rawId == 0xFFFFFFFFu) ? -1 : (int32_t)(rawId & 0xFFFFu);
                dctrTagIds[i] = dctrId;
                dctrBitmapIds[i] = ResolveDctrTextureBitmapTagId(cache, dctrId);
            }
        }
    }
    NativeDiag("RuntimeDecGeom sbsp=0x%X: decorator_sets count=%d", sbspTagId, (int)dctrTagIds.size());

    // Read decorator_instance_buffer (global_render_geometry_struct at sbsp+0x280)
    const uint8_t* instBuf = sbsp + kSbspDecoratorInstanceBufOff;
    TagBlockRef meshesBlk = ReadTagBlock(instBuf + kDecInstBuf_Meshes);
    if (meshesBlk.count <= 0 || meshesBlk.count > 256) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: meshes count=%d (none or insane)", sbspTagId, meshesBlk.count);
        return true;
    }
    TagBlockRef bboxBlk = ReadTagBlock(instBuf + kDecInstBuf_BoundingBoxes);
    // DECORATOR_GEOM_DIAG: the resource handle at instBuf+0x90 reads 0 even on
    // maps that DO have decorator sets (Forge Halo 0x3056: 5 sets), so 0x90 is
    // the wrong offset for this global_render_geometry_struct (the meshes
    // tagblock above parsed fine). Dump the struct header so we can locate the
    // real resource handle -- a dword whose (&0xFFFF) is a plausible resource
    // index. The +0x280 region was bounds-checked to 168 bytes above.
    {
        char hex[3 * 0xA8 + 16]; int p = 0;
        for (int b = 0; b < 0xA8; ++b)
            p += snprintf(hex + p, sizeof(hex) - (size_t)p, "%02X ", instBuf[b]);
        NativeDiag("RuntimeDecGeom sbsp=0x%X: instBufHdr[0x00..0xA8]= %s", sbspTagId, hex);
    }
    int32_t resourceIdRaw = R32(instBuf + kDecInstBuf_ResourcePtr);
    if (resourceIdRaw == 0 || resourceIdRaw == -1) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: resourcePtr=%d (null) @+0x%X", sbspTagId, resourceIdRaw, kDecInstBuf_ResourcePtr);
        return true;
    }

    NativeDiag("RuntimeDecGeom sbsp=0x%X: meshes=%d bboxes=%d resRaw=0x%X",
        sbspTagId, meshesBlk.count, bboxBlk.count, (uint32_t)resourceIdRaw);

    // DECORATOR_MESHES_FROM_RESOURCE: the inline
    // global_render_geometry_struct's meshes-tagblock has count>0 but
    // pointer=0 in cooked Reach maps - the meshes don't live in tag-meta,
    // they're packed into the gestalt fixup blob slice that this resource
    // owns. Load gestalt + fixups FIRST so we can resolve resource-relative
    // pointers (both meshes and bounding-boxes) before attempting the
    // section parse.
    if (!ParseGestalt(cache)) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: ParseGestalt failed", sbspTagId);
        return true;
    }
    int32_t resourceIndex = resourceIdRaw & 0xFFFF;
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size()) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: resourceIndex=%d OOB", sbspTagId, resourceIndex);
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(cache->parseMutex);
        if (!EnsureResourceFixups(cache, (size_t)resourceIndex)) {
            NativeDiag("RuntimeDecGeom sbsp=0x%X: EnsureResourceFixups failed", sbspTagId);
            return true;
        }
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];

    // Map the gestalt's FixupData blob so we can read resource-relative
    // pointers. Same constants the BSP main path uses (zone+328 fixupSize,
    // zone+340 fixupPtr).
    size_t gestaltFixupSize = 0;
    {
        int zoneIdx = FindGlobalTag(cache, "zone");
        if (zoneIdx >= 0) {
            int64_t zoneMetaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
            if (zoneMetaOff >= 0 && (size_t)zoneMetaOff + 350 <= cache->size) {
                const uint8_t* zoneMeta = cache->base + zoneMetaOff;
                int32_t fSize  = R32(zoneMeta + 328);
                uint32_t fPtrR = RU32(zoneMeta + 340);
                int64_t fOff   = TagMetaFileOff(cache, fPtrR);
                if (fSize > 0 && fOff >= 0 && (size_t)fOff + (size_t)fSize <= cache->size)
                    gestaltFixupSize = (size_t)fSize;
            }
        }
    }

    // Dump fixup table for diagnostic - helps confirm the layout on first
    // forge_halo decode and any future map with different fixup count.
    {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: entry rIdx=%d fOff=%d fSz=%d fixups=%d gestaltFixupSz=%llu",
            sbspTagId, resourceIndex, entry.fixupOffset, entry.fixupSize,
            (int)entry.fixups.size(), (unsigned long long)gestaltFixupSize);
        for (size_t fi = 0; fi < entry.fixups.size() && fi < 32; ++fi) {
            uint32_t masked = (uint32_t)(entry.fixups[fi].offset & 0x0FFFFFFFu);
            NativeDiag("RuntimeDecGeom sbsp=0x%X: fixup[%llu] unk=0x%08x off=0x%08x (masked=0x%x abs=%lld)",
                sbspTagId, (unsigned long long)fi, (uint32_t)entry.fixups[fi].unknown,
                (uint32_t)entry.fixups[fi].offset, masked,
                (long long)((int64_t)entry.fixupOffset + (int64_t)masked));
        }
    }

    // Load the resource DATA PAYLOAD (page data). The per-VB placement streams
    // are addressed by fixups whose masked offset is a payload-relative byte
    // offset (resourceData + (fixup.offset & 0x0FFFFFFF), no fixupOffset add).
    constexpr size_t kMaxRead = 64 * 1024 * 1024;
    size_t resourceSize = 0;
    uint8_t* resourceData = ReadResourceData(cache, resourceIdRaw, kMaxRead, &resourceSize);
    if (!resourceData) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: ReadResourceData failed", sbspTagId);
        return true;
    }
    NativeDiag("RuntimeDecGeom sbsp=0x%X: resourcePayload size=%llu",
        sbspTagId, (unsigned long long)resourceSize);

    // Parse the resource's VB/IB fixup region (vbCounts/vbLens/strides + per-VB
    // payload offsets in entry.fixups). Decorators have ibCount==0; the VBs are
    // the per-set stride-16 instance placement streams.
    std::vector<uint32_t> vbCounts, vbLens, vbStrides, ibLens;
    std::vector<uint8_t>  ibFmts;
    if (!ParseFixupRegion(cache, entry, vbCounts, vbLens, vbStrides, ibFmts, ibLens)) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: ParseFixupRegion failed", sbspTagId);
        free(resourceData);
        return true;
    }

    // =========================================================================
    // GPU-INSTANCED DECORATOR DECODE - static, byte-grounded.
    //
    // RE'd entirely from on-disk bytes (forge_halo.map, sbsp 0x3056/0x3B57) via
    // the offline deco_probe_harness + DECO_PROBE diags, cross-checked against
    // HREK tags/shaders/decorators/decorators.hlsl_include (DX11 path) and the
    // blam-tags scenario_structure_bsp schema. Findings:
    //
    //   * The decorator_instance_buffer resource holds N stride-16 VB streams,
    //     ONE per decorator set (N == decorator_sets count). Each stream is the
    //     per-instance placement stream s_decorator_instance_input, packed to 16B:
    //         +0x00  uint   position      UHEND3N (11/11/10 unsigned-normalized)
    //         +0x04  4xbyte auxilary_info  (B0=type_index, B1=motion_scale=0xCD,
    //                                       B2=per-inst aux.w, B3=0)
    //         +0x08  4xint16 quaternion    SNORM16 x,y,z,w  (NON-unit: |q|^2 is
    //                                       the per-instance scale, max ~2.0 - 
    //                                       decorators.hlsl_include:256)
    //     (Byte-evidence: B5 const 0xCD; quaternion w-component dominant so B15
    //      ~const 0x7B/0x7C/0x82; |q|^2 measured 1.28..2.2.)
    //
    //   * There is NO color stream in this resource. Byte-verified on forge_halo
    //     (deco_probe ZH_DBG_DumpDecoInstBuf): the instance buffer has EXACTLY one
    //     stride-16 VB stream per decorator set (sbsp 0x3056: 5 streams / 5 sets;
    //     0x3B57: 2 / 2), each = placement only (pos+aux+quat = 4+4+8). The HLSL
    //     s_decorator_instance_input.color:COLOR1 (instance_color.rgb *
    //     exp2(a*63.75-31.75), decorators.hlsl_include:303) is a 4th input slot the
    //     MCC DX11 cooker did NOT emit into this resource, and the render_model
    //     instance data carries none either. So per-instance baked RGBE color is
    //     DEFERRED (normal-only emit + log)
    // - never guessed.
    //
    //   * Per-cluster decorator_groups (cluster+0x50, 60B, decorator_runtime_
    //     cluster_block) slice each set's stream into world-space ranges:
    //         +0x00 u16 placement_count   +0x02 u8 set_index
    //         +0x03 u8  buffer_index      +0x04 i32 buffer_offset (BYTES; ==idx*16)
    //         +0x08 f32x3 position_bounds_min   (== instance_compression_offset)
    //         +0x18 f32x3 position_bounds_size  (== instance_compression_scale)
    //     world_pos = UnpackUHEND3N(rec.position) * bounds_size + bounds_min.
    //     (Verified: forge_halo group bounds land in the BSP world AABB; the
    //      buffer_offset stepping equals cumulative placement_count*16.)
    //
    //   * The blade TEMPLATE mesh (s_decorator_vertex_input) is NOT in this
    //     resource; in cooked maps it is the dctr's RENDER_MODEL. The dctr meta
    //     carries up to 4 render_model (mode) tag-refs (LOD1..4) at +0x00/+0x10/
    //     +0x20/+0x30 (16-byte tag-refs; null = 0xFFFFFFFF at +0xC), and the
    //     texture bitm at +0x50. Each render_model holds ONE VFMT_DECORATOR
    //     section (fmt 0x0F, stride 0x20): Float32_3 position decompressed via the
    //     section posMin/posMax (== vertex_compression_offset/scale,
    //     decorators.hlsl_include:242 - gives ABSOLUTE world-unit size), Float32_2
    //     texcoord @ +0x0C, Float32_3 normal @ +0x14. We load the HIGHEST-detail
    //     LOD per set (LOD1, +0x00 - DECO_LOD_FIX; env MMS_DECO_LOD overrides) via
    //     the existing model parser (ZH_MMP_*), and instance THAT mesh per
    //     placement - the real blade/tuft/rock silhouette + absolute size, NOT a
    //     fixed card. (Byte-verified template bboxes: flowers +-0.115, ground_cover
    //     ~+-0.45, bushes ~+-0.40x0.57h, rocks ~+-0.30, tree z up to ~10.) If a set's
    //     render_model can't be read we fall back to a fixed card for that set
    //     ONLY and log it (never guess).
    //
    //   * world transform (decorators.hlsl_include:271-272, DX11):
    //         rotated = quaternion_transform_point(quat, template_vertex)
    //         world   = rotated + instance_position.xyz
    //     quaternion_transform_point = q * ({pt,0} * conjugate(q))  (quaternions.
    //     hlsl_include:65-94); non-unit q bakes scale.
    //
    // One output ZH_RuntimeDecoratorMesh is produced PER decorator set (so the
    // consumer binds one dctr bitmap per mesh). Output is plain world-space
    // positions + uv + normal + indices.
    // =========================================================================

    // VBstream summary (one concise line; full byte dump only relevant during RE).
    NativeDiag("RuntimeDecGeom sbsp=0x%X: instanced-decode vbStreams=%d sets=%d",
        sbspTagId, (int)vbCounts.size(), (int)dctrTagIds.size());

    // Helpers (HREK packed_vector.fx / quaternions.fx, ported).
    auto unpackUHEND3N = [](uint32_t u, float out[3]) {
        out[0] = (float)(u & 0x7FFu) / 2047.0f;
        out[1] = (float)((u >> 11) & 0x7FFu) / 2047.0f;
        out[2] = (float)((u >> 22) & 0x3FFu) / 1023.0f;
    };
    // quaternion_transform_point(q, pt) - the TRUE sandwich q*{pt,0}*conjugate(q)
    // exactly as HREK shaders/shared/quaternions.hlsl_include:65-93 computes it
    // (q * (pt q')). For q=(u,w) the closed form is
    //     pt' = (w^2 - |u|^2)*pt + 2(u*pt)*u + 2w*(uxpt)
    // which for a NON-UNIT q equals |q|^2*R_unit(pt): a uniform |q|^2-scaled pure
    // ROTATION that PRESERVES the rotated direction. decorators.hlsl_include:256
    // takes that |q|^2=dot(q,q) only for the wind height term; line 272's comment
    // ("max scale of 2.0 is built into vertex compression") confirms the position
    // scale rides the vertex compression, and the quaternion contributes its own
    // |q|^2 uniform scale via this sandwich.
    //
    // DECO_QUAT_FIX: the PRIOR lambda used the unit-quaternion
    // optimized form  v + 2w(qvxv) + 2qvx(qvxv) = (1-2|u|^2)v + 2(u*pt)u + 2w(uxpt).
    // That equals the sandwich ONLY when |q|^2=1 (w^2+|u|^2=1). For the cooked SNORM16
    // decorator quats |q|^2 ~= 1.3-2.2, so (1-2|u|^2) != (w^2-|u|^2) and the old form
    // SHEARED the rotated axes - pulling the blade up-vector OFF vertical. The
    // exhaustive offline sweep (tools/deco_quat_probe.cpp, 24 perm x 16 sign over
    // 243k TALL instances on 6 maps) measured this directly: the old optimized form
    // clusters the template +Z up at mean(up.z)=+0.548; the TRUE sandwich (identical
    // to normalizing q first, since |q|^2R preserves direction) lifts it to
    // mean(up.z)=+0.736 - a real, consistent vertical improvement with no per-BSP
    // overfitting (raw xyzw is ALSO the global #1 component order, so only the
    // FORMULA matters, not the lane order). Harness-grounded: the remaining
    // ~0.74 (not 1.0) is genuine authored terrain-follow tilt, not a decode
    // error.
    auto quatTransform = [](const float q[4], const float v[3], float out[3]) {
        const float ux = q[0], uy = q[1], uz = q[2], w = q[3];
        const float uu = ux*ux + uy*uy + uz*uz;          // |u|^2
        const float a  = w*w - uu;                       // (w^2 - |u|^2)
        const float d  = 2.0f * (ux*v[0] + uy*v[1] + uz*v[2]);   // 2(u*pt)
        // 2w(u x pt)
        const float cx = uy*v[2] - uz*v[1];
        const float cy = uz*v[0] - ux*v[2];
        const float cz = ux*v[1] - uy*v[0];
        out[0] = a*v[0] + d*ux + 2.0f*w*cx;
        out[1] = a*v[1] + d*uy + 2.0f*w*cy;
        out[2] = a*v[2] + d*uz + 2.0f*w*cz;
    };

    // -------------------------------------------------------------------------
    // PER-SET BLADE TEMPLATE - the real render_model geometry.
    //
    // Each decorator set's dctr references a render_model (mode) holding the
    // absolute-sized blade/tuft/rock template (VFMT_DECORATOR, posMin/posMax =
    // vertex_compression_offset/scale). We load the HIGHEST-detail LOD per set
    // (LOD1, +0x00 - DECO_LOD_FIX; env MMS_DECO_LOD overrides) and
    // instance THAT mesh per placement - replacing the prior fixed billboard
    // card. If a set's template can't be read, we fall back to the fixed card
    // for that set ONLY and log it (never emit nothing, never guess a size).
    std::vector<DecoTemplate> setTemplates(dctrTagIds.size());
    for (size_t s = 0; s < dctrTagIds.size(); ++s) {
        setTemplates[s] = LoadDecoratorTemplate(cache, cacheHandleId, dctrTagIds[s], sbspTagId, (uint32_t)s);
    }

    // DECO_COLOR_RE: load the BSP SH airprobe lighting-point grid
    // ONCE for this sbsp when the gate is enabled. Used below to reconstruct the
    // per-instance baked ambient (instance_color) by nearest-point sampling at
    // each blade's world position. Default OFF - when disabled the grid stays
    // empty and no per-vertex color is emitted (Colors==NULL), keeping the
    // shipping normal-only foliage path byte-identical.
    const bool decoColorOn = DecoColorGateEnabled();
    DecoAirprobePoint* airGrid = nullptr;
    uint32_t airGridCount = 0;
    if (decoColorOn) {
        int gok = ZH_LBSP_GetAirprobeGrid(cacheHandleId, sbspTagId, &airGrid, &airGridCount);
        NativeDiag("[DECO_COLOR] sbsp=0x%X gate=ON airprobeGrid ok=%d count=%u",
                   sbspTagId, gok, airGridCount);
        if (!gok || airGridCount == 0) {
            // No usable grid - emit normal-only (Colors stays NULL). Honest:
            // do not invent a color when there's no on-disk source for this BSP.
            if (airGrid) { ZH_LBSP_FreeAirprobeGrid(airGrid); airGrid = nullptr; }
            airGridCount = 0;
        }
    }
    // Per-instance color reconstruction is emitted only when the grid resolved.
    const bool emitDecoColor = decoColorOn && airGrid != nullptr && airGridCount > 0;

    // Fixed billboard-card fallback (ONLY used when a set's render_model template
    // is missing/unreadable). Sized at a typical grass-blade aspect. (The decoded
    // quaternion is a UNIT rotation now - DECO_QUAT_PACKING - so it contributes
    // orientation only, no scale; the card size is the final blade size.)
    constexpr float kCardW = 0.35f;   // half-width applied as +-kCardW
    constexpr float kCardH = 0.7f;    // height (local z, [0..kCardH])
    struct TmplVert { float p[3]; float uv[2]; };
    static const TmplVert kCard[4] = {
        { { -kCardW, 0.0f, 0.0f  }, { 0.0f, 1.0f } },   // bottom-left
        { {  kCardW, 0.0f, 0.0f  }, { 1.0f, 1.0f } },   // bottom-right
        { {  kCardW, 0.0f, kCardH}, { 1.0f, 0.0f } },   // top-right
        { { -kCardW, 0.0f, kCardH}, { 0.0f, 0.0f } },   // top-left
    };
    static const uint16_t kCardIdx[6] = { 0, 1, 2, 0, 2, 3 };
    static const float kCardNormal[3] = { 0.0f, -1.0f, 0.0f }; // card faces -Y in local space

    // Read cluster decorator_groups -> flat list of placement ranges with bounds.
    // DECO_PLACEMENT_CULL_RE: the authoritative cooked layout
    // of decorator_runtime_cluster_block (s_decorator_runtime_block, 60 B, schema
    // GUID 303a54e2..., decorator_tag_definitions.cpp) is:
    //   +0x00 short  placement count   +0x02 char setIdx   +0x03 char bufIdx
    //   +0x04 long   buffer offset      +0x08 vec3 position_bounds_min
    //   +0x14 real   bounding sphere RADIUS  <- decoded here (was skipped)
    //   +0x18 vec3   position_bounds_size    +0x24 vec3 bounding sphere CENTER <-
    //   +0x30 block  model start index (per-LOD-model start word list)
    // The bounding sphere (center @0x24 / radius @0x14) is the engine's coarse
    // per-group cull primitive: at runtime it distance/frustum-culls and LOD-
    // decimates a whole group by this sphere before submitting any placements
    // (cvars render_decorators_decimation_test / render_decorators_lod_mask;
    // per-vertex fade decorators.hlsl_include:187-191). MMS has no distance
    // pipeline yet (gap #5 / docs section 12b) so it draws every group at LOD1 - the
    // boardwalk-bushes over-population. There is NO surface/material cull in the
    // record or on the cluster (only structure_cluster_flags bit-4 "decorators
    // are lit", a lighting flag), so the boardwalk exclusion is a Sapien COOK-TIME
    // decision baked into the placement stream - NOT something MMS may re-derive
    // (never guess). We decode the sphere now so a future
    // renderer pass can implement the engine's distance/sphere cull faithfully;
    // it does NOT change current geometry emission.
    struct DecoGroup {
        uint16_t count; uint8_t setIdx; uint8_t bufIdx; int32_t bufOff;
        float bMin[3]; float bSize[3];
        float sphereCenter[3]; float sphereRadius;   // engine coarse cull primitive
    };
    std::vector<DecoGroup> groups;
    // DECO_GRP30 (T0-5): the 60-byte decorator_group record is mapped only through
    // 0x30 (count/setIdx/bufIdx/bufOff/bMin/sphere/bSize). Review #20 hypothesized a
    // per-LOD model-start-index INSTANCE sub-range at +0x30 so we could draw only the
    // chosen LOD's placements instead of the full G.count (claimed over-draw).
    //
    // CONFIRMED REFUTED: the 12-byte
    // tail at 0x30 is a tag_block { i32 count=2 @0x30; u64 pointer @0x34 } - a per-LOD
    // DESCRIPTOR (2 LODs), NOT an instance sub-range. The 0x34 value increments by +1
    // PER GROUP, not by G.count (e.g. gi0 cnt=147 ptr=...893; gi1 cnt=59 ptr=...894),
    // so it cannot be an instance start-index. G.count IS the correct single-LOD
    // per-group instance count - MMS is NOT over-drawing LODs. So the LOD-subrange fix
    // is invalid; "extra/wrong decorators" is cross-BSP + multi-set overlap (see the
    // DECO_DUP_DIAG multiSetPos accumulator), addressed by the per-group sphere cull
    // (Tier-1), not here. The dump below is retained as a confirmed decorator-RE probe.
    int grp30Budget = 12;
    {
        SbspLayout SLp = PickSbspLayout(cache->cacheType);
        TagBlockRef clBlk = ReadTagBlock(sbsp + SLp.OFF_CLUSTERS);
        if (clBlk.count > 0 && clBlk.count <= 0x10000) {
            int64_t clOff = TagMetaFileOff(cache, clBlk.pointer);
            if (clOff >= 0 &&
                (size_t)clOff + (size_t)clBlk.count * SLp.CLUSTER_BLOCK_SIZE <= cache->size) {
                for (int ci = 0; ci < clBlk.count; ++ci) {
                    const uint8_t* c = cache->base + clOff + (size_t)ci * SLp.CLUSTER_BLOCK_SIZE;
                    TagBlockRef dg = ReadTagBlock(c + 0x50);
                    if (dg.count <= 0 || dg.count > 65536) continue;
                    int64_t dgOff = TagMetaFileOff(cache, dg.pointer);
                    if (dgOff < 0 || (size_t)dgOff + (size_t)dg.count * 60 > cache->size) continue;
                    for (int gi = 0; gi < dg.count; ++gi) {
                        const uint8_t* g = cache->base + dgOff + (size_t)gi * 60;
                        DecoGroup G;
                        G.count  = RU16(g + 0x00);
                        NativeDiag("[DECO_BLK] sbsp=0x%X ci=%d gi=%d count=%u setIdx=%u bufIdx=%u bufOff=%d palette=%d", sbspTagId, ci, gi, (unsigned)RU16(g + 0x00), (unsigned)g[0x02], (unsigned)g[0x03], (int)R32(g + 0x04), (int)dctrTagIds.size());
                        G.setIdx = g[0x02];
                        G.bufIdx = g[0x03];
                        G.bufOff = R32(g + 0x04);
                        memcpy(G.bMin,  g + 0x08, 12);
                        memcpy(&G.sphereRadius, g + 0x14, 4);     // bounding sphere radius
                        memcpy(G.bSize, g + 0x18, 12);
                        memcpy(G.sphereCenter, g + 0x24, 12);     // bounding sphere center
                        if (G.count == 0) continue;
                        if (G.bufIdx >= vbCounts.size()) continue;
                        // #97 PHANTOM-CLUSTER GUARD: reject groups whose placement bounds
                        // are non-finite or absurd - these are corrupt/phantom cluster
                        // records that scatter foliage into wrong places on some maps.
                        // The threshold (1e6 wu) is far beyond any real Reach map extent
                        // (~+-10k wu) so legitimate distant foliage is untouched; only
                        // NaN/inf/garbage groups are dropped. (Bounds-intersection culling
                        // is deliberately avoided without a repro map to prevent regressing
                        // the 6-map-verified decode.)
                        {
                            bool bad = false;
                            for (int a = 0; a < 3; ++a) {
                                if (!isfinite(G.bMin[a]) || !isfinite(G.bSize[a]) ||
                                    fabsf(G.bMin[a]) > 1.0e6f || fabsf(G.bSize[a]) > 1.0e6f) {
                                    bad = true; break;
                                }
                            }
                            if (bad) {
                                NativeDiag("DecoGeom: sbsp=0x%X ci=%d gi=%d PHANTOM group dropped "
                                           "(bMin=%.1f,%.1f,%.1f bSize=%.1f,%.1f,%.1f)",
                                           sbspTagId, ci, gi,
                                           G.bMin[0], G.bMin[1], G.bMin[2],
                                           G.bSize[0], G.bSize[1], G.bSize[2]);
                                continue;
                            }
                        }
                        // GRPDUMP: unconditional per-group dump (setIdx/bounds) to
                        // triage the "broadleaf spans whole plaza" bug - is a set built
                        // from many tight per-cluster groups, or few groups with BSP-wide
                        // bounds? Concise, one line/group.
                        if (getenv("MMS_DECO_GRPDUMP")) {
                            NativeDiag("[GRPDUMP] sbsp=0x%X ci=%d gi=%d set=%u buf=%u cnt=%u "
                                       "bMin=(%.1f,%.1f,%.1f) bSize=(%.2f,%.2f,%.2f) sphR=%.1f",
                                sbspTagId, ci, gi, G.setIdx, G.bufIdx, G.count,
                                G.bMin[0], G.bMin[1], G.bMin[2],
                                G.bSize[0], G.bSize[1], G.bSize[2], G.sphereRadius);
                        }
                        // DECO_POS_DIAG: dump cluster index + raw 60B for EXTREME-bMin
                        // groups (phantom-cluster vs real distant data triage).
                        if (getenv("MMS_DECO_POS_DIAG") &&
                            (fabsf(G.bMin[0]) > 500.f || fabsf(G.bMin[1]) > 500.f || fabsf(G.bMin[2]) > 200.f)) {
                            char hex[200]; int hp = 0;
                            for (int b = 0; b < 60 && hp < 190; ++b) hp += snprintf(hex+hp, sizeof(hex)-hp, "%02X", g[b]);
                            NativeDiag("[DECO_POS_RAW] sbsp=0x%X ci=%d gi=%d/%d cnt=%u setIdx=%u bufIdx=%u bufOff=%d "
                                       "bMin=(%.1f,%.1f,%.1f) bSize=(%.2f,%.2f,%.2f) raw60=%s",
                                sbspTagId, ci, gi, dg.count, G.count, G.setIdx, G.bufIdx, G.bufOff,
                                G.bMin[0],G.bMin[1],G.bMin[2], G.bSize[0],G.bSize[1],G.bSize[2], hex);
                        }
                        // DECO_GRP30 (T0-5): dump the un-mapped 0x30..0x3C tail for the
                        // first groups so the per-LOD sub-range hypothesis can be checked
                        // against real bytes. Several interpretations side-by-side: if it
                        // is a TagBlockRef, i32@0x30 is a small count and the 8 bytes at
                        // 0x34 a plausible pointer; if it is an inline LOD start-index
                        // word-list, the u16 lane(s) should partition G.count (=%u).
                        if (getenv("MMS_DECO_POS_DIAG") && grp30Budget > 0) {
                            --grp30Budget;
                            int32_t  i30 = R32(g + 0x30), i34 = R32(g + 0x34), i38 = R32(g + 0x38);
                            uint16_t w0=RU16(g+0x30), w1=RU16(g+0x32), w2=RU16(g+0x34),
                                     w3=RU16(g+0x36), w4=RU16(g+0x38), w5=RU16(g+0x3A);
                            char hx[40]; int hp=0;
                            for (int b=0x30; b<0x3C && hp<38; ++b) hp += snprintf(hx+hp,sizeof(hx)-hp,"%02X",g[b]);
                            NativeDiag("[DECO_GRP30] sbsp=0x%X gi=%d setIdx=%u count=%u | "
                                       "i32@30=%d i32@34=%d i32@38=%d | "
                                       "u16=[%u %u %u %u %u %u] | raw=%s",
                                sbspTagId, gi, G.setIdx, G.count,
                                i30, i34, i38, w0,w1,w2,w3,w4,w5, hx);
                        }
                        groups.push_back(G);
                    }
                }
            }
        }
    }

    if (groups.empty()) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: no decorator_groups (no instances) - 0 meshes",
            sbspTagId);
        if (airGrid) ZH_LBSP_FreeAirprobeGrid(airGrid);
        free(resourceData);
        return true;
    }

    // -------------------------------------------------------------------------
    // DECO_QUAT_SWEEP (gated, offline). When MMS_DECO_QUAT_PROBE=1 run the
    // exhaustive 384-candidate (24 perm x 16 sign) per-instance up-clustering
    // sweep over the TALL sets and print the ranked table to the native log,
    // then RETURN without baking geometry. The production decode below is
    // UNCHANGED - this is a read-only RE probe.
    // -------------------------------------------------------------------------
    if (DecoQuatProbeGateEnabled()) {
        // Per-set probe info: TALL classification + the template's REAL longest
        // local axis (used as "up" - tests the up-axis assumption directly).
        constexpr float kProbeTallZ = 0.3f;   // task spec: heightZ >= ~0.3 => tall
        std::vector<DecoSetProbeInfo> setInfo(setTemplates.size());
        for (size_t s = 0; s < setTemplates.size(); ++s) {
            DecoSetProbeInfo si{}; si.valid = false;
            const DecoTemplate& T = setTemplates[s];
            if (!T.valid || T.pos.size() < 3) { setInfo[s] = si; continue; }
            float mn[3] = { 1e30f,1e30f,1e30f }, mx[3] = { -1e30f,-1e30f,-1e30f };
            for (size_t v = 0; v + 2 < T.pos.size(); v += 3)
                for (int a = 0; a < 3; ++a) {
                    float c = T.pos[v + a];
                    if (c < mn[a]) mn[a] = c;
                    if (c > mx[a]) mx[a] = c;
                }
            float ext[3] = { mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2] };
            int longAxis = 0;
            if (ext[1] > ext[longAxis]) longAxis = 1;
            if (ext[2] > ext[longAxis]) longAxis = 2;
            si.upAxis[0]=si.upAxis[1]=si.upAxis[2]=0.0f; si.upAxis[longAxis]=1.0f;
            // TALL = template z-extent (heightZ) >= threshold. (heightZ is the
            // local-Z extent; the long-axis test is reported separately below.)
            si.tall = (T.heightZ >= kProbeTallZ);
            si.valid = true;
            setInfo[s] = si;
            NativeDiag("[DECO_QUAT_PROBE] sbsp=0x%X set=%zu heightZ=%.3f ext=(%.3f,%.3f,%.3f) "
                       "longAxis=%c tall=%d",
                sbspTagId, s, T.heightZ, ext[0], ext[1], ext[2],
                "XYZ"[longAxis], si.tall ? 1 : 0);
        }

        std::vector<QuatCandStat> stats(24 * 16);
        for (int p = 0; p < 24; ++p)
            for (int sg = 0; sg < 16; ++sg) {
                QuatCandStat& cs = stats[p*16 + sg];
                cs.perm = p; cs.sign = sg;
            }
        if (!g_quatGlobalInit) {
            for (int p = 0; p < 24; ++p)
                for (int sg = 0; sg < 16; ++sg) {
                    g_quatGlobal[p*16+sg].perm = p; g_quatGlobal[p*16+sg].sign = sg;
                }
            g_quatGlobalInit = true;
        }

        // Walk the SAME instance ranges the production decode uses; for each TALL
        // instance evaluate every candidate against (a) the template true long
        // axis and (b) the fixed local +Z axis.
        const float zAxis[3] = { 0.0f, 0.0f, 1.0f };
        uint64_t tallInst = 0, flatInst = 0;
        for (const auto& G : groups) {
            if (G.setIdx >= setInfo.size() || !setInfo[G.setIdx].valid) continue;
            if ((size_t)G.bufIdx >= vbCounts.size() || (size_t)G.bufIdx >= entry.fixups.size()) continue;
            uint32_t vbFoff = (uint32_t)(entry.fixups[G.bufIdx].offset & 0x0FFFFFFFu);
            uint32_t vbLen  = vbLens[G.bufIdx];
            if ((size_t)vbFoff >= resourceSize) continue;
            const uint8_t* vbBase = resourceData + vbFoff;
            if (G.bufOff < 0) continue;
            size_t rangeStart = (size_t)G.bufOff;
            size_t needBytes  = (size_t)G.count * 16;
            if (rangeStart + needBytes > vbLen) continue;
            if ((size_t)vbFoff + rangeStart + needBytes > resourceSize) continue;

            const DecoSetProbeInfo& si = setInfo[G.setIdx];
            const bool isTall = si.tall;
            if (isTall) { tallInst += G.count; g_quatGlobalTallInst += G.count; }
            else flatInst += G.count;

            // DECO_TYPE_TILT: accumulate per-candidate up.z by the set's
            // PRODUCTION DecoType (heightZ>=0.5 => 1 woody/blade, else 0 ground-
            // scatter/rock/debris) for ALL sets - including the short scatter sets the
            // tall-only sweep below skips. This is the table that answers "are the solid
            // debris over-tilted vs blades, and does cand20 (4-lane) fix it vs cand18".
            {
                if (!g_decoTypeInit) {
                    for (int c = 0; c < kDecoPackCandN; ++c) { g_decoPackWoody[c]=DecoPackStat(); g_decoPackScatter[c]=DecoPackStat(); }
                    g_decoTypeInit = true;
                }
                float setHZ = (G.setIdx < setTemplates.size() && setTemplates[G.setIdx].valid)
                                  ? setTemplates[G.setIdx].heightZ : 0.0f;
                bool woody = (setHZ >= 0.5f);
                if (woody) g_decoWoodyInst += G.count; else g_decoScatterInst += G.count;
                const float upZty[3] = { 0.0f, 0.0f, 1.0f };
                const uint8_t* precTy = vbBase + rangeStart;
                for (uint32_t pi = 0; pi < G.count; ++pi) {
                    const uint8_t* qb = precTy + (size_t)pi * 16 + 8;
                    int16_t s4[4]; memcpy(s4, qb, 8);
                    if (((int)s4[0]*s4[0] + (int)s4[1]*s4[1] + (int)s4[2]*s4[2] + (int)s4[3]*s4[3]) == 0) continue;
                    for (int c = 0; c < kDecoPackCandN; ++c) {
                        float q[4];
                        if (!DecoDecodePack(c, qb, q)) continue;
                        float up[3]; QuatSandwichDir(q, upZty, up);
                        DecoPackStat& st = woody ? g_decoPackWoody[c] : g_decoPackScatter[c];
                        st.sumUpZ += up[2]; st.n++;
                        if (up[2] > 0.9f) st.gt09++;
                        if (up[2] > 0.7f) st.gt07++;
                        float aUpZ = up[2] < 0.0f ? -up[2] : up[2];
                        st.sumAbsUpZ += aUpZ; if (aUpZ > 0.9f) st.absGt09++;
                    }
                }
            }

            if (!isTall) continue;   // TALL sets are the diagnostic

            // DECO_QUAT_PACK_PROBE: FLAT-GROUND classification. A decorator_group's
            // authored position_bounds_size.z (G.bSize[2], == instance_compression_
            // scale.z, decorators.hlsl_include:184) is the Z-extent its placements
            // span. A group sitting on FLAT terrain spans a thin Z slab; a group on a
            // slope/wall spans a tall one. Flat => the blades MUST be vertical, so it
            // is the decisive ground-truth set for the packing test. Threshold 0.5 wu
            // (~half a blade height) is conservative; report both buckets + the count.
            if (!g_decoPackInit) {
                for (int c = 0; c < kDecoPackCandN; ++c) { g_decoPackFlat[c]=DecoPackStat(); g_decoPackAll[c]=DecoPackStat(); }
                g_decoPackInit = true;
            }
            const bool flatGround = (G.bSize[2] <= 0.5f);
            if (flatGround) g_decoFlatInst += G.count; else g_decoSlopeInst += G.count;
            const float upZ[3] = { 0.0f, 0.0f, 1.0f };
            {
                const uint8_t* prec0 = vbBase + rangeStart;
                if (!g_decoScaleInit) { for (int s=0;s<5;++s) g_decoScale[s]=DecoScaleStat(); g_decoScaleInit=true; }
                for (uint32_t pi = 0; pi < G.count; ++pi) {
                    const uint8_t* qb = prec0 + (size_t)pi * 16 + 8;
                    // DECO_SCALE_PROBE: candidate per-instance scale sources from the
                    // SAME 8 quaternion bytes (see DecoScaleStat header). Skip the
                    // all-zero degenerate record (no orientation, no scale).
                    {
                        int16_t s4[4]; memcpy(s4, qb, 8);
                        const float invn = 1.0f/32767.0f;
                        float rx=s4[0]*invn, ry=s4[1]*invn, rz=s4[2]*invn, rw=s4[3]*invn;
                        double rawDot = (double)rx*rx + (double)ry*ry + (double)rz*rz + (double)rw*rw;
                        if (rawDot > 1e-8) {
                            DecoScaleAccum(0, rawDot);                 // srcA = dot(q4raw)
                            DecoScaleAccum(1, sqrt(rawDot));           // srcB = |q4raw|
                            float qu[4]; DecoDecodePack(18, qb, qu);   // cand18 (recon-w, unit)
                            DecoScaleAccum(2, (double)qu[0]*qu[0]+(double)qu[1]*qu[1]
                                              +(double)qu[2]*qu[2]+(double)qu[3]*qu[3]); // srcC ~1 (control)
                            // qi0 (rec+0x08) as the task's alt scale-lane hypotheses.
                            DecoScaleAccum(3, fabs((double)s4[0]) * invn);          // srcD SNORM16 |qi0|
                            DecoScaleAccum(4, ((double)(uint16_t)s4[0]) / 65535.0);  // srcE UNORM16 qi0
                        }
                    }
                    for (int c = 0; c < kDecoPackCandN; ++c) {
                        float q[4];
                        if (!DecoDecodePack(c, qb, q)) continue;
                        float ql = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
                        // SIGN RESOLUTION: the cand3 family rotates template +Z to
                        // world -Z (|up.z|=0.96 but NEGATIVE). The engine VS rotates
                        // the template vertex by the SAME quat for BOTH position and
                        // normal - so if the decoded quat is correct, the verticality
                        // is intrinsic and the residual sign is the template's modeled
                        // up convention. Rotate +Z; the dump's |up.z| is the decisive
                        // (sign-free) verticality. (The production fix flips the
                        // template up to land +Z, validated by frac after the flip.)
                        float up[3]; QuatSandwichDir(q, upZ, up);
                        float tlt = sqrtf(up[0]*up[0]+up[1]*up[1]);
                        float aUpZ = up[2] < 0.0f ? -up[2] : up[2];
                        DecoPackStat& st = flatGround ? g_decoPackFlat[c] : g_decoPackAll[c];
                        st.sumUpZ += up[2]; st.n++;
                        if (up[2] > 0.9f) st.gt09++;
                        if (up[2] > 0.7f) st.gt07++;
                        st.sumQlen += ql; st.sumYawSpread += tlt;
                        st.sumAbsUpZ += aUpZ; if (aUpZ > 0.9f) st.absGt09++;
                        // also accumulate flat candidates into the ALL bucket so the
                        // control (all-tall) row is complete.
                        if (flatGround) {
                            DecoPackStat& sa = g_decoPackAll[c];
                            sa.sumUpZ += up[2]; sa.n++;
                            if (up[2] > 0.9f) sa.gt09++;
                            if (up[2] > 0.7f) sa.gt07++;
                            sa.sumQlen += ql; sa.sumYawSpread += tlt;
                            sa.sumAbsUpZ += aUpZ; if (aUpZ > 0.9f) sa.absGt09++;
                        }
                    }
                }
            }

            const uint8_t* rec0 = vbBase + rangeStart;
            for (uint32_t i = 0; i < G.count; ++i) {
                const uint8_t* rec = rec0 + (size_t)i * 16;
                int16_t qi[4]; memcpy(qi, rec + 8, 8);
                float lane[4] = {
                    (float)qi[0]/32767.0f, (float)qi[1]/32767.0f,
                    (float)qi[2]/32767.0f, (float)qi[3]/32767.0f,
                };
                // skip degenerate (all-zero) quats - they carry no orientation.
                if (lane[0]*lane[0]+lane[1]*lane[1]+lane[2]*lane[2]+lane[3]*lane[3] < 1e-8f)
                    continue;
                for (int p = 0; p < 24; ++p) {
                    const uint8_t* pm = kQuatPerms[p];
                    float base[4] = { lane[pm[0]], lane[pm[1]], lane[pm[2]], lane[pm[3]] };
                    for (int sg = 0; sg < 16; ++sg) {
                        float q[4] = {
                            (sg & 1) ? -base[0] : base[0],
                            (sg & 2) ? -base[1] : base[1],
                            (sg & 4) ? -base[2] : base[2],
                            (sg & 8) ? -base[3] : base[3],
                        };
                        QuatCandStat& cs = stats[p*16 + sg];
                        QuatCandStat& gs = g_quatGlobal[p*16 + sg];
                        // (a) template true long-axis as up
                        float up[3]; QuatRotateUnit(q, si.upAxis, up);
                        cs.tSumZ += up[2]; cs.tN++;
                        gs.tSumZ += up[2]; gs.tN++;
                        if (up[2] > 0.9f) { cs.tGt09++; gs.tGt09++; }
                        if (up[2] > 0.7f) { cs.tGt07++; gs.tGt07++; }
                        // horizontal spread proxy: angle of (up.x,up.y) - but for a
                        // near-vertical up this is the residual tilt direction.
                        float tlt = sqrtf(up[0]*up[0]+up[1]*up[1]);
                        cs.tSumYawAbs += tlt; gs.tSumYawAbs += tlt;
                        // (b) fixed local +Z as up
                        float upz[3]; QuatRotateUnit(q, zAxis, upz);
                        cs.tzSumZ += upz[2]; cs.tzN++;
                        gs.tzSumZ += upz[2]; gs.tzN++;
                        if (upz[2] > 0.9f) { cs.tzGt09++; gs.tzGt09++; }
                        // (c) NORMALIZE-Q-FIRST variant: unit q => pure rotation.
                        float ql = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
                        float qn[4];
                        if (ql > 1e-12f) { float inv=1.0f/sqrtf(ql); qn[0]=q[0]*inv; qn[1]=q[1]*inv; qn[2]=q[2]*inv; qn[3]=q[3]*inv; }
                        else { qn[0]=0; qn[1]=0; qn[2]=0; qn[3]=1; }
                        float nup[3]; QuatRotateUnit(qn, si.upAxis, nup);
                        cs.nSumZ += nup[2]; cs.nN++; gs.nSumZ += nup[2]; gs.nN++;
                        if (nup[2] > 0.9f) { cs.nGt09++; gs.nGt09++; }
                        if (nup[2] > 0.7f) { cs.nGt07++; gs.nGt07++; }
                        float nupz[3]; QuatRotateUnit(qn, zAxis, nupz);
                        cs.nzSumZ += nupz[2]; cs.nzN++; gs.nzSumZ += nupz[2]; gs.nzN++;
                        if (nupz[2] > 0.9f) { cs.nzGt09++; gs.nzGt09++; }
                    }
                }
            }
        }

        // Rank by tall-set mean(up.z) on the template LONG axis. Report the best
        // axis interpretation per candidate (long-axis vs +Z) so a +Z-modeled
        // template isn't penalized for a long-axis misread.
        std::vector<int> order(stats.size());
        for (size_t k = 0; k < stats.size(); ++k) order[k] = (int)k;
        auto meanZ = [&](const QuatCandStat& c) -> double {
            double a = c.tN ? c.tSumZ / (double)c.tN : -2.0;
            double b = c.tzN ? c.tzSumZ / (double)c.tzN : -2.0;
            return (a > b) ? a : b;   // best of the two up-axis interpretations
        };
        std::sort(order.begin(), order.end(), [&](int x, int y){
            return meanZ(stats[x]) > meanZ(stats[y]);
        });

        static const char kLane[4] = { 'x','y','z','w' };
        NativeDiag("[DECO_QUAT_PROBE] sbsp=0x%X tallInst=%llu flatInst=%llu - RANKED top-12 "
                   "(by best tall mean up.z; perm shows lane->xyzw, sign -=negated):",
            sbspTagId, (unsigned long long)tallInst, (unsigned long long)flatInst);
        int printed = 0;
        for (size_t oi = 0; oi < order.size() && printed < 12; ++oi) {
            const QuatCandStat& c = stats[order[oi]];
            if (c.tN == 0) continue;
            const uint8_t* pm = kQuatPerms[c.perm];
            // describe the lane->xyzw mapping with sign, e.g. "x=-q1 y=q2 z=q0 w=q3"
            char desc[96];
            int signs[4] = { (c.sign&1)?1:0, (c.sign&2)?1:0, (c.sign&4)?1:0, (c.sign&8)?1:0 };
            snprintf(desc, sizeof(desc), "x=%sq%d y=%sq%d z=%sq%d w=%sq%d",
                signs[0]?"-":"", pm[0], signs[1]?"-":"", pm[1],
                signs[2]?"-":"", pm[2], signs[3]?"-":"", pm[3]);
            double mL = c.tN ? c.tSumZ/(double)c.tN : -2.0;
            double fL = c.tN ? (double)c.tGt09/(double)c.tN : 0.0;
            double f7 = c.tN ? (double)c.tGt07/(double)c.tN : 0.0;
            double mZ = c.tzN ? c.tzSumZ/(double)c.tzN : -2.0;
            double fZ = c.tzN ? (double)c.tzGt09/(double)c.tzN : 0.0;
            double tilt = c.tN ? c.tSumYawAbs/(double)c.tN : 0.0;
            NativeDiag("[DECO_QUAT_PROBE]  #%d %-40s | LONGaxis mean.z=%+.3f frac>.9=%.2f frac>.7=%.2f tilt=%.3f"
                       " | +Zaxis mean.z=%+.3f frac>.9=%.2f",
                printed + 1, desc, mL, fL, f7, tilt, mZ, fZ);
            (void)kLane;
            ++printed;
        }
        NativeDiag("[DECO_QUAT_PROBE] sbsp=0x%X DONE (probe-only, no geometry baked)", sbspTagId);
        if (airGrid) ZH_LBSP_FreeAirprobeGrid(airGrid);
        free(resourceData);
        *outMeshes = nullptr; *outMeshCount = 0;
        return true;
    }

    // Per-set template vertex count (real render_model template, else 4-vert card).
    uint32_t setCount0 = (uint32_t)dctrTagIds.size();
    auto templateVertCount = [&](uint32_t s) -> uint32_t {
        if (s < setTemplates.size() && setTemplates[s].valid)
            return (uint32_t)(setTemplates[s].pos.size() / 3);
        return 4u;   // fixed card
    };

    // Total instances + a sane cap on expanded vertices. With the real templates
    // a single instance emits templateVertCount(set) verts (24..1344 vs the old
    // fixed 4), so compute the true expanded-vert total and subsample evenly if
    // it exceeds the cap (deterministic stride; coverage stays representative).
    //
    // DECO_CAP_FIX: raised 6M -> 24M. The 6M cap was DECIMATING the
    // decorators (stride 2 = HALF dropped, silently) on every dense map - measured:
    // forge_halo 0x3056 needs 7.36M (133811 inst), m30 BSPs up to 7.81M. Half the
    // foliage was simply not emitted to the surface - the user's "much of them still
    // aren't being put onto the surface" report. 24M clears the observed maxes with
    // 3x headroom so typical maps decimate ZERO instances. The cap still exists as a
    // runaway guard for pathological maps; env MMS_DECO_VERT_CAP_M overrides it
    // (value in millions, 1..256) for coverage-vs-memory tuning without a rebuild.
    uint32_t kCapMillions = 24u;
    if (const char* capEnv = getenv("MMS_DECO_VERT_CAP_M")) {
        int v = atoi(capEnv);
        if (v >= 1 && v <= 256) kCapMillions = (uint32_t)v;
    }
    const uint64_t kMaxExpandedVerts = (uint64_t)kCapMillions * 1024u * 1024u;
    uint64_t totalInst = 0;
    uint64_t wantVerts = 0;
    for (const auto& G : groups) {
        totalInst += G.count;
        wantVerts += (uint64_t)G.count * templateVertCount(G.setIdx < setCount0 ? G.setIdx : 0);
    }
    uint32_t instStride = 1;
    if (wantVerts > kMaxExpandedVerts) {
        instStride = (uint32_t)((wantVerts + kMaxExpandedVerts - 1) / kMaxExpandedVerts);
        NativeDiag("RuntimeDecGeom sbsp=0x%X: CAP - %llu instances would emit %llu verts; "
                   "subsampling every %u-th instance (cap=%llu)",
            sbspTagId, (unsigned long long)totalInst, (unsigned long long)wantVerts,
            instStride, (unsigned long long)kMaxExpandedVerts);
    }

    // Bucket groups by decorator set, flushing to a finished-mesh list whenever a
    // set's vertex accumulator approaches the uint16 index ceiling. A dense set
    // (e.g. forge_halo set[2] ~ 68k verts) therefore yields several meshes that
    // share the same dctr bitmap. Each finished chunk -> one ZH_RuntimeDecoratorMesh.
    uint32_t setCount = (uint32_t)dctrTagIds.size();
    if (setCount == 0) { if (airGrid) ZH_LBSP_FreeAirprobeGrid(airGrid); free(resourceData); return true; }
    constexpr uint32_t kMaxVertsPerMesh = 65532u;   // < 0xFFFF, multiple of 4

    struct Chunk {
        std::vector<float> pos, uv, nrm, col;   // col: DECO_COLOR_RE per-vertex float3 (empty when gate off)
        std::vector<float> sway;                // DECO_WIND_RE per-vertex float3 world-space sway basis
        std::vector<uint16_t> idx;
        uint32_t setIdx = 0;
        uint32_t vbase = 0;
        uint32_t instCount = 0;                 // DECO_BUDGET_RE: instances baked into this chunk
    };
    std::vector<Chunk> finished;
    std::vector<Chunk> openChunk(setCount);   // current open chunk per set
    for (uint32_t s = 0; s < setCount; ++s) openChunk[s].setIdx = s;

    auto flushSet = [&](uint32_t s) {
        if (!openChunk[s].pos.empty()) {
            finished.push_back(std::move(openChunk[s]));
            openChunk[s] = Chunk();
            openChunk[s].setIdx = s;
        }
    };

    // DECO_SEAT_RE: track the world-Z range of authored instance
    // positions. The engine does NOT runtime-snap decorators to the terrain - the
    // authored UHEND3N instance_position (decoded above as pn*bSize+bMin, byte-
    // identical to decorators.hlsl_include:184) IS the final, surface-seated world
    // placement Sapien baked at scenario-compile. So MMS must NOT add a raycast snap
    // (that would move correctly-placed foliage).
    //
    // SEAT VERDICT:
    // Compared the decoded decorator blade-base verts to the decoded BSP cluster
    // (terrain) surface Z at the SAME world XY, across 40+ BSPs on 5 maps
    // (forge_halo, m10, m20, m30, 30_settlement, 70_boneyard). Result: wherever the
    // decorator XY overlaps a real terrain cluster, median(decoZ - terrainZ) ~= 0
    // (forge_halo 0x3056: median -0.005, p10/p90 +-0.4; m20/m30/m10 dense BSPs:
    // median 0.000..0.030). i.e. the foliage DATA is correctly surface-seated and
    // the BSP terrain Z is correctly decoded. The BSP terrain clusters are uniformly
    // vfmt 0x00 (VFMT_WORLD) with an identity [0..1] bbox -> decoded as RAW world-space
    // float (the world-format guard in DecodeBspPositions, NOT the bbox-multiply
    // flavour), so there is NO project_mms_bsp_uncompress collapse here. Positive p90
    // tails are just tall decorators (tree template z up to ~10) whose mid/tip verts
    // legitimately rise above the seat. Both decode paths PROVEN correct - neither the
    // decorator decode nor the BSP uncompress is the floating-foliage cause; do not
    // add a snap.
    float seatZMin = 1e30f, seatZMax = -1e30f;
    uint64_t seatCount = 0;

    // DECO_POS_DIAG: per-group placement diagnostic. The seat harness
    // only validated VERTICAL seating (decoZ vs terrainZ at the decoded XY) - it is
    // BLIND to horizontal mis-placement. User reports decorators "extremely far from
    // where they should be" => some groups get wrong per-cluster bounds (bMin/bSize)
    // or wrong instance ranges. Gate MMS_DECO_POS_DIAG=1: log each group's bounds +
    // first decoded world pos, and the global decoded-position AABB, to spot the
    // garbage-bounds groups. Read-only; no behavior change.
    const bool g_decoPosDiag = (getenv("MMS_DECO_POS_DIAG") != nullptr);
    double gpMin[3] = { 1e30,1e30,1e30 }, gpMax[3] = { -1e30,-1e30,-1e30 };
    // DECO_DUP_DIAG: quantized world-pos -> bitmask of setIdx seen at that spot.
    // multiSetPos = # spots with >1 distinct decorator set (leaf+plant stacked);
    // dupSameSet = # repeated instances of the SAME set at one spot.
    // STATIC so it accumulates across BOTH BSP GetDeco calls in one process run - 
    // exposes CROSS-BSP stacking (0x3056 leaf + 0x3B57 plant at the same world spot).
    static std::unordered_map<int64_t, uint32_t> posSetMask;
    uint64_t dupSameSet = 0, multiSetPos = 0;

    // DECO_SCALE_RE: per-set applied-scale spread accumulator (verifies the PRODUCTION
    // bake - not just the probe - now varies blade size per instance). Emitted once
    // per set after the bake (gated MMS_NATIVE_LOG=1).
    std::vector<float>    setScaleMin(setCount, 1e30f), setScaleMax(setCount, -1e30f);
    std::vector<double>   setScaleSum(setCount, 0.0);
    std::vector<uint64_t> setScaleN(setCount, 0);

    for (const auto& G : groups) {
        if (G.setIdx >= setCount) continue;
        if ((size_t)G.bufIdx >= vbCounts.size() || (size_t)G.bufIdx >= entry.fixups.size())
            continue;
        uint32_t vbFoff = (uint32_t)(entry.fixups[G.bufIdx].offset & 0x0FFFFFFFu);
        uint32_t vbLen  = vbLens[G.bufIdx];
        if ((size_t)vbFoff >= resourceSize) continue;
        const uint8_t* vbBase = resourceData + vbFoff;
        if (G.bufOff < 0) continue;
        size_t rangeStart = (size_t)G.bufOff;
        size_t needBytes  = (size_t)G.count * 16;
        if (rangeStart + needBytes > vbLen) continue;
        if ((size_t)vbFoff + rangeStart + needBytes > resourceSize) continue;

        // Pick this set's template (real render_model blade, else fixed card).
        const DecoTemplate* TS = (G.setIdx < setTemplates.size() && setTemplates[G.setIdx].valid)
                                    ? &setTemplates[G.setIdx] : nullptr;

        const uint8_t* rec0 = vbBase + rangeStart;
        for (uint32_t i = 0; i < G.count; i += instStride) {
            const uint8_t* rec = rec0 + (size_t)i * 16;
            if (i == 0) {
                uint32_t hist[8] = {0,0,0,0,0,0,0,0}; uint32_t b0h[4] = {0,0,0,0};
                for (uint32_t j = 0; j < G.count; j += instStride) { const uint8_t* r2 = rec0 + (size_t)j * 16; hist[r2[7] & 7]++; b0h[r2[4] >> 6]++; }
                NativeDiag("[DECO_SUB] sbsp=0x%X set=%u count=%u subpart hist=[%u %u %u %u %u %u %u %u] B0>>6 hist=[%u %u %u %u] B1=0x%02X",
                    sbspTagId, G.setIdx, G.count, hist[0],hist[1],hist[2],hist[3],hist[4],hist[5],hist[6],hist[7], b0h[0],b0h[1],b0h[2],b0h[3], rec[5]);
            }
            // DECO_SUBPART: byte 0x7 = subpart_index -> this placement draws ONE section of the set model.
            // (Before: every placement drew ALL sections merged - "leaf+plant stacked" and the planter plant
            // showing up at every placement of the set, e.g. Panopticon stairs.)
            const DecoTemplate* T = TS;
            if (TS && !TS->subs.empty()) {
                uint32_t sub = rec[0x07];
                if (TS->typeCount > 0 && sub >= TS->typeCount) continue;   // invalid type -> engine draws nothing
                if (sub >= TS->subs.size()) sub = 0;
                T = &TS->subs[sub];
            }
            const uint32_t tVerts = T ? (uint32_t)(T->pos.size() / 3) : 4u;
            const uint32_t tIdx   = T ? (uint32_t)(T->idx.size())     : 6u;
            if (openChunk[G.setIdx].vbase + tVerts > kMaxVertsPerMesh) flushSet(G.setIdx);
            Chunk& A = openChunk[G.setIdx];
            uint32_t posPacked; memcpy(&posPacked, rec + 0, 4);
            float pn[3]; unpackUHEND3N(posPacked, pn);
            float wpos[3] = {
                pn[0] * G.bSize[0] + G.bMin[0],
                pn[1] * G.bSize[1] + G.bMin[1],
                pn[2] * G.bSize[2] + G.bMin[2],
            };
            // DECO_SEAT_RE: accumulate authored seat-Z range (see flushSet header).
            if (wpos[2] < seatZMin) seatZMin = wpos[2];
            if (wpos[2] > seatZMax) seatZMax = wpos[2];
            ++seatCount;
            if (g_decoPosDiag && G.setIdx < 31) {
                // Quantize to 0.25u grid; track which sets land on each spot.
                int64_t qx = (int64_t)llroundf(wpos[0] * 50.0f);
                int64_t qy = (int64_t)llroundf(wpos[1] * 50.0f);
                int64_t qz = (int64_t)llroundf(wpos[2] * 50.0f);
                int64_t key = (qx & 0x1FFFFF) | ((qy & 0x1FFFFF) << 21) | ((qz & 0x3FFFFF) << 42);
                uint32_t bit = 1u << G.setIdx;
                auto it = posSetMask.find(key);
                if (it == posSetMask.end()) { posSetMask[key] = bit; }
                else {
                    if (it->second & bit) ++dupSameSet;        // same set repeated here
                    else { if ((it->second & (it->second - 1)) == 0) ++multiSetPos; it->second |= bit; }
                }
            }
            if (g_decoPosDiag) {
                for (int a = 0; a < 3; ++a) {
                    if (wpos[a] < gpMin[a]) gpMin[a] = wpos[a];
                    if (wpos[a] > gpMax[a]) gpMax[a] = wpos[a];
                }
                if (i == 0) {
                    NativeDiag("[DECO_POS] sbsp=0x%X clusterGrp set=%u bufIdx=%u cnt=%u "
                               "bMin=(%.2f,%.2f,%.2f) bSize=(%.2f,%.2f,%.2f) wpos0=(%.2f,%.2f,%.2f)",
                        sbspTagId, G.setIdx, G.bufIdx, G.count,
                        G.bMin[0], G.bMin[1], G.bMin[2],
                        G.bSize[0], G.bSize[1], G.bSize[2],
                        wpos[0], wpos[1], wpos[2]);
                }
            }
            // 8-byte quaternion field @ rec+0x08 (4xSNORM16 lanes qi0..qi3).
            //
            // DECO_QUAT_YAW (ENGINE-FAITHFUL - supersedes the earlier
            // "3-component compressed quat + reconstructed-w + Rx180" decode). The
            // engine decorator VS reads ALL FOUR lanes .wzyx as the quaternion
            // (x,y,z,w) = (qi3, qi2, qi1, qi0) and applies the standard Hamilton
            // sandwich, with NO w-reconstruction and NO axis flip:
            //   decorators.hlsl_include:202  instance_quaternion = instance_input.quaternion.wzyx
            //   :271  rotated_position = quaternion_transform_point(instance_quaternion, vertex)
            //   quaternions.hlsl_include:65  quaternion_transform_point = q*v*q'  (q.xyz vec, q.w scalar)
            // qi0 is the REAL scalar w. The render_model template is modeled +Z-up
            // (verts run z~=0..heightZ from the base), so the raw wzyx quaternion maps
            // template +Z -> world +Z directly - no template-up flip exists in the
            // engine. This is DecoDecodePack cand1 ("RAW wzyx").
            //
            // WHY THIS FIXES THE YAW: the old path DROPPED qi0 and reconstructed
            // w=sqrt(1-x^2-y^2-z^2), which discards w's SIGN -> recovers the rotation only up
            // to a flip, then composed an Rx180 to undo the resulting up-axis
            // inversion. On flat ground Rx180 and Ry180 gave identical verticality but
            // differed by a 180 deg YAW (an offline-undecidable mirror - the source of the
            // open ambiguity). Reading qi0 as the real w keeps the FULL authored
            // rotation incl. yaw, so the flip - and the ambiguity - vanish. The blade
            // winding/normals are unchanged: the decode is a single proper rotation
            // (det +1) applied identically to positions and normals (engine: line 280),
            // so it can never invert winding or flip a normal.
            //
            // DECO_SCALE_RE: the engine's sandwich is UN-normalized, so a non-unit q
            // contributes |q|^2 as the per-instance UNIFORM SCALE (instance_scale =
            // dot(quaternion,quaternion), decorators.hlsl_include:256). We split that
            // out: normalize q for ORIENTATION below and re-apply |q_raw|^2 as an
            // explicit uniform multiply at the vertex transform - exactly equivalent to
            // the engine's single un-normalized sandwich (world = |q|^2*R(v) + instPos).
            // HARNESS EVIDENCE (tools/deco_quat_probe.cpp [DECO_SCALE], 6-map 243k tall
            // instances, env MMS_DECO_QUAT_PROBE=1): src0 dot(q4raw) mean=1.474 sd=0.313
            // min~=0.0 max=2.728 - a smooth bell over ~0.5..2.0 (matches the engine "max
            // scale 2.0", line 193). The unit-recon control (src2) floors at exactly 1.0
            // (mean 1.30, min 1.000) - proving the normalize collapsed size variation.
            // qi0-as-a-lane candidates (src3 |qi0|: mean 0.39 max 0.99; src4 UNORM: mean
            // 0.20) are clamped <1 and clustered near 0 - NOT a plausible scale; rejected.
            // So scale = dot of the 4 RAW SNORM16 lanes (the engine's literal value).
            int16_t qi[4];
            memcpy(qi, rec + 8, 8);
            // DECO_I8Q: engine s_decorator_runtime_placement = Q_I..Q_W as 4 SIGNED BYTES at 0x8..0xB (i8 * sqrt(2)/127;
            // the quaternion is NON-unit, |q|^2 = instance scale, decorators.hlsl_include:217) and baked RGBE colour at
            // 0xC..0xF. Reading 4 x int16 over 0x8..0xF mixed the colour bytes into rotation AND scale (clamped x2 ->
            // small plant variants rendered as full, tilted planter plants on Panopticon's stairs). MMS_DECO_I8Q=0 reverts.
            static int s_i8q = -1;
            if (s_i8q < 0) { const char* e = getenv("MMS_DECO_I8Q"); s_i8q = (e && *e) ? atoi(e) : 1; }
            float i8q[4] = {0.0f, 0.0f, 0.0f, 1.0f}; float i8q2 = 1.0f;
            if (s_i8q) { const int8_t* qb = (const int8_t*)(rec + 8); const float k = 1.41421356f / 127.0f;
                i8q[0] = qb[0] * k; i8q[1] = qb[1] * k; i8q[2] = qb[2] * k; i8q[3] = qb[3] * k;
                i8q2 = i8q[0]*i8q[0] + i8q[1]*i8q[1] + i8q[2]*i8q[2] + i8q[3]*i8q[3]; }
            float instScale;
            float q[4];
            {
                const float invn = 1.0f / 32767.0f;
                // DECO_SCALE_FIX: per-instance GEOMETRY scale = |q_raw|
                // (LINEAR magnitude of the 4 raw SNORM16 lanes), NOT |q_raw|^2.
                //
                // WHY (the prior |q|^2 was over-scaling "most" decorators, user-reported):
                // - The engine documents a hard MAX SCALE OF 2.0 (decorators.hlsl_include
                //    :193/:272 "max scale of 2.0 is built into vertex compression").
                // - HARNESS (deco_quat_probe, forge_halo, 6319 tall inst): src0 dot(q4raw)
                //    = |q|^2 maxes at 2.606 - it VIOLATES the engine's 2.0 cap. src1 |q4raw|
                //    (linear) maxes at 1.614 and means 1.273 - it RESPECTS the 2.0 cap.
                //    => the authored geometry scale is the LINEAR |q|, not |q|^2.
                // - The dot(q,q)=|q|^2 at decorators.hlsl_include:256 is the WIND
                //    height_squared term ONLY (a naturally squared quantity for the bend
                //    math), NOT the geometry size. Path A (line 91) confirms geometry uses
                //    an explicit LINEAR scalar (instance_position_and_scale.w) with a
                //    pre-normalized quaternion. The prior code conflated the two.
                // - Symptom match: |q|^2 inflated the ~5000/6319 instances above 1.25 (the
                //    "most" the user saw as wrong size) up to 2.6x; the near-unit few looked
                //    right. |q| tempers the high end (cap 1.6x) and leaves near-unit alone.
                float r0 = qi[0]*invn, r1 = qi[1]*invn, r2 = qi[2]*invn, r3 = qi[3]*invn;
                float q2 = r0*r0 + r1*r1 + r2*r2 + r3*r3;
                // #227b DECO_SCALE: engine literal is dot(q,q)=|q|^2 (decorators.hlsl_include:206),
                // CLAMPED to the documented 2.0 cap (decorators.hlsl_include:272). RE_deco_orientation_oracle
                // proved the AUTHORING quat is exactly UNIT with a separate explicit scale float - the runtime
                // quat carries scale in its magnitude^2, but uncapped it reaches 2.6-2.76, over-inflating plants.
                // DECO_SCALE_MODE2: DEFAULT = 1 (|q| LINEAR, clamped 2.0). The
                // |q|^2 alternative over-scales rubble/debris - |q|^2 reaches 2.6,
                // VIOLATING the engine's documented "max scale 2.0"
                // (decorators.hlsl_include:272); linear |q| maxes ~1.6 and respects it (the
                // original harness-backed DECO_SCALE_FIX choice, before the |q|^2 revert). Plants
                // still read correct at linear |q| (Sapien A/B). Modes: 0=|q|^2, 1=|q| lin, 2=none.
                static int s_scaleMode = -1;
                if (s_scaleMode < 0) { const char* e = getenv("MMS_DECO_SCALE_MODE"); s_scaleMode = (e&&*e)?atoi(e):1; }
                if (s_i8q)                 { instScale = i8q2 > 8.0f ? 8.0f : (i8q2 < 1e-3f ? 1.0f : i8q2); }
                else if (s_scaleMode == 2) instScale = 1.0f;
                else if (s_scaleMode == 1) { instScale = sqrtf(q2 > 4.0f ? 4.0f : q2); if (instScale > 2.0f) instScale = 2.0f; }
                else                       instScale = q2 > 2.0f ? 2.0f : q2;   // |q|^2 clamped 2.0
                if (instScale < 1e-2f) instScale = 1.0f;    // degenerate all-zero -> no scale
                // DECO_QUAT_HARNESS_REVERT: ORIENTATION = cand18
                // (cand3  o  Rx180). HARNESS-PROVEN by tools/deco_quat_probe.exe on
                // forge_halo.map (DECO_QUAT_PACK flat-ground table is authoritative):
                //   cand18 (cand3 o Rx180): FLAT up.z = +0.996, f.9=0.99, f.7=1.00 - 
                //     CORRECT (blades vertical). [cand19 = cand3 o Ry180 also +0.996.]
                //   cand1 (RAW wzyx = (qi3,qi2,qi1,qi0)): FLAT up.z = -0.767 - UPSIDE
                //     DOWN. This was the "engine-faithful wzyx, qi0=real-w"
                //     change; its "engine source" reasoning was FALSIFIED by the
                //     harness - the decorator template is +Z-up (template z runs
                //     0..height) and cand1 sends template +Z to -0.767 (DOWN). The
                //     harness is ground truth here; production reverts to cand18.
                // We call DecoDecodePack(18, ...) verbatim - the SAME code path the
                // harness's cand18 uses - so the production orientation is DEFINITIONALLY
                // the +0.996 decode (qb = rec+8 = the 4xSNORM16 quaternion lanes).
                // The Rx180-vs-Ry180 split is a cosmetic 180 deg YAW only (both +0.996
                // vertical); we keep Rx180 (cand18, the prior shipped choice).
                //
                // We normalize for ORIENTATION only; the |q_raw|^2 magnitude rides
                // instScale above and is re-applied explicitly at the vertex transform
                // (equivalent to the engine's single un-normalized sandwich).
                // * DECO_QUAT_FIX (2026-08, RE_decorator_instance_transform - engine-source
                // + real-byte PROVEN): the on-disk quaternion is 4xSNORM16 in NATURAL
                // (x,y,z,w) memory order - NOT smallest-three, NOT fp16, and NOT the 3-lane
                // reconstruct-w that cand18/cand19 used. decorators.hlsl_include reads all
                // FOUR lanes (`float4 quaternion : NORMAL1`) and takes dot(q,q) for scale - 
                // there is no drop-index / sqrt(1-Sigma) reconstruct / largest-component logic
                // anywhere, so a 3-lane packing is structurally impossible. cand18/cand19
                // FABRICATE w=sqrt(1-x^2-y^2-z^2), forcing |q|=1 and DISCARDING the real scalar +
                // its sign; on flat ground the invented w happens to look vertical, but on
                // SLOPED placements it has the wrong axis - which is exactly the "foliage
                // floating / growing from wrong places / clipping" on panopticon's varied
                // terrain. The correct decode is cand0: read s0,s1,s2,s3 at rec+0x08 as
                // x,y,z,w (each /32767), normalize for orientation (the |q|^2 scale rides
                // instScale, decoded above). PROVEN: over 12654 tall instances cand0's mean
                // rotated template-up = +0.732 (OUT of the ground); cand1 (.wzyx, s0=scalar)
                // = -0.732 (INTO the ground - physically impossible). The HREK `.wzyx`
                // swizzle is a 360 vfetch-order artifact; the MCC re-cook stores natural
                // (x,y,z,w) order so HMS reads it straight.
                // * ORIENTATION = cand18 (3-lane wzyx + reconstruct-w + Rx180), HARDCODED. This
                // SUPERSEDES the cand1 claim below it, which is WRONG (visibly broken:
                // debris lay flat/inverted, plants tilted). DECISIVE EMPIRICAL PROOF - the
                // DECO_QUAT_TYPE harness over 146,940 real instances:
                //   cand1 (RAW wzyx, shipped): up.z=-0.73/-0.77, s.9=0.00  -> ZERO instances upright (inverted INTO ground)
                //   cand18 (cand3 o Rx180):    up.z=+0.99,       s.9=0.98  -> 98% stand vertical
                // The on-disk quaternion is 3-LANE (scalar s0 dropped, w reconstructed = sqrt(1-x^2-y^2-z^2));
                // cand3 (wzyx+reconW) yields a valid UNIT quat but 180 deg-flipped (up.z=-0.99), and the Rx180
                // corrects it upright. This is NOT "fabricating w" - the packing genuinely omits the scalar.
                // Templates are +Z-up (Blam); the correct quat rotates +Z -> world--Z (Sapien blade-up=local -Z),
                // which the Rx180 compensates. The prior "engine reads .wzyx as raw 4-lane" note mistook the
                // post-input-assembler HLSL swizzle for the on-disk lane order. Scale still rides |q|^2 clamped
                // 2.0 (separate, below). No env knob.
                // DECO_QUAT_CAND: A/B knob to find the packing that stands
                // the panopticon broadleaf ferns upright (they render FLAT with cand18;
                // engine bb_1 shows them vertical). Default keeps cand18; MMS_DECO_QUAT_CAND
                // overrides for calibration. Removed once the correct cand is pinned.
                static int s_quatCand = -1;
                if (s_quatCand < 0) {
                    const char* e = getenv("MMS_DECO_QUAT_CAND");
                    s_quatCand = (e && *e) ? atoi(e) : 18;
                }
                float fq[4];
                if (s_i8q) {
                    if (i8q2 > 1e-8f) { float r = 1.0f/sqrtf(i8q2); q[0]=i8q[0]*r; q[1]=i8q[1]*r; q[2]=i8q[2]*r; q[3]=i8q[3]*r; }
                    else { q[0]=0.0f; q[1]=0.0f; q[2]=0.0f; q[3]=1.0f; }
                    // DECO_I8Q_FIX: model-local frame convention (template local Z vs engine): compose a fixed
                    // rotation r into q (q' = q * r, r applied first). MMS_DECO_I8Q_FIX selects the variant.
                    static int s_fix = -1;
                    if (s_fix < 0) { const char* e = getenv("MMS_DECO_I8Q_FIX"); s_fix = (e && *e) ? atoi(e) : 5; } // default 5 = Rx180: model-local frame (blades point +Z); same composition the old empirically-verified cand18 decode carried
                    if (s_fix > 0) {
                        const float h = 0.70710678f;
                        // (x,y,z,w) fixed rotations
                        // 8 = Rx180*Rz+90 = (h,-h,0,0)? computed: (1,0,0,0)*(0,0,h,h) = (h, -h, 0, 0)... use explicit table
                        static const float R[12][4] = {
                            {0,0,0,1}, {h,0,0,h}, {-h,0,0,h}, {0,h,0,h}, {0,-h,0,h}, {1,0,0,0}, {0,0,h,h}, {0,0,-h,h},
                            {h,-h,0,0}, {h,h,0,0}, {0,0,1,0}, {0,1,0,0} };
                        const float* r = R[s_fix % 12];
                        // Hamilton product q*r with (x,y,z,w)
                        float ax=q[0],ay=q[1],az=q[2],aw=q[3], bx=r[0],by=r[1],bz=r[2],bw=r[3];
                        q[0] = aw*bx + ax*bw + ay*bz - az*by;
                        q[1] = aw*by - ax*bz + ay*bw + az*bx;
                        q[2] = aw*bz + ax*by - ay*bx + az*bw;
                        q[3] = aw*bw - ax*bx - ay*by - az*bz;
                    }
                } else if (DecoDecodePack(s_quatCand, rec + 8, fq)) {
                    float l = fq[0]*fq[0] + fq[1]*fq[1] + fq[2]*fq[2] + fq[3]*fq[3];
                    if (l > 1e-8f) { float r = 1.0f/sqrtf(l); q[0]=fq[0]*r; q[1]=fq[1]*r; q[2]=fq[2]*r; q[3]=fq[3]*r; }
                    else { q[0]=0.0f; q[1]=0.0f; q[2]=0.0f; q[3]=1.0f; }
                } else { q[0]=0.0f; q[1]=0.0f; q[2]=0.0f; q[3]=1.0f; }   // degenerate -> identity
            }

            // DECO_SCALE_RE: track the per-set applied-scale spread for the post-bake
            // verification log (proves size now varies per instance in production).
            if (G.setIdx < setScaleN.size()) {
                if (instScale < setScaleMin[G.setIdx]) setScaleMin[G.setIdx] = instScale;
                if (instScale > setScaleMax[G.setIdx]) setScaleMax[G.setIdx] = instScale;
                setScaleSum[G.setIdx] += instScale; setScaleN[G.setIdx]++;
            }

            // DECO_WIND_RE: per-instance motion_scale +
            // the world-space sway-axis basis. The engine takes
            //     motion_scale = instance_auxilary_info.y / 256   (decorators.hlsl_include:208)
            // BUT instance_auxilary_info is the SWIZZLED aux dword:
            //     instance_auxilary_info = instance_input.auxilary_info.wzyx   (line 182)
            // The aux dword's 4 UBYTE lanes map to on-disk record bytes in memory order
            // (B0=rec+0x04 -> .x, B1=rec+0x05 -> .y, B2=rec+0x06 -> .z, B3=rec+0x07 -> .w).
            // After the `.wzyx` reverse, instance_auxilary_info.y == auxilary_info.z ==
            // B2 == rec+0x06. So motion_scale is rec[0x06], NOT rec[0x05].
            //   (type_index = instance_auxilary_info.x = auxilary_info.w = B3 = rec+0x07.)
            // The original DECO_WIND_RE read rec[0x05] (B1) - that lane is
            // instance_auxilary_info.z, which the shader NEVER reads; the prior "0xCD
            // constant at B1" was a red-herring constant in an unused lane, so trees
            // whose real motion_scale (B2) is non-zero got a wrong/zero amplitude and
            // STOPPED swaying (user-reported regression). Reading the correct B2 byte
            // restores per-type sway: static ground-cover/flowers carry low/zero B2
            // (stay planted), grass + tall tree decorators carry high B2 (full sway).
            // The wave is added to the TEMPLATE-LOCAL x then rotated by the quaternion
            // (lines 263-271); since quat*v is linear, the world sway displacement for
            // amplitude `a` is quat*(a,0,0). We bake the per-vertex world sway basis =
            // quat*(1,0,0) * (motion_scale*heightGate) so the VS only multiplies by the
            // animated sin(phase). When B2 is 0 the basis is zero -> "no sway" for that
            // instance (data-driven, not a guessed default).
            float motionScale = (float)rec[0x06] / 256.0f;   // aux B2 (= instance_auxilary_info.y after .wzyx)
            float swayAxis[3]; { const float ex[3] = {1.0f, 0.0f, 0.0f}; quatTransform(q, ex, swayAxis); }

            // DECO_WIND_RE FIX: dump the raw aux bytes for the first few
            // instances of each set (bounded) so the user's next run reveals which byte
            // is the real per-type motion_scale. Trees should show a non-zero B2 (now
            // used); if instead a different byte is non-zero for trees, this surfaces it.
            if (g_decoAuxLogBudget > 0 && i == 0) {
                --g_decoAuxLogBudget;
                NativeDiag("[DECO_AUX] sbsp=0x%X set=%u i0 aux B0=0x%02X B1=0x%02X B2=0x%02X B3=0x%02X "
                           "(motion_scale=B2/256=%.3f, type_index=B3) wpos=(%.2f,%.2f,%.2f)",
                           sbspTagId, G.setIdx, rec[0x04], rec[0x05], rec[0x06], rec[0x07],
                           motionScale, wpos[0], wpos[1], wpos[2]);
                // #227 FLOAT DIAG: template local-Z base/extent + applied scale. If T->zmin > 0 the
                // model geometry sits ABOVE its origin -> seating the origin on the floor floats it by
                // zmin*instScale (the suspected cause the #166 re-seat never handled). If zmin~=0 the
                // float is a placement/position issue instead. seat wpos.z + zmin*scale = rendered base Z.
                NativeDiag("  [DECO_FLOAT] sbsp=0x%X set=%u zmin=%.3f zmax=%.3f heightZ=%.3f instScale=%.3f "
                           "seatZ=%.3f renderedBaseZ=%.3f",
                           sbspTagId, G.setIdx, T ? T->zmin : 0.0f, T ? T->zmax : 0.0f,
                           T ? T->heightZ : 0.0f, instScale, wpos[2],
                           wpos[2] + (T ? T->zmin : 0.0f) * instScale);
            }

            // DECO_COLOR_RE: per-instance baked ambient - sample the SH airprobe
            // grid at this blade's world position (nearest point). Same color is
            // applied to every vertex of this instance (per-instance granularity,
            // matching the engine's per-placement instance_color). Gate-off path
            // leaves instCol unused (emitDecoColor==false).
            float instCol[3] = { 1.0f, 1.0f, 1.0f };
            if (emitDecoColor && s_i8q && rec[0x0F] != 0) {
                // DECO_I8Q: engine baked instance colour (light_placement RGBE @0xC..0xF); decorators.hlsl_include:254
                const float ee = exp2f((float)rec[0x0F] / 255.0f * 63.75f - 31.75f);
                instCol[0] = rec[0x0C] / 255.0f * ee; instCol[1] = rec[0x0D] / 255.0f * ee; instCol[2] = rec[0x0E] / 255.0f * ee;
            } else if (emitDecoColor) {
                int aprIdx = -1;
                SampleNearestAirprobe(airGrid, airGridCount, wpos, instCol, &aprIdx);
                if (g_decoColorLogBudget > 0) {
                    --g_decoColorLogBudget;
                    NativeDiag("[DECO_COLOR] sbsp=0x%X set=%u blade i=%u pos=(%.2f,%.2f,%.2f) "
                               "sampledRGB=(%.4f,%.4f,%.4f) ptIdx=%d",
                               sbspTagId, G.setIdx, i, wpos[0], wpos[1], wpos[2],
                               instCol[0], instCol[1], instCol[2], aprIdx);
                }
            }

            const uint32_t vbaseLocal = A.vbase;
            for (uint32_t v = 0; v < tVerts; ++v) {
                const float* tp; const float* tuv; const float* tn;
                if (T) { tp = &T->pos[v*3]; tuv = &T->uv[v*2]; tn = &T->nrm[v*3]; }
                else   { tp = kCard[v].p;   tuv = kCard[v].uv; tn = kCardNormal;  }
                // world = instScale * quaternion_transform_point(q, template_vertex)
                //         + instance_pos.
                // DECO_SCALE_RE: q is a UNIT rotation (orientation only); the
                // per-instance UNIFORM SCALE = |q_raw| (LINEAR, instScale, decoded above
                // - see DECO_SCALE_FIX: linear respects the engine's max-scale-2.0 cap
                // where |q|^2 overshot to 2.6) is applied explicitly here. Uniform scale
                // commutes with rotation, so
                // multiplying the rotated vertex == scaling the template vertex first.
                // The template ORIGIN is the blade base (verts run z~=0..heightZ; the
                // seat note confirms decoZ~=terrainZ at the base), and we scale BEFORE
                // adding instance_pos - so the base stays seated at instance_pos and the
                // blade grows up/out (no lift/sink). Matches the engine exactly.
                float rp[3];
                quatTransform(q, tp, rp);
                A.pos.push_back(rp[0] * instScale + wpos[0]);
                A.pos.push_back(rp[1] * instScale + wpos[1]);
                A.pos.push_back(rp[2] * instScale + wpos[2]);
                A.uv.push_back(tuv[0]);
                A.uv.push_back(tuv[1]);
                // world_normal = quaternion_transform_point(q, template_normal), then
                // renormalize to drop the scale (decorators.hlsl_include:280,282).
                float rn[3];
                quatTransform(q, tn, rn);
                float nl = rn[0]*rn[0] + rn[1]*rn[1] + rn[2]*rn[2];
                if (nl > 1e-12f) { float inv = 1.0f / sqrtf(nl); rn[0]*=inv; rn[1]*=inv; rn[2]*=inv; }
                else { rn[0]=0; rn[1]=0; rn[2]=1; }
                A.nrm.push_back(rn[0]); A.nrm.push_back(rn[1]); A.nrm.push_back(rn[2]);
                // DECO_COLOR_RE: per-instance baked ambient, replicated per vertex.
                if (emitDecoColor) {
                    A.col.push_back(instCol[0]); A.col.push_back(instCol[1]); A.col.push_back(instCol[2]);
                }
                // DECO_WIND_RE: per-vertex world-space sway basis. heightGate uses the
                // TEMPLATE-LOCAL height tp[2] (saturate(abs(z)), engine line 174) so the
                // root (z~=0) is planted and the tip (z up the blade) bends - NOT the
                // world altitude (which would translate the whole blade). Pre-rotated
                // X axis * (motion_scale * heightGate); the VS multiplies by sin(phase).
                float heightGate = fabsf(tp[2]); if (heightGate > 1.0f) heightGate = 1.0f;
                // DECO_SCALE_RE: scale the world-space sway displacement by instScale
                // too - in the engine the wave is added to the template-local vertex and
                // then runs through the SAME un-normalized |q|^2 sandwich, so a larger
                // blade sways proportionally further (keeps sway consistent with size).
                float swayAmp = motionScale * heightGate * instScale;
                // DECO_TRUNK_FIX: the heightGate saturates at z=1, so a
                // TALL woody decorator (tree/bush - heightZ>=0.5) gives its ENTIRE
                // trunk above 1u the FULL horizontal sway, which reads as the trunk
                // "expanding/contracting" in the wind. Trees should be rigid -> zero
                // the sway for woody sets. Ground scatter (grass, heightZ<0.5) still
                // sways. T is this group's set template; heightZ is its local height.
                if (T != nullptr && T->heightZ >= 0.5f) swayAmp = 0.0f;
                A.sway.push_back(swayAxis[0] * swayAmp);
                A.sway.push_back(swayAxis[1] * swayAmp);
                A.sway.push_back(swayAxis[2] * swayAmp);
            }
            for (uint32_t k = 0; k < tIdx; ++k) {
                uint16_t e = T ? T->idx[k] : kCardIdx[k];
                A.idx.push_back((uint16_t)(vbaseLocal + e));
            }
            A.vbase += tVerts;
            A.instCount += 1;
        }
        // DECO_BUDGET_RE: emit ONE mesh per per-cluster group (flush the
        // set's open chunk at each group boundary) so the renderer can distance-sort
        // groups and draw nearest-first up to an instance budget - the engine's
        // decorator decimation (haloreach.dll sub_1806F9FAC). Previously all of a set's
        // groups accumulated into a few plaza-spanning chunks that could only be drawn
        // whole-or-nothing, so HMS drew 100% of authored candidates (gap #5).
        flushSet(G.setIdx);
    }
    // DECO_SCALE_RE: per-set applied-scale spread (verifies blade size now varies).
    for (uint32_t s = 0; s < setCount; ++s) {
        if (setScaleN[s] == 0) continue;
        NativeDiag("[DECO_SCALE_APPLIED] sbsp=0x%X set=%u n=%llu scale min=%.3f max=%.3f mean=%.3f",
            sbspTagId, s, (unsigned long long)setScaleN[s],
            setScaleMin[s], setScaleMax[s], setScaleSum[s] / (double)setScaleN[s]);
    }
    for (uint32_t s = 0; s < setCount; ++s) flushSet(s);

    if (finished.empty()) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: instanced-decode produced 0 meshes "
                   "(groups=%llu, all ranges OOB?)",
            sbspTagId, (unsigned long long)groups.size());
        if (airGrid) ZH_LBSP_FreeAirprobeGrid(airGrid);
        free(resourceData);
        return true;
    }

    auto* meshArr = (ZH_RuntimeDecoratorMesh*)calloc(finished.size(), sizeof(ZH_RuntimeDecoratorMesh));
    if (!meshArr) { if (airGrid) ZH_LBSP_FreeAirprobeGrid(airGrid); free(resourceData); return false; }

    int outIdx = 0;
    for (auto& A : finished) {
        if (A.pos.empty()) continue;
        uint32_t vc = (uint32_t)(A.pos.size() / 3);
        uint32_t ic = (uint32_t)A.idx.size();
        float* positions = (float*)malloc(A.pos.size() * sizeof(float));
        float* uvs       = (float*)malloc(A.uv.size()  * sizeof(float));
        float* normals   = (float*)malloc(A.nrm.size() * sizeof(float));
        uint16_t* idx16  = (uint16_t*)malloc(A.idx.size() * sizeof(uint16_t));
        if (!positions || !uvs || !normals || !idx16) {
            if (positions) free(positions); if (uvs) free(uvs);
            if (normals) free(normals); if (idx16) free(idx16);
            continue;
        }
        memcpy(positions, A.pos.data(), A.pos.size() * sizeof(float));
        memcpy(uvs,       A.uv.data(),  A.uv.size()  * sizeof(float));
        memcpy(normals,   A.nrm.data(), A.nrm.size() * sizeof(float));
        memcpy(idx16,     A.idx.data(), A.idx.size() * sizeof(uint16_t));

        // DECO_COLOR_RE: per-vertex baked ambient (NULL when gate off / no grid).
        float* colors = nullptr;
        if (emitDecoColor && A.col.size() == A.pos.size() && !A.col.empty()) {
            colors = (float*)malloc(A.col.size() * sizeof(float));
            if (colors) memcpy(colors, A.col.data(), A.col.size() * sizeof(float));
        }

        // DECO_WIND_RE: per-vertex world-space sway basis (always present on a
        // successful decode; size matches positions). NULL only on alloc failure.
        float* sway = nullptr;
        if (A.sway.size() == A.pos.size() && !A.sway.empty()) {
            sway = (float*)malloc(A.sway.size() * sizeof(float));
            if (sway) memcpy(sway, A.sway.data(), A.sway.size() * sizeof(float));
        }

        uint32_t s = A.setIdx;
        uint32_t bitmapTagId = 0xFFFFFFFFu, dctrId = 0xFFFFFFFFu;
        if (s < dctrBitmapIds.size() && dctrBitmapIds[s] >= 0)
            bitmapTagId = (uint32_t)dctrBitmapIds[s];
        if (s < dctrTagIds.size() && dctrTagIds[s] >= 0)
            dctrId = (uint32_t)dctrTagIds[s];

        // DECO_TYPE_RE: classify this set as ground-scatter (0) vs tall-woody (1)
        // from its template's local-Z height. kDecoTallZ=0.5 sits cleanly between
        // the byte-verified bush height (~0.57) and the tallest ground scatter
        // (ground_cover ~0.45) - trees (z~10) and bushes land in class 1, grass/
        // flower/ground_cover/rock in class 0. Fixed-card fallback (no template,
        // heightZ=0) classifies as ground scatter (the safe low-cutoff default).
        constexpr float kDecoTallZ = 0.5f;
        float setHeightZ = (s < setTemplates.size() && setTemplates[s].valid)
                               ? setTemplates[s].heightZ : 0.0f;
        uint32_t decoType = (setHeightZ >= kDecoTallZ) ? 1u : 0u;

        ZH_RuntimeDecoratorMesh& m = meshArr[outIdx++];
        m.Positions   = positions;
        m.UVs         = uvs;
        m.Normals     = normals;
        m.Indices     = idx16;
        m.VertexCount = vc;
        m.IndexCount  = ic;
        m.BitmapTagId = bitmapTagId;
        m.DctrTagId   = dctrId;
        m.Colors      = colors;   // NULL unless DECO_COLOR gate produced a grid sample
        m.Sway        = sway;     // DECO_WIND_RE per-vertex world-space sway basis
        m.DecoType    = decoType; // DECO_TYPE_RE: 0=ground scatter, 1=tall woody (tree/bush)
        m.InstanceCount = A.instCount; // DECO_BUDGET_RE: instances in this group-mesh

        // Free THIS chunk's SOURCE vectors now that everything is copied into the malloc'd
        // out-arrays. Without this, `finished` holds EVERY chunk's geometry simultaneously with the
        // full set of malloc'd copies -> a ~2x peak on dense-foliage maps (Forge World ~= +3.5 GB).
        // Freeing as we iterate halves the decorator peak; output bytes are unchanged.
        std::vector<float>().swap(A.pos);
        std::vector<float>().swap(A.uv);
        std::vector<float>().swap(A.nrm);
        std::vector<float>().swap(A.col);
        std::vector<float>().swap(A.sway);
        std::vector<uint16_t>().swap(A.idx);
    }

    if (airGrid) ZH_LBSP_FreeAirprobeGrid(airGrid);
    free(resourceData);
    *outMeshes = meshArr;
    *outMeshCount = (uint32_t)outIdx;
    NativeDiag("RuntimeDecGeom sbsp=0x%X: DONE instanced meshes=%d sets=%u "
               "decoColorGate=%d airprobes=%u seatZ=[%.2f..%.2f] inst=%llu "
               "(authored seat = engine surface placement; MMS adds NO runtime snap)",
        sbspTagId, outIdx, setCount, decoColorOn ? 1 : 0, airGridCount,
        (seatCount ? seatZMin : 0.0f), (seatCount ? seatZMax : 0.0f),
        (unsigned long long)seatCount);
    if (g_decoPosDiag && seatCount) {
        NativeDiag("[DECO_POS] sbsp=0x%X GLOBAL decoded-pos AABB X[%.1f..%.1f] Y[%.1f..%.1f] Z[%.1f..%.1f] "
                   "(compare to BSP world bounds - instances far outside = wrong bMin/bSize)",
            sbspTagId, gpMin[0], gpMax[0], gpMin[1], gpMax[1], gpMin[2], gpMax[2]);
        NativeDiag("[DECO_DUP] sbsp=0x%X spots=%llu multiSetPos=%llu (different decorator types stacked at one spot) "
                   "dupSameSet=%llu (same type repeated at one spot)",
            sbspTagId, (unsigned long long)posSetMask.size(),
            (unsigned long long)multiSetPos, (unsigned long long)dupSameSet);
    }
    return true;
}

bool SehGetRuntimeDecoratorGeometry(
    CacheHandle* cache, uint64_t cacheHandleId, uint32_t sbspTagId,
    ZH_RuntimeDecoratorMesh** outMeshes, uint32_t* outMeshCount)
{
    __try { return GetRuntimeDecoratorGeometryInner(cache, cacheHandleId, sbspTagId, outMeshes, outMeshCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        NativeDiag("RuntimeDecGeom sbsp=0x%X: SEH fault", sbspTagId);
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

// Verified runtime decorator offsets (panopticon scan):
//   sbsp+0x274 = runtime_decorator_sets[]  (12-byte block descriptor - 
//                tag-refs to dctr; first dword at the resolved memory is
//                literally 'rtcd' little-endian, the dctr class code).
//   sbsp+0x280 = decorator_instance_buffer (assumed; 4-byte shifted from
//                the RE-notes +0x284, mirrors the same -4 shift the sets
//                block uses).
//
// This is a -4 schema shift from the RE notes' nominal +0x278/+0x284,
// matching the same legacy-vs-U13 4-byte drift the scnr layout has
// between OFF_STRUCTURE_BSPS=80 (U13) / 76 (legacy) and
// OFF_SCENARIO_LIGHTMAP_REF=1808 (U13) / 1856 (legacy).

static void LogRuntimeDecoratorDescriptors(CacheHandle* cache, uint32_t sbspTagId) {
    __try {
        if (sbspTagId >= cache->tags.size()) return;
        const TagEntry& te = cache->tags[sbspTagId];
        if (memcmp(te.classCode, "sbsp", 4) != 0) return;
        int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
        if (metaOff < 0) return;
        if ((size_t)metaOff + 0x500 > cache->size) return;
        const uint8_t* sbsp = cache->base + metaOff;

        TagBlockRef setsBlk = ReadTagBlock(sbsp + kSbspRuntimeDecoratorSetsOff);
        if (setsBlk.count <= 0 || setsBlk.count > 64) {
            zh_mcc::NativeDiag(
                "RuntimeDec sbsp=0x%X +0x%X: count=%d ptr=0x%X (no decorators or insane)",
                sbspTagId, kSbspRuntimeDecoratorSetsOff, setsBlk.count, setsBlk.pointer);
            return;
        }
        int64_t setsOff = TagMetaFileOff(cache, setsBlk.pointer);
        if (setsOff < 0) {
            zh_mcc::NativeDiag(
                "RuntimeDec sbsp=0x%X +0x%X: count=%d ptr=0x%X (ptr unresolvable)",
                sbspTagId, kSbspRuntimeDecoratorSetsOff, setsBlk.count, setsBlk.pointer);
            return;
        }
        // 16 bytes per runtime_decorator_set_block (it's just a tag-ref).
        if ((size_t)setsOff + (size_t)setsBlk.count * 16 > cache->size) return;

        // Walk first up-to-8 sets and resolve each dctr tag-ref so we can
        // confirm the chain works end-to-end.
        char buf[1024]; size_t used = 0;
        int dumpN = setsBlk.count > 8 ? 8 : setsBlk.count;
        for (int i = 0; i < dumpN; ++i) {
            const uint8_t* s = cache->base + setsOff + (size_t)i * 16;
            uint32_t tagIdRaw = RU32(s + 12);
            int32_t dctrTagId = (tagIdRaw == 0xFFFFFFFFu) ? -1 : (int32_t)(tagIdRaw & 0xFFFFu);
            const char* dctrName = "?";
            if (dctrTagId >= 0 && (uint32_t)dctrTagId < cache->tags.size() &&
                memcmp(cache->tags[dctrTagId].classCode, "dctr", 4) == 0) {
                dctrName = cache->tags[dctrTagId].tagName.c_str();
                const char* shortName = dctrName;
                for (const char* p = dctrName; *p; ++p)
                    if (*p == '\\' || *p == '/') shortName = p + 1;
                dctrName = shortName;
            }
            int n = _snprintf_s(buf + used, sizeof(buf) - used, _TRUNCATE,
                "%s[%d]=0x%X(%s)", used == 0 ? "" : " ", i,
                (uint32_t)dctrTagId, dctrName);
            if (n > 0) used += (size_t)n;
        }
        zh_mcc::NativeDiag(
            "RuntimeDec sbsp=0x%X sets count=%d (showing %d): %s",
            sbspTagId, setsBlk.count, dumpN, buf);

        // Quick peek at decorator_instance_buffer - first 16 bytes of the
        // inline render-geometry struct.
        const uint8_t* ibuf = sbsp + kSbspDecoratorInstanceBufOff;
        zh_mcc::NativeDiag(
            "RuntimeDec sbsp=0x%X instBuf@+0x%X: %02X %02X %02X %02X %02X %02X %02X %02X | "
            "%02X %02X %02X %02X %02X %02X %02X %02X",
            sbspTagId, kSbspDecoratorInstanceBufOff,
            ibuf[0],  ibuf[1],  ibuf[2],  ibuf[3],
            ibuf[4],  ibuf[5],  ibuf[6],  ibuf[7],
            ibuf[8],  ibuf[9],  ibuf[10], ibuf[11],
            ibuf[12], ibuf[13], ibuf[14], ibuf[15]);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        zh_mcc::NativeDiag("RuntimeDec sbsp=0x%X SEH fault during diag", sbspTagId);
    }
}

extern "C" __declspec(dllexport) ZH_BspHandle __stdcall ZH_BSP_OpenBsp(
    uint64_t cacheHandle, uint32_t sbspTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;

    auto* bsp = new (std::nothrow) BspData();
    if (!bsp) return 0;
    bsp->cacheHandle = cacheHandle;
    bsp->sbspTagId   = sbspTagId;
    bsp->resourceIndex = -1;
    bsp->lbspResourceIndex = -1;

    LogRuntimeDecoratorDescriptors(cache, sbspTagId);

    bool ok = SehParseSbspTag(cache, sbspTagId, *bsp);
    if (!ok) {
        if (bsp->resourceData) free(bsp->resourceData);
        delete bsp;
        return 0;
    }

    uint64_t handle = g_nextBspHandle.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_bspHandlesMutex);
        g_bspHandles[handle] = bsp;
    }
    return handle;
}

extern "C" __declspec(dllexport) void __stdcall ZH_BSP_CloseBsp(ZH_BspHandle h)
{
    BspData* bsp = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_bspHandlesMutex);
        auto it = g_bspHandles.find(h);
        if (it == g_bspHandles.end()) return;
        bsp = it->second;
        g_bspHandles.erase(it);
    }
    if (!bsp) return;
    if (bsp->resourceData) free(bsp->resourceData);
    delete bsp;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMeshCount(ZH_BspHandle h)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0;
    return (uint32_t)bsp->meshes.size();
}

extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_GetMesh(
    ZH_BspHandle h, uint32_t i, ZH_BspMesh* outMesh)
{
    if (!outMesh) return false;
    memset(outMesh, 0, sizeof(*outMesh));
    BspData* bsp = LookupBsp(h);
    if (!bsp || i >= bsp->meshes.size()) return false;
    const BspMesh& m = bsp->meshes[i];
    if (m.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[m.sectionIndex];

    // Diagnostic: log the first few mesh queries so we can see what vertex
    // format / counts the viewer is being told. Helps diagnose
    // "BSP parses fine but adapter rejects everything" (e.g. all meshes have
    // vfmt=0xFF or vertexCount=0 -> adapter's gate / decode gate trips).
    static std::atomic<int> s_meshDiagBudget{ 6 };
    int v = s_meshDiagBudget.load(std::memory_order_relaxed);
    while (v > 0) {
        if (s_meshDiagBudget.compare_exchange_weak(v, v - 1,
            std::memory_order_relaxed, std::memory_order_relaxed))
        {
            zh_mcc::NativeDiag(
                "Bsp mesh[%u]: secIdx=%u vfmt=0x%02x vc=%u ic=%u matIdx=%d isInst=%d posBounds=(%g..%g,%g..%g,%g..%g)",
                i, m.sectionIndex, sec.vertexFormat, sec.vertexCount, sec.indexCount,
                m.materialIndex, m.isInstance ? 1 : 0,
                m.posMin[0], m.posMax[0], m.posMin[1], m.posMax[1], m.posMin[2], m.posMax[2]);
            break;
        }
    }

    outMesh->VertexCount  = sec.vertexCount;
    outMesh->IndexCount   = sec.indexCount;
    outMesh->VertexStride = 12;
    outMesh->IndexStride  = (sec.vertexCount > 0xFFFF) ? 4u : 2u;
    outMesh->VertexFormat = sec.vertexFormat;
    outMesh->Flags        = m.isInstance ? 1u : 0u;
    outMesh->MaterialIndex = m.materialIndex;
    memcpy(outMesh->BoundsMin, m.posMin, sizeof(m.posMin));
    memcpy(outMesh->BoundsMax, m.posMax, sizeof(m.posMax));
    memcpy(outMesh->UvMin,     m.uvMin,  sizeof(m.uvMin));
    memcpy(outMesh->UvMax,     m.uvMax,  sizeof(m.uvMax));
    memcpy(outMesh->AppliedTransform, m.transform, sizeof(m.transform));
    outMesh->InstanceOrdinal = m.instanceOrdinal;
    // Surface the Lbsp `Meshes` (a.k.a. `Sections`) ordinal. The Lbsp
    // Meshes block at 0x7C is parallel to the BSP parser's `data.sections[]`
    // - sectionIndex IS the Lbsp mesh ordinal.
    //
    // (Carried in the Reserved1 slot; the Rust mirror reads it as
    // `lbsp_mesh_ordinal`.)
    outMesh->Reserved1       = m.sectionIndex;
    // Owning SBSP cluster index (parallel to Lbsp.clusters[]); sentinel for
    // instances. The viewer indexes Lbsp.clusters[] by this so the per-cluster
    // lightmap slice (baked shadows) is sampled.
    outMesh->LightmapClusterIndex = m.isInstance ? 0xFFFFFFFFu : m.lightmapClusterIndex;
    return true;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshGeometry(
    ZH_BspHandle h, uint32_t meshIndex,
    uint8_t** outVertexBytes, uint32_t* outVertexBytesLen,
    uint8_t** outIndexBytes,  uint32_t* outIndexBytesLen)
{
    if (!outVertexBytes || !outVertexBytesLen || !outIndexBytes || !outIndexBytesLen)
        return false;
    *outVertexBytes = nullptr;
    *outVertexBytesLen = 0;
    *outIndexBytes = nullptr;
    *outIndexBytesLen = 0;

    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehDecodeMeshGeometry(bsp, meshIndex,
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

// DIRECT-VB: hand back the section's PREBUILT compressed VB/IB VERBATIM (pointers into
// the resource mmap) + de-quant constants + this submesh's index window, so the viewer
// can upload straight to the GPU and decompress in the vertex shader. PURE ADDITIVE:
// returns existing data by reference, allocates/mutates nothing. See
// reference_hms_direct_vb_pipeline_re.md.
extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_GetRawGeometry(
    ZH_BspHandle h, uint32_t meshIndex, ZH_RawGeom* outGeom)
{
    if (!outGeom) return false;
    memset(outGeom, 0, sizeof(*outGeom));
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    if (meshIndex >= bsp->meshes.size()) return false;
    const BspMesh& mesh = bsp->meshes[meshIndex];
    if (mesh.sectionIndex >= bsp->sections.size()) return false;
    const BspSection& sec = bsp->sections[mesh.sectionIndex];
    if (!IsFormatSupported(sec.vertexFormat)) return false;
    if (sec.vertexCount == 0) return false;

    // Vertex buffer - verbatim slice of the resource mmap.
    if ((size_t)sec.vbResourceOffset + (size_t)sec.vbDataLength > bsp->resourceSize) return false;
    uint32_t stride = StrideForFormat(sec.vertexFormat);
    if ((size_t)sec.vertexCount * stride > sec.vbDataLength) return false;
    outGeom->VbPtr = bsp->resourceData + sec.vbResourceOffset;
    outGeom->VbLen = sec.vbDataLength;
    outGeom->VertexFormat = sec.vertexFormat;
    outGeom->VertexStride = stride;
    outGeom->SectionVertexCount = sec.vertexCount;

    // Index buffer - verbatim, plus this submesh's window into it. Reach BSP indices are
    // 16-bit; indexFormat!=0 flags 32-bit. NOTE: the raw IB may be triangle STRIPS with
    // restart sentinels - the viewer decides list-vs-strip topology (positions decode is
    // independent of this, so the first verification slice ignores it).
    uint32_t indexStride = (sec.indexFormat != 0) ? 4u : 2u;
    outGeom->IndexStride = indexStride;
    outGeom->IsUnindexed = sec.isUnindexed ? 1u : 0u;
    if (!sec.isUnindexed) {
        if ((size_t)sec.ibResourceOffset + (size_t)sec.ibDataLength > bsp->resourceSize) return false;
        outGeom->IbPtr = bsp->resourceData + sec.ibResourceOffset;
        outGeom->IbLen = sec.ibDataLength;
    }
    uint32_t smIndex = sec.submeshStart + mesh.submeshIndexInSec;
    if (smIndex < bsp->submeshes.size()) {
        const BspSubmesh& sm = bsp->submeshes[smIndex];
        outGeom->IndexStart = sm.indexStart;
        outGeom->IndexCount = sm.indexLength;
    }

    outGeom->IsInstance = mesh.isInstance ? 1u : 0u;
    memcpy(outGeom->PosMin, sec.posMin, sizeof(float) * 3);
    memcpy(outGeom->PosMax, sec.posMax, sizeof(float) * 3);
    memcpy(outGeom->UvMin,  sec.uvMin,  sizeof(float) * 2);
    memcpy(outGeom->UvMax,  sec.uvMax,  sizeof(float) * 2);
    memcpy(outGeom->Transform, mesh.transform, sizeof(float) * 16);
    outGeom->UniformScale = mesh.uniformScale;
    outGeom->LightmapClusterIndex = mesh.lightmapClusterIndex;
    outGeom->InstanceOrdinal = mesh.instanceOrdinal;
    return true;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshUVs(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outUvFloat2, uint32_t* outUvFloatCount)
{
    if (!outUvFloat2 || !outUvFloatCount) return false;
    *outUvFloat2 = nullptr;
    *outUvFloatCount = 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehDecodeMeshUVs(bsp, meshIndex, outUvFloat2, outUvFloatCount);
    if (!ok) {
        if (*outUvFloat2) { free(*outUvFloat2); *outUvFloat2 = nullptr; }
        *outUvFloatCount = 0;
    }
    return ok;
}

// Secondary lightmap UV (UV2). Same shape as ZH_BSP_DecodeMeshUVs but reads
// the +0x20 Float16x2 slot from the world / flat-world cluster vertex format.
// Returns false on rigid / skinned / decorator clusters (no UV2 stream),
// caller falls back to flat shading.
extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshUV2(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outUvFloat2, uint32_t* outUvFloatCount)
{
    if (!outUvFloat2 || !outUvFloatCount) return false;
    *outUvFloat2 = nullptr;
    *outUvFloatCount = 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehDecodeMeshUV2s(bsp, meshIndex, outUvFloat2, outUvFloatCount);
    if (!ok) {
        if (*outUvFloat2) { free(*outUvFloat2); *outUvFloat2 = nullptr; }
        *outUvFloatCount = 0;
    }
    return ok;
}

// =============================================================================
// ZH_BSP_GetInstancePvlVb -- the per-instance PER-VERTEX LIGHTPROBE vertex buffer.
//
// CORRECTED (reach_tag_test.exe struct definitions): the 2-byte block at
// Lbsp+0xF4 is `s_scenario_lightmap_pervertex_data_run_time { vertex buffer index }`
// -- per-vertex LIGHTING, not texcoords. An earlier version of this export decoded it
// as UInt16x2 UV2, which is wrong (the values look plausible because per-vertex
// lighting is smooth across a mesh, exactly like a UV chart).
//
// Why it matters: ZH_LBSP_GetInstancePvlVb (the normal per-instance PVL fetch) fails
// for most instances on some maps (830 of 1205 on Zealot), dropping that geometry to a
// flat airprobe ambient -- the "grey where it should be purple" bug. This block names
// the VB that holds their real baked per-vertex lighting.
//
// Returns the RAW buffer bytes (caller decodes with the existing PVL unpacker, which
// already knows the packed VMF-lobe layout), plus the VB's stride and vertex count.
// Free the buffer with ZH_BSP_Free.
// =============================================================================
bool ExtractInstanceUV2VbIndex(CacheHandle* cache,
                               uint32_t sbspTagId,
                               uint32_t instanceOrdinal,
                               int16_t* outVbIndex);

static bool GetInstancePvlVbImpl(BspData* bsp, uint32_t instanceOrdinal,
                                 uint8_t** outBytes, uint32_t* outLen,
                                 uint32_t* outStride, uint32_t* outCount)
{
    if (!bsp || !bsp->resourceData || bsp->resourceSize == 0) return false;

    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return false;

    int16_t vbIndex = -1;
    if (!ExtractInstanceUV2VbIndex(cache, bsp->sbspTagId, instanceOrdinal, &vbIndex)) return false;
    if (vbIndex < 0) return false;

    const size_t vi = (size_t)vbIndex;
    if (vi >= bsp->lbspVbFixupOffsets.size() || vi >= bsp->lbspVbCounts.size()
        || vi >= bsp->lbspVbStrides.size() || vi >= bsp->lbspVbLens.size())
        return false;

    const uint32_t off    = bsp->lbspVbFixupOffsets[vi];
    const uint32_t count  = bsp->lbspVbCounts[vi];
    const uint32_t stride = bsp->lbspVbStrides[vi];
    const uint32_t len    = bsp->lbspVbLens[vi];
    if (count == 0 || len == 0) return false;

    // Every byte handed back must sit inside the decoded resource page.
    if ((size_t)off > bsp->resourceSize) return false;
    if ((size_t)len > bsp->resourceSize - (size_t)off) return false;

    uint8_t* dst = (uint8_t*)malloc((size_t)len);
    if (!dst) return false;
    memcpy(dst, bsp->resourceData + off, (size_t)len);

    *outBytes  = dst;
    *outLen    = len;
    *outStride = stride;
    *outCount  = count;
    return true;
}

static bool SehGetInstancePvlVb(BspData* bsp, uint32_t instanceOrdinal,
                                uint8_t** outBytes, uint32_t* outLen,
                                uint32_t* outStride, uint32_t* outCount)
{
    __try {
        return GetInstancePvlVbImpl(bsp, instanceOrdinal, outBytes, outLen, outStride, outCount);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_GetInstancePvlVb(
    ZH_BspHandle h, uint32_t instanceOrdinal,
    uint8_t** outBytes, uint32_t* outLen, uint32_t* outStride, uint32_t* outCount)
{
    if (!outBytes || !outLen || !outStride || !outCount) return false;
    *outBytes = nullptr; *outLen = 0; *outStride = 0; *outCount = 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehGetInstancePvlVb(bsp, instanceOrdinal, outBytes, outLen, outStride, outCount);
    if (!ok) {
        if (*outBytes) { free(*outBytes); *outBytes = nullptr; }
        *outLen = 0; *outStride = 0; *outCount = 0;
    }
    return ok;
}

// Per-vertex normals (float3[vc]). Same shape as ZH_BSP_DecodeMeshUVs but
// returns float3 per vertex instead of float2. Returns false on decorator
// meshes that lack packed normal data, or on OOB/format errors.
extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshNormals(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outNormals, uint32_t* outNormalFloatCount)
{
    if (!outNormals || !outNormalFloatCount) return false;
    *outNormals = nullptr;
    *outNormalFloatCount = 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehDecodeMeshNormals(bsp, meshIndex, outNormals, outNormalFloatCount);
    if (!ok) {
        if (*outNormals) { free(*outNormals); *outNormals = nullptr; }
        *outNormalFloatCount = 0;
    }
    return ok;
}

// Per-vertex tangents (float3[vc]). Same shape as normals.
// Returns false on decorator meshes or on error.
extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshTangents(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outTangents, uint32_t* outTangentFloatCount)
{
    if (!outTangents || !outTangentFloatCount) return false;
    *outTangents = nullptr;
    *outTangentFloatCount = 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehDecodeMeshTangents(bsp, meshIndex, outTangents, outTangentFloatCount);
    if (!ok) {
        if (*outTangents) { free(*outTangents); *outTangents = nullptr; }
        *outTangentFloatCount = 0;
    }
    return ok;
}

// Per-vertex binormals (float3[vc]). Computed as cross(normal, tangent) * sign
// where sign comes from the tangent's W component (binormal handedness).
// Returns false on decorator meshes or on error.
extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshBinormals(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outBinormals, uint32_t* outBinormalFloatCount)
{
    if (!outBinormals || !outBinormalFloatCount) return false;
    *outBinormals = nullptr;
    *outBinormalFloatCount = 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    bool ok = SehDecodeMeshBinormals(bsp, meshIndex, outBinormals, outBinormalFloatCount);
    if (!ok) {
        if (*outBinormals) { free(*outBinormals); *outBinormals = nullptr; }
        *outBinormalFloatCount = 0;
    }
    return ok;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMaterialDiffuseBitmapTagId(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFFFFFFFFu;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size())
        return 0xFFFFFFFFu;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFFFFFFFu;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    return SehResolveDiffuse(cache, shaderTagId);
}

// #204: classify a material's shader so the renderer can pick the right fallback
// when the diffuse resolves to "no bitmap". Some Reach BSP surfaces intentionally
// use textureless shaders - `shaders\invalid` (broken/placeholder, all slots are
// engine-internal LUT/default bitmaps) and `...\simple\black` (renders pure black,
// used for occluder shells, vent interiors, backfaces). Painting these with the
// default mid-gray fallback makes them read as pale, flat, see-through panels
// (Countdown material 50 = the 39 gantry/shell meshes). Returns:
//   0 = normal / has a real diffuse or unknown shader
//   1 = black    (name contains "\black")
//   2 = invalid  (name contains "invalid")
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMaterialShaderKind(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0u;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size())
        return 0u;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0u;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0u;
    if ((uint32_t)shaderTagId >= cache->tags.size()) return 0u;
    const char* name = cache->tags[shaderTagId].tagName.c_str();
    if (!name || !*name) return 0u;
    if (strstr(name, "invalid") != nullptr) return 2u;
    // Match "\black" (or "/black") so simple\black and any *_black variant classify.
    size_t len = strlen(name);
    if (len >= 6) {
        for (size_t i = 0; i + 6 <= len; ++i) {
            if ((name[i] == '\\' || name[i] == '/') &&
                name[i+1] == 'b' && name[i+2] == 'l' && name[i+3] == 'a' &&
                name[i+4] == 'c' && name[i+5] == 'k' &&
                (name[i+6] == '\0')) {
                return 1u;
            }
        }
    }
    return 0u;
}

// #285: the BSP material's shader tag CLASS packed as 4 little-endian bytes (e.g. 'r','m','g','l'
// -> 0x6C676D72), 0 when unresolved. Mirrors ZH_MMP_GetShaderClass for the render-model path.
// Lets the Rust side route glass (rmgl) BSP panes by shader class instead of the brittle
// blend-mode / diffuse-name heuristics - rmgl glass has no blend_mode category so the blend
// resolver returns the 0xFF sentinel and the name prong only fires for *glass* bitmaps.
// The render_method (rmsh/rmgl/...) TAG ID of a BSP material (0xFFFFFFFF when unresolved). Lets the
// caller walk the shader's postprocess block (rmt2 template name -> option digits) via ZH_TAG_ReadMeta.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMaterialShaderTagId(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFFFFFFFFu;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFFFFFFFFu;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFFFFFFFu;
    return (uint32_t)shaderTagId;
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetShaderClass(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0u;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0u;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0u;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0u;
    if ((uint32_t)shaderTagId >= cache->tags.size()) return 0u;
    const char* c = cache->tags[shaderTagId].classCode;
    return (uint32_t)(uint8_t)c[0]
         | ((uint32_t)(uint8_t)c[1] << 8)
         | ((uint32_t)(uint8_t)c[2] << 16)
         | ((uint32_t)(uint8_t)c[3] << 24);
}

extern "C" __declspec(dllexport) uint8_t __stdcall ZH_BSP_GetMaterialBlendMode(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFF;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFF;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveBlendMode(cache, shaderTagId);
}

// MAT-1: per-material material_model (0..9) for a BSP material index; 0xFF unresolved.
// Mirrors ZH_BSP_GetMaterialBlendMode. Rust maps 0xFF -> default 1 (cook_torrance).
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_BSP_GetMaterialModel(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFF;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFF;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveMaterialModel(cache, shaderTagId);
}

// ALBEDO-VARIANT: per-material albedo option (0 default / 1 two_detail / 2 black_point /
// 3 overlay / 4 detail_blend / 5 three_detail / 6 color_mask / 7 constant_color); 0xFF
// unresolved. Mirrors ZH_BSP_GetMaterialModel. Rust maps 0xFF -> 0 (default single-detail).
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_BSP_GetMaterialAlbedoOption(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFF;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFF;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveAlbedoOption(cache, shaderTagId);
}

// LIT-SI-3: per-material self_illumination MODE (0..12) for a BSP material index;
// 0xFF unresolved. Mirrors ZH_BSP_GetMaterialModel. See ResolveShaderSelfIllumMode
// for the enum. Rust maps 0xFF -> 1 (simple = current single-composite path).
extern "C" __declspec(dllexport) uint8_t __stdcall ZH_BSP_GetSelfIllumMode(
    ZH_BspHandle h, int32_t materialIndex)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFF;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFF;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFF;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFF;
    return SehResolveSelfIllumMode(cache, shaderTagId);
}

// Per-material diffuse-slot UV tiling. Walks rmsh -> rmt2.Arguments[] to
// find the index of the picked diffuse usage, reads
// rmsh.ShaderProperties[0].TilingData[argIdx].XY. Falls back to (1.0, 1.0)
// on any failure (older DLL won't export this; older callers see no scale).
//
// Returns true iff TilingData was found and looked sane (positive, finite,
// not absurdly huge); outTileX / outTileY always written. Caller can choose
// to skip multiplying when both are 1.0 to avoid wasted work.
extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_GetMaterialDiffuseTiling(
    ZH_BspHandle h, int32_t materialIndex,
    float* outTileX, float* outTileY)
{
    if (outTileX) *outTileX = 1.0f;
    if (outTileY) *outTileY = 1.0f;
    if (!outTileX || !outTileY) return false;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size())
        return false;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return false;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return false;
    float tx = 1.0f, ty = 1.0f;
    bool ok = SehResolveDiffuseTiling(cache, shaderTagId, materialIndex, tx, ty);
    *outTileX = tx;
    *outTileY = ty;
    return ok;
}

// The rmt2 shader-constants walk (Arguments[] real/scalar decode) is IMMUTABLE per
// shader tag but was re-run on EVERY call - decode_bsp_mesh calls this ~7x per mesh, so a
// terrain/interior BSP with thousands of meshes spent 100s+ re-walking the same shaders.
// Memoize the resolved constants per (cache, shaderTagId). Post-load-immutable -> safe to
// share across the parallel decode; the mutex only guards the small hash lookup/insert.
static std::shared_mutex g_shaderConstCacheMutex;
static std::unordered_map<uint64_t, std::pair<bool, ZH_ShaderConstantsImpl>> g_shaderConstCache;

extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_GetMaterialShaderConstants(
    ZH_BspHandle h, int32_t materialIndex, ZH_ShaderConstants* outConsts)
{
    if (!outConsts) return false;
    static_assert(sizeof(ZH_ShaderConstants) == sizeof(ZH_ShaderConstantsImpl),
                  "ZH_ShaderConstants struct size mismatch with internal impl");
    memset(outConsts, 0, sizeof(*outConsts));

    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return false;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return false;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return false;

    uint64_t key = ((uint64_t)(uintptr_t)cache << 20) ^ (uint32_t)shaderTagId;
    {
        std::shared_lock<std::shared_mutex> lk(g_shaderConstCacheMutex);
        auto it = g_shaderConstCache.find(key);
        if (it != g_shaderConstCache.end()) {
            memcpy(outConsts, &it->second.second, sizeof(*outConsts));
            return it->second.first;
        }
    }
    ZH_ShaderConstantsImpl tmp;
    memset(&tmp, 0, sizeof(tmp));
    bool ok = SehResolveShaderConstants(cache, shaderTagId, materialIndex, &tmp);
    {
        std::unique_lock<std::shared_mutex> lk(g_shaderConstCacheMutex);
        g_shaderConstCache[key] = std::make_pair(ok, tmp);
    }
    memcpy(outConsts, &tmp, sizeof(*outConsts));
    return ok;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_BSP_GetMaterialTerrainLayers(
    ZH_BspHandle h, int32_t materialIndex, ZH_TerrainLayers* outLayers)
{
    if (!outLayers) return false;
    memset(outLayers, 0, sizeof(*outLayers));
    outLayers->BaseMap_M_0 = 0xFFFFFFFFu;
    outLayers->BaseMap_M_1 = 0xFFFFFFFFu;
    outLayers->BaseMap_M_2 = 0xFFFFFFFFu;
    outLayers->BaseMap_M_3 = 0xFFFFFFFFu;
    outLayers->BumpMap_M_0 = 0xFFFFFFFFu;
    outLayers->BumpMap_M_1 = 0xFFFFFFFFu;
    outLayers->BumpMap_M_2 = 0xFFFFFFFFu;
    outLayers->BumpMap_M_3 = 0xFFFFFFFFu;
    outLayers->DetailMap_M_0 = 0xFFFFFFFFu;
    outLayers->DetailMap_M_1 = 0xFFFFFFFFu;
    outLayers->DetailMap_M_2 = 0xFFFFFFFFu;
    outLayers->DetailMap_M_3 = 0xFFFFFFFFu;
    outLayers->DetailBumpMap_M_0 = 0xFFFFFFFFu;
    outLayers->DetailBumpMap_M_1 = 0xFFFFFFFFu;
    outLayers->DetailBumpMap_M_2 = 0xFFFFFFFFu;
    outLayers->DetailBumpMap_M_3 = 0xFFFFFFFFu;
    outLayers->DetailTile_M_0_X = 1.0f; outLayers->DetailTile_M_0_Y = 1.0f;
    outLayers->DetailTile_M_1_X = 1.0f; outLayers->DetailTile_M_1_Y = 1.0f;
    outLayers->DetailTile_M_2_X = 1.0f; outLayers->DetailTile_M_2_Y = 1.0f;
    outLayers->DetailTile_M_3_X = 1.0f; outLayers->DetailTile_M_3_Y = 1.0f;
    outLayers->BaseTile_M_0_X = 1.0f; outLayers->BaseTile_M_0_Y = 1.0f;
    outLayers->BaseTile_M_1_X = 1.0f; outLayers->BaseTile_M_1_Y = 1.0f;
    outLayers->BaseTile_M_2_X = 1.0f; outLayers->BaseTile_M_2_Y = 1.0f;
    outLayers->BaseTile_M_3_X = 1.0f; outLayers->BaseTile_M_3_Y = 1.0f;
    outLayers->BumpTile_M_0_X = 1.0f; outLayers->BumpTile_M_0_Y = 1.0f;
    outLayers->BumpTile_M_1_X = 1.0f; outLayers->BumpTile_M_1_Y = 1.0f;
    outLayers->BumpTile_M_2_X = 1.0f; outLayers->BumpTile_M_2_Y = 1.0f;
    outLayers->BumpTile_M_3_X = 1.0f; outLayers->BumpTile_M_3_Y = 1.0f;
    outLayers->DetailBumpTile_M_0_X = 1.0f; outLayers->DetailBumpTile_M_0_Y = 1.0f;
    outLayers->DetailBumpTile_M_1_X = 1.0f; outLayers->DetailBumpTile_M_1_Y = 1.0f;
    outLayers->DetailBumpTile_M_2_X = 1.0f; outLayers->DetailBumpTile_M_2_Y = 1.0f;
    outLayers->DetailBumpTile_M_3_X = 1.0f; outLayers->DetailBumpTile_M_3_Y = 1.0f;
    outLayers->GlobalAlbedoTint[0] = 1.0f;
    outLayers->GlobalAlbedoTint[1] = 1.0f;
    outLayers->GlobalAlbedoTint[2] = 1.0f;
    outLayers->GlobalAlbedoTint[3] = 1.0f;
    // TERRAIN_BLEND_FIX: engine-identity defaults for the appended
    // fields so an early return (no BSP / non-terrain shader) yields today's
    // exact behaviour. memset already zeroed the .zw offsets + AuthoredActiveMask;
    // only the blend xform SCALE needs the explicit (1,1) identity.
    outLayers->BlendXform_X = 1.0f; outLayers->BlendXform_Y = 1.0f;
    outLayers->BlendXform_Z = 0.0f; outLayers->BlendXform_W = 0.0f;
    outLayers->BlendType = 0xFFFFFFFFu;   // MAT-15: unresolved => morph (no-op)
    outLayers->BlendMap    = 0xFFFFFFFFu;
    outLayers->IsTerrainBlend = 0;

    BspData* bsp = LookupBsp(h);
    if (!bsp) return false;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return false;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return false;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return false;

    TerrainLayersResolved resolved{};
    bool ok = SehResolveTerrainLayers(cache, shaderTagId, resolved);
    outLayers->BaseMap_M_0 = resolved.base_m[0];
    outLayers->BaseMap_M_1 = resolved.base_m[1];
    outLayers->BaseMap_M_2 = resolved.base_m[2];
    outLayers->BaseMap_M_3 = resolved.base_m[3];
    outLayers->BumpMap_M_0 = resolved.bump_m[0];
    outLayers->BumpMap_M_1 = resolved.bump_m[1];
    outLayers->BumpMap_M_2 = resolved.bump_m[2];
    outLayers->BumpMap_M_3 = resolved.bump_m[3];
    outLayers->DetailMap_M_0 = resolved.detail_m[0];
    outLayers->DetailMap_M_1 = resolved.detail_m[1];
    outLayers->DetailMap_M_2 = resolved.detail_m[2];
    outLayers->DetailMap_M_3 = resolved.detail_m[3];
    outLayers->DetailTile_M_0_X = resolved.detail_tile[0][0];
    outLayers->DetailTile_M_0_Y = resolved.detail_tile[0][1];
    outLayers->DetailTile_M_1_X = resolved.detail_tile[1][0];
    outLayers->DetailTile_M_1_Y = resolved.detail_tile[1][1];
    outLayers->DetailTile_M_2_X = resolved.detail_tile[2][0];
    outLayers->DetailTile_M_2_Y = resolved.detail_tile[2][1];
    outLayers->DetailTile_M_3_X = resolved.detail_tile[3][0];
    outLayers->DetailTile_M_3_Y = resolved.detail_tile[3][1];
    outLayers->BaseTile_M_0_X = resolved.base_tile[0][0];
    outLayers->BaseTile_M_0_Y = resolved.base_tile[0][1];
    outLayers->BaseTile_M_1_X = resolved.base_tile[1][0];
    outLayers->BaseTile_M_1_Y = resolved.base_tile[1][1];
    outLayers->BaseTile_M_2_X = resolved.base_tile[2][0];
    outLayers->BaseTile_M_2_Y = resolved.base_tile[2][1];
    outLayers->BaseTile_M_3_X = resolved.base_tile[3][0];
    outLayers->BaseTile_M_3_Y = resolved.base_tile[3][1];
    outLayers->GlobalAlbedoTint[0] = resolved.global_albedo_tint[0];
    outLayers->GlobalAlbedoTint[1] = resolved.global_albedo_tint[1];
    outLayers->GlobalAlbedoTint[2] = resolved.global_albedo_tint[2];
    outLayers->GlobalAlbedoTint[3] = resolved.global_albedo_tint[3];
    outLayers->BumpTile_M_0_X = resolved.bump_tile[0][0];
    outLayers->BumpTile_M_0_Y = resolved.bump_tile[0][1];
    outLayers->BumpTile_M_1_X = resolved.bump_tile[1][0];
    outLayers->BumpTile_M_1_Y = resolved.bump_tile[1][1];
    outLayers->BumpTile_M_2_X = resolved.bump_tile[2][0];
    outLayers->BumpTile_M_2_Y = resolved.bump_tile[2][1];
    outLayers->BumpTile_M_3_X = resolved.bump_tile[3][0];
    outLayers->BumpTile_M_3_Y = resolved.bump_tile[3][1];
    outLayers->DetailBumpMap_M_0 = resolved.detail_bump_m[0];
    outLayers->DetailBumpMap_M_1 = resolved.detail_bump_m[1];
    outLayers->DetailBumpMap_M_2 = resolved.detail_bump_m[2];
    outLayers->DetailBumpMap_M_3 = resolved.detail_bump_m[3];
    outLayers->DetailBumpTile_M_0_X = resolved.detail_bump_tile[0][0];
    outLayers->DetailBumpTile_M_0_Y = resolved.detail_bump_tile[0][1];
    outLayers->DetailBumpTile_M_1_X = resolved.detail_bump_tile[1][0];
    outLayers->DetailBumpTile_M_1_Y = resolved.detail_bump_tile[1][1];
    outLayers->DetailBumpTile_M_2_X = resolved.detail_bump_tile[2][0];
    outLayers->DetailBumpTile_M_2_Y = resolved.detail_bump_tile[2][1];
    outLayers->DetailBumpTile_M_3_X = resolved.detail_bump_tile[3][0];
    outLayers->DetailBumpTile_M_3_Y = resolved.detail_bump_tile[3][1];
    // TERRAIN_BLEND_FIX: carry the blend xform (Fix 1), per-layer .zw
    // offsets (Fix 3) and the authored active mask (Fix 2) out to the interop.
    outLayers->BlendXform_X = resolved.blend_xform[0];
    outLayers->BlendXform_Y = resolved.blend_xform[1];
    outLayers->BlendXform_Z = resolved.blend_xform[2];
    outLayers->BlendXform_W = resolved.blend_xform[3];
    outLayers->BaseOffset_M_0_X = resolved.base_offset[0][0]; outLayers->BaseOffset_M_0_Y = resolved.base_offset[0][1];
    outLayers->BaseOffset_M_1_X = resolved.base_offset[1][0]; outLayers->BaseOffset_M_1_Y = resolved.base_offset[1][1];
    outLayers->BaseOffset_M_2_X = resolved.base_offset[2][0]; outLayers->BaseOffset_M_2_Y = resolved.base_offset[2][1];
    outLayers->BaseOffset_M_3_X = resolved.base_offset[3][0]; outLayers->BaseOffset_M_3_Y = resolved.base_offset[3][1];
    outLayers->DetailOffset_M_0_X = resolved.detail_offset[0][0]; outLayers->DetailOffset_M_0_Y = resolved.detail_offset[0][1];
    outLayers->DetailOffset_M_1_X = resolved.detail_offset[1][0]; outLayers->DetailOffset_M_1_Y = resolved.detail_offset[1][1];
    outLayers->DetailOffset_M_2_X = resolved.detail_offset[2][0]; outLayers->DetailOffset_M_2_Y = resolved.detail_offset[2][1];
    outLayers->DetailOffset_M_3_X = resolved.detail_offset[3][0]; outLayers->DetailOffset_M_3_Y = resolved.detail_offset[3][1];
    outLayers->BumpOffset_M_0_X = resolved.bump_offset[0][0]; outLayers->BumpOffset_M_0_Y = resolved.bump_offset[0][1];
    outLayers->BumpOffset_M_1_X = resolved.bump_offset[1][0]; outLayers->BumpOffset_M_1_Y = resolved.bump_offset[1][1];
    outLayers->BumpOffset_M_2_X = resolved.bump_offset[2][0]; outLayers->BumpOffset_M_2_Y = resolved.bump_offset[2][1];
    outLayers->BumpOffset_M_3_X = resolved.bump_offset[3][0]; outLayers->BumpOffset_M_3_Y = resolved.bump_offset[3][1];
    outLayers->DetailBumpOffset_M_0_X = resolved.detail_bump_offset[0][0]; outLayers->DetailBumpOffset_M_0_Y = resolved.detail_bump_offset[0][1];
    outLayers->DetailBumpOffset_M_1_X = resolved.detail_bump_offset[1][0]; outLayers->DetailBumpOffset_M_1_Y = resolved.detail_bump_offset[1][1];
    outLayers->DetailBumpOffset_M_2_X = resolved.detail_bump_offset[2][0]; outLayers->DetailBumpOffset_M_2_Y = resolved.detail_bump_offset[2][1];
    outLayers->DetailBumpOffset_M_3_X = resolved.detail_bump_offset[3][0]; outLayers->DetailBumpOffset_M_3_Y = resolved.detail_bump_offset[3][1];
    outLayers->AuthoredActiveMask = resolved.authored_active_mask;
    // MAT-15: distance_blend_base mode + params.
    outLayers->BlendType   = resolved.blend_type;
    outLayers->BlendSlope  = resolved.blend_slope;
    outLayers->BlendOffset = resolved.blend_offset;
    for (int n = 0; n < 4; ++n) {
        outLayers->BlendTarget[n][0] = resolved.blend_target[n][0];
        outLayers->BlendTarget[n][1] = resolved.blend_target[n][1];
        outLayers->BlendTarget[n][2] = resolved.blend_target[n][2];
        outLayers->BlendTarget[n][3] = resolved.blend_target[n][3];
        outLayers->BlendMax[n] = resolved.blend_max[n];
    }
    outLayers->BlendMap    = resolved.blend_map;
    outLayers->IsTerrainBlend = resolved.is_terrain_blend ? 1u : 0u;

    // One-line diag per material that resolves as terrain blend so the user
    // can verify against Reclaimer's "expected 5 textures" output. Bounded - 
    // we only log the first ~256 distinct materials. SEH-protected name reads.
    if (ok) {
        static std::atomic<int> s_diagBudget{ 256 };
        int v = s_diagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_diagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                auto NameOf = [&](uint32_t id) -> const char* {
                    if (id == 0xFFFFFFFFu) return "(none)";
                    if (id >= cache->tags.size()) return "(oob)";
                    const std::string& n = cache->tags[id].tagName;
                    return n.empty() ? "(unnamed)" : n.c_str();
                };
                __try {
                    zh_mcc::NativeDiag(
                        "BspTerrainBlend mat=%d shaderTag=0x%04x m0='%s' m1='%s' m2='%s' m3='%s' blend='%s'",
                        materialIndex, (unsigned)(shaderTagId & 0xFFFF),
                        NameOf(resolved.base_m[0]),
                        NameOf(resolved.base_m[1]),
                        NameOf(resolved.base_m[2]),
                        NameOf(resolved.base_m[3]),
                        NameOf(resolved.blend_map));
                } __except (EXCEPTION_EXECUTE_HANDLER) { }
                break;
            }
        }
    }

    return ok;
}

// W22/W23 investigation probe (HMS_UWPROBE, stderr). For a water material, walk
// rmsh->rmt2 and dump: rmt2.Arguments[] count + every resolved arg name/index, and
// whether the literal strings "underwater_fog_color"/"underwater_murkiness" occur
// anywhere in the cache string blob (i.e. are GLOBAL strings) or only as tag-local
// bytes. Tells us where the underwater WaterSharedPS globals live so W22/W23 can be
// wired to the right block. No-op unless HMS_UWPROBE is set. Never called in normal runs.
extern "C" __declspec(dllexport) void __stdcall ZH_BSP_UnderwaterProbe(
    ZH_BspHandle h, int32_t materialIndex)
{
    if (getenv("HMS_UWPROBE") == nullptr) return;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return;

    // Scan the global string blob for the two names (once).
    static int s_scanned = 0;
    if (!s_scanned) {
        s_scanned = 1;
        const char* blob = (const char*)cache->stringBlob.data();
        size_t blobSz = cache->stringBlob.size();
        auto blobHas = [&](const char* needle) -> bool {
            size_t nl = strlen(needle);
            if (blobSz < nl) return false;
            for (size_t i = 0; i + nl <= blobSz; ++i)
                if (memcmp(blob + i, needle, nl) == 0) return true;
            return false;
        };
        fprintf(stderr, "HMS_UWPROBE stringblob has underwater_fog_color=%d underwater_murkiness=%d (blobSz=%zu)\n",
                (int)blobHas("underwater_fog_color"), (int)blobHas("underwater_murkiness"), blobSz);
    }

    __try {
        const TagEntry& te = cache->tags[shaderTagId];
        if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return;
        int64_t rmshOff = TagMetaFileOff(cache, te.metaPointerRaw);
        if (rmshOff < 0) return;
        const uint8_t* rmsh = cache->base + rmshOff;
        TagBlockRef propsBlk = ReadTagBlock(rmsh + OFF_SHADER_PROPS);
        if (propsBlk.count <= 0) return;
        int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
        if (propsOff < 0) return;
        const uint8_t* props = cache->base + propsOff;
        int32_t rmtRawId = R32(props + 12);
        int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu) ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
        if (rmtTagId < 0) return;
        const TagEntry& rmtTe = cache->tags[rmtTagId];
        int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
        if (rmtOff < 0) return;
        const uint8_t* rmtMeta = cache->base + rmtOff;
        TagBlockRef argsBlk = ReadTagBlock(rmtMeta + FWD_OFF_RMT_ARGUMENTS);
        int64_t argsOff = TagMetaFileOff(cache, argsBlk.pointer);
        fprintf(stderr, "HMS_UWPROBE mat=%d rmt=0x%X argsCount=%d\n",
                materialIndex, (unsigned)rmtTagId, argsBlk.count);
        if (argsOff >= 0) {
            for (int i = 0; i < argsBlk.count && i < 64; ++i) {
                int32_t sid = R32(cache->base + argsOff + (size_t)i * STRINGID_BLOCK_SIZE);
                const char* nm = ResolveStringId(cache, sid);
                fprintf(stderr, "  arg[%d] sid=0x%08X name='%s'\n", i, (unsigned)sid, nm ? nm : "<null>");
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) { }
}

// =============================================================================
// FIX B (W22/W23): ZH_BSP_GetUnderwaterFog - the authored underwater fog color +
// murkiness from the atmosphere_globals ("atgf") tag's underwater_setting block.
//
// The engine's k_ps_water_underwater_fog_color / k_ps_water_underwater_murkiness
// (WaterSharedPS cbuffer) are filled at runtime from atmosphere_globals, NOT the
// per-material water rmt2 (the ZH_BSP_UnderwaterProbe confirmed the rmt2 Arguments
// carry only SURFACE params). The underwater_setting element (sizeof(s_underwater_setting),
// confirmed from reach_tag_test.exe tag defs + the .rdata field-name strings
// "Murkiness"@RVA 0x18401B0 / "Fog Color"@RVA 0x18401C0) is 0x14 bytes:
//   +0x00  Name        (string_id)       e.g. "water_physics00"
//   +0x04  Murkiness   (real_fraction)   [0..1]
//   +0x08  Fog Color   (real_rgb float3) linear, pre-exposure
// Empirically the underwater_setting_block {count,ptr} sits at atgf meta +0x024 on the
// campaign/ocean caches (35_island, m10/m20/m50/m70, 30_settlement all read
// murk=1.0 rgb=(0.067,0.094,0.094), Name="water_physics00"). Rather than hardcode the
// offset (block field offsets shift by tag version), scan the meta head for a block
// whose element[0] matches the underwater_setting SIGNATURE - a resolvable Name
// containing "water", murk in [0,1], rgb in [0,8] with a non-zero channel. If no such
// block is found the caller keeps its deep-colour/water_murkiness stand-in (this never
// returns a guessed colour). Returns 1 on success (out params written), else 0.
// =============================================================================
extern "C" __declspec(dllexport) int __stdcall ZH_BSP_GetUnderwaterFog(
    ZH_BspHandle h, float* outMurk, float* outR, float* outG, float* outB)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0;
    int found = 0;
    __try {
        int atgfIdx = FindGlobalTag(cache, "atgf");
        if (atgfIdx < 0) return 0;
        const TagEntry& ate = cache->tags[atgfIdx];
        int64_t mo = TagMetaFileOff(cache, ate.metaPointerRaw);
        if (mo < 0) return 0;
        const uint8_t* m = cache->base + mo;
        for (int off = 0x08; off + 8 <= 0x400 && !found; off += 4) {
            TagBlockRef b = ReadTagBlock(m + off);
            if (b.count <= 0 || b.count > 16) continue;
            int64_t eo = TagMetaFileOff(cache, b.pointer);
            if (eo < 0 || (size_t)eo + 0x14 > cache->size) continue;
            const uint8_t* e = cache->base + eo;
            int32_t nameSid; memcpy(&nameSid, e + 0x00, 4);
            const char* nm = ResolveStringId(cache, nameSid);
            if (!nm || !nm[0]) continue;
            size_t nl = strnlen(nm, 64);
            bool hasWater = false;
            for (size_t i = 0; i + 5 <= nl; ++i) {
                if ((nm[i]|0x20)=='w' && (nm[i+1]|0x20)=='a' && (nm[i+2]|0x20)=='t' &&
                    (nm[i+3]|0x20)=='e' && (nm[i+4]|0x20)=='r') { hasWater = true; break; }
            }
            if (!hasWater) continue;
            float murk, r, g, bl;
            memcpy(&murk, e + 0x04, 4); memcpy(&r, e + 0x08, 4);
            memcpy(&g, e + 0x0C, 4);   memcpy(&bl, e + 0x10, 4);
            if (!(murk >= 0.f && murk <= 1.0001f)) continue;
            if (!(r>=0.f&&r<=8.f && g>=0.f&&g<=8.f && bl>=0.f&&bl<=8.f)) continue;
            if (r + g + bl <= 0.0001f) continue;
            if (outMurk) *outMurk = murk;
            if (outR) *outR = r; if (outG) *outG = g; if (outB) *outB = bl;
            found = 1;
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
    return found;
}

// =============================================================================
// rmsh.Postprocess[0].Overlays - animation walker.
//
// Per Lord Zedd's ReachMCC plugin, each overlay element is 0x24 bytes:
//   +0x00  Type (enum32)             0=Value, 1=Color, 2=ScaleUniform,
//                                     3=ScaleX, 4=ScaleY, 5=TranslationX,
//                                     6=TranslationY, 7=FrameIndex, 8=Alpha
//   +0x04  Input Name (stringid)     which rmt2 argument this animates
//   +0x08  Range Name (stringid)     stage / driver name (often empty)
//   +0x0C  Time Period (float32)     animation cycle in seconds
//   +0x10  Function (dataRef, 16B)   curve bytes - we surface the raw data
//                                     pointer + size; viewer side can evaluate.
//
// The engine evaluates Function at (currentTime mod TimePeriod) -> output
// value, then offsets the named rmt2 argument's relevant component (X/Y/A/etc)
// by that value. shader_water typically has TranslationX/Y overlays on
// base_map and bump_map slots - driving the visible wave/foam scroll.
// =============================================================================

constexpr int OFF_OVERLAYS_IN_PROPS  = 0x5C;
constexpr int OVERLAY_BLOCK_SIZE     = 0x24;

struct ZH_MaterialOverlay {
    uint32_t Type;            // engine enum (0..8)
    int32_t  InputNameSid;    // stringid
    int32_t  RangeNameSid;
    float    TimePeriod;
    char     InputName[32];   // resolved string-id name
    uint32_t FunctionByteSize;// optional - 0 if Function dataref couldn't resolve
    uint32_t Reserved[2];
};

// Diag budget for overlay resolution - first N calls dump to native log.
static std::atomic<int> s_overlayDiagBudget{ 128 };

static int ResolveOverlaysInner(CacheHandle* cache, int32_t shaderTagId,
                                ZH_MaterialOverlay* outArr, int capacity)
{
    bool logThis = false;
    {
        int v = s_overlayDiagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_overlayDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            { logThis = true; break; }
        }
    }

    if (!cache || !outArr || capacity <= 0) return 0;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0 || te.classCode[0] != 'r' || te.classCode[1] != 'm') {
        if (logThis) NativeDiag("ResolveOverlays: tag 0x%X class '%c%c%c%c' not rm*",
            shaderTagId, te.classCode[0], te.classCode[1], te.classCode[2], te.classCode[3]);
        return 0;
    }

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size) return 0;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) {
        if (logThis) NativeDiag("ResolveOverlays: tag 0x%X propsBlk.count=%d (empty postprocess)",
            shaderTagId, propsBlk.count);
        return 0;
    }
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 || (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size) return 0;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef ovBlk = ReadTagBlock(props + OFF_OVERLAYS_IN_PROPS);
    if (logThis) NativeDiag("ResolveOverlays: tag 0x%X class='%c%c%c%c' props.count=%d ovBlk.count=%d ovBlk.ptr=0x%X",
        shaderTagId, te.classCode[0], te.classCode[1], te.classCode[2], te.classCode[3],
        propsBlk.count, ovBlk.count, ovBlk.pointer);
    if (ovBlk.count <= 0 || ovBlk.count > 0x100) return 0;
    int64_t ovOff = TagMetaFileOff(cache, ovBlk.pointer);
    if (ovOff < 0 ||
        (size_t)ovOff + (size_t)ovBlk.count * OVERLAY_BLOCK_SIZE > cache->size) return 0;

    int n = ovBlk.count > capacity ? capacity : ovBlk.count;
    for (int i = 0; i < n; ++i) {
        const uint8_t* e = cache->base + ovOff + (size_t)i * OVERLAY_BLOCK_SIZE;
        ZH_MaterialOverlay& o = outArr[i];
        memset(&o, 0, sizeof(o));
        o.Type         = (uint32_t)R32(e + 0x00);
        o.InputNameSid = R32(e + 0x04);
        o.RangeNameSid = R32(e + 0x08);
        memcpy(&o.TimePeriod, e + 0x0C, 4);
        // dataRef at +0x10 is 16 bytes: (count, raw_ptr, raw_ptr2, defOff_or_size).
        // The first 4 bytes are the byte count of the function's serialized
        // representation. We surface only the size for now - the viewer can
        // request the actual bytes through a separate export when we wire up
        // function-curve evaluation. (For first-pass animation, knowing the
        // Type + TimePeriod + InputName is enough to drive a linear scroll
        // at the right rate.)
        o.FunctionByteSize = (uint32_t)R32(e + 0x10);

        if (cache->stringTableParsed) {
            const char* nm = ResolveStringId(cache, o.InputNameSid);
            if (nm) {
                size_t L = strnlen(nm, 31);
                memcpy(o.InputName, nm, L);
                o.InputName[L] = 0;
            }
        }
    }
    if (logThis && n > 0) {
        for (int i = 0; i < n; ++i)
            NativeDiag("  overlay[%d] type=%u input='%s' period=%.3f",
                       i, outArr[i].Type, outArr[i].InputName, outArr[i].TimePeriod);
    }
    return n;
}

static int SehResolveOverlays(CacheHandle* cache, int32_t shaderTagId,
                              ZH_MaterialOverlay* outArr, int capacity)
{
    __try { return ResolveOverlaysInner(cache, shaderTagId, outArr, capacity); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}

extern "C" __declspec(dllexport) int32_t __stdcall ZH_BSP_GetMaterialOverlays(
    ZH_BspHandle h, int32_t materialIndex,
    ZH_MaterialOverlay* outArr, int32_t capacity)
{
    if (!outArr || capacity <= 0) return 0;
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0;
    return SehResolveOverlays(cache, shaderTagId, outArr, capacity);
}

// =============================================================================
// rmsh.Postprocess[0].Textures x rmt2.Usages - bitmap by usage-name lookup.
// Walks the parallel ShaderMaps + Usages arrays and returns the bitmap tag
// id for the FIRST slot whose usage stringid matches the given name. Used
// for shader_water to find the wave-displacement bitmap (which the existing
// generic-diffuse walker doesn't surface because it picks base_map/foam).
// =============================================================================

extern "C" void ZH_Logf(const char* fmt, ...);
static uint32_t ResolveBitmapByUsageInner(CacheHandle* cache, int32_t shaderTagId,
                                          const char* usageNeedle, uint8_t* outSampler = nullptr)
{
    if (outSampler) *outSampler = 0xFF;
    if (!cache || !usageNeedle || !*usageNeedle) return 0xFFFFFFFFu;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0xFFFFFFFFu;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0 || te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0xFFFFFFFFu;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + (OFF_SHADER_PROPS + 8) > cache->size) return 0xFFFFFFFFu;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS);
    if (propsBlk.count <= 0) return 0xFFFFFFFFu;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 || (size_t)propsOff + SHADER_PROPS_BLOCK_SIZE > cache->size) return 0xFFFFFFFFu;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef mapsBlk = ReadTagBlock(props + OFF_SHADER_MAPS_IN_PROPS);
    if (mapsBlk.count <= 0) return 0xFFFFFFFFu;
    int64_t mapsOff = TagMetaFileOff(cache, mapsBlk.pointer);
    if (mapsOff < 0 ||
        (size_t)mapsOff + (size_t)mapsBlk.count * SHADER_MAP_BLOCK_SIZE > cache->size)
        return 0xFFFFFFFFu;

    // Walk rmt2.Usages[] in parallel - slot i's usage name is Usages[i].
    int32_t rmtRawId = R32(props + 12);
    int32_t rmtTagId = ((uint32_t)rmtRawId == 0xFFFFFFFFu)
                       ? -1 : (int32_t)((uint32_t)rmtRawId & 0xFFFFu);
    if (rmtTagId < 0 || (uint32_t)rmtTagId >= cache->tags.size()) return 0xFFFFFFFFu;
    if (!cache->stringTableParsed) return 0xFFFFFFFFu;
    const TagEntry& rmtTe = cache->tags[rmtTagId];
    if (rmtTe.classIndex < 0 || rmtTe.classCode[0] != 'r' ||
        rmtTe.classCode[1] != 'm' || rmtTe.classCode[2] != 't') return 0xFFFFFFFFu;
    int64_t rmtOff = TagMetaFileOff(cache, rmtTe.metaPointerRaw);
    if (rmtOff < 0 || (size_t)rmtOff + OFF_RMT_USAGES + 8 > cache->size) return 0xFFFFFFFFu;
    const uint8_t* rmtMeta = cache->base + rmtOff;
    TagBlockRef usagesBlk = ReadTagBlock(rmtMeta + OFF_RMT_USAGES);
    if (usagesBlk.count <= 0 || usagesBlk.count > 0x1000) return 0xFFFFFFFFu;
    int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
    if (usagesOff < 0 ||
        (size_t)usagesOff + (size_t)usagesBlk.count * STRINGID_BLOCK_SIZE > cache->size)
        return 0xFFFFFFFFu;

    int slotMax = mapsBlk.count;
    if (slotMax > usagesBlk.count) slotMax = usagesBlk.count;
    for (int i = 0; i < slotMax; ++i) {
        int32_t sid = R32(cache->base + usagesOff + (size_t)i * STRINGID_BLOCK_SIZE);
        const char* usageName = ResolveStringId(cache, sid);
        {
            // MMS_USAGE_DIAG=1: log every rmt2 usage slot walked (native log) - debugging aid.
            static int s_usageDiag = -1;
            if (s_usageDiag < 0) { char dv[8]; s_usageDiag = GetEnvironmentVariableA("MMS_USAGE_DIAG", dv, sizeof(dv)) ? 1 : 0; }
            if (s_usageDiag == 1) {
                // + the raw 24-byte texture-constant record (bitmap ref, sampler state bytes).
                const uint8_t* me = cache->base + mapsOff + (size_t)i * SHADER_MAP_BLOCK_SIZE;
                fprintf(stderr, "[USAGE] shader=%d slot=%d/%d usages=%d name=%s needle=%s bytes=", shaderTagId, i, mapsBlk.count, usagesBlk.count, usageName ? usageName : "(null)", usageNeedle);
                for (int b = 0; b < SHADER_MAP_BLOCK_SIZE; ++b) fprintf(stderr, "%02x", me[b]);
                fprintf(stderr, "\n"); fflush(stderr);
            }
        }
        if (!usageName || !*usageName) continue;
        if (strcmp(usageName, usageNeedle) != 0) continue;
        const uint8_t* mapEntry = cache->base + mapsOff +
                                  (size_t)i * SHADER_MAP_BLOCK_SIZE;
        int32_t rawId = R32(mapEntry + 12);
        if ((uint32_t)rawId == 0xFFFFFFFFu) continue;
        uint32_t bmpId = (uint32_t)rawId & 0xFFFFu;
        if (bmpId >= cache->tags.size()) continue;
        if (memcmp(cache->tags[bmpId].classCode, "bitm", 4) != 0) continue;
        // Texture-constant byte 18 = sampler address mode nibbles (u low, v high:
        // 0 wrap, 1 clamp, 2 mirror, 3 black border) -- the value the engine binds this slot with.
        if (outSampler) *outSampler = mapEntry[18];
        return bmpId;
    }
    return 0xFFFFFFFFu;
}

// Sampler address-mode byte of the texture constant bound to `usageNeedle`
// (see ResolveBitmapByUsageInner); 0xFF when the usage is absent / unbound.
uint32_t SehResolveSamplerByUsage(CacheHandle* cache, int32_t shaderTagId, const char* usageNeedle)
{
    __try {
        uint8_t smp = 0xFF;
        uint32_t t = ResolveBitmapByUsageInner(cache, shaderTagId, usageNeedle, &smp);
        return (t == 0xFFFFFFFFu) ? 0xFFu : (uint32_t)smp;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFu; }
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMaterialSamplerByUsage(
    ZH_BspHandle h, int32_t materialIndex, const char* usageName)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFFu;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFFu;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFu;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFFu;
    return SehResolveSamplerByUsage(cache, shaderTagId, usageName);
}

// #285: NON-static so MapModelParser.cpp can reuse it for model-shader (rmhg/rmsh) bitmaps - 
// the resolver walks any render_method tag id's rmt2 usages, so a model shaderTagId works too.
uint32_t SehResolveBitmapByUsage(CacheHandle* cache, int32_t shaderTagId,
                                        const char* usageNeedle)
{
    __try { return ResolveBitmapByUsageInner(cache, shaderTagId, usageNeedle); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

// Same as shader-constants - the bitmap-by-usage rmt2 walk is immutable per
// (shaderTagId, usageName) but was re-run ~8x per mesh (detail_map/2/3, self_illum,
// alpha_test_map, wave arrays, watercolor). Memoize per (cache, shaderTagId, usageHash).
static std::shared_mutex g_bitmapUsageCacheMutex;
static std::unordered_map<uint64_t, uint32_t> g_bitmapUsageCache;

// Drop the per-shader material-walk caches when a cache closes. Keys fold the
// cache pointer into a hash so we can't erase selectively - but these hold only small POD
// per shader tag (a handful of KB total), so clearing wholesale on any close is fine and
// prevents stale (cache*, tag) collisions after a CacheHandle* address is reused.
extern "C" void PurgeBspMaterialCachesForCache(void* /*cachePtr*/)
{
    { std::unique_lock<std::shared_mutex> lk(g_shaderConstCacheMutex); g_shaderConstCache.clear(); }
    { std::unique_lock<std::shared_mutex> lk(g_bitmapUsageCacheMutex); g_bitmapUsageCache.clear(); }
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMaterialBitmapByUsage(
    ZH_BspHandle h, int32_t materialIndex, const char* usageName)
{
    BspData* bsp = LookupBsp(h);
    if (!bsp) return 0xFFFFFFFFu;
    if (materialIndex < 0 || (size_t)materialIndex >= bsp->shaders.size()) return 0xFFFFFFFFu;
    int32_t shaderTagId = bsp->shaders[materialIndex].shaderTagId;
    if (shaderTagId < 0) return 0xFFFFFFFFu;
    CacheHandle* cache = LookupHandle(bsp->cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    // FNV-1a hash of usageName folded with cache+shaderTagId for the cache key.
    uint64_t uh = 1469598103934665603ull;
    for (const char* p = usageName; p && *p; ++p) { uh ^= (uint8_t)*p; uh *= 1099511628211ull; }
    uint64_t key = (((uint64_t)(uintptr_t)cache << 20) ^ (uint32_t)shaderTagId) * 31 + uh;
    {
        std::shared_lock<std::shared_mutex> lk(g_bitmapUsageCacheMutex);
        auto it = g_bitmapUsageCache.find(key);
        if (it != g_bitmapUsageCache.end()) return it->second;
    }
    uint32_t r = SehResolveBitmapByUsage(cache, shaderTagId, usageName);
    {
        std::unique_lock<std::shared_mutex> lk(g_bitmapUsageCacheMutex);
        g_bitmapUsageCache[key] = r;
    }
    return r;
}

extern "C" __declspec(dllexport) void __stdcall ZH_BSP_FreeBuffer(uint8_t* buf)
{
    if (buf) free(buf);
}

// Public export: decode all VFMT_DECORATOR meshes from sbsp's
// decorator_instance_buffer. Returns 1 on success (count may be 0), 0 on
// hard failure. Caller frees via ZH_BSP_FreeRuntimeDecoratorMeshes.
extern "C" __declspec(dllexport) int __stdcall ZH_BSP_GetRuntimeDecoratorGeometry(
    uint64_t cacheHandle,
    uint32_t sbspTagId,
    ZH_RuntimeDecoratorMesh** outMeshes,
    uint32_t* outMeshCount)
{
    if (outMeshes) *outMeshes = nullptr;
    if (outMeshCount) *outMeshCount = 0;
    if (!outMeshes || !outMeshCount) return 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;

    bool ok = SehGetRuntimeDecoratorGeometry(cache, cacheHandle, sbspTagId, outMeshes, outMeshCount);
    if (!ok) {
        // Clean up any partial allocation
        if (*outMeshes) {
            for (uint32_t i = 0; i < *outMeshCount; ++i) {
                auto& m = (*outMeshes)[i];
                if (m.Positions) free(m.Positions);
                if (m.UVs) free(m.UVs);
                if (m.Normals) free(m.Normals);
                if (m.Indices) free(m.Indices);
                if (m.Colors) free(m.Colors);
                if (m.Sway) free(m.Sway);
            }
            free(*outMeshes);
            *outMeshes = nullptr;
        }
        *outMeshCount = 0;
        return 0;
    }
    return 1;
}

// Free the array returned by ZH_BSP_GetRuntimeDecoratorGeometry.
extern "C" __declspec(dllexport) void __stdcall ZH_BSP_FreeRuntimeDecoratorMeshes(
    ZH_RuntimeDecoratorMesh* meshes, uint32_t count)
{
    if (!meshes) return;
    for (uint32_t i = 0; i < count; ++i) {
        if (meshes[i].Positions) free(meshes[i].Positions);
        if (meshes[i].UVs)       free(meshes[i].UVs);
        if (meshes[i].Normals)   free(meshes[i].Normals);
        if (meshes[i].Indices)   free(meshes[i].Indices);
        if (meshes[i].Colors)    free(meshes[i].Colors);
        if (meshes[i].Sway)      free(meshes[i].Sway);
    }
    free(meshes);
}

// =============================================================================
// PREPLACED-DECAL BAKED GEOMETRY DECODE
// -----------------------------------------------------------------------------
// The preplaced-decal triple in the sbsp is:
//   SET  block @ sbsp+0x328  (world center + decs ref + ref range)
//   REF  block @ sbsp+0x334  (per-decal mesh slice: index/vertex start+count)
//   GEOM struct@ sbsp+0x340  (`preplaced decal geometry!*` =
//                             global_render_geometry_struct, inline)
//
// The GEOM struct is the SAME struct layout the decorator instance buffer uses
// (sbsp+0x280): Meshes tag_block @+0x00, BoundingBoxes tag_block @+0x0C, and the
// geometry resource handle @+0x94. The baked, artist-conformed frost/ice mesh
// that wraps whole walls/floors lives here - one flat VB/IB the engine slices
// per decal via the REF block (index_start/count, vertex_start/count).
//
// This export decodes that whole buffer ONCE into a flat interleaved vertex
// array {pos.x,pos.y,pos.z, u,v} (5 floats/vertex) + a u32 index buffer. Rust
// then slices per decal (REF ranges) and renders. Positions come back already in
// WORLD space (VFMT_WORLD sections are raw Float32; other formats are de-quant'd
// via the section bbox), so there is NO conform/projection step downstream.
//
// Index buffer is emitted VERBATIM (no strip expansion) so the REF index_start/
// count offsets stay 1:1 with the engine's layout; index VALUES are rebased to
// the concatenated (global) vertex array. Reach preplaced-decal geometry is a
// triangle LIST (index_count ~= 3xtriangles per the on-disk REF slices), so Rust
// consumes each REF slice as a tri-list.
// =============================================================================
constexpr int PPGEOM_STRUCT_OFFSET   = 0x340;  // inline global_render_geometry_struct
constexpr int PPGEOM_OFF_MESHES      = 0x00;   // tag_block (sections)
constexpr int PPGEOM_OFF_BBOXES      = 0x0C;   // tag_block (bounding boxes)
constexpr int PPGEOM_OFF_RESOURCE    = 0x94;   // geometry resource handle

static bool DecodePreplacedGeometryInner(
    CacheHandle* cache, uint32_t sbspTagId,
    float** outVerts, uint32_t* outVertFloatCount,
    uint32_t** outIndices, uint32_t* outIndexCount, uint32_t* outVertexCount,
    uint32_t** outMeshVertBase, uint32_t** outMeshIdxBase, uint32_t* outMeshCount)
{
    *outVerts = nullptr; *outVertFloatCount = 0;
    *outIndices = nullptr; *outIndexCount = 0; *outVertexCount = 0;
    *outMeshVertBase = nullptr; *outMeshIdxBase = nullptr; *outMeshCount = 0;

    if (sbspTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[sbspTagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, TC_SBSP, 4) != 0) return false;

    int64_t sbspMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (sbspMetaOff < 0) return false;
    if ((size_t)sbspMetaOff + PPGEOM_STRUCT_OFFSET + 0xA8 > cache->size) return false;
    const uint8_t* geom = cache->base + sbspMetaOff + PPGEOM_STRUCT_OFFSET;

    if (getenv("HMS_PPGEOM_DIAG")) {
        char hex[3 * 0xA8 + 16]; int p = 0;
        for (int b = 0; b < 0xA8; ++b) p += snprintf(hex + p, sizeof(hex) - (size_t)p, "%02X ", geom[b]);
        NativeDiag("[PPGEOM] sbsp=0x%X geomHdr[0x00..0xA8]= %s", sbspTagId, hex);
    }

    // The inline global_render_geometry_struct uses 12-byte tag_block headers
    // (int32 count @+0, int32 unk @+4, uint32 pointer @+8) - the MCC loaded-
    // metadata BlockCollection stride the decorator instance buffer also uses
    // (kDecInstBuf_Meshes=0x00, kDecInstBuf_BoundingBoxes=0x0C: 12 apart). The
    // shared ReadTagBlock reads an 8-byte header (pointer @+4) which lands on the
    // `unk` field for these nested structs, so read the pointer at +8 here.
    auto Read12Block = [](const uint8_t* p) -> TagBlockRef {
        return TagBlockRef{ R32(p), RU32(p + 8) };
    };
    TagBlockRef meshesBlk = Read12Block(geom + PPGEOM_OFF_MESHES);
    TagBlockRef bboxBlk   = Read12Block(geom + PPGEOM_OFF_BBOXES);
    int32_t resourceIdRaw = R32(geom + PPGEOM_OFF_RESOURCE);
    NativeDiag("[PPGEOM] sbsp=0x%X meshes=%d(ptr=0x%x) bboxes=%d(ptr=0x%x) resRaw=0x%X",
        sbspTagId, meshesBlk.count, meshesBlk.pointer, bboxBlk.count, bboxBlk.pointer,
        (uint32_t)resourceIdRaw);
    if (getenv("HMS_PPGEOM_DIAG")) {
        int64_t mo = TagMetaFileOff(cache, meshesBlk.pointer);
        NativeDiag("[PPGEOM] sbsp=0x%X meshesMetaOff=%lld", sbspTagId, (long long)mo);
        if (mo >= 0 && (size_t)mo + 92 <= cache->size) {
            const uint8_t* s = cache->base + mo;
            char hex[3*92+16]; int p=0;
            for (int b=0;b<92;++b) p+=snprintf(hex+p,sizeof(hex)-(size_t)p,"%02X ",s[b]);
            NativeDiag("[PPGEOM] sbsp=0x%X sec[0] rec92= %s", sbspTagId, hex);
        }
    }

    if (meshesBlk.count <= 0 || meshesBlk.count > 4096) return true;   // valid: no preplaced geometry
    if (resourceIdRaw == 0 || resourceIdRaw == -1) return true;

    // Gestalt + resource fixups + payload.
    if (!ParseGestalt(cache)) return false;
    int32_t resourceIndex = resourceIdRaw & 0xFFFF;
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size()) return false;
    {
        std::lock_guard<std::mutex> lk(cache->parseMutex);
        if (!EnsureResourceFixups(cache, (size_t)resourceIndex)) return false;
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];

    std::vector<uint32_t> vbCounts, vbLens, vbStrides, ibLens;
    std::vector<uint8_t>  ibFmts;
    if (!ParseFixupRegion(cache, entry, vbCounts, vbLens, vbStrides, ibFmts, ibLens)) {
        NativeDiag("[PPGEOM] sbsp=0x%X ParseFixupRegion failed", sbspTagId);
        return false;
    }

    if (getenv("HMS_PPGEOM_DIAG")) {
        NativeDiag("[PPGEOM] sbsp=0x%X fixupRegion vbCount=%llu ibCount=%llu",
            sbspTagId, (unsigned long long)vbCounts.size(), (unsigned long long)ibLens.size());
        for (size_t i = 0; i < vbCounts.size() && i < 16; ++i)
            NativeDiag("[PPGEOM]   VB[%llu] vc=%u len=%u stride=%u foff=0x%x",
                (unsigned long long)i, vbCounts[i], vbLens[i], vbStrides[i],
                i < entry.fixups.size() ? (uint32_t)(entry.fixups[i].offset & 0x0FFFFFFF) : 0);
        for (size_t i = 0; i < ibLens.size() && i < 16; ++i) {
            size_t fi = vbCounts.size() * 2 + i;
            NativeDiag("[PPGEOM]   IB[%llu] len=%u fmt=%u foff=0x%x",
                (unsigned long long)i, ibLens[i], ibFmts[i],
                fi < entry.fixups.size() ? (uint32_t)(entry.fixups[fi].offset & 0x0FFFFFFF) : 0);
        }
    }

    constexpr size_t kMaxRead = 64 * 1024 * 1024;
    size_t resourceSize = 0;
    uint8_t* resourceData = ReadResourceData(cache, resourceIdRaw, kMaxRead, &resourceSize);
    if (!resourceData) { NativeDiag("[PPGEOM] sbsp=0x%X ReadResourceData failed", sbspTagId); return false; }

    // The preplaced-decal geometry buffer is NOT a section-driven mesh - the
    // meshes tag_block resolves to garbage. Instead it is a set of independent
    // VB/IB meshes in the geometry resource (one per decal-material group). Each
    // decal's REF row selects a mesh via `definition block index` and slices it
    // with (vertex_start,vertex_count,index_start,index_count) LOCAL to that
    // mesh. So we decode EVERY VB/IB pair (VB[m] with IB[m]) into one flat global
    // array, and hand back per-mesh base offsets so the renderer can locate a
    // decal's mesh, then apply the REF slice within it.
    //
    // All preplaced VBs are stride-36 VFMT_WORLD (raw Float32 world positions +
    // Float16 UV) - verified on cex_prisoner (every VB stride==36). Index buffers
    // are tri-list-equivalent (REF index_count ~= 3xtriangles). Index VALUES are
    // kept RAW (mesh-local); the caller resolves them against the mesh vertex
    // base (+ vertex_start if the engine's IB is per-decal-relative).
    (void)meshesBlk; (void)bboxBlk;

    size_t nMesh = vbCounts.size();
    if (ibLens.size() < nMesh) nMesh = ibLens.size();
    if (nMesh == 0) { free(resourceData); return true; }

    std::vector<float>    globalVerts;   // 5 floats/vertex (pos.xyz, u, v)
    std::vector<uint32_t> globalIndices; // raw (mesh-local) index values
    std::vector<uint32_t> meshVertBase(nMesh, 0);
    std::vector<uint32_t> meshIdxBase(nMesh, 0);

    const uint32_t WORLD_STRIDE = 36;
    for (size_t m = 0; m < nMesh; ++m) {
        meshVertBase[m] = (uint32_t)(globalVerts.size() / 5);
        meshIdxBase[m]  = (uint32_t)globalIndices.size();

        uint32_t vc     = vbCounts[m];
        uint32_t stride = vbStrides[m];
        uint32_t vbOff  = (m < entry.fixups.size()) ? (uint32_t)(entry.fixups[m].offset & 0x0FFFFFFF) : 0;

        // Only stride-36 world VBs are preplaced-decal geometry. Non-36 strides
        // (should not occur here) are recorded as empty meshes to keep the mesh
        // index alignment with the decals' definition-block index.
        if (stride == WORLD_STRIDE && vc > 0 &&
            (size_t)vbOff + (size_t)vc * stride <= resourceSize)
        {
            const uint8_t* vbSrc = resourceData + vbOff;
            std::vector<float> pos((size_t)vc * 3);
            std::vector<float> uv((size_t)vc * 2);
            const float idMin[3] = { 0,0,0 }, idMax[3] = { 1,1,1 };
            DecodeBspPositions(vbSrc, vc, stride, VFMT_WORLD, idMin, idMax,
                               reinterpret_cast<uint8_t*>(pos.data()));
            DecodeBspUVs(vbSrc, vc, stride, VFMT_WORLD, idMin, idMax, uv.data());
            globalVerts.reserve(globalVerts.size() + (size_t)vc * 5);
            for (uint32_t v = 0; v < vc; ++v) {
                globalVerts.push_back(pos[v*3+0]);
                globalVerts.push_back(pos[v*3+1]);
                globalVerts.push_back(pos[v*3+2]);
                globalVerts.push_back(uv[v*2+0]);
                globalVerts.push_back(uv[v*2+1]);
            }

            uint32_t ibLen = ibLens[m];
            uint32_t indexStride = (vc > 0xFFFF) ? 4u : 2u;
            size_t ibFixupIdx = vbCounts.size() * 2 + m;
            uint32_t ibOff = (ibFixupIdx < entry.fixups.size())
                ? (uint32_t)(entry.fixups[ibFixupIdx].offset & 0x0FFFFFFF) : 0;
            uint32_t rawIdxCount = ibLen / indexStride;
            if ((size_t)ibOff + (size_t)ibLen <= resourceSize && rawIdxCount > 0) {
                const uint8_t* ibSrc = resourceData + ibOff;
                globalIndices.reserve(globalIndices.size() + rawIdxCount);
                if (indexStride == 2)
                    for (uint32_t k = 0; k < rawIdxCount; ++k) globalIndices.push_back((uint32_t)RU16(ibSrc + (size_t)k*2));
                else
                    for (uint32_t k = 0; k < rawIdxCount; ++k) globalIndices.push_back(RU32(ibSrc + (size_t)k*4));
            }

            NativeDiag("[PPGEOM] sbsp=0x%X mesh[%llu] vc=%u vbOff=%u ibLen=%u vbase=%u ibase=%u first=(%.2f,%.2f,%.2f)",
                sbspTagId, (unsigned long long)m, vc, vbOff, ibLen, meshVertBase[m], meshIdxBase[m],
                pos[0], pos[1], pos[2]);
        } else {
            NativeDiag("[PPGEOM] sbsp=0x%X mesh[%llu] SKIP vc=%u stride=%u vbOff=%u (non-world / OOB)",
                sbspTagId, (unsigned long long)m, vc, stride, vbOff);
        }
    }

    free(resourceData);

    uint32_t vcount = (uint32_t)(globalVerts.size() / 5);
    if (vcount == 0 || globalIndices.empty()) {
        NativeDiag("[PPGEOM] sbsp=0x%X decoded EMPTY (vc=%u ic=%llu)",
            sbspTagId, vcount, (unsigned long long)globalIndices.size());
        return true;
    }

    float*    vbuf = (float*)malloc(globalVerts.size() * sizeof(float));
    uint32_t* ibuf = (uint32_t*)malloc(globalIndices.size() * sizeof(uint32_t));
    uint32_t* mvb  = (uint32_t*)malloc(nMesh * sizeof(uint32_t));
    uint32_t* mib  = (uint32_t*)malloc(nMesh * sizeof(uint32_t));
    if (!vbuf || !ibuf || !mvb || !mib) {
        if (vbuf) free(vbuf); if (ibuf) free(ibuf); if (mvb) free(mvb); if (mib) free(mib);
        return false;
    }
    memcpy(vbuf, globalVerts.data(), globalVerts.size() * sizeof(float));
    memcpy(ibuf, globalIndices.data(), globalIndices.size() * sizeof(uint32_t));
    memcpy(mvb, meshVertBase.data(), nMesh * sizeof(uint32_t));
    memcpy(mib, meshIdxBase.data(), nMesh * sizeof(uint32_t));

    *outVerts          = vbuf;
    *outVertFloatCount = (uint32_t)globalVerts.size();
    *outIndices        = ibuf;
    *outIndexCount     = (uint32_t)globalIndices.size();
    *outVertexCount    = vcount;
    *outMeshVertBase   = mvb;
    *outMeshIdxBase    = mib;
    *outMeshCount      = (uint32_t)nMesh;

    NativeDiag("[PPGEOM] sbsp=0x%X DECODED verts=%u indices=%u meshes=%llu",
        sbspTagId, vcount, *outIndexCount, (unsigned long long)nMesh);
    return true;
}

// Decode the sbsp's preplaced-decal baked geometry into a flat interleaved
// vertex array {pos.xyz, uv} (5 floats/vertex) + a u32 index buffer. The
// per-decal slices come from HaloMapStudio_BSP_EnumeratePreplacedDecals (REF
// index/vertex start+count). Returns 1 on success (counts may be 0 = no
// preplaced geometry), 0 on hard failure. Caller frees via
// HaloMapStudio_BSP_FreePreplacedGeometry.
extern "C" __declspec(dllexport) int __stdcall HaloMapStudio_BSP_DecodePreplacedGeometry(
    uint64_t cacheHandle, uint32_t sbspTagId,
    float** outVerts, uint32_t* outVertFloatCount,
    uint32_t** outIndices, uint32_t* outIndexCount, uint32_t* outVertexCount,
    uint32_t** outMeshVertBase, uint32_t** outMeshIdxBase, uint32_t* outMeshCount)
{
    if (outVerts) *outVerts = nullptr;
    if (outVertFloatCount) *outVertFloatCount = 0;
    if (outIndices) *outIndices = nullptr;
    if (outIndexCount) *outIndexCount = 0;
    if (outVertexCount) *outVertexCount = 0;
    if (outMeshVertBase) *outMeshVertBase = nullptr;
    if (outMeshIdxBase) *outMeshIdxBase = nullptr;
    if (outMeshCount) *outMeshCount = 0;
    if (!outVerts || !outVertFloatCount || !outIndices || !outIndexCount || !outVertexCount ||
        !outMeshVertBase || !outMeshIdxBase || !outMeshCount)
        return 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;

    __try {
        return DecodePreplacedGeometryInner(cache, sbspTagId, outVerts, outVertFloatCount,
                                            outIndices, outIndexCount, outVertexCount,
                                            outMeshVertBase, outMeshIdxBase, outMeshCount) ? 1 : 0;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        NativeDiag("[PPGEOM] SEH fault sbsp=0x%X", sbspTagId);
        if (*outVerts)        { free(*outVerts);        *outVerts = nullptr; }
        if (*outIndices)      { free(*outIndices);      *outIndices = nullptr; }
        if (*outMeshVertBase) { free(*outMeshVertBase); *outMeshVertBase = nullptr; }
        if (*outMeshIdxBase)  { free(*outMeshIdxBase);  *outMeshIdxBase = nullptr; }
        *outVertFloatCount = 0; *outIndexCount = 0; *outVertexCount = 0; *outMeshCount = 0;
        return 0;
    }
}

extern "C" __declspec(dllexport) void __stdcall HaloMapStudio_BSP_FreePreplacedGeometry(
    float* verts, uint32_t* indices, uint32_t* meshVertBase, uint32_t* meshIdxBase)
{
    if (verts)        free(verts);
    if (indices)      free(indices);
    if (meshVertBase) free(meshVertBase);
    if (meshIdxBase)  free(meshIdxBase);
}

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_BSP_EnumerateSbspsInScenario(
    uint64_t cacheHandle, uint32_t scnrTagId,
    uint32_t* outSbspTagIds, uint32_t maxCount)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    return SehEnumerateSbsps(cache, scnrTagId, outSbspTagIds, maxCount);
}

// Drain all open BSP handles (called from ZH_MMP_PrepareUnload). Used by the
// hot-reload path so the viewer can FreeLibrary cleanly.
extern "C" void MapBspParser_DrainAllHandles()
{
    std::vector<BspData*> doomed;
    {
        std::lock_guard<std::mutex> lk(g_bspHandlesMutex);
        doomed.reserve(g_bspHandles.size());
        for (auto& kv : g_bspHandles) doomed.push_back(kv.second);
        g_bspHandles.clear();
    }
    for (auto* bsp : doomed) {
        if (!bsp) continue;
        if (bsp->resourceData) free(bsp->resourceData);
        delete bsp;
    }
}
