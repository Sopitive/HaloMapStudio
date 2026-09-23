// LightmapParser.cpp
// =============================================================================
// Native walker for the sbsp -> active scenario_lightmap (sLdT) -> active
// scenario_lightmap_bsp_data (Lbsp) -> per-cluster lightmap state chain in
// Halo MCC HaloReach .map files.
//
// What it exposes (all loaded by crates/hms-native):
//   * ZH_LBSP_GetActiveLbspTagId  - scnr -> sbsp -> sLdT -> active Lbsp.
//   * ZH_LBSP_GetClusterEntry     - per-cluster DM / SDM atlas bitmap tag ids +
//                                   submap indices (baked lightmap sampling).
//   * ZH_LBSP_GetInstanceAtlasSubmap / GetInstancePvl / GetInstanceProbe -
//                                   the three per-instance lighting tiers
//                                   (atlas, per-vertex, single probe).
//   * ZH_LBSP_GetClusterPvlVb     - per-cluster per-vertex lighting stream.
//   * ZH_LBSP_GetLightprobeAtlas  - the per-BSP lightprobe atlas.
//   * ZH_LBSP_GetAirprobeGrid     - the SH lighting-point grid (airprobes)
//                                   used for decorator / object ambient.
//   * ZH_LBSP_GetBrightness       - Lbsp brightness scale.
//
// Background: the runtime 0x148-byte cluster-state struct (TLS slot 0x578 via
// scenario_lightmap_get_struct_ptr, sapien.exe+0x47A2E0) is built at engine
// load from the Lbsp's RESOURCE PAGE ("scenario_cluster_data_resource",
// string @ sapien.exe+0x18F5C80); the on-disk Lbsp stores no pool indices.
// This parser decompresses the resource page (ReadResourceData), caches it on
// the CacheHandle and reads the per-cluster / per-instance data from it.
//
// Defensive contract:
//   * Every cross-tag deref is wrapped in __try / __except so a busted
//     scnr/sbsp/sLdT/Lbsp returns false rather than crashing the worker.
//   * Index validation against cache->tags.size() guards every TagId.
//   * Class-code checks confirm the tag at each link is what we expect.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <cmath>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "vmf_diffuse_lut.h"

using namespace zh_mcc;

// Engine g_sample_vmf_diffuse LUT lookup (byte-exact port of lightprobe.rs::vmf_diffuse_coeff):
// bilinear sample of the 64x64 A8 table at x = dot(dir,N)*0.5+0.5, y = bandwidth (identity map).
// This IS the vMF -> irradiance convolution - replaces the analytic half-Lambert approximation.
static float VmfDiffuseCoeff(float ndotd, float bandwidth)
{
    constexpr int N = 64;
    float fx = (ndotd * 0.5f + 0.5f);
    if (fx < 0.0f) fx = 0.0f; if (fx > 1.0f) fx = 1.0f;
    float fy = bandwidth;
    if (fy < 0.0f) fy = 0.0f; if (fy > 1.0f) fy = 1.0f; // texcoord clamps bandwidth>1 to sharpest row
    fx *= (N - 1); fy *= (N - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1 < N ? x0 + 1 : N - 1;
    int y1 = y0 + 1 < N ? y0 + 1 : N - 1;
    float tx = fx - x0, ty = fy - y0;
    auto s = [](int xi, int yi) { return kVmfDiffuseLut[yi * N + xi] / 255.0f; };
    float top = s(x0, y0) * (1.0f - tx) + s(x1, y0) * tx;
    float bot = s(x0, y1) * (1.0f - tx) + s(x1, y1) * tx;
    return top * (1.0f - ty) + bot * ty;
}

// =============================================================================
// Public ABI surface - must match the Rust mirror in crates/hms-native/src/lib.rs.
// =============================================================================

#pragma pack(push, 1)
struct ZH_LbspClusterEntry {
    uint32_t DmBitmapTagId;     // 0xFFFFFFFF = no DM atlas / not yet RE'd
    uint32_t SdmBitmapTagId;    // 0xFFFFFFFF = no SDM atlas / not yet RE'd
    uint32_t SubmapIndexDM;     // = lightprobe_texture_array_index (cluster +0x00 i16)
    uint32_t SubmapIndexSDM;    // = same i16 (engine binds DM+SDM with one index)
    // LBSP_TAG_LAYOUT_RE - schema-authoritative per-cluster
    // fields. Source: lbsp_reach.json `scenario_lightmap_cluster_data`
    // (sizeof 8, cross-checked in SAPIEN_LIGHTMAP_PHASE3_RE.md section 8.3).
    int32_t  PervertexBlockIndex;   // cluster +0x02 i16 (cross-ref into bsp_per_vertex_data)
    int32_t  PervertexBlockOffset;  // cluster +0x04 i32 (byte offset within block)
    // Per-cluster UV scale/bias for atlas sampling - identity (1,1,0,0) when
    // the parser cannot derive a real value. The original SAPIEN_LIGHTMAP_NOTES
    // claim that shader register 0x570001 carries a 4-component scale/bias was
    // disproven by haloreach.dll +0x258F3C disasm: 0x570001 is actually a
    // single scalar (vec4 with .w = per-cluster HDR scale, xy/zw = 0). See
    // LBSP_TAG_LAYOUT_RE.md "Per-cluster shader constant 0x570001". The
    // scale/bias slot is reserved here so a future per-cluster atlas-tile RE
    // pass can surface real values without an ABI break.
    float    UvScaleU;          // identity 1.0f
    float    UvScaleV;          // identity 1.0f
    float    UvBiasU;           // identity 0.0f
    float    UvBiasV;           // identity 0.0f
    // Per-cluster HDR scale (engine register 0x570001 .w). Default 1.0 when
    // no on-disk source is known; the engine derives this at resource-page
    // load and stores it at runtime state[i]+0x100. Multiplying it into the
    // atlas sample matches the engine's per-cluster light scaling.
    float    HdrScale;          // identity 1.0f
};
#pragma pack(pop)

// PHASE_B_RE - schema-authoritative per-cluster per-vertex
// lightprobe (PVL) descriptor. Returned by ZH_LBSP_GetClusterPvlVb.
//
// Routing chain (LBSP_TAG_LAYOUT_RE.md / LIGHTPROBE_TRUE_ENCODING_RE.md):
//   clusters[ci] (+0x054, stride 8): { i16 lpti, i16 pervertex_block_index,
//                                       i32 pervertex_block_offset }
//   bsp_per_vertex_run_time_data[pvb_index] (+0x048, stride 4):
//                                     { i16 vertex_buffer_index, i16 hdr_scale }
//   lbsp_vb_pool[vertex_buffer_index] = THE stride-4 PVL ByteAddressBuffer
//
// On success Found=1 and Bytes points at a malloc'd copy of the stride-4 VB
// (free via ZH_LBSP_FreeLightprobeBuffer). On "cluster carries no PVL" the
// export returns true with Found=0 (this is engine-normal: most clusters
// have pervertex_block_index == -1).
#pragma pack(push, 1)
struct ZH_LbspClusterPvlVb {
    uint32_t Found;        // 1 = PVL VB resolved; 0 = cluster has no PVL (normal)
    uint8_t* Bytes;        // malloc'd stride-4 ByteAddressBuffer contents (or null)
    uint32_t Len;          // length of Bytes in bytes (== ElemCount*4)
    int32_t  ElemCount;    // VB-pool declared element count (Len/4)
    int32_t  HdrScaleRaw;  // i16 hdr_scale from the +0x48 run-time table
    float    HdrScale;     // decoded float multiplier (see decode note below)
    int32_t  PvbIndex;     // clusters[ci].pervertex_block_index (i16, -1 = none)
    int32_t  PvbOffset;    // clusters[ci].pervertex_block_offset (i32)
    int32_t  VbIndex;      // resolved lbsp_vb_pool index (-1 if none)
};
#pragma pack(pop)

namespace {

// Tag class codes
constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_SBSP = "sbsp";
constexpr const char* TC_LTMP = "sLdT";   // scenario_lightmap - stored on disk as 'sLdT' fourcc, NOT 'ltmp'
constexpr const char* TC_LBSP = "Lbsp";   // scenario_lightmap_bsp_data
constexpr const char* TC_BITM = "bitm";

// scnr layout - mirrors PickScnrLayout in MapBspParser.cpp.
struct LpScnrLayout { int OFF_STRUCTURE_BSPS; int OFF_SCENARIO_LIGHTMAP_REF; };
LpScnrLayout PickScnr(CacheType ct) {
    LpScnrLayout L;
    switch (ct) {
        case CacheType::MccHaloReachU13:
            L.OFF_STRUCTURE_BSPS        = 80;
            L.OFF_SCENARIO_LIGHTMAP_REF = 1800;
            break;
        case CacheType::MccHaloReach:
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
        case CacheType::MccHaloReachU10:
        default:
            L.OFF_STRUCTURE_BSPS        = 76;
            L.OFF_SCENARIO_LIGHTMAP_REF = 1856;
            break;
    }
    return L;
}

// sbsp layout - only need clusters block for cluster-index bounds-check.
struct LpSbspLayout { int OFF_CLUSTERS; int CLUSTER_BLOCK_SIZE; };
LpSbspLayout PickSbsp(CacheType /*ct*/) {
    // MccHaloReach (release through U13) common defaults - Reclaimer
    // Cluster offsets agree across U10/U13.
    LpSbspLayout L;
    L.OFF_CLUSTERS       = 312;
    L.CLUSTER_BLOCK_SIZE = 140;
    return L;
}

// Lbsp layout (matches MapBspParser's LbspLayout, plus Phase-3 atlas offsets).
//
// Spec source: SAPIEN_LIGHTMAP_PHASE3_RE.md section 8.2 - verified against
// camden-smallwood/blam-tags' scenario_lightmap_bsp_data.json (Reach), sizeof
// = 0x168 = 360 bytes. Offsets here are relative to the Lbsp tag main body.
//
// Path A in section 8.4: the per-cluster atlas tuple at runtime
// (+0xF0/+0xF6/+0xF8/+0xFE) is synthesized by the engine from these on-disk
// fields:
//   light direction texture reference  -> DM pool @ +0xF0
//   light intensity texture reference  -> SDM pool @ +0xF8
//   clusters[i].lightprobe_texture_array_index -> DM and SDM submap @ +0xF6/+0xFE
struct LpLbspLayout {
    int OFF_SECTIONS;
    int OFF_RESOURCE_POINTER;
    int OFF_LIGHT_DIRECTION_TAGREF;   // 16B tag_reference  -> DM bitm
    int OFF_LIGHT_INTENSITY_TAGREF;   // 16B tag_reference  -> SDM bitm
    int OFF_CLUSTERS_BLOCK;           // 12B block descriptor (count, ptr, def)
    int CLUSTER_DATA_BLOCK_SIZE;      // 8B per-cluster entry
    int CLUSTER_OFFSET_LIGHTPROBE_INDEX; // i16 lightprobe_texture_array_index
};
LpLbspLayout PickLbsp(CacheType /*ct*/) {
    // Plugin (Lord Zedd ReachMCC) - verified offsets:
    //   0x18 float32   Brightness
    //   0x1C tagRef    Primary Map     (DM - light direction)
    //   0x2C tagRef    Intensity Map   (SDM - light intensity)
    //   0x48 tagblock  Unknown A       (VB-index pool, elementSize 0x4)
    //   0x54 tagblock  Unknown B       (elementSize 0x8) <- current "clusters" reads from here
    //   0x60 tagblock  Instanced Geometry (per-instance index record, 0xC)
    //   0x6C tagblock  Colors          (per-instance ambient, 0x24)
    //   0x7C tagblock  Meshes          (runtime mesh data, 0x5C)
    //   0xF4 tagblock  Per-Instance Lightmap Texcoords (VB indices, 0x2)
    //
    // NOTE: `OFF_CLUSTERS_BLOCK = 0x54` is a misnomer - the plugin
    // documents 0x54 as `Unknown B`, not a clusters block. The int16 we
    // read at +0x00 of each 8B entry happens to produce usable submap
    // values for some meshes (lbspHits=11/92 in settlement) but the
    // semantics are heuristic, not authoritative.
    LpLbspLayout L;
    L.OFF_SECTIONS                       = 124;
    L.OFF_RESOURCE_POINTER               = 268;
    L.OFF_LIGHT_DIRECTION_TAGREF         = 0x01C;   // 28
    L.OFF_LIGHT_INTENSITY_TAGREF         = 0x02C;   // 44
    L.OFF_CLUSTERS_BLOCK                 = 0x054;   // 84
    L.CLUSTER_DATA_BLOCK_SIZE            = 8;
    L.CLUSTER_OFFSET_LIGHTPROBE_INDEX    = 0x00;    // i16 at start of 8B entry
    return L;
}

// =============================================================================
// Layout diagnostic - dump every documented Lbsp tagblock's count + raw
// pointer the FIRST time we touch each Lbsp tag (ground truth for block
// sizes vs instance counts, e.g. "is 0xF4 parallel to instances or to
// Unknown A?").
// Output goes to HaloMapStudio_native.log via NativeDiag.
//
// The plugin XML names the visible blocks; we dump count + ptr + the first
// 16 raw bytes of each block so a quick hex check can verify the element
// layout matches the plugin.
// =============================================================================
// Forward-declare - LpReadTagRefId is defined further down in this file.
int32_t LpReadTagRefId(const uint8_t* tagRef);
constexpr int OFF_UNKNOWN_A          = 0x48;
constexpr int OFF_UNKNOWN_B          = 0x54;
constexpr int OFF_INSTANCED_GEOMETRY = 0x60;
constexpr int OFF_COLORS             = 0x6C;
constexpr int OFF_MESHES             = 0x7C;
constexpr int OFF_PERINSTANCE_LM_UV2 = 0xF4;
constexpr int OFF_AIRPROBES          = 0x12C;

static void Phase0DumpLbspLayout(CacheHandle* cache, uint32_t lbspTagId, int64_t lbspMetaOff)
{
    if (!cache || lbspTagId >= cache->tags.size()) return;
    if (lbspMetaOff < 0) return;
    const size_t kMaxRead = (size_t)OFF_AIRPROBES + 16;
    if ((size_t)lbspMetaOff + kMaxRead > cache->size) return;
    const uint8_t* m = cache->base + lbspMetaOff;

    auto dumpBlk = [&](const char* name, int off) {
        if ((size_t)lbspMetaOff + (size_t)off + 8 > cache->size) return;
        TagBlockRef blk = ReadTagBlock(m + off);
        int64_t dataOff = (blk.count > 0) ? TagMetaFileOff(cache, blk.pointer) : -1;
        char head[64] = "(empty)";
        if (dataOff > 0 && (size_t)dataOff + 16 <= cache->size) {
            const uint8_t* p = cache->base + dataOff;
            // First 16 bytes as hex - gives a peek at the element layout
            // without committing to an interpretation.
            snprintf(head, sizeof(head),
                "%02x %02x %02x %02x  %02x %02x %02x %02x  %02x %02x %02x %02x  %02x %02x %02x %02x",
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
        }
        NativeDiag("LbspLayout[lbsp=0x%04x] %-30s @0x%03x: count=%-5d ptr=0x%08x  head16=[%s]",
                   (unsigned)lbspTagId, name, (unsigned)off,
                   (int)blk.count, (unsigned)blk.pointer, head);
    };

    // Also dump the scalar/tagref fields so we can sanity-check our
    // existing readers against the plugin one final time.
    float brightness = 0;
    memcpy(&brightness, m + 0x18, 4);
    int32_t primaryId  = LpReadTagRefId(m + 0x1C);
    int32_t intensityId = LpReadTagRefId(m + 0x2C);
    NativeDiag("LbspLayout[lbsp=0x%04x] BASE brightness=%.3f primaryMap=0x%04x intensityMap=0x%04x",
               (unsigned)lbspTagId, brightness,
               (unsigned)(primaryId & 0xFFFF), (unsigned)(intensityId & 0xFFFF));

    // LBSP_TAG_LAYOUT_RE - schema-authoritative block names.
    // Source: lbsp_reach.json (sizeof-validated by compute_lbsp_offsets.py).
    // The two previous "Unknown" labels were placeholders from Lord Zedd's
    // plugin (which doesn't name runtime-only structures); the schema does
    // name them. See LBSP_TAG_LAYOUT_RE.md for the full layout table.
    dumpBlk("bsp_per_vertex_run_time_data (0x48)", OFF_UNKNOWN_A);
    dumpBlk("clusters (0x54)",                     OFF_UNKNOWN_B);
    dumpBlk("instances (0x60)",                    OFF_INSTANCED_GEOMETRY);
    dumpBlk("probes (0x6C)",                       OFF_COLORS);
    dumpBlk("meshes (0x7C, via imported_geometry)", OFF_MESHES);
    dumpBlk("per_instance_lightmap_texcoords (0xF4)", OFF_PERINSTANCE_LM_UV2);
    dumpBlk("airprobes (0x12C)",                   OFF_AIRPROBES);
}

// Per-process dedup - we want one dump per unique Lbsp tag id, not one
// per `ExtractClusterAtlasPair` call (which fires once per cluster mesh).
static std::atomic<uint64_t> g_phase0DumpedLbsps[64] = {};
static std::atomic<int> g_phase0DumpIndex{0};

static bool Phase0AlreadyDumped(uint32_t lbspTagId) {
    int n = g_phase0DumpIndex.load(std::memory_order_acquire);
    for (int i = 0; i < n && i < 64; i++) {
        if ((uint32_t)g_phase0DumpedLbsps[i].load(std::memory_order_relaxed) == lbspTagId)
            return true;
    }
    int slot = g_phase0DumpIndex.fetch_add(1, std::memory_order_acq_rel);
    if (slot < 64) g_phase0DumpedLbsps[slot].store(lbspTagId, std::memory_order_release);
    return false;
}

// Read a TagReference's TagId. Mirrors MapBspParser.cpp's ReadTagRefId - Gen3+
// layout has TagId at +12, with 0xFFFFFFFFu being the genuine null sentinel.
int32_t LpReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// =============================================================================
// Walk scnr -> StructureBsps[i] to find the bspIndex for the given sbsp tag id.
// Returns -1 on any failure.
// =============================================================================
int32_t FindBspIndexForSbsp(CacheHandle* cache, uint32_t /*scnrIdx*/, uint32_t sbspTagId,
                            const uint8_t* scnrMeta, int OFF_STRUCTURE_BSPS)
{
    TagBlockRef bspsBlk = ReadTagBlock(scnrMeta + OFF_STRUCTURE_BSPS);
    if (bspsBlk.count <= 0 || bspsBlk.count > 0x10000) return -1;
    int64_t bspsOff = TagMetaFileOff(cache, bspsBlk.pointer);
    constexpr int STRUCTURE_BSP_BLOCK_SIZE = 172;
    if (bspsOff < 0 ||
        (size_t)bspsOff + (size_t)bspsBlk.count * STRUCTURE_BSP_BLOCK_SIZE > cache->size)
        return -1;

    uint16_t sbspMasked = (uint16_t)(sbspTagId & 0xFFFFu);
    for (int i = 0; i < bspsBlk.count; ++i) {
        const uint8_t* b = cache->base + bspsOff + (size_t)i * STRUCTURE_BSP_BLOCK_SIZE;
        int32_t bspRefId   = LpReadTagRefId(b);
        uint16_t maskedRef = (bspRefId < 0) ? 0xFFFFu : (uint16_t)(bspRefId & 0xFFFFu);
        if (bspRefId == (int32_t)sbspTagId || maskedRef == sbspMasked)
            return i;
    }
    return -1;
}

// =============================================================================
// Resolve the active Lbsp tag id + resource pointer for the given sbsp.
//   sbsp -> [via scnr.StructureBsps walk] -> bspIndex
//   scnr.ScenarioLightmapReference -> sLdT (scenario_lightmap)
//   sLdT.LightmapRefs[bspIndex].LightmapDataReference -> Lbsp tag id
//   Lbsp+OFF_RESOURCE_POINTER -> resourceIdRaw
// Returns 0xFFFFFFFFu on any failure. Out params populated regardless.
// =============================================================================
uint32_t ResolveActiveLbspTagIdInner(CacheHandle* cache, uint32_t sbspTagId,
                                     int32_t* outResourceIdRaw)
{
    if (outResourceIdRaw) *outResourceIdRaw = 0;

    // Per-call diag budget - log up to 8 distinct sbsp resolves.
    static std::atomic<int> s_resolveDiagBudget{ 8 };
    bool logThis = false;
    {
        int v = s_resolveDiagBudget.load(std::memory_order_relaxed);
        while (v > 0) {
            if (s_resolveDiagBudget.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            { logThis = true; break; }
        }
    }

    // 1. Validate sbsp.
    if (sbspTagId >= cache->tags.size()) {
        if (logThis) NativeDiag("Lbsp[%u]: sbspTagId OOB tags=%llu",
            sbspTagId, (unsigned long long)cache->tags.size());
        return 0xFFFFFFFFu;
    }
    if (memcmp(cache->tags[sbspTagId].classCode, TC_SBSP, 4) != 0) {
        if (logThis) NativeDiag("Lbsp[%u]: not sbsp (class=%.4s)",
            sbspTagId, cache->tags[sbspTagId].classCode);
        return 0xFFFFFFFFu;
    }

    // 2. scnr global tag.
    int scnrIdx = FindGlobalTag(cache, TC_SCNR);
    if (scnrIdx < 0) {
        if (logThis) NativeDiag("Lbsp[%u]: no scnr global tag", sbspTagId);
        return 0xFFFFFFFFu;
    }
    int64_t scnrMetaOff = TagMetaFileOff(cache, cache->tags[scnrIdx].metaPointerRaw);
    if (scnrMetaOff < 0) {
        if (logThis) NativeDiag("Lbsp[%u]: bad scnr meta off scnrIdx=%d", sbspTagId, scnrIdx);
        return 0xFFFFFFFFu;
    }

    LpScnrLayout SL = PickScnr(cache->cacheType);
    if ((size_t)scnrMetaOff + (size_t)SL.OFF_SCENARIO_LIGHTMAP_REF + 16 > cache->size) {
        if (logThis) NativeDiag("Lbsp[%u]: scnr meta truncated metaOff=%lld lmRef=%d",
            sbspTagId, (long long)scnrMetaOff, SL.OFF_SCENARIO_LIGHTMAP_REF);
        return 0xFFFFFFFFu;
    }
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    // 3. Find the sbsp's index in scnr.StructureBsps[].
    int32_t bspIndex = FindBspIndexForSbsp(cache, (uint32_t)scnrIdx, sbspTagId,
                                            scnrMeta, SL.OFF_STRUCTURE_BSPS);
    if (bspIndex < 0) {
        if (logThis) NativeDiag("Lbsp[%u]: bspIndex not found in scnr.StructureBsps",
            sbspTagId);
        return 0xFFFFFFFFu;
    }

    // 4. scnr.ScenarioLightmapReference -> sLdT.
    int32_t ltmpId = LpReadTagRefId(scnrMeta + SL.OFF_SCENARIO_LIGHTMAP_REF);
    if (ltmpId < 0 || (uint32_t)ltmpId >= cache->tags.size()) {
        if (logThis) NativeDiag("Lbsp[%u]: bad ltmp tag id=%d", sbspTagId, ltmpId);
        return 0xFFFFFFFFu;
    }
    if (memcmp(cache->tags[ltmpId].classCode, TC_LTMP, 4) != 0) {
        if (logThis) NativeDiag("Lbsp[%u]: ltmp tag class wrong (got %.4s)",
            sbspTagId, cache->tags[ltmpId].classCode);
        return 0xFFFFFFFFu;
    }

    // 5. sLdT.LightmapRefs[bspIndex].LightmapDataReference -> Lbsp.
    int64_t ltmpMetaOff = TagMetaFileOff(cache, cache->tags[ltmpId].metaPointerRaw);
    if (ltmpMetaOff < 0 || (size_t)ltmpMetaOff + 16 > cache->size) {
        if (logThis) NativeDiag("Lbsp[%u]: bad ltmp meta off=%lld",
            sbspTagId, (long long)ltmpMetaOff);
        return 0xFFFFFFFFu;
    }
    const uint8_t* ltmpMeta = cache->base + ltmpMetaOff;

    // scenario_lightmap.LightmapRefs @ +4 (BlockCollection<LightmapDataInfoBlock>)
    constexpr int OFF_LIGHTMAP_REFS               = 4;
    constexpr int LIGHTMAP_DATA_INFO_BLOCK_SIZE   = 32;
    TagBlockRef lmRefsBlk = ReadTagBlock(ltmpMeta + OFF_LIGHTMAP_REFS);
    if (lmRefsBlk.count <= bspIndex) {
        if (logThis) NativeDiag("Lbsp[%u]: lmRefs.count=%d <= bspIndex=%d",
            sbspTagId, lmRefsBlk.count, bspIndex);
        return 0xFFFFFFFFu;
    }

    int64_t lmRefsOff = TagMetaFileOff(cache, lmRefsBlk.pointer);
    if (lmRefsOff < 0 ||
        (size_t)lmRefsOff + (size_t)lmRefsBlk.count * LIGHTMAP_DATA_INFO_BLOCK_SIZE > cache->size)
    {
        if (logThis) NativeDiag("Lbsp[%u]: lmRefs OOB off=%lld count=%d",
            sbspTagId, (long long)lmRefsOff, lmRefsBlk.count);
        return 0xFFFFFFFFu;
    }

    const uint8_t* lmInfo = cache->base + lmRefsOff +
                            (size_t)bspIndex * LIGHTMAP_DATA_INFO_BLOCK_SIZE;
    int32_t lbspId = LpReadTagRefId(lmInfo);
    if (lbspId < 0 || (uint32_t)lbspId >= cache->tags.size()) {
        if (logThis) NativeDiag("Lbsp[%u]: bad lbsp tag id=%d (bspIndex=%d)",
            sbspTagId, lbspId, bspIndex);
        return 0xFFFFFFFFu;
    }
    if (memcmp(cache->tags[lbspId].classCode, TC_LBSP, 4) != 0) {
        if (logThis) NativeDiag("Lbsp[%u]: lbsp class wrong (got %.4s)",
            sbspTagId, cache->tags[lbspId].classCode);
        return 0xFFFFFFFFu;
    }

    // 6. Read Lbsp+268 ResourcePointer.
    if (outResourceIdRaw) {
        int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspId].metaPointerRaw);
        LpLbspLayout LL = PickLbsp(cache->cacheType);
        if (lbspMetaOff >= 0 &&
            (size_t)lbspMetaOff + (size_t)LL.OFF_RESOURCE_POINTER + 4 <= cache->size)
        {
            const uint8_t* lbspMeta = cache->base + lbspMetaOff;
            *outResourceIdRaw = R32(lbspMeta + LL.OFF_RESOURCE_POINTER);
        }
    }

    if (logThis) NativeDiag(
        "Lbsp[%u]: OK bspIndex=%d ltmpId=%d lbspId=%d resRaw=0x%x",
        sbspTagId, bspIndex, ltmpId, lbspId,
        outResourceIdRaw ? (unsigned)*outResourceIdRaw : 0u);
    return (uint32_t)lbspId;
}

uint32_t SehResolveActiveLbsp(CacheHandle* cache, uint32_t sbspTagId,
                              int32_t* outResourceIdRaw)
{
    __try { return ResolveActiveLbspTagIdInner(cache, sbspTagId, outResourceIdRaw); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0xFFFFFFFFu; }
}

// Validate the cluster index against the sbsp's Clusters[] count.
// Returns the cluster count on success (>= 0), -1 on failure.
int32_t GetSbspClusterCountInner(CacheHandle* cache, uint32_t sbspTagId)
{
    if (sbspTagId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[sbspTagId].classCode, TC_SBSP, 4) != 0) return -1;
    int64_t metaOff = TagMetaFileOff(cache, cache->tags[sbspTagId].metaPointerRaw);
    if (metaOff < 0) return -1;

    LpSbspLayout SL = PickSbsp(cache->cacheType);
    if ((size_t)metaOff + (size_t)SL.OFF_CLUSTERS + 8 > cache->size) return -1;

    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef blk = ReadTagBlock(meta + SL.OFF_CLUSTERS);
    if (blk.count < 0 || blk.count > 0x10000) return -1;
    return blk.count;
}

// =============================================================================
// Extract per-cluster
// (DM, SDM) bitm tag ids + submap index DIRECTLY from the Lbsp tag main body.
//
// The runtime per-cluster atlas tuple at scenario_lightmap+0xC8+i*0x148 +
// 0xF0/+0xF6/+0xF8/+0xFE is synthesized by the engine at resource-page-load
// time from these on-disk fields (verified static schema):
//   Lbsp_meta + 0x01C  -> tag_reference  (light DIRECTION) -> DM bitm tag id
//   Lbsp_meta + 0x02C  -> tag_reference  (light INTENSITY) -> SDM bitm tag id
//   Lbsp_meta + 0x054  -> 12B block descriptor (count, ptr_raw, def_id)
//     cluster[i] @ resolved_ptr + i*8:
//       +0x00  i16  lightprobe_texture_array_index  -> DM submap = SDM submap
//       +0x02  i16  pervertex_block_index
//       +0x04  i32  pervertex_block_offset
//
// Hypothesis A (section 8.4): one i16 lightprobe_texture_array_index feeds BOTH
// the DM and SDM submap; pool indices are scenario-wide (one DM + one SDM
// bitm per Lbsp). Verified against camden-smallwood/blam-tags JSON +
// Sapien struct-defs section 7.
//
// Failure modes:
//   * tag_id at +0x01C / +0x02C is the 0xFFFFFFFF null sentinel: graceful
//     no-lightmap fallback (return false).
//   * resolved tag isn't a `bitm` (e.g. malformed Lbsp): same.
//   * clusters block descriptor count <= clusterIndex: bounds-failure,
//     same.
//   * cluster_ptr_raw doesn't resolve via TagMetaFileOff: same.
// =============================================================================
bool ExtractClusterAtlasPair(CacheHandle* cache,
                             uint32_t sbspTagId,
                             uint32_t clusterIndex,
                             uint32_t lbspTagId,
                             ZH_LbspClusterEntry* outEntry)
{
    if (lbspTagId >= cache->tags.size()) return false;

    int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspTagId].metaPointerRaw);
    if (lbspMetaOff < 0) return false;

    // One-time per-Lbsp layout diagnostic. Fires for the first
    // cluster-mesh lookup of each Lbsp tag so we capture every loaded
    // BSP's layout without flooding the log. The dump tells us the
    // count + ptr + first-16-bytes of every plugin-named block, so we
    // can validate the LBSP_IMPLEMENTATION_PLAN.md assumptions about
    // which blocks parallel the instance list. Cheap (<200 bytes of
    // log per Lbsp), bounded (64-entry dedup cap).
    if (!Phase0AlreadyDumped(lbspTagId)) {
        Phase0DumpLbspLayout(cache, lbspTagId, lbspMetaOff);
    }

    LpLbspLayout LL = PickLbsp(cache->cacheType);
    // Need to read up to OFF_CLUSTERS_BLOCK + 8 bytes (block count + ptr +
    // def_id is 12B but TagBlockRef only consumes the first 8). Use 0x60 as a
    // generous bound that still covers all three fields.
    const size_t kMinSize = (size_t)LL.OFF_CLUSTERS_BLOCK + 8;
    if ((size_t)lbspMetaOff + kMinSize > cache->size) return false;
    const uint8_t* lbspMeta = cache->base + lbspMetaOff;

    // 1. DM bitm tag id (light direction texture reference).
    int32_t dmId = LpReadTagRefId(lbspMeta + LL.OFF_LIGHT_DIRECTION_TAGREF);
    if (dmId < 0 || (uint32_t)dmId >= cache->tags.size()) return false;
    if (memcmp(cache->tags[dmId].classCode, TC_BITM, 4) != 0) return false;

    // 2. SDM bitm tag id (light intensity texture reference).
    int32_t sdmId = LpReadTagRefId(lbspMeta + LL.OFF_LIGHT_INTENSITY_TAGREF);
    bool sdmOk = (sdmId >= 0 && (uint32_t)sdmId < cache->tags.size() &&
                  memcmp(cache->tags[sdmId].classCode, TC_BITM, 4) == 0);
    if (!sdmOk) {
        // SDM is optional - some BSPs ship only DM (intensity-less). Reuse
        // DM so callers still get a working atlas; the visual diff is
        // small (intensity = full-bright assumption).
        sdmId = dmId;
    }

    // 3. Clusters block - read count + ptr_raw, resolve cluster[i] entry.
    // LBSP_TAG_LAYOUT_RE: also surface the schema's
    // per-cluster pervertex_block_index/offset so callers have the
    // complete cluster record, not just the lightprobe submap index.
    TagBlockRef clustersBlk = ReadTagBlock(lbspMeta + LL.OFF_CLUSTERS_BLOCK);
    int32_t submapIdx = 0;
    bool submapOk = false;
    int32_t pervertexBlockIdx = -1;
    int32_t pervertexBlockOff = 0;
    if (clustersBlk.count > 0 &&
        clustersBlk.count <= 0x10000 &&
        (int32_t)clusterIndex < clustersBlk.count)
    {
        int64_t clustersOff = TagMetaFileOff(cache, clustersBlk.pointer);
        if (clustersOff >= 0 &&
            (size_t)clustersOff +
            (size_t)clustersBlk.count * (size_t)LL.CLUSTER_DATA_BLOCK_SIZE <=
            cache->size)
        {
            const uint8_t* cluster = cache->base + clustersOff +
                (size_t)clusterIndex * (size_t)LL.CLUSTER_DATA_BLOCK_SIZE;
            // Per scenario_lightmap_cluster_data (sizeof 8):
            //   +0x00 i16 lightprobe_texture_array_index
            //   +0x02 i16 pervertex_block_index
            //   +0x04 i32 pervertex_block_offset
            int16_t lp = (int16_t)RU16(cluster + LL.CLUSTER_OFFSET_LIGHTPROBE_INDEX);
            // Negative lightprobe index = "no lightmap on this cluster"
            // (e.g. clusters that fall outside the lightmap volume). Treat
            // as submap 0 fallback rather than a hard failure.
            if (lp >= 0) {
                submapIdx = (int32_t)lp;
                submapOk = true;
            }
            pervertexBlockIdx = (int32_t)(int16_t)RU16(cluster + 0x02);
            pervertexBlockOff = R32(cluster + 0x04);
        }
    }

    outEntry->DmBitmapTagId       = (uint32_t)dmId;
    outEntry->SdmBitmapTagId      = (uint32_t)sdmId;
    outEntry->SubmapIndexDM       = (uint32_t)submapIdx;
    outEntry->SubmapIndexSDM      = (uint32_t)submapIdx;   // Hypothesis A: one i16 feeds both
    outEntry->PervertexBlockIndex = pervertexBlockIdx;
    outEntry->PervertexBlockOffset = pervertexBlockOff;
    // Per-cluster UV scale/bias is identity here - each submap is an
    // independent decoded BitmapSource, and UV2 is in [0,1] of the submap.
    // Reserved for future per-cluster atlas-tile decoding work; identity
    // means "sample UV2 as-is" which matches the submapResolver path.
    outEntry->UvScaleU = 1.0f;
    outEntry->UvScaleV = 1.0f;
    outEntry->UvBiasU  = 0.0f;
    outEntry->UvBiasV  = 0.0f;
    // Per-cluster HDR scale (engine constant 0x570001 .w). Default 1.0
    // because the on-disk source field is not statically resolvable - the
    // engine derives it from the resource-page-load path in haloreach.dll.
    // A future RE pass with a runtime probe at state[i]+0x100 can surface
    // a real value without an ABI change.
    outEntry->HdrScale = 1.0f;

    // First-cluster diag per BSP - confirms the new path is firing and
    // shows what got resolved. Budget is per-process to keep log bounded.
    static std::atomic<int> s_extractDiagBudget{ 32 };
    int v = s_extractDiagBudget.load(std::memory_order_relaxed);
    if (clusterIndex == 0 && v > 0 &&
        s_extractDiagBudget.compare_exchange_weak(v, v - 1,
            std::memory_order_relaxed, std::memory_order_relaxed))
    {
        NativeDiag(
            "LbspExtract[%u]: DM=0x%04x ('%s') SDM=0x%04x ('%s') "
            "cluster[0].submap=%d (clusters_count=%d, submapOk=%d)",
            sbspTagId,
            (unsigned)dmId,
            cache->tags[dmId].tagName.c_str(),
            (unsigned)sdmId,
            cache->tags[sdmId].tagName.c_str(),
            submapIdx, clustersBlk.count, submapOk ? 1 : 0);
    }
    return true;
}

bool SehGetClusterEntry(CacheHandle* cache, uint32_t sbspTagId,
                        uint32_t clusterIndex, ZH_LbspClusterEntry* outEntry)
{
    __try {
        // Always start with sentinels so a partial failure leaves the
        // caller with a well-defined "no lightmap" struct.
        memset(outEntry, 0, sizeof(*outEntry));
        outEntry->DmBitmapTagId        = 0xFFFFFFFFu;
        outEntry->SdmBitmapTagId       = 0xFFFFFFFFu;
        outEntry->PervertexBlockIndex  = -1;
        outEntry->PervertexBlockOffset = 0;
        // Identity UV scale/bias + identity HDR scale on the failure path so
        // a partial resolution still leaves the viewer with values it can
        // safely multiply (no NaN / no zero-scale collapse).
        outEntry->UvScaleU = 1.0f;
        outEntry->UvScaleV = 1.0f;
        outEntry->UvBiasU  = 0.0f;
        outEntry->UvBiasV  = 0.0f;
        outEntry->HdrScale = 1.0f;

        // Resolve the active Lbsp - both validates the chain and gives us
        // the lbsp tag id Path A reads its tag_refs from.
        int32_t resourceIdRaw = 0;
        uint32_t lbspId = ResolveActiveLbspTagIdInner(cache, sbspTagId, &resourceIdRaw);
        if (lbspId == 0xFFFFFFFFu) return false;

        // Bounds-check the cluster index against sbsp.Clusters[].
        int32_t clusterCount = GetSbspClusterCountInner(cache, sbspTagId);
        if (clusterCount <= 0) return false;
        if ((int32_t)clusterIndex >= clusterCount) return false;

        // Pull the DM / SDM tag ids + per-cluster submap
        // index from the Lbsp tag main body - this avoids the resource-page
        // bitm-class scan that fails on Reach BSPs (panopticon).
        ExtractClusterAtlasPair(cache, sbspTagId, clusterIndex, lbspId, outEntry);

        // ALWAYS return true if the chain resolved + cluster is valid,
        // regardless of atlas extraction success. The contract from the
        // viewer side is "DM/SDM == 0xFFFFFFFFu means no atlas, fall back".
        return true;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outEntry) {
            memset(outEntry, 0, sizeof(*outEntry));
            outEntry->DmBitmapTagId        = 0xFFFFFFFFu;
            outEntry->SdmBitmapTagId       = 0xFFFFFFFFu;
            outEntry->PervertexBlockIndex  = -1;
            outEntry->PervertexBlockOffset = 0;
            outEntry->UvScaleU = 1.0f;
            outEntry->UvScaleV = 1.0f;
            outEntry->UvBiasU  = 0.0f;
            outEntry->UvBiasV  = 0.0f;
            outEntry->HdrScale = 1.0f;
        }
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

extern "C" __declspec(dllexport) uint32_t __stdcall ZH_LBSP_GetActiveLbspTagId(
    uint64_t cacheHandle, uint32_t sbspTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0xFFFFFFFFu;
    int32_t unused = 0;
    return SehResolveActiveLbsp(cache, sbspTagId, &unused);
}

extern "C" __declspec(dllexport) bool __stdcall ZH_LBSP_GetClusterEntry(
    uint64_t cacheHandle, uint32_t sbspTagId, uint32_t clusterIndex,
    ZH_LbspClusterEntry* outEntry)
{
    if (!outEntry) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        memset(outEntry, 0, sizeof(*outEntry));
        outEntry->DmBitmapTagId        = 0xFFFFFFFFu;
        outEntry->SdmBitmapTagId       = 0xFFFFFFFFu;
        outEntry->PervertexBlockIndex  = -1;
        outEntry->PervertexBlockOffset = 0;
        outEntry->UvScaleU = 1.0f;
        outEntry->UvScaleV = 1.0f;
        outEntry->UvBiasU  = 0.0f;
        outEntry->UvBiasV  = 0.0f;
        outEntry->HdrScale = 1.0f;
        return false;
    }
    return SehGetClusterEntry(cache, sbspTagId, clusterIndex, outEntry);
}

// =============================================================================
// ExtractInstanceUV2VbIndex - per-instance lightmap-UV2 vertex-buffer index
// from the Lbsp tag's `Per-Instance Lightmap Texcoords` block at offset 0xF4
// (Lord Zedd plugin: `tagblock elementSize=0x2` -> `int16 Vertex Buffer
// Index`). Consumed by MapBspParser's ZH_BSP_GetInstancePvlVb.
//
// The block is PARALLEL to the sbsp's GeometryInstance list - both have the
// same count (3069 for settlement, 26 for a sky-only Lbsp). So instance
// ordinal k indexes directly into the block; no two-level redirection
// through `Unknown A` (0x48) or `Instanced Geometry` (0x60) is needed.
//
// Returns false on failure (no Lbsp, instance OOB, block missing). On
// success `outVbIndex` is the int16 VB index that the engine uses to
// fetch this instance's lightmap UV2 stream from the Lbsp's resource page.
// =============================================================================
constexpr int OFF_PERINSTANCE_LM_UV2_BLOCK     = 0xF4;
constexpr int PERINSTANCE_LM_UV2_ELEMENT_SIZE  = 0x2;   // one int16

bool ExtractInstanceUV2VbIndex(CacheHandle* cache,
                               uint32_t sbspTagId,
                               uint32_t instanceOrdinal,
                               int16_t* outVbIndex)
{
    if (!outVbIndex) return false;
    *outVbIndex = -1;
    if (!cache) return false;

    int32_t unused = 0;
    uint32_t lbspId = SehResolveActiveLbsp(cache, sbspTagId, &unused);
    if (lbspId == 0xFFFFFFFFu || lbspId >= cache->tags.size()) return false;

    int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspId].metaPointerRaw);
    if (lbspMetaOff < 0) return false;
    if ((size_t)lbspMetaOff + (size_t)OFF_PERINSTANCE_LM_UV2_BLOCK + 8 > cache->size)
        return false;

    const uint8_t* lbspMeta = cache->base + lbspMetaOff;
    TagBlockRef blk = ReadTagBlock(lbspMeta + OFF_PERINSTANCE_LM_UV2_BLOCK);
    if (blk.count <= 0 || blk.count > 0x100000) return false;
    if (instanceOrdinal >= (uint32_t)blk.count) return false;

    int64_t blkDataOff = TagMetaFileOff(cache, blk.pointer);
    if (blkDataOff < 0) return false;
    size_t need = (size_t)instanceOrdinal * (size_t)PERINSTANCE_LM_UV2_ELEMENT_SIZE
                + (size_t)PERINSTANCE_LM_UV2_ELEMENT_SIZE;
    if ((size_t)blkDataOff + need > cache->size) return false;

    const uint8_t* entry = cache->base + blkDataOff
                         + (size_t)instanceOrdinal * (size_t)PERINSTANCE_LM_UV2_ELEMENT_SIZE;
    *outVbIndex = (int16_t)RU16(entry);
    return true;
}

// =============================================================================
// ZH_LBSP_GetInstanceAtlasSubmap.
//
// Resolves the per-instance DM/SDM atlas submap + pool tag ids for a given
// instance ordinal. Walks Lbsp.Meshes[].InstanceBuckets[].instances[] looking
// for the matching instance_index, then returns:
//   * bucket.definition_index (i16) -> atlas submap selector (same for DM/SDM)
//   * Lbsp+0x1C / +0x2C tag refs -> DM / SDM bitm pool tag ids
//   * bucket.mesh_index + bucket ordinal -> diag fields
//
// Per LBSP_INSTANCE_BUCKETS_RE.md, the InstanceOrdinal -> bucket mapping is a
// reverse-scan, NOT a parallel array. Returns false on any failure; caller
// falls back to cluster-0 atlas (existing path) per the doc's section 7.5 contract.
// =============================================================================

// CORRECTED (dumped from Zealot's lbsp 0x15b2): `bucket.definition_index` is an
// **Lbsp MESH INDEX** -- it indexes Lbsp.Meshes[] (+0x7C, stride 0x5C, 250 entries on Zealot), NOT a
// bitmap submap. Every sampled bucket has mesh_index == definition_index, and the values run to
// ~240 against 250 meshes. Feeding it to find_bitmap(pool_tag, submap) fails for ~all instances
// (the pool bitmap has exactly ONE submap; only an index of 0 ever resolved). The per-instance page
// / chart transform must be read from Lbsp.Meshes[mesh_index] instead. Field names below kept for
// ABI compatibility -- read SubmapDM as "lbsp mesh index".
#pragma pack(push, 1)
struct ZH_LbspInstanceAtlasSubmap {
    uint32_t SubmapDM;       // bucket.definition_index == LBSP MESH INDEX (not a bitmap submap)
    uint32_t SubmapSDM;      // same value
    uint32_t PoolDM;         // DM bitm tag id from Lbsp+0x01C, 0xFFFFFFFF = none
    uint32_t PoolSDM;        // SDM bitm tag id from Lbsp+0x02C, 0xFFFFFFFF = none
    uint16_t MeshIndex;      // bucket.mesh_index - index into Lbsp.Meshes[]
    uint16_t BucketOrdinal;  // which bucket within mesh.InstanceBuckets[]
    uint32_t Reserved0;
    uint32_t Reserved1;
    uint32_t Reserved2;
};
#pragma pack(pop)

static void InstanceAtlasSubmap_InitSentinel(ZH_LbspInstanceAtlasSubmap* o)
{
    memset(o, 0, sizeof(*o));
    o->SubmapDM      = 0xFFFFFFFFu;
    o->SubmapSDM     = 0xFFFFFFFFu;
    o->PoolDM        = 0xFFFFFFFFu;
    o->PoolSDM       = 0xFFFFFFFFu;
    o->MeshIndex     = 0xFFFFu;
    o->BucketOrdinal = 0xFFFFu;
}

static bool ExtractInstanceAtlasSubmapInner(
    CacheHandle* cache,
    uint32_t sbspTagId,
    uint32_t instanceOrdinal,
    ZH_LbspInstanceAtlasSubmap* outEntry)
{
    // Local copies of the Lbsp.Meshes block offsets - the shared constants
    // at the bottom of this file are out of scope here, so duplicate them
    // (kept in sync via doc - both reflect Reach Lbsp stride 0x5C @ +0x7C).
    constexpr int LBSP_OFF_MESHES_LOC               = 0x7C;
    constexpr int LBSP_MESH_ELEMENT_SIZE_LOC        = 0x5C;

    InstanceAtlasSubmap_InitSentinel(outEntry);

    int32_t unused = 0;
    uint32_t lbspId = SehResolveActiveLbsp(cache, sbspTagId, &unused);
    if (lbspId == 0xFFFFFFFFu || lbspId >= cache->tags.size()) return false;

    int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspId].metaPointerRaw);
    if (lbspMetaOff < 0) return false;
    if ((size_t)lbspMetaOff + LBSP_OFF_MESHES_LOC + 8 > cache->size) return false;
    const uint8_t* lbspMeta = cache->base + lbspMetaOff;

    // Scenario-wide DM/SDM bitm pool ids (Lbsp+0x01C, Lbsp+0x02C).
    LpLbspLayout LL = PickLbsp(cache->cacheType);
    if ((size_t)lbspMetaOff + LL.OFF_LIGHT_INTENSITY_TAGREF + 16 > cache->size) return false;
    int32_t dmId  = LpReadTagRefId(lbspMeta + LL.OFF_LIGHT_DIRECTION_TAGREF);
    int32_t sdmId = LpReadTagRefId(lbspMeta + LL.OFF_LIGHT_INTENSITY_TAGREF);
    if (dmId  >= 0 && (uint32_t)dmId  < cache->tags.size() &&
        memcmp(cache->tags[dmId].classCode,  TC_BITM, 4) == 0)
        outEntry->PoolDM  = (uint32_t)dmId;
    if (sdmId >= 0 && (uint32_t)sdmId < cache->tags.size() &&
        memcmp(cache->tags[sdmId].classCode, TC_BITM, 4) == 0)
        outEntry->PoolSDM = (uint32_t)sdmId;
    else
        outEntry->PoolSDM = outEntry->PoolDM;  // SDM optional, fall back to DM

    // Walk Lbsp.Meshes[] looking for the bucket whose instances[] contains
    // the requested instance ordinal. O(meshes * buckets * instances) but
    // total instance count is bounded by sbsp.GeometryInstances cardinality
    // (typically <10K) and runs once per instance at material build, so the
    // amortised cost is fine.
    TagBlockRef meshes = ReadTagBlock(lbspMeta + LBSP_OFF_MESHES_LOC);
    if (meshes.count <= 0 || meshes.count > 0x10000) return false;
    int64_t meshesOff = TagMetaFileOff(cache, meshes.pointer);
    if (meshesOff < 0) return false;
    if ((size_t)meshesOff + (size_t)meshes.count * LBSP_MESH_ELEMENT_SIZE_LOC > cache->size)
        return false;

    constexpr int LBSP_MESH_OFF_INSTANCE_BUCKETS = 0x34;
    constexpr int INSTANCE_BUCKET_ELEMENT_SIZE   = 0x10;
    constexpr int BUCKET_OFF_MESH_INDEX          = 0x00;
    constexpr int BUCKET_OFF_DEFINITION_INDEX    = 0x02;
    constexpr int BUCKET_OFF_INSTANCES_BLOCK     = 0x04;
    constexpr int INSTANCE_INDEX_ELEMENT_SIZE    = 0x02;

    for (int mi = 0; mi < meshes.count; ++mi) {
        const uint8_t* mesh = cache->base + meshesOff
                            + (size_t)mi * LBSP_MESH_ELEMENT_SIZE_LOC;
        TagBlockRef bk = ReadTagBlock(mesh + LBSP_MESH_OFF_INSTANCE_BUCKETS);
        if (bk.count <= 0 || bk.count > 0x10000) continue;
        int64_t bkOff = TagMetaFileOff(cache, bk.pointer);
        if (bkOff < 0) continue;
        if ((size_t)bkOff + (size_t)bk.count * INSTANCE_BUCKET_ELEMENT_SIZE > cache->size)
            continue;

        for (int bi = 0; bi < bk.count; ++bi) {
            const uint8_t* bucket = cache->base + bkOff
                                  + (size_t)bi * INSTANCE_BUCKET_ELEMENT_SIZE;
            TagBlockRef inst = ReadTagBlock(bucket + BUCKET_OFF_INSTANCES_BLOCK);
            if (inst.count <= 0 || inst.count > 0x10000) continue;
            int64_t instOff = TagMetaFileOff(cache, inst.pointer);
            if (instOff < 0) continue;
            if ((size_t)instOff + (size_t)inst.count * INSTANCE_INDEX_ELEMENT_SIZE > cache->size)
                continue;

            for (int ii = 0; ii < inst.count; ++ii) {
                int16_t instIdx = (int16_t)RU16(cache->base + instOff
                                              + (size_t)ii * INSTANCE_INDEX_ELEMENT_SIZE);
                if ((int32_t)instIdx != (int32_t)instanceOrdinal) continue;

                int16_t defIdx = (int16_t)RU16(bucket + BUCKET_OFF_DEFINITION_INDEX);
                int16_t bkMesh = (int16_t)RU16(bucket + BUCKET_OFF_MESH_INDEX);
                outEntry->SubmapDM      = (defIdx >= 0) ? (uint32_t)defIdx : 0xFFFFFFFFu;
                outEntry->SubmapSDM     = outEntry->SubmapDM;
                outEntry->MeshIndex     = (uint16_t)(bkMesh & 0xFFFFu);
                outEntry->BucketOrdinal = (uint16_t)bi;
                return true;
            }
        }
    }
    return false;
}

static bool SehGetInstanceAtlasSubmap(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t instanceOrdinal,
    ZH_LbspInstanceAtlasSubmap* outEntry)
{
    __try {
        return ExtractInstanceAtlasSubmapInner(cache, sbspTagId,
                                               instanceOrdinal, outEntry);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outEntry) InstanceAtlasSubmap_InitSentinel(outEntry);
        return false;
    }
}

extern "C" __declspec(dllexport) bool __stdcall ZH_LBSP_GetInstanceAtlasSubmap(
    uint64_t cacheHandle, uint32_t sbspTagId, uint32_t instanceOrdinal,
    ZH_LbspInstanceAtlasSubmap* outEntry)
{
    if (!outEntry) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) { InstanceAtlasSubmap_InitSentinel(outEntry); return false; }
    return SehGetInstanceAtlasSubmap(cache, sbspTagId, instanceOrdinal, outEntry);
}

// =============================================================================
// ZH_LBSP_GetBrightness - engine-authored Lbsp scene brightness multiplier.
//
// Per Lord Zedd's ReachMCC plugin, the Lbsp tag carries a Brightness float at
// offset 0x18:
//   <float32 name="Brightness" offset="0x18" />
//
// This is the engine's per-BSP scene-light scalar - applied uniformly across
// all surfaces lit by the BSP's lightmap. Daylight maps author it ~1.0+;
// shadowed-interior maps author lower values to dim the whole scene. We
// surface it so the viewer can scale ambient + shader-driven
// lighting consistently to match what the engine renders.
//
// Returns 1.0 (engine-neutral) on any failure - caller treats that as
// "no brightness override available".
// =============================================================================

constexpr int OFF_LBSP_BRIGHTNESS = 0x18;

static float ResolveLbspBrightnessInner(CacheHandle* cache, uint32_t sbspTagId)
{
    if (!cache) return 1.0f;
    int32_t unused = 0;
    uint32_t lbspId = SehResolveActiveLbsp(cache, sbspTagId, &unused);
    if (lbspId == 0xFFFFFFFFu || lbspId >= cache->tags.size()) return 1.0f;
    const TagEntry& te = cache->tags[lbspId];
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + OFF_LBSP_BRIGHTNESS + 4 > cache->size) return 1.0f;
    float b;
    memcpy(&b, cache->base + metaOff + OFF_LBSP_BRIGHTNESS, 4);
    if (!std::isfinite(b) || b <= 0.001f || b > 1000.0f) return 1.0f;
    return b;
}

static float SehResolveLbspBrightness(CacheHandle* cache, uint32_t sbspTagId)
{
    __try { return ResolveLbspBrightnessInner(cache, sbspTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 1.0f; }
}

extern "C" __declspec(dllexport) float __stdcall ZH_LBSP_GetBrightness(
    uint64_t cacheHandle, uint32_t sbspTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 1.0f;
    return SehResolveLbspBrightness(cache, sbspTagId);
}

// =============================================================================
// LIGHTMAP_VMF_RE Path B - per-pixel lightprobe atlas surface.
//
// Spec source: LIGHTMAP_VMF_RE_FROM_HREK.md section "Per-pixel (BSP geometry,
// static_per_pixel_ps)" + lightmap_sampling.hlsl_include:34-54. The engine
// declares two atlas samplers per BSP:
//
//   lightprobe_hdr_color_ps - 3-slice Texture3D, RGB HDR per slice:
//       slice 0 = dominant lobe color
//       slice 1 = (analytical_mask + cloud_mask flags) [scalar slice]
//       slice 2 = fill lobe color
//   lightprobe_dir_and_bandwidth_ps - Texture2D, RGBA8, encoding per pixel:
//       sample 0 .a = dominant dir x (x 2 - 1)
//       sample 1 .a = dominant dir y
//       sample 2 .a = dominant dir z
//       sample 3 .a = bandwidth (kappa via exp(-6.238325 * alpha))
//
// Whatever the on-disk Lbsp ships, these are produced by the cooker's
// `Lightmap_BitmapName_LpArray` stage 10 (see docs/rendering/lightmap_cooking.html
// stage 10): two `bitm` tag references that the engine binds as
// (direction, intensity) - the SAME pair this file already exposes via
// `ZH_LBSP_GetClusterEntry.DmBitmapTagId / SdmBitmapTagId` (Lbsp+0x1C / +0x2C).
// The HLSL "Texture3D 3-slice" vs "Texture2D 4-sample" framing is a sampling
// convention applied at PS-bind time, not a separate on-disk asset:
//   * DM (Lbsp+0x1C, plugin "Primary Map / light_direction") -> the engine's
//     `lightprobe_dir_and_bandwidth_ps` atlas, RGBA8 per pixel.
//   * SDM (Lbsp+0x2C, plugin "Intensity Map / light_intensity") -> the engine's
//     `lightprobe_hdr_color_ps` atlas. Sliced 3x across V (or layer-array
//     index) by `sample_lightprobe_texture` to pull dom / mask / fill bands.
//
// This export surfaces the two tag ids + brightness + the raw resource id of
// the Lbsp's bitmap resource page (used by callers that want to extract the
// underlying BC3 / DXT5 bytes themselves through MapBitmapParser's
// existing path). The viewer samples the 2D SDM with the mesh UV2.
//
// Sentinel contract: every field is 0xFFFFFFFFu / 1.0f / 0u on failure so
// callers can treat (DmTagId == 0xFFFFFFFFu || SdmTagId == 0xFFFFFFFFu) as
// "no per-pixel atlas" and fall back to per-vertex VMF or Lambert+ambient.
// =============================================================================

#pragma pack(push, 1)
struct ZH_LbspLightprobeAtlas {
    uint32_t DmTagId;           // Lbsp+0x1C (lightprobe_dir_and_bandwidth bitm)
    uint32_t SdmTagId;          // Lbsp+0x2C (lightprobe_hdr_color bitm)
    float    Brightness;        // Lbsp+0x18 engine scene scalar (1.0 = neutral)
    int32_t  ResourceIdRaw;     // Lbsp+0x10C ResourcePointer (0 = no resource page)
    uint32_t LbspTagId;         // resolved Lbsp tag id (echo of GetActiveLbspTagId)
    uint32_t HasBrightness;     // 1 = Brightness was authored, 0 = walker fallback
    uint32_t Reserved0;
    uint32_t Reserved1;         // future: fill_color slice index
    uint32_t Reserved2;
    uint32_t Reserved3;
};
#pragma pack(pop)

static void LightprobeAtlas_InitSentinel(ZH_LbspLightprobeAtlas* o)
{
    if (!o) return;
    memset(o, 0, sizeof(*o));
    o->DmTagId       = 0xFFFFFFFFu;
    o->SdmTagId      = 0xFFFFFFFFu;
    o->Brightness    = 1.0f;
    o->ResourceIdRaw = 0;
    o->LbspTagId     = 0xFFFFFFFFu;
    o->HasBrightness = 0u;
}

static bool ExtractLightprobeAtlasInner(
    CacheHandle* cache, uint32_t sbspTagId, ZH_LbspLightprobeAtlas* outAtlas)
{
    LightprobeAtlas_InitSentinel(outAtlas);
    if (!cache) return false;

    int32_t resourceIdRaw = 0;
    uint32_t lbspId = SehResolveActiveLbsp(cache, sbspTagId, &resourceIdRaw);
    if (lbspId == 0xFFFFFFFFu || lbspId >= cache->tags.size()) return false;

    int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspId].metaPointerRaw);
    if (lbspMetaOff < 0) return false;

    LpLbspLayout LL = PickLbsp(cache->cacheType);
    if ((size_t)lbspMetaOff + (size_t)LL.OFF_LIGHT_INTENSITY_TAGREF + 16 > cache->size)
        return false;
    const uint8_t* lbspMeta = cache->base + lbspMetaOff;

    // Atlas tag refs (lightprobe_dir_and_bandwidth + lightprobe_hdr_color
    // - i.e. DM + SDM in plugin terms). Validate bitm class so a malformed
    // Lbsp doesn't surface a non-bitmap tag id to the caller.
    int32_t dmId  = LpReadTagRefId(lbspMeta + LL.OFF_LIGHT_DIRECTION_TAGREF);
    int32_t sdmId = LpReadTagRefId(lbspMeta + LL.OFF_LIGHT_INTENSITY_TAGREF);
    if (dmId  >= 0 && (uint32_t)dmId  < cache->tags.size() &&
        memcmp(cache->tags[dmId].classCode,  TC_BITM, 4) == 0)
        outAtlas->DmTagId  = (uint32_t)dmId;
    if (sdmId >= 0 && (uint32_t)sdmId < cache->tags.size() &&
        memcmp(cache->tags[sdmId].classCode, TC_BITM, 4) == 0)
        outAtlas->SdmTagId = (uint32_t)sdmId;
    else
        outAtlas->SdmTagId = outAtlas->DmTagId; // SDM optional - fall back to DM

    // Brightness (+0x18) with same validation as ZH_LBSP_GetBrightness.
    if ((size_t)lbspMetaOff + OFF_LBSP_BRIGHTNESS + 4 <= cache->size) {
        float b;
        memcpy(&b, lbspMeta + OFF_LBSP_BRIGHTNESS, 4);
        if (std::isfinite(b) && b > 0.001f && b <= 1000.0f) {
            outAtlas->Brightness    = b;
            outAtlas->HasBrightness = 1u;
        }
    }

    outAtlas->ResourceIdRaw = resourceIdRaw;
    outAtlas->LbspTagId     = lbspId;

    // One-shot per-Lbsp diag so we can confirm what got surfaced.
    static std::atomic<int> s_atlasDiagBudget{ 16 };
    int v = s_atlasDiagBudget.load(std::memory_order_relaxed);
    if (v > 0 && s_atlasDiagBudget.compare_exchange_weak(v, v - 1,
            std::memory_order_relaxed, std::memory_order_relaxed))
    {
        const char* dmName  = (outAtlas->DmTagId  < cache->tags.size())
            ? cache->tags[outAtlas->DmTagId ].tagName.c_str() : "(null)";
        const char* sdmName = (outAtlas->SdmTagId < cache->tags.size())
            ? cache->tags[outAtlas->SdmTagId].tagName.c_str() : "(null)";
        NativeDiag(
            "LbspAtlas[sbsp=0x%04x lbsp=0x%04x]: DM=0x%04x ('%s') SDM=0x%04x ('%s') "
            "B=%.3f (auth=%u) resRaw=0x%x",
            (unsigned)sbspTagId, (unsigned)lbspId,
            (unsigned)outAtlas->DmTagId,  dmName,
            (unsigned)outAtlas->SdmTagId, sdmName,
            outAtlas->Brightness, outAtlas->HasBrightness,
            (unsigned)resourceIdRaw);
    }

    return (outAtlas->DmTagId != 0xFFFFFFFFu) || (outAtlas->SdmTagId != 0xFFFFFFFFu);
}

extern "C" __declspec(dllexport) bool __stdcall ZH_LBSP_GetLightprobeAtlas(
    uint64_t cacheHandle, uint32_t sbspTagId, ZH_LbspLightprobeAtlas* outAtlas)
{
    if (!outAtlas) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) { LightprobeAtlas_InitSentinel(outAtlas); return false; }
    __try { return ExtractLightprobeAtlasInner(cache, sbspTagId, outAtlas); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        LightprobeAtlas_InitSentinel(outAtlas);
        return false;
    }
}

// =============================================================================
// Reach lightprobe per-vertex VMF lobe - diagnostic + decode
//
// Spec source: captured_shaders/settlement/LIGHTMAP_DECODE_RE.md.
//
// At runtime the engine binds a per-vertex 8-byte VMF lobe stream as t17
// (ByteAddressBuffer per_vertex_lightprobe_buffer) and the VS unpacks it into
// (bounceDir, bounceColor, ambientColor) - see disasm lines 15-94 of
// lightprobe_vs.asm. That stream lives in the Lbsp tag's RESOURCE PAGE as one
// of its vertex buffers. The VB's stride is 8 (1 dir-index byte + 3 primary
// RGB bytes + 1 confidence byte + 3 secondary RGB bytes - disasm line 21
// loads two dwords = 8 bytes per vertex).
//
// We need to learn:
//   (a) Which Vertex Buffer Index slot (1..8) in the Lbsp `Meshes` block at
//       offset 0x18..0x26 carries the lightprobe VB index for each mesh.
//   (b) Which lbsp VB indices have stride==4 (those are lightprobe streams).
//
// LIGHTPROBE_STRIDE_RE: the stride is NOT 8. HREK shader source
// (HREK/tags/shaders/templated/lightmap_sampling.hlsl_include line 154)
// shows the per_vertex_lightprobe_buffer is a `BYTE_ADDRESS_BUFFER` accessed
// with `Load2(floor(vid * 1.5) * 4)` - a sliding 8-byte window at 6-byte
// effective per-vertex stride. The Lbsp VB pool declares it with raw stride 4.
// =============================================================================

namespace {

// Open the cache's Lbsp resource page + VB-pool fixup region for a given sbsp.
// Returns false on any failure. On success outResourceData / outResourceSize
// own a freshly-malloc'd page payload (caller frees via free()), and the
// vbCounts/vbLens/vbStrides/vbFixupOffsets vectors carry the parsed VB pool.
//
// Wrapped via the public Lbsp resolver - we don't actually need BspData
// because the Lbsp resource page is independent of any open ZH_BspHandle.
// This means the decoder can run before the BSP is opened.
struct LbspResAndFixup {
    uint8_t*              resData = nullptr;
    size_t                resSize = 0;
    std::vector<uint32_t> vbCounts;
    std::vector<uint32_t> vbLens;
    std::vector<uint32_t> vbStrides;
    std::vector<uint32_t> vbFixupOffsets;
    uint32_t              lbspTagId = 0xFFFFFFFFu;
    int64_t               lbspMetaOff = -1;
    // Cached entries (in g_lbspResCache below) own their resData - the
    // destructor is suppressed via `owned=false` on hand-out wrappers.
    bool                  owned = true;

    ~LbspResAndFixup() {
        if (owned && resData) free(resData);
    }
};

// =============================================================================
// LBSP resource page cache.
//
// `LoadLbspResourceAndFixup` was being called per-mesh by
// `ExtractMeshLightprobeBytesInner` and re-decompressing the 67 MB -> 155 MB
// LBSP resource page on EVERY call. For settlement's 489 cluster meshes
// that's ~33 GB of decompression work, freezing the viewer for the full
// duration of a BSP load (user-visible symptom: "meshes don't load").
//
// Fix: cache the loaded data per (cache*, lbspTagId). The resource page is
// immutable post-load, so it's safe to share one decompressed copy across
// every mesh in the BSP. Per-process map; entries leak for cache lifetime
// - typical map has <8 Lbsps x ~155 MB each = ~1.2 GB worst case, which is
// acceptable for the viewer's debug-build memory budget. Production might
// want an LRU but for now leaking is the simplest correct path.
// =============================================================================
struct LbspResCacheKey {
    void*    cache;
    uint32_t lbspTagId;
    bool operator==(const LbspResCacheKey& o) const noexcept
    { return cache == o.cache && lbspTagId == o.lbspTagId; }
};
struct LbspResCacheKeyHash {
    size_t operator()(const LbspResCacheKey& k) const noexcept
    {
        return ((size_t)(uintptr_t)k.cache) ^ ((size_t)k.lbspTagId << 16);
    }
};
struct LbspResCacheEntry {
    uint8_t*              resData = nullptr;
    size_t                resSize = 0;
    std::vector<uint32_t> vbCounts;
    std::vector<uint32_t> vbLens;
    std::vector<uint32_t> vbStrides;
    std::vector<uint32_t> vbFixupOffsets;
    int64_t               lbspMetaOff = -1;
    bool                  hasStride4 = false; // fast-path negative-cache flag (LIGHTPROBE_STRIDE_RE)
};
static std::mutex g_lbspResCacheMutex;
static std::unordered_map<LbspResCacheKey, LbspResCacheEntry*,
                          LbspResCacheKeyHash> g_lbspResCache;

// SINGLE-FLIGHT guard against the LBSP-page decompression
// STAMPEDE. The Rust load pool runs the BSP par-decode on ~ncpu-2 threads (30 on a
// 32-thread box). Every one of a cluster's meshes resolves to the SAME active Lbsp
// resource page, so at load start all 30 workers MISS the cache simultaneously and
// EACH decompress the same ~155 MB page (ReadResourceData) before any of them inserts - 
// 30 x 155 MB ~= 4.6 GB of transient decompression buffers alive at once (measured peak
// WorkingSet ~10.5 GB on forge_halo; 29 copies are freed immediately after the insert
// race). Serialize per-KEY: the first thread claims the load, decompresses ONCE, and
// wakes the rest, who then read the cached page. Different Lbsp pages still load in
// parallel. `g_lbspResLoading` = keys currently being decompressed; the CV wakes waiters.
static std::condition_variable g_lbspResCacheCv;
static std::unordered_set<LbspResCacheKey, LbspResCacheKeyHash> g_lbspResLoading;

// Free ALL cached lightmap resource pages for a cache being closed. These are
// the decompressed LBSP pages (~155 MB each) - without this they leaked for the whole
// process lifetime, so loading several maps ballooned RAM to multiple GB. Called from the
// cache-release path (MapCacheCommon). `cachePtr` is the CacheHandle* being deleted.
extern "C" void PurgeLightmapCachesForCache(void* cachePtr)
{
    {
        std::lock_guard<std::mutex> lk(g_lbspResCacheMutex);
        for (auto it = g_lbspResCache.begin(); it != g_lbspResCache.end(); ) {
            if (it->first.cache == cachePtr) {
                if (it->second) {
                    if (it->second->resData) free(it->second->resData);
                    delete it->second;
                }
                it = g_lbspResCache.erase(it);
            } else {
                ++it;
            }
        }
    }
}

// Walk the gestalt's zone fixup table to recover the Lbsp's per-VB byte
// offsets within the resource page. Mirrors MapBspParser.cpp's
// ParseFixupRegion in shape - but we can't call that directly because it
// lives in another translation unit and operates on a slightly different
// data type (`ResourceEntry`). Instead we replicate the offsets and bounds-
// checks here. Kept private because the moment we have a stable Lbsp ABI
// for VB-list extraction we should unify with MapBspParser's path.
//
// Constants 328 (fixupSize) / 340 (fixupPtr) / 24 (trailer offset) /
// VERTEX_BUFFER_INFO_SIZE 28 match MapBspParser.cpp exactly.
constexpr int LBSP_FX_VB_INFO_SIZE = 28;
static bool ParseLbspFixupRegion(CacheHandle* cache,
                                 int32_t resourceIdRaw,
                                 LbspResAndFixup& out)
{
    int32_t resourceIndex = resourceIdRaw & 0xFFFF;
    if (resourceIndex < 0 || resourceIndex >= (int)cache->resourceEntries.size())
        return false;
    {
        std::lock_guard<std::mutex> lk(cache->parseMutex);
        if (!EnsureResourceFixups(cache, (size_t)resourceIndex)) return false;
    }
    const ResourceEntry& entry = cache->resourceEntries[resourceIndex];

    int zoneIdx = FindGlobalTag(cache, "zone");
    if (zoneIdx < 0) return false;
    int64_t metaOff = TagMetaFileOff(cache, cache->tags[zoneIdx].metaPointerRaw);
    if (metaOff < 0 || (size_t)metaOff + 350 > cache->size) return false;
    const uint8_t* meta = cache->base + metaOff;

    int32_t fixupSize  = R32(meta + 328);
    uint32_t fixupPtrR = RU32(meta + 340);
    int64_t fixupOff   = TagMetaFileOff(cache, fixupPtrR);
    if (fixupSize < 0 || fixupOff < 0) return false;
    if ((size_t)fixupOff + (size_t)fixupSize > cache->size) return false;
    const uint8_t* fixupBase = cache->base + fixupOff;

    if (entry.fixupSize < 24) return false;
    int64_t trailerOff = (int64_t)entry.fixupOffset + (int64_t)entry.fixupSize - 24;
    if (trailerOff < 0 || (size_t)trailerOff + 16 > (size_t)fixupSize) return false;
    int32_t vbCount = R32(fixupBase + trailerOff);
    if (vbCount < 0 || vbCount > 0x10000) return false;

    size_t cursor = (size_t)entry.fixupOffset;
    if (cursor + (size_t)vbCount * LBSP_FX_VB_INFO_SIZE > (size_t)fixupSize)
        return false;

    out.vbCounts.assign((size_t)vbCount, 0u);
    out.vbLens.assign((size_t)vbCount, 0u);
    out.vbStrides.assign((size_t)vbCount, 0u);
    for (int i = 0; i < vbCount; ++i) {
        const uint8_t* p = fixupBase + cursor + i * LBSP_FX_VB_INFO_SIZE;
        out.vbCounts[i] = (uint32_t)R32(p + 0);
        out.vbLens[i]   = (uint32_t)R32(p + 8);
        if (out.vbCounts[i] > 0)
            out.vbStrides[i] = out.vbLens[i] / out.vbCounts[i];
    }

    out.vbFixupOffsets.clear();
    out.vbFixupOffsets.reserve((size_t)vbCount);
    for (size_t fi = 0; fi < (size_t)vbCount; ++fi) {
        uint32_t foff = 0;
        if (fi < entry.fixups.size())
            foff = (uint32_t)(entry.fixups[fi].offset & 0x0FFFFFFFu);
        out.vbFixupOffsets.push_back(foff);
    }
    return true;
}

// Resolve sbsp -> Lbsp + load resource page + parse VB pool.
// Returns nullptr on any failure. SEH-wrapped at the entry call site.
//
// Caches the loaded data per (cache*, lbspTagId) in g_lbspResCache. Hot path
// after first call is a hash-lookup + memcpy of metadata; the 155 MB res
// page stays resident in the cache entry and `out.resData` points at it
// (with `out.owned=false` so the destructor doesn't double-free).
static bool LoadLbspResourceAndFixup(CacheHandle* cache, uint32_t sbspTagId,
                                     LbspResAndFixup& out)
{
    int32_t resourceIdRaw = 0;
    uint32_t lbspId = SehResolveActiveLbsp(cache, sbspTagId, &resourceIdRaw);
    if (lbspId == 0xFFFFFFFFu) return false;
    if (resourceIdRaw == 0 || resourceIdRaw == -1) return false;
    int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspId].metaPointerRaw);
    if (lbspMetaOff < 0) return false;

    // Cache lookup - first hit on a given lbspTagId loads + caches; every
    // subsequent call within the process lifetime returns the cached entry.
    // Mutex protects only the map insert/lookup; the cached resData itself
    // is immutable post-load so readers don't need locking.
    LbspResCacheKey key{ cache, lbspId };
    // Single-flight admission. Loop until EITHER the page is cached (return
    // it) OR we are elected the sole loader (break and decompress). If another thread is
    // already decompressing this exact page, WAIT on the CV instead of decompressing a
    // redundant 155 MB copy - this is what caps the load-time memory peak.
    {
        std::unique_lock<std::mutex> lk(g_lbspResCacheMutex);
        for (;;) {
            auto it = g_lbspResCache.find(key);
            if (it != g_lbspResCache.end()) {
                const LbspResCacheEntry* e = it->second;
                // Populate `out` as a NON-owning view over the cached bytes.
                out.resData       = e->resData;
                out.resSize       = e->resSize;
                out.vbCounts      = e->vbCounts;
                out.vbLens        = e->vbLens;
                out.vbStrides     = e->vbStrides;
                out.vbFixupOffsets= e->vbFixupOffsets;
                out.lbspTagId     = lbspId;
                out.lbspMetaOff   = e->lbspMetaOff;
                out.owned         = false; // critical - the cache owns resData
                return true;
            }
            if (g_lbspResLoading.find(key) != g_lbspResLoading.end()) {
                // Peer is decompressing this page; block until it publishes or aborts.
                g_lbspResCacheCv.wait(lk);
                continue;
            }
            // We win the election - claim the load and decompress below (unlocked).
            g_lbspResLoading.insert(key);
            break;
        }
    }
    // If we abort past this point we MUST clear the loading claim and wake waiters, or
    // they deadlock. Helper for the failure returns.
    auto abort_load = [&]() {
        std::lock_guard<std::mutex> lk(g_lbspResCacheMutex);
        g_lbspResLoading.erase(key);
        g_lbspResCacheCv.notify_all();
    };

    // Cache miss - do the real load (we are the sole loader for `key`).
    constexpr size_t kMaxRead = 64ull * 1024ull * 1024ull;
    size_t resSize = 0;
    uint8_t* res = ReadResourceData(cache, resourceIdRaw, kMaxRead, &resSize);
    if (!res || resSize == 0) {
        if (res) free(res);
        abort_load();
        return false;
    }
    out.resData = res;
    out.resSize = resSize;
    out.lbspTagId = lbspId;
    out.lbspMetaOff = lbspMetaOff;
    out.owned = true; // we own this until we hand it to the cache
    if (!ParseLbspFixupRegion(cache, resourceIdRaw, out)) {
        // resData ownership stays on `out` so its destructor frees it.
        abort_load();
        return false;
    }

    // Move ownership into the cache. Detect stride==4 presence in pass - 
    // future lookups can short-circuit on `!hasStride4` if we want, but for
    // now we leave the byte extractor to do its own check.
    // LIGHTPROBE_STRIDE_RE: stride==4 is the engine truth - the
    // per_vertex_lightprobe_buffer is a ByteAddressBuffer fetched with a
    // sliding 8-byte window at 6-byte effective stride.
    bool hasStride4 = false;
    for (uint32_t s : out.vbStrides) { if (s == 4) { hasStride4 = true; break; } }

    LbspResCacheEntry* entry = new LbspResCacheEntry();
    entry->resData        = out.resData;     // take ownership
    entry->resSize        = out.resSize;
    entry->vbCounts       = out.vbCounts;
    entry->vbLens         = out.vbLens;
    entry->vbStrides      = out.vbStrides;
    entry->vbFixupOffsets = out.vbFixupOffsets;
    entry->lbspMetaOff    = out.lbspMetaOff;
    entry->hasStride4     = hasStride4;

    {
        std::lock_guard<std::mutex> lk(g_lbspResCacheMutex);
        // Re-check in case a parallel worker raced us in. With single-flight admission
        // this should not happen for the same key (we hold the sole load claim), but the
        // check is a cheap safety net. Either way, publish + release the claim + wake
        // any waiters that blocked on this key.
        auto it2 = g_lbspResCache.find(key);
        if (it2 != g_lbspResCache.end()) {
            // Parallel worker won; drop ours.
            delete entry;
            const LbspResCacheEntry* w = it2->second;
            out.resData       = w->resData;
            out.resSize       = w->resSize;
            out.vbCounts      = w->vbCounts;
            out.vbLens        = w->vbLens;
            out.vbStrides     = w->vbStrides;
            out.vbFixupOffsets= w->vbFixupOffsets;
            out.lbspMetaOff   = w->lbspMetaOff;
            out.owned         = false;
            g_lbspResLoading.erase(key);
            g_lbspResCacheCv.notify_all();
            return true;
        }
        g_lbspResCache[key] = entry;
        g_lbspResLoading.erase(key);
        g_lbspResCacheCv.notify_all();
    }
    // Our `out` no longer owns resData - the cache does.
    out.owned = false;
    NativeDiag("LbspResCache: stored lbsp=0x%04x resSize=%zu stride4=%s",
               (unsigned)lbspId, resSize, hasStride4 ? "yes" : "no");
    return true;
}

// PHASE_B_RE - schema-authoritative per-cluster PVL VB resolver.
//
// Replaces the KNOWN-UNSAFE mesh-slot scan in ExtractMeshLightprobeBytesInner.
// Walks the documented routing chain:
//   clusters[clusterIndex].pervertex_block_index  (already read for us by
//       ExtractClusterAtlasPair -> ZH_LbspClusterEntry, but we re-read locally
//       so this function is self-contained and SEH-bounded)
//   -> bsp_per_vertex_run_time_data[pvb_index] (+0x048, stride 4):
//        { i16 vertex_buffer_index, i16 hdr_scale }
//   -> lbsp_vb_pool[vertex_buffer_index]  = THE stride-4 PVL ByteAddressBuffer
//
// HDR-SCALE DECODE (i16 hdr_scale -> float):  This is the engine's
// `per_vertex_lighting_offset.y` (see HREK lightmap_sampling.hlsl_include
// lines 141/144: `vmf1.xyz *= vmf1.xyz * per_vertex_lighting_offset.y`). The
// on-disk field is `vertex_buffer_hdr_scale` (i16). The exact i16 -> float
// encoding is NOT documented in lbsp_reach.json and could not be settled
// statically (the engine populates the VS constant at resource-page load).
// We surface the RAW i16 (HdrScaleRaw) AND a best-effort decoded HdrScale so
// the viewer can either use it or fall back to lbspBrightness. Decode used:
// treat the i16 as an IEEE half-float (the values seen live, e.g. 13026 /
// 17343, are plausible halfs ~ 0.34 / 1.93 - NOT plausible as raw ints or a
// fixed /256). This is FLAGGED uncertain; the viewer keeps lbspBrightness as the
// production scale and the half decode is diag-only until a live VS capture
// confirms it.
static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            // subnormal
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400) == 0);
            mant &= 0x3FF;
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

static bool ExtractClusterPvlVbInner(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t clusterIndex,
    ZH_LbspClusterPvlVb* outVb)
{
    if (!outVb) return false;
    memset(outVb, 0, sizeof(*outVb));
    outVb->PvbIndex = -1;
    outVb->VbIndex  = -1;
    outVb->HdrScale = 1.0f;

    LbspResAndFixup r;
    if (!LoadLbspResourceAndFixup(cache, sbspTagId, r)) return false;

    const size_t cacheSz = cache->size;
    if (r.lbspMetaOff < 0) return false;
    const uint8_t* lbspMeta = cache->base + r.lbspMetaOff;

    // --- 1. clusters[clusterIndex] -> pervertex_block_index / offset ---
    LpLbspLayout LL = PickLbsp(cache->cacheType);
    if ((size_t)r.lbspMetaOff + (size_t)LL.OFF_CLUSTERS_BLOCK + 8 > cacheSz) return false;
    TagBlockRef clustersBlk = ReadTagBlock(lbspMeta + LL.OFF_CLUSTERS_BLOCK);
    if (clustersBlk.count <= 0 || clustersBlk.count > 0x10000) return true; // no PVL, Found=0
    if ((int32_t)clusterIndex >= clustersBlk.count) return true;            // OOB -> no PVL
    int64_t clustersOff = TagMetaFileOff(cache, clustersBlk.pointer);
    if (clustersOff < 0 ||
        (size_t)clustersOff + (size_t)clustersBlk.count * (size_t)LL.CLUSTER_DATA_BLOCK_SIZE > cacheSz)
        return true;
    const uint8_t* cluster = cache->base + clustersOff +
        (size_t)clusterIndex * (size_t)LL.CLUSTER_DATA_BLOCK_SIZE;
    int32_t pvbIdx = (int32_t)(int16_t)RU16(cluster + 0x02);
    int32_t pvbOff = R32(cluster + 0x04);
    outVb->PvbIndex  = pvbIdx;
    outVb->PvbOffset = pvbOff;
    if (pvbIdx < 0) return true; // cluster carries no PVL - engine-normal, Found=0

    // --- 2. bsp_per_vertex_run_time_data[pvbIdx] (+0x048, stride 4) ---
    //        { i16 vertex_buffer_index, i16 hdr_scale }
    constexpr int OFF_PV_RUNTIME = 0x48;
    if ((size_t)r.lbspMetaOff + OFF_PV_RUNTIME + 8 > cacheSz) return true;
    TagBlockRef rtBlk = ReadTagBlock(lbspMeta + OFF_PV_RUNTIME);
    if (rtBlk.count <= 0 || pvbIdx >= rtBlk.count) { if (getenv("ZH_PVLDIAG")) fprintf(stderr, "ZHPD cluster#%u pvbIdx=%d FAIL runtime-block count=%d\n", clusterIndex, pvbIdx, rtBlk.count); return true; }
    int64_t rtOff = TagMetaFileOff(cache, rtBlk.pointer);
    if (rtOff < 0 || (size_t)rtOff + (size_t)rtBlk.count * 4 > cacheSz) return true;
    const uint8_t* rtEntry = cache->base + rtOff + (size_t)pvbIdx * 4;
    int16_t vbi      = (int16_t)RU16(rtEntry + 0);
    int16_t hdrRaw   = (int16_t)RU16(rtEntry + 2);
    outVb->VbIndex     = vbi;
    outVb->HdrScaleRaw = (int32_t)hdrRaw;
    outVb->HdrScale    = HalfToFloat((uint16_t)hdrRaw);
    if (!(outVb->HdrScale > 0.0f) || !(outVb->HdrScale < 1e4f)) outVb->HdrScale = 1.0f;
    if (vbi < 0 || (size_t)vbi >= r.vbCounts.size()) { if (getenv("ZH_PVLDIAG")) fprintf(stderr, "ZHPD cluster#%u pvbIdx=%d FAIL unbound vbi=%d pool=%zu rtcount=%d hdr=%g\n", clusterIndex, pvbIdx, (int)vbi, r.vbCounts.size(), rtBlk.count, outVb->HdrScale); return true; } // unbound slot -> no PVL

    // --- 3. lbsp_vb_pool[vbi] = the stride-4 PVL ByteAddressBuffer ---
    uint32_t cnt  = r.vbCounts[(size_t)vbi];
    uint32_t len  = r.vbLens[(size_t)vbi];
    uint32_t foff = r.vbFixupOffsets[(size_t)vbi];
    if (cnt == 0 || len != cnt * 4) {
        static std::atomic<int> s_pvlSizeDiag{ 16 };
        int v = s_pvlSizeDiag.load(std::memory_order_relaxed);
        if (v > 0 && s_pvlSizeDiag.compare_exchange_weak(v, v - 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            NativeDiag("LbspClusterPvlVb[sbsp=0x%04x] cluster#%u pvbIdx=%d vbi=%d "
                       "stride!=4 (cnt=%u len=%u) - skipping",
                       (unsigned)sbspTagId, clusterIndex, pvbIdx, (int)vbi, cnt, len);
        if (getenv("ZH_PVLDIAG")) fprintf(stderr, "ZHPD cluster#%u pvbIdx=%d vbi=%d FAIL stride cnt=%u len=%u\n", clusterIndex, pvbIdx, (int)vbi, cnt, len);
        return true; // not a stride-4 VB -> treat as no PVL rather than rainbow
    }
    if ((size_t)foff + (size_t)len > r.resSize) { if (getenv("ZH_PVLDIAG")) fprintf(stderr, "ZHPD cluster#%u pvbIdx=%d vbi=%d FAIL fixup foff=%u len=%u resSize=%zu\n", clusterIndex, pvbIdx, (int)vbi, foff, len, r.resSize); return true; }

    uint8_t* buf = (uint8_t*)malloc(len);
    if (!buf) return false;
    memcpy(buf, r.resData + foff, len);
    if (getenv("ZH_PVLDIAG")) fprintf(stderr, "ZHPD cluster#%u pvbIdx=%d vbi=%d OK cnt=%u len=%u foff=%u\n", clusterIndex, pvbIdx, (int)vbi, cnt, len, foff);
    outVb->Found     = 1;
    outVb->Bytes     = buf;
    outVb->Len       = len;
    outVb->ElemCount = (int32_t)cnt;

    static std::atomic<int> s_pvlOkDiag{ 24 };
    int v = s_pvlOkDiag.load(std::memory_order_relaxed);
    if (v > 0 && s_pvlOkDiag.compare_exchange_weak(v, v - 1,
            std::memory_order_relaxed, std::memory_order_relaxed))
        NativeDiag("LbspClusterPvlVb[sbsp=0x%04x] cluster#%u OK pvbIdx=%d vbi=%d "
                   "cnt=%u len=%u pvbOff=%d hdrRaw=%d hdrHalf=%.4f",
                   (unsigned)sbspTagId, clusterIndex, pvbIdx, (int)vbi,
                   cnt, len, pvbOff, (int)hdrRaw, outVb->HdrScale);
    return true;
}

static bool SehExtractClusterPvlVb(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t clusterIndex,
    ZH_LbspClusterPvlVb* outVb)
{
    __try {
        return ExtractClusterPvlVbInner(cache, sbspTagId, clusterIndex, outVb);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outVb && outVb->Bytes) { free(outVb->Bytes); }
        if (outVb) { memset(outVb, 0, sizeof(*outVb)); outVb->PvbIndex = -1; outVb->VbIndex = -1; outVb->HdrScale = 1.0f; }
        return false;
    }
}

// PER-INSTANCE PVL - the instanced-geometry (rocks/cliffs) twin of
// ExtractClusterPvlVbInner. Reads the per-instance record at Lbsp+0x60
// (scenario_lightmap_instance_data, stride 12: i16 lpti@0, i16 pervertex_block_
// index@2, i16 probe_block_index@4, i32 analytical@8), then routes the
// pervertex_block_index through the SAME +0x48 run-time table + lbsp_vb_pool the
// cluster path uses. Instances get their OWN VB -> pvbOffset is always 0. Returns
// the same ZH_LbspClusterPvlVb the cluster path returns so the viewer decodes it
// identically (decode_pvl_tint with pvb_offset=0). See docs/hrek_re/06.
static bool ExtractInstancePvlVbInner(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t instanceOrdinal,
    ZH_LbspClusterPvlVb* outVb)
{
    if (!outVb) return false;
    memset(outVb, 0, sizeof(*outVb));
    outVb->PvbIndex = -1;
    outVb->VbIndex  = -1;
    outVb->HdrScale = 1.0f;

    LbspResAndFixup r;
    if (!LoadLbspResourceAndFixup(cache, sbspTagId, r)) return false;
    const size_t cacheSz = cache->size;
    if (r.lbspMetaOff < 0) return false;
    const uint8_t* lbspMeta = cache->base + r.lbspMetaOff;

    // --- 1. instances[instanceOrdinal] (Lbsp+0x60, stride 12) ---
    constexpr int OFF_INSTANCES_BLOCK = 0x60;
    constexpr int INSTANCE_DATA_BLOCK_SIZE = 12;
    if ((size_t)r.lbspMetaOff + OFF_INSTANCES_BLOCK + 8 > cacheSz) return false;
    TagBlockRef instBlk = ReadTagBlock(lbspMeta + OFF_INSTANCES_BLOCK);
    if (instBlk.count <= 0 || instBlk.count > 0x40000) return true; // no instance LM
    if ((int32_t)instanceOrdinal >= instBlk.count) return true;
    int64_t instOff = TagMetaFileOff(cache, instBlk.pointer);
    if (instOff < 0 ||
        (size_t)instOff + (size_t)instBlk.count * (size_t)INSTANCE_DATA_BLOCK_SIZE > cacheSz)
        return true;
    const uint8_t* inst = cache->base + instOff +
        (size_t)instanceOrdinal * (size_t)INSTANCE_DATA_BLOCK_SIZE;
    int32_t pvbIdx = (int32_t)(int16_t)RU16(inst + 0x02); // pervertex_block_index
    outVb->PvbIndex  = pvbIdx;
    outVb->PvbOffset = 0; // instances own their VB - start at vertex 0
    if (getenv("ZH_TIERDIAG")) {
        int32_t ppIdx = (int32_t)(int16_t)RU16(inst + 0x00);   // lightprobe_texture_array_index (per_pixel)
        int32_t probeIdx = (int32_t)(int16_t)RU16(inst + 0x04); // probe_block_index (single_probe)
        fprintf(stderr, "ZH_TIERDIAG ord=%u pp=%d pv=%d probe=%d\n", instanceOrdinal, ppIdx, pvbIdx, probeIdx);
    }
    if (pvbIdx < 0) return true; // single-probe / per-pixel / bogus -> no per-vertex PVL

    // --- 2. bsp_per_vertex_run_time_data[pvbIdx] (+0x048, stride 4) ---
    constexpr int OFF_PV_RUNTIME = 0x48;
    if ((size_t)r.lbspMetaOff + OFF_PV_RUNTIME + 8 > cacheSz) return true;
    TagBlockRef rtBlk = ReadTagBlock(lbspMeta + OFF_PV_RUNTIME);
    if (rtBlk.count <= 0 || pvbIdx >= rtBlk.count) return true;
    int64_t rtOff = TagMetaFileOff(cache, rtBlk.pointer);
    if (rtOff < 0 || (size_t)rtOff + (size_t)rtBlk.count * 4 > cacheSz) return true;
    const uint8_t* rtEntry = cache->base + rtOff + (size_t)pvbIdx * 4;
    int16_t vbi    = (int16_t)RU16(rtEntry + 0);
    int16_t hdrRaw = (int16_t)RU16(rtEntry + 2);
    outVb->VbIndex     = vbi;
    outVb->HdrScaleRaw = (int32_t)hdrRaw;
    outVb->HdrScale    = HalfToFloat((uint16_t)hdrRaw);
    if (!(outVb->HdrScale > 0.0f) || !(outVb->HdrScale < 1e4f)) outVb->HdrScale = 1.0f;
    if (vbi < 0 || (size_t)vbi >= r.vbCounts.size()) return true;

    // --- 3. lbsp_vb_pool[vbi] = stride-4 PVL ByteAddressBuffer ---
    uint32_t cnt  = r.vbCounts[(size_t)vbi];
    uint32_t len  = r.vbLens[(size_t)vbi];
    uint32_t foff = r.vbFixupOffsets[(size_t)vbi];
    if (cnt == 0 || len != cnt * 4) return true;
    if ((size_t)foff + (size_t)len > r.resSize) return true;
    uint8_t* buf = (uint8_t*)malloc(len);
    if (!buf) return false;
    memcpy(buf, r.resData + foff, len);
    outVb->Found     = 1;
    outVb->Bytes     = buf;
    outVb->Len       = len;
    outVb->ElemCount = (int32_t)cnt;
    return true;
}

static bool SehExtractInstancePvlVb(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t instanceOrdinal,
    ZH_LbspClusterPvlVb* outVb)
{
    __try {
        return ExtractInstancePvlVbInner(cache, sbspTagId, instanceOrdinal, outVb);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outVb && outVb->Bytes) { free(outVb->Bytes); }
        if (outVb) { memset(outVb, 0, sizeof(*outVb)); outVb->PvbIndex = -1; outVb->VbIndex = -1; outVb->HdrScale = 1.0f; }
        return false;
    }
}

// Per-instance PVL VB (rocks/cliffs). Same output shape + free path as
// ZH_LBSP_GetClusterPvlVb. Found==0 (return true) = this instance has no
// per-vertex baked lighting (single-probe/per-pixel/bogus) -> caller falls back.
extern "C" __declspec(dllexport) bool __stdcall ZH_LBSP_GetInstancePvl(
    uint64_t cacheHandle, uint32_t sbspTagId, uint32_t instanceOrdinal,
    ZH_LbspClusterPvlVb* outVb)
{
    if (outVb) { memset(outVb, 0, sizeof(*outVb)); outVb->PvbIndex = -1; outVb->VbIndex = -1; outVb->HdrScale = 1.0f; }
    if (!outVb) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;
    return SehExtractInstancePvlVb(cache, sbspTagId, instanceOrdinal, outVb);
}

// =============================================================================
// SH lighting-point grid parser - airprobes block (lbsp+0x12C).
//
// DECO_COLOR_RE: the engine derives the decorator per-instance
// baked `instance_color` in `structure_bsp_light_decorators_from_scenario` by
// sampling the BSP SH/airprobe lighting-point grid at each placement position.
// The on-disk grid lives in the Lbsp's `airprobes` block at +0x12C; each entry
// (`scenario_lightmap_airprobe_value`, sizeof 56) carries an EXPLICIT world
// position + the `half_dual_vmf_lightprobe_with_analytical_light_index` SH
// payload (the SAME dual-VMF format MMS already decodes for the per-pixel
// lightmap atlas - see LIGHTMAP_VMF_RE_FROM_HREK.md). Layout (authoritative,
// lbsp_reach.json scenario_lightmap_airprobe_value, GUID 4a1d8551...):
//
//   +0x00  real_point_3d  airprobe position (world XYZ, 12B)
//   +0x0C  string_id      airprobe name (4B)
//   +0x10  int16          manual bsp flags
//   +0x12  pad            (2B)
//   +0x14  dual_vmf[16]   vmf terms - 16 int16 = 16 half-floats (32B)
//   +0x34  int32          analytical light index (4B)
//   = 56 bytes
//
// The 16 halfs map to vmf_coefficients[4] (4x vec4 half - lightmap_sampling.
// hlsl_include:9-23 / spherical_harmonics.hlsl_include:68-80):
//   half[0..2]  = dominant lobe direction (unit)      half[3]  = analytical mask
//   half[4..6]  = dominant lobe HDR color (rgb)        half[7]  = bandwidth (vMF kappa)
//   half[8..10] = fill lobe direction                  half[11] = cloud mask
//   half[12..14]= fill lobe HDR color (rgb)            half[15] = fill bandwidth
//
// We evaluate the engine's `dual_vmf_diffuse` at a sample direction N (decorator
// ground pattern -> world-up) to produce a LINEAR HDR ambient color:
//   dom_response = pow(saturate(dot(domDir, N) * 0.5 + 0.5), bandwidthExp)
//   color = dom_response * domColor + 0.25 * fillColor      (per /pi normalized)
// (We fold the 1/pi into the engine's downstream exposure; the result is a
// linear HDR tint matching `instance_color.rgb` before the RGBE exp2 encode.)
// =============================================================================
namespace {

#pragma pack(push, 1)
struct ZH_AirprobePoint {
    float pos[3];       // world XYZ
    float ambient[3];   // LINEAR HDR ambient (dual-VMF evaluated up-facing)
    // OBJECT_PROBE_SH: directional terms for per-object
    // directional lighting. APPENDED after pos/ambient so the decorator path
    // (which only reads pos/ambient) is byte-unaffected. domDir = dominant lobe
    // direction (unit, toward peak incoming radiance); mask = analytical
    // visibility [0,1] (1=fully sun-lit); dirWeight = directional fraction of the
    // ambient [0,0.6] (0=isotropic -> no directional change in the shader).
    float domDir[3];
    float mask;
    float dirWeight;
    // OBJ-PROBE (engine per-object lighting): the raw dual-vMF lobes so the renderer can evaluate
    // irradiance(N) = [LUT(N.domDir, bandwidth) * domRgb + 0.25 * fillRgb] / pi per PIXEL (doc 00 s4).
    float domRgb[3];
    float fillRgb[3];
    float bandwidth;
    // obj-light: FILL lobe direction (terms 8..10, unit or zero). The engine's airprobe blend
    // (sub_1406BAEA0) sums it per probe and the lobes+sun merge (sub_140828CD0) mixes it into the
    // object's analytical light direction. Appended -> record is 84 bytes.
    float fillDir[3];
};
#pragma pack(pop)
static_assert(sizeof(ZH_AirprobePoint) == 84, "ZH_AirprobePoint layout drift");

constexpr int OFF_AIRPROBES_BLOCK        = 0x12C;
constexpr int AIRPROBE_ELEMENT_SIZE      = 56;   // sizeof scenario_lightmap_airprobe_value
constexpr int AIRPROBE_OFF_POSITION      = 0x00;
constexpr int AIRPROBE_OFF_VMF_TERMS     = 0x14; // 16 int16 = dual_vmf[16]
constexpr int AIRPROBE_MAX_COUNT_SANITY  = 0x40000;

// Evaluate the engine dual-VMF lobe at sample direction N (unit). Returns a
// linear HDR rgb ambient. Mirrors spherical_harmonics.hlsl_include:68-80 with
// the kappa=1 analytic stand-in for the `g_sample_vmf_diffuse` LUT (the LUT is the
// hemispherical cosine integral of the vMF lobe; for the per-probe bandwidth we
// approximate it with a normalized half-Lambert^k response, exactly the form
// NativeBspMeshAdapter.cs already uses for the per-vertex VMF dominant lobe).
static void EvalDualVmf(const uint16_t* terms, const float N[3], float outRgb[3],
                        float outDomDir[3] = nullptr, float* outMask = nullptr,
                        float* outDirWeight = nullptr)
{
    // Decode the 16 half-floats into the 4x vec4 coefficient table.
    float c[16];
    for (int i = 0; i < 16; ++i) c[i] = HalfToFloat(terms[i]);

    float domDir[3]  = { c[0],  c[1],  c[2]  };
    float analyticalMask = c[3];
    float domRgb[3]  = { c[4],  c[5],  c[6]  };
    float bandwidth  = c[7];
    float fillRgb[3] = { c[12], c[13], c[14] };

    // Normalize the dominant direction (authored unit, guard against junk).
    float dl = domDir[0]*domDir[0] + domDir[1]*domDir[1] + domDir[2]*domDir[2];
    if (dl > 1e-8f) { float inv = 1.0f / sqrtf(dl); domDir[0]*=inv; domDir[1]*=inv; domDir[2]*=inv; }
    else { domDir[0]=0; domDir[1]=0; domDir[2]=1; }

    // bandwidth (vMF kappa) shapes the lobe sharpness. `k` (guarded) is retained only for the
    // directional-fraction heuristic (outDirWeight) below.
    float k = bandwidth;
    if (!(k > 0.25f && k < 64.0f) || !std::isfinite(k)) k = 1.0f;

    float ndl = domDir[0]*N[0] + domDir[1]*N[1] + domDir[2]*N[2];
    // ENGINE-EXACT dual_vmf_diffuse (spherical_harmonics.hlsl_include:55-80):
    // coeff_dom from the g_sample_vmf_diffuse LUT (NOT the analytic half-Lambert), fill = 0.25 hard
    // literal, whole thing / pi. Now consistent with the atlas/PVL paths (both absolute-HDR, / pi).
    float lutBw = std::isfinite(bandwidth) ? bandwidth : 1.0f;
    float coeffDom = VmfDiffuseCoeff(ndl, lutBw);
    const float invPi = 0.31830988618f;
    for (int j = 0; j < 3; ++j) {
        float v = (coeffDom * domRgb[j] + 0.25f * fillRgb[j]) * invPi;
        if (!std::isfinite(v) || v < 0.0f) v = 0.0f;
        outRgb[j] = v;
    }

    // OBJECT_PROBE_SH (T1-5): directional terms for per-object lighting.
    if (outDomDir) { outDomDir[0]=domDir[0]; outDomDir[1]=domDir[1]; outDomDir[2]=domDir[2]; }
    if (outMask) {
        float m = analyticalMask;
        if (!std::isfinite(m)) m = 1.0f;
        if (m < 0.0f) m = 0.0f; if (m > 1.0f) m = 1.0f;
        *outMask = m;
    }
    if (outDirWeight) {
        // Directional fraction = dominant-lobe energy / (dominant + fill) at the
        // peak. domLuma uses the peak response ((k+1)*0.25), fillLuma the constant
        // 0.25 integral. Bounded to [0,0.6] so the shader's mean-preserving
        // lerp(1, 0.5+halfL, w) stays a modest +-30% (a single junk probe cannot
        // black out or blow up an object).
        auto luma = [](const float r[3]) { return 0.299f*r[0] + 0.587f*r[1] + 0.114f*r[2]; };
        float domPeak = (k + 1.0f) * 0.25f;
        float domLuma = luma(domRgb) * domPeak;
        float fillLuma = luma(fillRgb) * 0.25f;
        if (domLuma < 0.0f || !std::isfinite(domLuma)) domLuma = 0.0f;
        if (fillLuma < 0.0f || !std::isfinite(fillLuma)) fillLuma = 0.0f;
        float denom = domLuma + fillLuma;
        float w = (denom > 1e-6f) ? (domLuma / denom) : 0.0f;
        if (w < 0.0f) w = 0.0f; if (w > 0.6f) w = 0.6f;
        *outDirWeight = w;
    }
}

// Per-instance SINGLE-PROBE baked lighting (rocks/cliffs baked with the engine's
// _connected_geometry_poop_lighting_single_probe policy). ~18% of Forge instances use
// this tier; ZH_LBSP_GetInstancePvl handles only the per-vertex tier (returns Found=0
// here); without this they would fall through to the flat airprobe-CENTROID
// ambient (a directionless region guess -> too dark). This reads the instance's OWN
// baked probe: instances[ord].probe_block_index (+0x04) -> Lbsp+0x6C probes[] (36B,
// dual-VMF terms @ +0x00) and evaluates the up-facing dual-VMF ambient (linear HDR) via
// the SAME EvalDualVmf used for airprobes/per-vertex. Layout: docs/hrek_re/06 section 2.3.
static bool ExtractInstanceProbeInner(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t instanceOrdinal, float outRgb[3])
{
    outRgb[0] = outRgb[1] = outRgb[2] = 0.0f;
    LbspResAndFixup r;
    if (!LoadLbspResourceAndFixup(cache, sbspTagId, r)) return false;
    const size_t cacheSz = cache->size;
    if (r.lbspMetaOff < 0) return false;
    const uint8_t* lbspMeta = cache->base + r.lbspMetaOff;

    // instances[ord] (Lbsp+0x60, stride 12): probe_block_index @ +0x04
    constexpr int OFF_INSTANCES_BLOCK = 0x60;
    constexpr int INSTANCE_DATA_BLOCK_SIZE = 12;
    if ((size_t)r.lbspMetaOff + OFF_INSTANCES_BLOCK + 8 > cacheSz) return false;
    TagBlockRef instBlk = ReadTagBlock(lbspMeta + OFF_INSTANCES_BLOCK);
    if (instBlk.count <= 0 || instBlk.count > 0x40000) return false;
    if ((int32_t)instanceOrdinal >= instBlk.count) return false;
    int64_t instOff = TagMetaFileOff(cache, instBlk.pointer);
    if (instOff < 0 ||
        (size_t)instOff + (size_t)instBlk.count * (size_t)INSTANCE_DATA_BLOCK_SIZE > cacheSz)
        return false;
    const uint8_t* inst = cache->base + instOff +
        (size_t)instanceOrdinal * (size_t)INSTANCE_DATA_BLOCK_SIZE;
    int32_t probeIdx = (int32_t)(int16_t)RU16(inst + 0x04); // probe_block_index
    if (probeIdx < 0) return false; // not single-probe (per-vertex/per-pixel/bogus)

    // probes[probeIdx] (Lbsp+0x6C, stride 36): dual_vmf terms[16 half] @ +0x00
    constexpr int OFF_PROBES_BLOCK = 0x6C;
    constexpr int PROBE_ELEMENT_SIZE = 36;
    if ((size_t)r.lbspMetaOff + OFF_PROBES_BLOCK + 8 > cacheSz) return false;
    TagBlockRef probeBlk = ReadTagBlock(lbspMeta + OFF_PROBES_BLOCK);
    if (probeBlk.count <= 0 || probeIdx >= probeBlk.count) return false;
    int64_t probeOff = TagMetaFileOff(cache, probeBlk.pointer);
    if (probeOff < 0 ||
        (size_t)probeOff + (size_t)probeBlk.count * (size_t)PROBE_ELEMENT_SIZE > cacheSz)
        return false;
    const uint8_t* probe = cache->base + probeOff + (size_t)probeIdx * PROBE_ELEMENT_SIZE;
    uint16_t terms[16];
    memcpy(terms, probe + 0x00, 32);
    const float worldUp[3] = { 0.0f, 0.0f, 1.0f };
    EvalDualVmf(terms, worldUp, outRgb);   // linear HDR up-facing ambient
    return true;
}

static bool SehExtractInstanceProbe(
    CacheHandle* cache, uint32_t sbspTagId, uint32_t instanceOrdinal, float outRgb[3])
{
    __try { return ExtractInstanceProbeInner(cache, sbspTagId, instanceOrdinal, outRgb); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outRgb) { outRgb[0] = outRgb[1] = outRgb[2] = 0.0f; }
        return false;
    }
}

// Returns true + fills outRgb (linear HDR) if this instance uses the single-probe
// lighting tier; false (outRgb=0) otherwise. Caller then tones it the same way as the
// airprobe fallback it replaces (per-instance flat tint), but with the CORRECT baked value.
extern "C" __declspec(dllexport) bool __stdcall ZH_LBSP_GetInstanceProbe(
    uint64_t cacheHandle, uint32_t sbspTagId, uint32_t instanceOrdinal, float* outRgb)
{
    if (outRgb) { outRgb[0] = outRgb[1] = outRgb[2] = 0.0f; }
    if (!outRgb) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;
    return SehExtractInstanceProbe(cache, sbspTagId, instanceOrdinal, outRgb);
}

// Walk the airprobes block and decode each point's world position + up-facing
// dual-VMF HDR ambient. Caller frees *outBuf. Returns false on hard failure.
bool ExtractAirprobeGridInner(
    CacheHandle* cache, uint32_t sbspTagId,
    ZH_AirprobePoint** outBuf, uint32_t* outCount)
{
    *outBuf = nullptr;
    *outCount = 0;
    if (!cache) return false;

    int32_t unused = 0;
    uint32_t lbspId = SehResolveActiveLbsp(cache, sbspTagId, &unused);
    if (lbspId == 0xFFFFFFFFu || lbspId >= cache->tags.size()) {
        NativeDiag("AirprobeGrid[sbsp=0x%X]: ResolveActiveLbsp failed", sbspTagId);
        return false;
    }
    int64_t lbspMetaOff = TagMetaFileOff(cache, cache->tags[lbspId].metaPointerRaw);
    if (lbspMetaOff < 0) return false;
    if ((size_t)lbspMetaOff + (size_t)OFF_AIRPROBES_BLOCK + 12 > cache->size) return false;

    const uint8_t* lbspMeta = cache->base + lbspMetaOff;
    TagBlockRef blk = ReadTagBlock(lbspMeta + OFF_AIRPROBES_BLOCK);
    if (blk.count <= 0) {
        NativeDiag("AirprobeGrid[sbsp=0x%X lbsp=0x%X]: airprobes count=%d (empty)",
                   sbspTagId, lbspId, blk.count);
        return true;  // valid empty result
    }
    if (blk.count > AIRPROBE_MAX_COUNT_SANITY) {
        NativeDiag("AirprobeGrid[sbsp=0x%X]: airprobes count insane (%d)", sbspTagId, blk.count);
        return false;
    }
    int64_t arrOff = TagMetaFileOff(cache, blk.pointer);
    if (arrOff < 0) return false;
    if ((size_t)arrOff + (size_t)blk.count * AIRPROBE_ELEMENT_SIZE > cache->size) {
        NativeDiag("AirprobeGrid[sbsp=0x%X]: airprobes array OOB (count=%d)", sbspTagId, blk.count);
        return false;
    }

    auto* buf = (ZH_AirprobePoint*)malloc((size_t)blk.count * sizeof(ZH_AirprobePoint));
    if (!buf) return false;

    const float worldUp[3] = { 0.0f, 0.0f, 1.0f };   // decorator ground sample pattern
    const uint8_t* base = cache->base + arrOff;
    int diagBudget = 4;
    for (int i = 0; i < blk.count; ++i) {
        const uint8_t* e = base + (size_t)i * AIRPROBE_ELEMENT_SIZE;
        memcpy(buf[i].pos, e + AIRPROBE_OFF_POSITION, 12);
        uint16_t terms[16];
        memcpy(terms, e + AIRPROBE_OFF_VMF_TERMS, 32);
        // OBJECT_PROBE_SH (T1-5): the up-facing ambient feeds the decorator path
        // (unchanged); the appended domDir/mask/dirWeight feed per-object directional
        // lighting. All decoded from the same dual-VMF terms in one pass.
        EvalDualVmf(terms, worldUp, buf[i].ambient,
                    buf[i].domDir, &buf[i].mask, &buf[i].dirWeight);
        {   // OBJ-PROBE raw lobes
            float bw = HalfToFloat(terms[7]);
            if (!(bw > 0.25f && bw < 64.0f) || !std::isfinite(bw)) bw = 1.0f;
            buf[i].bandwidth = bw;
            for (int j = 0; j < 3; ++j) {
                float dv = HalfToFloat(terms[4 + j]), fv = HalfToFloat(terms[12 + j]);
                buf[i].domRgb[j]  = (std::isfinite(dv) && dv > 0.0f) ? dv : 0.0f;
                buf[i].fillRgb[j] = std::isfinite(fv) ? fv : 0.0f;   // signed: engine 0.25*fill may be negative
            }
            // obj-light: fill lobe direction (terms 8..10), normalised; zero when absent/junk.
            float fd[3] = { HalfToFloat(terms[8]), HalfToFloat(terms[9]), HalfToFloat(terms[10]) };
            float fl = 0.0f;
            for (int j = 0; j < 3; ++j) { if (!std::isfinite(fd[j])) fd[j] = 0.0f; fl += fd[j] * fd[j]; }
            if (fl > 1e-8f) { float inv = 1.0f / sqrtf(fl); for (int j = 0; j < 3; ++j) buf[i].fillDir[j] = fd[j] * inv; }
            else { buf[i].fillDir[0] = buf[i].fillDir[1] = buf[i].fillDir[2] = 0.0f; }
        }
        if (diagBudget-- > 0) {
            // OBJECT_PROBE_SH (T1-5): also dump the directional terms so the decode
            // is verifiable. domDir should trend toward +Z (sky-dominant) for open
            // probes; mask <1 in occluded pockets; dirWeight in [0,0.6].
            // #220 aftship: ALSO dump the raw dominant + fill lobe COLOURS (terms 4..6 /
            // 12..14) to confirm the interior dominant is purple (R,B >> G).
            float dR = HalfToFloat(terms[4]),  dG = HalfToFloat(terms[5]),  dB = HalfToFloat(terms[6]);
            float fR = HalfToFloat(terms[12]), fG = HalfToFloat(terms[13]), fB = HalfToFloat(terms[14]);
            NativeDiag("AirprobeGrid[sbsp=0x%X] pt[%d] pos=(%.2f,%.2f,%.2f) "
                       "ambRGB=(%.4f,%.4f,%.4f) domDir=(%.3f,%.3f,%.3f) mask=%.3f dirW=%.3f "
                       "domRGB=(%.4f,%.4f,%.4f) fillRGB=(%.4f,%.4f,%.4f)",
                       sbspTagId, i,
                       buf[i].pos[0], buf[i].pos[1], buf[i].pos[2],
                       buf[i].ambient[0], buf[i].ambient[1], buf[i].ambient[2],
                       buf[i].domDir[0], buf[i].domDir[1], buf[i].domDir[2],
                       buf[i].mask, buf[i].dirWeight, dR, dG, dB, fR, fG, fB);
        }
    }
    NativeDiag("AirprobeGrid[sbsp=0x%X lbsp=0x%X]: DONE airprobes=%d", sbspTagId, lbspId, blk.count);
    *outBuf = buf;
    *outCount = (uint32_t)blk.count;
    return true;
}

bool SehExtractAirprobeGrid(
    CacheHandle* cache, uint32_t sbspTagId,
    ZH_AirprobePoint** outBuf, uint32_t* outCount)
{
    __try { return ExtractAirprobeGridInner(cache, sbspTagId, outBuf, outCount); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outBuf && *outBuf) { free(*outBuf); *outBuf = nullptr; }
        if (outCount) *outCount = 0;
        NativeDiag("AirprobeGrid[sbsp=0x%X]: SEH fault", sbspTagId);
        return false;
    }
}

} // anonymous namespace (airprobe SH grid)

} // anonymous namespace

// =============================================================================
// Public exports - lightprobe / PVL
// =============================================================================

// Schema-authoritative per-CLUSTER PVL VB lookup (a mesh-slot heuristic
// produces rainbow garbage instead). Caller keys by the BSP's
// owning cluster index (ZH_BspMesh.LightmapClusterIndex). On success
// outVb->Found==1 and outVb->Bytes is a malloc'd stride-4 ByteAddressBuffer
// (free via ZH_LBSP_FreeLightprobeBuffer). Found==0 with return true means
// "this cluster legitimately carries no per-vertex lightprobe data" (the
// engine-normal case for most clusters); caller falls back to flat/unlit.
extern "C" __declspec(dllexport) bool __stdcall ZH_LBSP_GetClusterPvlVb(
    uint64_t cacheHandle, uint32_t sbspTagId, uint32_t clusterIndex,
    ZH_LbspClusterPvlVb* outVb)
{
    if (outVb) { memset(outVb, 0, sizeof(*outVb)); outVb->PvbIndex = -1; outVb->VbIndex = -1; outVb->HdrScale = 1.0f; }
    if (!outVb) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;
    return SehExtractClusterPvlVb(cache, sbspTagId, clusterIndex, outVb);
}

extern "C" __declspec(dllexport) void __stdcall ZH_LBSP_FreeLightprobeBuffer(uint8_t* buf)
{
    if (buf) free(buf);
}

// =============================================================================
// SH lighting-point grid export - ZH_LBSP_GetAirprobeGrid
//
// DECO_COLOR_RE: returns the BSP airprobe SH lighting-point grid
// for `sbspTagId` - a malloc'd flat array of ZH_AirprobePoint{ float pos[3];
// float ambient[3]; } (24B each). `pos` is world XYZ; `ambient` is the LINEAR
// HDR ambient color from the up-facing dual-VMF evaluation. The decorator
// color reconstruction samples the NEAREST point to each blade's world position
// to recover the engine's baked `instance_color`. Caller frees via
// ZH_LBSP_FreeAirprobeGrid. Returns 1 on success (count may be 0 - "no airprobe
// grid on this BSP"), 0 on hard failure.
// =============================================================================
extern "C" __declspec(dllexport) int __stdcall ZH_LBSP_GetAirprobeGrid(
    uint64_t cacheHandle, uint32_t sbspTagId,
    ZH_AirprobePoint** outBuf, uint32_t* outCount)
{
    if (outBuf)  *outBuf = nullptr;
    if (outCount) *outCount = 0;
    if (!outBuf || !outCount) return 0;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    return SehExtractAirprobeGrid(cache, sbspTagId, outBuf, outCount) ? 1 : 0;
}

extern "C" __declspec(dllexport) void __stdcall ZH_LBSP_FreeAirprobeGrid(ZH_AirprobePoint* buf)
{
    if (buf) free(buf);
}
