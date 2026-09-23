// PlanarFogWalker.cpp
// =============================================================================
// Native walker for planar fog volumes authored in the structure_design (sddt)
// tag, referenced by the scenario via scnr.StructureDesigns[].
//
// Tag chain (Reach MCC):
//
//   scnr (scenario)
//     Structure Designs          @ scnr + 0x5C  (elementSize 0x20)
//       +0x00  tagRef(16)  -> sddt (structure_design)
//       +0x10  tagRef(16)  -> unknown (ignored)
//
//   scnr.Fog palette            @ scnr + 0x534  (elementSize 0x18)
//       +0x00  stringId          Name
//       +0x04  int16             Unknown
//       +0x06  int16             Unknown
//       +0x08  tagRef(16)  -> fogg (the fog appearance tag)
//
//   sddt (structure_design, baseSize 0x160)
//     Planar Fog                 @ sddt + 0x70  (elementSize 0x3C)
//       +0x00  stringId          Name
//       +0x04  tagRef(16)        Appearance Settings -> fogg tag
//       +0x14  tagblock(12)      Vertices (elementSize 0xC: float3 Position)
//       +0x20  tagblock(12)      Triangles
//                 -> sub-block Planes (elementSize 0x10: plane3 = float4 ABCD)
//       +0x2C  float             Depth (thickness of the fog volume below plane)
//       +0x30  float3            Normal (world-space up-vector of the fog plane)
//
//   fogg tag (fog appearance settings) - decoded field offsets from
//   FogParser.cpp / SAPIEN_FOG_NOTES.md. The fields differ between
//   atmospheric fog (screen-space) and planar fog appearance; for the v1
//   viewer we read the same inscatter color A at FOGG_INSCATTER_A_OFF
//   and density at FOGG_DENSITY_OFF that FogParser already uses.
//
//   scnr.ScenarioClusterData[].Fog[] maps per-cluster fog index back to the
//   scnr.Fog palette. This tells us which clusters are "inside" a fog zone.
//   For the viewer's v1 we render every planar fog volume as a visible plane
//   regardless of cluster membership - it's always correct to show them.
//
// Schema reference: Assembly ReachMCC sddt.xml + scnr.xml (Lord Zedd plugins).
//
// Output ABI: flat array of ZH_PlanarFogVolume via the public export
// ZH_PFOG_Enumerate(cacheHandle, scnrTagId, ...). The caller frees with
// ZH_PFOG_FreeBuffer.
//
// Defensive contract (same as DecalWalker / FogParser):
//   * SEH-wrapped at the public boundary.
//   * Class-code checks at every cross-tag dereference.
//   * Index validation against cache->tags.size().
//   * Hard cap on total fog volume count (PLANAR_FOG_CAP).
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <vector>

using namespace zh_mcc;

// ---------------------------------------------------------------------------
// Public ABI struct - must match the Rust mirror of ZH_PlanarFogVolume in crates/hms-native/src/lib.rs
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct ZH_PlanarFogVolume {
    float    Normal[3];         // plane normal (world-space)
    float    PlaneD;            // plane equation D: dot(Normal, P) + D = 0
    float    Depth;             // fog thickness below the plane
    float    Color[4];          // RGBA - from fogg inscatter A, alpha = density
    float    Density;           // fog density / murkiness
    // Bounding box of the fog volume vertices (for culling / quad sizing)
    float    BoundsMin[3];
    float    BoundsMax[3];
    // Centroid of the fog volume vertices
    float    Centroid[3];
    // Number of vertices in the polygon outline (for debug / future use)
    uint32_t VertexCount;
    uint32_t _pad[3];
};
#pragma pack(pop)

namespace {

// Hard cap to prevent runaway allocations on busted tags.
constexpr int PLANAR_FOG_CAP = 64;

// Tag class codes
constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_SDDT = "sddt";
constexpr const char* TC_FOGG = "fogg";

// scnr.StructureDesigns tagblock offset
int PickStructureDesignsOffset(CacheType ct) {
    switch (ct) {
        case CacheType::MccHaloReachU13: return 0x5C;
        default:                          return 0x58;  // pre-U13 may be 4 bytes earlier
    }
}

// sddt.PlanarFog tagblock offset
constexpr int SDDT_PLANAR_FOG_OFF = 0x70;
constexpr int SDDT_PLANAR_FOG_ENTRY_SIZE = 0x3C;

// sddt.WaterInstances tagblock offset (for water fog color/murkiness)
constexpr int SDDT_WATER_INSTANCES_OFF = 0x64;
constexpr int SDDT_WATER_INSTANCE_SIZE = 0x54;

// fogg field offsets (from FogParser.cpp)
constexpr int FOGG_INSCATTER_A_OFF = 0x1C;   // sky_fog_color RGB (float3)
constexpr int FOGG_DENSITY_OFF     = 0x14;   // sky_fog_thickness (0.0-1.0)

// TagReference: ClassId @ +0, padding @ +4..11, TagId @ +12
int32_t ReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// Read fogg tag for appearance parameters. Returns false on any failure.
bool ReadFoggAppearance(CacheHandle* cache, int32_t foggTagId,
                        float outColor[4], float* outDensity)
{
    if (foggTagId < 0 || (uint32_t)foggTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[foggTagId];
    if (memcmp(te.classCode, TC_FOGG, 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + FOGG_INSCATTER_A_OFF + 12 > cache->size) return false;

    const uint8_t* m = cache->base + metaOff;
    // Read inscatter color A (3 floats)
    memcpy(&outColor[0], m + FOGG_INSCATTER_A_OFF + 0, 4);
    memcpy(&outColor[1], m + FOGG_INSCATTER_A_OFF + 4, 4);
    memcpy(&outColor[2], m + FOGG_INSCATTER_A_OFF + 8, 4);
    outColor[3] = 1.0f; // full alpha, will be modulated by density

    if ((size_t)metaOff + FOGG_DENSITY_OFF + 4 > cache->size) {
        *outDensity = 0.5f; // fallback
    } else {
        memcpy(outDensity, m + FOGG_DENSITY_OFF, 4);
    }
    return true;
}

// Walk a single sddt tag's PlanarFog[] block and append results.
bool WalkSddtPlanarFogs(CacheHandle* cache, uint32_t sddtTagId,
                         std::vector<ZH_PlanarFogVolume>& out)
{
    if (sddtTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[sddtTagId];
    if (memcmp(te.classCode, TC_SDDT, 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + SDDT_PLANAR_FOG_OFF + 12 > cache->size) return false;

    const uint8_t* meta = cache->base + metaOff;

    // Read PlanarFog tagblock
    TagBlockRef fogBlk = ReadTagBlock(meta + SDDT_PLANAR_FOG_OFF);
    if (fogBlk.count <= 0 || fogBlk.count > PLANAR_FOG_CAP) {
        if (fogBlk.count == 0) return true; // no planar fogs in this design
        return false;
    }

    int64_t fogArrayOff = TagMetaFileOff(cache, fogBlk.pointer);
    if (fogArrayOff < 0 ||
        (size_t)fogArrayOff + (size_t)fogBlk.count * SDDT_PLANAR_FOG_ENTRY_SIZE > cache->size)
        return false;

    for (int i = 0; i < fogBlk.count && (int)out.size() < PLANAR_FOG_CAP; ++i) {
        const uint8_t* entry = cache->base + fogArrayOff +
                               (size_t)i * SDDT_PLANAR_FOG_ENTRY_SIZE;

        ZH_PlanarFogVolume vol;
        memset(&vol, 0, sizeof(vol));

        // Read normal (float3 at entry+0x30)
        memcpy(&vol.Normal[0], entry + 0x30, 4);
        memcpy(&vol.Normal[1], entry + 0x34, 4);
        memcpy(&vol.Normal[2], entry + 0x38, 4);

        // Read depth (float at entry+0x2C)
        memcpy(&vol.Depth, entry + 0x2C, 4);

        // Read Appearance Settings tag ref at entry+0x04 (16 bytes)
        int32_t foggId = ReadTagRefId(entry + 0x04);
        float color[4] = { 0.6f, 0.7f, 0.8f, 1.0f }; // default blueish fog
        float density = 0.5f;
        if (foggId >= 0) {
            ReadFoggAppearance(cache, foggId, color, &density);
        }
        memcpy(vol.Color, color, sizeof(float) * 4);
        vol.Density = density;

        // Read Vertices tagblock at entry+0x14
        TagBlockRef vertBlk = ReadTagBlock(entry + 0x14);
        vol.VertexCount = (vertBlk.count > 0) ? (uint32_t)vertBlk.count : 0;

        // Compute bounding box and centroid from vertices
        float bmin[3] = { 1e30f, 1e30f, 1e30f };
        float bmax[3] = { -1e30f, -1e30f, -1e30f };
        float csum[3] = { 0, 0, 0 };
        bool havePlaneD = false;

        if (vertBlk.count > 0 && vertBlk.count < 10000) {
            int64_t vertOff = TagMetaFileOff(cache, vertBlk.pointer);
            if (vertOff >= 0 &&
                (size_t)vertOff + (size_t)vertBlk.count * 12 <= cache->size)
            {
                for (int v = 0; v < vertBlk.count; ++v) {
                    const uint8_t* vp = cache->base + vertOff + (size_t)v * 12;
                    float x, y, z;
                    memcpy(&x, vp + 0, 4);
                    memcpy(&y, vp + 4, 4);
                    memcpy(&z, vp + 8, 4);

                    if (x < bmin[0]) bmin[0] = x;
                    if (y < bmin[1]) bmin[1] = y;
                    if (z < bmin[2]) bmin[2] = z;
                    if (x > bmax[0]) bmax[0] = x;
                    if (y > bmax[1]) bmax[1] = y;
                    if (z > bmax[2]) bmax[2] = z;
                    csum[0] += x;
                    csum[1] += y;
                    csum[2] += z;
                }
                float invN = 1.0f / (float)vertBlk.count;
                vol.Centroid[0] = csum[0] * invN;
                vol.Centroid[1] = csum[1] * invN;
                vol.Centroid[2] = csum[2] * invN;

                // Compute plane D from normal and centroid
                vol.PlaneD = -(vol.Normal[0] * vol.Centroid[0] +
                               vol.Normal[1] * vol.Centroid[1] +
                               vol.Normal[2] * vol.Centroid[2]);
                havePlaneD = true;
            }
        }

        // If we got plane data from the Triangles sub-block, use that instead
        // (more accurate - the tag may carry explicit plane equations).
        TagBlockRef triBlk = ReadTagBlock(entry + 0x20);
        if (triBlk.count > 0 && triBlk.count < 10000 && !havePlaneD) {
            int64_t triOff = TagMetaFileOff(cache, triBlk.pointer);
            if (triOff >= 0 && (size_t)triOff + 12 <= cache->size) {
                // Each triangle entry has a sub-block of planes (float4 ABCD)
                TagBlockRef planeBlk = ReadTagBlock(cache->base + triOff);
                if (planeBlk.count > 0) {
                    int64_t planeOff = TagMetaFileOff(cache, planeBlk.pointer);
                    if (planeOff >= 0 && (size_t)planeOff + 16 <= cache->size) {
                        const uint8_t* pp = cache->base + planeOff;
                        // plane3 = float4 (A, B, C, D)
                        float A, B, C, D;
                        memcpy(&A, pp + 0, 4);
                        memcpy(&B, pp + 4, 4);
                        memcpy(&C, pp + 8, 4);
                        memcpy(&D, pp + 12, 4);
                        vol.Normal[0] = A;
                        vol.Normal[1] = B;
                        vol.Normal[2] = C;
                        vol.PlaneD = D;
                    }
                }
            }
        }

        memcpy(vol.BoundsMin, bmin, sizeof(float) * 3);
        memcpy(vol.BoundsMax, bmax, sizeof(float) * 3);

        // If no vertices found, place the fog at origin with large bounds
        if (vol.VertexCount == 0) {
            for (int j = 0; j < 3; ++j) {
                vol.BoundsMin[j] = -500.0f;
                vol.BoundsMax[j] =  500.0f;
                vol.Centroid[j]  = 0.0f;
            }
        }

        NativeDiag("PlanarFogWalker: sddt=%u fog[%d] normal=(%.2f,%.2f,%.2f) "
                   "depth=%.2f density=%.4f color=(%.2f,%.2f,%.2f) "
                   "verts=%u bounds=[(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)]",
                   sddtTagId, i,
                   vol.Normal[0], vol.Normal[1], vol.Normal[2],
                   vol.Depth, vol.Density,
                   vol.Color[0], vol.Color[1], vol.Color[2],
                   vol.VertexCount,
                   vol.BoundsMin[0], vol.BoundsMin[1], vol.BoundsMin[2],
                   vol.BoundsMax[0], vol.BoundsMax[1], vol.BoundsMax[2]);

        out.push_back(vol);
    }

    // Also walk WaterInstances for water-fog volumes (water body fog)
    TagBlockRef waterBlk = ReadTagBlock(meta + SDDT_WATER_INSTANCES_OFF);
    if (waterBlk.count > 0 && waterBlk.count < 100) {
        int64_t waterOff = TagMetaFileOff(cache, waterBlk.pointer);
        if (waterOff >= 0 &&
            (size_t)waterOff + (size_t)waterBlk.count * SDDT_WATER_INSTANCE_SIZE <= cache->size)
        {
            for (int w = 0; w < waterBlk.count && (int)out.size() < PLANAR_FOG_CAP; ++w) {
                const uint8_t* we = cache->base + waterOff +
                                    (size_t)w * SDDT_WATER_INSTANCE_SIZE;

                // Water Instance layout (sddt.xml):
                //   +0x10 colorf FogColor (RGBA, 4 floats)
                //   +0x20 float  FogMurkiness
                //   +0x24 tagblock WaterPlanes (elementSize 0x10: plane3 = float4)
                //   +0x3C rangef BoundsX (min, max)
                //   +0x44 rangef BoundsY
                //   +0x4C rangef BoundsZ

                float fogColor[4];
                memcpy(&fogColor[0], we + 0x10, 4);
                memcpy(&fogColor[1], we + 0x14, 4);
                memcpy(&fogColor[2], we + 0x18, 4);
                memcpy(&fogColor[3], we + 0x1C, 4);

                float murkiness;
                memcpy(&murkiness, we + 0x20, 4);

                // Skip water instances with no fog color / zero murkiness
                if (murkiness <= 0.001f &&
                    fogColor[0] <= 0.001f && fogColor[1] <= 0.001f && fogColor[2] <= 0.001f)
                    continue;

                TagBlockRef wpBlk = ReadTagBlock(we + 0x24);
                if (wpBlk.count <= 0) continue;

                int64_t wpOff = TagMetaFileOff(cache, wpBlk.pointer);
                if (wpOff < 0 || (size_t)wpOff + 16 > cache->size) continue;

                // Read first water plane (float4 ABCD)
                const uint8_t* pp = cache->base + wpOff;
                float A, B, C, D;
                memcpy(&A, pp + 0, 4);
                memcpy(&B, pp + 4, 4);
                memcpy(&C, pp + 8, 4);
                memcpy(&D, pp + 12, 4);

                ZH_PlanarFogVolume vol;
                memset(&vol, 0, sizeof(vol));

                vol.Normal[0] = A;
                vol.Normal[1] = B;
                vol.Normal[2] = C;
                vol.PlaneD = D;
                vol.Depth = 10.0f; // default water fog depth

                // Water fog color - use tag's authored color
                vol.Color[0] = fogColor[0];
                vol.Color[1] = fogColor[1];
                vol.Color[2] = fogColor[2];
                vol.Color[3] = fogColor[3];
                vol.Density = murkiness;

                // Read bounds
                memcpy(&vol.BoundsMin[0], we + 0x3C, 4);
                memcpy(&vol.BoundsMax[0], we + 0x40, 4);
                memcpy(&vol.BoundsMin[1], we + 0x44, 4);
                memcpy(&vol.BoundsMax[1], we + 0x48, 4);
                memcpy(&vol.BoundsMin[2], we + 0x4C, 4);
                memcpy(&vol.BoundsMax[2], we + 0x50, 4);

                vol.Centroid[0] = (vol.BoundsMin[0] + vol.BoundsMax[0]) * 0.5f;
                vol.Centroid[1] = (vol.BoundsMin[1] + vol.BoundsMax[1]) * 0.5f;
                vol.Centroid[2] = (vol.BoundsMin[2] + vol.BoundsMax[2]) * 0.5f;
                vol.VertexCount = 0; // water plane has no explicit vertex polygon

                NativeDiag("PlanarFogWalker: sddt=%u water[%d] normal=(%.2f,%.2f,%.2f) "
                           "D=%.2f murkiness=%.4f color=(%.2f,%.2f,%.2f) "
                           "bounds=[(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)]",
                           sddtTagId, w, A, B, C, D, murkiness,
                           fogColor[0], fogColor[1], fogColor[2],
                           vol.BoundsMin[0], vol.BoundsMin[1], vol.BoundsMin[2],
                           vol.BoundsMax[0], vol.BoundsMax[1], vol.BoundsMax[2]);

                out.push_back(vol);
            }
        }
    }

    return true;
}

// Resolve scnr.StructureDesigns[] and walk each sddt for planar fogs.
bool EnumerateInner(CacheHandle* cache, uint32_t scnrTagId,
                    std::vector<ZH_PlanarFogVolume>& out)
{
    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& scnrTag = cache->tags[scnrTagId];
    if (memcmp(scnrTag.classCode, TC_SCNR, 4) != 0) return false;

    int64_t scnrMetaOff = TagMetaFileOff(cache, scnrTag.metaPointerRaw);
    if (scnrMetaOff < 0) return false;

    // Try both offsets for Structure Designs
    int sdOff = PickStructureDesignsOffset(cache->cacheType);
    if ((size_t)scnrMetaOff + (size_t)sdOff + 12 > cache->size) return false;

    const uint8_t* scnrMeta = cache->base + scnrMetaOff;
    TagBlockRef sdBlk = ReadTagBlock(scnrMeta + sdOff);

    // Fallback: try the alternate offset
    if (sdBlk.count <= 0 || sdBlk.count > 64) {
        int alt = (sdOff == 0x5C) ? 0x58 : 0x5C;
        if ((size_t)scnrMetaOff + (size_t)alt + 12 <= cache->size) {
            TagBlockRef altBlk = ReadTagBlock(scnrMeta + alt);
            if (altBlk.count > 0 && altBlk.count <= 64) {
                sdBlk = altBlk;
                sdOff = alt;
            }
        }
    }

    if (sdBlk.count <= 0 || sdBlk.count > 64) {
        NativeDiag("PlanarFogWalker: scnr=%u no StructureDesigns block found "
                   "(offset=0x%X count=%d)", scnrTagId, sdOff, sdBlk.count);
        return false;
    }

    constexpr int SD_ENTRY_SIZE = 0x20; // 2 tag refs (16B each)
    int64_t sdArrayOff = TagMetaFileOff(cache, sdBlk.pointer);
    if (sdArrayOff < 0 ||
        (size_t)sdArrayOff + (size_t)sdBlk.count * SD_ENTRY_SIZE > cache->size)
        return false;

    NativeDiag("PlanarFogWalker: scnr=%u found %d StructureDesigns at scnr+0x%X",
               scnrTagId, sdBlk.count, sdOff);

    for (int i = 0; i < sdBlk.count; ++i) {
        const uint8_t* sdEntry = cache->base + sdArrayOff +
                                 (size_t)i * SD_ENTRY_SIZE;

        // Read the Design tag ref at +0x00 (16 bytes, TagId at +12)
        int32_t sddtId = ReadTagRefId(sdEntry + 0x00);
        if (sddtId < 0 || (uint32_t)sddtId >= cache->tags.size()) continue;

        // Verify it's an sddt tag
        if (memcmp(cache->tags[sddtId].classCode, TC_SDDT, 4) != 0) continue;

        NativeDiag("PlanarFogWalker: StructureDesigns[%d] -> sddt tag %d",
                   i, sddtId);

        WalkSddtPlanarFogs(cache, (uint32_t)sddtId, out);
    }

    return true;
}

// Inner (C++ objects allowed) - collects fogs into a malloc'd output buffer.
// Separated from the SEH wrapper because __try can't coexist with C++ dtors.
bool EnumerateToBuffer(CacheHandle* cache, uint32_t scnrTagId,
                       ZH_PlanarFogVolume** outBuffer, uint32_t* outCount)
{
    std::vector<ZH_PlanarFogVolume> fogs;
    if (!EnumerateInner(cache, scnrTagId, fogs) || fogs.empty()) {
        *outBuffer = nullptr;
        *outCount = 0;
        return true; // not an error - just no planar fogs
    }

    size_t bytes = fogs.size() * sizeof(ZH_PlanarFogVolume);
    auto* buf = (ZH_PlanarFogVolume*)malloc(bytes);
    if (!buf) {
        *outBuffer = nullptr;
        *outCount = 0;
        return false;
    }
    memcpy(buf, fogs.data(), bytes);
    *outBuffer = buf;
    *outCount = (uint32_t)fogs.size();

    NativeDiag("PlanarFogWalker: scnr=%u total %u planar fog volumes",
               scnrTagId, (uint32_t)fogs.size());
    return true;
}

// SEH wrapper (no C++ objects in scope).
bool SehEnumerate(CacheHandle* cache, uint32_t scnrTagId,
                  ZH_PlanarFogVolume** outBuffer, uint32_t* outCount)
{
    __try {
        return EnumerateToBuffer(cache, scnrTagId, outBuffer, outCount);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        *outBuffer = nullptr;
        *outCount = 0;
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

extern "C" {

__declspec(dllexport) bool __stdcall ZH_PFOG_Enumerate(
    uint64_t cacheHandle, uint32_t scnrTagId,
    ZH_PlanarFogVolume** outBuffer, uint32_t* outCount)
{
    if (!outBuffer || !outCount) return false;
    *outBuffer = nullptr;
    *outCount = 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;

    return SehEnumerate(cache, scnrTagId, outBuffer, outCount);
}

__declspec(dllexport) void __stdcall ZH_PFOG_FreeBuffer(ZH_PlanarFogVolume* buf)
{
    if (buf) free(buf);
}

} // extern "C"
