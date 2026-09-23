// SimpleLightsWalker.cpp
// =============================================================================
// SIMPLE_LIGHTS RE - Native walker for Halo Reach's `simple_lights_analytical`
// scene-light pool. These are the point/spot lights authored in BSP that the
// engine packs into the `SimpleLightsPS` cbuffer (cb5, 656 bytes) each frame
// and consumes via the HREK `simple_lights.hlsl_include` per-pixel loop. They
// are what makes interior caves, sword-base corridors, and structure tunnels
// look lit instead of flat.
//
// MMS doesn't load these at all (TERRAIN_SHADER_RE_FROM_HREK.md section "Simple
// lights"), so all those interiors render against the lightmap diffuse alone.
// This walker is phase 1: extract the placements and surface them to the viewer for
// diagnostic logging + future PS upload.
//
// ----- Tag chain -----------------------------------------------------------
//
//   scnr (scenario)
//     Light Volumes        @ +0x210   elementSize 0x90   (per-instance placement)
//       +0x00 i16   Palette Index           -> Light Volumes Palette index
//       +0x02 i16   Name Index
//       +0x04 u32   Placement Flags         (Never Placed bit 6 = skip)
//       +0x08 fp3   Position (world XYZ)
//       +0x14 deg3  Rotation (pitch/yaw/roll, degrees)
//       +0x20 f32   Scale
//       ... node-orientation tagblocks ...
//       +0x60 e16   Type (0=Sphere, 1=Projective)
//       +0x62 f16   Flags (bit 2 = Cinematic Only - skip)
//       ... lightmap fields ...
//       +0x70 fp3   Target Point  (projective light target)
//       +0x7C f32   Width
//       +0x80 f32   Height Scale
//       +0x84 deg   Field of View (cone full-angle, degrees)
//       +0x88 f32   Falloff Distance
//       +0x8C f32   Cutoff Distance
//
//     Light Volumes Palette @ +0x21C   elementSize 0x10  (palette[])
//       +0x00 tagref Light Volume   -> ltvl tag
//
//   ltvl (Light Volume System) - render-method tag with color/gel params in
//        Postprocess[0].Float Constants[]; the per-light tint is resolved
//        from there (white when unauthored).
//
// ----- Why scnr.LightVolumes is "simple_lights" -----------------------------
//
// In Halo Reach, the HREK schema/source calls these "simple lights" because
// they get packed into the engine's `simple_lights[]` cbuffer at frame time
// (HREK `simple_lights.hlsl_include`, 5 vec4 per light). MCC's renderer
// preserves this shape: 8 lights x 5 vec4 = 640 B + 16 B header = 656 B
// (matches the captured `SimpleLightsPS` cbuffer in
// captured_shaders/liveDump_2026_05_29). The tag-side authoring entry for
// each point/spot light is the scnr Light Volume placement; the ltvl tag
// referenced by Palette[i] carries the color and shader template.
//
// ----- Cap -----------------------------------------------------------------
//
// HREK source caps the PS loop at 8 active lights ("god damn PC compiler
// likes to unroll these loops"). We cap the walker at the same 8 to keep
// the viewer's eventual cbuffer upload bounded. Maps with more authored
// lights (Forge World) will surface only the first 8 - phase 2 can add
// per-frustum culling.
//
// Defensive contract - same shape as SkyWalker / LightWalker:
//   * SEH wrapper at the public boundary.
//   * Class-code + bounds checks at every cross-tag deref.
//   * Index validation against cache->tags.size() before each ltvl deref.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

using namespace zh_mcc;

// =============================================================================
// Public ABI - must match the Rust mirror in crates/hms-native/src/lib.rs (Pack=1, 80 B per entry).
// 5 vec4 layout mirrors HREK's simple_lights.hlsl_include DX11 stride so the
// viewer side can upload the buffer to the PS without a per-light copy.
// =============================================================================
#pragma pack(push, 1)
struct ZH_SimpleLight {
    // vec4[0] - position.xyz + bounding_radius.w
    float Pos[3];
    float BoundingRadius;
    // vec4[1] - direction.xyz + sphere_pct.w (0=spot, 1=omni)
    float Dir[3];
    float SpherePct;
    // vec4[2] - color.rgb + smooth.w (phase 1: white; phase 2 = ltvl tint)
    float Color[3];
    float Smooth;
    // vec4[3] - cos(cutoff_angle).x + angle_falloff_ratio.y + angle_falloff_power.z
    float CosCutoff;
    float AngleFalloffRatio;
    float AngleFalloffPower;
    float Pad3;
    // vec4[4] - bounding_radius.x + far_atten_end.y + far_atten_ratio.z
    float Pad4_BoundingRadius;
    float FarAttenEnd;
    float FarAttenRatio;
    float Pad4;
};
#pragma pack(pop)
static_assert(sizeof(ZH_SimpleLight) == 80,
              "ZH_SimpleLight must be 5 vec4 (HREK simple_lights stride)");

namespace {

constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_LTVL = "ltvl";

// scnr.Light Volumes block - per ReachMCC sbsp/scnr plugin XML.
constexpr int SCNR_LIGHT_VOLUMES_OFFSET   = 0x210;
constexpr int LIGHT_VOLUMES_ELEM_SIZE     = 0x90;

// scnr.Light Volumes Palette block.
constexpr int SCNR_LIGHT_PALETTE_OFFSET   = 0x21C;
constexpr int LIGHT_PALETTE_ELEM_SIZE     = 0x10;

// Per-instance Light Volume placement field offsets (within 0x90 block).
constexpr int LV_PALETTE_INDEX_OFFSET     = 0x00;   // i16
constexpr int LV_PLACEMENT_FLAGS_OFFSET   = 0x04;   // u32
constexpr int LV_POSITION_OFFSET          = 0x08;   // fp3
constexpr int LV_ROTATION_OFFSET          = 0x14;   // deg3
constexpr int LV_TYPE_OFFSET              = 0x60;   // e16  (0=Sphere, 1=Projective)
constexpr int LV_FLAGS_OFFSET             = 0x62;   // f16
constexpr int LV_FIELD_OF_VIEW_OFFSET     = 0x84;   // degrees (cone full-angle)
constexpr int LV_FALLOFF_DISTANCE_OFFSET  = 0x88;   // f32
constexpr int LV_CUTOFF_DISTANCE_OFFSET   = 0x8C;   // f32

// ----- ltvl (Light Volume System) tag layout (ReachMCC ltvl.xml plugin) -----
// baseSize 0xC; "Light Volume System" tagblock @ +0x0 elementSize 0x1B4.
//   ltvlsys + 0x3C  tagblock Postprocess (elementSize 0xB4)
//     postprocess + 0x00  tagref Shader Template (rmt2)
//     postprocess + 0x1C  tagblock Float Constants (elementSize 0x10)
//                          { float a,b,c,d }   (a=R, b=G, c=B per plugin note)
// The rmt2 referenced by Shader Template names each Float Constants slot
// positionally; the slot named "profile_color" carries the beam tint and
// "profile_intensity" the brightness scalar (HREK
// light_volume_property.hlsl_include _index_profile_color / _intensity).
constexpr int LTVL_SYS_BLOCK_OFFSET       = 0x00;   // tagblock Light Volume System
constexpr int LTVL_SYS_ELEM_SIZE          = 0x1B4;
constexpr int LTVL_POSTPROCESS_OFFSET     = 0x3C;   // tagblock Postprocess
constexpr int LTVL_POSTPROCESS_ELEM_SIZE  = 0xB4;
constexpr int LTVL_PP_SHADER_TMPL_OFFSET  = 0x00;   // tagref rmt2
constexpr int LTVL_PP_FLOATCONST_OFFSET   = 0x1C;   // tagblock Float Constants
constexpr int LTVL_FLOATCONST_ELEM_SIZE   = 0x10;   // { a,b,c,d } f32
// rmt2 Float Constants (Arguments) name list - same offset DecalWalker uses.
constexpr int RMT2_FLOAT_CONSTANTS_OFFSET = 0x48;
constexpr int RMT2_FLOATCONST_NAME_SIZE   = 0x04;   // one stringid per arg

constexpr const char* TC_RMT2             = "rmt2";

// Bit 6 of placement flags = "Never Placed" - skip these (used for editor-
// only annotation markers).
constexpr uint32_t LV_FLAG_NEVER_PLACED   = 0x40;
// Bit 2 of LV_FLAGS = "Cinematic Only" - skip; not part of gameplay lights.
constexpr uint16_t LV_LIGHTFLAG_CINEMATIC = 0x04;

// HREK simple_lights[] cap - PC compiler unroll limit, see hlsl_include note.
constexpr int SIMPLE_LIGHT_HARD_CAP       = 8;

// Sanity bounds - Forge World has hundreds of Light Volumes; cap walker
// reads at this before stride-multiplying to avoid OOB.
constexpr int LV_COUNT_SANITY             = 0x4000;
constexpr int LP_COUNT_SANITY             = 0x200;

// Convert HREK Field-Of-View (cone full-angle in degrees, sometimes 0 for
// omnidirectional spheres) to the simple_lights cos(cutoff_angle) form the
// PS expects. The HREK PS halves the FOV and takes cos; for omni lights
// (Sphere type, fov<=0) we return -1 (matches "no cone cull").
float FovDegToCosCutoff(float fovDeg) {
    if (fovDeg <= 0.0f) return -1.0f;       // omnidirectional
    if (fovDeg >= 360.0f) return -1.0f;
    float halfRad = (fovDeg * 0.5f) * (3.14159265358979323846f / 180.0f);
    return cosf(halfRad);
}

// Halo Reach euler -> forward vector. Engine convention is pitch/yaw/roll
// in degrees with the canonical forward = +X. Same shape as the AI facing
// helper in HostByteHelper.
void EulerToForward(float pitch, float yaw, float roll, float out[3]) {
    (void)roll;
    const float DEG2RAD = 3.14159265358979323846f / 180.0f;
    float cp = cosf(pitch * DEG2RAD), sp = sinf(pitch * DEG2RAD);
    float cy = cosf(yaw   * DEG2RAD), sy = sinf(yaw   * DEG2RAD);
    out[0] = cy * cp;
    out[1] = sy * cp;
    out[2] = -sp;
}

bool LocateScnrLightVolumes(CacheHandle* cache, uint32_t scnrTagId,
                            const uint8_t** outScnrMeta,
                            const uint8_t** outBlockBase, int32_t* outCount)
{
    *outScnrMeta = nullptr;
    *outBlockBase = nullptr;
    *outCount = 0;
    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scnrTagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return false;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + (size_t)SCNR_LIGHT_VOLUMES_OFFSET + 8 > cache->size)
        return false;
    const uint8_t* scnr = cache->base + metaOff;
    *outScnrMeta = scnr;
    TagBlockRef blk = ReadTagBlock(scnr + SCNR_LIGHT_VOLUMES_OFFSET);
    if (blk.count <= 0 || blk.count > LV_COUNT_SANITY) return false;
    int64_t arrOff = TagMetaFileOff(cache, blk.pointer);
    if (arrOff < 0) return false;
    if ((size_t)arrOff + (size_t)blk.count * LIGHT_VOLUMES_ELEM_SIZE > cache->size)
        return false;
    *outBlockBase = cache->base + arrOff;
    *outCount = blk.count;
    return true;
}

// Resolve Light Volumes Palette[idx] -> ltvl tag id. Returns -1 on failure or
// when the entry is unbound. Caller only uses this for diag logging in phase
// 1 (color/intensity stays white until phase 2).
int32_t ResolvePaletteLtvlTagId(CacheHandle* cache, const uint8_t* scnr, int16_t paletteIdx)
{
    if (paletteIdx < 0) return -1;
    TagBlockRef palBlk = ReadTagBlock(scnr + SCNR_LIGHT_PALETTE_OFFSET);
    if (palBlk.count <= 0 || palBlk.count > LP_COUNT_SANITY) return -1;
    if (paletteIdx >= palBlk.count) return -1;
    int64_t palArrOff = TagMetaFileOff(cache, palBlk.pointer);
    if (palArrOff < 0) return -1;
    if ((size_t)palArrOff + (size_t)palBlk.count * LIGHT_PALETTE_ELEM_SIZE > cache->size)
        return -1;
    const uint8_t* palEntry =
        cache->base + palArrOff + (size_t)paletteIdx * LIGHT_PALETTE_ELEM_SIZE;
    // TagReference: classId@+0, padding@+4..11, tagId@+12.
    uint32_t rawId = RU32(palEntry + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    int32_t id = (int32_t)(rawId & 0xFFFFu);
    if ((uint32_t)id >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[id].classCode, TC_LTVL, 4) != 0) return -1;
    return id;
}

// Reclaimer Gen3+ TagReference: ClassId @ +0, padding @ +4..11, TagId @ +12.
int32_t ReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// Resolve the ltvl beam tint (and an intensity scalar) from the ltvl tag's
// Postprocess[0].Float Constants[] block, using the rmt2 Shader Template's
// argument-name list to identify the "profile_color" / "profile_intensity"
// slots. Mirrors DecalWalker's rmt2-arg walk on the decs Postprocess block.
//
// Returns true if a usable color was recovered. On false the caller keeps the
// phase-1 white. outColor is linear RGB, outIntensity a >=0 scalar (1.0 if no
// explicit intensity slot was authored).
bool ResolveLtvlBeamTint(CacheHandle* cache, int32_t ltvlId,
                         float outColor[3], float* outIntensity)
{
    outColor[0] = outColor[1] = outColor[2] = 1.0f;
    *outIntensity = 1.0f;
    if (ltvlId < 0 || (uint32_t)ltvlId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[ltvlId];
    if (te.classIndex < 0 || memcmp(te.classCode, TC_LTVL, 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    const uint8_t* sys = cache->base + metaOff;

    // Light Volume System tagblock @ +0x0 (1 element expected).
    if ((size_t)metaOff + LTVL_SYS_BLOCK_OFFSET + 8 > cache->size) return false;
    TagBlockRef sysBlk = ReadTagBlock(sys + LTVL_SYS_BLOCK_OFFSET);
    if (sysBlk.count <= 0 || sysBlk.count > 8) return false;
    int64_t sysOff = TagMetaFileOff(cache, sysBlk.pointer);
    if (sysOff < 0) return false;
    if ((size_t)sysOff + (size_t)sysBlk.count * LTVL_SYS_ELEM_SIZE > cache->size)
        return false;
    const uint8_t* sysEntry = cache->base + sysOff;  // element [0]

    // Postprocess tagblock @ sysEntry+0x3C (1 element).
    if ((size_t)((sysEntry - cache->base) + LTVL_POSTPROCESS_OFFSET + 8) > cache->size)
        return false;
    TagBlockRef ppBlk = ReadTagBlock(sysEntry + LTVL_POSTPROCESS_OFFSET);
    if (ppBlk.count <= 0 || ppBlk.count > 8) return false;
    int64_t ppOff = TagMetaFileOff(cache, ppBlk.pointer);
    if (ppOff < 0) return false;
    if ((size_t)ppOff + (size_t)ppBlk.count * LTVL_POSTPROCESS_ELEM_SIZE > cache->size)
        return false;
    const uint8_t* pp = cache->base + ppOff;  // Postprocess[0]

    // Float Constants tagblock @ pp+0x1C.
    TagBlockRef fcBlk = ReadTagBlock(pp + LTVL_PP_FLOATCONST_OFFSET);
    if (fcBlk.count <= 0 || fcBlk.count > 64) return false;
    int64_t fcOff = TagMetaFileOff(cache, fcBlk.pointer);
    if (fcOff < 0) return false;
    if ((size_t)fcOff + (size_t)fcBlk.count * LTVL_FLOATCONST_ELEM_SIZE > cache->size)
        return false;

    // Resolve rmt2 argument NAMES to find which Float-Constant slot is the
    // beam color / intensity. Fall back to slot heuristics if names absent.
    int colorSlot = -1, intensitySlot = -1;
    int32_t rmt2Id = ReadTagRefId(pp + LTVL_PP_SHADER_TMPL_OFFSET);
    if (rmt2Id >= 0 && (uint32_t)rmt2Id < cache->tags.size()) {
        const TagEntry& rt = cache->tags[rmt2Id];
        if (rt.classIndex >= 0 && memcmp(rt.classCode, TC_RMT2, 4) == 0) {
            int64_t rmt2Off = TagMetaFileOff(cache, rt.metaPointerRaw);
            if (rmt2Off >= 0 &&
                (size_t)rmt2Off + RMT2_FLOAT_CONSTANTS_OFFSET + 8 <= cache->size)
            {
                const uint8_t* rmt2 = cache->base + rmt2Off;
                TagBlockRef argBlk = ReadTagBlock(rmt2 + RMT2_FLOAT_CONSTANTS_OFFSET);
                if (argBlk.count > 0 && argBlk.count < 256) {
                    int64_t argOff = TagMetaFileOff(cache, argBlk.pointer);
                    if (argOff >= 0 &&
                        (size_t)argOff + (size_t)argBlk.count * RMT2_FLOATCONST_NAME_SIZE
                            <= cache->size)
                    {
                        for (int k = 0; k < argBlk.count && k < (int)fcBlk.count; ++k) {
                            int32_t sid;
                            memcpy(&sid, cache->base + argOff +
                                   (size_t)k * RMT2_FLOATCONST_NAME_SIZE, 4);
                            const char* name = ResolveStringId(cache, sid);
                            if (!name) continue;
                            if (colorSlot < 0 && strstr(name, "color"))
                                colorSlot = k;
                            if (intensitySlot < 0 && strstr(name, "intensity"))
                                intensitySlot = k;
                        }
                    }
                }
            }
        }
    }

    // Heuristic fallback: slot 0 is conventionally the tint on these one-arg
    // beam templates (profile_color is the dominant authored param).
    if (colorSlot < 0) colorSlot = 0;

    auto readFc = [&](int slot, float out[4]) -> bool {
        if (slot < 0 || slot >= (int)fcBlk.count) return false;
        size_t off = (size_t)fcOff + (size_t)slot * LTVL_FLOATCONST_ELEM_SIZE;
        if (off + LTVL_FLOATCONST_ELEM_SIZE > cache->size) return false;
        memcpy(out, cache->base + off, 16);
        return true;
    };

    float c[4];
    bool got = false;
    if (readFc(colorSlot, c)) {
        // Plugin note: a=R, b=G, c=B. Reject all-zero / NaN as "unauthored".
        bool finite = isfinite(c[0]) && isfinite(c[1]) && isfinite(c[2]);
        bool nonzero = (c[0] != 0.0f || c[1] != 0.0f || c[2] != 0.0f);
        if (finite && nonzero) {
            outColor[0] = c[0]; outColor[1] = c[1]; outColor[2] = c[2];
            got = true;
        }
    }
    if (intensitySlot >= 0) {
        float in[4];
        if (readFc(intensitySlot, in) && isfinite(in[0]) && in[0] > 0.0f)
            *outIntensity = in[0];
    }
    return got;
}

bool EnumerateInner(CacheHandle* cache, uint32_t scnrTagId,
                    ZH_SimpleLight* outBuf, uint32_t* outCount,
                    uint32_t* outTotalSeen, uint32_t* outSkipped)
{
    *outCount = 0;
    *outTotalSeen = 0;
    *outSkipped = 0;
    const uint8_t* scnr = nullptr;
    const uint8_t* base = nullptr;
    int32_t total = 0;
    if (!LocateScnrLightVolumes(cache, scnrTagId, &scnr, &base, &total)) {
        NativeDiag("SimpleLightsWalker: scnr=0x%X no Light Volumes block (or empty)",
                   scnrTagId);
        return true;  // not an error - many maps have no simple lights
    }
    *outTotalSeen = (uint32_t)total;
    NativeDiag("SimpleLightsWalker: scnr=0x%X total placements=%d (cap=%d)",
               scnrTagId, total, SIMPLE_LIGHT_HARD_CAP);

    uint32_t written = 0;
    for (int i = 0; i < total && (int)written < SIMPLE_LIGHT_HARD_CAP; ++i) {
        const uint8_t* P = base + (size_t)i * LIGHT_VOLUMES_ELEM_SIZE;

        uint32_t placementFlags = RU32(P + LV_PLACEMENT_FLAGS_OFFSET);
        if (placementFlags & LV_FLAG_NEVER_PLACED) {
            (*outSkipped)++;
            continue;
        }
        uint16_t lightFlags = RU16(P + LV_FLAGS_OFFSET);
        if (lightFlags & LV_LIGHTFLAG_CINEMATIC) {
            (*outSkipped)++;
            continue;
        }

        float pos[3], rot[3];
        memcpy(pos, P + LV_POSITION_OFFSET, 12);
        memcpy(rot, P + LV_ROTATION_OFFSET, 12);

        uint16_t typeEnum = RU16(P + LV_TYPE_OFFSET);
        float fov, falloff, cutoff;
        memcpy(&fov,     P + LV_FIELD_OF_VIEW_OFFSET,    4);
        memcpy(&falloff, P + LV_FALLOFF_DISTANCE_OFFSET, 4);
        memcpy(&cutoff,  P + LV_CUTOFF_DISTANCE_OFFSET,  4);

        int16_t palIdx = (int16_t)R16(P + LV_PALETTE_INDEX_OFFSET);
        int32_t ltvlId = ResolvePaletteLtvlTagId(cache, scnr, palIdx);

        // Sanity: ditch obvious garbage values.
        if (!(cutoff > 0.0f) || !isfinite(cutoff)) cutoff = 8.0f;        // default 8 wu
        if (!(falloff >= 0.0f) || !isfinite(falloff)) falloff = cutoff * 0.5f;
        if (falloff > cutoff) falloff = cutoff * 0.5f;
        if (!isfinite(fov) || fov < 0.0f || fov > 360.0f) fov = 0.0f;
        bool isOmni = (typeEnum == 0) || (fov <= 0.001f);

        ZH_SimpleLight& L = outBuf[written];
        memset(&L, 0, sizeof(L));
        L.Pos[0] = pos[0]; L.Pos[1] = pos[1]; L.Pos[2] = pos[2];
        L.BoundingRadius = cutoff * cutoff;          // PS compares dist^2 < radius^2

        float dir[3] = {0, 0, -1};
        if (!isOmni) EulerToForward(rot[0], rot[1], rot[2], dir);
        L.Dir[0] = dir[0]; L.Dir[1] = dir[1]; L.Dir[2] = dir[2];
        L.SpherePct = isOmni ? 1.0f : 0.0f;

        // Resolve the per-light tint from the ltvl tag's
        // Postprocess[0].Float Constants[] (profile_color slot). Falls back
        // to white when the tag/slot is unauthored - never an invented color.
        float beamColor[3] = {1.0f, 1.0f, 1.0f};
        float beamIntensity = 1.0f;
        bool tinted = ResolveLtvlBeamTint(cache, ltvlId, beamColor, &beamIntensity);
        L.Color[0] = beamColor[0] * beamIntensity;
        L.Color[1] = beamColor[1] * beamIntensity;
        L.Color[2] = beamColor[2] * beamIntensity;
        L.Smooth = 1.0f;

        L.CosCutoff = FovDegToCosCutoff(fov);
        // angle_falloff_ratio: how sharply the spot edge falls off; HREK
        // tag layer authored as 1.0 by default. We don't have a placement-
        // level field for this so default to 1.
        L.AngleFalloffRatio = isOmni ? 0.0f : 1.0f;
        L.AngleFalloffPower = isOmni ? 0.0f : 1.0f;

        L.Pad4_BoundingRadius = cutoff;
        L.FarAttenEnd = cutoff;
        // (cutoff - falloff) is the linear-attenuation distance; ratio is 1/d.
        float attenDist = (cutoff - falloff);
        L.FarAttenRatio = (attenDist > 0.0001f) ? (1.0f / attenDist) : 1.0f;

        NativeDiag("SimpleLightsWalker:   light[%u] pal=%d ltvl=0x%X type=%s "
                   "pos=(%.1f,%.1f,%.1f) fov=%.1fdeg fall=%.1f cut=%.1f "
                   "tint=%s(%.2f,%.2f,%.2f)x%.2f",
                   written, (int)palIdx,
                   (unsigned)(ltvlId < 0 ? 0xFFFFFFFFu : (uint32_t)ltvlId),
                   isOmni ? "omni" : "spot",
                   pos[0], pos[1], pos[2], fov, falloff, cutoff,
                   tinted ? "ltvl" : "white",
                   beamColor[0], beamColor[1], beamColor[2], beamIntensity);
        ++written;
    }

    if (total > SIMPLE_LIGHT_HARD_CAP) {
        NativeDiag("SimpleLightsWalker: scnr=0x%X capped at %d (had %d placements, skipped=%u)",
                   scnrTagId, SIMPLE_LIGHT_HARD_CAP, total, *outSkipped);
    }

    *outCount = written;
    return true;
}

bool EnumerateSeh(CacheHandle* cache, uint32_t scnrTagId,
                  ZH_SimpleLight* outBuf, uint32_t* outCount,
                  uint32_t* outTotalSeen, uint32_t* outSkipped)
{
    __try { return EnumerateInner(cache, scnrTagId, outBuf, outCount, outTotalSeen, outSkipped); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        NativeDiag("SimpleLightsWalker: scnr=0x%X SEH fault during enumerate", scnrTagId);
        *outCount = 0;
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public export - ZH_SCNR_EnumerateSimpleLights
// =============================================================================
//
// Caller-allocated buffer model (matches FogParser / SunLight): the caller passes a
// fixed-size ZH_SimpleLight[8] array; native writes up to 8 entries and
// returns the count actually written via outCount.
//
//   outBuf        : ZH_SimpleLight[8] (must be at least 8 entries)
//   outCount      : actual lights written (0..8)
//   outTotalSeen  : total placements in scnr.LightVolumes (for diag log)
//   outSkipped    : count of "Never Placed" or "Cinematic Only" entries
//
// Returns 1 on success (count may be 0 - "valid query, no simple lights"),
// 0 on hard failure (bad cache handle, bad scnr tag id).

extern "C" __declspec(dllexport) int __stdcall ZH_SCNR_EnumerateSimpleLights(
    uint64_t cacheHandle,
    uint32_t scnrTagId,
    ZH_SimpleLight* outBuf,
    uint32_t* outCount,
    uint32_t* outTotalSeen,
    uint32_t* outSkipped)
{
    if (outCount)     *outCount     = 0;
    if (outTotalSeen) *outTotalSeen = 0;
    if (outSkipped)   *outSkipped   = 0;
    if (!outBuf || !outCount || !outTotalSeen || !outSkipped) return 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        NativeDiag("SimpleLightsWalker: bad cacheHandle=0x%llX",
                   (unsigned long long)cacheHandle);
        return 0;
    }

    return EnumerateSeh(cache, scnrTagId, outBuf, outCount, outTotalSeen, outSkipped) ? 1 : 0;
}
