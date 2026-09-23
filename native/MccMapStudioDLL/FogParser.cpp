// FogParser.cpp
// =============================================================================
// Native walker for the scnr -> Fog palette -> active fogg ->
// (inscatter A, inscatter B, sky tint, density, start distance) chain in
// Halo MCC HaloReach .map files.
//
// FOG_RE_V3: WRONG SCNR OFFSET FIX.
// =====================================================================
//
// The prior FOG_RE_V2 revision walked scnr+0x528 (with a 0x4-back fallback
// to 0x524) and used an entry stride of 0x8C, then scanned each entry for
// an embedded 'fogg' fourcc. Per Lord Zedd's authoritative
// `Assembly/Plugins/ReachMCC/scnr.xml` plugin (line 6009 and 6059):
//
//   scnr+0x528  =  Background Sound Environment Palette  (elementSize 0x78)
//   scnr+0x534  =  Fog palette                            (elementSize 0x18)
//                  +0x00 stringId Name
//                  +0x04 int16    Unknown
//                  +0x06 int16    Unknown
//                  +0x08 tagRef   Fog (16 bytes, fogg tagref)
//
// FOG_RE_V2 was reading the Background Sound Env palette and overrun-
// scanning into adjacent palette data (Fog @ 0x534, Camera FX @ 0x540,
// Weather @ 0x54C). The 'gggf' (fogg BE-fourcc) scanner accidentally
// caught real fogg tagrefs inside the Fog palette only because each Sound
// Env entry (0x78 bytes) overlapped the start of the next palette block
// when stepped at stride 0x8C. The values it surfaced were plausible but
// the walk was unprincipled and brittle to schema drift.
//
// This revision walks the actual Fog palette at scnr+0x534 with stride
// 0x18 and reads the fogg tagref directly at entry+0x08 - no fourcc
// scanning required.
//
// Layout reference (verified against ReachMCC/scnr.xml):
//
//   scnr.Fog palette          @ scnr + 0x534 (elementSize 0x18)
//     +0x00 stringId Name
//     +0x04 int16    Unknown
//     +0x06 int16    Unknown
//     +0x08 tagRef   Fog        (16 B; fogg class)
//
//   fogg tag - .map cache layout. Offset verification state (RENDER_RE_R2):
//     VERIFIED (used live):
//       +0x04 distance_bias
//       +0x14 sky_fog_thickness
//       +0x18 sky_fog_falloff_end (== sky_fog_max_distance)
//       +0x1C sky_fog_color (RGB)
//       +0x3C ground_fog_color (RGB)  [read but only used as InscatterB tint]
//     DERIVED - high confidence, anchored to the confirmed +0x14 thickness via
//     the fogg.xml "sky fog" field order (flags, base height, fog height,
//     thickness, falloff, color). Guarded by a sanity check; falls back to a
//     no-op flat band if the read looks wrong:
//       +0x0C sky_fog_base_height
//       +0x10 sky_fog_height
//     UNVERIFIED - NOT wired (would mis-fog if guessed). Kept OFF + dumped via
//     the MMS_NATIVE_LOG byte dump in ReadFoggFields so the user can confirm:
//       ground band thickness/height/base_height/max_distance, fog-light block,
//       extinction_threshold.
//   The cbuffer FIELD NAMES come from HREK atmosphere_structs.hlsl_include
//   (s_atmosphere_constants); the ENGINE MODEL from atmosphere_core.hlsl_include
//   compute_scattering_core / get_fog_thickness_at_relative_height.
//
// Defensive contract:
//   * Every cross-tag deref is wrapped in __try / __except so a busted
//     scnr or fogg layout returns false rather than tearing down the
//     worker thread.
//   * Index validation against cache->tags.size() guards every TagId
//     before lookup.
//   * Class-code checks confirm the tag at each link is what we expect.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

// Public ABI surface - must match the Rust mirror in crates/hms-native/src/lib.rs.
#pragma pack(push, 1)
struct ZH_FogParams {
    float    InscatterA[3];   // near color, RGB
    float    InscatterB[3];   // far color, RGB
    float    SkyTint[3];      // sky-tint RGB
    float    Density;         // SkyFogThickness - 0..1 max opacity cap (NOT exp rate)
    float    StartDistance;   // DistanceBias - world units, negative = into screen
    uint32_t HasFog;          // 0 = scnr has no fog palette / null fogg ref, 1 = OK
    float    FalloffEnd;      // distance (wu) at which fog reaches max thickness
    uint32_t _pad;

    // FOG_HEIGHT_BAND: the per-pixel quadratic
    // height falloff + sight-ray-in-band fraction is the #1 reason MMS fog
    // "looks off" (uniform wall regardless of looking up/down). These two
    // fields drive the engine `get_fog_thickness_at_relative_height` +
    // `dist_ratio_in_fog` (atmosphere_core.hlsl_include:34-43,63).
    //
    // SkyFogHeight / SkyFogBaseHeight offsets are DERIVED (high confidence)
    // from the VERIFIED sky_fog_thickness @ +0x14 plus the fogg "sky fog"
    // struct field ORDER in the Assembly plugin (fogg.xml: flags, base
    // height, fog height, thickness[+0x14], falloff end[+0x18], color[+0x1C]).
    // Stepping back from the confirmed +0x14 thickness in 4-byte float
    // strides forces fog_height @ +0x10 and base_height @ +0x0C. See
    // FOGG_SKY_FOG_HEIGHT_OFF / FOGG_SKY_BASE_HEIGHT_OFF below.
    float    SkyFogHeight;       // height band thickness above base (wu); 0 / huge = flat
    float    SkyFogBaseHeight;   // world Z of the fog band floor (wu)
    // GROUND BAND + FOG LIGHT - FOGG_GROUND_BAND_RE: now VERIFIED
    // against HREK standalone .atmosphere_fog tags (byte-exact across 30_settlement,
    // 35_island, condemned, preserve) whose first half (+0x04..+0x28) is itself
    // independently confirmed by the production FogParser reading sane runtime
    // values from the live MCC cache. Ground band = the second `solo_fog_parameters`
    // struct contiguous after the sky band at +0x2C..+0x48. Color = fog_color *
    // fog_color_intensity (HDR). HasGroundBand set only when the read passes a
    // sanity test; else 0 (shader collapses the band - no-op).
    float    GroundFogThickness; // +0x34 (0..1); 0 => ground band disabled
    float    GroundFogColor[3];  // +0x3C RGB * intensity (+0x48)
    float    GroundFogHeight;    // +0x30 band thickness (wu)
    float    GroundFogBaseHeight;// +0x2C band floor world Z (wu)
    float    GroundFogMaxDistance;// +0x38 (wu)
    // FOG LIGHT 1 (directional sun-through-fog inscatter). The on-disk struct
    // stores pitch/yaw ANGLES (+0x4C/+0x50) for the light direction; the runtime
    // recovers the dir vector in the shader from the scene analytical sun instead.
    // We export the color * intensity + the two falloff exponents + nearby cutoff.
    float    FogLightColor[3];   // +0x58 tint RGB * intensity (+0x64)
    float    FogLightAngularFalloff;  // +0x68 pow() exponent on the angular disc
    float    FogLightDistanceFalloff; // +0x6C pow() exponent on (1-extinction)
    float    FogLightNearbyCutoff;    // +0x70 near-distance cutoff [0..1]
    uint32_t HasHeightBand;      // 1 => SkyFogHeight/BaseHeight are valid & live
    uint32_t HasGroundBand;      // 1 => ground-band fields valid & live
    uint32_t HasFogLight;        // 1 => fog-light fields valid & live
    uint32_t _pad2;
    // World-space direction TO the fog light (from pitch +0x4C / yaw +0x50) and the
    // disc shape (radius_scale/offset) derived from angular_radius (+0x54). The engine ratio is
    // saturate(dot(view,dir)*radius_scale + radius_offset); pow(ratio, angular_falloff)*color.
    float    FogLightDir[3];
    float    FogLightRadiusScale;
    float    FogLightRadiusOffset;
};
#pragma pack(pop)

namespace {

// scnr.Fog palette layout (verified against ReachMCC/scnr.xml line 6059).
constexpr int FOG_PALETTE_ENTRY_SIZE = 0x18;
constexpr int FOG_PALETTE_FOGG_TAGREF_OFF = 0x08;  // 16-byte TagReference

// fogg field offsets (verified vs sapien.exe disassembly).
// .map cache layout (Reach). Verified on forge_halo.map tag #11740.
constexpr int FOGG_INSCATTER_A_OFF = 0x1C;   // sky_fog_color RGB (float3)
constexpr int FOGG_INSCATTER_B_OFF = 0x3C;   // ground_fog_color RGB (float3)
constexpr int FOGG_SKY_TINT_OFF    = 0x1C;   // same as sky color (engine sky_tint is separate)
constexpr int FOGG_DENSITY_OFF     = 0x14;   // sky_fog_thickness (0.0-1.0)
constexpr int FOGG_FALLOFF_END_OFF = 0x18;   // sky_fog_falloff_end (world units)
constexpr int FOGG_START_DIST_OFF  = 0x04;   // distance_bias

// FOG_HEIGHT_BAND (RENDER_RE_R2): DERIVED from the VERIFIED +0x14 thickness +
// fogg.xml "sky fog" field order (flags, base height, fog height, thickness).
// 4-byte float strides back from +0x14: fog_height @ +0x10, base_height @ +0x0C.
// HIGH confidence (anchored to a confirmed offset), but NOT independently
// byte-verified - guarded by a plausibility check + the byte-dump diag below.
constexpr int FOGG_SKY_FOG_HEIGHT_OFF  = 0x10;   // sky_fog_height  (band thickness, wu)
constexpr int FOGG_SKY_BASE_HEIGHT_OFF = 0x0C;   // sky_fog_base_height (band floor Z, wu)
constexpr int FOGG_SKY_INTENSITY_OFF   = 0x28;   // sky fog_color_intensity (HDR scale)

// FOGG_GROUND_BAND_RE: the ground band is a second
// solo_fog_parameters struct contiguous after the sky band. Byte-verified
// against HREK standalone .atmosphere_fog tags (30_settlement, 35_island,
// condemned, preserve). See FOGG_GROUND_BAND_RE.md.
constexpr int FOGG_GND_BASE_HEIGHT_OFF = 0x2C;   // ground base_height (wu)
constexpr int FOGG_GND_HEIGHT_OFF      = 0x30;   // ground fog_height (band thickness, wu)
constexpr int FOGG_GND_THICKNESS_OFF   = 0x34;   // ground fog_thickness [0..1]
constexpr int FOGG_GND_MAX_DIST_OFF    = 0x38;   // ground max_fog_distance (wu)
constexpr int FOGG_GND_COLOR_OFF       = 0x3C;   // ground fog_color RGB
constexpr int FOGG_GND_INTENSITY_OFF   = 0x48;   // ground fog_color_intensity (HDR scale)

// Fog light 1 (directional sun-through-fog inscatter), byte-verified vs HREK.
constexpr int FOGG_FL_TINT_COLOR_OFF   = 0x58;   // tint_color RGB
constexpr int FOGG_FL_TINT_INTENS_OFF  = 0x64;   // tint_color_intensity (HDR scale)
constexpr int FOGG_FL_ANG_FALLOFF_OFF  = 0x68;   // angular_falloff_steepness (pow exponent)
constexpr int FOGG_FL_DIST_FALLOFF_OFF = 0x6C;   // distance_falloff_steepness (pow exponent)
constexpr int FOGG_FL_NEARBY_CUTOFF_OFF= 0x70;   // nearby_cutoff_percentage [0..1]

// Guard: largest field byte we read. Now covers the full sky + ground +
// fog-light set up to the nearby cutoff at +0x70.
constexpr int FOGG_REQUIRED_BYTES  = FOGG_FL_NEARBY_CUTOFF_OFF + 4;

constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_FOGG = "fogg";

// scnr.Fog palette offset. Per Lord Zedd's ReachMCC/scnr.xml plugin
// (line 6059), the Fog palette tagblock header lives at scnr+0x534 across
// all known MCC builds. (Note: non-MCC Reach is at +0x548 per
// Plugins/Reach/scnr.xml line 6051 - we don't target that.)
//
// We retain a per-build switch in case a future U13+ revision drifts the
// offset, but the current default (and only verified value) is 0x534.
int PickFogPaletteOffset(CacheType ct) {
    switch (ct) {
        case CacheType::MccHaloReach:
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
        case CacheType::MccHaloReachU10:
        case CacheType::MccHaloReachU13:
        default:
            return 0x534;
    }
}

// TagReference layout (Gen3+): ClassId @ +0, padding @ +4..11, TagId @ +12.
// Returns -1 only for the genuine 0xFFFFFFFF null sentinel - tag ids with
// high-bit identity salt (e.g. 0xA60Cxxxx) are valid and we mask to 16 bits.
int32_t FogReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// Locate scnr.Fog[] tagblock pointer/count.  Returns false on any failure
// (wrong class, OOB metaOff, OOB block). paletteOffset is the scnr-relative
// offset to the tagblock header (12 bytes: count + uint32 unused + ptr,
// per ReadTagBlock contract).
bool LocateFogPalette(CacheHandle* cache, uint32_t scnrTagId,
                      int paletteOffset,
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
    if ((size_t)metaOff + (size_t)paletteOffset + 8 > cache->size) return false;

    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef blk = ReadTagBlock(meta + paletteOffset);
    // Fog palettes are typically 1-16 entries - anything > 0x100 is a sign
    // we read the wrong offset (e.g. a misaligned slot that happens to
    // contain a large int32).
    if (blk.count <= 0 || blk.count > 0x100) return false;

    int64_t arrayOff = TagMetaFileOff(cache, blk.pointer);
    if (arrayOff < 0 ||
        (size_t)arrayOff + (size_t)blk.count * FOG_PALETTE_ENTRY_SIZE > cache->size)
        return false;

    *outBlockBase = cache->base + arrayOff;
    *outCount = blk.count;
    return true;
}

// Read the fogg TagRef directly from a Fog palette entry. Returns -1 if
// the ref is null or doesn't resolve to a valid fogg tag.
int32_t ReadFoggFromPaletteEntry(CacheHandle* cache, const uint8_t* entry) {
    int32_t id = FogReadTagRefId(entry + FOG_PALETTE_FOGG_TAGREF_OFF);
    if (id < 0) return -1;
    if ((uint32_t)id >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[id].classCode, TC_FOGG, 4) != 0) return -1;
    return id;
}

// Walk every palette entry, return the first non-null fogg tag id.
// -1 if no entry resolves.
int32_t ResolveActiveFoggTagId(CacheHandle* cache, uint32_t scnrTagId,
                               int paletteOffset)
{
    const uint8_t* base = nullptr;
    int32_t count = 0;
    if (!LocateFogPalette(cache, scnrTagId, paletteOffset, &base, &count))
        return -1;

    for (int i = 0; i < count; ++i) {
        const uint8_t* entry = base + (size_t)i * FOG_PALETTE_ENTRY_SIZE;
        int32_t fogg = ReadFoggFromPaletteEntry(cache, entry);
        if (fogg >= 0) return fogg;
    }
    return -1;
}

// Read the 5 fogg fields into outParams.  Returns false on any read fault
// or class mismatch.
bool ReadFoggFields(CacheHandle* cache, uint32_t foggTagId, ZH_FogParams* outParams)
{
    if (foggTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[foggTagId];
    if (memcmp(te.classCode, TC_FOGG, 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + (size_t)FOGG_REQUIRED_BYTES > cache->size) return false;

    const uint8_t* m = cache->base + metaOff;
    // float3 reads - Reach stores RGB inline as 3 little-endian floats.
    for (int i = 0; i < 3; ++i) {
        memcpy(&outParams->InscatterA[i], m + FOGG_INSCATTER_A_OFF + i * 4, 4);
        memcpy(&outParams->InscatterB[i], m + FOGG_INSCATTER_B_OFF + i * 4, 4);
        memcpy(&outParams->SkyTint[i],    m + FOGG_SKY_TINT_OFF    + i * 4, 4);
    }
    // #atm-GAP5 (engine-exact): _sky_fog_color = fog_color * fog_color_intensity (HDR), exactly
    // like the ground band (+0x48) and fog light (+0x64) which ALREADY pre-multiply. Sky was the
    // odd one out - SkyTint was taken raw, so sky inscatter under-bright vs engine when intensity
    // != 1. Pre-multiply, guarded by a plausibility check (fall back to raw tint if implausible).
    {
        float skyIntensity = 1.0f;
        memcpy(&skyIntensity, m + FOGG_SKY_INTENSITY_OFF, 4);
        if (skyIntensity >= 0.0f && skyIntensity < 100.0f && skyIntensity == skyIntensity) {
            outParams->SkyTint[0] *= skyIntensity;
            outParams->SkyTint[1] *= skyIntensity;
            outParams->SkyTint[2] *= skyIntensity;
        }
    }
    memcpy(&outParams->Density,       m + FOGG_DENSITY_OFF,     4);
    memcpy(&outParams->StartDistance, m + FOGG_START_DIST_OFF,  4);
    memcpy(&outParams->FalloffEnd,    m + FOGG_FALLOFF_END_OFF, 4);
    // FalloffEnd is the "distance at which fog reaches max thickness" in
    // world units. If the value is implausible (<= 0 or > 1e6), substitute
    // a sensible Reach default. This guards against reading from a slot
    // that isn't actually falloff_end on some cache variants.
    if (!(outParams->FalloffEnd > 1.0f) || outParams->FalloffEnd > 100000.0f)
        outParams->FalloffEnd = 1000.0f;

    // -----------------------------------------------------------------------
    // FOG_HEIGHT_BAND (RENDER_RE_R2): read the DERIVED sky height fields and
    // mark them live only if both pass a sanity test. fog_height is a positive
    // band thickness in world units; base_height can be any finite Z. A bad
    // read (NaN / inf / non-positive height) leaves the band OFF so the PS
    // falls back to the existing flat behaviour (no-op) rather than mis-fogging.
    // -----------------------------------------------------------------------
    float skyHeight = 0.0f, skyBase = 0.0f;
    memcpy(&skyHeight, m + FOGG_SKY_FOG_HEIGHT_OFF,  4);
    memcpy(&skyBase,   m + FOGG_SKY_BASE_HEIGHT_OFF, 4);

    bool heightOk =
        // finite
        (skyHeight == skyHeight) && (skyBase == skyBase) &&
        (skyHeight < 1e30f && skyHeight > -1e30f) &&
        (skyBase   < 1e30f && skyBase   > -1e30f) &&
        // a real band: positive thickness within a plausible world range
        (skyHeight > 0.01f && skyHeight < 100000.0f) &&
        (skyBase   > -100000.0f && skyBase < 100000.0f);

    if (heightOk) {
        outParams->SkyFogHeight     = skyHeight;
        outParams->SkyFogBaseHeight = skyBase;
        outParams->HasHeightBand    = 1;
    } else {
        // Safe default: huge band, base at 0 -> height weight ~= 1 everywhere,
        // dist_ratio ~= 1 -> identical to the current flat fog (no-op).
        outParams->SkyFogHeight     = 1e9f;
        outParams->SkyFogBaseHeight = 0.0f;
        outParams->HasHeightBand    = 0;
    }

    // -----------------------------------------------------------------------
    // GROUND BAND - FOGG_GROUND_BAND_RE: real reads at the
    // byte-verified offsets. The ground band is a second solo_fog_parameters
    // struct at +0x2C..+0x48 (color = fog_color * fog_color_intensity, HDR).
    // Gated on a sanity test: a positive thickness in [0,1.5] with a finite
    // band height. If anything looks wrong we leave thickness 0 so the shader
    // collapses the band (no-op) rather than mis-fogging.
    // -----------------------------------------------------------------------
    float gndThick = 0.0f, gndHeight = 0.0f, gndBase = 0.0f, gndMaxDist = 0.0f;
    float gndCol[3] = {0,0,0}, gndIntensity = 1.0f;
    memcpy(&gndThick,    m + FOGG_GND_THICKNESS_OFF,   4);
    memcpy(&gndHeight,   m + FOGG_GND_HEIGHT_OFF,      4);
    memcpy(&gndBase,     m + FOGG_GND_BASE_HEIGHT_OFF, 4);
    memcpy(&gndMaxDist,  m + FOGG_GND_MAX_DIST_OFF,    4);
    memcpy(&gndIntensity,m + FOGG_GND_INTENSITY_OFF,   4);
    for (int i = 0; i < 3; ++i)
        memcpy(&gndCol[i], m + FOGG_GND_COLOR_OFF + i * 4, 4);

    auto finite1 = [](float v){ return v == v && v < 1e30f && v > -1e30f; };
    bool groundOk =
        finite1(gndThick) && finite1(gndHeight) && finite1(gndBase) &&
        finite1(gndMaxDist) && finite1(gndIntensity) &&
        finite1(gndCol[0]) && finite1(gndCol[1]) && finite1(gndCol[2]) &&
        // a real band: positive opacity in a plausible range + positive height
        (gndThick > 0.0001f && gndThick <= 1.5f) &&
        (gndHeight > 0.01f && gndHeight < 100000.0f) &&
        (gndBase > -100000.0f && gndBase < 100000.0f) &&
        (gndMaxDist >= 0.0f && gndMaxDist < 1e7f) &&
        (gndIntensity >= 0.0f && gndIntensity < 100.0f) &&
        (gndCol[0] >= 0.0f && gndCol[1] >= 0.0f && gndCol[2] >= 0.0f);

    if (groundOk) {
        outParams->GroundFogThickness   = gndThick;
        // Pre-multiply color by intensity (engine fog_color * fog_color_intensity).
        outParams->GroundFogColor[0]    = gndCol[0] * gndIntensity;
        outParams->GroundFogColor[1]    = gndCol[1] * gndIntensity;
        outParams->GroundFogColor[2]    = gndCol[2] * gndIntensity;
        outParams->GroundFogHeight      = gndHeight;
        outParams->GroundFogBaseHeight  = gndBase;
        outParams->GroundFogMaxDistance = (gndMaxDist > 1.0f) ? gndMaxDist : 1000.0f;
        outParams->HasGroundBand        = 1;
    } else {
        outParams->GroundFogThickness   = 0.0f;
        outParams->GroundFogColor[0]    = 0.0f;
        outParams->GroundFogColor[1]    = 0.0f;
        outParams->GroundFogColor[2]    = 0.0f;
        outParams->GroundFogHeight      = 0.0f;
        outParams->GroundFogBaseHeight  = 0.0f;
        outParams->GroundFogMaxDistance = 0.0f;
        outParams->HasGroundBand        = 0;
    }

    // -----------------------------------------------------------------------
    // FOG LIGHT 1 - directional sun-through-fog inscatter. The on-disk struct
    // stores the light DIRECTION as pitch/yaw angles (+0x4C/+0x50); we don't
    // export those because the shader recovers the direction from the scene
    // analytical sun (SceneAnalyticalSunDir) instead. We export the tint color
    // (* intensity), the two pow() falloff exponents, and the nearby cutoff so
    // the shader's fog-light term uses the per-scenario authored values rather
    // than hardcoded constants. Gated on sanity; defaults are NO-OP-safe.
    // -----------------------------------------------------------------------
    float flCol[3] = {0,0,0}, flIntensity = 1.0f;
    float flAngFall = 1.0f, flDistFall = 1.0f, flNearCut = 0.0f;
    if (getenv("ZH_FOGLIGHTDIAG")) {
        for (int off = 0x44; off <= 0x58; off += 4) {
            float v; memcpy(&v, m + off, 4);
            fprintf(stderr, "ZH_FOGLIGHTDIAG +0x%02X = %.5f\n", off, v);
        }
    }
    if (getenv("ZH_FOGGDUMP")) {
        for (int off = 0x00; off <= 0x4C; off += 4) {
            float v; memcpy(&v, m + off, 4);
            uint32_t iv; memcpy(&iv, m + off, 4);
            fprintf(stderr, "ZH_FOGGDUMP +0x%02X = %.5f (0x%08X)\n", off, v, iv);
        }
    }
    for (int i = 0; i < 3; ++i)
        memcpy(&flCol[i], m + FOGG_FL_TINT_COLOR_OFF + i * 4, 4);
    memcpy(&flIntensity, m + FOGG_FL_TINT_INTENS_OFF,  4);
    memcpy(&flAngFall,   m + FOGG_FL_ANG_FALLOFF_OFF,  4);
    memcpy(&flDistFall,  m + FOGG_FL_DIST_FALLOFF_OFF, 4);
    memcpy(&flNearCut,   m + FOGG_FL_NEARBY_CUTOFF_OFF,4);

    bool fogLightOk =
        finite1(flCol[0]) && finite1(flCol[1]) && finite1(flCol[2]) &&
        finite1(flIntensity) && finite1(flAngFall) && finite1(flDistFall) &&
        finite1(flNearCut) &&
        (flCol[0] >= 0.0f && flCol[1] >= 0.0f && flCol[2] >= 0.0f) &&
        (flIntensity >= 0.0f && flIntensity < 100.0f) &&
        // a real light: non-black tint after intensity scale
        ((flCol[0] + flCol[1] + flCol[2]) * flIntensity > 0.0005f) &&
        (flAngFall >= 0.0f && flAngFall < 1000.0f) &&
        (flDistFall >= 0.0f && flDistFall < 1000.0f) &&
        (flNearCut >= 0.0f && flNearCut <= 1.0f);

    if (fogLightOk) {
        outParams->FogLightColor[0]      = flCol[0] * flIntensity;
        outParams->FogLightColor[1]      = flCol[1] * flIntensity;
        outParams->FogLightColor[2]      = flCol[2] * flIntensity;
        // Guard against a degenerate 0 exponent (pow(x,0)==1 would make the
        // whole disc full-bright); the engine authors >= 1 in every sampled tag.
        outParams->FogLightAngularFalloff  = (flAngFall > 0.01f) ? flAngFall : 1.0f;
        outParams->FogLightDistanceFalloff = (flDistFall > 0.01f) ? flDistFall : 1.0f;
        outParams->FogLightNearbyCutoff    = flNearCut;
        outParams->HasFogLight             = 1;
        // World direction TO the light from pitch(+0x4C)/yaw(+0x50) (Halo Z-up), and
        // the disc shape from angular_radius(+0x54): ratio = saturate(cosine*scale + offset) maps the
        // half-angle to [0,1]. Guarded - a garbage read falls back to a wide neutral disc (scale 1).
        float flPitch = 0.0f, flYaw = 0.0f, flAngRad = 0.0f;
        memcpy(&flPitch,  m + 0x4C, 4);
        memcpy(&flYaw,    m + 0x50, 4);
        memcpy(&flAngRad, m + 0x54, 4);
        if (finite1(flPitch) && finite1(flYaw) && fabsf(flPitch) < 6.3f && fabsf(flYaw) < 6.3f) {
            float cp = cosf(flPitch), sp = sinf(flPitch), cy = cosf(flYaw), sy = sinf(flYaw);
            outParams->FogLightDir[0] = cp * cy;
            outParams->FogLightDir[1] = cp * sy;
            outParams->FogLightDir[2] = sp;
        } else {
            outParams->FogLightDir[0] = 0.0f; outParams->FogLightDir[1] = 0.0f; outParams->FogLightDir[2] = 1.0f;
        }
        if (finite1(flAngRad) && flAngRad > 0.001f && flAngRad < 3.14f) {
            float car = cosf(flAngRad);
            float denom = (1.0f - car); if (denom < 1e-4f) denom = 1e-4f;
            // Clamp scale so an ultra-tight authored radius still reads as a small disc, not 1 pixel.
            float scale = 1.0f / denom; if (scale > 200.0f) scale = 200.0f;
            outParams->FogLightRadiusScale  = scale;
            outParams->FogLightRadiusOffset = -car / denom;
        } else {
            outParams->FogLightRadiusScale = 1.0f; outParams->FogLightRadiusOffset = 0.0f;
        }
    } else {
        outParams->FogLightColor[0]      = 0.0f;
        outParams->FogLightColor[1]      = 0.0f;
        outParams->FogLightColor[2]      = 0.0f;
        outParams->FogLightAngularFalloff  = 32.0f; // legacy fallback exponent
        outParams->FogLightDistanceFalloff = 2.0f;
        outParams->FogLightNearbyCutoff    = 0.0f;
        outParams->HasFogLight             = 0;
        outParams->FogLightDir[0] = 0.0f; outParams->FogLightDir[1] = 0.0f; outParams->FogLightDir[2] = 1.0f;
        outParams->FogLightRadiusScale = 1.0f; outParams->FogLightRadiusOffset = 0.0f;
    }

    outParams->HasFog = 1;
    outParams->_pad  = 0;
    outParams->_pad2 = 0;

    // ---------------------------------------------------------------------
    // OFFSET-VERIFICATION BYTE DUMP (MMS_NATIVE_LOG-gated). The ground-band
    // and fog-light fogg offsets are NOT confirmed. This one-shot hex dump of
    // fogg meta +0x00..+0x70 lets the user read the raw floats and confirm
    // where ground thickness / ground color / base+height / fog-light live.
    // NativeDiag is already MMS_NATIVE_LOG-gated. Bounded read: +0x70 worst
    // case; only dump when the tag is large enough.
    // ---------------------------------------------------------------------
    if ((size_t)metaOff + 0x74 <= cache->size) {
        const uint8_t* d = m;
        for (int row = 0; row < 0x70; row += 0x10) {
            float f0, f1, f2, f3;
            memcpy(&f0, d + row + 0x0, 4);
            memcpy(&f1, d + row + 0x4, 4);
            memcpy(&f2, d + row + 0x8, 4);
            memcpy(&f3, d + row + 0xC, 4);
            NativeDiag("FogParser[DUMP] fogg+0x%02X: %08X %08X %08X %08X "
                       "| f=(%.4f, %.4f, %.4f, %.4f)",
                       row,
                       RU32(d + row + 0x0), RU32(d + row + 0x4),
                       RU32(d + row + 0x8), RU32(d + row + 0xC),
                       f0, f1, f2, f3);
        }
    }
    return true;
}

// Inner walker - caller must wrap in SEH.
bool GetParamsInner(CacheHandle* cache, uint32_t scnrTagId, ZH_FogParams* outParams)
{
    // Zero-init every field; HasFog stays 0 until a successful read.
    memset(outParams, 0, sizeof(*outParams));

    int paletteOff = PickFogPaletteOffset(cache->cacheType);
    int32_t foggId = ResolveActiveFoggTagId(cache, scnrTagId, paletteOff);

    if (foggId < 0) {
        // Diag identifies the corrected schema: scnr.Fog palette @ 0x534,
        // stride 0x18, fogg tagref at entry+0x08. If a future build shifts
        // any of these, this line is the obvious regression beacon (0x528 with
        // stride 0x8C is the Background Sound Environment Palette, not fog).
        NativeDiag("FogParser: scnr=%u no Fog palette fogg resolved "
                   "(scnr+0x%X stride=0x%X tagref@+0x%X) [FOG_RE_V3]",
                   scnrTagId, paletteOff, FOG_PALETTE_ENTRY_SIZE,
                   FOG_PALETTE_FOGG_TAGREF_OFF);
        return false;
    }

    bool ok = ReadFoggFields(cache, (uint32_t)foggId, outParams);
    if (!ok) {
        NativeDiag("FogParser: scnr=%u fogg=%d field read failed",
                   scnrTagId, foggId);
        return false;
    }
    NativeDiag("FogParser: scnr=%u fogg=%d thickness=%.4f bias=%.2f falloff=%.2f "
               "skyHeight=%.2f skyBase=%.2f heightBand=%u "
               "A=(%.2f,%.2f,%.2f) B=(%.2f,%.2f,%.2f) tint=(%.2f,%.2f,%.2f) "
               "[FOG_HEIGHT_BAND paletteOff=0x%X]",
               scnrTagId, foggId, outParams->Density, outParams->StartDistance, outParams->FalloffEnd,
               outParams->SkyFogHeight, outParams->SkyFogBaseHeight, outParams->HasHeightBand,
               outParams->InscatterA[0], outParams->InscatterA[1], outParams->InscatterA[2],
               outParams->InscatterB[0], outParams->InscatterB[1], outParams->InscatterB[2],
               outParams->SkyTint[0],    outParams->SkyTint[1],    outParams->SkyTint[2],
               paletteOff);
    NativeDiag("FogParser: scnr=%u fogg=%d GROUND band=%u thick=%.4f h=%.2f base=%.2f maxd=%.1f "
               "col=(%.3f,%.3f,%.3f) | FOGLIGHT on=%u col=(%.3f,%.3f,%.3f) angFall=%.2f distFall=%.2f nearCut=%.3f "
               "[FOGG_GROUND_BAND_RE]",
               scnrTagId, foggId, outParams->HasGroundBand, outParams->GroundFogThickness,
               outParams->GroundFogHeight, outParams->GroundFogBaseHeight, outParams->GroundFogMaxDistance,
               outParams->GroundFogColor[0], outParams->GroundFogColor[1], outParams->GroundFogColor[2],
               outParams->HasFogLight, outParams->FogLightColor[0], outParams->FogLightColor[1],
               outParams->FogLightColor[2], outParams->FogLightAngularFalloff,
               outParams->FogLightDistanceFalloff, outParams->FogLightNearbyCutoff);
    return true;
}

bool SehGetParams(CacheHandle* cache, uint32_t scnrTagId, ZH_FogParams* outParams)
{
    __try { return GetParamsInner(cache, scnrTagId, outParams); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outParams) memset(outParams, 0, sizeof(*outParams));
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

extern "C" __declspec(dllexport) bool __stdcall ZH_FOG_GetParams(
    uint64_t cacheHandle, uint32_t scnrTagId, ZH_FogParams* outParams)
{
    if (!outParams) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        memset(outParams, 0, sizeof(*outParams));
        return false;
    }
    return SehGetParams(cache, scnrTagId, outParams);
}
