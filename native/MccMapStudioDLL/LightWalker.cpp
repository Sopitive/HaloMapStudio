// LightWalker.cpp
// =============================================================================
// Native walker for the analytical sun
// `ligh` tag referenced from a Halo Reach .map's scnr metadata.
//
// === R2 Schema correction (this revision) ===
//
// The previous revision walked `scnr+0x528` as an "AtmospherePalette" tagblock
// and scanned its entries (stride 0x8C) for embedded 'ligh' fourccs. That was
// wrong on two counts:
//
//   1) Reach scnr has NO `AtmospherePalette` block at all. The 0x528 offset
//      was inherited from a copy-paste in FogParser.cpp that ALSO targets
//      0x528 - but per Lord Zedd's `ReachMCC/scnr.xml` plugin (line 6009),
//      scnr+0x528 is actually the `Background Sound Environment Palette`
//      tagblock (elementSize 0x78). FogParser happens to find plausible
//      fogg tagrefs there only because the scanner reads past the end of
//      each 0x78-byte sound-env entry into adjacent palette data.
//
//   2) Reach's per-cluster atmosphere is decomposed into four separate
//      palettes (Background Sound Env @ 0x528, Fog @ 0x534, Camera FX @ 0x540,
//      Weather @ 0x54C). None of these contain `ligh` tagrefs, so scanning
//      for 'ligh' in any of them returns -1 every time - exactly the
//      `paletteOff=0x528` log line that motivated this fix.
//
// === The actual scnr fields with sun/light data (ReachMCC schema) ===
//
//   scnr+0x6D8  tagRef "Sky Parameters"       (sun direction lives downstream)
//   scnr+0x6E8  tagRef "Fog Parameters"
//   scnr+0x6F8  tagRef "Global Lighting"      (gldf class - color/intensity)
//   scnr+0x788  tagblock "Cinematic Lights"   (elementSize 0x14):
//                  +0x00 stringid Name
//                  +0x04 tagRef   Light       <- direct `ligh` tagref (16 B)
//
// The Cinematic Lights tagblock is the most reliable single source of
// directly-resolvable `ligh` tagrefs reachable from scnr in Reach. The sky
// scenery walk (scnr.Skies -> scen -> render_model -> light marker -> ligh)
// would be more "engine-faithful" for analytical sun but requires four
// dereferences and a model parse - out of scope for this walker.
//
// === Fallback strategy ===
//
// Many Reach maps have no Cinematic Lights authored (the campaign uses them
// for in-engine cutscenes; multiplayer/forge maps typically have an empty
// block). For those cases we fall back to scanning the global tag table for
// `ligh` tags and picking the most sun-like one:
//
//   * Type == 1 (Projective) - Reach's "sun" is encoded as
//                                          Projective with effectively infinite
//                                          range (Sphere lights are point/area).
//   * Largest LightRange wins - sun has the longest reach.
//   * On tie, larger SunDiskFovDeg wins - sun has a defined disk; non-sun
//                                          projectives tend to be tight cones.
//
// This produces consistent, deterministic results per map without needing
// scene-specific authoring metadata. If a future RE pass adds a proper
// scnr.Skies -> scen -> ligh walk we can promote it ahead of this fallback.
//
// === ligh field layout (unchanged from R1, baseSize 0xCC) ===
//
//     +0x02 flags16    Flags (bit 1 = ShadowCasting, ...)
//     +0x04 enum16     Type (0=Sphere, 1=Projective)
//     +0x08 float      Light Range
//     +0x0C float      Maximum Intensity Range (intensity scalar)
//     +0x1C float      Field Of View (sun-disk angular DIAMETER in degrees)
//     +0x20 float      Outer cone half-angle (degrees)
//     +0x34 int16      Shadow Resolution Selector
//     +0x80 tagref(16) Gel Map bitmap          (TagId at +0x80+12 = +0x8C)
//     +0x9C tagref(16) Lens Flare              (TagId at +0x9C+12 = +0xA8)
//   COLOR is a runtime-evaluated function dataref at +0x48 - not statically
//   decoded by this walker. Consumer should provide its own default tint.
//
// Defensive contract (same as FogParser):
//   * SEH wrapper around the full walk - any read fault returns false.
//   * Index validation against cache->tags.size() before each tag deref.
//   * Class-code check confirms 'ligh' at the resolved tag id.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>

using namespace zh_mcc;

// Public ABI - must match the Rust mirror in crates/hms-native/src/lib.rs.
#pragma pack(push, 1)
struct ZH_SunLight {
    uint32_t HasLight;           // 0 = no atmospheric ligh / unresolved, 1 = OK
    uint32_t LightType;          // 0=Sphere, 1=Projective
    float    Intensity;          // ligh+0x0C
    float    SunDiskFovDeg;      // ligh+0x1C
    float    OuterConeDeg;       // ligh+0x20
    float    LightRange;         // ligh+0x08
    uint16_t Flags;              // ligh+0x02
    uint16_t ShadowResSelector;  // ligh+0x34
    uint32_t GelMapTagId;        // ligh+0x8C
    uint32_t LensFlareTagId;     // ligh+0xA8
    uint32_t _pad[4];
};

// Analytical "sky light" (the map's outdoor sun) authored INSIDE the sky
// render_model, NOT the ligh tag. Ground truth (MCC Reach U13):
//   Sky Light direction : mode+0x224 (point3; the authored vector, negate for L)
//   Sky Light RGB (HDR) : mode+0x230 / +0x234 / +0x238
//   Natural-light SH DC : R @ mode+0x124, G @ +0x164, B @ +0x1A4 (per-channel SH,
//                         0x40 stride; DC term = flat ambient colour)
// The ligh tag carries only a colourless intensity/FOV (colour is a Color-Change
// function), so THIS is where a Reach map's real sun colour + ambient live.
struct ZH_SkyLight {
    uint32_t HasLight;   // 0 = unresolved, 1 = OK
    float    Dir[3];     // authored sky-light vector (mode+0x224)
    float    Color[3];   // HDR linear sun RGB (mode+0x230..0x238)
    float    Ambient[3]; // natural-light SH DC per channel (mode+0x124/0x164/0x1A4)
    uint32_t _pad[3];
};

static inline float RfLE(const uint8_t* p) { float v; memcpy(&v, p, sizeof(v)); return v; }

// One reflection (flare sprite) element. Pack=1, MUST match the Rust mirror of ZH_LensFlareElement.
struct ZH_LensFlareElement {
    float    AxisOffset;        // refl+0x08 - corona-axis fraction (0=source..1=edge)
    float    RotationOffset;    // refl+0x04
    float    RadiusMin;         // refl+0x0C - 0 => use radius curve (viewer default)
    float    RadiusMax;         // refl+0x10
    float    BrightnessMin;     // refl+0x14
    float    BrightnessMax;     // refl+0x18
    float    ModulationFactor;  // refl+0x44 - shader modulation_factor.x
    float    ColorR;            // refl+0x48
    float    ColorG;            // refl+0x4C
    float    ColorB;            // refl+0x50
    float    TintPower;         // refl+0x54 - shader modulation_factor.y (gamma)
    int32_t  BitmapIndex;       // refl+0x02 - sprite-sequence index into base atlas
    uint32_t Flags8;            // refl+0x00
    uint32_t RadiusCurveSize;   // raw dataref size @ refl+0x1C (uncertain - logged)
    uint32_t BrightCurveSize;   // raw dataref size @ refl+0x30 (uncertain - logged)
};

// Per-lens header info. Pack=1, MUST match the Rust mirror of ZH_LensFlareInfo.
struct ZH_LensFlareInfo {
    float    FalloffAngleRad;             // lens+0x00
    float    CutoffAngleRad;              // lens+0x04
    float    OcclusionInnerRadiusScale;   // lens+0x10
    uint32_t Flags;                       // lens+0x34 (flags16)
    uint32_t BitmapTagId;                 // lens+0x30 (bitm atlas tag id, 0xFFFF=none)
};
#pragma pack(pop)

namespace {

// --- ligh field offsets (Reach schema, baseSize 0xCC) ---
constexpr int LGHT_FLAGS_OFF         = 0x02;
constexpr int LGHT_TYPE_OFF          = 0x04;
constexpr int LGHT_RANGE_OFF         = 0x08;
constexpr int LGHT_MAX_INTENSITY_OFF = 0x0C;
constexpr int LGHT_FOV_OFF           = 0x1C;
constexpr int LGHT_OUTER_DEG_OFF     = 0x20;
constexpr int LGHT_SHADOW_RES_OFF    = 0x34;
constexpr int LGHT_GEL_TAGREF_OFF    = 0x80;
constexpr int LGHT_LENS_TAGREF_OFF   = 0x9C;
constexpr int LGHT_REQUIRED_BYTES    = 0xCC;

constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_LIGH = "ligh";
constexpr const char* TC_SCEN = "scen";   // scenery tag class (sky is scenery in Reach)
constexpr const char* TC_HLMT = "hlmt";   // model tag class
constexpr const char* TC_MODE = "mode";   // render_model tag class

// --- scnr.Skies[] -> scenery -> hlmt -> render_model chain ---
// Mirrors SkyWalker.cpp (the SAME walk used to render the sky dome). Reclaimer's
// HaloReach scenario.config.cs:
//   Skies @ scnr+132 (MccHaloReach .. U10), scnr+136 (MccHaloReachU13)
//   SkyReferenceBlock fixed size = 48 B; SkyReference (TagReference) @ +0
//   scenery (ObjectTagBase) Model TagReference @ scen+100  -> hlmt
//   model (hlmt) RenderModel TagReference @ hlmt+0          -> mode
// The map's authored OUTDOOR sun is the `ligh` referenced by a light marker in
// that sky scenery's own render_model. We resolve it by scanning the render_model
// meta HEADER region (where the model's tagblock/tagref fields live) for the
// first VALIDATED `ligh` tagref. This is scoped to the map's own sky model, so
// the resolved light is the sun that sky authored - NOT an arbitrary cache-wide
// "most sun-like" ligh (the sunset-hue regression). If the sky model
// carries no ligh tagref, we return -1 (no global heuristic fallback).
constexpr int SKY_BLOCK_SIZE             = 48;
constexpr int SKY_REF_OFFSET             = 0;
constexpr int SCENERY_MODEL_REF_OFFSET   = 100;  // scen+100 -> hlmt tagref
constexpr int HLMT_RENDER_MODEL_REF_OFF  = 0;    // hlmt+0   -> mode tagref
// How far into the render_model meta to scan for a `ligh` tagref. The Reach
// render_model header (flags, bounds, marker groups, node/region/section block
// headers, and the light-marker block) all live in the first few hundred bytes
// of meta before the geometry resource pointers. 0x600 covers the header field
// region across U3..U13 schema drift without walking into geometry payload.
constexpr int MODE_LIGH_SCAN_BYTES       = 0x600;

// --- scnr.CinematicLights tagblock (per ReachMCC scnr.xml line 6750) ---
//   ReachMCC builds: offset 0x788, entry stride 0x14, Light tagref @ +0x04.
// Pre-U13 MCC builds may shift this - we try a window of candidate offsets
// because the (small) handful of fields between scnr+0x600 and scnr+0x788
// have drifted across U3/U8/U10 schema revisions.
constexpr int CINEMATIC_LIGHTS_ENTRY_SIZE = 0x14;
constexpr int CINEMATIC_LIGHTS_LIGHT_REF  = 0x04;

// Candidate offsets to try for scnr.CinematicLights tagblock header (8 B:
// int32 count + uint32 pointer). 0x788 is the verified ReachMCC value; the
// remaining offsets are +-0x14 / +-0x24 windows to absorb cross-build drift
// without having to maintain a per-CacheType table. A wrong offset reads a
// bogus count/pointer that fails our bounds checks and we move on.
const int kCinematicLightsCandidates[] = {
    0x788, 0x774, 0x77C, 0x780, 0x784, 0x78C, 0x790, 0x794, 0x79C, 0x7A8, 0x7B4,
};

// --- Tag-reference helpers ---
// TagReference layout (Gen3+): ClassId @ +0, padding @ +4..+11, TagId @ +12.
// Returns -1 only for the 0xFFFFFFFF null sentinel; high-bit identity salt
// (e.g. 0xA60Cxxxx) is valid and we mask to 16 bits.
int32_t ReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// Confirm `tagId` exists and has class 'ligh'. Returns true if usable.
bool IsValidLighTag(CacheHandle* cache, uint32_t tagId) {
    if (tagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[tagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, TC_LIGH, 4) != 0) return false;
    return true;
}

// Read scnr tagblock header at (scnr_meta + offset). Returns false on any
// bounds failure (the offset is wrong for this build) or implausible count.
bool ReadScnrTagBlock(CacheHandle* cache, const uint8_t* scnrMeta, size_t scnrAvail,
                      int offset, int32_t* outCount, uint32_t* outPointer)
{
    if (offset < 0 || (size_t)offset + 8 > scnrAvail) return false;
    TagBlockRef blk = ReadTagBlock(scnrMeta + offset);
    if (blk.count < 0 || blk.count > 0x100) return false;
    *outCount   = blk.count;
    *outPointer = blk.pointer;
    return true;
}

// Walk a CinematicLights-shaped tagblock (stride 0x14, Light tagref @ +0x04)
// and return the first valid `ligh` tag id, or -1 if none.
int32_t FindLighInCinematicBlock(CacheHandle* cache, int32_t count, uint32_t blockPointer)
{
    if (count <= 0) return -1;
    int64_t arrayOff = TagMetaFileOff(cache, blockPointer);
    if (arrayOff < 0) return -1;
    if ((size_t)arrayOff + (size_t)count * CINEMATIC_LIGHTS_ENTRY_SIZE > cache->size)
        return -1;
    const uint8_t* base = cache->base + arrayOff;
    for (int i = 0; i < count; ++i) {
        const uint8_t* entry = base + (size_t)i * CINEMATIC_LIGHTS_ENTRY_SIZE;
        // Tagref is 16 B starting at entry+0x04. Sanity: confirm the entry
        // fully fits before the tagref read (already covered by the array
        // bounds check above, since 0x04+16 = 0x14 = stride).
        int32_t id = ReadTagRefId(entry + CINEMATIC_LIGHTS_LIGHT_REF);
        if (id < 0) continue;
        if (!IsValidLighTag(cache, (uint32_t)id)) continue;
        return id;
    }
    return -1;
}

// NOTE: The old cache-wide "most sun-like" heuristic
// (ScoreSunLikeness + FindBestSunLighInCache) was REMOVED. It picked an
// arbitrary global Projective ligh on MP/Forge maps and resolved to a cinematic
// sunset light -> user-visible reddish-sunset hue regression. The
// map-scoped sky-chain resolver below replaces it.

// ----- Strategy B (replacement): scnr.Skies -> scen -> hlmt -> render_model,
//       then scan THAT render_model's meta header for the authored sun `ligh`.
//
// This re-uses the exact tag chain SkyWalker.cpp walks to render the sky dome.
// Because the search is scoped to the map's own sky scenery model, any `ligh`
// it references IS that map's outdoor sun - so it does NOT reintroduce the
// cache-wide "most sun-like" sunset-hue regression. -1 if the chain
// breaks or the model has no ligh ref.

int PickScnrSkiesOffset(CacheType ct) {
    switch (ct) {
        case CacheType::MccHaloReachU13: return 136;
        default:                         return 132;
    }
}

// scnr.Skies[skyIndex] -> validated scenery (scen) tag id, or -1.
int32_t ResolveSkySceneryTagId(CacheHandle* cache, const uint8_t* scnrMeta,
                               size_t scnrAvail, int skyIndex, int32_t* outSkyCount)
{
    if (outSkyCount) *outSkyCount = 0;
    int skiesOff = PickScnrSkiesOffset(cache->cacheType);
    if (skiesOff < 0 || (size_t)skiesOff + 8 > scnrAvail) return -1;
    TagBlockRef blk = ReadTagBlock(scnrMeta + skiesOff);
    if (blk.count <= 0 || blk.count > 0x10000) return -1;
    if (outSkyCount) *outSkyCount = blk.count;
    if (skyIndex >= blk.count) return -1;

    int64_t arrOff = TagMetaFileOff(cache, blk.pointer);
    if (arrOff < 0) return -1;
    if ((size_t)arrOff + (size_t)blk.count * SKY_BLOCK_SIZE > cache->size) return -1;

    const uint8_t* entry = cache->base + arrOff + (size_t)skyIndex * SKY_BLOCK_SIZE;
    int32_t scenId = ReadTagRefId(entry + SKY_REF_OFFSET);
    if (scenId < 0 || (uint32_t)scenId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[scenId].classCode, TC_SCEN, 4) != 0) return -1;
    return scenId;
}

// Follow a tagref at (tag meta + refOff) and confirm the target class. -1 on fail.
int32_t FollowTagRef(CacheHandle* cache, uint32_t srcTagId, const char* srcClass4,
                     int refOff, const char* dstClass4)
{
    if (srcTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[srcTagId];
    if (memcmp(te.classCode, srcClass4, 4) != 0) return -1;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return -1;
    if ((size_t)metaOff + (size_t)refOff + 16 > cache->size) return -1;
    int32_t dstId = ReadTagRefId(cache->base + metaOff + refOff);
    if (dstId < 0 || (uint32_t)dstId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[dstId].classCode, dstClass4, 4) != 0) return -1;
    return dstId;
}

// Scan a render_model's (mode) meta header region for the first VALIDATED
// `ligh` tagref. Tagrefs store the TagId at +12, so we step word-aligned and
// read each candidate's +12 dword, validate it indexes a real `ligh` tag, and
// additionally require the tagref's ClassId dword (at +0) to look like a class
// pointer (high bit set / non-zero) - this filters out raw geometry ints that
// happen to alias a valid low-16 ligh index. Returns the ligh id, or -1.
int32_t FindSunLighInRenderModel(CacheHandle* cache, uint32_t modeTagId)
{
    if (modeTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[modeTagId];
    if (memcmp(te.classCode, TC_MODE, 4) != 0) return -1;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return -1;

    size_t avail = (size_t)cache->size - (size_t)metaOff;
    size_t scanLen = avail < (size_t)MODE_LIGH_SCAN_BYTES ? avail : (size_t)MODE_LIGH_SCAN_BYTES;
    if (scanLen < 16) return -1;
    const uint8_t* m = cache->base + metaOff;

    // A TagReference is 16 B: [ClassId u32][unused u32][unused u32][TagId u32].
    // Step by 4 (tagrefs are 4-aligned inside meta) and test each as a tagref base.
    for (size_t off = 0; off + 16 <= scanLen; off += 4) {
        const uint8_t* ref = m + off;
        uint32_t classId = RU32(ref + 0);
        // Reach class ids are fourcc-packed (e.g. 'ligh'=0x6C696768) with a high
        // identity salt; a zero/0xFFFFFFFF classId word is never a live tagref.
        if (classId == 0 || classId == 0xFFFFFFFFu) continue;
        int32_t id = ReadTagRefId(ref);
        if (id < 0) continue;
        if (!IsValidLighTag(cache, (uint32_t)id)) continue;
        // Extra guard: the ClassId word of a `ligh` tagref decodes to the 'ligh'
        // fourcc in its low bytes on these builds. Accept either exact low-fourcc
        // match OR (defensively) any classId whose target tag we already
        // class-validated as 'ligh' above - the latter covers salt-only builds.
        return id;
    }
    return -1;
}

// Walk every sky in scnr.Skies, returning the first authored sun `ligh`.
int32_t ResolveSunFromSkies(CacheHandle* cache, const uint8_t* scnrMeta,
                            size_t scnrAvail, int* outSkyUsed, int32_t* outModeId)
{
    if (outSkyUsed) *outSkyUsed = -1;
    if (outModeId)  *outModeId  = -1;
    int32_t skyCount = 0;
    // Probe count via index 0 (also returns count).
    ResolveSkySceneryTagId(cache, scnrMeta, scnrAvail, 0, &skyCount);
    if (skyCount <= 0) return -1;
    if (skyCount > 64) skyCount = 64;  // bound the loop

    for (int i = 0; i < skyCount; ++i) {
        int32_t scenId = ResolveSkySceneryTagId(cache, scnrMeta, scnrAvail, i, nullptr);
        if (scenId < 0) continue;
        int32_t hlmtId = FollowTagRef(cache, (uint32_t)scenId, TC_SCEN,
                                      SCENERY_MODEL_REF_OFFSET, TC_HLMT);
        if (hlmtId < 0) continue;
        int32_t modeId = FollowTagRef(cache, (uint32_t)hlmtId, TC_HLMT,
                                      HLMT_RENDER_MODEL_REF_OFF, TC_MODE);
        if (modeId < 0) continue;
        int32_t lighId = FindSunLighInRenderModel(cache, (uint32_t)modeId);
        if (lighId >= 0) {
            if (outSkyUsed) *outSkyUsed = i;
            if (outModeId)  *outModeId  = modeId;
            return lighId;
        }
    }
    return -1;
}

// Resolve the active sun-light tag id for this scenario.
//   Strategy A: scnr.CinematicLights tagblock (try a window of candidate
//               offsets to absorb cross-build schema drift).
//   Strategy B: scnr.Skies -> scenery -> hlmt -> render_model, scan that
//               model's meta for the authored sun ligh (map-scoped, safe).
//
// Returns the tag id, or -1 if nothing resolves. `outSource` is filled with
// a short human-readable label for the diag log.
int32_t ResolveSunLighTagId(CacheHandle* cache, uint32_t scnrTagId,
                            const char** outSource, int* outSourceOffset)
{
    *outSource = "(none)";
    *outSourceOffset = -1;

    if (scnrTagId >= cache->tags.size()) return -1;
    const TagEntry& te = cache->tags[scnrTagId];
    if (te.classIndex < 0) return -1;
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return -1;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return -1;
    size_t scnrAvail = (size_t)cache->size - (size_t)metaOff;
    const uint8_t* scnrMeta = cache->base + metaOff;

    // Strategy A: walk CinematicLights at each candidate offset.
    for (int off : kCinematicLightsCandidates) {
        int32_t  count   = 0;
        uint32_t pointer = 0;
        if (!ReadScnrTagBlock(cache, scnrMeta, scnrAvail, off, &count, &pointer))
            continue;
        if (count == 0) continue;  // empty block (very common) - try next offset / fallback
        int32_t id = FindLighInCinematicBlock(cache, count, pointer);
        if (id >= 0) {
            *outSource = "scnr.CinematicLights";
            *outSourceOffset = off;
            return id;
        }
    }

    // Strategy B: scnr.Skies -> scenery -> hlmt -> render_model,
    // scan that model's meta for the authored sun `ligh`. Map-scoped, so it
    // resolves the sun THIS sky authored (no cache-wide "most sun-like"
    // heuristic - that was the sunset-hue regression, now avoided).
    {
        int  skyUsed = -1;
        int32_t modeId = -1;
        int32_t id = ResolveSunFromSkies(cache, scnrMeta, scnrAvail, &skyUsed, &modeId);
        if (id >= 0) {
            *outSource = "scnr.Skies->scen->mode";
            *outSourceOffset = skyUsed;  // reused as sky index for the diag log
            NativeDiag("LightWalker: scnr=%u sky[%d] mode=%d -> ligh=%d (authored sun)",
                       scnrTagId, skyUsed, modeId, id);
            return id;
        }
    }

    // Nothing resolved. Return -1; the caller's HasLight=0 sentinel makes
    // RecomputeSunDiffuse fall back to its NEUTRAL default sun (near-white,
    // unit-luma, a=1) so directional lighting + shadows still work. We do NOT
    // fall back to a cache-wide "most sun-like" scan (that was the sunset-hue
    // regression).
    return -1;
}

bool ReadLighFields(CacheHandle* cache, uint32_t lighTagId, ZH_SunLight* out)
{
    if (!IsValidLighTag(cache, lighTagId)) return false;
    const TagEntry& te = cache->tags[lighTagId];
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + (size_t)LGHT_REQUIRED_BYTES > cache->size) return false;
    const uint8_t* m = cache->base + metaOff;

    out->Flags       = (uint16_t)RU16(m + LGHT_FLAGS_OFF);
    out->LightType   = (uint32_t)RU16(m + LGHT_TYPE_OFF);
    memcpy(&out->LightRange,    m + LGHT_RANGE_OFF,         4);
    memcpy(&out->Intensity,     m + LGHT_MAX_INTENSITY_OFF, 4);
    memcpy(&out->SunDiskFovDeg, m + LGHT_FOV_OFF,           4);
    memcpy(&out->OuterConeDeg,  m + LGHT_OUTER_DEG_OFF,     4);
    out->ShadowResSelector = (uint16_t)RU16(m + LGHT_SHADOW_RES_OFF);
    out->GelMapTagId    = RU32(m + LGHT_GEL_TAGREF_OFF + 12) & 0xFFFFu;
    out->LensFlareTagId = RU32(m + LGHT_LENS_TAGREF_OFF + 12) & 0xFFFFu;
    // Sanity defaults - engine values typically positive + finite. Anything
    // exotic (NaN, negative, > 1e6) -> clear so consumer falls back to defaults.
    auto isSane = [](float v, float lo, float hi) {
        return (v == v) && v >= lo && v <= hi;
    };
    if (!isSane(out->Intensity,     0.0f, 1.0e6f)) out->Intensity     = 0.0f;
    if (!isSane(out->SunDiskFovDeg, 0.0f, 180.0f)) out->SunDiskFovDeg = 0.0f;
    if (!isSane(out->OuterConeDeg,  0.0f, 180.0f)) out->OuterConeDeg  = 0.0f;
    if (!isSane(out->LightRange,    0.0f, 1.0e9f)) out->LightRange    = 0.0f;
    out->HasLight = 1;
    return true;
}

bool GetSunLightInner(CacheHandle* cache, uint32_t scnrTagId, ZH_SunLight* out)
{
    memset(out, 0, sizeof(*out));
    const char* source = "(none)";
    int sourceOff = -1;
    int32_t lighId = ResolveSunLighTagId(cache, scnrTagId, &source, &sourceOff);
    if (lighId < 0) {
        NativeDiag("LightWalker: scnr=%u no atmospheric ligh resolved "
                   "(tried CinematicLights window + cache-wide scan)",
                   scnrTagId);
        return false;
    }
    if (!ReadLighFields(cache, (uint32_t)lighId, out)) {
        NativeDiag("LightWalker: scnr=%u ligh=%d field read failed (source=%s)",
                   scnrTagId, lighId, source);
        return false;
    }
    if (sourceOff > 0) {
        NativeDiag("LightWalker: scnr=%u ligh=%d source=%s@scnr+0x%X type=%u "
                   "intensity=%.3f fov=%.2f outer=%.2f range=%.1f gel=%u lens=%u",
                   scnrTagId, lighId, source, sourceOff,
                   out->LightType, out->Intensity, out->SunDiskFovDeg,
                   out->OuterConeDeg, out->LightRange,
                   out->GelMapTagId, out->LensFlareTagId);
    } else {
        NativeDiag("LightWalker: scnr=%u ligh=%d source=%s type=%u "
                   "intensity=%.3f fov=%.2f outer=%.2f range=%.1f gel=%u lens=%u",
                   scnrTagId, lighId, source,
                   out->LightType, out->Intensity, out->SunDiskFovDeg,
                   out->OuterConeDeg, out->LightRange,
                   out->GelMapTagId, out->LensFlareTagId);
    }
    return true;
}

bool SehGetSunLight(CacheHandle* cache, uint32_t scnrTagId, ZH_SunLight* out)
{
    __try { return GetSunLightInner(cache, scnrTagId, out); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (out) memset(out, 0, sizeof(*out));
        return false;
    }
}

// ============================================================================
// LENS FLARE element walker  (LENS_FLARE_RE)
// ============================================================================
//
// Recovers the per-element "reflections" chain from a `lens` (lens_flare) tag - 
// the marching across-screen flare sprites. The sky-light's LensFlareTagId
// (ligh+0xA8, already captured by ZH_SCNR_GetSunLight) is the entry point.
//
// === Tag layout - GROUND TRUTH ===
// Verified by parsing real shipped Reach .map `lens` tags through THIS cache
// parser (boneyard flare_m10_*, 35_island levels\shared\post_fx\bright_sun_flare
// with 21 reflections, ghost/wraith projectile flares). The MCC HaloReach
// `lens` tag uses the SAME field layout as the Xbox-360 Reach Assembly plugin
// (Plugins\Reach\lens.xml, baseSize 0x9C, reflection elementSize 0x58):
//
//   lens base struct:
//     +0x00 float   Falloff Angle (radians)
//     +0x04 float   Cutoff Angle  (radians)
//     +0x10 float   Occlusion Inner Radius Scale
//     +0x24 tagRef  Bitmap (sprite atlas; TagId @ +0x24+12 = +0x30)
//     +0x34 flags16 Flags
//     +0x44 tagblock Reflections (8-byte header: int32 count + uint32 ptr)
//                    element stride = 0x58
//
//   Reflection element (stride 0x58):
//     +0x00 uint8   Flags
//     +0x02 int16   Bitmap (sprite-sequence) Index into the base Bitmap atlas
//     +0x04 float   Rotation Offset
//     +0x08 float   Axis Offset   (fraction along corona axis: 0=on corona/at
//                                  source, 1=primary side screen edge, -1=opp.)
//     +0x0C rangef  Radius     (min,max) - 0,0 => size comes from radius curve
//     +0x14 rangef  Brightness (min,max) - 0,0 => from brightness curve
//     +0x1C dataref Radius Curve Function    (s_function, NOT decoded here)
//     +0x30 dataref Brightness Curve Function(s_function, NOT decoded here)
//     +0x44 float   Modulation Factor   (shader modulation_factor.x)
//     +0x48 colorf  Color RGB           (tint_color.rgb)
//     +0x54 float   Tint Power          (shader modulation_factor.y, gamma pow)
//
// Each element maps 1:1 to the lens_flare PS (HREK effects/lens_flare):
//   out = modulation_factor.x*color^pow + color*tint_color;
//   brightness = tint_color.a*ILLUM_EXPOSURE*scale.r*modulation_factor.z
// (alpha/scale.r/mod.z are runtime exposure terms supplied by the consumer.)
//
// We DO NOT decode the radius/brightness curve datarefs (packed s_function
// blobs). Elements whose radius rangef is (0,0) leave RadiusMin/Max=0 - the
// viewer substitutes a fixed engine-faithful default radius. RawCurveSizes
// are surfaced so a future RE pass can decode them.
// ============================================================================

// lens base offsets
constexpr int LENS_OFF_FALLOFF_ANGLE  = 0x00;
constexpr int LENS_OFF_CUTOFF_ANGLE   = 0x04;
constexpr int LENS_OFF_OCC_INNER_SCALE= 0x10;
constexpr int LENS_OFF_BITMAP_TAGREF  = 0x24;   // TagId at +0x30
constexpr int LENS_OFF_FLAGS          = 0x34;
constexpr int LENS_OFF_REFLECTIONS    = 0x44;   // tagblock (count + ptr)
constexpr int LENS_REFL_STRIDE        = 0x58;
constexpr int LENS_REQUIRED_BYTES     = 0x9C;

// reflection element offsets
constexpr int REFL_OFF_FLAGS8         = 0x00;
constexpr int REFL_OFF_BITMAP_INDEX   = 0x02;
constexpr int REFL_OFF_ROT_OFFSET     = 0x04;
constexpr int REFL_OFF_AXIS_OFFSET    = 0x08;
constexpr int REFL_OFF_RADIUS_MIN     = 0x0C;
constexpr int REFL_OFF_RADIUS_MAX     = 0x10;
constexpr int REFL_OFF_BRIGHT_MIN     = 0x14;
constexpr int REFL_OFF_BRIGHT_MAX     = 0x18;
constexpr int REFL_OFF_RADIUS_CURVE   = 0x1C;   // dataref (size @ +0)
constexpr int REFL_OFF_BRIGHT_CURVE   = 0x30;   // dataref (size @ +0)
constexpr int REFL_OFF_MODULATION     = 0x44;
constexpr int REFL_OFF_COLOR_RGB      = 0x48;
constexpr int REFL_OFF_TINT_POWER     = 0x54;

constexpr const char* TC_LENS = "lens";

bool IsValidLensTag(CacheHandle* cache, uint32_t tagId) {
    if (tagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[tagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, TC_LENS, 4) != 0) return false;
    return true;
}

bool GetLensElementsInner(CacheHandle* cache, uint32_t lensTagId,
                          ZH_LensFlareElement* outBuf, int maxElems,
                          uint32_t* outCount, uint32_t* outTotalSeen,
                          ZH_LensFlareInfo* outInfo)
{
    *outCount = 0;
    *outTotalSeen = 0;
    if (outInfo) memset(outInfo, 0, sizeof(*outInfo));

    if (!IsValidLensTag(cache, lensTagId)) {
        NativeDiag("LensWalker: tag %u is not a valid 'lens' tag", lensTagId);
        return false;
    }
    const TagEntry& te = cache->tags[lensTagId];
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + (size_t)LENS_REQUIRED_BYTES > cache->size) return false;
    const uint8_t* m = cache->base + metaOff;

    if (outInfo) {
        memcpy(&outInfo->FalloffAngleRad, m + LENS_OFF_FALLOFF_ANGLE, 4);
        memcpy(&outInfo->CutoffAngleRad,  m + LENS_OFF_CUTOFF_ANGLE,  4);
        memcpy(&outInfo->OcclusionInnerRadiusScale, m + LENS_OFF_OCC_INNER_SCALE, 4);
        outInfo->Flags        = (uint32_t)RU16(m + LENS_OFF_FLAGS);
        outInfo->BitmapTagId  = RU32(m + LENS_OFF_BITMAP_TAGREF + 12) & 0xFFFFu;
    }

    TagBlockRef blk = ReadTagBlock(m + LENS_OFF_REFLECTIONS);
    if (blk.count <= 0 || blk.count > 0x400) {
        NativeDiag("LensWalker: lens %u has implausible reflection count=%d",
                   lensTagId, blk.count);
        return true;  // valid lens, just no usable reflections
    }
    *outTotalSeen = (uint32_t)blk.count;

    int64_t arrOff = TagMetaFileOff(cache, blk.pointer);
    if (arrOff < 0) return false;
    if ((size_t)arrOff + (size_t)blk.count * LENS_REFL_STRIDE > cache->size) {
        NativeDiag("LensWalker: lens %u reflection array OOB (count=%d)",
                   lensTagId, blk.count);
        return false;
    }
    const uint8_t* base = cache->base + arrOff;

    int n = blk.count < maxElems ? blk.count : maxElems;
    for (int i = 0; i < n; ++i) {
        const uint8_t* el = base + (size_t)i * LENS_REFL_STRIDE;
        ZH_LensFlareElement* o = &outBuf[i];
        memset(o, 0, sizeof(*o));
        o->Flags8       = el[REFL_OFF_FLAGS8];
        o->BitmapIndex  = (int32_t)R16(el + REFL_OFF_BITMAP_INDEX);
        memcpy(&o->RotationOffset,   el + REFL_OFF_ROT_OFFSET,  4);
        memcpy(&o->AxisOffset,       el + REFL_OFF_AXIS_OFFSET, 4);
        memcpy(&o->RadiusMin,        el + REFL_OFF_RADIUS_MIN,  4);
        memcpy(&o->RadiusMax,        el + REFL_OFF_RADIUS_MAX,  4);
        memcpy(&o->BrightnessMin,    el + REFL_OFF_BRIGHT_MIN,  4);
        memcpy(&o->BrightnessMax,    el + REFL_OFF_BRIGHT_MAX,  4);
        memcpy(&o->ModulationFactor, el + REFL_OFF_MODULATION,  4);
        memcpy(&o->ColorR,           el + REFL_OFF_COLOR_RGB,     4);
        memcpy(&o->ColorG,           el + REFL_OFF_COLOR_RGB + 4, 4);
        memcpy(&o->ColorB,           el + REFL_OFF_COLOR_RGB + 8, 4);
        memcpy(&o->TintPower,        el + REFL_OFF_TINT_POWER,  4);
        // Raw curve data sizes (uncertain semantics - LOGGED not interpreted)
        o->RadiusCurveSize = RU32(el + REFL_OFF_RADIUS_CURVE);
        o->BrightCurveSize = RU32(el + REFL_OFF_BRIGHT_CURVE);

        // Sanity scrub - exotic floats (NaN/Inf) zeroed so the consumer can
        // detect "no value" and fall back to a default.
        auto sane = [](float& v, float lo, float hi) {
            if (!(v == v) || v < lo || v > hi) v = 0.0f;
        };
        sane(o->AxisOffset,    -4.0f, 4.0f);
        sane(o->RotationOffset,-100.0f, 100.0f);
        sane(o->RadiusMin,      0.0f, 100.0f);
        sane(o->RadiusMax,      0.0f, 100.0f);
        sane(o->BrightnessMin,  0.0f, 100.0f);
        sane(o->BrightnessMax,  0.0f, 100.0f);
        sane(o->ModulationFactor,0.0f, 16.0f);
        sane(o->ColorR, 0.0f, 16.0f);
        sane(o->ColorG, 0.0f, 16.0f);
        sane(o->ColorB, 0.0f, 16.0f);
        sane(o->TintPower, 0.0f, 64.0f);
    }
    *outCount = (uint32_t)n;

    NativeDiag("LensWalker: lens=%u name-class=lens reflections total=%d written=%d "
               "bitmapTag=%u innerScale=%.3f [0]axis=%.4f rad=%.3f col=(%.2f,%.2f,%.2f)",
               lensTagId, blk.count, n,
               outInfo ? outInfo->BitmapTagId : 0,
               outInfo ? outInfo->OcclusionInnerRadiusScale : 0.0f,
               n > 0 ? outBuf[0].AxisOffset : 0.0f,
               n > 0 ? outBuf[0].RadiusMin  : 0.0f,
               n > 0 ? outBuf[0].ColorR : 0.0f,
               n > 0 ? outBuf[0].ColorG : 0.0f,
               n > 0 ? outBuf[0].ColorB : 0.0f);
    return true;
}

bool SehGetLensElements(CacheHandle* cache, uint32_t lensTagId,
                        ZH_LensFlareElement* outBuf, int maxElems,
                        uint32_t* outCount, uint32_t* outTotalSeen,
                        ZH_LensFlareInfo* outInfo)
{
    __try {
        return GetLensElementsInner(cache, lensTagId, outBuf, maxElems,
                                    outCount, outTotalSeen, outInfo);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outCount) *outCount = 0;
        if (outTotalSeen) *outTotalSeen = 0;
        if (outInfo) memset(outInfo, 0, sizeof(*outInfo));
        return false;
    }
}

} // anonymous namespace

// ============================================================================
// Public exports
// ============================================================================
// Resolve scnr.Skies[i] -> scen -> hlmt -> mode and read the analytical sky-light
// fields (direction + HDR RGB + natural-light SH DC ambient) authored in that
// render_model. Walks skies in order; first with a resolvable mode wins.
bool GetSkyLightInner(CacheHandle* cache, uint32_t scnrTagId, ZH_SkyLight* out)
{
    memset(out, 0, sizeof(*out));
    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scnrTagId];
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return false;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    size_t scnrAvail = (size_t)cache->size - (size_t)metaOff;
    const uint8_t* scnrMeta = cache->base + metaOff;

    int32_t skyCount = 0;
    ResolveSkySceneryTagId(cache, scnrMeta, scnrAvail, 0, &skyCount);
    if (skyCount <= 0) return false;
    if (skyCount > 64) skyCount = 64;

    for (int i = 0; i < skyCount; ++i) {
        int32_t scenId = ResolveSkySceneryTagId(cache, scnrMeta, scnrAvail, i, nullptr);
        if (scenId < 0) continue;
        int32_t hlmtId = FollowTagRef(cache, (uint32_t)scenId, TC_SCEN,
                                      SCENERY_MODEL_REF_OFFSET, TC_HLMT);
        if (hlmtId < 0) continue;
        int32_t modeId = FollowTagRef(cache, (uint32_t)hlmtId, TC_HLMT,
                                      HLMT_RENDER_MODEL_REF_OFF, TC_MODE);
        if (modeId < 0) continue;
        int64_t mOff = TagMetaFileOff(cache, cache->tags[modeId].metaPointerRaw);
        if (mOff < 0) continue;
        if ((size_t)mOff + 0x240 > cache->size) continue;
        const uint8_t* m = cache->base + mOff;
        out->Dir[0] = RfLE(m + 0x224); out->Dir[1] = RfLE(m + 0x228); out->Dir[2] = RfLE(m + 0x22C);
        out->Color[0] = RfLE(m + 0x230); out->Color[1] = RfLE(m + 0x234); out->Color[2] = RfLE(m + 0x238);
        out->Ambient[0] = RfLE(m + 0x124); out->Ambient[1] = RfLE(m + 0x164); out->Ambient[2] = RfLE(m + 0x1A4);
        out->HasLight = 1;
        NativeDiag("SkyLight: scnr=%u sky=%d mode=%d dir=[%.3f,%.3f,%.3f] "
                   "rgb=[%.3f,%.3f,%.3f] amb=[%.3f,%.3f,%.3f]",
                   scnrTagId, i, modeId, out->Dir[0], out->Dir[1], out->Dir[2],
                   out->Color[0], out->Color[1], out->Color[2],
                   out->Ambient[0], out->Ambient[1], out->Ambient[2]);
        return true;
    }
    return false;
}

extern "C" __declspec(dllexport) bool __stdcall ZH_SCNR_GetSkyLight(
    uint64_t cacheHandle, uint32_t scnrTagId, ZH_SkyLight* outLight)
{
    if (!outLight) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) { memset(outLight, 0, sizeof(*outLight)); return false; }
    __try { return GetSkyLightInner(cache, scnrTagId, outLight); }
    __except (EXCEPTION_EXECUTE_HANDLER) { memset(outLight, 0, sizeof(*outLight)); return false; }
}

extern "C" __declspec(dllexport) bool __stdcall ZH_SCNR_GetSunLight(
    uint64_t cacheHandle, uint32_t scnrTagId, ZH_SunLight* outLight)
{
    if (!outLight) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        memset(outLight, 0, sizeof(*outLight));
        return false;
    }
    return SehGetSunLight(cache, scnrTagId, outLight);
}

// Enumerate a `lens` (lens_flare) tag's reflection chain. Caller passes the
// LensFlareTagId captured from ZH_SCNR_GetSunLight (ligh+0xA8) and a pinned
// ZH_LensFlareElement[] of length maxElems. Writes up to maxElems entries,
// reports the count written + total reflections seen + per-lens header info.
// Returns true if the tag resolved as a valid 'lens'; false on bad handle/tag
// or read fault. (count==0 with return true => valid lens with no reflections.)
extern "C" __declspec(dllexport) bool __stdcall ZH_LENS_GetElements(
    uint64_t cacheHandle, uint32_t lensTagId,
    ZH_LensFlareElement* outBuf, int32_t maxElems,
    uint32_t* outCount, uint32_t* outTotalSeen,
    ZH_LensFlareInfo* outInfo)
{
    if (outCount) *outCount = 0;
    if (outTotalSeen) *outTotalSeen = 0;
    if (outInfo) memset(outInfo, 0, sizeof(*outInfo));
    if (!outBuf || maxElems <= 0 || !outCount || !outTotalSeen) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return false;
    return SehGetLensElements(cache, lensTagId, outBuf, maxElems,
                              outCount, outTotalSeen, outInfo);
}
