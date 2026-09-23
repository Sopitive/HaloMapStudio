// ScenarioObjectWalker.cpp
// =============================================================================
// Native walker for a scenario's BAKED object placements (scenery / vehicles /
// weapons / equipment / machines / controls / crates / giants / effect scenery /
// bipeds). Lets the viewer render a map's designer-placed objects with NO live
// game. Mirrors ScenarioForgePaletteWalker.cpp.
//
// NOTE: user-placed FORGE objects are NOT here - those live in the .mvar variant
// or in live process memory. This walker returns only the cooked scenario's
// authored objects.
//
// scnr layout (MCC Halo Reach). Each object category is a pair of 12-byte
// BlockCollection descriptors laid out contiguously: placement block, then its
// palette block +0x0C after. Offsets are LEGACY (MccHaloReach/U3/U8/U10);
// U13 subtracts 0x14 from every object placement/palette offset.
//
//   placement element - shared s_scenario_object_datum header (constant across
//   categories; only trailing per-category data differs, hence differing strides):
//     +0x00  int16   palette_index   (-1 => skip)
//     +0x02  int16   name_index
//     +0x04  uint32  placement_flags
//     +0x08  float3  position (world XYZ)
//     +0x14  float3  rotation (real_euler_angles_3d, RADIANS: yaw(+Z),pitch(+Y),roll(+X))
//     +0x20  float   scale (<=0 => 1.0 default)
//
//   palette entry (stride 0x10): tag_reference @ +0x00; tagId @ +0x0C.
//     obj short id (& 0xFFFF) -> ZH_TAG_ResolveModeTagId -> render_model (mode).
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

using namespace zh_mcc;

// Resolve obj (scen/vehi/weap/...) tag -> render_model (mode) tag id. Exported
// from MapBitmapParser.cpp; forward-declared here (same __stdcall ABI).
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolveModeTagId(
    uint64_t cacheHandle, uint32_t primaryTagId);
// Resolve an effect_scenery (efsc) tag -> representative bitmap for a billboard.
extern "C" __declspec(dllexport) uint32_t __stdcall ZH_TAG_ResolveEfscBitmap(
    uint64_t cacheHandle, uint32_t efscTagId);

namespace {

#pragma pack(push, 1)
struct ZH_ScnrObjectInstance {
    uint32_t renderModelTagId;  // 0  resolved mode id (0xFFFFFFFF if none)
    uint32_t objTagShortId;     // 4  raw obj tag (scen/vehi/...) short id
    uint16_t categoryId;        // 8  which block (see kCategories)
    int16_t  paletteIndex;      // 10
    int16_t  nameIndex;         // 12
    uint16_t pad;               // 14
    uint32_t placementFlags;    // 16
    float    pos[3];            // 20 world XYZ
    float    fwd[3];            // 32 forward basis (from euler)
    float    up[3];             // 44 up basis (from euler)
    float    euler[3];          // 56 raw yaw,pitch,roll radians
    float    scale;             // 68 normalized (<=0 -> 1.0)
};
#pragma pack(pop)
static_assert(sizeof(ZH_ScnrObjectInstance) == 72, "ZH_ScnrObjectInstance layout drift");

constexpr const char* TC_SCNR = "scnr";

constexpr int PLACEMENT_HDR_PAL_IDX  = 0x00;
constexpr int PLACEMENT_HDR_NAME_IDX = 0x02;
constexpr int PLACEMENT_HDR_FLAGS    = 0x04;
constexpr int PLACEMENT_HDR_POS      = 0x08;
constexpr int PLACEMENT_HDR_ROT      = 0x14;
constexpr int PLACEMENT_HDR_SCALE    = 0x20;
constexpr int PALETTE_STRIDE         = 0x10;
constexpr int PALETTE_TAGID_OFF      = 0x0C;

constexpr int PLACEMENT_COUNT_SANITY = 0x40000;
constexpr int PALETTE_COUNT_SANITY   = 4096;
constexpr int TOTAL_INSTANCE_CAP     = 20000;

struct Category {
    uint16_t id;
    int      legacyPlacementOff;
    int      legacyPaletteOff;
    int      stride;
};

// Renderable categories only (each resolves obj -> mode). Skip sound_scenery,
// light_volumes, terminals (no render_model).
constexpr Category kCategories[] = {
    { 0,  0x110, 0x11C, 0xDC },  // scenery
    { 1,  0x128, 0x134, 0x78 },  // bipeds
    { 2,  0x140, 0x14C, 0xD0 },  // vehicles
    { 3,  0x158, 0x164, 0xB4 },  // equipment
    { 4,  0x170, 0x17C, 0xD0 },  // weapons
    { 5,  0x194, 0x1A0, 0xE4 },  // machines
    { 6,  0x1C4, 0x1D0, 0xDC },  // controls
    { 7,  0x1F4, 0x200, 0x88 },  // giants
    { 8,  0x20C, 0x218, 0xB0 },  // effect_scenery
    { 9,  0x614, 0x620, 0xD8 },  // crates
};

// U13 subtracts 0x14 from object placement/palette offsets (proven for the
// object region only).
int U13Delta(CacheType ct) { return (ct == CacheType::MccHaloReachU13) ? 0x14 : 0; }

// Euler (yaw about +Z, pitch about +Y, roll about +X, radians) -> fwd/up basis,
// matching the live-object render path (object_matrix re-orthonormalizes).
void EulerToBasis(const float e[3], float fwd[3], float up[3]) {
    float sy = sinf(e[0]), cy = cosf(e[0]);
    float sp = sinf(e[1]), cp = cosf(e[1]);
    float sr = sinf(e[2]), cr = cosf(e[2]);
    fwd[0] = cy * cp;  fwd[1] = sy * cp;  fwd[2] = -sp;
    up[0]  = cy * sp * cr + sy * sr;
    up[1]  = sy * sp * cr - cy * sr;
    up[2]  = cp * cr;
}

// Walk one category's placements into the growable buffer.
void WalkCategory(CacheHandle* cache, uint64_t cacheHandle, const uint8_t* scnr,
                  const Category& cat, int delta,
                  ZH_ScnrObjectInstance** outBuf, size_t* outCap, size_t* outLen)
{
    int placementOff = cat.legacyPlacementOff - delta;
    int paletteOff   = cat.legacyPaletteOff   - delta;
    if (placementOff < 0 || paletteOff < 0) return;
    if ((size_t)placementOff + 12 > cache->size || (size_t)paletteOff + 12 > cache->size) return;

    TagBlockRef placeBlk = ReadTagBlock(scnr + placementOff);
    if (placeBlk.count <= 0 || placeBlk.count > PLACEMENT_COUNT_SANITY) return;
    int64_t placeArrOff = TagMetaFileOff(cache, placeBlk.pointer);
    if (placeArrOff < 0) return;
    if ((size_t)placeArrOff + (size_t)placeBlk.count * cat.stride > cache->size) return;

    // Palette (for the obj tag refs). May legitimately be empty.
    TagBlockRef palBlk = ReadTagBlock(scnr + paletteOff);
    const uint8_t* palArr = nullptr;
    int palCount = 0;
    if (palBlk.count > 0 && palBlk.count <= PALETTE_COUNT_SANITY) {
        int64_t palArrOff = TagMetaFileOff(cache, palBlk.pointer);
        if (palArrOff >= 0 &&
            (size_t)palArrOff + (size_t)palBlk.count * PALETTE_STRIDE <= cache->size) {
            palArr = cache->base + palArrOff;
            palCount = palBlk.count;
        }
    }
    if (!palArr || palCount == 0) return;  // no palette -> no renderable objects

    const uint8_t* placeArr = cache->base + placeArrOff;

    for (int i = 0; i < placeBlk.count; ++i) {
        if (*outLen >= (size_t)TOTAL_INSTANCE_CAP) return;
        const uint8_t* pl = placeArr + (size_t)i * cat.stride;

        int16_t palIdx = (int16_t)RU16(pl + PLACEMENT_HDR_PAL_IDX);
        if (palIdx < 0 || palIdx >= palCount) continue;

        // Resolve palette obj -> mode.
        const uint8_t* pe = palArr + (size_t)palIdx * PALETTE_STRIDE;
        uint32_t tagDatum = RU32(pe + PALETTE_TAGID_OFF);
        if (tagDatum == 0u || tagDatum == 0xFFFFFFFFu) continue;
        uint32_t objShort = tagDatum & 0xFFFFu;
        if (objShort >= cache->tags.size()) continue;
        uint32_t modeId;
        if (cat.id == 8) {
            // Effect scenery (efsc) has NO render_model - resolve a representative
            // bitmap (efsc->effe->prt3->bitm) so the viewer can draw a billboard
            // stand-in. Emit even when the bitmap can't be resolved
            // (renderModelTagId=0xFFFFFFFF) so the effect's POSITION still shows as a
            // generic sprite; the Rust side routes category 8 to the billboard path.
            modeId = ZH_TAG_ResolveEfscBitmap(cacheHandle, objShort);
        } else {
            modeId = ZH_TAG_ResolveModeTagId(cacheHandle, objShort);
            if (modeId == 0xFFFFFFFFu) continue;  // no render_model (blocker/light/etc.)
        }

        // Grow buffer geometrically.
        if (*outLen >= *outCap) {
            size_t newCap = (*outCap == 0) ? 512 : (*outCap * 2);
            if (newCap > (size_t)TOTAL_INSTANCE_CAP) newCap = TOTAL_INSTANCE_CAP;
            ZH_ScnrObjectInstance* nb = (ZH_ScnrObjectInstance*)realloc(
                *outBuf, newCap * sizeof(ZH_ScnrObjectInstance));
            if (!nb) return;
            *outBuf = nb;
            *outCap = newCap;
        }

        ZH_ScnrObjectInstance& D = (*outBuf)[*outLen];
        memset(&D, 0, sizeof(D));
        D.renderModelTagId = modeId;
        D.objTagShortId    = objShort;
        D.categoryId       = cat.id;
        D.paletteIndex     = palIdx;
        D.nameIndex        = (int16_t)RU16(pl + PLACEMENT_HDR_NAME_IDX);
        D.placementFlags   = RU32(pl + PLACEMENT_HDR_FLAGS);
        memcpy(D.pos,   pl + PLACEMENT_HDR_POS, 12);
        memcpy(D.euler, pl + PLACEMENT_HDR_ROT, 12);
        float scale; memcpy(&scale, pl + PLACEMENT_HDR_SCALE, 4);
        if (!(scale > 0.0f) || !isfinite(scale)) scale = 1.0f;
        D.scale = scale;
        EulerToBasis(D.euler, D.fwd, D.up);
        ++(*outLen);
    }
}

bool EnumerateInner(CacheHandle* cache, uint64_t cacheHandle, uint32_t scnrTagId,
                    ZH_ScnrObjectInstance** outBuf, size_t* outCap, size_t* outLen)
{
    *outBuf = nullptr; *outCap = 0; *outLen = 0;
    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scnrTagId];
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return false;
    int64_t scnrMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (scnrMetaOff < 0) return false;
    const uint8_t* scnr = cache->base + scnrMetaOff;

    int delta = U13Delta(cache->cacheType);
    for (const Category& cat : kCategories) {
        WalkCategory(cache, cacheHandle, scnr, cat, delta, outBuf, outCap, outLen);
    }
    NativeDiag("ScnrObjects[scnr=0x%X] cacheType=%d emitted=%zu",
               scnrTagId, (int)cache->cacheType, *outLen);
    return true;
}

bool EnumerateSeh(CacheHandle* cache, uint64_t cacheHandle, uint32_t scnrTagId,
                  ZH_ScnrObjectInstance** outBuf, size_t* outCap, size_t* outLen)
{
    __try { return EnumerateInner(cache, cacheHandle, scnrTagId, outBuf, outCap, outLen); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        NativeDiag("ScnrObjects: scnr=0x%X SEH fault", scnrTagId);
        if (*outBuf) { free(*outBuf); *outBuf = nullptr; }
        *outCap = 0; *outLen = 0;
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================
extern "C" __declspec(dllexport) int __stdcall ZH_SCNR_EnumerateObjects(
    uint64_t cacheHandle,
    uint32_t scnrTagId,
    ZH_ScnrObjectInstance** outBuffer,
    uint32_t* outCount)
{
    if (outBuffer) *outBuffer = nullptr;
    if (outCount)  *outCount  = 0;
    if (!outBuffer || !outCount) return 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || cache->base == nullptr) return 0;

    ZH_ScnrObjectInstance* buf = nullptr;
    size_t cap = 0, len = 0;
    bool ok = EnumerateSeh(cache, cacheHandle, scnrTagId, &buf, &cap, &len);
    if (!ok) { if (buf) free(buf); return 0; }

    *outBuffer = buf;
    *outCount  = (uint32_t)len;
    return 1;
}

extern "C" __declspec(dllexport) void __stdcall ZH_SCNR_FreeObjectBuffer(
    ZH_ScnrObjectInstance* buf)
{
    if (buf) free(buf);
}
