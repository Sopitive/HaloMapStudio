// OverlayWalker.cpp
// =============================================================================
// Native walker for rmsh.Postprocess[0].Overlays
// animation curves. Walks each overlay entry, decodes its s_animation_function
// payload, and pre-samples 16 evenly-spaced curve points so the viewer's per-frame
// tick can do a single LUT lookup at frac(globalTime / TimePeriod) without
// porting Bungie's full function library.
//
// Layout reference: RMSH_OVERLAYS_RE.md (full RE, including binary scan of
// 500 overlays from 30_settlement.map). Five primary function-type variants
// covered explicitly (0 Identity, 2 Direct Transition, 3 Linear+Periodic,
// 8 Multi-spline-key, 9 Color two-color blend) + a linear-ramp fallback for
// unseen variants. Twelve secondary periodic types covered per the
// "Wobble Function" enum from Plugins/Reach/csdt.xml.
//
// ABI:
//   int32_t ZH_MMP_GetShaderOverlayCount(modelHandle, shaderIndex)
//   bool    ZH_MMP_GetShaderOverlayAt(modelHandle, shaderIndex, overlayIndex, &out)
//
// Used by the viewer's sky material build to drive engine-correct
// cloud scroll / lightning pulse / distant_booms cadence in SkyUnlitMaterial.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"
#include "MapModelParser.h"   // ZH_ModelHandle typedef

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

using namespace zh_mcc;

// Public ABI - must match the Rust mirror in crates/hms-native/src/lib.rs.
#pragma pack(push, 1)
struct ZH_ShaderOverlay {
    int32_t  Type;              // engine enum 0..10
    int32_t  InputNameSid;
    int32_t  RangeNameSid;
    float    TimePeriod;        // seconds per cycle
    char     InputName[32];

    uint8_t  FuncType;
    uint8_t  FuncOutputFlags;
    uint8_t  FuncSubType;
    uint8_t  SecondaryType;     // 0xFF when no secondary

    float    InMin, InMax;
    float    OutMin, OutMax;

    uint32_t ColorMinRgba;
    uint32_t ColorMaxRgba;

    float    CurvePoints[16];

    uint32_t Reserved[2];
};
#pragma pack(pop)

static_assert(sizeof(ZH_ShaderOverlay) == 148,
              "ZH_ShaderOverlay layout drift - keep in sync with the Rust mirror in crates/hms-native/src/lib.rs");

namespace {

constexpr int OFF_SHADER_PROPS_OW    = 0x38;   // rmsh+0x38 -> Postprocess block
constexpr int SHADER_PROPS_SIZE_OW   = 0xAC;
constexpr int OFF_OVERLAYS_IN_PROPS_OW = 0x5C; // pp+0x5C -> Overlays tagblock
constexpr int OVERLAY_BLOCK_SIZE_OW  = 0x24;
constexpr int OV_OFF_TYPE        = 0x00;
constexpr int OV_OFF_INPUT_SID   = 0x04;
constexpr int OV_OFF_RANGE_SID   = 0x08;
constexpr int OV_OFF_PERIOD      = 0x0C;
constexpr int OV_OFF_FUNCREF     = 0x10;   // dataref(16): size@+0, ptr@+0xC
constexpr int OV_DATAREF_PTR_OFF = 0x0C;   // raw ptr inside the dataref

// Helpers exposed from MapModelParser.cpp so this file doesn't have to touch
// internal model-data structs. ZH_MMP_GetShaderTagId is a public export;
// MapModelParser_GetCacheForModel is internal extern "C" linkage.
extern "C" __declspec(dllexport) int32_t __stdcall
ZH_MMP_GetShaderTagId(ZH_ModelHandle h, int32_t shaderIndex);

extern "C" CacheHandle* MapModelParser_GetCacheForModel(ZH_ModelHandle h);

// ----------- Primary stage decode ------------------------------------------

void DecodePrimaryStage(const uint8_t* payload, int32_t /*size*/,
                        ZH_ShaderOverlay* o)
{
    // bytes 4..19 are the primary stage (16 bytes). Interpretation depends on
    // function_type byte at payload[0]. See RMSH_OVERLAYS_RE.md section 3.
    uint8_t ft = o->FuncType;
    if (ft == 9 || ft == 2 || ft == 3) {
        // Color/Transition variants: rgba_min @ +4, rgba_max @ +16.
        // For Type 3 (Linear+Periodic) the four floats actually mean
        // (in_min, in_max, out_min, out_max) - but in the observed
        // payload most non-trivial Periodic curves use type=3 with
        // stage_a as scalar (in_min/in_max/out_min/out_max). Read both
        // and prefer the scalar interpretation when the bytes pattern
        // is a plausible float (avoids garbage rgba misinterpretation
        // when the underlying author chose floats).
        memcpy(&o->InMin,  payload + 4,  4);
        memcpy(&o->InMax,  payload + 8,  4);
        memcpy(&o->OutMin, payload + 12, 4);
        memcpy(&o->OutMax, payload + 16, 4);
        memcpy(&o->ColorMinRgba, payload + 4,  4);
        memcpy(&o->ColorMaxRgba, payload + 16, 4);
    } else {
        // Type 0 / 8 (and unknown) -> scalar in_min/in_max/out_min/out_max
        memcpy(&o->InMin,  payload + 4,  4);
        memcpy(&o->InMax,  payload + 8,  4);
        memcpy(&o->OutMin, payload + 12, 4);
        memcpy(&o->OutMax, payload + 16, 4);
        o->ColorMinRgba = 0;
        o->ColorMaxRgba = 0;
    }
    // Sanity clamp - values outside engine-sensible ranges -> defaults.
    auto isSane = [](float v){ return v == v && fabsf(v) < 1.0e6f; };
    if (!isSane(o->InMin))  o->InMin  = 0.0f;
    if (!isSane(o->InMax))  o->InMax  = 1.0f;
    if (!isSane(o->OutMin)) o->OutMin = 0.0f;
    if (!isSane(o->OutMax)) o->OutMax = 0.0f;
}

// ----------- Curve evaluators ----------------------------------------------

static float EvalPrimary(uint8_t ft, float in_min, float in_max,
                         float out_min, float out_max, float t)
{
    (void)in_min; (void)in_max; // not used in current evaluators
    switch (ft) {
        case 0: return out_min;                                   // Identity / no-op
        case 1: return out_min;                                   // Constant
        case 2: return out_min + (out_max - out_min) * t;         // Direct transition
        case 3: return out_min + (out_max - out_min) * t;         // Linear+Periodic
        case 4: return out_min + (out_max - out_min) * t;         // Linear
        case 8: return out_min + (out_max - out_min) * t;         // Multi-spline (fallback)
        case 9: return 0.5f * (out_min + out_max);                // Color mid
        default: return out_min;
    }
}

static float EvalSecondary(uint8_t st, float t)
{
    const float TWO_PI = 6.28318530718f;
    switch (st) {
        case 0:  return 1.0f;
        case 1:  return 0.0f;
        case 2:  /* Cosine */
        case 3:  return 0.5f - 0.5f * cosf(TWO_PI * t);
        case 4:  /* Diagonal Wave / Triangle */
        case 5:  { float v = 2.0f * t - 1.0f; return fabsf(v); }
        case 6:  /* Slide / Sawtooth */
        case 7:  return t;
        case 8: { /* Noise */
            uint32_t h = (uint32_t)(t * 1024.0f);
            h ^= h * 0x85ebca6bu; h ^= h >> 13;
            return (float)(h & 0xFFFFu) / 65535.0f;
        }
        case 9: { /* Jitter - low-pass over 4 hash samples */
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                uint32_t h = (uint32_t)((t + i * 0.05f) * 1024.0f);
                h ^= h * 0x85ebca6bu; h ^= h >> 13;
                acc += (float)(h & 0xFFFFu) / 65535.0f;
            }
            return acc * 0.25f;
        }
        case 10: { /* Wander - smoothstepped random walk */
            float a = floorf(t * 8.0f), b = a + 1.0f;
            uint32_t ha = (uint32_t)(a * 17.0f) * 0x85ebca6bu;
            uint32_t hb = (uint32_t)(b * 17.0f) * 0x85ebca6bu;
            float u = t * 8.0f - a;
            u = u * u * (3.0f - 2.0f * u);
            float va = (float)(ha & 0xFFFFu) / 65535.0f;
            float vb = (float)(hb & 0xFFFFu) / 65535.0f;
            return va * (1.0f - u) + vb * u;
        }
        case 11: { /* Spark - sparse impulses */
            float fr = t * 8.0f - floorf(t * 8.0f);
            return (fr > 0.85f) ? 1.0f : 0.0f;
        }
        default: return 1.0f;
    }
}

void BakeCurveLut(ZH_ShaderOverlay* o)
{
    for (int i = 0; i < 16; ++i) {
        float t = (float)i / 15.0f;
        float a = EvalPrimary(o->FuncType, o->InMin, o->InMax,
                              o->OutMin, o->OutMax, t);
        float b = (o->SecondaryType != 0xFF)
                ? EvalSecondary(o->SecondaryType, t)
                : 1.0f;
        o->CurvePoints[i] = a * b;
    }
}

// ----------- Walker ---------------------------------------------------------

bool ReadOverlayCommon(CacheHandle* cache, const uint8_t* entry,
                       ZH_ShaderOverlay* outOv)
{
    memset(outOv, 0, sizeof(*outOv));
    outOv->Type         = (int32_t)R32(entry + OV_OFF_TYPE);
    outOv->InputNameSid = R32(entry + OV_OFF_INPUT_SID);
    outOv->RangeNameSid = R32(entry + OV_OFF_RANGE_SID);
    memcpy(&outOv->TimePeriod, entry + OV_OFF_PERIOD, 4);
    outOv->SecondaryType = 0xFF;

    if (cache->stringTableParsed) {
        const char* nm = ResolveStringId(cache, outOv->InputNameSid);
        if (nm) {
            size_t L = strnlen(nm, 31);
            memcpy(outOv->InputName, nm, L);
            outOv->InputName[L] = 0;
        }
    }

    // dataRef at +0x10. Size@+0, raw ptr@+0xC. Resolve via TagMetaFileOff.
    int32_t fnSize = R32(entry + OV_OFF_FUNCREF + 0x00);
    uint32_t rawPtr = (uint32_t)R32(entry + OV_OFF_FUNCREF + OV_DATAREF_PTR_OFF);
    if (fnSize < 28 || fnSize > 1024) return true;   // sanity; keep defaults
    int64_t fnOff = TagMetaFileOff(cache, rawPtr);
    if (fnOff < 0 || (size_t)fnOff + (size_t)fnSize > cache->size) return true;
    const uint8_t* payload = cache->base + fnOff;

    // Common 28-byte header.
    outOv->FuncType        = payload[0];
    outOv->FuncOutputFlags = payload[1];
    outOv->FuncSubType     = payload[2];
    DecodePrimaryStage(payload, fnSize, outOv);

    if (fnSize >= 32) {
        uint32_t secSize = R32(payload + 28);
        if (secSize > 0 && (size_t)32 + (size_t)secSize <= (size_t)fnSize) {
            uint8_t st = payload[32];
            if (st < 12) outOv->SecondaryType = st;
        }
    }
    return true;
}

bool ReadShaderOverlayInner(CacheHandle* cache, int32_t shaderTagId,
                            int32_t overlayIndex, ZH_ShaderOverlay* outOv)
{
    if (!outOv) return false;
    memset(outOv, 0, sizeof(*outOv));
    outOv->SecondaryType = 0xFF;
    if (!cache) return false;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return false;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + OFF_SHADER_PROPS_OW + 8 > cache->size) return false;
    const uint8_t* meta = cache->base + metaOff;

    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS_OW);
    if (propsBlk.count <= 0) return false;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 || (size_t)propsOff + SHADER_PROPS_SIZE_OW > cache->size) return false;
    const uint8_t* props = cache->base + propsOff;

    TagBlockRef ovBlk = ReadTagBlock(props + OFF_OVERLAYS_IN_PROPS_OW);
    if (ovBlk.count <= 0 || ovBlk.count > 256) return false;
    if (overlayIndex < 0 || overlayIndex >= ovBlk.count) return false;
    int64_t ovOff = TagMetaFileOff(cache, ovBlk.pointer);
    if (ovOff < 0) return false;
    if ((size_t)ovOff + (size_t)ovBlk.count * OVERLAY_BLOCK_SIZE_OW > cache->size) return false;

    const uint8_t* entry = cache->base + ovOff
                         + (size_t)overlayIndex * OVERLAY_BLOCK_SIZE_OW;
    ReadOverlayCommon(cache, entry, outOv);
    BakeCurveLut(outOv);
    // FF-ROUTING: expose the postprocess "Routing Info" (pp+0x50, stride 4) so the caller can
    // map this overlay to the rmt2 argument it drives. Reserved[0] = raw routing entry
    // [overlayIndex] (0xFFFFFFFF when out of range), Reserved[1] = routingCount | overlayCount<<16.
    {
        // Routing entry = { u8 register, u8 flags (0x8f seen), u8 function(overlay) index, u8 argument index }.
        // Verified on forge_halo shield doors: every animated function routes to a texture-xform
        // argument (noise_map_a/b, warp_map, overlay_map/_detail_map) matching the rmt2 arg order.
        TagBlockRef rtBlk = ReadTagBlock(props + 0x50);
        uint32_t argIdx = 0xFFFFFFFFu;
        if (rtBlk.count > 0 && rtBlk.count < 4096) {
            int64_t rtOff = TagMetaFileOff(cache, rtBlk.pointer);
            if (rtOff >= 0 && (size_t)rtOff + (size_t)rtBlk.count * 4 <= cache->size) {
                for (int32_t r = 0; r < rtBlk.count; ++r) {
                    const uint8_t* re = cache->base + rtOff + (size_t)r * 4;
                    if ((int32_t)re[2] == overlayIndex) { argIdx = re[3]; break; }
                }
            }
        }
        outOv->Reserved[0] = argIdx;
        outOv->Reserved[1] = ((uint32_t)rtBlk.count & 0xFFFFu) | ((uint32_t)ovBlk.count << 16);
    }
    return true;
}

int32_t CountShaderOverlaysInner(CacheHandle* cache, int32_t shaderTagId)
{
    if (!cache) return 0;
    if (shaderTagId < 0 || (uint32_t)shaderTagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[shaderTagId];
    if (te.classIndex < 0) return 0;
    if (te.classCode[0] != 'r' || te.classCode[1] != 'm') return 0;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return 0;
    if ((size_t)metaOff + OFF_SHADER_PROPS_OW + 8 > cache->size) return 0;
    const uint8_t* meta = cache->base + metaOff;
    TagBlockRef propsBlk = ReadTagBlock(meta + OFF_SHADER_PROPS_OW);
    if (propsBlk.count <= 0) return 0;
    int64_t propsOff = TagMetaFileOff(cache, propsBlk.pointer);
    if (propsOff < 0 || (size_t)propsOff + SHADER_PROPS_SIZE_OW > cache->size) return 0;
    const uint8_t* props = cache->base + propsOff;
    TagBlockRef ovBlk = ReadTagBlock(props + OFF_OVERLAYS_IN_PROPS_OW);
    if (ovBlk.count <= 0 || ovBlk.count > 256) return 0;
    return (int32_t)ovBlk.count;
}

int32_t SehCountShaderOverlays(CacheHandle* cache, int32_t shaderTagId)
{
    __try { return CountShaderOverlaysInner(cache, shaderTagId); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return 0; }
}

bool SehReadShaderOverlay(CacheHandle* cache, int32_t shaderTagId,
                          int32_t overlayIndex, ZH_ShaderOverlay* outOv)
{
    __try { return ReadShaderOverlayInner(cache, shaderTagId, overlayIndex, outOv); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outOv) {
            memset(outOv, 0, sizeof(*outOv));
            outOv->SecondaryType = 0xFF;
        }
        return false;
    }
}

} // anonymous namespace

// ============================================================================
// Public exports
// ============================================================================

// By-TAG variants (any render_method tag id, e.g. a BSP material's shader): same records/routing.
extern "C" __declspec(dllexport) int32_t __stdcall ZH_TAG_GetShaderOverlayCount(
    uint64_t cacheHandle, int32_t shaderTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || shaderTagId < 0) return 0;
    return SehCountShaderOverlays(cache, shaderTagId);
}
extern "C" __declspec(dllexport) bool __stdcall ZH_TAG_GetShaderOverlayAt(
    uint64_t cacheHandle, int32_t shaderTagId, int32_t overlayIndex, ZH_ShaderOverlay* outOv)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache || shaderTagId < 0 || !outOv) return false;
    return SehReadShaderOverlay(cache, shaderTagId, overlayIndex, outOv);
}

extern "C" __declspec(dllexport) int32_t __stdcall ZH_MMP_GetShaderOverlayCount(
    ZH_ModelHandle h, int32_t shaderIndex)
{
    int32_t shaderTagId = ZH_MMP_GetShaderTagId(h, shaderIndex);
    if (shaderTagId < 0) return 0;
    // Cache handle is needed to walk the metadata; reuse the one stored on
    // the model. The shaderTagId is in the cache the model was opened from.
    CacheHandle* cache = MapModelParser_GetCacheForModel(h);
    if (!cache) return 0;
    return SehCountShaderOverlays(cache, shaderTagId);
}

extern "C" __declspec(dllexport) bool __stdcall ZH_MMP_GetShaderOverlayAt(
    ZH_ModelHandle h, int32_t shaderIndex, int32_t overlayIndex,
    ZH_ShaderOverlay* outOv)
{
    if (!outOv) return false;
    int32_t shaderTagId = ZH_MMP_GetShaderTagId(h, shaderIndex);
    if (shaderTagId < 0) {
        memset(outOv, 0, sizeof(*outOv));
        outOv->SecondaryType = 0xFF;
        return false;
    }
    CacheHandle* cache = MapModelParser_GetCacheForModel(h);
    if (!cache) {
        memset(outOv, 0, sizeof(*outOv));
        outOv->SecondaryType = 0xFF;
        return false;
    }
    return SehReadShaderOverlay(cache, shaderTagId, overlayIndex, outOv);
}
