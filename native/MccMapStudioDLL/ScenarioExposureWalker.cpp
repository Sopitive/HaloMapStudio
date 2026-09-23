// ScenarioExposureWalker.cpp
// =============================================================================
// Native walker for the per-scenario global brightness/exposure fields. Reads
// scnr+0x330 (camera_exposure_stops) and optionally chases the
// scripted_exposure_globals tag at scnr+0x6B4 to surface the global
// log2-space exposure offset + sky/ambient/sun RGB tint.
//
// Why: maps in the viewer render too-dark / too-bright inconsistently - see
// GLOBAL_BRIGHTNESS_RE.md. The engine's per-frame g_exposure scalar is
//
//   g_exposure = powf(2, scnr[+0x330] - sum(sceg fields))
//
// where the "sum" is the accumulator over the per-domain offsets at sceg
// +0x20/+0x38/+0x48/+0x88/+0x98/+0xa8 (master / sky / ambient / detail /
// tonemap_min / tonemap_max). For an offline viewer most of those reduce to
// a single net log2-stops scalar that's a static property of the scenario.
//
// Field offsets - verified vs haloreachnew!FUN_18025e3ec (the per-frame
// exposure-accumulator). The .map serialized scnr tag uses the same field
// layout as the runtime struct (tag-schema parity). U13 increments most
// scnr offsets by +4, so we fall back from +0x334 -> +0x330 when the U13
// build is detected and the primary read looks bogus.
//
// The scripted_exposure_globals tag's tag-class fourcc is engine-internal
// and never appears as a string in haloreachnew (the schema-name strings
// are stripped from shipping). We rely on the TagRef-by-id walk:
// scnr+0x6B4 stores a 16-byte TagReference whose tag-id (+12) points to
// the sceg-equivalent tag - we deref it without enforcing a class-code
// match. If the offset isn't authored (TagRef == FFFFFFFF) we just report
// HasSceg=0 and only return the scnr stops.
//
// Failure mode:
//   * Older DLL (export missing): caller sees HasExposure=0, falls back
//     to the legacy K/Brightness heuristic.
//   * scnr+0x330 reads NaN/Inf or |stops| > 6: clamp to 0 and warn - 
//     prevents catastrophic over/under exposure from a layout mismatch.
//   * sceg deref AVs (bad TagRef): SEH returns HasSceg=0; the stops field
//     from scnr is still surfaced.
//
// COLOR_MATRIX_RE - follow-up map for the final-composite
// color_matrix (NOT wired here; documented so the next pass needn't re-RE):
//   The HREK final_composite_base.hlsl_include applies
//     color = saturate(mul(float4(color,1), color_matrix))   // float4x3
//   where color_matrix is FinalCompositePS PS reg 129 (3 regs) - see
//   final_composite_registers.hlsl_include:38. It is ENGINE-RUNTIME-COMPUTED,
//   NOT a serialized scenario field:
//     - camera_fx_settings (cfxs, class 'sxfc' - scnr default_camera_fx):
//       exposure + bloom + bling + self_illum ONLY. No grading block in any of
//       the 46 Reach cfxs tags.
//     - area_screen_effect (asse): the only tag with grading fields
//       (single_screen_effect block, field order: exposure boost, hue left,
//       hue right, saturation, desaturation, contrast enhance, gamma enhance,
//       gamma reduce, bright noise, dark noise, "color filter" rgb [white
//       balance], color floor, ...). Each is an ANIMATABLE mapping_function
//       (transient triggered effect: vision mode / damage / area volume), not a
//       static scalar. forge_halo / 70_boneyard / m45 author them all
//       default-identity (block bodies are 0xCD-uninitialized).
//   => For an offline viewer with no active screen effect, the engine-correct
//      color_matrix is IDENTITY. Per-map "mood" comes from lighting/sky/sun-tint
//      (already surfaced as Sky/Ambient/SunTintRGB below), not a post matrix.
//   If a future pass wants per-effect grading it must (a) resolve the scnr
//      area_screen_effect references, (b) evaluate each single_screen_effect
//      mapping_function at a chosen sim time, (c) compose the 4x3 the way the
//      engine's c_postprocess does. No static-field shortcut exists.
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

using namespace zh_mcc;

// Public ABI - must match the Rust mirror in crates/hms-native/src/lib.rs's
// ZH_ScenarioExposure struct exactly. Pack=1 to avoid implicit padding
// between the float fields and the trailing uint flags.
#pragma pack(push, 1)
struct ZH_ScenarioExposure {
    float    Stops;             // scnr[+0x330] base camera_exposure_stops
                                // (log2 space, +-1 typical)
    float    ScriptedOffset;    // sceg[+0x20].x - master exposure offset
                                // (additive in log2 space). 0 if no sceg.
    float    SkyTintRGB[3];     // sceg[+0x58] RGB sky tint (1,1,1) if no sceg
    float    AmbientTintRGB[3]; // sceg[+0x68] RGB ambient tint
    float    SunTintRGB[3];     // sceg[+0x78] RGB sun tint
    uint32_t HasSceg;           // 0 = scnr+0x6B4 was null / unresolvable
    uint32_t HasExposure;       // 0 = walker failed entirely (scnr stops
                                // unread). When 0, ALL fields are zero-init
                                // and caller should fall back to legacy
                                // brightness derivation.
    // #189: auto-exposure adaptation from the camera_fx_settings (cfxs) tag referenced by
    // scnr Default Camera FX @ scnr+0x6A8. Log2/stops space (composes with Stops above).
    uint32_t HasCameraFx;       // 1 = cfxs resolved + read
    uint32_t AutoEnabled;       // cfxs Flags@0x00 bit2 (0x04 Auto-Adjust Target)
    float    ManualExposure;    // cfxs Exposure@0x04 (used when AutoEnabled==0)
    float    AutoMinEV;         // cfxs Range@0x10 (auto-exposure clamp min)
    float    AutoMaxEV;         // cfxs Range@0x14 (auto-exposure clamp max)
    float    AutoTarget;        // cfxs Auto-Exposure Screen Brightness@0x18 (target key)
    // The engine authors BLOOM PER MAP in this same cfxs tag. HMS had these as
    // hard-coded shader constants (point 1.0 / inherent 0.05) and applied bloom with NO intensity
    // term. Zealot authors 0.15 / 0.25 / 0.4, so our bloom ran 2.5x hot.
    // cfxs parameter structs are { u32 flags, float value, float max_change, float blend_speed }
    // after the EXPOSURE block (0x00..0x1F) and the 8-byte SENSITIVITY struct (0x20..0x27):
    //   0x28 bloom HIGHLIGHT (value @0x2C = "highlight bloom{bloom point}")
    //   0x38 bloom INHERENT  (value @0x3C)
    //   0x48 bloom INTENSITY (value @0x4C)
    float    BloomPoint;
    float    BloomInherent;
    float    BloomIntensity;
    uint32_t HasBloom;
};
#pragma pack(pop)

namespace {

constexpr const char* TC_SCNR = "scnr";

// scnr layout offsets. Verified for U10 / Retail; U13 ships the same scnr
// with most field offsets shifted by +4.
constexpr int SCNR_EXPOSURE_STOPS_OFF_U10 = 0x330;
constexpr int SCNR_EXPOSURE_STOPS_OFF_U13 = 0x334;
constexpr int SCNR_SCEG_TAGREF_OFF_U10    = 0x6B4;
constexpr int SCNR_SCEG_TAGREF_OFF_U13    = 0x6B8;
// #189: scnr Default Camera FX TagReference (-> cfxs camera_fx_settings). Per Assembly
// ReachMCC scnr.xml the tagRef is @ scnr+0x6A8 (ClassId@+0, TagId@+0xC -> 0x6B4). U13 shifts +4.
constexpr int SCNR_CAMERAFX_TAGREF_OFF_U10 = 0x6A8;
constexpr int SCNR_CAMERAFX_TAGREF_OFF_U13 = 0x6AC;
// cfxs EXPOSURE block field offsets (Assembly Reach cfxs.xml, tag-relative).
constexpr int CFXS_FLAGS_OFF          = 0x00; // flags16; bit2 (0x04) = Auto-Adjust Target (AE on)
constexpr int CFXS_EXPOSURE_OFF       = 0x04; // manual exposure (used when AE off)
constexpr int CFXS_RANGE_MIN_OFF      = 0x10; // auto-exposure clamp min (EV/log2)
constexpr int CFXS_RANGE_MAX_OFF      = 0x14; // auto-exposure clamp max
constexpr int CFXS_SCREEN_BRIGHT_OFF  = 0x18; // Auto-Exposure Screen Brightness (target key)
constexpr int CFXS_REQUIRED_BYTES     = 0x1C;

// sceg-equivalent layout offsets (per GLOBAL_BRIGHTNESS_RE.md section 3 +
// haloreachnew!FUN_18025e3ec accumulator). The engine sums SIX log2-space
// offsets to compute `accum`; g_exposure = powf(2, scnr_stops - accum).
// Reading only +0x20 gave us a partial picture and underrepresented
// per-map exposure on scenarios that author the per-domain offsets
// (sky/ambient/detail) but leave the master at 0.
constexpr int SCEG_MASTER_OFFSET_OFF  = 0x20;  // master exposure stops (.x)
constexpr int SCEG_SKY_OFFSET_OFF     = 0x38;  // sky-domain stops (.x)
constexpr int SCEG_AMBIENT_OFFSET_OFF = 0x48;  // ambient-domain stops (.x)
constexpr int SCEG_DETAIL_OFFSET_OFF  = 0x88;  // detail-domain stops (.x)
constexpr int SCEG_TONEMAP_MIN_OFF    = 0x98;  // tonemap min stops (.x)
constexpr int SCEG_TONEMAP_MAX_OFF    = 0xa8;  // tonemap max stops (.x)
constexpr int SCEG_SKY_TINT_OFF       = 0x58;  // sky-domain RGB tint (float3)
constexpr int SCEG_AMBIENT_TINT_OFF   = 0x68;  // ambient-domain RGB tint
constexpr int SCEG_SUN_TINT_OFF       = 0x78;  // sun-domain RGB tint
constexpr int SCEG_REQUIRED_BYTES     = SCEG_TONEMAP_MAX_OFF + 4;

// Reach TagReference layout: ClassId @ +0 (4 bytes ASCII), padding +4..11,
// TagId @ +12 (uint32). Mirrors the convention used in FogParser.cpp.
int32_t ReadTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// Sanity clamp for the stops scalar: Reach authors +-1 stop typical, +-2 in
// extreme cases. Anything beyond +-6 (= 64x brightness swing) is almost
// certainly a layout mismatch - log and zero it.
bool LooksReasonableStops(float v) {
    if (!isfinite(v)) return false;
    return (v > -6.0f && v < 6.0f);
}

void InitOutDefaults(ZH_ScenarioExposure* outParams) {
    memset(outParams, 0, sizeof(*outParams));
    // Identity tint defaults - neutral grey-scale. The brightness scalar
    // multiplies these in channel-wise; (1,1,1) is the no-op default.
    outParams->SkyTintRGB[0]     = 1.0f;
    outParams->SkyTintRGB[1]     = 1.0f;
    outParams->SkyTintRGB[2]     = 1.0f;
    outParams->AmbientTintRGB[0] = 1.0f;
    outParams->AmbientTintRGB[1] = 1.0f;
    outParams->AmbientTintRGB[2] = 1.0f;
    outParams->SunTintRGB[0]     = 1.0f;
    outParams->SunTintRGB[1]     = 1.0f;
    outParams->SunTintRGB[2]     = 1.0f;
}

// Picks the U10-vs-U13 scnr field offsets. Mirrors FogParser's strategy:
// try the build-specific guess first, fall back to the alternate when the
// primary read looks malformed.
void PickScnrOffsets(CacheType ct, int* outStopsOff, int* outScegTagRefOff) {
    switch (ct) {
        case CacheType::MccHaloReachU13:
            *outStopsOff = SCNR_EXPOSURE_STOPS_OFF_U13;
            *outScegTagRefOff = SCNR_SCEG_TAGREF_OFF_U13;
            return;
        case CacheType::MccHaloReach:
        case CacheType::MccHaloReachU3:
        case CacheType::MccHaloReachU8:
        case CacheType::MccHaloReachU10:
        default:
            *outStopsOff = SCNR_EXPOSURE_STOPS_OFF_U10;
            *outScegTagRefOff = SCNR_SCEG_TAGREF_OFF_U10;
            return;
    }
}

bool ReadScegFields(CacheHandle* cache, uint32_t scegTagId,
                    ZH_ScenarioExposure* outParams)
{
    if (scegTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scegTagId];

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + (size_t)SCEG_REQUIRED_BYTES > cache->size) return false;

    const uint8_t* m = cache->base + metaOff;

    // GLOBAL_BRIGHTNESS_RE.md initially named all six fields as log2-stops
    // offsets that the engine sums. Live runtime data on settlement /
    // forge_world (mesh log: scegAccum=0.980, expoMul=0.507) shows that
    // summing all six produces a NET DARKENING - exactly opposite of
    // what the user observes in MCC. Likely interpretation: the four
    // "offset" fields (master / sky / ambient / detail at +0x20/+0x38/
    // +0x48/+0x88) are real log2 stops the engine subtracts to compensate
    // for HDR over-exposure, but the +0x98/+0xa8 "tonemap min/max" fields
    // are filmic-curve control points (not log2 stops) that we'd been
    // mis-reading as additive terms. Without our viewer doing the HDR
    // over-exposure that those offsets compensate against, applying the
    // subtraction makes the viewer dimmer than the engine output.
    //
    // Until the formula is fully RE'd against the engine PS, only the
    // master offset at +0x20 is read. TODO: pin down (a) the actual
    // accumulator member list, (b) the sign convention vs. the LDR
    // pipeline, (c) whether the engine separately applies +0x98/+0xa8 as
    // tone-curve points.
    float scripted = 0.0f;
    memcpy(&scripted, m + SCEG_MASTER_OFFSET_OFF, 4);
    if (LooksReasonableStops(scripted))
        outParams->ScriptedOffset = scripted;
    // SCEG_SKY/AMBIENT/DETAIL/TONEMAP_* offsets stay declared above so
    // the next RE pass on FUN_18025e3ec can wire them in without
    // re-finding the constants. They are intentionally NOT summed here.
    (void)SCEG_SKY_OFFSET_OFF;
    (void)SCEG_AMBIENT_OFFSET_OFF;
    (void)SCEG_DETAIL_OFFSET_OFF;
    (void)SCEG_TONEMAP_MIN_OFF;
    (void)SCEG_TONEMAP_MAX_OFF;

    // * SCEG TINT - DISABLED / forced NEUTRAL (2026-08, PROVEN garbage by a real-byte float
    // sweep of 0x40..0x90 across settlement(7413) / aftship(5546) / panopticon(7035)):
    //   40=-1.0 44=2.0 48=125 4C=50 50=[-120|-30] 54=[238.5|59.9] 58=[-180|-45] 5C=[358.5|89.9]
    //   60=17.0 64=-15.9 68=11.9 6C=-10.8 70=[0.9|0.7] 74=[0.1|0.3] 78=-0.02 7C=[-82|-123]
    //   80=0.5 84=0.5 88=-0.02 8C=2.5 90=[0.63|0.70]
    // The SCEG_*_TINT_OFF offsets (sky 0x58 / ambient 0x68 / sun 0x78) do NOT point at RGB
    // colour tints - this whole region is scalar RANGES / ANGLES / min-max PAIRS (0x5C~=360 deg|90 deg
    // is a rotation range; 0x70/0x74 is a pair summing to 1.0; the "R" lanes at 0x58/0x68/0x78
    // are -180/11.9/-0.02 = clearly not colour). Reading them as RGB with a [0..8] sanity
    // clamp fabricates "sun=[1,0.5,0.5]"-style tints out of garbage lanes, and a
    // "present_flags bit0" gate on them is just the low 16 bits of a float (aftship bit0=0,
    // settlement bit0=1 - luck). The engine's real per-map sun/ambient COLOUR lives
    // in the 'lght'/atmosphere/sky tags (sceg only SCALES, normally identity) - that's a separate
    // tag-layout RE (TODO). Until then the tints are forced NEUTRAL so NO map gets a spurious
    // warm/coloured cast - the correct identity default.
    for (int i = 0; i < 3; ++i) {
        outParams->SkyTintRGB[i]     = 1.0f;
        outParams->AmbientTintRGB[i] = 1.0f;
        outParams->SunTintRGB[i]     = 1.0f;
    }
    return true;
}

// #189: read the cfxs (camera_fx_settings) EXPOSURE block -> auto-exposure band.
bool ReadCameraFxFields(CacheHandle* cache, uint32_t cfxsTagId,
                        ZH_ScenarioExposure* outParams)
{
    if (cfxsTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[cfxsTagId];
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;
    if ((size_t)metaOff + (size_t)CFXS_REQUIRED_BYTES > cache->size) return false;
    const uint8_t* m = cache->base + metaOff;

    uint16_t flags = 0;
    memcpy(&flags, m + CFXS_FLAGS_OFF, 2);
    outParams->AutoEnabled = (flags & 0x04u) ? 1u : 0u;

    if (getenv("ZH_CFXSDIAG")) {
        fprintf(stderr, "CFXS meta dump (metaOff=%lld) flags=0x%x - floats:\n", (long long)metaOff, flags);
        for (int o = 0; o < 0x50; o += 4) {
            float v; memcpy(&v, m + o, 4);
            uint32_t iv; memcpy(&iv, m + o, 4);
            fprintf(stderr, "  +0x%02x: f=%g  u=0x%08x\n", o, v, iv);
        }
        fflush(stderr);
    }
    float manual = 0.f, rmin = 0.f, rmax = 0.f, tgt = 0.f;
    memcpy(&manual, m + CFXS_EXPOSURE_OFF, 4);
    memcpy(&rmin,   m + CFXS_RANGE_MIN_OFF, 4);
    memcpy(&rmax,   m + CFXS_RANGE_MAX_OFF, 4);
    memcpy(&tgt,    m + CFXS_SCREEN_BRIGHT_OFF, 4);
    // Sanity: EV bounds in log2/stops space (~+-6 typical); target screen brightness 0..16.
    if (isfinite(manual) && manual > -12.f && manual < 12.f) outParams->ManualExposure = manual;
    if (isfinite(rmin)   && rmin   > -12.f && rmin   < 12.f) outParams->AutoMinEV = rmin;
    if (isfinite(rmax)   && rmax   > -12.f && rmax   < 12.f) outParams->AutoMaxEV = rmax;
    // Auto-Exposure Screen Brightness is authored in LOG2/EV space (forge_halo = -3.32193 =
    // log2(0.1)), NOT linear. The old `tgt >= 0` check rejected the negative EV value and fell
    // back to a hardcoded linear 0.25 - a target ~2.5x too bright. Convert EV -> the linear key the
    // meter compares against: key = 2^tgt (forge -> 0.1).
    if (isfinite(tgt) && tgt > -16.f && tgt < 8.f) outParams->AutoTarget = powf(2.0f, tgt);
    // The three per-map bloom parameters (layout in the struct comment above).
    {
        constexpr int CFXS_BLOOM_POINT_OFF     = 0x2C;
        constexpr int CFXS_BLOOM_INHERENT_OFF  = 0x3C;
        constexpr int CFXS_BLOOM_INTENSITY_OFF = 0x4C;
        if ((size_t)metaOff + CFXS_BLOOM_INTENSITY_OFF + 4 <= cache->size) {
            float pt = 0.f, inh = 0.f, inten = 0.f;
            memcpy(&pt,    m + CFXS_BLOOM_POINT_OFF, 4);
            memcpy(&inh,   m + CFXS_BLOOM_INHERENT_OFF, 4);
            memcpy(&inten, m + CFXS_BLOOM_INTENSITY_OFF, 4);
            // Authored values are small positives; reject anything outside a sane band rather
            // than let a misread offset drive the tonemap.
            if (isfinite(pt) && pt >= 0.f && pt <= 16.f &&
                isfinite(inh) && inh >= 0.f && inh <= 4.f &&
                isfinite(inten) && inten >= 0.f && inten <= 8.f) {
                outParams->BloomPoint     = pt;
                outParams->BloomInherent  = inh;
                outParams->BloomIntensity = inten;
                outParams->HasBloom       = 1u;
            }
        }
    }
    // A resolved cfxs whose Range is degenerate (min>=max) is unusable -> treat as not-read.
    if (outParams->AutoMaxEV <= outParams->AutoMinEV) return false;
    return true;
}

// Try the requested set of scnr offsets, falling back to the alternate
// build's layout when the primary read looks malformed (NaN/Inf or |v|>6).
bool ReadScnrAndSceg(CacheHandle* cache, uint32_t scnrTagId,
                     int stopsOff, int scegOff,
                     ZH_ScenarioExposure* outParams,
                     bool* outStopsOk, int32_t* outScegId)
{
    *outStopsOk = false;
    *outScegId  = -1;

    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scnrTagId];
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return false;

    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return false;

    // We need stopsOff+4 AND scegOff+16 bytes from the scnr meta. Bound
    // check both before touching memory.
    size_t needBytes = (size_t)stopsOff + 4;
    if ((size_t)scegOff + 16 > needBytes) needBytes = (size_t)scegOff + 16;
    if ((size_t)metaOff + needBytes > cache->size) return false;

    const uint8_t* m = cache->base + metaOff;

    float stops = 0.0f;
    memcpy(&stops, m + stopsOff, 4);
    if (LooksReasonableStops(stops)) {
        outParams->Stops = stops;
        *outStopsOk = true;
    }

    *outScegId = ReadTagRefId(m + scegOff);
    return true;
}

// Inner walker - caller wraps in SEH.
bool GetExposureInner(CacheHandle* cache, uint32_t scnrTagId,
                      ZH_ScenarioExposure* outParams)
{
    InitOutDefaults(outParams);

    int stopsOff = 0, scegOff = 0;
    PickScnrOffsets(cache->cacheType, &stopsOff, &scegOff);

    bool stopsOk = false;
    int32_t scegId = -1;
    if (!ReadScnrAndSceg(cache, scnrTagId, stopsOff, scegOff,
                         outParams, &stopsOk, &scegId))
        return false;

    // Fallback to the alternate build's offsets if the primary read
    // resolved nothing useful (stops bogus AND scegId null).
    if (!stopsOk && scegId < 0) {
        int altStopsOff = (stopsOff == SCNR_EXPOSURE_STOPS_OFF_U10)
            ? SCNR_EXPOSURE_STOPS_OFF_U13 : SCNR_EXPOSURE_STOPS_OFF_U10;
        int altScegOff  = (scegOff  == SCNR_SCEG_TAGREF_OFF_U10)
            ? SCNR_SCEG_TAGREF_OFF_U13  : SCNR_SCEG_TAGREF_OFF_U10;
        if (!ReadScnrAndSceg(cache, scnrTagId, altStopsOff, altScegOff,
                             outParams, &stopsOk, &scegId)) {
            return false;
        }
    }

    outParams->HasExposure = 1;

    if (scegId >= 0) {
        if (ReadScegFields(cache, (uint32_t)scegId, outParams)) {
            outParams->HasSceg = 1;
        }
    }

    // #189: chase scnr Default Camera FX -> cfxs for the auto-exposure adaptation band.
    // scnrTagId was validated inside ReadScnrAndSceg. Try both build layouts (U10/U13).
    if (scnrTagId < cache->tags.size()) {
        const TagEntry& te = cache->tags[scnrTagId];
        int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
        if (metaOff >= 0) {
            const int camOffs[2] = { SCNR_CAMERAFX_TAGREF_OFF_U10, SCNR_CAMERAFX_TAGREF_OFF_U13 };
            for (int k = 0; k < 2 && outParams->HasCameraFx == 0; ++k) {
                int camOff = camOffs[k];
                if ((size_t)metaOff + (size_t)camOff + 16 > cache->size) continue;
                int32_t cfxsId = ReadTagRefId(cache->base + metaOff + camOff);
                if (cfxsId >= 0 && ReadCameraFxFields(cache, (uint32_t)cfxsId, outParams)) {
                    outParams->HasCameraFx = 1;
                }
            }
        }
    }

    NativeDiag("ScenarioExposure: scnr=%u stops=%.3f sceg=%d scripted=%.3f "
               "skyTint=(%.2f,%.2f,%.2f) ambTint=(%.2f,%.2f,%.2f) "
               "sunTint=(%.2f,%.2f,%.2f)",
               scnrTagId, outParams->Stops, scegId, outParams->ScriptedOffset,
               outParams->SkyTintRGB[0], outParams->SkyTintRGB[1], outParams->SkyTintRGB[2],
               outParams->AmbientTintRGB[0], outParams->AmbientTintRGB[1], outParams->AmbientTintRGB[2],
               outParams->SunTintRGB[0], outParams->SunTintRGB[1], outParams->SunTintRGB[2]);
    return true;
}

bool SehGetExposure(CacheHandle* cache, uint32_t scnrTagId,
                    ZH_ScenarioExposure* outParams)
{
    __try { return GetExposureInner(cache, scnrTagId, outParams); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        if (outParams) InitOutDefaults(outParams);
        return false;
    }
}

} // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

extern "C" __declspec(dllexport) bool __stdcall ZH_SCNR_GetExposure(
    uint64_t cacheHandle, uint32_t scnrTagId, ZH_ScenarioExposure* outParams)
{
    if (!outParams) return false;
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        InitOutDefaults(outParams);
        return false;
    }
    return SehGetExposure(cache, scnrTagId, outParams);
}
