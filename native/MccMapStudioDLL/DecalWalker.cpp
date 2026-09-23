// DecalWalker.cpp
// =============================================================================
// Native walker for Halo Reach sbsp `Runtime Decals` (small-quad surface decals
// - bullet holes, scuff marks, scenery stamps). Companion to MapBspParser
// (which handles cluster + instance geometry and the runtime decorators).
//
// Tag chain (Reach MCC):
//
//   scnr (scenario)
//     Decals Palette                @ scnr + 0x368  (elementSize 0x10)
//       +0x00  tag_reference (16B)  ->  decs (decal_system)
//
//   sbsp (scenario_structure_bsp)
//     Runtime Decals                @ sbsp + 0x1D0  (elementSize 0x28)
//       +0x00  int16   palette index    (-> scnr Decals Palette[])
//       +0x02  int16   manual_bsp_flags
//       +0x04  float4  rotation quat    (XYZW)
//       +0x14  float3  position         (world XYZ)
//       +0x20  float2  scale x / y      (half-extents of the decal quad)
//
//   decs (decal_system, baseSize 0x3C)
//     Decal System tagblock @ +0x2C   (elementSize 0x98)
//       +0x40   Postprocess tagblock   (elementSize 0xB4)
//         +0x10  Textures tagblock     (elementSize 0x18)
//           +0x00  tag_reference (16B) -> bitm (the decal bitmap)
//
// Schema reference: AssemblySource/Plugins/ReachMCC/sbsp.xml + scnr.xml +
// decs.xml (Lord Zedd plugins). Sizes cross-checked against the in-repo
// blam-tags scenario_structure_bsp.json (`scenario_decal` size=40).
//
// Layout caveats:
//   * sbsp.Runtime Decals offset 0x1D0 is the U13/MccHaloReach common offset.
//     U3-U10 hasn't been verified separately yet; if the tag header was
//     compacted in an earlier build the offset will differ. We expose a
//     cache-type-aware picker so per-build overrides can land without code
//     surgery.
//   * scnr.Decals Palette offset 0x368 is the U13/MccHaloReach offset. Same
//     caveat - additional offsets land in PickScnrDecalsPaletteOffset.
//   * `Decals Palette` uses the FULL 16-byte tag_reference (class id at +0,
//     tag id at +12), not the truncated 4-byte ref used inside sbsp's
//     `Decals` block at +0x328 (which is `s_bsp_preplaced_decal_reference`,
//     a different block).
//
// Output ABI: flat array of ZH_DecalInstance via the public export
// ZH_BSP_EnumerateDecals(cacheHandle, sbspTagId, scnrTagId, ...). The caller
// frees with ZH_BSP_FreeDecalBuffer.
//
// Defensive contract - same shape as SkyWalker:
//   * SEH-wrapped at the public boundary.
//   * Class-code checks at every cross-tag dereference (scnr / sbsp / decs / bitm).
//   * Index validation against cache->tags.size().
//   * Hard cap on per-bsp instance count (DECAL_TOTAL_INSTANCE_CAP).
// =============================================================================

#include "pch.h"
#include "MapCacheCommon.h"

#include <windows.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits>

using namespace zh_mcc;

namespace {

// ---------------------------------------------------------------------------
// Public ABI
// ---------------------------------------------------------------------------
//
// One element per runtime decal. Bitmap resolution may fail (deca/decs broken,
// palette index OOB) in which case bitmapTagId == 0xFFFFFFFFu - caller can
// skip those rather than rendering a magenta quad.
//
// halfExtents are the per-axis half-widths of the decal quad in world space
// - sbsp's Runtime Decal `scale x / y` ARE the half-extents (verified against
// engine code by inspection of scenario_decal's real_vector_2d field).
//
// facing is the unit normal vector derived from the quaternion's rotated +Z
// (i.e. the decal sticks out of a surface whose normal points along this
// vector). Pre-rotated to be ready for the viewer scene to build a quad.
//
// `u`, `v` are the in-plane basis vectors orthogonal to facing - also derived
// from the quaternion (rotated +X / +Y). The viewer uses them to build the four
// quad corners as `position +- u*halfExtents.x +- v*halfExtents.y`.
#pragma pack(push, 1)
struct ZH_DecalInstance {
    float    position[3];        // world XYZ
    float    facing[3];          // surface normal (unit vector)
    float    uAxis[3];           // in-plane U basis (unit, perpendicular to facing)
    float    vAxis[3];           // in-plane V basis (unit, perpendicular to facing AND uAxis)
    float    halfExtents[2];     // quad half-width / half-height (world units)
    uint32_t bitmapTagId;        // 0xFFFFFFFFu when unresolved
    int32_t  decsTagId;          // -1 when unresolved
    int16_t  paletteIndex;       // raw sbsp.Runtime Decals[i].palette index
    int16_t  pad0;
    int32_t  blendModeRaw;       // decs.DecalSystem[0].Postprocess[0]+0x68 (int32)
                                 //   engine enum; the viewer maps it to its blend-mode enum. -1 if
                                 //   not read (decs unresolved / OOB).
    // === FloatConstants entries 0..3 from Postprocess[0] ===
    // FloatConstants is a tagblock at Postprocess+0x1C with elementSize 0x10
    // (four float32 a/b/c/d). The render-method template (rmt2) defines what
    // each entry means - typical convention is FC[0] = UV transform
    // (scale_u, scale_v, offset_u, offset_v), FC[1] = albedo/tint colour
    // (R, G, B, A), FC[2..] = specular / emissive / per-shader params.
    //
    // Diag showed FC[0]=(1,1,0,0) on every Countdown decal which is the
    // identity UV transform - confirming the convention. The actual
    // per-decal tint is in one of FC[1..3]; the viewer's decal install path selects
    // via MMS_DECAL_TINT_SLOT.
    //
    // All NaN if the entry doesn't exist.
    float    floatConst0[4];     // (was named `tint` - kept for backward-compat in name)
    float    floatConst1[4];
    float    floatConst2[4];
    float    floatConst3[4];

    float    decsUnk08;          // decs+0x8 float (candidate "Maximum Decal Radius")
    float    decsUnk10;          // decs+0x10 float (candidate scale multiplier or bias)
    float    decsUnk24;          // decs+0x24 float (candidate fade param)
    float    decsUnk28;          // decs+0x28 float
    float    decsUnk38;          // decs+0x38 float (after Decal System tagblock)

    // === rmt2 (shader template) walk results ===
    // rmt2 ID resolved from decs.Postprocess[0]+0x0 (Shader Template tagref).
    // -1 if not resolved.
    int32_t  rmt2TagId;
    // Parameter NAMES from rmt2+0x48 Float Constants tagblock (one stringid
    // per entry; positionally indexes into decs.Postprocess[0].FloatConstants).
    // Up to 4 entries surfaced for diag; full count and overflow flag below.
    // Strings are NUL-terminated; truncated to 31 bytes + NUL = 32 bytes each.
    int32_t  fcParamCount;       // total rmt2.FloatConstants entry count
    char     fcParamName0[32];
    char     fcParamName1[32];
    char     fcParamName2[32];
    char     fcParamName3[32];

    // Sapien-RE'd scale-resolution fields (FUN_1408AD440):
    //   final_scale_x = (sbsp.scale_x or random(rangeMin, rangeMax)) * scaleXMul
    //   final_scale_y =  sbsp.scale_y or random(rangeMin, rangeMax)
    // the viewer's decal install path performs the same multiplication. scaleXMul defaults to
    // 1.0 if the decs tag couldn't be read (so existing behaviour is preserved
    // - only decals whose decs supplies a non-1 multiplier change size).
    // fields read from DecalSystem entry (size 0x98),
    // NOT from Postprocess. The walker's prior pp+0x70/0x94 reads landed in
    // V3 as "drop because int16 indices, not floats" - but the right struct
    // is the DecalSystem entry. Halo Infinite decs.xml end-of-entry schema
    // verifies layout 1:1 with Reach.
    float    scaleXDefaultMin;   // sys+0x70  (NaN when unresolved)
    float    scaleXDefaultMax;   // sys+0x74  (NaN when unresolved)
    float    scaleYDefaultMin;   // sys+0x78  V4 NEW
    float    scaleYDefaultMax;   // sys+0x7C  V4 NEW
    float    clampAngleDeg;      // sys+0x88  V4 NEW (cosClamp used in surface-fit)
    float    cullAngleDeg;       // sys+0x8C  V4 NEW (back-face cull half-angle)
    float    depthBias;          // sys+0x90  V4 NEW (per-decs rasterizer bias)
    float    scaleXMul;          // sys+0x94  (1.0 when unresolved)

    // Sapien-RE'd bitm sprite-sequence sub-region (FUN_1408AD300). For
    // sprite-flagged decals (decs.PP[0]+0x4 bit 2 set), the bitm tag has
    // a SpriteSequences tagblock @ +0x30 (or @ +0x70 fallback), each
    // sequence holding a Sprites sub-block @ +0x34 (0x20-byte elements).
    // The first sprite's (u_min, u_max, v_min, v_max) carries a sub-rect
    // of the atlas - the rendered decal samples ONLY that sub-rect.
    //
    // Without these the texture sampled is the WHOLE atlas: "yellow
    // squares should be stripes" + "text way too small" + "content takes
    // tiny portion of the quad" all come from missing this step. A decal
    // authored as a square (scale_x = scale_y) with a stripe sprite
    // (u_size = 1, v_size = 0.1) IS rendered as a 10:1 stripe by the
    // engine - the sprite's aspect overrides the tag's square scale.
    //
    // Default (no sprite / non-sprite shader): hasSprite=0, full UV.
    uint8_t  hasSprite;          // 1 if the sprite sub-region is valid
    uint8_t  spritePad0;
    uint8_t  spritePad1;
    uint8_t  spritePad2;
    float    spriteUMin;         // 0.0 default
    float    spriteVMin;         // 0.0 default
    float    spriteUSize;        // 1.0 default
    float    spriteVSize;        // 1.0 default

    // === Full Textures[] roster from decs.DecalSystem[0].Postprocess[0]. ===
    // bitmapTagId (above) is Textures[0]. These are Textures[1..3], each
    // 0xFFFFFFFFu when the slot is empty / not a bitm tagref (some Reach
    // rmt2 templates put FloatConstant params in the same slot - those are
    // filtered out by class-code check in the walker).
    //
    // Per shaderTemplate parameter NAME, the slot roles are:
    //   * Standard bitmap (0x16E9): [base_map, tint_color, intensity, mod]
    // - only slot 0 is a bitm (others are FCs).
    //   * Tiling 3-slot (0x10A3, 0x00E5, 0x0E6B, 0x1652): [base, u, v, ""]
    //   * Tiling 4-slot (0x09F0, 0x01DB, 0x168B, 0x16D4):
    //       [base_map, bump_map, u_tiles, v_tiles] - slot 1 = bump
    //   * Tiling 5-slot (0x1675):
    //       [base_map, alpha_map, bump_map, u_tiles] - slot 1 = alpha
    //   * Vector / SDF (0x16BA, 0x16C5): [base, vec_sharp, aa_tweak, tint]
    //   * Glass / refraction (0x16FC, 0x1704):
    //       [base_map, bump_map, interier, mask_threshold] - slot 1 = bump
    //
    // The install path picks the right interpretation by comparing the rmt2
    // FloatConstant param NAMES (fcParamNameN above) - slot index alone
    // doesn't disambiguate alpha_map vs bump_map.
    uint32_t bitmapTagId2;       // Textures[1]
    uint32_t bitmapTagId3;       // Textures[2]
    uint32_t bitmapTagId4;       // Textures[3]

    // Sort Layer enum8 from DecalSystem+0x62. The viewer sorts decals by
    // this ascending so Post-Pass signage renders last (on top of Normal
    // scorches). Default = 2 (Normal).
    uint8_t  sortLayer;
    uint8_t  pad1;
    uint16_t pad2;

    // === DECAL_SLOT_FIX: rmt2.Usages[] TEXTURE-slot names ===
    // The bitmap -> sampler binding is POSITIONAL: rmt2.Usages[i] (read from
    // rmt2+0x6C, the Usages tagblock) names sampler slot i, and slot i is
    // decs.Postprocess[0].Textures[i] (Textures[0]=bitmapTagId, [1..3]=
    // bitmapTagId2..4). This is the SAME contract the BSP resolver uses
    // (MapBspParser.cpp ResolveTerrainLayers: Usages[i] <-> ShaderMaps[i]).
    //
    // Prior code identified texture-slot roles from the rmt2 ARGUMENTS
    // block (rmt2+0x48, the FLOAT-param names: tint/u_tiles/...) - a
    // DIFFERENT, independently-ordered list - capped at 4. Using the
    // float-param names to label texture slots is why exactly one texture
    // per map (whichever shader's bitmap/float ordering diverged) bound to
    // the wrong slot. usageName[i] is the authoritative slot label.
    int32_t  usageCount;          // total rmt2.Usages[] entry count
    char     usageName[8][32];    // Usages[0..7] (slot labels, NUL-term, 31+NUL)
};
#pragma pack(pop)
// 360 (prior) + 4 (usageCount) + 256 (8x32 usageName) = 620
static_assert(sizeof(ZH_DecalInstance) == 620,
              "ZH_DecalInstance layout drift - keep in sync with the Rust mirror in crates/hms-native/src/lib.rs");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_SBSP = "sbsp";
constexpr const char* TC_DECS = "decs";
constexpr const char* TC_DECA = "deca";
constexpr const char* TC_BITM = "bitm";

// Lord Zedd ReachMCC sbsp.xml: Runtime Decals @ 0x1D0, elementSize 0x28.
constexpr int  RUNTIME_DECAL_BLOCK_SIZE      = 0x28;  // 40 bytes
constexpr int  RD_OFFSET_PALETTE_INDEX       = 0x00;
constexpr int  RD_OFFSET_ROTATION_QUAT       = 0x04;  // float4 (xyzw)
constexpr int  RD_OFFSET_POSITION            = 0x14;  // float3
constexpr int  RD_OFFSET_SCALE_XY            = 0x20;  // float2

// Lord Zedd ReachMCC scnr.xml: Decals Palette @ 0x368, elementSize 0x10.
constexpr int  DECAL_PALETTE_BLOCK_SIZE      = 0x10;  // 16 bytes (one tag_reference)
constexpr int  DP_OFFSET_DECAL_TAGREF        = 0x00;  // full 16B tagref

// Lord Zedd ReachMCC decs.xml chain:
//   decs + 0x2C   Decal System tagblock (elementSize 0x98)
//     +0x40        Postprocess tagblock (elementSize 0xB4)
//       +0x10      Textures tagblock (elementSize 0x18)
//         +0x00     tagref bitmap (16B)
constexpr int  DECS_OFF_DECAL_SYSTEM         = 0x2C;
// Sort Layer enum8 inside DecalSystem entry.
constexpr int  DECS_DECAL_SYSTEM_SORT_LAYER  = 0x62;
constexpr int  DECS_DECAL_SYSTEM_ELEM_SIZE   = 0x98;
constexpr int  DECS_OFF_POSTPROCESS          = 0x40;  // relative to Decal System entry
constexpr int  DECS_POSTPROCESS_ELEM_SIZE    = 0xB4;
constexpr int  DECS_OFF_TEXTURES             = 0x10;  // relative to Postprocess entry
constexpr int  DECS_TEXTURES_ELEM_SIZE       = 0x18;
constexpr int  DECS_TEXTURES_BITMAP_OFFSET   = 0x00;  // tagref @ start of texture entry
constexpr int  DECS_POSTPROCESS_BLENDMODE    = 0x68;  // int32 blend mode (engine enum)
constexpr int  DECS_POSTPROCESS_FLOATCONSTS  = 0x1C;  // tagblock Float Constants
// Scale-resolution fields (RE'd from Sapien FUN_1408AD440, the decal render
// allocate-from-pool function called once per Postprocess[N] pass).
//
// Sapien formula:
//   sx_in = sbsp.RuntimeDecal[i].scale_x  (or random(pp+0x70, pp+0x74) if ~0)
//   sy_in = sbsp.RuntimeDecal[i].scale_y  (or random(pp+0x70, pp+0x74) if ~0)
//   decal_pool[i] + 0x6c  =  sx_in * (pp + 0x94)   <- THE scale_x multiplier
//   decal_pool[i] + 0x70  =  sy_in                  (NO multiplier on Y)
//
// pp+0x94 is a per-pass scalar that makes Reach's tiny scale_x authored values
// (often 0.05..0.5 in tags) render at the correct world size - for some
// decals this is 1.0, for others it's 5..20+, which is why some decals
// should cover a whole wall.
constexpr int  DECS_POSTPROCESS_SCALE_X_RANGE = 0x70;  // float[2] min/max range for default
constexpr int  DECS_POSTPROCESS_SCALE_X_MUL   = 0x94;  // float - applied to scale_x always
constexpr int  DECS_POSTPROCESS_FLAGS         = 0x04;  // uint32 flags; bit 2 = "use sprite sub-rect"

// Reach bitm tag layout for sprite sequences. Sapien's FUN_1408AD300 reads
// bitm+0x30 - but that's the RUNTIME (in-memory) offset after the engine
// reshapes the tag. The OFFLINE (file cache) layout we walk has different
// offsets, documented in Lord Zedd's reach bitm.xml plugin:
//   bitm + 0x68  -> Sequences tagblock, elementSize 0x40
//                  (Left/Right = U min/max, Top/Bottom = V min/max
//                   match the Sapien sprite-walker output)
//   sequence + 0x34 -> Sprites tagblock, elementSize 0x20
//   sprite + 0x08  -> Left   (float, u_min)
//   sprite + 0x0C  -> Right  (float, u_max)
//   sprite + 0x10  -> Top    (float, v_min)
//   sprite + 0x14  -> Bottom (float, v_max)
//
// Earlier mistake (offsets 0x30/0x70) was reading garbage / structurally
// invalid tagblocks - every decal logged `sprite=n` because the count
// was 0 or out of bounds. Lord Zedd's plugin is the source-of-truth for
// the offline schema.
constexpr int  BITM_OFF_SPRITE_SEQUENCES_A    = 0x68;
constexpr int  BITM_OFF_SPRITE_SEQUENCES_B    = 0x68;  // no fallback for offline schema
constexpr int  BITM_SPRITE_SEQ_ELEM_SIZE      = 0x40;
constexpr int  BITM_SPRITE_SEQ_OFF_SPRITES    = 0x34;  // tagblock inside the sequence
constexpr int  BITM_SPRITE_ELEM_SIZE          = 0x20;
constexpr int  BITM_SPRITE_OFF_UMIN           = 0x08;
constexpr int  BITM_SPRITE_OFF_UMAX           = 0x0C;
constexpr int  BITM_SPRITE_OFF_VMIN           = 0x10;
constexpr int  BITM_SPRITE_OFF_VMAX           = 0x14;
constexpr int  DECS_FLOATCONST_ELEM_SIZE     = 0x10;  // 4xfloat per entry
constexpr int  DECS_POSTPROCESS_SHADER_TMPL  = 0x00;  // tagref at +0x0 = rmt2 ref

// rmt2 schema (Reach ReachMCC plugin):
//   baseSize=0x84
//   +0x00 tagref VertexShader
//   +0x10 tagref PixelShader
//   +0x48 tagblock FloatConstants  elementSize=4 (one stringid per entry)
//   +0x6C tagblock Textures        elementSize=4 (one stringid per entry)
constexpr int  RMT2_OFF_FLOAT_CONSTANTS      = 0x48;  // Arguments (float-param names)
// DECAL_SLOT_FIX: rmt2.Usages tagblock - the TEXTURE-slot
// label list, parallel to the shader's bitmap array. Same offset (108/0x6C)
// the BSP terrain resolver uses (MapBspParser.cpp OFF_RMT_USAGES). Entry is
// a single stringid (4 bytes) but the tagblock element carries the standard
// stringid-block stride; we read the first 4 bytes of each element.
constexpr int  RMT2_OFF_USAGES               = 0x6C;
constexpr int  RMT2_USAGE_ELEM_SIZE          = 4;
constexpr int  RMT2_FLOATCONST_ELEM_SIZE     = 0x04;  // just stringid (param name)
constexpr const char* TC_RMT2                = "rmt2";
// Base-tag candidate fields (Reach decs baseSize=0x3C). The xml has them as
// "Unknown" - we surface them per-decal so the viewer diag can correlate values
// with visual outcomes and we can pick whichever field is the size multiplier.
constexpr int  DECS_OFF_UNK_08               = 0x08;  // candidate Max Decal Radius
constexpr int  DECS_OFF_UNK_10               = 0x10;  // candidate scale multiplier
constexpr int  DECS_OFF_UNK_24               = 0x24;
constexpr int  DECS_OFF_UNK_28               = 0x28;
constexpr int  DECS_OFF_UNK_38               = 0x38;

// Aggregated decs info pulled in one pass. Defaulted to "missing" sentinels
// (NaN for floats, -1 for blend) so the viewer can detect absence.
struct DecsResolveInfo {
    int32_t blendModeRaw;
    float   floatConst[4][4]; // FloatConstants[0..3].(a,b,c,d). All NaN if absent.
    float   unk08;
    float   unk10;
    float   unk24;
    float   unk28;
    float   unk38;

    // rmt2 walk results
    int32_t rmt2TagId;
    int32_t fcParamCount;     // total rmt2.FloatConstants (Arguments) count
    char    fcParamName[4][32];
    // DECAL_SLOT_FIX: rmt2.Usages[] - the TEXTURE-slot label
    // list (rmt2+0x6C), parallel to decs.Textures[]. Distinct from the
    // FloatConstants/Arguments names above. usageName[i] labels Textures[i].
    int32_t usageCount;
    char    usageName[8][32];

    // scale fields live on the DecalSystem entry
    // (sysMeta, size 0x98), NOT on the Postprocess block as V3 thought.
    // Sapien FUN_1408AD440 reads from `lVar4 = FUN_1408ADC10()` which is the
    // DecalSystem ptr. Halo Infinite's decs.xml end-of-entry schema confirms:
    //   sys+0x70 float scale_x_default_min
    //   sys+0x74 float scale_x_default_max
    //   sys+0x78 float scale_y_default_min     (V4 added)
    //   sys+0x7C float scale_y_default_max     (V4 added)
    //   sys+0x88 float clamp_angle_deg         (V4 added)
    //   sys+0x8C float cull_angle_deg          (V4 added)
    //   sys+0x90 float depth_bias              (V4 added)
    //   sys+0x94 float runtime_bitmap_aspect (scale_x multiplier)
    // Walker now reads from sysMeta. Final pose formula:
    //   final_scale_x = (sbsp.scale_x or random(rangeMin, rangeMax)) * mul
    //   final_scale_y =  sbsp.scale_y or random(yRangeMin, yRangeMax)
    float   scaleXDefaultMin;  // sys+0x70
    float   scaleXDefaultMax;  // sys+0x74
    float   scaleYDefaultMin;  // sys+0x78  (V4 new)
    float   scaleYDefaultMax;  // sys+0x7C  (V4 new)
    float   clampAngleDeg;     // sys+0x88  (V4 new)
    float   cullAngleDeg;      // sys+0x8C  (V4 new)
    float   depthBias;         // sys+0x90  (V4 new)
    float   scaleXMul;         // sys+0x94 - THE missing scale field

    // Sprite sub-region from bitm.SpriteSequences[0].Sprites[0]
    // (Sapien FUN_1408AD300). Only valid when pp.flags bit 2 is set.
    // Default (no sprite): (0,0,1,1) = full atlas.
    bool    hasSprite;
    float   spriteUMin;
    float   spriteVMin;
    float   spriteUSize;
    float   spriteVSize;

    // Textures[1..3] bitm tag ids (0xFFFFFFFFu when absent / non-bitm).
    uint32_t bitmapTagId2;
    uint32_t bitmapTagId3;
    uint32_t bitmapTagId4;

    // DecalSystem+0x62 Sort Layer enum8:
    //   0 / 1 = Pre-Pass (drawn FIRST, before normal decals)
    //   2     = Normal   (default)
    //   3     = Post-Pass (drawn LAST, on top of others)
    // the viewer's decal install path sorts decals by SortLayer ASCENDING before adding
    // to the scene group so render order matches engine intent.
    uint8_t sortLayer;
};

static void ClearDecsInfo(DecsResolveInfo* di) {
    di->blendModeRaw = -1;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            di->floatConst[i][j] = std::numeric_limits<float>::quiet_NaN();
    di->unk08 = di->unk10 = di->unk24 = di->unk28 = di->unk38
              = std::numeric_limits<float>::quiet_NaN();
    di->rmt2TagId = -1;
    di->fcParamCount = 0;
    for (int i = 0; i < 4; ++i) di->fcParamName[i][0] = '\0';
    di->usageCount = 0;
    for (int i = 0; i < 8; ++i) di->usageName[i][0] = '\0';
    // Default scaleXMul = 1.0 so callers can multiply unconditionally; NaN
    // here would propagate to a NaN scale and the decal would be invisible.
    di->scaleXDefaultMin = di->scaleXDefaultMax = std::numeric_limits<float>::quiet_NaN();
    di->scaleYDefaultMin = di->scaleYDefaultMax = std::numeric_limits<float>::quiet_NaN();
    di->clampAngleDeg    = std::numeric_limits<float>::quiet_NaN();
    di->cullAngleDeg     = std::numeric_limits<float>::quiet_NaN();
    di->depthBias        = 0.0f;
    di->scaleXMul = 1.0f;
    di->hasSprite    = false;
    di->spriteUMin   = 0.0f;
    di->spriteVMin   = 0.0f;
    di->spriteUSize  = 1.0f;
    di->spriteVSize  = 1.0f;
    di->bitmapTagId2 = 0xFFFFFFFFu;
    di->bitmapTagId3 = 0xFFFFFFFFu;
    di->bitmapTagId4 = 0xFFFFFFFFu;
    di->sortLayer    = 2;  // Normal default
}

// Walk bitm.SpriteSequences[0].Sprites[0] (or fallback @ +0x70) to read the
// first sprite's UV sub-rect. Returns true if a valid sprite was found.
// `out*` always populated - defaults (0,0,1,1) on failure.
static bool ResolveBitmFirstSpriteUV(CacheHandle* cache, int32_t bitmTagId,
                                     float* outUMin, float* outVMin,
                                     float* outUSize, float* outVSize)
{
    *outUMin  = 0.0f;
    *outVMin  = 0.0f;
    *outUSize = 1.0f;
    *outVSize = 1.0f;
    if (bitmTagId < 0) return false;
    if ((uint32_t)bitmTagId >= cache->tags.size()) return false;
    const TagEntry& bte = cache->tags[bitmTagId];
    if (bte.classIndex < 0) return false;
    if (memcmp(bte.classCode, TC_BITM, 4) != 0) return false;

    int64_t bitmMetaOff = TagMetaFileOff(cache, bte.metaPointerRaw);
    if (bitmMetaOff < 0) return false;
    if ((size_t)bitmMetaOff + (size_t)BITM_OFF_SPRITE_SEQUENCES_B + 12 > cache->size) return false;
    const uint8_t* bitmMeta = cache->base + bitmMetaOff;

    // Try +0x30 first, fall back to +0x70 if empty (per Sapien preferred
    // order in FUN_1408AD300).
    TagBlockRef seqBlk = ReadTagBlock(bitmMeta + BITM_OFF_SPRITE_SEQUENCES_A);
    if (seqBlk.count <= 0) {
        seqBlk = ReadTagBlock(bitmMeta + BITM_OFF_SPRITE_SEQUENCES_B);
    }
    if (seqBlk.count <= 0 || seqBlk.count > 1024) return false;

    int64_t seqOff = TagMetaFileOff(cache, seqBlk.pointer);
    if (seqOff < 0) return false;
    // Sprite-sequence element is 0x40 bytes.
    if ((size_t)seqOff + (size_t)BITM_SPRITE_SEQ_ELEM_SIZE > cache->size) return false;
    const uint8_t* seqMeta = cache->base + seqOff;  // sequence[0]

    // Sequence + 0x34 is a tagblock of Sprites (0x20-byte elements).
    if ((size_t)BITM_SPRITE_SEQ_OFF_SPRITES + 12 > (size_t)BITM_SPRITE_SEQ_ELEM_SIZE) return false;
    TagBlockRef spriteBlk = ReadTagBlock(seqMeta + BITM_SPRITE_SEQ_OFF_SPRITES);
    if (spriteBlk.count <= 0 || spriteBlk.count > 1024) return false;

    int64_t spriteOff = TagMetaFileOff(cache, spriteBlk.pointer);
    if (spriteOff < 0) return false;
    if ((size_t)spriteOff + (size_t)BITM_SPRITE_ELEM_SIZE > cache->size) return false;
    const uint8_t* spriteMeta = cache->base + spriteOff;  // sprite[0]

    float uMin, uMax, vMin, vMax;
    memcpy(&uMin, spriteMeta + BITM_SPRITE_OFF_UMIN, 4);
    memcpy(&uMax, spriteMeta + BITM_SPRITE_OFF_UMAX, 4);
    memcpy(&vMin, spriteMeta + BITM_SPRITE_OFF_VMIN, 4);
    memcpy(&vMax, spriteMeta + BITM_SPRITE_OFF_VMAX, 4);

    // Sanity: NaN / inverted ranges -> bail. Values outside [0,1] are
    // suspicious but technically valid for clamp-sampled atlases - clamp
    // to a wide safe band rather than rejecting.
    if (!(uMin == uMin) || !(uMax == uMax) || !(vMin == vMin) || !(vMax == vMax)) return false;
    if (uMax <= uMin || vMax <= vMin) return false;
    if (uMin < -2.0f || uMax > 3.0f || vMin < -2.0f || vMax > 3.0f) return false;

    *outUMin  = uMin;
    *outVMin  = vMin;
    *outUSize = uMax - uMin;
    *outVSize = vMax - vMin;
    return true;
}

// Sanity caps.
constexpr int  DECAL_RUNTIME_COUNT_SANITY    = 0x10000;   // 65k
constexpr int  DECAL_PALETTE_COUNT_SANITY    = 0x1000;    // 4k
constexpr int  DECAL_TOTAL_INSTANCE_CAP      = 4096;      // hard cap per sbsp

// ---------------------------------------------------------------------------
// Schema offset pickers
// ---------------------------------------------------------------------------
//
// sbsp Runtime Decals + scnr Decals Palette offsets. Reach MCC release-U10
// share the same layout (no header drift in this region per Lord Zedd's
// plugin), but we expose a per-cache-type picker so a future U13-only build
// can be wedged in without touching every call site.

int PickSbspRuntimeDecalsOffset(CacheType ct) {
    (void)ct;
    return 0x1D0;
}

int PickScnrDecalsPaletteOffset(CacheType ct) {
    (void)ct;
    return 0x368;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int32_t ReadFullTagRefId(const uint8_t* tagRef) {
    // Reclaimer Gen3+ TagReference: ClassId @ +0, padding @ +4..11, TagId @ +12.
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// Quaternion (x,y,z,w) -> 3x3 rotation matrix; decompose into the basis
// Sapien/the-engine ACTUALLY uses for sbsp/scnr decal data. Sapien's
// decal-load function (sapien.exe!FUN_1404FD7F0) decodes the quat with the same matrix
// builder we use (FUN_1404378C0, col0/col1/col2 = rotated +X/+Y/+Z) and
// THEN performs this transformation before passing to
// c_decal_system::create_at_index:
//
//   col0_out = col0 XOR 0x80000000       // negate col0 (forward)
//   col1     = thrown away                // engine reconstructs via cross
//   col2_out = col2 XOR 0x80000000       // negate col2 (up)
//
// The engine stores ONLY two basis vectors per decal - the negated col0
// and negated col2 - and reconstructs the third axis on-the-fly. Sapien
// authors the quat such that col0 (forward) points INTO the wall (matching
// the entity convention "forward is where the entity faces"); the engine
// negates to get the OUTWARD surface normal. col2 is similarly flipped.
//
// To match the engine, our basis is:
//   N (surface normal) = -col0   (outward, away from the wall)
//   V (in-plane up)    = -col2   (also negated by the engine)
//   U (in-plane left)  = -col1   (= N x V via right-hand rule, which works
//                                  out to -col1 for our right-handed quat)
//
// This is identical to "negate every column" of the matrix decoded by
// the standard quat -> 3x3 builder. See:
//   sapien.exe!FUN_140179EB0 - walks scenario decals (vtable iterator)
//   sapien.exe!FUN_1404FEA60 - palette lookup wrapper
//   sapien.exe!FUN_1404FD7F0 - decodes quat, negates col0+col2, calls
//                                create_at_index (THE smoking gun)
//   sapien.exe!FUN_1404FB8C0 - c_decal_system::create_at_index
//   sapien.exe!FUN_1404FCDE0 - c_decal_system::initialize_for_game
//                                (allocates 0x2C0 x 0x60-byte datums)
//
// Why this and not the standard graphics convention (+Z = N):
//
//   * haloreach.dll!FUN_18015DB8C (the runtime decal-create handler)
//     stores the input surface normal at offsets +0x1C..+0x24 of its
//     internal decal struct. That offset is the "forward" (col0) slot
//     of a Bungie matrix4x3 - the engine itself treats N as col0.
//
//   * sapien.exe!FUN_1404E5B00 (scenario_decal parent-transform helper)
//     XORs the sign bit of scale_x (at +0x20 in the 40-byte record) when
//     the parent transform's determinant is negative (mirror). This only
//     makes sense if scale_x is the half-extent along an in-plane axis - 
//     specifically col1 (left), by the Bungie convention col0=forward,
//     col1=left, col2=up. Pins scale_x to col1, scale_y to col2.
//
//   * sapien.exe!FUN_1404385D0 builds a Bungie matrix4x3 from a
//     (position, quat) pair by calling the standard XYZW quat -> 3x3
//     decompose and packing col0/col1/col2 directly into the
//     forward/left/up slots. So a Sapien-authored sbsp quat decodes to
//     a matrix4x3 whose col0 is "forward" = N.
//
// DEFINITIVE (RE'd from Sapien's FUN_1404FAAD0 / collision raycast): N is
// NOT -col0. Sapien's decal raycast uses -col2 as the cast direction (into the surface), which
// means +col2 is the OUTWARD surface normal. The author convention is:
//   col2 = up of decal = surface outward normal (the floor's +Z, a wall's
//          horizontal outward normal, etc).
//
// Sapien FUN_1404FAAD0 confirms (param_3 = pointer to engine A-slot = -col2):
//     fVar4 = -col2.x;  fVar5 = -col2.y;  fVar2 = -col2.z;
//     normalize(-col2);
//     cast_start = pos - normalized * SOME_DIST   // = pos + (col2 * D)
//     cast_dir   = -col2                          // INTO the surface
//     FUN_1404FECF0(piVar17, ..., &cast_start, &cast_dir);  // RAYCAST
//
// So the cast goes FROM "above the surface along +col2" TO "into the surface
// along -col2". The hit is the actual rendered surface. The +col2 direction
// is the surface outward normal. (For impact decals N comes from the impact
// surface normal directly; for sbsp.RuntimeDecal it comes from the authored
// quat's col2.)
//
// Note the column NEGATION inside create_at_index - the engine negates
// col0 + col2 for internal storage. But
// the negated -col2 is the cast direction, not the surface normal. The
// surface normal is the un-negated +col2. (This is what the cast actually
// hits when it shoots in the -col2 direction.)
//
// References:
//   sapien.exe!FUN_1404378C0 - quat -> 3x3 matrix, column-major output:
//                                col0 @ matrix[0..2], col1 @ [3..5], col2 @ [6..8]
//   sapien.exe!FUN_1404FD7F0 - sbsp.RuntimeDecal loader; passes -col2 as
//                                normal-slot arg, -col0 as up-slot arg to FB8C0
//   sapien.exe!FUN_1404FB8C0 - create_at_index; stores -col2 at +0x1c (A),
//                                -col0 at +0x34 (C); B = C x A = col0 x col2
//   sapien.exe!FUN_1404FAAD0 - uses +0x1c (= -col2) as raycast direction
//   sapien.exe!FUN_1408AD440 - render allocate; decal+0x6c = scale_x * pp+0x94
//
// Texture U/V mapping: col2 = N is the texture's perpendicular direction.
// The in-plane axes col0 (forward) and col1 (left) define U/V. By Bungie's
// usual authoring convention:
//   U (texture-X "right")     = -col1  (col1 is "left", so -col1 is "right")
//   V (texture-Y "down" or "up") = +col0  (col0 is "forward" = decal's "up")
// MMS_DECAL_UV_VARIANT env var can swap / negate these if the texture comes
// out rotated 90 deg on specific decals.
void QuatToBasis(float qx, float qy, float qz, float qw,
                 float outU[3], float outV[3], float outN[3])
{
    // Normalise defensively - engine quats are unit but garbage from a busted
    // tag could produce NaNs downstream.
    float len2 = qx*qx + qy*qy + qz*qz + qw*qw;
    if (len2 < 1e-12f) {
        // Identity quat fallback. Identity quat -> col0=+X, col1=+Y, col2=+Z.
        // Phase-4 mapping: N=+col2=+Z, U=-col1=-Y, V=+col0=+X.
        outN[0] = 0;  outN[1] = 0;  outN[2] = 1;
        outU[0] = 0;  outU[1] = -1; outU[2] = 0;
        outV[0] = 1;  outV[1] = 0;  outV[2] = 0;
        return;
    }
    float inv = 1.0f / sqrtf(len2);
    qx *= inv; qy *= inv; qz *= inv; qw *= inv;

    float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    float wx = qw * qx, wy = qw * qy, wz = qw * qz;

    // Column-major quat -> 3x3:
    //   col0 = (1-2(y^2+z^2),  2(xy+wz),   2(xz-wy))   (rotated +X = forward)
    //   col1 = (2(xy-wz),    1-2(x^2+z^2), 2(yz+wx))   (rotated +Y = left)
    //   col2 = (2(xz+wy),    2(yz-wx),   1-2(x^2+y^2)) (rotated +Z = up)
    //
    // N (outward surface normal) = +col2
    // U (texture-X right)        = -col1
    // V (texture-Y up)           = +col0
    outN[0] =  2.0f * (xz + wy);
    outN[1] =  2.0f * (yz - wx);
    outN[2] =  1.0f - 2.0f * (xx + yy);

    outU[0] = -(2.0f * (xy - wz));
    outU[1] = -(1.0f - 2.0f * (xx + zz));
    outU[2] = -(2.0f * (yz + wx));

    outV[0] =  1.0f - 2.0f * (yy + zz);
    outV[1] =  2.0f * (xy + wz);
    outV[2] =  2.0f * (xz - wy);
}

// Resolve decs.Decal System[0].Postprocess[0].Textures[0].Bitmap Reference -> bitm tag id.
// Also extracts:
//   * Postprocess+0x68 (Blend Mode int32) - selects D3D blend state per decal
//   * Postprocess+0x1C FloatConstants[0] (a,b,c,d) - per-decal tint colour
//   * decs base unknown floats at +0x08/+0x10/+0x24/+0x28/+0x38 - candidates
//     for a scale multiplier / fade param the engine may apply on top of the
//     sbsp scale_x/y. These are surfaced for diagnosis; pick whichever
//     correlates with visual size.
//
// Returns -1 on any failure. `info` is written even on failure (with NaN /
// sentinel values for the fields that couldn't be read).
int32_t ResolveDecsBitmapTagId(CacheHandle* cache, int32_t decsTagId,
                               DecsResolveInfo* info)
{
    if (info) ClearDecsInfo(info);
    if (decsTagId < 0) return -1;
    if ((uint32_t)decsTagId >= cache->tags.size()) return -1;
    const TagEntry& dte = cache->tags[decsTagId];
    if (dte.classIndex < 0) return -1;
    // Accept either 'decs' (Reach Decal System) or 'deca' (older single-decal
    // tag). The bitmap chain for `deca` is structurally similar - first
    // Postprocess[0].Textures[0] - so the same offsets work when the tag
    // is decs-shaped. Defensive: bail on any other class to avoid mis-reads.
    bool isDecs = memcmp(dte.classCode, TC_DECS, 4) == 0;
    bool isDeca = memcmp(dte.classCode, TC_DECA, 4) == 0;
    if (!isDecs && !isDeca) return -1;

    int64_t metaOff = TagMetaFileOff(cache, dte.metaPointerRaw);
    if (metaOff < 0) return -1;
    // Need at least up to 0x3C for the base-tag unknown floats; the Decal
    // System tagblock starts at 0x2C (12 bytes inline = up to 0x38), then
    // the trailing float at 0x38. Cap at 0x3C bytes minimum.
    if ((size_t)metaOff + 0x3C > cache->size) return -1;
    const uint8_t* decsMeta = cache->base + metaOff;

    // Read the base-tag candidate floats. Cheap raw reads - the caller will
    // pick whichever (if any) correlates with the visual decal sizing.
    if (info) {
        memcpy(&info->unk08, decsMeta + DECS_OFF_UNK_08, 4);
        memcpy(&info->unk10, decsMeta + DECS_OFF_UNK_10, 4);
        memcpy(&info->unk24, decsMeta + DECS_OFF_UNK_24, 4);
        memcpy(&info->unk28, decsMeta + DECS_OFF_UNK_28, 4);
        memcpy(&info->unk38, decsMeta + DECS_OFF_UNK_38, 4);
    }

    // Decal System block (one entry expected per decs).
    TagBlockRef sysBlk = ReadTagBlock(decsMeta + DECS_OFF_DECAL_SYSTEM);
    if (sysBlk.count <= 0 || sysBlk.count > 16) return -1;
    int64_t sysOff = TagMetaFileOff(cache, sysBlk.pointer);
    if (sysOff < 0) return -1;
    if ((size_t)sysOff + DECS_DECAL_SYSTEM_ELEM_SIZE > cache->size) return -1;
    const uint8_t* sysMeta = cache->base + sysOff;

    // Sort Layer (enum8) at +0x62 inside the
    // DecalSystem entry. Values: 0/1 = Pre-Pass, 2 = Normal (default),
    // 3 = Post-Pass. The the viewer's decal install path sorts the decal list by this
    // ascending before adding to the scene group, so Post-Pass signage
    // renders on top of Normal scorches/scuffs as the engine does.
    if ((size_t)DECS_DECAL_SYSTEM_SORT_LAYER + 1 <= (size_t)DECS_DECAL_SYSTEM_ELEM_SIZE) {
        info->sortLayer = sysMeta[DECS_DECAL_SYSTEM_SORT_LAYER];
    }

    // Postprocess block (one entry expected).
    if ((size_t)DECS_OFF_POSTPROCESS + 12 > (size_t)DECS_DECAL_SYSTEM_ELEM_SIZE) return -1;
    TagBlockRef ppBlk = ReadTagBlock(sysMeta + DECS_OFF_POSTPROCESS);
    if (ppBlk.count <= 0 || ppBlk.count > 16) return -1;
    int64_t ppOff = TagMetaFileOff(cache, ppBlk.pointer);
    if (ppOff < 0) return -1;
    if ((size_t)ppOff + DECS_POSTPROCESS_ELEM_SIZE > cache->size) return -1;
    const uint8_t* ppMeta = cache->base + ppOff;

    // Textures block (any count; pick first; also slurp 1..3 if present).
    if ((size_t)DECS_OFF_TEXTURES + 12 > (size_t)DECS_POSTPROCESS_ELEM_SIZE) return -1;
    TagBlockRef texBlk = ReadTagBlock(ppMeta + DECS_OFF_TEXTURES);
    if (texBlk.count <= 0 || texBlk.count > 64) return -1;
    int64_t texOff = TagMetaFileOff(cache, texBlk.pointer);
    if (texOff < 0) return -1;
    // Must cover up to min(texBlk.count, 4) entries to read the full roster.
    int rosterCount = texBlk.count < 4 ? texBlk.count : 4;
    if ((size_t)texOff + (size_t)rosterCount * (size_t)DECS_TEXTURES_ELEM_SIZE > cache->size) return -1;
    const uint8_t* texMeta = cache->base + texOff;

    // First texture entry's bitmap tagref.
    int32_t bitmapId = ReadFullTagRefId(texMeta + DECS_TEXTURES_BITMAP_OFFSET);
    if (bitmapId < 0) return -1;
    if ((uint32_t)bitmapId >= cache->tags.size()) return -1;
    if (memcmp(cache->tags[bitmapId].classCode, TC_BITM, 4) != 0) return -1;

    // Textures[1..3] - bitm refs only (FloatConstant-payload slots return
    // -1 here when their class code isn't 'bitm', which is the same sentinel
    // we surface to callers as 0xFFFFFFFFu).
    if (info) {
        auto readBitmAtSlot = [&](int slotIdx) -> uint32_t {
            if (slotIdx >= texBlk.count) return 0xFFFFFFFFu;
            const uint8_t* entry = texMeta + (size_t)slotIdx * (size_t)DECS_TEXTURES_ELEM_SIZE;
            int32_t id = ReadFullTagRefId(entry + DECS_TEXTURES_BITMAP_OFFSET);
            if (id < 0) return 0xFFFFFFFFu;
            if ((uint32_t)id >= cache->tags.size()) return 0xFFFFFFFFu;
            if (memcmp(cache->tags[id].classCode, TC_BITM, 4) != 0) return 0xFFFFFFFFu;
            return (uint32_t)id;
        };
        info->bitmapTagId2 = readBitmAtSlot(1);
        info->bitmapTagId3 = readBitmAtSlot(2);
        info->bitmapTagId4 = readBitmAtSlot(3);
    }

    // Postprocess + 0x68 = int32 Blend Mode. Lives in the SAME postprocess
    // entry we just walked, so the bounds check above (sizeof entry >= 0xB4)
    // already covers the read.
    if (info)
    {
        if ((size_t)DECS_POSTPROCESS_BLENDMODE + 4 <= (size_t)DECS_POSTPROCESS_ELEM_SIZE)
        {
            int32_t bm = 0;
            memcpy(&bm, ppMeta + DECS_POSTPROCESS_BLENDMODE, 4);
            info->blendModeRaw = bm;
        }

        // scale + range + clamp/cull angle + depth_bias
        // ALL live on the DecalSystem entry (sysMeta), NOT on the Postprocess
        // block. Sapien FUN_1408AD440 reads lVar4+0x70..
        // lVar4+0x94 where lVar4 = FUN_1408ADC10() returns the DecalSystem
        // entry ptr (stride 0x98). Halo Infinite decs.xml end-of-entry schema
        // (size 0x98) maps 1:1: +0x70 scale_x_min, +0x74 max, +0x78 sy_min,
        // +0x7C sy_max, +0x88 clamp_angle, +0x8C cull_angle, +0x90 depth_bias,
        // +0x94 runtime_bitmap_aspect.
        if ((size_t)DECS_DECAL_SYSTEM_ELEM_SIZE >= 0x98)
        {
            memcpy(&info->scaleXDefaultMin, sysMeta + 0x70, 4);
            memcpy(&info->scaleXDefaultMax, sysMeta + 0x74, 4);
            memcpy(&info->scaleYDefaultMin, sysMeta + 0x78, 4);
            memcpy(&info->scaleYDefaultMax, sysMeta + 0x7C, 4);
            memcpy(&info->clampAngleDeg,    sysMeta + 0x88, 4);
            memcpy(&info->cullAngleDeg,     sysMeta + 0x8C, 4);
            memcpy(&info->depthBias,        sysMeta + 0x90, 4);
            memcpy(&info->scaleXMul,        sysMeta + 0x94, 4);
            // Sanity: out-of-range / non-finite -> safe defaults.
            auto isSane = [](float v, float lo, float hi){ return std::isfinite(v) && v >= lo && v <= hi; };
            if (!isSane(info->scaleXMul,     0.001f, 1024.0f)) info->scaleXMul = 1.0f;
            if (!isSane(info->depthBias,   -1024.0f, 1024.0f)) info->depthBias = 0.0f;
            if (!isSane(info->clampAngleDeg, 0.0f,   180.0f))  info->clampAngleDeg = std::numeric_limits<float>::quiet_NaN();
            if (!isSane(info->cullAngleDeg,  0.0f,   180.0f))  info->cullAngleDeg  = std::numeric_limits<float>::quiet_NaN();
            if (!isSane(info->scaleXDefaultMin, 0.0f, 1024.0f)) info->scaleXDefaultMin = std::numeric_limits<float>::quiet_NaN();
            if (!isSane(info->scaleXDefaultMax, 0.0f, 1024.0f)) info->scaleXDefaultMax = std::numeric_limits<float>::quiet_NaN();
            if (!isSane(info->scaleYDefaultMin, 0.0f, 1024.0f)) info->scaleYDefaultMin = std::numeric_limits<float>::quiet_NaN();
            if (!isSane(info->scaleYDefaultMax, 0.0f, 1024.0f)) info->scaleYDefaultMax = std::numeric_limits<float>::quiet_NaN();
        }
        else
        {
            info->scaleXDefaultMin = std::numeric_limits<float>::quiet_NaN();
            info->scaleXDefaultMax = std::numeric_limits<float>::quiet_NaN();
            info->scaleYDefaultMin = std::numeric_limits<float>::quiet_NaN();
            info->scaleYDefaultMax = std::numeric_limits<float>::quiet_NaN();
            info->clampAngleDeg    = std::numeric_limits<float>::quiet_NaN();
            info->cullAngleDeg     = std::numeric_limits<float>::quiet_NaN();
            info->depthBias        = 0.0f;
            info->scaleXMul        = 1.0f;
        }

        // -- DIAGNOSTIC: dump every interesting float in the decs.PP[0] block
        // and ALSO in the decs base tag itself. One of these holds the
        // scale multiplier that turns 0.791 -> ~4m at render time.
        // (Sapien runtime says pp+0x94; offline schema offset may differ.)
        NativeDiag("DecsDiag bitm=0x%X pp_floats:", bitmapId);
        for (int dofs = 0x00; dofs < DECS_POSTPROCESS_ELEM_SIZE; dofs += 4) {
            if ((size_t)dofs + 4 > (size_t)DECS_POSTPROCESS_ELEM_SIZE) break;
            float fv = 0.0f;
            memcpy(&fv, ppMeta + dofs, 4);
            // Plausible scale candidates: finite, positive, in [0.1, 100].
            if (fv > 0.1f && fv < 100.0f && fv == fv) {
                NativeDiag("  pp+0x%02X = %.3f", dofs, fv);
            }
        }
        // Also dump decs base tag floats - Maximum Decal Radius / world size
        // candidates might live there (offset 0x0..0x40 inline).
        NativeDiag("DecsDiag bitm=0x%X base_floats:", bitmapId);
        for (int dofs = 0x00; dofs < 0x40; dofs += 4) {
            float fv = 0.0f;
            memcpy(&fv, decsMeta + dofs, 4);
            if (fv > 0.1f && fv < 100.0f && fv == fv) {
                NativeDiag("  decs+0x%02X = %.3f", dofs, fv);
            }
        }

        // Sprite sub-region - Sapien only uses it when pp.flags bit 2 is
        // set, but for our purposes we ALWAYS look it up: if the bitm has
        // a sprite sequence, the artist authored a specific sub-rect that
        // the engine SHOULD respect. The pp-flag gate is a perf/feature
        // toggle in the engine; for the viewer the sprite UV always
        // produces the right visual.
        //
        // Even non-flagged decals can have a default first sprite that
        // matches (0,0,1,1) - in which case ResolveBitmFirstSpriteUV
        // returns false and we keep the (0,0,1,1) default.
        float spUMin, spVMin, spUSize, spVSize;
        if (ResolveBitmFirstSpriteUV(cache, bitmapId, &spUMin, &spVMin, &spUSize, &spVSize)) {
            info->hasSprite   = true;
            info->spriteUMin  = spUMin;
            info->spriteVMin  = spVMin;
            info->spriteUSize = spUSize;
            info->spriteVSize = spVSize;
        }

        // Postprocess + 0x1C = Float Constants tagblock (elementSize 0x10).
        // Read FloatConstants[0..3] = (a,b,c,d) each. Convention based on
        // [DECAL DIAG] dump from Countdown:
        //   [0] = UV transform (scale_u, scale_v, offset_u, offset_v)
        // - observed as (1,1,0,0) on every decal
        //   [1] = albedo / tint colour (R, G, B, A)
        //   [2..] = additional shader params (specular, emissive, fade...)
        // Each entry that doesn't exist stays NaN.
        TagBlockRef fcBlk = ReadTagBlock(ppMeta + DECS_POSTPROCESS_FLOATCONSTS);
        if (fcBlk.count > 0 && fcBlk.count < 64) {
            int64_t fcOff = TagMetaFileOff(cache, fcBlk.pointer);
            int readEntries = fcBlk.count > 4 ? 4 : fcBlk.count;
            for (int k = 0; k < readEntries; ++k) {
                size_t entryStart = (size_t)fcOff + (size_t)k * DECS_FLOATCONST_ELEM_SIZE;
                if (entryStart + DECS_FLOATCONST_ELEM_SIZE <= cache->size) {
                    memcpy(info->floatConst[k], cache->base + entryStart,
                           DECS_FLOATCONST_ELEM_SIZE);
                }
            }
        }

        // Postprocess+0x0 = Shader Template tagref -> rmt2. Walk it to get
        // the parameter NAME list - the viewer diag uses this to identify
        // which FloatConstants slot encodes the tint colour (Reach decals
        // share an rmt2 template whose Float Constants block names every
        // parameter positionally; the slot named "albedo_color" /
        // "tint_color" / "color" IS the tint).
        int32_t rmt2Id = ReadFullTagRefId(ppMeta + DECS_POSTPROCESS_SHADER_TMPL);
        if (rmt2Id >= 0 && (uint32_t)rmt2Id < cache->tags.size()) {
            const TagEntry& rmt2Te = cache->tags[rmt2Id];
            if (rmt2Te.classIndex >= 0 &&
                memcmp(rmt2Te.classCode, TC_RMT2, 4) == 0)
            {
                int64_t rmt2Off = TagMetaFileOff(cache, rmt2Te.metaPointerRaw);
                if (rmt2Off >= 0 &&
                    (size_t)rmt2Off + (size_t)RMT2_OFF_FLOAT_CONSTANTS + 12
                        <= cache->size)
                {
                    info->rmt2TagId = rmt2Id;
                    const uint8_t* rmt2Meta = cache->base + rmt2Off;
                    TagBlockRef paramBlk = ReadTagBlock(
                        rmt2Meta + RMT2_OFF_FLOAT_CONSTANTS);
                    if (paramBlk.count > 0 && paramBlk.count < 256) {
                        info->fcParamCount = paramBlk.count;
                        int64_t paramOff = TagMetaFileOff(cache, paramBlk.pointer);
                        if (paramOff >= 0 &&
                            (size_t)paramOff +
                                (size_t)paramBlk.count * RMT2_FLOATCONST_ELEM_SIZE
                                    <= cache->size)
                        {
                            int n = paramBlk.count > 4 ? 4 : paramBlk.count;
                            for (int k = 0; k < n; ++k) {
                                size_t off = (size_t)paramOff +
                                             (size_t)k * RMT2_FLOATCONST_ELEM_SIZE;
                                int32_t sid;
                                memcpy(&sid, cache->base + off, 4);
                                const char* name = ResolveStringId(cache, sid);
                                if (name) {
                                    size_t len = strlen(name);
                                    if (len > 31) len = 31;
                                    memcpy(info->fcParamName[k], name, len);
                                    info->fcParamName[k][len] = '\0';
                                } else {
                                    snprintf(info->fcParamName[k], 32,
                                             "sid=0x%X", (unsigned)sid);
                                }
                            }
                        }
                    }

                    // DECAL_SLOT_FIX: walk rmt2.Usages[] (0x6C) - 
                    // the TEXTURE-slot label list, parallel to decs.Textures[].
                    // usageName[i] labels Textures[i]; index i IS the sampler
                    // register. This is the authoritative slot binding (same
                    // contract as the BSP resolver). Up to 8 slots surfaced.
                    if ((size_t)rmt2Off + (size_t)RMT2_OFF_USAGES + 12 <= cache->size) {
                        TagBlockRef usagesBlk = ReadTagBlock(
                            rmt2Meta + RMT2_OFF_USAGES);
                        if (usagesBlk.count > 0 && usagesBlk.count < 256) {
                            info->usageCount = usagesBlk.count;
                            int64_t usagesOff = TagMetaFileOff(cache, usagesBlk.pointer);
                            if (usagesOff >= 0 &&
                                (size_t)usagesOff +
                                    (size_t)usagesBlk.count * RMT2_USAGE_ELEM_SIZE
                                        <= cache->size)
                            {
                                int un = usagesBlk.count > 8 ? 8 : usagesBlk.count;
                                for (int k = 0; k < un; ++k) {
                                    size_t off = (size_t)usagesOff +
                                                 (size_t)k * RMT2_USAGE_ELEM_SIZE;
                                    int32_t sid;
                                    memcpy(&sid, cache->base + off, 4);
                                    const char* name = ResolveStringId(cache, sid);
                                    if (name) {
                                        size_t len = strlen(name);
                                        if (len > 31) len = 31;
                                        memcpy(info->usageName[k], name, len);
                                        info->usageName[k][len] = '\0';
                                    } else {
                                        snprintf(info->usageName[k], 32,
                                                 "sid=0x%X", (unsigned)sid);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return bitmapId;
}

// ---------------------------------------------------------------------------
// Palette walker - read scnr.Decals Palette[] -> array of decs tag ids
// (with -1 for null/unresolved entries).
// ---------------------------------------------------------------------------

bool ReadScnrDecalsPalette(CacheHandle* cache, uint32_t scnrTagId,
                           int32_t** outPalette, int32_t* outCount)
{
    *outPalette = nullptr;
    *outCount   = 0;

    if (scnrTagId >= cache->tags.size()) return false;
    const TagEntry& te = cache->tags[scnrTagId];
    if (te.classIndex < 0) return false;
    if (memcmp(te.classCode, TC_SCNR, 4) != 0) return false;

    int64_t scnrMetaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (scnrMetaOff < 0) return false;
    int paletteOff = PickScnrDecalsPaletteOffset(cache->cacheType);
    if ((size_t)scnrMetaOff + (size_t)paletteOff + 12 > cache->size) return false;
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    TagBlockRef pbk = ReadTagBlock(scnrMeta + paletteOff);
    if (pbk.count <= 0) return true;            // valid: empty palette
    if (pbk.count > DECAL_PALETTE_COUNT_SANITY) return false;

    int64_t paletteArrOff = TagMetaFileOff(cache, pbk.pointer);
    if (paletteArrOff < 0) return false;
    if ((size_t)paletteArrOff + (size_t)pbk.count * DECAL_PALETTE_BLOCK_SIZE > cache->size)
        return false;

    int32_t* arr = (int32_t*)malloc(sizeof(int32_t) * (size_t)pbk.count);
    if (!arr) return false;

    for (int i = 0; i < pbk.count; ++i) {
        const uint8_t* entry = cache->base + paletteArrOff + i * DECAL_PALETTE_BLOCK_SIZE;
        arr[i] = ReadFullTagRefId(entry + DP_OFFSET_DECAL_TAGREF);
        // Class validation: must be 'decs' or 'deca' to be useful. Anything
        // else gets demoted to -1 so the bitmap-resolver short-circuits.
        if (arr[i] >= 0 && (uint32_t)arr[i] < cache->tags.size()) {
            const TagEntry& dte = cache->tags[arr[i]];
            bool isDecs = memcmp(dte.classCode, TC_DECS, 4) == 0;
            bool isDeca = memcmp(dte.classCode, TC_DECA, 4) == 0;
            if (!isDecs && !isDeca) arr[i] = -1;
        } else {
            arr[i] = -1;
        }
    }

    *outPalette = arr;
    *outCount   = pbk.count;
    return true;
}

// ---------------------------------------------------------------------------
// sbsp.Runtime Decals walker
// ---------------------------------------------------------------------------

bool EnumerateInner(CacheHandle* cache,
                    uint32_t sbspTagId, uint32_t scnrTagId,
                    ZH_DecalInstance** outBuf, uint32_t* outLen,
                    int* diagPaletteCount, int* diagRuntimeCount,
                    int* diagResolved, int* diagUnresolved)
{
    *outBuf = nullptr;
    *outLen = 0;
    *diagPaletteCount = 0;
    *diagRuntimeCount = 0;
    *diagResolved     = 0;
    *diagUnresolved   = 0;

    if (sbspTagId >= cache->tags.size()) return false;
    const TagEntry& sbspTe = cache->tags[sbspTagId];
    if (sbspTe.classIndex < 0) return false;
    if (memcmp(sbspTe.classCode, TC_SBSP, 4) != 0) return false;

    // Read the scnr palette first so we can resolve palette index -> decs id.
    int32_t* palette = nullptr;
    int32_t paletteCount = 0;
    bool palOk = ReadScnrDecalsPalette(cache, scnrTagId, &palette, &paletteCount);
    if (!palOk) {
        NativeDiag("DecalWalker: scnr=0x%X decals palette read failed (sbsp=0x%X)",
                   scnrTagId, sbspTagId);
        // Not fatal - sbsp may have runtime decals without a palette (each
        // bitmap-id stays unresolved), so continue with paletteCount=0.
        palette = nullptr;
        paletteCount = 0;
    }
    *diagPaletteCount = paletteCount;

    int64_t sbspMetaOff = TagMetaFileOff(cache, sbspTe.metaPointerRaw);
    if (sbspMetaOff < 0) {
        if (palette) free(palette);
        return false;
    }
    int runtimeDecalsOff = PickSbspRuntimeDecalsOffset(cache->cacheType);
    if ((size_t)sbspMetaOff + (size_t)runtimeDecalsOff + 12 > cache->size) {
        if (palette) free(palette);
        return false;
    }
    const uint8_t* sbspMeta = cache->base + sbspMetaOff;

    TagBlockRef rd = ReadTagBlock(sbspMeta + runtimeDecalsOff);
    if (rd.count <= 0) {
        // No runtime decals on this sbsp - valid, empty result.
        if (palette) free(palette);
        return true;
    }
    if (rd.count > DECAL_RUNTIME_COUNT_SANITY) {
        NativeDiag("DecalWalker: sbsp=0x%X runtime decal count %d insane (>%d)",
                   sbspTagId, rd.count, DECAL_RUNTIME_COUNT_SANITY);
        if (palette) free(palette);
        return false;
    }

    int64_t rdArrOff = TagMetaFileOff(cache, rd.pointer);
    if (rdArrOff < 0) {
        if (palette) free(palette);
        return false;
    }
    if ((size_t)rdArrOff + (size_t)rd.count * RUNTIME_DECAL_BLOCK_SIZE > cache->size) {
        if (palette) free(palette);
        return false;
    }
    *diagRuntimeCount = rd.count;

    int capped = rd.count;
    if (capped > DECAL_TOTAL_INSTANCE_CAP) capped = DECAL_TOTAL_INSTANCE_CAP;

    ZH_DecalInstance* buf = (ZH_DecalInstance*)malloc(sizeof(ZH_DecalInstance) * (size_t)capped);
    if (!buf) {
        if (palette) free(palette);
        return false;
    }
    memset(buf, 0, sizeof(ZH_DecalInstance) * (size_t)capped);

    // Per-decs cache to avoid re-resolving the same palette entry per decal
    // hit. paletteCount can be 0 (no palette) in which case we never touch
    // this cache. The cache holds both the bitmap id AND the full
    // DecsResolveInfo (tint, blend mode, unknown floats) since they're all
    // read in the same pass.
    int32_t*           bmCache    = nullptr;
    DecsResolveInfo*   infoCache  = nullptr;
    if (paletteCount > 0) {
        bmCache   = (int32_t*)malloc(sizeof(int32_t) * (size_t)paletteCount);
        infoCache = (DecsResolveInfo*)malloc(sizeof(DecsResolveInfo) * (size_t)paletteCount);
        if (bmCache && infoCache) {
            for (int i = 0; i < paletteCount; ++i) {
                bmCache[i] = -2;   // sentinel "not yet computed"
                ClearDecsInfo(&infoCache[i]);
            }
        } else {
            if (bmCache)   { free(bmCache);   bmCache = nullptr; }
            if (infoCache) { free(infoCache); infoCache = nullptr; }
        }
    }

    uint32_t emitted = 0;
    for (int i = 0; i < capped; ++i) {
        const uint8_t* rec = cache->base + rdArrOff + i * RUNTIME_DECAL_BLOCK_SIZE;
        int16_t palIdx = R16(rec + RD_OFFSET_PALETTE_INDEX);

        float qx, qy, qz, qw;
        memcpy(&qx, rec + RD_OFFSET_ROTATION_QUAT + 0,  4);
        memcpy(&qy, rec + RD_OFFSET_ROTATION_QUAT + 4,  4);
        memcpy(&qz, rec + RD_OFFSET_ROTATION_QUAT + 8,  4);
        memcpy(&qw, rec + RD_OFFSET_ROTATION_QUAT + 12, 4);

        float px, py, pz;
        memcpy(&px, rec + RD_OFFSET_POSITION + 0, 4);
        memcpy(&py, rec + RD_OFFSET_POSITION + 4, 4);
        memcpy(&pz, rec + RD_OFFSET_POSITION + 8, 4);

        float sx, sy;
        memcpy(&sx, rec + RD_OFFSET_SCALE_XY + 0, 4);
        memcpy(&sy, rec + RD_OFFSET_SCALE_XY + 4, 4);

        // preserve sx==0 / sy==0 as the engine's
        // "use authored default range" sentinel. Sapien FUN_1408AD440 only
        // substitutes the per-decs default range when |input.scale| < epsilon
        // - clobbering to 0.5 here masked that signal, so every decal that
        // authored its size via the decs default range collapsed to a
        // uniform 0.5 m stamp regardless of intent. the viewer's decal install path now
        // detects sx==0 and substitutes 0.5 * (decsRangeMin + decsRangeMax)
        // from the V4-resolved DecalSystem entry. Only clamp truly absurd
        // values here.
        if (!std::isfinite(sx) || sx < 0.0f || sx > 1024.0f) sx = 0.0f;
        if (!std::isfinite(sy) || sy < 0.0f || sy > 1024.0f) sy = 0.0f;

        float u[3], v[3], n[3];
        QuatToBasis(qx, qy, qz, qw, u, v, n);

        ZH_DecalInstance& D = buf[emitted];
        D.position[0] = px; D.position[1] = py; D.position[2] = pz;
        D.facing[0]   = n[0]; D.facing[1]   = n[1]; D.facing[2]   = n[2];
        D.uAxis[0]    = u[0]; D.uAxis[1]    = u[1]; D.uAxis[2]    = u[2];
        D.vAxis[0]    = v[0]; D.vAxis[1]    = v[1]; D.vAxis[2]    = v[2];
        D.halfExtents[0] = sx;
        D.halfExtents[1] = sy;
        D.paletteIndex = palIdx;
        D.pad0 = 0;
        D.blendModeRaw = -1;
        for (int t = 0; t < 4; ++t) {
            D.floatConst0[t] = std::numeric_limits<float>::quiet_NaN();
            D.floatConst1[t] = std::numeric_limits<float>::quiet_NaN();
            D.floatConst2[t] = std::numeric_limits<float>::quiet_NaN();
            D.floatConst3[t] = std::numeric_limits<float>::quiet_NaN();
        }
        D.decsUnk08 = D.decsUnk10 = D.decsUnk24 = D.decsUnk28 = D.decsUnk38
                    = std::numeric_limits<float>::quiet_NaN();
        D.rmt2TagId    = -1;
        D.fcParamCount = 0;
        D.fcParamName0[0] = D.fcParamName1[0] =
        D.fcParamName2[0] = D.fcParamName3[0] = '\0';
        D.usageCount = 0;
        for (int u = 0; u < 8; ++u) D.usageName[u][0] = '\0';
        D.bitmapTagId = 0xFFFFFFFFu;
        D.bitmapTagId2 = 0xFFFFFFFFu;
        D.bitmapTagId3 = 0xFFFFFFFFu;
        D.bitmapTagId4 = 0xFFFFFFFFu;
        D.decsTagId   = -1;
        // Scale fields default - unmultiplied (1x) so unresolved decals
        // render at the same size as before; resolved decals get the real
        // pp+0x94 from the decs tag.
        D.scaleXDefaultMin = std::numeric_limits<float>::quiet_NaN();
        D.scaleXDefaultMax = std::numeric_limits<float>::quiet_NaN();
        D.scaleYDefaultMin = std::numeric_limits<float>::quiet_NaN();
        D.scaleYDefaultMax = std::numeric_limits<float>::quiet_NaN();
        D.clampAngleDeg    = std::numeric_limits<float>::quiet_NaN();
        D.cullAngleDeg     = std::numeric_limits<float>::quiet_NaN();
        D.depthBias        = 0.0f;
        D.scaleXMul        = 1.0f;
        D.hasSprite   = 0;
        D.spritePad0  = D.spritePad1 = D.spritePad2 = 0;
        D.spriteUMin  = 0.0f;
        D.spriteVMin  = 0.0f;
        D.spriteUSize = 1.0f;
        D.spriteVSize = 1.0f;
        D.sortLayer   = 2;        // Normal default ()
        D.pad1        = 0;
        D.pad2        = 0;

        // Resolve palette index -> decs tag id -> bitmap tag id + extended info.
        if (palette && palIdx >= 0 && palIdx < paletteCount) {
            int32_t decsId = palette[palIdx];
            D.decsTagId = decsId;
            if (decsId >= 0) {
                int32_t          bmId = -1;
                DecsResolveInfo  localInfo;
                DecsResolveInfo* pInfo = &localInfo;
                if (bmCache && infoCache) {
                    if (bmCache[palIdx] == -2) {
                        bmCache[palIdx] = ResolveDecsBitmapTagId(cache, decsId, &infoCache[palIdx]);
                    }
                    bmId  = bmCache[palIdx];
                    pInfo = &infoCache[palIdx];
                } else {
                    bmId = ResolveDecsBitmapTagId(cache, decsId, &localInfo);
                }
                D.blendModeRaw = pInfo->blendModeRaw;
                for (int t = 0; t < 4; ++t) {
                    D.floatConst0[t] = pInfo->floatConst[0][t];
                    D.floatConst1[t] = pInfo->floatConst[1][t];
                    D.floatConst2[t] = pInfo->floatConst[2][t];
                    D.floatConst3[t] = pInfo->floatConst[3][t];
                }
                D.rmt2TagId    = pInfo->rmt2TagId;
                D.fcParamCount = pInfo->fcParamCount;
                memcpy(D.fcParamName0, pInfo->fcParamName[0], 32);
                memcpy(D.fcParamName1, pInfo->fcParamName[1], 32);
                memcpy(D.fcParamName2, pInfo->fcParamName[2], 32);
                memcpy(D.fcParamName3, pInfo->fcParamName[3], 32);
                D.usageCount = pInfo->usageCount;
                for (int u = 0; u < 8; ++u)
                    memcpy(D.usageName[u], pInfo->usageName[u], 32);
                D.decsUnk08 = pInfo->unk08;
                D.decsUnk10 = pInfo->unk10;
                D.decsUnk24 = pInfo->unk24;
                D.decsUnk28 = pInfo->unk28;
                D.decsUnk38 = pInfo->unk38;
                D.scaleXDefaultMin = pInfo->scaleXDefaultMin;
                D.scaleXDefaultMax = pInfo->scaleXDefaultMax;
                D.scaleYDefaultMin = pInfo->scaleYDefaultMin;
                D.scaleYDefaultMax = pInfo->scaleYDefaultMax;
                D.clampAngleDeg    = pInfo->clampAngleDeg;
                D.cullAngleDeg     = pInfo->cullAngleDeg;
                D.depthBias        = pInfo->depthBias;
                D.scaleXMul        = pInfo->scaleXMul;
                D.sortLayer        = pInfo->sortLayer;
                D.hasSprite        = pInfo->hasSprite ? (uint8_t)1 : (uint8_t)0;
                D.spriteUMin       = pInfo->spriteUMin;
                D.spriteVMin       = pInfo->spriteVMin;
                D.spriteUSize      = pInfo->spriteUSize;
                D.spriteVSize      = pInfo->spriteVSize;
                D.bitmapTagId2     = pInfo->bitmapTagId2;
                D.bitmapTagId3     = pInfo->bitmapTagId3;
                D.bitmapTagId4     = pInfo->bitmapTagId4;
                if (bmId >= 0) {
                    D.bitmapTagId = (uint32_t)bmId;
                    ++(*diagResolved);
                } else {
                    ++(*diagUnresolved);
                }
            } else {
                ++(*diagUnresolved);
            }
        } else {
            ++(*diagUnresolved);
        }

        ++emitted;
    }

    *outBuf = buf;
    *outLen = emitted;

    if (palette)   free(palette);
    if (bmCache)   free(bmCache);
    if (infoCache) free(infoCache);
    return true;
}

bool EnumerateSeh(CacheHandle* cache,
                  uint32_t sbspTagId, uint32_t scnrTagId,
                  ZH_DecalInstance** outBuf, uint32_t* outLen,
                  int* diagPaletteCount, int* diagRuntimeCount,
                  int* diagResolved, int* diagUnresolved)
{
    __try {
        return EnumerateInner(cache, sbspTagId, scnrTagId,
                              outBuf, outLen,
                              diagPaletteCount, diagRuntimeCount,
                              diagResolved, diagUnresolved);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        NativeDiag("DecalWalker: SEH fault sbsp=0x%X scnr=0x%X",
                   sbspTagId, scnrTagId);
        if (*outBuf) { free(*outBuf); *outBuf = nullptr; }
        *outLen = 0;
        return false;
    }
}

}  // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

// Enumerate every runtime decal on the given sbsp. Resolves palette indices
// through scnr.Decals Palette to a decs tag and decs Postprocess[0].Textures[0]
// to a bitmap tag.
//
//   outBuffer - malloc'd ZH_DecalInstance[*outCount]. Caller frees
//                      via ZH_BSP_FreeDecalBuffer.
//   outCount - count actually emitted.
//
// Returns 1 on success (count may be zero - valid "no decals" result),
// 0 on hard failure (bad cache handle, bad sbsp/scnr tag id, OOB reads).
extern "C" __declspec(dllexport) int __stdcall ZH_BSP_EnumerateDecals(
    uint64_t cacheHandle,
    uint32_t sbspTagId,
    uint32_t scnrTagId,
    ZH_DecalInstance** outBuffer,
    uint32_t* outCount)
{
    if (outBuffer) *outBuffer = nullptr;
    if (outCount)  *outCount  = 0;
    if (!outBuffer || !outCount) return 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        NativeDiag("DecalWalker: bad cacheHandle=0x%llX",
                   (unsigned long long)cacheHandle);
        return 0;
    }

    ZH_DecalInstance* buf = nullptr;
    uint32_t          len = 0;
    int dPal = 0, dRT = 0, dRes = 0, dUn = 0;
    bool ok = EnumerateSeh(cache, sbspTagId, scnrTagId,
                           &buf, &len,
                           &dPal, &dRT, &dRes, &dUn);

    NativeDiag("Decals[sbsp=0x%X scnr=0x%X] palette=%d runtime=%d emitted=%u resolved=%d unresolved=%d",
               sbspTagId, scnrTagId, dPal, dRT, len, dRes, dUn);

    if (!ok) {
        if (buf) { free(buf); buf = nullptr; }
        return 0;
    }

    *outBuffer = buf;
    *outCount  = len;
    return 1;
}

// Free a buffer returned by ZH_BSP_EnumerateDecals. Safe with NULL.
extern "C" __declspec(dllexport) void __stdcall ZH_BSP_FreeDecalBuffer(
    ZH_DecalInstance* buf)
{
    if (buf) free(buf);
}
