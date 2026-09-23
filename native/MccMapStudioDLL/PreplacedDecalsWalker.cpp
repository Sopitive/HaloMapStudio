// PreplacedDecalsWalker.cpp
// =============================================================================
// Native walker for Halo Reach sbsp PREPLACED decals - rewritten to the
// engine-source 3-block schema (ReverseMe/scenario_structure_bsp.json,
// structure_bsp_definitions.cpp). Companion to DecalWalker (which handles the
// OTHER decal block, `Runtime Decals` at sbsp+0x1D0). The two walkers are
// INDEPENDENT - no shared state, no shared output type.
//
// -----------------------------------------------------------------------------
// THE SCHEMA
// -----------------------------------------------------------------------------
// Preplaced decals are THREE coupled sbsp blocks, in this field order (json
// sbsp root ~1168-1182):
//
//   1. `preplaced decal sets*`  -> bsp_preplaced_decal_set_reference_block
//   2. `preplaced decals*`      -> bsp_preplaced_decal_reference_block
//   3. `preplaced decal geometry!*` -> global_render_geometry_struct (inline)
//
// In the MCC loaded-metadata layout a tag_block header is 12 bytes (int32
// count + int32 + uint32 pointer - same stride PickScnr... / OFF_PAGES vs
// OFF_SEGMENTS exhibit: 12-byte gap between consecutive tag_block fields).
// The previous (fabricated) walker read a single "Decals" tag_block @ +0x328
// and a phantom "Decal Properties" tag_block @ +0x334. Those two offsets ARE
// real tag_blocks, but they are:
//
//   sbsp + 0x328  -> SET   block (bsp_preplaced_decal_set_reference_block)
//   sbsp + 0x334  -> REF   block (bsp_preplaced_decal_reference_block)
//   (sbsp + 0x340  -> the inline geometry struct - not walked here)
//
// i.e. +0x328 lands on the SET block (block order is sets-then-decals), and
// +0x334 (the next tag_block header, 12 bytes later) is the REFERENCE block - 
// NOT a "Decal Properties" block (no such block exists in engine source).
//
//   SET block (element size 28 = 0x1C):
//     +0x00  short tagref  Decal             (DIRECT decs/deca datum index - 
//                                              4-byte short tagref in the MCC
//                                              loaded layout; the tag id is the
//                                              low 16 bits. NOT a palette index.)
//     +0x04  8x char  location bsp/cluster pairs (4 pairs; overlap the tagref's
//                                              upper bytes in tag-file order)
//     +0x0C  real_point_3d  center           (WORLD XYZ - the decal position)
//     +0x18  short  first decal ref index    (start into the REF block)
//     +0x1A  short  decal ref count          (count into the REF block)
//
//   STATIC CONFIRMATION (Assembly ReachMCC/sbsp.xml - the SAME
//   authoritative MCC loaded-metadata plugin that proved the runtime-decal
//   offset 0x1D0). The "Decals" tagblock @0x328 (elementSize 0x1C) renders
//   field +0x00 as `<tagref name="Decal" withGroup="false">` - a 4-byte SHORT
//   tagref holding the decs datum index DIRECTLY, then Position point3 @0x0C,
//   `Decal Property Index` int16 @0x18 (= first ref index), Unknown int16 @0x1A
//   (= ref count). The companion "Decal Properties" tagblock @0x334 (the REF
//   block) has point2 @0x0C (spirit corner) + point2 @0x14 (spirit size). This
//   reconciles the engine-source JSON `decal definition index` with the MCC
//   layout: MCC resolves the authored palette index into a direct tag datum at
//   LOAD time, so the loaded field at +0x00 is the decs id itself. We therefore
//   read +0x00 as a DIRECT short tagref (primary), and fall back to scnr-palette
//   index resolution only when the direct read fails class validation.
//
//   REF block (element size 28 = 0x1C):
//     +0x00  short  index start              \  mesh slice into the baked
//     +0x02  short  index count              |  preplaced-decal geometry
//     +0x04  short  vertex start             |  buffer (block 3)
//     +0x06  short  vertex count             /
//     +0x08  short  definition block index
//     +0x0A  2B pad
//     +0x0C  real_point_2d  spirit corner    (INLINE sprite-UV Umin/Vmin)
//     +0x14  real_vector_2d spirit size      (INLINE sprite-UV Usize/Vsize)
//
// The decs tag link is the SET block's `decal definition index`, resolved
// through the scenario decals palette (scnr + 0x368, same palette the runtime
// DecalWalker uses). It is an INDEX, not an inline tagref. If the palette
// fails to resolve we log the raw index and leave the decal unresolved rather
// than crash (per the no-procedural-placeholders rule).
//
// The sprite-UV sub-rect is INLINE on each REF entry (spirit corner/size),
// NOT in a separate properties block.
//
//   decs (decal_system, baseSize 0x3C) - same chain DecalWalker uses
//     Decal System tagblock @ +0x2C  -> Postprocess @ +0x40 -> Textures @ +0x10
//       Textures[0] -> bitm (base map); +0x68 blend mode; +0x70/+0x94 scale.
//
// -----------------------------------------------------------------------------
// Output ABI: flat array of ZH_PreplacedDecal via the public export
// HaloMapStudio_BSP_EnumeratePreplacedDecals. Caller frees with
// HaloMapStudio_BSP_FreePreplacedDecals. The ABI is UNCHANGED (80 bytes) so the
// Rust mirror / .def need no edit - the SET.center world pos, resolved decs/bitm,
// inline spirit UV, blend and scale all map onto the existing fields. The
// former forensic "Unk" int16 echo fields now carry the geometry mesh-slice
// (index/vertex start/count) for logging, and the raw def-index sits in
// PropertyIndex. Per-SET emission is one record (a SET may carry multiple REF
// rows; we use the FIRST REF row's inline UV + slice - the common authored
// case is one ref per set; multi-ref sets log all refs and render the first).
//
// Defensive contract - same shape as DecalWalker:
//   * SEH-wrapped at the public boundary.
//   * Class-code checks at every cross-tag dereference (sbsp / scnr / decs / bitm).
//   * Index validation against cache->tags.size() and tagblock count vs. cache size.
//   * Hard cap on per-bsp count (PREPLACED_INSTANCE_CAP).
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
// Public ABI - UNCHANGED layout (80 bytes). Keep in sync with the Rust mirror in crates/hms-native/src/lib.rs
// PreplacedDecalInterop.ZH_PreplacedDecal.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct ZH_PreplacedDecal {
    float    Position[3];          // world XYZ from SET.center (+0x0C)
    int32_t  DecsTagId;            // resolved decs id (-1 unresolved)
    uint32_t BitmapTagId;          // resolved bitm id (0xFFFFFFFFu unresolved)
    int32_t  PropertyIndex;        // raw SET.decal_definition_index (palette idx)
    int32_t  BlendModeRaw;         // decs.PP[0]+0x68 (-1 unresolved)

    // Inline sprite sub-rect from REF.spirit_corner (+0x0C) / spirit_size
    // (+0x14). Default (0,0,1,1) when degenerate / no ref row.
    float    SpriteUMin;
    float    SpriteVMin;
    float    SpriteUSize;
    float    SpriteVSize;

    // Sapien scale-resolution fields, mirrored from the runtime-decal walker.
    float    ScaleXMul;            // decs.PP[0]+0x94 (1.0 unresolved)
    float    ScaleXDefaultMin;     // decs.PP[0]+0x70 (NaN unresolved)
    float    ScaleXDefaultMax;     // decs.PP[0]+0x74 (NaN unresolved)

    // Geometry mesh-slice from the FIRST REF row (block 2 -> block 3). These
    // formerly held forensic "Unk" int16 echoes; they now surface the baked-
    // geometry slice the engine renders for this decal. Not consumed by the
    // viewer's quad-projection install (which projects at SET.center) - emitted
    // for logging / future baked-geometry rendering.
    int16_t  IndexStart;           // REF +0x00
    int16_t  IndexCount;           // REF +0x02
    int16_t  VertexStart;          // REF +0x04
    int16_t  VertexCount;          // REF +0x06
    int16_t  DecalRefCount;        // SET decal_ref_count (number of REF rows)
    int16_t  Pad0;

    // Full Textures[] roster - Textures[0] is the existing BitmapTagId (above).
    // Textures[1..3] surface additional bitmap-tagref slots authored by Reach's
    // multi-bitmap rmt2 families. Slots that don't hold a bitm tagref are
    // 0xFFFFFFFFu. The install path consults rmt2 param names per slot.
    uint32_t BitmapTagId2;  // Textures[1] (0xFFFFFFFFu if absent / non-bitm)
    uint32_t BitmapTagId3;  // Textures[2]
    uint32_t BitmapTagId4;  // Textures[3]
};
#pragma pack(pop)
static_assert(sizeof(ZH_PreplacedDecal) == 80,
              "ZH_PreplacedDecal layout drift - keep in sync with the Rust mirror in crates/hms-native/src/lib.rs");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr const char* TC_SBSP = "sbsp";
constexpr const char* TC_SCNR = "scnr";
constexpr const char* TC_DECS = "decs";
constexpr const char* TC_DECA = "deca";
constexpr const char* TC_BITM = "bitm";

// Block 1: SET block (bsp_preplaced_decal_set_reference_block) @ sbsp+0x328.
constexpr int  SET_BLOCK_OFFSET              = 0x328;
constexpr int  SET_BLOCK_ELEM_SIZE           = 0x1C;   // 28
constexpr int  SET_OFF_DECAL_REF             = 0x00;   // short tagref (direct decs datum index in MCC loaded layout)
constexpr int  SET_OFF_CENTER                = 0x0C;   // real_point_3d (world)
constexpr int  SET_OFF_FIRST_REF_INDEX       = 0x18;   // short
constexpr int  SET_OFF_REF_COUNT             = 0x1A;   // short

// Block 2: REF block (bsp_preplaced_decal_reference_block) @ sbsp+0x334.
constexpr int  REF_BLOCK_OFFSET              = 0x334;
constexpr int  REF_BLOCK_ELEM_SIZE           = 0x1C;   // 28
constexpr int  REF_OFF_INDEX_START           = 0x00;   // short
constexpr int  REF_OFF_INDEX_COUNT           = 0x02;   // short
constexpr int  REF_OFF_VERTEX_START          = 0x04;   // short
constexpr int  REF_OFF_VERTEX_COUNT          = 0x06;   // short
constexpr int  REF_OFF_DEF_BLOCK_INDEX       = 0x08;   // short
constexpr int  REF_OFF_SPIRIT_CORNER         = 0x0C;   // real_point_2d (Umin,Vmin)
constexpr int  REF_OFF_SPIRIT_SIZE           = 0x14;   // real_vector_2d (Usize,Vsize)

// scnr decals palette - SAME offset the runtime DecalWalker uses (scnr+0x368,
// elementSize 0x10, full 16B tagref with the tag id at +0x0C).
constexpr int  SCNR_DECALS_PALETTE_OFFSET    = 0x368;
constexpr int  DECAL_PALETTE_BLOCK_SIZE      = 0x10;
constexpr int  DECAL_PALETTE_COUNT_SANITY    = 0x1000;

// decs chain offsets - IDENTICAL to DecalWalker's. Duplicated here so a
// refactor of one walker doesn't perturb the other.
constexpr int  DECS_OFF_DECAL_SYSTEM         = 0x2C;
constexpr int  DECS_DECAL_SYSTEM_ELEM_SIZE   = 0x98;
constexpr int  DECS_OFF_POSTPROCESS          = 0x40;
constexpr int  DECS_POSTPROCESS_ELEM_SIZE    = 0xB4;
constexpr int  DECS_OFF_TEXTURES             = 0x10;
constexpr int  DECS_TEXTURES_ELEM_SIZE       = 0x18;
constexpr int  DECS_TEXTURES_BITMAP_OFFSET   = 0x00;
constexpr int  DECS_POSTPROCESS_BLENDMODE    = 0x68;
constexpr int  DECS_POSTPROCESS_SCALE_X_RANGE= 0x70;
constexpr int  DECS_POSTPROCESS_SCALE_X_MUL  = 0x94;

// Sanity caps.
constexpr int  PREPLACED_COUNT_SANITY        = 0x10000;  // 65k
constexpr int  PREPLACED_INSTANCE_CAP        = 16384;    // hard cap per sbsp
constexpr int  REF_COUNT_SANITY              = 0x10000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Full 16B Reclaimer Gen3+ tagref: ClassId @ +0, pad @ +4..11, TagId @ +12.
inline int32_t ReadFullTagRefId(const uint8_t* tagRef) {
    uint32_t rawId = RU32(tagRef + 12);
    if (rawId == 0xFFFFFFFFu) return -1;
    return (int32_t)(rawId & 0xFFFFu);
}

// ---------------------------------------------------------------------------
// scnr decals palette walker - read scnr.Decals Palette[] -> array of decs
// tag ids (with -1 for null/non-decs entries). The SET block's
// `decal definition index` indexes this array. Same semantics as the runtime
// DecalWalker's ReadScnrDecalsPalette (replicated here to keep the walkers
// independent).
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
    if ((size_t)scnrMetaOff + (size_t)SCNR_DECALS_PALETTE_OFFSET + 12 > cache->size) return false;
    const uint8_t* scnrMeta = cache->base + scnrMetaOff;

    TagBlockRef pbk = ReadTagBlock(scnrMeta + SCNR_DECALS_PALETTE_OFFSET);
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
        arr[i] = ReadFullTagRefId(entry);
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

// Resolve decs.DecalSystem[0].Postprocess[0].Textures[].BitmapReference roster.
// Returns Textures[0]'s bitm id (or -1 on failure); fills outBitmapTagId2..4
// with Textures[1..3]'s bitm ids. Scale/blend out-params are sentineled on
// failure. IDENTICAL chain to DecalWalker's resolver.
int32_t ResolveDecsToBitmap(CacheHandle* cache, int32_t decsTagId,
                            int32_t* outBlendMode,
                            float*   outScaleMul,
                            float*   outScaleDefMin,
                            float*   outScaleDefMax,
                            uint32_t* outBitmapTagId2,
                            uint32_t* outBitmapTagId3,
                            uint32_t* outBitmapTagId4)
{
    *outBlendMode   = -1;
    *outScaleMul    = 1.0f;
    *outScaleDefMin = std::numeric_limits<float>::quiet_NaN();
    *outScaleDefMax = std::numeric_limits<float>::quiet_NaN();
    *outBitmapTagId2 = 0xFFFFFFFFu;
    *outBitmapTagId3 = 0xFFFFFFFFu;
    *outBitmapTagId4 = 0xFFFFFFFFu;

    if (decsTagId < 0) return -1;
    if ((uint32_t)decsTagId >= cache->tags.size()) return -1;
    const TagEntry& dte = cache->tags[decsTagId];
    if (dte.classIndex < 0) return -1;
    bool isDecs = memcmp(dte.classCode, TC_DECS, 4) == 0;
    bool isDeca = memcmp(dte.classCode, TC_DECA, 4) == 0;
    if (!isDecs && !isDeca) return -1;

    int64_t metaOff = TagMetaFileOff(cache, dte.metaPointerRaw);
    if (metaOff < 0) return -1;
    if ((size_t)metaOff + 0x3C > cache->size) return -1;
    const uint8_t* decsMeta = cache->base + metaOff;

    TagBlockRef sysBlk = ReadTagBlock(decsMeta + DECS_OFF_DECAL_SYSTEM);
    if (sysBlk.count <= 0 || sysBlk.count > 16) return -1;
    int64_t sysOff = TagMetaFileOff(cache, sysBlk.pointer);
    if (sysOff < 0) return -1;
    if ((size_t)sysOff + DECS_DECAL_SYSTEM_ELEM_SIZE > cache->size) return -1;
    const uint8_t* sysMeta = cache->base + sysOff;

    TagBlockRef ppBlk = ReadTagBlock(sysMeta + DECS_OFF_POSTPROCESS);
    if (ppBlk.count <= 0 || ppBlk.count > 16) return -1;
    int64_t ppOff = TagMetaFileOff(cache, ppBlk.pointer);
    if (ppOff < 0) return -1;
    if ((size_t)ppOff + DECS_POSTPROCESS_ELEM_SIZE > cache->size) return -1;
    const uint8_t* ppMeta = cache->base + ppOff;

    TagBlockRef texBlk = ReadTagBlock(ppMeta + DECS_OFF_TEXTURES);
    if (texBlk.count <= 0 || texBlk.count > 64) return -1;
    int64_t texOff = TagMetaFileOff(cache, texBlk.pointer);
    if (texOff < 0) return -1;
    int rosterCount = texBlk.count < 4 ? texBlk.count : 4;
    if ((size_t)texOff + (size_t)rosterCount * (size_t)DECS_TEXTURES_ELEM_SIZE > cache->size) return -1;
    const uint8_t* texMeta = cache->base + texOff;

    auto readBitmAtSlot = [&](int slotIdx) -> uint32_t {
        if (slotIdx >= texBlk.count) return 0xFFFFFFFFu;
        const uint8_t* entry = texMeta + (size_t)slotIdx * (size_t)DECS_TEXTURES_ELEM_SIZE;
        int32_t id = ReadFullTagRefId(entry + DECS_TEXTURES_BITMAP_OFFSET);
        if (id < 0) return 0xFFFFFFFFu;
        if ((uint32_t)id >= cache->tags.size()) return 0xFFFFFFFFu;
        if (memcmp(cache->tags[id].classCode, TC_BITM, 4) != 0) return 0xFFFFFFFFu;
        return (uint32_t)id;
    };

    uint32_t slot0 = readBitmAtSlot(0);
    if (slot0 == 0xFFFFFFFFu) return -1;
    int32_t bitmapId = (int32_t)slot0;

    *outBitmapTagId2 = readBitmAtSlot(1);
    *outBitmapTagId3 = readBitmAtSlot(2);
    *outBitmapTagId4 = readBitmAtSlot(3);

    int32_t bm = 0;
    memcpy(&bm, ppMeta + DECS_POSTPROCESS_BLENDMODE, 4);
    *outBlendMode = bm;

    float defMin = 0.0f, defMax = 0.0f, mul = 1.0f;
    memcpy(&defMin, ppMeta + DECS_POSTPROCESS_SCALE_X_RANGE + 0, 4);
    memcpy(&defMax, ppMeta + DECS_POSTPROCESS_SCALE_X_RANGE + 4, 4);
    memcpy(&mul,    ppMeta + DECS_POSTPROCESS_SCALE_X_MUL,        4);
    if (!(mul > 0.0f) || mul > 1024.0f) mul = 1.0f;
    *outScaleMul    = mul;
    *outScaleDefMin = defMin;
    *outScaleDefMax = defMax;

    return bitmapId;
}

// ---------------------------------------------------------------------------
// REF block loader - load the bsp_preplaced_decal_reference_block[] into a
// flat array of the fields the SET walk needs:
//   { indexStart, indexCount, vertexStart, vertexCount, defBlockIndex,
//     uMin, vMin, uSize, vSize } per row. Out array length = REF block count.
// Caller frees on exit. Sanitises degenerate spirit-UV to (0,0,1,1).
// ---------------------------------------------------------------------------
struct RefRow {
    int16_t indexStart, indexCount, vertexStart, vertexCount, defBlockIndex;
    float   uMin, vMin, uSize, vSize;
};

bool LoadRefBlock(CacheHandle* cache, const uint8_t* sbspMeta,
                  RefRow** outArr, int* outCount)
{
    *outArr = nullptr;
    *outCount = 0;

    TagBlockRef blk = ReadTagBlock(sbspMeta + REF_BLOCK_OFFSET);
    if (blk.count <= 0) return true;  // empty is valid
    if (blk.count > REF_COUNT_SANITY) return false;

    int64_t arrOff = TagMetaFileOff(cache, blk.pointer);
    if (arrOff < 0) return false;
    if ((size_t)arrOff + (size_t)blk.count * REF_BLOCK_ELEM_SIZE > cache->size) return false;

    RefRow* arr = (RefRow*)malloc(sizeof(RefRow) * (size_t)blk.count);
    if (!arr) return false;
    for (int i = 0; i < blk.count; ++i) {
        const uint8_t* e = cache->base + arrOff + i * REF_BLOCK_ELEM_SIZE;
        RefRow& r = arr[i];
        r.indexStart    = R16(e + REF_OFF_INDEX_START);
        r.indexCount    = R16(e + REF_OFF_INDEX_COUNT);
        r.vertexStart   = R16(e + REF_OFF_VERTEX_START);
        r.vertexCount   = R16(e + REF_OFF_VERTEX_COUNT);
        r.defBlockIndex = R16(e + REF_OFF_DEF_BLOCK_INDEX);
        memcpy(&r.uMin,  e + REF_OFF_SPIRIT_CORNER + 0, 4);
        memcpy(&r.vMin,  e + REF_OFF_SPIRIT_CORNER + 4, 4);
        memcpy(&r.uSize, e + REF_OFF_SPIRIT_SIZE   + 0, 4);
        memcpy(&r.vSize, e + REF_OFF_SPIRIT_SIZE   + 4, 4);
        // Sanitise degenerate / NaN inline UV -> full atlas.
        if (!(r.uMin  == r.uMin))  r.uMin  = 0.0f;
        if (!(r.vMin  == r.vMin))  r.vMin  = 0.0f;
        if (!(r.uSize == r.uSize) || r.uSize <= 0.0f || r.uSize > 32.0f) r.uSize = 1.0f;
        if (!(r.vSize == r.vSize) || r.vSize <= 0.0f || r.vSize > 32.0f) r.vSize = 1.0f;
    }
    *outArr   = arr;
    *outCount = blk.count;
    return true;
}

// ---------------------------------------------------------------------------
// Preplaced decal walker - SET block driven.
// ---------------------------------------------------------------------------
bool EnumerateInner(CacheHandle* cache, uint32_t sbspTagId, uint32_t scnrTagId,
                    ZH_PreplacedDecal** outBuf, uint32_t* outLen,
                    int* diagSetCount, int* diagRefCount, int* diagPaletteCount,
                    int* diagResolved, int* diagUnresolved)
{
    *outBuf = nullptr;
    *outLen = 0;
    *diagSetCount = 0;
    *diagRefCount = 0;
    *diagPaletteCount = 0;
    *diagResolved = 0;
    *diagUnresolved = 0;

    if (sbspTagId >= cache->tags.size()) return false;
    const TagEntry& sbspTe = cache->tags[sbspTagId];
    if (sbspTe.classIndex < 0) return false;
    if (memcmp(sbspTe.classCode, TC_SBSP, 4) != 0) return false;

    int64_t sbspMetaOff = TagMetaFileOff(cache, sbspTe.metaPointerRaw);
    if (sbspMetaOff < 0) return false;
    if ((size_t)sbspMetaOff + (size_t)REF_BLOCK_OFFSET + 12 > cache->size) return false;
    const uint8_t* sbspMeta = cache->base + sbspMetaOff;

    // --- Block 1: SET block @ +0x328 ---
    TagBlockRef sets = ReadTagBlock(sbspMeta + SET_BLOCK_OFFSET);
    if (sets.count <= 0) return true;  // valid empty
    if (sets.count > PREPLACED_COUNT_SANITY) {
        NativeDiag("[PREPLACED] sbsp=0x%X SET count %d insane (>%d)",
                   sbspTagId, sets.count, PREPLACED_COUNT_SANITY);
        return false;
    }
    int64_t setsArrOff = TagMetaFileOff(cache, sets.pointer);
    if (setsArrOff < 0) return false;
    if ((size_t)setsArrOff + (size_t)sets.count * SET_BLOCK_ELEM_SIZE > cache->size) return false;
    *diagSetCount = sets.count;

    // --- Block 2: REF block @ +0x334 ---
    RefRow* refs = nullptr;
    int refCount = 0;
    if (!LoadRefBlock(cache, sbspMeta, &refs, &refCount)) {
        NativeDiag("[PREPLACED] sbsp=0x%X REF block read failed", sbspTagId);
        refs = nullptr;
        refCount = 0;
    }
    *diagRefCount = refCount;

    // --- scnr decals palette (resolves SET.decal_definition_index -> decs) ---
    int32_t* palette = nullptr;
    int32_t  paletteCount = 0;
    if (!ReadScnrDecalsPalette(cache, scnrTagId, &palette, &paletteCount)) {
        NativeDiag("[PREPLACED] sbsp=0x%X scnr=0x%X decals palette read failed",
                   sbspTagId, scnrTagId);
        palette = nullptr;
        paletteCount = 0;
    }
    *diagPaletteCount = paletteCount;

    int capped = sets.count;
    if (capped > PREPLACED_INSTANCE_CAP) capped = PREPLACED_INSTANCE_CAP;

    ZH_PreplacedDecal* buf = (ZH_PreplacedDecal*)malloc(sizeof(ZH_PreplacedDecal) * (size_t)capped);
    if (!buf) {
        if (refs)    free(refs);
        if (palette) free(palette);
        return false;
    }
    memset(buf, 0, sizeof(ZH_PreplacedDecal) * (size_t)capped);

    // Per-decs resolution cache keyed by global tag id.
    struct DecsCacheEntry {
        int32_t  bitmapId;
        int32_t  blendMode;
        float    scaleMul;
        float    scaleDefMin;
        float    scaleDefMax;
        uint32_t bitmapId2;
        uint32_t bitmapId3;
        uint32_t bitmapId4;
        bool     computed;
    };
    DecsCacheEntry* decsCache = (DecsCacheEntry*)calloc(
        cache->tags.size(), sizeof(DecsCacheEntry));

    uint32_t emitted = 0;
    for (int i = 0; i < capped; ++i) {
        const uint8_t* set = cache->base + setsArrOff + i * SET_BLOCK_ELEM_SIZE;

        // +0x00 is a SHORT tagref in the MCC loaded layout (Assembly ReachMCC
        // sbsp.xml: <tagref name="Decal" withGroup="false">). The decs/deca
        // datum index is held DIRECTLY here; the tag id is the low 16 bits.
        // We keep the raw 32-bit read for the palette-index fallback and a
        // 16-bit-masked id for the primary direct-tagref resolve.
        uint32_t decalRefRaw = (uint32_t)R32(set + SET_OFF_DECAL_REF);
        int32_t  defIndex    = (int32_t)decalRefRaw;                 // raw (palette fallback)
        int32_t  directId    = (decalRefRaw == 0xFFFFFFFFu)
                                   ? -1
                                   : (int32_t)(decalRefRaw & 0xFFFFu); // direct decs id
        float cx, cy, cz;
        memcpy(&cx, set + SET_OFF_CENTER + 0, 4);
        memcpy(&cy, set + SET_OFF_CENTER + 4, 4);
        memcpy(&cz, set + SET_OFF_CENTER + 8, 4);
        int16_t firstRef = R16(set + SET_OFF_FIRST_REF_INDEX);
        int16_t refCnt   = R16(set + SET_OFF_REF_COUNT);

        ZH_PreplacedDecal& D = buf[emitted];
        D.Position[0] = cx; D.Position[1] = cy; D.Position[2] = cz;
        D.DecsTagId   = -1;
        D.BitmapTagId = 0xFFFFFFFFu;
        D.PropertyIndex = defIndex;     // raw SET+0x00 (forensic: direct decs datum index)
        D.BlendModeRaw  = -1;
        D.SpriteUMin = 0.0f; D.SpriteVMin = 0.0f;
        D.SpriteUSize = 1.0f; D.SpriteVSize = 1.0f;
        D.ScaleXMul = 1.0f;
        D.ScaleXDefaultMin = std::numeric_limits<float>::quiet_NaN();
        D.ScaleXDefaultMax = std::numeric_limits<float>::quiet_NaN();
        D.IndexStart = 0; D.IndexCount = 0; D.VertexStart = 0; D.VertexCount = 0;
        D.DecalRefCount = refCnt; D.Pad0 = 0;
        D.BitmapTagId2 = 0xFFFFFFFFu;
        D.BitmapTagId3 = 0xFFFFFFFFu;
        D.BitmapTagId4 = 0xFFFFFFFFu;

        // Resolve the SET's +0x00 Decal reference -> decs/deca tag id.
        // PRIMARY (MCC loaded layout, Assembly ReachMCC-confirmed): +0x00 holds
        // the decs datum index DIRECTLY (short tagref). Read the low 16 bits and
        // class-validate. FALLBACK: only if the direct id fails validation, treat
        // the raw value as an index into the scnr decals palette (covers any
        // build/path where the field is still an authored palette index rather
        // than a resolved datum). Never crash; log which path resolved.
        int32_t decsId = -1;
        const char* resoTag = "?";
        if (directId >= 0 && (uint32_t)directId < cache->tags.size()) {
            const TagEntry& dte = cache->tags[directId];
            bool isDecs = (dte.classIndex >= 0) && memcmp(dte.classCode, TC_DECS, 4) == 0;
            bool isDeca = (dte.classIndex >= 0) && memcmp(dte.classCode, TC_DECA, 4) == 0;
            if (isDecs || isDeca) { decsId = directId; resoTag = "direct"; }
        }
        if (decsId < 0 && palette && defIndex >= 0 && defIndex < paletteCount) {
            decsId = palette[defIndex];           // already class-validated
            if (decsId >= 0) resoTag = "palette";
        }

        if (decsId >= 0 && (uint32_t)decsId < cache->tags.size()) {
            D.DecsTagId = decsId;

            int32_t  bmId; int32_t bm;
            float    mul, defMin, defMax;
            uint32_t bm2 = 0xFFFFFFFFu, bm3 = 0xFFFFFFFFu, bm4 = 0xFFFFFFFFu;
            if (decsCache && !decsCache[decsId].computed) {
                bmId = ResolveDecsToBitmap(cache, decsId, &bm, &mul, &defMin, &defMax,
                                           &bm2, &bm3, &bm4);
                decsCache[decsId].bitmapId    = bmId;
                decsCache[decsId].blendMode   = bm;
                decsCache[decsId].scaleMul    = mul;
                decsCache[decsId].scaleDefMin = defMin;
                decsCache[decsId].scaleDefMax = defMax;
                decsCache[decsId].bitmapId2   = bm2;
                decsCache[decsId].bitmapId3   = bm3;
                decsCache[decsId].bitmapId4   = bm4;
                decsCache[decsId].computed    = true;
            }
            if (decsCache) {
                bmId   = decsCache[decsId].bitmapId;
                bm     = decsCache[decsId].blendMode;
                mul    = decsCache[decsId].scaleMul;
                defMin = decsCache[decsId].scaleDefMin;
                defMax = decsCache[decsId].scaleDefMax;
                bm2    = decsCache[decsId].bitmapId2;
                bm3    = decsCache[decsId].bitmapId3;
                bm4    = decsCache[decsId].bitmapId4;
            } else {
                bmId = ResolveDecsToBitmap(cache, decsId, &bm, &mul, &defMin, &defMax,
                                           &bm2, &bm3, &bm4);
            }

            D.BlendModeRaw     = bm;
            D.ScaleXMul        = mul;
            D.ScaleXDefaultMin = defMin;
            D.ScaleXDefaultMax = defMax;
            D.BitmapTagId2     = bm2;
            D.BitmapTagId3     = bm3;
            D.BitmapTagId4     = bm4;
            if (bmId >= 0) {
                D.BitmapTagId = (uint32_t)bmId;
                ++(*diagResolved);
            } else {
                ++(*diagUnresolved);
            }
        } else {
            ++(*diagUnresolved);
        }

        // Apply the FIRST REF row's inline UV + mesh slice for this SET.
        // (A SET may reference [firstRef .. firstRef+refCnt) REF rows; the
        // viewer install projects a single quad at SET.center, so we surface
        // the first row's sub-rect. Multi-ref sets are logged with the full
        // range so the user's launch can reveal whether 1-ref-per-set holds.)
        if (refs && refCount > 0 && firstRef >= 0 && firstRef < refCount) {
            const RefRow& r0 = refs[firstRef];
            D.SpriteUMin  = r0.uMin;
            D.SpriteVMin  = r0.vMin;
            D.SpriteUSize = r0.uSize;
            D.SpriteVSize = r0.vSize;
            D.IndexStart  = r0.indexStart;
            D.IndexCount  = r0.indexCount;
            D.VertexStart = r0.vertexStart;
            D.VertexCount = r0.vertexCount;
            // REF +0x08 `definition block index` selects WHICH baked-geometry mesh
            // (VB/IB) in the preplaced-decal geometry buffer this decal's slice
            // indexes into (the buffer holds one mesh per decal-material group).
            // Surfaced via the formerly-unused Pad0 so the renderer can pick the
            // right mesh before applying VertexStart/IndexStart within it.
            D.Pad0 = r0.defBlockIndex;
        }

        // Gate-on validation logging (NativeDiag is itself gated by
        // MMS_NATIVE_LOG). One line per SET: position, def-index resolution,
        // resolved decs, ref range, and the first ref's inline UV + slice.
        if (i < 256) {
            NativeDiag("[PREPLACED] set=%-4d center=(%.2f,%.2f,%.2f) defIdx=%d(%s) "
                       "->decs=0x%X bitm=0x%X refs=[%d..%d) | ref=%d uv=(%.3f,%.3f,%.3f,%.3f) "
                       "tris=idx[%d+%d] vtx[%d+%d]",
                       i, cx, cy, cz, defIndex, resoTag,
                       (D.DecsTagId < 0 ? 0xFFFFu : (uint32_t)D.DecsTagId),
                       D.BitmapTagId,
                       (int)firstRef, (int)(firstRef + refCnt),
                       (int)firstRef,
                       D.SpriteUMin, D.SpriteVMin, D.SpriteUSize, D.SpriteVSize,
                       (int)D.IndexStart, (int)D.IndexCount,
                       (int)D.VertexStart, (int)D.VertexCount);
        }

        ++emitted;
    }

    *outBuf = buf;
    *outLen = emitted;

    if (decsCache) free(decsCache);
    if (refs)      free(refs);
    if (palette)   free(palette);
    return true;
}

bool EnumerateSeh(CacheHandle* cache, uint32_t sbspTagId, uint32_t scnrTagId,
                  ZH_PreplacedDecal** outBuf, uint32_t* outLen,
                  int* diagSetCount, int* diagRefCount, int* diagPaletteCount,
                  int* diagResolved, int* diagUnresolved)
{
    __try {
        return EnumerateInner(cache, sbspTagId, scnrTagId,
                              outBuf, outLen,
                              diagSetCount, diagRefCount, diagPaletteCount,
                              diagResolved, diagUnresolved);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        NativeDiag("[PREPLACED] SEH fault sbsp=0x%X", sbspTagId);
        if (*outBuf) { free(*outBuf); *outBuf = nullptr; }
        *outLen = 0;
        return false;
    }
}

}  // anonymous namespace

// =============================================================================
// Public exports
// =============================================================================

// Enumerate preplaced decals on the given sbsp via the 3-block schema:
//   SET block (+0x328): world center + decal_definition_index + ref range
//   REF block (+0x334): inline spirit UV + baked-geometry mesh slice
//   scnr decals palette (+0x368): resolves decal_definition_index -> decs id
// One ZH_PreplacedDecal is emitted per SET (using its first REF row's UV/slice).
//
//   outBuffer - malloc'd ZH_PreplacedDecal[*outCount]. Caller frees via
//               HaloMapStudio_BSP_FreePreplacedDecals.
//   outCount - count emitted (may be 0 - valid "no decals").
//
// Returns 1 on success, 0 on hard failure (bad handle, bad sbsp, OOB reads).
extern "C" __declspec(dllexport) int __stdcall HaloMapStudio_BSP_EnumeratePreplacedDecals(
    uint64_t cacheHandle, uint32_t sbspTagId, uint32_t scnrTagId,
    ZH_PreplacedDecal** outBuffer, uint32_t* outCount)
{
    if (outBuffer) *outBuffer = nullptr;
    if (outCount)  *outCount  = 0;
    if (!outBuffer || !outCount) return 0;

    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) {
        NativeDiag("[PREPLACED] bad cacheHandle=0x%llX",
                   (unsigned long long)cacheHandle);
        return 0;
    }

    ZH_PreplacedDecal* buf = nullptr;
    uint32_t           len = 0;
    int dSets = 0, dRefs = 0, dPal = 0, dRes = 0, dUn = 0;
    bool ok = EnumerateSeh(cache, sbspTagId, scnrTagId, &buf, &len,
                           &dSets, &dRefs, &dPal, &dRes, &dUn);

    NativeDiag("[PREPLACED] sbsp=0x%X scnr=0x%X sets=%d refs=%d palette=%d "
               "emitted=%u resolved=%d unresolved=%d",
               sbspTagId, scnrTagId, dSets, dRefs, dPal, len, dRes, dUn);

    if (!ok) {
        if (buf) { free(buf); buf = nullptr; }
        return 0;
    }
    *outBuffer = buf;
    *outCount  = len;
    return 1;
}

extern "C" __declspec(dllexport) void __stdcall HaloMapStudio_BSP_FreePreplacedDecals(
    ZH_PreplacedDecal* buf)
{
    if (buf) free(buf);
}

extern "C" __declspec(dllexport) uint32_t __stdcall HaloMapStudio_BSP_GetPreplacedDecalCount(
    uint64_t cacheHandle, uint32_t sbspTagId)
{
    CacheHandle* cache = LookupHandle(cacheHandle);
    if (!cache) return 0;
    if (sbspTagId >= cache->tags.size()) return 0;
    const TagEntry& te = cache->tags[sbspTagId];
    if (te.classIndex < 0) return 0;
    if (memcmp(te.classCode, TC_SBSP, 4) != 0) return 0;
    int64_t metaOff = TagMetaFileOff(cache, te.metaPointerRaw);
    if (metaOff < 0) return 0;
    if ((size_t)metaOff + (size_t)SET_BLOCK_OFFSET + 12 > cache->size) return 0;
    __try {
        TagBlockRef blk = ReadTagBlock(cache->base + metaOff + SET_BLOCK_OFFSET);
        if (blk.count <= 0) return 0;
        if (blk.count > PREPLACED_COUNT_SANITY) return 0;
        return (uint32_t)blk.count;
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        return 0;
    }
}
