// MapBspParser.h
// =============================================================================
// Self-contained C++ scenario_structure_bsp (sbsp) decoder for Halo MCC
// HaloReach .map files (the tag layouts follow Reclaimer's
// Blam.HaloReach scenario_structure_bsp + scenario_lightmap definitions).
//
// Operates on a cache handle previously opened via ZH_MBP_OpenCache.
// Sharing the handle is intentional - bitmap, model, and bsp parsers walk
// the same tag index and resource gestalt, so re-mapping the file would be
// wasteful.
//
// Output model:
//   * Cluster meshes appear first in the iteration order (rendered at world
//     origin).
//   * Instance meshes (BspGeometryInstanceBlock) follow, with their per-
//     instance transform pre-baked into the position stream.
//
// First-cut format coverage:
//   * Cluster format 0x00 (world) - primary BSP cluster layout.
//   * Format 0x01 (rigid) - same decoder as MapModelParser.
//   * Other formats - return false from the geometry decode
//                                    so the viewer's Reclaimer fallback
//                                    handles them.
//
// Threading model mirrors MapModelParser.
// =============================================================================

#pragma once
#include <cstdint>

extern "C" {

// One BSP cluster's metadata. Cluster geometry is rendered at world origin
// (identity transform); instance geometry has its own transform baked in.
struct ZH_BspMesh {
    uint32_t VertexCount;
    uint32_t IndexCount;
    uint32_t VertexStride;     // packed float3 only (12 bytes) for first cut
    uint32_t IndexStride;      // 2 or 4
    uint32_t VertexFormat;
    uint32_t Flags;             // bit 0 = is_instance, else cluster
    int32_t  MaterialIndex;     // -1 = no material
    float    BoundsMin[3];
    float    BoundsMax[3];
    float    UvMin[2];
    float    UvMax[2];
    // Bake-in transform that's already been applied to vertex positions.
    // Useful for de-baking in the viewer (rare).
    float    AppliedTransform[16];
    // For instance meshes (Flags bit 0 == 1) this is the index into the
    // sbsp's GeometryInstance list - i.e. which instance this mesh
    // represents. Used by the Lbsp lookup to fetch the per-instance
    // lightmap UV2 vertex-buffer index from
    // `Per-Instance Lightmap Texcoords` at tag offset 0xF4 (parallel
    // array, indexed by this ordinal). For cluster meshes (bit 0 == 0)
    // this field is UINT32_MAX (sentinel "not an instance"). Mirrored
    // in the hms-native ZhBspMesh mirror.
    uint32_t InstanceOrdinal;
    uint32_t Reserved1;
    // BAKED_SHADOW_FIX: owning SBSP cluster index for cluster
    // meshes (Flags bit0==0). Lets the viewer's Lbsp lookup index Lbsp.clusters[]
    // by the REAL cluster index instead of the submesh ordinal (clusters emit
    // one mesh per submesh, so the ordinal over-runs the 11-entry block).
    // 0xFFFFFFFF for instance meshes. Mirrored in the hms-native ZhBspMesh mirror.
    uint32_t LightmapClusterIndex;
};

typedef uint64_t ZH_BspHandle;

// Open + decode all clusters + instance permutations of a single
// scenario_structure_bsp. Returns 0 on failure.
__declspec(dllexport) ZH_BspHandle __stdcall ZH_BSP_OpenBsp(
    uint64_t cacheHandle, uint32_t sbspTagId);

__declspec(dllexport) void __stdcall ZH_BSP_CloseBsp(ZH_BspHandle h);

// Mesh enumeration. Cluster meshes appear first, then instances.
__declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMeshCount(ZH_BspHandle h);
__declspec(dllexport) bool     __stdcall ZH_BSP_GetMesh(ZH_BspHandle h, uint32_t i, ZH_BspMesh* outMesh);

// Decode one mesh's vertex+index byte arrays. Same shape as MapModelParser's
// ZH_MMP_DecodeSectionGeometry - packed float3 positions (already
// world-space if instance - bake the per-instance transform during decode),
// uint16 or uint32 indices.
__declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshGeometry(
    ZH_BspHandle h, uint32_t meshIndex,
    uint8_t** outVertexBytes, uint32_t* outVertexBytesLen,
    uint8_t** outIndexBytes,  uint32_t* outIndexBytesLen);

__declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshUVs(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outUvFloat2, uint32_t* outUvFloatCount);

// Per-vertex normals decoded from Int16_N4 at +0x14 in the vertex buffer.
// Output is a malloc'd float3[vc] array (3 floats per vertex). Normals are
// already rotated by the instance transform for instance meshes. Free with
// ZH_BSP_FreeBuffer. Returns false for unsupported formats.
__declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshNormals(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outNormals, uint32_t* outNormalFloatCount);

// Per-vertex tangents decoded from Int16_N4 at +0x1C in the vertex buffer.
// Same shape as normals (float3[vc]). Returns false on decorator meshes
// (which have no tangent data) or on error.
__declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshTangents(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outTangents, uint32_t* outTangentFloatCount);

// Per-vertex binormals computed as cross(normal, tangent) * tangent_w_sign.
// The tangent's 4th component carries the binormal handedness needed for
// correct bump mapping on mirrored UV geometry. Same shape as normals.
__declspec(dllexport) bool __stdcall ZH_BSP_DecodeMeshBinormals(
    ZH_BspHandle h, uint32_t meshIndex,
    float** outBinormals, uint32_t* outBinormalFloatCount);

// Material -> diffuse bitmap tag id resolution. Same pattern as MapModelParser's
// ZH_MMP_GetShaderDiffuseBitmapTagId. Feeds into the bitmap decoder.
__declspec(dllexport) uint32_t __stdcall ZH_BSP_GetMaterialDiffuseBitmapTagId(
    ZH_BspHandle h, int32_t materialIndex);

// Material -> blend mode index 0..5 (or 0xFF on failure). See
// MapModelParser.h::ZH_MMP_GetShaderBlendMode for the mapping.
__declspec(dllexport) uint8_t __stdcall ZH_BSP_GetMaterialBlendMode(
    ZH_BspHandle h, int32_t materialIndex);

// MAT-1: Material -> material_model enum (0..9), 0xFF on failure. See
// MapModelParser.h::ZH_MMP_GetShaderMaterialModel for the mapping.
__declspec(dllexport) uint8_t __stdcall ZH_BSP_GetMaterialModel(
    ZH_BspHandle h, int32_t materialIndex);

// LIT-SI-3: Material -> self_illumination MODE enum (0..12), 0xFF on failure.
// 0 off / 1 simple / 2 three_channel / 3 plasma / 4 from_albedo / 5 detail /
// 6 meter / 7 times_diffuse / 8 simple_with_alpha_mask / 9 multilayer /
// 10 palette / 11 change_color / 12 change_color_detail.
__declspec(dllexport) uint8_t __stdcall ZH_BSP_GetSelfIllumMode(
    ZH_BspHandle h, int32_t materialIndex);

// Reach `rmtr` (terrain) 4-layer blend descriptor. Each base_map_m_<n> field
// is a bitm tag id (or 0xFFFFFFFFu if that slot didn't resolve). BlendMap is
// the RGBA mask used to weight the 4 layers. IsTerrainBlend = 1 iff the
// shader tag class is `rmtr` and at least one layer + the blend map resolved.
// Caller falls back to the existing single-bitmap path when IsTerrainBlend=0.
//
// Bumpmap layers (`bump_map_m_<n>`) added to give the viewer
// proper rocky relief instead of flat-shaded gray surfaces. Detail maps
// (`detail_map_m_<n>`) + per-layer detail tile factors + global_albedo_tint
// added so the CPU composite multiplies each base layer by its
// per-layer detail texture before the 4-way blend mask combine. Without
// detail layers the terrain reads as low-frequency mush; the engine
// expects (base * detail * 2) per layer, then blend, then * tint.
//
// The struct is 364 bytes (the TERRAIN_BLEND_FIX fields - blend xform +
// per-layer .zw offsets + authored active mask - are appended at the end):
// the Rust `ZhTerrainLayers` mirror MUST stay in lockstep - there's no
// version negotiation, both sides are memcpy'd by the FFI layer using
// the field declaration order.
//
// Bump tile factors (`BumpTile_M_<n>_X/Y`) and per-layer detail-bump fields
// (`DetailBumpMap_M_<n>` + `DetailBumpTile_M_<n>_X/Y`) let the viewer bind
// each layer's normal map at its own tile (Reach normal-map convention:
// GREEN near-1 = upward, near-0 = sloped).
#pragma pack(push, 1)
struct ZH_TerrainLayers {
    uint32_t BaseMap_M_0;
    uint32_t BaseMap_M_1;
    uint32_t BaseMap_M_2;
    uint32_t BaseMap_M_3;
    uint32_t BlendMap;
    uint32_t IsTerrainBlend;
    uint32_t BumpMap_M_0;
    uint32_t BumpMap_M_1;
    uint32_t BumpMap_M_2;
    uint32_t BumpMap_M_3;
    // Detail maps (one per base layer). 0xFFFFFFFFu when the rmt2 has no
    // detail_map_m_<n> usage or it failed to resolve to a bitm.
    uint32_t DetailMap_M_0;
    uint32_t DetailMap_M_1;
    uint32_t DetailMap_M_2;
    uint32_t DetailMap_M_3;
    // Per-layer detail tile factors, read from rmt2.Float Constants[argIdx]
    // where argIdx is the position of "detail_map_m_<n>" in rmt2.Arguments[].
    // Defaults to (1.0, 1.0) when the slot wasn't found - caller can detect
    // "no detail" by DetailMap_M_<n> == 0xFFFFFFFFu first.
    float    DetailTile_M_0_X, DetailTile_M_0_Y;
    float    DetailTile_M_1_X, DetailTile_M_1_Y;
    float    DetailTile_M_2_X, DetailTile_M_2_Y;
    float    DetailTile_M_3_X, DetailTile_M_3_Y;
    // Per-layer base tile factors (for computing detail/base ratio at
    // composite time - detail UV tiles relative to base tile space).
    float    BaseTile_M_0_X, BaseTile_M_0_Y;
    float    BaseTile_M_1_X, BaseTile_M_1_Y;
    float    BaseTile_M_2_X, BaseTile_M_2_Y;
    float    BaseTile_M_3_X, BaseTile_M_3_Y;
    // global_albedo_tint Vec4 (R, G, B, A). Defaults to (1,1,1,1) when
    // the shader doesn't carry one (engine no-op multiply).
    float    GlobalAlbedoTint[4];
    // Per-layer bump tile factors (rmt2.Float Constants[bump_map_m_N]).
    // Defaults to (1,1) when missing. Bump UV ratio is bumpTile/baseTile.
    float    BumpTile_M_0_X, BumpTile_M_0_Y;
    float    BumpTile_M_1_X, BumpTile_M_1_Y;
    float    BumpTile_M_2_X, BumpTile_M_2_Y;
    float    BumpTile_M_3_X, BumpTile_M_3_Y;
    // Detail-bump layers (one per base layer). 0xFFFFFFFFu when the rmt2
    // has no detail_bump_m_<n> usage or it failed to resolve to a bitm.
    uint32_t DetailBumpMap_M_0;
    uint32_t DetailBumpMap_M_1;
    uint32_t DetailBumpMap_M_2;
    uint32_t DetailBumpMap_M_3;
    // Per-layer detail-bump tile factors (rmt2.Float Constants[detail_bump_m_N]).
    float    DetailBumpTile_M_0_X, DetailBumpTile_M_0_Y;
    float    DetailBumpTile_M_1_X, DetailBumpTile_M_1_Y;
    float    DetailBumpTile_M_2_X, DetailBumpTile_M_2_Y;
    float    DetailBumpTile_M_3_X, DetailBumpTile_M_3_Y;
    // TERRAIN_BLEND_FIX: APPENDED at struct END (memcpy-by-order,
    // Pack=1 - the Rust mirror MUST append the same fields
    // in the same order). All default to engine identity so an older field-set
    // degrades to today's exact behaviour.
    //
    // Fix 1: blend_map_xform (transform_texcoord, texture_xform.hlsl_include:26).
    // The engine samples the weight mask at uv*xy + zw (terrain_new:162); MMS
    // sampled at bare uv. (sx, sy, ox, oy). Default (1,1,0,0) = identity.
    float    BlendXform_X, BlendXform_Y, BlendXform_Z, BlendXform_W;
    // Fix 3: per-layer base/detail/bump/detail-bump UV TRANSLATION (.zw of each
    // map's xform). The engine's transform_texcoord adds .zw; MMS dropped it
    // (scale-only). Default (0,0) = engine identity offset.
    float    BaseOffset_M_0_X, BaseOffset_M_0_Y;
    float    BaseOffset_M_1_X, BaseOffset_M_1_Y;
    float    BaseOffset_M_2_X, BaseOffset_M_2_Y;
    float    BaseOffset_M_3_X, BaseOffset_M_3_Y;
    float    DetailOffset_M_0_X, DetailOffset_M_0_Y;
    float    DetailOffset_M_1_X, DetailOffset_M_1_Y;
    float    DetailOffset_M_2_X, DetailOffset_M_2_Y;
    float    DetailOffset_M_3_X, DetailOffset_M_3_Y;
    float    BumpOffset_M_0_X, BumpOffset_M_0_Y;
    float    BumpOffset_M_1_X, BumpOffset_M_1_Y;
    float    BumpOffset_M_2_X, BumpOffset_M_2_Y;
    float    BumpOffset_M_3_X, BumpOffset_M_3_Y;
    float    DetailBumpOffset_M_0_X, DetailBumpOffset_M_0_Y;
    float    DetailBumpOffset_M_1_X, DetailBumpOffset_M_1_Y;
    float    DetailBumpOffset_M_2_X, DetailBumpOffset_M_2_Y;
    float    DetailBumpOffset_M_3_X, DetailBumpOffset_M_3_Y;
    // Fix 2: bake-authored material_N_type active set (bit n = layer n authored
    // diffuse_only|diffuse_plus_specular). 0 = unresolved -> caller falls back to
    // the bitmap-derived mask (today's behaviour). Read via the rmsh ShaderOptions
    // -> rmdf categories walk (same as ResolveShaderBlendMode).
    uint32_t AuthoredActiveMask;
    // MAT-15: distance_blend_base far-distance base->target colour lerp
    // (terrain_new.hlsl_include:102-129). APPENDED at struct END (Pack=1). All
    // default to engine identity (BlendType 0xFFFFFFFF = unresolved => morph => no
    // lerp; slope/offset/max 0 => base_blend 0 => no lerp) so a shorter DLL degrades
    // to today's behaviour. BlendType: 0 morph, 1 distance_blend_base, 0xFFFFFFFF
    // unresolved. Engine: base_blend = saturate(dist*BlendSlope + BlendOffset);
    // amount_N = min(base_blend, BlendMax[N]); base = lerp(base, BlendTarget[N], amount_N).
    uint32_t BlendType;
    float    BlendSlope;
    float    BlendOffset;
    float    BlendTarget[4][4];   // per-layer target colour (rgba)
    float    BlendMax[4];         // per-layer max blend amount (scalar)
};
#pragma pack(pop)

__declspec(dllexport) bool __stdcall ZH_BSP_GetMaterialTerrainLayers(
    ZH_BspHandle h, int32_t materialIndex, ZH_TerrainLayers* outLayers);

// Per-material diffuse-slot UV tiling (TileX, TileY). Reach BSP textures are
// authored to TILE - the rmt2 carries a per-argument tiling table that the
// engine multiplies by vertex UVs at sample time. Without this scale walls/
// floors render at native UV space and look stretched.
//
// Walks rmsh -> rmt2.Arguments[] to find the index of the picked diffuse
// usage, then reads ShaderProperties[0].TilingData[argIdx].XY. Returns true
// on success. On any failure outTileX / outTileY are set to 1.0 (engine no-op
// scale) and false is returned - caller can skip multiplying when both are
// 1.0. Mirrors Reclaimer.Blam.Common.Gen3.Gen3MaterialHelper's
// `tilingData[arguments.IndexOf(usage)]` resolution.
__declspec(dllexport) bool __stdcall ZH_BSP_GetMaterialDiffuseTiling(
    ZH_BspHandle h, int32_t materialIndex,
    float* outTileX, float* outTileY);

// =============================================================================
// Shader-constants surface - RealConstants (Vector4 per slot) + ScalarConstants
// (single float per slot, bit-cast from Reach's Integer Constants block). Both
// blocks live inside rmsh.ShaderProperties[0]; the rmt2's Arguments[] list
// names what each slot means for that template (e.g. "wave_height",
// "specular_power", "fresnel_coefficient", "ripple_normal_a").
//
// Reach layout (ShaderProperties block, total 172 bytes - see
// AssemblySource/Plugins/Reach/rmsh.xml "Postprocess" tagblock):
//   0x00 (16B): Template TagReference
//   0x10 (12B): ShaderMaps tagblock
//   0x1C (12B): "Float Constants" tagblock (16B/elem) -> RealConstants
//                NOTE: physically the SAME block Reclaimer calls "TilingData".
//                Reach packs both UV-tile data (for texture slots) and tint /
//                wave-height / specular vectors (for non-texture slots) into
//                ONE Vector4 array. The rmt2.Arguments[i] string is what
//                distinguishes one role from the other.
//   0x28 (12B): "Integer Constants" tagblock (4B/elem) -> ScalarConstants
//                Reach has no separate float-scalar block - the engine writes
//                shader scalars into this int32 slot. We surface them as
//                bit-cast floats for caller inspection.
//   0x34 (4B):  Boolean Constants (flags32) - not surfaced.
//
// ArgNames[] is rmt2.Arguments[] resolved to strings; index = argument slot
// (the index into RealConstants). Each name is null-padded to 32 bytes;
// absent names are zero-filled. A shader's "wave_height" slot is at
// argIdx = position of "wave_height" in ArgNames -> read RealConstants[argIdx*4..].
//
// Confidence: RealConstants offset/stride is HIGH (matches the existing
// OFF_TILING_DATA_IN_PROPS=28 used by the working tiling resolver - same
// physical block). ScalarConstants offset (40) is MEDIUM - derived from
// Assembly's rmsh.xml; the int->float bit-cast is the questionable bit (Reach
// stores scalars as int32 but the engine consumes them as floats in shader
// register set 1). Diag log surfaces both raw-int and as-float so the user
// can verify against real water-shader values.
// =============================================================================
#pragma pack(push, 1)
struct ZH_ShaderConstants {
    uint32_t RealCount;                 // count of vec4 entries actually populated
    uint32_t ScalarCount;               // count of scalar entries actually populated
    float    RealConstants[64 * 4];     // [argIdx*4 + (x|y|z|w)], up to 64 vec4s
    float    ScalarConstants[64];       // up to 64 scalars (bit-cast int32 in Reach)
    // ARG_NAME_WIDEN: 48-byte slot (was 32). 31 usable chars
    // truncated the longest Reach render_method arg names (e.g.
    // environment_map_specular_contribution, 37 chars) - see MapBspParser.cpp
    // SHADER_CONSTS_NAME_LEN. Kept in lockstep with crates/hms-native/src/lib.rs and
    // the BspMeshDiskCache version bump.
    char     ArgNames[64 * 48];         // null-padded usage name per slot, indexed by argIdx
    uint32_t Reserved[8];
};
#pragma pack(pop)

__declspec(dllexport) bool __stdcall ZH_BSP_GetMaterialShaderConstants(
    ZH_BspHandle h, int32_t materialIndex, ZH_ShaderConstants* outConsts);

// DIRECT-VB pipeline (engine-style): hand back the .map's PREBUILT, compressed
// vertex/index bytes VERBATIM (pointers into the resource mmap - do NOT free) plus
// the per-section de-quantization constants, so the viewer can upload them straight
// to the GPU and decompress in the vertex shader (position = raw*Scale+Offset), exactly
// like deform_flat_rigid - instead of the CPU decode+re-interleave in DecodeMeshGeometry.
// PURE ADDITIVE: returns existing data by reference, mutates nothing. See
// reference_hms_direct_vb_pipeline_re.md.
#pragma pack(push, 1)
struct ZH_RawGeom {
    const uint8_t* VbPtr;      // section VB, verbatim (into resource mmap; do NOT free)
    uint32_t       VbLen;
    const uint8_t* IbPtr;      // section IB, verbatim (nullptr if unindexed)
    uint32_t       IbLen;
    uint32_t       VertexFormat; // VFMT_* (0x00 world, 0x01 rigid, 0x02 skinned, 0x04/05/06 flat, 0x0F decorator)
    uint32_t       VertexStride; // StrideForFormat: 0x24 world/rigid, 0x2C skinned, 0x20 decorator
    uint32_t       SectionVertexCount; // total verts in the section VB
    uint32_t       IndexStride;  // 2 or 4
    uint32_t       IsUnindexed;
    uint32_t       IsInstance;
    // This mesh = ONE submesh of the section: draw IbPtr[IndexStart .. IndexStart+IndexCount).
    uint32_t       IndexStart;   // in indices (not bytes)
    uint32_t       IndexCount;
    // Per-section de-quant constants (Position_Compression_Scale=(Max-Min), Offset=Min).
    // WORLD/FLAT_WORLD ignore these (positions already world-space float32).
    float          PosMin[3];
    float          PosMax[3];
    float          UvMin[2];     // rigid/skinned UInt16-N2 de-quant; world uses Float16 (ignored)
    float          UvMax[2];
    // Instance transform (identity for cluster meshes) - apply to positions/normals in-shader.
    float          Transform[16];
    float          UniformScale;
    uint32_t       LightmapClusterIndex; // for the PVL / lightmap-texture lookup
    uint32_t       InstanceOrdinal;
};
#pragma pack(pop)
__declspec(dllexport) bool __stdcall ZH_BSP_GetRawGeometry(
    ZH_BspHandle h, uint32_t meshIndex, ZH_RawGeom* outGeom);

__declspec(dllexport) void __stdcall ZH_BSP_FreeBuffer(uint8_t* buf);

// Scenario-level convenience: enumerate all loaded sbsp tag ids from a scnr.
// The viewer can use this to find what BSPs the engine has loaded without
// having to read scnr itself. Returns count; pass null buffer to query size.
__declspec(dllexport) uint32_t __stdcall ZH_BSP_EnumerateSbspsInScenario(
    uint64_t cacheHandle, uint32_t scnrTagId,
    uint32_t* outSbspTagIds, uint32_t maxCount);

}  // extern "C"
