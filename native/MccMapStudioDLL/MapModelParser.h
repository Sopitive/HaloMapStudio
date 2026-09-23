// MapModelParser.h
// =============================================================================
// Self-contained C++ render_model (mode tag) decoder for Halo MCC HaloReach
// .map files (tag layouts follow Reclaimer's Blam.HaloReach.render_model
// definitions).
//
// Operates on a cache handle previously opened via ZH_MBP_OpenCache.
// Sharing the handle is intentional - the bitmap and model parsers walk the
// same tag index and resource gestalt, so re-mapping the file would be wasteful.
//
// First-cut format coverage:
//   * Format 0x01 (rigid) - full vertex + UV decode, position de-quantized
//                             from section bounds. The hot path for forge maps.
//   * Other formats - return false from the geometry decode so the
//                             viewer's existing Reclaimer fallback handles them.
//
// Threading model mirrors MapBitmapParser:
//   * One model handle per (cache, mode tag) pair.
//   * The shared CacheHandle owns the mmap; the model handle just holds
//     decoded section / submesh / shader-mapping metadata.
//   * Decode is lock-free once the model handle is created.
// =============================================================================

#pragma once
#include <cstdint>

extern "C" {

// One section's metadata - caller iterates these. Vertex/index byte counts
// are post-decompression, ready for upload to GPU.
struct ZH_ModelSection {
    uint32_t VertexCount;
    uint32_t IndexCount;
    uint32_t VertexStride;     // post-decode (always 12 for first cut: float3 pos)
    uint32_t IndexStride;      // 2 or 4
    uint32_t VertexFormat;     // engine-format byte (rigid=01, world=00, etc.)
    uint32_t Flags;
    int16_t  NodeIndex;        // for rigid-single-node sections; -1 = unset
    // Format class (was Reserved0):
    //   0 = rigid / world / decorator / flat-rigid / flat-world (vertex data
    //       is in mesh-local frame, expects the engine's per-mesh 90 degZ bake
    //       in the viewer to match the live forward/up axes from the engine).
    //   1 = skinned / flat-skinned (vertex data is in MODEL frame after the
    //       bind-pose bones have been applied; the per-mesh 90 degZ would
    //       double-rotate the mesh and is omitted by the viewer).
    int16_t  FormatClass;
    float    BoundsMin[3];
    float    BoundsMax[3];
    float    UvMin[2];
    float    UvMax[2];
    uint32_t SubmeshCount;
    int32_t  MaterialIndex;    // -1 = no material

    // Per-section bone transform - composition of the bone[NodeIndex] local
    // transform up the parent chain to the model root. Reclaimer doesn't bake
    // this at parse time; it stores BoneIndex on the mesh and lets the
    // renderer apply the bone hierarchy at draw time. Our viewer doesn't carry
    // a bone hierarchy for static rigid forge meshes, so we surface the
    // composed transform here and the viewer applies it as a
    // per-segment GeometryModel3D.Transform.
    //
    // Default for sections with NodeIndex == 0xFF (no bone): translation is
    // (0,0,0), rotation is the identity quaternion (0,0,0,1).
    float    NodeTranslation[3];
    float    NodeRotation[4];   // x, y, z, w (unit quaternion)

    uint32_t Reserved1[2];
};

// One submesh inside a section (index range + shader index).
// Flags is the per-submesh `SubmeshFlags` ushort from the engine (offset 18 in
// the 24-byte SubmeshBlock per Reclaimer). Bit 8 (0x100) = IsTransparent - 
// caller uses this to switch alpha-blend on cloud-layer sky panels.
struct ZH_ModelSubmesh {
    uint32_t SectionIndex;
    int32_t  ShaderIndex;
    uint32_t IndexStart;
    uint32_t IndexLength;
    uint32_t Flags;
    uint32_t Reserved;
};

// Public handle obtained from ZH_MMP_OpenModel, freed by ZH_MMP_CloseModel.
typedef uint64_t ZH_ModelHandle;

// Open + decode a render_model tag from a previously-opened cache handle.
// Returns 0 on failure. The caller owns the handle and must close it.
__declspec(dllexport) ZH_ModelHandle __stdcall ZH_MMP_OpenModel(
    uint64_t cacheHandle, uint32_t modeTagId);

__declspec(dllexport) void __stdcall ZH_MMP_CloseModel(ZH_ModelHandle h);

// Section-level queries.
__declspec(dllexport) uint32_t __stdcall ZH_MMP_GetSectionCount(ZH_ModelHandle h);
__declspec(dllexport) bool     __stdcall ZH_MMP_GetSection(ZH_ModelHandle h, uint32_t i, ZH_ModelSection* outSec);

// Returns 1 iff the section is part of permutation 0 of any Region - used
// by the viewer to render ONE armor variant for player bipeds rather
// than every variant overlapping. For models without a Regions block all
// sections are reported as allowed.
__declspec(dllexport) uint8_t __stdcall ZH_MMP_IsSectionAllowed(
    ZH_ModelHandle h, uint32_t sectionIndex);

// Decode a section's vertex buffer to packed float3 positions (12 bytes each)
// + uint16/uint32 indices, both already de-quantized to world space (positions
// pre-multiplied by section bounds). Caller must free with FreeBuffer.
//
// Returns false on unsupported vertex formats - caller must handle (e.g. by
// falling back to Reclaimer's path during migration).
__declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionGeometry(
    ZH_ModelHandle h, uint32_t sectionIndex,
    uint8_t** outVertexBytes, uint32_t* outVertexBytesLen,
    uint8_t** outIndexBytes,  uint32_t* outIndexBytesLen);

// Optional: surface UV channel separately so the viewer can build textured
// MeshGeometry3D without re-walking the vertex stream. Returns null if the
// section's format has no UVs.
__declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionUVs(
    ZH_ModelHandle h, uint32_t sectionIndex,
    float** outUvFloat2, uint32_t* outUvFloatCount);

// Per-vertex normals (float3[VertexCount]). Decorator fmt = Float32_3 @ +0x14;
// rigid/world/skinned = Int16_N4 @ +0x14 (/32767). Caller frees with FreeBuffer.
__declspec(dllexport) bool __stdcall ZH_MMP_DecodeSectionNormals(
    ZH_ModelHandle h, uint32_t sectionIndex,
    float** outNrmFloat3, uint32_t* outNrmFloatCount);

// Submesh + material lookups so the viewer can build per-segment materials.
__declspec(dllexport) uint32_t __stdcall ZH_MMP_GetSubmeshCount(ZH_ModelHandle h);
__declspec(dllexport) bool     __stdcall ZH_MMP_GetSubmesh(ZH_ModelHandle h, uint32_t i, ZH_ModelSubmesh* outSm);

// Per-shader -> diffuse bitmap tag id resolution. The viewer feeds this id
// into ZH_MBP_DecodeBitmap to get the texture pixels.
// Returns 0xFFFFFFFFu if no shader / no diffuse / unable to resolve.
__declspec(dllexport) uint32_t __stdcall ZH_MMP_GetShaderDiffuseBitmapTagId(
    ZH_ModelHandle h, int32_t shaderIndex);

// Per-shader -> blend mode index 0..5. Walks rmsh.RenderMethodDefinitionRef
// (rmdf), finds the "blend_mode" Category, indexes into rmsh.ShaderOptions[]
// at the same ordinal to get the option index, then resolves the option's
// name via the cache string table (e.g. "alpha_blend", "additive", "opaque").
//
// Mapping (matches Sapien's rasterizer_set_blend_mode_register at
// sapien.exe+0xB40500):
//   0 = opaque
//   1 = additive
//   2 = multiply
//   3 = double_multiply / maximum
//   4 = alpha_blend / pre_multiplied_alpha
//   5 = add_src_times_srcalpha / add_src_times_dstalpha
//   0xFF = unresolved (caller should treat as opaque default).
__declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderBlendMode(
    ZH_ModelHandle h, int32_t shaderIndex);

// MAT-1: per-shader material_model enum (0..9) chosen by the rmsh `material_model`
// rmdf category. 0=diffuse_only 1=cook_torrance 2=two_lobe_phong 3=foliage 4=none
// 5=glass 6=organism 7=single_lobe_phong 8=hair 9=custom_specular. 0xFF = unresolved
// (caller maps to the default cook_torrance path). Sibling of ZH_MMP_GetShaderBlendMode.
__declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderMaterialModel(
    ZH_ModelHandle h, int32_t shaderIndex);

// LIT-SI-3: per-shader self_illumination MODE enum (0..12) chosen by the rmsh
// `self_illumination` rmdf category. 0 off / 1 simple / 2 three_channel / 3 plasma /
// 4 from_albedo / 5 detail / 6 meter / 7 times_diffuse / 8 simple_with_alpha_mask /
// 9 multilayer / 10 palette / 11 change_color / 12 change_color_detail. 0xFF =
// unresolved (caller maps to the default simple single-composite path). Model-side
// mirror of ZH_BSP_GetSelfIllumMode. Sibling of ZH_MMP_GetShaderMaterialModel.
__declspec(dllexport) uint8_t __stdcall ZH_MMP_GetShaderSelfIllumMode(
    ZH_ModelHandle h, int32_t shaderIndex);

// Per-shader RealConstants (Vector4 per slot) + ScalarConstants (single float
// per slot, bit-cast from Reach's Integer Constants block). Walks the same
// rmsh -> rmt2 chain as ZH_MMP_GetShaderDiffuseBitmapTagId. See
// MapBspParser.h::ZH_BSP_GetMaterialShaderConstants for the layout and the
// confidence rating; this is the model-side mirror of that export so the
// non-BSP shader chain (render_models / forge palette / sky / etc.) gets
// access to the same shader-instance values.
__declspec(dllexport) bool __stdcall ZH_MMP_GetShaderConstants(
    ZH_ModelHandle h, int32_t shaderIndex, struct ZH_ShaderConstants* outConsts);

// Free any buffer returned by Decode* above.
__declspec(dllexport) void __stdcall ZH_MMP_FreeBuffer(uint8_t* buf);

}  // extern "C"
