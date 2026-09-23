//! hms-native — FFI to the C++ map-parser library (`HaloMapStudioDLL.dll` /
//! `libhalomapstudio.so`, source under `native/`).
//!
//! The library is loaded with `libloading` and its `ZH_*` C exports are
//! resolved into function pointers: cache / BSP / render-model / bitmap decode
//! plus the scenario walkers (collision, physics, sky, fog, lights, decals,
//! lightmaps, terrain layers, shader constants).
//!
//! ABI notes: handles are u64, scalars u32/i32/f32, out-params `*mut T`, and
//! boolean returns are taken as u8/i32 and interpreted `!= 0`. Structs are
//! `#[repr(C)]` mirrors of the `ZH_*` structs — keep them byte-exact with the
//! C++ headers (MapBspParser.h / MapModelParser.h / MapBitmapParser.h).

use std::ffi::{c_char, c_void};
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use libloading::Library;

// ---------------------------------------------------------------------------
// #[repr(C)] struct mirrors.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhBitmapInfo {
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub mip_count: u32,
    pub submap_count: u32,
    pub flags: u32,
    pub depth: u32,
    /// Submap bitmapType byte: 0=2D, 1=3D/volume, 2=CUBEMAP (6 faces), 3=array.
    /// A cube's `depth` is 1 while it physically stores 6 faces — the authored
    /// environment_map cube path keys off `bitmap_type == 2`, not `depth`.
    pub bitmap_type: u32,
    /// Submap byte 17: bitmap gamma curve. Reach enum: 0=unknown/xRGB, 1=linear(raw),
    /// 2=sRGB, 3=gamma2(pow2.0), 4=offset_log. The engine samples through the texture
    /// FORMAT this curve selects (sRGB → auto decode; linear/offset_log → raw). Used to
    /// decode terrain base/detail per-curve instead of a blanket pow(2.2) (which
    /// over-darkens linear-authored detail/utility maps like cliff_detail).
    pub curve: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhBspMesh {
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_stride: u32,
    pub index_stride: u32,
    pub vertex_format: u32,
    pub flags: u32,
    pub material_index: i32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// 4x4 transform already applied to positions (row-major).
    pub matrix: [f32; 16],
    pub instance_ordinal: u32,
    pub reserved1: u32,
    /// Owning SBSP cluster index for cluster meshes (0xFFFFFFFF for instances).
    /// Keys the per-vertex VMF lightprobe lookup (ZH_LBSP_GetClusterPvlVb).
    /// MUST be present — the native struct is 144B; omitting it overflows the
    /// out-param on every GetMesh call.
    pub lightmap_cluster_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhModelSection {
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_stride: u32,
    pub index_stride: u32,
    pub vertex_format: u32,
    pub flags: u32,
    pub node_index: i16,
    pub format_class: i16,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub submesh_count: u32,
    pub material_index: i32,
    pub node_translation: [f32; 3],
    pub node_rotation: [f32; 4],
    pub reserved1a: u32,
    pub reserved1b: u32,
}

/// One submesh (material segment) of a render model — mirrors ZH_ModelSubmesh.
/// A section's index range is subdivided into submeshes, each with its OWN shader
/// index → its own diffuse/blend. Multi-material objects (e.g. the coliseum wall)
/// need per-submesh texturing; using shader 0 for the whole model shows one
/// (wrong) material on every part.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhModelSubmesh {
    pub section_index: u32,
    pub shader_index: i32,
    pub index_start: u32,
    pub index_length: u32,
}

// ---------------------------------------------------------------------------
// Export function-pointer types.
// ---------------------------------------------------------------------------
// ZH_MBP_OpenCache takes a wide path (const wchar_t*), NOT narrow. wchar_t is
// 16-bit on Windows (UTF-16) and 32-bit on Linux (UTF-32), so the element type
// follows the target rather than being hard-coded.
#[cfg(windows)]
pub type WideChar = u16;
#[cfg(not(windows))]
pub type WideChar = u32;
type FnOpenCache = unsafe extern "C" fn(*const WideChar) -> u64;
type FnU64 = unsafe extern "C" fn(u64);
type FnFindBitmap = unsafe extern "C" fn(u64, u32, u32, *mut ZhBitmapInfo) -> i32;
type FnDecodeBitmap =
    unsafe extern "C" fn(u64, u32, u32, u32, *mut *mut u8, *mut u32, *mut u32) -> i32;
// ZH_MBP_DecodeBitmapSlice(cache, tag, submap, mip, sliceIndex, out ptr, out w, out h) -> bool.
// Decodes ONE z-slice of a volume submap (the dual-VMF SDM lightprobe_hdr_color has 3).
type FnDecodeBitmapSlice =
    unsafe extern "C" fn(u64, u32, u32, u32, u32, *mut *mut u8, *mut u32, *mut u32) -> i32;
// ZH_MBP_DecodeBitmapFace(cache, tag, submap, mip, faceIndex, strideMode, out ptr, out w, out h)
// -> bool. Decodes ONE face of a cubemap submap (bitmapType==2 → 6 faces). strideMode:
// 0=face-major (DDS/D3D layout, correct for Reach), 1=mip-major.
type FnDecodeBitmapFace =
    unsafe extern "C" fn(u64, u32, u32, u32, u32, u32, *mut *mut u8, *mut u32, *mut u32) -> i32;
type FnFree = unsafe extern "C" fn(*mut u8);
// raw DDS (DXT10 header + BC blocks + map's own mips) for direct GPU upload —
// skips the CPU BCn-decode + CPU mip-regen that dominate load time.
type FnGetRawDds = unsafe extern "C" fn(u64, u32, u32, *mut *mut u8, *mut u32) -> i32;
type FnOpenBsp = unsafe extern "C" fn(u64, u32) -> u64;
type FnU64ToU32 = unsafe extern "C" fn(u64) -> u32;
type FnU64ToU64 = unsafe extern "C" fn(u64) -> u64;
type FnGetBspMesh = unsafe extern "C" fn(u64, u32, *mut ZhBspMesh) -> i32;
type FnDecodeGeom =
    unsafe extern "C" fn(u64, u32, *mut *mut u8, *mut u32, *mut *mut u8, *mut u32) -> i32;
type FnDecodeVec = unsafe extern "C" fn(u64, u32, *mut *mut u8, *mut u32) -> i32;
// DIRECT-VB: ZH_BSP_GetRawGeometry(bsp, meshIndex, out ZhRawGeom*) -> bool. Hands back the
// .map's prebuilt compressed VB/IB verbatim (pointers into the resource mmap) + de-quant
// constants, for engine-style verbatim GPU upload + in-shader decompress.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ZhRawGeom {
    pub vb_ptr: *const u8,
    pub vb_len: u32,
    pub ib_ptr: *const u8,
    pub ib_len: u32,
    pub vertex_format: u32,
    pub vertex_stride: u32,
    pub section_vertex_count: u32,
    pub index_stride: u32,
    pub is_unindexed: u32,
    pub is_instance: u32,
    pub index_start: u32,
    pub index_count: u32,
    pub pos_min: [f32; 3],
    pub pos_max: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub transform: [f32; 16],
    pub uniform_scale: f32,
    pub lightmap_cluster_index: u32,
    pub instance_ordinal: u32,
}
type FnGetRawGeom = unsafe extern "C" fn(u64, u32, *mut ZhRawGeom) -> i32;

/// Owned copy of a mesh's raw (compressed) vertex/index bytes + de-quant constants.
#[derive(Clone)]
pub struct RawGeomOwned {
    pub vb: Vec<u8>,
    pub ib: Vec<u8>,
    pub vertex_format: u32,
    pub vertex_stride: u32,
    pub section_vertex_count: u32,
    pub index_stride: u32,
    pub index_start: u32,
    pub index_count: u32,
    pub is_instance: bool,
    pub pos_min: [f32; 3],
    pub pos_max: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub transform: [f32; 16],
    pub uniform_scale: f32,
    pub lightmap_cluster_index: u32,
}
// ZH_BSP_EnumerateSbspsInScenario(cache, scnrTagId, out, max) -> count.
type FnEnumSbsps = unsafe extern "C" fn(u64, u32, *mut u32, u32) -> u32;
// ZH_TAG_EnumerateByGroup(cache, groupBytes, out, max) -> total.
type FnEnumByGroup = unsafe extern "C" fn(u64, u32, *mut u32, u32) -> u32;
// ZH_SCNR_EnumerateTriggerVolumes(cache, scnr, out[100B each], max) -> count.
type FnEnumTriggers = unsafe extern "C" fn(u64, u32, *mut u8, u32) -> u32;
// ZH_TAG_ResolveCollTagId(cache, primaryTag) -> collTag.
type FnResolveTag = unsafe extern "C" fn(u64, u32) -> u32;
type FnReadMeta = unsafe extern "C" fn(u64, u32, u32, u32, *mut u8) -> i32;
type FnReadPtr = unsafe extern "C" fn(u64, u32, u32, *mut u8) -> i32;
type FnTagOverlayCount = unsafe extern "C" fn(u64, i32) -> i32;
type FnTagOverlayAt = unsafe extern "C" fn(u64, i32, i32, *mut u8) -> u8;
// ZH_COLL_DecodeGeometry(cache, coll, out verts*, out vcount, out idx*, out icount) -> bool.
type FnCollDecode = unsafe extern "C" fn(u64, u32, *mut *mut f32, *mut u32, *mut *mut u32, *mut u32) -> u8;

// ZH_SDDT_EnumerateSoftCeilings(cache, scnr, out[64B each], max) -> count;
// ZH_SDDT_EnumerateSoftCeilingTriangles(cache, scnr, out f32[9 per tri], maxTris) -> count.
type FnEnumSoftCeilings = unsafe extern "C" fn(u64, u32, *mut u8, u32) -> u32;
type FnEnumSoftCeilingTris = unsafe extern "C" fn(u64, u32, *mut f32, u32) -> u32;

/// One structure-design soft ceiling (the invisible planes bounding the playable
/// volume). `tris` are world-space triangles (v0, v1, v2).
#[derive(Clone, Default, Debug)]
pub struct SoftCeiling {
    pub name: String,
    /// 0 = acceleration, 1 = soft kill, 2 = slip surface (engine `soft_ceiling_type_enum`).
    pub kind: u32,
    /// scnr metadata flags: 1 ignore bipeds, 2 ignore vehicles, 4 ignore camera, 8 ignore huge vehicles.
    pub flags: u32,
    pub tris: Vec<[[f32; 3]; 3]>,
}

impl SoftCeiling {
    pub fn kind_name(&self) -> &'static str {
        soft_ceiling_kind_name(self.kind)
    }
}

/// Engine name of a soft-ceiling type value.
pub fn soft_ceiling_kind_name(kind: u32) -> &'static str {
    match kind {
        0 => "acceleration",
        1 => "soft kill",
        2 => "slip surface",
        _ => "unknown",
    }
}

/// One scenario trigger volume (kill/safe/plain), oriented box.
#[derive(Clone, Default)]
pub struct TriggerVolume {
    pub category: u32, // 1=kill, 2=safe, 0=plain
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    pub ext: [f32; 3],
}
type FnOpenModel = unsafe extern "C" fn(u64, u32) -> u64;
type FnGetSection = unsafe extern "C" fn(u64, u32, *mut ZhModelSection) -> i32;
type FnGetSubmesh = unsafe extern "C" fn(u64, u32, *mut ZhModelSubmesh) -> i32;
type FnTagByShader = unsafe extern "C" fn(u64, i32) -> u32;
type FnShaderTemplate = unsafe extern "C" fn(u64, i32) -> i32;
// ZH_SKY_GetSkyCount(cache, scnrTag) -> count; ZH_SKY_GetSkyTagId /
// ZH_SKY_GetRenderModelTagId(cache, scnrTag, skyIndex) -> tag id.
type FnSkyCount = unsafe extern "C" fn(u64, u32) -> u32;
type FnSkyTag = unsafe extern "C" fn(u64, u32, u32) -> u32;
// ZH_TAG_GetName(cache, tagId, buf, bufLen) -> chars written.
type FnTagName = unsafe extern "C" fn(u64, u32, *mut c_char, i32) -> i32;
// ZH_BSP_GetMaterialBitmapByUsage(bsp, materialIndex, usageName) -> bitmap tag.
type FnBitmapByUsage = unsafe extern "C" fn(u64, i32, *const c_char) -> u32;
// ZH_MMP_GetShaderConstants(model, shaderIndex, out ZhShaderConstants) -> bool.
type FnShaderConstants = unsafe extern "C" fn(u64, i32, *mut ZhShaderConstants) -> u8;
// ZH_BSP_GetMaterialDiffuseTiling(bsp, matIdx, out f32 tileX, out f32 tileY) -> bool.
type FnBspTiling = unsafe extern "C" fn(u64, i32, *mut f32, *mut f32) -> u8;
// ZH_BSP_GetMaterialTerrainLayers(bsp, matIdx, out ZH_TerrainLayers[5416B]) -> bool.
type FnTerrainLayers = unsafe extern "C" fn(u64, i32, *mut u8) -> u8;
// ZH_BSP_GetMaterialOverlays(bsp, matIdx, out ZH_MaterialOverlay[60B]*, cap) -> count.
type FnMaterialOverlays = unsafe extern "C" fn(u64, i32, *mut u8, i32) -> i32;
// ZH_MMP_GetShaderOverlayCount(model, shader) -> count.
type FnMmpOverlayCount = unsafe extern "C" fn(u64, i32) -> i32;
// ZH_MMP_GetShaderOverlayAt(model, shader, idx, out ZH_ShaderOverlay[148]) -> bool.
type FnMmpOverlayAt = unsafe extern "C" fn(u64, i32, i32, *mut u8) -> u8;
// ZH_BSP_EnumerateDecals(cache, sbsp, scnr, out buf, out count) -> 1 on success.
type FnEnumDecals = unsafe extern "C" fn(u64, u32, u32, *mut *mut u8, *mut u32) -> i32;
// HaloMapStudio_BSP_GetPreplacedDecalCount(cache, sbsp) -> count. __stdcall on x64 == C ABI.
type FnPreplacedCount = unsafe extern "C" fn(u64, u32) -> u32;
// HaloMapStudio_BSP_EnumeratePreplacedDecals(cache, sbsp, scnr, out ZH_PreplacedDecal[80]**, out count) -> 1.
type FnEnumPreplaced = unsafe extern "C" fn(u64, u32, u32, *mut *mut u8, *mut u32) -> i32;
// HaloMapStudio_BSP_DecodePreplacedGeometry(cache, sbsp, out verts**(5f/vertex), out vertFloatCount,
// out indices**(u32), out indexCount, out vertexCount) -> 1. The baked, world-space, artist-conformed
// preplaced-decal frost/ice mesh — sliced per decal by the REF ranges from EnumeratePreplacedDecals.
// Out-params: verts**, vertFloatCount, indices**, indexCount, vertexCount, meshVertBase**, meshIdxBase**, meshCount.
type FnDecodePpGeom = unsafe extern "C" fn(
    u64, u32,
    *mut *mut f32, *mut u32, *mut *mut u32, *mut u32, *mut u32,
    *mut *mut u32, *mut *mut u32, *mut u32,
) -> i32;
// HaloMapStudio_BSP_FreePreplacedGeometry(verts, indices, meshVertBase, meshIdxBase).
type FnFreePpGeom = unsafe extern "C" fn(*mut f32, *mut u32, *mut u32, *mut u32);
// ZH_MMP_IsSectionAllowed(model, sectionIndex) -> 1 if the section belongs to permutation 0 of
// its region (the default, non-damaged variant). Vehicle-permutation filter.
type FnMmpSectionAllowed = unsafe extern "C" fn(u64, u32) -> u8;
// ZH_MMP_EnumerateAttachments(cache, objTagId, out[], maxOut) -> count. Turret/attachments.
type FnEnumAttachments = unsafe extern "C" fn(u64, u32, *mut ZhAttachment, i32) -> i32;
// ZH_MMP_GetVariantSectionMask(cache, objTagId, variantSid, outMask[], maxMask) -> nSections
// written (0 = variant not found / default). Vehicle body-variant permutation swap.
type FnVariantSectionMask = unsafe extern "C" fn(u64, u32, u32, *mut u8, i32) -> i32;
// ZH_MMP_GetDefaultVariantSectionMask(cache, objTag, outMask[], max) -> nSections.
type FnDefaultVariantSectionMask = unsafe extern "C" fn(u64, u32, *mut u8, i32) -> i32;

/// one hlmt Variants→Objects attachment (turret etc.), with the child render_model resolved
/// and its MODEL-SPACE transform already composed (parent node chain ∘ marker). Mirrors the C++
/// `ZH_Attachment` (#pragma pack(1), 48B).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug)]
pub struct ZhAttachment {
    pub child_mode: u32,
    pub child_obj: u32,
    pub marker_sid: u32,
    pub variant_name_sid: u32,
    pub variant_index: i32,
    pub pos: [f32; 3],
    pub rot: [f32; 4], // quaternion x,y,z,w
    pub scale: f32,
}

/// Native `ZH_DecalInstance` total size (Pack=1) — the array stride.
const DECAL_INSTANCE_SIZE: usize = 620;

/// One decoded runtime decal: an oriented, textured quad projected on a surface.
/// Corners = pos ± u_axis*half.x ± v_axis*half.y.
#[derive(Clone, Default)]
pub struct DecalInstance {
    pub pos: [f32; 3],
    pub facing: [f32; 3],
    pub u_axis: [f32; 3],
    pub v_axis: [f32; 3],
    pub half: [f32; 2],
    pub bitmap: u32,
    // Widened per the decal RE (the flat-quad reader dropped all of these → decals
    // rendered totally wrong). Offsets into the 620B ZH_DecalInstance.
    pub blend_mode: i32,          // @68  0=opaque,1=add,2=multiply,4=alpha,...
    pub scale_x_default: [f32; 2], // @292/296 (used when half.x ~0)
    pub scale_y_default: [f32; 2], // @300/304
    pub depth_bias: f32,          // @316
    pub scale_x_mul: f32,         // @320 (X-only multiplier, sys+0x94)
    pub has_sprite: bool,         // @324
    pub sprite: [f32; 4],         // @328 UMin,VMin,USize,VSize (atlas sub-rect)
    pub sort_layer: u8,           // @356 0/1 pre,2 normal,3 post
    pub clamp_angle: f32,         // @308 decal clamp cone (deg) — surface-fit tolerance
    pub cull_angle: f32,          // @312 decal cull cone (deg) — reject fits beyond this
    /// Per-decal tint (albedo_color/tint_color) RGB from the rmt2 FloatConstants, selected
    /// by matching the rmt2 param name. None → untinted (white). The shader multiplies the
    /// base bitmap by this, so a white-swatch decal (yellow "45", coloured signage) gets its
    /// colour ONLY from here — omitting it renders those decals colourless.
    pub tint: Option<[f32; 3]>,
    /// SDF "vector" decal (numbers/text/glyphs). base_map is a flat swatch; the glyph
    /// silhouette is a signed-distance field in `vector_map` (bitmapTagId2 @344). Detected by
    /// the rmt2 params `vector_sharpness`/`antialias_tweak`.
    pub is_vector: bool,
    pub vector_map: u32,
}
/// One decoded PREPLACED decal — the baked, artist-painted surface-coating decals
/// (`sbsp.Decals`, distinct from runtime scenario decals). These carry a world position and
/// a slice into the sbsp's baked preplaced-decal geometry buffer (the already-conformed mesh
/// that wraps walls/floors continuously — the "whole wall covered" ice). Mirrors native
/// `ZH_PreplacedDecal` (80 bytes, Pack=1).
#[derive(Clone, Default, Debug)]
pub struct PreplacedDecal {
    pub pos: [f32; 3],
    pub decs: i32,             // resolved decs (decal_system) tag id, -1 unresolved
    pub bitmap: u32,           // Textures[0] base map, 0xFFFFFFFF unresolved
    pub property_index: i32,   // raw palette index
    pub blend_mode: i32,       // -1 unresolved
    pub sprite: [f32; 4],      // Umin,Vmin,Usize,Vsize (inline sprite sub-rect)
    pub scale_x_mul: f32,
    pub scale_x_default: [f32; 2],
    pub index_start: i16,      // slice into the baked preplaced-decal geometry buffer
    pub index_count: i16,
    pub vertex_start: i16,
    pub vertex_count: i16,
    pub ref_count: i16,
    pub def_block_index: i16,  // REF +0x08: selects which baked-geometry mesh (VB/IB) this slice indexes
    pub bitmap2: u32,          // Textures[1] = alpha/shape mask (like DecalInstance.vector_map)
}

/// The decoded preplaced-decal baked geometry buffer for one sbsp. Holds every VB/IB mesh
/// concatenated into flat arrays; `PreplacedDecal.def_block_index` selects a mesh and the REF
/// slice (vertex_start/count, index_start/count) addresses within it. Positions are already
/// WORLD-space & artist-conformed (no projection needed downstream).
#[derive(Clone, Default, Debug)]
pub struct PreplacedGeometry {
    /// 5 floats per vertex: pos.x, pos.y, pos.z, u, v (all meshes concatenated).
    pub verts: Vec<f32>,
    /// RAW (mesh-local) index values (all meshes concatenated).
    pub indices: Vec<u32>,
    /// Total vertex count (verts.len()/5).
    pub vertex_count: u32,
    /// Per-mesh first-vertex offset into `verts` (in vertices).
    pub mesh_vert_base: Vec<u32>,
    /// Per-mesh first-index offset into `indices`.
    pub mesh_idx_base: Vec<u32>,
}
type FnVoid = unsafe extern "C" fn();

/// Native `ZH_TerrainLayers` total size (Pack=1). Mirrored as a byte buffer;
/// the fields we consume are read by offset (see `bsp_terrain_layers`).
const TERRAIN_LAYERS_SIZE: usize = 5416;

/// The full terrain-blend material data the viewer renders: up to 4 base + 4
/// detail colour maps blended by an RGBA blend mask, with per-layer base AND
/// detail UV transforms (scale in `_tile`, translation in `_offset` — the engine
/// `transform_texcoord(uv) = uv*xy + zw`), the blend mask's own UV transform, the
/// global albedo tint, and the bake-authored active-layer mask. Read by byte
/// offset from the native `ZH_TerrainLayers` (Pack=1, 364B); every field defaults
/// to engine identity so an older DLL that memcpy's a shorter struct (leaving the
/// appended fields zeroed) degrades to scale-only behaviour instead of breaking.
#[derive(Clone, Default)]
pub struct TerrainLayers {
    pub base: [u32; 4],       // 4 base bitmap tags (0xFFFFFFFF = unused)
    pub detail: [u32; 4],     // 4 detail bitmap tags (0xFFFFFFFF = unused)
    pub blend: u32,           // RGBA blend-weight bitmap tag
    pub is_terrain: bool,
    pub base_tile: [[f32; 2]; 4],   // base_map_m_N_xform.xy (scale)
    pub base_offset: [[f32; 2]; 4], // base_map_m_N_xform.zw (translation)
    pub detail_tile: [[f32; 2]; 4], // detail_map_m_N_xform.xy
    pub detail_offset: [[f32; 2]; 4], // detail_map_m_N_xform.zw
    pub blend_xform: [f32; 4],      // blend_map_xform (xy scale, zw offset)
    pub global_tint: [f32; 4],      // global_albedo_tint (rgb; a unused)
    pub active_mask: u32,           // bit n = layer n authored live (0 = unresolved)
    pub bump: [u32; 4],             // bump_map_m_N tags (@24) — terrain normal maps
    pub bump_tile: [[f32; 2]; 4],   // bump_map_m_N_xform.xy (@136 scale)
    pub bump_offset: [[f32; 2]; 4], // bump_map_m_N_xform.zw (@296 translation)
    // MAT-14: per-layer detail_bump_m_N (a SECOND normal map added unweighted to bump
    // when ACTIVE_MATERIAL_COUNT<4 — terrain_new.hlsl_include:84,214-216). Tags @168,
    // tile @184, offset @328. 0xFFFFFFFF when absent (engine samples a flat default).
    pub detail_bump: [u32; 4],
    pub detail_bump_tile: [[f32; 2]; 4],
    pub detail_bump_offset: [[f32; 2]; 4],
    // MAT-15: distance_blend_base far-distance base->target colour lerp.
    // blend_type: 0 morph (no-op), 1 distance_blend_base, u32::MAX unresolved (=morph).
    pub blend_type: u32,            // @364
    pub blend_slope: f32,           // @368
    pub blend_offset: f32,          // @372
    pub blend_target: [[f32; 4]; 4],// @376 per-layer target colour rgba (4×16B)
    pub blend_max: [f32; 4],        // @440 per-layer max blend amount
}

/// Mirrors native `ZH_ShaderConstants` (4392B). Named render-method args with
/// their vec4 / scalar values — used to read authored self_illum_color +
/// self_illum_intensity so the self-illum shader is faithful, not a proxy.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZhShaderConstants {
    pub real_count: u32,
    pub scalar_count: u32,
    pub real_constants: [f32; 64 * 4],
    pub scalar_constants: [f32; 64],
    pub arg_names: [u8; 64 * 48],
    pub reserved: [u32; 8],
}

impl ZhShaderConstants {
    fn zeroed() -> Self {
        // Big POD; zero-init without requiring Default on the large arrays.
        unsafe { std::mem::zeroed() }
    }
    /// Slot index whose arg name matches `name` (null-padded, 48B per slot).
    fn slot_of(&self, name: &str) -> Option<usize> {
        let n = self.real_count.max(self.scalar_count).min(64) as usize;
        for i in 0..n {
            let base = i * 48;
            let raw = &self.arg_names[base..base + 48];
            let end = raw.iter().position(|&b| b == 0).unwrap_or(48);
            if std::str::from_utf8(&raw[..end]).map(|s| s == name).unwrap_or(false) {
                return Some(i);
            }
        }
        None
    }
    /// The vec4 authored for a named arg (e.g. "self_illum_color").
    pub fn real(&self, name: &str) -> Option<[f32; 4]> {
        let i = self.slot_of(name)?;
        let o = i * 4;
        Some([
            self.real_constants[o],
            self.real_constants[o + 1],
            self.real_constants[o + 2],
            self.real_constants[o + 3],
        ])
    }
    /// The scalar authored for a named arg (e.g. "self_illum_intensity").
    pub fn scalar(&self, name: &str) -> Option<f32> {
        let i = self.slot_of(name)?;
        self.scalar_constants.get(i).copied()
    }
    /// Debug: (index, name, real vec4, scalar) for every populated arg slot. Used to discover
    /// which authored param names the DLL actually exposes for a material.
    pub fn dump_args(&self) -> Vec<(usize, String, [f32; 4], f32)> {
        let n = self.real_count.max(self.scalar_count).min(64) as usize;
        let mut out = Vec::new();
        for i in 0..n {
            let base = i * 48;
            let raw = &self.arg_names[base..base + 48];
            let end = raw.iter().position(|&b| b == 0).unwrap_or(48);
            let name = String::from_utf8_lossy(&raw[..end]).into_owned();
            let o = i * 4;
            out.push((i, name,
                [self.real_constants[o], self.real_constants[o+1], self.real_constants[o+2], self.real_constants[o+3]],
                self.scalar_constants.get(i).copied().unwrap_or(0.0)));
        }
        out
    }
}
/// Mirrors native `ZH_FogParams` (#pragma pack(1)) — the scenario's atmospheric
/// fog (`fogg` tag) two-band params. Read via ZH_FOG_GetParams.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ZhFogParams {
    pub inscatter_a: [f32; 3],
    pub inscatter_b: [f32; 3],
    pub sky_tint: [f32; 3],
    pub density: f32,
    pub start_distance: f32,
    pub has_fog: u32,
    pub falloff_end: f32,
    pub _pad: u32,
    pub sky_fog_height: f32,
    pub sky_fog_base_height: f32,
    pub ground_fog_thickness: f32,
    pub ground_fog_color: [f32; 3],
    pub ground_fog_height: f32,
    pub ground_fog_base_height: f32,
    pub ground_fog_max_distance: f32,
    pub fog_light_color: [f32; 3],
    pub fog_light_angular_falloff: f32,
    pub fog_light_distance_falloff: f32,
    pub fog_light_nearby_cutoff: f32,
    pub has_height_band: u32,
    pub has_ground_band: u32,
    pub has_fog_light: u32,
    pub _pad2: u32,
    // World direction TO the fog light + disc shape (from fogg pitch/yaw/angular_radius).
    pub fog_light_dir: [f32; 3],
    pub fog_light_radius_scale: f32,
    pub fog_light_radius_offset: f32,
}
// ZH_FOG_GetParams(cache, scnrTag, out ZhFogParams) -> bool.
type FnFogParams = unsafe extern "C" fn(u64, u32, *mut ZhFogParams) -> u8;

/// Mirrors native `ZH_ChangeColors` (#pragma pack(1)). The 11 predefined engine
/// team/player "change colors" (LINEAR RGB, 0..1) in Color-enum order:
/// 0 red,1 blue,2 green,3 orange,4 purple,5 gold,6 brown,7 pink,8 neutral,9 black,
/// 10 zombie. Entries 8..10 are engine-synthetic (not authored in the tag) and
/// come back as fixed defaults. Read via `ZH_GLOBALS_GetChangeColors`.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ZhChangeColors {
    pub rgb: [[f32; 3]; 11],
    pub count: u32,
    pub found: u32,
}
// ZH_GLOBALS_GetChangeColors(cache, out ZhChangeColors) -> bool.
type FnChangeColors = unsafe extern "C" fn(u64, *mut ZhChangeColors) -> u8;

/// Per-scenario exposure + colour-grade tints (ZH_SCNR_GetExposure). Mirrors the
/// C++ `ZH_ScenarioExposure` (pack 1). g_exposure = pow(2, Stops). Extends it with
/// the cfxs camera_fx auto-exposure adaptation band (log2/stops space).
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct ZhScenarioExposure {
    pub stops: f32,
    pub scripted_offset: f32,
    pub sky_tint: [f32; 3],
    pub ambient_tint: [f32; 3],
    pub sun_tint: [f32; 3],
    pub has_sceg: u32,
    pub has_exposure: u32,
    // cfxs auto-exposure adaptation band (0 fields when has_camera_fx==0).
    pub has_camera_fx: u32,
    pub auto_enabled: u32,
    pub manual_exposure: f32,
    pub auto_min_ev: f32,
    pub auto_max_ev: f32,
    pub auto_target: f32,
    /// Per-map bloom, authored in the same cfxs tag (value fields at +0x2C/+0x3C/+0x4C).
    pub bloom_point: f32,
    pub bloom_inherent: f32,
    pub bloom_intensity: f32,
    pub has_bloom: u32,
}
type FnScnrExposure = unsafe extern "C" fn(u64, u32, *mut ZhScenarioExposure) -> u8;

// ZH_LBSP_GetAirprobeGrid(cache, sbsp, out buf, out count) -> 1 on success.
type FnAirprobeGrid = unsafe extern "C" fn(u64, u32, *mut *mut u8, *mut u32) -> i32;
// ZH_LBSP_GetBrightness(cache, sbsp) -> float scene-brightness scalar.
type FnBrightness = unsafe extern "C" fn(u64, u32) -> f32;
// ZH_BSP_GetRuntimeDecoratorGeometry(cache, sbsp, out meshes, out count) -> 1 ok.
type FnDecoGeom = unsafe extern "C" fn(u64, u32, *mut *mut u8, *mut u32) -> i32;
// ZH_BSP_FreeRuntimeDecoratorMeshes(meshes, count).
type FnFreeDeco = unsafe extern "C" fn(*mut u8, u32);
// ZH_SCNR_EnumerateObjects(cache, scnr, out **buf, out *count) -> 1 ok.
type FnScnrObjects = unsafe extern "C" fn(u64, u32, *mut *mut u8, *mut u32) -> i32;
// ZH_SCNR_FreeObjectBuffer(buf).
type FnFreeScnrObjects = unsafe extern "C" fn(*mut u8);
// ZH_SCNR_EnumerateSimpleLights(cache, scnr, out buf[>=8], out *count, out *seen, out
// *skipped) -> 1 ok. Caller-allocated fixed buffer (cap 8); NO free. x64 → extern "C".
type FnSimpleLights =
    unsafe extern "C" fn(u64, u32, *mut ZhSimpleLight, *mut u32, *mut u32, *mut u32) -> i32;
// ZH_PFOG_Enumerate(cache, scnr, out **buf, out *count) -> bool. Heap buffer; free via
// ZH_PFOG_FreeBuffer(buf). Struct ZH_PlanarFogVolume = 92B.
type FnPfog = unsafe extern "C" fn(u64, u32, *mut *mut ZhPlanarFogVolume, *mut u32) -> u8;
type FnFreePfog = unsafe extern "C" fn(*mut ZhPlanarFogVolume);
// ZH_SCNR_GetSunLight(cache, scnr, out ZhSunLight) -> bool. Resolves the sun ligh +
// its lens tag id.
type FnSunLight = unsafe extern "C" fn(u64, u32, *mut ZhSunLight) -> u8;
type FnSkyLight = unsafe extern "C" fn(u64, u32, *mut ZhSkyLight) -> u8;
// ZH_LENS_GetElements(cache, lensTag, out buf[maxElems], maxElems, out *count,
// out *seen, out ZhLensFlareInfo) -> bool. Caller-allocated buffer.
type FnLensElements = unsafe extern "C" fn(
    u64, u32, *mut ZhLensFlareElement, i32, *mut u32, *mut u32, *mut ZhLensFlareInfo,
) -> u8;
// ZH_LBSP_GetClusterPvlVb(cache, sbsp, clusterIndex, out ZhLbspClusterPvlVb) -> bool.
type FnClusterPvl = unsafe extern "C" fn(u64, u32, u32, *mut ZhLbspClusterPvlVb) -> u8;
// ZH_LBSP_GetInstanceProbe(cache, sbsp, ordinal, out rgb[3]) -> true if single-probe tier.
type FnInstanceProbe = unsafe extern "C" fn(u64, u32, u32, *mut f32) -> u8;

/// Mirrors native `ZH_LbspClusterPvlVb` (Pack=1, 40B). `bytes` is a malloc'd
/// stride-4 per-vertex-lightprobe buffer freed via ZH_LBSP_FreeLightprobeBuffer.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ZhLbspClusterPvlVb {
    found: u32,
    bytes: *mut u8,
    len: u32,
    elem_count: i32,
    hdr_scale_raw: i32,
    hdr_scale: f32,
    pvb_index: i32,
    pvb_offset: i32,
    vb_index: i32,
}

// ---- Per-texel lightmap atlas ----
// ZH_LBSP_GetLightprobeAtlas(cache, sbsp, out ZhLbspAtlas) -> bool.
type FnLbspAtlas = unsafe extern "C" fn(u64, u32, *mut ZhLbspAtlas) -> u8;
// ZH_LBSP_GetClusterEntry(cache, sbsp, clusterIndex, out ZhLbspClusterEntry) -> bool.
type FnClusterEntry = unsafe extern "C" fn(u64, u32, u32, *mut ZhLbspClusterEntry) -> u8;
// ZH_BSP_DecodeMeshUV2(bsp, meshIndex, out **f32, out count) -> bool.
type FnBspUv2 = unsafe extern "C" fn(u64, u32, *mut *mut f32, *mut u32) -> u8;

/// Mirrors native `ZH_LbspLightprobeAtlas` (Pack=1, 40B). Per-BSP DM/SDM atlas
/// tag ids + scene Brightness for the per-pixel baked lightmap path.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct ZhLbspAtlas {
    pub dm_tag: u32,
    pub sdm_tag: u32,
    pub brightness: f32,
    pub resource_id: i32,
    pub lbsp_tag: u32,
    pub has_brightness: u32,
    _reserved: [u32; 4],
}

/// Mirrors native `ZH_LbspClusterEntry` (Pack=1, 44B). Per-cluster atlas submap +
/// the per-vertex-vs-per-pixel selector (`pervertex_block_index`: >=0 has PVL, -1 per-pixel).
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct ZhLbspClusterEntry {
    pub dm_tag: u32,
    pub sdm_tag: u32,
    pub submap_dm: u32,
    pub submap_sdm: u32,
    pub pervertex_block_index: i32,
    pub pervertex_block_offset: i32,
    pub uv_scale_u: f32,
    pub uv_scale_v: f32,
    pub uv_bias_u: f32,
    pub uv_bias_v: f32,
    pub hdr_scale: f32,
}

/// Mirrors native `ZH_LbspInstanceAtlasSubmap` (Pack=1). Per-INSTANCE atlas submap
/// result: the DM/SDM pool tag ids + the submap index this instance's lightmap lives in
/// (Lbsp.Meshes[].InstanceBuckets[] → definition_index). Used to light BSP instance
/// geometry (cluster == 0xFFFFFFFF) from the per-texel atlas when instance PVL fetch fails.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct ZhLbspInstanceAtlas {
    pub submap_dm: u32,
    pub submap_sdm: u32,
    pub pool_dm: u32,
    pub pool_sdm: u32,
    pub mesh_index: u16,
    pub bucket_ordinal: u16,
    _reserved: [u32; 3],
}

// ZH_LBSP_GetInstanceAtlasSubmap(cache, sbsp, instanceOrdinal, out ZhLbspInstanceAtlas) -> bool.
type FnInstanceAtlas = unsafe extern "C" fn(u64, u32, u32, *mut ZhLbspInstanceAtlas) -> u8;

/// Decoded per-cluster per-vertex lightprobe (PVL) stream + block params.
#[derive(Clone, Default)]
pub struct ClusterPvl {
    pub bytes: Vec<u8>, // raw stride-4 VMF-lobe buffer
    pub pvb_offset: i32,
    pub hdr_scale: f32,
}

/// One decoded runtime decorator mesh: pre-baked WORLD-space geometry (grass /
/// foliage), ready to render with an identity transform. `colors` is the
/// per-vertex baked ambient (airprobe SH sample) when the DLL produced it.
#[derive(Clone, Default)]
pub struct DecoratorMesh {
    pub positions: Vec<f32>,   // float3 * vertex_count
    pub uvs: Vec<f32>,         // float2 * vertex_count
    pub normals: Vec<f32>,     // float3 * vertex_count
    pub colors: Option<Vec<f32>>, // float3 * vertex_count (baked ambient)
    pub sway: Option<Vec<f32>>,   // float3 * vertex_count (world-space sway basis)
    pub indices: Vec<u16>,
    pub vertex_count: u32,
    pub bitmap_tag: u32,
    /// the decorator_set (dctr) tag this group came from (0xFFFFFFFF if unresolved)
    pub dctr_tag: u32,
    /// DECO_BUDGET_RE: decorator instances baked into this mesh (one mesh == one
    /// per-cluster group). The renderer distance-sorts groups and draws nearest-first
    /// up to a total instance budget — the engine's decorator decimation (gap #5).
    pub instance_count: u32,
}

/// One baked scenario object placement (scenery/vehicle/weapon/etc.), enumerated
/// statically from the scnr via `ZH_SCNR_EnumerateObjects` — a map's DESIGNER
/// objects, renderable with no live game. (Forge-editor objects are NOT here.)
#[derive(Clone, Copy, Debug)]
pub struct ScnrObject {
    pub mode_tag: u32,  // resolved render_model tag id
    pub obj_tag: u32,   // raw obj (scen/vehi/…) short id
    pub category: u16,  // 0=scenery,1=biped,2=vehicle,3=equip,4=weapon,5=mach,6=ctrl,7=giant,8=efsc,9=crate
    /// the placement's index into its category's scnr palette block (int16 @+10).
    pub palette_index: i16,
    /// the placement's scnr object-names index (int16 @+12; -1 = unnamed).
    pub name_index: i16,
    /// s_scenario_object_datum placement_flags. bit0="not automatically" (spawned by
    /// script/gametype, NOT at map load), bit6="never placed". Uses these to
    /// suppress objects the engine wouldn't auto-place.
    pub placement_flags: u32,
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    pub scale: f32,
}

/// A sandbox forge-palette entry resolved from the cache, with the coordinates a
/// .mvar object placement references: `(palette_index, entry_within, variant_within)`
/// → `tag_short` (the obj tag, resolve to a render_model via `resolve_mode`).
#[derive(Clone, Debug)]
pub struct ForgePaletteEntry {
    pub palette_index: u32,
    pub entry_within: u32,
    pub variant_within: u32,
    pub tag_short: u32,
    pub name: String,
    /// Per-variant name (native struct offset 104, e.g. "room_double") — the entry `name`
    /// (offset 64) is shared across an entry's variants, this disambiguates the permutation.
    pub variant_name: String,
    /// the top-level palette CATEGORY name (native struct offset 24, resolved from the
    /// scnr palette[p].Name StringId — e.g. "STRUCTURES", "WEAPONS", "VEHICLES"). The real forge
    /// category the object lives under; empty if the cache had no string for it.
    pub category_name: String,
    /// raw variant Name stringId (native struct offset 144) — the engine's key for
    /// selecting the object's hlmt model variant. Matches `ZhAttachment.variant_name_sid`
    /// and drives the variant section mask. 0 = default / no variant.
    pub variant_name_sid: u32,
}

/// One scenario SimpleLight (point/spot), mirroring native `ZH_SimpleLight`
/// (#pragma pack(1), 80B = 5×vec4) from `ZH_SCNR_EnumerateSimpleLights`.
/// Field offsets (see SimpleLightsWalker.cpp): pos@0, cutoff²@12, dir@16, sphere%@28,
/// color(linear=tint×intensity)@32, smooth@44, cosCutoff@48 (-1=omni), angleRatio@52,
/// anglePower@56, cutoff(linear)@64, farEnd@68, farRatio@72.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhSimpleLight {
    pub pos: [f32; 3],
    pub bounding_radius_sq: f32,
    pub dir: [f32; 3],
    pub sphere_pct: f32,
    pub color: [f32; 3],
    pub smooth: f32,
    pub cos_cutoff: f32,
    pub angle_falloff_ratio: f32,
    pub angle_falloff_power: f32,
    pub _pad3: f32,
    pub cutoff: f32,
    pub far_atten_end: f32,
    pub far_atten_ratio: f32,
    pub _pad4: f32,
}

/// Sun light + lens tag id (mirrors native `ZH_SunLight`, #pragma pack(1), 52B). #[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhSunLight {
    pub has_light: u32,
    pub light_type: u32, // 0=Sphere, 1=Projective
    pub intensity: f32,
    pub sun_disk_fov_deg: f32,
    pub outer_cone_deg: f32,
    pub light_range: f32,
    pub flags: u16,
    pub shadow_res_selector: u16,
    pub gel_map_tag_id: u32,
    pub lens_flare_tag_id: u32, // 0xFFFF = none
    pub _pad: [u32; 4],
}

/// Analytical "sky light" — the map's outdoor sun authored in the sky render_model
/// (NOT the ligh tag). From `ZH_SCNR_GetSkyLight` (LightWalker.cpp). `dir` is the
/// authored vector (mode+0x224), `color` the HDR-linear sun RGB (mode+0x230..238),
/// `ambient` the natural-light SH DC per channel (mode+0x124/0x164/0x1A4).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhSkyLight {
    pub has_light: u32,
    pub dir: [f32; 3],
    pub color: [f32; 3],
    pub ambient: [f32; 3],
    pub _pad: [u32; 3],
}

/// One lens-flare reflection element (mirrors native `ZH_LensFlareElement`, packed, 60B). #[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhLensFlareElement {
    pub axis_offset: f32, // 0=at sun, 1≈screen centre, 2=antipode
    pub rotation_offset: f32,
    pub radius_min: f32,
    pub radius_max: f32,
    pub brightness_min: f32,
    pub brightness_max: f32,
    pub modulation_factor: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    pub tint_power: f32,
    pub bitmap_index: i32,
    pub flags8: u32,
    pub radius_curve_size: u32,
    pub bright_curve_size: u32,
}

/// Lens-flare system info (mirrors native `ZH_LensFlareInfo`, packed, 20B). #[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhLensFlareInfo {
    pub falloff_angle_rad: f32,
    pub cutoff_angle_rad: f32,
    pub occlusion_inner_radius_scale: f32,
    pub flags: u32,
    pub bitmap_tag_id: u32, // sprite atlas bitm; 0xFFFF=none
}

/// One planar fog volume (mirrors native `ZH_PlanarFogVolume`, #pragma pack(1), 92B)
/// from `ZH_PFOG_Enumerate` — Reach `sddt.PlanarFog[]` / WaterInstances.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ZhPlanarFogVolume {
    pub normal: [f32; 3],
    pub plane_d: f32,
    pub depth: f32,
    pub color: [f32; 4], // rgb = fogg inscatter, a = density
    pub density: f32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub centroid: [f32; 3],
    pub vertex_count: u32,
    pub _pad: [u32; 3],
}

/// One BSP airprobe SH lighting point (mirrors native `ZH_AirprobePoint`, 44B).
/// `ambient` is the LINEAR HDR up-facing ambient the engine's dual-VMF eval
/// produces at this probe — the same value baked into a decorator's
/// per-instance color. The directional terms describe the dominant lobe.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AirprobePoint {
    pub pos: [f32; 3],
    pub ambient: [f32; 3],
    pub dom_dir: [f32; 3],
    pub mask: f32,
    pub dir_weight: f32,
    /// OBJ-PROBE: raw dual-vMF lobes (dominant colour, fill colour, bandwidth κ) for per-pixel object lighting.
    pub dom_rgb: [f32; 3],
    pub fill_rgb: [f32; 3],
    pub bandwidth: f32,
    /// the FILL lobe's direction (terms 8..10, unit or zero) — the engine's airprobe
    /// blend sums it per probe and its `sub_140828CD0` merge mixes it into the object's analytical
    /// light direction (doc 16 §1.1/§2.1). Record is 84B.
    pub fill_dir: [f32; 3],
}

/// Loaded DLL + resolved core exports. `Library` is kept alive for the lifetime
/// of the resolved function pointers (they're valid only while it's loaded).
pub struct NativeDll {
    // Wrapped in ManuallyDrop so NativeDll's Drop impl can quiesce the DLL and then
    // LEAK the module instead of FreeLibrary'ing it. HaloMapStudioDLL's DllMain spawns
    // a background worker thread that outlives FreeLibrary; unmapping the DLL while that
    // thread is being dispatched executes freed code → access violation. Crash signature
    // (all dumps, MCC open or not): Unloaded_HaloMapStudioDLL.dll+0x145d, thread start
    // routine dispatched by ntdll with the module already gone. Never unmapping it makes
    // the thread's code page always valid; at process exit the OS terminates threads
    // before final teardown, so leaking is safe.
    _lib: std::mem::ManuallyDrop<Library>,
    // bitmap / cache
    mbp_open_cache: FnOpenCache,
    mbp_close_cache: FnU64,
    mbp_find_bitmap: Option<FnFindBitmap>,
    mbp_decode_bitmap: Option<FnDecodeBitmap>,
    mbp_decode_bitmap_keep: Option<FnDecodeBitmap>,
    mbp_decode_bitmap_slice: Option<FnDecodeBitmapSlice>,
    mbp_decode_bitmap_face: Option<FnDecodeBitmapFace>,
    mbp_free: Option<FnFree>,
    mbp_get_raw_dds: Option<FnGetRawDds>,
    mbp_free_raw_dds: Option<FnFree>,
    // bsp
    bsp_open: FnOpenBsp,
    bsp_close: FnU64,
    bsp_mesh_count: FnU64ToU32,
    bsp_get_mesh: FnGetBspMesh,
    bsp_decode_geom: FnDecodeGeom,
    bsp_raw_geom: Option<FnGetRawGeom>,
    bsp_decode_normals: Option<FnDecodeVec>,
    // G1: per-vertex tangent frame (Int16_N4 @ +0x1C, /32767). Tangents = float3[vc];
    // binormals = cross(N,T)·sign(tan.w) so the tangent's discarded 4th-component
    // handedness is recovered. Feeds the stored-tangent TBN in the mesh shader.
    bsp_decode_tangents: Option<FnDecodeVec>,
    bsp_decode_binormals: Option<FnDecodeVec>,
    bsp_decode_uvs: Option<FnDecodeVec>,
    bsp_mat_diffuse: Option<FnTagByShader>,
    bsp_mat_shader_kind: Option<FnTagByShader>,
    bsp_mat_blend: Option<FnTagByShader>,
    // MAT-1: per-BSP-material material_model enum (0..9); 0xFF unresolved.
    bsp_mat_material_model: Option<FnTagByShader>,
    bsp_mat_albedo_option: Option<FnTagByShader>,
    // LIT-SI-3: per-BSP-material self_illumination mode enum (0..12); 0xFF unresolved.
    bsp_mat_self_illum_mode: Option<FnTagByShader>,
    bsp_mat_shader_class: Option<FnTagByShader>,
    bsp_mat_shader_tag: Option<FnTagByShader>,
    bsp_mat_bitmap_by_usage: Option<FnBitmapByUsage>,
    bsp_mat_sampler_by_usage: Option<FnBitmapByUsage>, // #condemned-fx
    bsp_terrain_layers: Option<FnTerrainLayers>,
    bsp_material_overlays: Option<FnMaterialOverlays>,
    mmp_overlay_count: Option<FnMmpOverlayCount>,
    mmp_overlay_at: Option<FnMmpOverlayAt>,
    tag_overlay_count: Option<FnTagOverlayCount>,
    tag_overlay_at: Option<FnTagOverlayAt>,
    bsp_enum_decals: Option<FnEnumDecals>,
    bsp_free_decals: Option<FnFree>,
    // preplaced decals (the baked artist-painted surface-coating decals — the
    // "whole wall covered" ice that runtime decals don't provide).
    bsp_preplaced_count: Option<FnPreplacedCount>,
    bsp_enum_preplaced: Option<FnEnumPreplaced>,
    bsp_free_preplaced: Option<FnFree>,
    bsp_decode_pp_geom: Option<FnDecodePpGeom>,
    bsp_free_pp_geom: Option<FnFreePpGeom>,
    bsp_free: FnFree,
    bsp_enum_sbsps: Option<FnEnumSbsps>,
    tag_enum_by_group: Option<FnEnumByGroup>,
    tag_class: Option<unsafe extern "system" fn(u64, u32) -> u32>,
    scnr_enum_triggers: Option<FnEnumTriggers>,
    sddt_enum_soft_ceilings: Option<FnEnumSoftCeilings>,
    sddt_enum_soft_ceiling_tris: Option<FnEnumSoftCeilingTris>,
    tag_resolve_coll: Option<FnResolveTag>,
    tag_resolve_mode: Option<FnResolveTag>,
    tag_read_meta: Option<FnReadMeta>,
    tag_read_ptr: Option<FnReadPtr>,
    /// active scenario_lightmap_bsp_data tag for an sbsp (research + atlas work).
    lbsp_active_tag: Option<unsafe extern "C" fn(u64, u32, *mut i32) -> u32>,
    tag_resolve_armor_icon: Option<FnResolveTag>,
    scnr_forge_palette: Option<FnEnumTriggers>,
    coll_decode: Option<FnCollDecode>,
    coll_free: Option<FnFree>,
    tag_resolve_phmo: Option<FnResolveTag>,
    phmo_decode: Option<FnCollDecode>,
    phmo_free: Option<FnFree>,
    // model
    mmp_open: FnOpenModel,
    mmp_close: FnU64,
    mmp_section_count: FnU64ToU32,
    mmp_get_section: FnGetSection,
    mmp_submesh_count: Option<FnU64ToU32>,
    mmp_section_allowed: Option<FnMmpSectionAllowed>,
    mmp_enum_attachments: Option<FnEnumAttachments>,
    mmp_variant_section_mask: Option<FnVariantSectionMask>,
    mmp_default_variant_section_mask: Option<FnDefaultVariantSectionMask>,
    mmp_resolve_default_variant_mask: Option<FnDefaultVariantSectionMask>,
    mmp_get_submesh: Option<FnGetSubmesh>,
    mbp_page_stats: Option<unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64, *mut u64)>,
    mmp_decode_geom: FnDecodeGeom,
    mmp_decode_uvs: Option<FnDecodeVec>,
    mmp_decode_colors: Option<FnDecodeVec>,
    mmp_shader_diffuse: Option<FnTagByShader>,
    mmp_shader_emissive: Option<FnTagByShader>,
    mmp_shader_blend: Option<FnTagByShader>,
    // MAT-1: per-model-shader material_model enum (0..9); 0xFF unresolved.
    mmp_shader_material_model: Option<FnTagByShader>,
    // LIT-SI-3: per-model-shader self_illumination MODE enum (0..12); 0xFF unresolved.
    mmp_shader_self_illum_mode: Option<FnTagByShader>,
    // SKY-3: per-model-shader sky class (1=sky_dome_simple, 0=other, 0xFF unresolved).
    mmp_shader_sky_class: Option<FnTagByShader>,
    mmp_shader_class: Option<FnTagByShader>,
    mmp_shader_bitmap_by_usage: Option<FnBitmapByUsage>,
    mmp_shader_sampler_by_usage: Option<FnBitmapByUsage>, // #condemned-fx
    mmp_shader_template: Option<FnShaderTemplate>,
    /// ZH_MMP_GetShaderTagId -- the render_method tag id a material references (-1 = none).
    mmp_shader_tag: Option<FnShaderTemplate>,
    mmp_shader_alpha_test: Option<FnTagByShader>,
    mmp_shader_constants: Option<FnShaderConstants>,
    // sky (scenario Skies[] → scenery → hlmt → render_model)
    sky_count: Option<FnSkyCount>,
    sky_render_model: Option<FnSkyTag>,
    tag_name: Option<FnTagName>,
    bsp_shader_constants: Option<FnShaderConstants>,
    // W22/W23 investigation probe (HMS_UWPROBE); no-op export otherwise.
    bsp_underwater_probe: Option<unsafe extern "C" fn(u64, i32)>,
    // W22/W23: authored underwater fog color + murkiness from atmosphere_globals (atgf).
    bsp_underwater_fog: Option<unsafe extern "C" fn(u64, *mut f32, *mut f32, *mut f32, *mut f32) -> i32>,
    bsp_diffuse_tiling: Option<FnBspTiling>,
    scnr_objects: Option<FnScnrObjects>,
    scnr_free_objects: Option<FnFreeScnrObjects>,
    scnr_simple_lights: Option<FnSimpleLights>,
    pfog_enumerate: Option<FnPfog>,
    pfog_free: Option<FnFreePfog>,
    scnr_sun_light: Option<FnSunLight>,
    scnr_sky_light: Option<FnSkyLight>,
    lens_elements: Option<FnLensElements>,
    mmp_free: FnFree,
    // lightmap (baked lighting)
    lbsp_airprobe_grid: Option<FnAirprobeGrid>,
    lbsp_free_airprobe_grid: Option<FnFree>,
    lbsp_brightness: Option<FnBrightness>,
    bsp_deco_geom: Option<FnDecoGeom>,
    bsp_free_deco: Option<FnFreeDeco>,
    lbsp_cluster_pvl: Option<FnClusterPvl>,
    lbsp_atlas: Option<FnLbspAtlas>,
    lbsp_cluster_entry: Option<FnClusterEntry>,
    bsp_decode_uv2: Option<FnBspUv2>,
    /// the per-instance PER-VERTEX LIGHTPROBE vertex buffer (raw bytes + layout).
    bsp_instance_pvl_vb: Option<
        unsafe extern "C" fn(u64, u32, *mut *mut u8, *mut u32, *mut u32, *mut u32) -> u8,
    >,
    lbsp_instance_pvl: Option<FnClusterPvl>,
    lbsp_instance_probe: Option<FnInstanceProbe>,
    lbsp_instance_atlas: Option<FnInstanceAtlas>,
    fog_get_params: Option<FnFogParams>,
    globals_change_colors: Option<FnChangeColors>,
    scnr_exposure: Option<FnScnrExposure>,
    lbsp_free_lightprobe: Option<FnFree>,
    // lifecycle
    prepare_unload: Option<FnVoid>,
    /// _heapmin() — return the CRT's freed transient-decode heap to the OS.
    trim_heaps: Option<FnVoid>,
    mbp_clear_page_cache: Option<FnU64>,
    /// bytes held by the native inflate-once page cache (HMS_MEMDIAG report).
    mbp_page_cache_bytes: Option<FnU64ToU64>,
    /// drop the resident pages of the .map file mapping(s) (re-fault on demand).
    mbp_drop_map_pages: Option<FnU64>,
}

// SAFETY: NativeDll is a loaded Library + raw C fn pointers + no interior mutable
// state. The C++ DLL serializes all cache access internally (parseMutex), so
// concurrent calls from multiple threads are safe. Sharing via Arc<NativeDll>
// across the load-worker + main thread is therefore sound.
unsafe impl Send for NativeDll {}
unsafe impl Sync for NativeDll {}

macro_rules! req {
    ($lib:expr, $name:literal) => {
        unsafe {
            *$lib
                .get(concat!($name, "\0").as_bytes())
                .map_err(|e| anyhow!("required export {} missing: {e}", $name))?
        }
    };
}
macro_rules! opt {
    ($lib:expr, $name:literal) => {
        unsafe { $lib.get(concat!($name, "\0").as_bytes()).ok().map(|s| *s) }
    };
}

impl NativeDll {
    /// Load the DLL at `dll_path` and resolve the core parsing exports.
    pub fn load(dll_path: &Path) -> Result<NativeDll> {
        let lib = unsafe { Library::new(dll_path) }
            .map_err(|e| anyhow!("LoadLibrary {}: {e}", dll_path.display()))?;
        let dll = NativeDll {
            mbp_open_cache: req!(lib, "ZH_MBP_OpenCache"),
            mbp_close_cache: req!(lib, "ZH_MBP_CloseCache"),
            mbp_find_bitmap: opt!(lib, "ZH_MBP_FindBitmapTag"),
            mbp_decode_bitmap: opt!(lib, "ZH_MBP_DecodeBitmap"),
            mbp_decode_bitmap_keep: opt!(lib, "ZH_MBP_DecodeBitmapKeepAlpha"),
            mbp_decode_bitmap_slice: opt!(lib, "ZH_MBP_DecodeBitmapSlice"),
            mbp_decode_bitmap_face: opt!(lib, "ZH_MBP_DecodeBitmapFace"),
            mbp_free: opt!(lib, "ZH_MBP_FreeBuffer"),
            mbp_get_raw_dds: opt!(lib, "ZH_MBP_GetRawDDS"),
            mbp_free_raw_dds: opt!(lib, "ZH_MBP_FreeRawDDS"),
            bsp_open: req!(lib, "ZH_BSP_OpenBsp"),
            bsp_close: req!(lib, "ZH_BSP_CloseBsp"),
            bsp_mesh_count: req!(lib, "ZH_BSP_GetMeshCount"),
            bsp_get_mesh: req!(lib, "ZH_BSP_GetMesh"),
            bsp_decode_geom: req!(lib, "ZH_BSP_DecodeMeshGeometry"),
            bsp_raw_geom: opt!(lib, "ZH_BSP_GetRawGeometry"),
            bsp_decode_normals: opt!(lib, "ZH_BSP_DecodeMeshNormals"),
            bsp_decode_tangents: opt!(lib, "ZH_BSP_DecodeMeshTangents"),
            bsp_decode_binormals: opt!(lib, "ZH_BSP_DecodeMeshBinormals"),
            bsp_decode_uvs: opt!(lib, "ZH_BSP_DecodeMeshUVs"),
            bsp_mat_diffuse: opt!(lib, "ZH_BSP_GetMaterialDiffuseBitmapTagId"),
            bsp_mat_shader_kind: opt!(lib, "ZH_BSP_GetMaterialShaderKind"),
            bsp_mat_blend: opt!(lib, "ZH_BSP_GetMaterialBlendMode"),
            bsp_mat_material_model: opt!(lib, "ZH_BSP_GetMaterialModel"),
            bsp_mat_albedo_option: opt!(lib, "ZH_BSP_GetMaterialAlbedoOption"),
            bsp_mat_self_illum_mode: opt!(lib, "ZH_BSP_GetSelfIllumMode"),
            bsp_mat_shader_class: opt!(lib, "ZH_BSP_GetShaderClass"),
            bsp_mat_shader_tag: opt!(lib, "ZH_BSP_GetMaterialShaderTagId"),
            bsp_mat_bitmap_by_usage: opt!(lib, "ZH_BSP_GetMaterialBitmapByUsage"),
            bsp_mat_sampler_by_usage: opt!(lib, "ZH_BSP_GetMaterialSamplerByUsage"),
            bsp_terrain_layers: opt!(lib, "ZH_BSP_GetMaterialTerrainLayers"),
            bsp_material_overlays: opt!(lib, "ZH_BSP_GetMaterialOverlays"),
            mmp_overlay_count: opt!(lib, "ZH_MMP_GetShaderOverlayCount"),
            mmp_overlay_at: opt!(lib, "ZH_MMP_GetShaderOverlayAt"),
            tag_overlay_count: opt!(lib, "ZH_TAG_GetShaderOverlayCount"),
            tag_overlay_at: opt!(lib, "ZH_TAG_GetShaderOverlayAt"),
            bsp_enum_decals: opt!(lib, "ZH_BSP_EnumerateDecals"),
            bsp_free_decals: opt!(lib, "ZH_BSP_FreeDecalBuffer"),
            bsp_preplaced_count: opt!(lib, "HaloMapStudio_BSP_GetPreplacedDecalCount"),
            bsp_enum_preplaced: opt!(lib, "HaloMapStudio_BSP_EnumeratePreplacedDecals"),
            bsp_free_preplaced: opt!(lib, "HaloMapStudio_BSP_FreePreplacedDecals"),
            bsp_decode_pp_geom: opt!(lib, "HaloMapStudio_BSP_DecodePreplacedGeometry"),
            bsp_free_pp_geom: opt!(lib, "HaloMapStudio_BSP_FreePreplacedGeometry"),
            bsp_free: req!(lib, "ZH_BSP_FreeBuffer"),
            bsp_enum_sbsps: opt!(lib, "ZH_BSP_EnumerateSbspsInScenario"),
            tag_enum_by_group: opt!(lib, "ZH_TAG_EnumerateByGroup"),
            tag_class: opt!(lib, "ZH_TAG_GetClass"),
            scnr_enum_triggers: opt!(lib, "ZH_SCNR_EnumerateTriggerVolumes"),
            sddt_enum_soft_ceilings: opt!(lib, "ZH_SDDT_EnumerateSoftCeilings"),
            sddt_enum_soft_ceiling_tris: opt!(lib, "ZH_SDDT_EnumerateSoftCeilingTriangles"),
            tag_resolve_coll: opt!(lib, "ZH_TAG_ResolveCollTagId"),
            tag_resolve_mode: opt!(lib, "ZH_TAG_ResolveModeTagId"),
            tag_read_meta: opt!(lib, "ZH_TAG_ReadMeta"),
            tag_read_ptr: opt!(lib, "ZH_TAG_ReadPtr"),
            lbsp_active_tag: opt!(lib, "ZH_LBSP_GetActiveLbspTagId"),
            tag_resolve_armor_icon: opt!(lib, "ZH_TAG_ResolveArmorIconBitmap"),
            scnr_forge_palette: opt!(lib, "ZH_SCNR_EnumerateForgePalette"),
            scnr_simple_lights: opt!(lib, "ZH_SCNR_EnumerateSimpleLights"),
            pfog_enumerate: opt!(lib, "ZH_PFOG_Enumerate"),
            pfog_free: opt!(lib, "ZH_PFOG_FreeBuffer"),
            scnr_sun_light: opt!(lib, "ZH_SCNR_GetSunLight"),
            scnr_sky_light: opt!(lib, "ZH_SCNR_GetSkyLight"),
            lens_elements: opt!(lib, "ZH_LENS_GetElements"),
            coll_decode: opt!(lib, "ZH_COLL_DecodeGeometry"),
            coll_free: opt!(lib, "ZH_COLL_FreeBuffer"),
            tag_resolve_phmo: opt!(lib, "ZH_TAG_ResolvePhmoTagId"),
            phmo_decode: opt!(lib, "ZH_PHMO_DecodeGeometry"),
            phmo_free: opt!(lib, "ZH_PHMO_FreeBuffer"),
            mmp_open: req!(lib, "ZH_MMP_OpenModel"),
            mmp_close: req!(lib, "ZH_MMP_CloseModel"),
            mmp_section_count: req!(lib, "ZH_MMP_GetSectionCount"),
            mmp_get_section: req!(lib, "ZH_MMP_GetSection"),
            mmp_submesh_count: opt!(lib, "ZH_MMP_GetSubmeshCount"),
            mmp_section_allowed: opt!(lib, "ZH_MMP_IsSectionAllowed"),
            mmp_enum_attachments: opt!(lib, "ZH_MMP_EnumerateAttachments"),
            mmp_variant_section_mask: opt!(lib, "ZH_MMP_GetVariantSectionMask"),
            mmp_default_variant_section_mask: opt!(lib, "ZH_MMP_GetDefaultVariantSectionMask"),
            mmp_resolve_default_variant_mask: opt!(lib, "ZH_MMP_ResolveDefaultVariantMask"),
            mmp_get_submesh: opt!(lib, "ZH_MMP_GetSubmesh"),
            mbp_page_stats: opt!(lib, "ZH_MBP_GetPageStats"),
            mmp_decode_geom: req!(lib, "ZH_MMP_DecodeSectionGeometry"),
            mmp_decode_uvs: opt!(lib, "ZH_MMP_DecodeSectionUVs"),
            mmp_decode_colors: opt!(lib, "ZH_MMP_DecodeSectionColors"),
            mmp_shader_diffuse: opt!(lib, "ZH_MMP_GetShaderDiffuseBitmapTagId"),
            mmp_shader_emissive: opt!(lib, "ZH_MMP_GetShaderEmissiveBitmapTagId"),
            mmp_shader_blend: opt!(lib, "ZH_MMP_GetShaderBlendMode"),
            mmp_shader_material_model: opt!(lib, "ZH_MMP_GetShaderMaterialModel"),
            mmp_shader_self_illum_mode: opt!(lib, "ZH_MMP_GetShaderSelfIllumMode"),
            mmp_shader_sky_class: opt!(lib, "ZH_MMP_GetShaderSkyClass"),
            mmp_shader_class: opt!(lib, "ZH_MMP_GetShaderClass"),
            mmp_shader_bitmap_by_usage: opt!(lib, "ZH_MMP_GetShaderBitmapByUsage"),
            mmp_shader_sampler_by_usage: opt!(lib, "ZH_MMP_GetShaderSamplerByUsage"),
            mmp_shader_template: opt!(lib, "ZH_MMP_GetShaderTemplateTagId"),
            mmp_shader_tag: opt!(lib, "ZH_MMP_GetShaderTagId"),
            mmp_shader_alpha_test: opt!(lib, "ZH_MMP_GetShaderAlphaTestBitmapTagId"),
            mmp_shader_constants: opt!(lib, "ZH_MMP_GetShaderConstants"),
            sky_count: opt!(lib, "ZH_SKY_GetSkyCount"),
            sky_render_model: opt!(lib, "ZH_SKY_GetRenderModelTagId"),
            tag_name: opt!(lib, "ZH_TAG_GetName"),
            bsp_shader_constants: opt!(lib, "ZH_BSP_GetMaterialShaderConstants"),
            bsp_underwater_probe: opt!(lib, "ZH_BSP_UnderwaterProbe"),
            bsp_underwater_fog: opt!(lib, "ZH_BSP_GetUnderwaterFog"),
            bsp_diffuse_tiling: opt!(lib, "ZH_BSP_GetMaterialDiffuseTiling"),
            scnr_objects: opt!(lib, "ZH_SCNR_EnumerateObjects"),
            scnr_free_objects: opt!(lib, "ZH_SCNR_FreeObjectBuffer"),
            lbsp_airprobe_grid: opt!(lib, "ZH_LBSP_GetAirprobeGrid"),
            lbsp_free_airprobe_grid: opt!(lib, "ZH_LBSP_FreeAirprobeGrid"),
            lbsp_brightness: opt!(lib, "ZH_LBSP_GetBrightness"),
            bsp_deco_geom: opt!(lib, "ZH_BSP_GetRuntimeDecoratorGeometry"),
            bsp_free_deco: opt!(lib, "ZH_BSP_FreeRuntimeDecoratorMeshes"),
            lbsp_cluster_pvl: opt!(lib, "ZH_LBSP_GetClusterPvlVb"),
            lbsp_atlas: opt!(lib, "ZH_LBSP_GetLightprobeAtlas"),
            lbsp_cluster_entry: opt!(lib, "ZH_LBSP_GetClusterEntry"),
            bsp_decode_uv2: opt!(lib, "ZH_BSP_DecodeMeshUV2"),
            bsp_instance_pvl_vb: opt!(lib, "ZH_BSP_GetInstancePvlVb"),
            lbsp_instance_pvl: opt!(lib, "ZH_LBSP_GetInstancePvl"),
            lbsp_instance_probe: opt!(lib, "ZH_LBSP_GetInstanceProbe"),
            lbsp_instance_atlas: opt!(lib, "ZH_LBSP_GetInstanceAtlasSubmap"),
            fog_get_params: opt!(lib, "ZH_FOG_GetParams"),
            globals_change_colors: opt!(lib, "ZH_GLOBALS_GetChangeColors"),
            scnr_exposure: opt!(lib, "ZH_SCNR_GetExposure"),
            lbsp_free_lightprobe: opt!(lib, "ZH_LBSP_FreeLightprobeBuffer"),
            mmp_free: req!(lib, "ZH_MMP_FreeBuffer"),
            prepare_unload: opt!(lib, "ZH_MMP_PrepareUnload"),
            trim_heaps: opt!(lib, "ZH_TrimHeaps"),
            mbp_clear_page_cache: opt!(lib, "ZH_MBP_ClearPageCache"),
            mbp_page_cache_bytes: opt!(lib, "ZH_MBP_GetPageCacheBytes"),
            mbp_drop_map_pages: opt!(lib, "ZH_MBP_DropMapPages"), // #mem2
            _lib: std::mem::ManuallyDrop::new(lib),
        };
        Ok(dll)
    }

    // ---- cache ----
    pub fn open_cache(&self, path: &str) -> Result<CacheHandle> {
        // The native side takes a `const wchar_t*`, and wchar_t is NOT the same
        // width everywhere: 16-bit on Windows, 32-bit on Linux/glibc. Encode to
        // match the target's wchar_t so the same C++ source works either way.
        #[cfg(windows)]
        let wide: Vec<WideChar> = path.encode_utf16().chain(std::iter::once(0)).collect();
        #[cfg(not(windows))]
        let wide: Vec<WideChar> = path.chars().map(|c| c as WideChar).chain(std::iter::once(0)).collect();
        let h = unsafe { (self.mbp_open_cache)(wide.as_ptr()) };
        if h == 0 {
            bail!("ZH_MBP_OpenCache failed for {path}");
        }
        Ok(CacheHandle(h))
    }
    pub fn close_cache(&self, h: CacheHandle) {
        unsafe { (self.mbp_close_cache)(h.0) }
    }

    /// Enumerate the structure-BSP tag ids referenced by a scenario tag.
    pub fn enumerate_sbsps(&self, cache: CacheHandle, scnr_tag: u32, max: u32) -> Vec<u32> {
        let Some(f) = self.bsp_enum_sbsps else { return vec![] };
        let mut ids = vec![0u32; max as usize];
        let n = unsafe { f(cache.0, scnr_tag, ids.as_mut_ptr(), max) } as usize;
        ids.truncate(n.min(max as usize));
        ids
    }

    /// Enumerate cache tag ids of a 4-char group code (e.g. "scnr"). Used to
    /// resolve the scenario tag for the offline (.map cache) path.
    pub fn tags_by_group(&self, cache: CacheHandle, group: &str) -> Vec<u32> {
        let Some(f) = self.tag_enum_by_group else { return vec![] };
        let b = group.as_bytes();
        if b.len() != 4 {
            return vec![];
        }
        let group_bytes = (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16) | ((b[3] as u32) << 24);
        let total = unsafe { f(cache.0, group_bytes, std::ptr::null_mut(), 0) };
        let total = total.min(0x10000);
        if total == 0 {
            return vec![];
        }
        let mut ids = vec![0u32; total as usize];
        let written = unsafe { f(cache.0, group_bytes, ids.as_mut_ptr(), total) } as usize;
        ids.truncate(written.min(total as usize));
        ids
    }

    /// The scenario (scnr) tag id for this cache, or None. First match — an MCC
    /// map cache carries exactly one scenario.
    pub fn find_scenario(&self, cache: CacheHandle) -> Option<u32> {
        self.tags_by_group(cache, "scnr").into_iter().next()
    }

    /// Resolve an object/primary tag to its collision_model tag (0xFFFFFFFF none).
    /// Resolve an object-definition tag (obje/bloc/scen/…) → its render_model (mode)
    /// tag, so statically-placed / locally-placed forge objects can be rendered
    /// without a live game. 0/0xFFFFFFFF when unresolved or the export is missing.
    /// resolve an armor-ability equipment (eqip) obj tag → its floating ICON bitmap tag
    /// (the per-ability `unsc_equipment_drop_holo_icon_*`). 0xFFFFFFFF when none / export missing.
    pub fn resolve_armor_icon(&self, cache: CacheHandle, eqip_tag: u32) -> u32 {
        self.tag_resolve_armor_icon.map_or(0xFFFF_FFFF, |f| unsafe { f(cache.0, eqip_tag) })
    }

    /// Bytes addressed by a raw tag-block pointer (from a TagBlockRef {count, ptr}).
    /// the active Lbsp tag id for an sbsp (0xFFFFFFFF = none).
    pub fn lbsp_active_tag(&self, cache: CacheHandle, sbsp: u32) -> Option<u32> {
        let f = self.lbsp_active_tag?;
        let mut idx = 0i32;
        let id = unsafe { f(cache.0, sbsp, &mut idx) };
        (id != 0xFFFF_FFFF).then_some(id)
    }

    pub fn tag_ptr_bytes(&self, cache: CacheHandle, raw_ptr: u32, len: u32) -> Option<Vec<u8>> {
        let f = self.tag_read_ptr?;
        let mut buf = vec![0u8; len as usize];
        let n = unsafe { f(cache.0, raw_ptr, len, buf.as_mut_ptr()) };
        (n as u32 == len).then_some(buf)
    }

    /// render_method template (rmt2) tag PATH for a shader tag (rmsh/rmgl/rmhg/...): postprocess[0].template.
    /// The basename digits encode the option choices (e.g. rmsh `_0_2_0_2_1_2_0_3_0_0_0_1`: albedo, bump, alpha_test,
    /// specular_mask, material_model, ENVIRONMENT_MAPPING (0 none,1 per_pixel,2 dynamic,3 from_flat_texture),
    /// self_illumination, blend_mode, parallax, misc, distortion, alpha_blend_source).
    pub fn rm_template_name(&self, cache: CacheHandle, shader_tag: u32) -> Option<String> {
        let m = self.tag_meta(cache, shader_tag, 0x38, 8)?;
        let cnt = u32::from_le_bytes([m[0], m[1], m[2], m[3]]);
        let ptr = u32::from_le_bytes([m[4], m[5], m[6], m[7]]);
        if cnt == 0 { return None; }
        let pp = self.tag_ptr_bytes(cache, ptr, 16)?;
        let datum = u32::from_le_bytes([pp[12], pp[13], pp[14], pp[15]]);
        if datum == 0xFFFF_FFFF { return None; }
        self.tag_name(cache, datum & 0xFFFF)
    }

    /// Overlay records of ANY render_method tag id: (type, routed arg name, period, out_min, out_max, func_type,
    /// curve_points[16] baked by the native walker over one period).
    pub fn shader_overlays_by_tag(&self, cache: CacheHandle, shader_tag: u32, arg_names: &[String]) -> Vec<(i32, String, f32, f32, f32, u8, [f32; 16])> {
        let (Some(cnt), Some(at)) = (self.tag_overlay_count, self.tag_overlay_at) else { return Vec::new(); };
        let n = unsafe { cnt(cache.0, shader_tag as i32) };
        let mut out = Vec::new();
        let mut buf = [0u8; 148];
        let rd_i32 = |b: &[u8], o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let rd_f32 = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        for i in 0..n.min(64) {
            if unsafe { at(cache.0, shader_tag as i32, i, buf.as_mut_ptr()) } == 0 { continue; }
            let arg_idx = u32::from_le_bytes([buf[140], buf[141], buf[142], buf[143]]);
            let name = arg_names.get(arg_idx as usize).cloned().unwrap_or_else(|| format!("arg{arg_idx}"));
            let mut cp = [0f32; 16];
            for k in 0..16 { cp[k] = rd_f32(&buf, 76 + k * 4); }
            out.push((rd_i32(&buf, 0), name, rd_f32(&buf, 12), rd_f32(&buf, 60), rd_f32(&buf, 64), buf[48], cp));
        }
        out
    }

    /// Per-PARAMETER UV scroll rate for ANY render_method tag id (BSP materials): TranslationX/Y overlays
    /// whose routed argument name (looked up in `arg_names`, the shader's real-constant list in order)
    /// starts with `param`. [0,0] when not animated.
    pub fn shader_scroll_param_by_tag(&self, cache: CacheHandle, shader_tag: u32, arg_names: &[String], param: &str) -> [f32; 2] {
        let (Some(cnt), Some(at)) = (self.tag_overlay_count, self.tag_overlay_at) else { return [0.0, 0.0]; };
        let n = unsafe { cnt(cache.0, shader_tag as i32) };
        let mut scroll = [0.0f32, 0.0f32];
        let mut buf = [0u8; 148];
        let rd_i32 = |b: &[u8], o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let rd_f32 = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        for i in 0..n.min(64) {
            if unsafe { at(cache.0, shader_tag as i32, i, buf.as_mut_ptr()) } == 0 { continue; }
            let ty = rd_i32(&buf, 0);
            if ty != 5 && ty != 6 { continue; }
            let arg_idx = u32::from_le_bytes([buf[140], buf[141], buf[142], buf[143]]);
            let name = arg_names.get(arg_idx as usize).map(|s| s.as_str()).unwrap_or("");
            if !name.starts_with(param) { continue; }
            let period = rd_f32(&buf, 12);
            if !(period.is_finite() && period > 1e-3) { continue; }
            let amp = rd_f32(&buf, 64) - rd_f32(&buf, 60);
            // Sign: the engine's animated xform offset moves the texture the OTHER way from `uv + rate*t`
            // (verified on the forge_halo waterfalls, which flowed upstream with +1/period) -> negate.
            let rate = -(if amp.is_finite() && amp.abs() > 1e-6 { amp / period } else { 1.0 / period });
            if !rate.is_finite() { continue; }
            if ty == 5 { scroll[0] += rate; } else { scroll[1] += rate; }
        }
        [scroll[0].clamp(-4.0, 4.0), scroll[1].clamp(-4.0, 4.0)]
    }

    /// Raw tag-meta bytes [offset, offset+len) (None when the export is missing or the read fails).
    pub fn tag_meta(&self, cache: CacheHandle, tag: u32, offset: u32, len: u32) -> Option<Vec<u8>> {
        let f = self.tag_read_meta?;
        let mut buf = vec![0u8; len as usize];
        let n = unsafe { f(cache.0, tag, offset, len, buf.as_mut_ptr()) };
        (n as u32 == len).then_some(buf)
    }

    pub fn resolve_mode(&self, cache: CacheHandle, obj_tag: u32) -> u32 {
        self.tag_resolve_mode.map_or(0, |f| unsafe { f(cache.0, obj_tag) })
    }

    /// Enumerate the scenario's forge PALETTE statically (no live game) — the object
    /// TYPES a forge session could place. Returns (palette_index, object_tag, name).
    /// ZH_ScnrForgePaletteEntry is 148B: tagShortId@20, entryName@64[40], variantNameSid@144.
    /// STRIDE MUST match the native struct — a mismatch misaligns every entry after the first,
    /// corrupting the palette-name classification (kill/safe/hill colours). Keep in sync with
    /// `scnr_forge_palette_full`.
    pub fn scnr_forge_palette(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<(u32, u32, String)> {
        let Some(f) = self.scnr_forge_palette else { return Vec::new() };
        const STRIDE: usize = 148;
        const CAP: u32 = 512;
        let mut buf = vec![0u8; STRIDE * CAP as usize];
        let n = unsafe { f(cache.0, scnr_tag, buf.as_mut_ptr(), CAP) };
        let n = (n as usize).min(CAP as usize);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * STRIDE;
            let pidx = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            let tag = u32::from_le_bytes([buf[o + 20], buf[o + 21], buf[o + 22], buf[o + 23]]);
            let name_bytes = &buf[o + 64..o + 104];
            let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(40);
            let name = String::from_utf8_lossy(&name_bytes[..end]).into_owned();
            if tag != 0 && tag != 0xFFFF_FFFF {
                out.push((pidx, tag, name));
            }
        }
        out
    }

    /// Full sandbox forge palette with the (paletteIndex, entryWithinPalette,
    /// variantWithinEntry) coordinates a .mvar placement resolves against. Mirrors
    /// the 144B ZH_ScnrForgePaletteEntry struct: pidx@0, entry@4, variant@8,
    /// tagShort@20, entryName@64. Used to resolve a .mvar `(folder,item)` → obj tag
    /// with no live game.
    pub fn scnr_forge_palette_full(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<ForgePaletteEntry> {
        let Some(f) = self.scnr_forge_palette else { return Vec::new() };
        const STRIDE: usize = 148; // includes variantNameSid@144
        const CAP: u32 = 1024;
        let mut buf = vec![0u8; STRIDE * CAP as usize];
        let n = unsafe { f(cache.0, scnr_tag, buf.as_mut_ptr(), CAP) };
        let n = (n as usize).min(CAP as usize);
        let rd = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * STRIDE;
            // KEEP tag_short==0 rows: they are placeholder ENTRY SLOTS the native walker emits
            // for empty palette entries, so the flat (paletteIndex,entryWithin) enumeration
            // matches the engine's positional variant_quota_index (a .mvar folder can index an
            // empty slot; resolve then returns 0 → the object is skipped, but the INDEX stays
            // aligned so all other objects resolve to the correct type).
            let tag_short = rd(&buf, o + 20);
            // paletteName (category) @24, entryName @64, variantName @104 — each 40B, NUL-term.
            let cat_bytes = &buf[o + 24..o + 64];
            let cend = cat_bytes.iter().position(|&b| b == 0).unwrap_or(40);
            let name_bytes = &buf[o + 64..o + 104];
            let end = name_bytes.iter().position(|&b| b == 0).unwrap_or(40);
            let vn_bytes = &buf[o + 104..o + 144];
            let vend = vn_bytes.iter().position(|&b| b == 0).unwrap_or(40);
            out.push(ForgePaletteEntry {
                palette_index: rd(&buf, o),
                entry_within: rd(&buf, o + 4),
                variant_within: rd(&buf, o + 8),
                tag_short,
                name: String::from_utf8_lossy(&name_bytes[..end]).into_owned(),
                variant_name: String::from_utf8_lossy(&vn_bytes[..vend]).into_owned(),
                category_name: String::from_utf8_lossy(&cat_bytes[..cend]).into_owned(),
                variant_name_sid: rd(&buf, o + 144),
            });
        }
        out
    }

    /// Enumerate a scenario's BAKED object placements (designer scenery/vehicles/
    /// weapons/etc.) with resolved render_model + world transform, so a map's own
    /// objects render on load with no live game. 72B stride ZH_ScnrObjectInstance.
    pub fn scnr_objects(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<ScnrObject> {
        let (Some(get), Some(free)) = (self.scnr_objects, self.scnr_free_objects) else {
            return Vec::new();
        };
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut count: u32 = 0;
        let ok = unsafe { get(cache.0, scnr_tag, &mut buf, &mut count) };
        if ok == 0 || buf.is_null() || count == 0 {
            if !buf.is_null() {
                unsafe { free(buf) };
            }
            return Vec::new();
        }
        const STRIDE: usize = 72;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let b = unsafe { buf.add(i * STRIDE) };
            let ru32 = |o: usize| unsafe { (b.add(o) as *const u32).read_unaligned() };
            let ru16 = |o: usize| unsafe { (b.add(o) as *const u16).read_unaligned() };
            let f1 = |o: usize| unsafe { (b.add(o) as *const f32).read_unaligned() };
            let f3 = |o: usize| [f1(o), f1(o + 4), f1(o + 8)];
            out.push(ScnrObject {
                mode_tag: ru32(0),
                obj_tag: ru32(4),
                category: ru16(8),
                palette_index: ru16(10) as i16, // #obj-identity
                name_index: ru16(12) as i16,    // #obj-identity
                placement_flags: ru32(16),
                pos: f3(20),
                fwd: f3(32),
                up: f3(44),
                scale: f1(68),
            });
        }
        unsafe { free(buf) };
        out
    }

    /// Enumerate the scenario's SimpleLights (point/spot), up to 8 (engine PC cap),
    /// via `ZH_SCNR_EnumerateSimpleLights`. Caller-allocated buffer; no free.
    pub fn scnr_simple_lights(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<ZhSimpleLight> {
        let Some(get) = self.scnr_simple_lights else { return Vec::new() };
        let mut buf = [ZhSimpleLight::default(); 8];
        let mut count: u32 = 0;
        let mut seen: u32 = 0;
        let mut skipped: u32 = 0;
        let ok = unsafe {
            get(cache.0, scnr_tag, buf.as_mut_ptr(), &mut count, &mut seen, &mut skipped)
        };
        if std::env::var("HMS_DIAG").is_ok() {
            eprintln!("HMS_DIAG SIMPLELIGHTS scnr={scnr_tag:#x} ok={ok} count={count} seen={seen} skipped={skipped}");
        }
        if ok == 0 || count == 0 {
            return Vec::new();
        }
        buf.into_iter().take((count as usize).min(8)).collect()
    }

    /// Enumerate the scenario's planar fog volumes (sddt.PlanarFog + WaterInstances)
    /// via `ZH_PFOG_Enumerate`. Heap buffer freed via ZH_PFOG_FreeBuffer.
    pub fn planar_fog(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<ZhPlanarFogVolume> {
        let (Some(get), Some(free)) = (self.pfog_enumerate, self.pfog_free) else {
            return Vec::new();
        };
        let mut buf: *mut ZhPlanarFogVolume = std::ptr::null_mut();
        let mut count: u32 = 0;
        let ok = unsafe { get(cache.0, scnr_tag, &mut buf, &mut count) };
        if ok == 0 || buf.is_null() || count == 0 {
            if !buf.is_null() {
                unsafe { free(buf) };
            }
            return Vec::new();
        }
        let out = unsafe { std::slice::from_raw_parts(buf, count as usize).to_vec() };
        unsafe { free(buf) };
        out
    }

    /// Resolve the scenario's sun light + its lens-flare tag id. None if absent.
    pub fn sun_light(&self, cache: CacheHandle, scnr_tag: u32) -> Option<ZhSunLight> {
        let get = self.scnr_sun_light?;
        let mut out = ZhSunLight::default();
        let ok = unsafe { get(cache.0, scnr_tag, &mut out) };
        if ok == 0 || out.has_light == 0 {
            None
        } else {
            Some(out)
        }
    }

    /// Analytical sky light (outdoor sun colour/dir + natural ambient) from the sky
    /// render_model — the map's real per-map lighting. `ZH_SCNR_GetSkyLight`.
    pub fn sky_light(&self, cache: CacheHandle, scnr_tag: u32) -> Option<ZhSkyLight> {
        let get = self.scnr_sky_light?;
        let mut out = ZhSkyLight::default();
        let ok = unsafe { get(cache.0, scnr_tag, &mut out) };
        if ok == 0 || out.has_light == 0 { None } else { Some(out) }
    }

    /// Enumerate a lens-flare tag's reflection elements. Returns (info, elements),
    /// up to 32. Empty vec = valid lens with no reflections; None = tag didn't resolve.
    pub fn lens_elements(&self, cache: CacheHandle, lens_tag: u32) -> Option<(ZhLensFlareInfo, Vec<ZhLensFlareElement>)> {
        let get = self.lens_elements?;
        let mut buf = [ZhLensFlareElement::default(); 32];
        let mut count: u32 = 0;
        let mut seen: u32 = 0;
        let mut info = ZhLensFlareInfo::default();
        let ok = unsafe {
            get(cache.0, lens_tag, buf.as_mut_ptr(), 32, &mut count, &mut seen, &mut info)
        };
        if ok == 0 {
            return None;
        }
        Some((info, buf.into_iter().take((count as usize).min(32)).collect()))
    }

    pub fn resolve_coll(&self, cache: CacheHandle, primary_tag: u32) -> u32 {
        self.tag_resolve_coll.map_or(0xFFFF_FFFF, |f| unsafe { f(cache.0, primary_tag) })
    }

    /// Decode a collision model into (positions, indices) triangle soup.
    pub fn coll_geometry(&self, cache: CacheHandle, coll_tag: u32) -> Option<(Vec<f32>, Vec<u32>)> {
        let (Some(dec), Some(free)) = (self.coll_decode, self.coll_free) else { return None };
        let mut vp: *mut f32 = std::ptr::null_mut();
        let mut vc: u32 = 0;
        let mut ip: *mut u32 = std::ptr::null_mut();
        let mut ic: u32 = 0;
        let ok = unsafe { dec(cache.0, coll_tag, &mut vp, &mut vc, &mut ip, &mut ic) };
        if ok == 0 {
            return None;
        }
        let verts = if vp.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(vp, vc as usize * 3).to_vec() } };
        let indices = if ip.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(ip, ic as usize).to_vec() } };
        if !vp.is_null() {
            unsafe { free(vp as *mut u8) };
        }
        if !ip.is_null() {
            unsafe { free(ip as *mut u8) };
        }
        Some((verts, indices))
    }

    /// Resolve an object/primary tag to its physics_model tag (0xFFFFFFFF none).
    pub fn resolve_phmo(&self, cache: CacheHandle, primary_tag: u32) -> u32 {
        self.tag_resolve_phmo.map_or(0xFFFF_FFFF, |f| unsafe { f(cache.0, primary_tag) })
    }

    /// Decode a physics model into (positions, indices) triangle soup.
    pub fn phmo_geometry(&self, cache: CacheHandle, phmo_tag: u32) -> Option<(Vec<f32>, Vec<u32>)> {
        let (Some(dec), Some(free)) = (self.phmo_decode, self.phmo_free) else { return None };
        let mut vp: *mut f32 = std::ptr::null_mut();
        let mut vc: u32 = 0;
        let mut ip: *mut u32 = std::ptr::null_mut();
        let mut ic: u32 = 0;
        let ok = unsafe { dec(cache.0, phmo_tag, &mut vp, &mut vc, &mut ip, &mut ic) };
        if ok == 0 {
            return None;
        }
        let verts = if vp.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(vp, vc as usize * 3).to_vec() } };
        let indices = if ip.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(ip, ic as usize).to_vec() } };
        if !vp.is_null() {
            unsafe { free(vp as *mut u8) };
        }
        if !ip.is_null() {
            unsafe { free(ip as *mut u8) };
        }
        Some((verts, indices))
    }

    /// Enumerate scenario trigger volumes (kill/safe/plain oriented boxes).
    pub fn trigger_volumes(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<TriggerVolume> {
        let Some(f) = self.scnr_enum_triggers else { return Vec::new() };
        const STRIDE: usize = 100;
        const MAX: u32 = 512;
        let mut buf = vec![0u8; STRIDE * MAX as usize];
        let n = unsafe { f(cache.0, scnr_tag, buf.as_mut_ptr(), MAX) }.min(MAX) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let b = i * STRIDE;
            let u = |o: usize| u32::from_le_bytes([buf[b + o], buf[b + o + 1], buf[b + o + 2], buf[b + o + 3]]);
            let f = |o: usize| f32::from_le_bytes([buf[b + o], buf[b + o + 1], buf[b + o + 2], buf[b + o + 3]]);
            out.push(TriggerVolume {
                category: u(0),
                pos: [f(8), f(12), f(16)],
                fwd: [f(20), f(24), f(28)],
                up: [f(32), f(36), f(40)],
                ext: [f(44), f(48), f(52)],
            });
        }
        out
    }

    /// every soft ceiling of every structure design (sddt) the scenario
    /// references, with its world-space triangles. Empty when the native library predates the
    /// export or the map has none (Halo 4 caches, non-Reach).
    pub fn soft_ceilings(&self, cache: CacheHandle, scnr_tag: u32) -> Vec<SoftCeiling> {
        let (Some(fc), Some(ft)) = (self.sddt_enum_soft_ceilings, self.sddt_enum_soft_ceiling_tris) else {
            return Vec::new();
        };
        const STRIDE: usize = 64;
        let n = unsafe { fc(cache.0, scnr_tag, std::ptr::null_mut(), 0) } as usize;
        if n == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; STRIDE * n];
        let n = (unsafe { fc(cache.0, scnr_tag, buf.as_mut_ptr(), n as u32) } as usize).min(n);
        let nt = unsafe { ft(cache.0, scnr_tag, std::ptr::null_mut(), 0) } as usize;
        let mut xyz = vec![0f32; nt * 9];
        let nt = if nt > 0 { (unsafe { ft(cache.0, scnr_tag, xyz.as_mut_ptr(), nt as u32) } as usize).min(nt) } else { 0 };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let b = i * STRIDE;
            let u = |o: usize| u32::from_le_bytes([buf[b + o], buf[b + o + 1], buf[b + o + 2], buf[b + o + 3]]);
            let name_bytes = &buf[b + 24..b + 64];
            let name_len = name_bytes.iter().position(|&c| c == 0).unwrap_or(name_bytes.len());
            let name = String::from_utf8_lossy(&name_bytes[..name_len]).into_owned();
            let (start, count) = (u(8) as usize, u(12) as usize);
            let mut tris = Vec::with_capacity(count);
            for t in start..(start + count).min(nt) {
                let f = &xyz[t * 9..t * 9 + 9];
                tris.push([[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]]);
            }
            out.push(SoftCeiling { name, kind: u(0), flags: u(4), tris });
        }
        out
    }

    // ---- bsp ----
    pub fn open_bsp(&self, cache: CacheHandle, sbsp_tag: u32) -> Result<BspHandle> {
        let h = unsafe { (self.bsp_open)(cache.0, sbsp_tag) };
        if h == 0 {
            bail!("ZH_BSP_OpenBsp failed for tag {sbsp_tag:#x}");
        }
        Ok(BspHandle(h))
    }
    pub fn close_bsp(&self, h: BspHandle) {
        unsafe { (self.bsp_close)(h.0) }
    }
    pub fn bsp_mesh_count(&self, h: BspHandle) -> u32 {
        unsafe { (self.bsp_mesh_count)(h.0) }
    }
    pub fn bsp_mesh(&self, h: BspHandle, i: u32) -> Option<ZhBspMesh> {
        let mut m = ZhBspMesh::default();
        (unsafe { (self.bsp_get_mesh)(h.0, i, &mut m) } != 0).then_some(m)
    }
    /// Decode a BSP mesh's vertex + index buffers into owned Vecs.
    pub fn bsp_decode_geometry(&self, h: BspHandle, i: u32) -> Option<GeomBuffers> {
        decode_geom(self.bsp_decode_geom, self.bsp_free, h.0, i)
    }
    /// DIRECT-VB: the mesh's PREBUILT compressed vertex/index bytes (verbatim from the .map
    /// resource) + de-quant constants. Bytes are copied out immediately (the native pointers
    /// alias the resource mmap and must not outlive the call). None if the export is absent
    /// or the mesh has no supported geometry.
    pub fn bsp_raw_geometry(&self, h: BspHandle, i: u32) -> Option<RawGeomOwned> {
        let f = self.bsp_raw_geom?;
        let mut g: ZhRawGeom = unsafe { std::mem::zeroed() };
        if unsafe { f(h.0, i, &mut g) } == 0 { return None; }
        if g.vb_ptr.is_null() || g.vb_len == 0 { return None; }
        let vb = unsafe { std::slice::from_raw_parts(g.vb_ptr, g.vb_len as usize).to_vec() };
        let ib = if !g.ib_ptr.is_null() && g.ib_len > 0 {
            unsafe { std::slice::from_raw_parts(g.ib_ptr, g.ib_len as usize).to_vec() }
        } else { Vec::new() };
        // Copy packed fields into locals (no refs into a packed struct).
        Some(RawGeomOwned {
            vb, ib,
            vertex_format: g.vertex_format,
            vertex_stride: g.vertex_stride,
            section_vertex_count: g.section_vertex_count,
            index_stride: g.index_stride,
            index_start: g.index_start,
            index_count: g.index_count,
            is_instance: g.is_instance != 0,
            pos_min: g.pos_min,
            pos_max: g.pos_max,
            uv_min: g.uv_min,
            uv_max: g.uv_max,
            transform: g.transform,
            uniform_scale: g.uniform_scale,
            lightmap_cluster_index: g.lightmap_cluster_index,
        })
    }
    pub fn bsp_decode_normals(&self, h: BspHandle, i: u32) -> Option<Vec<u8>> {
        decode_vec(self.bsp_decode_normals?, self.bsp_free, h.0, i)
    }
    /// G1: per-vertex tangents (float3[vc]) decoded from Int16_N4 @ +0x1C (/32767),
    /// instance-rotated. Bytes = vc·3·4. None on decorator/unsupported meshes.
    pub fn bsp_decode_tangents(&self, h: BspHandle, i: u32) -> Option<Vec<u8>> {
        decode_vec(self.bsp_decode_tangents?, self.bsp_free, h.0, i)
    }
    /// G1: per-vertex binormals (float3[vc]) computed by the DLL as
    /// cross(N,T)·sign(tan.w) — recovers the handedness the tangent export discards.
    pub fn bsp_decode_binormals(&self, h: BspHandle, i: u32) -> Option<Vec<u8>> {
        decode_vec(self.bsp_decode_binormals?, self.bsp_free, h.0, i)
    }
    pub fn bsp_decode_uvs(&self, h: BspHandle, i: u32) -> Option<Vec<u8>> {
        decode_vec(self.bsp_decode_uvs?, self.bsp_free, h.0, i)
    }
    /// Lightmap UV2 stream (float2 per vertex, UNORM-decoded to [0,1]) for a BSP mesh —
    /// indexes the per-texel lightmap atlas. None on meshes without UV2.
    /// the per-instance PER-VERTEX LIGHTPROBE buffer for an instanced bsp mesh,
    /// keyed by its GeometryInstance ordinal — the baked lighting most instances have but whose
    /// normal `instance_pvl` fetch fails. Returns (raw bytes, stride, vertex count); decode with
    /// the existing PVL unpacker.
    pub fn bsp_instance_pvl_vb(&self, h: BspHandle, ordinal: u32) -> Option<(Vec<u8>, u32, u32)> {
        let f = self.bsp_instance_pvl_vb?;
        let mut p: *mut u8 = std::ptr::null_mut();
        let (mut len, mut stride, mut count) = (0u32, 0u32, 0u32);
        let ok = unsafe { f(h.0, ordinal, &mut p, &mut len, &mut stride, &mut count) } != 0;
        if !ok || p.is_null() || len == 0 {
            if !p.is_null() { unsafe { (self.bsp_free)(p) } }
            return None;
        }
        let v = unsafe { std::slice::from_raw_parts(p, len as usize).to_vec() };
        unsafe { (self.bsp_free)(p) };
        Some((v, stride, count))
    }

    pub fn bsp_decode_uv2(&self, h: BspHandle, i: u32) -> Option<Vec<f32>> {
        let f = self.bsp_decode_uv2?;
        let mut p: *mut f32 = std::ptr::null_mut();
        let mut cnt = 0u32; // FLOAT count (vertexCount*2)
        let ok = unsafe { f(h.0, i, &mut p, &mut cnt) } != 0;
        if !ok || p.is_null() || cnt == 0 {
            if !p.is_null() { unsafe { (self.bsp_free)(p as *mut u8) } }
            return None;
        }
        let v = unsafe { std::slice::from_raw_parts(p, cnt as usize).to_vec() };
        unsafe { (self.bsp_free)(p as *mut u8) };
        Some(v)
    }
    /// Per-BSP DM/SDM lightmap atlas tag ids + scene Brightness. None if the
    /// BSP has no atlas or the export is unavailable.
    pub fn lbsp_atlas(&self, cache: CacheHandle, sbsp: u32) -> Option<ZhLbspAtlas> {
        let f = self.lbsp_atlas?;
        let mut a = ZhLbspAtlas::default();
        if unsafe { f(cache.0, sbsp, &mut a) } != 0 { Some(a) } else { None }
    }
    /// Per-cluster lightmap entry: atlas submap + PVL-vs-per-pixel selector.
    pub fn lbsp_cluster_entry(&self, cache: CacheHandle, sbsp: u32, cluster: u32) -> Option<ZhLbspClusterEntry> {
        let f = self.lbsp_cluster_entry?;
        let mut e = ZhLbspClusterEntry::default();
        if unsafe { f(cache.0, sbsp, cluster, &mut e) } != 0 { Some(e) } else { None }
    }
    /// Per-INSTANCE lightmap atlas submap: the DM/SDM pool tags + submap index for
    /// a BSP instance's baked lightmap. Lets instance geometry (cluster == 0xFFFFFFFF) whose
    /// per-instance PVL fetch fails be lit from the per-texel atlas like clusters are.
    /// Returns None if the export is absent or the instance has no atlas submap.
    /// True if the DLL exports ZH_LBSP_GetInstanceAtlasSubmap (older DLLs don't).
    pub fn has_instance_atlas(&self) -> bool { self.lbsp_instance_atlas.is_some() }
    pub fn instance_atlas_submap(&self, cache: CacheHandle, sbsp: u32, ordinal: u32) -> Option<ZhLbspInstanceAtlas> {
        let f = self.lbsp_instance_atlas?;
        let mut a = ZhLbspInstanceAtlas::default();
        if unsafe { f(cache.0, sbsp, ordinal, &mut a) } != 0 { Some(a) } else { None }
    }
    pub fn bsp_material_diffuse(&self, h: BspHandle, material_index: i32) -> u32 {
        self.bsp_mat_diffuse
            .map_or(0, |f| unsafe { f(h.0, material_index) })
    }
    pub fn bsp_material_blend(&self, h: BspHandle, material_index: i32) -> u32 {
        self.bsp_mat_blend.map_or(0, |f| unsafe { f(h.0, material_index) })
    }
    /// MAT-1: per-material material_model enum for a BSP material (0..9). Native 0xFF
    /// (unresolved) maps to 1 (cook_torrance = the current unconditional path) so an
    /// unresolved read is a no-op. See `model_shader_material_model` for the enum.
    pub fn bsp_material_model(&self, h: BspHandle, material_index: i32) -> u8 {
        let v = self.bsp_mat_material_model.map_or(0xFFu32, |f| unsafe { f(h.0, material_index) }) as u8;
        if v == 0xFF { 1 } else { v }
    }
    /// ALBEDO-VARIANT: per-material albedo option (0 default / 1 two_detail / 2 black_point /
    /// 3 overlay / 4 detail_blend / 5 three_detail / 6 color_mask / 7 constant_color). Native
    /// 0xFF (unresolved / old DLL) maps to 0 (default = the single-detail path).
    pub fn bsp_material_albedo_option(&self, h: BspHandle, material_index: i32) -> u8 {
        let v = self.bsp_mat_albedo_option.map_or(0xFFu32, |f| unsafe { f(h.0, material_index) }) as u8;
        if v == 0xFF { 0 } else { v }
    }
    /// LIT-SI-3: per-material self_illumination MODE for a BSP material (0..12). Native 0xFF
    /// (unresolved / no `self_illumination` category) maps to 1 (simple = the current HMS
    /// single-composite path) so an unresolved read is a no-op. Enum: 0 off / 1 simple /
    /// 2 three_channel / 3 plasma / 4 from_albedo / 5 detail / 6 meter / 7 times_diffuse /
    /// 8 simple_with_alpha_mask / 9 multilayer / 10 palette / 11 change_color / 12 change_color_detail.
    pub fn bsp_self_illum_mode(&self, h: BspHandle, material_index: i32) -> u8 {
        let v = self.bsp_mat_self_illum_mode.map_or(0xFFu32, |f| unsafe { f(h.0, material_index) }) as u8;
        if v == 0xFF { 1 } else { v }
    }
    /// the BSP material's shader tag CLASS (e.g. "rmgl" for glass), decoded from the
    /// packed LE bytes. Empty on unresolved. The authoritative glass signal for BSP surfaces —
    /// the blend-mode resolver returns 0xFF for rmgl (no blend_mode category) so blend alone
    /// can't distinguish glass, and the old diffuse-name heuristic missed non-"*glass*" panes.
    /// render_method tag id of a BSP material (None when unresolved).
    pub fn bsp_material_shader_tag(&self, h: BspHandle, material_index: i32) -> Option<u32> {
        let t = self.bsp_mat_shader_tag.map_or(0xFFFF_FFFF, |f| unsafe { f(h.0, material_index) });
        (t != 0xFFFF_FFFF && t != 0).then_some(t)
    }
    pub fn bsp_material_shader_class(&self, h: BspHandle, material_index: i32) -> String {
        let packed = self.bsp_mat_shader_class.map_or(0, |f| unsafe { f(h.0, material_index) });
        if packed == 0 {
            return String::new();
        }
        let b = packed.to_le_bytes();
        String::from_utf8_lossy(&b).trim_end_matches('\0').to_string()
    }
    /// shader-kind classification for picking the no-diffuse fallback color.
    /// 0 = normal/unknown, 1 = `...\black` (render black), 2 = `shaders\invalid`
    /// (render near-black). Textureless shells/occluders otherwise render as pale
    /// gray and read as see-through panels.
    pub fn bsp_material_shader_kind(&self, h: BspHandle, material_index: i32) -> u32 {
        self.bsp_mat_shader_kind.map_or(0, |f| unsafe { f(h.0, material_index) })
    }
    /// Resolve a BSP material's bitmap for a named shader usage (e.g.
    /// "watercolor_texture" to detect + texture shader_water surfaces). Returns
    /// 0xFFFFFFFF when the usage isn't present or the export is unavailable.
    pub fn bsp_material_bitmap_by_usage(&self, h: BspHandle, material_index: i32, usage: &str) -> u32 {
        let Some(f) = self.bsp_mat_bitmap_by_usage else { return 0xFFFF_FFFF };
        let Ok(c) = std::ffi::CString::new(usage) else { return 0xFFFF_FFFF };
        unsafe { f(h.0, material_index, c.as_ptr()) }
    }
    /// the sampler address-mode byte the engine binds a BSP material's usage with
    /// (low nibble u, high nibble v: 0 wrap, 1 clamp, 2 mirror, 3 black border); 0xFF when absent.
    pub fn bsp_material_sampler_by_usage(&self, h: BspHandle, material_index: i32, usage: &str) -> u32 {
        let Some(f) = self.bsp_mat_sampler_by_usage else { return 0xFF };
        let Ok(c) = std::ffi::CString::new(usage) else { return 0xFF };
        unsafe { f(h.0, material_index, c.as_ptr()) }
    }

    /// Terrain-blend layers for a BSP material (4 base maps + RGBA blend mask +
    /// per-layer tiling). None when the material isn't a terrain blend or the
    /// export is unavailable — caller uses the single-diffuse path instead.
    pub fn bsp_terrain_layers(&self, h: BspHandle, material_index: i32) -> Option<TerrainLayers> {
        let f = self.bsp_terrain_layers?;
        let mut buf = vec![0u8; TERRAIN_LAYERS_SIZE];
        let ok = unsafe { f(h.0, material_index, buf.as_mut_ptr()) };
        if ok == 0 {
            return None;
        }
        let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        let f32_at = |o: usize| f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
        if u32_at(20) == 0 {
            return None; // IsTerrainBlend == 0
        }
        // Native ZH_TerrainLayers layout (Pack=1, 364B). Byte offsets:
        //   BaseMap0..3 @0..16, BlendMap @16, IsTerrainBlend @20, Bump0..3 @24..40,
        //   Detail0..3 @40..56, DetailTile(8f) @56..88, BaseTile(8f) @88..120,
        //   GlobalAlbedoTint(4f) @120..136, BumpTile(8f) @136..168,
        //   DetailBumpMap0..3 @168..184, DetailBumpTile(8f) @184..216,
        //   BlendXform(4f) @216..232, BaseOffset(8f) @232..264,
        //   DetailOffset(8f) @264..296, BumpOffset(8f) @296..328,
        //   DetailBumpOffset(8f) @328..360, AuthoredActiveMask @360.
        let base = [u32_at(0), u32_at(4), u32_at(8), u32_at(12)];
        let detail = [u32_at(40), u32_at(44), u32_at(48), u32_at(52)];
        let blend = u32_at(16);
        // A scale of 0/NaN means "field absent" → engine identity 1.0. Offsets
        // legitimately can be 0, so only NaN-guard those (default 0.0).
        let scale_at = |o: usize| {
            let v = f32_at(o);
            if v.is_finite() && v.abs() > 1e-4 { v } else { 1.0 }
        };
        let off_at = |o: usize| {
            let v = f32_at(o);
            if v.is_finite() { v } else { 0.0 }
        };
        let mut base_tile = [[1.0f32, 1.0]; 4];
        let mut detail_tile = [[1.0f32, 1.0]; 4];
        let mut base_offset = [[0.0f32, 0.0]; 4];
        let mut detail_offset = [[0.0f32, 0.0]; 4];
        for l in 0..4 {
            detail_tile[l] = [scale_at(56 + l * 8), scale_at(56 + l * 8 + 4)];
            base_tile[l] = [scale_at(88 + l * 8), scale_at(88 + l * 8 + 4)];
            base_offset[l] = [off_at(232 + l * 8), off_at(232 + l * 8 + 4)];
            detail_offset[l] = [off_at(264 + l * 8), off_at(264 + l * 8 + 4)];
        }
        // blend_map_xform: default identity (1,1,0,0) when the DLL left it zeroed
        // (older struct) — sampling at uv*0+0 would collapse the mask to one texel.
        let bx = [f32_at(216), f32_at(220), f32_at(224), f32_at(228)];
        let blend_xform = if bx[0].abs() > 1e-4 || bx[1].abs() > 1e-4 {
            [bx[0], bx[1], bx[2], bx[3]]
        } else {
            [1.0, 1.0, 0.0, 0.0]
        };
        // global_albedo_tint: default (1,1,1,1) no-op multiply when absent/zeroed.
        let gt = [f32_at(120), f32_at(124), f32_at(128), f32_at(132)];
        let global_tint = if gt[0] > 1e-4 || gt[1] > 1e-4 || gt[2] > 1e-4 {
            [gt[0], gt[1], gt[2], if gt[3] > 1e-4 { gt[3] } else { 1.0 }]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        };
        let active_mask = u32_at(360);
        // Bump (normal) maps: tags @24, tile @136, offset @296.
        let bump = [u32_at(24), u32_at(28), u32_at(32), u32_at(36)];
        let mut bump_tile = [[1.0f32, 1.0]; 4];
        let mut bump_offset = [[0.0f32, 0.0]; 4];
        for l in 0..4 {
            bump_tile[l] = [scale_at(136 + l * 8), scale_at(136 + l * 8 + 4)];
            bump_offset[l] = [off_at(296 + l * 8), off_at(296 + l * 8 + 4)];
        }
        // MAT-14: detail_bump tags @168, tile @184, offset @328 (see the layout note above).
        let detail_bump = [u32_at(168), u32_at(172), u32_at(176), u32_at(180)];
        let mut detail_bump_tile = [[1.0f32, 1.0]; 4];
        let mut detail_bump_offset = [[0.0f32, 0.0]; 4];
        for l in 0..4 {
            detail_bump_tile[l] = [scale_at(184 + l * 8), scale_at(184 + l * 8 + 4)];
            detail_bump_offset[l] = [off_at(328 + l * 8), off_at(328 + l * 8 + 4)];
        }
        // MAT-15: distance_blend_base fields appended at 364. u32::MAX / zeros when the
        // DLL is older (buffer stayed zeroed → blend_type 0 = morph = no-op either way).
        let blend_type = u32_at(364);
        let blend_slope = off_at(368);
        let blend_offset = off_at(372);
        let mut blend_target = [[0.0f32; 4]; 4];
        let mut blend_max = [0.0f32; 4];
        for l in 0..4 {
            let o = 376 + l * 16;
            blend_target[l] = [f32_at(o), f32_at(o + 4), f32_at(o + 8), f32_at(o + 12)];
            blend_max[l] = off_at(440 + l * 4);
        }
        Some(TerrainLayers {
            base,
            detail,
            blend,
            is_terrain: true,
            base_tile,
            base_offset,
            detail_tile,
            detail_offset,
            blend_xform,
            global_tint,
            active_mask,
            bump,
            bump_tile,
            bump_offset,
            detail_bump,
            detail_bump_tile,
            detail_bump_offset,
            blend_type,
            blend_slope,
            blend_offset,
            blend_target,
            blend_max,
        })
    }

    /// The authored V-scroll RATE (tiles/sec) of a material's animated overlays —
    /// used for waterfalls (a `shader` rmsh with `Type=6 TranslationY` overlays).
    /// Returns the rate as a POSITIVE magnitude (caller negates for downward flow);
    /// None when the material has no Y-scroll overlay or the export is unavailable.
    /// Engine rate = 1/TimePeriod; prefers the `self_illum_map` overlay (the visible
    /// flow layer), else any TranslationY overlay. (ZH_MaterialOverlay is 60B:
    /// Type@0, InputSid@4, RangeSid@8, TimePeriod@12, InputName[32]@16, ...)
    pub fn bsp_material_yscroll(&self, h: BspHandle, material_index: i32) -> Option<f32> {
        let f = self.bsp_material_overlays?;
        const STRIDE: usize = 60;
        const CAP: i32 = 8;
        let mut buf = vec![0u8; STRIDE * CAP as usize];
        let n = unsafe { f(h.0, material_index, buf.as_mut_ptr(), CAP) };
        if n <= 0 {
            return None;
        }
        let n = (n as usize).min(CAP as usize);
        let mut any_rate: Option<f32> = None;
        let mut selfillum_rate: Option<f32> = None;
        for i in 0..n {
            let o = i * STRIDE;
            let ty = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            if ty != 6 {
                continue; // 6 = TranslationY
            }
            let period = f32::from_le_bytes([buf[o + 12], buf[o + 13], buf[o + 14], buf[o + 15]]);
            if !(period.is_finite() && period > 1e-3) {
                continue;
            }
            let rate = 1.0 / period;
            // InputName[32] at +16 — prefer the self_illum layer (the visible flow).
            let name_end = buf[o + 16..o + 48].iter().position(|&b| b == 0).unwrap_or(32);
            let name = String::from_utf8_lossy(&buf[o + 16..o + 16 + name_end]).to_lowercase();
            if name.contains("self_illum") {
                selfillum_rate = Some(rate);
            }
            any_rate.get_or_insert(rate);
        }
        selfillum_rate.or(any_rate)
    }

    /// Per-material WATER wave scroll rates (layerA, layerB), in the engine's
    /// `time_warp` units. The rmt2 `time_warp` real is authored 0 on every map — the real
    /// per-map rate is `Σ 1/TimePeriod` over the material's `TranslationX(5)`/`TranslationY(6)`
    /// animation overlays (RE_water_rate.md: mat54 = 1/12 exactly). Routed to layer A (the
    /// wave_displacement layer) vs layer B (wave_slope) by the overlay's InputName. Returns
    /// (0,0) when the material authors no translation overlay → genuinely STATIC water.
    pub fn bsp_material_water_rates(&self, h: BspHandle, material_index: i32) -> (f32, f32) {
        let Some(f) = self.bsp_material_overlays else { return (0.0, 0.0) };
        const STRIDE: usize = 60;
        const CAP: i32 = 8;
        let mut buf = vec![0u8; STRIDE * CAP as usize];
        let n = unsafe { f(h.0, material_index, buf.as_mut_ptr(), CAP) };
        if n <= 0 { return (0.0, 0.0); }
        let n = (n as usize).min(CAP as usize);
        // route the two wave layers by the raw InputNameSid (ZH_MaterialOverlay +4 — already
        // in the buffer; the resolved InputName@+16 is empty because the water shader param names
        // are engine-GLOBAL stringids, absent from the map-local blob). The engine-global INDEX is
        // the low 16 bits of the sid (RE2_water_layers.md §1.2):
        //   layer A = wave_displacement_array_xform 0x8311 / time_warp 0x82CA
        //   layer B = wave_slope_array_xform        0x8312 / time_warp_aux 0x82DE
        // EVERY other Type-5/6 overlay (foam / watercolor / bump / subwave / detail) is NOT a wave
        // layer — routing them in was the "sum-all" over-count. rate = 1/TimePeriod.
        const A_XFORM: u32 = 0x8311; // wave_displacement_array_xform (33553)
        const A_TIME:  u32 = 0x82CA; // time_warp (33482)
        const B_XFORM: u32 = 0x8312; // wave_slope_array_xform (33554)
        const B_TIME:  u32 = 0x82DE; // time_warp_aux (33502)
        let diag = std::env::var("HMS_WATERDIAG").is_ok();
        let (mut a, mut b) = (0.0f32, 0.0f32);
        let (mut main, mut slope) = (0.0f32, 0.0f32); // fallback accumulators (old heuristic)
        for i in 0..n {
            let o = i * STRIDE;
            let ty = u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            let sid = (i32::from_le_bytes([buf[o + 4], buf[o + 5], buf[o + 6], buf[o + 7]]) as u32) & 0xFFFF;
            let period = f32::from_le_bytes([buf[o + 12], buf[o + 13], buf[o + 14], buf[o + 15]]);
            if diag {
                let name_end = buf[o + 16..o + 48].iter().position(|&x| x == 0).unwrap_or(32);
                let name = String::from_utf8_lossy(&buf[o + 16..o + 16 + name_end]);
                eprintln!("HMS_WATERDIAG mat={} overlay#{} ty={} sid=0x{:04X} period={:.3} name='{}'", material_index, i, ty, sid, period, name);
            }
            if ty != 5 && ty != 6 { continue; } // 5 = TranslationX, 6 = TranslationY
            if !(period.is_finite() && period > 1e-3) { continue; }
            let rate = 1.0 / period;
            if diag {
                let name_end = buf[o + 16..o + 48].iter().position(|&x| x == 0).unwrap_or(32);
                let name = String::from_utf8_lossy(&buf[o + 16..o + 16 + name_end]);
                eprintln!("HMS_WATERDIAG mat={} overlay ty={} sid=0x{:04X} period={:.3} rate={:.4} name='{}'",
                    material_index, ty, sid, period, rate, name);
            }
            match sid {
                A_XFORM | A_TIME => a += rate,
                B_XFORM | B_TIME => b += rate,
                _ => {} // foam/watercolor/bump/subwave/detail — not a wave layer, drop
            }
            // fallback bucket (only used if SID routing finds nothing)
            let name_end = buf[o + 16..o + 48].iter().position(|&x| x == 0).unwrap_or(32);
            let name = String::from_utf8_lossy(&buf[o + 16..o + 16 + name_end]).to_lowercase();
            if name.contains("slope") { slope += rate; } else { main += rate; }
        }
        // If SID routing produced a real layer-A rate, trust it (drops foam/bump correctly). Else
        // the map's sids didn't match (version skew) → fall back to the old sum-all heuristic so
        // water still animates (RE2_water_layers.md "order-preserving fallback").
        if a > 0.0 || b > 0.0 {
            (a, if b > 0.0 { b } else { a * 0.85 })
        } else {
            (main, if slope > 0.0 { slope } else { main * 0.85 })
        }
    }

    /// Per-shader animated UV SCROLL rate (tiles/sec) for a render_model section —
    /// drives sky cloud/atmosphere drift. Sums TranslationX(Type5)/Y(Type6)
    /// overlays: rate = (OutMax−OutMin)/TimePeriod, else 1/TimePeriod. [0,0] when the
    /// section has no translation overlay. ZH_ShaderOverlay is 148B: Type@0,
    /// TimePeriod@12, OutMin@60, OutMax@64.
    /// Raw animated-parameter ("overlay") records of a model shader: (type, driven arg name, period,
    /// out_min, out_max, func_type). Type 5/6 = TranslationX/Y. Empty when the export is missing.
    pub fn model_shader_overlays(&self, h: ModelHandle, shader: i32) -> Vec<(i32, String, f32, f32, f32, u8)> {
        let (Some(cnt), Some(at)) = (self.mmp_overlay_count, self.mmp_overlay_at) else { return Vec::new(); };
        let n = unsafe { cnt(h.0, shader) };
        let mut out = Vec::new();
        if n <= 0 { return out; }
        let arg_names: Vec<String> = self.model_shader_constants(h, shader).map(|c| c.dump_args().into_iter().map(|a| a.1).collect()).unwrap_or_default();
        let mut buf = [0u8; 148];
        let rd_i32 = |b: &[u8], o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let rd_f32 = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        for i in 0..n.min(64) {
            let ok = unsafe { at(h.0, shader, i, buf.as_mut_ptr()) };
            if ok == 0 { continue; }
            let name_end = buf[16..48].iter().position(|&c| c == 0).unwrap_or(32);
            let name = String::from_utf8_lossy(&buf[16..16 + name_end]).to_string();
            // FF-ROUTING: the DLL resolves the driven rmt2 ARGUMENT index via the postprocess routing table.
            let arg_idx = u32::from_le_bytes([buf[140], buf[141], buf[142], buf[143]]);
            let name = if name.is_empty() && arg_idx != 0xFFFF_FFFF {
                arg_names.get(arg_idx as usize).cloned().unwrap_or_else(|| format!("arg{arg_idx}"))
            } else { name };
            out.push((rd_i32(&buf, 0), name, rd_f32(&buf, 12), rd_f32(&buf, 60), rd_f32(&buf, 64), buf[48]));
        }
        out
    }

    /// Per-PARAMETER UV scroll rate (units/sec) from the TranslationX/Y overlays whose driven arg
    /// name starts with `param` (e.g. "noise_map_a"); [0,0] when that parameter is not animated.
    pub fn model_shader_scroll_param(&self, h: ModelHandle, shader: i32, param: &str) -> [f32; 2] {
        let mut scroll = [0.0f32, 0.0f32];
        for (ty, name, period, omin, omax, _) in self.model_shader_overlays(h, shader) {
            if ty != 5 && ty != 6 { continue; }
            if !name.starts_with(param) { continue; }
            if !(period.is_finite() && period > 1e-3) { continue; }
            let amp = omax - omin;
            // Sign: see shader_scroll_param_by_tag (engine offset direction is opposite to uv + rate*t).
            let rate = -(if amp.is_finite() && amp.abs() > 1e-6 { amp / period } else { 1.0 / period });
            if !rate.is_finite() { continue; }
            if ty == 5 { scroll[0] += rate; } else { scroll[1] += rate; }
        }
        [scroll[0].clamp(-4.0, 4.0), scroll[1].clamp(-4.0, 4.0)]
    }

    pub fn model_shader_scroll(&self, h: ModelHandle, shader: i32) -> [f32; 2] {
        let (Some(cnt), Some(at)) = (self.mmp_overlay_count, self.mmp_overlay_at) else {
            return [0.0, 0.0];
        };
        let n = unsafe { cnt(h.0, shader) };
        if n <= 0 {
            return [0.0, 0.0];
        }
        let mut buf = [0u8; 148];
        let mut scroll = [0.0f32, 0.0f32];
        let rd_i32 = |b: &[u8], o: usize| i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let rd_f32 = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        for i in 0..n.min(16) {
            let ok = unsafe { at(h.0, shader, i, buf.as_mut_ptr()) };
            if ok == 0 {
                continue;
            }
            let ty = rd_i32(&buf, 0);
            if ty != 5 && ty != 6 {
                continue; // 5 = TranslationX, 6 = TranslationY
            }
            let period = rd_f32(&buf, 12);
            if !(period.is_finite() && period > 1e-3) {
                continue;
            }
            let amp = rd_f32(&buf, 64) - rd_f32(&buf, 60); // OutMax − OutMin
            let rate = if amp.is_finite() && amp.abs() > 1e-6 { amp / period } else { 1.0 / period };
            if !rate.is_finite() {
                continue;
            }
            if ty == 5 { scroll[0] += rate; } else { scroll[1] += rate; }
        }
        // Clamp to sane cloud speeds so a mis-decoded curve can't fling UVs.
        [scroll[0].clamp(-1.0, 1.0), scroll[1].clamp(-1.0, 1.0)]
    }

    /// Enumerate a BSP's runtime decals (oriented textured quads) for a scenario.
    /// Empty when the map has no decals or the export is unavailable.
    pub fn decals(&self, cache: CacheHandle, sbsp_tag: u32, scnr_tag: u32) -> Vec<DecalInstance> {
        let (Some(get), Some(free)) = (self.bsp_enum_decals, self.bsp_free_decals) else {
            return Vec::new();
        };
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut count: u32 = 0;
        let ok = unsafe { get(cache.0, sbsp_tag, scnr_tag, &mut buf, &mut count) };
        if ok == 0 || buf.is_null() || count == 0 {
            if !buf.is_null() {
                unsafe { free(buf) };
            }
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let base = unsafe { buf.add(i * DECAL_INSTANCE_SIZE) };
            let f = |off: usize| -> f32 {
                unsafe { (base.add(off) as *const f32).read_unaligned() }
            };
            let u = |off: usize| -> u32 {
                unsafe { (base.add(off) as *const u32).read_unaligned() }
            };
            let b = |off: usize| -> u8 { unsafe { *base.add(off) } };
            // per-decal tint: the rmt2 FloatConstants carry the decal's colour, and
            // which slot is the colour is named in the rmt2 param-name table. Scan the 4
            // surfaced param names (offsets 164/196/228/260, 32B each) for a colour slot;
            // read that FloatConstant's RGB (floatConst0..3 at 72/88/104/120). Without this
            // a white-swatch decal (coloured signage / yellow "45") renders colourless.
            let read_name = |off: usize| -> String {
                let mut s = String::new();
                for k in 0..32 {
                    let c = unsafe { *base.add(off + k) };
                    if c == 0 { break; }
                    s.push(c as char);
                }
                s.to_ascii_lowercase()
            };
            let fc_rgb = |slot: usize| -> [f32; 3] {
                let o = 72 + slot * 16;
                [f(o), f(o + 4), f(o + 8)]
            };
            let fc_count = unsafe { (base.add(160) as *const i32).read_unaligned() }.max(0) as usize;
            let name_offs = [164usize, 196, 228, 260];
            let mut tint = None;
            let mut is_vector = false;
            for slot in 0..fc_count.min(4) {
                let nm = read_name(name_offs[slot]);
                if tint.is_none() && (nm.contains("tint_color") || nm.contains("albedo_color") || nm == "color") {
                    let c = fc_rgb(slot);
                    if c.iter().all(|v| v.is_finite()) {
                        tint = Some(c);
                    }
                }
                // SDF glyph decals (numbers/text) author these rmt2 params.
                if nm.contains("vector_sharpness") || nm.contains("antialias_tweak") {
                    is_vector = true;
                }
            }
            // bitmapTagId2 @344 = the vector_map (SDF glyph field) for vector/text decals.
            let vector_map = u(344);
            out.push(DecalInstance {
                pos: [f(0), f(4), f(8)],
                facing: [f(12), f(16), f(20)],
                u_axis: [f(24), f(28), f(32)],
                v_axis: [f(36), f(40), f(44)],
                half: [f(48), f(52)],
                bitmap: u(56),
                blend_mode: unsafe { (base.add(68) as *const i32).read_unaligned() },
                scale_x_default: [f(292), f(296)],
                scale_y_default: [f(300), f(304)],
                depth_bias: f(316),
                scale_x_mul: f(320),
                has_sprite: b(324) != 0,
                sprite: [f(328), f(332), f(336), f(340)],
                sort_layer: b(356),
                clamp_angle: f(308),
                cull_angle: f(312),
                tint,
                is_vector,
                vector_map,
            });
        }
        unsafe { free(buf) };
        out
    }

    /// Count of preplaced (baked, artist-painted) decals on this sbsp. 0 if the
    /// export is absent. Used to diagnose the "whole wall covered" coverage gap.
    pub fn preplaced_decal_count(&self, cache: CacheHandle, sbsp_tag: u32) -> u32 {
        match self.bsp_preplaced_count {
            Some(f) => unsafe { f(cache.0, sbsp_tag) },
            None => 0,
        }
    }

    /// Enumerate a BSP's PREPLACED decals — the baked artist-painted surface-coating
    /// decals (the "whole wall covered" ice/frost). Distinct from `decals()` (runtime
    /// scenario decals). Each carries a world Position, resolved bitmap(s), blend mode,
    /// sprite sub-rect, scale, and the baked-geometry mesh slice (index/vertex start+count)
    /// into the sbsp preplaced-decal geometry buffer. Empty if the export is unavailable.
    pub fn preplaced_decals(&self, cache: CacheHandle, sbsp_tag: u32, scnr_tag: u32) -> Vec<PreplacedDecal> {
        let (Some(get), Some(free)) = (self.bsp_enum_preplaced, self.bsp_free_preplaced) else {
            return Vec::new();
        };
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut count: u32 = 0;
        let ok = unsafe { get(cache.0, sbsp_tag, scnr_tag, &mut buf, &mut count) };
        if ok == 0 || buf.is_null() || count == 0 {
            if !buf.is_null() { unsafe { free(buf) }; }
            return Vec::new();
        }
        const STRIDE: usize = 80; // sizeof(ZH_PreplacedDecal)
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let base = unsafe { buf.add(i * STRIDE) };
            let f = |off: usize| -> f32 { unsafe { (base.add(off) as *const f32).read_unaligned() } };
            let i32r = |off: usize| -> i32 { unsafe { (base.add(off) as *const i32).read_unaligned() } };
            let u32r = |off: usize| -> u32 { unsafe { (base.add(off) as *const u32).read_unaligned() } };
            let i16r = |off: usize| -> i16 { unsafe { (base.add(off) as *const i16).read_unaligned() } };
            out.push(PreplacedDecal {
                pos: [f(0), f(4), f(8)],           // +0x00
                decs: i32r(12),                    // +0x0C
                bitmap: u32r(16),                  // +0x10
                property_index: i32r(20),          // +0x14
                blend_mode: i32r(24),              // +0x18
                sprite: [f(28), f(32), f(36), f(40)], // +0x1C..0x28 (Umin,Vmin,Usize,Vsize)
                scale_x_mul: f(44),                // +0x2C
                scale_x_default: [f(48), f(52)],   // +0x30,0x34
                index_start: i16r(56),             // +0x38
                index_count: i16r(58),             // +0x3A
                vertex_start: i16r(60),            // +0x3C
                vertex_count: i16r(62),            // +0x3E
                ref_count: i16r(64),               // +0x40
                def_block_index: i16r(66),         // +0x42 (Pad0) = REF def block index (mesh selector)
                bitmap2: u32r(68),                 // +0x44 Textures[1] (mask)
            });
        }
        unsafe { free(buf) };
        out
    }

    /// Decode the sbsp's PREPLACED-decal baked geometry buffer. The buffer holds one VB/IB
    /// mesh per decal-material group; each decal's REF row selects a mesh via `def_block_index`
    /// and slices it with (vertex_start,vertex_count,index_start,index_count) LOCAL to that mesh.
    /// Returns the concatenated world-space vertices (5 floats/vertex: pos.xyz, u, v), the RAW
    /// (mesh-local) index values, and per-mesh vertex/index base offsets into those flat arrays.
    /// None when the export is absent or the sbsp has no preplaced geometry.
    pub fn preplaced_geometry(&self, cache: CacheHandle, sbsp_tag: u32) -> Option<PreplacedGeometry> {
        let (get, free) = (self.bsp_decode_pp_geom?, self.bsp_free_pp_geom?);
        let mut verts: *mut f32 = std::ptr::null_mut();
        let mut vfloat: u32 = 0;
        let mut idx: *mut u32 = std::ptr::null_mut();
        let mut icount: u32 = 0;
        let mut vcount: u32 = 0;
        let mut mvb: *mut u32 = std::ptr::null_mut();
        let mut mib: *mut u32 = std::ptr::null_mut();
        let mut mcount: u32 = 0;
        let ok = unsafe {
            get(cache.0, sbsp_tag, &mut verts, &mut vfloat, &mut idx, &mut icount, &mut vcount,
                &mut mvb, &mut mib, &mut mcount)
        };
        if ok == 0 || verts.is_null() || idx.is_null() || vfloat == 0 || icount == 0 {
            unsafe { free(verts, idx, mvb, mib) };
            return None;
        }
        let v = unsafe { std::slice::from_raw_parts(verts, vfloat as usize) }.to_vec();
        let i = unsafe { std::slice::from_raw_parts(idx, icount as usize) }.to_vec();
        let mvb_v = if mvb.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(mvb, mcount as usize) }.to_vec() };
        let mib_v = if mib.is_null() { Vec::new() } else { unsafe { std::slice::from_raw_parts(mib, mcount as usize) }.to_vec() };
        unsafe { free(verts, idx, mvb, mib) };
        Some(PreplacedGeometry {
            verts: v,
            indices: i,
            vertex_count: vcount,
            mesh_vert_base: mvb_v,
            mesh_idx_base: mib_v,
        })
    }

    // ---- model ----
    pub fn open_model(&self, cache: CacheHandle, mode_tag: u32) -> Result<ModelHandle> {
        let h = unsafe { (self.mmp_open)(cache.0, mode_tag) };
        if h == 0 {
            bail!("ZH_MMP_OpenModel failed for tag {mode_tag:#x}");
        }
        Ok(ModelHandle(h))
    }
    pub fn close_model(&self, h: ModelHandle) {
        unsafe { (self.mmp_close)(h.0) }
    }
    pub fn model_section_count(&self, h: ModelHandle) -> u32 {
        unsafe { (self.mmp_section_count)(h.0) }
    }
    pub fn model_section(&self, h: ModelHandle, i: u32) -> Option<ZhModelSection> {
        let mut s = ZhModelSection::default();
        (unsafe { (self.mmp_get_section)(h.0, i, &mut s) } != 0).then_some(s)
    }
    /// Page-cache profiling: (hits, inflates, inflate_bytes, inflate_ns, stores) since the
    /// last call (counters reset on read). All zero if the DLL predates the export.
    pub fn page_stats(&self) -> (u64, u64, u64, u64, u64) {
        let Some(f) = self.mbp_page_stats else { return (0, 0, 0, 0, 0) };
        let (mut h, mut i, mut b, mut n, mut s) = (0u64, 0u64, 0u64, 0u64, 0u64);
        unsafe { f(&mut h, &mut i, &mut b, &mut n, &mut s) };
        (h, i, b, n, s)
    }

    /// Total submesh (material segment) count for the model (across all sections).
    pub fn model_submesh_count(&self, h: ModelHandle) -> u32 {
        self.mmp_submesh_count.map_or(0, |f| unsafe { f(h.0) })
    }

    /// whether a render-model SECTION belongs to permutation 0 of its region (the default,
    /// non-damaged variant). Vehicles/objects pack every damage state + variant as separate
    /// sections; rendering all of them overlaps damaged+pristine geometry. Defaults to `true` when
    /// the export is missing (older DLL) so nothing is hidden by accident.
    pub fn model_section_allowed(&self, h: ModelHandle, section: u32) -> bool {
        self.mmp_section_allowed.map_or(true, |f| unsafe { f(h.0, section) } != 0)
    }

    /// enumerate an object's hlmt attachments (turrets etc.) — each with the child
    /// render_model resolved and its model-space transform composed. Empty when the object has none
    /// or the export is missing.
    pub fn enumerate_attachments(&self, cache: CacheHandle, obj_tag: u32) -> Vec<ZhAttachment> {
        let Some(f) = self.mmp_enum_attachments else { return Vec::new() };
        let n = unsafe { f(cache.0, obj_tag, std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Vec::new();
        }
        let n = (n as usize).min(64);
        let mut buf: Vec<ZhAttachment> = vec![
            ZhAttachment { child_mode: 0, child_obj: 0, marker_sid: 0, variant_name_sid: 0, variant_index: 0, pos: [0.0; 3], rot: [0.0, 0.0, 0.0, 1.0], scale: 1.0 };
            n
        ];
        let got = unsafe { f(cache.0, obj_tag, buf.as_mut_ptr(), n as i32) };
        buf.truncate((got.max(0) as usize).min(n));
        buf
    }

    /// per-section allow mask for a specific model VARIANT of `obj_tag` (e.g. rocket warthog).
    /// `variant_sid` is the hlmt model-variant Name stringId (from the forge palette's variant block
    /// @0x14, == `ForgePaletteEntry::variant_name_sid`). Returns a per-section bool where `true` =
    /// the section belongs to this variant's chosen region-permutations. Empty vec ⇒ variant not
    /// found / sid 0 / export missing → caller keeps its default (all sections / perm[0]) behaviour.
    pub fn variant_section_mask(&self, cache: CacheHandle, obj_tag: u32, variant_sid: u32) -> Vec<bool> {
        if variant_sid == 0 { return Vec::new(); }
        let Some(f) = self.mmp_variant_section_mask else { return Vec::new() };
        const MAX: usize = 4096;
        let mut buf = vec![0u8; MAX];
        let n = unsafe { f(cache.0, obj_tag, variant_sid, buf.as_mut_ptr(), MAX as i32) };
        if n <= 0 { return Vec::new(); }
        let n = (n as usize).min(MAX);
        buf.truncate(n);
        buf.into_iter().map(|b| b != 0).collect()
    }

    /// per-section allow mask for an object's DEFAULT hlmt variant (variant[0]) — the
    /// engine's resting appearance. Use INSTEAD of the perm[0]-of-every-region gate for vehicles
    /// whose rotor/tail/wings need the default-variant permutation selection (Falcon). Empty vec ⇒
    /// no hlmt variants / export missing → caller keeps the perm[0] behaviour.
    pub fn default_variant_section_mask(&self, cache: CacheHandle, obj_tag: u32) -> Vec<bool> {
        if obj_tag == 0 || obj_tag == 0xFFFF_FFFF { return Vec::new(); }
        let Some(f) = self.mmp_default_variant_section_mask else { return Vec::new() };
        const MAX: usize = 4096;
        let mut buf = vec![0u8; MAX];
        let n = unsafe { f(cache.0, obj_tag, buf.as_mut_ptr(), MAX as i32) };
        if n <= 0 { return Vec::new(); }
        let n = (n as usize).min(MAX);
        buf.truncate(n);
        buf.into_iter().map(|b| b != 0).collect()
    }

    /// DETERMINISTIC resting-appearance section mask. For EVERY render-model region, allows the
    /// single permutation the object's hlmt default variant (variant[0]) selects (else perm 0), plus
    /// orphan sections. Fixes vehicles that showed overlapping static+rotating wheels / leaked damage
    /// or blur geometry / dropped body parts under the old name-token heuristic. Empty vec ⇒ the obj
    /// has no hlmt variants (plain prop) or the export is missing ⇒ caller keeps `model_section_allowed`.
    pub fn resolve_default_variant_mask(&self, cache: CacheHandle, obj_tag: u32) -> Vec<bool> {
        if obj_tag == 0 || obj_tag == 0xFFFF_FFFF { return Vec::new(); }
        let Some(f) = self.mmp_resolve_default_variant_mask else { return Vec::new() };
        const MAX: usize = 4096;
        let mut buf = vec![0u8; MAX];
        let n = unsafe { f(cache.0, obj_tag, buf.as_mut_ptr(), MAX as i32) };
        if n <= 0 { return Vec::new(); }
        let n = (n as usize).min(MAX);
        buf.truncate(n);
        buf.into_iter().map(|b| b != 0).collect()
    }
    /// Submesh `i` — its owning section, shader index, and index range within that section.
    pub fn model_submesh(&self, h: ModelHandle, i: u32) -> Option<ZhModelSubmesh> {
        let f = self.mmp_get_submesh?;
        let mut s = ZhModelSubmesh::default();
        (unsafe { f(h.0, i, &mut s) } != 0).then_some(s)
    }
    pub fn model_decode_geometry(&self, h: ModelHandle, i: u32) -> Option<GeomBuffers> {
        decode_geom(self.mmp_decode_geom, self.mmp_free, h.0, i)
    }
    pub fn model_decode_uvs(&self, h: ModelHandle, i: u32) -> Option<Vec<u8>> {
        decode_vec(self.mmp_decode_uvs?, self.mmp_free, h.0, i)
    }
    /// per-vertex COLOR (float3 RGB in [0,1]) for a section, if it carries a
    /// vertex-color stream (vertex-gradient sky domes). None when the section has no color stream.
    pub fn model_decode_colors(&self, h: ModelHandle, i: u32) -> Option<Vec<[f32; 3]>> {
        let bytes = decode_vec(self.mmp_decode_colors?, self.mmp_free, h.0, i)?;
        if bytes.len() < 12 { return None; }
        let mut out = Vec::with_capacity(bytes.len() / 12);
        for c in bytes.chunks_exact(12) {
            let r = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let g = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            let b = f32::from_le_bytes([c[8], c[9], c[10], c[11]]);
            out.push([r, g, b]);
        }
        Some(out)
    }
    pub fn model_shader_diffuse(&self, h: ModelHandle, shader: i32) -> u32 {
        self.mmp_shader_diffuse
            .map_or(0, |f| unsafe { f(h.0, shader) })
    }
    /// Self-illum / emissive bitmap tag for a shader index (0 if none / unsupported).
    pub fn model_shader_emissive(&self, h: ModelHandle, shader: i32) -> u32 {
        self.mmp_shader_emissive
            .map_or(0, |f| unsafe { f(h.0, shader) })
    }
    /// Blend-mode index for a model shader (0=opaque, additive/alpha/etc).
    pub fn model_shader_blend(&self, h: ModelHandle, shader: i32) -> u32 {
        self.mmp_shader_blend.map_or(0, |f| unsafe { f(h.0, shader) })
    }
    /// MAT-1: per-shader material_model enum for a MODEL shader (0..9). Native returns
    /// 0xFF when unresolved (no `material_model` category / bad tag / missing export);
    /// we map that to 1 (cook_torrance) — the current unconditional shading path — so an
    /// unresolved read leaves behaviour unchanged. 0=diffuse_only 1=cook_torrance
    /// 2=two_lobe_phong 3=foliage 4=none 5=glass 6=organism 7=single_lobe_phong 8=hair
    /// 9=custom_specular.
    pub fn model_shader_material_model(&self, h: ModelHandle, shader: i32) -> u8 {
        let v = self.mmp_shader_material_model.map_or(0xFFu32, |f| unsafe { f(h.0, shader) }) as u8;
        if v == 0xFF { 1 } else { v }
    }
    /// LIT-SI-3: per-shader self_illumination MODE for a MODEL shader (0..12). Native 0xFF
    /// (unresolved / no `self_illumination` category / missing export) maps to 1 (simple =
    /// the current HMS single-composite emissive path) so an unresolved read is a no-op.
    /// Enum: 0 off / 1 simple / 2 three_channel / 3 plasma / 4 from_albedo / 5 detail /
    /// 6 meter / 7 times_diffuse / 8 simple_with_alpha_mask / 9 multilayer / 10 palette /
    /// 11 change_color / 12 change_color_detail. Model-side mirror of `bsp_self_illum_mode`.
    pub fn model_shader_self_illum_mode(&self, h: ModelHandle, shader: i32) -> u8 {
        let v = self.mmp_shader_self_illum_mode.map_or(0xFFu32, |f| unsafe { f(h.0, shader) }) as u8;
        if v == 0xFF { 1 } else { v }
    }
    /// SKY-3: is this MODEL sky shader a `sky_dome_simple` gradient-only dome?
    /// Native returns 1 = sky_dome_simple (vColor·exposure, NO texture sampler),
    /// 0 = other/textured sky template, 0xFF = unresolved (bad tag / missing export).
    /// Only `Some(true)` when definitively sky_dome_simple; `Some(false)` when
    /// definitively another template; `None` when unresolved (caller keeps its
    /// existing is_vgradient_dome heuristic and samples base+emis as before).
    pub fn model_shader_sky_class(&self, h: ModelHandle, shader: i32) -> Option<bool> {
        let f = self.mmp_shader_sky_class?;
        let v = unsafe { f(h.0, shader) } as u8;
        match v {
            1 => Some(true),
            0 => Some(false),
            _ => None, // 0xFF unresolved
        }
    }
    /// The shader's tag CLASS as a 4-byte string (e.g. "rmgl" for glass, "rmsh"
    /// for the standard shader). Empty when unresolved. Used to route forge glass
    /// panes (rmgl) to the alpha-blend pass — their blend mode resolves to the
    /// 0xFF "unresolved" sentinel (the resolver reads the rmsh layout) so class is
    /// the only reliable glass signal.
    /// the rmt2 (render_method_template) tag id for a model shader; -1 on failure. The
    /// tag NAME (via tag_name) encodes the per-category option digits — parse the self_illum
    /// number to classify change_color vs fixed vs multilayer_additive markers.
    pub fn model_shader_template_tag(&self, h: ModelHandle, shader: i32) -> i32 {
        self.mmp_shader_template.map_or(-1, |f| unsafe { f(h.0, shader) })
    }
    /// the render_method tag id a model material references (-1 = none / old DLL). Every
    /// `model_shader_*` answer is a pure function of it, so it keys the cross-model texture dedupe.
    pub fn model_shader_tag(&self, h: ModelHandle, shader: i32) -> i32 {
        self.mmp_shader_tag.map_or(-1, |f| unsafe { f(h.0, shader) })
    }
    /// a MODEL shader's bitmap for a named render-method usage ("noise_map_a", "palette",
    /// "alpha_mask_map") — feeds the palettized_plasma forcefield shader. 0xFFFF_FFFF when absent.
    pub fn model_shader_bitmap_by_usage(&self, h: ModelHandle, shader: i32, usage: &str) -> u32 {
        let Some(f) = self.mmp_shader_bitmap_by_usage else { return 0xFFFF_FFFF };
        let Ok(c) = std::ffi::CString::new(usage) else { return 0xFFFF_FFFF };
        unsafe { f(h.0, shader, c.as_ptr()) }
    }
    /// sampler address-mode byte of a model shader's usage (see bsp_material_sampler_by_usage).
    pub fn model_shader_sampler_by_usage(&self, h: ModelHandle, shader: i32, usage: &str) -> u32 {
        let Some(f) = self.mmp_shader_sampler_by_usage else { return 0xFF };
        let Ok(c) = std::ffi::CString::new(usage) else { return 0xFF };
        unsafe { f(h.0, shader, c.as_ptr()) }
    }
    pub fn model_shader_class(&self, h: ModelHandle, shader: i32) -> String {
        let packed = self.mmp_shader_class.map_or(0, |f| unsafe { f(h.0, shader) });
        if packed == 0 {
            return String::new();
        }
        let b = packed.to_le_bytes();
        String::from_utf8_lossy(&b).trim_end_matches('\0').to_string()
    }
    /// the DEFINITIVE cutout signal for a scenery/object shader — the
    /// `alpha_test_map` bitmap tag id (the same slot the BSP path reads). Returns
    /// a valid bitmap tag when the material authors alpha-test cutout (tree canopy
    /// cards, plants), else 0xFFFF_FFFF. More reliable than the base-map alpha
    /// histogram, which misses graded-alpha canopies (some Panopticon trees).
    pub fn model_shader_alpha_test(&self, h: ModelHandle, shader: i32) -> u32 {
        self.mmp_shader_alpha_test.map_or(0xFFFF_FFFF, |f| unsafe { f(h.0, shader) })
    }
    /// Authored render-method constants for a model shader (named vec4/scalar
    /// args). None when unavailable. Boxed — the struct is ~4.4 KB.
    pub fn model_shader_constants(&self, h: ModelHandle, shader: i32) -> Option<Box<ZhShaderConstants>> {
        let f = self.mmp_shader_constants?;
        let mut c = Box::new(ZhShaderConstants::zeroed());
        let ok = unsafe { f(h.0, shader, c.as_mut() as *mut ZhShaderConstants) };
        if ok != 0 { Some(c) } else { None }
    }

    /// Number of skies referenced by the scenario (scnr Skies[] block).
    pub fn sky_count(&self, cache: CacheHandle, scnr: u32) -> u32 {
        self.sky_count.map_or(0, |f| unsafe { f(cache.0, scnr) })
    }
    /// The render_model (mode) tag id for sky `idx` (scnr→scen→hlmt→mode).
    /// Returns 0xFFFFFFFF/0 when unresolved.
    pub fn sky_render_model(&self, cache: CacheHandle, scnr: u32, idx: u32) -> u32 {
        self.sky_render_model.map_or(0xFFFF_FFFF, |f| unsafe { f(cache.0, scnr, idx) })
    }
    /// Tag path/name for a tag id (lowercased). None if unavailable/empty.
    /// the tag's 4-char class code (e.g. "sbsp"), None when unmapped or the export is missing.
    pub fn tag_class(&self, cache: CacheHandle, tag: u32) -> Option<String> {
        let f = self.tag_class?;
        let v = unsafe { f(cache.0, tag) };
        if v == 0 { return None; }
        Some(String::from_utf8_lossy(&v.to_le_bytes()).to_string())
    }
    pub fn tag_name(&self, cache: CacheHandle, tag: u32) -> Option<String> {
        let f = self.tag_name?;
        let mut buf = [0u8; 256];
        let n = unsafe { f(cache.0, tag, buf.as_mut_ptr() as *mut c_char, buf.len() as i32) };
        if n <= 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..n as usize]).to_lowercase())
    }

    /// Named render-method args for a BSP material (e.g. water shader params:
    /// water_murkiness, fresnel_coefficient, slope_scaler, watercolor_coefficient).
    /// Reach packs logically-scalar water args into the RealConstant `.x`.
    pub fn bsp_material_shader_constants(&self, h: BspHandle, mat: i32) -> Option<Box<ZhShaderConstants>> {
        let f = self.bsp_shader_constants?;
        let mut c = Box::new(ZhShaderConstants::zeroed());
        let ok = unsafe { f(h.0, mat, c.as_mut() as *mut ZhShaderConstants) };
        if ok != 0 { Some(c) } else { None }
    }

    /// W22/W23 investigation probe. No-op unless HMS_UWPROBE is set (native reads it).
    pub fn bsp_underwater_probe(&self, h: BspHandle, mat: i32) {
        if let Some(f) = self.bsp_underwater_probe { unsafe { f(h.0, mat) } }
    }

    /// W22/W23: the authored underwater fog `(rgb, murkiness)` from the map's
    /// `atmosphere_globals` (atgf) tag underwater_setting block, or None when the
    /// cache has no atgf underwater setting (caller keeps its deep-colour stand-in).
    /// `rgb` is a linear, pre-exposure colour; `murkiness` is the dedicated underwater
    /// extinction (distinct from the surface `water_murkiness`).
    pub fn bsp_underwater_fog(&self, h: BspHandle) -> Option<([f32; 3], f32)> {
        let f = self.bsp_underwater_fog?;
        let (mut murk, mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let ok = unsafe { f(h.0, &mut murk, &mut r, &mut g, &mut b) };
        if ok != 0 { Some(([r, g, b], murk)) } else { None }
    }

    /// Per-material base-map UV tiling scale (engine `base_map_xform.xy`, from
    /// `ZH_BSP_GetMaterialDiffuseTiling` — reads `rmt2.TilingData[diffuseArg].XY`).
    /// (1,1) when unresolved so callers sample raw UVs. Fixes opaque BSP surfaces
    /// (boardwalk floor etc.) that authored a tile > 1 and were stretched to one sample.
    pub fn bsp_material_diffuse_tiling(&self, h: BspHandle, mat: i32) -> [f32; 2] {
        let (mut x, mut y) = (1.0f32, 1.0f32);
        if let Some(f) = self.bsp_diffuse_tiling {
            unsafe { f(h.0, mat, &mut x, &mut y); }
        }
        // NO structural cap: the engine applies base_map_xform.xy verbatim
        // (transform_texcoord = uv * xform.xy + xform.zw), and real materials tile far above 16
        // (forge_halo's far-field grass plane base_map 20, Ivory's pebble gardens 40 / grate
        // floors 32 / wood 20). A "tiles <= 16" clamp would silently untile all of them. Only
        // NaN/non-positive (and the native's own 1024 sanity cap) reject.
        if !x.is_finite() || x <= 0.0 { x = 1.0; }
        if !y.is_finite() || y <= 0.0 { y = 1.0; }
        [x, y]
    }

    // ---- lightmap (baked lighting) ----
    /// The BSP airprobe SH grid for `sbsp_tag`: one point per authored probe,
    /// each carrying the linear-HDR baked ambient (+ dominant lobe). Empty when
    /// the map has no grid or the DLL lacks the export — callers then keep flat
    /// ambient (never invent a probe).
    pub fn airprobe_grid(&self, cache: CacheHandle, sbsp_tag: u32) -> Vec<AirprobePoint> {
        let (Some(get), Some(free)) = (self.lbsp_airprobe_grid, self.lbsp_free_airprobe_grid)
        else {
            return Vec::new();
        };
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut count: u32 = 0;
        let ok = unsafe { get(cache.0, sbsp_tag, &mut buf, &mut count) };
        if ok == 0 || buf.is_null() || count == 0 {
            if !buf.is_null() {
                unsafe { free(buf) };
            }
            return Vec::new();
        }
        let mut out = Vec::with_capacity(count as usize);
        // Native array is `count` tightly-packed ZH_AirprobePoint (44B each).
        for i in 0..count as usize {
            let p = unsafe {
                (buf as *const AirprobePoint).add(i).read_unaligned()
            };
            out.push(p);
        }
        unsafe { free(buf) };
        out
    }

    /// Engine-authored per-BSP scene-brightness scalar (Lbsp+0x18). 1.0 is
    /// neutral; returns 1.0 when unavailable.
    pub fn bsp_brightness(&self, cache: CacheHandle, sbsp_tag: u32) -> f32 {
        let b = self.lbsp_brightness.map_or(1.0, |f| unsafe { f(cache.0, sbsp_tag) });
        if b.is_finite() && b > 0.0 { b } else { 1.0 }
    }

    /// Decode the per-BSP runtime decorator geometry (pre-baked world-space
    /// grass/foliage meshes). Empty when the map has no decorators or the DLL
    /// lacks the export.
    pub fn decorator_geometry(&self, cache: CacheHandle, sbsp_tag: u32) -> Vec<DecoratorMesh> {
        let (Some(get), Some(free)) = (self.bsp_deco_geom, self.bsp_free_deco) else {
            return Vec::new();
        };
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut count: u32 = 0;
        let ok = unsafe { get(cache.0, sbsp_tag, &mut buf, &mut count) };
        if ok == 0 || buf.is_null() || count == 0 {
            if !buf.is_null() {
                unsafe { free(buf, count) };
            }
            return Vec::new();
        }
        // Native descriptor (x64, 72B): 4 ptrs, 4 u32, Colors ptr@48, Sway ptr@56,
        // DecoType u32@64 (ZH_RuntimeDecoratorMesh in MapBspParser.cpp).
        const STRIDE: usize = 72;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let base = unsafe { buf.add(i * STRIDE) };
            let read_ptr = |off: usize| -> *const u8 {
                unsafe { (base.add(off) as *const *const u8).read_unaligned() }
            };
            let read_u32 = |off: usize| -> u32 {
                unsafe { (base.add(off) as *const u32).read_unaligned() }
            };
            let pos_ptr = read_ptr(0) as *const f32;
            let uv_ptr = read_ptr(8) as *const f32;
            let nrm_ptr = read_ptr(16) as *const f32;
            let idx_ptr = read_ptr(24) as *const u16;
            let vcount = read_u32(32);
            let icount = read_u32(36);
            let bitmap = read_u32(40);
            let dctr = read_u32(44);
            let col_ptr = read_ptr(48) as *const f32;
            let sway_ptr = read_ptr(56) as *const f32;
            // DECO_BUDGET_RE: InstanceCount at struct offset 68 (fills DecoType@64's
            // tail pad; 72B stride unchanged). Instances baked into this group-mesh.
            let inst_count = read_u32(68);

            let copy_f = |p: *const f32, n: usize| -> Vec<f32> {
                if p.is_null() || n == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(p, n).to_vec() }
                }
            };
            let positions = copy_f(pos_ptr, vcount as usize * 3);
            let uvs = copy_f(uv_ptr, vcount as usize * 2);
            let normals = copy_f(nrm_ptr, vcount as usize * 3);
            let colors = {
                let c = copy_f(col_ptr, vcount as usize * 3);
                if c.is_empty() { None } else { Some(c) }
            };
            let sway = {
                let s = copy_f(sway_ptr, vcount as usize * 3);
                if s.is_empty() { None } else { Some(s) }
            };
            let indices = if idx_ptr.is_null() || icount == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(idx_ptr, icount as usize).to_vec() }
            };
            out.push(DecoratorMesh {
                positions,
                uvs,
                normals,
                colors,
                sway,
                indices,
                vertex_count: vcount,
                bitmap_tag: bitmap,
                dctr_tag: dctr,
                instance_count: inst_count,
            });
        }
        unsafe { free(buf, count) };
        out
    }

    /// Fetch the per-vertex lightprobe (VMF-lobe) stream for one BSP cluster.
    /// Returns None when the cluster carries no PVL (engine-normal for most) or
    /// the DLL lacks the export. See `decode_pvl_color` for the byte decode.
    /// The scenario's atmospheric fog params (`fogg` tag). None if the map has no
    /// fog palette / null fogg ref.
    pub fn fog_params(&self, cache: CacheHandle, scnr_tag: u32) -> Option<ZhFogParams> {
        let f = self.fog_get_params?;
        let mut p: ZhFogParams = unsafe { std::mem::zeroed() };
        let ok = unsafe { f(cache.0, scnr_tag, &mut p) };
        let has = p.has_fog;
        if ok != 0 && has != 0 { Some(p) } else { None }
    }

    /// The 11 predefined engine team/player "change colors" (LINEAR RGB) read
    /// from the game_globals ('matg') tag. `found==0` means the block wasn't
    /// located (caller keeps its built-in table); entries 8..10 are always the
    /// engine-synthetic neutral/black/zombie defaults.
    pub fn change_colors(&self, cache: CacheHandle) -> Option<ZhChangeColors> {
        let f = self.globals_change_colors?;
        let mut o: ZhChangeColors = unsafe { std::mem::zeroed() };
        let ok = unsafe { f(cache.0, &mut o) };
        if ok != 0 && o.found != 0 { Some(o) } else { None }
    }

    /// The scenario's configured exposure + colour-grade tints (scnr camera
    /// exposure stops + sceg offsets). None if unsupported / walker failed.
    pub fn scenario_exposure(&self, cache: CacheHandle, scnr_tag: u32) -> Option<ZhScenarioExposure> {
        let f = self.scnr_exposure?;
        let mut o = ZhScenarioExposure::default();
        let ok = unsafe { f(cache.0, scnr_tag, &mut o) };
        if ok != 0 && o.has_exposure != 0 { Some(o) } else { None }
    }

    /// Per-INSTANCE baked per-vertex lightprobe (rocks/cliffs). Same decode as
    /// `cluster_pvl` (pvb_offset is always 0 for instances). None = this instance
    /// has no per-vertex baked lighting (single-probe/per-pixel/bogus) → caller
    /// falls back to the airprobe tint.
    pub fn instance_pvl(&self, cache: CacheHandle, sbsp_tag: u32, ordinal: u32) -> Option<ClusterPvl> {
        let get = self.lbsp_instance_pvl?;
        self.pvl_via(get, cache, sbsp_tag, ordinal)
    }

    /// Single-probe baked lighting tier for an instance (rocks/cliffs baked with the
    /// `single_probe` policy). Returns the instance's own up-facing dual-VMF ambient
    /// (linear HDR rgb), or None if this instance isn't single-probe. ~18% of Forge
    /// instances; without this they fall to the generic airprobe-centroid ambient.
    pub fn instance_probe(&self, cache: CacheHandle, sbsp_tag: u32, ordinal: u32) -> Option<[f32; 3]> {
        let get = self.lbsp_instance_probe?;
        let mut rgb = [0.0f32; 3];
        let ok = unsafe { get(cache.0, sbsp_tag, ordinal, rgb.as_mut_ptr()) };
        if ok == 0 { return None; }
        if !rgb.iter().all(|c| c.is_finite() && *c >= 0.0) { return None; }
        Some(rgb)
    }

    pub fn cluster_pvl(&self, cache: CacheHandle, sbsp_tag: u32, cluster: u32) -> Option<ClusterPvl> {
        let get = self.lbsp_cluster_pvl?;
        self.pvl_via(get, cache, sbsp_tag, cluster)
    }

    fn pvl_via(&self, get: FnClusterPvl, cache: CacheHandle, sbsp_tag: u32, index: u32) -> Option<ClusterPvl> {
        let cluster = index;
        let mut vb = ZhLbspClusterPvlVb {
            found: 0,
            bytes: std::ptr::null_mut(),
            len: 0,
            elem_count: 0,
            hdr_scale_raw: 0,
            hdr_scale: 1.0,
            pvb_index: -1,
            pvb_offset: 0,
            vb_index: -1,
        };
        let ok = unsafe { get(cache.0, sbsp_tag, cluster, &mut vb) };
        // Copy out of the packed struct into locals (avoid unaligned refs).
        let found = vb.found;
        let bytes_ptr = vb.bytes;
        let len = vb.len;
        let pvb_offset = vb.pvb_offset;
        let hdr_scale = vb.hdr_scale;
        if ok == 0 || found == 0 || bytes_ptr.is_null() || len == 0 {
            if !bytes_ptr.is_null() {
                if let Some(f) = self.lbsp_free_lightprobe {
                    unsafe { f(bytes_ptr) };
                }
            }
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, len as usize).to_vec() };
        if let Some(f) = self.lbsp_free_lightprobe {
            unsafe { f(bytes_ptr) };
        }
        Some(ClusterPvl { bytes, pvb_offset, hdr_scale: if hdr_scale.is_finite() && hdr_scale > 0.0 { hdr_scale } else { 1.0 } })
    }

    // ---- bitmap ----
    pub fn find_bitmap(&self, cache: CacheHandle, tag: u32, submap: u32) -> Option<ZhBitmapInfo> {
        let f = self.mbp_find_bitmap?;
        let mut info = ZhBitmapInfo::default();
        (unsafe { f(cache.0, tag, submap, &mut info) } != 0).then_some(info)
    }
    /// Decode a bitmap mip to RGBA8. Returns (rgba, width, height).
    pub fn decode_bitmap(
        &self,
        cache: CacheHandle,
        tag: u32,
        submap: u32,
        mip: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let f = self.mbp_decode_bitmap?;
        let free = self.mbp_free?;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let (mut w, mut h) = (0u32, 0u32);
        let ok = unsafe { f(cache.0, tag, submap, mip, &mut ptr, &mut w, &mut h) } != 0;
        if !ok || ptr.is_null() || w == 0 || h == 0 {
            return None;
        }
        let len = (w as usize) * (h as usize) * 4;
        let rgba = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        unsafe { free(ptr) };
        Some((rgba, w, h))
    }

    /// Decode a bitmap KEEPING its real alpha channel (the normal `decode_bitmap`
    /// stamps alpha=255). Needed for alpha-test foliage cutout masks. Returns
    /// BGRA bytes (same convention as `decode_bitmap`).
    pub fn decode_bitmap_keep_alpha(
        &self,
        cache: CacheHandle,
        tag: u32,
        submap: u32,
        mip: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let f = self.mbp_decode_bitmap_keep?;
        let free = self.mbp_free?;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let (mut w, mut h) = (0u32, 0u32);
        let ok = unsafe { f(cache.0, tag, submap, mip, &mut ptr, &mut w, &mut h) } != 0;
        if !ok || ptr.is_null() || w == 0 || h == 0 {
            return None;
        }
        let len = (w as usize) * (h as usize) * 4;
        let rgba = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        unsafe { free(ptr) };
        Some((rgba, w, h))
    }

    /// Decode ONE z-slice of a volume submap, KEEPING alpha (BGRA). The dual-VMF SDM
    /// `lightprobe_hdr_color` is a depth>=3 volume: slice0/1 reconstruct the dominant-lobe
    /// colour (`slice0.rgb + slice1.rgb*2-1`) and slices' ALPHA carry the packed direction;
    /// slice2.rgb is the fill (ambient) lobe. Returns None if the export is absent (older
    /// DLL) or the bitmap isn't deep enough — callers fall back to the single-slice proxy.
    pub fn decode_bitmap_slice(
        &self,
        cache: CacheHandle,
        tag: u32,
        submap: u32,
        mip: u32,
        slice: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let f = self.mbp_decode_bitmap_slice?;
        let free = self.mbp_free?;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let (mut w, mut h) = (0u32, 0u32);
        let ok = unsafe { f(cache.0, tag, submap, mip, slice, &mut ptr, &mut w, &mut h) } != 0;
        if !ok || ptr.is_null() || w == 0 || h == 0 {
            if !ptr.is_null() {
                unsafe { free(ptr) };
            }
            return None;
        }
        let len = (w as usize) * (h as usize) * 4;
        let rgba = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        unsafe { free(ptr) };
        Some((rgba, w, h))
    }

    /// Decode ONE face (0..5) of a CUBEMAP submap (bitmapType==2), KEEPING alpha (BGRA).
    /// A cube's `depth` is 1 even though it stores 6 faces, so `decode_bitmap_slice`
    /// refuses faces 1..5; this keys off a fixed 6-face count. `stride_mode`: 0 = face-major,
    /// 1 = mip-major (the CORRECT layout for Reach bitm cubes — empirically confirmed).
    /// Returns None
    /// when the export is absent (older DLL), the bitmap isn't a cube, or the face is
    /// out of range — the authored env-cube path then falls back to the captured sky dome.
    pub fn decode_bitmap_face(
        &self,
        cache: CacheHandle,
        tag: u32,
        submap: u32,
        mip: u32,
        face: u32,
        stride_mode: u32,
    ) -> Option<(Vec<u8>, u32, u32)> {
        let f = self.mbp_decode_bitmap_face?;
        let free = self.mbp_free?;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let (mut w, mut h) = (0u32, 0u32);
        let ok =
            unsafe { f(cache.0, tag, submap, mip, face, stride_mode, &mut ptr, &mut w, &mut h) }
                != 0;
        if !ok || ptr.is_null() || w == 0 || h == 0 {
            if !ptr.is_null() {
                unsafe { free(ptr) };
            }
            return None;
        }
        let len = (w as usize) * (h as usize) * 4;
        let rgba = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        unsafe { free(ptr) };
        Some((rgba, w, h))
    }

    /// fetch the bitmap's RAW DDS (148-byte DXT10 header + BC blocks + the map's own
    /// mip chain) for direct GPU upload — skips the CPU BCn-decode + CPU mip-regen that
    /// dominate load time. Returns None when the format isn't block-compressed (the caller
    /// falls back to the CPU decode path).
    pub fn raw_dds(&self, cache: CacheHandle, tag: u32, submap: u32) -> Option<Vec<u8>> {
        let f = self.mbp_get_raw_dds?;
        let free = self.mbp_free_raw_dds?;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut len = 0u32;
        let ok = unsafe { f(cache.0, tag, submap, &mut ptr, &mut len) } != 0;
        if !ok || ptr.is_null() || len < 148 {
            if !ptr.is_null() { unsafe { free(ptr) }; }
            return None;
        }
        let dds = unsafe { std::slice::from_raw_parts(ptr, len as usize).to_vec() };
        unsafe { free(ptr) };
        Some(dds)
    }

    /// Cleanly detach hooks + drain handles in THIS process's copy (used by the
    /// viewer before it FreeLibrary's its own parsing copy).
    pub fn prepare_unload(&self) {
        if let Some(f) = self.prepare_unload {
            unsafe { f() }
        }
    }

    /// ask the native DLL to return its freed transient-decode heap to the OS.
    pub fn trim_heaps(&self) {
        if let Some(f) = self.trim_heaps {
            unsafe { f() }
        }
    }
    /// Free the inflate-once page cache (parent + shared children) for `cache`. Call from the
    /// POST-load trim only — NOT the periodic during-load trims, which would evict the pages that
    /// make the load fast. Returns steady memory to the ~200MB after the load settles.
    pub fn clear_page_cache(&self, cache: CacheHandle) {
        if let Some(f) = self.mbp_clear_page_cache {
            unsafe { f(cache.0) }
        }
    }
    /// bytes currently held by the native inflate-once page cache (parent + shared children).
    pub fn page_cache_bytes(&self, cache: CacheHandle) -> u64 {
        self.mbp_page_cache_bytes.map_or(0, |f| unsafe { f(cache.0) })
    }
    /// drop the resident pages of the read-only .map mapping(s) from this process (they
    /// re-fault from the OS file cache on the next access). Linux only; a no-op on Windows / old DLLs.
    pub fn drop_map_pages(&self, cache: CacheHandle) {
        if let Some(f) = self.mbp_drop_map_pages {
            unsafe { f(cache.0) }
        }
    }
}

impl Drop for NativeDll {
    fn drop(&mut self) {
        // Quiesce the DLL (detach hooks / drain handles) if it exports the hook, then
        // deliberately DO NOT drop `_lib` — leaking the HMODULE keeps the DLL mapped so
        // its DllMain-spawned worker thread never runs freed code. See the field comment
        // for the crash (Unloaded_HaloMapStudioDLL.dll+0x145d). One leaked module per
        // process is negligible and this NativeDll normally lives the whole session.
        self.prepare_unload();
        // `_lib` is ManuallyDrop → not calling ManuallyDrop::drop leaks it (no FreeLibrary).
    }
}

/// Owned vertex + index buffers copied out of the DLL (already freed there).
pub struct GeomBuffers {
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
}

fn decode_geom(f: FnDecodeGeom, free: FnFree, handle: u64, i: u32) -> Option<GeomBuffers> {
    let mut vb: *mut u8 = std::ptr::null_mut();
    let mut ib: *mut u8 = std::ptr::null_mut();
    let (mut vlen, mut ilen) = (0u32, 0u32);
    let ok = unsafe { f(handle, i, &mut vb, &mut vlen, &mut ib, &mut ilen) } != 0;
    if !ok {
        return None;
    }
    let vertices = if !vb.is_null() && vlen > 0 {
        let v = unsafe { std::slice::from_raw_parts(vb, vlen as usize).to_vec() };
        unsafe { free(vb) };
        v
    } else {
        vec![]
    };
    let indices = if !ib.is_null() && ilen > 0 {
        let v = unsafe { std::slice::from_raw_parts(ib, ilen as usize).to_vec() };
        unsafe { free(ib) };
        v
    } else {
        vec![]
    };
    Some(GeomBuffers { vertices, indices })
}

fn decode_vec(f: FnDecodeVec, free: FnFree, handle: u64, i: u32) -> Option<Vec<u8>> {
    let mut p: *mut u8 = std::ptr::null_mut();
    let mut len = 0u32;
    let ok = unsafe { f(handle, i, &mut p, &mut len) } != 0;
    if !ok || p.is_null() || len == 0 {
        return None;
    }
    // The DLL's UV/normal exports return the FLOAT count (vertexCount·2 for UVs,
    // ·3 for normals), NOT a byte count — so the buffer is `len * 4` bytes. Reading
    // only `len` bytes truncated UVs to 1/4, collapsing most vertices to (0,0)
    // (flat single-colour instanced cliffs/ground).
    let bytes = len as usize * 4;
    let v = unsafe { std::slice::from_raw_parts(p, bytes).to_vec() };
    unsafe { free(p) };
    Some(v)
}

// Typed handles so callers can't mix cache/bsp/model handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheHandle(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BspHandle(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelHandle(pub u64);

// keep c_void referenced (used in future export signatures)
const _: usize = std::mem::size_of::<*const c_void>();
