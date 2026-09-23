//! Halo 4 lightmaps: the `Lbsp` atlas bitmaps, their decode, the per-instance lightmap UV
//! streams and the baked sun. Companion doc section: docs/halo4_geometry_layout.md "10. Lightmaps".
//!
//! Layout (wraparound + ca_forge_ravine, cross-checked by the tests at the bottom; m10_crash for
//! the object/per-vertex-only Lbsps):
//!   * Lbsp main struct: `+0x14 f32`, `+0x18 f32` = per-pixel scale constants (present exactly when
//!     the +0x3C intensity map exists; Reach's K sits at +0x18), `+0x1C == +0x20 f32` = per-VERTEX
//!     scale (present on per-vertex-only object Lbsps), `+0x24 f32x3` = SUN direction (unit, the
//!     direction light TRAVELS; negate for the to-sun vector), `+0x30 f32x3` = sun RGB intensity
//!     (30/25.4/22.4 wraparound, 14/12.3/8.3 Ravine - both Ravine BSPs carry the same values).
//!     Sun proof: 91.6 % of the vertices whose analytic texel says "sun visible" face -dir, and
//!     the dominant lobe of those texels points at -dir (dot 0.78 wraparound / 0.65 Ravine).
//!   * bitmap refs: +0x3C `*_dualvmf_hybrid_pixel_intensity_dxt5` = DXT5 (BC3) texture ARRAY of
//!     2 slices; +0x4C `*_pixel_direction_dxtn` = DXN (BC5 **SNORM**) array of 3 slices;
//!     +0x5C `*_pixel_analytic_dxt5a` = format 44 (DXN_mono_alpha = BC5 unorm, 1 B/px), 1 slice.
//!     All single-level, primary page stream only, slices stored back to back (slice-major).
//!   * Direction map = TWO lobe vectors in world space, length <= 1 (length = sharpness as in
//!     Reach's DecompressVMF): lobe A = (s2.x, s0.x, s0.y), lobe B = (s2.y, s1.x, s1.y) where
//!     `sK.c` is channel c of slice K. Axis-group medians on wraparound: +x -> P4/P5 only,
//!     +y -> P0/P2 only, +z -> P1/P3 only, all positive. Lobe A carries the BAKED SUN (on
//!     sun-visible texels A points at the sun, dot 0.78/0.65; |A| grows 0.59 -> 0.78 in sun);
//!     lobe B aligns with the surface normal on the enclosed map (dot 0.73, 99 % positive).
//!   * Intensity slice 0 (rgb + a) pairs with lobe A (lum0 ~ |A| 0.76), slice 1 with lobe B.
//!   * Analytic map: x = sun visibility (0/1 mask, hard shadows), y = the floating sun's extra
//!     multiplier (bimodal 0/1 on the enclosed map, ~1.0 everywhere outdoors; anti-correlates
//!     -0.84 with slice-1 alpha; only the per-vertex shader path reads it, see below).
//!   * Lightmap UVs: the type-4 / stride-4 vertex buffers are u16x2 UNORM, ALREADY in atlas space
//!     (texel-centre offsets: min u = 0.5/1792). No per-cluster / per-instance sub-rect exists:
//!     Lbsp +0xBC clusters (8 B: i16 array index -1, i16 -1, i32 0) select nothing and the
//!     per-instance record carries only the VB index. 100.0 % (wraparound) / 97.4 % (Ravine) of
//!     the lightmapped vertices land on a non-padding texel of slice 0.
//!   * Lbsp +0x1B8 = per-INSTANCE records, **28 B**: two empty 12-B blocks, then `i16 vb @+24`
//!     (index of the instance's type-4 UV stream, -1 = per-vertex lit) + i16 0. The 553 non-null
//!     indices on wraparound are unique, all type-4, and each VB's vertex count equals the
//!     instance mesh's vertex count. Lbsp +0xC8 = per-instance 36 B: `i32 mesh index @+20`
//!     (== the type-5 instance record's mesh, 1046/1046), `f32x3 bounds centre @+24`.
//!   * Cluster meshes on both MP maps have NO geometry (everything is instanced), so there is no
//!     per-cluster lightmap path (`mesh_lightmap_uvs` is per instance).
//!
//! ENGINE LAW (from the shipped pixel shaders' DXBC and the constant setup in halo4.dll;
//! evidence in docs/halo4_geometry_layout.md section 10):
//!   * Shader (`srf_blinn` PS entry 31 = forward static per-pixel, entry 04/06/10 = deferred; RDEF
//!     names: t25 `ps_bsp_lightprobe_dir_and_bandwidth_texture` = the +0x4C direction array,
//!     t26 `ps_bsp_lightprobe_hdr_color_texture` = the +0x3C intensity array, t28
//!     `ps_bsp_lightprobe_analytic_texture` = +0x5C, t27 `ps_lightmap_sharpen_falloff_texture` =
//!     the `rasterizer\sharpen_falloff_64` bitmap, cb1 `EngineBSPPS` = {compress_constant_1,
//!     compress_constant_2, scale_constants}, cb4 `EngineModelLightingPS` regs 12..14 =
//!     floating_shadow_light_direction / _intensity / static_floating_shadow_sharpening):
//!       I_k     = exp2(-9 * slice_k.a) * cc1.xy[k] + cc1.zw[k]          (k = 0: lobe A, 1: lobe B)
//!       col_k   = slice_k.rgb * I_k                                      (NO channel swap)
//!       d_A     = (s2.x, s0.x, s0.y),  d_B = (s2.y, s1.x, s1.y),  w_k = sqrt(1 - |d_k|^2)
//!       u_k     = 0.2821 * w_k + 0.3257 * dot(d_k, N_pixel)   (SH L1 irradiance / pi; Y00 and
//!                 the cosine-convolved Y1 coefficient)   v_k = same with the VERTEX normal
//!       f_k     = sharpen_falloff_64(u_k, v_k)   (64x64 mono LUT; identity f = u on the diagonal,
//!                 a contrast S-curve around the vertex-normal value off it)
//!       gate_A  = lerp(sat(mask.g + scale.y), 1, analytic.x) with the floating sun, scale.y = 0.3
//!                 (sub_18034F2D0); mask = the per-frame shadow mask (1 with no dynamic occluder)
//!       diffuse = col_A * f_A * gate_A + col_B * f_B + sun_rgb * sat(vis * (s+1) - (s+1)/2 + 1/2)
//!                 * max(dot(N, -sun_dir), 0) / pi,  vis = analytic.x (x mask.r inside the shadow
//!                 frustum, x mask.b), s = scnr structure_bsps[+184] + lightmapStaticShadowSharpeningFactor
//!       out     = albedo * diffuse + specular (Blinn NdotH^p / pi against the two lobe directions
//!                 and the sun), then * ps_view_exposure.x
//!     The lobes are NOT vMF lobes: |d| is the SH L1 / L0 ratio and the LUT does the convolution.
//!   * Constants (halo4.dll sub_18038C1E0, reached from sub_18034F2D0 via the Lbsp accessor
//!     sub_18034F6C8): with x = direct_scalar * Lbsp[+0x14], y = indirect_scalar * Lbsp[+0x18]
//!     (scalars = the `set_lightmap_direct/indirect_scalar_bsp` script globals, 1.0 unless scripted;
//!     the global brightness multipliers at 0x180E85500 / 0x180E854FC are 1.0 in the image):
//!       cc1 = (x * 512/511, y * 512/511, -x/511, -y/511)   =>  I_k = K_k * (512 * exp2(-9 a) - 1) / 511
//!       cc2 = (0.65, 1.0, Lbsp i16 +0x130, 0.5 / that)      (hybrid overlay texture size; unused here)
//!       vs_bsp_lightmap_compress_constant = (direct * Lbsp[+0x1C], indirect * Lbsp[+0x20], 1, 1)
//!     so +0x14 = K_DIRECT (lobe A = slice 0 = "direct": sun / sky / emissive), +0x18 = K_INDIRECT
//!     (lobe B = slice 1 = bounce), +0x1C / +0x20 = the per-VERTEX pair (5-slice R5G6B5 array
//!     `*_vertex_565`, 1024 wide, indexed by vertex id + vs_mesh_lightmap_compress_constant.z).
//!   * Floating sun (sub_18035F24C / sub_18035E814): Lbsp flags16 bit 0x200 = has floating sun
//!     (set on wraparound / Ravine / vista), bit 0x2000 = no shadow cascade (the sun stays on);
//!     light direction = Lbsp +0x24
//!     (shader uses -dir), intensity = Lbsp +0x30; sharpening s = scnr structure_bsps[+184] (1.0 on
//!     both MP BSPs) + the debug global (0): sat(2 * vis - 0.5). The dual-VMF lightmap is
//!     therefore "hybrid": lobe A on analytic.x = 1 texels is the sky / residual (blue on Ravine),
//!     the direct sun is analytic there; where analytic.x = 0 the baked lobe A carries it.
//!   * analytic.y is never read by the per-pixel shaders (only the per-vertex path carries an
//!     extra visibility multiplier in its slice-2 y channel).
//!
//! HMS decode (`lobe_intensity`, `compose_atlas_textures`, mesh shader `h4_lightmap_tint`): the
//! four composed BGRA8 textures are lobe A rgb+a, lobe B rgb+a, dir A unorm + analytic.x, dir B
//! unorm + the 64x64 sharpen LUT stashed in the top-left corner of the .w channel (analytic.y is
//! unused, and the renderer's fixed vMF LUT slot cannot carry a second table). gate_A = 1 (no
//! dynamic shadow mask; HMS's own cast shadow darkens lobe A + the sun downstream) and the sun
//! visibility returned in .a is sqrt(sat(2 vis - 0.5)) because the caller squares it (Reach law).
//! Default ON for Halo 4 (`HMS_H4_LIGHTMAP=0` reverts to the flat placeholder).

use anyhow::{anyhow, bail, Result};

use super::bitmaps::{bitmap_info, BitmapInfo, OFF_BITM_RESOURCES};
use super::cache::{ByteRead, H4Cache};
use super::geometry::{H4Bsp, MESH_ELEM, OFF_SBSP_MESHES};

pub const OFF_LBSP_K_DOM: usize = 0x14;
pub const OFF_LBSP_K: usize = 0x18;
pub const OFF_LBSP_K_VERTEX: usize = 0x1C;
pub const OFF_LBSP_SUN_DIR: usize = 0x24;
pub const OFF_LBSP_SUN_RGB: usize = 0x30;
pub const OFF_LBSP_INTENSITY: usize = 0x3C;
pub const OFF_LBSP_DIRECTION: usize = 0x4C;
pub const OFF_LBSP_ANALYTIC: usize = 0x5C;
pub const OFF_LBSP_VERTEX_565: usize = 0x9C;
pub const OFF_LBSP_VERTEX_AO: usize = 0xAC;
/// Lbsp +0xBC cluster records (8 B: i16 lightprobe array index, i16 per-vertex block, i32
/// offset; all -1 / 0 on the shipped MP maps - read by the tests only).
#[cfg(test)]
pub const OFF_LBSP_CLUSTERS: usize = 0xBC;
pub const OFF_LBSP_INSTANCE_INFO: usize = 0xC8;
pub const OFF_LBSP_INSTANCE_LM: usize = 0x1B8;
#[cfg(test)]
pub const CLUSTER_ELEM: usize = 8;
pub const INSTANCE_INFO_ELEM: usize = 36;
pub const INSTANCE_LM_ELEM: usize = 28;
/// Bitmap format 44 = DXN_mono_alpha (BC5 unorm; x = mono, y = alpha) - the analytic map.
pub const FMT_DXN_MONO_ALPHA: i16 = 44;
/// Bitmap format 36 = BC4 (0.5 B/px, one channel) - the analytic map of sun-less m10_crash BSPs.
pub const FMT_BC4: i16 = 36;
pub const FMT_DXN: i16 = 38;
pub const FMT_DXT5: i16 = 16;
pub const FMT_A8R8G8B8: i16 = 11;
/// Lbsp +0 flags16: floating (analytic) sun present (halo4.dll sub_18035F24C byte 0).
pub const LBSP_FLAG_FLOATING_SUN: u16 = 0x200;
/// Bit 0x2000 = NO floating-shadow cascade (sub_18035F24C byte 1 = `(flags & 0x2000)
/// == 0`, consumed only by sub_18035E814's frustum build); the sun is unaffected.
pub const LBSP_FLAG_NO_SHADOW_CASCADE: u16 = 0x2000;

/// scnr structure_bsps block: count @+0xA4, pointer @+0xA8 (the engine reads
/// `*(scenario + 168)`), 336-B elements; `f32 @+184` = static floating-shadow sharpening.
pub const OFF_SCNR_STRUCTURE_BSPS: usize = 0xA4;
pub const SCNR_STRUCTURE_BSP_ELEM: usize = 336;
pub const OFF_SCNR_BSP_SHADOW_SHARPEN: usize = 184;
/// The engine's `rasterizer\sharpen_falloff_64` LUT (64x64 A8R8G8B8, mono) stashed in the
/// top-left corner of the composed dir-B texture's alpha channel.
pub const SHARPEN_LUT_SIZE: u32 = 64;
pub const SHARPEN_LUT_NAME: &str = "rasterizer\\sharpen_falloff_64";

#[derive(Clone, Copy, Debug)]
pub struct H4LightmapAtlas {
    pub lbsp_tag: usize,
    /// +0x3C intensity map (DXT5 array, `dm_slices` slices).
    pub dm_tag: usize,
    /// +0x4C direction map (DXN snorm array, `sdm_slices` slices).
    pub sdm_tag: usize,
    /// +0x5C analytic (sun visibility) map, format 44, 1 slice.
    pub analytic_tag: Option<usize>,
    /// The analytic map carries a .y channel (format 44 DXN_mono_alpha; the BC4
    /// sun-less variant has none -> the shader's multiplier is 1).
    pub analytic_has_y: bool,
    /// Lbsp +0x18 = K_INDIRECT: scale of lobe B / intensity slice 1 (compress_constant_1.y,
    /// halo4.dll sub_18038C1E0). 14.42 wraparound / 5.17 Ravine.
    pub k: f32,
    /// Lbsp +0x14 = K_DIRECT: scale of lobe A / intensity slice 0 (compress_constant_1.x).
    /// 277.7 wraparound (light strips) / 9.33 Ravine.
    pub k_dom: f32,
    /// Lbsp +0x1C: per-VERTEX direct scale (vs_bsp_lightmap_compress_constant.x).
    pub k_vertex: f32,
    /// Lbsp +0x20: per-VERTEX indirect scale (vs_bsp_lightmap_compress_constant.y).
    pub k_vertex_indirect: f32,
    /// Lbsp +0 flags: bit 0x200 = floating (analytic) sun present, bit 0x2000 = no
    /// floating-shadow cascade (the sun stays on).
    pub flags: u16,
    /// True when the engine builds the floating-sun record for this BSP (flags 0x200 set,
    /// non-zero direction): the analytic sun is added on top of the lobes (every MP BSP,
    /// including the 0x6202 ones - Redoubt, Tower, the vistas - which only lack the cascade).
    pub floating_sun: bool,
    /// scnr structure_bsps[+184] static floating-shadow sharpening `s` for this BSP (1.0 on
    /// the MP maps; the engine adds the debug global lightmapStaticShadowSharpeningFactor = 0):
    /// sun visibility = sat(vis * (s + 1) - (s + 1) / 2 + 1/2).
    pub shadow_sharpen: f32,
    pub w: u32,
    pub h: u32,
    pub dm_slices: u32,
    pub sdm_slices: u32,
    /// Unit vector TOWARDS the sun (= -(Lbsp +0x24)).
    pub sun_dir: [f32; 3],
    /// Sun RGB intensity (Lbsp +0x30).
    pub sun_rgb: [f32; 3],
}

/// Resolve the Lbsp's atlas bitmaps and constants. Fails when the Lbsp has no per-pixel atlas
/// (object / per-vertex-only Lbsps carry only +0x5C / +0x9C / +0xAC).
pub fn load_atlas(c: &H4Cache, lbsp_tag: usize) -> Result<H4LightmapAtlas> {
    if c.tag_class(lbsp_tag).as_ref() != Some(b"Lbsp") { bail!("tag {lbsp_tag} is not an Lbsp"); }
    let m = c.tag_meta(lbsp_tag).ok_or_else(|| anyhow!("Lbsp meta"))?;
    let d = c.data();
    let dm_tag = c.tag_ref_of(m + OFF_LBSP_INTENSITY, b"bitm").ok_or_else(|| anyhow!("Lbsp has no per-pixel intensity map (+0x3C)"))?;
    let sdm_tag = c.tag_ref_of(m + OFF_LBSP_DIRECTION, b"bitm").ok_or_else(|| anyhow!("Lbsp has no direction map (+0x4C)"))?;
    let analytic_tag = c.tag_ref_of(m + OFF_LBSP_ANALYTIC, b"bitm");
    let dm = bitmap_info(c, dm_tag).ok_or_else(|| anyhow!("intensity bitmap element"))?;
    let sdm = bitmap_info(c, sdm_tag).ok_or_else(|| anyhow!("direction bitmap element"))?;
    if dm.format != FMT_DXT5 { bail!("intensity map format {} (expected DXT5 = 16)", dm.format); }
    if sdm.format != FMT_DXN { bail!("direction map format {} (expected DXN = 38)", sdm.format); }
    if (dm.w, dm.h) != (sdm.w, sdm.h) { bail!("intensity {}x{} vs direction {}x{}", dm.w, dm.h, sdm.w, sdm.h); }
    let f = |o: usize| d.f32_at(m + o);
    let dir = [f(OFF_LBSP_SUN_DIR), f(OFF_LBSP_SUN_DIR + 4), f(OFF_LBSP_SUN_DIR + 8)];
    let flags = d.u16_at(m);
    // The floating-sun record exists when bit 0x200 is set and the direction is non-zero
    // (sub_18035F24C byte 0 = `using_floating_sun`); bit 0x2000 only clears record byte 1,
    // which sub_18035E814 tests to build / upload the floating-SHADOW cascade frustum - the sun
    // itself stays on (Redoubt / Tower / the vistas: flags 0x6202 = sun, no cascade; reading
    // 0x2000 as "sun disabled" would drop the analytic sun of every 0x6202 BSP).
    let sun_on = flags & LBSP_FLAG_FLOATING_SUN != 0 && dir.iter().map(|v| v * v).sum::<f32>() >= 1e-7;
    let analytic_has_y = analytic_tag.and_then(|t| bitmap_info(c, t)).map_or(false, |i| i.format == FMT_DXN_MONO_ALPHA);
    Ok(H4LightmapAtlas {
        lbsp_tag,
        dm_tag,
        sdm_tag,
        analytic_tag,
        analytic_has_y,
        k: f(OFF_LBSP_K),
        k_dom: f(OFF_LBSP_K_DOM),
        k_vertex: f(OFF_LBSP_K_VERTEX),
        k_vertex_indirect: f(OFF_LBSP_K_VERTEX + 4),
        flags,
        floating_sun: sun_on,
        shadow_sharpen: scnr_shadow_sharpen(c, lbsp_tag).unwrap_or(0.0),
        w: dm.w as u32,
        h: dm.h as u32,
        dm_slices: dm.depth.max(1) as u32,
        sdm_slices: sdm.depth.max(1) as u32,
        sun_dir: [-dir[0], -dir[1], -dir[2]],
        sun_rgb: if sun_on { [f(OFF_LBSP_SUN_RGB), f(OFF_LBSP_SUN_RGB + 4), f(OFF_LBSP_SUN_RGB + 8)] } else { [0.0; 3] },
    })
}

/// scnr structure_bsps (count @+0xA4, ptr @+0xA8, 336 B elements; the engine indexes it as
/// `*(scenario + 168) + 84 dwords * bsp`) -> `f32 @+184` = the static floating-shadow
/// sharpening of the BSP whose Lbsp is `lbsp_tag`. The element's `bitm`-less sbsp ref sits at +0;
/// the Lbsp is matched through the sLdT/sbsp name (the Lbsp of BSP i is named after its sbsp).
pub fn scnr_shadow_sharpen(c: &H4Cache, lbsp_tag: usize) -> Option<f32> {
    let lname = c.tag_name(lbsp_tag).to_ascii_lowercase();
    let d = c.data();
    for scnr in c.find_tags(b"scnr") {
        let sm = c.tag_meta(scnr)?;
        let (n, o) = c.block(sm + OFF_SCNR_STRUCTURE_BSPS)?;
        for i in 0..n {
            let e = o + i * SCNR_STRUCTURE_BSP_ELEM;
            let Some(sbsp) = c.tag_ref_of(e, b"sbsp") else { continue };
            if c.tag_name(sbsp).to_ascii_lowercase() == lname { return Some(d.f32_at(e + OFF_SCNR_BSP_SHADOW_SHARPEN)); }
        }
    }
    None
}

/// scnr structure_bsps floating-shadow cascade fields (Assembly plugin / halo4.dll
/// sub_18035F7EC: `u8 @+0xB4` cascade count, `u8 @+0xB6` quality (0 = 8 tap, 1 = 12 tap, 2 = 6 tap
/// poisson apply), `u8 @+0xB7` resolution (0 = 512, 1 = 800), 24-B cascade records @+0xBC =
/// {half_width, length, offset, bias, filter width, sun direction offset}). Every shipped MP map
/// authors ONE cascade (3 wu half-width, 35..100 wu long; Ravine 6-tap 800² bias 0.0006 filter 3
/// sun offset 18). Only cascade 0 is returned (the multi-cascade maps - Deadly Crossing / Bonanza
/// - stack a 12 wu second box that HMS does not draw).
pub fn scnr_cascade(c: &H4Cache, lbsp_tag: usize) -> Option<hms_render::H4CascadeCfg> {
    let lname = c.tag_name(lbsp_tag).to_ascii_lowercase();
    let d = c.data();
    // Lbsp flag 0x2000 = the engine never builds the cascade frustum for this BSP
    if d.u16_at(c.tag_meta(lbsp_tag)?) & LBSP_FLAG_NO_SHADOW_CASCADE != 0 { return None; }
    for scnr in c.find_tags(b"scnr") {
        let sm = c.tag_meta(scnr)?;
        let (n, o) = c.block(sm + OFF_SCNR_STRUCTURE_BSPS)?;
        for i in 0..n {
            let e = o + i * SCNR_STRUCTURE_BSP_ELEM;
            let Some(sbsp) = c.tag_ref_of(e, b"sbsp") else { continue };
            if c.tag_name(sbsp).to_ascii_lowercase() != lname { continue; }
            let count = d.u8_at(e + 0xB4);
            if count == 0 { return None; }
            let quality = d.u8_at(e + 0xB6);
            let res = if d.u8_at(e + 0xB7) == 1 { 800 } else { 512 };
            let f = |k: usize| d.f32_at(e + 0xBC + k * 4);
            let cfg = hms_render::H4CascadeCfg {
                half_width: f(0), length: f(1), offset: f(2), bias: f(3), filter: f(4), sun_offset: f(5),
                resolution: res,
                taps: match quality { 1 => 12, 2 => 6, _ => 8 },
            };
            if !(cfg.half_width > 0.0 && cfg.length > 0.0) { return None; }
            return Some(cfg);
        }
    }
    None
}

/// Bytes per pixel (x1) of the single-level formats used by the atlases.
fn bytes_per_slice(info: &BitmapInfo) -> Result<usize> {
    let px = info.w as usize * info.h as usize;
    match info.format {
        FMT_DXT5 | FMT_DXN | FMT_DXN_MONO_ALPHA => Ok(px), // 16 B per 4x4 block
        14 | FMT_BC4 => Ok(px / 2),                          // 8 B per 4x4 block
        FMT_A8R8G8B8 => Ok(px * 4),                          // the sharpen LUT
        FMT_R5G6B5 => Ok(px * 2),                            // per-vertex arrays
        f => bail!("unsupported atlas bitmap format {f}"),
    }
}

/// Raw mip-0 bytes of every slice of a single-level (array) bitmap, from its primary stream.
pub fn bitmap_slices_raw(c: &H4Cache, tag: usize) -> Result<(BitmapInfo, Vec<u8>)> {
    let info = bitmap_info(c, tag).ok_or_else(|| anyhow!("no bitmap element"))?;
    let m = c.tag_meta(tag).ok_or_else(|| anyhow!("meta"))?;
    let (_, ro) = c.block(m + OFF_BITM_RESOURCES).ok_or_else(|| anyhow!("no resources block"))?;
    let e = c.resource_by_id(c.data().u32_at(ro)).ok_or_else(|| anyhow!("resource id does not resolve"))?;
    if !c.kind_is(e.kind, H4Cache::RES_BITMAP) { bail!("resource kind {} is not a bitmap", e.kind); }
    let dd = c.definition(e);
    let slices = info.depth.max(1) as usize;
    let need = bytes_per_slice(&info)? * slices;
    // single-level atlases ship one (primary, nibble 4) stream holding every slice back to back
    let mut best: Option<(usize, u32)> = None;
    for k in 0..3 {
        let size = dd.i32_at(k * 20).max(0) as usize;
        if let Some(addr) = e.fixup_at((k * 20 + 12) as u32) {
            if size >= need && best.map_or(true, |b| size < b.0) { best = Some((size, addr)); }
        }
    }
    let (size, addr) = best.ok_or_else(|| anyhow!("no page stream holds {need} B ({}x{}x{} fmt {})", info.w, info.h, slices, info.format))?;
    // The DLC-era atlases (levels\dlc\*, zd_02_grind) ship every atlas stream at exactly
    // TWICE the slice total: the real slices first, then zero padding (intensity / analytic) or
    // a flat-normal fill (direction, the "second unused chain" of bitmaps.rs) - on
    // dlc_forge_island + ca_basin the last non-zero byte sits at 49.9 % of the intensity and
    // analytic streams and the direction tail is constant (128,128,0..) BC5 blocks. Only the
    // prefix is the texture, so a larger stream is fine; a smaller one is not.
    if size < need { bail!("stream size {size} < {slices} slices x {} B", need / slices); }
    Ok((info, c.stream_bytes(e, addr, need)?))
}

/// Decoded intensity slices as BGRA (straight BC3 decode, no HDR reconstruction).
pub fn dm_slices_bgra(c: &H4Cache, atlas: &H4LightmapAtlas) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    let (info, raw) = bitmap_slices_raw(c, atlas.dm_tag)?;
    let per = bytes_per_slice(&info)?;
    Ok((0..info.depth.max(1) as usize)
        .map(|s| (decode_bc3_bgra(&raw[s * per..(s + 1) * per], info.w as u32, info.h as u32), info.w as u32, info.h as u32))
        .collect())
}

/// The three direction slices decoded as BGRA: r = x, g = y (snorm mapped to 0..255 as
/// (v + 1) / 2), b = 0, a = 255. Lobe A = (slice2.r, slice0.r, slice0.g), lobe B =
/// (slice2.g, slice1.r, slice1.g) after the unorm -> snorm inverse.
pub fn sdm_slices_bgra(c: &H4Cache, atlas: &H4LightmapAtlas) -> Result<Vec<(Vec<u8>, u32, u32)>> {
    let (info, raw) = bitmap_slices_raw(c, atlas.sdm_tag)?;
    let per = bytes_per_slice(&info)?;
    Ok((0..info.depth.max(1) as usize)
        .map(|s| (decode_bc5_bgra(&raw[s * per..(s + 1) * per], info.w as u32, info.h as u32, true), info.w as u32, info.h as u32))
        .collect())
}

/// The analytic map as BGRA: r = sun visibility, g = the second channel (format 44 = BC5 unorm;
/// format 36 = BC4, one channel, g = 0).
pub fn analytic_bgra(c: &H4Cache, atlas: &H4LightmapAtlas) -> Result<Option<(Vec<u8>, u32, u32)>> {
    let Some(tag) = atlas.analytic_tag else { return Ok(None) };
    let (info, raw) = bitmap_slices_raw(c, tag)?;
    let per = bytes_per_slice(&info)?;
    let (w, h) = (info.w as u32, info.h as u32);
    match info.format {
        FMT_DXN_MONO_ALPHA | FMT_DXN => Ok(Some((decode_bc5_bgra(&raw[..per], w, h, false), w, h))),
        FMT_BC4 => Ok(Some((decode_bc4_bgra(&raw[..per], w, h), w, h))),
        f => bail!("analytic map format {f}"),
    }
}

/// Two world-space lobe vectors (A, B) from the three decoded direction slices at texel (x, y).
pub fn lobe_dirs(sdm: &[(Vec<u8>, u32, u32)], x: u32, y: u32) -> Option<([f32; 3], [f32; 3])> {
    if sdm.len() < 3 { return None; }
    let ch = |s: usize, k: usize| -> Option<f32> {
        let (b, w, h) = &sdm[s];
        if x >= *w || y >= *h { return None; }
        // BGRA: r at +2, g at +1
        let o = ((y * *w + x) * 4) as usize + if k == 0 { 2 } else { 1 };
        Some(b[o] as f32 / 255.0 * 2.0 - 1.0)
    };
    Some(([ch(2, 0)?, ch(0, 0)?, ch(0, 1)?], [ch(2, 1)?, ch(1, 0)?, ch(1, 1)?]))
}

/// ENGINE lobe intensity from the slice alpha `a` (0..1) and the lobe's K (module doc;
/// halo4.dll sub_18038C1E0: compress_constant_1 = (K*512/511, .., -K/511, ..) applied to
/// exp2(-9a)): `K * (512 * exp2(-9a) - 1) / 511` - exactly K at a = 0 and exactly 0 at a = 1.
pub fn lobe_intensity(k: f32, a: f32) -> f32 {
    k * (512.0 * (-9.0 * a).exp2() - 1.0) / 511.0
}

/// Lobe colour of an intensity slice texel (BGRA buffer) as the engine samples it: (r, g, b) in
/// the plain BC3 decode order (the shaders multiply t26.rgb straight, no swizzle).
pub fn lobe_rgb(slice: &[u8], x: u32, y: u32, w: u32) -> [f32; 3] {
    let o = ((y * w + x) * 4) as usize;
    [slice[o + 2] as f32 / 255.0, slice[o + 1] as f32 / 255.0, slice[o] as f32 / 255.0]
}

/// The SH-L1 irradiance/pi coefficient the engine feeds the sharpen LUT: `d` is the raw
/// lobe vector (|d| <= 1), `n` the (unit) normal: `0.2821 * sqrt(1 - |d|^2) + 0.3257 * dot(d, n)`.
pub fn lobe_sh_coeff(d: [f32; 3], n: [f32; 3]) -> f32 {
    let l2: f32 = d.iter().map(|v| v * v).sum();
    let w = (1.0 - l2).max(0.0).sqrt();
    0.282_094_8 * w + 0.325_735 * (d[0] * n[0] + d[1] * n[1] + d[2] * n[2])
}

/// The engine's `rasterizer\sharpen_falloff_64` LUT as 64x64 bytes (row-major, y = the
/// vertex-normal coefficient, x = the pixel-normal coefficient; the shader samples `.x` = R of the
/// A8R8G8B8 bitmap, r == g == b). None when the map has no such bitmap.
pub fn sharpen_lut(c: &H4Cache) -> Option<Vec<u8>> {
    let tag = c.find_tags(b"bitm").into_iter().find(|&t| c.tag_name(t).eq_ignore_ascii_case(SHARPEN_LUT_NAME))?;
    let (info, raw) = bitmap_slices_raw(c, tag).ok()?;
    if info.format != FMT_A8R8G8B8 || (info.w as u32, info.h as u32) != (SHARPEN_LUT_SIZE, SHARPEN_LUT_SIZE) { return None; }
    // A8R8G8B8 in the file = BGRA bytes; take R (@+2)
    Some(raw.chunks_exact(4).map(|p| p[2]).collect())
}

/// The identity fallback of the sharpen LUT (f = u, the exact diagonal of the engine table).
fn identity_lut() -> Vec<u8> {
    (0..SHARPEN_LUT_SIZE * SHARPEN_LUT_SIZE).map(|i| (((i % SHARPEN_LUT_SIZE) as f32 + 0.5) / SHARPEN_LUT_SIZE as f32 * 255.0).round() as u8).collect()
}

/// The four BGRA8 textures the GPU decode samples (`h4_lightmap_tint` in the mesh shader, bound
/// as dm / sdm / sdm1 / sdm2), each `(bgra bytes, w, h)`:
///   0: lobe A (slice 0) rgb + alpha,  1: lobe B (slice 1) rgb + alpha,
///   2: direction A as unorm (v+1)/2 + analytic.x (sun visibility),
///   3: direction B as unorm + the 64x64 sharpen LUT in the top-left corner of .w (1 elsewhere;
///      analytic.y is never read by the engine's per-pixel shaders).
pub fn compose_atlas_textures(c: &H4Cache, atlas: &H4LightmapAtlas) -> Result<[(Vec<u8>, u32, u32); 4]> {
    let dm = dm_slices_bgra(c, atlas)?;
    if dm.len() < 2 { bail!("intensity map has {} slices, need 2", dm.len()); }
    let sdm = sdm_slices_bgra(c, atlas)?;
    if sdm.len() < 3 { bail!("direction map has {} slices, need 3", sdm.len()); }
    let ana = analytic_bgra(c, atlas)?;
    let (w, h) = (atlas.w, atlas.h);
    if w < SHARPEN_LUT_SIZE || h < SHARPEN_LUT_SIZE { bail!("atlas {w}x{h} too small to carry the sharpen LUT"); }
    let n = (w * h) as usize;
    let t0 = dm[0].0.clone();
    let t1 = dm[1].0.clone();
    let lut = sharpen_lut(c).unwrap_or_else(identity_lut);
    let mut t2 = vec![0u8; n * 4];
    let mut t3 = vec![0u8; n * 4];
    for y in 0..h {
        for x in 0..w {
            let p = ((y * w + x) * 4) as usize;
            let (a, b) = lobe_dirs(&sdm, x, y).unwrap_or(([0.0; 3], [0.0; 3]));
            let enc = |v: f32| ((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round() as u8;
            // BGRA byte order: b @0, g @1, r @2 -> shader .rgb = (x, y, z)
            t2[p] = enc(a[2]); t2[p + 1] = enc(a[1]); t2[p + 2] = enc(a[0]);
            t3[p] = enc(b[2]); t3[p + 1] = enc(b[1]); t3[p + 2] = enc(b[0]);
            t2[p + 3] = match &ana { Some((ab, _, _)) => ab[p + 2], None => 255 };
            // analytic.y = the floating sun's extra multiplier (entry 06:
            // `mul r0.w, r9.y, r0.w` AFTER the sharpening and the cascade lerp; 0.78..1.0 on the
            // sun-lit texels of the MP maps) rides .w outside the LUT corner (the corner reads 1)
            t3[p + 3] = if x < SHARPEN_LUT_SIZE && y < SHARPEN_LUT_SIZE { lut[(y * SHARPEN_LUT_SIZE + x) as usize] } else { match &ana { Some((ab, _, _)) if atlas.analytic_has_y => ab[p + 1], _ => 255 } };
        }
    }
    Ok([(t0, w, h), (t1, w, h), (t2, w, h), (t3, w, h)])
}

/// Print-only diag (HMS_H4_LMDIAG): the ENGINE-law diffuse the composed textures imply at every
/// lightmapped vertex of `bsp` - lobe magnitudes I_A / I_B (`lobe_intensity`), the SH-weighted
/// lobe luminance at the vertex normal (col * f, f = the LUT diagonal = the SH coefficient), the
/// analytic sun term (sun_rgb . n / pi, sun-visible verts) and the sun-visibility mean - so the
/// render's brightness can be judged in numbers. Returns (verts, mean diffuse lum, mean sun lum,
/// mean vis).
pub fn engine_stats(c: &H4Cache, bsp: &H4Bsp, atlas: &H4LightmapAtlas, tex: &[(Vec<u8>, u32, u32); 4]) -> (usize, f32, f32, f32) {
    let (w, h) = (atlas.w, atlas.h);
    let (mut n, mut s_lm, mut s_sun, mut s_vis) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    let (mut s_ia, mut s_ib) = (0.0f64, 0.0f64);
    let mut lums: Vec<f32> = Vec::new();
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    for (ii, inst) in bsp.instances.iter().enumerate() {
        let mi = inst.mesh.max(0) as usize;
        let Ok(Some(uvs)) = mesh_lightmap_uvs(c, bsp, mi, ii) else { continue };
        let Ok(Some(mesh)) = super::geometry::decode_mesh(c, bsp, mi) else { continue };
        let mat = bsp.instance_matrix(inst);
        for (k, uv) in uvs.iter().enumerate() {
            let Some(v) = mesh.verts.get(k) else { continue };
            let nrm = mat.transform_vector3(glam::Vec3::from(v.normal)).normalize_or_zero();
            let x = ((uv[0] * w as f32) as u32).min(w - 1);
            let y = ((uv[1] * h as f32) as u32).min(h - 1);
            let p = ((y * w + x) * 4) as usize;
            let dec = |t: &[u8]| [t[p + 2] as f32 / 255.0 * 2.0 - 1.0, t[p + 1] as f32 / 255.0 * 2.0 - 1.0, t[p] as f32 / 255.0 * 2.0 - 1.0];
            let (da, db) = (dec(&tex[2].0), dec(&tex[3].0));
            let i_a = lobe_intensity(atlas.k_dom, tex[0].0[p + 3] as f32 / 255.0);
            let i_b = lobe_intensity(atlas.k, tex[1].0[p + 3] as f32 / 255.0);
            let ca = lobe_rgb(&tex[0].0, x, y, w).map(|v| v * i_a);
            let cb = lobe_rgb(&tex[1].0, x, y, w).map(|v| v * i_b);
            let fa = lobe_sh_coeff(da, nrm.into()).max(0.0);
            let fb = lobe_sh_coeff(db, nrm.into()).max(0.0);
            let diffuse = [ca[0] * fa + cb[0] * fb, ca[1] * fa + cb[1] * fb, ca[2] * fa + cb[2] * fb];
            let vis = tex[2].0[p + 3] as f32 / 255.0;
            let ndl = nrm.dot(glam::Vec3::from(atlas.sun_dir)).max(0.0);
            let sun = lum(atlas.sun_rgb) * ndl * vis / std::f32::consts::PI;
            lums.push(lum(diffuse));
            n += 1;
            s_lm += lum(diffuse) as f64;
            s_sun += sun as f64;
            s_vis += vis as f64;
            s_ia += i_a as f64;
            s_ib += i_b as f64;
        }
    }
    if n > 0 {
        lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("HMS_H4_LMDIAG {}: {} lightmapped verts, baked lum p10/p50/p90 = {:.3}/{:.3}/{:.3} (mean {:.3}), analytic sun lum mean {:.3}, mean I_A {:.2} I_B {:.2}, sun-vis mean {:.3}, K_direct {:.3} K_indirect {:.3} sun rgb {:?} sharpen {:.2} flags {:#x}",
            bsp.name, n, lums[n / 10], lums[n / 2], lums[n * 9 / 10], s_lm / n as f64, s_sun / n as f64, s_ia / n as f64, s_ib / n as f64, s_vis / n as f64, atlas.k_dom, atlas.k, atlas.sun_rgb, atlas.shadow_sharpen, atlas.flags);
    }
    (n, (s_lm / n.max(1) as f64) as f32, (s_sun / n.max(1) as f64) as f32, (s_vis / n.max(1) as f64) as f32)
}

/// Index of the type-4 (u16x2 unorm lightmap UV) vertex buffer of instance `inst`, or None when
/// the instance is per-vertex lit. Lbsp +0x1B8[inst] @+24 (see the module doc).
pub fn instance_uv_vb(c: &H4Cache, lbsp_tag: usize, inst: usize) -> Option<i16> {
    let m = c.tag_meta(lbsp_tag)?;
    let (n, o) = c.block(m + OFF_LBSP_INSTANCE_LM)?;
    if inst >= n { return None; }
    let vb = c.data().i16_at(o + inst * INSTANCE_LM_ELEM + 24);
    (vb >= 0).then_some(vb)
}

// ---- per-VERTEX lighting --------------------------------------------------------------------
//
// From the shipped `srf_blinn` vertex shader for entry 32 (per-vertex static lighting;
// RDEF: t0 `vs_bsp_lightprobe_data_texture`, cb1 `vs_bsp_lightmap_compress_constant`, cb3
// `vs_mesh_lightmap_compress_constant`): the Lbsp +0x9C `*_vertex_565` bitmap is a 5-slice
// R5G6B5 array, 1024 texels wide, addressed by `idx = vertex_id + vs_mesh_lightmap_compress_constant.z`
// as (x = idx & 1023, y = idx >> 10). Slice roles (VS `ld` sequence):
//   slice 1 .rgb * cc.x (= Lbsp +0x1C) -> lobe A (direct) colour, slice 0 .rgb * cc.y (+0x20) -> lobe B,
//   both times exp2(-9 * slice4.y) (ONE shared alpha, no 512/511 remap on this path),
//   dir A = (s2.x, s3.x, s4.x) * 2 - 1, dir B = (s2.z, s3.z, s4.z) * 2 - 1,
//   analytic sun visibility = s3.y, extra floating-sun multiplier = s2.y (per-vertex only).
// The pixel shader (entry 32) then applies the same sharpen-LUT / SH-L1 law as the atlas path.
// The per-instance vertex offset is Lbsp +0xC8 `i32 @+12` (contiguous with the mesh
// vertex counts across instances; max end 83235 <= 1024 x 84 on wraparound, 3844 <= 1024 x 4
// on Ravine) and `i16 @+2` selects the bitmap: 0 = +0x9C vertex_565 (5 slices), 1 = +0xAC
// vertex_ao_565 (1 slice; Ravine 1024 x 76, max end 76231 - its shader is NOT decoded, those
// instances stay flat). `i16 @+6` = lighting mode (0x200 / 0x300 per-vertex, 0x400 / 0x500
// per-pixel atlas, 0x100 / 0x600 / 0x700 probe-style, 0 = none; the finer distinctions are not
// decoded).

/// Bitmap format 6 = R5G6B5 (2 B/px) - the per-vertex lighting arrays.
pub const FMT_R5G6B5: i16 = 6;

/// The decoded 5-slice per-vertex lighting array of an Lbsp (`(x, y)` texel -> rgb in 0..1).
pub struct H4PerVertexLighting {
    pub w: u32,
    pub h: u32,
    /// 5 slices, each `w * h` rgb triples.
    pub slices: Vec<Vec<[f32; 3]>>,
}

impl H4PerVertexLighting {
    fn texel(&self, slice: usize, idx: u32) -> Option<[f32; 3]> {
        let (x, y) = (idx & 1023, idx >> 10);
        if x >= self.w || y >= self.h { return None; }
        self.slices.get(slice)?.get((y * self.w + x) as usize).copied()
    }
}

/// Load the Lbsp +0x9C `*_vertex_565` array (None when the Lbsp has none).
pub fn load_pervertex(c: &H4Cache, lbsp_tag: usize) -> Result<Option<H4PerVertexLighting>> {
    let m = c.tag_meta(lbsp_tag).ok_or_else(|| anyhow!("Lbsp meta"))?;
    let Some(tag) = c.tag_ref_of(m + OFF_LBSP_VERTEX_565, b"bitm") else { return Ok(None) };
    let (info, raw) = bitmap_slices_raw(c, tag)?;
    if info.format != FMT_R5G6B5 { bail!("vertex_565 format {} (expected R5G6B5 = 6)", info.format); }
    let n = info.w as usize * info.h as usize;
    let slices = (0..info.depth.max(1) as usize).map(|s| {
        let b = &raw[s * n * 2..(s + 1) * n * 2];
        (0..n).map(|i| { let v = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]); let c = rgb565(v); [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] }).collect()
    }).collect::<Vec<Vec<[f32; 3]>>>();
    if slices.len() < 5 { bail!("vertex_565 has {} slices, need 5", slices.len()); }
    Ok(Some(H4PerVertexLighting { w: info.w as u32, h: info.h as u32, slices }))
}

/// Per-vertex lighting record of instance `inst` (Lbsp +0xC8): `(bitmap select @+2, vertex
/// offset @+12, mode @+6)`; None when the instance has no vertex offset.
pub fn instance_pervertex(c: &H4Cache, lbsp_tag: usize, inst: usize) -> Option<(i16, i32, i16)> {
    let m = c.tag_meta(lbsp_tag)?;
    let (n, o) = c.block(m + OFF_LBSP_INSTANCE_INFO)?;
    if inst >= n { return None; }
    let e = o + inst * INSTANCE_INFO_ELEM;
    let d = c.data();
    let off = d.i32_at(e + 12);
    (off >= 0).then(|| (d.i16_at(e + 2), off, d.i16_at(e + 6)))
}

/// Per-vertex baked colour (engine law, evaluated at the vertex normal so the sharpen LUT
/// is its identity diagonal): `rgb = col_A * u_A + col_B * u_B`, `a = sqrt(vis * mult)` (the
/// mesh shader squares the alpha for the analytic sun). `normals` are WORLD-space unit normals;
/// None when any vertex falls outside the array.
pub fn pervertex_colors(pv: &H4PerVertexLighting, atlas: &H4LightmapAtlas, offset: i32, normals: &[[f32; 3]]) -> Option<Vec<[f32; 4]>> {
    let mut out = Vec::with_capacity(normals.len());
    for (k, n) in normals.iter().enumerate() {
        let idx = (offset as i64 + k as i64) as u32;
        let (s0, s1, s2, s3, s4) = (pv.texel(0, idx)?, pv.texel(1, idx)?, pv.texel(2, idx)?, pv.texel(3, idx)?, pv.texel(4, idx)?);
        let i = (-9.0 * s4[1]).exp2();
        let ia = i * atlas.k_vertex;
        let ib = i * atlas.k_vertex_indirect;
        let da = [s2[0] * 2.0 - 1.0, s3[0] * 2.0 - 1.0, s4[0] * 2.0 - 1.0];
        let db = [s2[2] * 2.0 - 1.0, s3[2] * 2.0 - 1.0, s4[2] * 2.0 - 1.0];
        let fa = lobe_sh_coeff(da, *n).max(0.0);
        let fb = lobe_sh_coeff(db, *n).max(0.0);
        let vis = if atlas.floating_sun { (s3[1] * s2[1]).clamp(0.0, 1.0).sqrt() } else { 0.0 };
        out.push([s1[0] * ia * fa + s0[0] * ib * fb, s1[1] * ia * fa + s0[1] * ib * fb, s1[2] * ia * fa + s0[2] * ib * fb, vis]);
    }
    Some(out)
}

// ---- instance light PROBES (Lbsp +0xD4, 0x5C B) ---------------------------------------------
//
// Instances of lighting mode 0x100 (probe) and 0x200 / 0x300 with bitmap select 1 (probe +
// per-vertex AO) are lit by ONE probe each (Lbsp +0xC8 `i16 @+4` = probe block index), through the
// `srf_*` entry-37 shaders (VS: `ld vs_bsp_lightprobe_ao_data_texture` at
// idx = vertex_id + vs_mesh_lightmap_compress_constant.z -> o6 = (r, 2 g, b, cc.w); PS: cb3 =
// EngineModelLightingPS: regs 0-6 = the quadratic SH irradiance of the probe, reg 7.w a lobe
// scale, regs 8-11 = lobe A dir + w / colour, lobe B dir + w / colour, regs 12-13 the floating
// sun) as
//   E(N) = [SH(N) + col_A * sat(k * (0.2821 w_A + 0.3257 dot(d_A, N))) * gate_A
//               + col_B * sat(k * (0.2821 w_B + 0.3257 dot(d_B, N)))] * ao + sun * vis * sat(N.L) / pi
//   with ao = 2 * s.g, vis = s.r, sun multiplier = s.b of the +0xAC `*_vertex_ao_565` texel.
// Probe element (f16 data: |dir| = 1.000 on every probe of both MP maps):
//   +0x00 8 x f16 = lobe A dir xyz, w, colour rgb, pad;  +0x10 8 x f16 = lobe B (same shape);
//   +0x20 u32 analytic light index; +0x24 27 x f16 = SH r[9], g[9], b[9].
// PROVISIONAL: the SH evaluation order (DC, y, z, x, xy, yz, z^2, zx, x^2 - y^2 mirrors the
// PS's cb3[0..6] packing, the on-disk order is assumed identical) and the reg-7.w lobe scale
// (taken as 1) come from the shader only - the CPU packer was not located in halo4.dll.
pub const OFF_LBSP_PROBES: usize = 0xD4;
pub const PROBE_ELEM: usize = 0x5C;

#[derive(Clone, Copy, Debug)]
pub struct H4Probe {
    pub dir_a: [f32; 3],
    pub w_a: f32,
    pub col_a: [f32; 3],
    pub dir_b: [f32; 3],
    pub w_b: f32,
    pub col_b: [f32; 3],
    /// Per channel: 9 SH-ish irradiance coefficients as stored.
    pub sh: [[f32; 9]; 3],
}

/// IEEE half -> f32 (the engine's own routine is halo4.dll sub_1801BB3C0).
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut e2 = -1i32; let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            (s << 31) | (((e2 + 1 + 112) as u32) << 23) | ((m2 & 0x3FF) << 13)
        }
    } else if e == 31 { (s << 31) | 0x7F80_0000 | (m << 13) }
    else { (s << 31) | (((e + 112) as u32) << 23) | (m << 13) };
    f32::from_bits(bits)
}

/// Every probe of an Lbsp (empty when the block is absent).
pub fn load_probes(c: &H4Cache, lbsp_tag: usize) -> Vec<H4Probe> {
    let mut out = Vec::new();
    let Some(m) = c.tag_meta(lbsp_tag) else { return out };
    let Some((n, o)) = c.block(m + OFF_LBSP_PROBES) else { return out };
    let d = c.data();
    let h = |off: usize| f16_to_f32(d.u16_at(off));
    for i in 0..n {
        let e = o + i * PROBE_ELEM;
        let v: Vec<f32> = (0..16).map(|k| h(e + 2 * k)).collect();
        let mut sh = [[0.0f32; 9]; 3];
        for ch in 0..3 { for k in 0..9 { sh[ch][k] = h(e + 0x24 + 2 * (ch * 9 + k)); } }
        out.push(H4Probe {
            dir_a: [v[0], v[1], v[2]], w_a: v[3], col_a: [v[4], v[5], v[6]],
            dir_b: [v[8], v[9], v[10]], w_b: v[11], col_b: [v[12], v[13], v[14]],
            sh,
        });
    }
    out
}

// ---- AIRPROBES (Lbsp +0x3E8) + the CPU surface-probe texel sample ----------------------------
//
// halo4.dll object lighting: every placed object is lit from its
// 292-B lighting record, filled by `sub_1802822B4` -> `sub_1802E5208`:
//   1. SURFACE PROBE `sub_1802E4388`: a collision ray from the object's lighting point (bounding
//      centre + 0.2 wu when the model has no lighting marker, direction `qword_180C9380C` =
//      (0, 0, -1)) hits the BSP; `sub_1802E5A28` samples the hit's lightmap texel on the CPU:
//      lobe A dir (s2.x, s0.x, s0.y) * 2 - 1, colour = intensity.rgb * exp(-6.238 a) * K_direct
//      (= exp2(-9 a)), lobe B likewise with K_indirect, sun visibility = analytic.x. The SH
//      registers stay ZERO (`sub_1802E11C8`), the lobe scale `cb3[7].w` = 1 (`sub_1802E688C`
//      returns t / duration != 1.0) -> the model lighting entry (09 / 11 / 36 / 37) computes
//      diffuse = col_A * sat(0.2821 w_A + 0.3257 dot(d_A, N)) + col_B * sat(..B..) + sun * vis
//      * sat(N.L) / pi - NO sharpen LUT on the object path, and the SH term is 0.
//   2. AIRPROBE VOLUME `sub_1802E3900` (when the ray finds no lightmapped surface, or obje
//      flag 0x800): the Lbsp +0x3E8 block (80-B records: position, u32 volume id, u16 x2, 27 f16
//      radiance SH @+24, f16 sun visibility @+78; 113 `lightprobevolume*` points on Forge
//      Island) - the nearest probe picks the volume, up to 6 probes of that volume within
//      60 wu are blended with 1 / max(0.01, d)^3 weights; `+267 = 1` makes cb3[7].w = 0 ->
//      the lobes are OFF, diffuse = SH(N) (the packer law `probe_sh`) + sun * vis.
// Neither path is the Lbsp INSTANCE probe (+0xD4) - those light BSP instances (entry 37,
// `sub_180351F00` passes a3 = 1.0 -> lobes off, SH only); HMS keeps them as a last-resort
// stand-in for objects on maps without airprobes.
pub const OFF_LBSP_AIRPROBES: usize = 0x3E8;
pub const AIRPROBE_ELEM: usize = 80;
/// The engine's airprobe search radius (`sub_1802822B4` passes 60.0 to `sub_1802E3900`).
pub const AIRPROBE_RADIUS: f32 = 60.0;

#[derive(Clone, Copy, Debug)]
pub struct H4AirProbe {
    pub pos: [f32; 3],
    /// `lightprobevolume*` string id: probes are only blended within one volume.
    pub volume: u32,
    pub sh: [[f32; 9]; 3],
    /// f16 @+78: the probe's static sun visibility (0..1).
    pub vis: f32,
}

/// Every airprobe of an Lbsp (+0x3E8, 80-B records; empty when the block is absent).
pub fn load_airprobes(c: &H4Cache, lbsp_tag: usize) -> Vec<H4AirProbe> {
    let mut out = Vec::new();
    let Some(m) = c.tag_meta(lbsp_tag) else { return out };
    let Some((n, o)) = c.block(m + OFF_LBSP_AIRPROBES) else { return out };
    let d = c.data();
    for i in 0..n {
        let e = o + i * AIRPROBE_ELEM;
        if e + AIRPROBE_ELEM > d.len() { break; }
        let pos = [d.f32_at(e), d.f32_at(e + 4), d.f32_at(e + 8)];
        if !pos.iter().all(|v| v.is_finite()) { continue; }
        let mut sh = [[0.0f32; 9]; 3];
        for ch in 0..3 { for k in 0..9 { sh[ch][k] = f16_to_f32(d.u16_at(e + 24 + 2 * (ch * 9 + k))); } }
        out.push(H4AirProbe { pos, volume: d.u32_at(e + 12), sh, vis: f16_to_f32(d.u16_at(e + 78)).clamp(0.0, 1.0) });
    }
    out
}

/// The engine's airprobe blend at `pos` (`sub_1802E3900`): the nearest probe of any BSP picks
/// the volume; up to the 6 nearest probes of THAT volume within `AIRPROBE_RADIUS` are weighted
/// 1 / max(0.01, d)^3 (normalised). Returns (SH r/g/b x 9, sun visibility) or None when no
/// probe lies within the radius.
pub fn airprobe_sample(probes: &[H4AirProbe], pos: [f32; 3]) -> Option<([[f32; 9]; 3], f32)> {
    let d2 = |p: &H4AirProbe| (p.pos[0] - pos[0]).powi(2) + (p.pos[1] - pos[1]).powi(2) + (p.pos[2] - pos[2]).powi(2);
    let nearest = probes.iter().min_by(|a, b| d2(a).partial_cmp(&d2(b)).unwrap_or(std::cmp::Ordering::Equal))?;
    let r2 = AIRPROBE_RADIUS * AIRPROBE_RADIUS;
    if d2(nearest) >= r2 { return None; }
    let volume = nearest.volume;
    let mut cand: Vec<(f32, usize)> = probes.iter().enumerate().filter(|(_, p)| p.volume == volume).map(|(i, p)| (d2(p), i)).filter(|(d, _)| *d < r2).collect();
    cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    cand.truncate(6);
    if cand.is_empty() { return None; }
    let mut w: Vec<f32> = cand.iter().map(|(d, _)| { let dd = d.sqrt().max(0.01); 1.0 / (dd * dd * dd) }).collect();
    let sum: f32 = w.iter().sum();
    if sum <= 1e-15 { w = vec![0.0; cand.len()]; w[0] = 1.0; } else { for v in &mut w { *v /= sum; } }
    let mut sh = [[0.0f32; 9]; 3];
    let mut vis = 0.0f32;
    for ((_, i), wi) in cand.iter().zip(&w) {
        let p = &probes[*i];
        for ch in 0..3 { for k in 0..9 { sh[ch][k] += wi * p.sh[ch][k]; } }
        vis += wi * p.vis;
    }
    Some((sh, vis.clamp(0.0, 1.0)))
}

/// One lightmap texel as the engine's CPU surface-probe sampler reads it (`sub_1802E5A28`) from
/// the composed atlas textures (`compose_atlas_textures` order): (dir A, w A, colour A, dir B,
/// w B, colour B, sun visibility = analytic.x). `u`/`v` in atlas space, nearest texel.
#[allow(clippy::type_complexity)]
pub fn atlas_texel_lobes(tex: &[(Vec<u8>, u32, u32); 4], k_dom: f32, k: f32, uv: [f32; 2]) -> ([f32; 3], f32, [f32; 3], [f32; 3], f32, [f32; 3], f32) {
    let (w, h) = (tex[0].1, tex[0].2);
    let x = ((uv[0] * w as f32) as i64).clamp(0, w as i64 - 1) as u32;
    let y = ((uv[1] * h as f32) as i64).clamp(0, h as i64 - 1) as u32;
    let p = ((y * w + x) * 4) as usize;
    let dec = |t: &[u8]| [t[p + 2] as f32 / 255.0 * 2.0 - 1.0, t[p + 1] as f32 / 255.0 * 2.0 - 1.0, t[p] as f32 / 255.0 * 2.0 - 1.0];
    let (da, db) = (dec(&tex[2].0), dec(&tex[3].0));
    let wid = |d: [f32; 3]| (1.0 - d.iter().map(|v| v * v).sum::<f32>()).max(0.0).sqrt();
    let i_a = lobe_intensity(k_dom, tex[0].0[p + 3] as f32 / 255.0);
    let i_b = lobe_intensity(k, tex[1].0[p + 3] as f32 / 255.0);
    let ca = lobe_rgb(&tex[0].0, x, y, w).map(|v| (v * i_a).max(0.0));
    let cb = lobe_rgb(&tex[1].0, x, y, w).map(|v| (v * i_b).max(0.0));
    let vis = tex[2].0[p + 3] as f32 / 255.0;
    (da, wid(da), ca, db, wid(db), cb, vis)
}

/// Probe block index of instance `inst` (Lbsp +0xC8 `i16 @+4`), None when -1.
pub fn instance_probe(c: &H4Cache, lbsp_tag: usize, inst: usize) -> Option<usize> {
    let m = c.tag_meta(lbsp_tag)?;
    let (n, o) = c.block(m + OFF_LBSP_INSTANCE_INFO)?;
    if inst >= n { return None; }
    let p = c.data().i16_at(o + inst * INSTANCE_INFO_ELEM + 4);
    (p >= 0).then_some(p as usize)
}

/// Load the Lbsp +0xAC `*_vertex_ao_565` array (1 slice: r = sun visibility, g = ao / 2, b = sun
/// multiplier per vertex; the VS scales g by 2).
pub fn load_vertex_ao(c: &H4Cache, lbsp_tag: usize) -> Result<Option<H4PerVertexLighting>> {
    let m = c.tag_meta(lbsp_tag).ok_or_else(|| anyhow!("Lbsp meta"))?;
    let Some(tag) = c.tag_ref_of(m + OFF_LBSP_VERTEX_AO, b"bitm") else { return Ok(None) };
    let (info, raw) = bitmap_slices_raw(c, tag)?;
    if info.format != FMT_R5G6B5 { bail!("vertex_ao_565 format {} (expected R5G6B5 = 6)", info.format); }
    let n = info.w as usize * info.h as usize;
    let b = &raw[..(n * 2).min(raw.len())];
    let slice: Vec<[f32; 3]> = (0..b.len() / 2).map(|i| { let v = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]); let c = rgb565(v); [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] }).collect();
    Ok(Some(H4PerVertexLighting { w: info.w as u32, h: info.h as u32, slices: vec![slice] }))
}

/// The quadratic-SH part of a probe alone (no lobes), max 0 - the object lane's ambient.
/// ENGINE probe SH irradiance (halo4.dll sub_18038CFFC, the `ps_model_sh_lighting`
/// packer = `EngineModelLightingPS` regs 0..6, consumed by entry 11 / 37 as `dot(cA, (x, y, z, 1)) +
/// dot(cB, (xy, yz, zz, zx)) + cC * (x^2 - y^2)`): the 9 f16 per channel are standard RADIANCE SH
/// coefficients (L00, L1-1, L10, L11, L2-2, L2-1, L20, L21, L22) and the packer applies the
/// cosine-convolution constants over pi with the sun-dir sign convention (x, y, yz, zx negated):
///   cA = (-0.3257 c3, -0.3257 c1, 0.3257 c2, 0.2821 c0 - 0.0788 c6)
///   cB = (0.2731 c4, -0.2731 c5, 0.2364 c6, -0.2731 c7),  cC = 0.1366 c8
/// (sqrt(pi) = s: 1/(2s), sqrt(3)/(3s), sqrt(15)/(8s), sqrt(5)/(16s), 3 sqrt(5)/(16s), sqrt(15)/(16s)).
/// Using the raw coefficients as the polynomial instead (DC x 3.5, wrong signs) lights the Forge
/// pieces and probe-lit instances ~1.8 stops too bright.
pub fn probe_sh(p: &H4Probe, n: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = (n[0], n[1], n[2]);
    let sp = std::f32::consts::PI.sqrt();
    let k0 = 1.0 / (2.0 * sp);           // 0.2821
    let k1 = 3f32.sqrt() / (3.0 * sp);   // 0.3257
    let k2 = 15f32.sqrt() / (8.0 * sp);  // 0.2731
    let k3 = 5f32.sqrt() / (16.0 * sp);  // 0.0788
    let mut out = [0.0f32; 3];
    for ch in 0..3 {
        let c = &p.sh[ch];
        let a = [-(k1 * c[3]), -(k1 * c[1]), k1 * c[2], k0 * c[0] - k3 * c[6]];
        let b = [k2 * c[4], -(k2 * c[5]), 3.0 * k3 * c[6], -(k2 * c[7])];
        let cc = 0.5 * k2 * c[8];
        out[ch] = (a[0] * x + a[1] * y + a[2] * z + a[3] + b[0] * x * y + b[1] * y * z + b[2] * z * z + b[3] * z * x + cc * (x * x - y * y)).max(0.0);
    }
    out
}

/// Probe irradiance at world normal `n` (the entry-37 PS law above, k = 1).
pub fn probe_irradiance(p: &H4Probe, n: [f32; 3]) -> [f32; 3] {
    let (x, y, z) = (n[0], n[1], n[2]);
    let lobe = |d: [f32; 3], w: f32| (0.2820948 * w + 0.325735 * (d[0] * x + d[1] * y + d[2] * z)).clamp(0.0, 1.0);
    let (fa, fb) = (lobe(p.dir_a, p.w_a), lobe(p.dir_b, p.w_b));
    let sh = probe_sh(p, n);
    let mut out = [0.0f32; 3];
    for ch in 0..3 {
        out[ch] = (sh[ch] + p.col_a[ch] * fa + p.col_b[ch] * fb).max(0.0);
    }
    out
}

/// Per-vertex colour of a probe-lit instance: rgb = probe irradiance(N) * ao,
/// a = sqrt(vis * mult) (the shader squares it for the floating sun). `ao` = the vertex_ao array
/// + this instance's vertex offset; None = ao 1 / vis 1 (mode 0x100 instances). `normals` are
/// WORLD-space unit normals.
pub fn probe_vertex_colors(p: &H4Probe, ao: Option<(&H4PerVertexLighting, i32)>, floating_sun: bool, normals: &[[f32; 3]]) -> Option<Vec<[f32; 4]>> {
    let mut out = Vec::with_capacity(normals.len());
    for (k, n) in normals.iter().enumerate() {
        let (aov, vis) = match ao {
            Some((pv, off)) => { let s = pv.texel(0, (off as i64 + k as i64) as u32)?; ((2.0 * s[1]).clamp(0.0, 1.0), (s[0] * s[2]).clamp(0.0, 1.0)) }
            None => (1.0, 1.0),
        };
        let e = probe_irradiance(p, *n);
        out.push([e[0] * aov, e[1] * aov, e[2] * aov, if floating_sun { vis.sqrt() } else { 0.0 }]);
    }
    Some(out)
}

/// Mesh index recorded for instance `inst` in Lbsp +0xC8 (@+20).
pub fn instance_mesh(c: &H4Cache, lbsp_tag: usize, inst: usize) -> Option<i32> {
    let m = c.tag_meta(lbsp_tag)?;
    let (n, o) = c.block(m + OFF_LBSP_INSTANCE_INFO)?;
    if inst >= n { return None; }
    Some(c.data().i32_at(o + inst * INSTANCE_INFO_ELEM + 20))
}

/// Atlas-space lightmap UVs (uv2 in [0,1]) for `mesh_idx` as lit through instance `inst`, which
/// reads its own type-4 stream; None when the instance is per-vertex lit. (Clusters have no
/// per-pixel path on the shipped maps - every lit mesh is an instance.)
pub fn mesh_lightmap_uvs(c: &H4Cache, bsp: &H4Bsp, mesh_idx: usize, inst: usize) -> Result<Option<Vec<[f32; 2]>>> {
    let lbsp = bsp.lbsp_tag.ok_or_else(|| anyhow!("bsp has no Lbsp"))?;
    let g = bsp.geometry.as_ref().ok_or_else(|| anyhow!("bsp has no geometry resource"))?;
    let e = c.resources.get(g.entry).ok_or_else(|| anyhow!("geometry entry"))?;
    if let Some(mi) = instance_mesh(c, lbsp, inst) {
        if mi != mesh_idx as i32 { bail!("instance {inst} is mesh {mi}, not {mesh_idx}"); }
    }
    let vb_idx = match instance_uv_vb(c, lbsp, inst) { Some(v) => v, None => return Ok(None) };
    let vb = g.vbs.get(vb_idx as usize).ok_or_else(|| anyhow!("uv vb {vb_idx} out of range"))?;
    if vb.kind != 4 || vb.stride != 4 { bail!("vb {vb_idx} is type {} stride {}, not a lightmap UV stream", vb.kind, vb.stride); }
    let sm = c.tag_meta(bsp.sbsp_tag).ok_or_else(|| anyhow!("sbsp meta"))?;
    if let Some((n, o)) = c.block(sm + OFF_SBSP_MESHES) {
        if mesh_idx < n {
            let vb0 = c.data().i16_at(o + mesh_idx * MESH_ELEM + 0x18);
            if let Some(pv) = g.vertex_buffer(vb0.max(0)) {
                if vb0 >= 0 && pv.count != vb.count { bail!("mesh {mesh_idx} has {} verts, uv stream {vb_idx} has {}", pv.count, vb.count); }
            }
        }
    }
    let raw = c.stream_bytes(e, vb.addr, vb.size as usize)?;
    Ok(Some((0..vb.count as usize).map(|k| [raw.u16_at(k * 4) as f32 / 65535.0, raw.u16_at(k * 4 + 2) as f32 / 65535.0]).collect()))
}

// ---- block decoders -------------------------------------------------------------------------

/// 8-entry BC4 palette (unorm 0..255 or snorm -127..127 as f32).
fn bc4_palette(b0: u8, b1: u8, snorm: bool) -> [f32; 8] {
    let (a0, a1) = if snorm { ((b0 as i8).max(-127) as f32, (b1 as i8).max(-127) as f32) } else { (b0 as f32, b1 as f32) };
    let mut p = [0.0f32; 8];
    p[0] = a0;
    p[1] = a1;
    if a0 > a1 {
        for i in 1..7 { p[i + 1] = ((7 - i) as f32 * a0 + i as f32 * a1) / 7.0; }
    } else {
        for i in 1..5 { p[i + 1] = ((5 - i) as f32 * a0 + i as f32 * a1) / 5.0; }
        p[6] = if snorm { -127.0 } else { 0.0 };
        p[7] = if snorm { 127.0 } else { 255.0 };
    }
    p
}

/// Decode one 8-byte BC4 block into 16 values in 0..1 (unorm) or -1..1 (snorm).
fn bc4_block(b: &[u8], snorm: bool) -> [f32; 16] {
    let pal = bc4_palette(b[0], b[1], snorm);
    let mut bits: u64 = 0;
    for i in 0..6 { bits |= (b[2 + i] as u64) << (8 * i); }
    let mut out = [0.0f32; 16];
    for (k, o) in out.iter_mut().enumerate() {
        let v = pal[((bits >> (3 * k)) & 7) as usize];
        *o = if snorm { v / 127.0 } else { v / 255.0 };
    }
    out
}

fn rgb565(c: u16) -> [f32; 3] {
    [((c >> 11) & 31) as f32 * (255.0 / 31.0), ((c >> 5) & 63) as f32 * (255.0 / 63.0), (c & 31) as f32 * (255.0 / 31.0)]
}

/// BC3 (DXT5) -> BGRA8. The colour block is always decoded in 4-colour mode (BC3 semantics).
pub fn decode_bc3_bgra(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let mut out = vec![0u8; (w * h * 4) as usize];
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * 16;
            if o + 16 > data.len() { break; }
            let b = &data[o..o + 16];
            let alpha = bc4_block(&b[0..8], false);
            let c0 = u16::from_le_bytes([b[8], b[9]]);
            let c1 = u16::from_le_bytes([b[10], b[11]]);
            let (p0, p1) = (rgb565(c0), rgb565(c1));
            let pal = [p0, p1, [0.0; 3], [0.0; 3]].map(|c| c);
            let mut pal = pal;
            for k in 0..3 {
                pal[2][k] = (2.0 * p0[k] + p1[k]) / 3.0;
                pal[3][k] = (p0[k] + 2.0 * p1[k]) / 3.0;
            }
            let bits = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
            for k in 0..16 {
                let (x, y) = (bx as u32 * 4 + (k % 4) as u32, by as u32 * 4 + (k / 4) as u32);
                if x >= w || y >= h { continue; }
                let c = pal[((bits >> (2 * k)) & 3) as usize];
                let p = ((y * w + x) * 4) as usize;
                out[p] = c[2].round() as u8;
                out[p + 1] = c[1].round() as u8;
                out[p + 2] = c[0].round() as u8;
                out[p + 3] = (alpha[k] * 255.0).round() as u8;
            }
        }
    }
    out
}

/// BC4 (one unorm channel) -> BGRA8 with r = x, g = b = 0, a = 255.
pub fn decode_bc4_bgra(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let mut out = vec![0u8; (w * h * 4) as usize];
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * 8;
            if o + 8 > data.len() { break; }
            let xs = bc4_block(&data[o..o + 8], false);
            for k in 0..16 {
                let (x, y) = (bx as u32 * 4 + (k % 4) as u32, by as u32 * 4 + (k / 4) as u32);
                if x >= w || y >= h { continue; }
                let p = ((y * w + x) * 4) as usize;
                out[p + 2] = (xs[k].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[p + 3] = 255;
            }
        }
    }
    out
}

/// BC5 (DXN) -> BGRA8 with r = x, g = y, b = 0, a = 255. Snorm values map to (v + 1) / 2.
pub fn decode_bc5_bgra(data: &[u8], w: u32, h: u32, snorm: bool) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let mut out = vec![0u8; (w * h * 4) as usize];
    let enc = |v: f32| -> u8 { let u = if snorm { (v + 1.0) * 0.5 } else { v }; (u.clamp(0.0, 1.0) * 255.0).round() as u8 };
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * 16;
            if o + 16 > data.len() { break; }
            let xs = bc4_block(&data[o..o + 8], snorm);
            let ys = bc4_block(&data[o + 8..o + 16], snorm);
            for k in 0..16 {
                let (x, y) = (bx as u32 * 4 + (k % 4) as u32, by as u32 * 4 + (k / 4) as u32);
                if x >= w || y >= h { continue; }
                let p = ((y * w + x) * 4) as usize;
                out[p] = 0;
                out[p + 1] = enc(ys[k]);
                out[p + 2] = enc(xs[k]);
                out[p + 3] = 255;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;
    use crate::h4::geometry::{decode_mesh, load_bsp};

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Output folder of the dump tools below (HMS_H4_SCRATCH, else the temp dir).
    fn scratch() -> std::path::PathBuf {
        let p = std::env::var("HMS_H4_SCRATCH").map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("hms_h4_lightmaps"));
        let _ = std::fs::create_dir_all(&p);
        p
    }

    struct Stats { mean: [f64; 4], min: [f64; 4], max: [f64; 4], hist: [[u32; 10]; 4] }

    /// Per-channel mean/min/max + 10-bin histogram of a BGRA buffer (channels reported as r,g,b,a).
    fn channel_stats(bgra: &[u8]) -> Stats {
        let n = (bgra.len() / 4).max(1) as f64;
        let mut s = Stats { mean: [0.0; 4], min: [1.0; 4], max: [0.0; 4], hist: [[0; 10]; 4] };
        for px in bgra.chunks_exact(4) {
            for (ci, &src) in [2usize, 1, 0, 3].iter().enumerate() {
                let v = px[src] as f64 / 255.0;
                s.mean[ci] += v / n;
                s.min[ci] = s.min[ci].min(v);
                s.max[ci] = s.max[ci].max(v);
                s.hist[ci][((v * 10.0) as usize).min(9)] += 1;
            }
        }
        s
    }

    fn print_stats(label: &str, bgra: &[u8]) {
        let s = channel_stats(bgra);
        let n = (bgra.len() / 4).max(1) as f64;
        for (ci, name) in ["r", "g", "b", "a"].iter().enumerate() {
            let h: Vec<String> = s.hist[ci].iter().map(|v| format!("{:.3}", *v as f64 / n)).collect();
            eprintln!("  {label} {name}: mean {:.4} min {:.3} max {:.3} hist [{}]", s.mean[ci], s.min[ci], s.max[ci], h.join(" "));
        }
    }

    /// Load every per-pixel atlas of a map, decode, print stats, and check the instance UV
    /// streams land on chart texels. Returns (instances with UVs, verts, chart hits).
    fn check_map(c: &H4Cache, min_hit: f64) -> (usize, usize, usize) {
        let mut totals = (0usize, 0usize, 0usize);
        for sbsp in c.find_tags(b"sbsp") {
            let bsp = load_bsp(c, sbsp).unwrap();
            let Some(lbsp) = bsp.lbsp_tag else { continue };
            let atlas = match load_atlas(c, lbsp) {
                Ok(a) => a,
                Err(e) => { eprintln!("{}: no per-pixel atlas ({e})", bsp.name); continue; }
            };
            eprintln!("{}: atlas {}x{} dm {} slices sdm {} slices k {:.4} k_dom {:.4} k_vertex {:.4} sun_dir {:?} sun_rgb {:?}",
                bsp.name, atlas.w, atlas.h, atlas.dm_slices, atlas.sdm_slices, atlas.k, atlas.k_dom, atlas.k_vertex, atlas.sun_dir, atlas.sun_rgb);
            // interiors (m10_crash) bake no sun: direction and colour are all zero
            let sl = atlas.sun_dir.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(sl < 1e-3 || (sl - 1.0).abs() < 1e-3, "sun dir neither zero nor unit: {:?}", atlas.sun_dir);
            assert!(atlas.k >= 0.0 && atlas.k_dom >= 0.0 && atlas.k_dom.is_finite());
            let dm = dm_slices_bgra(c, &atlas).unwrap();
            assert_eq!(dm.len(), atlas.dm_slices as usize);
            let sdm = sdm_slices_bgra(c, &atlas).unwrap();
            assert_eq!(sdm.len(), atlas.sdm_slices as usize);
            for (i, (b, w, h)) in dm.iter().enumerate() {
                assert_eq!((*w, *h, b.len()), (atlas.w, atlas.h, (atlas.w * atlas.h * 4) as usize));
                print_stats(&format!("intensity[{i}]"), b);
            }
            for (i, (b, w, h)) in sdm.iter().enumerate() {
                assert_eq!((*w, *h, b.len()), (atlas.w, atlas.h, (atlas.w * atlas.h * 4) as usize));
                print_stats(&format!("direction[{i}] (snorm->unorm)"), b);
            }
            if let Some((b, w, h)) = analytic_bgra(c, &atlas).unwrap() {
                assert_eq!((w, h), (atlas.w, atlas.h));
                print_stats("analytic", &b);
            }
            // UVs: every lightmapped instance's uv2 is in [0,1] and hits a non-padding texel
            let (mut n_inst, mut n_verts, mut hits) = (0usize, 0usize, 0usize);
            let mut sun_dot = (0.0f64, 0usize);
            let ana = analytic_bgra(c, &atlas).unwrap();
            for (ii, inst) in bsp.instances.iter().enumerate() {
                let mi = inst.mesh.max(0) as usize;
                let Some(uvs) = mesh_lightmap_uvs(c, &bsp, mi, ii).unwrap() else { continue };
                n_inst += 1;
                let mesh = decode_mesh(c, &bsp, mi).unwrap();
                let mat = bsp.instance_matrix(inst);
                for (k, uv) in uvs.iter().enumerate() {
                    assert!((0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]), "uv out of range {uv:?}");
                    let x = ((uv[0] * atlas.w as f32) as u32).min(atlas.w - 1);
                    let y = ((uv[1] * atlas.h as f32) as u32).min(atlas.h - 1);
                    n_verts += 1;
                    if dm[0].0[((y * atlas.w + x) * 4 + 3) as usize] > 0 { hits += 1; }
                    // lobe A points at the sun where the analytic map says the sun is visible
                    if let (Some(m), Some((ab, _, _))) = (mesh.as_ref(), ana.as_ref()) {
                        if ab[((y * atlas.w + x) * 4 + 2) as usize] > 128 {
                            if let Some(v) = m.verts.get(k) {
                                let n = mat.transform_vector3(glam::Vec3::from(v.normal)).normalize_or_zero();
                                if n.dot(glam::Vec3::from(atlas.sun_dir)) > 0.3 {
                                    if let Some((a, _)) = lobe_dirs(&sdm, x, y) {
                                        let a = glam::Vec3::from(a);
                                        if a.length() > 0.1 { sun_dot.0 += a.normalize().dot(glam::Vec3::from(atlas.sun_dir)) as f64; sun_dot.1 += 1; }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let hit = hits as f64 / n_verts.max(1) as f64;
            let sd = sun_dot.0 / sun_dot.1.max(1) as f64;
            eprintln!("  {} lightmapped instances, {} verts, chart-hit {:.4}, lobe-A.sun on {} sun-lit verts = {:.3}", n_inst, n_verts, hit, sun_dot.1, sd);
            assert!(n_inst > 0, "no lightmapped instances");
            assert!(hit >= min_hit, "chart hit {hit}");
            assert!(sun_dot.1 == 0 || sd > 0.4, "lobe A does not point at the sun ({sd})");
            totals.0 += n_inst;
            totals.1 += n_verts;
            totals.2 += hits;
        }
        totals
    }

    #[test]
    fn wraparound_atlas() {
        let Some(c) = open("wraparound.map") else { return };
        let (n_inst, _, _) = check_map(&c, 0.99);
        assert_eq!(n_inst, 553, "553 type-4 UV streams = 553 lightmapped instances");
        let sbsp = c.find_tags(b"sbsp")[0];
        let bsp = load_bsp(&c, sbsp).unwrap();
        let atlas = load_atlas(&c, bsp.lbsp_tag.unwrap()).unwrap();
        assert_eq!((atlas.w, atlas.h, atlas.dm_slices, atlas.sdm_slices), (1792, 1792, 2, 3));
        assert!((atlas.k - 14.42).abs() < 0.01 && (atlas.k_dom - 277.7).abs() < 0.1 && (atlas.k_vertex - 21.84).abs() < 0.01);
        // engine-law inputs: per-vertex pair, floating sun on (flags 0x4202), scnr
        // structure_bsps[+184] sharpening 1.0, the sharpen LUT is present and its diagonal is the
        // identity (texel (i, i) ~ 255 * (i + 0.5) / 64)
        assert!((atlas.k_vertex_indirect - 21.84).abs() < 0.01);
        assert_eq!(atlas.flags, 0x4202);
        assert!(atlas.floating_sun);
        assert!((atlas.shadow_sharpen - 1.0).abs() < 1e-6);
        let lut = sharpen_lut(&c).expect("rasterizer\\sharpen_falloff_64");
        assert_eq!(lut.len(), (SHARPEN_LUT_SIZE * SHARPEN_LUT_SIZE) as usize);
        for i in [0u32, 8, 16, 32, 48, 56] {
            let v = lut[(i * SHARPEN_LUT_SIZE + i) as usize] as f32 / 255.0;
            assert!((v - (i as f32 + 0.5) / 64.0).abs() < 0.03, "LUT diagonal {i}: {v}");
        }
        assert_eq!(lut[0], 0);
        // lobe_intensity: exactly K at alpha 0, exactly 0 at alpha 1, exp2 slope in between
        assert!((lobe_intensity(277.7, 0.0) - 277.7).abs() < 1e-3 && lobe_intensity(277.7, 1.0).abs() < 1e-6);
        assert!((lobe_intensity(1.0, 0.5) - (512.0 * 2f32.powf(-4.5) - 1.0) / 511.0).abs() < 1e-6);
        // the composed dir-B texture carries the LUT in its top-left .w corner
        let tex = compose_atlas_textures(&c, &atlas).unwrap();
        assert_eq!(tex[3].0[3], lut[0]);
        assert_eq!(tex[3].0[((32 * atlas.w + 32) * 4 + 3) as usize], lut[32 * 64 + 32]);
        // outside the corner .w = analytic.y (the sun's extra multiplier; 0.54 mean
        // on wraparound, 0.78 on its sun-lit texels), 255 where the map has no y channel
        {
            let ana = analytic_bgra(&c, &atlas).unwrap().unwrap();
            let p = ((100 * atlas.w + 100) * 4) as usize;
            assert_eq!(tex[3].0[p + 3], if atlas.analytic_has_y { ana.0[p + 1] } else { 255 });
            assert!(atlas.analytic_has_y);
        }
        assert!((atlas.sun_dir[1] - 0.884).abs() < 0.001);
        // instance UV bookkeeping: unique VBs, vertex counts agree with the mesh
        let mut vbs: Vec<i16> = (0..bsp.instances.len()).filter_map(|i| instance_uv_vb(&c, atlas.lbsp_tag, i)).collect();
        assert_eq!(vbs.len(), 553);
        vbs.sort();
        vbs.dedup();
        assert_eq!(vbs.len(), 553);
        for i in 0..bsp.instances.len() {
            assert_eq!(instance_mesh(&c, atlas.lbsp_tag, i), Some(bsp.instances[i].mesh as i32), "Lbsp +0xC8[{i}] mesh");
        }
        // clusters carry no atlas selector on this map (Lbsp +0xBC, 8 B: i16 -1, i16 -1, i32 0)
        let m = c.tag_meta(atlas.lbsp_tag).unwrap();
        let (n, o) = c.block(m + OFF_LBSP_CLUSTERS).unwrap();
        assert_eq!(n, 8);
        for i in 0..n { assert_eq!(c.data().i16_at(o + i * CLUSTER_ELEM), -1); }
        // HMS_H4_DUMP=1: raw BGRA dumps of the intensity slices into the scratch dir (RE aid)
        if std::env::var("HMS_H4_DUMP").is_ok() {
            let dm = dm_slices_bgra(&c, &atlas).unwrap();
            for (i, (b, w, h)) in dm.iter().enumerate() { std::fs::write(scratch().join(format!("wrap_dm{i}_{w}x{h}.bgra")), b).unwrap(); }
        }
    }

    #[test]
    fn ravine_atlases() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let (n_inst, n_verts, _) = check_map(&c, 0.95);
        assert!(n_inst > 0 && n_verts > 0);
    }

    /// Campaign: the BSP Lbsps live under environments\solo; object Lbsps have no +0x3C map and
    /// must be rejected cleanly. Every per-pixel atlas that exists decodes to its byte size.
    #[test]
    fn m10_crash_atlases() {
        let Some(c) = open("m10_crash.map") else { return };
        let mut ok = 0usize;
        let mut none = 0usize;
        for lbsp in c.find_tags(b"Lbsp") {
            match load_atlas(&c, lbsp) {
                Ok(a) => {
                    ok += 1;
                    let dm = dm_slices_bgra(&c, &a).unwrap();
                    let sdm = sdm_slices_bgra(&c, &a).unwrap();
                    assert_eq!(dm.len(), a.dm_slices as usize);
                    assert_eq!(sdm.len(), 3, "{}: direction slices", c.tag_name(lbsp));
                    for (b, _, _) in sdm.iter().chain(dm.iter()) { assert_eq!(b.len(), (a.w * a.h * 4) as usize); }
                }
                Err(_) => none += 1,
            }
        }
        eprintln!("m10_crash: {ok} per-pixel atlases, {none} Lbsps without one");
        assert!(ok > 0);
        let (n_inst, _, _) = check_map(&c, 0.9);
        eprintln!("m10_crash lightmapped instances: {n_inst}");
    }

    /// The DLC-era atlases ship every stream at twice the slice total (real slices
    /// first, then padding): ca_basin's 1536^2 intensity / direction / analytic maps must decode.
    #[test]
    fn dlc_padded_atlas_streams_decode() {
        let Some(c) = open("ca_basin.map") else { return };
        let sbsp = c.find_tags(b"sbsp").into_iter().find(|&t| c.tag_name(t).ends_with("ca_basin_bsp01")).expect("basin bsp");
        let bsp = load_bsp(&c, sbsp).unwrap();
        let atlas = load_atlas(&c, bsp.lbsp_tag.expect("lbsp")).expect("basin atlas");
        assert_eq!((atlas.w, atlas.h, atlas.dm_slices, atlas.sdm_slices), (1536, 1536, 2, 3));
        assert!((atlas.k_dom - 13.366935).abs() < 1e-4 && (atlas.k - 7.233471).abs() < 1e-4);
        let dm = dm_slices_bgra(&c, &atlas).unwrap();
        assert_eq!(dm.len(), 2);
        let sdm = sdm_slices_bgra(&c, &atlas).unwrap();
        assert_eq!(sdm.len(), 3);
        // the real slices are not padding: a spread of intensity alphas
        let n = dm[0].0.len() / 4;
        let mean_a = dm[0].0.chunks(4).map(|p| p[3] as f64).sum::<f64>() / n as f64;
        assert!(mean_a > 40.0 && mean_a < 250.0, "slice 0 alpha mean {mean_a}");
        assert!(analytic_bgra(&c, &atlas).unwrap().is_some());
    }

    /// RE tool: atlas stats + decoded lobe irradiance means of HMS_H4_PS_MAP (the
    /// engine-law I = K (512 2^-9a - 1) / 511 over chart texels), to compare maps' absolute scale.
    #[test]
    #[ignore]
    fn dump_atlas_irradiance() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let Some(c) = open(&map) else { return };
        for sbsp in c.find_tags(b"sbsp") {
            let bsp = load_bsp(&c, sbsp).unwrap();
            let Some(lbsp) = bsp.lbsp_tag else { continue };
            let Ok(atlas) = load_atlas(&c, lbsp) else { continue };
            let dm = dm_slices_bgra(&c, &atlas).unwrap();
            let ana = analytic_bgra(&c, &atlas).unwrap();
            for (k, (b, w, h)) in dm.iter().enumerate().take(2) {
                let kk = if k == 0 { atlas.k_dom } else { atlas.k };
                let (mut n, mut sum_i, mut sum_lum, mut sum_a, mut hist) = (0usize, 0.0f64, 0.0f64, 0.0f64, [0usize; 10]);
                let mut sum_vis = 0.0f64;
                for i in 0..(*w * *h) as usize {
                    let p = &b[i * 4..i * 4 + 4];
                    let a = p[3] as f64 / 255.0;
                    // padding texels: alpha 1 (I = 0) and black
                    if p[3] == 255 && p[0] == 0 && p[1] == 0 && p[2] == 0 { continue; }
                    let ii = kk as f64 * (512.0 * (-9.0 * a).exp2() - 1.0) / 511.0;
                    let lum = (0.3086 * p[2] as f64 + 0.6094 * p[1] as f64 + 0.082 * p[0] as f64) / 255.0;
                    n += 1; sum_i += ii; sum_lum += lum * ii; sum_a += a;
                    hist[((a * 10.0) as usize).min(9)] += 1;
                    if let Some((ab, _, _)) = &ana { sum_vis += ab[i * 4 + 2] as f64 / 255.0; }
                }
                let nn = n.max(1) as f64;
                eprintln!("{}: lobe {} K {:.3}: {} chart texels, mean a {:.3}, mean I {:.3}, mean luma*I {:.4}, sun vis {:.3}, a hist {:?}", bsp.name, if k == 0 { "A(direct)" } else { "B(indirect)" }, kk, n, sum_a / nn, sum_i / nn, sum_lum / nn, sum_vis / nn, hist.iter().map(|v| format!("{:.2}", *v as f64 / nn)).collect::<Vec<_>>());
            }
            eprintln!("  sun {:?} floating {} flags {:#x}", atlas.sun_rgb, atlas.floating_sun, atlas.flags);
        }
    }

    /// DXN direction map: 3 slices x (w*h) bytes on disk -> 3 x (w*h*4) BGRA; a synthetic block
    /// round-trips both palette modes and the snorm mapping.
    #[test]
    fn dxn_direction_decodes_to_size() {
        // synthetic: 4x4, one BC5 block, x = snorm 127 (1.0) flat, y = snorm -127 (-1.0) flat
        let mut blk = [0u8; 16];
        blk[0] = 127; blk[1] = 127; // x endpoints equal -> 6-entry mode, indices 0 -> 127
        blk[8] = 0x81; blk[9] = 0x81; // -127
        let out = decode_bc5_bgra(&blk, 4, 4, true);
        assert_eq!(out.len(), 64);
        assert!(out.chunks_exact(4).all(|p| p[2] == 255 && p[1] == 0 && p[3] == 255));
        let outu = decode_bc5_bgra(&blk, 4, 4, false);
        assert!(outu.chunks_exact(4).all(|p| p[2] == 127 && p[1] == 129));
        // BC3 alpha block in 8-entry mode: a0 = 255 > a1 = 0, all indices 0 -> alpha 255, colour c0
        let mut b3 = [0u8; 16];
        b3[0] = 255; b3[8] = 0xFF; b3[9] = 0xFF; // c0 = white
        let o3 = decode_bc3_bgra(&b3, 4, 4);
        assert!(o3.chunks_exact(4).all(|p| p == [255, 255, 255, 255]));
        let Some(c) = open("wraparound.map") else { return };
        let sbsp = c.find_tags(b"sbsp")[0];
        let bsp = load_bsp(&c, sbsp).unwrap();
        let atlas = load_atlas(&c, bsp.lbsp_tag.unwrap()).unwrap();
        let (info, raw) = bitmap_slices_raw(&c, atlas.sdm_tag).unwrap();
        assert_eq!((info.format, info.depth, info.kind), (FMT_DXN, 3, 3));
        assert_eq!(raw.len(), 3 * 1792 * 1792, "3 BC5 slices at 1 B/px");
        let sdm = sdm_slices_bgra(&c, &atlas).unwrap();
        assert_eq!(sdm.iter().map(|s| s.0.len()).sum::<usize>(), 3 * 1792 * 1792 * 4);
        // the intensity map is a 2-slice DXT5 array of the same size
        let (i2, raw2) = bitmap_slices_raw(&c, atlas.dm_tag).unwrap();
        assert_eq!((i2.format, i2.depth, raw2.len()), (FMT_DXT5, 2, 2 * 1792 * 1792));
    }

    /// RE tool: dump every pixel-shader DXBC blob of the `mats` whose name contains
    /// HMS_H4_PS_MATS (default `srf_blinn`) on HMS_H4_PS_MAP (default wraparound.map) into the
    /// scratch dir as `<mats-leaf>_ps<entry>_<A|B>.dxbc` plus the RDEF resource names, so an
    /// external disassembler can recover the lightmap sampling law. Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_ps_blobs() {
        use crate::h4::materials::{MTSB_REC_DXBC_A, MTSB_REC_DXBC_B, MTSB_SHADER_REC, OFF_MATS_MTSB, OFF_MATS_PS_ENTRIES, OFF_MTSB_PS};
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let want = std::env::var("HMS_H4_PS_MATS").unwrap_or_else(|_| "srf_blinn".into());
        let Some(c) = open(&map) else { return };
        let out = scratch();
        let d = c.data();
        let mut dumped = 0usize;
        for mats in c.find_tags(b"mats") {
            let name = c.tag_name(mats).to_string();
            if !name.contains(&want) { continue; }
            let leaf = name.rsplit('\\').next().unwrap_or(&name).to_string();
            let Some(mm) = c.tag_meta(mats) else { continue };
            let Some(mtsb) = c.tag_ref_of(mm + OFF_MATS_MTSB, b"mtsb") else { continue };
            let Some(mb) = c.tag_meta(mtsb) else { continue };
            let Some((ne, pe)) = c.block(mm + OFF_MATS_PS_ENTRIES) else { continue };
            let Some((np, pp)) = c.block(mb + OFF_MTSB_PS) else { continue };
            eprintln!("mats {name}: {ne} PS entry points, mtsb {} ({np} PS records)", c.tag_name(mtsb));
            for k in 0..ne {
                let hash = d.i32_at(pe + k * 8);
                let idx = d.i32_at(pe + k * 8 + 4);
                if idx < 0 || idx as usize >= np { continue; }
                let rec = pp + idx as usize * MTSB_SHADER_REC;
                for (tag, td) in [("A", MTSB_REC_DXBC_A), ("B", MTSB_REC_DXBC_B)] {
                    let size = d.i32_at(rec + td);
                    let ptr = d.u32_at(rec + td + 12);
                    if size <= 0 || ptr == 0 || ptr == 0xCDCD_CDCD { continue; }
                    let Some(o) = c.meta_off(ptr) else { continue };
                    let blob = &d[o..o + size as usize];
                    let p = out.join(format!("{leaf}_ps{k:02}_{tag}.dxbc"));
                    std::fs::write(&p, blob).unwrap();
                    dumped += 1;
                    eprintln!("  entry {k:02} hash {hash:#x} rec {idx} blob {tag} {size} B -> {}", p.display());
                }
            }
            // HMS_H4_PS_VS=1 also dumps the vertex shaders: +0x34 = 38 entry points x block of
            // {vertex type} x {hash, VS index}
            if std::env::var("HMS_H4_PS_VS").is_ok() {
                use crate::h4::materials::{OFF_MATS_VS_ENTRIES, OFF_MTSB_VS};
                let (Some((nv, pv)), Some((nrec, prec))) = (c.block(mm + OFF_MATS_VS_ENTRIES), c.block(mb + OFF_MTSB_VS)) else { continue };
                for k in 0..nv {
                    let Some((nt, pt)) = c.block(pv + k * 12) else { continue };
                    for t in 0..nt {
                        let idx = d.i32_at(pt + t * 8 + 4);
                        if idx < 0 || idx as usize >= nrec { continue; }
                        let rec = prec + idx as usize * MTSB_SHADER_REC;
                        for (tag, td) in [("A", MTSB_REC_DXBC_A), ("B", MTSB_REC_DXBC_B)] {
                            let size = d.i32_at(rec + td);
                            let ptr = d.u32_at(rec + td + 12);
                            if size <= 0 || ptr == 0 || ptr == 0xCDCD_CDCD { continue; }
                            let Some(o) = c.meta_off(ptr) else { continue };
                            let p = out.join(format!("{leaf}_vs{k:02}_t{t:02}_{tag}.dxbc"));
                            std::fs::write(&p, &d[o..o + size as usize]).unwrap();
                            dumped += 1;
                        }
                    }
                }
            }
        }
        eprintln!("dumped {dumped} blobs to {}", out.display());
    }

    /// RE tool: print every Lbsp atlas bitmap element (w, h, depth, type, format,
    /// levels, total) with its resource definition streams (size, fixup) and the D3D11 desc bytes
    /// - the DLC maps ship intensity streams twice the 2-slice size. Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_lbsp_bitmaps() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let Some(c) = open(&map) else { return };
        let d = c.data();
        for lbsp in c.find_tags(b"Lbsp") {
            let Some(m) = c.tag_meta(lbsp) else { continue };
            eprintln!("Lbsp {}: flags {:#06x} K {:?}", c.tag_name(lbsp), d.u16_at(m), (5..9).map(|k| d.f32_at(m + 4 * k)).collect::<Vec<_>>());
            for (label, off) in [("intensity", OFF_LBSP_INTENSITY), ("direction", OFF_LBSP_DIRECTION), ("analytic", OFF_LBSP_ANALYTIC), ("vertex565", OFF_LBSP_VERTEX_565), ("vertex_ao", OFF_LBSP_VERTEX_AO)] {
                let Some(t) = c.tag_ref_of(m + off, b"bitm") else { continue };
                let Some(bm) = c.tag_meta(t) else { continue };
                let Some((nb, bo)) = c.block(bm + crate::h4::bitmaps::OFF_BITM_BITMAPS) else { continue };
                for i in 0..nb {
                    let o = bo + i * crate::h4::bitmaps::BITMAP_ELEM;
                    let el: Vec<String> = (0..48).map(|k| format!("{:02x}", d.u8_at(o + k))).collect();
                    eprintln!("  {label} '{}' elem[{i}] w {} h {} depth {} flags {:#x} type {} fmt {} levels {} total {} | {}",
                        c.tag_name(t).rsplit('\\').next().unwrap_or(""), d.u16_at(o), d.u16_at(o + 2), d.u8_at(o + 4), d.u8_at(o + 5), d.u8_at(o + 6), d.i16_at(o + 8), d.u8_at(o + 16) as u32 + 1, d.u32_at(o + 24), el.join(" "));
                }
                let Some((_, ro)) = c.block(bm + OFF_BITM_RESOURCES) else { continue };
                let Some(e) = c.resource_by_id(d.u32_at(ro)) else { eprintln!("  (resource unresolved)"); continue };
                let dd = c.definition(e);
                let streams: Vec<String> = (0..3).map(|k| format!("[{} B @{:?}]", dd.i32_at(k * 20), e.fixup_at((k * 20 + 12) as u32).map(|a| format!("{a:#x}")))).collect();
                let desc: Vec<String> = (0x3C..0x3C.min(dd.len()).max(0x50.min(dd.len()))).map(|k| format!("{:02x}", dd[k])).collect();
                eprintln!("    def {} B streams {} desc@3C {}", dd.len(), streams.join(" "), desc.join(" "));
                // HMS_H4_DUMP_STREAMS=1: write the primary stream bytes to the scratch dir
                if std::env::var("HMS_H4_DUMP_STREAMS").is_ok() {
                    if let (size, Some(addr)) = (dd.i32_at(0).max(0) as usize, e.fixup_at(12)) {
                        if let Ok(b) = c.stream_bytes(e, addr, size) {
                            let p = scratch().join(format!("{}.bin", c.tag_name(t).rsplit('\\').next().unwrap_or("x")));
                            std::fs::write(&p, &b).unwrap();
                            eprintln!("    wrote {} ({} B)", p.display(), b.len());
                        }
                    }
                }
            }
        }
    }

    /// RE tool: print the scnr structure_bsps block (+0xA4, 336 B) fields the engine's
    /// floating-sun setup reads (sub_18035E814 / sub_18034F2D0 in halo4.dll: count @+0xA4, ptr @+0xA8) and the Lbsp flags.
    #[test]
    #[ignore]
    fn dump_scnr_bsp_lighting_fields() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let Some(c) = open(&map) else { return };
        let d = c.data();
        for scnr in c.find_tags(b"scnr") {
            let Some(sm) = c.tag_meta(scnr) else { continue };
            eprintln!("scnr dwords @0xA0..0xC0: {:?}", (0..8).map(|k| format!("{:#x}", d.u32_at(sm + 0xA0 + 4 * k))).collect::<Vec<_>>());
            let Some((n, o)) = c.block(sm + 0xA4) else { continue };
            for i in 0..n {
                let e = o + i * 336;
                let sbsp = c.tag_ref(e).map(|(cl, t)| format!("{} {}", String::from_utf8_lossy(&cl), c.tag_name(t)));
                let u16_116 = d.u16_at(e + 116);
                let bytes: Vec<u8> = (180..188).map(|k| d.u8_at(e + k)).collect();
                let f184 = d.f32_at(e + 184);
                let frus: Vec<Vec<f32>> = (0..d.u8_at(e + 180) as usize).map(|k| (0..6).map(|j| d.f32_at(e + 188 + k * 24 + j * 4)).collect()).collect();
                eprintln!("scnr bsp[{i}] {sbsp:?}: u16@116 {u16_116} bytes@180 {bytes:?} f32@184 {f184} frustums {frus:?}");
            }
        }
        for lbsp in c.find_tags(b"Lbsp") {
            let Some(m) = c.tag_meta(lbsp) else { continue };
            eprintln!("Lbsp {}: flags16 {:#06x} u16@2 {:#06x} f32@4..: {:?}", c.tag_name(lbsp), d.u16_at(m), d.u16_at(m + 2), (1..9).map(|k| d.f32_at(m + 4 * k)).collect::<Vec<_>>());
        }
    }

    /// RE tool: for the per-VERTEX-lit instances (no UV stream) print the Lbsp +0xC8 /
    /// +0x1B8 record dwords next to the mesh vertex count, to find the vertex offset into the
    /// 1024-wide `*_vertex_565` array (vs_mesh_lightmap_compress_constant.z). Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_pervertex_records() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let Some(c) = open(&map) else { return };
        let d = c.data();
        for sbsp in c.find_tags(b"sbsp") {
            let bsp = load_bsp(&c, sbsp).unwrap();
            let Some(lbsp) = bsp.lbsp_tag else { continue };
            let m = c.tag_meta(lbsp).unwrap();
            if let Some(vt) = c.tag_ref_of(m + OFF_LBSP_VERTEX_565, b"bitm") {
                let info = bitmap_info(&c, vt).unwrap();
                eprintln!("{}: vertex_565 {}x{} depth {} fmt {}", bsp.name, info.w, info.h, info.depth, info.format);
            }
            if let Some(g) = bsp.geometry.as_ref() {
                let mut hist: std::collections::BTreeMap<(i16, i16), (usize, u64)> = Default::default();
                for vb in &g.vbs { let e = hist.entry((vb.kind, vb.stride)).or_default(); e.0 += 1; e.1 += vb.count as u64; }
                eprintln!("  VB (kind, stride) -> (count, total verts): {hist:?}");
            }
            if let Some(vt) = c.tag_ref_of(m + OFF_LBSP_VERTEX_AO, b"bitm") {
                let info = bitmap_info(&c, vt).unwrap();
                eprintln!("  vertex_ao_565 {}x{} depth {} fmt {}", info.w, info.h, info.depth, info.format);
            }
            let (ni, oi) = c.block(m + OFF_LBSP_INSTANCE_INFO).unwrap_or((0, 0));
            let (nl, ol) = c.block(m + OFF_LBSP_INSTANCE_LM).unwrap_or((0, 0));
            let (mut shown, mut sum_pv) = (0usize, 0usize);
            let (mut expect, mut maxend, mut mism, mut lit) = (0usize, 0usize, 0usize, 0usize);
            for (ii, inst) in bsp.instances.iter().enumerate() {
                if inst.mesh < 0 || instance_uv_vb(&c, lbsp, ii).is_some() { continue; }
                let mi = inst.mesh as usize;
                let nverts = decode_mesh(&c, &bsp, mi).ok().flatten().map(|mm| mm.verts.len()).unwrap_or(0);
                let info: Vec<String> = if ii < ni { (0..9).map(|k| format!("{:#x}", d.u32_at(oi + ii * INSTANCE_INFO_ELEM + 4 * k))).collect() } else { vec![] };
                let lm: Vec<String> = if ii < nl { (0..7).map(|k| format!("{:#x}", d.u32_at(ol + ii * INSTANCE_LM_ELEM + 4 * k))).collect() } else { vec![] };
                // HMS_H4_PV_BIG=1 also prints every record whose vertex offset is past 4096
                let big = d.i32_at(oi + ii * INSTANCE_INFO_ELEM + 12) > 4096 && std::env::var("HMS_H4_PV_BIG").is_ok();
                if shown < 24 || big { eprintln!("  inst {ii} mesh {mi} verts {nverts} (running {sum_pv}): +0xC8 {info:?} +0x1B8 {lm:?}"); }
                shown += 1;
                sum_pv += nverts;
                let off = d.i32_at(oi + ii * INSTANCE_INFO_ELEM + 12);
                if off >= 0 && nverts > 0 {
                    if off as usize != expect { mism += 1; if mism <= 5 { eprintln!("  MISMATCH inst {ii}: offset {off} expected {expect}"); } }
                    expect = off as usize + nverts;
                    maxend = maxend.max(off as usize + nverts);
                    lit += 1;
                }
            }
            eprintln!("  {shown} per-vertex-lit instances, {sum_pv} verts total; {lit} with an offset, max end {maxend}, {mism} non-contiguous");
            // (b @+2, mode @+6) -> (instances, max offset+count)
            let mut groups: std::collections::BTreeMap<(i16, i16), (usize, i64)> = Default::default();
            for (ii, inst) in bsp.instances.iter().enumerate() {
                if inst.mesh < 0 || ii >= ni { continue; }
                let e = oi + ii * INSTANCE_INFO_ELEM;
                let nverts = decode_mesh(&c, &bsp, inst.mesh as usize).ok().flatten().map(|mm| mm.verts.len()).unwrap_or(0) as i64;
                let off = d.i32_at(e + 12) as i64;
                let g = groups.entry((d.i16_at(e + 2), d.i16_at(e + 6))).or_default();
                g.0 += 1;
                if off >= 0 { g.1 = g.1.max(off + nverts); }
            }
            let uvlit = bsp.instances.iter().enumerate().filter(|(ii, _)| instance_uv_vb(&c, lbsp, *ii).is_some()).count();
            eprintln!("  groups (b, mode) -> (n, max end): {groups:?}; {uvlit} instances with a UV stream");
            // per-vertex colours vs the atlas: luminance percentiles of both paths
            if let (Ok(atlas), Ok(Some(pv))) = (load_atlas(&c, lbsp), load_pervertex(&c, lbsp)) {
                let mut lums: Vec<f32> = Vec::new();
                let mut vis = 0.0f64;
                for (ii, inst) in bsp.instances.iter().enumerate() {
                    let Some((0, off, _)) = instance_pervertex(&c, lbsp, ii) else { continue };
                    let Ok(Some(mesh)) = decode_mesh(&c, &bsp, inst.mesh.max(0) as usize) else { continue };
                    let mat = bsp.instance_matrix(inst);
                    let normals: Vec<[f32; 3]> = mesh.verts.iter().map(|v| mat.transform_vector3(glam::Vec3::from(v.normal)).normalize_or_zero().to_array()).collect();
                    if let Some(cols) = pervertex_colors(&pv, &atlas, off, &normals) {
                        for c4 in cols { lums.push(0.2126 * c4[0] + 0.7152 * c4[1] + 0.0722 * c4[2]); vis += (c4[3] * c4[3]) as f64; }
                    }
                }
                if !lums.is_empty() {
                    let n = lums.len();
                    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    eprintln!("  per-vertex colours: {n} verts, lum p10/p50/p90 = {:.3}/{:.3}/{:.3}, sun-vis mean {:.3}", lums[n / 10], lums[n / 2], lums[n * 9 / 10], vis / n as f64);
                }
                let tex = compose_atlas_textures(&c, &atlas).unwrap();
                engine_stats(&c, &bsp, &atlas, &tex);
            }
        }
    }

    /// Render helper: print the scnr placement positions by class (bipd = player spawns)
    /// so a ground-level HMS_CAM can be chosen for a map. Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_placement_positions() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let Some(c) = open(&map) else { return };
        let mut by: std::collections::BTreeMap<String, Vec<[f32; 3]>> = Default::default();
        for p in crate::h4::objects::load_placements(&c) {
            by.entry(String::from_utf8_lossy(&p.class).into_owned()).or_default().push(p.pos);
        }
        for (cl, ps) in by {
            let n = ps.len() as f32;
            let c3 = ps.iter().fold([0.0f32; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
            eprintln!("{cl}: {} placements, centroid {c3:?}, first {:?}", ps.len(), &ps[..ps.len().min(6)]);
        }
    }

    /// RE tool: dump the raw page streams of every `bitm` whose name contains
    /// HMS_H4_BITM (default `rasterizer\sharpen_falloff`) with its element fields, for the LUT
    /// decode. Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_lut_bitmaps() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let want = std::env::var("HMS_H4_BITM").unwrap_or_else(|_| "rasterizer\\sharpen_falloff".into());
        let Some(c) = open(&map) else { return };
        let out = scratch();
        for tag in c.find_tags(b"bitm") {
            let name = c.tag_name(tag).to_string();
            if !name.contains(&want) { continue; }
            let leaf = name.rsplit('\\').next().unwrap_or(&name).to_string();
            let Some(info) = bitmap_info(&c, tag) else { continue };
            let Some(m) = c.tag_meta(tag) else { continue };
            let Some((_, ro)) = c.block(m + OFF_BITM_RESOURCES) else { continue };
            let Some(e) = c.resource_by_id(c.data().u32_at(ro)) else { continue };
            let dd = c.definition(e);
            eprintln!("bitm {name}: {}x{} depth {} kind {} format {} levels {} total {}", info.w, info.h, info.depth, info.kind, info.format, info.levels, info.total);
            for k in 0..3 {
                let size = dd.i32_at(k * 20).max(0) as usize;
                if let Some(addr) = e.fixup_at((k * 20 + 12) as u32) {
                    if size == 0 { continue; }
                    match c.stream_bytes(e, addr, size) {
                        Ok(b) => {
                            let p = out.join(format!("{leaf}_stream{k}_nib{}.bin", addr >> 28));
                            std::fs::write(&p, &b).unwrap();
                            eprintln!("  stream {k} nibble {} size {size} -> {}", addr >> 28, p.display());
                        }
                        Err(err) => eprintln!("  stream {k} size {size}: {err}"),
                    }
                }
            }
        }
    }

    struct ZhAtlasCopy { dm_tag: u32, sdm_tag: u32 }

    /// Forge Island's Lbsp +0x3E8 airprobe block (113 `lightprobevolume*` points, z = -62.28 on
    /// the deck level) and the engine blend at the variant's loadout camera: the radiance SH
    /// EXCLUDES the direct sun (E/pi to-sun ~= anti-sun) and lights an up-facing surface ~4-5
    /// (the mean-bake stand-in gives 6.4 flat, with the vista's baked sun in it).
    #[test]
    fn forge_island_airprobes() {
        let Some(c) = open("dlc_forge_island.map") else { return };
        let lbsp = c.find_tags(b"Lbsp").into_iter().find(|&t| c.tag_name(t).ends_with("dlc_forge_island_bsp01")).expect("bsp01 Lbsp");
        let probes = load_airprobes(&c, lbsp);
        assert_eq!(probes.len(), 113);
        let p0 = &probes[0];
        assert!((p0.pos[0] - 237.05).abs() < 0.01 && (p0.pos[1] - 94.74).abs() < 0.01 && (p0.pos[2] + 62.28).abs() < 0.01, "{:?}", p0.pos);
        assert_eq!(p0.volume, 0x3481);
        assert!((p0.sh[0][0] - 12.586).abs() < 0.01, "DC r {}", p0.sh[0][0]);
        assert!(probes.iter().all(|p| p.vis >= 0.0 && p.vis <= 1.0));
        let (sh, vis) = airprobe_sample(&probes, [234.69, -102.01, -66.35]).expect("a probe within 60 wu");
        let p = H4Probe { dir_a: [0.0; 3], w_a: 0.0, col_a: [0.0; 3], dir_b: [0.0; 3], w_b: 0.0, col_b: [0.0; 3], sh };
        let up = probe_sh(&p, [0.0, 0.0, 1.0]);
        let down = probe_sh(&p, [0.0, 0.0, -1.0]);
        let sun = probe_sh(&p, [0.18304859, -0.9450339, 0.270932]);
        let anti = probe_sh(&p, [-0.18304859, 0.9450339, -0.270932]);
        eprintln!("airprobe blend at the loadout camera: up {up:?} down {down:?} to-sun {sun:?} anti-sun {anti:?} vis {vis}");
        assert!(up[1] > 3.5 && up[1] < 6.0, "up {up:?}");
        assert!(down[1] < 1.0, "down {down:?}");
        assert!((sun[1] - anti[1]).abs() < 1.0, "the direct sun is not in the SH: {sun:?} vs {anti:?}");
        // no probe within 60 wu of the far vista -> None (the caller falls back)
        assert!(airprobe_sample(&probes, [-6000.0, -6000.0, 5000.0]).is_none());
        // the texel sampler agrees with the shader decode on a padding texel (black) and stays finite
        let atlas = load_atlas(&c, lbsp).unwrap();
        let tex = compose_atlas_textures(&c, &atlas).unwrap();
        let (da, wa, ca, _db, _wb, _cb, vis) = atlas_texel_lobes(&tex, atlas.k_dom, atlas.k, [0.5, 0.5]);
        assert!(da.iter().all(|v| v.is_finite()) && wa >= 0.0 && wa <= 1.0 && ca.iter().all(|v| *v >= 0.0) && (0.0..=1.0).contains(&vis));
    }

    /// Reach reference dump through the native DLL: DM (raw DDS) + SDM slices 0..2 (BGRA) of the
    /// first BSP of a Reach map (HMS_REACH_MAP, default forge_halo.map) into the scratch dir,
    /// with the same channel statistics, so the H4 encoding can be compared side by side. Needs
    /// libhalomapstudio.so next to the test binary or at HMS_NATIVE_DLL, and the Reach maps folder
    /// (HMS_REACH_MAPS or the Steam default).
    #[test]
    #[ignore]
    fn reach_reference_dump() {
        let dll = std::env::var("HMS_NATIVE_DLL").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("libhalomapstudio.so"))).unwrap_or_default()
        });
        let Ok(nat) = hms_native::NativeDll::load(&dll) else { eprintln!("skip: native dll {} not loadable", dll.display()); return };
        let maps = std::env::var("HMS_REACH_MAPS").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            std::path::PathBuf::from("/mnt/games/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach/maps")
        });
        let name = std::env::var("HMS_REACH_MAP").unwrap_or_else(|_| "forge_halo.map".into());
        let p = maps.join(&name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return; }
        let cache = nat.open_cache(&p.to_string_lossy()).expect("open reach cache");
        let scnr = nat.find_scenario(cache).expect("scnr");
        let sbsps = nat.enumerate_sbsps(cache, scnr, 64);
        let out = scratch();
        for &sbsp in sbsps.iter().take(2) {
            let Some(atlas) = nat.lbsp_atlas(cache, sbsp) else { eprintln!("sbsp {sbsp:#x}: no Lbsp atlas"); continue };
            // copy out of the packed struct before borrowing
            let (dm_tag, sdm_tag, k, has_k) = (atlas.dm_tag, atlas.sdm_tag, atlas.brightness, atlas.has_brightness);
            let atlas = ZhAtlasCopy { dm_tag, sdm_tag };
            eprintln!("reach {} sbsp {sbsp:#x}: dm {:#x} ({}) sdm {:#x} ({}) K {:.4} has_brightness {}",
                name, dm_tag, nat.tag_name(cache, dm_tag).unwrap_or_default(), sdm_tag,
                nat.tag_name(cache, sdm_tag).unwrap_or_default(), k, has_k);
            if let Some(dds) = nat.raw_dds(cache, atlas.dm_tag, 0) {
                std::fs::write(out.join(format!("reach_{sbsp:x}_dm.dds")), &dds).unwrap();
                eprintln!("  dm dds {} B", dds.len());
            }
            if let Some((bgra, w, h)) = nat.decode_bitmap_keep_alpha(cache, atlas.dm_tag, 0, 0) {
                std::fs::write(out.join(format!("reach_{sbsp:x}_dm_{w}x{h}.bgra")), &bgra).unwrap();
                print_stats("reach DM", &bgra);
            }
            for sl in 0..3u32 {
                let r = if sl == 0 { nat.decode_bitmap_keep_alpha(cache, atlas.sdm_tag, 0, 0) } else { nat.decode_bitmap_slice(cache, atlas.sdm_tag, 0, 0, sl) };
                if let Some((bgra, w, h)) = r {
                    std::fs::write(out.join(format!("reach_{sbsp:x}_sdm{sl}_{w}x{h}.bgra")), &bgra).unwrap();
                    print_stats(&format!("reach SDM[{sl}]"), &bgra);
                }
            }
        }
        nat.close_cache(cache);
    }
}
