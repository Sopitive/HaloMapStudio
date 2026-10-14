//! Halo 3 BSP albedo composite (`albedo_fx.hlsl` ported offline). #h3-albedo
//!
//! THE BUG THIS FIXES. The Halo 3 BSP path bound `base_map` alone as the surface colour. That is
//! only ever the whole albedo for the *one* shader option that has no detail layer; every other
//! `shader` albedo option multiplies one to three DETAIL maps into the base and scales the
//! product by `DETAIL_MULTIPLIER = 4.59479`. For the options whose base is a *blend/overlay mask*
//! rather than a colour (`detail_blend`, `two_detail`, `two_detail_overlay`,
//! `three_detail_blend`) the base carries almost no colour at all - zanzibar's courtyard is
//! `zan_cementlot`, albedo option 1, whose `base_map` is `zan_cementlot_overlay` (a near-white
//! plate whose ALPHA selects cement vs dirt) with the real colour in `detail_map`
//! (`zan_cementlot_dif`) and `detail_map2` (`zan_ground_dirt_dif`). Binding the base alone drew
//! the whole lot as blown-out featureless white - "basically no ground at all".
//!
//! The option ordinal -> law table is pinned by `materials::tests::albedo_option_sampler_sets`,
//! which sweeps every installed Halo 3 cache and asserts each option names a FIXED sampler set
//! (option 1 always names `detail_map2`, 5 `detail_map3`, 6 `detail_map_overlay`, 8
//! `color_mask_map`, and option 2 `constant_color` never names a `base_map` at all).
//!
//! The laws are `camden-smallwood/protomorph`'s ports of `albedo_fx.hlsl`
//! (`assets/halo_shaders/albedo_fx_*.wgsl`), reproduced per-texel here:
//!
//!   default (0)             base.rgb x (detail.rgb x DM) x albedo_color   [albedo_fx_default.wgsl:24-28]
//!   detail_blend (1)        DM x base.rgb x ((1-base.a) d1 + base.a d2)   [albedo_fx_detail_blend.wgsl:25-33]
//!   three_detail_blend (5)  b1=sat(2 base.a), b2=sat(2 base.a - 1);
//!                           f=(1-b1) d1 + b1 d2; s=(1-b2) f + b2 d3;
//!                           DM x base.rgb x s.rgb                         [albedo_fx_three_detail_blend.wgsl:35-45]
//!   two_detail_overlay (6)  db=(1-base.a) d1 + base.a d2;
//!                           base.rgb x DM^2 x db.rgb x overlay.rgb        [albedo_fx_two_detail_overlay.wgsl:30-38]
//!   two_detail (7)          base.rgb x DM^2 x d1.rgb x d2.rgb             [albedo_fx_two_detail.wgsl:26-32]
//!   color_mask (8)          base.rgb x (detail.rgb x DM) x tint(mask)     [albedo_fx_color_mask.wgsl:34-47]
//!   two_detail_black_point  as two_detail, alpha through apply_black_point [albedo_fx_two_detail_black_point.wgsl:24-47]
//!
//! Why offline and not in the shader: the shared mesh shader carries ONE detail sampler and no
//! base-alpha blend, and it is also Reach's and Halo 4's - adding three samplers to it would put
//! every other game's BSP through a new code path. Compositing into the uploaded texture keeps
//! the change inside `h3/`. The cost is resolution: a detail map that tiles 8x over the base is
//! resolved at base resolution, so its fine grain is softened (the same trade the Reach detail
//! bake makes). The COLOUR, which is what was missing, is exact.
//!
//! Space: the mesh shader samples a NON-sRGB view and does `pow(rgb, 2.2)` itself (mesh.rs
//! `mesh_shade`), so this module decodes bytes with `^2.2`, runs the law in linear, saturates
//! (the engine's `albedo = saturate(...)`) and re-encodes with `^(1/2.2)` - exactly inverting
//! what the shader will do.

use std::collections::HashMap;

use super::bitmaps::{bitmap_pixels_avail, BitmapInfo};
use super::cache::H3Cache;
use super::materials::H3Material;

/// `DETAIL_MULTIPLIER` (`albedo_fx.hlsl`): the constant every detail layer is scaled by. Note
/// 4.59479 == 2^2.2, so in the shader's gamma-2.2 byte space one detail layer is exactly x2.
pub const DETAIL_MULTIPLIER: f32 = 4.59479;

/// Largest composite we build (a Halo 3 BSP base map is 512 or 1024; this only clamps).
const MAX_DIM: u32 = 1024;

/// One decoded mip-0 image, RGBA8 straight from the cache's bytes (no colour conversion).
pub struct Img {
    pub w: u32,
    pub h: u32,
    /// RGBA8, `w * h * 4` bytes.
    pub px: Vec<u8>,
}

impl Img {
    /// Nearest texel at a WRAPPED uv (the engine's samplers wrap; detail maps tile).
    #[inline]
    fn at(&self, u: f32, v: f32) -> [f32; 4] {
        let wrap = |t: f32, n: u32| {
            let t = t - t.floor();
            ((t * n as f32) as u32).min(n - 1) as usize
        };
        let i = (wrap(v, self.h) * self.w as usize + wrap(u, self.w)) * 4;
        let p = &self.px[i..i + 4];
        [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0]
    }
}

/// Decode a `bitm`'s mip 0 to RGBA8. BC1 / BC2 / BC3 and the two 32-bit uncompressed layouts -
/// every format a Halo 3 BSP albedo / detail map actually uses (the sweep in
/// `materials::tests` sees only 0x0B, 0x0E and 0x10 on colour parameters).
pub fn decode_rgba(c: &H3Cache, tag: usize) -> Option<Img> {
    let (info, bytes) = bitmap_pixels_avail(c, tag).ok()?;
    decode_rgba_bytes(&info, &bytes)
}

fn decode_rgba_bytes(info: &BitmapInfo, bytes: &[u8]) -> Option<Img> {
    if info.kind != 0 || info.w == 0 || info.h == 0 { return None; }
    let (w, h) = (info.w as u32, info.h as u32);
    let n = (w as usize) * (h as usize) * 4;
    let mut px = vec![255u8; n];
    match info.format {
        0x0E => decode_bc(bytes, w, h, &mut px, 8, false),
        0x0F => decode_bc(bytes, w, h, &mut px, 16, false),
        0x10 => decode_bc(bytes, w, h, &mut px, 16, true),
        // A8R8G8B8 / X8R8G8B8: stored BGRA byte order
        0x0A | 0x0B => {
            if bytes.len() < n { return None; }
            for (o, p) in px.chunks_exact_mut(4).enumerate() {
                let s = &bytes[o * 4..o * 4 + 4];
                p[0] = s[2]; p[1] = s[1]; p[2] = s[0];
                p[3] = if info.format == 0x0A { 255 } else { s[3] };
            }
        }
        _ => return None,
    }
    Some(Img { w, h, px })
}

/// BC1 (`block = 8`) / BC2 / BC3 (`block = 16`) colour blocks into RGBA8. `bc3_alpha` decodes the
/// BC4 alpha block of a BC3; BC2's 4-bit alpha nibbles are expanded when `block == 16 &&
/// !bc3_alpha`.
fn decode_bc(data: &[u8], w: u32, h: u32, out: &mut [u8], block: usize, bc3_alpha: bool) {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let c565 = |c: u16| -> [f32; 3] {
        [((c >> 11) & 31) as f32 / 31.0, ((c >> 5) & 63) as f32 / 63.0, (c & 31) as f32 / 31.0]
    };
    let bc2 = block == 16 && !bc3_alpha;
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * block;
            if o + block > data.len() { return; }
            let co = if block == 16 { o + 8 } else { o };
            let c0 = u16::from_le_bytes([data[co], data[co + 1]]);
            let c1 = u16::from_le_bytes([data[co + 2], data[co + 3]]);
            let (p0, p1) = (c565(c0), c565(c1));
            let mut pal = [p0, p1, [0.0; 3], [0.0; 3]];
            // BC2/BC3 colour blocks ALWAYS use the 4-colour (c0 > c1) interpolation, whatever the
            // endpoint order; only BC1 switches to the 3-colour + punch-through mode.
            let four = block == 16 || c0 > c1;
            for k in 0..3 {
                if four {
                    pal[2][k] = (2.0 * p0[k] + p1[k]) / 3.0;
                    pal[3][k] = (p0[k] + 2.0 * p1[k]) / 3.0;
                } else {
                    pal[2][k] = 0.5 * (p0[k] + p1[k]);
                    pal[3][k] = 0.0;
                }
            }
            let bits = u32::from_le_bytes([data[co + 4], data[co + 5], data[co + 6], data[co + 7]]);
            // alpha palette (BC3 / BC4 block)
            let mut apal = [255u8; 8];
            if bc3_alpha {
                let (a0, a1) = (data[o], data[o + 1]);
                apal[0] = a0; apal[1] = a1;
                if a0 > a1 {
                    for k in 1..7 { apal[k + 1] = (((7 - k) as u32 * a0 as u32 + k as u32 * a1 as u32) / 7) as u8; }
                } else {
                    for k in 1..5 { apal[k + 1] = (((5 - k) as u32 * a0 as u32 + k as u32 * a1 as u32) / 5) as u8; }
                    apal[6] = 0; apal[7] = 255;
                }
            }
            let abits = if bc3_alpha {
                let mut v = 0u64;
                for k in 0..6 { v |= (data[o + 2 + k] as u64) << (8 * k); }
                v
            } else { 0 };
            for py in 0..4usize {
                for px4 in 0..4usize {
                    let (x, y) = (bx * 4 + px4, by * 4 + py);
                    if x >= w as usize || y >= h as usize { continue; }
                    let k = py * 4 + px4;
                    let idx = ((bits >> (2 * k)) & 3) as usize;
                    let p = (y * w as usize + x) * 4;
                    out[p] = (pal[idx][0] * 255.0 + 0.5) as u8;
                    out[p + 1] = (pal[idx][1] * 255.0 + 0.5) as u8;
                    out[p + 2] = (pal[idx][2] * 255.0 + 0.5) as u8;
                    out[p + 3] = if bc3_alpha {
                        apal[((abits >> (3 * k)) & 7) as usize]
                    } else if bc2 {
                        let nib = data[o + k / 2];
                        let v = if k % 2 == 0 { nib & 0x0F } else { nib >> 4 };
                        v * 17
                    } else if block == 8 && c0 <= c1 && idx == 3 { 0 } else { 255 };
                }
            }
        }
    }
}

#[inline]
fn to_linear(v: f32) -> f32 { v.powf(2.2) }
#[inline]
fn to_gamma(v: f32) -> f32 { v.clamp(0.0, 1.0).powf(1.0 / 2.2) }

/// `apply_black_point` (`utilities_fx.hlsl:123`, via
/// `albedo_fx_two_detail_black_point.wgsl:24-28`): soft remap of `alpha` around a pivot.
pub fn apply_black_point(black_point: f32, alpha: f32) -> f32 {
    let mid = (black_point + 1.0) * 0.5;
    let d = mid - black_point;
    if d.abs() < 1e-6 { return alpha.clamp(0.0, 1.0); }
    (mid * ((alpha - black_point) / d).clamp(0.0, 1.0) + (alpha - mid).clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Does this albedo option read more than `base_map` (i.e. is a composite needed at all)?
pub fn needs_composite(m: &H3Material) -> bool {
    if m.base.is_none() { return false; }
    // every option composites at least a detail_map; option 2 (constant_color) has no base
    m.param("detail_map").is_some()
}

/// Build the composited albedo for one BSP render method: BGRA8 bytes + dimensions, ready for
/// `upload_texture_bgra` (which also builds the mip chain). `None` when a required layer will
/// not decode - the caller then falls back to the plain compressed `base_map` upload.
pub fn composite(c: &H3Cache, m: &H3Material, cache: &mut HashMap<usize, Option<std::sync::Arc<Img>>>) -> Option<(Vec<u8>, u32, u32)> {
    let base_tag = m.base?;
    let mut get = |tag: usize| -> Option<std::sync::Arc<Img>> {
        cache.entry(tag).or_insert_with(|| decode_rgba(c, tag).map(std::sync::Arc::new)).clone()
    };
    let base = get(base_tag)?;
    let layer = |name: &str, g: &mut dyn FnMut(usize) -> Option<std::sync::Arc<Img>>| -> Option<(std::sync::Arc<Img>, [f32; 4])> {
        let t = m.param(name)?;
        let img = g(t)?;
        Some((img, m.xform(name)))
    };
    let d1 = layer("detail_map", &mut get);
    let d2 = layer("detail_map2", &mut get);
    let d3 = layer("detail_map3", &mut get);
    let dov = layer("detail_map_overlay", &mut get);
    let cmask = layer("color_mask_map", &mut get);
    d1.as_ref()?; // nothing to composite without a first detail layer

    let opt = m.albedo_opt();
    let base_xf = m.xform("base_map");
    // The composite resolves at the BASE map's grid: the base xform is the UV the mesh shader
    // already applies to this texture, so the output stays a drop-in replacement for base_map
    // and every detail is re-projected onto it.
    let (ow, oh) = (base.w.min(MAX_DIM), base.h.min(MAX_DIM));
    if ow == 0 || oh == 0 { return None; }

    let tint = m.float("albedo_color").unwrap_or([1.0, 1.0, 1.0, 1.0]);
    let tint = {
        let ok = |v: f32| v.is_finite() && (0.0..=8.0).contains(&v);
        if ok(tint[0]) && ok(tint[1]) && ok(tint[2]) && tint[0] + tint[1] + tint[2] > 1e-3 { tint } else { [1.0, 1.0, 1.0, 1.0] }
    };
    let tint2 = m.float("albedo_color2").unwrap_or([1.0; 4]);
    let tint3 = m.float("albedo_color3").unwrap_or([1.0; 4]);
    let neutral = m.float("neutral_gray").unwrap_or([1.0; 4]);

    let dm = DETAIL_MULTIPLIER;
    let dm2 = dm * dm;
    let mut out = vec![0u8; (ow as usize) * (oh as usize) * 4];
    for y in 0..oh {
        // uv of this texel in the BASE map's own space, then back to the mesh uv the detail
        // xforms are authored against: base_uv = uv * base_xf.xy + base_xf.zw.
        let bv = (y as f32 + 0.5) / oh as f32;
        for x in 0..ow {
            let bu = (x as f32 + 0.5) / ow as f32;
            let sx = if base_xf[0].abs() > 1e-6 { base_xf[0] } else { 1.0 };
            let sy = if base_xf[1].abs() > 1e-6 { base_xf[1] } else { 1.0 };
            let (mu, mv) = ((bu - base_xf[2]) / sx, (bv - base_xf[3]) / sy);
            let smp = |l: &Option<(std::sync::Arc<Img>, [f32; 4])>| -> [f32; 4] {
                match l {
                    Some((img, xf)) => {
                        let (sx, sy) = (if xf[0].abs() > 1e-6 { xf[0] } else { 1.0 }, if xf[1].abs() > 1e-6 { xf[1] } else { 1.0 });
                        let t = img.at(mu * sx + xf[2], mv * sy + xf[3]);
                        [to_linear(t[0]), to_linear(t[1]), to_linear(t[2]), t[3]]
                    }
                    None => [1.0, 1.0, 1.0, 1.0],
                }
            };
            let bi = ((y as usize) * ow as usize + x as usize) * 4;
            let bp = &base.px[bi..bi + 4];
            let b = [to_linear(bp[0] as f32 / 255.0), to_linear(bp[1] as f32 / 255.0), to_linear(bp[2] as f32 / 255.0), bp[3] as f32 / 255.0];
            let s1 = smp(&d1);
            let s2 = smp(&d2);
            let s3 = smp(&d3);
            let so = smp(&dov);

            let (mut rgb, mut a) = match opt {
                // detail_blend
                1 => {
                    let mix = |i: usize| (1.0 - b[3]) * s1[i] + b[3] * s2[i];
                    ([dm * b[0] * mix(0), dm * b[1] * mix(1), dm * b[2] * mix(2)], mix(3))
                }
                // three_detail_blend
                5 => {
                    let b1 = (2.0 * b[3]).clamp(0.0, 1.0);
                    let b2 = (2.0 * b[3] - 1.0).clamp(0.0, 1.0);
                    let s = |i: usize| {
                        let f = (1.0 - b1) * s1[i] + b1 * s2[i];
                        (1.0 - b2) * f + b2 * s3[i]
                    };
                    ([dm * b[0] * s(0), dm * b[1] * s(1), dm * b[2] * s(2)], s(3))
                }
                // two_detail_overlay
                6 => {
                    let db = |i: usize| (1.0 - b[3]) * s1[i] + b[3] * s2[i];
                    ([b[0] * dm2 * db(0) * so[0], b[1] * dm2 * db(1) * so[1], b[2] * dm2 * db(2) * so[2]], db(3) * so[3])
                }
                // two_detail / two_detail_black_point
                7 | 9 => {
                    let rgb = [b[0] * dm2 * s1[0] * s2[0], b[1] * dm2 * s1[1] * s2[1], b[2] * dm2 * s1[2] * s2[2]];
                    let a = if opt == 9 { apply_black_point(b[3], s1[3] * s2[3]) } else { b[3] * s1[3] * s2[3] };
                    (rgb, a)
                }
                // color_mask
                8 => {
                    let cm = match &cmask { Some((img, xf)) => img.at(mu * xf[0] + xf[2], mv * xf[1] + xf[3]), None => [0.0; 4] };
                    let stage = |w: f32, t: [f32; 4], i: usize| {
                        let nv = if i < 3 { if neutral[i].abs() > 1e-4 { neutral[i] } else { 1.0 } } else { 1.0 };
                        (1.0 - w) + w * (t[i] / nv)
                    };
                    let tn = |i: usize| stage(cm[0], tint, i) * stage(cm[1], tint2, i) * stage(cm[2], tint3, i);
                    ([b[0] * s1[0] * dm * tn(0), b[1] * s1[1] * dm * tn(1), b[2] * s1[2] * dm * tn(2)], b[3] * s1[3] * tn(3))
                }
                // default (0) and every option whose extra inputs are per-object (change colour /
                // chameleon): base x detail x DM x albedo_color
                _ => ([b[0] * s1[0] * dm * tint[0], b[1] * s1[1] * dm * tint[1], b[2] * s1[2] * dm * tint[2]], b[3] * s1[3] * tint[3]),
            };
            // the engine saturates the composed albedo (entry_points.hlsl_include)
            for v in rgb.iter_mut() { *v = v.clamp(0.0, 1.0); }
            a = a.clamp(0.0, 1.0);
            // BGRA for upload_texture_bgra
            out[bi] = (to_gamma(rgb[2]) * 255.0 + 0.5) as u8;
            out[bi + 1] = (to_gamma(rgb[1]) * 255.0 + 0.5) as u8;
            out[bi + 2] = (to_gamma(rgb[0]) * 255.0 + 0.5) as u8;
            out[bi + 3] = (a * 255.0 + 0.5) as u8;
        }
    }
    Some((out, ow, oh))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `DETAIL_MULTIPLIER` is 2^2.2 exactly, which is why one detail layer is a x2 in the
    /// shader's gamma-2.2 byte space (the identity this module's encode/decode relies on).
    #[test]
    fn detail_multiplier_is_two_in_gamma_space() {
        assert!((DETAIL_MULTIPLIER - 2f32.powf(2.2)).abs() < 1e-4, "{DETAIL_MULTIPLIER}");
        assert!((DETAIL_MULTIPLIER.powf(1.0 / 2.2) - 2.0).abs() < 1e-4);
    }

    /// `apply_black_point` (utilities_fx.hlsl:123) at its three fixed points.
    #[test]
    fn black_point_fixed_points() {
        assert!((apply_black_point(0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((apply_black_point(0.0, 1.0) - 1.0).abs() < 1e-6);
        // at alpha == the pivot the remap is 0; at the mid-point it is the mid-point
        assert!((apply_black_point(0.5, 0.5) - 0.0).abs() < 1e-6);
        assert!((apply_black_point(0.5, 0.75) - 0.75).abs() < 1e-6);
    }

    /// A `default` composite of a mid-grey base with the engine's neutral detail (sRGB 128,
    /// whose linear value x4.59479 is ~1) must return ~the base: that is the calibration the
    /// `DETAIL_MULTIPLIER` is authored around.
    #[test]
    fn neutral_detail_is_identity() {
        let d = to_linear(128.0 / 255.0) * DETAIL_MULTIPLIER;
        assert!((d - 1.0).abs() < 0.06, "neutral detail x DM = {d}");
    }
}
