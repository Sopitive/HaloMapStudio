//! Halo 4 (MCC) `mat ` / `mats` / `mtsb` decode: texture parameter NAMES, blend modes and
//! a material-kind classification for the BSP render path.
//!
//! Layout (probed on wraparound, ca_forge_ravine, m10_crash, m30_cryptum; every offset agrees
//! with the Assembly `Halo4MCC/mat.xml` + `mats.xml` plugins):
//!
//! `mat ` main struct (0x44 B):
//!   +0x00 tag ref -> `mats`;  +0x10 block "material parameters" (ALWAYS EMPTY in the cache: the
//!   parameter-name table was stripped at cache build);  +0x1C block (n=1) "postprocess
//!   definition" (0xA8 B, below);  +0x28/+0x2C/+0x30/+0x34 physics global-material string ids
//!   (`hard_metal_thick`, `brittle_glass`, `energy`, one per layer on layered shaders; ALL ZERO on
//!   fx / sky / non-collidable materials - 8 of 31 on Ravine);
//!   +0x38 f32 sort offset;  +0x3C u8 ALPHA BLEND MODE (enum below);  +0x3D u8 sort layer;
//!   +0x3E u8 flags;  +0x3F u8 render flags;  +0x40 u8 transparent shadow policy.
//!
//! postprocess element (0xA8 B):
//!   @0x00 textures block, 0x18 B each: `bitm` ref @0, u8 address mode @0x10 (0x11 = cube),
//!         u8 filter mode @0x11, u8 frame-index parameter @0x12 (0xFF), u8 SAMPLER INDEX @0x13
//!         (== the element index == the pixel shader's sampler/texture register), i8 min/max mip
//!         @0x14/0x15 (-1), u8 render phase mask @0x16 (5/6/7);
//!   @0x0C texture xforms (16 B: scale u, scale v, offset u, offset v), one per texture;
//!   @0x18 float constants (float4), @0x24 vertex float constants, @0x30 int constants,
//!   @0x3C bool constants (flags32), @0x54 functions block (0x2C B; animated materials),
//!   @0x60 function parameters, @0x6C extern parameters, @0x78 u8 alpha blend mode (equals
//!   the main-struct byte at +0x3C on every `mat ` of both test maps), @0x79 u8 layer blend
//!   mode (1 blended, 2 layered), @0x7A u16 flags (bit 4 = has texture transform functions).
//!
//! `mats` (0x64 B): +0x00 flags32 (bit 1 decal, bit 4 volume fog, bit 5 water, bit 7 hologram,
//!   bit 12 ALPHA CLIP), +0x24 ref -> `mtsb` shader bank, +0x34 vertex-shader entry points
//!   (38 x 12 B, each a block of 52 x {i32 hash, i32 index into mtsb VS table}; shared/deduped
//!   across shaders), +0x40 PIXEL-shader entry points (38 x {i32 hash, i32 index into the mtsb
//!   PS table}; -1 = none), +0x4C "material parameters" (EMPTY in the cache).
//!
//! `mtsb`: +0x00 vertex shaders (128 B records), +0x24 pixel shaders (128 B records); each
//!   record carries two DXBC blobs as tag_data {i32 size @0, u32 ptr @12} at +0x18 and +0x2C.
//!   1199 of 5091 pixel shaders (25 of the 38 entry points of every surface shader) still carry
//!   their `RDEF` reflection chunk, which names the samplers `UserSampler_<parameter>` at bind
//!   point == the `mat` texture slot. That is the ONLY place the texture parameter names survive
//!   (0 conflicts / 0 unmapped slots over all 100 wraparound + Ravine BSP materials). Float
//!   constants are anonymous there (`UserParametersPS.user_parameter_160..191`), so the float
//!   "names" are the cbuffer variable names at 16*i.

use std::collections::HashMap;

use super::bitmaps::bitmap_info;
use super::cache::{ByteRead, H4Cache};

// ---- `mat ` main struct (the full layout is in the module doc) --------------------------------
pub const OFF_MAT_MATS: usize = 0x00;
pub const OFF_MAT_POSTPROCESS: usize = 0x1C;
pub const OFF_MAT_PHYSICS_SID: usize = 0x28;
pub const OFF_MAT_SORT_OFFSET: usize = 0x38;
pub const OFF_MAT_BLEND_MODE: usize = 0x3C;
pub const OFF_MAT_SORT_LAYER: usize = 0x3D;
// ---- postprocess element ---------------------------------------------------------------------
pub const PP_TEXTURES: usize = 0x00;
pub const PP_TEX_XFORMS: usize = 0x0C;
pub const PP_FLOATS: usize = 0x18;
pub const PP_BOOLS: usize = 0x3C;
pub const PP_BLEND_MODE: usize = 0x78;
pub const PP_LAYER_BLEND: usize = 0x79;
pub const PP_FLAGS: usize = 0x7A;
pub const TEX_ELEM: usize = 0x18;
pub const TEX_SAMPLER_INDEX: usize = 0x13;
// ---- `mats` ----------------------------------------------------------------------------------
pub const OFF_MATS_FLAGS: usize = 0x00;
pub const OFF_MATS_MTSB: usize = 0x24;
/// Vertex-shader entry points (read only by the `dump_ps_blobs` RE tool in lightmaps.rs).
#[cfg(test)]
pub const OFF_MATS_VS_ENTRIES: usize = 0x34;
pub const OFF_MATS_PS_ENTRIES: usize = 0x40;
pub const MATS_FLAG_VOLUME_FOG: u32 = 1 << 4;
pub const MATS_FLAG_ALPHA_CLIP: u32 = 1 << 12;
// ---- `mtsb` ----------------------------------------------------------------------------------
#[cfg(test)]
pub const OFF_MTSB_VS: usize = 0x00;
pub const OFF_MTSB_PS: usize = 0x24;
pub const MTSB_SHADER_REC: usize = 0x80;
pub const MTSB_REC_DXBC_A: usize = 0x18;
pub const MTSB_REC_DXBC_B: usize = 0x2C;

/// Halo 4 alpha blend mode (Assembly `Halo4MCC/mat.xml`, same order as Reach's rmt2 enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum H4BlendMode {
    Opaque = 0,
    Additive = 1,
    Multiply = 2,
    AlphaBlend = 3,
    DoubleMultiply = 4,
    PreMultipliedAlpha = 5,
    Maximum = 6,
    MultiplyAdd = 7,
    AddSrcTimesDstAlpha = 8,
    AddSrcTimesSrcAlpha = 9,
    InvAlphaBlend = 10,
    MotionBlurStatic = 11,
    MotionBlurInhibit = 12,
    ApplyShadowIntoShadowMask = 13,
    AlphaBlendConstant = 14,
    OverdrawApply = 15,
    WetScreenEffect = 16,
    Minimum = 17,
    ReverseSubtract = 18,
    ForgeLightmap = 19,
    ForgeLightmapInv = 20,
    ReplaceAllChannels = 21,
    AlphaBlendMax = 22,
    OpaqueAlphaBlend = 23,
    AlphaBlendAdditiveTransparent = 24,
    Unknown = 255,
}

impl H4BlendMode {
    pub fn from_u8(v: u8) -> H4BlendMode {
        use H4BlendMode::*;
        match v {
            0 => Opaque, 1 => Additive, 2 => Multiply, 3 => AlphaBlend, 4 => DoubleMultiply,
            5 => PreMultipliedAlpha, 6 => Maximum, 7 => MultiplyAdd, 8 => AddSrcTimesDstAlpha,
            9 => AddSrcTimesSrcAlpha, 10 => InvAlphaBlend, 11 => MotionBlurStatic,
            12 => MotionBlurInhibit, 13 => ApplyShadowIntoShadowMask, 14 => AlphaBlendConstant,
            15 => OverdrawApply, 16 => WetScreenEffect, 17 => Minimum, 18 => ReverseSubtract,
            19 => ForgeLightmap, 20 => ForgeLightmapInv, 21 => ReplaceAllChannels,
            22 => AlphaBlendMax, 23 => OpaqueAlphaBlend, 24 => AlphaBlendAdditiveTransparent,
            _ => Unknown,
        }
    }
}

/// Render-lane classification of a BSP material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum H4MatKind {
    Opaque,
    /// Cutout: `mats` flag bit 12 ("is alpha clip") / `*_clip` shaders / foliage.
    AlphaTest,
    /// Glass, decals, foam sheets, vertex-alpha fades: src-alpha over.
    AlphaBlend,
    /// Holograms, plasma, boundaries, light shafts, energy: additive glow.
    Additive,
    Multiply,
    /// Not a rendered surface (volume fog / explicit effect shaders / null material).
    Invisible,
}

#[derive(Clone, Debug, Default)]
pub struct H4TexParam {
    /// Shader parameter name from the pixel-shader reflection (`color_map`, `normal_map`,
    /// `diffuseMap`, `layer0_coMap`, ...). `slot_N` when no reflection survived.
    pub name: String,
    pub bitm: Option<usize>,
    /// (scale u, scale v, offset u, offset v) from the postprocess texture xform block.
    pub xform: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct H4Material {
    pub mat_tag: usize,
    pub name: String,
    pub mats_tag: Option<usize>,
    pub mats_name: String,
    pub kind: H4MatKind,
    pub textures: Vec<H4TexParam>,
    /// Float4 constants with the reflected cbuffer variable name (`user_parameter_160`...).
    pub floats: Vec<(String, [f32; 4])>,
    /// 1.0 opaque; a stand-in constant alpha for blend kinds when the shader carries none.
    pub base_alpha: f32,
    /// Raw fields kept for routing / diagnostics.
    pub blend_mode: H4BlendMode,
    pub blend_mode_raw: u8,
    pub sort_layer: u8,
    pub sort_offset: f32,
    pub mats_flags: u32,
    pub pp_flags: u16,
    pub layer_blend: u8,
    /// Postprocess bool constants (flags32 @0x3C; bit 0 = the family's first
    /// `user_parameter_bool_0`, e.g. `normal_detail_map` enabled on srf_blinn*).
    pub bools: u32,
    /// Physics global material names (`hard_metal_thick`, `brittle_glass`, ...), layer 0 first.
    pub physics: Vec<String>,
    /// True when texture names came from a surviving RDEF chunk (else `slot_N` placeholders).
    pub names_from_reflection: bool,
}

impl Default for H4MatKind {
    fn default() -> Self { H4MatKind::Opaque }
}

impl Default for H4BlendMode {
    fn default() -> Self { H4BlendMode::Opaque }
}

impl H4Material {
    fn find_tex(&self, names: &[&str]) -> Option<usize> {
        for n in names {
            if let Some(t) = self.textures.iter().find(|t| t.name.eq_ignore_ascii_case(n)) { return t.bitm; }
        }
        None
    }
    /// Diffuse / albedo bitmap. Falls back to texture parameter 0 (the diffuse on every
    /// wraparound material, the blend map on layered terrain).
    pub fn diffuse(&self) -> Option<usize> {
        self.find_tex(&["color_map", "diffuseMap", "diffuse_map", "base_map", "baseMap", "layer0_coMap", "color1_map"])
            .or_else(|| self.textures.first().and_then(|t| t.bitm))
    }
    /// `*_vertmask` shaders (srf_ca_color_warp_vertmask: the Ravine foam sheets) fade by the
    /// per-vertex blend scalar (vertex bytes 12..16, a 0/1 mask on those meshes - on Ravine the
    /// foam is 33 % zero / 67 % one; cliff / rock meshes carry ~0.5 LAYER weights instead, so
    /// the mask is only honoured for these shaders).
    pub fn uses_vertex_mask(&self) -> bool { self.mats_name.to_ascii_lowercase().contains("vertmask") }
}

/// Classification rule table (documented; blend mode is the primary key, the shader NAME only
/// breaks ties when the blend byte says opaque):
///   1. `mats` "volume fog" flag or an `explicit_shaders\` / `volume_fog` shader   -> Invisible
///   2. blend 1 additive, 6 maximum, 7 multiply-add, 8/9 add-src-times-alpha, 24   -> Additive
///   3. blend 2 multiply, 4 double multiply                                         -> Multiply
///   4. blend 3 alpha, 5 premultiplied, 10 inv alpha, 14 alpha constant, 22 max, 23 -> AlphaBlend
///   5. blend 0 + `mats` alpha-clip flag or name `*_clip` / `foliage`               -> AlphaTest
///   6. blend 0 + name `glass` / `vertalpha` / `falpha` / `hologram` / `halogram`
///      / `color_warp`                                                              -> AlphaBlend
///   7. blend 0 + name `plasma` / `boundary` / `softedge` / `energy` / `lightshafts`
///      / `edge_glow`                                                               -> Additive
///   8. otherwise                                                                    -> Opaque
pub fn classify(mats_name: &str, blend_mode: u8, mats_flags: u32) -> H4MatKind {
    let n = mats_name.to_ascii_lowercase();
    let has = |s: &str| n.contains(s);
    if mats_flags & MATS_FLAG_VOLUME_FOG != 0 || has("explicit_shaders\\") || has("volume_fog") {
        return H4MatKind::Invisible;
    }
    match blend_mode {
        1 | 6 | 7 | 8 | 9 | 24 => return H4MatKind::Additive,
        2 | 4 => return H4MatKind::Multiply,
        3 | 5 | 10 | 14 | 22 | 23 => return H4MatKind::AlphaBlend,
        _ => {}
    }
    if mats_flags & MATS_FLAG_ALPHA_CLIP != 0 || has("_clip") || has("foliage") { return H4MatKind::AlphaTest; }
    if has("glass") || has("vertalpha") || has("falpha") || has("hologram") || has("halogram") || has("color_warp") {
        return H4MatKind::AlphaBlend;
    }
    if has("plasma") || has("boundary") || has("softedge") || has("energy") || has("lightshafts") || has("edge_glow") {
        return H4MatKind::Additive;
    }
    H4MatKind::Opaque
}

// ---- DXBC reflection ---------------------------------------------------------------------

/// Sampler bind point -> parameter name, plus float cbuffer variable names by byte offset.
#[derive(Default)]
struct Reflection {
    samplers: HashMap<u32, String>,
    float_vars: HashMap<u32, String>,
}

fn cstr(b: &[u8], o: usize) -> String {
    let Some(s) = b.get(o..) else { return String::new() };
    let n = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    String::from_utf8_lossy(&s[..n.min(256)]).into_owned()
}

/// Parse the RDEF chunk of a DXBC blob: `UserSampler_<name>` sampler bindings and the
/// `UserParametersPS` cbuffer variables. None when the blob has no reflection chunk.
fn parse_rdef(blob: &[u8]) -> Option<Reflection> {
    if blob.len() < 36 || &blob[0..4] != b"DXBC" { return None; }
    let nchunks = blob.u32_at(28) as usize;
    for k in 0..nchunks.min(32) {
        let co = blob.u32_at(32 + 4 * k) as usize;
        if blob.get(co..co + 8).map(|h| &h[0..4]) != Some(b"RDEF") { continue; }
        let size = blob.u32_at(co + 4) as usize;
        let r = blob.get(co + 8..(co + 8 + size).min(blob.len()))?;
        let (ncb, cbo, nres, reso) = (r.u32_at(0) as usize, r.u32_at(4) as usize, r.u32_at(8) as usize, r.u32_at(12) as usize);
        let rd11 = r.get(28..32) == Some(b"RD11");
        let (res_stride, var_stride) = if rd11 { (r.u32_at(40) as usize, r.u32_at(44) as usize) } else { (32, 24) };
        if res_stride < 28 || var_stride < 12 { return None; }
        let mut out = Reflection::default();
        for i in 0..nres.min(256) {
            let e = reso + i * res_stride;
            let name = cstr(r, r.u32_at(e) as usize);
            let ty = r.u32_at(e + 4);
            let bind = r.u32_at(e + 20);
            if ty == 3 {
                if let Some(p) = name.strip_prefix("UserSampler_") { out.samplers.insert(bind, p.to_string()); }
            }
        }
        for i in 0..ncb.min(64) {
            let e = cbo + i * 24;
            let name = cstr(r, r.u32_at(e) as usize);
            if name != "UserParametersPS" { continue; }
            let (nv, vo) = (r.u32_at(e + 4) as usize, r.u32_at(e + 8) as usize);
            for j in 0..nv.min(256) {
                let v = vo + j * var_stride;
                out.float_vars.insert(r.u32_at(v + 4), cstr(r, r.u32_at(v) as usize));
            }
        }
        return Some(out);
    }
    None
}

/// One of the two DXBC blobs of a `mtsb` shader record (`table` = OFF_MTSB_VS / OFF_MTSB_PS).
fn mtsb_blob<'a>(c: &'a H4Cache, mtsb_meta: usize, table: usize, idx: usize, td: usize) -> Option<&'a [u8]> {
    let (n, p) = c.block(mtsb_meta + table)?;
    if idx >= n { return None; }
    let rec = p + idx * MTSB_SHADER_REC;
    let d = c.data();
    let size = d.i32_at(rec + td);
    let ptr = d.u32_at(rec + td + 12);
    if size <= 0 || ptr == 0 || ptr == 0xCDCD_CDCD { return None; }
    let o = c.meta_off(ptr)?;
    d.get(o..o + size as usize)
}

/// Union of the reflection data over every pixel-shader entry point of a `mats` that still
/// carries an RDEF chunk (they agree with each other: 0 conflicts on both test maps).
fn reflect_mats(c: &H4Cache, mats_tag: usize) -> Reflection {
    let mut out = Reflection::default();
    let Some(mm) = c.tag_meta(mats_tag) else { return out };
    let Some(mtsb) = c.tag_ref_of(mm + OFF_MATS_MTSB, b"mtsb") else { return out };
    let Some(mb) = c.tag_meta(mtsb) else { return out };
    let Some((ne, pe)) = c.block(mm + OFF_MATS_PS_ENTRIES) else { return out };
    let d = c.data();
    for k in 0..ne.min(64) {
        let idx = d.i32_at(pe + k * 8 + 4);
        if idx < 0 { continue; }
        for td in [MTSB_REC_DXBC_B, MTSB_REC_DXBC_A] {
            let Some(blob) = mtsb_blob(c, mb, OFF_MTSB_PS, idx as usize, td) else { continue };
            if let Some(r) = parse_rdef(blob) {
                for (b, n) in r.samplers { out.samplers.entry(b).or_insert(n); }
                for (o, n) in r.float_vars { out.float_vars.entry(o).or_insert(n); }
                break;
            }
        }
    }
    out
}

/// Decode one `mat ` tag (texture params by name, float4 constants, blend mode, kind).
pub fn load_material(c: &H4Cache, mat_tag: usize) -> H4Material {
    let d = c.data();
    let mut m = H4Material { mat_tag, name: c.tag_name(mat_tag).to_string(), base_alpha: 1.0, ..Default::default() };
    let Some(mt) = c.tag_meta(mat_tag) else { m.kind = H4MatKind::Invisible; return m };
    m.mats_tag = c.tag_ref_of(mt + OFF_MAT_MATS, b"mats");
    if let Some(ms) = m.mats_tag {
        m.mats_name = c.tag_name(ms).to_string();
        if let Some(mm) = c.tag_meta(ms) { m.mats_flags = d.u32_at(mm + OFF_MATS_FLAGS); }
    }
    for k in 0..4 {
        let v = d.u32_at(mt + OFF_MAT_PHYSICS_SID + 4 * k);
        if v != 0 { m.physics.push(c.sid(v)); }
    }
    m.sort_offset = d.f32_at(mt + OFF_MAT_SORT_OFFSET);
    m.blend_mode_raw = d.u8_at(mt + OFF_MAT_BLEND_MODE);
    m.sort_layer = d.u8_at(mt + OFF_MAT_SORT_LAYER);
    let refl = m.mats_tag.map(|ms| reflect_mats(c, ms)).unwrap_or_default();
    m.names_from_reflection = !refl.samplers.is_empty();
    if let Some((_, pe)) = c.block(mt + OFF_MAT_POSTPROCESS) {
        // the element's own copy of the blend mode wins if the main-struct byte is unset
        let pp_blend = d.u8_at(pe + PP_BLEND_MODE);
        if m.blend_mode_raw == 0 { m.blend_mode_raw = pp_blend; }
        m.layer_blend = d.u8_at(pe + PP_LAYER_BLEND);
        m.pp_flags = d.u16_at(pe + PP_FLAGS);
        m.bools = d.u32_at(pe + PP_BOOLS);
        let (xn, xp) = c.block(pe + PP_TEX_XFORMS).unwrap_or((0, 0));
        if let Some((tn, tp)) = c.block(pe + PP_TEXTURES) {
            for k in 0..tn.min(64) {
                let e = tp + k * TEX_ELEM;
                let slot = d.u8_at(e + TEX_SAMPLER_INDEX) as u32;
                let name = refl.samplers.get(&slot).cloned().unwrap_or_else(|| format!("slot_{slot}"));
                let xform = if k < xn { let x = xp + k * 16; [d.f32_at(x), d.f32_at(x + 4), d.f32_at(x + 8), d.f32_at(x + 12)] } else { [1.0, 1.0, 0.0, 0.0] };
                m.textures.push(H4TexParam { name, bitm: c.tag_ref_of(e, b"bitm"), xform });
            }
        }
        if let Some((fnum, fp)) = c.block(pe + PP_FLOATS) {
            for k in 0..fnum.min(64) {
                let f = fp + k * 16;
                let name = refl.float_vars.get(&(16 * k as u32)).cloned().unwrap_or_else(|| format!("user_parameter_{}", 160 + k));
                m.floats.push((name, [d.f32_at(f), d.f32_at(f + 4), d.f32_at(f + 8), d.f32_at(f + 12)]));
            }
        }
    }
    m.blend_mode = H4BlendMode::from_u8(m.blend_mode_raw);
    m.kind = classify(&m.mats_name, m.blend_mode_raw, m.mats_flags);
    // Stand-in constant alpha: glass shaders are texture / fresnel driven in the engine; 0.5 keeps
    // them see-through until the fresnel lane exists (a heuristic constant).
    m.base_alpha = match m.kind {
        H4MatKind::AlphaBlend if m.mats_name.to_ascii_lowercase().contains("glass") => 0.5,
        _ => 1.0,
    };
    m
}

/// All materials of one `sbsp` (its +0x160 block; null `mat ` refs become Invisible entries so
/// the index space matches the parts' material indices).
pub fn load_bsp_materials(c: &H4Cache, sbsp_tag: usize) -> Vec<H4Material> {
    let mut out = Vec::new();
    let Some(sb) = c.tag_meta(sbsp_tag) else { return out };
    let Some((n, o)) = c.block(sb + super::geometry::OFF_SBSP_MATERIALS) else { return out };
    for i in 0..n {
        match c.tag_ref_of(o + i * super::geometry::MATERIAL_ELEM, b"mat ") {
            Some(mt) => out.push(load_material(c, mt)),
            None => out.push(H4Material { name: "<null>".into(), kind: H4MatKind::Invisible, base_alpha: 1.0, ..Default::default() }),
        }
    }
    out
}

/// One line per material: index, mat name, mats name, kind, base_alpha, blend, and every texture
/// parameter as `name=bitmap-tag-name (format id)`.
pub fn print_material_diag(c: &H4Cache, mats: &[H4Material]) {
    for (i, m) in mats.iter().enumerate() {
        let short = |s: &str| s.rsplit('\\').next().unwrap_or(s).to_string();
        let tex: Vec<String> = m.textures.iter().map(|t| {
            match t.bitm {
                Some(b) => format!("{}={} ({})", t.name, short(c.tag_name(b)), bitmap_info(c, b).map(|i| i.format).unwrap_or(-1)),
                None => format!("{}=<none>", t.name),
            }
        }).collect();
        eprintln!("  [{i:3}] {} | {} | {:?} a={:.2} blend={:?}({}) clip={} flags={:#x} phys={:?} | {}",
            short(&m.name), short(&m.mats_name), m.kind, m.base_alpha, m.blend_mode, m.blend_mode_raw,
            m.mats_flags & MATS_FLAG_ALPHA_CLIP != 0, m.pp_flags, m.physics, tex.join(" "));
    }
}

#[cfg(test)]
mod tests {
    use super::super::cache::maps_dir;
    use super::super::geometry::load_bsp;
    use super::*;
    use std::collections::BTreeMap;

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    const DIFFUSE_ISH: &[&str] = &["color_map", "diffuseMap", "diffuse_map", "base_map", "baseMap", "blend_map", "selfillum_map", "color1_map"];

    /// Every BSP material of wraparound + Ravine loads with reflected, ASCII texture names;
    /// texture 0 is diffuse-ish; the main-struct blend byte equals the postprocess copy.
    #[test]
    fn bsp_materials_load_and_name() {
        let mut hist: BTreeMap<H4MatKind, usize> = BTreeMap::new();
        let mut total = 0;
        let mut with_physics = 0;
        for map in ["wraparound.map", "ca_forge_ravine.map"] {
            let Some(c) = open(map) else { return };
            let d = c.data();
            for sb in c.find_tags(b"sbsp") {
                let mats = load_bsp_materials(&c, sb);
                assert!(!mats.is_empty(), "{map}: no materials");
                eprintln!("== {} {}: {} materials", map, c.tag_name(sb), mats.len());
                print_material_diag(&c, &mats);
                for m in &mats {
                    *hist.entry(m.kind).or_default() += 1;
                    if m.mat_tag == 0 && m.name == "<null>" { continue; }
                    total += 1;
                    assert!(m.names_from_reflection, "{}: no RDEF texture names", m.name);
                    assert!(!m.textures.is_empty(), "{}: no textures", m.name);
                    for t in &m.textures {
                        assert!(!t.name.is_empty() && t.name.is_ascii() && !t.name.starts_with("slot_"), "{}: bad name {:?}", m.name, t.name);
                    }
                    assert!(DIFFUSE_ISH.contains(&m.textures[0].name.as_str()), "{}: tex0 = {}", m.name, m.textures[0].name);
                    assert!(m.diffuse().is_some(), "{}: no diffuse", m.name);
                    for (n, _) in &m.floats { assert!(n.starts_with("user_parameter_"), "{}: float name {n}", m.name); }
                    // main-struct blend byte @0x3C == postprocess element @0x78
                    let mt = c.tag_meta(m.mat_tag).unwrap();
                    let (_, pe) = c.block(mt + OFF_MAT_POSTPROCESS).unwrap();
                    assert_eq!(d.u8_at(mt + OFF_MAT_BLEND_MODE), d.u8_at(pe + PP_BLEND_MODE), "{}: blend byte mismatch", m.name);
                    // physics sids are OPTIONAL: fx / sky / non-collidable materials carry
                    // all-zero words at +0x28 (Ravine: waterfall_thin/thick, water_foam2, wet_01a,
                    // skybox_clouds4/5, island, island_large = 8 of 31; wraparound 0 of 69).
                    for p in &m.physics { assert!(!p.is_empty() && p.is_ascii(), "{}: physics sid {:?}", m.name, m.physics); }
                    if !m.physics.is_empty() { with_physics += 1; }
                }
            }
        }
        eprintln!("== kind histogram over {total} named materials: {hist:?}; {with_physics} with physics sids");
        assert!(total >= 100, "expected >= 100 materials, got {total}");
        assert!(with_physics * 100 >= total * 85, "physics sid coverage {with_physics}/{total}");
    }

    /// The Ravine "white slab with black speckle" under the tower base = the two
    /// `ca_ravine_terrain_water_foam` sheets (srf_ca_color_warp_vertmask, blend mode 3) drawn
    /// opaque. They must classify AlphaBlend with the DXT5 foam as `diffuseMap`.
    #[test]
    fn ravine_slab_is_alpha_blend_foam() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let sb = c.find_tags(b"sbsp").into_iter().find(|&t| c.tag_name(t).ends_with("ca_forge_ravine_bsp")).expect("playable bsp");
        let mats = load_bsp_materials(&c, sb);
        let (idx, foam) = mats.iter().enumerate().find(|(_, m)| m.name.ends_with("ca_ravine_terrain_water_foam")).expect("foam material");
        eprintln!("slab material [{idx}]:");
        print_material_diag(&c, std::slice::from_ref(foam));
        assert_eq!(foam.kind, H4MatKind::AlphaBlend);
        assert_eq!(foam.blend_mode, H4BlendMode::AlphaBlend);
        assert!(foam.mats_name.ends_with("srf_ca_color_warp_vertmask"));
        let names: Vec<&str> = foam.textures.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["diffuseMap", "alphaMap", "selfIllumMap", "uvOffsetMap"]);
        let diff = foam.diffuse().expect("foam diffuse");
        assert!(c.tag_name(diff).ends_with("ca_ravine_wave_foam"));
        assert_eq!(bitmap_info(&c, diff).unwrap().format, 16, "foam diffuse is DXT5 (alpha in the texture)");
        // where it is: every instance of a mesh that uses this material, world bounds
        let bsp = load_bsp(&c, sb).expect("load_bsp");
        let mut hits = 0;
        for inst in &bsp.instances {
            let Some(mesh) = bsp.meshes.get(inst.mesh as usize) else { continue };
            if mesh.parts.iter().any(|p| p.material as usize == idx) {
                hits += 1;
                eprintln!("  foam instance mesh {} world min {:?} max {:?}", inst.mesh, inst.world_min, inst.world_max);
            }
        }
        assert!(hits >= 1, "foam material has no instances");
        // the glass and the water plane route as expected too
        let water = mats.iter().find(|m| m.name.ends_with("ca_ravine_terrain_water2")).unwrap();
        assert_eq!(water.kind, H4MatKind::Opaque);
        let glass = mats.iter().find(|m| m.name.ends_with("ca_forge_ravine_glass_pattern_01")).unwrap();
        assert_eq!(glass.kind, H4MatKind::AlphaBlend);
    }

    /// Campaign map (24 BSPs, 1054 materials): every named material decodes, >= 95 % of the
    /// non-Invisible ones get reflected texture names, and the kind histogram is printed.
    #[test]
    fn campaign_m10_materials() {
        let Some(c) = open("m10_crash.map") else { return };
        let mut hist: BTreeMap<H4MatKind, usize> = BTreeMap::new();
        let mut tex0: BTreeMap<String, usize> = BTreeMap::new();
        let (mut named, mut reflected, mut unreflected) = (0usize, 0usize, Vec::new());
        for sb in c.find_tags(b"sbsp") {
            for m in load_bsp_materials(&c, sb) {
                *hist.entry(m.kind).or_default() += 1;
                if m.name == "<null>" || m.kind == H4MatKind::Invisible { continue; }
                named += 1;
                if let Some(t) = m.textures.first() { *tex0.entry(t.name.clone()).or_default() += 1; }
                if m.names_from_reflection { reflected += 1; } else { unreflected.push(m.mats_name.clone()); }
                assert!(m.textures.iter().all(|t| t.name.is_ascii()), "{}", m.name);
            }
        }
        unreflected.sort();
        unreflected.dedup();
        eprintln!("== m10_crash kinds {hist:?}\n   tex0 names {tex0:?}\n   reflected {reflected}/{named}; without RDEF: {unreflected:?}");
        assert!(named > 900, "named {named}");
        assert!(reflected * 100 >= named * 95, "reflection coverage {reflected}/{named}");
    }

    /// Name-rule table sanity (no cache needed).
    #[test]
    fn classify_rules() {
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_blinn", 0, 0), H4MatKind::Opaque);
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_blinn_clip", 0, MATS_FLAG_ALPHA_CLIP), H4MatKind::AlphaTest);
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_glass", 3, 0), H4MatKind::AlphaBlend);
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_glass_simple", 0, 0), H4MatKind::AlphaBlend);
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_ca_hologram", 1, 0), H4MatKind::Additive);
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_constant_lightshafts", 9, 0), H4MatKind::Additive);
        assert_eq!(classify("shaders\\material_shaders\\materials\\srf_ca_skybox_gradient", 2, 0), H4MatKind::Multiply);
        assert_eq!(classify("shaders\\material_shaders\\explicit_shaders\\effects\\volume_fog", 1, MATS_FLAG_VOLUME_FOG), H4MatKind::Invisible);
    }
}
