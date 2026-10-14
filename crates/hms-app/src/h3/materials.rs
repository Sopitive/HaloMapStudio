//! Halo 3 (MCC) render-method base-texture resolve. #h3
//!
//! Halo 3 BSP materials reference a render-method tag DIRECTLY (rmsh/rmtr/rmd/rmw/rmcs/rmfl/...,
//! no `mat ` indirection like Halo 4). Every render-method group shares the classic Gen3
//! `render_method` base struct (Halo3MCC/*.xml, verified identical across rmsh/rmtr/rmd/rmw/rmcs/
//! rmfl/rmhg on guardian):
//!   +0x00 Definition tagref (rmdf);  +0x10 Options block (0x2);
//!   +0x1C **Parameters** block (0x3C): +0x00 name stringid, +0x04 type enum32 (0 Bitmap, 1 Color,
//!         2 Real, 3 Int, 4 Bool, 5 ARGB), +0x08 `bitm` tagref, +0x18 real, ... — a Bitmap
//!         parameter names its texture (`base_map`, `bump_map`, `detail_map`, ...) here;
//!   +0x28 Postprocess block (0x8C): +0x00 rmt2 (render_method_template) ref, +0x10 Textures block
//!         (0x1C: +0x00 `bitm` ref) = the COMPILED bitmap references in SAMPLER order.
//!
//! In shipped caches the render_method's own Parameters block is EMPTY (stripped at cache build,
//! like Halo 4), so the textures come from the postprocess block. Their NAMES come from the rmt2
//! shader template's Textures block (+0x6C, 0x4 stride, one Parameter Name stringid per texture, in
//! the SAME order) - this is the H3 analogue of the Halo 4 pixel-shader reflection. Matching gives
//! the true `base_map` for every shader family (a plain rmsh names texture 0 `base_map`; terrain
//! (rmtr) names texture 0 the blend/control map and its layer albedos `base_map_m_0`...).
//!
//! Each postprocess texture element also names (int8 @0x16) the postprocess Float Constant
//! (+0x1C, vec4) holding its transform_texcoord [tile.xy, offset.xy]; -1 = untransformed.
//!
//! The diffuse/albedo bitmap is the `base_map` parameter (or the closest base/albedo/diffuse
//! layer, else the first non-auxiliary texture). Read-only; a null / unresolved material yields
//! base=None and the caller binds its mid-grey placeholder.

use super::cache::H3Cache;
use crate::h4::cache::ByteRead;

pub const OFF_RM_PARAMETERS: usize = 0x1C;
pub const OFF_RM_POSTPROCESS: usize = 0x28;
pub const PARAM_ELEM: usize = 0x3C;
pub const PARAM_TYPE_BITMAP: u32 = 0;
pub const PARAM_BITMAP_REF: usize = 0x08;
pub const PP_SHADER_TEMPLATE: usize = 0x00; // rmt2 tagref
pub const PP_TEXTURES: usize = 0x10;
pub const PP_TEX_ELEM: usize = 0x1C;
/// Postprocess texture element: int8 index into the postprocess Float Constants (-1 = none) of
/// the sampler's transform_texcoord constant [scale.x, scale.y, offset.x, offset.y].
pub const PP_TEX_XFORM_INDEX: usize = 0x16;
pub const PP_FLOAT_CONSTANTS: usize = 0x1C;
pub const PP_FLOAT_ELEM: usize = 0x10;
pub const OFF_RMT2_TEXTURES: usize = 0x6C; // stringid per texture, 0x4 stride, sampler order
pub const OFF_RMT2_FLOATS: usize = 0x48;   // stringid per float constant, 0x4 stride (postprocess order)
/// `render_method_postprocess` +0x5C: the ANIMATED PARAMETER (Functions) block that drives the
/// scrolling sky layers. Element 0x24 = `{ u32 type; sid input_name; sid range_name; f32
/// time_period; tag_data function }` - the classic `render_method_animated_parameter_block` with
/// the postprocess `type` word in front. #h3-sky-anim
pub const PP_FUNCTIONS: usize = 0x5C;
pub const PP_FUNC_ELEM: usize = 0x24;
pub const PP_FUNC_PERIOD: usize = 0x0C;
pub const PP_FUNC_DATA: usize = 0x10;   // tag_data {size, flags, stream_off, address, definition}
/// `type`: the texture-coordinate component this function drives. Pinned on the shipped sky
/// materials (see `sky_animation_parameters`): zanzibar's sky `water` authors 5 and 6 at the SAME
/// 40 s period - an ocean scrolling diagonally - which only reads as translation X and Y.
pub const ANIM_TRANSLATION_X: u32 = 5;
pub const ANIM_TRANSLATION_Y: u32 = 6;

#[derive(Clone, Debug, Default)]
pub struct H3Material {
    pub render_tag: usize,
    pub name: String,
    /// The diffuse/albedo `bitm` tag, or None (null shader / no readable bitmap parameter).
    pub base: Option<usize>,
    /// Every bitmap parameter (name, `bitm` tag), in sampler order.
    pub params: Vec<(String, usize)>,
    /// Per `params` entry: the sampler's texcoord transform [tile.x, tile.y, off.x, off.y]
    /// (the postprocess float constant its element names; [1,1,0,0] when none).
    pub xforms: Vec<[f32; 4]>,
    /// The compiled float constants (name from the rmt2 Float Constants block, value from the
    /// postprocess block): `albedo_color`, `self_illum_color`, `*_xform`, ...
    pub floats: Vec<(String, [f32; 4])>,
    /// True when `base` came from the postprocess textures block, not the named parameters.
    pub from_postprocess: bool,
    /// The rmt2 shader template's option digits (`shaders\shader_templates\_a_b_c_...`), in
    /// the render-method definition's option order; empty when unresolved.
    pub template_opts: Vec<u8>,
    /// The render method's animated parameters (`PP_FUNCTIONS`), in authored order. #h3-sky-anim
    pub anim: Vec<H3AnimParam>,
    /// The render-method class (rmsh / rmtr / rmhg / rmd / rmfl / ...), `????` when unknown.
    pub class: [u8; 4],
}

/// Blend mode of a Halo 3 render method, from its template's `blend_mode` option (the classic
/// `shader` enum: 0 opaque, 1 additive, 2 multiply, 3 alpha_blend, 4 double_multiply,
/// 5 pre_multiplied_alpha).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H3Blend { Opaque, Additive, Multiply, AlphaBlend, DoubleMultiply, PremulAlpha }

impl H3Material {
    fn opt(&self, i: usize) -> u8 { self.template_opts.get(i).copied().unwrap_or(0) }
    /// The template option index of each family's `blend_mode` (Halo 3 render-method
    /// definitions: `shader`/`custom` albedo, bump_mapping, alpha_test, specular_mask,
    /// material_model, environment_mapping, self_illumination, **blend_mode**(7), parallax, misc,
    /// distortion, soft_fade; `halogram` albedo, self_illumination, **blend_mode**(2), ...;
    /// `decal` albedo, **blend_mode**(1), ...). Terrain / foliage / water are opaque families.
    pub fn blend(&self) -> H3Blend {
        let raw = match &self.class {
            b"rmsh" | b"rmcs" | b"rmct" => self.opt(7),
            b"rmhg" => self.opt(2),
            b"rmd " => self.opt(1),
            _ => 0,
        };
        match raw { 1 => H3Blend::Additive, 2 => H3Blend::Multiply, 3 => H3Blend::AlphaBlend, 4 => H3Blend::DoubleMultiply, 5 => H3Blend::PremulAlpha, _ => H3Blend::Opaque }
    }
    /// Does the template alpha-test (`shader`/`custom` option 2, `foliage` option 1 = 1 simple)?
    /// Cross-checked with the compiled `alpha_test_map` sampler the template then names.
    pub fn alpha_test(&self) -> bool {
        let opt_on = match &self.class { b"rmsh" | b"rmcs" | b"rmct" => self.opt(2) != 0, b"rmfl" => self.opt(1) != 0, _ => false };
        opt_on || self.params.iter().any(|(n, _)| n.eq_ignore_ascii_case("alpha_test_map"))
    }
    /// The `shader` albedo option (0 default, 1 detail_blend, 2 constant_color, 5
    /// three_detail_blend, 7 two_detail, ...) - the sky lane's albedo switch uses the same numbering.
    pub fn albedo_opt(&self) -> u8 {
        match &self.class { b"rmsh" | b"rmcs" | b"rmct" | b"rmhg" | b"rmd " | b"rmfl" => self.opt(0), _ => 0 }
    }
    /// A named bitmap parameter (`detail_map`, `self_illum_map`, ...).
    pub fn param(&self, name: &str) -> Option<usize> {
        self.params.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, b)| *b)
    }
    /// A named float constant (`albedo_color`, ...).
    pub fn float(&self, name: &str) -> Option<[f32; 4]> {
        self.floats.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| *v)
    }
    /// The texcoord transform of a named bitmap parameter ([1,1,0,0] when unnamed / untransformed).
    pub fn xform(&self, name: &str) -> [f32; 4] {
        self.params.iter().position(|(n, _)| n.eq_ignore_ascii_case(name)).and_then(|i| self.xforms.get(i).copied()).unwrap_or([1.0, 1.0, 0.0, 0.0])
    }
}

/// The `shader` material-model option (digit 4): what `calc_material_<model>_ps` the engine links
/// for this render method. 0 diffuse_only, 1 cook_torrance, 2 two_lobe_phong, 3 foliage, 4 none,
/// 5 glass, 6 organism, 7 single_lobe_phong, 8 car_paint, 9 hair, 10 cook_torrance_pbr_maps.
pub const MM_DIFFUSE_ONLY: u8 = 0;
pub const MM_COOK_TORRANCE: u8 = 1;
pub const MM_NONE: u8 = 4;
pub const MM_FOLIAGE: u8 = 3;

/// The engine's cook-torrance / environment constants of one Halo 3 render method - the
/// `cook_torrance_fx.hlsl` PARAMs the shipped shader reads (all present as named postprocess
/// float constants, materials.rs `floats`). This is what a dynamic OBJECT's material lane
/// carries to the shader (#h3-objlight); a material that authors none leaves the defaults, which
/// make the specular lobe vanish (`specular_coefficient` 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H3Shading {
    /// `material_model` option digit (see MM_*).
    pub model: u8,
    /// `specular_mask` option digit: 0 none, 1 from_diffuse, 2 mult_diffuse, 3 from_texture.
    pub spec_mask: u8,
    /// `environment_mapping` option digit: 0 none, 1 per_pixel, 2 dynamic, 3 from_flat_texture...
    pub env_mode: u8,
    pub diffuse_coefficient: f32,
    pub specular_coefficient: f32,
    pub roughness: f32,
    pub albedo_blend: f32,
    pub specular_tint: [f32; 3],
    pub fresnel_color: [f32; 3],
    pub analytical_specular: f32,
    pub area_specular: f32,
    pub env_specular: f32,
    pub env_tint: [f32; 3],
    pub env_roughness_scale: f32,
    /// `albedo_color` (rgb x a): the constant tint every `shader` albedo option multiplies in.
    pub albedo_color: [f32; 4],
}

impl Default for H3Shading {
    fn default() -> Self {
        H3Shading { model: MM_DIFFUSE_ONLY, spec_mask: 0, env_mode: 0, diffuse_coefficient: 1.0, specular_coefficient: 0.0,
            roughness: 0.4, albedo_blend: 1.0, specular_tint: [1.0; 3], fresnel_color: [0.5; 3], analytical_specular: 0.0,
            area_specular: 0.0, env_specular: 0.0, env_tint: [1.0; 3], env_roughness_scale: 1.0, albedo_color: [1.0; 4] }
    }
}

impl H3Material {
    /// The `shader`-family template option digits this lane reads (material_model = 4,
    /// specular_mask = 3, environment_mapping = 5); non-`shader` families keep 0.
    fn shader_opt(&self, i: usize) -> u8 {
        match &self.class { b"rmsh" | b"rmcs" | b"rmct" => self.opt(i), _ => 0 }
    }

    /// This material's cook-torrance / environment constants. `two_lobe_phong` (model 2) authors
    /// `normal_specular_tint` / `glancing_specular_tint` instead of `specular_tint` /
    /// `fresnel_color` (banshee_metal), and no `roughness` - its lobe is driven by
    /// `glancing_specular_power`; those are mapped onto the same slots so one shader lane serves
    /// both (roughness from the power: m = sqrt(2/(p+2)), the standard Beckmann<->Phong bridge).
    pub fn shading(&self) -> H3Shading {
        let f1 = |n: &str| self.float(n).map(|v| v[0]);
        let f3 = |n: &str| self.float(n).map(|v| [v[0], v[1], v[2]]);
        let mut s = H3Shading { model: self.shader_opt(4), spec_mask: self.shader_opt(3), env_mode: self.shader_opt(5), ..Default::default() };
        if let Some(v) = f1("diffuse_coefficient") { s.diffuse_coefficient = v; }
        if let Some(v) = f1("specular_coefficient") { s.specular_coefficient = v; }
        if let Some(v) = f1("roughness") { s.roughness = v; }
        if let Some(v) = f1("albedo_blend") { s.albedo_blend = v; }
        if let Some(v) = f3("specular_tint") { s.specular_tint = v; }
        if let Some(v) = f3("fresnel_color") { s.fresnel_color = v; }
        if let Some(v) = f1("analytical_specular_contribution") { s.analytical_specular = v; }
        if let Some(v) = f1("area_specular_contribution") { s.area_specular = v; }
        if let Some(v) = f1("environment_map_specular_contribution") { s.env_specular = v; }
        if let Some(v) = f3("env_tint_color") { s.env_tint = v; }
        if let Some(v) = f1("env_roughness_scale") { s.env_roughness_scale = v; }
        if let Some(v) = self.float("albedo_color") { s.albedo_color = v; }
        // two_lobe_phong / single_lobe_phong authoring
        if let Some(v) = f3("normal_specular_tint") { s.specular_tint = v; }
        if let Some(v) = f3("glancing_specular_tint") { s.fresnel_color = v; }
        if self.float("roughness").is_none() {
            if let Some(p) = f1("glancing_specular_power").filter(|p| *p > 0.0) {
                s.roughness = (2.0 / (p + 2.0)).sqrt().clamp(0.02, 1.0);
            }
        }
        // `albedo_specular_tint_blend` is two_lobe_phong's albedo_blend
        if self.float("albedo_blend").is_none() {
            if let Some(v) = f1("albedo_specular_tint_blend") { s.albedo_blend = v; }
        }
        s.roughness = s.roughness.clamp(0.02, 1.0);
        s.albedo_blend = s.albedo_blend.clamp(0.0, 1.0);
        s
    }

    /// The `specular_mask_texture` sampler (specular_mask option 3), if any.
    pub fn spec_mask_texture(&self) -> Option<usize> {
        self.param("specular_mask_texture").or_else(|| self.param("specular_mask_map"))
    }

    /// The `environment_map` cube sampler (environment_mapping per_pixel / from_flat_texture).
    pub fn env_map(&self) -> Option<usize> { self.param("environment_map") }
}

/// One `render_method` animated parameter: a texture-coordinate component driven by a mapping
/// function over a wall-clock period. #h3-sky-anim
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H3AnimParam {
    /// `ANIM_TRANSLATION_X` / `_Y` (the only kinds the shipped sky materials author).
    pub kind: u32,
    /// `time_period` (seconds): how long the function's input takes to traverse its range.
    pub period: f32,
    /// The mapping function's type byte (0 = constant, 3 = transition, ...).
    pub fn_type: u8,
    /// The constant function's output (`+-1` on every shipped sky layer = one full uv wrap).
    pub value: f32,
}

impl H3AnimParam {
    /// The translation rate in uv/second: the function's output traversed once per `time_period`.
    /// A `clouds_horizon` layer authors value -1 over 500 s = one tile every 500 s westward; the
    /// Halo 4 / H2A sky turned out to be the same law at a fixed 600 s ([[h2a-fog-law-and-sky-scroll]]).
    pub fn rate(&self) -> f32 {
        if !(self.period.abs() > 1e-3) || !self.value.is_finite() { return 0.0 }
        self.value / self.period
    }
}

/// Per-LAYER `[rate_x, rate_y]` (uv/s) for base_map, detail_map, detail_map2, detail_map3.
///
/// The functions are a flat list with no register index, so they are consumed in authored order:
/// a layer takes at most one X and one Y, and a repeated component starts the next layer. That
/// rule reproduces every shipped sky material exactly - zanzibar `water` 5(40 s) 6(40 s) 6(60 s)
/// = a diagonally scrolling ocean plus a second layer drifting in Y; riverworld `clouds_patchy`
/// 5(400) 6(500) 5(700) 6(600) = two layers; `clouds_fore_frontlit` 5(167) 5(35) = two layers
/// each drifting in X only.
pub fn layer_scroll_rates(anim: &[H3AnimParam]) -> [[f32; 2]; 4] {
    let mut out = [[0.0f32; 2]; 4];
    let (mut layer, mut have_x, mut have_y) = (0usize, false, false);
    for a in anim {
        let (slot, seen) = match a.kind {
            ANIM_TRANSLATION_X => (0usize, have_x),
            ANIM_TRANSLATION_Y => (1, have_y),
            _ => continue,
        };
        if seen { layer += 1; have_x = false; have_y = false; }
        if layer >= out.len() { break }
        out[layer][slot] = a.rate();
        if slot == 0 { have_x = true } else { have_y = true }
    }
    out
}

/// The option digits of a template tag name (`..\_0_2_0_1_0_0_0_0_0_0_0` -> [0,2,0,1,...]).
pub fn template_options(name: &str) -> Vec<u8> {
    let tail = name.rsplit('\\').next().unwrap_or(name);
    if !tail.starts_with('_') { return Vec::new(); }
    tail[1..].split('_').map(|d| d.parse::<u8>().unwrap_or(0)).collect()
}

/// The diffuse-ish texture parameter: exact `base_map` first, then a base/albedo/diffuse layer
/// (terrain `base_map_m_0`, glass `foam_texture`... never match; unnamed → `tex_N`), else the first
/// texture that is NOT a blend/control/detail/bump map, else texture 0.
fn pick_base(params: &[(String, usize)]) -> Option<usize> {
    for p in ["base_map", "albedo_map", "color_map", "diffuse_map"] {
        if let Some((_, b)) = params.iter().find(|(n, _)| n.eq_ignore_ascii_case(p)) { return Some(*b); }
    }
    for key in ["base_map", "albedo", "diffuse", "base"] {
        if let Some((_, b)) = params.iter().find(|(n, _)| n.to_ascii_lowercase().contains(key)) { return Some(*b); }
    }
    // Nothing base-named (e.g. a self-illum-only or blend-only shader): the first texture that is
    // not obviously a mask / control / detail layer. A material whose only textures are auxiliary
    // (riverworld's `ground_lake_deep`: one DXN bump) has NO diffuse - binding the bump as the
    // albedo drew the whole lake bed as a flat blown-white plane.
    let is_aux = |n: &str| { let n = n.to_ascii_lowercase();
        ["blend", "control", "detail", "bump", "normal", "height", "mask", "alpha", "noise", "warp"].iter().any(|k| n.contains(k)) };
    params.iter().find(|(n, _)| !is_aux(n)).map(|(_, b)| *b)
}

/// Decode one render-method tag: its named textures (from the rmt2 template) and the diffuse bitmap.
pub fn load_material(c: &H3Cache, render_tag: usize) -> H3Material {
    let mut m = H3Material { render_tag, name: c.tag_name(render_tag).to_string(), class: c.tag_class(render_tag).unwrap_or(*b"????"), ..Default::default() };
    let Some(mt) = c.tag_meta(render_tag) else { return m };
    let d = c.data();
    // Path A: the render_method's own Parameters block, if the cache kept it (named bitmaps).
    if let Some((n, o)) = c.block(mt + OFF_RM_PARAMETERS) {
        for i in 0..n.min(4096) {
            let e = o + i * PARAM_ELEM;
            if d.u32_at(e + 4) != PARAM_TYPE_BITMAP { continue; }
            if let Some(b) = c.tag_ref_of(e + PARAM_BITMAP_REF, b"bitm") {
                m.params.push((c.sid(d.u32_at(e)), b));
            }
        }
    }
    // Path B (the shipped case): the compiled postprocess textures, named by the rmt2 template's
    // Textures block (same sampler order). Element 0 of the postprocess block.
    if m.params.is_empty() {
        if let Some((_, po)) = c.block(mt + OFF_RM_POSTPROCESS) {
            // texture parameter names from the rmt2 template (in sampler order)
            let rmt2 = c.tag_ref_of(po + PP_SHADER_TEMPLATE, b"rmt2");
            m.template_opts = rmt2.map(|t| template_options(c.tag_name(t))).unwrap_or_default();
            let names: Vec<String> = rmt2
                .and_then(|rt| c.tag_meta(rt))
                .and_then(|rm| c.block(rm + OFF_RMT2_TEXTURES))
                .map(|(tn, to)| (0..tn).map(|i| c.sid(d.u32_at(to + i * 4))).collect())
                .unwrap_or_default();
            let consts = c.block(po + PP_FLOAT_CONSTANTS);
            // the float constants, named by the template's Float Constants block (same order)
            if let (Some((nc, co)), Some(rt)) = (consts, rmt2) {
                let fnames: Vec<String> = c.tag_meta(rt).and_then(|rm| c.block(rm + OFF_RMT2_FLOATS))
                    .map(|(fnn, fo)| (0..fnn).map(|i| c.sid(d.u32_at(fo + i * 4))).collect()).unwrap_or_default();
                for i in 0..nc.min(64) {
                    let o = co + i * PP_FLOAT_ELEM;
                    let name = fnames.get(i).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| format!("float_{i}"));
                    m.floats.push((name, [d.f32_at(o), d.f32_at(o + 4), d.f32_at(o + 8), d.f32_at(o + 12)]));
                }
            }
            if let Some((tn, to)) = c.block(po + PP_TEXTURES) {
                for i in 0..tn.min(64) {
                    let e = to + i * PP_TEX_ELEM;
                    if let Some(b) = c.tag_ref_of(e, b"bitm") {
                        let name = names.get(i).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| format!("tex_{i}"));
                        m.params.push((name, b));
                        let xi = d.u8_at(e + PP_TEX_XFORM_INDEX) as i8;
                        let xf = match consts {
                            Some((nc, co)) if xi >= 0 && (xi as usize) < nc => {
                                let o = co + xi as usize * PP_FLOAT_ELEM;
                                let v = [d.f32_at(o), d.f32_at(o + 4), d.f32_at(o + 8), d.f32_at(o + 12)];
                                if v.iter().all(|x| x.is_finite()) && v[0].abs() > 1e-6 && v[1].abs() > 1e-6 { v } else { [1.0, 1.0, 0.0, 0.0] }
                            }
                            _ => [1.0, 1.0, 0.0, 0.0],
                        };
                        m.xforms.push(xf);
                    }
                }
            }
            // the animated parameters (scrolling sky layers)
            if let Some((fn_n, fo)) = c.block(po + PP_FUNCTIONS) {
                for i in 0..fn_n.min(32) {
                    let e = fo + i * PP_FUNC_ELEM;
                    let kind = d.u32_at(e);
                    let period = d.f32_at(e + PP_FUNC_PERIOD);
                    let (mut fn_type, mut value) = (0u8, 0.0f32);
                    let size = d.u32_at(e + PP_FUNC_DATA) as usize;
                    if (12..=4096).contains(&size) {
                        if let Some(o) = c.meta_off(d.u32_at(e + PP_FUNC_DATA + 0xC)).filter(|o| o + 12 <= d.len()) {
                            fn_type = d.u8_at(o);
                            value = d.f32_at(o + 8);
                        }
                    }
                    m.anim.push(H3AnimParam { kind, period, fn_type, value });
                }
            }
            m.from_postprocess = true;
        }
    }
    m.base = pick_base(&m.params);
    m
}

#[cfg(test)]
mod tests {
    use super::super::cache::maps_dir;
    use super::super::geometry::load_bsp;
    use super::*;

    fn open(name: &str) -> Option<H3Cache> {
        let p = maps_dir()?.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H3Cache::open(&p).expect("open"))
    }

    /// guardian: every BSP material's render tag decodes to at least one named bitmap parameter,
    /// and the vast majority resolve a diffuse bitmap that is a readable `bitm`.
    #[test]
    fn guardian_materials_resolve_diffuse() {
        let Some(c) = open("guardian.map") else { return };
        let sbsp = c.find_tags(b"sbsp")[0];
        let bsp = load_bsp(&c, sbsp).unwrap();
        let short = |s: &str| s.rsplit('\\').next().unwrap_or(s).to_string();
        let (mut resolved, mut with_bitmap, mut total) = (0usize, 0usize, 0usize);
        for bm in &bsp.materials {
            let Some(rt) = bm.render_tag else { continue };
            total += 1;
            let m = load_material(&c, rt);
            if !m.params.is_empty() { resolved += 1; }
            if let Some(b) = m.base {
                assert_eq!(c.tag_class(b), Some(*b"bitm"), "{}: base is not a bitm", short(&m.name));
                with_bitmap += 1;
            }
        }
        eprintln!("guardian materials: {total} render tags, {resolved} with bitmap params, {with_bitmap} with a diffuse");
        assert!(total > 10);
        assert!(with_bitmap * 100 / total >= 80, "diffuse resolved {with_bitmap}/{total}");
    }

    /// The rmt2 template digits: on every guardian + construct BSP / sky `rmsh` material the
    /// alpha_test option (digit 2) is set exactly when the compiled template names an
    /// `alpha_test_map` sampler - the cross-check that pins the `shader` option order the
    /// blend-mode read (digit 7) relies on.
    #[test]
    fn template_options_match_samplers() {
        let (mut checked, mut agree) = (0usize, 0usize);
        for map in ["guardian.map", "construct.map"] {
            let Some(c) = open(map) else { continue };
            let mut tags: Vec<usize> = Vec::new();
            for sbsp in c.find_tags(b"sbsp") { tags.extend(load_bsp(&c, sbsp).unwrap().materials.iter().filter_map(|m| m.render_tag)); }
            // + the sky model's materials
            if let Some(scnr) = c.find_tags(b"scnr").first().copied() {
                if let Some((_, so)) = c.tag_meta(scnr).and_then(|sm| c.block(sm + super::super::objects::OFF_SCNR_SKIES)) {
                    if let Some(mode) = c.tag_ref_of(so, b"scen").and_then(|s| super::super::objects::model_of(&c, s)) {
                        let model = super::super::objects::load_model(&c, mode).unwrap();
                        tags.extend(model.materials.iter().flatten().copied());
                        for &t in model.materials.iter().flatten() {
                            let m = load_material(&c, t);
                            eprintln!("{map} sky mat {} ({}) opts {:?} blend {:?} albedo {} alpha_test {} tex {:?}", m.name.rsplit('\\').next().unwrap_or(""), String::from_utf8_lossy(&m.class), m.template_opts, m.blend(), m.albedo_opt(), m.alpha_test(),
                                m.params.iter().map(|(n, b)| format!("{n}={}", c.tag_name(*b).rsplit('\\').next().unwrap_or(""))).collect::<Vec<_>>());
                        }
                    }
                }
            }
            tags.sort();
            tags.dedup();
            for t in tags {
                let m = load_material(&c, t);
                if &m.class != b"rmsh" || m.template_opts.len() < 8 { continue; }
                let sampler = m.params.iter().any(|(n, _)| n.eq_ignore_ascii_case("alpha_test_map"));
                checked += 1;
                if (m.template_opts[2] != 0) == sampler { agree += 1; } else { eprintln!("  MISMATCH {} opts {:?} sampler {sampler}", m.name, m.template_opts); }
                if m.blend() != H3Blend::Opaque { eprintln!("  {map} blend {:?}: {} opts {:?}", m.blend(), m.name.rsplit('\\').next().unwrap_or(""), m.template_opts); }
            }
        }
        if checked == 0 { eprintln!("skip: no maps"); return; }
        eprintln!("template alpha_test digit vs sampler: {agree}/{checked} agree");
        assert_eq!(agree, checked);
    }

    /// SWEEP (pins the `shader` albedo option enum): over every installed Halo 3 cache,
    /// histogram the rmt2 template's albedo digit against the texture parameters the template
    /// names. Each option must name a FIXED sampler set - that correspondence is what the
    /// albedo composite in `albedo_layers` keys off.
    #[test]
    fn albedo_option_sampler_sets() {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<(String, u8), BTreeMap<String, usize>> = BTreeMap::new();
        for p in super::super::cache::installed_caches() {
            let Ok(c) = H3Cache::open(&p) else { continue };
            let mut tags: Vec<usize> = Vec::new();
            for sbsp in c.find_tags(b"sbsp") {
                let Ok(b) = load_bsp(&c, sbsp) else { continue };
                tags.extend(b.materials.iter().filter_map(|m| m.render_tag));
            }
            tags.sort(); tags.dedup();
            for t in tags {
                let m = load_material(&c, t);
                if m.template_opts.is_empty() { continue }
                let cls = String::from_utf8_lossy(&m.class).to_string();
                let mut names: Vec<&str> = m.params.iter().map(|(n, _)| n.as_str())
                    .filter(|n| !n.contains("bump") && !n.contains("_m_")).collect();
                names.sort(); names.dedup();
                *hist.entry((cls, m.albedo_opt())).or_default().entry(names.join("+")).or_default() += 1;
            }
        }
        if hist.is_empty() { eprintln!("skip: no halo3 maps"); return }
        for ((cls, opt), sets) in &hist {
            let n: usize = sets.values().sum();
            eprintln!("{cls} albedo={opt}: {n} materials over {} distinct sampler sets, e.g. {:?}",
                sets.len(), sets.iter().max_by_key(|(_, n)| **n).map(|(k, n)| format!("{k} x{n}")));
        }
        // the options the BSP composite implements, with the sampler set each one must name
        // every option that names a base_map also names the detail sampler its law multiplies in
        for (opt, must) in [(1u8, "detail_map2"), (5, "detail_map3"), (7, "detail_map2"), (6, "detail_map_overlay"), (8, "color_mask_map"), (3, "change_color_map")] {
            if let Some(sets) = hist.get(&("rmsh".to_string(), opt)) {
                // a handful of materials bind only a base_map (the rest of the samplers compiled
                // out as the template default); every set that binds more names the option's own
                assert!(sets.iter().all(|(k, n)| k.contains(must) || k == "base_map" || *n < 5),
                    "rmsh albedo={opt} must name {must}, got {sets:?}");
            }
        }
        // option 0 (`default`) is base x detail: a base_map always comes with a detail_map
        if let Some(sets) = hist.get(&("rmsh".to_string(), 0)) {
            for (k, n) in sets {
                assert!(!k.contains("base_map") || k.contains("detail_map") || *n < 5, "rmsh albedo=0 '{k}' x{n} has a base but no detail");
            }
        }
        // `constant_color` (2) never names a base_map at all - it is the vertex-colour-only option
        if let Some(sets) = hist.get(&("rmsh".to_string(), 2)) {
            assert!(sets.keys().all(|k| !k.contains("base_map")), "rmsh albedo=2 named a base_map: {sets:?}");
        }
    }

    /// PROBE (print-only): dump every tagblock-looking slot of a sky material's
    /// `render_method` postprocess element, so the animated-parameter (function) block that
    /// drives the scrolling cloud layers can be located. #h3-sky-anim
    #[test]
    fn probe_render_method_postprocess_blocks() {
        if std::env::var("HMS_H3_PROBE_RM").is_err() { eprintln!("skip (set HMS_H3_PROBE_RM=1)"); return }
        for map in ["zanzibar.map", "riverworld.map", "guardian.map"] {
            let Some(c) = open(map) else { continue };
            let d = c.data();
            let Some(scnr) = c.find_tags(b"scnr").first().copied() else { continue };
            let Some((_, so)) = c.tag_meta(scnr).and_then(|sm| c.block(sm + super::super::objects::OFF_SCNR_SKIES)) else { continue };
            let Some(mode) = c.tag_ref_of(so, b"scen").and_then(|s| super::super::objects::model_of(&c, s)) else { continue };
            let model = super::super::objects::load_model(&c, mode).unwrap();
            for &t in model.materials.iter().flatten().take(6) {
                let m = load_material(&c, t);
                let Some(mt) = c.tag_meta(t) else { continue };
                eprintln!("{map} {} ({}) meta@{mt:#x}", m.name.rsplit('\\').next().unwrap_or(""), String::from_utf8_lossy(&m.class));
                // the render_method body: every 4-aligned (count, ptr) that resolves
                for off in (0..0xC0).step_by(4) {
                    if let Some((n, o)) = c.block(mt + off) {
                        if n > 4096 { continue }
                        let head: Vec<u8> = d.get(o..o + 48.min(d.len() - o)).unwrap_or(&[]).to_vec();
                        eprintln!("   body +{off:#04x}: {n} elems @ {o:#x}  {:02x?}", &head[..head.len().min(32)]);
                    }
                }
                let Some((_, po)) = c.block(mt + OFF_RM_POSTPROCESS) else { continue };
                eprintln!("   postprocess elem @ {po:#x}");
                {
                    // the template's float-constant names, in postprocess register order
                    let rmt2 = c.tag_ref_of(po + PP_SHADER_TEMPLATE, b"rmt2");
                    let fnames: Vec<String> = rmt2.and_then(|rt| c.tag_meta(rt)).and_then(|rm| c.block(rm + OFF_RMT2_FLOATS))
                        .map(|(fnn, fo2)| (0..fnn).map(|i| c.sid(d.u32_at(fo2 + i * 4))).collect()).unwrap_or_default();
                    eprintln!("     rmt2 floats: {:?}", fnames.iter().enumerate().map(|(i, n)| format!("{i}={n}")).collect::<Vec<_>>());
                    if let Some((cn, co)) = c.block(po + PP_FLOAT_CONSTANTS) {
                        let vals: Vec<String> = (0..cn).map(|i| { let o = co + i * PP_FLOAT_ELEM; format!("{i}:[{:.3} {:.3} {:.3} {:.3}]", d.f32_at(o), d.f32_at(o+4), d.f32_at(o+8), d.f32_at(o+12)) }).collect();
                        eprintln!("     pp floats: {:?}", vals);
                    }
                }
                if let Some((fn_n, fo)) = c.block(po + 0x5C) {
                    eprintln!("     FUNCTIONS {fn_n} elems @ {fo:#x}");
                    for k in 0..fn_n {
                        let b = fo + k * 0x24;
                        let blob = d.u32_at(b + 0x1C);
                        let sz = d.u32_at(b + 0x10) as usize;
                        let val = c.meta_off(blob).filter(|o| o + 12 <= d.len()).map(|o| (d.u8_at(o), d.u8_at(o + 1), d.f32_at(o + 8)));
                        eprintln!("       FN{k}: hdr [{} {} {}] period {:.2} size {sz} blob {:?}",
                            d.u32_at(b), d.u32_at(b + 4), d.u32_at(b + 8), d.f32_at(b + 0x0C), val);
                    }
                    for k in 0..fn_n.min(6) {
                        // probe every 4-byte slot of a 0x40 window as (u32, f32, sid)
                        let base = fo + k * 0x18;
                        let row: Vec<String> = (0..16).map(|j| {
                            let v = d.u32_at(base + j * 4);
                            let f = d.f32_at(base + j * 4);
                            let sid = c.sid(v);
                            format!("[{j}] {v:#010x} f={f:.4}{}", if sid.is_empty() { String::new() } else { format!(" sid='{sid}'") })
                        }).collect();
                        eprintln!("       fn{k} @ {base:#x}\n         {}", row.join("\n         "));
                        // tag_data shape {size, flags, stream_off, address, definition}: follow it
                        for probe_off in [0x10usize, 0x14, 0x18] {
                            let size = d.u32_at(base + probe_off) as usize;
                            if !(4..=512).contains(&size) { continue }
                            for ptr_off in [probe_off + 4, probe_off + 8, probe_off + 12] {
                                let ptr = d.u32_at(base + ptr_off);
                                if ptr == 0 || ptr == 0xCDCDCDCD { continue }
                                if let Some(o2) = c.meta_off(ptr) {
                                    if o2 + size <= d.len() {
                                        eprintln!("         -> size@{probe_off:#x}={size} ptr@{ptr_off:#x}={ptr:#x} data {:02x?}", &d[o2..o2 + size]);
                                    }
                                }
                            }
                        }
                    }
                }
                for off in (0..0xC0).step_by(4) {
                    if let Some((n, o)) = c.block(po + off) {
                        if n > 4096 { continue }
                        let head: Vec<u8> = d.get(o..o + 64.min(d.len() - o)).unwrap_or(&[]).to_vec();
                        let sids: Vec<String> = (0..4).map(|k| c.sid(d.u32_at(o + k * 4))).collect();
                        eprintln!("     pp +{off:#04x}: {n} elems @ {o:#x} sid0..3 {sids:?} {:02x?}", &head[..head.len().min(48)]);
                    }
                }
            }
        }
    }

    /// The sky SCROLL RATES, pinned to the values the shipped caches author. The Halo 3 sky
    /// animates through the render method's postprocess FUNCTIONS block (+0x5C): a translation-X
    /// (type 5) / translation-Y (type 6) parameter whose constant mapping function outputs +-1
    /// over an authored `time_period`, so the layer advances one full uv per period.
    ///
    /// riverworld `clouds_horizon` authors ONE function: translation X, -1 over 500 s.
    /// riverworld `clouds_patchy` authors FOUR: two layers, X then Y (400/500 s, 700/600 s).
    /// zanzibar's sky `water` authors THREE: X and Y at the SAME 40 s (a diagonally scrolling
    /// ocean) plus a second layer's Y at 60 s - the pair at an identical period is what pins
    /// 5/6 as translation X/Y rather than a scale.
    #[test]
    fn sky_animation_parameters() {
        let want: &[(&str, &str, &[(u32, f32, f32)])] = &[
            ("riverworld.map", "clouds_horizon", &[(5, 500.0, -1.0)]),
            ("riverworld.map", "clouds_patchy", &[(5, 400.0, -1.0), (6, 500.0, -1.0), (5, 700.0, -1.0), (6, 600.0, -1.0)]),
            ("riverworld.map", "clouds_cumulus", &[(5, 450.0, -1.0), (6, 200.0, 1.0)]),
            ("zanzibar.map", "water", &[(5, 40.0, -1.0), (6, 40.0, -1.0), (6, 60.0, -1.0)]),
            ("guardian.map", "fog_plane", &[(5, 20.0, 1.0)]),
            // a STATIC layer: zanzibar's puffy clouds author no function at all
            ("zanzibar.map", "clouds_puffy", &[]),
        ];
        let mut checked = 0usize;
        for (map, mat, exp) in want {
            let Some(c) = open(map) else { continue };
            let Some(scnr) = c.find_tags(b"scnr").first().copied() else { continue };
            let Some((_, so)) = c.tag_meta(scnr).and_then(|sm| c.block(sm + super::super::objects::OFF_SCNR_SKIES)) else { continue };
            let Some(mode) = c.tag_ref_of(so, b"scen").and_then(|s| super::super::objects::model_of(&c, s)) else { continue };
            let model = super::super::objects::load_model(&c, mode).unwrap();
            let Some(m) = model.materials.iter().flatten().map(|&t| load_material(&c, t))
                .find(|m| m.name.rsplit('\\').next().unwrap_or("") == *mat) else { panic!("{map}: no sky material '{mat}'") };
            let got: Vec<(u32, f32, f32)> = m.anim.iter().map(|a| (a.kind, a.period, a.value)).collect();
            assert_eq!(&got[..], *exp, "{map} {mat} animated parameters");
            for a in &m.anim {
                assert_eq!(a.fn_type, 0, "{map} {mat}: expected a CONSTANT mapping function");
                assert!((a.rate() - a.value / a.period).abs() < 1e-9);
            }
            checked += 1;
        }
        if checked == 0 { eprintln!("skip: no halo3 maps"); return }
        // the layer assignment: one X and one Y per layer, a repeat starts the next layer
        let r = layer_scroll_rates(&[
            H3AnimParam { kind: 5, period: 40.0, fn_type: 0, value: -1.0 },
            H3AnimParam { kind: 6, period: 40.0, fn_type: 0, value: -1.0 },
            H3AnimParam { kind: 6, period: 60.0, fn_type: 0, value: -1.0 },
        ]);
        assert!((r[0][0] + 0.025).abs() < 1e-6 && (r[0][1] + 0.025).abs() < 1e-6, "{r:?}");
        assert!(r[1][0] == 0.0 && (r[1][1] + 1.0 / 60.0).abs() < 1e-6, "{r:?}");
        assert_eq!(r[2], [0.0, 0.0]);
        // NO layer may scroll anywhere near a full uv per second (the defect this replaced: the
        // sky pass read the (1,1) TILE as its detail translation rate)
        for (map, _mat, _) in want {
            let Some(c) = open(map) else { continue };
            let Some(scnr) = c.find_tags(b"scnr").first().copied() else { continue };
            let Some((_, so)) = c.tag_meta(scnr).and_then(|sm| c.block(sm + super::super::objects::OFF_SCNR_SKIES)) else { continue };
            let Some(mode) = c.tag_ref_of(so, b"scen").and_then(|s| super::super::objects::model_of(&c, s)) else { continue };
            let model = super::super::objects::load_model(&c, mode).unwrap();
            for t in model.materials.iter().flatten() {
                let m = load_material(&c, *t);
                for l in layer_scroll_rates(&m.anim) {
                    assert!(l[0].abs() < 0.1 && l[1].abs() < 0.1, "{map} {}: runaway sky scroll {l:?}", m.name);
                }
            }
        }
    }

    /// riverworld's terrain (`rmtr`) materials: every one names a `blend_map` and at least two
    /// `base_map_m_N` layers through its template (the layer albedos the terrain lane blends).
    #[test]
    fn riverworld_terrain_layers_resolve() {
        let Some(c) = open("riverworld.map") else { return };
        let (mut n, mut ok) = (0usize, 0usize);
        for sbsp in c.find_tags(b"sbsp") {
            for bm in load_bsp(&c, sbsp).unwrap().materials.iter().filter(|m| &m.class == b"rmtr") {
                let m = load_material(&c, bm.render_tag.unwrap());
                let short = |s: &str| s.rsplit('\\').next().unwrap_or(s).to_string();
                eprintln!("rmtr {} opts {:?} tex {:?}", short(&m.name), m.template_opts, m.params.iter().zip(&m.xforms).map(|((k, b), x)| format!("{k}={} x{:?}", short(c.tag_name(*b)), x)).collect::<Vec<_>>());
                n += 1;
                let layers = (0..4).filter(|i| m.param(&format!("base_map_m_{i}")).is_some()).count();
                if m.param("blend_map").is_some() && layers >= 2 { ok += 1; }
            }
        }
        eprintln!("riverworld rmtr: {ok}/{n} with blend_map + >= 2 layers");
        assert!(n > 0 && ok == n);
    }

    /// Probe (print-only): every `rmw` (water) render method of riverworld / construct /
    /// guardian - template, textures and the named float constants the water pass needs.
    #[test]
    fn water_materials_probe() {
        for map in ["riverworld.map", "guardian.map", "construct.map", "zanzibar.map"] {
            let Some(c) = maps_dir().map(|d| d.join(map)).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { continue };
            for t in c.find_tags(b"rmw ") {
                let m = load_material(&c, t);
                eprintln!("{map} rmw {} opts {:?} textures {:?} floats {:?}", m.name.rsplit('\\').next().unwrap_or(""), m.template_opts,
                    m.params.iter().map(|(n, b)| format!("{n}={}", c.tag_name(*b).rsplit('\\').next().unwrap_or(""))).collect::<Vec<_>>(),
                    m.floats.iter().map(|(n, v)| format!("{n}=[{:.3},{:.3},{:.3},{:.3}]", v[0], v[1], v[2], v[3])).collect::<Vec<_>>());
            }
        }
    }
    /// Probe (print-only, ignored): every render method of every scenario-placed object model
    /// (weapons / vehicles / scenery) of HMS_H3_PROBE_MAP - class, template digits, the named
    /// float constants (cook_torrance parameters) and samplers. #h3-objlight
    #[test]
    #[ignore]
    fn object_material_floats_probe() {
        use super::super::objects::{load_model, load_placements, model_of};
        let map = std::env::var("HMS_H3_PROBE_MAP").unwrap_or_else(|_| "riverworld.map".into());
        let Some(c) = open(&map) else { return };
        let short = |s: &str| s.rsplit('\\').next().unwrap_or(s).to_string();
        let mut seen = std::collections::BTreeSet::new();
        let mut tags: Vec<usize> = load_placements(&c).iter().map(|p| p.palette_tag).collect();
        tags.sort(); tags.dedup();
        for t in tags {
            let Some(mode) = model_of(&c, t) else { continue };
            let Ok(model) = load_model(&c, mode) else { continue };
            for &rt in model.materials.iter().flatten() {
                if !seen.insert(rt) { continue; }
                let m = load_material(&c, rt);
                eprintln!("{map} {} :: {} ({}) opts {:?}\n    floats {:?}\n    tex {:?}", short(&model.name), short(&m.name), String::from_utf8_lossy(&m.class), m.template_opts,
                    m.floats.iter().map(|(n, v)| format!("{n}=[{:.3},{:.3},{:.3},{:.3}]", v[0], v[1], v[2], v[3])).collect::<Vec<_>>(),
                    m.params.iter().map(|(n, b)| format!("{n}={}", short(c.tag_name(*b)))).collect::<Vec<_>>());
            }
        }
    }
}

