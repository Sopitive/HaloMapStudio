//! Halo 4 material SHADING model: what the shipped `srf_*` pixel shaders compute, as a
//! per-material descriptor the renderer's Halo 4 lane (`h4_shade` in hms-render mesh.rs) evaluates.
//!
//! Every formula here was read from the DXBC of the material shaders in the cache (`mtsb` PS
//! tables, disassembled with d3dcompiler_47 `D3DDisassemble`; scratch listings under
//! h4light/dis/). Entry 01 of each `mats` is the G-BUFFER pass (albedo, encoded normal, spec
//! mask, per-family alpha lane), entries 04/06 the deferred STATIC LIGHTING passes (04 = analytic
//! point light variant, 06 = floating sun variant); both compute the same BRDF. See
//! docs/halo4_lighting_model.md for the listings and the per-family constant maps.
//!
//! Common law (from the asm), per lobe k in {A, B} of the lightmap and the analytic sun L:
//!   diffuse  = sum_k col_k * LUT(u_k, v_k) [* gate_A for k = A] + sun_rgb * sat(N.L) / pi
//!   Blinn    spec_k = pow(sat(N . H_k), 1 / r) / pi     H_k = normalize(d_k + E), E = to-camera
//!            r = lerp(rough_min, rough_max, 1 - gloss)  (`srf_blinn*`, `srf_ca_blinn*`)
//!   Phong    spec_k = pow(sat(R . d_k), p) / pi          R = reflect(-E, N)
//!            (`srf_ca_snow_detail`: p = spec_power; `srf_ca_layered_*`: p = 1 / lerp(rough))
//!   spec     = spec_colour * sum_k col_k * spec_k        (lobe colours WITHOUT the sharpen LUT)
//!   env      = cube(reflect(-E, lerp(Nv, N, nb))) * refl_tint * refl_int * Rf * fresnel * gate_A
//!              fresnel = lerp(1, scale * pow(lerp(NdotV, 1 - NdotV, invert), power), weight)
//!              env = lerp(env, env * diffuse, env_lit_by_diffuse)
//!   out      = (albedo * diffuse_intensity) * diffuse + spec + env + self_illum,  * exposure
//! `exp`/`log` in DXBC are base 2: the literal -1.65149617 in every spec lane is log2(1 / pi).
//!
//! Colour maps are sRGB-decoded by the hardware (the per-bitmap DXGI format at resource
//! definition +0x4C: 72 / 78 / 91 = *_SRGB, checked on 5 000+ bitmaps of both MP maps);
//! `color_detail_map` / `all_layers_color_detail_map` are LINEAR-format textures multiplied by
//! 4.59479 (= 2^2.2, mid-grey neutral); DXN normal maps are BC5_SNORM (DXGI 84) and the shader
//! uses .xy raw, z = sqrt(1 - x^2 - y^2).

use super::materials::H4Material;

/// Which specular lobe the family evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecModel {
    /// No specular term (`srf_lambert`, `srf_constant`, unknown families).
    None,
    /// (N.H)^(1/r) / pi, r = lerp(rough_min, rough_max, 1 - gloss).
    Blinn,
    /// (R.L)^p / pi with p = spec_power (a constant, no gloss map).
    PhongPower,
    /// (R.L)^(1/r) / pi, r = lerp(rough_min, rough_max, 1 - gloss).
    PhongRough,
}

/// Where the per-texel specular colour / gloss come from (`H4Shading::spec_src` in the shader:
/// 0 none, 1 specular_map (rgb colour, a gloss), 2 control SpGlRf (r spec, g gloss, b reflection),
/// 3 control SpGlSi (r spec, g gloss, b self-illum), 4 diffspec (spec colour derived from the colour
/// map), 5 specular_map rgb x control SpGlRf, 6 control SpGlRf with self-illum in .a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecSource { None, SpecularMap, ControlSpGlRf, ControlSpGlSi, DiffSpec, SpecMapAndControl, ControlSpGlRfSiAlpha }

/// Texture roles the Halo 4 lane binds (resolved from the material's reflected parameter names).
#[derive(Clone, Debug, Default)]
pub struct H4TexRoles {
    pub color: Option<usize>,
    pub color_xform: [f32; 4],
    pub normal: Option<usize>,
    pub normal_xform: [f32; 4],
    pub normal_detail: Option<usize>,
    pub normal_detail_xform: [f32; 4],
    pub color_detail: Option<usize>,
    pub color_detail_xform: [f32; 4],
    /// `specular_map` (rgb spec colour, a gloss on srf_blinn / srf_blinn_detail).
    pub specular: Option<usize>,
    pub specular_xform: [f32; 4],
    /// `control_map_SpGlRf` / `control_map_SpGlSi`.
    pub control: Option<usize>,
    pub control_xform: [f32; 4],
    pub reflection_cube: Option<usize>,
    pub self_illum: Option<usize>,
    pub self_illum_xform: [f32; 4],
    /// `srf_ca_snow_detail` top layer (`color1_map` / `normal1_map`).
    pub color1: Option<usize>,
    pub color1_xform: [f32; 4],
    pub normal1: Option<usize>,
    pub normal1_xform: [f32; 4],
    /// `pcc_amount_map` of the `*_colorchangemap` families (entry 01 asm: albedo =
    /// lerp(c, luma709(c) * change_colour.rgb, pcc.r * change_colour.w) * tint; the engine's
    /// neutral / no-team primary change colour is (0.5, 0.5, 0.5, 1) - the Forge pieces).
    pub pcc: Option<usize>,
    pub pcc_xform: [f32; 4],
}

/// The decoded shading constants of one material (read from each family's shader unless noted).
#[derive(Clone, Debug)]
pub struct H4Shading {
    /// Leaf name of the `mats` (`srf_blinn_reflection`, ...).
    pub family: String,
    /// True when the family's constant map was read from its shader (else the generic fallback).
    pub known: bool,
    pub spec_model: SpecModel,
    pub spec_src: SpecSource,
    pub tex: H4TexRoles,
    pub albedo_tint: [f32; 3],
    pub diffuse_intensity: f32,
    /// Spec mask from the colour map alpha: mask = lerp(1, color.a, w). `srf_blinn_detail`
    /// multiplies color.a directly (w = 1) and adds lerp(1, detail.a, detail_spec_weight).
    pub spec_mask_alpha_weight: f32,
    pub detail_spec_weight: f32,
    pub spec_color: [f32; 3],
    pub spec_intensity: f32,
    /// Blinn / PhongRough: roughness at gloss 1 / gloss 0. PhongPower: x = the exponent.
    pub rough_min: f32,
    pub rough_max: f32,
    pub spec_tint_by_albedo: f32,
    /// diffspec: spec colour = sat(pow(lerp(c, lum(c), desat), pow) * scale + bias).
    pub diffspec: [f32; 4],
    pub refl_tint: [f32; 3],
    pub refl_intensity: f32,
    pub refl_normal_blend: f32,
    /// fresnel = lerp(1, scale * pow(lerp(NdotV, 1 - NdotV, invert), power), weight).
    pub fresnel_scale: f32,
    pub fresnel_power: f32,
    pub fresnel_weight: f32,
    pub fresnel_invert: f32,
    pub env_lit_by_diffuse: f32,
    pub normal_detail_fade_end: f32,
    pub normal_detail_fade_start: f32,
    /// srf_ca_snow_detail: lerp(base normal, base + detail, strength); 1 elsewhere.
    pub normal_detail_strength: f32,
    pub self_illum_color: [f32; 3],
    pub self_illum_intensity: f32,
    /// 0 none, 1 = control.b * col * int^2 (srf_blinn_selfillum), 2 = albedo * col * int * control.a
    /// (srf_blinn_reflection_selfillum), 3 = selfillum_map.rgb * col * int (PROVISIONAL),
    /// 4 = col * int * color_map.a (`srf_char_cov_selfillum`), 5 = col * int, a pure constant
    /// (`srf_char_constant`).
    pub self_illum_mode: u8,
    pub self_illum_blend: f32,
    /// srf_ca_snow_detail second layer.
    pub snow: Option<SnowLayer>,
    /// srf_ca_layered_three_height_detailnormal (PROVISIONAL subset: the colour blend is exact, the
    /// per-layer spec constants collapse to layer 0).
    pub layered: Option<Layered3>,
    /// `srf_ca_boundary*`: the additive self-illum palette family (drawn by the renderer's
    /// halogram lane, not by `h4_shade`).
    pub boundary: Option<Boundary>,
    /// `srf_char_cov*` (`#h4-veh`): the Covenant three-colour view-angle ramp.
    pub char_cov: Option<CharCov>,
}

// #h4-veh
/// `srf_char_cov` / `srf_char_cov_specdetail` / `srf_char_cov_selfillum`: the Covenant
/// character / vehicle body family (the Wraith, Ghost, Revenant, Banshee hulls). Its albedo is NOT
/// the colour map - the colour map is a sheen / pattern sheet that gets crushed to a few per cent
/// and re-tinted by a **three-colour SCREEN ramp driven by the view angle** and masked by the
/// control map's ALPHA. Read from entry 01 of both families (byte-identical albedo block; a second,
/// swizzle-unambiguous reading in the forward entry 21):
///
/// ```text
/// c  = sat(dot(N, E))        E = to-camera, N = the mapped tangent normal
/// f  = 1 - c
/// wA = f^up163.w      wC = f^up162.w      wB = c^up161.w
/// tC = (1 - wA) * wC
/// tB = wB * (1 - tC)
/// ramp = 1 - control.a * (1 - tB*up161.rgb) * (1 - wA*up163.rgb) * (1 - tC*up162.rgb)
/// albedo = color_map.rgb * up160.rgb * ramp
/// ```
///
/// So `up161` is the FACE-ON colour (c = 1 -> albedo = color * up160 * up161), `up163` the GRAZING
/// colour (c = 0) and `up162` the mid-angle one. `up160.w` is never read.
///
/// The rest of the family: diffuse intensity `up164.x`; specular = the standard three-lobe Blinn
/// with `spec_col = control.r`, `gloss = control.g`, roughness `lerp(up165.y, up165.z, 1 - gloss)`,
/// colour `lerp(up164.yzw, albedo, up165.w)`, intensity `up165.x`, and - `_specdetail` only -
/// x `spec_detail_map.r` (the G-buffer spec mask; `srf_char_cov` writes 0 there and has no
/// spec_detail_map at all). Reflection = `cube(reflect(-E, N))` x `up166.rgb`, saturated by
/// `lerp(luma709, tinted, up166.w)`, x `(1 - NdotV)^up163.w` (the SAME exponent as the ramp's
/// `wA` - there is no fresnel scale/weight/invert triple here) x `control.b` x **`diffuse.r`**
/// (the family is hard-wired env-lit-by-diffuse, and only by the RED channel).
/// `_selfillum` adds `up167.rgb * up167.w * color_map.a`.
///
/// These families never bind `EngineMaterialPS`, so a Covenant vehicle body takes NO change
/// colour - its colour is entirely `up160`..`up163`.
#[derive(Clone, Debug)]
pub struct CharCov {
    /// up161 rgb + w: the face-on colour and its `c^w` exponent.
    pub face: [f32; 4],
    /// up162 rgb + w: the mid-angle colour and its `f^w` exponent.
    pub mid: [f32; 4],
    /// up163 rgb + w: the grazing colour, its `f^w` exponent, and the reflection fresnel power.
    pub graze: [f32; 4],
    /// up166.w: the reflection saturation (`lerp(luma709, tinted, w)`).
    pub refl_saturation: f32,
    /// `spec_detail_map` (`_specdetail` only): a plain multiplier on the whole specular term.
    pub spec_detail: Option<usize>,
    pub spec_detail_xform: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct SnowLayer {
    pub color1_tint: [f32; 3],
    pub spec1_color: [f32; 3],
    pub spec1_intensity: f32,
    pub spec1_power: f32,
    /// World-space direction the layer accumulates on (normalized `up8.zxy`).
    pub dir: [f32; 3],
    pub bias: f32,
    pub coverage_scale: f32,
    pub coverage_power: f32,
    pub normal_blend: f32,
    pub alpha_mask_weight: f32,
}

#[derive(Clone, Debug)]
pub struct Layered3 {
    pub blend: Option<usize>,
    pub layer_co: [Option<usize>; 3],
    pub layer_co_xform: [[f32; 4]; 3],
    pub layer_nm: [Option<usize>; 3],
    pub layer_nm_xform: [[f32; 4]; 3],
    pub layer_tint: [[f32; 3]; 3],
    pub all_detail: Option<usize>,
    pub all_detail_xform: [f32; 4],
    /// [height-vs-mask weight, softness weight] for layers 1 and 2.
    pub blend1: [f32; 2],
    pub blend2: [f32; 2],
    /// 0 = the height/softness law (`srf_ca_layered_*_height*`), 1 = plain mask sum
    /// `l0 * B.r + l1 * B.g + l2 * B.b` (`srf_layered_three`), 2 = the same with B normalised by
    /// |B.rgba| (`srf_layered_two_detail*`).
    pub mask_mode: u8,
}

// #grid-render
/// `srf_ca_boundary` / `srf_ca_boundary_noscale` (the Halo 4 Forge GRID piece `fw_grid`, the
/// boundary walls): a fully self-illuminated ADDITIVE family with no lit term at all. Read from
/// the forward transparent entries (`ps21/22/23/31/32/33`, byte-identical in both families and
/// between them; the deferred entry 01 writes o0 = 0 and only the encoded normal, so the family
/// never reaches the G-buffer lighting passes). cbNN: cb11[k] = the `mat` texture xform of slot k,
/// cb13[i] = `user_parameter_16i`, cb2[0] = ps_material_object_parameters (the change colour),
/// cb3[0] = psDepthConstants, cb0[7] = ps_view_exposure.
///
/// ```text
/// NdotV  = sat(dot(-normalize(P - eye), N))
/// fres   = sat(up161.y * pow(sat(lerp(NdotV, 1 - NdotV, up161.w)), up161.z))
/// hfade  = sat(1 - uv.y / up163.z)                       (up163.z = 0 on every shipped material)
/// edge   = lerp(fres, 1, hfade)
/// soft   = (1 - sat((scene_depth - frag_depth) / up163.x))^4        (soft intersection glow)
/// v      = sat(sqrt(edge^2 + soft) * (1 - up163.w * |noise_a.r - noise_b.r|))
/// pal    = palette(v * xf5.x + xf5.z, up162.w * xf5.y + xf5.w).r
/// si     = pow(|pal|, up163.y) * up160.rgb * change_colour.rgb * up161.x * alpha_mask.a
/// ovl    = overlay.rgb * overlay_detail.rgb * up160.rgb * up160.w * 4.59479
/// fade   = 1 - sat(up162.x * pow(sat(lerp(NdotV, 1 - NdotV, up162.z)), up162.y))
/// out    = max(0, ((si + ovl) * fade) * fog_extinction + fog_inscatter) * exposure
/// ```
#[derive(Clone, Debug)]
pub struct Boundary {
    pub overlay: Option<usize>,
    pub overlay_xform: [f32; 4],
    pub overlay_detail: Option<usize>,
    pub overlay_detail_xform: [f32; 4],
    pub alpha_mask: Option<usize>,
    pub alpha_mask_xform: [f32; 4],
    pub noise_a: Option<usize>,
    pub noise_a_xform: [f32; 4],
    pub noise_b: Option<usize>,
    pub noise_b_xform: [f32; 4],
    pub palette: Option<usize>,
    pub palette_xform: [f32; 4],
    /// up160: the family's ONE colour - it tints both the palette self-illum and the overlay.
    pub tint: [f32; 3],
    /// up160.w: overlay intensity (x the 4.59479 linear-detail constant).
    pub overlay_intensity: f32,
    /// up161: self-illum intensity, then the palette-index fresnel (scale, power, invert).
    pub si_intensity: f32,
    pub fres_scale: f32,
    pub fres_power: f32,
    pub fres_invert: f32,
    /// up162: the OUTPUT fade fresnel (scale, power, invert) + the palette v coordinate.
    pub fade_scale: f32,
    pub fade_power: f32,
    pub fade_invert: f32,
    pub palette_v: f32,
    /// up163: soft-intersection range (wu), palette exponent, uv.y fade denominator, noise weight.
    pub depth_fade_range: f32,
    pub palette_power: f32,
    pub height_fade: f32,
    pub noise_diff_scale: f32,
}

fn f4(m: &H4Material, i: usize) -> [f32; 4] { m.floats.get(i).map(|f| f.1).unwrap_or([0.0; 4]) }
fn rgb(v: [f32; 4]) -> [f32; 3] { [v[0], v[1], v[2]] }

fn tex(m: &H4Material, names: &[&str]) -> (Option<usize>, [f32; 4]) {
    for n in names {
        if let Some(t) = m.textures.iter().find(|t| t.name.eq_ignore_ascii_case(n)) { return (t.bitm, t.xform); }
    }
    (None, [1.0, 1.0, 0.0, 0.0])
}

impl H4Shading {
    fn base(m: &H4Material, family: &str) -> H4Shading {
        let mut t = H4TexRoles::default();
        // layer0_coMap / r_color: the layered terrain families that are not decoded
        // (srf_env_m30_rock_occlusion, ...) at least get their first layer instead of white
        (t.color, t.color_xform) = tex(m, &["color_map", "diffuseMap", "diffuse_map", "base_map", "baseMap", "color1_map", "layer0_coMap", "r_color"]);
        (t.normal, t.normal_xform) = tex(m, &["normal_map", "normalMap"]);
        (t.normal_detail, t.normal_detail_xform) = tex(m, &["normal_detail_map"]);
        (t.color_detail, t.color_detail_xform) = tex(m, &["color_detail_map"]);
        (t.specular, t.specular_xform) = tex(m, &["specular_map"]);
        (t.control, t.control_xform) = tex(m, &["control_map_SpGlRf", "control_map_SpGlSi", "control_map_SpGlRfSi"]);
        (t.reflection_cube, _) = tex(m, &["reflection_map"]);
        (t.self_illum, t.self_illum_xform) = tex(m, &["selfillum_map", "selfIllumMap", "self_illum_map"]);
        (t.color1, t.color1_xform) = tex(m, &["color1_map"]);
        (t.normal1, t.normal1_xform) = tex(m, &["normal1_map"]);
        (t.pcc, t.pcc_xform) = tex(m, &["pcc_amount_map"]);
        // srf_ca_snow_detail names its base as color_map too; color1_map must not be the base
        if family.contains("snow") { (t.color, t.color_xform) = tex(m, &["color_map"]); }
        H4Shading {
            family: family.to_string(),
            known: false,
            spec_model: SpecModel::None,
            spec_src: SpecSource::None,
            tex: t,
            albedo_tint: [1.0; 3],
            diffuse_intensity: 1.0,
            spec_mask_alpha_weight: 0.0,
            detail_spec_weight: 0.0,
            spec_color: [1.0; 3],
            spec_intensity: 0.0,
            rough_min: 0.05,
            rough_max: 1.0,
            spec_tint_by_albedo: 0.0,
            diffspec: [0.0, 1.0, 1.0, 0.0],
            refl_tint: [1.0; 3],
            refl_intensity: 0.0,
            refl_normal_blend: 1.0,
            fresnel_scale: 1.0,
            fresnel_power: 1.0,
            fresnel_weight: 0.0,
            fresnel_invert: 1.0,
            env_lit_by_diffuse: 0.0,
            normal_detail_fade_end: 0.0,
            normal_detail_fade_start: 0.0,
            normal_detail_strength: 1.0,
            self_illum_color: [0.0; 3],
            self_illum_intensity: 0.0,
            self_illum_mode: 0,
            self_illum_blend: 1.0,
            snow: None,
            layered: None,
            boundary: None,
            char_cov: None,
        }
    }

    /// Decode a material's shading constants from its `mats` family + float4 constants.
    pub fn from_material(m: &H4Material) -> H4Shading {
        let fam = m.mats_name.rsplit('\\').next().unwrap_or("").to_ascii_lowercase();
        let mut s = H4Shading::base(m, &fam);
        let p = |i: usize| f4(m, i);
        let detail_on = m.bools & 1 != 0;
        match fam.as_str() {
            // srf_blinn: color, normal, specular (rgb colour * gloss a), normal_detail
            "srf_blinn" | "srf_blinn_clip" | "srf_blinn_vertalpha" | "srf_blinn_falpha" | "srf_blinn_colorchangemap" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::SpecularMap;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = p(0)[3];
                s.spec_mask_alpha_weight = p(1)[0]; s.spec_color = [p(1)[1], p(1)[2], p(1)[3]];
                s.spec_intensity = p(2)[0]; s.rough_min = p(2)[1]; s.rough_max = p(2)[2]; s.spec_tint_by_albedo = p(2)[3];
                s.normal_detail_fade_end = p(3)[0]; s.normal_detail_fade_start = p(3)[1];
            }
            // srf_blinn_reflection: + control SpGlRf, reflection cube (the colorchangemap variant =
            // the same lighting constants (entry 06 asm identical cb13 usage) + the pcc lane)
            "srf_blinn_reflection" | "srf_blinn_reflection_clip" | "srf_blinn_reflection_colorchangemap" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::SpecMapAndControl;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = p(0)[3];
                s.env_lit_by_diffuse = p(1)[0]; s.spec_mask_alpha_weight = p(1)[1];
                s.spec_color = rgb(p(2)); s.spec_intensity = p(2)[3];
                s.rough_min = p(3)[0]; s.rough_max = p(3)[1]; s.spec_tint_by_albedo = p(3)[2];
                s.refl_tint = rgb(p(4)); s.refl_intensity = p(4)[3];
                s.refl_normal_blend = p(5)[0]; s.fresnel_scale = p(5)[1]; s.fresnel_power = p(5)[2]; s.fresnel_weight = p(5)[3];
                s.fresnel_invert = p(6)[0]; s.normal_detail_fade_end = p(6)[1]; s.normal_detail_fade_start = p(6)[2];
            }
            // srf_blinn_reflection_selfillum: self-illum = albedo * up1.rgb * up1.w * control.a
            "srf_blinn_reflection_selfillum" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::ControlSpGlRfSiAlpha;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = p(0)[3];
                s.self_illum_color = rgb(p(1)); s.self_illum_intensity = p(1)[3]; s.self_illum_mode = 2;
                s.self_illum_blend = p(2)[0].min(1.0); s.env_lit_by_diffuse = p(2)[1]; s.spec_mask_alpha_weight = p(2)[2];
                s.spec_color = rgb(p(3)); s.spec_intensity = p(3)[3];
                s.rough_min = p(4)[0]; s.rough_max = p(4)[1]; s.spec_tint_by_albedo = p(4)[2];
                s.refl_tint = rgb(p(5)); s.refl_intensity = p(5)[3];
                s.refl_normal_blend = p(6)[0]; s.fresnel_scale = p(6)[1]; s.fresnel_power = p(6)[2]; s.fresnel_weight = p(6)[3];
                s.fresnel_invert = p(7)[0]; s.normal_detail_fade_end = p(7)[1]; s.normal_detail_fade_start = p(7)[2];
            }
            // srf_blinn_selfillum: control SpGlSi, no colour-alpha spec mask
            "srf_blinn_selfillum" | "srf_blinn_selfillum_clip" | "srf_blinn_selfillum_mod" | "srf_blinn_selfillum_colorchangemap" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::ControlSpGlSi;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = p(0)[3];
                s.spec_color = rgb(p(1)); s.spec_intensity = p(1)[3];
                s.rough_min = p(2)[0]; s.rough_max = p(2)[1]; s.spec_tint_by_albedo = p(2)[2];
                s.self_illum_color = rgb(p(3)); s.self_illum_intensity = p(3)[3]; s.self_illum_mode = 1;
                s.normal_detail_fade_end = p(4)[0]; s.normal_detail_fade_start = p(4)[1];
            }
            // srf_blinn_detail: albedo = color * detail * 4.59; spec map rgb + gloss a
            "srf_blinn_detail" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::SpecularMap;
                s.detail_spec_weight = p(0)[0]; s.albedo_tint = [p(0)[1], p(0)[2], p(0)[3]];
                s.spec_mask_alpha_weight = 1.0;
                s.diffuse_intensity = p(1)[0];
                s.spec_color = rgb(p(2)); s.spec_intensity = p(2)[3];
                s.rough_min = p(3)[0]; s.rough_max = p(3)[1]; s.spec_tint_by_albedo = p(3)[2];
                s.normal_detail_fade_end = p(3)[3]; s.normal_detail_fade_start = p(4)[0];
            }
            // srf_ca_blinn_diffspec_reflection: spec colour from the colour map + SpGlRf + cube
            "srf_ca_blinn_diffspec_reflection" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::DiffSpec;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = p(0)[3];
                s.env_lit_by_diffuse = p(1)[0]; s.spec_mask_alpha_weight = p(1)[1];
                s.spec_color = rgb(p(2)); s.spec_intensity = p(2)[3];
                s.rough_min = p(3)[0]; s.rough_max = p(3)[1]; s.spec_tint_by_albedo = p(3)[2];
                s.diffspec = [p(3)[3], p(4)[0], p(4)[1], p(4)[2]];
                s.refl_tint = rgb(p(5)); s.refl_intensity = p(5)[3];
                s.refl_normal_blend = p(6)[0]; s.fresnel_scale = p(6)[1]; s.fresnel_power = p(6)[2]; s.fresnel_weight = p(6)[3];
                s.fresnel_invert = p(7)[0]; s.normal_detail_fade_end = p(7)[1]; s.normal_detail_fade_start = p(7)[2];
            }
            // srf_ca_blinn_diffspec_detail: colour detail * 4.59, diffspec, FIXED roughness up3.x
            "srf_ca_blinn_diffspec_detail" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::DiffSpec;
                s.detail_spec_weight = p(0)[0]; s.albedo_tint = [p(0)[1], p(0)[2], p(0)[3]];
                s.spec_mask_alpha_weight = 1.0;
                s.diffuse_intensity = p(1)[0];
                s.spec_color = rgb(p(2)); s.spec_intensity = p(2)[3];
                s.rough_min = p(3)[0]; s.rough_max = p(3)[0]; s.spec_tint_by_albedo = p(3)[2];
                s.diffspec = [p(3)[3], p(4)[0], p(4)[1], p(4)[2]];
                s.normal_detail_fade_end = p(5)[0]; s.normal_detail_fade_start = p(4)[3];
            }
            // srf_ca_snow_detail: Phong; top layer (color1 / normal1) by world direction coverage
            "srf_ca_snow_detail" => {
                s.known = true;
                s.spec_model = SpecModel::PhongPower;
                s.spec_src = SpecSource::None;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = 1.0;
                s.detail_spec_weight = p(1)[0]; s.normal_detail_fade_end = p(1)[1]; s.normal_detail_fade_start = p(1)[2];
                s.normal_detail_strength = p(1)[3];
                s.spec_color = rgb(p(2)); s.spec_intensity = p(2)[3];
                s.rough_min = p(3)[0]; s.spec_tint_by_albedo = p(3)[1]; s.spec_mask_alpha_weight = p(3)[2];
                let d = [p(8)[2], p(8)[0], p(8)[1]];
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
                s.snow = Some(SnowLayer {
                    color1_tint: rgb(p(4)),
                    spec1_color: rgb(p(5)), spec1_intensity: p(5)[3],
                    spec1_power: p(6)[0],
                    bias: p(6)[3],
                    coverage_scale: p(7)[0], coverage_power: p(7)[1], normal_blend: p(7)[2], alpha_mask_weight: p(7)[3],
                    dir: [d[0] / l, d[1] / l, d[2] / l],
                });
            }
            // srf_ca_layered_three_height_detailnormal: height-blended 3-layer terrain, Phong
            "srf_ca_layered_three_height_detailnormal" | "srf_ca_layered_three_height" => {
                s.known = true;
                s.spec_model = SpecModel::PhongRough;
                s.spec_src = SpecSource::DiffSpec;
                let (blend, _) = tex(m, &["blend_map"]);
                let l0 = tex(m, &["layer0_coMap"]); let l1 = tex(m, &["layer1_coMap"]); let l2 = tex(m, &["layer2_coMap"]);
                let n0 = tex(m, &["layer0_nmMap"]); let n1 = tex(m, &["layer1_nmMap"]); let n2 = tex(m, &["layer2_nmMap"]);
                let ad = tex(m, &["all_layers_color_detail_map"]);
                s.tex.color = l0.0; s.tex.color_xform = l0.1;
                s.tex.normal = n0.0; s.tex.normal_xform = n0.1;
                s.albedo_tint = rgb(p(1)); s.diffuse_intensity = 1.0;
                // layer 0 spec constants stand in for the blended triple (PROVISIONAL)
                s.spec_color = rgb(p(2)); s.spec_tint_by_albedo = p(2)[3];
                s.diffspec = [p(3)[1], p(3)[2], p(3)[3], p(4)[0]];
                s.rough_min = p(13)[0]; s.rough_max = p(13)[1];
                s.spec_intensity = p(12)[3];
                s.layered = Some(Layered3 {
                    blend,
                    layer_co: [l0.0, l1.0, l2.0], layer_co_xform: [l0.1, l1.1, l2.1],
                    layer_nm: [n0.0, n1.0, n2.0], layer_nm_xform: [n0.1, n1.1, n2.1],
                    layer_tint: [rgb(p(1)), rgb(p(5)), [p(9)[1], p(9)[2], p(9)[3]]],
                    all_detail: ad.0, all_detail_xform: ad.1,
                    blend1: [p(8)[1], p(8)[2]], blend2: [p(12)[1], p(12)[2]],
                    mask_mode: 0,
                });
            }
            // The other layered terrain families (entry 01 of each disassembled,
            // h4rad/dis/*_ps01_A.asm): the SAME height / softness colour blend as
            // srf_ca_layered_three_height_detailnormal, only the constant slots differ (without
            // this table they fall to the generic lane with NO colour map: white albedo = the
            // washed-out terrains of Redoubt / Basin / Erosion / Bonanza / Blood Crash).
            //   redoubt / basin / basin_detail: l0 tint p0.yzw, l1 p3.xyz, l2 p6.yzw, blend1 (p5.z, p5.w),
            //     blend2 (p8.z, p8.w); one detailNmMap; basin samples the blend map (and the colour
            //     detail) at uv2 (PROVISIONAL: HMS samples it at uv1)
            //   three_height_detailnormalx2: the decoded family's slots (+ a second layer0 detail normal)
            //   one_reflection: l0 p1, l1 p4, l2 p8, blend1 (p6.z, p6.w), blend2 (p10.z, p10.w)
            //   srf_layered_three_height(_detailnormal): l0 p0, l1 p1.xyz, l2 p2.yzw, blend1 (p1.w, p2.x),
            //     blend2 (p3.x, p3.y)
            //   two_height_detailnormal_reflection: 2 layers: l0 p1, l1 p7 masked by blend.g with
            //     (p10.y, p10.z) -> layer slot 2 (slot 1 = layer 0 again, weight 0)
            // Spec constants stay the layer-0 stand-in of the decoded family (PROVISIONAL).
            "srf_ca_layered_redoubt" | "srf_ca_layered_basin" | "srf_ca_layered_basin_detail"
            | "srf_ca_layered_three_height_detailnormalx2" | "srf_ca_layered_one_reflection"
            | "srf_layered_three_height" | "srf_layered_three_height_detailnormal"
            | "srf_ca_layered_two_height_detailnormal_reflection" => {
                s.known = true;
                s.spec_model = SpecModel::PhongRough;
                s.spec_src = SpecSource::DiffSpec;
                let (blend, _) = tex(m, &["blend_map"]);
                let l0 = tex(m, &["layer0_coMap"]); let mut l1 = tex(m, &["layer1_coMap"]); let mut l2 = tex(m, &["layer2_coMap"]);
                let n0 = tex(m, &["layer0_nmMap"]); let mut n1 = tex(m, &["layer1_nmMap"]); let mut n2 = tex(m, &["layer2_nmMap"]);
                let ad = tex(m, &["all_layers_color_detail_map", "color_detail_map"]);
                let (t0, t1, t2, b1, b2): ([f32; 3], [f32; 3], [f32; 3], [f32; 2], [f32; 2]) = match fam.as_str() {
                    "srf_ca_layered_three_height_detailnormalx2" => (rgb(p(1)), rgb(p(5)), [p(9)[1], p(9)[2], p(9)[3]], [p(8)[1], p(8)[2]], [p(12)[1], p(12)[2]]),
                    "srf_ca_layered_one_reflection" => (rgb(p(1)), rgb(p(4)), rgb(p(8)), [p(6)[2], p(6)[3]], [p(10)[2], p(10)[3]]),
                    "srf_layered_three_height" | "srf_layered_three_height_detailnormal" => (rgb(p(0)), rgb(p(1)), [p(2)[1], p(2)[2], p(2)[3]], [p(1)[3], p(2)[0]], [p(3)[0], p(3)[1]]),
                    "srf_ca_layered_two_height_detailnormal_reflection" => {
                        // two layers: slot 1 = layer 0 (weight 0), slot 2 = layer 1 by blend.g
                        l2 = l1; n2 = n1; l1 = l0; n1 = n0;
                        (rgb(p(1)), rgb(p(1)), rgb(p(7)), [0.0, 0.0], [p(10)[1], p(10)[2]])
                    }
                    _ => ([p(0)[1], p(0)[2], p(0)[3]], rgb(p(3)), [p(6)[1], p(6)[2], p(6)[3]], [p(5)[2], p(5)[3]], [p(8)[2], p(8)[3]]),
                };
                s.tex.color = l0.0; s.tex.color_xform = l0.1;
                s.tex.normal = n0.0; s.tex.normal_xform = n0.1;
                s.albedo_tint = t0; s.diffuse_intensity = 1.0;
                s.spec_color = [1.0; 3]; s.spec_tint_by_albedo = 0.0;
                s.diffspec = [0.0, 1.0, 1.0, 0.0];
                s.rough_min = 0.05; s.rough_max = 1.0;
                s.spec_intensity = 0.0;
                s.layered = Some(Layered3 {
                    blend,
                    layer_co: [l0.0, l1.0, l2.0], layer_co_xform: [l0.1, l1.1, l2.1],
                    layer_nm: [n0.0, n1.0, n2.0], layer_nm_xform: [n0.1, n1.1, n2.1],
                    layer_tint: [t0, t1, t2],
                    all_detail: ad.0, all_detail_xform: ad.1,
                    blend1: b1, blend2: b2,
                    mask_mode: 0,
                });
            }
            // Mask-blended layered families (entry 01 asm): albedo = r_color * p0.rgb *
            // B.r + g_color * p1.rgb * B.g + b_color * p2.rgb * B.b, B = blend_map (the two_detail
            // variants normalise B by |B.rgba| first). Valhalla / Forge Island / Grind / Shatter /
            // Wreckage terrains.
            "srf_layered_two_detail" | "srf_layered_two_detail_spec" | "srf_layered_two_detail_specmap_vertcolor" | "srf_layered_three" => {
                s.known = true;
                s.spec_model = SpecModel::None;
                let (blend, _) = tex(m, &["blend_map"]);
                let l0 = tex(m, &["r_color"]); let l1 = tex(m, &["g_color"]); let l2 = tex(m, &["b_color"]);
                let n0 = tex(m, &["r_normal"]); let n1 = tex(m, &["g_normal"]); let n2 = tex(m, &["b_normal"]);
                s.tex.color = l0.0; s.tex.color_xform = l0.1;
                s.tex.normal = n0.0; s.tex.normal_xform = n0.1;
                s.tex.normal_detail = None;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = 1.0;
                s.spec_intensity = 0.0;
                s.layered = Some(Layered3 {
                    blend,
                    layer_co: [l0.0, l1.0, l2.0], layer_co_xform: [l0.1, l1.1, l2.1],
                    layer_nm: [n0.0, n1.0, n2.0], layer_nm_xform: [n0.1, n1.1, n2.1],
                    layer_tint: [rgb(p(0)), rgb(p(1)), rgb(p(2))],
                    all_detail: None, all_detail_xform: [1.0, 1.0, 0.0, 0.0],
                    blend1: [0.0, 0.0], blend2: [0.0, 0.0],
                    mask_mode: if fam == "srf_layered_three" { 1 } else { 2 },
                });
            }
            // The *_detail reflection variants share srf_blinn_detail's G-buffer law
            // (entry 01 asm: albedo = color * color_detail * p0.yzw * 4.59479, spec mask
            // lerp(1, detail.a, p0.x) * color.a); the lighting constants are srf_blinn_detail's
            // (PROVISIONAL for the reflection lanes, which stay off).
            "srf_blinn_reflection_detail" | "srf_blinn_reflection_detail_uv2" | "srf_ca_blinn_diffspec_reflection_detail" | "srf_blinn_detail_uv2" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = if s.tex.specular.is_some() { SpecSource::SpecularMap } else { SpecSource::None };
                s.detail_spec_weight = p(0)[0]; s.albedo_tint = [p(0)[1], p(0)[2], p(0)[3]];
                s.spec_mask_alpha_weight = 1.0;
                s.diffuse_intensity = p(1)[0];
                s.spec_color = rgb(p(2)); s.spec_intensity = if s.tex.specular.is_some() { p(2)[3] } else { 0.0 };
                s.rough_min = p(3)[0]; s.rough_max = p(3)[1]; s.spec_tint_by_albedo = p(3)[2];
                s.normal_detail_fade_end = p(3)[3]; s.normal_detail_fade_start = p(4)[0];
            }
            // srf_special_sky_ocean_calm (Forge Island's sea; entries 01/02 disassembled,
            // h4expo/ocean): albedo = color_map * up1.rgb, diffuse intensity up1.w, normal = normal_map *
            // up0.x + normal1_map * up0.y faded by distance (up0.z end, up0.w start), G-buffer alpha =
            // fresnel lerp(1, up3.x * pow(lerp(NdotV, 1 - NdotV, up3.w), up3.y), up3.z) * (1 - sat((dist -
            // up4.y) / (up4.x - up4.y))); lit: lightmap diffuse + cube(reflect) * up2.rgb * up2.w * alpha
            // * cube.a, NO specular. The second normal map rides the normal_detail slot (PROVISIONAL:
            // the distance fade applies to both maps in the engine, HMS fades the detail only).
            "srf_special_sky_ocean_calm" => {
                s.known = true;
                s.spec_model = SpecModel::None;
                s.albedo_tint = rgb(p(1)); s.diffuse_intensity = p(1)[3];
                s.refl_tint = rgb(p(2)); s.refl_intensity = p(2)[3];
                s.refl_normal_blend = 1.0;
                s.fresnel_scale = p(3)[0]; s.fresnel_power = p(3)[1]; s.fresnel_weight = p(3)[2]; s.fresnel_invert = p(3)[3];
                s.normal_detail_strength = p(0)[1];
                s.normal_detail_fade_end = p(0)[2]; s.normal_detail_fade_start = p(0)[3];
                if s.tex.normal_detail.is_none() { s.tex.normal_detail = s.tex.normal1; s.tex.normal_detail_xform = s.tex.normal1_xform; }
            }
            // srf_ca_boundary / srf_ca_boundary_noscale: the additive self-illum palette family
            // (the Forge grid). See the `Boundary` doc for the asm-derived law; the two families
            // share one pixel-shader body, the "noscale" variant only differs in its vertex shader.
            "srf_ca_boundary" | "srf_ca_boundary_noscale" => {
                s.known = true;
                s.spec_model = SpecModel::None;
                let tx = |n: &str| tex(m, &[n]);
                let (overlay, overlay_xform) = tx("overlay_map");
                let (overlay_detail, overlay_detail_xform) = tx("overlay_detail_map");
                let (alpha_mask, alpha_mask_xform) = tx("self_illum_alpha_mask_map");
                let (noise_a, noise_a_xform) = tx("self_illum_noise_map_a");
                let (noise_b, noise_b_xform) = tx("self_illum_noise_map_b");
                let (palette, palette_xform) = tx("self_illum_palette_map");
                s.boundary = Some(Boundary {
                    overlay, overlay_xform, overlay_detail, overlay_detail_xform,
                    alpha_mask, alpha_mask_xform, noise_a, noise_a_xform, noise_b, noise_b_xform,
                    palette, palette_xform,
                    tint: rgb(p(0)), overlay_intensity: p(0)[3],
                    si_intensity: p(1)[0], fres_scale: p(1)[1], fres_power: p(1)[2], fres_invert: p(1)[3],
                    fade_scale: p(2)[0], fade_power: p(2)[1], fade_invert: p(2)[2], palette_v: p(2)[3],
                    depth_fade_range: p(3)[0], palette_power: p(3)[1], height_fade: p(3)[2], noise_diff_scale: p(3)[3],
                });
            }
            // #h4-veh srf_char_cov*: the Covenant body ramp (see the `CharCov` doc for the asm).
            "srf_char_cov" | "srf_char_cov_specdetail" | "srf_char_cov_selfillum" => {
                s.known = true;
                s.spec_model = SpecModel::Blinn;
                s.spec_src = SpecSource::ControlSpGlRf;
                let (sd, sd_xform) = tex(m, &["spec_detail_map"]);
                s.albedo_tint = rgb(p(0));
                s.diffuse_intensity = p(4)[0];
                s.spec_color = [p(4)[1], p(4)[2], p(4)[3]];
                s.spec_intensity = p(5)[0]; s.rough_min = p(5)[1]; s.rough_max = p(5)[2]; s.spec_tint_by_albedo = p(5)[3];
                s.refl_tint = rgb(p(6)); s.refl_intensity = 1.0;   // the intensity is folded into up166.rgb
                s.refl_normal_blend = 1.0;
                s.env_lit_by_diffuse = 1.0;                        // by diffuse.r - see the shader lane
                if fam.ends_with("selfillum") { s.self_illum_color = rgb(p(7)); s.self_illum_intensity = p(7)[3]; s.self_illum_mode = 4; }
                s.char_cov = Some(CharCov {
                    face: p(1), mid: p(2), graze: p(3),
                    refl_saturation: p(6)[3],
                    spec_detail: if fam.ends_with("specdetail") { sd } else { None },
                    spec_detail_xform: sd_xform,
                });
            }
            // #h4-veh srf_char_constant: NO textures, NO samplers, no lighting at all - entry 01
            // writes `o0.rgb = up160.rgb` and entries 04/06 are byte-identical and emit
            // `ps_view_albedo * up160.w` through the self-illum exposure lerp. A pure emissive
            // constant (the Wraith's light strips / warning lights).
            "srf_char_constant" => {
                s.known = true;
                s.spec_model = SpecModel::None;
                s.albedo_tint = [0.0; 3];          // no lit term
                s.diffuse_intensity = 0.0;
                s.self_illum_color = rgb(p(0));
                s.self_illum_intensity = p(0)[3];
                s.self_illum_mode = 5;             // constant self-illum
            }
            "srf_lambert" | "srf_lambert_clip" | "srf_constant" => {
                s.known = true;
                s.spec_model = SpecModel::None;
                s.albedo_tint = rgb(p(0)); s.diffuse_intensity = p(0)[3];
                if fam == "srf_constant" { s.diffuse_intensity = 1.0; s.albedo_tint = [1.0; 3]; }
            }
            _ => {
                // Generic fallback: diffuse + normal map; a specular_map or control map, when
                // present, drives a Blinn lobe with neutral constants (PROVISIONAL).
                // Albedo tint / diffuse intensity = user_parameter_160 (p0.rgb / p0.w) for the
                // plain colour-map families (entry 01 / 06 asm of srf_ca_blinn_diffspec, srf_phong,
                // srf_foliage_clip: `color * cb13[0].xyz`, lit `* cb13[0].w`, the same slot as the
                // srf_blinn* families). Families whose p0 is NOT a tint
                // (the *_detail layouts, tintable, snow, char) keep the neutral constants.
                let p0_is_tint = (fam.starts_with("srf_blinn") || fam.starts_with("srf_ca_blinn") || fam.starts_with("srf_phong") || fam.starts_with("srf_foliage") || fam.starts_with("srf_lambert"))
                    && !fam.contains("detail") && !fam.contains("tintable") && !fam.contains("twotone") && !fam.contains("colorchange");
                if p0_is_tint {
                    let t = p(0);
                    if t[0] > 0.0 && t[1] > 0.0 && t[2] > 0.0 && t[3] > 0.0 { s.albedo_tint = rgb(t); s.diffuse_intensity = t[3]; }
                }
                if s.tex.control.is_some() { s.spec_model = SpecModel::Blinn; s.spec_src = SpecSource::ControlSpGlRf; s.spec_intensity = 1.0; s.rough_min = 0.01; s.rough_max = 1.0; }
                else if s.tex.specular.is_some() { s.spec_model = SpecModel::Blinn; s.spec_src = SpecSource::SpecularMap; s.spec_intensity = 1.0; s.rough_min = 0.01; s.rough_max = 1.0; }
            }
        }
        if !detail_on && !fam.contains("layered") { s.tex.normal_detail = None; }
        if s.tex.reflection_cube.is_none() { s.refl_intensity = 0.0; }
        if s.rough_min <= 0.0 { s.rough_min = 1e-5; }
        if s.rough_max <= 0.0 { s.rough_max = s.rough_min; }
        s
    }

    /// Shader lane code for `spec_src` (see the WGSL `h4_shade`).
    pub fn spec_src_code(&self) -> f32 {
        match self.spec_src {
            SpecSource::None => 0.0,
            SpecSource::SpecularMap => 1.0,
            SpecSource::ControlSpGlRf => 2.0,
            SpecSource::ControlSpGlSi => 3.0,
            SpecSource::DiffSpec => 4.0,
            SpecSource::SpecMapAndControl => 5.0,
            SpecSource::ControlSpGlRfSiAlpha => 6.0,
        }
    }
    /// `matmodel` lane: -(1 + model) is the renderer's Halo 4 sentinel (Reach passes >= 0).
    pub fn matmodel_code(&self) -> f32 {
        let model = match self.spec_model { SpecModel::None => 0.0, SpecModel::Blinn => 1.0, SpecModel::PhongPower => 2.0, SpecModel::PhongRough => 3.0 };
        -(1.0 + model)
    }
}

/// One-line description for diagnostics (HMS_H4_MATDIAG).
pub fn describe(s: &H4Shading) -> String {
    format!("{} {:?}/{:?} tint {:?} di {:.2} spec {:?}x{:.2} rough {:.3}..{:.3} tba {:.2} refl {:?}x{:.2} fres s{:.2} p{:.2} w{:.2} i{:.0} si {:?}x{:.1} m{} nd {:.0}..{:.0}{}{}",
        s.family, s.spec_model, s.spec_src, s.albedo_tint, s.diffuse_intensity, s.spec_color, s.spec_intensity, s.rough_min, s.rough_max,
        s.spec_tint_by_albedo, s.refl_tint, s.refl_intensity, s.fresnel_scale, s.fresnel_power, s.fresnel_weight, s.fresnel_invert,
        s.self_illum_color, s.self_illum_intensity, s.self_illum_mode, s.normal_detail_fade_start, s.normal_detail_fade_end,
        if s.snow.is_some() { " +snow" } else { "" },
        if s.layered.is_some() { " +layered3" } else if s.boundary.is_some() { " +boundary" } else { "" })
        + &s.char_cov.as_ref().map(|c| format!(" +charcov face {:?}^{:.2} mid {:?}^{:.2} graze {:?}^{:.2} refl-sat {:.2}{}",
            [c.face[0], c.face[1], c.face[2]], c.face[3], [c.mid[0], c.mid[1], c.mid[2]], c.mid[3],
            [c.graze[0], c.graze[1], c.graze[2]], c.graze[3], c.refl_saturation,
            if c.spec_detail.is_some() { " +specdetail" } else { "" })).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::{maps_dir, H4Cache};
    use crate::h4::materials::load_bsp_materials;

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Every BSP material of both MP maps decodes; the families read from the shaders cover
    /// >= 85 % of them; spot-check the constants of known materials against the dumped floats.
    #[test]
    fn mp_families_decode() {
        let mut total = 0usize;
        let mut known = 0usize;
        for map in ["wraparound.map", "ca_forge_ravine.map"] {
            let Some(c) = open(map) else { return };
            for sb in c.find_tags(b"sbsp") {
                for m in load_bsp_materials(&c, sb) {
                    if m.name == "<null>" { continue; }
                    let s = H4Shading::from_material(&m);
                    total += 1;
                    if s.known { known += 1; } else { eprintln!("unknown family: {} ({})", s.family, m.name); }
                    assert!(s.rough_min > 0.0 && s.rough_max > 0.0, "{}", m.name);
                    if m.name.ends_with("mp_forerunner_floortile_01_dark") {
                        // wraparound: up2 = (0.666, 0.694, 0.797, 4.5), up3 = (0.015, 0.1, 0, 0), up4.w = 0.6, up5 = (1, 0.2, 2, 0.75)
                        assert_eq!(s.spec_model, SpecModel::Blinn);
                        assert!((s.spec_intensity - 4.5).abs() < 1e-4 && (s.rough_min - 0.015).abs() < 1e-5 && (s.rough_max - 0.1).abs() < 1e-5, "{}", describe(&s));
                        assert!((s.refl_intensity - 0.6).abs() < 1e-4 && (s.fresnel_power - 2.0).abs() < 1e-4 && (s.fresnel_weight - 0.75).abs() < 1e-4, "{}", describe(&s));
                        assert!(s.tex.reflection_cube.is_some() && s.tex.control.is_some() && s.tex.normal.is_some());
                    }
                    if m.name.ends_with("ca_forge_ravine_cliffs") {
                        assert_eq!(s.spec_model, SpecModel::PhongPower);
                        let sn = s.snow.as_ref().expect("snow layer");
                        assert!((s.rough_min - 10.0).abs() < 1e-4 && (sn.coverage_power - 9.375).abs() < 1e-3, "{}", describe(&s));
                        assert!((sn.dir[2] - 1.0).abs() < 1e-4, "snow dir {:?}", sn.dir);
                        assert!(s.tex.color_detail.is_some() && s.tex.normal_detail.is_some() && s.tex.color1.is_some());
                        assert!((s.normal_detail_fade_end - 300.0).abs() < 1e-3 && (s.normal_detail_fade_start - 20.0).abs() < 1e-3);
                    }
                    if m.name.ends_with("ca_forge_ravine_floortile_01") {
                        assert_eq!(s.spec_src, SpecSource::DiffSpec);
                        assert!((s.diffspec[1] - 1.0).abs() < 1e-4 && (s.refl_intensity - 1.0).abs() < 1e-4 && (s.fresnel_power - 3.0).abs() < 1e-4, "{}", describe(&s));
                    }
                    if m.name.ends_with("mp_forerunner_emissive") {
                        assert_eq!(s.self_illum_mode, 2);
                        assert!((s.self_illum_intensity - 100.0).abs() < 1e-3, "{}", describe(&s));
                    }
                }
            }
        }
        eprintln!("families known {known}/{total}");
        assert!(known * 100 >= total * 85, "known {known}/{total}");
    }

    /// The Forge GRID piece's material (`fw_grid` -> `srf_ca_boundary_noscale`) decodes into the
    /// additive self-illum palette lane with the constants the `mat ` tag carries: up160
    /// (0.0055, 0.1573, 0.8122, 2.3), up161 (1024, 1, 2, 1), up162 all zero, up163 (2, 1.5, 0, 3),
    /// and all six texture roles resolved by their reflected parameter names.
    #[test]
    fn forge_grid_boundary_decodes() {
        let Some(c) = open("dlc_forge_island - Copy.map").or_else(|| open("ca_forge_ravine.map")) else { return };
        let Some(&mat) = c.find_tags(b"mat ").iter().find(|&&t| c.tag_name(t).ends_with("fw_grid")) else {
            eprintln!("skip: no fw_grid material in this cache");
            return;
        };
        let m = crate::h4::materials::load_material(&c, mat);
        assert_eq!(m.mats_name.rsplit('\\').next(), Some("srf_ca_boundary_noscale"));
        assert_eq!(m.kind, crate::h4::materials::H4MatKind::Additive);
        let s = H4Shading::from_material(&m);
        let b = s.boundary.as_ref().expect("boundary lane");
        assert!(b.overlay.is_some() && b.overlay_detail.is_some() && b.alpha_mask.is_some(), "{:?}", m.textures);
        assert!(b.noise_a.is_some() && b.noise_b.is_some() && b.palette.is_some(), "{:?}", m.textures);
        assert!((b.si_intensity - 1024.0).abs() < 1e-3 && (b.overlay_intensity - 2.3).abs() < 1e-4, "{b:?}");
        assert!((b.fres_power - 2.0).abs() < 1e-4 && (b.fres_invert - 1.0).abs() < 1e-4, "{b:?}");
        assert!((b.depth_fade_range - 2.0).abs() < 1e-4 && (b.palette_power - 1.5).abs() < 1e-4, "{b:?}");
        assert!((b.noise_diff_scale - 3.0).abs() < 1e-4 && b.height_fade == 0.0, "{b:?}");
        assert!(b.tint[2] > b.tint[1] && b.tint[1] > b.tint[0], "the grid's tint is blue: {:?}", b.tint);
        // the overlay map tiles 3x, the noise B map 3x, everything else 1x
        assert_eq!(b.overlay_xform[0], 3.0);
        assert_eq!(b.noise_b_xform[0], 3.0);
        assert_eq!(b.alpha_mask_xform, [1.0, 1.0, 0.0, 0.0]);
    }

    // #h4-veh
    /// The Covenant body family decodes into the `CharCov` ramp with the constants the `mat `
    /// tags carry: `storm_wraith_pld_03` (`srf_char_cov_specdetail`) up160 0.5114 grey, up161
    /// (0.0220, 0.0511, 0.1087, 5), up162 (0.0220, 0.0987, 0.1480, 3), up163 (0.0290, 0.0127,
    /// 0.0017, 2), up164 (1, 0.6456, 0.7299, 1), up165 (75, 0.005, 0.015, 0), up166 (0.6456,
    /// 0.7818, 1, 3), plus a `spec_detail_map` at 20x tiling. Every `srf_char_cov*` material in
    /// the cache must take the lane (else the vehicle draws raw white albedo again).
    #[test]
    fn char_cov_family_decodes() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let mut seen = 0usize;
        for mat in c.find_tags(b"mat ") {
            let m = crate::h4::materials::load_material(&c, mat);
            let fam = m.mats_name.rsplit('\\').next().unwrap_or("").to_ascii_lowercase();
            if !fam.starts_with("srf_char_cov") { continue; }
            let s = H4Shading::from_material(&m);
            seen += 1;
            assert!(s.known, "{fam} is not decoded");
            let cv = s.char_cov.as_ref().unwrap_or_else(|| panic!("{} ({fam}) has no char_cov lane", m.name));
            assert!(cv.face[3] > 0.0 && cv.mid[3] > 0.0 && cv.graze[3] > 0.0, "{} ramp exponents {:?}", m.name, (cv.face[3], cv.mid[3], cv.graze[3]));
            assert_eq!(s.spec_src, SpecSource::ControlSpGlRf, "{}", m.name);
            // only the `_specdetail` family multiplies the spec_detail_map into the specular
            // (the plain family's entry 01 writes o1.z = 0), so the others must not bind it
            if !fam.ends_with("specdetail") { assert!(cv.spec_detail.is_none(), "{} ({fam}) took a spec_detail_map", m.name); }
            if m.name.ends_with("storm_wraith_pld_03") {
                assert!((s.albedo_tint[0] - 0.51139784).abs() < 1e-5, "{}", describe(&s));
                assert!((cv.face[0] - 0.022012994).abs() < 1e-6 && (cv.face[3] - 5.0).abs() < 1e-4, "{cv:?}");
                assert!((cv.mid[2] - 0.14799805).abs() < 1e-6 && (cv.mid[3] - 3.0).abs() < 1e-4, "{cv:?}");
                assert!((cv.graze[0] - 0.02899119).abs() < 1e-6 && (cv.graze[3] - 2.0).abs() < 1e-4, "{cv:?}");
                assert!((s.diffuse_intensity - 1.0).abs() < 1e-4 && (s.spec_color[0] - 0.6455554).abs() < 1e-6 && (s.spec_color[2] - 1.0).abs() < 1e-6, "{}", describe(&s));
                assert!((s.spec_intensity - 75.0).abs() < 1e-3 && (s.rough_min - 0.005).abs() < 1e-6 && (s.rough_max - 0.015).abs() < 1e-6, "{}", describe(&s));
                assert!((s.refl_tint[2] - 1.0).abs() < 1e-5 && (cv.refl_saturation - 3.0).abs() < 1e-4, "{}", describe(&s));
                assert_eq!(cv.spec_detail_xform[0], 20.0, "spec_detail tiling");
                assert!(cv.spec_detail.is_some(), "the Wraith binds a spec_detail_map");
                // the face-on colour is BLUE and the grazing one RED-brown: the Covenant hull
                assert!(cv.face[2] > cv.face[0] && cv.graze[0] > cv.graze[2], "{cv:?}");
            }
        }
        assert!(seen >= 2, "only {seen} srf_char_cov* materials found");
    }

    // #h4-veh
    /// `srf_char_constant` (the Wraith's light strips) is a pure emissive constant: no lit term,
    /// self-illum mode 5, colour + intensity from up160.
    #[test]
    fn char_constant_is_pure_emissive() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let Some(&mat) = c.find_tags(b"mat ").iter().find(|&&t| {
            let m = crate::h4::materials::load_material(&c, t);
            m.mats_name.ends_with("srf_char_constant")
        }) else { eprintln!("skip: no srf_char_constant material"); return };
        let m = crate::h4::materials::load_material(&c, mat);
        let s = H4Shading::from_material(&m);
        assert!(s.known, "{}", describe(&s));
        assert_eq!(s.self_illum_mode, 5);
        assert_eq!(s.albedo_tint, [0.0; 3], "no lit term: {}", describe(&s));
        assert_eq!(s.diffuse_intensity, 0.0);
        assert_eq!(s.spec_model, SpecModel::None);
    }
}
