//! Forge "special FX" objects -> area_screen_effect (`sefc`) decode + ENGINE composition.
//!
//! Object -> effect chain (verified on forge_halo, docs/hrek_re/20_forge_screen_fx.md):
//!   obje +0x104 attachments block (0x20 B: tagref @0 [class @0, datum @0xC])
//!     -> effe +0x2C events block (0x40 B: parts block @0x1C)
//!       -> part (0x64 B: runtime group fourcc @0xC, type tagref @0x14 [datum @0x20]) of class 'sefc'
//!         -> sefc tag (16 B: word flags x2 + block{count @4, ptr @8}) of 240-B
//!            `s_single_screen_effect_definition` elements (reach_tag_test.exe field table 0x141e97100).
//!
//! Composition rule (reach_tag_test.exe screen_effect.cpp, accumulate = sub_1402A7BE0, disassembled):
//!   s = clamp(falloff, 0, 1)
//!   every scalar term (exposure boost/deboost, hue l/r, saturation, desaturation, contrast, gamma
//!   enhance/reduce, noise, floor rgb, tron, blur, vision, hud fade) = max(acc, s * def)
//!   colour filter rgb = min(acc, s * def + (1 - s))           (starts at 1,1,1)
//!   motion suck = max(acc, s * mag) with a summed direction.
//! The accumulator is reset per frame (sub_1402A98F0), the SCENARIO default sefc (scnr 'cfes' ref)
//! is folded in first with falloff 1 (sub_1402A84C0), then every live screen-effect instance.
//! => two identical effects do NOT stack (max(a, a) == a) and the map default composes by the
//! same per-term max. The tag doc says the same: "in the case of overlapping effects, the maximum
//! will be taken".
//!
//! Colour matrix (sub_1407E5390, disassembled; applied by final_composite AFTER pow(color, gamma.z)):
//!   hue    = fmod(hueRight - hueLeft + ctl.hue, 360)  (+360 if negative)      -> Haeberli hue rotation
//!   S      = clamp(ctl.sat(=1) + saturation - desaturation, -10, 10)          -> lerp(luma709, c, S)
//!   C      = 1 + 4 * contrast^2 ; FC = diag(sqrt(filter) * C), translation = (-floor - contrast^2)
//!            (asm: `mulss xmm14,xmm14` squares the contrast once; both the diagonal and the
//!            translation use the SQUARE -- so the contrast pivots about 0.25 in gamma-2 space)
//!   M      = Sat * FC * Hue   (row-vector convention, out = saturate(v * M))
//!   gamma  = clamp(1 + gammaEnhance - gammaReduce, 0.01, 10) ; shader exponent gamma.z = 0.5 * gamma
//!   exposure boost - deboost = stops added to the exposure (linear gain 2^stops before the sqrt).
//! Bright/dark noise (noise_params) are a screen-space noise texture the engine composites last;
//! not ported (HMS has no noise pass).

use crate::scene::SceneController;

/// One decoded `s_single_screen_effect_definition` (240 B).
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenFxDef {
    /// sefc tag id (0xFFFF-masked datum) + element index within the tag block.
    pub tag: u32,
    pub index: u32,
    /// Tag path of the sefc (lowercased), for the status line.
    pub name: String,
    pub name_sid: u32,
    /// area_screen_effect_flags: b0 debug disable, b1 allow effect outside radius, b2 unattached,
    /// b3 first person only, b4 third person only, b5 disable camera falloffs (distance + angle),
    /// b6 only affects attached object.
    pub flags: u16,
    /// hidden flags (b1 = runtime "valid" gate the engine tests before applying the element).
    pub hidden_flags: u16,
    pub max_distance: f32,
    pub delay: f32,
    pub lifetime: f32,
    /// The four falloff functions (distance @0xC, time @0x28, angle @0x3C, object @0x58) when
    /// they are CONSTANT mapping functions (type 1): the constant. None = non-constant curve.
    pub falloff_const: [Option<f32>; 4],
    pub reals: ScreenFxReals,
    /// +0xE0 "shader effect" tagref datum (0xFFFFFFFF = none).
    pub shader_effect: u32,
}

/// The real-valued terms the engine accumulates (+0x6C..+0xDC of the element).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ScreenFxReals {
    pub exposure_boost: f32,
    pub exposure_deboost: f32,
    pub hue_left: f32,
    pub hue_right: f32,
    pub saturation: f32,
    pub desaturation: f32,
    pub contrast_enhance: f32,
    pub gamma_enhance: f32,
    pub gamma_reduce: f32,
    pub bright_noise: f32,
    pub dark_noise: f32,
    pub color_filter: [f32; 3],
    pub color_floor: [f32; 3],
    pub tron: f32,
    pub motion_suck: f32,
    pub motion_dir: [f32; 3],
    pub hblur: f32,
    pub vblur: f32,
    pub vision_mode: f32,
    pub hud_fade: f32,
    pub fov_in: f32,
    pub fov_out: f32,
    pub screen_shake: f32,
}

/// The engine's per-frame accumulator (sub_1402A98F0 layout, 0x68 B). Only the terms the post
/// pass can honour are consumed; the rest are kept for the diagnostics / status line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenFxAccum {
    pub exposure_boost: f32,
    pub exposure_deboost: f32,
    pub hue_left: f32,
    pub hue_right: f32,
    pub saturation: f32,
    pub desaturation: f32,
    pub contrast_enhance: f32,
    pub gamma_enhance: f32,
    pub gamma_reduce: f32,
    pub bright_noise: f32,
    pub dark_noise: f32,
    pub color_filter: [f32; 3],
    pub color_floor: [f32; 3],
    pub tron: f32,
    pub motion_suck: f32,
    pub motion_dir: [f32; 3],
    pub hblur: f32,
    pub vblur: f32,
    pub vision_mode: f32,
    pub hud_fade: f32,
}

impl Default for ScreenFxAccum {
    fn default() -> Self { Self::identity() }
}

impl ScreenFxAccum {
    /// The reset state: everything 0 except the colour filter (1,1,1).
    pub fn identity() -> Self {
        Self {
            exposure_boost: 0.0, exposure_deboost: 0.0, hue_left: 0.0, hue_right: 0.0,
            saturation: 0.0, desaturation: 0.0, contrast_enhance: 0.0, gamma_enhance: 0.0,
            gamma_reduce: 0.0, bright_noise: 0.0, dark_noise: 0.0,
            color_filter: [1.0; 3], color_floor: [0.0; 3], tron: 0.0, motion_suck: 0.0,
            motion_dir: [0.0; 3], hblur: 0.0, vblur: 0.0, vision_mode: 0.0, hud_fade: 0.0,
        }
    }

    /// True when nothing has been accumulated (the post pass can skip the matrix).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_identity(&self) -> bool { *self == Self::identity() }

    /// Engine `accumulate` (sub_1402A7BE0): per-term max, filter = min(acc, s*f + (1-s)).
    pub fn accumulate(&mut self, r: &ScreenFxReals, falloff: f32) {
        let s = if falloff.is_finite() { falloff.clamp(0.0, 1.0) } else { 0.0 };
        let fin = |v: f32| if v.is_finite() { v } else { 0.0 };
        let mx = |acc: &mut f32, v: f32| { *acc = acc.max(s * fin(v)); };
        mx(&mut self.exposure_boost, r.exposure_boost);
        mx(&mut self.exposure_deboost, r.exposure_deboost);
        mx(&mut self.hue_left, r.hue_left);
        mx(&mut self.hue_right, r.hue_right);
        mx(&mut self.saturation, r.saturation);
        mx(&mut self.desaturation, r.desaturation);
        mx(&mut self.contrast_enhance, r.contrast_enhance);
        mx(&mut self.gamma_enhance, r.gamma_enhance);
        mx(&mut self.gamma_reduce, r.gamma_reduce);
        mx(&mut self.bright_noise, r.bright_noise);
        mx(&mut self.dark_noise, r.dark_noise);
        for i in 0..3 {
            let f = s * fin(r.color_filter[i]) + (1.0 - s);
            self.color_filter[i] = self.color_filter[i].min(f);
            mx(&mut self.color_floor[i], r.color_floor[i]);
        }
        mx(&mut self.tron, r.tron);
        // motion suck: magnitude max, direction summed (normalised first when non-degenerate).
        let mag = s * fin(r.motion_suck);
        let mut d = [fin(r.motion_dir[0]), fin(r.motion_dir[1]), fin(r.motion_dir[2])];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len >= 1e-4 { for v in &mut d { *v /= len; } }
        for i in 0..3 { self.motion_dir[i] += d[i] * mag; }
        self.motion_suck = self.motion_suck.max(mag);
        mx(&mut self.hblur, r.hblur);
        mx(&mut self.vblur, r.vblur);
        mx(&mut self.vision_mode, r.vision_mode);
        mx(&mut self.hud_fade, r.hud_fade);
    }
}

/// What the post pass needs: exposure gain (linear, before the gamma-2 sqrt), the gamma exponent
/// (0.5 = plain sqrt) and the 4x3 colour matrix as three float4 COLUMNS (out[j] = dot(v, col[j]),
/// v = (r, g, b, 1)) -- the same layout the engine uploads to k_ps_color_matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenFxRender {
    pub gain: f32,
    pub gamma_exp: f32,
    pub cols: [[f32; 4]; 3],
    /// (dark noise, bright noise) -- reported only, not rendered.
    pub noise: [f32; 2],
}

impl Default for ScreenFxRender {
    fn default() -> Self { Self::identity() }
}

impl ScreenFxRender {
    pub fn identity() -> Self {
        Self { gain: 1.0, gamma_exp: 0.5, cols: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], noise: [0.0; 2] }
    }
    /// Apply to a gamma-2 (post-sqrt) colour, exactly like the shader's saturate(mul(v, M)).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        let v = [c[0], c[1], c[2], 1.0];
        let mut o = [0.0f32; 3];
        for j in 0..3 {
            o[j] = (v[0] * self.cols[j][0] + v[1] * self.cols[j][1] + v[2] * self.cols[j][2] + v[3] * self.cols[j][3]).clamp(0.0, 1.0);
        }
        o
    }
}

// ---- 4x4 row-major matrices, row-vector convention (v' = v * M) ------------------------------

pub type Mat4 = [[f32; 4]; 4];

const MAT4_IDENTITY: Mat4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// out[i] = sum_k a[i][k] * b[k]  (engine sub_1407E4CD0: A rows times B).
fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut o = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    o
}

/// Rec.709 luma weights the engine uses for both the saturation lerp and the hue shear.
const LUMA_R: f32 = 0.212656;
const LUMA_G: f32 = 0.715158;
const LUMA_B: f32 = 0.0721856;

fn xrotate(m: &Mat4, rs: f32, rc: f32) -> Mat4 {
    let mut r = MAT4_IDENTITY;
    r[1][1] = rc; r[1][2] = rs; r[2][1] = -rs; r[2][2] = rc;
    mat4_mul(m, &r)
}
fn yrotate(m: &Mat4, rs: f32, rc: f32) -> Mat4 {
    let mut r = MAT4_IDENTITY;
    r[0][0] = rc; r[0][2] = -rs; r[2][0] = rs; r[2][2] = rc;
    mat4_mul(m, &r)
}
fn zrotate(m: &Mat4, rs: f32, rc: f32) -> Mat4 {
    let mut r = MAT4_IDENTITY;
    r[0][0] = rc; r[0][1] = rs; r[1][0] = -rs; r[1][1] = rc;
    mat4_mul(m, &r)
}
fn zshear(m: &Mat4, dx: f32, dy: f32) -> Mat4 {
    let mut r = MAT4_IDENTITY;
    r[0][2] = dx; r[1][2] = dy;
    mat4_mul(m, &r)
}

/// Engine hue rotation (sub_1407E4E20) = Haeberli's luminance-preserving rotation about the grey
/// axis: rotate (1,1,1) onto +z (x by 45 deg, y by atan(1/sqrt2)), shear so luma stays in the
/// z=const plane, rotate about z by `deg`, un-shear, un-rotate.
fn hue_matrix(deg: f32) -> Mat4 {
    let mut m = MAT4_IDENTITY;
    let xrs = 1.0 / 2.0f32.sqrt();
    let xrc = xrs;
    m = xrotate(&m, xrs, xrc);
    let yrs = -1.0 / 3.0f32.sqrt();
    let yrc = 2.0f32.sqrt() / 3.0f32.sqrt();
    m = yrotate(&m, yrs, yrc);
    // transform the luma vector (row vector * M, with the translation row)
    let lx = LUMA_R * m[0][0] + LUMA_G * m[1][0] + LUMA_B * m[2][0] + m[3][0];
    let ly = LUMA_R * m[0][1] + LUMA_G * m[1][1] + LUMA_B * m[2][1] + m[3][1];
    let lz = LUMA_R * m[0][2] + LUMA_G * m[1][2] + LUMA_B * m[2][2] + m[3][2];
    let zsx = lx / lz;
    let zsy = ly / lz;
    m = zshear(&m, zsx, zsy);
    let rad = deg.to_radians();
    m = zrotate(&m, rad.sin(), rad.cos());
    m = zshear(&m, -zsx, -zsy);
    m = yrotate(&m, -yrs, yrc);
    m = xrotate(&m, -xrs, xrc);
    m
}

/// Engine saturation matrix: out = (1 - s) * luma709(c) + s * c.
fn saturation_matrix(s: f32) -> Mat4 {
    let a = 1.0 - s;
    [
        [a * LUMA_R + s, a * LUMA_R, a * LUMA_R, 0.0],
        [a * LUMA_G, a * LUMA_G + s, a * LUMA_G, 0.0],
        [a * LUMA_B, a * LUMA_B, a * LUMA_B + s, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Engine filter / contrast / floor matrix: diag(sqrt(filter) * (1 + 4c^2)), translation
/// (-floor - c^2). The sqrt is because the matrix runs in gamma-2 space (a linear multiply by
/// `filter` is a multiply by sqrt(filter) after the sqrt).
fn filter_contrast_floor_matrix(filter: [f32; 3], contrast: f32, floor: [f32; 3]) -> Mat4 {
    let c2 = contrast * contrast;
    let k = 1.0 + 4.0 * c2;
    let d = |f: f32| f.max(0.0).sqrt() * k;
    [
        [d(filter[0]), 0.0, 0.0, 0.0],
        [0.0, d(filter[1]), 0.0, 0.0],
        [0.0, 0.0, d(filter[2]), 0.0],
        // (the engine builds this row with w = 0; only the three rgb columns are ever uploaded, so
        // w = 1 here keeps the 4x4 a proper affine matrix without changing the result)
        [-floor[0] - c2, -floor[1] - c2, -floor[2] - c2, 1.0],
    ]
}

/// Engine colour-matrix construction (sub_1407E5390) with the "hue saturation control" globals at
/// their defaults (hue 0, saturation 1, rgb scale 1). Returns the full 4x4 (row-vector).
pub fn color_matrix(acc: &ScreenFxAccum) -> Mat4 {
    let mut hue = (acc.hue_right - acc.hue_left) % 360.0;
    if hue < 0.0 { hue += 360.0; }
    let s = (1.0 + acc.saturation - acc.desaturation).clamp(-10.0, 10.0);
    let contrast = acc.contrast_enhance;
    let filter = [acc.color_filter[0].clamp(0.0, 1.0), acc.color_filter[1].clamp(0.0, 1.0), acc.color_filter[2].clamp(0.0, 1.0)];
    let sat = saturation_matrix(s);
    let fc = filter_contrast_floor_matrix(filter, contrast, acc.color_floor);
    let h = hue_matrix(hue);
    mat4_mul(&mat4_mul(&sat, &fc), &h)
}

/// Engine gamma exponent: pow(color, 0.5 * clamp(1 + enhance - reduce, 0.01, 10)).
fn gamma_exponent(acc: &ScreenFxAccum) -> f32 {
    0.5 * (1.0 + acc.gamma_enhance - acc.gamma_reduce).clamp(0.01, 10.0)
}

/// Compose an accumulator into post-pass parameters.
pub fn compose(acc: &ScreenFxAccum) -> ScreenFxRender {
    let m = color_matrix(acc);
    let mut cols = [[0.0f32; 4]; 3];
    for j in 0..3 {
        for i in 0..4 { cols[j][i] = m[i][j]; }
    }
    let stops = (acc.exposure_boost - acc.exposure_deboost).clamp(-8.0, 8.0);
    ScreenFxRender { gain: stops.exp2(), gamma_exp: gamma_exponent(acc), cols, noise: [acc.dark_noise, acc.bright_noise] }
}

// ---- cache decode ----------------------------------------------------------------------------

fn rd32(b: &[u8], o: usize) -> u32 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) }
fn rd16(b: &[u8], o: usize) -> u16 { u16::from_le_bytes([b[o], b[o + 1]]) }
fn rdf(b: &[u8], o: usize) -> f32 { let v = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]); if v.is_finite() { v } else { 0.0 } }

/// Decode a mapping-function blob (20-B data ref at `off`: size @0, ptr @0xC) when it is a
/// CONSTANT function (type 1, kind 0: constant = range lo). Else None.
fn falloff_const(scene: &SceneController, e: &[u8], off: usize) -> Option<f32> {
    let size = rd32(e, off);
    let ptr = rd32(e, off + 0xC);
    if size < 8 || ptr == 0 { return None; }
    let b = scene.raw_tag_ptr(ptr, size.min(0x40))?;
    if b.len() < 8 { return None; }
    if b[0] == 1 && b[2] == 0 { Some(rdf(&b, 4)) } else { None }
}

/// Decode every element of a `sefc` tag. Empty when the tag is unreadable.
pub fn decode_sefc(scene: &SceneController, tag: u32) -> Vec<ScreenFxDef> {
    let mut out = Vec::new();
    let Some(m) = scene.raw_tag_meta(tag, 0, 0x10) else { return out };
    if m.len() < 0x10 { return out; }
    let count = rd32(&m, 4).min(16);
    let ptr = rd32(&m, 8);
    if count == 0 || ptr == 0 { return out; }
    let Some(blob) = scene.raw_tag_ptr(ptr, count * 0xF0) else { return out };
    let name = scene.tag_name_of(tag);
    for i in 0..count as usize {
        let e = &blob[i * 0xF0..];
        if e.len() < 0xF0 { break; }
        let f = |o: usize| rdf(e, o);
        let reals = ScreenFxReals {
            exposure_boost: f(0x6C), exposure_deboost: f(0x70), hue_left: f(0x74), hue_right: f(0x78),
            saturation: f(0x7C), desaturation: f(0x80), contrast_enhance: f(0x84), gamma_enhance: f(0x88),
            gamma_reduce: f(0x8C), bright_noise: f(0x90), dark_noise: f(0x94),
            color_filter: [f(0x98), f(0x9C), f(0xA0)], color_floor: [f(0xA4), f(0xA8), f(0xAC)],
            tron: f(0xB0), motion_suck: f(0xB4), motion_dir: [f(0xB8), f(0xBC), f(0xC0)],
            hblur: f(0xC4), vblur: f(0xC8), vision_mode: f(0xCC), hud_fade: f(0xD0),
            fov_in: f(0xD4), fov_out: f(0xD8), screen_shake: f(0xDC),
        };
        out.push(ScreenFxDef {
            tag, index: i as u32, name: name.clone(),
            name_sid: rd32(e, 0), flags: rd16(e, 4), hidden_flags: rd16(e, 6),
            max_distance: f(8), delay: f(0x20), lifetime: f(0x24),
            falloff_const: [falloff_const(scene, e, 0xC), falloff_const(scene, e, 0x28), falloff_const(scene, e, 0x3C), falloff_const(scene, e, 0x58)],
            reals,
            shader_effect: rd32(e, 0xE0 + 0xC),
        });
    }
    out
}

// Tag-ref class codes are stored big-endian ('sefc' reads as the bytes "cfes" in a hex dump).
const FOURCC_EFFE: u32 = u32::from_be_bytes(*b"effe");
const FOURCC_SEFC: u32 = u32::from_be_bytes(*b"sefc");

/// The `sefc` tags an object spawns through its attachments (obje +0x104 -> effe -> events ->
/// parts of class 'sefc'). Deduplicated, in discovery order. Empty for ordinary objects.
fn object_screen_fx_tags(scene: &SceneController, obj_tag: u32) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let Some(hdr) = scene.raw_tag_meta(obj_tag, 0x104, 8) else { return out };
    if hdr.len() < 8 { return out; }
    let (cnt, ptr) = (rd32(&hdr, 0), rd32(&hdr, 4));
    if cnt == 0 || cnt > 32 || ptr == 0 { return out; }
    let Some(att) = scene.raw_tag_ptr(ptr, cnt * 0x20) else { return out };
    for i in 0..cnt as usize {
        let e = &att[i * 0x20..];
        if e.len() < 0x20 { break; }
        let cls = rd32(e, 0);
        let datum = rd32(e, 0xC);
        if datum == 0xFFFF_FFFF || datum == 0 { continue; }
        let tag = datum & 0xFFFF;
        if cls == FOURCC_SEFC {
            if !out.contains(&tag) { out.push(tag); }
        } else if cls == FOURCC_EFFE {
            for t in effect_screen_fx_tags(scene, tag) {
                if !out.contains(&t) { out.push(t); }
            }
        }
    }
    out
}

/// The `sefc` parts of an `effe` (events @0x2C, 0x40 B; parts @event+0x1C, 0x64 B; type tagref @0x14).
fn effect_screen_fx_tags(scene: &SceneController, effe_tag: u32) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let Some(hdr) = scene.raw_tag_meta(effe_tag, 0x2C, 8) else { return out };
    if hdr.len() < 8 { return out; }
    let (ecnt, eptr) = (rd32(&hdr, 0), rd32(&hdr, 4));
    if ecnt == 0 || ecnt > 64 || eptr == 0 { return out; }
    let Some(ev) = scene.raw_tag_ptr(eptr, ecnt * 0x40) else { return out };
    for i in 0..ecnt as usize {
        let e = &ev[i * 0x40..];
        if e.len() < 0x40 { break; }
        let (pcnt, pptr) = (rd32(e, 0x1C), rd32(e, 0x20));
        if pcnt == 0 || pcnt > 64 || pptr == 0 { continue; }
        let Some(parts) = scene.raw_tag_ptr(pptr, pcnt * 0x64) else { continue };
        for p in 0..pcnt as usize {
            let pe = &parts[p * 0x64..];
            if pe.len() < 0x64 { break; }
            let cls = rd32(pe, 0x14);
            let datum = rd32(pe, 0x20);
            if cls != FOURCC_SEFC || datum == 0xFFFF_FFFF || datum == 0 { continue; }
            let tag = datum & 0xFFFF;
            if scene.raw_tag_class(tag).as_deref().map_or(true, |c| c == "sefc") && !out.contains(&tag) {
                out.push(tag);
            }
        }
    }
    out
}

/// Runtime falloff for a placed Forge FX object's element. The orbs author "disable camera
/// falloffs" (b5) + constant-1 curves, so they are map-wide. HMS evaluates no per-camera distance
/// or angle curve, so an element WITHOUT b5 (or b1 "allow effect outside radius") is treated as a
/// proximity effect the camera is outside of (falloff 0) -- e.g. the Kill Ball's 2.9-wu red flash.
/// Elements flagged debug-disabled (b0) or hidden-invalid contribute nothing.
fn element_falloff(def: &ScreenFxDef) -> f32 {
    if def.flags & 0x1 != 0 { return 0.0; }
    if def.hidden_flags & 0x2 == 0 { return 0.0; }
    if def.flags & 0x22 == 0 { return 0.0; }
    let mut s = 1.0;
    // The engine multiplies the evaluated object-falloff curve in whenever the effect is attached
    // to an object, and the time-falloff curve when the element has a finite lifetime. Only their
    // CONSTANT forms are honoured here (the orbs author constant 1.0 for all four curves).
    if let Some(c) = def.falloff_const[3] { s *= c.clamp(0.0, 1.0); }
    if def.lifetime > 1e-4 {
        if let Some(c) = def.falloff_const[1] { s *= c.clamp(0.0, 1.0); }
    }
    s
}

/// Per-map cache: obje tag -> its sefc tags, sefc tag -> decoded elements.
#[derive(Default)]
pub struct ScreenFxCache {
    obj_to_sefc: std::collections::HashMap<u32, Vec<u32>>,
    defs: std::collections::HashMap<u32, Vec<ScreenFxDef>>,
}

impl ScreenFxCache {
    pub fn clear(&mut self) { self.obj_to_sefc.clear(); self.defs.clear(); }

    pub fn sefc_for_object(&mut self, scene: &SceneController, obj_tag: u32) -> &[u32] {
        if obj_tag == 0 || obj_tag == 0xFFFF_FFFF {
            return &[];
        }
        self.obj_to_sefc.entry(obj_tag).or_insert_with(|| object_screen_fx_tags(scene, obj_tag)).as_slice()
    }

    pub fn defs(&mut self, scene: &SceneController, sefc_tag: u32) -> &[ScreenFxDef] {
        self.defs.entry(sefc_tag).or_insert_with(|| decode_sefc(scene, sefc_tag)).as_slice()
    }
}

/// The active screen-effect set for one frame: the DISTINCT sefc tags spawned by the placed
/// objects (dedupe = the engine's per-term max makes duplicates a no-op) plus the scenario default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActiveScreenFx {
    /// Scenario default sefc tag (0 = none).
    pub default_tag: u32,
    /// Distinct Forge FX sefc tags, sorted (order does not matter to a max-accumulate).
    pub forge_tags: Vec<u32>,
    /// Display names of the Forge FX (last path component of the sefc tag path), matching forge_tags.
    pub forge_names: Vec<String>,
    /// The composed accumulator (default + forge effects).
    pub accum: ScreenFxAccum,
    /// Post-pass parameters for `accum`.
    pub render: ScreenFxRender,
    /// Post-pass parameters for the scenario default ALONE (what "Forge special FX off" shows).
    pub render_default_only: ScreenFxRender,
}

/// Datum prefix of the base map's own (scnr-placed) scenery/machines. Their sefc attachments are
/// script / event / proximity gated in the engine (Zealot's corvette-power flash, Reflection's
/// globe) and are NOT permanently on, so only Forge-placed objects (variant / local / live) count.
const SCENARIO_DATUM_BASE: u32 = 0xE000_0000;
fn is_scenario_datum(d: u32) -> bool { (d & 0xF000_0000) == SCENARIO_DATUM_BASE }

/// Build the active set from the (datum, object tag) pairs present in the scene.
pub fn active_screen_fx(scene: &SceneController, cache: &mut ScreenFxCache, default_tag: u32, objects: impl Iterator<Item = (u32, u32)>) -> ActiveScreenFx {
    let mut forge_tags: Vec<u32> = Vec::new();
    for (datum, t) in objects {
        if is_scenario_datum(datum) { continue; }
        for &s in cache.sefc_for_object(scene, t) {
            if !forge_tags.contains(&s) { forge_tags.push(s); }
        }
    }
    forge_tags.sort_unstable();
    let mut acc = ScreenFxAccum::identity();
    if default_tag != 0 {
        // The scenario default is folded in at falloff 1 with NO curve / distance evaluation
        // (engine sub_1402A84C0; the tag doc: "not used for scenario global effects").
        for d in cache.defs(scene, default_tag).to_vec() {
            let s = if d.flags & 0x1 != 0 || d.hidden_flags & 0x2 == 0 { 0.0 } else { 1.0 };
            acc.accumulate(&d.reals, s);
        }
    }
    let render_default_only = if default_tag != 0 { compose(&acc) } else { ScreenFxRender::identity() };
    let mut forge_names = Vec::new();
    for &s in &forge_tags {
        let defs = cache.defs(scene, s).to_vec();
        let name = defs.first().map(|d| short_name(&d.name)).unwrap_or_else(|| format!("{s:#x}"));
        forge_names.push(name);
        for d in defs {
            acc.accumulate(&d.reals, element_falloff(&d));
        }
    }
    let render = if default_tag != 0 || !forge_tags.is_empty() { compose(&acc) } else { ScreenFxRender::identity() };
    ActiveScreenFx { default_tag, forge_tags, forge_names, accum: acc, render, render_default_only }
}

/// Upload `active` to the post pass. `forge_enabled` false = the scenario default alone (the
/// user's "Forge special FX" toggle / `screenfx off` / HMS_FORGE_FX=0). Returns a log line.
pub fn push_to_renderer(renderer: &hms_render::SceneRenderer, queue: &eframe::wgpu::Queue, active: &ActiveScreenFx, forge_enabled: bool) -> String {
    let r = if forge_enabled { active.render } else { active.render_default_only };
    renderer.set_screen_fx(queue, r.gain, r.gamma_exp, r.cols);
    describe_active(active, forge_enabled)
}

/// One-line summary of the active set for logs / the status line.
fn describe_active(active: &ActiveScreenFx, forge_enabled: bool) -> String {
    let r = if forge_enabled { &active.render } else { &active.render_default_only };
    let a = &active.accum;
    format!(
        "screen fx: default sefc {:#x}, forge fx [{}]{} | gain={:.3} gamma_exp={:.3} | composed: hue={:.1} sat={:.2}/{:.2} contrast={:.2} gamma+={:.2}/-={:.2} filter=[{:.3},{:.3},{:.3}] floor=[{:.3},{:.3},{:.3}] noise b/d={:.3}/{:.3} (noise/tron/blur/vision not rendered)",
        active.default_tag, active.forge_names.join(" + "), if forge_enabled { "" } else { " (DISABLED)" },
        r.gain, r.gamma_exp, a.hue_right - a.hue_left, a.saturation, a.desaturation, a.contrast_enhance, a.gamma_enhance, a.gamma_reduce,
        a.color_filter[0], a.color_filter[1], a.color_filter[2], a.color_floor[0], a.color_floor[1], a.color_floor[2], a.bright_noise, a.dark_noise,
    )
}

/// Headless / script override: HMS_FORGE_FX=0|off|false disables the placed Forge FX (the map
/// default still applies), anything else (or unset) = on.
pub fn env_forge_fx_enabled() -> bool {
    match std::env::var("HMS_FORGE_FX") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no"),
        Err(_) => true,
    }
}

const SETTING_FILE: &str = "forge_fx.txt";

/// The persisted "Forge special FX" toggle (settings dir, default ON when absent/unreadable).
pub fn load_enabled_setting() -> bool {
    let Some(dir) = crate::project::settings_dir() else { return true };
    match std::fs::read_to_string(dir.join(SETTING_FILE)) {
        Ok(s) => !matches!(s.trim(), "0" | "off" | "false"),
        Err(_) => true,
    }
}

pub fn save_enabled_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(SETTING_FILE), if on { "1" } else { "0" });
    }
}

/// "objects\levels\shared\screen_effects\colorblind" -> "colorblind".
pub fn short_name(path: &str) -> String {
    path.rsplit(['\\', '/']).next().unwrap_or(path).to_string()
}

/// One-line diagnostic for a decoded element.
pub fn describe(d: &ScreenFxDef) -> String {
    let r = &d.reals;
    format!(
        "sefc {:#x}[{}] '{}' flags={:#06x} hidden={:#06x} maxdist={:.1} delay={:.2} life={:.2} falloff_const={:?} | boost={:.3} deboost={:.3} hueL={:.1} hueR={:.1} sat={:.3} desat={:.3} contrast={:.3} gamma+={:.3} gamma-={:.3} noise b/d={:.3}/{:.3} filter=[{:.3},{:.3},{:.3}] floor=[{:.3},{:.3},{:.3}] tron={:.2} suck={:.2} blur={:.2}/{:.2} vision={:.2} hud={:.2} fov={:.2}/{:.2} shake={:.2} shader={:#x}",
        d.tag, d.index, d.name, d.flags, d.hidden_flags, d.max_distance, d.delay, d.lifetime, d.falloff_const,
        r.exposure_boost, r.exposure_deboost, r.hue_left, r.hue_right, r.saturation, r.desaturation, r.contrast_enhance,
        r.gamma_enhance, r.gamma_reduce, r.bright_noise, r.dark_noise, r.color_filter[0], r.color_filter[1], r.color_filter[2],
        r.color_floor[0], r.color_floor[1], r.color_floor[2], r.tron, r.motion_suck, r.hblur, r.vblur, r.vision_mode, r.hud_fade,
        r.fov_in, r.fov_out, r.screen_shake, d.shader_effect,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-4 }
    fn mat_approx(a: &Mat4, b: &Mat4) -> bool { a.iter().zip(b.iter()).all(|(x, y)| x.iter().zip(y.iter()).all(|(p, q)| approx(*p, *q))) }
    fn reals(f: impl Fn(&mut ScreenFxReals)) -> ScreenFxReals { let mut r = ScreenFxReals { color_filter: [1.0; 3], ..Default::default() }; f(&mut r); r }

    #[test]
    fn identity_accumulator_is_identity_render() {
        let acc = ScreenFxAccum::identity();
        assert!(acc.is_identity());
        let r = compose(&acc);
        assert!(approx(r.gain, 1.0) && approx(r.gamma_exp, 0.5));
        assert!(mat_approx(&color_matrix(&acc), &MAT4_IDENTITY));
        let o = r.apply([0.2, 0.5, 0.9]);
        assert!(approx(o[0], 0.2) && approx(o[1], 0.5) && approx(o[2], 0.9), "{o:?}");
    }

    #[test]
    fn hue_zero_is_identity_and_rotation_preserves_luma_and_grey() {
        assert!(mat_approx(&hue_matrix(0.0), &MAT4_IDENTITY));
        for deg in [30.0f32, 120.0, 200.0, 359.0] {
            let m = hue_matrix(deg);
            // grey axis fixed
            let g = [0.5f32, 0.5, 0.5, 1.0];
            for j in 0..3 {
                let v = g[0] * m[0][j] + g[1] * m[1][j] + g[2] * m[2][j] + g[3] * m[3][j];
                assert!(approx(v, 0.5), "grey not fixed at {deg}: {v}");
            }
            // luma preserved for an arbitrary colour
            let c = [0.8f32, 0.3, 0.1];
            let mut o = [0.0f32; 3];
            for j in 0..3 { o[j] = c[0] * m[0][j] + c[1] * m[1][j] + c[2] * m[2][j] + m[3][j]; }
            let l0 = LUMA_R * c[0] + LUMA_G * c[1] + LUMA_B * c[2];
            let l1 = LUMA_R * o[0] + LUMA_G * o[1] + LUMA_B * o[2];
            assert!(approx(l0, l1), "luma changed at {deg}: {l0} vs {l1}");
        }
        // 360 == 0
        assert!(mat_approx(&hue_matrix(360.0), &MAT4_IDENTITY));
    }

    #[test]
    fn saturation_matrix_extremes() {
        let m0 = saturation_matrix(0.0);
        let c = [0.8f32, 0.3, 0.1];
        let l = LUMA_R * c[0] + LUMA_G * c[1] + LUMA_B * c[2];
        for j in 0..3 {
            let v = c[0] * m0[0][j] + c[1] * m0[1][j] + c[2] * m0[2][j];
            assert!(approx(v, l));
        }
        assert!(mat_approx(&saturation_matrix(1.0), &MAT4_IDENTITY));
    }

    #[test]
    fn filter_is_sqrt_in_gamma2_space_and_floor_subtracts() {
        let mut acc = ScreenFxAccum::identity();
        acc.color_filter = [0.25, 1.0, 0.64];
        acc.color_floor = [0.0, 0.1, 0.0];
        let r = compose(&acc);
        let o = r.apply([1.0, 0.5, 1.0]);
        assert!(approx(o[0], 0.5) && approx(o[1], 0.4) && approx(o[2], 0.8), "{o:?}");
    }

    #[test]
    fn contrast_and_gamma_terms() {
        let mut acc = ScreenFxAccum::identity();
        acc.contrast_enhance = 0.5; // k = 1 + 4*0.25 = 2, translation -0.25 (pivot 0.25)
        let r = compose(&acc);
        let o = r.apply([0.5, 0.75, 0.25]);
        assert!(approx(o[0], 0.75) && approx(o[1], 1.0) && approx(o[2], 0.25), "{o:?}");
        acc.gamma_enhance = 0.4;
        acc.gamma_reduce = 0.1;
        assert!(approx(gamma_exponent(&acc), 0.5 * 1.3));
        acc.gamma_reduce = 40.0;
        assert!(approx(gamma_exponent(&acc), 0.5 * 0.01));
    }

    #[test]
    fn accumulate_is_per_term_max_and_identical_effects_do_not_stack() {
        let a = reals(|r| { r.exposure_boost = 0.3; r.desaturation = 0.5; r.color_filter = [0.5, 0.8, 1.0]; r.color_floor = [0.1, 0.0, 0.0]; });
        let mut once = ScreenFxAccum::identity();
        once.accumulate(&a, 1.0);
        let mut twice = once;
        twice.accumulate(&a, 1.0);
        assert_eq!(once, twice);
        assert!(approx(once.exposure_boost, 0.3) && approx(once.desaturation, 0.5));
        assert_eq!(once.color_filter, [0.5, 0.8, 1.0]);
        // a second, different effect: max per term, min on the filter
        let b = reals(|r| { r.exposure_boost = 0.1; r.desaturation = 0.9; r.color_filter = [0.9, 0.4, 1.0]; r.contrast_enhance = 0.2; });
        let mut both = once;
        both.accumulate(&b, 1.0);
        assert!(approx(both.exposure_boost, 0.3) && approx(both.desaturation, 0.9) && approx(both.contrast_enhance, 0.2));
        assert_eq!(both.color_filter, [0.5, 0.4, 1.0]);
        // order independence
        let mut rev = ScreenFxAccum::identity();
        rev.accumulate(&b, 1.0);
        rev.accumulate(&a, 1.0);
        assert_eq!(both, rev);
    }

    #[test]
    fn falloff_scales_and_lerps_the_filter_toward_white() {
        let a = reals(|r| { r.desaturation = 1.0; r.color_filter = [0.0, 0.0, 0.0]; });
        let mut acc = ScreenFxAccum::identity();
        acc.accumulate(&a, 0.25);
        assert!(approx(acc.desaturation, 0.25));
        assert!(acc.color_filter.iter().all(|&v| approx(v, 0.75)));
        let mut off = ScreenFxAccum::identity();
        off.accumulate(&a, 0.0);
        assert!(off.is_identity());
    }

    #[test]
    fn exposure_boost_minus_deboost_is_stops() {
        let mut acc = ScreenFxAccum::identity();
        acc.exposure_boost = 1.0;
        acc.exposure_deboost = 0.5;
        assert!(approx(compose(&acc).gain, 2f32.powf(0.5)));
    }

    #[test]
    fn columns_match_row_vector_matrix() {
        let mut acc = ScreenFxAccum::identity();
        acc.hue_right = 40.0;
        acc.saturation = 0.3;
        acc.contrast_enhance = 0.1;
        acc.color_filter = [0.9, 0.95, 1.0];
        acc.color_floor = [0.02, 0.0, 0.01];
        let m = color_matrix(&acc);
        let r = compose(&acc);
        let v = [0.3f32, 0.6, 0.2, 1.0];
        for j in 0..3 {
            let a = (v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j] + v[3] * m[3][j]).clamp(0.0, 1.0);
            let b = r.apply([0.3, 0.6, 0.2])[j];
            assert!(approx(a, b));
        }
    }

    #[test]
    fn hue_direction_report() {
        // Documented in docs/hrek_re/20_forge_screen_fx.md: at +120 deg the engine's rotation moves
        // pure red to the green or blue axis. Pin whichever it is so a regression is visible.
        let m = hue_matrix(120.0);
        let c = [1.0f32, 0.0, 0.0];
        let mut o = [0.0f32; 3];
        for j in 0..3 { o[j] = c[0] * m[0][j] + c[1] * m[1][j] + c[2] * m[2][j] + m[3][j]; }
        let dominant = if o[1] > o[0] && o[1] > o[2] { 'g' } else if o[2] > o[0] && o[2] > o[1] { 'b' } else { 'r' };
        assert_eq!(dominant, 'g', "hue +120 mapped red to {o:?}");
    }
}
