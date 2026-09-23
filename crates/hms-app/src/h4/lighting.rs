//! Halo 4 scene lighting tags: atmosphere fog (`fogg`) -> the renderer's fog uniform, and the
//! camera fx (`cfxs`) -> exposure meter, bloom, filmic curve and colour-grading LUT.
//!
//! Fog law (halo4.dll `sub_18035B330` fog blend state, `sub_18039E6F4` fogg accumulate,
//! `sub_1803A3C90` density = thickness * 0.01, `sub_1803A3CC8` fog light; explicit shader
//! `explicit_shaders\postprocess\screen_atmospheric_fog` entries 0 (table) / 4 (apply), scratch
//! listings h4light/fog/dis): Halo 4's atmospheric fog is a deferred screen-space pass over the
//! HDR buffer, `final = scene * T + inscatter`, per height layer
//!   dh   = min(hi - base, H) - (lo - base)          (ray extent inside the layer, lo/hi = min/max z)
//!   d    = min(max(0, sat(dh / (hi - lo)) * (dist + dist_bias)), falloff_end)
//!   k    = sat((H - (lo - base)) / H)^2
//!   T    = exp2(-d * thickness * 0.01 * k),  inscatter = colour * (1 - T)
//! plus the optional fog light (flags bit 3): `f = sat((dot(view, L) - cos r) / (1.001 - cos r))
//! ^ ang_falloff * sat(1 + T / (nearby_cutoff - 1)) ^ dist_falloff`, colour = tint * tint_int,
//! or with flags bit 7 = tint_int * the BSP sun colour along -sun_dir. The scnr Fog block
//! (+0x56C, 16-B elements, `fogg` ref @0) selects the tag (weighted by atmosphere blend
//! volumes on campaign maps - the MP maps have exactly one). Haven's `fogg` has both
//! thicknesses 0 = no atmospheric fog; Ravine: ground layer base 5 / height 2000 / thickness
//! 0.05 / falloff 2000, colour (0.571, 0.730, 0.956), fog light 0.1 x sun.
//!
//! `fogg` layout (== the X360 Assembly fogg.xml, matching the DLL reader): flags u16 @0,
//! dist bias f32 @4; sky layer @0x8 {base, height, thickness, falloff end, rgb @0x18}; ground
//! layer @0x34 {same, rgb @0x44}; ceiling @0x60; fog light @0x8C {pitch, yaw, angular radius,
//! tint rgb @0x98, tint intensity @0xA4, angular falloff @0xA8, distance falloff @0xAC, nearby
//! cutoff @0xB0}.

use eframe::wgpu;
use super::cache::{ByteRead, H4Cache};

pub const OFF_SCNR_FOG: usize = 0x56C;
pub const FOGG_FLAG_SKY: u16 = 1 << 0;
pub const FOGG_FLAG_GROUND: u16 = 1 << 1;
pub const FOGG_FLAG_LIGHT: u16 = 1 << 3;
pub const FOGG_FLAG_LIGHT_IS_SUN: u16 = 1 << 7;

#[derive(Clone, Copy, Debug, Default)]
pub struct FogLayer {
    pub base: f32,
    pub height: f32,
    pub thickness: f32,
    pub falloff_end: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct H4Fog {
    pub name: String,
    pub flags: u16,
    pub dist_bias: f32,
    pub sky: FogLayer,
    pub ground: FogLayer,
    pub light_pitch: f32,
    pub light_yaw: f32,
    pub light_radius_deg: f32,
    pub light_tint: [f32; 3],
    pub light_intensity: f32,
    pub light_ang_falloff: f32,
    pub light_dist_falloff: f32,
    pub light_nearby_cutoff: f32,
}

fn layer(c: &H4Cache, o: usize) -> FogLayer {
    let d = c.data();
    FogLayer { base: d.f32_at(o), height: d.f32_at(o + 4), thickness: d.f32_at(o + 8), falloff_end: d.f32_at(o + 12), color: [d.f32_at(o + 16), d.f32_at(o + 20), d.f32_at(o + 24)] }
}

/// The scenario's first `fogg` (None when the scnr Fog block is empty or the ref is null).
pub fn load_fog(c: &H4Cache) -> Option<H4Fog> {
    let scnr = c.find_tags(b"scnr").first().copied()?;
    let sm = c.tag_meta(scnr)?;
    let (_, fo) = c.block(sm + OFF_SCNR_FOG)?;
    let tag = c.tag_ref_of(fo, b"fogg")?;
    let m = c.tag_meta(tag)?;
    let d = c.data();
    Some(H4Fog {
        name: c.tag_name(tag).to_string(),
        flags: d.u16_at(m),
        dist_bias: d.f32_at(m + 4),
        sky: layer(c, m + 0x8),
        ground: layer(c, m + 0x34),
        light_pitch: d.f32_at(m + 0x8C),
        light_yaw: d.f32_at(m + 0x90),
        light_radius_deg: d.f32_at(m + 0x94),
        light_tint: [d.f32_at(m + 0x98), d.f32_at(m + 0x9C), d.f32_at(m + 0xA0)],
        light_intensity: d.f32_at(m + 0xA4),
        light_ang_falloff: d.f32_at(m + 0xA8),
        light_dist_falloff: d.f32_at(m + 0xAC),
        light_nearby_cutoff: d.f32_at(m + 0xB0),
    })
}

impl H4Fog {
    /// True when a layer would fog anything (the engine skips the pass at thickness 0).
    pub fn active(&self) -> bool {
        (self.flags & FOGG_FLAG_GROUND != 0 && self.ground.thickness > 2e-4) || (self.flags & FOGG_FLAG_SKY != 0 && self.sky.thickness > 2e-4)
    }

    /// The renderer's 28-float fog uniform packed for the Halo 4 lane (mesh.rs `h4_fog`):
    /// atm0 = 0 (the Reach sky-tint lane), atm1 = ground colour, atm2 = fog light colour,
    /// atm3 = 0 (keeps the Reach `apply_fog` off), atm4 = (density, base, height, falloff_end) of
    /// the ground layer, atm5 = (dist bias, light on, cos radius, angular falloff), atm6 = (light
    /// dir xyz, 4.0 = the Halo 4 marker); the sky layer density rides in atm1.w, its (base,
    /// height, falloff) in atm2.w / atm5.w is NOT carried (Ravine + Haven: sky thickness 0).
    /// `sun_dir` = unit vector TO the sun, `sun_rgb` = the BSP sun colour (fog light flag 7).
    pub fn uniform(&self, sun_dir: [f32; 3], sun_rgb: [f32; 3]) -> [f32; 28] {
        let mut f = [0.0f32; 28];
        let g = &self.ground;
        // atm0 stays zero: the renderer reads fog[0..3] as the Reach sky-gradient tint and a
        // Halo 4 sky is a scenery object, so the procedural gradient keeps its neutral default
        f[4..7].copy_from_slice(&g.color);
        f[7] = if self.flags & FOGG_FLAG_SKY != 0 { self.sky.thickness * 0.01 } else { 0.0 };
        let light_on = self.flags & FOGG_FLAG_LIGHT != 0 && self.light_intensity > 0.0;
        let (lc, ld) = if self.flags & FOGG_FLAG_LIGHT_IS_SUN != 0 {
            ([sun_rgb[0] * self.light_intensity, sun_rgb[1] * self.light_intensity, sun_rgb[2] * self.light_intensity], sun_dir)
        } else {
            let (p, y) = (self.light_pitch.to_radians(), self.light_yaw.to_radians());
            ([self.light_tint[0] * self.light_intensity, self.light_tint[1] * self.light_intensity, self.light_tint[2] * self.light_intensity],
             [p.cos() * y.cos(), p.cos() * y.sin(), p.sin()])
        };
        f[8..11].copy_from_slice(&lc);
        f[11] = self.light_dist_falloff;
        f[16] = if self.flags & FOGG_FLAG_GROUND != 0 { g.thickness * 0.01 } else { 0.0 };
        f[17] = g.base;
        f[18] = g.height;
        f[19] = g.falloff_end.max(100.0);
        f[20] = self.dist_bias;
        f[21] = if light_on { 1.0 } else { 0.0 };
        f[22] = self.light_radius_deg.to_radians().cos();
        f[23] = self.light_ang_falloff;
        f[24..27].copy_from_slice(&ld);
        f[27] = 4.0;
        // nearby cutoff rides in atm1.w's neighbour slot? no free lane: fold 1/(cutoff-1) into atm3.x
        f[12] = 1.0 / (self.light_nearby_cutoff - 1.0).min(-1e-3);
        f
    }
}

// ---- camera fx (`cfxs`) -> exposure meter, bloom, filmic, LUT -------------------------------
//
// Engine law (halo4.dll `sub_18035A470` cfxs consumer, `sub_180359C04` state update, `sub_180359768`
// auto-exposure, `sub_180380FD0` meter chain, `sub_18034E7AC` ps_view_exposure = (E / 2^s, 2^s, E,
// 1) with s = render_HDR_target_stops = 3, `sub_180375EAC` self-illum exposure, `sub_180381578`
// colour-grading LUT constants, `sub_1803818E0` filmic constants, explicit `final_composite` /
// `downsample_block_bloom_2x2` / `exposure_downsample` PS asm - h4light/post/, doc section 4):
//   E = 2^stops (absolute); the meter measures M = blend(mean log2, log2 mean; sensitivity) of the
//   centre-weighted (auto_exposure_weight 18x10) luma of w * c, c = min(E L, 8), w = highlight *
//   luma(c) + inherent + self_illum * min(si luma, 1/32); the stops adapt toward
//   screen_brightness - M (bit 2) inside the [+0x10, +0x14] range with delay / blend / max change;
//   self-illum stops = (stops - pref) * change + pref; composite: c = scene * E + bloom; filmic
//   t = Hable(c) / Hable(W) when +0x100 bit 5; out = sqrt(t) then the 16^3 colour-grading LUT
//   (+0xF0) sampled at out * 15/16 + 1/32.
// cfxs layout (0x120 B, == the X360 Assembly plugin; 16-B function blocks {u16 flags, pad, f32
// value, f32 max change, f32 blend}): exposure block @0, range (min, max stops) @0x10, screen
// brightness (log2 target) @0x18, delay @0x1C, sensitivity block @0x20, highlight / inherent /
// self-illum bloom blocks @0x28 / 0x38 / 0x48, bloom intensity @0x58, large / medium / small
// colour blocks @0x68 / 0x78 / 0x88 (rgb @+4), self-illum stops @0xCC / 0xDC, colour grading
// `bitm` @0xF0, filmic flags @0x100 (bit 5 = enabled), A B C D E F W @0x104.
pub const OFF_SCNR_DEFAULT_CFXS: usize = 0x70C;
/// halo4.dll sub_180375EAC: `view+16 = 2^(stops + offsets) * (F / 1.4938016)` when the
/// render-settings byte at 0x180E84F69 is set, F = the float at 0x180E84F6C - both are 1 in the
/// shipped image's defaults (byte 0x180E84F40..0x77 block; the same 0.66943294 Reach applies
/// through its white point). The meter and the composite both see this E (ps_view_exposure =
/// (E/8, 8, E, 1)), so it is folded into the absolute band here. The per-frame copy of that
/// block comes from a bss mirror whose writer was not located, so the runtime value is inferred
/// from the image defaults.
pub const H4_EXPOSURE_SCALE: f32 = 1.0 / 1.4938016;
/// The absolute exposure gain of a cfxs-band stops value: `E = 2^stops * H4_EXPOSURE_SCALE`
/// (the inverse of `H4CameraFx::stops_of_gain`). `// #h4-expo-3` used by the Lighting panel's
/// band sliders so an edit stays on the Halo 4 lane instead of Reach's `set_auto_exposure`.
pub fn gain_of_stops(stops: f32) -> f32 { stops.exp2() * H4_EXPOSURE_SCALE }
pub const OFF_CFXS_COLOR_GRADING: usize = 0xF0;
pub const CFXS_FILMIC_ENABLED: u16 = 1 << 5;

#[derive(Clone, Debug, Default)]
pub struct H4CameraFx {
    pub name: String,
    pub exposure_flags: u16,
    pub exposure_target: f32,
    pub exposure_range: [f32; 2],
    /// log2 of the target measured luminance.
    pub screen_brightness: f32,
    pub sensitivity: f32,
    pub bloom_highlight: f32,
    pub bloom_inherent: f32,
    pub bloom_self_illum: f32,
    pub bloom_intensity: f32,
    pub bloom_colors: [[f32; 3]; 3],
    pub filmic_flags: u16,
    /// A (shoulder), B (linear), C (angle), D (toe), E (toe num), F (toe den), W (white).
    pub filmic: [f32; 7],
    /// +0xF0 colour-grading `bitm` (16^3 LUT sampled after the sqrt), None = no grading.
    pub color_grading: Option<usize>,
    /// Exposure block: max change per frame (stops, +0x8), blend (+0xC), delay
    /// (seconds, +0x1C) - the adaptation dynamics of `sub_180359768` (informational in HMS).
    pub exposure_max_change: f32,
    pub exposure_blend: f32,
    pub exposure_delay: f32,
    /// Self-illum exposure: preferred stops (+0xD0) and change (+0xE0);
    /// si_stops = (stops - preferred) * change + preferred (sub_180359C04 state +600).
    pub si_preferred_stops: f32,
    pub si_change: f32,
    /// Bit i set = block i of `CFXS_BLOCKS` carried the "use default" flag (bit 0 of its
    /// u16) and was resolved from the fallback chain (`resolve_cfxs_defaults`); diag only.
    pub defaulted: u32,
}

/// Every cfxs function block HMS reads: (offset, size). halo4.dll `sub_180359AD8`:
/// a block whose u16 flags bit 0 ("use default") is set is taken from the next cfxs of the chain
/// current (area / scripted) -> scenario default (+0x70C) -> the `matg` (globals\\globals) default
/// (+0x730 `globals\\defaults\\default`, halo4.dll sub_180375EAC: globals +1852); the matg
/// default's own values are final.
pub const CFXS_BLOCKS: [(usize, usize); 14] = [
    (0x00, 0x20), (0x20, 0x08), (0x28, 0x10), (0x38, 0x10), (0x48, 0x10), (0x58, 0x10), (0x68, 0x10),
    (0x78, 0x10), (0x88, 0x10), (0xCC, 0x10), (0xDC, 0x10), (0xEC, 0x14), (0x100, 0x20), (0x98, 0x10),
];
pub const OFF_MATG_DEFAULT_CFXS: usize = 0x730;
/// The engine's bloom-intensity scale at render resolutions above 1280x720
/// (halo4.dll `sub_1803812B4`: `intensity *= min(1, 921600 / (w h))`, render-settings float
/// 0x180E84F78 = 1 keeps the rule on; 1080p = 0.444, 1440p = 0.25).
pub fn h4_bloom_res_factor(w: u32, h: u32) -> f32 {
    if w > 1280 || h > 720 { (921_600.0 / (w as f32 * h as f32)).clamp(0.0, 1.0) } else { 1.0 }
}

/// Resolve the "use default" blocks of the cfxs at `m` into an owned 0x120-byte copy:
/// `chain` = the fallback metas in order (scenario default, matg default). Returns (bytes, mask).
pub fn resolve_cfxs_defaults(c: &H4Cache, m: usize, chain: &[usize]) -> (Vec<u8>, u32) {
    let d = c.data();
    let mut out = d[m..m + 0x120].to_vec();
    let mut mask = 0u32;
    for (i, &(o, sz)) in CFXS_BLOCKS.iter().enumerate() {
        if d.u16_at(m + o) & 1 == 0 { continue; }
        for &fm in chain {
            if fm == m { continue; }
            let src = &d[fm + o..fm + o + sz];
            out[o..o + sz].copy_from_slice(src);
            mask |= 1 << i;
            if d.u16_at(fm + o) & 1 == 0 { break; }
        }
    }
    (out, mask)
}

pub const OFF_SCNR_CAMERA_FX: usize = 0x578;
pub const SCNR_CAMERA_FX_ELEM: usize = 0x30;

/// The scenario's `cfxs`: the scnr +0x70C default, unless the named Camera FX block (+0x578,
/// 0x30-B elements: name sid @0, `cfxs` ref @4; script-selected per area on campaign maps)
/// carries a `*default` entry - m10_crash's scnr default is the cryo-intro settings, the level
/// plays under `m10_default` (PROVISIONAL selection: the hs `camera_fx_set` scripts are not run).
pub fn load_camera_fx(c: &H4Cache) -> Option<H4CameraFx> {
    let scnr = c.find_tags(b"scnr").first().copied()?;
    let sm = c.tag_meta(scnr)?;
    let mut tag = c.tag_ref_of(sm + OFF_SCNR_DEFAULT_CFXS, b"cfxs");
    if let Some((n, o)) = c.block(sm + OFF_SCNR_CAMERA_FX) {
        for i in 0..n.min(64) {
            let e = o + i * SCNR_CAMERA_FX_ELEM;
            if c.sid(c.data().u32_at(e)).ends_with("default") {
                if let Some(t) = c.tag_ref_of(e + 4, b"cfxs") { tag = Some(t); }
            }
        }
    }
    let tag = tag?;
    let m = c.tag_meta(tag)?;
    // "use default" chain: the scenario default cfxs (when the Camera FX entry differs)
    // then the rasg default (`globals\defaults\default`).
    let mut chain = Vec::new();
    if let Some(t) = c.tag_ref_of(sm + OFF_SCNR_DEFAULT_CFXS, b"cfxs") { if let Some(fm) = c.tag_meta(t) { chain.push(fm); } }
    if let Some(r) = c.find_tags(b"matg").first().copied() {
        if let Some(rm) = c.tag_meta(r) {
            if let Some(fm) = c.tag_ref_of(rm + OFF_MATG_DEFAULT_CFXS, b"cfxs").and_then(|t| c.tag_meta(t)) { chain.push(fm); }
        }
    }
    let (bytes, defaulted) = resolve_cfxs_defaults(c, m, &chain);
    // the grading `bitm` ref is read from whichever cfxs the block resolved to (chain order)
    let cg_meta = if c.data().u16_at(m + 0xEC) & 1 == 0 { m } else { chain.iter().copied().find(|&fm| fm != m && c.data().u16_at(fm + 0xEC) & 1 == 0).or(chain.last().copied()).unwrap_or(m) };
    let d: &[u8] = &bytes;
    let m = 0usize;
    let rgb = |o: usize| [d.f32_at(o), d.f32_at(o + 4), d.f32_at(o + 8)];
    Some(H4CameraFx {
        name: c.tag_name(tag).to_string(),
        defaulted,
        exposure_flags: d.u16_at(m),
        exposure_target: d.f32_at(m + 4),
        exposure_range: [d.f32_at(m + 0x10), d.f32_at(m + 0x14)],
        screen_brightness: d.f32_at(m + 0x18),
        sensitivity: d.f32_at(m + 0x24),
        bloom_highlight: d.f32_at(m + 0x2C),
        bloom_inherent: d.f32_at(m + 0x3C),
        bloom_self_illum: d.f32_at(m + 0x4C),
        bloom_intensity: d.f32_at(m + 0x5C),
        bloom_colors: [rgb(m + 0x6C), rgb(m + 0x7C), rgb(m + 0x8C)],
        filmic_flags: d.u16_at(m + 0x100),
        filmic: [d.f32_at(m + 0x104), d.f32_at(m + 0x108), d.f32_at(m + 0x10C), d.f32_at(m + 0x110), d.f32_at(m + 0x114), d.f32_at(m + 0x118), d.f32_at(m + 0x11C)],
        color_grading: c.tag_ref_of(cg_meta + OFF_CFXS_COLOR_GRADING, b"bitm"),
        exposure_max_change: d.f32_at(m + 0x8),
        exposure_blend: d.f32_at(m + 0xC),
        exposure_delay: d.f32_at(m + 0x1C),
        si_preferred_stops: d.f32_at(m + 0xD0),
        si_change: d.f32_at(m + 0xE0),
    })
}

/// Push a Halo 4 map's post-process state to the renderer (the one place the GUI,
/// the headless renderer and the script host share): the absolute exposure band + the engine
/// meter (`set_h4_meter`), the filmic curve, the bloom curve / colours, the colour-grading LUT
/// and the self-illum exposure law. `stats` carries the cfxs + decoded LUT of the loaded map.
pub fn apply_post(renderer: &mut hms_render::SceneRenderer, device: &wgpu::Device, queue: &wgpu::Queue, x: Option<&H4CameraFx>, lut: Option<&(Vec<u8>, u32)>) {
    apply_post_ex(renderer, device, queue, x, lut, true)
}

/// `apply_post` without the one-line bloom report - for the SECOND renderer a caller pushes the
/// same map state to (the palette preview's offscreen renderer, #h4-preview), which must not print
/// the load line twice.
pub fn apply_post_quiet(renderer: &mut hms_render::SceneRenderer, device: &wgpu::Device, queue: &wgpu::Queue, x: Option<&H4CameraFx>, lut: Option<&(Vec<u8>, u32)>) {
    apply_post_ex(renderer, device, queue, x, lut, false)
}

fn apply_post_ex(renderer: &mut hms_render::SceneRenderer, device: &wgpu::Device, queue: &wgpu::Queue, x: Option<&H4CameraFx>, lut: Option<&(Vec<u8>, u32)>, verbose: bool) {
    renderer.set_exposure(queue, 1.0);
    let Some(x) = x else {
        renderer.set_h4_filmic(queue, None);
        renderer.set_h4_meter(queue, None);
        renderer.set_h4_color_grading(device, queue, None);
        renderer.set_h4_bloom(queue, false);
        renderer.set_illum_exposure(None);
        return;
    };
    let (lo, hi) = x.gain_range();
    // key 1.0: the H4 meter writes log2(key) - stops, so the post's key / 2^meter = 2^stops.
    // The band carries the engine's display factor: E = 2^stops * H4_EXPOSURE_SCALE
    // (`gain_range` applies it), so the solved gain IS the engine's ps_view_exposure.z.
    renderer.set_exposure_band_abs(queue, 1.0, lo, hi);
    renderer.set_h4_meter(queue, Some([x.bloom_highlight, x.bloom_inherent, x.bloom_self_illum, x.sensitivity, x.screen_brightness]));
    renderer.set_h4_filmic(queue, x.filmic_params());
    renderer.set_bloom_curve(queue, x.bloom_highlight, x.bloom_inherent, x.bloom_intensity);
    renderer.set_bloom_colors(queue, x.bloom_colors);
    // The Halo 4 bloom chain (own curve level / kernels / alpha-weighted combine / resolution
    // scale, hms-render BLOOM_WGSL `*_h4`); the lanes above carry the resolved cfxs values.
    renderer.set_h4_bloom(queue, true);
    let (w, h) = renderer.size();
    let dflt = |i: usize| if x.defaulted >> i & 1 != 0 { "*" } else { "" };
    if verbose { eprintln!("h4 bloom: '{}' highlight {}{} inherent {}{} self_illum {}{} intensity {}{} x res {:.3} ({}x{}) = {:.4} | large {:?}{} medium {:?}{} small {:?}{} | LUT {} (* = cfxs 'use default' -> scenario / matg default)",
        x.name.rsplit('\\').next().unwrap_or(""), x.bloom_highlight, dflt(2), x.bloom_inherent, dflt(3), x.bloom_self_illum, dflt(4), x.bloom_intensity, dflt(5),
        h4_bloom_res_factor(w, h), w, h, x.bloom_intensity * h4_bloom_res_factor(w, h), x.bloom_colors[0], dflt(6), x.bloom_colors[1], dflt(7), x.bloom_colors[2], dflt(8), lut.is_some()); }
    renderer.set_h4_color_grading(device, queue, lut.map(|(b, n)| (b.as_slice(), *n)));
    // self-illum exposure (sub_180359C04 state +600): si_stops = (stops - pref) * change + pref, so
    // E_si / E = 2^((1 - change) (pref - stops)) = the renderer's Reach ILLUM_SCALE law
    // 2^((1 - s)(P - e)) with e = log2(gain / (key * 10)) -> P' = pref - log2(10) (key = 1).
    // The gain carries H4_EXPOSURE_SCALE, so e = log2(gain / 10) sits log2(scale) below the
    // cfxs stops; shift P' by the same amount (the engine's E_si / E has no such factor:
    // view+20 = 2^(si_stops - stops)).
    renderer.set_illum_exposure(Some((x.si_preferred_stops - 10f32.log2() + H4_EXPOSURE_SCALE.log2(), x.si_change)));
}

/// The engine's auto-exposure ADAPTATION (halo4.dll `sub_180359768`), in the cfxs's own stops.
///
/// The meter gives the target the exposure converges to; the engine does not jump there. Each
/// frame it pushes the band-clamped target into a 120-entry ring and only moves when EVERY target
/// in the last `delay` seconds (at the display refresh rate, 1..120 frames) lies on the same side
/// of the current value - the minimum of the window must be above it to rise, the maximum below it
/// to fall. The step is `(target - cur) * blend` with `blend` divided by the integer number of
/// 30 Hz ticks in one display frame (`blend / (Hz / 30)`, clamped to 1) and only when exposure
/// flags bit 1 is set, then hard-clamped to +-`max change`.
///
/// `cur` = the adapted stops (None = not adapted yet: snap, like the engine's instant-adopt frames
/// after a load), `hist` = the ring, `target` = the band-clamped stops the meter solved for
/// (the engine's `prev + screen_brightness - M(prev)` converges to the same point), `dt` = the
/// frame time (HMS's frame rate is variable, so the engine's per-frame constants are evaluated at
/// the measured rate instead of a fixed 60 Hz). Returns the stops to expose with.
/// `// #h4-expo-3`
pub const H4_AE_HISTORY: usize = 120;
pub const CFXS_EXPOSURE_BLEND: u16 = 1 << 1;
pub fn adapt_stops(
    cur: &mut Option<f32>,
    hist: &mut std::collections::VecDeque<f32>,
    x: &H4CameraFx,
    target: f32,
    dt: f32,
) -> f32 {
    hist.push_back(target);
    while hist.len() > H4_AE_HISTORY { hist.pop_front(); }
    let Some(c) = *cur else { *cur = Some(target); return target };
    // Converged: the geometric step below never lands exactly on the target, so snap once the
    // remaining error is far under what a display can show (1e-4 stops = 0.007 % of the gain).
    // A still camera then renders a STEADY frame instead of drifting by float dust forever.
    if (target - c).abs() <= 1e-4 { *cur = Some(target); return target; }
    let hz = (1.0 / dt.clamp(1.0 / 240.0, 1.0 / 20.0)).clamp(20.0, 240.0);
    // the delay window in frames: the engine truncates `Hz * delay` and clamps it to [1, 120]
    let n = ((hz * x.exposure_delay.max(0.0)) as usize).clamp(1, H4_AE_HISTORY).min(hist.len());
    let win = hist.iter().rev().take(n);
    let agrees = if target > c {
        win.fold(f32::MAX, |a, &b| a.min(b)) > c
    } else {
        win.fold(f32::MIN, |a, &b| a.max(b)) < c
    };
    if !agrees { return c; }
    let mut d = target - c;
    if x.exposure_flags & CFXS_EXPOSURE_BLEND != 0 {
        let ticks = ((hz as u32) / 30).max(1) as f32;
        d *= (x.exposure_blend / ticks).clamp(0.0, 1.0);
    }
    // max change per frame; 0 would freeze the exposure, which no shipped cfxs asks for
    let m = x.exposure_max_change.abs();
    if m > 0.0 { d = d.clamp(-m, m); }
    let v = c + d;
    *cur = Some(v);
    v
}

impl H4CameraFx {
    /// The composite's filmic constants (P0..P4 of t = c (P0 c + P1) / (c (P2 c + P3) + P4) =
    /// Hable(c) / Hable(W)), None when the curve is disabled or degenerate (identity).
    pub fn filmic_params(&self) -> Option<[f32; 5]> {
        if self.filmic_flags & CFXS_FILMIC_ENABLED == 0 { return None; }
        let [a, b, cc, dd, e, f, w] = self.filmic;
        if [a, b, cc, dd, e, f, w].iter().any(|v| v.abs() < 1e-6) { return None; }
        let h = (w * (a * w + cc * b) + dd * e) / (w * (a * w + b) + dd * f) - e / f;
        if h.abs() < 1e-6 { return None; }
        Some([a * (f - e) / h, b * (cc * f - e) / h, a * f, b * f, dd * f * f])
    }
    /// Absolute exposure gain range of the adaptation: [2^min, 2^max] x `H4_EXPOSURE_SCALE`
    /// (the engine's `E = 2^stops / 1.4938016`, sub_180375EAC).
    pub fn gain_range(&self) -> (f32, f32) { (self.exposure_range[0].exp2() * H4_EXPOSURE_SCALE, self.exposure_range[1].exp2() * H4_EXPOSURE_SCALE) }
    /// The cfxs-band stops of an absolute gain (undoes `H4_EXPOSURE_SCALE`).
    pub fn stops_of_gain(gain: f32) -> f32 { (gain / H4_EXPOSURE_SCALE).max(1e-9).log2() }
    /// The closed-form key of the engine meter for inherent = 0, self-illum = 0, sensitivity = 0 and
    /// no clamp: the geomean exposed luminance that satisfies mean(log2(highlight * Y^2)) =
    /// screen_brightness is sqrt(2^sb / highlight). Used by the HMS_H4_POSTDBG=1 diagnostic (Reach
    /// meter) and the load log; the default path solves the full law on the GPU (`apply_post`).
    pub fn meter_key(&self) -> f32 {
        let target = self.screen_brightness.exp2();
        (target / self.bloom_highlight.max(0.05)).sqrt().clamp(0.02, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;

    /// `// #h4-expo-3` The engine's adaptation law (`sub_180359768`): snap on the first frame,
    /// then a `max change` per-frame clamp, the `blend` rate (flags bit 1) and the `delay`
    /// window that refuses to move while the recent targets straddle the current value.
    #[test]
    fn adapt_stops_follows_the_engine_law() {
        // dlc_forge_island's block: max change 0.1, blend 0.1, delay 0.0333 s, flags 0x56.
        let fi = H4CameraFx { exposure_flags: 0x56, exposure_max_change: 0.1, exposure_blend: 0.1, exposure_delay: 1.0 / 30.0, ..Default::default() };
        let dt = 1.0 / 60.0;
        let (mut cur, mut hist) = (None, std::collections::VecDeque::new());
        // first frame: snap (no fade-in from nowhere)
        assert_eq!(adapt_stops(&mut cur, &mut hist, &fi, -1.0, dt), -1.0);
        // a step up is HELD for the delay window (60 Hz x 1/30 s = 2 frames): the first frame
        // after the jump still sees the old target in the ring, so the engine does not move yet
        assert_eq!(adapt_stops(&mut cur, &mut hist, &fi, 1.0, dt), -1.0);
        // once the whole window agrees: blend 0.1 / (60/30 ticks) = 0.05 of the 2-stop error =
        // 0.1, exactly at the max-change clamp
        let v1 = adapt_stops(&mut cur, &mut hist, &fi, 1.0, dt);
        assert!((v1 - -0.9).abs() < 1e-5, "first step {v1}");
        // ... and it converges to the target with the still camera held
        let mut v = v1;
        for _ in 0..4000 { v = adapt_stops(&mut cur, &mut hist, &fi, 1.0, dt); }
        assert!((v - 1.0).abs() < 1e-3, "converged {v}");
        // an oscillating target never satisfies the delay window's same-side rule, so the
        // exposure holds still (the engine's anti-flicker gate)
        let held = v;
        for k in 0..60 { v = adapt_stops(&mut cur, &mut hist, &fi, if k % 2 == 0 { held - 1.0 } else { held + 1.0 }, dt); }
        assert!((v - held).abs() < 1e-3, "delay window let it drift: {v} vs {held}");
        // without the blend flag the step is the raw error clamped to max change
        let flat = H4CameraFx { exposure_flags: 0x54, exposure_max_change: 0.02, exposure_blend: 0.1, exposure_delay: 0.0, ..Default::default() };
        let (mut c2, mut h2) = (Some(0.0f32), std::collections::VecDeque::new());
        let v2 = adapt_stops(&mut c2, &mut h2, &flat, 1.0, dt);
        assert!((v2 - 0.02).abs() < 1e-6, "max-change step {v2}");
    }

    /// RE tool: the `rasg` (rasterizer globals) tag refs of HMS_H4_PS_MAP - the
    /// exposure meter's weight bitmap is read from rasg +0x10 (datum) / +0x9C (sub_180380FD0).
    #[test]
    #[ignore]
    fn dump_rasg_refs() {
        let map = std::env::var("HMS_H4_PS_MAP").unwrap_or_else(|_| "wraparound.map".into());
        let Some(dir) = maps_dir() else { return };
        let Ok(c) = H4Cache::open(&dir.join(&map)) else { return };
        for t in c.find_tags(b"rasg") {
            let Some(m) = c.tag_meta(t) else { continue };
            eprintln!("rasg {} meta {m:#x}", c.tag_name(t));
            let d = c.data();
            eprintln!("  dwords: {:?}", (0..16).map(|k| format!("{:#x}", d.u32_at(m + 4 * k))).collect::<Vec<_>>());
            if let Some((n, o)) = c.block(m) {
                eprintln!("  block@0: {n} elems at {o:#x}");
                for i in 0..n.min(64) {
                    let e = o + i * 16;
                    let r = c.tag_ref(e).map(|(cl, idx)| format!("{} {} {}", String::from_utf8_lossy(&cl), c.tag_name(idx), crate::h4::bitmaps::bitmap_info(&c, idx).map(|i| format!("{}x{}x{} kind {} fmt {}", i.w, i.h, i.depth, i.kind, i.format)).unwrap_or_default()));
                    eprintln!("    [{i}] @{:#x} {:?}", e - o, r);
                    if let Some(idx) = c.tag_ref_of(e, b"bitm") {
                        if c.tag_name(idx).ends_with("auto_exposure_weight") {
                            if let Ok(crate::h4::bitmaps::Texel::Bgra { bytes, w, h }) = crate::h4::bitmaps::load_base_texture(&c, idx) {
                                eprintln!("      auto_exposure_weight {w}x{h} (b g r a of row-major texels):");
                                for y in 0..h as usize {
                                    let row: Vec<String> = (0..w as usize).map(|x| { let p = &bytes[(y * w as usize + x) * 4..][..4]; format!("{:3}/{:3}/{:3}/{:3}", p[0], p[1], p[2], p[3]) }).collect();
                                    eprintln!("      {}", row.join(" "));
                                }
                            }
                        }
                    }
                }
            }
            for o in (0..0x400).step_by(4) {
                if let Some((cl, idx)) = c.tag_ref(m + o) {
                    if cl.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
                        let info = crate::h4::bitmaps::bitmap_info(&c, idx).map(|i| format!("{}x{}x{} kind {} fmt {}", i.w, i.h, i.depth, i.kind, i.format)).unwrap_or_default();
                        eprintln!("  +{:#x} {} {} {}", o, String::from_utf8_lossy(&cl), c.tag_name(idx), info);
                    }
                }
            }
        }
    }

    /// RE tool: every map's cfxs exposure / bloom / grading fields on one line each
    /// (HMS_H4_MAPS_GLOB = substring filter). Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_cfxs_all() {
        let Some(dir) = maps_dir() else { return };
        let filt = std::env::var("HMS_H4_MAPS_GLOB").unwrap_or_default();
        let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".map") && n.contains(&filt)).collect();
        names.sort();
        for n in names {
            let Ok(c) = H4Cache::open(&dir.join(&n)) else { continue };
            let Some(x) = load_camera_fx(&c) else { eprintln!("{n}: no cfxs"); continue };
            let lut = x.color_grading.map(|t| {
                let i = crate::h4::bitmaps::bitmap_info(&c, t);
                let v = crate::h4::bitmaps::load_volume_lut(&c, t).map(|(b, n, dxgi)| {
                    // identity check: texel (i,j,k) should be ~ (i,j,k)/(n-1) in r,g,b; print corners + mean abs dev
                    let mut dev = 0.0f64;
                    for z in 0..n { for y in 0..n { for xx in 0..n {
                        let p = &b[(((z * n + y) * n + xx) * 4) as usize..][..4];
                        dev += ((p[2] as f64 / 255.0 - xx as f64 / (n - 1) as f64).abs() + (p[1] as f64 / 255.0 - y as f64 / (n - 1) as f64).abs() + (p[0] as f64 / 255.0 - z as f64 / (n - 1) as f64).abs()) / 3.0;
                    }}}
                    let t = |xx: u32, y: u32, z: u32| { let p = &b[(((z * n + y) * n + xx) * 4) as usize..][..4]; format!("({},{},{})", p[2], p[1], p[0]) };
                    format!("n {n} dxgi {dxgi} mean|dev| {:.4} black {} white {} mid {} red {} green {} blue {}", dev / (n * n * n) as f64, t(0, 0, 0), t(n - 1, n - 1, n - 1), t(n / 2, n / 2, n / 2), t(n - 1, 0, 0), t(0, n - 1, 0), t(0, 0, n - 1))
                }).unwrap_or_else(|e| format!("ERR {e}"));
                format!("{} {:?} {v}", c.tag_name(t).rsplit('\\').next().unwrap_or(""), i.map(|i| (i.w, i.h, i.depth, i.kind, i.format, i.levels, i.total)))
            });
            eprintln!("{n}: '{}' expo flags {:#x} target {} range [{}, {}] sb {:.3} maxchg {} blend {} delay {} sens {} si_pref {} si_change {} | bloom hl {} inh {} si {} int {} | filmic {:?} | LUT {:?}",
                x.name.rsplit('\\').next().unwrap_or(""), x.exposure_flags, x.exposure_target, x.exposure_range[0], x.exposure_range[1], x.screen_brightness, x.exposure_max_change, x.exposure_blend, x.exposure_delay, x.sensitivity, x.si_preferred_stops, x.si_change,
                x.bloom_highlight, x.bloom_inherent, x.bloom_self_illum, x.bloom_intensity, x.filmic_params(), lut);
        }
    }

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Ravine: ground layer 5 / 2000 / 0.05 / 2000, colour (0.571, 0.730, 0.956), sun-tinted fog
    /// light; Haven: both thicknesses 0 (no atmospheric fog).
    #[test]
    fn fogg_values() {
        if let Some(c) = open("ca_forge_ravine.map") {
            let f = load_fog(&c).expect("ravine fogg");
            assert!(f.name.ends_with("ca_forge_ravine_fog"), "{}", f.name);
            assert_eq!(f.flags, 0x8A);
            assert!((f.dist_bias + 100.0).abs() < 1e-3);
            assert!((f.ground.base - 5.0).abs() < 1e-3 && (f.ground.height - 2000.0).abs() < 1e-3 && (f.ground.thickness - 0.05).abs() < 1e-4 && (f.ground.falloff_end - 2000.0).abs() < 1e-3);
            assert!((f.ground.color[0] - 0.571125).abs() < 1e-4 && (f.ground.color[2] - 0.955973).abs() < 1e-4);
            assert!((f.light_radius_deg - 110.0).abs() < 1e-3 && (f.light_intensity - 0.1).abs() < 1e-5 && (f.light_ang_falloff - 30.0).abs() < 1e-3);
            assert!(f.active());
            let u = f.uniform([0.0, 0.0, 1.0], [14.0, 12.3, 8.3]);
            assert!((u[16] - 0.0005).abs() < 1e-6 && u[27] == 4.0 && (u[8] - 1.4).abs() < 1e-4);
        }
        if let Some(c) = open("wraparound.map") {
            let f = load_fog(&c).expect("haven fogg");
            assert!(!f.active(), "{f:?}");
        }
    }

    /// Ravine cfxs: range [-0.5, 1.5] stops, screen brightness -4.366, bloom 1 / 0 / 1 / 0.5,
    /// colours 0.2 / 0.4 / 0.7, filmic (.2 .3 .2 .2 .01 .4 4) enabled -> P = (0.10494, 0.02825, 0.08, 0.12, 0.032).
    #[test]
    fn cfxs_values() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let x = load_camera_fx(&c).expect("ravine cfxs");
        assert!(x.name.ends_with("post_fx\\ca_forge_ravine"), "{}", x.name);
        assert!((x.exposure_range[0] + 0.5).abs() < 1e-4 && (x.exposure_range[1] - 1.5).abs() < 1e-4);
        assert!((x.screen_brightness + 4.36578).abs() < 1e-3 && (x.bloom_highlight - 1.0).abs() < 1e-5 && (x.bloom_intensity - 0.5).abs() < 1e-5);
        assert!((x.bloom_colors[0][0] - 0.2).abs() < 1e-5 && (x.bloom_colors[2][2] - 0.7).abs() < 1e-5);
        let p = x.filmic_params().expect("filmic on");
        for (got, want) in p.iter().zip([0.10494f32, 0.02825, 0.08, 0.12, 0.032]) { assert!((got - want).abs() < 2e-4, "{p:?}"); }
        assert!((x.meter_key() - 0.2202).abs() < 1e-3, "{}", x.meter_key());
        // adaptation dynamics, self-illum exposure law, colour-grading volume
        assert!((x.exposure_max_change - 0.5).abs() < 1e-5 && (x.exposure_blend - 0.15).abs() < 1e-5 && (x.exposure_delay - 0.1).abs() < 1e-5);
        assert!(x.si_preferred_stops.abs() < 1e-6 && (x.si_change - 0.2).abs() < 1e-6);
        let lut = x.color_grading.expect("ravine cfxs grading LUT");
        assert!(c.tag_name(lut).ends_with("default_16_color_grading"), "{}", c.tag_name(lut));
        let (b, n, dxgi) = crate::h4::bitmaps::load_volume_lut(&c, lut).expect("volume LUT");
        assert_eq!((n, dxgi, b.len()), (16, 87, 16 * 16 * 16 * 4));
        // corners: black, white, pure red at x = n-1 (BGRA texels, x fastest)
        assert_eq!(&b[..3], &[0, 0, 0]);
        assert_eq!(&b[(15 * 4) as usize..(15 * 4 + 3) as usize], &[0, 0, 254]);
        assert_eq!(&b[b.len() - 4..b.len() - 1], &[254, 254, 254]);
    }

    /// `apply_post`'s self-illum exposure mapping: E_si / E = 2^((1 - change) (pref - stops))
    /// through the renderer's ILLUM_SCALE law 2^((1 - s)(P' - e)) with e = log2(gain / (key 10)), key = 1.
    #[test]
    fn self_illum_exposure_mapping() {
        let (pref, change, stops) = (1.0f32, 0.2f32, 2.5f32);
        let p_prime = pref - 10f32.log2() + H4_EXPOSURE_SCALE.log2();
        let e = (stops.exp2() * H4_EXPOSURE_SCALE / 10.0).log2();
        let illum = ((1.0 - change) * (p_prime - e)).exp2();
        let want = ((1.0 - change) * (pref - stops)).exp2();
        assert!((illum - want).abs() < 1e-4, "{illum} vs {want}");
    }
}
