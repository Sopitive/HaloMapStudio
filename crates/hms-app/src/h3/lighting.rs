//! Halo 3 scene atmosphere: the `skya` (sky_atmosphere_parameters) tag -> the renderer's fog
//! uniform. #h3-fog
//!
//! Layout (Halo3/skya.xml, Halo3MCC scnr.xml / sbsp.xml):
//!   * skya +0x34 Atmosphere Settings (0xA4): flags u16 @0, name sid @4, sun pitch @8, heading
//!     @0xC, colour rgb @0x10, intensity @0x1C, sea level @0x20, rayleigh height scale @0x24,
//!     mie height scale @0x28, rayleigh multiplier @0x2C, mie multiplier @0x30, sun phase
//!     function (HG g) @0x34, desaturation @0x38, distance bias @0x3C, max fog thickness @0x40,
//!     beta_m rgb @0x44, beta_p rgb @0x50, beta_m theta prefix @0x5C, beta_p theta prefix @0x68
//!     (the engine's precomputed scattering coefficients, cache-baked), sheet density @0x74,
//!     full / half intensity height @0x78 / @0x7C, wind @0x80.
//!   * scnr +0x4F0 Atmosphere (0x8: name sid @0, setting index i16 @4); sbsp +0x94 Atmosphere
//!     Palette (0x8: name sid @0, setting index i16 @4), cluster element +0x19 atmosphere index.
//! The engine (halo3.dll, protomorph's port `atmosphere_fx.wgsl`): per pixel
//!   dist = min(max(|view| + distance_bias, 0), max_fog_thickness);
//!   dm/dp = dist x the exponential height integral of exp(-h / rayleigh_h) / exp(-h / mie_h);
//!   extinction = exp2(-(beta_m log2e . dm + beta_p log2e . dp)),
//!   inscatter = sun / (beta_m + beta_p) x (beta_m_theta (1 + c^2) + beta_p_theta HG(c, g)) x (1 - ext).
//! The renderer's fog module is Reach's two-band height fog (mesh.rs `apply_fog`), so the two
//! species are COLLAPSED onto its sky band (an approximation, stated): thickness = the
//! luminance of beta_m + beta_p, height scale = the beta-weighted mean of the two height
//! scales, base = sea level, colour = the side-lit inscatter (c = 0) chroma, distance bias and
//! max distance as authored. The directional (sun-phase) term is not carried.

use super::cache::H3Cache;
use crate::h4::cache::ByteRead;

pub const OFF_SKYA_SETTINGS: usize = 0x34;
pub const SKYA_SETTING_ELEM: usize = 0xA4;
pub const OFF_SCNR_ATMOSPHERE: usize = 0x4F0;
pub const OFF_SBSP_ATMOSPHERE_PALETTE: usize = 0x94;
pub const OFF_SBSP_CLUSTERS: usize = 0xC4;
pub const SBSP_CLUSTER_ELEM: usize = 0x118;
pub const OFF_CLUSTER_ATMOSPHERE: usize = 0x19;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct H3Atmosphere {
    pub name: String,
    pub flags: u16,
    pub sun_pitch: f32,
    pub sun_heading: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub sea_level: f32,
    pub rayleigh_height: f32,
    pub mie_height: f32,
    pub rayleigh_mult: f32,
    pub mie_mult: f32,
    pub phase_g: f32,
    pub desaturation: f32,
    pub distance_bias: f32,
    pub max_fog_thickness: f32,
    pub beta_m: [f32; 3],
    pub beta_p: [f32; 3],
    pub beta_m_theta: [f32; 3],
    pub beta_p_theta: [f32; 3],
}

impl H3Atmosphere {
    pub fn enabled(&self) -> bool { self.flags & 1 != 0 }
}

fn v3(d: &[u8], o: usize) -> [f32; 3] { [d.f32_at(o), d.f32_at(o + 4), d.f32_at(o + 8)] }

/// Every atmosphere setting of every `skya` tag in the cache (tag, setting index, values).
pub fn atmospheres(c: &H3Cache) -> Vec<(usize, usize, H3Atmosphere)> {
    let d = c.data();
    let mut out = Vec::new();
    for t in c.find_tags(b"skya") {
        let Some(m) = c.tag_meta(t) else { continue };
        let Some((n, o)) = c.block(m + OFF_SKYA_SETTINGS) else { continue };
        for i in 0..n.min(64) {
            let e = o + i * SKYA_SETTING_ELEM;
            out.push((t, i, H3Atmosphere {
                name: c.sid(d.u32_at(e + 4)), flags: d.u16_at(e), sun_pitch: d.f32_at(e + 8), sun_heading: d.f32_at(e + 0xC), color: v3(d, e + 0x10),
                intensity: d.f32_at(e + 0x1C), sea_level: d.f32_at(e + 0x20), rayleigh_height: d.f32_at(e + 0x24), mie_height: d.f32_at(e + 0x28),
                rayleigh_mult: d.f32_at(e + 0x2C), mie_mult: d.f32_at(e + 0x30), phase_g: d.f32_at(e + 0x34), desaturation: d.f32_at(e + 0x38),
                distance_bias: d.f32_at(e + 0x3C), max_fog_thickness: d.f32_at(e + 0x40), beta_m: v3(d, e + 0x44), beta_p: v3(d, e + 0x50),
                beta_m_theta: v3(d, e + 0x5C), beta_p_theta: v3(d, e + 0x68),
            }));
        }
    }
    out
}

/// The atmosphere setting the map renders with: the scnr Atmosphere block's setting (its
/// first entry's index into the sky's `skya` settings: construct -> #9 `bsp_fog`, sea level
/// -200 with a -150 wu distance bias, so nothing inside the map fogs - the interior stays
/// clear as in-game) when it names one, else setting 0 of the scenario's `skya` when the map
/// carries no selector at all (guardian, riverworld: every cluster's atmosphere index is -1).
/// Per-cluster blending (the engine weighs the neighbouring clusters' settings) is not done:
/// the shipped MP maps name at most one.
pub fn scene_atmosphere(c: &H3Cache) -> Option<H3Atmosphere> {
    let all = atmospheres(c);
    let d = c.data();
    let scnr = c.tag_meta(*c.find_tags(b"scnr").first()?)?;
    match c.block(scnr + OFF_SCNR_ATMOSPHERE) {
        Some((n, o)) if n > 0 => {
            let idx = d.i16_at(o + 4);
            all.into_iter().find(|(_, i, _)| *i as i64 == idx as i64).map(|(_, _, a)| a).filter(|a| a.enabled())
        }
        _ => all.into_iter().next().map(|(_, _, a)| a).filter(|a| a.enabled()),
    }
}

/// Luminance (BT.709).
fn lum(c: [f32; 3]) -> f32 { 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] }

impl H3Atmosphere {
    /// The sun radiance the atmosphere scatters (skya flags bit 1 = override the sky's sun with
    /// the setting's colour / intensity: the engine keeps the colour's chroma and sets its
    /// luminance to `intensity`, protomorph `get_sun_parameters`). Without the override the sky
    /// model's own sun would apply; its light list is not decoded, so the same rule is used.
    pub fn sun_rgb(&self) -> [f32; 3] {
        let l = lum(self.color).max(1e-4);
        let k = self.intensity / l;
        [self.color[0] * k, self.color[1] * k, self.color[2] * k]
    }

    /// The renderer's 28-float fog uniform in the HALO 3 slot map (mesh.rs `h3_fog`, picked by
    /// the `atm3.w == 2` sentinel): the ENGINE two-species law, not Reach's band fog.
    ///
    /// Ported from `c_atmosphere_fog_interface::set_constant @ 0x1803AE530` as protomorph's
    /// `atmosphere_fog_interface.rs` / `atmosphere_fx.wgsl` record it. The three places the
    /// runtime transforms the authored tag values (and this therefore must too):
    ///   * `beta_m` / `beta_p` are scaled by `log2(e)` (the shader integrates with `exp2`),
    ///   * `beta_p_angular` is pre-multiplied by `(1 - g^2)` (the HG normaliser),
    ///   * `slot6 = 2g`, `slot2.w = g + 1` (the shader rebuilds `g + 1` from `2g`),
    ///   * the sun is carried as `sun_intensity / (beta_m + beta_p)` per channel.
    /// `scene_gain` = the scene's bake gain (the fixed-exposure stand-in every lit lane carries),
    /// so the inscatter sits in the same units as the lit map and the sky.
    ///
    /// `max_lum` caps the inscatter luminance (scene units): the fixed-exposure stand-in cannot
    /// adapt to a haze brighter than the lit map the way the engine's meter does (construct's
    /// `bsp_fog` scatters 7x the mean lit surface), so the normalised sun is scaled down to the
    /// white point (1 / exposure) when the side-lit inscatter would clip - an approximation,
    /// stated, and the ONLY one left in this path.
    pub fn fog_uniform(&self, scene_gain: f32, max_lum: f32) -> [f32; 28] {
        const LOG2E: f32 = 1.442695;
        let mut f = [0.0f32; 28];
        let bm = self.beta_m;
        let bp = self.beta_p;
        let tot = [bm[0] + bp[0], bm[1] + bp[1], bm[2] + bp[2]];
        if !(lum(tot) > 1e-9) { return f; }
        let g = self.phase_g.clamp(-0.995, 0.995);
        let sun = self.sun_rgb();
        let mut norm = [0.0f32; 3];
        for k in 0..3 { norm[k] = if tot[k] > 1e-9 { (sun[k] / tot[k]) * scene_gain } else { 0.0 }; }
        // the engine's (1 - g^2) prefix on the Mie angular term (set_constant slot5)
        let hg_prefix = 1.0 - g * g;
        let bpa = [self.beta_p_theta[0] * hg_prefix, self.beta_p_theta[1] * hg_prefix, self.beta_p_theta[2] * hg_prefix];
        // the side-lit (cos 0) inscatter: the representative colour the sky gradient grades to,
        // and the luminance the exposure cap is solved against
        let hg0 = (1.0 + g * g).max(1e-4).powf(-1.5);
        let mut col = [0.0f32; 3];
        for k in 0..3 { col[k] = (norm[k] * (self.beta_m_theta[k] + bpa[k] * hg0)).max(0.0); }
        let l = lum(col);
        if max_lum > 0.0 && l > max_lum {
            let s = max_lum / l;
            for c in col.iter_mut() { *c *= s; }
            for c in norm.iter_mut() { *c *= s; }
        }
        // the sun DIRECTION the phase functions use: pitch / heading (engine Z-up, pointing the
        // way the light travels - the shader dots it against the view vector camera->fragment)
        let (sp, cp2) = (self.sun_pitch.sin(), self.sun_pitch.cos());
        let (sh, ch) = (self.sun_heading.sin(), self.sun_heading.cos());
        let sun_dir = [-ch * cp2, -sh * cp2, -sp];

        f[0..3].copy_from_slice(&col);           // atm0.xyz: representative inscatter (sky gradient)
        f[3] = self.distance_bias;               // atm0.w
        f[4..7].copy_from_slice(&norm);          // atm1.xyz: sun / (beta_m + beta_p) x gain
        f[7] = self.max_fog_thickness.max(0.0);  // atm1.w
        for k in 0..3 { f[8 + k] = bm[k] * LOG2E; }   // atm2.xyz
        f[11] = self.sea_level;                  // atm2.w: reference_datum_plane
        for k in 0..3 { f[12 + k] = bp[k] * LOG2E; }  // atm3.xyz
        f[15] = 2.0;                             // atm3.w: the Halo 3 atmosphere sentinel
        f[16..19].copy_from_slice(&self.beta_m_theta); // atm4.xyz
        f[19] = self.mie_height.max(1e-4);       // atm4.w
        f[20..23].copy_from_slice(&bpa);         // atm5.xyz
        f[23] = self.rayleigh_height.max(1e-4);  // atm5.w
        f[24..27].copy_from_slice(&sun_dir);     // atm6.xyz
        f[27] = 2.0 * g;                         // atm6.w
        f
    }
}

// ---------------------------------------------------------------------------------------------
// camera_fx_settings (`cfxs`): the map's own exposure + bloom authoring. #h3-expo
// ---------------------------------------------------------------------------------------------
//
// Halo 3 has the same `cfxs` tag group Reach and Halo 4 do, and the scenario names one at
// `scnr +0x660` ("default camera fx"); the globals (`matg +0x4CC`) name the fallback
// `globals\default`. The meta layout was read field-by-field off the shipped caches (every
// halo3 + halo3odst map; the probe test below prints it) and sits BETWEEN Reach's and Halo 4's:
//
//   +0x00  EXPOSURE (`s_real_exposure_parameter`)            0x20
//          flags @0x00 (bit2 0x04 = "auto-adjust target"), manual exposure @0x04,
//          max change @0x08, blend @0x0C, band min / max stops @0x10 / @0x14,
//          AUTO-EXPOSURE SCREEN BRIGHTNESS @0x18 **log2-encoded** (the metering key),
//          delay @0x1C.
//   +0x20  AUTO-EXPOSURE SENSITIVITY (flags, value)          0x08   (value @0x24)
//   +0x28  a second instant parameter, PURPOSE UNKNOWN       0x08   (Halo 4 has no counterpart;
//          authored 0 on most maps, 0.1/0.2/1.0 on a few campaign + Lockout settings)
//   +0x30  HIGHLIGHT BLOOM {bloom point}  (flags, value, max change, blend)   0x10 (value @0x34)
//   +0x40  INHERENT BLOOM                                                     0x10 (value @0x44)
//   +0x50  BLOOM INTENSITY                                                    0x10 (value @0x54)
//   +0x60 / +0x70 / +0x80  BLOOM LARGE / MEDIUM / SMALL COLOUR (flags, rgb)   0x10 each
//   +0x90  BLOOM SIZE                                                         0x10
//
// Halo 4 is this layout MINUS the +0x28 struct and PLUS a fourth bloom parameter (self-illum),
// which is why its offsets run 8 lower from the bloom block on. Every real parameter's flags
// bit0 = "use default": the engine then takes that parameter from the globals' default cfxs
// instead of the map's own value (the same rule as Reach / Halo 4).

/// Offsets inside the Halo 3 `cfxs` meta (see the block comment above).
pub const OFF_SCNR_DEFAULT_CFXS: usize = 0x660;
pub const OFF_MATG_DEFAULT_CFXS: usize = 0x4CC;
/// `cfxs` exposure flags bit 2: auto-adjust the target (clear = fixed `manual` exposure).
pub const CFXS_AUTO_ADJUST: u16 = 0x04;

/// A Halo 3 map's resolved `camera_fx_settings`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H3CameraFx {
    pub name: String,
    pub flags: u16,
    /// Manual exposure in stops (used when `CFXS_AUTO_ADJUST` is clear).
    pub manual: f32,
    /// The auto-exposure band, in stops (tag order: min, max).
    pub band: [f32; 2],
    /// log2 of the metering target ("auto-exposure screen brightness"); `key()` decodes it.
    pub screen_brightness: f32,
    /// Auto-exposure sensitivity: the meter's blend between the log of the mean and the mean of
    /// the log (0 = geometric mean, 1 = arithmetic mean).
    pub sensitivity: f32,
    pub bloom_point: f32,
    pub bloom_inherent: f32,
    pub bloom_intensity: f32,
    /// Bloom large / medium / small colour.
    pub bloom_colors: [[f32; 3]; 3],
    /// Bit k set = parameter k took the globals default ("use default" flag): 0 exposure,
    /// 1 sensitivity, 2 point, 3 inherent, 4 intensity, 5..7 colours.
    pub defaulted: u8,
}

impl H3CameraFx {
    /// The metering key (the exposed luminance the meter converges to): the tag stores log2 of
    /// it. A map that authors nothing usable (chillout: -16.6 = 1e-5) falls back to the globals
    /// default's 0.035.
    pub fn key(&self) -> f32 {
        let k = self.screen_brightness.exp2();
        if k.is_finite() && k > 1e-3 { k } else { 0.035 }
    }
    pub fn auto(&self) -> bool { self.flags & CFXS_AUTO_ADJUST != 0 }
    /// The band as (min, max) stops, ordered and widened if the tag is degenerate.
    pub fn stops_band(&self) -> (f32, f32) {
        let (lo, hi) = (self.band[0].min(self.band[1]), self.band[0].max(self.band[1]));
        if hi - lo < 1e-3 || !lo.is_finite() || !hi.is_finite() { (-6.0, 6.0) } else { (lo, hi) }
    }
}

/// Read one `cfxs` meta, taking any parameter whose "use default" flag (bit 0) is set from
/// `dflt` instead (the engine's rule; `dflt` is the globals default cfxs meta).
fn read_cfxs(c: &H3Cache, m: usize, dflt: Option<usize>, name: String) -> H3CameraFx {
    let d = c.data();
    // one real parameter: its own meta offset unless bit0 of the flags says "use default"
    let mut defaulted = 0u8;
    let mut src = |bit: u8, off: usize| -> usize {
        if d.u32_at(m + off) & 1 != 0 {
            if let Some(f) = dflt { defaulted |= 1 << bit; return f + off; }
        }
        m + off
    };
    let e = src(0, 0x00);
    let sens = src(1, 0x20);
    let pt = src(2, 0x30);
    let inh = src(3, 0x40);
    let int = src(4, 0x50);
    let cols = [src(5, 0x60), src(6, 0x70), src(7, 0x80)];
    let rgb = |o: usize| [d.f32_at(o + 4), d.f32_at(o + 8), d.f32_at(o + 12)];
    H3CameraFx {
        name,
        flags: d.u16_at(e),
        manual: d.f32_at(e + 4),
        band: [d.f32_at(e + 0x10), d.f32_at(e + 0x14)],
        screen_brightness: d.f32_at(e + 0x18),
        sensitivity: d.f32_at(sens + 4).clamp(0.0, 1.0),
        bloom_point: d.f32_at(pt + 4),
        bloom_inherent: d.f32_at(inh + 4),
        bloom_intensity: d.f32_at(int + 4),
        bloom_colors: [rgb(cols[0]), rgb(cols[1]), rgb(cols[2])],
        defaulted,
    }
}

/// The map's camera FX: `scnr +0x660`, with every "use default" parameter taken from the
/// globals' default (`matg +0x4CC`, `globals\default`). None when the scenario names none.
pub fn load_camera_fx(c: &H3Cache) -> Option<H3CameraFx> {
    let scnr = c.tag_meta(*c.find_tags(b"scnr").first()?)?;
    let dflt = c.find_tags(b"matg").first().copied()
        .and_then(|t| c.tag_meta(t))
        .and_then(|m| c.tag_ref_of(m + OFF_MATG_DEFAULT_CFXS, b"cfxs"))
        .and_then(|t| c.tag_meta(t));
    let tag = c.tag_ref_of(scnr + OFF_SCNR_DEFAULT_CFXS, b"cfxs")?;
    let m = c.tag_meta(tag)?;
    Some(read_cfxs(c, m, dflt, c.tag_name(tag).to_string()))
}

/// The engine's neutral exposure: `render_exposure = 2^stops * tone_curve_white_point / 1.4938016`
/// (halo3.dll `0x1802847A8`, the player-view exposure setup, writing `player_view + 0x2B8`), and
/// `tone_curve_white_point` is 1.0 in the shipped `c_screen_postprocess` initialiser
/// (`.data:0x1808AD040 + 0x2C`). So a Halo 3 map at 0 stops renders at 0.669, not 1.0.
pub const H3_WHITE_POINT: f32 = 1.0;
pub const H3_EXPOSURE_SCALE: f32 = H3_WHITE_POINT / 1.4938016;

/// The absolute render exposure of a cfxs-band stops value (the engine's `g_exposure.r`).
pub fn exposure_of_stops(stops: f32) -> f32 { stops.exp2() * H3_EXPOSURE_SCALE }

impl H3CameraFx {
    /// The exposure band as absolute gains (the engine's `render_exposure` at each end).
    pub fn gain_range(&self) -> (f32, f32) {
        let (lo, hi) = self.stops_band();
        (exposure_of_stops(lo), exposure_of_stops(hi))
    }
}

/// Push a Halo 3 map's post-process state to the renderer: the engine TONE CURVE and the
/// authored bloom curve / colours. The one place the headless renderer and the GUI load share,
/// so the two can never drift (the Reach and Halo 4 paths have the same shape: `apply_post`).
///
/// Tone curve: Halo 3's composite ends on a CUBIC, not Reach's bare gamma-2 `sqrt`
/// (`copy_accumulation_target @ halo3.dll 0x18029A1C0`; see `set_h3_tone_curve`). HMS put every
/// game through Reach's encode, which costs Halo 3 about 1.22x everywhere below white.
///
/// BLOOM IS DELIBERATELY NOT PUSHED HERE, and the reason is worth keeping. Every Halo 3 map
/// authors a bloom curve in this same `cfxs` (point / inherent / intensity + the three pyramid
/// colours), and HMS runs all of them on the built-in Reach defaults (0.1 / 0.0 / 0.3). But the
/// two engines shape bloom with DIFFERENT laws, and the authored numbers do not transfer:
///   * HMS / Reach (BLOOM_WGSL `fs_bright`): `S = 2 E (point lum(E) + inherent)` - `point` is a
///     MULTIPLIER, so a bigger value means more bloom.
///   * Halo 3 (`halo3.dll postprocess_player_view @0x18029C058` -> shader 48
///     `downsample_4x4_block_bloom`): `I = luma(E)`, `bright = max(I inherent, I - point v16)`,
///     `out.rgb = E bright / I` with `v16 = max(1, 2^(stops anti_bloom)) white / 1.4938016` -
///     `point` is a THRESHOLD, so a bigger value means LESS bloom.
/// Guardian authors point 1.5; fed to the Reach law that is 15x the default highlight weight,
/// and a measured A/B put the frame mean up 1.35x on bloom alone - an artefact, not a fix. The
/// authored values can only be used once the Halo 3 bright pass itself is ported. (The `+0x28`
/// instant parameter is almost certainly the `anti_bloom` that law needs: the runtime orders
/// sensitivity at `c_exposure+0x118` and anti-bloom at `+0x11C`, matching the tag's `+0x20` and
/// `+0x28`, and the shipped values - 0 on most maps, 1.0 on Lockout - fit a 0..1 switch.)
pub fn apply_post(renderer: &hms_render::SceneRenderer, queue: &eframe::wgpu::Queue, _x: Option<&H3CameraFx>) {
    renderer.set_h3_tone_curve(queue, Some(H3_WHITE_POINT));
}

/// The tags whose meta holds a reference to a `skya` tag (a probe helper: the referrer scan).
pub fn skya_referrers(c: &H3Cache) -> Vec<(usize, usize)> {
    let d = c.data();
    let mut out = Vec::new();
    let skyas = c.find_tags(b"skya");
    if skyas.is_empty() { return out; }
    // tag metas in file order, to attribute a hit to the tag whose meta precedes it
    let mut metas: Vec<(usize, usize)> = (0..c.tag_count()).filter_map(|t| c.tag_meta(t).map(|m| (m, t))).collect();
    metas.sort();
    let mut i = 0;
    while i + 16 <= d.len() {
        if &d[i..i + 4] == b"ayks" {
            let datum = d.u32_at(i + 12);
            let idx = (datum & 0xFFFF) as usize;
            if skyas.contains(&idx) {
                let owner = metas.iter().rev().find(|(m, _)| *m <= i).map(|(_, t)| *t);
                if let Some(o) = owner { out.push((o, idx)); }
            }
        }
        i += 4;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::cache::{maps_dir, H3Cache};
    use super::*;

    /// guardian / riverworld resolve their setting 0, construct its selector's #9 `bsp_fog`
    /// (200 wu below the map, biased 150 wu out: the interior stays clear), and the fog uniform
    /// is finite with a positive thickness.
    #[test]
    fn scene_atmosphere_selection() {
        for (map, expect) in [("guardian.map", "forest_base"), ("riverworld.map", "haze_level"), ("construct.map", "bsp_fog")] {
            let Some(c) = maps_dir().map(|d| d.join(map)).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { continue };
            let a = scene_atmosphere(&c);
            assert_eq!(a.as_ref().map(|a| a.name.as_str()), Some(expect), "{map}: {a:?}");
            if map == "construct.map" { let a = a.as_ref().unwrap(); assert!(a.sea_level <= -150.0 && a.distance_bias <= -100.0); }
            if let Some(a) = a {
                // the HALO 3 slot map (mesh.rs `h3_fog`): atm3.w == 2 is the sentinel that picks
                // the engine two-species path over Reach's band fog
                let f = a.fog_uniform(14.0, 4.0);
                assert!(f.iter().all(|v| v.is_finite()), "{map}: {f:?}");
                assert_eq!(f[15], 2.0, "{map}: atm3.w must be the Halo 3 atmosphere sentinel");
                assert_eq!(f[3], a.distance_bias, "{map}: atm0.w = distance_bias");
                assert_eq!(f[7], a.max_fog_thickness.max(0.0), "{map}: atm1.w = max_fog_thickness");
                assert_eq!(f[11], a.sea_level, "{map}: atm2.w = reference_datum_plane");
                assert_eq!(f[19], a.mie_height.max(1e-4), "{map}: atm4.w = mie_height_scale");
                assert_eq!(f[23], a.rayleigh_height.max(1e-4), "{map}: atm5.w = rayleigh_height_scale");
                assert!((f[27] - 2.0 * a.phase_g.clamp(-0.995, 0.995)).abs() < 1e-6, "{map}: atm6.w = 2g");
                // beta x log2(e) (the shader integrates with exp2)
                for k in 0..3 {
                    assert!((f[8 + k] - a.beta_m[k] * 1.442695).abs() < 1e-7, "{map}: atm2.xyz = beta_m x log2e");
                    assert!((f[12 + k] - a.beta_p[k] * 1.442695).abs() < 1e-7, "{map}: atm3.xyz = beta_p x log2e");
                }
                // the sun direction is a unit vector
                let l = (f[24] * f[24] + f[25] * f[25] + f[26] * f[26]).sqrt();
                assert!((l - 1.0).abs() < 1e-4, "{map}: sun dir not unit ({l})");
                // the inscatter is capped at the stated white point
                assert!(lum([f[0], f[1], f[2]]) <= 4.0 + 1e-3, "{map}: inscatter over the cap");
                eprintln!("{map}: '{}' inscatter {:?} bias {:.1} max {:.0} ref {:.1} mie_h {:.1} ray_h {:.1} g {:.2}",
                    a.name, &f[0..3], f[3], f[7], f[11], f[19], f[23], f[27] * 0.5);
            }
        }
    }

    /// #h3-expo The `cfxs` decode, pinned on the shipped caches: the scenario names one at
    /// `scnr +0x660`, its key is the exp2 of a NEGATIVE log2 field (never the raw value), the
    /// band is a real ordered stops range, and the bloom parameters are in 0..2. Snowbound is the
    /// "use default" case (all three bloom flags set -> the globals' 1.2 / 0.2 / 0.2); Sandbox is
    /// the MANUAL case (exposure flags without `CFXS_AUTO_ADJUST`).
    #[test]
    fn camera_fx_decodes() {
        let Some(dir) = maps_dir() else { return };
        let mut seen = 0;
        for (map, key, band, bloom) in [
            ("guardian.map", 0.045, [1.25, 2.5], [1.5, 0.1, 0.1]),
            ("construct.map", 0.045, [-0.5, 1.0], [1.2, 0.1, 0.5]),
            ("riverworld.map", 0.035, [0.0, 3.0], [1.5, 0.1, 0.1]),
            ("isolation.map", 0.030, [0.0, 3.0], [0.5, 0.01, 0.45]),
            ("snowbound.map", 0.035, [0.0, 2.5], [1.2, 0.2, 0.2]),
        ] {
            let p = dir.join(map);
            if !p.is_file() { continue }
            let c = H3Cache::open(&p).expect("open");
            let x = load_camera_fx(&c).unwrap_or_else(|| panic!("{map}: no cfxs"));
            seen += 1;
            assert!((x.key() - key).abs() < 1e-4, "{map}: key {} (log2 {})", x.key(), x.screen_brightness);
            assert!(x.screen_brightness < 0.0, "{map}: the key is stored log2-encoded");
            assert_eq!(x.stops_band(), (band[0], band[1]), "{map}: band");
            let got = [x.bloom_point, x.bloom_inherent, x.bloom_intensity];
            for k in 0..3 { assert!((got[k] - bloom[k]).abs() < 1e-5, "{map}: bloom {got:?} != {bloom:?}"); }
            assert_eq!(x.bloom_colors[0], [1.0; 3], "{map}: the bloom colours are white on every MP map");
            // snowbound takes all three bloom parameters from globals\default
            assert_eq!(x.defaulted & 0x1C != 0, map == "snowbound.map", "{map}: use-default flags {:#x}", x.defaulted);
            assert!(x.auto(), "{map}: auto-adjust");
            // the band bounds the engine's own render_exposure
            let (lo, hi) = x.gain_range();
            assert!(lo > 0.0 && hi > lo && hi < 64.0, "{map}: gain range {lo}..{hi}");
        }
        // Sandbox authors MANUAL exposure (no auto-adjust bit) - the one shipped map that does
        let p = dir.join("sandbox.map");
        if p.is_file() {
            let x = load_camera_fx(&H3Cache::open(&p).expect("open")).expect("sandbox cfxs");
            assert!(!x.auto(), "sandbox: flags {:#x} should have no auto-adjust bit", x.flags);
            seen += 1;
        }
        if seen == 0 { eprintln!("skip: no halo3 maps installed"); }
    }

    /// Probe (print-only): guardian / riverworld / construct - the skya tags, who references
    /// them, every atmosphere setting's values, and the scnr / sbsp atmosphere selectors.
    #[test]
    fn atmosphere_probe() {
        for map in ["guardian.map", "riverworld.map", "construct.map"] {
            let Some(c) = maps_dir().map(|d| d.join(map)).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { continue };
            let d = c.data();
            eprintln!("== {map}: skya tags {:?}", c.find_tags(b"skya").iter().map(|t| c.tag_name(*t).to_string()).collect::<Vec<_>>());
            for (owner, skya) in skya_referrers(&c) {
                eprintln!("  ref from {} [{}] -> {}", c.tag_name(owner), String::from_utf8_lossy(&c.tag_class(owner).unwrap_or(*b"????")), c.tag_name(skya));
            }
            for (t, i, a) in atmospheres(&c) {
                eprintln!("  {} #{i} '{}' flags {:#x} sun pitch {} heading {} colour {:?} int {} sea {} rayH {} mieH {} rayM {} mieM {} g {} desat {} bias {} maxT {} beta_m {:?} beta_p {:?} bm_theta {:?} bp_theta {:?}",
                    c.tag_name(t).rsplit('\\').next().unwrap_or(""), a.name, a.flags, a.sun_pitch, a.sun_heading, a.color, a.intensity, a.sea_level, a.rayleigh_height, a.mie_height,
                    a.rayleigh_mult, a.mie_mult, a.phase_g, a.desaturation, a.distance_bias, a.max_fog_thickness, a.beta_m, a.beta_p, a.beta_m_theta, a.beta_p_theta);
            }
            let scnr = c.tag_meta(c.find_tags(b"scnr")[0]).unwrap();
            if let Some((n, o)) = c.block(scnr + OFF_SCNR_ATMOSPHERE) {
                eprintln!("  scnr atmosphere block: {}", (0..n).map(|i| format!("{}:{} raw {:02x?}", c.sid(d.u32_at(o + i * 8)), d.i16_at(o + i * 8 + 4), &d[o + i * 8..o + i * 8 + 8])).collect::<Vec<_>>().join(" "));
            }
            for sb in c.find_tags(b"sbsp") {
                let sm = c.tag_meta(sb).unwrap();
                if let Some((n, o)) = c.block(sm + OFF_SBSP_ATMOSPHERE_PALETTE) {
                    eprintln!("  sbsp atmosphere palette: {}", (0..n).map(|i| format!("{}:{} raw {:02x?}", c.sid(d.u32_at(o + i * 8)), d.i16_at(o + i * 8 + 4), &d[o + i * 8..o + i * 8 + 8])).collect::<Vec<_>>().join(" "));
                }
                if let Some((n, o)) = c.block(sm + OFF_SBSP_CLUSTERS) {
                    let mut hist = std::collections::BTreeMap::new();
                    for i in 0..n { *hist.entry(d.u8_at(o + i * SBSP_CLUSTER_ELEM + OFF_CLUSTER_ATMOSPHERE) as i8).or_insert(0) += 1; }
                    eprintln!("  sbsp clusters {n}: atmosphere index histogram {hist:?}");
                }
            }
        }
    }
}
