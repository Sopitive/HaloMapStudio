//! Halo 4 / H2A Forge special-FX screen effects: the placed FX object -> `sefc`
//! (area_screen_effect) chain decoded against the pure-Rust `H4Cache`, then composed with the
//! ENGINE-GENERIC accumulate / colour-matrix math in `crate::screenfx` (shared with the Reach
//! path). Only the tag DECODE differs between the engines; the composition and the renderer upload
//! (`crate::screenfx::push_to_renderer`) are identical.
//!
//! ## Object -> sefc chain (RE'd on ca_forge_ravine.map, cross-checked on the DLC/H2A caches)
//! The Forge "Special FX" palette (`scnr` map-variant palette entry `ff_special_fx`, 6 variants:
//! Colorblind / Next Gen / Juicy / Nova / Olde Timey / Pen and Ink) places `bloc` objects
//! `objects\levels\shared\screen_fx_orb\fx\<name>`. Each carries ONE effect attachment:
//! ```
//! obje +0x118  attachments block (0x20 B/elem, same stride as Reach): tagref @0 {fourcc, datum @0xC}
//!   -> class 'effe'  objects\levels\shared\screen_fx_orb\fx\<name>
//!      (H4's effe is a tiny stub; the orb's screen effect is NOT reached through effe events)
//! ```
//! The `sefc` (`objects\levels\shared\screen_effects\<x>`) is not referenced by a fixed field of
//! the placed `bloc` (5 of 6 carry a stray `sefc` ref in their variable tail at +0x224, but
//! Colorblind does not, and the tail offset is not stable). The reliable, uniform link is by the
//! attachment effect's leaf NAME: the `sefc` whose leaf is `<name>` or `reach_<name>`
//! (nova -> reach_nova, gloomy -> reach_gloomy; the rest are identity). Verified: all six orbs
//! resolve to exactly their Reach-equivalent sefc.
//!
//! ## `sefc` tag + `s_single_screen_effect_definition` (H4)
//! sefc tag header (16 B): u16 flags @0, u16 hidden @2, tag block {count @4, ptr @8}, pad @0xC,
//! then a stray `effe` tag ref @0x10 (the orb's effect twin - unused here).
//! The element is **0x118 B** (Reach's 0xF0 + an inserted 16-B tag ref at +0x6C + a u32) and its
//! layout is Reach's shifted by +0x14 from +0x6C on:
//! ```
//!  +0x00 name sid   +0x04 u16 flags   +0x06 u16 hidden   +0x08 max distance
//!  +0x0C distance falloff (20-B data ref: size @0, ptr @0xC)   +0x20 delay   +0x24 lifetime
//!  +0x28 time falloff   +0x3C angle falloff   +0x50/54 object fn sids   +0x58 object falloff
//!  +0x6C (16-B tag ref, secondary screen effect; null on the orbs)   +0x7C u32
//!  +0x80 exposure boost   +0x84 exposure deboost   +0x88 hue left   +0x8C hue right
//!  +0x90 saturation       +0x94 desaturation       +0x98 contrast enhance
//!  +0x9C gamma enhance     +0xA0 gamma reduce        +0xA4 bright noise   +0xA8 dark noise
//!  +0xAC color filter rgb  +0xB8 color floor rgb     +0xC4 tron   +0xC8 motion suck  +0xCC dir xyz
//!  +0xD8 hblur  +0xDC vblur  +0xE0 vision mode  +0xE4 hud fade  +0xE8 fov in  +0xEC fov out
//!  +0xF0 screen shake      +0xF4 shader effect tag ref
//! ```
//! (verified: Colorblind desat 1.0 / contrast 0.2 / gamma+ 0.3 / filter 1,1,1; Gloomy deboost 1.0 /
//! desat 0.7 / contrast 0.3 / gamma+/- 0.3 / noise .02 / filter .837,.767,.709 / floor .013 - the
//! exact Reach values.) H4 Forge maps carry NO scenario-default screen effect (the scnr camera-fx
//! `cfxs` ref at +0x70c has no adjacent sefc), so the active set is the placed orbs alone.

use super::cache::{ByteRead, H4Cache};
use crate::screenfx::{compose, ActiveScreenFx, ScreenFxAccum, ScreenFxDef, ScreenFxReals, ScreenFxRender};

/// sefc element stride (bytes).
const SEFC_ELEM: usize = 0x118;
/// obje attachments block offset + element stride (H4 base-obje layout).
const OFF_OBJE_ATTACHMENTS: usize = 0x118;
const ATTACH_ELEM: usize = 0x20;

fn rdf(d: &[u8], o: usize) -> f32 { let v = d.f32_at(o); if v.is_finite() { v } else { 0.0 } }

/// Leaf (last `\`/`/` component) of a tag path.
fn leaf(name: &str) -> &str { name.rsplit(['\\', '/']).next().unwrap_or(name) }

/// A CONSTANT mapping-function falloff (20-B data ref at element+`off`: size @0, ptr @0xC). Returns
/// the constant when the blob is a constant function (type 1, kind 0), else None (non-constant, or
/// unreadable). Mirrors the Reach `screenfx::falloff_const`.
fn falloff_const(c: &H4Cache, elem: usize, off: usize) -> Option<f32> {
    let d = c.data();
    let size = d.i32_at(elem + off);
    let ptr = d.u32_at(elem + off + 0xC);
    if size < 8 || ptr == 0 || ptr == 0xCDCD_CDCD { return None; }
    let b = c.meta_off(ptr)?;
    if b + 8 > d.len() { return None; }
    if d.u8_at(b) == 1 && d.u8_at(b + 2) == 0 { Some(rdf(d, b + 4)) } else { None }
}

/// Decode every element of a Halo 4 / H2A `sefc` tag. Empty when unreadable.
pub fn decode_sefc(c: &H4Cache, sefc_tag: usize) -> Vec<ScreenFxDef> {
    let mut out = Vec::new();
    let Some(m) = c.tag_meta(sefc_tag) else { return out };
    let d = c.data();
    let Some((count, e0)) = c.block(m + 4) else { return out };
    let name = c.tag_name(sefc_tag).to_ascii_lowercase();
    for i in 0..count.min(16) {
        let e = e0 + i * SEFC_ELEM;
        if e + SEFC_ELEM > d.len() { break; }
        let f = |o: usize| rdf(d, e + o);
        let reals = ScreenFxReals {
            exposure_boost: f(0x80), exposure_deboost: f(0x84), hue_left: f(0x88), hue_right: f(0x8C),
            saturation: f(0x90), desaturation: f(0x94), contrast_enhance: f(0x98), gamma_enhance: f(0x9C),
            gamma_reduce: f(0xA0), bright_noise: f(0xA4), dark_noise: f(0xA8),
            color_filter: [f(0xAC), f(0xB0), f(0xB4)], color_floor: [f(0xB8), f(0xBC), f(0xC0)],
            tron: f(0xC4), motion_suck: f(0xC8), motion_dir: [f(0xCC), f(0xD0), f(0xD4)],
            hblur: f(0xD8), vblur: f(0xDC), vision_mode: f(0xE0), hud_fade: f(0xE4),
            fov_in: f(0xE8), fov_out: f(0xEC), screen_shake: f(0xF0),
        };
        out.push(ScreenFxDef {
            tag: sefc_tag as u32, index: i as u32, name: name.clone(),
            name_sid: d.u32_at(e), flags: d.u16_at(e + 4), hidden_flags: d.u16_at(e + 6),
            max_distance: f(8), delay: f(0x20), lifetime: f(0x24),
            falloff_const: [falloff_const(c, e, 0xC), falloff_const(c, e, 0x28), falloff_const(c, e, 0x3C), falloff_const(c, e, 0x58)],
            reals,
            shader_effect: d.u32_at(e + 0xF4 + 0xC),
        });
    }
    out
}

/// The distinct `sefc` tags a placed Forge object spawns. Walks the obje +0x118 attachments: a
/// direct `sefc` attachment is taken as-is; an `effe` attachment resolves to the `sefc` whose leaf
/// name equals the effe's leaf or `reach_<leaf>` (the screen_fx_orb naming). Deduplicated.
pub fn object_sefc_tags(c: &H4Cache, obje_tag: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let Some(m) = c.tag_meta(obje_tag) else { return out };
    let Some((n, e0)) = c.block(m + OFF_OBJE_ATTACHMENTS) else { return out };
    for k in 0..n.min(32) {
        let el = e0 + k * ATTACH_ELEM;
        if let Some(si) = c.tag_ref_of(el, b"sefc") {
            if !out.contains(&si) { out.push(si); }
        } else if let Some(ei) = c.tag_ref_of(el, b"effe") {
            if let Some(si) = sefc_for_effe_leaf(c, leaf(c.tag_name(ei))) {
                if !out.contains(&si) { out.push(si); }
            }
        }
    }
    out
}

/// The `sefc` whose leaf name is `l` or `reach_<l>` (the screen_fx_orb effect->sefc pairing).
fn sefc_for_effe_leaf(c: &H4Cache, l: &str) -> Option<usize> {
    let reach = format!("reach_{l}");
    c.find_tags(b"sefc").into_iter().find(|&s| { let sl = leaf(c.tag_name(s)); sl == l || sl == reach })
}

/// Runtime falloff of a placed orb's element (mirrors `crate::screenfx`'s private rule). The orbs
/// author flag b5 "disable camera falloffs" + constant-1 curves, so their falloff is 1 map-wide;
/// an element without b1/b5 is a proximity effect the camera is outside of (0).
fn element_falloff(def: &ScreenFxDef) -> f32 {
    if def.flags & 0x1 != 0 { return 0.0; }         // debug disable
    if def.hidden_flags & 0x2 == 0 { return 0.0; }  // runtime "valid" gate
    if def.flags & 0x22 == 0 { return 0.0; }        // needs b1 (outside radius) or b5 (no falloffs)
    let mut s = 1.0;
    if let Some(cst) = def.falloff_const[3] { s *= cst.clamp(0.0, 1.0); }
    if def.lifetime > 1e-4 {
        if let Some(cst) = def.falloff_const[1] { s *= cst.clamp(0.0, 1.0); }
    }
    s
}

/// Per-map decode cache: obje tag -> its sefc tags, sefc tag -> decoded elements.
#[derive(Default)]
pub struct H4ScreenFxCache {
    obj_to_sefc: std::collections::HashMap<usize, Vec<usize>>,
    defs: std::collections::HashMap<usize, Vec<ScreenFxDef>>,
}

impl H4ScreenFxCache {
    pub fn clear(&mut self) { self.obj_to_sefc.clear(); self.defs.clear(); }

    fn sefc_for_object(&mut self, c: &H4Cache, obje_tag: usize) -> &[usize] {
        self.obj_to_sefc.entry(obje_tag).or_insert_with(|| object_sefc_tags(c, obje_tag)).as_slice()
    }

    fn defs(&mut self, c: &H4Cache, sefc_tag: usize) -> &[ScreenFxDef] {
        self.defs.entry(sefc_tag).or_insert_with(|| decode_sefc(c, sefc_tag)).as_slice()
    }
}

/// Build the active screen-effect set for a Halo 4 / H2A map: the DISTINCT sefc tags the placed
/// Forge FX objects spawn (H4 has no scenario-default sefc). `objects` = the placed objects' obje
/// tags. Returns a `crate::screenfx::ActiveScreenFx` so the shared `push_to_renderer` applies it.
pub fn active_screen_fx(c: &H4Cache, cache: &mut H4ScreenFxCache, objects: impl Iterator<Item = usize>) -> ActiveScreenFx {
    let mut forge_tags: Vec<u32> = Vec::new();
    let mut tag_ids: Vec<usize> = Vec::new();
    for t in objects {
        for &s in cache.sefc_for_object(c, t) {
            if !tag_ids.contains(&s) { tag_ids.push(s); forge_tags.push(s as u32); }
        }
    }
    // sort by u32 for a stable key (order does not matter to the per-term max accumulate)
    let mut order: Vec<usize> = (0..tag_ids.len()).collect();
    order.sort_by_key(|&i| forge_tags[i]);
    let sorted_ids: Vec<usize> = order.iter().map(|&i| tag_ids[i]).collect();
    let mut sorted_u32: Vec<u32> = order.iter().map(|&i| forge_tags[i]).collect();
    sorted_u32.dedup();

    let mut acc = ScreenFxAccum::identity();
    let mut forge_names = Vec::new();
    for &s in &sorted_ids {
        let defs = cache.defs(c, s).to_vec();
        let nm = defs.first().map(|d| crate::screenfx::short_name(&d.name)).unwrap_or_else(|| format!("{s:#x}"));
        forge_names.push(nm);
        for d in defs { acc.accumulate(&d.reals, element_falloff(&d)); }
    }
    let render = if sorted_ids.is_empty() { ScreenFxRender::identity() } else { compose(&acc) };
    ActiveScreenFx {
        default_tag: 0,
        forge_tags: sorted_u32,
        forge_names,
        accum: acc,
        render,
        // H4 has no scenario default: "Forge FX off" == identity (the map's cfxs base look, applied
        // separately by lighting.rs, is unchanged).
        render_default_only: ScreenFxRender::identity(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(map: &str) -> Option<H4Cache> {
        let p = crate::h4::cache::maps_dir()?.join(map);
        p.exists().then(|| H4Cache::open(&p).ok()).flatten()
    }

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-3 }

    /// The decode matches the known (Reach-identical) FX values on the shipped Halo 4 cache.
    #[test]
    fn ravine_fx_decode_matches_known_values() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let leafmap = |want: &str| c.find_tags(b"sefc").into_iter().find(|&s| c.tag_name(s).ends_with(want));

        // Colorblind: desat 1.0, contrast 0.2, gamma+ 0.3, filter 1,1,1, else 0.
        let cb = leafmap("screen_effects\\colorblind").expect("colorblind sefc");
        let d = decode_sefc(&c, cb);
        assert_eq!(d.len(), 1);
        let r = &d[0].reals;
        assert!(approx(r.desaturation, 1.0) && approx(r.contrast_enhance, 0.2) && approx(r.gamma_enhance, 0.3), "{r:?}");
        assert!(approx(r.color_filter[0], 1.0) && approx(r.color_filter[1], 1.0) && approx(r.color_filter[2], 1.0));
        assert!(approx(r.exposure_boost, 0.0) && approx(r.exposure_deboost, 0.0) && approx(r.saturation, 0.0));
        assert_eq!(d[0].flags, 0x20);
        assert_eq!(d[0].hidden_flags & 0x3, 0x3);
        assert!(approx(d[0].max_distance, 225.0));

        // Gloomy (reach_gloomy): deboost 1.0, desat 0.7, contrast 0.3, gamma+/- 0.3, noise .02,
        // filter .837/.767/.709, floor .013.
        let gl = leafmap("screen_effects\\reach_gloomy").expect("gloomy sefc");
        let r = &decode_sefc(&c, gl)[0].reals;
        assert!(approx(r.exposure_deboost, 1.0) && approx(r.desaturation, 0.7) && approx(r.contrast_enhance, 0.3));
        assert!(approx(r.gamma_enhance, 0.3) && approx(r.gamma_reduce, 0.3));
        assert!(approx(r.bright_noise, 0.02) && approx(r.dark_noise, 0.02));
        assert!(approx(r.color_filter[0], 0.837) && approx(r.color_filter[1], 0.767) && approx(r.color_filter[2], 0.709), "{:?}", r.color_filter);
        assert!(approx(r.color_floor[0], 0.013) && approx(r.color_floor[2], 0.013));

        // Juicy: saturation 1.0, gamma+ 0.5, filter 1/.922/.831.
        let ju = leafmap("screen_effects\\juicy").expect("juicy sefc");
        let r = &decode_sefc(&c, ju)[0].reals;
        assert!(approx(r.saturation, 1.0) && approx(r.gamma_enhance, 0.5));
        assert!(approx(r.color_filter[1], 0.922) && approx(r.color_filter[2], 0.831), "{:?}", r.color_filter);

        // Nova: boost 6.5, contrast 0.1, gamma+ 0.2, filter .056.
        let nv = leafmap("screen_effects\\reach_nova").expect("nova sefc");
        let r = &decode_sefc(&c, nv)[0].reals;
        assert!(approx(r.exposure_boost, 6.5) && approx(r.contrast_enhance, 0.1) && approx(r.gamma_enhance, 0.2), "{r:?}");
        assert!(approx(r.color_filter[0], 0.056), "{:?}", r.color_filter);
    }

    /// The obje -> sefc resolver finds each FX orb's screen effect.
    #[test]
    fn ravine_object_sefc_resolves_all_six() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        for (fx, want) in [("colorblind", "colorblind"), ("gloomy", "reach_gloomy"), ("juicy", "juicy"),
                           ("nova", "reach_nova"), ("olde_timey", "olde_timey"), ("pen_and_ink", "pen_and_ink")] {
            let obje = c.find_tags(b"bloc").into_iter().find(|&t| c.tag_name(t).ends_with(&format!("fx\\{fx}"))).expect(fx);
            let sefcs = object_sefc_tags(&c, obje);
            assert_eq!(sefcs.len(), 1, "{fx}: {sefcs:?}");
            assert!(c.tag_name(sefcs[0]).ends_with(&format!("screen_effects\\{want}")), "{fx} -> {}", c.tag_name(sefcs[0]));
        }
    }

    /// End-to-end: a scene containing the Colorblind orb composes to a full desaturate (S=0), and
    /// two identical orbs do not stack (engine per-term max).
    #[test]
    fn ravine_active_compose_colorblind() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let obje = c.find_tags(b"bloc").into_iter().find(|&t| c.tag_name(t).ends_with("fx\\colorblind")).unwrap();
        let mut cache = H4ScreenFxCache::default();
        let one = active_screen_fx(&c, &mut cache, [obje].into_iter());
        assert_eq!(one.forge_names, vec!["colorblind".to_string()]);
        // colorblind: S = 1 + sat - desat = 1 + 0 - 1 = 0 -> greyscale. apply a colour, expect luma.
        let out = one.render.apply([0.8, 0.3, 0.1]);
        assert!(approx(out[0], out[1]) && approx(out[1], out[2]), "not greyscale: {out:?}");
        // two identical orbs == one (max(a,a)=a)
        let two = active_screen_fx(&c, &mut cache, [obje, obje].into_iter());
        assert_eq!(one.render, two.render);
        assert_eq!(two.forge_tags.len(), 1);
    }

    /// H2A cache (engine-identical to H4): the FX orbs decode the same when the map is present.
    #[test]
    fn h2a_fx_decode() {
        let Some(dir) = crate::h4::cache::h2a_maps_dir() else { return };
        // any H2A cache carrying the shared screen_fx tags
        let Some(c) = std::fs::read_dir(&dir).ok().and_then(|rd| rd.filter_map(|e| e.ok())
            .filter_map(|e| H4Cache::open(&e.path()).ok())
            .find(|c| c.find_tags(b"sefc").iter().any(|&s| c.tag_name(s).ends_with("screen_effects\\colorblind")))) else { return };
        let cb = c.find_tags(b"sefc").into_iter().find(|&s| c.tag_name(s).ends_with("screen_effects\\colorblind")).unwrap();
        let r = &decode_sefc(&c, cb)[0].reals;
        assert!(approx(r.desaturation, 1.0) && approx(r.contrast_enhance, 0.2), "H2A colorblind {r:?}");
    }

    /// Scan every shipped H4 hopper variant for one that places an FX orb (prints matches).
    /// HMS_MVAR_DIR overrides the variant folder; needs the caches.
    #[test]
    #[ignore]
    fn scan_variants_for_fx() {
        let Some(maps) = crate::h4::cache::maps_dir() else { eprintln!("no maps"); return };
        let vdir = std::env::var("HMS_MVAR_DIR").map(std::path::PathBuf::from)
            .unwrap_or_else(|_| maps.parent().unwrap().join("hopper_map_variants"));
        // cache maps by map_id
        let mut caches: std::collections::HashMap<u32, (std::path::PathBuf, H4Cache)> = std::collections::HashMap::new();
        for e in std::fs::read_dir(&maps).into_iter().flatten().filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().map_or(false, |x| x == "map") {
                if let Ok(c) = H4Cache::open(&p) {
                    if let Some(id) = crate::mapcat::read_map_id(&p) { caches.entry(id).or_insert((p, c)); }
                }
            }
        }
        let mut found = 0;
        for e in std::fs::read_dir(&vdir).into_iter().flatten().filter_map(|e| e.ok()) {
            let p = e.path();
            if p.extension().map_or(true, |x| x != "mvar") { continue; }
            if !crate::h4::mvar::is_h4_variant(&p) { continue; }
            let Ok(v) = crate::h4::mvar::parse_h4_variant(&p) else { continue };
            let Some((_, c)) = caches.get(&v.map_id) else { continue };
            let pal = crate::h4::palette::palette(c);
            let mut fx = Vec::new();
            for o in &v.objects {
                let Some(q) = o.quota else { continue };
                if let Some((_, pv)) = pal.resolve(Some(q), o.variant) {
                    if let Some(tag) = pv.tag {
                        if c.tag_name(tag).contains("screen_fx_orb") { fx.push(c.tag_name(tag).rsplit('\\').next().unwrap().to_string()); }
                    }
                }
            }
            if !fx.is_empty() {
                eprintln!("{} (map {}): FX {:?}", p.file_name().unwrap().to_string_lossy(), v.map_id, fx);
                found += 1;
            }
        }
        eprintln!("variants with FX orbs: {found}");
    }

    /// Print dump for manual inspection (HMS_H4_MAP overrides the map).
    #[test]
    #[ignore]
    fn dump() {
        let map = std::env::var("HMS_H4_MAP").unwrap_or_else(|_| "ca_forge_ravine.map".into());
        let Some(c) = open(&map) else { eprintln!("{map} missing"); return };
        for s in c.find_tags(b"sefc") {
            if !c.tag_name(s).contains("screen_effects") { continue; }
            for d in decode_sefc(&c, s) { eprintln!("{}", crate::screenfx::describe(&d)); }
        }
    }
}
