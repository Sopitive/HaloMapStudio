//! #hud-raster  TEST-ONLY. A CPU rasteriser for egui's draw list, so the HUD can be looked at
//! without a window: `play_hud_draw::paint_play_hud` runs against a headless `egui::Context`, the
//! shapes it emits are tessellated by egui itself, and this file composites the resulting meshes
//! into an RGBA buffer with the same equation egui's shader uses -- premultiplied source over the
//! destination, `out = src + dst * (1 - src.a)`, in sRGB bytes, vertex colour times texture sample.
//! Additive widgets are alpha-0 premultiplied colours (see `reach-chud-render-fields`), so they
//! land here exactly as they would on the GPU.
//!
//! The textures are the REAL decoded bitmaps and the REAL Reach font atlases: they arrive in the
//! frame's `textures_delta` and are kept across frames, because `reach_hud::hud_art` uploads a
//! `chdt`'s art once per context.

use eframe::egui;
use std::collections::HashMap;

pub struct Raster {
    pub w: usize,
    pub h: usize,
    /// Premultiplied RGBA, 0..255 as f32.
    px: Vec<[f32; 4]>,
    textures: HashMap<egui::TextureId, (usize, usize, Vec<egui::Color32>)>,
}

impl Raster {
    pub fn new(w: usize, h: usize) -> Raster {
        Raster { w, h, px: vec![[0.0; 4]; w * h], textures: HashMap::new() }
    }

    /// Start a new image of `w` x `h`, keeping every texture ingested so far.
    pub fn reset(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.px = vec![[0.0; 4]; w * h];
    }

    /// Fill the frame with a vertical gradient between two sRGB colours (a stand-in for the map).
    pub fn background(&mut self, top: [u8; 3], bottom: [u8; 3]) {
        for y in 0..self.h {
            let t = y as f32 / (self.h - 1).max(1) as f32;
            for x in 0..self.w {
                let c = |k: usize| top[k] as f32 * (1.0 - t) + bottom[k] as f32 * t;
                self.px[y * self.w + x] = [c(0), c(1), c(2), 255.0];
            }
        }
    }

    pub fn ingest(&mut self, d: &egui::TexturesDelta) {
        for (id, delta) in &d.set {
            let (size, pixels): ([usize; 2], Vec<egui::Color32>) = match &delta.image {
                egui::ImageData::Color(c) => (c.size, c.pixels.clone()),
                egui::ImageData::Font(f) => (f.size, f.srgba_pixels(None).collect()),
            };
            match delta.pos {
                None => {
                    self.textures.insert(*id, (size[0], size[1], pixels));
                }
                Some([x0, y0]) => {
                    if let Some((tw, _, tp)) = self.textures.get_mut(id) {
                        for y in 0..size[1] {
                            for x in 0..size[0] {
                                tp[(y0 + y) * *tw + x0 + x] = pixels[y * size[0] + x];
                            }
                        }
                    }
                }
            }
        }
    }

    fn sample(&self, id: egui::TextureId, u: f32, v: f32) -> [f32; 4] {
        let Some((tw, th, tp)) = self.textures.get(&id) else { return [255.0; 4] };
        // Bilinear, texel centres at +0.5, clamped.
        let x = (u * *tw as f32 - 0.5).clamp(0.0, (*tw - 1) as f32);
        let y = (v * *th as f32 - 0.5).clamp(0.0, (*th - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(*tw - 1), (y0 + 1).min(*th - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| {
            let c = tp[yy * *tw + xx];
            [c.r() as f32, c.g() as f32, c.b() as f32, c.a() as f32]
        };
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        std::array::from_fn(|k| {
            (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy
        })
    }

    fn tri(&mut self, id: egui::TextureId, v: [&egui::epaint::Vertex; 3], clip: egui::Rect) {
        let (p0, p1, p2) = (v[0].pos, v[1].pos, v[2].pos);
        let area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
        if area.abs() < 1e-6 {
            return;
        }
        let min_x = p0.x.min(p1.x).min(p2.x).max(clip.min.x).max(0.0).floor() as i32;
        let max_x = p0.x.max(p1.x).max(p2.x).min(clip.max.x).min(self.w as f32).ceil() as i32;
        let min_y = p0.y.min(p1.y).min(p2.y).max(clip.min.y).max(0.0).floor() as i32;
        let max_y = p0.y.max(p1.y).max(p2.y).min(clip.max.y).min(self.h as f32).ceil() as i32;
        for y in min_y..max_y {
            for x in min_x..max_x {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((p1.x - px) * (p2.y - py) - (p2.x - px) * (p1.y - py)) / area;
                let w1 = ((p2.x - px) * (p0.y - py) - (p0.x - px) * (p2.y - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                    continue;
                }
                let u = w0 * v[0].uv.x + w1 * v[1].uv.x + w2 * v[2].uv.x;
                let vv = w0 * v[0].uv.y + w1 * v[1].uv.y + w2 * v[2].uv.y;
                let t = self.sample(id, u, vv);
                let col = |k: usize| -> f32 {
                    let c = |i: usize| match k {
                        0 => v[i].color.r(),
                        1 => v[i].color.g(),
                        2 => v[i].color.b(),
                        _ => v[i].color.a(),
                    } as f32;
                    w0 * c(0) + w1 * c(1) + w2 * c(2)
                };
                let s: [f32; 4] = std::array::from_fn(|k| t[k] * col(k) / 255.0);
                let d = &mut self.px[y as usize * self.w + x as usize];
                let inv = 1.0 - s[3] / 255.0;
                for k in 0..4 {
                    d[k] = (s[k] + d[k] * inv).min(255.0);
                }
            }
        }
    }

    /// Composite one egui frame. `ppp` is the context's pixels-per-point (1.0 for these runs).
    pub fn draw(&mut self, ctx: &egui::Context, out: egui::FullOutput) {
        self.ingest(&out.textures_delta);
        let ppp = out.pixels_per_point;
        for prim in ctx.tessellate(out.shapes, ppp) {
            if let egui::epaint::Primitive::Mesh(m) = prim.primitive {
                for t in m.indices.chunks_exact(3) {
                    let v = [&m.vertices[t[0] as usize], &m.vertices[t[1] as usize], &m.vertices[t[2] as usize]];
                    self.tri(m.texture_id, v, prim.clip_rect);
                }
            }
        }
    }

    /// The composited sRGB of one pixel, 0..255.
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        let p = self.px[y * self.w + x];
        [p[0], p[1], p[2]]
    }

    /// Count of pixels whose colour differs from `bg` -- "something was painted here".
    pub fn painted_in(&self, r: egui::Rect, bg: &Raster) -> usize {
        let mut n = 0;
        for y in (r.min.y.max(0.0) as usize)..(r.max.y.min(self.h as f32) as usize) {
            for x in (r.min.x.max(0.0) as usize)..(r.max.x.min(self.w as f32) as usize) {
                let (a, b) = (self.px[y * self.w + x], bg.px[y * self.w + x]);
                if (0..3).any(|k| (a[k] - b[k]).abs() > 6.0) {
                    n += 1;
                }
            }
        }
        n
    }

    pub fn save(&self, path: &std::path::Path) {
        if let Some(d) = path.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let f = std::fs::File::create(path).expect("create png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("png header");
        let data: Vec<u8> = self.px.iter().flat_map(|p| [p[0] as u8, p[1] as u8, p[2] as u8]).collect();
        w.write_image_data(&data).expect("png data");
    }
}

#[cfg(test)]
mod frames {
    //! `cargo test --release -p hms-app hud_raster::frames -- --ignored --nocapture --test-threads=1`
    //! writes the PNGs listed by the test to `$HUD_OUT` (default: the session scratchpad `hud/`).

    use super::*;
    use crate::play_hud::PlayHud;
    use crate::play_hud_draw::{paint_play_hud, HudFrame};
    use crate::reach_unit_hud::SeatHuds;
    use crate::reach_seats::UnitSeats;
    use crate::reach_hud as rh;

    fn out_dir() -> std::path::PathBuf {
        std::env::var("HUD_OUT").map(Into::into).unwrap_or_else(|_| {
            "/tmp/claude-1000/-home-sopitive/05268b5e-9d50-4b4a-9c35-93b66f9dc5fa/scratchpad/hud".into()
        })
    }

    fn find(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, group: &str, tail: &str) -> Option<u32> {
        let want = format!("\\{tail}");
        native.tags_by_group(cache, group).into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(&want)))
    }

    #[test]
    #[ignore = "writes PNGs; needs the forge_halo cache"]
    fn render_hud_frames() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let ctx = egui::Context::default();
        let mut r = Raster::new(8, 8);
        let dir = out_dir();
        let biped = find(native, cache, "bipd", "spartans\\spartans").expect("spartan");
        let mut written = Vec::new();

        let mut shoot = |r: &mut Raster, name: &str, w: usize, h: usize, hud: &PlayHud, f: &HudFrame| -> Vec<String> {
            r.reset(w, h);
            r.background([128, 140, 150], [92, 84, 70]);
            let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
            let mut lines = Vec::new();
            let out = ctx.run(egui::RawInput { screen_rect: Some(vp), ..Default::default() }, |ctx| {
                let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hud")));
                lines = paint_play_hud(ctx, &painter, vp, native, cache, hud, f);
            });
            r.draw(&ctx, out);
            let p = dir.join(format!("{name}_{w}x{h}.png"));
            r.save(&p);
            written.push(p.file_name().unwrap().to_string_lossy().to_string());
            lines
        };

        let foot = |weapon: u32| HudFrame {
            weapon_tag: weapon, weapon_usable: true, hud_on: true, crosshair_on: true, key_label: "E".into(), ..Default::default()
        };
        for (w, h) in [(1920, 1080), (1280, 720), (2560, 1080)] {
            for wname in ["assault_rifle", "magnum", "plasma_pistol", "rocket_launcher", "dmr", "shotgun"] {
                let Some(weap) = find(native, cache, "weap", wname) else { continue };
                let mut hud = PlayHud::default();
                hud.bind(native, cache, biped);
                hud.set_weapon(native, cache, weap, 0);
                for state in ["full", "half", "low"] {
                    if let Some(v) = hud.vitality.as_mut() {
                        v.restore();
                        match state {
                            "half" => v.shield = v.shield_max().unwrap_or(0.0) * 0.5,
                            "low" => {
                                v.shield = 0.0;
                                v.health = v.health_max().unwrap_or(0.0) * 0.25;
                            }
                            _ => {}
                        }
                    }
                    if state == "low" {
                        if let Some(a) = hud.ammo.as_mut() {
                            a.loaded = (a.mag.rounds_loaded_maximum / 5).max(1);
                        }
                    }
                    let lines = shoot(&mut r, &format!("foot_{wname}_{state}"), w, h, &hud, &foot(weap));
                    if w == 1920 && state == "full" {
                        eprintln!("--- {wname}:\n  {}", lines.join("\n  "));
                    }
                }
            }
            // Empty hands.
            let mut hud = PlayHud::default();
            hud.bind(native, cache, biped);
            hud.set_weapon(native, cache, 0, 0);
            shoot(&mut r, "foot_unarmed_full", w, h, &hud, &foot(0));

            // ---- vehicles ----
            let ar = find(native, cache, "weap", "assault_rifle").unwrap_or(0);
            let mut hud = PlayHud::default();
            hud.bind(native, cache, biped);
            hud.set_weapon(native, cache, ar, 0);
            for (name, tail, seat) in [
                ("warthog_driver", "warthog\\warthog", 0),
                ("warthog_passenger", "warthog\\warthog", 1),
                ("warthog_gunner", "warthog_chaingun\\warthog_chaingun", 0),
                ("warthog_gauss", "warthog_gauss\\warthog_gauss", 0),
                ("warthog_rocket", "warthog_rocket\\warthog_rocket", 0),
                ("ghost", "ghost\\ghost", 0),
                ("banshee", "banshee\\banshee", 0),
                ("mongoose_driver", "mongoose\\mongoose", 0),
                ("falcon_pilot", "falcon\\falcon", 0),
                ("scorpion_driver", "scorpion\\scorpion", 0),
                ("scorpion_coax", "scorpion_anti_infantry\\scorpion_anti_infantry", 0),
                ("wraith_driver", "wraith\\wraith", 0),
            ] {
                let Some(v) = find(native, cache, "vehi", tail) else { continue };
                let Some(us) = UnitSeats::read(native, cache, v) else { continue };
                let huds = SeatHuds::for_seat(native, cache, v, &us.seats[seat]);
                let f = HudFrame {
                    weapon_tag: ar,
                    weapon_usable: us.seats[seat].allows_weapons(),
                    seat_huds: Some(huds),
                    riding: Some((seat, name.into())),
                    hud_on: true,
                    crosshair_on: true,
                    key_label: "E".into(),
                    ..Default::default()
                };
                shoot(&mut r, &format!("veh_{name}"), w, h, &hud, &f);
            }
        }
        eprintln!("HUD PNGs in {}:\n{}", dir.display(), written.join("\n"));
    }

    /// #use-hold  **THE ACTION PROMPT SITS WHERE THE PLAYER HUD AUTHORS IT.** `paint_play_hud` with
    /// a prompt draws the `player_training` collection of `ui\chud\spartan`: anchor 4 (centre) at
    /// `(0, +164)` virtual units, 1.2 scale, the widget's own authored colour -- at 1280x720 and
    /// 1920x1080 -- and with no prompt draws NOTHING of it. Writes `action_prompt_*.png`.
    #[test]
    #[ignore = "writes PNGs; needs the forge_halo cache"]
    fn the_action_prompt_sits_lower_centre_in_the_authored_style() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let ctx = egui::Context::default();
        let biped = find(native, cache, "bipd", "spartans\\spartans").expect("spartan");
        let mut hud = PlayHud::default();
        hud.bind(native, cache, biped);
        let skin = crate::reach_chud::read_globals(native, cache).and_then(|g| g.default_skin().cloned()).expect("skin");
        let t = rh::chud_by_path(native, cache, rh::PLAYER_CHUD_SPARTAN).unwrap();
        let chud = rh::read_chud(native, cache, t).unwrap();
        let col = chud.get(rh::TRAINING_COLLECTION).expect("player_training");
        let text_w = col.widgets.iter().find(|w| matches!(&w.art, rh::WidgetArt::Text { input, .. } if input == rh::CHUD_TRAINING_TEXT)).expect("text widget");
        let cp = col.base.placement().unwrap();
        assert_eq!((cp.anchor, cp.offset), (rh::ANCHOR_CENTER, [0.0, 164.0]), "the authored placement this gate is written against");
        let want_colour = rh::widget_color(&text_w.base, Some(&skin));
        let mut r = Raster::new(8, 8);
        for (w, h) in [(1280usize, 720usize), (1920, 1080)] {
            let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
            let vs = rh::VirtualScreen::from_skin(vp, &skin);
            let mut shoot = |prompt: Option<&str>, name: Option<&str>| -> (Vec<rh::WidgetDraw>, Vec<String>) {
                let f = HudFrame {
                    weapon_usable: true, hud_on: true, crosshair_on: true, key_label: "E".into(),
                    prompt: prompt.map(str::to_string), ..Default::default()
                };
                r.reset(w, h);
                r.background([128, 140, 150], [92, 84, 70]);
                rh::set_widget_log(true);
                let mut lines = Vec::new();
                let out = ctx.run(egui::RawInput { screen_rect: Some(vp), ..Default::default() }, |ctx| {
                    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hud")));
                    lines = paint_play_hud(ctx, &painter, vp, native, cache, &hud, &f);
                });
                let log = rh::take_widget_log();
                rh::set_widget_log(false);
                r.draw(&ctx, out);
                if let Some(n) = name {
                    r.save(&out_dir().join(format!("{n}_{w}x{h}.png")));
                }
                (log.into_iter().filter(|d| d.collection == rh::TRAINING_COLLECTION).collect(), lines)
            };
            let (log, _) = shoot(Some("Hold E to drive Warthog"), Some("action_prompt"));
            let text = log.iter().find(|d| d.kind == "text" && d.outcome == "drawn").unwrap_or_else(|| panic!("the prompt text was not drawn: {log:#?}"));
            assert_eq!(text.text.as_deref(), Some("Hold E to drive Warthog"));
            assert_eq!(text.color, Some(want_colour), "the authored widget colour");
            let rect = text.rect.expect("text rect");
            // Centre of the screen + 164 virtual units down; the text's own origin is its centre.
            let want_y = vp.center().y + 164.0 * vs.vpx;
            assert!((rect.center().y - want_y).abs() < 14.0 * vs.vpx.max(1.0), "{w}x{h}: prompt at y {} wanted ~{want_y}", rect.center().y);
            assert!((rect.center().x - vp.center().x).abs() < 3.0, "{w}x{h}: prompt not horizontally centred: {}", rect.center().x);
            assert!(rect.center().y > vp.center().y && rect.center().y < vp.bottom(), "lower half of the screen");
            let (none, lines) = shoot(None, None);
            assert!(none.iter().all(|d| d.outcome != "drawn"), "{w}x{h}: with no prompt nothing of the collection draws: {none:#?}");
            assert!(!lines.iter().any(|l| l.starts_with("Action prompt")), "{w}x{h}: no prompt, yet the HUD reports painting one: {lines:?}");
        }
    }
}

#[cfg(test)]
mod audit {
    //! #hud-audit  GATES on the painter's output, each with a saboteur in the round that added it.
    //! `cargo test --release -p hms-app hud_raster::audit -- --ignored --nocapture --test-threads=1`

    use super::*;
    use crate::reach_hud as rh;

    type Open = (std::sync::MutexGuard<'static, ()>, std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle);

    fn open() -> Option<Open> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    fn bounds(shapes: &[egui::epaint::ClippedShape]) -> Option<egui::Rect> {
        fn walk(s: &egui::Shape, out: &mut Option<egui::Rect>) {
            match s {
                egui::Shape::Mesh(m) => {
                    for v in &m.vertices {
                        let r = egui::Rect::from_min_max(v.pos, v.pos);
                        *out = Some(out.map_or(r, |u| u.union(r)));
                    }
                }
                egui::Shape::Vec(v) => v.iter().for_each(|x| walk(x, out)),
                _ => {}
            }
        }
        let mut out = None;
        shapes.iter().for_each(|c| walk(&c.shape, &mut out));
        out
    }

    struct Painted {
        log: Vec<rh::WidgetDraw>,
        union: Option<egui::Rect>,
        vs: rh::VirtualScreen,
    }

    /// Paint one named collection of one chud through the production painter, headlessly.
    fn paint(ctx: &egui::Context, native: &hms_native::NativeDll, cache: hms_native::CacheHandle, chud: &str, coll: &str, v: &rh::HudValues, repeat: usize) -> Painted {
        let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1920.0, 1080.0));
        let skin = crate::reach_chud::read_globals(native, cache).and_then(|g| g.default_skin().cloned());
        let vs = rh::VirtualScreen::from_skin(vp, skin.as_ref().unwrap());
        let t = rh::chud_by_path(native, cache, chud).unwrap_or_else(|| panic!("no {chud}"));
        let art = rh::hud_art(ctx, native, cache, t).unwrap();
        let ci = art.chud.collections.iter().position(|c| c.base.name == coll).unwrap_or_else(|| panic!("no {coll}"));
        rh::set_widget_log(true);
        let out = ctx.run(egui::RawInput { screen_rect: Some(vp), ..Default::default() }, |ctx| {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("a")));
            rh::paint_collection(ctx, &painter, &vs, &art, ci, skin.as_ref(), v, repeat);
        });
        let log = rh::take_widget_log();
        rh::set_widget_log(false);
        Painted { log, union: bounds(&out.shapes), vs }
    }

    fn authored(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, chud: &str, coll: &str, widget: &str) -> (rh::Placement, rh::Placement, [f32; 2], u32) {
        let t = rh::chud_by_path(native, cache, chud).unwrap();
        let c = rh::read_chud(native, cache, t).unwrap();
        let cl = c.get(coll).unwrap();
        let w = cl.get(widget).unwrap();
        let rh::WidgetArt::Bitmap { sprite, flags, .. } = &w.art else { panic!("not a bitmap") };
        (*cl.base.placement().unwrap(), *w.base.placement().unwrap(), sprite.size, *flags)
    }

    /// Where the painter must put a meter's visible part at `fill`, from the tag alone: the
    /// collection's anchor + offset, the widget's offset, the sprite pivoted by `widget origin`,
    /// and the level taken from the pivot-side (`rising`: left part, else right part).
    fn expected_meter(vs: &rh::VirtualScreen, cp: &rh::Placement, wp: &rh::Placement, size: [f32; 2], fill: f32, rising: bool) -> egui::Rect {
        let a = vs.anchor(cp.anchor).unwrap();
        let (px, py) = (a.x + (cp.offset[0] + wp.offset[0]) * vs.vpx, a.y + (cp.offset[1] + wp.offset[1]) * vs.vpx);
        let (w, h) = (size[0] * wp.scale[0].abs() * vs.vpx, size[1] * wp.scale[1].abs() * vs.vpx);
        let left = px - wp.origin[0] * w;
        let top = py - wp.origin[1] * h;
        if rising {
            egui::Rect::from_min_size(egui::pos2(left, top), egui::vec2(w * fill, h))
        } else {
            egui::Rect::from_min_size(egui::pos2(left + w * (1.0 - fill), top), egui::vec2(w * fill, h))
        }
    }

    /// **METERS.** A meter shows the part of its shape the level names, counted from the pivot
    /// side: the shield bar (ramp rises away from its pivot) a LEFT part, the health bar and the
    /// assault rifle's magazine (ramps fall) a RIGHT part. Positions come from the authored
    /// placement and pivot, not from the painter.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn a_meter_shows_the_part_of_its_shape_the_level_names() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ctx = egui::Context::default();
        let near = |a: egui::Rect, b: egui::Rect| (a.min - b.min).abs().max_elem() < 0.6 && (a.max - b.max).abs().max_elem() < 0.6;
        let mut checked = 0;
        for (chud, coll, widget, rising) in [
            ("ui\\chud\\spartan", "shield_meter", "meter", true),
            ("ui\\chud\\spartan", "health_meter", "health_meter", false),
            ("ui\\chud\\assault_rifle", "ammo_area", "meter", false),
        ] {
            let (cp, wp, size, _) = authored(native, cache, chud, coll, widget);
            for fill in [1.0f32, 0.5, 0.25] {
                let v = rh::HudValues {
                    weapon_tag: Some(1),
                    shield: rh::Field::recorded(fill),
                    health: rh::Field::recorded(fill),
                    loaded: rh::Field::recorded((fill * 32.0) as i32),
                    loaded_max: Some(32),
                    ..Default::default()
                };
                let p = paint(&ctx, native, cache, chud, coll, &v, 0);
                let d = p.log.iter().find(|d| d.widget == widget && d.outcome == "drawn").unwrap_or_else(|| panic!("{coll}/{widget} not drawn at {fill}"));
                let want = expected_meter(&p.vs, &cp, &wp, size, fill, rising);
                assert!(near(d.rect.unwrap(), want), "{coll}/{widget} at {fill}: drew {:?}, authored pivot and ramp say {want:?}", d.rect);
                checked += 1;
            }
        }
        assert_eq!(checked, 9);
    }

    /// **MIRRORING.** `chud_widget_bitmap_flags` bit 0: the shield (right half, pivot at the
    /// centre) and the health bar (left half) are drawn reflected about the screen centre, so the
    /// pair is symmetric. The union of every mesh the two collections emit is centred on the
    /// anchor, and at half level it is still symmetric.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn a_mirrored_widget_is_drawn_on_both_sides_of_its_point() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ctx = egui::Context::default();
        for coll in ["shield_meter", "health_meter"] {
            for fill in [1.0f32, 0.4] {
                let v = rh::HudValues { shield: rh::Field::recorded(fill), health: rh::Field::recorded(fill), ..Default::default() };
                let p = paint(&ctx, native, cache, "ui\\chud\\spartan", coll, &v, 0);
                let u = p.union.unwrap_or_else(|| panic!("{coll} emitted nothing"));
                let cx = p.vs.anchor(rh::ANCHOR_TOP_CENTER).unwrap().x;
                assert!((u.center().x - cx).abs() < 1.5, "{coll} at {fill}: spans {:.0}..{:.0}, centre {:.1}, the anchor is {cx:.1}", u.min.x, u.max.x, u.center().x);
                assert!(u.width() > 60.0, "{coll} at {fill} is only {:.0} pt wide", u.width());
            }
        }
    }

    /// **PER-ELEMENT SPRITES.** The grenade icons are `auto permute external input1`: element k of
    /// the datasource draws sequence `sequence + k`, so the plasma slot is not the frag picture.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn each_grenade_slot_draws_its_own_sequence() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ctx = egui::Context::default();
        let v = rh::HudValues { frags: rh::Field::recorded(2), plasmas: rh::Field::recorded(1), ..Default::default() };
        let (_, wp, _, flags) = authored(native, cache, "ui\\chud\\spartan", "grenades", "unhilight");
        assert!(flags & rh::BITMAP_AUTO_PERMUTE_INPUT1 != 0, "unhilight is a permuted widget");
        let _ = wp;
        let t = rh::chud_by_path(native, cache, "ui\\chud\\spartan").unwrap();
        let c = rh::read_chud(native, cache, t).unwrap();
        let rh::WidgetArt::Bitmap { sprite, sequence, .. } = &c.get("grenades").unwrap().get("unhilight").unwrap().art else { panic!() };
        let sprites: Vec<_> = (0..2)
            .map(|r| {
                let p = paint(&ctx, native, cache, "ui\\chud\\spartan", "grenades", &v, r);
                p.log.iter().filter(|d| d.outcome == "drawn" && d.kind == "bitmap").filter_map(|d| d.sprite).collect::<Vec<_>>()
            })
            .collect();
        assert!(!sprites[0].is_empty() && !sprites[1].is_empty(), "both slots must draw an icon: {sprites:?}");
        // The unselected icon of slot 1 is sequence + 1 of the same sheet, not slot 0's picture.
        let want = crate::reach_chud::read_sprite(native, cache, sprite.bitmap_tag, *sequence as usize + 1, 0).unwrap();
        assert!(sprites[1].iter().any(|s| s.2 == want.size && s.1 == want.bitmap_index), "slot 1 did not draw sequence+1: {sprites:?} vs {:?}", want.size);
        assert!(sprites[0] != sprites[1], "the two slots drew the same sprites");
    }

    /// **THE RETICLES EXIST AND ARE COLOURED.** Rasterise a real frame: the rocket launcher (its
    /// reticle lives in `crosshair_locking`, gated on `NOT 115`), the assault rifle's mirrored ring
    /// and a Warthog driver's `v_driver` chevron must put BLUE pixels (the skin's `crosshair
    /// normal`, #2494FF) around the screen centre. Black or nothing is the bug these caught.
    #[test]
    #[ignore = "needs the forge_halo cache"]
    fn every_reticle_is_painted_in_the_hud_colour_around_the_centre() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ctx = egui::Context::default();
        let mut r = Raster::new(1920, 1080);
        let mut bg = Raster::new(1920, 1080);
        let biped = native.tags_by_group(cache, "bipd").into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with("spartans\\spartans"))).unwrap();
        let weap = |n: &str| {
            let w = format!("\\{n}");
            native.tags_by_group(cache, "weap").into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|x| x.ends_with(&w))).unwrap()
        };
        let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1920.0, 1080.0));
        let centre = egui::Rect::from_center_size(vp.center(), egui::vec2(260.0, 260.0));
        bg.background([110, 110, 110], [110, 110, 110]);
        let warthog = native.tags_by_group(cache, "vehi").into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with("warthog\\warthog"))).unwrap();
        let seat0 = crate::reach_seats::UnitSeats::read(native, cache, warthog).unwrap().seats[0].clone();
        let driver = crate::reach_unit_hud::SeatHuds::for_seat(native, cache, warthog, &seat0);
        for (label, weapon, seat) in [("assault_rifle", weap("assault_rifle"), None), ("rocket_launcher", weap("rocket_launcher"), None), ("warthog driver", 0, Some(driver))] {
            let mut hud = crate::play_hud::PlayHud::default();
            hud.bind(native, cache, biped);
            hud.set_weapon(native, cache, weapon, 0);
            let f = crate::play_hud_draw::HudFrame {
                weapon_tag: weapon, weapon_usable: seat.is_none(), seat_huds: seat.clone(), riding: seat.as_ref().map(|_| (0usize, String::new())),
                hud_on: false, crosshair_on: true, key_label: "E".into(), ..Default::default()
            };
            r.reset(1920, 1080);
            r.background([110, 110, 110], [110, 110, 110]);
            let out = ctx.run(egui::RawInput { screen_rect: Some(vp), ..Default::default() }, |ctx| {
                let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("r")));
                crate::play_hud_draw::paint_play_hud(ctx, &painter, vp, native, cache, &hud, &f);
            });
            r.draw(&ctx, out);
            let n = r.painted_in(centre, &bg);
            assert!(n > 40, "{label}: only {n} pixels painted around the centre");
            let (mut blue, mut dark) = (0usize, 0usize);
            for y in (centre.min.y as usize)..(centre.max.y as usize) {
                for x in (centre.min.x as usize)..(centre.max.x as usize) {
                    let p = r.pixel(x, y);
                    if (0..3).any(|k| (p[k] - 110.0).abs() > 20.0) {
                        if p[2] > p[0] * 1.5 && p[2] > 120.0 { blue += 1 } else { dark += 1 }
                    }
                }
            }
            assert!(blue > 40 && dark * 4 < blue, "{label}: {blue} blue pixels vs {dark} off-colour ones");
        }
    }
}
