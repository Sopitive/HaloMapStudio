//! GUI load path for Halo 3 / Halo 3: ODST caches. #h3
//!
//! `App::load_map` dispatches here when the picked cache is a Halo 3 engine
//! (`cache::is_halo3_cache`), BEFORE the Reach default. The load runs on a worker thread exactly
//! like the Halo 4 one (h4/gui.rs): it opens the cache, builds + uploads meshes (scene.rs
//! `build_meshes` - the same builder the headless screenshot runs) and streams batches over a
//! channel; `App::h3_drive_load` (called every frame from `drive_load`) appends them to the
//! renderer's lanes, lands the camera and writes the status line on `Done`. The Reach
//! SceneController's cache is CLOSED first and the Halo 4 state reset, so the Reach / Halo 4
//! only paths (particles, lights, editor scene, palette preview) stay off for a Halo 3 map.
//! Read-only viewing: no editor scene, no palette, no save. A Halo 3 map variant
//! (`h3_import_mvar`, from the variant browser / `import_mvar`) rides on the load: the worker
//! parses it (mvar.rs) and `build_meshes` places its Forge objects next to the scenario's.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use hms_render::GpuMesh;

use super::edit_scene::H3EditorAssets;
use super::render::append_lane;
use super::scene::{build_meshes, default_fog, load_map_with_variant, placeholder_sun_dir, H3EditorCollect, H3Stats, Lane};

pub enum H3Msg {
    Meshes(Lane, Vec<GpuMesh>),
    Progress(u32, u32),
    Log(String),
    Done(Result<(H3Stats, H3EditorAssets), String>),
}

/// A Halo 3 load in progress (the worker owns the cache; the main thread drains `rx`).
pub struct H3Load {
    rx: mpsc::Receiver<H3Msg>,
    handle: Option<std::thread::JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
    map: String,
    /// Engine label for status strings ("Halo 3" / "Halo 3: ODST").
    game: &'static str,
    t0: std::time::Instant,
}

impl crate::App {
    /// Abort a running Halo 3 load (no-op when none) and drop the "Halo 3 map loaded" flag.
    /// Called by every `load_map` (next to `h4_reset`), so switching games never leaves a
    /// worker running or stale meshes behind.
    pub(crate) fn h3_reset(&mut self) {
        if let Some(mut l) = self.h3_load.take() {
            l.cancel.store(true, Ordering::Relaxed);
            drop(l.rx);
            if let Some(h) = l.handle.take() { let _ = h.join(); }
        }
        self.h3_active = false;
        self.h3_scene = None;
        self.h3_edit = Default::default();
    }

    /// Start loading a Halo 3 cache on a worker thread. The Halo 4 state and the renderer's
    /// per-map lighting state are reset to neutral first (the previous map's fog / tints /
    /// screen-fx / post chain must not leak in).
    pub(crate) fn h3_load_map(&mut self, map_path: String) {
        self.h3_reset();
        self.h4_reset(); // drops the Halo 4 scene + post chain (the Reach load does the same)
        // the Halo 3 variant queued by `h3_import_mvar` rides on this load (or the last-opened
        // variant queued for the startup map, when it is a Halo 3 file); a Reach / Halo 4 one
        // cannot ride on a Halo 3 map. It becomes the open variant for the Variant Info panel
        // only - File > Save stays off (read-only).
        let variant = self.h3_pending_mvar.take()
            .or_else(|| self.autoload_variant.take_if(|p| super::mvar::is_h3_variant(p)));
        self.autoload_variant = None;
        self.current_variant_path = variant.clone();
        self.new_variant_template = None;
        if let Some(v) = variant.as_deref().and_then(|p| super::mvar::parse_h3_variant(p).ok()) {
            self.variant_title = v.header.name.clone();
            self.variant_description = v.header.description.clone();
            self.variant_author = v.header.author.clone();
            self.variant_editor = String::new();
            self.variant_header_dirty = false;
        }
        if let Some(s) = self.scene_ctl.as_mut() { s.close_cache(); }
        let queue = self.render_state.queue.clone();
        self.renderer.set_fog(&queue, &default_fog());
        self.renderer.set_sky_atmosphere(true);
        self.renderer.set_space_sky(false);
        self.renderer.set_scene_tints([1.0; 3], [1.0; 3]);
        self.renderer.set_screen_fx(&queue, 1.0, 0.5, crate::screenfx::ScreenFxRender::identity().cols);
        self.renderer.set_simple_lights(&queue, 0, &[]);
        self.renderer.set_planar_fog_volumes(Vec::new());
        self.renderer.set_lens_flare(Vec::new());
        self.renderer.set_sky_meshes(Vec::new());
        self.renderer.set_particle_meshes(Vec::new());
        self.renderer.set_overlay_lines(&self.render_state.device, &[]);
        // the fixed Halo 3 exposure (scene.rs: the bake gain is solved against it). The
        // interactive meter computes `base * clamp(key / metered, lo, hi)` (`exposure_params`),
        // so a unit band [1, 1] pins the GUI to the same fixed exposure as the headless shot.
        self.renderer.set_exposure(&queue, super::scene::fixed_exposure());
        self.renderer.set_exposure_band_abs(&queue, super::scene::fixed_exposure(), 1.0, 1.0);
        self.renderer.set_h4_filmic(&queue, None);
        self.renderer.set_sun_dir(placeholder_sun_dir());
        self.renderer.set_h4_shadow(None, None);
        self.load_t0 = std::time::Instant::now();
        self.load_frac = 0.0;
        let game = super::cache::cache_engine(std::path::Path::new(&map_path)).map(|e| e.label()).unwrap_or("Halo 3");
        self.map_status = format!("Loading {game} map {}...", map_path);
        log::info!("h3: loading '{map_path}' as {game}");

        let mesh = self.renderer.mesh_renderer_arc();
        let device = self.render_state.device.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (tx, rx) = mpsc::channel();
        let path = map_path.clone();
        let variant_path = variant.clone();
        let handle = std::thread::Builder::new().name("hms-h3-load".into()).spawn(move || {
            let send = |m: H3Msg| { let _ = tx.send(m); };
            let variant = variant_path;
            let loaded = match load_map_with_variant(&path, variant.as_deref()) {
                Ok(l) => l,
                Err(e) => { send(H3Msg::Done(Err(format!("{e:#}")))); return; }
            };
            for l in &loaded.log { send(H3Msg::Log(l.clone())); }
            let mut emit = |lane: Lane, v: Vec<GpuMesh>| send(H3Msg::Meshes(lane, v));
            let mut progress = |d: u32, t: u32| send(H3Msg::Progress(d, t));
            let mut log = |s: String| send(H3Msg::Log(s));
            // the GUI always builds the editor scene: the variant's objects are drawn through
            // H3ObjectScene's dynamic lanes (pickable / editable), not statically here
            let mut collect = H3EditorCollect::new();
            let r = build_meshes(&loaded, &device, &queue, &mesh, &mut emit, &mut progress, Some(&worker_cancel), &mut log, Some(&mut collect));
            send(H3Msg::Done(r.map(|st| {
                let (bmin, bmax) = st.frame_bounds();
                let assets = H3EditorAssets { cache: loaded.cache.clone(), variant: variant.as_ref().zip(loaded.variant.clone()).map(|(p, v)| (p.clone(), v)), collect, world_bounds: (bmin, bmax) };
                (st, assets)
            }).map_err(|e| format!("{e:#}"))));
        }).expect("spawn h3 load thread");
        self.h3_load = Some(H3Load { rx, handle: Some(handle), cancel, map: map_path, game, t0: std::time::Instant::now() });
        self.h3_active = true;
    }

    /// Drain the Halo 3 load worker (non-blocking). Called every frame; no-op without a load.
    pub(crate) fn h3_drive_load(&mut self) {
        let Some(load) = self.h3_load.as_ref() else { return };
        let mut done: Option<Result<(H3Stats, H3EditorAssets), String>> = None;
        let mut disconnected = false;
        loop {
            match load.rx.try_recv() {
                Ok(H3Msg::Meshes(lane, v)) => append_lane(&mut self.renderer, lane, v),
                Ok(H3Msg::Progress(d, t)) => {
                    self.load_frac = if t > 0 { (d as f32 / t as f32).min(0.99) } else { 0.0 };
                    let pct = if t > 0 { (d * 100 / t).min(99) } else { 0 };
                    self.map_status = format!("Loading {} map {}... {pct}%", load.game, load.map);
                }
                Ok(H3Msg::Log(s)) => log::info!("{s}"),
                Ok(H3Msg::Done(r)) => { done = Some(r); break; }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => { disconnected = true; break; }
            }
        }
        if done.is_none() && !disconnected { return; }
        let mut load = self.h3_load.take().unwrap();
        if let Some(h) = load.handle.take() { let _ = h.join(); }
        let secs = load.t0.elapsed().as_secs_f32();
        self.load_frac = 1.0;
        match done {
            Some(Ok((stats, assets))) => {
                // the editor scene over the worker's assets; the variant's objects become editor
                // objects (h3/app.rs) and the palette panel fills
                let n_variant = self.h3_install_editor_scene(assets);
                self.renderer.set_sun_dir(placeholder_sun_dir());
                self.renderer.set_scene_tints([1.0; 3], [1.0; 3]);
                let q = self.render_state.queue.clone();
                self.renderer.set_exposure(&q, stats.expo);
                self.renderer.set_exposure_band_abs(&q, stats.expo, 1.0, 1.0);
                // #h3-tone  Halo 3's own tone curve (1.5c - 0.5c^3, read out of halo3.dll's
                // copy_accumulation_target) plus the map's authored cfxs. The headless path
                // (h3/render.rs) already did this; without it here the GUI -- which is what the
                // user actually looks at -- kept rendering Halo 3 through Reach's bare sqrt.
                super::lighting::apply_post(&self.renderer, &q, stats.camera_fx.as_ref());
                // the map's atmosphere fog (lighting.rs), else the neutral default pushed at load #h3-fog
                if let Some(f) = stats.fog { self.renderer.set_fog(&q, &f); }
                let (fmin, fmax) = stats.frame_bounds();
                // frame the playable clusters (the headless framing); HMS_CAM / HMS_AUTOSHOT
                // are honoured by the shared post-load hooks in main.rs (`autoshot_frames`)
                if std::env::var("HMS_CAM").is_err() {
                    // land IN the map at a spawn (spawn_camera.rs through the editor scene: the
                    // variant's spawns, the scenario's, the scnr starting locations), stood off
                    // the geometry; the bounds overview only when the map has none of those
                    let spawn = self.h3_scene.as_deref().and_then(|s| s.spawn_camera_framed(Some((fmin, fmax))));
                    if let Some(p) = spawn {
                        self.camera.pos = p.pos;
                        self.camera.yaw = p.yaw;
                        self.camera.pitch = p.pitch;
                        log::info!("h3 spawn camera: {}", p.describe());
                    } else {
                        let cam = crate::h4::render::build_camera(fmin, fmax);
                        self.camera.pos = cam.pos;
                        self.camera.yaw = cam.yaw;
                        self.camera.pitch = cam.pitch;
                        log::info!("h3 camera (bounds overview, no spawn found): pos={:?} yaw={:.1} pitch={:.1}", cam.pos.to_array(), cam.yaw.to_degrees(), cam.pitch.to_degrees());
                    }
                } else {
                    self.apply_cam_env();
                }
                let lanes: Vec<String> = { let mut v: Vec<(Lane, usize)> = stats.lanes.iter().map(|(l, n)| (*l, *n)).collect(); v.sort_by_key(|(l, _)| format!("{l:?}")); v.iter().map(|(l, n)| format!("{l:?} {n}")).collect() };
                let variant = match self.current_variant_path.as_ref() {
                    Some(p) => format!(", {n_variant} variant objects ({}, editor scene)", p.file_name().unwrap_or_default().to_string_lossy()),
                    None => String::new(),
                };
                self.map_status = format!(
                    "{}: {} - {} draws / {} tris, {} BSP instances, {} scenario objects{variant}, {} sky parts, {} textures ({}) in {:.1}s. {}.",
                    load.game, load.map, stats.draws, stats.tris, stats.instances_drawn, stats.objects_placed, stats.sky_parts, stats.tex_ok, lanes.join(", "), secs,
                    if stats.has_lightmap { format!("Baked lighting ({} per-pixel / {} per-vertex uses)", stats.lm_pp, stats.lm_pv) } else { "Placeholder sun (no lightmap)".to_string() });
                log::info!("{}", self.map_status);
                crate::mapcat::save_last_map(&self.map_path);
                self.note_recent_map(); // File ▸ Open recent
                // HMS_AUTOSHOT (dev aid): capture the settled viewport and exit, like the Reach
                // load-complete tick does
                if self.autoshot.is_some() { self.autoshot_frames = 6; }
            }
            Some(Err(e)) => { self.h3_active = false; self.map_status = format!("{} load failed: {e}", load.game); log::error!("{}", self.map_status); }
            None => { self.h3_active = false; self.map_status = format!("{} load worker died", load.game); log::error!("{}", self.map_status); }
        }
    }

    /// Open a Halo 3 `.mvar`: find its base map among the detected Halo 3 caches (mapcat reads
    /// the id from the `.mapinfo`), queue the variant and load that map; the load worker places
    /// the objects (scene.rs `build_variant_objects`). Same shape as `h4_import_mvar`. #h3
    pub(crate) fn h3_import_mvar(&mut self, path: std::path::PathBuf) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let variant = match super::mvar::parse_h3_variant(&path) {
            Ok(v) => v,
            Err(e) => { self.spawn_status = format!("{name}: not a readable Halo 3 variant ({e})"); return; }
        };
        let want_game = crate::mapcat::Game::Halo3;
        let Some(idx) = self.map_candidates.iter().position(|c| c.game == want_game && c.map_id == Some(variant.map_id)) else {
            self.spawn_status = format!("{name}: base map id {} is not among the detected {} maps", variant.map_id, want_game.display_name());
            return;
        };
        let label = self.map_candidates[idx].label();
        self.picker_game = want_game; // #map-picker
        self.show_modded_maps = self.map_candidates[idx].modded;
        self.selected_map = Some(idx);
        self.h3_pending_mvar = Some(path);
        self.spawn_status = format!("{name} ('{}', {} objects): loading {} base map {label} (map id {}) with the variant (read-only)...",
            variant.header.name, variant.placed().count(), want_game.display_name(), variant.map_id);
        self.load_map();
    }

    /// The browser row hint of a Halo 3 variant: "Title (base map, N objects, author)".
    pub(crate) fn h3_summary_line(&self, sm: &super::mvar::H3VariantSummary) -> String {
        let title = if sm.title.is_empty() { "(untitled)" } else { sm.title.as_str() };
        let map = self.map_candidates.iter().find(|c| c.game == crate::mapcat::Game::Halo3 && c.map_id == Some(sm.map_id))
            .map(|c| c.stem.clone()).unwrap_or_else(|| format!("map id {}", sm.map_id));
        let author = if sm.author.is_empty() { String::new() } else { format!(", {}", sm.author) };
        format!("{title} [Halo 3] ({map}, {} objects{author})", sm.objects)
    }

    /// HMS_CAM="x,y,z,yawDeg,pitchDeg" (+ HMS_FOV): the exact-view override shared with the
    /// Reach load-complete tick.
    fn apply_cam_env(&mut self) {
        if let Ok(s) = std::env::var("HMS_CAM") {
            let v: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
            if v.len() >= 5 {
                self.camera.pos = glam::Vec3::new(v[0], v[1], v[2]);
                self.camera.yaw = v[3].to_radians();
                self.camera.pitch = v[4].to_radians();
                if let Some(f) = std::env::var("HMS_FOV").ok().and_then(|x| x.trim().parse::<f32>().ok()) {
                    if f > 1.0 && f < 179.0 { self.camera.fov_y = f.to_radians(); }
                }
            }
        }
    }
}
