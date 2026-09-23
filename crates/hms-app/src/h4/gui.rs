//! GUI load path for Halo 4 and Halo 2 Anniversary caches.
//!
//! `App::load_map` dispatches here when the picked cache is any non-Reach engine
//! (`cache::cache_engine(..).is_some()` - Halo 4 or H2A; the reader auto-detects which). The load
//! runs on a worker thread exactly like the Reach `run_load_worker`: it decodes + uploads meshes
//! (scene.rs `build_meshes`) and streams batches over a channel; `App::h4_drive_load` (called
//! every frame from `drive_load`) appends them to the renderer's lanes, installs the editor scene
//! (h4_app.rs `h4_install_editor_scene`: palette + the variant's objects), lands the
//! camera and writes the status line on `Done`. The Reach SceneController's cache is CLOSED first,
//! so the Reach-only paths (particles, lights, the 3D palette preview) stay off for a Halo 4 map
//! instead of touching stale Reach state; the editor verbs go through `objscene()`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use hms_render::GpuMesh;

use super::render::build_camera;
use super::scene::{build_meshes, load_map, load_map_with_variant, placeholder_sun_dir, H4EditorAssets, H4EditorCollect, H4Stats, Lane};

pub enum H4Msg {
    Meshes(Lane, Vec<GpuMesh>),
    Progress(u32, u32),
    Log(String),
    /// The stats + everything the editor scene needs (the cache moves out of the worker).
    Done(Result<(H4Stats, H4EditorAssets), String>),
}

/// A Halo 4 load in progress (the worker owns the cache; the main thread drains `rx`).
pub struct H4Load {
    rx: mpsc::Receiver<H4Msg>,
    handle: Option<std::thread::JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
    map: String,
    /// Engine label for status strings ("Halo 4" / "Halo 2 Anniversary").
    game: &'static str,
    t0: std::time::Instant,
}

impl crate::App {
    /// Abort a running Halo 4 load (no-op when none) and drop the "Halo 4 map loaded" flag.
    /// Called by the Reach `load_map` too, so switching back to Reach never leaves a worker running.
    pub(crate) fn h4_reset(&mut self) {
        if let Some(mut l) = self.h4_load.take() {
            l.cancel.store(true, Ordering::Relaxed);
            drop(l.rx);
            if let Some(h) = l.handle.take() { let _ = h.join(); }
        }
        self.h4_active = false;
        self.h4_scene = None;
        // `// #h4-expo-3` the cfxs adaptation block belongs to the map being unloaded
        self.h4_cfxs = None;
        self.h4_ae_stops = None;
        self.h4_ae_hist.clear();
        self.renderer.set_marker_meshes(Vec::new(), Vec::new(), Vec::new()); // map spawn markers (a Reach load calls this too)
        self.h4_map_spawn_markers = 0;
        // #h4-preview: drop the Halo 4 post chain from BOTH renderers (the viewport and the palette
        // preview's offscreen one), so a Reach map loaded after a Halo 4 map is composited the Reach
        // way in both. Every load runs through here (main.rs `load_map`), and a Halo 4 load pushes
        // its own state again at `Done`; the Reach load path sets its exposure / illum / bloom after.
        let (d, q) = (self.render_state.device.clone(), self.render_state.queue.clone());
        super::lighting::apply_post_quiet(&mut self.renderer, &d, &q, None, None);
        super::lighting::apply_post_quiet(&mut self.preview_renderer, &d, &q, None, None);
        self.preview_renderer.set_h4_shadow(None, None);
        self.preview_for = None;
    }

    /// Start loading a Halo 4 cache on a worker thread. The renderer's per-map lighting state is
    /// reset to neutral first (the previous Reach map's fog/tints/screen-fx must not leak in).
    pub(crate) fn h4_load_map(&mut self, map_path: String) {
        self.h4_reset();
        // a Halo 4 variant queued by `h4_import_mvar` rides along with this load (or the
        // last-opened / project variant queued for the startup map, when it is a Halo 4 file -
        // the Reach load-complete tick never runs for a Halo 4 map).
        let variant = self.h4_pending_mvar.take()
            .or_else(|| self.autoload_variant.take_if(|p| super::mvar::is_h4_variant(p)));
        // the variant becomes the OPEN variant (File > Save / `save as` write it
        // back through `h4_save_variant_to`); its header strings feed the Variant Info panel.
        self.current_variant_path = variant.clone();
        self.new_variant_template = None;
        if let Some(v) = variant.as_deref().and_then(|p| super::mvar::parse_h4_variant(p).ok()) {
            self.variant_title = v.title_key.clone();
            self.variant_description = v.description_key.clone();
            self.variant_author = v.author.clone();
            self.variant_editor = v.editor.clone();
            self.mvar_labels = v.labels.clone();
            self.variant_header_dirty = false;
            self.seed_variant_globals(v.globals());
        }
        if let Some(s) = self.scene_ctl.as_mut() { s.close_cache(); }
        let queue = self.render_state.queue.clone();
        self.renderer.set_fog(&queue, &[0.0f32; 28]);
        self.renderer.set_sky_atmosphere(true);
        self.renderer.set_space_sky(false);
        self.renderer.set_scene_tints([1.0; 3], [1.0; 3]);
        self.renderer.set_screen_fx(&queue, 1.0, 0.5, crate::screenfx::ScreenFxRender::identity().cols);
        self.renderer.set_simple_lights(&queue, 0, &[]);
        self.renderer.set_planar_fog_volumes(Vec::new());
        self.renderer.set_lens_flare(Vec::new());
        self.renderer.set_sky_meshes(Vec::new());
        // (the previous map's geometry in every other lane + the object lists were already
        // dropped by App::load_map's clear_all_scene_objects before it dispatched here)
        self.renderer.set_particle_meshes(Vec::new());
        self.renderer.set_overlay_lines(&self.render_state.device, &[]);
        self.renderer.set_exposure(&queue, 1.0);
        self.renderer.set_h4_filmic(&queue, None); // a map without cfxs keeps the Reach composite
        self.renderer.set_sun_dir(placeholder_sun_dir());
        self.renderer.set_h4_shadow(None, None); // back to the Reach shadow fit until the map is in
        self.load_t0 = std::time::Instant::now();
        self.load_frac = 0.0;
        let game = super::cache::cache_engine(std::path::Path::new(&map_path)).map(|e| e.label()).unwrap_or("Halo 4");
        self.map_status = format!("Loading {game} map {}...", map_path);
        log::info!("h4: loading '{map_path}' as {game}");

        let mesh = self.renderer.mesh_renderer_arc();
        let device = self.render_state.device.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (tx, rx) = mpsc::channel();
        let path = map_path.clone();
        let handle = std::thread::Builder::new().name("hms-h4-load".into()).spawn(move || {
            let send = |m: H4Msg| { let _ = tx.send(m); };
            let loaded = match match &variant { Some(v) => load_map_with_variant(&path, Some(v)), None => load_map(&path) } {
                Ok(l) => l,
                Err(e) => { send(H4Msg::Done(Err(format!("{e:#}")))); return; }
            };
            for l in &loaded.log { send(H4Msg::Log(l.clone())); }
            let mut emit = |lane: Lane, v: Vec<GpuMesh>| send(H4Msg::Meshes(lane, v));
            let mut progress = |d: u32, t: u32| send(H4Msg::Progress(d, t));
            let mut log = |s: String| send(H4Msg::Log(s));
            // the GUI always builds the editor scene: the variant objects are NOT drawn
            // statically here, they go through H4ObjectScene's dynamic lanes (pickable / editable)
            let mut collect = H4EditorCollect::new();
            let r = build_meshes(&loaded, &device, &queue, &mesh, &mut emit, &mut progress, Some(&worker_cancel), &mut log, Some(&mut collect));
            send(H4Msg::Done(r.map(|st| { let assets = H4EditorAssets::from_loaded(loaded, collect, &st); (st, assets) }).map_err(|e| format!("{e:#}"))));
        }).expect("spawn h4 load thread");
        self.h4_load = Some(H4Load { rx, handle: Some(handle), cancel, map: map_path, game, t0: std::time::Instant::now() });
        self.h4_active = true;
    }

    /// Drain the Halo 4 load worker (non-blocking). Called every frame; no-op without a load.
    pub(crate) fn h4_drive_load(&mut self) {
        let Some(load) = self.h4_load.as_ref() else { return };
        let mut done: Option<Result<(H4Stats, H4EditorAssets), String>> = None;
        let mut disconnected = false;
        loop {
            match load.rx.try_recv() {
                Ok(H4Msg::Meshes(lane, v)) => match lane {
                    Lane::Opaque => self.renderer.append_static_meshes(v),
                    Lane::AlphaTest => self.renderer.append_alphatest_meshes(v),
                    Lane::Blend => self.renderer.append_blend_meshes(v),
                    Lane::Additive => self.renderer.append_additive_meshes(v),
                    Lane::Sky => self.renderer.set_sky_meshes(v.into_iter().map(|m| { let b = m.blend_mode(); let k = m.centroid()[0]; (m, b, k) }).collect()),
                },
                Ok(H4Msg::Progress(d, t)) => {
                    self.load_frac = if t > 0 { (d as f32 / t as f32).min(0.99) } else { 0.0 };
                    let pct = if t > 0 { (d * 100 / t).min(99) } else { 0 };
                    self.map_status = format!("Loading {} map {}... {pct}% ({d}/{t} meshes)", load.game, load.map);
                }
                Ok(H4Msg::Log(s)) => log::info!("{s}"),
                Ok(H4Msg::Done(r)) => { done = Some(r); break; }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => { disconnected = true; break; }
            }
        }
        if done.is_none() && !disconnected { return; }
        let mut load = self.h4_load.take().unwrap();
        if let Some(h) = load.handle.take() { let _ = h.join(); }
        let secs = load.t0.elapsed().as_secs_f32();
        self.load_frac = 1.0;
        match done {
            Some(Ok((stats, mut assets)) ) => {
                // the scenario's spawn-family markers -> the renderer's marker lane,
                // drawn only while View > "Show map spawn points" is on (the persisted Reach switch)
                let (mo, ma, mb) = super::scene::split_marker_meshes(std::mem::take(&mut assets.collect.map_spawn_meshes));
                self.renderer.set_marker_meshes(mo, ma, mb);
                self.renderer.set_show_markers(self.show_map_spawns);
                // the editor scene (dynamic lanes + picks) over the worker's assets; the
                // variant's objects become editor objects (`h4_render_variant_objects` builds
                // their ObjMeta / source records)
                let n_variant = self.h4_install_editor_scene(assets);
                self.h4_map_spawn_markers = stats.scnr_spawn_markers;
                log::info!("h4 scnr: {} forge-palette placements skipped (variant-owned), {} spawn-family markers on the marker lane ({})", stats.scnr_forge_owned, stats.scnr_spawn_markers, if self.show_map_spawns { "shown" } else { "hidden - View > Show map spawn points" });
                self.renderer.set_sun_dir(stats.sun_dir());
                self.renderer.set_scene_tints(stats.sun_tint(), [1.0; 3]);
                // the object sun shadow fit (static scenario casters; the editor's own objects
                // join after their first rebuild, main.rs) + the BSP's cascade
                self.renderer.set_h4_shadow(stats.caster_bounds, stats.cascade);
                log::info!("h4 shadows: static caster bounds {:?}, cascade {:?}", stats.caster_bounds.map(|(a, b)| (a.to_array(), b.to_array())), stats.cascade);
                // the map's atmosphere fog (lighting.rs)
                if let Some(f) = stats.fog { let q = self.render_state.queue.clone(); self.renderer.set_fog(&q, &f); }
                // cfxs: exposure band + engine meter, filmic, bloom, colour
                // grading LUT, self-illum exposure (lighting.rs `apply_post`)
                {
                    let q = self.render_state.queue.clone();
                    let d = self.render_state.device.clone();
                    super::lighting::apply_post(&mut self.renderer, &d, &q, stats.camera_fx.as_ref(), stats.color_grading_lut.as_ref());
                    // `// #h4-expo-3` keep the cfxs: the Lighting panel seeds its band sliders
                    // from the authored stops, and the interactive auto-exposure runs the engine's
                    // delay / blend / max-change adaptation (`adapt_stops`) from it.
                    self.h4_cfxs = stats.camera_fx.clone();
                    self.h4_ae_stops = None;
                    self.h4_ae_hist.clear();
                    // #h4-preview: the palette preview is a SECOND SceneRenderer, so it needs the
                    // same per-map state to light a previewed object like the viewport does: the
                    // cfxs post chain (band + engine meter, filmic, bloom, grading LUT, self-illum
                    // law), the baked sun direction / tint and the object shadow fit. The Reach
                    // mirror does the same a few lines into `drive_load` (main.rs).
                    super::lighting::apply_post_quiet(&mut self.preview_renderer, &d, &q, stats.camera_fx.as_ref(), stats.color_grading_lut.as_ref());
                    self.preview_renderer.set_sun_dir(stats.sun_dir());
                    self.preview_renderer.set_scene_tints(stats.sun_tint(), [1.0; 3]);
                    self.preview_renderer.set_h4_shadow(stats.caster_bounds, stats.cascade);
                    self.preview_for = None; // lit per map -> rebuild the panel for this one
                }
                let (fmin, fmax) = stats.frame_bounds();
                if std::env::var("HMS_CAM").is_err() {
                    // land IN the map like the Reach load: the variant's loadout
                    // camera / initial spawn, else the scenario's starting location / spawn
                    // placements (spawncam.rs, stood off the geometry); the bounds overview only
                    // when the map has none of those.
                    let spawn = self.h4_scene.as_deref().and_then(super::spawncam::spawn_camera);
                    if let Some(c) = spawn {
                        self.camera.pos = c.pos;
                        self.camera.yaw = c.yaw;
                        self.camera.pitch = c.pitch;
                        log::info!("h4 spawn camera ({}): pos={:?} yaw={:.1} pitch={:.1} (stand-off moved {:.2} wu)", c.source.label(), c.pos.to_array(), c.yaw.to_degrees(), c.pitch.to_degrees(), (c.pos - c.raw_pos).length());
                    } else {
                        let cam = build_camera(fmin, fmax);
                        self.camera.pos = cam.pos;
                        self.camera.yaw = cam.yaw;
                        self.camera.pitch = cam.pitch;
                        log::info!("h4 camera (bounds overview, no spawn found): pos={:?} yaw={:.1} pitch={:.1}", cam.pos.to_array(), cam.yaw.to_degrees(), cam.pitch.to_degrees());
                    }
                }
                let lanes: Vec<String> = { let mut v: Vec<(Lane, usize)> = stats.lanes.iter().map(|(l, n)| (*l, *n)).collect(); v.sort_by_key(|(l, _)| format!("{l:?}")); v.iter().map(|(l, n)| format!("{l:?} {n}")).collect() };
                let variant_note = if n_variant > 0 { format!(" + {} map-variant objects (editor scene)", n_variant) } else { String::new() };
                self.map_status = format!(
                    "{}: {} - {} draws / {} tris, {} objects{}, {} textures ({}) in {:.1}s. {} sun, {}.",
                    load.game, load.map, stats.draws, stats.tris, stats.objects_placed, variant_note, stats.tex_ok, lanes.join(", "), secs, if stats.sun.is_some() { "Baked" } else { "Placeholder" },
                    if stats.lm_instances > 0 { format!("PROVISIONAL lightmap atlas on {} instances (HMS_H4_LIGHTMAP)", stats.lm_instances) } else { "flat lighting (no lightmaps)".to_string() });
                log::info!("{}", self.map_status);
            }
            Some(Err(e)) => { self.h4_active = false; self.map_status = format!("{} load failed: {e}", load.game); log::error!("{}", self.map_status); }
            None => { self.h4_active = false; self.map_status = format!("{} load worker died", load.game); log::error!("{}", self.map_status); }
        }
    }

    /// Open a Halo 4 map variant: find its base map among the detected
    /// Halo 4 caches by map id (mapcat reads `.mapinfo`), select it and load it with the variant
    /// queued in `h4_pending_mvar` (the load worker resolves the objects through the base map's
    /// forge palette, scene.rs). The Reach variant flow (`import_mvar`) dispatches here when the
    /// file's `mvar` chunk is v50, so the Reach editor state is never touched by a Halo 4 file.
    pub(crate) fn h4_import_mvar(&mut self, path: std::path::PathBuf) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let variant = match super::mvar::parse_h4_variant(&path) {
            Ok(v) => v,
            Err(e) => { self.spawn_status = format!("{name}: not a readable Halo 4 variant ({e})"); return; }
        };
        let Some(idx) = self.map_candidates.iter().position(|c| c.game == crate::mapcat::Game::Halo4 && c.map_id == Some(variant.map_id)) else {
            self.spawn_status = format!("{name}: base map id {} is not among the detected Halo 4 maps", variant.map_id);
            return;
        };
        let label = self.map_candidates[idx].label();
        self.picker_game = crate::mapcat::Game::Halo4; // #map-picker
        self.show_modded_maps = self.map_candidates[idx].modded;
        self.selected_map = Some(idx);
        self.h4_pending_mvar = Some(path);
        self.spawn_status = format!("{name} ('{}', {} objects): loading Halo 4 base map {label} (map id {}) with the variant (editor scene)...", variant.title, variant.objects.len(), variant.map_id);
        self.load_map();
    }
}
