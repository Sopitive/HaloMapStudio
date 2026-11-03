//! Headless script host: the batch / automation engine. Runs the same command grammar
//! (`crate::script`) as the interactive editor against a windowless offscreen renderer that loads
//! maps synchronously, so a script can `loadmap / loadvariant / camera / screenshot` in sequence
//! and, with `foreach`, drive whole batches (e.g. screenshot every variant of a map).
//!
//! Entry: `hms-app --script <file>` or `hms-app --exec "<commands>"`. `HMS_SHOT_SIZE` sets the
//! initial render size (a per-shot `screenshot ... WxH` resizes on demand).

use std::collections::HashMap;
use std::time::Instant;

use eframe::wgpu;
use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::NativeDll;
use hms_render::{Camera, SceneRenderer};

use crate::forge_scale;
use crate::h4::edit::{h4_tag, H4PaletteItem};
use crate::h4::edit_scene::H4ObjectScene;
use crate::h4::mvar::{H4PlacedObject, H4Variant};
use crate::h4::palette::ForgePalette;
use crate::mapcat::{self, MapCandidate};
use crate::mvar;
use crate::objscene::ObjectScene;
use crate::scene::{self, LoadMsg, SceneController};
use crate::script;

/// Per-object editable data the host needs for `set`/`get`/filtering (subset of the App's ObjMeta).
#[derive(Clone, Default)]
struct HMeta {
    name: String,
    team: u8,
    color: i32,
    cached_type: u8,
    spawn_seq: i32,
    respawn: u8,
    label: String,
    boundary_shape: u8,
    boundary: [u16; 4],
    /// SCALED / SHADOW pseudo-flag overrides (None = derived default).
    flags: forge_scale::ObjFlags,
    /// Halo 4 only: the record's own `variant-object-scale` field (datum +0x2C dequantised; the
    /// shipped default q 7 is 1.048). MCC stores it but does not draw it, so it is reported
    /// (`get` / `flags`) and never applied: a Halo 4 object scales by the gametype rule alone.
    /// None for a Reach object.
    h4_scale: Option<f32>,
}

impl HMeta {
    fn scale(&self) -> f32 {
        // Halo 4: 1.0 unless the gametype rule (SCALED / scale label) is on; the record's scale
        // field is not what MCC draws.
        if self.h4_scale.is_some() {
            return crate::h4::edit::h4_effective_scale(self.flags.scaled_on(&self.label), self.spawn_seq, self.team, forge_scale::ScaleConvention::X330);
        }
        // The SCALED flag (default: has the "scale" label) decides.
        if self.flags.scaled_on(&self.label) {
            forge_scale::object_scale(self.spawn_seq, self.team)
                .clamp(0.01, forge_scale::object_max_scale(self.team))
        } else {
            1.0
        }
    }
    fn shadow_on(&self) -> bool {
        self.flags.shadow_on(self.team, &self.label)
    }
    /// "0xF0000012 name: scaled ON (auto, x2.5) | shadow off (auto)".
    fn flags_line(&self, datum: u32, g: &forge_scale::GlobalFlags) -> String {
        let how = |o: Option<bool>| if o.is_some() { "set" } else { "auto" };
        let (fs, fc) = forge_scale::effective_flags(g, &self.flags, self.team, &self.label);
        // A Halo 4 object: the SCALED rule is the only drawn scale; the record's own field is
        // reported for the file's sake (MCC stores it, does not draw it).
        if let Some(hs) = self.h4_scale {
            let on = self.flags.scaled_on(&self.label);
            return format!(
                "0x{datum:08X} {}: team={} label='{}' spawnseq={} | scaled {} ({}, x{:.3}{}; variant-object-scale x{:.3}, not drawn by MCC) | shadow {} ({}{})",
                self.name, self.team as i32, self.label, self.spawn_seq, if on { "ON" } else { "off" }, how(self.flags.scaled), self.scale(),
                if !g.scaled { ", global OFF -> x1" } else { "" }, hs,
                if self.shadow_on() { "ON" } else { "off" }, how(self.flags.shadow),
                if !g.shadowcasters && fc != self.shadow_on() { ", global OFF" } else { "" },
            );
        }
        format!(
            "0x{datum:08X} {}: team={} label='{}' seq={} | scaled {} ({}, x{:.3}{}) | shadow {} ({}{})",
            self.name, self.team as i32, self.label, self.spawn_seq,
            if self.flags.scaled_on(&self.label) { "ON" } else { "off" }, how(self.flags.scaled), self.scale(),
            if !g.scaled && fs != self.flags.scaled_on(&self.label) { ", global OFF" } else { "" },
            if self.shadow_on() { "ON" } else { "off" }, how(self.flags.shadow),
            if !g.shadowcasters && fc != self.shadow_on() { ", global OFF" } else { "" },
        )
    }
}

/// The headless scripting engine.
pub struct ScriptHost {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: SceneRenderer,
    dll_path: std::path::PathBuf,
    cands: Vec<MapCandidate>,

    scene: Option<Box<SceneController>>,
    /// #reach-skin The posed Reach character actors (`actor ...`). Survives between commands so
    /// a script can place, pose, arm and then `screenshot`.
    actors: crate::reach_actor::ActorScene,
    map_path: String,
    map_id: Option<u32>,
    /// #film  the saved film the `film` verb last opened (header only).
    film: Option<Box<crate::film::Film>>,
    film_path: Option<std::path::PathBuf>,
    /// #film  the decoded playback index + clock, once `film index` has run.
    film_play: Option<crate::film_play::Playback>,
    /// #film-action  Has this film's Forge scene been drawn (or refused, with a reason)? The
    /// GUI draws it when a film is OPENED; a script opens films header-only on purpose, so the
    /// scene is drawn on the first playback command instead and the outcome is reported there.
    film_scene: Option<String>,
    film_driver: crate::film_play::ActorDriver,
    /// #projectile  What the FILM effect lane installed last frame, for `film status`.
    film_fx: crate::film_projectile::FxReport,
    /// #gamevariant  Has the film's own loadout table been resolved onto the tracks yet? `Some`
    /// once it has landed (or once it cannot), so the attempt is made once and not per command.
    film_spawn_weapon: Option<crate::film_variant::SpawnWeaponApplied>,
    /// #fp-arms  Which player's armour the first-person arm models are currently wearing.
    film_fp_armour: Option<(u32, u8)>,
    /// #fp-frame  Print-only diagnostic state for `film fp`: the aspect the framing report
    /// projects through, and whether it lists every node. Nothing in the render reads either.
    shot_aspect: f32,
    fp_verbose: bool,
    /// #fp-fov  The script camera's own vertical FOV, parked while a first-person film camera is
    /// live. Mirrors the window.
    fov_before_first_person: Option<f32>,
    /// #film-camera  Did the last `apply_film_camera` actually SOLVE the requested camera? A
    /// camera naming a slot or owner group this film does not have solves to nothing, and the
    /// capture then shows a stale view. `shot` says so rather than labelling it with the request.
    film_camera_resolved: bool,
    /// #fp-fov  The first-person field of view in the game's own degrees, the SAME persisted
    /// value the window's slider drives -- so a scripted capture frames like the window.
    fp_fov_deg: f32,

    // Halo 4 editing state (empty while a Reach map is loaded; `scene` is None while
    // `h4_scene` is Some - a Halo 4 load drops the Reach controller).
    h4_scene: Option<Box<H4ObjectScene>>,
    /// The rich palette (`palette::palette`) + its flat item list (`place <name>` / `list palette`).
    h4_pal: Option<ForgePalette>,
    h4_items: Vec<H4PaletteItem>,
    /// The FULL edited record per datum: the decoded slot for a loaded object, the instantiated
    /// record for a placed one, a clone (slot = new) for a dup. `set` edits these directly; a
    /// save starts from them (`build_h4_save_list_core` with them as `src`).
    h4_records: HashMap<u32, H4PlacedObject>,
    /// Records HMS cannot display (carried through a save verbatim).
    h4_unresolved: Vec<H4PlacedObject>,
    /// The open variant (path + parsed header: bounds / budget / labels) - the save source.
    h4_variant: Option<(std::path::PathBuf, H4Variant)>,
    /// The label table (`set <d> label <name>` appends a missing name; saved as a header edit).
    h4_labels: Vec<String>,
    /// The open variant's global fields (either game) + the pending global edits
    /// (`variant set`); the Halo 4 save writes them, a Reach edit needs the interactive app.
    variant_globals: Option<mvar::VariantGlobals>,
    global_edits: mvar::GlobalEdits,
    /// Header strings a `variant set name|description|author|editor` changed (Halo 4 save input).
    global_strings: Option<(String, String, String, String)>,

    // Warm base-map caches (rebuilt on each map load).
    palette: Vec<hms_native::ForgePaletteEntry>,
    types: Vec<(u32, u32)>,
    place_palette: Vec<(u32, u32, String, u32)>, // (mode_tag, obj_tag, name, variant_name_sid)
    /// The scenario's placements that are currently DRAWN (always the head of `objects`).
    base_objects: Vec<ObjectInfo>,
    /// Every scenario placement (drawn or not) + the obj tags among them that are
    /// spawn-family markers; `show_map_spawns` (HMS_MAP_SPAWNS seeds it, `mapspawns on|off` changes
    /// it for this run) decides which of them land in `base_objects`.
    base_all: Vec<ObjectInfo>,
    /// Per scenario datum (category, palette_index, name_index) for `get` / `pick`.
    base_ident: HashMap<u32, (u16, i16, i16)>,
    base_spawn_tags: std::collections::HashSet<u32>,
    show_map_spawns: bool,
    /// Draw the hidden-block physics hulls (HMS_PHYSICS_OUTLINES seeds it, default
    /// on; `outlines on|off` changes it for this run). Off = the game's look.
    show_physics_outlines: bool,
    /// #h4-phys the two selected-object hull overlays (run-local; the host renders no overlay).
    show_collision_hull: bool,
    show_physics_hull: bool,
    /// #wire-visible  Draw the SELECTION wireframe through other objects (HMS_WIRE_XRAY seeds
    /// it, default off; `wirexray on|off` changes it for this run).
    wire_xray: bool,
    /// #quant-error  The save-precision ghost (run-local; `HMS_QUANT_GHOST=1` / `1x40` seeds it,
    /// `precision ghost on|off|<factor>` changes it), so a batch capture can show the predicted
    /// pose exactly as the interactive viewport does.
    quant_ghost: crate::quant_error_ui::GhostSetting,
    /// #save-precision  Run-local (default ON; `precision on|off`): after `save` the objects adopt the
    /// poses the game will load. `last_save_report` is what `precision report` prints.
    save_precision_on: bool,
    last_save_report: Option<crate::quant_error::SaveReport>,
    /// #cannon-preview  run-local (`cannontraj on|off`), default off.
    cannon_preview: crate::man_cannon_preview::State,
    /// Draw the structure-design soft ceilings (HMS_SOFT_CEILINGS seeds it,
    /// default off; `softceilings on|off` changes it for this run).
    show_soft_ceilings: bool,
    /// The scenario trigger-volume boxes (`triggers on|off`, default off).
    show_triggers: bool,
    /// The playable BSPs' world bounds + floor (HMS_HARD_FLOOR seeds it, default off).
    show_hard_floor: bool,
    /// The playable-BSP boxes (HMS_PLAYABLE_BOUNDS seeds it, default off).
    show_playable_bounds: bool,

    // Live object set (base scenery + forge/placed) + per-object data.
    objects: Vec<ObjectInfo>,
    colors: HashMap<u32, (u8, u8)>,
    meta: HashMap<u32, HMeta>,
    next_datum: u32,
    selected: Vec<u32>,

    camera: Camera,
    size: (u32, u32),
    cur_stem: String, // stem of the last-loaded variant (else base map), for `screenshot <dir>`
    dirty: bool,      // object set changed → rebuild before next render
    /// Placed Forge special-FX screen effects on/off (HMS_FORGE_FX env seeds it,
    /// `screenfx on|off` changes it) + the per-map decode cache and the last-pushed key.
    forge_fx_enabled: bool,
    screenfx_cache: crate::screenfx::ScreenFxCache,
    /// Halo 4 / H2A decode cache (its tag reader differs; composition is shared). #forge-fx-h4
    h4_screenfx_cache: crate::h4::screenfx::H4ScreenFxCache,
    screenfx_pushed: Option<(bool, u32, Vec<u32>)>,
    active_screenfx: crate::screenfx::ActiveScreenFx,
    /// The global "Scaled objects" / "Shadow casters (Forge)" switches: the persisted
    /// settings with HMS_SCALED / HMS_SHADOWCASTERS on top; `scale on|off` / `shadowcasters on|off`
    /// change them for this run only (a batch render must not rewrite the GUI's settings).
    globals: forge_scale::GlobalFlags,
}

/// CLI entry: run a script file (`--script path`) or an inline string (`--exec "…"`).
pub fn run(source_text: String) -> anyhow::Result<()> {
    hide_console();
    let mut host = ScriptHost::new()?;
    let out = script::run_program(&mut host, &source_text);
    // Print the log to stderr (stdout stays clean) so batch runs are observable.
    eprint!("{out}");
    // HMS_DUMPONE=<hex,hex,...> (print-only diagnostic): dump the decoded bitmaps with those tag
    // ids to <diag dir>/tex_<tag>.png and print their mean RGBA.
    if let Ok(tags) = std::env::var("HMS_DUMPONE") {
        if let Some(scene) = host.scene.as_ref() {
            for t in tags.split(',') {
                let ts = t.trim().trim_start_matches("0x");
                if let Ok(tag) = u32::from_str_radix(ts, 16) {
                    match scene.decode_bitmap_keep_alpha_public(tag) {
                        Some((mut px, w, h)) => {
                            for p in px.chunks_exact_mut(4) { p.swap(0, 2); }
                            let path = format!("{}/tex_{:x}.png", crate::headless::diag_dir().display(), tag);
                            let _ = crate::headless::write_png(&path, &px, w, h);
                            let (mut r,mut g,mut b,mut a,mut n)=(0u64,0u64,0u64,0u64,0u64);
                            for q in px.chunks_exact(4){ r+=q[0] as u64;g+=q[1] as u64;b+=q[2] as u64;a+=q[3] as u64;n+=1; }
                            let n=n.max(1);
                            eprintln!("dumped tex {:x} {}x{} -> {} meanRGBA=({},{},{},{})", tag,w,h,path,r/n,g/n,b/n,a/n);
                        }
                        None => eprintln!("decode {:x} FAILED", tag),
                    }
                }
            }
        } else { eprintln!("HMS_DUMPONE: no scene loaded"); }
    }
    Ok(())
}

impl ScriptHost {
    fn new() -> anyhow::Result<Self> {
        let (w, h) = parse_size(std::env::var("HMS_SHOT_SIZE").ok().as_deref());
        let instance = wgpu::Instance::default();
        // Same policy as headless: HMS_SOFTWARE_GPU=1 forces the software rasterizer,
        // otherwise hardware first with an automatic software fallback.
        let want_soft = std::env::var("HMS_SOFTWARE_GPU").is_ok();
        let request = |fallback: bool| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: fallback,
        }));
        let adapter = if want_soft { request(true) } else { request(false).or_else(|| request(true)) }
            .ok_or_else(|| anyhow::anyhow!("no wgpu adapter (script host) — no GPU driver and no software rasterizer"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("hms-script-host"),
                required_features: adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage, // see headless.rs
            },
            None,
        ))?;
        let mut renderer = SceneRenderer::new(&device, &queue, (w, h));
        renderer.show_grid = false;
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();
        // The native library sits beside the app; a `cargo test` binary lives one level down in
        // `deps/`, so look there too (this is what lets a gate drive the real script host).
        let dll_path = [exe_dir.join(crate::PAYLOAD_DLL), exe_dir.join("..").join(crate::PAYLOAD_DLL)]
            .into_iter()
            .find(|p| p.is_file())
            .unwrap_or_else(|| exe_dir.join(crate::PAYLOAD_DLL));
        let mut cam = Camera::default();
        cam.far = 8000.0;
        Ok(Self {
            device,
            queue,
            renderer,
            dll_path,
            cands: mapcat::enumerate(),
            scene: None,
            actors: Default::default(),
            map_path: String::new(),
            map_id: None,
            film: None,
            film_path: None,
            film_play: None,
            film_scene: None,
            film_driver: Default::default(),
            film_fx: Default::default(),
            film_spawn_weapon: None,
            film_fp_armour: None,
            shot_aspect: 16.0 / 9.0,
            fp_verbose: false,
            fov_before_first_person: None,
            film_camera_resolved: false,
            fp_fov_deg: crate::film_play::load_fov_setting(),
            h4_scene: None,
            h4_pal: None,
            h4_items: Vec::new(),
            h4_records: HashMap::new(),
            h4_unresolved: Vec::new(),
            h4_variant: None,
            h4_labels: Vec::new(),
            variant_globals: None,
            global_edits: mvar::GlobalEdits::default(),
            global_strings: None,
            palette: Vec::new(),
            types: Vec::new(),
            place_palette: Vec::new(),
            base_objects: Vec::new(),
            base_all: Vec::new(),
            base_ident: HashMap::new(),
            base_spawn_tags: Default::default(),
            show_map_spawns: crate::map_spawns::env_show_map_spawns(),
            show_physics_outlines: crate::physics_outlines::env_show_physics_outlines(),
            show_collision_hull: false,
            show_physics_hull: false,
            wire_xray: crate::wire_xray::env_wire_xray(),
            quant_ghost: crate::quant_error_ui::env_setting().unwrap_or_default(),
            save_precision_on: true,
            last_save_report: None,
            cannon_preview: Default::default(),
            show_soft_ceilings: crate::soft_ceilings::env_show_soft_ceilings(),
            show_triggers: false,
            show_hard_floor: crate::hard_floor::env_show_hard_floor(),
            show_playable_bounds: crate::hard_floor::env_show_playable_bounds(),
            objects: Vec::new(),
            colors: HashMap::new(),
            meta: HashMap::new(),
            next_datum: 0xD800_0000,
            selected: Vec::new(),
            camera: cam,
            size: (w, h),
            cur_stem: String::new(),
            dirty: false,
            forge_fx_enabled: crate::screenfx::env_forge_fx_enabled(),
            screenfx_cache: crate::screenfx::ScreenFxCache::default(),
            h4_screenfx_cache: crate::h4::screenfx::H4ScreenFxCache::default(),
            globals: forge_scale::GlobalFlags::load(),
            screenfx_pushed: None,
            active_screenfx: crate::screenfx::ActiveScreenFx::default(),
        })
    }

    /// Compose the scenario default + placed Forge special-FX screen effects for the current
    /// object set and upload when the set changed (the scenario default is part of it: Zealot's
    /// blue filter belongs in a scripted capture).
    fn refresh_screen_fx(&mut self) {
        // Halo 4 / H2A path (#forge-fx-h4): decode through the pure-Rust reader; no scenario default.
        if let Some(cache) = self.h4_scene.as_deref().map(|s| s.cache().clone()) {
            let mut obje_tags: Vec<usize> = Vec::new();
            for o in &self.objects {
                if let Some(idx) = crate::h4::edit::h4_tag_index(o.primary_tag) {
                    if !obje_tags.contains(&idx) { obje_tags.push(idx); }
                }
            }
            let active = crate::h4::screenfx::active_screen_fx(&cache, &mut self.h4_screenfx_cache, obje_tags.into_iter());
            let key = (self.forge_fx_enabled, 0u32, active.forge_tags.clone());
            if self.screenfx_pushed.as_ref() != Some(&key) {
                let line = crate::screenfx::push_to_renderer(&self.renderer, &self.queue, &active, self.forge_fx_enabled);
                if std::env::var("HMS_DIAG").is_ok() { eprintln!("HMS_DIAG h4 {line}"); }
                self.screenfx_pushed = Some(key);
            }
            self.active_screenfx = active;
            return;
        }
        let Some(scene) = self.scene.as_ref() else { return };
        let default_tag = scene.screen_fx().map(|f| f.tag).unwrap_or(0);
        let active = crate::screenfx::active_screen_fx(scene, &mut self.screenfx_cache, default_tag, self.objects.iter().map(|o| (o.datum, o.primary_tag)));
        let key = (self.forge_fx_enabled, default_tag, active.forge_tags.clone());
        if self.screenfx_pushed.as_ref() != Some(&key) {
            let line = crate::screenfx::push_to_renderer(&self.renderer, &self.queue, &active, self.forge_fx_enabled);
            if std::env::var("HMS_DIAG").is_ok() { eprintln!("HMS_DIAG {line}"); }
            self.screenfx_pushed = Some(key);
        }
        self.active_screenfx = active;
    }

    fn screenfx_status(&self) -> String {
        let a = &self.active_screenfx;
        if a.forge_names.is_empty() {
            format!("Forge FX: none placed ({})", if self.forge_fx_enabled { "on" } else { "off" })
        } else {
            format!("Forge FX{}: {}", if self.forge_fx_enabled { "" } else { " (off)" }, a.forge_names.join(" + "))
        }
    }

    /// The object scene the editor verbs talk to: the Halo 4 scene while one is
    /// loaded, else the Reach controller (None before any map is loaded).
    fn objscene(&self) -> Option<&dyn ObjectScene> {
        if let Some(h) = self.h4_scene.as_deref() { return Some(h); }
        self.scene.as_deref().map(|s| s as &dyn ObjectScene)
    }
    fn objscene_mut(&mut self) -> Option<&mut dyn ObjectScene> {
        if let Some(h) = self.h4_scene.as_deref_mut() { return Some(h); }
        self.scene.as_deref_mut().map(|s| s as &mut dyn ObjectScene)
    }

    // -------- map / variant loading (synchronous) --------

    fn resolve_map_index(&self, want: &str) -> Option<usize> {
        let wl = want.to_lowercase();
        if let Some(i) = self.cands.iter().position(|c| c.path.to_string_lossy().eq_ignore_ascii_case(want)) {
            return Some(i);
        }
        self.cands
            .iter()
            .position(|c| c.stem.to_lowercase() == wl)
            .or_else(|| self.cands.iter().position(|c| {
                c.label().to_lowercase().contains(&wl)
                    || c.stem.to_lowercase().contains(&wl)
                    || c.path.to_string_lossy().to_lowercase().contains(&wl)
            }))
    }

    /// Fresh renderer state (drop the previous map's meshes).
    fn clear_renderer_lanes(&mut self) {
        self.renderer.set_static_meshes(Vec::new());
        self.renderer.set_terrain_meshes(Vec::new());
        self.renderer.set_water_meshes(Vec::new());
        self.renderer.set_alphatest_meshes(Vec::new());
        self.renderer.set_foliage_meshes(Vec::new());
        self.renderer.set_blend_meshes(Vec::new());
        self.renderer.set_additive_meshes(Vec::new());
        self.renderer.set_decal_meshes(Vec::new());
        self.renderer.set_sky_meshes(Vec::new());
        self.renderer.set_dynamic_meshes(Vec::new());
        self.renderer.set_dynamic_cutout_meshes(Vec::new());
        self.renderer.set_dynamic_holo_meshes(Vec::new());
        self.renderer.set_dynamic_blend_meshes(Vec::new());
    }

    /// Load a base map fully (blocking), warm the forge palette, resolve base scenery objects.
    fn load_base_map(&mut self, path: &str) -> Result<String, String> {
        // A Halo 4 cache goes through the pure-Rust reader + the H4ObjectScene.
        // #h2a: a groundhog cache reads through the same H4Cache path.
        if crate::h4::is_halo4_cache(std::path::Path::new(path)) || crate::h2a::is_h2a_cache(std::path::Path::new(path)) {
            return self.h4_load_base_map(path, None);
        }
        self.h4_drop(); // a Reach map replaces any Halo 4 state
        self.clear_renderer_lanes();

        let dll = NativeDll::load(&self.dll_path).map_err(|e| format!("native DLL load failed: {e}"))?;
        let mut scene = SceneController::new(dll);
        scene.open_cache(path).map_err(|e| format!("open_cache('{path}') failed: {e}"))?;
        scene.begin_bsp_load();
        self.renderer.set_fog(&self.queue, &scene.fog_uniform());
        self.renderer.set_sky_atmosphere(scene.has_atmosphere());
        self.renderer.set_space_sky(scene.is_space_sky());

        let mesh_arc = self.renderer.mesh_renderer_arc();
        let (dv, qv) = (self.device.clone(), self.queue.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let handle = std::thread::spawn(move || scene::run_load_worker(scene, mesh_arc, dv, qv, tx, cancel));
        let mut done_scene: Option<Box<SceneController>> = None;
        while let Ok(msg) = rx.recv() {
            match msg {
                LoadMsg::Opaque(m) => self.renderer.append_static_meshes(m),
                LoadMsg::Terrain(m) => self.renderer.append_terrain_meshes(m),
                LoadMsg::Water(m) => self.renderer.append_water_meshes(m),
                LoadMsg::WaterPlanes(p) => self.renderer.append_water_planes(p),
                LoadMsg::Alphatest(m) => self.renderer.append_alphatest_meshes(m),
                LoadMsg::Foliage(m) => self.renderer.append_foliage_meshes(m),
                LoadMsg::Blend(m) => self.renderer.append_blend_meshes(m),
                LoadMsg::Additive(m) => self.renderer.append_additive_meshes(m),
                LoadMsg::StaticExtra(m) => self.renderer.append_decal_meshes(m),
                LoadMsg::Sky(s) => self.renderer.set_sky_meshes(s),
                LoadMsg::Progress { .. } => {}
                LoadMsg::Done { scene, exposure } => {
                    self.renderer.set_exposure(&self.queue, exposure);
                    done_scene = Some(scene);
                    break;
                }
            }
        }
        let _ = handle.join();
        let mut scene = done_scene.ok_or_else(|| "load worker died before Done".to_string())?;

        // One-time renderer setup (exposure band / tints / sun / lights).
        if let Some((key, lo, hi)) = scene.autoexposure_band() {
            self.renderer.set_auto_exposure(&self.queue, key, lo, hi);
            // HMS_ENGINE_DECODE / HMS_GEXP (diagnostics, same as headless.rs): a fixed exposure at
            // the cfxs key (the engine's steady-state g_exposure) instead of the meter;
            // HMS_GEXP=<f> overrides the multiplier.
            if std::env::var("HMS_ENGINE_DECODE").is_ok() || std::env::var("HMS_GEXP").is_ok() {
                let g = std::env::var("HMS_GEXP").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(key);
                self.renderer.set_fixed_exposure(&self.queue, g);
            }
        }
        let (st, at) = scene.scene_tints();
        self.renderer.set_scene_tints(st, at);
        // Like main.rs / headless.rs: gate the glass sun lanes by the map's baked sun-visibility
        // mean (this path must render what the app renders).
        self.renderer.set_sun_reach(scene.baked_sun_reach());
        // Diagnostic lighting multipliers, same knobs headless.rs exposes, so a sweep can A/B the
        // analytical sun's contribution across many views in ONE map load.
        {
            let f = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
            self.renderer.set_lighting_mults(f("HMS_SUN_MULT"), f("HMS_AMB_MULT"), f("HMS_LM_MULT"));
        }
        // HMS_DIAG prints the sky light and sun direction.
        let sky_light_dbg = scene.sky_light();
        if std::env::var("HMS_DIAG").is_ok() {
            match &sky_light_dbg {
                Some(s) => eprintln!("HMS_DIAG SKYLIGHT has_light={} ambient=[{:.4},{:.4},{:.4}]", s.has_light, s.ambient[0], s.ambient[1], s.ambient[2]),
                None => eprintln!("HMS_DIAG SKYLIGHT none"),
            }
        }
        self.renderer.set_obj_dir_strength(scene.obj_dir_strength());
        if let Some(sun) = scene.scene_sun_dir() {
            self.renderer.set_sun_dir(Vec3::from(sun));
            if std::env::var("HMS_DIAG").is_ok() {
                eprintln!("HMS_DIAG SUNDIR=[{:.3},{:.3},{:.3}]", sun[0], sun[1], sun[2]);
            }
        }
        // Sun lens-flare elements (headless too, so captures show the flare). Mirrors main.rs.
        let mut lens = scene.lens_flare();
        // HMS_LENS_TEST (diagnostic) injects synthetic elements (8 floats: axis, radius, bright,
        // r, g, b, tint_power, mod) to check the shaping on maps whose scenario resolves no
        // sun-light lens tag (Forge).
        if std::env::var("HMS_LENS_TEST").is_ok() {
            lens.extend_from_slice(&[
                0.05, 0.6, 1.0, 1.0, 0.6, 0.3, 2.0, 1.0,   // soft warm halo (low tint_power)
                0.20, 0.35, 1.0, 0.4, 0.9, 1.0, 9.0, 1.0,  // tight cyan core (high tint_power)
                0.38, 0.5, 1.0, 0.4, 0.5, 1.0, 1.5, 1.0,   // soft blue (very low tint_power)
                0.55, 0.3, 1.0, 1.0, 0.3, 0.3, 6.0, 1.0,   // tight red
            ]);
        }
        self.renderer.set_lens_flare(lens);
        let sl = scene.simple_lights();
        self.renderer.set_simple_lights(&self.queue, (sl.len() / 20) as u32, &sl);
        // Authored fog volumes for the screen-space depth composite (headless too, so captures
        // show it). Mirrors the interactive path in main.rs.
        let mut pfog = scene.planar_fog_volumes();
        // HMS_PFOG_TEST (diagnostic) injects a synthetic volume (plane z=18, big footprint) to
        // check the pass on maps that author none (Forge).
        if std::env::var("HMS_PFOG_TEST").is_ok() {
            pfog.push(hms_render::PlanarFogVolume {
                plane: [0.0, 0.0, 1.0, -18.0], // normal +Z, plane at z=18 (fog below)
                color: [0.35, 0.55, 0.7],
                density: 0.6,
                bmin: [-400.0, -400.0, -200.0],
                bmax: [400.0, 400.0, 18.0],
                base_depth: 20.0, // exercises the quadratic near-plane soften
            });
        }
        self.renderer.set_planar_fog_volumes(pfog);

        // Base scenery objects (constant across variants). Placements of objects in the map's
        // Forge sandbox palette are canvas / default-variant content the map variant owns, so
        // they are never placed or tracked from the scenario (same filter as main.rs /
        // headless.rs; otherwise the canvas's 300+ spawn markers draw on top of the variant's
        // own). HMS_SCNR_FORGE=1 (diagnostic) places them anyway.
        let forge_owned: std::collections::HashSet<u32> = if std::env::var("HMS_SCNR_FORGE").is_ok() {
            Default::default()
        } else {
            scene.forge_owned_tags()
        };
        let scnr_all = scene.scenario_objects();
        // Which obj tags are spawn-family markers (respawn_point_invisible etc.).
        self.base_spawn_tags = crate::map_spawns::spawn_obj_tags(scnr_all.iter().map(|o| o.obj_tag), |t| scene.tag_name_of(t));
        self.base_ident.clear();
        let mut base_ident = HashMap::new();
        self.base_all = scnr_all
            .into_iter()
            .filter(|o| o.category != 8)
            .filter(|o| (o.placement_flags & 0x41) == 0)
            .filter(|o| !forge_owned.contains(&o.obj_tag))
            .enumerate()
            .map(|(i, o)| {
                let datum = 0xE000_0000u32.wrapping_add(i as u32);
                base_ident.insert(datum, (o.category, o.palette_index, o.name_index));
                ObjectInfo {
                datum,
                type_sig: 0, sig0: 0, sig1: 0,
                pos: o.pos, health: 1.0, shield: 1.0,
                mode_tag: o.mode_tag, fwd: o.fwd, up: o.up,
                attached: [0; 8], primary_tag: o.obj_tag,
                variant_name_sid: 0,
            }})
            .collect();
        self.base_ident = base_ident;
        self.base_objects = self.base_objects_visible();

        // Warm the whole forge palette so variant loads are decode-free (HMS_NO_PREWARM skips the
        // warm-up, for load-time profiling); build a placement palette.
        self.palette = scene.forge_palette_full();
        self.types = scene.forge_type_order(&self.palette);
        let tags = scene.forge_palette_model_tags();
        let n_warm = if std::env::var("HMS_NO_PREWARM").is_ok() { 0 } else { scene.predecode_tags(&tags) };
        self.place_palette.clear();
        {
            let mut seen = std::collections::HashSet::new();
            for e in &self.palette {
                let mode = scene.resolve_object_mode(e.tag_short);
                if mode == 0 || mode == 0xFFFF_FFFF {
                    continue;
                }
                let name = if !e.variant_name.is_empty() { e.variant_name.clone() } else { e.name.clone() };
                if seen.insert((e.tag_short, name.clone())) {
                    self.place_palette.push((mode, e.tag_short, name, e.variant_name_sid));
                }
            }
        }

        self.scene = Some(scene);
        self.map_path = path.to_string();
        self.map_id = mapcat::read_map_id(std::path::Path::new(path)).or_else(|| {
            self.cands.iter().find(|c| c.path.to_string_lossy() == path).and_then(|c| c.map_id)
        });
        self.objects = self.base_objects.clone();
        self.colors.clear();
        self.meta.clear();
        self.screenfx_cache.clear();
        self.h4_screenfx_cache.clear();
        self.screenfx_pushed = None;
        self.cur_stem = std::path::Path::new(path)
            .file_stem().unwrap_or_default().to_string_lossy().into_owned();
        self.dirty = true;
        // Default camera: frame the scene.
        self.frame_scene();
        Ok(format!("loaded map {} (palette {} models warmed, {} base objects; {})", self.cur_stem, n_warm, self.base_objects.len(), self.map_spawns_status()))
    }

    /// The scenario placements to draw under the current `show_map_spawns`.
    fn base_objects_visible(&self) -> Vec<ObjectInfo> {
        self.base_all
            .iter()
            .filter(|o| self.show_map_spawns || !self.base_spawn_tags.contains(&o.primary_tag))
            .cloned()
            .collect()
    }

    fn map_spawns_status(&self) -> String {
        let n = self.base_all.iter().filter(|o| self.base_spawn_tags.contains(&o.primary_tag)).count();
        crate::map_spawns::status_line(self.show_map_spawns, n)
    }

    /// "hard floor: shown (1 playable BSP(s) of 2; floor z -25.0)".
    fn hard_floor_status(&self) -> String {
        let wb = self.scene.as_ref().and_then(|s| crate::hard_floor::world_box(s.structure_bsp_flags(), &s.structure_bsp_mopp_bounds()));
        crate::hard_floor::status_line(self.show_hard_floor, wb.as_ref())
    }

    fn playable_bounds_status(&self) -> String {
        let bsps = self.scene.as_ref().map(|s| s.structure_bsp_flags()).unwrap_or(&[]);
        crate::hard_floor::playable_status_line(self.show_playable_bounds, bsps)
    }

    /// "soft ceilings: shown (5 ceiling(s): ...)".
    fn soft_ceilings_status(&self) -> String {
        let cs = self.scene.as_ref().map(|s| s.soft_ceilings()).unwrap_or_default();
        crate::soft_ceilings::status_line(self.show_soft_ceilings, &cs)
    }

    /// #wire-visible  "selection wireframe: draws THROUGH objects (N edge(s) selected)".
    fn wire_xray_status(&self) -> String {
        let n: usize = self.objscene()
            .map(|sc| self.selected.iter().filter_map(|d| sc.selection_wireframe(*d)).map(|w| w.len()).sum())
            .unwrap_or(0);
        crate::wire_xray::status_line(self.wire_xray, n)
    }

    // ===================== #quant-error  save-precision prediction =====================

    /// The quantiser the next save would use. Reach and Halo 4 / H2A both seed `variant_globals`
    /// on load (`seed_globals`), so the box is the edited one when `variant set bounds` moved it
    /// and the file's otherwise -- exactly what the encoders write with. Halo 3 has no headless
    /// path at all, so it cannot appear here.
    fn pose_quant(&self) -> Option<crate::quant_error::PoseQuant> {
        let g = self.variant_globals.as_ref()?;
        let b = if self.global_edits.bounds != g.bounds { self.global_edits.bounds } else { g.bounds };
        let game = if self.h4_variant.is_some() { crate::quant_error::QuantGame::Halo4 } else { crate::quant_error::QuantGame::Reach };
        Some(crate::quant_error::PoseQuant::new(game, b))
    }

    fn pose_quant_game_name(&self) -> &'static str {
        match self.h4_variant.as_ref() {
            Some((_, v)) if v.chunk_version == crate::h4::mvar::H2A_MVAR_VERSION => "Halo 2 Anniversary",
            Some(_) => "Halo 4",
            None => "Halo: Reach",
        }
    }

    /// Every object a save would write (the ones with an editable record), with its predicted
    /// error. Scenario placements are not in the variant, so they are not here.
    fn predict_all_pose_errors(&self) -> Vec<crate::quant_error::WorstRow> {
        let Some(q) = self.pose_quant() else { return Vec::new() };
        self.objects.iter()
            .filter(|o| self.meta.contains_key(&o.datum))
            .map(|o| crate::quant_error::WorstRow {
                datum: o.datum,
                name: self.meta.get(&o.datum).map(|m| m.name.clone()).unwrap_or_default(),
                err: q.predict(o.pos, o.fwd, o.up),
            })
            .collect()
    }

    /// Move objects onto the nearest pose the file stores exactly. The host keeps no undo
    /// history, so this is a straight edit of the live objects (and of the Halo 4 records a save
    /// starts from), reported the same way the app reports it.
    fn snap_datums_to_representable(&mut self, datums: &[u32]) -> Result<String, String> {
        let q = self.pose_quant().ok_or("no variant is open, so there is no save quantiser to snap to")?;
        let mut work: Vec<(usize, [f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        for &d in datums {
            let Some(i) = self.movable_index(d) else { continue };
            if !self.meta.contains_key(&d) { continue; }
            let o = &self.objects[i];
            let e = q.predict(o.pos, o.fwd, o.up);
            if e.is_exact() { continue; }
            work.push((i, e.saved_pos, e.saved_fwd, e.saved_up));
        }
        if work.is_empty() {
            return Ok(format!("snap: {} object(s) already survive the save exactly - nothing to do", datums.len()));
        }
        let n = work.len();
        let (mut moved, mut turned) = (0.0f32, 0.0f32);
        for (i, p, f, u) in work {
            let o = &mut self.objects[i];
            moved = moved.max((glam::Vec3::from(p) - glam::Vec3::from(o.pos)).length());
            turned = turned.max(crate::quant_error::basis_angle_deg(o.fwd, o.up, f, u));
            o.pos = p;
            o.fwd = f;
            o.up = u;
            let d = o.datum;
            if let Some(rec) = self.h4_records.get_mut(&d) {
                rec.pos = p;
                rec.set_orientation(f, u);
            }
        }
        self.dirty = true;
        Ok(format!("snap: {n} object(s) moved onto the nearest representable pose (up to {moved:.6} wu / {turned:.5} deg); predicted save error is now zero"))
    }

    /// #save-precision  After a save: adopt the poses the written file decodes to (the host has no
    /// undo stack, so this is a straight edit of the live objects and the records a save starts from).
    fn apply_save_precision(&mut self, written: &crate::h4::mvar::H4Variant) -> String {
        let q = crate::quant_error::PoseQuant::new(crate::quant_error::QuantGame::Halo4, written.bounds);
        let bits: Vec<crate::quant_error::PoseBits> = written.objects.iter().map(|o| crate::quant_error::pose_bits(o.pos, o.fwd, o.up)).collect();
        let objs: Vec<(u32, [f32; 3], [f32; 3], [f32; 3])> = self.objects.iter().filter(|o| self.meta.contains_key(&o.datum)).map(|o| (o.datum, o.pos, o.fwd, o.up)).collect();
        let name_of = |d: u32| self.meta.get(&d).map(|m| m.name.clone()).unwrap_or_default();
        let (changes, report) = crate::quant_error::plan_save_precision(&q, &bits, &objs, &name_of, self.save_precision_on, self.pose_quant_game_name());
        if self.save_precision_on {
            for c in &changes {
                let Some(i) = self.movable_index(c.datum) else { continue };
                let o = &mut self.objects[i];
                o.pos = c.new_pos;
                o.fwd = c.new_fwd;
                o.up = c.new_up;
                if let Some(rec) = self.h4_records.get_mut(&c.datum) {
                    rec.pos = c.new_pos;
                    rec.set_orientation(c.new_fwd, c.new_up);
                }
            }
        }
        let line = report.status();
        self.last_save_report = Some(report);
        line
    }

    /// The ghost's violet wireframe for the selected object(s), appended to the depth-ALWAYS
    /// lane. Mirrors `App::push_quant_ghost` through the shared `ghost_transform`.
    fn push_quant_ghost(&self, xray: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>) {
        if !self.quant_ghost.on { return; }
        let Some(q) = self.pose_quant() else { return };
        let k = self.quant_ghost.exaggerate;
        for o in self.objects.iter().filter(|o| self.selected.contains(&o.datum) && self.meta.contains_key(&o.datum)) {
            let e = q.predict(o.pos, o.fwd, o.up);
            if e.is_exact() { continue; }
            let xf = crate::quant_error_ui::ghost_transform(o.pos, o.fwd, o.up, e.saved_pos, e.saved_fwd, e.saved_up, k);
            match self.objscene().and_then(|s| s.selection_wireframe(o.datum)) {
                Some(mut w) => {
                    crate::objscene::fit_wire_budget(&mut w, crate::quant_error_ui::GHOST_MAX_LINES);
                    for seg in w.chunks_exact(2) {
                        xray.push((
                            xf.transform_point3(Vec3::from(seg[0])).to_array(),
                            xf.transform_point3(Vec3::from(seg[1])).to_array(),
                            crate::quant_error_ui::GHOST_RGB,
                        ));
                    }
                }
                None => {
                    let Some(obb) = self.objscene().and_then(|s| s.object_obb(o.datum)) else { continue };
                    for (a, b) in obb.edge_segments() {
                        xray.push((xf.transform_point3(a).to_array(), xf.transform_point3(b).to_array(), crate::quant_error_ui::GHOST_RGB));
                    }
                }
            }
        }
    }

    /// `precision [get] | <datum> | worst [n] | snap [selection|all|<datum>] | ghost [...]`
    fn precision_cmd(&mut self, sub: &str, args: &[String]) -> Result<String, String> {
        match sub {
            "" | "get" | "summary" | "variant" => {
                let q = self.pose_quant().ok_or("no variant is open, so there is no save quantiser")?;
                let rows = self.predict_all_pose_errors();
                let mut st = crate::quant_error::ErrStats::default();
                for r in &rows { st.add(&r.err); }
                let step = q.pos_step();
                Ok(format!(
                    "{} save precision\n  world-bounds x {:.2}..{:.2}  y {:.2}..{:.2}  z {:.2}..{:.2}\n  position {} / {} / {} bits -> grid {:.6} / {:.6} / {:.6} wu (worst error = half of each)\n  rotation worst {:.4} deg\n  {} object(s): {} exactly representable, {} quantising badly\n  position error mean {:.6} wu, max {:.6} wu\n  rotation error mean {:.5} deg, max {:.5} deg",
                    self.pose_quant_game_name(),
                    q.bounds[0], q.bounds[1], q.bounds[2], q.bounds[3], q.bounds[4], q.bounds[5],
                    q.axis_bits[0], q.axis_bits[1], q.axis_bits[2], step[0], step[1], step[2],
                    q.rot_worst_deg(), st.count, st.exact, st.bad,
                    st.pos_mean(), st.pos_max, st.rot_mean(), st.rot_max
                ))
            }
            "worst" => {
                let n = args.first().and_then(|s| s.parse::<usize>().ok()).unwrap_or(10).clamp(1, 200);
                let rows = self.predict_all_pose_errors();
                if rows.is_empty() { return Err("no variant is open (or it has no placed objects)".into()); }
                let (by_pos, by_rot) = crate::quant_error::rank_worst(rows, n);
                let mut s = format!("worst {n} by POSITION error:\n");
                for r in &by_pos {
                    s.push_str(&format!("  0x{:08X}  {:.6} wu  ({:+.6} {:+.6} {:+.6})  {}\n", r.datum, r.err.pos_total, r.err.pos_delta[0], r.err.pos_delta[1], r.err.pos_delta[2], r.name));
                }
                s.push_str(&format!("worst {n} by ROTATION error:\n"));
                for r in &by_rot {
                    s.push_str(&format!("  0x{:08X}  {:.5} deg  (roll {:+.4} pitch {:+.4} yaw {:+.4})  {}\n", r.datum, r.err.rot_deg, r.err.euler_delta_deg[0], r.err.euler_delta_deg[1], r.err.euler_delta_deg[2], r.name));
                }
                Ok(s)
            }
            "snap" => {
                let targets: Vec<u32> = match args.first().map(|s| s.to_lowercase()).as_deref() {
                    None | Some("selection") | Some("sel") => {
                        if self.selected.is_empty() { return Err("precision snap: nothing is selected (use `precision snap all`, or a datum)".into()); }
                        self.selected.clone()
                    }
                    Some("all") => self.objects.iter().filter(|o| self.meta.contains_key(&o.datum)).map(|o| o.datum).collect(),
                    Some(_) => vec![crate::script::parse_datum(&args[0])?],
                };
                self.snap_datums_to_representable(&targets)
            }
            "report" => match &self.last_save_report {
                Some(r) => Ok(r.text()),
                None => Err("no save has happened yet in this run".into()),
            },
            "mode" | "on" | "off" => {
                let want = match (sub, args.first().map(|s| s.to_lowercase()).as_deref()) {
                    ("on", _) => Some(true),
                    ("off", _) => Some(false),
                    (_, Some(a)) => crate::map_spawns::parse_flag(a),
                    _ => None,
                };
                if let Some(on) = want { self.save_precision_on = on; }
                Ok(format!("save precision: {}", if self.save_precision_on { "ON (after save the objects adopt the poses the game will load)" } else { "off (saving leaves the objects as authored)" }))
            }
            "ghost" => {
                let mut g = self.quant_ghost;
                for a in args {
                    let al = a.to_lowercase();
                    if al == "get" { continue; }
                    if let Some(on) = crate::map_spawns::parse_flag(&al) {
                        g.on = on;
                    } else if let Some(f) = al.trim_start_matches('x').parse::<f32>().ok().filter(|v| v.is_finite()) {
                        g.exaggerate = f.clamp(1.0, crate::quant_error_ui::EXAGGERATE_MAX);
                    } else {
                        return Err(format!("precision ghost: '{a}' is not on|off|get or an exaggeration factor"));
                    }
                }
                self.quant_ghost = g;
                let n = self.objects.iter().filter(|o| self.selected.contains(&o.datum) && self.meta.contains_key(&o.datum)).count();
                Ok(format!(
                    "save-precision ghost: {}{} - {n} selected object(s) (run-local in the batch host; the next `screenshot` draws it)",
                    if g.on { "on" } else { "off" },
                    if g.on && g.exaggerate > 1.0 { format!(", EXAGGERATED x{:.0}", g.exaggerate) } else { String::new() }
                ))
            }
            other => {
                let d = crate::script::parse_datum(other)?;
                let q = self.pose_quant().ok_or("no variant is open, so there is no save quantiser")?;
                let o = self.objects.iter().find(|o| o.datum == d).ok_or_else(|| format!("no object 0x{d:08X}"))?;
                let e = q.predict(o.pos, o.fwd, o.up);
                let step = q.pos_step();
                Ok(format!(
                    "0x{d:08X} ({})\n  live    {:.6} {:.6} {:.6}\n  stored  {:.6} {:.6} {:.6}\n  {}\n  grid {:.6} / {:.6} / {:.6} wu; worst-case use {:.0}% of position, {:.0}% of rotation{}",
                    self.pose_quant_game_name(),
                    o.pos[0], o.pos[1], o.pos[2], e.saved_pos[0], e.saved_pos[1], e.saved_pos[2],
                    e.summary(), step[0], step[1], step[2], e.pos_frac * 100.0, e.rot_frac * 100.0,
                    if e.is_exact() { "  - EXACT, nothing is lost" } else if e.quantises_badly() { "  - QUANTISES BADLY, nudge or snap it" } else { "" }
                ))
            }
        }
    }

    /// #h4-phys "collision hull: shown (2190 edge(s)) | physics hull: hidden (192 edge(s))" -
    /// the counts come from the SELECTED object through the trait, so a batch run can measure the
    /// two overlays on either game without a window.
    fn hull_overlay_status(&self) -> String {
        let (mut ce, mut pe) = (0usize, 0usize);
        // #settle the world extent of each hull too, so a script can prove a scaled object's hull
        // is drawn / simulated at its scale
        let mut ext = String::new();
        if let (Some(&d), Some(s)) = (self.selected.first(), self.objscene()) {
            if let Some(o) = self.objects.iter().find(|o| o.datum == d) {
                let extent = |edges: &[([f32; 3], [f32; 3])]| -> String {
                    if edges.is_empty() { return String::new(); }
                    let mut mn = Vec3::splat(f32::MAX);
                    let mut mx = Vec3::splat(f32::MIN);
                    for (a, b) in edges { mn = mn.min(Vec3::from(*a)).min(Vec3::from(*b)); mx = mx.max(Vec3::from(*a)).max(Vec3::from(*b)); }
                    let e = mx - mn;
                    format!(" extent {:.3} x {:.3} x {:.3} wu, z [{:.3}, {:.3}]", e.x, e.y, e.z, mn.z, mx.z)
                };
                let cw = s.collision_world_edges(o);
                let pw = s.physics_world_edges(o);
                ce = cw.len();
                pe = pw.len();
                ext = format!(" | collision{} | physics{}", extent(&cw), extent(&pw));
                if let Some(b) = s.physics_body(o) {
                    let (bmn, bmx) = b.aabb();
                    ext.push_str(&format!(" | settle body: {} at scale x{:.2}, local box {:.3} x {:.3} x {:.3} wu", b.desc, b.scale, bmx.x - bmn.x, bmx.y - bmn.y, bmx.z - bmn.z));
                }
            }
        }
        let (n, natt) = match (self.selected.first(), self.objscene()) {
            (Some(&d), Some(s)) => s.pick_entry_counts(d),
            _ => (0, 0),
        };
        format!(
            "collision hull: {} ({ce} edge(s)) | physics hull: {} ({pe} edge(s)) | selection geometry: {n} pick entr(ies), {natt} attachment(s){ext}",
            if self.show_collision_hull { "shown" } else { "hidden" },
            if self.show_physics_hull { "shown" } else { "hidden" },
        )
    }

    /// "hidden-block hulls: shown (252 hull triangle(s))".
    fn physics_outlines_status(&self) -> String {
        let tris = self.objscene().map(|s| s.blocker_overlay().0.len() / 3).unwrap_or(0);
        crate::physics_outlines::status_line(self.show_physics_outlines, tris)
    }

    /// `mapspawns on|off`: swap the drawn scenario head of `objects` (the
    /// variant / placed tail is kept as is, so selection, colours and meta are untouched).
    fn set_map_spawns(&mut self, on: bool) {
        if on == self.show_map_spawns { return; }
        self.show_map_spawns = on;
        let tail: Vec<ObjectInfo> = self.objects.iter().skip(self.base_objects.len()).cloned().collect();
        self.base_objects = self.base_objects_visible();
        self.objects = self.base_objects.clone();
        self.objects.extend(tail);
        self.dirty = true;
    }

    /// Apply a .mvar: ensure its base map is loaded, then resolve its forge objects.
    fn apply_variant(&mut self, path: &str) -> Result<String, String> {
        // A Halo 4 variant (mvar chunk v50) never meets the Reach codec.
        if crate::h4::mvar::is_h4_variant(std::path::Path::new(path)) {
            return self.h4_apply_variant(path);
        }
        let t_parse0 = Instant::now();
        let variant = mvar::parse_variant(std::path::Path::new(path))
            .ok_or_else(|| format!("cannot parse variant '{path}'"))?;
        let stem = std::path::Path::new(path).file_stem().unwrap_or_default().to_string_lossy().into_owned();
        self.apply_variant_value(variant, stem, t_parse0.elapsed())
    }

    /// Apply an already-parsed variant (a file's, or the film's own in-memory one -- see
    /// `film_export::variant_from_film`). `stem` names the scene for later captures.
    fn apply_variant_value(&mut self, variant: mvar::Variant, stem: String, t_parse: std::time::Duration) -> Result<String, String> {
        let t_all = Instant::now();
        self.seed_globals(variant.globals.clone());
        // The base-map (BSP + palette pre-decode) load is a one-time cost excluded from the
        // between-variant timing: it only fires when the base map id changes. Reported
        // separately ("skip" on reuse).
        let mut t_base = std::time::Duration::ZERO;
        let mut base_loaded = false;
        if self.scene.is_none() || self.map_id != Some(variant.map_id) || mvar::is_unassigned_map_id(variant.map_id) {
            let idx = mapcat::find_map_for_variant(&self.cands, variant.map_id, variant.mcc_map_id)
                .ok_or_else(|| format!("no installed base map for variant map_id 0x{:04X}", variant.map_id))?;
            let base = self.cands[idx].path.to_string_lossy().into_owned();
            let t_base0 = Instant::now();
            self.load_base_map(&base)?;
            t_base = t_base0.elapsed();
            base_loaded = true;
        }
        let t_build0 = Instant::now();
        // Reset to base scenery, then add the variant's forge objects.
        self.objects = self.base_objects.clone();
        self.colors.clear();
        self.meta.clear();
        let scene = self.scene.as_ref().unwrap();
        let mut loadout_cam: Option<([f32; 3], [f32; 3])> = None;
        // Not every variant places a spawning_camera. Collect the initial (and any) spawn
        // markers too so a variant with no authored loadout camera still previews from a
        // player's-eye view at a spawn, rather than the bird's-eye frame_forge fallback.
        let mut initial_spawns: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
        let mut any_spawns: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
        let mut placed_pos: Vec<glam::Vec3> = Vec::new();
        let mut placed = 0usize;
        for o in variant.objects.iter() {
            let (mode_tag, obj_tag, variant_sid) =
                scene.resolve_forge_model_v(&self.palette, &self.types, o.folder, o.item);
            let name = self
                .types
                .get(o.folder as usize)
                .and_then(|&(pi, ew)| self.palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.clone()))
                .unwrap_or_default();
            if loadout_cam.is_none() && name.contains("spawning_camera") {
                loadout_cam = Some((o.pos, o.fwd));
            }
            // Spawn markers are in-bounds even though they're skipped from the render below —
            // capture them for the camera fallback before the `continue` drops them.
            if name.contains("spawning_initial") {
                initial_spawns.push((glam::Vec3::from(o.pos), glam::Vec3::from(o.fwd)));
            } else if name.contains("spawning_respawn") {
                any_spawns.push((glam::Vec3::from(o.pos), glam::Vec3::from(o.fwd)));
            }
            // Markers (spawns / respawn / objective) are skipped in preview shots;
            // HMS_SHOW_MARKERS (diagnostic) renders them.
            let is_marker = name.starts_with("spawning_") || name.starts_with("respawn_") || name.starts_with("obj_");
            if (is_marker && std::env::var("HMS_SHOW_MARKERS").is_err()) || mode_tag == 0 || mode_tag == 0xFFFF_FFFF {
                continue;
            }
            if o.pos.iter().any(|c| c.abs() > 8000.0) {
                continue; // parked/unplaced palette item
            }
            let datum = 0xD000_0000u32.wrapping_add(placed as u32);
            let cu = if o.color < 0 { 0xFFu8 } else { o.color as u8 };
            self.colors.insert(datum, (o.team, cu));
            let label = variant.labels.get(o.label_idx as usize).cloned().unwrap_or_default();
            self.meta.insert(datum, HMeta {
                name: name.clone(),
                team: o.team,
                color: o.color,
                cached_type: o.cached_type,
                spawn_seq: o.spawn_seq as i32,
                respawn: o.respawn,
                label,
                boundary_shape: o.boundary_shape,
                boundary: o.boundary,
                flags: Default::default(),
                h4_scale: None,
            });
            self.objects.push(ObjectInfo {
                datum,
                type_sig: 0, sig0: 0, sig1: 0,
                pos: o.pos, health: 1.0, shield: 1.0,
                mode_tag, fwd: o.fwd, up: o.up,
                attached: [0; 8], primary_tag: obj_tag,
                variant_name_sid: variant_sid,
            });
            placed_pos.push(glam::Vec3::from(o.pos));
            placed += 1;
        }
        self.cur_stem = stem;
        self.dirty = true;
        // Camera priority — mirror the "initial loadout camera" intent on every variant:
        //   1. the authored spawning_camera object (the true initial loadout camera)
        //   2. else stand at the initial spawn nearest the map centre, look into the map
        //   3. else any (respawn) spawn, same standing pose
        //   4. else the scenario's built-in spawn proxy
        //   5. else frame the placed objects (bird's-eye — last resort)
        let centroid = if placed_pos.is_empty() {
            Vec3::ZERO
        } else {
            placed_pos.iter().fold(Vec3::ZERO, |a, &p| a + p) / (placed_pos.len() as f32)
        };
        let nearest = |spawns: &[(Vec3, Vec3)]| -> Option<(Vec3, Vec3)> {
            spawns
                .iter()
                .min_by(|a, b| (a.0 - centroid).length_squared().total_cmp(&(b.0 - centroid).length_squared()))
                .copied()
        };
        // Stand at a spawn marker (~player eye height) and look along its forward, or toward the
        // map centre when the spawn faces nowhere useful.
        let stand_at = |eye_spawn: Vec3, fwd: Vec3, this: &mut Self| {
            let eye = eye_spawn + Vec3::Z * 0.62;
            let dir = if fwd.x.hypot(fwd.y) > 0.1 { fwd } else { centroid - eye };
            this.set_camera_look(eye, dir);
        };
        let cam_src: &str;
        if let Some((pos, fwd)) = loadout_cam {
            self.set_camera_look(Vec3::from(pos), Vec3::from(fwd));
            cam_src = "loadout-camera";
        } else if let Some((spawn, fwd)) = nearest(&initial_spawns) {
            stand_at(spawn, fwd, self);
            cam_src = "initial-spawn";
        } else if let Some((spawn, fwd)) = nearest(&any_spawns) {
            stand_at(spawn, fwd, self);
            cam_src = "respawn";
        } else if let Some((p, yaw, pitch)) = self.scene.as_ref().and_then(|s| s.spawn_camera_pose()) {
            self.camera.pos = p;
            self.camera.yaw = yaw;
            self.camera.pitch = pitch;
            cam_src = "scenario-spawn";
        } else {
            self.frame_forge();
            cam_src = "frame";
        }
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let build = ms(t_build0.elapsed());
        let base = if base_loaded { format!("{:.1}ms", ms(t_base)) } else { "skip".into() };
        Ok(format!(
            "loaded variant {} ({placed} forge objects) | cam={cam_src} initial_spawns={} | LOAD parse={:.1}ms base={base} build={:.1}ms total={:.1}ms",
            self.cur_stem, initial_spawns.len(), ms(t_parse), build, ms(t_all.elapsed())
        ))
    }

    // -------- rendering / capture --------

    fn ensure_built(&mut self) {
        if !self.dirty {
            return;
        }
        // The active object scene: Halo 4 while one is loaded, else the Reach controller.
        let scene: &mut dyn ObjectScene = if let Some(h) = self.h4_scene.as_deref_mut() { h } else if let Some(s) = self.scene.as_deref_mut() { s } else { return };
        // Same derivation as the GUI's rebuild tick: per-object flags under the globals.
        let mut scales = HashMap::new();
        let mut casters = std::collections::HashSet::new();
        for (d, m) in &self.meta {
            let (fs, fc) = forge_scale::effective_flags(&self.globals, &m.flags, m.team, &m.label);
            // A Halo 4 object scales by the gametype rule only (`m.scale()` is 1.0 otherwise;
            // the record field is not drawn by MCC); the global switch still zeroes it.
            let s = if m.h4_scale.is_some() { if self.globals.scaled { m.scale() } else { 1.0 } } else if fs { m.scale() } else { 1.0 };
            if (s - 1.0).abs() > 1e-3 {
                scales.insert(*d, s);
            }
            if fc {
                casters.insert(*d);
            }
        }
        // #damage the DISPLAYED damage state per object (only overrides)
        let damage: HashMap<u32, u8> = self.meta.iter().filter_map(|(d, m)| m.flags.damage.filter(|s| *s > 0).map(|s| (*d, s))).collect();
        scene.set_object_scales(scales);
        scene.set_object_casters(casters);
        scene.set_object_damage(damage);
        let mut obj_meshes = None;
        for _ in 0..4096 {
            obj_meshes = scene
                .maybe_rebuild(&self.objects, &self.colors, self.renderer.mesh_renderer(), &self.device, &self.queue)
                .or(obj_meshes);
            if !scene.rebuild_pending() {
                break;
            }
        }
        if let Some(l) = obj_meshes {
            self.renderer.set_dynamic_meshes(l.opaque);
            self.renderer.set_dynamic_cutout_meshes(l.cutout);
            self.renderer.set_dynamic_holo_meshes(l.holo);
            self.renderer.set_dynamic_holo_solid_meshes(l.holo_solid);
            self.renderer.set_dynamic_blend_meshes(l.blend);
        }
        self.dirty = false;
    }

    /// #film-camera  Put the VIEW where the film camera is, and give a first-person camera
    /// Reach's own field of view.
    ///
    /// Called from `screenshot`, NOT from the `film scene` handler where this logic used to sit.
    /// That placement made the effect depend on VERB ORDER: the natural script is
    /// `film scene` ... then ... `film camera firstbody N`, and selecting the camera after the
    /// scene meant nothing re-applied it, so every `shot` kept firing from the map-load camera.
    /// Six different `firstbody` targets produced six frames differing by a maxdiff of 6 — which
    /// is exactly how a first-person view can be "checked" from a script and found fine while the
    /// window shows something else. Applying it at capture time makes it order-independent.
    fn apply_film_camera(&mut self) {
        let Some(pb) = self.film_play.as_ref() else { return };
        if matches!(pb.camera, crate::film_play::FilmCamera::Free) {
            // Leaving free camera restores whatever FOV was parked when first person took over.
            if let Some(f) = self.fov_before_first_person.take() {
                self.camera.fov_y = f;
            }
            return;
        }
        let pbc = pb.clone();
        // Whether the camera RESOLVED, not merely which one was asked for. A camera naming an
        // owner group or slot that this film does not have solves to nothing and leaves the view
        // wherever it was -- and a line that printed the REQUEST would then label a stale frame
        // with the camera the script wanted, which is the same false confidence that let the
        // order-dependence bug above go unseen.
        self.film_camera_resolved = false;
        if let Some(p) = crate::film_play::solve_camera_at(&pbc, self.film_driver.eye_height(&pbc)) {
            self.camera.pos = Vec3::from(p.pos);
            self.camera.yaw = p.yaw;
            self.camera.pitch = p.pitch;
            self.film_camera_resolved = true;
        }
        if pbc.camera.is_first_person() {
            if self.fov_before_first_person.is_none() {
                self.fov_before_first_person = Some(self.camera.fov_y);
            }
            self.camera.fov_y = crate::film_play::reach_vertical_fov_for(self.fp_fov_deg);
        } else if let Some(f) = self.fov_before_first_person.take() {
            self.camera.fov_y = f;
        }
    }

    /// #armour-fx Hand the ACTOR PARTICLE LANE to the scene, from the pose the actors are
    /// currently in. Called after every `ActorScene::update` here, exactly as the window's
    /// `update_actors` does it, so a scripted capture and the window simulate the same sources.
    /// #fp-fov Install one actor rebuild's two lane sets: the world's, and the FIRST-PERSON RIG's
    /// (drawn through the engine's narrower first-person projection).
    fn install_actor_build(&mut self, b: crate::reach_actor::ActorBuild) {
        self.renderer.set_first_person_fov_scale(crate::film_play::first_person_fov_scale());
        let (l, f) = (b.world, b.first_person);
        self.renderer.set_actor_meshes(l.opaque, l.cutout, l.holo, l.holo_solid, l.blend);
        self.renderer.set_first_person_meshes(f.opaque, f.cutout, f.holo, f.holo_solid, f.blend);
    }

    fn push_actor_particles(&mut self) {
        let srcs = self.actors.particle_sources();
        if let Some(sc) = self.scene.as_mut() {
            if !srcs.is_empty() || sc.actor_particle_source_count() > 0 {
                sc.set_actor_particle_sources(srcs);
            }
        }
        // #projectile  The FILM lane (muzzle flashes, projectile trails, detonations), from the
        // pose that has just been written. The same two steps the GUI host takes, so a scripted
        // capture and the app install the same sources.
        let want_fx = match (self.film_play.as_ref(), self.scene.as_ref()) {
            (Some(pb), Some(scene)) => match scene.cache_handle() {
                Some(cache) => {
                    let native = scene.native_dll();
                    self.film_driver.film_effects(pb, &self.actors, native, cache)
                }
                None => Vec::new(),
            },
            _ => Vec::new(),
        };
        if let Some(sc) = self.scene.as_mut() {
            if !want_fx.is_empty() || sc.film_particle_source_count() > 0 {
                self.film_fx = crate::film_projectile::install(sc, &want_fx);
            }
            if !self.film_driver.beams.is_empty() || !self.film_driver.contrails.is_empty() || sc.film_beam_count() > 0 {
                crate::film_projectile::install_beams(sc, &self.film_driver.beams);
                crate::film_projectile::install_contrails(sc, self.film_driver.contrails.clone());
            }
        }
    }

    fn screenshot(&mut self, path: &str, size: Option<(u32, u32)>) -> Result<String, String> {
        let t_all = Instant::now();
        // #film-camera  Order-independent: whatever camera the film is on, apply it NOW.
        self.apply_film_camera();
        if let Some((w, h)) = size {
            if (w, h) != self.size {
                self.renderer.resize(&self.device, (w, h));
                self.size = (w, h);
            }
        }
        let t_rebuild0 = Instant::now();
        self.ensure_built();
        let t_rebuild = t_rebuild0.elapsed();
        self.refresh_screen_fx();
        self.refresh_zone_overlay();
        let out = self.resolve_shot_path(path);
        if let Some(parent) = std::path::Path::new(&out).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // HMS_TIME picks the animation time of the capture (default 0): wave-slope cycling,
        // scrolling textures, flashing lights.
        //
        // #blade-anim  With a FILM open the default is the playback position instead of 0, so a
        // capture shows the animated materials where that tick puts them — a scrolling sword
        // blade, a flashing light — and two captures of the same tick are identical while
        // captures of different ticks are not. `HMS_TIME` still overrides, for sweeping one
        // frame through an animation.
        let anim_t = std::env::var("HMS_TIME")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .or_else(|| self.film_play.as_ref().map(|pb| pb.anim_seconds()))
            .unwrap_or(0.0);
        // Forge lights illuminate: pack the nearest to the camera into SimpleLights at the
        // shot's animation time so flashing lights sample their cycle.
        if let Some(scene) = self.scene.as_ref() {
            if scene.has_forge_lights() {
                let sl = scene.simple_lights_with_forge(self.camera.pos, anim_t);
                self.renderer.set_simple_lights(&self.queue, (sl.len() / 20) as u32, &sl);
            }
        }
        // Simulate the effect PARTICLE systems for this capture. The scripted host never did,
        // so no `shot` has ever shown a grav-lift plume, a waterfall or (now) a player's armour
        // effect — the one thing that cannot be checked any other way than by looking. Same call,
        // same clock (`anim_t`) and same camera basis the window uses, so a capture shows what
        // the window shows. `HMS_NO_LIFT_PARTICLES=1` still turns it off inside the call.
        {
            let fwd = self.camera.forward();
            let right = self.camera.right();
            let up = right.cross(fwd).normalize();
            let pos = self.camera.pos;
            let pm = {
                let mr = self.renderer.mesh_renderer();
                self.scene
                    .as_mut()
                    .map(|sc| sc.build_particle_meshes(pos, fwd, right, up, anim_t, mr, &self.device, &self.queue))
            };
            if let Some(pm) = pm { self.renderer.set_particle_meshes(pm); }
            // #projectile-sim  And the projectiles' light volumes.
            let bm = self.scene.as_mut().map(|sc| sc.build_film_beam_meshes(anim_t, self.renderer.mesh_renderer(), &self.device, &self.queue));
            if let Some(bm) = bm { self.renderer.set_beam_meshes(bm); }
        }
        let t_render0 = Instant::now();
        self.renderer.render(&self.device, &self.queue, &self.camera, anim_t);
        // HMS_EXPODIAG (print-only): the meter's mean log2 luminance and the raw engine gain
        // (the engine's gain is clamp(key / metered, 2^min, 2^max)).
        if std::env::var("HMS_EXPODIAG").is_ok() {
            if let Some(ml) = self.renderer.read_mean_log(&self.device, &self.queue) {
                let metered = 2f32.powf(ml);
                eprintln!("HMS_EXPODIAG mean_log2={:.3} metered_lum={:.4} raw_gain(0.1/m)={:.3}", ml, metered, 0.1 / metered.max(1e-4));
            }
        }
        let (rgba, rw, rh) = self.renderer.capture_rgba(&self.device, &self.queue);
        let t_render = t_render0.elapsed();
        let t_png0 = Instant::now();
        crate::headless::write_png(&out, &rgba, rw, rh).map_err(|e| format!("write {out}: {e}"))?;
        let t_png = t_png0.elapsed();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        Ok(format!(
            "wrote {out} ({rw}x{rh}) | cam {} ({:.2},{:.2},{:.2}) yaw {:.1} pitch {:.1} fov {:.1} \
             | SHOT rebuild={:.1}ms render+capture={:.1}ms png={:.1}ms total={:.1}ms",
            // #film-camera  Say WHICH camera this frame was shot from, and from where. A capture
            // that silently used the editor camera while the script had asked for first person is
            // exactly the failure this line makes impossible to miss.
            self.film_play
                .as_ref()
                .map(|p| match p.camera {
                    crate::film_play::FilmCamera::Free => "free".to_string(),
                    crate::film_play::FilmCamera::FirstPerson(s) => format!("first:slot{s}"),
                    crate::film_play::FilmCamera::Follow(s) => format!("follow:slot{s}"),
                    crate::film_play::FilmCamera::FirstPersonOwner(o) => format!("first:owner{o}"),
                    crate::film_play::FilmCamera::FollowOwner(o) => format!("follow:owner{o}"),
                })
                .map(|c| {
                    if matches!(self.film_play.as_ref().map(|p| p.camera), Some(crate::film_play::FilmCamera::Free))
                        || self.film_camera_resolved
                    {
                        c
                    } else {
                        format!("{c}!UNRESOLVED-stale-view")
                    }
                })
                .unwrap_or_else(|| "editor".to_string()),
            self.camera.pos.x,
            self.camera.pos.y,
            self.camera.pos.z,
            self.camera.yaw.to_degrees(),
            self.camera.pitch.to_degrees(),
            self.camera.fov_y.to_degrees(),
            ms(t_rebuild), ms(t_render), ms(t_png), ms(t_all.elapsed())
        ))
    }

    /// Build the holographic zone volume + outline for the SELECTED object(s) with a boundary
    /// shape (mirrors the interactive overlay), so `select <datum>` + `screenshot` shows the zone.
    fn refresh_zone_overlay(&mut self) {
        let mut segs: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        let mut ztris: Vec<([f32; 3], [f32; 4])> = Vec::new();
        for o in &self.objects {
            if !self.selected.contains(&o.datum) {
                continue;
            }
            let Some(m) = self.meta.get(&o.datum) else { continue };
            let (bsegs, btris) = crate::boundary_geometry(
                m.boundary_shape, m.boundary, m.cached_type,
                Vec3::from(o.pos), Vec3::from(o.fwd), Vec3::from(o.up),
            );
            segs.extend(bsegs);
            ztris.extend(btris);
        }
        // The hidden-blocker physics hulls, only when switched on (HMS_PHYSICS_OUTLINES /
        // `outlines on`): mirrors the GUI's View > "Show physics outlines".
        if self.show_physics_outlines {
            if let Some(scene) = self.objscene() {
                let (btris, blines) = scene.blocker_overlay();
                ztris.extend(btris.iter().copied());
                segs.extend(blines.iter().copied());
            }
        }
        // The kill floor / acceleration / slip planes + the trigger-volume boxes,
        // the same lanes and colours as the GUI's View switches.
        let mut xray: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        if self.show_soft_ceilings {
            if let Some(scene) = self.scene.as_ref() {
                let (t, l) = crate::soft_ceilings::overlay_geometry(&scene.soft_ceilings());
                ztris.extend(t);
                xray.extend(l);
            }
        }
        if let Some(scene) = self.scene.as_ref() {
            if self.show_hard_floor {
                if let Some(wb) = crate::hard_floor::world_box(scene.structure_bsp_flags(), &scene.structure_bsp_mopp_bounds()) {
                    let (t, l) = crate::hard_floor::world_geometry(&wb, 16);
                    ztris.extend(t);
                    xray.extend(l);
                }
            }
            if self.show_playable_bounds {
                let (t, l) = crate::hard_floor::overlay_geometry(scene.structure_bsp_flags(), 8);
                ztris.extend(t);
                xray.extend(l);
            }
        }
        // #quant-error  the save-precision ghost of the SELECTED object(s), the same violet
        // wireframe in the same dedicated depth-ALWAYS lane the interactive viewport uses
        let mut ghost: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        self.push_quant_ghost(&mut ghost);
        // #cannon-preview  the selected cannon's arc, in the same lane the GUI draws it in
        let mut retry = false;
        match (self.cannon_preview.on, self.scene.as_ref()) {
            (true, Some(sc)) => { self.cannon_preview.sync(sc, &self.objects, &self.selected, &mut retry); }
            _ => { self.cannon_preview.clear(); }
        }
        ghost.extend(self.cannon_preview.segments());
        self.renderer.set_ghost_lines(&self.device, &ghost);
        self.renderer.set_xray_lines(&self.device, &xray);
        if self.show_triggers {
            if let Some(scene) = self.scene.as_ref() {
                segs.extend(crate::trigger_volume_segments(&scene.trigger_volumes()));
            }
        }
        self.renderer.set_overlay_lines(&self.device, &segs);
        self.renderer.set_zone_tris(&self.device, if ztris.is_empty() { None } else { Some(&ztris) });
        // #wire-visible  The SELECTION wireframe (the GUI's apply_selection_highlight), so a
        // scripted `select <datum>` + `screenshot` carries the same highlight the editor draws --
        // and `wirexray on|off` can be checked from a script.
        let (wire_lines, wire_box) = {
            let sel = self.selected.clone();
            let mut lines: Vec<[f32; 3]> = Vec::new();
            let mut fallback: Option<([f32; 3], [f32; 3])> = None;
            if let Some(sc) = self.objscene() {
                for d in &sel {
                    match sc.selection_wireframe(*d) {
                        Some(w) => lines.extend(w),
                        None if fallback.is_none() => fallback = sc.aabb_of(*d),
                        None => {}
                    }
                }
            }
            (lines, fallback)
        };
        self.renderer.set_highlight_xray(self.wire_xray);
        if wire_lines.is_empty() {
            self.renderer.set_highlight(&self.device, wire_box);
        } else {
            self.renderer.set_highlight_lines(&self.device, Some(&wire_lines));
        }
    }

    fn resolve_shot_path(&self, path: &str) -> String {
        let p = std::path::Path::new(path);
        let is_dir = path.ends_with('/') || path.ends_with('\\') || p.is_dir();
        if is_dir {
            let stem = if self.cur_stem.is_empty() { "shot" } else { &self.cur_stem };
            p.join(format!("{stem}.png")).to_string_lossy().into_owned()
        } else {
            path.to_string()
        }
    }

    // -------- camera helpers --------

    fn set_camera_look(&mut self, pos: Vec3, fwd: Vec3) {
        self.camera.pos = pos;
        let f = if fwd.length_squared() > 1e-6 { fwd.normalize() } else { Vec3::X };
        self.camera.yaw = f.y.atan2(f.x);
        self.camera.pitch = f.z.clamp(-1.0, 1.0).asin();
    }

    fn frame_points(&mut self, center: Vec3, radius: f32) {
        let radius = radius.max(5.0);
        let dir = Vec3::new(0.6, 0.6, -0.5).normalize();
        let dist = radius / (self.camera.fov_y * 0.5).tan() * 1.15 + radius * 0.2;
        self.camera.pos = center - dir * dist;
        let to = (center - self.camera.pos).normalize();
        self.camera.yaw = to.y.atan2(to.x);
        self.camera.pitch = to.z.clamp(-1.0, 1.0).asin();
        self.camera.far = (radius * 6.0).max(5000.0);
    }

    fn frame_forge(&mut self) {
        let forge: Vec<Vec3> = self.objects.iter().skip(self.base_objects.len()).map(|o| Vec3::from(o.pos)).collect();
        if forge.is_empty() {
            self.frame_scene();
            return;
        }
        let n = forge.len() as f32;
        let c = forge.iter().fold(Vec3::ZERO, |a, p| a + *p) / n;
        let spread = (forge.iter().map(|p| (*p - c).length_squared()).sum::<f32>() / n).sqrt();
        self.frame_points(c, spread * 2.5);
    }

    fn frame_scene(&mut self) {
        // Prefer object centroid; fall back to scene bounds.
        let centroid = if !self.objects.is_empty() {
            let n = self.objects.len() as f32;
            let c = self.objects.iter().fold(Vec3::ZERO, |a, o| a + Vec3::from(o.pos)) / n;
            let r = (self.objects.iter().map(|o| (Vec3::from(o.pos) - c).length_squared()).sum::<f32>() / n).sqrt();
            Some((c, r.max(5.0) * 2.5))
        } else {
            self.objscene().and_then(|s| s.scene_bounds()).map(|(mn, mx)| {
                let mn = Vec3::from(mn);
                let mx = Vec3::from(mx);
                ((mn + mx) * 0.5, (mx - mn).length() * 0.5)
            })
        };
        if let Some((c, r)) = centroid {
            self.frame_points(c, r);
        }
    }

    // -------- command execution --------

    fn movable_index(&self, datum: u32) -> Option<usize> {
        self.objects.iter().position(|o| o.datum == datum)
    }

    /// #settle Physics-only settle, the same driver as `App::settle_object`
    /// (`physics_settle::settle_in_scene`): the object's phmo rigid-body shapes at its editor
    /// scale, dropped under gravity onto the level geometry + the other objects' physics hulls,
    /// stepped until rest. The scene must be built (`ensure_built`) so the scales are in place.
    fn settle_one(&mut self, datum: u32) -> Result<String, String> {
        let outcome = {
            let scene = self.objscene().ok_or_else(|| "no map loaded".to_string())?;
            crate::physics_settle::settle_in_scene(scene, &self.objects, datum, &[], &crate::physics_settle::SettleParams::default())?
        };
        if outcome.moved() {
            if let Some(i) = self.movable_index(datum) {
                self.objects[i].pos = outcome.pos.into();
                self.objects[i].fwd = outcome.fwd.into();
                self.objects[i].up = outcome.up.into();
            }
            self.dirty = true;
        }
        Ok(outcome.summary())
    }

    /// #film  `film <path> | info | card | machine | extract variant|gametype|both [dir] |
    /// records | scene`. `scene` loads the film's base map and applies its embedded Forge variant
    /// through the ordinary variant path, so a headless screenshot of a film's scene is possible.
    fn film_verb(&mut self, sub: &str, args: &[String]) -> Result<String, String> {
        use crate::{film, film_export, film_ui};
        let label = |f: &film::Film| -> Option<String> {
            let i = crate::mapcat::find_map_for_variant(&self.cands, f.content.map_id, None)?;
            Some(self.cands[i].label())
        };
        let rows = |f: &film::Film, lbl: Option<String>| -> String {
            film_ui::metadata_rows(f, false, lbl.as_deref())
                .into_iter()
                .map(|(k, v)| format!("{k:<18}{v}\n"))
                .collect()
        };
        // #film-data  megalo / variant / events / seats / score: shared with the GUI command server.
        if let Some(r) = crate::film_data::dispatch(self.film_path.as_deref(), self.film.as_deref(), self.film_play.as_ref().map(|p| &p.index), sub, args) {
            return r;
        }
        match sub {
            "info" | "summary" => {
                let f = self.film.as_deref().ok_or("film: no film open (film <path>)")?;
                Ok(f.summary())
            }
            "card" | "details" => {
                let f = self.film.as_deref().ok_or("film: no film open (film <path>)")?;
                let l = label(f);
                Ok(rows(f, l))
            }
            // #film-stats  `film stats [json]` -- the headless statistics, also reachable from
            // the script host. The CLI verb `--film-stats` is the primary interface and needs no
            // host (and no GPU); this exists so a running session can ask the same question.
            "stats" | "statistics" => {
                let path = self.film_path.clone().ok_or("film: no film open (film <path>)")?;
                let names = match (self.scene.as_ref(), self.scene.as_ref().and_then(|s| s.cache_handle())) {
                    (Some(sc), Some(c)) => Some(crate::film_stats::MapNames::build(sc.native_dll(), c)),
                    _ => None,
                };
                let st = match &names {
                    Some(n) => crate::film_stats::from_path_with(&path, n),
                    None => crate::film_stats::from_path(&path),
                }?;
                Ok(match args.first().map(|a| a.as_str()) {
                    Some("json") => st.json(),
                    _ => st.table(),
                })
            }
            "identity" | "bind" | "who" => {
                // Which recorded player drove which network entity. Nothing in the stream says
                // so, so this is a measured correlation between each player's aim yaw and each
                // biped's heading -- the report prints the cost and the runner-up margin so the
                // match can be judged rather than trusted. One binding per LIFE: a player gets a
                // new entity on every respawn.
                let path = self.film_path.clone().ok_or("film: no film open (film <path>)")?;
                let f = film::open(&path).map_err(|e| e.to_string())?;
                let idx = crate::film_decode::build_index_film(&f);
                let mut out = String::new();
                let aimed: Vec<usize> =
                    (0..idx.player_aims.len()).filter(|&i| idx.player_aims[i].len() >= 30).collect();
                let bipeds = idx.tracks.iter().filter(|t| t.entity_type == 11).count();
                out.push_str(&format!(
                    "{} player slots with aim samples, {} biped entities, {} bound ({} confident)\n",
                    aimed.len(),
                    bipeds,
                    idx.bindings.len(),
                    idx.bindings.iter().filter(|b| b.is_confident()).count()
                ));
                for s in 0..16u8 {
                    let named = f.header.players.get(s as usize).filter(|p| p.present());
                    let lives = idx.lives_of_slot(s);
                    let any = idx.bindings.iter().filter(|b| b.slot == s).count();
                    if named.is_none() && any == 0 {
                        continue;
                    }
                    let name = named.map(|p| p.gamertag.clone()).unwrap_or_else(|| "(empty slot)".into());
                    let range = idx
                        .slot_tick_range(s)
                        .map(|(a, b)| {
                            format!(
                                " present {}..{}",
                                crate::film_play::fmt_time(a as f32 / 60.0),
                                crate::film_play::fmt_time(b as f32 / 60.0)
                            )
                        })
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "slot {s:<2} {name:<20} {} confident live(s) of {any} bound, {} aim samples{range}\n",
                        lives.len(),
                        idx.player_aims.get(s as usize).map(|v| v.len()).unwrap_or(0)
                    ));
                    for b in lives.iter().take(6) {
                        // #catchall  SAY HOW the name was arrived at. A body named by elimination
                        // rather than by its own owner stamp carries no correlation score, and
                        // printing 0.000 next to it would read like a perfect match.
                        out.push_str(&match b.origin {
                            crate::film_decode::BindOrigin::SpeciesElimination => format!(
                                "     0x{:08X}  {} .. {}  named by ELIMINATION ({})\n",
                                b.net_id,
                                crate::film_play::fmt_time(b.first_tick as f32 / 60.0),
                                crate::film_play::fmt_time(b.last_tick as f32 / 60.0),
                                b.origin.label()
                            ),
                            _ => format!(
                                "     0x{:08X}  {} .. {}  cost {:.3} margin {:.3} n={} ({}) via {}\n",
                                b.net_id,
                                crate::film_play::fmt_time(b.first_tick as f32 / 60.0),
                                crate::film_play::fmt_time(b.last_tick as f32 / 60.0),
                                b.cost,
                                b.margin,
                                b.samples,
                                if b.from_facing { "facing" } else { "travel" },
                                b.origin.label()
                            ),
                        });
                    }
                    if lives.len() > 6 {
                        out.push_str(&format!("     ... and {} more\n", lives.len() - 6));
                    }
                }
                self.film_play = Some(crate::film_play::Playback::new(idx));
                self.film_spawn_weapon = None; // #gamevariant re-arm for the new index
                Ok(out)
            }
            // #reach-hud #crosshair  `film hud [on|off]` / `film crosshair [on|off]` /
            // `film respawnwidget [on|off]`. The GUI has these as checkboxes under "In first
            // person"; the verbs exist so a scripted capture can set them, because a toggle that
            // can only be reached by clicking cannot be put in a screenshot.
            "hud" | "crosshair" | "reticule" | "respawnwidget" => {
                let which = sub.to_string();
                let pb = self.film_play.as_mut().ok_or("film: the recording is not decoded yet (film index)")?;
                let flag = match which.as_str() {
                    "hud" => &mut pb.hud,
                    "respawnwidget" => &mut pb.respawn_widget,
                    _ => &mut pb.crosshair,
                };
                match args.first().map(|s| s.to_lowercase()) {
                    Some(w) if w == "on" || w == "1" || w == "yes" => *flag = true,
                    Some(w) if w == "off" || w == "0" || w == "no" => *flag = false,
                    Some(w) => return Err(format!("film {which} {w}: expected on | off")),
                    None => *flag = !*flag,
                }
                let on = *flag;
                match which.as_str() {
                    "hud" => crate::reach_hud::save_hud_setting(on),
                    "respawnwidget" => crate::reach_hud::save_respawn_setting(on),
                    _ => crate::reach_hud::save_crosshair_setting(on),
                }
                let mut out = format!("film {which}: {}\n", if on { "drawn" } else { "hidden" });
                for l in crate::film_ui::hud_field_report() {
                    out.push_str(&format!("  {l}\n"));
                }
                Ok(out)
            }
            // #film-action  `film assassination [next|prev]` -- the script half of the two
            // transport buttons, so a capture can frame a kill without a human at the keyboard.
            "assassination" | "assassinations" => {
                let pb = self.film_play.as_mut().ok_or("film: the recording is not decoded yet (film index)")?;
                let total = pb.assassinations().len();
                let dir = match args.first().map(|s| s.as_str()) {
                    None | Some("next") => 1,
                    Some("prev") | Some("previous") | Some("back") => -1,
                    Some(w) if w == "list" => {
                        let mut out = format!("{total} assassination(s)\n");
                        let mut all = pb.assassinations().to_vec();
                        all.sort_by_key(|a| a.tick);
                        for a in &all {
                            out.push_str(&format!(
                                "  {} {} {}\n",
                                crate::film_play::fmt_time(a.tick as f32 / 60.0),
                                a.incident,
                                if a.stated { "(stated)" } else { "(derived)" }
                            ));
                        }
                        return Ok(out);
                    }
                    Some(w) => return Err(format!("film assassination {w}: expected next | prev | list")),
                };
                match pb.jump_to_assassination(dir) {
                    Some(a) => Ok(format!(
                        "assassination at {} ({}, {}); playhead parked {} ticks early, camera on the victim\n",
                        crate::film_play::fmt_time(a.tick as f32 / 60.0),
                        a.incident,
                        if a.stated { "stated" } else { "derived" },
                        crate::film_play::Playback::ASSASSINATION_LEAD_TICKS
                    )),
                    None => Ok(format!(
                        "no assassination {} this point ({total} in the clip)\n",
                        if dir >= 0 { "after" } else { "before" }
                    )),
                }
            }
            "tags" | "nametags" => {
                let pb = self.film_play.as_mut().ok_or("film: the recording is not decoded yet (film index)")?;
                // #nametag-mode  always | lookedat | hidden. `on`/`off` are kept because that is
                // what this verb took while the setting was a bool; no argument CYCLES.
                pb.name_tags = match args.first().map(|s| s.as_str()) {
                    Some(w) => crate::reach_hud::NameTagMode::parse(w)
                        .ok_or_else(|| format!("film nametags {w}: expected always | lookedat | hidden"))?,
                    None => match pb.name_tags {
                        crate::reach_hud::NameTagMode::Always => crate::reach_hud::NameTagMode::LookedAt,
                        crate::reach_hud::NameTagMode::LookedAt => crate::reach_hud::NameTagMode::Hidden,
                        crate::reach_hud::NameTagMode::Hidden => crate::reach_hud::NameTagMode::Always,
                    },
                };
                crate::reach_hud::save_name_tag_mode(pb.name_tags);
                let snapshot = pb.clone();
                let players: Vec<crate::film::FilmPlayer> =
                    self.film.as_deref().map(|f| f.header.players.clone()).unwrap_or_default();
                let tags = crate::film_ui::name_tags(&snapshot, &players, 16);
                let mut out = format!(
                    "name tags {}; {} tag(s) at this moment\n",
                    snapshot.name_tags.keyword(),
                    tags.len()
                );
                // #nametag-mode  Be clear about what this list IS. In `lookedat` the drawing code
                // culls it to the ONE body inside the engine's own 3-degree cone, and that cull
                // happens at paint time against the live camera — so the rows below are the
                // CANDIDATES, not what ends up on screen.
                if snapshot.name_tags == crate::reach_hud::NameTagMode::LookedAt {
                    out.push_str(
                        "  (these are the candidates: in 'lookedat' the draw keeps only the one body inside \
                         the engine's 3-degree, 40 wu cone, nearest first)\n",
                    );
                }
                for t in tags.iter().take(16) {
                    // #catchall  A tag whose name was worked out by ELIMINATION rather than read
                    // off the body's own owner stamp says so, so a name that is an inference is
                    // never presented as a stated fact.
                    let how = match snapshot.index.origin_of_entity(t.net_id) {
                        Some(crate::film_decode::BindOrigin::SpeciesElimination) => {
                            "  [named by elimination: the only header slot left, and its species]"
                        }
                        _ if !t.named => "  [unnamed: no confident player match]",
                        _ => "",
                    };
                    out.push_str(&format!(
                        "  {:<22} at ({:7.2}, {:7.2}, {:7.2}){how}\n",
                        t.text,
                        t.pos[0],
                        t.pos[1],
                        t.pos[2],
                    ));
                }
                Ok(out)
            }
            "camera" | "cam" => {
                // #film  `film camera free | first <slot> | follow <slot>`. Headless, so this
                // reports the solved pose rather than moving a window.
                if self.film_play.is_none() {
                    let path = self.film_path.clone().ok_or("film: no film open (film <path>)")?;
                    let f = film::open(&path).map_err(|e| e.to_string())?;
                    let idx = crate::film_decode::build_index_film(&f);
                    self.film_play = Some(crate::film_play::Playback::new(idx));
                self.film_spawn_weapon = None; // #gamevariant re-arm for the new index
                }
                let pb = self.film_play.as_mut().unwrap();
                let what = args.first().map(|s| s.as_str()).unwrap_or("free");
                let slot = |which: &str| -> Result<u8, String> {
                    args.get(1)
                        .ok_or_else(|| format!("film camera {which} <player slot>"))?
                        .parse::<u8>()
                        .map_err(|_| format!("film camera {which} <player slot 0..15>"))
                };
                // #film-camera  `film camera mode` REPORTS which engine camera mode the current
                // subject is in and why, and `film camera context on|off` sets (and persists)
                // whether Theater follows the engine into them at all. Both are reports/settings
                // rather than a camera selection, so they return before the `pb.camera` match.
                match what {
                    "mode" | "engine" | "ctx" => {
                        let mut out = String::new();
                        out.push_str(&format!(
                            "context cameras: {}\n",
                            if pb.context_cameras { "ON" } else { "off" }
                        ));
                        match pb.camera_context() {
                            Some(c) => {
                                out.push_str(&format!("  {}\n", c.label()));
                                out.push_str(&format!(
                                    "  entered at tick {} ({:.0}% through its {} s transition)\n",
                                    c.entered,
                                    c.blend * 100.0,
                                    crate::film_camera::ENTER_VEHICLE_TRANSITION_SECONDS
                                ));
                                let tl = pb.camera_timeline();
                                out.push_str(&format!(
                                    "  this body has {} override segment(s); the film has {} over {} bodies\n",
                                    tl.segments(c.subject).len(),
                                    tl.total_segments(),
                                    tl.per_body.len()
                                ));
                                for s in tl.segments(c.subject) {
                                    out.push_str(&format!(
                                        "    {:>7} .. {:<7}  mode {} ({:<14}) {:?}\n",
                                        s.from_tick,
                                        s.to_tick,
                                        s.mode.index(),
                                        s.mode.name(),
                                        s.detail
                                    ));
                                }
                            }
                            None => out.push_str("  no subject (free camera, or that player has no body here)\n"),
                        }
                        // Every context camera in the whole film, keyed by the tick it starts, so
                        // a scripted check can seek straight to one instead of hunting. Printed
                        // whether or not the current subject has a body, because "which body is
                        // the camera on right now" and "where are this film's context cameras"
                        // are different questions and the second is the one a capture needs.
                        {
                            let tl = pb.camera_timeline();
                            let mut rows: Vec<(u32, u32, String)> = Vec::new();
                            for (net, segs) in &tl.per_body {
                                for s in segs {
                                    if let crate::film_camera::ContextDetail::SyncAction { attacker, partner, .. } =
                                        &s.detail
                                    {
                                        rows.push((
                                            s.from_tick,
                                            s.to_tick,
                                            format!(
                                                "sync action: 0x{net:08X} is the {} of 0x{:08X}",
                                                if *attacker { "ATTACKER" } else { "victim" },
                                                partner.unwrap_or(0)
                                            ),
                                        ));
                                    }
                                }
                            }
                            rows.sort();
                            out.push_str(&format!(
                                "  this film's SYNC ACTION cameras ({} half-segments = {} kill(s)):\n",
                                rows.len(),
                                rows.len() / 2
                            ));
                            for (a, b, w) in rows.iter().take(40) {
                                out.push_str(&format!(
                                    "    tick {a:>7}..{b:<7} ({:>7.2} s)  {w}\n",
                                    *a as f32 / crate::film_camera::TICK_HZ
                                ));
                            }
                        }
                        out.push_str("  the engine's thirteen modes: ");
                        out.push_str(
                            &crate::film_camera::ENGINE_CAMERA_MODE_NAMES
                                .iter()
                                .enumerate()
                                .map(|(i, n)| format!("{i}={n}"))
                                .collect::<Vec<_>>()
                                .join(", "),
                        );
                        out.push('\n');
                        return Ok(out);
                    }
                    // #film-camera  Which of the OPEN MAP's weapons take the view out of first
                    // person — `weap +0x204` bit 15, `uses 3rd person camera`. A report rather
                    // than a setting: the flag is the engine's, and the only honest way to say
                    // whether a detached turret changes the camera is to read it.
                    "weapons" | "weapon" => {
                        let Some(scene) = self.scene.as_ref() else {
                            return Err("film camera weapons: no map open (openmap <path>)".into());
                        };
                        let Some(cache) = scene.cache_handle() else {
                            return Err("film camera weapons: the open map has no cache".into());
                        };
                        let native = scene.native_dll();
                        let mut rows: Vec<(bool, bool, String)> = Vec::new();
                        for t in native.tags_by_group(cache, "weap") {
                            let Some(name) = native.tag_name(cache, t) else { continue };
                            let third = crate::film_camera::decode_weapon_third_person(native, cache, t);
                            let support = native.tag_meta(cache, t, 0, 0x208).map(|b| {
                                u32::from_le_bytes([b[0x204], b[0x205], b[0x206], b[0x207]])
                                    & crate::film_camera::WEAPON_FLAG_SUPPORT
                                    != 0
                            });
                            rows.push((
                                third == Some(true),
                                support == Some(true),
                                name.rsplit('\\').next().unwrap_or(&name).to_string(),
                            ));
                        }
                        rows.sort();
                        let n3 = rows.iter().filter(|r| r.0).count();
                        let mut out = format!(
                            "{} weapon tag(s) in this map; {n3} set `uses 3rd person camera` \
                             (weap +0x204 bit 15)\n",
                            rows.len()
                        );
                        for (third, support, name) in rows.iter().rev() {
                            if !*third && !*support {
                                continue;
                            }
                            out.push_str(&format!(
                                "  {:<34} {}{}\n",
                                name,
                                if *third { "USES 3RD PERSON CAMERA" } else { "" },
                                if *support { "  [support weapon, bit 8]" } else { "" }
                            ));
                        }
                        return Ok(out);
                    }
                    "context" => {
                        let on = match args.get(1).map(|s| s.as_str()) {
                            Some("on" | "1" | "true" | "yes") => true,
                            Some("off" | "0" | "false" | "no") => false,
                            None => !pb.context_cameras,
                            Some(o) => return Err(format!("film camera context on|off (got {o})")),
                        };
                        pb.context_cameras = on;
                        crate::film_camera::save_context_setting(on);
                        return Ok(format!(
                            "context cameras {} (persisted)\n",
                            if on { "ON: Theater follows the engine into its assassination, vehicle and death cameras" } else { "off: the plain camera stays where you put it" }
                        ));
                    }
                    _ => {}
                }
                pb.camera = match what {
                    "free" | "off" | "none" => crate::film_play::FilmCamera::Free,
                    "first" | "fp" | "firstperson" | "eyes" => {
                        crate::film_play::FilmCamera::FirstPerson(slot("first")?)
                    }
                    "follow" | "chase" | "third" => crate::film_play::FilmCamera::Follow(slot("follow")?),
                    // #film-camera  The OWNER-keyed modes, which are what the window's own camera
                    // list actually uses -- it offers one camera per owner GROUP so that a body
                    // the recording never named is still watchable. This host could only select
                    // the SLOT-keyed modes, so the path the user is really looking through could
                    // not be reproduced from a script at all.
                    "firstbody" | "fpbody" | "firstowner" => {
                        crate::film_play::FilmCamera::FirstPersonOwner(slot("firstbody")?)
                    }
                    "followbody" | "chasebody" | "followowner" => {
                        crate::film_play::FilmCamera::FollowOwner(slot("followbody")?)
                    }
                    other => {
                        return Err(format!(
                            "film camera {other}: expected free | first <slot> | follow <slot> | firstbody <owner> | followbody <owner> | mode | context on|off"
                        ))
                    }
                };
                let status = crate::film_play::camera_status(pb);
                Ok(match crate::film_play::solve_camera(pb) {
                    Some(p) => format!(
                        "{status}\n  {} at ({:.2}, {:.2}, {:.2}) yaw {:.1} pitch {:.1}\n",
                        p.label, p.pos[0], p.pos[1], p.pos[2], p.yaw.to_degrees(), p.pitch.to_degrees()
                    ),
                    None => format!("{status}\n"),
                })
            }
            "armour" | "armor" | "appearance" => {
                // The recorded armour customisation per player. Ten slot bytes, each an index
                // into one block of `globals\\player_model_customization`.
                let f = self.film.as_deref().ok_or("film: no film open (film <path>)")?;
                let mut out = String::new();
                for p in f.present_players() {
                    match p.appearance() {
                        Some(a) => {
                            out.push_str(&format!(
                                "slot {:<2} {:<20} {}{}\n",
                                p.index,
                                p.gamertag,
                                a.summary(),
                                if a.in_range() { "" } else { "  [OUT OF RANGE: an index exceeds its option list]" }
                            ));
                            for (i, v) in a.slots.iter().enumerate() {
                                out.push_str(&format!(
                                    "    {:<16} option {:>3} of {:>3}   -> {}\n",
                                    crate::film::APPEARANCE_SLOT_NAMES[i],
                                    v,
                                    crate::film::APPEARANCE_COUNTS[i],
                                    crate::film::APPEARANCE_SLOT_REGIONS[i]
                                ));
                            }
                        }
                        None => out.push_str(&format!("slot {:<2} {:<20} (record too short)\n", p.index, p.gamertag)),
                    }
                }
                if out.is_empty() {
                    out.push_str("no player slots are occupied in this film\n");
                }
                Ok(out)
            }
            "scan" | "diagnose" => {
                // The decoder's own health report on this film: how many records decoded, where
                // the rest stopped, and what the parameter solver chose. This is the verb to
                // reach for when a film yields no entities.
                let path = self.film_path.clone().ok_or("film: no film open (film <path>)")?;
                let f = film::open(&path).map_err(|e| e.to_string())?;
                let st = crate::film_decode::scan(&f.bytes, &f.data);
                Ok(format!("{}\n", st.report()))
            }
            "roster" | "players" | "names" => {
                let f = self.film.as_deref().ok_or("film: no film open (film <path>)")?;
                let mut out = String::new();
                for p in f.present_players() {
                    out.push_str(&format!(
                        "slot {:<2} {:<20} [{:<4}] {}\n",
                        p.index,
                        if p.gamertag.is_empty() { "(no gamertag)" } else { p.gamertag.as_str() },
                        p.service_tag,
                        film_ui::team_name(p.team)
                    ));
                }
                if out.is_empty() {
                    out.push_str("no player slots are occupied in this film\n");
                }
                // Population over time, so a caller picking a capture timestamp never lands in
                // the pre-spawn dead zone at the start of a round.
                if let Some(pb) = self.film_play.as_ref() {
                    match pb.index.first_player_tick() {
                        Some(t) => out.push_str(&format!(
                            "first player biped appears at tick {t} ({})\n",
                            crate::film_play::fmt_time(t as f32 / 60.0)
                        )),
                        None => out.push_str("the entity stream records no player bipeds\n"),
                    }
                    for (tick, n) in pb.index.player_population_changes(12) {
                        out.push_str(&format!(
                            "  {} -> {n} alive\n",
                            crate::film_play::fmt_time(tick as f32 / 60.0)
                        ));
                    }
                } else {
                    out.push_str("(run `film index` first for spawn times)\n");
                }
                // The CAMERA list the GUI actually offers, which is built per OWNER GROUP from
                // the entity stream rather than from the header — so a row here that shows an
                // entity id instead of a gamertag is a body whose owner HMS could not tie to a
                // player slot, and that is exactly what the panel shows the user.
                if let Some(pb) = self.film_play.as_ref() {
                    let rows = film_ui::roster_lines(f, pb, &self.film_driver.biped_of);
                    out.push_str(&format!("\ncamera list ({} row(s)):\n", rows.len()));
                    for r in &rows {
                        out.push_str(&format!(
                            "  {:<22} owner {:<6} slot {:<4} named={} lives={} first_tick={}\n",
                            r.label,
                            r.owner.map(|o| o.to_string()).unwrap_or_else(|| "-".into()),
                            r.slot.map(|s| s.to_string()).unwrap_or_else(|| "-".into()),
                            r.named,
                            r.lives,
                            r.first_tick.map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
                        ));
                    }
                    // #species  Cross-check the two independent statements about a player's
                    // SPECIES: the header record's own `player_model_choice` (+0x4C, 0 Spartan /
                    // 1 Elite) against the biped the entity stream names for that player's
                    // bodies. They must agree; where they do not, the body is being dressed from
                    // one species' loadout while drawn as the other, which is how armour goes
                    // missing. Needs an open map, because resolving a body's biped does.
                    let mut rows: Vec<String> = Vec::new();
                    for p in f.present_players() {
                        let Some(a) = p.appearance() else { continue };
                        let said = if a.model_choice == 1 { "elite" } else { "spartan" };
                        let owner = pb.index.owner_of_slot(p.index as u8);
                        let mut seen: Vec<&str> = Vec::new();
                        // #catchall  Ask for the slot's BODIES, not its owner group: a body named
                        // by elimination sits in a catch-all group that names nobody, so going
                        // through the group would report "(no body resolved)" for a player whose
                        // body is in fact identified.
                        for t in pb.index.bodies_of_slot(p.index as u8) {
                            if let Some(b) = self.film_driver.biped_of.get(&t.net_id) {
                                // The LEAF of the biped tag path, not a two-way is_elite test:
                                // a Forge clip also spawns `monitor_editor` bodies, which an
                                // elite/not-elite split would mislabel as Spartans.
                                let k = b.rsplit('\\').next().unwrap_or(b.as_str());
                                if !seen.contains(&k) { seen.push(k); }
                            }
                        }
                        let drawn = if seen.is_empty() { "(no body resolved)".to_string() } else { seen.join("+") };
                        let agree = seen.len() == 1 && seen[0].contains(said);
                        rows.push(format!(
                            "  slot {:<2} {:<20} record says {:<7} bodies drawn as {:<16} {}\n",
                            p.index, p.gamertag, said, drawn,
                            if seen.is_empty() { "-" } else if agree { "agree" } else { "** DISAGREE **" }
                        ));
                        // A disagreement is either one player who changed species, or an owner
                        // group that has collected bodies belonging to more than one player. The
                        // two look completely different per body, so print them.
                        if !agree && !seen.is_empty() {
                            if let Some(o) = owner {
                                let mut ts: Vec<_> = pb
                                    .index
                                    .tracks
                                    .iter()
                                    .filter(|t| t.entity_type == 11 && t.owner_raw == Some(o))
                                    .collect();
                                ts.sort_by_key(|t| t.first_tick);
                                for t in ts {
                                    rows.push(format!(
                                        "      body {:#010x} ticks {:>6}..{:<6} ({:>5.1}s..{:<5.1}s)  {}\n",
                                        t.net_id, t.first_tick, t.last_tick,
                                        t.first_tick as f32 / 60.0, t.last_tick as f32 / 60.0,
                                        self.film_driver
                                            .biped_of
                                            .get(&t.net_id)
                                            .map(|b| b.rsplit('\\').next().unwrap_or(b.as_str()).to_string())
                                            .unwrap_or_else(|| "(unresolved)".into())
                                    ));
                                }
                            }
                        }
                    }
                    if !rows.is_empty() {
                        out.push_str("\nspecies cross-check (header `player_model_choice` vs the biped the stream names):\n");
                        for r in rows { out.push_str(&r); }
                    }
                    out.push_str(&film_ui::roster_coverage(pb, f.present_players().count()));
                    out.push('\n');
                }
                Ok(out)
            }
            "machine" => {
                let f = self.film.as_deref().ok_or("film: no film open")?;
                Ok(format!(
                    "recording machine: {}\n(the Windows account and computer name of the PC that recorded the film)\n",
                    f.header.session_id
                ))
            }
            "records" | "stream" => {
                let path = self.film_path.clone().ok_or("film: no film open")?;
                let f = film::open(&path).map_err(|e| e.to_string())?;
                let tick = f.data.records.iter().filter(|r| r.kind == film::RecordKind::Tick).count();
                let total: usize = f.data.records.iter().map(|r| r.len).sum();
                Ok(format!(
                    "{} records ({tick} tick), {total} payload bytes, framing {}, flmh says {} ticks\n",
                    f.data.records.len(),
                    if f.data.framing_exact { "exact" } else { "BROKEN" },
                    f.header.tick_count
                ))
            }
            "extract" => {
                let f = self.film.clone().ok_or("film: no film open (film <path>)")?;
                let dir = args
                    .get(1)
                    .map(std::path::PathBuf::from)
                    .or_else(|| self.film_path.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())))
                    .unwrap_or_else(std::env::temp_dir);
                let what = args.first().map(|s| s.as_str()).unwrap_or("both");
                let mut out = String::new();
                if matches!(what, "variant" | "mvar" | "map" | "both") {
                    let stem = film_export::safe_stem(&f.header.map_variant.metadata.title, "extracted_variant");
                    match film_export::export_map_variant(&f, &dir.join(format!("{stem}.mvar"))) {
                        Ok(r) => out.push_str(&format!("map variant: {} objects -> {}\n", r.objects, r.path.display())),
                        Err(e) => out.push_str(&format!("map variant: {e}\n")),
                    }
                }
                if matches!(what, "gametype" | "gv" | "game" | "both") {
                    let stem = film_export::safe_stem(&f.header.game_variant.title, "extracted_gametype");
                    match film_export::export_game_variant(&f, &dir.join(format!("{stem}.bin"))) {
                        Ok(r) => match (&r.bin_path, &r.why_no_bin) {
                            (Some(bin), _) => out.push_str(&format!(
                                "gametype: {} bytes -> {} (a loadable Reach game variant .bin; notes in {})\n",
                                r.bytes, bin.display(),
                                r.notes_path.file_name().unwrap_or_default().to_string_lossy()
                            )),
                            (None, why) => out.push_str(&format!(
                                "gametype: no .bin could be built ({}); raw struct ({} bytes) -> {} instead\n",
                                why.clone().unwrap_or_default(), r.bytes,
                                r.struct_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
                            )),
                        },
                        Err(e) => out.push_str(&format!("gametype: {e}\n")),
                    }
                }
                if out.is_empty() {
                    return Err(format!("film extract {what}: expected variant | gametype | both"));
                }
                Ok(out)
            }
            // #ragdoll  `film ragdoll [on|off]`: the ARTICULATED solver for dead bodies, so a
            // capture can be taken both ways and compared. It is ON by default.
            "ragdoll" => {
                if let Some(a) = args.first() {
                    let on = matches!(a.to_ascii_lowercase().as_str(), "on" | "1" | "true" | "yes");
                    self.film_driver.ragdoll = on;
                }
                let mut s = format!(
                    "ragdoll solver {} -- {} body(ies) simulated at the last sync, furthest simulated {} tick(s){}\n",
                    if self.film_driver.ragdoll { "on" } else { "off" },
                    self.film_driver.ragdolls_shown,
                    self.film_driver.ragdoll_steps,
                    match &self.film_driver.ragdoll_missing {
                        Some(w) => format!("; {w}"),
                        None => String::new(),
                    }
                );
                for (net, p, age) in &self.film_driver.ragdoll_bodies {
                    s.push_str(&format!(
                        "  entity {net:#010x} at ({:7.2}, {:7.2}, {:7.2})  dead {:.2}s\n",
                        p[0], p[1], p[2], *age as f32 / 60.0
                    ));
                }
                // #melee  The replacement lane, reported beside it: which bodies are swinging,
                // where, and on what clip.
                s.push_str(&format!(
                    "melee: {} swinging, the clip carries {} of the skeleton's {} nodes\n",
                    self.film_driver.melee_shown, self.film_driver.melee_nodes.0, self.film_driver.melee_nodes.1
                ));
                for (p, clip) in &self.film_driver.melee_bodies {
                    s.push_str(&format!("  swing at ({:7.2}, {:7.2}, {:7.2})  {clip}\n", p[0], p[1], p[2]));
                }
                Ok(s)
            }
            // ---- playback -------------------------------------------------------------
            "index" => {
                let path = self.film_path.clone().ok_or("film: no film open (film <path>)")?;
                let f = film::open(&path).map_err(|e| e.to_string())?;
                let idx = crate::film_decode::build_index_film(&f);
                let text = idx.report();
                self.film_play = Some(crate::film_play::Playback::new(idx));
                self.film_spawn_weapon = None; // #gamevariant re-arm for the new index
                Ok(format!("{text}\n"))
            }
            "seek" | "at" | "step" | "speed" | "play" | "pause" | "show" | "frame" => {
                if self.film_play.is_none() {
                    // `film seek` implies an index; build it rather than scolding the user.
                    let path = self.film_path.clone().ok_or("film: no film open (film <path>)")?;
                    let f = film::open(&path).map_err(|e| e.to_string())?;
                    let idx = crate::film_decode::build_index_film(&f);
                    self.film_play = Some(crate::film_play::Playback::new(idx));
                self.film_spawn_weapon = None; // #gamevariant re-arm for the new index
                }
                let pb = self.film_play.as_mut().unwrap();
                match sub {
                    "seek" | "at" => {
                        let v: f32 = args.first().ok_or("film seek <seconds>")?.parse().map_err(|_| "film seek <seconds>")?;
                        pb.seek_seconds(v);
                    }
                    "step" => {
                        let n: i32 = args.first().map(|s| s.parse().unwrap_or(1)).unwrap_or(1);
                        pb.step(n);
                    }
                    "speed" => {
                        let v: f32 = args.first().ok_or("film speed <x>")?.parse().map_err(|_| "film speed <x>")?;
                        pb.speed = v.clamp(0.25, 4.0);
                    }
                    "play" => pb.playing = true,
                    "pause" => pb.playing = false,
                    _ => {}
                }
                // #film-action  Draw the film's own Forge scene, once, before the first frame
                // is placed. Without this a scripted or headless capture shows bare BSP with the
                // players apparently hovering in mid-air -- which is not a playback bug but
                // looks exactly like one, and it quietly misled our own verification for a
                // while. `film <path>` stays header-only (it must work with no map loaded at
                // all), so this is the first point where a scene is both wanted and possible.
                if self.film_scene.is_none() {
                    let note = match self.film.as_deref().map(crate::film_export::variant_from_film) {
                        None => "no film open".to_string(),
                        Some(Err(why)) => format!("no Forge scene: {why}"),
                        Some(Ok(v)) => {
                            let n = v.objects.len();
                            let title = v.title.clone();
                            match self.apply_variant_value(v, "film_scene".into(), std::time::Duration::ZERO) {
                                Ok(_) => format!("Forge scene: {n} object(s) from {title:?}"),
                                Err(e) => format!("no Forge scene: the variant would not apply ({e})"),
                            }
                        }
                    };
                    self.film_scene = Some(note);
                }
                // #gamevariant  Give every body the weapon its player SPAWNED with, out of the
                // film's own game variant -- the same step the GUI takes before it places
                // anything. This host keeps its own driver AND its own `Playback`, so a step
                // installed on only one of them silently changes the answer: without it every
                // body here reported no weapon at all (the track's weapon field is only written
                // when a weapon CHANGES HANDS, and a loadout never does), which is exactly the
                // "NOBODY is holding a weapon" a scripted seek printed while the GUI showed a
                // whole team armed.
                self.film_resolve_spawn_weapon();
                // #fp-arms  The first-person arm models + their section picks, and the subject
                // player's own armour on them. Same two calls the GUI makes, and they live in
                // `film_play` so there is only one implementation of either.
                self.film_resolve_first_person();
                // Place the actors for this moment (headless: the same call the GUI makes).
                let mut placed = String::new();
                if let Some(scene) = self.scene.as_deref() {
                    if let Some(cache) = scene.cache_handle() {
                        let native = scene.native_dll();
                        let pbc = self.film_play.clone().unwrap();
                        // #film-action  Name each body's BIPED before placing it, so an Elite
                        // draws as an Elite here exactly as it does in the GUI. One
                        // implementation, in `film_play`.
                        let mp = scene.open_map_path().map(|p| p.to_path_buf());
                        self.film_driver.resolve_bipeds(&pbc, native, cache, mp.as_deref());
                        // #corpse-fall  The same landing the GUI does -- this host keeps its own
                        // driver, so anything installed on one has to be installed on both.
                        self.film_driver.ground_corpses(&pbc, &|p: [f32; 3]| {
                            let o = glam::Vec3::new(p[0], p[1], p[2] + 0.05);
                            scene
                                .raycast_solid(o, glam::Vec3::NEG_Z, crate::film_play::GROUND_REACH)
                                .map(|t| o.z - t)
                        });
                        let rep = self.film_driver.sync(&pbc, &mut self.actors, native, cache);
                        // #ragdoll  Then the ARTICULATED solver poses every dead body, after
                        // `sync` has placed them and chosen their seed clip. Same call the window
                        // makes -- this host keeps its own driver, so a step installed on one
                        // only is exactly how a scripted check of a ragdoll can pass while the
                        // window draws something else.
                        self.film_driver.simulate_ragdolls(&pbc, &mut self.actors, native, cache, scene, &self.objects);
                        // #skin  Dress every body in the armour and colours its player actually
                        // had -- the SAME call the window makes, so a scripted capture and the
                        // window show the same cast. Runs after `sync`, because it works off the
                        // actor assignments that call makes.
                        if self.film.is_some() {
                            // Split the borrows by field: the driver and the actors are mutable,
                            // the scene is mutable (the synthetic variants are installed on it),
                            // and the film is read-only.
                            let (film, driver, actors, sc) =
                                (self.film.as_deref(), &mut self.film_driver, &mut self.actors, self.scene.as_mut());
                            if let (Some(f), Some(sc)) = (film, sc) {
                                if driver.apply_recorded_appearance(&pbc, f, sc, actors) {
                                    actors.rebuild = true;
                                }
                            }
                        }
                        // #film-camera  The camera is applied in `apply_film_camera`, called
                        // from `screenshot`. It used to be applied HERE, which made it depend on
                        // verb order: `film camera firstbody N` issued after `film scene` -- the
                        // natural way to write a script -- never re-ran this, so every `shot`
                        // fired from the map-load camera and six different targets gave six
                        // frames differing by a maxdiff of 6. Applying it once, at capture time,
                        // is order-independent; there must not be a second copy here.
                        placed = format!(
                            "  actors: {} shown, {} parked, {} created  |  {}\n",
                            rep.shown,
                            rep.parked,
                            rep.created,
                            self.film_scene.as_deref().unwrap_or("scene not attempted")
                        );
                        // #projectile-sim  Theater's simulated shots, before the mesh build, exactly
                        // where the window runs them.
                        {
                            let sc = self.scene.as_deref().unwrap();
                            let world = crate::play_projectile::SceneWorld(sc as &dyn crate::objscene::ObjectScene);
                            if let Some(cache) = sc.cache_handle() {
                                self.film_driver.synth_shots(&pbc, Some(&world), &mut self.actors, sc.native_dll(), cache);
                            }
                        }
                        if self.actors.rebuild {
                            let b = crate::reach_actor::build_actor_meshes(
                                &mut self.actors, self.scene.as_mut().unwrap(),
                                self.renderer.mesh_renderer(), &self.device, &self.queue);
                            self.install_actor_build(b);
                        }
                        // #fp-frame  A scripted seek is a jump: settle the cross-fades before
                        // posing, or `dt = 0` leaves every actor frozen part-way between clips.
                        self.actors.settle_blends();
                        self.actors.update(0.0, &self.queue);
                        self.push_actor_particles();
                    }
                }
                let pb = self.film_play.as_ref().unwrap();
                let mut out = format!("{}\n{placed}", pb.status());
                out.push_str(&format!("  {}\n", self.film_driver.caption(pb)));
                // #fp-frame  Print-only: where the first-person rig lands IN THE FRAME. A posed
                // rig that sits below the frustum is invisible to every other report here.
                if pb.camera.is_first_person() {
                    let aspect = self.shot_aspect;
                    out.push_str(&self.film_driver.fp_framing(&self.actors, &self.camera, aspect, self.fp_verbose));
                }
                if sub == "show" || sub == "frame" {
                    for p in pb.frame().iter().take(24) {
                        // The BIPED the recording names for this body, so "which of these is an
                        // Elite" is answerable from the listing instead of by aiming a camera.
                        let who = match self.film_driver.biped_of.get(&p.net_id) {
                            Some(b) => format!(" {}", b.rsplit('\\').next().unwrap_or(b.as_str())),
                            None => String::new(),
                        };
                        out.push_str(&format!(
                            "  entity {:#010x} type {:>2} at ({:7.2}, {:7.2}, {:7.2}) yaw {:6.1}{}{}\n",
                            p.net_id, p.entity_type, p.pos[0], p.pos[1], p.pos[2], p.yaw,
                            if p.has_yaw { "" } else { " (no orientation yet)" }, who
                        ));
                    }
                    // #vehicle-rot #vehicle-damage  The VEHICLES the stream draws, with the two
                    // quantities a yaw-only placement could not show: TILT (how far the recorded
                    // basis is off level) and the recorded damage state (`W` = a wreck the engine
                    // had not collected yet). `film show` is how a scripted session checks what is
                    // on screen, so the vehicle rows belong beside the bodies.
                    if let Some(vt) = &self.film_driver.vehicles {
                        let tick = pb.tick.round().max(0.0) as u32;
                        let wreck = |net: u32| self.film_driver.vehicle_damage_at(net, tick).1;
                        for (v, pos, orient) in vt.to_draw_at_with(&pb.index, tick, &wreck).iter().take(24) {
                            let (st, wr) = self.film_driver.vehicle_damage_at(v.net_id, tick);
                            out.push_str(&format!(
                                "  vehicle {:#010x} {:<14} at ({:7.2}, {:7.2}, {:7.2}) yaw {:6.1} tilt {:5.1} dmg {}{}{}\n",
                                v.net_id, v.name, pos[0], pos[1], pos[2],
                                orient.map(|(f, _)| f[1].atan2(f[0]).to_degrees()).unwrap_or(f32::NAN),
                                orient.map(|(_, u)| crate::film_vehicle::tilt_degrees(u)).unwrap_or(f32::NAN),
                                crate::forge_scale::damage_state_name(st),
                                if wr { " WRECK" } else { "" },
                                match self.film_driver.wreck_persistence(v.net_id) {
                                    Some(k) => format!(" (collected {:.1}s after it blew up)", k as f32 / 60.0),
                                    None => String::new(),
                                }
                            ));
                        }
                    }
                    // #film-objects  The OBJECTIVE objects, the loose weapons and the debris the
                    // stream draws -- the lane that used to drop the flag entirely. Each row
                    // carries the recorded tilt (what a yaw-only placement would have thrown
                    // away) and, for an objective, who the recording says is carrying it and
                    // whether that is STATED or derived.
                    if let Some(ot) = &self.film_driver.objects {
                        let tick = pb.tick.round().max(0.0) as u32;
                        let bipeds: Vec<(u32, [f32; 3])> =
                            pb.bipeds(self.film_driver.cap).into_iter().map(|p| (p.net_id, p.pos)).collect();
                        for (o, pos, orient) in
                            ot.to_draw_capped(&pb.index, tick, self.film_driver.object_cap).iter().take(32)
                        {
                            // Only an OBJECTIVE is asked this. A dropped sword lying a world
                            // unit from a player is not "carried by" him, and printing that it
                            // is would be a claim the recording does not make.
                            let carry = match Some(o.draw)
                                .filter(|d| *d == crate::film_objective::Draw::Objective)
                                .and_then(|_| {
                                    crate::film_objective::carrier_at(&pb.index, o.net_id, tick, *pos, &bipeds)
                                })
                            {
                                Some((b, how)) => format!(
                                    " carried by {:#010x} ({})",
                                    b,
                                    match how {
                                        crate::film_objective::Carry::Stated => "stated",
                                        crate::film_objective::Carry::Derived => "derived",
                                    }
                                ),
                                None => String::new(),
                            };
                            out.push_str(&format!(
                                "  {:<9} {:#010x} {:<20} at ({:7.2}, {:7.2}, {:7.2}) yaw {:6.1} tilt {:5.1}{}\n",
                                o.draw.label(),
                                o.net_id,
                                o.name,
                                pos[0],
                                pos[1],
                                pos[2],
                                orient.map(|(f, _)| f[1].atan2(f[0]).to_degrees()).unwrap_or(f32::NAN),
                                orient.map(|(_, u)| crate::film_vehicle::tilt_degrees(u)).unwrap_or(f32::NAN),
                                carry
                            ));
                        }
                        // #film-objects  And the WAYPOINT records, which a headless capture
                        // cannot show: `screenshot` renders the 3D scene only, so the chud
                        // overlay exists solely in the window. Printing the data -- the anchor,
                        // the authored colour and where it projects in THIS camera -- is how a
                        // scripted session checks it without opening one.
                        let bipeds2: Vec<(u32, [f32; 3])> =
                            pb.bipeds(self.film_driver.cap).into_iter().map(|p| (p.net_id, p.pos)).collect();
                        let teams = self
                            .film
                            .as_deref()
                            .and_then(|f| f.header.gametype.as_ref())
                            .is_some_and(|g| g.misc.teams_enabled());
                        let players: Vec<crate::film::FilmPlayer> =
                            self.film.as_deref().map(|f| f.header.players.clone()).unwrap_or_default();
                        let team_of = |net: u32| -> u8 {
                            pb.index
                                .slot_of_entity(net)
                                .and_then(|s| players.iter().find(|q| q.present() && q.index == s as usize))
                                .map(|q| q.team)
                                .unwrap_or(crate::reach_chud::TEAM_NONE)
                        };
                        let viewer_team = pb
                            .camera
                            .owner(&pb.index)
                            .and_then(|o| pb.index.slot_of_owner(o))
                            .or_else(|| self.film.as_deref().and_then(|f| f.recorder_slot()))
                            .and_then(|s| players.iter().find(|q| q.present() && q.index == s as usize))
                            .map(|q| q.team)
                            .unwrap_or(crate::reach_chud::TEAM_NONE);
                        for w in crate::film_objective::waypoints(
                            ot, &pb.index, tick, &bipeds2, viewer_team, teams, &team_of,
                        ) {
                            let d = (Vec3::from(w.pos) - self.camera.pos).length();
                            // Where it would land IN THE FRAME. `screenshot` renders the 3D
                            // scene only -- the chud overlay is painted by egui in the window --
                            // so a headless capture can never show the waypoint. Printing the
                            // pixel it projects to, in the SAME camera the next `shot` uses, is
                            // what makes the record checkable against that capture instead of
                            // taken on trust.
                            let (sw, sh) = self.size;
                            let clip = self.camera.view_proj(sw as f32 / sh.max(1) as f32)
                                * glam::Vec4::new(w.pos[0], w.pos[1], w.pos[2], 1.0);
                            let screen = if clip.w > 1e-6 {
                                let n = clip.truncate() / clip.w;
                                format!(
                                    "  screen ({:.0}, {:.0}) of {sw}x{sh}",
                                    (n.x * 0.5 + 0.5) * sw as f32,
                                    (0.5 - n.y * 0.5) * sh as f32
                                )
                            } else {
                                "  behind the camera".to_string()
                            };
                            out.push_str(&format!(
                                "  waypoint  {:#010x} {:<10} anchor ({:7.2}, {:7.2}, {:7.2}) colour {} ({}) \
                                 scale {:.2} at {:.1} wu{}{}\n",
                                w.net_id,
                                w.text,
                                w.pos[0],
                                w.pos[1],
                                w.pos[2],
                                w.color_index,
                                match w.color_index {
                                    0 => "navpt normal",
                                    3 => "navpt enemy",
                                    _ => "navpt neutral",
                                },
                                crate::film_ui::nameplate_scale(d, &crate::film_ui::NamePlateRules::default()),
                                d,
                                match w.carrier {
                                    Some((b, crate::film_objective::Carry::Stated)) =>
                                        format!("  carried by {b:#010x} (STATED by the recording)"),
                                    Some((b, crate::film_objective::Carry::Derived)) =>
                                        format!("  carried by {b:#010x} (derived from the recorded positions)"),
                                    None => "  on the ground".to_string(),
                                },
                                screen
                            ));
                        }
                    }
                }
                Ok(out)
            }
            // #film-objects  `film objects [on|off]` -- draw the objective objects, the loose
            // weapons and the debris the recording carries. On by default; this is the toggle.
            "objects" | "objective" | "objectives" => {
                match args.first().map(|s| s.to_lowercase()) {
                    Some(w) if w == "on" || w == "yes" || w == "1" => self.film_driver.show_objects = true,
                    Some(w) if w == "off" || w == "no" || w == "0" => self.film_driver.show_objects = false,
                    Some(w) => return Err(format!("film objects {w}: expected on | off")),
                    None => self.film_driver.show_objects = !self.film_driver.show_objects,
                }
                Ok(format!(
                    "recorded objects: {}  |  {}\n",
                    if self.film_driver.show_objects { "drawn" } else { "hidden" },
                    if self.film_driver.object_status.is_empty() {
                        "nothing resolved yet (seek once with a map open)".to_string()
                    } else {
                        self.film_driver.object_status.clone()
                    }
                ))
            }
            // #fp-frame  Print-only: `film fp [verbose|quiet] [WxH]` steers the first-person
            // framing report that `film show` prints. Diagnostic only -- nothing in the render
            // path reads either setting.
            "fp" => {
                for a in args {
                    let a = a.to_lowercase();
                    match a.as_str() {
                        "verbose" | "nodes" | "on" => self.fp_verbose = true,
                        "quiet" | "off" => self.fp_verbose = false,
                        _ => {
                            let (w, h) = a.split_once('x').ok_or_else(|| {
                                format!("film fp {a}: expected verbose | quiet | WxH")
                            })?;
                            let w: f32 = w.parse().map_err(|_| format!("film fp {a}: bad width"))?;
                            let h: f32 = h.parse().map_err(|_| format!("film fp {a}: bad height"))?;
                            if w <= 0.0 || h <= 0.0 {
                                return Err(format!("film fp {a}: bad size"));
                            }
                            self.shot_aspect = w / h;
                        }
                    }
                }
                Ok(format!(
                    "first-person framing report: aspect {:.4}, {}\n",
                    self.shot_aspect,
                    if self.fp_verbose { "every node" } else { "summary only" }
                ))
            }
            // #fp-fov  `film fov [degrees]` -- the FIRST-PERSON field of view, in the game's own
            // units (horizontal at the engine's 4:3 reference, the number MCC's slider shows).
            // This is the SAME persisted setting the window's slider drives, so a scripted capture
            // and the window frame alike. (`cam fov` is a different thing: a raw vertical override
            // of the editor camera, for measurement.)
            "fov" => {
                if let Some(a) = args.first() {
                    let v: f32 = a.parse().map_err(|_| format!("film fov {a}: expected degrees"))?;
                    if !(crate::film_play::FOV_MIN_DEG..=crate::film_play::FOV_MAX_DEG).contains(&v) {
                        return Err(format!(
                            "film fov {v}: expected {}..{} degrees",
                            crate::film_play::FOV_MIN_DEG,
                            crate::film_play::FOV_MAX_DEG
                        ));
                    }
                    self.fp_fov_deg = v;
                    crate::film_play::save_fov_setting(v);
                }
                let vert = crate::film_play::reach_vertical_fov_for(self.fp_fov_deg);
                // #fp-fov ...and the field of view the RIG itself is drawn at, which is not the
                // world's: the engine renders the first-person model through its own camera at
                // 0.88 of the world camera's vertical FOV, and that is what stops the arms
                // splaying as this number climbs.
                let rig = crate::film_play::reach_first_person_model_fov_for(self.fp_fov_deg);
                Ok(format!(
                    "first-person FOV {:.1}\u{00B0} (Reach's own units) = {:.2}\u{00B0} vertical, {:.2}\u{00B0} horizontal at 16:9; \
                     the arms + weapon are drawn through their own camera at {:.2}\u{00B0} vertical (x{:.3}, the engine's)\n",
                    self.fp_fov_deg,
                    vert.to_degrees(),
                    2.0 * ((16.0 / 9.0) * (vert * 0.5).tan()).atan().to_degrees(),
                    rig.to_degrees(),
                    crate::film_play::first_person_fov_scale(),
                ))
            }
            "folders" => {
                let dirs = crate::film_ui::film_search_folders();
                if dirs.is_empty() {
                    return Ok("no MCC saved-film folders found\n".into());
                }
                Ok(dirs.iter().map(|d| format!("{}\n", d.display())).collect())
            }
            // #film-action  Still here to FORCE a redraw; playback now does it by itself. One
            // decision, in `film_export::variant_from_film`.
            "scene" | "draw" => {
                let f = self.film.clone().ok_or("film: no film open (film <path>)")?;
                let v = film_export::variant_from_film(&f).map_err(|e| format!("film scene: {e}"))?;
                let (n, title) = (v.objects.len(), v.title.clone());
                let applied = self.apply_variant_value(v, "film_scene".into(), std::time::Duration::ZERO)?;
                self.film_scene = Some(format!("Forge scene: {n} object(s) from {title:?}"));
                Ok(format!("film scene: {n} objects from {title:?}\n{applied}"))
            }
            other => {
                let path = std::path::PathBuf::from(other);
                if !path.is_file() {
                    return Err(format!("film: no such file '{}'", path.display()));
                }
                // Drop whatever was open BEFORE trying the new file. MCC leaves zero-byte
                // `.film` stubs in its autosave folder, and a failed open that left the previous
                // film in place made every later verb silently report the OLD film's numbers.
                self.film = None;
                self.film_path = None;
                self.film_play = None;
                self.film_scene = None;
                let f = film::open_header(&path).map_err(|e| e.to_string())?;
                let text = rows(&f, label(&f));
                self.film = Some(Box::new(f));
                self.film_spawn_weapon = None; // #gamevariant a new film has a new variant
                self.film_path = Some(path);
                Ok(text)
            }
        }
    }

    fn exec_cmd(&mut self, cmd: script::EditorCommand) -> Result<String, String> {
        use script::{CameraCmd, EditorCommand, Placement};
        match cmd {
            EditorCommand::LoadMap(name) => {
                let idx = self.resolve_map_index(&name).ok_or_else(|| format!("no map matching '{name}'"))?;
                let path = self.cands[idx].path.to_string_lossy().into_owned();
                self.load_base_map(&path)
            }
            EditorCommand::LoadVariant(path) => self.apply_variant(&path),
            EditorCommand::Wait => Ok(String::new()), // loads are synchronous here
            EditorCommand::Pick { delete, radius } => {
                self.ensure_built(); // build the pick set for the current objects
                let dir = self.camera.forward();
                let hit = self
                    .objscene()
                    .and_then(|s| s.pick(self.camera.pos, dir))
                    .ok_or("nothing under the camera crosshair")?;
                let center = self.movable_index(hit).map(|i| Vec3::from(self.objects[i].pos)).unwrap_or(Vec3::ZERO);
                let mut set = vec![hit];
                if let Some(r) = radius {
                    let r2 = r * r;
                    for o in &self.objects {
                        if o.datum != hit && (Vec3::from(o.pos) - center).length_squared() <= r2 {
                            set.push(o.datum);
                        }
                    }
                }
                if delete {
                    self.objects.retain(|o| !set.contains(&o.datum));
                    for d in &set { self.meta.remove(d); self.colors.remove(d); self.h4_records.remove(d); }
                    self.selected.clear();
                    self.dirty = true;
                    Ok(format!("deleted {} object(s) under crosshair", set.len()))
                } else {
                    self.selected = set.clone();
                    // Say what was hit, same line as the GUI.
                    let what = self.object_identity(hit).map(|id| id.status_line()).unwrap_or_else(|| format!("0x{hit:08X}"));
                    Ok(format!("under crosshair: {what} — {} selected", set.len()))
                }
            }
            EditorCommand::Screenshot { path, size } => self.screenshot(&path, size),
            // A Halo 4 variant saves / duplicates here (pure record work, no live game); Reach
            // variant writing stays on the interactive app (it owns the edit lists and the
            // save / edit bookkeeping): refuse clearly rather than pretending to save.
            EditorCommand::Save { path } if self.h4_scene.is_some() => self.h4_save(path.as_deref()),
            EditorCommand::Dup { datum, off } if self.h4_scene.is_some() => self.h4_dup(datum, off),
            EditorCommand::Save { .. } | EditorCommand::Dup { .. } => {
                Err("save/dup need the interactive app (Reach); a Halo 4 variant saves here".into())
            }
            // #snap-array: a line/axis array STAMPS copies, so it rides on the same duplication
            // the headless host does not own for Reach.
            EditorCommand::ArrayLine { .. } | EditorCommand::ArrayAxis { .. } => {
                Err("array line/axis needs the interactive app (it duplicates objects)".into())
            }
            // #snap-array: the scripted Ctrl magnet -- a pure translation, so it works here.
            EditorCommand::SnapTo { target, to, axis } => {
                let moving: Vec<u32> = match target {
                    Some(d) => vec![d],
                    None => self.selected.clone(),
                };
                if moving.is_empty() {
                    return Err("snapto needs a datum or a selection".into());
                }
                let candidates: Vec<u32> = match to {
                    Some(d) => vec![d],
                    None => self.objects.iter().map(|o| o.datum).filter(|d| !moving.contains(d)).collect(),
                };
                let tags: HashMap<u32, u32> = self.objects.iter().map(|o| (o.datum, o.mode_tag)).collect();
                let mut cache = crate::snap::PlaneCache::default();
                let cfg = crate::snap::SnapCfg::default();
                let res = {
                    let scene = self.objscene().ok_or("snapto: no map loaded")?;
                    // An explicit target means only that object's faces; otherwise the level too.
                    let ax = axis.map(Vec3::from);
                    crate::snap::solve_scene(scene, &moving, &candidates, &|d| tags.get(&d).copied().unwrap_or(0), ax, to.is_none(), &mut cache, &cfg)
                }
                .ok_or_else(|| format!("snapto: no face within {:.2} wu of the selection", cfg.range))?;
                for &d in &moving {
                    if let Some(i) = self.movable_index(d) {
                        let p = Vec3::from(self.objects[i].pos) + res.delta;
                        self.objects[i].pos = p.to_array();
                    }
                    if let Some(sc) = self.objscene_mut() {
                        sc.translate_pick(d, res.delta);
                    }
                }
                self.dirty = true;
                Ok(format!(
                    "snapped {} object(s) by ({:.3}, {:.3}, {:.3}) — {}",
                    moving.len(), res.delta.x, res.delta.y, res.delta.z, res.describe()
                ))
            }
            EditorCommand::Count { type_filter, .. } => {
                // Headless `count` takes no name / label filter (documented); a type filter
                // matches nothing here.
                let n = if type_filter.is_none() { self.objects.len() } else { 0 };
                Ok(format!("{n} object(s)"))
            }
            EditorCommand::Camera(c) => {
                match c {
                    CameraCmd::To(p) => self.camera.pos = Vec3::from(p),
                    CameraCmd::LookAt(p) => {
                        let to = (Vec3::from(p) - self.camera.pos).normalize_or_zero();
                        self.camera.yaw = to.y.atan2(to.x);
                        self.camera.pitch = to.z.clamp(-1.0, 1.0).asin();
                    }
                    CameraCmd::Spawn => {
                        // the shared load / teleport rule (spawn_camera.rs); the source is logged
                        if let Some(p) = self.objscene().and_then(|s| s.spawn_camera()) {
                            self.camera.pos = p.pos;
                            self.camera.yaw = p.yaw;
                            self.camera.pitch = p.pitch;
                            eprintln!("camera spawn: {}", p.describe());
                        } else {
                            eprintln!("camera spawn: no spawn / starting location / soft ceiling / bounds source - framing the scene");
                            self.frame_scene();
                        }
                    }
                    CameraCmd::Frame => self.frame_forge(),
                    CameraCmd::Nudge(d) => self.camera.pos += Vec3::from(d),
                    // #fp-frame  Print-only diagnostic. The vertical field of view a scripted
                    // capture shoots at; the app's own default is unchanged by it.
                    CameraCmd::Fov(deg) => {
                        if let Some(d) = deg {
                            if !(1.0..179.0).contains(&d) {
                                return Err(format!("camera fov {d}: expected 1..179 degrees"));
                            }
                            self.camera.fov_y = d.to_radians();
                        }
                        return Ok(format!(
                            "camera fov_y {:.2}° (={:.2}° horizontal at {:.4})\n",
                            self.camera.fov_y.to_degrees(),
                            2.0 * ((self.camera.fov_y * 0.5).tan() * self.shot_aspect).atan().to_degrees(),
                            self.shot_aspect
                        ));
                    }
                    CameraCmd::Get => {
                        return Ok(format!(
                            "camera pos {:.2} {:.2} {:.2}  yaw {:.1}°  pitch {:.1}°",
                            self.camera.pos.x, self.camera.pos.y, self.camera.pos.z,
                            self.camera.yaw.to_degrees(), self.camera.pitch.to_degrees()
                        ));
                    }
                    CameraCmd::Clearance | CameraCmd::Standoff(_) => {
                        // The stand-off solver lives on the interactive App (it needs the live
                        // object pick list). Say so rather than silently doing nothing.
                        return Err("camera standoff/clearance need the interactive app (they probe placed objects)".into());
                    }
                    CameraCmd::Orbit { target, dist, yaw, pitch } => {
                        let focus = match target {
                            crate::script::OrbitTarget::Point(p) => Vec3::from(p),
                            crate::script::OrbitTarget::Datum(d) => self
                                .objects
                                .iter()
                                .find(|o| o.datum == d)
                                .map(|o| Vec3::from(o.pos))
                                .ok_or_else(|| format!("no object 0x{d:08X}"))?,
                            crate::script::OrbitTarget::Selection => {
                                return Err("camera orbit selection needs the interactive app".into())
                            }
                        };
                        let (y, p) = (yaw.to_radians(), pitch.to_radians().clamp(-1.5, 1.5));
                        self.camera.pos = focus - Vec3::new(y.cos() * p.cos(), y.sin() * p.cos(), p.sin()) * dist;
                        self.camera.yaw = y;
                        self.camera.pitch = p;
                        return Ok(format!("orbiting ({:.2},{:.2},{:.2}) at {dist:.1} wu", focus.x, focus.y, focus.z));
                    }
                }
                Ok("camera set".into())
            }
            EditorCommand::MapId(r) => {
                let id = self.resolve_map_id(&r)?;
                Ok(format!("0x{id:08X} ({id})"))
            }
            EditorCommand::ListMaps(filter) => {
                let f = filter.unwrap_or_default().to_lowercase();
                let mut s = String::new();
                let mut n = 0;
                for c in &self.cands {
                    let label = c.label();
                    if !f.is_empty() && !label.to_lowercase().contains(&f) {
                        continue;
                    }
                    let id = c.map_id.map(|v| format!("0x{v:04X}")).unwrap_or_else(|| "?".into());
                    s.push_str(&format!("  [{}] id={id}  {label}\n", if c.modded { "mod" } else { "stk" }));
                    n += 1;
                }
                Ok(format!("{n} maps:\n{s}"))
            }
            EditorCommand::ListVariants { sel, filter } => {
                let (items, s) = self.variant_listing(&sel, filter.as_deref());
                Ok(format!("{} variants:\n{s}", items.len()))
            }
            EditorCommand::ListPalette(filter) if self.h4_scene.is_some() => Ok(self.h4_list_palette(filter.as_deref())),
            EditorCommand::ListPalette(filter) => {
                let f = filter.unwrap_or_default().to_lowercase();
                let mut s = String::new();
                let mut n = 0;
                for (i, (_, _, name, _)) in self.place_palette.iter().enumerate() {
                    if !f.is_empty() && !name.to_lowercase().contains(&f) {
                        continue;
                    }
                    s.push_str(&format!("  #{i}  {name}\n"));
                    n += 1;
                    if n >= 400 {
                        s.push_str("  … (truncated)\n");
                        break;
                    }
                }
                Ok(format!("{n} palette objects:\n{s}"))
            }
            EditorCommand::ListObjects { type_filter, name_filter, label_filter } => {
                let tf = type_filter.map(|s| s.to_lowercase());
                let nf = name_filter.map(|s| s.to_lowercase());
                let lf = label_filter.map(|s| s.to_lowercase());
                let mut s = String::new();
                let mut n = 0;
                for o in &self.objects {
                    let m = self.meta.get(&o.datum);
                    let name = m.map(|m| m.name.clone()).unwrap_or_default();
                    let label = m.map(|m| m.label.clone()).unwrap_or_default();
                    let ct = m.map(|m| m.cached_type).unwrap_or(0);
                    if let Some(tf) = &tf {
                        let num_ok = tf.parse::<u8>().map(|v| v == ct).unwrap_or(false);
                        if !num_ok && !name.to_lowercase().contains(tf) {
                            continue;
                        }
                    }
                    if nf.as_ref().map(|nf| !name.to_lowercase().contains(nf)).unwrap_or(false) {
                        continue;
                    }
                    if lf.as_ref().map(|lf| !label.to_lowercase().contains(lf)).unwrap_or(false) {
                        continue;
                    }
                    s.push_str(&format!("  0x{:08X}  ({:.1},{:.1},{:.1})  type={ct}  {name}\n", o.datum, o.pos[0], o.pos[1], o.pos[2]));
                    n += 1;
                    if n >= 1000 {
                        break;
                    }
                }
                Ok(format!("{n} objects:\n{s}"))
            }
            EditorCommand::ListTypes => {
                let mut hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                for o in &self.objects {
                    let name = self.meta.get(&o.datum).map(|m| m.name.clone()).unwrap_or_else(|| format!("tag_{:08X}", o.primary_tag));
                    *hist.entry(name).or_default() += 1;
                }
                let mut v: Vec<(String, usize)> = hist.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                let mut s = String::new();
                for (name, c) in &v {
                    s.push_str(&format!("  {c:>4}  {name}\n"));
                }
                Ok(format!("{} distinct types:\n{s}", v.len()))
            }
            EditorCommand::Get(datum) => Ok(self.dump_object(datum)),
            EditorCommand::Place { obj, pos } => {
                let PlaceTarget { mode, obj_tag, name, variant_sid, h4 } = self.resolve_place(&obj)?;
                let world = match pos {
                    Placement::At(p) => Vec3::from(p),
                    Placement::Camera => {
                        let fwd = self.camera.forward();
                        self.objscene().and_then(|s| s.raycast_scene(self.camera.pos, fwd)).unwrap_or(self.camera.pos + fwd * 12.0)
                    }
                    Placement::Relative { datum, off } => {
                        let i = self.movable_index(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                        Vec3::from(self.objects[i].pos) + Vec3::from(off)
                    }
                    Placement::OnFace { datum, dir } => {
                        let i = self.movable_index(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                        let p = Vec3::from(self.objects[i].pos);
                        let d = Vec3::from(dir).normalize_or_zero();
                        let half = self.objscene().and_then(|s| s.aabb_of(datum)).map(|(mn, mx)| {
                            let e = Vec3::from(mx) - Vec3::from(mn);
                            0.5 * (e.x * d.x.abs() + e.y * d.y.abs() + e.z * d.z.abs())
                        }).unwrap_or(1.0);
                        p + d * half
                    }
                };
                let datum = self.next_datum;
                self.next_datum = self.next_datum.wrapping_add(1);
                let (mut fwd, mut up) = ([1.0f32, 0.0, 0.0], [0.0f32, 0.0, 1.0]);
                if let Some((quota, variant)) = h4 {
                    // The placed object gets its full .mvar record at once (the engine's Forge
                    // defaults for that palette entry) so it saves; the live pose is the record's
                    // decoded basis (what the file will hold).
                    let rec = self.h4_instantiate(quota, variant, world.into(), fwd, up)?;
                    fwd = rec.fwd;
                    up = rec.up;
                    self.meta.insert(datum, h4_hmeta(&rec, &name, &self.h4_labels));
                    self.colors.insert(datum, (rec.team as u8, rec.color.unwrap_or(0xFF)));
                    self.h4_records.insert(datum, rec);
                } else {
                    self.meta.insert(datum, HMeta { name: name.clone(), team: mvar::TEAM_NEUTRAL, color: -1, ..Default::default() });
                }
                self.objects.push(ObjectInfo {
                    datum, type_sig: 0, sig0: 0, sig1: 0,
                    pos: world.into(), health: 1.0, shield: 1.0,
                    mode_tag: mode, fwd, up,
                    attached: [0; 8], primary_tag: obj_tag,
                    variant_name_sid: variant_sid,
                });
                self.selected = vec![datum];
                self.dirty = true;
                Ok(format!("placed '{name}' → 0x{datum:08X} at ({:.1},{:.1},{:.1})", world.x, world.y, world.z))
            }
            EditorCommand::Move { datum, delta } => {
                let i = self.movable_index(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                for k in 0..3 { self.objects[i].pos[k] += delta[k]; }
                self.dirty = true;
                Ok(format!("moved 0x{datum:08X}"))
            }
            EditorCommand::MoveTo { datum, pos } => {
                let i = self.movable_index(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                self.objects[i].pos = pos;
                self.dirty = true;
                Ok(format!("moved 0x{datum:08X}"))
            }
            // Face-to-face coincident constraint (the scripted form of the Construct panel's
            // "Mate to B"). Moves object `a` only, so a script can state an exact relationship
            // between two placements without hand-computing the offset.
            EditorCommand::Coincident { a, af, b, bf, centered, turn } => {
                let scene = self.objscene().ok_or("coincident: no map loaded")?;
                let oa = scene.object_obb(a).ok_or_else(|| format!("coincident: no bounds for 0x{a:08X}"))?;
                let ob = scene.object_obb(b).ok_or_else(|| format!("coincident: no bounds for 0x{b:08X}"))?;
                let bn = ob.face_normal(bf);
                if bn.length_squared() < 0.5 {
                    return Err(format!("coincident: 0x{b:08X} face {bf} has no normal (zero-extent axis)"));
                }
                let mate = if centered { crate::construct::FaceMate::Centered } else { crate::construct::FaceMate::Plane };
                let ac = oa.face_center(af);
                let (q, delta) = crate::construct::face_mate_transform(ac, oa.face_normal(af), ob.face_center(bf), bn, mate, turn);
                let i = self.movable_index(a).ok_or_else(|| format!("no object 0x{a:08X}"))?;
                let p = glam::Vec3::from(self.objects[i].pos);
                // rotate about face A's centre, then translate
                let np = ac + q * (p - ac) + delta;
                self.objects[i].pos = np.to_array();
                if turn {
                    let f = glam::Vec3::from(self.objects[i].fwd);
                    let u = glam::Vec3::from(self.objects[i].up);
                    self.objects[i].fwd = (q * f).to_array();
                    self.objects[i].up = (q * u).to_array();
                }
                // Keep the cached pick box in step (turned and moved the same way), so a second
                // constraint in the same script sees the new pose instead of recomputing the same
                // delta and double-moving.
                if let Some(sc) = self.objscene_mut() {
                    sc.rotate_pick(a, q, ac);
                    sc.translate_pick(a, delta);
                }
                self.dirty = true;
                let how = if centered { "centred" } else { "flush" };
                Ok(format!("coincident: moved 0x{a:08X} {how} onto 0x{b:08X} (delta {:.3},{:.3},{:.3})", delta.x, delta.y, delta.z))
            }
            EditorCommand::Settle(which) => {
                let targets: Vec<u32> = match which {
                    Some(d) => vec![d],
                    None => self.selected.clone(),
                };
                if targets.is_empty() {
                    return Err("settle needs a datum or a selection".into());
                }
                // the scales / picks must be current before the bodies are built
                self.ensure_built();
                let mut n = 0;
                let mut lines = Vec::new();
                for d in &targets {
                    match self.settle_one(*d) {
                        Ok(line) => {
                            if !line.starts_with("No contact") { n += 1; }
                            lines.push(format!("0x{d:08X}: {line}"));
                        }
                        Err(e) => lines.push(format!("0x{d:08X}: {e}")),
                    }
                }
                self.dirty = true;
                Ok(format!("settled {n}/{} object(s)\n{}", targets.len(), lines.join("\n")))
            }
            EditorCommand::Rotate { datum, axis, deg } => {
                let i = self.movable_index(datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
                let a = [Vec3::X, Vec3::Y, Vec3::Z][axis];
                let q = glam::Quat::from_axis_angle(a, deg.to_radians());
                self.objects[i].fwd = (q * Vec3::from(self.objects[i].fwd)).into();
                self.objects[i].up = (q * Vec3::from(self.objects[i].up)).into();
                // Keep the cached oriented box in step, so a following `coincident` reads the
                // object's real face normals instead of its pre-rotation ones.
                let pivot = Vec3::from(self.objects[i].pos);
                if let Some(sc) = self.objscene_mut() { sc.rotate_pick(datum, q, pivot); }
                self.dirty = true;
                Ok(format!("rotated 0x{datum:08X} {deg}°"))
            }
            EditorCommand::Set { datum, field, value } if datum == crate::script::SET_SELECTION => {
                // Apply the same field to every selected object.
                let targets = self.selected.clone();
                if targets.is_empty() {
                    return Err("set selection: nothing selected".into());
                }
                let mut n = 0usize;
                let mut last_err: Option<String> = None;
                for d in targets {
                    match self.exec_cmd(EditorCommand::Set { datum: d, field: field.clone(), value: value.clone() }) {
                        Ok(_) => n += 1,
                        Err(e) => last_err = Some(e),
                    }
                }
                if n == 0 {
                    return Err(last_err.unwrap_or_else(|| "set selection: no editable objects".into()));
                }
                Ok(format!("set {field} on {n} object(s)"))
            }
            // Halo 4 objects: edit the full record.
            EditorCommand::Set { datum, field, value } if self.h4_records.contains_key(&datum) => self.h4_set(datum, &field, &value),
            EditorCommand::Set { datum, field, value } => {
                let m = self.meta.get_mut(&datum).ok_or_else(|| format!("no editable object 0x{datum:08X}"))?;
                match field.as_str() {
                    "team" => m.team = script::parse_team(&value)?,
                    "color" => m.color = value.parse().map_err(|_| "color must be -1..7")?,
                    "label" => m.label = value.clone(),
                    "scaled" => m.flags.scaled = forge_scale::parse_flag_value(&value)?,
                    "shadow" | "castshadow" | "shadowcaster" => m.flags.shadow = forge_scale::parse_flag_value(&value)?,
                    "damage" | "damagestate" | "damage_state" | "state" => m.flags.damage = forge_scale::parse_damage_value(&value)?,
                    "spawnseq" | "spawn_seq" => m.spawn_seq = value.parse().map_err(|_| "spawnseq must be int")?,
                    "scale" => {
                        m.label = "scale".into();
                        let s: f32 = value.parse().map_err(|_| "scale must be a number")?;
                        m.spawn_seq = forge_scale::object_scale_to_seq(s).0 as i32;
                    }
                    "cachedtype" | "cached_type" => m.cached_type = value.parse().map_err(|_| "cached_type must be int")?,
                    "respawn" => m.respawn = value.parse().map_err(|_| "respawn must be int")?,
                    "shape" | "boundary" => {
                        m.boundary_shape = match value.to_lowercase().as_str() {
                            "none" => 0, "sphere" => 1, "cylinder" => 2, "box" => 3,
                            _ => return Err("shape must be none/sphere/cylinder/box".into()),
                        };
                    }
                    "radius" | "width" => m.boundary[0] = crate::wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "length" => m.boundary[1] = crate::wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "top" => m.boundary[2] = crate::wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "bottom" => m.boundary[3] = crate::wu_to_bval(value.parse().map_err(|_| "need a number")?),
                    "b0" => m.boundary[0] = value.parse().map_err(|_| "b0 must be 0..2047")?,
                    "b1" => m.boundary[1] = value.parse().map_err(|_| "b1 must be 0..2047")?,
                    "b2" => m.boundary[2] = value.parse().map_err(|_| "b2 must be 0..2047")?,
                    "b3" => m.boundary[3] = value.parse().map_err(|_| "b3 must be 0..2047")?,
                    other => return Err(format!("unknown field '{other}'")),
                }
                let cu = if m.color < 0 { 0xFFu8 } else { m.color as u8 };
                let team = m.team;
                self.colors.insert(datum, (team, cu));
                self.dirty = true;
                Ok(format!("set 0x{datum:08X} {field}={value}"))
            }
            EditorCommand::Select(d) => { self.selected = vec![d]; Ok(format!("selected 0x{d:08X}")) }
            EditorCommand::SelectAll => { self.selected = self.objects.iter().map(|o| o.datum).collect(); Ok(format!("selected {}", self.selected.len())) }
            EditorCommand::Deselect => { self.selected.clear(); Ok("deselected".into()) }
            EditorCommand::Delete(which) => {
                let dd: Vec<u32> = which.map(|d| vec![d]).unwrap_or_else(|| self.selected.clone());
                if dd.is_empty() {
                    return Err("nothing to delete".into());
                }
                self.objects.retain(|o| !dd.contains(&o.datum));
                for d in &dd { self.meta.remove(d); self.colors.remove(d); self.h4_records.remove(d); }
                self.selected.retain(|d| !dd.contains(d));
                self.dirty = true;
                Ok(format!("deleted {} object(s)", dd.len()))
            }
            EditorCommand::Echo(t) => Ok(t),
            // #reach-skin The headless host has no renderer, so `actor` here resolves and
            // REPORTS (bipeds, weapons, animation names, the marker pair, the pose state) but
            // draws nothing; the GUI executor in main.rs does the same commands AND rebuilds
            // the actor meshes.
            EditorCommand::Actor(args) => {
                let Some(scene) = self.scene.as_deref_mut() else {
                    return Err("`actor` needs an open Reach map".into());
                };
                let cache = scene.cache_handle();
                let r = crate::reach_actor::run_actor_command(&args, &mut self.actors, scene.native_dll(), cache)?;
                if self.actors.rebuild {
                    let t0 = std::time::Instant::now();
                    loop {
                        let b = crate::reach_actor::build_actor_meshes(
                            &mut self.actors, scene, self.renderer.mesh_renderer(), &self.device, &self.queue);
                        let done = !b.pending || t0.elapsed().as_secs() > 120;
                        if done {
                            self.install_actor_build(b);
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
                self.actors.update(0.0, &self.queue);
                self.push_actor_particles();
                Ok(r)
            }
            EditorCommand::ScreenFx(set) => {
                if let Some(on) = set { self.forge_fx_enabled = on; }
                self.refresh_screen_fx();
                Ok(self.screenfx_status())
            }
            // Run-local globals (not persisted from a batch run).
            EditorCommand::ScaledGlobal(set) => {
                if let Some(on) = set { self.globals.scaled = on; self.dirty = true; }
                Ok(self.flags_status())
            }
            EditorCommand::ShadowCastersGlobal(set) => {
                if let Some(on) = set { self.globals.shadowcasters = on; self.dirty = true; }
                Ok(self.flags_status())
            }
            // Run-local (not persisted from a batch run), like the flag globals.
            EditorCommand::MapSpawns(set) => {
                if let Some(on) = set { self.set_map_spawns(on); }
                Ok(self.map_spawns_status())
            }
            // World-box select over the run's objects with meta (variant / placed), by the
            // scene's pick bounds (a hidden block's hull): the GUI marquee's test.
            EditorCommand::SelectBox { min, max, additive } => {
                self.ensure_built(); // pick bounds come from the object rebuild
                let scene = self.objscene().ok_or("no map loaded")?;
                let mut hits = if additive { self.selected.clone() } else { Vec::new() };
                for o in &self.objects {
                    if !self.meta.contains_key(&o.datum) || hits.contains(&o.datum) {
                        continue;
                    }
                    if crate::physics_outlines::box_overlaps(min, max, o.pos, scene.aabb_of(o.datum)) {
                        hits.push(o.datum);
                    }
                }
                self.selected = hits;
                Ok(format!("selected {} objects (box)", self.selected.len()))
            }
            EditorCommand::PhysicsOutlines(set) => {
                if let Some(on) = set { self.show_physics_outlines = on; }
                Ok(self.physics_outlines_status())
            }
            // #h4-phys The batch host draws no selection overlay; the verbs are accepted (and
            // report the selected object's real edge counts) so one script runs on both hosts.
            EditorCommand::CollisionOverlay(set) => {
                if let Some(on) = set { self.show_collision_hull = on; }
                Ok(self.hull_overlay_status())
            }
            EditorCommand::PhysicsOverlay(set) => {
                if let Some(on) = set { self.show_physics_hull = on; }
                Ok(self.hull_overlay_status())
            }
            // The script host has no window; the verb is accepted so a shared script runs on both.
            EditorCommand::SettingsPanel(_) => Ok("settings panel: not applicable (no window)".into()),
            // File ▸ Open recent is the app's own menu history; the headless host opens
            // exactly what its script names, so it keeps none.
            EditorCommand::RecentDbg { .. } => Ok("recent: not applicable (headless, no menus)".into()),
            // #dialogs The browser is GUI-only; the headless host has no window.
            EditorCommand::DialogDbg { .. } => Ok("dialog: not applicable (headless, no window)".into()),
            // #model-import: the imported-mesh lane and its gizmo live in the GUI only.
            EditorCommand::ImportModel { .. } => Ok("import: not applicable (headless host has no imported-model lane; run it in the app)".into()),
            // #film  Reach Theater saved films. Everything but drawing works headless.
            EditorCommand::Film { sub, args } => self.film_verb(&sub, &args),
            // #wire-visible  Run-local; the selection lane is re-applied on every screenshot.
            EditorCommand::WireXray(set) => {
                if let Some(on) = set { self.wire_xray = on; }
                Ok(self.wire_xray_status())
            }
            // #quant-error  The save-precision prediction, headless. Same numbers and the same
            // snap as the interactive app (the host has no viewport, so `ghost` only reports).
            EditorCommand::Precision { sub, args } => self.precision_cmd(&sub, &args),
            // #cannon-preview  Run-local; recomputed and drawn on every screenshot.
            EditorCommand::CannonPreview(set) => {
                if let Some(on) = set { self.cannon_preview.on = on; }
                let mut retry = false;
                if let (true, Some(sc)) = (self.cannon_preview.on, self.scene.as_ref()) {
                    self.cannon_preview.sync(sc, &self.objects, &self.selected, &mut retry);
                }
                let lines: Vec<String> = self.cannon_preview.results.iter().map(|t| format!("  {:#x}: {:?} {}", t.datum, t.verdict, t.label().join(" | "))).collect();
                Ok(format!("man cannon trajectory: {}\n{}", if self.cannon_preview.on { "on" } else { "off" }, lines.join("\n")))
            }
            // Run-local; the overlay is re-merged on every screenshot.
            EditorCommand::SoftCeilings(set) => {
                if let Some(on) = set { self.show_soft_ceilings = on; }
                let mut out = self.soft_ceilings_status();
                if set.is_none() {
                    let cs = self.scene.as_ref().map(|s| s.soft_ceilings()).unwrap_or_default();
                    for l in crate::soft_ceilings::listing(&cs) {
                        out.push('\n');
                        out.push_str(&l);
                    }
                }
                Ok(out)
            }
            EditorCommand::HardFloor(set) => {
                if let Some(on) = set { self.show_hard_floor = on; }
                let mut out = self.hard_floor_status();
                if set.is_none() {
                    if let Some(scene) = self.scene.as_ref() {
                        for l in crate::hard_floor::listing(scene.structure_bsp_flags(), &scene.structure_bsp_mopp_bounds()) {
                            out.push('\n');
                            out.push_str(&l);
                        }
                    }
                }
                Ok(out)
            }
            EditorCommand::PlayableBounds(set) => {
                if let Some(on) = set { self.show_playable_bounds = on; }
                Ok(self.playable_bounds_status())
            }
            EditorCommand::TriggerVolumes(set) => {
                if let Some(on) = set { self.show_triggers = on; }
                let n = self.scene.as_ref().map(|s| s.trigger_volumes().len()).unwrap_or(0);
                Ok(format!("trigger volumes: {} ({n} volume(s))", if self.show_triggers { "shown" } else { "hidden" }))
            }
            EditorCommand::NewVariant => Err("newvariant: interactive app only (headless: loadvariant a template, then save as)".into()),
            // The headless host reports either game's globals; edits apply to a Halo 4 variant
            // (its save lives here), a Reach edit needs the interactive app.
            EditorCommand::VariantGet(field) => self.variant_get(field.as_deref()),
            EditorCommand::VariantSet { field, value } => self.variant_set(&field, &value),
            EditorCommand::Undo | EditorCommand::Redo => Err("undo/redo: the headless batch host keeps no edit history (interactive app only)".into()),
            // A hover preview only exists in the interactive app.
            EditorCommand::PreviewHover { .. } | EditorCommand::PointerSim(_) => Err("preview: the dropdown hover preview is interactive-app only (no panel to hover here)".into()),
            // #construct-h4: the Construct tool is a viewport tool -- there is no viewport here.
            EditorCommand::Construct(_) => Err("construct: the Construct (CAD) tool is interactive-app only (no viewport to click in here)".into()),
            EditorCommand::FlagsGet => {
                let mut out = self.flags_status();
                let mut sel = self.selected.clone();
                sel.sort();
                for d in sel {
                    if let Some(m) = self.meta.get(&d) {
                        out.push('\n');
                        out.push_str(&m.flags_line(d, &self.globals));
                    }
                }
                Ok(out)
            }
            // `bspwarn list`: structure BSP flags + every variant / placed object (those
            // with meta, i.e. not the base map's scenery) whose position is in a non-playable BSP.
            // #prefab saved object groups; the same executor the app and the panel use
            EditorCommand::Prefab(args) => self.prefab_command(&args),
            EditorCommand::ConvertVariant(args) => crate::xgame::convert_cli(&args),
            EditorCommand::XGame { what, filter } => {
                let cat = crate::xgame::catalog::Catalog::load();
                Ok(match what.as_str() {
                    "table" => crate::xgame::table::describe(&crate::xgame::table::Table::load(), &cat, filter.as_deref()),
                    "canvases" => {
                        let want = filter.as_deref().and_then(crate::xgame::game_from_str);
                        let mut s = String::new();
                        for g in crate::xgame::GAMES {
                            if want.map_or(false, |w| w != g) { continue; }
                            let cs = crate::xgame::convert::canvases(&cat, g);
                            s.push_str(&format!("{} - {} canvases\n", g.display_name(), cs.len()));
                            for (stem, id) in cs {
                                let n = cat.on_canvas(g, &stem).len();
                                s.push_str(&format!("  {stem:<28} map id {:<8} {n} placeable objects\n", id.map(|i| i.to_string()).unwrap_or_else(|| "-".into())));
                            }
                        }
                        s
                    }
                    _ => crate::xgame::catalog::describe(&cat, filter.as_deref()),
                })
            }
            // #play  Test mode is a WINDOW mode: it needs the viewport's own input, camera and
            // per-frame clock, none of which the headless host has. Refused in words rather than
            // silently doing nothing, which is what a "not supported" that returns Ok would be.
            // `play propdiag [fixed]`: the print-only prop-world report on the host's REAL scene
            // (the one `play` verb that works headless; see `prop_diag`).
            EditorCommand::Play(a) if a.first().is_some_and(|w| w.eq_ignore_ascii_case("propdiag")) => {
                self.ensure_built();
                let scene = self.scene.as_deref().ok_or("no map loaded")?;
                Ok(crate::prop_diag::run(scene, &self.objects, a.get(1).is_some_and(|w| w.eq_ignore_ascii_case("fixed")), 8.0))
            }
            EditorCommand::Play(_) => Err(
                "test mode runs in the window (it needs the viewport's input and camera);                  use the GUI's `play` verb over the command server"
                    .into(),
            ),
            EditorCommand::BspWarnList => {
                if self.h4_scene.is_some() { return Ok(self.h4_bounds_report()); }
                let scene = self.scene.as_ref().ok_or("no map loaded")?;
                let mut out = scene.bsp_flags_report();
                let mut n = 0usize;
                let mut rows = String::new();
                for o in &self.objects {
                    let Some(m) = self.meta.get(&o.datum) else { continue };
                    if let Some(i) = scene.non_playable_bsp_at(Vec3::from(o.pos)) {
                        let bsp_name = scene.structure_bsp_flags().get(i).map(|b| b.name.rsplit('\\').next().unwrap_or("").to_string()).unwrap_or_default();
                        rows.push_str(&format!("\n  0x{:08X}  {}{}   BSP {} ({})   ({:.1}, {:.1}, {:.1})",
                            o.datum, m.name, if m.label.is_empty() { String::new() } else { format!("  \"{}\"", m.label) },
                            i, bsp_name, o.pos[0], o.pos[1], o.pos[2]));
                        n += 1;
                    }
                }
                out.push_str(&format!("{n} object(s) outside playable space{rows}"));
                Ok(out)
            }
        }
    }

    /// Globals + counts of affected objects, plus what the renderer will actually put
    /// into the shadow pass after the next rebuild (engine-default casters included).
    fn flags_status(&mut self) -> String {
        let (mut n_scaled, mut n_cast) = (0usize, 0usize);
        for m in self.meta.values() {
            let (fs, fc) = forge_scale::effective_flags(&self.globals, &m.flags, m.team, &m.label);
            n_scaled += fs as usize;
            n_cast += fc as usize;
        }
        self.ensure_built();
        let (cm, ci, tm, ti) = self.renderer.shadow_caster_counts();
        format!("{} | {n_scaled} scaled, {n_cast} casting | shadow pass: {ci}/{ti} object instances ({cm}/{tm} meshes)", self.globals.status_line())
    }

    fn resolve_place(&self, obj: &script::ObjRef) -> Result<PlaceTarget, String> {
        if self.h4_scene.is_some() { return self.h4_resolve_place(obj); }
        let t = match obj {
            script::ObjRef::Index(i) => self.place_palette.get(*i).cloned().ok_or_else(|| format!("palette index {i} out of range"))?,
            script::ObjRef::Name(n) => {
                let nl = n.to_lowercase();
                self.place_palette.iter().find(|(_, _, name, _)| name.to_lowercase().contains(&nl)).cloned()
                    .ok_or_else(|| format!("no palette object matching '{n}'"))?
            }
        };
        Ok(PlaceTarget { mode: t.0, obj_tag: t.1, name: t.2, variant_sid: t.3, h4: None })
    }

    fn resolve_map_id(&self, r: &script::MapRef) -> Result<u32, String> {
        match r {
            script::MapRef::Current => self.map_id.ok_or_else(|| "no map loaded / id unknown".into()),
            script::MapRef::Named(n) => {
                let i = self.resolve_map_index(n).ok_or_else(|| format!("no map matching '{n}'"))?;
                self.cands[i].map_id.ok_or_else(|| format!("map '{n}' has no readable id"))
            }
            script::MapRef::Variant(p) => mvar::parse_variant(std::path::Path::new(p)).map(|v| v.map_id).ok_or_else(|| format!("cannot parse variant '{p}'")),
        }
    }

    fn variant_listing(&self, sel: &script::VariantSel, filter: Option<&str>) -> (Vec<std::path::PathBuf>, String) {
        let want_id: Option<u32> = match sel {
            script::VariantSel::All => None,
            script::VariantSel::Id(id) => Some(*id),
            script::VariantSel::Current => self.map_id,
            script::VariantSel::MapName(n) => self.resolve_map_index(n).and_then(|i| self.cands[i].map_id),
        };
        let f = filter.map(|s| s.to_lowercase());
        let mut items = Vec::new();
        let mut s = String::new();
        for p in crate::variant_catalog() {
            let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
            if let Some(f) = &f {
                if !stem.contains(f.as_str()) {
                    continue;
                }
            }
            let vid = mvar::parse_variant(&p).map(|v| v.map_id);
            if let Some(want) = want_id {
                if vid != Some(want) {
                    continue;
                }
            }
            let ids = vid.map(|v| format!("0x{v:04X}")).unwrap_or_else(|| "?".into());
            s.push_str(&format!("  id={ids}  {}\n", p.display()));
            items.push(p);
        }
        (items, s)
    }

    /// Tag path / class / tag id / placement of any datum in `objects` (scenario,
    /// variant or placed). Same shape as the GUI's `App::object_identity`.
    fn object_identity(&self, datum: u32) -> Option<crate::obj_identity::ObjIdentity> {
        use crate::obj_identity::{self, ObjIdentity, ObjSource};
        let o = self.objects.iter().find(|o| o.datum == datum)?;
        let (tag_path, class) = match self.objscene() {
            Some(s) if o.primary_tag != 0 && o.primary_tag != 0xFFFF_FFFF => (
                s.tag_name_of(o.primary_tag),
                obj_identity::class_from_code(&s.raw_tag_class(o.primary_tag).unwrap_or_default()),
            ),
            _ => (String::new(), String::new()),
        };
        let scnr = self.base_ident.get(&datum).copied();
        let source = match scnr {
            Some((_, palette_index, name_index)) => {
                ObjSource::Scenario { index: datum.wrapping_sub(0xE000_0000), palette_index, name_index }
            }
            None => ObjSource::Variant {
                palette_name: self.meta.get(&datum).map(|m| m.name.clone()).unwrap_or_default(),
                // A Halo 4 record knows its slot (a new one has none yet).
                slot: self.h4_records.get(&datum).map(|r| r.slot).filter(|s| (*s as usize) < crate::h4::mvar::H4_SLOTS),
            },
        };
        let class = match scnr {
            Some((cat, _, _)) if class.is_empty() => obj_identity::category_name(cat).to_string(),
            _ => class,
        };
        Some(ObjIdentity { datum, obj_tag: o.primary_tag, mode_tag: o.mode_tag, tag_path, class, source })
    }

    fn dump_object(&self, datum: u32) -> String {
        let Some(o) = self.objects.iter().find(|o| o.datum == datum) else {
            return format!("no object 0x{datum:08X}");
        };
        let mut s = format!(
            "0x{datum:08X}\n  pos=({:.2},{:.2},{:.2})\n  fwd=({:.2},{:.2},{:.2}) up=({:.2},{:.2},{:.2})\n  primary_tag=0x{:08X} mode_tag=0x{:08X}\n",
            o.pos[0], o.pos[1], o.pos[2], o.fwd[0], o.fwd[1], o.fwd[2], o.up[0], o.up[1], o.up[2], o.primary_tag, o.mode_tag
        );
        if let Some(id) = self.object_identity(datum) {
            s.push_str(&id.dump_lines());
        }
        if let Some(m) = self.meta.get(&datum) {
            s.push_str(&format!(
                "  name={}\n  team={} ({}) color={} cached_type={} spawn_seq={} respawn={}\n  label='{}' boundary_shape={} scale={:.3}\n",
                m.name, if m.team == 0xFF { -1 } else { m.team as i32 }, crate::forge_team_name(if m.team == 0xFF { -1 } else { m.team as i32 }), // 8 = neutral, -1 = none
                m.color, m.cached_type, m.spawn_seq, m.respawn, m.label, m.boundary_shape, m.scale()
            ));
            s.push_str(&format!("  {}\n", m.flags_line(datum, &self.globals)));
            // #damage the DAMAGE display state + the states this object's model (and its
            // attachments) actually author — what the object panel's DAMAGE dropdown offers.
            let authored = self.objscene().map(|sc| sc.damage_states_of(o)).unwrap_or_default();
            s.push_str(&format!(
                "  damage={} authored=[{}]\n",
                forge_scale::damage_state_name(m.flags.damage_state()),
                authored.iter().map(|&st| forge_scale::damage_state_name(st)).collect::<Vec<_>>().join(", ")
            ));
        }
        if let Some(r) = self.h4_records.get(&datum) { s.push_str(&h4_record_lines(r, &self.h4_labels)); }
        s
    }
}

/// What `resolve_place` found for a `place` target: the editor tags + name, and for a Halo 4
/// palette item its (quota, variant) so the placement gets a full `.mvar` record.
struct PlaceTarget {
    mode: u32,
    obj_tag: u32,
    name: String,
    variant_sid: u32,
    h4: Option<(u8, u8)>,
}

/// The host's shared per-object view of a Halo 4 record (what the Reach verbs read:
/// team / colour / type / spawn / label / boundary + the record's scale); the flags are kept by
/// the caller across re-derivations.
fn h4_hmeta(rec: &H4PlacedObject, name: &str, labels: &[String]) -> HMeta {
    HMeta {
        name: name.to_string(),
        team: rec.team as u8,
        color: rec.color.map(|c| c as i32).unwrap_or(-1),
        cached_type: rec.object_type,
        spawn_seq: rec.spawn_sequence as i32,
        respawn: rec.spawn_time,
        label: rec.labels[0].and_then(|l| labels.get(l as usize).cloned()).unwrap_or_default(),
        boundary_shape: rec.shape as u8,
        boundary: rec.shape_values,
        flags: Default::default(),
        h4_scale: Some(rec.scale()),
    }
}

/// The `get` block of a Halo 4 record: every field, raw + decoded.
fn h4_record_lines(r: &H4PlacedObject, labels: &[String]) -> String {
    let lab = |l: Option<u8>| l.map(|i| format!("{i}:'{}'", labels.get(i as usize).map(String::as_str).unwrap_or("?"))).unwrap_or_else(|| "-".into());
    let wu = r.shape_wu();
    format!(
        "  h4: slot={} flags={} quota={:?} variant={:?} type={} ({}) parent={} in_bounds={}\n  h4: team={} color={:?} spawnseq={} spawntime={}s userdata={} placement={:#05x} (phys {}, hi {}) variant_object_scale={:.3} (q {}{}; stored, not drawn by MCC) locked={}\n  h4: labels [{} {} {} {}]\n  h4: shape={:?} values={:?} (wu {:.2} {:.2} {:.2} {:.2}) typedata={:?}\n",
        r.slot, r.flags, r.quota, r.variant, r.object_type, crate::h4_app::h4_type_name(r.object_type), r.parent, r.in_bounds,
        r.team, r.color, r.spawn_sequence, r.spawn_time, r.user_data, r.placement, r.physics(), (r.placement >> 8) & 3, r.scale(), r.scale_q, if r.is_scaled() { "" } else { ", default" }, r.locked,
        lab(r.labels[0]), lab(r.labels[1]), lab(r.labels[2]), lab(r.labels[3]),
        r.shape, &r.shape_values[..r.shape.value_count()], wu[0], wu[1], wu[2], wu[3], r.type_data,
    )
}

// The Halo 4 branch of the headless host: load through the pure-Rust reader into the
// H4ObjectScene, variants through `variant_to_editor`, `place` from the Forge palette with a full
// record, `set` / `get` over the record, `dup`, and `save` through `build_h4_save_list_core` ->
// gates -> `save_objects`.
impl ScriptHost {
    /// #fp-arms  The GUI's first-person wiring, for this host: resolve the `@player
    /// representation` rows once, then dress them in whichever player the camera is riding.
    fn film_resolve_first_person(&mut self) {
        let Some(sc) = self.scene.as_deref_mut() else { return };
        if self.film_driver.fp_reps.is_empty() {
            self.film_driver.resolve_first_person(sc);
        }
        let dress: Option<(u8, Vec<u8>)> = self.film_play.as_ref().and_then(|pb| {
            if !pb.camera.is_first_person() {
                return None;
            }
            // #fp-arms  The CAMERA's subject, by slot -- including the owner-keyed route the
            // window's camera list uses. Asking only the body's slot binding returned None for an
            // owner group the recording never bound, and no armour was installed at all.
            pb.subject_armour()
        });
        if let (Some((slot, app)), Some(sc)) = (dress, self.scene.as_deref_mut()) {
            self.film_fp_armour = self.film_driver.dress_first_person(sc, slot, &app, self.film_fp_armour);
        }
    }

    /// #gamevariant  The GUI's `film_resolve_spawn_weapon`, for this host.
    ///
    /// Resolving the variant's `motl` loadout collection to a `weap` tag needs an open map, and
    /// the map and the index arrive in either order, so it is attempted until it lands and is a
    /// no-op after that. Clearing `film_spawn_weapon` is what re-arms it for the next film.
    fn film_resolve_spawn_weapon(&mut self) {
        if self.film_spawn_weapon.is_some() || self.film_play.is_none() {
            return;
        }
        let Some(scene) = self.scene.as_deref() else { return };
        let Some(cache) = scene.cache_handle() else { return };
        let native = scene.native_dll();
        let Some(gv) = self.film.as_deref().and_then(|f| f.header.gametype.clone()) else {
            self.film_spawn_weapon = Some(crate::film_variant::SpawnWeaponApplied {
                why: "this recording carries no game variant",
                ..Default::default()
            });
            return;
        };
        let names = crate::film_variant::VariantNames::build(native, cache);
        let weap_tags = native.tags_by_group(cache, "weap");
        let Some(pb) = self.film_play.as_mut() else { return };
        self.film_spawn_weapon = Some(crate::film_variant::apply_spawn_weapons(
            &mut pb.index.tracks,
            crate::film_play::BIPED_TYPE,
            &gv,
            &names,
            &weap_tags,
        ));
    }

    /// Drop every Halo 4 state (a Reach load, or before a fresh Halo 4 load).
    fn h4_drop(&mut self) {
        self.h4_scene = None;
        self.h4_pal = None;
        self.h4_items.clear();
        self.h4_records.clear();
        self.h4_unresolved.clear();
        self.h4_variant = None;
        self.h4_labels.clear();
    }

    /// Load a Halo 4 cache (blocking) into the renderer's static lanes + an `H4ObjectScene`
    /// (dynamic lanes / picks / raycast soup), the Forge palette, and the variant riding along.
    /// Mirrors `h4/gui.rs::h4_load_map` + `h4_drive_load` and `h4/render.rs::headless_run`.
    fn h4_load_base_map(&mut self, path: &str, variant: Option<&std::path::Path>) -> Result<String, String> {
        use crate::h4::scene::{build_meshes, load_map_with_variant, placeholder_sun_dir, H4EditorAssets, H4EditorCollect, Lane};
        let t0 = Instant::now();
        self.scene = None; // the Reach controller goes (a Halo 4 map never touches it)
        self.h4_drop();
        self.clear_renderer_lanes();
        // per-map renderer state back to neutral (the previous map's fog / tints / fx must not leak)
        self.renderer.set_fog(&self.queue, &[0.0f32; 28]);
        self.renderer.set_sky_atmosphere(true);
        self.renderer.set_space_sky(false);
        self.renderer.set_scene_tints([1.0; 3], [1.0; 3]);
        self.renderer.set_screen_fx(&self.queue, 1.0, 0.5, crate::screenfx::ScreenFxRender::identity().cols);
        self.renderer.set_simple_lights(&self.queue, 0, &[]);
        self.renderer.set_planar_fog_volumes(Vec::new());
        self.renderer.set_lens_flare(Vec::new());
        self.renderer.set_particle_meshes(Vec::new());
        self.renderer.set_beam_meshes(Vec::new()); // #projectile-sim
        self.renderer.set_overlay_lines(&self.device, &[]);
        self.renderer.set_xray_lines(&self.device, &[]);
        self.renderer.set_exposure(&self.queue, 1.0);
        self.renderer.set_h4_filmic(&self.queue, None);
        self.renderer.set_sun_dir(placeholder_sun_dir());
        self.screenfx_cache.clear();
        self.h4_screenfx_cache.clear();
        self.screenfx_pushed = None;

        let loaded = load_map_with_variant(path, variant).map_err(|e| format!("{e:#}"))?;
        for l in &loaded.log { log::info!("{l}"); }
        let mr = self.renderer.mesh_renderer_arc();
        let mut batches: Vec<(Lane, Vec<hms_render::GpuMesh>)> = Vec::new();
        let mut collect = H4EditorCollect::new();
        let stats = {
            let mut emit = |lane: Lane, v: Vec<hms_render::GpuMesh>| batches.push((lane, v));
            let mut progress = |_d: u32, _t: u32| {};
            let mut lg = |s: String| log::info!("{s}");
            build_meshes(&loaded, &self.device, &self.queue, &mr, &mut emit, &mut progress, None, &mut lg, Some(&mut collect)).map_err(|e| format!("{e:#}"))?
        };
        for (lane, v) in batches {
            match lane {
                Lane::Opaque => self.renderer.append_static_meshes(v),
                Lane::AlphaTest => self.renderer.append_alphatest_meshes(v),
                Lane::Blend => self.renderer.append_blend_meshes(v),
                Lane::Additive => self.renderer.append_additive_meshes(v),
                Lane::Water => self.renderer.append_water_meshes(v),
                Lane::Sky => self.renderer.set_sky_meshes(v.into_iter().map(|m| { let b = m.blend_mode(); let k = m.centroid()[0]; (m, b, k) }).collect()),
            }
        }
        // lighting / post, as the GUI does it (gui.rs h4_drive_load)
        self.renderer.set_sun_dir(stats.sun_dir());
        self.renderer.set_scene_tints(stats.sun_tint(), [1.0; 3]);
        if let Some(f) = stats.fog { self.renderer.set_fog(&self.queue, &f); }
        // exposure band + engine meter, filmic, bloom, grading LUT, self-illum exposure
        crate::h4::lighting::apply_post(&mut self.renderer, &self.device, &self.queue, stats.camera_fx.as_ref(), stats.color_grading_lut.as_ref());
        let (fmin, fmax) = stats.frame_bounds();
        let n_static = stats.objects_placed;
        let (draws, tris) = (stats.draws, stats.tris);
        let assets = H4EditorAssets::from_loaded(loaded, collect, &stats);
        let es = H4ObjectScene::new(assets, &self.device, &self.queue);
        let pal = crate::h4::palette::palette(es.cache());
        self.h4_items = crate::h4::edit::palette_items(&pal);
        self.h4_pal = Some(pal);
        let riding = es.variant().cloned();
        self.h4_scene = Some(Box::new(es));

        // base-map bookkeeping: no Reach palette / scenery lists; the object set starts empty
        // (scenario placements are static and unpickable)
        self.palette.clear();
        self.types.clear();
        self.place_palette.clear();
        self.base_objects.clear();
        self.base_all.clear();
        self.base_ident.clear();
        self.base_spawn_tags.clear();
        self.objects.clear();
        self.colors.clear();
        self.meta.clear();
        self.selected.clear();
        self.map_path = path.to_string();
        self.map_id = mapcat::read_map_id(std::path::Path::new(path)).or_else(|| {
            self.cands.iter().find(|c| c.path.to_string_lossy() == path).and_then(|c| c.map_id)
        });
        self.cur_stem = std::path::Path::new(path).file_stem().unwrap_or_default().to_string_lossy().into_owned();
        self.dirty = true;
        let cam = crate::h4::render::build_camera(fmin, fmax);
        self.camera.pos = cam.pos;
        self.camera.yaw = cam.yaw;
        self.camera.pitch = cam.pitch;
        self.camera.far = cam.far.max(self.camera.far);
        let n_variant = match riding {
            Some((vp, v)) => self.h4_install_variant(vp, v),
            None => 0,
        };
        Ok(format!("loaded Halo 4 map {} ({draws} draws / {tris} tris, {n_static} scenario objects static, {} palette items{}) in {:.1}s",
            self.cur_stem, self.h4_items.len(), if n_variant > 0 { format!(", {n_variant} variant objects") } else { String::new() }, t0.elapsed().as_secs_f32()))
    }

    /// `loadvariant <halo4 .mvar>`: load its base map when a different one (or none) is open, then
    /// install the variant's records as the editable object set.
    fn h4_apply_variant(&mut self, path: &str) -> Result<String, String> {
        let t0 = Instant::now();
        let vp = std::path::PathBuf::from(path);
        let v = crate::h4::mvar::parse_h4_variant(&vp).map_err(|e| format!("cannot parse Halo 4 variant '{path}': {e}"))?;
        let base_loaded = if self.h4_scene.is_none() || self.map_id != Some(v.map_id) {
            let idx = self.cands.iter().position(|c| c.game == mapcat::Game::Halo4 && c.map_id == Some(v.map_id))
                .ok_or_else(|| format!("no installed Halo 4 base map for variant map id {} ({})", v.map_id, v.title))?;
            let base = self.cands[idx].path.to_string_lossy().into_owned();
            // the variant rides along with the load (the static path then skips its objects)
            self.h4_load_base_map(&base, Some(&vp))?;
            true
        } else {
            self.h4_install_variant(vp, v);
            false
        };
        let (_, v) = self.h4_variant.as_ref().ok_or("variant did not install")?;
        Ok(format!("loaded variant {} ({} forge objects, {} unresolved, {} labels {:?}, budget {}/{}) | base={} | {:.1}ms",
            self.cur_stem, self.objects.len(), self.h4_unresolved.len(), self.h4_labels.len(), self.h4_labels, v.budget_spent, v.budget_max,
            if base_loaded { "loaded" } else { "skip" }, t0.elapsed().as_secs_f64() * 1000.0))
    }

    /// The parsed variant -> `objects` / `colors` / `meta` / `h4_records` (`variant_to_editor`),
    /// the scene's bounds + spawn camera, the header labels. Returns the object count.
    fn h4_install_variant(&mut self, vp: std::path::PathBuf, v: H4Variant) -> usize {
        let Some(es) = self.h4_scene.as_mut() else { return 0 };
        es.set_variant(Some((vp.clone(), v.clone())));
        let set = crate::h4::edit::variant_to_editor(es.cache(), es.palette(), &v);
        self.objects = set.objects;
        self.colors = set.colors;
        self.selected.clear();
        self.meta.clear();
        self.h4_records.clear();
        for o in &self.objects {
            let m = &set.meta[&o.datum];
            self.meta.insert(o.datum, h4_hmeta(&set.src[&o.datum], &m.name, &set.labels));
            self.h4_records.insert(o.datum, set.src[&o.datum].clone());
        }
        self.h4_unresolved = set.unresolved;
        self.h4_labels = set.labels;
        self.seed_globals(v.globals());
        self.h4_variant = Some((vp.clone(), v));
        self.cur_stem = vp.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        self.dirty = true;
        // camera: the variant's loadout camera / initial spawn (spawncam.rs), else frame the objects
        if let Some((pos, yaw, pitch)) = self.h4_scene.as_deref().and_then(|s| s.spawn_camera_pose()) {
            self.camera.pos = pos;
            self.camera.yaw = yaw;
            self.camera.pitch = pitch;
        } else {
            self.frame_forge();
        }
        self.objects.len()
    }

    fn h4_instantiate(&self, quota: u8, variant: u8, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> Result<H4PlacedObject, String> {
        let es = self.h4_scene.as_ref().ok_or("no Halo 4 map loaded")?;
        let pal = self.h4_pal.as_ref().ok_or("no Halo 4 palette")?;
        crate::h4::edit::instantiate_record(es.cache(), pal, quota, (variant > 0).then_some(variant), pos, fwd, up)
            .ok_or_else(|| format!("palette entry {quota}.{variant} does not instantiate"))
    }

    /// `place <#index | name>` over the flat Halo 4 palette: exact raw / display name first
    /// (`sp_respawn_point`, `Respawn Point`), then "entry:variant", then a substring.
    fn h4_resolve_place(&self, obj: &script::ObjRef) -> Result<PlaceTarget, String> {
        let item = match obj {
            script::ObjRef::Index(i) => self.h4_items.get(*i).ok_or_else(|| format!("palette index {i} out of range ({} items)", self.h4_items.len()))?,
            script::ObjRef::Name(n) => {
                let nl = n.to_lowercase();
                let full = |it: &H4PaletteItem| format!("{}:{}", it.name, it.variant_name).to_lowercase();
                self.h4_items.iter().find(|it| it.name.eq_ignore_ascii_case(n) || it.display.eq_ignore_ascii_case(n) || it.variant_name.eq_ignore_ascii_case(n) || it.variant_display.eq_ignore_ascii_case(n))
                    .or_else(|| self.h4_items.iter().find(|it| full(it) == nl))
                    .or_else(|| self.h4_items.iter().find(|it| full(it).contains(&nl) || it.display.to_lowercase().contains(&nl) || it.variant_display.to_lowercase().contains(&nl)))
                    .ok_or_else(|| format!("no Halo 4 palette item matching '{n}' (try `list palette {n}`)"))?
            }
        };
        let obje = item.obje_tag.ok_or_else(|| format!("palette item {} has a null tag", item.name))?;
        Ok(PlaceTarget { mode: item.editor_mode_tag, obj_tag: h4_tag(obje), name: format!("{}:{}", item.name, item.variant_name), variant_sid: 0, h4: Some((item.quota, item.variant)) })
    }

    fn h4_list_palette(&self, filter: Option<&str>) -> String {
        let f = filter.unwrap_or_default().to_lowercase();
        let mut s = String::new();
        let mut n = 0;
        for it in &self.h4_items {
            let line = format!("{}:{} '{}{}' {}", it.name, it.variant_name, it.display, if it.variant_display.is_empty() { String::new() } else { format!(" / {}", it.variant_display) }, it.palette);
            if !f.is_empty() && !line.to_lowercase().contains(&f) { continue; }
            s.push_str(&format!("  #{}  q{}.{}  {line}  type {} price {} max {}{}\n", it.index, it.quota, it.variant, it.object_type, it.price, it.max,
                if it.mode_tag.is_none() { " [marker: no render model]" } else { "" }));
            n += 1;
            if n >= 600 { s.push_str("  ... (truncated)\n"); break; }
        }
        format!("{n} Halo 4 palette items:\n{s}")
    }

    /// `set <datum> <field> <value>` on a Halo 4 record: the shared names keep working (`team`,
    /// `color`, `label`, `shape`, `radius|width|length|top|bottom` in wu, `b0..b3` raw, `respawn`,
    /// `spawnseq`) plus `spawnorder`, `spawntime`, `label2..4`, `traitzone`, `userdata`,
    /// `placement` (raw 10-bit), `placement_hi`, `physics` (normal|fixed|phased), `parent` (slot),
    /// `clips`, `channel`, `passability`, `locked`, and `scale` / `variant_object_scale` (the
    /// record's own scale, 0..10 in the engine's 64 steps; `scale_q` = the raw quantum; `real6` /
    /// `real6q` are accepted as older aliases).
    fn h4_set(&mut self, datum: u32, field: &str, value: &str) -> Result<String, String> {
        use crate::h4::edit::shape_wu_to_raw;
        use crate::h4::mvar::{H4Shape, H4TypeData};
        let name = self.meta.get(&datum).map(|m| m.name.clone()).unwrap_or_default();
        let mut flags = self.meta.get(&datum).map(|m| m.flags).unwrap_or_default();
        let labels = &mut self.h4_labels;
        let r = self.h4_records.get_mut(&datum).ok_or_else(|| format!("no Halo 4 object 0x{datum:08X}"))?;
        let num = |what: &str| value.trim().parse::<f32>().map_err(|_| format!("{what} must be a number"));
        let int = |what: &str, lo: i64, hi: i64| value.trim().parse::<i64>().ok().filter(|v| (lo..=hi).contains(v)).ok_or_else(|| format!("{what} must be {lo}..{hi}"));
        // label by NAME: an existing table entry, else appended (saved as a header edit); `none` clears
        let mut label_index = |v: &str| -> Result<Option<u8>, String> {
            let t = v.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("none") || t == "-" { return Ok(None); }
            if let Some(i) = labels.iter().position(|l| l.eq_ignore_ascii_case(t)) { return Ok(Some(i as u8)); }
            if labels.len() >= 255 { return Err("label table full".into()); }
            labels.push(t.to_string());
            Ok(Some((labels.len() - 1) as u8))
        };
        // shape value slots: sphere [radius]; cylinder [radius, top, bottom]; box [width, length, top, bottom]
        let slot_of = |shape: H4Shape, which: &str| -> Result<usize, String> {
            Ok(match (shape, which) {
                (_, "radius") | (_, "width") => 0,
                (H4Shape::Box, "length") => 1,
                (H4Shape::Cylinder, "top") => 1,
                (H4Shape::Box, "top") => 2,
                (H4Shape::Cylinder, "bottom") => 2,
                (H4Shape::Box, "bottom") => 3,
                _ => return Err(format!("'{which}' does not apply to shape {shape:?} (set shape first)")),
            })
        };
        match field {
            "team" => { let t = script::parse_team(value)?; r.team = if t == 0xFF { -1 } else { t as i8 }; }
            "color" | "colour" => { let c = int("color", -1, 7)?; r.color = (c >= 0).then_some(c as u8); }
            "label" | "label1" => r.labels[0] = label_index(value)?,
            "label2" => r.labels[1] = label_index(value)?,
            "label3" => r.labels[2] = label_index(value)?,
            "label4" => r.labels[3] = label_index(value)?,
            "spawnseq" | "spawn_seq" | "spawnsequence" | "spawn_sequence" | "spawnorder" | "spawn_order" => r.spawn_sequence = (int("spawnseq", -128, 255)? as u8) as i8, // signed byte, -100..100 in game
            "spawntime" | "spawn_time" | "respawn" => r.spawn_time = int("spawntime", 0, 255)? as u8,
            "userdata" | "user_data" => r.user_data = int("userdata", -128, 127)? as i8,
            "traitzone" | "trait_zone" => {
                if r.object_type != 31 { return Err(format!("traitzone applies to trait zones (type 31); this object is type {}", r.object_type)); }
                r.type_data = H4TypeData::TraitZone(int("traitzone", 0, 3)? as u8);
            }
            "clips" | "weapon_clips" | "weaponclips" => match r.type_data {
                H4TypeData::Byte(_) => r.type_data = H4TypeData::Byte(int("clips", 0, 255)? as u8),
                _ => return Err(format!("clips applies to weapons / dominion pads (types 1, 12); this object is type {}", r.object_type)),
            },
            "channel" | "tele_channel" => match r.type_data {
                H4TypeData::Pair(_, b) => r.type_data = H4TypeData::Pair(int("channel", 0, 31)? as u8, b),
                _ => return Err(format!("channel does not apply to type {}", r.object_type)),
            },
            "passability" | "tele_passability" => match r.type_data {
                H4TypeData::Pair(a, _) => r.type_data = H4TypeData::Pair(a, int("passability", 0, 31)? as u8),
                _ => return Err(format!("passability does not apply to type {}", r.object_type)),
            },
            "shape" | "boundary" => {
                r.shape = match value.trim().to_lowercase().as_str() {
                    "none" | "0" => H4Shape::None, "sphere" | "1" => H4Shape::Sphere, "cylinder" | "2" => H4Shape::Cylinder, "box" | "3" => H4Shape::Box,
                    _ => return Err("shape must be none/sphere/cylinder/box".into()),
                };
            }
            "radius" | "width" | "length" | "top" | "bottom" => { let k = slot_of(r.shape, field)?; r.shape_values[k] = shape_wu_to_raw(num(field)?); }
            "b0" => r.shape_values[0] = int("b0", 0, 65535)? as u16,
            "b1" => r.shape_values[1] = int("b1", 0, 65535)? as u16,
            "b2" => r.shape_values[2] = int("b2", 0, 65535)? as u16,
            "b3" => r.shape_values[3] = int("b3", 0, 65535)? as u16,
            "placement" => r.placement = int("placement", 0, 0x3FF)? as u16,
            "placement_hi" | "placementhi" => r.placement = (r.placement & 0xFF) | ((int("placement_hi", 0, 3)? as u16) << 8),
            "physics" => {
                let p = match value.trim().to_lowercase().as_str() { "normal" | "0" => 0, "fixed" | "1" => 1, "phased" | "3" => 3, _ => return Err("physics must be normal/fixed/phased".into()) };
                r.placement = (r.placement & !0xC0) | (p << 6);
            }
            "parent" | "spawnrel" | "spawn_rel" => r.parent = int("parent", -1, 32767)? as i16,
            "locked" => r.locked = forge_scale::parse_flag_value(value)?.unwrap_or(false),
            // the record's object scale (datum +0x2C; the game copies it into the spawned object)
            "scale" | "variant_object_scale" | "real6" => {
                let s = num("scale")?;
                if !(0.0..=10.0).contains(&s) { return Err("scale must be 0..10 (the record stores 64 steps over that range)".into()); }
                r.set_scale(s);
            }
            "scale_q" | "real6q" => r.scale_q = int("scale_q", 0, 63)? as u8,
            "scaled" => flags.scaled = forge_scale::parse_flag_value(value)?,
            "shadow" | "castshadow" | "shadowcaster" => flags.shadow = forge_scale::parse_flag_value(value)?,
            "damage" | "damagestate" | "damage_state" | "state" => flags.damage = forge_scale::parse_damage_value(value)?,
            "cachedtype" | "cached_type" | "type" => return Err("the Halo 4 object type is dictated by the palette entry (read-only)".into()),
            other => return Err(format!("unknown Halo 4 field '{other}'")),
        }
        let rec = r.clone();
        let mut m = h4_hmeta(&rec, &name, &self.h4_labels);
        m.flags = flags;
        self.meta.insert(datum, m);
        self.colors.insert(datum, (rec.team as u8, rec.color.unwrap_or(0xFF)));
        self.dirty = true;
        Ok(format!("set 0x{datum:08X} {field}={value}"))
    }

    /// `dup [datum] [dx dy dz]`: copies (new datums, records with `slot = NEW_SLOT`, same parent /
    /// fields / colours) offset by `off`; the copies become the selection.
    fn h4_dup(&mut self, datum: Option<u32>, off: [f32; 3]) -> Result<String, String> {
        let targets: Vec<u32> = match datum { Some(d) => vec![d], None => self.selected.clone() };
        if targets.is_empty() { return Err("dup: nothing selected".into()); }
        let mut made = Vec::new();
        for d in targets {
            let Some(o) = self.objects.iter().find(|o| o.datum == d).cloned() else { return Err(format!("no object 0x{d:08X}")) };
            let Some(rec) = self.h4_records.get(&d).cloned() else { return Err(format!("0x{d:08X} has no Halo 4 record")) };
            let nd = self.next_datum;
            self.next_datum = self.next_datum.wrapping_add(1);
            let mut no = o;
            no.datum = nd;
            for k in 0..3 { no.pos[k] += off[k]; }
            let mut nr = rec;
            nr.slot = crate::h4::mvar::NEW_SLOT;
            nr.pos = no.pos;
            self.objects.push(no);
            if let Some(m) = self.meta.get(&d).cloned() { self.meta.insert(nd, m); }
            if let Some(c) = self.colors.get(&d).copied() { self.colors.insert(nd, c); }
            self.h4_records.insert(nd, nr);
            made.push(nd);
        }
        self.selected = made.clone();
        self.dirty = true;
        Ok(format!("duplicated {} object(s) -> {}", made.len(), made.iter().map(|d| format!("0x{d:08X}")).collect::<Vec<_>>().join(" ")))
    }

    /// `save [as <path>]` for the open Halo 4 variant: `build_h4_save_list_core` over the edited
    /// records -> gates (out-of-bounds = WARNING + clamped, over budget / past 651 slots = refused,
    /// quota maximum / type = WARNING) -> `h4::mvar::save_objects`. Only the matchmaking folder
    /// (`hopper_map_variants`) is refused: the game never lists a file saved there. After the save the
    /// written file is the open variant and the records carry their final slots, so a second
    /// `save` is a fixed point.
    fn h4_save(&mut self, path: Option<&str>) -> Result<String, String> {
        use crate::h4::edit::{build_h4_save_list_core, clamp_into_bounds, palette_object_type, H4GateInputs};
        let (src, v) = self.h4_variant.clone().ok_or("no Halo 4 variant is open (loadvariant first)")?;
        let dst = match path { Some(p) => std::path::PathBuf::from(p.trim()), None => src.clone() };
        if let Some(dir) = dst.parent().and_then(|d| d.canonicalize().ok()) {
            if dir.file_name().is_some_and(|n| n.eq_ignore_ascii_case("hopper_map_variants")) {
                return Err(format!("hopper_map_variants is the matchmaking folder ({}) - the game never lists a file saved there; save into halo4\\map_variants or LocalFiles\\<xuid>\\Halo4\\Map", dir.display()));
            }
        }
        let es = self.h4_scene.as_ref().ok_or("no Halo 4 map loaded")?;
        let pal = self.h4_pal.as_ref().ok_or("no Halo 4 palette")?;
        let ty = |q: u8, vi: Option<u8>| palette_object_type(pal, q, vi);
        let (list, report) = build_h4_save_list_core(&self.h4_records, &self.objects, &[], &self.h4_unresolved, &HashMap::new(), &self.colors,
            &H4GateInputs { palette: es.palette(), bounds: self.global_edits.bounds, palette_type: Some(&ty) }); // the edited box quantises the positions
        let warnings = report.warnings();
        if report.blocks_save() { return Err(format!("save refused:\n{warnings}")); }
        // a script cannot answer the GUI's "Save anyway" - it saves anyway (the encoder clamps),
        // and the live objects are clamped too so the editor agrees with the file
        let clamped = clamp_into_bounds(&mut self.objects, self.global_edits.bounds);
        // pending `variant set` edits (globals + header strings)
        let (category, budget_max, bounds, quotas) = self.global_edits.diff(&v.globals());
        // `maximum_budget` is kept at or above what the objects cost so the file stays
        // self-consistent; it gates nothing in game (the loader recomputes the spend and restores
        // the map's own sandbox budget), so an over-budget variant is written, not refused.
        let want_max = crate::h4::edit::budget_max_for(budget_max.unwrap_or(v.budget_max), report.budget_spent);
        let budget_max = (want_max != v.budget_max).then_some(want_max);
        let strs = self.global_strings.clone();
        let edits = crate::h4::mvar::H4HeaderEdits {
            labels: (self.h4_labels != v.labels).then(|| self.h4_labels.clone()),
            budget_spent: (report.budget_spent != v.budget_spent).then_some(report.budget_spent),
            title: strs.as_ref().map(|t| t.0.clone()).filter(|t| *t != v.title_key),
            description: strs.as_ref().map(|t| t.1.clone()).filter(|t| *t != v.description_key),
            author: strs.as_ref().map(|t| t.2.clone()).filter(|t| *t != v.author),
            editor: strs.as_ref().map(|t| t.3.clone()).filter(|t| *t != v.editor),
            category,
            budget_max,
            bounds,
            quotas,
            ..Default::default()
        };
        let edits = (!edits.is_empty()).then_some(edits);
        let slots = crate::h4::mvar::save_objects(&src, &dst, &list, edits.as_ref()).map_err(|e| e.to_string())?;
        // bookkeeping: the edited records (with their final slots) are the next save's source
        for (i, o) in self.objects.iter().enumerate() {
            if let (Some(rec), Some(slot)) = (list.get(i), slots.get(i)) {
                let mut r = rec.clone();
                r.slot = *slot;
                self.h4_records.insert(o.datum, r);
            }
        }
        if !clamped.is_empty() { self.dirty = true; }
        let reparsed = crate::h4::mvar::parse_h4_variant(&dst).map_err(|e| format!("saved but cannot re-read {}: {e}", dst.display()))?;
        self.seed_globals(reparsed.globals()); // the file carries the edits from here on
        // #save-precision: the objects adopt exactly what the written file decodes to
        let prec = self.apply_save_precision(&reparsed);
        self.h4_variant = Some((dst.clone(), reparsed));
        self.cur_stem = dst.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        Ok(format!("saved {} objects -> {} ({} unresolved carried, budget {}{}{}){}\n{prec}",
            slots.len(), dst.display(), self.h4_unresolved.len(), report.budget_spent, if edits.is_some() { ", header edited" } else { "" },
            if clamped.is_empty() { String::new() } else { format!(", {} clamped into the bounds", clamped.len()) },
            if warnings.is_empty() { String::new() } else { format!("\n{}", warnings.trim_end()) }))
    }

    /// Seed the global view + edit set from a freshly parsed variant (either game).
    fn seed_globals(&mut self, g: mvar::VariantGlobals) {
        self.global_edits = mvar::GlobalEdits::from_globals(&g);
        self.variant_globals = Some(g);
        self.global_strings = None;
    }

    /// Quota-row names: Halo 4 from the flattened palette, Reach from the type order.
    fn quota_names(&self, h4: bool, n: usize) -> Vec<String> {
        let mut out = vec![String::new(); n];
        if h4 {
            if let Some(pal) = self.h4_pal.as_ref() {
                for (i, e) in pal.entries.iter().enumerate().take(n) {
                    let cat = pal.categories.get(e.category).map(|c| c.display.as_str()).unwrap_or("");
                    out[i] = if cat.is_empty() { e.display.clone() } else { format!("{cat} / {}", e.display) };
                }
            }
        } else {
            for (i, &(pi, ew)) in self.types.iter().enumerate().take(n) {
                if let Some(e) = self.palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew) {
                    out[i] = if e.category_name.is_empty() { e.name.clone() } else { format!("{} / {}", e.category_name, e.name) };
                }
            }
        }
        out
    }

    /// `variant get render-pack <out.json>`: write the open variant's base-map RENDER PACK (see
    /// `render_pack.rs`) built from the user's own installed game. Print-only apart from that one
    /// file; refuses to overwrite nothing it did not just name.
    fn variant_render_pack(&self, out: &str) -> Result<String, String> {
        use crate::render_pack::{build_pack, summary, PackItem};
        if out.is_empty() { return Err("variant get render-pack <out.json>  (writes the local render pack for the open variant's map)".into()); }
        let g = self.variant_globals.as_ref().ok_or("no variant is open (loadvariant first)")?;
        let scene = self.objscene().ok_or("no map is loaded")?;
        let (game, items): (&str, Vec<PackItem>) = if let Some(pal) = self.h4_pal.as_ref() {
            let items = pal.entries.iter().map(|e| {
                let v = e.variants.first();
                PackItem {
                    index: e.quota as usize,
                    name: e.display.clone(),
                    category: pal.categories.get(e.category).map(|c| c.display.clone()).unwrap_or_default(),
                    class: v.map(|v| v.class_str()).unwrap_or_default(),
                    obj_tag: v.and_then(|v| v.tag).map(h4_tag).unwrap_or(0),
                    variants: e.variants.iter().map(|v| (v.index as u32, if v.name.is_empty() { v.display.clone() } else { v.name.clone() }, v.tag.map(h4_tag).unwrap_or(0))).collect(),
                }
            }).collect();
            ("halo4", items)
        } else {
            let items = self.types.iter().enumerate().map(|(i, &(pi, ew))| {
                let e = self.palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew);
                PackItem {
                    index: i,
                    name: e.map(|e| e.name.clone()).unwrap_or_default(),
                    category: e.map(|e| e.category_name.clone()).unwrap_or_default(),
                    class: e.and_then(|e| scene.raw_tag_class(e.tag_short)).unwrap_or_default(),
                    obj_tag: e.map(|e| e.tag_short).unwrap_or(0),
                    variants: self.palette.iter().filter(|p| p.palette_index == pi && p.entry_within == ew).map(|p| (p.variant_within, if p.variant_name.is_empty() { scene.tag_name_of(p.tag_short).rsplit('\\').next().unwrap_or("").to_string() } else { p.variant_name.clone() }, p.tag_short)).collect(),
                }
            }).collect();
            ("reach", items)
        };
        let pack = build_pack(game, g.map_id, g.bounds, scene, &items, &|t| self.scene.as_deref().map(|s| s.render_info_for_pack(t)).unwrap_or_default());
        std::fs::write(out, pack.to_string()).map_err(|e| format!("render-pack: cannot write {out}: {e}"))?;
        Ok(format!("{} -> {out}", summary(&pack)))
    }

    /// `variant get [field]`.
    fn variant_get(&self, field: Option<&str>) -> Result<String, String> {
        let g = self.variant_globals.as_ref().ok_or("no variant is open (loadvariant first)")?;
        if let Some(rest) = field.and_then(|f| f.strip_prefix("render-pack")) {
            return self.variant_render_pack(rest.trim());
        }
        let names = self.quota_names(g.game == "Halo 4", g.quotas.len());
        let (t, d, a, e) = match (&self.global_strings, &self.h4_variant) {
            (Some(s), _) => s.clone(),
            (None, Some((_, v))) => (v.title_key.clone(), v.description_key.clone(), v.author.clone(), v.editor.clone()),
            (None, None) => (String::new(), String::new(), String::new(), String::new()),
        };
        let labels: Vec<String> = if self.h4_variant.is_some() { self.h4_labels.clone() } else { Vec::new() };
        mvar::globals_report(g, &self.global_edits, &names, field, &t, &d, &a, &e, &labels)
    }

    /// `variant set <field> <value>`: applies to a Halo 4 variant (saved by `save`); a Reach
    /// variant's globals are written by the interactive app only, like its objects.
    fn variant_set(&mut self, field: &str, value: &str) -> Result<String, String> {
        let g = self.variant_globals.clone().ok_or("no variant is open (loadvariant first)")?;
        if self.h4_variant.is_none() { return Err("variant set: a Reach variant's globals are written by the interactive app (headless saves Halo 4 only)".into()); }
        let (_, v) = self.h4_variant.as_ref().unwrap();
        let (mut t, mut d, mut a, mut e) = self.global_strings.clone().unwrap_or_else(|| (v.title_key.clone(), v.description_key.clone(), v.author.clone(), v.editor.clone()));
        let (msg, _) = mvar::apply_global_set(&g, &mut self.global_edits, field, value, &mut t, &mut d, &mut a, &mut e)?;
        self.global_strings = Some((t, d, a, e));
        if field.eq_ignore_ascii_case("bounds") {
            let n = crate::h4::edit::out_of_bounds(&self.objects, self.global_edits.bounds).len();
            if n > 0 { return Ok(format!("{msg} WARNING: {n} object(s) lie outside the new box and would be clamped on save")); }
        }
        Ok(msg)
    }

    /// `bspwarn list` on a Halo 4 map: the variant bounds + every object outside them.
    fn h4_bounds_report(&self) -> String {
        let mut out = self.h4_scene.as_deref().map(|s| s.bsp_flags_report()).unwrap_or_default();
        let Some((_, v)) = self.h4_variant.as_ref() else { out.push_str("no variant loaded"); return out };
        let oob = crate::h4::edit::out_of_bounds(&self.objects, v.bounds);
        out.push_str(&format!("{} object(s) outside the variant bounds", oob.len()));
        for d in oob {
            if let Some(o) = self.objects.iter().find(|o| o.datum == d) {
                out.push_str(&format!("\n  0x{d:08X}  {}   ({:.1}, {:.1}, {:.1})", self.meta.get(&d).map(|m| m.name.as_str()).unwrap_or(""), o.pos[0], o.pos[1], o.pos[2]));
            }
        }
        out
    }
}

impl script::ScriptRunner for ScriptHost {
    fn exec_line(&mut self, line: &str) -> String {
        match script::parse_line(line) {
            Ok(None) => String::new(),
            Ok(Some(cmd)) => match self.exec_cmd(cmd) {
                Ok(m) => m,
                Err(e) => format!("ERROR {e}"),
            },
            Err(e) => format!("PARSE {e}"),
        }
    }

    fn enumerate(&mut self, source: &str) -> Result<Vec<script::Item>, String> {
        match script::parse_source(source)? {
            script::Source::Variants { sel, filter } => {
                let (paths, _) = self.variant_listing(&sel, filter.as_deref());
                Ok(paths.into_iter().map(|p| script::Item {
                    stem: p.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
                    value: p.to_string_lossy().into_owned(),
                }).collect())
            }
            script::Source::Maps(filter) => {
                let f = filter.unwrap_or_default().to_lowercase();
                Ok(self.cands.iter().filter(|c| f.is_empty() || c.label().to_lowercase().contains(&f))
                    .map(|c| script::Item { stem: c.stem.clone(), value: c.path.to_string_lossy().into_owned() }).collect())
            }
            script::Source::Objects { type_filter, name_filter, label_filter } => {
                let tf = type_filter.map(|s| s.to_lowercase());
                let nf = name_filter.map(|s| s.to_lowercase());
                let lf = label_filter.map(|s| s.to_lowercase());
                let mut out = Vec::new();
                for o in &self.objects {
                    let m = self.meta.get(&o.datum);
                    let name = m.map(|m| m.name.clone()).unwrap_or_default();
                    let label = m.map(|m| m.label.clone()).unwrap_or_default();
                    let ct = m.map(|m| m.cached_type).unwrap_or(0);
                    if let Some(tf) = &tf {
                        let num_ok = tf.parse::<u8>().map(|v| v == ct).unwrap_or(false);
                        if !num_ok && !name.to_lowercase().contains(tf) { continue; }
                    }
                    if nf.as_ref().map(|nf| !name.to_lowercase().contains(nf)).unwrap_or(false) { continue; }
                    if lf.as_ref().map(|lf| !label.to_lowercase().contains(lf)).unwrap_or(false) { continue; }
                    out.push(script::Item { value: format!("0x{:08X}", o.datum), stem: name });
                }
                Ok(out)
            }
        }
    }
}

fn parse_size(s: Option<&str>) -> (u32, u32) {
    if let Some(s) = s {
        if let Some((a, b)) = s.split_once(['x', 'X']) {
            if let (Ok(w), Ok(h)) = (a.trim().parse(), b.trim().parse()) {
                return (w, h);
            }
        }
    }
    (1600, 900)
}

#[cfg(windows)]
fn hide_console() {
    #[link(name = "kernel32")]
    extern "system" { fn GetConsoleWindow() -> isize; }
    #[link(name = "user32")]
    extern "system" { fn ShowWindow(hwnd: isize, n: i32) -> i32; }
    unsafe {
        let hwnd = GetConsoleWindow();
        if hwnd != 0 {
            ShowWindow(hwnd, 0);
        }
    }
}
#[cfg(not(windows))]
fn hide_console() {}

// =================================================================================================
// #prefab  PREFABS in the headless host
// =================================================================================================
//
// The same prefab core the app uses (`prefab.rs`), over the host's own object lists, so
// `prefab list / show / place / rename / delete / import / export / where` all work from
// `--exec` / `--script` with no window - which is how a prefab placement gets a screenshot and a
// regression test without a human.
//
// WHAT THE HOST CANNOT SEE: the host keeps a thin per-object record for Reach (`HMeta`: team,
// colour, respawn, spawn sequence, label, boundary, the pseudo-flags) because it cannot save a
// Reach variant at all - the placement flags, the teleporter channel and the spare-clip count
// only exist on the interactive app's full `ObjMeta`. So a Reach `prefab save` here says what it
// could not capture instead of pretending. Halo 4 / H2A keep the FULL `.mvar` record
// (`h4_records`), so a Halo 4 capture here is complete.
impl crate::prefab::PrefabScene for ScriptHost {
    fn pf_target(&self) -> Option<(mapcat::Game, String)> {
        if self.map_path.is_empty() {
            return None;
        }
        let mp = std::path::Path::new(&self.map_path);
        let stem = mp.file_stem().map(|s| s.to_string_lossy().to_ascii_lowercase())?;
        let game = self
            .cands
            .iter()
            .find(|c| c.path == mp)
            .map(|c| c.game)
            .unwrap_or(if self.h4_scene.is_some() { mapcat::Game::Halo4 } else { mapcat::Game::Reach });
        Some((game, stem))
    }

    fn pf_palette(&self) -> Vec<crate::prefab::PalRow> {
        use crate::prefab::PalRow;
        if let Some(pal) = self.h4_pal.as_ref() {
            return self
                .h4_items
                .iter()
                .enumerate()
                .map(|(i, it)| {
                    let v = pal.resolve(Some(it.quota), Some(it.variant));
                    let cat = pal.entries.get(it.quota as usize).map(|e| e.category as u32).unwrap_or(0);
                    PalRow {
                        index: i,
                        key: v.map(|(_, v)| crate::xgame::object_key(&v.tag_path, &v.name, v.variant_sid)).unwrap_or_default(),
                        display: if it.variant_display.is_empty() { it.display.clone() } else { format!("{} \u{b7} {}", it.display, it.variant_display) },
                        tag: v.map(|(_, v)| v.tag_path.clone()).unwrap_or_default(),
                        variant: v.map(|(_, v)| v.name.clone()).unwrap_or_default(),
                        variant_sid: v.map(|(_, v)| v.variant_sid).unwrap_or(0),
                        spot: [cat, it.quota as u32, it.variant as u32],
                        price: it.price as f64,
                        max_allowed: it.max,
                    }
                })
                .collect();
        }
        let Some(scene) = self.scene.as_deref() else { return Vec::new() };
        self.place_palette
            .iter()
            .enumerate()
            .map(|(i, (_, obj_tag, name, sid))| {
                let e = self.palette.iter().find(|e| e.tag_short == *obj_tag && e.variant_name_sid == *sid)
                    .or_else(|| self.palette.iter().find(|e| e.tag_short == *obj_tag));
                let tag = scene.tag_name_of(*obj_tag);
                let quota = e
                    .and_then(|e| self.types.iter().position(|&(p, w)| p == e.palette_index && w == e.entry_within))
                    .unwrap_or(0) as u32;
                PalRow {
                    index: i,
                    key: crate::xgame::object_key(&tag, e.map(|e| e.variant_name.as_str()).unwrap_or(""), *sid),
                    display: name.clone(),
                    tag,
                    variant: e.map(|e| e.variant_name.clone()).unwrap_or_default(),
                    variant_sid: *sid,
                    spot: [e.map(|e| e.palette_index).unwrap_or(0), quota, e.map(|e| e.variant_within).unwrap_or(0)],
                    price: 0.0,
                    max_allowed: 0,
                }
            })
            .collect()
    }

    fn pf_capacity(&self) -> crate::prefab::Capacity {
        let pal = <Self as crate::prefab::PrefabScene>::pf_palette(self);
        let mut per_quota = std::collections::HashMap::new();
        let mut spent = 0.0f64;
        for (&d, _) in self.meta.iter() {
            if let Some(row) = self.pf_pal_index(d).and_then(|i| pal.get(i)) {
                *per_quota.entry(row.spot[1]).or_insert(0u32) += 1;
                spent += row.price;
            }
        }
        let priced = self.h4_scene.is_some();
        crate::prefab::Capacity {
            slots_used: self.meta.len() + self.h4_unresolved.len(),
            slots_max: crate::h4::mvar::H4_SLOTS,
            budget_spent: spent,
            budget_max: self.variant_globals.as_ref().map(|g| g.budget_max as f64).unwrap_or(0.0),
            budget_priced: priced,
            per_quota,
        }
    }

    fn pf_selection(&self) -> Vec<u32> {
        self.selected.clone()
    }
    fn pf_set_selection(&mut self, datums: Vec<u32>) {
        self.selected = datums;
    }

    fn pf_read(&self, datum: u32) -> Option<(Option<usize>, crate::prefab::PrefabObject)> {
        let o = self.objects.iter().find(|o| o.datum == datum)?;
        let m = self.meta.get(&datum)?;
        let team_u8 = self.colors.get(&datum).map(|(t, _)| *t).unwrap_or(m.team);
        let rec = self.h4_records.get(&datum);
        let mut labels: Vec<String> = Vec::new();
        if !m.label.is_empty() {
            labels.push(m.label.clone());
        }
        if let Some(r) = rec {
            for l in r.labels.iter().skip(1).flatten() {
                if let Some(s) = self.h4_labels.get(*l as usize).filter(|s| !s.is_empty()) {
                    if !labels.iter().any(|x| x.eq_ignore_ascii_case(s)) {
                        labels.push(s.clone());
                    }
                }
            }
        }
        let shape_wu = match rec {
            Some(r) => r.shape_wu(),
            None => {
                let n = [0usize, 1, 3, 4][(m.boundary_shape as usize).min(3)];
                let mut w = [0.0f32; 4];
                for k in 0..n {
                    w[k] = (m.boundary[k] as f32 / 2047.0) * 200.0;
                }
                w
            }
        };
        let po = crate::prefab::PrefabObject {
            key: String::new(), // `capture` fills it from the palette row
            display: m.name.clone(),
            tag: self.objscene().map(|s| s.tag_name_of(o.primary_tag)).unwrap_or_default(),
            variant: String::new(),
            variant_sid: o.variant_name_sid,
            spot: None,
            pos: o.pos,
            fwd: o.fwd,
            up: o.up,
            team: if team_u8 == 0xFF { -1 } else { team_u8.min(8) as i16 },
            color: (m.color >= 0).then(|| m.color as u8),
            spawn_time: m.respawn,
            spawn_sequence: m.spawn_seq,
            user_data: rec.map(|r| r.user_data as i32).unwrap_or(0),
            placement: rec.map(|r| (r.placement & 0xFF) as u8).unwrap_or(mvar::PLACEMENT_DEFAULT_SPAWNABLE),
            engine_flags: 0,
            placement_hi: rec.map(|r| (r.placement >> 8) as u8).unwrap_or(0),
            shape: m.boundary_shape,
            shape_wu,
            labels,
            tele_channel: 0,
            tele_passability: 0,
            weapon_clips: 0,
            object_type: m.cached_type,
            scale_q: rec.map(|r| r.scale_q).filter(|q| *q != crate::h4::mvar::SCALE_Q_DEFAULT),
            locked: rec.map(|r| r.locked).unwrap_or(false),
            trait_zone: 0,
            scaled: m.flags.scaled,
            shadow: m.flags.shadow,
            damage: m.flags.damage,
        };
        Some((self.pf_pal_index(datum), po))
    }

    fn pf_place(&mut self, pal: usize, o: &crate::prefab::PrefabObject) -> Result<u32, String> {
        let datum = self.next_datum;
        let team_u8: u8 = if o.team < 0 { 0xFF } else { o.team.min(8) as u8 };
        let (mode, obj_tag, name, variant_sid, h4) = match (self.h4_pal.is_some(), self.h4_items.get(pal), self.place_palette.get(pal)) {
            (true, Some(it), _) => (
                it.editor_mode_tag,
                it.obje_tag.map(h4_tag).unwrap_or(0),
                if it.variant_name.is_empty() { it.name.clone() } else { format!("{}:{}", it.name, it.variant_name) },
                0u32,
                Some((it.quota, it.variant)),
            ),
            (false, _, Some(t)) => (t.0, t.1, t.2.clone(), t.3, None),
            _ => return Err(format!("palette row {pal} is out of range")),
        };
        let mut hm = HMeta { name: name.clone(), team: team_u8, color: o.color.map(|c| c as i32).unwrap_or(-1),
            cached_type: o.object_type, spawn_seq: o.spawn_sequence, respawn: o.spawn_time,
            label: o.labels.first().cloned().unwrap_or_default(), boundary_shape: o.shape, boundary: [0; 4],
            flags: forge_scale::ObjFlags { scaled: o.scaled, shadow: o.shadow, damage: o.damage }, h4_scale: None };
        let (mut fwd, mut up) = (o.fwd, o.up);
        if let Some((quota, variant)) = h4 {
            let mut rec = self.h4_instantiate(quota, variant, o.pos, o.fwd, o.up)?;
            rec.team = o.team.clamp(-1, 8) as i8;
            rec.color = o.color;
            rec.spawn_time = o.spawn_time;
            rec.spawn_sequence = o.spawn_sequence.clamp(-100, 100) as i8;
            rec.user_data = o.user_data.clamp(-100, 100) as i8;
            rec.placement = ((o.placement_hi as u16) << 8) | o.placement as u16;
            rec.locked = o.locked;
            if let Some(q) = o.scale_q {
                rec.scale_q = q.min(63);
            }
            rec.shape = match o.shape { 1 => crate::h4::mvar::H4Shape::Sphere, 2 => crate::h4::mvar::H4Shape::Cylinder, 3 => crate::h4::mvar::H4Shape::Box, _ => crate::h4::mvar::H4Shape::None };
            rec.shape_values = [0; 4];
            for k in 0..rec.shape.value_count() {
                rec.shape_values[k] = crate::h4::edit::shape_wu_to_raw(o.shape_wu[k]);
            }
            // Resolve the label indices against a COPY of the table: the closure must not keep a
            // borrow of `self` alive into the `h4_records.insert` below.
            let table = self.h4_labels.clone();
            for k in 0..4 {
                rec.labels[k] = o.labels.get(k).and_then(|s| table.iter().position(|t| t.eq_ignore_ascii_case(s)).map(|i| i as u8));
            }
            fwd = rec.fwd;
            up = rec.up;
            hm = h4_hmeta(&rec, &name, &table);
            hm.flags = forge_scale::ObjFlags { scaled: o.scaled, shadow: o.shadow, damage: o.damage };
            self.h4_records.insert(datum, rec);
        } else {
            let n = [0usize, 1, 3, 4][(o.shape as usize).min(3)];
            for k in 0..n {
                hm.boundary[k] = crate::wu_to_bval(o.shape_wu[k]);
            }
        }
        self.next_datum = self.next_datum.wrapping_add(1);
        self.colors.insert(datum, (team_u8, o.color.unwrap_or(0xFF)));
        self.meta.insert(datum, hm);
        self.objects.push(ObjectInfo {
            datum, type_sig: 0, sig0: 0, sig1: 0, pos: o.pos, health: 1.0, shield: 1.0,
            mode_tag: mode, fwd, up, attached: [0; 8], primary_tag: obj_tag, variant_name_sid: variant_sid,
        });
        self.dirty = true;
        Ok(datum)
    }

    fn pf_remove(&mut self, datums: &[u32]) {
        if datums.is_empty() {
            return;
        }
        self.objects.retain(|o| !datums.contains(&o.datum));
        for d in datums {
            self.meta.remove(d);
            self.colors.remove(d);
            self.h4_records.remove(d);
        }
        self.selected.retain(|d| !datums.contains(d));
        self.dirty = true;
    }

    fn pf_labels(&self) -> Vec<String> {
        if self.h4_scene.is_some() {
            return self.h4_labels.clone();
        }
        // The host does not keep the Reach label table (it cannot save a Reach variant), so a
        // Reach placement here reports every label as dropped rather than inventing an index.
        Vec::new()
    }

    fn pf_drop_point(&self) -> [f32; 3] {
        let fwd = self.camera.forward();
        self.objscene()
            .and_then(|s| s.raycast_scene(self.camera.pos, fwd))
            .unwrap_or(self.camera.pos + fwd * 12.0)
            .to_array()
    }

    fn pf_invalidate(&mut self) {
        self.dirty = true;
        if let Some(s) = self.objscene_mut() {
            s.invalidate();
        }
    }
}

impl ScriptHost {
    /// #prefab Palette row of a placed object: the Halo 4 record's own (quota, variant), else the
    /// Reach object tag + model-variant sid.
    fn pf_pal_index(&self, datum: u32) -> Option<usize> {
        if let Some(r) = self.h4_records.get(&datum) {
            let (q, v) = (r.quota?, r.variant.unwrap_or(0));
            return self.h4_items.iter().position(|it| it.quota == q && it.variant == v);
        }
        let o = self.objects.iter().find(|o| o.datum == datum)?;
        self.place_palette
            .iter()
            .position(|(_, t, _, sid)| *t == o.primary_tag && *sid == o.variant_name_sid)
            .or_else(|| self.place_palette.iter().position(|(_, t, _, _)| *t == o.primary_tag))
    }

    /// #prefab The `prefab …` verb in the headless host.
    fn prefab_command(&mut self, args: &str) -> Result<String, String> {
        let saving = args.trim_start().to_ascii_lowercase().starts_with("save");
        let reach = self.h4_scene.is_none() && self.scene.is_some();
        let out = crate::prefab::command(self, args)?;
        if saving && reach {
            return Ok(format!(
                "{out}\n  ! captured headlessly: the placement flags, teleporter channel and spare-clip count are not tracked \
                 outside the interactive app, so they were left at their defaults. Save the prefab from the app for the full record."
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod film_draw_gates {
    //! #projectile-draw  Does a projectile reach the SCREEN?
    //!
    //! Every earlier projectile gate stopped at "the source was installed" or "the actor was
    //! posed". The owner reported nothing visible, which is exactly what those cannot see. These
    //! gates drive the real script host (the same `film` / `shot` verbs a scripted capture uses and
    //! the same lane order `App::update_actors` runs), render on a real offscreen GPU device and
    //! compare PIXELS: the frame with the projectile lanes installed against the same frame with
    //! them emptied. A render of the same frame twice must be identical (the noise floor), which
    //! is what makes any difference near the projectile evidence of the projectile.
    //!
    //! Map- and GPU-backed, so `#[ignore]`d; run ONE per process:
    //! `cargo test --release -p hms-app film_draw_gates::<name> -- --ignored --nocapture --test-threads=1 --skip sound::probes`
    use super::*;

    /// The least number of pixels near the projectile that must differ for it to count as drawn.
    const MIN_PX: usize = 40;

    /// A host with `forge_halo` open, or None when this machine cannot (no GPU, no map, no DLL).
    fn host() -> Option<ScriptHost> {
        let mut h = ScriptHost::new().ok()?;
        let out = script::run_program(&mut h, "openmap forge_halo");
        if h.scene.is_none() {
            eprintln!("film_draw_gates SKIPPED: forge_halo did not open: {out}");
            return None;
        }
        Some(h)
    }

    /// Every real Reach film recorded on `forge_halo` with `min` projectile tracks, best first.
    fn films_with_projectiles(min: usize) -> Vec<(std::path::PathBuf, usize)> {
        let mut v = Vec::new();
        for p in crate::film::spec_corpus() {
            let Ok(f) = crate::film::open(&p) else { continue };
            if f.header.scenario_name() != "forge_halo" {
                continue;
            }
            let idx = crate::film_decode::build_index_film(&f);
            let n = idx.tracks.iter().filter(|t| t.entity_type == crate::film_projectile::PROJECTILE_TYPE && t.positions.len() >= 3).count();
            if n >= min {
                v.push((p, n));
            }
        }
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// RGBA of the last rendered frame.
    fn last_frame(h: &mut ScriptHost) -> (Vec<u8>, u32, u32) {
        h.renderer.capture_rgba(&h.device, &h.queue)
    }

    /// Pixels whose largest channel differs by more than `thr`, split into those within `r` px
    /// of `(cx, cy)` and the rest.
    fn diff_points(a: &[u8], b: &[u8], w: u32, cx: f32, cy: f32, r: f32, thr: i32) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for (i, (pa, pb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            let d = (0..3).map(|k| (pa[k] as i32 - pb[k] as i32).abs()).max().unwrap();
            if d > thr {
                let (x, y) = ((i as u32 % w) as f32, (i as u32 / w) as f32);
                if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= r {
                    out.push((x, y));
                }
            }
        }
        out
    }

    /// Principal axis of a point cloud: `(angle of the major axis in radians, major/minor spread)`.
    fn principal_axis(pts: &[(f32, f32)]) -> Option<(f32, f32)> {
        if pts.len() < 12 {
            return None;
        }
        let n = pts.len() as f32;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f32>() / n, pts.iter().map(|p| p.1).sum::<f32>() / n);
        let (mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32);
        for p in pts {
            let (dx, dy) = (p.0 - mx, p.1 - my);
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        let tr = sxx + syy;
        let det = sxx * syy - sxy * sxy;
        let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
        let (l1, l2) = (tr / 2.0 + disc, (tr / 2.0 - disc).max(1.0e-6));
        Some((0.5 * (2.0 * sxy).atan2(sxx - syy), (l1 / l2).sqrt()))
    }

    fn diff_split(a: &[u8], b: &[u8], w: u32, cx: f32, cy: f32, r: f32, thr: i32) -> (usize, usize) {
        let (mut near, mut far) = (0usize, 0usize);
        for (i, (pa, pb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            let d = (0..3).map(|k| (pa[k] as i32 - pb[k] as i32).abs()).max().unwrap();
            if d > thr {
                let (x, y) = ((i as u32 % w) as f32, (i as u32 / w) as f32);
                if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= r { near += 1 } else { far += 1 }
            }
        }
        (near, far)
    }

    /// Where `p` lands on the screen of the host's current camera, in pixels.
    fn project(h: &ScriptHost, p: Vec3, w: u32, hh: u32) -> (f32, f32) {
        let c = h.camera.view_proj(w as f32 / hh as f32) * p.extend(1.0);
        ((c.x / c.w * 0.5 + 0.5) * w as f32, (1.0 - (c.y / c.w * 0.5 + 0.5)) * hh as f32)
    }

    /// Empty every projectile lane (what the CONTROL frame is rendered with).
    fn empty_lanes(h: &mut ScriptHost, actor_ids: &[u32]) {
        if let Some(sc) = h.scene.as_mut() {
            sc.set_film_particle_sources(Vec::new());
            sc.set_film_beams(&[]);
            sc.set_film_contrails(Vec::new());
        }
        for &id in actor_ids {
            if let Some(a) = h.actors.get_mut(id) {
                a.pos = Vec3::new(0.0, 0.0, -9000.0);
                a.world_override = None;
            }
        }
        h.actors.update(0.0, &h.queue);
    }

    /// Which of a field's lanes a frame installs.
    #[derive(Clone, Copy, PartialEq)]
    enum Lanes {
        All,
        /// only the render models on the actor lane
        ModelsOnly,
        /// only the impact / detonation bursts
        BurstsOnly,
    }

    /// What one frame did to the picture.
    #[derive(Default)]
    struct Pts(Vec<(f32, f32)>);
    impl std::fmt::Debug for Pts {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} px", self.0.len())
        }
    }

    #[derive(Debug, Default)]
    struct View {
        /// the differing pixels near the focus (for measuring the SHAPE of what was drawn)
        pts: Pts,
        /// pixels that differ within / outside 90 px of the focus between WITH and WITHOUT
        near: usize,
        far: usize,
        /// the same frame rendered twice: must be 0
        noise: usize,
        sources: usize,
        beams: usize,
    }

    /// Render the CURRENT lanes, render again (the noise floor), empty the lanes and render the
    /// control. `focus` is the world point the camera looks at and where "near" is measured.
    fn compare(h: &mut ScriptHost, focus: Vec3, actor_ids: &[u32], out: &std::path::Path, name: &str, radius: f32) -> Option<View> {
        let shot = out.join(format!("{name}.png"));
        h.screenshot(&shot.to_string_lossy(), None).ok()?;
        let (full, w, hh) = last_frame(h);
        let (cx, cy) = project(h, focus, w, hh);
        h.screenshot(&shot.to_string_lossy(), None).ok()?;
        let (again, _, _) = last_frame(h);
        let (noise, _) = diff_split(&full, &again, w, cx, cy, radius, 6);
        let sources = h.scene.as_ref().map_or(0, |s| s.film_particle_source_count());
        let beams = h.scene.as_ref().map_or(0, |s| s.film_beam_count());
        empty_lanes(h, actor_ids);
        h.screenshot(&out.join(format!("{name}_ctl.png")).to_string_lossy(), None).ok()?;
        let (ctl, _, _) = last_frame(h);
        let (near, far) = diff_split(&full, &ctl, w, cx, cy, radius, 6);
        let pts = Pts(diff_points(&full, &ctl, w, cx, cy, radius, 6));
        Some(View { near, far, noise, sources, beams, pts })
    }

    /// The `weap` tag whose leaf is `leaf`.
    fn weapon_named(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, leaf: &str) -> Option<u32> {
        native.tags_by_group(cache, "weap").into_iter().find(|&t| native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf)))
    }

    /// Fire barrel `barrel` of weapon `leaf` from `origin` along `dir` in the REAL scene, fly it
    /// `ticks` ticks through `play_projectile::Field` (the production integrator and the production
    /// `SceneWorld`), draw it through the production lane -- `sync_actors` + `effects` + `beams` +
    /// `contrails` + `film_projectile::install*` + the actor mesh build, in the order
    /// `App::update_actors` runs them -- and compare pixels.
    ///
    /// The scene clock is arranged so the frame is rendered at t = 0 and the shot was launched
    /// `ticks` ticks before it (negative birth times), because a scripted capture runs the particle
    /// lane on `HMS_TIME` and there is no film here.
    fn shoot(h: &mut ScriptHost, leaf: &str, barrel: usize, origin: Vec3, dir: Vec3, ticks: u32, out: &std::path::Path, lanes: Lanes) -> Option<(View, std::sync::Arc<crate::play_projectile::Law>, Vec3)> {
        use crate::play_projectile::{Field, Launch, SceneWorld, TICK};
        let (native, cache) = {
            let sc = h.scene.as_ref()?;
            (sc.native_dll() as *const hms_native::NativeDll, sc.cache_handle()?)
        };
        // SAFETY: the NativeDll is owned by the scene, which outlives this function and is not replaced in it.
        let native: &hms_native::NativeDll = unsafe { &*native };
        let weap = weapon_named(native, cache, leaf)?;
        let proj = crate::reach_projectile::weapon_barrels(native, cache, weap).get(barrel)?.projectile?;
        let mut field = Field::new();
        let law = field.law(native, cache, proj)?;
        let t0 = -(ticks as f32) * TICK;
        field.launch(&law, &[Launch { origin, dir, tracer: true }], t0);
        {
            let sc = h.scene.as_ref()?;
            let world = SceneWorld(sc.as_ref() as &dyn crate::objscene::ObjectScene);
            for k in 0..ticks {
                field.advance(TICK, t0 + (k + 1) as f32 * TICK, &world);
            }
        }
        let focus = field.live.first()?.pos;
        let travel = field.live.first()?.vel.normalize_or_zero();
        draw_field(h, &mut field, native, cache, lanes);
        let side = dir.cross(Vec3::Z).normalize_or_zero();
        let cam = focus + side * 4.0 + Vec3::Z * 0.6;
        script::run_program(h, &format!("camera to {} {} {}\ncamera lookat {} {} {}", cam.x, cam.y, cam.z, focus.x, focus.y, focus.z));
        let v = compare(h, focus, &field.pool_ids(), out, &format!("play_{leaf}{}", match lanes { Lanes::All => "", Lanes::ModelsOnly => "_model", Lanes::BurstsOnly => "_burst" }), 90.0)?;
        Some((v, law, travel))
    }

    /// Install `field`'s lanes exactly as the window's `update_actors` does.
    fn draw_field(h: &mut ScriptHost, field: &mut crate::play_projectile::Field, native: &hms_native::NativeDll, cache: hms_native::CacheHandle, lanes: Lanes) {
        field.sync_actors(native, cache, &mut h.actors);
        let mut fx = field.effects();
        if lanes == Lanes::BurstsOnly {
            fx.retain(|e| e.label.starts_with("impact") || e.label.starts_with("detonation"));
        }
        if let Some(sc) = h.scene.as_mut() {
            // `models_only` leaves the particle / light-volume / contrail lanes empty, so what is
            // measured is the render model on the actor lane and nothing else.
            if lanes == Lanes::ModelsOnly {
                sc.set_film_particle_sources(Vec::new());
                sc.set_film_beams(&[]);
                sc.set_film_contrails(Vec::new());
            } else if lanes == Lanes::BurstsOnly {
                h.film_fx = crate::film_projectile::install(sc, &fx);
                sc.set_film_beams(&[]);
                sc.set_film_contrails(Vec::new());
            } else {
                h.film_fx = crate::film_projectile::install(sc, &fx);
                crate::film_projectile::install_beams(sc, &field.beams());
                crate::film_projectile::install_contrails(sc, field.contrails());
            }
        }
        if h.actors.rebuild {
            let b = crate::reach_actor::build_actor_meshes(&mut h.actors, h.scene.as_mut().unwrap(), h.renderer.mesh_renderer(), &h.device, &h.queue);
            h.install_actor_build(b);
        }
        h.actors.update(0.0, &h.queue);
    }

    /// An airborne point inside the map (the middle of the bounds).
    fn open_air(h: &ScriptHost) -> Vec3 {
        let (lo, hi) = h.scene.as_ref().and_then(|s| s.scene_bounds()).unwrap_or(([0.0; 3], [0.0; 3]));
        Vec3::new((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, lo[2] + (hi[2] - lo[2]) * 0.5)
    }

    /// Projectiles whose tag authors NO visual of their own: no render model, no attached effect
    /// that draws, no light volume, no contrail. PROVEN from the tags (`probe_projectile_table`,
    /// `probe_projectile_attachments_of_every_group`). The engine draws nothing for these in
    /// flight either; their shots are seen through the muzzle flash and the impact. KNOWN ZERO: a
    /// gate that goes red the day one of them gains a visual.
    const AUTHORED_INVISIBLE: &[&str] = &["magnum", "shotgun", "focus_rifle", "spartan_laser"];

    /// Weapons whose projectile must be on screen: model, light volume, contrail or particle effect.
    const MUST_DRAW: &[&str] = &[
        "assault_rifle", "dmr", "sniper_rifle", "plasma_pistol", "plasma_rifle", "plasma_repeater", "needler",
        "rocket_launcher", "grenade_launcher", "concussion_rifle", "plasma_launcher", "spike_rifle",
        "machinegun_turret", "plasma_turret", "scorpion_cannon",
    ];

    /// **TEST MODE: every shot the player can fire is on screen**, or is one of the four whose tag
    /// authors no projectile visual at all.
    #[test]
    #[ignore = "opens the native library, a GPU and forge_halo; run alone"]
    fn test_mode_shots_are_drawn_on_screen() {
        let Some(mut h) = host() else { return };
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        let origin = open_air(&h);
        let dir = Vec3::new(1.0, 0.2, 0.05).normalize();
        let mut drawn = 0usize;
        let mut report = Vec::new();
        for &leaf in MUST_DRAW {
            let (v, law, _) = shoot(&mut h, leaf, 0, origin, dir, 6, &out, Lanes::All).unwrap_or_else(|| panic!("{leaf}: the shot could not be set up"));
            report.push(format!("{leaf}: near {} (noise {}) model {} effects {} beams {} contrails {}", v.near, v.noise, law.has_model(), law.fx.attached.len(), law.fx.beams.len(), law.fx.contrails.len()));
            assert_eq!(v.noise, 0, "{leaf}: the same frame rendered twice differs ({}) -- the comparison is not trustworthy", v.noise);
            assert!(
                v.near >= MIN_PX,
                "{leaf} ({}) is NOT DRAWN: only {} pixels near it differ with its lanes installed (need {MIN_PX}). model {} / {} effect(s) / {} light volume(s) / {} contrail(s); sources {} beams+contrails {}",
                law.fx.name, v.near, law.has_model(), law.fx.attached.len(), law.fx.beams.len(), law.fx.contrails.len(), v.sources, v.beams
            );
            drawn += 1;
        }
        assert_eq!(drawn, MUST_DRAW.len(), "liveness: every listed weapon was shot");
        for &leaf in AUTHORED_INVISIBLE {
            let (v, law, _) = shoot(&mut h, leaf, 0, origin, dir, 6, &out, Lanes::All).unwrap_or_else(|| panic!("{leaf}: the shot could not be set up"));
            assert!(
                !law.has_model() && law.fx.attached.is_empty() && law.fx.beams.is_empty() && law.fx.contrails.is_empty(),
                "{leaf}'s projectile now authors a visual: it belongs in MUST_DRAW"
            );
            assert_eq!(v.near, 0, "{leaf} draws {} pixels but its tag authors no projectile visual", v.near);
        }
        for r in &report {
            eprintln!("  {r}");
        }
    }

    /// **TEST MODE: a projectile's RENDER MODEL** is drawn on the actor lane (the lane Theater draws a
    /// recorded one on), with every particle / light-volume / contrail lane left empty.
    #[test]
    #[ignore = "opens the native library, a GPU and forge_halo; run alone"]
    fn a_projectile_model_is_drawn_on_the_actor_lane() {
        let Some(mut h) = host() else { return };
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        let origin = open_air(&h);
        let dir = Vec3::new(1.0, 0.2, 0.05).normalize();
        let mut seen = 0usize;
        for leaf in ["rocket_launcher", "grenade_launcher", "concussion_rifle", "plasma_launcher", "needler"] {
            let (v, law, _) = shoot(&mut h, leaf, 0, origin, dir, 6, &out, Lanes::ModelsOnly).unwrap_or_else(|| panic!("{leaf}: the shot could not be set up"));
            assert!(law.has_model(), "{leaf}: expected a projectile with a render model");
            eprintln!("  {leaf}: model-only near {} noise {}", v.near, v.noise);
            assert_eq!(v.noise, 0, "{leaf}: render noise");
            if v.near >= MIN_PX {
                seen += 1;
            }
        }
        assert!(seen >= 4, "only {seen} of 5 projectile models reached the screen on the actor lane");
    }

    /// **TEST MODE: a projectile model POINTS ALONG ITS TRAVEL.** The picture, not the matrix: the
    /// model is drawn alone (every other lane empty) from the side, and the principal axis of what it
    /// covers on screen must lie along the screen projection of its velocity -- for two different
    /// flight directions, so a model drawn along a fixed axis cannot pass by luck.
    #[test]
    #[ignore = "opens the native library, a GPU and forge_halo; run alone"]
    fn a_projectile_model_points_along_its_travel_on_screen() {
        let Some(mut h) = host() else { return };
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        let origin = open_air(&h);
        let mut tested = 0usize;
        for leaf in ["rocket_launcher", "needler", "spike_rifle"] {
            for dir in [Vec3::new(1.0, 0.2, 0.05).normalize(), Vec3::new(0.15, 1.0, 0.1).normalize()] {
                let (v, law, travel) = shoot(&mut h, leaf, 0, origin, dir, 6, &out, Lanes::ModelsOnly).unwrap_or_else(|| panic!("{leaf}: no shot"));
                assert!(law.has_model(), "{leaf}");
                let sc_pts = &v.pts.0;
                let Some((axis, elong)) = principal_axis(sc_pts) else { panic!("{leaf}: the model covered only {} px", sc_pts.len()) };
                // screen direction of travel
                let (w, hh) = h.size;
                let focus = h.camera.pos + h.camera.forward() * 4.0; // looking at the focus from 4 wu
                let _ = focus;
                let a = project(&h, h.camera.pos + h.camera.forward() * 4.0, w, hh);
                let b = project(&h, h.camera.pos + h.camera.forward() * 4.0 + travel * 0.5, w, hh);
                let want = (b.1 - a.1).atan2(b.0 - a.0);
                let mut d = (axis - want).abs() % std::f32::consts::PI;
                if d > std::f32::consts::FRAC_PI_2 {
                    d = std::f32::consts::PI - d;
                }
                eprintln!("  {leaf} dir {dir:?}: major axis {:.1} deg, travel {:.1} deg, off by {:.1}, elongation {elong:.2}", axis.to_degrees(), want.to_degrees(), d.to_degrees());
                if elong >= 1.6 {
                    assert!(d.to_degrees() < 20.0, "{leaf} is drawn {:.0} degrees off its direction of travel (elongation {elong:.2})", d.to_degrees());
                    tested += 1;
                }
            }
        }
        assert!(tested >= 4, "liveness: only {tested} elongated model shots could be measured");
    }

    /// **TEST MODE: a hit PLAYS ITS EFFECT ON SCREEN.** The projectile is fired at the real ground of
    /// `forge_halo` and flown through the real collision until it ends; a quarter second later the frame
    /// with the impact / detonation bursts installed is compared with the same frame without them.
    #[test]
    #[ignore = "opens the native library, a GPU and forge_halo; run alone"]
    fn a_hit_plays_its_effect_on_screen() {
        use crate::play_projectile::{Field, Launch, SceneWorld, TICK};
        let Some(mut h) = host() else { return };
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        let sky = open_air(&h);
        // 12 wu above the ground under the middle of the map
        let ground = {
            let sc = h.scene.as_ref().unwrap();
            let t = crate::objscene::ObjectScene::raycast_solid(sc.as_ref(), sky, Vec3::NEG_Z, 5000.0).expect("ground under the map centre");
            sky - Vec3::Z * t
        };
        let origin = ground + Vec3::Z * 12.0;
        let dir = Vec3::new(0.25, 0.1, -1.0).normalize();
        let mut seen = 0usize;
        for leaf in ["rocket_launcher", "concussion_rifle", "assault_rifle", "sniper_rifle", "plasma_rifle"] {
            // how long until it ends, from a throwaway flight in the same world
            let ticks_to_end = {
                let sc = h.scene.as_ref().unwrap();
                let (native, cache) = (sc.native_dll(), sc.cache_handle().unwrap());
                let weap = weapon_named(native, cache, leaf).unwrap();
                let proj = crate::reach_projectile::weapon_barrels(native, cache, weap)[0].projectile.unwrap();
                let mut f = Field::new();
                let law = f.law(native, cache, proj).unwrap();
                f.launch(&law, &[Launch { origin, dir, tracer: true }], 0.0);
                let w = SceneWorld(sc.as_ref() as &dyn crate::objscene::ObjectScene);
                let mut e = None;
                for k in 0..1200u32 {
                    f.tick(k as f32 * TICK, &w);
                    if f.live[0].ended.is_some() {
                        e = Some(k + 1);
                        break;
                    }
                }
                e
            };
            let Some(e) = ticks_to_end else { panic!("{leaf}: never hit the ground it was fired at") };
            let (v, law, _) = shoot(&mut h, leaf, 0, origin, dir, e + 12, &out, Lanes::BurstsOnly).unwrap_or_else(|| panic!("{leaf}: no shot"));
            eprintln!("  {leaf}: ended after {e} ticks; burst near {} noise {} sources {} (impact {:?} material {:?} detonation {:?}); install: {}", v.near, v.noise, v.sources, law.fx.def.impact_effect, law.material_impact, law.fx.def.detonation_effect(false), h.film_fx.line());
            assert_eq!(v.noise, 0, "{leaf}: render noise");
            assert!(v.near >= MIN_PX, "{leaf}: its hit played NOTHING on screen ({} px near the contact)", v.near);
            seen += 1;
        }
        assert_eq!(seen, 5);
    }

    /// **TEST MODE: the shotgun's pellets** come from the `proj`'s `conical spread` (PROVEN,
    /// `projectile_initial_update`): 5 x 3 = 15 pellets per shot, all inside the authored 4 degree cone.
    #[test]
    #[ignore = "opens the native library and forge_halo; run alone"]
    fn the_shotgun_launches_its_authored_pellets() {
        let Some(h) = host() else { return };
        let sc = h.scene.as_ref().unwrap();
        let (native, cache) = (sc.native_dll(), sc.cache_handle().unwrap());
        let weap = weapon_named(native, cache, "shotgun").expect("forge_halo has a shotgun");
        let proj = crate::reach_projectile::weapon_barrels(native, cache, weap)[0].projectile.expect("the shotgun fires a projectile");
        let law = crate::play_projectile::Law::read(native, cache, proj).expect("readable");
        let c = law.extra.conical.expect("the shotgun's projectile authors a conical spread");
        assert_eq!((c.azimuth_strata, c.radial_strata), (5, 3), "the authored strata");
        assert!((c.max_angle - 0.0698).abs() < 1.0e-3, "the authored cone is 4 degrees, got {} rad", c.max_angle);
        let mut f = crate::play_projectile::Field::new();
        let law = std::sync::Arc::new(law);
        f.launch(&law, &[crate::play_projectile::Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
        assert_eq!(f.live.len(), 15, "one shot launches 5 x 3 pellets");
        let widest = f.live.iter().map(|p| p.fwd.dot(Vec3::X).clamp(-1.0, 1.0).acos()).fold(0.0f32, f32::max);
        assert!(widest <= c.max_angle + 1.0e-4 && widest > 0.02, "pellets spread out to {widest} rad of a {} rad cone", c.max_angle);
    }

    /// **TEST MODE: the muzzle flash** -- the barrel's own firing effect played on the gun model's
    /// markers -- is on screen.
    #[test]
    #[ignore = "opens the native library, a GPU and forge_halo; run alone"]
    fn a_muzzle_flash_is_drawn_on_screen() {
        let Some(mut h) = host() else { return };
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        let origin = open_air(&h);
        let (native, cache) = {
            let sc = h.scene.as_ref().unwrap();
            (sc.native_dll() as *const hms_native::NativeDll, sc.cache_handle().unwrap())
        };
        // SAFETY: owned by the scene, which is not replaced below.
        let native: &hms_native::NativeDll = unsafe { &*native };
        let mut seen = 0usize;
        for leaf in ["assault_rifle", "plasma_rifle", "rocket_launcher"] {
            let weap = weapon_named(native, cache, leaf).unwrap();
            let b = crate::reach_projectile::weapon_barrels(native, cache, weap)[0].clone();
            let fx = b.firing_effect.expect("a firing effect");
            let mode = native.resolve_mode(cache, weap);
            let mut field = crate::play_projectile::Field::new();
            field.flash(fx, mode, glam::Mat4::from_translation(origin), 0.0);
            draw_field(&mut h, &mut field, native, cache, Lanes::All);
            let cam = origin + Vec3::new(-1.2, 0.8, 0.3);
            script::run_program(&mut h, &format!("camera to {} {} {}\ncamera lookat {} {} {}", cam.x, cam.y, cam.z, origin.x, origin.y, origin.z));
            let v = compare(&mut h, origin, &[], &out, &format!("flash_{leaf}"), 160.0).expect("frame");
            assert_eq!(v.noise, 0, "{leaf}: render noise");
            eprintln!("  {leaf}: muzzle flash near {} sources {}", v.near, v.sources);
            if v.near >= MIN_PX {
                seen += 1;
            }
        }
        assert!(seen >= 2, "only {seen} of 3 muzzle flashes drew anything: the flash lane is not on screen");
    }

    /// **THEATER: the projectiles the film REPLICATES** (grenades, rockets, missiles, flak) are on
    /// screen mid-flight, one of each kind the clip holds.
    #[test]
    #[ignore = "opens the native library, a GPU, forge_halo and the films on this machine; run alone"]
    fn theater_recorded_projectiles_are_drawn_on_screen() {
        let Some(mut h) = host() else { return };
        let films = films_with_projectiles(40);
        let Some((path, _)) = films.first().cloned() else {
            eprintln!("theater gate SKIPPED: no Reach film with projectiles on this machine");
            return;
        };
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        script::run_program(&mut h, &format!("film {}\nfilm seek 20", path.to_string_lossy()));
        // one mid-flight sample per projectile KIND: the longest-lived track of each that moves
        let mut pick: std::collections::BTreeMap<String, (u32, usize, Vec3, Vec3)> = Default::default();
        {
            let pb = h.film_play.as_ref().unwrap();
            let ot = h.film_driver.objects.as_ref().unwrap();
            for tr in &pb.index.tracks {
                let Some(o) = ot.objects.get(&tr.net_id) else { continue };
                if o.draw != crate::film_objective::Draw::Projectile || tr.positions.len() < 12 {
                    continue;
                }
                let m = tr.positions.len() / 2;
                let d = Vec3::from(tr.positions[m + 1].1) - Vec3::from(tr.positions[m].1);
                if d.length() < 1.0e-3 {
                    continue;
                }
                if pick.get(&o.name).map_or(true, |b| tr.positions.len() > b.1) {
                    pick.insert(o.name.clone(), (tr.positions[m].0, tr.positions.len(), Vec3::from(tr.positions[m].1), d.normalize()));
                }
            }
        }
        assert!(pick.len() >= 3, "liveness: the clip holds {} projectile kinds with a moving track, need >= 3", pick.len());
        let mut drawn = 0usize;
        for (name, (tick, _n, pos, fwd)) in pick {
            let side = fwd.cross(Vec3::Z).normalize_or_zero();
            let cam = pos + side * 5.0 + Vec3::Z * 0.8;
            script::run_program(&mut h, &format!("film seek {}\ncamera to {} {} {}\ncamera lookat {} {} {}", tick as f32 / 60.0, cam.x, cam.y, cam.z, pos.x, pos.y, pos.z));
            let ids: Vec<u32> = h.film_driver.object_actor.values().copied().collect();
            let v = compare(&mut h, pos, &ids, &out, &format!("theater_{name}"), 90.0).expect("frame");
            eprintln!("  {name:<26} near {} noise {} sources {} beams+contrails {}", v.near, v.noise, v.sources, v.beams);
            // A kind whose effect keeps emitter state (flak, frag smoke) does not render the same
            // twice. Its own re-render noise is subtracted: it counts only by the margin it beats it.
            if v.near >= v.noise + MIN_PX {
                drawn += 1;
            }
            script::run_program(&mut h, "film seek 20");
        }
        assert!(drawn >= 3, "only {drawn} of the clip's projectile kinds are on screen mid-flight (beating their own re-render noise by 40 px)");
    }

    /// **THEATER: the rounds the film does NOT replicate** (plasma, rifle, needler ...) are flown from
    /// their fire events and are on screen a few ticks after firing.
    #[test]
    #[ignore = "opens the native library, a GPU, forge_halo and the films on this machine; run alone"]
    fn theater_simulated_shots_are_drawn_on_screen() {
        let Some(mut h) = host() else { return };
        let films = films_with_projectiles(10);
        if films.is_empty() {
            eprintln!("theater gate SKIPPED: no Reach film with projectiles on this machine");
            return;
        }
        let out = std::env::temp_dir().join("hms_proj_gate");
        let _ = std::fs::create_dir_all(&out);
        let (mut drawn, mut tried, mut sources, mut scrubbed) = (0usize, 0usize, std::collections::BTreeSet::new(), false);
        let mut kinds = std::collections::BTreeSet::new();
        for (path, _) in films.iter().take(8) {
            if tried >= 12 {
                break;
            }
            script::run_program(&mut h, &format!("film {}\nfilm seek 20", path.to_string_lossy()));
            // Fire events of bodies whose weapon launches a projectile the film does not carry as an
            // entity (a sword's swing is also an `action_weapon_fire`, and most films are mostly
            // swords or rockets).
            let shooter = |h: &ScriptHost, net: u32, t: u32| -> bool {
                let (Some(pb), Some(pt)) = (h.film_play.as_ref(), h.film_driver.projectiles.as_ref()) else { return false };
                let Some(&ti) = h.film_driver.track_of.get(&net) else { return false };
                let Some(w) = pb.index.tracks.get(ti).and_then(|tr| tr.weapon_at(t)).and_then(|i| h.film_driver.weap_tags.get(i as usize)) else { return false };
                pt.by_weapon.get(w).and_then(|s| s.projectile).map_or(false, |(p, _)| {
                    !h.film_driver.objects.as_ref().map_or(false, |ot| ot.objects.values().any(|o| o.obj_tag == p && o.draw == crate::film_objective::Draw::Projectile))
                })
            };
            let mut events: Vec<(u32, u32)> = Vec::new();
            for (&net, ticks) in &h.film_driver.fire_of {
                let ok: Vec<u32> = ticks.iter().copied().filter(|&t| shooter(&h, net, t)).collect();
                for &t in ok.iter().skip(2).step_by(9).take(3) {
                    events.push((net, t));
                }
            }
            events.sort();
            // spread over every body rather than the first few ids (some bodies hold a sword)
            let step = (events.len() / 24).max(1);
            for (net, ft) in events.into_iter().step_by(step).take(24) {
                script::run_program(&mut h, &format!("film seek {}", (ft + 2) as f32 / 60.0));
                let key = (net, ft);
                let Some(Some(fl)) = h.film_driver.flights.get(&key).cloned() else {
                    eprintln!("  body {net:#x} tick {ft}: no flight ({:?})", h.film_driver.shots_report);
                    continue;
                };
                if fl.pellets[0].path.len() < 3 {
                    continue; // a shot that ended in the shooter's face has no flight to look at
                }
                let pe = &fl.pellets[0];
                let i = 2usize.min(pe.path.len() - 1);
                let (pos, dir) = (pe.path[i], (pe.path[i] - pe.path[0]).normalize_or_zero());
                if dir == Vec3::ZERO {
                    continue;
                }
                let side = dir.cross(Vec3::Z).normalize_or_zero();
                let cam = pos - dir * 3.0 + side * 3.0 + Vec3::Z * 0.5;
                script::run_program(&mut h, &format!("camera to {} {} {}\ncamera lookat {} {} {}", cam.x, cam.y, cam.z, pos.x, pos.y, pos.z));
                let ids = h.film_driver.shots.pool_ids();
                let src = h.film_driver.shot_sources.get(&key).copied();
                let v = compare(&mut h, pos, &ids, &out, &format!("synth_{net:x}_{ft}"), 60.0).expect("frame");
                eprintln!("  body {net:#x} tick {ft} {:<24} aim {src:?} near {} noise {}", fl.law.fx.name, v.near, v.noise);
                // a projectile whose tag authors no visual (magnum, shotgun ...) cannot be seen in flight
                if AUTHORED_INVISIBLE.iter().any(|w| fl.law.fx.name.starts_with(w)) {
                    continue;
                }
                tried += 1;
                if v.near >= v.noise + MIN_PX {
                    drawn += 1;
                    kinds.insert(fl.law.fx.name.clone());
                    if let Some(s) = src {
                        sources.insert(format!("{s:?}"));
                    }
                    // SCRUB: away and back lands on the very same bullet (the flight is a pure function
                    // of the tick, not state integrated frame to frame).
                    if !scrubbed {
                        let here: Vec<Vec3> = h.film_driver.shots.live.iter().map(|p| p.pos).collect();
                        assert!(!here.is_empty(), "liveness: the shot being scrubbed is on screen");
                        script::run_program(&mut h, &format!("film seek {}", (ft + 400) as f32 / 60.0));
                        script::run_program(&mut h, &format!("film seek {}", (ft + 2) as f32 / 60.0));
                        let again: Vec<Vec3> = h.film_driver.shots.live.iter().map(|p| p.pos).collect();
                        assert_eq!(here, again, "scrubbing away and back moved the bullet");
                        scrubbed = true;
                    }
                }
            }
        }
        assert!(tried >= 4, "liveness: only {tried} simulated shots were examined");
        assert!(drawn >= 2 && kinds.len() >= 2, "only {drawn} of {tried} simulated shots ({} kinds: {kinds:?}) are on screen", kinds.len());
        assert!(!sources.is_empty(), "the shots that drew had no recorded aim source");
    }
}

#[cfg(test)]
mod cannon_draw_gates {
    //! #cannon-preview  Does the selected man cannon's arc reach the SCREEN?
    //!
    //! Same method as `film_draw_gates`: the real script host (`cannontraj` verb, the same
    //! `ScriptHost::sync` + lane merge a screenshot runs) renders on an offscreen GPU device and
    //! the frame with the preview on is compared pixel by pixel to the same frame with it off.
    //! Map- and GPU-backed, so `#[ignore]`d; run alone:
    //! `cargo test --release -p hms-app cannon_draw_gates -- --ignored --nocapture --test-threads=1 --skip sound::probes`
    use super::*;

    fn frame(h: &mut ScriptHost, name: &str) -> (Vec<u8>, u32, u32) {
        let shot = std::env::temp_dir().join(format!("hms_cannon_{name}.png"));
        h.screenshot(&shot.to_string_lossy(), None).expect("a screenshot");
        h.renderer.capture_rgba(&h.device, &h.queue)
    }

    fn project(h: &ScriptHost, p: Vec3, w: u32, hh: u32) -> (f32, f32) {
        let c = h.camera.view_proj(w as f32 / hh as f32) * p.extend(1.0);
        ((c.x / c.w * 0.5 + 0.5) * w as f32, (1.0 - (c.y / c.w * 0.5 + 0.5)) * hh as f32)
    }

    /// Pixels that differ by more than `thr`, within `r` px of `(cx, cy)` and in the whole frame.
    fn diff(a: &[u8], b: &[u8], w: u32, cx: f32, cy: f32, r: f32) -> (usize, usize) {
        let (mut near, mut all) = (0, 0);
        for (i, (pa, pb)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
            if (0..3).map(|k| (pa[k] as i32 - pb[k] as i32).abs()).max().unwrap() > 6 {
                all += 1;
                let (x, y) = ((i as u32 % w) as f32, (i as u32 / w) as f32);
                if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() <= r {
                    near += 1;
                }
            }
        }
        (near, all)
    }

    #[test]
    #[ignore = "opens the native library, a GPU and forge_halo; run alone"]
    fn the_selected_cannons_arc_and_landing_marker_are_drawn_on_screen() {
        let mut h = match ScriptHost::new() {
            Ok(h) => h,
            Err(_) => return,
        };
        script::run_program(&mut h, "openmap forge_halo");
        let Some((native, cache)) = h.scene.as_ref().and_then(|s| Some((s.native_dll() as *const hms_native::NativeDll, s.cache_handle()?))) else { return };
        // SAFETY: owned by the scene, which outlives this test body and is not replaced in it.
        let native: &hms_native::NativeDll = unsafe { &*native };
        let tag = native.tags_by_group(cache, "bloc").into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with("ff_man_cannon_forge\\ff_man_cannon_forge"))).expect("the man cannon");
        let site = crate::play_launch::tests::flat_site(native, cache, h.scene.as_ref().unwrap()).expect("a flat site");
        let mut cannon = crate::play_launch::tests::placed(tag, 0x7100, site, 0.0);
        cannon.fwd = [1.0, 0.0, 0.0];
        h.objects.push(cannon);
        h.selected = vec![cannon.datum];
        h.dirty = true;
        // a side-on view of the whole arc (it flies ~15 wu toward -X)
        let mid = site + Vec3::new(-7.0, 0.0, 2.0);
        let cam = mid + Vec3::new(0.0, 22.0, 6.0);
        script::run_program(&mut h, &format!("camera to {} {} {}\ncamera lookat {} {} {}", cam.x, cam.y, cam.z, mid.x, mid.y, mid.z));

        script::run_program(&mut h, "cannontraj off");
        let (off, w, hh) = frame(&mut h, "off");
        let (off2, _, _) = frame(&mut h, "off2");
        let out = script::run_program(&mut h, "cannontraj on");
        eprintln!("{out}");
        let (on, _, _) = frame(&mut h, "on");
        let t = h.cannon_preview.results.first().cloned().expect("liveness: the selected cannon was predicted");
        let landing = t.landing.expect("it lands");
        let (lx, ly) = project(&h, landing, w, hh);
        let (ax, ay) = project(&h, t.apex, w, hh);
        let noise = diff(&off, &off2, w, lx, ly, 40.0).1;
        let (near_land, all) = diff(&off, &on, w, lx, ly, 40.0);
        let (near_apex, _) = diff(&off, &on, w, ax, ay, 40.0);
        eprintln!("CANNON-DRAW noise {noise}, differing px {all}, near landing {near_land}, near apex {near_apex}, landing at px ({lx:.0},{ly:.0})");
        assert_eq!(noise, 0, "the same frame rendered twice differs: the comparison is not trustworthy");
        assert!(all >= 300, "the arc is not on screen: only {all} pixels differ with the preview on");
        assert!(near_land >= 60, "no landing marker where the body lands: {near_land} px near ({lx:.0},{ly:.0})");
        assert!(near_apex >= 8, "the arc is not drawn through its apex: {near_apex} px");
        // selecting nothing removes it again: the frame equals the same nothing-selected frame
        // rendered with the preview switched off
        h.selected.clear();
        let (none_on, _, _) = frame(&mut h, "none_on");
        script::run_program(&mut h, "cannontraj off");
        let (none_off, _, _) = frame(&mut h, "none_off");
        assert_eq!(diff(&none_off, &none_on, w, lx, ly, 40.0).1, 0, "with nothing selected the preview is still on screen");
        assert!(h.cannon_preview.results.is_empty());
    }
}
