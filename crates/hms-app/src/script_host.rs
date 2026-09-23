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
    map_path: String,
    map_id: Option<u32>,

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
        let dll_path = exe_dir.join(crate::PAYLOAD_DLL);
        let mut cam = Camera::default();
        cam.far = 8000.0;
        Ok(Self {
            device,
            queue,
            renderer,
            dll_path,
            cands: mapcat::enumerate(),
            scene: None,
            map_path: String::new(),
            map_id: None,
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
            globals: forge_scale::GlobalFlags::load(),
            screenfx_pushed: None,
            active_screenfx: crate::screenfx::ActiveScreenFx::default(),
        })
    }

    /// Compose the scenario default + placed Forge special-FX screen effects for the current
    /// object set and upload when the set changed (the scenario default is part of it: Zealot's
    /// blue filter belongs in a scripted capture).
    fn refresh_screen_fx(&mut self) {
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

    /// #h4-phys "collision hull: shown (2190 edge(s)) | physics hull: hidden (192 edge(s))" -
    /// the counts come from the SELECTED object through the trait, so a batch run can measure the
    /// two overlays on either game without a window.
    fn hull_overlay_status(&self) -> String {
        let (mut ce, mut pe) = (0usize, 0usize);
        if let (Some(&d), Some(s)) = (self.selected.first(), self.objscene()) {
            if let Some(o) = self.objects.iter().find(|o| o.datum == d) {
                ce = s.collision_world_edges(o).len();
                pe = s.physics_world_edges(o).len();
            }
        }
        let (n, natt) = match (self.selected.first(), self.objscene()) {
            (Some(&d), Some(s)) => s.pick_entry_counts(d),
            _ => (0, 0),
        };
        format!(
            "collision hull: {} ({ce} edge(s)) | physics hull: {} ({pe} edge(s)) | selection geometry: {n} pick entr(ies), {natt} attachment(s)",
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
        let t_all = Instant::now();
        let t_parse0 = Instant::now();
        let variant = mvar::parse_variant(std::path::Path::new(path))
            .ok_or_else(|| format!("cannot parse variant '{path}'"))?;
        let t_parse = t_parse0.elapsed();
        self.seed_globals(variant.globals.clone());
        // The base-map (BSP + palette pre-decode) load is a one-time cost excluded from the
        // between-variant timing: it only fires when the base map id changes. Reported
        // separately ("skip" on reuse).
        let mut t_base = std::time::Duration::ZERO;
        let mut base_loaded = false;
        if self.scene.is_none() || self.map_id != Some(variant.map_id) {
            let idx = self
                .cands
                .iter()
                .position(|c| c.game != mapcat::Game::Halo4 && c.map_id == Some(variant.map_id))
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
        self.cur_stem = std::path::Path::new(path).file_stem().unwrap_or_default().to_string_lossy().into_owned();
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
        scene.set_object_scales(scales);
        scene.set_object_casters(casters);
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

    fn screenshot(&mut self, path: &str, size: Option<(u32, u32)>) -> Result<String, String> {
        let t_all = Instant::now();
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
        let anim_t = std::env::var("HMS_TIME").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
        // Forge lights illuminate: pack the nearest to the camera into SimpleLights at the
        // shot's animation time so flashing lights sample their cycle.
        if let Some(scene) = self.scene.as_ref() {
            if scene.has_forge_lights() {
                let sl = scene.simple_lights_with_forge(self.camera.pos, anim_t);
                self.renderer.set_simple_lights(&self.queue, (sl.len() / 20) as u32, &sl);
            }
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
            "wrote {out} ({rw}x{rh}) | SHOT rebuild={:.1}ms render+capture={:.1}ms png={:.1}ms total={:.1}ms",
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

    /// Gravity settle (headless mirror of App::settle_object): drop `datum` onto the surface
    /// below and tilt it flush to the slope (all four wheels on the ground). Samples the surface
    /// under the footprint, averages the hit normals, aligns up to that normal preserving heading.
    fn settle_one(&mut self, datum: u32) -> bool {
        let Some(i) = self.movable_index(datum) else { return false };
        let pos = Vec3::from(self.objects[i].pos);
        let fwd = Vec3::from(self.objects[i].fwd);
        let result: Option<(Vec3, Vec3, Vec3)> = {
            let Some(scene) = self.objscene() else { return false };
            let (mn, mx) = scene
                .aabb_of(datum)
                .map(|(a, b)| (Vec3::from(a), Vec3::from(b)))
                .unwrap_or((pos - Vec3::ONE, pos + Vec3::ONE));
            let bottom_offset = (pos.z - mn.z).max(0.0);
            let cx = (mn.x + mx.x) * 0.5;
            let cy = (mn.y + mx.y) * 0.5;
            let hx = (mx.x - mn.x) * 0.35;
            let hy = (mx.y - mn.y) * 0.35;
            let samples = [
                (cx, cy),
                (cx - hx, cy - hy),
                (cx + hx, cy - hy),
                (cx - hx, cy + hy),
                (cx + hx, cy + hy),
            ];
            let top_z = mx.z + 2.0;
            let dir = Vec3::NEG_Z;
            let mut center_z = None;
            let mut nsum = Vec3::ZERO;
            let mut nhits = 0;
            for (k, (sx, sy)) in samples.iter().enumerate() {
                let o = Vec3::new(*sx, *sy, top_z);
                let mut best: Option<(Vec3, Vec3)> = None;
                if let Some((p, nn)) = scene.raycast_scene_n(o, dir) {
                    best = Some((p, nn));
                }
                if let Some(p) = scene.raycast_objects_excluding(o, dir, &[datum]) {
                    if best.map(|(bp, _)| p.z > bp.z).unwrap_or(true) {
                        best = Some((p, Vec3::Z));
                    }
                }
                if let Some((p, nn)) = best {
                    let nn = if nn.z < 0.0 { -nn } else { nn };
                    nsum += nn;
                    nhits += 1;
                    if k == 0 {
                        center_z = Some(p.z);
                    }
                }
            }
            if nhits == 0 {
                None
            } else {
                let center_z = center_z.unwrap_or(pos.z);
                let mut up = (nsum / nhits as f32).normalize_or_zero();
                if !up.is_finite() || up.z < 0.2 {
                    up = Vec3::Z;
                }
                let mut f = fwd - up * fwd.dot(up);
                if f.length_squared() < 1e-6 {
                    f = Vec3::X - up * Vec3::X.dot(up);
                    if f.length_squared() < 1e-6 {
                        f = Vec3::Y - up * Vec3::Y.dot(up);
                    }
                }
                Some((Vec3::new(pos.x, pos.y, center_z + bottom_offset), f.normalize_or_zero(), up))
            }
        };
        if let Some((np, f, up)) = result {
            self.objects[i].pos = np.into();
            self.objects[i].fwd = f.into();
            self.objects[i].up = up.into();
            true
        } else {
            false
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
                        if let Some((pos, yaw, pitch)) = self.objscene().and_then(|s| s.spawn_camera_pose()) {
                            self.camera.pos = pos;
                            self.camera.yaw = yaw;
                            self.camera.pitch = pitch;
                        } else {
                            self.frame_scene();
                        }
                    }
                    CameraCmd::Frame => self.frame_forge(),
                    CameraCmd::Nudge(d) => self.camera.pos += Vec3::from(d),
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
                let mut n = 0;
                for d in &targets {
                    if self.settle_one(*d) {
                        n += 1;
                    }
                }
                self.dirty = true;
                Ok(format!("settled {n}/{} object(s)", targets.len()))
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
            // #dialogs The browser is GUI-only; the headless host has no window.
            EditorCommand::DialogDbg { .. } => Ok("dialog: not applicable (headless, no window)".into()),
            // #wire-visible  Run-local; the selection lane is re-applied on every screenshot.
            EditorCommand::WireXray(set) => {
                if let Some(on) = set { self.wire_xray = on; }
                Ok(self.wire_xray_status())
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
        self.renderer.set_overlay_lines(&self.device, &[]);
        self.renderer.set_xray_lines(&self.device, &[]);
        self.renderer.set_exposure(&self.queue, 1.0);
        self.renderer.set_h4_filmic(&self.queue, None);
        self.renderer.set_sun_dir(placeholder_sun_dir());
        self.screenfx_cache.clear();
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
        self.h4_variant = Some((dst.clone(), reparsed));
        self.cur_stem = dst.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        Ok(format!("saved {} objects -> {} ({} unresolved carried, budget {}{}{}){}",
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

    /// `variant get [field]`.
    fn variant_get(&self, field: Option<&str>) -> Result<String, String> {
        let g = self.variant_globals.as_ref().ok_or("no variant is open (loadvariant first)")?;
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
