//! Headless (windowless) offscreen renderer + screenshot: renders one map to a PNG with a
//! scripted camera, no window, so rendering changes can be verified from the pixels.
//!
//! Entered from `main()` when `HMS_SHOT` is set. The documented inputs (site/headless.html):
//! `HMS_SHOT` (output PNG), `HMS_MAP` (stem / substring / path), `HMS_SHOT_SIZE` ("WxH", default
//! 1600x900), `HMS_CAM` ("x,y,z,yawDeg,pitchDeg"), `HMS_VIEW` / `HMS_DIST` / `HMS_FOV`,
//! `HMS_SPAWNCAM`, `HMS_STANDOFF`, `HMS_TIME`, `HMS_MVAR` / `HMS_MVAR_LIST`, `HMS_PLACE` /
//! `HMS_PLACE_COLOR`, the overlay switches and `HMS_PREVIEW_SHOT`. Every other `HMS_*` read here
//! is a print-only development diagnostic (each is described where it is read). A companion
//! "<png>.log" records the scene statistics.

use std::collections::HashMap;

use eframe::wgpu;
use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::NativeDll;
use hms_render::{Camera, SceneRenderer};

use crate::mapcat;
use crate::scene::{self, LoadMsg, SceneController};

/// Entry point. Returns Ok(()) after writing the PNG, or an error string.
pub fn run() -> anyhow::Result<()> {
    // Truly headless: no visible console window. Release builds are already
    // windows_subsystem="windows" (no console at all), but debug / `cargo run` builds
    // attach a console that flashes on screen — hide it immediately in headless mode so a
    // capture shows NO window. stdout still works (the "<png>.log" is written to disk anyway).
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetConsoleWindow() -> isize;
        }
        #[link(name = "user32")]
        extern "system" {
            fn ShowWindow(hwnd: isize, n_cmd_show: i32) -> i32;
        }
        unsafe {
            let hwnd = GetConsoleWindow();
            if hwnd != 0 {
                const SW_HIDE: i32 = 0;
                ShowWindow(hwnd, SW_HIDE);
            }
        }
    }
    let out = std::env::var("HMS_SHOT").unwrap_or_else(|_| "shot.png".into());
    let want = std::env::var("HMS_MAP").unwrap_or_default();
    let (w, h) = parse_size(std::env::var("HMS_SHOT_SIZE").ok().as_deref());
    let mut log = String::new();
    macro_rules! logln { ($($a:tt)*) => {{ let s = format!($($a)*); eprintln!("{s}"); log.push_str(&s); log.push('\n'); }} }

    // Resolve the map from the detected catalog (stem / substring / path).
    let cands = mapcat::enumerate();
    let map_path = resolve_map(&cands, &want)
        .ok_or_else(|| anyhow::anyhow!("HMS_MAP='{want}' not found among {} detected maps", cands.len()))?;
    logln!("headless: map='{}' size={}x{}", map_path, w, h);
    // Halo 4 caches take the pure-Rust path (no native DLL; h4/render.rs). It honours
    // HMS_MVAR=<halo4 .mvar> (+ HMS_MVAR_LIST) like the Reach block below; a Reach .mvar handed
    // to a Halo 4 map is refused there.
    if crate::h4::is_halo4_cache(std::path::Path::new(&map_path)) {
        return crate::h4::render::headless_run(&map_path, &out, w, h);
    }
    // #h2a: Halo 2 Anniversary (groundhog) caches take the same pure-Rust path -- its cache format
    // IS Halo 4's apart from the pointer expander (docs/h2a_support_plan.md).
    if crate::h2a::is_h2a_cache(std::path::Path::new(&map_path)) {
        return crate::h2a::render::headless_run(&map_path, &out, w, h);
    }

    // --- windowless wgpu device (DX12/Vulkan offscreen; no surface) ---
    let instance = wgpu::Instance::default();
    // HMS_SOFTWARE_GPU=1 forces the software rasterizer (lavapipe on Vulkan, WARP on
    // DX12); otherwise prefer a hardware adapter (discrete or integrated) and fall back to the
    // software one automatically when no hardware adapter exists (VMs, servers, no driver).
    let want_soft = std::env::var("HMS_SOFTWARE_GPU").is_ok();
    let request = |fallback: bool| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: fallback,
    }));
    let adapter = if want_soft { request(true) } else { request(false).or_else(|| { logln!("no hardware adapter; trying the software rasterizer"); request(true) }) }
        .ok_or_else(|| anyhow::anyhow!("no wgpu adapter (headless) — no GPU driver and no software Vulkan/DX12 rasterizer (install lavapipe / vulkan-swrast on Linux; WARP is built into Windows)"))?;
    let info = adapter.get_info();
    logln!("adapter: {:?} ({:?}, {:?}, BC={})", info.name, info.device_type, info.backend, adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC));
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("hms-headless"),
            required_features: adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC,
            required_limits: adapter.limits(),
            // Smaller allocator blocks (16-128 MB instead of 128-512 MB) so a single live
            // allocation cannot pin a half-gigabyte host-visible staging chunk after the load.
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))?;

    let mut renderer = SceneRenderer::new(&device, &queue, (w, h));
    renderer.show_grid = false; // no debug grid in screenshots

    // --- load the map via the same worker the live app uses ---
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())).unwrap_or_default();
    let dll_path = exe_dir.join(crate::PAYLOAD_DLL);
    let mut scene = SceneController::new(NativeDll::load(&dll_path)?);
    scene.open_cache(&map_path)?;
    // HMS_PALETTE_DUMP=1|<filter> (print-only): the Forge palette census, every entry (hidden
    // categories included) with its object / model tags and the hlmt model-variant sid (@0x14 of
    // the palette variant), then the per-part decode diagnostics (blend / emissive / forcefield
    // flags) for the entries whose entry, variant or category name matches the filter.
    if let Ok(filter) = std::env::var("HMS_PALETTE_DUMP") {
        let pal = scene.forge_palette_full();
        eprintln!("PALETTE {} entries", pal.len());
        for p in &pal {
            let mode = scene.resolve_object_mode(p.tag_short);
            eprintln!("PALETTE_ENTRY cat='{}' name='{}' variant='{}' vsid={:#x} obj={:#x} mode={:#x} '{}'", p.category_name, p.name, p.variant_name, p.variant_name_sid, p.tag_short, mode, scene.tag_name_of(mode));
            if filter != "1" && !p.name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) && !p.variant_name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) && !format!("{}", p.category_name).to_ascii_lowercase().contains(&filter.to_ascii_lowercase()) { continue; }
            if filter != "1" { scene.dump_object_parts(p.tag_short); }
        }
    }
    // Print-only tag inspection diagnostics: HMS_PPDUMP=<tag hex,...> (post-process tags),
    // HMS_FIND_TAG=<substring> (tag paths), HMS_TAGDUMP=<tag>[:<off>[:<len>]] (raw tag bytes),
    // HMS_PTRDUMP=<ptr>[:<len>] (raw bytes at a tag pointer).
    if let Ok(v) = std::env::var("HMS_PPDUMP") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.dump_postprocess_diag(tag); } } }
    if let Ok(v) = std::env::var("HMS_FIND_TAG") { scene.find_tags_diag(&v); }
    if let Ok(v) = std::env::var("HMS_TAGDUMP") {
        for spec in v.split(',') {
            let p: Vec<u32> = spec.split(':').filter_map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect();
            if let Some(&tag) = p.first() { scene.dump_tag_bytes(tag, p.get(1).copied().unwrap_or(0), p.get(2).copied().unwrap_or(0x200)); }
        }
    }
    if let Ok(v) = std::env::var("HMS_PTRDUMP") {
        for spec in v.split(',') {
            let p: Vec<u32> = spec.split(':').filter_map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect();
            if let Some(&ptr) = p.first() { scene.dump_ptr_bytes(ptr, p.get(1).copied().unwrap_or(0x100)); }
        }
    }
    // HMS_LENS_SCAN=1: every lens-flare tag the map references. HMS_LIFTFX_DIAG=<effe hex,...>:
    // the gravlift light-volume effects.
    if std::env::var("HMS_LENS_SCAN").is_ok() { scene.lens_scan_diag(); }
    if let Ok(v) = std::env::var("HMS_LIFTFX_DIAG") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.lift_fx_diag(tag); } } }
    // HMS_PARTICLE_DIAG=<effe hex,...> dumps an effect's decoded particle systems / emitters / prt3.
    if let Ok(v) = std::env::var("HMS_PARTICLE_DIAG") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.particle_diag(tag); } } }
    // HMS_SCREENFX_DIAG=1 decodes every Forge-palette object's screen effects (obje ->
    // effe -> sefc) plus the scenario default, printing all 240-B fields; =<obj hex,...> for a subset.
    if let Ok(v) = std::env::var("HMS_SCREENFX_DIAG") {
        let mut cache = crate::screenfx::ScreenFxCache::default();
        let objs: Vec<(u32, String)> = if v == "1" {
            scene.forge_palette_full().iter().map(|p| (p.tag_short, format!("{}/{}/{}", p.category_name, p.name, p.variant_name))).collect()
        } else {
            v.split(',').filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok()).map(|t| (t, String::new())).collect()
        };
        if let Some(fx) = scene.screen_fx() {
            for d in cache.defs(&scene, fx.tag) { eprintln!("SCREENFX DEFAULT {}", crate::screenfx::describe(d)); }
        }
        for (obj, label) in objs {
            let tags = cache.sefc_for_object(&scene, obj).to_vec();
            if tags.is_empty() { continue; }
            eprintln!("SCREENFX OBJ {obj:#x} '{}' {label} -> sefc {:x?}", scene.tag_name_of(obj), tags);
            for t in tags { for d in cache.defs(&scene, t) { eprintln!("SCREENFX   {}", crate::screenfx::describe(d)); } }
        }
    }
    // HMS_OBJMETA=<obj hex,...>: an object's decoded obje / hlmt metadata.
    if let Ok(v) = std::env::var("HMS_OBJMETA") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.dump_object_meta(tag); } } }
    // HMS_FORCELOOP_DIAG=1 (print-only) lists every Forge-palette object whose obje attachment
    // effects are force-looping (effe flag bit 8): one-shot effects the particle simulation keeps
    // replaying (the armor-ability pickup icons are the only ones without an object-function
    // scale, i.e. always on).
    if std::env::var("HMS_FORCELOOP_DIAG").is_ok() {
        let mut seen = std::collections::HashSet::new();
        for p in scene.forge_palette_full() {
            if p.tag_short == 0 || p.tag_short == 0xFFFF_FFFF || !seen.insert(p.tag_short) { continue; }
            for (effe, marker, scale_sid) in scene.object_effects(p.tag_short) {
                let systems = scene.effect_particle_systems(effe);
                let Some(fx) = systems.first().map(|s| s.effect.clone()) else { continue };
                if fx.flags & 0x100 == 0 { continue; }
                eprintln!("FORCELOOP obj={:#x} '{}' effe={:#x} '{}' flags={:#x} loop_start={} marker={marker:#x} scale_sid={scale_sid:#x} systems={} persistent={}",
                    p.tag_short, p.name, effe, scene.tag_name_of(effe), fx.flags, fx.loop_start, systems.len(), scale_sid == 0);
            }
        }
    }
    // HMS_REGIONDIAG=<obj hex,...>: render-model regions / permutations.
    if let Ok(v) = std::env::var("HMS_REGIONDIAG") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.dump_model_regions(tag); } } }
    // HMS_VEHMAT=<obj hex,...> dumps every render-model shader (template options, constants, bitmap usages).
    if let Ok(v) = std::env::var("HMS_VEHMAT") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.dump_object_materials(tag); } } }
    // HMS_OBJE_CC=<obj hex,...>: the obje change-colour block. HMS_DUMP_BITMAP=<bitm hex,...>: bitmap headers.
    if let Ok(v) = std::env::var("HMS_OBJE_CC") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.scan_change_colors(tag); } } }
    if let Ok(v) = std::env::var("HMS_DUMP_BITMAP") { for t in v.split(',') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.dump_bitmap_diag(tag); } } }
    // HMS_DUMP_LUT3D=<tag hex>:<out path> writes a 3D bitmap (engine LUT) as a raw u8 volume.
    if let Ok(v) = std::env::var("HMS_DUMP_LUT3D") { if let Some((t, p)) = v.split_once(':') { if let Ok(tag) = u32::from_str_radix(t.trim().trim_start_matches("0x"), 16) { scene.dump_lut3d_diag(tag, p.trim()); } } }
    // HMS_VBPROBE=1: compare the .map's raw compressed vertex buffers against the CPU decode, then exit.
    if std::env::var("HMS_VBPROBE").is_ok() {
        let report = scene.direct_vb_probe();
        eprintln!("{report}");
        let _ = std::fs::write(format!("{out}.log"), &report);
        return Ok(());
    }
    scene.begin_bsp_load();
    if std::env::var("HMS_NOFOG").is_err() {
        renderer.set_fog(&queue, &scene.fog_uniform());
    }
    renderer.set_sky_atmosphere(scene.has_atmosphere());
    renderer.set_space_sky(scene.is_space_sky());
    { let f = scene.fog_uniform(); logln!("has_atmosphere={} space_sky={} fog_sky_tint=[{:.4},{:.4},{:.4}] dens={:.4}", scene.has_atmosphere(), scene.is_space_sky(), f[0], f[1], f[2], f[3]); }

    let mesh_arc = renderer.mesh_renderer_arc();
    let (dv, qv) = (device.clone(), queue.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handle = std::thread::spawn(move || scene::run_load_worker(scene, mesh_arc, dv, qv, tx, cancel));

    let mut done_scene: Option<Box<SceneController>> = None;
    let (mut n_op, mut n_ter, mut n_wat, mut n_at, mut n_fol, mut n_bl, mut n_add, mut n_dec, mut n_sky) =
        (0usize, 0, 0, 0, 0, 0, 0, 0, 0);
    while let Ok(msg) = rx.recv() {
        match msg {
            LoadMsg::Opaque(m) => { n_op += m.len(); renderer.append_static_meshes(m); }
            LoadMsg::Terrain(m) => { n_ter += m.len(); renderer.append_terrain_meshes(m); }
            LoadMsg::Water(m) => { n_wat += m.len(); renderer.append_water_meshes(m); }
            LoadMsg::WaterPlanes(p) => renderer.append_water_planes(p),
            LoadMsg::Alphatest(m) => { n_at += m.len(); renderer.append_alphatest_meshes(m); }
            LoadMsg::Foliage(m) => { n_fol += m.len(); renderer.append_foliage_meshes(m); }
            LoadMsg::Blend(m) => { n_bl += m.len(); renderer.append_blend_meshes(m); }
            LoadMsg::Additive(m) => { n_add += m.len(); renderer.append_additive_meshes(m); }
            LoadMsg::StaticExtra(m) => { n_dec += m.len(); if std::env::var("HMS_NODECALS").is_err() { renderer.append_decal_meshes(m); } }
            LoadMsg::Sky(s) => { n_sky += s.len(); renderer.set_sky_meshes(s); }
            LoadMsg::Progress { .. } => {}
            LoadMsg::Done { scene, exposure } => { renderer.set_exposure(&queue, exposure); done_scene = Some(scene); break; }
        }
    }
    let _ = handle.join();
    let mut scene = *done_scene.ok_or_else(|| anyhow::anyhow!("load worker died before Done"))?;
    logln!("meshes: opaque={n_op} terrain={n_ter} water={n_wat} alphatest={n_at} foliage={n_fol} blend={n_bl} additive={n_add} decal={n_dec} sky={n_sky}");
    // HMS_PATHTRACED=1 replaces the shipped lightmaps with the GPU path-traced bake: bake, fill
    // baked_atlas, clear the BSP-derived meshes, reload (load_step then builds LightmapInputs from
    // the baked atlas). The same flow as the interactive toggle.
    if std::env::var("HMS_PATHTRACED").is_ok() {
        // HMS_BAKE_QUALITY=direct_only|draft|low|medium|high|super_slow (tool.exe's ladder)
        let q = std::env::var("HMS_BAKE_QUALITY").unwrap_or_else(|_| "medium".into());
        let (qs, qb, _, _) = scene::SceneController::bake_quality(&q);
        let samples: u32 = std::env::var("HMS_BAKE_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(qs);
        let bounces: u32 = std::env::var("HMS_BAKE_BOUNCES").ok().and_then(|s| s.parse().ok()).unwrap_or(qb);
        logln!("HMS_PATHTRACED: quality '{q}' -> {samples} samples / {bounces} bounces");
        let k: f32 = std::env::var("HMS_PATHTRACE_K").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let baker = hms_render::lightbake_gpu::GpuLightBaker::new(&device);
        let prog = std::sync::atomic::AtomicU32::new(0);
        scene.set_pathtraced_bake(&baker, &device, &queue, samples, bounces, k, &prog);
        logln!("HMS_PATHTRACED: baked (samples={samples} bounces={bounces} k={k}); reloading BSP with baked atlas");
        renderer.set_static_meshes(Vec::new()); renderer.set_terrain_meshes(Vec::new());
        renderer.set_water_meshes(Vec::new()); renderer.set_alphatest_meshes(Vec::new());
        renderer.set_foliage_meshes(Vec::new()); renderer.set_blend_meshes(Vec::new());
        renderer.set_additive_meshes(Vec::new()); renderer.set_decal_meshes(Vec::new());
        scene.begin_bsp_load();
        let mesh_arc = renderer.mesh_renderer_arc();
        let (dv, qv) = (device.clone(), queue.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let handle = std::thread::spawn(move || scene::run_load_worker(scene, mesh_arc, dv, qv, tx, cancel));
        let mut done2: Option<Box<SceneController>> = None;
        while let Ok(msg) = rx.recv() {
            match msg {
                LoadMsg::Opaque(m) => renderer.append_static_meshes(m),
                LoadMsg::Terrain(m) => renderer.append_terrain_meshes(m),
                LoadMsg::Water(m) => renderer.append_water_meshes(m),
                LoadMsg::WaterPlanes(p) => renderer.append_water_planes(p),
                LoadMsg::Alphatest(m) => renderer.append_alphatest_meshes(m),
                LoadMsg::Foliage(m) => renderer.append_foliage_meshes(m),
                LoadMsg::Blend(m) => renderer.append_blend_meshes(m),
                LoadMsg::Additive(m) => renderer.append_additive_meshes(m),
                LoadMsg::StaticExtra(m) => { if std::env::var("HMS_NODECALS").is_err() { renderer.append_decal_meshes(m); } }
                LoadMsg::Sky(s) => renderer.set_sky_meshes(s),
                LoadMsg::Progress { .. } => {}
                LoadMsg::Done { scene: s, exposure } => { renderer.set_exposure(&queue, exposure); done2 = Some(s); break; }
            }
        }
        let _ = handle.join();
        scene = *done2.ok_or_else(|| anyhow::anyhow!("pathtraced reload died before Done"))?;
        logln!("HMS_PATHTRACED: reload complete");
    }
    // HMS_BAKETEST=1 (print-only): the CPU bake pipeline (geometry retain -> texel raster ->
    // path-trace) with per-submap texel coverage + timing (samples via HMS_BAKE_SAMPLES,
    // bounces via HMS_BAKE_BOUNCES).
    if std::env::var("HMS_BAKETEST").is_ok() {
        let samples: u32 = std::env::var("HMS_BAKE_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let bounces: u32 = std::env::var("HMS_BAKE_BOUNCES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let t0 = std::time::Instant::now();
        let grids = scene.bake_pathtraced_lightmaps(samples, bounces);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut total_texels = 0usize;
        let mut covered = 0usize;
        for (sub, (grid, w, h)) in &grids {
            let cov = grid.iter().filter(|t| t.ambient.length_squared() > 0.0 || t.bandwidth > 0.0).count();
            total_texels += (w * h) as usize;
            covered += cov;
            logln!("  bake submap tag={:#010x} sub={} {}x{} covered={}/{}", sub.0, sub.1, w, h, cov, w * h);
        }
        logln!("BAKETEST (CPU): {} submaps, {} texels ({} covered), samples={} bounces={} in {:.0} ms",
            grids.len(), total_texels, covered, samples, bounces, ms);
    }
    // HMS_BAKETEST_GPU=1 (print-only): the same bake on the GPU compute path.
    if std::env::var("HMS_BAKETEST_GPU").is_ok() {
        let samples: u32 = std::env::var("HMS_BAKE_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let bounces: u32 = std::env::var("HMS_BAKE_BOUNCES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let gpu_baker = hms_render::lightbake_gpu::GpuLightBaker::new(&device);
        let prog = std::sync::atomic::AtomicU32::new(0);
        let (grids, ms) = scene.bake_pathtraced_gpu(&gpu_baker, &device, &queue, samples, bounces, &prog);
        let mut total = 0usize;
        let mut lit = 0usize;
        for (sub, (grid, w, h)) in &grids {
            let l = grid.iter().filter(|o| o.ambient[0] + o.ambient[1] + o.ambient[2] > 0.0).count();
            total += (w * h) as usize; lit += l;
            logln!("  GPU bake submap tag={:#010x} sub={} {}x{} lit={}", sub.0, sub.1, w, h, l);
        }
        logln!("BAKETEST (GPU): {} submaps, {} texels ({} lit), samples={} bounces={} in {:.0} ms",
            grids.len(), total, lit, samples, bounces, ms);
    }
    // HMS_SKYLIGHT_DIAG=1 prints the scenario's sky lights.
    if std::env::var("HMS_SKYLIGHT_DIAG").is_ok() { let _ = scene.sky_lights(); }
    // HMS_BAKE_COMPARE=<dir>|1: GPU bake vs the shipped lightmap atlas, per texel
    // (bake_compare.rs); per-submap PNGs + a report go to the dir ("1" = cwd). Samples / bounces
    // as above.
    if let Ok(v) = std::env::var("HMS_BAKE_COMPARE") {
        let samples: u32 = std::env::var("HMS_BAKE_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let bounces: u32 = std::env::var("HMS_BAKE_BOUNCES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let out_dir = if v == "1" || v.is_empty() { ".".to_string() } else { v };
        let stem = std::path::Path::new(&map_path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "map".into());
        let gpu_baker = hms_render::lightbake_gpu::GpuLightBaker::new(&device);
        let report = crate::bake_compare::run(&scene, &gpu_baker, &device, &queue, samples, bounces, &out_dir, &stem);
        for line in report.lines() { logln!("{line}"); }
        let rp = format!("{}/bake_cmp_{}_{}spp_report.txt", out_dir.trim_end_matches('/'), stem, samples);
        if let Err(e) = std::fs::write(&rp, &report) { logln!("BAKE_COMPARE: report write failed {rp}: {e}"); } else { logln!("BAKE_COMPARE: report written to {rp}"); }
    }
    // HMS_MEMPROF=1 (print-only): GPU memory accounting + process working set (the post-load
    // trim itself lives in main.rs).
    if std::env::var("HMS_MEMPROF").is_ok() {
        let (unc, bc, buf, tc, mc) = hms_render::memprof::report();
        let mb = |b: u64| (b as f64) / (1024.0 * 1024.0);
        logln!(
            "MEMPROF: textures {:.0} MB uncompressed + {:.0} MB BC ({} textures) | mesh buffers {:.0} MB ({} meshes)",
            mb(unc), mb(bc), tc, mb(buf), mc
        );
        // Settled process memory (what Task Manager shows) vs peak (transient decode).
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX, PROCESS_MEMORY_COUNTERS};
            use windows::Win32::System::Threading::GetCurrentProcess;
            let mut c = PROCESS_MEMORY_COUNTERS_EX::default();
            if GetProcessMemoryInfo(GetCurrentProcess(),
                &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32).is_ok() {
                logln!("MEMPROF process: WorkingSet {:.0} MB (peak {:.0}) | PrivateCommit {:.0} MB",
                    mb(c.WorkingSetSize as u64), mb(c.PeakWorkingSetSize as u64), mb(c.PrivateUsage as u64));
            }
            // Return the native CRT heap to the OS, then trim the working set.
            scene.trim_native_heaps();
            use windows::Win32::System::ProcessStatus::EmptyWorkingSet;
            let _ = EmptyWorkingSet(GetCurrentProcess());
            let mut c2 = PROCESS_MEMORY_COUNTERS_EX::default();
            if GetProcessMemoryInfo(GetCurrentProcess(),
                &mut c2 as *mut _ as *mut PROCESS_MEMORY_COUNTERS,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32).is_ok() {
                logln!("MEMPROF after-trim: WorkingSet {:.0} MB | PrivateCommit {:.0} MB",
                    mb(c2.WorkingSetSize as u64), mb(c2.PrivateUsage as u64));
            }
        }
    }

    // The GUI's post-load trim (main.rs `trim_at`): free the native inflate-once page cache (up
    // to 1.5 GB of decompressed pages that only the load needed) and return freed heap to the
    // OS. Done after the optional bake / reload blocks (they re-read the map and benefit from the
    // warm cache), so the steady-state memory numbers below match the interactive app.
    scene.clear_page_cache();
    let t_trim = std::time::Instant::now();
    scene.trim_native_heaps();
    if std::env::var("HMS_MEMDIAG").is_ok() { logln!("HMS_MEMDIAG post-load heap trim took {:.1} ms", t_trim.elapsed().as_secs_f64() * 1000.0); }
    scene.print_mem_diag("post-load-trim");

    // --- post-load setup (mirrors drive_load Done handler) ---
    // The tag-driven auto-exposure band (mirrors main.rs) so headless renders match the
    // interactive app's exposure.
    if let Some((key, lo, hi)) = scene.autoexposure_band() {
        renderer.set_auto_exposure(&queue, key, lo, hi);
        logln!("auto-exposure (cfxs tag): key={key:.3} min_ev={lo:.2} max_ev={hi:.2}");
    }
    renderer.set_illum_exposure(scene.illum_exposure()); // cfxs self-illum P/s
    if let Some((p, sc)) = scene.illum_exposure() { logln!("self-illum exposure (cfxs tag): P={p:.3} s={sc:.3}"); }
    // Mirrors main.rs: gate the glass sun lanes by the map's baked sun-visibility mean.
    {
        let r = scene.baked_sun_reach();
        renderer.set_sun_reach(r);
        logln!("sun reach (baked visibility population): {r:.4}");
    }
    // The map's own bloom (knee + intensity), unconditional, exactly as the GUI does it: headless
    // must render what a double-click renders.
    if let Some((pt, inh, inten)) = scene.bloom_curve() {
        // Its own lane (see SceneRenderer::set_bloom_curve), not the user slider.
        renderer.set_bloom_curve(&queue, pt, inh, inten);
        logln!("bloom (cfxs tag): point={pt:.3} inherent={inh:.3} intensity={inten:.3}");
    }
    if let Some(c) = scene.bloom_colors() {
        renderer.set_bloom_colors(&queue, c);
        logln!("bloom colours (cfxs tag): large={:?} medium={:?} small={:?}", c[0], c[1], c[2]);
    }
    // The scenario default screen effect (every colour term, engine matrix);
    // re-pushed below once the placed objects are known (Forge special-FX orbs add to it).
    let mut screenfx_cache = crate::screenfx::ScreenFxCache::default();
    let forge_fx_enabled = crate::screenfx::env_forge_fx_enabled();
    {
        let default_tag = scene.screen_fx().map(|f| f.tag).unwrap_or(0);
        let active = crate::screenfx::active_screen_fx(&scene, &mut screenfx_cache, default_tag, std::iter::empty());
        logln!("{}", crate::screenfx::push_to_renderer(&renderer, &queue, &active, forge_fx_enabled));
    }
    // Calibration knobs (diagnostics): HMS_AE="key,min_ev,max_ev" overrides the auto-exposure
    // band; HMS_EXPO=<gain> the exposure stops_gain (p.x); HMS_BLOOM=<scale> the bloom strength
    // (1 = engine default, 0 = off); HMS_EXPCAL=<f> the exposure calibration; all for matching
    // engine captures.
    if let Ok(s) = std::env::var("HMS_AE") {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() == 3 { renderer.set_auto_exposure(&queue, p[0], p[1], p[2]); logln!("HMS_AE override: key={} min_ev={} max_ev={}", p[0], p[1], p[2]); }
    }
    if let Some(e) = std::env::var("HMS_EXPO").ok().and_then(|s| s.trim().parse::<f32>().ok()) {
        renderer.set_exposure(&queue, e); logln!("HMS_EXPO override: stops_gain={e}");
    }
    if let Some(b) = std::env::var("HMS_BLOOM").ok().and_then(|s| s.trim().parse::<f32>().ok()) {
        renderer.set_bloom(&queue, b); logln!("HMS_BLOOM override: bloom_scale={b}");
    }
    if let Some(c) = std::env::var("HMS_EXPCAL").ok().and_then(|s| s.trim().parse::<f32>().ok()) {
        renderer.set_exp_cal(&queue, c); logln!("HMS_EXPCAL override: exp_cal={c}");
    }
    // HMS_ENGINE_DECODE=1 (diagnostic): replace the auto-exposure meter with the engine's fixed
    // g_exposure scalar (Sapien 0.125); HMS_FIXED_EXPOSURE=<f> overrides the value.
    if std::env::var("HMS_ENGINE_DECODE").is_ok() {
        let g = std::env::var("HMS_FIXED_EXPOSURE").ok().and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(0.125);
        renderer.set_exposure(&queue, 1.0);      // neutralize p.x (fixed gain replaces stops×meter)
        renderer.set_fixed_exposure(&queue, g);
        logln!("HMS_ENGINE_DECODE: fixed g_exposure={g}");
    }
    // HMS_DUMPDATA=<path.json>: dump the scene-global lighting inputs (sun / ambient / fog /
    // exposure / per-BSP K / scnr lights) for the HMS <-> Sapien parity diff.
    if let Ok(dp) = std::env::var("HMS_DUMPDATA") {
        let json = scene.dump_scene_data_json();
        match std::fs::write(&dp, &json) {
            Ok(()) => logln!("HMS_DUMPDATA: wrote scene-global lighting data to {dp}"),
            Err(e) => logln!("HMS_DUMPDATA: write failed for {dp}: {e}"),
        }
    }
    // HMS_DUMPONE=<bitm hex,...>: dump those bitmaps to <diag dir>/tex_<tag>.png with alpha (to
    // inspect what a material's textures contain), print their channel means, then exit.
    if let Ok(tags) = std::env::var("HMS_DUMPONE") {
        for t in tags.split(',') {
            let ts = t.trim().trim_start_matches("0x");
            if let Ok(tag) = u32::from_str_radix(ts, 16) {
                match scene.decode_bitmap_keep_alpha_public(tag) {
                    Some((mut px, w, h)) => {
                        for p in px.chunks_exact_mut(4) { p.swap(0, 2); } // BGRA -> RGBA
                        let path = format!("{}/tex_{:x}.png", diag_dir().display(), tag);
                        let _ = write_png(&path, &px, w, h);
                        // channel means to see where content lives
                        let (mut r,mut g,mut b,mut a,mut n)=(0u64,0u64,0u64,0u64,0u64);
                        for q in px.chunks_exact(4){ r+=q[0] as u64;g+=q[1] as u64;b+=q[2] as u64;a+=q[3] as u64;n+=1; }
                        let n=n.max(1);
                        logln!("dumped tex {:x} {}x{} -> {} meanRGBA=({},{},{},{})", tag,w,h,path, r/n,g/n,b/n,a/n);
                    }
                    None => logln!("decode {:x} FAILED", tag),
                }
            }
        }
        return Ok(());
    }
    // HMS_PROBEDIAG=1 (print-only): the airprobe ambient at a fixed set of Ivory Tower points
    // (interior vs exterior spatial variation).
    if std::env::var("HMS_PROBEDIAG").is_ok() {
        for &(x,y,z) in &[(-20.0f32,-175.0f32,243.0f32),(-30.0,-190.0,242.0),(0.0,-160.0,244.0),(-45.0,-150.0,245.0),(20.0,-185.0,243.0),(-15.0,-175.0,252.0),(-60.0,-200.0,240.0),(40.0,-140.0,246.0)] {
            let a = scene.debug_probe([x,y,z]);
            logln!("PROBE ({:.0},{:.0},{:.0}) ambient=[{:.3},{:.3},{:.3}] luma={:.3}", x,y,z, a[0],a[1],a[2], 0.299*a[0]+0.587*a[1]+0.114*a[2]);
        }
    }
    let (st, at) = scene.scene_tints();
    renderer.set_scene_tints(st, at);
    // HMS_SUN_MULT / HMS_AMB_MULT / HMS_LM_MULT (diagnostics): the Lighting Lab multipliers, so
    // the interactive slider findings can be reproduced and measured.
    {
        let f = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.trim().parse::<f32>().ok()).unwrap_or(d);
        renderer.set_lighting_mults(f("HMS_SUN_MULT", 1.0), f("HMS_AMB_MULT", 1.0), f("HMS_LM_MULT", 1.0));
    }
    renderer.set_obj_dir_strength(scene.obj_dir_strength()); // per-map object soft-shade
    logln!("tints: sun=[{:.2},{:.2},{:.2}] ambient=[{:.2},{:.2},{:.2}]", st[0], st[1], st[2], at[0], at[1], at[2]);
    // The atmosphere fog sky tint + the baked ambient at the scene centre (which channel
    // carries a map's colour cast).
    let fg = scene.fog_uniform();
    logln!("fog sky tint=[{:.3},{:.3},{:.3}] density={:.3}", fg[0], fg[1], fg[2], fg[3]);
    logln!("fog GROUND color=[{:.3},{:.3},{:.3}] thickness={:.4} height={:.3} base={:.3} maxdist={:.3}",
        fg[8], fg[9], fg[10], fg[11], fg[12], fg[13], fg[14]);
    if let Some((mn, mx)) = scene.scene_bounds() {
        logln!("scene bounds: min=[{:.1},{:.1},{:.1}] max=[{:.1},{:.1},{:.1}]", mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
        let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
        let amb = scene.debug_probe(c);
        logln!("baked ambient @center=[{:.3},{:.3},{:.3}]", amb[0], amb[1], amb[2]);
    }
    // HMS_SUN="x,y,z" (diagnostic) overrides the probe-derived sun direction, for A/B tests
    // against a captured engine directional (the probe dom_dir average is near-vertical on
    // enclosed maps; the engine's ivory_tower directional is (0.2985,0.5385,0.7880)).
    let sun_override = std::env::var("HMS_SUN").ok().and_then(|s| {
        let v: Vec<f32> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if v.len() == 3 { Some(Vec3::new(v[0], v[1], v[2])) } else { None }
    });
    if let Some(sun) = sun_override {
        renderer.set_sun_dir(sun);
        logln!("sun dir (HMS_SUN): [{:.3},{:.3},{:.3}]", sun.x, sun.y, sun.z);
    } else if let Some(sun) = scene.scene_sun_dir() {
        renderer.set_sun_dir(Vec3::from(sun));
        logln!("sun dir: [{:.3},{:.3},{:.3}]", sun[0], sun[1], sun[2]);
    } else {
        logln!("sun dir: fallback (no probe)");
    }
    let sl = scene.simple_lights();
    renderer.set_simple_lights(&queue, (sl.len() / 20) as u32, &sl);

    logln!("scnr tag id: {:#x}", scene.scnr_tag());
    // Does the scenario author a sun light at all?
    match scene.scenario_sun() {
        Some((i, t)) => logln!("scenario sun ligh: intensity={i:.3} type={t}"),
        None => logln!("scenario sun ligh: NONE (map authors no sun light)"),
    }
    {
        let (h, mean, n) = scene.atlas_vis_hist();
        let pv = crate::lightprobe::vis_hist_peek();
        logln!("atlas sun-vis (DM.g) at {n} verts: mean={mean:.3} bins8={h:?}; PVL 2-bit pop={pv:?}");
    }
    // The map's authored outdoor sun + ambient (sky light).
    match scene.sky_light() {
        Some(s) => logln!("sky light: has_light={} dir=[{:.3},{:.3},{:.3}] rgb=[{:.3},{:.3},{:.3}] ambient=[{:.3},{:.3},{:.3}]",
            { let h = s.has_light; h }, s.dir[0], s.dir[1], s.dir[2], s.color[0], s.color[1], s.color[2], s.ambient[0], s.ambient[1], s.ambient[2]),
        None => logln!("sky light: (none resolved)"),
    }

    // Effect scenery billboards (additive), same skip-on-undecodable rule as the app.
    let fx = scene.effect_scenery_billboards();
    let mut fx_kept = 0usize;
    if !fx.is_empty() {
        let mut fx_meshes = Vec::new();
        {
            let mr = renderer.mesh_renderer();
            for (verts, indices, bitm) in &fx {
                let tv = if *bitm != 0 && *bitm != 0xFFFF_FFFF {
                    scene.decode_bitmap_public(*bitm).map(|t| hms_render::upload_texture_bgra_nomip(&device, &queue, &t.0, t.1, t.2))
                } else { None };
                let Some((view, _tex)) = tv else { continue };
                fx_meshes.push(mr.upload_mesh(&device, &queue, verts, indices, &[glam::Mat4::IDENTITY], None, Some(&view), None, None, None, None, [0.0, 0.0], [1.0, 1.0], [0.0, 0.0, 0.0, 0.0], 0.0, 1.0, [0.0, -1.0], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None, None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None));
                fx_kept += 1;
            }
        }
        renderer.append_additive_meshes(fx_meshes);
    }
    logln!("effect scenery: {} placements, {} drawn", fx.len(), fx_kept);
    // BSP-placed effect_scenery light volumes (Sword Base oni gravlifts etc.) -> additive
    // camera-facing ribbons (bump_xform.w = 4 lane, see scene.effect_scenery_light_volumes).
    let lv = scene.effect_scenery_light_volumes();
    if !lv.is_empty() {
        let mut lv_meshes = Vec::with_capacity(lv.len());
        {
            let mr = renderer.mesh_renderer();
            let (white, _wt) = hms_render::upload_texture_bgra_nomip(&device, &queue, &[255, 255, 255, 255], 1, 1);
            for (verts, indices, fxc) in &lv {
                lv_meshes.push(mr.upload_mesh(&device, &queue, verts, indices, &[glam::Mat4::IDENTITY], None, Some(&white), Some(&white), None, None, None, [0.0, 0.0], [1.0, 1.0], [0.0; 4], 0.0, 0.0, [0.0, -1.0], [0.0; 4], [0.0, 0.0, 0.0, 4.0], *fxc, [0.0; 4], [0.0; 4], None, false, 0.0, [0.0; 4], None, None, [0.0; 4], [0.0; 4], None, None, [0.0; 4], [0.0; 4], None));
            }
        }
        logln!("effect scenery light volumes: {}", lv_meshes.len());
        renderer.append_additive_meshes(lv_meshes);
    }

    // HMS_SCNRFLAGS=1|list (print-only): the scenario placement-flags distribution (how many
    // objects are "not automatically" / "never placed").
    if std::env::var("HMS_SCNRFLAGS").is_ok() {
        let raw = scene.scenario_objects();
        let mut hist: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
        let (mut na, mut np) = (0u32, 0u32);
        for o in &raw {
            *hist.entry(o.placement_flags).or_default() += 1;
            if o.placement_flags & 0x1 != 0 { na += 1; }
            if o.placement_flags & 0x40 != 0 { np += 1; }
        }
        logln!("SCNRFLAGS: total={} not_auto(bit0)={} never_placed(bit6)={}", raw.len(), na, np);
        for (f, c) in hist.iter().take(24) {
            logln!("  flags=0x{f:08X}  count={c}");
        }
        // HMS_SCNRFLAGS=list also prints every placement with its obj tag path,
        // forge-owned / spawn-family classification and position (which built-in spawns a map ships).
        if std::env::var("HMS_SCNRFLAGS").map(|v| v == "list").unwrap_or(false) {
            let owned = scene.forge_owned_tags();
            for o in &raw {
                let nm = scene.tag_name_of(o.obj_tag);
                logln!("SCNROBJ obj={:#x} mode={:#x} cat={} flags={:#04x} owned={} spawn={} pos=({:.1},{:.1},{:.1}) '{}'",
                    o.obj_tag, o.mode_tag, o.category, o.placement_flags, owned.contains(&o.obj_tag) as u8,
                    crate::map_spawns::is_map_spawn_name(&nm) as u8, o.pos[0], o.pos[1], o.pos[2], nm);
            }
        }
    }

    // Scenario objects (trees / props / etc.) -> maybe_rebuild -> dynamic / cutout / holo passes.
    // Placements of objects in the map's Forge sandbox palette are canvas / default-variant
    // content the map variant owns, so they are never placed or tracked from the scenario (the
    // same filter as main.rs). HMS_SCNR_FORGE=1 (diagnostic) places them anyway.
    let forge_owned: std::collections::HashSet<u32> = if std::env::var("HMS_SCNR_FORGE").is_ok() {
        Default::default()
    } else {
        scene.forge_owned_tags()
    };
    let scnr_all = scene.scenario_objects();
    let scnr_total = scnr_all.len();
    // The scenario's own spawn-family markers (respawn_point_invisible etc.) are hidden unless
    // HMS_MAP_SPAWNS=1, the same default the GUI's View > "Show map spawn points" has.
    let show_map_spawns = crate::map_spawns::env_show_map_spawns();
    let spawn_tags = crate::map_spawns::spawn_obj_tags(scnr_all.iter().map(|o| o.obj_tag), |t| scene.tag_name_of(t));
    let mut spawns_seen = 0usize;
    let obj_skip: std::collections::HashSet<u32> = std::env::var("HMS_OBJ_SKIP").ok().map(|v| v.split(',').filter_map(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default();
    let mut objects: Vec<ObjectInfo> = scnr_all
        .into_iter()
        .filter(|o| o.category != 8)
        // Skip "not automatically" (bit0) / "never placed" (bit6) placements
        // (HMS_SHOW_ALL_SCNR=1, diagnostic, keeps them).
        .filter(|o| (o.placement_flags & 0x41) == 0 || std::env::var("HMS_SHOW_ALL_SCNR").is_ok())
        .filter(|o| !forge_owned.contains(&o.obj_tag))
        // HMS_OBJ_SKIP=<obj hex,...> (diagnostic) leaves those scenario objects out of the shot,
        // so a suspect surface can be attributed to an object vs the BSP from the pixels.
        .filter(|o| !obj_skip.contains(&o.obj_tag))
        .filter(|o| {
            let spawn = spawn_tags.contains(&o.obj_tag);
            if spawn { spawns_seen += 1; }
            show_map_spawns || !spawn
        })
        .enumerate()
        .map(|(i, o)| ObjectInfo {
            datum: 0xE000_0000u32.wrapping_add(i as u32),
            type_sig: 0, sig0: 0, sig1: 0,
            pos: o.pos, health: 1.0, shield: 1.0,
            mode_tag: o.mode_tag, fwd: o.fwd, up: o.up,
            attached: [0; 8], primary_tag: o.obj_tag,
            variant_name_sid: 0,
        })
        .collect();
    logln!("scenario objects: {} (of {scnr_total} raw; forge-palette placements are owned by the map variant and skipped)", objects.len());
    logln!("{}", crate::map_spawns::status_line(show_map_spawns, spawns_seen));
    // HMS_MVAR=<path>: place the variant's forge objects, so the headless render matches what the
    // interactive app shows with that variant open.
    if let Ok(mvar_path) = std::env::var("HMS_MVAR") {
        if let Some(variant) = crate::mvar::parse_variant(std::path::Path::new(&mvar_path)) {
            let palette = scene.forge_palette_full();
            let types = scene.forge_type_order(&palette);
            let mut placed = 0usize;
            let mut unresolved_list: Vec<(usize, u16, u8)> = Vec::new();
            // The same scale / shadow-caster derivation the GUI applies to a variant
            // (SCALED <- "scale" label; SHADOW <- GREEN + "scale"), under the globals (persisted
            // settings, HMS_SCALED / HMS_SHADOWCASTERS on top). Headless has no per-object overrides.
            let globals = crate::forge_scale::GlobalFlags::load();
            let mut scales: HashMap<u32, f32> = HashMap::new();
            let mut casters: std::collections::HashSet<u32> = Default::default();
            for (i, o) in variant.objects.iter().enumerate() {
                let (mode_tag, obj_tag, vsid) = scene.resolve_forge_model_v(&palette, &types, o.folder, o.item);
                if mode_tag == 0 || mode_tag == 0xFFFF_FFFF {
                    // These still occupy a slot in the variant. Report what they are.
                    unresolved_list.push((i, o.folder, o.item));
                    continue;
                }
                if std::env::var("HMS_MVAR_LIST").is_ok() {
                    let nm = types.get(o.folder as usize).and_then(|&(pi, ew)| {
                        palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew && e.variant_within == o.item as u32)
                            .or_else(|| palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew))
                            .map(|e| if e.variant_name.trim().is_empty() { e.name.clone() } else { e.variant_name.clone() })
                    }).unwrap_or_default();
                    eprintln!("OBJ slot={:3} f={:3} i={:2} pos=({:8.2},{:8.2},{:8.2}) team={} {}",
                        o.slot, o.folder, o.item, o.pos[0], o.pos[1], o.pos[2], o.team, nm);
                }
                // HMS_MVAR_DIAG=1 (print-only): the resolved model / shadow-caster flag per object.
                if std::env::var("HMS_MVAR_DIAG").is_ok() {
                    eprintln!("MVAR_OBJ mode={:#x} obj={:#x} casts={} name='{}' pos=({:.1},{:.1},{:.1})", mode_tag, obj_tag, scene.object_casts_shadow(obj_tag), scene.tag_name_of(mode_tag), o.pos[0], o.pos[1], o.pos[2]);
                }
                let datum = 0xF000_0000u32.wrapping_add(i as u32);
                objects.push(ObjectInfo {
                    datum,
                    type_sig: 0, sig0: 0, sig1: 0,
                    pos: o.pos, health: 1.0, shield: 1.0,
                    mode_tag, fwd: o.fwd, up: o.up,
                    attached: [0; 8], primary_tag: obj_tag, variant_name_sid: vsid,
                });
                placed += 1;
                {
                    let label = if o.label_idx != 0xFFFF { variant.labels.get(o.label_idx as usize).cloned().unwrap_or_default() } else { String::new() };
                    let (fs, fc) = crate::forge_scale::effective_flags(&globals, &Default::default(), o.team, &label);
                    if fs {
                        let sc = crate::forge_scale::object_scale(o.spawn_seq, o.team).clamp(0.01, crate::forge_scale::object_max_scale(o.team));
                        if (sc - 1.0).abs() > 1e-3 { scales.insert(datum, sc); }
                    }
                    if fc { casters.insert(datum); }
                }
            }
            logln!("HMS_MVAR flags: {} | {} scaled, {} casting", globals.status_line(), scales.len(), casters.len());
            scene.set_object_scales(scales);
            scene.set_object_casters(casters);
            // Same as the GUI's variant open: decode the variant's models on a background
            // thread; the per-frame rebuild takes delivery (objects pop in over the decode).
            scene.start_background_predecode(&objects.iter().map(|o| o.mode_tag).collect::<Vec<u32>>());
            logln!("HMS_MVAR: {} variant objects, {} placed (total objects now {})", variant.objects.len(), placed, objects.len());
            logln!("HMS_MVAR: quota count in file = {}, distinct palette types derived = {}", variant.num_quotas, types.len());
            // How many of the variant's objects sit in a BSP flagged "not normally
            // playable space in MP" (the save-time warning list). Every record counts, resolved or not.
            {
                let outside: Vec<(usize, usize)> = variant.objects.iter().enumerate()
                    .filter_map(|(i, o)| scene.non_playable_bsp_at(Vec3::from(o.pos)).map(|b| (i, b)))
                    .collect();
                logln!("HMS_MVAR: {} objects outside playable space (non-playable structure BSPs: {})",
                    outside.len(),
                    scene.structure_bsp_flags().iter().filter(|b| b.non_playable()).map(|b| b.index.to_string()).collect::<Vec<_>>().join(","));
                for &(i, b) in outside.iter().take(40) {
                    let o = &variant.objects[i];
                    logln!("   slot#{i} f={} i={} pos=({:.1},{:.1},{:.1}) -> BSP {b}", o.folder, o.item, o.pos[0], o.pos[1], o.pos[2]);
                }
            }
            if !unresolved_list.is_empty() {
                logln!("HMS_MVAR: {} UNRESOLVED objects (occupy slots, not displayed):", unresolved_list.len());
                for &(i, folder, item) in unresolved_list.iter().take(40) {
                    let group = types.get(folder as usize).and_then(|&(pi, ew)| {
                        palette.iter().find(|e| e.palette_index == pi && e.entry_within == ew).map(|e| e.name.clone())
                    }).unwrap_or_else(|| "<folder past the end of this map's palette>".into());
                    logln!("   slot#{i} folder={folder} item={item}  {group}");
                }
            }

        } else {
            logln!("HMS_MVAR: parse_variant FAILED for {mvar_path}");
        }
    }
    // HMS_PLACE=<obj hex>[,<obj hex>...]@x,y,z : place Forge palette objects (by OBJECT tag id) in a row along +x
    // starting at x,y,z (3 wu apart) — lets a headless render show palette items without a saved variant.
    // Several `ids@x,y,z` groups may be joined with ';' (each group starts its own row; the datum
    // counter keeps running so every placed object stays unique) — e.g. a glass pane at one spot
    // and a marker deliberately in FRONT of it: HMS_PLACE="2144@14,5,9;90@14,2,9".
    if let Ok(spec) = std::env::var("HMS_PLACE") {
        let mut k = 0usize;
        for group in spec.split(';') {
            let Some((ids, at)) = group.split_once('@') else { continue };
            let p: Vec<f32> = at.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            if p.len() == 3 {
                let mut n = 0usize;
                for id in ids.split(',') {
                    // `<obj hex>[:<variant sid hex>]`: the optional sid is the hlmt model-variant
                    // Name the Forge palette entry (@0x14) / .mvar placement would carry (e.g. 69d:6c0 = the
                    // rocket warthog), so variant selection can be rendered without a saved variant.
                    let (id, vs) = id.trim().split_once(':').map_or((id.trim(), ""), |(a, b)| (a, b.trim()));
                    let Ok(obj) = u32::from_str_radix(id.trim_start_matches("0x"), 16) else { continue };
                    let vsid = if vs.is_empty() { 0 } else { u32::from_str_radix(vs.trim_start_matches("0x"), 16).unwrap_or(0) };
                    let mode = scene.resolve_object_mode(obj);
                    if mode == 0 || mode == 0xFFFF_FFFF { logln!("HMS_PLACE: no model for {obj:#x}"); continue; }
                    objects.push(ObjectInfo {
                        datum: 0xF100_0000u32.wrapping_add(k as u32), type_sig: 0, sig0: 0, sig1: 0,
                        pos: [p[0] + 3.0 * n as f32, p[1], p[2]], health: 1.0, shield: 1.0,
                        mode_tag: mode, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
                        attached: [0; 8], primary_tag: obj, variant_name_sid: vsid,
                    });
                    k += 1;
                    n += 1;
                }
                logln!("HMS_PLACE: placed {n} objects at {:?}", p);
            }
        }
    }
    // Playable-area centroid from object positions (robust to distant skybox geometry that
    // blows up scene_bounds). Used as the default camera target so shots land on the map,
    // not 2500u away in empty space. Also log a few positions so specific spots can be framed.
    let obj_centroid: Option<(Vec3, f32)> = if !objects.is_empty() {
        let n = objects.len() as f32;
        let sum = objects.iter().fold(Vec3::ZERO, |a, o| a + Vec3::from(o.pos));
        let c = sum / n;
        let spread = (objects.iter().map(|o| (Vec3::from(o.pos) - c).length_squared()).sum::<f32>() / n).sqrt();
        for o in objects.iter().take(6) {
            log.push_str(&format!("  obj pos=[{:.1},{:.1},{:.1}]\n", o.pos[0], o.pos[1], o.pos[2]));
        }
        Some((c, spread.max(5.0)))
    } else {
        None
    };
    for (verts, _i, _b) in fx.iter().take(6) {
        if let Some(v0) = verts.first() {
            log.push_str(&format!("  efsc near=[{:.1},{:.1},{:.1}]\n", v0.pos[0], v0.pos[1], v0.pos[2]));
        }
    }
    if let Some((c, r)) = obj_centroid {
        logln!("object centroid=[{:.1},{:.1},{:.1}] spread={:.1}", c.x, c.y, c.z, r);
    }
    // HMS_PLACE_COLOR=<team>,<color>: give every HMS_PLACE'd object this forge (team, colour) pair
    // (change-colour lanes / team tint), to reproduce coloured-object rendering headlessly.
    let mut empty: HashMap<u32, (u8, u8)> = HashMap::new();
    if let Ok(pc) = std::env::var("HMS_PLACE_COLOR") {
        let v: Vec<u8> = pc.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 2 {
            for o in objects.iter().filter(|o| o.datum & 0xFF00_0000 == 0xF100_0000) { empty.insert(o.datum, (v[0], v[1])); }
            logln!("HMS_PLACE_COLOR: team={} color={} on {} placed objects", v[0], v[1], empty.len());
        }
    }
    // Compose the placed Forge special-FX objects' screen effects (dedup + per-term max
    // with the scenario default) exactly like the interactive app; HMS_FORGE_FX=0 renders the
    // default alone for batch screenshots.
    {
        let default_tag = scene.screen_fx().map(|f| f.tag).unwrap_or(0);
        let active = crate::screenfx::active_screen_fx(&scene, &mut screenfx_cache, default_tag, objects.iter().map(|o| (o.datum, o.primary_tag)));
        logln!("{}", crate::screenfx::push_to_renderer(&renderer, &queue, &active, forge_fx_enabled));
    }
    // maybe_rebuild works under a per-frame time budget (interactive smoothness), so it may
    // leave models un-decoded and set rebuild_pending. Headless wants the complete set in one
    // shot: loop until it settles (each call returns the full current set; the last is complete).
    let mut obj_meshes = None;
    for _ in 0..4096 {
        obj_meshes = scene.maybe_rebuild(&objects, &empty, renderer.mesh_renderer(), &device, &queue).or(obj_meshes);
        if !scene.rebuild_pending() { break; }
    }
    if let Some((op, cut, holo, holo_solid, blend)) = obj_meshes {
        logln!("object meshes: opaque={} cutout={} holo={} holo_solid={} blend={}", op.len(), cut.len(), holo.len(), holo_solid.len(), blend.len());
        renderer.set_dynamic_meshes(op);
        renderer.set_dynamic_cutout_meshes(cut);
        renderer.set_dynamic_holo_meshes(holo);
        renderer.set_dynamic_holo_solid_meshes(holo_solid);
        renderer.set_dynamic_blend_meshes(blend);
    }
    // The phmo overlay (solid translucent + outline) for hidden forge blocks: shown unless
    // HMS_PHYSICS_OUTLINES=0, the same default the GUI's View > "Show hidden-block physics
    // hulls" has. Only variant / user-placed objects ever get a hull.
    {
        let show = crate::physics_outlines::env_show_physics_outlines();
        let (btris, blines): (Vec<([f32;3],[f32;4])>, Vec<([f32;3],[f32;3],[f32;3])>) = {
            let (t, l) = scene.blocker_overlay();
            (t.to_vec(), l.to_vec())
        };
        logln!("{} (HMS_PHYSICS_OUTLINES; hull lines={})", crate::physics_outlines::status_line(show, btris.len() / 3), blines.len());
        let (mut ztris, segs) = if show { (btris, blines) } else { (Vec::new(), Vec::new()) };
        // HMS_SOFT_CEILINGS=1 draws the structure-design soft ceilings (kill floor /
        // acceleration / slip planes), the GUI's View > "Soft ceilings" lane; always logs the list.
        {
            let show_sc = crate::soft_ceilings::env_show_soft_ceilings();
            let cs = scene.soft_ceilings();
            logln!("{} (HMS_SOFT_CEILINGS)", crate::soft_ceilings::status_line(show_sc, &cs));
            for l in crate::soft_ceilings::listing(&cs) { logln!("{l}"); }
            let mut xray = Vec::new();
            if show_sc {
                let (t, l) = crate::soft_ceilings::overlay_geometry(&cs);
                ztris.extend(t);
                xray.extend(l);
            }
            // HMS_HARD_FLOOR=1 draws the playable BSPs' world bounds + floor plane.
            let show_hf = crate::hard_floor::env_show_hard_floor();
            let bsps = scene.structure_bsp_flags();
            let mopp = scene.structure_bsp_mopp_bounds();
            let wb = crate::hard_floor::world_box(bsps, &mopp);
            logln!("{} (HMS_HARD_FLOOR)", crate::hard_floor::status_line(show_hf, wb.as_ref()));
            for l in crate::hard_floor::listing(bsps, &mopp) { logln!("{l}"); }
            if let (true, Some(wb)) = (show_hf, wb.as_ref()) {
                let (t, l) = crate::hard_floor::world_geometry(wb, 16);
                ztris.extend(t);
                xray.extend(l);
            }
            // HMS_PLAYABLE_BOUNDS=1 draws the playable BSPs' own boxes (violet).
            let show_pb = crate::hard_floor::env_show_playable_bounds();
            logln!("{} (HMS_PLAYABLE_BOUNDS)", crate::hard_floor::playable_status_line(show_pb, bsps));
            if show_pb {
                let (t, l) = crate::hard_floor::overlay_geometry(bsps, 8);
                ztris.extend(t);
                xray.extend(l);
            }
            if !xray.is_empty() { renderer.set_xray_lines(&device, &xray); }
        }
        if !ztris.is_empty() { renderer.set_zone_tris(&device, Some(&ztris)); }
        if !segs.is_empty() { renderer.set_overlay_lines(&device, &segs); }
    }
    // The authored planar fog volumes through the same screen-space path as the interactive
    // app (main.rs load_map), so headless renders match what the user sees.
    {
        let fog_vols = scene.planar_fog_volumes();
        if !fog_vols.is_empty() {
            logln!("planar fog volumes: {}", fog_vols.len());
            renderer.set_planar_fog_volumes(fog_vols);
        }
    }

    // --- camera ---
    let mut cam = build_camera(&scene, obj_centroid);
    logln!("camera: pos=[{:.1},{:.1},{:.1}] yaw={:.1} pitch={:.1}", cam.pos.x, cam.pos.y, cam.pos.z, cam.yaw.to_degrees(), cam.pitch.to_degrees());

    // HMS_VISHIST=1 (print-only): the airprobe sky-mask histogram and, below, the raw 2-bit
    // per-vertex sun-visibility population.
    if std::env::var("HMS_VISHIST").is_ok() {
        let (n, mean, lo, hi) = scene.airprobe_mask_hist();
        logln!("HMS_VISHIST airprobes: n={n} mean_mask={mean:.3} (<0.05: {lo:.1}%, >0.95: {hi:.1}%)");
    }
    // HMS_LBSPDUMP="<hdr_off>,<elem_size>[,<count>[,<lbsp_tag>]]": dump one Lbsp block's raw elements.
    if let Ok(v) = std::env::var("HMS_LBSPDUMP") {
        let p: Vec<usize> = v.split(',').filter_map(|t| {
            let t = t.trim();
            if let Some(h) = t.strip_prefix("0x") { usize::from_str_radix(h, 16).ok() } else { t.parse().ok() }
        }).collect();
        if p.len() >= 2 {
            // "<hdr_off>,<elem>[,<count>[,<lbsp_tag>]]"; lbsp_tag omitted/0 = every bsp's Lbsp.
            for line in scene.lbsp_block_dump(*p.get(3).unwrap_or(&0) as u32, p[0], p[1], *p.get(2).unwrap_or(&16)) {
                logln!("HMS_LBSPDUMP {line}");
            }
        }
    }
    // HMS_TAGMETA="<tag>,<off>,<len>": raw tag meta bytes.
    if let Ok(v) = std::env::var("HMS_TAGMETA") {
        let p: Vec<u64> = v.split(',').filter_map(|t| {
            let t = t.trim();
            if let Some(h) = t.strip_prefix("0x") { u64::from_str_radix(h, 16).ok() } else { t.parse().ok() }
        }).collect();
        if p.len() >= 3 {
            for line in scene.tag_meta_dump(p[0] as u32, p[1] as u32, p[2] as u32) {
                logln!("HMS_TAGMETA {line}");
            }
        }
    }
    // HMS_PTRDUMP="<ptr>,<len>[,<elem>]": raw bytes at a tag pointer, after the load (the
    // pre-load form above takes `<ptr>:<len>`).
    if let Ok(v) = std::env::var("HMS_PTRDUMP") {
        let p: Vec<u64> = v.split(',').filter_map(|t| {
            let t = t.trim();
            if let Some(h) = t.strip_prefix("0x") { u64::from_str_radix(h, 16).ok() } else { t.parse().ok() }
        }).collect();
        if p.len() >= 2 {
            for line in scene.tag_ptr_dump(p[0] as u32, p[1] as u32, *p.get(2).unwrap_or(&16) as usize) {
                logln!("HMS_PTRDUMP {line}");
            }
        }
    }
    // HMS_LBSPSCAN=1: every Lbsp block header.
    if std::env::var("HMS_LBSPSCAN").is_ok() {
        for line in scene.lbsp_block_scan() { logln!("HMS_LBSPSCAN {line}"); }
    }
    if std::env::var("HMS_VISHIST").is_ok() {
        let h = crate::lightprobe::vis_hist_take();
        let n: u64 = h.iter().sum();
        if n > 0 {
            logln!("HMS_VISHIST: n={n}  0(occluded)={:.1}%  1={:.1}%  2={:.1}%  3(full sun)={:.1}%",
                100.0 * h[0] as f64 / n as f64, 100.0 * h[1] as f64 / n as f64,
                100.0 * h[2] as f64 / n as f64, 100.0 * h[3] as f64 / n as f64);
        }
    }
    // HMS_STANDOFF=<wu> pushes the camera clear of solid geometry before rendering, through the
    // same solver the interactive app uses.
    if let Ok(v) = std::env::var("HMS_STANDOFF") {
        let want: f32 = v.trim().parse().unwrap_or(2.0);
        let start = cam.pos;
        let (before, _, blocked0) = scene.nearest_solid(start, want);
        cam.pos = scene.standoff(start, want);
        let (after, _, blocked1) = scene.nearest_solid(cam.pos, want);
        logln!("HMS_STANDOFF: want {want:.2} wu | before nearest={:?} blocked={blocked0}/64 | moved {:.2} wu to ({:.2},{:.2},{:.2}) | after nearest={:?} blocked={blocked1}/64",
            before.map(|t| (t * 100.0).round() / 100.0), (cam.pos - start).length(),
            cam.pos.x, cam.pos.y, cam.pos.z,
            after.map(|t| (t * 100.0).round() / 100.0));
    }

    // HMS_OBJLIGHT_SCAN="x0,y0,x1,y1,step,z" (print-only): ground height / material / lightmap
    // sample under every grid point (the crate ray cast straight down from z).
    if let Ok(s) = std::env::var("HMS_OBJLIGHT_SCAN") {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() == 6 { scene.obj_light_scan(p[0], p[1], p[2], p[3], p[4], p[5]); }
        else { logln!("HMS_OBJLIGHT_SCAN: expected x0,y0,x1,y1,step,z"); }
    }
    // HMS_OBJLIGHT_AT="x,y,z[,type[,radius]];..." (print-only): the full engine object-lighting sample at points.
    if let Ok(s) = std::env::var("HMS_OBJLIGHT_AT") { scene.obj_light_at(&s); }
    // HMS_LIGHTMAP_SCAN="x0,y0,x1,y1,step[,radius]" (diagnostic): the probe-grid light map, what a
    // Forge piece resting on the ground at each grid cell would be shaded with (engine
    // object-lighting sample + the mesh_shade merge for a flat top face). Writes <HMS_SHOT stem>_lightscan.png
    // (top-down, +x right, +y up, grid every 25 wu, legend) and <stem>_lightscan.csv.
    // HMS_LIGHTBAKE_EXPO=<gain> fixes the tonemap exposure for every light-bake output (default: auto key).
    if let Ok(s) = std::env::var("HMS_LIGHTMAP_SCAN") {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() >= 5 {
            let radius = p.get(5).copied().unwrap_or(0.4);
            let mut scan = scene.light_bake_scan(p[0], p[1], p[2], p[3], p[4], radius);
            if let Some(e) = std::env::var("HMS_LIGHTBAKE_EXPO").ok().and_then(|v| v.parse::<f32>().ok()) { if e > 0.0 { scan.exposure = e; } }
            let kinds = { let mut k = [0usize; 6]; for c in scan.cells.iter().filter(|c| c.valid) { k[(c.kind as usize).min(5)] += 1; } k };
            logln!("LIGHTMAP_SCAN {}x{} cells step={} radius={:.2} valid={} ({} lightmap, {} pvl, {} pvl-colour, {} probe, {} airprobe) exposure={:.4} sun=({:.2},{:.2},{:.2}) in {:.0} ms",
                scan.nx, scan.ny, scan.step, scan.radius, scan.valid_count(), kinds[1], kinds[2], kinds[3], kinds[4], kinds[5], scan.exposure, scan.sun_rgb[0], scan.sun_rgb[1], scan.sun_rgb[2], scan.elapsed_ms);
            let stem = out.trim_end_matches(".png").to_string();
            let img = scan.render_image(0, 25.0);
            match write_png(&format!("{stem}_lightscan.png"), &img.rgba, img.w as u32, img.h as u32) {
                Ok(()) => logln!("LIGHTMAP_SCAN wrote {stem}_lightscan.png ({}x{}, {} px/cell)", img.w, img.h, img.scale),
                Err(e) => logln!("LIGHTMAP_SCAN png failed: {e}"),
            }
            match std::fs::write(format!("{stem}_lightscan.csv"), scan.to_csv()) {
                Ok(()) => logln!("LIGHTMAP_SCAN wrote {stem}_lightscan.csv"),
                Err(e) => logln!("LIGHTMAP_SCAN csv failed: {e}"),
            }
        } else { logln!("HMS_LIGHTMAP_SCAN: expected x0,y0,x1,y1,step[,radius]"); }
    }
    // Baked-colour enumeration + palette + placement search (diagnostics):
    //   HMS_LIGHT_PALETTE=<n>        enumerate every baked colour (lightmap texels / PVL verts / probes /
    //                                airprobes), quantize into n swatches; prints them, writes
    //                                <stem>_lightbake.png (top-down map painted from the samples, swatch
    //                                colours) and <stem>_palette.png (numbered swatch strip).
    //   HMS_LIGHT_FIND=<swatch>[,n]  or  r,g,b[,n] (0-255 display values → nearest swatch first): find
    //                                where the swatch occurs, probe the real object placement ONLY there
    //                                (5×5 @ 1 wu around the densest 4 wu cells + example samples), print
    //                                the best n placements and write <stem>_lightbake_sel.png (highlight).
    //   HMS_LIGHT_RADIUS=<r>         probe radius for the placement probe (default 0.4 = crate-like).
    if std::env::var("HMS_LIGHT_PALETTE").is_ok() || std::env::var("HMS_LIGHT_FIND").is_ok() {
        let n: usize = std::env::var("HMS_LIGHT_PALETTE").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64);
        let expo = std::env::var("HMS_LIGHTBAKE_EXPO").ok().and_then(|v| v.parse::<f32>().ok()).filter(|e| *e > 0.0);
        let colors = scene.light_bake_enumerate(expo, n, None);
        let c = &colors.counts;
        logln!("LIGHT_PALETTE enumerated {} samples from {} soup tris: {} lightmap texels, {} pvl verts, {} pvl-colour verts, {} probe tris, {} airprobes → {} colour buckets (4-bit) → {} swatches; exposure={:.4} sun=({:.2},{:.2},{:.2}) in {:.0} ms",
            colors.total, colors.tris, c[1], c[2], c[3], c[4], c[5], colors.bucket_count, colors.swatches.len(), colors.exposure, colors.sun_rgb[0], colors.sun_rgb[1], colors.sun_rgb[2], colors.elapsed_ms);
        logln!("LIGHT_PALETTE swatches (display rgb 0-255 = PNG pixel values; linear = renderer HDR units):\n{}", colors.palette_text().trim_end());
        let stem = out.trim_end_matches(".png").to_string();
        let img = colors.render_map(0, 25.0, None);
        match write_png(&format!("{stem}_lightbake.png"), &img.rgba, img.w as u32, img.h as u32) {
            Ok(()) => logln!("LIGHT_PALETTE wrote {stem}_lightbake.png ({}x{}, {} px per {} wu)", img.w, img.h, img.scale, colors.raster.step),
            Err(e) => logln!("LIGHT_PALETTE png failed: {e}"),
        }
        let pal = colors.render_palette();
        match write_png(&format!("{stem}_palette.png"), &pal.rgba, pal.w as u32, pal.h as u32) {
            Ok(()) => logln!("LIGHT_PALETTE wrote {stem}_palette.png"),
            Err(e) => logln!("LIGHT_PALETTE palette png failed: {e}"),
        }
        if let Ok(f) = std::env::var("HMS_LIGHT_FIND") {
            let q: Vec<f32> = f.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let radius = std::env::var("HMS_LIGHT_RADIUS").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.4);
            let (swatch, nres) = if q.len() >= 3 {
                let target = [q[0].clamp(0.0, 255.0) as u8, q[1].clamp(0.0, 255.0) as u8, q[2].clamp(0.0, 255.0) as u8];
                let sw = colors.nearest_swatch(target);
                logln!("LIGHT_FIND target rgb=({},{},{}) → nearest swatch {:?}", target[0], target[1], target[2], sw);
                (sw, q.get(3).map(|v| *v as usize).unwrap_or(10))
            } else if !q.is_empty() {
                (Some(q[0] as usize), q.get(1).map(|v| *v as usize).unwrap_or(10))
            } else { (None, 10) };
            match swatch.and_then(|s| colors.swatches.get(s).map(|w| (s, w))) {
                Some((si, sw)) => {
                    logln!("LIGHT_FIND swatch {si}: rgb=({},{},{}) lin=({:.4},{:.4},{:.4}) pop={} occurs in {} cells (4 wu); densest: {}",
                        sw.disp[0], sw.disp[1], sw.disp[2], sw.lin[0], sw.lin[1], sw.lin[2], sw.population, sw.cells.len(),
                        sw.cells.iter().take(6).map(|(p, n)| format!("({:.0},{:.0},{:.0})×{n}", p[0], p[1], p[2])).collect::<Vec<_>>().join(" "));
                    let t1 = std::time::Instant::now();
                    let (places, probes) = scene.light_bake_find_placements(&colors, si, nres.max(1), radius);
                    logln!("LIGHT_FIND {} local placement probes in {:.0} ms → {} placements (radius {:.2}):", probes, t1.elapsed().as_secs_f32() * 1000.0, places.len(), radius);
                    for (rank, p) in places.iter().enumerate() {
                        logln!("  #{:<2} x={:>8.2} y={:>8.2} ground z={:>7.2}  place at z={:>7.2}  rgb=({:>3},{:>3},{:>3}) dist={:>5.1}  mask={:.2} {}  lin=({:.4},{:.4},{:.4}) key=({:.3},{:.3},{:.3}) fill=({:.3},{:.3},{:.3})",
                            rank + 1, p.x, p.y, p.z, p.place_z, p.disp[0], p.disp[1], p.disp[2], p.dist, p.mask, crate::scene::lb_kind_name(p.kind),
                            p.color_lin[0], p.color_lin[1], p.color_lin[2], p.key_rgb[0], p.key_rgb[1], p.key_rgb[2], p.fill_rgb[0], p.fill_rgb[1], p.fill_rgb[2]);
                    }
                    let img = colors.render_map(0, 25.0, Some(si));
                    if write_png(&format!("{stem}_lightbake_sel.png"), &img.rgba, img.w as u32, img.h as u32).is_ok() { logln!("LIGHT_FIND wrote {stem}_lightbake_sel.png (swatch {si} highlighted)"); }
                }
                None => logln!("HMS_LIGHT_FIND: expected <swatch index>[,n] or r,g,b[,n] (swatch out of range?)"),
            }
        }
    }
    // HMS_RAYPROBE=1 (print-only): cast a fan of rays from the camera and report hit (geometry
    // decoded + present) vs miss (void).
    if std::env::var("HMS_RAYPROBE").is_ok() {
        logln!("RAYPROBE soup_tris={} cam=[{:.1},{:.1},{:.1}] yaw={:.1} pitch={:.1}",
            scene.soup_len(), cam.pos.x, cam.pos.y, cam.pos.z, cam.yaw.to_degrees(), cam.pitch.to_degrees());
        let vp = cam.view_proj(w as f32 / h as f32);
        let inv = vp.inverse();
        // Sample a grid of NDC pixels across the frame (x:-1..1 L→R, y:-1..1 B→T).
        for ndy in [0.6f32, 0.3, 0.0, -0.3, -0.6] {
            for ndx in [-0.5f32, -0.2, 0.1, 0.4, 0.7, 0.95] {
                let np = inv * glam::Vec4::new(ndx, ndy, 0.0, 1.0);
                let fp = inv * glam::Vec4::new(ndx, ndy, 1.0, 1.0);
                let n = np.truncate() / np.w;
                let f = fp.truncate() / fp.w;
                let d = (f - n).normalize_or_zero();
                match scene.raycast_material(cam.pos, d) {
                    Some((t, diff, blend, mat)) => logln!("  RAYPROBE ndc=({:+.2},{:+.2}) HIT dist={:.1} mat={} blend={} diffuse={:#x}",
                        ndx, ndy, t, mat, blend, diff),
                    None => logln!("  RAYPROBE ndc=({:+.2},{:+.2}) MISS(void)", ndx, ndy),
                }
            }
        }
    }

    // --- render one frame + read back ---
    // HMS_TIME picks the animation time of the shot (water waves, holo drift, scrolling
    // textures; default 0).
    let shot_time: f32 = std::env::var("HMS_TIME").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    // Simulate the effect particle systems (grav lifts, waterfalls) at the shot time with
    // the shot camera (CPU billboarding) and hand the batches to the renderer.
    {
        let t0 = std::time::Instant::now();
        let fwd = cam.forward(); let right = cam.right(); let up = right.cross(fwd).normalize();
        let pm = scene.build_particle_meshes(cam.pos, fwd, right, up, shot_time, renderer.mesh_renderer(), &device, &queue);
        logln!("particles: {} batches ({:.1} ms)", pm.len(), t0.elapsed().as_secs_f64() * 1000.0);
        renderer.set_particle_meshes(pm);
        // Placed lens flares (street-light glares etc.), projected + occlusion-tested per frame.
        let lf = scene.lens_flare_instances(&device, &queue);
        logln!("lens flares: {} instances", lf.len());
        renderer.set_lens_flare_instances(lf);
    }
    // Re-pack the SimpleLights uniform once the objects are built so placed Forge lights
    // (HMS_PLACE / HMS_MVAR) light the capture: the nearest-to-camera dynamic lights at the shot
    // time, like the GUI does per frame.
    if scene.has_forge_lights() {
        let sl = scene.simple_lights_with_forge(cam.pos, shot_time);
        renderer.set_simple_lights(&queue, (sl.len() / 20) as u32, &sl);
        logln!("forge lights: {} packed into SimpleLights (of {} scenario)", sl.len() / 20, scene.simple_lights().len() / 20);
    }
    if std::env::var("HMS_RTGI").is_ok() {
        // one warm render first so the sky env cube (the tracer's sky radiance) is captured
        renderer.render(&device, &queue, &cam, shot_time);
    }
    // HMS_RTGI=1: real-time probe GI replaces the baked lightmaps; HMS_RTGI_WARMUP=<n> update
    // passes before the shot (default 48) so the multi-bounce field has converged.
    if std::env::var("HMS_RTGI").is_ok() {
        let t0 = std::time::Instant::now();
        match scene.rtgi_scene_data() {
            Some(sd) => {
                renderer.set_rtgi_scene(&device, &queue, &sd);
                renderer.set_rtgi_enabled(&queue, true);
                let n: u32 = std::env::var("HMS_RTGI_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
                let t1 = std::time::Instant::now();
                renderer.rtgi_warm_up(&device, &queue, n);
                let (np, sp, d, _) = renderer.rtgi_info();
                logln!("HMS_RTGI: {} probes ({}x{}x{}, {:.2} wu), scene {} ms, warm-up {} passes {} ms", np, d[0], d[1], d[2], sp, (t1 - t0).as_millis(), n, t1.elapsed().as_millis());
            }
            None => logln!("HMS_RTGI: no occluder geometry captured — GI disabled"),
        }
        // HMS_RTGI_PROBE_AT="x,y,z": dump the 8 probe records around a world point (diagnostic)
        if let Ok(v) = std::env::var("HMS_RTGI_PROBE_AT") {
            let c: Vec<f32> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            if c.len() == 3 { for line in renderer.rtgi_dump_at(&device, &queue, [c[0], c[1], c[2]]) { logln!("{line}"); } }
        }
    }
    // The 1x1 luminance meter is written by a pass that runs after the scene pass, so the mesh /
    // terrain shaders read the previous frame's meter; a single render would read a zero-cleared
    // meter, drive illum_scale_now() to the band floor and leave every self-illum surface ~3x
    // brighter than in the interactive app. Two warm frames populate the meter first.
    for _ in 0..2 { renderer.render(&device, &queue, &cam, shot_time); }
    // HMS_FRAMES=<n> renders n frames after load with a moving camera (slow orbit +
    // yaw sweep) through the SAME per-frame path the interactive app runs each update(): particle
    // sim + upload, the object snapshot / rebuild check, forge-light repack, render (encode+submit)
    // and the non-blocking exposure-meter readback. The queue is pipelined two frames deep like
    // a swapchain (wait for frame i-2's submission before encoding frame i), so `gpu_wait` is the
    // present-equivalent backpressure. Prints per-stage mean/p99/max and every frame slower than
    // 2x the median with its dominant stage. Print-only diagnostic; the shot is unaffected.
    if let Some(n) = std::env::var("HMS_FRAMES").ok().and_then(|s| s.trim().parse::<usize>().ok()).filter(|&n| n > 0) {
        const STAGES: [&str; 6] = ["particles", "objsnap", "lights", "render", "meter", "gpu_wait"];
        let mut rows: Vec<[f32; 7]> = Vec::with_capacity(n); // per-frame [total, stages...]
        let mut pending: std::collections::VecDeque<wgpu::SubmissionIndex> = std::collections::VecDeque::new();
        let base = cam.clone();
        let mut meter_hits = 0usize; // frames on which the async meter had a value
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f32() * 1000.0;
        let mut ps_settled: Option<(u64, u64, f64, f64, u64)> = None; // page stats at the settle point
        for i in 0..n {
            let f = i as f32;
            // What the GUI does once the post-load object tail has settled (main.rs
            // settle_pending -> release_burst_memory): the first frames' on-demand decodes refill
            // the native page cache, so the settled footprint is measured after that release.
            if i == 30 { scene.release_burst_memory(); scene.print_mem_diag("frames-settled"); ps_settled = Some(scene.page_stats()); }
            // moving camera: yaw sweep +-20 deg and a small orbit around the start position so
            // sort orders, particle billboards, the meter and the sun shadow frustum all change.
            cam.yaw = base.yaw + (f * 0.021).sin() * 0.35;
            cam.pitch = base.pitch + (f * 0.013).cos() * 0.08;
            cam.pos = base.pos + Vec3::new((f * 0.017).sin() * 3.0, (f * 0.011).cos() * 3.0, (f * 0.009).sin() * 1.0);
            let t_frame = std::time::Instant::now();
            let mut st = [0f32; 6];
            // present-equivalent: block until frame i-2 has finished on the GPU.
            let t = std::time::Instant::now();
            if pending.len() >= 2 {
                if let Some(idx) = pending.pop_front() { let _ = device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx)); }
            }
            st[5] = ms(t);
            let time = shot_time + f / 60.0;
            let t = std::time::Instant::now();
            {
                let fwd = cam.forward(); let right = cam.right(); let up = right.cross(fwd).normalize();
                let pm = scene.build_particle_meshes(cam.pos, fwd, right, up, time, renderer.mesh_renderer(), &device, &queue);
                renderer.set_particle_meshes(pm);
                renderer.set_lens_flare_instances(scene.lens_flare_instances(&device, &queue));
            }
            st[0] = ms(t);
            let t = std::time::Instant::now();
            {
                // main.rs tick_scene: fresh object snapshot every frame + signature check. Frames
                // 100..160 also nudge the last object (a grab-drag: the signature changes every
                // frame, so the full object rebuild pass runs) to measure the move cost.
                if (100..160).contains(&i) {
                    if let Some(o) = objects.last_mut() { o.pos[2] += 0.01; }
                }
                let snap = objects.clone();
                if let Some((op, cut, holo, hs, bl)) = scene.maybe_rebuild(&snap, &empty, renderer.mesh_renderer(), &device, &queue) {
                    renderer.set_dynamic_meshes(op); renderer.set_dynamic_cutout_meshes(cut);
                    renderer.set_dynamic_holo_meshes(holo); renderer.set_dynamic_holo_solid_meshes(hs); renderer.set_dynamic_blend_meshes(bl);
                }
            }
            st[1] = ms(t);
            let t = std::time::Instant::now();
            if scene.has_forge_lights() {
                let sl = scene.simple_lights_with_forge(cam.pos, time);
                renderer.set_simple_lights(&queue, (sl.len() / 20) as u32, &sl);
            }
            st[2] = ms(t);
            let t = std::time::Instant::now();
            renderer.render(&device, &queue, &cam, time);
            pending.push_back(queue.submit(std::iter::empty()));
            st[3] = ms(t);
            let t = std::time::Instant::now();
            if renderer.read_mean_log_async(&device, &queue).is_some() { meter_hits += 1; } // main.rs: every frame, non-blocking
            st[4] = ms(t);
            let total = ms(t_frame);
            rows.push([total, st[0], st[1], st[2], st[3], st[4], st[5]]);
        }
        let _ = device.poll(wgpu::Maintain::Wait);
        cam = base;
        // GUI-only per-frame costs that the loop above cannot mirror, timed in isolation:
        // the properties panel's material inspector (main.rs: object_material per selected object).
        if let Some(tag) = objects.iter().map(|o| o.mode_tag).find(|&t| t != 0 && t != 0xFFFF_FFFF) {
            let t = std::time::Instant::now();
            for _ in 0..10 { let _ = scene.object_material(tag); }
            logln!("HMS_FRAMES   object_material({tag:#x}) x10 = {:.2} ms each", ms(t) / 10.0);
        }
        // the collision/physics hull overlays (main.rs rebuild_overlays, rebuilt after every object
        // rebuild = every frame of a drag when the overlay toggles are on).
        if let Some(o) = objects.iter().find(|o| o.mode_tag != 0 && o.mode_tag != 0xFFFF_FFFF) {
            let t = std::time::Instant::now();
            let mut n = 0;
            for _ in 0..10 { n = scene.collision_world_edges(o).len(); }
            let tc = ms(t) / 10.0;
            let t = std::time::Instant::now();
            let mut np = 0;
            for _ in 0..10 { np = scene.physics_world_edges(o).len(); }
            logln!("HMS_FRAMES   collision_world_edges x10 = {tc:.2} ms each ({n} edges); physics_world_edges x10 = {:.2} ms each ({np} edges)", ms(t) / 10.0);
        }
        let stat = |k: usize| -> (f32, f32, f32, f32) {
            let mut v: Vec<f32> = rows.iter().map(|r| r[k]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
            let med = v[v.len() / 2];
            let p99 = v[((v.len() as f32 * 0.99) as usize).min(v.len() - 1)];
            (mean, med, p99, *v.last().unwrap_or(&0.0))
        };
        let (tm, tmed, tp99, tmax) = stat(0);
        logln!("HMS_FRAMES {} frames {}x{} (SS x2): total mean={tm:.2} median={tmed:.2} p99={tp99:.2} max={tmax:.2} ms | async meter had a value on {meter_hits}/{n} frames", n, w, h);
        // HMS_MEMDIAG: what the render loop pulled back into the native page cache after the
        // settle release; the steady-state page-cache footprint is exactly these re-inflates.
        if let (Some(a), true) = (ps_settled, std::env::var("HMS_MEMDIAG").is_ok()) {
            let b = scene.page_stats();
            logln!("HMS_MEMDIAG frames 30..{n}: page cache hits={} inflates={} inflated={:.0} MB ({:.0} ms CPU) stores={}", b.0 - a.0, b.1 - a.1, b.2 - a.2, b.3 - a.3, b.4 - a.4);
        }
        // resident set after the run (Linux): a swapped-out working set is itself a hitch source.
        if let Ok(st) = std::fs::read_to_string("/proc/self/status") {
            for l in st.lines().filter(|l| l.starts_with("VmRSS") || l.starts_with("VmSwap") || l.starts_with("VmHWM")) { logln!("HMS_FRAMES   {}", l.split_whitespace().collect::<Vec<_>>().join(" ")); }
        }
        for (k, name) in STAGES.iter().enumerate() {
            let (m, med, p99, mx) = stat(k + 1);
            logln!("HMS_FRAMES   {name:<10} mean={m:>7.2} median={med:>7.2} p99={p99:>7.2} max={mx:>7.2} ms");
        }
        let thresh = tmed * 2.0;
        let slow: Vec<usize> = (0..rows.len()).filter(|&i| rows[i][0] > thresh).collect();
        logln!("HMS_FRAMES   frames > 2x median ({thresh:.2} ms): {}", slow.len());
        for i in slow.iter().take(40) {
            let r = &rows[*i];
            let (mut top, mut topv) = (0usize, 0f32);
            for k in 0..6 { if r[k + 1] > topv { topv = r[k + 1]; top = k; } }
            logln!("HMS_FRAMES     frame {i:>4}: {:>7.2} ms  top={} {:.2} ms  [{}]", r[0], STAGES[top], topv,
                (0..6).map(|k| format!("{}={:.2}", STAGES[k], r[k + 1])).collect::<Vec<_>>().join(" "));
        }
    }
    renderer.render(&device, &queue, &cam, shot_time);
    // HMS_HDR_DUMP=<path.f32> writes the linear pre-tonemap HDR scene (SS x size, RGBA f32,
    // little-endian, row-major): the per-pixel probe that does not go through the tonemap +
    // auto-exposure. The log line gives the dimensions.
    if let Ok(hp) = std::env::var("HMS_HDR_DUMP") {
        let (hdr, hw, hh) = renderer.dump_hdr(&device, &queue);
        let mut bytes = Vec::with_capacity(hdr.len() * 4);
        for v in &hdr { bytes.extend_from_slice(&v.to_le_bytes()); }
        match std::fs::write(&hp, &bytes) {
            Ok(()) => logln!("HMS_HDR_DUMP: wrote {}x{} RGBA-f32 linear HDR to {}", hw, hh, hp),
            Err(e) => logln!("HMS_HDR_DUMP: write failed: {e}"),
        }
    }
    let (rgba, rw, rh) = renderer.capture_rgba(&device, &queue);
    // HMS_SAMPLE="x0,y0,x1,y1" (pixels, print-only): the mean sRGB of that region + the share of
    // near-white (blown, >0.96) pixels, so a colour claim can be checked numerically.
    if let Ok(s) = std::env::var("HMS_SAMPLE") {
        let n: Vec<i64> = s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        if n.len() == 4 {
            let (x0, y0, x1, y1) = (n[0].max(0) as u32, n[1].max(0) as u32, (n[2] as u32).min(rw), (n[3] as u32).min(rh));
            let (mut r, mut g, mut b, mut cnt, mut blown) = (0u64, 0u64, 0u64, 0u64, 0u64);
            for y in y0..y1 { for x in x0..x1 {
                let i = ((y * rw + x) * 4) as usize;
                if i + 2 < rgba.len() {
                    r += rgba[i] as u64; g += rgba[i + 1] as u64; b += rgba[i + 2] as u64; cnt += 1;
                    let l = 0.2126 * rgba[i] as f32 + 0.7152 * rgba[i + 1] as f32 + 0.0722 * rgba[i + 2] as f32;
                    if l > 245.0 { blown += 1; }
                }
            }}
            let c = cnt.max(1) as f32;
            logln!("HMS_SAMPLE region=({x0},{y0})-({x1},{y1}) mean_rgb=[{:.3},{:.3},{:.3}] blown={:.1}%",
                r as f32 / c / 255.0, g as f32 / c / 255.0, b as f32 / c / 255.0, 100.0 * blown as f32 / c);
        }
    }
    // HMS_PICKTEST=1 (print-only): cast rays through several screen points and report the BSP
    // material each hits (per-triangle picking gives different materials per ray).
    if std::env::var("HMS_PICKTEST").is_ok() {
        let aspect = rw as f32 / rh as f32;
        let inv = cam.view_proj(aspect).inverse();
        // HMS_PICK_NDC="x,y;x,y;...": any number of extra probe points (NDC, y up).
        let extra: Vec<(f32, f32, &str)> = std::env::var("HMS_PICK_NDC").ok()
            .map(|s| s.split(';').filter_map(|pt| { let n: Vec<f32> = pt.split(',').filter_map(|t| t.trim().parse().ok()).collect();
                if n.len() == 2 { Some((n[0], n[1], "custom")) } else { None } }).collect::<Vec<_>>())
            .unwrap_or_default();
        for (nx, ny, label) in [
            (0.0f32, 0.0f32, "center"), (-0.5, -0.5, "lower-left"), (0.5, -0.5, "lower-right"),
            (-0.5, 0.5, "upper-left"), (0.5, 0.5, "upper-right"), (0.0, -0.7, "bottom"),
        ].into_iter().chain(extra) {
            let near = inv.project_point3(Vec3::new(nx, ny, 0.0));
            let far = inv.project_point3(Vec3::new(nx, ny, 1.0));
            let dir = (far - near).normalize_or_zero();
            match scene.bsp_material_at(near, dir) {
                Some((d, b, mi)) => {
                    let hp = scene.raycast_scene(near, dir).map(|p| format!("({:.2},{:.2},{:.2})", p.x, p.y, p.z)).unwrap_or_default();
                    logln!("HMS_PICKTEST {label} ndc=({nx:.3},{ny:.3}): diffuse=0x{d:04x} '{}' blend={b} mat_idx={mi} hit={hp} | {}", scene.tag_name_of(d), scene.pick_light_note(near, dir));
                }
                None => logln!("HMS_PICKTEST {label}: (no hit)"),
            }
        }
    }
    write_png(&out, &rgba, rw, rh)?;
    logln!("wrote {out} ({rw}x{rh})");
    // HMS_PREVIEW_SHOT=<obj hex>[:<variant sid hex>]:<png path> (several specs may be joined with
    // ';') renders the palette preview of that object exactly as the GUI's palette panel does: a
    // second 280x280 SceneRenderer (no sky / grid), the same per-map lighting mirror,
    // `build_preview_meshes` (the viewport's object path) for one placement at the shot camera
    // position, the same orbit camera (`preview_camera`, default 3/4 view yaw 0.7 / pitch 0.5 /
    // zoom 1). HMS_PREVIEW_SIZE=<px> overrides the square size; HMS_PREVIEW_ORBIT="yaw,pitch,zoom"
    // the orbit. So the preview can be compared against the viewport render of the same object.
    for spec in std::env::var("HMS_PREVIEW_SHOT").ok().iter().flat_map(|v| v.split(';')).map(str::trim).filter(|s| !s.is_empty()) {
        let mut it = spec.splitn(3, ':');
        let obj = it.next().and_then(|t| u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok());
        let (vsid, ppath) = match (it.next(), it.next()) {
            (Some(v), Some(pth)) => (u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).unwrap_or(0), pth.to_string()),
            (Some(pth), None) => (0, pth.to_string()),
            _ => (0, String::new()),
        };
        match obj {
            Some(obj) if !ppath.is_empty() => {
                let psz: u32 = std::env::var("HMS_PREVIEW_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(280);
                let orbit: Vec<f32> = std::env::var("HMS_PREVIEW_ORBIT").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
                let (pyaw, ppitch, pzoom) = (orbit.first().copied().unwrap_or(0.7), orbit.get(1).copied().unwrap_or(0.5), orbit.get(2).copied().unwrap_or(1.0));
                let mut pr = SceneRenderer::new(&device, &queue, (psz, psz));
                pr.show_sky = false;
                pr.show_grid = false;
                // The same per-map lighting the GUI mirrors into its preview renderer at map load.
                let (sun_tint, amb_tint) = scene.scene_tints();
                pr.set_scene_tints(sun_tint, amb_tint);
                pr.set_obj_dir_strength(scene.obj_dir_strength());
                if let Some((key, lo, hi)) = scene.autoexposure_band() { pr.set_auto_exposure(&queue, key, lo, hi); }
                pr.set_illum_exposure(scene.illum_exposure());
                pr.set_sun_reach(scene.baked_sun_reach());
                pr.set_exposure(&queue, scene.exposure());
                if let Some((pt, inh, inten)) = scene.bloom_curve() { pr.set_bloom_curve(&queue, pt, inh, inten); }
                if let Some(c) = scene.bloom_colors() { pr.set_bloom_colors(&queue, c); }
                if let Some(sun) = scene.scene_sun_dir() { pr.set_sun_dir(Vec3::from(sun)); }
                let built = scene.build_preview_meshes(obj, vsid, cam.pos, renderer.mesh_renderer(), &device, &queue);
                match built {
                    Ok((op, cut, holo, hs, bl, mn, mx)) => {
                        logln!("HMS_PREVIEW_SHOT obj={obj:#x} vsid={vsid:#x}: meshes opaque={} cutout={} holo={} holo_solid={} blend={} aabb=[{:.2},{:.2},{:.2}]-[{:.2},{:.2},{:.2}]",
                            op.len(), cut.len(), holo.len(), hs.len(), bl.len(), mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
                        pr.set_dynamic_meshes(op);
                        pr.set_dynamic_cutout_meshes(cut);
                        pr.set_dynamic_holo_meshes(holo);
                        pr.set_dynamic_holo_solid_meshes(hs);
                        pr.set_dynamic_blend_meshes(bl);
                        let center = (mn + mx) * 0.5;
                        let radius = ((mx - mn).length() * 0.5).max(0.5);
                        let pcam = crate::preview_camera(center, mn, mx, radius, pyaw, ppitch, pzoom);
                        // A few frames so the exposure meter settles like the live panel's periodic refresh.
                        for _ in 0..4 { pr.render(&device, &queue, &pcam, shot_time); }
                        let (prgba, pw, ph) = pr.capture_rgba(&device, &queue);
                        match write_png(&ppath, &prgba, pw, ph) {
                            Ok(()) => logln!("HMS_PREVIEW_SHOT: wrote {ppath} ({pw}x{ph}) cam=[{:.2},{:.2},{:.2}] yaw={:.1} pitch={:.1}", pcam.pos.x, pcam.pos.y, pcam.pos.z, pcam.yaw.to_degrees(), pcam.pitch.to_degrees()),
                            Err(e) => logln!("HMS_PREVIEW_SHOT: write failed: {e}"),
                        }
                    }
                    Err(inflight) => logln!("HMS_PREVIEW_SHOT obj={obj:#x}: no preview meshes (decode in flight={inflight})"),
                }
            }
            _ => logln!("HMS_PREVIEW_SHOT: expected <obj hex>[:<variant sid hex>]:<png path>"),
        }
    }
    // HMS_EXPODIAG=1 (print-only): metered luminance + where the auto-gain lands vs the cfxs band.
    if std::env::var("HMS_EXPODIAG").is_ok() {
        if let Some(ml) = renderer.read_mean_log(&device, &queue) {
            let lum = 2f32.powf(ml);
            let (key, min_ev, max_ev) = scene.autoexposure_band().map(|b| (b.0, b.1, b.2)).unwrap_or((0.25, -2.0, 1.0));
            // Mirrors Renderer::set_auto_exposure (band = key * C * 2^ev, C = 10).
            let lo = key * 10.0 * 2f32.powf(min_ev);
            let hi = key * 10.0 * 2f32.powf(max_ev);
            // Mirrors the shader's POST_EXPOSURE_CAL (1.0); the correction is applied before the clamp.
            let cal = 1.0f32;
            let corrected = (key / lum.max(1e-4)) * cal;
            let gain = corrected.clamp(lo.min(hi), hi);
            let clamped = if corrected >= hi { "CLAMPED@max" } else if corrected <= lo { "CLAMPED@min" } else { "adapting" };
            logln!("HMS_EXPODIAG mean_log={ml:.3} lum={lum:.4} corrected_gain={corrected:.2} band=[{lo:.2},{hi:.2}] -> net={gain:.3} ({clamped})");
        }
    }
    // Working set after the (post-trim) render: the steady footprint the interactive viewer
    // settles at, since only render-touched pages fault back after the load-heap trim.
    #[cfg(windows)]
    if std::env::var("HMS_MEMPROF").is_ok() { unsafe {
        use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX, PROCESS_MEMORY_COUNTERS};
        use windows::Win32::System::Threading::GetCurrentProcess;
        let mut c = PROCESS_MEMORY_COUNTERS_EX::default();
        if GetProcessMemoryInfo(GetCurrentProcess(),
            &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32).is_ok() {
            logln!("MEMPROF steady-after-render: WorkingSet {:.0} MB",
                (c.WorkingSetSize as f64) / (1024.0 * 1024.0));
        }
    } }
    scene.print_mem_diag("steady-after-render");

    let _ = std::fs::write(format!("{out}.log"), &log);
    Ok(())
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

fn resolve_map(cands: &[mapcat::MapCandidate], want: &str) -> Option<String> {
    if want.is_empty() {
        return cands.first().map(|c| c.path.to_string_lossy().into_owned());
    }
    let w = want.to_lowercase();
    // exact stem, then substring on stem/path.
    cands
        .iter()
        .find(|c| c.stem.to_lowercase() == w)
        .or_else(|| cands.iter().find(|c| c.stem.to_lowercase().contains(&w) || c.path.to_string_lossy().to_lowercase().contains(&w)))
        .map(|c| c.path.to_string_lossy().into_owned())
}

/// Explicit HMS_CAM, else auto-frame the object centroid (playable area) or scene bounds.
fn build_camera(scene: &SceneController, obj_centroid: Option<(Vec3, f32)>) -> Camera {
    let mut cam = Camera::default();
    // HMS_FOV (vertical FOV in degrees) overrides the default, e.g. to match an engine capture.
    if let Some(fov) = std::env::var("HMS_FOV").ok().and_then(|s| s.trim().parse::<f32>().ok()) {
        if fov > 1.0 && fov < 179.0 { cam.fov_y = fov.to_radians(); }
    }
    // HMS_SPAWNCAM: the interactive spawn-point camera.
    if std::env::var("HMS_SPAWNCAM").is_ok() {
        if let Some((pos, yaw, pitch)) = scene.spawn_camera_pose() {
            cam.pos = pos; cam.yaw = yaw; cam.pitch = pitch;
            return cam;
        }
    }
    if let Ok(s) = std::env::var("HMS_CAM") {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() == 5 {
            cam.pos = Vec3::new(p[0], p[1], p[2]);
            cam.yaw = p[3].to_radians();
            cam.pitch = p[4].to_radians();
            return cam;
        }
    }
    // Prefer the object centroid (playable area) over raw bounds (which distant skybox
    // geometry blows up). Fall back to bounds when there are no objects.
    let framed = obj_centroid
        .map(|(c, r)| (c, r * 2.5))
        .or_else(|| scene.scene_bounds().map(|(mn, mx)| {
            let mn = Vec3::from(mn);
            let mx = Vec3::from(mx);
            ((mn + mx) * 0.5, (mx - mn).length() * 0.5)
        }));
    if let Some((center, radius)) = framed {
        let radius = radius.max(5.0);
        let view = std::env::var("HMS_VIEW").unwrap_or_default();
        let dmul: f32 = std::env::var("HMS_DIST").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
        let dist = (radius / (cam.fov_y * 0.5).tan() * 1.15 + radius * 0.2) * dmul;
        let (dir, pitch) = match view.as_str() {
            "top" => (Vec3::new(0.0, 0.0, -1.0), -std::f32::consts::FRAC_PI_2 + 0.01),
            "front" => (Vec3::new(0.0, 1.0, -0.15), -0.15),
            _ => (Vec3::new(0.6, 0.6, -0.5).normalize(), -0.5), // 3/4 overview (camera ABOVE, looking down)
        };
        cam.pos = center - dir * dist;
        let to = (center - cam.pos).normalize();
        cam.yaw = to.y.atan2(to.x);
        cam.pitch = if view == "top" { pitch } else { to.z.clamp(-1.0, 1.0).asin() };
        cam.far = (radius * 6.0).max(5000.0);
    }
    cam
}

pub(crate) fn write_png(path: &str, rgba: &[u8], w: u32, h: u32) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let bw = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(bw, w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}

/// Directory for print/dump diagnostics (PNG dumps etc.): `HMS_DIAG_DIR` if set, else the OS temp
/// directory. Never a hard-coded user path.
pub fn diag_dir() -> std::path::PathBuf {
    let d = std::env::var("HMS_DIAG_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| std::env::temp_dir().join("hms-diag"));
    let _ = std::fs::create_dir_all(&d);
    d
}
