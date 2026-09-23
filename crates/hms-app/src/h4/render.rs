//! Halo 4 headless screenshot: `HMS_SHOT=<png> HMS_MAP=<halo4 map>` lands here from
//! `headless::run` when the map is a Halo 4 cache. The scene is built by scene.rs (the same
//! path the GUI load worker runs), rendered once through the wgpu renderer and written as PNG
//! with a `<png>.log` beside it.
//!
//! Env knobs: HMS_CAM (x,y,z,yawDeg,pitchDeg), HMS_VIEW=top|front, HMS_DIST, HMS_FOV,
//! HMS_H4_NOTEX=1 (no textures), HMS_H4_MAXMESH=<n> (debug: cap the mesh count),
//! HMS_H4_NOOBJ=1 (no scenario objects), HMS_H4_MATDIAG=1 (print every material),
//! HMS_MVAR=<halo4 .mvar> (place the variant's Forge objects; HMS_MVAR_LIST=1 lists them;
//! HMS_MVAR_SAVE=<dst.mvar> writes the variant back through h4::mvar::save_objects,
//! HMS_MVAR_SAVE_MOVE=dx,dy,dz moves every object first - a scripted edit for headless checks),
//! HMS_H4_LIGHTMAP=0 (disable the engine-law lightmap atlas decode, see lightmaps.rs),
//! HMS_H4_LMDIAG=1 (print the implied baked / sun luminance stats), HMS_FRAMES=<n> (n extra
//! moving-camera frames), HMS_SPAWNCAM=1 (the GUI's post-load camera: variant loadout camera /
//! spawn, else the scenario's starting location / spawn placements, stood off the geometry -
//! builds the editor scene like HMS_H4_SOUP=1; the log line names the source),
//! HMS_PREVIEW_SHOT=<item>:<png>[;...] (#h4-preview: the palette preview of a Forge palette row -
//! q<quota>[.<variant>] / n<raw name substring> / <obje tag hex> - through the viewport's own
//! object path; HMS_PREVIEW_SIZE, HMS_PREVIEW_ORBIT="yaw,pitch,zoom" as on the Reach path).

use anyhow::{anyhow, Result};
use eframe::wgpu;
use glam::Vec3;
use hms_render::{Camera, SceneRenderer};

use super::scene::{build_meshes, load_map, load_map_with_variant, H4EditorAssets, H4EditorCollect, Lane};

pub fn headless_run(map_path: &str, out: &str, w: u32, h: u32) -> Result<()> {
    let mut log = String::new();
    macro_rules! logln { ($($a:tt)*) => {{ let s = format!($($a)*); eprintln!("{s}"); log.push_str(&s); log.push('\n'); }} }
    let t0 = std::time::Instant::now();
    // The cache + BSP parse and the mesh build are shared with the GUI worker (scene.rs).
    // HMS_MVAR=<halo4 .mvar> places the variant's Forge objects (like the Reach path);
    // HMS_MVAR_LIST=1 also prints every decoded object
    let variant = std::env::var("HMS_MVAR").ok().filter(|s| !s.trim().is_empty()).map(std::path::PathBuf::from);
    let loaded = match &variant {
        Some(vp) => {
            if !super::mvar::is_h4_variant(vp) { anyhow::bail!("HMS_MVAR='{}' is not a Halo 4 map variant (mvar chunk v50)", vp.display()); }
            load_map_with_variant(map_path, Some(vp))?
        }
        None => load_map(map_path)?,
    };
    for l in &loaded.log { logln!("{l}"); }
    // HMS_MVAR_SAVE=<dst.mvar>: the headless `save as` - write the loaded variant back through
    // the one save function (its own object list, no edits; the result must be byte-identical
    // to the source, which the tests gate on all 388 shipped files). HMS_MVAR_SAVE_MOVE=dx,dy,dz
    // applies a scripted edit first - every object moved by that delta - so an edited save can
    // be checked headlessly by reloading the file.
    if let (Some(vp), Ok(dst)) = (&variant, std::env::var("HMS_MVAR_SAVE")) {
        let v = super::mvar::parse_h4_variant(vp)?;
        let mut objects = v.objects.clone();
        let delta: Option<[f32; 3]> = std::env::var("HMS_MVAR_SAVE_MOVE").ok().and_then(|m| {
            let p: Vec<f32> = m.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (p.len() == 3).then(|| [p[0], p[1], p[2]])
        });
        if let Some(d) = delta { for o in objects.iter_mut() { for k in 0..3 { o.pos[k] += d[k]; } } }
        let slots = super::mvar::save_objects(vp, std::path::Path::new(&dst), &objects, None)?;
        let identical = super::mvar::read_h4_expanded(vp).ok() == std::fs::read(&dst).ok();
        logln!("HMS_MVAR_SAVE: wrote {} objects -> {dst} (body v{}{}; {})", slots.len(), v.version,
            delta.map(|d| format!(", every object moved by {d:?}")).unwrap_or_default(),
            if identical { "byte-identical to the source" } else if delta.is_some() { "edited" } else { "DIFFERS from the source" });
    }

    // --- windowless wgpu device (same selection as headless::run) ---
    let instance = wgpu::Instance::default();
    let want_soft = std::env::var("HMS_SOFTWARE_GPU").is_ok();
    let request = |fallback: bool| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: fallback,
    }));
    let adapter = if want_soft { request(true) } else { request(false).or_else(|| request(true)) }
        .ok_or_else(|| anyhow!("no wgpu adapter (headless)"))?;
    let info = adapter.get_info();
    let has_bc = adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
    logln!("adapter: {:?} ({:?}, {:?}, BC={})", info.name, info.device_type, info.backend, has_bc);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("hms-headless-h4"),
            required_features: adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))?;
    let mut renderer = SceneRenderer::new(&device, &queue, (w, h));
    renderer.show_grid = false;

    // --- meshes: decode + upload through the shared builder, straight into the renderer lanes ---
    // HMS_SPAWNCAM needs the editor scene too (the raycast soup for the stand-off)
    let spawncam_wanted = std::env::var("HMS_SPAWNCAM").is_ok();
    // HMS_PREVIEW_SHOT needs the editor scene too (the palette preview runs through it)
    let preview_wanted = std::env::var("HMS_PREVIEW_SHOT").map(|v| !v.trim().is_empty()).unwrap_or(false);
    let editor_wanted = std::env::var("HMS_H4_SOUP").map(|v| v == "1").unwrap_or(false) || spawncam_wanted || preview_wanted;
    let mut spawn_cam: Option<super::spawncam::SpawnCam> = None;
    let mut editor_shadow: Option<(Option<(Vec3, Vec3)>, Option<hms_render::H4CascadeCfg>)> = None;
    let collect_out: Option<H4EditorCollect>;
    // kept alive past the build block for HMS_PREVIEW_SHOT (#h4-preview)
    let mut editor_scene: Option<super::edit_scene::H4ObjectScene> = None;
    let stats = {
        let mr = renderer.mesh_renderer_arc();
        let mut batches: Vec<(Lane, Vec<hms_render::GpuMesh>)> = Vec::new();
        let mut emit = |lane: Lane, v: Vec<hms_render::GpuMesh>| batches.push((lane, v));
        let mut progress = |_d: u32, _t: u32| {};
        let mut lg = |s: String| { eprintln!("{s}"); log.push_str(&s); log.push('\n'); };
        // HMS_H4_SOUP=1: build the editor scene too and draw the variant objects through its
        // dynamic lanes (the pixel-parity check of the editor path against the static one)
        let mut collect = editor_wanted.then(H4EditorCollect::new);
        let stats = build_meshes(&loaded, &device, &queue, &mr, &mut emit, &mut progress, None, &mut lg, collect.as_mut())?;
        for (lane, v) in batches {
            match lane {
                Lane::Opaque => renderer.append_static_meshes(v),
                Lane::AlphaTest => renderer.append_alphatest_meshes(v),
                Lane::Blend => renderer.append_blend_meshes(v),
                Lane::Additive => renderer.append_additive_meshes(v),
                Lane::Sky => renderer.set_sky_meshes(v.into_iter().map(|m| { let b = m.blend_mode(); let k = m.centroid()[0]; (m, b, k) }).collect()),
            }
        }
        collect_out = collect;
        stats
    };
    // Editor path: the variant objects through H4ObjectScene -> dynamic lanes
    if let Some(mut collect) = collect_out {
        // The scenario's spawn markers ride the renderer's marker lane, shown only with
        // HMS_MAP_SPAWNS=1 (the GUI's View toggle)
        let (mo, ma, mb) = super::scene::split_marker_meshes(std::mem::take(&mut collect.map_spawn_meshes));
        logln!("h4 map spawns: {} marker meshes (opaque {} additive {} blend {}; {})", mo.len() + ma.len() + mb.len(), mo.len(), ma.len(), mb.len(), if crate::map_spawns::env_show_map_spawns() { "drawn, HMS_MAP_SPAWNS=1" } else { "hidden" });
        renderer.set_marker_meshes(mo, ma, mb);
        renderer.set_show_markers(crate::map_spawns::env_show_map_spawns());
        let variant_n = loaded.variant_full.as_ref().map_or(0, |(_, v)| v.objects.len());
        let assets = H4EditorAssets::from_loaded(loaded, collect, &stats);
        let mut es = super::edit_scene::H4ObjectScene::new(assets, &device, &queue);
        let (objects, colors) = es.variant_editor_objects();
        let mr = renderer.mesh_renderer_arc();
        let mut lanes: Option<crate::objscene::RebuiltLanes> = None;
        let mut passes = 0;
        loop {
            if let Some(l) = crate::objscene::ObjectScene::maybe_rebuild(&mut es, &objects, &colors, &mr, &device, &queue) { lanes = Some(l); }
            passes += 1;
            if !crate::objscene::ObjectScene::rebuild_pending(&es) || passes > 10_000 { break; }
        }
        if let Some(l) = lanes {
            logln!("h4 editor: {} variant objects -> {} pick entries, dynamic lanes opaque {} cutout {} blend {} ({} rebuild passes; models {} decoded); object light sources [surface probe, airprobe, instance probe, mean bake] = {:?}", variant_n, es.pick_count(), l.opaque.len(), l.cutout.len(), l.blend.len(), passes, es.model_count(), es.light_source_tally());
            // the raycast soup + object pick from above the spawn camera (the editor's floor query)
            if let Some((sp, _, _)) = crate::objscene::ObjectScene::spawn_camera_pose(&es) {
                let o = sp + Vec3::new(0.0, 0.0, 5.0);
                let floor = crate::objscene::ObjectScene::raycast_scene_n(&es, o, Vec3::NEG_Z);
                let obj = crate::objscene::ObjectScene::pick(&es, o, Vec3::NEG_Z);
                logln!("h4 editor: soup {} tris; from {:?} straight down: BSP floor {:?}, object {:?}", es.soup_len(), o.to_array(), floor.map(|(p, n)| (p.to_array(), n.to_array())), obj.map(|d| format!("{d:#x}")));
            }
            renderer.set_dynamic_meshes(l.opaque);
            renderer.set_dynamic_cutout_meshes(l.cutout);
            renderer.set_dynamic_holo_meshes(l.holo);
            renderer.set_dynamic_holo_solid_meshes(l.holo_solid);
            renderer.set_dynamic_blend_meshes(l.blend);
            editor_shadow = crate::objscene::ObjectScene::h4_shadow_setup(&es);
        } else {
            logln!("h4 editor: {} variant objects, nothing rebuilt", variant_n);
        }
        // The GUI's post-load camera (spawncam.rs) over the same scene + soup
        if spawncam_wanted {
            spawn_cam = super::spawncam::spawn_camera(&es);
            match spawn_cam {
                Some(c) => logln!("HMS_SPAWNCAM: source={} pos=({:.2},{:.2},{:.2}) yaw={:.1} pitch={:.1} raw=({:.2},{:.2},{:.2}) stand-off moved {:.2} wu (soup {} tris)",
                    c.source.label(), c.pos.x, c.pos.y, c.pos.z, c.yaw.to_degrees(), c.pitch.to_degrees(), c.raw_pos.x, c.raw_pos.y, c.raw_pos.z, (c.pos - c.raw_pos).length(), es.soup_len()),
                None => logln!("HMS_SPAWNCAM: source=none (no variant camera / spawn, no scenario starting location / spawn placement / camera) - bounds overview"),
            }
        }
        editor_scene = Some(es);
    }

    // --- lighting: the Lbsp's baked sun direction when it carries one (else the placeholder) ---
    logln!("h4 sun: to-sun dir {:?} tint {:?} ({}; lightmap atlas {})", stats.sun_dir().to_array(), stats.sun_tint(), if stats.sun.is_some() { "Lbsp baked sun" } else { "placeholder" },
        if stats.lm_instances > 0 { "engine-law atlas decode" } else { "off - flat placeholder lighting" });
    renderer.set_sun_dir(stats.sun_dir());
    // Lbsp +0x30 sun colour: absolute HDR rgb with the atlas path (same units as its lobes),
    // hue only (max 1) for the flat placeholder lighting.
    renderer.set_scene_tints(stats.sun_tint(), [1.0; 3]);
    // The object sun shadow (forge-lightmap burn law) over every caster: the static scenario
    // objects + the editor scene's variant objects (HMS_H4_SOUP=1 path), and the cascade
    {
        let mut b = stats.caster_bounds;
        if let Some((Some(eb), _)) = editor_shadow {
            b = Some(match b { Some((a, c)) => (a.min(eb.0), c.max(eb.1)), None => eb });
        }
        renderer.set_h4_shadow(b, stats.cascade);
        logln!("h4 shadows: caster bounds {:?} cascade {:?}", b.map(|(a, c)| (a.to_array(), c.to_array())), stats.cascade);
    }
    // The map's atmosphere fog (lighting.rs); zero = none. HMS_H4_NOFOG=1 renders without it
    // (a per-term diagnostic, like HMS_NOFOG on the Reach path).
    renderer.set_fog(&queue, &if std::env::var("HMS_H4_NOFOG").is_ok() { [0.0; 28] } else { stats.fog.unwrap_or([0.0; 28]) });
    // cfxs: exposure band + engine meter, filmic composite, bloom, the colour-grading LUT and
    // the self-illum exposure law (lighting.rs `apply_post`, shared with the GUI + script host).
    // HMS_H4_POSTDBG bit mask (diag only): 1 = Reach meter (key/geomean) instead of the engine
    // meter, 2 = no filmic, 4 = no colour-grading LUT.
    let pdbg: u32 = std::env::var("HMS_H4_POSTDBG").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    super::lighting::apply_post(&mut renderer, &device, &queue, stats.camera_fx.as_ref(), if pdbg & 4 == 0 { stats.color_grading_lut.as_ref() } else { None });
    if let Some(x) = &stats.camera_fx {
        if pdbg & 1 != 0 { let (lo, hi) = x.gain_range(); renderer.set_h4_meter(&queue, None); renderer.set_exposure_band_abs(&queue, x.meter_key(), lo, hi); }
        if pdbg & 2 != 0 { renderer.set_h4_filmic(&queue, None); }
    }
    // HMS_EXPO_FIXED=<gain>: pin the exposure gain instead of metering (per-term diagnostic)
    if let Some(g) = std::env::var("HMS_EXPO_FIXED").ok().and_then(|s| s.parse::<f32>().ok()) { renderer.set_fixed_exposure(&queue, g); }

    // --- camera: frame the PLAYABLE clusters (skip vista-sized ones), else the whole scene ---
    let (fmin, fmax) = stats.frame_bounds();
    logln!("camera frames min {:?} max {:?} ({})", fmin.to_array(), fmax.to_array(), if fmin == Vec3::from(stats.bmin) && fmax == Vec3::from(stats.bmax) { "whole scene" } else { "playable clusters" });
    let mut cam = build_camera(fmin, fmax);
    if let (Some(c), Err(_)) = (spawn_cam, std::env::var("HMS_CAM")) {
        cam.pos = c.pos;
        cam.yaw = c.yaw;
        cam.pitch = c.pitch;
    }
    logln!("camera pos {:?} yaw {:.1} pitch {:.1} far {:.0}", cam.pos.to_array(), cam.yaw.to_degrees(), cam.pitch.to_degrees(), cam.far);
    for _ in 0..3 { renderer.render(&device, &queue, &cam, 0.0); }
    // print-only diag of the burn-map fit: texel (wu), depth range (wu), square side (wu)
    {
        let d = renderer.h4_shadow_diag.get();
        if d[3] > 0.5 {
            logln!("h4 shadows: burn map {}² texel {:.4} wu ({:.1} texels/wu; engine 40), depth range {:.1} wu (receiver bias 0.002 = {:.3} wu), footprint {:.1} wu square, 4x4 bilinear PCF = {:.3} wu penumbra (engine 0.075)",
                hms_render::SHADOW_MAP_SIZE, d[0], 1.0 / d[0], d[1], 0.002 * d[1], d[2], 3.0 * d[0]);
        } else {
            logln!("h4 shadows: no object casters - burn map off");
        }
    }
    // the engine meter's solved stops (LUM_WGSL h4_solve writes log2(key) - stops, key = 1)
    if let (Some(ml), Some(x)) = (renderer.read_mean_log(&device, &queue), &stats.camera_fx) {
        logln!("h4 exposure: solved {:.3} stops (band [{}, {}], screen brightness {:.3}) -> gain {:.3}", super::lighting::H4CameraFx::stops_of_gain((-ml).exp2()), x.exposure_range[0], x.exposure_range[1], x.screen_brightness, (-ml).exp2());
    }
    // HMS_HDR_DUMP=<path.f32>: the linear pre-tonemap HDR frame (RGBA f32, row-major) -
    // the only per-term probe that is not confounded by the exposure meter (same as headless.rs).
    if let Ok(hp) = std::env::var("HMS_HDR_DUMP") {
        let (hdr, hw, hh) = renderer.dump_hdr(&device, &queue);
        let mut bytes = Vec::with_capacity(hdr.len() * 4);
        for v in &hdr { bytes.extend_from_slice(&v.to_le_bytes()); }
        match std::fs::write(&hp, &bytes) {
            Ok(()) => logln!("HMS_HDR_DUMP: wrote {}x{} RGBA-f32 linear HDR to {}", hw, hh, hp),
            Err(e) => logln!("HMS_HDR_DUMP: write failed: {e}"),
        }
    }
    // HMS_BLOOM_DUMP=<path.f32>: the final bloom level (1/4 res RGBA f32) the composite
    // adds to scene * gain; with HMS_HDR_DUMP + the solved gain a CPU replica of the chain can be
    // checked against the GPU. Diag only.
    if let Ok(bp) = std::env::var("HMS_BLOOM_DUMP") {
        let (bl, bw, bh) = renderer.dump_bloom(&device, &queue);
        let mut bytes = Vec::with_capacity(bl.len() * 4);
        for v in &bl { bytes.extend_from_slice(&v.to_le_bytes()); }
        let n = (bw * bh) as f64;
        let mean: [f64; 3] = [0, 1, 2].map(|c| bl.chunks(4).map(|p| p[c] as f64).sum::<f64>() / n);
        match std::fs::write(&bp, &bytes) {
            Ok(()) => logln!("HMS_BLOOM_DUMP: wrote {}x{} RGBA-f32 bloom to {} (mean rgb {:.4} {:.4} {:.4})", bw, bh, bp, mean[0], mean[1], mean[2]),
            Err(e) => logln!("HMS_BLOOM_DUMP: write failed: {e}"),
        }
    }
    let (rgba, rw, rh) = renderer.capture_rgba(&device, &queue);
    // quick visibility stat: fraction of non-background pixels (any pixel differing from the corner)
    let bg = [rgba[0], rgba[1], rgba[2]];
    let n_px = (rw * rh) as usize;
    let mut differ = 0usize;
    let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
    for i in 0..n_px {
        let p = &rgba[i * 4..i * 4 + 3];
        if (p[0] as i32 - bg[0] as i32).abs() + (p[1] as i32 - bg[1] as i32).abs() + (p[2] as i32 - bg[2] as i32).abs() > 24 { differ += 1; }
        sr += p[0] as u64; sg += p[1] as u64; sb += p[2] as u64;
    }
    logln!("frame {}x{}: {:.1}% pixels differ from the corner colour {:?}; mean rgb ({:.0},{:.0},{:.0})",
        rw, rh, 100.0 * differ as f32 / n_px as f32, bg, sr as f32 / n_px as f32, sg as f32 / n_px as f32, sb as f32 / n_px as f32);
    crate::headless::write_png(out, &rgba, rw, rh)?;
    logln!("wrote {out} ({rw}x{rh}) in {:.2}s", t0.elapsed().as_secs_f32());
    // HMS_PREVIEW_SHOT=<item>:<png>[;<item>:<png>...] (#h4-preview): render the palette preview of
    // a Halo 4 palette row exactly as the GUI's palette panel does - a second square SceneRenderer
    // (no sky / grid) carrying this map's post chain (`apply_post`), sun and object-shadow fit,
    // `H4ObjectScene::build_preview_meshes` (the viewport's own object path) for one placement at
    // the shot camera position, framed by the shared orbit camera (`preview_camera`, default 3/4
    // view yaw 0.7 / pitch 0.5 / zoom 1). `<item>` is one of
    //   q<quota>[.<variant>]  the Forge palette quota / variant, decimal (the .mvar addressing)
    //   n<text>               the first row whose RAW entry / variant string id contains <text>
    //   <hex>                 a raw obje cache tag index, hex (0x prefix optional)
    // HMS_PREVIEW_SIZE=<px> overrides the square size, HMS_PREVIEW_ORBIT="yaw,pitch,zoom" the orbit.
    // The editor scene is built for this (like HMS_H4_SOUP=1), so the preview shot and the viewport
    // shot in the same run come from one scene.
    for spec in std::env::var("HMS_PREVIEW_SHOT").ok().iter().flat_map(|v| v.split(';')).map(str::trim).filter(|s| !s.is_empty()) {
        let Some((item, ppath)) = spec.rsplit_once(':') else { logln!("HMS_PREVIEW_SHOT '{spec}': expected <item>:<png>"); continue };
        let (item, ppath) = (item.trim(), ppath.trim());
        let Some(es) = editor_scene.as_mut() else { logln!("HMS_PREVIEW_SHOT '{spec}': no editor scene (the variant / palette did not load)"); continue };
        let obje: Option<usize> = if let Some(rest) = item.strip_prefix('q') {
            let mut it = rest.splitn(2, '.');
            let q: Option<u8> = it.next().and_then(|t| t.trim().parse().ok());
            let v: u8 = it.next().and_then(|t| t.trim().parse().ok()).unwrap_or(0);
            q.and_then(|q| es.resolve_item(q, Some(v)).map(|(t, _, _)| t))
        } else if let Some(name) = item.strip_prefix('n') {
            let needle = name.trim().to_ascii_lowercase();
            es.palette().iter().find_map(|e| e.variants.iter().find_map(|pv| {
                (e.name.to_ascii_lowercase().contains(&needle) || pv.name.to_ascii_lowercase().contains(&needle)).then_some(pv.tag).flatten()
            }))
        } else {
            usize::from_str_radix(item.trim_start_matches("0x"), 16).ok()
        };
        let Some(obje) = obje else { logln!("HMS_PREVIEW_SHOT '{item}': no such palette item / tag"); continue };
        let obj_tag = super::edit::h4_tag(obje);
        let psz: u32 = std::env::var("HMS_PREVIEW_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(280);
        let orbit: Vec<f32> = std::env::var("HMS_PREVIEW_ORBIT").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();
        let (pyaw, ppitch, pzoom) = (orbit.first().copied().unwrap_or(0.7), orbit.get(1).copied().unwrap_or(0.5), orbit.get(2).copied().unwrap_or(1.0));
        let mut pr = SceneRenderer::new(&device, &queue, (psz, psz));
        pr.show_sky = false;
        pr.show_grid = false;
        // the same per-map state the GUI mirrors into its preview renderer at map load (h4/gui.rs)
        super::lighting::apply_post_quiet(&mut pr, &device, &queue, stats.camera_fx.as_ref(), stats.color_grading_lut.as_ref());
        pr.set_sun_dir(stats.sun_dir());
        pr.set_scene_tints(stats.sun_tint(), [1.0; 3]);
        pr.set_h4_shadow(stats.caster_bounds, stats.cascade);
        // HMS_EXPO_FIXED pins the preview's gain too, so a preview and a viewport shot of the same
        // object can be compared without the per-frame meter (their frame content differs).
        if let Some(g) = std::env::var("HMS_EXPO_FIXED").ok().and_then(|s| s.parse::<f32>().ok()) { pr.set_fixed_exposure(&queue, g); }
        let mut built = Err(true);
        for _ in 0..64 {
            built = es.build_preview_meshes(obj_tag, 0, cam.pos, renderer.mesh_renderer(), &device, &queue);
            if !matches!(built, Err(true)) { break; }
        }
        match built {
            Ok((op, cut, holo, hs, bl, mn, mx)) => {
                logln!("HMS_PREVIEW_SHOT '{item}' obje={obje:#x} '{}': meshes opaque={} cutout={} holo={} holo_solid={} blend={} aabb=[{:.2},{:.2},{:.2}]-[{:.2},{:.2},{:.2}]",
                    crate::objscene::ObjectScene::tag_name_of(es, obj_tag), op.len(), cut.len(), holo.len(), hs.len(), bl.len(), mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
                pr.set_dynamic_meshes(op);
                pr.set_dynamic_cutout_meshes(cut);
                pr.set_dynamic_holo_meshes(holo);
                pr.set_dynamic_holo_solid_meshes(hs);
                pr.set_dynamic_blend_meshes(bl);
                let center = (mn + mx) * 0.5;
                let radius = ((mx - mn).length() * 0.5).max(0.5);
                let pcam = crate::preview_camera(center, mn, mx, radius, pyaw, ppitch, pzoom);
                // a few frames so the exposure meter settles like the live panel's periodic refresh
                for _ in 0..4 { pr.render(&device, &queue, &pcam, 0.0); }
                let (prgba, pw, ph) = pr.capture_rgba(&device, &queue);
                match crate::headless::write_png(ppath, &prgba, pw, ph) {
                    Ok(()) => logln!("HMS_PREVIEW_SHOT: wrote {ppath} ({pw}x{ph}) cam=[{:.2},{:.2},{:.2}] yaw={:.1} pitch={:.1}", pcam.pos.x, pcam.pos.y, pcam.pos.z, pcam.yaw.to_degrees(), pcam.pitch.to_degrees()),
                    Err(e) => logln!("HMS_PREVIEW_SHOT: write failed: {e}"),
                }
            }
            Err(inflight) => logln!("HMS_PREVIEW_SHOT '{item}' obje={obje:#x}: no preview meshes (decode in flight={inflight})"),
        }
    }
    // HMS_FRAMES=<n>: render n more frames with a moving camera (yaw sweep + small orbit, the same
    // motion as the Reach headless perf loop) through the renderer path the GUI runs per frame;
    // prints frame-time mean / p99 / max. Print-only.
    if let Some(n) = std::env::var("HMS_FRAMES").ok().and_then(|s| s.trim().parse::<usize>().ok()).filter(|&n| n > 0) {
        let base = cam.clone();
        let mut cam = cam.clone();
        let mut times: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32;
            cam.yaw = base.yaw + (f * 0.021).sin() * 0.35;
            cam.pitch = base.pitch + (f * 0.013).cos() * 0.08;
            cam.pos = base.pos + Vec3::new((f * 0.017).sin() * 3.0, (f * 0.011).cos() * 3.0, (f * 0.009).sin() * 1.0);
            let t = std::time::Instant::now();
            renderer.render(&device, &queue, &cam, f / 60.0);
            let _ = device.poll(wgpu::Maintain::Wait);
            times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = times.iter().sum::<f32>() / n as f32;
        let rss = std::fs::read_to_string("/proc/self/status").ok().and_then(|t| t.lines().find(|l| l.starts_with("VmRSS")).map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())).unwrap_or_default();
        logln!("HMS_FRAMES {} frames {}x{}: mean={:.2} p99={:.2} max={:.2} ms; RSS {} kB", n, w, h, mean, sorted[(n * 99 / 100).min(n - 1)], sorted[n - 1], rss);
    }
    let _ = std::fs::write(format!("{out}.log"), &log);
    Ok(())
}

/// HMS_CAM, else a 3/4 overview framed on the decoded geometry bounds (mirrors headless.rs).
pub fn build_camera(bmin: Vec3, bmax: Vec3) -> Camera {
    let mut cam = Camera::default();
    if let Some(fov) = std::env::var("HMS_FOV").ok().and_then(|s| s.trim().parse::<f32>().ok()) {
        if fov > 1.0 && fov < 179.0 { cam.fov_y = fov.to_radians(); }
    }
    if let Ok(s) = std::env::var("HMS_CAM") {
        let p: Vec<f32> = s.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        if p.len() == 5 {
            cam.pos = Vec3::new(p[0], p[1], p[2]);
            cam.yaw = p[3].to_radians();
            cam.pitch = p[4].to_radians();
            cam.far = 20000.0;
            return cam;
        }
    }
    let center = (bmin + bmax) * 0.5;
    let radius = ((bmax - bmin).length() * 0.5).max(5.0);
    let view = std::env::var("HMS_VIEW").unwrap_or_default();
    let dmul: f32 = std::env::var("HMS_DIST").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let dist = (radius / (cam.fov_y * 0.5).tan() * 1.15 + radius * 0.2) * dmul;
    let (dir, pitch) = match view.as_str() {
        "top" => (Vec3::new(0.0, 0.0, -1.0), -std::f32::consts::FRAC_PI_2 + 0.01),
        "front" => (Vec3::new(0.0, 1.0, -0.15), -0.15),
        _ => (Vec3::new(0.6, 0.6, -0.5).normalize(), -0.5),
    };
    cam.pos = center - dir * dist;
    let to = (center - cam.pos).normalize();
    cam.yaw = to.y.atan2(to.x);
    cam.pitch = if view == "top" { pitch } else { to.z.clamp(-1.0, 1.0).asin() };
    cam.far = (radius * 6.0).max(5000.0);
    cam
}
