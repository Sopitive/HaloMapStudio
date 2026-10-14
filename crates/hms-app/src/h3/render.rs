//! Halo 3 (MCC) headless screenshot: `HMS_SHOT=<png> HMS_MAP=<halo3 map>` lands here from
//! `headless::run` when the map is a Halo 3 cache. The scene is built by scene.rs (the same
//! path the GUI load worker runs), rendered once through the wgpu renderer and written as a PNG
//! with a `<png>.log` beside it. Env: HMS_CAM / HMS_VIEW / HMS_DIST / HMS_FOV (shared with the
//! Reach / Halo 4 paths, via `h4::render::build_camera`) + the scene.rs diagnostics, plus the
//! two print/write-only exposure instruments the other two paths already carry: HMS_HDR_DUMP
//! (the linear pre-tonemap frame as RGBA f32) and HMS_EXPODIAG (the metered scene luminance
//! against the map's authored `cfxs` key). #h3-expo

use anyhow::{anyhow, Result};
use eframe::wgpu;
use hms_render::SceneRenderer;

use super::scene::{build_meshes, default_fog, placeholder_sun_dir, Lane};

/// Route one batch into the renderer lane it was built for (the GUI drain does the same).
pub fn append_lane(renderer: &mut SceneRenderer, lane: Lane, v: Vec<hms_render::GpuMesh>) {
    match lane {
        Lane::Opaque => renderer.append_static_meshes(v),
        Lane::AlphaTest => renderer.append_alphatest_meshes(v),
        Lane::Blend => renderer.append_blend_meshes(v),
        Lane::Additive => renderer.append_additive_meshes(v),
        Lane::Water => renderer.append_water_meshes(v),
        Lane::Terrain => renderer.append_terrain_meshes(v),
        Lane::Sky => renderer.set_sky_meshes(v.into_iter().map(|m| { let b = m.blend_mode(); let k = m.centroid()[0]; (m, b, k) }).collect()),
    }
}

pub fn headless_run(map_path: &str, out: &str, w: u32, h: u32) -> Result<()> {
    let mut log = String::new();
    macro_rules! logln { ($($a:tt)*) => {{ let s = format!($($a)*); eprintln!("{s}"); log.push_str(&s); log.push('\n'); }} }
    let t0 = std::time::Instant::now();

    // HMS_MVAR=<halo3 .mvar> places the variant's Forge objects (like the Reach / Halo 4 paths) #h3
    let variant = std::env::var("HMS_MVAR").ok().filter(|s| !s.trim().is_empty()).map(std::path::PathBuf::from);
    if let Some(vp) = &variant {
        if !super::mvar::is_h3_variant(vp) { anyhow::bail!("HMS_MVAR='{}' is not a Halo 3 map variant (mvar chunk v12)", vp.display()); }
    }
    let loaded = super::scene::load_map_with_variant(map_path, variant.as_deref())?;
    for l in &loaded.log { logln!("{l}"); }

    // --- windowless wgpu device ---
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
    logln!("adapter: {:?} ({:?}, {:?})", info.name, info.device_type, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("hms-headless-h3"),
            required_features: adapter.features() & wgpu::Features::TEXTURE_COMPRESSION_BC,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))?;
    let mut renderer = SceneRenderer::new(&device, &queue, (w, h));
    renderer.show_grid = false;

    // --- meshes: decode + upload through the shared builder, straight into the renderer lanes ---
    let stats = {
        let mr = renderer.mesh_renderer_arc();
        let mut batches: Vec<(Lane, Vec<hms_render::GpuMesh>)> = Vec::new();
        let mut emit = |lane: Lane, v: Vec<hms_render::GpuMesh>| batches.push((lane, v));
        let mut progress = |_d: u32, _t: u32| {};
        let mut lg = |s: String| { eprintln!("{s}"); log.push_str(&s); log.push('\n'); };
        let stats = build_meshes(&loaded, &device, &queue, &mr, &mut emit, &mut progress, None, &mut lg, None)?;
        for (lane, v) in batches { append_lane(&mut renderer, lane, v); }
        stats
    };

    // Sun + neutral tints, fixed exposure (no cfxs meter yet, so the engine auto-exposure is
    // bypassed - HMS_EXPO_FIXED overrides). With the bake bound every lit vertex carries alpha 0,
    // so the placeholder sun only reaches BSPs without a lightmap.
    renderer.set_sun_dir(placeholder_sun_dir());
    renderer.set_scene_tints([1.0; 3], [1.0; 3]);
    renderer.set_fog(&queue, &default_fog()); // replaced by the map's atmosphere below when it has one #h3-fog
    if let Some(mut f) = stats.fog {
        // HMS_H3_FOGTEST="<index>=<value>,..." (diagnostic): override fog uniform slots
        if let Ok(spec) = std::env::var("HMS_H3_FOGTEST") {
            for kv in spec.split(',') { if let Some((k, v)) = kv.split_once('=') { if let (Ok(k), Ok(v)) = (k.trim().parse::<usize>(), v.trim().parse::<f32>()) { if k < 28 { f[k] = v; } } } }
        }
        logln!("h3 fog uniform: {:?}", &f[..28]);
        renderer.set_fog(&queue, &f);
    }
    renderer.set_fixed_exposure(&queue, stats.expo);
    // the map's own post-process state: the Halo 3 tone curve + the authored bloom curve #h3-expo
    super::lighting::apply_post(&renderer, &queue, stats.camera_fx.as_ref());
    let (bmin, bmax) = stats.frame_bounds();
    let mut cam = crate::h4::render::build_camera(bmin, bmax);
    // land at a spawn like the GUI (spawn_camera.rs; no soup here, so the pose is used raw)
    if std::env::var("HMS_CAM").is_err() {
        match super::edit_scene::spawn_camera_for(&loaded.cache, loaded.variant.as_ref(), Some((bmin, bmax)), None, None) {
            Some(p) => { cam.pos = p.pos; cam.yaw = p.yaw; cam.pitch = p.pitch; logln!("h3 spawn camera: {}", p.describe()); }
            None => logln!("h3 spawn camera: none (bounds overview)"),
        }
    }
    logln!("camera pos {:?} yaw {:.1} pitch {:.1} far {:.0}", cam.pos.to_array(), cam.yaw.to_degrees(), cam.pitch.to_degrees(), cam.far);
    for _ in 0..3 { renderer.render(&device, &queue, &cam, 0.0); }

    // HMS_HDR_DUMP=<path.f32> (diagnostic, print/write-only): the linear pre-tonemap HDR frame
    // (RGBA f32, row-major) - the one measurement of the lit scene that the exposure chain cannot
    // confound. Mirrors the Reach (headless.rs) and Halo 4 (h4/render.rs) dumps. #h3-expo
    if let Ok(hp) = std::env::var("HMS_HDR_DUMP") {
        let (hdr, hw, hh) = renderer.dump_hdr(&device, &queue);
        let mut bytes = Vec::with_capacity(hdr.len() * 4);
        for v in &hdr { bytes.extend_from_slice(&v.to_le_bytes()); }
        match std::fs::write(&hp, &bytes) {
            Ok(()) => logln!("HMS_HDR_DUMP: wrote {}x{} RGBA-f32 linear HDR to {}", hw, hh, hp),
            Err(e) => logln!("HMS_HDR_DUMP: write failed: {e}"),
        }
    }
    // HMS_EXPODIAG (print-only): the 1x1 meter's mean log2 scene luminance against the map's
    // authored cfxs key, so "how far off is this map's exposure" is a number, not an impression.
    if std::env::var("HMS_EXPODIAG").is_ok() {
        if let Some(ml) = renderer.read_mean_log(&device, &queue) {
            let metered = ml.exp2();
            match stats.camera_fx.as_ref() {
                Some(x) => logln!("HMS_EXPODIAG mean_log2={ml:.3} metered_lum={metered:.5} vs cfxs key {:.5} -> the scene is {:.2}x off the map's own metering target",
                    x.key(), x.key() / metered.max(1e-9)),
                None => logln!("HMS_EXPODIAG mean_log2={ml:.3} metered_lum={metered:.5} (the scenario names no cfxs)"),
            }
        }
    }
    let (rgba, rw, rh) = renderer.capture_rgba(&device, &queue);
    let bg = [rgba[0], rgba[1], rgba[2]];
    let n_px = (rw * rh) as usize;
    let mut differ = 0usize;
    for i in 0..n_px {
        let p = &rgba[i * 4..i * 4 + 3];
        if (p[0] as i32 - bg[0] as i32).abs() + (p[1] as i32 - bg[1] as i32).abs() + (p[2] as i32 - bg[2] as i32).abs() > 24 { differ += 1; }
    }
    logln!("frame {}x{}: {:.1}% pixels differ from the corner colour {:?}", rw, rh, 100.0 * differ as f32 / n_px as f32, bg);
    crate::headless::write_png(out, &rgba, rw, rh)?;
    logln!("wrote {out} ({rw}x{rh}) in {:.2}s", t0.elapsed().as_secs_f32());
    let _ = std::fs::write(format!("{out}.log"), &log);
    Ok(())
}
