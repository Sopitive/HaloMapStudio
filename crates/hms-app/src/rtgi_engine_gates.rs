//! Gates for the engine-mode real-time probe GI at the APPLICATION level (the pure math and the
//! shader live in `hms_render::engine_gi` / `rtgi` and have their own gates there).
//!
//! * `placed_blocks_*` runs the REAL edit path (`rtgi_dyn::DynCache` -> `build_dynamic` ->
//!   `RtGi::set_dynamic` / `invalidate_region` -> the probe-update compute shader) on a SOFTWARE adapter
//!   (Mesa llvmpipe through GL: no GPU, no window, no display; any other adapter is refused).
//! * `forge_world_sky_*` reads the sky list and the first-bounce scalers through the scene from the
//!   user's `forge_halo.map` (native library only, no GPU).
//! * `forge_world_probe_field_*` builds Forge World's tracer scene from the map's COLLISION BSP (the
//!   same CPU path the collision gates use, no renderer) and compares the probe field with the map's
//!   shipped airprobes, legacy sky model vs engine mode.
//!
//!   cargo test --release -p hms-app rtgi_engine_gates -- --ignored --test-threads=1 --nocapture

use std::time::Instant;

use eframe::wgpu;
use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_render::engine_gi::*;
use hms_render::rtgi::{RtGi, RtGiScene, REGION_PASSES};

use crate::rtgi_dyn::{build_dynamic, object_sig, set_sig, DynCache};
use crate::scene::SceneController;

/// A CPU (llvmpipe) device; hardware adapters are refused.
pub(crate) fn test_device() -> (wgpu::Device, wgpu::Queue) {
    for (k, v) in [
        ("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json"),
        ("EGL_PLATFORM", "surfaceless"),
        ("LIBGL_ALWAYS_SOFTWARE", "1"),
        ("GALLIUM_DRIVER", "llvmpipe"),
    ] {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, v);
        }
    }
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::GL, ..Default::default() });
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::GL)
        .into_iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
        .expect("no software (llvmpipe) GL adapter: these gates refuse hardware GPUs");
    eprintln!("rtgi gates: adapter {:?}", adapter.get_info().name);
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("rtgi-gates"), required_features: wgpu::Features::empty(), required_limits: adapter.limits(), memory_hints: wgpu::MemoryHints::MemoryUsage },
        None,
    ))
    .expect("device")
}

type Tri = [Vec3; 3];

/// A 96-record sky cap (z 0.09..0.95, bluish) + the sun record, like Forge World's list.
fn test_sky() -> SkyLights {
    let n = 96;
    let sky: Vec<SkyRecord> = (0..n)
        .map(|i| {
            let z = 0.09 + 0.86 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let p = i as f64 * 2.399963229728653;
            SkyRecord { dir: [(r * p.cos()) as f32, (r * p.sin()) as f32, z as f32], rgb: [0.7, 0.95, 1.2], omega: 3.14 / n as f32 }
        })
        .collect();
    let sun = SkyRecord { dir: [0.4, -0.3, 0.866], rgb: [1400.0, 1300.0, 1170.0], omega: 0.00215 };
    SkyLights { sky, sun: Some(sun), sun_1st: 0.3, sky_1st: 1.0 }
}

/// An infinite floor (albedo 0.45) under a probe volume, lit by `test_sky`.
fn floor_scene(bounds: ([f32; 3], [f32; 3])) -> RtGiScene {
    let h = 4000.0;
    let tris: Vec<Tri> = vec![
        [Vec3::new(-h, -h, 0.0), Vec3::new(h, -h, 0.0), Vec3::new(h, h, 0.0)],
        [Vec3::new(-h, -h, 0.0), Vec3::new(h, h, 0.0), Vec3::new(-h, h, 0.0)],
    ];
    let bvh = crate::lightbake::GpuBvh::build(&tris);
    let sky = test_sky();
    RtGiScene {
        nodes: bytemuck::cast_slice(&bvh.nodes).to_vec(),
        tris: bytemuck::cast_slice(&bvh.tris).to_vec(),
        tri_mat: vec![0; bvh.tris.len()],
        mats: vec![[0.45, 0.45, 0.45, 0.0], [0.0; 4], [0.0; 4]],
        lights: vec![],
        emitters: vec![],
        total_emitter_weight: 0.0,
        bounds_min: bounds.0,
        bounds_max: bounds.1,
        sun_dir: sky.sun.unwrap().dir,
        sun_color: sky.sun_irradiance(),
        sky_color: [0.5; 3],
        fog_sigma: 0.0,
        fog_top: 0.0,
        sky: Some(sky),
    }
}

fn block(o: &ObjectInfo) -> Vec<Tri> {
    let c = Vec3::from(o.pos);
    let p = |x: f32, y: f32, z: f32| c + Vec3::new(x, y, z) * 1.5; // a 3 wu cube hull
    let q = |a: Vec3, b: Vec3, cc: Vec3, d: Vec3| [[a, b, cc], [a, cc, d]];
    let f = [
        q(p(-1., -1., -1.), p(1., -1., -1.), p(1., 1., -1.), p(-1., 1., -1.)), q(p(-1., -1., 1.), p(1., -1., 1.), p(1., 1., 1.), p(-1., 1., 1.)),
        q(p(-1., -1., -1.), p(1., -1., -1.), p(1., -1., 1.), p(-1., -1., 1.)), q(p(-1., 1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(-1., 1., 1.)),
        q(p(-1., -1., -1.), p(-1., 1., -1.), p(-1., 1., 1.), p(-1., -1., 1.)), q(p(1., -1., -1.), p(1., 1., -1.), p(1., 1., 1.), p(1., -1., 1.)),
    ];
    f.iter().flatten().copied().collect()
}

fn piece(datum: u32, x: f32, y: f32) -> ObjectInfo {
    ObjectInfo { datum, type_sig: 0, sig0: 0, sig1: 0, pos: [x, y, 4.5], health: 1.0, shield: 1.0, mode_tag: 7, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: 7, variant_name_sid: 0 }
}

/// Sync the placed pieces into the tracer exactly as `App::rtgi_dyn_tick` does, then run the region passes.
fn sync(g: &mut RtGi, device: &wgpu::Device, queue: &wgpu::Queue, cache: &mut DynCache, objs: &[ObjectInfo]) -> usize {
    let sig = set_sig(objs, &|_| None);
    let delta = cache.refresh(objs, &|o| object_sig(o, None), &mut |o| block(o).into_iter().collect(), sig, Instant::now());
    let (tris, dropped) = cache.triangles();
    assert_eq!(dropped, 0);
    assert!(g.set_dynamic(queue, build_dynamic(&tris).as_ref()), "dynamic set refused");
    let mut queued = 0;
    for (lo, hi) in &delta.regions { queued += g.invalidate_region(*lo, *hi); }
    for _ in 0..REGION_PASSES {
        let mut enc = device.create_command_encoder(&Default::default());
        g.dispatch_part(&mut enc, queue, None, 1, 0);
        queue.submit(Some(enc.finish()));
    }
    queued
}

/// **THE EDIT GATE.** Three blocks are placed over a floor; one is moved 12 wu. Through the real glue
/// (`DynCache` -> `build_dynamic` -> `RtGi`), the probes around the block's old and new position change
/// (the sky it shaded comes back, the sky it now shades goes away) and every probe farther than two
/// spacings from both stays BIT-IDENTICAL.
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn placed_blocks_update_the_probes_they_reach_and_no_others() {
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MIN_SPACING", "3.0");
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &floor_scene(([-24.0, -24.0, 1.0], [24.0, 24.0, 5.0])), None, None);
    g.set_enabled(&queue, true);
    g.warm_up(&device, &queue, 12);
    let empty = g.read_probes(&device, &queue);
    let np = empty.len();
    assert!(np >= 200, "{np} probes");
    let (o, sp, d) = ([g.params.origin[0], g.params.origin[1], g.params.origin[2]], g.spacing, g.dims);

    let mut cache = DynCache::default();
    let mut objs = vec![piece(1, -12.0, -12.0), piece(2, 12.0, -12.0), piece(3, -12.0, 12.0)];
    let queued = sync(&mut g, &device, &queue, &mut cache, &objs);
    assert!(queued > 0 && queued < np, "first population queued {queued} of {np}");
    let before = g.read_probes(&device, &queue);

    // move block 3 across the volume
    let old = objs[2].pos;
    objs[2].pos = [12.0, 12.0, 4.5];
    let queued = sync(&mut g, &device, &queue, &mut cache, &objs);
    assert!(queued > 0 && queued < np / 2, "the move queued {queued} of {np} probes");
    let after = g.read_probes(&device, &queue);

    let pos = |i: usize| [o[0] + (i as u32 % d[0]) as f32 * sp, o[1] + ((i as u32 / d[0]) % d[1]) as f32 * sp, o[2] + (i as u32 / (d[0] * d[1])) as f32 * sp];
    let reach = |p: [f32; 3], c: [f32; 3]| (0..3).all(|a| (p[a] - c[a]).abs() <= 1.5 + 2.0 * sp + 1e-3);
    let (mut near_n, mut near_changed, mut far_n, mut far_same) = (0, 0, 0, 0);
    for i in 0..np {
        let p = pos(i);
        let same = (0..9).all(|k| (0..3).all(|c| before[i].total[k][c] == after[i].total[k][c]));
        if reach(p, old) || reach(p, objs[2].pos) {
            near_n += 1;
            if !same { near_changed += 1; }
        } else {
            far_n += 1;
            if same { far_same += 1; }
        }
    }
    eprintln!("moved one block: {near_n} probes in reach ({near_changed} changed), {far_n} outside ({far_same} bit-identical)");
    assert!(near_n >= 8 && far_n >= 50, "region {near_n}, outside {far_n}");
    assert_eq!(far_same, far_n, "{} probes outside the edit changed", far_n - far_same);
    assert!(near_changed * 2 >= near_n, "only {near_changed} of {near_n} probes in reach changed");
    // the sky the block used to shade is back above its old spot
    let under_old = (0..np).filter(|&i| { let p = pos(i); (p[0] - old[0]).abs() < 1.0 && (p[1] - old[1]).abs() < 1.0 && p[2] < old[2] - 1.5 }).collect::<Vec<_>>();
    assert!(!under_old.is_empty());
    for i in under_old {
        assert!(after[i].sky[0][1] > before[i].sky[0][1] * 1.15, "probe {i} under the old position did not get its sky back: {} -> {}", before[i].sky[0][1], after[i].sky[0][1]);
        // ...to within the grazing sky records the other two blocks (24 wu away, 6 wu tall) still clip: the
        // restored probe is far closer to the block-free field than to the shadowed one
        let diff = |a: &[[f32; 3]; 9], b: &[[f32; 3]; 9]| (0..9).flat_map(|k| (0..3).map(move |c| (a[k][c] - b[k][c]).abs())).fold(0.0f32, f32::max);
        let (restored, shadowed) = (diff(&after[i].sky, &empty[i].sky), diff(&before[i].sky, &empty[i].sky));
        assert!(restored < 0.04 * empty[i].sky[0][1] && restored < 0.25 * shadowed, "probe {i}: restored sky differs from the block-free field by {restored}, the shadowed one did by {shadowed}");
    }
}

/// The scene hands the tracer the map's real sky list (sun last and brightest) and the scalers 0.30/1.00.
#[test]
#[ignore = "opens the native library and the user's forge_halo.map"]
fn forge_world_sky_list_and_scalers_reach_the_probe_tracer() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
        eprintln!("rtgi gates SKIPPED: no forge_halo cache");
        return;
    };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test(); // also resolves the scenario tag the sky reads go through
    let k = sc.engine_sky_lights().expect("forge_halo has a sky list");
    assert_eq!(k.sky.len(), 102, "sky_halo: 102 sky records + the sun");
    assert!(k.sun.is_some() && k.sun_is_brightest());
    let (sun1, sky1) = sc.sky_bounce_scalers().expect("the sky tag carries the scalers");
    assert!((sun1 - 0.30).abs() < 1e-6 && (sky1 - 1.0).abs() < 1e-6, "scalers {sun1}/{sky1} (HREK sky_halo: 0.30 / 1.00)");
    assert_eq!((k.sun_1st, k.sky_1st), (sun1, sky1));
    let p = radiance_integral(&sky_sh_from_list(&k.sky, Some([0.0, 0.0, 1.0]), &|_| [1.0; 3]));
    assert!((p[0] - 2.191).abs() < 0.01 && (p[1] - 3.014).abs() < 0.01 && (p[2] - 3.824).abs() < 0.01, "P_up {p:?} (07: 2.19, 3.01, 3.82)");
    // `mode.flags` (cache +4) is 0x10 on Forge World: bit 3 clear = the last record IS the sun
    assert_eq!(sc.sky_flags(), Some(0x10), "sky_halo flags (HREK: 0x10)");
    eprintln!("forge_halo: {} sky records, scalers {sun1}/{sky1}, flags 0x10, P_up ({:.3},{:.3},{:.3})", k.sky.len(), p[0], p[1], p[2]);
}

/// Forge World's tracer scene from the map's COLLISION geometry (no renderer): every collision triangle
/// is an opaque occluder of one flat albedo (the collision surfaces carry no material), the sky is the
/// map's real light list, the probe volume is the airprobe cloud padded by `pad`.
fn forge_world_scene(sc: &mut SceneController, native: &hms_native::NativeDll, cache: hms_native::CacheHandle, albedo: f32, engine: bool) -> (RtGiScene, Vec<crate::rtgi_compare::ShippedProbe>, usize) {
    sc.load_collision_for_test();
    let mut tris: Vec<Tri> = Vec::new();
    sc.static_tris_in_aabb(Vec3::splat(-1.0e6), Vec3::splat(1.0e6), &mut tris);
    assert!(tris.len() > 10_000, "only {} collision triangles", tris.len());
    // The collision BSP also carries the playable-bounds shell (a prism whose lid sits at z = 1201 and whose
    // triangles are millions of wu^2): it is a boundary volume, not geometry, and would shut the sky out.
    let before = tris.len();
    tris.retain(|t| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length() < 2.0e5);
    // ...and Forge World's playable-space lid, a flat cap at z = 117.79 over the whole island
    tris.retain(|t| !t.iter().all(|p| (p.z - 117.79).abs() < 0.02));
    eprintln!("forge_halo collision: dropped {} boundary-shell triangles of {before}", before - tris.len());
    let scnr = native.find_scenario(cache).expect("scenario");
    let mut ship = Vec::new();
    for sbsp in native.enumerate_sbsps(cache, scnr, 64) {
        for p in native.airprobe_grid(cache, sbsp) {
            ship.push(crate::rtgi_compare::ShippedProbe { pos: p.pos, dom: p.dom_rgb, fill: p.fill_rgb, t: p.bandwidth, dir: p.dom_dir });
        }
    }
    assert!(ship.len() >= 100, "forge_halo ships 127 airprobes, found {}", ship.len());
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &ship { lo = lo.min(Vec3::from(p.pos)); hi = hi.max(Vec3::from(p.pos)); }
    let pad = Vec3::splat(12.0);
    let sky = sc.engine_sky_lights().expect("sky list");
    let n_tris = tris.len();
    let bvh = crate::lightbake::GpuBvh::build(&tris);
    // legacy sky = the old derivation: uniform radiance E_up / pi, sun = the brightest record (scene.rs sky_bake_params)
    let mut e_up = [0.0f32; 3];
    for r in &sky.sky { for c in 0..3 { e_up[c] += r.rgb[c] * r.omega * r.dir[2].max(0.0); } }
    let s = sky.sun.unwrap();
    (
        RtGiScene {
            nodes: bytemuck::cast_slice(&bvh.nodes).to_vec(),
            tris: bytemuck::cast_slice(&bvh.tris).to_vec(),
            tri_mat: vec![0; bvh.tris.len()],
            mats: vec![[albedo, albedo, albedo, 0.0], [0.0; 4], [0.0; 4]],
            lights: vec![],
            emitters: vec![],
            total_emitter_weight: 0.0,
            bounds_min: (lo - pad).into(),
            bounds_max: (hi + pad).into(),
            sun_dir: s.dir,
            sun_color: sky.sun_irradiance(),
            sky_color: [e_up[0] / std::f32::consts::PI, e_up[1] / std::f32::consts::PI, e_up[2] / std::f32::consts::PI],
            fog_sigma: 0.0,
            fog_top: 0.0,
            sky: if engine { Some(sky) } else { None },
        },
        ship,
        n_tris,
    )
}

/// Probe-field map for a human. For every (x, y) column of the grid, the LOWEST valid probe's up-facing
/// irradiance/pi (exact cosine convolution of its total SH); None where the column is solid.
fn field_columns(probes: &[hms_render::rtgi::ProbeRecord], d: [u32; 3]) -> Vec<Option<f64>> {
    (0..d[0] * d[1]).map(|c| {
        let (ix, iy) = (c % d[0], c / d[0]);
        (0..d[2]).find_map(|iz| {
            let r = &probes[(ix + iy * d[0] + iz * d[0] * d[1]) as usize];
            r.valid.then(|| { let e = sh_irradiance(&from_graphics_sh(&r.total), [0.0, 0.0, 1.0]); (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]) / std::f64::consts::PI })
        })
    }).collect()
}

/// Paint columns as an image (north up, 16 px per cell) with the shipped airprobes as red dots.
/// `colour(v)` maps a column value to RGB. `HMS_GATES_PNG_DIR=<dir>` writes them; nothing is written otherwise.
fn write_field_png(path: &std::path::Path, cols: &[Option<f64>], o: [f32; 3], sp: f32, d: [u32; 3], ship: &[crate::rtgi_compare::ShippedProbe], colour: &dyn Fn(f64) -> [u8; 3]) {
    const PX: u32 = 16;
    let (w, h) = (d[0] * PX, d[1] * PX);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for p in rgba.chunks_exact_mut(4) { p.copy_from_slice(&[24, 24, 28, 255]); }
    for iy in 0..d[1] { for ix in 0..d[0] {
        let Some(v) = cols[(ix + iy * d[0]) as usize] else { continue };
        let c = colour(v);
        for dy in 0..PX { for dx in 0..PX {
            let (x, y) = (ix * PX + dx, (d[1] - 1 - iy) * PX + dy);
            rgba[((y * w + x) * 4) as usize..][..4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        } }
    } }
    for s in ship {
        let (fx, fy) = ((s.pos[0] - o[0]) / sp, (s.pos[1] - o[1]) / sp);
        if fx < 0.0 || fy < 0.0 || fx >= d[0] as f32 || fy >= d[1] as f32 { continue; }
        let (px, py) = ((fx * PX as f32) as i64, ((d[1] as f32 - fy) * PX as f32) as i64);
        for dy in -2..=2i64 { for dx in -2..=2i64 {
            let (x, y) = (px + dx, py + dy);
            if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h { rgba[((y as u32 * w + x as u32) * 4) as usize..][..4].copy_from_slice(&[255, 40, 40, 255]); }
        } }
    }
    crate::headless::write_png(&path.to_string_lossy(), &rgba, w, h).expect("png");
}

/// Forge World's probe field against the map's shipped airprobes on the engine's own collision geometry,
/// BEFORE (legacy sky model: flat ambient, 1.3 gain) and AFTER (engine mode). Prints the agreement; asserts
/// liveness and the structural facts that hold whatever the shipped probes' radiometry is.
#[test]
#[ignore = "software GL + the user's forge_halo.map (a minute)"]
fn forge_world_probe_field_against_the_shipped_airprobes() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
        eprintln!("rtgi gates SKIPPED: no forge_halo cache");
        return;
    };
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MAX_PROBES", "6000");
    std::env::set_var("HMS_RTGI_MIN_SPACING", "4.0");
    let albedo: f32 = std::env::var("HMS_GATES_ALBEDO").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4);
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let mut stats = Vec::new();
    let mut columns = Vec::new();
    for (label, engine) in [("BEFORE (legacy sky model, no cube)", false), ("AFTER  (engine sky lights)", true)] {
        let t0 = Instant::now();
        let (scene, ship, n_tris) = forge_world_scene(&mut sc, &native, cache, albedo, engine);
        let mut g = RtGi::new(&device);
        g.set_scene(&device, &queue, &scene, None, None);
        g.set_enabled(&queue, true);
        assert_eq!(g.engine, engine);
        let t1 = Instant::now();
        g.warm_up(&device, &queue, 40);
        let probes = g.read_probes(&device, &queue);
        let o = [g.params.origin[0], g.params.origin[1], g.params.origin[2]];
        {
            let valid = probes.iter().filter(|p| p.valid).count();
            let dc = |f: &dyn Fn(&hms_render::rtgi::ProbeRecord) -> f32| { let mut v: Vec<f32> = probes.iter().filter(|p| p.valid).map(|p| f(p)).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); if v.is_empty() { [0.0; 3] } else { [v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10]] } };
            eprintln!("probes valid {valid}/{}; total DC g p10/50/90 {:?}; sky DC {:?}; gather DC {:?}", probes.len(), dc(&|p| p.total[0][1]), dc(&|p| p.sky[0][1]), dc(&|p| p.gather[0][1]));
        }
        if engine {
            let mut rows: Vec<(f32, [f32; 3], f64, f64, f64, f64)> = Vec::new();
            for s in &ship {
                let Some(sh) = crate::rtgi_compare::field_sh_at(&probes, o, g.spacing, g.dims, s.pos) else { continue };
                let pl = |c: [f32; 3]| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64;
                let ship_p = pl([s.dom[0] + s.fill[0], s.dom[1] + s.fill[1], s.dom[2] + s.fill[2]]);
                let p = radiance_integral(&sh);
                let hms_p = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                let f = fit_sh_to_dualvmf(&sh);
                let fp = f.dom.iter().zip(f.fill.iter()).map(|(a, b)| a + b).collect::<Vec<_>>();
                let fit_p = 0.2126 * fp[0] + 0.7152 * fp[1] + 0.0722 * fp[2];
                rows.push((s.pos[2], s.pos, ship_p, hms_p, fit_p, f.t));
            }
            rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            for r in rows.iter().take(6) { eprintln!("highest shipped probe z {:5.1} pos ({:6.0},{:6.0}): shipped P {:.3} | HMS SH P {:.3} fit P {:.3} t {:.2} | ratio {:.3}", r.0, r.1[0], r.1[1], r.2, r.3, r.4, r.5, r.2 / r.3.max(1e-9)); }
        }
        columns.push((field_columns(&probes, g.dims), o, g.spacing, g.dims, ship.clone()));
        let st = crate::rtgi_compare::compare(&ship, &probes, o, g.spacing, g.dims);
        eprintln!("{n_tris} collision tris, {} probes ({:.1} wu), scene {:.1} s, 40 passes {:.1} s, gain {}", probes.len(), g.spacing, (t1 - t0).as_secs_f32(), t1.elapsed().as_secs_f32(), g.params.bias[3]);
        eprintln!("{}", crate::rtgi_compare::report(label, &st));
        assert!(st.n >= 100, "{label}: only {} airprobes fell inside the probe grid", st.n);
        stats.push(st);
    }
    if let Some(dir) = std::env::var_os("HMS_GATES_PNG_DIR") {
        let dir = std::path::Path::new(&dir);
        let grey = |v: f64| { let g = ((v / 1.5).clamp(0.0, 1.0).sqrt() * 255.0) as u8; [g, g, g] };
        for (i, name) in ["forge_world_probe_field_legacy.png", "forge_world_probe_field_engine.png"].iter().enumerate() {
            let (c, o, sp, d, ship) = &columns[i];
            write_field_png(&dir.join(name), c, *o, *sp, *d, ship, &grey);
        }
        // engine / legacy up-facing irradiance per column: blue = engine darker, red = brighter (+-1 stop)
        let (c0, o, sp, d, ship) = &columns[0];
        let ratio: Vec<Option<f64>> = c0.iter().zip(&columns[1].0).map(|(a, b)| match (a, b) { (Some(a), Some(b)) if *a > 1e-6 => Some(b / a), _ => None }).collect();
        let heat = |r: f64| { let t = (r.max(1e-3).log2() / 1.0).clamp(-1.0, 1.0); if t < 0.0 { [(255.0 * (1.0 + t)) as u8, (255.0 * (1.0 + t)) as u8, 255] } else { [255, (255.0 * (1.0 - t)) as u8, (255.0 * (1.0 - t)) as u8] } };
        write_field_png(&dir.join("forge_world_probe_field_ratio_engine_over_legacy.png"), &ratio, *o, *sp, *d, ship, &heat);
    }
    let (before, after) = (&stats[0], &stats[1]);
    // What the engine sky model fixes, measured on the map's own probes (floors from the numbers above:
    // direction dot 0.22 -> 0.95, bandwidth error -0.64 -> +0.11 on Forge World with a flat 0.4 albedo):
    assert!(after.dir_dot_median >= 0.85 && after.dir_dot_median > before.dir_dot_median + 0.3,
        "dominant direction: median dot {:.3} (was {:.3} with the legacy sky)", after.dir_dot_median, before.dir_dot_median);
    assert!(after.t_diff_median.abs() < 0.25 && after.t_diff_median.abs() < before.t_diff_median.abs(),
        "bandwidth error {:+.3} (was {:+.3})", after.t_diff_median, before.t_diff_median);
    // What it does NOT fix, printed so nobody mistakes it for fixed: the shipped open-sky probes are ~1.7x
    // brighter than the sky list alone gives (the atlas is not), so the up-facing irradiance stays about half of
    // theirs in both modes. Unreconciled; see the engine_gi docs and the report.
    eprintln!("up-facing E/pi HMS/shipped median: legacy {:.2}, engine {:.2} (shipped open-sky probes are ~1.7x the sky-list radiometry: UNRESOLVED)", before.median_ratio, after.median_ratio);
}
