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
    std::env::set_var("HMS_RTGI_MAX_PROBES", "16384"); // (other gates of this module lower it; env is per process)
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
    forge_world_scene_geom(sc, native, cache, albedo, engine, None)
}

/// `forge_world_scene` with the occluder geometry chosen: `None` = the engine's collision world (the default stand-in),
/// `Some((tris, keep_collision, tri_mesh, mesh_albedo))` = these (render) triangles (each tagged with its mesh, whose loader albedo is `mesh_albedo[mesh]`; an empty `tri_mesh` = the flat `albedo`), plus the collision ones when `keep_collision`.
fn forge_world_scene_geom(sc: &mut SceneController, native: &hms_native::NativeDll, cache: hms_native::CacheHandle, albedo: f32, engine: bool, geom: Option<(&[Tri], bool, &[u32], &[[f32; 3]])>) -> (RtGiScene, Vec<crate::rtgi_compare::ShippedProbe>, usize) {
    sc.load_collision_for_test();
    let mut tris: Vec<Tri> = Vec::new();
    sc.static_tris_in_aabb(Vec3::splat(-1.0e6), Vec3::splat(1.0e6), &mut tris);
    assert!(tris.len() > 10_000, "only {} collision triangles", tris.len());
    // The collision BSP also carries the playable-bounds shell (a prism whose lid sits at z = 1201 and whose
    // triangles are millions of wu^2): it is a boundary volume, not geometry, and would shut the sky out.
    let before = tris.len();
    tris.retain(|t| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length() < 2.0e5);
    let scnr = native.find_scenario(cache).expect("scenario");
    let mut ship = Vec::new();
    for sbsp in native.enumerate_sbsps(cache, scnr, 64) {
        for p in native.airprobe_grid(cache, sbsp) {
            ship.push(crate::rtgi_compare::ShippedProbe { pos: p.pos, dom: p.dom_rgb, fill: p.fill_rgb, t: p.bandwidth, dir: p.dom_dir, mask: p.mask });
        }
    }
    assert!(ship.len() >= 30, "too few airprobes ({}) to compare against", ship.len());
    // The collision BSP also carries the playable space's LID (a large flat cap far above everything the
    // players reach) and the boundary walls hanging from it. They are volumes, not geometry: a lid shuts the
    // sky out of every probe. Found without a per-map constant: horizontal triangles of > 2e4 wu^2 more than 30 wu
    // above the highest airprobe; every triangle touching one of those levels goes.
    let top = ship.iter().fold(f32::MIN, |m, p| m.max(p.pos[2]));
    let mut lids: Vec<f32> = Vec::new();
    for t in &tris {
        let nrm = (t[1] - t[0]).cross(t[2] - t[0]);
        let area = 0.5 * nrm.length();
        if area > 2.0e4 && (nrm.z / nrm.length().max(1e-9)).abs() > 0.99 && t[0].z > top + 30.0 && !lids.iter().any(|z| (z - t[0].z).abs() < 0.05) { lids.push(t[0].z); }
    }
    tris.retain(|t| !t.iter().any(|p| lids.iter().any(|z| (p.z - z).abs() < 0.05)));
    let mut tri_mesh: Vec<u32> = Vec::new();
    let mut mesh_albedo: &[[f32; 3]] = &[];
    if let Some((extra, keep, tm, ma)) = geom {
        if !keep { tris.clear(); }
        let n0 = tris.len();
        tris.extend_from_slice(extra);
        if !tm.is_empty() { tri_mesh = vec![u32::MAX; n0]; tri_mesh.extend_from_slice(tm); mesh_albedo = ma; }
    }
    eprintln!("collision: dropped {} boundary-shell triangles of {before} (lid levels {lids:?})", before - tris.len());
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
            tri_mat: if tri_mesh.is_empty() { vec![0; bvh.tris.len()] } else { bvh.order.iter().map(|&o| match tri_mesh[o as usize] { u32::MAX => 0, m => m + 1 }).collect() },
            mats: { let mut m = vec![[albedo, albedo, albedo, 0.0], [0.0; 4], [0.0; 4]]; for a in mesh_albedo { m.extend_from_slice(&[[a[0], a[1], a[2], 0.0], [0.0; 4], [0.0; 4]]); } m },
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
    // What the engine sky model fixes, measured on the map's own probes (floors from the measured numbers on Forge World with a flat 0.4 albedo:
    // direction dot 0.44 -> 0.99, bandwidth error -0.64 -> -0.26):
    assert!(after.dir_dot_median >= 0.85 && after.dir_dot_median > before.dir_dot_median + 0.3,
        "dominant direction: median dot {:.3} (was {:.3} with the legacy sky)", after.dir_dot_median, before.dir_dot_median);
    assert!(after.t_diff_median.abs() < 0.35 && after.t_diff_median.abs() < before.t_diff_median.abs(),
        "bandwidth error {:+.3} (was {:+.3})", after.t_diff_median, before.t_diff_median);
    // What it does NOT fix, printed so nobody mistakes it for fixed: the shipped open-sky probes are ~1.7x
    // brighter than the sky list alone gives (the atlas is not), so the up-facing irradiance stays about half of
    // theirs in both modes. Unreconciled; see the engine_gi docs and the report.
    eprintln!("up-facing E/pi HMS/shipped median: legacy {:.2}, engine {:.2} (shipped open-sky probes are ~1.7x the sky-list radiometry: UNRESOLVED)", before.median_ratio, after.median_ratio);
}

/// THE SHADOW GATE on real data. The shipped airprobes carry the Tool's baked sun visibility (`mask`, the stored
/// sqrt form) at 127 points; the real-time mode's ray-traced cascades are asked for the same points. BEFORE this
/// round the mode shaded every point at full sun (visibility 1: no shadow at all), so the shadowed set it agreed
/// on was empty (IoU 0). The geometry is the engine's collision BSP minus its boundary shell and lid, which is a
/// proxy for the render geometry the Tool traced, so the floors below come from the measured numbers, not from 1.0.
#[test]
#[ignore = "software GL + the user's forge_halo.map (a minute)"]
fn forge_world_sun_shadow_against_the_shipped_airprobe_masks() {
    // HMS_GATES_MAP=<map stem> picks another Reach map (one per process: the native cache is shared)
    let stem = std::env::var("HMS_GATES_MAP").unwrap_or_else(|_| "forge_halo".into());
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(&stem) else {
        eprintln!("rtgi gates SKIPPED: no {stem} cache");
        return;
    };
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MAX_PROBES", "200");
    std::env::set_var("HMS_RTGI_MIN_SPACING", "40.0");
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let (scene, ship, _) = forge_world_scene(&mut sc, &native, cache, 0.4, true);
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &scene, None, None);
    g.set_enabled(&queue, true);
    let c = ship.iter().fold([0.0f32; 3], |a, p| [a[0] + p.pos[0] / ship.len() as f32, a[1] + p.pos[1] / ship.len() as f32, a[2] + p.pos[2] / ship.len() as f32]);
    let t0 = Instant::now();
    g.update_sun_now(&device, &queue, c);
    let pts: Vec<([f32; 3], [f32; 3])> = ship.iter().map(|p| (p.pos, [0.0, 0.0, 1.0])).collect();
    let vis = g.sample_sun_vis(&device, &queue, &pts, false);
    let masks: Vec<f32> = ship.iter().map(|p| p.mask).collect();
    let st = crate::rtgi_compare::mask_compare(&masks, &vis);
    let before = crate::rtgi_compare::mask_compare(&masks, &vec![1.0; masks.len()]);
    eprintln!("sun cascades traced in {:.1} s", t0.elapsed().as_secs_f32());
    for (p, (m, v)) in ship.iter().zip(masks.iter().zip(&vis)) { if (*m < 0.5) != (*v < 0.5) { eprintln!("DISAGREE pos ({:7.1},{:7.1},{:5.1}) shipped mask {m:.2} real-time {v:.2}", p.pos[0], p.pos[1], p.pos[2]); } }
    eprintln!("{}", crate::rtgi_compare::mask_report("BEFORE (no real-time shadow: visibility 1 everywhere)", &before));
    eprintln!("{}", crate::rtgi_compare::mask_report("AFTER  (ray-traced cascades)", &st));
    assert!(st.n >= 30 && st.shadow_a >= 3, "liveness: {} points, {} of them shadowed in the shipped bake", st.n, st.shadow_a);
    // measured on Forge World: 127 of 127 points agree, shadow IoU 1.000 (13 of 13 shadowed points found)
    assert!(st.agree >= 0.95 && st.shadow_iou >= 0.8 && st.shadow_iou > before.shadow_iou + 0.2,
        "sun shadow vs the shipped airprobe masks: agree {:.3}, IoU {:.3} (was {:.3})", st.agree, st.shadow_iou, before.shadow_iou);
}

/// HOW MUCH OF THE "less shadowed" LOOK IS THE PROBE GRID. The sky-direct SH of a probe is deterministic (every
/// sky record through the scene), so a probe grid of 1.25 wu spacing over a cliff-and-tower patch of Forge World is
/// the TRUTH for the sky light at any point of it; coarser grids of the same patch (what the viewer's ~16 wu grid does
/// at the scale of the whole island) are interpolated at the same points and compared. Reports the error of the
/// up-facing sky irradiance and how much of the light that truly sits in shadow the coarse grid fills in.
#[test]
#[ignore = "software GL + the user's forge_halo.map (a few minutes)"]
fn forge_world_probe_spacing_vs_sky_occlusion_truth() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
        eprintln!("rtgi gates SKIPPED: no forge_halo cache");
        return;
    };
    let (device, queue) = test_device();
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let (scene0, ship, _) = forge_world_scene(&mut sc, &native, cache, 0.4, true);
    drop(scene0);
    // a patch around the highest airprobe (a tower / cliff region)
    let hi = ship.iter().fold(ship[0], |a, p| if p.pos[2] > a.pos[2] { *p } else { a });
    let c = hi.pos;
    let (bmin, bmax) = ([c[0] - 20.0, c[1] - 20.0, c[2] - 28.0], [c[0] + 20.0, c[1] + 20.0, c[2] + 6.0]);
    let mut run = |spacing: f32| -> (Vec<hms_render::rtgi::ProbeRecord>, [f32; 3], f32, [u32; 3]) {
        std::env::set_var("HMS_RTGI_MIN_SPACING", format!("{spacing}"));
        std::env::set_var("HMS_RTGI_MAX_PROBES", "32768");
        let (mut s2, _, _) = forge_world_scene(&mut sc, &native, cache, 0.4, true);
        s2.bounds_min = bmin;
        s2.bounds_max = bmax;
        let mut g = RtGi::new(&device);
        g.set_scene(&device, &queue, &s2, None, None);
        g.set_enabled(&queue, true);
        g.warm_up(&device, &queue, 1);
        let p = g.read_probes(&device, &queue);
        (p, [g.params.origin[0], g.params.origin[1], g.params.origin[2]], g.spacing, g.dims)
    };
    let (fine, fo, fs, fd) = run(1.25);
    let up = |sh: &hms_render::engine_gi::Sh3| -> f64 { let e = sh_irradiance(sh, [0.0, 0.0, 1.0]); (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]) / std::f64::consts::PI };
    // truth: the fine probes themselves (valid ones, away from the patch border so coarse grids bracket them)
    let mut truth: Vec<([f32; 3], f64)> = Vec::new();
    for iz in 0..fd[2] { for iy in 0..fd[1] { for ix in 0..fd[0] {
        let r = &fine[(ix + iy * fd[0] + iz * fd[0] * fd[1]) as usize];
        if !r.valid { continue; }
        let p = [fo[0] + ix as f32 * fs + r.offset[0], fo[1] + iy as f32 * fs + r.offset[1], fo[2] + iz as f32 * fs + r.offset[2]];
        if (0..3).any(|a| p[a] < bmin[a] + 6.0 || p[a] > bmax[a] - 6.0) { continue; }
        if (ix + iy + iz) % 3 != 0 { continue; }
        truth.push((p, up(&hms_render::engine_gi::from_graphics_sh(&r.sky))));
    } } }
    let open = truth.iter().fold(0.0f64, |m, t| m.max(t.1));
    let dark: Vec<usize> = (0..truth.len()).filter(|&i| truth[i].1 < 0.5 * open).collect();
    eprintln!("patch around ({:.0},{:.0},{:.0}): {} truth points, open-sky up irradiance/pi {open:.3}, {} of them in shadow (< half of open)", c[0], c[1], c[2], truth.len(), dark.len());
    assert!(truth.len() > 200 && dark.len() > 20, "liveness: {} points, {} dark", truth.len(), dark.len());
    let mut worst_fill = 0.0f64;
    for spacing in [2.5f32, 5.0, 10.0, 16.5] {
        let (p, o, s, d) = run(spacing);
        let (mut within, mut n, mut errs, mut fill, mut nd) = (0usize, 0usize, Vec::new(), 0.0f64, 0usize);
        for (i, (pos, t)) in truth.iter().enumerate() {
            let Some(sh) = crate::rtgi_compare::field_sh_slot_at(&p, o, s, d, *pos, true) else { continue };
            let v = up(&sh);
            n += 1;
            let e = (v - t).abs() / open;
            errs.push(e);
            if (v / t.max(1e-6) - 1.0).abs() < 0.25 { within += 1; }
            if dark.contains(&i) { fill += (v - t) / open; nd += 1; }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let fill = fill / nd.max(1) as f64;
        worst_fill = worst_fill.max(fill);
        eprintln!("spacing {spacing:5.2} wu ({:?} probes): {n} points | error / open-sky median {:.3}, p90 {:.3} | within +-25% of the point's own truth {:.1}% | light leaked INTO truly shadowed points: +{:.3} of open-sky ({} points)", d, errs[errs.len() / 2], errs[errs.len() * 9 / 10], 100.0 * within as f64 / n as f64, fill, nd);
    }
    assert!(worst_fill >= 0.0);
}


/// WHERE the real-time field departs from the shipped probes, point by point (report). Both sides are the
/// up-facing diffuse irradiance/pi the viewer would show a +z-facing object at the airprobe: shipped = the engine
/// decode of its lobes + the analytic sun x baked mask^2; real-time = the field's total SH + the sun x traced
/// visibility^2. The global scale between them is the unresolved 1.6-2x probe radiometry, so the table normalises by
/// the MEDIAN ratio and lists the points that depart from it. `HMS_GATES_MAP=<stem>` picks the map.
#[test]
#[ignore = "software GL + a Reach map (a minute)"]
fn rtgi_vs_shipped_airprobes_where_it_departs() {
    let stem = std::env::var("HMS_GATES_MAP").unwrap_or_else(|_| "forge_halo".into());
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(&stem) else {
        eprintln!("rtgi gates SKIPPED: no {stem} cache");
        return;
    };
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MAX_PROBES", "32768");
    std::env::set_var("HMS_RTGI_MIN_SPACING", "1.5");
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let (scene, ship, n_tris) = forge_world_scene(&mut sc, &native, cache, 0.4, true);
    let sky = scene.sky.clone().expect("sky list");
    let sun_e = sky.sun_irradiance();
    let sun_l = sky.sun.map_or([0.0, 0.0, 1.0], |s| s.dir);
    let lights = native.scnr_simple_lights(cache, native.find_scenario(cache).unwrap());
    let stli = sc.stli_baked_positions();
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &scene, None, None);
    g.set_enabled(&queue, true);
    g.warm_up(&device, &queue, 48);
    let c = ship.iter().fold([0.0f32; 3], |a, p| [a[0] + p.pos[0] / ship.len() as f32, a[1] + p.pos[1] / ship.len() as f32, a[2] + p.pos[2] / ship.len() as f32]);
    g.update_sun_now(&device, &queue, c);
    let probes = g.read_probes(&device, &queue);
    let o = [g.params.origin[0], g.params.origin[1], g.params.origin[2]];
    let pts: Vec<([f32; 3], [f32; 3])> = ship.iter().map(|p| (p.pos, [0.0, 0.0, 1.0])).collect();
    let vis = g.sample_sun_vis(&device, &queue, &pts, false);
    let lum = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let sun_up = lum([sun_e[0] as f64, sun_e[1] as f64, sun_e[2] as f64]) * (sun_l[2].max(0.0) as f64) / std::f64::consts::PI;
    struct Row { pos: [f32; 3], ship_amb: f64, ship_tot: f64, hms_amb: f64, hms_tot: f64, sky_frac: f64, mask: f32, vis: f32, light_d: f32, stli_d: f32 }
    let mut rows: Vec<Row> = Vec::new();
    // open-sky reference: the largest sky-direct DC among valid probes
    let open_sky = probes.iter().filter(|p| p.valid).map(|p| p.sky[0][1]).fold(0.0f32, f32::max).max(1e-6);
    for (i, s) in ship.iter().enumerate() {
        let Some(sh) = crate::rtgi_compare::field_sh_at(&probes, o, g.spacing, g.dims, s.pos) else { continue };
        let sky_dc = crate::rtgi_compare::field_sh_slot_at(&probes, o, g.spacing, g.dims, s.pos, true).map_or(0.0, |k| k[1][0] as f64);
        let theirs = DualVmf { dir: [s.dir[0] as f64, s.dir[1] as f64, s.dir[2] as f64], kappa: kappa_of_t(s.t as f64), t: s.t as f64, dom: [s.dom[0] as f64, s.dom[1] as f64, s.dom[2] as f64], fill: [s.fill[0] as f64, s.fill[1] as f64, s.fill[2] as f64], err: 0.0 };
        let sa = lum(diffuse_from_dual_vmf([0.0, 0.0, 1.0], &theirs));
        let ha = lum(sh_irradiance(&sh, [0.0, 0.0, 1.0]).map(|v| v / std::f64::consts::PI));
        if sa < 1e-4 { continue; }
        let ld = lights.iter().map(|l| ((l.pos[0] - s.pos[0]).powi(2) + (l.pos[1] - s.pos[1]).powi(2) + (l.pos[2] - s.pos[2]).powi(2)).sqrt()).fold(f32::MAX, f32::min);
        let sd = stli.iter().map(|l| ((l.0[0] - s.pos[0]).powi(2) + (l.0[1] - s.pos[1]).powi(2) + (l.0[2] - s.pos[2]).powi(2)).sqrt()).fold(f32::MAX, f32::min);
        rows.push(Row { pos: s.pos, ship_amb: sa, ship_tot: sa + sun_up * (s.mask as f64).powi(2), hms_amb: ha, hms_tot: ha + sun_up * (vis[i] as f64).powi(2), sky_frac: sky_dc / open_sky as f64, mask: s.mask, vis: vis[i], light_d: ld, stli_d: sd });
    }
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let mut ra: Vec<f64> = rows.iter().map(|r| r.hms_amb / r.ship_amb).collect();
    let mut rt: Vec<f64> = rows.iter().map(|r| r.hms_tot / r.ship_tot).collect();
    let (ma, mt) = (med(&mut ra), med(&mut rt));
    eprintln!("{stem}: {} airprobes compared, {n_tris} collision tris, {} scenario simple lights; median HMS/shipped ambient {ma:.3}, total (ambient + sun) {mt:.3}", rows.len(), lights.len());
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&a, &b| (rows[a].hms_tot / rows[a].ship_tot).partial_cmp(&(rows[b].hms_tot / rows[b].ship_tot)).unwrap());
    let (dark_n, bright_n) = (idx.iter().filter(|&&i| rows[i].hms_tot / rows[i].ship_tot / mt < 0.6).count(), idx.iter().filter(|&&i| rows[i].hms_tot / rows[i].ship_tot / mt > 1.6).count());
    eprintln!("  darker than the median by > 1/0.6: {dark_n}; brighter by > 1.6x: {bright_n}");
    let show = |label: &str, it: Vec<usize>| {
        for i in it { let r = &rows[i]; eprintln!("  {label} ({:7.1},{:7.1},{:5.1}) ratio/median total {:.2} (amb {:.2}) | ship amb {:.3} tot {:.3} hms amb {:.3} tot {:.3} | sky-direct {:.2} of open | sun mask ship {:.2} rt {:.2} | nearest stli (baked) light {:.0} wu", r.pos[0], r.pos[1], r.pos[2], r.hms_tot / r.ship_tot / mt, r.hms_amb / r.ship_amb / ma, r.ship_amb, r.ship_tot, r.hms_amb, r.hms_tot, r.sky_frac, r.mask, r.vis, r.stli_d.min(9999.0)); }
    };
    // is the shortfall where the baked stli lights are? (RTGI gives them to probes as bounce sources only)
    let (mut near, mut far): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for r in &rows { if r.stli_d < 20.0 { near.push(r.hms_amb / r.ship_amb); } else { far.push(r.hms_amb / r.ship_amb); } }
    if !near.is_empty() && !far.is_empty() { eprintln!("  ambient HMS/shipped within 20 wu of a baked stli light: median {:.3} ({} probes); elsewhere {:.3} ({} probes)", med(&mut near.clone()), near.len(), med(&mut far.clone()), far.len()); }
    show("DARK  ", idx.iter().take(6).copied().collect());
    show("BRIGHT", idx.iter().rev().take(6).copied().collect());
}

/// SURFACE AUDIT of the real-time mode on a real map's geometry (report + liveness): for a few thousand points on
/// the level's own surfaces (collision triangles, offset along their sun-facing normal) it measures
///  (1) DARK-BY-CONSTRUCTION: points none of whose 8 surrounding probes is valid (the field returns black there);
///  (2) SHADOW ERROR: the shadow-map visibility against 32 real sun-disc rays through the same BVH, split into
///      acne (map says shadowed, rays say lit: too DARK) and leaks (map lit, rays shadowed: too BRIGHT), per cascade;
///  (3) the map's light inventory (what RTGI is and is not given).
#[test]
#[ignore = "software GL + a Reach map (a minute)"]
fn rtgi_surface_audit() {
    let stem = std::env::var("HMS_GATES_MAP").unwrap_or_else(|_| "forge_halo".into());
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(&stem) else {
        eprintln!("rtgi gates SKIPPED: no {stem} cache");
        return;
    };
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MAX_PROBES", "32768");
    std::env::set_var("HMS_RTGI_MIN_SPACING", "1.5");
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let (scene, ship, _) = forge_world_scene(&mut sc, &native, cache, 0.4, true);
    let sky = scene.sky.clone().unwrap();
    let l = sky.sun.map_or([0.0, 0.0, 1.0], |s| { let n = (s.dir[0] * s.dir[0] + s.dir[1] * s.dir[1] + s.dir[2] * s.dir[2]).sqrt(); [s.dir[0] / n, s.dir[1] / n, s.dir[2] / n] });
    let (nl, nb, nr, nspot) = sc.stli_light_inventory();
    let lights = native.scnr_simple_lights(cache, native.find_scenario(cache).unwrap());
    eprintln!("{stem} light inventory: {} scenario simple lights (runtime, per-pixel), stli {nl} lights ({nb} bake their direct term into the lightmap [{nspot} omni/spot], {nr} runtime-only analytic), sky list {} records + sun", lights.len(), sky.sky.len());
    let mut tris: Vec<[Vec3; 3]> = Vec::new();
    sc.static_tris_in_aabb(Vec3::splat(-1.0e6), Vec3::splat(1.0e6), &mut tris);
    let mut g = RtGi::new(&device);
    let (bmin, bmax) = (scene.bounds_min, scene.bounds_max);
    g.set_scene(&device, &queue, &scene, None, None);
    g.set_enabled(&queue, true);
    g.warm_up(&device, &queue, 8);
    let c = ship.iter().fold([0.0f32; 3], |a, p| [a[0] + p.pos[0] / ship.len() as f32, a[1] + p.pos[1] / ship.len() as f32, a[2] + p.pos[2] / ship.len() as f32]);
    g.update_sun_now(&device, &queue, c);
    let probes = g.read_probes(&device, &queue);
    let (o, sp, d) = ([g.params.origin[0], g.params.origin[1], g.params.origin[2]], g.spacing, g.dims);
    // sample points: triangle centroids inside the probe volume, offset along the normal that faces the sun
    let mut pts: Vec<([f32; 3], [f32; 3])> = Vec::new();
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 11) as f64 / (1u64 << 53) as f64 };
    for _ in 0..400_000 {
        if pts.len() >= 6000 { break; }
        let t = &tris[(next() * tris.len() as f64) as usize % tris.len()];
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        if n.length() < 1e-6 { continue; }
        let mut n = n.normalize();
        if n.dot(Vec3::from(l)) < 0.0 { n = -n; }
        if n.dot(Vec3::from(l)) < 0.15 { continue; }
        let p = (t[0] + t[1] + t[2]) / 3.0;
        if (0..3).any(|a| p[a] < bmin[a] || p[a] > bmax[a]) { continue; }
        pts.push(((p + n * 0.02).into(), n.into()));
    }
    assert!(pts.len() >= 2000, "only {} surface points", pts.len());
    // (1) black holes
    let corners_valid = |p: [f32; 3], n: [f32; 3]| -> usize {
        let q = [p[0] + n[0] * 0.2 * sp, p[1] + n[1] * 0.2 * sp, p[2] + n[2] * 0.2 * sp];
        let gc: Vec<f32> = (0..3).map(|a| ((q[a] - o[a]) / sp).clamp(0.0, d[a] as f32 - 1.0)).collect();
        let base: Vec<u32> = (0..3).map(|a| (gc[a].floor() as u32).min(d[a].saturating_sub(2))).collect();
        (0..8u32).filter(|k| { let c = [base[0] + (k & 1), base[1] + ((k >> 1) & 1), base[2] + ((k >> 2) & 1)]; c[0] < d[0] && c[1] < d[1] && c[2] < d[2] && probes[(c[0] + c[1] * d[0] + c[2] * d[0] * d[1]) as usize].valid }).count()
    };
    let mut hist = [0usize; 9];
    for (p, n) in &pts { hist[corners_valid(*p, *n)] += 1; }
    let tot = pts.len() as f64;
    eprintln!("{stem}: field coverage of {} surface points (probe spacing {sp:.1} wu): 0 valid corners {:.1}% (black), 1-2 {:.1}%, 3-5 {:.1}%, 6-8 {:.1}%", pts.len(), 100.0 * hist[0] as f64 / tot, 100.0 * (hist[1] + hist[2]) as f64 / tot, 100.0 * (hist[3] + hist[4] + hist[5]) as f64 / tot, 100.0 * (hist[6] + hist[7] + hist[8]) as f64 / tot);
    // (2) shadow map vs rays
    let truth = g.trace_sun_fraction(&device, &queue, &pts);
    let map: Vec<f32> = g.sample_sun_vis(&device, &queue, &pts, false).iter().map(|v| v * v).collect();
    let tex = |c: usize| 2.0 * SUN_HALF_PUB[c] / 1024.0;
    let (mut acne, mut leak, mut bias, mut n_in) = ([0usize; 5], [0usize; 5], [0.0f64; 5], [0usize; 5]);
    let (bx, by) = hms_render::rtgi::sun_basis(l);
    for (i, (p, _)) in pts.iter().enumerate() {
        let (pu, pv) = (p[0] * bx[0] + p[1] * bx[1] + p[2] * bx[2], p[0] * by[0] + p[1] * by[1] + p[2] * by[2]);
        let (cu, cv) = (c[0] * bx[0] + c[1] * bx[1] + c[2] * bx[2], c[0] * by[0] + c[1] * by[1] + c[2] * by[2]);
        let k = (0..4).find(|&k| ((pu - cu).abs() / SUN_HALF_PUB[k]).max((pv - cv).abs() / SUN_HALF_PUB[k]) < 0.9).unwrap_or(4);
        n_in[k] += 1;
        bias[k] += (map[i] - truth[i]) as f64;
        if map[i] < 0.2 && truth[i] > 0.8 { acne[k] += 1; }
        if map[i] > 0.8 && truth[i] < 0.2 { leak[k] += 1; }
    }
    for k in 0..5 {
        if n_in[k] == 0 { continue; }
        let label = if k < 4 { format!("cascade {k} (texel {:.3} wu)", tex(k)) } else { "outside the cascades (lit by construction)".to_string() };
        eprintln!("  shadow map vs 32 rays, {label}: {} points | mean (map - rays) {:+.3} | too DARK (map<0.2, rays>0.8) {:.2}% | too BRIGHT (map>0.8, rays<0.2) {:.2}%", n_in[k], bias[k] / n_in[k] as f64, 100.0 * acne[k] as f64 / n_in[k] as f64, 100.0 * leak[k] as f64 / n_in[k] as f64);
    }
}
const SUN_HALF_PUB: [f32; 4] = hms_render::rtgi::SUN_HALF;

/// SHADOW FILL vs the shipped atlas, point by point. For atlas texels the shipped bake marks as in sun shadow
/// (DM visibility ~0) the shipped diffuse is the lightmap irradiance alone, E_lm(n) = dual_vmf_diffuse. The real-time
/// mode's diffuse there is `gi_irradiance` (+ whatever sun the traced cascades let through), the very WGSL the
/// mesh shader runs, so the ratio is the ratio of the two linear radiances before albedo / tonemap. Prints the
/// distribution, where it is worst, and the share carried by each term (each switched off in the shader text in turn).
/// `HMS_GATES_MAP=<stem>` picks the map; `HMS_GATES_SPACING` the minimum probe spacing (default 1.5 -> the cap).
#[test]
#[ignore = "software GL + a Reach map (minutes)"]
fn rtgi_shadow_fill_vs_the_shipped_atlas() {
    let stem = std::env::var("HMS_GATES_MAP").unwrap_or_else(|_| "forge_halo".into());
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(&stem) else {
        eprintln!("rtgi gates SKIPPED: no {stem} cache");
        return;
    };
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MAX_PROBES", "32768");
    std::env::set_var("HMS_RTGI_MIN_SPACING", std::env::var("HMS_GATES_SPACING").unwrap_or_else(|_| "1.5".into()));
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let mesh_albedo = sc.load_lm_geom_for_test(&device, &queue);
    eprintln!("atlas-lit meshes decoded headlessly: {}", mesh_albedo.len());
    // HMS_GATES_GEOM = render (default: the atlas-lit render meshes only, with their loader albedo) | collision | both
    let geom_mode = std::env::var("HMS_GATES_GEOM").unwrap_or_else(|_| "render".into());
    let mut tri_mesh: Vec<u32> = Vec::new();
    let render_tris: Vec<Tri> = sc.lm_geom().iter().enumerate().flat_map(|(mi, m)| m.indices.chunks_exact(3).filter_map(|t| {
        let (a, b, c) = (m.verts.get(t[0] as usize)?, m.verts.get(t[1] as usize)?, m.verts.get(t[2] as usize)?);
        Some(([Vec3::from(a.0), Vec3::from(b.0), Vec3::from(c.0)], mi as u32))
    }).collect::<Vec<_>>()).map(|(t, mi)| { tri_mesh.push(mi); t }).collect();
    let (scene, ship, n_occ) = match geom_mode.as_str() {
        "render" => forge_world_scene_geom(&mut sc, &native, cache, 0.4, true, Some((&render_tris, false, &tri_mesh, &mesh_albedo))),
        "both" => forge_world_scene_geom(&mut sc, &native, cache, 0.4, true, Some((&render_tris, true, &tri_mesh, &mesh_albedo))),
        _ => forge_world_scene(&mut sc, &native, cache, 0.4, true),
    };
    // inner-grid configurations measured on the patch below: the shipped default, plus any in HMS_GATES_INNER="s:nx:ny:nz,..."
    let mut inner_cfgs: Vec<(String, hms_render::rtgi::InnerPolicy)> = vec![("default (Auto)".into(), hms_render::rtgi::InnerPolicy::Auto)];
    if let Ok(v) = std::env::var("HMS_GATES_INNER") {
        for c in v.split(',') {
            let f: Vec<f32> = c.split(':').filter_map(|x| x.parse().ok()).collect();
            if f.len() == 4 { inner_cfgs.push((format!("{c}"), hms_render::rtgi::InnerPolicy::Fixed(hms_render::rtgi::InnerGrid { spacing: f[0], dims: [f[1] as u32, f[2] as u32, f[3] as u32] }))); }
        }
    }
    let mut inner_scenes: Vec<RtGiScene> = if stem == "forge_halo" {
        inner_cfgs.iter().map(|_| match geom_mode.as_str() {
            "render" => forge_world_scene_geom(&mut sc, &native, cache, 0.4, true, Some((&render_tris, false, &tri_mesh, &mesh_albedo))).0,
            _ => forge_world_scene(&mut sc, &native, cache, 0.4, true).0,
        }).collect()
    } else { Vec::new() };
    eprintln!("occluder geometry: {geom_mode} ({n_occ} triangles; {} render triangles available)", render_tris.len());
    let mut scene = scene;
    if let (Some(k), Some(sky)) = (std::env::var("HMS_GATES_SUN1ST").ok().and_then(|v| v.parse::<f32>().ok()), scene.sky.as_mut()) { eprintln!("experiment: sun first-bounce scaler {} -> {}", sky.sun_1st, k); sky.sun_1st = k; }
    let (bmin, bmax) = (scene.bounds_min, scene.bounds_max);
    let mut g = RtGi::new(&device);
    g.inner_policy = hms_render::rtgi::InnerPolicy::Off; // the single-grid baseline every figure above is measured with
    g.set_scene(&device, &queue, &scene, None, None);
    g.set_enabled(&queue, true);
    g.warm_up(&device, &queue, 8);
    let c = ship.iter().fold([0.0f32; 3], |a, p| [a[0] + p.pos[0] / ship.len() as f32, a[1] + p.pos[1] / ship.len() as f32, a[2] + p.pos[2] / ship.len() as f32]);
    g.update_sun_now(&device, &queue, c);
    // the sun the traced cascades use (the loader's scene_sun_dir is not set on this renderer-less path)
    let sun_dir = Vec3::from(scene.sun_dir).normalize_or_zero();
    let sun_col = scene.sun_color;
    // atlas sample points: random barycentric points of the atlas-lit meshes
    let geom = sc.lm_geom();
    let mut tri_list: Vec<(u32, u32)> = Vec::new();
    for (mi, m) in geom.iter().enumerate() { for t in 0..(m.indices.len() / 3) { tri_list.push((mi as u32, t as u32)); } }
    assert!(tri_list.len() > 1000, "only {} atlas triangles", tri_list.len());
    let mut rng = 0x2545F4914F6CDD1Du64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 11) as f64 / (1u64 << 53) as f64 };
    let mut imgs: std::collections::HashMap<usize, Option<crate::scene::LmAtlasImages>> = Default::default();
    let lum3_ = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    struct Pt { p: [f32; 3], n: [f32; 3], e_lm: [f32; 3], vis: f32, mi: usize, fill: f32, dom: f32, hdr: f32, k: f32 }
    let (mut shadow, mut lit): (Vec<Pt>, Vec<Pt>) = (Vec::new(), Vec::new());
    let mut dead = 0usize;
    for _ in 0..3_000_000 {
        if shadow.len() >= 8000 && lit.len() >= 2000 { break; }
        let (mi, ti) = tri_list[(next() * tri_list.len() as f64) as usize % tri_list.len()];
        let m = &geom[mi as usize];
        let t = &m.indices[ti as usize * 3..ti as usize * 3 + 3];
        let (Some(v0), Some(v1), Some(v2)) = (m.verts.get(t[0] as usize), m.verts.get(t[1] as usize), m.verts.get(t[2] as usize)) else { continue };
        let (mut a, mut b) = (next() as f32, next() as f32);
        if a + b > 1.0 { a = 1.0 - a; b = 1.0 - b; }
        let w = [1.0 - a - b, a, b];
        let pos = Vec3::from(v0.0) * w[0] + Vec3::from(v1.0) * w[1] + Vec3::from(v2.0) * w[2];
        let n = (Vec3::from(v0.1) * w[0] + Vec3::from(v1.1) * w[1] + Vec3::from(v2.1) * w[2]).normalize_or_zero();
        if n.length_squared() < 0.5 || (0..3).any(|k| pos[k] < bmin[k] || pos[k] > bmax[k]) { continue; }
        let uv = [v0.2[0] * w[0] + v1.2[0] * w[1] + v2.2[0] * w[2], v0.2[1] * w[0] + v1.2[1] * w[1] + v2.2[1] * w[2]];
        let Some(im) = imgs.entry(mi as usize).or_insert_with(|| sc.shipped_lm_images(m.dm, m.sdm)) else { continue };
        let st = crate::scene::shipped_lm_texel(im, m.hdr, m.k, uv);
        let e = crate::lightprobe::dual_vmf_diffuse(n, Vec3::from(st.dom_dir), st.dom_rgb, st.fill_rgb, st.bandwidth);
        let lum = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
        if !(lum > 1e-5) { dead += 1; continue; }
        if lum < 0.01 && st.vis < 0.05 { dead += 1; continue; }
        let vis = st.vis.clamp(0.0, 1.0);
        let pt = Pt { p: (pos + n * 0.05).into(), n: n.into(), e_lm: e, vis, mi: mi as usize, fill: lum3_(st.fill_rgb), dom: lum3_(st.dom_rgb), hdr: m.hdr, k: m.k };
        if vis < 0.05 { if shadow.len() < 8000 { shadow.push(pt); } } else if vis > 0.95 && n.dot(sun_dir) > 0.2 { if lit.len() < 2000 { lit.push(pt); } }
    }
    eprintln!("{dead} random atlas points were DEAD (atlas irradiance < 0.01 and visibility < 0.05: hidden/buried surfaces, excluded)");
    eprintln!("{stem}: {} shadowed atlas points (DM visibility < 0.05), {} sunlit controls, probe spacing {:.1} wu, sun dir ({:.2},{:.2},{:.2})", shadow.len(), lit.len(), g.spacing, sun_dir.x, sun_dir.y, sun_dir.z);
    assert!(shadow.len() >= 500, "liveness: only {} shadowed points", shadow.len());
    let lum3 = |c: [f32; 3]| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64;
    let pct = |v: &mut Vec<f64>, f: f64| -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * f) as usize] };
    let eval = |pts: &[Pt], slot: u32, edits: &[(&str, &str)]| -> Vec<f64> {
        let q: Vec<([f32; 3], [f32; 3])> = pts.iter().map(|p| (p.p, p.n)).collect();
        g.sample_irradiance_variant(&device, &queue, &q, slot, edits).iter().map(|c| lum3(*c)).collect()
    };
    // exposure classes from the RENDER triangles (what the player sees): buried = a 3 wu ray along the normal hits geometry;
    // sky-clear = the straight-up ray escapes. Brute force over the atlas-lit triangles, in parallel.
    let (class, nbur): (Vec<u8>, usize) = {
        use rayon::prelude::*;
        let hit = |o: Vec3, d: Vec3, tmax: f32| -> bool {
            render_tris.iter().any(|t| {
                let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
                let pv = d.cross(e2);
                let det = e1.dot(pv);
                if det.abs() < 1e-9 { return false; }
                let inv = 1.0 / det;
                let tv = o - t[0];
                let u = tv.dot(pv) * inv;
                if !(0.0..=1.0).contains(&u) { return false; }
                let qv = tv.cross(e1);
                let v = d.dot(qv) * inv;
                if v < 0.0 || u + v > 1.0 { return false; }
                let tt = e2.dot(qv) * inv;
                tt > 0.02 && tt < tmax
            })
        };
        let cl: Vec<u8> = shadow.par_iter().map(|p| {
            let (o, n) = (Vec3::from(p.p), Vec3::from(p.n));
            if hit(o, n, 3.0) { 0 } else if hit(o, Vec3::Z, 1000.0) { 1 } else { 2 }
        }).collect();
        let nb = cl.iter().filter(|&&c| c == 0).count();
        (cl, nb)
    };
    eprintln!("exposure of the {} shadowed points by RENDER geometry: buried (3 wu normal ray hits) {nbur}; exposed but roofed {}; exposed and sky-clear {}", shadow.len(),
        class.iter().filter(|&&c| c == 1).count(), class.iter().filter(|&&c| c == 2).count());
    let sky_all = eval(&shadow, 19, &[]);
    let sky_open = { let mut v = sky_all.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() * 9 / 10] };
    let covered: Vec<bool> = sky_all.iter().map(|s| *s < 0.5 * sky_open).collect();
    let e_lm: Vec<f64> = shadow.iter().map(|p| lum3(p.e_lm)).collect();
    let report = |label: &str, hms: &Vec<f64>, mask: &dyn Fn(usize) -> bool| {
        let mut r: Vec<f64> = (0..hms.len()).filter(|&i| mask(i)).map(|i| hms[i] / e_lm[i]).collect();
        if r.len() < 10 { eprintln!("  {label:<46} ({} points)", r.len()); return; }
        let n = r.len();
        eprintln!("  {label:<46} n {n:5} | HMS/atlas ratio p10 {:.2}  median {:.2}  p90 {:.2}  | > 1.5x: {:.1}%  < 0.67x: {:.1}%", pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9),
            100.0 * r.iter().filter(|&&v| v > 1.5).count() as f64 / n as f64, 100.0 * r.iter().filter(|&&v| v < 0.67).count() as f64 / n as f64);
    };
    let all = |_: usize| true;
    let total = eval(&shadow, 0, &[]);
    eprintln!("== (1) shadowed points: real-time ambient (gi_irradiance, slot 0) over the atlas lightmap irradiance E_lm(n) ==");
    report("all shadowed points", &total, &all);
    report("sky-open (probe sky-direct >= half of p90)", &total, &|i| !covered[i]);
    report("covered/enclosed (sky-direct < half of p90)", &total, &|i| covered[i]);
    report("up-facing (n.z > 0.5)", &total, &|i| shadow[i].n[2] > 0.5);
    report("walls (|n.z| <= 0.5)", &total, &|i| shadow[i].n[2].abs() <= 0.5);
    report("down-facing (n.z < -0.5)", &total, &|i| shadow[i].n[2] < -0.5);
    let zs: Vec<f32> = { let mut z: Vec<f32> = shadow.iter().map(|p| p.p[2]).collect(); z.sort_by(|a, b| a.partial_cmp(b).unwrap()); z };
    let (z1, z2) = (zs[zs.len() / 3], zs[2 * zs.len() / 3]);
    report(&format!("low  (z < {z1:.0})"), &total, &|i| shadow[i].p[2] < z1);
    report(&format!("mid  ({z1:.0} <= z < {z2:.0})"), &total, &|i| shadow[i].p[2] >= z1 && shadow[i].p[2] < z2);
    report(&format!("high (z >= {z2:.0})"), &total, &|i| shadow[i].p[2] >= z2);
    report("EXPOSED + ROOFED (interior / under overhang)", &total, &|i| class[i] == 1);
    report("EXPOSED + SKY-CLEAR (shadowed only from the sun)", &total, &|i| class[i] == 2);
    report("BURIED (inside render geometry)", &total, &|i| class[i] == 0);
    // the sun that the cascades let into the cache-shadowed points, and the final (ambient + sun) ratio
    let q: Vec<([f32; 3], [f32; 3])> = shadow.iter().map(|p| (p.p, p.n)).collect();
    let svis = g.sample_sun_vis(&device, &queue, &q, false);
    let sun_e: Vec<f64> = shadow.iter().zip(&svis).map(|(p, v)| lum3([sun_col[0] , sun_col[1], sun_col[2]]) * Vec3::from(p.n).dot(sun_dir).max(0.0) as f64 * (*v as f64).powi(2) / std::f64::consts::PI).collect();
    let fin: Vec<f64> = total.iter().zip(&sun_e).map(|(a, b)| a + b).collect();
    eprintln!("== (2) FINAL diffuse in cache-shadowed points: ambient + the sun the traced cascades let through (sun x N.L x vis^2 / pi) ==");
    report("ambient + sun", &fin, &all);
    {
        // is the sun that gets into cache-shadowed points the shadow MAP's doing, or do real rays through the same geometry reach it too?
        let truth = g.trace_sun_fraction(&device, &queue, &q);
        let ok: Vec<usize> = (0..shadow.len()).filter(|&i| Vec3::from(shadow[i].n).dot(sun_dir) > 0.1).collect();
        let (mut map_leak, mut ray_leak, mut both) = (0usize, 0usize, 0usize);
        let mut dsum = 0.0f64;
        for &i in &ok {
            let m = (svis[i] as f64).powi(2);
            let t = truth[i] as f64;
            if m > 0.25 { map_leak += 1; }
            if t > 0.25 { ray_leak += 1; }
            if m > 0.25 && t > 0.25 { both += 1; }
            dsum += m - t;
        }
        eprintln!("  sun-facing cache-shadowed points: {} | shadow map says lit (>25%): {:.1}% | 32 rays through the SAME geometry say lit: {:.1}% | both: {:.1}% | mean (map - rays) {:+.3}",
            ok.len(), 100.0 * map_leak as f64 / ok.len() as f64, 100.0 * ray_leak as f64 / ok.len() as f64, 100.0 * both as f64 / ok.len() as f64, dsum / ok.len() as f64);
    }
    let leaked = sun_e.iter().zip(&e_lm).filter(|(s, e)| **s > 0.25 * **e).count();
    eprintln!("  points where the traced sun alone adds > 25% of the atlas irradiance: {:.1}%", 100.0 * leaked as f64 / shadow.len() as f64);
    let mut lit_stats: Option<(f64, f64)> = None;
    // control: the sunlit points, ambient+sun vs atlas (the 'every sunlit area' reference)
    if lit.len() > 50 {
        let a = eval(&lit, 0, &[]);
        let q2: Vec<([f32; 3], [f32; 3])> = lit.iter().map(|p| (p.p, p.n)).collect();
        let sv = g.sample_sun_vis(&device, &queue, &q2, false);
        let mut r: Vec<f64> = lit.iter().enumerate().map(|(i, p)| {
            let ndl = Vec3::from(p.n).dot(sun_dir).max(0.0) as f64;
            let hms = a[i] + lum3(sun_col) * ndl * (sv[i] as f64).powi(2) / std::f64::consts::PI;
            let ship = lum3(p.e_lm) + lum3(sun_col) * ndl * (p.vis as f64).powi(2) / std::f64::consts::PI;
            hms / ship }).collect();
        {
            let elit: Vec<f64> = lit.iter().map(|p| lum3(p.e_lm)).collect();
            for (slot, name) in [(0u32, "total "), (19, "sky   "), (10, "gather")] {
                let v = eval(&lit, slot, &[]);
                let mut r: Vec<f64> = v.iter().zip(&elit).map(|(a, b)| a / b).collect();
                eprintln!("  sunlit control, ambient only, {name}: HMS/atlas p10 {:.2} median {:.2} p90 {:.2}", pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9));
            }
        }
        lit_stats = Some((pct(&mut r.clone(), 0.1), pct(&mut r.clone(), 0.5)));
        eprintln!("== control: sunlit atlas points (vis > 0.95, N.L > 0.2): HMS/atlas final ratio p10 {:.2} median {:.2} p90 {:.2} ({} points)", pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9), r.len());
    }
    // (3) the terms
    eprintln!("== (3) decomposition of the shadow ambient (ratio to E_lm; median over all shadowed points) ==");
    let sky = sky_all.clone();
    let gather = eval(&shadow, 10, &[]);
    report("sky-direct SH only (slot 19)", &sky, &all);
    report("bounce/emitter gather SH only (slot 10)", &gather, &all);
    let nofb = eval(&shadow, 0, &[("for (var k = 0u; k < 64u; k = k + 1u) {", "for (var k = 0u; k < 0u; k = k + 1u) {")]);
    report("4x4x4 hole fallback OFF (black where it applied)", &nofb, &all);
    let nl2 = eval(&shadow, 0, &[("e = e + a2 * (", "e = e + 0.0 * a2 * (")]);
    report("SH L2 band removed (L0 + L1 only)", &nl2, &all);
    let novis = eval(&shadow, 0, &[("w = w * max(cheb, 0.02);", "w = w * 1.0;"), ("w = w * (wb * wb + 0.2);", "w = w * 1.0;")]);
    report("no Chebyshev visibility / backface weights", &novis, &all);
    // (3c) interpolation variants: which probe-weighting choices move the shadow tail (non-buried points only)
    if std::env::var("HMS_GATES_VARIANTS").is_ok() {
        eprintln!("== (3c) interpolation variants (non-buried points; ratio to the atlas) ==");
        let variants: [(&str, Vec<(&str, &str)>); 6] = [
            ("current", vec![]),
            ("Chebyshev floor 0.02 -> 0", vec![("w = w * max(cheb, 0.02);", "w = w * cheb;")]),
            ("Chebyshev power 3 -> 6", vec![("cheb = max(cheb * cheb * cheb, 0.0);", "cheb = max(cheb * cheb * cheb * cheb * cheb * cheb, 0.0);")]),
            ("backface weight wb^2+0.2 -> wb^4+0.05", vec![("w = w * (wb * wb + 0.2);", "w = w * (wb * wb * wb * wb + 0.05);")]),
            ("normal bias 0.2 -> 0.5 cell", vec![("let bp = p + (n * gi.bias.x + v * gi.bias.y) * sp;", "let bp = p + (n * 0.5 + v * gi.bias.y) * sp;")]),
            ("all four", vec![("w = w * max(cheb, 0.02);", "w = w * cheb;"), ("cheb = max(cheb * cheb * cheb, 0.0);", "cheb = max(cheb * cheb * cheb * cheb * cheb * cheb, 0.0);"), ("w = w * (wb * wb + 0.2);", "w = w * (wb * wb * wb * wb + 0.05);"), ("let bp = p + (n * gi.bias.x + v * gi.bias.y) * sp;", "let bp = p + (n * 0.5 + v * gi.bias.y) * sp;")]),
        ];
        for (name, ed) in &variants {
            let v = eval(&shadow, 0, ed);
            for (cname, cls) in [("sky-clear", 2u8), ("roofed", 1u8)] {
                let mut r: Vec<f64> = (0..shadow.len()).filter(|&i| class[i] == cls).map(|i| v[i] / e_lm[i]).collect();
                let n = r.len() as f64;
                eprintln!("  {name:<40} {cname:<9}: p10 {:.2} median {:.2} p90 {:.2} | > 1.5x {:.1}% < 0.67x {:.1}%", pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9), 100.0 * r.iter().filter(|&&x| x > 1.5).count() as f64 / n, 100.0 * r.iter().filter(|&&x| x < 0.67).count() as f64 / n);
            }
        }
    }
    // fallback usage
    let fb_used = (0..shadow.len()).filter(|&i| nofb[i] < 0.5 * total[i]).count();
    eprintln!("  points that rely on the hole fallback (result < half without it): {:.1}%", 100.0 * fb_used as f64 / shadow.len() as f64);
    // is the shortfall where the shipped bake's own (direct) lights are? Those 41-ish stli lights light the atlas but reach the probes only as
    // bounce sources, and the receivers get no direct term from them in this mode.
    {
        let lights = sc.stli_baked_positions();
        if !lights.is_empty() {
            let dist: Vec<f64> = shadow.iter().map(|p| lights.iter().map(|l| ((0..3).map(|a| (l.0[a] - p.p[a]).powi(2)).sum::<f32>()).sqrt() as f64).fold(f64::MAX, f64::min)).collect();
            eprintln!("== (3a) HMS/atlas ratio by distance to the nearest baked stli light ({} lights); non-buried points ==", lights.len());
            for (lo, hi) in [(0.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, 1.0e9)] {
                let mut r: Vec<f64> = (0..shadow.len()).filter(|&i| class[i] != 0 && dist[i] >= lo && dist[i] < hi).map(|i| total[i] / e_lm[i]).collect();
                if r.len() >= 10 { eprintln!("  {:5.0}..{:<6.0} wu: n {:5} | median {:.2} p10 {:.2} p90 {:.2}", lo, hi.min(9999.0), r.len(), pct(&mut r, 0.5), pct(&mut r, 0.1), pct(&mut r, 0.9)); }
            }
        }
    }
    // fallback-reliant points: what the fallback gives there, against the atlas
    {
        let mut r: Vec<f64> = (0..shadow.len()).filter(|&i| class[i] != 0 && nofb[i] < 0.5 * total[i]).map(|i| total[i] / e_lm[i]).collect();
        if r.len() >= 10 { eprintln!("  the hole fallback's own points (not buried): n {} | HMS/atlas p10 {:.2} median {:.2} p90 {:.2}", r.len(), pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9)); }
    }
    // the fallback's normalisation / visibility floor, on the points that use it (cross-map evidence for the chosen form)
    {
        let fbp: Vec<usize> = (0..shadow.len()).filter(|&i| class[i] != 0 && nofb[i] < 0.5 * total[i]).collect();
        if fbp.len() >= 10 {
            let fv: [(&str, Vec<(&str, &str)>); 4] = [
                ("energy-conserving, vis floor 0.25 (current)", vec![]),
                ("renormalised by visible weight (old)", vec![("return (fsum / fden) * (gi.bias.w / GI_PI);", "return (fsum / max(fw, 1e-6)) * (gi.bias.w / GI_PI);")]),
                ("energy-conserving, vis floor 0.5", vec![("w = w * max(cheb, 0.25);", "w = w * max(cheb, 0.5);")]),
                ("energy-conserving, vis floor 0.75", vec![("w = w * max(cheb, 0.25);", "w = w * max(cheb, 0.75);")]),
            ];
            for (name, ed) in &fv {
                let v = eval(&shadow, 0, ed);
                let mut r: Vec<f64> = fbp.iter().map(|&i| v[i] / e_lm[i]).collect();
                eprintln!("  fallback form {name:<46} n {:3}: HMS/atlas p10 {:.2} median {:.2} p90 {:.2}", r.len(), pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9));
            }
        }
    }
    // (3b) INNER PROBE GRID: the same points through the shipped camera-centred grid, against the single grid
    let mut inner_gate: Option<[f64; 8]> = None;
    if !inner_scenes.is_empty() {
        let (hx, hz) = (18.0f32, 5.0f32);
        let ok: Vec<usize> = (0..shadow.len()).filter(|&i| class[i] != 0).collect();
        let near = |i: usize, j: usize| (shadow[i].p[0] - shadow[j].p[0]).abs() < hx && (shadow[i].p[1] - shadow[j].p[1]).abs() < hx && (shadow[i].p[2] - shadow[j].p[2]).abs() < hz;
        let best = ok.iter().copied().max_by_key(|&i| ok.iter().filter(|&&j| near(i, j)).count()).unwrap();
        let patch: Vec<usize> = ok.iter().copied().filter(|&j| near(best, j)).collect();
        let c0 = shadow[best].p;
        eprintln!("== (3b) inner probe grid: {} non-buried shadowed points within the window core around ({:.0},{:.0},{:.0}); single grid {:.1} wu ==", patch.len(), c0[0], c0[1], c0[2], g.spacing);
        let el: Vec<f64> = patch.iter().map(|&i| e_lm[i]).collect();
        let q: Vec<([f32; 3], [f32; 3])> = patch.iter().map(|&i| (shadow[i].p, shadow[i].n)).collect();
        let stats = |gg: &RtGi, slot: u32| -> (f64, f64, f64, f64, f64, f64, f64) {
            let v: Vec<f64> = gg.sample_irradiance_variant(&device, &queue, &q, slot, &[]).iter().map(|c| lum3(*c)).collect();
            let mut r: Vec<f64> = v.iter().zip(&el).map(|(a, b)| a / b).collect();
            (pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9), r.iter().filter(|&&x| x > 1.5).count() as f64 / r.len() as f64, r.iter().filter(|&&x| x < 0.67).count() as f64 / r.len() as f64,
             { let mut l: Vec<f64> = r.iter().map(|x| x.max(1e-3).ln().abs()).collect(); l.sort_by(|a, b| a.partial_cmp(b).unwrap()); l.iter().sum::<f64>() / l.len() as f64 },
             { let mut l: Vec<f64> = r.iter().map(|x| x.max(1e-3).ln().abs()).collect(); l.sort_by(|a, b| a.partial_cmp(b).unwrap()); l[(l.len() - 1) * 9 / 10] })
        };
        let pack = sc.stli_direct_pack();
        let lights_on = |gg: &RtGi| -> (f64, f64, f64, f64, f64) {
            let amb: Vec<f64> = gg.sample_irradiance_variant(&device, &queue, &q, 0, &[]).iter().map(|c| lum3(*c)).collect();
            let dl = stli_direct_lum(&pack, &q, Some((gg, &device, &queue)));
            let mut r: Vec<f64> = amb.iter().zip(&dl).zip(&el).map(|((a, d), e)| (a + d) / e).collect();
            let ml = r.iter().map(|x| x.max(1e-3).ln().abs()).sum::<f64>() / r.len() as f64;
            (pct(&mut r, 0.5), pct(&mut r, 0.9), r.iter().filter(|&&x| x > 1.5).count() as f64 / r.len() as f64, r.iter().filter(|&&x| x < 0.67).count() as f64 / r.len() as f64, ml)
        };
        let base = stats(&g, 0);
        eprintln!("  single grid           ({:5.1} wu): total HMS/atlas p10 {:.2} median {:.2} p90 {:.2} | > 1.5x {:.1}% | < 0.67x {:.1}% | mean|ln r| {:.3} p90|ln r| {:.3}", g.spacing, base.0, base.1, base.2, 100.0 * base.3, 100.0 * base.4, base.5, base.6);
        { let w = lights_on(&g); eprintln!("     + baked stli lights (probe-visibility): median {:.2} p90 {:.2} | > 1.5x {:.1}% | < 0.67x {:.1}% | mean|ln r| {:.3}", w.0, w.1, 100.0 * w.2, 100.0 * w.3, w.4); }
        for ((label, pol), sc2) in inner_cfgs.iter().zip(inner_scenes.iter()) {
            let mut g2 = RtGi::new(&device);
            g2.inner_policy = *pol;
            g2.set_scene(&device, &queue, sc2, None, None);
            g2.set_enabled(&queue, true);
            let moved = g2.focus(c0);
            g2.warm_up(&device, &queue, 14);
            let inn = g2.inner;
            let st = stats(&g2, 0);
            eprintln!("  inner {:<22} (outer {:4.1} wu, {} inner probes{}): total p10 {:.2} median {:.2} p90 {:.2} | > 1.5x {:.1}% | < 0.67x {:.1}% | mean|ln r| {:.3} p90|ln r| {:.3}   [{} cells queued]", label, g2.spacing, inn.map_or(0, |i| i.count()), inn.map_or(String::new(), |i| format!(" at {:.2} wu", i.spacing)), st.0, st.1, st.2, 100.0 * st.3, 100.0 * st.4, st.5, st.6, moved);
            { let w = lights_on(&g2); eprintln!("     + baked stli lights (probe-visibility): median {:.2} p90 {:.2} | > 1.5x {:.1}% | < 0.67x {:.1}% | mean|ln r| {:.3}", w.0, w.1, 100.0 * w.2, 100.0 * w.3, w.4); }
            if std::env::var("HMS_GATES_DARKLIST").is_ok() && !matches!(pol, hms_render::rtgi::InnerPolicy::Auto) {
                let vi = g2.sample_irradiance_variant(&device, &queue, &q, 0, &[]);
                let vs = g.sample_irradiance_variant(&device, &queue, &q, 0, &[]);
                let si = g2.sample_irradiance_variant(&device, &queue, &q, 19, &[]);
                let ss = g.sample_irradiance_variant(&device, &queue, &q, 19, &[]);
                let mut rows: Vec<usize> = (0..patch.len()).filter(|&k| lum3(vi[k]) < 0.5 * lum3(vs[k])).collect();
                rows.sort_by(|&a, &b| (lum3(vi[a]) / el[a]).partial_cmp(&(lum3(vi[b]) / el[b])).unwrap());
                eprintln!("  darker with the inner grid than the single grid by > 2x: {} of {} points", rows.len(), patch.len());
                for &k in rows.iter().take(10) {
                    let i = patch[k];
                    eprintln!("    ({:6.1},{:6.1},{:5.1}) n({:5.2},{:5.2},{:5.2}) {} atlas {:.4} | single {:.4} (sky {:.4}) inner {:.4} (sky {:.4}) | window cell dist: x {:.1} y {:.1} z {:.1}", shadow[i].p[0], shadow[i].p[1], shadow[i].p[2], shadow[i].n[0], shadow[i].n[1], shadow[i].n[2], ["buried", "roofed", "sky-clear"][class[i] as usize], el[k], lum3(vs[k]), lum3(ss[k]), lum3(vi[k]), lum3(si[k]), shadow[i].p[0] - c0[0], shadow[i].p[1] - c0[1], shadow[i].p[2] - c0[2]);
                }
            }
            if matches!(pol, hms_render::rtgi::InnerPolicy::Auto) { assert!(inn.is_some(), "the default inner grid did not activate (outer spacing {:.1})", g2.spacing); let w = lights_on(&g2); inner_gate = Some([base.5, base.2, base.3, base.4, w.4, w.1, w.2, w.3]); }
        }
    }
    // worst
    let mut idx: Vec<usize> = (0..shadow.len()).collect();
    idx.sort_by(|&a, &b| (total[b] / e_lm[b]).partial_cmp(&(total[a] / e_lm[a])).unwrap());
    eprintln!("== (4) the 12 worst (brightest vs atlas) shadowed points ==");
    for &i in idx.iter().take(12) {
        let p = &shadow[i];
        eprintln!("      mesh {} '{}' hdr {:.3} k {:.3} raw dom {:.4} fill {:.4} vis {:.3}", p.mi, sc.tag_name_of(geom[p.mi].diffuse), p.hdr, p.k, p.dom, p.fill, p.vis);
        eprintln!("  ({:7.1},{:7.1},{:6.1}) n({:5.2},{:5.2},{:5.2}) {} | atlas {:.4} HMS {:.4} ratio {:.1} | sky {:.4} gather {:.4} nofallback {:.4} noL2 {:.4}", p.p[0], p.p[1], p.p[2], p.n[0], p.n[1], p.n[2], if covered[i] { "covered" } else { "open   " }, e_lm[i], total[i], total[i] / e_lm[i], sky[i], gather[i], nofb[i], nl2[i]);
    }
    eprintln!("== (5) the 8 darkest vs atlas ==");
    for &i in idx.iter().rev().take(8) {
        let p = &shadow[i];
        eprintln!("  ({:7.1},{:7.1},{:6.1}) n({:5.2},{:5.2},{:5.2}) {} | atlas {:.4} HMS {:.4} ratio {:.2}", p.p[0], p.p[1], p.p[2], p.n[0], p.n[1], p.n[2], if covered[i] { "covered" } else { "open   " }, e_lm[i], total[i], total[i] / e_lm[i]);
    }
    // ---- the gate (Forge World; other maps print only: their render geometry stand-in is poorer) ----
    let black = (0..shadow.len()).filter(|&i| class[i] != 0 && e_lm[i] > 0.01 && total[i] < 1e-3 * e_lm[i]).count();
    let nonburied = (0..shadow.len()).filter(|&i| class[i] != 0).count();
    eprintln!("non-buried shadowed points shaded (near) BLACK where the atlas has light: {black} of {nonburied}");
    if let Some([bml, bp90, bshare, bdark, cml, cp90, cshare, cdark]) = inner_gate {
        // the shipped combination (fine camera-centred grid + baked lights drawn per pixel) against the single grid with no
        // lights, on the same points: measured 0.390 -> 0.320 (mean |ln ratio|), p90 2.33 -> 2.12, over 1.5x 20.3 % -> 18.9 %, under 0.67x 5.0 % -> 2.4 %
        assert!(cml < 0.9 * bml, "the inner grid + baked lights do not beat the single grid: mean |ln r| {cml:.3} vs {bml:.3}");
        assert!(cp90 <= bp90 && cshare <= bshare + 0.005 && cdark <= bdark, "a tail got worse: p90 {cp90:.2} ({bp90:.2}), over 1.5x {:.1}% ({:.1}%), under 0.67x {:.1}% ({:.1}%)", 100.0 * cshare, 100.0 * bshare, 100.0 * cdark, 100.0 * bdark);
    }
    if stem == "forge_halo" {
        let sky_clear: Vec<usize> = (0..shadow.len()).filter(|&i| class[i] == 2).collect();
        let mut r: Vec<f64> = sky_clear.iter().map(|&i| total[i] / e_lm[i]).collect();
        let (p10, med) = (pct(&mut r, 0.1), pct(&mut r, 0.5));
        let (lp10, lmed) = lit_stats.expect("sunlit control");
        // measured on Forge World with the loader's render geometry + albedo: shadowed sky-clear median 1.19 (p10 1.02),
        // sunlit final median 1.01 (p10 0.95), hole-fallback points median 1.01, 36 of 6847 (0.5%) black points (deep roofed pockets), gate <= 1%
        assert!(med > 0.85 && med < 1.45 && p10 > 0.8, "shadowed (sky-clear) ambient vs the atlas: median {med:.2} p10 {p10:.2}");
        assert!(lmed > 0.9 && lmed < 1.15 && lp10 > 0.85, "sunlit final vs the atlas: median {lmed:.2} p10 {lp10:.2}");
        assert!(black * 100 <= nonburied, "{black} of {nonburied} non-buried shadowed points are black where the atlas has light");
        let fbp: Vec<usize> = (0..shadow.len()).filter(|&i| class[i] != 0 && nofb[i] < 0.5 * total[i]).collect();
        assert!(fbp.len() >= 50, "liveness: only {} hole-fallback points", fbp.len());
        let mut fr: Vec<f64> = fbp.iter().map(|&i| total[i] / e_lm[i]).collect();
        let fmed = pct(&mut fr, 0.5);
        assert!(fmed > 0.5 && fmed < 2.0, "the hole fallback's light vs the atlas: median {fmed:.2} (renormalising by the visible weight gives 3.9, a pocket left unlit 0.1)");
    }
}

/// THE BAKED STLI LIGHTS AGAINST THE ATLAS. The shipped lightmap holds the direct term of the map's bake-direct lights (omni /
/// spot with bungie type 0/1); the real-time mode's probes carry only their bounce. Points on atlas-lit surfaces near such a
/// light, with a clear line to it and no other light close by, give the light's own contribution as
/// (atlas irradiance - the field's ambient there); that is compared with `stli_direct_pack` evaluated the way the pixel shader
/// does (`calc_simple_lights`: colour x (N.L + 0.06) x cone x (1 - d/range)^2 / pi). The same pack, unshadowed, over EVERY point
/// of the rings, is what the app would draw: the second table is the leak through walls.
#[test]
#[ignore = "software GL + a Reach map (minutes)"]
fn rtgi_stli_direct_light_vs_the_shipped_atlas() {
    let stem = std::env::var("HMS_GATES_MAP").unwrap_or_else(|_| "70_boneyard".into());
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(&stem) else {
        eprintln!("rtgi gates SKIPPED: no {stem} cache");
        return;
    };
    let (device, queue) = test_device();
    std::env::set_var("HMS_RTGI_MAX_PROBES", "32768");
    std::env::set_var("HMS_RTGI_MIN_SPACING", "1.5");
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let mesh_albedo = sc.load_lm_geom_for_test(&device, &queue);
    let mut tri_mesh: Vec<u32> = Vec::new();
    let render_tris: Vec<Tri> = sc.lm_geom().iter().enumerate().flat_map(|(mi, m)| m.indices.chunks_exact(3).filter_map(|t| {
        let (a, b, c) = (m.verts.get(t[0] as usize)?, m.verts.get(t[1] as usize)?, m.verts.get(t[2] as usize)?);
        Some(([Vec3::from(a.0), Vec3::from(b.0), Vec3::from(c.0)], mi as u32))
    }).collect::<Vec<_>>()).map(|(t, mi)| { tri_mesh.push(mi); t }).collect();
    let (scene, _ship, _) = forge_world_scene_geom(&mut sc, &native, cache, 0.4, true, Some((&render_tris, false, &tri_mesh, &mesh_albedo)));
    let (bmin, bmax) = (scene.bounds_min, scene.bounds_max);
    let mut g = RtGi::new(&device);
    g.inner_policy = hms_render::rtgi::InnerPolicy::Off;
    g.set_scene(&device, &queue, &scene, None, None);
    g.set_enabled(&queue, true);
    g.warm_up(&device, &queue, 8);
    let pack = sc.stli_direct_pack();
    eprintln!("{stem}: {} baked direct lights (omni/spot, bungie type 0/1); {} atlas-lit meshes; probe spacing {:.1} wu", pack.len(), sc.lm_geom().len(), g.spacing);
    if pack.is_empty() { eprintln!("no baked lights on this map"); return; }
    let geom = sc.lm_geom();
    let near_light = |p: Vec3, r: f32| pack.iter().any(|l| (Vec3::from(l.0) - p).length() < r);
    let mut tri_list: Vec<(u32, u32)> = Vec::new();
    for (mi, m) in geom.iter().enumerate() { for t in 0..(m.indices.len() / 3) {
        let (Some(a), Some(b), Some(c)) = (m.verts.get(m.indices[t * 3] as usize), m.verts.get(m.indices[t * 3 + 1] as usize), m.verts.get(m.indices[t * 3 + 2] as usize)) else { continue };
        let ctr = (Vec3::from(a.0) + Vec3::from(b.0) + Vec3::from(c.0)) / 3.0;
        if near_light(ctr, 18.0) { tri_list.push((mi as u32, t as u32)); }
    } }
    assert!(tri_list.len() > 200, "only {} triangles near lights", tri_list.len());
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng >> 11) as f64 / (1u64 << 53) as f64 };
    let mut imgs: std::collections::HashMap<usize, Option<crate::scene::LmAtlasImages>> = Default::default();
    struct P { p: [f32; 3], n: [f32; 3], e_lm: [f32; 3] }
    let mut pts: Vec<P> = Vec::new();
    for _ in 0..400_000 {
        if pts.len() >= 2500 { break; }
        let (mi, ti) = tri_list[(next() * tri_list.len() as f64) as usize % tri_list.len()];
        let m = &geom[mi as usize];
        let t = &m.indices[ti as usize * 3..ti as usize * 3 + 3];
        let (v0, v1, v2) = (&m.verts[t[0] as usize], &m.verts[t[1] as usize], &m.verts[t[2] as usize]);
        let (mut a, mut b) = (next() as f32, next() as f32);
        if a + b > 1.0 { a = 1.0 - a; b = 1.0 - b; }
        let w = [1.0 - a - b, a, b];
        let pos = Vec3::from(v0.0) * w[0] + Vec3::from(v1.0) * w[1] + Vec3::from(v2.0) * w[2];
        let n = (Vec3::from(v0.1) * w[0] + Vec3::from(v1.1) * w[1] + Vec3::from(v2.1) * w[2]).normalize_or_zero();
        if n.length_squared() < 0.5 || (0..3).any(|k| pos[k] < bmin[k] || pos[k] > bmax[k]) || !near_light(pos, 16.0) { continue; }
        let uv = [v0.2[0] * w[0] + v1.2[0] * w[1] + v2.2[0] * w[2], v0.2[1] * w[0] + v1.2[1] * w[1] + v2.2[1] * w[2]];
        let Some(im) = imgs.entry(mi as usize).or_insert_with(|| sc.shipped_lm_images(m.dm, m.sdm)) else { continue };
        let st = crate::scene::shipped_lm_texel(im, m.hdr, m.k, uv);
        let e = crate::lightprobe::dual_vmf_diffuse(n, Vec3::from(st.dom_dir), st.dom_rgb, st.fill_rgb, st.bandwidth);
        pts.push(P { p: (pos + n * 0.05).into(), n: n.into(), e_lm: e });
    }
    assert!(pts.len() >= 500, "only {} points near lights", pts.len());
    let lum3 = |c: [f32; 3]| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64;
    let q: Vec<([f32; 3], [f32; 3])> = pts.iter().map(|p| (p.p, p.n)).collect();
    let amb: Vec<f64> = g.sample_irradiance(&device, &queue, &q).iter().map(|c| lum3(*c)).collect();
    // the shader's per-light term (mesh.rs calc_simple_lights), luminance
    let term = |l: &[f32; 20], p: Vec3, n: Vec3| -> f64 {
        let to = Vec3::new(l[0], l[1], l[2]) - p;
        let d2 = to.length_squared();
        if d2 >= l[3] { return 0.0; }
        let d = d2.sqrt().max(1e-4);
        let lh = to / d;
        let att = ((l[17] - d) * l[18]).clamp(0.0, 1.0).powi(2);
        let ca = (-lh).dot(Vec3::new(l[4], l[5], l[6])).clamp(0.0, 1.0);
        let cone = ((ca - l[12]) * l[13].max(1e-4)).clamp(0.0, 1.0).powf(l[14].max(1.0));
        let ang = cone + (1.0 - cone) * l[7];
        let ndl = (n.dot(lh) + 0.06).clamp(0.0, 1.0);
        lum3([l[8] * ndl * ang * att * 0.3183, l[9] * ndl * ang * att * 0.3183, l[10] * ndl * ang * att * 0.3183]) as f64
    };
    use rayon::prelude::*;
    let blocked = |o: Vec3, to: Vec3| -> bool {
        let d = to - o;
        let len = d.length();
        let dir = d / len.max(1e-6);
        render_tris.iter().any(|t| {
            let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
            let pv = dir.cross(e2);
            let det = e1.dot(pv);
            if det.abs() < 1e-9 { return false; }
            let inv = 1.0 / det;
            let tv = o - t[0];
            let u = tv.dot(pv) * inv;
            if !(0.0..=1.0).contains(&u) { return false; }
            let qv = tv.cross(e1);
            let v = dir.dot(qv) * inv;
            if v < 0.0 || u + v > 1.0 { return false; }
            let tt = e2.dot(qv) * inv;
            tt > 0.05 && tt < len - 0.1
        })
    };
    // per point: the nearest light, its term, the others' terms, and whether the nearest is in line of sight
    struct R { d: f64, ndl: f64, own: f64, others: f64, los: bool, all_unshadowed: f64, all_visible: f64, near8: Vec<(usize, f64)>, los4: Vec<(usize, bool)> }
    let rows: Vec<R> = pts.par_iter().map(|pt| {
        let (p, n) = (Vec3::from(pt.p), Vec3::from(pt.n));
        let mut terms: Vec<(f32, f64, usize)> = pack.iter().enumerate().map(|(i, l)| ((Vec3::from(l.0) - p).length(), term(&l.1, p, n), i)).collect();
        terms.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (d0, own, i0) = terms[0];
        let others: f64 = terms.iter().skip(1).map(|t| t.1).sum();
        let all: f64 = terms.iter().take(8).map(|t| t.1).sum();
        let los = !blocked(p, Vec3::from(pack[i0].0));
        // the same sum with each light's line of sight checked (the 4 nearest that actually contribute): the price of doing it right
        let mut vis = 0.0;
        let mut checked = 0;
        for t in terms.iter().take(8) {
            if t.1 <= 0.0 { continue; }
            if checked >= 4 { break; }
            checked += 1;
            if !blocked(p, Vec3::from(pack[t.2].0)) { vis += t.1; }
        }
        let near8: Vec<(usize, f64)> = terms.iter().take(8).map(|t| (t.2, t.1)).collect();
        let los4: Vec<(usize, bool)> = terms.iter().take(8).filter(|t| t.1 > 0.0).take(4).map(|t| (t.2, !blocked(p, Vec3::from(pack[t.2].0)))).collect();
        R { d: d0 as f64, ndl: n.dot((Vec3::from(pack[i0].0) - p).normalize()) as f64, own, others, los, all_unshadowed: all, all_visible: vis, near8, los4 }
    }).collect();
    // probe-field light visibility (the WGSL the pixel shaders use) for the 8 nearest lights of each point
    let mut trip: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
    let mut idx: Vec<(usize, usize)> = Vec::new();
    for (i, r) in rows.iter().enumerate() { for (li, _) in &r.near8 { trip.push((pts[i].p, pts[i].n, pack[*li].0)); idx.push((i, *li)); } }
    let pv = g.sample_light_vis(&device, &queue, &trip);
    let mut probe_vis_sum = vec![0.0f64; rows.len()];
    let (mut agree, mut tot_c, mut false_vis, mut false_hid) = (0usize, 0usize, 0usize, 0usize);
    {
        let mut k = 0usize;
        for (i, r) in rows.iter().enumerate() {
            for (li, tm) in &r.near8 {
                probe_vis_sum[i] += tm * pv[k] as f64 * std::env::var("HMS_GATES_STLI_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
                if let Some((_, los)) = r.los4.iter().find(|(l, _)| l == li) {
                    tot_c += 1;
                    let vis = pv[k] > 0.5;
                    if vis == *los { agree += 1; } else if vis { false_vis += 1; } else { false_hid += 1; }
                }
                k += 1;
            }
        }
    }
    eprintln!("probe-field light visibility vs ray-traced line of sight: {} (point, light) pairs: agree {:.1}%, probe says visible but a wall is in the way {:.1}%, probe says hidden but it is visible {:.1}%", tot_c, 100.0 * agree as f64 / tot_c.max(1) as f64, 100.0 * false_vis as f64 / tot_c.max(1) as f64, 100.0 * false_hid as f64 / tot_c.max(1) as f64);
    let pct = |v: &mut Vec<f64>, f: f64| -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * f) as usize] };
    eprintln!("== (1) the light's own term: (atlas - ambient) / formula, nearest light in line of sight, other lights < 10% of it, formula > 5% of the ambient ==");
    let mut medians: Vec<(f64, f64, usize)> = Vec::new();
    for (lo, hi) in [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0)] {
        let mut r: Vec<f64> = rows.iter().enumerate().filter(|(i, x)| x.los && x.d >= lo && x.d < hi && x.ndl > 0.2 && x.others < 0.1 * x.own && x.own > 0.05 * amb[*i].max(1e-3)).map(|(i, x)| (lum3(pts[i].e_lm) - amb[i]) / x.own).collect();
        if r.len() < 15 { eprintln!("  {lo:>4.0}..{hi:<4.0} wu: {} points (too few)", r.len()); continue; }
        let (m, p10, p90) = (pct(&mut r, 0.5), pct(&mut r, 0.1), pct(&mut r, 0.9));
        eprintln!("  {lo:>4.0}..{hi:<4.0} wu: n {:4} | measured / formula  median {:.2}  p10 {:.2}  p90 {:.2}", r.len(), m, p10, p90);
        medians.push((lo, m, r.len()));
    }
    eprintln!("== (2) what the app would draw: ambient + the 8 nearest lights UNSHADOWED, over every point of the ring, against the atlas ==");
    let ring_gate = |lo: f64, hi: f64, with: u8| -> Option<(f64, f64, f64, usize, f64, f64, f64)> {
        let mut r: Vec<f64> = rows.iter().enumerate().filter(|(_, x)| x.d >= lo && x.d < hi).map(|(i, x)| (amb[i] + match with { 1 => x.all_unshadowed, 2 => x.all_visible, 3 => probe_vis_sum[i], _ => 0.0 }) / lum3(pts[i].e_lm).max(1e-4)).collect();
        if r.len() < 30 { return None; }
        let n = r.len();
        let mean_ln = r.iter().map(|x| x.max(1e-3).ln().abs()).sum::<f64>() / n as f64;
        let dark = r.iter().filter(|&&x| x < 0.5).count() as f64 / n as f64;
        let bright = r.iter().filter(|&&x| x > 2.0).count() as f64 / n as f64;
        Some((pct(&mut r, 0.1), pct(&mut r, 0.5), pct(&mut r, 0.9), n, mean_ln, dark, bright))
    };
    for (lo, hi) in [(0.0, 8.0), (8.0, 16.0)] {
        for (with, name) in [(0u8, "off          "), (1, "ON unshadowed"), (2, "ON line-of-sight"), (3, "ON probe-visibility")] {
            if let Some((p10, m, p90, n, ml, dk, br)) = ring_gate(lo, hi, with) { eprintln!("  ring {lo:.0}..{hi:.0} wu, lights {name}: n {n:4} | final / atlas p10 {p10:.2} median {m:.2} p90 {p90:.2} | mean|ln r| {ml:.3}, below 0.5x {:.1}%, above 2x {:.1}%", 100.0 * dk, 100.0 * br); }
        }
    }
    let leak = rows.iter().enumerate().filter(|(_, x)| !x.los && x.d < 16.0 && x.own > 0.5 * lum3(pts[0].e_lm).min(1.0)).count();
    let occl = rows.iter().filter(|x| !x.los && x.d < 16.0).count();
    eprintln!("  points whose nearest light is NOT in line of sight: {} of {} ({} of them would still receive a visible term)", occl, rows.len(), leak);
    // the gate: the lights, drawn the way the app draws them (probe-field visibility), bring the 0..8 wu ring to the atlas
    {
        let off = ring_gate(0.0, 8.0, 0).expect("ring points");
        let on = ring_gate(0.0, 8.0, 3).expect("ring points");
        let unshadowed = ring_gate(0.0, 8.0, 1).expect("ring points");
        // measured: Boneyard median 0.62 -> 0.90 (mean |ln r| 2.15 -> 1.71), Settlement 0.84 -> 0.96 (1.34 -> 1.17); unshadowed p90 24.4 / 7.7 vs 12.0 / 6.8
        assert!(on.1 > 0.75 && on.1 <= 1.0, "ring 0..8 wu with the lights: final/atlas median {:.2} (lights off {:.2})", on.1, off.1);
        assert!(on.4 < off.4, "the lights do not bring the ring closer to the atlas: mean |ln r| {:.3} vs {:.3} without", on.4, off.4);
        assert!(on.2 <= unshadowed.2, "hiding the lights behind walls did not cut the bright tail: p90 {:.2} vs {:.2} unshadowed", on.2, unshadowed.2);
    }
}

/// Sum over the 8 nearest baked stli lights of the pixel shader's per-light term (luminance), each multiplied by the probe-field
/// visibility of the light (`gi_light_vis`) read from `g`; the unshadowed sum when `g` is None.
fn stli_direct_lum(pack: &[([f32; 3], [f32; 20])], pts: &[([f32; 3], [f32; 3])], g: Option<(&RtGi, &wgpu::Device, &wgpu::Queue)>) -> Vec<f64> {
    let lum3 = |c: [f32; 3]| 0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64;
    let term = |l: &[f32; 20], p: Vec3, n: Vec3| -> f64 {
        let to = Vec3::new(l[0], l[1], l[2]) - p;
        let d2 = to.length_squared();
        if d2 >= l[3] { return 0.0; }
        let d = d2.sqrt().max(1e-4);
        let lh = to / d;
        let att = ((l[17] - d) * l[18]).clamp(0.0, 1.0).powi(2);
        let ca = (-lh).dot(Vec3::new(l[4], l[5], l[6])).clamp(0.0, 1.0);
        let cone = ((ca - l[12]) * l[13].max(1e-4)).clamp(0.0, 1.0).powf(l[14].max(1.0));
        let ang = cone + (1.0 - cone) * l[7];
        let ndl = (n.dot(lh) + 0.06).clamp(0.0, 1.0);
        lum3([l[8] * ndl * ang * att * 0.3183, l[9] * ndl * ang * att * 0.3183, l[10] * ndl * ang * att * 0.3183]) as f64
    };
    let mut trip: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::new();
    let mut owners: Vec<(usize, f64)> = Vec::new();
    for (i, (p, n)) in pts.iter().enumerate() {
        let (pv, nv) = (Vec3::from(*p), Vec3::from(*n));
        let mut t: Vec<(f32, f64, usize)> = pack.iter().enumerate().map(|(k, l)| ((Vec3::from(l.0) - pv).length(), term(&l.1, pv, nv), k)).collect();
        t.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for x in t.iter().take(8) { if x.1 > 0.0 { trip.push((*p, *n, pack[x.2].0)); owners.push((i, x.1)); } }
    }
    let vis: Vec<f32> = match g { Some((gg, d, q)) if !trip.is_empty() => gg.sample_light_vis(d, q, &trip), _ => vec![1.0; trip.len()] };
    let mut out = vec![0.0f64; pts.len()];
    // test seam for the saboteurs: HMS_GATES_STLI_SCALE multiplies the light term (x0 = lights removed, x2 = doubled)
    let scale: f64 = std::env::var("HMS_GATES_STLI_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    for ((i, t), v) in owners.iter().zip(&vis) { out[*i] += t * *v as f64 * scale; }
    out
}
