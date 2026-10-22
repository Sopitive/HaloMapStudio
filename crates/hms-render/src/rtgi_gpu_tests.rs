//! The real-time probe update run FOR REAL on a SOFTWARE adapter (Mesa llvmpipe through GL): no GPU,
//! no window, no display, nothing of the user's session is touched. The harness refuses any adapter
//! that is not a CPU rasteriser, so these can never run on (or wake) a hardware GPU.
//!
//!   cargo test --release -p hms-render rtgi_gpu -- --ignored --test-threads=1
//!
//! They are `#[ignore]`d only because they need Mesa's llvmpipe and take a few seconds. Each scene is
//! small enough that the expected probe SH can be computed in closed form from the engine's own
//! conventions (`engine_gi`), which makes these the gates for the WGSL, not for a copy of it.

use crate::engine_gi::*;
use crate::rtgi::*;
use eframe::wgpu;

/// A CPU (llvmpipe) device. Env vars are set before the first EGL call so libglvnd picks Mesa, never
/// the NVIDIA EGL vendor; the adapter type is then checked and anything but `Cpu` is refused.
fn soft_device() -> (wgpu::Device, wgpu::Queue) {
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
        .expect("no software (llvmpipe) GL adapter: install Mesa's swrast; these tests refuse hardware GPUs");
    eprintln!("rtgi_gpu: adapter {:?}", adapter.get_info().name);
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("rtgi-soft"), required_features: wgpu::Features::empty(), required_limits: adapter.limits(), memory_hints: wgpu::MemoryHints::MemoryUsage },
        None,
    ))
    .expect("software device")
}

type Tri = [[f32; 3]; 3];

/// Median-split BVH, same layout as `lightbake::GpuBvh` (the app builds the production one):
/// node = aabb_min[3], left_first, aabb_max[3], count (8 words); triangle = 3 x (xyz + pad).
pub(crate) fn build_bvh(tris: &[Tri]) -> (Vec<u8>, Vec<u8>, Vec<u32>) {
    let n = tris.len();
    let cen = |t: &Tri| [(t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0, (t[0][2] + t[1][2] + t[2][2]) / 3.0];
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut nodes: Vec<[u32; 8]> = vec![[0; 8]];
    let mut stack = vec![(0usize, 0usize, n)];
    while let Some((ni, start, end)) = stack.pop() {
        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
        for &oi in &order[start..end] {
            for v in &tris[oi as usize] {
                for a in 0..3 { mn[a] = mn[a].min(v[a]); mx[a] = mx[a].max(v[a]); }
            }
        }
        let cnt = end - start;
        let f = |x: f32| x.to_bits();
        if cnt <= 4 {
            nodes[ni] = [f(mn[0]), f(mn[1]), f(mn[2]), start as u32, f(mx[0]), f(mx[1]), f(mx[2]), cnt as u32];
            continue;
        }
        let ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
        let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 } else if ext[1] >= ext[2] { 1 } else { 2 };
        order[start..end].sort_by(|&a, &b| cen(&tris[a as usize])[axis].partial_cmp(&cen(&tris[b as usize])[axis]).unwrap());
        let mid = start + cnt / 2;
        let l = nodes.len();
        nodes.push([0; 8]);
        nodes.push([0; 8]);
        nodes[ni] = [f(mn[0]), f(mn[1]), f(mn[2]), l as u32, f(mx[0]), f(mx[1]), f(mx[2]), 0];
        stack.push((l, start, mid));
        stack.push((l + 1, mid, end));
    }
    let mut tb: Vec<f32> = Vec::with_capacity(n * 12);
    for &oi in &order {
        for v in &tris[oi as usize] { tb.extend_from_slice(&[v[0], v[1], v[2], 0.0]); }
    }
    (bytemuck::cast_slice(&nodes).to_vec(), bytemuck::cast_slice(&tb).to_vec(), order)
}

fn quad(z: f32, half: f32) -> [Tri; 2] {
    let (a, b, c, d) = ([-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]);
    [[a, b, c], [a, c, d]]
}

fn cuboid(lo: [f32; 3], hi: [f32; 3]) -> Vec<Tri> {
    let p = |x: usize, y: usize, z: usize| [[lo[0], hi[0]][x], [lo[1], hi[1]][y], [lo[2], hi[2]][z]];
    let faces = [
        [p(0, 0, 0), p(1, 0, 0), p(1, 1, 0), p(0, 1, 0)], [p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)],
        [p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1)], [p(0, 1, 0), p(1, 1, 0), p(1, 1, 1), p(0, 1, 1)],
        [p(0, 0, 0), p(0, 1, 0), p(0, 1, 1), p(0, 0, 1)], [p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1)],
    ];
    faces.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect()
}

/// A 96-record cap of sky (z in 0.09..0.95, like Forge World's, radiance bluish) + the sun record.
fn test_sky(sun_1st: f32, sky_1st: f32) -> SkyLights {
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
    SkyLights { sky, sun: Some(sun), sun_1st, sky_1st }
}

/// Floor material 0 (opaque, albedo rho), material 1 = sky shell (never hit; keeps the BVH non-empty).
fn mats(rho: f32) -> Vec<[f32; 4]> {
    vec![[rho, rho, rho, 0.0], [0.0; 4], [0.0; 4], [0.0, 0.0, 0.0, 2.0], [0.0; 4], [0.0; 4]]
}

struct Scene {
    tris: Vec<Tri>,
    tri_mat: Vec<u32>,
    rho: f32,
    sky: Option<SkyLights>,
    bounds: ([f32; 3], [f32; 3]),
}

impl Scene {
    fn build(&self) -> RtGiScene {
        let (nodes, tris, order) = build_bvh(&self.tris);
        let sun = self.sky.as_ref().and_then(|k| k.sun).map_or([0.0, 0.0, 1.0], |s| s.dir);
        let sun_e = self.sky.as_ref().map_or([0.0; 3], |k| k.sun_irradiance());
        RtGiScene {
            nodes, tris, tri_mat: order.iter().map(|&o| self.tri_mat[o as usize]).collect(), mats: mats(self.rho),
            lights: vec![], emitters: vec![], total_emitter_weight: 0.0,
            bounds_min: self.bounds.0, bounds_max: self.bounds.1,
            sun_dir: sun, sun_color: sun_e, sky_color: [0.5, 0.5, 0.5],
            fog_sigma: 0.0, fog_top: 0.0, sky: self.sky.clone(),
        }
    }
}

fn floor_scene(rho: f32, sky: Option<SkyLights>) -> Scene {
    let mut tris: Vec<Tri> = quad(0.0, 4000.0).to_vec();
    tris.push([[0.0, 0.0, -9000.0], [1.0, 0.0, -9000.0], [0.0, 1.0, -9000.0]]); // a sky-flagged speck
    Scene { tri_mat: vec![0, 0, 3], tris, rho, sky, bounds: ([-3.0, -3.0, 1.0], [3.0, 3.0, 5.0]) }
}

fn empty_scene(sky: Option<SkyLights>) -> Scene {
    Scene { tris: vec![[[0.0, 0.0, -9000.0], [1.0, 0.0, -9000.0], [0.0, 1.0, -9000.0]]], tri_mat: vec![3], rho: 0.5, sky, bounds: ([-3.0, -3.0, 1.0], [3.0, 3.0, 5.0]) }
}

/// Build, upload and run `passes` full updates; returns the renderer-side object and the probes.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, passes: u32, gain: Option<f32>) -> (RtGi, Vec<ProbeRecord>) {
    std::env::set_var("HMS_RTGI_MIN_SPACING", "1.0"); // 7 x 7 x 5 probes over the 6 x 6 x 4 test box
    let mut g = RtGi::new(device);
    g.set_scene(device, queue, &scene.build(), None, None);
    g.set_enabled(queue, true);
    if let Some(gn) = gain { g.set_gain(queue, gn); }
    g.warm_up(device, queue, passes);
    let p = g.read_probes(device, queue);
    (g, p)
}

fn tool_sh(p: &ProbeRecord, which: u8) -> Sh3 {
    from_graphics_sh(match which { 0 => &p.total, 1 => &p.gather, _ => &p.sky })
}

/// Lower-hemisphere integrals of the Tool basis (what a uniform floor of radiance L contributes to a probe above it).
fn lower_hemisphere_sh() -> Sh9 {
    let n = 40000;
    let mut out = [0.0; 9];
    for i in 0..n {
        let z = -1.0 + (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let ph = i as f64 * 2.399963229728653;
        let y = basis([r * ph.cos(), r * ph.sin(), z]);
        for k in 0..9 { out[k] += y[k] * 2.0 * std::f64::consts::PI / n as f64; }
    }
    out
}

fn unit_close(got: f64, want: f64, abs: f64, rel: f64) -> bool { (got - want).abs() <= abs + rel * want.abs() }

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_sky_only_probe_equals_the_engine_sky_sh() {
    let (device, queue) = soft_device();
    let sc = empty_scene(Some(test_sky(0.3, 1.0)));
    let (g, probes) = run(&device, &queue, &sc, 3, None);
    assert!(g.engine, "a scene with a sky list must run in engine mode");
    assert_eq!(probes.len(), g.nprobes as usize);
    let want = sky_sh_from_list(&sc.sky.as_ref().unwrap().sky, None, &|_| [1.0; 3]);
    let (mut n, mut worst) = (0usize, 0.0f64);
    for p in &probes {
        assert!(p.valid, "an open-air probe must be valid");
        // an open-air probe is never relocated: slot 9 is the relocation record and must stay exactly zero
        // (an earlier shader let a tenth thread blend SH data into it, drifting probes off their cell)
        assert_eq!(p.offset, [0.0, 0.0, 0.0], "probe relocation record polluted");
        let got = tool_sh(p, 0);
        for c in 0..3 {
            for k in 0..9 {
                let tol = 1e-3 + 2e-3 * want[c][k].abs();
                assert!(unit_close(got[c][k], want[c][k], 1e-3, 2e-3), "probe SH[{c}][{k}] = {} want {} (tol {tol})", got[c][k], want[c][k]);
                worst = worst.max((got[c][k] - want[c][k]).abs());
            }
        }
        n += 1;
    }
    assert!(n >= 100, "only {n} probes compared");
    // the gather part is exactly zero in an empty world, and the sky part IS the total
    for p in &probes { assert!(p.gather.iter().all(|c| c.iter().all(|v| v.abs() < 1e-6)), "gather must be 0 in an empty scene"); }
    eprintln!("sky-only: {n} probes, worst |dSH| {worst:.2e} vs the CPU sky_sh_from_list");
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_the_sun_is_never_in_the_probe_sh() {
    let (device, queue) = soft_device();
    let mut a = empty_scene(Some(test_sky(0.3, 1.0)));
    let (_g, pa) = run(&device, &queue, &a, 3, None);
    // a hundred times the sun, and no sun at all: the probe SH must not move
    let mut sky = a.sky.take().unwrap();
    let mut s = sky.sun.unwrap();
    s.rgb = [s.rgb[0] * 100.0, s.rgb[1] * 100.0, s.rgb[2] * 100.0];
    sky.sun = Some(s);
    a.sky = Some(sky.clone());
    let (_g, pb) = run(&device, &queue, &a, 3, None);
    sky.sun = None;
    a.sky = Some(sky);
    let (_g, pc) = run(&device, &queue, &a, 3, None);
    for ((x, y), z) in pa.iter().zip(&pb).zip(&pc) {
        for k in 0..9 { for c in 0..3 {
            assert_eq!(x.total[k][c], y.total[k][c], "sun x100 changed the probe SH");
            assert_eq!(x.total[k][c], z.total[k][c], "removing the sun changed the probe SH");
        } }
    }
    eprintln!("sun x1 / x100 / absent: {} probes bit-identical", pa.len());
}

/// Probe above an infinite floor: sky visible above, floor (albedo rho) below. The floor's exit radiance
/// is `rho (sun_1st E_sun cos + sky_1st E_sky) / pi` (Tool C7/C11, no higher bounce on a plane), seen over
/// the lower hemisphere, so the gather SH is that radiance times the lower-hemisphere basis integrals.
fn floor_expect(sky: &SkyLights, rho: f32) -> [f64; 3] {
    let s = sky.sun.unwrap();
    let mut l = [0.0f64; 3];
    for c in 0..3 {
        let e_sky: f64 = sky.sky.iter().map(|r| r.rgb[c] as f64 * r.omega as f64 * (r.dir[2] as f64).max(0.0)).sum();
        let e_sun = s.rgb[c] as f64 * s.omega as f64 * s.dir[2] as f64;
        l[c] = rho as f64 * (sky.sun_1st as f64 * e_sun + sky.sky_1st as f64 * e_sky) / std::f64::consts::PI;
    }
    l
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_a_floor_bounces_the_engine_amount() {
    let (device, queue) = soft_device();
    let lo = lower_hemisphere_sh();
    let rho = 0.45;
    let sc = floor_scene(rho, Some(test_sky(0.3, 1.0)));
    let (g, probes) = run(&device, &queue, &sc, 40, None);
    assert!(g.engine);
    let l = floor_expect(sc.sky.as_ref().unwrap(), rho);
    let (mut n, mut worst) = (0, 0.0f64);
    for p in &probes {
        if !p.valid { continue; }
        let gat = tool_sh(p, 1);
        for c in 0..3 {
            for k in 0..9 {
                let want = l[c] * lo[k];
                // Monte-Carlo noise sits on every coefficient at the scale of the DC term, so the l>0 terms
                // (which are ~0 or small for a floor) are held to a fraction of the DC, the DC itself tight
                let dc = (l[c] * lo[0]).abs();
                let tol = if k == 0 { 0.05 * dc } else { 0.04 * dc };
                assert!((gat[c][k] as f64 - want).abs() <= tol, "gather SH[{c}][{k}] = {} want {want} (tol {tol})", gat[c][k]);
                if k == 0 { worst = worst.max((gat[c][k] - want).abs() / want.abs().max(1e-9)); }
            }
        }
        // the sky part is untouched by the floor (every record is above the horizon)
        let want_sky = sky_sh_from_list(&sc.sky.as_ref().unwrap().sky, None, &|_| [1.0; 3]);
        let got_sky = tool_sh(p, 2);
        for c in 0..3 { for k in 0..9 { assert!(unit_close(got_sky[c][k], want_sky[c][k], 2e-3, 2e-3), "sky SH[{c}][{k}]"); } }
        // and the total is their sum
        let tot = tool_sh(p, 0);
        for c in 0..3 { for k in 0..9 { assert!((tot[c][k] - got_sky[c][k] - gat[c][k]).abs() < 1e-4, "total != sky + gather"); } }
        n += 1;
    }
    assert!(n >= 100, "only {n} probes compared");
    eprintln!("floor: {n} probes, worst DC rel err {:.3} (floor radiance L = {:.4?})", worst, l);
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_the_first_bounce_scalers_change_only_the_bounce() {
    let (device, queue) = soft_device();
    let rho = 0.45;
    let gather_dc = |sun_1st: f32, sky_1st: f32| -> ([f32; 3], [f32; 3], usize) {
        let sc = floor_scene(rho, Some(test_sky(sun_1st, sky_1st)));
        let (_g, p) = run(&device, &queue, &sc, 40, None);
        let n = p.iter().filter(|r| r.valid).count();
        let mut gat = [0.0f32; 3];
        let mut sky = [0.0f32; 3];
        for r in p.iter().filter(|r| r.valid) { for c in 0..3 { gat[c] += r.gather[0][c] / n as f32; sky[c] += r.sky[0][c] / n as f32; } }
        (gat, sky, n)
    };
    let (full, sky_full, n) = gather_dc(1.0, 1.0);
    let (sun03, sky_03, _) = gather_dc(0.3, 1.0);
    let (sky0, _, _) = gather_dc(0.0, 1.0);
    let (nosky, _, _) = gather_dc(1.0, 0.0);
    assert!(n >= 100);
    for c in 0..3 {
        // the sky-direct SH does not know either scaler
        assert_eq!(sky_full[c], sky_03[c], "the sun scaler changed the sky-direct SH");
        // gather(sun 0.3) = gather(sun 0) + 0.3 (gather(sun 1) - gather(sun 0))
        let want = sky0[c] + 0.3 * (full[c] - sky0[c]);
        assert!((sun03[c] - want).abs() < 0.03 * full[c], "ch {c}: sun 0.3 gather {} want {want} (sun0 {}, sun1 {})", sun03[c], sky0[c], full[c]);
        // the sky scaler removes exactly the sky's bounce
        assert!(nosky[c] < full[c] && nosky[c] > 0.0, "ch {c}: sky_1st = 0 must leave the sun bounce only");
        // the two parts add up to the whole
        assert!((nosky[c] + sky0[c] - full[c]).abs() < 0.04 * full[c], "ch {c}: sun part {} + sky part {} != {}", nosky[c], sky0[c], full[c]);
    }
    eprintln!("scalers: gather DC full {full:.4?}, sun_1st=0.3 {sun03:.4?}, sun_1st=0 {sky0:.4?}, sky_1st=0 {nosky:.4?} ({n} probes)");
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_moving_a_block_updates_the_nearby_probes_only() {
    let (device, queue) = soft_device();
    std::env::set_var("HMS_RTGI_MIN_SPACING", "3.0");
    let mut sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    sc.bounds = ([-12.0, -12.0, 1.0], [12.0, 12.0, 5.0]);
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &sc.build(), None, None);
    g.set_enabled(&queue, true);
    g.warm_up(&device, &queue, 20);
    let before = g.read_probes(&device, &queue);
    let np = before.len();
    assert!(np >= 100, "{np} probes");
    let (o, sp, d) = ([g.params.origin[0], g.params.origin[1], g.params.origin[2]], g.spacing, g.dims);
    // a 4 x 4 x 1 slab hovering 2 wu above the probes' lowest row
    let (lo, hi) = ([-2.0, -2.0, 3.0], [2.0, 2.0, 4.0]);
    let slab = cuboid(lo, hi);
    let (nodes, tris, order) = build_bvh(&slab);
    let dynamic = RtGiDynamic { nodes, tris, tri_mat: order.iter().map(|_| 0u32).collect(), mats: vec![[0.5, 0.5, 0.5, 0.0], [0.0; 4], [0.0; 4]] };
    assert!(g.set_dynamic(&queue, Some(&dynamic)), "dynamic set refused");
    let queued = g.invalidate_region(lo, hi);
    assert!(queued > 0 && queued < np, "region of {queued} / {np} probes");
    for _ in 0..REGION_PASSES {
        let mut enc = device.create_command_encoder(&Default::default());
        g.dispatch_part(&mut enc, &queue, None, 1, 0);
        queue.submit(Some(enc.finish()));
    }
    let after = g.read_probes(&device, &queue);
    let pos = |i: usize| [o[0] + (i as u32 % d[0]) as f32 * sp, o[1] + ((i as u32 / d[0]) % d[1]) as f32 * sp, o[2] + (i as u32 / (d[0] * d[1])) as f32 * sp];
    let (mut near_changed, mut near_n, mut far_n, mut far_same) = (0, 0, 0, 0);
    let mut under_drop = 0.0f32;
    for i in 0..np {
        let p = pos(i);
        let inside = (0..3).all(|a| p[a] >= lo[a] - 2.0 * sp - 1e-3 && p[a] <= hi[a] + 2.0 * sp + 1e-3);
        let same = (0..9).all(|k| (0..3).all(|c| before[i].total[k][c] == after[i].total[k][c]));
        if inside {
            near_n += 1;
            if !same { near_changed += 1; }
            // the probe straight under the slab lost sky
            if p[0].abs() < 1.6 && p[1].abs() < 1.6 && p[2] < lo[2] {
                under_drop = under_drop.max(1.0 - after[i].sky[0][1] / before[i].sky[0][1]);
            }
        } else {
            far_n += 1;
            if same { far_same += 1; }
        }
    }
    assert!(near_n >= 5 && far_n >= 20, "region {near_n} probes, outside {far_n}");
    assert_eq!(far_same, far_n, "{} probes outside the edit region changed", far_n - far_same);
    assert!(near_changed * 2 >= near_n, "only {near_changed} of {near_n} probes in the region changed");
    assert!(under_drop > 0.15, "the probes under the slab lost only {:.1} % of their sky", under_drop * 100.0);
    eprintln!("edit: {near_n} probes in the region ({near_changed} changed), {far_n} outside (all unchanged); sky under the slab -{:.0} %", under_drop * 100.0);

    // removing the slab restores the sky-direct SH (deterministic) of every probe in the region
    assert!(g.set_dynamic(&queue, None));
    g.invalidate_region(lo, hi);
    for _ in 0..REGION_PASSES {
        let mut enc = device.create_command_encoder(&Default::default());
        g.dispatch_part(&mut enc, &queue, None, 1, 0);
        queue.submit(Some(enc.finish()));
    }
    let restored = g.read_probes(&device, &queue);
    for i in 0..np {
        for k in 0..9 { for c in 0..3 {
            assert!((restored[i].sky[k][c] - before[i].sky[k][c]).abs() < 1e-4, "probe {i} sky SH[{k}][{c}] not restored");
        } }
    }
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_a_map_without_a_sky_list_keeps_the_legacy_sky() {
    let (device, queue) = soft_device();
    let sc = empty_scene(None);
    let (g, probes) = run(&device, &queue, &sc, 6, None);
    assert!(!g.engine, "no sky list must fall back to the legacy model");
    assert_eq!(g.params.bias[3], 1.3, "the legacy fallback keeps its fitted 1.3 gain");
    // legacy: escaped rays see the flat sky ambient 0.5 above, 0.25x below: P = 2 pi L (1 + 0.25)
    let want = 2.0 * std::f64::consts::PI * 0.5 * 1.25;
    for p in probes.iter().filter(|p| p.valid) {
        let dc = 3.5449077 * p.total[0][1] as f64;
        assert!((dc - want).abs() / want < 0.05, "legacy P_g = {dc} want {want}");
    }
}
