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

// ---------------------------------------------------------------------------------------------------------
// Ray-traced sun shadow cascades (`RtGi::update_sun` + the shading-side `gi_sun_vis`)
// ---------------------------------------------------------------------------------------------------------

fn sun_vis_at(device: &wgpu::Device, queue: &wgpu::Queue, rt: &RtGi, pts: &[([f32; 3], [f32; 3])]) -> Vec<f32> {
    rt.sample_sun_vis(device, queue, pts, false)
}

/// Does the ray `p + s*l` (s > 0) cross the box? (slab method)
fn ray_hits_box(p: [f32; 3], l: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> bool {
    let (mut t0, mut t1) = (0.0f32, f32::MAX);
    for a in 0..3 {
        if l[a].abs() < 1e-9 { if p[a] < lo[a] || p[a] > hi[a] { return false; } continue; }
        let (mut ta, mut tb) = ((lo[a] - p[a]) / l[a], (hi[a] - p[a]) / l[a]);
        if ta > tb { std::mem::swap(&mut ta, &mut tb); }
        t0 = t0.max(ta); t1 = t1.min(tb);
        if t0 > t1 { return false; }
    }
    true
}

/// Rotate `l` by `ang` about the unit axis `ax` (Rodrigues).
fn rotate(l: [f32; 3], ax: [f32; 3], ang: f32) -> [f32; 3] {
    let (c, s) = (ang.cos(), ang.sin());
    let d = ax[0] * l[0] + ax[1] * l[1] + ax[2] * l[2];
    let x = [ax[1] * l[2] - ax[2] * l[1], ax[2] * l[0] - ax[0] * l[2], ax[0] * l[1] - ax[1] * l[0]];
    [l[0] * c + x[0] * s + ax[0] * d * (1.0 - c), l[1] * c + x[1] * s + ax[1] * d * (1.0 - c), l[2] * c + x[2] * s + ax[2] * d * (1.0 - c)]
}

/// Sun-visible / shadowed floor points by ANALYTIC ray-vs-box geometry; `certain` only when every ray inside a
/// cone 3x wider than the sun's agrees (so the soft edge can not decide the gate).
fn floor_truth(l: [f32; 3], lo: [f32; 3], hi: [f32; 3], half: f32, x_range: f32) -> Vec<([f32; 3], Option<bool>)> {
    let (bx, by) = sun_basis(l);
    let mut out = Vec::new();
    let n = (2.0 * x_range / 0.25) as i32;
    for iy in 0..=n { for ix in 0..=n {
        let p = [-x_range + ix as f32 * 0.25, -x_range + iy as f32 * 0.25, 0.0];
        let mut votes = [false; 2];
        for (k, (ax, a)) in [(bx, 0.0f32), (bx, 3.0 * half), (bx, -3.0 * half), (by, 3.0 * half), (by, -3.0 * half)].iter().enumerate() {
            let r = rotate(l, *ax, *a);
            let h = ray_hits_box(p, r, lo, hi);
            if k == 0 { votes = [h, h]; } else if h != votes[0] { votes[1] = !votes[0]; }
        }
        out.push((p, if votes[0] == votes[1] { Some(votes[0]) } else { None }));
    } }
    out
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_the_sun_shadow_follows_the_geometry() {
    let (device, queue) = soft_device();
    let (lo, hi) = ([-2.0f32, -2.0, 3.0], [2.0f32, 2.0, 4.0]);
    let mut sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    sc.tris.extend(cuboid(lo, hi));
    sc.tri_mat.extend(std::iter::repeat(0).take(12));
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &sc.build(), None, None);
    g.set_enabled(&queue, true);
    g.update_sun_now(&device, &queue, [0.0, 0.0, 1.0]);
    let l = { let s = sc.sky.as_ref().unwrap().sun.unwrap().dir; let n = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt(); [s[0] / n, s[1] / n, s[2] / n] };
    let half = sun_tan_half(sc.sky.as_ref().unwrap().sun.unwrap().omega).atan();
    let truth = floor_truth(l, lo, hi, half, 8.0);
    let pts: Vec<_> = truth.iter().map(|(p, _)| (*p, [0.0, 0.0, 1.0])).collect();
    let vis = sun_vis_at(&device, &queue, &g, &pts);
    let (mut lit_n, mut sh_n, mut lit_ok, mut sh_ok, mut soft) = (0, 0, 0, 0, 0);
    for ((p, t), v) in truth.iter().zip(&vis) {
        let frac = v * v; // the shader returns sqrt(visible fraction)
        match t {
            Some(false) => { lit_n += 1; if frac > 0.95 { lit_ok += 1; } else { eprintln!("lit point {p:?} got {frac}"); } }
            Some(true) => { sh_n += 1; if frac < 0.05 { sh_ok += 1; } else { eprintln!("shadowed point {p:?} got {frac}"); } }
            None => { if frac > 0.05 && frac < 0.95 { soft += 1; } }
        }
    }
    eprintln!("sun shadow: {lit_n} certain-lit ({lit_ok} ok), {sh_n} certain-shadow ({sh_ok} ok), {soft} penumbra points");
    assert!(lit_n > 500 && sh_n > 50, "scene too small to mean anything: {lit_n} lit / {sh_n} shadowed");
    assert_eq!((lit_ok, sh_ok), (lit_n, sh_n), "the traced sun shadow disagrees with analytic ray geometry");
    assert!(soft >= 1, "no penumbra at all: the sun disc has no effect");
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_a_moved_piece_moves_its_sun_shadow() {
    let (device, queue) = soft_device();
    let sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &sc.build(), None, None);
    g.set_enabled(&queue, true);
    g.update_sun_now(&device, &queue, [0.0, 0.0, 1.0]);
    let l = { let s = sc.sky.as_ref().unwrap().sun.unwrap().dir; [s[0], s[1], s[2]] };
    let spot = |lo: [f32; 3]| -> [f32; 3] { // the floor point under a 2x2x1 slab at `lo`, along the sun ray
        let c = [lo[0] + 1.0, lo[1] + 1.0, lo[2] + 0.5];
        let s = c[2] / l[2];
        [c[0] - l[0] * s, c[1] - l[1] * s, 0.0]
    };
    let place = |g: &mut RtGi, lo: [f32; 3]| {
        let slab = cuboid(lo, [lo[0] + 2.0, lo[1] + 2.0, lo[2] + 1.0]);
        let (nodes, tris, order) = build_bvh(&slab);
        let d = RtGiDynamic { nodes, tris, tri_mat: order.iter().map(|_| 0u32).collect(), mats: vec![[0.5, 0.5, 0.5, 0.0], [0.0; 4], [0.0; 4]] };
        assert!(g.set_dynamic(&queue, Some(&d)));
        g.update_sun_now(&device, &queue, [0.0, 0.0, 1.0]);
    };
    let (a, b) = ([-6.0, -4.0, 3.0], [4.0, 5.0, 3.0]);
    let q = [([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]); 1];
    let _ = q;
    let pts = [(spot(a), [0.0, 0.0, 1.0]), (spot(b), [0.0, 0.0, 1.0])];
    let before = sun_vis_at(&device, &queue, &g, &pts);
    place(&mut g, a);
    let at_a = sun_vis_at(&device, &queue, &g, &pts);
    place(&mut g, b);
    let at_b = sun_vis_at(&device, &queue, &g, &pts);
    eprintln!("floor sun visibility (under A, under B): empty {before:?}, piece at A {at_a:?}, piece at B {at_b:?}");
    assert!(before[0] > 0.98 && before[1] > 0.98, "no piece: lit everywhere");
    assert!(at_a[0] < 0.2 && at_a[1] > 0.98, "piece at A: shadow under A only");
    assert!(at_b[1] < 0.2 && at_b[0] > 0.98, "piece moved to B: the shadow moved with it");
    // an OBJECT receiver sees the static map only: a placed piece's coarse physics hull must not shadow the
    // surface of the very object it came from (the raster shadow pass handles piece-on-object)
    let as_object = g.sample_sun_vis(&device, &queue, &pts, true);
    assert!(as_object[0] > 0.98 && as_object[1] > 0.98, "object receivers must ignore the placed-piece layer: {as_object:?}");
}

/// Fraction of the sun DISC (a cone of half angle `half`) visible from `p`, by brute-force ray-vs-box over 240
/// uniformly spread directions in the cone — the quantity the Tool's sun rays estimate (`faux_light_sky.cpp`).
fn disc_fraction(p: [f32; 3], l: [f32; 3], half: f32, lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let (bx, by) = sun_basis(l);
    let th = half.tan();
    let n = 240;
    let mut vis = 0;
    for k in 0..n {
        let r = ((k as f32 + 0.5) / n as f32).sqrt() * th;
        let a = k as f32 * 2.399963;
        let d = [l[0] + (bx[0] * a.cos() + by[0] * a.sin()) * r, l[1] + (bx[1] * a.cos() + by[1] * a.sin()) * r, l[2] + (bx[2] * a.cos() + by[2] * a.sin()) * r];
        let m = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if !ray_hits_box(p, [d[0] / m, d[1] / m, d[2] / m], lo, hi) { vis += 1; }
    }
    vis as f32 / n as f32
}

/// The soft edge IS the sun disc: a slab 12 wu above the floor throws a ~0.7 wu penumbra, and the traced +
/// filtered visibility must follow the analytic disc fraction through it (mean error over the penumbra points),
/// not a hard edge and not a fixed blur. (The saboteur is a fixed 1-texel filter: hard shadow.)
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_the_penumbra_follows_the_sun_disc() {
    let (device, queue) = soft_device();
    let (lo, hi) = ([-2.0f32, -2.0, 11.0], [2.0f32, 2.0, 12.0]);
    let mut sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    sc.tris.extend(cuboid(lo, hi));
    sc.tri_mat.extend(std::iter::repeat(0).take(12));
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &sc.build(), None, None);
    g.set_enabled(&queue, true);
    let sun = sc.sky.as_ref().unwrap().sun.unwrap();
    let l = { let n = (sun.dir[0] * sun.dir[0] + sun.dir[1] * sun.dir[1] + sun.dir[2] * sun.dir[2]).sqrt(); [sun.dir[0] / n, sun.dir[1] / n, sun.dir[2] / n] };
    let half = sun_tan_half(sun.omega).atan();
    g.update_sun_now(&device, &queue, [0.0, 0.0, 1.0]);
    // a line of floor points straight across the shadow's trailing edge (along the sun's ground direction)
    let mut pts = Vec::new();
    let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let s = c[2] / l[2];
    let centre = [c[0] - l[0] * s, c[1] - l[1] * s];
    for i in -150..=150 { let t = i as f32 * 0.03; pts.push(([centre[0] + t * 0.8, centre[1] - t * 0.6, 0.0], [0.0f32, 0.0, 1.0])); }
    let vis = g.sample_sun_vis(&device, &queue, &pts, false);
    let (mut err, mut n, mut width) = (0.0f32, 0, 0);
    for ((p, _), v) in pts.iter().zip(&vis) {
        let truth = disc_fraction(*p, l, half, lo, hi);
        if truth > 0.02 && truth < 0.98 { err += (v * v - truth).abs(); n += 1; }
        if v * v > 0.05 && v * v < 0.95 { width += 1; }
    }
    eprintln!("penumbra: {n} analytic penumbra points, mean |traced - disc| {:.3}, traced penumbra spans {width} of the 301 samples (0.03 wu apart)", err / n.max(1) as f32);
    assert!(n >= 15, "the analytic penumbra is only {n} samples wide: the test line misses the edge");
    assert!(err / (n as f32) < 0.12, "the traced penumbra does not follow the sun disc: mean error {:.3}", err / n as f32);
}

/// The first frame after the mode is switched on traces ONE cascade; the others hold zeros. A point that only a
/// not-yet-traced cascade covers must read as LIT (a zero distance would read as an occluder at the sun).
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_an_untraced_cascade_is_not_shadow() {
    let (device, queue) = soft_device();
    let sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &sc.build(), None, None);
    g.set_enabled(&queue, true);
    let mut enc = device.create_command_encoder(&Default::default());
    g.update_sun(&mut enc, &queue, [0.0, 0.0, 1.0]); // traces cascade 0 only
    queue.submit(Some(enc.finish()));
    let pts = [([3.0, 2.0, 0.0], [0.0, 0.0, 1.0]), ([100.0, 40.0, 0.0], [0.0, 0.0, 1.0]), ([300.0, -200.0, 0.0], [0.0, 0.0, 1.0])];
    let v = g.sample_sun_vis(&device, &queue, &pts, false);
    assert!(v.iter().all(|x| *x > 0.99), "an untraced cascade shaded as shadow: {v:?}");
}

/// A shadow edge must be SMOOTH along its own length: no texel-scale steps, no per-pixel grain. The coarsest
/// cascade's texel is 0.84 wu; samples are 0.06 wu apart, so on a smooth edge neighbouring samples differ by a few
/// hundredths while a nearest-texel filter (steps of the texel size) or a per-pixel-rotated 16-tap pattern
/// (grain of +-0.12) jumps by 0.1 - 0.3. The slow drift of the level along the edge (<= +-0.13) is the depth map's
/// edge-position quantisation of up to half a texel and is NOT what this gate measures.
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_a_shadow_edge_is_smooth_along_its_length() {
    let (device, queue) = soft_device();
    let (lo, hi) = ([-2.0f32, -2.0, 11.0], [2.0f32, 2.0, 12.0]);
    let mut sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    sc.tris.extend(cuboid(lo, hi));
    sc.tri_mat.extend(std::iter::repeat(0).take(12));
    let mut g = RtGi::new(&device);
    g.set_scene(&device, &queue, &sc.build(), None, None);
    g.set_enabled(&queue, true);
    let sun = sc.sky.as_ref().unwrap().sun.unwrap();
    let l = { let n = (sun.dir[0] * sun.dir[0] + sun.dir[1] * sun.dir[1] + sun.dir[2] * sun.dir[2]).sqrt(); [sun.dir[0] / n, sun.dir[1] / n, sun.dir[2] / n] };
    let half = sun_tan_half(sun.omega).atan();
    // the camera is 300 wu away: only the coarsest cascade (window half extent 432) covers the slab
    g.update_sun_now(&device, &queue, [300.0, 0.0, 1.0]);
    // find, at y = 4, the x where the analytic disc fraction crosses 0.5 (the trailing edge of the shadow)
    let (mut a, mut b) = (-12.0f32, 6.0f32);
    for _ in 0..40 { let m = 0.5 * (a + b); if disc_fraction([m, 4.0, 0.0], l, half, lo, hi) < 0.5 { a = m } else { b = m } }
    let x_edge = 0.5 * (a + b);
    let mut pts = Vec::new();
    for i in 0..40 { pts.push(([x_edge, 2.8 + i as f32 * 0.06, 0.0], [0.0f32, 0.0, 1.0])); }
    let v = g.sample_sun_vis(&device, &queue, &pts, false);
    let f: Vec<f32> = v.iter().map(|x| x * x).collect();
    let (mn, mx) = f.iter().fold((1.0f32, 0.0f32), |(a, b), x| (a.min(*x), b.max(*x)));
    for (i, ((p, _), x)) in pts.iter().zip(&f).enumerate() { if i % 4 == 0 { eprintln!("  y {:.2} traced {x:.3} truth {:.3}", p[1], disc_fraction(*p, l, half, lo, hi)); } }
    eprintln!("edge smoothness: lit fraction along the edge min {mn:.3} max {mx:.3} peak-to-peak {:.3}", mx - mn);
    let step = f.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    eprintln!("edge smoothness: largest step between neighbouring samples {step:.3}");
    assert!(step < 0.04, "the shadow edge steps / has grain: {step:.3} between samples 0.06 wu apart");
    assert!(mx - mn < 0.4, "the shadow edge level wanders by {:.3} along a straight edge", mx - mn);
}

/// A surface whose whole probe cell is invalid used to read BLACK (`gi_irradiance` returned 0 when no corner survived).
/// With only ONE valid probe in the grid, a point in a far cell must still receive that probe's lighting, not zero.
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_a_cell_with_no_valid_probe_is_not_black() {
    let (device, queue) = soft_device();
    let sc = empty_scene(Some(test_sky(0.3, 1.0)));
    let (g, probes) = run(&device, &queue, &sc, 3, None);
    let n = probes.len();
    assert!(n > 100 && probes.iter().all(|p| p.valid));
    let pt = [([-1.5f32, -1.5, 2.2], [0.0f32, 0.0, 1.0])];
    let ctrl = g.sample_irradiance(&device, &queue, &pt)[0];
    assert!(ctrl[0] > 0.01, "control: an open-air point must be lit, got {ctrl:?}");
    // invalidate every probe but the first (the grid corner nearest the origin of the box)
    let zero = [0.0f32; 4];
    for pi in 1..n { queue.write_buffer(g.probes_buf(), ((pi * 28 + 9) * 16) as u64, bytemuck::cast_slice(&zero)); }
    queue.submit(None);
    let got = g.sample_irradiance(&device, &queue, &pt)[0];
    eprintln!("control {ctrl:?}, one-valid-probe {got:?}");
    assert!(got[0] > 0.01 && got[1] > 0.01 && got[2] > 0.01, "a cell with no valid corner was shaded black: {got:?}");
}

// ---------------------------------------------------------------------------------------------------------------
// The camera-centred INNER probe grid (rtgi.rs `InnerGrid`).
// ---------------------------------------------------------------------------------------------------------------

/// Floor + a roof slab 2.6..3.0 wu up (a sky shadow with a hard edge) in a 6 x 6 x 4 box; the outer grid is 1 wu.
fn roofed_scene() -> Scene {
    let mut sc = floor_scene(0.45, Some(test_sky(0.3, 1.0)));
    for t in cuboid([-1.5, -1.5, 2.6], [1.5, 1.5, 3.0]) { sc.tris.push(t); sc.tri_mat.push(0); }
    sc
}

const TEST_INNER: InnerGrid = InnerGrid { spacing: 0.5, dims: [24, 24, 16] };

fn run_inner(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, grid: Option<InnerGrid>, cam: [f32; 3], passes: u32) -> RtGi {
    std::env::set_var("HMS_RTGI_MIN_SPACING", "1.0");
    let mut g = RtGi::new(device);
    g.inner_policy = match grid { Some(i) => InnerPolicy::Fixed(i), None => InnerPolicy::Off };
    g.set_scene(device, queue, &scene.build(), None, None);
    g.set_enabled(queue, true);
    g.focus(cam);
    g.warm_up(device, queue, passes);
    g
}

/// cell -> sky-direct SH (lum of the DC term and the 9 coefficients of green) for every inner probe of a field.
fn inner_cells(g: &RtGi, probes: &[ProbeRecord]) -> std::collections::HashMap<[i32; 3], (bool, [f32; 9])> {
    let (gr, base, _) = g.inner_window().expect("inner grid");
    let mut m = std::collections::HashMap::new();
    for slot in 0..gr.count() {
        let p = &probes[(g.n_outer + slot) as usize];
        let mut sh = [0f32; 9];
        for k in 0..9 { sh[k] = p.sky[k][1]; }
        m.insert(RtGi::inner_cell_of(&gr, base, slot), (p.valid, sh));
    }
    m
}

#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_the_inner_grid_probes_are_the_engine_sky_where_the_window_says() {
    let (device, queue) = soft_device();
    let mut sc = roofed_scene();
    // lattice origin at the world origin, so the roof's cells have NEGATIVE coordinates (the toroidal wrap must handle them)
    sc.bounds = ([0.0, 0.0, 1.0], [3.0, 3.0, 5.0]);
    let a = run_inner(&device, &queue, &sc, Some(TEST_INNER), [0.0, 0.0, 0.5], 6);
    assert!(a.inner.is_some() && a.n_outer > 0, "inner grid not active: {:?} outer {}", a.inner, a.n_outer);
    assert_eq!(a.nprobes, a.n_outer + TEST_INNER.count());
    let pa = a.read_probes(&device, &queue);
    let ca = inner_cells(&a, &pa);
    // shaded cells exist (under the roof) and open ones exist
    let dc: Vec<f32> = ca.values().map(|v| v.1[0]).collect();
    let (lo, hi) = (dc.iter().cloned().fold(f32::MAX, f32::min), dc.iter().cloned().fold(f32::MIN, f32::max));
    let nz = dc.iter().filter(|v| **v > 1e-4).count();
    let nvalid = ca.values().filter(|v| v.0).count();
    eprintln!("inner cells: {} valid of {}, {} with sky light, DC {lo:.4}..{hi:.4}", nvalid, ca.len(), nz);
    assert!(lo < 0.7 * hi, "no sky shadow in the inner grid: DC {lo}..{hi}");
    // absolute check of the cell <-> slot mapping: the cells ABOVE the roof see the whole sky, those under its middle a fraction
    {
        let o = a.params.origin;
        let sp = TEST_INNER.spacing;
        let pos = |c: &[i32; 3]| [o[0] + c[0] as f32 * sp, o[1] + c[1] as f32 * sp, o[2] + c[2] as f32 * sp];
        let (mut above, mut below) = (Vec::new(), Vec::new());
        for (c, (valid, sh)) in &ca {
            let p = pos(c);
            if !*valid { continue; }
            if p[2] > 3.3 && p[2] < 3.9 { above.push(sh[0]); }
            if p[0].abs() < 0.8 && p[1].abs() < 0.8 && p[2] > 1.0 && p[2] < 2.3 { below.push(sh[0]); }
        }
        assert!(above.len() > 20 && below.len() > 5, "cells above {} / below {}", above.len(), below.len());
        let ma = above.iter().sum::<f32>() / above.len() as f32;
        let mb = below.iter().sum::<f32>() / below.len() as f32;
        eprintln!("cells above the roof: mean sky DC {ma:.3} ({} cells); under its middle: {mb:.3} ({} cells)", above.len(), below.len());
        assert!(mb < 0.7 * ma, "the probes under the roof are not darker than those above it ({mb:.3} vs {ma:.3}): the slot <-> cell mapping is wrong");
    }
    // the same window reached by MOVING: start elsewhere, focus to the same place, trace again
    let b = run_inner(&device, &queue, &sc, Some(TEST_INNER), [-2.0, -2.0, 0.5], 6);
    b.focus([0.0, 0.0, 0.5]);
    assert!(b.inner_window().unwrap().1 != RtGi::default_probe_base_for_test(), "window did not move");
    b.warm_up(&device, &queue, 8);
    let pb = b.read_probes(&device, &queue);
    let cb = inner_cells(&b, &pb);
    let (mut n, mut worst) = (0usize, 0f32);
    for (cell, (va, sa)) in &ca {
        let Some((vb, sb)) = cb.get(cell) else { continue };
        n += 1;
        // (a probe inside the roof relocates / is declared invalid by a stateful search that can settle differently; compare what both trust)
        if !(*va && *vb) { continue; }
        for k in 0..9 { worst = worst.max((sa[k] - sb[k]).abs() / (0.05 + sa[0].abs())); }
    }
    assert!(n >= 300, "only {n} common valid cells");
    assert!(worst < 0.02, "cells of a moved window differ from a fresh one: worst relative SH difference {worst:.4}");
    eprintln!("inner grid: {} cells in common between a fresh and a moved window, worst relative sky-SH difference {worst:.2e}", n);
}

fn path_samples(g: &RtGi, device: &wgpu::Device, queue: &wgpu::Queue, from: [f32; 3], to: [f32; 3], n: usize) -> Vec<f64> {
    let pts: Vec<([f32; 3], [f32; 3])> = (0..n).map(|i| { let t = i as f32 / (n - 1) as f32; ([from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t, from[2] + (to[2] - from[2]) * t], [0.0, 0.0, 1.0]) }).collect();
    g.sample_irradiance(device, queue, &pts).iter().map(|c| c[1] as f64).collect()
}

/// A walk across the inner window's edge must not show a seam: neighbouring samples 0.02 wu apart differ by a smooth
/// amount whether or not the path crosses the fade to the coarse grid.
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_there_is_no_seam_where_the_inner_grid_meets_the_coarse_one() {
    let (device, queue) = soft_device();
    // a thin slab 0.5 wu over the floor (x 0..2, y -1..1): the fine and the coarse grid resolve its sky shadow differently,
    // and it sits inside the fade zone of the window's +x edge
    let mut sc = roofed_scene();
    for t in cuboid([0.0, -1.0, 0.5], [2.0, 1.0, 0.7]) { sc.tris.push(t); sc.tri_mat.push(0); }
    let g = run_inner(&device, &queue, &sc, Some(TEST_INNER), [-4.0, 0.0, 0.5], 12);
    let coarse = run_inner(&device, &queue, &sc, None, [0.0; 3], 12);
    // y = 2.0 is beyond the roof (shadow edge at |y| = 1.5), the path runs along x over the whole box and out of the window
    let n = 300;
    let (a, b) = ([-2.9, 0.0, 0.05], [2.9, 0.0, 0.05]);
    let e = path_samples(&g, &device, &queue, a, b, n);
    let e0 = path_samples(&coarse, &device, &queue, a, b, n);
    let step = |v: &[f64]| v.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max);
    let mean = e.iter().sum::<f64>() / n as f64;
    let (s, s0) = (step(&e), step(&e0));
    let dmax = e.iter().zip(&e0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("seam walk: mean E {mean:.3}; largest step with the inner grid {s:.4}, coarse grid alone {s0:.4}; largest inner-vs-coarse difference {dmax:.4}");
    assert!(dmax > 0.02, "the inner grid changes nothing on this walk ({dmax:.4}): the seam gate would be vacuous");
    {
        let (i, st) = e.windows(2).enumerate().map(|(i, w)| (i, (w[1] - w[0]).abs())).fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
        eprintln!("largest step at x = {:.2}: {:.4} -> {:.4} (coarse alone {:.4} -> {:.4})", -2.9 + 5.8 * i as f32 / (n - 1) as f32, e[i], e[i + 1], e0[i], e0[i + 1]); let _ = st;
    }
    assert!(mean > 0.05, "dead scene");
    assert!(s < 1.25 * s0 + 0.004, "a seam: largest step {s:.4} (coarse alone {s0:.4}, mean {mean:.3})");
    // the same slab walk with the window centred where its probes sit in the slab and the floor (cells with no valid corner):
    // such a cell must hand over to the coarse grid smoothly, not jump
    let g2 = run_inner(&device, &queue, &sc, Some(TEST_INNER), [-1.0, 0.0, 0.5], 12);
    let e3 = path_samples(&g2, &device, &queue, a, b, n);
    let s3 = step(&e3);
    eprintln!("seam walk, window over the slab: largest step {s3:.4} (coarse alone {s0:.4})");
    assert!(s3 < 1.25 * s0 + 0.004, "a jump where a cell has no valid probe: step {s3:.4} (coarse alone {s0:.4})");
    // the same walk 1 wu to the side (clear of the slab)
    let e2 = path_samples(&g, &device, &queue, [-2.9, 2.0, 0.05], [2.9, 2.0, 0.05], n);
    let e20 = path_samples(&coarse, &device, &queue, [-2.9, 2.0, 0.05], [2.9, 2.0, 0.05], n);
    let (s2, s20) = (step(&e2), step(&e20));
    let m2 = e2.iter().sum::<f64>() / n as f64;
    eprintln!("seam walk under the roof: mean {m2:.3}; largest step {s2:.4} (coarse alone {s20:.4})");
    assert!(s2 < 0.06 * m2.max(mean) + s20, "a seam under the roof: step {s2:.4} (coarse {s20:.4})");
}

/// No swimming: once converged, further frames change the field by noise only (no more than the coarse grid's own),
/// a camera that drifts less than the hysteresis moves nothing, and a window move leaves the light at points deep inside
/// both windows where it was.
#[test]
#[ignore = "software GL (llvmpipe) compute"]
fn rtgi_gpu_the_inner_grid_does_not_swim() {
    let (device, queue) = soft_device();
    let sc = roofed_scene();
    let g = run_inner(&device, &queue, &sc, Some(TEST_INNER), [0.0, 0.0, 0.5], 16);
    let coarse = run_inner(&device, &queue, &sc, None, [0.0; 3], 16);
    let pts: Vec<([f32; 3], [f32; 3])> = (0..60).map(|i| { let a = i as f32 * 0.7; ([1.8 * a.cos(), 1.8 * a.sin(), 0.05 + 0.02 * (i % 5) as f32], [0.0, 0.0, 1.0]) }).collect();
    let frame_noise = |r: &RtGi| -> f64 {
        let mut prev: Vec<f64> = r.sample_irradiance(&device, &queue, &pts).iter().map(|c| c[1] as f64).collect();
        let mut worst = 0f64;
        for _ in 0..12 {
            let mut enc = device.create_command_encoder(&Default::default());
            r.dispatch_part(&mut enc, &queue, None, 1, 0);
            queue.submit(Some(enc.finish()));
            let cur: Vec<f64> = r.sample_irradiance(&device, &queue, &pts).iter().map(|c| c[1] as f64).collect();
            for (a, b) in prev.iter().zip(&cur) { worst = worst.max((a - b).abs() / a.max(0.05)); }
            prev = cur;
        }
        worst
    };
    let (ni, nc) = (frame_noise(&g), frame_noise(&coarse));
    eprintln!("frame-to-frame change at 60 floor points: inner grid {ni:.4}, coarse grid alone {nc:.4}");
    assert!(ni < 0.05 + 1.5 * nc, "the field swims: {ni:.4} per frame (coarse {nc:.4})");
    // drift below the hysteresis (one cell): nothing is queued, the window stays
    let base0 = g.inner_window().unwrap().1;
    assert_eq!(g.focus([0.5, 0.0, 0.5]), 0);
    assert_eq!(g.inner_window().unwrap().1, base0, "the window moved on a one-cell drift");
    // a real move (2 cells) re-owns only the cells at the trailing/leading slab
    let moved = g.focus([1.0, 0.0, 0.5]);
    assert!(moved > 0 && moved < TEST_INNER.count() as usize, "{moved} cells re-owned on a 2-cell move");
    let before: Vec<f64> = pts.iter().map(|_| 0.0).collect();
    let _ = before;
    let inside: Vec<([f32; 3], [f32; 3])> = (0..40).map(|i| { let a = i as f32 * 0.5; ([0.4 + 0.6 * a.cos(), 0.6 * a.sin(), 0.05], [0.0, 0.0, 1.0]) }).collect();
    let e_before: Vec<f64> = coarse_free(&g, &device, &queue, &inside);
    g.warm_up(&device, &queue, 8);
    let e_after: Vec<f64> = coarse_free(&g, &device, &queue, &inside);
    let worst = e_before.iter().zip(&e_after).map(|(a, b)| (a - b).abs() / a.max(0.05)).fold(0.0, f64::max);
    eprintln!("a 2-cell window move changes the light at 40 interior points by at most {worst:.4} (relative)");
    assert!(worst < 0.08, "the light moved with the window: {worst:.4}");
}
fn coarse_free(g: &RtGi, device: &wgpu::Device, queue: &wgpu::Queue, pts: &[([f32; 3], [f32; 3])]) -> Vec<f64> {
    g.sample_irradiance(device, queue, pts).iter().map(|c| c[1] as f64).collect()
}
