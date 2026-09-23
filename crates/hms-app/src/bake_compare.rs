//! Per-texel comparison harness — OUR GPU path-traced lightmap bake vs the SHIPPED
//! lightmap atlas of the loaded map. Headless: `HMS_BAKE_COMPARE=<out_dir|1>` (+ HMS_BAKE_SAMPLES /
//! HMS_BAKE_BOUNCES). Measurement only — the light model itself lives in `lightbake_gpu.rs`.
//!
//! Units (read `lightbake_gpu::encode_dual_vmf` + `scene::lm_texel_decode`):
//!   * SHIPPED: lobes `dom_rgb`/`fill_rgb` already × hdr·K. Irradiance for a normal n is the engine
//!     `dual_vmf_diffuse` = (LUT(n·dir,bw)·dom + 0.25·fill)/π. At render time the mesh shader ADDS the
//!     analytical sun `sun_col·sat(n·sun)·vis²/π` (vis = DM visibility) — so the *rendered* shipped
//!     lighting is `E_lm + E_sun`. We evaluate both (`ship_lm`, `ship`).
//!   * OURS: `GpuTexelOut.ambient` = the tracer's total (direct sun + lights + emitters + indirect)
//!     irradiance estimate at the texel normal (`E_raw`). `encode_dual_vmf(k)` is built so that the
//!     shipped decode (hdr=1, K=k) reproduces `E_enc(n) = ambient + LUT(n·dom_dir,bw)·dom_color`
//!     (fill = ambient·4π, dom = dom_color·π, both ÷k → ×k on decode). We evaluate both and report the
//!     best-fit scale `k` (least squares in log space) rather than assuming 1.
//!   Primary metric: luminance of E(n) at the texel's own surface normal, ours(raw) vs shipped(rendered).

use glam::Vec3;
use std::collections::HashMap;

use crate::scene::{BspLmMesh, SceneController};

const PI: f32 = std::f32::consts::PI;

fn lum(c: [f32; 3]) -> f32 { 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] }
fn edge(a: glam::Vec2, b: glam::Vec2, p: glam::Vec2) -> f32 { (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) }

/// One covered texel's evaluation (compact; a few million of these per map).
#[derive(Clone, Copy)]
struct Rec {
    sub: u16,
    mat: u16,
    /// 0 = up-facing (n.z > 0.5), 1 = wall, 2 = down-facing (n.z < -0.5)
    nclass: u8,
    ship: f32,     // lum of shipped RENDERED irradiance at n (lightmap + analytical sun×vis²)
    ship_lm: f32,  // lum of shipped LIGHTMAP-only irradiance at n
    ours: f32,     // lum of our raw `ambient` (E at n)
    ours_enc: f32, // lum of what the encode→decode round trip renders at n
    dot: f32,      // dominant-direction agreement
    bw_s: f32,
    bw_o: f32,
    fill_frac_s: f32, // shipped 0.25·fill/π ÷ E_lm (isotropic share)
    vis: f32,
    pos: [f32; 3],
}

#[derive(Default)]
struct Group {
    n: usize,
    logr: Vec<f32>, // ln(ours/ship) (unscaled), for the median
    sum_ship: f64,
    sum_ours: f64,
    sum_dot: f64,
    sum_bw_s: f64,
    sum_bw_o: f64,
    sum_fill: f64,
    sum_pos: [f64; 3],
    zmin: f32,
    zmax: f32,
}
impl Group {
    fn push(&mut self, r: &Rec, lr: f32) {
        if self.n == 0 { self.zmin = r.pos[2]; self.zmax = r.pos[2]; }
        self.n += 1;
        self.logr.push(lr);
        self.sum_ship += r.ship as f64;
        self.sum_ours += r.ours as f64;
        self.sum_dot += r.dot as f64;
        self.sum_bw_s += r.bw_s as f64;
        self.sum_bw_o += r.bw_o as f64;
        self.sum_fill += r.fill_frac_s as f64;
        for k in 0..3 { self.sum_pos[k] += r.pos[k] as f64; }
        self.zmin = self.zmin.min(r.pos[2]);
        self.zmax = self.zmax.max(r.pos[2]);
    }
    /// (median ratio unscaled, log-RMS after `ls` scale, fraction within ±25% after scale)
    fn stats(&mut self, ls: f64) -> (f64, f64, f64) {
        if self.n == 0 { return (0.0, 0.0, 0.0); }
        self.logr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = self.logr[self.n / 2] as f64;
        let (mut s2, mut within) = (0f64, 0usize);
        let (lo, hi) = ((0.75f64).ln(), (1.25f64).ln());
        for &l in &self.logr {
            let e = l as f64 + ls;
            s2 += e * e;
            if e >= lo && e <= hi { within += 1; }
        }
        (med.exp(), (s2 / self.n as f64).sqrt(), within as f64 / self.n as f64)
    }
}

/// Run the GPU bake at (samples, bounces) and compare every covered texel with the shipped atlas.
/// Returns the text report; writes side-by-side PNGs (shipped | ours×k | log-ratio heatmap) per submap
/// into `out_dir` as `bake_cmp_<map>_<spp>spp_<tag>_<sub>.png`.
pub fn run(
    scene: &SceneController,
    baker: &hms_render::lightbake_gpu::GpuLightBaker,
    device: &eframe::wgpu::Device,
    queue: &eframe::wgpu::Queue,
    samples: u32,
    bounces: u32,
    out_dir: &str,
    map_stem: &str,
) -> String {
    let mut rep = String::new();
    macro_rules! out { ($($t:tt)*) => { rep.push_str(&format!($($t)*)); rep.push('\n'); } }

    let prog = std::sync::atomic::AtomicU32::new(0);
    let t0 = std::time::Instant::now();
    let (grids, bake_ms) = scene.bake_pathtraced_gpu(baker, device, queue, samples, bounces, &prog);
    let (sun_dir, sun_col, sky_col) = scene.bake_sun_params();
    out!("BAKE_COMPARE map={map_stem} samples={samples} bounces={bounces}: {} submaps baked in {:.0} ms; sun_dir=({:.3},{:.3},{:.3}) sun_col=({:.2},{:.2},{:.2}) sky_col=({:.2},{:.2},{:.2})",
        grids.len(), bake_ms, sun_dir.x, sun_dir.y, sun_dir.z, sun_col[0], sun_col[1], sun_col[2], sky_col[0], sky_col[1], sky_col[2]);

    let geom: &[BspLmMesh] = scene.lm_geom();
    let mut by_sub: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, m) in geom.iter().enumerate() { by_sub.entry(m.dm).or_default().push(i); }
    let mut subs: Vec<(u32, u32)> = by_sub.keys().copied().collect();
    subs.sort();

    // material id table (diffuse tag → dense id + name)
    let mut mat_ids: HashMap<u32, u16> = HashMap::new();
    let mut mat_names: Vec<String> = Vec::new();
    let mut mat_of = |tag: u32| -> u16 {
        *mat_ids.entry(tag).or_insert_with(|| {
            let n = scene.tag_name_of(tag);
            mat_names.push(if n.is_empty() { format!("{tag:#010x}") } else { n });
            (mat_names.len() - 1) as u16
        })
    };

    let mut recs: Vec<Rec> = Vec::new();
    // per-submap image buffers (rgb shipped rendered, rgb ours raw, coverage/flags) kept for the PNGs
    struct SubImg { w: u32, h: u32, ship: Vec<[f32; 3]>, ours: Vec<[f32; 3]>, flag: Vec<u8> } // flag 0 uncovered,1 ok,2 ship≈0,3 ours≈0,4 both≈0
    let mut imgs: Vec<((u32, u32), SubImg)> = Vec::new();
    let (mut n_uncov_bake, mut n_noimg) = (0usize, 0usize);
    const ZERO: f32 = 1e-4;

    for (si, sub) in subs.iter().enumerate() {
        let Some((grid, w, h)) = grids.get(sub) else { continue };
        let (wu, hu) = (*w as usize, *h as usize);
        let mesh_ids = &by_sub[sub];
        // Re-raster the texel G-buffer exactly as the bake does (pos, normal, owning mesh).
        let mut gpos = vec![[0f32; 3]; wu * hu];
        let mut gnrm = vec![[0f32; 3]; wu * hu];
        let mut gmesh = vec![u32::MAX; wu * hu];
        for &mi in mesh_ids {
            let m = &geom[mi];
            for tri in m.indices.chunks_exact(3) {
                let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let (Some(v0), Some(v1), Some(v2)) = (m.verts.get(i0), m.verts.get(i1), m.verts.get(i2)) else { continue };
                let p = |v: &([f32; 3], [f32; 3], [f32; 2])| glam::Vec2::new(v.2[0] * wu as f32, v.2[1] * hu as f32);
                let (a, b, c) = (p(v0), p(v1), p(v2));
                let min_x = a.x.min(b.x).min(c.x).floor().max(0.0) as usize;
                let max_x = (a.x.max(b.x).max(c.x).ceil() as isize).clamp(0, wu as isize) as usize;
                let min_y = a.y.min(b.y).min(c.y).floor().max(0.0) as usize;
                let max_y = (a.y.max(b.y).max(c.y).ceil() as isize).clamp(0, hu as isize) as usize;
                let area = edge(a, b, c);
                if area.abs() < 1e-8 { continue; }
                let inv = 1.0 / area;
                for py in min_y..max_y {
                    for px in min_x..max_x {
                        let pt = glam::Vec2::new(px as f32 + 0.5, py as f32 + 0.5);
                        let (w0, w1, w2) = (edge(b, c, pt) * inv, edge(c, a, pt) * inv, edge(a, b, pt) * inv);
                        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) && (w0 > 0.0 || w1 > 0.0 || w2 > 0.0) { continue; }
                        let idx = py * wu + px;
                        if gmesh[idx] != u32::MAX { continue; }
                        let pos = Vec3::from(v0.0) * w0 + Vec3::from(v1.0) * w1 + Vec3::from(v2.0) * w2;
                        let nrm = (Vec3::from(v0.1) * w0 + Vec3::from(v1.1) * w1 + Vec3::from(v2.1) * w2).normalize_or_zero();
                        gpos[idx] = pos.into(); gnrm[idx] = nrm.into(); gmesh[idx] = mi as u32;
                    }
                }
            }
        }
        // shipped atlas images per mesh (all meshes of a submap share the DM; SDM/hdr/k are per mesh)
        let mut ship_imgs: HashMap<usize, Option<crate::scene::LmAtlasImages>> = HashMap::new();
        let mut img = SubImg { w: *w, h: *h, ship: vec![[0.0; 3]; wu * hu], ours: vec![[0.0; 3]; wu * hu], flag: vec![0u8; wu * hu] };
        for idx in 0..wu * hu {
            let mi = gmesh[idx];
            if mi == u32::MAX { continue; }
            let o = &grid[idx];
            if o.ambient[3] < 0.5 { n_uncov_bake += 1; continue; }
            let m = &geom[mi as usize];
            let si_ = ship_imgs.entry(mi as usize).or_insert_with(|| scene.shipped_lm_images(m.dm, m.sdm));
            let Some(si_) = si_ else { n_noimg += 1; continue };
            let n = Vec3::from(gnrm[idx]);
            let uv = [((idx % wu) as f32 + 0.5) / wu as f32, ((idx / wu) as f32 + 0.5) / hu as f32];
            let st = crate::scene::shipped_lm_texel(si_, m.hdr, m.k, uv);
            // SHIPPED: lightmap irradiance at n + the renderer's analytical sun (sun_col·sat(n·sun)·vis²/π)
            let e_lm = crate::lightprobe::dual_vmf_diffuse(n, Vec3::from(st.dom_dir), st.dom_rgb, st.fill_rgb, st.bandwidth);
            let ndl = n.dot(sun_dir).max(0.0);
            let vis = st.vis.clamp(0.0, 1.0);
            let e_ship = [e_lm[0] + sun_col[0] * ndl * vis * vis / PI, e_lm[1] + sun_col[1] * ndl * vis * vis / PI, e_lm[2] + sun_col[2] * ndl * vis * vis / PI];
            let e_fill = lum([0.25 * st.fill_rgb[0] / PI, 0.25 * st.fill_rgb[1] / PI, 0.25 * st.fill_rgb[2] / PI]);
            // OURS: raw irradiance + the encode→decode round trip (hdr=1, K=1)
            let e_raw = [o.ambient[0], o.ambient[1], o.ambient[2]];
            let odir = Vec3::from([o.dom[0], o.dom[1], o.dom[2]]).normalize_or_zero();
            let bw_o = o.dom[3].clamp(0.0, 1.0);
            let co = crate::lightprobe::vmf_diffuse_coeff(n.dot(odir), bw_o);
            let e_enc = [e_raw[0] + co * o.dom_color[0], e_raw[1] + co * o.dom_color[1], e_raw[2] + co * o.dom_color[2]];
            let (ls, lo) = (lum(e_ship), lum(e_raw));
            img.ship[idx] = e_ship; img.ours[idx] = e_raw;
            img.flag[idx] = match (ls > ZERO, lo > ZERO) { (true, true) => 1, (false, true) => 2, (true, false) => 3, (false, false) => 4 };
            let nclass = if n.z > 0.5 { 0 } else if n.z < -0.5 { 2 } else { 1 };
            recs.push(Rec {
                sub: si as u16, mat: mat_of(m.diffuse), nclass,
                ship: ls, ship_lm: lum(e_lm), ours: lo, ours_enc: lum(e_enc),
                dot: odir.dot(Vec3::from(st.dom_dir)), bw_s: st.bandwidth, bw_o,
                fill_frac_s: if lum(e_lm) > ZERO { e_fill / lum(e_lm) } else { 1.0 },
                vis, pos: gpos[idx],
            });
        }
        imgs.push((*sub, img));
    }
    out!("covered texels: {} evaluated ({} bake-uncovered, {} no shipped image); {} materials; eval {:.1} s",
        recs.len(), n_uncov_bake, n_noimg, mat_names.len(), t0.elapsed().as_secs_f64());
    if recs.is_empty() { out!("nothing to compare (no atlas-lit meshes?)"); return rep; }

    // ---- global best-fit scale (least squares in log space) over texels where both are > ZERO ----
    let valid: Vec<&Rec> = recs.iter().filter(|r| r.ship > ZERO && r.ours > ZERO).collect();
    let n_zero_ship = recs.iter().filter(|r| r.ship <= ZERO && r.ours > ZERO).count();
    let n_zero_ours = recs.iter().filter(|r| r.ship > ZERO && r.ours <= ZERO).count();
    let n_zero_both = recs.iter().filter(|r| r.ship <= ZERO && r.ours <= ZERO).count();
    let fit = |f: &dyn Fn(&Rec) -> (f32, f32)| -> (f64, f64, f64) {
        // returns (scale k = exp(mean(ln ship − ln ours)), log-RMS after scale, frac within ±25% after scale)
        let mut s = 0f64;
        for r in &valid { let (a, b) = f(r); s += (b as f64).ln() - (a as f64).ln(); }
        let ls = s / valid.len().max(1) as f64;
        let (mut s2, mut w) = (0f64, 0usize);
        let (lo, hi) = ((0.75f64).ln(), (1.25f64).ln());
        for r in &valid { let (a, b) = f(r); let e = (a as f64).ln() + ls - (b as f64).ln(); s2 += e * e; if e >= lo && e <= hi { w += 1; } }
        (ls.exp(), (s2 / valid.len().max(1) as f64).sqrt(), w as f64 / valid.len().max(1) as f64)
    };
    let (k_raw, rms_raw, w_raw) = fit(&|r| (r.ours, r.ship));
    let (k_raw_lm, rms_raw_lm, w_raw_lm) = fit(&|r| (r.ours, r.ship_lm.max(ZERO)));
    let (k_enc, rms_enc, w_enc) = fit(&|r| (r.ours_enc.max(ZERO), r.ship));
    // unscaled (k=1) error of the primary metric, for reference
    let (rms1, w1) = {
        let (mut s2, mut w) = (0f64, 0usize);
        let (lo, hi) = ((0.75f64).ln(), (1.25f64).ln());
        for r in &valid { let e = (r.ours as f64).ln() - (r.ship as f64).ln(); s2 += e * e; if e >= lo && e <= hi { w += 1; } }
        ((s2 / valid.len().max(1) as f64).sqrt(), w as f64 / valid.len().max(1) as f64)
    };
    let mean = |f: &dyn Fn(&Rec) -> f32| -> f64 { valid.iter().map(|r| f(r) as f64).sum::<f64>() / valid.len().max(1) as f64 };
    out!("");
    out!("== GLOBAL ({} texels with both > {ZERO}; ship≈0&ours>0: {n_zero_ship}, ours≈0&ship>0: {n_zero_ours}, both≈0: {n_zero_both}) ==", valid.len());
    out!("mean lum: shipped(rendered)={:.4} shipped(lightmap-only)={:.4} ours(raw)={:.4} ours(encoded)={:.4}",
        mean(&|r| r.ship), mean(&|r| r.ship_lm), mean(&|r| r.ours), mean(&|r| r.ours_enc));
    out!("PRIMARY ours(raw) vs shipped(rendered): best-fit k={:.4} (HMS_PATHTRACE_K); log-RMS after scale={:.3} (×/÷{:.2}); within ±25%={:.1}% | unscaled k=1: log-RMS={:.3} within ±25%={:.1}%",
        k_raw, rms_raw, rms_raw.exp(), w_raw * 100.0, rms1, w1 * 100.0);
    out!("        ours(raw) vs shipped(lightmap-only): k={:.4} log-RMS={:.3} within={:.1}%", k_raw_lm, rms_raw_lm, w_raw_lm * 100.0);
    out!("        ours(encoded round trip) vs shipped(rendered): k={:.4} log-RMS={:.3} within={:.1}%", k_enc, rms_enc, w_enc * 100.0);
    out!("dominant dir: mean dot={:.3}, frac dot>0.7={:.1}%, frac dot<0={:.1}%; bandwidth mean shipped={:.3} ours={:.3}; shipped isotropic share (0.25·fill/π ÷ E_lm) mean={:.3}; mean vis={:.3}",
        mean(&|r| r.dot), valid.iter().filter(|r| r.dot > 0.7).count() as f64 * 100.0 / valid.len() as f64,
        valid.iter().filter(|r| r.dot < 0.0).count() as f64 * 100.0 / valid.len() as f64,
        mean(&|r| r.bw_s), mean(&|r| r.bw_o), mean(&|r| r.fill_frac_s), mean(&|r| r.vis));
    let ls = k_raw.ln();
    // luminance percentiles (distribution shape, both sides)
    let pct = |vals: &mut Vec<f32>| -> [f32; 5] {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| vals[((vals.len() - 1) as f64 * f) as usize];
        [q(0.10), q(0.50), q(0.90), q(0.99), q(0.999)]
    };
    let ps = pct(&mut recs.iter().map(|r| r.ship).collect());
    let po = pct(&mut recs.iter().map(|r| r.ours).collect());
    out!("percentiles p10/p50/p90/p99/p99.9 (all covered texels): shipped(rendered) {:.4}/{:.4}/{:.4}/{:.4}/{:.4} | ours(raw) {:.5}/{:.5}/{:.5}/{:.5}/{:.5}",
        ps[0], ps[1], ps[2], ps[3], ps[4], po[0], po[1], po[2], po[3], po[4]);

    // ---- SUN/SKY REACH PROBE: does the bake's occluder set let the sun / sky reach texels the SHIPPED
    // bake says are sun-lit (vis ≥ 0.5)? Tallies the blocking material (glass is skipped like the bake). ----
    {
        let occl = scene.bake_occluder_mesh_set();
        let cand: Vec<&Rec> = recs.iter().filter(|r| r.vis >= 0.5).collect();
        let step = (cand.len() / 3000).max(1);
        let mut blockers: HashMap<String, (usize, f64)> = HashMap::new();
        let mut sky_blockers: HashMap<String, (usize, f64)> = HashMap::new();
        let (mut n_probe, mut sun_clear, mut sky_clear, mut glass_pass) = (0usize, 0usize, 0usize, 0u32);
        for r in cand.iter().step_by(step) {
            let p = Vec3::from(r.pos);
            let n = Vec3::new(0.0, 0.0, if r.nclass == 2 { -1.0 } else { 1.0 }); // approx: offset along ±Z
            n_probe += 1;
            match scene.bake_blocker(p + n * 0.05, sun_dir, &occl, &mut glass_pass) {
                None => sun_clear += 1,
                Some((t, name)) => { let e = blockers.entry(name).or_insert((0, 0.0)); e.0 += 1; e.1 += t as f64; }
            }
            match scene.bake_blocker(p + n * 0.05, Vec3::Z, &occl, &mut glass_pass) {
                None => sky_clear += 1,
                Some((t, name)) => { let e = sky_blockers.entry(name).or_insert((0, 0.0)); e.0 += 1; e.1 += t as f64; }
            }
        }
        out!("");
        out!("-- SUN/SKY REACH PROBE ({n_probe} texels the SHIPPED bake marks sun-visible (vis≥0.5), rays through the bake's occluder set; glass layers passed: {glass_pass}) --");
        out!("to-sun ray reaches sky: {} ({:.1}%); straight-up (+Z) ray reaches sky: {} ({:.1}%)", sun_clear, sun_clear as f64 * 100.0 / n_probe.max(1) as f64, sky_clear, sky_clear as f64 * 100.0 / n_probe.max(1) as f64);
        let mut bl: Vec<_> = blockers.into_iter().collect();
        bl.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (name, (n, td)) in bl.iter().take(10) { out!("  sun blocked by {:<6} ({:>5.1}%) mean dist {:>7.1}  {}", n, *n as f64 * 100.0 / n_probe.max(1) as f64, td / *n as f64, name); }
        let mut bl: Vec<_> = sky_blockers.into_iter().collect();
        bl.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (name, (n, td)) in bl.iter().take(6) { out!("  +Z blocked by  {:<6} ({:>5.1}%) mean dist {:>7.1}  {}", n, *n as f64 * 100.0 / n_probe.max(1) as f64, td / *n as f64, name); }
    }

    // ---- breakdowns: normal class, shipped-brightness quintile, sun visibility ----
    let breakdown = |name: &str, key: &dyn Fn(&Rec) -> Option<usize>, labels: &[&str], rep: &mut String| {
        let mut gs: Vec<Group> = (0..labels.len()).map(|_| Group::default()).collect();
        for r in &valid { if let Some(i) = key(r) { gs[i].push(r, (r.ours / r.ship).ln()); } }
        rep.push_str(&format!("-- by {name} (after global k={k_raw:.3}) --\n"));
        rep.push_str(&format!("{:<22} {:>9} {:>9} {:>8} {:>9} {:>10} {:>10} {:>7} {:>7} {:>7}\n", "class", "texels", "med k·r", "logRMS", "±25%", "mean_ship", "mean_ours", "dot", "bw_s", "bw_o"));
        for (i, g) in gs.iter_mut().enumerate() {
            if g.n == 0 { continue; }
            let (med, rms, w) = g.stats(ls);
            rep.push_str(&format!("{:<22} {:>9} {:>9.3} {:>8.3} {:>8.1}% {:>10.4} {:>10.4} {:>7.3} {:>7.3} {:>7.3}\n", labels[i], g.n, med * k_raw, rms, w * 100.0,
                g.sum_ship / g.n as f64, g.sum_ours / g.n as f64, g.sum_dot / g.n as f64, g.sum_bw_s / g.n as f64, g.sum_bw_o / g.n as f64));
        }
    };
    out!("");
    breakdown("normal class", &|r| Some(r.nclass as usize), &["up-facing (n.z>0.5)", "wall", "down-facing"], &mut rep);
    // brightness quintiles of the shipped value
    let mut ship_sorted: Vec<f32> = valid.iter().map(|r| r.ship).collect();
    ship_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| ship_sorted[((ship_sorted.len() - 1) as f64 * f) as usize];
    let qs = [q(0.2), q(0.4), q(0.6), q(0.8)];
    let qlabels: Vec<String> = vec![
        format!("ship<{:.3}", qs[0]), format!("{:.3}..{:.3}", qs[0], qs[1]), format!("{:.3}..{:.3}", qs[1], qs[2]), format!("{:.3}..{:.3}", qs[2], qs[3]), format!("ship>{:.3}", qs[3]),
    ];
    let ql: Vec<&str> = qlabels.iter().map(|s| s.as_str()).collect();
    out!("");
    breakdown("shipped brightness quintile", &|r| Some(qs.iter().filter(|&&t| r.ship > t).count()), &ql, &mut rep);
    out!("");
    breakdown("shipped sun visibility", &|r| Some(if r.vis < 0.05 { 0 } else if r.vis < 0.5 { 1 } else if r.vis < 0.95 { 2 } else { 3 }), &["vis≈0 (enclosed)", "vis<0.5", "vis<0.95", "vis≈1 (open)"], &mut rep);

    // ---- per submap ----
    let mut gsub: Vec<Group> = (0..subs.len()).map(|_| Group::default()).collect();
    let mut gmat: Vec<Group> = (0..mat_names.len()).map(|_| Group::default()).collect();
    for r in &valid { let lr = (r.ours / r.ship).ln(); gsub[r.sub as usize].push(r, lr); gmat[r.mat as usize].push(r, lr); }
    out!("");
    out!("-- per SUBMAP (after global k={k_raw:.3}; 'med k·r' = median ours×k/shipped) --");
    out!("{:<22} {:>9} {:>9} {:>8} {:>9} {:>10} {:>10} {:>7} {:>7}", "submap", "texels", "med k·r", "logRMS", "±25%", "mean_ship", "mean_ours", "dot", "vis");
    for (i, g) in gsub.iter_mut().enumerate() {
        if g.n == 0 { continue; }
        let (med, rms, w) = g.stats(ls);
        let vis = valid.iter().filter(|r| r.sub as usize == i).map(|r| r.vis as f64).sum::<f64>() / g.n as f64;
        out!("{:<22} {:>9} {:>9.3} {:>8.3} {:>8.1}% {:>10.4} {:>10.4} {:>7.3} {:>7.3}", format!("{:#010x}/{}", subs[i].0, subs[i].1), g.n, med * k_raw, rms, w * 100.0,
            g.sum_ship / g.n as f64, g.sum_ours / g.n as f64, g.sum_dot / g.n as f64, vis);
    }

    // ---- per material (all, sorted by count) + worst 15 by log-RMS ----
    let mut mstats: Vec<(usize, usize, f64, f64, f64)> = Vec::new(); // (mat, n, med, rms, within)
    for (i, g) in gmat.iter_mut().enumerate() { if g.n == 0 { continue; } let (med, rms, w) = g.stats(ls); mstats.push((i, g.n, med, rms, w)); }
    mstats.sort_by(|a, b| b.1.cmp(&a.1));
    out!("");
    out!("-- per MATERIAL (diffuse tag), by texel count (after global k={k_raw:.3}) --");
    out!("{:<9} {:>9} {:>8} {:>9} {:>10} {:>10} {:>7} {:>7} {:>7}  {}", "texels", "med k·r", "logRMS", "±25%", "mean_ship", "mean_ours", "dot", "vis", "fill_s", "material");
    for &(i, n, med, rms, w) in mstats.iter().take(60) {
        let g = &gmat[i];
        let vis = valid.iter().filter(|r| r.mat as usize == i).map(|r| r.vis as f64).sum::<f64>() / n as f64;
        out!("{:<9} {:>9.3} {:>8.3} {:>8.1}% {:>10.4} {:>10.4} {:>7.3} {:>7.3} {:>7.3}  {}", n, med * k_raw, rms, w * 100.0,
            g.sum_ship / n as f64, g.sum_ours / n as f64, g.sum_dot / n as f64, vis, g.sum_fill / n as f64, mat_names[i]);
    }
    let mut worst: Vec<_> = mstats.iter().filter(|m| m.1 >= 500).cloned().collect();
    worst.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    out!("");
    out!("-- WORST 15 materials by log-RMS (≥500 texels; mean values are luminance of E(n); pos = mean world pos, z-range) --");
    out!("{:<9} {:>8} {:>9} {:>10} {:>10} {:>7}  {:<28} {}", "texels", "logRMS", "med k·r", "mean_ship", "mean_ours×k", "dot", "pos (x,y,z) z-range", "material");
    for &(i, n, med, rms, _w) in worst.iter().take(15) {
        let g = &gmat[i];
        let p = [g.sum_pos[0] / n as f64, g.sum_pos[1] / n as f64, g.sum_pos[2] / n as f64];
        out!("{:<9} {:>8.3} {:>9.3} {:>10.4} {:>10.4} {:>7.3}  {:<28} {}", n, rms, med * k_raw, g.sum_ship / n as f64, g.sum_ours / n as f64 * k_raw, g.sum_dot / n as f64,
            format!("({:.0},{:.0},{:.0}) {:.0}..{:.0}", p[0], p[1], p[2], g.zmin, g.zmax), mat_names[i]);
    }

    // ---- PNGs: shipped | ours×k | log2-ratio heatmap ----
    // display: sqrt(E·expo), expo so the shipped median lands at 0.5² → common for both panels
    let med_ship = ship_sorted[ship_sorted.len() / 2].max(1e-5) as f64;
    let expo = 0.25 / med_ship;
    let to8 = |v: f64| ((v.max(0.0) * expo).sqrt().min(1.0) * 255.0) as u8;
    let _ = std::fs::create_dir_all(out_dir);
    for (sub, im) in &imgs {
        let (w, h) = (im.w as usize, im.h as usize);
        let mut rgba = vec![0u8; w * h * 4 * 3];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let f = im.flag[i];
                let mut put = |panel: usize, c: [u8; 4]| { let o = (y * w * 3 + panel * w + x) * 4; rgba[o..o + 4].copy_from_slice(&c); };
                if f == 0 { put(0, [0, 0, 0, 255]); put(1, [0, 0, 0, 255]); put(2, [0, 0, 0, 255]); continue; }
                let s = im.ship[i]; let o = im.ours[i];
                put(0, [to8(s[0] as f64), to8(s[1] as f64), to8(s[2] as f64), 255]);
                put(1, [to8(o[0] as f64 * k_raw), to8(o[1] as f64 * k_raw), to8(o[2] as f64 * k_raw), 255]);
                let hm = match f {
                    1 => {
                        // log2(ours·k / shipped) in [-2,+2]: blue = ours darker, white = match, red = ours brighter
                        let l2 = ((lum(o) as f64 * k_raw) / lum(s) as f64).log2().clamp(-2.0, 2.0) / 2.0;
                        let t = l2.abs();
                        let (r, g, b) = if l2 < 0.0 { (1.0 - t, 1.0 - t, 1.0) } else { (1.0, 1.0 - t, 1.0 - t) };
                        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
                    }
                    2 => [255, 0, 255, 255], // shipped ≈ 0, ours lit (magenta)
                    3 => [0, 255, 0, 255],   // ours ≈ 0, shipped lit (green)
                    _ => [64, 64, 64, 255],  // both ≈ 0
                };
                put(2, hm);
            }
        }
        let path = format!("{}/bake_cmp_{}_{}spp_{:08x}_{}.png", out_dir.trim_end_matches('/'), map_stem, samples, sub.0, sub.1);
        match crate::headless::write_png(&path, &rgba, (w * 3) as u32, h as u32) {
            Ok(()) => { out!("wrote {path} ({}x{} ×3 panels: shipped | ours×k | log2 ratio heatmap [blue=ours darker, red=brighter, magenta=ship≈0, green=ours≈0])", w, h); }
            Err(e) => { out!("PNG write failed {path}: {e}"); }
        }
    }
    out!("BAKE_COMPARE done in {:.1} s", t0.elapsed().as_secs_f64());
    rep
}
