//! Data-backed gates of `engine_gi` against the SHIPPED Halo Reach EK content (read-only, loose tags
//! under `HREK/tags`). `#[ignore]`d: they need the EK on this machine.
//!
//!   cargo test --release -p hms-render --test engine_gi_data -- --ignored --nocapture
//!
//! RUNG A of `research/new-maps/07-tool-lightmapper-re.md` s9: the Tool's direct-sky lightmap of an
//! open floor (sky list, hemisphere clip, sun excluded, no bounce) fitted by the Tool's encoder must
//! land on the SHIPPED `box` atlas' brightest texels. The prediction is made with the production
//! functions `sky_sh_from_list` + `fit_sh_to_dualvmf`; nothing of the shipped atlas is fed into them.

use hms_render::engine_gi::*;
use std::path::PathBuf;

fn hrek() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HREK_DIR") {
        return Some(PathBuf::from(p));
    }
    for d in std::fs::read_dir("/run/media").ok()?.flatten() {
        for e in std::fs::read_dir(d.path()).ok()?.flatten() {
            let p = e.path().join("Program Files (x86)/Steam/steamapps/common/HREK");
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    None
}

fn f32_at(d: &[u8], o: usize) -> f32 { f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) }
fn u32_at(d: &[u8], o: usize) -> u32 { u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) }

/// The sky light run of a sky render_model: the longest run of 28-byte records (unit direction, rgb >= 0,
/// 0 < solid angle < 7) — the same scan `sky_params.py` uses for the older `box_sky` layout. The
/// Tool's own convention is the last record = the sun. The run is NOT 4-byte aligned in the older tags (box_sky), so the scan steps by one byte.
fn read_sky_lights(d: &[u8]) -> Vec<SkyRecord> {
    let valid = |o: usize| -> Option<SkyRecord> {
        if o + 28 > d.len() { return None; }
        let v: Vec<f32> = (0..7).map(|k| f32_at(d, o + 4 * k)).collect();
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if v.iter().any(|x| !x.is_finite()) || (n - 1.0).abs() > 2e-3 || v[3].min(v[4]).min(v[5]) < 0.0 || !(v[6] > 0.0 && v[6] < 7.0) { return None; }
        Some(SkyRecord { dir: [v[0], v[1], v[2]], rgb: [v[3], v[4], v[5]], omega: v[6] })
    };
    let mut best: Vec<SkyRecord> = Vec::new();
    let mut o = 0usize;
    while o + 28 <= d.len() {
        if valid(o).is_some() {
            let mut run = Vec::new();
            let mut p = o;
            while let Some(r) = valid(p) { run.push(r); p += 28; }
            if run.len() > best.len() { best = run; }
            o = p;
        } else {
            o += 1;
        }
    }
    best
}

/// BC4 block -> 16 values in 0..1.
fn bc4(b: &[u8]) -> [f32; 16] {
    let (a0, a1) = (b[0] as f32, b[1] as f32);
    let mut pal = [0f32; 8];
    pal[0] = a0;
    pal[1] = a1;
    if a0 > a1 {
        for k in 1..7 { pal[1 + k] = ((7 - k) as f32 * a0 + k as f32 * a1) / 7.0; }
    } else {
        for k in 1..5 { pal[1 + k] = ((5 - k) as f32 * a0 + k as f32 * a1) / 5.0; }
        pal[6] = 0.0;
        pal[7] = 255.0;
    }
    let mut bits = 0u64;
    for i in 0..6 { bits |= (b[2 + i] as u64) << (8 * i); }
    let mut out = [0f32; 16];
    for i in 0..16 { out[i] = pal[((bits >> (3 * i)) & 7) as usize] / 255.0; }
    out
}

/// BC1 colour block (always the 4-colour mode, as the DXT5 colour half is) -> 16 RGB.
fn bc1(b: &[u8]) -> [[f32; 3]; 16] {
    let c0 = b[0] as u32 | (b[1] as u32) << 8;
    let c1 = b[2] as u32 | (b[3] as u32) << 8;
    let rgb = |c: u32| [((c >> 11) & 31) as f32 * 255.0 / 31.0, ((c >> 5) & 63) as f32 * 255.0 / 63.0, (c & 31) as f32 * 255.0 / 31.0];
    let (p0, p1) = (rgb(c0), rgb(c1));
    let mut pal = [p0, p1, [0.0; 3], [0.0; 3]];
    for k in 0..3 { pal[2][k] = (2.0 * p0[k] + p1[k]) / 3.0; pal[3][k] = (p0[k] + 2.0 * p1[k]) / 3.0; }
    let bits = b[4] as u32 | (b[5] as u32) << 8 | (b[6] as u32) << 16 | (b[7] as u32) << 24;
    let mut out = [[0.0; 3]; 16];
    for i in 0..16 { let p = pal[((bits >> (2 * i)) & 3) as usize]; out[i] = [p[0] / 255.0, p[1] / 255.0, p[2] / 255.0]; }
    out
}

/// The pixel chunk of a loose bitmap tag: the `adgt` chunk whose size is exactly `n` bytes.
fn pixel_chunk(d: &[u8], n: usize) -> Option<&[u8]> {
    let mut i = 0;
    while i + 12 <= d.len() {
        if &d[i..i + 4] == b"adgt" && u32_at(d, i + 8) as usize == n && i + 12 + n <= d.len() { return Some(&d[i + 12..i + 12 + n]); }
        i += 1;
    }
    None
}

struct Texel {
    dom: [f32; 3],
    fill: [f32; 3],
    dir: [f32; 3],
    t: f32,
}

/// Decode the shipped dual-vMF atlas (engine `DecompressVMF`, 04 s1.5): DM = DXN (R intensity code, G sun
/// visibility), SDM = 3 DXT5 slices (RGB colour parts, A direction/bandwidth components).
fn decode_atlas(dm: &[u8], sdm: &[u8], k: f32, w: usize) -> Vec<Option<Texel>> {
    let nb = w / 4;
    let mut code = vec![0f32; w * w];
        for by in 0..nb { for bx in 0..nb {
        let blk = &dm[(by * nb + bx) * 16..][..16];
        let r = bc4(&blk[..8]); // (the DM green channel is the sun visibility: not needed for the radiometry rungs)
        for i in 0..16 { let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4); code[y * w + x] = r[i]; }
    } }
    let mut sl: Vec<(Vec<[f32; 3]>, Vec<f32>)> = Vec::new();
    for s in 0..3 {
        let raw = &sdm[s * w * w..(s + 1) * w * w];
        let mut rgb = vec![[0f32; 3]; w * w];
        let mut a = vec![0f32; w * w];
        for by in 0..nb { for bx in 0..nb {
            let blk = &raw[(by * nb + bx) * 16..][..16];
            let (al, c) = (bc4(&blk[..8]), bc1(&blk[8..]));
            for i in 0..16 { let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4); rgb[y * w + x] = c[i]; a[y * w + x] = al[i]; }
        } }
        sl.push((rgb, a));
    }
    (0..w * w).map(|i| {
        let c = code[i];
        if c >= 0.97 { return None; } // background / unlit texels carry code 250/255
        let v = [sl[0].1[i] * 2.0 - 1.0, sl[1].1[i] * 2.0 - 1.0, sl[2].1[i] * 2.0 - 1.0];
        let t = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let dir = [v[0] / t.max(1e-6), v[1] / t.max(1e-6), v[2] / t.max(1e-6)];
        let f = (-9.0 * c).exp2() * k;
        let mut dom = [0f32; 3];
        let mut fill = [0f32; 3];
        for ch in 0..3 {
            dom[ch] = (sl[0].0[i][ch] + 2.0 * sl[1].0[i][ch] - 1.0) * f;
            fill[ch] = sl[2].0[i][ch] * f;
        }
        Some(Texel { dom, fill, dir, t: t.min(1.0) })
    }).collect()
}

fn lbsp_k(d: &[u8]) -> f32 {
    let i = d.windows(4).position(|w| w == b"tadb").expect("tadb chunk");
    f32_at(d, i + 12 + 12 + 8 + 0x18)
}

fn median(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }

/// What the gate measures, so the negative controls can reuse it.
struct Rung {
    n: usize,
    pred: DualVmf,
    pred_p: [f64; 3],
    ship_p: [f64; 3],
    ship_t: f64,
    ship_share: f64,
    ship_dir: [f64; 3],
}

fn rung_a(sky: &[SkyRecord], normal: Option<[f32; 3]>, texels: &[Option<Texel>]) -> Rung {
    let sh = sky_sh_from_list(sky, normal, &|_| [1.0; 3]);
    let pred = fit_sh_to_dualvmf(&sh);
    let pred_p = radiance_integral(&sh);
    let lum = |t: &Texel| -> f64 { (0..3).map(|c| (t.dom[c] + t.fill[c]) as f64 * [0.2126, 0.7152, 0.0722][c]).sum() };
    // the brightest population: open-floor texels (07 s5.4 used P in 1.89..2.31)
    let sel: Vec<&Texel> = texels.iter().flatten().filter(|t| lum(t) >= 1.89).collect();
    let n = sel.len();
    assert!(n > 5000, "only {n} bright texels: the atlas did not decode (liveness)");
    let mut pc = [vec![], vec![], vec![]];
    let (mut ts, mut shs, mut dx, mut dy, mut dz) = (vec![], vec![], vec![], vec![], vec![]);
    for t in &sel {
        for c in 0..3 { pc[c].push((t.dom[c] + t.fill[c]) as f64); }
        ts.push(t.t as f64);
        let p = lum(t);
        shs.push((0..3).map(|c| t.dom[c] as f64 * [0.2126, 0.7152, 0.0722][c]).sum::<f64>() / p);
        dx.push(t.dir[0] as f64); dy.push(t.dir[1] as f64); dz.push(t.dir[2] as f64);
    }
    let ship_p = [median(&mut pc[0]), median(&mut pc[1]), median(&mut pc[2])];
    Rung { n, pred, pred_p, ship_p, ship_t: median(&mut ts), ship_share: median(&mut shs), ship_dir: [median(&mut dx), median(&mut dy), median(&mut dz)] }
}

fn pass_rung_a(r: &Rung) -> Result<(), String> {
    let dirdot = r.pred.dir.iter().zip(r.ship_dir).map(|(a, b)| a * b).sum::<f64>();
    let pred_share = (0..3).map(|c| r.pred.dom[c] * [0.2126, 0.7152, 0.0722][c]).sum::<f64>() / (0..3).map(|c| r.pred_p[c] * [0.2126, 0.7152, 0.0722][c]).sum::<f64>();
    let mut why = Vec::new();
    if dirdot < 0.995 { why.push(format!("direction dot {dirdot:.4} < 0.995")); }
    if (r.pred.t - r.ship_t).abs() > 0.015 { why.push(format!("bandwidth t {:.3} vs shipped {:.3}", r.pred.t, r.ship_t)); }
    if (pred_share - r.ship_share).abs() > 0.03 { why.push(format!("dominant share {pred_share:.3} vs shipped {:.3}", r.ship_share)); }
    for c in 0..3 {
        let e = (r.pred_p[c] - r.ship_p[c]) / r.ship_p[c];
        // 07 measured a one-sided -3..-5 % (shipped lower) it could not explain; 7 % covers it
        if e.abs() > 0.07 { why.push(format!("P[{c}] {:.3} vs shipped {:.3} ({:+.1} %)", r.pred_p[c], r.ship_p[c], e * 100.0)); }
    }
    if why.is_empty() { Ok(()) } else { Err(why.join("; ")) }
}

#[test]
#[ignore = "needs the Halo Reach EK on this machine"]
fn rung_a_box_open_floor_matches_the_shipped_atlas() {
    let Some(h) = hrek() else { eprintln!("engine_gi_data SKIPPED: no HREK"); return; };
    let box_dir = h.join("tags/levels/test/box");
    let sky_tag = std::fs::read(box_dir.join("box_sky/box_sky.render_model")).expect("box_sky");
    let lights = read_sky_lights(&sky_tag);
    assert_eq!(lights.len(), 51, "box_sky has 50 sky records + the sun (07 s5.4), found {}", lights.len());
    let sl = SkyLights::from_records(lights.clone(), true, 1.0, 1.0);
    assert!(sl.sun_is_brightest(), "the last record must be the brightest (the sun)");
    let lbsp = std::fs::read(box_dir.join("box_000.scenario_lightmap_bsp_data")).expect("lbsp");
    let k = lbsp_k(&lbsp);
    let dm = std::fs::read(box_dir.join("box_lightmap_box_000_16f_lp_array_dualvmf_pixel_direction_dxtn.bitmap")).expect("dm");
    let sdm = std::fs::read(box_dir.join("box_lightmap_box_000_16f_lp_array_dualvmf_pixel_intensity_dxt5.bitmap")).expect("sdm");
    let w = 512;
    let dmc = pixel_chunk(&dm, w * w).expect("DM pixels");
    let sdc = pixel_chunk(&sdm, 3 * w * w).expect("SDM pixels");
    let tex = decode_atlas(dmc, sdc, k, w);
    let lit = tex.iter().flatten().count();
    eprintln!("box atlas: K {k:.4}, {lit} lit texels of {}", w * w);
    assert!((k - 3.6493).abs() < 1e-3, "K = {k}");

    // THE PREDICTION: sky records only (sun excluded), hemisphere-clipped for an up-facing floor texel.
    let r = rung_a(&sl.sky, Some([0.0, 0.0, 1.0]), &tex);
    eprintln!(
        "box rung A over {} brightest texels\n  predicted: dir ({:.3},{:.3},{:.3}) t {:.3} kappa {:.2}  P ({:.3},{:.3},{:.3})\n  shipped:   dir ({:.3},{:.3},{:.3}) t {:.3}            P ({:.3},{:.3},{:.3}) share {:.3}",
        r.n, r.pred.dir[0], r.pred.dir[1], r.pred.dir[2], r.pred.t, r.pred.kappa, r.pred_p[0], r.pred_p[1], r.pred_p[2],
        r.ship_dir[0], r.ship_dir[1], r.ship_dir[2], r.ship_t, r.ship_p[0], r.ship_p[1], r.ship_p[2], r.ship_share
    );
    if let Err(e) = pass_rung_a(&r) { panic!("rung A failed: {e}"); }

    // NEGATIVE CONTROLS: the gate must bite. Each of the four things 07 found, undone, must fail it.
    // (1) the sun left in the SH (a lightmapper that is not `analytical`)
    let with_sun = rung_a(&lights, Some([0.0, 0.0, 1.0]), &tex);
    assert!(pass_rung_a(&with_sun).is_err(), "gate does not notice the sun in the SH: {with_sun:?}", with_sun = (with_sun.pred_p, with_sun.ship_p));
    // (2) no hemisphere clip (every record seen): the dome is above the horizon, so for this sky the clip only
    //     matters with a tilted receiver; a wall facing +x loses half the dome and must NOT match the floor
    let wall = rung_a(&sl.sky, Some([1.0, 0.0, 0.0]), &tex);
    assert!(pass_rung_a(&wall).is_err(), "gate does not tell a wall from a floor");
    // (3) a sky scaled by pi (the 'divided by pi' mistake the old HMS sky had in reverse)
    let scaled: Vec<SkyRecord> = sl.sky.iter().map(|s| SkyRecord { rgb: [s.rgb[0] * 3.14159, s.rgb[1] * 3.14159, s.rgb[2] * 3.14159], ..*s }).collect();
    let big = rung_a(&scaled, Some([0.0, 0.0, 1.0]), &tex);
    assert!(pass_rung_a(&big).is_err(), "gate does not notice a pi-scaled sky");
}

#[test]
#[ignore = "needs the Halo Reach EK on this machine"]
fn forge_world_sky_list_and_scalers_read_as_documented() {
    let Some(h) = hrek() else { eprintln!("engine_gi_data SKIPPED: no HREK"); return; };
    let d = std::fs::read(h.join("tags/levels/multi/forge_halo/sky_halo/sky_halo.render_model")).expect("sky_halo");
    let lights = read_sky_lights(&d);
    assert_eq!(lights.len(), 103, "sky_halo: 102 sky records + the sun");
    let sl = SkyLights::from_records(lights, true, 0.3, 1.0);
    assert!(sl.sun_is_brightest());
    let sun_e = sl.sun_irradiance();
    assert!((sun_e[0] - 3.0).abs() < 0.01 && (sun_e[1] - 2.824).abs() < 0.01 && (sun_e[2] - 2.529).abs() < 0.01, "sun irradiance {sun_e:?}");
    let sh = sky_sh_from_list(&sl.sky, Some([0.0, 0.0, 1.0]), &|_| [1.0; 3]);
    let p = radiance_integral(&sh);
    eprintln!("sky_halo P_up = ({:.3},{:.3},{:.3})", p[0], p[1], p[2]);
    assert!((p[0] - 2.191).abs() < 0.01 && (p[1] - 3.014).abs() < 0.01 && (p[2] - 3.824).abs() < 0.01, "P_up {p:?}");
    // the on-disk scalers (HREK root +0x238 / +0x23C = cache mode +0x244 / +0x248)
    let i = d.windows(4).position(|w| w == b"tadb").expect("tadb");
    let root = i + 32;
    let (sun1, sky1) = (f32_at(&d, root + 0x238), f32_at(&d, root + 0x23C));
    assert!((sun1 - 0.30).abs() < 1e-6 && (sky1 - 1.0).abs() < 1e-6, "sun/sky 1st bounce scalers {sun1}/{sky1}");
}
