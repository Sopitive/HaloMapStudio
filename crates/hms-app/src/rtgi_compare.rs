//! Agreement of the real-time probe field with the map's SHIPPED airprobes (print-only measurement).
//!
//! A shipped airprobe is the Tool's lightmapper result at a point (`Lbsp+0x12C`: a dual-vMF lobe pair).
//! The probe field is HMS's own estimate of the same quantity. This module turns both into the same
//! three numbers and reports their distribution: the up-facing diffuse lighting the ENGINE would give
//! an object there (`E(+z)/pi` through the engine decode), the dominant direction, and the bandwidth.
//!
//! Used by `headless.rs` under `HMS_RTGI_COMPARE=1` (before = legacy sky model, after = engine mode)
//! and by the gates. Nothing here reads a GPU: it takes the probe records.

use hms_render::engine_gi::*;
use hms_render::rtgi::ProbeRecord;

/// One shipped airprobe in engine units (`hms_native::AirprobePoint`).
#[derive(Clone, Copy, Debug)]
pub struct ShippedProbe {
    pub pos: [f32; 3],
    pub dom: [f32; 3],
    pub fill: [f32; 3],
    /// engine bandwidth t in [0,1]
    pub t: f32,
    pub dir: [f32; 3],
    /// baked sun visibility at the probe (`vmf[0].w`, the stored sqrt form)
    pub mask: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// shipped probes with at least four valid field probes around them
    pub n: usize,
    pub skipped: usize,
    /// fraction of probes whose HMS / shipped up-facing luminance lies in [0.9,1.1] / [0.75,1.25]
    pub within10: f64,
    pub within25: f64,
    pub median_ratio: f64,
    /// geometric mean of HMS/shipped E(+z)/pi per channel (the colour bias)
    pub bias_rgb: [f64; 3],
    /// median / 10th-percentile dot of the fitted dominant directions
    pub dir_dot_median: f64,
    pub dir_dot_p10: f64,
    /// median of (HMS t - shipped t)
    pub t_diff_median: f64,
    /// median of HMS P_lum / shipped P_lum (the radiance-integral ratio)
    pub p_ratio_median: f64,
}

fn lum(c: [f64; 3]) -> f64 { 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] }

/// The field's total SH at a world position: trilinear over the 8 surrounding probes that are valid
/// (None when fewer than four are), in the Tool's basis.
pub fn field_sh_at(probes: &[ProbeRecord], origin: [f32; 3], spacing: f32, dims: [u32; 3], p: [f32; 3]) -> Option<Sh3> {
    field_sh_slot_at(probes, origin, spacing, dims, p, false)
}

/// Same, optionally of the SKY-DIRECT slot (the deterministic sky-through-the-scene term, no bounce noise).
pub fn field_sh_slot_at(probes: &[ProbeRecord], origin: [f32; 3], spacing: f32, dims: [u32; 3], p: [f32; 3], sky_slot: bool) -> Option<Sh3> {
    let g = [(p[0] - origin[0]) / spacing, (p[1] - origin[1]) / spacing, (p[2] - origin[2]) / spacing];
    // a point outside the grid (beyond half a cell) has no field around it: say so rather than extrapolate
    if (0..3).any(|a| g[a] < -0.5 || g[a] > dims[a] as f32 - 0.5) { return None; }
    let base: [i64; 3] = [0, 1, 2].map(|a| (g[a].floor() as i64).clamp(0, (dims[a] as i64 - 2).max(0)));
    let f: [f64; 3] = [0, 1, 2].map(|a| (g[a] as f64 - base[a] as f64).clamp(0.0, 1.0));
    let mut acc: Sh3 = [[0.0; 9]; 3];
    let (mut wsum, mut nvalid) = (0.0f64, 0);
    for k in 0..8u32 {
        let off = [(k & 1) as i64, ((k >> 1) & 1) as i64, ((k >> 2) & 1) as i64];
        let c = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
        if (0..3).any(|a| c[a] < 0 || c[a] >= dims[a] as i64) { continue; }
        let idx = (c[0] + c[1] * dims[0] as i64 + c[2] * dims[0] as i64 * dims[1] as i64) as usize;
        let Some(pr) = probes.get(idx) else { continue };
        if !pr.valid { continue; }
        nvalid += 1;
        let w = (0..3).map(|a| if off[a] == 1 { f[a] } else { 1.0 - f[a] }).product::<f64>();
        let sh = from_graphics_sh(if sky_slot { &pr.sky } else { &pr.total });
        for c in 0..3 { for j in 0..9 { acc[c][j] += w * sh[c][j]; } }
        wsum += w;
    }
    if nvalid < 4 || wsum <= 1e-9 { return None; }
    for c in 0..3 { for j in 0..9 { acc[c][j] /= wsum; } }
    Some(acc)
}

/// Compare the field against the shipped probes.
pub fn compare(shipped: &[ShippedProbe], probes: &[ProbeRecord], origin: [f32; 3], spacing: f32, dims: [u32; 3]) -> Stats {
    let mut st = Stats::default();
    let (mut ratios, mut dots, mut tdiff, mut pratio) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut logsum = [0.0f64; 3];
    for s in shipped {
        let Some(sh) = field_sh_at(probes, origin, spacing, dims, s.pos) else { st.skipped += 1; continue };
        let theirs = DualVmf {
            dir: [s.dir[0] as f64, s.dir[1] as f64, s.dir[2] as f64],
            kappa: kappa_of_t(s.t as f64),
            t: s.t as f64,
            dom: [s.dom[0] as f64, s.dom[1] as f64, s.dom[2] as f64],
            fill: [s.fill[0] as f64, s.fill[1] as f64, s.fill[2] as f64],
            err: 0.0,
        };
        let e_ship = diffuse_from_dual_vmf([0.0, 0.0, 1.0], &theirs);
        let e_hms = sh_irradiance(&sh, [0.0, 0.0, 1.0]).map(|v| v / std::f64::consts::PI);
        if lum(e_ship) <= 1e-6 { st.skipped += 1; continue; }
        ratios.push(lum(e_hms) / lum(e_ship));
        for c in 0..3 { logsum[c] += (e_hms[c].max(1e-9) / e_ship[c].max(1e-9)).ln(); }
        let mine = fit_sh_to_dualvmf(&sh);
        dots.push(mine.dir[0] * theirs.dir[0] + mine.dir[1] * theirs.dir[1] + mine.dir[2] * theirs.dir[2]);
        tdiff.push(mine.t - theirs.t);
        pratio.push(lum([mine.dom[0] + mine.fill[0], mine.dom[1] + mine.fill[1], mine.dom[2] + mine.fill[2]]) / lum([theirs.dom[0] + theirs.fill[0], theirs.dom[1] + theirs.fill[1], theirs.dom[2] + theirs.fill[2]]).max(1e-9));
        st.n += 1;
    }
    if st.n == 0 { return st; }
    let frac = |lo: f64, hi: f64| ratios.iter().filter(|r| **r >= lo && **r <= hi).count() as f64 / st.n as f64;
    st.within10 = frac(0.9, 1.1);
    st.within25 = frac(0.75, 1.25);
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); v[v.len() / 2] };
    st.median_ratio = med(&mut ratios);
    for c in 0..3 { st.bias_rgb[c] = (logsum[c] / st.n as f64).exp(); }
    dots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    st.dir_dot_p10 = dots[dots.len() / 10];
    st.dir_dot_median = dots[dots.len() / 2];
    st.t_diff_median = med(&mut tdiff);
    st.p_ratio_median = med(&mut pratio);
    st
}

pub fn report(label: &str, s: &Stats) -> String {
    format!(
        "RTGI_COMPARE {label}: {} shipped airprobes compared ({} skipped) | E(+z)/pi HMS/shipped: median {:.3}, within +-10% {:.1}%, within +-25% {:.1}% | colour bias (R,G,B) ({:.3},{:.3},{:.3}) | dominant direction dot: median {:.3}, p10 {:.3} | t diff median {:+.3} | P_lum ratio median {:.3}",
        s.n, s.skipped, s.median_ratio, 100.0 * s.within10, 100.0 * s.within25, s.bias_rgb[0], s.bias_rgb[1], s.bias_rgb[2], s.dir_dot_median, s.dir_dot_p10, s.t_diff_median, s.p_ratio_median
    )
}

/// Agreement of two sun-visibility sets (stored sqrt form, 0 = shadowed, 1 = lit) sampled at the same points.
#[derive(Clone, Debug, Default)]
pub struct MaskStats {
    pub n: usize,
    /// fraction of points where the two differ by less than 0.25
    pub agree: f64,
    /// shadowed = visibility < 0.5: intersection over union of the two shadowed sets
    pub shadow_iou: f64,
    pub shadow_a: usize,
    pub shadow_b: usize,
    /// mean visibility of each set
    pub mean_a: f64,
    pub mean_b: f64,
}

/// `a` = reference (the shipped bake), `b` = the real-time field.
pub fn mask_compare(a: &[f32], b: &[f32]) -> MaskStats {
    let n = a.len().min(b.len());
    let mut st = MaskStats { n, ..Default::default() };
    if n == 0 { return st; }
    let (mut agree, mut inter, mut uni) = (0usize, 0usize, 0usize);
    for i in 0..n {
        if (a[i] - b[i]).abs() < 0.25 { agree += 1; }
        let (sa, sb) = (a[i] < 0.5, b[i] < 0.5);
        if sa { st.shadow_a += 1; }
        if sb { st.shadow_b += 1; }
        if sa && sb { inter += 1; }
        if sa || sb { uni += 1; }
        st.mean_a += a[i] as f64;
        st.mean_b += b[i] as f64;
    }
    st.agree = agree as f64 / n as f64;
    st.shadow_iou = if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
    st.mean_a /= n as f64;
    st.mean_b /= n as f64;
    st
}

pub fn mask_report(label: &str, s: &MaskStats) -> String {
    format!("RTGI_COMPARE {label}: {} points | sun visibility agree (|d|<0.25) {:.1}% | shadow IoU {:.3} (shadowed: shipped {}, real-time {}) | mean vis shipped {:.3}, real-time {:.3}",
        s.n, 100.0 * s.agree, s.shadow_iou, s.shadow_a, s.shadow_b, s.mean_a, s.mean_b)
}

/// Two renders of the same view (RGBA8, display values): the baked-lightmap frame and the real-time-lighting
/// frame. Luminance is linearised (pow 2.2). The SHADOW MASK of a frame = pixels darker than 0.55 x that frame's
/// 80th-percentile luminance, so the mask is independent of an overall brightness difference and answers only
/// "where is the shadow".
#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub n: usize,
    pub mean_base: f64,
    pub mean_rt: f64,
    /// per-pixel log2(rt/base) over pixels lit in either frame: median, 10th and 90th percentile
    pub log2_median: f64,
    pub log2_p10: f64,
    pub log2_p90: f64,
    pub within25: f64,
    pub shadow_frac_base: f64,
    pub shadow_frac_rt: f64,
    pub shadow_iou: f64,
}

pub fn frame_stats(base: &[u8], rt: &[u8]) -> FrameStats {
    let n = (base.len().min(rt.len())) / 4;
    let lin = |p: &[u8]| -> f64 { (0.2126 * (p[0] as f64 / 255.0).powf(2.2)) + 0.7152 * (p[1] as f64 / 255.0).powf(2.2) + 0.0722 * (p[2] as f64 / 255.0).powf(2.2) };
    let lb: Vec<f64> = (0..n).map(|i| lin(&base[i * 4..])).collect();
    let lr: Vec<f64> = (0..n).map(|i| lin(&rt[i * 4..])).collect();
    let p80 = |v: &[f64]| { let mut s = v.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); s[(s.len() * 4 / 5).min(s.len().saturating_sub(1))] };
    let mut st = FrameStats { n, ..Default::default() };
    if n == 0 { return st; }
    let (tb, tr) = (0.55 * p80(&lb), 0.55 * p80(&lr));
    let (mut inter, mut uni, mut sb, mut sr) = (0usize, 0usize, 0usize, 0usize);
    let mut logs: Vec<f64> = Vec::new();
    for i in 0..n {
        st.mean_base += lb[i];
        st.mean_rt += lr[i];
        let (a, b) = (lb[i] < tb, lr[i] < tr);
        if a { sb += 1; }
        if b { sr += 1; }
        if a && b { inter += 1; }
        if a || b { uni += 1; }
        if lb[i] > 1e-4 || lr[i] > 1e-4 { logs.push(((lr[i] + 1e-4) / (lb[i] + 1e-4)).log2()); }
    }
    st.mean_base /= n as f64;
    st.mean_rt /= n as f64;
    st.shadow_frac_base = sb as f64 / n as f64;
    st.shadow_frac_rt = sr as f64 / n as f64;
    st.shadow_iou = if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
    if !logs.is_empty() {
        logs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        st.log2_median = logs[logs.len() / 2];
        st.log2_p10 = logs[logs.len() / 10];
        st.log2_p90 = logs[logs.len() * 9 / 10];
        st.within25 = logs.iter().filter(|l| l.abs() < 0.3219).count() as f64 / logs.len() as f64; // +-25 %
    }
    st
}

pub fn frame_report(label: &str, s: &FrameStats) -> String {
    format!("RTGI_COMPARE {label}: {} px | mean linear lum baked {:.4} / real-time {:.4} | per-pixel log2(rt/baked) median {:+.2}, p10 {:+.2}, p90 {:+.2}, within +-25% {:.1}% | shadow mask (dark < 0.55 x p80) baked {:.1}% of px, real-time {:.1}%, IoU {:.3}",
        s.n, s.mean_base, s.mean_rt, s.log2_median, s.log2_p10, s.log2_p90, 100.0 * s.within25, 100.0 * s.shadow_frac_base, 100.0 * s.shadow_frac_rt, s.shadow_iou)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field whose every probe holds the SH of the same lobe, and the shipped probes that ARE that lobe.
    fn field_of(f: &DualVmf, scale: f64, dims: [u32; 3]) -> Vec<ProbeRecord> {
        // build the SH of the lobe with the exact model, store it in the graphics basis
        let z = |k: f64| { let c = 1.0 / k.tanh(); [1.0, c - 1.0 / k, 1.0 - 3.0 * c / k + 3.0 / (k * k)] };
        let zb = z(f.kappa);
        let band = [1.0, zb[1], zb[1], zb[1], zb[2], zb[2], zb[2], zb[2], zb[2]];
        let y = basis(f.dir);
        let mut g = [[0f32; 3]; 9];
        for c in 0..3 {
            let p = f.dom[c] + f.fill[c];
            let share = f.dom[c] / p;
            for j in 0..9 {
                let tool = if j == 0 { p * 0.282094806 } else { p * share * band[j] * y[j] };
                let sign = if matches!(j, 1 | 3 | 5 | 7) { -1.0 } else { 1.0 };
                g[j][c] = (scale * sign * tool) as f32;
            }
        }
        (0..dims[0] * dims[1] * dims[2]).map(|_| ProbeRecord { total: g, gather: [[0.0; 3]; 9], sky: [[0.0; 3]; 9], offset: [0.0; 3], valid: true }).collect()
    }

    fn shipped_from(f: &DualVmf, n: usize) -> Vec<ShippedProbe> {
        (0..n).map(|i| ShippedProbe {
            pos: [1.0 + 0.5 * i as f32, 2.0, 2.5],
            dom: [f.dom[0] as f32, f.dom[1] as f32, f.dom[2] as f32], fill: [f.fill[0] as f32, f.fill[1] as f32, f.fill[2] as f32],
            t: f.t as f32, dir: [f.dir[0] as f32, f.dir[1] as f32, f.dir[2] as f32], mask: 1.0,
        }).collect()
    }

    /// The metric is an identity when the field IS the shipped lobe, and it bites: a field 1.5x too bright
    /// leaves ~0 % within +-25 %, reports the ratio and a colour-neutral bias; a field with the lobe turned
    /// away fails the direction check.
    #[test]
    fn the_comparison_is_an_identity_and_it_bites() {
        let lobe = DualVmf { dir: [0.0, 0.2, 0.98f64.min(1.0)], kappa: 2.2, t: t_of_kappa(2.2), dom: [3.7, 5.0, 6.2], fill: [0.0; 3], err: 0.0 };
        let n = { let l = (lobe.dir[0] * lobe.dir[0] + lobe.dir[1] * lobe.dir[1] + lobe.dir[2] * lobe.dir[2]).sqrt(); [lobe.dir[0] / l, lobe.dir[1] / l, lobe.dir[2] / l] };
        let lobe = DualVmf { dir: n, ..lobe };
        let dims = [6, 6, 6];
        let ship = shipped_from(&lobe, 8);
        let same = compare(&ship, &field_of(&lobe, 1.0, dims), [0.0; 3], 1.0, dims);
        assert_eq!(same.n, 8, "all 8 compared");
        assert!((same.median_ratio - 1.0).abs() < 0.04, "ratio {}", same.median_ratio);
        assert!(same.within10 > 0.99 && same.dir_dot_median > 0.999 && same.t_diff_median.abs() < 0.03, "{same:?}");
        let bright = compare(&ship, &field_of(&lobe, 1.5, dims), [0.0; 3], 1.0, dims);
        assert!(bright.within25 < 0.01 && (bright.median_ratio - 1.5).abs() < 0.08, "{bright:?}");
        assert!((bright.bias_rgb[0] - bright.bias_rgb[2]).abs() < 0.02, "a grey scale is colour neutral: {:?}", bright.bias_rgb);
        let turned = DualVmf { dir: [1.0, 0.0, 0.0], ..lobe };
        let off = compare(&ship, &field_of(&turned, 1.0, dims), [0.0; 3], 1.0, dims);
        assert!(off.dir_dot_median < 0.5, "a turned lobe must fail the direction check: {off:?}");
        // probes outside the field (fewer than four valid neighbours) are skipped, not guessed
        let far = vec![ShippedProbe { pos: [900.0, 900.0, 900.0], ..ship[0] }];
        let none = compare(&far, &field_of(&lobe, 1.0, dims), [0.0; 3], 1.0, dims);
        assert_eq!((none.n, none.skipped), (0, 1));
    }

    /// The frame metric answers "where is the shadow", not "how bright": the same image brightened 1.6x keeps
    /// IoU 1 and moves the median ratio by log2(1.6^2.2-ish); a frame whose shadow moved drops the IoU. The mask
    /// metric agrees with itself and sees a flipped half. (Saboteur: drop the per-frame p80 normalisation.)
    #[test]
    fn the_frame_and_mask_metrics_measure_shadow_position_not_brightness() {
        let (w, h) = (64usize, 64usize);
        let mk = |shadow_x0: usize, gain: f32| -> Vec<u8> {
            let mut v = Vec::new();
            for _y in 0..h { for x in 0..w {
                let lit = !(shadow_x0..shadow_x0 + 20).contains(&x);
                let g = (if lit { 200.0 } else { 60.0 } * gain).min(255.0) as u8;
                v.extend_from_slice(&[g, g, g, 255]);
            } }
            v
        };
        let a = mk(10, 1.0);
        let same = frame_stats(&a, &mk(10, 1.0));
        assert!((same.shadow_iou - 1.0).abs() < 1e-9 && same.log2_median.abs() < 1e-6, "{same:?}");
        let bright = frame_stats(&a, &mk(10, 1.2));
        assert!((bright.shadow_iou - 1.0).abs() < 1e-9 && bright.log2_median > 0.3, "brightness must not move the shadow mask: {bright:?}");
        let moved = frame_stats(&a, &mk(40, 1.0));
        assert!(moved.shadow_iou < 0.1, "a moved shadow must drop the IoU: {moved:?}");
        let m = mask_compare(&[1.0, 1.0, 0.0, 0.0, 0.5], &[1.0, 0.0, 0.0, 1.0, 0.45]);
        assert_eq!((m.n, m.shadow_a, m.shadow_b), (5, 2, 3));
        assert!((m.shadow_iou - 1.0 / 4.0).abs() < 1e-9 && (m.agree - 3.0 / 5.0).abs() < 1e-9, "{m:?}");
    }
}
