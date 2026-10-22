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
        let sh = from_graphics_sh(&pr.total);
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
            t: f.t as f32, dir: [f.dir[0] as f32, f.dir[1] as f32, f.dir[2] as f32],
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
}
