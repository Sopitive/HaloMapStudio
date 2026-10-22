//! The engine's lighting radiometry, as pure functions (no GPU): what the Tool's lightmapper
//! (`faux`) stores and what the game reconstructs. `rtgi.rs` (the real-time probe GI) is built on
//! these conventions, and the later HMS lightmap baker will write its results through the same
//! functions, so nothing here may drift from the engine.
//!
//! Every constant below was read out of `reachek_tool.exe` (the HREK `tool.exe`) or measured on
//! shipped data in `research/new-maps/07-tool-lightmapper-re.md`; the `sub_` addresses are its VAs.
//! PROVEN = decompiled / measured, INFERRED = follows from proven facts.
//!
//! * SH basis and order (`sub_140610B70`): `[Y00, -.4886 y, .4886 z, -.4886 x, 1.0925 xy, -1.0925 yz,
//!   .9462 z^2 - .3154, -1.0925 xz, .5463 (x^2 - y^2)]` — Condon-Shortley signs on the odd terms.
//!   `rtgi.rs` stores the plain graphics basis (+ signs); `from_graphics_sh` converts. PROVEN.
//! * SKY (`sub_140190040`/`sub_1401904E0`): the sky is the sky render_model's "sky lights" list, each
//!   record `rgb * solid_angle` as a point light toward `dir`; a record contributes to a receiver
//!   with normal `N` only if `N.L > 0` (a probe, `|N| == 0`, accepts every direction); NO cosine in
//!   the SH; the last record is the sun. PROVEN.
//! * SUN: with `analytical = true` (the shipped farm script) the sun is NOT in the SH. It only gives
//!   the per-texel visibility `DM.g` and the bounce source. PROVEN (`sub_1401974A0`).
//! * SH accumulate: `SH_c[lm] += rgb_c * (Omega/N) * V * Y_lm(l)` — SH of incident RADIANCE, so
//!   `2 sqrt(pi) * SH_c[0]` = the radiance integral `P` over the visible directions. PROVEN.
//! * BOUNCE SOURCE (`sub_1401974A0`, `sub_140196D50`): `D = 1/pi * sum clamp(m * N.L * w, 0, 1) *
//!   ctl * rgb`, `ctl` = `sun 1st bounce scaler` (sun) / `sky light 1st bounce scaler` (sky);
//!   exit radiance of the first bounce = `albedo * D`. Photon transport clamps the albedo so its
//!   largest channel is <= 0.8 (`sub_14018DAB0`). PROVEN.
//! * ENCODER (`sub_140610490`, `sub_140620190`): the SH is fitted by ONE directional vMF + ONE
//!   isotropic lobe; direction from the luminance l=1 band, bandwidth law `kappa(t)`
//!   (`sub_140610C30`), colours in units of `P`. PROVEN.
//! * DECODER: `E(N) = (LUT(N.d, t) * dom + 0.25 * fill) / pi`, the `rasterizer\diffusetable` LUT being
//!   the vMF(kappa(t)) convolved with the clamped cosine (max error 1.3 %). PROVEN.

use std::f64::consts::PI;

/// One sky-light record: `dir` points TOWARD the light (as stored by `SceneController::sky_lights`),
/// `rgb` is radiance, `omega` the solid angle in steradians.
#[derive(Clone, Copy, Debug)]
pub struct SkyRecord {
    pub dir: [f32; 3],
    pub rgb: [f32; 3],
    pub omega: f32,
}

/// The sky render_model's light list split the way the Tool splits it: the sun (the last record
/// unless `mode.flags & 8`; in practice always the brightest) and the sky records.
#[derive(Clone, Debug)]
pub struct SkyLights {
    pub sky: Vec<SkyRecord>,
    pub sun: Option<SkyRecord>,
    /// `sun 1st bounce scaler` / `sky light 1st bounce scaler` (sky `mode` +0x244 / +0x248).
    pub sun_1st: f32,
    pub sky_1st: f32,
}

impl SkyLights {
    /// Split a record list. The Tool takes the LAST record as the sun; `sky_lights()` returns the
    /// records in authored order, so `sun_is_last` is the faithful choice. A list whose last
    /// record is not the brightest is reported by `sun_is_brightest` (a gate pins the real maps).
    pub fn from_records(mut rec: Vec<SkyRecord>, sun_is_last: bool, sun_1st: f32, sky_1st: f32) -> Self {
        let sun = if sun_is_last { rec.pop() } else { None };
        SkyLights { sky: rec, sun, sun_1st, sky_1st }
    }
    pub fn sun_is_brightest(&self) -> bool {
        let p = |r: &SkyRecord| (r.rgb[0] + r.rgb[1] + r.rgb[2]) * r.omega;
        match &self.sun {
            Some(s) => self.sky.iter().all(|r| p(r) <= p(s)),
            None => true,
        }
    }
    /// Sun irradiance on a surface facing it (`rgb * Omega`).
    pub fn sun_irradiance(&self) -> [f32; 3] {
        self.sun.map_or([0.0; 3], |s| [s.rgb[0] * s.omega, s.rgb[1] * s.omega, s.rgb[2] * s.omega])
    }
}

/// SH of incident radiance, per channel, in the TOOL basis/order (see module docs).
pub type Sh9 = [f64; 9];
pub type Sh3 = [Sh9; 3];

/// The Tool's real SH basis (`sub_140610B70`).
pub fn basis(d: [f64; 3]) -> [f64; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282094806,
        -0.488602519 * y,
        0.488602519 * z,
        -0.488602519 * x,
        1.09254849 * x * y,
        -1.09254849 * y * z,
        0.946174681 * z * z - 0.31539157,
        -1.09254849 * x * z,
        0.546274245 * (x * x - y * y),
    ]
}

/// The plain graphics SH basis `rtgi.rs` stores (all `+` signs).
pub fn basis_graphics(d: [f64; 3]) -> [f64; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Convert nine RGB coefficients stored in the graphics basis into the Tool's basis. The two
/// bases differ only by the sign of terms 1, 3, 5 and 7.
pub fn from_graphics_sh(g: &[[f32; 3]; 9]) -> Sh3 {
    let mut out: Sh3 = [[0.0; 9]; 3];
    for k in 0..9 {
        let s = if matches!(k, 1 | 3 | 5 | 7) { -1.0 } else { 1.0 };
        for c in 0..3 {
            out[c][k] = s * g[k][c] as f64;
        }
    }
    out
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The Tool's direct sky SH for a receiver: `SH_c[k] += rgb_c * Omega * V * Y_k(l)` over the sky
/// records (the sun is NOT among them), hemisphere-clipped by `N.L > 0` when `normal` is given
/// (`None` = a probe: every direction). `vis(i)` is the per-record transmittance (1 = open sky).
/// Port of `faux_light_sky.cpp` + the sink `sub_1401974A0`.
pub fn sky_sh_from_list(sky: &[SkyRecord], normal: Option<[f32; 3]>, vis: &dyn Fn(usize) -> [f32; 3]) -> Sh3 {
    let mut sh: Sh3 = [[0.0; 9]; 3];
    for (i, r) in sky.iter().enumerate() {
        let l = [r.dir[0] as f64, r.dir[1] as f64, r.dir[2] as f64];
        if let Some(n) = normal {
            if dot3([n[0] as f64, n[1] as f64, n[2] as f64], l) <= 0.0 {
                continue;
            }
        }
        let t = vis(i);
        let y = basis(l);
        for c in 0..3 {
            let w = r.rgb[c] as f64 * r.omega as f64 * t[c] as f64;
            for k in 0..9 {
                sh[c][k] += w * y[k];
            }
        }
    }
    sh
}

/// Radiance integral `P_c = 2 sqrt(pi) * SH_c[0]` (`sub_14061ADC0`: `DC * 3.54490757`).
pub fn radiance_integral(sh: &Sh3) -> [f64; 3] {
    [3.5449077 * sh[0][0], 3.5449077 * sh[1][0], 3.5449077 * sh[2][0]]
}

/// Exact cosine-convolved irradiance of an SH of radiance at `n`, in the Tool's basis.
pub fn sh_irradiance(sh: &Sh3, n: [f64; 3]) -> [f64; 3] {
    let y = basis(n);
    let a = [PI, 2.0 * PI / 3.0, 2.0 * PI / 3.0, 2.0 * PI / 3.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0, PI / 4.0];
    let mut e = [0.0; 3];
    for c in 0..3 {
        for k in 0..9 {
            e[c] += a[k] * sh[c][k] * y[k];
        }
        e[c] = e[c].max(0.0);
    }
    e
}

// ---------------------------------------------------------------------------------------------
// Bandwidth law and the vMF band factors
// ---------------------------------------------------------------------------------------------

/// Tool `sub_140610C30`: engine bandwidth `t` in [0,1] -> vMF concentration `kappa`.
/// `kappa(0) = 0.456`, `kappa(0.5) = 1.575`, `kappa(1) = 55.79`.
pub fn kappa_of_t(t: f64) -> f64 {
    55.3843117 * (56.9768753 - 56.0 * t).powf(-0.93375814) - 0.814276755
}
/// Inverse of [`kappa_of_t`] (`sub_14060FF40`).
pub fn t_of_kappa(k: f64) -> f64 {
    (56.9768753 - ((k + 0.814276755) / 55.3843117).powf(1.0 / -0.93375814)) / 56.0
}

/// Band factors `(1, A1, A2)` of a normalised vMF (`sub_14060FDF0`).
fn zonal_bands(kappa: f64) -> [f64; 3] {
    let k = kappa.max(1e-4);
    let coth = 1.0 / k.tanh();
    [1.0, coth - 1.0 / k, 1.0 - 3.0 * coth / k + 3.0 / (k * k)]
}

/// The Tool's luminance weights (`sub_140610490`: 0.212656006 / 0.715157986 / 0.0721855983).
pub const TOOL_LUMA: [f64; 3] = [0.212656006, 0.715157986, 0.0721855983];

/// One directional vMF lobe + one isotropic lobe: what an atlas texel / probe stores.
#[derive(Clone, Copy, Debug)]
pub struct DualVmf {
    /// Unit direction TOWARD the dominant light.
    pub dir: [f64; 3],
    pub kappa: f64,
    /// Engine bandwidth in [0,1] (`t_of_kappa`).
    pub t: f64,
    /// Dominant lobe colour = `P * share` (radiance integral units).
    pub dom: [f64; 3],
    /// Isotropic lobe colour = `P * (1 - share)`.
    pub fill: [f64; 3],
    /// The fit objective at the optimum.
    pub err: f64,
}

/// Number of `kappa` samples of the 1-D search (the Tool runs BFGS; for a lobe-shaped SH both reach
/// the same optimum, `07-tool-lightmapper-re.md` s4.1).
pub const FIT_KAPPA_SAMPLES: usize = 400;

/// Port of `lm_compare.py:fit_sh_to_dualvmf` (the Tool objective of `sub_140620190`): direction from
/// the luminance l=1 band, band weights 10 (l=1) and 1 (l=2), per-channel closed-form share clipped
/// to [0,1], 1-D geometric search over `kappa` in [0.456, 55.8].
pub fn fit_sh_to_dualvmf(sh: &Sh3) -> DualVmf {
    let mut lum = [0.0f64; 9];
    for k in 0..9 {
        lum[k] = TOOL_LUMA[0] * sh[0][k] + TOOL_LUMA[1] * sh[1][k] + TOOL_LUMA[2] * sh[2][k];
    }
    let mut d = [-lum[3], -lum[1], lum[2]];
    let n = norm3(d);
    d = if n > 1e-9 { [d[0] / n, d[1] / n, d[2] / n] } else { [0.0, 0.0, 1.0] };
    let y = basis(d);
    const W: [f64; 9] = [0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let p = radiance_integral(sh);
    let (k0, k1) = (0.456f64, 55.8f64);
    let mut best: Option<(f64, f64, [f64; 3])> = None;
    for i in 0..FIT_KAPPA_SAMPLES {
        let k = k0 * (k1 / k0).powf(i as f64 / (FIT_KAPPA_SAMPLES - 1) as f64);
        let z = zonal_bands(k);
        let band = [1.0, z[1], z[1], z[1], z[2], z[2], z[2], z[2], z[2]];
        let mut err = 0.0;
        let mut share = [0.0f64; 3];
        for c in 0..3 {
            let (mut num, mut den) = (0.0, 0.0);
            for j in 1..9 {
                let m = band[j] * y[j] * p[c];
                num += W[j] * m * sh[c][j];
                den += W[j] * m * m;
            }
            let s = if den > 1e-30 { (num / den).clamp(0.0, 1.0) } else { 0.0 };
            share[c] = s;
            for j in 1..9 {
                let m = band[j] * y[j] * p[c];
                err += W[j] * (s * m - sh[c][j]).powi(2);
            }
        }
        if best.map_or(true, |b| err < b.0) {
            best = Some((err, k, share));
        }
    }
    let (err, kappa, share) = best.unwrap();
    let pc = [p[0].max(0.0), p[1].max(0.0), p[2].max(0.0)];
    DualVmf {
        dir: d,
        kappa,
        t: t_of_kappa(kappa).clamp(0.0, 1.0),
        dom: [pc[0] * share[0], pc[1] * share[1], pc[2] * share[2]],
        fill: [pc[0] * (1.0 - share[0]), pc[1] * (1.0 - share[1]), pc[2] * (1.0 - share[2])],
        err,
    }
}

// ---------------------------------------------------------------------------------------------
// The decoder: engine LUT and E(N)
// ---------------------------------------------------------------------------------------------

/// The exact value the engine's `rasterizer\diffusetable` approximates: the normalised vMF(kappa(t))
/// around `d` convolved with `max(0, n.w)`, for `n.d = c`.
pub fn lut_exact(c: f64, t: f64) -> f64 {
    let k = kappa_of_t(t);
    let (nt, np) = (720usize, 96usize);
    let norm = k / (2.0 * PI * (1.0 - (-2.0 * k).exp()));
    let n = [(1.0 - c * c).max(0.0).sqrt(), 0.0, c];
    let mut acc = 0.0;
    for i in 0..nt {
        let th = (i as f64 + 0.5) / nt as f64 * PI;
        let f = norm * (k * (th.cos() - 1.0)).exp();
        let w = th.sin() * (PI / nt as f64) * (2.0 * PI / np as f64);
        for j in 0..np {
            let ph = (j as f64 + 0.5) / np as f64 * 2.0 * PI;
            let om = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
            acc += f * dot3(om, n).max(0.0) * w;
        }
    }
    acc
}

static VMF_DIFFUSE_LUT: &[u8] = include_bytes!("vmf_diffuse_lut.bin"); // 128x128 A8 (rasterizer\diffusetable)

/// The engine LUT as shipped (bilinear, `x = n.d * 0.5 + 0.5`, `y = t`).
pub fn lut_table(c: f64, t: f64) -> f64 {
    const W: usize = 128;
    let x = ((c * 0.5 + 0.5).clamp(0.0, 1.0)) * (W - 1) as f64;
    let y = t.clamp(0.0, 1.0) * (W - 1) as f64;
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(W - 1), (y0 + 1).min(W - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let s = |xi: usize, yi: usize| VMF_DIFFUSE_LUT[yi * W + xi] as f64 / 255.0;
    (s(x0, y0) * (1.0 - fx) + s(x1, y0) * fx) * (1.0 - fy) + (s(x0, y1) * (1.0 - fx) + s(x1, y1) * fx) * fy
}

/// The engine's reconstruction `E(N) = (LUT(N.d, t) * dom + 0.25 * fill) / pi` of a dual vMF — what
/// the game shows for a surface (lightmap texel) or an object (probe) with this normal. The result
/// is irradiance / pi in the units of the lobes' radiance integral `P`.
pub fn diffuse_from_dual_vmf(n: [f64; 3], f: &DualVmf) -> [f64; 3] {
    let lut = lut_table(dot3(n, f.dir), f.t);
    let mut e = [0.0; 3];
    for c in 0..3 {
        e[c] = (lut * f.dom[c] + 0.25 * f.fill[c]) / PI;
    }
    e
}

/// 16-half probe record (airprobe `+0x14` / single probe `+0x00`):
/// `[dir0(3), mask0, colour0(3), bw0, dir1(3), mask1, colour1(3), bw1]`, the layout
/// `scene.rs::probe_vmf_from_terms` decodes. `mask` is the baked sun visibility (1 = unoccluded).
pub fn engine_probe_terms(f: &DualVmf, sun_vis: f32) -> [f32; 16] {
    [
        f.dir[0] as f32, f.dir[1] as f32, f.dir[2] as f32, sun_vis,
        f.dom[0] as f32, f.dom[1] as f32, f.dom[2] as f32, f.t as f32,
        0.0, 0.0, 0.0, sun_vis,
        f.fill[0] as f32, f.fill[1] as f32, f.fill[2] as f32, 0.0,
    ]
}

// ---------------------------------------------------------------------------------------------
// Bounce source and the albedo clamp
// ---------------------------------------------------------------------------------------------

/// Photon-transport albedo (`sub_14018DAB0`): scaled so the largest channel is <= 0.8.
pub fn photon_albedo(rho: [f32; 3]) -> [f32; 3] {
    let m = rho[0].max(rho[1]).max(rho[2]);
    if m <= 0.8 { rho } else { [rho[0] * 0.8 / m, rho[1] * 0.8 / m, rho[2] * 0.8 / m] }
}

/// The first-bounce SOURCE `D` (RGB) of a receiver with normal `n` (`sub_1401974A0` +
/// `sub_140196D50`): `D = 1/pi * sum clamp(m * N.L * Omega, 0, 1) * ctl * rgb * V` over the sky
/// records (`ctl = sky_1st`) and the sun (`ctl = sun_1st`, with `sun_vis` its visibility).
/// `m` is the receiving material's "analytical/normal light absorb" (1.0 by default).
pub fn bounce_source(
    n: [f32; 3], lights: &SkyLights, sky_vis: &dyn Fn(usize) -> [f32; 3], sun_vis: [f32; 3], absorb_normal: f32, absorb_analytic: f32,
) -> [f32; 3] {
    let mut d = [0.0f64; 3];
    let ndl = |l: [f32; 3]| (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]) as f64;
    for (i, r) in lights.sky.iter().enumerate() {
        let c = ndl(r.dir);
        if c <= 0.0 {
            continue;
        }
        let w = (absorb_normal as f64 * c * r.omega as f64).clamp(0.0, 1.0);
        let t = sky_vis(i);
        for k in 0..3 {
            d[k] += w * lights.sky_1st as f64 * r.rgb[k] as f64 * t[k] as f64;
        }
    }
    if let Some(s) = lights.sun {
        let c = ndl(s.dir);
        if c > 0.0 {
            let w = (absorb_analytic as f64 * c * s.omega as f64).clamp(0.0, 1.0);
            for k in 0..3 {
                d[k] += w * lights.sun_1st as f64 * s.rgb[k] as f64 * sun_vis[k] as f64;
            }
        }
    }
    [(d[0] / PI) as f32, (d[1] / PI) as f32, (d[2] / PI) as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift for the property tests (no `rand` dependency in this crate).
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn unit(&mut self) -> [f64; 3] {
            let z = 2.0 * self.f() - 1.0;
            let p = 2.0 * PI * self.f();
            let r = (1.0 - z * z).sqrt();
            [r * p.cos(), r * p.sin(), z]
        }
    }

    /// Spherical-Fibonacci sky of `n` records covering the upper hemisphere with uniform radiance.
    pub(crate) fn dome(n: usize, rgb: [f32; 3]) -> Vec<SkyRecord> {
        (0..n)
            .map(|i| {
                let z = 1.0 - (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).sqrt();
                let p = i as f64 * 2.399963229728653;
                SkyRecord { dir: [(r * p.cos()) as f32, (r * p.sin()) as f32, z as f32], rgb, omega: (2.0 * PI / n as f64) as f32 }
            })
            .collect()
    }

    /// The bandwidth law against the numbers the decompile gives (`sub_140610C30`).
    #[test]
    fn bandwidth_law_matches_the_decompiled_curve() {
        for (t, want) in [(0.0, 0.456), (0.5, 1.575), (1.0, 55.79)] {
            let k = kappa_of_t(t);
            assert!((k - want).abs() / want < 0.005, "kappa({t}) = {k}, decompile says {want}");
            assert!((t_of_kappa(k) - t).abs() < 1e-9, "t_of_kappa is not the inverse at {t}");
        }
    }

    /// `rasterizer\diffusetable` IS vMF(kappa(t)) convolved with the clamped cosine (07 s4.2):
    /// the table and the exact integral agree at the five measured points and over a grid. The
    /// saboteur is the exponent of the bandwidth law (07 used -0.85: this goes red).
    #[test]
    fn the_engine_lut_is_the_vmf_cosine_convolution() {
        for (c, t, tab) in [(1.0, 0.0, 0.329), (0.0, 0.0, 0.247), (1.0, 0.5, 0.514), (1.0, 1.0, 0.969), (0.5, 0.75, 0.414)] {
            let e = lut_exact(c, t);
            assert!((e - tab).abs() / tab < 0.02, "LUT({c},{t}): convolution {e:.4}, engine {tab}");
            let g = lut_table(c, t);
            assert!((g - tab).abs() / tab < 0.01, "shipped table sample LUT({c},{t}) = {g:.4}, expected {tab}");
        }
        let (mut worst, mut n) = (0.0f64, 0);
        for ci in 0..=10 {
            for ti in 0..=10 {
                let (c, t) = (-1.0 + 0.2 * ci as f64, 0.1 * ti as f64);
                let (a, b) = (lut_exact(c, t), lut_table(c, t));
                worst = worst.max((a - b).abs());
                n += 1;
            }
        }
        assert_eq!(n, 121);
        assert!(worst < 0.016, "table vs convolution worst abs error {worst:.4} over 121 grid points");
    }

    /// Round trip: build the SH of a known dual-vMF with the exact model, fit, recover it. 200 random
    /// lobes (07's selftest). Saboteur: change a band weight or `zonal_bands` and the direction or
    /// share goes wrong.
    #[test]
    fn the_fit_recovers_the_lobe_it_was_built_from() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut worst = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..200 {
            let d = rng.unit();
            let k = (0.5f64.ln() + (50.0f64 / 0.5).ln() * rng.f()).exp();
            let share = [0.2 + 0.8 * rng.f(), 0.2 + 0.8 * rng.f(), 0.2 + 0.8 * rng.f()];
            let p = [0.5 + 19.5 * rng.f(), 0.5 + 19.5 * rng.f(), 0.5 + 19.5 * rng.f()];
            let y = basis(d);
            let z = zonal_bands(k);
            let band = [1.0, z[1], z[1], z[1], z[2], z[2], z[2], z[2], z[2]];
            let mut sh: Sh3 = [[0.0; 9]; 3];
            for c in 0..3 {
                for j in 1..9 {
                    sh[c][j] = p[c] * share[c] * band[j] * y[j];
                }
                sh[c][0] = p[c] * 0.282094806;
            }
            let f = fit_sh_to_dualvmf(&sh);
            let derr = 1.0 - dot3(f.dir, d);
            let kerr = (f.kappa / k).ln().abs();
            let serr = (0..3).map(|c| (f.dom[c] / p[c] - share[c]).abs()).fold(0.0, f64::max);
            worst = (worst.0.max(derr), worst.1.max(kerr), worst.2.max(serr));
            assert!(derr < 1e-3 && kerr < 0.05 && serr < 0.03, "lobe not recovered: dir err {derr}, ln(kappa) err {kerr}, share err {serr}");
        }
        eprintln!("fit round trip (200 lobes): worst dir err {:.2e}, ln-kappa err {:.3}, share err {:.3}", worst.0, worst.1, worst.2);
    }

    /// The graphics<->Tool basis conversion: an SH built with the graphics basis, converted, reproduces
    /// the Tool-basis SH of the same radiance.
    #[test]
    fn graphics_and_tool_bases_differ_only_by_odd_term_signs() {
        let mut rng = Rng(77);
        for _ in 0..50 {
            let d = rng.unit();
            let (g, t) = (basis_graphics(d), basis(d));
            let mut sh = [[0f32; 3]; 9];
            for k in 0..9 {
                sh[k] = [g[k] as f32 * 2.0, g[k] as f32 * 3.0, g[k] as f32 * 0.5];
            }
            let conv = from_graphics_sh(&sh);
            for k in 0..9 {
                assert!((conv[0][k] - 2.0 * t[k]).abs() < 1e-5 && (conv[1][k] - 3.0 * t[k]).abs() < 1e-5, "term {k}");
            }
        }
    }

    /// A uniform hemisphere sky on an open floor: DC = L * 2 pi * Y00 (P = 2 pi L), the l=1 terms
    /// point up, and the hemisphere clip removes the records below the horizon.
    #[test]
    fn a_uniform_sky_integrates_to_two_pi_l() {
        let rec = dome(400, [0.5, 1.0, 2.0]);
        let sh = sky_sh_from_list(&rec, Some([0.0, 0.0, 1.0]), &|_| [1.0; 3]);
        let p = radiance_integral(&sh);
        for c in 0..3 {
            let want = 2.0 * PI * rec[0].rgb[c] as f64;
            assert!((p[c] - want).abs() / want < 0.01, "P[{c}] = {} want {}", p[c], want);
        }
        // a downward-facing receiver sees nothing of an upper-hemisphere dome
        let down = sky_sh_from_list(&rec, Some([0.0, 0.0, -1.0]), &|_| [1.0; 3]);
        assert_eq!(radiance_integral(&down), [0.0; 3]);
        // a probe (no normal) accepts every record
        let probe = sky_sh_from_list(&rec, None, &|_| [1.0; 3]);
        assert!((radiance_integral(&probe)[1] - p[1]).abs() < 1e-9, "the dome is above the horizon either way");
        // the fit of an open-floor sky is a broad lobe straight up
        let f = fit_sh_to_dualvmf(&sh);
        assert!(f.dir[2] > 0.999, "dir {:?}", f.dir);
        assert!(f.t > 0.4 && f.t < 0.8, "t = {}", f.t);
    }

    /// The engine reconstruction of the fit is close to the exact cosine convolution of the SH it
    /// was fitted to (the whole point of using it for objects): +-10 % over many normals on a sky
    /// that is a real lobe.
    #[test]
    fn the_dual_vmf_decode_tracks_the_exact_irradiance() {
        let rec = dome(400, [1.0, 1.0, 1.0]);
        let sh = sky_sh_from_list(&rec, Some([0.0, 0.0, 1.0]), &|_| [1.0; 3]);
        let f = fit_sh_to_dualvmf(&sh);
        let mut rng = Rng(5);
        let (mut n, mut worst) = (0, 0.0f64);
        for _ in 0..200 {
            let nrm = rng.unit();
            if nrm[2] < 0.2 {
                continue; // grazing/below-horizon normals are where a 2-lobe model is allowed to differ
            }
            let exact = sh_irradiance(&sh, nrm)[1] / PI;
            let eng = diffuse_from_dual_vmf(nrm, &f)[1];
            worst = worst.max((eng - exact).abs() / exact);
            n += 1;
        }
        assert!(n > 50, "only {n} normals tested");
        assert!(worst < 0.10, "engine decode differs from the exact irradiance by {:.1} % over {n} normals", worst * 100.0);
    }

    /// The sun is not part of the probe SH: the SH of a list with and without the sun record is
    /// identical, and the sun only shows up as the separate bounce term. (Saboteur: put the sun
    /// into the `sky` vector and the first assertion goes red.)
    #[test]
    fn the_sun_never_enters_the_sky_sh() {
        let sky = dome(100, [0.4, 0.5, 0.6]);
        let sun = SkyRecord { dir: [0.5, 0.0, 0.866], rgb: [1000.0, 1000.0, 900.0], omega: 0.003 };
        let mut all = sky.clone();
        all.push(sun);
        let lights = SkyLights::from_records(all, true, 0.3, 1.0);
        assert_eq!(lights.sky.len(), 100);
        assert!(lights.sun.is_some() && lights.sun_is_brightest());
        let with = sky_sh_from_list(&lights.sky, None, &|_| [1.0; 3]);
        let without = sky_sh_from_list(&sky, None, &|_| [1.0; 3]);
        assert_eq!(with, without, "the sun record leaked into the SH");
        // ...and the bounce source does see it, scaled by the sun's first-bounce scaler
        let n = [0.0, 0.0, 1.0];
        let only_sun = bounce_source(n, &SkyLights { sky: vec![], ..lights.clone() }, &|_| [1.0; 3], [1.0; 3], 1.0, 1.0);
        let want = 0.3 * 1000.0 * 0.003 * 0.866 / PI as f32;
        assert!((only_sun[0] - want).abs() / want < 1e-4, "sun bounce {} want {}", only_sun[0], want);
    }

    /// The first-bounce scalers multiply ONLY the bounce source (07 C7/C8): 0.30 vs 1.0 on the sun
    /// scales the sun's part by exactly 0.3 and leaves the sky part alone.
    #[test]
    fn the_first_bounce_scalers_act_on_the_bounce_source_only() {
        let sky = dome(100, [0.4, 0.5, 0.6]);
        let sun = SkyRecord { dir: [0.0, 0.6, 0.8], rgb: [800.0, 700.0, 600.0], omega: 0.004 };
        let mk = |sun_1st: f32, sky_1st: f32| SkyLights { sky: sky.clone(), sun: Some(sun), sun_1st, sky_1st };
        let n = [0.0, 0.0, 1.0];
        let full = bounce_source(n, &mk(1.0, 1.0), &|_| [1.0; 3], [1.0; 3], 1.0, 1.0);
        let sun_scaled = bounce_source(n, &mk(0.3, 1.0), &|_| [1.0; 3], [1.0; 3], 1.0, 1.0);
        let sky_only = bounce_source(n, &SkyLights { sun: None, ..mk(1.0, 1.0) }, &|_| [1.0; 3], [1.0; 3], 1.0, 1.0);
        for c in 0..3 {
            let sun_part = full[c] - sky_only[c];
            assert!(sun_part > 0.0);
            assert!((sun_scaled[c] - (sky_only[c] + 0.3 * sun_part)).abs() < 1e-5 * full[c].max(1.0), "channel {c}: the sun scaler changed more than the sun term");
        }
        // the SH does not depend on either scaler
        let a = sky_sh_from_list(&mk(1.0, 1.0).sky, None, &|_| [1.0; 3]);
        let b = sky_sh_from_list(&mk(0.3, 0.5).sky, None, &|_| [1.0; 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn photon_albedo_is_clamped_by_scaling_to_point_eight() {
        assert_eq!(photon_albedo([0.5, 0.4, 0.3]), [0.5, 0.4, 0.3]);
        let a = photon_albedo([1.0, 0.5, 0.25]);
        assert!((a[0] - 0.8).abs() < 1e-6 && (a[1] - 0.4).abs() < 1e-6 && (a[2] - 0.2).abs() < 1e-6, "{a:?}");
    }

    /// The probe record round-trips through the layout `probe_vmf_from_terms` reads.
    #[test]
    fn engine_probe_terms_use_the_airprobe_layout() {
        let f = DualVmf { dir: [0.0, 0.6, 0.8], kappa: 2.0, t: 0.55, dom: [3.0, 2.0, 1.0], fill: [0.5, 0.4, 0.3], err: 0.0 };
        let t = engine_probe_terms(&f, 1.0);
        assert_eq!([t[0], t[1], t[2]], [0.0, 0.6, 0.8]);
        assert_eq!([t[4], t[5], t[6]], [3.0, 2.0, 1.0]);
        assert_eq!(t[7], 0.55);
        assert_eq!([t[12], t[13], t[14]], [0.5, 0.4, 0.3]);
        assert_eq!([t[8], t[9], t[10]], [0.0; 3], "fill direction is zero (isotropic)");
    }
}
