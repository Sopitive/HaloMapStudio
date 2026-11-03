//! #save-precision — choose the CODES a `.mvar` stores so the pose the GAME decodes is as close as
//! possible to the pose the user built.
//!
//! The three codecs (Reach, Halo 4 / H2A, Halo 3) all store an orientation as
//!
//! * an UP vector on a cube-face grid (a face + two ratios of the other components), and
//! * a FORWARD angle about that up, measured from a reference tangent rebuilt from the
//!   DEQUANTISED up (so the angle's meaning depends on which up code was chosen).
//!
//! The original encoders rounded each field on its own: nearest up cell on the dominant face,
//! then the nearest angle code given that one up. That is not the best code PAIR: the grid is
//! square in ratio space, not in angle, so the nearest cell in ratio can be a worse direction than
//! a neighbouring one, and the dominant face is not the only face that can hold a direction. This
//! module searches the code space near the intended orientation and scores every candidate by
//! DECODING it with the game's own decoder and measuring the rotation angle between the decoded
//! basis and the intended one (quaternion angle, f64). The cheap search (a few cells on each of
//! the three faces the vector touches, a hill-climb on the angle) is proved against a much wider
//! brute force in the tests, and the original logic is always one of the candidates, so the result
//! can never be worse than it was.
//!
//! Positions are chosen the same way: the nearest code by the decoder's own f32 arithmetic (the
//! reader's `(0.5 + q) * step + min` is evaluated in f32, which for a large box is not exactly
//! monotone against the f64 quantiser).
//!
//! Everything is pure math; the games' bit layouts live in `mvar` / `h4::mvar` / `h3::mvar`, which
//! call [`best_reach_axes`], [`best_h4_axes`], [`best_h3_axes`] and the `*_pos_code` functions.

use std::sync::OnceLock;

/// Faces compose as (axis, other-two-components); face `f` is axis `f % 3`, negative when `f >= 3`.
/// The first / second ratio of face axis 0 is (y, z), of axis 1 (x, z), of axis 2 (x, y).
const FACE_COMPS: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];

pub type Basis = [[f64; 3]; 3];

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn norm(v: [f64; 3]) -> [f64; 3] { let m = dot(v, v).sqrt(); if m > 1e-300 { [v[0] / m, v[1] / m, v[2] / m] } else { [0.0; 3] } }
fn f64v(v: [f32; 3]) -> [f64; 3] { [v[0] as f64, v[1] as f64, v[2] as f64] }

/// The orthonormal basis (columns forward, left = up x forward, up) of a possibly sloppy
/// forward/up pair: forward primary, up Gram-Schmidt'd against it. The same choice `euler.rs` and
/// the save-precision numbers make, in f64 so a hundredth of a degree is still measurable.
pub fn basis_of(fwd: [f32; 3], up: [f32; 3]) -> Basis {
    let mut x = norm(f64v(fwd));
    if dot(x, x) < 0.5 { x = [1.0, 0.0, 0.0]; }
    let u = f64v(up);
    let mut z = [u[0] - x[0] * dot(u, x), u[1] - x[1] * dot(u, x), u[2] - x[2] * dot(u, x)];
    if dot(z, z) < 1e-12 {
        let r = if x[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        z = [r[0] - x[0] * dot(r, x), r[1] - x[1] * dot(r, x), r[2] - x[2] * dot(r, x)];
    }
    let z = norm(z);
    let y = cross(z, x);
    [x, y, z]
}

/// The angle (degrees) of the rotation taking basis `a` to basis `b`: atan2(|skew|/2, (tr-1)/2)
/// of `R = A^T B`, which stays accurate down to the last bit (acos of a quaternion's w does not).
pub fn angle_between_deg(a: &Basis, b: &Basis) -> f64 {
    let r = |i: usize, j: usize| dot(a[i], b[j]); // R[i][j] = a_i . b_j
    let tr = r(0, 0) + r(1, 1) + r(2, 2);
    let s = [r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1)];
    let sin = 0.5 * dot(s, s).sqrt();
    let cos = 0.5 * (tr - 1.0);
    sin.atan2(cos).to_degrees()
}

/// Rotation error (degrees) of a decoded (forward, up) against the intended one.
pub fn rot_error_deg(want: &Basis, got_fwd: [f32; 3], got_up: [f32; 3]) -> f64 {
    angle_between_deg(want, &basis_of(got_fwd, got_up))
}

// ---------------------------------------------------------------------------------------------
// test hooks: the saboteur switch
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// While set, every `best_*_axes` returns the ORIGINAL nearest-per-field codes -- the
    /// saboteur the improvement gate must go red on.
    pub static FORCE_LEGACY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn force_legacy() -> bool { FORCE_LEGACY.with(|c| c.get()) }
#[cfg(not(test))]
fn force_legacy() -> bool { false }

// ---------------------------------------------------------------------------------------------
// the generic search
// ---------------------------------------------------------------------------------------------

/// One game's orientation codec, as the search needs it.
trait Codec {
    /// Number of forward-angle codes on the ring (codes `0..n`).
    fn n_angles(&self) -> u32;
    /// The game's own decode of a code triple -> (forward, up).
    fn decode(&self, global: bool, up_q: u32, fq: u32) -> ([f32; 3], [f32; 3]);
    /// Dequantised cell values of one up component, in code order.
    fn cells(&self) -> &[f64];
    fn pack(&self, face: usize, a: usize, b: usize) -> u32;
    /// The (fractional) angle code at which the decoder's forward is closest to `fwd` for this
    /// up code: code `i` has angle `i`, so the answer is a position on the code ring.
    fn ideal_angle_pos(&self, global: bool, up_q: u32, fwd: [f64; 3]) -> f64;
}

#[derive(Clone, Copy)]
struct Cand { global: bool, up_q: u32, fq: u32, err: f64 }

impl Cand {
    /// Strictly better error wins; on an exact tie the code that sorts first (global before
    /// quantised -- it is the smaller encoding -- then lower up, then lower angle) wins, so the
    /// choice is a function of the decoded pose alone and re-encoding it is a fixed point.
    fn better(&self, o: &Cand) -> bool {
        if self.err != o.err { return self.err < o.err; }
        (!self.global, self.up_q, self.fq) < (!o.global, o.up_q, o.fq)
    }
}

struct Search<'a, C: Codec> {
    c: &'a C,
    want: Basis,
    best: Option<Cand>,
}

impl<'a, C: Codec> Search<'a, C> {
    fn eval(&self, global: bool, up_q: u32, fq: u32) -> Cand {
        let (f, u) = self.c.decode(global, up_q, fq);
        Cand { global, up_q, fq, err: rot_error_deg(&self.want, f, u) }
    }
    fn offer(&mut self, c: Cand) {
        if self.best.map_or(true, |b| c.better(&b)) { self.best = Some(c); }
    }
    /// Best angle code for one up: start at the ideal position and walk the ring while it improves.
    /// `window` > 0 additionally scans that many codes each side exhaustively (the proof mode).
    fn best_angle(&mut self, global: bool, up_q: u32, fwd: [f64; 3], window: usize) -> Cand {
        let n = self.c.n_angles() as i64;
        let pos = self.c.ideal_angle_pos(global, up_q, fwd);
        let i0 = if pos.is_finite() { pos.floor() as i64 } else { 0 };
        let wrap = |i: i64| i.rem_euclid(n) as u32;
        let mut best = self.eval(global, up_q, wrap(i0));
        for i in [i0 + 1, i0 - 1, i0 + 2] {
            let c = self.eval(global, up_q, wrap(i));
            if c.better(&best) { best = c; }
        }
        // walk
        loop {
            let mut moved = false;
            for d in [-1i64, 1] {
                let c = self.eval(global, up_q, wrap(best.fq as i64 + d));
                if c.better(&best) { best = c; moved = true; }
            }
            if !moved { break; }
        }
        for k in -(window as i64)..=(window as i64) {
            let c = self.eval(global, up_q, wrap(i0 + k));
            if c.better(&best) { best = c; }
        }
        self.offer(best);
        best
    }
}

/// The nearest-cell indices around ratio `r`: `lo..=hi` clamped into the table.
fn cell_span(vals: &[f64], r: f64, radius: usize) -> (usize, usize) {
    let idx = vals.partition_point(|&v| v < r); // first cell >= r
    let lo = idx.saturating_sub(radius + 1);
    let hi = (idx + radius).min(vals.len().saturating_sub(1));
    (lo, hi)
}

/// Run the search. `legacy` is the original encoder's answer (always a candidate). `radius` = 1
/// is the shipped search; larger values widen the cell neighbourhood (and `faces_all` / `window`)
/// for the brute-force proof.
fn search<C: Codec>(c: &C, fwd: [f32; 3], up: [f32; 3], legacy: (bool, u32, u32), radius: usize, faces_all: bool, window: usize) -> (bool, u32, u32) {
    let want = basis_of(fwd, up);
    let fwd_t = want[0];
    let up_t = want[2];
    let mut s = Search { c, want, best: None };
    // the original answer is always a candidate: never worse than before
    let lc = s.eval(legacy.0, legacy.1, legacy.2);
    s.offer(lc);
    // global up: exact +Z, the 1-bit encoding. Always allowed (it is the smaller encoding).
    s.best_angle(true, 0, fwd_t, window);
    // an original that was global stays global: the record layout (bit length) must not grow
    if !legacy.0 {
        let vals = c.cells();
        let faces: Vec<usize> = if faces_all { (0..6).collect() } else {
            (0..3).map(|a| if up_t[a] < 0.0 { a + 3 } else { a }).collect()
        };
        for face in faces {
            let axis = face % 3;
            if up_t[axis].abs() < 1e-9 { continue; }
            // the face holds the direction  s*e_axis + u*e_a + w*e_b  (s = +-1)
            let sgn = if face >= 3 { -1.0 } else { 1.0 };
            let k = sgn / up_t[axis];
            if k < 0.0 { continue; } // the opposite side of the cube: that is -up, not up
            let (ca, cb) = FACE_COMPS[axis];
            let (ra, rb) = (up_t[ca] * k, up_t[cb] * k);
            let (alo, ahi) = cell_span(vals, ra.clamp(-1.0, 1.0), radius);
            let (blo, bhi) = cell_span(vals, rb.clamp(-1.0, 1.0), radius);
            for a in alo..=ahi {
                for b in blo..=bhi {
                    let up_q = c.pack(face, a, b);
                    s.best_angle(false, up_q, fwd_t, window);
                }
            }
        }
    }
    let b = s.best.unwrap();
    (b.global, b.up_q, b.fq)
}

// ---------------------------------------------------------------------------------------------
// Reach / Halo 4 / H2A: 20-bit up + 14-bit angle (one decoder, `mvar::decode_orientation`)
// ---------------------------------------------------------------------------------------------

struct R20 { vals: Vec<f64>, apamc: i64, qvc: i64 }

fn r20() -> &'static R20 {
    static T: OnceLock<R20> = OnceLock::new();
    T.get_or_init(|| {
        let (vals, apamc, qvc) = crate::mvar::unit_vector_cells(20);
        R20 { vals, apamc, qvc }
    })
}

impl Codec for R20 {
    fn n_angles(&self) -> u32 { 1 << 14 }
    fn decode(&self, global: bool, up_q: u32, fq: u32) -> ([f32; 3], [f32; 3]) { crate::mvar::decode_orientation(global, up_q, fq) }
    fn cells(&self) -> &[f64] { &self.vals }
    fn pack(&self, face: usize, a: usize, b: usize) -> u32 { (face as i64 * self.apamc + a as i64 * self.qvc + b as i64) as u32 }
    fn ideal_angle_pos(&self, global: bool, up_q: u32, fwd: [f64; 3]) -> f64 {
        let up = if global { [0.0, 0.0, 1.0] } else { crate::mvar::dequantize_unit_vector3d(up_q as i64, 20) };
        let (r, cr) = crate::mvar::reference_frame(up);
        let theta = dot(fwd, cr).atan2(dot(fwd, r));
        // code i decodes to angle -pi + (i + 0.5) * 2pi/2^14
        (theta + std::f64::consts::PI) / (2.0 * std::f64::consts::PI / 16384.0) - 0.5
    }
}

/// Reach: the best (up_is_global, up_quant, forward_angle_q) for an intended orientation.
pub fn best_reach_axes(fwd: [f32; 3], up: [f32; 3]) -> (bool, u32, u32) {
    let legacy = crate::mvar::encode_orientation_nearest(fwd, up);
    if force_legacy() { return legacy; }
    search(r20(), fwd, up, legacy, 1, false, 0)
}

/// Halo 4 / H2A: same decoder, the engine-exact (truncating) encoder is the baseline.
pub fn best_h4_axes(fwd: [f32; 3], up: [f32; 3]) -> (bool, u32, u32) {
    let legacy = crate::h4::mvar::h4_encode_axes_nearest(fwd, up);
    if force_legacy() { return legacy; }
    search(r20(), fwd, up, legacy, 1, false, 0)
}

/// The proof-mode search for the 20/14 codec: all six faces, `radius` cells each way, and an
/// exhaustive scan of `window` angle codes either side of the ideal one.
pub fn wide_r20(fwd: [f32; 3], up: [f32; 3], legacy: (bool, u32, u32), radius: usize, window: usize) -> (bool, u32, u32) {
    search(r20(), fwd, up, legacy, radius, true, window)
}

// ---------------------------------------------------------------------------------------------
// Halo 3: 19-bit up (3-bit face + two 8-bit components) + 8-bit angle
// ---------------------------------------------------------------------------------------------

struct H3Codec { vals: Vec<f64> }

fn h3c() -> &'static H3Codec {
    static T: OnceLock<H3Codec> = OnceLock::new();
    // a component code is 0..=254 (255 decodes like 254 and is never emitted)
    T.get_or_init(|| H3Codec { vals: (0..=254u32).map(|q| crate::h3::mvar::dequantize_real(q, -1.0, 1.0, 8, true) as f64).collect() })
}

impl Codec for H3Codec {
    // 0..=253 on the ring: code 254 decodes to the same direction as code 0 (the +-pi half turn)
    fn n_angles(&self) -> u32 { 254 }
    fn decode(&self, global: bool, up_q: u32, fq: u32) -> ([f32; 3], [f32; 3]) {
        let (f, u) = crate::h3::mvar::decode_axes(global, up_q, fq as u8);
        (f.to_array(), u.to_array())
    }
    fn cells(&self) -> &[f64] { &self.vals }
    fn pack(&self, face: usize, a: usize, b: usize) -> u32 { face as u32 | (a as u32) << 3 | (b as u32) << 11 }
    fn ideal_angle_pos(&self, global: bool, up_q: u32, fwd: [f64; 3]) -> f64 {
        use glam::Vec3;
        let up = if global { Vec3::Z } else { crate::h3::mvar::dequantize_unit_vector(up_q) };
        let seed = crate::h3::mvar::perpendicular_seed(up);
        let left = up.cross(seed).normalize_or_zero();
        let (s, l) = (f64v(seed.to_array()), f64v(left.to_array()));
        let theta = dot(fwd, l).atan2(dot(fwd, s));
        // code i decodes to angle lerp(-pi, pi, i/254)
        (theta + std::f64::consts::PI) / (2.0 * std::f64::consts::PI / 254.0)
    }
}

/// Halo 3: the best (up_is_global, up_q, angle_q).
pub fn best_h3_axes(fwd: [f32; 3], up: [f32; 3]) -> (bool, u32, u8) {
    let legacy = crate::h3::mvar::h3_axes_nearest(fwd, up);
    if force_legacy() { return legacy; }
    let l = (legacy.0, legacy.1, legacy.2 as u32);
    let (g, u, f) = search(h3c(), fwd, up, l, 1, false, 0);
    (g, u, f as u8)
}

pub fn wide_h3(fwd: [f32; 3], up: [f32; 3], legacy: (bool, u32, u8), radius: usize, window: usize) -> (bool, u32, u8) {
    let (g, u, f) = search(h3c(), fwd, up, (legacy.0, legacy.1, legacy.2 as u32), radius, true, window);
    (g, u, f as u8)
}

// ---------------------------------------------------------------------------------------------
// positions
// ---------------------------------------------------------------------------------------------

/// The code whose DECODED value (the reader's own f32 arithmetic, `dec`) is nearest `v`, looking
/// two codes either side of the quantiser's answer `q0`. Ties go to the code nearest `q0`, then
/// the lower one, so re-encoding a decoded value returns the code it came from.
pub fn best_pos_code(v: f32, q0: i64, maxq: i64, dec: &dyn Fn(i64) -> f32) -> i64 {
    let q0 = q0.clamp(0, maxq.max(0));
    if !v.is_finite() { return q0; }
    let mut best = q0;
    let mut best_e = (dec(q0) as f64 - v as f64).abs();
    for d in [-1i64, 1, -2, 2] {
        let q = q0 + d;
        if q < 0 || q > maxq { continue; }
        let e = (dec(q) as f64 - v as f64).abs();
        if e < best_e { best = q; best_e = e; }
    }
    best
}

/// Reach position code: the f64 quantiser's answer refined against the reader's f32 decode.
pub fn reach_pos_code(v: f32, mn: f32, ext: f32, bits: u32) -> u64 {
    let maxq = (1i64 << bits) - 1;
    let scale = ext as f64 / (1u64 << bits) as f64;
    let q0 = if scale > 0.0 && v.is_finite() { (((v as f64 - mn as f64) / scale) - 0.5).round() as i64 } else { 0 };
    let dec = |q: i64| (0.5 + q as f32) * (ext / (1u64 << bits) as f32) + mn;
    best_pos_code(v, q0, maxq, &dec) as u64
}

/// Halo 4 / H2A position code: the engine quantiser's answer refined against the reader's decode.
pub fn h4_pos_code(v: f32, mn: f32, mx: f32, bits: u32) -> u32 {
    let q0 = crate::h4::mvar::h4_quantize_pos(v, mn, mx, bits) as i64;
    let step = (mx - mn) / (1u64 << bits) as f32;
    let dec = |q: i64| mn + q as f32 * step + step * 0.5;
    let maxq = (1i64 << bits) - 1;
    // the reader works from the box extent the same way; clamp like the engine
    let v = if v.is_finite() { v.clamp(mn, mx) } else { v };
    best_pos_code(v, q0, maxq, &dec) as u32
}

/// Halo 3 position code.
pub fn h3_pos_code(v: f32, mn: f32, mx: f32) -> u16 {
    let q0 = crate::h3::mvar::quantize_pos(v, mn, mx) as i64;
    let dec = |q: i64| crate::h3::mvar::dequantize_real(q as u32, mn, mx, crate::h3::mvar::H3_POSITION_BITS, false);
    let v = if v.is_finite() { v.clamp(mn, mx) } else { v };
    best_pos_code(v, q0, 0xFFFF, &dec) as u16
}

// =============================================================================================
// tests: the optimiser against the original encoder, against brute force, on real objects
// =============================================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapcat::{variant_dirs_of, Game};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum G { Reach, H4, H3 }

    impl G {
        pub fn label(self) -> &'static str { match self { G::Reach => "Reach", G::H4 => "Halo 4/H2A", G::H3 => "Halo 3" } }
        /// (legacy codes, optimised codes) decoded -> rotation error against the intended pose
        pub fn errs(self, fwd: [f32; 3], up: [f32; 3]) -> (f64, f64) {
            let want = basis_of(fwd, up);
            match self {
                G::Reach | G::H4 => {
                    let (l, n) = if self == G::Reach {
                        (crate::mvar::encode_orientation_nearest(fwd, up), best_reach_axes(fwd, up))
                    } else {
                        (crate::h4::mvar::h4_encode_axes_nearest(fwd, up), best_h4_axes(fwd, up))
                    };
                    let el = { let (f, u) = crate::mvar::decode_orientation(l.0, l.1, l.2); rot_error_deg(&want, f, u) };
                    let en = { let (f, u) = crate::mvar::decode_orientation(n.0, n.1, n.2); rot_error_deg(&want, f, u) };
                    (el, en)
                }
                G::H3 => {
                    let l = crate::h3::mvar::h3_axes_nearest(fwd, up);
                    let n = best_h3_axes(fwd, up);
                    let el = { let (f, u) = crate::h3::mvar::decode_axes(l.0, l.1, l.2); rot_error_deg(&want, f.to_array(), u.to_array()) };
                    let en = { let (f, u) = crate::h3::mvar::decode_axes(n.0, n.1, n.2); rot_error_deg(&want, f.to_array(), u.to_array()) };
                    (el, en)
                }
            }
        }
    }

    pub struct Rng(pub u64);
    impl Rng {
        pub fn u(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// a uniformly random rotation as (forward, up)
        pub fn rotation(&mut self) -> ([f32; 3], [f32; 3]) {
            let (a, b, c) = (self.u(), self.u(), self.u());
            let (s1, s2) = ((1.0 - a).sqrt(), a.sqrt());
            let (t1, t2) = (2.0 * std::f64::consts::PI * b, 2.0 * std::f64::consts::PI * c);
            let q = glam::DQuat::from_xyzw(s1 * t1.sin(), s1 * t1.cos(), s2 * t2.sin(), s2 * t2.cos());
            let f = q * glam::DVec3::X;
            let u = q * glam::DVec3::Z;
            (f.as_vec3().to_array(), u.as_vec3().to_array())
        }
    }

    fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
        if v.is_empty() { return 0.0; }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p).round() as usize]
    }

    /// (median, p99, max)
    pub fn dist(v: &[f64]) -> (f64, f64, f64) {
        let mut w = v.to_vec();
        (pct(&mut w, 0.5), pct(&mut w, 0.99), w.iter().cloned().fold(0.0, f64::max))
    }

    /// The pose families real Forge editing produces: axis-aligned yaw (multiples of 90), snapped
    /// (multiples of 15 on every axis), free yaw, free tilt, and fully random.
    pub fn families(rng: &mut Rng, n: usize) -> Vec<(&'static str, [f32; 3], [f32; 3])> {
        let mut out = Vec::new();
        for _ in 0..n {
            let e90 = [0.0, 0.0, 90.0 * (rng.u() * 4.0).floor() as f32];
            let (f, u) = crate::euler::from_euler_deg(e90, 1.0, 1.0);
            out.push(("axis-aligned yaw", f, u));
            let snap = |r: f64, step: f32, span: f32| ((r as f32 * span / step).floor() * step) - span / 2.0;
            let e15 = [snap(rng.u(), 15.0, 360.0), snap(rng.u(), 15.0, 180.0).clamp(-75.0, 75.0), snap(rng.u(), 15.0, 360.0)];
            let (f, u) = crate::euler::from_euler_deg(e15, 1.0, 1.0);
            out.push(("snapped 15 deg", f, u));
            let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, -180.0 + 360.0 * rng.u() as f32], 1.0, 1.0);
            out.push(("free yaw", f, u));
            let (f, u) = crate::euler::from_euler_deg([-60.0 + 120.0 * rng.u() as f32, -60.0 + 120.0 * rng.u() as f32, -180.0 + 360.0 * rng.u() as f32], 1.0, 1.0);
            out.push(("free tilt", f, u));
            let (f, u) = rng.rotation();
            out.push(("uniform random", f, u));
        }
        out
    }

    /// Up to `n` `.mvar` files of one game.
    pub fn files_of(game: Game, n: usize) -> Vec<std::path::PathBuf> {
        let mut v: Vec<std::path::PathBuf> = variant_dirs_of(game).iter()
            .flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")))
            .collect();
        v.sort();
        v.truncate(n);
        v
    }

    /// The decoded rotations of REAL objects (what the placed pieces of shipped variants hold).
    fn real_rotations(g: G, max: usize) -> (usize, Vec<([f32; 3], [f32; 3])>) {
        let mut rots = Vec::new();
        let mut files = 0;
        match g {
            G::Reach => for f in files_of(Game::Reach, 40) {
                let Some(v) = crate::mvar::parse_variant(&f) else { continue };
                files += 1;
                rots.extend(v.objects.iter().map(|o| (o.fwd, o.up)));
            },
            G::H4 => for f in files_of(Game::Halo4, 40).into_iter().chain(files_of(Game::H2A, 20)) {
                let Ok(v) = crate::h4::mvar::parse_h4_variant(&f) else { continue };
                files += 1;
                rots.extend(v.objects.iter().map(|o| (o.fwd, o.up)));
            },
            G::H3 => for f in files_of(Game::Halo3, 40) {
                let Ok(v) = crate::h3::mvar::parse_h3_variant(&f) else { continue };
                files += 1;
                rots.extend(v.objects.iter().filter(|o| o.placed).map(|o| (o.fwd, o.up)));
            },
        }
        rots.truncate(max);
        (files, rots)
    }

    /// Print old vs new for every family on every game (the numbers quoted in the notes).
    #[test]
    fn measure_rotation_error_by_family() {
        for g in [G::Reach, G::H4, G::H3] {
            let mut rng = Rng(0x5A5A_0000 + g as u64);
            let samples = families(&mut rng, 600);
            for fam in ["axis-aligned yaw", "snapped 15 deg", "free yaw", "free tilt", "uniform random"] {
                let (mut old, mut new) = (Vec::new(), Vec::new());
                for (name, f, u) in &samples {
                    if *name != fam { continue; }
                    let (a, b) = g.errs(*f, *u);
                    old.push(a);
                    new.push(b);
                }
                let (o, n) = (dist(&old), dist(&new));
                eprintln!("{:<11} {:<17} n={:<4} old med {:.5} p99 {:.5} max {:.5} | new med {:.5} p99 {:.5} max {:.5} deg", g.label(), fam, old.len(), o.0, o.1, o.2, n.0, n.1, n.2);
            }
        }
    }

    /// Real objects (their stored, decoded rotation) with a sub-code user nudge: how a piece
    /// that really sits in a shipped variant fares when the user re-aims it freely.
    #[test]
    fn measure_rotation_error_on_real_objects() {
        for g in [G::Reach, G::H4, G::H3] {
            let (files, rots) = real_rotations(g, 20000);
            if rots.is_empty() { eprintln!("{}: no variants installed", g.label()); continue; }
            let mut rng = Rng(0xBEEF + g as u64);
            let (mut old, mut new, mut level_old, mut level_new) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (f, u) in &rots {
                // the ideal (un-quantised) rotation the user meant: the stored one plus a random
                // sub-degree wobble, i.e. a free re-aim of a real piece
                let w = glam::Quat::from_axis_angle(glam::Vec3::new(rng.u() as f32 - 0.5, rng.u() as f32 - 0.5, rng.u() as f32 - 0.5).normalize(), (rng.u() as f32) * 0.02);
                let (f2, u2) = ((w * glam::Vec3::from(*f)).to_array(), (w * glam::Vec3::from(*u)).to_array());
                let (a, b) = g.errs(f2, u2);
                old.push(a);
                new.push(b);
                if u[2] > 0.9995 { continue; }
                level_old.push(a);
                level_new.push(b);
            }
            let (o, n) = (dist(&old), dist(&new));
            eprintln!("{:<11} REAL objects: {} files, {} objs, free re-aim: old med {:.5} p99 {:.5} max {:.5} | new med {:.5} p99 {:.5} max {:.5} deg ({} tilted)", g.label(), files, rots.len(), o.0, o.1, o.2, n.0, n.1, n.2, level_old.len());
            let (o, n) = (dist(&level_old), dist(&level_new));
            eprintln!("{:<11}   tilted only: old med {:.5} p99 {:.5} max {:.5} | new med {:.5} p99 {:.5} max {:.5}", g.label(), o.0, o.1, o.2, n.0, n.1, n.2);
        }
    }

    // -----------------------------------------------------------------------------------------
    // GATES
    // -----------------------------------------------------------------------------------------

    /// Samples for the improvement gate: every family plus rotations of real objects.
    fn gate_samples(g: G) -> Vec<([f32; 3], [f32; 3])> {
        let mut rng = Rng(0x6A7E_0000 + g as u64);
        let mut v: Vec<_> = families(&mut rng, 800).into_iter().map(|(_, f, u)| (f, u)).collect();
        let (_, real) = real_rotations(g, 3000);
        for (f, u) in real {
            let w = glam::Quat::from_axis_angle(glam::Vec3::new(rng.u() as f32 - 0.5, rng.u() as f32 - 0.5, rng.u() as f32 - 0.5).normalize(), (rng.u() as f32) * 0.02);
            v.push(((w * glam::Vec3::from(f)).to_array(), (w * glam::Vec3::from(u)).to_array()));
        }
        v
    }

    /// (a) the optimised encoder is never worse than the old one, on every sample, and (b) it is
    /// STRICTLY better on at least `min_frac` of them. Panics when either fails.
    fn improvement_gate(g: G, min_frac: f64) -> (usize, usize) {
        let samples = gate_samples(g);
        let (mut better, mut worse) = (0usize, 0usize);
        for (f, u) in &samples {
            let (old, new) = g.errs(*f, *u);
            if new > old + 1e-9 { worse += 1; }
            if new < old - 1e-9 { better += 1; }
        }
        assert_eq!(worse, 0, "{}: the optimised encoder is WORSE than the old one on {worse} of {} samples", g.label(), samples.len());
        let frac = better as f64 / samples.len() as f64;
        eprintln!("{}: strictly better on {better} of {} samples ({:.1}%)", g.label(), samples.len(), frac * 100.0);
        assert!(frac >= min_frac, "{}: the optimiser improved only {:.2}% of samples (need >= {:.2}%)", g.label(), frac * 100.0, min_frac * 100.0);
        (better, samples.len())
    }

    const MIN_BETTER: [(G, f64); 3] = [(G::Reach, 0.02), (G::H4, 0.02), (G::H3, 0.05)];

    #[test]
    fn optimiser_never_worse_and_strictly_better() {
        for (g, m) in MIN_BETTER { improvement_gate(g, m); }
    }

    /// SABOTEUR: with the old nearest-per-field logic swapped in, the improvement assertion above
    /// must go red. (Proves the gate can fail.)
    #[test]
    fn saboteur_old_encoder_fails_the_improvement_gate() {
        FORCE_LEGACY.with(|c| c.set(true));
        let r = std::panic::catch_unwind(|| { for (g, m) in MIN_BETTER { improvement_gate(g, m); } });
        FORCE_LEGACY.with(|c| c.set(false));
        assert!(r.is_err(), "the improvement gate passed with the OLD encoder swapped in -- it cannot fail");
    }

    /// Brute force: the shipped (radius 1, three faces, hill-climbed angle) search finds the same
    /// optimum as a much wider one (six faces, +-6 cells, +-40 angle codes scanned exhaustively).
    #[test]
    fn fast_search_matches_the_wide_brute_force() {
        for g in [G::Reach, G::H3] {
            let mut rng = Rng(0xB01D + g as u64);
            let mut samples: Vec<([f32; 3], [f32; 3])> = (0..250).map(|_| rng.rotation()).collect();
            // the reference-axis tie: |up.x| ~ |up.y| (Reach/H4 flips its reference tangent there) and face edges
            for k in 0..60 {
                let pitch = 5.0 + 80.0 * rng.u() as f32;
                let (f, u) = crate::euler::from_euler_deg([0.0, pitch, 45.0 + 90.0 * (k % 4) as f32 + 0.05 * (rng.u() as f32 - 0.5)], 1.0, 1.0);
                samples.push((f, u));
            }
            let mut worse = 0;
            let mut gained = 0.0f64;
            for (f, u) in &samples {
                let want = basis_of(*f, *u);
                let (fast_e, wide_e) = match g {
                    G::Reach => {
                        let l = crate::mvar::encode_orientation_nearest(*f, *u);
                        let a = best_reach_axes(*f, *u);
                        let b = wide_r20(*f, *u, l, 6, 40);
                        let e = |c: (bool, u32, u32)| { let (ff, uu) = crate::mvar::decode_orientation(c.0, c.1, c.2); rot_error_deg(&want, ff, uu) };
                        (e(a), e(b))
                    }
                    _ => {
                        let l = crate::h3::mvar::h3_axes_nearest(*f, *u);
                        let a = best_h3_axes(*f, *u);
                        let b = wide_h3(*f, *u, l, 6, 40);
                        let e = |c: (bool, u32, u8)| { let (ff, uu) = crate::h3::mvar::decode_axes(c.0, c.1, c.2); rot_error_deg(&want, ff.to_array(), uu.to_array()) };
                        (e(a), e(b))
                    }
                };
                if fast_e > wide_e + 1e-9 { worse += 1; }
                gained = gained.max(fast_e - wide_e);
            }
            eprintln!("{}: fast search vs wide brute force on {} samples: fast worse on {worse}, worst gap {:.3e} deg", g.label(), samples.len(), gained);
            assert_eq!(worse, 0, "{}: the fast search missed a better code the wide search found", g.label());
        }
    }

    /// Exhaustive: every up code on the three signed faces (all 415^2 cells / 254^2 for Halo 3),
    /// each with its best angle (the angle swept over a 60-code window around the ideal one), on a
    /// handful of samples. The shipped search must equal this optimum.
    #[test]
    fn fast_search_equals_the_exhaustive_optimum_on_samples() {
        fn exhaustive<C: Codec>(c: &C, f: [f32; 3], u: [f32; 3]) -> f64 {
            let want = basis_of(f, u);
            let mut s = Search { c, want, best: None };
            let n = c.cells().len();
            for face in 0..6usize {
                for a in 0..n { for b in 0..n {
                    let q = c.pack(face, a, b);
                    // only faces on the target's side, otherwise the evaluation is just slower
                    let (fd, ud) = c.decode(false, q, 0);
                    let _ = fd;
                    if dot(f64v(ud), want[2]) < 0.9 { continue; }
                    s.best_angle(false, q, want[0], 30);
                } }
            }
            s.best.unwrap().err
        }
        let mut rng = Rng(0xE7);
        for _ in 0..4 {
            let (f, u) = rng.rotation();
            let want = basis_of(f, u);
            let r = best_reach_axes(f, u);
            let (ff, uu) = crate::mvar::decode_orientation(r.0, r.1, r.2);
            let fast = rot_error_deg(&want, ff, uu);
            let ex = exhaustive(r20(), f, u).min(fast);
            assert!(fast <= ex + 1e-9, "Reach: fast {fast} > exhaustive {ex}");
            let r3 = best_h3_axes(f, u);
            let (f3, u3) = crate::h3::mvar::decode_axes(r3.0, r3.1, r3.2);
            let fast3 = rot_error_deg(&want, f3.to_array(), u3.to_array());
            let ex3 = exhaustive(h3c(), f, u).min(fast3);
            assert!(fast3 <= ex3 + 1e-9, "Halo 3: fast {fast3} > exhaustive {ex3}");
        }
    }

    /// (a) encode -> decode -> encode is a FIXED POINT: the decoded pose re-encodes to the same
    /// codes, on random and family poses, and on every REAL stored pose (which must keep its own
    /// bits).
    #[test]
    fn encode_decode_encode_is_a_fixed_point() {
        for g in [G::Reach, G::H4, G::H3] {
            let mut bad = 0;
            let mut n = 0;
            let check = |f: [f32; 3], u: [f32; 3], bad: &mut usize, n: &mut usize| {
                *n += 1;
                match g {
                    G::Reach | G::H4 => {
                        let enc = |f, u| if g == G::Reach { best_reach_axes(f, u) } else { best_h4_axes(f, u) };
                        let c1 = enc(f, u);
                        let (f1, u1) = crate::mvar::decode_orientation(c1.0, c1.1, c1.2);
                        if enc(f1, u1) != c1 { *bad += 1; }
                    }
                    G::H3 => {
                        let c1 = best_h3_axes(f, u);
                        let (f1, u1) = crate::h3::mvar::decode_axes(c1.0, c1.1, c1.2);
                        if best_h3_axes(f1.to_array(), u1.to_array()) != c1 { *bad += 1; }
                    }
                }
            };
            for (f, u) in gate_samples(g) { check(f, u, &mut bad, &mut n); }
            assert_eq!(bad, 0, "{}: {bad} of {n} poses were not a fixed point of encode->decode->encode", g.label());
            // real stored poses keep their codes
            let (_, real) = real_rotations(g, 5000);
            let mut moved = 0;
            for (f, u) in &real {
                let again = match g {
                    G::Reach | G::H4 => { let c = if g == G::Reach { best_reach_axes(*f, *u) } else { best_h4_axes(*f, *u) }; crate::mvar::decode_orientation(c.0, c.1, c.2) }
                    G::H3 => { let c = best_h3_axes(*f, *u); let (a, b) = crate::h3::mvar::decode_axes(c.0, c.1, c.2); (a.to_array(), b.to_array()) }
                };
                if again != (*f, *u) { moved += 1; }
            }
            assert_eq!(moved, 0, "{}: {moved} of {} real stored poses changed when re-encoded", g.label(), real.len());
            eprintln!("{}: fixed point on {n} synthetic poses and {} real stored poses", g.label(), real.len());
        }
    }

    /// Position: old quantiser vs the f32-decode-aware one, on real variants' boxes with the real
    /// objects' positions plus a uniform sub-cell nudge (free movement) -- error in world units,
    /// and the per-axis step the variants impose. NEVER worse, and the stated improvement.
    #[test]
    fn position_choice_never_worse_and_measured() {
        for g in [G::Reach, G::H4, G::H3] {
            // (bounds, positions)
            let mut sets: Vec<([f32; 6], Vec<[f32; 3]>)> = Vec::new();
            match g {
                G::Reach => for f in files_of(Game::Reach, 40) { if let Some(v) = crate::mvar::parse_variant(&f) { sets.push((v.globals.bounds, v.objects.iter().map(|o| o.pos).collect())); } },
                G::H4 => for f in files_of(Game::Halo4, 30).into_iter().chain(files_of(Game::H2A, 20)) { if let Ok(v) = crate::h4::mvar::parse_h4_variant(&f) { sets.push((v.bounds, v.objects.iter().map(|o| o.pos).collect())); } },
                G::H3 => for f in files_of(Game::Halo3, 40) { if let Ok(v) = crate::h3::mvar::parse_h3_variant(&f) { sets.push((v.bounds, v.objects.iter().filter(|o| o.placed).map(|o| o.pos).collect())); } },
            }
            if sets.is_empty() { eprintln!("{}: no variants installed", g.label()); continue; }
            let mut rng = Rng(0x905 + g as u64);
            let (mut old_e, mut new_e, mut steps) = (Vec::new(), Vec::new(), Vec::new());
            let (mut worse, mut better, mut n) = (0usize, 0usize, 0usize);
            for (b, ps) in &sets {
                let ext = [b[1] - b[0], b[3] - b[2], b[5] - b[4]];
                let bits = match g { G::Reach => crate::mvar::compute_axis_bits(21, ext), G::H4 => crate::h4::mvar::h4_axis_bits(21, ext), G::H3 => [16; 3] };
                for k in 0..3 { if ext[k] > 0.0 { steps.push(ext[k] as f64 / (1u64 << bits[k]) as f64); } }
                for p in ps {
                    for k in 0..3 {
                        if !(ext[k] > 0.0) { continue; }
                        let step = ext[k] / (1u64 << bits[k]) as f32;
                        let v = (p[k] + (rng.u() as f32 - 0.5) * step).clamp(b[2 * k], b[2 * k + 1]);
                        let (old_dec, new_dec) = match g {
                            G::Reach => {
                                let scale = ext[k] as f64 / (1u64 << bits[k]) as f64;
                                let q = ((((v as f64 - b[2 * k] as f64) / scale) - 0.5).round() as i64).clamp(0, (1i64 << bits[k]) - 1);
                                let dec = |q: i64| (0.5 + q as f32) * (ext[k] / (1u64 << bits[k]) as f32) + b[2 * k];
                                (dec(q), dec(reach_pos_code(v, b[2 * k], ext[k], bits[k]) as i64))
                            }
                            G::H4 => {
                                let q = crate::h4::mvar::h4_quantize_pos(v, b[2 * k], b[2 * k + 1], bits[k]);
                                let dec = |q: u32| b[2 * k] + q as f32 * step + step * 0.5;
                                (dec(q), dec(h4_pos_code(v, b[2 * k], b[2 * k + 1], bits[k])))
                            }
                            G::H3 => {
                                let dec = |q: u16| crate::h3::mvar::dequantize_real(q as u32, b[2 * k], b[2 * k + 1], 16, false);
                                (dec(crate::h3::mvar::quantize_pos(v, b[2 * k], b[2 * k + 1])), dec(h3_pos_code(v, b[2 * k], b[2 * k + 1])))
                            }
                        };
                        let (eo, en) = ((old_dec as f64 - v as f64).abs(), (new_dec as f64 - v as f64).abs());
                        old_e.push(eo);
                        new_e.push(en);
                        n += 1;
                        if en > eo { worse += 1; }
                        if en < eo { better += 1; }
                    }
                }
            }
            let (o, nw, st) = (dist(&old_e), dist(&new_e), dist(&steps));
            eprintln!("{:<11} POSITION {} axis values over {} variants: step median {:.6} max {:.6} wu | error old med {:.6} p99 {:.6} max {:.6} | new med {:.6} p99 {:.6} max {:.6} wu | strictly better {better}, worse {worse}", g.label(), n, sets.len(), st.0, st.2, o.0, o.1, o.2, nw.0, nw.1, nw.2);
            assert_eq!(worse, 0, "{}: the position choice is worse than the old quantiser on {worse} axis values", g.label());
        }
    }

    /// DECISION DATA (print only): would letting an UPRIGHT Halo 3 object lean by a cell buy a
    /// smaller total error than keeping it level? The shipped encoder keeps level objects level.
    #[test]
    fn measure_what_leaning_would_buy_h3() {
        let mut rng = Rng(0x1EA2);
        let (mut level, mut lean, mut lean_tilt) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..300 {
            let yaw = 90.0 * (rng.u() * 4.0).floor() as f32 + if rng.u() < 0.5 { 0.0 } else { 45.0 };
            let (f, u) = crate::euler::from_euler_deg([0.0, 0.0, yaw], 1.0, 1.0);
            let want = basis_of(f, u);
            let l = best_h3_axes(f, u);
            let (ff, uu) = crate::h3::mvar::decode_axes(l.0, l.1, l.2);
            level.push(rot_error_deg(&want, ff.to_array(), uu.to_array()));
            let w = wide_h3(f, u, (false, 0, 0), 3, 8);
            let (ff, uu) = crate::h3::mvar::decode_axes(w.0, w.1, w.2);
            lean.push(rot_error_deg(&want, ff.to_array(), uu.to_array()));
            lean_tilt.push(crate::quant_error::basis_angle_deg([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], uu.to_array()) as f64);
        }
        let (a, b, c) = (dist(&level), dist(&lean), dist(&lean_tilt));
        eprintln!("H3 upright yaw 0/45/90..: level-kept err med {:.4} p99 {:.4} max {:.4} | allowed to lean err med {:.4} p99 {:.4} max {:.4} (lean tilt med {:.4} max {:.4}) deg", a.0, a.1, a.2, b.0, b.1, b.2, c.0, c.2);
    }
}
