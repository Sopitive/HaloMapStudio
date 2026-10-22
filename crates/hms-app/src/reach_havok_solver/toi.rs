//! Continuous collision: the quality-pair table, the time-of-impact search and the one-shot impulse
//! of `hkpContinuousSimulation` (`03-continuous-toi.md`; `sub_180D0C1D0`, `sub_180D2C670`,
//! `sub_180E2AAD0`, `sub_180D7CB80`, `sub_180D7E340`).
//!
//! # What the engine does (PROVEN)
//!
//! Every pair has a quality record picked by `matrix[qualityA][qualityB]`. Only records 2..6 are
//! "continuous". Reach creates FIXED(0), KEYFRAMED(1), DEBRIS(2), CRITICAL(5) and CHARACTER(8)
//! bodies, so: prop vs terrain = record 5 (N = 20), prop vs prop = record 3 (N = 3), CHARACTER pairs
//! = record 6, DEBRIS pairs = record 1 (discrete only: they never get a TOI and are the ones that
//! tunnel). After `integrate` has moved the bodies, the agent predicts the end-of-step separation
//! `v16 = dist0 + n . dispA_rel - angleBound`; if it is below the TOI trigger depth
//! `depth1 = min(d f6, d f7 + dist0)` (`d` = the smaller allowed penetration depth, `f6 = -0.5`
//! or `-0.4`) a conservative-advancement search finds the first time the pair reaches
//! `depth2 = min(d f10, d f11 + dist0)` (`-0.25 d` / `-0.2 d`). At the event the bodies are stepped
//! back to the event time, a one-shot impulse changes the normal velocity to
//! `-(1 + e) vn + 0.1`, and both are integrated from the event time to the end of the step.
//!
//! The "TOI solver" for neighbours that the impulse leaves violated (tau 0.5, damping 1.4, one
//! step over a growing island) is NOT ported: the next PSI solve resolves what remains.

use crate::reach_havok::QualityType;
use glam::{Quat, Vec3};

/// `hkpCollisionQualityInfo` for one pair (0x40 B in the engine).
#[derive(Clone, Copy, Debug)]
pub struct ToiRecord {
    pub index: u8,
    /// `useContinuous`
    pub continuous: bool,
    pub f6: f32,
    pub f7: f32,
    pub f8: f32,
    pub f9: f32,
    pub f10: f32,
    pub f11: f32,
    pub f12: f32,
    pub f14: f32,
}

/// `matrix[qA][qB]` of `sub_180D0C1D0` (symmetric).
const MATRIX: [[u8; 10]; 10] = [
    [0, 0, 1, 2, 4, 5, 4, 1, 6, 3],
    [0, 0, 1, 1, 3, 3, 3, 1, 6, 3],
    [1, 1, 1, 1, 1, 1, 3, 1, 1, 1],
    [2, 1, 1, 1, 1, 1, 3, 1, 1, 1],
    [4, 3, 1, 1, 1, 3, 3, 1, 3, 3],
    [5, 3, 1, 1, 3, 3, 3, 1, 6, 3],
    [4, 3, 3, 3, 3, 3, 3, 1, 3, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [6, 6, 1, 1, 3, 6, 3, 1, 6, 6],
    [3, 3, 1, 1, 3, 3, 3, 1, 6, 3],
];

/// The record index the engine uses for a pair of qualities (0 = the pair makes no agent).
pub fn record_index(a: QualityType, b: QualityType) -> u8 {
    let (ia, ib) = (a as i32, b as i32);
    if !(0..10).contains(&ia) || !(0..10).contains(&ib) {
        return 1;
    }
    MATRIX[ia as usize][ib as usize]
}

impl ToiRecord {
    /// Build the record the world ctor builds for `index` (1..6): `max_lin_vel` is
    /// `expectedMaxLinearVelocity` (60). Records 0 and 1 are not continuous.
    pub fn engine(index: u8, max_lin_vel: f32) -> ToiRecord {
        let none = ToiRecord { index, continuous: false, f6: -5.0e19, f7: -5.0e19, f8: 1.0, f9: 1.0, f10: -1.0e19, f11: -1.0e19, f12: 5.0e18, f14: 1.0 };
        let simple = |n: f32, f14_numer: f32| {
            let f7 = -0.5 / (n - 1.0);
            let f11 = -0.35 / (n - 1.0);
            ToiRecord {
                index,
                continuous: true,
                f6: -0.5,
                f7,
                f8: 5.0 / max_lin_vel,
                f9: 0.005 / max_lin_vel,
                f10: -0.25,
                f11,
                f12: -0.3 * f11,
                f14: f14_numer / (n - 1.0) / max_lin_vel,
            }
        };
        let forced = || {
            let n = 20.0;
            ToiRecord {
                index,
                continuous: true,
                f6: -0.4,
                f7: -0.6 / (n - 1.0),
                f8: 5.0 / max_lin_vel,
                f9: 0.005 / max_lin_vel,
                f10: -0.2,
                f11: -0.42 / (n - 1.0),
                f12: 0.126 / (n - 1.0),
                f14: 0.6 / (n - 1.0) / max_lin_vel,
            }
        };
        match index {
            2 | 3 => simple(3.0, 1.0),
            4 => simple(4.0, 0.5),
            5 | 6 => forced(),
            _ => none,
        }
    }
}

/// The result of a successful search.
#[derive(Clone, Copy, Debug)]
pub struct Toi {
    /// Event time as a fraction of the step (0..1).
    pub t: f32,
    pub normal: Vec3,
    pub pos: Vec3,
    /// Closing speed at the event, wu/s (negative = closing).
    pub sep_vel: f32,
    pub gap: f32,
}

/// Interpolated pose of a swept body.
#[derive(Clone, Copy, Debug)]
pub struct Swept {
    pub com0: Vec3,
    pub com1: Vec3,
    pub q0: Quat,
    pub q1: Quat,
}

impl Swept {
    pub fn at(&self, t: f32) -> (Vec3, Quat) {
        (self.com0.lerp(self.com1, t), self.q0.slerp(self.q1, t))
    }
    /// Rotation angle over the step (the `deltaAngle.w` bound).
    pub fn angle(&self) -> f32 {
        2.0 * self.q0.dot(self.q1).abs().min(1.0).acos()
    }
}

/// Everything the search needs about a pair.
pub struct PairQuery<'a> {
    /// Signed gap, normal (b into a) and a point at fraction `t` of the step.
    pub gap_at: &'a dyn Fn(f32) -> Option<(f32, Vec3, Vec3)>,
    /// Displacement of A relative to B over the whole step (world), for the closing bound.
    pub rel_disp: Vec3,
    /// Surface displacement due to rotation over the step (`angle * radius`, both bodies).
    pub ang_bound: f32,
    /// Smaller allowed penetration depth of the pair (`d`).
    pub d: f32,
    pub dt: f32,
}

/// The search of `sub_180E2AAD0`: conservative advancement to the trigger depth, then a clamped
/// secant refinement (at most 10 iterations, factor clamped to [0.1, 0.9]) to the target depth.
/// `dist0` is the gap at the start of the step. Returns `None` when the pair cannot reach the
/// trigger depth this step or the event would fall in the last `f14` seconds.
pub fn find_toi(q: &PairQuery, rec: &ToiRecord, dist0: f32) -> Option<Toi> {
    let d = q.d;
    let depth1 = (d * rec.f6).min(d * rec.f7 + dist0);
    let depth2 = (d * rec.f10).min(d * rec.f11 + dist0);
    let tol_band = (d * rec.f12).abs();
    // INFERRED: the engine reads `max(d f8, f9) * invDt` as a floor of the advance, which for
    // `d f8 = 0.0133 s` would be 0.8 of a step and skip the whole approach; only `f9` is used here
    let min_step = (rec.f9 / q.dt).min(1.0);
    // v16 >= dist0 - |relMove| - angBound and depth1 <= d f7 + dist0: a pair that does not move by
    // more than -d f7 this step can never trigger (cheap early-out before any GJK)
    if q.rel_disp.length() + q.ang_bound <= -d * rec.f7 {
        return None;
    }
    let (g0, n0, p0) = (q.gap_at)(0.0)?;
    // predicted end-of-step separation
    let v16 = g0 + n0.dot(q.rel_disp) - q.ang_bound;
    if v16 >= depth1 {
        return None;
    }
    let mut t_lo = 0.0f32;
    let (mut g_lo, mut n, mut p) = (g0, n0, p0);
    let mut t = 0.0f32;
    let mut guard = 0;
    // phase 1: advance while above the target
    let mut hit: Option<(f32, f32)> = if g0 <= depth2 { Some((0.0, g0)) } else { None };
    while hit.is_none() && guard < 64 {
        guard += 1;
        let approach = (-n.dot(q.rel_disp) + q.ang_bound).max(0.0);
        if approach <= 1.0e-9 || g_lo - approach * (1.0 - t) > depth1 {
            return None;
        }
        let adv = (g_lo - depth1) / approach;
        let nt = (t + adv.max(min_step)).min(1.0);
        let Some((g, nn, pp)) = (q.gap_at)(nt) else { return None };
        if g <= depth2 {
            // bracket found: [t, nt]: above the target at t, at or below it at nt
            hit = Some((nt, g));
            t_lo = t;
            n = nn;
            p = pp;
            break;
        }
        t = nt;
        g_lo = g;
        n = nn;
        p = pp;
        if nt >= 1.0 {
            return None;
        }
    }
    let (t_hi0, g_hi0) = hit?;
    let (mut lo, mut hi) = ((t_lo, g_lo), (t_hi0, g_hi0));
    if t_hi0 > 0.0 {
        // phase 2: clamped secant between the sample above the target and the one at or below it
        for _ in 0..10 {
            if (hi.1 - depth2).abs() < tol_band {
                break;
            }
            let denom = lo.1 - hi.1;
            let f = if denom.abs() > 1.0e-9 { ((lo.1 - depth2) / denom).clamp(0.1, 0.9) } else { 0.5 };
            let tm = lo.0 + (hi.0 - lo.0) * f;
            let Some((g, nn, pp)) = (q.gap_at)(tm) else { break };
            n = nn;
            p = pp;
            if g > depth2 {
                lo = (tm, g);
            } else {
                hi = (tm, g);
            }
        }
        if let Some((_, nn, pp)) = (q.gap_at)(hi.0) {
            n = nn;
            p = pp;
        }
    }
    let t_event = (hi.0 * q.dt).max(rec.f14) / q.dt;
    if t_event * q.dt >= q.dt - rec.f14 {
        return None;
    }
    // closing speed at the event: the secant slope of the gap across the final bracket (the
    // engine's `s`, `sepVel = s * invDt`), never faster than the conservative bound
    let bound = (n.dot(q.rel_disp) - q.ang_bound) / q.dt;
    let slope = if hi.0 > lo.0 { (hi.1 - lo.1) / ((hi.0 - lo.0) * q.dt) } else { bound };
    let sep_vel = slope.max(bound).min(0.0);
    Some(Toi { t: t_event, normal: n, pos: p, sep_vel, gap: hi.1 })
}

/// What the one-shot impulse needs about one body (`None` for the fixed world).
pub struct ImpulseBody<'a> {
    pub com: Vec3,
    pub v: &'a mut Vec3,
    pub w: &'a mut Vec3,
    pub inv_mass: f32,
    /// World-frame inverse inertia tensor.
    pub inv_inertia: glam::Mat3,
}

fn skew(a: Vec3) -> glam::Mat3 {
    glam::Mat3::from_cols(Vec3::new(0.0, a.z, -a.y), Vec3::new(-a.z, 0.0, a.x), Vec3::new(a.y, -a.x, 0.0))
}

/// The one-shot TOI response (`sub_180D7E340` -> `sub_180DA3960` / `sub_180DA3FC0`, `03` sec 6):
/// the normal relative velocity becomes `-(1 + e) vn + 0.1` (`vn` capped at the event's closing
/// speed), tangential velocity is preserved while the impulse stays inside the friction cone,
/// otherwise Coulomb-clipped, and a "rotate normal" share (0.2) of the remaining slide speed is added
/// to the normal impulse. Returns true when an impulse was applied.
pub fn toi_impulse(a: &mut ImpulseBody, mut b: Option<&mut ImpulseBody>, pos: Vec3, n: Vec3, sep_vel: f32, mu: f32, e: f32, rotate_normal: f32) -> bool {
    let ra = pos - a.com;
    let rb = b.as_ref().map_or(Vec3::ZERO, |b| pos - b.com);
    let va = *a.v + a.w.cross(ra);
    let vb = b.as_ref().map_or(Vec3::ZERO, |b| *b.v + b.w.cross(rb));
    let vrel = va - vb;
    let mut vn = vrel.dot(n);
    if vn > sep_vel {
        vn = sep_vel;
        if sep_vel >= 0.1 {
            return false;
        }
    }
    if vn >= 0.1 {
        return false; // already separating at least as fast as the bias
    }
    let target = (-vn) * (1.0 + e) + 0.1;
    // K = (imA + imB) I - [rA] IA [rA] - [rB] IB [rB]   (world frame)
    let mut k = glam::Mat3::IDENTITY * (a.inv_mass + b.as_ref().map_or(0.0, |b| b.inv_mass));
    k -= skew(ra) * a.inv_inertia * skew(ra);
    if let Some(bb) = b.as_ref() {
        k -= skew(rb) * bb.inv_inertia * skew(rb);
    }
    let knn = n.dot(k * n);
    if knn <= 1.0e-12 {
        return false;
    }
    let vt = vrel - n * vrel.dot(n);
    let mut j = k.inverse() * (n * target);
    let jn = j.dot(n);
    let jt = j - n * jn;
    if !(jn >= 0.0 && jt.length_squared() <= mu * mu * jn * jn) {
        let jn = target / knn;
        let t_dir = if vt.length_squared() > 1.0e-12 { -vt.normalize() } else { Vec3::ZERO };
        let ktt = t_dir.dot(k * t_dir).max(1.0e-12);
        let jt_mag = (vt.length() / ktt).min(mu * jn);
        let after = (vt.length() - jt_mag * ktt).max(0.0);
        let boost = rotate_normal * vt.length().min(after) / knn;
        j = n * (jn + boost) + t_dir * jt_mag;
    }
    *a.v += j * a.inv_mass;
    *a.w += a.inv_inertia * ra.cross(j);
    if let Some(bb) = b.as_mut() {
        *bb.v -= j * bb.inv_mass;
        *bb.w -= bb.inv_inertia * rb.cross(j);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The records the world ctor builds, by the formulas of `03` 2.3 with N = 3 / 20 and
    /// maxLinVel = 60 (numbers worked out by hand).
    #[test]
    fn record_numbers() {
        let r3 = ToiRecord::engine(3, 60.0);
        assert!(r3.continuous && r3.f6 == -0.5 && (r3.f7 + 0.25).abs() < 1e-7 && r3.f10 == -0.25 && (r3.f11 + 0.175).abs() < 1e-7);
        assert!((r3.f12 - 0.0525).abs() < 1e-7 && (r3.f14 - 1.0 / 2.0 / 60.0).abs() < 1e-7, "{r3:?}");
        let r5 = ToiRecord::engine(5, 60.0);
        assert!(r5.f6 == -0.4 && (r5.f7 + 0.6 / 19.0).abs() < 1e-7 && r5.f10 == -0.2 && (r5.f11 + 0.42 / 19.0).abs() < 1e-7);
        assert!((r5.f14 - 0.6 / 19.0 / 60.0).abs() < 1e-8);
        assert!(!ToiRecord::engine(1, 60.0).continuous);
    }

    /// The pair matrix (`03` 2.2): prop vs terrain = 5, prop vs prop = 3, character = 6, debris = 1,
    /// fixed vs keyframed makes no agent.
    #[test]
    fn matrix_entries() {
        use QualityType::*;
        assert_eq!(record_index(Critical, Fixed), 5);
        assert_eq!(record_index(Fixed, Critical), 5);
        assert_eq!(record_index(Critical, Critical), 3);
        assert_eq!(record_index(Critical, Character), 6);
        assert_eq!(record_index(Debris, Critical), 1);
        assert_eq!(record_index(Debris, Fixed), 1);
        assert_eq!(record_index(Fixed, Keyframed), 0);
        assert_eq!(record_index(Keyframed, Critical), 3);
    }

    /// A ball closing on a plane at 2 wu per step from a gap of 1 wu: the gap is 1 - 2t, the target
    /// depth is `-0.2 d` (record 5, d = 0.2: -0.04), so the event is at t = 0.52.
    #[test]
    fn search_finds_the_target_depth_crossing() {
        let rec = ToiRecord::engine(5, 60.0);
        let gap = |t: f32| Some((1.0 - 2.0 * t, Vec3::Z, Vec3::ZERO));
        let q = PairQuery { gap_at: &gap, rel_disp: Vec3::new(0.0, 0.0, -2.0), ang_bound: 0.0, d: 0.2, dt: 1.0 / 60.0 };
        let toi = find_toi(&q, &rec, 1.0).expect("event");
        assert!((toi.t - 0.52).abs() < 0.002, "t {}", toi.t);
        assert!((toi.sep_vel + 120.0).abs() < 1.0, "closing 2 wu per 1/60 s = -120 wu/s, got {}", toi.sep_vel);
        // too slow to reach the trigger depth (-0.4 d = -0.08) this step: no event
        let slow = PairQuery { gap_at: &gap, rel_disp: Vec3::new(0.0, 0.0, -0.5), ang_bound: 0.0, d: 0.2, dt: 1.0 / 60.0 };
        assert!(find_toi(&slow, &rec, 1.0).is_none());
        // a DEBRIS-record pair is never continuous
        assert!(!ToiRecord::engine(record_index(QualityType::Debris, QualityType::Fixed), 60.0).continuous);
    }

    /// Head-on impulse against the fixed world: `v_after = e |vn| + 0.1` (the normal velocity change
    /// is `-(1 + e) vn + 0.1`).
    #[test]
    fn impulse_is_one_plus_e_plus_a_bias() {
        for e in [0.0f32, 0.5] {
            let (mut v, mut w) = (Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO);
            let mut body = ImpulseBody { com: Vec3::new(0.0, 0.0, 1.0), v: &mut v, w: &mut w, inv_mass: 0.5, inv_inertia: glam::Mat3::ZERO };
            assert!(toi_impulse(&mut body, None, Vec3::ZERO + Vec3::new(0.0, 0.0, 0.0), Vec3::Z, -10.0, 1.0, e, 0.2));
            let want = e * 10.0 + 0.1;
            assert!((v.z - want).abs() < 1.0e-4, "e {e}: {} vs {want}", v.z);
        }
        // already separating faster than 0.1: nothing happens
        let (mut v, mut w) = (Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO);
        let mut body = ImpulseBody { com: Vec3::new(0.0, 0.0, 1.0), v: &mut v, w: &mut w, inv_mass: 0.5, inv_inertia: glam::Mat3::ZERO };
        assert!(!toi_impulse(&mut body, None, Vec3::ZERO, Vec3::Z, 3.0, 1.0, 0.0, 0.2));
    }
}
