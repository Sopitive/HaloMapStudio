//! The engine's constraint solver for contact constraints: the stream builder
//! (`hkpSimpleContactConstraintAtom` -> jacobian rows, `sub_180D99B00`), the solve loop
//! (`hkSolve`, `sub_180D9AEE0`), the post-solve bookkeeping (`sub_180DA26B0`) and the motion
//! write-back (`sub_180D883A0`) -- ported function by function from `haloreach_debug.dll`, with the
//! retail twin `sub_1808E4120` (solve) checked for the same row switch.
//!
//! # What the solver is (PROVEN from the decompiles; read this before changing anything)
//!
//! It is NOT sequential impulses with a split-impulse push. It is a Gauss-Seidel solver on the
//! bodies' **velocity accumulators** with three engine-specific features that decide how a resting
//! box behaves:
//!
//! 1. **`iterations` (8) sub-steps of `dt / 8`.** One pass over every row per step; gravity
//!    (`gravity * dt/8 * gravityFactor`) is added at the start of every step. The contact rhs is
//!    `-(distance deficit) * (tau/damp) * 8 / dt` and every row's inverse effective mass is
//!    multiplied by `damp` (0.9): a row removes 90 % of its residual per pass.
//! 2. **A running average per body** (`acc+0x40/+0x50`). After each step the body's velocity `v`
//!    becomes `(v - avg) + avg'` with `avg' = avg + (tau/damp) (v - avg)`: the rows see the true
//!    velocity PLUS a memory of earlier steps (an over-relaxation / extrapolation term), and after
//!    the LAST step the memory is dropped again: the velocity handed back to the motion is
//!    `v - avg` while the POSITION integrates the mean velocity `avg * (damp/tau)/8`. This is what
//!    turns the large penetration-recovery velocity of the early steps into a displacement without
//!    leaving it in the velocity (Havok's version of a split impulse).
//! 3. **Slow-object damping**: before the average update, a body whose per-axis
//!    `|w| w_inv + |v| v_inv <= 1` (a deactivation-class threshold) has its velocity multiplied by
//!    `slow_mult` (~0.984) per step, and zeroed once it is below `zero_below`.
//!
//! Contact rows (`0x10` unbounded-above, `0x12` with a max impulse, `0x14` a coupled PAIR of
//! normals solved as a 2x2 LCP, `0x16`/`0x17` the friction pair (with a torsional third row for a
//! multi-point manifold) clamped to a CIRCULAR / spherical cone whose radius is
//! `mu_mean * (sum of the points' last normal impulses) / 8`) are built from the persistent point
//! state: see [`super::manifold`].
//!
//! Restitution IS active: the world's `contactRestingVelocity = FLT_MAX` only disables the
//! alternative "impact solver" branch of the new-contact setup (`sub_180D8DFA0`), whose condition
//! `-FLT_MAX <= approachVelocity || restitution <= 0.3` is then always true; the surviving branch
//! seeds the penetration memory with `1.3 * e * |vn| * dt/8` so the first solves push the pair
//! apart at ~`e * |vn|` (see [`init_new_points`]). The earlier note that Reach rigid bodies do not
//! bounce was wrong.

use super::info::SolverInfo;
use super::manifold::Manifold;
use crate::reach_havok::{HavokBody, MotionType};
use glam::{Quat, Vec3};

/// `FLT_EPSILON`, the guard the engine adds to every effective-mass denominator.
const EPS: f32 = 1.192_092_9e-7;

/// Which kind of accumulator a body is (`acc+0`): `0` dynamic (gets gravity), `1` keyframed / fixed,
/// `2` character (damped, no gravity). `sub_180D87580`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AccKind {
    Dynamic,
    Kinematic,
    Character,
}

/// One body's velocity accumulator (`hkpVelocityAccumulator`, 0x80 bytes).
#[derive(Clone, Copy, Debug)]
struct Acc {
    kind: AccKind,
    lin: Vec3,
    /// Angular velocity in the BODY frame (the solver works with diagonal local inertia).
    ang: Vec3,
    avg_lin: Vec3,
    avg_ang: Vec3,
    inv_mass: f32,
    inv_inertia: Vec3,
    /// `acc+8`: the body's gravity factor.
    gravity: f32,
    class: usize,
    com: Vec3,
    rot: Quat,
}

impl Acc {
    /// The fixed world: zero inverse mass and velocity.
    fn fixed() -> Self {
        Acc {
            kind: AccKind::Kinematic,
            lin: Vec3::ZERO,
            ang: Vec3::ZERO,
            avg_lin: Vec3::ZERO,
            avg_ang: Vec3::ZERO,
            inv_mass: 0.0,
            inv_inertia: Vec3::ZERO,
            gravity: 0.0,
            class: 1,
            com: Vec3::ZERO,
            rot: Quat::IDENTITY,
        }
    }

    fn from_body(b: &HavokBody, class: u8) -> Self {
        let rot = b.q;
        let diag = Vec3::new(b.inv_inertia_local.x_axis.x, b.inv_inertia_local.y_axis.y, b.inv_inertia_local.z_axis.z);
        let (kind, im, ii) = match b.motion {
            MotionType::Fixed => (AccKind::Kinematic, 0.0, Vec3::ZERO),
            MotionType::Keyframed => (AccKind::Kinematic, 0.0, Vec3::ZERO),
            MotionType::Character => (AccKind::Character, b.inv_mass, diag),
            _ => (AccKind::Dynamic, b.inv_mass, diag),
        };
        Acc {
            kind,
            lin: b.v,
            ang: rot.inverse() * b.w,
            avg_lin: Vec3::ZERO,
            avg_ang: Vec3::ZERO,
            inv_mass: im,
            inv_inertia: ii,
            gravity: if kind == AccKind::Dynamic { b.gravity_factor() } else { 0.0 },
            class: class as usize,
            com: b.com,
            rot,
        }
    }
}

/// A body-pair's effective inverse masses for one stream (the mass-changer atom scales them).
#[derive(Clone, Copy, Debug)]
struct PairMass {
    ma: f32,
    ia: Vec3,
    mb: f32,
    ib: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct NormalRow {
    lin: Vec3,
    ang_a: Vec3,
    ang_b: Vec3,
    rhs: f32,
    inv_eff: f32,
    denom: f32,
    acc: f32,
}

#[derive(Clone, Copy, Debug)]
struct FrictionRows {
    d: [Vec3; 2],
    ang_a: [Vec3; 2],
    ang_b: [Vec3; 2],
    rhs: [f32; 2],
    /// `r2C`, `r5C`, offdiagonal `r60`/`r80`.
    c00: f32,
    c11: f32,
    c01: f32,
    limit: f32,
    acc: [f32; 2],
    ratio: f32,
    /// torsion: (angA, angB, rhs, 1/eff (already / radius), radius, acc)
    tors: Option<Torsion>,
}

#[derive(Clone, Copy, Debug)]
struct Torsion {
    ang_a: Vec3,
    ang_b: Vec3,
    rhs: f32,
    inv_eff: f32,
    radius: f32,
    acc: f32,
}

#[derive(Clone, Debug)]
enum Row {
    Normal(NormalRow),
    /// Two normals solved together (`0x14`): rows, plus `r2C`, `r5C`, `r60`.
    Pair(NormalRow, NormalRow, f32, f32, f32),
    Friction(FrictionRows),
}

#[derive(Clone, Debug)]
struct Stream {
    a: usize,
    b: usize,
    pm: PairMass,
    rows: Vec<Row>,
    manifold: usize,
}

fn cross_local(rot: Quat, v: Vec3) -> Vec3 {
    rot.inverse() * v
}

fn safe_rsqrt(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / x.sqrt()
    } else {
        0.0
    }
}

/// `sub_180D8DFA0`: initialise the memory of every NEW point of a manifold from the bodies' current
/// velocities. **PROVEN** formulas (see the module note for the branch that survives):
///
/// ```text
///   vn   = n . ((vA + wA x rA) - (vB + wB x rB))          approach (negative) velocity
///   e    = restitution code / 128
///   rec0 = 1/(imA + imB + 1e-10) * -0.2 * vn * (e + 1)    impulse estimate (the friction cap seed)
///   rec4 = e * vn * (dt/8) * -1.3
///   mem  = e <= 0 ? 0 : rec4 + dist
/// ```
pub(super) fn init_new_points(m: &mut Manifold, bodies: &[HavokBody], info: &SolverInfo) {
    let pa = &bodies[m.a];
    let (ima, vb_lin, vb_ang, comb, imb) = match m.b {
        Some(b) => (pa.inv_mass, bodies[b].v, bodies[b].w, bodies[b].com, bodies[b].inv_mass),
        None => (pa.inv_mass, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.0),
    };
    let (ima, imb) = (ima * m.mass_scale.0, imb * m.mass_scale.1);
    let e = m.material.restitution_code as f32 * 0.007_812_5;
    for p in m.points.iter_mut().filter(|p| p.st.is_new) {
        let va = pa.v + pa.w.cross(p.pos - pa.com);
        let vb = vb_lin + vb_ang.cross(p.pos - comb);
        let vn = p.normal.dot(va - vb);
        p.st.impulse = 1.0 / (ima + imb + 1.0e-10) * -0.2 * vn * (e + 1.0);
        let rec4 = e * vn * info.sub_dt * -1.3;
        p.st.pending = rec4;
        p.st.mem = if e <= 0.0 { 0.0 } else { rec4 + p.dist };
        p.st.is_new = false;
    }
}

/// Build the rows of one manifold (`sub_180D99B00`). Writes the penetration memory back into the
/// points exactly as the engine does at build time.
fn build_stream(mi: usize, m: &mut Manifold, accs: &[Acc], fixed: usize, info: &SolverInfo) -> Stream {
    let (a, b) = (m.a, m.b.unwrap_or(fixed));
    let (aa, ab) = (&accs[a], &accs[b]);
    let pm = PairMass { ma: aa.inv_mass * m.mass_scale.0, ia: aa.inv_inertia * m.mass_scale.0, mb: ab.inv_mass * m.mass_scale.1, ib: ab.inv_inertia * m.mass_scale.1 };
    let n_pts = m.points.len();
    let mut rows: Vec<Row> = Vec::with_capacity(n_pts + 1);
    let mut sum_imp = 0.0f32;
    let mut sum_mu = 0.0f32;
    let mut sum_pos = Vec3::ZERO;
    let mut sum_n = Vec3::ZERO;
    let mu = m.material.friction();
    let dt = info.dt;
    let mut singles: Vec<NormalRow> = Vec::with_capacity(n_pts);
    for p in m.points.iter_mut() {
        let n = p.normal;
        let ra = p.pos - aa.com;
        let rb = p.pos - ab.com;
        let ang_a = cross_local(aa.rot, ra.cross(n));
        let ang_b = cross_local(ab.rot, n.cross(rb));
        let denom = pm.ma + pm.mb + (ang_a * ang_a).dot(pm.ia) + (ang_b * ang_b).dot(pm.ib) + EPS;
        sum_imp += p.st.impulse;
        sum_mu += mu;
        let mem = p.st.mem;
        let v38 = p.dist - mem;
        let v39 = 1.0 * dt;
        let v40 = -p.st.pending - v38;
        let v41 = (-(mem * 0.05)).min(v39);
        let mut v42 = mem + v41;
        let mut v43 = v38 - v41;
        if v40 > v41 + v41 + v39 {
            v42 -= v40;
            v43 += v40;
        }
        p.st.mem = v42.min(-EPS);
        sum_pos += p.pos;
        sum_n += n;
        singles.push(NormalRow { lin: n, ang_a, ang_b, rhs: -v43 * info.rhs_scale, inv_eff: info.damp / denom, denom, acc: 0.0 });
    }
    // solver path 2: the points at odd indices are coupled with their predecessor
    let mut i = 0;
    while i < singles.len() {
        if i + 1 < singles.len() {
            let (r0, r1) = (singles[i], singles[i + 1]);
            let dd = r0.lin * r1.lin;
            let m01 = ((r0.ang_a * r1.ang_a).dot(pm.ia) + (r0.ang_b * r1.ang_b).dot(pm.ib) + (pm.ma + pm.mb) * (dd.x + dd.y + dd.z)) * 0.99;
            let det = info.damp / (r1.denom * r0.denom - m01 * m01);
            rows.push(Row::Pair(r0, r1, det * r1.denom, det * r0.denom, -(det * m01)));
            i += 2;
        } else {
            rows.push(Row::Normal(singles[i]));
            i += 1;
        }
    }
    // friction rows exist only when last tick's impulses were positive
    let inv_n = 1.0 / n_pts as f32;
    let mut cap = sum_mu * sum_imp * inv_n;
    if cap > 0.0 {
        let s2 = sum_n.length_squared();
        let first = m.points[0].normal;
        let n_avg = if inv_n * inv_n * s2 <= 0.9999 {
            if s2 <= 0.1 {
                if 0.9 > first.length_squared() {
                    cap = 0.0;
                    Vec3::Y
                } else {
                    first
                }
            } else {
                sum_n * safe_rsqrt(s2)
            }
        } else {
            first
        };
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        let mut t = axes[m.atom.axis as usize].cross(n_avg);
        if t.length_squared() <= 0.1 {
            let a = n_avg.abs();
            m.atom.axis = if a.y <= a.x { if a.z <= a.y { 2 } else { 1 } } else if a.z > a.x { 0 } else { 2 };
            m.atom.slip[0] = 0.0;
            m.atom.slip[1] = 0.0;
            t = axes[m.atom.axis as usize].cross(n_avg);
        }
        let t1 = t * safe_rsqrt(t.length_squared());
        let t2 = n_avg.cross(t1);
        let pc = sum_pos * inv_n;
        let (ra, rb) = (pc - aa.com, pc - ab.com);
        let dirs = [t2, t1];
        let mut ang_a = [Vec3::ZERO; 2];
        let mut ang_b = [Vec3::ZERO; 2];
        let mut den = [0.0f32; 2];
        for k in 0..2 {
            ang_a[k] = cross_local(aa.rot, ra.cross(dirs[k]));
            ang_b[k] = cross_local(ab.rot, dirs[k].cross(rb));
            den[k] = pm.ma + pm.mb + (ang_a[k] * ang_a[k]).dot(pm.ia) + (ang_b[k] * ang_b[k]).dot(pm.ib) + EPS;
        }
        let dd = dirs[0] * dirs[1];
        let m01 = (ang_b[1] * ang_b[0]).dot(pm.ib) + (ang_a[1] * ang_a[0]).dot(pm.ia) + (pm.mb + pm.ma) * (dd.x + dd.y + dd.z);
        let k = info.damp / (den[0] * den[1] - m01 * m01);
        let mut fr = FrictionRows {
            d: dirs,
            ang_a,
            ang_b,
            rhs: [m.atom.slip[0] * info.anchor_scale, m.atom.slip[1] * info.anchor_scale],
            c00: k * den[1],
            c11: k * den[0],
            c01: -(k * m01),
            limit: cap * info.friction_pass,
            acc: [0.0; 2],
            ratio: 1.0,
            tors: None,
        };
        if n_pts > 1 {
            if m.atom.radius_dirty {
                let mut r = 0.0;
                for p in &m.points {
                    r += (p.pos - pc).length();
                }
                m.atom.radius = r * inv_n;
                m.atom.radius_dirty = false;
            }
            if m.atom.radius >= 1.0e-6 {
                let ta = cross_local(aa.rot, n_avg);
                let tb = cross_local(ab.rot, -n_avg);
                let d = (ta * ta).dot(pm.ia) + (tb * tb).dot(pm.ib) + EPS;
                fr.tors = Some(Torsion { ang_a: ta, ang_b: tb, rhs: m.atom.slip[2] * info.anchor_scale, inv_eff: info.damp / d * (1.0 / m.atom.radius), radius: m.atom.radius, acc: 0.0 });
            }
        }
        rows.push(Row::Friction(fr));
    }
    Stream { a, b, pm, rows, manifold: mi }
}

fn jv(accs: &[Acc], a: usize, b: usize, lin: Vec3, ang_a: Vec3, ang_b: Vec3) -> f32 {
    (accs[a].lin - accs[b].lin).dot(lin) + accs[a].ang.dot(ang_a) + accs[b].ang.dot(ang_b)
}

fn push(accs: &mut [Acc], pm: &PairMass, a: usize, b: usize, lin: Vec3, ang_a: Vec3, ang_b: Vec3, delta: f32) {
    accs[a].lin += lin * (delta * pm.ma);
    accs[b].lin -= lin * (delta * pm.mb);
    accs[a].ang += ang_a * pm.ia * delta;
    accs[b].ang += ang_b * pm.ib * delta;
}

fn push_ang(accs: &mut [Acc], pm: &PairMass, a: usize, b: usize, ang_a: Vec3, ang_b: Vec3, delta: f32) {
    accs[a].ang += ang_a * pm.ia * delta;
    accs[b].ang += ang_b * pm.ib * delta;
}

/// Row 0x10 (the `0x12` variant with a max impulse never occurs: the engine writes `maxImpulse = 0`).
fn solve_normal(accs: &mut [Acc], pm: &PairMass, a: usize, b: usize, r: &mut NormalRow) {
    let res = (r.rhs - jv(accs, a, b, r.lin, r.ang_a, r.ang_b)) * r.inv_eff;
    let d = res.max(0.0 - r.acc);
    push(accs, pm, a, b, r.lin, r.ang_a, r.ang_b, d);
    r.acc += d;
}

/// Row 0x14: two coupled normals (`sub_180D9AEE0` case 0x14).
fn solve_pair(accs: &mut [Acc], pm: &PairMass, a: usize, b: usize, r0: &mut NormalRow, r1: &mut NormalRow, c2c: f32, c5c: f32, c60: f32) {
    let ra = r0.rhs - jv(accs, a, b, r0.lin, r0.ang_a, r0.ang_b);
    let rb = r1.rhs - jv(accs, a, b, r1.lin, r1.ang_a, r1.ang_b);
    let x0 = ra * c2c + rb * c60;
    if x0 <= -r0.acc {
        let x1 = rb * r1.inv_eff;
        if -r1.acc < x1 {
            push(accs, pm, a, b, r1.lin, r1.ang_a, r1.ang_b, x1);
            r1.acc += x1;
            return;
        }
    } else {
        let x1 = rb * c5c + ra * c60;
        if x1 > -r1.acc {
            push(accs, pm, a, b, r0.lin, r0.ang_a, r0.ang_b, x0);
            r0.acc += x0;
            push(accs, pm, a, b, r1.lin, r1.ang_a, r1.ang_b, x1);
            r1.acc += x1;
            return;
        }
    }
    // row 0 alone (LABEL_5)
    let d = (0.0 - r0.acc).max(ra * r0.inv_eff);
    push(accs, pm, a, b, r0.lin, r0.ang_a, r0.ang_b, d);
    r0.acc += d;
}

/// Row 0x16 / 0x17: the friction cone.
fn solve_friction(accs: &mut [Acc], pm: &PairMass, a: usize, b: usize, f: &mut FrictionRows) {
    let r0 = f.rhs[0] - jv(accs, a, b, f.d[0], f.ang_a[0], f.ang_b[0]);
    let r1 = f.rhs[1] - jv(accs, a, b, f.d[1], f.ang_a[1], f.ang_b[1]);
    let mut x0 = r0 * f.c00 + r1 * f.c01;
    let mut x1 = r1 * f.c11 + r0 * f.c01;
    let mut x2 = 0.0;
    if let Some(t) = &f.tors {
        let rt = (t.rhs - (accs[a].ang.dot(t.ang_a) + accs[b].ang.dot(t.ang_b))) * t.inv_eff;
        x2 = rt;
        let n2 = x0 * x0 + x1 * x1 + x2 * x2;
        if n2 <= f.limit * f.limit {
            f.ratio = 1.0;
        } else {
            f.ratio = (f.limit * f.limit / n2).sqrt();
            x0 *= f.ratio;
            x1 *= f.ratio;
            x2 *= f.ratio;
        }
    } else {
        let n2 = x0 * x0 + x1 * x1;
        if n2 <= f.limit * f.limit {
            f.ratio = 1.0;
        } else {
            f.ratio = (f.limit * f.limit / n2).sqrt();
            x0 *= f.ratio;
            x1 *= f.ratio;
        }
    }
    push(accs, pm, a, b, f.d[0], f.ang_a[0], f.ang_b[0], x0);
    f.acc[0] += x0;
    push(accs, pm, a, b, f.d[1], f.ang_a[1], f.ang_b[1], x1);
    f.acc[1] += x1;
    if let Some(t) = &mut f.tors {
        let torque = x2 * t.radius;
        push_ang(accs, pm, a, b, t.ang_a, t.ang_b, torque);
        t.acc += torque;
    }
}

fn finite_box(v: Vec3) -> bool {
    v.abs().max_element() < 1.0e6 && v.is_finite()
}

/// The per-step pass over every accumulator after the rows (`sub_180D9AEE0` tail, cases 0/1/2).
fn step_average(accs: &mut [Acc], info: &SolverInfo, last: bool) {
    for acc in accs.iter_mut() {
        if acc.kind != AccKind::Kinematic {
            if !(finite_box(acc.lin) && finite_box(acc.ang)) {
                acc.lin = Vec3::ZERO;
                acc.ang = Vec3::ZERO;
            }
            let c = &info.classes[acc.class.min(5)];
            let t = acc.ang.abs() * c.ang_inv + acc.lin.abs() * c.lin_inv;
            if t.max_element() <= 1.0 {
                if t.max_element() > c.zero_below {
                    acc.lin *= c.slow_mult;
                    acc.ang *= c.slow_mult;
                } else {
                    acc.lin = Vec3::ZERO;
                    acc.ang = Vec3::ZERO;
                }
            }
        }
        let dv = acc.lin - acc.avg_lin;
        let dw = acc.ang - acc.avg_ang;
        acc.avg_lin += dv * info.avg_gain;
        acc.avg_ang += dw * info.avg_gain;
        if last {
            acc.avg_lin *= info.avg_scale;
            acc.avg_ang *= info.avg_scale;
            acc.lin = dv;
            acc.ang = dw;
        } else {
            let g = if acc.kind == AccKind::Dynamic { info.gravity_sub * acc.gravity } else { Vec3::ZERO };
            acc.lin = dv + acc.avg_lin + g;
            acc.ang = dw + acc.avg_ang;
        }
    }
}

/// `hkSolve` over the whole stream.
fn run_solver(accs: &mut [Acc], streams: &mut [Stream], info: &SolverInfo) {
    for acc in accs.iter_mut() {
        if acc.kind == AccKind::Dynamic {
            acc.lin += info.gravity_sub * acc.gravity;
        }
        acc.avg_lin = Vec3::ZERO;
        acc.avg_ang = Vec3::ZERO;
    }
    for step in 0..info.iterations {
        for _ in 0..info.micro_steps {
            for s in streams.iter_mut() {
                let (a, b, pm) = (s.a, s.b, s.pm);
                for row in s.rows.iter_mut() {
                    match row {
                        Row::Normal(r) => solve_normal(accs, &pm, a, b, r),
                        Row::Pair(r0, r1, c2c, c5c, c60) => solve_pair(accs, &pm, a, b, r0, r1, *c2c, *c5c, *c60),
                        Row::Friction(f) => solve_friction(accs, &pm, a, b, f),
                    }
                }
            }
        }
        step_average(accs, info, step + 1 == info.iterations);
    }
}

/// `sub_180DA26B0`: write the impulses and the remaining-separation bookkeeping back into the
/// persistent point / atom memory (read by the NEXT tick's [`build_stream`]).
fn post_solve(streams: &[Stream], accs: &[Acc], manifolds: &mut [Manifold], info: &SolverInfo) {
    let dt = info.dt;
    let v4 = info.sub_dt * (info.damp / info.tau);
    let v7 = info.sub_dt * (info.damp / (info.tau * 0.5));
    for s in streams {
        let (aa, ab) = (&accs[s.a], &accs[s.b]);
        let d_lin = aa.avg_lin - ab.avg_lin;
        let jd = |lin: Vec3, ang_a: Vec3, ang_b: Vec3| d_lin.dot(lin) + aa.avg_ang.dot(ang_a) + ab.avg_ang.dot(ang_b);
        let m = &mut manifolds[s.manifold];
        let mut pi = 0;
        for row in &s.rows {
            match row {
                Row::Normal(r) => {
                    m.points[pi].st.impulse = r.acc;
                    m.points[pi].st.pending = v4 * r.rhs - jd(r.lin, r.ang_a, r.ang_b) * dt;
                    pi += 1;
                }
                Row::Pair(r0, r1, ..) => {
                    for r in [r0, r1] {
                        m.points[pi].st.impulse = r.acc;
                        m.points[pi].st.pending = v4 * r.rhs - jd(r.lin, r.ang_a, r.ang_b) * dt;
                        pi += 1;
                    }
                }
                Row::Friction(f) => {
                    let slip0 = (v7 * f.rhs[0] - jd(f.d[0], f.ang_a[0], f.ang_b[0]) * dt) * f.ratio;
                    let slip1 = (v7 * f.rhs[1] - jd(f.d[1], f.ang_a[1], f.ang_b[1]) * dt) * f.ratio;
                    m.atom.slip[0] = slip0;
                    m.atom.slip[1] = slip1;
                    m.atom.impulse[0] = f.acc[0];
                    m.atom.impulse[1] = f.acc[1];
                    if let Some(t) = &f.tors {
                        m.atom.slip[2] = (v7 * t.rhs - (aa.avg_ang.dot(t.ang_a) + ab.avg_ang.dot(t.ang_b)) * dt) * f.ratio;
                        m.atom.impulse[2] = t.acc;
                    }
                }
            }
        }
    }
}

/// Per-body state outside [`HavokBody`] that the solver and the deactivation need.
#[derive(Clone, Debug)]
pub struct BodyAux {
    /// `hkMotionState.deactivationClass` (0 invalid, 1 off, 2 low, 3 medium, 4 high, 5 aggressive).
    pub class: u8,
    /// `hkMotionState.objectRadius` (the bounding radius of the body's shapes); only
    /// `min(1, radius)` enters the sleep windows' energy figure.
    pub radius: f32,
    pub sleep: super::sleep::SleepState,
}

impl BodyAux {
    pub fn new(class: u8) -> Self {
        BodyAux { class, radius: 1.0, sleep: super::sleep::SleepState::default() }
    }
}

/// What a step reports.
#[derive(Clone, Debug, Default)]
pub struct StepReport {
    /// Islands (groups of bodies) that passed the deactivation test this tick; the engine puts
    /// them on the dirty list and executes the request at the start of the next integrate.
    pub deactivate: Vec<Vec<usize>>,
    pub streams: usize,
    pub rows: usize,
}

/// Rotate `q` by the world-frame angular velocity `w` over `dt`, with the engine's step-angle
/// clamp and its quaternion polynomial (`sub_180D883A0`, identical to [`HavokBody::step_psi`]).
pub(super) fn rotate_by_pub(q: Quat, w: Vec3, dt: f32, max_ang: f32) -> Quat {
    rotate_by(q, w, dt, max_ang)
}

fn rotate_by(q: Quat, w: Vec3, dt: f32, max_ang: f32) -> Quat {
    let mut h = w * (dt * 0.5);
    let step_cap = (dt * max_ang).min(0.9);
    let mut s = h.length_squared() * 0.405_284_703;
    if step_cap * step_cap < s {
        let k = step_cap / s.sqrt();
        h *= k;
        s = step_cap * step_cap;
    }
    let wq = 1.0 - s * 0.822_947_979 - s * s * 0.130_529_001 - s * 0.044_408_001 * s * s;
    let dq = Quat::from_xyzw(h.x, h.y, h.z, wq);
    let n = (dq * q).normalize();
    if n.is_finite() {
        n
    } else {
        q
    }
}

fn clamp_len(v: Vec3, cap: f32) -> Vec3 {
    let l2 = v.length_squared();
    if l2.is_nan() {
        Vec3::ZERO
    } else if l2 > cap * cap {
        v * (cap / l2.sqrt())
    } else {
        v
    }
}

/// `sub_180D883A0` for one body that went through the solver.
fn write_back(info: &SolverInfo, b: &mut HavokBody, acc: &Acc) {
    let cap_l = b.max_linear_velocity();
    let cap_a = b.max_angular_velocity();
    match b.motion {
        MotionType::Fixed => {}
        MotionType::Keyframed => {
            b.com += clamp_len(b.v, cap_l) * info.dt;
            b.q = rotate_by(b.q, b.w, info.dt, cap_a);
        }
        _ => {
            let w_world = b.q * acc.ang;
            let avg_w_world = b.q * acc.avg_ang;
            b.v = acc.lin;
            b.w = w_world;
            let mut avg = acc.avg_lin;
            let mut avg_w = avg_w_world;
            if !(finite_box(avg) && finite_box(avg_w)) {
                avg = Vec3::ZERO;
                avg_w = Vec3::ZERO;
            }
            b.com += clamp_len(avg, cap_l) * info.dt;
            b.q = rotate_by(b.q, avg_w, info.dt, cap_a);
            b.v = clamp_len(b.v, cap_l);
            b.w = clamp_len(b.w, cap_a);
        }
    }
}

/// Everything one tick does to a set of awake bodies and their contact manifolds, in the engine's
/// order (`hkpContinuousSimulation::integrate`, `sub_180D72850`):
///
/// 1. damping of every dynamic / character body (`sub_180D87580`): `v *= max(0, 1 - dt * damping)`;
/// 2. new-contact setup ([`init_new_points`]);
/// 3. stream build, solve, post-solve;
/// 4. write-back: velocity = the solver's `v - avg`, position += `dt * mean velocity`;
/// 5. bodies that touch nothing take the free path (`sub_180D86820`: gravity, damping, caps, move);
/// 6. the deactivation windows ([`super::sleep`]), and the island rule: an island whose slowest
///    body has more than 5 inactive windows goes to sleep.
///
/// `awake[i]` false marks a sleeping body: it is not integrated and a manifold touching it treats
/// it as immovable. `links` are the broad-phase pairs that tie bodies into islands.
pub fn step(info: &SolverInfo, bodies: &mut [HavokBody], aux: &mut [BodyAux], awake: &[bool], manifolds: &mut [Manifold], links: &[(usize, usize)]) -> StepReport {
    let n = bodies.len();
    let mut report = StepReport::default();
    let mut touched = vec![false; n];
    for m in manifolds.iter().filter(|m| !m.points.is_empty()) {
        if awake[m.a] {
            touched[m.a] = true;
        }
        if let Some(b) = m.b {
            if awake[b] {
                touched[b] = true;
            }
        }
    }
    // 1. damping of the bodies that go through the solver (a body with no constraint takes the free
    // path below, which damps AFTER gravity)
    for i in 0..n {
        let b = &mut bodies[i];
        if awake[i] && touched[i] && !matches!(b.motion, MotionType::Fixed | MotionType::Keyframed) {
            b.v *= (1.0 - info.dt * b.lin_damping).max(0.0);
            b.w *= (1.0 - info.dt * b.ang_damping).max(0.0);
        }
    }
    // accumulators: awake bodies as they are, sleepers as immovable, plus the fixed world
    let mut accs: Vec<Acc> = (0..n)
        .map(|i| {
            let mut a = Acc::from_body(&bodies[i], aux[i].class);
            if !awake[i] {
                a.inv_mass = 0.0;
                a.inv_inertia = Vec3::ZERO;
                a.lin = Vec3::ZERO;
                a.ang = Vec3::ZERO;
                a.kind = AccKind::Kinematic;
            }
            a
        })
        .collect();
    let fixed = accs.len();
    accs.push(Acc::fixed());
    // 2.
    for m in manifolds.iter_mut().filter(|m| !m.points.is_empty()) {
        init_new_points(m, bodies, info);
    }
    // 3.
    let mut streams: Vec<Stream> = Vec::new();
    for (mi, m) in manifolds.iter_mut().enumerate() {
        if !m.points.is_empty() {
            streams.push(build_stream(mi, m, &accs, fixed, info));
        }
    }
    report.streams = streams.len();
    report.rows = streams.iter().map(|s| s.rows.len()).sum();
    run_solver(&mut accs, &mut streams, info);
    post_solve(&streams, &accs, manifolds, info);
    // 4./5.
    for i in 0..n {
        if !awake[i] {
            continue;
        }
        if touched[i] {
            write_back(info, &mut bodies[i], &accs[i]);
        } else {
            free_step(info, &mut bodies[i]);
        }
    }
    // 6.
    let counts: Vec<u32> = (0..n).map(|i| if awake[i] { super::sleep::tick(info, &mut aux[i], &bodies[i]) } else { u32::MAX }).collect();
    report.deactivate = super::sleep::islands_to_sleep(&counts, awake, links, aux);
    report
}

/// `sub_180D86820` for a body in an island with no constraints: gravity first, then damping
/// (the order differs from the solver path, where damping is applied before the solve).
fn free_step(info: &SolverInfo, b: &mut HavokBody) {
    match b.motion {
        MotionType::Fixed => {}
        MotionType::Keyframed => {
            b.com += clamp_len(b.v, b.max_linear_velocity()) * info.dt;
            b.q = rotate_by(b.q, b.w, info.dt, b.max_angular_velocity());
        }
        MotionType::Character => {
            b.v *= (1.0 - info.dt * b.lin_damping).max(0.0);
            b.w *= (1.0 - info.dt * b.ang_damping).max(0.0);
            b.com += clamp_len(b.v, b.max_linear_velocity()) * info.dt;
            b.q = rotate_by(b.q, b.w, info.dt, b.max_angular_velocity());
        }
        _ => {
            b.v += info.gravity_tick * b.gravity_factor();
            b.v *= (1.0 - info.dt * b.lin_damping).max(0.0);
            b.w *= (1.0 - info.dt * b.ang_damping).max(0.0);
            if !(finite_box(b.v) && finite_box(b.w)) {
                b.v = Vec3::ZERO;
                b.w = Vec3::ZERO;
            }
            b.v = clamp_len(b.v, b.max_linear_velocity());
            b.com += b.v * info.dt;
            let w = b.w;
            b.q = rotate_by(b.q, w, info.dt, b.max_angular_velocity());
            b.w = clamp_len(b.w, b.max_angular_velocity());
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::reach_havok::{HavokBody, MotionType};
    use glam::Mat3;

    const G: f32 = 4.1712594;

    fn info() -> SolverInfo {
        SolverInfo::engine(60.0, Vec3::new(0.0, 0.0, -G))
    }

    /// a body that only translates (zero inverse inertia), mass 10
    fn point_mass(pos: Vec3) -> HavokBody {
        HavokBody::new(MotionType::BoxInertia, 10.0, Mat3::ZERO, pos, Quat::IDENTITY)
    }

    /// The eight sub-steps of an unconstrained dynamic body, derived by hand from the loop in
    /// `sub_180D9AEE0`: with g' = g dt/8 the per-step velocity change is `dv_n = (n+1) g'` for
    /// n = 0..7, the velocity handed back is `8 g' = g dt` and the scaled average is the MEAN of
    /// the eight step velocities, `4.5 g'` (not the final one, `8 g'`, and not the exact-integral
    /// `4 g'`): the displacement of the tick is therefore `4.5 g' dt`.
    #[test]
    fn free_fall_through_the_solver_loop() {
        let info = info();
        let b = point_mass(Vec3::ZERO);
        // class 0 has a 1.19e-7 threshold: the slow-object damping (a fresh fall IS slower than the
        // MEDIUM threshold for its first tick) stays out of the way of the algebra under test
        let mut accs = vec![Acc::from_body(&b, 0)];
        run_solver(&mut accs, &mut [], &info);
        let g1 = G * info.dt / 8.0;
        assert!((accs[0].lin.z + 8.0 * g1).abs() < 1e-6, "final velocity {} want {}", accs[0].lin.z, -8.0 * g1);
        assert!((accs[0].avg_lin.z + 4.5 * g1).abs() < 1e-6, "mean velocity {} want {}", accs[0].avg_lin.z, -4.5 * g1);
    }

    /// The row of one resting point, from the numbers of the builder: a point 0.01 deep with a
    /// fresh memory has `d43 = dist - mem - min(-mem*0.05, dt) = -0.01`, so
    /// `rhs = 0.01 * (tau/damp) * 8 / dt = 0.01 * 160 = 1.6` and, for a body that only translates
    /// (inverse mass 0.1 against the fixed world), `invEff = damp / (0.1 + FLT_EPS) = 9.0`.
    #[test]
    fn builder_row_numbers() {
        let info = info();
        let b = point_mass(Vec3::new(0.0, 0.0, 0.5));
        let accs = vec![Acc::from_body(&b, 3), Acc::fixed()];
        let mut m = Manifold::new(0, None, super::super::manifold::ContactMaterial::mix(0.9, 0.0, 0.9, 0.0));
        m.sync(&[super::super::manifold::ContactPoint::new(7, Vec3::new(0.0, 0.0, 0.0), Vec3::Z, -0.01)]);
        m.points[0].st.is_new = false;
        let s = build_stream(0, &mut m, &accs, 1, &info);
        let Row::Normal(r) = &s.rows[0] else { panic!("one point is one single row") };
        assert!((r.rhs - 1.6).abs() < 1e-4, "rhs {}", r.rhs);
        assert!((r.inv_eff - 9.0).abs() < 1e-3, "invEff {}", r.inv_eff);
        // the memory is rewritten at build time: min(mem + d41, -eps) = -eps for a fresh point
        assert!((m.points[0].st.mem + EPS).abs() < 1e-9, "mem {}", m.points[0].st.mem);
        assert_eq!(s.rows.len(), 1, "no friction row: nothing was pressed last tick");
    }

    /// New-contact seed (`sub_180D8DFA0`): approaching at 2 wu/s, e = 64/128 = 0.5, dt/8 = 1/480:
    /// rec4 = e * vn * sub_dt * -1.3 = 0.5 * -2 / 480 * -1.3 = 0.0027083, mem = rec4 + dist,
    /// rec0 = -0.2 * vn * (e + 1) / (1/10) = 6.0 against the fixed world.
    #[test]
    fn new_point_seed_numbers() {
        let info = info();
        let mut a = point_mass(Vec3::new(0.0, 0.0, 0.5));
        a.v = Vec3::new(0.0, 0.0, -2.0);
        let mut m = Manifold::new(0, None, super::super::manifold::ContactMaterial { friction_code: 40, restitution_code: 64 });
        m.sync(&[super::super::manifold::ContactPoint::new(1, Vec3::new(0.0, 0.0, 0.0), Vec3::Z, 0.01)]);
        init_new_points(&mut m, &[a], &info);
        let p = &m.points[0];
        assert!((p.st.pending - 0.0027083).abs() < 1e-6, "rec4 {}", p.st.pending);
        assert!((p.st.mem - (0.0027083 + 0.01)).abs() < 1e-6, "mem {}", p.st.mem);
        assert!((p.st.impulse - 6.0).abs() < 1e-3, "rec0 {}", p.st.impulse);
        assert!(!p.st.is_new);
    }

    /// Friction cap: `mu_mean * sum(last normal impulses) / 8`.
    #[test]
    fn friction_cap_numbers() {
        let info = info();
        let b = point_mass(Vec3::new(0.0, 0.0, 0.5));
        let accs = vec![Acc::from_body(&b, 3), Acc::fixed()];
        let mat = super::super::manifold::ContactMaterial::mix(0.5, 0.0, 0.5, 0.0);
        let mut m = Manifold::new(0, None, mat);
        m.sync(&[super::super::manifold::ContactPoint::new(1, Vec3::ZERO, Vec3::Z, 0.0)]);
        m.points[0].st.is_new = false;
        m.points[0].st.impulse = 2.0;
        let s = build_stream(0, &mut m, &accs, 1, &info);
        let Some(Row::Friction(f)) = s.rows.last() else { panic!("a pressed point gets a friction row") };
        let want = mat.friction() * 2.0 / 8.0;
        assert!((f.limit - want).abs() < 1e-6, "limit {} want {}", f.limit, want);
        // and the two tangents are perpendicular to the normal and to each other
        assert!(f.d[0].dot(Vec3::Z).abs() < 1e-6 && f.d[1].dot(Vec3::Z).abs() < 1e-6 && f.d[0].dot(f.d[1]).abs() < 1e-6);
    }

    /// Slow-object damping (`sub_180D9F030`): with a class MEDIUM body (lin weight 14.10, zero
    /// limit 0.01, multiplier 0.97647) a velocity of 0.0004 wu/s (weighted 0.0056 <= 0.01) is
    /// snapped to zero, one of 0.02 (0.28) is only damped, one of 0.2 (2.8 > 1) is left alone.
    #[test]
    fn slow_objects_are_damped_then_zeroed() {
        let zero_g = SolverInfo::engine(60.0, Vec3::new(0.0, 0.0, -G));
        let run = |vx: f32| -> f32 {
            let mut b = point_mass(Vec3::ZERO);
            b.v = Vec3::new(vx, 0.0, 0.0);
            b.set_gravity_factor(0.0); // the damping alone, no fall
            let mut accs = vec![Acc::from_body(&b, 3)];
            run_solver(&mut accs, &mut [], &zero_g);
            accs[0].lin.x
        };
        assert_eq!(run(0.0004), 0.0, "below the zero limit");
        let damped = run(0.02);
        assert!(damped > 0.0 && damped < 0.02 * 0.99, "damped: {damped}");
        let fast = run(0.2);
        assert!((fast - 0.2).abs() < 1e-6, "untouched: {fast}");
    }
}
