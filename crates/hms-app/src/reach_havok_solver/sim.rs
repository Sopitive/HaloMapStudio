//! A complete rigid-body world around the engine's solver: broad phase, narrow phase with the
//! engine's contact admission rules, time of impact, manifold persistence, wake-on-touch and sleep.
//! It is the unit the gates drive and the unit the play-mode prop world can adopt as-is (see the
//! wiring notes in `reach_havok_solver.rs`).
//!
//! One [`Sim::tick`] is one engine step in the engine's order (`hkpContinuousSimulation::
//! stepDeltaTime`, `03-continuous-toi.md` 1): the deferred sleep requests run, then
//! `integrate` (the solver, using the contacts found LAST tick), then `collide` (new poses: contact
//! admission, time-of-impact search), then `advanceTime` (the TOI events).

use super::info::SolverInfo;
use super::manifold::{Admission, Candidate, ContactMaterial, ContactPoint, Manifold, MAX_POINTS};
use super::poly::{Poly, RawContact};
use super::shape::{collide, signed_gap, LocalShape, WShape};
use super::solve::{step, BodyAux, StepReport};
use super::toi::{find_toi, record_index, toi_impulse, ImpulseBody, PairQuery, Swept, Toi, ToiRecord};
use crate::reach_havok::{HavokBody, MotionType, QualityType};
use glam::{Mat3, Quat, Vec3};
use std::collections::HashMap;

/// One simulated body.
#[derive(Clone, Debug)]
pub struct SimBody {
    pub hav: HavokBody,
    pub aux: BodyAux,
    pub shapes: Vec<LocalShape>,
    pub friction: f32,
    pub restitution: f32,
    pub awake: bool,
    /// `objectQualityType`: CRITICAL for props (TOI), DEBRIS (discrete only), CHARACTER, ...
    pub quality: QualityType,
    /// `allowedPenetrationDepth`: `max(0.4 * minDimension, 0.01)`.
    pub allowed_penetration: f32,
    /// Search for time-of-impact events with THIS body as the mover. False for a body some other law places each tick
    /// (the player's), whose penetration the solver resolves; bodies that hit it still get theirs.
    pub toi: bool,
    com0: Vec3,
    q0: Quat,
}

/// `get_minimum_dimension` of `havok_utilities.cpp` (`05` sec 7): box `2 (min h + r)`, sphere `r`,
/// capsule `2 r`, hull `2 min(aabb half extents) + 2 r`.
pub fn min_dimension(shapes: &[LocalShape]) -> f32 {
    let mut m = f32::MAX;
    for s in shapes {
        m = m.min(match s {
            LocalShape::Hull { verts, radius, .. } => {
                let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for v in verts {
                    mn = mn.min(*v);
                    mx = mx.max(*v);
                }
                (mx - mn).min_element() + 2.0 * radius
            }
            LocalShape::Sphere { radius, .. } => *radius,
            LocalShape::Capsule { radius, .. } => 2.0 * radius,
        });
    }
    m
}

fn bounding_radius(shapes: &[LocalShape]) -> f32 {
    shapes
        .iter()
        .map(|s| match s {
            LocalShape::Hull { verts, radius, .. } => verts.iter().map(|v| v.length()).fold(0.0, f32::max) + radius,
            LocalShape::Sphere { center, radius } => center.length() + radius,
            LocalShape::Capsule { a, b, radius } => a.length().max(b.length()) + radius,
        })
        .fold(0.0, f32::max)
}

impl SimBody {
    /// A prop: CRITICAL quality, deactivation class MEDIUM (`05` sec 6), penetration depth from the shapes.
    pub fn new(hav: HavokBody, shapes: Vec<LocalShape>, friction: f32, restitution: f32) -> Self {
        let mut aux = BodyAux::new(3);
        aux.radius = bounding_radius(&shapes);
        let allowed = crate::reach_havok::allowed_penetration_depth(1.0, min_dimension(&shapes), crate::reach_havok::object_type::SCENERY, false, false);
        let (com0, q0) = (hav.com, hav.q);
        SimBody { hav, aux, shapes, friction, restitution, awake: true, quality: QualityType::Critical, allowed_penetration: allowed, toi: true, com0, q0 }
    }
    fn swept(&self) -> Swept {
        Swept { com0: self.com0, com1: self.hav.com, q0: self.q0, q1: self.hav.q }
    }
    fn dynamic(&self) -> bool {
        !matches!(self.hav.motion, MotionType::Fixed | MotionType::Keyframed)
    }
}

/// A static convex polygon (one BSP collision surface: up to 8 coplanar points; a triangle is the
/// 3-point case) with its convex radius ([`ENV_RADIUS`] = 0.0164 in the engine). The surface has no
/// material of its own: the solver mixes the two BODIES' materials, and the environment is one
/// body ([`StaticSource::material`]).
#[derive(Clone, Debug)]
pub struct StaticPoly {
    pub verts: Vec<Vec3>,
    pub radius: f32,
}

/// `havok_shape_radius` (0.05) * 0.328: the convex radius of every environment polygon and of the
/// phmo boxes / triangles / polyhedra (`01-collision-detection.md` 1.2).
pub const ENV_RADIUS: f32 = 0.0164;
/// The materials the engine mixes (`sub_180D78040` reads the BODIES' `hkpMaterial`): a phmo body is
/// created with friction 1.0 / restitution 0.0 (`05` sec 4.3), the environment body keeps the
/// `hkpRigidBodyCinfo` defaults 0.5 / 0.4. So a prop on the level has `sqrt(1.0 * 0.5) = 0.707`
/// (-> 0.716 in the hkUFloat8 table) and `sqrt(0 * 0.4) = 0` restitution.
pub const PROP_BODY_MATERIAL: (f32, f32) = (1.0, 0.0);
pub const ENV_BODY_MATERIAL: (f32, f32) = (0.5, 0.4);

impl StaticPoly {
    pub fn triangle(a: Vec3, b: Vec3, c: Vec3, radius: f32) -> Self {
        StaticPoly { verts: vec![a, b, c], radius }
    }
    #[cfg(test)]
    pub fn shape_for_test(&self) -> WShape {
        self.shape()
    }
    /// Is `p` in the INTERIOR of the polygon (projected on its plane, at least 1 mm from every
    /// edge)? A contact there has the polygon's FACE as its B feature; one on the boundary has an
    /// edge or vertex (a "potential" contact, `sub_180D45890`).
    fn interior(&self, p: Vec3) -> bool {
        let n = self.verts.len();
        if n < 3 {
            return false;
        }
        let normal = (self.verts[1] - self.verts[0]).cross(self.verts[2] - self.verts[0]).normalize_or_zero();
        let mut sign = 0.0f32;
        for k in 0..n {
            let (a, b) = (self.verts[k], self.verts[(k + 1) % n]);
            let e = b - a;
            let side = normal.cross(e).normalize_or_zero().dot(p - a);
            // inside = to the left of every edge for a counter-clockwise loop (either winding is fine)
            if sign == 0.0 {
                sign = if side >= 0.0 { 1.0 } else { -1.0 };
            }
            if side * sign < 1.0e-3 {
                return false;
            }
        }
        true
    }

    fn shape(&self) -> WShape {
        let n = self.verts.len();
        let l: Vec<usize> = (0..n).collect();
        let mut r = l.clone();
        r.reverse();
        WShape::Poly(Poly::hull(&self.verts, &[l, r], Vec3::ZERO, Quat::IDENTITY, self.radius))
    }
}

/// Where the static polygons come from (the level collision BSP in the game).
pub trait StaticSource {
    fn gather(&self, min: Vec3, max: Vec3, out: &mut Vec<(u32, StaticPoly)>);
    /// The environment body's `(friction, restitution)`.
    fn material(&self) -> (f32, f32) {
        ENV_BODY_MATERIAL
    }
    /// Are the polygons wound OUTWARD (their front is the open side of the solid they bound: the engine's
    /// collision BSP and the object hull shells)? Then a body found BEHIND one is inside the solid and is
    /// sent out the front (`send_out_the_front`) instead of on through it. A bare sheet of unknown winding
    /// (the default) is double-sided.
    fn outward_wound(&self) -> bool {
        false
    }
}

/// A plain polygon list (brute force); the gates use it.
pub struct PolyList {
    pub polys: Vec<StaticPoly>,
    pub material: (f32, f32),
    /// The polygons are wound outward (see [`StaticSource::outward_wound`]).
    pub outward: bool,
}

impl PolyList {
    pub fn new(polys: Vec<StaticPoly>, material: (f32, f32)) -> Self {
        PolyList { polys, material, outward: false }
    }
    /// With the engine's environment material.
    pub fn engine(polys: Vec<StaticPoly>) -> Self {
        PolyList { polys, material: ENV_BODY_MATERIAL, outward: false }
    }
    /// The same polygons, declared outward-wound.
    pub fn outward_wound(mut self) -> Self {
        self.outward = true;
        self
    }
}

impl StaticSource for PolyList {
    fn material(&self) -> (f32, f32) {
        self.material
    }
    fn outward_wound(&self) -> bool {
        self.outward
    }
    fn gather(&self, min: Vec3, max: Vec3, out: &mut Vec<(u32, StaticPoly)>) {
        for (i, t) in self.polys.iter().enumerate() {
            let (mut tmin, mut tmax) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for v in &t.verts {
                tmin = tmin.min(*v);
                tmax = tmax.max(*v);
            }
            if tmax.cmpge(min).all() && tmin.cmple(max).all() {
                out.push((i as u32, t.clone()));
            }
        }
    }
}

/// Collision tolerance of the world (`hkpWorldCinfo::collisionTolerance`, **PROVEN** 0.0328).
pub const COLLISION_TOLERANCE: f32 = 0.0328;
/// `expectedMaxLinearVelocity` (60), which scales the TOI records.
pub const EXPECTED_MAX_LINEAR_VELOCITY: f32 = 60.0;
/// `toiCollisionResponseRotateNormal`.
pub const TOI_ROTATE_NORMAL: f32 = 0.2;

#[derive(Default)]
struct PairFresh {
    cands: Vec<Candidate>,
    mat: Option<ContactMaterial>,
}

pub struct Sim {
    pub info: SolverInfo,
    pub bodies: Vec<SimBody>,
    manifolds: HashMap<(usize, Option<usize>), Manifold>,
    pub tolerance: f32,
    /// Apply the 16:1 mass clamp (`havok_utilities.cpp:365`) to body-body contacts. INFERRED to be
    /// active (the function exists, no caller was found): see `02-contact-solver.md` section 8.
    pub mass_clamp: bool,
    /// Run the time-of-impact search (`hkpContinuousSimulation`). Always on in the engine.
    pub continuous: bool,
    pending_sleep: Vec<Vec<usize>>,
    /// broad-phase pairs of the last collide (they make the islands)
    links: Vec<(usize, usize)>,
    /// TOI events handled so far (a statistic the gates use for liveness).
    pub toi_count: usize,
}

impl Sim {
    pub fn new(info: SolverInfo) -> Self {
        Sim { info, bodies: Vec::new(), manifolds: HashMap::new(), tolerance: COLLISION_TOLERANCE, mass_clamp: true, continuous: true, pending_sleep: Vec::new(), links: Vec::new(), toi_count: 0 }
    }

    pub fn add(&mut self, b: SimBody) -> usize {
        self.bodies.push(b);
        self.bodies.len() - 1
    }

    /// Take a body out of the simulation without disturbing the indices of the others: it
    /// stops colliding, stops integrating and its manifolds disappear on the next tick.
    pub fn retire(&mut self, i: usize) {
        let b = &mut self.bodies[i];
        b.shapes.clear();
        b.awake = false;
        b.hav.motion = MotionType::Fixed;
        b.hav.inv_mass = 0.0;
        b.hav.inv_inertia_local = Mat3::ZERO;
        b.hav.v = Vec3::ZERO;
        b.hav.w = Vec3::ZERO;
        b.quality = QualityType::Fixed;
    }

    /// The contact points the body has with anything right now (static world and bodies), as the
    /// last `collide` left them: what the solver will use on the next tick.
    pub fn body_contacts(&self, i: usize) -> Vec<ContactPoint> {
        let mut out = Vec::new();
        for (k, m) in &self.manifolds {
            if k.0 == i || k.1 == Some(i) {
                out.extend(m.points.iter().copied());
            }
        }
        out
    }

    /// Contact points held right now: (against the static world, between bodies).
    pub fn contact_counts(&self) -> (usize, usize) {
        let (mut st, mut pr) = (0, 0);
        for (k, m) in &self.manifolds {
            if k.1.is_none() {
                st += m.points.len();
            } else {
                pr += m.points.len();
            }
        }
        (st, pr)
    }

    /// Activate a sleeping body (the engine's `activate`: the inactivity windows restart).
    pub fn wake(&mut self, i: usize) {
        let b = &mut self.bodies[i];
        if !b.awake {
            b.awake = true;
            b.aux.sleep.reset();
        }
    }

    /// One engine step.
    pub fn tick(&mut self, world: &dyn StaticSource) -> StepReport {
        self.execute_pending_sleep();
        for b in &mut self.bodies {
            b.com0 = b.hav.com;
            b.q0 = b.hav.q;
        }
        let rep = self.solve();
        self.pending_sleep.extend(rep.deactivate.iter().cloned());
        self.collide(world, None);
        if self.continuous {
            self.advance_time(world);
        }
        rep
    }

    /// `sub_180D5CC00`: the dirty list is processed first thing in `integrate`: an island goes to
    /// sleep (velocities zeroed) unless a body of it vetoes (`sub_180D88D20`).
    fn execute_pending_sleep(&mut self) {
        for group in std::mem::take(&mut self.pending_sleep) {
            if group.iter().any(|&i| !self.bodies[i].awake) {
                continue;
            }
            let veto = group.iter().any(|&i| {
                let b = &self.bodies[i];
                b.aux.sleep.vetoes_sleep(b.aux.radius, b.hav.v, b.hav.w)
            });
            // not part of the engine's rule: a body still sunk into something does not sleep there (the
            // saved poses of loose objects overlap their floors); it keeps being pushed out until it clears
            let sunk = group.iter().any(|&i| self.body_contacts(i).iter().any(|c| c.dist < -SLEEP_PENETRATION));
            if veto || sunk {
                continue;
            }
            for i in group {
                let b = &mut self.bodies[i];
                b.awake = false;
                b.hav.v = Vec3::ZERO;
                b.hav.w = Vec3::ZERO;
            }
        }
    }

    fn solve(&mut self) -> StepReport {
        let n = self.bodies.len();
        let mut hav: Vec<HavokBody> = self.bodies.iter().map(|b| b.hav.clone()).collect();
        let mut aux: Vec<BodyAux> = self.bodies.iter().map(|b| b.aux.clone()).collect();
        let awake: Vec<bool> = self.bodies.iter().map(|b| b.awake).collect();
        let mut keys: Vec<(usize, Option<usize>)> = self.manifolds.keys().copied().collect();
        keys.sort();
        let mut ms: Vec<Manifold> = keys.iter().map(|k| self.manifolds.remove(k).unwrap()).collect();
        let rep = step(&self.info, &mut hav, &mut aux, &awake, &mut ms, &self.links);
        for (k, m) in keys.into_iter().zip(ms) {
            self.manifolds.insert(k, m);
        }
        for i in 0..n {
            self.bodies[i].hav = hav[i].clone();
            self.bodies[i].aux = aux[i].clone();
        }
        rep
    }

    fn polys(&self, i: usize) -> Vec<WShape> {
        let b = &self.bodies[i];
        b.shapes.iter().map(|s| s.world(b.hav.com, b.hav.q)).collect()
    }

    fn body_aabb(shapes: &[WShape], margin: f32) -> (Vec3, Vec3) {
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in shapes {
            let (a, b) = p.aabb(margin);
            mn = mn.min(a);
            mx = mx.max(b);
        }
        (mn, mx)
    }

    /// `n . relMove + angBound` of the engine's end-of-step prediction (negative = closing):
    /// the centre-of-mass motion of the pair along the normal over one tick, minus the
    /// bound `dt (|wA| rA + |wB| rB)` on what rotation can add.
    fn step_approach(&self, a: usize, b: Option<usize>, n: Vec3) -> f32 {
        let dt = self.info.dt;
        let (ha, ra) = (&self.bodies[a].hav, self.bodies[a].aux.radius);
        let mut rel = ha.v;
        let mut ang = ha.w.length() * ra;
        if let Some(b) = b {
            let hb = &self.bodies[b].hav;
            rel -= hb.v;
            if self.bodies[b].awake && self.bodies[b].dynamic() {
                ang += hb.w.length() * self.bodies[b].aux.radius;
            }
        }
        (n.dot(rel) - ang) * dt
    }

    /// `collideInternal`: broad phase, PRED_GSK-style candidate points, admission, manifold update.
    fn collide(&mut self, world: &dyn StaticSource, only: Option<&[usize]>) {
        let tol = self.tolerance;
        let n = self.bodies.len();
        let polys: Vec<Vec<WShape>> = (0..n).map(|i| self.polys(i)).collect();
        let boxes: Vec<(Vec3, Vec3)> = polys.iter().map(|ps| Self::body_aabb(ps, tol)).collect();
        let g = self.info.gravity.length();
        let adm = Admission::engine(tol, g, self.info.dt);
        let mut fresh: HashMap<(usize, Option<usize>), PairFresh> = HashMap::new();
        if only.is_none() {
            self.links.clear();
        }
        let touched = |x: usize| only.map_or(true, |o| o.contains(&x));
        let mut statics: Vec<(u32, StaticPoly)> = Vec::new();
        let mut raw: Vec<RawContact> = Vec::new();
        for i in 0..n {
            if !self.bodies[i].awake || !self.bodies[i].dynamic() || !touched(i) {
                continue;
            }
            // --- the static world (quality FIXED) ---
            if record_index(self.bodies[i].quality, QualityType::Fixed) != 0 {
                statics.clear();
                world.gather(boxes[i].0, boxes[i].1, &mut statics);
                let mut pair_c: Vec<(Candidate, f32)> = Vec::new(); // candidate, b_face
                let mut sheets: Vec<Sheet> = Vec::new(); // the polygon each candidate came from
                for (si, sp) in polys[i].iter().enumerate() {
                    for (ti, t) in &statics {
                        raw.clear();
                        if collide(sp, &t.shape(), tol, 8, &mut raw) {
                            let sheet = Sheet::of(t, sp, self.bodies[i].hav.com);
                            for c in raw.drain(..) {
                                let id = (c.id.wrapping_add(ti.wrapping_mul(0x9E37_79B1))) ^ ((si as u32) << 28);
                                let approach = self.step_approach(i, None, c.normal);
                                let cand = Candidate { point: ContactPoint::new(id, c.pos, c.normal, c.dist), edge: c.edge, approach };
                                pair_c.push((cand, if !c.edge && t.interior(c.pos) { 1.0 } else { 0.0 }));
                                sheets.push(sheet);
                            }
                        } else if world.outward_wound() {
                            // A body WHOLLY behind a face but under it (inside a slab or a block, saved a hand's width
                            // into a Forge piece) touches the bare sheet nowhere. If the polygon, moved back to the
                            // body's centre plane, does meet it, the body is inside the solid this face bounds: offer
                            // the exit through the front as a candidate opposed to the front, which
                            // `send_out_the_front` then flips (nearest exit only) like any other.
                            let com = self.bodies[i].hav.com;
                            let sheet = Sheet::of(t, sp, com);
                            let depth = sheet.plane - sheet.body_min;
                            if sheet.front != Vec3::ZERO && !sheet.com_front && depth > 0.0 && depth < 2.0 * self.bodies[i].aux.radius + 0.1 {
                                let shift = sheet.front * sheet.front.dot(com - t.verts[0]);
                                let moved = StaticPoly { verts: t.verts.iter().map(|v| *v + shift).collect(), radius: t.radius };
                                raw.clear();
                                if collide(sp, &moved.shape(), tol, 8, &mut raw) && !raw.is_empty() {
                                    let pos = raw.iter().map(|c| c.pos).sum::<Vec3>() / raw.len() as f32;
                                    let id = ti.wrapping_mul(0x9E37_79B1) ^ 0x4000_0000 ^ ((si as u32) << 28);
                                    let n = -sheet.front;
                                    let approach = self.step_approach(i, None, n);
                                    pair_c.push((Candidate { point: ContactPoint::new(id, pos, n, -depth), edge: false, approach }, 0.0));
                                    sheets.push(sheet);
                                    raw.clear();
                                }
                            }
                        }
                    }
                }
                if world.outward_wound() {
                    send_out_the_front(&mut pair_c, &sheets, self.bodies[i].aux.radius);
                }
                if !pair_c.is_empty() {
                    let (env_f, env_r) = world.material();
                    let radius_a = polys[i].iter().map(|p| match p { WShape::Poly(q) => q.radius, WShape::Sphere { r, .. } | WShape::Capsule { r, .. } => *r }).fold(0.0, f32::max);
                    filter_potentials(&mut pair_c, radius_a, tol);
                    let e = fresh.entry((i, None)).or_default();
                    e.mat = Some(ContactMaterial::mix(self.bodies[i].friction, self.bodies[i].restitution, env_f, env_r));
                    e.cands.extend(pair_c.into_iter().map(|(c, _)| c));
                }
            }
        }
        // --- body pairs: sweep and prune along x, then the narrow phase ---
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| boxes[a].0.x.partial_cmp(&boxes[b].0.x).unwrap_or(std::cmp::Ordering::Equal));
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (k, &a) in order.iter().enumerate() {
            for &b in &order[k + 1..] {
                if boxes[b].0.x > boxes[a].1.x {
                    break;
                }
                if !(boxes[a].0.cmple(boxes[b].1).all() && boxes[a].1.cmpge(boxes[b].0).all()) {
                    continue;
                }
                let live = |x: usize| self.bodies[x].awake && self.bodies[x].dynamic();
                // A is the awake dynamic one (the lower index when both are)
                let pair = match (live(a), live(b)) {
                    (true, true) => (a.min(b), a.max(b)),
                    (true, false) => (a, b),
                    (false, true) => (b, a),
                    (false, false) => continue,
                };
                if touched(pair.0) || touched(pair.1) {
                    pairs.push(pair);
                }
            }
        }
        pairs.sort_unstable();
        for (i, j) in pairs {
            if record_index(self.bodies[i].quality, self.bodies[j].quality) == 0 {
                continue;
            }
            // the broad-phase pair exists: the islands merge, a sleeper wakes (`sub_180D5A280`)
            if self.bodies[j].dynamic() {
                self.links.push((i, j));
                if !self.bodies[j].awake {
                    self.wake(j);
                }
            }
            for (si, sp) in polys[i].iter().enumerate() {
                for (sj, sq) in polys[j].iter().enumerate() {
                    raw.clear();
                    if collide(sp, sq, tol, 8, &mut raw) {
                        let e = fresh.entry((i, Some(j))).or_default();
                        e.mat = Some(ContactMaterial::mix(self.bodies[i].friction, self.bodies[i].restitution, self.bodies[j].friction, self.bodies[j].restitution));
                        for c in raw.drain(..) {
                            let id = c.id ^ ((si as u32) << 12) ^ ((sj as u32) << 20);
                            let approach = self.step_approach(i, Some(j), c.normal);
                            e.cands.push(Candidate { point: ContactPoint::new(id, c.pos, c.normal, c.dist), edge: c.edge, approach });
                        }
                    }
                }
            }
        }
        // manifolds: pairs without candidates this tick lose their points
        let keys: Vec<_> = self.manifolds.keys().copied().collect();
        for k in keys {
            if !fresh.contains_key(&k) && (touched(k.0) || k.1.is_some_and(touched)) {
                self.manifolds.remove(&k);
            }
        }
        for (k, f) in fresh {
            let mat = f.mat.unwrap_or(ContactMaterial { friction_code: 40, restitution_code: 0 });
            let m = self.manifolds.entry(k).or_insert_with(|| Manifold::new(k.0, k.1, mat));
            m.material = mat;
            if let (true, Some(b)) = (self.mass_clamp, k.1) {
                let (ma, mb) = (self.bodies[k.0].hav.inv_mass, self.bodies[b].hav.inv_mass);
                if ma > 0.0 && mb > 0.0 {
                    let (fa, fb) = crate::reach_havok::mass_multipliers(1.0 / ma, 1.0 / mb);
                    m.mass_scale = (1.0 / fa, 1.0 / fb);
                }
            }
            let mut adm = adm;
            adm.max_points = MAX_POINTS * match k.1 {
                None => self.bodies[k.0].shapes.len(),
                Some(b) => self.bodies[k.0].shapes.len().max(self.bodies[b].shapes.len()),
            }.clamp(1, 8);
            m.update(&f.cands, &adm);
            if m.points.is_empty() {
                self.manifolds.remove(&k);
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // advanceTime: time-of-impact events
    // ---------------------------------------------------------------------------------------

    fn shape_gap_at(&self, i: usize, other: Other<'_>, t: f32) -> Option<(f32, Vec3, Vec3)> {
        let (ca, qa) = self.bodies[i].swept().at(t);
        let mut best: Option<(f32, Vec3, Vec3)> = None;
        for sa in &self.bodies[i].shapes {
            let wa = sa.world(ca, qa);
            match other {
                Other::Static(poly) => {
                    if let Some(g) = signed_gap(&wa, &poly.shape()) {
                        if best.map_or(true, |b| g.0 < b.0) {
                            best = Some(g);
                        }
                    }
                }
                Other::Body(j) => {
                    let (cb, qb) = self.bodies[j].swept().at(t);
                    for sb in &self.bodies[j].shapes {
                        if let Some(g) = signed_gap(&wa, &sb.world(cb, qb)) {
                            if best.map_or(true, |b| g.0 < b.0) {
                                best = Some(g);
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// Find the earliest TOI event of body `i` this step (`sub_180D2C670` + `sub_180E2AAD0`).
    fn find_event(&self, i: usize, world: &dyn StaticSource, boxes: &[(Vec3, Vec3)]) -> Option<(Toi, Option<usize>, f32, f32)> {
        let b = &self.bodies[i];
        let dt = self.info.dt;
        let sw = b.swept();
        let (lo, hi) = boxes[i];
        let mut best: Option<(Toi, Option<usize>, f32, f32)> = None;
        let consider = |toi: Toi, partner: Option<usize>, mu: f32, e: f32, best: &mut Option<(Toi, Option<usize>, f32, f32)>| {
            if best.as_ref().map_or(true, |b| toi.t < b.0.t) {
                *best = Some((toi, partner, mu, e));
            }
        };
        // static world
        let rec_s = ToiRecord::engine(record_index(b.quality, QualityType::Fixed), EXPECTED_MAX_LINEAR_VELOCITY);
        // a pair that moves by less than -d f7 this step can never trigger (`find_toi` early-out);
        // checked here too so that quiet bodies do no GJK at all
        let own_move = (sw.com1 - sw.com0).length() + sw.angle() * b.aux.radius;
        if rec_s.continuous && own_move > -b.allowed_penetration * rec_s.f7 {
            let mut polys = Vec::new();
            world.gather(lo, hi, &mut polys);
            for (_, p) in &polys {
                let gap = |t: f32| self.shape_gap_at(i, Other::Static(p), t);
                let Some((dist0, _, _)) = gap(0.0) else { continue };
                // Already deeper in the sheet than the allowed penetration when the step began (a saved pose that
                // overlaps the level): no time of impact, whose impulse would push along the sheet's nearest
                // side -- the BACK of it when the body's centre is behind -- and the solver, which pushes out the
                // front (`send_out_the_front`), recovers it.
                if world.outward_wound() && dist0 < -b.allowed_penetration {
                    continue;
                }
                // ...and no time of impact against a face the body's centre is BEHIND: that face is the far side of
                // a solid the body is inside (or outside of, for the other face of a wall), not something to land on.
                if let Some(sh) = self.bodies[i].shapes.first().filter(|_| world.outward_wound()).map(|s| Sheet::of(p, &s.world(sw.com0, sw.q0), sw.com0)) {
                    if sh.front != Vec3::ZERO && sh.front.dot(sw.com0 - p.verts[0]) < 0.0 {
                        continue;
                    }
                }
                let q = PairQuery { gap_at: &gap, rel_disp: sw.com1 - sw.com0, ang_bound: sw.angle() * b.aux.radius, d: b.allowed_penetration, dt };
                if let Some(toi) = find_toi(&q, &rec_s, dist0) {
                    let (env_f, env_r) = world.material();
                    let mu = ContactMaterial::mix(b.friction, b.restitution, env_f, env_r);
                    consider(toi, None, mu.friction(), mu.restitution(), &mut best);
                }
            }
        }
        // other bodies
        for j in 0..self.bodies.len() {
            if j == i {
                continue;
            }
            let o = &self.bodies[j];
            let rec = ToiRecord::engine(record_index(b.quality, o.quality), EXPECTED_MAX_LINEAR_VELOCITY);
            if !rec.continuous {
                continue;
            }
            let other_move = if o.awake && o.dynamic() { (o.hav.com - o.com0).length() + o.swept().angle() * o.aux.radius } else { 0.0 };
            if own_move + other_move <= -b.allowed_penetration.min(o.allowed_penetration) * rec.f7 {
                continue;
            }
            let (olo, ohi) = boxes[j];
            if !(lo.cmple(ohi).all() && hi.cmpge(olo).all()) {
                continue;
            }
            let moving_other = o.awake && o.dynamic();
            let gap = |t: f32| self.shape_gap_at(i, Other::Body(j), t);
            let Some((dist0, _, _)) = gap(0.0) else { continue };
            let osw = o.swept();
            let rel_disp = (sw.com1 - sw.com0) - if moving_other { osw.com1 - osw.com0 } else { Vec3::ZERO };
            let q = PairQuery { gap_at: &gap, rel_disp, ang_bound: sw.angle() * b.aux.radius + if moving_other { osw.angle() * o.aux.radius } else { 0.0 }, d: b.allowed_penetration.min(o.allowed_penetration), dt };
            if let Some(toi) = find_toi(&q, &rec, dist0) {
                let mu = ContactMaterial::mix(b.friction, b.restitution, o.friction, o.restitution);
                consider(toi, Some(j), mu.friction(), mu.restitution(), &mut best);
            }
        }
        best
    }

    /// Where the TOI impulse acts. The search returns ONE closest point, and a box landing flat reports one of its
    /// four bottom corners: an impulse there spins the box at 30 rad/s and kicks it back up, which no flat landing
    /// does. Every vertex of the body within 2 cm of its lowest one along the contact normal is part of the same
    /// face or edge contact, and the impulse acts at their centroid; a corner, a round shape or a side hit (normal more than 45 degrees off vertical) is left alone.
    /// INFERRED (the engine solves the pair's whole contact manifold at the event time).
    fn patch_centre(&self, i: usize, t: f32, pos: Vec3, n: Vec3) -> Vec3 {
        // only a LANDING (a mostly vertical normal): a body pressed into a wall keeps the single-point response
        if n.z.abs() < 0.7 {
            return pos;
        }
        let polys = self.polys_at(i, t);
        let low = polys
            .iter()
            .filter_map(|w| if let WShape::Poly(p) = w { Some(p.verts.iter().map(|v| n.dot(*v - pos)).fold(f32::MAX, f32::min)) } else { None })
            .fold(f32::MAX, f32::min);
        let (mut sum, mut count) = (Vec3::ZERO, 0);
        for w in &polys {
            if let WShape::Poly(p) = w {
                for v in &p.verts {
                    if n.dot(*v - pos) < low + 0.02 {
                        sum += *v;
                        count += 1;
                    }
                }
            }
        }
        if count >= 2 {
            let c = sum / count as f32;
            c + n * (pos - c).dot(n)
        } else {
            pos
        }
    }

    /// The box body `k` sweeps over this step.
    fn swept_box(&self, k: usize) -> (Vec3, Vec3) {
        let (a0, a1) = (Self::body_aabb(&self.polys_at(k, 0.0), self.tolerance), Self::body_aabb(&self.polys_at(k, 1.0), self.tolerance));
        (a0.0.min(a1.0), a0.1.max(a1.1))
    }

    fn polys_at(&self, i: usize, t: f32) -> Vec<WShape> {
        let (c, q) = self.bodies[i].swept().at(t);
        self.bodies[i].shapes.iter().map(|s| s.world(c, q)).collect()
    }

    /// `advanceTime`: handle the TOI events, earliest first. Each event steps the bodies back to the
    /// event time, applies the one-shot impulse and re-integrates them to the end of the step.
    fn advance_time(&mut self, world: &dyn StaticSource) {
        let mut handled = vec![0u8; self.bodies.len()];
        // one event per round, earliest first, at most 3 per body: the rounds are not capped (a world of many bodies landing on the
        // same tick has many events; capping them let the late ones through). Each body's next event is cached and only the
        // ones an event can have changed are searched again: the bodies it touched, those whose cached partner was one of them,
        // and those whose swept box meets theirs.
        type Event = Option<(Toi, Option<usize>, f32, f32)>;
        let n = self.bodies.len();
        let mut boxes: Vec<(Vec3, Vec3)> = (0..n).map(|k| self.swept_box(k)).collect();
        let mut cache: Vec<Option<Event>> = vec![None; n];
        for _round in 0..(3 * n).max(4) {
            let mut best: Option<(usize, Toi, Option<usize>, f32, f32)> = None;
            for i in 0..n {
                let b = &self.bodies[i];
                if !b.awake || !b.dynamic() || !b.toi || handled[i] >= 3 {
                    continue;
                }
                if cache[i].is_none() {
                    cache[i] = Some(self.find_event(i, world, &boxes));
                }
                if let Some(Some((toi, partner, mu, e))) = cache[i] {
                    if best.as_ref().map_or(true, |x| toi.t < x.1.t) {
                        best = Some((i, toi, partner, mu, e));
                    }
                }
            }
            let Some((i, toi, partner, mu, e)) = best else { break };
            handled[i] += 1;
            self.toi_count += 1;
            let dt = self.info.dt;
            // back-step to the event time
            let touched: Vec<usize> = std::iter::once(i).chain(partner.filter(|&j| self.bodies[j].awake && self.bodies[j].dynamic())).collect();
            for &k in &touched {
                let (c, q) = self.bodies[k].swept().at(toi.t);
                self.bodies[k].hav.com = c;
                self.bodies[k].hav.q = q;
            }
            // impulse
            let toi_pos = self.patch_centre(i, toi.t, toi.pos, toi.normal);
            {
                let (mut va, mut wa) = (self.bodies[i].hav.v, self.bodies[i].hav.w);
                let ha = &self.bodies[i].hav;
                let mut ia = ImpulseBody { com: ha.com, v: &mut va, w: &mut wa, inv_mass: ha.inv_mass, inv_inertia: ha.inv_inertia_world() };
                let applied;
                let (mut vb, mut wb);
                match partner {
                    Some(j) => {
                        let hb = &self.bodies[j].hav;
                        let dynb = self.bodies[j].awake && self.bodies[j].dynamic();
                        vb = hb.v;
                        wb = hb.w;
                        let mut ib = ImpulseBody { com: hb.com, v: &mut vb, w: &mut wb, inv_mass: if dynb { hb.inv_mass } else { 0.0 }, inv_inertia: if dynb { hb.inv_inertia_world() } else { Mat3::ZERO } };
                        applied = toi_impulse(&mut ia, Some(&mut ib), toi_pos, toi.normal, toi.sep_vel, mu, e, TOI_ROTATE_NORMAL);
                    }
                    None => {
                        applied = toi_impulse(&mut ia, None, toi_pos, toi.normal, toi.sep_vel, mu, e, TOI_ROTATE_NORMAL);
                        vb = Vec3::ZERO;
                        wb = Vec3::ZERO;
                    }
                }
                if applied {
                    // A collision with restitution below one does not ADD energy. The response is a single-point impulse sized
                    // by the event's closing speed, which for a tumbling body can exceed the speed the contact point really
                    // has (a corner swinging up while the centre falls): it then throws the body back harder than it came.
                    // The kinetic energy after the event is held to what it was before (plus the 0.1 wu/s separation bias).
                    let ke = |h: &HavokBody, v: Vec3, w: Vec3| -> f32 {
                        let m = if h.inv_mass > 0.0 { 1.0 / h.inv_mass } else { 0.0 };
                        let iw = h.inv_inertia_world();
                        let rot = if iw.determinant().abs() > 1.0e-18 { 0.5 * w.dot(iw.inverse() * w) } else { 0.0 };
                        0.5 * m * v.length_squared() + rot
                    };
                    let (ha, mut before, mut after) = (self.bodies[i].hav.clone(), 0.0f32, 0.0f32);
                    before += ke(&ha, ha.v, ha.w);
                    after += ke(&ha, va, wa);
                    let mut slack = 0.5 * (if ha.inv_mass > 0.0 { 1.0 / ha.inv_mass } else { 0.0 }) * 0.2 * 0.2;
                    let hb = partner.filter(|&j| self.bodies[j].awake && self.bodies[j].dynamic()).map(|j| self.bodies[j].hav.clone());
                    if let Some(hb) = &hb {
                        before += ke(hb, hb.v, hb.w);
                        after += ke(hb, vb, wb);
                        slack += 0.5 * (if hb.inv_mass > 0.0 { 1.0 / hb.inv_mass } else { 0.0 }) * 0.2 * 0.2;
                    }
                    // (only a body against the level that no other body is pressing on: the push of a third body, the player's
                    // above all, brings energy from outside the world)
                    let pushed = self.manifolds.keys().any(|k| (k.0 == i && k.1.is_some()) || k.1 == Some(i));
                    if after > before + slack && partner.is_none() && !pushed {
                        // the largest share of the impulse that keeps the energy (bisection; the energy is convex in the share)
                        let total = |f: f32| -> f32 {
                            let mut e = ke(&ha, ha.v + (va - ha.v) * f, ha.w + (wa - ha.w) * f);
                            if let Some(hb) = &hb {
                                e += ke(hb, hb.v + (vb - hb.v) * f, hb.w + (wb - hb.w) * f);
                            }
                            e
                        };
                        let (mut lo, mut hi) = (0.0f32, 1.0f32);
                        for _ in 0..12 {
                            let mid = 0.5 * (lo + hi);
                            if total(mid) <= before + slack {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        let f = lo;
                        va = ha.v + (va - ha.v) * f;
                        wa = ha.w + (wa - ha.w) * f;
                        if let Some(hb) = &hb {
                            vb = hb.v + (vb - hb.v) * f;
                            wb = hb.w + (wb - hb.w) * f;
                        }
                    }
                    self.bodies[i].hav.v = va;
                    self.bodies[i].hav.w = wa;
                    if let Some(j) = partner {
                        if self.bodies[j].awake && self.bodies[j].dynamic() {
                            self.bodies[j].hav.v = vb;
                            self.bodies[j].hav.w = wb;
                        }
                    }
                }
            }
            // re-integrate from the event time to the end of the step with the post-TOI velocity
            let rem = dt * (1.0 - toi.t);
            for &k in &touched {
                let h = &mut self.bodies[k].hav;
                h.com += h.v * rem;
                h.q = super::solve::rotate_by_pub(h.q, h.w, rem, h.max_angular_velocity());
                // the swept start for any further event of this step is the event pose
                self.bodies[k].com0 = self.bodies[k].hav.com - self.bodies[k].hav.v * rem;
            }
            // collide again for the new poses (of the bodies it touched), and forget the events it can have changed
            let mut moved = touched.clone();
            if let Some(j) = partner {
                if !moved.contains(&j) {
                    moved.push(j);
                }
            }
            self.collide(world, Some(&moved));
            for &k in &moved {
                boxes[k] = self.swept_box(k);
                cache[k] = None;
            }
            for k in 0..n {
                let hit_partner = matches!(cache[k], Some(Some((_, Some(p), _, _))) if moved.contains(&p));
                let near = moved.iter().any(|&m| boxes[k].0.cmple(boxes[m].1).all() && boxes[k].1.cmpge(boxes[m].0).all());
                if hit_partner || near {
                    cache[k] = None;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Other<'a> {
    Static(&'a StaticPoly),
    Body(usize),
}

/// The "potential contact" pass for contacts on a polygon's vertices and edges
/// (`sub_180D45890`, `01-collision-detection.md` 4.4): a contact whose B feature is a polygon vertex
/// or edge is dropped when a deeper face contact already covers it. `radius` = convex radius of
/// the body's shape (`r`), `tol` the collision tolerance.
fn filter_potentials(cands: &mut Vec<(Candidate, f32)>, r: f32, tol: f32) {
    let reps: Vec<Candidate> = cands.iter().filter(|(_, face)| *face > 0.5).map(|(c, _)| *c).collect();
    cands.retain(|(c, face)| {
        if *face > 0.5 {
            return true;
        }
        // every representative here is a committed face contact, so the engine's inner condition
        // (`R.id != 0xFFFF || c >= 0.8 || ...`) is true and only the plane test and the depth test remain
        for rep in &reps {
            let cc = rep.point.normal.dot(c.point.normal);
            let h = rep.point.normal.dot(c.point.pos - rep.point.pos);
            let g = (cc - 1.0) * r + h;
            if g <= r && rep.point.dist - 0.05 * tol < c.point.dist {
                return false;
            }
        }
        true
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(x: f32, dist: f32) -> (Candidate, f32) {
        (Candidate { point: ContactPoint::new(1, Vec3::new(x, 0.0, 0.0), Vec3::Z, dist), edge: false, approach: 0.0 }, 1.0)
    }
    fn pot(x: f32, dist: f32, n: Vec3) -> (Candidate, f32) {
        (Candidate { point: ContactPoint::new(2, Vec3::new(x, 0.0, 0.0), n, dist), edge: false, approach: 0.0 }, 0.0)
    }

    /// `sub_180D45890`: a contact on a polygon's vertex/edge is dropped when a DEEPER committed face
    /// contact's plane (within the radius r = 0.0164 + the point's height) covers it; a deeper
    /// potential, or one far above the plane, survives.
    #[test]
    fn potential_contacts_are_shadowed_by_deeper_face_contacts() {
        let r = 0.0164;
        // face contact at -10 mm; the potential at -5 mm lies on its plane: dropped
        let mut c = vec![rep(0.0, -0.010), pot(0.1, -0.005, Vec3::new(0.2, 0.0, 0.98).normalize())];
        filter_potentials(&mut c, r, 0.0328);
        assert_eq!(c.len(), 1, "the shadowed potential is dropped");
        // the potential is deeper than the face contact: it stays
        let mut c = vec![rep(0.0, -0.010), pot(0.1, -0.020, Vec3::Z)];
        filter_potentials(&mut c, r, 0.0328);
        assert_eq!(c.len(), 2);
        // the potential is 5 cm above the face plane (g = h > r): not covered, it stays
        let mut c = vec![rep(0.0, -0.010), (Candidate { point: ContactPoint::new(3, Vec3::new(0.1, 0.0, 0.05), Vec3::Z, -0.005), edge: false, approach: 0.0 }, 0.0)];
        filter_potentials(&mut c, r, 0.0328);
        assert_eq!(c.len(), 2);
    }
}


/// Deepest overlap (wu) a body may have and still go to sleep.
const SLEEP_PENETRATION: f32 = 0.02;

/// A static polygon as one body saw it: its FRONT normal (the winding, which for the engine's collision BSP and
/// the object hull shells points out of the solid), the plane offset plus its convex radius, and how far
/// the body reaches below that plane.
#[derive(Clone, Copy)]
struct Sheet {
    front: Vec3,
    /// `front . v0 + radius`
    plane: f32,
    /// the body's lowest point along `front`, less its own radius
    body_min: f32,
    /// the body's centre is on the front side of the plane
    com_front: bool,
}

impl Sheet {
    fn of(t: &StaticPoly, body: &WShape, com: Vec3) -> Sheet {
        let v = &t.verts;
        let mut n = Vec3::ZERO;
        for k in 0..v.len() {
            let (p, q) = (v[k], v[(k + 1) % v.len()]);
            n += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
        }
        let front = n.normalize_or_zero();
        let body_min = match body {
            WShape::Poly(p) => p.verts.iter().map(|x| front.dot(*x)).fold(f32::MAX, f32::min) - p.radius,
            WShape::Sphere { c, r } => front.dot(*c) - r,
            WShape::Capsule { a, b, r } => front.dot(*a).min(front.dot(*b)) - r,
        };
        Sheet { front, plane: front.dot(v[0]) + t.radius, body_min, com_front: front.dot(com - v[0]) >= 0.0 }
    }
}

/// **A body that has ended up BEHIND a face is pushed out the FRONT of it.** The engine's polygons are
/// double-sided sheets and its continuous collision keeps a body from ever getting behind one; the saved poses
/// of a map's loose objects do not come with that guarantee (a weapon lying a few cm into a floor, a crate
/// sunk into a wall), and a separating-axis contact on a sheet whose body centre is behind it pushes the
/// body on through. Where a contact's normal opposes the polygon's front, the body is moved out the front
/// instead -- unless the body also touches the front of an OPPOSED face (it is outside a thin slab or wall
/// and this is the solid's far side) -- and only the NEAREST such exit pushes (the far faces of a solid a
/// body is inside are the other side of it). Same rule as `physics_settle::shape_contacts_sided`.
fn send_out_the_front(cands: &mut Vec<(Candidate, f32)>, sheets: &[Sheet], body_radius: f32) {
    debug_assert_eq!(cands.len(), sheets.len());
    let reach = 2.0 * body_radius + 0.1;
    let fronts: Vec<Vec3> = cands.iter().zip(sheets).filter(|(c, s)| s.com_front && c.0.point.dist < COLLISION_TOLERANCE && c.0.point.normal.dot(s.front) > 0.5).map(|(_, s)| s.front).collect();
    let mut flips: Vec<(usize, f32)> = Vec::new();
    for (k, (c, s)) in cands.iter().zip(sheets).enumerate() {
        if s.front == Vec3::ZERO || c.0.point.normal.dot(s.front) >= -0.5 {
            continue;
        }
        if fronts.iter().any(|f| f.dot(s.front) < -0.5) {
            continue;
        }
        let depth = s.plane - s.body_min;
        if depth > 0.0 && depth < reach {
            flips.push((k, depth));
        }
    }
    // The nearest exit pushes, but an exit DOWN counts double: a body saved inside a thin deck or slab is better left on
    // top of it than dropped out of its underside, off the world.
    let eff = |&(k, d): &(usize, f32)| if sheets[k].front.z < -0.5 { d * 2.0 } else { d };
    let Some(&(best, depth)) = flips.iter().min_by(|a, b| eff(a).total_cmp(&eff(b))) else { return };
    // every flipped point of the nearest exit's PATCH (same front) is kept at the same depth, so the push is
    // level; a single point would tip the body
    let s = sheets[best];
    let mut keep = vec![true; cands.len()];
    for &(k, _) in &flips {
        let same = sheets[k].front.dot(s.front) > 0.99;
        keep[k] = same;
        if same {
            let p = &mut cands[k].0.point;
            p.normal = s.front;
            p.dist = -depth;
        }
    }
    let mut idx = 0;
    cands.retain(|_| {
        idx += 1;
        keep[idx - 1]
    });
}
