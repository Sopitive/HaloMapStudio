//! #driven-sim  A DRIVEN vehicle's hull against the static world, resolved by the engine's own
//! rigid-body pipeline ([`crate::reach_havok_solver`]) instead of the vertex-against-face sweep.
//!
//! # Why this exists
//!
//! Reported: *"the vehicle stuff basically won't let me drive off any ledge at all, it just
//! teleports me backward when I'm about to drive off the edge of basically anything."* The old
//! stand-in ([`crate::play_world::sweep_hull`]) took each hull vertex's closest point on a level
//! triangle and, when that point was a triangle EDGE, flipped the triangle's normal to the side of
//! the hull's centre and pushed the vertex back "through the face". At a ledge the nose is over the
//! drop, above the cliff face's top edge and on the cliff face's OPEN side: no contact exists, but
//! the flip called it a vertex that had crossed the sheet and pushed the whole hull back along the
//! cliff polygon's normal (measured at `forge_halo` (69.5, 207.1, 44.9): 0.19 wu in one tick, along
//! the horizontal normal of a ledge wall).
//!
//! The engine has no such rule (`research/havok/07-audit-vehicle.md` section 6): a vehicle is one
//! CRITICAL rigid body whose hull meets the level's polygons through the PRED_GSK agent with the
//! admission and edge-suppression rules of `01-collision-detection.md` (a hull contact on a polygon
//! edge or vertex is only a "potential" contact, dropped when a deeper face contact covers it), the
//! contact is resolved along the contact normal as a velocity bias, and the time of impact stops the
//! hull before it passes a wall. Nothing in the vehicle code moves the body (PROVEN, 07 section 6.5).
//!
//! # Order, as the engine has it
//!
//! The vehicle law runs first and only SETS velocities ([`crate::reach_vehicle`]'s `jeep_tick`,
//! `sub_18098AD00`); the world step then integrates, collides and solves. Here the law has already
//! moved the pose by the time this runs, so the pose it came from (`prev`) and the pose it proposes
//! define the velocities the law "set" (`(proposed - prev) / dt`, linear and angular); the world
//! steps those from `prev`; what the contacts changed is added back to the ride's own velocities.
//! A frame in which the hull touches nothing is returned UNTOUCHED (the pose and velocities are the
//! law's, bit for bit), so free flight and the wheel springs are not re-integrated.
//!
//! # Stand-ins (labelled)
//!
//! * The static world is the level's triangle soup (and the placed objects' hulls) with the
//!   engine's convex radius 0.0164, not the BSP's own n-gons (INFERRED harmless: coplanar
//!   neighbours give the same normal).
//! * Frames are stepped in whole 60 Hz ticks (a frame shorter than a tick is one shorter step);
//!   the engine steps once per 60 Hz tick.
//! * Gravity is NOT applied here (the law's velocities already carry it); the body's gravity
//!   factor is 0 and the world gravity only feeds the admission thresholds.
//! * Hulls that are not convex polyhedra with faces (more than 64 vertices) become their box.

use crate::physics_settle::{ConvexShape, PhysBody, ShapeGeom, StaticWorld};
use crate::play_world::{HullStep, LEVEL_FRICTION, LEVEL_RESTITUTION};
use crate::reach_havok::{HavokBody, MotionType};
use crate::reach_havok_solver::{LocalShape, Sim, SimBody, SolverInfo, StaticPoly, StaticSource, ENV_RADIUS};
use glam::{Mat3, Quat, Vec3};

/// The engine's tick, Hz.
const TICK_HZ: f32 = 60.0;

/// The face loops of a convex polyhedron given by its vertices and outward planes
/// (`n . x == d` on the face): the vertices on each plane, in cyclic order about its normal.
fn loops_from_planes(pts: &[Vec3], planes: &[(Vec3, f32)]) -> Vec<Vec<usize>> {
    let ext = pts.iter().fold(0.0f32, |m, p| m.max(p.length())).max(1.0);
    let eps = 1.0e-3 * ext;
    let mut out = Vec::new();
    for (n, d) in planes {
        let mut idx: Vec<usize> = (0..pts.len()).filter(|&i| (n.dot(pts[i]) - d).abs() < eps).collect();
        if idx.len() < 3 {
            continue;
        }
        let c = idx.iter().map(|&i| pts[i]).sum::<Vec3>() / idx.len() as f32;
        let u = (n.any_orthonormal_vector()).normalize();
        let v = n.cross(u);
        idx.sort_by(|&a, &b| {
            let (pa, pb) = (pts[a] - c, pts[b] - c);
            pa.dot(v).atan2(pa.dot(u)).total_cmp(&pb.dot(v).atan2(pb.dot(u)))
        });
        out.push(idx);
    }
    out
}

/// `phmo` shape -> solver shape, relative to the centre of mass `com`.
pub(crate) fn local_shape(s: &ConvexShape, com: Vec3) -> LocalShape {
    match &s.geom {
        ShapeGeom::Sphere { c, r } => LocalShape::Sphere { center: *c - com, radius: *r },
        ShapeGeom::Capsule { a, b, r } => LocalShape::Capsule { a: *a - com, b: *b - com, radius: *r },
        ShapeGeom::Hull { pts, planes, r } => {
            let loops = if pts.len() <= crate::physics_settle::HULL_PLANE_LIMIT { loops_from_planes(pts, planes) } else { Vec::new() };
            if loops.len() >= 4 {
                LocalShape::Hull { verts: pts.iter().map(|p| *p - com).collect(), loops, radius: *r }
            } else {
                // the box of the vertices
                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for p in pts {
                    lo = lo.min(*p);
                    hi = hi.max(*p);
                }
                let centre = (lo + hi) * 0.5 - com;
                match LocalShape::cuboid(((hi - lo) * 0.5).max(Vec3::splat(1.0e-3)), *r) {
                    LocalShape::Hull { verts, loops, radius } => LocalShape::Hull { verts: verts.into_iter().map(|v| v + centre).collect(), loops, radius },
                    other => other,
                }
            }
        }
    }
}

/// A stable id for a soup triangle (the solver matches contact points across ticks by it).
pub(crate) fn tri_id(t: &[Vec3; 3]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for p in t {
        for c in p.to_array() {
            h = (h ^ c.to_bits()).wrapping_mul(0x0100_0193);
            h ^= h >> 15;
        }
    }
    h
}

/// The level soup as the solver's static source.
pub struct SoupSource<'a>(pub &'a dyn StaticWorld);

impl StaticSource for SoupSource<'_> {
    fn material(&self) -> (f32, f32) {
        (LEVEL_FRICTION, LEVEL_RESTITUTION)
    }
    fn gather(&self, min: Vec3, max: Vec3, out: &mut Vec<(u32, StaticPoly)>) {
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        self.0.gather(min, max, &mut tris);
        for t in tris {
            if (t[1] - t[0]).cross(t[2] - t[0]).length_squared() < 1.0e-10 {
                continue;
            }
            out.push((tri_id(&t), StaticPoly::triangle(t[0], t[1], t[2], ENV_RADIUS)));
        }
    }
}

/// One driven vehicle's hull as a CRITICAL rigid body in a world of its own.
pub struct DrivenSim {
    sim: Sim,
    idx: usize,
    /// Centre of mass in model space.
    com: Vec3,
    /// The ride keeps a full rotation (the jeep law); otherwise the body is translation-only.
    rotates: bool,
}

impl DrivenSim {
    /// `None` when the body has no hull shape.
    pub fn new(body: &PhysBody, rotates: bool) -> Option<Self> {
        let hull: Vec<&ConvexShape> = body.shapes.iter().filter(|s| !s.support).collect();
        if hull.is_empty() {
            return None;
        }
        let shapes: Vec<LocalShape> = hull.iter().map(|s| local_shape(s, body.com)).collect();
        let friction = hull.iter().map(|s| s.mat.friction).sum::<f32>() / hull.len() as f32;
        let inv_inertia = if rotates && body.inertia.determinant().abs() > 1.0e-12 { body.inertia.inverse() } else { Mat3::ZERO };
        let inv_inertia = if inv_inertia.to_cols_array().iter().all(|x| x.is_finite()) { inv_inertia } else { Mat3::ZERO };
        let mut hav = HavokBody::new(MotionType::BoxInertia, body.mass.max(1.0), inv_inertia, Vec3::ZERO, Quat::IDENTITY);
        hav.set_gravity_factor(0.0);
        let mut sim = Sim::new(SolverInfo::engine(TICK_HZ, crate::reach_havok::world_gravity()));
        // restitution 0: the engine's resting velocity is FLT_MAX and the hull's material is 0
        let idx = sim.add(SimBody::new(hav, shapes, friction, 0.0));
        Some(DrivenSim { sim, idx, com: body.com, rotates })
    }

    /// `allowedPenetrationDepth` of the hull body (`max(0.4 * minDimension, 0.01)`): the depth a
    /// continuous-collision pair is brought back to is `0.2` of it (`03-continuous-toi.md`).
    pub fn allowed_penetration(&self) -> f32 {
        self.sim.bodies[self.idx].allowed_penetration
    }

    /// Take the pose `prev` -> (`pos`, `rot`) the vehicle law proposes over `dt` seconds through
    /// the world; write back the corrected pose and add the contacts' velocity change to `vel` /
    /// `ang`. A frame with no contact leaves all of them exactly as they were.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        prev_pos: Vec3,
        prev_rot: Quat,
        pos: &mut Vec3,
        rot: &mut Quat,
        vel: &mut Vec3,
        ang: &mut Vec3,
        dt: f32,
        world: &dyn StaticWorld,
    ) -> HullStep {
        let mut out = HullStep::default();
        if !(dt > 1.0e-6) {
            return out;
        }
        let com_prev = prev_pos + prev_rot * self.com;
        let com_new = *pos + *rot * self.com;
        let v_in = (com_new - com_prev) / dt;
        let w_in = if self.rotates {
            let mut dq = (*rot * prev_rot.conjugate()).normalize();
            if dq.w < 0.0 {
                dq = -dq;
            }
            let (axis, angle) = dq.to_axis_angle();
            if axis.is_finite() { axis * angle / dt } else { Vec3::ZERO }
        } else {
            Vec3::ZERO
        };
        {
            let b = &mut self.sim.bodies[self.idx];
            b.hav.com = com_prev;
            b.hav.q = prev_rot.normalize();
            b.hav.v = v_in;
            b.hav.w = w_in;
        }
        let n = ((dt * TICK_HZ).round() as usize).clamp(1, 12);
        let dt_i = dt / n as f32;
        let src = SoupSource(world);
        let g = crate::reach_havok::world_gravity();
        let toi0 = self.sim.toi_count;
        let mut touched = false;
        for _ in 0..n {
            self.sim.info = SolverInfo::engine(1.0 / dt_i, g);
            // a driven body never sleeps: its velocities are set by the law every frame
            self.sim.wake(self.idx);
            self.sim.bodies[self.idx].aux.sleep.reset();
            let rep = self.sim.tick(&src);
            out.steps += 1;
            let cs = self.sim.body_contacts(self.idx);
            if rep.rows > 0 || !cs.is_empty() {
                touched = true;
                out.hits += 1;
                for c in &cs {
                    out.deepest = out.deepest.max(-c.dist);
                }
                if out.first.is_none() {
                    if let Some(c) = cs.iter().min_by(|a, b| a.dist.total_cmp(&b.dist)) {
                        out.first = Some((c.pos, c.normal));
                    }
                }
            }
        }
        touched |= self.sim.toi_count > toi0;
        if !touched {
            return out;
        }
        out.hits = out.hits.max(1);
        let b = &self.sim.bodies[self.idx];
        let (q, com) = (b.hav.q, b.hav.com);
        let dv = b.hav.v - v_in;
        out.impact = (-dv.dot(v_in.normalize_or_zero())).max(0.0);
        *pos = com - q * self.com;
        if self.rotates {
            *rot = q;
            *ang += b.hav.w - w_in;
        }
        *vel += dv;
        out
    }
}

#[cfg(test)]
mod tests {
    //! Gates for [`DrivenSim`] on synthetic worlds whose answers are geometry (a wall's plane, a
    //! ledge's edge), not the code under test. The vehicle "law" here is the simplest one that
    //! can be wrong in the owner's way: a constant velocity it re-asserts every frame, so a
    //! hull that is stopped or pushed has to be stopped or pushed by the world alone.

    use super::*;
    use crate::physics_settle::ShapeMaterial;

    struct Tris(Vec<[Vec3; 3]>);
    impl StaticWorld for Tris {
        fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            for t in &self.0 {
                let lo = t[0].min(t[1]).min(t[2]);
                let hi = t[0].max(t[1]).max(t[2]);
                if hi.cmpge(mn).all() && lo.cmple(mx).all() {
                    out.push(*t);
                }
            }
        }
    }

    fn quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> [[Vec3; 3]; 2] {
        [[a, b, c], [a, c, d]]
    }

    /// A Mongoose-sized hull: 3.2 x 1.8 x 0.7 wu, origin at its centre.
    fn hull_body() -> PhysBody {
        let mat = ShapeMaterial { friction: 1.0, restitution: 0.0, mass: 0.0 };
        let half = Vec3::new(1.6, 0.9, 0.35);
        let shape = ConvexShape::aabb_box(Vec3::ZERO, half, 0.0164, mat).expect("box");
        let m = 600.0;
        let inertia = Mat3::from_diagonal(Vec3::new(
            m / 12.0 * (4.0 * (half.y * half.y + half.z * half.z)),
            m / 12.0 * (4.0 * (half.x * half.x + half.z * half.z)),
            m / 12.0 * (4.0 * (half.x * half.x + half.y * half.y)),
        ));
        PhysBody { shapes: vec![shape], mass: m, com: Vec3::ZERO, inertia, lin_damping: 0.0, ang_damping: 0.0, scale: 1.0, fixed: false, desc: String::new() }
    }

    /// One frame's record.
    #[derive(Clone, Copy)]
    struct Frame {
        pos: Vec3,
        vel: Vec3,
        hits: usize,
        step: HullStep,
    }

    /// Drive `law` (position, the velocity the last frame ended with -> the velocity the law sets) for `secs` at frame time `dt`.
    fn drive(world: &Tris, rotates: bool, start: Vec3, dt: f32, secs: f32, law: &dyn Fn(Vec3, Vec3) -> Vec3) -> Vec<Frame> {
        let mut sim = DrivenSim::new(&hull_body(), rotates).expect("sim");
        let (mut pos, mut rot) = (start, Quat::IDENTITY);
        let mut out = Vec::new();
        let mut last_v = Vec3::ZERO;
        for _ in 0..(secs / dt) as usize {
            let v = law(pos, last_v);
            let (mut p, mut r, mut vel, mut ang) = (pos + v * dt, rot, v, Vec3::ZERO);
            let step = sim.step(pos, rot, &mut p, &mut r, &mut vel, &mut ang, dt, world);
            pos = p;
            rot = r;
            last_v = vel;
            out.push(Frame { pos, vel, hits: step.hits, step });
        }
        out
    }

    const FRAMES: [f32; 3] = [1.0 / 60.0, 1.0 / 30.0, 1.0 / 144.0];

    /// A frame in which the hull touches nothing is the law's, bit for bit.
    #[test]
    fn a_frame_without_contact_is_returned_untouched() {
        let floor = Tris(quad(Vec3::new(-50.0, -50.0, 0.0), Vec3::new(50.0, -50.0, 0.0), Vec3::new(50.0, 50.0, 0.0), Vec3::new(-50.0, 50.0, 0.0)).to_vec());
        let mut sim = DrivenSim::new(&hull_body(), true).expect("sim");
        let prev = Vec3::new(0.0, 0.0, 2.0);
        let v = Vec3::new(7.0, 1.0, 0.0);
        let q = Quat::from_rotation_z(0.3);
        let (mut p, mut r, mut vel, mut ang) = (prev + v / 60.0, q, v, Vec3::new(0.0, 0.0, 0.2));
        let (p0, r0, v0, a0) = (p, r, vel, ang);
        let st = sim.step(prev, q, &mut p, &mut r, &mut vel, &mut ang, 1.0 / 60.0, &floor);
        assert_eq!(st.hits, 0, "a hull 1.6 wu above the floor touches nothing");
        assert_eq!((p, r, vel, ang), (p0, r0, v0, a0), "an untouched frame must be the law's, bit for bit");
    }

    /// A vehicle driven at a wall at 11 wu/s stops AT the wall, at every frame time: its front
    /// never passes the wall's plane by more than the allowed penetration, never ends behind it,
    /// and is never moved backward by more than a push-out of a few millimetres.
    #[test]
    fn a_vehicle_driven_into_a_wall_stops_at_it_and_is_not_thrown_back() {
        let wall_x = 10.0;
        let wall = Tris(quad(Vec3::new(wall_x, -20.0, -5.0), Vec3::new(wall_x, 20.0, -5.0), Vec3::new(wall_x, 20.0, 5.0), Vec3::new(wall_x, -20.0, 5.0)).to_vec());
        // the front surface (core + radius) and the wall polygon's padded surface (radius 0.0164)
        let half_len = 1.6 + 0.0164;
        let surface = wall_x - ENV_RADIUS;
        let allowed = DrivenSim::new(&hull_body(), true).expect("sim").allowed_penetration();
        for dt in FRAMES {
            let fr = drive(&wall, true, Vec3::new(0.0, 0.0, 0.0), dt, 3.0, &|_, _| Vec3::new(11.0, 0.0, 0.0));
            assert!(fr.iter().any(|f| f.hits > 0), "liveness: the hull never met the wall at dt {dt}");
            let mut prev_x = 0.0f32;
            let mut worst_back = 0.0f32;
            let mut deepest = 0.0f32;
            for f in &fr {
                worst_back = worst_back.min(f.pos.x - prev_x);
                prev_x = f.pos.x;
                deepest = deepest.max(f.pos.x + half_len - surface);
            }
            let last = fr.last().unwrap();
            assert!(worst_back > -0.01, "dt {dt}: the hull was moved backward by {worst_back} in one frame");
            // the engine brings a CRITICAL pair back to 0.2 * allowedPenetration (`03`): twice that is slack
            assert!(deepest < 0.4 * allowed, "dt {dt}: the front went {deepest} wu into the wall's padded surface (allowed {allowed})");
            assert!(fr.iter().all(|f| f.pos.x < wall_x), "dt {dt}: the hull's centre passed the wall");
            assert!(last.pos.x + half_len > surface - 0.1, "dt {dt}: stopped {} wu short of the wall", surface - last.pos.x - half_len);
            assert!(last.vel.x.abs() < 1.0, "dt {dt}: still moving {} into the wall", last.vel.x);
        }
    }

    /// The owner's report, on a synthetic ledge: driving over the edge of a slab. The origin
    /// passes the edge, the law lets go of the ground (gravity from there), and the hull must go
    /// FORWARD every frame (no frame moves it against its heading by more than a millimetre) and
    /// fall away from the slab, not hang on it.
    #[test]
    fn driving_off_a_ledge_never_moves_the_hull_backward() {
        let edge = 10.0;
        let top = 0.0;
        let mut tris = quad(Vec3::new(-40.0, -10.0, top), Vec3::new(edge, -10.0, top), Vec3::new(edge, 10.0, top), Vec3::new(-40.0, 10.0, top)).to_vec();
        // the cliff face (normal +x), 6 wu deep
        tris.extend(quad(Vec3::new(edge, -10.0, top), Vec3::new(edge, -10.0, -6.0), Vec3::new(edge, 10.0, -6.0), Vec3::new(edge, 10.0, top)));
        let slab = Tris(tris);
        let hull_h = 0.35 + 0.0164;
        for dt in FRAMES {
            for speed in [3.0f32, 9.0] {
                for rotates in [false, true] {
                    // the law: hold the speed; gravity acts on the vertical velocity the last frame ended
                    // with once the origin is past the edge (a supported hull has it cancelled by contact)
                    let g = 4.17f32;
                    let law = move |p: Vec3, last: Vec3| Vec3::new(speed, 0.0, if p.x > edge { last.z - g * dt } else { 0.0 });
                    let fr = drive(&slab, rotates, Vec3::new(0.0, 0.0, top + hull_h + 0.004), dt, 8.0, &law);
                    let mut prev = Vec3::new(0.0, 0.0, 0.0);
                    let mut worst = 0.0f32;
                    for f in &fr {
                        worst = worst.min(f.pos.x - prev.x);
                        prev = f.pos;
                    }
                    let last = fr.last().unwrap();
                    assert!(worst > -0.001, "dt {dt} speed {speed} rotates {rotates}: moved backward by {worst} in one frame");
                    assert!(last.pos.x > edge + 1.0, "dt {dt} speed {speed} rotates {rotates}: hung on the ledge at x {}", last.pos.x);
                    assert!(last.pos.z < top - 2.0, "dt {dt} speed {speed} rotates {rotates}: did not fall, z {}", last.pos.z);
                }
            }
        }
    }

    /// The owner's tiny steps (0.0083 and 0.0625 wu) under a hull that rides above them, the way
    /// a wheeled vehicle's hull does: it is never touched, so it is never stopped or moved.
    #[test]
    fn tiny_steps_under_a_riding_hull_never_touch_it() {
        for h in [0.0083f32, 0.0625] {
            let x0 = 6.0;
            let mut tris = quad(Vec3::new(-40.0, -10.0, 0.0), Vec3::new(x0, -10.0, 0.0), Vec3::new(x0, 10.0, 0.0), Vec3::new(-40.0, 10.0, 0.0)).to_vec();
            tris.extend(quad(Vec3::new(x0, -10.0, 0.0), Vec3::new(x0, -10.0, h), Vec3::new(x0, 10.0, h), Vec3::new(x0, 10.0, 0.0)));
            tris.extend(quad(Vec3::new(x0, -10.0, h), Vec3::new(40.0, -10.0, h), Vec3::new(40.0, 10.0, h), Vec3::new(x0, 10.0, h)));
            let world = Tris(tris);
            for dt in FRAMES {
                // 0.2 wu of clearance under the hull's padded bottom, over the higher side
                let z = 0.35 + 0.0164 + h + 0.2;
                let fr = drive(&world, true, Vec3::new(0.0, 0.0, z), dt, 2.0, &|_, _| Vec3::new(8.0, 0.0, 0.0));
                assert!(fr.iter().all(|f| f.hits == 0), "step {h} dt {dt}: a hull riding 0.2 wu clear of the step was touched");
                assert!(fr.last().unwrap().pos.x > 15.0, "step {h} dt {dt}: did not travel");
            }
        }
    }

    /// A hull whose bottom edge overlaps a riser by a hair (the wheels' bounce, 0.0083 wu) scrapes
    /// over it: it is not stopped, and what the contact moves it back by is bounded by the overlap
    /// plus the radius pad (a push-out along the contact normal, never a rewind of the frame).
    #[test]
    fn a_hull_edge_grazing_a_tiny_step_slides_over_it() {
        let h = 0.0625f32;
        let x0 = 6.0;
        let mut tris = quad(Vec3::new(-40.0, -10.0, 0.0), Vec3::new(x0, -10.0, 0.0), Vec3::new(x0, 10.0, 0.0), Vec3::new(-40.0, 10.0, 0.0)).to_vec();
        tris.extend(quad(Vec3::new(x0, -10.0, 0.0), Vec3::new(x0, -10.0, h), Vec3::new(x0, 10.0, h), Vec3::new(x0, 10.0, 0.0)));
        tris.extend(quad(Vec3::new(x0, -10.0, h), Vec3::new(40.0, -10.0, h), Vec3::new(40.0, 10.0, h), Vec3::new(x0, 10.0, h)));
        let world = Tris(tris);
        let overlap = 0.0083f32;
        for dt in FRAMES {
            // the hull's padded bottom is `overlap` below the step's top: z0 = h - overlap + 0.35 + r
            let z0 = h - overlap + 0.35 + 0.0164;
            let law = |p: Vec3, last: Vec3| {
                // vertical: the wheels hold the ride height over whichever side is under the front
                let want = z0 + if p.x + 1.6 > x0 { overlap + 0.01 } else { 0.0 };
                Vec3::new(8.0, 0.0, ((want - p.z) * 30.0).clamp(-2.0, 2.0) + 0.0 * last.z)
            };
            let fr = drive(&world, true, Vec3::new(0.0, 0.0, z0 + 0.0), dt, 2.0, &law);
            let mut prev_x = 0.0;
            let mut worst = 0.0f32;
            for f in &fr {
                worst = worst.min(f.pos.x - prev_x);
                prev_x = f.pos.x;
            }
            assert!(fr.iter().any(|f| f.hits > 0), "liveness: the hull never touched the step at dt {dt}");
            assert!(worst > -(overlap + ENV_RADIUS + 0.0164), "dt {dt}: a {overlap} wu graze moved the hull back {worst} in one frame");
            assert!(fr.last().unwrap().pos.x > 12.0, "dt {dt}: stopped at x {}", fr.last().unwrap().pos.x);
        }
    }
}
