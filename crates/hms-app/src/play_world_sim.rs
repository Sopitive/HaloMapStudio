//! #prop-sim  The loose props, weapons and the player's push, on the engine's own Havok pipeline.
//!
//! [`PropWorld`] keeps the slots, the lazy decode, the wake rules and the public API; the physics of every
//! NON-vehicle body is one persistent [`crate::reach_havok_solver::Sim`] (broad phase, contact admission,
//! manifold persistence, the 8-iteration solver, time of impact against the level, deactivation), stepped once
//! per 60 Hz tick. `Live.hav` stays the canonical state the rest of the world reads and writes (launches,
//! `set_body_velocity`, `restore_all`); it is copied into the Sim at the start of the tick and back out after.
//!
//! * **Statics.** The level soup plus the loaded Fixed hulls, as triangles with the environment convex radius,
//!   declared OUTWARD-WOUND ([`PropSoup`]): a saved pose that overlaps the level is sent out the front of the
//!   face (the same rule as `physics_settle::shape_contacts_sided`), never on through it.
//! * **The player** is one dynamic Character body (125 kg, no inverse inertia, no gravity) put at the
//!   controller's pose with the controller's velocity every tick: the controller stays the authority for where
//!   the player is, the Sim decides what a prop does about it. A prop pinned against a wall pushes back on the
//!   body, the wall holds the prop.
//! * **Vehicles stay on the old solver** (`PropWorld::substep`: hover pads, suspension-like support spheres,
//!   passive wheeled hulls), which the Sim, being all-rigid, cannot replace. They join the Sim as Keyframed
//!   proxies (pose and velocity from the vehicle) so props collide with them and are pushed by them; a driven
//!   vehicle's hull (a [`Kinematic`] pusher) is the same kind of proxy. In the other direction the old solver
//!   sees an awake prop as a body of its true mass whose result is discarded, so a vehicle is slowed by what
//!   it hits by exactly the momentum the prop takes.
//! * **Sleep** is the engine's deactivation (with one veto of ours: no sleeping while sunk more than
//!   [`crate::play_world::PEN_NO_SLEEP`] into something). A Sim wake (a body touched by an awake one) is
//!   mirrored to the slot, and a slot wake (a pusher, a script, a launch) to the Sim.

use super::*;
use crate::play_driven::{local_shape, tri_id};
use crate::reach_havok::{QualityType, TICK_HZ};
use crate::reach_havok_solver::{LocalShape, Sim, SimBody, SolverInfo, StaticPoly, StaticSource, ENV_RADIUS};

/// The level soup and the loaded Fixed hulls as the Sim's static source.
pub(super) struct PropSoup<'a> {
    pub level: &'a dyn StaticWorld,
    pub statics: &'a TriGrid,
}

impl StaticSource for PropSoup<'_> {
    fn material(&self) -> (f32, f32) {
        (LEVEL_FRICTION, LEVEL_RESTITUTION)
    }
    fn outward_wound(&self) -> bool {
        true
    }
    fn gather(&self, min: Vec3, max: Vec3, out: &mut Vec<(u32, StaticPoly)>) {
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        self.level.gather(min, max, &mut tris);
        self.statics.gather(min, max, &mut tris);
        for t in tris {
            if (t[1] - t[0]).cross(t[2] - t[0]).length_squared() < 1.0e-10 {
                continue;
            }
            out.push((tri_id(&t), StaticPoly::triangle(t[0], t[1], t[2], ENV_RADIUS)));
        }
    }
}

/// The Sim and which of its bodies is whose.
#[derive(Default)]
pub(super) struct PropSim {
    sim: Option<Sim>,
    /// `lives` index of a loose body -> its Sim body
    by_live: HashMap<usize, usize>,
    /// `lives` index of a vehicle -> its Keyframed proxy (None: nothing collidable)
    proxy: HashMap<usize, Option<usize>>,
    /// pusher geometry key -> the Sim bodies built for it
    pool: HashMap<u64, Vec<usize>>,
    /// Sim bodies of pushers placed this tick, and the tick before
    placed: Vec<usize>,
    placed_before: Vec<usize>,
    /// bodies whose saved pose has been freed from the level
    repaired: std::collections::HashSet<usize>,
    /// last tick's contact counts (static, pair)
    pub last_contacts: (usize, usize),
}

/// A single capsule is the player (a dynamic Character body); any other pusher is a driven hull (Keyframed).
fn is_character(k: &Kinematic) -> bool {
    k.shapes.len() == 1 && matches!(k.shapes[0].geom, ShapeGeom::Capsule { .. })
}

fn kin_key(k: &Kinematic) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u32| h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3);
    mix(k.mass.to_bits());
    mix(k.com.x.to_bits());
    mix(k.com.y.to_bits());
    mix(k.com.z.to_bits());
    for s in k.shapes {
        match &s.geom {
            ShapeGeom::Sphere { c, r } => {
                mix(1);
                c.to_array().iter().for_each(|x| mix(x.to_bits()));
                mix(r.to_bits());
            }
            ShapeGeom::Capsule { a, b, r } => {
                mix(2);
                a.to_array().iter().chain(b.to_array().iter()).for_each(|x| mix(x.to_bits()));
                mix(r.to_bits());
            }
            ShapeGeom::Hull { pts, r, .. } => {
                mix(3);
                mix(pts.len() as u32);
                pts.iter().for_each(|p| p.to_array().iter().for_each(|x| mix(x.to_bits())));
                mix(r.to_bits());
            }
        }
    }
    h
}

/// How fast the player's body may change velocity toward the controller's, wu/s^2 (INFERRED: the biped reaches its run
/// speed in about an eighth of a second).
const PUSH_ACCEL: f32 = 20.0;

/// Deeper than this (wu) into the level a saved pose is freed, and for at most this many rounds.
const REPAIR_DEPTH: f32 = 0.05;
const REPAIR_ROUNDS: usize = 40;

const FAR: Vec3 = Vec3::new(0.0, 0.0, -1.0e6);

impl PropWorld {
    fn sim_prop_slot(&self, l: usize) -> Option<usize> {
        let s = self.owner[l];
        (!self.slots[s].seed.vehicle).then_some(s)
    }

    /// One tick of every loose body on the Sim. Returns without doing any work when nothing loose is awake.
    pub(super) fn sim_tick(&mut self, level: &dyn StaticWorld, src: &dyn PropSource, kin: &[Kinematic], st: &mut StepStats) {
        let awake: Vec<usize> = (0..self.slots.len())
            .filter(|&i| !self.slots[i].asleep && !self.slots[i].driven && !self.slots[i].seed.vehicle && matches!(self.slots[i].kind, Kind::Live(_)))
            .collect();
        if awake.is_empty() {
            return;
        }
        // every dormant neighbour of an awake body is decoded BEFORE the gather: a Fixed object contributes its
        // hull to the static grid, a body becomes an obstacle (or a partner) of the Sim
        let mut near: Vec<usize> = Vec::new();
        for &i in &awake {
            let Kind::Live(l) = self.slots[i].kind else { continue };
            let ab = self.slots[i].aabb;
            let sweep = self.lives[l].hav.v * TICK_SECONDS;
            let (mn, mx) = (ab.0.min(ab.0 + sweep) - Vec3::splat(0.05), ab.1.max(ab.1 + sweep) + Vec3::splat(0.05));
            self.neighbours((mn, mx), i, &mut near);
            for &j in near.clone().iter() {
                if matches!(self.slots[j].kind, Kind::Dormant) {
                    self.materialise(j, src);
                }
            }
        }
        let mut ps = std::mem::take(&mut self.simw);
        let sim = ps.sim.get_or_insert_with(|| Sim::new(SolverInfo::engine(TICK_HZ, world_gravity())));
        let dt = TICK_SECONDS;
        // --- the first tick of a body: free a saved pose that is inside the level ---
        // A map's loose objects are saved where the editor dropped them: a hand's width into a Forge piece, or wholly inside
        // its walls. The Havok world sends a body out of what it overlaps the way the engine's continuous collision would have
        // kept it out of it, but a body already INSIDE a thick solid, past the faces' sheets, touches nothing there. So, once,
        // when a body first takes part, this world's own depth-aware rule (`shape_contacts_sided`: only the nearest exit of a
        // vertex pushes, and never DOWN: a body saved in a floor slab is brought up, not out of its underside) moves it out of anything deeper than [`REPAIR_DEPTH`] (a capped share of the penetration per
        // round, up to [`REPAIR_ROUNDS`] rounds), and the Havok world takes over from there.
        {
            let mut tris: Vec<[Vec3; 3]> = Vec::new();
            let mut cbuf: Vec<Contact> = Vec::new();
            for &i in &awake {
                let Kind::Live(l) = self.slots[i].kind else { continue };
                if !ps.repaired.insert(l) {
                    continue;
                }
                for _ in 0..REPAIR_ROUNDS {
                    let ab = self.slots[i].aabb;
                    tris.clear();
                    level.gather(ab.0 - Vec3::splat(0.05), ab.1 + Vec3::splat(0.05), &mut tris);
                    self.statics.gather(ab.0 - Vec3::splat(0.05), ab.1 + Vec3::splat(0.05), &mut tris);
                    if tris.is_empty() {
                        break;
                    }
                    let (rot_q, origin) = (self.slots[i].rot, self.slots[i].origin);
                    let (rot, inv) = (Mat3::from_quat(rot_q), Mat3::from_quat(rot_q).transpose());
                    let (mut sum, mut n) = (Vec3::ZERO, 0u32);
                    for sh in &self.lives[l].shapes {
                        cbuf.clear();
                        crate::physics_settle::shape_contacts_sided(sh, rot, origin, inv, &tris, true, &mut cbuf);
                        for c in cbuf.iter().filter(|c| c.depth > REPAIR_DEPTH && c.n.z > -0.5) {
                            sum += c.n * c.depth;
                            n += 1;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                    let push = sum / n as f32 * POSITION_PUSH;
                    let push = if push.length() > MAX_PUSH_PER_SUBSTEP { push.normalize() * MAX_PUSH_PER_SUBSTEP } else { push };
                    self.lives[l].hav.com += push;
                    self.sync_slot(i);
                }
            }
        }
        // --- bodies: create what is new, copy the canonical state in ---
        for l in 0..self.lives.len() {
            let s = self.owner[l];
            let lv = &self.lives[l];
            if self.slots[s].seed.vehicle {
                let entry = ps.proxy.entry(l).or_insert_with(|| {
                    let hull: Vec<&ConvexShape> = lv.shapes.iter().filter(|x| !x.support).collect();
                    if hull.is_empty() {
                        return None;
                    }
                    let shapes: Vec<LocalShape> = hull.iter().map(|x| local_shape(x, lv.com_local)).collect();
                    let mass = if lv.hav.inv_mass > 0.0 { 1.0 / lv.hav.inv_mass } else { 1.0 };
                    let mut b = SimBody::new(HavokBody::new(MotionType::Keyframed, mass, Mat3::ZERO, lv.hav.com, lv.hav.q), shapes, hull.iter().map(|x| x.mat.friction).sum::<f32>() / hull.len() as f32, 0.0);
                    b.quality = QualityType::Keyframed;
                    Some(sim.add(b))
                });
                if let Some(&mut b) = entry.as_mut() {
                    let hb = &mut sim.bodies[b].hav;
                    if self.slots[s].driven {
                        hb.com = FAR;
                        hb.set_velocity(Vec3::ZERO, Vec3::ZERO);
                    } else {
                        hb.com = lv.hav.com;
                        hb.q = lv.hav.q;
                        hb.set_velocity(lv.hav.v, lv.hav.w);
                    }
                }
                continue;
            }
            let b = *ps.by_live.entry(l).or_insert_with(|| {
                let shapes: Vec<LocalShape> = lv.shapes.iter().map(|x| local_shape(x, lv.com_local)).collect();
                let (f, e) = shape_mat(&lv.shapes);
                let mut sb = SimBody::new(lv.hav.clone(), shapes, f, e);
                sb.awake = false;
                sim.add(sb)
            });
            sim.bodies[b].hav = lv.hav.clone();
            if self.slots[s].asleep {
                sim.bodies[b].awake = false;
                sim.bodies[b].hav.set_velocity(Vec3::ZERO, Vec3::ZERO);
            } else if !sim.bodies[b].awake {
                sim.wake(b);
            }
        }
        // --- pushers ---
        ps.placed_before = std::mem::take(&mut ps.placed);
        let mut taken: HashMap<u64, usize> = HashMap::new();
        for k in kin {
            let key = kin_key(k);
            let n = taken.entry(key).or_insert(0);
            let list = ps.pool.entry(key).or_default();
            if *n == list.len() {
                let character = is_character(k);
                let shapes: Vec<LocalShape> = k.shapes.iter().map(|x| local_shape(x, k.com)).collect();
                let (f, e) = shape_mat(k.shapes);
                let mass = k.mass.max(1.0);
                let motion = if character { MotionType::BoxInertia } else { MotionType::Keyframed };
                let mut hav = HavokBody::new(motion, mass, Mat3::ZERO, k.origin + k.rot * k.com, k.rot);
                if character {
                    hav.set_gravity_factor(0.0);
                }
                let mut b = SimBody::new(hav, shapes, f, e);
                b.quality = if character { QualityType::Character } else { QualityType::Keyframed };
                // the controller places the player's body: its penetration is the solver's to resolve, not a
                // time-of-impact event's (whose one-shot impulse drove a weapon pinned against a wall through it)
                b.toi = !character;
                list.push(sim.add(b));
            }
            let b = list[*n];
            *n += 1;
            let character = is_character(k);
            let fresh = !ps.placed_before.contains(&b);
            let sb = &mut sim.bodies[b];
            // The pusher is put where the controller left it (the controller is the authority for position and a
            // prop's hull stops it at touching). A character's VELOCITY is the engine's: what the solver left of it
            // (a body that cannot move stops it) accelerated toward the wish velocity by at most [`PUSH_ACCEL`],
            // so a player leaning on a weapon pinned against a wall does not drive it through the wall.
            sb.hav.com = k.origin + k.rot * k.com;
            sb.hav.q = k.rot;
            if character {
                let v = if fresh { k.velocity } else { sb.hav.v };
                let d = k.velocity - v;
                let step = PUSH_ACCEL * dt;
                let d = if d.length() > step { d / d.length() * step } else { d };
                sb.hav.set_velocity(v + d, Vec3::ZERO);
                sb.awake = true;
                sb.aux.sleep.reset();
            } else {
                // a driven hull is Keyframed: its velocity is the vehicle's
                sb.hav.set_velocity(k.velocity, k.angular);
            }
            ps.placed.push(b);
        }
        for list in ps.pool.values() {
            for &b in list {
                if !ps.placed.contains(&b) {
                    sim.bodies[b].hav.com = FAR;
                    sim.bodies[b].hav.set_velocity(Vec3::ZERO, Vec3::ZERO);
                    sim.bodies[b].awake = false;
                }
            }
        }
        // --- one engine tick ---
        let soup = PropSoup { level, statics: &self.statics };
        sim.tick(&soup);
        ps.last_contacts = sim.contact_counts();
        st.static_contacts += ps.last_contacts.0;
        st.pair_contacts += ps.last_contacts.1;
        // --- copy the result out, mirror wake and sleep ---
        for (&l, &b) in ps.by_live.iter() {
            let Some(s) = self.sim_prop_slot(l) else { continue };
            let sb = &sim.bodies[b];
            let was_asleep = self.slots[s].asleep;
            if was_asleep && !sb.awake {
                continue;
            }
            self.lives[l].hav = sb.hav.clone();
            if sb.awake {
                if was_asleep {
                    self.slots[s].asleep = false;
                    self.slots[s].ever_woken = true;
                    let lv = &mut self.lives[l];
                    lv.calm = 0;
                    lv.quiet = 0;
                    lv.ref_com = lv.hav.com;
                    lv.ref_q = lv.hav.q;
                    st.woke += 1;
                }
            } else {
                self.lives[l].hav.set_velocity(Vec3::ZERO, Vec3::ZERO);
                self.slots[s].asleep = true;
                st.slept += 1;
            }
            self.sync_slot(s);
        }
        self.simw = ps;
    }
}
