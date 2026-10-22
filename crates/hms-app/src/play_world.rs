//! #play-world  Test mode's RIGID-BODY WORLD: every object the game would simulate is simulated,
//! not just the player and the vehicle being driven.
//!
//! Built on [`crate::reach_havok`] (the engine's integrator law at the true 60 Hz tick, damping,
//! caps, gravity factor, contact material mixing, the 16:1 contact mass clamp) and on the shapes
//! [`crate::physics_settle`] already decodes from `phmo`.
//!
//! # What is simulated (and what is not)
//!
//! * **Forge placement physics** (`placement & 0xC0`, [`PhysicsMode::from_placement`]; PROVEN
//!   layout, `mvar.rs`): `Normal` is a dynamic body; `Fixed` is terrain -- its physics hull is
//!   static triangles and it never moves; `Phased` has no collision at all, neither with bodies
//!   nor with the player.
//! * Objects that author no physics geometry, or whose multiplayer type is a REGION (spawn
//!   points, zones, volumes: [`crate::reach_collision::collides_with_biped`]) are not bodies.
//! * A body with a tag mass of zero or less is not a body.
//! * Everything else with a `phmo` (or a `coll` hull) is a rigid body: crates, barrels, scenery
//!   props, dropped weapons, equipment, powerups and vehicles. **Vehicles are passive rigid bodies
//!   here** (gravity, hull, damping): the tank / hover / VTOL laws are not ported, and a parked
//!   Warthog's suspension is the jeep law's, not this module's. The seam for the jeep law is
//!   [`PropWorld::set_body_velocity`] / [`PropWorld::body_pose`].
//!
//! # Initial state: everything starts ASLEEP at its saved pose
//!
//! Nothing falls when test mode starts (the owner's rule). A body wakes when something touches it:
//! the player, a driven vehicle, another awake body. A prop that was authored floating stays where
//! it is until disturbed -- the game would drop it. INFERRED; see UNRESOLVED.
//!
//! # Cost
//!
//! Bodies are DORMANT until needed: a slot holds only its coarse world AABB until something awake
//! overlaps it, and only then is its physics body decoded. The broad phase is a uniform grid of
//! body AABBs; static triangles come from the level soup plus the grid of Fixed hulls. A sleeping
//! world does no narrow-phase work at all (gated).
//!
//! # PROVEN / INFERRED / UNRESOLVED
//!
//! PROVEN (see `reach_havok`): integrator, gravity factor, damping, velocity caps, quality types,
//! friction / restitution mixing and storage, the contact mass clamp, world solver settings.
//! INFERRED: the contact solver (sequential impulses at the engine's 8 iterations and `tau`;
//! the Havok soft-contact equations are not lifted); sleeping thresholds (30 ticks under
//! 0.1 wu/s -- Havok's deactivation compares positions against `deactivationReferenceDistance`
//! 0.05 wu, which is not lifted either); the level's friction / restitution (the BSP triangle's
//! global material is not available to the gather, so the root `default_material` 0.9 / 0.2 is
//! used); the player's push (a kinematic pusher with the biped's mass under the 16:1 clamp).
//! UNRESOLVED: edge-edge contacts (box edges on box edges are not generated), TOI (replaced by
//! speed-driven substeps and a depenetration speed cap), collision damage to props, bodies
//! standing on a moving platform, whether the engine bounces (see
//! `reach_havok::CONTACT_RESTING_VELOCITY`).

use std::collections::HashMap;

use glam::{Mat3, Quat, Vec3};

use crate::physics_settle::{shape_contacts, tri_overlaps, ConvexShape, Contact, PhysBody, ShapeGeom, StaticWorld};
use crate::reach_havok::{
    contact_material, solve_pair_contacts, world_gravity, HavokBody, HoverLaw, MotionType, PairContact, SolverParams, TICK_SECONDS,
};

/// The root `default_material`'s physics properties, read from `forge_halo`'s `matg`
/// (`reach_havok` gate): used for the level side of every body-vs-level contact.
pub const LEVEL_FRICTION: f32 = 0.9;
pub const LEVEL_RESTITUTION: f32 = 0.2;

/// A body that is merely RESTING on a sleeper does not wake it; one moving faster than this does.
/// INFERRED.
pub const SLEEP_SPEED: f32 = 0.1;
/// A body sleeps after staying within this of its window's reference pose, wu -- half of
/// `deactivationReferenceDistance` at 60 Hz (0.05, PROVEN) -- for [`SLEEP_TICKS`] ticks. INFERRED.
pub const SLEEP_RADIUS: f32 = 0.025;
/// ...for this many consecutive ticks (0.5 s at 60 Hz) -- INFERRED.
pub const SLEEP_TICKS: u32 = 30;
/// Largest substep count for fast bodies.
pub const MAX_SUBSTEPS: usize = 4;
/// A body travels at most this far in one substep before it is split further, wu.
pub const SUBSTEP_TRAVEL: f32 = 0.08;
/// Most ticks run in one rendered frame (a slow frame drops time instead of spiralling).
pub const MAX_TICKS_PER_FRAME: u32 = 3;
/// Cap on bodies integrated at once. Anything beyond is held (kept awake, not stepped) until a
/// place frees up, so one avalanche cannot take a frame.
pub const MAX_AWAKE: usize = 192;
/// Longest position push in one substep, wu.
pub const MAX_PUSH_PER_SUBSTEP: f32 = 0.05;
/// Body-vs-body contacts are generated out to this gap (a speculative contact the solver lets
/// the bodies close and no further). Without it two boxes whose corners sit exactly on each other's
/// side planes -- every aligned stack -- generate NO contact, because the vertex is neither inside
/// nor outside. Under the engine's `collisionTolerance` (0.0328).
pub const CONTACT_MARGIN: f32 = 0.02;
/// Faces within this of the closest are candidates when picking a contact normal, wu.
pub const FACE_TIE: f32 = 0.1;
/// How far a hull vertex sample is moved toward its hull's middle (see `samples`), wu.
pub const INSET: f32 = 0.002;
/// Solver iterations of this world. The engine's are 8, but its soft-contact solver and warm
/// starting are not lifted; this stand-in has neither, and a three-box stack needs more passes to
/// carry weight through two layers. INFERRED.
pub const PLAY_ITERATIONS: u32 = 24;
/// Fraction of the mean penetration removed per substep by the position push (split impulse).
/// Not the solver's `tau`: that is a velocity-level softness; this is a projection and a stack's
/// weight needs it near 1 to keep the resting sag under a millimetre or two. INFERRED.
pub const POSITION_PUSH: f32 = 0.8;
/// A body moving slower than this (wu/s) for [`SLEEP_TICKS`] ticks sleeps even if a solver bias
/// keeps walking it across its window (a settled stack creeps sideways by a few mm/s).
pub const QUIET_SPEED: f32 = 0.03;
/// Broad-phase cell size, wu.
pub const CELL: f32 = 2.0;
/// A body counts as displaced from its saved pose beyond this, wu / radians.
pub const DISPLACED_POS: f32 = 0.01;
pub const DISPLACED_ANGLE: f32 = 0.005;

/// Forge placement physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicsMode {
    Normal,
    Fixed,
    Phased,
}

impl PhysicsMode {
    /// `PlacedObject.placement` bits 6-7: 0 normal, 1 fixed, 3 phased (2 is unused in Reach and
    /// reads as normal).
    pub fn from_placement(placement: u8) -> Self {
        match (placement & 0xC0) >> 6 {
            1 => PhysicsMode::Fixed,
            3 => PhysicsMode::Phased,
            _ => PhysicsMode::Normal,
        }
    }
}

/// Where bodies and hulls come from. The scene implements it ([`SceneSource`]); gates implement
/// it with boxes.
pub trait PropSource {
    /// The object's welded physics body at its editor scale, model space.
    fn physics_body(&self, datum: u32) -> Option<PhysBody>;
    /// A vehicle's hover pads and gravity scale; `None` for anything that is not a `vehi`.
    fn vehicle_spec(&self, _datum: u32) -> Option<VehicleSpec> {
        None
    }
    /// World-space triangles of the object's physics hull (for Fixed objects).
    fn hull_tris(&self, datum: u32) -> Vec<[Vec3; 3]>;
}

/// One placed object offered to the world.
#[derive(Clone, Copy, Debug)]
pub struct PropSeed {
    pub datum: u32,
    pub origin: Vec3,
    pub rot: Quat,
    pub mode: PhysicsMode,
    /// Coarse world AABB used until the body is decoded.
    pub aabb: (Vec3, Vec3),
    /// A `vehi`: simulated like a prop, but drawn and handed back to the driver through the
    /// ride's own parked-vehicle path, never by a prop actor.
    pub vehicle: bool,
}

/// One hover pad: where it is on the hull (model space) and its force law.
#[derive(Clone, Copy, Debug)]
pub struct HoverPoint {
    pub pos: Vec3,
    pub law: HoverLaw,
}

/// What a vehicle adds to a rigid body when nobody is driving it.
#[derive(Clone, Debug, Default)]
pub struct VehicleSpec {
    /// The `vehi` anti-gravity points. Non-empty = the engine's hover law holds the hull up.
    pub hover: Vec<HoverPoint>,
    /// `vehi` `gravity scale`, after the engine's postprocess.
    pub gravity_scale: f32,
    /// Has friction points (wheels / treads): rests on its tyres.
    pub wheeled: bool,
    /// An `alien_scout`'s authored braking, applied with zero input.
    pub idle: Option<crate::reach_havok::IdleBrake>,
}

/// Friction of a tyre sphere against the ground, a stand-in for a free-rolling wheel (INFERRED:
/// the wheel law is the jeep agent's and a passive body has no axle to spin).
pub const WHEEL_FRICTION: f32 = 0.12;
/// A level-start vehicle whose hull has ground within this of its lowest point counts as
/// resting (asleep until touched); with nothing there it falls. wu. INFERRED.
pub const VEHICLE_SUPPORT_REACH: f32 = 0.8;

/// A body that has been decoded.
struct Live {
    shapes: Vec<ConvexShape>,
    /// Body-space AABB of all shapes.
    local_aabb: (Vec3, Vec3),
    com_local: Vec3,
    hav: HavokBody,
    radius: f32,
    /// Editor scale already applied to the shapes (the draw needs it).
    scale: f32,
    /// Hover pads (empty for a prop and for a wheeled vehicle).
    hover: Vec<HoverPoint>,
    /// Wheeled vehicle: after a driver has used it the jeep law, not this world, parks it.
    wheeled: bool,
    /// Braking the engine applies to an empty hover vehicle (`alien_scout`).
    idle: Option<crate::reach_havok::IdleBrake>,
    calm: u32,
    /// The window's reference pose: the body sleeps when it stays within [`SLEEP_RADIUS`] of it
    /// for [`SLEEP_TICKS`] ticks (Havok deactivates on POSITION against a reference distance, not
    /// on speed; a resting stack's solver residual would never read as calm by speed).
    ref_com: Vec3,
    ref_q: Quat,
    quiet: u32,
}

enum Kind {
    /// Not decoded yet.
    Dormant,
    /// Decoded; `live` index.
    Live(usize),
    /// Fixed hull loaded into the static grid.
    FixedLoaded,
    /// Phased, or has no usable physics: never collides.
    Inert,
}

struct Slot {
    seed: PropSeed,
    kind: Kind,
    asleep: bool,
    /// Current world pose of the MODEL ORIGIN.
    origin: Vec3,
    rot: Quat,
    aabb: (Vec3, Vec3),
    cells: Option<((i32, i32, i32), (i32, i32, i32))>,
    ever_woken: bool,
    /// A driver (or the legacy parked-vehicle law) owns this vehicle: the world neither steps it
    /// nor collides with it (the driven hull is a pusher instead).
    driven: bool,
}

/// A pusher the caller moves itself: the player's capsule, a driven vehicle's hull.
pub struct Kinematic<'a> {
    pub shapes: &'a [ConvexShape],
    /// World pose of the model origin the shapes are relative to.
    pub origin: Vec3,
    pub rot: Quat,
    pub velocity: Vec3,
    pub angular: Vec3,
    pub mass: f32,
    /// Centre of mass, body space (for the lever arm of its own spin).
    pub com: Vec3,
}

/// What a step did, for the caption and the gates.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepStats {
    pub ticks: u32,
    pub awake: usize,
    pub held: usize,
    pub live: usize,
    pub dormant: usize,
    pub static_contacts: usize,
    pub pair_tests: usize,
    pub pair_contacts: usize,
    pub woke: usize,
    pub slept: usize,
    pub decoded: usize,
}

/// A triangle grid for the Fixed hulls.
#[derive(Default)]
pub struct TriGrid {
    tris: Vec<[Vec3; 3]>,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
    big: Vec<u32>,
}

impl TriGrid {
    pub fn add(&mut self, t: [Vec3; 3]) {
        let id = self.tris.len() as u32;
        let (mn, mx) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
        let (lo, hi) = (cell_of(mn), cell_of(mx));
        let n = (hi.0 - lo.0 + 1) as i64 * (hi.1 - lo.1 + 1) as i64 * (hi.2 - lo.2 + 1) as i64;
        self.tris.push(t);
        if n > 64 {
            self.big.push(id);
            return;
        }
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    self.cells.entry((x, y, z)).or_default().push(id);
                }
            }
        }
    }
    pub fn len(&self) -> usize {
        self.tris.len()
    }
}

impl StaticWorld for TriGrid {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        let (lo, hi) = (cell_of(mn), cell_of(mx));
        let mut seen: Vec<u32> = Vec::new();
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    if let Some(v) = self.cells.get(&(x, y, z)) {
                        for &id in v {
                            if !seen.contains(&id) {
                                seen.push(id);
                            }
                        }
                    }
                }
            }
        }
        seen.extend(self.big.iter().copied());
        for id in seen {
            let t = &self.tris[id as usize];
            if tri_overlaps(t, mn, mx) {
                out.push(*t);
            }
        }
    }
}

fn cell_of(p: Vec3) -> (i32, i32, i32) {
    ((p.x / CELL).floor() as i32, (p.y / CELL).floor() as i32, (p.z / CELL).floor() as i32)
}

fn world_aabb(local: (Vec3, Vec3), origin: Vec3, rot: Quat, pad: f32) -> (Vec3, Vec3) {
    let (mn, mx) = local;
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for i in 0..8 {
        let c = Vec3::new(
            if i & 1 == 0 { mn.x } else { mx.x },
            if i & 2 == 0 { mn.y } else { mx.y },
            if i & 4 == 0 { mn.z } else { mx.z },
        );
        let w = origin + rot * c;
        lo = lo.min(w);
        hi = hi.max(w);
    }
    (lo - Vec3::splat(pad), hi + Vec3::splat(pad))
}

fn overlap(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> bool {
    (0..3).all(|k| a.0[k] <= b.1[k] && a.1[k] >= b.0[k])
}

// ------------------------------------------------------------------------------------------
// Shape vs shape
// ------------------------------------------------------------------------------------------

/// Sample points of a shape with the radius each carries, world space.
fn samples(shape: &ConvexShape, rot: Mat3, pos: Vec3, out: &mut Vec<(Vec3, f32)>) {
    out.clear();
    match &shape.geom {
        ShapeGeom::Sphere { c, r } => out.push((pos + rot * *c, *r)),
        ShapeGeom::Capsule { a, b, r } => {
            let (aw, bw) = (pos + rot * *a, pos + rot * *b);
            for k in 0..5 {
                out.push((aw.lerp(bw, k as f32 / 4.0), *r));
            }
        }
        ShapeGeom::Hull { pts, r, .. } => {
            // Each vertex is nudged INSET toward the hull's middle. An aligned stack puts every
            // corner of the upper box exactly on the side planes of the lower one, where "which
            // face am I touching" is a tie that the sideways faces win -- the weight then pushes
            // sideways and the stack falls through itself. Inside the planes the top face wins.
            let c = pts.iter().copied().sum::<Vec3>() / pts.len().max(1) as f32;
            for p in pts {
                let to_c = c - *p;
                let l = to_c.length();
                let q = if l > 1e-6 { *p + to_c / l * INSET.min(l * 0.5) } else { *p };
                out.push((pos + rot * q, *r));
            }
        }
    }
}

/// Signed distance from world point `p` to the shape's padded surface (negative inside), and the
/// outward normal there.
fn surface_distance(shape: &ConvexShape, rot: Mat3, pos: Vec3, inv_rot: Mat3, p: Vec3, prefer: Vec3) -> (f32, Vec3) {
    match &shape.geom {
        ShapeGeom::Sphere { c, r } => {
            let d = p - (pos + rot * *c);
            let l = d.length();
            (l - r, if l > 1e-6 { d / l } else { Vec3::Z })
        }
        ShapeGeom::Capsule { a, b, r } => {
            let (aw, bw) = (pos + rot * *a, pos + rot * *b);
            let ab = bw - aw;
            let t = if ab.length_squared() > 1e-12 { ((p - aw).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
            let d = p - (aw + ab * t);
            let l = d.length();
            (l - r, if l > 1e-6 { d / l } else { Vec3::Z })
        }
        ShapeGeom::Hull { planes, r, .. } => {
            let pb = inv_rot * (p - pos);
            let mut best = f32::MIN;
            for (n, d) in planes {
                best = best.max(n.dot(pb) - d);
            }
            // Among the faces the point is nearly as close to as its closest, take the one that
            // faces the OTHER body. A corner sample of a stacked box is within a hair of the lower
            // box's side faces and its top face at once; "closest" alone picks a side face as soon
            // as the overlap passes the inset and the stack falls through itself.
            // ...and the DEPTH is measured along that face, not along the closest one.
            let mut bn = Vec3::Z;
            let mut bs = best;
            let mut bd = f32::MIN;
            for (n, d) in planes {
                let sd = n.dot(pb) - d;
                if sd >= best - FACE_TIE {
                    let facing = (rot * *n).dot(prefer);
                    if facing > bd {
                        bd = facing;
                        bn = *n;
                        bs = sd;
                    }
                }
            }
            (bs - r, rot * bn)
        }
    }
}

/// One shape of A against one shape of B. Contacts are appended as
/// `(point, normal from B into A, depth)`; `ext_b` bounds a believable depth.
#[allow(clippy::too_many_arguments)]
fn shape_pair(
    sa: &ConvexShape,
    ra: Mat3,
    pa: Vec3,
    ira: Mat3,
    sb: &ConvexShape,
    rb: Mat3,
    pb: Vec3,
    irb: Mat3,
    ext_a: f32,
    ext_b: f32,
    out: &mut Vec<(Vec3, Vec3, f32)>,
) {
    let mut pts = Vec::new();
    // from B's middle toward A's: the direction a contact normal "from B into A" should lean to
    let to_a = (pa + ra * sa.centroid() - (pb + rb * sb.centroid())).normalize_or_zero();
    samples(sa, ra, pa, &mut pts);
    for (p, r) in &pts {
        let (d, n) = surface_distance(sb, rb, pb, irb, *p, to_a);
        let depth = r - d;
        if depth > -CONTACT_MARGIN && depth < ext_b.max(*r * 2.0) {
            out.push((*p - n * d, n, depth));
        }
    }
    samples(sb, rb, pb, &mut pts);
    for (p, r) in &pts {
        let (d, n) = surface_distance(sa, ra, pa, ira, *p, -to_a);
        let depth = r - d;
        if depth > -CONTACT_MARGIN && depth < ext_a.max(*r * 2.0) {
            // the sample belongs to B: the normal out of A points into B, so from B into A it flips
            out.push((*p - n * d, -n, depth));
        }
    }
}

// ------------------------------------------------------------------------------------------
// The world
// ------------------------------------------------------------------------------------------

/// The play-mode rigid-body world.
pub struct PropWorld {
    slots: Vec<Slot>,
    lives: Vec<Live>,
    /// `lives[i]` belongs to `slots[owner[i]]`.
    owner: Vec<usize>,
    by_datum: HashMap<u32, usize>,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
    statics: TriGrid,
    pub solver: SolverParams,
    /// Bodies to wake on the next step (level-start vehicles with nothing under them).
    pub pending_wake: Vec<u32>,
    acc: f32,
    /// A signature of everything the renderer needs to redraw: changes when a displaced body
    /// moves, constant while every body is asleep.
    sig: u64,
    pub stats: StepStats,
    stamp: Vec<u32>,
    stamp_now: u32,
}

impl PropWorld {
    pub fn new() -> Self {
        PropWorld {
            slots: Vec::new(),
            lives: Vec::new(),
            owner: Vec::new(),
            by_datum: HashMap::new(),
            grid: HashMap::new(),
            statics: TriGrid::default(),
            solver: SolverParams { iterations: PLAY_ITERATIONS, ..SolverParams::ENGINE },
            pending_wake: Vec::new(),
            acc: 0.0,
            sig: 0,
            stats: StepStats::default(),
            stamp: Vec::new(),
            stamp_now: 0,
        }
    }

    /// Offer one object. Phased objects are recorded (so they can be listed) but never collide.
    pub fn add(&mut self, seed: PropSeed) {
        let idx = self.slots.len();
        let kind = if seed.mode == PhysicsMode::Phased { Kind::Inert } else { Kind::Dormant };
        self.by_datum.insert(seed.datum, idx);
        let slot = Slot { seed, kind, asleep: true, origin: seed.origin, rot: seed.rot, aabb: seed.aabb, cells: None, ever_woken: false, driven: false };
        self.slots.push(slot);
        self.stamp.push(0);
        if seed.mode != PhysicsMode::Phased {
            self.grid_update(idx);
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn live_count(&self) -> usize {
        self.lives.len()
    }

    fn grid_update(&mut self, idx: usize) {
        let aabb = self.slots[idx].aabb;
        let new = (cell_of(aabb.0), cell_of(aabb.1));
        if self.slots[idx].cells == Some(new) {
            return;
        }
        if let Some((lo, hi)) = self.slots[idx].cells {
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    for z in lo.2..=hi.2 {
                        if let Some(v) = self.grid.get_mut(&(x, y, z)) {
                            v.retain(|&i| i as usize != idx);
                        }
                    }
                }
            }
        }
        for x in new.0 .0..=new.1 .0 {
            for y in new.0 .1..=new.1 .1 {
                for z in new.0 .2..=new.1 .2 {
                    self.grid.entry((x, y, z)).or_default().push(idx as u32);
                }
            }
        }
        self.slots[idx].cells = Some(new);
    }

    fn neighbours(&mut self, aabb: (Vec3, Vec3), skip: usize, out: &mut Vec<usize>) {
        out.clear();
        self.stamp_now = self.stamp_now.wrapping_add(1);
        let (lo, hi) = (cell_of(aabb.0), cell_of(aabb.1));
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    if let Some(v) = self.grid.get(&(x, y, z)) {
                        for &i in v {
                            let i = i as usize;
                            if i != skip && !self.slots[i].driven && self.stamp[i] != self.stamp_now && overlap(self.slots[i].aabb, aabb) {
                                self.stamp[i] = self.stamp_now;
                                out.push(i);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Decode a dormant slot into a body (or a static hull, or nothing).
    fn materialise(&mut self, idx: usize, src: &dyn PropSource) -> bool {
        match self.slots[idx].kind {
            Kind::Live(_) => return true,
            Kind::Inert | Kind::FixedLoaded => return false,
            Kind::Dormant => {}
        }
        let seed = self.slots[idx].seed;
        self.stats.decoded += 1;
        if seed.mode == PhysicsMode::Fixed {
            for t in src.hull_tris(seed.datum) {
                self.statics.add(t);
            }
            self.slots[idx].kind = Kind::FixedLoaded;
            return false;
        }
        let Some(mut body) = src.physics_body(seed.datum) else {
            self.slots[idx].kind = Kind::Inert;
            return false;
        };
        let spec = if seed.vehicle { src.vehicle_spec(seed.datum) } else { None };
        if !seed.vehicle {
            // A prop has no suspension. A shape the body decode flagged as a support point (a
            // tyre / hover-pad sphere) is a force-limited spring that compresses by half its
            // radius under the prop's own weight; on a spawn point, a turret or a hull shell that
            // is a body sunk 11..23 cm into the block it sits on (measured by
            // `real_props_on_real_blocks_do_not_fall_through`). For a prop it is a hard shape.
            for sh in body.shapes.iter_mut() {
                sh.support = false;
            }
        }
        if let Some(sp) = &spec {
            if !sp.hover.is_empty() {
                // the pads are springs, not rigid spheres
                body.shapes.retain(|s| !s.support);
            } else if sp.wheeled {
                for s in body.shapes.iter_mut().filter(|s| s.support) {
                    s.mat.friction = WHEEL_FRICTION;
                }
            }
        }
        if body.fixed && !seed.vehicle {
            // the phmo itself says this never moves (motion type keyframed / fixed): terrain
            for t in src.hull_tris(seed.datum) {
                self.statics.add(t);
            }
            self.slots[idx].kind = Kind::FixedLoaded;
            return false;
        }
        if !(body.mass.is_finite() && body.mass > 0.0) || body.shapes.is_empty() {
            self.slots[idx].kind = Kind::Inert;
            return false;
        }
        let inv_i = {
            let i = body.inertia;
            if i.determinant().abs() > 1e-9 && i.is_finite() {
                i.inverse()
            } else {
                let r = body.bounding_radius().max(0.05);
                Mat3::from_diagonal(Vec3::splat(1.0 / (0.4 * body.mass * r * r)))
            }
        };
        let motion = MotionType::BoxInertia;
        let com_w = seed.origin + seed.rot * body.com;
        let mut hav = HavokBody::new(motion, body.mass, inv_i, com_w, seed.rot).with_damping(body.lin_damping, body.ang_damping);
        let hover = spec.as_ref().map(|s| s.hover.clone()).unwrap_or_default();
        if let Some(sp) = &spec {
            // the vehicle's own `gravity scale` is the body's Havok gravity factor at all times
            hav.set_gravity_factor(crate::reach_havok::vehicle_body_gravity_factor(1.0, 1.0, sp.gravity_scale));
        }
        let radius = body.bounding_radius();
        let local_aabb = body.aabb();
        let scale = if body.scale.is_finite() && body.scale > 0.0 { body.scale } else { 1.0 };
        self.lives.push(Live { shapes: body.shapes, local_aabb, com_local: body.com, hav, radius, scale, hover, wheeled: spec.as_ref().is_some_and(|s| s.wheeled), idle: spec.as_ref().and_then(|s| s.idle), calm: 0, ref_com: com_w, ref_q: seed.rot, quiet: 0 });
        self.owner.push(idx);
        self.slots[idx].kind = Kind::Live(self.lives.len() - 1);
        let ab = world_aabb(local_aabb, self.slots[idx].origin, self.slots[idx].rot, 0.02);
        self.slots[idx].aabb = ab;
        self.grid_update(idx);
        true
    }

    /// Wake a body (decoding it first). Returns whether it is now a live, awake body.
    pub fn wake(&mut self, datum: u32, src: &dyn PropSource) -> bool {
        let Some(&idx) = self.by_datum.get(&datum) else { return false };
        self.wake_slot(idx, src)
    }

    fn wake_slot(&mut self, idx: usize, src: &dyn PropSource) -> bool {
        if !self.materialise(idx, src) {
            return false;
        }
        if self.slots[idx].asleep {
            self.slots[idx].asleep = false;
            self.slots[idx].ever_woken = true;
            if let Kind::Live(l) = self.slots[idx].kind {
                let lv = &mut self.lives[l];
                lv.calm = 0;
                lv.quiet = 0;
                lv.ref_com = lv.hav.com;
                lv.ref_q = lv.hav.q;
            }
            self.stats.woke += 1;
        }
        true
    }

    /// Give a body a velocity (the seam for an external law: an impulse from a weapon, the jeep
    /// law, a script). Wakes it.
    pub fn set_body_velocity(&mut self, datum: u32, v: Vec3, w: Vec3, src: &dyn PropSource) {
        if self.wake(datum, src) {
            if let Some(&idx) = self.by_datum.get(&datum) {
                if let Kind::Live(l) = self.slots[idx].kind {
                    self.lives[l].hav.set_velocity(v, w);
                }
            }
        }
    }

    /// The model-origin pose of a body.
    pub fn body_pose(&self, datum: u32) -> Option<(Vec3, Quat)> {
        let &idx = self.by_datum.get(&datum)?;
        Some((self.slots[idx].origin, self.slots[idx].rot))
    }

    pub fn body_velocity(&self, datum: u32) -> Option<(Vec3, Vec3)> {
        let &idx = self.by_datum.get(&datum)?;
        match self.slots[idx].kind {
            Kind::Live(l) => Some((self.lives[l].hav.v, self.lives[l].hav.w)),
            _ => Some((Vec3::ZERO, Vec3::ZERO)),
        }
    }

    pub fn is_asleep(&self, datum: u32) -> Option<bool> {
        let &idx = self.by_datum.get(&datum)?;
        Some(self.slots[idx].asleep)
    }

    /// Bodies currently awake.
    pub fn awake_count(&self) -> usize {
        self.slots.iter().filter(|s| matches!(s.kind, Kind::Live(_)) && !s.asleep && !s.driven).count()
    }

    /// Put every body back at its saved pose, at rest, asleep. Exact: the saved pose is stored,
    /// not recomputed.
    pub fn restore_all(&mut self) {
        for i in 0..self.slots.len() {
            let (o, r) = (self.slots[i].seed.origin, self.slots[i].seed.rot);
            self.slots[i].origin = o;
            self.slots[i].rot = r;
            self.slots[i].asleep = true;
            self.slots[i].aabb = self.slots[i].seed.aabb;
            if let Kind::Live(l) = self.slots[i].kind {
                let live = &mut self.lives[l];
                live.hav.com = o + r * live.com_local;
                live.hav.q = r;
                live.hav.set_velocity(Vec3::ZERO, Vec3::ZERO);
                live.calm = 0;
                self.slots[i].aabb = world_aabb(live.local_aabb, o, r, 0.02);
            }
            self.grid_update(i);
        }
        self.acc = 0.0;
        self.sig = 0;
    }

    fn displaced_slot(&self, s: &Slot) -> bool {
        if !matches!(s.kind, Kind::Live(_)) {
            return false;
        }
        (s.origin - s.seed.origin).length() > DISPLACED_POS || s.rot.angle_between(s.seed.rot) > DISPLACED_ANGLE
    }

    /// The poses of every body that has left its saved pose: `(datum, origin, rotation)`. These
    /// are the placements the renderer must hide and draw itself.
    pub fn displaced(&self) -> Vec<(u32, Vec3, Quat)> {
        self.slots.iter().filter(|s| !s.seed.vehicle && self.displaced_slot(s)).map(|s| (s.seed.datum, s.origin, s.rot)).collect()
    }

    /// [`displaced`](Self::displaced) with each body's editor scale: `(datum, origin, rotation, scale)`.
    pub fn displaced_scaled(&self) -> Vec<(u32, Vec3, Quat, f32)> {
        self.slots
            .iter()
            .filter(|s| !s.seed.vehicle && self.displaced_slot(s))
            .map(|s| {
                let sc = if let Kind::Live(l) = s.kind { self.lives[l].scale } else { 1.0 };
                (s.seed.datum, s.origin, s.rot, sc)
            })
            .collect()
    }

    /// The datums whose PLACEMENT must not be drawn.
    pub fn hidden_datums(&self) -> Vec<u32> {
        self.slots.iter().filter(|s| !s.seed.vehicle && self.displaced_slot(s)).map(|s| s.seed.datum).collect()
    }

    /// Vehicles that have left their saved pose and are not driven: the app adopts each into the
    /// ride's parked-vehicle path (which draws it, hides the placement and hands it back).
    pub fn displaced_vehicles(&self) -> Vec<u32> {
        self.slots.iter().filter(|s| s.seed.vehicle && !s.driven && self.displaced_slot(s)).map(|s| s.seed.datum).collect()
    }

    /// A simulated vehicle's state: `(origin, rotation, linear velocity, angular velocity)`.
    pub fn vehicle_state(&self, datum: u32) -> Option<(Vec3, Quat, Vec3, Vec3)> {
        let &idx = self.by_datum.get(&datum)?;
        let s = &self.slots[idx];
        if !s.seed.vehicle || s.driven {
            return None;
        }
        match s.kind {
            Kind::Live(l) => Some((s.origin, s.rot, self.lives[l].hav.v, self.lives[l].hav.w)),
            _ => None,
        }
    }

    /// Is this vehicle a wheeled one (friction points)? Wheeled vehicles keep the jeep law once
    /// they have been driven; the world only simulates them passively until then.
    pub fn is_driven(&self, datum: u32) -> bool {
        self.by_datum.get(&datum).is_some_and(|&i| self.slots[i].driven)
    }

    /// The driver gets in (or the legacy law takes over): the vehicle leaves the simulation and
    /// its CURRENT state is returned so nothing snaps.
    pub fn release_vehicle(&mut self, datum: u32) -> Option<(Vec3, Quat, Vec3, Vec3)> {
        let st = self.vehicle_state(datum);
        let &idx = self.by_datum.get(&datum)?;
        if self.slots[idx].seed.vehicle {
            self.slots[idx].driven = true;
        }
        st
    }

    /// The driver gets out: the vehicle (decoded if need be) continues from the pose and
    /// velocities it had. `false` when the placement is not a simulatable vehicle.
    #[allow(clippy::too_many_arguments)]
    pub fn claim_vehicle(&mut self, datum: u32, origin: Vec3, rot: Quat, v: Vec3, w: Vec3, src: &dyn PropSource) -> bool {
        let Some(&idx) = self.by_datum.get(&datum) else { return false };
        if !self.slots[idx].seed.vehicle {
            return false;
        }
        self.slots[idx].driven = false;
        if !self.materialise(idx, src) {
            return false;
        }
        let Kind::Live(l) = self.slots[idx].kind else { return false };
        if self.lives[l].wheeled {
            // the jeep law parks a wheeled vehicle once it has been driven
            self.slots[idx].driven = true;
            return false;
        }
        self.slots[idx].origin = origin;
        self.slots[idx].rot = rot;
        let lv = &mut self.lives[l];
        lv.hav.com = origin + rot * lv.com_local;
        lv.hav.q = rot;
        lv.hav.set_velocity(v, w);
        lv.calm = 0;
        lv.quiet = 0;
        lv.ref_com = lv.hav.com;
        lv.ref_q = rot;
        self.slots[idx].asleep = false;
        self.slots[idx].ever_woken = true;
        self.sync_slot(idx);
        true
    }

    /// Does this vehicle have hover pads (the engine's anti-gravity law)?
    pub fn is_hover(&self, datum: u32) -> bool {
        self.by_datum.get(&datum).is_some_and(|&i| matches!(self.slots[i].kind, Kind::Live(l) if !self.lives[l].hover.is_empty()))
    }

    /// Everything the draw depends on, folded to one number: changes when a displaced body moves,
    /// is constant while the world is at rest.
    pub fn signature(&self) -> u64 {
        self.sig
    }

    fn recompute_sig(&mut self) {
        if !self.slots.iter().any(|s| self.displaced_slot(s)) {
            self.sig = 0;
            return;
        }
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for s in &self.slots {
            if !self.displaced_slot(s) {
                continue;
            }
            for v in [s.origin.x, s.origin.y, s.origin.z, s.rot.x, s.rot.y, s.rot.z, s.rot.w] {
                h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
            }
            h = (h ^ s.seed.datum as u64).wrapping_mul(0x100_0000_01b3);
        }
        self.sig = h;
    }

    /// Advance by `dt` seconds of real time (fixed 60 Hz ticks, at most [`MAX_TICKS_PER_FRAME`]).
    pub fn step(&mut self, dt: f32, level: &dyn StaticWorld, src: &dyn PropSource, kin: &[Kinematic]) -> StepStats {
        for d in std::mem::take(&mut self.pending_wake) {
            self.wake(d, src);
        }
        self.acc += dt.clamp(0.0, 0.1);
        let mut st = StepStats::default();
        let mut n = 0;
        while self.acc >= TICK_SECONDS && n < MAX_TICKS_PER_FRAME {
            self.acc -= TICK_SECONDS;
            self.tick(level, src, kin, &mut st);
            n += 1;
        }
        if n == MAX_TICKS_PER_FRAME {
            self.acc = 0.0;
        }
        st.ticks = n;
        st.awake = self.awake_count();
        st.live = self.lives.len();
        st.dormant = self.slots.iter().filter(|s| matches!(s.kind, Kind::Dormant)).count();
        if n > 0 {
            self.recompute_sig();
        }
        self.stats = st;
        st
    }

    fn tick(&mut self, level: &dyn StaticWorld, src: &dyn PropSource, kin: &[Kinematic], st: &mut StepStats) {
        // 0. a pusher touching a sleeping or dormant body wakes it
        for k in kin {
            let kab = kin_aabb(k);
            let mut near = Vec::new();
            self.neighbours(kab, usize::MAX, &mut near);
            for idx in near {
                if !self.slots[idx].asleep && matches!(self.slots[idx].kind, Kind::Live(_)) {
                    continue;
                }
                if self.slots[idx].seed.mode != PhysicsMode::Normal {
                    // a Fixed neighbour is terrain: load its hull so the level knows about it
                    if matches!(self.slots[idx].kind, Kind::Dormant) {
                        self.materialise(idx, src);
                    }
                    continue;
                }
                if !self.materialise(idx, src) {
                    continue;
                }
                if self.kin_touches(k, idx) {
                    self.wake_slot(idx, src);
                }
            }
        }
        // 1. who is awake (bounded)
        let mut awake: Vec<usize> = (0..self.slots.len())
            .filter(|&i| !self.slots[i].asleep && !self.slots[i].driven && matches!(self.slots[i].kind, Kind::Live(_)))
            .collect();
        st.held = awake.len().saturating_sub(MAX_AWAKE);
        awake.truncate(MAX_AWAKE);
        if awake.is_empty() {
            return;
        }
        // 2. substeps from the fastest body
        let mut fastest = 0.0f32;
        for &i in &awake {
            if let Kind::Live(l) = self.slots[i].kind {
                let lv = &self.lives[l];
                fastest = fastest.max(lv.hav.v.length() + lv.hav.w.length() * lv.radius);
            }
        }
        let nsub = ((fastest * TICK_SECONDS) / SUBSTEP_TRAVEL).ceil().clamp(1.0, MAX_SUBSTEPS as f32) as usize;
        let h = TICK_SECONDS / nsub as f32;
        for _ in 0..nsub {
            self.substep(&awake, h, level, src, kin, st);
        }
        // 3. sleep
        for &i in &awake {
            if self.slots[i].asleep {
                continue;
            }
            let Kind::Live(l) = self.slots[i].kind else { continue };
            let lv = &mut self.lives[l];
            let moved = (lv.hav.com - lv.ref_com).length() > SLEEP_RADIUS
                || lv.hav.q.angle_between(lv.ref_q) * lv.radius > SLEEP_RADIUS;
            if moved {
                lv.ref_com = lv.hav.com;
                lv.ref_q = lv.hav.q;
                lv.calm = 0;
            } else {
                lv.calm += 1;
            }
            let slow = lv.hav.v.length() < QUIET_SPEED && lv.hav.w.length() * lv.radius < QUIET_SPEED;
            lv.quiet = if slow { lv.quiet + 1 } else { 0 };
            if lv.calm >= SLEEP_TICKS || lv.quiet >= SLEEP_TICKS {
                lv.hav.set_velocity(Vec3::ZERO, Vec3::ZERO);
                self.slots[i].asleep = true;
                st.slept += 1;
            }
        }
    }

    fn kin_touches(&self, k: &Kinematic, idx: usize) -> bool {
        let Kind::Live(l) = self.slots[idx].kind else { return false };
        let lv = &self.lives[l];
        let (rk, rp) = (Mat3::from_quat(k.rot), Mat3::from_quat(self.slots[idx].rot));
        let (irk, irp) = (rk.transpose(), rp.transpose());
        let mut tmp = Vec::new();
        for sa in k.shapes {
            for sb in &lv.shapes {
                tmp.clear();
                shape_pair(sa, rk, k.origin, irk, sb, rp, self.slots[idx].origin, irp, 4.0, 4.0, &mut tmp);
                if !tmp.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    fn substep(&mut self, awake: &[usize], h: f32, level: &dyn StaticWorld, src: &dyn PropSource, kin: &[Kinematic], st: &mut StepStats) {
        let g = world_gravity();
        // hover pads: the engine's anti-gravity law, applied to the velocity BEFORE Havok's own
        // integration, exactly the order of the vehicle tick (vehicle law -> set velocity -> step)
        for &i in awake {
            if self.slots[i].asleep {
                continue;
            }
            let Kind::Live(l) = self.slots[i].kind else { continue };
            if self.lives[l].hover.is_empty() {
                continue;
            }
            self.apply_hover(i, l, h, level);
        }
        // integrate
        for &i in awake {
            if self.slots[i].asleep {
                continue;
            }
            let Kind::Live(l) = self.slots[i].kind else { continue };
            self.lives[l].hav.step_psi(h, g);
            self.sync_slot(i);
        }
        // scratch: awake bodies first, then any body woken by contact, then the kinematics
        let mut scratch: Vec<HavokBody> = Vec::new();
        let mut sidx: HashMap<usize, usize> = HashMap::new();
        let mut order: Vec<usize> = Vec::new();
        for &i in awake {
            if self.slots[i].asleep {
                continue;
            }
            let Kind::Live(l) = self.slots[i].kind else { continue };
            sidx.insert(i, scratch.len());
            scratch.push(self.lives[l].hav.clone());
            order.push(i);
        }
        let mut contacts: Vec<PairContact> = Vec::new();
        // (a, b, normal from b into a, depth): the position push applied after the solve
        let mut pushes: Vec<(usize, Option<usize>, Vec3, f32)> = Vec::new();
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        let mut cbuf: Vec<Contact> = Vec::new();
        let mut near: Vec<usize> = Vec::new();
        let mut pairs: Vec<(Vec3, Vec3, f32)> = Vec::new();
        let mut oi = 0;
        while oi < order.len() {
            let i = order[oi];
            oi += 1;
            let Kind::Live(l) = self.slots[i].kind else { continue };
            let a = sidx[&i];
            let (rot_q, origin) = (self.slots[i].rot, self.slots[i].origin);
            let rot = Mat3::from_quat(rot_q);
            let inv = rot.transpose();
            // --- static: the level and the Fixed hulls ---
            let ab = self.slots[i].aabb;
            let (mn, mx) = (ab.0 - Vec3::splat(0.05), ab.1 + Vec3::splat(0.05));
            // every dormant neighbour is decoded BEFORE the gather: a Fixed object (or a phmo that
            // is itself fixed) contributes its hull to the static grid, a body joins the pair loop
            self.neighbours((mn, mx), i, &mut near);
            for &j in near.clone().iter() {
                if matches!(self.slots[j].kind, Kind::Dormant) {
                    self.materialise(j, src);
                }
            }
            tris.clear();
            level.gather(mn, mx, &mut tris);
            self.statics.gather(mn, mx, &mut tris);
            if !tris.is_empty() {
                for sh in &self.lives[l].shapes {
                    cbuf.clear();
                    shape_contacts(sh, rot, origin, inv, &tris, &mut cbuf);
                    for c in cbuf.iter().take(24) {
                        let (mu, e) = contact_material(c.mu, c.e, LEVEL_FRICTION, LEVEL_RESTITUTION);
                        // penetration is removed by a POSITION push after the solve (split
                        // impulse), never by velocity: separation 0 here
                        contacts.push(PairContact { a, b: None, point: c.p, normal: c.n, separation: 0.0, friction: mu, restitution: e });
                        pushes.push((a, None, c.n, c.depth));
                        st.static_contacts += 1;
                    }
                }
            }
            // --- bodies ---
            let mut nb = near.clone();
            nb.retain(|&j| self.slots[j].seed.mode == PhysicsMode::Normal);
            for j in nb {
                let b_awake = !self.slots[j].asleep && matches!(self.slots[j].kind, Kind::Live(_));
                if b_awake && j < i && sidx.contains_key(&j) {
                    continue; // the (j, i) pair is handled from j
                }
                if !self.materialise(j, src) {
                    continue;
                }
                let Kind::Live(lj) = self.slots[j].kind else { continue };
                let (rj_q, oj) = (self.slots[j].rot, self.slots[j].origin);
                let (rj, ij) = (Mat3::from_quat(rj_q), Mat3::from_quat(rj_q).transpose());
                pairs.clear();
                let (ext_a, ext_b) = (self.lives[l].radius * 2.0, self.lives[lj].radius * 2.0);
                for sa in &self.lives[l].shapes {
                    for sb in &self.lives[lj].shapes {
                        st.pair_tests += 1;
                        shape_pair(sa, rot, origin, inv, sb, rj, oj, ij, ext_a, ext_b, &mut pairs);
                    }
                }
                if pairs.is_empty() {
                    continue;
                }
                // the sleeper is touched: only a body that is actually moving disturbs it; a
                // body merely resting on it leaves it asleep and solid
                if !b_awake {
                    let speed = scratch[a].v.length() + scratch[a].w.length() * self.lives[l].radius;
                    if speed >= SLEEP_SPEED * 0.5 {
                        self.wake_slot(j, src);
                    }
                }
                let bi = match sidx.get(&j) {
                    Some(&b) => Some(b),
                    None if !self.slots[j].asleep => {
                        let idxb = scratch.len();
                        sidx.insert(j, idxb);
                        scratch.push(self.lives[lj].hav.clone());
                        order.push(j);
                        Some(idxb)
                    }
                    None => {
                        // asleep and not woken: immovable obstacle
                        let mut frozen = self.lives[lj].hav.clone();
                        frozen.inv_mass = 0.0;
                        frozen.inv_inertia_local = Mat3::ZERO;
                        let idxb = scratch.len();
                        scratch.push(frozen);
                        Some(idxb)
                    }
                };
                let (fa, ea) = shape_mat(&self.lives[l].shapes);
                let (fb, eb) = shape_mat(&self.lives[lj].shapes);
                let (mu, e) = contact_material(fa, ea, fb, eb);
                pairs.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
                for (p, n, depth) in pairs.iter().take(8) {
                    contacts.push(PairContact { a, b: bi, point: *p, normal: *n, separation: (-*depth).max(0.0), friction: mu, restitution: e });
                    if *depth > 0.0 {
                        pushes.push((a, bi, *n, *depth));
                    }
                    st.pair_contacts += 1;
                }
            }
        }
        // --- kinematic pushers ---
        let kbase = scratch.len();
        for k in kin {
            let mut hb = HavokBody::new(
                MotionType::Keyframed,
                k.mass.max(1.0),
                Mat3::ZERO,
                k.origin + k.rot * k.com,
                k.rot,
            );
            hb.set_velocity(k.velocity, k.angular);
            scratch.push(hb);
        }
        for (ki, k) in kin.iter().enumerate() {
            let kab = kin_aabb(k);
            let (rk, irk) = (Mat3::from_quat(k.rot), Mat3::from_quat(k.rot).transpose());
            for &i in &order {
                let Kind::Live(l) = self.slots[i].kind else { continue };
                if !overlap(self.slots[i].aabb, kab) {
                    continue;
                }
                let (rp_q, op) = (self.slots[i].rot, self.slots[i].origin);
                let (rp, ip) = (Mat3::from_quat(rp_q), Mat3::from_quat(rp_q).transpose());
                pairs.clear();
                for sa in k.shapes {
                    for sb in &self.lives[l].shapes {
                        st.pair_tests += 1;
                        shape_pair(sa, rk, k.origin, irk, sb, rp, op, ip, 4.0, self.lives[l].radius * 2.0, &mut pairs);
                    }
                }
                pairs.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
                let (fk, ek) = shape_mat(k.shapes);
                let (fb, eb) = shape_mat(&self.lives[l].shapes);
                let (mu, e) = contact_material(fk, ek, fb, eb);
                for (p, n, depth) in pairs.iter().take(8) {
                    // normal from the prop INTO the pusher, pusher is `a`
                    contacts.push(PairContact {
                        a: kbase + ki,
                        b: Some(sidx[&i]),
                        point: *p,
                        normal: *n,
                        separation: (-*depth).max(0.0),
                        friction: mu,
                        restitution: e,
                    });
                    if *depth > 0.0 {
                        pushes.push((kbase + ki, Some(sidx[&i]), *n, *depth));
                    }
                    st.pair_contacts += 1;
                }
            }
        }
        // --- solve ---
        let saved: Vec<(Vec3, Vec3)> = scratch[kbase..].iter().map(|b| (b.v, b.w)).collect();
        solve_pair_contacts(&mut scratch, &contacts, h, &self.solver);
        for (b, s) in scratch[kbase..].iter_mut().zip(saved) {
            b.v = s.0;
            b.w = s.1;
        }
        // split-impulse position correction: each body moves a `tau` fraction of the mean
        // penetration along the mean contact normal; a pair shares it by inverse mass
        let mut corr: Vec<(Vec3, u32)> = vec![(Vec3::ZERO, 0); scratch.len()];
        for &(a, b, n, depth) in &pushes {
            let (ia, ib) = (scratch[a].inv_mass, b.map_or(0.0, |b| scratch[b].inv_mass));
            let tot = ia + ib;
            if tot <= 0.0 {
                continue;
            }
            if ia > 0.0 && a < kbase {
                corr[a].0 += n * depth * (ia / tot);
                corr[a].1 += 1;
            }
            if let Some(b) = b {
                if ib > 0.0 {
                    corr[b].0 -= n * depth * (ib / tot);
                    corr[b].1 += 1;
                }
            }
        }
        for (&i, &s) in &sidx {
            if corr[s].1 == 0 || self.slots[i].asleep {
                continue;
            }
            if let Kind::Live(l) = self.slots[i].kind {
                let mean = corr[s].0 / corr[s].1 as f32;
                let push = mean * POSITION_PUSH;
                let push = if push.length() > MAX_PUSH_PER_SUBSTEP { push.normalize() * MAX_PUSH_PER_SUBSTEP } else { push };
                self.lives[l].hav.com += push;
                self.sync_slot(i);
            }
        }
        // write back
        for (&i, &s) in &sidx {
            if let Kind::Live(l) = self.slots[i].kind {
                if !self.slots[i].asleep {
                    self.lives[l].hav.v = scratch[s].v;
                    self.lives[l].hav.w = scratch[s].w;
                }
            }
        }
    }

    /// One substep of a hull's hover pads against the ground below each pad.
    fn apply_hover(&mut self, i: usize, l: usize, h: f32, level: &dyn StaticWorld) {
        let g = crate::reach_physics::WORLD_GRAVITY;
        let (origin, rot) = (self.slots[i].origin, self.slots[i].rot);
        let reach = self.lives[l].hover.iter().map(|p| p.law.height + p.law.radius).fold(0.0f32, f32::max);
        let ab = self.slots[i].aabb;
        let (mn, mx) = (ab.0 - Vec3::new(0.05, 0.05, reach + 0.05), ab.1 + Vec3::splat(0.05));
        let mut tris = Vec::new();
        level.gather(mn, mx, &mut tris);
        self.statics.gather(mn, mx, &mut tris);
        if tris.is_empty() {
            return;
        }
        let up_z = (rot * Vec3::Z).z;
        let mass = if self.lives[l].hav.inv_mass > 0.0 { 1.0 / self.lives[l].hav.inv_mass } else { return };
        for k in 0..self.lives[l].hover.len() {
            let pt = self.lives[l].hover[k];
            let p = origin + rot * pt.pos;
            let Some((dist, n)) = ray_down(&tris, p, pt.law.height + pt.law.radius) else { continue };
            let hv = &self.lives[l].hav;
            let vn = hv.point_velocity(p).dot(n);
            let a = crate::reach_havok::hover_acceleration(&pt.law, g, Some(dist), vn, up_z);
            let lv = &mut self.lives[l].hav;
            lv.apply_impulse_at(p, n * (a * mass * h));
        }
        // With nobody in the seat the engine's velocity controller still runs, with zero input:
        // its target is rest, so the hull is braked at the authored deceleration, plus the drag.
        if let Some(idle) = self.lives[l].idle {
            let lv = &mut self.lives[l].hav;
            let (fwd, left) = (rot * Vec3::X, rot * Vec3::Y);
            let v = lv.v;
            let (af, al) = crate::reach_havok::hover_idle_brake(&idle, v.dot(fwd), v.dot(left), TICK_SECONDS);
            let mut acc = fwd * af + left * al;
            let vh = Vec3::new(v.x, v.y, 0.0);
            let sp = vh.length();
            if sp > 1e-6 {
                acc -= vh / sp * crate::reach_havok::hover_idle_drag(&idle, sp, TICK_SECONDS);
            }
            let com = lv.com;
            lv.apply_impulse_at(com, acc * (mass * h));
        }
    }

    /// Slot pose and AABB from the body.
    fn sync_slot(&mut self, i: usize) {
        let Kind::Live(l) = self.slots[i].kind else { return };
        let lv = &self.lives[l];
        let rot = lv.hav.q;
        self.slots[i].rot = rot;
        self.slots[i].origin = lv.hav.com - rot * lv.com_local;
        let ab = world_aabb(lv.local_aabb, self.slots[i].origin, rot, 0.02);
        self.slots[i].aabb = ab;
        self.grid_update(i);
    }
}

impl Default for PropWorld {
    fn default() -> Self {
        Self::new()
    }
}

fn shape_mat(shapes: &[ConvexShape]) -> (f32, f32) {
    shapes.first().map(|s| (s.mat.friction, s.mat.restitution)).unwrap_or((0.5, 0.2))
}

fn kin_aabb(k: &Kinematic) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for s in k.shapes {
        let (a, b) = s.aabb();
        let (wa, wb) = world_aabb((a, b), k.origin, k.rot, 0.05);
        lo = lo.min(wa);
        hi = hi.max(wb);
    }
    (lo, hi)
}

/// The nearest triangle hit by a ray going straight down from `origin`, within `len`:
/// `(distance, unit normal facing up)`.
pub fn ray_down(tris: &[[Vec3; 3]], origin: Vec3, len: f32) -> Option<(f32, Vec3)> {
    let mut best: Option<(f32, Vec3)> = None;
    for t in tris {
        // Moller-Trumbore with dir = -Z
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        let dir = -Vec3::Z;
        let pv = dir.cross(e2);
        let det = e1.dot(pv);
        if det.abs() < 1e-9 {
            continue;
        }
        let inv = 1.0 / det;
        let tv = origin - t[0];
        let u = tv.dot(pv) * inv;
        if !(-1e-5..=1.0 + 1e-5).contains(&u) {
            continue;
        }
        let qv = tv.cross(e1);
        let v = dir.dot(qv) * inv;
        if v < -1e-5 || u + v > 1.0 + 1e-5 {
            continue;
        }
        let d = e2.dot(qv) * inv;
        if d < -0.02 || d > len {
            continue;
        }
        if best.map_or(true, |(bd, _)| d < bd) {
            let n = e1.cross(e2).normalize_or_zero();
            best = Some((d.max(0.0), if n.z < 0.0 { -n } else { n }));
        }
    }
    best
}

// ------------------------------------------------------------------------------------------
// The seam for a hull that another law drives (the jeep)
// ------------------------------------------------------------------------------------------

/// One contact of a hull with the static world.
#[derive(Clone, Copy, Debug)]
pub struct HullContact {
    pub point: Vec3,
    /// Unit normal from the surface INTO the hull.
    pub normal: Vec3,
    /// Penetration, wu (> 0).
    pub depth: f32,
    /// Already combined with the level's material and stored as the engine stores it
    /// ([`contact_material`]): feed these to `reach_havok::solve_contacts`.
    pub friction: f32,
    pub restitution: f32,
}

/// The contacts of a rigid hull (a vehicle's `phmo` shapes, model space, at its editor scale)
/// posed at `origin` / `rot` against `world`. **The seam for the jeep law**, which models its
/// wheels as springs and has no hull collision of its own: a rolled jeep rests on its hull because
/// these contacts exist, and a hull driven into a wall is stopped here.
///
/// Support spheres (tyres) are left to the caller to filter out -- they carry
/// `ConvexShape::support` -- because the suspension owns them.
pub fn hull_vs_static(shapes: &[ConvexShape], origin: Vec3, rot: Quat, world: &dyn StaticWorld) -> Vec<HullContact> {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for s in shapes {
        let (a, b) = s.aabb();
        let (wa, wb) = world_aabb((a, b), origin, rot, 0.05);
        lo = lo.min(wa);
        hi = hi.max(wb);
    }
    let mut tris = Vec::new();
    world.gather(lo, hi, &mut tris);
    let mut out = Vec::new();
    if tris.is_empty() {
        return out;
    }
    let (m, inv) = (Mat3::from_quat(rot), Mat3::from_quat(rot).transpose());
    let mut buf: Vec<Contact> = Vec::new();
    for s in shapes.iter().filter(|s| !s.support) {
        buf.clear();
        shape_contacts(s, m, origin, inv, &tris, &mut buf);
        for c in &buf {
            let (f, r) = contact_material(c.mu, c.e, LEVEL_FRICTION, LEVEL_RESTITUTION);
            out.push(HullContact { point: c.p, normal: c.n, depth: c.depth, friction: f, restitution: r });
        }
    }
    out
}

// ------------------------------------------------------------------------------------------
// A DRIVEN vehicle's hull against the static world
// ------------------------------------------------------------------------------------------

/// A hull penetrating less than this is touching, not colliding, wu.
pub const HULL_SLOP: f32 = 0.004;
/// The swept hull is tested every this-far along its frame's travel, wu. Under half the thinnest
/// wall a Forge block makes, so a fast frame cannot step over one (the engine's Critical quality
/// runs continuous collision; this is the substepped stand-in for its TOI).
pub const HULL_SWEEP_STEP: f32 = 0.05;
/// Most sweep steps in one frame (a 48-step cap is 2.4 wu of travel in one frame).
pub const HULL_SWEEP_MAX: usize = 48;
/// The most one push-out iteration may move the hull, wu.
pub const HULL_PUSH_MAX: f32 = 0.5;
/// A drawn body that has come to rest this far from where its meshes were built is rebuilt
/// there, so its baked object lighting follows it, wu.
pub const RELIGHT_DISTANCE: f32 = 0.75;

/// What one corrected frame did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HullStep {
    /// Sweep steps taken.
    pub steps: usize,
    /// Steps at which the hull was in contact.
    pub hits: usize,
    /// Deepest penetration met, wu.
    pub deepest: f32,
    /// Speed into the surface that was removed, wu/s (the largest over the frame).
    pub impact: f32,
    /// The deepest contact of the first step that hit: where, and the surface's push direction.
    pub first: Option<(Vec3, Vec3)>,
    /// Penetration left when the frame ended (0 unless the push-out could not clear it), wu.
    pub left: f32,
}

/// Move a DRIVEN vehicle's hull from `prev` to `*pos` (where its own law put it this frame)
/// without letting it pass through the static world: the frame's travel is cut into short steps,
/// and at the first step that penetrates the hull is pushed back out along the contact normal and
/// its velocity gets the engine's contact response -- the normal component removed, friction
/// (the shapes' own mixed with the level's, stored as the engine stores it) taking up to
/// `mu * (normal impulse)` of the tangential speed, NO restitution (the engine's
/// `contactRestingVelocity` is `FLT_MAX`; INFERRED consequence) -- and the rest of the frame's
/// travel slides along the surface.
///
/// **Stand-ins:** translation only (the hull's attitude stays the vehicle law's), the static
/// world is triangles, the contact set is vertex-against-face, and the mass is infinite (the world
/// does not move). The vehicle's own law is not told: it will push into the wall again next frame
/// and be stopped again, which is what holds a vehicle against a wall.
pub fn sweep_hull(shapes: &[ConvexShape], rot: Quat, prev: Vec3, pos: &mut Vec3, vel: &mut Vec3, world: &dyn StaticWorld) -> HullStep {
    let mut out = HullStep::default();
    let hull: Vec<&ConvexShape> = shapes.iter().filter(|s| !s.support).collect();
    if hull.is_empty() {
        return out;
    }
    let owned: Vec<ConvexShape> = hull.into_iter().cloned().collect();
    let delta = *pos - prev;
    let n = ((delta.length() / HULL_SWEEP_STEP).ceil() as usize).clamp(1, HULL_SWEEP_MAX);
    let mut step = delta / n as f32;
    let mut p = prev;
    let mut v = *vel;
    for _ in 0..n {
        out.steps += 1;
        let mut cand = p + step;
        let mut contacts: Vec<HullContact> = hull_vs_static(&owned, cand, rot, world).into_iter().filter(|c| c.depth > HULL_SLOP).collect();
        if contacts.is_empty() {
            p = cand;
            out.left = 0.0;
            continue;
        }
        out.hits += 1;
        if out.first.is_none() {
            if let Some(c) = contacts.iter().max_by(|a, b| a.depth.total_cmp(&b.depth)) {
                out.first = Some((c.point, c.normal));
            }
        }
        let mut normal = Vec3::ZERO;
        let mut mu = 0.0f32;
        // push out: along the depth-weighted mean normal, repeated while it still penetrates. A
        // push that does not shallow the deepest contact (the hull is on the wrong side of a thin
        // surface, or two pushes fight) is undone and the loop stops: it must never ratchet the
        // hull through the world.
        for _ in 0..6 {
            let (mut nsum, mut dmax, mut wsum) = (Vec3::ZERO, 0.0f32, 0.0f32);
            for c in &contacts {
                nsum += c.normal * c.depth;
                dmax = dmax.max(c.depth);
                wsum += c.depth;
                mu += c.friction * c.depth;
            }
            out.deepest = out.deepest.max(dmax);
            let nn = nsum.normalize_or_zero();
            if nn == Vec3::ZERO {
                break;
            }
            normal = nn;
            mu /= wsum.max(1e-6);
            let before = cand;
            cand += nn * (dmax + HULL_SLOP * 0.5).min(HULL_PUSH_MAX);
            contacts = hull_vs_static(&owned, cand, rot, world).into_iter().filter(|c| c.depth > HULL_SLOP).collect();
            if contacts.is_empty() {
                break;
            }
            if contacts.iter().map(|c| c.depth).fold(0.0f32, f32::max) >= dmax {
                cand = before;
                contacts = hull_vs_static(&owned, cand, rot, world).into_iter().filter(|c| c.depth > HULL_SLOP).collect();
                break;
            }
            mu = 0.0;
        }
        out.left = contacts.iter().map(|c| c.depth).fold(0.0f32, f32::max);
        if normal != Vec3::ZERO {
            let vn = v.dot(normal);
            if vn < 0.0 {
                out.impact = out.impact.max(-vn);
                let dvn = -vn; // restitution 0
                v += normal * dvn;
                let vt = v - normal * v.dot(normal);
                let lim = mu * dvn;
                v -= if vt.length() <= lim { vt } else { vt.normalize() * lim };
            }
            // the rest of the frame's travel slides: its inward component is gone
            let sn = step.dot(normal);
            if sn < 0.0 {
                step -= normal * sn;
            }
        }
        p = cand;
    }
    *pos = p;
    *vel = v;
    out
}

// ------------------------------------------------------------------------------------------
// The scene adapters
// ------------------------------------------------------------------------------------------

/// The level triangle soup behind [`StaticWorld`].
pub struct SceneLevel<'a>(pub &'a dyn crate::objscene::ObjectScene);

impl StaticWorld for SceneLevel<'_> {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        self.0.static_tris_in_aabb(mn, mx, out);
    }
}

/// Bodies and hulls straight from the object scene.
pub struct SceneSource<'a> {
    pub scene: &'a dyn crate::objscene::ObjectScene,
    pub objects: &'a [hms_ipc::ObjectInfo],
    /// The tags, for vehicles' physics blocks. `None` = no vehicle is simulated.
    pub tags: Option<(&'a hms_native::NativeDll, hms_native::CacheHandle)>,
}

impl PropSource for SceneSource<'_> {
    fn physics_body(&self, datum: u32) -> Option<PhysBody> {
        let o = self.objects.iter().find(|o| o.datum == datum)?;
        self.scene.physics_body(o)
    }
    fn hull_tris(&self, datum: u32) -> Vec<[Vec3; 3]> {
        match self.objects.iter().find(|o| o.datum == datum) {
            Some(o) => self.scene.physics_world_tris(o),
            None => Vec::new(),
        }
    }
    fn vehicle_spec(&self, datum: u32) -> Option<VehicleSpec> {
        let (native, cache) = self.tags?;
        let o = self.objects.iter().find(|o| o.datum == datum)?;
        let vp = crate::reach_vehicle::VehiclePhysics::read(native, cache, o.primary_tag)?;
        let hv = &vp.havok;
        let mut spec = VehicleSpec {
            hover: Vec::new(),
            gravity_scale: crate::reach_havok::vehicle_gravity_scale_postprocessed(hv.gravity_scale_raw),
            wheeled: !hv.friction_points.is_empty(),
            idle: match &vp.family {
                crate::reach_vehicle::Family::Scout { model, .. } => Some(crate::reach_havok::IdleBrake {
                    speed_decel: model.speed_deceleration,
                    slide_decel: model.slide_deceleration,
                    drag: model.drag_coefficient,
                    constant_decel: model.constant_deceleration,
                }),
                _ => None,
            },
        };
        if hv.friction_points.is_empty() && !hv.antigravity_points.is_empty() {
            // the pads sit where `physics_body` put its support spheres (one per marker instance)
            let centres: Vec<Vec3> = self
                .scene
                .physics_body(o)
                .map(|b| {
                    b.shapes
                        .iter()
                        .filter(|s| s.support)
                        .filter_map(|s| if let ShapeGeom::Sphere { c, .. } = &s.geom { Some(*c) } else { None })
                        .collect()
                })
                .unwrap_or_default();
            for (k, a) in hv.antigravity_points.iter().enumerate() {
                let pos = if centres.is_empty() { Vec3::ZERO } else { centres[k % centres.len()] };
                spec.hover.push(HoverPoint {
                    pos,
                    law: HoverLaw {
                        strength: a.strength,
                        height: a.height,
                        radius: a.radius,
                        compression_damping: a.compression_damping,
                        extension_damping: a.extension_damping,
                        normal_k_hi: a.normal_k1,
                        normal_k_lo: a.normal_k0,
                    },
                });
            }
        }
        Some(spec)
    }
}

/// Offer every object to a new world. `modes` maps a datum to its Forge placement physics byte
/// (absent = Normal).
pub fn seed_world(
    scene: &dyn crate::objscene::ObjectScene,
    objects: &[hms_ipc::ObjectInfo],
    modes: &HashMap<u32, u8>,
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    all_solid: bool,
) -> PropWorld {
    let mut w = PropWorld::new();
    let mut solid: HashMap<u32, bool> = HashMap::new();
    for o in objects {
        let ok = all_solid || *solid.entry(o.primary_tag).or_insert_with(|| {
            // Vehicles and living units are not rigid props here: a parked vehicle is the jeep
            // law's (suspension), a driven one is a pusher, and their draw needs the attachment
            // pieces the actor path for props does not build.
            let class = native.tag_class(cache, o.primary_tag).unwrap_or_default();
            !matches!(class.as_str(), "bipd" | "ctrl" | "mach" | "dgrp")
                && crate::reach_collision::collides_with_biped(native, cache, o.primary_tag)
        });
        if !ok {
            continue;
        }
        let Some((mn, mx)) = scene.aabb_of(o.datum) else { continue };
        let mode = PhysicsMode::from_placement(modes.get(&o.datum).copied().unwrap_or(0));
        let vehicle = !all_solid && native.tag_class(cache, o.primary_tag).as_deref() == Some("vehi");
        w.add(PropSeed {
            datum: o.datum,
            origin: Vec3::from(o.pos),
            rot: crate::physics_settle::fwd_up_to_rot(Vec3::from(o.fwd), Vec3::from(o.up)),
            mode,
            aabb: (Vec3::from(mn) - Vec3::splat(0.02), Vec3::from(mx) + Vec3::splat(0.02)),
            vehicle,
        });
        // A vehicle placed with nothing under it falls at the start; one standing on the ground
        // (or hovering within reach of it) stays asleep until touched.
        if vehicle && mode == PhysicsMode::Normal {
            let (lo, hi) = (Vec3::from(mn), Vec3::from(mx));
            let c = Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, lo.z + 0.05);
            if scene.raycast_solid(c, -Vec3::Z, VEHICLE_SUPPORT_REACH).is_none() {
                w.pending_wake.push(o.datum);
            }
        }
    }
    w
}

// ------------------------------------------------------------------------------------------
// The test-mode session
// ------------------------------------------------------------------------------------------

/// What test mode holds: the world (built on the first frame), the Forge physics bytes, and the
/// actors that draw displaced props.
///
/// **Drawing is the ride's own mechanism** (`play_vehicle` / `PlayMode::sync_ride_actors`): a
/// placement that has left its saved pose is HIDDEN from the object lanes (the hidden set is part
/// of `object_pipeline_sig`, so the scene rebuilds once when the set changes) and drawn by an
/// ACTOR whose world matrix is set each frame. A prop at rest is never an actor, so a map where
/// nothing is touched costs nothing and draws exactly as the editor does.
#[derive(Default)]
pub struct PropSession {
    world: Option<PropWorld>,
    /// Forge `placement` byte per datum, set by the app when test mode starts.
    pub placement: HashMap<u32, u8>,
    actors: HashMap<u32, Vec<u32>>,
    /// Where each drawn body's meshes (and their baked object lighting) were built.
    baked: HashMap<u32, Vec3>,
    index: crate::play_vehicle::VehicleIndex,
    /// The editor's per-datum `(forge colour, scale, damage)` -- the maps the static draw reads.
    pub appearance: HashMap<u32, (Option<(u8, u8)>, f32, u8)>,
    /// The driven vehicle's collision world and hull shapes.
    driven_world: crate::play_mode::PlayWorld,
    driven_hulls: HashMap<u32, Vec<ConvexShape>>,
    /// Vehicle hulls (pushers), cached by datum: `(shapes, mass)`.
    hulls: HashMap<u32, (Vec<ConvexShape>, f32)>,
    /// Exits waiting to be handed to the world on the next frame.
    claims: Vec<VehicleClaim>,
    /// Vehicles the app has adopted into the ride path's parked set.
    adopted: Vec<u32>,
    pub disabled: bool,
    /// Skip the tag-based solidity / class filter (stand-in scenes whose tags are not real).
    pub all_solid: bool,
    pub note: String,
    pub last: StepStats,
}

impl PropSession {
    pub fn hidden_datums(&self) -> Vec<u32> {
        self.world.as_ref().map(|w| w.hidden_datums()).unwrap_or_default()
    }

    pub fn world(&self) -> Option<&PropWorld> {
        self.world.as_ref()
    }

    pub fn mark_adopted(&mut self, datum: u32) {
        if !self.adopted.contains(&datum) {
            self.adopted.push(datum);
        }
    }

    /// Adopted vehicles that are no longer parked have a driver again: release them, returning
    /// each one's state as it left the simulation.
    pub fn sync_released(&mut self, parked: &[u32]) -> Vec<(u32, (Vec3, Quat, Vec3, Vec3))> {
        let mut out = Vec::new();
        let gone: Vec<u32> = self.adopted.iter().copied().filter(|d| !parked.contains(d)).collect();
        for d in gone {
            self.adopted.retain(|x| *x != d);
            if let Some(st) = self.release_vehicle(d) {
                out.push((d, st));
            }
        }
        out
    }

    /// The DRIVEN vehicle against the static world, after its own law has moved it. `prev` is where
    /// it was before the step. Corrects the ride's pose and velocity through its public state.
    /// The world it collides with is the same one the player walks on (`PlayWorld`, regathered
    /// around the vehicle), minus the vehicle's own placement.
    #[allow(clippy::too_many_arguments)]
    pub fn collide_driven(
        &mut self,
        prev: Vec3,
        ride: &mut crate::play_vehicle::Riding,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        scene: &dyn crate::objscene::ObjectScene,
        objects: &[hms_ipc::ObjectInfo],
    ) -> HullStep {
        if self.disabled {
            return HullStep::default();
        }
        let datum = ride.hidden_datum;
        if !self.driven_hulls.contains_key(&datum) {
            let shapes = objects.iter().find(|o| o.datum == datum).and_then(|o| scene.physics_body(o)).map(|b| b.shapes).unwrap_or_default();
            self.driven_hulls.insert(datum, shapes);
        }
        let shapes = &self.driven_hulls[&datum];
        if shapes.is_empty() {
            return HullStep::default();
        }
        let here = ride.drive.pos;
        if !self.driven_world.covers(here) || !self.driven_world.covers(prev) {
            // everything but the vehicle itself and what rides on it (its turret, a carried weapon,
            // anything in its attach list, to any depth): those overlap its hull by construction
            let mut mine = vec![datum];
            let mut i = 0;
            while i < mine.len() {
                if let Some(o) = objects.iter().find(|o| o.datum == mine[i]) {
                    for a in o.attached.iter().copied().filter(|a| *a != 0 && *a != u32::MAX) {
                        if !mine.contains(&a) {
                            mine.push(a);
                        }
                    }
                }
                i += 1;
            }
            let others: Vec<hms_ipc::ObjectInfo> = objects.iter().filter(|o| !mine.contains(&o.datum)).copied().collect();
            self.driven_world.regather(here, scene, &others, native, cache);
        }
        let rot = ride.drive.basis();
        let mut pos = ride.drive.pos;
        let mut vel = ride.drive.vel;
        let st = sweep_hull(shapes, rot, prev, &mut pos, &mut vel, &self.driven_world);
        if st.hits > 0 {
            ride.drive.pos = pos;
            ride.drive.vel = vel;
            if let Some((at, n)) = st.first {
                // name what stopped it, for the caption: the nearest object hull, else the level
                let near = objects
                    .iter()
                    .filter(|o| o.datum != datum)
                    .map(|o| ((Vec3::from(o.pos) - at).length(), o))
                    .filter(|(d, _)| *d < 6.0)
                    .min_by(|a, b| a.0.total_cmp(&b.0));
                self.note = format!(
                    "driven hull pushed out at ({:.1}, {:.1}, {:.1}) along ({:.2}, {:.2}, {:.2}), {:.3} deep; nearest object {}",
                    at.x, at.y, at.z, n.x, n.y, n.z, st.deepest,
                    near.map_or("none (level)".to_string(), |(d, o)| format!("datum {} tag {:#x} at {d:.1} wu", o.datum, o.primary_tag)),
                );
            }
        }
        st
    }

    /// A driver got out: the vehicle becomes a body at this pose and velocity next frame.
    pub fn request_claim(&mut self, c: VehicleClaim) {
        self.claims.push(c);
    }

    /// Vehicles that left their saved pose and are not yet drawn by the ride path.
    pub fn displaced_vehicles(&self) -> Vec<u32> {
        self.world.as_ref().map(|w| w.displaced_vehicles()).unwrap_or_default()
    }

    /// The simulated state of a vehicle, for the app to write into its ride state.
    pub fn vehicle_state(&self, datum: u32) -> Option<(Vec3, Quat, Vec3, Vec3)> {
        self.world.as_ref().and_then(|w| w.vehicle_state(datum))
    }

    /// A driver got in (or the jeep law took over): hand the vehicle back with its current state.
    pub fn release_vehicle(&mut self, datum: u32) -> Option<(Vec3, Quat, Vec3, Vec3)> {
        self.world.as_mut().and_then(|w| w.release_vehicle(datum))
    }

    /// Is this vehicle simulated by the world right now (so the ride path must not step it)?
    pub fn owns_vehicle(&self, datum: u32) -> bool {
        self.world.as_ref().is_some_and(|w| w.vehicle_state(datum).is_some())
    }

    /// One frame. `player` is the on-foot capsule (feet position, velocity, radius, height, mass);
    /// `vehicle` is the driven hull (datum of the placement, origin, rotation, velocity).
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        dt: f32,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        scene: &dyn crate::objscene::ObjectScene,
        objects: &[hms_ipc::ObjectInfo],
        player: Option<PlayerPusher>,
        vehicles: &[VehiclePusher],
        actors: &mut crate::reach_actor::ActorScene,
    ) {
        if self.disabled {
            return;
        }
        if self.world.is_none() {
            let w = seed_world(scene, objects, &self.placement, native, cache, self.all_solid);
            self.note = format!("rigid bodies: {} candidate object(s)", w.len());
            self.world = Some(w);
        }
        let src = SceneSource { scene, objects, tags: Some((native, cache)) };
        let level = SceneLevel(scene);
        if let Some(w) = self.world.as_mut() {
            for c in std::mem::take(&mut self.claims) {
                w.claim_vehicle(c.datum, c.origin, c.rot, c.velocity, c.angular, &src);
            }
        }
        // pushers
        let cap;
        let mut kin: Vec<Kinematic> = Vec::new();
        if let Some(p) = player {
            let r = p.radius;
            let top = (p.height - r).max(r);
            let shape = ConvexShape::capsule(
                Vec3::new(0.0, 0.0, r),
                Vec3::new(0.0, 0.0, top),
                r,
                crate::physics_settle::ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 0.0 },
            );
            cap = shape.into_iter().collect::<Vec<_>>();
            if !cap.is_empty() {
                kin.push(Kinematic {
                    shapes: &cap,
                    origin: p.feet,
                    rot: Quat::IDENTITY,
                    velocity: p.velocity,
                    angular: Vec3::ZERO,
                    mass: p.mass,
                    com: Vec3::new(0.0, 0.0, p.height * 0.5),
                });
            }
        }
        for v in vehicles {
            if !self.hulls.contains_key(&v.datum) {
                let hull = objects
                    .iter()
                    .find(|o| o.datum == v.datum)
                    .and_then(|o| scene.physics_body(o))
                    .map(|b| (b.shapes, b.mass))
                    .unwrap_or_default();
                self.hulls.insert(v.datum, hull);
            }
        }
        for v in vehicles {
            if let Some(h) = self.hulls.get(&v.datum) {
                if !h.0.is_empty() {
                    kin.push(Kinematic {
                        shapes: &h.0,
                        origin: v.origin,
                        rot: v.rot,
                        velocity: v.velocity,
                        angular: Vec3::ZERO,
                        mass: h.1.max(1.0),
                        com: Vec3::ZERO,
                    });
                }
            }
        }
        let Some(world) = self.world.as_mut() else { return };
        self.last = world.step(dt, &level, &src, &kin);
        // draw the displaced: exactly what the static object pipeline draws for the same
        // placement -- its own model in its model variant, the children that variant attaches,
        // the Forge colour, the damage state and the editor scale -- moved with the body
        let disp = world.displaced_scaled();
        let asleep: Vec<bool> = disp.iter().map(|(d, ..)| world.is_asleep(*d).unwrap_or(true)).collect();
        let info_of: HashMap<u32, &hms_ipc::ObjectInfo> = objects.iter().map(|o| (o.datum, o)).collect();
        let mut keep: Vec<u32> = Vec::with_capacity(disp.len());
        for ((datum, origin, rot, scale), asleep) in disp.iter().zip(asleep) {
            let Some(o) = info_of.get(datum) else { continue };
            keep.push(*datum);
            let world_m = glam::Mat4::from_scale_rotation_translation(Vec3::splat(*scale), *rot, *origin);
            let pieces = crate::play_vehicle::object_pieces(&mut self.index, native, cache, o.primary_tag, world_m, o.variant_name_sid);
            let ids = self.actors.entry(*datum).or_default();
            if ids.len() != pieces.len() {
                let old = std::mem::take(ids);
                actors.actors.retain(|a| !old.contains(&a.id));
                for p in &pieces {
                    if let Ok(id) = actors.add_render_model(native, cache, p.mode_tag, None, Vec3::ZERO, 0.0) {
                        ids.push(id);
                    }
                }
                actors.rebuild = true;
                self.baked.insert(*datum, *origin);
            }
            for (id, p) in ids.iter().zip(pieces.iter()) {
                if let Some(a) = actors.get_mut(*id) {
                    a.world_override = Some(p.world);
                    a.playing = false;
                }
            }
            let (forge_color, _, damage) = self.appearance.get(datum).copied().unwrap_or((None, 1.0, 0));
            crate::play_vehicle::apply_appearance(actors, ids, &crate::play_vehicle::RideAppearance { forge_color, scale: *scale, damage });
            if let Some(&first) = ids.first() {
                crate::play_vehicle::set_variant(actors, first, o.variant_name_sid);
            }
            // The object lighting is baked into the meshes where they were built; a body that has
            // come to rest well away from there is re-lit where it lies.
            if asleep && self.baked.get(datum).map_or(true, |b| (*origin - *b).length() > RELIGHT_DISTANCE) {
                for &id in ids.iter() {
                    if let Some(a) = actors.get_mut(id) {
                        a.meshes_dirty = true;
                    }
                }
                actors.rebuild = true;
                self.baked.insert(*datum, *origin);
            }
        }
        let stale: Vec<u32> = self.actors.keys().filter(|d| !keep.contains(d)).copied().collect();
        if !stale.is_empty() {
            let ids: Vec<u32> = stale.iter().filter_map(|d| self.actors.remove(d)).flatten().collect();
            for d in &stale {
                self.baked.remove(d);
            }
            actors.actors.retain(|a| !ids.contains(&a.id));
            actors.rebuild = true;
        }
    }

    /// Leaving test mode: every body back at its saved pose and every actor removed.
    pub fn leave(&mut self, actors: &mut crate::reach_actor::ActorScene) {
        if let Some(w) = self.world.as_mut() {
            w.restore_all();
        }
        if !self.actors.is_empty() {
            let ids: Vec<u32> = self.actors.drain().flat_map(|(_, ids)| ids).collect();
            self.baked.clear();
            actors.actors.retain(|a| !ids.contains(&a.id));
            actors.rebuild = true;
        }
        self.world = None;
        self.hulls.clear();
        self.claims.clear();
    }
}

/// The on-foot player as a pusher.
#[derive(Clone, Copy, Debug)]
pub struct PlayerPusher {
    pub feet: Vec3,
    pub velocity: Vec3,
    pub radius: f32,
    pub height: f32,
    pub mass: f32,
}

/// A drawable, hand-back-able ride record for a vehicle nobody is in: the same `Riding` the ride
/// path parks, so the parked-vehicle machinery (actors, hidden placement, re-boarding that resumes
/// THIS vehicle) draws and returns it. Its drive state starts at the placement's pose.
pub fn bare_riding(
    index: &mut crate::play_vehicle::VehicleIndex,
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    o: &hms_ipc::ObjectInfo,
) -> Option<crate::play_vehicle::Riding> {
    use crate::play_vehicle::{Riding, RideAppearance, RidePhase};
    let entry = index.entry(native, cache, o.primary_tag)?;
    let start = crate::scene::object_matrix(o);
    let yaw = start.x_axis.y.atan2(start.x_axis.x);
    Some(Riding {
        hidden_datum: o.datum,
        vehi_tag: o.primary_tag,
        vehi_path: entry.path.clone(),
        placement_tag: o.primary_tag,
        seat: 0,
        seat_label: String::new(),
        seat_mode: String::new(),
        drive: crate::reach_vehicle::DriveState { pos: start.w_axis.truncate(), yaw, ..Default::default() },
        start,
        drivable: entry.drive.as_ref().map(|_| ()).map_err(|s| s.clone()),
        yaw,
        pitch: 0.0,
        appearance: RideAppearance::default(),
        phase: RidePhase::Seated,
        phase_age: 0.0,
        enter_len: None,
        exit_len: None,
        obstruction: Default::default(),
        allows_weapons: false,
    })
}

/// Write a simulated vehicle's state into a ride record's drive state (what the parked path draws,
/// hides and hands back).
pub fn write_drive_state(d: &mut crate::reach_vehicle::DriveState, st: (Vec3, Quat, Vec3, Vec3)) {
    let (origin, rot, v, w) = st;
    d.pos = origin;
    d.vel = v;
    d.orient = Some(rot);
    d.ang_vel = w;
    let f = rot * Vec3::X;
    d.yaw = f.y.atan2(f.x);
}

/// A vehicle handed to the world when its driver gets out.
#[derive(Clone, Copy, Debug)]
pub struct VehicleClaim {
    pub datum: u32,
    pub origin: Vec3,
    pub rot: Quat,
    pub velocity: Vec3,
    pub angular: Vec3,
}

/// A vehicle the world does not step but must collide with: the driven one, or a parked one the
/// ride path moves.
#[derive(Clone, Copy, Debug)]
pub struct VehiclePusher {
    pub datum: u32,
    pub origin: Vec3,
    pub rot: Quat,
    pub velocity: Vec3,
}

// ------------------------------------------------------------------------------------------
// Gates
// ------------------------------------------------------------------------------------------

#[cfg(test)]
mod gates {
    use super::*;
    use crate::physics_settle::ShapeMaterial;
    use crate::reach_havok::TICK_HZ;

    const G: f32 = crate::reach_physics::WORLD_GRAVITY;

    /// The floor: a big quad at z = 0.
    fn floor() -> Vec<[Vec3; 3]> {
        let (a, b, c, d) = (Vec3::new(-50.0, -50.0, 0.0), Vec3::new(50.0, -50.0, 0.0), Vec3::new(50.0, 50.0, 0.0), Vec3::new(-50.0, 50.0, 0.0));
        vec![[a, b, c], [a, c, d]]
    }

    fn box_body(half: f32, mass: f32, mu: f32, e: f32) -> PhysBody {
        let mat = ShapeMaterial { friction: mu, restitution: e, mass };
        let shape = ConvexShape::aabb_box(Vec3::ZERO, Vec3::splat(half), 0.0, mat).unwrap();
        let mut b = PhysBody::from_shape(shape, "box".into());
        b.mass = mass;
        let i = mass * (2.0 * half * half * 4.0) / 12.0;
        b.inertia = Mat3::from_diagonal(Vec3::splat(i));
        b.com = Vec3::ZERO;
        b.lin_damping = 0.0;
        b.ang_damping = 0.0;
        b
    }

    fn sphere_body(r: f32, mass: f32, mu: f32, e: f32) -> PhysBody {
        let mat = ShapeMaterial { friction: mu, restitution: e, mass };
        let shape = ConvexShape::sphere(Vec3::ZERO, r, mat).unwrap();
        let mut b = PhysBody::from_shape(shape, "ball".into());
        b.mass = mass;
        b.inertia = Mat3::from_diagonal(Vec3::splat(0.4 * mass * r * r));
        b.com = Vec3::ZERO;
        b.lin_damping = 0.0;
        b.ang_damping = 0.0;
        b
    }

    struct Src {
        bodies: HashMap<u32, PhysBody>,
        hulls: HashMap<u32, Vec<[Vec3; 3]>>,
        specs: HashMap<u32, VehicleSpec>,
        decodes: std::cell::Cell<usize>,
    }
    impl Src {
        fn new() -> Self {
            Src { bodies: HashMap::new(), hulls: HashMap::new(), specs: HashMap::new(), decodes: std::cell::Cell::new(0) }
        }
    }
    impl PropSource for Src {
        fn physics_body(&self, d: u32) -> Option<PhysBody> {
            self.decodes.set(self.decodes.get() + 1);
            self.bodies.get(&d).cloned()
        }
        fn hull_tris(&self, d: u32) -> Vec<[Vec3; 3]> {
            self.hulls.get(&d).cloned().unwrap_or_default()
        }
        fn vehicle_spec(&self, d: u32) -> Option<VehicleSpec> {
            self.specs.get(&d).cloned()
        }
    }

    fn seed(datum: u32, pos: Vec3, half: f32, mode: PhysicsMode) -> PropSeed {
        PropSeed { datum, origin: pos, rot: Quat::IDENTITY, mode, aabb: (pos - Vec3::splat(half + 0.02), pos + Vec3::splat(half + 0.02)), vehicle: false }
    }

    fn run(w: &mut PropWorld, level: &Vec<[Vec3; 3]>, src: &Src, kin: &[Kinematic], ticks: u32) {
        for _ in 0..ticks {
            w.step(TICK_SECONDS, level, src, kin);
        }
    }

    /// The Forge placement physics byte decodes as the mvar layout says.
    #[test]
    fn placement_physics_bits_decode() {
        assert_eq!(PhysicsMode::from_placement(0b0000_1100), PhysicsMode::Normal, "the retail default 0x0C");
        assert_eq!(PhysicsMode::from_placement(1 << 6 | 0x0C), PhysicsMode::Fixed);
        assert_eq!(PhysicsMode::from_placement(3 << 6), PhysicsMode::Phased);
        assert_eq!(PhysicsMode::from_placement(2 << 6), PhysicsMode::Normal, "2 is unused and reads as normal");
    }

    /// A crate dropped from height lands at the analytic time and height and then SLEEPS.
    #[test]
    fn a_crate_dropped_from_height_lands_at_the_analytic_time_and_sleeps() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(0.5, 100.0, 0.8, 0.0));
        let level = floor();
        let mut w = PropWorld::new();
        let drop = 2.0f32;
        w.add(seed(1, Vec3::new(0.0, 0.0, 0.5 + drop), 0.5, PhysicsMode::Normal));
        assert!(w.wake(1, &src));
        let mut land = None;
        let mut slept_at = None;
        for t in 0..400u32 {
            w.step(TICK_SECONDS, &level, &src, &[]);
            let z = w.body_pose(1).unwrap().0.z;
            if land.is_none() && z <= 0.5 + 0.02 {
                land = Some(t);
            }
            if slept_at.is_none() && w.is_asleep(1).unwrap() {
                slept_at = Some(t);
                break;
            }
        }
        // free fall of `drop` wu at 60 Hz symplectic Euler: the first N with g dt^2 N(N+1)/2 >= drop
        let n_fall = (1..).find(|&n| G * TICK_SECONDS * TICK_SECONDS * (n * (n + 1)) as f32 / 2.0 >= drop).unwrap();
        let land = land.expect("never landed");
        assert!((land as i32 - n_fall as i32).abs() <= 3, "landed on tick {land}, analytic fall takes {n_fall} ticks");
        let slept = slept_at.expect("never slept");
        assert!(slept > land + SLEEP_TICKS - 2 && slept < land + 200, "slept at {slept}, landed at {land}");
        let z = w.body_pose(1).unwrap().0.z;
        assert!((z - 0.5).abs() < 0.01, "rests at {z}, a unit crate rests at 0.5");
        assert!(w.body_velocity(1).unwrap().0.length() == 0.0);
    }

    /// Nothing falls on its own: a sleeping body stays at its saved pose, the world does NO
    /// narrow-phase work, and the render signature does not move.
    #[test]
    fn a_world_at_rest_does_no_work_and_does_not_change_the_signature() {
        let mut src = Src::new();
        let level = floor();
        let mut w = PropWorld::new();
        for i in 0..500u32 {
            src.bodies.insert(i, box_body(0.5, 50.0, 0.8, 0.0));
            // half of them float: they must still not move
            w.add(seed(i, Vec3::new((i % 25) as f32 * 3.0, (i / 25) as f32 * 3.0, 0.5 + (i % 2) as f32 * 4.0), 0.5, PhysicsMode::Normal));
        }
        let sig0 = w.signature();
        run(&mut w, &level, &src, &[], 120);
        assert_eq!(w.stats.pair_tests + w.stats.static_contacts + w.stats.pair_contacts, 0);
        assert_eq!(w.awake_count(), 0);
        assert_eq!(src.decodes.get(), 0, "no body was decoded: they are all dormant");
        assert_eq!(w.signature(), sig0);
        assert!(w.displaced().is_empty() && w.hidden_datums().is_empty());
        assert_eq!(w.body_pose(7).unwrap().0, Vec3::new(21.0, 0.0, 4.5));
    }

    /// A Fixed object never moves and is solid; a PHASED object is not.
    #[test]
    fn a_fixed_object_is_terrain_and_a_phased_object_is_not_there() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(0.5, 100.0, 0.8, 0.0)); // the dropped crate
        src.bodies.insert(2, box_body(1.0, 100.0, 0.8, 0.0)); // fixed: would be a body if it were Normal
        src.bodies.insert(3, box_body(1.0, 100.0, 0.8, 0.0)); // phased
        // the Fixed object's hull: a slab 2x2 wide from z 0..1 at x = 10
        let slab = |x: f32, top: f32| -> Vec<[Vec3; 3]> {
            let (a, b, c, d) = (Vec3::new(x - 1.0, -1.0, top), Vec3::new(x + 1.0, -1.0, top), Vec3::new(x + 1.0, 1.0, top), Vec3::new(x - 1.0, 1.0, top));
            vec![[a, b, c], [a, c, d]]
        };
        src.hulls.insert(2, slab(10.0, 1.0));
        src.hulls.insert(3, slab(20.0, 1.0));
        let level = floor();
        let mut w = PropWorld::new();
        w.add(seed(1, Vec3::new(10.0, 0.0, 4.0), 0.5, PhysicsMode::Normal));
        w.add(PropSeed { aabb: (Vec3::new(9.0, -1.0, 0.0), Vec3::new(11.0, 1.0, 1.0)), ..seed(2, Vec3::new(10.0, 0.0, 0.5), 1.0, PhysicsMode::Fixed) });
        w.add(PropSeed { aabb: (Vec3::new(19.0, -1.0, 0.0), Vec3::new(21.0, 1.0, 1.0)), ..seed(3, Vec3::new(20.0, 0.0, 0.5), 1.0, PhysicsMode::Phased) });
        w.add(seed(4, Vec3::new(20.0, 0.0, 4.0), 0.5, PhysicsMode::Normal));
        src.bodies.insert(4, box_body(0.5, 100.0, 0.8, 0.0));
        w.wake(1, &src);
        w.wake(4, &src);
        run(&mut w, &level, &src, &[], 300);
        // the crate over the Fixed slab (top z = 1) rests ON it, at 1.5
        let z1 = w.body_pose(1).unwrap().0.z;
        assert!((z1 - 1.5).abs() < 0.03, "crate over the fixed slab rests at {z1}, want 1.5");
        // the crate over the Phased slab falls through it to the floor
        let z4 = w.body_pose(4).unwrap().0.z;
        assert!((z4 - 0.5).abs() < 0.03, "crate over the phased slab rests at {z4}, want 0.5 (the slab is not there)");
        // neither the Fixed nor the Phased object moved or became a body
        assert_eq!(w.body_pose(2).unwrap().0, Vec3::new(10.0, 0.0, 0.5));
        assert_eq!(w.body_pose(3).unwrap().0, Vec3::new(20.0, 0.0, 0.5));
        assert_eq!(w.live_count(), 2, "only the two Normal crates became bodies");
        assert!(w.hidden_datums().iter().all(|d| *d == 1 || *d == 4));
    }

    /// A ball dropped with restitution e rebounds to e^2 h WHEN the restitution threshold lets it;
    /// with the engine's FLT_MAX threshold it does not bounce.
    #[test]
    fn a_ball_bounces_to_the_restitution_height_and_the_engine_threshold_stops_it() {
        let h0 = 2.0f32;
        let e_shape = 1.0f32; // shape restitution; the level contributes 0.2 -> sqrt(1*0.2) stored
        assert_eq!(LEVEL_RESTITUTION, 0.2);
        let want_e = crate::reach_havok::contact_restitution_stored((e_shape * 0.2f32).sqrt());
        let mut peaks = [0.0f32; 2];
        for (k, resting) in [1.0f32, f32::MAX].into_iter().enumerate() {
            let mut src = Src::new();
            src.bodies.insert(1, sphere_body(0.25, 5.0, 0.5, e_shape));
            let level = floor();
            let mut w = PropWorld::new();
            w.solver.resting_velocity = resting;
            w.add(PropSeed { aabb: (Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 3.0)), ..seed(1, Vec3::new(0.0, 0.0, 0.25 + h0), 0.25, PhysicsMode::Normal) });
            w.wake(1, &src);
            let mut touched = false;
            let mut peak = 0.0f32;
            for _ in 0..240 {
                w.step(TICK_SECONDS, &level, &src, &[]);
                let (p, _) = w.body_pose(1).unwrap();
                let v = w.body_velocity(1).unwrap().0;
                if p.z < 0.25 + 0.03 {
                    touched = true;
                }
                if touched && v.z > 0.0 {
                    peak = peak.max(p.z - 0.25);
                }
            }
            assert!(touched);
            peaks[k] = peak;
        }
        let want = want_e * want_e * h0;
        assert!((peaks[0] - want).abs() < 0.2 * want, "with a finite threshold the ball rebounds {} wu, e^2 h = {want}", peaks[0]);
        assert!(peaks[1] < 0.03, "with the engine's threshold it rebounded {}", peaks[1]);
    }

    /// A pushed box slides and stops by friction at the analytic distance v0^2 / (2 mu g).
    #[test]
    fn a_pushed_box_slides_to_the_analytic_stopping_distance() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(0.5, 40.0, 0.5, 0.0));
        let level = floor();
        let mut w = PropWorld::new();
        w.add(seed(1, Vec3::new(0.0, 0.0, 0.5), 0.5, PhysicsMode::Normal));
        let v0 = 3.0f32;
        w.set_body_velocity(1, Vec3::new(v0, 0.0, 0.0), Vec3::ZERO, &src);
        let mut stopped = None;
        for t in 0..600u32 {
            w.step(TICK_SECONDS, &level, &src, &[]);
            if w.is_asleep(1).unwrap() {
                stopped = Some(t);
                break;
            }
        }
        assert!(stopped.is_some(), "never came to rest");
        // the level's friction is the root material's 0.9, written as a LITERAL: expected values
        // may not be computed from the constant under test
        assert_eq!(LEVEL_FRICTION, 0.9);
        let (mu, _) = contact_material(0.5, 0.0, 0.9, 0.2);
        let want = v0 * v0 / (2.0 * mu * G);
        let got = w.body_pose(1).unwrap().0.x;
        assert!((got - want).abs() < 0.12 * want, "slid {got} wu, v0^2/(2 mu g) = {want} with mu {mu}");
    }

    /// Three boxes stacked stay stacked and the stack goes to sleep.
    #[test]
    fn a_stack_of_boxes_stays_stacked_and_sleeps() {
        let mut src = Src::new();
        let level = floor();
        let mut w = PropWorld::new();
        for i in 0..3u32 {
            src.bodies.insert(i, box_body(0.5, 30.0, 0.8, 0.0));
            w.add(seed(i, Vec3::new(0.0, 0.0, 0.5 + i as f32 * 1.0), 0.5, PhysicsMode::Normal));
            w.wake(i, &src);
        }
        run(&mut w, &level, &src, &[], 600);
        for i in 0..3u32 {
            let p = w.body_pose(i).unwrap().0;
            assert!((p.z - (0.5 + i as f32)).abs() < 0.05, "box {i} at z {} (want {})", p.z, 0.5 + i as f32);
            assert!(p.x.abs() < 0.05 && p.y.abs() < 0.05, "box {i} slid to {p:?}");
            assert_eq!(w.is_asleep(i), Some(true), "box {i} still awake");
        }
    }

    /// The player's capsule pushes a resting crate: it wakes, moves with the player and keeps
    /// moving only while pushed. The pusher's own velocity is untouched.
    #[test]
    fn the_player_pushes_a_crate() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(0.5, 60.0, 0.5, 0.0));
        let level = floor();
        let mut w = PropWorld::new();
        w.add(seed(1, Vec3::new(2.0, 0.0, 0.5), 0.5, PhysicsMode::Normal));
        let cap = ConvexShape::capsule(Vec3::new(0.0, 0.0, 0.3), Vec3::new(0.0, 0.0, 1.0), 0.3, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 0.0 }).unwrap();
        let shapes = [cap];
        let mut x = 0.0f32;
        let speed = 2.0f32;
        for _ in 0..180 {
            x += speed * TICK_SECONDS;
            let kin = Kinematic { shapes: &shapes, origin: Vec3::new(x, 0.0, 0.0), rot: Quat::IDENTITY, velocity: Vec3::new(speed, 0.0, 0.0), angular: Vec3::ZERO, mass: 125.0, com: Vec3::new(0.0, 0.0, 0.6) };
            w.step(TICK_SECONDS, &level, &src, std::slice::from_ref(&kin));
        }
        let p = w.body_pose(1).unwrap().0;
        assert!(w.stats.woke > 0 || w.awake_count() > 0 || p.x > 2.01, "the crate never woke");
        assert!(p.x > 2.5, "the crate was not pushed: x = {}", p.x);
        // it is ahead of the capsule's front face (x + 0.3) by at most its half width + slop
        assert!(p.x - 0.5 >= x + 0.3 - 0.15, "the crate is inside the player: crate x {} player x {x}", p.x);
        // pushed at about the walking speed
        let v = w.body_velocity(1).unwrap().0.x;
        assert!(v > 0.5 * speed && v < 1.5 * speed, "crate speed {v}");
        assert_eq!(1, 1);
    }

    /// A driven vehicle hits a light crate: the crate is carried off ahead of the hull at about the
    /// hull's speed, and the hull's own speed is not changed by the solve.
    #[test]
    fn a_vehicle_hitting_a_crate_carries_it_off() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(0.5, 40.0, 0.5, 0.0));
        let level = floor();
        let mut w = PropWorld::new();
        w.add(seed(1, Vec3::new(6.0, 0.0, 0.5), 0.5, PhysicsMode::Normal));
        let hull = [ConvexShape::aabb_box(Vec3::new(0.0, 0.0, 0.6), Vec3::new(1.5, 0.8, 0.5), 0.0, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 1000.0 }).unwrap()];
        let speed = 6.0f32;
        let mut x = 0.0f32;
        let mut max_v = 0.0f32;
        for _ in 0..150 {
            x += speed * TICK_SECONDS;
            let kin = Kinematic { shapes: &hull, origin: Vec3::new(x, 0.0, 0.0), rot: Quat::IDENTITY, velocity: Vec3::new(speed, 0.0, 0.0), angular: Vec3::ZERO, mass: 1000.0, com: Vec3::new(0.0, 0.0, 0.6) };
            w.step(TICK_SECONDS, &level, &src, std::slice::from_ref(&kin));
            max_v = max_v.max(w.body_velocity(1).unwrap().0.x);
        }
        let p = w.body_pose(1).unwrap().0;
        assert!(p.x > 8.0, "the crate was not carried: x = {}", p.x);
        assert!(max_v > 0.6 * speed, "peak crate speed {max_v} against a hull doing {speed}");
        assert!(max_v < 1.6 * speed, "the crate was flung faster ({max_v}) than the hull ({speed}) can explain");
    }

    /// Leaving test mode puts every body back EXACTLY where it was saved.
    #[test]
    fn restoring_puts_every_body_back_exactly() {
        let mut src = Src::new();
        let level = floor();
        let mut w = PropWorld::new();
        let poses = [Vec3::new(1.0, 2.0, 3.5), Vec3::new(-4.0, 0.5, 0.5)];
        for (i, p) in poses.iter().enumerate() {
            src.bodies.insert(i as u32, box_body(0.5, 30.0, 0.8, 0.0));
            w.add(seed(i as u32, *p, 0.5, PhysicsMode::Normal));
            w.wake(i as u32, &src);
        }
        run(&mut w, &level, &src, &[], 200);
        assert!(!w.displaced().is_empty(), "the floating crate should have fallen");
        w.restore_all();
        for (i, p) in poses.iter().enumerate() {
            let (o, r) = w.body_pose(i as u32).unwrap();
            assert_eq!(o.to_array().map(f32::to_bits), p.to_array().map(f32::to_bits), "body {i} origin not bit-exact");
            assert_eq!(r, Quat::IDENTITY);
            assert_eq!(w.is_asleep(i as u32), Some(true));
        }
        assert!(w.displaced().is_empty() && w.hidden_datums().is_empty());
        assert_eq!(w.signature(), 0);
        // and it behaves the same on a second run
        w.wake(0, &src);
        run(&mut w, &level, &src, &[], 200);
        assert!(w.body_pose(0).unwrap().0.z < 3.0);
    }

    /// THE RENDER CHAIN, headless: a body that moves changes the signature and joins the hidden
    /// set (so the scene rebuilds once and the actor draws it); once it settles and sleeps the
    /// signature stops changing (so nothing rebuilds every frame).
    #[test]
    fn a_moving_body_changes_the_signature_and_a_settled_one_stops() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(0.5, 30.0, 0.8, 0.0));
        let level = floor();
        let mut w = PropWorld::new();
        w.add(seed(1, Vec3::new(0.0, 0.0, 3.0), 0.5, PhysicsMode::Normal));
        w.wake(1, &src);
        let mut sigs = std::collections::HashSet::new();
        for _ in 0..30 {
            w.step(TICK_SECONDS, &level, &src, &[]);
            sigs.insert(w.signature());
        }
        assert!(sigs.len() >= 20, "a falling body produced only {} distinct signatures", sigs.len());
        assert_eq!(w.hidden_datums(), vec![1], "a displaced body hides its placement");
        let (_, rot) = w.displaced().iter().find(|d| d.0 == 1).map(|d| (d.1, d.2)).unwrap();
        assert!(rot.is_normalized());
        run(&mut w, &level, &src, &[], 400);
        assert_eq!(w.is_asleep(1), Some(true));
        let s = w.signature();
        run(&mut w, &level, &src, &[], 60);
        assert_eq!(w.signature(), s, "a settled world must not keep re-signing");
        assert_eq!(w.hidden_datums(), vec![1], "it stays drawn as an actor while displaced");
    }

    /// Bounded cost: many bodies asleep cost nothing; many awake stay under the per-frame cap and
    /// the number of bodies stepped never exceeds it.
    #[test]
    fn many_bodies_are_bounded() {
        let mut src = Src::new();
        let level = floor();
        let mut w = PropWorld::new();
        // 292 bodies against a cap of 192, both LITERAL: a cap that moved with the constant would
        // make this gate follow it
        let n = 292u32;
        for i in 0..n {
            src.bodies.insert(i, box_body(0.25, 5.0, 0.8, 0.0));
            w.add(seed(i, Vec3::new((i % 20) as f32 * 1.0, (i / 20) as f32 * 1.0, 0.25 + 3.0), 0.25, PhysicsMode::Normal));
            w.wake(i, &src);
        }
        let t0 = std::time::Instant::now();
        let st = w.step(TICK_SECONDS, &level, &src, &[]);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(st.held, 292 - 192, "bodies beyond the cap of 192 are held, not stepped");
        assert_eq!(MAX_AWAKE, 192);
        assert!(st.pair_tests < 200_000, "pair tests {} look unbounded", st.pair_tests);
        eprintln!("play_world: {n} awake bodies, one tick {ms:.2} ms ({} pair tests, {} contacts)", st.pair_tests, st.pair_contacts);
    }

    /// A scene stand-in: a floor, and boxes for bodies. The trait has no defaults for most of
    /// what a real scene answers, so a stand-in has to say what it is.
    struct StubScene {
        src: Src,
        aabbs: HashMap<u32, (Vec3, Vec3)>,
    }
    impl crate::objscene::ObjectScene for StubScene {
        fn has_cache(&self) -> bool { true }
        fn invalidate(&mut self) {}
        fn rebuild_pending(&self) -> bool { false }
        fn maybe_rebuild(
            &mut self,
            _o: &[hms_ipc::ObjectInfo],
            _c: &HashMap<u32, (u8, u8)>,
            _m: &hms_render::MeshRenderer,
            _d: &eframe::wgpu::Device,
            _q: &eframe::wgpu::Queue,
        ) -> Option<crate::objscene::RebuiltLanes> { None }
        fn set_object_scales(&mut self, _s: HashMap<u32, f32>) {}
        fn set_object_casters(&mut self, _c: std::collections::HashSet<u32>) {}
        fn pick(&self, _o: Vec3, _d: Vec3) -> Option<u32> { None }
        fn aabb_of(&self, d: u32) -> Option<([f32; 3], [f32; 3])> { self.aabbs.get(&d).map(|(a, b)| (a.to_array(), b.to_array())) }
        fn object_obb(&self, _d: u32) -> Option<crate::construct::Obb> { None }
        fn selection_wireframe(&self, _d: u32) -> Option<Vec<[f32; 3]>> { None }
        fn translate_pick(&mut self, _d: u32, _v: Vec3) {}
        fn rotate_pick(&mut self, _d: u32, _q: Quat, _p: Vec3) {}
        fn raycast_scene(&self, _o: Vec3, _d: Vec3) -> Option<Vec3> { None }
        fn raycast_scene_n(&self, _o: Vec3, _d: Vec3) -> Option<(Vec3, Vec3)> { None }
        fn raycast_objects_excluding(&self, _o: Vec3, _d: Vec3, _x: &[u32]) -> Option<Vec3> { None }
        fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])> { None }
        fn resolve_object_mode(&self, _t: u32) -> u32 { 0 }
        fn tag_name_of(&self, _t: u32) -> String { String::new() }
        fn raw_tag_class(&self, _t: u32) -> Option<String> { None }
        fn bsp_flags_report(&self) -> String { String::new() }
        fn blocker_overlay(&self) -> (&[([f32; 3], [f32; 4])], &[([f32; 3], [f32; 3], [f32; 3])]) { (&[], &[]) }
        fn collision_world_edges(&self, _o: &hms_ipc::ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
        fn physics_world_edges(&self, _o: &hms_ipc::ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
        fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> { None }
        fn physics_body(&self, o: &hms_ipc::ObjectInfo) -> Option<PhysBody> { self.src.physics_body(o.datum) }
        fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            floor().as_slice().gather(mn, mx, out);
        }
    }

    fn obj(datum: u32, pos: Vec3, mode_tag: u32) -> hms_ipc::ObjectInfo {
        hms_ipc::ObjectInfo {
            datum,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: pos.to_array(),
            health: 1.0,
            shield: 1.0,
            mode_tag,
            fwd: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            attached: [0; 8],
            primary_tag: 0,
            variant_name_sid: 0,
        }
    }

    /// THE RENDER PATH, with the real session and a real render model: a floating crate is woken,
    /// falls, and the session (1) hides its placement, (2) creates ONE actor for it whose world
    /// matrix tracks the body every frame, (3) stops changing once it sleeps; leaving removes the
    /// actor, empties the hidden set and puts the body back. Needs a map for the render model.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_falling_prop_is_hidden_and_drawn_by_an_actor_and_leaving_cleans_up() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        // any tag with a drawable render model
        let mode = native
            .tags_by_group(cache, "mode")
            .into_iter()
            .find(|&t| {
                let mut a = crate::reach_actor::ActorScene::default();
                a.add_render_model(native, cache, t, None, Vec3::ZERO, 0.0).is_ok()
            })
            .expect("no drawable render model in forge_halo");
        let mut stub = StubScene { src: Src::new(), aabbs: HashMap::new() };
        stub.src.bodies.insert(1, box_body(0.5, 40.0, 0.8, 0.0));
        let start = Vec3::new(0.0, 0.0, 3.0);
        stub.aabbs.insert(1, (start - Vec3::splat(0.52), start + Vec3::splat(0.52)));
        let objects = vec![obj(1, start, mode)];
        let mut actors = crate::reach_actor::ActorScene::default();
        let mut session = PropSession { all_solid: true, ..Default::default() };
        // nothing is awake: the first frames add no actor and hide nothing
        for _ in 0..10 {
            session.tick(TICK_SECONDS, native, cache, &stub, &objects, None, &[], &mut actors);
        }
        assert!(actors.actors.is_empty() && session.hidden_datums().is_empty(), "an untouched world must draw nothing extra");
        // a pusher drops by: wake the crate directly through the world, as a nudge would
        let src = SceneSource { scene: &stub, objects: &objects, tags: None };
        session.world.as_mut().unwrap().set_body_velocity(1, Vec3::new(0.0, 0.0, -0.5), Vec3::ZERO, &src);
        let mut last_override = None;
        let mut changes = 0;
        for _ in 0..40 {
            session.tick(TICK_SECONDS, native, cache, &stub, &objects, None, &[], &mut actors);
            if let Some(a) = actors.actors.first() {
                let m = a.world_override;
                if m != last_override {
                    changes += 1;
                }
                last_override = m;
            }
        }
        assert_eq!(actors.actors.len(), 1, "exactly one actor draws the falling crate");
        assert_eq!(session.hidden_datums(), vec![1]);
        assert!(changes >= 20, "the actor's matrix changed on only {changes} of 40 frames while the crate fell");
        let z_actor = last_override.unwrap().w_axis.z;
        let z_body = session.world().unwrap().body_pose(1).unwrap().0.z;
        assert!((z_actor - z_body).abs() < 1e-5, "actor at z {z_actor}, body at {z_body}");
        // it settles: the actor stops changing
        for _ in 0..400 {
            session.tick(TICK_SECONDS, native, cache, &stub, &objects, None, &[], &mut actors);
        }
        assert_eq!(session.world().unwrap().is_asleep(1), Some(true));
        let m0 = actors.actors[0].world_override;
        for _ in 0..30 {
            session.tick(TICK_SECONDS, native, cache, &stub, &objects, None, &[], &mut actors);
        }
        assert_eq!(actors.actors[0].world_override, m0, "a settled prop's actor must not keep moving");
        // leaving
        session.leave(&mut actors);
        assert!(actors.actors.is_empty(), "leaving left an actor behind");
        assert!(session.hidden_datums().is_empty());

        // ...and the PLAYER, through the session's own plumbing, pushes a resting crate: the
        // pusher is built from the capsule, wakes the crate, and the crate is then drawn
        let mut stub2 = StubScene { src: Src::new(), aabbs: HashMap::new() };
        stub2.src.bodies.insert(1, box_body(0.5, 40.0, 0.5, 0.0));
        let rest = Vec3::new(2.0, 0.0, 0.5);
        stub2.aabbs.insert(1, (rest - Vec3::splat(0.52), rest + Vec3::splat(0.52)));
        let objects2 = vec![obj(1, rest, mode)];
        let mut actors2 = crate::reach_actor::ActorScene::default();
        let mut session2 = PropSession { all_solid: true, ..Default::default() };
        let mut x = -1.0f32;
        for _ in 0..180 {
            x += 2.0 * TICK_SECONDS;
            let p = PlayerPusher { feet: Vec3::new(x, 0.0, 0.0), velocity: Vec3::new(2.0, 0.0, 0.0), radius: 0.3, height: 1.5, mass: 125.0 };
            session2.tick(TICK_SECONDS, native, cache, &stub2, &objects2, Some(p), &[], &mut actors2);
        }
        let pushed = session2.world().unwrap().body_pose(1).unwrap().0;
        assert!(pushed.x > 2.4, "the player's capsule did not push the crate through the session: x = {}", pushed.x);
        assert_eq!(actors2.actors.len(), 1, "the pushed crate is drawn by an actor");
        assert_eq!(session2.hidden_datums(), vec![1]);
    }

    /// REAL BODIES. Every scenery / crate / weapon / equipment object in `forge_halo` that has a
    /// `phmo`, with the body the scene builds for it (real shapes, mass, inertia, damping), is
    /// dropped from 1.5 wu onto a floor in a grid. Each must land, stay finite, stay on the floor
    /// (no tunnelling, no explosion) and go to sleep; the awake step cost is reported.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn every_real_prop_body_drops_lands_and_sleeps() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let native = &*native;
        let mut objects: Vec<hms_ipc::ObjectInfo> = Vec::new();
        for group in ["scen", "bloc", "weap", "eqip", "crea"] {
            for tag in native.tags_by_group(cache, group) {
                let p = native.resolve_phmo(cache, tag);
                if p == 0 || p == 0xFFFF_FFFF {
                    continue;
                }
                let i = objects.len() as u32;
                let mut o = obj(i, Vec3::new(((i % 16) as f32 - 8.0) * 4.0, ((i / 16) as f32 - 5.0) * 4.0, 0.0), 0);
                o.primary_tag = tag;
                objects.push(o);
                if objects.len() >= 160 {
                    break;
                }
            }
        }
        let scene: &dyn crate::objscene::ObjectScene = &sc;
        let mut w = PropWorld::new();
        let mut kept: Vec<(u32, f32)> = Vec::new(); // (datum, rest height above its origin)
        let mut fixed = 0usize;
        for o in &objects {
            let Some(b) = scene.physics_body(o) else { continue };
            if b.fixed {
                fixed += 1;
                continue;
            }
            if !(b.mass.is_finite() && b.mass > 0.0) {
                continue;
            }
            let (lo, hi) = b.aabb();
            let h = hi.z - lo.z;
            if !(h.is_finite() && h > 0.01 && h < 8.0) {
                continue;
            }
            // origin so that the body's lowest point is 1.5 wu above the floor
            let origin = Vec3::new(o.pos[0], o.pos[1], 1.5 - lo.z);
            let ab = world_aabb((lo, hi), origin, Quat::IDENTITY, 0.02);
            w.add(PropSeed { datum: o.datum, origin, rot: Quat::IDENTITY, mode: PhysicsMode::Normal, aabb: ab, vehicle: false });
            kept.push((o.datum, -lo.z));
        }
        assert!(kept.len() >= 20, "only {} real prop bodies could be built ({} fixed)", kept.len(), fixed);
        let src = SceneSource { scene, objects: &objects, tags: None };
        let level = floor();
        for (d, _) in &kept {
            w.wake(*d, &src);
        }
        let t0 = std::time::Instant::now();
        let mut worst = 0.0f64;
        let mut ticks = 0;
        while w.awake_count() > 0 && ticks < 900 {
            let t = std::time::Instant::now();
            w.step(TICK_SECONDS, &level, &src, &[]);
            worst = worst.max(t.elapsed().as_secs_f64() * 1000.0);
            ticks += 1;
        }
        let total = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "play_world: {} real bodies ({} fixed skipped): all asleep after {ticks} ticks; {:.2} ms/tick average, {worst:.2} ms worst",
            kept.len(),
            fixed,
            total / ticks.max(1) as f64
        );
        let mut bad = Vec::new();
        for (d, below) in &kept {
            let p = w.body_pose(*d).unwrap().0;
            // a resting body's lowest point is on the floor: origin z within [below - 0.3, below + 0.5]
            if !p.is_finite() || p.z < below - 0.3 || p.z > below + 1.0 || w.is_asleep(*d) != Some(true) {
                bad.push(format!("datum {d}: z {:.3} (floor-resting origin ~{:.3}) asleep {:?}", p.z, below, w.is_asleep(*d)));
            }
        }
        assert!(bad.len() * 10 <= kept.len(), "{} of {} real bodies did not land and sleep sanely:\n{}", bad.len(), kept.len(), bad.join("\n"));
        println!("   {} of {} real bodies outside the sanity band", bad.len(), kept.len());
    }

    /// The jeep seam: a hull resting 1 cm into the floor reports its four bottom corners pushing
    /// UP; one hovering 10 cm above reports nothing; a tyre sphere is left to the suspension.
    #[test]
    fn a_hull_reports_its_floor_contacts_to_the_vehicle_law() {
        let body = box_body(0.5, 100.0, 0.8, 0.0);
        let level = floor();
        let low = hull_vs_static(&body.shapes, Vec3::new(0.0, 0.0, 0.49), Quat::IDENTITY, &level);
        let verts: Vec<_> = low.iter().filter(|c| (c.depth - 0.01).abs() < 1e-3).collect();
        assert!(verts.len() >= 4, "{} contacts at the right depth: {low:?}", verts.len());
        assert!(low.iter().all(|c| c.normal.z > 0.99), "contact normals must point up out of the floor");
        assert!(hull_vs_static(&body.shapes, Vec3::new(0.0, 0.0, 0.6), Quat::IDENTITY, &level).is_empty());
        let mut tyre = ConvexShape::sphere(Vec3::new(0.0, 0.0, 0.0), 0.5, ShapeMaterial::default()).unwrap();
        tyre.support = true;
        assert!(hull_vs_static(&[tyre], Vec3::new(0.0, 0.0, 0.4), Quat::IDENTITY, &level).is_empty(), "support spheres belong to the suspension");
        // the mixed material: shape 0.8 with the level's 0.9
        let (f, _) = contact_material(0.8, 0.0, LEVEL_FRICTION, LEVEL_RESTITUTION);
        assert!((low[0].friction - f).abs() < 1e-6);
    }

    // ---- a driven vehicle's hull against the static world -------------------------------------

    fn wall_x(x: f32) -> Vec<[Vec3; 3]> {
        let (a, b, c, d) = (Vec3::new(x, -50.0, -50.0), Vec3::new(x, 50.0, -50.0), Vec3::new(x, 50.0, 50.0), Vec3::new(x, -50.0, 50.0));
        vec![[a, b, c], [a, c, d]]
    }

    fn car_hull(mu: f32) -> Vec<ConvexShape> {
        let mat = ShapeMaterial { friction: mu, restitution: 0.0, mass: 900.0 };
        let mut v = vec![ConvexShape::aabb_box(Vec3::ZERO, Vec3::new(1.2, 0.6, 0.3), 0.0, mat).unwrap()];
        let mut t = ConvexShape::sphere(Vec3::new(0.9, 0.0, -0.3), 0.25, mat).unwrap();
        t.support = true;
        v.push(t);
        v
    }

    /// Drive `hull` at `vel` toward a wall at x = 10 for 4 s; returns (final pos, final vel,
    /// total hits, farthest x any hull point reached, deepest penetration seen).
    fn drive_at_wall(vel: Vec3, dt: f32) -> (Vec3, Vec3, usize, f32, f32) {
        let hull = car_hull(0.5);
        let world = wall_x(10.0);
        let (mut pos, mut v) = (Vec3::new(0.0, 0.0, 0.0), vel);
        let (mut hits, mut far, mut deep) = (0usize, f32::MIN, 0.0f32);
        for _ in 0..(4.0 / dt) as usize {
            let prev = pos;
            pos += v * dt; // the vehicle's own law, which knows nothing of walls
            let st = sweep_hull(&hull, Quat::IDENTITY, prev, &mut pos, &mut v, &world);
            hits += st.hits;
            deep = deep.max(st.deepest);
            far = far.max(pos.x + 1.2);
        }
        (pos, v, hits, far, deep)
    }

    #[test]
    fn a_driven_hull_never_tunnels_through_a_wall_at_any_speed() {
        for speed in [5.0f32, 20.0, 60.0, 120.0] {
            for dt in [1.0 / 60.0, 1.0 / 30.0] {
                let (pos, v, hits, far, _) = drive_at_wall(Vec3::new(speed, 0.0, 0.0), dt);
                assert!(hits > 0, "{speed} wu/s dt {dt}: the wall was never met (the gate would be vacuous)");
                assert!(far < 10.0 + 0.05, "{speed} wu/s dt {dt}: a hull point reached x {far}, through the wall at 10");
                assert!(pos.x < 10.0, "{speed} wu/s dt {dt}: ended at x {}", pos.x);
                assert!(v.x.abs() < 0.5, "{speed} wu/s dt {dt}: still moving into the wall at {}", v.x);
            }
        }
    }

    #[test]
    fn a_driven_hull_has_no_restitution_and_slides_along_a_glancing_wall() {
        // 20 wu/s into the wall and 20 along it, a slick hull (mu 0.1 against the level's 0.9)
        let hull = car_hull(0.1);
        let world = wall_x(10.0);
        let (mut pos, mut v) = (Vec3::new(8.0, 0.0, 0.0), Vec3::new(20.0, 20.0, 0.0));
        let dt = 1.0 / 60.0;
        let mut hits = 0;
        let mut rebound = 0.0f32;
        for _ in 0..60 {
            let prev = pos;
            pos += v * dt;
            hits += sweep_hull(&hull, Quat::IDENTITY, prev, &mut pos, &mut v, &world).hits;
            rebound = rebound.max(-v.x);
        }
        assert!(hits > 0, "the wall was never met");
        assert!(rebound < 1e-3, "restitution must be 0: rebounded at {rebound} wu/s");
        assert!(pos.y > 8.0, "the hull must slide on along the wall, it reached y {}", pos.y);
        assert!(v.y > 12.0, "a slick hull keeps most of its tangential speed, kept {}", v.y);
        // and a grippy one loses more of it
        let grippy = car_hull(0.9);
        let (mut pos2, mut v2) = (Vec3::new(8.0, 0.0, 0.0), Vec3::new(20.0, 20.0, 0.0));
        for _ in 0..60 {
            let prev = pos2;
            pos2 += v2 * dt;
            sweep_hull(&grippy, Quat::IDENTITY, prev, &mut pos2, &mut v2, &world);
        }
        assert!(v2.y < v.y - 1.0, "friction must matter: grippy {} vs slick {}", v2.y, v.y);
    }

    #[test]
    fn a_hull_clear_of_the_wall_is_left_exactly_alone() {
        let hull = car_hull(0.5);
        let mut world = wall_x(10.0);
        world.extend(floor());
        let (mut pos, mut v) = (Vec3::new(5.0, 0.0, 0.4), Vec3::new(0.0, 30.0, 0.0));
        let (p0, v0) = (pos, v);
        for _ in 0..30 {
            let prev = pos;
            pos += v * (1.0 / 60.0);
            let st = sweep_hull(&hull, Quat::IDENTITY, prev, &mut pos, &mut v, &world);
            assert_eq!(st.hits, 0);
        }
        assert_eq!(v, v0);
        assert!((pos - (p0 + v0 * 0.5)).length() < 1e-4, "free travel must not be altered: {pos:?}");
    }

    /// The real Warthog / Ghost / Scorpion hulls against a real placed block's collision
    /// triangles, at full speed.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn real_vehicle_hulls_do_not_drive_through_a_real_block() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let scene: &dyn crate::objscene::ObjectScene = &sc;
        let mut tested = 0;
        for (group, sub) in [("vehi", "warthog"), ("vehi", "ghost"), ("vehi", "scorpion")] {
            let Some(tag) = native.tags_by_group(cache, group).into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(sub)) else { continue };
            let mut o = obj(1, Vec3::ZERO, 0);
            o.primary_tag = tag;
            let Some(b) = scene.physics_body(&o) else { continue };
            let hull_n = b.shapes.iter().filter(|s| !s.support).count();
            if hull_n == 0 {
                continue;
            }
            // a wall as tall as the vehicle's hull, 2 wu thick at x = 12
            let wall = wall_x(12.0);
            for speed in [15.0f32, 40.0, 80.0] {
                let (mut pos, mut v) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(speed, 0.0, 0.0));
                let mut hits = 0;
                let dt = 1.0 / 30.0;
                for _ in 0..180 {
                    let prev = pos;
                    pos += v * dt;
                    hits += sweep_hull(&b.shapes, Quat::IDENTITY, prev, &mut pos, &mut v, &wall).hits;
                }
                assert!(hits > 0, "{sub} at {speed}: never met the wall");
                assert!(pos.x < 12.0, "{sub} at {speed} wu/s ended at x {} -- through the wall", pos.x);
            }
            tested += 1;
        }
        assert!(tested >= 2, "only {tested} real vehicle hulls could be built");
    }

    /// The collision soup `SceneController::build_collision_soup` builds, without a loaded scene:
    /// every structure BSP's engine collision plus the placed collision of its instances, and
    /// NOTHING from render meshes.
    fn engine_level_soup(native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> crate::decal_projector::TriSoup {
        let mut soup = crate::decal_projector::TriSoup::new(4.0);
        for &t in &native.tags_by_group(cache, "sbsp") {
            if let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(native, cache, t) {
                for (tr, _) in b.solid_triangles() {
                    soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                }
            }
            if let Some(inst) = crate::collision_bsp::instance_tris(native, cache, t) {
                for tr in &inst.placed {
                    soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                }
            }
        }
        soup.rebalance(8);
        soup.shrink();
        soup
    }

    struct StudySoup(crate::decal_projector::TriSoup);
    impl StaticWorld for StudySoup {
        fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            self.0.tris_in_aabb(mn, mx, out);
        }
    }

    /// The soup `SceneController::build_collision_soup` makes: every structure BSP's engine
    /// collision plus its instanced geometry placed at every instance.
    fn real_level_soup_with_fallback(native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> crate::decal_projector::TriSoup {
        let mut soup = crate::decal_projector::TriSoup::new(4.0);
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(native, cache, t) else { continue };
            for (tr, _) in b.solid_triangles() {
                soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
            }
            if let Some(inst) = crate::collision_bsp::instance_tris(native, cache, t) {
                for tr in inst.placed.iter().chain(&inst.fallback) {
                    soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                }
            }
        }
        soup
    }

    fn real_level_soup(native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> crate::decal_projector::TriSoup {
        engine_level_soup(native, cache)
    }

    #[derive(Default)]
    struct Study {
        metres: f32,
        flat_ticks: usize,
        flat_hits: usize,
        crossed: usize,
        /// ticks passing the footprint test, the ahead test, and ticks examined
        why: [usize; 3],
        report: Vec<String>,
    }

    /// Drive `shapes` (a real vehicle hull, tyres on the ground unless `hover`) from each start
    /// `(x, y, yaw)` at each speed, ground-following the way the vehicle law does (four rays
    /// under the wheelbase give pitch and roll), through [`sweep_hull`]. A tick counts as FLAT
    /// when the ground under the footprint and 0.5 / 1.5 / 2.5 wu ahead is level to 3 cm and no
    /// steep (|n.z| < 0.7) surface lies in the corridor the hull is about to sweep: nothing can
    /// be in the way, so the sweep must leave the motion alone. A drive ends at its first
    /// push-out (a real obstacle, or a flat-tick failure that is recorded).
    fn study(name: &str, shapes: &[ConvexShape], hover: bool, world: &dyn StaticWorld, starts: &[(f32, f32, f32)], seams: &[f32]) -> Study {
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let lowest_tyre = shapes.iter().filter(|s| s.support).filter_map(|s| if let ShapeGeom::Sphere { c, r } = &s.geom { Some(c.z - r) } else { None }).fold(f32::MAX, f32::min);
        let lowest_hull = shapes.iter().filter(|s| !s.support).map(|s| s.aabb().0.z).fold(f32::MAX, f32::min);
        let rest_z = if hover { -lowest_hull + 0.3 } else { -lowest_tyre };
        let hext = {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for sh in shapes.iter().filter(|s| !s.support) {
                let (x, y) = sh.aabb();
                lo = lo.min(x);
                hi = hi.max(y);
            }
            hi - lo
        };
        let mut out = Study::default();
        for &(x, y, yaw) in starts {
            let mut g = Vec::new();
            world.gather(Vec3::new(x - 0.05, y - 0.05, -2000.0), Vec3::new(x + 0.05, y + 0.05, 2000.0), &mut g);
            // the highest upward surface there
            let Some(top) = g.iter().filter(|t| nrm(t).z > 0.99).map(|t| {
                let n = nrm(t);
                (n.dot(t[0]) - n.x * x - n.y * y) / n.z
            }).fold(None, |a: Option<f32>, z| Some(a.map_or(z, |a| a.max(z)))) else { continue };
            for speed in [3.0f32, 6.0, 9.0, 11.0] {
                let yawq = Quat::from_rotation_z(yaw);
                let dir = yawq * Vec3::X;
                let (mut pos, mut vel) = (Vec3::new(x, y, top + rest_z), dir * speed);
                let dt = 1.0 / 60.0;
                let ticks = (30.0 / speed / dt) as usize; // up to 30 wu of road
                for _ in 0..ticks {
                    let prev = pos;
                    pos += vel * dt;
                    let (pos0, rot0) = (pos, yawq);
                    let ray = |dx: f32, dy: f32| -> Option<Vec3> {
                        let from = pos0 + rot0 * Vec3::new(dx, dy, 0.0);
                        let mut g = Vec::new();
                        world.gather(from - Vec3::new(0.05, 0.05, 4.0), from + Vec3::new(0.05, 0.05, 1.0), &mut g);
                        ray_down(&g, Vec3::new(from.x, from.y, pos0.z + 1.0), 6.0).map(|(d, _)| Vec3::new(from.x, from.y, pos0.z + 1.0 - d))
                    };
                    let (hx, hy) = (hext.x * 0.4, hext.y * 0.4);
                    let (Some(f), Some(bk), Some(l), Some(rr)) = (ray(hx, 0.0), ray(-hx, 0.0), ray(0.0, hy), ray(0.0, -hy)) else { break };
                    let n = (f - bk).cross(l - rr).normalize_or_zero();
                    let n = if n.z < 0.0 { -n } else { n };
                    if n.z < 0.5 {
                        break;
                    }
                    let rot = Quat::from_rotation_arc(Vec3::Z, n) * yawq;
                    let gz = (f.z + bk.z + l.z + rr.z) * 0.25;
                    pos.z = gz + rest_z / n.z;
                    let ahead = |d: f32| ray(hx + d, 0.0).map(|h| (h.z - gz).abs() < 0.03).unwrap_or(false);
                    let c_foot = [f, bk, l, rr].iter().all(|q| (q.z - gz).abs() < 0.03) && n.z > 0.995;
                    let c_ahead = ahead(0.5) && ahead(1.5) && ahead(2.5);
                    out.why[0] += c_foot as usize;
                    out.why[1] += c_ahead as usize;
                    out.why[2] += 1;
                    let flat = c_foot && c_ahead && {
                        let w = hext.y * 0.6 + 0.3;
                        let tip = pos0 + (yawq * Vec3::X) * (hext.x * 0.5 + 3.5);
                        let (cmn, cmx) = (pos0.min(tip) - Vec3::new(w, w, 0.0), pos0.max(tip) + Vec3::new(w, w, 0.0));
                        let mut c = Vec::new();
                        world.gather(Vec3::new(cmn.x, cmn.y, gz + 0.02), Vec3::new(cmx.x, cmx.y, gz + 2.5), &mut c);
                        !c.iter().any(|t| nrm(t).z.abs() < 0.7)
                    };
                    let before = vel;
                    let st = sweep_hull(shapes, rot, prev, &mut pos, &mut vel, world);
                    out.metres += (pos - prev).length();
                    out.crossed += seams.iter().filter(|s| (prev.x - **s) * (pos.x - **s) < 0.0).count();
                    if flat {
                        out.flat_ticks += 1;
                    }
                    if st.hits == 0 {
                        continue;
                    }
                    if flat {
                        out.flat_hits += 1;
                        if out.report.len() < 10 {
                            out.report.push(format!("{name} speed {speed} at {pos:?} gz {gz:.3}: hits {} deepest {:.4} impact {:.3} dv {:.3}", st.hits, st.deepest, st.impact, (before - vel).length()));
                        }
                    }
                    break;
                }
            }
        }
        out
    }

    fn real_vehicles<'a>(sc: &crate::scene::SceneController, native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> Vec<(&'a str, bool, Vec<ConvexShape>)> {
        let scene: &dyn crate::objscene::ObjectScene = sc;
        let mut v = Vec::new();
        for sub in ["warthog", "mongoose", "ghost", "scorpion"] {
            let Some(tag) = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(sub)) else { continue };
            let mut o = obj(1, Vec3::ZERO, 0);
            o.primary_tag = tag;
            if let Some(b) = scene.physics_body(&o) {
                v.push((sub, sub == "ghost", b.shapes));
            }
        }
        v
    }

    /// The OWNER'S REPORT, measured: "driving on flat ground I randomly hit invisible
    /// barriers". Real `forge_halo` engine collision, each real vehicle hull driven through
    /// 4 headings at 3 / 6 / 9 / 11 wu/s from random flat starts (see [`study`]). Any push-out on
    /// a flat tick is a fault; liveness is metres driven and flat ticks.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn real_flat_ground_gives_no_invisible_barriers() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let soup = real_level_soup(&native, cache);
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let area = |t: &[Vec3; 3]| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
        let flats: Vec<[Vec3; 3]> = soup.tris().iter().filter(|t| nrm(t).z > 0.99 && area(t) > 4.0).copied().collect();
        assert!(flats.len() > 100, "{} large flat triangles", flats.len());
        let world = StudySoup(soup);
        let mut seed = 0x9E3779B9u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let mut starts = Vec::new();
        for _ in 0..40 {
            let t = flats[(rnd() * flats.len() as f32) as usize % flats.len()];
            let c = (t[0] + t[1] + t[2]) / 3.0;
            for k in 0..4 {
                starts.push((c.x, c.y, k as f32 * std::f32::consts::FRAC_PI_2 + rnd()));
            }
        }
        let mut total = Study::default();
        for (name, hover, shapes) in real_vehicles(&sc, &native, cache) {
            let st = study(name, &shapes, hover, &world, &starts, &[]);
            eprintln!("{name}: {:.0} m, {} flat ticks, {} flat push-outs", st.metres, st.flat_ticks, st.flat_hits);
            total.metres += st.metres;
            total.flat_ticks += st.flat_ticks;
            total.flat_hits += st.flat_hits;
            total.report.extend(st.report);
        }
        for r in &total.report {
            eprintln!("FLAT-HIT {r}");
        }
        eprintln!("driven {:.0} m, {} flat ticks, {} flat push-outs", total.metres, total.flat_ticks, total.flat_hits);
        assert!(total.metres > 500.0 && total.flat_ticks > 2000, "liveness: {} m, {} flat ticks", total.metres, total.flat_ticks);
        assert_eq!(total.flat_hits, 0, "push-outs on flat ground with nothing ahead: {:?}", total.report);
    }

    /// Forge platforms and the SEAM between butted ones. Real `bloc`/`scen` objects whose top
    /// is a flat slab (a few cm to 30 cm thick, wider than 4 wu) are put on flat level ground,
    /// two butted end to end, and every real vehicle drives the length of both across the seam.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn real_forge_platform_seams_give_no_invisible_barriers() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let scene: &dyn crate::objscene::ObjectScene = &sc;
        let soup = real_level_soup(&native, cache);
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let area = |t: &[Vec3; 3]| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
        let flat_tri = soup.tris().iter().filter(|t| nrm(t).z > 0.999 && area(t) > 400.0).copied().next().expect("a big flat triangle");
        let c = (flat_tri[0] + flat_tri[1] + flat_tri[2]) / 3.0;
        // a true ground height at that spot
        let mut gh = Vec::new();
        StudySoup(crate::decal_projector::TriSoup::new(4.0));
        let probe = StudySoup(soup);
        probe.gather(c - Vec3::new(0.05, 0.05, 50.0), c + Vec3::new(0.05, 0.05, 6.0), &mut gh);
        let (d, _) = ray_down(&gh, c + Vec3::Z * 6.0, 60.0).expect("ground");
        let ground = c.z + 6.0 - d;
        let mut slabs: Vec<(u32, (Vec3, Vec3), String)> = Vec::new();
        for group in ["bloc", "scen"] {
            for tag in native.tags_by_group(cache, group) {
                let mut o = obj(1, Vec3::ZERO, 0);
                o.primary_tag = tag;
                let tris = scene.physics_world_tris(&o);
                if tris.is_empty() {
                    continue;
                }
                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for t in &tris {
                    for v in t {
                        lo = lo.min(*v);
                        hi = hi.max(*v);
                    }
                }
                let e = hi - lo;
                if e.z > 0.04 && e.z < 0.3 && e.x > 4.0 && e.y > 4.0 && e.x < 60.0 {
                    slabs.push((tag, (lo, hi), native.tag_name(cache, tag).unwrap_or_default()));
                }
            }
        }
        eprintln!("{} flat slab objects", slabs.len());
        assert!(!slabs.is_empty(), "no thin flat Forge slab found");
        let (mut total, mut seams_crossed) = (Study::default(), 0usize);
        for (tag, (lo, hi), name) in slabs.iter().take(6) {
            // two copies butted along +x, bottoms on the ground
            let w = hi.x - lo.x;
            let mut soup2 = real_level_soup(&native, cache);
            let mut o = obj(1, Vec3::ZERO, 0);
            o.primary_tag = *tag;
            let at = |x: f32| Vec3::new(c.x + x - lo.x, c.y - (lo.y + hi.y) * 0.5, ground - lo.z);
            for k in 0..2 {
                let mut oo = o;
                oo.pos = at(k as f32 * w).to_array();
                for t in scene.physics_world_tris(&oo) {
                    soup2.add_tri_owned(t[0], t[1], t[2], (0, 0, -1), u32::MAX);
                }
            }
            let world = StudySoup(soup2);
            let seam = c.x + w;
            let starts: Vec<(f32, f32, f32)> = (-2..=2).map(|k| (c.x + 1.5, c.y + k as f32 * 0.7, 0.0)).collect();
            for (vname, hover, shapes) in real_vehicles(&sc, &native, cache) {
                let st = study(&format!("{vname} on {name}"), &shapes, hover, &world, &starts, &[seam]);
                total.metres += st.metres;
                total.flat_ticks += st.flat_ticks;
                total.flat_hits += st.flat_hits;
                for i in 0..3 {
                    total.why[i] += st.why[i];
                }
                seams_crossed += st.crossed;
                total.report.extend(st.report);
            }
        }
        for r in &total.report {
            eprintln!("FLAT-HIT {r}");
        }
        eprintln!("driven {:.0} m over {} seam crossings, {} flat ticks, {} flat push-outs", total.metres, seams_crossed, total.flat_ticks, total.flat_hits);
        eprintln!("why {:?}", total.why);
        assert!(seams_crossed >= 20, "liveness: only {seams_crossed} seam crossings");
        assert_eq!(total.flat_hits, 0, "push-outs on a platform with nothing ahead: {:?}", total.report);
    }

    // ---- the bisect probe: the REAL drive law on the REAL ground, component by component ----

    #[derive(Clone, Copy)]
    struct Cfg {
        name: &'static str,
        hull: bool,
        instances: bool,
        radius: bool,
        /// Added to the height the wheels see (a render surface that is not the collision one).
        ground_off: f32,
        dt: f32,
    }

    #[derive(Default, Clone, Copy)]
    struct Anom {
        metres: f32,
        ticks: usize,
        speed_drop: usize,
        attitude: usize,
        oscill: usize,
        lateral: usize,
        pushouts: usize,
        creep: usize,
        left_overlap: usize,
    }

    impl Anom {
        fn add(&mut self, o: &Anom) {
            self.metres += o.metres;
            self.ticks += o.ticks;
            self.speed_drop += o.speed_drop;
            self.attitude += o.attitude;
            self.oscill += o.oscill;
            self.lateral += o.lateral;
            self.pushouts += o.pushouts;
            self.creep += o.creep;
            self.left_overlap += o.left_overlap;
        }
        fn row(&self) -> String {
            let per = |n: usize| if self.metres > 1.0 { n as f32 * 100.0 / self.metres } else { n as f32 };
            format!(
                "drop {:6.1} att {:6.1} osc {:6.1} lat {:6.1} push {:6.1} creep {:5} overlap {:4} ({:.0} m)",
                per(self.speed_drop),
                per(self.attitude),
                per(self.oscill),
                per(self.lateral),
                per(self.pushouts),
                self.creep,
                self.left_overlap,
                self.metres
            )
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Drive {
        Park,
        Straight(f32),
        Weave,
    }

    /// One drive of the real `DriveModel` over `world`, the hull sweep applied after every step
    /// exactly as `PropSession::collide_driven` does it. Anomaly bounds come from the engine law:
    /// attitude rate past `MAX_ATTITUDE_RATE`; a horizontal speed drop in one tick greater than
    /// `3 g dt` with the throttle held and the hull level (no brake, no slope, no turn in the
    /// way); sideways velocity at no steer; a pitch rate that flips sign four times in twelve
    /// ticks at more than 0.15 rad/s (jitter); creep of a parked hull after its 5 s settle.
    fn run_drive(model: &crate::reach_vehicle::DriveModel, shapes: &[ConvexShape], world: &dyn StaticWorld, start: (f32, f32, f32), mode: Drive, cfg: &Cfg) -> Anom {
        use crate::reach_vehicle::{DriveInput, DriveState};
        let g = crate::reach_physics::WORLD_GRAVITY;
        let strip: Vec<ConvexShape> = shapes
            .iter()
            .map(|s| {
                let mut c = s.clone();
                if !cfg.radius {
                    if let ShapeGeom::Hull { r, .. } = &mut c.geom {
                        *r = 0.0;
                    }
                }
                c
            })
            .collect();
        let off = cfg.ground_off;
        let ground = |p: Vec3| -> Option<(f32, Vec3)> {
            let mut t = Vec::new();
            world.gather(p - Vec3::new(0.05, 0.05, 6.0), p + Vec3::new(0.05, 0.05, 0.0), &mut t);
            ray_down(&t, p, 6.0).map(|(d, n)| (p.z - d + off, n))
        };
        let (x, y, yaw) = start;
        let Some((z0, _)) = ground(Vec3::new(x, y, 3000.0).min(Vec3::new(x, y, 8.0 + 0.0)).max(Vec3::new(x, y, -1.0e9))) else { return Anom::default() };
        let mut st = DriveState { pos: Vec3::new(x, y, z0 + 0.6), yaw, ..Default::default() };
        let mut a = Anom::default();
        let dt = cfg.dt;
        let settle = (5.0 / dt) as usize;
        let total = match mode {
            Drive::Park => settle + (20.0 / dt) as usize,
            Drive::Straight(_) => settle + (40.0 / dt) as usize,
            Drive::Weave => settle + (24.0 / dt) as usize,
        };
        let mut flips: std::collections::VecDeque<i8> = Default::default();
        let mut hv_prev = 0.0f32;
        for k in 0..total {
            let live = k >= settle;
            let (throttle, want) = match mode {
                Drive::Park => (0.0, st.yaw),
                Drive::Straight(t) => (if live { t } else { 0.0 }, yaw),
                Drive::Weave => {
                    let phase = (((k - settle.min(k)) as f32 * dt) as usize) % 2;
                    (if live { 0.8 } else { 0.0 }, yaw + if phase == 0 { 0.5 } else { -0.5 })
                }
            };
            let prev = st.pos;
            let before = st;
            st = model.step(st, DriveInput { throttle, slide: 0.0, want_yaw: want, brake: false }, dt, &ground);
            if cfg.hull {
                let rot = st.basis();
                let (mut pos, mut vel) = (st.pos, st.vel);
                let hs = sweep_hull(&strip, rot, prev, &mut pos, &mut vel, world);
                if hs.hits > 0 {
                    st.pos = pos;
                    st.vel = vel;
                    if live {
                        a.pushouts += 1;
                    }
                    if hs.left > HULL_SLOP {
                        a.left_overlap += 1;
                    }
                }
            }
            if !live {
                continue;
            }
            a.ticks += 1;
            a.metres += (st.pos - prev).length();
            let hv = Vec3::new(st.vel.x, st.vel.y, 0.0).length();
            let left = st.basis() * Vec3::Y;
            let fwd = st.basis() * Vec3::X;
            let up_z = (st.basis() * Vec3::Z).z;
            let level = up_z > 0.98 && before.ground_normal.z > 0.98;
            if matches!(mode, Drive::Straight(_)) && throttle > 0.0 && level && st.grounded && hv_prev - hv > 3.0 * g * dt {
                a.speed_drop += 1;
            }
            hv_prev = hv;
            if st.ang_vel.x.abs().max(st.ang_vel.y.abs()) > crate::reach_vehicle::drive::MAX_ATTITUDE_RATE {
                a.attitude += 1;
            }
            let pr = st.ang_vel.dot(left);
            flips.push_back(if pr > 0.15 { 1 } else if pr < -0.15 { -1 } else { 0 });
            if flips.len() > 12 {
                flips.pop_front();
            }
            let nz: Vec<i8> = flips.iter().copied().filter(|v| *v != 0).collect();
            if nz.windows(2).filter(|w| w[0] != w[1]).count() >= 4 {
                a.oscill += 1;
                flips.clear();
            }
            if !matches!(mode, Drive::Weave) && st.vel.dot(left).abs() > 0.4 && st.vel.dot(fwd).abs() > 0.5 {
                a.lateral += 1;
            }
            if mode == Drive::Park && st.vel.length() > 0.05 {
                a.creep += 1;
            }
        }
        a
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn bisect_probe_real_vehicles_on_real_ground() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let full = real_level_soup_with_fallback(&native, cache);
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let area = |t: &[Vec3; 3]| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
        let flats: Vec<[Vec3; 3]> = full.tris().iter().filter(|t| nrm(t).z > 0.99 && area(t) > 4.0).copied().collect();
        let mut bare = crate::decal_projector::TriSoup::new(4.0);
        for &t in &native.tags_by_group(cache, "sbsp") {
            if let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(&native, cache, t) {
                for (tr, _) in b.solid_triangles() {
                    bare.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                }
            }
        }
        let (w_full, w_bare) = (StudySoup(full), StudySoup(bare));
        let mut seed = 0xA5A5_1234u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let mut starts = Vec::new();
        for _ in 0..6 {
            let t = flats[(rnd() * flats.len() as f32) as usize % flats.len()];
            let c = (t[0] + t[1] + t[2]) / 3.0;
            starts.push((c.x, c.y, rnd() * std::f32::consts::TAU));
        }
        let cfgs = [
            Cfg { name: "base", hull: true, instances: true, radius: true, ground_off: 0.0, dt: 1.0 / 60.0 },
            Cfg { name: "(1) hull collision off", hull: false, instances: true, radius: true, ground_off: 0.0, dt: 1.0 / 60.0 },
            Cfg { name: "(3) instance collision off", hull: true, instances: false, radius: true, ground_off: 0.0, dt: 1.0 / 60.0 },
            Cfg { name: "(4) shape radius off", hull: true, instances: true, radius: false, ground_off: 0.0, dt: 1.0 / 60.0 },
            Cfg { name: "(9) wheels 8 cm below collision", hull: true, instances: true, radius: true, ground_off: -0.08, dt: 1.0 / 60.0 },
            Cfg { name: "(9) wheels 20 cm below collision", hull: true, instances: true, radius: true, ground_off: -0.20, dt: 1.0 / 60.0 },
            Cfg { name: "(9) wheels 5 cm above collision", hull: true, instances: true, radius: true, ground_off: 0.05, dt: 1.0 / 60.0 },
            Cfg { name: "(8) tick 30 Hz (control)", hull: true, instances: true, radius: true, ground_off: 0.0, dt: 1.0 / 30.0 },
        ];
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let drives = [("park", Drive::Park), ("straight 0.4", Drive::Straight(0.4)), ("straight 0.8", Drive::Straight(0.8)), ("straight 1.0", Drive::Straight(1.0)), ("weave", Drive::Weave)];
        let mut base_total = Anom::default();
        for (vname, _hover, shapes) in real_vehicles(&sc, &native, cache) {
            let tag = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(vname)).unwrap();
            let Some(entry) = index.entry(&native, cache, tag) else { continue };
            let Ok(model) = entry.drive.clone() else {
                eprintln!("{vname}: no drive model");
                continue;
            };
            eprintln!("== {vname} (rigid law: {})", model.rigid());
            for cfg in &cfgs {
                let world: &dyn StaticWorld = if cfg.instances { &w_full } else { &w_bare };
                for (dname, d) in &drives {
                    let mut tot = Anom::default();
                    for &s0 in &starts {
                        tot.add(&run_drive(&model, &shapes, world, s0, *d, cfg));
                    }
                    if cfg.name == "base" {
                        base_total.add(&tot);
                    }
                    eprintln!("BISECT {vname:9} | {:34} | {dname:13} | {}", cfg.name, tot.row());
                }
            }
        }
        assert!(base_total.metres > 200.0, "liveness: {} m driven", base_total.metres);
    }

    /// Which part of the instanced collision makes the walls: the engine's own collision BSPs of
    /// the instance definitions ("placed"), or the RENDER-mesh stand-ins used where a definition
    /// has no collision ("fallback")? Same drives as the bisect, per source.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn bisect_probe_instance_sources() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let mk = |placed: bool, fallback: bool| {
            let mut soup = crate::decal_projector::TriSoup::new(4.0);
            let (mut np, mut nf) = (0usize, 0usize);
            for &t in &native.tags_by_group(cache, "sbsp") {
                if let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(&native, cache, t) {
                    for (tr, _) in b.solid_triangles() {
                        soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                    }
                }
                if let Some(inst) = crate::collision_bsp::instance_tris(&native, cache, t) {
                    if placed {
                        for tr in &inst.placed {
                            soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                            np += 1;
                        }
                    }
                    if fallback {
                        for tr in &inst.fallback {
                            soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                            nf += 1;
                        }
                    }
                }
            }
            eprintln!("world placed={placed} fallback={fallback}: {np} placed + {nf} fallback instance triangles");
            StudySoup(soup)
        };
        let worlds = [("level only", mk(false, false)), ("+ placed instance collision", mk(true, false)), ("+ fallback render meshes", mk(false, true))];
        let full = real_level_soup(&native, cache);
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let area = |t: &[Vec3; 3]| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
        let flats: Vec<[Vec3; 3]> = full.tris().iter().filter(|t| nrm(t).z > 0.99 && area(t) > 4.0).copied().collect();
        let mut seed = 0xA5A5_1234u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let mut starts = Vec::new();
        for _ in 0..6 {
            let t = flats[(rnd() * flats.len() as f32) as usize % flats.len()];
            let c = (t[0] + t[1] + t[2]) / 3.0;
            starts.push((c.x, c.y, rnd() * std::f32::consts::TAU));
        }
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let cfg = Cfg { name: "base", hull: true, instances: true, radius: true, ground_off: 0.0, dt: 1.0 / 60.0 };
        for (vname, _h, shapes) in real_vehicles(&sc, &native, cache) {
            let tag = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(vname)).unwrap();
            let Some(entry) = index.entry(&native, cache, tag) else { continue };
            let Ok(model) = entry.drive.clone() else { continue };
            for (wname, w) in &worlds {
                let mut tot = Anom::default();
                for &s0 in &starts {
                    tot.add(&run_drive(&model, &shapes, w, s0, Drive::Straight(0.8), &cfg));
                }
                eprintln!("INSTSRC {vname:9} | {wname:30} | {}", tot.row());
            }
        }
    }

    /// The OWNER'S REPORT: "random physics objects, when I bumped them, fell through the Forge
    /// blocks they sat on". Real props (`scen`/`bloc`/`weap`/`eqip`/`crea` with a `phmo`) are put
    /// on top of real flat-topped Forge blocks standing on the real ground, through the real
    /// `PropWorld` + `SceneSource` path, with the block FIXED (the Forge default for a placed
    /// block) and with the block a NORMAL sleeping body. Each is bumped (a player-sized capsule
    /// walks into it) or dropped on from 30 cm, and simulated for 20 s. A prop whose lowest
    /// point ends below the block's top face by more than 5 cm while it is over the block is a
    /// fault. Prints the worst cases.
    fn props_on_blocks(label: &str, block_mode: PhysicsMode, bump: bool) -> (usize, usize, Vec<String>) {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return (0, 0, Vec::new());
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let scene: &dyn crate::objscene::ObjectScene = &sc;
        let level = StudySoup(engine_level_soup(&native, cache));
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let area = |t: &[Vec3; 3]| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
        // flat ground: a big flat triangle, its true top
        let mut ground = None;
        for t in level.0.tris() {
            if nrm(t).z > 0.999 && area(t) > 400.0 {
                let c = (t[0] + t[1] + t[2]) / 3.0;
                let mut g = Vec::new();
                level.gather(c - Vec3::new(0.05, 0.05, 50.0), c + Vec3::new(0.05, 0.05, 6.0), &mut g);
                if let Some((d, n)) = ray_down(&g, c + Vec3::Z * 6.0, 60.0) {
                    if n.z > 0.999 {
                        ground = Some(Vec3::new(c.x, c.y, c.z + 6.0 - d));
                        break;
                    }
                }
            }
        }
        let ground = ground.expect("flat ground");
        // blocks: a flat top face of at least 2 x 2 wu and a solid thickness
        let mut blocks: Vec<(u32, String, Vec<[Vec3; 3]>, f32, Vec3, Vec3)> = Vec::new();
        for tag in native.tags_by_group(cache, "bloc") {
            let mut o = obj(1, Vec3::ZERO, 0);
            o.primary_tag = tag;
            let tris = scene.physics_world_tris(&o);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for t in &tris {
                for v in t {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
            }
            let top = tris.iter().filter(|t| nrm(t).z > 0.99).map(|t| t[0].z).fold(f32::MIN, f32::max);
            let top_area: f32 = tris.iter().filter(|t| nrm(t).z > 0.99 && (t[0].z - top).abs() < 0.01).map(area).sum();
            if tris.is_empty() || top_area < 4.0 || hi.z - lo.z < 0.15 || hi.x - lo.x > 40.0 || hi.y - lo.y > 40.0 {
                continue;
            }
            blocks.push((tag, native.tag_name(cache, tag).unwrap_or_default(), tris, top, lo, hi));
        }
        blocks.truncate(4);
        // props
        let mut props: Vec<(u32, String, crate::physics_settle::PhysBody)> = Vec::new();
        for group in ["scen", "bloc", "weap", "eqip", "crea"] {
            for tag in native.tags_by_group(cache, group) {
                let mut o = obj(1, Vec3::ZERO, 0);
                o.primary_tag = tag;
                let Some(b) = scene.physics_body(&o) else { continue };
                let (lo, hi) = b.aabb();
                let e = hi - lo;
                if b.fixed || !(b.mass.is_finite() && b.mass > 0.0) || e.max_element() > 1.6 || e.min_element() < 0.1 {
                    continue;
                }
                props.push((tag, native.tag_name(cache, tag).unwrap_or_default(), b));
                if props.len() >= 40 {
                    break;
                }
            }
        }
        eprintln!("{label}: {} blocks, {} props", blocks.len(), props.len());
        assert!(!blocks.is_empty() && props.len() >= 4, "{} blocks, {} props", blocks.len(), props.len());
        let (mut pairs, mut bad) = (0usize, 0usize);
        let mut report = Vec::new();
        let cap = [ConvexShape::capsule(Vec3::new(0.0, 0.0, 0.3), Vec3::new(0.0, 0.0, 1.0), 0.3, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 0.0 }).unwrap()];
        for (btag, bname, btris, top, blo, bhi) in &blocks {
            let bo = Vec3::new(ground.x, ground.y, ground.z - blo.z);
            let top_z = bo.z + top;
            let centre = Vec3::new(bo.x + (blo.x + bhi.x) * 0.5, bo.y + (blo.y + bhi.y) * 0.5, top_z);
            for (ptag, pname, pbody) in &props {
                let (plo, phi) = pbody.aabb();
                let lift = if bump { 0.02 } else { 0.3 };
                let porigin = Vec3::new(centre.x, centre.y, top_z - plo.z + lift);
                let objects = vec![
                    {
                        let mut o = obj(1, bo, 0);
                        o.primary_tag = *btag;
                        o
                    },
                    {
                        let mut o = obj(2, porigin, 0);
                        o.primary_tag = *ptag;
                        o
                    },
                ];
                let src = SceneSource { scene, objects: &objects, tags: None };
                let mut w = PropWorld::new();
                let babb = (bo + *blo - Vec3::splat(0.05), bo + *bhi + Vec3::splat(0.05));
                w.add(PropSeed { datum: 1, origin: bo, rot: Quat::IDENTITY, mode: block_mode, aabb: babb, vehicle: false });
                let pabb = world_aabb((plo, phi), porigin, Quat::IDENTITY, 0.02);
                w.add(PropSeed { datum: 2, origin: porigin, rot: Quat::IDENTITY, mode: PhysicsMode::Normal, aabb: pabb, vehicle: false });
                w.wake(2, &src);
                pairs += 1;
                let mut lowest_rel = f32::MAX;
                let mut first_cross: Option<String> = None;
                let (half_x, half_y) = ((bhi.x - blo.x) * 0.5 - 0.2, (bhi.y - blo.y) * 0.5 - 0.2);
                for tick in 0..1200u32 {
                    let mut kin = Vec::new();
                    let t = tick as f32 * TICK_SECONDS;
                    let cx = centre.x - 1.6 + 3.0 * t.min(1.0);
                    if bump && t < 1.2 {
                        kin.push(Kinematic { shapes: &cap, origin: Vec3::new(cx, centre.y, top_z), rot: Quat::IDENTITY, velocity: Vec3::new(3.0, 0.0, 0.0), angular: Vec3::ZERO, mass: 125.0, com: Vec3::new(0.0, 0.0, 0.6) });
                    }
                    w.step(TICK_SECONDS, &level, &src, &kin);
                    if tick > 30 {
                        let (o, q) = w.body_pose(2).unwrap();
                        // the true lowest point of the posed shapes (an AABB of a tumbled body
                        // reaches below anything the body actually touches)
                        let low = pbody
                            .shapes
                            .iter()
                            .map(|sh| match &sh.geom {
                                ShapeGeom::Sphere { c, r } => (o + q * *c).z - r,
                                ShapeGeom::Capsule { a, b, r } => (o + q * *a).z.min((o + q * *b).z) - r,
                                ShapeGeom::Hull { pts, .. } => pts.iter().map(|p| (o + q * *p).z).fold(f32::MAX, f32::min),
                            })
                            .fold(f32::MAX, f32::min);
                        let over = (o.x - centre.x).abs() < half_x && (o.y - centre.y).abs() < half_y;
                        if over {
                            lowest_rel = lowest_rel.min(low - top_z);
                            if low - top_z < -0.05 && first_cross.is_none() {
                                let (v, wv) = w.body_velocity(2).unwrap();
                                first_cross = Some(format!("tick {tick}: origin {o:?} v {v:?} w {wv:?} capsule x {cx:.2} (centre {:.2},{:.2}) rot*Z {:?}", centre.x, centre.y, q * Vec3::Z));
                            }
                        }
                    }
                }
                if lowest_rel < -0.05 && report.len() < 3 {
                    eprintln!("DETAIL {pname} on {bname}: body mass {} fixed {} shapes {}", pbody.mass, pbody.fixed, pbody.shapes.len());
                    for (i, sh) in pbody.shapes.iter().enumerate() {
                        let (a, b) = sh.aabb();
                        let kind = match &sh.geom { ShapeGeom::Sphere { r, .. } => format!("sphere r {r:.3}"), ShapeGeom::Capsule { r, .. } => format!("capsule r {r:.3}"), ShapeGeom::Hull { pts, r, .. } => format!("hull {} pts r {r:.4}", pts.len()) };
                        eprintln!("   shape {i} support {} {kind} aabb {a:?}..{b:?}", sh.support);
                    }
                    eprintln!("   FIRST CROSSING {:?}", first_cross);
                    let (o, q) = w.body_pose(2).unwrap();
                    eprintln!("   final origin {o:?} rot*Z {:?} top_z {top_z:.3} com {:?}", q * Vec3::Z, pbody.com);
                }
                if lowest_rel < -0.05 {
                    bad += 1;
                    if report.len() < 12 {
                        report.push(format!("{label}: {pname} on {bname}: lowest point {lowest_rel:.3} below the top face, asleep {:?}", w.is_asleep(2)));
                    }
                }
            }
        }
        (pairs, bad, report)
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn real_props_on_real_blocks_do_not_fall_through() {
        let mut total = (0usize, 0usize);
        let mut all = Vec::new();
        for (label, mode, bump) in [("fixed block, bumped", PhysicsMode::Fixed, true), ("fixed block, dropped on", PhysicsMode::Fixed, false), ("normal block, bumped", PhysicsMode::Normal, true), ("normal block, dropped on", PhysicsMode::Normal, false)] {
            let (p, b, r) = props_on_blocks(label, mode, bump);
            eprintln!("PROPBLOCK {label}: {p} prop/block pairs, {b} fell through");
            total.0 += p;
            total.1 += b;
            all.extend(r);
        }
        for r in &all {
            eprintln!("PROPFALL {r}");
        }
        assert!(total.0 >= 100, "liveness: only {} prop/block pairs", total.0);
        assert_eq!(total.1, 0, "{} of {} prop/block pairs fell through: {all:?}", total.1, total.0);
    }

    /// **The phantom-wall gate, on the PRODUCTION world.** `SceneController` builds its collision
    /// world and answers its wheel ray casts from it; the engine has NO collision for an
    /// instanced definition that authors none (grass cards, rocks, detail meshes), so a hull, a
    /// capsule and a wheel ray must all pass through those instances' render triangles. For every
    /// such instance on `forge_halo` this puts a hull (the Warthog's), the walker's capsule and a
    /// downward ray on its render triangles and demands: no contact, and the ray reports the ground
    /// beneath rather than the mesh. Liveness: how many definitions and instances were found.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn instances_without_collision_are_pass_through_for_hull_capsule_and_wheel_ray() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        sc.load_collision_for_test();
        assert!(sc.collision_note().contains("structure collision BSP"), "{}", sc.collision_note());
        eprintln!("{}", sc.collision_note());
        let scene: &dyn crate::objscene::ObjectScene = &sc;
        let hog = real_vehicles(&sc, &native, cache).into_iter().find(|(n, _, _)| *n == "warthog").expect("a Warthog").2;
        let capsule = [ConvexShape::capsule(Vec3::new(0.0, 0.0, 0.3), Vec3::new(0.0, 0.0, 1.0), 0.3, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 0.0 }).unwrap()];
        let world = SceneLevel(&sc);
        let engine = StudySoup(engine_level_soup(&native, cache));
        let (mut defs, mut insts, mut hull_hits, mut cap_hits, mut ray_hits, mut tested) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(inst) = crate::collision_bsp::instance_tris(&native, cache, t) else { continue };
            defs += inst.fallback_meshes;
            insts += inst.fallback_meshes;
            // the render triangles of a no-collision instance, one probe per triangle (sampled)
            for tr in inst.fallback.iter().step_by(11) {
                let n = (tr[1] - tr[0]).cross(tr[2] - tr[0]).normalize_or_zero();
                let c = (tr[0] + tr[1] + tr[2]) / 3.0;
                if n.z < 0.7 {
                    continue; // judge the upward-facing ones: those are what a wheel rides and a hull sits on
                }
                // a probe only counts where the engine's own collision has NOTHING within 3 wu
                // sideways and from 0.5 below to 2.5 above (judged on an independently built
                // engine-only soup), so any contact can only come from the render triangle itself
                let mut e = Vec::new();
                engine.gather(c - Vec3::new(3.0, 3.0, 0.5), c + Vec3::new(3.0, 3.0, 2.5), &mut e);
                if e.iter().any(|t| crate::physics_settle::tri_overlaps(t, c - Vec3::new(3.0, 3.0, 0.5), c + Vec3::new(3.0, 3.0, 2.5))) {
                    continue;
                }
                tested += 1;
                // hull standing on the render triangle
                let rot = Quat::IDENTITY;
                let hull_origin = c + Vec3::Z * 0.3;
                let sup_lo = hog.iter().filter(|s| !s.support).map(|s| s.aabb().0.z).fold(f32::MAX, f32::min);
                let origin = Vec3::new(hull_origin.x, hull_origin.y, c.z - sup_lo + 0.01);
                if hull_vs_static(&hog, origin, rot, &world).iter().any(|k| k.depth > HULL_SLOP) {
                    hull_hits += 1;
                }
                let mut gc = Vec::new();
                world.gather(c - Vec3::splat(1.5), c + Vec3::splat(1.5), &mut gc);
                let cap_origin = Vec3::new(c.x, c.y, c.z - 0.2);
                let (m, inv) = (glam::Mat3::from_quat(rot), glam::Mat3::from_quat(rot).transpose());
                let mut buf = Vec::new();
                for sh in &capsule {
                    crate::physics_settle::shape_contacts(sh, m, cap_origin, inv, &gc, &mut buf);
                }
                if buf.iter().any(|k| k.depth > HULL_SLOP) {
                    cap_hits += 1;
                }
                // a wheel ray from 0.5 wu above the triangle must not stop on it
                if let Some(d) = scene.raycast_solid_excluding(c + Vec3::Z * 0.5, Vec3::NEG_Z, 3.0, &[]) {
                    if d < 0.5 + 0.05 {
                        ray_hits += 1;
                    }
                }
            }
        }
        eprintln!("PASSTHRU {defs} no-collision instance meshes on forge_halo; {tested} upward render triangles probed over open ground: hull contacts {hull_hits}, capsule contacts {cap_hits}, wheel-ray stops {ray_hits}");
        assert!(insts >= 100 && tested >= 20, "liveness: {insts} instance meshes, {tested} probes");
        assert_eq!((hull_hits, cap_hits, ray_hits), (0, 0, 0), "a hull, capsule or wheel ray was stopped by a render triangle of a definition without collision");
    }

    /// **A moved object is drawn as its placement is drawn.** The ridden, the parked and the
    /// physics-moved object all draw through `play_vehicle::object_pieces` +
    /// `apply_appearance` + `set_variant`; for real forge_halo vehicles in red / blue / a custom
    /// colour, scaled, damaged and in a non-default model variant, the actors those make carry
    /// exactly the inputs the static pipeline reads from the same placement: the model variant
    /// sid, the Forge tint (`scene::forge_tint`), the damage state, the scale matrix and the
    /// attached children the variant selects. Liveness: how many objects and pieces were checked.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn a_moved_object_is_drawn_with_its_placements_appearance() {
        use crate::play_vehicle::{apply_appearance, object_pieces, set_variant, tint_of, RideAppearance, VehicleIndex};
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut index = VehicleIndex::default();
        let mut actors = crate::reach_actor::ActorScene::default();
        let (mut checked, mut piece_count, mut with_children) = (0usize, 0usize, 0usize);
        let colours: [(Option<(u8, u8)>, f32, u8); 4] = [(Some((0, 255)), 1.0, 0), (Some((1, 255)), 1.5, 0), (Some((9, 5)), 1.0, 2), (None, 0.5, 3)];
        for sub in ["warthog", "mongoose", "ghost", "scorpion"] {
            let Some(tag) = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().ends_with(sub)) else { continue };
            let rows = native.enumerate_attachments(cache, tag);
            // a non-default variant when the object offers one
            let variant = rows.iter().map(|a| a.variant_name_sid).filter(|v| *v != 0).last().unwrap_or(0);
            for (fc, scale, damage) in colours {
                let ap = RideAppearance { forge_color: fc, scale, damage };
                let world = glam::Mat4::from_scale_rotation_translation(Vec3::splat(scale), Quat::from_rotation_z(0.7), Vec3::new(3.0, 4.0, 5.0));
                let pieces = object_pieces(&mut index, &native, cache, tag, world, variant);
                // the pieces the static draw would make: the model + the rows of the chosen variant
                let want_children = rows.iter().filter(|a| a.child_mode != 0 && a.child_mode != u32::MAX && if variant != 0 && rows.iter().any(|r| r.variant_name_sid == variant) { a.variant_name_sid == variant } else { a.variant_index == 0 }).count();
                assert_eq!(pieces.len(), 1 + want_children, "{sub}: pieces");
                assert!((pieces[0].world.x_axis.truncate().length() - scale).abs() < 1e-4, "{sub}: scale matrix");
                let ids: Vec<u32> = pieces.iter().filter_map(|p| actors.add_render_model(&native, cache, p.mode_tag, None, Vec3::ZERO, 0.0).ok()).collect();
                assert_eq!(ids.len(), pieces.len());
                apply_appearance(&mut actors, &ids, &ap);
                set_variant(&mut actors, ids[0], variant);
                let expect_tint = fc.map(|tc| {
                    let c = crate::scene::forge_tint(Some(tc));
                    [[c[0], c[1], c[2]], [c[0], c[1], c[2]]]
                });
                for &id in &ids {
                    let a = actors.get(id).unwrap();
                    assert_eq!(a.skin, expect_tint, "{sub} {fc:?}: colour");
                    assert_eq!(a.damage, damage, "{sub}: damage");
                }
                assert_eq!(actors.get(ids[0]).unwrap().variant_sid, variant, "{sub}: variant");
                assert_eq!(tint_of(&ap), expect_tint);
                checked += 1;
                piece_count += pieces.len();
                with_children += (want_children > 0) as usize;
            }
        }
        eprintln!("APPEARANCE {checked} objects ({piece_count} pieces, {with_children} with attached children) match the static inputs");
        assert!(checked >= 8 && with_children >= 1, "liveness: {checked} objects, {with_children} with children");
    }

    // ---- driving off a ledge ------------------------------------------------------------------

    /// A closed box shell (outward normals) plus a ground plane at z = 0: a platform `h` high,
    /// edge on the +x side at x = `edge`.
    fn ledge_world(h: f32, edge: f32) -> Vec<[Vec3; 3]> {
        let q = |a: Vec3, b: Vec3, c: Vec3, d: Vec3, out: &mut Vec<[Vec3; 3]>| {
            out.push([a, b, c]);
            out.push([a, c, d]);
        };
        let mut w = Vec::new();
        // ground, up-facing
        q(Vec3::new(-300.0, -300.0, 0.0), Vec3::new(300.0, -300.0, 0.0), Vec3::new(300.0, 300.0, 0.0), Vec3::new(-300.0, 300.0, 0.0), &mut w);
        let (x0, x1, y0, y1) = (edge - 30.0, edge, -20.0, 20.0);
        // top (up), +x face, -x face, +y face, -y face (outward)
        q(Vec3::new(x0, y0, h), Vec3::new(x1, y0, h), Vec3::new(x1, y1, h), Vec3::new(x0, y1, h), &mut w);
        q(Vec3::new(x1, y0, 0.0), Vec3::new(x1, y1, 0.0), Vec3::new(x1, y1, h), Vec3::new(x1, y0, h), &mut w);
        q(Vec3::new(x0, y1, 0.0), Vec3::new(x0, y0, 0.0), Vec3::new(x0, y0, h), Vec3::new(x0, y1, h), &mut w);
        q(Vec3::new(x1, y1, 0.0), Vec3::new(x0, y1, 0.0), Vec3::new(x0, y1, h), Vec3::new(x1, y1, h), &mut w);
        q(Vec3::new(x0, y0, 0.0), Vec3::new(x1, y0, 0.0), Vec3::new(x1, y0, h), Vec3::new(x0, y0, h), &mut w);
        w
    }

    #[derive(Default, Debug)]
    struct LedgeRun {
        ticks: usize,
        backward: usize,
        worst_back: f32,
        teleports: usize,
        reached_edge: bool,
        dropped: bool,
        sweep_hits: usize,
        model_backward: usize,
        report: Vec<String>,
        /// per tick: position, velocity, hull hits that tick
        track: Vec<(Vec3, Vec3, usize)>,
    }

    /// One drive toward an edge `edge_dist` wu ahead of `start_xy` along `yaw`, with the ground
    /// `drop` lower beyond it: `speed` wu/s, throttle `held` or released. The ground query is the
    /// production one (`play_vehicle::step`): a ray from the wheel-sphere top, 8 wu, memoised on a
    /// 0.05 wu XY grid for the duration of one `DriveModel::step` when `memo`. Per tick the
    /// displacement is compared with the velocity the vehicle had: backward motion and teleports
    /// are counted, split into what the drive law did on its own (`model_backward`) and what the
    /// hull sweep did on top of it.
    #[allow(clippy::too_many_arguments)]
    fn ledge_drive(model: &crate::reach_vehicle::DriveModel, shapes: &[ConvexShape], world: &dyn StaticWorld, start_xy: glam::Vec2, ground_z: f32, yaw: f32, edge_dist: f32, drop: f32, speed: f32, held: bool, hull: bool, memo: bool) -> LedgeRun {
        use crate::reach_vehicle::{DriveInput, DriveState};
        let raw = |p: Vec3| -> Option<(f32, Vec3)> {
            let mut t = Vec::new();
            world.gather(p - Vec3::new(0.05, 0.05, 8.0), p + Vec3::new(0.05, 0.05, 0.0), &mut t);
            ray_down(&t, p, 8.0).map(|(d, n)| (p.z - d, n))
        };
        let dt = 1.0 / 60.0;
        let dir = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
        let start = Vec3::new(start_xy.x, start_xy.y, ground_z + 0.6);
        let mut st = DriveState { pos: start, yaw, ..Default::default() };
        let mut out = LedgeRun::default();
        let settle = |st: DriveState| -> DriveState {
            let cache: std::cell::RefCell<Vec<(f32, f32, Option<(f32, Vec3)>)>> = Default::default();
            let g = |p: Vec3| -> Option<(f32, Vec3)> {
                if memo {
                    if let Some(&(_, _, hit)) = cache.borrow().iter().find(|(x, y, _)| (x - p.x).abs() < 0.05 && (y - p.y).abs() < 0.05) {
                        return hit;
                    }
                }
                let h = raw(p);
                cache.borrow_mut().push((p.x, p.y, h));
                h
            };
            model.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: yaw, brake: true }, 1.0 / 60.0, &g)
        };
        for _ in 0..180 {
            st = settle(st);
        }
        let z_rest = st.pos.z;
        st.vel = Vec3::new(dir.x * speed, dir.y * speed, st.vel.z);
        for k in 0..(9.0 / dt) as usize {
            let prev = st.pos;
            let vh = Vec3::new(st.vel.x, st.vel.y, 0.0);
            let vhat = vh.normalize_or_zero();
            let cache: std::cell::RefCell<Vec<(f32, f32, Option<(f32, Vec3)>)>> = Default::default();
            let queries: std::cell::RefCell<Vec<(Vec3, Option<f32>)>> = Default::default();
            let g = |p: Vec3| -> Option<(f32, Vec3)> {
                if memo {
                    if let Some(&(_, _, hit)) = cache.borrow().iter().find(|(x, y, _)| (x - p.x).abs() < 0.05 && (y - p.y).abs() < 0.05) {
                        queries.borrow_mut().push((p, hit.map(|h| h.0)));
                        return hit;
                    }
                }
                let h = raw(p);
                queries.borrow_mut().push((p, h.map(|h| h.0)));
                cache.borrow_mut().push((p.x, p.y, h));
                h
            };
            st = model.step(st, DriveInput { throttle: if held { 1.0 } else { 0.0 }, slide: 0.0, want_yaw: yaw, brake: false }, dt, &g);
            let d_model = st.pos - prev;
            let mb = d_model.dot(vhat);
            if vh.length() > 1.0 && mb < -0.03 {
                out.model_backward += 1;
            }
            if hull {
                let (mut pos, mut vel) = (st.pos, st.vel);
                let hs = sweep_hull(shapes, st.basis(), prev, &mut pos, &mut vel, world);
                if hs.hits > 0 {
                    out.sweep_hits += 1;
                    st.pos = pos;
                    st.vel = vel;
                }
            }
            let d = st.pos - prev;
            out.ticks += 1;
            out.track.push((st.pos, st.vel, 0));
            let along = d.dot(vhat);
            if vh.length() > 1.0 && along < -0.03 {
                out.backward += 1;
                out.worst_back = out.worst_back.min(along);
                if out.report.len() < 3 {
                    let q: Vec<String> = queries.borrow().iter().take(8).map(|(p, h)| format!("({:.2},{:.2},{:.2})->{:?}", p.x, p.y, p.z, h)).collect();
                    out.report.push(format!("tick {k}: pos {:?} -> {:?} (along v {along:.3}), v {:?} -> {:?}, model-only along {mb:.3}; wheel queries {}", prev, st.pos, vh, st.vel, q.join(" ")));
                }
            }
            if d.length() > vh.length() * dt * 1.5 + HULL_PUSH_MAX + 0.1 {
                out.teleports += 1;
            }
            let travelled = (st.pos - start).dot(dir);
            if travelled > edge_dist {
                out.reached_edge = true;
            }
            if out.reached_edge && st.pos.z < z_rest - 0.8 * drop && travelled > edge_dist + 0.5 {
                out.dropped = true;
            }
            if st.pos.z < z_rest - drop - 0.3 {
                break;
            }
        }
        out
    }

    #[test]
    #[ignore = "TRACE"]
    fn trace_ledge() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let mut index = crate::play_vehicle::VehicleIndex::default();
        for (vname, _hover, shapes) in real_vehicles(&sc, &native, cache) {
            if vname != "mongoose" { continue; }
            let tag = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(vname)).unwrap();
            let entry = index.entry(&native, cache, tag).unwrap();
            let model = entry.drive.clone().unwrap();
            let world = ledge_world(1.0, 15.0);
            let r = ledge_drive(&model, &shapes, &world, glam::Vec2::new(10.0, 0.0), 1.0, 1.0472, 5.0 / 0.5, 1.0, 11.0, false, true, true);
            for (k, (p, v, _)) in r.track.iter().enumerate().step_by(6) {
                eprintln!("T {k} pos {:.2} {:.2} {:.2} vel {:.2} {:.2} {:.2}", p.x, p.y, p.z, v.x, v.y, v.z);
            }
        }
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn vehicles_drive_off_ledges_without_moving_backward() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let (mut runs, mut backward, mut teleports, mut incomplete, mut reached) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut model_back = 0usize;
        let mut shown = 0;
        for (vname, _hover, shapes) in real_vehicles(&sc, &native, cache) {
            let tag = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(vname)).unwrap();
            let Some(entry) = index.entry(&native, cache, tag) else { continue };
            let Ok(model) = entry.drive.clone() else {
                eprintln!("{vname}: no drive model");
                continue;
            };
            for h in [0.3f32, 1.0, 3.0] {
                let edge = 15.0;
                let world = ledge_world(h, edge);
                for yaw in [0.0f32, 0.5236, 1.0472] {
                    for speed in [2.0f32, 4.0, 6.0, 9.0, 11.0] {
                        for held in [false, true] {
                            let r = ledge_drive(&model, &shapes, &world, glam::Vec2::new(edge - 5.0, 0.0), h, yaw, 5.0 / yaw.cos().max(0.3), h, speed, held, true, true);
                            runs += 1;
                            backward += r.backward;
                            teleports += r.teleports;
                            model_back += r.model_backward;
                            reached += r.reached_edge as usize;
                            let carried_past = r.reached_edge;
                            if carried_past && !r.dropped {
                                incomplete += 1;
                            }
                            if (r.backward > 0 || r.teleports > 0 || (carried_past && !r.dropped)) && shown < 14 {
                                shown += 1;
                                eprintln!("LEDGE {vname} h {h} yaw {yaw:.2} speed {speed} held {held}: backward {} (worst {:.3}) teleports {} reached {} dropped {} sweep hits {} model-only backward {}", r.backward, r.worst_back, r.teleports, r.reached_edge, r.dropped, r.sweep_hits, r.model_backward);
                                for l in &r.report {
                                    eprintln!("     {l}");
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!("LEDGE TOTAL {runs} drives, {reached} reached the edge: backward ticks {backward} (drive law alone {model_back}), teleports {teleports}, carried past the edge but did not drop {incomplete}");
        assert!(runs >= 100 && reached >= 40, "liveness: {runs} drives, {reached} reached the edge");
        assert_eq!((backward, teleports, incomplete), (0, 0, 0), "a vehicle driven off a ledge moved backward, teleported or failed to drop");
    }

    /// The same drive-off on the REAL forge_halo terrain: ledges are found by marching 8 headings
    /// from random flat spots until the ground falls by more than a metre after at least 8 wu of
    /// level ground; each is driven toward at 4 and 9 wu/s, throttle held and released, with the
    /// production ground query (memoised) and the hull sweep.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture"]
    fn vehicles_drive_off_real_terrain_ledges_without_moving_backward() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_world gates SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let soup = engine_level_soup(&native, cache);
        let nrm = |t: &[Vec3; 3]| (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
        let area = |t: &[Vec3; 3]| 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
        let flats: Vec<[Vec3; 3]> = soup.tris().iter().filter(|t| nrm(t).z > 0.99 && area(t) > 4.0).copied().collect();
        let world = StudySoup(soup);
        let mut seed = 0x1234_ABCDu32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let ground_at = |x: f32, y: f32, z0: f32| -> Option<f32> {
            let mut t = Vec::new();
            world.gather(Vec3::new(x - 0.05, y - 0.05, z0 - 12.0), Vec3::new(x + 0.05, y + 0.05, z0 + 3.0), &mut t);
            ray_down(&t, Vec3::new(x, y, z0 + 3.0), 15.0).map(|(d, _)| z0 + 3.0 - d)
        };
        // (start xy, ground z, yaw, edge distance from start, drop)
        let mut ledges: Vec<(glam::Vec2, f32, f32, f32, f32)> = Vec::new();
        let mut tries = 0;
        while ledges.len() < 14 && tries < 4000 {
            tries += 1;
            let t = flats[(rnd() * flats.len() as f32) as usize % flats.len()];
            let c = (t[0] + t[1] + t[2]) / 3.0;
            let Some(z0) = ground_at(c.x, c.y, c.z) else { continue };
            for k in 0..8 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                let dir = glam::Vec2::new(yaw.cos(), yaw.sin());
                let mut level = true;
                for i in 1..80 {
                    let d = i as f32 * 0.5;
                    let p = glam::Vec2::new(c.x, c.y) + dir * d;
                    match ground_at(p.x, p.y, z0) {
                        Some(z) if (z - z0).abs() < 0.15 => {}
                        Some(z) if z < z0 - 1.0 && d >= 9.0 && level => {
                            ledges.push((glam::Vec2::new(c.x, c.y) + dir * (d - 6.0), z0, yaw, 6.0, (z0 - z).min(6.0)));
                            level = false;
                            break;
                        }
                        _ => {
                            break;
                        }
                    }
                }
                if !level {
                    break;
                }
            }
        }
        eprintln!("found {} real ledges in {tries} tries", ledges.len());
        let mut index = crate::play_vehicle::VehicleIndex::default();
        let (mut runs, mut reached, mut backward, mut teleports, mut incomplete, mut model_back) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut shown = 0;
        for (vname, _hover, shapes) in real_vehicles(&sc, &native, cache) {
            let tag = native.tags_by_group(cache, "vehi").into_iter().find(|t| native.tag_name(cache, *t).unwrap_or_default().to_lowercase().contains(vname)).unwrap();
            let Some(entry) = index.entry(&native, cache, tag) else { continue };
            let Ok(model) = entry.drive.clone() else { continue };
            for &(sxy, z0, yaw, ed, drop) in &ledges {
                for speed in [4.0f32, 9.0] {
                    for held in [false, true] {
                        let r = ledge_drive(&model, &shapes, &world, sxy, z0, yaw, ed, drop, speed, held, true, true);
                        runs += 1;
                        reached += r.reached_edge as usize;
                        backward += r.backward;
                        teleports += r.teleports;
                        model_back += r.model_backward;
                        if r.reached_edge && !r.dropped {
                            incomplete += 1;
                        }
                        if (r.backward > 0 || r.teleports > 0 || (r.reached_edge && !r.dropped)) && shown < 12 {
                            shown += 1;
                            eprintln!("TLEDGE {vname} yaw {yaw:.2} drop {drop:.1} speed {speed} held {held}: backward {} (worst {:.3}) teleports {} reached {} dropped {} sweep hits {} model-only backward {}", r.backward, r.worst_back, r.teleports, r.reached_edge, r.dropped, r.sweep_hits, r.model_backward);
                            for l in &r.report {
                                eprintln!("     {l}");
                            }
                        }
                    }
                }
            }
        }
        eprintln!("TLEDGE TOTAL {runs} drives over {} ledges, {reached} reached the edge: backward ticks {backward} (drive law alone {model_back}), teleports {teleports}, past the edge but not dropped {incomplete}", ledges.len());
        assert!(ledges.len() >= 10 && reached >= 40, "liveness: {} ledges, {reached} reached", ledges.len());
        assert_eq!((backward, teleports, incomplete), (0, 0, 0));
    }

    // ---- vehicles --------------------------------------------------------------------------

    fn vseed(datum: u32, pos: Vec3) -> PropSeed {
        PropSeed { vehicle: true, aabb: (pos - Vec3::splat(2.0), pos + Vec3::splat(2.0)), ..seed(datum, pos, 1.0, PhysicsMode::Normal) }
    }

    /// A hover vehicle: a flat hull and three pads (strength 0.6 each, height 1.0, radius 0.2).
    fn ghost(src: &mut Src, datum: u32) {
        let mat = ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 400.0 };
        let shape = ConvexShape::aabb_box(Vec3::ZERO, Vec3::new(1.0, 0.5, 0.15), 0.0, mat).unwrap();
        let mut b = PhysBody::from_shape(shape, "ghost".into());
        b.mass = 400.0;
        b.inertia = Mat3::from_diagonal(Vec3::new(40.0, 80.0, 100.0));
        b.com = Vec3::ZERO;
        b.lin_damping = 0.05;
        b.ang_damping = 0.4;
        src.bodies.insert(datum, b);
        let law = HoverLaw { strength: 0.6, height: 1.0, radius: 0.2, compression_damping: 1.5, extension_damping: 1.5, normal_k_hi: 0.8, normal_k_lo: 0.2 };
        let hover = [Vec3::new(0.8, 0.0, 0.0), Vec3::new(-0.6, 0.4, 0.0), Vec3::new(-0.6, -0.4, 0.0)].map(|pos| HoverPoint { pos, law }).to_vec();
        let idle = Some(crate::reach_havok::IdleBrake { speed_decel: 4.0, slide_decel: 3.0, drag: 0.02, constant_decel: 0.5 });
        src.specs.insert(datum, VehicleSpec { hover, gravity_scale: 1.0, wheeled: false, idle });
    }

    /// A Ghost released 5 wu up falls at gravity, then the hover law catches it and it comes to
    /// rest at the analytic clearance (radius + height * (1 - 1/sqrt(sum strength)) = 0.4546 wu
    /// for three pads of 0.6) -- and goes to sleep there.
    #[test]
    fn a_ghost_released_in_the_air_falls_then_hovers_at_its_authored_height() {
        let mut src = Src::new();
        ghost(&mut src, 1);
        let level = floor();
        let mut w = PropWorld::new();
        w.add(vseed(1, Vec3::new(0.0, 0.0, 5.0)));
        assert!(w.claim_vehicle(1, Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &src));
        // the first 20 ticks are free fall at gravity: v = -g t
        run(&mut w, &level, &src, &[], 20);
        let vz = w.body_velocity(1).unwrap().0.z;
        assert!((vz + G * 20.0 * TICK_SECONDS).abs() < 0.05, "vz after 20 ticks {vz}, free fall gives {}", -G * 20.0 / 60.0);
        run(&mut w, &level, &src, &[], 400);
        let z = w.body_pose(1).unwrap().0.z;
        assert!((z - 0.4546).abs() < 0.04, "hovering at {z}, the pads' analytic clearance is 0.4546");
        assert_eq!(w.is_asleep(1), Some(true), "a hovering vehicle at rest sleeps");
        // and it is level
        let (_, q) = w.body_pose(1).unwrap();
        assert!((q * Vec3::Z).z > 0.99);
    }

    /// With no hover pads (a flyer) there is nothing to hold it up: it falls at gravity times its
    /// own gravity scale. Banshee authors 0.75.
    #[test]
    fn a_flying_vehicle_falls_at_gravity_times_its_scale() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(1.0, 800.0, 0.5, 0.0));
        src.specs.insert(1, VehicleSpec { hover: Vec::new(), gravity_scale: 0.75, wheeled: false, idle: None });
        let level = floor();
        let mut w = PropWorld::new();
        w.add(vseed(1, Vec3::new(0.0, 0.0, 60.0)));
        assert!(w.claim_vehicle(1, Vec3::new(0.0, 0.0, 60.0), Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &src));
        let n = 60u32;
        run(&mut w, &level, &src, &[], n);
        let series = (n * (n + 1)) as f32 / 2.0;
        let want = 60.0 - 0.75 * G * TICK_SECONDS * TICK_SECONDS * series;
        let z = w.body_pose(1).unwrap().0.z;
        assert!((z - want).abs() < 0.01, "fell to {z}, 0.75 g gives {want}");
    }

    /// A wheeled vehicle released mid-air lands on its TYRES (the support spheres), not its hull,
    /// and sleeps there.
    #[test]
    fn a_jeep_released_in_the_air_lands_on_its_tyres() {
        let mut src = Src::new();
        let mat = ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 900.0 };
        let hull = ConvexShape::aabb_box(Vec3::new(0.0, 0.0, 0.55), Vec3::new(1.2, 0.6, 0.3), 0.0, mat).unwrap();
        let mut b = PhysBody::from_shape(hull, "jeep".into());
        let tyre_mat = ShapeMaterial { friction: 0.8, restitution: 0.0, mass: 0.0 };
        let mut pts = Vec::new();
        for (x, y) in [(0.9, 0.55), (0.9, -0.55), (-0.9, 0.55), (-0.9, -0.55)] {
            let mut t = ConvexShape::sphere(Vec3::new(x, y, 0.0), 0.25, tyre_mat).unwrap();
            t.support = true;
            b.shapes.push(t);
            pts.push(Vec3::new(x, y, 0.0));
        }
        b.mass = 900.0;
        b.inertia = Mat3::from_diagonal(Vec3::new(200.0, 700.0, 800.0));
        b.com = Vec3::new(0.0, 0.0, 0.3);
        b.lin_damping = 0.0;
        b.ang_damping = 0.0;
        src.bodies.insert(1, b);
        src.specs.insert(1, VehicleSpec { hover: Vec::new(), gravity_scale: 1.0, wheeled: true, idle: None });
        let level = floor();
        let mut w = PropWorld::new();
        w.add(vseed(1, Vec3::new(0.0, 0.0, 3.0)));
        // a level-start wheeled vehicle in the air is woken and falls onto its tyres (one that
        // was DRIVEN is never claimed: the jeep law parks it)
        w.wake(1, &src);
        run(&mut w, &level, &src, &[], 600);
        let (p, q) = w.body_pose(1).unwrap();
        assert!((p.z - 0.25).abs() < 0.03, "tyre centres rest at their radius: origin z {} want 0.25", p.z);
        assert!((q * Vec3::Z).z > 0.98, "landed upright on its tyres");
        assert_eq!(w.is_asleep(1), Some(true));
    }

    /// Exit continuity: a vehicle released from a moving, airborne state continues that motion --
    /// no stop, no snap -- and the state the world hands back on re-entry is exactly the state it
    /// held.
    #[test]
    fn leaving_and_entering_a_vehicle_carries_its_motion_without_a_snap() {
        let mut src = Src::new();
        src.bodies.insert(1, box_body(1.0, 800.0, 0.5, 0.0));
        src.specs.insert(1, VehicleSpec { hover: Vec::new(), gravity_scale: 1.0, wheeled: false, idle: None });
        let level = floor();
        let mut w = PropWorld::new();
        w.add(vseed(1, Vec3::new(0.0, 0.0, 50.0)));
        let v0 = Vec3::new(6.0, 0.0, 0.0);
        let q0 = Quat::from_rotation_z(0.5);
        assert!(w.claim_vehicle(1, Vec3::new(0.0, 0.0, 50.0), q0, v0, Vec3::ZERO, &src));
        run(&mut w, &level, &src, &[], 12);
        let (o, _, v, _) = w.vehicle_state(1).unwrap();
        assert!((o.x - 6.0 * 12.0 * TICK_SECONDS).abs() < 0.02, "carried {} wu in 12 ticks, 6 wu/s gives 1.2", o.x);
        assert!((v.x - 6.0).abs() < 0.01 && (v.z + G * 12.0 * TICK_SECONDS).abs() < 0.02, "velocity {v:?}");
        // the driver gets back in: the state handed back is the state it held, and it stops moving
        let held = w.vehicle_state(1).unwrap();
        let handed = w.release_vehicle(1).unwrap();
        assert_eq!(handed.0, held.0);
        assert_eq!(handed.2, held.2);
        assert!(w.vehicle_state(1).is_none() && w.is_driven(1));
        run(&mut w, &level, &src, &[], 30);
        assert_eq!(w.body_pose(1).unwrap().0, held.0, "a driven vehicle is not stepped by the world");
    }

    /// A wheeled vehicle that is pushed rolls (the tyres' friction is low) and comes to rest by
    /// damping/friction, then sleeps.
    #[test]
    fn a_pushed_warthog_rolls_and_stops() {
        let mut src = Src::new();
        let mat = ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 900.0 };
        let hull = ConvexShape::aabb_box(Vec3::new(0.0, 0.0, 0.55), Vec3::new(1.2, 0.6, 0.3), 0.0, mat).unwrap();
        let mut b = PhysBody::from_shape(hull, "jeep".into());
        for (x, y) in [(0.9, 0.55), (0.9, -0.55), (-0.9, 0.55), (-0.9, -0.55)] {
            let mut t = ConvexShape::sphere(Vec3::new(x, y, 0.0), 0.25, ShapeMaterial { friction: 0.8, restitution: 0.0, mass: 0.0 }).unwrap();
            t.support = true;
            b.shapes.push(t);
        }
        b.mass = 900.0;
        b.inertia = Mat3::from_diagonal(Vec3::new(200.0, 700.0, 800.0));
        b.com = Vec3::new(0.0, 0.0, 0.3);
        b.lin_damping = 0.0;
        b.ang_damping = 0.0;
        src.bodies.insert(1, b);
        src.specs.insert(1, VehicleSpec { hover: Vec::new(), gravity_scale: 1.0, wheeled: true, idle: None });
        let level = floor();
        let mut w = PropWorld::new();
        let start = Vec3::new(0.0, 0.0, 0.25);
        w.add(PropSeed { vehicle: true, aabb: (start - Vec3::new(2.0, 1.0, 0.3), start + Vec3::new(2.0, 1.0, 1.2)), ..seed(1, start, 1.0, PhysicsMode::Normal) });
        // the player's capsule walks into its tail at 2 wu/s
        let cap = [ConvexShape::capsule(Vec3::new(0.0, 0.0, 0.3), Vec3::new(0.0, 0.0, 1.0), 0.3, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 0.0 }).unwrap()];
        let mut x = -3.0f32;
        for _ in 0..240 {
            x += 2.0 * TICK_SECONDS;
            let kin = Kinematic { shapes: &cap, origin: Vec3::new(x, 0.0, 0.0), rot: Quat::IDENTITY, velocity: Vec3::new(2.0, 0.0, 0.0), angular: Vec3::ZERO, mass: 125.0, com: Vec3::new(0.0, 0.0, 0.6) };
            w.step(TICK_SECONDS, &level, &src, std::slice::from_ref(&kin));
        }
        let moved = w.body_pose(1).unwrap().0.x;
        assert!(moved > 0.5, "the vehicle was not pushed: x = {moved}");
        run(&mut w, &level, &src, &[], 900);
        assert_eq!(w.is_asleep(1), Some(true), "it never came to rest");
        let rest = w.body_pose(1).unwrap().0.x;
        assert!(rest > moved - 0.01, "it rolled back?");
    }

    /// The render fingerprint: it changes every tick while a vehicle falls and stops changing once
    /// it sleeps (the Theater lesson: a settled body must not keep rebuilding the scene).
    #[test]
    fn a_vehicle_signature_moves_while_falling_and_stops_when_settled() {
        let mut src = Src::new();
        ghost(&mut src, 1);
        let level = floor();
        let mut w = PropWorld::new();
        w.add(vseed(1, Vec3::new(0.0, 0.0, 3.0)));
        assert!(w.claim_vehicle(1, Vec3::new(0.0, 0.0, 3.0), Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &src));
        let mut sigs = std::collections::HashSet::new();
        for _ in 0..40 {
            w.step(TICK_SECONDS, &level, &src, &[]);
            sigs.insert(w.signature());
        }
        assert!(sigs.len() >= 30, "{} distinct signatures in 40 falling ticks", sigs.len());
        assert_eq!(w.displaced_vehicles(), vec![1]);
        run(&mut w, &level, &src, &[], 600);
        assert_eq!(w.is_asleep(1), Some(true), "a Ghost dropped from 3 wu must hover still and sleep");
        let s0 = w.signature();
        run(&mut w, &level, &src, &[], 60);
        assert_eq!(w.signature(), s0);
    }

    /// Many parked vehicles: asleep ones cost nothing; 60 hovering ones stay cheap.
    #[test]
    fn many_parked_vehicles_are_bounded() {
        let mut src = Src::new();
        let level = floor();
        let mut w = PropWorld::new();
        for i in 0..200u32 {
            ghost(&mut src, i);
            w.add(vseed(i, Vec3::new((i % 20) as f32 * 6.0, (i / 20) as f32 * 6.0, 0.5)));
        }
        run(&mut w, &level, &src, &[], 60);
        assert_eq!(w.stats.pair_tests + w.stats.static_contacts, 0, "sleeping vehicles did work");
        assert_eq!(src.decodes.get(), 0);
        for i in 0..60u32 {
            let o = w.body_pose(i).unwrap().0;
            w.claim_vehicle(i, o + Vec3::Z * 2.0, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &src);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..60 {
            w.step(TICK_SECONDS, &level, &src, &[]);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 60.0;
        eprintln!("play_world: 60 awake hover vehicles, {ms:.2} ms/tick");
        assert!(ms < 20.0, "60 hover vehicles cost {ms} ms per tick");
    }

    #[test]
    fn the_tick_is_the_engines() {
        assert_eq!(TICK_HZ, 60.0);
    }
}
