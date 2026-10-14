//! The articulated rigid-body solver: ONE physics core for the editor's "Simulate physics for
//! selection" and for Theater's dead bipeds. #ragdoll
//!
//! [`physics_settle`](crate::physics_settle) already owned the collision geometry (a `phmo`'s
//! rigid-body shapes, the Havok convex radius, the tag mass / centre of mass / inertia / friction
//! / restitution), the static-world query and a sequential-impulse contact solver -- but it welded
//! every rigid body of an object into ONE body, so a ragdoll's limbs could not articulate. This
//! module is that solver generalised to N bodies joined by the `phmo`'s own authored constraints,
//! and `physics_settle::settle` is now a one-link, no-joint call into it: the editor's settle and
//! a Theater ragdoll run the same `step`, the same contact rows and the same integrator.
//!
//! ## What a ragdoll is made of
//!
//! A `phmo` rigid body names a `mode` NODE (`phmo_shapes::rigid_bodies`), and the constraint
//! blocks name pairs of the same nodes with a pivot and a basis in each node's own space
//! (`phmo_shapes::constraints`). So the simulation is:
//!   * one [`Link`] per rigid body, its shapes in NODE space (built with an empty node-matrix
//!     list so `rigid_bodies` leaves them there), its mass / centre of mass / inertia from the
//!     tag through `PhysBody::from_rigid_bodies` of that ONE body;
//!   * one [`Joint`] per authored constraint, mapped onto the two links by node index.
//! The links start at the biped's current pose (`RagdollState::from_pose`), so a ragdoll begins
//! exactly where the death animation's first frame put the body, and the drawn skeleton is read
//! back out of the simulation with [`Ragdoll::pose_model`].
//!
//! ## The solver
//!
//! Sequential impulses (Havok's own solver is a projected-Gauss-Seidel / sequential-impulse
//! solver), at the engine's 30 Hz tick with speed-driven substepping, in this order per iteration:
//!   1. every joint's point-to-point row (a full 3x3 effective-mass solve, Baumgarte-biased);
//!   2. every joint's angular LIMIT rows -- twist / cone / plane for a ragdoll constraint, the
//!      axis lock and the angle limit for a hinge -- each a scalar angular row whose Jacobian
//!      axis is derived exactly (see [`Joint::limit_rows`]);
//!   3. every contact row: normal impulse with restitution, two-direction Coulomb friction
//!      clamped by the accumulated normal impulse, and a split impulse for penetration recovery.
//! Contacts come from `physics_settle::shape_contacts` unchanged, so a ragdoll collides with the
//! level and with placed objects through the same vertex-vs-triangle code the editor settle uses.
//!
//! ## What this is not
//!
//! It is not a Havok port. Specific simplifications, stated so nobody mistakes them:
//!   * a ragdoll constraint's cone is circular (the tag's `cone` range is read as its half-angle)
//!     and the twist angle comes from a swing-twist decomposition, so the twist row's Jacobian is
//!     exact only while the swing is small;
//!   * the authored MOTORS (`damped spring motors` / `position motors` / the per-edge
//!     `ragdoll motors` of the `node edges` block) are not simulated -- a Reach ragdoll is limp,
//!     and the only joint damping is the constraint's own friction torque;
//!   * self-collision is sphere-vs-sphere between links that are neither joined nor already
//!     touching at the BIND pose (a biped's pills overlap at every joint, so the overlapping
//!     pairs are measured once at build time and excluded forever);
//!   * link-vs-link contacts have no friction and no restitution.

use glam::{Mat3, Mat4, Quat, Vec3};

use crate::phmo_shapes::{ConstraintDef, ConstraintKind};
use crate::physics_settle::{
    integrate_quat, shape_contacts, Contact, ConvexShape, PhysBody, RigidBodyDef, ShapeGeom,
    StaticWorld, GRAVITY_WU_S2, SUPPORT_G, TICK_HZ,
};

/// One simulated body: a `phmo` rigid body's shapes in NODE space plus the node it drives.
#[derive(Clone, Debug)]
pub struct Link {
    pub body: PhysBody,
    /// The `mode` node this link is the physics of (-1 = none).
    pub node: i16,
    /// Keyframed (motion type 4) or fixed (5): never moved by the solver.
    ///
    /// NOT the infinite-inertia flag -- that is `no_spin`. A Reach `phmo` marks a rigid body
    /// keyframed because in-game it is driven by the animation until the biped dies, so a ragdoll
    /// whose bodies are ALL keyframed is un-pinned wholesale by [`build`]: a ragdoll is by
    /// definition the dynamic version of the same bodies.
    pub pinned: bool,
    /// `Infinite Inertia Tensor` (rigid body flags bit 1): the body never gains angular velocity.
    /// It still moves -- treating it as pinned would nail a limb to the air.
    pub no_spin: bool,
    /// Sphere approximation of the link's shapes, NODE space, for self-collision.
    spheres: Vec<(Vec3, f32)>,
}

impl Link {
    pub fn inv_mass(&self) -> f32 { if self.pinned { 0.0 } else { 1.0 / self.body.mass.max(1e-6) } }
    pub fn inv_inertia_body(&self) -> Mat3 {
        if self.pinned || self.no_spin { return Mat3::ZERO; }
        let i = self.body.inertia.inverse();
        if i.is_finite() { i } else { Mat3::from_diagonal(Vec3::splat(1.0 / self.body.mass.max(1e-6))) }
    }
}

/// What a joint allows, from the `phmo` constraint it was read from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointLimits {
    /// Point-to-point only (ball and socket / stiff spring, which is treated as a short rod).
    Ball,
    /// The child's twist axis is locked onto the parent's, with an optional angle limit about it.
    Hinge { limit: Option<(f32, f32)>, friction: f32 },
    /// Havok's ragdoll constraint: a twist about the frame's forward axis, a circular cone the
    /// forward axis may leave the parent's by, and a plane limit on the swing out of the frame's
    /// forward/up plane.
    Ragdoll { twist: (f32, f32), cone: f32, plane: (f32, f32), friction_torque: f32 },
}

/// One authored joint between two links.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub a: usize,
    pub b: usize,
    /// The pivot in each link's NODE space.
    pub pivot_a: Vec3,
    pub pivot_b: Vec3,
    /// The constraint basis in each link's NODE space, columns (forward = twist, left = plane
    /// axis, up).
    pub frame_a: Mat3,
    pub frame_b: Mat3,
    pub limits: JointLimits,
    /// The constraint's `name` string id, for the diagnostics.
    pub name: u32,
}

/// A built ragdoll: the links, the joints and the self-collision pairs.
#[derive(Clone, Debug)]
pub struct Ragdoll {
    pub links: Vec<Link>,
    pub joints: Vec<Joint>,
    /// Link pairs tested for self-collision (neither joined nor overlapping at the bind pose).
    pub self_pairs: Vec<(usize, usize)>,
    /// Status-line description.
    pub desc: String,
    /// Constraint elements present in the tag that could not be mapped onto two links.
    pub unmapped_constraints: usize,
}

/// One link's state. `com` is the centre of mass in WORLD space, `q` the node->world rotation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkState {
    pub com: Vec3,
    pub q: Quat,
    pub v: Vec3,
    pub w: Vec3,
}

impl Default for LinkState {
    fn default() -> Self { LinkState { com: Vec3::ZERO, q: Quat::IDENTITY, v: Vec3::ZERO, w: Vec3::ZERO } }
}

/// The whole simulation's state at one tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RagdollState {
    pub links: Vec<LinkState>,
    /// Ticks simulated since the start.
    pub ticks: u32,
    /// Consecutive quiet ticks (see [`SimParams::sleep_steps`]).
    pub quiet: u32,
    /// Any link has touched something.
    pub touched: bool,
    /// The simulation has come to rest and is no longer stepped.
    pub asleep: bool,
}

/// Simulation parameters. The defaults are the engine's tick and gravity.
#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    pub dt: f32,
    pub gravity: f32,
    pub solver_iters: usize,
    /// Penetration allowed before the split impulse pushes out (wu).
    pub slop: f32,
    /// Approach speed below which a contact does not bounce (wu/s).
    pub restitution_threshold: f32,
    /// Baumgarte factor for a joint's point-to-point error (fraction corrected per substep).
    pub joint_bias: f32,
    /// Baumgarte factor for an angular limit's violation.
    pub limit_bias: f32,
    /// Largest substeps per tick (the existing settle's cap).
    pub max_substeps: usize,
    pub self_collision: bool,
    /// Rest = below these speeds for `sleep_steps` consecutive ticks while touching something.
    pub sleep_lin: f32,
    pub sleep_ang: f32,
    pub sleep_steps: u32,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            dt: 1.0 / TICK_HZ,
            gravity: GRAVITY_WU_S2,
            solver_iters: 16,
            slop: 0.004,
            restitution_threshold: 0.5,
            joint_bias: 0.2,
            limit_bias: 0.2,
            max_substeps: 8,
            self_collision: true,
            sleep_lin: 0.02,
            sleep_ang: 0.05,
            sleep_steps: 15,
        }
    }
}

/// What one tick did, for the status line and the gates.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub contacts: usize,
    pub tris_tested: usize,
    pub substeps: usize,
    /// Largest point-to-point error left at any joint (wu).
    pub joint_error: f32,
    /// Largest angular-limit violation left at any joint (radians).
    pub limit_violation: f32,
}

// ----------------------------------------------------------------------------------------------
// Building a ragdoll from a phmo
// ----------------------------------------------------------------------------------------------

/// Sphere approximation of one shape in its own space (a capsule becomes three spheres, a hull
/// its bounding sphere): what self-collision tests.
fn shape_spheres(s: &ConvexShape) -> Vec<(Vec3, f32)> {
    match &s.geom {
        ShapeGeom::Sphere { c, r } => vec![(*c, *r)],
        ShapeGeom::Capsule { a, b, r } => (0..3).map(|k| (a.lerp(*b, k as f32 / 2.0), *r)).collect(),
        ShapeGeom::Hull { pts, r, .. } => {
            let c = pts.iter().copied().sum::<Vec3>() / pts.len().max(1) as f32;
            let rad = pts.iter().map(|p| (*p - c).length()).fold(0.0f32, f32::max) + *r;
            vec![(c, rad)]
        }
    }
}

/// Build a ragdoll from one `phmo`'s rigid bodies and constraints.
///
/// `defs` must be the rigid bodies in NODE space -- call
/// `phmo_shapes::rigid_bodies(&blocks, layout, &[])` with an EMPTY node-matrix list, because the
/// links carry their node frame in their own state, not baked into the geometry. `bind` is the
/// model-space bind matrix of every `mode` node (`reach_skel::Skeleton::bind_model`), used only
/// to measure which link pairs already overlap and must never self-collide.
///
/// `None` when fewer than two links have a shape (there is nothing to articulate).
pub fn build(defs: &[RigidBodyDef], constraints: &[ConstraintDef], bind: &[Mat4], desc: String) -> Option<Ragdoll> {
    let mut links: Vec<Link> = Vec::new();
    let mut duplicate_nodes = 0usize;
    for d in defs {
        if d.shapes.is_empty() { continue; }
        // ONE link per node. A `phmo` carries a rigid body per REGION/PERMUTATION (a rigid body
        // element has both indices), so a biped can author several bodies on the same node -- and
        // a second body on a node the joints already name would be an unattached lump flying off
        // on its own, because each constraint names a NODE and so can only reach one of them.
        // The first is kept, which is the base permutation; the permutation actually drawn is NOT
        // selected (`regions` / `permutations` are not read), and the count is reported.
        if links.iter().any(|l| l.node == d.node) {
            duplicate_nodes += 1;
            continue;
        }
        let one = [d.clone()];
        let Some(body) = PhysBody::from_rigid_bodies(&one, String::new()) else { continue };
        let spheres = body.shapes.iter().flat_map(shape_spheres).collect();
        links.push(Link {
            body,
            node: d.node,
            pinned: d.motion_type >= 4,
            no_spin: d.flags & 0x2 != 0,
            spheres,
        });
    }
    if links.len() < 2 { return None; }
    // A `phmo` whose every rigid body is keyframed is the ANIMATED version of the model (which is
    // what a biped is until it dies). Simulating that is simulating nothing, so the whole
    // assembly goes dynamic -- the engine does the same thing when it hands a unit to the ragdoll.
    if links.iter().all(|l| l.pinned) {
        for l in links.iter_mut() { l.pinned = false; }
    }

    let link_of = |node: i16| -> Option<usize> { links.iter().position(|l| l.node == node && node >= 0) };
    let mut joints: Vec<Joint> = Vec::new();
    let mut unmapped = 0usize;
    for c in constraints {
        let (Some(a), Some(b)) = (link_of(c.node_a), link_of(c.node_b)) else {
            unmapped += 1;
            continue;
        };
        if a == b { unmapped += 1; continue; }
        let limits = match c.kind {
            ConstraintKind::Ragdoll { twist, cone, plane, friction_torque } => JointLimits::Ragdoll {
                twist: order(twist),
                // the tag stores a range; the cone is circular, so its half-angle is the widest
                // end of that range
                cone: twist_sane(cone.0.abs().max(cone.1.abs())),
                plane: order(plane),
                friction_torque: friction_torque.max(0.0),
            },
            ConstraintKind::LimitedHinge { friction, limit } => JointLimits::Hinge { limit: Some(order(limit)), friction: friction.max(0.0) },
            ConstraintKind::Hinge => JointLimits::Hinge { limit: None, friction: 0.0 },
            ConstraintKind::BallSocket | ConstraintKind::StiffSpring { .. } => JointLimits::Ball,
            // a prismatic joint is a slider, which no biped authors; hold it as a ball so it does
            // not silently come apart
            ConstraintKind::Prismatic { .. } => JointLimits::Ball,
        };
        joints.push(Joint {
            a,
            b,
            pivot_a: c.pivot_a,
            pivot_b: c.pivot_b,
            frame_a: c.frame_a,
            frame_b: c.frame_b,
            limits,
            name: c.name,
        });
    }

    // A pile of unconnected rigid bodies is not a ragdoll: without the authored constraints the
    // limbs would simply fly apart, which is worse than the clip it replaces. Refusing here is
    // what makes the caller's fallback honest ("this biped has no articulated physics model").
    if joints.is_empty() { return None; }

    // Self-collision pairs: not joined, and not already overlapping at the bind pose.
    let mut self_pairs = Vec::new();
    for i in 0..links.len() {
        for j in (i + 1)..links.len() {
            if joints.iter().any(|t| (t.a == i && t.b == j) || (t.a == j && t.b == i)) { continue; }
            if links[i].pinned && links[j].pinned { continue; }
            if bind_overlap(&links[i], &links[j], bind) { continue; }
            self_pairs.push((i, j));
        }
    }

    let nj = joints.len();
    let desc = format!(
        "{desc}: {} link{}, {nj} joint{}{}{}",
        links.len(),
        if links.len() == 1 { "" } else { "s" },
        if nj == 1 { "" } else { "s" },
        if unmapped > 0 { format!(", {unmapped} constraint(s) with no rigid body") } else { String::new() },
        if duplicate_nodes > 0 {
            format!(", {duplicate_nodes} further rigid body(ies) on a node already taken (other region/permutation)")
        } else {
            String::new()
        }
    );
    Some(Ragdoll { links, joints, self_pairs, desc, unmapped_constraints: unmapped })
}

fn order((a, b): (f32, f32)) -> (f32, f32) {
    if a.is_finite() && b.is_finite() && a <= b { (a, b) } else if a.is_finite() && b.is_finite() { (b, a) } else { (0.0, 0.0) }
}

fn twist_sane(a: f32) -> f32 { if a.is_finite() { a.clamp(0.0, std::f32::consts::PI) } else { 0.0 } }

/// Do two links' sphere approximations already touch at the bind pose?
fn bind_overlap(a: &Link, b: &Link, bind: &[Mat4]) -> bool {
    let ma = bind_of(bind, a.node);
    let mb = bind_of(bind, b.node);
    for (ca, ra) in &a.spheres {
        let pa = ma.transform_point3(*ca);
        for (cb, rb) in &b.spheres {
            if (pa - mb.transform_point3(*cb)).length() < ra + rb { return true; }
        }
    }
    false
}

fn bind_of(bind: &[Mat4], node: i16) -> Mat4 {
    if node < 0 { return Mat4::IDENTITY; }
    bind.get(node as usize).copied().unwrap_or(Mat4::IDENTITY)
}

impl Ragdoll {
    /// The node->world matrix a link's state puts its node at.
    pub fn link_world(&self, i: usize, st: &RagdollState) -> Mat4 {
        let s = st.links[i];
        Mat4::from_rotation_translation(s.q, s.com - s.q * self.links[i].body.com)
    }

    /// Place every link at the node frames of `node_world` (world-space matrices for every `mode`
    /// node: `actor.world() * actor.pose.model[i]`), at rest.
    pub fn state_from_pose(&self, node_world: &[Mat4]) -> RagdollState {
        let links = self
            .links
            .iter()
            .map(|l| {
                let m = bind_of(node_world, l.node);
                let (_, q, t) = m.to_scale_rotation_translation();
                let q = if q.is_finite() { q.normalize() } else { Quat::IDENTITY };
                LinkState { com: t + q * l.body.com, q, v: Vec3::ZERO, w: Vec3::ZERO }
            })
            .collect();
        RagdollState { links, ticks: 0, quiet: 0, touched: false, asleep: false }
    }

    /// Read the simulation back out as MODEL-space node matrices for a skeleton of `n` nodes.
    ///
    /// A node a link drives takes that link's frame directly (converted out of world by
    /// `root^-1`). Every other node keeps its pose-relative transform to the nearest ancestor
    /// that IS driven, so a Spartan's fingers and his gun hand ride the forearm link instead of
    /// collapsing to the origin. `parents` / `locals` are the skeleton's parent indices and the
    /// per-node local matrices of the pose being replaced.
    pub fn pose_model(&self, st: &RagdollState, root_inv: Mat4, parents: &[i16], locals: &[Mat4], order: &[usize]) -> Vec<Mat4> {
        let n = parents.len().min(locals.len());
        let mut out = vec![Mat4::IDENTITY; n];
        let mut driven = vec![false; n];
        for (i, l) in self.links.iter().enumerate() {
            if l.node < 0 { continue; }
            let k = l.node as usize;
            if k < n {
                out[k] = root_inv * self.link_world(i, st);
                driven[k] = true;
            }
        }
        for &k in order {
            if k >= n || driven[k] { continue; }
            let p = parents[k];
            out[k] = if p >= 0 && (p as usize) < n { out[p as usize] * locals[k] } else { locals[k] };
        }
        out
    }

    /// The whole ragdoll's world AABB at `st`.
    pub fn aabb(&self, st: &RagdollState) -> (Vec3, Vec3) {
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for i in 0..self.links.len() {
            let m = self.link_world(i, st);
            let (lmn, lmx) = self.links[i].body.aabb();
            for k in 0..8u32 {
                let c = Vec3::new(
                    if k & 4 != 0 { lmx.x } else { lmn.x },
                    if k & 2 != 0 { lmx.y } else { lmn.y },
                    if k & 1 != 0 { lmx.z } else { lmn.z },
                );
                let cw = m.transform_point3(c);
                mn = mn.min(cw);
                mx = mx.max(cw);
            }
        }
        (mn, mx)
    }

    pub fn total_mass(&self) -> f32 { self.links.iter().map(|l| l.body.mass).sum() }
}

/// #settle ONE link and no joints: the shape the editor's "Simulate physics for selection" runs
/// through, so the editor and a Theater ragdoll share this module's `step` rather than two
/// solvers. Self-collision is meaningless with one link and is off for it.
pub fn one_link(body: PhysBody) -> Ragdoll {
    let spheres = body.shapes.iter().flat_map(shape_spheres).collect();
    Ragdoll {
        links: vec![Link { body, node: -1, pinned: false, no_spin: false, spheres }],
        joints: Vec::new(),
        self_pairs: Vec::new(),
        desc: String::new(),
        unmapped_constraints: 0,
    }
}

// ----------------------------------------------------------------------------------------------
// Joint rows
// ----------------------------------------------------------------------------------------------

/// One scalar angular inequality: `axis` is the Jacobian direction (relative angular velocity
/// about it increases `value`), and `value` must stay in [`lo`, `hi`].
#[derive(Clone, Copy, Debug)]
pub struct LimitRow {
    pub axis: Vec3,
    pub value: f32,
    pub lo: f32,
    pub hi: f32,
    /// Angular friction impulse cap (0 = none).
    pub friction: f32,
}

impl LimitRow {
    /// How far outside its range the row currently is (0 = inside).
    pub fn violation(&self) -> f32 { (self.lo - self.value).max(self.value - self.hi).max(0.0) }
}

/// Swing/twist split of `q` about `+X`: returns (twist angle about X, swing quaternion).
fn swing_twist_x(q: Quat) -> (f32, Quat) {
    let q = if q.w < 0.0 { -q } else { q };
    // the twist part is the rotation about X that q contains
    let twist = if q.x.abs() < 1e-7 && q.w.abs() < 1e-7 {
        Quat::IDENTITY
    } else {
        Quat::from_xyzw(q.x, 0.0, 0.0, q.w).normalize()
    };
    let twist = if twist.is_finite() { twist } else { Quat::IDENTITY };
    let swing = q * twist.conjugate();
    let angle = 2.0 * twist.x.clamp(-1.0, 1.0).atan2(twist.w);
    let angle = if angle > std::f32::consts::PI {
        angle - std::f32::consts::TAU
    } else if angle < -std::f32::consts::PI {
        angle + std::f32::consts::TAU
    } else {
        angle
    };
    (angle, swing)
}

impl Joint {
    /// The joint's two pivots in world space, given the link frames.
    fn pivots(&self, rd: &Ragdoll, st: &RagdollState) -> (Vec3, Vec3) {
        (
            rd.link_world(self.a, st).transform_point3(self.pivot_a),
            rd.link_world(self.b, st).transform_point3(self.pivot_b),
        )
    }

    /// The angular limit rows this joint imposes at the current state.
    ///
    /// Every row's Jacobian is derived rather than guessed:
    ///   * CONE -- `theta = acos(uA . uB)` between the two twist axes, and
    ///     `d(theta)/dt = w_rel . normalize(uA x uB)` exactly (because `|uA x uB| = sin theta`);
    ///   * PLANE -- `phi = asin(uB . pA)` with `pA` the parent frame's plane axis, and
    ///     `d(phi)/dt = w_rel . normalize(uB x pA)` exactly (because `|uB x pA| = cos phi`);
    ///   * TWIST -- the twist angle of the swing-twist split of `frameA^-1 frameB` about the
    ///     frame's forward axis, whose rate is `w_rel . uA` while the swing is small.
    /// A hinge's two locked angular degrees of freedom are one cone row with a zero limit: the
    /// row's axis is recomputed every iteration, so the pair of directions is covered.
    pub fn limit_rows(&self, rd: &Ragdoll, st: &RagdollState, out: &mut Vec<LimitRow>) {
        let ra = Mat3::from_quat(st.links[self.a].q) * self.frame_a;
        let rb = Mat3::from_quat(st.links[self.b].q) * self.frame_b;
        let (ua, pa) = (ra.x_axis.normalize_or_zero(), ra.y_axis.normalize_or_zero());
        let ub = rb.x_axis.normalize_or_zero();
        if ua == Vec3::ZERO || ub == Vec3::ZERO { return; }
        let (cone_max, twist, plane, friction) = match self.limits {
            JointLimits::Ball => return,
            JointLimits::Hinge { limit, friction } => (Some(0.0), limit, None, friction),
            JointLimits::Ragdoll { twist, cone, plane, friction_torque } => (Some(cone), Some(twist), Some(plane), friction_torque),
        };
        // cone (or, at zero, a hinge's axis lock)
        if let Some(cmax) = cone_max {
            let cross = ua.cross(ub);
            let sin = cross.length();
            let theta = sin.atan2(ua.dot(ub).clamp(-1.0, 1.0));
            let axis = if sin > 1e-5 {
                cross / sin
            } else if theta > 1.0 {
                // antiparallel: any perpendicular closes the cone
                ua.any_orthonormal_vector()
            } else {
                Vec3::ZERO
            };
            if axis != Vec3::ZERO {
                out.push(LimitRow { axis, value: theta, lo: -cmax, hi: cmax, friction: 0.0 });
            }
        }
        if let Some((lo, hi)) = plane {
            let cross = ub.cross(pa);
            let len = cross.length();
            if len > 1e-5 {
                let phi = ub.dot(pa).clamp(-1.0, 1.0).asin();
                out.push(LimitRow { axis: cross / len, value: phi, lo, hi, friction: 0.0 });
            }
        }
        if let Some((lo, hi)) = twist {
            let qrel = Quat::from_mat3(&orthonormalise(ra)).conjugate() * Quat::from_mat3(&orthonormalise(rb));
            if qrel.is_finite() {
                let (psi, _) = swing_twist_x(qrel.normalize());
                out.push(LimitRow { axis: ua, value: psi, lo, hi, friction });
            }
        }
    }
}

fn orthonormalise(m: Mat3) -> Mat3 {
    let x = m.x_axis.normalize_or_zero();
    let x = if x == Vec3::ZERO { Vec3::X } else { x };
    let mut y = m.y_axis - x * m.y_axis.dot(x);
    if y.length_squared() < 1e-10 { y = x.any_orthonormal_vector(); }
    let y = y.normalize();
    Mat3::from_cols(x, y, x.cross(y))
}

// ----------------------------------------------------------------------------------------------
// The step
// ----------------------------------------------------------------------------------------------

struct Row {
    a: usize,
    b: Option<usize>,
    ra: Vec3,
    rb: Vec3,
    n: Vec3,
    t1: Vec3,
    t2: Vec3,
    kn: f32,
    kt1: f32,
    kt2: f32,
    jn: f32,
    jt1: f32,
    jt2: f32,
    jp: f32,
    bounce: f32,
    target: f32,
    mu: f32,
    cap: f32,
}

/// Per-link cached inverses at the current orientation.
struct LinkCache {
    inv_m: f32,
    i_inv: Mat3,
}

#[inline]
fn point_vel(s: &LinkState, r: Vec3) -> Vec3 { s.v + s.w.cross(r) }

#[inline]
fn apply_impulse(st: &mut [LinkState], c: &[LinkCache], i: usize, r: Vec3, imp: Vec3, sign: f32) {
    st[i].v += imp * (c[i].inv_m * sign);
    st[i].w += c[i].i_inv * r.cross(imp * sign);
}

/// Step the whole system one engine tick. Returns what the tick did.
pub fn step(rd: &Ragdoll, st: &mut RagdollState, world: &dyn StaticWorld, p: &SimParams) -> StepStats {
    let mut stats = StepStats::default();
    if st.asleep || rd.links.is_empty() {
        return stats;
    }
    // substeps from the fastest link, the same rule the single-body settle used
    let mut speed = 0.0f32;
    for (i, l) in rd.links.iter().enumerate() {
        let brad = l.body.bounding_radius().max(1e-3);
        speed = speed.max(st.links[i].v.length() + st.links[i].w.length() * brad);
    }
    let nsub = ((speed * p.dt) / 0.05).ceil().clamp(1.0, p.max_substeps.max(1) as f32) as usize;
    let h = p.dt / nsub as f32;
    stats.substeps = nsub;

    let mut tris: Vec<[Vec3; 3]> = Vec::new();
    let mut contacts: Vec<Contact> = Vec::new();

    for _ in 0..nsub {
        // integrate forces
        for (i, l) in rd.links.iter().enumerate() {
            if l.pinned { st.links[i].v = Vec3::ZERO; st.links[i].w = Vec3::ZERO; continue; }
            st.links[i].v.z -= p.gravity * h;
            st.links[i].v /= 1.0 + l.body.lin_damping * h;
            st.links[i].w /= 1.0 + l.body.ang_damping * h;
        }
        let cache: Vec<LinkCache> = rd
            .links
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let r = Mat3::from_quat(st.links[i].q);
                LinkCache { inv_m: l.inv_mass(), i_inv: r * l.inv_inertia_body() * r.transpose() }
            })
            .collect();

        // ---- contacts against the static world -------------------------------------------
        let mut rows: Vec<Row> = Vec::new();
        for (i, l) in rd.links.iter().enumerate() {
            if l.pinned { continue; }
            let s = st.links[i];
            let r = Mat3::from_quat(s.q);
            let rt = r.transpose();
            let x = s.com - r * l.body.com;
            let (lmn, lmx) = l.body.aabb();
            let (mut wmn, mut wmx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for k in 0..8u32 {
                let c = Vec3::new(
                    if k & 4 != 0 { lmx.x } else { lmn.x },
                    if k & 2 != 0 { lmx.y } else { lmn.y },
                    if k & 1 != 0 { lmx.z } else { lmn.z },
                );
                let cw = x + r * c;
                wmn = wmn.min(cw);
                wmx = wmx.max(cw);
            }
            let sweep = s.v * h;
            let margin = Vec3::splat(0.02);
            tris.clear();
            world.gather(wmn.min(wmn + sweep) - margin, wmx.max(wmx + sweep) + margin, &mut tris);
            stats.tris_tested += tris.len();
            contacts.clear();
            for sh in &l.body.shapes {
                shape_contacts(sh, r, x, rt, &tris, &mut contacts);
            }
            if !contacts.is_empty() { st.touched = true; }
            let support_cap = l.body.mass * p.gravity * h * SUPPORT_G;
            for c in &contacts {
                let rr = c.p - s.com;
                push_contact_row(&mut rows, i, None, rr, Vec3::ZERO, c, &cache, st, p, support_cap, h);
            }
        }

        // ---- self-collision (sphere vs sphere, no friction, no restitution) --------------
        if p.self_collision {
            for &(i, j) in &rd.self_pairs {
                let (mi, mj) = (rd.link_world(i, st), rd.link_world(j, st));
                for (ci, ri) in &rd.links[i].spheres {
                    let pi = mi.transform_point3(*ci);
                    for (cj, rj) in &rd.links[j].spheres {
                        let pj = mj.transform_point3(*cj);
                        let d = pi - pj;
                        let len = d.length();
                        let sum = ri + rj;
                        if len >= sum || len < 1e-6 { continue; }
                        let n = d / len;
                        let c = Contact { p: pj + n * *rj, n, depth: sum - len, mu: 0.0, e: 0.0, soft: None };
                        let ra = (pj + n * *rj) - st.links[i].com;
                        let rb = (pj + n * *rj) - st.links[j].com;
                        push_contact_row(&mut rows, i, Some(j), ra, rb, &c, &cache, st, p, f32::INFINITY, h);
                    }
                }
            }
        }
        stats.contacts = rows.len();

        // ---- solve ------------------------------------------------------------------------
        // joint rows are rebuilt from the state each iteration (their axes rotate with the
        // bodies), so only the accumulated contact impulses are warm-started within a substep
        let mut acc_limit: Vec<f32> = Vec::new();
        let mut vp: Vec<Vec3> = vec![Vec3::ZERO; rd.links.len()];
        let mut wp: Vec<Vec3> = vec![Vec3::ZERO; rd.links.len()];
        let mut limit_rows: Vec<LimitRow> = Vec::new();
        for _ in 0..p.solver_iters {
            // 1. point to point
            for j in &rd.joints {
                let (pa, pb) = j.pivots(rd, st);
                let ra = pa - st.links[j.a].com;
                let rb = pb - st.links[j.b].com;
                let (ca, cb) = (&cache[j.a], &cache[j.b]);
                let k = eff_mass_matrix(ca, cb, ra, rb);
                let Some(kinv) = invert3(k) else { continue };
                let err = pb - pa;
                let vrel = point_vel(&st.links[j.b], rb) - point_vel(&st.links[j.a], ra);
                let bias = err * (p.joint_bias / h);
                let imp = kinv * -(vrel + bias);
                if !imp.is_finite() { continue; }
                apply_impulse(&mut st.links, &cache, j.b, rb, imp, 1.0);
                apply_impulse(&mut st.links, &cache, j.a, ra, imp, -1.0);
            }
            // 2. angular limits
            limit_rows.clear();
            let mut owners: Vec<(usize, usize)> = Vec::new();
            for (ji, j) in rd.joints.iter().enumerate() {
                let before = limit_rows.len();
                j.limit_rows(rd, st, &mut limit_rows);
                for r in before..limit_rows.len() { owners.push((ji, r)); }
            }
            if acc_limit.len() != limit_rows.len() { acc_limit = vec![0.0; limit_rows.len()]; }
            for (ri, row) in limit_rows.iter().enumerate() {
                let (ji, _) = owners[ri];
                let j = &rd.joints[ji];
                let (ca, cb) = (&cache[j.a], &cache[j.b]);
                let k = row.axis.dot((ca.i_inv + cb.i_inv) * row.axis);
                if !(k > 1e-9) { continue; }
                let wrel = st.links[j.b].w - st.links[j.a].w;
                let rate = wrel.dot(row.axis);
                // the inequality that is violated (or the friction row when neither is)
                let over = row.value - row.hi;
                let under = row.lo - row.value;
                let mut dj = 0.0f32;
                let mut lo_clamp = 0.0f32;
                let mut hi_clamp = 0.0f32;
                if over > 0.0 {
                    let bias = -p.limit_bias * over / h;
                    dj = (bias - rate) / k;
                    lo_clamp = f32::NEG_INFINITY;
                } else if under > 0.0 {
                    let bias = p.limit_bias * under / h;
                    dj = (bias - rate) / k;
                    hi_clamp = f32::INFINITY;
                } else if row.friction > 0.0 {
                    dj = -rate / k;
                    let cap = row.friction * h;
                    lo_clamp = -cap;
                    hi_clamp = cap;
                }
                if dj == 0.0 { continue; }
                let prev = acc_limit[ri];
                acc_limit[ri] = (prev + dj).clamp(lo_clamp.min(0.0), hi_clamp.max(0.0));
                let d = acc_limit[ri] - prev;
                if d == 0.0 || !d.is_finite() { continue; }
                let torque = row.axis * d;
                st.links[j.b].w += cb.i_inv * torque;
                st.links[j.a].w -= ca.i_inv * torque;
            }
            // 3. contacts
            for row in rows.iter_mut() {
                solve_contact_row(row, &mut st.links, &cache, &mut vp, &mut wp);
            }
        }

        // ---- integrate ---------------------------------------------------------------------
        for (i, l) in rd.links.iter().enumerate() {
            if l.pinned { continue; }
            let s = &mut st.links[i];
            if !(s.v.is_finite() && s.w.is_finite()) { s.v = Vec3::ZERO; s.w = Vec3::ZERO; }
            s.com += (s.v + vp[i]) * h;
            s.q = integrate_quat(s.q, s.w + wp[i], h);
        }
    }

    // residuals (what the gates assert)
    for j in &rd.joints {
        let (pa, pb) = j.pivots(rd, st);
        stats.joint_error = stats.joint_error.max((pb - pa).length());
        let mut rows = Vec::new();
        j.limit_rows(rd, st, &mut rows);
        for r in &rows { stats.limit_violation = stats.limit_violation.max(r.violation()); }
    }

    st.ticks += 1;
    let mut max_v = 0.0f32;
    let mut max_w = 0.0f32;
    for (i, l) in rd.links.iter().enumerate() {
        if l.pinned { continue; }
        max_v = max_v.max(st.links[i].v.length());
        max_w = max_w.max(st.links[i].w.length());
    }
    if stats.contacts > 0 && max_v < p.sleep_lin && max_w < p.sleep_ang {
        st.quiet += 1;
        if st.quiet >= p.sleep_steps { st.asleep = true; }
    } else {
        st.quiet = 0;
    }
    stats
}

#[allow(clippy::too_many_arguments)]
fn push_contact_row(
    rows: &mut Vec<Row>,
    a: usize,
    b: Option<usize>,
    ra: Vec3,
    rb: Vec3,
    c: &Contact,
    cache: &[LinkCache],
    st: &RagdollState,
    p: &SimParams,
    support_cap: f32,
    h: f32,
) {
    let ca = &cache[a];
    let rn_a = ra.cross(c.n);
    let mut kn = ca.inv_m + rn_a.dot(ca.i_inv * rn_a);
    let mut vrel = point_vel(&st.links[a], ra);
    if let Some(bi) = b {
        let cb = &cache[bi];
        let rn_b = rb.cross(c.n);
        kn += cb.inv_m + rn_b.dot(cb.i_inv * rn_b);
        vrel -= point_vel(&st.links[bi], rb);
    }
    if !(kn > 1e-9) { return; }
    let vn = vrel.dot(c.n);
    let bounce = if vn < -p.restitution_threshold && c.soft.is_none() { -c.e * vn } else { 0.0 };
    let cap = match c.soft { Some(travel) if c.depth < travel => support_cap, _ => f32::INFINITY };
    let vt = vrel - c.n * vn;
    let t1 = if vt.length_squared() > 1e-8 { vt.normalize() } else { c.n.any_orthonormal_vector() };
    let t2 = c.n.cross(t1).normalize_or_zero();
    let tan_k = |t: Vec3| -> f32 {
        let mut k = ca.inv_m + ra.cross(t).dot(ca.i_inv * ra.cross(t));
        if let Some(bi) = b {
            let cb = &cache[bi];
            k += cb.inv_m + rb.cross(t).dot(cb.i_inv * rb.cross(t));
        }
        k
    };
    let target = 0.2 * (c.depth - p.slop).max(0.0) / h;
    rows.push(Row {
        a,
        b,
        ra,
        rb,
        n: c.n,
        t1,
        t2,
        kn,
        kt1: tan_k(t1),
        kt2: tan_k(t2),
        jn: 0.0,
        jt1: 0.0,
        jt2: 0.0,
        jp: 0.0,
        bounce,
        target,
        mu: c.mu,
        cap,
    });
}

fn solve_contact_row(row: &mut Row, st: &mut [LinkState], cache: &[LinkCache], vp: &mut [Vec3], wp: &mut [Vec3]) {
    let rel = |st: &[LinkState], row: &Row| -> Vec3 {
        let mut v = point_vel(&st[row.a], row.ra);
        if let Some(bi) = row.b { v -= point_vel(&st[bi], row.rb); }
        v
    };
    // normal
    let vn = rel(st, row).dot(row.n);
    let dj = (row.bounce - vn) / row.kn;
    let j0 = row.jn;
    row.jn = (row.jn + dj).clamp(0.0, row.cap);
    let d = row.jn - j0;
    if d != 0.0 && d.is_finite() {
        let imp = row.n * d;
        apply_impulse(st, cache, row.a, row.ra, imp, 1.0);
        if let Some(bi) = row.b { apply_impulse(st, cache, bi, row.rb, imp, -1.0); }
    }
    // friction
    let max_f = row.mu * row.jn;
    if max_f > 0.0 {
        for k in 0..2 {
            let (t, kt) = if k == 0 { (row.t1, row.kt1) } else { (row.t2, row.kt2) };
            if !(kt > 1e-9) { continue; }
            let vt = rel(st, row).dot(t);
            let dj = -vt / kt;
            let acc = if k == 0 { row.jt1 } else { row.jt2 };
            let newacc = (acc + dj).clamp(-max_f, max_f);
            let d = newacc - acc;
            if k == 0 { row.jt1 = newacc } else { row.jt2 = newacc }
            if d != 0.0 && d.is_finite() {
                let imp = t * d;
                apply_impulse(st, cache, row.a, row.ra, imp, 1.0);
                if let Some(bi) = row.b { apply_impulse(st, cache, bi, row.rb, imp, -1.0); }
            }
        }
    }
    // split impulse: penetration recovery on the pseudo velocities only
    let mut vprel = vp[row.a] + wp[row.a].cross(row.ra);
    if let Some(bi) = row.b { vprel -= vp[bi] + wp[bi].cross(row.rb); }
    let vpn = vprel.dot(row.n);
    let dj = (row.target - vpn) / row.kn;
    let j0 = row.jp;
    row.jp = (row.jp + dj).max(0.0);
    let d = row.jp - j0;
    if d != 0.0 && d.is_finite() {
        let imp = row.n * d;
        vp[row.a] += imp * cache[row.a].inv_m;
        wp[row.a] += cache[row.a].i_inv * row.ra.cross(imp);
        if let Some(bi) = row.b {
            vp[bi] -= imp * cache[bi].inv_m;
            wp[bi] -= cache[bi].i_inv * row.rb.cross(imp);
        }
    }
}

/// `K = (1/ma + 1/mb) I - [ra]x Ia^-1 [ra]x - [rb]x Ib^-1 [rb]x`, the point-to-point effective
/// mass matrix.
fn eff_mass_matrix(ca: &LinkCache, cb: &LinkCache, ra: Vec3, rb: Vec3) -> Mat3 {
    let sa = skew(ra);
    let sb = skew(rb);
    Mat3::from_diagonal(Vec3::splat(ca.inv_m + cb.inv_m)) - sa * ca.i_inv * sa - sb * cb.i_inv * sb
}

fn skew(v: Vec3) -> Mat3 {
    Mat3::from_cols(Vec3::new(0.0, v.z, -v.y), Vec3::new(-v.z, 0.0, v.x), Vec3::new(v.y, -v.x, 0.0))
}

fn invert3(m: Mat3) -> Option<Mat3> {
    let d = m.determinant();
    if !d.is_finite() || d.abs() < 1e-12 { return None; }
    let i = m.inverse();
    i.is_finite().then_some(i)
}

// ----------------------------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics_settle::{ShapeMaterial, SettleParams};

    fn plane(z: f32, half: f32) -> Vec<[Vec3; 3]> {
        let a = Vec3::new(-half, -half, z);
        let b = Vec3::new(half, -half, z);
        let c = Vec3::new(half, half, z);
        let d = Vec3::new(-half, half, z);
        vec![[a, b, c], [a, c, d]]
    }

    /// A two-link chain: a pinned anchor at the origin and a capsule hanging from a ragdoll joint.
    fn chain(limits: JointLimits, pinned_anchor: bool) -> Ragdoll {
        let anchor = RigidBodyDef {
            shapes: vec![ConvexShape::sphere(Vec3::ZERO, 0.1, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 1.0 }).unwrap()],
            mass: 1.0, com: None, inertia: None, inertia_scale: Vec3::ONE, lin_damping: 0.0, ang_damping: 0.0,
            motion_type: if pinned_anchor { 5 } else { 2 }, flags: 0, node: 0,
        };
        let limb = RigidBodyDef {
            shapes: vec![ConvexShape::capsule(Vec3::ZERO, Vec3::new(0.0, 0.0, -0.5), 0.1, ShapeMaterial { friction: 0.5, restitution: 0.0, mass: 1.0 }).unwrap()],
            mass: 1.0, com: None, inertia: None, inertia_scale: Vec3::ONE, lin_damping: 0.0, ang_damping: 0.05,
            motion_type: 2, flags: 0, node: 1,
        };
        let c = crate::phmo_shapes::ConstraintDef {
            name: 0, node_a: 0, node_b: 1, scale_a: 1.0,
            frame_a: Mat3::IDENTITY, pivot_a: Vec3::ZERO,
            scale_b: 1.0, frame_b: Mat3::IDENTITY, pivot_b: Vec3::ZERO,
            edge: 0,
            kind: crate::phmo_shapes::ConstraintKind::BallSocket,
        };
        let mut rd = build(&[anchor, limb], &[c], &[], "chain".into()).unwrap();
        rd.joints[0].limits = limits;
        rd.self_pairs.clear();
        rd
    }

    #[test]
    fn a_joint_holds_its_two_pivots_together_under_gravity() {
        let rd = chain(JointLimits::Ball, true);
        let world: Vec<[Vec3; 3]> = Vec::new();
        let mut st = rd.state_from_pose(&[Mat4::IDENTITY, Mat4::IDENTITY]);
        let p = SimParams::default();
        let mut worst = 0.0f32;
        for _ in 0..300 {
            let s = step(&rd, &mut st, &world, &p);
            worst = worst.max(s.joint_error);
        }
        // the anchor is pinned, so the limb must still be hanging off it
        assert!(worst < 0.02, "joint drifted {worst} wu");
        assert!(st.links[0].com.length() < 1e-5, "the pinned anchor never moves: {:?}", st.links[0].com);
        assert!(st.links[1].com.z < 0.0, "the limb hangs below: {:?}", st.links[1].com);
    }

    /// #ragdoll A ragdoll constraint's CONE must hold: whatever the limb is spun with, the angle
    /// between the child's twist axis and the parent's cannot exceed the authored half-angle.
    ///
    /// The limb is kicked rather than left to gravity, because hanging straight down from its own
    /// pivot it is already in equilibrium and would never reach any limit at all -- which is how
    /// the first version of this test passed vacuously.
    #[test]
    fn a_ragdoll_cone_limit_holds_within_its_authored_angle() {
        let world: Vec<[Vec3; 3]> = Vec::new();
        let p = SimParams::default();
        let run = |cone: f32| -> f32 {
            let rd = chain(JointLimits::Ragdoll { twist: (-0.1, 0.1), cone, plane: (-cone, cone), friction_torque: 0.0 }, true);
            let mut st = rd.state_from_pose(&[Mat4::IDENTITY, Mat4::IDENTITY]);
            // a hard kick about +Y, which swings the child's +X axis away from the parent's
            st.links[1].w = Vec3::new(0.0, 9.0, 0.0);
            let mut worst = 0.0f32;
            for _ in 0..400 {
                step(&rd, &mut st, &world, &p);
                let ua = Mat3::from_quat(st.links[0].q).x_axis;
                let ub = Mat3::from_quat(st.links[1].q).x_axis;
                worst = worst.max(ua.dot(ub).clamp(-1.0, 1.0).acos());
            }
            worst
        };
        let narrow = run(20f32.to_radians());
        let wide = run(70f32.to_radians());
        assert!(narrow > 10f32.to_radians(), "the kick did drive it out: {} deg", narrow.to_degrees());
        assert!(
            narrow < 20f32.to_radians() + 0.12,
            "a 20 deg cone held: reached {} deg",
            narrow.to_degrees()
        );
        // the gate bites: it is the LIMIT that stopped it, not the dynamics -- a wider cone from
        // the same kick really does go further
        assert!(
            wide > narrow + 20f32.to_radians(),
            "a 70 deg cone lets the same kick go further: {} deg vs {} deg",
            wide.to_degrees(),
            narrow.to_degrees()
        );
        assert!(wide < 70f32.to_radians() + 0.12, "and still holds its own limit: {} deg", wide.to_degrees());
    }

    #[test]
    fn a_free_chain_falls_to_the_floor_and_comes_to_rest_without_exploding() {
        let rd = chain(JointLimits::Ragdoll { twist: (-0.5, 0.5), cone: 1.0, plane: (-1.0, 1.0), friction_torque: 0.0 }, false);
        let world = plane(0.0, 20.0);
        let mut st = rd.state_from_pose(&[Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0)), Mat4::from_translation(Vec3::new(0.0, 0.0, 3.0))]);
        let p = SimParams::default();
        for _ in 0..900 {
            step(&rd, &mut st, &world, &p);
            if st.asleep { break; }
        }
        assert!(st.asleep, "came to rest (quiet {} ticks {})", st.quiet, st.ticks);
        for (i, s) in st.links.iter().enumerate() {
            assert!(s.com.is_finite(), "link {i} is finite");
            assert!(s.com.z > -0.2, "link {i} did not tunnel through the floor: {:?}", s.com);
            assert!(s.com.length() < 50.0, "link {i} did not explode: {:?}", s.com);
            assert!(s.v.length() < 0.1, "link {i} is at rest: {:?}", s.v);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_state_at_the_same_tick() {
        let rd = chain(JointLimits::Ragdoll { twist: (-0.5, 0.5), cone: 1.0, plane: (-1.0, 1.0), friction_torque: 0.0 }, false);
        let world = plane(0.0, 20.0);
        let start = |v: Vec3| {
            let mut st = rd.state_from_pose(&[Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0)), Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0))]);
            for s in st.links.iter_mut() { s.v = v; }
            st
        };
        let run = |v: Vec3, n: usize| {
            let mut st = start(v);
            for _ in 0..n { step(&rd, &mut st, &world, &SimParams::default()); }
            st
        };
        let a = run(Vec3::new(1.0, 0.0, 0.0), 120);
        let b = run(Vec3::new(1.0, 0.0, 0.0), 120);
        assert_eq!(a, b, "the same seed reproduces the tick exactly");
        let c = run(Vec3::new(-1.0, 0.0, 0.0), 120);
        assert_ne!(a.links[1].com, c.links[1].com, "a different seed gives a different ragdoll");
    }

    #[test]
    fn self_collision_pairs_exclude_what_already_touches_at_the_bind_pose() {
        // two spheres 0.1 apart with radius 0.1: overlapping at bind, so never self-collided
        let mk = |x: f32, node: i16| RigidBodyDef {
            shapes: vec![ConvexShape::sphere(Vec3::ZERO, 0.1, ShapeMaterial::default()).unwrap()],
            mass: 1.0, com: None, inertia: None, inertia_scale: Vec3::ONE, lin_damping: 0.0, ang_damping: 0.0,
            motion_type: 2, flags: 0, node,
        }
        .tap(x);
        // Three links: 0 is joined to 2, so the pair under test is (0, 1) alone.
        let joint = crate::phmo_shapes::ConstraintDef {
            name: 0, node_a: 0, node_b: 2, scale_a: 1.0,
            frame_a: Mat3::IDENTITY, pivot_a: Vec3::ZERO,
            scale_b: 1.0, frame_b: Mat3::IDENTITY, pivot_b: Vec3::ZERO,
            edge: 0, kind: crate::phmo_shapes::ConstraintKind::BallSocket,
        };
        let far_away = Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0));
        let bind_close = [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.1, 0.0, 0.0)), far_away];
        let bind_far = [Mat4::IDENTITY, Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)), far_away];
        let bodies = [mk(0.0, 0), mk(0.0, 1), mk(0.0, 2)];
        let close = build(&bodies, &[joint], &bind_close, "three".into()).unwrap();
        assert!(!close.self_pairs.contains(&(0, 1)), "overlapping at bind: {:?}", close.self_pairs);
        let far = build(&bodies, &[joint], &bind_far, "three".into()).unwrap();
        assert!(far.self_pairs.contains(&(0, 1)), "apart at bind, they collide: {:?}", far.self_pairs);
        assert!(!far.self_pairs.contains(&(0, 2)), "a JOINED pair never self-collides");
    }

    trait Tap { fn tap(self, _x: f32) -> Self; }
    impl Tap for RigidBodyDef { fn tap(self, _x: f32) -> Self { self } }

    #[test]
    fn one_link_no_joint_reproduces_the_editors_box_on_a_plane() {
        // the shared-core gate: the same arithmetic the editor's settle runs
        let s = ConvexShape::aabb_box(Vec3::ZERO, Vec3::splat(0.5), 0.0, ShapeMaterial { friction: 0.6, restitution: 0.1, mass: 1.0 }).unwrap();
        let body = PhysBody::from_shape(s, "box".into());
        let rd = Ragdoll {
            links: vec![Link { spheres: body.shapes.iter().flat_map(shape_spheres).collect(), body, node: -1, pinned: false, no_spin: false }],
            joints: Vec::new(),
            self_pairs: Vec::new(),
            desc: "one".into(),
            unmapped_constraints: 0,
        };
        let world = plane(0.0, 10.0);
        let mut st = rd.state_from_pose(&[]);
        st.links[0].com = Vec3::new(0.0, 0.0, 3.0);
        let p = SimParams::default();
        for _ in 0..600 {
            step(&rd, &mut st, &world, &p);
            if st.asleep { break; }
        }
        assert!(st.asleep, "settled");
        assert!((st.links[0].com.z - 0.5).abs() < 0.02, "rest z {}", st.links[0].com.z);
        // and the editor's own params map onto SimParams with the same numbers
        let sp = SettleParams::default();
        assert_eq!((sp.dt, sp.gravity, sp.solver_iters, sp.slop), (p.dt, p.gravity, p.solver_iters, p.slop));
    }

    #[test]
    fn swing_twist_splits_a_pure_twist_exactly() {
        for deg in [-170.0f32, -90.0, -10.0, 0.0, 15.0, 90.0, 170.0] {
            let q = Quat::from_rotation_x(deg.to_radians());
            let (t, swing) = swing_twist_x(q);
            assert!((t.to_degrees() - deg).abs() < 1e-2, "twist {deg} -> {}", t.to_degrees());
            assert!(swing.angle_between(Quat::IDENTITY) < 1e-3, "no swing in a pure twist");
        }
        // a pure swing has no twist
        let q = Quat::from_rotation_y(50f32.to_radians());
        let (t, _) = swing_twist_x(q);
        assert!(t.abs() < 1e-3, "pure swing has no twist: {}", t.to_degrees());
    }


    /// #ragdoll The limit rows' JACOBIANS, checked against a finite difference of the angle they
    /// constrain. These are the one place a sign or a normalisation error would silently make a
    /// joint limit push the wrong way, so they are derived in the module docs and verified here:
    ///   * cone  `d/dt acos(uA . uB) == w_rel . normalize(uA x uB)`
    ///   * plane `d/dt asin(uB . pA) == w_rel . normalize(uB x pA)`
    #[test]
    fn the_cone_and_plane_jacobians_match_a_finite_difference() {
        // a deterministic spread of configurations (no RNG: the gate must be reproducible)
        let mut seed = 0x2545F491u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        // The finite difference is taken in f64: the claim being checked is a piece of
        // MATHEMATICS, and in f32 the difference quotient itself is the inaccurate part --
        // `acos` near +-1 loses most of its mantissa and dividing by a small step multiplies
        // what is left, which showed up as a 0.11 rad "error" in a correct identity.
        let dt = 1e-6f64;
        let mut worst_cone = 0.0f64;
        let mut worst_plane = 0.0f64;
        let mut tested = 0usize;
        let d = |v: Vec3| glam::DVec3::new(v.x as f64, v.y as f64, v.z as f64);
        for _ in 0..400 {
            let ua = d(Vec3::new(next(), next(), next())).normalize_or_zero();
            let ub = d(Vec3::new(next(), next(), next())).normalize_or_zero();
            let t = d(Vec3::new(next(), next(), next()));
            if ua == glam::DVec3::ZERO || ub == glam::DVec3::ZERO { continue; }
            let pa = (t - ua * t.dot(ua)).normalize_or_zero();
            if pa == glam::DVec3::ZERO { continue; }
            let wa = d(Vec3::new(next(), next(), next())) * 2.0;
            let wb = d(Vec3::new(next(), next(), next())) * 2.0;
            let wrel = wb - wa;
            // first-order rotation by an angular velocity
            let spin = |v: glam::DVec3, w: glam::DVec3| (v + w.cross(v) * dt).normalize();
            let (ua2, ub2, pa2) = (spin(ua, wa), spin(ub, wb), spin(pa, wa));
            // cone -- `acos` is ill-conditioned as the angle approaches 0 or pi, so the sweep
            // only scores configurations where the two axes are a real angle apart
            let cr = ua.cross(ub);
            let s = cr.length();
            if s > 0.2 {
                let th0 = ua.dot(ub).clamp(-1.0, 1.0).acos();
                let th1 = ua2.dot(ub2).clamp(-1.0, 1.0).acos();
                let numeric = (th1 - th0) / dt;
                let predicted = wrel.dot(cr / s);
                worst_cone = worst_cone.max((numeric - predicted).abs());
                tested += 1;
            }
            // plane
            let cr2 = ub.cross(pa);
            let s2 = cr2.length();
            if s2 > 0.2 {
                let ph0 = ub.dot(pa).clamp(-1.0, 1.0).asin();
                let ph1 = ub2.dot(pa2).clamp(-1.0, 1.0).asin();
                let numeric = (ph1 - ph0) / dt;
                let predicted = wrel.dot(cr2 / s2);
                worst_plane = worst_plane.max((numeric - predicted).abs());
            }
        }
        assert!(tested > 150, "the sweep really ran: {tested} configurations");
        assert!(worst_cone < 0.01, "cone Jacobian: worst error {worst_cone}");
        assert!(worst_plane < 0.01, "plane Jacobian: worst error {worst_plane}");
        // and the gate bites: the OPPOSITE sign is far outside that tolerance
        let ua = Vec3::X;
        let ub = Vec3::new(0.8, 0.6, 0.0).normalize();
        let wrel = Vec3::new(0.0, 0.0, 1.0);
        let axis = ua.cross(ub).normalize();
        assert!((wrel.dot(axis) - wrel.dot(-axis)).abs() > 1.0, "a flipped axis is a different answer");
    }
}


// ----------------------------------------------------------------------------------------------
// Reading one from the cache
// ----------------------------------------------------------------------------------------------

/// #ragdoll Read an object tag's `phmo` blocks (shapes AND constraints) through the native DLL.
///
/// Takes the OBJECT tag (a `bipd`): `resolve_phmo` walks `obje -> hlmt -> phmo` itself. The same
/// two primitives `SceneController::phmo_blocks` uses, but reachable from the actor path, which
/// had no route to a physics model at all.
pub fn phmo_blocks_for(
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    obj_tag: u32,
) -> Option<crate::phmo_shapes::PhmoBlocks> {
    let phmo = native.resolve_phmo(cache, obj_tag);
    if phmo == 0 || phmo == 0xFFFF_FFFF { return None; }
    let meta = native.tag_meta(cache, phmo, 0, crate::phmo_shapes::REACH_PHMO_META_SIZE as u32)?;
    let rd = |o: usize| -> u32 {
        meta.get(o..o + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    Some(crate::phmo_shapes::PhmoBlocks::read(&crate::phmo_shapes::REACH, |off, elem| {
        let n = rd(off) as i32;
        let ptr = rd(off + 4);
        if n <= 0 || n > 0x10000 || ptr == 0 || ptr == 0xFFFF_FFFF { return Vec::new(); }
        native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32).unwrap_or_default()
    }))
}

/// #ragdoll Build the ragdoll of one object tag against a skeleton.
///
/// The shapes are read in NODE space (an empty node-matrix list), because a link carries its node
/// frame in its own state; `bind` is the skeleton's model-space bind matrices, used only to
/// measure which link pairs already overlap and must never self-collide.
pub fn ragdoll_for(
    native: &hms_native::NativeDll,
    cache: hms_native::CacheHandle,
    obj_tag: u32,
    bind: &[Mat4],
    desc: String,
) -> Option<Ragdoll> {
    let b = phmo_blocks_for(native, cache, obj_tag)?;
    let (defs, _counts) = crate::phmo_shapes::rigid_bodies(&b, &crate::phmo_shapes::REACH, &[]);
    let cons = crate::phmo_shapes::constraints(&b);
    build(&defs, &cons, bind, desc)
}

/// #ragdoll The diagnostic behind the constraint decode: every authored joint, its limits in
/// degrees, and HOW FAR APART its two pivots land at the bind pose. That residual is the proof
/// that `a position` / `b position` are read at the right offsets and in the right spaces -- a
/// correct read puts both at the same world point, a wrong one does not.
///
/// Reachable two ways, and the first needs no GPU:
///   `hms-app --dump-ragdoll <map.map> [bipd-name-substring]`
///   `actor ragdoll <bipd-name-substring>` in the script host.
pub fn describe_constraints(
    b: &crate::phmo_shapes::PhmoBlocks,
    node_xforms: &[Mat4],
    node_name: &dyn Fn(i16) -> String,
    sid_name: &dyn Fn(u32) -> String,
) -> String {
    use crate::phmo_shapes as ps;
    let cons = ps::constraints(b);
    let (present, ok) = ps::constraint_counts(b);
    let edges = ps::node_edges(b);
    let pnodes = ps::phmo_node_names(b);
    let mut s = format!(
        "phmo constraints: {present} element(s) present, {ok} decoded, {} node edge(s), {} phmo node(s)\n",
        edges.len(),
        pnodes.len()
    );
    // The phmo's OWN node list must be the `mode` node list: a constraint's node index is only
    // meaningful against the render model's skeleton, so a disagreement here would invalidate
    // every pivot below. Printed, not assumed.
    let agree = pnodes
        .iter()
        .enumerate()
        .filter(|(i, sid)| sid_name(**sid) == node_name(*i as i16))
        .count();
    if !pnodes.is_empty() {
        s.push_str(&format!("  phmo node names match the mode node names: {agree} of {}\n", pnodes.len()));
    }
    // Per type, present vs decoded: this is what says whether the four tail blocks the engine's
    // own reflection does not reach really are the constraint types the plugin names.
    for (name, present, ok) in ps::constraint_counts_per_type(b) {
        s.push_str(&format!(
            "  block '{name}': {present} element(s), {ok} decoded{}\n",
            if ok == present { "" } else { "  <-- the rest are not orthonormal frames: that block is NOT this type" }
        ));
    }
    let mut worst = 0.0f32;
    for c in &cons {
        let res = c.pivot_residual(node_xforms);
        worst = worst.max(res);
        let lim = match c.kind {
            ps::ConstraintKind::Ragdoll { twist, cone, plane, friction_torque } => format!(
                "twist [{:.1},{:.1}] cone [{:.1},{:.1}] plane [{:.1},{:.1}] friction {friction_torque:.3}",
                twist.0.to_degrees(), twist.1.to_degrees(), cone.0.to_degrees(), cone.1.to_degrees(),
                plane.0.to_degrees(), plane.1.to_degrees()
            ),
            ps::ConstraintKind::LimitedHinge { friction, limit } => format!(
                "limit [{:.1},{:.1}] friction {friction:.3}", limit.0.to_degrees(), limit.1.to_degrees()
            ),
            ps::ConstraintKind::StiffSpring { length } => format!("length {length:.4}"),
            ps::ConstraintKind::Prismatic { limit, max_friction } => format!(
                "limit [{:.1},{:.1}] friction {max_friction:.3}", limit.0.to_degrees(), limit.1.to_degrees()
            ),
            _ => String::new(),
        };
        s.push_str(&format!(
            "  {:<22} {:<16} {} ({}) <-> {} ({})  pivot residual {:.5} wu  frames {}  {}\n",
            sid_name(c.name),
            c.kind.type_name(),
            c.node_a, node_name(c.node_a), c.node_b, node_name(c.node_b),
            res,
            if c.frames_are_orthonormal() { "ok" } else { "NOT ORTHONORMAL" },
            lim
        ));
    }
    s.push_str(&format!("  worst pivot residual at the bind pose: {worst:.5} wu\n"));
    s
}
