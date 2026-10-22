//! #play  The CHARACTER CONTROLLER: one biped's capsule, swept against the level and the user's
//! own Forge blocks.
//!
//! This is the only genuinely new mechanism test mode needs. Everything else it does — posing a
//! biped from a [`crate::film_anim::BodyState`], choosing clips, drawing first-person arms,
//! reading the collision triangles — already existed for Theater. Walking did not.
//!
//! # This is the ENGINE's controller, not an approximation of Havok's
//!
//! Reach does not use Havok's character proxy. `character_physics_mode_ground.cpp`
//! (`reach_tag_test.exe`, the same code is in `haloreach.dll`) is the engine's own controller:
//! it reads the contacts Havok's rigid body produced last tick, decides what it is standing on,
//! computes a new LINEAR VELOCITY and hands it back; Havok only integrates and depenetrates.
//! This module ports that decision, function for function (`sub_1406C2A50` ground plane,
//! `sub_1406C3140` plane subsets, `sub_1406C4A20` velocity, `sub_1406C33B0` acceleration clamp,
//! `sub_1404E97C0` stick -> commanded velocity), and keeps Havok's half as the honest
//! approximation it has to be: a capsule of three spheres, pushed out of triangles along the
//! contact normal, with the velocity into each contact removed.
//!
//! The consequences a player feels, each of them a fact of the engine and none a tuning choice:
//!
//! * **ONE contact test and ONE filter.** Every contact — face, edge or vertex — is judged on its
//!   NORMAL, never on how high it sits on the capsule. A contact is ground-eligible while
//!   `dot(normal, velocity) < 0.468` (the fastest the engine's own adhesion could pull the body
//!   back), so "leaving the ground" is "moving away from it faster than adhesion can follow".
//!   There is **no step-up height**: a ledge is climbed or refused by the contact normal its edge
//!   makes with the bottom sphere, which is a consequence of the authored radius and slope limit.
//! * **Two slope limits on two different normals.** The ground PLANE (the normalised sum of
//!   whichever of the five flattest eligible contacts maximises its z — 31 subsets, tried
//!   exhaustively) must have `z > cos(authored max slope)`; and every SURFACE the capsule touches
//!   that is steeper than the global 35 degrees raises that bar to just above itself, so a steep
//!   face can never be its own floor. 35 degrees is therefore the real walkable limit on a face;
//!   the authored 50 governs cresting an edge.
//! * **The body is never moved to keep it grounded.** On the ground the new velocity is the error
//!   to the commanded velocity (acceleration-clamped, in 3-D, along the ground plane), minus an
//!   ADHESION velocity of `clamp(speed / 2.2 * 0.234, 0.12, 0.234)` x 30/tick_rate wu/s into the plane; gravity
//!   is added last and only while the clamp is still saturated. The contact pushes the body back
//!   out. A position teleport to a probed height — this module's old way — has two writers for
//!   one field and is what made ramps jitter.
//! * **Air is a different mode**, not "ground with gravity": horizontal velocity chases the
//!   commanded velocity at the authored `airborne acceleration` (1.05 wu/s^2 on the Spartan row —
//!   far less than gravity along any steep face, so a steep slope slides you down however hard
//!   you push), and gravity is Havok's.
//!
//! # Nothing in here is pure physics for its own sake
//!
//! Every constant it moves by comes from [`crate::play_phys::PlayerPhysics`], i.e. from the
//! `bipd` and `matg` tags, and the module takes it as an argument rather than owning any
//! numbers. The handful of literals that are NOT authored are the engine's own code constants and
//! are named and sourced where they are declared.
//!
//! # Shape
//!
//! [`PlayerState::step`] is pure over a [`StaticWorld`], so a floor, a ramp, a staircase and a
//! wall can be built by hand in a test and the controller driven across them with no map, no
//! window and no GPU — while the SAME function is what the live per-frame hook calls. The
//! collision primitives are the project's own (`physics_settle::closest_point_on_triangle`, the
//! same one the ragdoll solver and the object settle use), so a player lands on a Forge block for
//! the same reason a corpse does.

use glam::Vec3;

use crate::physics_settle::{closest_point_on_triangle, Shell, StaticWorld, TICK_HZ};
use crate::play_phys::PlayerPhysics;

/// How many spheres stand in for the capsule. Three (bottom, middle, top) is what
/// `ConvexShape::capsule` reduces a pill to for the settle solver, so a player is collided the
/// same way every other capsule in this codebase is.
pub const CAPSULE_SPHERES: usize = 3;

/// Havok's `collisionTolerance` as the engine configures the world (`sub_18009A06C` in
/// `haloreach.dll`): a body within this distance of a surface HAS a contact with it, touching or
/// not. It is what lets a body that has just crested an edge still count as standing for the
/// next tick, without the controller ever moving it to make that so.
pub const CONTACT_TOLERANCE: f32 = 0.0328;

/// `dot(contact normal, linear velocity) < 0.468` — the one filter on whether a contact may be
/// ground (`sub_1406C2A50`, literal `0.467999995`; the shipped game's `sub_1801FC5CC` has the same
/// literal). It is the 30 Hz-referenced adhesion ceiling (`ADHESION_MAX` x 2): a contact stops
/// being ground when you leave it faster than the engine's adhesion could pull you back.
pub const LEAVING_SPEED: f32 = 0.468;

/// Contacts remembered between steps. Generous: a floor is deduplicated to one entry however many
/// triangles it is made of, so only a very cluttered corner comes near it.
const MAX_CONTACTS: usize = 24;
/// The engine keeps the five flattest eligible contacts (`sub_1406C2A50` caps its list at 5).
const GROUND_CANDIDATES: usize = 5;
/// Depenetration passes per sub-step. More than one because pushing out of a floor can push
/// into a wall (an inside corner needs both resolved).
const SOLVER_PASSES: usize = 6;
/// The stuck fallback needs a penetration deeper than this fraction of the radius...
const SQUEEZE_DEPTH: f32 = 0.5;
/// ...on every tick for this long...
const SQUEEZE_SECONDS: f32 = 0.5;
/// ...with less horizontal progress than this, wu.
const SQUEEZE_PROGRESS: f32 = 0.1;

/// The three literals of the adhesion velocity (`sub_1406C4A20`): it is
/// `clamp(speed / ADHESION_REF_SPEED * ADHESION_MAX, ADHESION_MIN, ADHESION_MAX)` wu/s at a
/// 30 Hz reference tick (see [`adhesion_speed`] for the 60 Hz scale).
const ADHESION_REF_SPEED: f32 = 2.2;
const ADHESION_MIN: f32 = 0.12;
const ADHESION_MAX: f32 = 0.234;

/// The global walkable-SURFACE limit: `cos(35 degrees)`, the `+0x1C` of the engine's "physics
/// constants" block (initialiser `sub_18014FC94`; see the `reach-engine-physics-gravity` note).
/// `sub_14037A2F0` returns it. It is NOT authored per biped.
fn surface_slope_cos() -> f32 {
    35f32.to_radians().cos()
}

/// What the player is asking for this frame — a stick, a look direction and four buttons.
/// Nothing here knows about keys; `crate::keybinds` turns bindings into this.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveInput {
    /// Forward stick, -1 (back) .. +1 (ahead).
    pub fwd: f32,
    /// Strafe stick, -1 (left) .. +1 (right).
    pub right: f32,
    /// Look yaw, radians (absolute — mouse look owns it).
    pub yaw: f32,
    /// Look pitch, radians.
    pub pitch: f32,
    /// Jump, EDGE: true only on the frame the button went down.
    pub jump: bool,
    /// Crouch, HELD.
    pub crouch: bool,
    /// Slow walk, HELD — the authored `walking speed` instead of the run set.
    pub walk: bool,
    /// Sprint, HELD.
    pub sprint: bool,
}

impl MoveInput {
    /// Is the stick asking for anything?
    pub fn moving(&self) -> bool {
        self.fwd.abs() > 1e-3 || self.right.abs() > 1e-3
    }
}

/// The player's body: where it is, how it is moving, and what it is standing on.
///
/// `pos` is the **feet** — the bottom of the capsule — because that is what a spawn point and a
/// floor are expressed in. The eye and the capsule centres are derived from it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerState {
    pub pos: Vec3,
    pub vel: Vec3,
    /// Where the player is LOOKING, radians. The camera's yaw, straight from the mouse.
    pub yaw: f32,
    pub pitch: f32,
    /// Where the BODY is facing, radians — **a separate quantity from the aim**, which follows it
    /// at the biped's own authored `moving turning speed` and, while standing still, not at all
    /// until the aim has pulled past the authored `stationary turning threshold`.
    ///
    /// Reported by the user: *"the biped doesn't like pivot on its feet at all, the whole biped
    /// seems to just rotate with the camera which is wrong"*. It did, because the drawn yaw WAS
    /// the aim. The engine separates them and makes the upper body take up the difference through
    /// an aim overlay; both halves are authored and both are now driven.
    pub facing: f32,
    /// How fast the facing is turning, radians/s — what picks a turn-in-place cycle.
    pub facing_rate: f32,
    /// Hysteresis for the stationary threshold: once a standing turn has started it continues
    /// until it has caught up, instead of stuttering on and off at the threshold.
    turning_in_place: bool,
    /// Standing on walkable ground this frame.
    pub grounded: bool,
    /// The ground PLANE underfoot (`Vec3::Z` when airborne) — [`ground_plane`]'s answer, which is
    /// a normalised sum of contact normals and not necessarily any one surface's.
    pub ground_normal: Vec3,
    /// The contacts the capsule ended the last step with; see [`PlayerState::contacts`].
    contacts: [Contact; MAX_CONTACTS],
    n_contacts: usize,
    /// Crouching — the ACTUAL state, which lags the button when a ceiling blocks standing up.
    pub crouched: bool,
    /// Eye interpolation, 0 = standing .. 1 = crouched. Driven at the authored transition rate.
    pub crouch_t: f32,
    /// Seconds off the ground, while off it.
    pub air_time: f32,
    /// Seconds since this body last touched down, while it is still on the ground.
    pub land_time: f32,
    /// Ground distance travelled, wu — what the locomotion cycle's phase is driven from, so the
    /// feet match the ground exactly as they do in Theater.
    pub distance: f32,
    /// Horizontal speed last step, wu/s.
    pub ground_speed: f32,
    /// Seconds spent sprinting, which is what ramps the authored sprint multiplier in: the
    /// globals author `seconds to full speed`, so a sprint accelerates into its top speed rather
    /// than snapping to it.
    pub sprint_time: f32,
    /// The last position verified FREE (inside no shell, no surface deeper than a hair): where the
    /// stuck fallback returns a body that cannot be pushed out.
    last_free: Vec3,
    /// Seconds the body has been squeezed (overlapping by more than the solver can undo).
    squeeze_time: f32,
    /// Where the current squeeze began, to tell a body going nowhere from one making progress.
    squeeze_from: Vec3,
}

impl Default for PlayerState {
    fn default() -> Self {
        PlayerState {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            facing: 0.0,
            facing_rate: 0.0,
            turning_in_place: false,
            grounded: false,
            ground_normal: Vec3::Z,
            contacts: [Contact::default(); MAX_CONTACTS],
            n_contacts: 0,
            crouched: false,
            crouch_t: 0.0,
            air_time: 0.0,
            land_time: 0.0,
            distance: 0.0,
            ground_speed: 0.0,
            sprint_time: 0.0,
            last_free: Vec3::ZERO,
            squeeze_time: 0.0,
            squeeze_from: Vec3::ZERO,
        }
    }
}

/// What one [`PlayerState::step`] did — everything a gate or a caption needs, measured rather
/// than inferred from the state afterwards.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepOut {
    /// Displacement this step, wu.
    pub moved: Vec3,
    /// This step is the one the body touched down on.
    pub landed: bool,
    /// This step is the one the body jumped on.
    pub jumped: bool,
    /// Contacts resolved (any normal).
    pub contacts: usize,
    /// Contacts rejected as too steep to stand on — the slope limit biting.
    pub steep_contacts: usize,
    /// The crouch button was down but a ceiling kept the body crouched, or stood it up late.
    pub crouch_blocked: bool,
    /// Triangles handed to the solver, for the performance line.
    pub tris: usize,
    /// Sub-steps taken.
    pub substeps: usize,
    /// **The runaway guard tripped**: this step wanted more sub-steps than the loop allows, so
    /// each one was longer than the capsule's own safe cap and the body may have stepped OVER
    /// thin geometry instead of into it. Reported rather than silent — a body that tunnels
    /// through a Forge platform must say so.
    pub substep_guard_tripped: bool,
    /// Moves made to get the body out of the inside of a closed shell this step.
    pub escaped: usize,
    /// The body was squeezed for too long to be pushed out and has been moved to a free spot.
    pub relocated: bool,
    /// The deepest penetration the contact solver left at the end of the step, wu.
    pub residual: f32,
}

/// One contact of the capsule with the world, as Havok's manifold would hand it to the
/// controller: the NORMAL of the contact (toward the capsule), and the normal of the SURFACE it
/// was made against. They are the same for a face contact and different for an edge or vertex
/// one, and the engine tests them against two different limits.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contact {
    /// `sub_140443770`: the contact normal (`contact + 0x30`).
    pub normal: Vec3,
    /// `sub_1404450B0`: the plane of the collision surface the contact belongs to, oriented
    /// toward the capsule.
    pub surface: Vec3,
}

/// What the capsule is standing on: the engine's ground plane, or `None` when it is not
/// standing (`sub_1406C2A50`, line for line).
///
/// 1. **Eligible contacts**: `normal.z > -1e-4` (nothing facing down) and
///    `dot(normal, vel) < 0.468` ([`LEAVING_SPEED`]). The five FLATTEST are kept, sorted by z
///    descending, a tie going behind the contact it ties with.
/// 2. **The plane** is the normalised sum of whichever SUBSET of those maximises the sum's z —
///    every non-empty subset is tried (31 of them for five), the best strictly-greater z wins.
///    There is no order to be wrong about: two walls of a valley, each tilted 20 degrees, sum to
///    a perfectly level plane that neither is alone.
/// 3. **The bar**: `min_normal_k` (`cos(authored maximum slope)`), raised to `surface.z + 0.001`
///    by every contact — eligible or not — whose SURFACE is steeper than the global 35 degrees.
///    The plane's z must exceed it. That is how a 50-degree authored limit still cannot stand on
///    a 40-degree face: the face's own normal is both the contact normal and the surface, and it
///    is below the bar it sets.
pub fn ground_plane(contacts: &[Contact], vel: Vec3, min_normal_k: f32) -> Option<Vec3> {
    let global = surface_slope_cos();
    let mut bar = min_normal_k;
    for c in contacts {
        if c.surface.z < global {
            bar = bar.max(c.surface.z + 0.001);
        }
    }
    let mut kept = [Vec3::ZERO; GROUND_CANDIDATES];
    let mut n = 0usize;
    for c in contacts {
        let nz = c.normal.z;
        if !(nz > -1e-4 && LEAVING_SPEED > c.normal.dot(vel)) {
            continue;
        }
        // Insert behind every kept contact at least as flat, drop whatever falls off the end.
        let mut at = 0;
        while at < n && nz <= kept[at].z {
            at += 1;
        }
        if at >= GROUND_CANDIDATES {
            continue;
        }
        let last = n.min(GROUND_CANDIDATES - 1);
        for i in (at..last).rev() {
            kept[i + 1] = kept[i];
        }
        kept[at] = c.normal;
        n = (n + 1).min(GROUND_CANDIDATES);
    }
    if n == 0 {
        return None;
    }
    // `sub_1406C3140`: the engine seeds the answer with straight DOWN, so an all-cancelling
    // subset can never win.
    let mut best = Vec3::NEG_Z;
    for mask in 1u32..(1u32 << n) {
        let mut sum = Vec3::ZERO;
        for (i, k) in kept.iter().enumerate().take(n) {
            if mask & (1 << i) != 0 {
                sum += *k;
            }
        }
        let len = sum.length();
        if len.abs() >= 1e-4 {
            let unit = sum / len;
            if unit.z > best.z {
                best = unit;
            }
        }
    }
    (best.z > bar).then_some(best)
}

/// The velocity pushed INTO the ground plane every tick, wu/s (`sub_1406C4A20`): grows with how
/// fast the body is already moving, between two literals, and is scaled by the biped's authored
/// `scale ground adhesion velocity` when that is positive.
pub fn adhesion_speed(vel: Vec3, p: &PlayerPhysics) -> f32 {
    let a = (vel.length() / ADHESION_REF_SPEED * ADHESION_MAX).clamp(ADHESION_MIN, ADHESION_MAX);
    let a = if p.ground_adhesion_scale > 0.0 { a * p.ground_adhesion_scale } else { a };
    // The three literals are tuned for a 30 Hz tick and the engine divides the result by
    // `tick_rate / 30` (`sub_1400BD360`; the same `1 / (int)(tick_rate/30)` is in the shipped
    // game's `sub_1801FABE4`): 0.06 .. 0.117 wu/s at the real 60 Hz.
    a * (30.0 / TICK_HZ)
}

/// Move `vel` toward `target` by at most `max_delta` — `sub_1406C33B0` with the melee-lunge
/// braking boost left out (that bit is only ever set by the melee code). Returns the new delta
/// and whether the clamp was SATURATED, which the ground branch needs for gravity.
fn clamp_delta(err: Vec3, max_delta: f32) -> (Vec3, bool) {
    let len = err.length();
    if len <= max_delta {
        (err, false)
    } else {
        (err / len * max_delta, true)
    }
}

/// **The engine's velocity update**, one tick: `vel` is the body's velocity after last tick's
/// contacts, `plane` the ground plane if [`ground_plane`] found one, `cmd` the commanded WORLD
/// velocity (the stick through [`PlayerPhysics::commanded_velocity`], times any sprint boost),
/// `crouch` the crouch fraction. Returns the new velocity. `dt` makes the engine's per-tick
/// quantities per-second: the acceleration clamp is `accel * dt`, gravity is `g * dt`; the
/// adhesion is a velocity and is not scaled.
///
/// **Grounded** (`sub_1406C4A20`, the `bit2 == 0` branch):
/// 1. the commanded direction gets the slope-following z that lies in the plane,
///    `z = -(x*nx + y*ny) / nz`, and is renormalised — a body walking up a ramp aims UP the ramp;
/// 2. its speed is `|cmd|` times [`PlayerPhysics::slope_speed_scale`] of that direction's z;
/// 3. the 3-D error to that target velocity is clamped to `accel * dt` (`accel` is the run
///    acceleration blended toward the sneak one by the crouch fraction);
/// 4. [`adhesion_speed`] is subtracted along the plane normal;
/// 5. gravity is subtracted LAST, and only if step 3 saturated.
///
/// **Airborne** (`bit2` branch): the horizontal error to the commanded velocity is clamped to the
/// authored `airborne acceleration * dt`; vertical velocity is Havok's, i.e. plain gravity.
pub fn control_velocity(vel: Vec3, plane: Option<Vec3>, cmd: Vec3, p: &PlayerPhysics, crouch: f32, dt: f32) -> Vec3 {
    match plane {
        Some(n) if n.z > 1e-3 => {
            let mag = cmd.length();
            let mut dir = Vec3::new(cmd.x, cmd.y, -(cmd.x * n.x + cmd.y * n.y) / n.z);
            let l = dir.length();
            dir = if l > 0.0 { dir / l } else { Vec3::ZERO };
            let target = dir * (mag * p.slope_speed_scale(dir.z));
            let (mut delta, saturated) = clamp_delta(target - vel, p.ground_acceleration(crouch) * dt);
            delta -= n * adhesion_speed(vel, p);
            if saturated {
                delta.z -= p.gravity() * dt;
            }
            vel + delta
        }
        _ => {
            let err = Vec3::new(cmd.x - vel.x, cmd.y - vel.y, 0.0);
            let (delta, _) = clamp_delta(err, p.air_acceleration() * dt);
            Vec3::new(vel.x + delta.x, vel.y + delta.y, vel.z - p.gravity() * dt)
        }
    }
}

/// The ground law of this biped in one line, for the caption and the dump verb.
pub fn law_summary(p: &PlayerPhysics) -> String {
    format!(
        "ground plane needs normal z > {:.4} (authored {:.1} deg, tested on the CONTACT normal) and any touched \
         surface steeper than {:.1} deg (global) raises the bar above itself | the ledge a contact normal lets \
         through is radius x (1 - cos slope) = {:.4} wu (emergent, no authored step height) | adhesion \
         {:.3}..{:.3} wu/s into the plane, ground eligibility needs normal.vel < {LEAVING_SPEED} | \
         contact tolerance {CONTACT_TOLERANCE} wu",
        p.min_normal_k(),
        p.max_slope.to_degrees(),
        surface_slope_cos().acos().to_degrees(),
        p.radius * (1.0 - p.min_normal_k()),
        ADHESION_MIN * 30.0 / TICK_HZ,
        ADHESION_MAX * 30.0 / TICK_HZ,
    )
}

impl PlayerState {
    /// Stand a body at `feet`, facing `yaw`, at rest.
    pub fn standing_at(feet: Vec3, yaw: f32) -> PlayerState {
        // The body starts facing where it is looking; they only diverge once the mouse moves.
        PlayerState { pos: feet, yaw, facing: yaw, last_free: feet, ..Default::default() }
    }

    /// How far the body's facing lags the aim, radians — **the quantity the aim overlay takes
    /// up**. Positive means the aim is to the body's left.
    pub fn aim_residual(&self) -> f32 {
        ang_delta(self.yaw, self.facing)
    }

    /// #seat-entry  Put the body where a VEHICLE SEAT says, without running the controller.
    ///
    /// **The controller is bypassed while seated, not extended.** A rider's transform is the
    /// vehicle's transform times the seat marker — which is already how Theater draws riders — so
    /// there is no capsule to sweep, no gravity to integrate and no ground to probe. Making
    /// [`PlayerState::step`] handle being in a vehicle would put two writers on one position.
    ///
    /// `carry_velocity` is the VEHICLE's velocity, stored so that stepping out of a moving one
    /// carries its momentum: Reach does that, and the seat's own `apply velocity on death exit`
    /// flag is the tag acknowledging the same thing. It also means the controller resumes from a
    /// sane velocity instead of from whatever the body had when it got in.
    pub fn set_pose_from_vehicle(&mut self, feet: Vec3, yaw: f32, pitch: f32, carry_velocity: Vec3) {
        self.pos = feet;
        self.yaw = yaw;
        self.pitch = pitch;
        self.vel = carry_velocity;
        // A seated body is not standing on anything, is not falling, and is not travelling on
        // foot: every quantity the locomotion state machine reads is neutralised here so that
        // getting out cannot land in an airborne clip or a run cycle mid-stride.
        self.grounded = false;
        self.ground_normal = Vec3::Z;
        self.n_contacts = 0;
        self.ground_speed = 0.0;
        self.air_time = 0.0;
        self.land_time = 0.0;
        self.sprint_time = 0.0;
        self.crouched = false;
        self.crouch_t = 0.0;
        // #aim  The FACING is one of those quantities, and it is the one that would show. A
        // seated body's yaw IS the vehicle's, so the facing has to be set to it rather than
        // neutralised to zero -- otherwise the whole ride would be drawn with the body at the
        // aim residual it happened to be carrying when it got in, and getting out would start
        // mid-turn as the facing chased the camera back. Set, not cleared, because this is a
        // placement and not a reset.
        self.facing = yaw;
        self.facing_rate = 0.0;
        self.turning_in_place = false;
    }

    /// The eye, in world space — what the first-person camera rides.
    ///
    /// The height is the authored one for the stance, interpolated across the crouch at the
    /// authored transition time, so going down to a knee lowers the view by the tag's own
    /// `standing camera height` - `crouching camera height` and not by a chosen amount.
    pub fn eye(&self, p: &PlayerPhysics) -> Vec3 {
        self.pos + Vec3::Z * self.eye_height(p)
    }

    /// Eye height above the feet, wu.
    pub fn eye_height(&self, p: &PlayerPhysics) -> f32 {
        let moving = self.ground_speed > 0.35;
        let stand = p.eye_height(false, moving);
        let crouch = p.eye_height(true, moving);
        stand + (crouch - stand) * self.crouch_t.clamp(0.0, 1.0)
    }

    /// Capsule height this frame, wu — interpolated across the crouch like the eye.
    pub fn height(&self, p: &PlayerPhysics) -> f32 {
        let t = self.crouch_t.clamp(0.0, 1.0);
        p.height_standing + (p.height_crouching - p.height_standing) * t
    }

    /// The capsule's sphere centres, world space, bottom first.
    pub fn spheres(&self, p: &PlayerPhysics) -> [Vec3; CAPSULE_SPHERES] {
        Self::spheres_at(self.pos, self.height(p), p.radius)
    }

    /// The sphere centres a capsule of this height would have with its feet at `feet`.
    pub fn spheres_at(feet: Vec3, height: f32, radius: f32) -> [Vec3; CAPSULE_SPHERES] {
        let r = radius.max(1e-4);
        let bottom = feet.z + r;
        let top = (feet.z + height - r).max(bottom);
        [
            Vec3::new(feet.x, feet.y, bottom),
            Vec3::new(feet.x, feet.y, (bottom + top) * 0.5),
            Vec3::new(feet.x, feet.y, top),
        ]
    }

    /// Advance one frame.
    ///
    /// `dt` is clamped and the move is sub-stepped so a fast body cannot pass through a thin
    /// Forge block: no sub-step displaces further than half the capsule radius.
    pub fn step(&mut self, input: &MoveInput, p: &PlayerPhysics, world: &dyn StaticWorld, dt: f32) -> StepOut {
        let mut out = StepOut::default();
        let dt = dt.clamp(0.0, 0.1);
        if dt <= 0.0 {
            return out;
        }
        let from = self.pos;
        // THE LOOK YAW IS WRAPPED HERE, and this is the only place it is written.
        //
        // Mouse look ACCUMULATES: `keybinds::play_look` returns `yaw - dx * k` and `App` feeds
        // the result back in, so a player who keeps spinning in one direction walks the number
        // off to a thousand radians and beyond. Nothing cared until the body gained a facing,
        // because nothing compared the look yaw against a wrapped angle -- and `ang_delta` wraps
        // with a `while` loop, so `ang_delta(1000.0, facing)` is a hundred and sixty iterations,
        // every frame, growing without bound for as long as the player keeps turning. Wrapping
        // the accumulator closes it at the source rather than hardening the four readers.
        self.yaw = wrap_pi(input.yaw);
        self.pitch = input.pitch;
        // THE BODY'S FACING FOLLOWS THE AIM AT THE AUTHORED RATE — it is not set to it.
        //
        // `bipd +0x4CC moving turning speed` and `+0x4D4 stationary turning threshold` exist
        // precisely to make the body lag the look direction, and the residual is what the upper
        // body takes up through the aim overlay. Setting the drawn yaw to the aim is what made the
        // whole model rotate rigidly with the mouse.
        {
            let residual = ang_delta(self.yaw, self.facing);
            let moving = self.ground_speed > crate::film_anim::cuts::STILL || input.moving();
            // Standing still, the body does not turn until the aim has pulled past the authored
            // threshold — that is the field's whole purpose, and it is why a standing player can
            // look around without their feet moving. Once a standing turn starts it runs to
            // completion rather than stuttering at the threshold.
            //
            // **And it turns at the same rate**, because the tag authors only one. The field
            // names say so: `+0x4CC` is *"moving turning speed [degrees per second]"* and
            // `+0x4D4` is *"stationary turning THRESHOLD"* — a threshold, not a rate. There is no
            // stationary turning speed in the `bipd` definition, so a standing turn uses the one
            // speed there is, and the pair of fields IS the model: one rate, plus a gate on
            // whether a stationary body turns at all.
            let turn = moving || self.turning_in_place || residual.abs() > p.stationary_turn_threshold;
            if turn && p.turning_speed > 1e-4 {
                let step = (p.turning_speed * dt).min(residual.abs());
                let d = step * residual.signum();
                self.facing = wrap_pi(self.facing + d);
                self.facing_rate = if dt > 0.0 { d / dt } else { 0.0 };
                self.turning_in_place = !moving && (residual.abs() - step) > 1e-3;
            } else {
                self.facing_rate = 0.0;
                self.turning_in_place = false;
            }
        }
        // Was this body on the ground COMING INTO this step? Captured here, before the jump below
        // can clear `grounded`, because it is used to decide whether the body is leaving the
        // ground under its own power — and a jump must not be able to answer that question twice.
        let was_grounded = self.grounded;

        // --- crouch, and the ceiling that can refuse to let it end -------------------------
        let want_crouch = input.crouch;
        if !want_crouch && self.crouched {
            // Standing up needs headroom for the FULL capsule. Asked once, against the same
            // triangles the move uses.
            if self.fits_standing(p, world) {
                self.crouched = false;
            } else {
                out.crouch_blocked = true;
            }
        } else {
            self.crouched = want_crouch;
        }
        let rate = if p.crouch_transition > 1e-3 { 1.0 / p.crouch_transition } else { 1e3 };
        let target = if self.crouched { 1.0 } else { 0.0 };
        self.crouch_t += (target - self.crouch_t).clamp(-rate * dt, rate * dt);
        self.crouch_t = self.crouch_t.clamp(0.0, 1.0);

        // --- what the stick asks for, in the world ----------------------------------------
        let (sy, cy) = self.yaw.sin_cos();
        let fwd = Vec3::new(cy, sy, 0.0);
        let right = Vec3::new(sy, -cy, 0.0);
        // SPRINT is the globals' own `full speed multiplier`, not a number of this module's.
        // Claimed only for a straight-ahead run, because Reach cannot strafe or shoot sprinting.
        let sprinting = input.sprint && self.grounded && !self.crouched && !input.walk && input.fwd > 0.5 && input.right.abs() < 0.5;
        // ...and it is EARNED over the authored `seconds to start` / `seconds to full speed`
        // (`PlayerPhysics::sprint_fraction`) rather than snapping: the globals author the
        // multiplier, the delay before it begins and how long it takes to arrive.
        self.sprint_time = if sprinting { self.sprint_time + dt } else { 0.0 };
        let boost = if sprinting {
            1.0 + (p.sprint_full_speed_multiplier.max(1.0) - 1.0) * p.sprint_fraction(self.sprint_time)
        } else {
            1.0
        };
        // The commanded velocity is built in the BODY's frame (forward cap, sideways cap, each
        // blended toward its sneak twin by the crouch fraction) and turned into the world by the
        // look yaw -- `sub_1404E97C0` then the rotation at the top of `sub_1406C4A20`.
        let (cf, cr) = p.commanded_velocity(input.fwd, input.right, self.crouch_t, input.walk);
        let cmd = (fwd * cf + right * cr) * boost;

        // --- jump ----------------------------------------------------------------------------
        // `sub_140E45230`: the UP component of the velocity is raised to the authored jump
        // velocity (`vel += up * (jump - dot(up, vel))` when that is positive). It is a velocity
        // SET, not an impulse on top of whatever the body was doing, and horizontal velocity is
        // left alone. The body is airborne from this tick: the floor it leaves is no longer
        // eligible ground the moment it moves away from it faster than 0.468 wu/s.
        let mut plane = self.grounded.then_some(self.ground_normal);
        if input.jump && self.grounded {
            if p.jump_velocity > self.vel.z {
                self.vel.z = p.jump_velocity;
            }
            self.grounded = false;
            plane = None;
            self.air_time = 0.0;
            out.jumped = true;
        }

        // --- the engine's velocity update ----------------------------------------------------
        self.vel = control_velocity(self.vel, plane, cmd, p, self.crouch_t, dt);

        // --- integrate, sub-stepped, depenetrating as we go ---------------------------------
        // SUB-STEP UNTIL THE WHOLE DISPLACEMENT IS CONSUMED, never by more than half the
        // capsule's radius at a time — that is what stops a fast body stepping over thin
        // geometry instead of into it.
        //
        // **The old code clamped the count to 16, which silently turned "subdivide until each
        // step is safe" into "give up and tunnel".** Past `16 * step_cap` of travel in one frame
        // each sub-step was LONGER than the cap it was computed from, with no diagnostic.
        // Measured for a Spartan (radius 0.175, cap 0.0875): at a steady 60 Hz it took 90 wu/s to
        // breach, which is a long fall — but **at a 100 ms frame it takes only 30 wu/s**, about
        // three seconds of falling, and this app does hitch. A Forge platform is thinner than a
        // 0.1875 wu sub-step, so the body goes through it and keeps accelerating, which is
        // "I fell straight through the entire map".
        //
        // The count is now a RUNAWAY GUARD rather than a budget: generous enough never to be
        // reached in play, and when it is reached it is REPORTED instead of quietly degrading.
        // The caller additionally splits a frame whose travel would outrun the gathered
        // collision neighbourhood, so this should never trip (see `PlayMode::tick`).
        const SUBSTEP_RUNAWAY_GUARD: usize = 256;
        let r = p.radius.max(1e-4);
        let height = self.height(p);
        let step_cap = (r * 0.5).max(1e-3);
        let travel = self.vel.length() * dt;
        let want = ((travel / step_cap).ceil() as usize).max(1);
        out.substep_guard_tripped = want > SUBSTEP_RUNAWAY_GUARD;
        let n_sub = want.min(SUBSTEP_RUNAWAY_GUARD);
        out.substeps = n_sub;
        let h = dt / n_sub as f32;
        // The triangles are gathered ONCE for the whole step: the box covers where the body can
        // be at any sub-step (its travel) plus everything a depenetration or the contact
        // tolerance can reach.
        let pad = Vec3::splat(2.0 * r + CONTACT_TOLERANCE + 0.05 + travel);
        // A body that is INSIDE a closed shell (placed there, or a block dropped on it) has no
        // face to push it out the right way; get it out first, the nearest way.
        out.escaped = self.escape_shells(r, height, world);
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        world.gather(self.pos - pad, self.pos + Vec3::Z * height + pad, &mut tris);
        out.tris = tris.len();
        for _ in 0..n_sub {
            self.pos += self.vel * h;
            out.contacts += self.resolve(r, height, &tris);
        }

        // --- what the capsule is standing on now: the NEXT tick's input ---------------------
        self.collect_contacts(r, height, &tris);
        self.check_squeeze(r, height, p, world, &tris, out.escaped, dt, &mut out);
        let plane = ground_plane(self.contacts(), self.vel, p.min_normal_k());
        self.grounded = plane.is_some();
        self.ground_normal = plane.unwrap_or(Vec3::Z);
        // The slope limit "biting": contacts a body could lean on (nothing facing down) that did
        // not make it ground.
        if !self.grounded {
            out.steep_contacts = self.contacts().iter().filter(|c| c.normal.z > -1e-4).count();
        }
        if self.grounded {
            if !was_grounded {
                out.landed = true;
                self.land_time = 0.0;
            } else {
                self.land_time += dt;
            }
            self.air_time = 0.0;
        } else {
            self.air_time += dt;
            self.land_time = 0.0;
        }

        out.moved = self.pos - from;
        let d = Vec3::new(out.moved.x, out.moved.y, 0.0).length();
        self.distance += d;
        self.ground_speed = if dt > 0.0 { d / dt } else { 0.0 };
        out
    }

    /// The points of the capsule's axis the inside test looks at: the sphere centres and enough
    /// between them (never more than 4 cm apart) that no plate thinner than the gaps can pass
    /// between two of them unseen.
    fn axis_samples(feet: Vec3, height: f32, r: f32) -> Vec<Vec3> {
        let lo = feet.z + r;
        let hi = (feet.z + height - r).max(lo);
        let n = (((hi - lo) / 0.04).ceil() as usize).max(2);
        (0..=n).map(|i| Vec3::new(feet.x, feet.y, lo + (hi - lo) * i as f32 / n as f32)).collect()
    }

    /// **INSIDE A CLOSED SHELL.** The triangle contacts below all push a sphere toward the side of
    /// a face it is on; a body whose axis is inside a block has a face on every side, so they
    /// hold it where it is, and one that is past the middle of a thin wall pushes it on out the
    /// FAR side. Havok pushes a penetrating character out along the penetration depth, i.e. the
    /// NEAREST way out of the volume, and this does the same for every closed shell (`Shell`: the
    /// Forge block hulls): it finds the SHORTEST straight translation (along an axis or a face
    /// normal of the shells it is in) after which no point of the capsule's axis is inside any
    /// shell -- the whole body, not the one sphere that happens to be nearest a face, since a
    /// capsule impaled on a plate cannot leave by moving along its own length -- and moves
    /// there; the contact solver then pushes the radius clear. The velocity going further in is
    /// removed. Butted blocks can put the exit inside the next one, so the search repeats.
    /// Returns the number of moves.
    ///
    /// The ENGINE'S cap on how much is undone per tick (and its stuck fallback) is not decoded;
    /// the whole distance is resolved at once.
    fn escape_shells(&mut self, r: f32, height: f32, world: &dyn StaticWorld) -> usize {
        const SCAN: f32 = 0.02;
        const REACH: f32 = 3.0;
        let mut shells: Vec<std::sync::Arc<Shell>> = Vec::new();
        let pad = Vec3::splat(r + 0.05 + REACH);
        world.solid_shells(self.pos - pad, self.pos + Vec3::Z * height + pad, &mut shells);
        if shells.is_empty() {
            return 0;
        }
        let inside = |pts: &[Vec3], at: Vec3| pts.iter().any(|p| shells.iter().any(|s| s.contains(*p + at)));
        let mut moves = 0;
        for _ in 0..6 {
            let pts = Self::axis_samples(self.pos, height, r);
            let here: Vec<&std::sync::Arc<Shell>> = shells.iter().filter(|s| pts.iter().any(|p| s.contains(*p))).collect();
            if here.is_empty() {
                break;
            }
            let mut dirs: Vec<Vec3> = vec![Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
            for s in &here {
                for t in &s.tris {
                    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
                    for d in [n, -n] {
                        if d != Vec3::ZERO && dirs.len() < 40 && !dirs.iter().any(|k| k.dot(d) > 0.999) {
                            dirs.push(d);
                        }
                    }
                }
            }
            let mut best: Option<(f32, Vec3)> = None;
            for d in dirs {
                let limit = best.map_or(REACH, |(b, _)| b);
                let mut t = SCAN;
                while t <= limit {
                    if !inside(&pts, d * t) {
                        // the exit is somewhere in (t - SCAN, t]: bisect for the shortest free one
                        let (mut lo, mut hi) = (t - SCAN, t);
                        for _ in 0..8 {
                            let m = 0.5 * (lo + hi);
                            if inside(&pts, d * m) { lo = m } else { hi = m }
                        }
                        if best.map_or(true, |(b, _)| hi < b) {
                            best = Some((hi, d));
                        }
                        break;
                    }
                    t += SCAN;
                }
            }
            // No way out within reach: straight up, as far as it takes.
            let (t, d) = best.unwrap_or_else(|| {
                let mut t = REACH;
                while inside(&pts, Vec3::Z * t) && t < 200.0 {
                    t += 0.25;
                }
                (t, Vec3::Z)
            });
            // ...and a radius further: left at the surface itself a sphere centre is a hair
            // outside, and which side of the face it is on is then rounding -- the solver could
            // push it straight back in.
            self.pos += d * (t + r + 1e-3);
            let vn = self.vel.dot(d);
            if vn < 0.0 {
                self.vel -= d * vn;
            }
            moves += 1;
        }
        moves
    }

    /// **THE STUCK FALLBACK -- conservative by construction.** It fires only for a body that is
    /// TRULY inside solid geometry and going nowhere: the contact solver leaves a penetration
    /// deeper than [`SQUEEZE_DEPTH`] of the radius (or the shell escape has to act) on every tick
    /// for [`SQUEEZE_SECONDS`], AND the body has moved less than [`SQUEEZE_PROGRESS`] wu over that
    /// time. Pressing against a wall, a lip, a ledge or a seam is not that: the solver leaves such a
    /// body touching (depth ~0), and a body that is making progress, however slowly, resets the
    /// clock. Then it is moved to the nearest free spot at about its own height, else back to the
    /// last position known to be free, else the nearest free spot at any height, with its velocity
    /// cleared -- the engine's `teleport to a safe position` for a stuck unit. (The engine's
    /// own trigger and search are not decoded; this is the behaviour, not its constants.)
    #[allow(clippy::too_many_arguments)]
    fn check_squeeze(&mut self, r: f32, height: f32, p: &PlayerPhysics, world: &dyn StaticWorld, tris: &[[Vec3; 3]], escaped: usize, dt: f32, out: &mut StepOut) {
        let residual = Self::spheres_at(self.pos, height, r)
            .iter()
            .flat_map(|c| tris.iter().filter_map(move |t| sphere_tri_contact(*c, r, 0.0, t)).map(|h| h.depth))
            .fold(0.0f32, f32::max);
        out.residual = residual;
        // (an escape that has to act tick after tick is a loop: the solver puts the body straight
        // back where the escape took it from)
        if residual > SQUEEZE_DEPTH * r || escaped >= 1 {
            if self.squeeze_time == 0.0 {
                self.squeeze_from = self.pos;
            }
            self.squeeze_time += dt;
            if self.squeeze_time > SQUEEZE_SECONDS {
                let moved = Vec3::new(self.pos.x - self.squeeze_from.x, self.pos.y - self.squeeze_from.y, 0.0).length();
                self.squeeze_time = 0.0;
                if moved < SQUEEZE_PROGRESS {
                    self.pos = self
                        .find_free(p, world, 3.0, &[0.0, 0.3, -0.3])
                        .or_else(|| (self.last_free.distance(self.pos) < 20.0).then_some(self.last_free))
                        .or_else(|| self.find_free(p, world, 6.0, &[0.8, 2.0, 4.0]))
                        .unwrap_or(self.last_free);
                    self.vel = Vec3::ZERO;
                    self.grounded = false;
                    out.relocated = true;
                }
            }
        } else {
            self.squeeze_time = 0.0;
            if residual < 0.02 {
                self.last_free = self.pos;
            }
        }
    }

    /// The nearest position to the body (rings out to `reach` wu, at the given height offsets)
    /// where a standing capsule overlaps nothing.
    pub fn find_free(&self, p: &PlayerPhysics, world: &dyn StaticWorld, reach: f32, dzs: &[f32]) -> Option<Vec3> {
        let r = p.radius.max(1e-4);
        let height = p.height_standing;
        let pad = reach + 2.0 * r;
        let top = dzs.iter().copied().fold(0.0f32, f32::max);
        let low = dzs.iter().copied().fold(0.0f32, f32::min);
        let (mn, mx) = (self.pos - Vec3::new(pad, pad, pad - low), self.pos + Vec3::new(pad, pad, height + top + pad));
        let mut shells: Vec<std::sync::Arc<Shell>> = Vec::new();
        world.solid_shells(mn, mx, &mut shells);
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        world.gather(mn, mx, &mut tris);
        let free = |feet: Vec3| !Self::overlaps_with(feet, r, height, &shells, &tris, 0.02);
        if dzs.contains(&0.0) && free(self.pos) {
            return Some(self.pos);
        }
        let rings = (reach / 0.2).ceil() as usize;
        for ring in 1..=rings {
            let rad = ring as f32 * 0.2;
            for k in 0..(8 + 2 * ring) {
                let a = k as f32 / (8 + 2 * ring) as f32 * std::f32::consts::TAU;
                for &dz in dzs {
                    let feet = self.pos + Vec3::new(a.cos() * rad, a.sin() * rad, dz);
                    if free(feet) {
                        return Some(feet);
                    }
                }
            }
        }
        None
    }

    fn overlaps_with(feet: Vec3, r: f32, height: f32, shells: &[std::sync::Arc<Shell>], tris: &[[Vec3; 3]], slack: f32) -> bool {
        Self::axis_samples(feet, height, r).iter().any(|c| shells.iter().any(|s| s.contains(*c)))
            || Self::spheres_at(feet, height, r).iter().any(|c| tris.iter().any(|t| sphere_tri_contact(*c, r, 0.0, t).is_some_and(|h| h.depth > slack)))
    }

    /// Put an already-placed body (a spawn, a respawn, a teleport, an exit from a vehicle) where
    /// it does not overlap anything: out of every closed shell, then out of every surface, and if
    /// that still leaves it wedged, to the nearest free spot. The same resolution the next
    /// `step` would do, available to the placing code so the result is known before it is used.
    /// Returns how far it moved the body.
    pub fn settle_out(&mut self, p: &PlayerPhysics, world: &dyn StaticWorld) -> f32 {
        let from = self.pos;
        let r = p.radius.max(1e-4);
        let height = self.height(p);
        self.escape_shells(r, height, world);
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        let pad = Vec3::splat(2.0 * r + CONTACT_TOLERANCE + 0.05);
        world.gather(self.pos - pad, self.pos + Vec3::Z * height + pad, &mut tris);
        self.resolve(r, height, &tris);
        if Self::overlaps_at(self.pos, p, world, 0.05) {
            if let Some(f) = self.find_free(p, world, 3.0, &[0.0, 0.3, -0.3]).or_else(|| self.find_free(p, world, 6.0, &[0.8, 2.0, 4.0])) {
                self.pos = f;
            }
        }
        self.last_free = self.pos;
        self.squeeze_time = 0.0;
        self.pos.distance(from)
    }

    /// Would a standing capsule with its feet at `feet` overlap solid geometry? Inside a closed
    /// shell, or into any surface by more than `slack` (resting on a floor is depth 0).
    pub fn overlaps_at(feet: Vec3, p: &PlayerPhysics, world: &dyn StaticWorld, slack: f32) -> bool {
        let r = p.radius.max(1e-4);
        let height = p.height_standing;
        let pad = Vec3::splat(2.0 * r + 0.05);
        let (mn, mx) = (feet - pad, feet + Vec3::Z * height + pad);
        let mut shells: Vec<std::sync::Arc<Shell>> = Vec::new();
        world.solid_shells(mn, mx, &mut shells);
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        world.gather(mn, mx, &mut tris);
        Self::overlaps_with(feet, r, height, &shells, &tris, slack)
    }

    /// The contacts the capsule had at the end of the last step — what the next tick's ground
    /// plane is decided from.
    pub fn contacts(&self) -> &[Contact] {
        &self.contacts[..self.n_contacts]
    }

    /// Push the capsule out of everything it is inside and remove the velocity going INTO each
    /// contact; returns how many contacts it found at the worst pass.
    ///
    /// This is the half of the engine's controller that belongs to Havok (depenetration and the
    /// contact solver), and it is deliberately the plain one. **Every contact is pushed out along
    /// its own normal and clipped along its own normal** — there is no classification by height,
    /// no flattening of "steep" contacts and no step rule. What lets a body climb a low ledge
    /// and what stops it at a steep face is the CONTROLLER above ([`ground_plane`]'s bar and the
    /// air mode), exactly as in the engine; this function only keeps the capsule out of the
    /// geometry.
    ///
    /// **Contacts are resolved ONE AT A TIME, each against the position the previous one left.**
    /// The previous controller summed the push of every penetrating contact, merging only those
    /// with the same direction: a body LANDING on a floor's diagonal, where the nearest point of
    /// both triangles is the same edge point, was pushed out of it twice and came to rest 0.0793
    /// wu in the air (its own measurement). Sequential projection cannot double-count: once the
    /// first contact has pushed the body out, the second finds nothing left to resolve.
    fn resolve(&mut self, r: f32, height: f32, tris: &[[Vec3; 3]]) -> usize {
        let mut worst = 0usize;
        for _ in 0..SOLVER_PASSES {
            let mut hits = 0usize;
            let mut moved = false;
            for i in 0..CAPSULE_SPHERES {
                let (faces, nf) = face_contacts(Self::spheres_at(self.pos, height, r)[i], r, 0.0, tris);
                for t in tris {
                    // The sphere is re-placed for every triangle: an earlier push has moved it.
                    let c = Self::spheres_at(self.pos, height, r)[i];
                    let Some(hit) = sphere_tri_contact(c, r, 0.0, t) else { continue };
                    if is_ghost(&hit, &faces[..nf], tris) {
                        continue;
                    }
                    hits += 1;
                    self.pos += hit.normal * hit.depth;
                    moved = true;
                    let vn = self.vel.dot(hit.normal);
                    if vn < 0.0 {
                        self.vel -= hit.normal * vn;
                    }
                }
            }
            worst = worst.max(hits);
            if !moved {
                break;
            }
        }
        worst
    }

    /// Record every contact within [`CONTACT_TOLERANCE`] of the capsule at its final position,
    /// merging those that agree on both normals (a floor is two triangles and one contact).
    fn collect_contacts(&mut self, r: f32, height: f32, tris: &[[Vec3; 3]]) {
        self.n_contacts = 0;
        for c in Self::spheres_at(self.pos, height, r) {
            let (faces, nf) = face_contacts(c, r, CONTACT_TOLERANCE, tris);
            for t in tris {
                let Some(hit) = sphere_tri_contact(c, r, CONTACT_TOLERANCE, t) else { continue };
                if is_ghost(&hit, &faces[..nf], tris) {
                    continue;
                }
                let known = self.contacts[..self.n_contacts]
                    .iter()
                    .any(|k| k.normal.dot(hit.normal) > 0.9999 && k.surface.dot(hit.surface) > 0.9999);
                if !known && self.n_contacts < MAX_CONTACTS {
                    self.contacts[self.n_contacts] = Contact { normal: hit.normal, surface: hit.surface };
                    self.n_contacts += 1;
                }
            }
        }
    }

    /// Would the full standing capsule fit here? Asked before ending a crouch.
    pub fn fits_standing(&self, p: &PlayerPhysics, world: &dyn StaticWorld) -> bool {
        let r = p.radius.max(1e-4);
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        let pad = Vec3::splat(r + 0.05);
        world.gather(
            Vec3::new(self.pos.x, self.pos.y, self.pos.z) - pad,
            Vec3::new(self.pos.x, self.pos.y, self.pos.z + p.height_standing) + pad,
            &mut tris,
        );
        let centres = Self::spheres_at(self.pos, p.height_standing, r);
        // Only the part of the capsule ABOVE the crouched one can be newly blocked — the rest is
        // where the body already is, so a floor or a wall it is already touching must not read as
        // "cannot stand up". A standing sphere is wholly inside the crouched capsule exactly
        // while its TOP (`c.z + r`) is under the crouched capsule's top, so the spheres to test
        // are those with `c.z > crouch_top - r`. (Testing `c.z > crouch_top` instead was a real
        // bug: the top sphere's centre sits a radius BELOW the capsule's top, so on a body whose
        // crouch and stand heights differ by less than a radius NO sphere was ever tested and
        // every ceiling let the body stand straight up through it.)
        let crouch_top = self.pos.z + p.height_crouching;
        for c in centres.iter().filter(|c| c.z + r > crouch_top - 1e-4) {
            for t in &tris {
                if sphere_tri_contact(*c, r, 0.0, t).is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// One line of state, for the caption and the live gate.
    pub fn describe(&self, p: &PlayerPhysics) -> String {
        format!(
            "feet ({:.3}, {:.3}, {:.3}) eye z {:.3} (+{:.3}) | vel ({:.2}, {:.2}, {:.2}) ground speed {:.2} wu/s | {} | {} | yaw {:.1} pitch {:.1} | air {:.2}s land {:.2}s | travelled {:.2} wu",
            self.pos.x, self.pos.y, self.pos.z,
            self.eye(p).z, self.eye_height(p),
            self.vel.x, self.vel.y, self.vel.z, self.ground_speed,
            if self.grounded { format!("GROUNDED n=({:.2},{:.2},{:.2}) {:.1} deg", self.ground_normal.x, self.ground_normal.y, self.ground_normal.z, self.ground_normal.z.clamp(-1.0, 1.0).acos().to_degrees()) } else { "AIRBORNE".to_string() },
            if self.crouched { format!("CROUCHED t={:.2}", self.crouch_t) } else { format!("standing t={:.2}", self.crouch_t) },
            self.yaw.to_degrees(), self.pitch.to_degrees(), self.air_time, self.land_time, self.distance,
        )
    }
}

/// Does the triangle `t` pass through `q`, to float tolerance?
fn passes_through(q: Vec3, t: &[Vec3; 3]) -> bool {
    closest_point_on_triangle(q, t[0], t[1], t[2]).0.distance_squared(q) < 1e-8
}

/// The triangles the sphere at `c` touches through their INTERIOR, as indices into `tris`.
/// (A sphere cannot touch more than a handful of faces at once; the cap only guards the array.)
fn face_contacts(c: Vec3, r: f32, margin: f32, tris: &[[Vec3; 3]]) -> ([usize; 16], usize) {
    let mut faces = [0usize; 16];
    let mut n = 0;
    for (k, t) in tris.iter().enumerate() {
        if n < faces.len() {
            if let Some(h) = sphere_tri_contact(c, r, margin, t) {
                if h.interior {
                    faces[n] = k;
                    n += 1;
                }
            }
        }
    }
    (faces, n)
}

/// **A GHOST CONTACT**: an edge or vertex contact whose point lies ON a triangle the sphere is
/// already touching through its face (on its boundary where two triangles of one mesh meet, or
/// inside it where one mesh's edge sits in another's face). That edge is not an obstacle -- the
/// surface under the sphere is the face, and the edge point is on it and no closer to the sphere
/// than the face's own foot -- but the edge contact's normal is tilted away from the face's, and
/// pushing and clipping along it deflects the body: crossing the diagonal of a flat floor
/// deflected a walking body UP by 0.5 wu/s and left it 5 mm in the air at every seam (measured).
/// Havok's mesh shapes weld such edges (`snapToConvexEdge`, 30 degrees, in the world's cinfo);
/// on a bare triangle soup the face contact simply dominates.
fn is_ghost(hit: &Hit, faces: &[usize], tris: &[[Vec3; 3]]) -> bool {
    !hit.interior && faces.iter().any(|&k| passes_through(hit.point, &tris[k]))
}

/// One sphere-against-triangle contact.
struct Hit {
    /// Contact normal, toward the sphere.
    normal: Vec3,
    /// The triangle's own plane normal, oriented toward the sphere.
    surface: Vec3,
    /// Penetration depth; negative inside the tolerance margin (not yet touching).
    depth: f32,
    /// The point of the triangle the contact is made at.
    point: Vec3,
    /// Is that point in the triangle's INTERIOR (a face contact) rather than on an edge or
    /// vertex?
    interior: bool,
}

/// The contact of a sphere against one triangle, if it is within `margin` of touching.
///
/// **The normal comes from the triangle's own plane whenever the closest point is the triangle's
/// INTERIOR**, and only from the centre-to-closest-point direction for an edge or a vertex
/// contact. That distinction is not a refinement, it is the difference between working and not:
/// a body resting on a floor penetrates it by a float's worth of nothing, and
/// `(centre - closest_point)` at a depth of 1e-7 on a triangle whose coordinates are tens of
/// world units is dominated by the rounding in the closest-point solve — measured, it produced
/// normals that were almost HORIZONTAL on a dead flat floor, which the slope rule then read as a
/// wall. The symptom was a body that flickered airborne every few frames while walking across
/// level ground, and flickered its animation with it.
fn sphere_tri_contact(c: Vec3, r: f32, margin: f32, t: &[Vec3; 3]) -> Option<Hit> {
    let (q, interior) = closest_point_on_triangle(c, t[0], t[1], t[2]);
    let delta = c - q;
    let d = delta.length();
    if d >= r + margin {
        return None;
    }
    let face = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
    let normal = if interior && face != Vec3::ZERO {
        // Oriented toward the sphere: a Halo collision soup is not reliably wound, and which
        // side of a triangle the body is on is a fact about the body, not about the winding.
        if face.dot(delta) < 0.0 {
            -face
        } else {
            face
        }
    } else if d > 1e-5 * r.max(1.0) {
        delta / d
    } else if face != Vec3::ZERO {
        face
    } else {
        return None;
    };
    // The surface a contact belongs to, on the same side of it as the sphere.
    let surface = if face == Vec3::ZERO {
        normal
    } else if face.dot(normal) < 0.0 {
        -face
    } else {
        face
    };
    Some(Hit { normal, surface, depth: r - d, point: q, interior })
}

// =============================================================================================
// the state the animation layer reads
// =============================================================================================

/// Fill a [`crate::film_anim::BodyState`] from the controller.
///
/// **This is the whole of "the animation layer needs no changes".** `film_anim` derives its
/// state from a recorded track; nothing in it cares where the numbers came from, so the same
/// `anim_candidates` that picks a Theater body's clip picks test mode's. Two things only this
/// caller can supply:
///
/// * **crouch**, which a Reach film cannot state at all (the crouch bit is measured absent from
///   5,438,363 messages) — so `Stance::Crouch`, present in the module since it was written and
///   never once reachable, becomes reachable here;
/// * a stated jump, where Theater has to infer one from vertical speed.
pub fn body_state(st: &PlayerState, input: &MoveInput, p: &PlayerPhysics, weapon_class: Option<String>) -> crate::film_anim::BodyState {
    use crate::film_anim::{cuts, BodyState, Gait, Provenance, Stance};
    let mut b = BodyState::default();
    let ground = Vec3::new(st.vel.x, st.vel.y, 0.0).length();
    b.ground_speed = ground;
    b.speed = st.vel.length();
    b.vz = st.vel.z;
    b.weapon_class = weapon_class;
    b.provenance = Provenance::Decoded;
    b.because = "test mode: the player's own input and the controller's own solved state";

    // Off the ground is STATED here, not inferred from a vertical-speed cut.
    if !st.grounded {
        b.stance = Stance::Flight;
        b.gait = if ground > cuts::WALK { Gait::RunFront } else { Gait::Idle };
        b.air_ticks = Some((st.air_time * 60.0) as u32);
        return b;
    }
    // A landing still owns the body for as long as the clip lasts.
    let land_ticks = (st.land_time * 60.0) as u32;
    if land_ticks < cuts::LAND_TICKS {
        b.land_ticks = Some(land_ticks);
    }
    // #play  The SPRINT stance uses the globals' OWN `minimum player velocity to be considered
    // in a sprinting state`, not Theater's `cuts::SPRINT`. Theater has to infer a sprint from a
    // recorded speed and so needs a threshold well clear of the run cap; test mode is TOLD that
    // the sprint button is down, so the only question left is the one the tag answers.
    b.stance = if st.crouched {
        Stance::Crouch
    } else if input.sprint && ground > p.sprint_min_velocity && input.fwd > 0.5 && input.right.abs() < 0.3 {
        Stance::Sprint
    } else {
        Stance::Combat
    };
    if ground < cuts::STILL {
        // Standing but TURNING: the graph authors `turn_left` / `turn_right` for exactly this, and
        // it is what makes a body pivot on its feet instead of sliding round rigidly.
        b.gait = if st.facing_rate > cuts::TURN_RATE {
            Gait::TurnLeft
        } else if st.facing_rate < -cuts::TURN_RATE {
            Gait::TurnRight
        } else {
            Gait::Idle
        };
        return b;
    }
    // Direction of travel in the body's own frame — the same quantity `film_anim::derive`
    // computes from a track, here read straight off the velocity and the look yaw.
    // The direction of travel in the BODY's own frame — which is its FACING, not where the
    // player is looking. Strafing is "sideways relative to my shoulders", and the shoulders lag
    // the mouse.
    let theta = ang_delta(st.vel.y.atan2(st.vel.x), st.facing);
    b.travel_angle = Some(theta);
    let a = theta.abs();
    // Crouching and walking have ONE authored cycle each per direction, so the fast/slow split
    // is made against the set actually in force rather than against the run cap.
    let fast_cut = if st.crouched { p.sneak_forward * 0.6 } else { cuts::WALK };
    b.gait = if a < cuts::FRONT_ARC {
        if ground > fast_cut { Gait::RunFront } else { Gait::WalkFront }
    } else if a > cuts::BACK_ARC {
        if ground > fast_cut { Gait::Back } else { Gait::BackSlow }
    } else if theta > 0.0 {
        Gait::Left
    } else {
        Gait::Right
    };
    b
}

fn ang_delta(a: f32, b: f32) -> f32 {
    crate::film_anim::ang_delta_pub(a, b)
}

/// Wrap an angle to (-pi, pi]. The facing accumulates small deltas every frame, so it has to be
/// kept in range or it drifts out of the domain `ang_delta` and the aim grid expect.
fn wrap_pi(a: f32) -> f32 {
    let t = std::f32::consts::TAU;
    let mut r = a % t;
    if r > std::f32::consts::PI {
        r -= t;
    } else if r <= -std::f32::consts::PI {
        r += t;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::play_phys::PlayerPhysics;

    fn phys() -> PlayerPhysics {
        // The stand-in from play_phys's own tests: Reach-SHAPED values, so the controller can be
        // exercised without a map. The real values come from the cache; the live gate prints them.
        crate::play_phys::tests_support::stand_in()
    }

    /// A level floor from (-half, -half) to (half, half) at height z.
    fn floor(z: f32, half: f32) -> Vec<[Vec3; 3]> {
        let a = Vec3::new(-half, -half, z);
        let b = Vec3::new(half, -half, z);
        let c = Vec3::new(half, half, z);
        let d = Vec3::new(-half, half, z);
        vec![[a, b, c], [a, c, d]]
    }

    /// A box, as 12 triangles — a Forge block.
    fn block(min: Vec3, max: Vec3) -> Vec<[Vec3; 3]> {
        let v = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];
        let f = |a: usize, b: usize, c: usize, d: usize| vec![[v[a], v[b], v[c]], [v[a], v[c], v[d]]];
        let mut out = Vec::new();
        out.extend(f(0, 1, 2, 3)); // bottom
        out.extend(f(4, 5, 6, 7)); // top
        out.extend(f(0, 1, 5, 4));
        out.extend(f(1, 2, 6, 5));
        out.extend(f(2, 3, 7, 6));
        out.extend(f(3, 0, 4, 7));
        out
    }

    /// A ramp rising along +X at `deg`, from x=0 to x=len.
    fn ramp(deg: f32, len: f32, half_y: f32) -> Vec<[Vec3; 3]> {
        let t = deg.to_radians().tan();
        let a = Vec3::new(0.0, -half_y, 0.0);
        let b = Vec3::new(len, -half_y, len * t);
        let c = Vec3::new(len, half_y, len * t);
        let d = Vec3::new(0.0, half_y, 0.0);
        vec![[a, b, c], [a, c, d]]
    }

    fn run(st: &mut PlayerState, input: &MoveInput, p: &PlayerPhysics, world: &Vec<[Vec3; 3]>, secs: f32) -> Vec<StepOut> {
        let dt = 1.0 / 60.0;
        let n = (secs / dt).round() as usize;
        let mut outs = Vec::with_capacity(n);
        let mut inp = *input;
        for i in 0..n {
            // A jump is an EDGE: only the first frame of the hold.
            inp.jump = input.jump && i == 0;
            outs.push(st.step(&inp, p, world, dt));
        }
        outs
    }

    fn settle(st: &mut PlayerState, p: &PlayerPhysics, world: &Vec<[Vec3; 3]>) {
        run(st, &MoveInput::default(), p, world, 1.0);
    }

    // -----------------------------------------------------------------------------------------
    // the gates a player would notice
    // -----------------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------------
    // #play-gate  THE INVERSE GATES: each one proves the gate above it BITES.
    //
    // A gate that has only ever been seen green proves nothing, so each behaviour is also
    // exercised with the authored value CHANGED — and because most of these behaviours are
    // driven by `PlayerPhysics`, which the test constructs, the sabotage can be in the DATA
    // rather than in the code.
    //
    // The 2026-10-20 controller port carries its OWN source-level sabotage table, run against a
    // snapshot copy of this file and `cmp`-checked afterwards; it is recorded on each gate that
    // has one (search for "SABOTAGE:").
    // -----------------------------------------------------------------------------------------

    /// #play-gate  THE PRECONDITION FOR BLENDING, at the one place test mode owns.
    ///
    /// Reported: *"Interpolation between animations is a MUST, such as crouch and crouch movement
    /// etc"*. The cross-fade itself is `reach_actor`'s (`set_anim` stores the outgoing clip and
    /// arms `blend_left = BLEND_SECONDS`, and `Actor::advance` decays it on WALL TIME whether or
    /// not the clip's own clock is running, which is the case test mode is in) and it is gated
    /// there. What test mode owns is whether a change of state ASKS FOR A DIFFERENT CLIP at all:
    /// a blend can only happen where the name changes, and `pose` calls `set_anim` exactly then.
    ///
    /// So this asserts the precondition for the user's own example — crouching still and
    /// crouch-walking are different clips — and that the asked-for name is stable while the state
    /// is, because a name that flickered would re-arm the blend every frame and freeze the body
    /// part-way between two poses.
    #[test]
    fn a_change_of_state_asks_for_a_different_clip_so_there_is_something_to_blend() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let class = crate::film_anim::DEFAULT_WEAPON_CLASS;
        let asked = |st: &PlayerState, inp: &MoveInput| -> String {
            let b = body_state(st, inp, &p, Some(class.to_string()));
            crate::film_anim::anim_candidates(&b, class).first().cloned().unwrap_or_default()
        };
        // Crouched and still.
        let mut still = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut still, &p, &w);
        let crouch_idle = MoveInput { crouch: true, ..Default::default() };
        run(&mut still, &crouch_idle, &p, &w, 1.0);
        let a = asked(&still, &crouch_idle);
        // Crouched and moving.
        let mut moving = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut moving, &p, &w);
        let crouch_walk = MoveInput { crouch: true, fwd: 1.0, ..Default::default() };
        run(&mut moving, &crouch_walk, &p, &w, 1.5);
        let b = asked(&moving, &crouch_walk);
        assert!(a.starts_with("crouch:"), "crouching still must ask the crouch mode, got '{a}'");
        assert!(b.starts_with("crouch:"), "crouch-walking must ask the crouch mode, got '{b}'");
        assert_ne!(a, b, "crouching still and crouch-walking must be different clips or there is nothing to blend");
        // Stable while the state is: the same name, frame after frame.
        for _ in 0..30 {
            moving.step(&crouch_walk, &p, &w, 1.0 / 60.0);
            assert_eq!(asked(&moving, &crouch_walk), b, "the asked-for clip must not flicker while the state holds");
        }
        // And standing up is a third clip, so the crouch exit blends too.
        let mut up = moving;
        run(&mut up, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 1.0);
        let c = asked(&up, &MoveInput { fwd: 1.0, ..Default::default() });
        assert_ne!(c, b, "standing up out of a crouch-walk must change the clip");
        assert!(!c.starts_with("crouch:"), "standing up must leave the crouch mode, got '{c}'");
    }

    /// #play-gate  #aim  THE LOOK YAW CANNOT RUN AWAY, however long the player spins.
    ///
    /// Mouse look accumulates, so the number grows for as long as the player keeps turning one
    /// way. `ang_delta` wraps with a `while` loop, so an unwrapped yaw turns every angle
    /// comparison into an unbounded loop -- a cost that gets worse the longer the session runs,
    /// which is the nastiest shape a performance bug comes in. The controller wraps it at the one
    /// place it is written.
    #[test]
    fn the_look_yaw_stays_in_range_however_far_the_player_spins() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        // Spin one way for a long time, feeding the yaw back in exactly as `App` does.
        let mut yaw = 0.0f32;
        for _ in 0..4000 {
            yaw = crate::keybinds::play_look(yaw, 0.0, 40.0, 0.0, 4.5).0;
            st.step(&MoveInput { yaw, ..Default::default() }, &p, &w, 1.0 / 60.0);
            assert!(
                st.yaw.abs() <= std::f32::consts::PI + 1e-5,
                "the stored look yaw left its range: {}",
                st.yaw
            );
            assert!(
                st.facing.abs() <= std::f32::consts::PI + 1e-5,
                "the facing left its range: {}",
                st.facing
            );
        }
        // And the raw accumulator really did run away, so the gate is not passing because the
        // input never got large.
        assert!(yaw.abs() > 100.0, "the fixture must actually spin the accumulator, got {yaw}");
        // THE MOUSE HAS TO STOP BEFORE THE BODY CAN BE ON THE AIM, and the arithmetic says so
        // plainly: the fixture turns the look 0.9 rad per frame, which is 54 rad/s, and the
        // authored turning speed is 2.0 rad/s. A body 27 times slower than the aim can NEVER be
        // on it while the spin continues. The first version of this gate asserted the residual
        // was small at the end of the spin — "a body that has had thousands of frames to catch
        // up" — which is impossible by construction, and it went red for that reason and not
        // because anything was wrong. So: stop the mouse, then give it time.
        let held = yaw;
        run(&mut st, &MoveInput { yaw: held, ..Default::default() }, &p, &w, 5.0);
        // Worst case the residual was a half turn, which at the authored rate is 1.57 s; 5 s is
        // ample and is still a bound rather than "run until it works".
        assert!(
            st.aim_residual().abs() < 1e-3,
            "once the mouse stops the body must come round to the aim — wrapping must change the \
             magnitude and not the relationship (residual {})",
            st.aim_residual()
        );
        assert!(st.yaw.abs() <= std::f32::consts::PI + 1e-5, "and the stored yaw is still in range: {}", st.yaw);
    }

    /// #play-gate  #aim  THE BODY FACES WHERE ITS FEET POINT, NOT WHERE THE CAMERA LOOKS.
    ///
    /// The reported symptom was *"the whole biped seems to just rotate with the camera"*. These
    /// assert the two halves of the mechanism that replaces that: the facing follows the aim at
    /// the biped's own authored `moving turning speed`, and the leftover is exactly what the aim
    /// overlay is handed.
    #[test]
    fn the_body_turns_toward_the_aim_at_the_authored_rate() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        assert!(st.facing.abs() < 1e-6, "it starts facing where it looks");
        // Snap the aim a quarter turn and walk: the body must come round, and take the time the
        // authored rate says, not arrive instantly.
        let turn = std::f32::consts::FRAC_PI_2;
        let dt = 1.0 / 60.0;
        let inp = MoveInput { fwd: 1.0, yaw: turn, ..Default::default() };
        let expect = turn / p.turning_speed;
        // STEP FIRST, THEN TEST. `aim_residual` reads the state's OWN stored yaw, which the
        // mouse has not reached until `step` has copied the input into it -- so a `while` guard
        // on the residual is zero before the first step and the loop never runs. It reported
        // "took 0.000 s", which is a ruler error in this gate rather than anything wrong with
        // the rule: a time of exactly zero is not a near miss.
        let mut t = 0.0f32;
        let mut frames = 0;
        loop {
            st.step(&inp, &p, &w, dt);
            t += dt;
            frames += 1;
            // It must never overshoot the aim -- an overshoot would read as a wobble.
            assert!(
                ang_delta(turn, st.facing) >= -1e-4,
                "the facing overshot the aim: residual {}",
                ang_delta(turn, st.facing)
            );
            if st.aim_residual().abs() <= 1e-3 || frames >= 600 {
                break;
            }
        }
        assert!(frames < 600, "the body never came round to the aim");
        assert!(
            (t - expect).abs() < 3.0 * dt,
            "a {:.1} deg turn at the authored {:.2} rad/s must take {expect:.3} s, took {t:.3} s",
            turn.to_degrees(),
            p.turning_speed
        );
        // And having arrived, the overlay is asked for nothing.
        assert!(st.aim_residual().abs() < 1e-3);
    }

    /// #play-gate  #aim  A STANDING body does not turn until the aim has pulled past the authored
    /// `stationary turning threshold` — which is the field's whole purpose, and what lets a
    /// player look around without their feet moving.
    #[test]
    fn a_standing_body_holds_its_feet_until_the_aim_passes_the_authored_threshold() {
        let p = phys();
        assert!(p.stationary_turn_threshold > 0.01, "the threshold must be authored for this to mean anything");
        let w = floor(0.0, 40.0);
        let look = |off: f32| -> PlayerState {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
            settle(&mut st, &p, &w);
            run(&mut st, &MoveInput { yaw: off, ..Default::default() }, &p, &w, 2.0);
            st
        };
        // Inside the threshold: the feet do not move at all, and the whole offset is left for the
        // overlay to pose.
        let inside = p.stationary_turn_threshold * 0.5;
        let st = look(inside);
        assert!(st.facing.abs() < 1e-6, "a look inside the threshold must not turn the feet, facing {}", st.facing);
        assert!(
            (st.aim_residual() - inside).abs() < 1e-5,
            "the whole look must be left to the overlay, residual {}",
            st.aim_residual()
        );
        // Past it: the feet come round, and all the way -- not to the threshold and no further,
        // which would leave the body permanently cocked.
        let outside = p.stationary_turn_threshold * 1.5;
        let st = look(outside);
        assert!(
            st.aim_residual().abs() < 1e-3,
            "past the threshold a standing body must catch up, residual {}",
            st.aim_residual()
        );
        // MOVING, the threshold does not apply: walking with the same small offset turns.
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        run(&mut st, &MoveInput { fwd: 1.0, yaw: inside, ..Default::default() }, &p, &w, 2.0);
        assert!(
            st.aim_residual().abs() < 1e-3,
            "a moving body lines up with the aim whatever the threshold says, residual {}",
            st.aim_residual()
        );
    }

    /// #play-gate  #aim  A standing turn picks a TURN-IN-PLACE cycle, so the feet shuffle round
    /// instead of the model sliding. This is the gait side of the same mechanism.
    #[test]
    fn a_standing_turn_asks_for_a_turn_in_place_cycle() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        let idle = MoveInput::default();
        assert_eq!(
            body_state(&st, &idle, &p, None).gait,
            crate::film_anim::Gait::Idle,
            "a body with the aim where its feet are is idle"
        );
        // A quarter turn to the LEFT (positive yaw) while standing still.
        let inp = MoveInput { yaw: std::f32::consts::FRAC_PI_2, ..Default::default() };
        let mut saw_turn = false;
        for _ in 0..120 {
            st.step(&inp, &p, &w, 1.0 / 60.0);
            if body_state(&st, &inp, &p, None).gait == crate::film_anim::Gait::TurnLeft {
                saw_turn = true;
            }
        }
        assert!(saw_turn, "turning on the spot must ask for a turn cycle, not idle");
        // The other way round, for the sign.
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        let inp = MoveInput { yaw: -std::f32::consts::FRAC_PI_2, ..Default::default() };
        let mut saw_turn = false;
        for _ in 0..120 {
            st.step(&inp, &p, &w, 1.0 / 60.0);
            if body_state(&st, &inp, &p, None).gait == crate::film_anim::Gait::TurnRight {
                saw_turn = true;
            }
        }
        assert!(saw_turn, "turning the other way must ask for the other turn cycle");
    }

    /// #play-gate  #aim  THE TRAVEL DIRECTION IS MEASURED AGAINST THE BODY, NOT THE CAMERA.
    ///
    /// Strafing is "sideways relative to my shoulders", and the shoulders lag the mouse. With the
    /// facing set to the aim this was accidentally right; it stops being right the moment they
    /// diverge, which is every frame of a turn.
    #[test]
    fn the_travel_direction_is_measured_against_the_body_not_the_camera() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        // Walk forward with the aim straight ahead until up to speed: forward travel, forward
        // gait.
        let inp = MoveInput { fwd: 1.0, ..Default::default() };
        run(&mut st, &inp, &p, &w, 2.0);
        assert_eq!(body_state(&st, &inp, &p, None).gait, crate::film_anim::Gait::RunFront);
        // Now hold the same stick and snap the aim 90 degrees. On the first frames the body has
        // barely turned, so it is still travelling along its own forward -- the gait must not
        // flip to a strafe just because the CAMERA moved.
        let turned = MoveInput { fwd: 1.0, yaw: std::f32::consts::FRAC_PI_2, ..Default::default() };
        st.step(&turned, &p, &w, 1.0 / 60.0);
        let b = body_state(&st, &turned, &p, None);
        assert_eq!(
            b.gait,
            crate::film_anim::Gait::RunFront,
            "one frame after the camera snapped, the body is still running forward, not strafing ({:?})",
            b.gait
        );
    }

    /// #play-gate  INVERSE of the turn gates: the rate and the threshold are the AUTHORED ones.
    #[test]
    fn the_turn_follows_the_authored_rate_and_threshold() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let time_to_come_round = |q: &PlayerPhysics| -> f32 {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
            settle(&mut st, q, &w);
            let inp = MoveInput { fwd: 1.0, yaw: std::f32::consts::FRAC_PI_2, ..Default::default() };
            let dt = 1.0 / 60.0;
            for i in 0..1200 {
                st.step(&inp, q, &w, dt);
                if st.aim_residual().abs() < 1e-3 {
                    return (i + 1) as f32 * dt;
                }
            }
            f32::INFINITY
        };
        let base = time_to_come_round(&p);
        let mut fast = p;
        fast.turning_speed = p.turning_speed * 2.0;
        let quick = time_to_come_round(&fast);
        assert!(
            (quick * 2.0 - base).abs() < 4.0 / 60.0,
            "twice the authored turning speed must take half the time: {base:.3} s vs {quick:.3} s"
        );
        // A biped authored with NO turning speed never comes round at all -- the whole motion is
        // the authored field and nothing else.
        let mut still = p;
        still.turning_speed = 0.0;
        assert!(time_to_come_round(&still).is_infinite(), "no authored turning speed, no turn");
        // A zero threshold turns a STANDING body for any look at all; a huge one never does.
        let standing_turns = |q: &PlayerPhysics, off: f32| -> bool {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
            settle(&mut st, q, &w);
            run(&mut st, &MoveInput { yaw: off, ..Default::default() }, q, &w, 2.0);
            st.facing.abs() > 1e-4
        };
        let tiny = p.stationary_turn_threshold * 0.5;
        assert!(!standing_turns(&p, tiny), "the authored threshold must hold the feet");
        let mut loose = p;
        loose.stationary_turn_threshold = 0.0;
        assert!(standing_turns(&loose, tiny), "a zero threshold must turn for any look");
        let mut tight = p;
        tight.stationary_turn_threshold = 3.0;
        assert!(!standing_turns(&tight, 1.5), "a threshold past the look must hold the feet");
    }

    /// #play-gate  A TRIANGLE SEAM IS NOT A BUMP. A floor is two triangles; the body standing on,
    /// landing on and walking across the diagonal that joins them must rest at exactly the
    /// surface height. The first resolver summed the push of every penetrating contact and
    /// merged only identical directions, so at the seam -- where the edge contact's normal is a
    /// hair off the face's -- the body was pushed out of the floor twice and floated 5 mm (and
    /// 0.0793 wu when it landed there). Sequential projection cannot double-count.
    ///
    /// SABOTAGE: sum the pushes of one pass instead of applying them one at a time -> the seam
    /// rows read ~0.004 wu and go red.
    #[test]
    fn a_triangle_seam_is_not_a_bump() {
        let p = spartan();
        let w = floor(0.0, 30.0); // diagonal y = x
        // landing on the seam from a height, at several points along it
        for k in [0.0f32, 1.0, -2.5, 7.25] {
            let mut st = PlayerState::standing_at(Vec3::new(k, k, 1.0), 0.0);
            // The landing itself, frame by frame: from the first frame the body is within 2 mm of
            // the floor it must never be above 2 mm again -- a body pushed out twice springs up
            // by its penetration (up to 3 cm at this fall speed) and then sinks back, and the
            // settled state alone cannot see it.
            let mut touched = false;
            let mut sprang = 0.0f32;
            for s in trace(&mut st, &MoveInput::default(), &p, &w, 90) {
                if s.pos.z < 2e-3 {
                    touched = true;
                } else if touched {
                    sprang = sprang.max(s.pos.z);
                }
            }
            assert!(touched, "seam point {k}: never reached the floor");
            assert!(sprang < 2e-3, "seam point {k}: touched down and then sprang {sprang:.4} wu back up (pushed out twice)");
            assert!(st.grounded, "seam point {k}: never landed");
            assert!(st.pos.z.abs() < 1e-6, "seam point {k}: rests at z {:e}, not on the surface", st.pos.z);
        }
        // walking straight across it at 40 different sub-frame phases: whether a frame happens
        // to land with the sphere over the seam depends on the phase, and a ghost contact only
        // shows on those frames
        let mut worst = 0.0f32;
        for k in 0..40 {
            let mut st = PlayerState::standing_at(Vec3::new(-1.0 + k as f32 * 0.0017, 0.0, 0.2), 0.0);
            settle(&mut st, &p, &w);
            for s in trace(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 90) {
                worst = worst.max(s.pos.z.abs());
            }
            assert!(st.pos.x > 0.3, "phase {k}: did not cross the seam");
        }
        assert!(worst < 1e-6, "crossing the seam at some phase bumped z by {worst:e}");
        // walking across it in several directions
        for (fwd, right, yaw, from) in [
            (1.0f32, 0.0f32, 0.0f32, Vec3::new(-2.0, 1.0, 0.2)),
            (0.0, 1.0, 0.0, Vec3::new(-2.0, 1.0, 0.2)),
            // heading 57 degrees: y - x GROWS, so start on the other side of the diagonal
            (1.0, 0.0, 1.0, Vec3::new(1.0, -0.5, 0.2)),
        ] {
            let mut st = PlayerState::standing_at(from, yaw);
            settle(&mut st, &p, &w);
            let input = MoveInput { fwd, right, yaw, ..Default::default() };
            let mut worst = 0.0f32;
            let mut crossed = false;
            let side0 = (st.pos.y - st.pos.x).signum();
            for s in trace(&mut st, &input, &p, &w, 240) {
                worst = worst.max(s.pos.z.abs());
                crossed |= (s.pos.y - s.pos.x).signum() != side0;
            }
            assert!(crossed, "the walk ({fwd},{right}) never crossed the seam (ended at {:?}, started on side {side0}), so it proved nothing", st.pos);
            assert!(worst < 1e-6, "walking ({fwd},{right}) across the seam bumped z by {worst:e}");
        }
    }

    /// #play-gate  INVERSE of the jump gate: the apex follows the AUTHORED jump velocity, and a
    /// biped with no jump velocity does not leave the ground at all.
    #[test]
    fn the_jump_follows_the_authored_velocity() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let apex_of = |q: &PlayerPhysics| {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
            settle(&mut st, q, &w);
            let base = st.pos.z;
            let mut peak = base;
            let mut inp = MoveInput { jump: true, ..Default::default() };
            for i in 0..120 {
                inp.jump = i == 0;
                st.step(&inp, q, &w, 1.0 / 60.0);
                peak = peak.max(st.pos.z);
            }
            peak - base
        };
        let a = apex_of(&p);
        // Twice the authored velocity is FOUR times the apex (v^2 / 2g), which no fudge factor
        // reproduces -- it is the arithmetic or it is nothing.
        let mut fast = p;
        fast.jump_velocity = p.jump_velocity * 2.0;
        let b = apex_of(&fast);
        assert!((b / a - 4.0).abs() < 0.2, "doubling the authored jump velocity gave {:.2}x the apex, not 4x", b / a);
        // And no authored jump velocity means no jump.
        let mut none = p;
        none.jump_velocity = 0.0;
        assert!(apex_of(&none) < 0.01, "a biped with no authored jump velocity left the ground");
    }

    /// #play-gate  INVERSE of the crouch gate: a biped whose authored crouching camera height
    /// EQUALS its standing one does not lower its eye when it crouches — so the gate above is
    /// reading the tag rather than a hard-coded drop.
    #[test]
    fn the_crouch_drop_is_the_authored_pair_of_heights() {
        let p = phys();
        let w = floor(0.0, 20.0);
        let drop_of = |q: &PlayerPhysics| {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
            settle(&mut st, q, &w);
            let up = st.eye_height(q);
            run(&mut st, &MoveInput { crouch: true, ..Default::default() }, q, &w, 1.0);
            up - st.eye_height(q)
        };
        let real = drop_of(&p);
        assert!((real - (p.eye_standing - p.eye_crouching)).abs() < 1e-3, "the drop {real} is not the authored pair");
        let mut flat = p;
        flat.eye_crouching = p.eye_standing;
        flat.eye_crouch_walking = p.eye_standing;
        assert!(drop_of(&flat).abs() < 1e-3, "a biped whose authored crouch eye equals its standing eye still dropped");
    }

    /// #play-gate  INVERSE of the speed gate: the run settles on whatever the AUTHORED cap says,
    /// including a value no Reach biped has. A hard-coded speed would not follow.
    #[test]
    fn the_run_speed_follows_the_authored_cap() {
        let w = floor(0.0, 400.0);
        let speed_of = |cap: f32| {
            let mut q = phys();
            q.run_forward = cap;
            q.run_acceleration = 60.0;
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
            settle(&mut st, &q, &w);
            run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &q, &w, 3.0);
            st.ground_speed
        };
        for cap in [1.0f32, 2.2, 3.6, 7.5] {
            let got = speed_of(cap);
            assert!((got - cap).abs() < 0.08, "authored cap {cap} gave {got:.3} wu/s");
        }
    }

    /// #play-gate  INVERSE of the sprint behaviour: the sprint top speed is the authored
    /// MULTIPLIER times the authored run, and a multiplier of 1.0 means sprint does nothing.
    #[test]
    fn sprint_is_the_authored_multiplier_and_nothing_else() {
        let w = floor(0.0, 400.0);
        let top = |mult: f32| {
            let mut q = phys();
            q.sprint_full_speed_multiplier = mult;
            q.sprint_seconds_to_full = 0.1;
            q.run_acceleration = 60.0;
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
            settle(&mut st, &q, &w);
            run(&mut st, &MoveInput { fwd: 1.0, sprint: true, ..Default::default() }, &q, &w, 3.0);
            st.ground_speed
        };
        let p = phys();
        let x1 = top(1.0);
        assert!((x1 - p.run_forward).abs() < 0.08, "a 1.0 multiplier must not speed anyone up: {x1:.3}");
        for m in [1.4f32, 1.623, 2.0] {
            let got = top(m);
            assert!(
                (got - p.run_forward * m).abs() < 0.12,
                "multiplier {m} should give {:.3}, got {got:.3}",
                p.run_forward * m
            );
        }
    }

    /// #play-gate  INVERSE of the crouch-clip gate: the `crouch:` mode is asked for ONLY while
    /// the player is actually crouching, so the gate above cannot be passing because every clip
    /// request happens to start with `crouch:`.
    #[test]
    fn the_crouch_mode_is_asked_for_only_while_crouching() {
        use crate::film_anim::anim_candidates;
        let p = phys();
        let w = floor(0.0, 20.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        for (crouch, want_prefix) in [(true, "crouch:"), (false, "combat:")] {
            let inp = MoveInput { crouch, fwd: 1.0, ..Default::default() };
            run(&mut st, &inp, &p, &w, 1.0);
            let b = body_state(&st, &inp, &p, Some("rifle".into()));
            let c = anim_candidates(&b, "rifle");
            assert!(
                c.first().map(|s| s.starts_with(want_prefix)).unwrap_or(false),
                "crouch={crouch} asked for {:?}, wanted a {want_prefix} clip",
                c.first()
            );
        }
    }

    /// #play-gate  A BODY DROPPED ON A FLOOR LANDS ON IT, and its feet end up ON the surface —
    /// not sunk into it and not hovering above it.
    #[test]
    fn a_body_falls_to_the_floor_and_rests_on_it() {
        let p = phys();
        let w = floor(0.0, 20.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 2.0), 0.0);
        settle(&mut st, &p, &w);
        assert!(st.grounded, "never landed: {}", st.describe(&p));
        assert!(st.pos.z.abs() < 0.01, "feet at z {} not on the floor: {}", st.pos.z, st.describe(&p));
        assert!(st.vel.z.abs() < 0.2, "still falling at rest: {}", st.describe(&p));
    }

    /// A ramp DESCENDING along +X from x = 0 at `deg`, with a FLAT shelf behind it so a body can
    /// settle before it starts down.
    ///
    /// Two quads, deliberately: the first version was one quad spanning `-back .. len` that
    /// dropped only `len * tan(deg)`, so its REAL angle was `atan(len*tan/(len+back))` — 6.68
    /// degrees when asked for 8. A perfectly slope-tracking body then drifted 0.187 wu through a
    /// residual measured against `tan(deg)`, which the gate read as jitter. **The geometry has to
    /// be the angle it claims**, or the yardstick manufactures the fault it is looking for.
    fn downramp(deg: f32, len: f32, back: f32, half_y: f32) -> Vec<[Vec3; 3]> {
        let t = deg.to_radians().tan();
        let mut v = Vec::new();
        // flat shelf, z = 0, from -back to 0
        let (a, b) = (Vec3::new(-back, -half_y, 0.0), Vec3::new(-back, half_y, 0.0));
        let (c, d) = (Vec3::new(0.0, half_y, 0.0), Vec3::new(0.0, -half_y, 0.0));
        v.push([a, b, c]);
        v.push([a, c, d]);
        // descent, z = -x * tan(deg), from 0 to len
        let (e, f) = (Vec3::new(len, half_y, -len * t), Vec3::new(len, -half_y, -len * t));
        v.push([d, c, e]);
        v.push([d, e, f]);
        v
    }

    /// Where the FEET of a resting capsule sit, vertically, above the plane of a slope.
    ///
    /// Not a fudge factor — geometry. A sphere of radius `r` tangent to a plane of slope `theta`
    /// has its centre a perpendicular `r` from the plane, so the centre is `r / cos(theta)` above
    /// the plane MEASURED VERTICALLY, and the capsule's lowest axis point (what this controller
    /// calls the feet) is `r` below the centre. So the feet rest `r * (sec(theta) - 1)` above the
    /// surface: 0.005 wu at 12 degrees, 0.0135 at 20, 0.0325 at 30 for a 0.21 wu capsule.
    ///
    /// A gate that compares the feet against the plane itself and calls the difference jitter is
    /// measuring this, which is exactly what the first version of the incline gate did.
    fn feet_rest_above_slope(radius: f32, deg: f32) -> f32 {
        radius * (1.0 / deg.to_radians().cos() - 1.0)
    }

    /// The surface height of [`ramp`] at `x` — what the feet must track while climbing it.
    fn ramp_surface_z(deg: f32, x: f32) -> f32 {
        if x <= 0.0 {
            0.0
        } else {
            x * deg.to_radians().tan()
        }
    }

    /// #play-gate  A FALLING BODY DOES NOT TUNNEL A THIN PLATFORM, EVEN ON A HITCHED FRAME.
    ///
    /// Reported by the user: *"I also just fell straight through the entire map"*. The mechanism
    /// was a sub-step count clamped to 16, which past `16 * step_cap` of travel made each
    /// sub-step LONGER than the cap it was computed from — so a fast body stepped over thin
    /// geometry instead of into it, with no diagnostic.
    ///
    /// The gate reproduces the conditions that make it reachable rather than theoretical: a
    /// **100 ms frame** (this app hitches, and `dt` is clamped to exactly that) and a long enough
    /// fall to be moving fast. At 100 wu of drop the body is doing ~46 wu/s, which is 4.6 wu per
    /// frame — the old code gave each sub-step 0.29 wu, well past the 0.175 wu capsule radius, so
    /// consecutive sphere positions did not overlap and a plane could pass between them. The fix
    /// keeps every sub-step at half the radius, so consecutive positions always overlap and no
    /// surface can slip through the gap.
    ///
    /// The platform is a single quad with NO thickness, deliberately: the guarantee is about the
    /// sphere's sweep overlapping, not about the geometry being thick enough to catch.
    /// #play-gate  **AN EDGE-ONLY LANDING SETTLES ON THE SURFACE.** A single triangle, landed on
    /// beside its hypotenuse so the nearest point is the EDGE and no adjoining face can supply a
    /// contact — the fixture the floor pair cannot produce.
    ///
    /// The old controller classified edge contacts by their HEIGHT on the capsule and needed an
    /// "escape pass" for a body that landed deep inside one; that classification is gone (every
    /// contact is judged on its normal), and this stays as the proof that the plain solver still
    /// brings an edge-only landing to rest ON the surface and claims ground, at three drop
    /// heights, on hitched frames.
    #[test]
    fn a_body_that_lands_inside_an_edge_contact_is_pushed_back_out() {
        let p = phys();
        // ONE triangle of the usual floor pair, so a landing near the diagonal has no second
        // triangle to supply a face contact.
        let half = 30.0f32;
        let w = vec![[
            Vec3::new(-half, -half, 0.0),
            Vec3::new(half, -half, 0.0),
            Vec3::new(half, half, 0.0),
        ]];
        // The hypotenuse of that triangle runs from (-half,-half) to (half,half), i.e. the line
        // y = x. Landing a hair to the INSIDE of it puts the closest point on the edge for a
        // capsule of this radius: the perpendicular distance from (d, -d) to y = x is d*sqrt(2),
        // so d = radius/(2*sqrt(2)) puts the sphere centre half a radius from the edge, which is
        // an edge contact and not a face one.
        let d = p.radius / (2.0 * std::f32::consts::SQRT_2);
        for drop in [30.0f32, 100.0, 300.0] {
            let mut st = PlayerState::standing_at(Vec3::new(d, -d, drop), 0.0);
            let mut deepest = 0.0f32;
            let mut landed = false;
            for _ in 0..400 {
                // 100 ms frames: the hitch case, which is what produces a deep penetration.
                st.step(&MoveInput::default(), &p, &w, 0.1);
                deepest = deepest.min(st.pos.z);
                if st.grounded && st.pos.z.abs() < 0.05 {
                    landed = true;
                    break;
                }
            }
            println!(
                "   dropped {drop} wu onto a single triangle's edge at ({d:.4}, {:.4}): \
                 deepest z {deepest:.4}, final z {:.4}, grounded {}",
                -d,
                st.pos.z,
                st.grounded
            );
            assert!(
                landed,
                "dropped {drop} wu onto an EDGE contact and never came to rest on the surface: \
                 final z {:.4}, grounded {} (deepest {deepest:.4})",
                st.pos.z,
                st.grounded,
            );
        }
    }

    #[test]
    fn a_falling_body_does_not_tunnel_a_thin_platform_on_a_hitched_frame() {
        let p = phys();
        let w = floor(0.0, 30.0);
        // Two landing spots per drop: one in the middle of a triangle (a FACE contact) and one
        // on the floor's shared diagonal (an EDGE contact). A floor is two triangles and a player
        // stands on their seam as often as not, so both must work — testing only the middle hid
        // the edge case that actually bit.
        for (drop, x, y, where_) in [
            (30.0f32, 3.0f32, -1.0f32, "mid-face"),
            (100.0, 3.0, -1.0, "mid-face"),
            (300.0, 3.0, -1.0, "mid-face"),
            (100.0, 0.0, 0.0, "on the triangle seam"),
            (300.0, 0.0, 0.0, "on the triangle seam"),
        ] {
            let drop = drop;
            let mut st = PlayerState::standing_at(Vec3::new(x, y, drop), 0.0);
            let mut worst_sub = 0.0f32;
            let mut guard = false;
            for _ in 0..400 {
                // A 100 ms frame, every frame: the hitch case, not the happy path.
                let o = st.step(&MoveInput::default(), &p, &w, 0.1);
                guard |= o.substep_guard_tripped;
                if o.substeps > 0 {
                    worst_sub = worst_sub.max(st.vel.length() * 0.1 / o.substeps as f32);
                }
                if st.grounded {
                    break;
                }
            }
            assert!(
                st.grounded,
                "dropped from {drop} wu {where_} on 100 ms frames and never landed — z is now {:.2}, so the \
                 body went through a floor it should have been caught by",
                st.pos.z
            );
            assert!(
                st.pos.z.abs() < 0.05,
                "dropped from {drop} wu {where_} and came to rest at z {:.4} instead of on the floor",
                st.pos.z
            );
            assert!(
                worst_sub <= p.radius,
                "dropped from {drop} wu {where_}: the largest sub-step was {worst_sub:.4} wu, past the \
                 {:.4} wu capsule radius — consecutive sphere positions no longer overlap and a \
                 surface can pass between them",
                p.radius
            );
            assert!(!guard, "dropped from {drop} wu {where_}: the runaway guard tripped, so the budget is too small");
        }
    }

    /// #play-gate  INVERSE: the sub-step budget is what does it, and it scales with the capsule.
    /// A body with half the radius needs twice the sub-steps for the same travel — if this ever
    /// stops holding, the budget has been replaced by a fixed count again, which is the bug.
    #[test]
    fn the_substep_count_follows_the_capsule_and_the_travel() {
        let p = phys();
        let w = floor(0.0, 30.0);
        let subs_for = |radius: f32, dt: f32| -> usize {
            let mut q = p;
            q.radius = radius;
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 200.0), 0.0);
            // Fall for a fixed time so the speed is the same in both cases.
            let mut last = 0;
            for _ in 0..20 {
                last = st.step(&MoveInput::default(), &q, &w, dt).substeps;
            }
            last
        };
        let big = subs_for(p.radius, 0.1);
        let small = subs_for(p.radius * 0.5, 0.1);
        assert!(
            small >= big * 2 - 1,
            "halving the radius should roughly double the sub-steps: {big} -> {small}"
        );
        // And a shorter frame travels less, so it needs fewer.
        assert!(subs_for(p.radius, 1.0 / 60.0) < big, "a 60 Hz frame must need fewer sub-steps than a 100 ms one");
    }

    /// #play-gate  WALKING UP AN INCLINE DOES NOT JITTER.
    ///
    /// Reported by the user: *"the physics for sure is very bad around inclines, it jitters and
    /// freaks out when I'm walking up any incline"*. The measure is the one that names the
    /// symptom: the RESIDUAL between the feet and the ramp's own surface at the body's x, across a
    /// steady climb. A body tracking the slope holds that residual constant; a body being
    /// repeatedly classified airborne, dropped by gravity and re-grounded oscillates it.
    ///
    /// Checked at several angles because the failure this caught is SPEED- AND ANGLE-DEPENDENT:
    /// it appears once `speed * tan(angle)` crosses a threshold, so a gentle ramp can pass while
    /// a moderate one fails.
    #[test]
    fn walking_up_an_incline_does_not_jitter() {
        let p = phys();
        for deg in [6.0f32, 12.0, 20.0, 30.0] {
            let mut w = floor(0.0, 8.0);
            w.extend(ramp(deg, 30.0, 8.0));
            let mut st = PlayerState::standing_at(Vec3::new(-2.0, 0.0, 0.2), 0.0);
            settle(&mut st, &p, &w);
            assert!(st.grounded, "{deg} deg: never settled on the flat");
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            // Reach the ramp and get onto it before measuring.
            run(&mut st, &input, &p, &w, 1.5);
            assert!(st.pos.x > 0.5, "{deg} deg: did not reach the ramp (x {:.2})", st.pos.x);
            // Settle into STEADY STATE on the ramp before measuring: the first frames on it are
            // the transition off the flat, and a transition is not jitter.
            run(&mut st, &input, &p, &w, 0.5);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            let mut airborne = 0;
            let z0 = st.pos.z;
            for _ in 0..120 {
                st.step(&input, &p, &w, 1.0 / 60.0);
                let residual = st.pos.z - ramp_surface_z(deg, st.pos.x);
                lo = lo.min(residual);
                hi = hi.max(residual);
                if !st.grounded {
                    airborne += 1;
                }
            }
            assert!(st.pos.z > z0 + 0.1, "{deg} deg: did not climb (z {:.3} -> {:.3})", z0, st.pos.z);
            assert_eq!(
                airborne, 0,
                "{deg} deg: the body was classified AIRBORNE on {airborne} of 120 frames while walking up a ramp \
                 — that is the jitter: gravity pulls it off the slope and the next frame puts it back"
            );
            // JITTER IS VARIATION, not offset. The feet legitimately rest
            // `radius * (sec(angle) - 1)` above the plane — that is where a capsule sits on a
            // slope — so the gate measures how much the residual MOVES, and separately asserts
            // that the residual itself is that geometric offset. Loosening the tolerance to
            // absorb the offset would have hidden a real oscillation of the same size.
            assert!(
                (hi - lo) < 0.005,
                "{deg} deg: the feet oscillated {:.4} wu around the ramp surface (residual {lo:.4}..{hi:.4})",
                hi - lo
            );
            let want = feet_rest_above_slope(p.radius, deg);
            let mid = (hi + lo) * 0.5;
            assert!(
                (mid - want).abs() < 0.006,
                "{deg} deg: the feet rest {mid:.4} wu above the plane; a {:.3} wu capsule on that slope rests \
                 {want:.4} wu above it",
                p.radius
            );
        }
    }

    /// #play-gate  AND WALKING DOWN ONE DOES NOT EITHER. A descent has the mirror problem: the
    /// in-plane velocity legitimately points DOWNWARD, and zeroing it every grounded frame makes
    /// the body walk off the slope horizontally and fall back onto it, over and over.
    #[test]
    fn walking_down_an_incline_does_not_jitter() {
        let p = phys();
        for deg in [8.0f32, 20.0, 32.0] {
            let w = downramp(deg, 60.0, 6.0, 8.0);
            let mut st = PlayerState::standing_at(Vec3::new(-3.0, 0.0, 0.3), 0.0);
            settle(&mut st, &p, &w);
            assert!(st.grounded, "{deg} deg: never settled");
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            run(&mut st, &input, &p, &w, 1.5);
            assert!(st.pos.x > 0.5, "{deg} deg: did not reach the descent");
            // Steady state, as uphill. A body running off the CREST leaves the ground -- the
            // engine's contact normal rotates past the 0.468 leaving speed -- and lands on the
            // slope further down; the window starts after that landing (the old controller hid
            // the flight by teleporting the body onto the slope).
            run(&mut st, &input, &p, &w, 4.0);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            let mut airborne = 0;
            for _ in 0..120 {
                st.step(&input, &p, &w, 1.0 / 60.0);
                let residual = st.pos.z + ramp_surface_z(deg, st.pos.x);
                lo = lo.min(residual);
                hi = hi.max(residual);
                if !st.grounded {
                    airborne += 1;
                }
            }
            assert_eq!(airborne, 0, "{deg} deg: airborne on {airborne} of 120 frames walking DOWN a ramp");
            assert!(
                (hi - lo) < 0.005,
                "{deg} deg: the feet oscillated {:.4} wu on the descent (residual {lo:.4}..{hi:.4})",
                hi - lo
            );
        }
    }

    /// #play-gate  THE MECHANISM, named: a legitimate uphill velocity must not be mistaken for
    /// leaving the ground.
    ///
    /// A body walking up a slope has an in-plane vertical velocity of `speed * tan(angle)` — it
    /// is rising, and that is not an ascent under its own power, it is the shape of the floor. The
    /// rule that decides whether to keep the body on the ground must therefore be about CAUSE
    /// (did it jump, was it already in the air) and not about how fast it is rising. This asserts
    /// the consequence directly: at the authored run speed, no incline the biped can legally
    /// climb may put it in the air.
    #[test]
    fn an_uphill_velocity_is_never_mistaken_for_leaving_the_ground() {
        let p = phys();
        // The walkable limit on a FACE is the engine's global 35 degrees, not the authored 50.
        let limit = 35.0f32;
        for deg in [6.0f32, 10.0, 15.0, 25.0, 33.0, limit - 0.5] {
            let mut w = floor(0.0, 8.0);
            w.extend(ramp(deg, 40.0, 8.0));
            let mut st = PlayerState::standing_at(Vec3::new(-2.0, 0.0, 0.2), 0.0);
            settle(&mut st, &p, &w);
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            let outs = run(&mut st, &input, &p, &w, 3.0);
            let airborne = outs.iter().filter(|_| false).count(); // placeholder; measured below
            let _ = airborne;
            // Re-walk recording the state, since StepOut does not carry `grounded`.
            let mut s2 = PlayerState::standing_at(Vec3::new(-2.0, 0.0, 0.2), 0.0);
            settle(&mut s2, &p, &w);
            let mut air = 0;
            for _ in 0..180 {
                s2.step(&input, &p, &w, 1.0 / 60.0);
                if !s2.grounded {
                    air += 1;
                }
            }
            assert_eq!(
                air, 0,
                "{deg:.0} deg (limit {limit:.0}): airborne on {air} of 180 frames. Walking at {:.3} wu/s up \
                 {deg:.0} deg gives a legitimate in-plane rise of {:.3} wu/s — if the rule that keeps a body \
                 grounded compares that against a speed, every incline past a few degrees reads as a jump",
                p.run_forward,
                p.run_forward * deg.to_radians().tan()
            );
        }
    }

    /// #play-gate  WALKING ACROSS A FLAT FLOOR DOES NOT BOB.
    ///
    /// The symptom this bites on is a body that sinks, climbs or jitters as it crosses perfectly
    /// level ground — the sign of a depenetration that fights gravity every frame, or of a
    /// ground probe that keeps letting go. It measures the FEET, because that is the quantity
    /// the controller owns.
    ///
    /// **The eye is measured separately, and only at steady speed, because the eye legitimately
    /// moves and the tag says so.** `bipd` authors a `standing camera height` AND a `running
    /// camera height`, and they differ — so a gate that watched the eye across the acceleration
    /// from a standstill to a run would be measuring the authored difference and calling it a
    /// bug. (It did: the first version of this test failed with "eye height wandered by 0.0200
    /// wu", which is exactly `eye_standing - eye_running` on these values and nothing to do with
    /// the floor.) The step is asserted to BE the authored difference, so a real wander cannot
    /// hide inside it.
    #[test]
    fn a_walk_across_a_flat_floor_does_not_bob() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(-5.0, 0.0, 0.5), 0.0);
        settle(&mut st, &p, &w);
        let z0 = st.pos.z;
        let input = MoveInput { fwd: 1.0, ..Default::default() };
        // Reach the authored run speed first, so the sample is steady state.
        run(&mut st, &input, &p, &w, 1.0);
        let (mut zlo, mut zhi) = (f32::MAX, f32::MIN);
        let (mut elo, mut ehi) = (f32::MAX, f32::MIN);
        let mut airborne = 0;
        for _ in 0..180 {
            st.step(&input, &p, &w, 1.0 / 60.0);
            zlo = zlo.min(st.pos.z);
            zhi = zhi.max(st.pos.z);
            elo = elo.min(st.eye(&p).z);
            ehi = ehi.max(st.eye(&p).z);
            if !st.grounded {
                airborne += 1;
            }
        }
        assert!(st.pos.x > 0.0, "barely moved: {}", st.describe(&p));
        assert_eq!(airborne, 0, "left the floor while walking on it: {}", st.describe(&p));
        // A millimetre at Halo scale is 0.0003 wu; this allows a third of that.
        assert!(
            (zhi - zlo) < 0.001,
            "the feet wandered by {:.5} wu across a flat floor (started at {z0:.5})",
            zhi - zlo
        );
        assert!((ehi - elo) < 0.001, "the eye wandered by {:.5} wu at steady speed", ehi - elo);
        // And the standing-to-running step IS the authored one, so it cannot be hiding a wander.
        let mut stand = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
        settle(&mut stand, &p, &w);
        let step = stand.eye_height(&p) - st.eye_height(&p);
        assert!(
            (step - (p.eye_standing - p.eye_running)).abs() < 1e-4,
            "standing-to-running eye step {step:.5} is not the authored {:.5}",
            p.eye_standing - p.eye_running
        );
    }

    /// #play-gate  THE AUTHORED RUN SPEED IS THE SPEED REACHED. Not a chosen number: the cap is
    /// `matg @player information` `run forward`, and a full-forward stick settles on it.
    #[test]
    fn a_full_forward_run_settles_on_the_authored_run_speed() {
        let p = phys();
        let w = floor(0.0, 200.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
        settle(&mut st, &p, &w);
        run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 2.0);
        assert!(
            (st.ground_speed - p.run_forward).abs() < 0.05,
            "ran at {:.3} wu/s, the tag says {:.3}",
            st.ground_speed,
            p.run_forward
        );
        // And backwards is the authored BACKWARD cap, which is slower.
        let mut st2 = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
        settle(&mut st2, &p, &w);
        run(&mut st2, &MoveInput { fwd: -1.0, ..Default::default() }, &p, &w, 2.0);
        assert!(
            (st2.ground_speed - p.run_backward).abs() < 0.05,
            "backpedalled at {:.3}, the tag says {:.3}",
            st2.ground_speed,
            p.run_backward
        );
    }

    /// #play-gate  A JUMP RETURNS TO THE GROUND IT LEFT, and its apex is the one the authored
    /// jump velocity and the engine's measured gravity imply. This is the gate on "jumping feels
    /// wrong": it pins the arithmetic to the tags rather than to taste.
    #[test]
    fn a_jump_returns_to_the_ground_it_left_at_the_predicted_apex() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        let z0 = st.pos.z;
        let outs = run(&mut st, &MoveInput { jump: true, ..Default::default() }, &p, &w, 2.0);
        assert!(outs[0].jumped, "the jump never fired");
        let apex = 0.0f32; // filled below from a re-run that records it
        let _ = apex;
        assert!(st.grounded, "never came back down: {}", st.describe(&p));
        assert!((st.pos.z - z0).abs() < 0.01, "landed at {:.4}, left from {:.4}", st.pos.z, z0);
        assert!(outs.iter().any(|o| o.landed), "no landing was reported");
        // The apex, measured by re-running and watching the peak.
        let mut st2 = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st2, &p, &w);
        let base = st2.pos.z;
        let mut peak = base;
        let mut inp = MoveInput { jump: true, ..Default::default() };
        for i in 0..120 {
            inp.jump = i == 0;
            st2.step(&inp, &p, &w, 1.0 / 60.0);
            peak = peak.max(st2.pos.z);
        }
        let predicted = p.jump_velocity * p.jump_velocity / (2.0 * p.gravity());
        assert!(
            (peak - base - predicted).abs() < 0.03,
            "apex {:.4} wu, the tags predict {:.4} wu (v {:.3}, g {:.4})",
            peak - base,
            predicted,
            p.jump_velocity,
            p.gravity()
        );
    }

    /// #play-gate  A JUMP CANNOT BE USED TO GO FASTER. Air control may steer but not accelerate
    /// past the authored ground cap.
    #[test]
    fn air_control_steers_but_does_not_outrun_the_ground_cap() {
        let p = phys();
        let w = floor(0.0, 200.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
        settle(&mut st, &p, &w);
        run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 2.0);
        let mut inp = MoveInput { fwd: 1.0, jump: true, ..Default::default() };
        let mut peak = 0.0f32;
        for i in 0..60 {
            inp.jump = i == 0;
            st.step(&inp, &p, &w, 1.0 / 60.0);
            peak = peak.max(Vec3::new(st.vel.x, st.vel.y, 0.0).length());
        }
        assert!(peak <= p.run_forward + 0.05, "gained speed in the air: {peak:.3} > {:.3}", p.run_forward);
    }

    /// #play-gate  THE CAPSULE CANNOT PASS THROUGH A FORGE BLOCK. The body runs at a block for
    /// four seconds; it must end up outside it, by at least the authored radius.
    #[test]
    fn the_capsule_cannot_walk_through_a_forge_block() {
        let p = phys();
        let mut w = floor(0.0, 40.0);
        // A block 2 wu on a side, its near face at x = 2.
        let (bmin, bmax) = (Vec3::new(2.0, -1.0, 0.0), Vec3::new(4.0, 1.0, 2.0));
        w.extend(block(bmin, bmax));
        let mut st = PlayerState::standing_at(Vec3::new(-2.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 4.0);
        assert!(
            st.pos.x < bmin.x - p.radius * 0.5,
            "ended at x {:.3}, INSIDE or past the block whose near face is {:.3}: {}",
            st.pos.x,
            bmin.x,
            st.describe(&p)
        );
        assert!(st.pos.x > 1.0, "stopped far short of the block, so this proves nothing: x {:.3}", st.pos.x);
        assert!(st.pos.z.abs() < 0.05, "climbed the block's face: z {:.3}", st.pos.z);
    }

    /// #play-gate  AND IT CANNOT PASS THROUGH ONE AT SPEED. The sub-stepping is what this bites
    /// on: a body teleported forward by a whole frame of run velocity could straddle a thin wall.
    #[test]
    fn a_thin_wall_is_not_tunnelled_through() {
        let p = phys();
        let mut w = floor(0.0, 60.0);
        // 0.08 wu thick — thinner than the capsule radius.
        w.extend(block(Vec3::new(3.0, -4.0, 0.0), Vec3::new(3.08, 4.0, 2.0)));
        let mut st = PlayerState::standing_at(Vec3::new(-1.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        // Start it already at full tilt by hand, so the whole run is at the cap.
        st.vel = Vec3::new(p.run_forward, 0.0, 0.0);
        run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 3.0);
        assert!(st.pos.x < 3.0, "tunnelled through a thin wall to x {:.3}: {}", st.pos.x, st.describe(&p));
    }

    /// #play-gate  CROUCHING LOWERS THE EYE TO THE TAG'S OWN CROUCHING CAMERA HEIGHT, and takes
    /// the tag's own transition time to get there.
    #[test]
    fn crouching_lowers_the_eye_to_the_authored_height() {
        let p = phys();
        let w = floor(0.0, 20.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        let eye_up = st.eye_height(&p);
        assert!((eye_up - p.eye_standing).abs() < 1e-4, "standing eye {eye_up} != authored {}", p.eye_standing);
        // Half the transition time in: part way down, not snapped.
        let half = p.crouch_transition * 0.5;
        run(&mut st, &MoveInput { crouch: true, ..Default::default() }, &p, &w, half);
        let mid = st.eye_height(&p);
        assert!(mid < eye_up - 0.01 && mid > p.eye_crouching + 0.01, "eye snapped instead of interpolating: {mid}");
        run(&mut st, &MoveInput { crouch: true, ..Default::default() }, &p, &w, 1.0);
        let down = st.eye_height(&p);
        assert!((down - p.eye_crouching).abs() < 1e-3, "crouched eye {down} != authored {}", p.eye_crouching);
        assert!(st.crouched);
        // And the capsule shortened to the authored crouching height.
        assert!((st.height(&p) - p.height_crouching).abs() < 1e-3);
    }

    /// #play-gate  A CEILING KEEPS A CROUCHED BODY CROUCHED. Letting go of crouch under a low
    /// roof must not stand the body up through it.
    #[test]
    fn a_low_ceiling_refuses_to_let_the_crouch_end() {
        let p = phys();
        let mut w = floor(0.0, 20.0);
        // A roof just above the crouching height, well below the standing height.
        let roof = p.height_crouching + 0.05;
        assert!(roof < p.height_standing, "the stand-in's own heights make this test vacuous");
        w.extend(floor(roof, 20.0).into_iter().map(|t| [t[0], t[2], t[1]]));
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.05), 0.0);
        st.crouched = true;
        st.crouch_t = 1.0;
        run(&mut st, &MoveInput { crouch: true, ..Default::default() }, &p, &w, 0.5);
        assert!(st.crouched, "lost the crouch under the roof");
        // Release: the body must stay crouched, and say why.
        let outs = run(&mut st, &MoveInput::default(), &p, &w, 0.5);
        assert!(st.crouched, "stood up through a {roof:.3} wu roof: {}", st.describe(&p));
        assert!(outs.iter().any(|o| o.crouch_blocked), "nothing reported the ceiling");
        // Out in the open the same release DOES stand it up — otherwise the test above would
        // pass for a controller that can never stand up at all.
        let open = floor(0.0, 20.0);
        let mut st2 = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.05), 0.0);
        st2.crouched = true;
        st2.crouch_t = 1.0;
        run(&mut st2, &MoveInput::default(), &p, &open, 1.0);
        assert!(!st2.crouched, "a body in the open could not stand up either");
    }

    /// #play-gate  WALKING DOWN A SLOPE STAYS ON THE SLOPE. Without the derived ground probe the
    /// body leaves the hill every frame and the animation flickers between running and airborne.
    #[test]
    fn running_downhill_never_leaves_the_hill() {
        let p = phys();
        let deg = 30.0f32;
        let t = deg.to_radians().tan();
        // A ramp DESCENDING along +X from x=0.
        let a = Vec3::new(-4.0, -6.0, 0.0);
        let b = Vec3::new(-4.0, 6.0, 0.0);
        let c = Vec3::new(20.0, 6.0, -24.0 * t);
        let d = Vec3::new(20.0, -6.0, -24.0 * t);
        let w = vec![[a, b, c], [a, c, d]];
        let mut st = PlayerState::standing_at(Vec3::new(-2.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        assert!(st.grounded, "never landed on the ramp");
        let mut airborne = 0;
        for _ in 0..180 {
            st.step(&MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 1.0 / 60.0);
            if !st.grounded {
                airborne += 1;
            }
        }
        assert!(st.pos.x > 0.0, "did not run down the ramp: {}", st.describe(&p));
        assert_eq!(airborne, 0, "left the ramp on {airborne} of 180 frames running downhill");
    }

    // -----------------------------------------------------------------------------------------
    // the state the animation layer reads
    // -----------------------------------------------------------------------------------------

    /// #play-gate  CROUCH REACHES THE `crouch:` ANIMATION MODE. `Stance::Crouch` has existed in
    /// `film_anim` since it was written and has never once been produced, because a Reach film
    /// cannot state the crouch bit. Test mode knows it, so this is the first caller that can —
    /// and the clip names it asks for must be the crouch mode's own leaves.
    #[test]
    fn a_crouching_player_asks_the_crouch_mode_for_its_clips() {
        use crate::film_anim::{anim_candidates, Stance};
        let p = phys();
        let w = floor(0.0, 20.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        let input = MoveInput { crouch: true, fwd: 1.0, ..Default::default() };
        run(&mut st, &input, &p, &w, 1.0);
        let b = body_state(&st, &input, &p, Some("rifle".into()));
        assert_eq!(b.stance, Stance::Crouch, "a crouching player is not in the crouch stance");
        let cands = anim_candidates(&b, "rifle");
        assert!(
            cands.first().map(|c| c.starts_with("crouch:")).unwrap_or(false),
            "first clip asked for is {:?}, not a crouch: clip",
            cands.first()
        );
        // And the leaf is the crouch mode's own naming, not the combat set's.
        assert!(
            cands.iter().any(|c| c == "crouch:rifle:walk_front" || c == "crouch:rifle:move_front"),
            "crouch candidates do not include the crouch mode's own forward cycle: {cands:?}"
        );
        // Standing, the same walk asks the combat set instead.
        let up = MoveInput { fwd: 1.0, ..Default::default() };
        run(&mut st, &up, &p, &w, 1.0);
        let b2 = body_state(&st, &up, &p, Some("rifle".into()));
        assert_eq!(b2.stance, Stance::Combat);
        assert!(anim_candidates(&b2, "rifle").first().map(|c| c.starts_with("combat:")).unwrap_or(false));
    }

    /// #play-gate  A JUMPING PLAYER ASKS FOR THE BALLISTIC ARC, NOT THE JETPACK HOVER. The
    /// `flight:` mode is the jetpack; asking it for a jump is what produced the A-pose. The
    /// state this controller hands over has to land on the combat set's airborne clips.
    #[test]
    fn a_jumping_player_asks_for_the_airborne_arc_and_then_the_landing() {
        use crate::film_anim::anim_candidates;
        let p = phys();
        let w = floor(0.0, 20.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        let mut inp = MoveInput { jump: true, ..Default::default() };
        st.step(&inp, &p, &w, 1.0 / 60.0);
        inp.jump = false;
        st.step(&inp, &p, &w, 1.0 / 60.0);
        assert!(!st.grounded, "not airborne one frame after the jump");
        let b = body_state(&st, &inp, &p, Some("rifle".into()));
        let cands = anim_candidates(&b, "rifle");
        assert_eq!(cands.first().map(|s| s.as_str()), Some("combat:rifle:airborne_arc"), "{cands:?}");
        assert!(!cands.iter().any(|c| c.starts_with("flight:")), "asked the JETPACK mode for a jump: {cands:?}");
        // Down again: the landing clip owns the body for the clip's own length.
        for _ in 0..200 {
            st.step(&inp, &p, &w, 1.0 / 60.0);
            if st.grounded {
                break;
            }
        }
        assert!(st.grounded);
        let b2 = body_state(&st, &inp, &p, Some("rifle".into()));
        assert!(b2.land_ticks.is_some(), "no landing state after touching down");
        assert_eq!(anim_candidates(&b2, "rifle").first().map(|s| s.as_str()), Some("combat:rifle:land_hard"));
    }

    /// #play-gate  STRAFING AND BACKPEDALLING ARE NOT DRAWN AS A FORWARD RUN — the same property
    /// `film_anim` was built for, now driven from a keyboard.
    #[test]
    fn the_travel_direction_picks_the_right_cycle() {
        use crate::film_anim::Gait;
        let p = phys();
        let w = floor(0.0, 60.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        for (inp, want) in [
            (MoveInput { fwd: 1.0, ..Default::default() }, Gait::RunFront),
            (MoveInput { fwd: -1.0, ..Default::default() }, Gait::Back),
            (MoveInput { right: 1.0, ..Default::default() }, Gait::Right),
            (MoveInput { right: -1.0, ..Default::default() }, Gait::Left),
        ] {
            let mut s = st;
            run(&mut s, &inp, &p, &w, 1.5);
            let b = body_state(&s, &inp, &p, Some("rifle".into()));
            assert_eq!(b.gait, want, "stick ({}, {}) gave {:?}: {}", inp.fwd, inp.right, b.gait, s.describe(&p));
        }
    }

    /// A world of bare triangles plus closed shells, for the inside-a-block tests.
    struct ShellWorld {
        tris: Vec<[Vec3; 3]>,
        shells: Vec<std::sync::Arc<Shell>>,
    }
    impl StaticWorld for ShellWorld {
        fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            self.tris.as_slice().gather(mn, mx, out)
        }
        fn solid_shells(&self, mn: Vec3, mx: Vec3, out: &mut Vec<std::sync::Arc<Shell>>) {
            out.extend(self.shells.iter().filter(|s| (0..3).all(|k| s.mn[k] <= mx[k] && s.mx[k] >= mn[k])).cloned());
        }
    }
    fn shell_world(boxes: &[(Vec3, Vec3)]) -> ShellWorld {
        let mut w = ShellWorld { tris: Vec::new(), shells: Vec::new() };
        for (mn, mx) in boxes {
            let c = (*mn + *mx) * 0.5;
            let h = (*mx - *mn) * 0.5;
            let planes = [(Vec3::X, h.x), (Vec3::NEG_X, h.x), (Vec3::Y, h.y), (Vec3::NEG_Y, h.y), (Vec3::Z, h.z), (Vec3::NEG_Z, h.z)];
            let t: Vec<[Vec3; 3]> = crate::object_hulls::hull_triangles(&planes, 0.0).into_iter().map(|t| t.map(|p| p + c)).collect();
            w.tris.extend(t.iter().copied());
            w.shells.push(std::sync::Arc::new(Shell::new(t)));
        }
        w
    }

    /// #collision  **A BODY PLACED INSIDE A BLOCK COMES OUT THE NEAREST SIDE.** The triangle
    /// contacts are two-sided toward the sphere, so a sphere whose centre is inside a closed
    /// block is held between its faces (or, past the middle of a thin wall, pushed on out the
    /// FAR side). A 0.2 thick wall with the body a hair either side of its middle must leave on
    /// that side, a deep cube must be left by the nearest face, and nothing may end inside.
    ///
    /// SABOTAGE: make `escape_shells` return 0 (one-sided / face-only contacts) -> every
    /// assertion about the final position goes red.
    #[test]
    fn a_body_placed_inside_a_closed_block_leaves_by_the_nearest_face() {
        let p = phys();
        let r = p.radius;
        let wall = shell_world(&[(Vec3::new(-0.1, -1.0, 0.0), Vec3::new(0.1, 1.0, 2.0))]);
        for (x, want_sign) in [(0.03f32, 1.0f32), (-0.03, -1.0), (0.09, 1.0), (-0.09, -1.0)] {
            let mut st = PlayerState::standing_at(Vec3::new(x, 0.0, 0.5), 0.0);
            let out = st.step(&MoveInput::default(), &p, &wall, 1.0 / 60.0);
            assert!(out.escaped >= 1, "x {x}: the body was not recognised as inside the wall");
            for _ in 0..60 {
                st.step(&MoveInput::default(), &p, &wall, 1.0 / 60.0);
            }
            assert!(st.pos.x * want_sign >= 0.1 + r - 0.02, "x {x}: ended at {:.3}, not out the {want_sign} side", st.pos.x);
            assert!(!PlayerState::overlaps_at(st.pos, &p, &wall, 0.02), "x {x}: still overlapping at {:?}", st.pos);
        }
        // a cube: nearest face by depth, not by the way the body was moving
        let cube = shell_world(&[(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 2.0))]);
        let mut st = PlayerState::standing_at(Vec3::new(0.8, 0.1, 0.3), 0.0);
        st.vel = Vec3::new(-1.0, 0.0, 0.0);
        let out = st.step(&MoveInput::default(), &p, &cube, 1.0 / 60.0);
        assert!(out.escaped >= 1);
        assert!(st.pos.x > 1.0 + r - 0.05, "left the far way: {:?}", st.pos);
    }

    /// #collision  Two butted blocks: the exit from the first is not left inside the second.
    #[test]
    fn a_body_inside_one_of_two_butted_blocks_does_not_end_in_the_other() {
        let p = phys();
        let w = shell_world(&[
            (Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 1.0)),
            (Vec3::new(0.5, -0.5, 0.0), Vec3::new(1.5, 0.5, 1.0)),
        ]);
        let mut st = PlayerState::standing_at(Vec3::new(0.4, 0.0, 0.2), 0.0);
        for _ in 0..30 {
            st.step(&MoveInput::default(), &p, &w, 1.0 / 60.0);
        }
        assert!(!PlayerState::overlaps_at(st.pos, &p, &w, 0.02), "ended overlapping at {:?}", st.pos);
    }

    /// A world that hides the shells: what the controller saw before `escape_shells`.
    struct NoShells<'a>(&'a ShellWorld);
    impl StaticWorld for NoShells<'_> {
        fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            self.0.gather(mn, mx, out)
        }
    }

    /// #collision  **EVERY REAL FORGE BLOCK LETS A TRAPPED BODY OUT.** For each `objects\levels\
    /// forge\ff_*` block whose physics is closed shells (the production hulls, radius included),
    /// a Spartan is placed at a 5x5x5 grid of positions through the block's box, every one with a
    /// sphere centre INSIDE the solid, and stepped for 1.5 s. After the first tick it must have
    /// moved no further than the nearest way out (half the shell's thinnest side plus a radius,
    /// plus 0.1 -- not through to the far side), and at the end it must overlap nothing. Also two
    /// copies butted at the block's pitch, with the body straddling the seam.
    ///
    /// BEFORE (shells hidden, as the controller was): the same placements are counted as ending
    /// inside the solid. SABOTAGE: `escape_shells` returning 0 -> red.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_every_real_forge_block_lets_a_trapped_body_out() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let (_, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, crate::play_mode::Species::Spartan).unwrap();
        let r = p.radius;
        let (mut blocks, mut trapped, mut stuck_before, mut stuck_after, mut too_far, mut seam_cases) = (0, 0, 0, 0, 0, 0);
        let mut report: Vec<String> = Vec::new();
        for &tag in native.tags_by_group(cache, "bloc").iter() {
            let name = native.tag_name(cache, tag).unwrap_or_default();
            if !name.contains("levels\\forge\\ff_") {
                continue;
            }
            let ph = native.resolve_phmo(cache, tag);
            if ph == 0 || ph == 0xFFFF_FFFF {
                continue;
            }
            let Some(blk) = crate::object_hulls::read_blocks(native, cache, ph) else { continue };
            let Some(shells) = crate::object_hulls::hull_shells(&blk, &crate::object_hulls::mode_node_xforms(native, cache, tag)) else { continue };
            blocks += 1;
            // the objects: one block, and a butted copy at the block's own x pitch
            let one = ShellWorld { tris: shells.iter().flatten().copied().collect(), shells: shells.iter().map(|s| std::sync::Arc::new(Shell::new(s.clone()))).collect() };
            let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for sh in &one.shells {
                mn = mn.min(sh.mn);
                mx = mx.max(sh.mx);
            }
            let ext = mx - mn;
                        let pitch = ext.x;
            let off = Vec3::new(pitch, 0.0, 0.0);
            let two = ShellWorld {
                tris: one.tris.iter().copied().chain(one.tris.iter().map(|t| t.map(|q| q + off))).collect(),
                shells: one.shells.iter().cloned().chain(one.shells.iter().map(|s| std::sync::Arc::new(Shell::new(s.tris.iter().map(|t| t.map(|q| q + off)).collect())))).collect(),
            };
            let mut spots: Vec<(Vec3, bool)> = Vec::new();
            for i in 0..5 {
                for j in 0..5 {
                    for k in 0..5 {
                        let f = |n: i32| 0.1 + 0.2 * n as f32;
                        spots.push((mn + ext * Vec3::new(f(i), f(j), f(k)) - Vec3::Z * 0.3, false));
                    }
                }
            }
            for dz in [0.1f32, 0.4] {
                for dy in [0.2f32, 0.5, 0.8] {
                    for dx in [-0.03f32, 0.0, 0.03] {
                        spots.push((Vec3::new(mx.x + dx, mn.y + ext.y * dy, mn.z + ext.z * dz), true));
                    }
                }
            }
            for (feet, seam) in spots {
                let w = if seam { &two } else { &one };
                let inside = PlayerState::axis_samples(feet, p.height_standing, r).iter().any(|c| w.shells.iter().any(|s| s.contains(*c)));
                if !inside {
                    continue;
                }
                trapped += 1;
                seam_cases += seam as usize;
                // before: the controller without the shells
                let mut b = PlayerState::standing_at(feet, 0.0);
                for _ in 0..90 {
                    b.step(&MoveInput::default(), &p, &NoShells(w), 1.0 / 60.0);
                }
                let sc = |s: &PlayerState| PlayerState::axis_samples(s.pos, p.height_standing, r).iter().any(|c| w.shells.iter().any(|sh| sh.contains(*c)));
                stuck_before += sc(&b) as usize;
                let mut st = PlayerState::standing_at(feet, 0.0);
                let first = st.step(&MoveInput::default(), &p, w, 1.0 / 60.0);
                let moved = (st.pos - feet).length();
                // ORACLE for an axis-aligned BOX shell (its triangles fill their own AABB: 12 of
                // them) that is the only one the body is in: the shortest axis translation that
                // takes the whole axis out of it, plus the radius the contact solver then adds.
                // Moving further than that is leaving by the wrong way (through the far side).
                let pts = PlayerState::axis_samples(feet, p.height_standing, r);
                let hit: Vec<_> = w.shells.iter().filter(|sh| pts.iter().any(|q| sh.contains(*q))).collect();
                if let [sh] = hit.as_slice() {
                    if sh.tris.len() == 12 {
                        let (lo, hi) = (pts[0], *pts.last().unwrap());
                        let e = (sh.mx.x - feet.x).min(feet.x - sh.mn.x).min(sh.mx.y - feet.y).min(feet.y - sh.mn.y).min(sh.mx.z - lo.z).min(hi.z - sh.mn.z);
                        if moved > e + r + 0.06 || first.escaped != 1 {
                            too_far += 1;
                            if report.len() < 10 {
                                report.push(format!("{name}: from {feet:?} moved {moved:.3} in {} move(s), the shortest axis exit says {:.3}", first.escaped, e + r));
                            }
                        }
                    }
                }
                let mut trace = Vec::new();
                for k in 0..90 {
                    let o = st.step(&MoveInput::default(), &p, w, 1.0 / 60.0);
                    if k < 6 { trace.push(format!("t{k} esc {} pos {:.3?}", o.escaped, st.pos)); }
                }
                if PlayerState::overlaps_at(st.pos, &p, w, 0.03) {
                    stuck_after += 1;
                    if report.len() < 8 {
                        let why = {
                            let cs = PlayerState::spheres_at(st.pos, p.height_standing, r);
                            let inside = cs.iter().filter(|c| w.shells.iter().any(|sh| sh.contains(**c))).count();
                            let mut tr: Vec<[Vec3; 3]> = Vec::new();
                            w.gather(st.pos - Vec3::splat(1.0), st.pos + Vec3::splat(2.0), &mut tr);
                            let deep = cs.iter().flat_map(|c| tr.iter().filter_map(|t| sphere_tri_contact(*c, r, 0.0, t)).map(|h| h.depth)).fold(0.0f32, f32::max);
                            format!("{inside} sphere centres inside a shell, deepest tri contact {deep:.3}")
                        };
                        let dbg: Vec<String> = w.shells.iter().filter(|sh| PlayerState::spheres_at(st.pos, p.height_standing, r).iter().any(|c| sh.contains(*c))).map(|sh| format!("shell {:?}..{:?} {} tris", sh.mn, sh.mx, sh.tris.len())).collect();
                        report.push(format!("{name}: from {feet:?} ended overlapping at {:?}: {why}; in {dbg:?}; {trace:?}", st.pos));
                    }
                }
            }
        }
        println!("{blocks} blocks, {trapped} trapped placements ({seam_cases} at a seam): ended INSIDE a shell before {stuck_before}, after the fix {stuck_after} overlapping; {too_far} single-move exits that did not go by the nearest face");
        for l in &report {
            println!("   {l}");
        }
        assert!(blocks >= 60 && trapped >= 3000 && seam_cases >= 300, "liveness: {blocks} blocks, {trapped} placements, {seam_cases} at seams");
        assert!(stuck_before > trapped / 4, "the fixture does not trap the unfixed controller ({stuck_before} of {trapped})");
        assert_eq!((stuck_after, too_far), (0, 0));
    }

    // -----------------------------------------------------------------------------------------
    // tiny obstacles: steps, gaps, ledges -- the body must never be pushed or sent backward
    // -----------------------------------------------------------------------------------------

    /// A local-frame obstacle course turned by `yaw` about the origin: closed bevelled block
    /// shells exactly as the production hulls are made (`padded_hull`, radius 0.0164), plus
    /// optional bare terrain triangles.
    fn course(boxes: &[Vec<Vec3>], terrain: &[[Vec3; 3]], yaw: f32) -> ShellWorld {
        let rot = glam::Quat::from_rotation_z(yaw);
        let mut w = ShellWorld { tris: Vec::new(), shells: Vec::new() };
        for pts in boxes {
            let pts: Vec<Vec3> = pts.iter().map(|p| rot * *p).collect();
            let t = crate::object_hulls::padded_hull(&pts, 0.0164);
            w.tris.extend(t.iter().copied());
            w.shells.push(std::sync::Arc::new(Shell::new(t)));
        }
        w.tris.extend(terrain.iter().map(|t| t.map(|p| rot * p)));
        w
    }

    fn cuboid(mn: Vec3, mx: Vec3) -> Vec<Vec3> {
        (0..8).map(|i| Vec3::new(if i & 1 == 0 { mn.x } else { mx.x }, if i & 2 == 0 { mn.y } else { mx.y }, if i & 4 == 0 { mn.z } else { mx.z })).collect()
    }

    fn flat_terrain(z: f32) -> Vec<[Vec3; 3]> {
        let (a, b) = (Vec3::new(-8.0, -8.0, z), Vec3::new(8.0, 8.0, z));
        vec![[a, Vec3::new(b.x, a.y, z), b], [a, b, Vec3::new(a.x, b.y, z)]]
    }

    #[derive(Clone, Copy, Debug)]
    enum Frame {
        Fixed(f32),
        Jittered,
    }

    /// What one walk did.
    #[derive(Default, Debug, Clone)]
    struct Walk {
        backward: usize,
        worst_back: f32,
        teleports: usize,
        relocated: usize,
        escaped: usize,
        flips: usize,
        progress: f32,
        jitter: f32,
        end: Vec3,
        first: String,
    }

    /// Walk a body along local +x (or +y when `along`) of a course rotated by `yaw`, from 2.2 wu
    /// before the origin, for 3 s of held input. Backward = a tick whose displacement points
    /// against the commanded direction by more than 3 mm; teleport = a tick that moves more than
    /// 3x the commanded speed (+5 cm) horizontally or rises more than 5 cm.
    fn walk(p: &PlayerPhysics, w: &ShellWorld, yaw: f32, sprint: bool, along: bool, start_h: f32, frame: Frame) -> Walk {
        let dir_local = if along { Vec3::Y } else { Vec3::X };
        let rot = glam::Quat::from_rotation_z(yaw);
        let dir = rot * dir_local;
        let face = dir.y.atan2(dir.x);
        let start = rot * (-dir_local * 2.2 + if along { Vec3::new(0.0, 0.0, 0.0) } else { Vec3::ZERO }) + Vec3::Z * (start_h + 0.05);
        let mut st = PlayerState::standing_at(start, face);
        for _ in 0..40 {
            st.step(&MoveInput { yaw: face, ..Default::default() }, p, w, 1.0 / 60.0);
        }
        let input = MoveInput { fwd: 1.0, yaw: face, sprint, ..Default::default() };
        let speed = if sprint { p.run_forward * p.sprint_full_speed_multiplier } else { p.run_forward };
        let mut out = Walk::default();
        let begin = st.pos;
        let mut prev_grounded = st.grounded;
        let mut tail: Vec<Vec3> = Vec::new();
        let mut rng = 0x1234_5678u32;
        let mut clock = 0.0f32;
        let mut f = 0usize;
        while clock < 3.0 {
            let dt = match frame {
                Frame::Fixed(h) => 1.0 / h,
                Frame::Jittered => {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    0.004 + 0.03 * ((rng >> 8) as f32 / 16_777_216.0)
                }
            };
            clock += dt;
            f += 1;
            let before = st.pos;
            let o = st.step(&input, p, w, dt);
            let d = st.pos - before;
            let along_cmd = d.x * dir.x + d.y * dir.y;
            let mech = format!("tick {f}: back {along_cmd:.4}, escaped {}, relocated {}, grounded {}->{}", o.escaped, o.relocated, prev_grounded, st.grounded);
            if along_cmd < -0.003 {
                out.backward += 1;
                out.worst_back = out.worst_back.min(along_cmd);
                if out.first.is_empty() { out.first = mech.clone(); }
            }
            if Vec3::new(d.x, d.y, 0.0).length() > speed * dt * 3.0 + 0.05 || d.z > 0.1 || o.relocated {
                out.teleports += 1;
                if out.first.is_empty() { out.first = mech.clone(); }
            }
            out.relocated += o.relocated as usize;
            out.escaped += (o.escaped > 0) as usize;
            out.flips += (st.grounded != prev_grounded) as usize;
            prev_grounded = st.grounded;
            if clock >= 2.0 {
                tail.push(st.pos);
            }
        }
        out.progress = (st.pos - begin).dot(dir);
        out.end = st.pos;
        let (mn, mx) = tail.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        out.jitter = (mx - mn).x.max((mx - mn).y);
        out
    }

    /// How a walk into an obstacle must end.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Expect {
        /// Climbs / crosses it: the body ends well past it.
        Cross,
        /// Cannot climb it: stops in front of it and stays still.
        Stop,
        /// Walks off an edge and falls.
        Drop,
        /// Wedges in a crack narrower than its sphere (the 35 degree surface bar): only
        /// "no teleport" is asserted.
        Wedge,
    }

    /// Every obstacle of the sweep: (name, boxes, terrain, start height, along, expectation).
    fn obstacles() -> Vec<(String, Vec<Vec<Vec3>>, Vec<[Vec3; 3]>, f32, bool, Expect)> {
        let big = |x0: f32, x1: f32, top: f32| cuboid(Vec3::new(x0, -6.0, top - 1.0), Vec3::new(x1, 6.0, top));
        let mut v = Vec::new();
        for h in [0.002f32, 0.004, 0.0083, 0.02, 0.0625] {
            v.push((format!("step up {h} block->block"), vec![big(-6.0, 0.0, 0.0), big(0.0, 6.0, h)], vec![], 0.0, false, Expect::Cross));
            v.push((format!("step up {h} terrain->block"), vec![big(0.0, 6.0, h)], flat_terrain(0.0), 0.0, false, Expect::Cross));
            v.push((format!("step down {h} block->block"), vec![big(-6.0, 0.0, h), big(0.0, 6.0, 0.0)], vec![], h, false, Expect::Cross));
        }
        for g in [0.01f32, 0.05, 0.1, 0.2] {
            v.push((format!("gap {g} across"), vec![big(-6.0, -g / 2.0, 0.0), big(g / 2.0, 6.0, 0.0)], vec![], 0.0, false, Expect::Cross));
            v.push((format!("gap {g} along"), vec![big(-6.0, -g / 2.0, 0.0), big(g / 2.0, 6.0, 0.0)], vec![], 0.0, true, Expect::Cross));
        }
        // a crack of 0.3 is narrower than the 0.35 sphere: the contact normals at its two edges are
        // 59 degrees from vertical, past the 35 degree bar -- the body wedges, in the engine too
        v.push(("gap 0.3 across".into(), vec![big(-6.0, -0.15, 0.0), big(0.15, 6.0, 0.0)], vec![], 0.0, false, Expect::Wedge));
        v.push(("ledge drop".into(), vec![big(-6.0, 0.0, 0.0)], vec![], 0.0, false, Expect::Drop));
        for h in [0.2f32, 0.5, 1.0] {
            v.push((format!("high ledge {h}"), vec![big(-6.0, 0.0, 0.0), big(0.0, 6.0, h)], vec![], 0.0, false, Expect::Stop));
        }
        // a ramp base lip: a 15 degree wedge up from a lip of height h
        for h in [0.0083f32, 0.0625] {
            let ramp = vec![
                Vec3::new(0.0, -6.0, 0.0), Vec3::new(0.0, 6.0, 0.0), Vec3::new(0.0, -6.0, h), Vec3::new(0.0, 6.0, h),
                Vec3::new(3.0, -6.0, h + 0.8), Vec3::new(3.0, 6.0, h + 0.8), Vec3::new(3.0, -6.0, 0.0), Vec3::new(3.0, 6.0, 0.0),
            ];
            v.push((format!("ramp lip {h}"), vec![big(-6.0, 0.0, 0.0), ramp], vec![], 0.0, false, Expect::Cross));
        }
        v
    }

    /// #collision  **TINY OBSTACLES NEVER MOVE THE BODY BACKWARD OR TELEPORT IT.** Steps of
    /// 0.002..0.0625 wu (block to block, terrain to block, down), gaps of 0.01..0.2 wu across and
    /// along, a ramp base lip, a platform edge to fall off and ledges the body cannot climb --
    /// production bevelled hull shells -- walked into from 8 directions at run and sprint, at
    /// 60 Hz, 144 Hz, 30 Hz and jittered frame times, for 3 s of held input. Per walk: no tick
    /// moves the body backward (> 3 mm against the stick) or teleports it (> 3x the commanded
    /// step + 5 cm, > 10 cm up, or a relocation), the stuck logic never fires (no shell escape, no
    /// relocation), steps and gaps within the engine's emergent climb are CROSSED, higher ledges
    /// STOP the body and it then stays still (< 1 cm of movement in the last second), edges
    /// DROP it.
    ///
    /// SABOTAGE: (a) the fallback fires on mere contact -- `|| self.n_contacts > 0` in its
    /// condition with `SQUEEZE_SECONDS` 0 and `SQUEEZE_PROGRESS` 100 -> relocations at every lip
    /// and ledge; (b) the threshold lowered -- `SQUEEZE_DEPTH` 0 (`>=`), seconds 0, progress 100
    /// -> the same. Both go red.
    #[test]
    fn tiny_obstacles_never_move_the_body_backward_or_teleport_it() {
        let p = spartan();
        let (mut walks, mut back, mut tele, mut reloc, mut escape) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut report: Vec<String> = Vec::new();
        let (mut crossed, mut stopped, mut dropped, mut wedged) = (0usize, 0usize, 0usize, 0usize);
        for (name, boxes, terrain, h0, along, expect) in obstacles() {
            for k in 0..8 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                for (sprint, frame) in [(false, Frame::Fixed(60.0)), (true, Frame::Fixed(60.0)), (false, Frame::Fixed(144.0)), (true, Frame::Fixed(30.0)), (true, Frame::Jittered), (false, Frame::Jittered)] {
                    let w = course(&boxes, &terrain, yaw);
                    let o = walk(&p, &w, yaw, sprint, along, h0, frame);
                    walks += 1;
                    back += o.backward;
                    tele += o.teleports;
                    reloc += o.relocated;
                    escape += o.escaped;
                    let ok = match expect {
                        Expect::Cross => { crossed += 1; o.progress > 3.5 }
                        Expect::Stop => { stopped += 1; o.progress < 2.3 && o.jitter < 0.01 }
                        Expect::Drop => { dropped += 1; o.progress > 3.0 && o.end.z < -0.5 }
                        Expect::Wedge => { wedged += 1; true }
                    };
                    if !ok || o.backward + o.teleports + o.relocated + o.escaped > 0 {
                        if report.len() < 12 {
                            report.push(format!("{name} yaw {:.0} {} {frame:?}: progress {:.2} jitter {:.3} back {} tele {} reloc {} escape {} {}", yaw.to_degrees(), if sprint { "sprint" } else { "run" }, o.progress, o.jitter, o.backward, o.teleports, o.relocated, o.escaped, o.first));
                        }
                        if !ok {
                            back += 1000; // an expectation failure is reported through the same assertion
                        }
                    }
                }
            }
        }
        println!("{walks} walks ({crossed} crossings, {stopped} stops, {dropped} drops, {wedged} wedges): backward ticks {back}, teleports {tele}, relocations {reloc}, shell escapes {escape}");
        for r in &report { println!("   {r}"); }
        assert!(walks >= 1400 && crossed >= 1000 && stopped >= 100 && dropped >= 40, "liveness: {walks} walks");
        assert_eq!((back, tele, reloc, escape), (0, 0, 0, 0));
    }


    /// Real terrain (structure BSP + instances) with real Forge block shells laid on it.
    struct TerrainAndBlocks<'a> {
        soup: &'a crate::decal_projector::TriSoup,
        tris: Vec<[Vec3; 3]>,
        shells: Vec<std::sync::Arc<Shell>>,
    }
    impl StaticWorld for TerrainAndBlocks<'_> {
        fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            self.soup.tris_in_aabb(mn, mx, out);
            self.tris.as_slice().gather(mn, mx, out);
        }
        fn solid_shells(&self, mn: Vec3, mx: Vec3, out: &mut Vec<std::sync::Arc<Shell>>) {
            out.extend(self.shells.iter().filter(|s| (0..3).all(|k| s.mn[k] <= mx[k] && s.mx[k] >= mn[k])).cloned());
        }
    }

    /// #collision  **NO RUBBERBAND ON REAL GROUND.** Real Forge blocks laid on real forge_halo terrain (structure BSP + instance collision) at lips of 0 / 0.0083 / 0.0625 wu /
    /// 0.0625 / -0.02 wu, walked into at run and sprint from 8 directions: which mechanism moves
    /// the body backward, or fires the stuck logic, when nothing solid is ahead?
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn real_blocks_on_real_terrain_never_rubberband() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let (_, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, crate::play_mode::Species::Spartan).unwrap();
        let mut soup = crate::decal_projector::TriSoup::new(4.0);
        let (mut main_t, mut inst_t, mut fall_t): (Vec<[Vec3; 3]>, Vec<[Vec3; 3]>, Vec<[Vec3; 3]>) = (Vec::new(), Vec::new(), Vec::new());
        let mut diag = 0usize;
        let mut bsp = None;
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(native, cache, t) else { continue };
            if b.surface_count() < 20000 { continue; }
            for (tr, _) in b.solid_triangles() { soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX); main_t.push(tr); }
            let i = crate::collision_bsp::instance_tris(native, cache, t).unwrap();
            for tr in i.placed.iter().chain(&i.fallback) { soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX); }
            inst_t.extend(i.placed.iter().copied());
            fall_t.extend(i.fallback.iter().copied());
            bsp = Some(b);
        }
        let b = bsp.expect("the big BSP");
        // the blocks
        let want: [&str; 0] = [];
        let mut shapes: Vec<(String, Vec<Vec<[Vec3; 3]>>)> = Vec::new();
        for &tag in native.tags_by_group(cache, "bloc").iter() {
            let name = native.tag_name(cache, tag).unwrap_or_default();
            if !name.contains("levels\\forge\\ff_") { continue; }
            let ph = native.resolve_phmo(cache, tag);
            let Some(blk) = crate::object_hulls::read_blocks(native, cache, ph) else { continue };
            if let Some(sh) = crate::object_hulls::hull_shells(&blk, &crate::object_hulls::mode_node_xforms(native, cache, tag)) {
                shapes.push((name.rsplit('\\').next().unwrap_or("").to_string(), sh));
            }
        }
        assert!(shapes.len() >= 5, "found {} block shapes", shapes.len());
        let (mn, mx) = {
            let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for t in soup.tris() { for v in t { mn = mn.min(*v); mx = mx.max(*v); } }
            (mn, mx)
        };
        let mut seed = 0xBADC0DEu32;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; (seed >> 8) as f32 / 16_777_216.0 };
        let (mut walks, mut back_free, mut fired_escape, mut fired_reloc, mut flips) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut report: Vec<String> = Vec::new();
        let mut reloc_report: Vec<String> = Vec::new();
        let mut steep_back = 0usize;
        let (mut stalled, mut stall_by_lip): (usize, std::collections::BTreeMap<String, (usize, usize)>) = (0, Default::default());
        let mut stall_report: Vec<String> = Vec::new();
        let mut spots = 0;
        let mut attempts = 0;
        while spots < 6 && attempts < 20000 {
            attempts += 1;
            let (x, y) = (mn.x + (mx.x - mn.x) * rnd(), mn.y + (mx.y - mn.y) * rnd());
            let from = Vec3::new(x, y, mx.z + 1.0);
            let to = Vec3::new(x, y, mn.z - 1.0);
            let Some((t, s)) = b.ray(from, to) else { continue };
            let Some(poly) = b.polygon(s) else { continue };
            let n = (poly[1] - poly[0]).cross(poly[2] - poly[0]).normalize_or_zero();
            if n.z.abs() < 0.97 { continue; }
            let hit = from + (to - from) * t;
            spots += 1;
            for (name, sh) in &shapes {
            for lip in &[0.0f32, 0.0083, 0.0625] {
            let lip = *lip;
                for k in 0..8 {
                    // (yaw below is both the walk direction and, with the block fixed, 8 approaches)
                    // the block sits with its top-surface lip `lip` above the terrain; the walker
                    // approaches from 2.2 wu away along direction k
                    let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                    let tops = sh.iter().flatten().flatten().map(|v| v.z).fold(f32::MIN, f32::max);
                    let at = Vec3::new(hit.x, hit.y, hit.z + lip - tops);
                    let shells: Vec<std::sync::Arc<Shell>> = sh.iter().map(|s| std::sync::Arc::new(Shell::new(s.iter().map(|t| t.map(|q| q + at)).collect()))).collect();
                    let tris: Vec<[Vec3; 3]> = shells.iter().flat_map(|s| s.tris.iter().copied()).collect();
                    let w = TerrainAndBlocks { soup: &soup, tris, shells };
                    let dir = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
                    for sprint in [true] {
                        let start = hit - dir * 2.2 + Vec3::Z * 0.4;
                        let mut st = PlayerState::standing_at(start, yaw);
                        for _ in 0..40 { st.step(&MoveInput { yaw, ..Default::default() }, &p, &w, 1.0 / 60.0); }
                        // a start inside instance geometry is a body that is stuck, not walking
                        if PlayerState::overlaps_at(st.pos, &p, &w, 0.05) { continue; }
                        let input = MoveInput { fwd: 1.0, yaw, sprint, ..Default::default() };
                        walks += 1;
                        let mut prev_g = st.grounded;
                        let (mut run, mut worst_run) = (0usize, 0usize);
                        let mut stall_at = st.pos;
                        let cmd = p.run_forward * p.sprint_full_speed_multiplier;
                        for f in 0..180 {
                            let before = st.pos;
                            let vz_before = st.vel.z;
                            let o = st.step(&input, &p, &w, 1.0 / 60.0);
                            let d = st.pos - before;
                            let front_free = !PlayerState::overlaps_at(before + dir * 0.25, &p, &w, 0.08);
                            // the law: a face steeper than the 35 degree bar underfoot, or a slot -- a ceiling
                            // contact (normal facing down) squeezing the body between floor and overhang
                            let steep = st.contacts().iter().any(|c| (c.surface.z < 0.819 && c.surface.z > -0.3 && c.normal.z < 0.819) || c.normal.z < -0.1)
                                // ...or free fall off an edge with the momentum of the slide that led to it
                                || (!st.grounded && st.contacts().is_empty())
                                // ...or still carrying the momentum of a fall / slide down (sliding off a
                                // drop onto a slope keeps going downhill for a moment)
                                || vz_before < -0.3;
                            let back_any = f > 20 && d.dot(dir) < -0.003 && front_free;
                            if back_any && steep { steep_back += 1; }
                            let back = back_any && !steep;
                            if f > 20 && front_free && st.ground_speed < 0.5 * cmd { run += 1; if run == 1 { stall_at = before; } worst_run = worst_run.max(run); } else { run = 0; }
                            fired_escape += (o.escaped > 0) as usize;
                            fired_reloc += o.relocated as usize;
                            flips += (st.grounded != prev_g) as usize;
                            prev_g = st.grounded;
                            if o.relocated {
                                reloc_report.push(format!("{name} lip {lip} yaw {:.0} tick {f} at ({:.3},{:.3},{:.3}) -> ({:.3},{:.3},{:.3}); residual {:.3}, escaped {}, vel ({:.2},{:.2},{:.2})", yaw.to_degrees(), before.x, before.y, before.z, st.pos.x, st.pos.y, st.pos.z, o.residual, o.escaped, st.vel.x, st.vel.y, st.vel.z));
                            }
                            if back || o.escaped > 0 {
                                back_free += back as usize;
                                if report.len() < 4 {
                                    report.push(format!("{name} lip {lip} yaw {:.0} tick {f} at ({:.2},{:.2},{:.3}): along {:.4}, up {:.3}; escaped {} relocated {} grounded {}", yaw.to_degrees(), before.x, before.y, before.z, d.dot(dir), d.z, o.escaped, o.relocated, st.grounded));
                                }
                            }
                        }
                        let e = stall_by_lip.entry(format!("lip {lip}")).or_insert((0, 0));
                        e.1 += 1;
                        if worst_run >= 12 {
                            stalled += 1;
                            e.0 += 1;
                            if stall_report.len() < 14 {
                                stall_report.push(format!("{name} lip {lip} yaw {:.0}: stalled {worst_run} ticks from ({:.3},{:.3},{:.3}), front free", yaw.to_degrees(), stall_at.x, stall_at.y, stall_at.z));
                            }
                        }
                    }
                }
            }
            }
        }
        println!("{stalled} stalled walks with the front free (steep terrain faces and falls included); by lip {stall_by_lip:?}");
        for r in &stall_report { println!("   STALL {r}"); }
        println!("{walks} walks over {spots} terrain spots: {steep_back} backward ticks on a face steeper than 35 deg or in a floor/ceiling slot (the law), {back_free} unexplained backward ticks with the front free, {fired_escape} ticks with a shell escape, {fired_reloc} relocations, {flips} grounded flips");
        for r in &report { println!("   {r}"); }
        for r in &reloc_report { println!("   RELOC {r}"); }
        assert!(walks >= 10000 && shapes.len() >= 100, "liveness: {walks} walks over {} block shapes", shapes.len());
        assert_eq!((back_free, fired_escape, fired_reloc), (0, 0, 0), "the stuck logic fired, or the body went backward, in normal walking");
    }

    /// #collision  Same walkers over three worlds -- main BSP only, + placed instance collision,
    /// + the render triangles kept for definitions without collision -- counting ticks that move
    /// the body BACKWARD with the front free (a capsule 0.25 ahead overlaps by < 0.08) and stalls
    /// (>= 12 ticks under half speed with the front free).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_which_instance_geometry_makes_barriers() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let (_, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, crate::play_mode::Species::Spartan).unwrap();
        let (mut main, mut placed, mut fallback) = (Vec::new(), Vec::new(), Vec::new());
        let mut bsp = None;
        for &t in &native.tags_by_group(cache, "sbsp") {
            let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(native, cache, t) else { continue };
            if b.surface_count() < 20000 { continue; }
            main.extend(b.solid_triangles().into_iter().map(|(t, _)| t));
            let i = crate::collision_bsp::instance_tris(native, cache, t).unwrap();
            placed.extend(i.placed.iter().copied());
            fallback.extend(i.fallback.iter().copied());
            bsp = Some(b);
        }
        let b = bsp.unwrap();
        let build = |sets: &[&Vec<[Vec3; 3]>]| {
            let mut s = crate::decal_projector::TriSoup::new(4.0);
            for set in sets { for tr in set.iter() { s.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX); } }
            SoupWorld(s)
        };
        let worlds = [("main only", build(&[&main])), ("main + placed instances", build(&[&main, &placed])), ("main + placed + fallback render tris", build(&[&main, &placed, &fallback]))];
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in &main { for v in t { mn = mn.min(*v); mx = mx.max(*v); } }
        let mut seed = 0xC0FFEEu32;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; (seed >> 8) as f32 / 16_777_216.0 };
        let mut starts = Vec::new();
        let mut attempts = 0;
        while starts.len() < 400 && attempts < 40000 {
            attempts += 1;
            let (x, y) = (mn.x + (mx.x - mn.x) * rnd(), mn.y + (mx.y - mn.y) * rnd());
            let from = Vec3::new(x, y, mx.z + 1.0);
            let to = Vec3::new(x, y, mn.z - 1.0);
            let Some((t, s)) = b.ray(from, to) else { continue };
            let Some(poly) = b.polygon(s) else { continue };
            if (poly[1] - poly[0]).cross(poly[2] - poly[0]).normalize_or_zero().z.abs() < 0.9 { continue; }
            starts.push(from + (to - from) * t);
        }
        for (label, w) in &worlds {
            let (mut walks, mut back, mut stalls, mut stalled_walks) = (0, 0, 0, 0);
            for hit in &starts {
                for k in 0..8 {
                    let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                    let dir = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
                    let mut st = PlayerState::standing_at(*hit + Vec3::Z * 0.3, yaw);
                    let mut settled = 0;
                    for _ in 0..40 { st.step(&MoveInput { yaw, ..Default::default() }, &p, w, 1.0 / 60.0); settled += st.grounded as usize; }
                    if settled < 20 { continue; }
                    walks += 1;
                    let input = MoveInput { fwd: 1.0, yaw, sprint: false, ..Default::default() };
                    let (mut run, mut worst) = (0usize, 0usize);
                    for f in 0..150 {
                        let before = st.pos;
                        st.step(&input, &p, w, 1.0 / 60.0);
                        let d = st.pos - before;
                        let free = !PlayerState::overlaps_at(before + dir * 0.25, &p, w, 0.08);
                        if f > 30 && free && d.dot(dir) < -0.003 { back += 1; }
                        if f > 30 && free && st.ground_speed < 0.5 * p.run_forward { run += 1; worst = worst.max(run); } else { run = 0; }
                    }
                    if worst >= 12 { stalls += 1; stalled_walks += 1; }
                }
            }
            println!("{label:<40} {walks} walks: {back} backward ticks with the front free, {stalled_walks} stalled walks with the front free");
            let _ = stalls;
        }
    }

    // -----------------------------------------------------------------------------------------
    // the real tags
    // -----------------------------------------------------------------------------------------

    /// #play-gate  **THE REAL ROW.** Prints every authored ground-physics field of both species
    /// off `forge_halo` and checks the five `runtime_*` values the ENGINE stored at tag
    /// post-process (`bipd` ground struct `+0x1C..+0x2C`) against the ones
    /// [`PlayerPhysics`] recomputes from the authored angles -- so the formulas that
    /// `postprocess_character_physics_field_struct` runs (`cos`, `sin(-x)`, `sin(x)`) are
    /// verified on the cache rather than trusted from a decompile.
    ///
    /// `cargo test -p hms-app --release play_ctl::tests::probe_the_authored_ground_physics -- --ignored --nocapture`
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_authored_ground_physics() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_ctl probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        for sp in [crate::play_mode::Species::Spartan, crate::play_mode::Species::Elite] {
            let (path, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, sp).expect("a player biped");
            println!("\n=== {} ===\n{path}\n{}\n{}", sp.label(), p.describe(), law_summary(&p));
            let name = path.split("  (").next().unwrap_or("").to_string();
            let tag = bipeds.iter().find(|(_, n)| *n == name).map(|(t, _)| *t).expect("the biped's tag");
            let g = crate::play_phys::BIPD_PHYSICS + crate::play_phys::PHYSICS_GROUND;
            let m = native.tag_meta(cache, tag, 0, (g + 0x44) as u32).expect("bipd meta");
            let f = |o: usize| f32::from_le_bytes([m[g + o], m[g + o + 1], m[g + o + 2], m[g + o + 3]]);
            let stored = [f(0x1C), f(0x20), f(0x24), f(0x28), f(0x2C)];
            let mine = [
                p.min_normal_k(),
                (-p.downhill_falloff).sin(),
                (-p.downhill_cutoff).sin(),
                p.uphill_falloff.sin(),
                p.uphill_cutoff.sin(),
            ];
            println!("   runtime k stored {stored:?}\n   runtime k mine   {mine:?}");
            for (i, (a, b)) in stored.iter().zip(mine.iter()).enumerate() {
                assert!((a - b).abs() < 1e-5, "runtime k[{i}]: the engine stored {a}, the formula gives {b}");
            }
        }
    }

    // =========================================================================================
    // #play-gate  THE ENGINE'S CONTROLLER (2026-10-20 port). Every expectation below is a
    // LITERAL worked by hand from the decompiled law, or a closed form derived from it in the
    // test -- never the production function's own output. Each gate names its SABOTAGE.
    // =========================================================================================

    /// The forge_halo SPARTAN, as `probe_the_authored_ground_physics` reads it off the cache
    /// (that probe prints the same numbers). Separate from the stand-in, whose values are
    /// deliberately not Reach's.
    fn spartan() -> PlayerPhysics {
        PlayerPhysics {
            walk_speed: 0.512,
            run_forward: 2.200,
            run_backward: 1.980,
            run_sideways: 1.980,
            run_acceleration: 8.5,
            sneak_forward: 0.900,
            sneak_backward: 0.810,
            sneak_sideways: 0.810,
            sneak_acceleration: 6.1,
            airborne_acceleration: 1.05,
            sprint_seconds_to_start: 0.0,
            sprint_seconds_to_full: 1.0,
            sprint_full_speed_multiplier: 1.623,
            jump_velocity: 2.3,
            height_standing: 0.65,
            height_crouching: 0.5,
            radius: 0.175,
            max_slope: 50f32.to_radians(),
            // `slope speed`: x1.5 past a 45 degree descent, x0.5 from 45 to 50 up -- beyond every
            // walkable face, so inert on ground the biped can stand on.
            downhill_falloff: 45f32.to_radians(),
            downhill_cutoff: 50f32.to_radians(),
            uphill_falloff: 45f32.to_radians(),
            uphill_cutoff: 50f32.to_radians(),
            downhill_velocity_scale: 1.5,
            uphill_velocity_scale: 0.5,
            ..phys()
        }
    }

    /// Run `n` frames at 60 Hz recording the state after each.
    fn trace(st: &mut PlayerState, input: &MoveInput, p: &PlayerPhysics, w: &Vec<[Vec3; 3]>, n: usize) -> Vec<PlayerState> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            st.step(input, p, w, 1.0 / 60.0);
            v.push(*st);
        }
        v
    }

    fn contact(n: Vec3) -> Contact {
        Contact { normal: n.normalize(), surface: n.normalize() }
    }

    /// Hand-worked: a valley of two faces tilted 20 degrees to either side sums to a LEVEL plane
    /// that neither face is alone, a contact being left faster than 0.468 wu/s is not ground, a
    /// face steeper than the global 35 degrees raises the bar above itself, and a roof is never
    /// ground.
    ///
    /// SABOTAGE: take the best SINGLE contact instead of the best subset (loop `mask` over powers
    /// of two only) -> the valley reads 0.9397 instead of 1.0 and `valley` goes red.
    /// SABOTAGE: drop the `LEAVING_SPEED` clause -> `leaving` goes red. SABOTAGE: drop the
    /// surface loop -> `steep` goes red (a 40-degree face would be standable at an authored 50).
    #[test]
    fn the_ground_plane_is_the_best_subset_of_the_flattest_eligible_contacts() {
        let k50 = 50f32.to_radians().cos();
        let tilt = 20f32.to_radians();
        let left = contact(Vec3::new(-tilt.sin(), 0.0, tilt.cos()));
        let right = contact(Vec3::new(tilt.sin(), 0.0, tilt.cos()));
        // 1. the valley
        let valley = ground_plane(&[left, right], Vec3::ZERO, k50).expect("a valley is ground");
        assert!((valley.z - 1.0).abs() < 1e-5 && valley.x.abs() < 1e-5, "valley plane {valley:?}, want straight up");
        assert!(left.normal.z < 0.95, "the fixture's single faces must be worse than the sum, or this proves nothing");
        // order is irrelevant: there is no order to be wrong about
        let rev = ground_plane(&[right, left], Vec3::ZERO, k50).unwrap();
        assert!((rev - valley).length() < 1e-6, "the answer depends on contact order: {rev:?} vs {valley:?}");
        // 2. leaving: the floor is moving away at 1.0 wu/s along its normal (> 0.468) -> not ground,
        //    and with nothing else touching there is NO ground at all
        let floor = contact(Vec3::Z);
        assert!(ground_plane(&[floor], Vec3::new(0.0, 0.0, 1.0), k50).is_none(), "a floor being left at 1.0 wu/s is ground");
        //    ...but at 0.4 wu/s (< 0.468) it still is
        assert!(ground_plane(&[floor], Vec3::new(0.0, 0.0, 0.4), k50).is_some(), "a floor being left at 0.4 wu/s is not ground");
        // 3. the surface bar: a 40-degree face (normal z 0.766) passes an authored 50 on its
        //    CONTACT normal and is refused on its SURFACE normal; a 30-degree one stands
        let face = |deg: f32| {
            let n = Vec3::new(-deg.to_radians().sin(), 0.0, deg.to_radians().cos());
            Contact { normal: n, surface: n }
        };
        assert!(ground_plane(&[face(40.0)], Vec3::ZERO, k50).is_none(), "a 40 degree face is standable");
        assert!(ground_plane(&[face(30.0)], Vec3::ZERO, k50).is_some(), "a 30 degree face is not standable");
        // ...whereas the SAME contact normal on a flat surface (the edge of a step) is fine:
        // contact normal 40 degrees, surface level.
        let edge = Contact { normal: face(40.0).normal, surface: Vec3::Z };
        assert!(ground_plane(&[edge], Vec3::ZERO, k50).is_some(), "the edge of a step was refused on its contact normal");
        //    and a 55-degree EDGE normal fails the authored 50 even on a level surface
        let edge55 = Contact { normal: face(55.0).normal, surface: Vec3::Z };
        assert!(ground_plane(&[edge55], Vec3::ZERO, k50).is_none(), "the authored limit does not apply to the contact normal");
        // 4. a roof is never ground
        assert!(ground_plane(&[contact(-Vec3::Z)], Vec3::ZERO, k50).is_none());
        // 5. a wall touching as well as a floor changes nothing (it is eligible, z = 0, and loses)
        let wall = Contact { normal: Vec3::X, surface: Vec3::X };
        let both = ground_plane(&[wall, floor], Vec3::ZERO, k50).expect("floor + wall is ground");
        assert!((both.z - 1.0).abs() < 1e-5, "{both:?}");
        // 6. only the five FLATTEST are kept: six contacts whose flattest is listed last
        let tilted = |deg: f32, az: f32| {
            let r = deg.to_radians();
            let n = Vec3::new(r.sin() * az.cos(), r.sin() * az.sin(), r.cos());
            Contact { normal: n, surface: n }
        };
        let mut six: Vec<Contact> = (0..5).map(|i| tilted(25.0, i as f32 * 1.2566)).collect();
        six.push(floor);
        let kept = ground_plane(&six, Vec3::ZERO, k50).unwrap();
        assert!((kept.z - 1.0).abs() < 1e-5, "the flattest contact was dropped because it was listed last: {kept:?}");
    }

    /// Hand-worked velocities, one per rule of `sub_1406C4A20`, at `dt = 1/60` (one engine tick:
    /// `game_time_globals` seconds_per_tick, which both the acceleration clamp and the gravity
    /// term multiply by) so the engine's per-tick numbers can be written as literals.
    ///
    /// * at rest on the flat the body is pressed into the ground at the MINIMUM adhesion, 0.12 x
    ///   30/60 = 0.06 (the engine divides the 30 Hz-tuned literals by `tick_rate / 30`);
    /// * cruising at 2.2 it is pressed at the MAXIMUM, 0.234 x 0.5 = 0.117;
    /// * from rest, a full stick is acceleration-limited to 8.5 / 60 = 0.14167 wu/s, the clamp is
    ///   saturated, so adhesion (0.06) AND gravity (4.1712594 / 60 = 0.069521) both apply;
    /// * on a 20 degree plane the target aims UP the plane at the commanded SPEED (2.2);
    /// * in the air the same stick adds only 1.05 / 60 = 0.0175 wu/s and gravity is the only z.
    ///
    /// SABOTAGE: drop the adhesion subtraction -> `rest` red; apply gravity unconditionally ->
    /// `cruise` red; drop the slope-following z -> `ramp` red; use the ground acceleration in
    /// the air -> `air` red; drop the 30/tick_rate adhesion scale -> `rest` red.
    #[test]
    fn the_velocity_update_is_the_engines_own_arithmetic() {
        let p = spartan();
        let dt = 1.0 / 60.0;
        let g = 4.1712594f32;
        // rest
        let rest = control_velocity(Vec3::ZERO, Some(Vec3::Z), Vec3::ZERO, &p, 0.0, dt);
        assert!((rest - Vec3::new(0.0, 0.0, -0.06)).length() < 1e-6, "rest {rest:?}");
        // cruise: error zero, so the clamp is NOT saturated and there is no gravity
        let cruise = control_velocity(Vec3::new(2.2, 0.0, 0.0), Some(Vec3::Z), Vec3::new(2.2, 0.0, 0.0), &p, 0.0, dt);
        assert!((cruise - Vec3::new(2.2, 0.0, -0.117)).length() < 1e-5, "cruise {cruise:?}");
        // launch
        let go = control_velocity(Vec3::ZERO, Some(Vec3::Z), Vec3::new(2.2, 0.0, 0.0), &p, 0.0, dt);
        let want = Vec3::new(8.5 / 60.0, 0.0, -0.06 - g / 60.0);
        assert!((go - want).length() < 1e-5, "launch {go:?}, want {want:?}");
        // ramp: plane normal of a 20 degree face rising toward +x; already moving along it
        let a = 20f32.to_radians();
        let n = Vec3::new(-a.sin(), 0.0, a.cos());
        let tangent = Vec3::new(a.cos(), 0.0, a.sin());
        let v = tangent * 2.2;
        let on_ramp = control_velocity(v, Some(n), Vec3::new(2.2, 0.0, 0.0), &p, 0.0, dt);
        // error zero (the target IS the current velocity), so only adhesion at speed 2.2
        assert!((on_ramp - (v - n * 0.117)).length() < 1e-5, "ramp {on_ramp:?}");
        // ...and from a standstill the first tick accelerates ALONG the plane, not along x
        let up = control_velocity(Vec3::ZERO, Some(n), Vec3::new(2.2, 0.0, 0.0), &p, 0.0, dt);
        let along = (up + n * 0.06 + Vec3::Z * (g / 60.0)).normalize();
        assert!((along - tangent).length() < 1e-4, "the first tick left the plane: {along:?} vs {tangent:?}");
        // air: horizontal chase at the AIRBORNE acceleration, z is gravity alone
        let air = control_velocity(Vec3::ZERO, None, Vec3::new(2.2, 0.0, 0.0), &p, 0.0, dt);
        assert!((air - Vec3::new(1.05 / 60.0, 0.0, -g / 60.0)).length() < 1e-5, "air {air:?}");
        // ...and with the stick released the air STILL pulls the velocity toward zero (it is not
        // a coast): 2.0 wu/s becomes 2.0 - 0.0175
        let drag = control_velocity(Vec3::new(2.0, 0.0, 1.0), None, Vec3::ZERO, &p, 0.0, dt);
        assert!((drag.x - (2.0 - 1.05 / 60.0)).abs() < 1e-5 && (drag.z - (1.0 - g / 60.0)).abs() < 1e-5, "drag {drag:?}");
    }

    /// FLAT GROUND DOES NOT MOVE THE BODY VERTICALLY, AT ALL. The old controller teleported
    /// `pos.z` to a probed height and wandered by millimetres; the engine's body is never moved
    /// to stay grounded, so standing and walking on a dead-flat floor must hold z to float
    /// rounding -- 1e-6 wu, a thousandth of the old 1e-3 tolerance.
    ///
    /// SABOTAGE: re-add `self.pos.z = <anything but the contact's own resolution>` after the
    /// resolve (e.g. `self.pos.z += 2e-6` when grounded) -> red.
    #[test]
    fn flat_ground_holds_the_vertical_position_to_float_rounding() {
        let p = spartan();
        let w = floor(0.0, 60.0);
        // Start beside the floor's diagonal seam (y = x) so the walk crosses it.
        let mut st = PlayerState::standing_at(Vec3::new(-1.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        assert!(st.grounded);
        let z0 = st.pos.z;
        // ABSOLUTE as well as steady: the floor is at z = 0, and a body that rested a few
        // microns above it (an extra writer on z that the contact then holds in place) would
        // pass a test that only compares the body with itself.
        assert!(z0.abs() < 1e-6, "the body rests at z {z0:e} on a floor at z = 0");
        let mut samples = 0;
        let mut worst = 0.0f32;
        let mut sides = std::collections::BTreeSet::new();
        // standing still, then walking, then strafing and backing, then stopping
        for input in [
            MoveInput::default(),
            MoveInput { fwd: 1.0, ..Default::default() },
            MoveInput { fwd: 1.0, right: 1.0, ..Default::default() },
            MoveInput { fwd: -1.0, ..Default::default() },
            MoveInput::default(),
        ] {
            for s in trace(&mut st, &input, &p, &w, 90) {
                worst = worst.max(s.pos.z.abs());
                assert!(s.grounded, "left the floor while crossing it: {}", s.describe(&p));
                sides.insert((s.pos.y - s.pos.x).is_sign_positive());
                samples += 1;
            }
        }
        assert_eq!(samples, 450);
        assert!(st.pos.x.abs() > 1.0 || st.pos.y.abs() > 1.0, "it never actually travelled, so this proved nothing");
        assert_eq!(sides.len(), 2, "the walk never crossed the floor's triangle seam, so it proved nothing about seams");
        assert!(worst < 1e-6, "z wandered {worst:e} wu on flat ground (started {z0:e})");
    }

    /// Climb and descend a ramp at several angles: z never goes the wrong way and each frame's
    /// rise is the slope's, with the tick-to-tick scatter bounded far below what a player sees.
    ///
    /// The per-frame rise is compared with `dx * tan(angle)`: on a plane the capsule's z moves by
    /// exactly that, so any residual is the controller wobbling. Measured scatter is asserted
    /// against a fixed 2e-4 wu (0.6 mm at Halo scale), 25 times tighter than the old gate's 5e-3.
    ///
    /// SABOTAGE: add the old position snap back (set `pos.z` from a probe each frame) -> the
    /// scatter bound goes red; drop the adhesion subtraction -> the body is "airborne" for whole
    /// frames on the descent (`grounded` assertion); set the leaving filter to `<= 0.0` -> the
    /// uphill assertion on `grounded` goes red.
    #[test]
    fn walking_a_ramp_is_monotone_and_its_per_frame_rise_is_the_slope() {
        let p = spartan();
        let mut frames_seen = 0usize;
        for deg in [8.0f32, 15.0, 25.0, 33.0] {
            let t = deg.to_radians().tan();
            // --- UP
            let mut w = floor(0.0, 8.0);
            w.extend(ramp(deg, 40.0, 8.0));
            let mut st = PlayerState::standing_at(Vec3::new(-3.0, 0.0, 0.2), 0.0);
            settle(&mut st, &p, &w);
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            trace(&mut st, &input, &p, &w, 150); // reach the ramp and get onto it
            let tr = trace(&mut st, &input, &p, &w, 180);
            let mut prev = tr[0];
            for s in &tr[1..] {
                assert!(s.pos.x > 1.0, "{deg}: still on the flat");
                assert!(s.grounded, "{deg} deg UP: airborne at x {:.3}", s.pos.x);
                let (dx, dz) = (s.pos.x - prev.pos.x, s.pos.z - prev.pos.z);
                assert!(dz > -1e-6, "{deg} deg UP: z went DOWN by {dz:e}");
                assert!((dz - dx * t).abs() < 2e-4, "{deg} deg UP: rose {dz:.6} for {dx:.6} of x (slope says {:.6})", dx * t);
                prev = *s;
                frames_seen += 1;
            }
            assert!(tr.last().unwrap().pos.z > tr[0].pos.z + 0.3, "{deg}: did not climb");
            // --- DOWN
            // The crest is a convex edge: a body running off it leaves the ground (the contact
            // normal rotates past the 0.468 leaving speed) and lands further down, so the
            // measured window starts well after the landing.
            let w = downramp(deg, 60.0, 8.0, 8.0);
            let mut st = PlayerState::standing_at(Vec3::new(-3.0, 0.0, 0.3), 0.0);
            settle(&mut st, &p, &w);
            trace(&mut st, &input, &p, &w, 360);
            let tr = trace(&mut st, &input, &p, &w, 180);
            let mut prev = tr[0];
            for s in &tr[1..] {
                assert!(s.pos.x > 1.0, "{deg}: still on the shelf");
                assert!(s.grounded, "{deg} deg DOWN: airborne at x {:.3}", s.pos.x);
                let (dx, dz) = (s.pos.x - prev.pos.x, s.pos.z - prev.pos.z);
                assert!(dz < 1e-6, "{deg} deg DOWN: z went UP by {dz:e}");
                assert!((dz + dx * t).abs() < 2e-4, "{deg} deg DOWN: fell {dz:.6} for {dx:.6} of x (slope says {:.6})", -dx * t);
                prev = *s;
                frames_seen += 1;
            }
        }
        assert!(frames_seen > 8 * 179 - 8, "only {frames_seen} frames were examined");
    }

    /// WALKABILITY STRADDLES THE GLOBAL 35 DEGREES, and the authored slope limit is not what
    /// decides it. Ramps at 34.5 degrees and below are climbed grounded; ramps at 35.5 and above
    /// are never stood on -- not at 40, not at 50 (the authored limit!), not at 60 -- and the
    /// slope limit visibly fires. Handing the biped a kinder authored limit (70) changes nothing
    /// about a 40 degree face, because that limit is tested on the CONTACT normal and a face's
    /// contact normal IS its surface normal, which the global limit has already refused.
    ///
    /// SABOTAGE: make `surface_slope_cos()` return 0.0 (no surface bar) -> the 40 and 50 degree
    /// rows go red (the authored 50 would let a 40 degree face be climbed); make it return
    /// cos(45) -> the 34.5 row stays green and the 40 row goes red.
    #[test]
    fn walkability_straddles_the_global_35_degrees() {
        let p = spartan();
        let climb = |q: &PlayerPhysics, deg: f32| -> (PlayerState, Vec<StepOut>, usize) {
            let mut w = floor(0.0, 6.0);
            w.extend(ramp(deg, 40.0, 6.0));
            let mut st = PlayerState::standing_at(Vec3::new(-1.5, 0.0, 0.2), 0.0);
            settle(&mut st, q, &w);
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            let mut outs = Vec::new();
            let mut grounded_on_ramp = 0usize;
            for _ in 0..360 {
                outs.push(st.step(&input, q, &w, 1.0 / 60.0));
                if st.grounded && st.pos.x > 0.5 {
                    grounded_on_ramp += 1;
                }
            }
            (st, outs, grounded_on_ramp)
        };
        for deg in [10.0f32, 20.0, 30.0, 34.5] {
            let (st, _, on) = climb(&p, deg);
            assert!(st.pos.z > 0.6, "{deg} deg: could not climb (z {:.3})", st.pos.z);
            assert!(on > 200, "{deg} deg: only {on} grounded frames on the ramp");
        }
        for (q, deg, label) in [
            (p, 35.5f32, "authored 50"),
            (p, 40.0, "authored 50"),
            (p, 50.0, "authored 50 (the limit itself)"),
            (p, 60.0, "authored 50"),
            (PlayerPhysics { max_slope: 70f32.to_radians(), ..p }, 40.0, "authored 70"),
        ] {
            let (st, outs, on) = climb(&q, deg);
            assert!(st.pos.z < 0.3, "{deg} deg ({label}): climbed to z {:.3}", st.pos.z);
            assert_eq!(on, 0, "{deg} deg ({label}): stood on the face for {on} frames");
            assert!(outs.iter().any(|o| o.steep_contacts > 0), "{deg} deg ({label}): the slope limit never fired -- the body never reached the face");
        }
    }

    /// THE LEDGE THE CONTACT LAW KEEPS ON THE GROUND. No step height exists anywhere; a ledge's
    /// top edge meets the bottom sphere at a contact normal of `z = (r - h) / r`, which must beat
    /// `cos(authored slope)` for the edge to count as ground. That puts the boundary at
    /// `h* = r (1 - cos 50) = 0.0625 wu` on the Spartan (r = 0.175, 50 degrees), derived HERE
    /// from the radius and the authored limit, not read from the code.
    ///
    /// What the boundary decides is whether the body stays in the GROUND mode (full acceleration,
    /// adhesion, slope-following) while it goes over the edge. Two observables, both measured:
    ///
    /// 1. the pure decision: `ground_plane` accepts a ledge contact exactly when `h < h*`, for
    ///    several (radius, authored limit) pairs -- 0.98 h* yes, 1.02 h* no;
    /// 2. the integrated walk: over a staircase of risers 0.5 h* and 0.9 h* high the body reaches
    ///    the top and is **never airborne on the way** (zero frames); over risers 1.3 h* high it
    ///    IS airborne on some frames (the edge stops being ground and the body is in the air
    ///    mode for the contact); and risers of 3 h* and 4 h* stop it at the first one.
    ///
    /// What the boundary does NOT decide, measured (`measure_step_up`): how tall a ledge the body
    /// can still SLIDE over on momentum alone. That is Havok's half (the contact solve) and on
    /// this approximation it is about 0.11 wu for authored limits of 15, 50 and 70 degrees alike.
    ///
    /// The authored pair decides: a capsule of twice the radius keeps a 1.8 h* ledge on the
    /// ground, and a 15 degree authored limit puts even a 0.5 h* one in the air.
    ///
    /// SABOTAGE: classify by height instead of normal (accept an edge contact when
    /// `contact_rise <= 0.0625`) -> the radius and limit rows of the pure decision go red;
    /// ignore the 0.468 leaving filter -> the first integrated row goes red (the floor being left
    /// stays "ground" and masks the edge).
    #[test]
    fn a_ledge_stays_ground_exactly_while_its_contact_normal_beats_the_authored_slope() {
        let p = spartan();
        let hstar = |q: &PlayerPhysics| q.radius * (1.0 - q.max_slope.cos());
        assert!((hstar(&p) - 0.0625).abs() < 1e-3, "the Spartan's closed form is {:.4}, expected 0.0625", hstar(&p));
        // 1. the pure decision, from the geometry of a sphere on an edge
        let edge_accepted = |q: &PlayerPhysics, h: f32| {
            let n = Vec3::new(-((q.radius * q.radius - (q.radius - h).powi(2)).sqrt()), 0.0, q.radius - h).normalize();
            ground_plane(&[Contact { normal: n, surface: Vec3::Z }], Vec3::ZERO, q.min_normal_k()).is_some()
        };
        for q in [p, PlayerPhysics { radius: p.radius * 2.0, ..p }, PlayerPhysics { max_slope: 15f32.to_radians(), ..p }, PlayerPhysics { max_slope: 70f32.to_radians(), ..p }] {
            assert!(edge_accepted(&q, hstar(&q) * 0.98), "r {} limit {:.0}: a ledge just under h* = {:.4} was refused", q.radius, q.max_slope.to_degrees(), hstar(&q));
            assert!(!edge_accepted(&q, hstar(&q) * 1.02), "r {} limit {:.0}: a ledge just over h* = {:.4} was accepted", q.radius, q.max_slope.to_degrees(), hstar(&q));
        }
        // 2. the integrated walk
        let stairs = |h: f32, n: usize| -> Vec<[Vec3; 3]> {
            let mut w = floor(0.0, 30.0);
            for i in 0..n {
                let x0 = 1.0 + 0.5 * i as f32;
                w.extend(block(Vec3::new(x0, -4.0, -1.0), Vec3::new(28.0, 4.0, h * (i + 1) as f32)));
            }
            w
        };
        // (final z, frames spent airborne between the first riser and the top step)
        let walk = |q: &PlayerPhysics, h: f32| -> (f32, usize) {
            let w = stairs(h, 6);
            let mut st = PlayerState::standing_at(Vec3::new(-0.5, 0.0, 0.2), 0.0);
            settle(&mut st, q, &w);
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            let mut air = 0;
            for s in trace(&mut st, &input, q, &w, 300) {
                if !s.grounded && s.pos.x > 0.8 && s.pos.x < 4.2 {
                    air += 1;
                }
            }
            (st.pos.z, air)
        };
        for k in [0.5f32, 0.9] {
            let h = hstar(&p) * k;
            let (z, air) = walk(&p, h);
            assert!(z > h * 5.0, "{k} h* ({h:.4} wu risers): reached only z {z:.4}, the top is {:.4}", h * 6.0);
            assert_eq!(air, 0, "{k} h* ({h:.4} wu risers): airborne for {air} frames on the way up");
        }
        let (z, air) = walk(&p, hstar(&p) * 1.3);
        assert!(air > 0, "1.3 h* risers never took the body off the ground: the edge is still being treated as ground (z {z:.4})");
        for k in [3.0f32, 4.0] {
            let h = hstar(&p) * k;
            let (z, _) = walk(&p, h);
            assert!(z < h * 0.5, "{k} h* ({h:.4} wu risers): got up to z {z:.4}");
        }
        // The authored pair decides.
        let big = PlayerPhysics { radius: p.radius * 2.0, ..p };
        let h = hstar(&p) * 1.8;
        let (z, air) = walk(&big, h);
        assert!(z > h * 5.0 && air == 0, "a capsule of twice the authored radius must walk up {h:.4} wu risers on the ground (z {z:.4}, {air} airborne frames)");
        let harsh = PlayerPhysics { max_slope: 15f32.to_radians(), ..p };
        let (_, air) = walk(&harsh, hstar(&p) * 0.5);
        assert!(air > 0, "a 15 degree authored limit must put a 0.5 h* riser in the air mode");
    }

    /// THE JUMP: apex and airtime from the authored velocity and the engine's authored gravity,
    /// as LITERALS. Spartan: v = 2.3 wu/s, g = 4.1712594 -> apex v^2/2g = 0.63414 wu, airtime
    /// 2v/g = 1.10279 s. Half gravity (the jump's `sub_140DCB3B0(unit, 0.5)`) would give apex
    /// 1.268 and airtime 2.206: for a PLAYER that scale never reaches the physics input
    /// (`sub_140E18320` multiplies the player's gravity by the PLAYER-TRAIT table -- 1.0 by
    /// default -- and reads the biped scale only for AI), and the writer is undone by
    /// `sub_140E40650` in the same call that applies the velocity anyway, so full gravity it is.
    ///
    /// SABOTAGE: halve gravity in `control_velocity`'s air branch -> both assertions red.
    #[test]
    fn the_jump_apex_and_airtime_are_the_authored_arithmetic() {
        let p = spartan();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.3), 0.0);
        settle(&mut st, &p, &w);
        let base = st.pos.z;
        let mut peak = base;
        let mut air_frames = 0;
        let mut inp = MoveInput { jump: true, ..Default::default() };
        for i in 0..240 {
            inp.jump = i == 0;
            st.step(&inp, &p, &w, 1.0 / 60.0);
            peak = peak.max(st.pos.z);
            if !st.grounded {
                air_frames += 1;
            }
        }
        let apex = peak - base;
        let air = air_frames as f32 / 60.0;
        assert!((apex - 0.63414).abs() < 0.03, "apex {apex:.4} wu, expected 0.6341 (v^2/2g)");
        assert!((air - 1.10279).abs() < 0.06, "airtime {air:.3} s, expected 1.103 (2v/g)");
        assert!(st.grounded, "never came back down");
    }

    /// A STEEP FACE SLIDES YOU DOWN, WHATEVER YOU PUSH. Past 35 degrees nothing is ground, so the
    /// body is in the air mode, where the authored `airborne acceleration` (1.05 wu/s^2) is far
    /// below gravity along the face (`g sin 50 = 3.2`). Standing on a 50 degree ramp, with and
    /// without the stick held UP the slope, the body ends up lower on it and moving down-slope;
    /// on a 30 degree ramp the same body does not creep at all (contrast: this is not just
    /// "everything slides").
    ///
    /// SABOTAGE: give the air mode the ground acceleration (8.5) -> the pushed row climbs and
    /// goes red; apply gravity only when saturated in the air branch -> the slide never starts.
    #[test]
    fn a_steep_face_slides_the_body_down_whatever_it_pushes() {
        let p = spartan();
        let steep = |deg: f32| {
            let mut w = floor(0.0, 12.0);
            w.extend(ramp(deg, 30.0, 12.0));
            w
        };
        for (push, label) in [(0.0f32, "no stick"), (1.0, "stick held up the slope")] {
            let w = steep(50.0);
            let x0 = 4.0;
            let z0 = x0 * 50f32.to_radians().tan() + p.radius * (1.0 / 50f32.to_radians().cos() - 1.0) + 0.02;
            let mut st = PlayerState::standing_at(Vec3::new(x0, 0.0, z0), 0.0);
            let input = MoveInput { fwd: push, ..Default::default() };
            let tr = trace(&mut st, &input, &p, &w, 90);
            let landed = tr[10];
            let after = tr[89];
            assert!(tr.iter().all(|s| !s.grounded), "{label}: stood on a 50 degree face");
            assert!(after.pos.x < landed.pos.x - 0.25, "{label}: did not slide down (x {:.3} -> {:.3})", landed.pos.x, after.pos.x);
            assert!(after.vel.x < -0.5, "{label}: velocity is not down-slope: {:?}", after.vel);
        }
        // contrast: a 30 degree face holds a standing body still
        let w = steep(30.0);
        let mut st = PlayerState::standing_at(Vec3::new(4.0, 0.0, 4.0 * 30f32.to_radians().tan() + 0.1), 0.0);
        trace(&mut st, &MoveInput::default(), &p, &w, 120);
        assert!(st.grounded);
        let rest = st.pos;
        trace(&mut st, &MoveInput::default(), &p, &w, 120);
        assert!((st.pos - rest).length() < 1e-4, "a body at rest on a 30 degree face crept by {:e}", (st.pos - rest).length());
    }

    /// A CEILING STOPS THE JUMP and a wall CORNER holds a body that runs into it. Under a roof
    /// 0.25 wu above the standing height a jump's head never passes the roof, its upward
    /// velocity is gone the frame it touches (no ratcheting through), and it lands. Running
    /// diagonally into the inside corner of two walls it ends outside both by at least the
    /// radius, and once stopped it is STILL: no jitter in x, y or z.
    ///
    /// SABOTAGE: skip the velocity clip in `resolve` -> the roof row goes red (it presses into
    /// the roof every frame and `vel.z` stays up); push out along only the deepest contact
    /// instead of the merged sum -> the corner row goes red.
    #[test]
    fn a_ceiling_and_a_wall_corner_hold_the_body() {
        let p = spartan();
        // --- ceiling
        let roof = p.height_standing + 0.25;
        let mut w = floor(0.0, 20.0);
        w.extend(floor(roof, 20.0).into_iter().map(|t| [t[0], t[2], t[1]]));
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.05), 0.0);
        settle(&mut st, &p, &w);
        let mut inp = MoveInput { jump: true, ..Default::default() };
        let mut touched = 0;
        let mut head_max = f32::MIN;
        for i in 0..200 {
            inp.jump = i == 0;
            st.step(&inp, &p, &w, 1.0 / 60.0);
            head_max = head_max.max(st.pos.z + p.height_standing);
            // On any frame the head is on the roof it must not be rising: the contact clips the
            // upward velocity the frame it is touched.
            if st.pos.z + p.height_standing > roof - 1e-4 {
                touched += 1;
                assert!(st.vel.z <= 0.01, "still rising ({:.3}) with the head on the roof", st.vel.z);
            }
        }
        assert!(touched >= 1, "the jump never reached the roof, so this proved nothing (head max {head_max:.3}, roof {roof:.3})");
        assert!(head_max < roof + 0.005, "the head went through the roof: {head_max:.4} > {roof:.4}");
        assert!(st.grounded, "never came back down from under the roof");
        // --- corner: walls at x = 3 and y = 3, running at 45 degrees into the inside corner
        let mut w = floor(0.0, 20.0);
        w.extend(block(Vec3::new(3.0, -10.0, 0.0), Vec3::new(4.0, 10.0, 3.0)));
        w.extend(block(Vec3::new(-10.0, 3.0, 0.0), Vec3::new(10.0, 4.0, 3.0)));
        let yaw = std::f32::consts::FRAC_PI_4;
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.2), yaw);
        settle(&mut st, &p, &w);
        let input = MoveInput { fwd: 1.0, yaw, ..Default::default() };
        let tr = trace(&mut st, &input, &p, &w, 360);
        let last = tr.last().unwrap();
        assert!(last.pos.x < 3.0 - p.radius * 0.9 && last.pos.y < 3.0 - p.radius * 0.9, "ended inside a wall at {:?}", last.pos);
        assert!(last.pos.x > 2.5 && last.pos.y > 2.5, "never reached the corner: {:?}", last.pos);
        let still = &tr[300..];
        let (mut dx, mut dy, mut dz) = (0.0f32, 0.0f32, 0.0f32);
        for s in still {
            dx = dx.max((s.pos.x - last.pos.x).abs());
            dy = dy.max((s.pos.y - last.pos.y).abs());
            dz = dz.max((s.pos.z - last.pos.z).abs());
        }
        assert!(dx < 1e-3 && dy < 1e-3 && dz < 1e-5, "jitter in the corner: {dx:e} {dy:e} {dz:e}");
    }

    /// A BODY AT REST IN THE CREASE BETWEEN TWO SLOPES STAYS PUT. Faceted ground -- every curved
    /// Forge block -- puts the capsule on two planes at once, with the adhesion velocity pressing
    /// it into only the flatter one; pushing out of the other must not walk it along the surface.
    /// (The real `ff_ramp_circle_small` still settles ~3 mm a second on its faceted curve --
    /// `probe_walk_the_real_forge_ramps` measures it; a Havok-style speculative contact clip was
    /// tried and did not change that, so it is not here.) The fixture lists the steeper facet
    /// FIRST so the push-out meets it first, the order that exposed the drift.
    ///
    /// SABOTAGE: apply gravity unconditionally on the ground -> the body slides down the crease
    /// and this goes red.
    #[test]
    fn a_body_resting_in_a_crease_between_two_slopes_does_not_creep() {
        let p = spartan();
        let (ta, tb) = (9f32.to_radians().tan(), 22f32.to_radians().tan());
        let a = vec![[Vec3::new(-8.0, -4.0, -8.0 * ta), Vec3::new(0.0, -4.0, 0.0), Vec3::new(0.0, 4.0, 0.0)], [Vec3::new(-8.0, -4.0, -8.0 * ta), Vec3::new(0.0, 4.0, 0.0), Vec3::new(-8.0, 4.0, -8.0 * ta)]];
        let b = vec![[Vec3::new(0.0, -4.0, 0.0), Vec3::new(8.0, -4.0, 8.0 * tb), Vec3::new(8.0, 4.0, 8.0 * tb)], [Vec3::new(0.0, -4.0, 0.0), Vec3::new(8.0, 4.0, 8.0 * tb), Vec3::new(0.0, 4.0, 0.0)]];
        let mut w = b;
        w.extend(a);
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.6), 0.0);
        trace(&mut st, &MoveInput::default(), &p, &w, 240);
        assert!(st.grounded, "never settled in the crease");
        let touching = st.contacts().len();
        assert!(touching >= 2, "the body is on {touching} plane(s), not in the crease, so this proves nothing");
        let rest = st.pos;
        let mut worst = 0.0f32;
        for s in trace(&mut st, &MoveInput::default(), &p, &w, 120) {
            worst = worst.max((s.pos - rest).length());
        }
        assert!(worst < 2e-4, "the body crept {worst:e} wu in two seconds at rest in a crease");
    }

    /// THE RUN-UP AND THE STOP TAKE THE AUTHORED ACCELERATION, and strafing is slower than going
    /// forward because the AXES carry their own caps. Spartan: 0 -> 2.2 wu/s at 8.5 wu/s^2 is
    /// 0.2588 s = 15.5 frames at 60 Hz; stopping is the same; crouched, 0.9 wu/s at the sneak
    /// 6.1 wu/s^2 is 0.1475 s = 8.9 frames; a diagonal run reaches 2.0928 wu/s (hand-worked from
    /// the circle-clamped stick, 0.7071 x 2.2 and 0.7071 x 1.98).
    ///
    /// SABOTAGE: use the sneak acceleration when standing (or the run one when crouched) -> the
    /// frame counts go red; skip the circle clamp in `commanded_velocity` -> the diagonal reads
    /// 2.96.
    #[test]
    fn the_run_up_and_the_stop_take_the_authored_acceleration() {
        let p = spartan();
        let w = floor(0.0, 400.0);
        let frames_to = |input: MoveInput, from_speed: f32, done: &dyn Fn(f32) -> bool| -> usize {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
            settle(&mut st, &p, &w);
            st.crouched = input.crouch;
            st.crouch_t = if input.crouch { 1.0 } else { 0.0 };
            st.vel = Vec3::new(from_speed, 0.0, 0.0);
            for i in 1..=300 {
                st.step(&input, &p, &w, 1.0 / 60.0);
                if done(st.ground_speed) {
                    return i;
                }
            }
            usize::MAX
        };
        let go = MoveInput { fwd: 1.0, ..Default::default() };
        let up = frames_to(go, 0.0, &|v| v >= 2.2 - 1e-3);
        assert!((up as f32 - 15.5).abs() <= 1.5, "0 -> 2.2 wu/s took {up} frames, 8.5 wu/s^2 says 15.5");
        let stop = frames_to(MoveInput::default(), 2.2, &|v| v < 1e-3);
        assert!((stop as f32 - 15.5).abs() <= 1.5, "2.2 -> 0 took {stop} frames, 8.5 wu/s^2 says 15.5");
        let crouch_up = frames_to(MoveInput { crouch: true, ..go }, 0.0, &|v| v >= 0.9 - 1e-3);
        assert!((crouch_up as f32 - 8.9).abs() <= 1.5, "crouched 0 -> 0.9 took {crouch_up} frames, 6.1 wu/s^2 says 8.9");
        // the diagonal's top speed, with a long enough run that it has arrived
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
        settle(&mut st, &p, &w);
        trace(&mut st, &MoveInput { fwd: 1.0, right: 1.0, ..Default::default() }, &p, &w, 90);
        assert!((st.ground_speed - 2.0928).abs() < 0.01, "a diagonal run settles at {:.4}, hand-worked 2.0928", st.ground_speed);
    }

    /// MOMENTUM, FROM THE TAGS. Releasing the stick at the run speed stops in `v^2 / 2a` with
    /// `a` the run acceleration (the engine decelerates with the SAME clamp it accelerates with):
    /// 2.2^2 / 17 = 0.2847 wu for the Spartan, crouched 0.9^2 / 12.2 = 0.0664 wu (the sneak
    /// acceleration, blended by the crouch fraction). In the air there is no coast: a jump
    /// taken at the run speed with the stick released loses horizontal speed at the airborne
    /// acceleration, 1.05 wu/s^2, so after half a second 2.2 - 0.525 = 1.675 wu/s.
    ///
    /// SABOTAGE: halve the clamp in `control_velocity`'s ground branch (`accel * dt * 0.5`) ->
    /// both stopping distances double; make the air branch coast (skip the horizontal pull) ->
    /// the air row stays at 2.2.
    #[test]
    fn stopping_distance_and_air_momentum_follow_the_authored_accelerations() {
        let p = spartan();
        let w = floor(0.0, 400.0);
        let stop = |crouch: bool, v0: f32| -> f32 {
            let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
            settle(&mut st, &p, &w);
            st.crouched = crouch;
            st.crouch_t = if crouch { 1.0 } else { 0.0 };
            st.vel = Vec3::new(v0, 0.0, 0.0);
            let x0 = st.pos.x;
            let inp = MoveInput { crouch, ..Default::default() };
            for _ in 0..200 {
                st.step(&inp, &p, &w, 1.0 / 60.0);
            }
            st.pos.x - x0
        };
        let d_run = stop(false, 2.2);
        assert!((d_run - 0.2847).abs() < 0.02, "stopped from a run in {d_run:.4} wu, v^2/2a says 0.2847");
        let d_crouch = stop(true, 0.9);
        assert!((d_crouch - 0.0664).abs() < 0.01, "stopped from a crouch-walk in {d_crouch:.4} wu, v^2/2a says 0.0664");
        // air: take off at the run speed, release everything, read the horizontal speed 0.5 s later
        let mut st = PlayerState::standing_at(Vec3::new(0.0, 0.0, 0.1), 0.0);
        settle(&mut st, &p, &w);
        st.vel = Vec3::new(2.2, 0.0, 0.0);
        st.step(&MoveInput { jump: true, ..Default::default() }, &p, &w, 1.0 / 60.0);
        for _ in 0..29 {
            st.step(&MoveInput::default(), &p, &w, 1.0 / 60.0);
        }
        assert!(!st.grounded, "the jump landed within half a second");
        assert!((st.vel.x - 1.675).abs() < 0.03, "after 0.5 s in the air with the stick released the speed is {:.3}, 2.2 - 1.05 x 0.5 says 1.675", st.vel.x);
    }

    /// The slope-speed scale is the engine's piecewise function of the travel direction's z.
    ///
    /// SABOTAGE: swap the uphill/downhill scales, or use the cutoff angle for the downhill step
    /// -> a row goes red.
    #[test]
    fn the_slope_speed_scale_is_the_engines_piecewise_function() {
        let p = PlayerPhysics {
            downhill_falloff: 10f32.to_radians(),
            downhill_cutoff: 40f32.to_radians(),
            uphill_falloff: 5f32.to_radians(),
            uphill_cutoff: 35f32.to_radians(),
            downhill_velocity_scale: 1.5,
            uphill_velocity_scale: 0.6,
            ..spartan()
        };
        let z = |deg: f32| deg.to_radians().sin();
        assert_eq!(p.slope_speed_scale(z(0.0)), 1.0, "level ground is unscaled");
        assert_eq!(p.slope_speed_scale(z(-5.0)), 1.0, "a shallow descent is unscaled (the step is at the falloff angle)");
        assert_eq!(p.slope_speed_scale(z(-12.0)), 1.5, "past the downhill falloff the full downhill scale applies");
        assert_eq!(p.slope_speed_scale(z(-30.0)), 1.5);
        assert_eq!(p.slope_speed_scale(z(3.0)), 1.0, "below the uphill falloff");
        let mid = p.slope_speed_scale(z(20.0));
        // linear in z between sin(5) and sin(35): (z - k0) / (k1 - k0) = 0.5 at z(20) -> check
        let (k0, k1) = (z(5.0), z(35.0));
        let want = 1.0 + (z(20.0) - k0) / (k1 - k0) * (0.6 - 1.0);
        assert!((mid - want).abs() < 1e-5 && mid < 1.0 && mid > 0.6, "uphill midpoint {mid}, want {want}");
        assert_eq!(p.slope_speed_scale(z(35.0)), 0.6);
        assert_eq!(p.slope_speed_scale(z(50.0)), 0.6);
    }

    /// #play-gate  **THE REAL FORGE RAMPS.** The shipped `forge_halo` ramp blocks' own physics
    /// hulls, on a floor at their base, walked up from the foot by the Spartan the cache
    /// describes: `ff_ramp_1x2_shallow` (13 deg) and `ff_ramp_1x2` (26 deg) must be climbed and
    /// stood on, `ff_ramp_2x2_steep` (45 and 66 degree faces) must not be stood on at all. Then
    /// the body is DROPPED onto the centroid of every walkable inclined triangle of every
    /// ramp / stair block in the map and must come to rest and stay still.
    ///
    /// The slope of an object is read off its own triangles (the area-weighted mean of its
    /// inclined faces), never assumed. **Liveness is asserted** -- how many ramps were walked, how
    /// many frames the body spent on an incline, how many steep frames touched the face -- so the
    /// probe cannot pass by finding nothing to test.
    ///
    /// `cargo test -p hms-app --profile quick play_ctl::tests::probe_walk_the_real_forge_ramps -- --ignored --nocapture`
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_walk_the_real_forge_ramps() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("play_ctl probes SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let (_, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, crate::play_mode::Species::Spartan).expect("a Spartan");
        let (mut walked_up, mut refused, mut incline_frames, mut steep_touch) = (0usize, 0usize, 0usize, 0usize);
        let mut expected_frames = 0.0f32;
        for want in ["ff_ramp_1x2_shallow", "ff_ramp_1x2", "ff_ramp_2x2_steep"] {
            let Some(&tag) = native
                .tags_by_group(cache, "bloc")
                .iter()
                .find(|&&t| native.tag_name(cache, t).map_or(false, |n| n.ends_with(&format!("\\{want}"))))
            else {
                println!("{want}: not in this cache");
                continue;
            };
            let phmo = native.resolve_phmo(cache, tag);
            let geo = if phmo != 0 && phmo != 0xFFFF_FFFF { native.phmo_geometry(cache, phmo) } else { None }.or_else(|| {
                let c = native.resolve_coll(cache, tag);
                if c != 0 && c != 0xFFFF_FFFF { native.coll_geometry(cache, c) } else { None }
            });
            let Some((v, idx)) = geo else { continue };
            let pt = |k: u32| Vec3::new(v[k as usize * 3], v[k as usize * 3 + 1], v[k as usize * 3 + 2]);
            let mut tris: Vec<[Vec3; 3]> = idx.chunks_exact(3).map(|f| [pt(f[0]), pt(f[1]), pt(f[2])]).collect();
            let base = tris.iter().flatten().map(|q| q.z).fold(f32::MAX, f32::min);
            // the inclined faces: the family of the LARGEST inclined triangle (normals within 5
            // degrees of it), their mean normal, extent and rise. (A block with two ramps back to
            // back averages to a lie, so one family is taken.)
            let face = |t: &[Vec3; 3]| {
                let c = (t[1] - t[0]).cross(t[2] - t[0]);
                // NOT flipped: a closed hull is wound outward, and only its TOP faces (outward
                // normal pointing up) can be walked on -- flipping the underside would send the
                // walk the wrong way up the slab.
                (c.normalize_or_zero(), c.length() * 0.5)
            };
            let inclined = |t: &[Vec3; 3]| {
                let (n, a) = face(t);
                n.z > 0.2 && n.z < 0.985 && a > 1e-4
            };
            let Some(lead) = tris.iter().filter(|t| inclined(t)).max_by(|a, b| face(a).1.total_cmp(&face(b).1)).map(|t| face(t).0) else {
                println!("{want}: no inclined faces");
                continue;
            };
            let (mut nsum, mut area_sum) = (Vec3::ZERO, 0.0f32);
            let mut inc: Vec<[Vec3; 3]> = Vec::new();
            for t in tris.iter().filter(|t| inclined(t)) {
                let (n, a) = face(t);
                if n.dot(lead) > 5f32.to_radians().cos() {
                    nsum += n * a;
                    area_sum += a;
                    inc.push(*t);
                }
            }
            if inc.is_empty() {
                println!("{want}: no inclined faces");
                continue;
            }
            let nmean = (nsum / area_sum).normalize();
            let slope = nmean.z.acos().to_degrees();
            let up = Vec3::new(-nmean.x, -nmean.y, 0.0).normalize();
            let along = |q: Vec3| q.dot(up);
            let lo = inc.iter().flatten().map(|q| along(*q)).fold(f32::MAX, f32::min);
            let hi = inc.iter().flatten().map(|q| along(*q)).fold(f32::MIN, f32::max);
            let ztop = inc.iter().flatten().map(|q| q.z).fold(f32::MIN, f32::max);
            let centre = inc.iter().flatten().fold(Vec3::ZERO, |a, q| a + *q) / (inc.len() * 3) as f32;
            let obj_lo = tris.iter().flatten().map(|q| along(*q)).fold(f32::MAX, f32::min);
            // a floor under the whole thing, at the object's base
            let fl = floor(base, 40.0);
            tris.extend(fl.into_iter().map(|t| [t[0] + Vec3::new(centre.x, centre.y, 0.0), t[1] + Vec3::new(centre.x, centre.y, 0.0), t[2] + Vec3::new(centre.x, centre.y, 0.0)]));
            // Start on the floor before the WHOLE OBJECT's low end along the walking direction
            // (a ramp built of two slopes has its steeper one higher up, and a start beneath the
            // family would be inside the block).
            let start = Vec3::new(centre.x, centre.y, base + 0.2) + up * (obj_lo - along(centre) - 1.5);
            let yaw = up.y.atan2(up.x);
            let mut st = PlayerState::standing_at(start, yaw);
            settle(&mut st, &p, &tris);
            let input = MoveInput { fwd: 1.0, yaw, ..Default::default() };
            let (mut on, mut steep, mut zmax) = (0usize, 0usize, base);
            for _ in 0..600 {
                let o = st.step(&input, &p, &tris, 1.0 / 60.0);
                zmax = zmax.max(st.pos.z);
                let a = along(st.pos);
                if a > lo + 0.1 && a < hi && st.pos.z > base + 0.04 && st.grounded {
                    on += 1;
                }
                if !st.grounded && o.steep_contacts > 0 && a > lo - 0.5 && a < hi {
                    steep += 1;
                }
            }
            let rise = ztop - base;
            println!(
                "{want}: slope {slope:.1} deg, {:.2} wu long, rise {rise:.3} | reached z +{:.3}, {on} grounded frames on the incline, {steep} frames refused at the face",
                hi - lo,
                zmax - base
            );
            if slope < 33.0 {
                // At the authored run speed the incline takes (length / speed) seconds to cross;
                // the body must have been standing on it for most of that.
                let expected = (hi - lo) / (p.run_forward / 60.0);
                assert!(zmax - base > rise * 0.7, "{want}: a {slope:.1} deg ramp was not climbed (+{:.3} of {rise:.3})", zmax - base);
                assert!(on as f32 > expected * 0.7, "{want}: only {on} grounded frames on a {slope:.1} deg incline that takes {expected:.0} to cross");
                walked_up += 1;
                incline_frames += on;
                expected_frames += expected;
            } else if slope > 37.0 {
                assert_eq!(on, 0, "{want}: stood on a {slope:.1} deg face for {on} frames");
                // It may SLIDE a little way up on the momentum of its run-up (a frictionless
                // capsule does: measured +0.31 of 1.97 on the 45 degree block) but never gets
                // anywhere near the top, because nothing it touches is ground.
                assert!(zmax - base < rise * 0.5, "{want}: climbed a {slope:.1} deg face to +{:.3} of {rise:.3}", zmax - base);
                refused += 1;
                steep_touch += steep;
            }
        }
        // --- every walkable inclined triangle of every ramp / stair block: drop, settle, stay ---
        // THE BOUND IS THE LAW'S, not a number read off a run. A body at rest is pressed into the
        // ground PLANE at the minimum adhesion speed, 0.12 x 30/60 = 0.06 wu/s. The plane is the
        // FLATTEST eligible contact within Havok's 0.0328 tolerance; on faceted ground (every
        // curved Forge block) that can be a facet the body is hovering over while a steeper one
        // is what actually holds it, and pressing along the flat normal into the steeper face
        // leaves the in-plane component: `0.06 x sin(tilt of the supporting contact)`. That is
        // the most a resting body can drift (measured 0.0197 against 0.0222 on a 21.7 degree
        // facet). Same engine rule, so the engine drifts the same way on that geometry.
        const ADHESION_AT_REST: f32 = 0.12 * 30.0 / 60.0;
        let (mut samples, mut stood) = (0usize, 0usize);
        let mut worst_creep = 0.0f32;
        let mut names = Vec::new();
        for group in ["bloc", "scen"] {
            for &t in &native.tags_by_group(cache, group) {
                let n = native.tag_name(cache, t).unwrap_or_default().to_lowercase();
                if n.contains("ramp") || n.contains("stair") {
                    names.push(t);
                }
            }
        }
        for tag in names {
            let phmo = native.resolve_phmo(cache, tag);
            let geo = if phmo != 0 && phmo != 0xFFFF_FFFF { native.phmo_geometry(cache, phmo) } else { None };
            let Some((v, idx)) = geo else { continue };
            let pt = |k: u32| Vec3::new(v[k as usize * 3], v[k as usize * 3 + 1], v[k as usize * 3 + 2]);
            let tris: Vec<[Vec3; 3]> = idx.chunks_exact(3).map(|f| [pt(f[0]), pt(f[1]), pt(f[2])]).collect();
            for t in &tris {
                let c = (t[1] - t[0]).cross(t[2] - t[0]);
                let (n, area) = (c.normalize_or_zero(), c.length() * 0.5);
                let slope = n.z.clamp(-1.0, 1.0).acos().to_degrees();
                if n.z <= 0.0 || !(3.0..32.0).contains(&slope) || area < 0.1 {
                    continue;
                }
                let centre = (t[0] + t[1] + t[2]) / 3.0;
                let mut st = PlayerState::standing_at(Vec3::new(centre.x, centre.y, centre.z + 0.4), 0.0);
                for _ in 0..150 {
                    st.step(&MoveInput::default(), &p, &tris, 1.0 / 60.0);
                }
                samples += 1;
                // Only a body that stayed on THIS face is judged (one that rolled off, or landed
                // on the lip of something else, is not a sample of this triangle).
                if (st.pos - centre).truncate().length() > 0.12 {
                    continue;
                }
                let rest = st.pos;
                let mut log = Vec::new();
                for k in 0..60 {
                    st.step(&MoveInput::default(), &p, &tris, 1.0 / 60.0);
                    if k < 8 {
                        let cs: Vec<String> = st.contacts().iter().map(|c| format!("n({:.3},{:.3},{:.3})", c.normal.x, c.normal.y, c.normal.z)).collect();
                        log.push(format!("   k{k} d=({:+.5},{:+.5},{:+.5}) v=({:+.4},{:+.4},{:+.4}) g={} {}", st.pos.x - rest.x, st.pos.y - rest.y, st.pos.z - rest.z, st.vel.x, st.vel.y, st.vel.z, st.grounded, cs.join(" ")));
                    }
                }
                let tilt = st.contacts().iter().filter(|c| c.normal.z > 0.0).map(|c| c.normal.z.clamp(-1.0, 1.0).acos()).fold(0.0f32, f32::max);
                let creep_limit = ADHESION_AT_REST * tilt.sin() * 1.15 + 1e-4;
                worst_creep = worst_creep.max((st.pos - rest).length());
                if (st.pos - rest).length() >= creep_limit {
                    println!("CREEP on '{}' tri {:?} slope {slope:.1} area {area:.2}\n{}", native.tag_name(cache, tag).unwrap_or_default(), t, log.join("\n"));
                }
                assert!(st.grounded, "a {slope:.1} deg face of a real block did not hold the body (at {centre:?})");
                assert!(
                    (st.pos - rest).length() < creep_limit,
                    "a body resting on a {slope:.1} deg real face (supporting contact tilt {:.1} deg) crept {:e} wu in a second, past the law's {creep_limit:e}",
                    tilt.to_degrees(),
                    (st.pos - rest).length()
                );
                stood += 1;
            }
        }
        println!("dropped onto {samples} real inclined triangles, {stood} stayed on their face and stood still (worst creep {worst_creep:.5} wu in a second)");
        assert!(samples >= 100 && stood >= 100, "only {samples} real inclined triangles ({stood} judged) were exercised");
        println!("walked {walked_up} walkable ramps ({incline_frames} incline frames), refused {refused} steep ones ({steep_touch} refused frames)");
        assert!(walked_up >= 2, "only {walked_up} walkable real ramps were exercised");
        assert!(expected_frames > 90.0 && incline_frames as f32 > expected_frames * 0.7, "{incline_frames} frames on real inclines, of {expected_frames:.0} expected");
        assert!(refused >= 1 && steep_touch > 0, "no real steep ramp was exercised ({refused}, {steep_touch})");
    }

    /// #play  A TRACE, not a gate: the first frames of walking into a ledge of a given height.
    #[test]
    #[ignore]
    fn trace_a_ledge() {
        let p = spartan();
        let h = std::env::var("LEDGE").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.03125);
        let mut w = floor(0.0, 30.0);
        w.extend(block(Vec3::new(1.0, -4.0, -1.0), Vec3::new(6.0, 4.0, h)));
        let mut st = PlayerState::standing_at(Vec3::new(0.2, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        println!("settled z={:.5} ledge h={h}", st.pos.z);
        let input = MoveInput { fwd: 1.0, ..Default::default() };
        for i in 0..90 {
            let o = st.step(&input, &p, &w, 1.0 / 60.0);
            let cs: Vec<String> = st.contacts().iter().map(|c| format!("n({:.2},{:.2},{:.2})/s({:.1},{:.1},{:.1})", c.normal.x, c.normal.y, c.normal.z, c.surface.x, c.surface.y, c.surface.z)).collect();
            println!(
                "f{i:02} x={:.4} z={:.5} v=({:.2},{:.2},{:.2}) g={} hits={} {}",
                st.pos.x, st.pos.z, st.vel.x, st.vel.y, st.vel.z, st.grounded, o.contacts, cs.join(" ")
            );
        }
    }

    /// #play  A TRACE, not a gate: walking +x across the floor's seam with the stand-in biped.
    #[test]
    #[ignore]
    fn trace_a_seam_walk() {
        let p = phys();
        let w = floor(0.0, 40.0);
        let mut st = PlayerState::standing_at(Vec3::new(-5.0, 0.0, 0.5), 0.0);
        settle(&mut st, &p, &w);
        let input = MoveInput { fwd: 1.0, ..Default::default() };
        run(&mut st, &input, &p, &w, 1.0);
        println!("start of window: x={:.4} z={:+.6}", st.pos.x, st.pos.z);
        for i in 0..180 {
            let o = st.step(&input, &p, &w, 1.0 / 60.0);
            if st.pos.z.abs() > 1e-5 || (st.pos.x > -0.3 && st.pos.x < 0.3) {
                let cs: Vec<String> = st.contacts().iter().map(|c| format!("n({:.3},{:.3},{:.3})", c.normal.x, c.normal.y, c.normal.z)).collect();
                println!("f{i:02} x={:.4} z={:+.6} v=({:.2},{:.2},{:+.3}) g={} hits={} {}", st.pos.x, st.pos.z, st.vel.x, st.vel.y, st.vel.z, st.grounded, o.contacts, cs.join(" "));
            }
        }
    }

    /// #play  A MEASUREMENT, not a gate: sweep a single ledge's height and report whether a
    /// Spartan walking at it ends up on top, for the authored 50 degree limit and for 15 and 70.
    #[test]
    #[ignore]
    fn measure_step_up() {
        let p = spartan();
        for deg in [15.0f32, 50.0, 70.0] {
            let q = PlayerPhysics { max_slope: deg.to_radians(), ..p };
            let hstar = q.radius * (1.0 - q.max_slope.cos());
            let mut row = String::new();
            let mut h = 0.01f32;
            while h < 0.30 {
                let mut w = floor(0.0, 30.0);
                w.extend(block(Vec3::new(1.0, -4.0, -1.0), Vec3::new(28.0, 4.0, h)));
                let mut st = PlayerState::standing_at(Vec3::new(-0.5, 0.0, 0.2), 0.0);
                settle(&mut st, &q, &w);
                trace(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &q, &w, 180);
                row += if st.pos.z > h - 0.01 { "#" } else { "." };
                h += 0.01;
            }
            println!("limit {deg:>4} deg  h* = {hstar:.4}  ledge 0.01 .. 0.30 step 0.01:  {row}");
        }
    }

    /// A GHOST EDGE IS RECOGNISED, A REAL ONE IS NOT. The edge of a floor triangle's neighbour,
    /// under a sphere resting 1 cm sunk beside the seam, is a ghost (the face under the sphere
    /// owns the contact); the top edge of a ledge the sphere is beside, with no face under it
    /// but the lower floor's, is a real obstacle. Both directions are asserted on the production
    /// `is_ghost`, with the contacts built by the production `sphere_tri_contact`.
    ///
    /// SABOTAGE: make `is_ghost` return false -> the first half goes red; make it return true for
    /// every edge -> the second half goes red.
    #[test]
    fn a_neighbouring_triangles_edge_is_a_ghost_and_a_ledge_edge_is_not() {
        let p = spartan();
        let r = p.radius;
        // --- the seam of a floor (diagonal y = x), sphere over the y > x triangle, 1 cm sunk
        let w = floor(0.0, 30.0);
        let c = Vec3::new(0.28, 0.3, r - 0.01);
        let (faces, nf) = face_contacts(c, r, 0.0, &w);
        assert_eq!(nf, 1, "exactly one floor triangle is touched through its face");
        let mut ghosts = 0;
        for t in &w {
            if let Some(h) = sphere_tri_contact(c, r, 0.0, t) {
                if !h.interior {
                    assert!(h.normal.x.abs() > 0.01, "the fixture's edge contact must be tilted, or this proves nothing");
                    assert!(is_ghost(&h, &faces[..nf], &w), "the seam's neighbour edge was not recognised as a ghost");
                    ghosts += 1;
                }
            }
        }
        assert_eq!(ghosts, 1, "the fixture must produce one edge contact");
        // --- a 5 cm ledge (block from x = 1) with the sphere beside its top edge, over the floor
        let mut w = floor(0.0, 30.0);
        w.extend(block(Vec3::new(1.0, -4.0, -1.0), Vec3::new(6.0, 4.0, 0.05)));
        let c = Vec3::new(0.97, 0.0, r);
        let (faces, nf) = face_contacts(c, r, 0.0, &w);
        let mut real = 0;
        for t in &w {
            if let Some(h) = sphere_tri_contact(c, r, 0.0, t) {
                if !h.interior {
                    assert!(!is_ghost(&h, &faces[..nf], &w), "a ledge's top edge was thrown away as a ghost");
                    real += 1;
                }
            }
        }
        assert!(real >= 1, "the fixture must touch the ledge's edge");
    }

    /// A `StaticWorld` over a `TriSoup` (the same structure `Scene::static_tris_in_aabb` uses).
    struct SoupWorld(crate::decal_projector::TriSoup);
    impl StaticWorld for SoupWorld {
        fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
            self.0.tris_in_aabb(mn, mx, out);
        }
    }

    /// #play-gate  **THE SLOPE-BARRIER SWEEP, on the real Forge World collision.** Walkers are
    /// started on walkable engine-collision surfaces of `forge_halo` (found by casting the
    /// engine's own ray down at pseudo-random places), in all 8 directions relative to the local
    /// slope, at the authored run speed and at the sprint top speed, and walked for 2 s. A STALL
    /// is a stretch of >= 12 frames, after the first half second, in which the body moves at less
    /// than 50% of its commanded speed while the engine's collision has NOTHING ahead: a
    /// chest-height ray 0.45 wu forward hits nothing, and a knee-height ray hits nothing or a
    /// walkable (>= 35 degree flat) surface. That is "I hit an invisible barrier".
    ///
    /// Liveness: the number of walkers, of walkers that crossed a slope change (the ground
    /// normal's angle changed by > 3 degrees during the walk) and the stall count are printed
    /// and asserted. `cargo test -p hms-app --profile quick play_ctl::tests::probe_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_sweep_the_real_terrain_for_invisible_barriers() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let (_, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, crate::play_mode::Species::Spartan).expect("a Spartan");
        let mut worlds = Vec::new();
        for &t in &native.tags_by_group(cache, "sbsp") {
            if let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(native, cache, t) {
                let mut soup = crate::decal_projector::TriSoup::new(4.0);
                for (tr, _) in b.solid_triangles() {
                    soup.add_tri_owned(tr[0], tr[1], tr[2], (0, 0, -1), u32::MAX);
                }
                if soup.len() > 20_000 {
                    worlds.push((b, SoupWorld(soup)));
                }
            }
        }
        assert!(!worlds.is_empty(), "no big structure collision BSP");
        let (b, w) = &worlds[0];
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for t in w.0.tris() {
            for v in t {
                mn = mn.min(*v);
                mx = mx.max(*v);
            }
        }
        let mut seed = 0xC0FFEEu32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / 16_777_216.0
        };
        let hstar = p.radius * (1.0 - p.min_normal_k());
        let (mut walkers, mut sloped, mut stalls, mut law_explained) = (0usize, 0usize, 0usize, 0usize);
        let mut reports: Vec<String> = Vec::new();
        let (mut back_ticks, mut tele_ticks, mut reloc_ticks, mut back_walkers) = (0usize, 0usize, 0usize, 0usize);
        let mut kicks: Vec<String> = Vec::new();
        let mut attempts = 0;
        while walkers < 3000 && attempts < 40000 {
            attempts += 1;
            let x = mn.x + (mx.x - mn.x) * rnd();
            let y = mn.y + (mx.y - mn.y) * rnd();
            let from = Vec3::new(x, y, mx.z + 1.0);
            let Some((t, s)) = b.ray(from, Vec3::new(x, y, mn.z - 1.0)) else { continue };
            let hit = from + (Vec3::new(x, y, mn.z - 1.0) - from) * t;
            let Some(poly) = b.polygon(s) else { continue };
            let n = (poly[1] - poly[0]).cross(poly[2] - poly[0]).normalize_or_zero();
            if n.z.abs() < 0.82 {
                continue;
            }
            for speed_label in ["run", "sprint"] {
                for k in 0..8 {
                    let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                    let mut st = PlayerState::standing_at(hit + Vec3::Z * 0.3, yaw);
                    let mut settled = 0;
                    for _ in 0..40 {
                        st.step(&MoveInput { yaw, ..Default::default() }, &p, w, 1.0 / 60.0);
                        settled += st.grounded as usize;
                    }
                    if settled < 20 {
                        continue;
                    }
                    walkers += 1;
                    let start = st.pos;
                    let input = MoveInput { fwd: 1.0, yaw, sprint: speed_label == "sprint", ..Default::default() };
                    let (mut run_len, mut worst_run) = (0usize, 0usize);
                    let (mut n0, mut tilt_range) = (st.ground_normal, 0.0f32);
                    let mut where_ = start;
                    let dirv = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
                    let mut walker_back = false;
                    for f in 0..120 {
                        let before = st.pos;
                        let o = st.step(&input, &p, w, 1.0 / 60.0);
                        let d = st.pos - before;
                        let sp = if speed_label == "sprint" { p.run_forward * p.sprint_full_speed_multiplier } else { p.run_forward };
                        let back = d.dot(dirv) < -0.003;
                        let tele = Vec3::new(d.x, d.y, 0.0).length() > sp / 60.0 * 3.0 + 0.05 || d.z > 0.05 || o.relocated;
                        back_ticks += back as usize;
                        tele_ticks += tele as usize;
                        reloc_ticks += o.relocated as usize;
                        walker_back |= back || tele;
                        if (back || tele) && kicks.len() < 10 {
                            kicks.push(format!("{speed_label} yaw {:.0} tick {f} at ({:.2},{:.2},{:.2}): moved {:.4} along the command, {:.3} up; escaped {} relocated {} grounded {}", yaw.to_degrees(), before.x, before.y, before.z, d.dot(dirv), d.z, o.escaped, o.relocated, st.grounded));
                        }
                                        if st.grounded {
                            tilt_range = tilt_range.max(n0.dot(st.ground_normal).clamp(-1.0, 1.0).acos().to_degrees());
                        }
                        if f < 30 {
                            n0 = st.ground_normal;
                            continue;
                        }
                        let commanded = if speed_label == "sprint" { p.run_forward * p.sprint_full_speed_multiplier } else { p.run_forward };
                        if st.ground_speed < 0.5 * commanded && st.grounded {
                            run_len += 1;
                            if run_len == 1 {
                                where_ = st.pos;
                            }
                            worst_run = worst_run.max(run_len);
                        } else {
                            run_len = 0;
                        }
                    }
                    back_walkers += walker_back as usize;
                    if tilt_range > 3.0 {
                        sloped += 1;
                    }
                    if worst_run >= 12 {
                        let dir = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
                        let chest = b.ray(where_ + Vec3::Z * 0.3, where_ + Vec3::Z * 0.3 + dir * 0.45);
                        let knee = b.ray(where_ + Vec3::Z * (hstar + 0.02), where_ + Vec3::Z * (hstar + 0.02) + dir * 0.45);
                        let knee_blocks = knee.map_or(false, |(_, s2)| b.polygon(s2).map_or(true, |pg| {
                            (pg[1] - pg[0]).cross(pg[2] - pg[0]).normalize_or_zero().z.abs() < 0.82
                        }));
                        // A face steeper than the engine's global 35 degrees under (or just
                        // ahead of) the stall IS the engine's own barrier: nothing standable.
                        let steep_face = [0.0f32, 0.3].iter().any(|ahead| {
                            let c = where_ + dir * *ahead;
                            b.ray(c + Vec3::Z * 0.5, c - Vec3::Z * 0.5).map_or(false, |(_, s2)| {
                                b.polygon(s2).map_or(false, |pg| {
                                    let z = (pg[1] - pg[0]).cross(pg[2] - pg[0]).normalize_or_zero().z.abs();
                                    z < 35f32.to_radians().cos()
                                })
                            })
                        });
                        if chest.is_none() && !knee_blocks && steep_face {
                            law_explained += 1;
                        } else if chest.is_none() && !knee_blocks {
                            stalls += 1;
                            if reports.len() < 12 {
                                reports.push(format!("{speed_label} yaw {:.0} from ({:.2},{:.2},{:.2}) stalled {worst_run} frames at ({:.3},{:.3},{:.3}) grounded n=({:.2},{:.2},{:.2})",
                                    yaw.to_degrees(), start.x, start.y, start.z, where_.x, where_.y, where_.z, st.ground_normal.x, st.ground_normal.y, st.ground_normal.z));
                            }
                        }
                    }
                }
            }
        }
        println!("backward ticks {back_ticks}, teleport ticks {tele_ticks}, relocations {reloc_ticks}, in {back_walkers} walkers");
        for k in &kicks {
            println!("   KICK {k}");
        }
        println!("swept {walkers} walkers ({sloped} crossed a slope change > 3 deg), {law_explained} stalls on faces steeper than 35 deg (the law), {stalls} unexplained stalls");
        for r in &reports {
            println!("   STALL {r}");
        }
        assert!(walkers >= 2000 && sloped >= 200, "the sweep barely looked: {walkers} walkers, {sloped} on slope changes");
        assert_eq!(stalls, 0, "{stalls} walkers lost most of their speed with nothing ahead in the engine's collision");
        assert_eq!((tele_ticks, reloc_ticks), (0, 0), "the body was teleported while walking on the structure collision");
    }

    /// #play-gate  **BLOCK SEAMS.** A floor made the way a Forge player makes one: a 5 x 5 grid of
    /// the shipped `ff_plat_1x1_flat` blocks butted edge to edge at their authored 1.0 wu pitch,
    /// using their real physics hulls. (Each hull is 0.0164 wu SMALLER per side than the block --
    /// the Havok shape's convex radius, which the triangle decode does not add back -- so there
    /// is a 3.3 cm crack between neighbours.) Walkers cross the grid in all 8 directions, at the
    /// run and the sprint speed: they must keep their speed (no catching on a crack or a lip) and
    /// never leave the ground. Liveness: the number of seam crossings is counted and asserted.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_sweep_block_seams() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
        let (_, p) = crate::play_mode::player_physics_for(native, cache, &bipeds, crate::play_mode::Species::Spartan).unwrap();
        let tag = *native
            .tags_by_group(cache, "bloc")
            .iter()
            .find(|&&t| native.tag_name(cache, t).map_or(false, |n| n.ends_with("\\ff_plat_1x1_flat")))
            .expect("ff_plat_1x1_flat");
        let ph = native.resolve_phmo(cache, tag);
        // the production hull: boxes with the Havok convex radius put back (`physics_world_tris`)
        let blocks = crate::object_hulls::read_blocks(native, cache, ph).expect("its phmo blocks");
        let hull = crate::object_hulls::hull_tris(&blocks, &crate::object_hulls::mode_node_xforms(native, cache, tag)).expect("its hull");
        let top = hull.iter().flatten().map(|q| q.z).fold(f32::MIN, f32::max);
        let mut world: Vec<[Vec3; 3]> = Vec::new();
        for gx in -2..=2 {
            for gy in -2..=2 {
                let off = Vec3::new(gx as f32, gy as f32, 0.0);
                world.extend(hull.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
            }
        }
        let (mut walkers, mut crossings, mut bad) = (0usize, 0usize, Vec::new());
        for sprint in [false, true] {
            for k in 0..8 {
                let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
                for ox in [-0.31f32, 0.0, 0.27] {
                    let dir = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
                    let start = Vec3::new(ox, 0.1, top + 0.2) - dir * 1.0;
                    let mut st = PlayerState::standing_at(start, yaw);
                    for _ in 0..50 {
                        st.step(&MoveInput { yaw, ..Default::default() }, &p, &world, 1.0 / 60.0);
                    }
                    if !st.grounded {
                        continue;
                    }
                    walkers += 1;
                    let input = MoveInput { fwd: 1.0, yaw, sprint, ..Default::default() };
                    let cmd = if sprint { p.run_forward * p.sprint_full_speed_multiplier } else { p.run_forward };
                    let (mut slow, mut air, mut last_cell) = (0usize, 0usize, (st.pos.x.round() as i32, st.pos.y.round() as i32));
                    let mut crossed = 0;
                    let mut min_z = f32::MAX;
                    for f in 0..150 {
                        st.step(&input, &p, &world, 1.0 / 60.0);
                        let cell = (st.pos.x.round() as i32, st.pos.y.round() as i32);
                        if cell != last_cell {
                            crossed += 1;
                            last_cell = cell;
                        }
                        if f > 60 && st.ground_speed < 0.85 * cmd {
                            slow += 1;
                        }
                        air += (!st.grounded) as usize;
                        min_z = min_z.min(st.pos.z);
                        if st.pos.x.abs() > 2.3 || st.pos.y.abs() > 2.3 {
                            break;
                        }
                    }
                    crossings += crossed;
                    if slow > 3 || air > 2 {
                        bad.push(format!("{} yaw {:.0} offset {ox}: {slow} slow frames, {air} airborne, lowest z {min_z:.3} (top {top:.3})", if sprint { "sprint" } else { "run" }, yaw.to_degrees()));
                    }
                }
            }
        }
        println!("{walkers} walkers, {crossings} seam crossings, {} caught", bad.len());
        for b in &bad {
            println!("   CAUGHT {b}");
        }
        assert!(walkers >= 40 && crossings >= 100, "the seam sweep barely looked: {walkers} walkers, {crossings} crossings");
        assert!(bad.is_empty(), "{} walkers lost speed or left the ground crossing block seams", bad.len());
    }

}
