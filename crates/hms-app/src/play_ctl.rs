//! #play  The CHARACTER CONTROLLER: one biped's capsule, swept against the level and the user's
//! own Forge blocks.
//!
//! This is the only genuinely new mechanism test mode needs. Everything else it does — posing a
//! biped from a [`crate::film_anim::BodyState`], choosing clips, drawing first-person arms,
//! reading the collision triangles — already existed for Theater. Walking did not.
//!
//! # Nothing in here is pure physics for its own sake
//!
//! Every constant it moves by comes from [`crate::play_phys::PlayerPhysics`], i.e. from the
//! `bipd` and `matg` tags, and the module takes it as an argument rather than owning any
//! numbers. The two quantities that are NOT authored anywhere are derived rather than chosen:
//!
//! * **There is no step-up height.** Halo's character proxy is a Havok capsule and Reach
//!   authors no step field. Stepping over a low ledge is emergent: the capsule's bottom
//!   hemisphere touches a ledge EDGE on a diagonal normal, and a contact whose normal is within
//!   the biped's own `maximum slope angle` of vertical is pushed out along that full normal — so
//!   the body rides up. A contact outside that angle is pushed out HORIZONTALLY, so a wall (and
//!   a ramp too steep for this biped) stops it. The ledge height this climbs is therefore a
//!   consequence of the authored radius and the authored slope limit, and
//!   [`max_step_height`] states it in closed form so it can be measured rather than guessed.
//! * **The ground probe depth**, which is what stops a body walking downhill from hopping off
//!   the slope every frame. It is the vertical distance the fastest authored run can descend in
//!   one engine tick on the steepest walkable slope — `run_forward × tan(max_slope) / TICK_HZ`.
//!   That is the exact physical requirement, with no free parameter: any shallower and a legal
//!   descent outruns the probe; any deeper and the body would cling to a ledge it has left.
//!
//! # Shape
//!
//! [`PlayerState::step`] is pure over a [`StaticWorld`], so a floor, a ramp, a ledge and a wall
//! can be built by hand in a test and the controller driven across them with no map, no window
//! and no GPU — while the SAME function is what the live per-frame hook calls. The collision
//! primitives are the project's own (`physics_settle::closest_point_on_triangle`, the same one
//! the ragdoll solver and the object settle use), so a player lands on a Forge block for the
//! same reason a corpse does.

use glam::Vec3;

use crate::physics_settle::{closest_point_on_triangle, StaticWorld, TICK_HZ};
use crate::play_phys::PlayerPhysics;

/// How many spheres stand in for the capsule. Three (bottom, middle, top) is what
/// `ConvexShape::capsule` reduces a pill to for the settle solver, so a player is collided the
/// same way every other capsule in this codebase is.
pub const CAPSULE_SPHERES: usize = 3;

/// Depenetration passes per sub-step. More than one because pushing out of a floor can push
/// into a wall (an inside corner needs both resolved).
const SOLVER_PASSES: usize = 4;
/// Extra passes run ONLY when the ordinary ones left the body inside something with nothing to
/// stand on. See `resolve`.
const ESCAPE_PASSES: usize = 2;

/// The depenetration accumulated over one solver pass.
///
/// **Contacts do not add up.** What each one asks for is `push · n >= depth` — a *constraint*, not
/// a contribution — and summing them double-counts every surface the capsule touches more than
/// once. A floor is TWO triangles sharing a diagonal, so a body that lands on that seam is
/// reported twice with the same vertical normal and gets pushed out twice as far as it went in.
/// MEASURED: dropped from 100 wu onto a seam it came to rest at z = +0.0793 with the floor at 0,
/// floating 8 cm in the air, instead of standing on it.
///
/// So contacts whose normals agree are MERGED, keeping the deepest, and only genuinely different
/// directions still add — which is what makes an inside corner push out of both walls. This is a
/// deduplication, not a solver: it does not project onto the intersection of the contact planes,
/// and in a corner the sum still over-shoots slightly. That is the known remaining
/// approximation.
#[derive(Default)]
struct Depenetration {
    dirs: Vec<(Vec3, f32)>,
}

impl Depenetration {
    /// cos of the angle within which two normals count as the same direction (~2.6 deg). The
    /// seam case produces normals that agree to the last bit; the tolerance is only so that a
    /// floor tessellated with a slight crease still merges.
    const SAME_DIR: f32 = 0.999;

    fn add(&mut self, n: Vec3, depth: f32) {
        for (dir, d) in self.dirs.iter_mut() {
            if dir.dot(n) > Self::SAME_DIR {
                if depth > *d {
                    *d = depth;
                    *dir = n;
                }
                return;
            }
        }
        self.dirs.push((n, depth));
    }

    fn total(&self) -> Vec3 {
        self.dirs
            .iter()
            .fold(Vec3::ZERO, |acc, (n, d)| acc + *n * *d)
    }
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
    /// The surface normal underfoot (`Vec3::Z` when airborne).
    pub ground_normal: Vec3,
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
            crouched: false,
            crouch_t: 0.0,
            air_time: 0.0,
            land_time: 0.0,
            distance: 0.0,
            ground_speed: 0.0,
            sprint_time: 0.0,
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
}

/// The ledge height this capsule rides up, wu — the closed form of the emergent step-up.
///
/// The bottom sphere's centre sits `radius` above the feet. Touching the top EDGE of a ledge
/// `s` high, the contact normal's vertical component is `(radius - s) / radius`, so the contact
/// counts as walkable — and is therefore pushed out along the full normal, lifting the body —
/// exactly while
///
/// ```text
///   (radius - s) / radius >= cos(max_slope)      =>      s <= radius * (1 - cos(max_slope))
/// ```
///
/// **Not a tuning knob**: both inputs are authored in `bipd`. This exists so the number can be
/// printed and compared against what a test actually climbs.
pub fn max_step_height(p: &PlayerPhysics) -> f32 {
    p.radius * (1.0 - p.max_slope.cos())
}

/// The downward probe that keeps a body walking downhill in contact with the hill, wu.
///
/// The requirement is exact: a body running at the fastest authored speed down the steepest
/// walkable slope descends `run_forward * tan(max_slope)` wu per second, so in one engine tick
/// it descends that over [`TICK_HZ`]. The probe has to reach at least that far or a legal
/// descent leaves the ground; it must not reach much further or the body clings to a ledge it
/// has walked off. No free parameter.
pub fn ground_probe_depth(p: &PlayerPhysics) -> f32 {
    (p.run_forward * p.max_slope.tan() / TICK_HZ).max(1e-4)
}

impl PlayerState {
    /// Stand a body at `feet`, facing `yaw`, at rest.
    pub fn standing_at(feet: Vec3, yaw: f32) -> PlayerState {
        // The body starts facing where it is looking; they only diverge once the mouse moves.
        PlayerState { pos: feet, yaw, facing: yaw, ..Default::default() }
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
        let stick = Vec3::new(input.fwd, input.right, 0.0);
        let stick_len = stick.length();
        let wish_dir = if stick_len > 1e-4 { (fwd * input.fwd + right * input.right) / stick_len } else { Vec3::ZERO };
        // SPRINT is the globals' own `full speed multiplier`, not a number of this module's.
        // Claimed only for a straight-ahead run, because Reach cannot strafe or shoot sprinting.
        let sprinting = input.sprint && self.grounded && !self.crouched && !input.walk && input.fwd > 0.5 && input.right.abs() < 0.5;
        // ...and it RAMPS IN over the authored `seconds to full speed` rather than snapping: the
        // globals author both the multiplier and how long it takes to reach it.
        self.sprint_time = if sprinting { self.sprint_time + dt } else { 0.0 };
        let base_cap = p.top_speed(input.fwd, input.right, self.crouched, input.walk);
        let full_cap = p.top_speed_with(input.fwd, input.right, self.crouched, input.walk, sprinting);
        let ramp = if p.sprint_seconds_to_full > 1e-3 {
            (self.sprint_time / p.sprint_seconds_to_full).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let cap = (base_cap + (full_cap - base_cap) * ramp) * stick_len.min(1.0);

        // --- jump --------------------------------------------------------------------------
        if input.jump && self.grounded {
            self.vel.z = p.jump_velocity;
            self.grounded = false;
            self.air_time = 0.0;
            out.jumped = true;
        }

        // --- accelerate ---------------------------------------------------------------------
        let accel = if self.grounded { p.acceleration(self.crouched) } else { p.air_acceleration() };
        let mut horiz = Vec3::new(self.vel.x, self.vel.y, 0.0);
        let target_v = wish_dir * cap;
        if self.grounded {
            // On the ground the authored acceleration both speeds up and stops: a Spartan does
            // not coast. The same rate pulls the velocity toward the target, which is zero when
            // the stick is centred.
            horiz = approach(horiz, target_v, accel * dt);
        } else if stick_len > 1e-4 {
            // Air control: the same pull toward the wish velocity but at the globals' own
            // `airborne acceleration`, and it may never push the body past its ground cap — so a
            // jump cannot be used to accelerate.
            let before = horiz.length();
            horiz = approach(horiz, target_v, accel * dt);
            let after = horiz.length();
            if after > before.max(cap) && after > 1e-6 {
                horiz *= before.max(cap) / after;
            }
        }
        self.vel.x = horiz.x;
        self.vel.y = horiz.y;

        // --- gravity, and the slope the body is standing on ---------------------------------
        if self.grounded {
            // Project the intended motion onto the ground plane, so running down a slope
            // converts into motion ALONG the slope rather than a sequence of little falls. This
            // is what makes a hill feel like a hill and not a staircase.
            let n = self.ground_normal;
            if n.z > 1e-3 {
                let h = Vec3::new(self.vel.x, self.vel.y, 0.0);
                self.vel.z = -(h.x * n.x + h.y * n.y) / n.z;
            } else {
                self.vel.z = 0.0;
            }
        } else {
            self.vel.z -= p.gravity() * dt;
        }

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
        let step_cap = (p.radius * 0.5).max(1e-3);
        let travel = self.vel.length() * dt;
        let want = ((travel / step_cap).ceil() as usize).max(1);
        out.substep_guard_tripped = want > SUBSTEP_RUNAWAY_GUARD;
        let n_sub = want.min(SUBSTEP_RUNAWAY_GUARD);
        out.substeps = n_sub;
        let h = dt / n_sub as f32;
        let mut grounded_any = false;
        let mut normal_best = Vec3::Z;
        let mut best_z = -2.0f32;
        for _ in 0..n_sub {
            self.pos += self.vel * h;
            let r = self.resolve(p, world);
            out.contacts += r.contacts;
            out.steep_contacts += r.steep;
            out.tris += r.tris;
            if r.grounded && r.normal.z > best_z {
                grounded_any = true;
                best_z = r.normal.z;
                normal_best = r.normal;
            }
        }

        // --- ground state, with the derived probe for hysteresis ----------------------------
        self.grounded = grounded_any;
        self.ground_normal = if grounded_any { normal_best } else { Vec3::Z };
        // IS THIS BODY LEAVING THE GROUND? The answer is about CAUSE, and that is the whole of
        // the incline fix.
        //
        // A body leaves the ground if it JUMPED on this step, or if it was already off the ground
        // coming into it. Nothing else does: rising because the floor under you slopes upward is
        // not an ascent, it is the shape of the floor.
        //
        // **This used to compare the vertical SPEED against a tenth of the authored jump
        // velocity, and that was the user's "it jitters and freaks out when I'm walking up any
        // incline".** A body walking up a slope has an in-plane vertical velocity of
        // `speed * tan(angle)` — entirely legitimate — so the old test fired whenever
        //
        //     speed * tan(angle) > jump_velocity * 0.1
        //
        // which for a Spartan running at the authored 2.200 wu/s with a 2.300 wu/s jump is
        // **any incline past 5.97 degrees** (an Elite: 5.82). Past that angle the probe was
        // skipped, the body was called airborne, gravity pulled it off the slope, the next
        // frame's depenetration put it back, and it did that sixty times a second. It also
        // flickered the ANIMATION between the run cycle and the jump arc, because the body state
        // reads `grounded`.
        //
        // Two earlier versions of this test were both wrong in the same way — they asked "how
        // fast is it going up?" when the question is "why is it going up?". `vel.z <= 0.0` failed
        // on the solver's own push-out residue; the speed threshold failed on every real hill.
        //
        // THE PATTERN, because this is the third distinct fault in these few lines and they are
        // all the same shape: each one was **a quantity that is only correct on FLAT GROUND, used
        // on a slope** — `vel.z <= 0.0`, then a speed threshold, then (in `probe_ground`) a
        // tangent point taken from the wrong place on the surface. Flat ground is the degenerate
        // case where all three happen to be right, which is exactly why sixty-one passing gates
        // never saw any of them. Anything added here that reasons about "the ground" should be
        // checked on a slope before it is believed.
        let leaving = out.jumped || !was_grounded;
        if !self.grounded && !leaving {
            // Not touching, but is there walkable ground within the distance a legal descent
            // could have covered this tick? If so the body is still on the hill.
            if let Some((z, n)) = self.probe_ground(p, world, ground_probe_depth(p)) {
                self.pos.z = z;
                self.grounded = true;
                self.ground_normal = n;
            }
        }
        if self.grounded {
            if !was_grounded {
                out.landed = true;
                self.land_time = 0.0;
            } else {
                self.land_time += dt;
            }
            self.air_time = 0.0;
            // Only a LANDING kills the fall. A body that was already walking keeps whatever
            // vertical velocity its own ground plane gives it — on a descent that is NEGATIVE and
            // correct, and zeroing it every grounded frame made the body walk off the slope
            // horizontally and drop back onto it, over and over, which is the same jitter from
            // the other direction. Nothing accumulates: the ground-plane projection at the top of
            // the next step is the only writer of `vel.z` while grounded, so it recomputes this
            // from the horizontal velocity regardless.
            if out.landed && self.vel.z < 0.0 {
                self.vel.z = 0.0;
            }
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

    /// Push the capsule out of anything it is inside, cancel the velocity into each contact, and
    /// report what it touched.
    ///
    /// The slope limit lives here, and it is the whole of "a steep ramp is not climbable": a
    /// contact whose normal is within the authored `maximum slope angle` of straight up is
    /// walkable and is pushed out along its full normal (which is also what carries the body up
    /// a low ledge); anything steeper is pushed out HORIZONTALLY, so it blocks without lifting.
    fn resolve(&mut self, p: &PlayerPhysics, world: &dyn StaticWorld) -> Resolved {
        let mut res = Resolved::default();
        let r = p.radius.max(1e-4);
        let height = self.height(p);
        let min_normal_z = p.max_slope.cos();
        let max_step = max_step_height(p);
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        let pad = Vec3::splat(r + 0.05);
        let lo = Vec3::new(self.pos.x, self.pos.y, self.pos.z) - pad;
        let hi = Vec3::new(self.pos.x, self.pos.y, self.pos.z + height) + pad;
        world.gather(lo, hi, &mut tris);
        res.tris = tris.len();
        if tris.is_empty() {
            return res;
        }
        // THE PASSES, and why the last ones are different.
        //
        // The ordinary passes answer "what is this body touching and what may it stand on". The
        // ESCAPE passes answer a question that only arises when those have failed: the body is
        // inside a surface, and the rules above have decided it is not something it could climb
        // onto, so there is no push and no ground and it stays there for ever. They run LAST and
        // only while nothing has been found to stand on — see `escaping` below for why that
        // condition, and not the geometry, is what separates the two cases.
        let mut pass = 0usize;
        while pass < SOLVER_PASSES + ESCAPE_PASSES {
            let escape = pass >= SOLVER_PASSES;
            if escape && res.grounded {
                break;
            }
            let mut push = Depenetration::default();
            let mut hits = 0usize;
            let centres = Self::spheres_at(self.pos, height, r);
            for (i, c) in centres.iter().enumerate() {
                for t in &tris {
                    let Some((n, depth, interior)) = sphere_tri_normal(*c, r, t) else { continue };
                    hits += 1;
                    // WHERE on the capsule the contact is, measured from the feet. This, not the
                    // normal, is what classifies it — see the comment below.
                    let (q, _) = closest_point_on_triangle(*c, t[0], t[1], t[2]);
                    let contact_rise = q.z - self.pos.z;
                    // THE SLOPE LIMIT, and with it the step-up. ONE rule, and it is a rule about
                    // WHERE THE CONTACT IS rather than about the direction of the normal:
                    //
                    //   a contact on the bottom sphere is something this biped can stand on
                    //   exactly while it is no higher above the feet than `max_step_height`.
                    //
                    // That is not a second law bolted beside the slope limit, it IS the slope
                    // limit. A capsule resting on a plane of slope `theta` touches it at
                    // `radius * (1 - cos theta)` above the feet, so "no higher than
                    // `radius * (1 - cos(max_slope))`" is precisely "no steeper than
                    // `max_slope`" for a plane — and it extends to a ledge EDGE, which has no
                    // slope of its own, in the only way consistent with it.
                    //
                    // **Why not the normal.** Classifying by `n.z` is correct only at first
                    // contact: once the body has penetrated, `n.z = (radius - rise) / d` with
                    // `d < radius`, so the same ledge reads as a floor or a wall depending on how
                    // deep the body happens to be — which depends on where the frames landed.
                    // MEASURED: with the normal rule, ledges between 0.06 and 0.155 wu were
                    // climbed or refused according to the approach, i.e. the same curb was
                    // sometimes steppable and sometimes not. The height rule has no such freedom:
                    // `q.z` is where the surface is.
                    //
                    // The three cases:
                    //   * low enough on the bottom sphere -> FLOOR: pushed out along its full
                    //     normal, which is also what lifts the body over a low ledge;
                    //   * a normal that points DOWN or dead sideways -> a ceiling or a wall:
                    //     pushed out along its full normal too, which cannot lift anything;
                    //   * anything else -> FLATTENED: pushed out horizontally only, so a ramp
                    //     past this biped's limit, and a ledge too tall to step onto, block
                    //     without ever carrying the body upward.
                    // A FACE contact is classified by its NORMAL; an EDGE or VERTEX contact by
                    // the HEIGHT of the contact above the feet. That split is the whole of it,
                    // and getting it wrong cost a body stuck inside the floor:
                    //
                    // The height rule exists to make the step-up deterministic, and an edge has no
                    // normal of its own worth trusting (it depends on penetration depth). But
                    // applied to a FACE it says something false. A body that lands hard penetrates
                    // the floor by more than `max_step` — 0.0513 wu for this capsule — and the
                    // height rule then reads the floor underneath it as "a ledge too tall to step
                    // onto", flattens the push-out to horizontal, which for a normal of (0,0,1) is
                    // ZERO, and never marks it grounded. **The body sits inside the floor for
                    // ever, neither pushed out nor standing on it.** Measured: dropped 100 wu, it
                    // came to rest at z = -0.08 and never grounded.
                    //
                    // A face's normal is exact, so for a face the question "can I stand on this"
                    // is exactly "is the normal within the slope limit", which is also the
                    // original rule and the one the slope gate was written against.
                    // A DEEP PENETRATION IS AN ESCAPE, NOT A STEP-UP, and this is the case that
                    // left a body stuck 8 cm inside a floor for 400 frames.
                    //
                    // The height rule asks "could I climb onto this?", which is the wrong question
                    // when the body is already INSIDE the surface: `contact_rise` is measured from
                    // the feet, and a penetration pushes the feet DOWN, so the rise inflates and
                    // the floor underneath reads as a ledge too tall to climb. The push is then
                    // flattened to horizontal — zero, for a normal of (0,0,1) — so there is no
                    // push and no ground. MEASURED: dropped 100 wu onto a floor, came to rest at
                    // z = -0.08 and never grounded.
                    //
                    // It is not only hard landings, and it is not only faces: the body lands on a
                    // triangle SEAM as often as not (a floor is two triangles and their shared
                    // diagonal is a real edge), so the edge branch sees it too — which is why
                    // fixing face-versus-edge alone did not clear it.
                    //
                    // So getting OUT has to take precedence over deciding whether the surface
                    // could have been climbed — but ONLY when there is nothing to stand on, and
                    // that condition is doing the real work. A first attempt made deep
                    // penetration of an upward-facing surface steppable unconditionally, on the
                    // reasoning that a wall's normal would fail the test. It does, but a LEDGE's
                    // rim does not: pressed against a curb 0.031 wu tall, the contact with its
                    // rim has `n = (0,0,1)` and `depth = 0.031` once the body is directly under
                    // it, which is geometrically indistinguishable from standing 0.031 wu inside
                    // a floor. MEASURED: it climbed a ledge a 15 deg slope limit must refuse.
                    //
                    // The two cases are not told apart by the contact at all. They are told apart
                    // by the rest of the body's situation: a body at a curb is ALREADY STANDING
                    // on the floor the curb rises from, and a body stuck inside a floor is
                    // standing on nothing. So the escape is a FALLBACK PASS -- it runs after the
                    // ordinary passes and only while `res.grounded` is still false. The curb case
                    // grounds on the floor in the first pass and never reaches it; the stuck case
                    // never grounds and always does.
                    let escaping = escape && i == 0 && depth > max_step && n.z >= min_normal_z;
                    let steppable = i == 0
                        && (escaping
                            || if interior { n.z >= min_normal_z } else { contact_rise <= max_step + 1e-5 });
                    let floor_like = steppable || (i > 0 && n.z >= min_normal_z);
                    let too_steep_to_climb = !floor_like && n.z > 0.0;
                    // The normal the POSITION and the VELOCITY are both resolved along. It is the
                    // contact normal except on a surface too steep to climb, where it is that
                    // normal flattened — see the measurement below.
                    let mut eff_n = n;
                    if floor_like {
                        // Ground is only claimed from the BOTTOM sphere: a floor-shaped normal on
                        // the head is an overhang the body is wedged under, not something it
                        // stands on.
                        if steppable {
                            res.grounded = true;
                            if n.z > res.normal.z || res.normal == Vec3::ZERO {
                                res.normal = n;
                            }
                        }
                        push.add(n, depth);
                    } else if too_steep_to_climb {
                        res.steep += 1;
                        let flat = Vec3::new(n.x, n.y, 0.0);
                        if flat.length_squared() > 1e-8 {
                            let flat = flat.normalize();
                            push.add(flat, depth);
                            // THE VELOCITY MUST BE CLIPPED AGAINST THE SAME FLATTENED NORMAL the
                            // position was pushed along, and this is not a detail.
                            //
                            // Clipping against the full normal instead removes the velocity going
                            // into the surface and leaves the component ALONG it — which on an
                            // upward-pointing normal points UP. MEASURED: a body running at 4.1
                            // wu/s into the edge of a ledge 50% past its step height had 3.23
                            // wu/s removed along a normal of (-0.79, 0, 0.62), which handed it
                            // **+2.0 wu/s straight up** and launched it 0.18 wu into the air —
                            // over the ledge it was supposed to be stopped by, landing on top and
                            // walking on. The two halves of the solver have to agree about what
                            // the surface is: a surface too steep to climb blocks horizontally,
                            // so it takes horizontal velocity and gives nothing back.
                            eff_n = flat;
                        }
                        // A normal with no horizontal part at all cannot be "too steep to
                        // climb" (it is straight up or straight down), so there is no third case.
                    } else {
                        // Walls (n.z == 0) and ceilings / overhangs (n.z < 0): out along the
                        // normal, which is horizontal for a wall and downward for a ceiling.
                        push.add(n, depth);
                    }
                    // Remove the velocity going INTO the surface — along the SAME normal the
                    // position was resolved with (see above).
                    let vn = self.vel.dot(eff_n);
                    if vn < 0.0 {
                        self.vel -= eff_n * vn;
                    }
                }
            }
            res.contacts = res.contacts.max(hits);
            let delta = push.total();
            if hits == 0 {
                break;
            }
            if delta.length_squared() < 1e-12 {
                if escape || res.grounded {
                    break;
                }
                // Nothing ordinary left to resolve and nothing to stand on: this is the stuck
                // case, so go straight to the escape rather than spending the remaining passes
                // recomputing the same nil push.
                pass = SOLVER_PASSES;
                continue;
            }
            self.pos += delta;
            pass += 1;
        }
        if res.grounded && res.normal == Vec3::ZERO {
            res.normal = Vec3::Z;
        }
        res
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
                if sphere_tri_normal(*c, r, t).is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// Walkable ground within `depth` below the feet: its z and its normal.
    fn probe_ground(&self, p: &PlayerPhysics, world: &dyn StaticWorld, depth: f32) -> Option<(f32, Vec3)> {
        let r = p.radius.max(1e-4);
        let min_normal_z = p.max_slope.cos();
        let mut tris: Vec<[Vec3; 3]> = Vec::new();
        let pad = Vec3::splat(r + 0.05);
        world.gather(
            Vec3::new(self.pos.x, self.pos.y, self.pos.z - depth) - pad,
            Vec3::new(self.pos.x, self.pos.y, self.pos.z + r) + pad,
            &mut tris,
        );
        // Sink the bottom sphere by `depth` and see what it would then be inside.
        let c = Vec3::new(self.pos.x, self.pos.y, self.pos.z + r - depth);
        let mut best: Option<(f32, Vec3)> = None;
        for t in &tris {
            let Some((n, _, interior)) = sphere_tri_normal(c, r, t) else { continue };
            if n.z < min_normal_z {
                continue;
            }
            let (q, _) = closest_point_on_triangle(c, t[0], t[1], t[2]);
            // WHERE THE FEET REST ON THIS SURFACE, under the body's own axis.
            //
            // A FACE contact has a plane, so the height under the body is read off that plane and
            // the feet sit `r*(sec - 1)` above it (see the long note in the jitter fix). An EDGE
            // or VERTEX contact **has no plane**, and pretending it does is wrong in a way that
            // matters at every crease: at the foot of a ramp the ramp's leading EDGE produced a
            // fictional plane sloping up under the body, which placed the feet ABOVE where the
            // body already was. For an edge, the exact answer is the tangent-sphere one — the
            // centre is `q + n*r` and the feet are `r` below it.
            let feet_z = if interior {
                let plane_z = q.z - (n.x * (self.pos.x - q.x) + n.y * (self.pos.y - q.y)) / n.z;
                plane_z + r * (1.0 / n.z - 1.0)
            } else {
                q.z + n.z * r - r
            };
            // ACCEPT, THEN COMPARE. Choosing the highest candidate first and filtering afterwards
            // let ONE unacceptable candidate mask a perfectly good one: at the crease the ramp's
            // edge scored higher than the floor directly underfoot, was then rejected for being
            // above the body, and the floor never got a look in — so the body was called airborne
            // while standing on it. MEASURED: exactly one frame in 180, at x = -0.063, which is
            // the crease.
            if feet_z > self.pos.z + 1e-4 || feet_z < self.pos.z - depth - 1e-4 {
                continue;
            }
            if best.map(|(z, _)| feet_z > z).unwrap_or(true) {
                best = Some((feet_z, n));
            }
        }
        best
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

#[derive(Default)]
struct Resolved {
    contacts: usize,
    steep: usize,
    tris: usize,
    grounded: bool,
    normal: Vec3,
}

/// The contact normal of a sphere against one triangle, and the penetration depth.
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
fn sphere_tri_normal(c: Vec3, r: f32, t: &[Vec3; 3]) -> Option<(Vec3, f32, bool)> {
    let (q, interior) = closest_point_on_triangle(c, t[0], t[1], t[2]);
    let delta = c - q;
    let d = delta.length();
    if d >= r {
        return None;
    }
    let face = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
    let n = if interior && face != Vec3::ZERO {
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
    Some((n, r - d, interior))
}

/// Move `v` toward `target` by at most `max_delta`.
fn approach(v: Vec3, target: Vec3, max_delta: f32) -> Vec3 {
    let d = target - v;
    let len = d.length();
    if len <= max_delta || len < 1e-9 {
        target
    } else {
        v + d * (max_delta / len)
    }
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

    /// Walk a body at a ledge `h` high whose face is at x = 1, for `secs`. The floor and the
    /// ledge both run far enough that a body which DOES climb cannot run off the end of the world
    /// and land somewhere misleading — the first version of this harness did exactly that, and the
    /// sweep read "blocked" for every ledge it had actually climbed.
    fn ledge_run(p: &PlayerPhysics, h: f32, secs: f32) -> (PlayerState, Vec<StepOut>) {
        let mut w = floor(0.0, 60.0);
        w.extend(block(Vec3::new(1.0, -8.0, -1.0), Vec3::new(50.0, 8.0, h)));
        let mut st = PlayerState::standing_at(Vec3::new(-0.5, 0.0, 0.2), 0.0);
        settle(&mut st, p, &w);
        let outs = run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, p, &w, secs);
        (st, outs)
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

    #[test]
    #[ignore]
    fn trace_one_ledge() {
        let p = phys();
        let h = max_step_height(&p) * 1.5;
        let mut w = floor(0.0, 60.0);
        w.extend(block(Vec3::new(1.0, -8.0, -1.0), Vec3::new(50.0, 8.0, h)));
        let mut st = PlayerState::standing_at(Vec3::new(-0.5, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        println!("ledge {h:.4}  max_step {:.4}  after settle z={:.6}", max_step_height(&p), st.pos.z);
        for i in 0..90 {
            let o = st.step(&MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 1.0 / 60.0);
            if i % 2 == 0 || (st.pos.x > 0.6 && st.pos.x < 1.6) {
                println!("f{i:02} x={:.4} z={:.5} g={} contacts={} steep={}", st.pos.x, st.pos.z, st.grounded, o.contacts, o.steep_contacts);
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // #play-gate  THE INVERSE GATES: each one proves the gate above it BITES.
    //
    // A gate that has only ever been seen green proves nothing, so each behaviour is also
    // exercised with the authored value CHANGED — and because every one of these behaviours is
    // driven by `PlayerPhysics`, which the test constructs, the sabotage is in the DATA rather
    // than in the code. That matters practically as well as philosophically: a code sabotage
    // costs a full recompile of a very large crate each time, and on a tree shared by five
    // agents that is an hour of a contended build lock; these cost nothing and they run in the
    // same build as the gates they guard.
    //
    // What they prove: that the gate's subject is genuinely driven by the authored value, and
    // that the assertion responds when it changes. What they do NOT prove: that the code path
    // could not be deleted outright — two of those were proven separately by editing the source
    // and watching the matching gate go red (the crouch transition rate, and the eye's crouch
    // interpolation).
    //
    // THE SOURCE-LEVEL SABOTAGES still owed, and the exact edit for each, so the next session can
    // run them without re-deriving them. Each one must turn the named gate red:
    //
    //   1. ~~`resolve`: make the escape pass unconditional (`if escape { break }` instead of
    //      `if escape && res.grounded`) ->
    //      `a_falling_body_does_not_tunnel_a_thin_platform_on_a_hitched_frame`.~~
    //      **RUN 2026-10-14: STAYED GREEN, and the prediction was wrong rather than the gate.**
    //      That edit disables the escape passes entirely (the break happens before any of their
    //      work), and the tunnelling fixture never needs them: a body landing on `floor()` gets a
    //      FACE contact, which is steppable on its normal alone with no depth test, so it grounds
    //      in the first ordinary pass. See
    //      `a_body_that_lands_inside_an_edge_contact_is_pushed_back_out`, which went looking for a
    //      fixture that does need the escape and establishes that **there isn't one**: the
    //      sub-step cap bounds penetration to half a capsule radius and the solver resolves inside
    //      the sub-step, so the deep-penetration state cannot be produced from outside. The escape
    //      pass is uncovered belt-and-braces, and that is written down there as a known gap.
    //   2. `Depenetration::add`: never merge (`if false && dir.dot(n) > SAME_DIR`) ->
    //      `contacts_in_the_same_direction_merge_instead_of_adding` and the tunnelling gate.
    //   3. `step`: turn instantly (`let step = residual.abs();`) ->
    //      `the_body_turns_toward_the_aim_at_the_authored_rate`.
    //   4. `step`: ignore the threshold (`let turn = true;`) ->
    //      `a_standing_body_holds_its_feet_until_the_aim_passes_the_authored_threshold`.
    //   5. `body_state`: measure travel against the aim (`st.yaw` for `st.facing`) ->
    //      `the_travel_direction_is_measured_against_the_body_not_the_camera`.
    //   6. `step`: drop the wrap (`self.yaw = input.yaw;`) ->
    //      `the_look_yaw_stays_in_range_however_far_the_player_spins`.
    //
    // **SABOTAGES 2 TO 6 WERE RUN 2026-10-14 AND ALL FIVE TURNED THEIR NAMED GATE RED**, each one
    // naming the actual defect, in a `rsync` snapshot with its own target directory so the shared
    // tree was never sabotaged. Baseline 35 passed / 0 failed; each sabotage 0 passed / 1 failed.
    // Only #1 above did not bite, for the reason recorded against it.
    //
    // **And one warning about the harness, which cost a whole pass.** A sabotage runner that
    // reads `cargo test`'s failure list and treats an EMPTY list as green will report every
    // sabotage as "stayed green" the moment a run stops producing a result line at all — a
    // compile error, or a kill. It happened: the baseline printed "NO RESULT LINE, failed=[]"
    // and the pass carried on. A missing result line has to be a hard error, not a quiet zero.
    // (The kill was self-inflicted: the runner's `cargo test` command line was byte-identical to
    // a stale run of mine, so killing the stale one by PID took the baseline with it.)
    // -----------------------------------------------------------------------------------------

    /// #play-gate  INVERSE of the slope gate: a biped whose authored limit is nearly vertical
    /// climbs the ramp the Spartan's 50 degrees refuses. If this passed with the slope rule
    /// removed, the slope gate would be passing for the wrong reason.
    #[test]
    fn a_kinder_authored_slope_limit_makes_the_same_ramp_climbable() {
        let p = phys();
        let steep = p.max_slope.to_degrees() + 20.0;
        let mut w = floor(0.0, 6.0);
        w.extend(ramp(steep, 20.0, 6.0));
        // As authored: refused.
        let mut a = PlayerState::standing_at(Vec3::new(-1.0, 0.0, 0.2), 0.0);
        settle(&mut a, &p, &w);
        run(&mut a, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 3.0);
        assert!(a.pos.z < 0.15, "the authored limit should refuse a {steep:.0} deg ramp, got z {:.3}", a.pos.z);
        // With a kinder authored limit: climbed. Only the TAG VALUE differs.
        let mut q = p;
        q.max_slope = (steep + 10.0).to_radians();
        let mut b = PlayerState::standing_at(Vec3::new(-1.0, 0.0, 0.2), 0.0);
        settle(&mut b, &q, &w);
        run(&mut b, &MoveInput { fwd: 1.0, ..Default::default() }, &q, &w, 3.0);
        assert!(
            b.pos.z > 0.5,
            "a biped whose authored limit is {:.0} deg must climb a {steep:.0} deg ramp, got z {:.3}",
            q.max_slope.to_degrees(),
            b.pos.z
        );
    }

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

    /// #play-gate  Contacts that point the same way are ONE constraint, not two.
    ///
    /// This is the rule in the small, where the arithmetic is visible: a floor is two triangles,
    /// a body landing on their seam is reported twice with the same normal, and if those add the
    /// body is pushed out twice as far as it went in. Measured as a floating body at z = +0.0793;
    /// here it is a single assertion.
    #[test]
    fn contacts_in_the_same_direction_merge_instead_of_adding() {
        let up = Vec3::Z;
        let mut d = Depenetration::default();
        d.add(up, 0.08);
        d.add(up, 0.08);
        assert!((d.total().z - 0.08).abs() < 1e-6, "a seam must push out once, got {:?}", d.total());
        // The deepest of the merged set wins: the shallower report must not shorten the push.
        let mut d = Depenetration::default();
        d.add(up, 0.02);
        d.add(up, 0.08);
        d.add(up, 0.05);
        assert!((d.total().z - 0.08).abs() < 1e-6, "the deepest must win, got {:?}", d.total());
        // Genuinely different directions DO add -- that is what gets a body out of a corner.
        let mut d = Depenetration::default();
        d.add(Vec3::X, 0.03);
        d.add(Vec3::Y, 0.04);
        let t = d.total();
        assert!((t.x - 0.03).abs() < 1e-6 && (t.y - 0.04).abs() < 1e-6, "a corner must push out of both walls, got {t:?}");
    }

    /// #play-gate  INVERSE of the step-up gate: a bigger authored RADIUS steps higher, and a
    /// harsher authored slope limit steps lower. The step-up has no constant of its own, so this
    /// is the whole of "it is the authored pair that decides".
    #[test]
    fn the_step_up_follows_the_authored_radius_and_slope_limit() {
        let p = phys();
        let climbs = |q: &PlayerPhysics, h: f32| {
            let (st, _) = ledge_run(q, h, 2.0);
            st.pos.z > h - 0.02 && st.pos.x > 1.2
        };
        let base = max_step_height(&p);
        // Twice the radius: twice the step, and a ledge the ordinary capsule refuses is climbed.
        let mut big = p;
        big.radius = p.radius * 2.0;
        assert!((max_step_height(&big) - base * 2.0).abs() < 1e-6);
        let over = base * 1.5;
        assert!(!climbs(&p, over), "the authored capsule must refuse a {over:.4} wu ledge");
        assert!(climbs(&big, over), "a capsule of twice the authored radius must climb it");
        // A harsher slope limit: a lower step, and a ledge the ordinary capsule climbs is refused.
        let mut harsh = p;
        harsh.max_slope = 15f32.to_radians();
        assert!(max_step_height(&harsh) < base * 0.5);
        let under = base * 0.5;
        assert!(climbs(&p, under), "the authored capsule must climb a {under:.4} wu ledge");
        assert!(!climbs(&harsh, under), "a 15 deg slope limit must refuse it");
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

    /// #play  A MEASUREMENT, not a gate: sweep the ledge height and report the tallest one the
    /// capsule actually walks onto, beside the closed form. Run with
    /// `cargo test -p hms-app --profile quick -- play_ctl::tests::measure_step_up --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measure_step_up() {
        let p = phys();
        println!("radius {:.4}  max_slope {:.2} deg  closed form {:.4} wu", p.radius, p.max_slope.to_degrees(), max_step_height(&p));
        let climbs = |h: f32| -> (bool, f32, f32) {
            let (st, _) = ledge_run(&p, h, 2.0);
            (st.pos.z > h - 0.02 && st.pos.x > 1.2, st.pos.x, st.pos.z)
        };
        let mut best = 0.0f32;
        let mut h = 0.005f32;
        while h < 0.30 {
            let (ok, x, z) = climbs(h);
            println!("ledge {h:.4} -> {} (ended x={x:.3} z={z:.4})", if ok { "CLIMBED" } else { "blocked" });
            if ok { best = best.max(h); }
            h += 0.005;
        }
        println!("MEASURED max step {best:.4} wu vs closed form {:.4} wu", max_step_height(&p));
    }

    /// #play  The contact normals at a ledge, printed — the evidence behind the closed form.
    #[test]
    #[ignore]
    fn measure_ledge_normals() {
        let p = phys();
        let r = p.radius;
        for s in [0.01f32, 0.03, 0.06, 0.1, 0.15, 0.2] {
            // The bottom sphere's centre sits `r` above the feet; touching the ledge's top EDGE
            // the normal's vertical part is (r - s) / d, and at first contact d == r.
            let nz = ((r - s) / r).max(-1.0);
            println!("ledge {s:.3}: n.z at first contact {nz:.4}  (limit cos {:.4}) -> {}",
                p.max_slope.cos(), if nz >= p.max_slope.cos() { "walkable" } else { "WALL" });
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
    #[test]
    /// #play-gate  **AN EDGE-ONLY LANDING SETTLES ON THE SURFACE.** A single triangle, landed on
    /// beside its hypotenuse so the nearest point is the EDGE and no adjoining face can supply a
    /// contact — the fixture the floor pair cannot produce.
    ///
    /// # What this gate is NOT, which is the point of writing it down
    ///
    /// It was written to cover the ESCAPE PASS (`escaping` in [`PlayerState::resolve`]), and **it
    /// does not reach it.** The chain, measured rather than reasoned:
    ///
    /// 1. The note above predicted that disabling the escape passes (`if escape { break }`, which
    ///    breaks before any of their work) would redden
    ///    `a_falling_body_does_not_tunnel_a_thin_platform_on_a_hitched_frame`. Run, it **stayed
    ///    green**: a body landing on `floor()` gets a FACE contact, and `steppable` for a face is
    ///    `n.z >= min_normal_z` with no depth test at all, so it grounds on the first ordinary
    ///    pass. Even the "on the triangle seam" cases ground through the two adjoining faces.
    /// 2. So this fixture removed the second triangle to force an edge contact — where the height
    ///    rule applies and a penetration deeper than `max_step_height` (0.0625 wu here) would
    ///    read as "a ledge too tall to climb", flatten the push to zero, and strand the body.
    /// 3. **It still never penetrates.** Dropped from 30, 100 and 300 wu on 100 ms frames the
    ///    feet never go below z = 0.0000: the sub-step cap bounds a sub-step to half a capsule
    ///    radius and the solver resolves within that sub-step, so a penetration deeper than
    ///    `max_step` cannot be produced from outside at all.
    ///
    /// **Conclusion: the escape pass is unreachable through the public API as the solver now
    /// stands, and it has no coverage.** The deep-penetration bug it was written for was
    /// separately cured by the sub-step cap and by the face-versus-edge split, and the escape
    /// remains as belt and braces. That is recorded as a KNOWN GAP rather than papered over with
    /// a gate that passes for the wrong reason — and it is a real one: if the sub-step cap is ever
    /// loosened, the escape becomes live code again with nothing watching it.
    ///
    /// What this gate does prove, which is worth having on its own: an edge-only landing comes to
    /// rest ON the surface and claims ground, at three drop heights, on hitched frames.
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
                 final z {:.4}, grounded {} (deepest {deepest:.4}) -- a body that penetrates an \
                 upward-facing surface deeper than its {:.4} wu step height has no push and no \
                 ground unless the ESCAPE pass runs",
                st.pos.z,
                st.grounded,
                max_step_height(&p)
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
            let w = downramp(deg, 30.0, 6.0, 8.0);
            let mut st = PlayerState::standing_at(Vec3::new(-3.0, 0.0, 0.3), 0.0);
            settle(&mut st, &p, &w);
            assert!(st.grounded, "{deg} deg: never settled");
            let input = MoveInput { fwd: 1.0, ..Default::default() };
            run(&mut st, &input, &p, &w, 1.5);
            assert!(st.pos.x > 0.5, "{deg} deg: did not reach the descent");
            // Steady state, as uphill.
            run(&mut st, &input, &p, &w, 0.5);
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
        let limit = p.max_slope.to_degrees();
        for deg in [6.0f32, 10.0, 15.0, 25.0, 35.0, limit - 2.0] {
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

    /// #play-gate  A SLOPE PAST THE AUTHORED LIMIT IS NOT CLIMBABLE, and one inside it is. The
    /// limit is `bipd` `maximum slope angle`, so this gate moves if the tag moves.
    #[test]
    fn a_slope_past_the_authored_limit_is_not_climbable() {
        let p = phys();
        let limit = p.max_slope.to_degrees();
        // Inside the limit: the body climbs.
        let gentle = limit - 15.0;
        let mut w = floor(0.0, 6.0);
        w.extend(ramp(gentle, 20.0, 6.0));
        let mut st = PlayerState::standing_at(Vec3::new(-1.0, 0.0, 0.2), 0.0);
        settle(&mut st, &p, &w);
        run(&mut st, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w, 3.0);
        assert!(
            st.pos.z > 0.5,
            "could not climb a {gentle:.0} deg ramp with a {limit:.0} deg limit: {}",
            st.describe(&p)
        );
        // Past the limit: it cannot.
        let steep = limit + 20.0;
        let mut w2 = floor(0.0, 6.0);
        w2.extend(ramp(steep, 20.0, 6.0));
        let mut st2 = PlayerState::standing_at(Vec3::new(-1.0, 0.0, 0.2), 0.0);
        settle(&mut st2, &p, &w2);
        let outs = run(&mut st2, &MoveInput { fwd: 1.0, ..Default::default() }, &p, &w2, 3.0);
        assert!(
            st2.pos.z < 0.15,
            "climbed a {steep:.0} deg ramp past a {limit:.0} deg limit to z {:.3}: {}",
            st2.pos.z,
            st2.describe(&p)
        );
        assert!(
            outs.iter().any(|o| o.steep_contacts > 0),
            "the slope limit never even fired, so this gate proves nothing"
        );
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

    /// #play-gate  THE STEP-UP IS THE EMERGENT ONE, AND IT IS THE PREDICTED SIZE.
    ///
    /// No step constant exists to tune, so this measures what the capsule plus the slope limit
    /// actually climbs and checks it against [`max_step_height`]'s closed form. A ledge inside
    /// that height is walked onto; one well past it is not.
    #[test]
    fn the_step_up_is_the_capsules_own_and_matches_the_closed_form() {
        let p = phys();
        let predicted = max_step_height(&p);
        assert!(predicted > 0.0 && predicted < p.radius, "closed form {predicted} is not between 0 and the radius");
        let climbs = |h: f32| -> bool {
            let (st, _) = ledge_run(&p, h, 2.0);
            st.pos.z > h - 0.02 && st.pos.x > 1.2
        };
        // Inside the predicted height: walked onto. Checked at several heights up to the limit,
        // because the point of the height rule is that this is DETERMINISTIC — under the normal
        // rule it depended on where the frames landed.
        for k in [0.1f32, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let low = predicted * k;
            assert!(climbs(low), "could not step onto a {low:.4} wu ledge, under the predicted {predicted:.4}");
        }
        // Well past it: never walked onto.
        for k in [1.5f32, 2.0, 3.0, 5.0] {
            let high = predicted * k;
            assert!(!climbs(high), "stepped onto a {high:.4} wu ledge, past the predicted {predicted:.4}");
        }
        // And the MEASURED limit, by bisection, against the closed form. This is the gate that
        // matters, because it is the number rather than a side of it.
        //
        // The two do not agree exactly and the reason is physical rather than a fudge: the body
        // rests a few thousandths of a world unit above the floor (the depenetration resolves to
        // touching, not to interpenetrating), so the rise it measures to a ledge is that much
        // smaller and it steps that much higher. 0.01 wu is 3 mm at Halo's scale.
        let (mut lo, mut hi) = (0.0f32, predicted * 3.0);
        for _ in 0..24 {
            let mid = (lo + hi) * 0.5;
            if climbs(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        assert!(
            (lo - predicted).abs() < 0.01,
            "MEASURED step-up {lo:.4} wu vs the closed form {predicted:.4} wu — they must agree to within 0.01 wu"
        );
        // And the limit is the AUTHORED pair's, not a number of this module's: a biped with a
        // bigger radius or a kinder slope limit steps higher, and nothing here can be tuned
        // without changing one of those two.
        let mut q = phys();
        q.radius *= 2.0;
        assert!((max_step_height(&q) - predicted * 2.0).abs() < 1e-6);
    }

    /// #play-gate  WALKING DOWN A SLOPE STAYS ON THE SLOPE. Without the derived ground probe the
    /// body leaves the hill every frame and the animation flickers between running and airborne.
    #[test]
    fn running_downhill_never_leaves_the_hill() {
        let p = phys();
        let deg = p.max_slope.to_degrees() - 10.0;
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
}
