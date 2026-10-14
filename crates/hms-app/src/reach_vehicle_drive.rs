//! #vehicle-drivetrain #vehicle-suspension  The ENGINE's own drivetrain and ground-contact law,
//! disassembled rather than guessed.
//!
//! This module is a child of [`crate::reach_vehicle`] and holds the two things that module used
//! to state it did not do:
//!
//! * **the drivetrain** — `engine torque -> gearbox -> wheel surface speed`, including the gear
//!   selection, the clutch and the two torque curves;
//! * **the ground contact** — where each `friction point` touches, how far its spring is
//!   compressed, and the hull attitude that follows.
//!
//! # Provenance: `reach_tag_test.exe`, which carries the whole engine WITH symbols
//!
//! The shipped `haloreach.dll` is stripped, but the MCC tool build is not: it carries the
//! original source file names in its asserts, and those name the compilation units exactly —
//! `omaha\units\vehicle_engine.cpp`, `omaha\units\vehicle_type_human_jeep.cpp`,
//! `omaha\physics\havok_vehicle_physics.cpp`. Every law below is quoted against the function and
//! the assert line that proves it. Where the engine's own variable name appears in an assert
//! string it is used verbatim (`cruising_to_loaded_transition`, `clutch_in_ticks`,
//! `clutch_shift_ticks`, `desired_gear`, `engine_definition->engine_moment`).
//!
//! ```text
//!   vehicle_engine.cpp        sub_140E5FE40  requested_gear()       (shift thresholds)
//!                             sub_140E60170  engine reset           (gear = -1, desired = -2)
//!                             sub_140E601B0  update()               (clutch + flywheel)
//!                             sub_140E60850  gear_ratio()           ( gear_block + 0x38 )
//!                             sub_140E60890  torque()               (lerp cruising<->loaded)
//!                             sub_140E60980  curve eval             (u = aav / max_aav, 0..1)
//!                             sub_140E600E0  wheel surface speed    ( ratio * aav )
//!   vehicle_type_human_jeep.cpp sub_140E69D40 the jeep's own tick
//!   havok_vehicle_physics.cpp sub_1406CC8C0  the per-wheel contact solve
//!                             sub_1406CF0F0  force/torque accumulation (iteration_count)
//! ```
//!
//! # What is MEASURED and what is INFERRED
//!
//! Measured (quoted above, byte-for-byte from the decompilation):
//!
//! * the gear selection predicate, both shift thresholds and both shift timers;
//! * the clutch: `max(ceil(0.05 * 30), 1) = 2` ticks, during which **the throttle is forced to
//!   zero** (`if (clutch_in_ticks != 0) throttle = 0`);
//! * torque `= (1 - t) * cruising + t * loaded` with `t = cruising_to_loaded_transition`, and `t`
//!   relaxing toward `|throttle|` with the time constant `0.022 s`;
//! * the torque curve's input is `angular_velocity / engine max angular velocity`, clamped
//!   `0..1`;
//! * the flywheel: `aav += dt * (torque + ratio * ground_torque) / engine_moment`, then clamped
//!   to `[0.05 * max, 1.3 * max]`; in neutral `aav *= 0.95` per tick instead;
//! * **`gear_ratio * angular_velocity` IS the wheel's surface speed in wu/s** — the jeep adds
//!   `that * dt` to a rolling phase and wraps it at the authored `wheel circumferance`
//!   (`sub_140E69D40`: `fmodf(phase, *((float *)jeep_block + 20))`), so the quantity is a
//!   distance rate and the circumference is only the wrap length. The pre-existing
//!   `top_speed_ceiling` multiplied by the circumference as well, which was a factor of 1.249 too
//!   fast on a Warthog;
//! * gear 0 is REVERSE, gear >= 1 forward, -1 neutral, -2 "none" — which is why the Warthog
//!   authors `-0.50` at index 0;
//! * **the hull's ground normal is the plane through the first THREE contacting friction
//!   points**, cross product, flipped to agree with world up (`sub_140E69D40` tail);
//! * the wheel normal force is a spring-damper: `(g * compression / scale - v_normal *
//!   ground_damp_factor) * mass`, capped at `maximum normal force contribution * mass`
//!   (`sub_1406CC8C0`, and the cap is stashed as `*((float *)physics + 7) * instance->mass`);
//! * a wheel only gets traction when the contact normal's up component **exceeds `ground maximum
//!   slope 1`** — which confirms the cosine encoding from the consumption side, independently of
//!   the postprocess that writes it;
//! * `iteration count` is the number of physics sub-steps: `sub_1406CF0F0` runs its whole body
//!   `iteration_count` times with `dt / n` and force fraction `1 / n`.
//!
//! Inferred, and labelled at each site:
//!
//! * **the spring's scale.** The engine's expression is `g * compression * radius / (ground_depth
//!   * extension)` where `extension` is a per-wheel runtime float this module cannot see. Taking
//!   `extension = radius` leaves `g * compression / ground_depth`, which is dimensionally exact
//!   and gives the right numbers on two very different vehicles: the static compression works out
//!   at `fraction_of_total_mass_share * ground_depth`, i.e. **0.031 wu (9 cm) on a Warthog**
//!   (depth 0.125, four wheels) and **0.083 wu on a Scorpion** (depth 1.000, twelve points). A
//!   stiff car and a heavy tracked hull, from one formula and no tuning.
//! * **the load that the ground puts back on the engine.** The engine reads it from the Havok
//!   instance (`sub_140E60130`: `engine_state + 8 = instance[30]`), which is the summed wheel
//!   reaction and not available without the full Havok wheel model. It is derived here from the
//!   no-slip identity instead: with `v = ratio * w`, `dv/dt = ratio * (T + ratio * T_g) / I`, so
//!   requiring `dv/dt = ratio * T / I - a_resist` gives `T_g = -a_resist * I / ratio^2`. That is
//!   an identity, not a constant, and `a_resist` is itself authored (the slope component of
//!   gravity, the family's own `drag coeficient` and `constant deceleration`).

use glam::Vec3;

/// Reach's tick rate. `seconds_to_ticks` in the engine is `ceil(s * 30) * tick_multiplier`, and
/// the multiplier is 1 at normal speed (`sub_1400BD130` / `sub_1400BD360`).
pub const TICK_RATE: f32 = 30.0;

/// `k_maximum_friction_points_per_vehicle_physics_definition`, from the engine's own bound check
/// `VALID_INDEX(friction_point_index, ...)` rejecting indices above 15
/// (`havok_vehicle_physics.cpp:1094`).
pub const MAX_CONTACT_POINTS: usize = 16;

/// The engine's `desired_gear == _vehicle_gear_none` sentinel: `sub_140E60170` resets the pair
/// `(gear, desired_gear)` to the 16-bit value `-257 = 0xFEFF`, so `gear = -1` and
/// `desired_gear = -2`.
pub const GEAR_NONE: i8 = -2;
/// Neutral. `requested_gear` returns this when the throttle is released and the engine has spun
/// below a quarter of its maximum.
pub const GEAR_NEUTRAL: i8 = -1;
/// Index 0 is REVERSE — the Warthog's gear 0 authors ratio `-0.50`.
pub const GEAR_REVERSE: i8 = 0;

/// `seconds_to_ticks`, exactly as `sub_1400BD130` computes it at normal game speed.
pub fn seconds_to_ticks(s: f32) -> u32 {
    if !(s > 0.0) {
        return 0;
    }
    (s * TICK_RATE).ceil() as u32
}

/// `clutch_shift_ticks` = `max(seconds_to_ticks(0.05), tick_multiplier)` = **2 ticks**, named by
/// the engine's own assert `IN_RANGE_INCLUSIVE(engine->clutch_in_ticks, 0, clutch_shift_ticks-1)`
/// (`vehicle_engine.cpp:335`).
pub const CLUTCH_SHIFT_TICKS: u8 = 2;

/// The engine's idle floor and overspeed ceiling on the flywheel, as fractions of `engine max
/// angular velocity` (`sub_140E601B0`: `min(1.3 * max, max(aav, 0.05 * max))`).
pub const AAV_IDLE_FRACTION: f32 = 0.05;
pub const AAV_CEILING_FRACTION: f32 = 1.3;
/// In neutral the flywheel decays by this factor each TICK (`sub_140E601B0`: `aav * 0.95`).
pub const NEUTRAL_DECAY_PER_TICK: f32 = 0.95;
/// The time constant the loaded/cruising blend relaxes on (`sub_140E601B0`: `dt / 0.022`).
pub const TRANSITION_TAU: f32 = 0.022;
/// Releasing the throttle only drops to neutral once the flywheel is below this fraction of its
/// maximum (`sub_140E5FE40`: `if (max_aav * 0.25 <= aav && (has_driver || ...)) keep the gear`).
pub const NEUTRAL_RELEASE_FRACTION: f32 = 0.25;

/// A vehicle's engine state. **Twenty bytes in the engine**, laid out exactly as
/// `sub_140E60170` resets it and `sub_140E601B0` steps it; the field names are the engine's own
/// where an assert gave them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineState {
    /// `+0x00 cruising_to_loaded_transition`, `0..1`. 1 = fully loaded.
    pub transition: f32,
    /// `+0x04` the flywheel's angular velocity. **In wu/s**, because `ratio * this` is the
    /// wheel's surface speed in wu/s.
    pub angular_velocity: f32,
    /// `+0x08` the torque the ground puts back on the engine, filled from the Havok instance by
    /// `sub_140E60130`. Derived here — see the module note.
    pub ground_torque: f32,
    /// `+0x0C` the current gear: -1 neutral, 0 reverse, 1.. forward.
    pub gear: i8,
    /// `+0x0D desired_gear`, [`GEAR_NONE`] when no shift is pending.
    pub desired_gear: i8,
    /// `+0x0E` ticks since the last completed shift, used by the engine for a 0.45 s reset.
    pub idle_ticks: u8,
    /// `+0x0F clutch_in_ticks`: 0 = engaged, 1..`CLUTCH_SHIFT_TICKS` = slipping.
    pub clutch_in_ticks: u8,
    /// `+0x10` ticks held in the current gear.
    pub gear_hold_ticks: u8,
    /// `+0x11` ticks the normalised flywheel speed has stayed ABOVE `upshift scale`.
    pub upshift_ticks: u8,
    /// `+0x12` ticks it has stayed BELOW `downshift scale`.
    pub downshift_ticks: u8,
    /// `+0x13` set when the gear is neutral.
    pub in_neutral: bool,
}

impl Default for EngineState {
    fn default() -> EngineState {
        EngineState {
            transition: 0.0,
            angular_velocity: 0.0,
            ground_torque: 0.0,
            gear: GEAR_NEUTRAL,
            desired_gear: GEAR_NONE,
            idle_ticks: 0,
            clutch_in_ticks: 0,
            gear_hold_ticks: 0,
            upshift_ticks: 0,
            downshift_ticks: 0,
            in_neutral: true,
        }
    }
}

/// The authored drivetrain: the engine block and the jeep/tank block's own `wheel circumferance`.
#[derive(Clone, Debug)]
pub struct Drivetrain {
    /// `engine moment` — "higher moments make engine spin up slower".
    pub moment: f32,
    /// `engine max angular velocity`.
    pub max_angular_velocity: f32,
    /// The gears, in authored index order. Index 0 is reverse.
    pub gears: Vec<super::Gear>,
    /// `wheel circumferance`, wu. The wrap length of the rolling phase, NOT a speed factor.
    pub wheel_circumference: f32,
}

impl Drivetrain {
    /// `None` when the family authors no usable engine, in which case the caller must not
    /// substitute one.
    pub fn new(engine: &super::Engine, wheel_circumference: f32) -> Option<Drivetrain> {
        if !(engine.moment > 1e-4) || !(engine.max_angular_velocity > 1e-3) || engine.gears.is_empty() {
            return None;
        }
        Some(Drivetrain {
            moment: engine.moment,
            max_angular_velocity: engine.max_angular_velocity,
            gears: engine.gears.clone(),
            wheel_circumference,
        })
    }

    /// `gear_ratio(gear)` — `sub_140E60850`, `gear_block + 0x38`. Zero outside the block.
    pub fn ratio(&self, gear: i8) -> f32 {
        if gear < 0 {
            return 0.0;
        }
        self.gears.get(gear as usize).map(|g| g.ratio).unwrap_or(0.0)
    }

    /// The flywheel speed normalised by `engine max angular velocity`.
    ///
    /// `sub_140E601B0` clamps to `[-0.5, 1.5]` before dividing; `sub_140E60980` (the curve's own
    /// input) clamps to `[0, 1]`. Both are reproduced, because the shift thresholds use the first
    /// and the torque lookup uses the second.
    pub fn normalised_speed(&self, aav: f32) -> f32 {
        let m = self.max_angular_velocity;
        aav.clamp(-0.5 * m, 1.5 * m) / m
    }

    /// `vehicle_engine_get_torque` — `sub_140E60890` + `sub_140E60980`.
    ///
    /// `(1 - t) * cruising(u) + t * loaded(u)` with `u = clamp(aav / max_aav, 0, 1)`. A gear that
    /// authors no curve contributes nothing from that curve rather than a substituted value.
    pub fn torque(&self, gear: i8, aav: f32, transition: f32) -> f32 {
        if gear < 0 {
            return 0.0;
        }
        let Some(g) = self.gears.get(gear as usize) else { return 0.0 };
        let u = (aav / self.max_angular_velocity).clamp(0.0, 1.0);
        let loaded = g.loaded_torque.as_ref().map(|f| f.eval(u)).unwrap_or(0.0);
        let cruising = g.cruising_torque.as_ref().map(|f| f.eval(u)).unwrap_or(0.0);
        let t = transition.clamp(0.0, 1.0);
        (1.0 - t) * cruising + t * loaded
    }

    /// **The wheel's surface speed, wu/s** — `sub_140E600E0`: `gear_ratio * angular_velocity`.
    ///
    /// Signed through the gear ratio, so reverse (gear 0, ratio negative) really does run
    /// backwards with the flywheel spinning the same way.
    pub fn wheel_surface_speed(&self, st: &EngineState) -> f32 {
        self.ratio(st.gear) * st.angular_velocity
    }

    /// The drivetrain's no-load ceiling, wu/s: `max angular velocity * the largest positive gear
    /// ratio`. **A ceiling, not a top speed** — the flywheel only reaches `max_angular_velocity`
    /// with no load at all, and the engine permits a transient 1.3x beyond it.
    pub fn speed_ceiling(&self) -> f32 {
        self.max_angular_velocity * self.gears.iter().map(|g| g.ratio).fold(0.0f32, f32::max)
    }

    /// Which gear the engine ASKS for this tick — `sub_140E5FE40`, in its own order.
    ///
    /// `throttle` is the signed throttle (the jeep passes the movement stick's magnitude signed by
    /// forward, `sub_140E61950`).
    pub fn requested_gear(&self, st: &EngineState, throttle: f32, ebrake: bool, has_driver: bool) -> i8 {
        let n = self.gears.len() as i8;
        let u = (st.angular_velocity / self.max_angular_velocity).clamp(0.0, 1.0);
        if throttle == 0.0 {
            // Released. The gear is HELD while the flywheel is still spinning (a quarter of
            // maximum) and somebody is aboard; otherwise neutral.
            let spinning = st.angular_velocity >= self.max_angular_velocity * NEUTRAL_RELEASE_FRACTION;
            if spinning && (has_driver || (st.gear_hold_ticks as f32) / TICK_RATE <= 1.0) {
                return st.gear;
            }
            return GEAR_NEUTRAL;
        }
        if throttle < 0.0 {
            // Reverse, unless the e-brake is down (then the gear is held).
            return if ebrake { st.gear } else { GEAR_REVERSE };
        }
        if st.gear < 1 {
            // Engage first.
            return if ebrake { st.gear } else { 1 };
        }
        let Some(g) = self.gears.get(st.gear as usize) else { return st.gear };
        let mut want = st.gear;
        // UPSHIFT: held above `upshift scale` for longer than `min time to upshift`, with the
        // throttle asking for at least as much as the engine is already giving.
        if u32::from(st.upshift_ticks) > seconds_to_ticks(g.min_time_to_upshift)
            && throttle >= u
            && u > g.upshift_scale
        {
            want = (st.gear + 1).clamp(1, (n - 1).max(1));
        }
        // DOWNSHIFT: held below `downshift scale` for longer than `min time to downshift`.
        if u32::from(st.downshift_ticks) > seconds_to_ticks(g.min_time_to_downshift) && g.downshift_scale > u {
            want = (st.gear - 1).max(1);
        }
        want
    }

    /// One engine tick — `sub_140E601B0`, in its own order.
    ///
    /// `dt` is the step; `throttle` the signed throttle; `requested` the gear from
    /// [`Drivetrain::requested_gear`]. Returns true when the gear CHANGED this step (the engine
    /// uses that to play the shift sound).
    pub fn update(&self, st: &mut EngineState, throttle: f32, requested: i8, ebrake: bool, dt: f32) -> bool {
        let m = self.max_angular_velocity;
        // "if (engine->clutch_in_ticks != 0 || ebrake) throttle = 0.0" -- the clutch really does
        // cut the throttle, which is the shift lurch.
        let eff_throttle = if st.clutch_in_ticks != 0 || ebrake { 0.0 } else { throttle.abs() };
        let u = self.normalised_speed(st.angular_velocity);
        // `cruising_to_loaded_transition` chases the error between what is asked for and what the
        // engine is already doing, at the authored 0.022 s.
        st.transition = (st.transition + (dt / TRANSITION_TAU) * (eff_throttle - u)).clamp(0.0, 1.0);
        st.idle_ticks = st.idle_ticks.saturating_add(1).min(0xFE);

        let mut changed = false;
        if requested != st.gear {
            if st.clutch_in_ticks != 0 {
                if st.clutch_in_ticks < CLUTCH_SHIFT_TICKS {
                    st.clutch_in_ticks += 1;
                } else {
                    // The clutch is in: the shift completes.
                    changed = st.gear != requested;
                    st.gear = if st.desired_gear == GEAR_NONE { requested } else { st.desired_gear };
                    st.desired_gear = GEAR_NONE;
                    st.gear_hold_ticks = 0;
                    st.upshift_ticks = 0;
                    st.downshift_ticks = 0;
                    st.clutch_in_ticks = 0;
                    if u32::from(st.idle_ticks) > seconds_to_ticks(0.45) {
                        st.idle_ticks = 0;
                    }
                }
            } else {
                st.desired_gear = requested;
                st.clutch_in_ticks = 1;
            }
        } else {
            st.clutch_in_ticks = 0;
            st.gear_hold_ticks = st.gear_hold_ticks.saturating_add(1);
            if let Some(g) = (st.gear >= 0).then(|| self.gears.get(st.gear as usize)).flatten() {
                // The two shift TIMERS: each counts ticks the normalised flywheel speed has
                // stayed on the far side of that gear's own threshold, and resets the moment it
                // comes back.
                if u <= g.upshift_scale {
                    st.upshift_ticks = 0;
                } else {
                    st.upshift_ticks = st.upshift_ticks.saturating_add(1);
                }
                if g.downshift_scale <= u {
                    st.downshift_ticks = 0;
                } else {
                    st.downshift_ticks = st.downshift_ticks.saturating_add(1);
                }
            }
        }

        // THE FLYWHEEL.
        if st.gear < 0 {
            // Neutral: decay. The engine's 0.95 is per TICK, so it is raised to dt*30 here
            // rather than applied once per frame -- otherwise a 120 Hz display spins the engine
            // down four times as fast as a 30 Hz one.
            st.angular_velocity *= NEUTRAL_DECAY_PER_TICK.powf(dt * TICK_RATE);
            st.in_neutral = true;
        } else {
            let torque = self.torque(st.gear, st.angular_velocity, st.transition);
            let ground = self.ratio(st.gear) * st.ground_torque;
            let aav = st.angular_velocity + dt * ((torque + ground) / self.moment);
            st.angular_velocity = aav.max(m * AAV_IDLE_FRACTION).min(m * AAV_CEILING_FRACTION);
            st.in_neutral = false;
        }
        changed
    }

    /// The ground torque that reproduces a given resisting acceleration, from the no-slip
    /// identity `T_g = -a_resist * I / ratio^2`.
    ///
    /// **An identity, not a fitted constant** — see the module note. Zero in neutral and at a
    /// ratio too small to divide by, which is the honest answer rather than an infinity.
    pub fn ground_torque_for_resistance(&self, gear: i8, a_resist: f32) -> f32 {
        let r = self.ratio(gear);
        if r.abs() < 1e-3 {
            return 0.0;
        }
        -a_resist * self.moment / (r * r)
    }
}

// ---------------------------------------------------------------------------------------------
// GROUND CONTACT: where the wheels touch, how far the springs are in, what the hull does
// ---------------------------------------------------------------------------------------------

/// What the friction-point flags say a wheel does with the steering angle.
///
/// Measured in `sub_140E69D40`: a wheel with flag bit 2 gets `+sin(delta/2)` in its steer
/// quaternion, one with bit 3 (or bit 4 while a turret is attached) gets `-sin(delta/2)`, and one
/// with neither is given the hull's own orientation unrotated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteerRole {
    /// Flag bit 2: steers with the wheel angle.
    Steered,
    /// Flag bit 3: steers INVERTED (a rear-steer axle).
    Inverted,
    /// Neither: fixed.
    Fixed,
}

impl SteerRole {
    /// From a `friction_point_definition_block` `flags` word.
    pub fn from_flags(flags: u32) -> SteerRole {
        if flags & 0x04 != 0 {
            SteerRole::Steered
        } else if flags & 0x08 != 0 {
            SteerRole::Inverted
        } else {
            SteerRole::Fixed
        }
    }
    pub fn sign(self) -> f32 {
        match self {
            SteerRole::Steered => 1.0,
            SteerRole::Inverted => -1.0,
            SteerRole::Fixed => 0.0,
        }
    }
}

/// One ground-contact point, resolved: the marker's model-space position plus what the tag says
/// about it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactPoint {
    /// The marker position in model space, from the render model's bind pose.
    pub local: Vec3,
    /// `radius`, wu. The wheel's own; for a hover point, its `radius`.
    pub radius: f32,
    /// How far above the contact the point RESTS: zero for a wheel (the radius is the stand-off),
    /// `antigrav height` for a hover point.
    pub rest_height: f32,
    /// This point's share of the vehicle's weight, normalised so the shares sum to 1.
    ///
    /// From `fraction of total mass`, which **is not a 0..1 fraction**: a Warthog authors 6.5 per
    /// wheel (sum 25.67) and a Scorpion 40.0 per point (sum 480). Only the ratio is used, so
    /// whatever unit Bungie meant, normalising makes it a load share — and that is what makes the
    /// static pitch and roll come out at exactly zero on any wheel layout (see
    /// [`ContactSolver::centroid`]).
    pub weight: f32,
    pub steer: SteerRole,
}

/// The authored ground spring, one per vehicle: `havok_vehicle_physics_struct`'s own fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundSpring {
    /// `ground depth`, wu. **The compression at which ONE contact alone would carry the whole
    /// vehicle**, which is the reading of the engine's `g * compression / scale` with the one
    /// runtime factor taken as 1 — see the module note. Warthog 0.125, Scorpion 1.000.
    pub depth: f32,
    /// `ground damp factor`: the coefficient on the contact-normal velocity. Warthog 1.350,
    /// Mongoose 1.450, Scorpion 1.000.
    pub damp: f32,
    /// `maximum normal force contribution` with its engine-confirmed default of 3 applied
    /// (`sub_1406D51E0`: `if (0.0 >= field) field = 3.0f`). A per-contact cap, wu/s^2.
    pub max_normal_force: f32,
    /// `ground maximum slope 0` / `1` as COSINES. A contact gives traction only when its normal's
    /// up component exceeds `[1]` (`sub_1406CC8C0`).
    pub slope_cos: [f32; 2],
    /// `iteration count`, the engine's own sub-step count, clamped to at least 1 by the
    /// postprocess (`sub_1406D51E0`: `field = max(field, 1)`).
    pub iterations: u32,
}

impl GroundSpring {
    /// True when this vehicle authors a usable wheel spring at all. **Every hover vehicle in
    /// `forge_halo` authors `ground depth = 0` and `ground damp factor = 0`** (measured: Ghost,
    /// and every turret), so the wheel law must not be run for them and must not be faked.
    pub fn authored(&self) -> bool {
        self.depth > 1e-5
    }
}

/// What one contact solve found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactSolve {
    /// Per point, the spring compression in wu — **the number the `suspension:*` overlays want**.
    pub compression: [f32; MAX_CONTACT_POINTS],
    /// Per point, whether it touched.
    pub touching: [bool; MAX_CONTACT_POINTS],
    /// How many points were solved (the valid prefix of the two arrays above).
    pub count: usize,
    /// How many of them touched.
    pub touching_count: usize,
    /// The hull's ground normal: **the plane through the first three touching points**, which is
    /// the engine's own rule (`sub_140E69D40`). `Vec3::Z` with fewer than three.
    pub normal: Vec3,
    /// Vertical acceleration from the springs, wu/s^2. Gravity is NOT included.
    pub spring_accel_z: f32,
    /// Angular acceleration about the hull's own lateral axis, nose-up positive, 1/s^2.
    pub alpha_pitch: f32,
    /// Angular acceleration about the hull's own forward axis, left-side-up positive, 1/s^2.
    pub alpha_roll: f32,
    /// The lowest over-penetration found: how far the hull has to be lifted so that no point is
    /// below its own hard floor. Zero when nothing is through the ground.
    pub penetration: f32,
}

impl Default for ContactSolve {
    fn default() -> ContactSolve {
        ContactSolve {
            compression: [0.0; MAX_CONTACT_POINTS],
            touching: [false; MAX_CONTACT_POINTS],
            count: 0,
            touching_count: 0,
            normal: Vec3::Z,
            spring_accel_z: 0.0,
            alpha_pitch: 0.0,
            alpha_roll: 0.0,
            penetration: 0.0,
        }
    }
}

/// The contact points of one vehicle, with the derived quantities that do not change as it
/// drives: the load centroid and the two pitch/roll inertia radii.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactSolver {
    pub points: Vec<ContactPoint>,
    /// The WEIGHTED centroid of the contact points, model space.
    ///
    /// **Torques are taken about this, not about the model origin, and that is what makes a
    /// vehicle at rest sit level.** Each point's static load is its own `weight`, so the static
    /// torque is `sum(weight_i * g * d_i)` and `sum(weight_i * d_i) = 0` by the definition of a
    /// weighted centroid. Any other reference point would give a permanent static pitch whenever
    /// the markers are not symmetric about the origin.
    pub centroid: Vec3,
    /// `sum(weight_i * (x_i - centroid.x)^2)`, wu^2 — the pitch inertia per unit mass.
    pub pitch_inertia: f32,
    /// The same about the forward axis.
    pub roll_inertia: f32,
}

impl ContactSolver {
    /// Build from the authored points. `None` when there are fewer than three, because with two
    /// or fewer there is no plane and the caller must say so rather than conform to an invented
    /// one.
    pub fn new(points: Vec<ContactPoint>) -> Option<ContactSolver> {
        if points.len() < 3 || points.len() > MAX_CONTACT_POINTS {
            return None;
        }
        let total: f32 = points.iter().map(|p| p.weight).sum();
        if !(total > 1e-6) {
            return None;
        }
        let centroid = points.iter().fold(Vec3::ZERO, |a, p| a + p.local * (p.weight / total));
        let pitch_inertia = points.iter().map(|p| p.weight / total * (p.local.x - centroid.x).powi(2)).sum::<f32>();
        let roll_inertia = points.iter().map(|p| p.weight / total * (p.local.y - centroid.y).powi(2)).sum::<f32>();
        Some(ContactSolver { points, centroid, pitch_inertia, roll_inertia })
    }

    /// Normalised load share of point `i`.
    pub fn share(&self, i: usize) -> f32 {
        let total: f32 = self.points.iter().map(|p| p.weight).sum();
        if total > 1e-6 { self.points[i].weight / total } else { 0.0 }
    }

    /// The compression this point sits at when the vehicle is parked on level ground:
    /// `load share * ground depth`.
    ///
    /// Derived, and **derivable without this code**: the springs must sum to gravity, so
    /// `sum(g * c_i / depth) = g` with `c_i` in proportion to the load share gives
    /// `c_i = share_i * depth`. A gate computes it that way and compares.
    pub fn static_compression(&self, i: usize, spring: &GroundSpring) -> f32 {
        self.share(i) * spring.depth
    }

    /// Solve one step's contacts.
    ///
    /// * `basis` is the hull's current rotation (yaw, then pitch, then roll);
    /// * `pos` the hull's model origin in the world;
    /// * `vel` the hull's linear velocity;
    /// * `rate` is `(pitch_rate, roll_rate)`, so the per-point normal velocity includes the
    ///   hull's own rotation — without it the damper cannot damp a pitch oscillation;
    /// * `ground(p)` answers "the ground height and normal under this world point", or `None`.
    pub fn solve(
        &self,
        spring: &GroundSpring,
        // `gravity` is the acceleration this vehicle is on, wu/s^2. **Passed in, not stored**:
        // there must be exactly one place that multiplies the world constant by the authored
        // `gravity scale`, or a caller can set one copy and watch nothing happen -- which is what
        // a first version of this did, and the free-fall gate caught it.
        gravity: f32,
        basis: glam::Quat,
        pos: Vec3,
        vel: Vec3,
        rate: (f32, f32),
        ground: &dyn Fn(Vec3) -> Option<(f32, Vec3)>,
    ) -> ContactSolve {
        let mut out = ContactSolve { count: self.points.len(), ..Default::default() };
        // The suspension's TRAVEL: the authored `ground depth`. At that compression one contact
        // alone carries the whole vehicle, so it is the bump stop as well as the spring's scale.
        let travel = spring.depth.max(1e-4);
        // PASS 1: where does each point touch, and how far is its spring in.
        //
        // One ray per point. The compression is a HEIGHT difference and nothing else, so this
        // pass needs no surface normal — which matters, because the ground query that covers the
        // Forge objects a map is built out of can only give a height.
        let mut contact: [Vec3; MAX_CONTACT_POINTS] = [Vec3::ZERO; MAX_CONTACT_POINTS];
        let mut first_normal: Option<Vec3> = None;
        for (i, p) in self.points.iter().enumerate() {
            let world = pos + basis * p.local;
            let Some((h, n)) = ground(world) else { continue };
            if first_normal.is_none() && n.length_squared() > 0.5 {
                first_normal = Some(n.normalize());
            }
            // Where this point rests when the spring is at zero compression.
            let c = (h + p.radius + p.rest_height) - world.z;
            if c <= 0.0 {
                // Off the ground: no force, no contact, and the spring is fully extended.
                continue;
            }
            out.touching[i] = true;
            out.touching_count += 1;
            contact[i] = Vec3::new(world.x, world.y, h);
            out.compression[i] = c.min(travel);
            out.penetration = out.penetration.max(c - travel);
        }
        // THE GROUND NORMAL: the plane through the first three TOUCHING contacts, flipped to
        // agree with world up. **This is the engine's own rule**, lifted from the tail of
        // `sub_140E69D40`, which does exactly this cross product over the first three contacting
        // friction points and flips the result against the world up vector. A least-squares plane
        // over all twelve of a Scorpion's points would be smoother and would not be what the
        // engine does, so it is not what happens here.
        let touching_pts: Vec<Vec3> =
            (0..self.points.len()).filter(|&i| out.touching[i]).take(3).map(|i| contact[i]).collect();
        if touching_pts.len() == 3 {
            let n = (touching_pts[2] - touching_pts[0]).cross(touching_pts[1] - touching_pts[0]);
            if n.length() > 1e-4 {
                let n = n.normalize();
                out.normal = if n.z < 0.0 { -n } else { n };
            }
        } else if let Some(n) = first_normal {
            // Fewer than three contacts: there is no plane, so the sampled surface normal is the
            // honest answer rather than a fabricated one.
            out.normal = n;
        }
        // PASS 2: the spring forces. A hover vehicle authors no wheel spring (measured: every
        // hover vehicle in `forge_halo` authors `ground depth = 0` and `ground damp factor = 0`),
        // so it gets the plane and the contact set and NOT a spring it never authored.
        if !spring.authored() {
            return out;
        }
        // A contact on too steep a face gives no support, exactly as the engine refuses traction
        // below `ground maximum slope 1` (`sub_1406CC8C0`: `if (normal.z <= physics[6]) no
        // traction`). Applied to the solved plane, because that is the surface the hull is
        // actually standing on.
        if spring.slope_cos[1] > 0.0 && out.normal.z <= spring.slope_cos[1] {
            return out;
        }
        let mut accel_sum = 0.0f32;
        let mut torque_pitch = 0.0f32;
        let mut torque_roll = 0.0f32;
        for (i, p) in self.points.iter().enumerate() {
            if !out.touching[i] {
                continue;
            }
            let d = p.local - self.centroid;
            // The point's own vertical velocity: the hull's, plus what its rotation does to a
            // point this far from the centroid (small-angle, which is all a suspension sees).
            let v_point = vel.z + rate.0 * d.x + rate.1 * d.y;
            // THE ENGINE'S SPRING-DAMPER, capped as the engine caps it. One accumulator: the
            // force is per-vehicle-mass in the engine too (`* instance->mass`), so the shares do
            // NOT multiply here -- they emerge as each point's own resting compression.
            let c = out.compression[i];
            // **THE CAP IS ON THE TOTAL, not on the spring half.** This line read
            //
            //     ((gravity * c / travel).min(cap) - v_point * damp).max(0.0)
            //
            // which left the DAMPER term unbounded, and because a contact cannot pull the
            // `.max(0.0)` makes it ONE-SIDED: it can push up hard and never pull down, so every
            // oscillation added energy. With `v_point` carrying the hull's angular rate
            // (`rate * d`), a wheel on a step pumped itself to **99,206 wu/s in one second** -- the
            // *"warthog just flew into the air"* report. One-sidedness is only safe when the total
            // is bounded, and the engine does bound it: `sub_1406CC8C0` stashes the cap ONCE as
            // `*((float *)physics + 7) * instance->mass` and applies it to the normal force, not to
            // a part of it. So the clamp goes round the whole expression, which is both the engine's
            // reading and the thing that makes the system unable to gain energy:
            // `a <= max_normal_force` always, so the hull's upward acceleration can never exceed
            // `contacts * max_normal_force` and its angular acceleration is bounded with it.
            let a = (gravity * c / travel - v_point * spring.damp).clamp(0.0, spring.max_normal_force);
            accel_sum += a;
            torque_pitch += a * d.x;
            torque_roll += a * d.y;
        }
        out.spring_accel_z = accel_sum;
        out.alpha_pitch = if self.pitch_inertia > 1e-6 { torque_pitch / self.pitch_inertia } else { 0.0 };
        out.alpha_roll = if self.roll_inertia > 1e-6 { torque_roll / self.roll_inertia } else { 0.0 };
        out
    }
}

/// #vehicle-suspension  The fastest the bump-stop correction may push a hull out of the ground,
/// wu/s.
///
/// **A numerical guard with no engine basis, and it is here because the correction used to be an
/// unbounded teleport.** `ContactSolve::penetration` was added straight to `pos.z` every sub-step,
/// so a ground query reporting a surface 1.25 wu above a wheel walked the hull upward at
/// **75 wu/s with a vertical velocity of 0.85** -- it was not flying, it was being teleported, and
/// it never escaped because the surface it was hitting was its own collision hull and moved with
/// it. Resolving an overlap is a position correction, so it must not create velocity and it must
/// not be able to look like flight; 2 wu/s is slower than a vehicle drives and still clears a
/// whole suspension travel in a sixteenth of a second.
pub const MAX_PENETRATION_PUSH_OUT: f32 = 2.0;

/// #vehicle-suspension  The largest pitch or roll rate the hull may carry, rad/s.
///
/// A guard, stated rather than assumed. The capped springs can produce at most
/// `contacts * max_normal_force * max|offset| / inertia` of angular acceleration -- 13.7 rad/s^2
/// on a Warthog -- so a bounded oscillator never approaches this; it exists so that an
/// unforeseen input cannot feed a large `rate * d` back into the damper term, which is the loop
/// that diverged.
pub const MAX_ATTITUDE_RATE: f32 = 4.0;

/// The hull attitude that a ground normal implies, in the hull's own frame.
///
/// Returns `(pitch, roll)` in radians, nose-up and left-side-up positive — the two angles that
/// put the hull's up axis on `normal` while keeping its heading at `yaw`. **This is the geometry
/// of a plane, not a model of anything**, so it is exact: a gate can give it a 30-degree ramp
/// normal and demand 30 degrees of pitch.
pub fn attitude_from_normal(normal: Vec3, yaw: f32) -> (f32, f32) {
    let n = normal.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return (0.0, 0.0);
    }
    // The normal in the hull's heading frame: +x forward, +y left.
    let (s, c) = (yaw.sin(), yaw.cos());
    let fwd = n.x * c + n.y * s;
    let left = -n.x * s + n.y * c;
    // With the basis `Rz(yaw) * Ry(-pitch) * Rx(roll)` the up axis in the heading frame is
    // `(-cos(roll) sin(pitch), -sin(roll), cos(roll) cos(pitch))`, so inverting it exactly:
    let pitch = (-fwd).atan2(n.z.max(1e-4));
    let roll = (-left).atan2((fwd * fwd + n.z * n.z).sqrt().max(1e-4));
    (pitch, roll)
}

#[cfg(test)]
mod gates {
    //! #vehicle-drivetrain #vehicle-suspension
    //! `cargo test -p hms-app --profile quick reach_vehicle_drive -- --nocapture`
    //!
    //! **No map and no cache.** Every number here is either authored into the fixture from a
    //! measured tag value or derived from the fixture by arithmetic that does not go through the
    //! code under test. Where a gate asserts a RELATIONSHIP (doubling this doubles that) it says
    //! so, because a relationship cannot be satisfied by a coincidence the way a single fitted
    //! number can.

    use super::*;
    use crate::reach_vehicle::{DriveInput, DriveModel, DriveState, SpeedSource, TurnLaw, Turning, VelocityModel};

    /// The Warthog's own measured contact geometry.
    ///
    /// Positions: the four friction-point markers are at `x = +-0.65` (a 1.3000 wu wheelbase,
    /// measured) and `y = +-0.39` (a 0.7800 wu track, measured), radius 0.250, and the authored
    /// `fraction of total mass` 6.5 / 6.5 / 6.333 / 6.333 front-to-back.
    fn warthog_contacts() -> ContactSolver {
        let p = |x: f32, y: f32, w: f32, steer: SteerRole| ContactPoint {
            local: Vec3::new(x, y, 0.0),
            radius: 0.25,
            rest_height: 0.0,
            weight: w,
            steer,
        };
        ContactSolver::new(vec![
            p(0.65, 0.39, 6.5, SteerRole::Steered),
            p(0.65, -0.39, 6.5, SteerRole::Steered),
            p(-0.65, 0.39, 6.333, SteerRole::Fixed),
            p(-0.65, -0.39, 6.333, SteerRole::Fixed),
        ])
        .expect("four points is a plane")
    }

    /// The Warthog's own authored `havok_vehicle_physics_struct`.
    fn warthog_spring() -> GroundSpring {
        GroundSpring { depth: 0.125, damp: 1.35, max_normal_force: 2.2, slope_cos: [0.766, 0.643], iterations: 1 }
    }

    /// The Warthog's own authored engine and gearbox.
    fn warthog_drivetrain() -> Drivetrain {
        let g = |index: usize, ratio: f32, up: f32, up_s: f32, down: f32, down_s: f32| crate::reach_vehicle::Gear {
            index,
            ratio,
            min_time_to_upshift: up,
            upshift_scale: up_s,
            min_time_to_downshift: down,
            downshift_scale: down_s,
            // A flat unit torque: the torque CURVE is tested on its own, and a gate about the
            // gearbox must not have its answer shaped by a curve.
            loaded_torque: Some(crate::scene::MapFn::Const(4000.0)),
            cruising_torque: Some(crate::scene::MapFn::Const(4000.0)),
        };
        Drivetrain {
            moment: 2000.0,
            max_angular_velocity: 9.5,
            gears: vec![
                g(0, -0.5, 0.0, 0.0, 0.0, 0.0),
                g(1, 0.75, 0.85, 0.90, 0.0, 0.0),
                g(2, 1.0, 0.85, 0.90, 0.25, 0.60),
                g(3, 1.2, 0.0, 0.0, 0.25, 0.60),
            ],
            wheel_circumference: 1.249,
        }
    }

    /// A Warthog-shaped [`DriveModel`]: the real contact geometry, the real spring, the real
    /// gearbox, and the turn law that was already gated.
    fn warthog_model() -> DriveModel {
        DriveModel {
            kind: "human_jeep",
            max_forward_speed: 9.0,
            max_reverse_speed: 4.5,
            speed_source: SpeedSource::SeatMaxSpeedReference,
            speed_acceleration: 10.6,
            speed_deceleration: 10.6,
            slide: VelocityModel::default(),
            turning: Some(Turning { max_left: 60.0f32.to_radians(), max_right: -60.0f32.to_radians(), turn_rate: 9.4248 }),
            wheelbase: Some(1.3),
            law: TurnLaw::Bicycle { wheelbase: 1.3 },
            law_note: None,
            turn: crate::reach_vehicle::TurnRate {
                yaw_deg: [50.0, 16.67],
                pitch_deg: [0.0, 0.0],
                max_speed_reference: 9.0,
                exponent: 2.0,
            },
            gravity_scale: 1.0,
            ride_height: 0.25,
            drivetrain: Some(warthog_drivetrain()),
            contacts: Some(warthog_contacts()),
            spring: warthog_spring(),
            ground_friction: 3.37,
        }
    }

    /// Drive the model to rest on a given ground function and report where it ended up.
    fn settle(d: &DriveModel, ground: &dyn Fn(Vec3) -> Option<(f32, Vec3)>, start_z: f32, secs: f32) -> DriveState {
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, start_z), yaw: 0.0, ..Default::default() };
        let n = (secs * 60.0) as usize;
        for _ in 0..n {
            st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, ground);
        }
        st
    }

    // -----------------------------------------------------------------------------------------
    // TERRAIN CONFORMANCE
    // -----------------------------------------------------------------------------------------

    /// #vehicle-suspension  **A hull parked on a RAMP must be tilted by the ramp's own angle, and
    /// one parked on the flat must be exactly level.**
    ///
    /// Reported: *"vehicles are still kinda useless right now, the physics aren't right"*, and the
    /// most visible part of that was a Warthog driving up a hill perfectly level because
    /// `DriveState` carried no attitude at all.
    ///
    /// The expectation is the RAMP's angle, which this gate computes from the ramp function it
    /// wrote — `atan(slope)` — and never from the solver. Two ramps at two angles, and a flat
    /// floor, so the gate cannot pass by returning a constant.
    #[test]
    fn a_parked_hull_takes_the_angle_of_the_ground_it_is_on() {
        let d = warthog_model();
        // Flat: exactly level, and the gate says EXACTLY because a plane through four coplanar
        // contacts has no tilt in it to find.
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let st = settle(&d, &flat, 1.0, 4.0);
        println!("flat: pitch {:+.5} rad roll {:+.5} rad, z {:.4}", st.pitch, st.roll, st.pos.z);
        assert!(st.pitch.abs() < 1e-3, "a hull on flat ground must be level, got pitch {:+.5} rad", st.pitch);
        assert!(st.roll.abs() < 1e-3, "a hull on flat ground must be level, got roll {:+.5} rad", st.roll);

        // Two ramps rising along +x. The hull's nose is UP by the ramp's angle.
        for slope in [0.2f32, 0.5] {
            let want = slope.atan();
            let ramp = move |p: Vec3| Some((p.x * slope, Vec3::new(-slope, 0.0, 1.0).normalize()));
            let st = settle(&d, &ramp, 2.0, 6.0);
            println!(
                "ramp slope {slope}: pitch {:+.4} rad ({:.2} deg) against atan(slope) {:+.4} ({:.2} deg)",
                st.pitch,
                st.pitch.to_degrees(),
                want,
                want.to_degrees()
            );
            assert!(
                (st.pitch - want).abs() < 0.02,
                "on a {slope} slope the hull must pitch to {want:+.4} rad, got {:+.4}",
                st.pitch
            );
            assert!(st.roll.abs() < 1e-2, "a ramp along +x must not ROLL the hull, got {:+.4}", st.roll);
        }

        // And across: a ramp rising along +y rolls and must not pitch. `+roll` is left-side-up and
        // +y IS left, so a floor that rises to the left lifts the left side: positive.
        let slope = 0.3f32;
        let cross = move |p: Vec3| Some((p.y * slope, Vec3::new(0.0, -slope, 1.0).normalize()));
        let st = settle(&d, &cross, 2.0, 6.0);
        println!("cross slope {slope}: roll {:+.4} rad ({:.2} deg), pitch {:+.4}", st.roll, st.roll.to_degrees(), st.pitch);
        assert!(
            (st.roll - slope.atan()).abs() < 0.02,
            "a floor rising to the left must roll the hull left-side-up by atan({slope}) = {:+.4}, got {:+.4}",
            slope.atan(),
            st.roll
        );
        assert!(st.pitch.abs() < 1e-2, "a ramp along +y must not PITCH the hull, got {:+.4}", st.pitch);
    }

    /// #vehicle-suspension  The plane rule itself, on its own: `attitude_from_normal` must invert
    /// the basis `Rz(yaw) * Ry(-pitch) * Rx(roll)` exactly.
    ///
    /// Round-tripped, which is the strongest form available: build a basis from a known
    /// `(yaw, pitch, roll)`, take its up axis, hand that to the function, and demand the angles
    /// back. **The expectation comes from `glam`, not from this module.**
    #[test]
    fn the_attitude_of_a_normal_round_trips_through_the_hulls_own_basis() {
        for &(yaw, pitch, roll) in &[
            (0.0f32, 0.0f32, 0.0f32),
            (0.0, 0.4, 0.0),
            (0.0, 0.0, -0.3),
            (1.1, 0.25, 0.15),
            (-2.3, -0.5, 0.45),
        ] {
            let basis = glam::Quat::from_rotation_z(yaw)
                * glam::Quat::from_rotation_y(-pitch)
                * glam::Quat::from_rotation_x(roll);
            let up = basis * Vec3::Z;
            let (p, r) = attitude_from_normal(up, yaw);
            println!("yaw {yaw:+.2}: pitch {pitch:+.4} -> {p:+.4}, roll {roll:+.4} -> {r:+.4}");
            assert!((p - pitch).abs() < 1e-4, "pitch {pitch:+.4} came back as {p:+.4}");
            assert!((r - roll).abs() < 1e-4, "roll {roll:+.4} came back as {r:+.4}");
        }
        // A degenerate normal must give nothing rather than a NaN.
        assert_eq!(attitude_from_normal(Vec3::ZERO, 0.7), (0.0, 0.0));
    }

    // -----------------------------------------------------------------------------------------
    // SUSPENSION
    // -----------------------------------------------------------------------------------------

    /// #vehicle-suspension  **A parked hull sits at the compression the TAG implies, each wheel at
    /// its own load share.**
    ///
    /// The expectation is derived here from the authored numbers and the force balance, not from
    /// the solver: the springs must sum to gravity, each wheel's share of the load is its
    /// normalised `fraction of total mass`, and the spring is `g * c / depth` — so
    /// `c_i = share_i * depth`. For the Warthog that is 0.0316 wu on each front wheel and 0.0309
    /// on each rear, out of 0.125 wu of travel: a quarter of the travel, which is what a car
    /// parked on its springs looks like.
    #[test]
    fn a_parked_hull_rests_at_the_compression_its_own_tag_implies() {
        let d = warthog_model();
        let c = d.contacts.as_ref().unwrap();
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let st = settle(&d, &flat, 1.0, 6.0);
        // Independently: the shares, and the compression each implies.
        let total: f32 = c.points.iter().map(|p| p.weight).sum();
        for i in 0..c.points.len() {
            let want = c.points[i].weight / total * d.spring.depth;
            println!(
                "wheel {i}: share {:.4} -> compression {want:.5} wu, solver has {:.5}",
                c.points[i].weight / total,
                st.suspension[i]
            );
            assert!(st.touching[i], "wheel {i} must be on the ground");
            assert!(
                (st.suspension[i] - want).abs() < 2.0e-3,
                "wheel {i} must rest at share*depth = {want:.5} wu, got {:.5}",
                st.suspension[i]
            );
            assert!(
                st.suspension[i] < d.spring.depth,
                "a parked wheel must not be bottomed out: {:.5} of {:.5} wu of travel",
                st.suspension[i],
                d.spring.depth
            );
        }
        // And the hull's own height follows: the wheel centre is at `ground + radius -
        // compression`, and the markers are at local z = 0, so the hull origin is exactly that.
        let want_z = 0.0 + 0.25 - c.static_compression(0, &d.spring);
        println!("hull z {:.5}, radius - static compression = {want_z:.5}", st.pos.z);
        assert!((st.pos.z - want_z).abs() < 3.0e-3, "the hull must rest at {want_z:.5}, got {:.5}", st.pos.z);
    }

    /// #vehicle-suspension  **The spring is a spring: it comes back.**
    ///
    /// Two relationships, neither a fitted number:
    ///
    /// 1. dropped from 0.5 wu up, the hull must return to the SAME rest height it settles to from
    ///    just above it — to within a millimetre, and the rest height is the one the gate above
    ///    derives from the tag;
    /// 2. **doubling `ground depth` doubles the resting compression** and halves the spring rate,
    ///    which is the relationship that makes `ground depth` the spring's scale rather than an
    ///    arbitrary constant.
    #[test]
    fn a_dropped_hull_returns_to_its_rest_height_and_the_authored_depth_sets_it() {
        let d = warthog_model();
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let near = settle(&d, &flat, 0.26, 6.0);
        let dropped = settle(&d, &flat, 0.75, 6.0);
        println!("settled from just above: z {:.5} | dropped from 0.75: z {:.5}", near.pos.z, dropped.pos.z);
        assert!(
            (near.pos.z - dropped.pos.z).abs() < 1e-3,
            "a spring must return to one rest height: {:.5} vs {:.5}",
            near.pos.z,
            dropped.pos.z
        );
        // ...and it must have been compressed MORE at some point on the way, or the drop never
        // loaded the spring and the assertion above is vacuous.
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.75), ..Default::default() };
        let mut peak = 0.0f32;
        for _ in 0..360 {
            st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &flat);
            peak = peak.max(st.suspension[0]);
        }
        let rest = d.contacts.as_ref().unwrap().static_compression(0, &d.spring);
        println!("peak compression on landing {peak:.5} wu against a resting {rest:.5}");
        assert!(peak > rest * 1.5, "a drop must compress the spring past rest: peak {peak:.5}, rest {rest:.5}");

        // THE RELATIONSHIP: twice the depth, twice the resting compression.
        let soft = DriveModel { spring: GroundSpring { depth: 0.25, ..d.spring }, ..d.clone() };
        let s_soft = settle(&soft, &flat, 1.0, 8.0);
        println!("depth 0.125 -> compression {:.5} | depth 0.250 -> {:.5}", near.suspension[0], s_soft.suspension[0]);
        let ratio = s_soft.suspension[0] / near.suspension[0];
        assert!(
            (ratio - 2.0).abs() < 0.08,
            "doubling `ground depth` must double the resting compression, got a ratio of {ratio:.4}"
        );
    }

    /// #vehicle-suspension  **`ground damp factor` damps.** With the authored value the hull
    /// settles; with zero it keeps bouncing. Counted in zero crossings of the vertical velocity,
    /// so the gate measures oscillation rather than a height.
    #[test]
    fn the_authored_damp_factor_is_what_stops_the_bouncing() {
        let d = warthog_model();
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let bounces = |m: &DriveModel| -> usize {
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.75), ..Default::default() };
            let mut prev = 0.0f32;
            let mut n = 0;
            for _ in 0..360 {
                st = m.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &flat);
                if prev < -1e-3 && st.vel.z > 1e-3 {
                    n += 1;
                }
                prev = st.vel.z;
            }
            n
        };
        let damped = bounces(&d);
        let undamped = bounces(&DriveModel { spring: GroundSpring { damp: 0.0, ..d.spring }, ..d.clone() });
        println!("damp 1.35: {damped} bounce(s) in 6 s | damp 0.0: {undamped}");
        assert!(undamped > damped, "removing `ground damp factor` must leave MORE bouncing: {undamped} vs {damped}");
        // **The bound is "it settles", not a bounce count.** A first version asserted at most two
        // bounces and went red at three; three is not wrong -- the authored numbers give a damping
        // ratio of `damp / (2 * sqrt(g / depth))` = 1.35 / (2 * 5.78) = **0.117**, a lightly
        // damped spring at 0.92 Hz, so a few visible bounces is what the tag asks for and a
        // tighter bound would have been a number invented to make the gate green. What the
        // authored damping must do is bring the hull to REST, and leave it bouncing forever
        // without it.
        let decay = |m: &DriveModel| -> (f32, f32) {
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.75), ..Default::default() };
            let mut early = 0.0f32;
            let mut late = 0.0f32;
            for i in 0..600 {
                st = m.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &flat);
                if i > 60 && i < 180 {
                    early = early.max(st.vel.z.abs());
                }
                if i > 480 {
                    late = late.max(st.vel.z.abs());
                }
            }
            (early, late)
        };
        let (e, l) = decay(&d);
        let (ue, ul) = decay(&DriveModel { spring: GroundSpring { damp: 0.0, ..d.spring }, ..d.clone() });
        println!("damped: peak |vz| 1-3 s {e:.5}, 8-10 s {l:.5} | undamped: {ue:.5} -> {ul:.5}");
        assert!(l < 1e-3, "the authored damping must bring the hull to rest, |vz| still {l:.5} after 8 s");
        assert!(e > l * 10.0, "...and it must have been moving earlier, or that assertion is vacuous: {e:.5}");
        assert!(ul > 0.05, "with NO damping it must still be moving after 8 s, got {ul:.5}");
        let _ = ue;
    }

    /// #vehicle-suspension  **The result must not depend on the sub-step size.** The equilibrium
    /// is a property of the authored numbers; the sub-step is a numerical choice, so halving it
    /// must not move where the vehicle comes to rest.
    ///
    /// Run at four frame times including **10 Hz**, which is the slowest `step` accepts (it
    /// clamps `dt` to 0.1 s). A spring as stiff as `gravity / ground depth` is 5.78 rad/s on a
    /// Warthog and a 100 ms explicit step of it is on the edge of unstable, so 10 Hz is where
    /// the sub-stepping earns its keep and the only rate at which removing it is visible — the
    /// first version of this gate stopped at 30 Hz and a sabotage that deleted the sub-step floor
    /// entirely left it GREEN, because the equilibrium is frame-rate independent either way and
    /// 30 Hz is still inside the stability limit.
    #[test]
    fn the_rest_height_does_not_depend_on_the_frame_rate() {
        let d = warthog_model();
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let rest = |fps: f32| -> (f32, f32) {
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.6), ..Default::default() };
            let mut late_motion = 0.0f32;
            let n = (10.0 * fps) as usize;
            for i in 0..n {
                st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / fps, &flat);
                if i * 2 > n {
                    late_motion = late_motion.max(st.vel.z.abs());
                }
            }
            (st.pos.z, late_motion)
        };
        let rows: Vec<(f32, (f32, f32))> = [10.0f32, 30.0, 60.0, 144.0].iter().map(|&f| (f, rest(f))).collect();
        for (fps, (z, m)) in &rows {
            println!("{fps:>5} Hz: rest height {z:.5}, residual |vz| in the last half {m:.6}");
        }
        let z0 = rows[0].1 .0;
        for (fps, (z, m)) in &rows {
            assert!(z.is_finite(), "{fps} Hz diverged to {z}");
            assert!(
                (z - z0).abs() < 1e-3,
                "the rest height must not follow the frame rate: {fps} Hz gave {z:.5} against {z0:.5} at 10 Hz"
            );
            assert!(*m < 1e-3, "{fps} Hz was still moving at the end ({m:.6}) -- the integrator is not settling");
        }
    }

    /// #vehicle-suspension  **The sub-step floor, on a spring stiff enough to need it.**
    ///
    /// This gate exists because a sabotage went GREEN. Deleting the 1/120 s sub-step floor from
    /// `DriveModel::step` — leaving only the authored `iteration count`, which is 1 on every
    /// vehicle in `forge_halo` — changed nothing that
    /// [`the_rest_height_does_not_depend_on_the_frame_rate`] could see, even at 10 Hz. The reading
    /// is that **the sabotage is a no-op for every shipped vehicle**, and the arithmetic says why:
    /// explicit Euler on a spring is stable while `dt * omega < 2`, the stiffest authored spring
    /// is the `truck_cab_large`'s `ground depth = 0.075` giving `omega = sqrt(g / depth) = 7.46`,
    /// and `0.1 * 7.46 = 0.75`. So the floor is **defensive, not load-bearing**, and saying a gate
    /// proved it would have been a false claim.
    ///
    /// What the floor does protect against is a spring stiffer than anything Bungie shipped — a
    /// modded vehicle, or a tag read wrong. The threshold is derivable: `dt * sqrt(g / depth) = 2`
    /// at `depth = g * dt^2 / 4`, which for `dt = 0.1` is **0.0104 wu**. So this gate takes a
    /// spring well past that and demands it still settle.
    #[test]
    fn a_spring_too_stiff_for_a_whole_frame_is_still_stable() {
        let d = warthog_model();
        // 0.002 wu of travel: omega = 45.7 rad/s, so a single 0.1 s step of it has
        // dt*omega = 4.57, more than twice the explicit stability limit.
        let stiff = DriveModel { spring: GroundSpring { depth: 0.002, damp: 1.35, ..d.spring }, ..d.clone() };
        let limit = crate::reach_physics::vehicle_fall_acceleration() * 0.1 * 0.1 / 4.0;
        let omega = (crate::reach_physics::vehicle_fall_acceleration() / 0.002f32).sqrt();
        println!(
            "depth 0.002 -> omega {omega:.2} rad/s, dt*omega at 10 Hz = {:.2} (stable below 2);              the authored-depth threshold at 10 Hz is {limit:.5} wu",
            0.1 * omega
        );
        assert!(0.1 * omega > 2.0, "the fixture must be past the stability limit, or this gate is vacuous");
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.30), ..Default::default() };
        for _ in 0..100 {
            st = stiff.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 0.1, &flat);
        }
        // Independently: a stiff spring parks the wheel centre within its own tiny travel of the
        // contact, so the hull sits essentially at the wheel radius.
        println!("10 s at 10 Hz on a very stiff spring: z {:.5}, |vz| {:.6}", st.pos.z, st.vel.z.abs());
        assert!(st.pos.z.is_finite(), "the integrator diverged to {}", st.pos.z);
        assert!(
            (st.pos.z - 0.25).abs() < 0.01,
            "a stiff spring must park the hull at the wheel radius 0.25, got {:.5}",
            st.pos.z
        );
        assert!(st.vel.z.abs() < 1e-2, "...and must be at rest, got |vz| {:.6}", st.vel.z.abs());
    }

    // -----------------------------------------------------------------------------------------
    // THE DRIVETRAIN
    // -----------------------------------------------------------------------------------------

    /// #vehicle-drivetrain  **The wheel's surface speed IS `gear ratio * engine angular
    /// velocity`** — `sub_140E600E0`, which is one multiply and must be exactly that.
    ///
    /// Including the sign: reverse is gear 0 with a negative ratio, so the same flywheel spinning
    /// the same way drives the vehicle backwards. And neutral is exactly zero.
    #[test]
    fn the_wheel_surface_speed_is_the_gear_ratio_times_the_flywheel() {
        let dt = warthog_drivetrain();
        for (gear, ratio) in [(0i8, -0.5f32), (1, 0.75), (2, 1.0), (3, 1.2)] {
            for aav in [0.0f32, 3.3, 9.5] {
                let st = EngineState { gear, angular_velocity: aav, ..Default::default() };
                let want = ratio * aav;
                assert_eq!(dt.wheel_surface_speed(&st), want, "gear {gear} at {aav} must give {want}");
            }
        }
        let n = EngineState { gear: GEAR_NEUTRAL, angular_velocity: 9.5, ..Default::default() };
        assert_eq!(dt.wheel_surface_speed(&n), 0.0, "neutral drives nothing");
        // The ceiling: top gear's ratio times max engine speed, and the Warthog's is 11.40 wu/s.
        println!("ceiling {:.4} wu/s", dt.speed_ceiling());
        assert!((dt.speed_ceiling() - 9.5 * 1.2).abs() < 1e-4);
    }

    /// #vehicle-drivetrain  **The torque is the authored lerp of two authored curves, and the
    /// curve's input is the flywheel speed normalised and CLAMPED to 0..1.**
    ///
    /// `sub_140E60890` + `sub_140E60980`. The two curves here are deliberately different and
    /// linear, so the expected value is arithmetic on the fixture: at `u`, cruising gives `100*u`
    /// and loaded `900*u`, and the blend at `t` is `(1-t)*100u + t*900u`.
    #[test]
    fn the_engine_torque_is_the_authored_blend_of_the_two_authored_curves() {
        let mut dt = warthog_drivetrain();
        dt.gears[2].cruising_torque = Some(crate::scene::MapFn::Identity { lo: 0.0, hi: 100.0 });
        dt.gears[2].loaded_torque = Some(crate::scene::MapFn::Identity { lo: 0.0, hi: 900.0 });
        let m = dt.max_angular_velocity;
        for &(aav, t) in &[(m * 0.5, 0.0f32), (m * 0.5, 1.0), (m * 0.5, 0.25), (m, 0.5), (m * 0.1, 0.5)] {
            let u = (aav / m).clamp(0.0, 1.0);
            let want = (1.0 - t) * (100.0 * u) + t * (900.0 * u);
            let got = dt.torque(2, aav, t);
            println!("aav {aav:.3} (u {u:.3}) transition {t:.2}: want {want:.4}, got {got:.4}");
            assert!((got - want).abs() < 1e-3, "torque at u={u:.3} t={t:.2} must be {want:.4}, got {got:.4}");
        }
        // CLAMPED: past the maximum the curve stops rising, which is what `sub_140E60980` does
        // before it evaluates. 1.3x the maximum is reachable -- the flywheel's own clamp allows
        // it -- so this is not a hypothetical.
        let at_max = dt.torque(2, m, 0.5);
        let past = dt.torque(2, m * 1.3, 0.5);
        println!("torque at max {at_max:.4}, at 1.3x max {past:.4}");
        assert_eq!(at_max, past, "the torque curve input must clamp at 1.0");
        // A gear outside the block, and neutral, give nothing rather than a panic.
        assert_eq!(dt.torque(GEAR_NEUTRAL, m, 1.0), 0.0);
        assert_eq!(dt.torque(99, m, 1.0), 0.0);
    }

    /// #vehicle-drivetrain  **The gearbox shifts on the authored thresholds and the authored
    /// times, through a 2-tick clutch that cuts the throttle.**
    ///
    /// The Warthog's own gearbox. The expectations are read off the fixture:
    ///
    /// * gear 1 may upshift once the normalised flywheel speed has been above **0.90** for
    ///   **0.85 s** (26 ticks), and not before;
    /// * gear 3 may downshift once it has been below **0.60** for **0.25 s** (8 ticks);
    /// * a shift takes `max(ceil(0.05*30), 1) = 2` ticks of clutch, during which the throttle the
    ///   engine sees is zero.
    #[test]
    fn the_gearbox_shifts_on_its_own_authored_thresholds_and_times() {
        let dt = warthog_drivetrain();
        let m = dt.max_angular_velocity;
        // --- the UPSHIFT threshold: 0.90 of maximum, and the TIME, 0.85 s ------------------
        let mut st = EngineState { gear: 1, angular_velocity: m * 0.95, ..Default::default() };
        // Just below the authored time: no upshift is even REQUESTED.
        st.upshift_ticks = seconds_to_ticks(0.85) as u8;
        assert_eq!(
            dt.requested_gear(&st, 1.0, false, true),
            1,
            "at exactly `min time to upshift` the engine must not have shifted yet"
        );
        st.upshift_ticks += 1;
        assert_eq!(dt.requested_gear(&st, 1.0, false, true), 2, "one tick past the authored time it must upshift");
        // ...and not if the flywheel is below the authored `upshift scale`, however long it waits.
        let slow = EngineState { angular_velocity: m * 0.85, upshift_ticks: 200, ..st };
        assert_eq!(dt.requested_gear(&slow, 1.0, false, true), 1, "below `upshift scale` there is no upshift");
        // ...and not if the throttle is asking for less than the engine is already giving.
        assert_eq!(dt.requested_gear(&st, 0.5, false, true), 1, "a lifted throttle must not upshift");

        // --- the DOWNSHIFT threshold: below 0.60 for 0.25 s -------------------------------
        let mut st = EngineState { gear: 3, angular_velocity: m * 0.4, ..Default::default() };
        st.downshift_ticks = seconds_to_ticks(0.25) as u8;
        assert_eq!(dt.requested_gear(&st, 1.0, false, true), 3, "at exactly the authored time, not yet");
        st.downshift_ticks += 1;
        assert_eq!(dt.requested_gear(&st, 1.0, false, true), 2, "one tick past it, down a gear");
        let fast = EngineState { angular_velocity: m * 0.7, ..st };
        assert_eq!(dt.requested_gear(&fast, 1.0, false, true), 3, "above `downshift scale` there is no downshift");

        // --- REVERSE is gear 0, and it is the ratio that makes it reverse -----------------
        let rolling = EngineState { gear: 2, angular_velocity: m * 0.5, ..Default::default() };
        assert_eq!(dt.requested_gear(&rolling, -1.0, false, true), GEAR_REVERSE, "a negative throttle selects gear 0");
        assert!(dt.ratio(GEAR_REVERSE) < 0.0, "and gear 0's authored ratio is negative");

        // --- NEUTRAL on release, but only once the flywheel is below a quarter -------------
        let spinning = EngineState { gear: 2, angular_velocity: m * 0.3, ..Default::default() };
        assert_eq!(dt.requested_gear(&spinning, 0.0, false, true), 2, "a spinning engine holds its gear");
        let idling = EngineState { gear: 2, angular_velocity: m * 0.2, ..Default::default() };
        assert_eq!(dt.requested_gear(&idling, 0.0, false, true), GEAR_NEUTRAL, "below a quarter it drops to neutral");

        // --- THE CLUTCH: two ticks, throttle cut ------------------------------------------
        let mut st = EngineState { gear: 1, angular_velocity: m * 0.95, transition: 1.0, ..Default::default() };
        let before = st.angular_velocity;
        dt.update(&mut st, 1.0, 2, false, 1.0 / 30.0);
        assert_eq!(st.gear, 1, "tick 1 of the shift: still in the old gear");
        assert_eq!(st.clutch_in_ticks, 1, "...with the clutch going in");
        assert_eq!(st.desired_gear, 2, "...and the desired gear recorded");
        // **The throttle cut lands on tick TWO, not tick one**, and that is the engine's own
        // ordering: `sub_140E601B0` reads `clutch_in_ticks` at the TOP of the function, before
        // the shift logic that sets it. So the first tick of a shift still sees the real throttle.
        // The first version of this gate asserted the cut on tick one and went red for that
        // reason -- an expectation error, and the clue was that the "wrong" value was exactly the
        // fixture's own starting transition.
        assert_eq!(st.transition, 1.0, "tick 1 still sees the real throttle, so the blend holds");
        dt.update(&mut st, 1.0, 2, false, 1.0 / 30.0);
        assert_eq!(st.clutch_in_ticks, CLUTCH_SHIFT_TICKS, "tick 2: the clutch is fully in");
        assert!(st.transition < 1.0, "...and NOW the throttle is cut, so the loaded blend falls");
        dt.update(&mut st, 1.0, 2, false, 1.0 / 30.0);
        assert_eq!(st.gear, 2, "tick 3: the shift completes");
        assert_eq!(st.clutch_in_ticks, 0, "...and the clutch comes back out");
        assert_eq!(st.desired_gear, GEAR_NONE, "...with no shift pending");
        assert_eq!(st.upshift_ticks, 0, "...and both shift timers reset");
        assert_eq!(st.downshift_ticks, 0);
        let _ = before;
    }

    /// #vehicle-drivetrain  **The flywheel is integrated, clamped and frame-rate independent.**
    ///
    /// * `aav += dt * (torque + ratio*ground_torque) / engine_moment` — one tick's worth is
    ///   arithmetic on the fixture: 4000 / 2000 * dt.
    /// * it is floored at `0.05 * max` and ceilinged at `1.3 * max`;
    /// * in neutral it decays 0.95 per TICK, which must be the same after one second whatever the
    ///   frame rate — the trap `play_mode::clock` records, and the reason the decay is raised to
    ///   `dt * 30` rather than applied once a frame.
    #[test]
    fn the_flywheel_integrates_the_authored_torque_and_clamps_it() {
        let dt = warthog_drivetrain();
        let m = dt.max_angular_velocity;
        // ONE TICK of the integration, from a gear that is already engaged so no clutch runs.
        let h = 1.0 / 30.0;
        let mut st = EngineState { gear: 2, angular_velocity: m * 0.5, transition: 1.0, ..Default::default() };
        let start = st.angular_velocity;
        dt.update(&mut st, 1.0, 2, false, h);
        let want = start + h * (4000.0 / 2000.0);
        println!("one tick: {start:.5} -> {:.5}, arithmetic says {want:.5}", st.angular_velocity);
        assert!((st.angular_velocity - want).abs() < 1e-4, "the flywheel must integrate to {want:.5}, got {:.5}", st.angular_velocity);

        // THE CEILING: 1.3 x max, reached and not passed.
        let mut st = EngineState { gear: 2, angular_velocity: m, transition: 1.0, ..Default::default() };
        for _ in 0..600 {
            dt.update(&mut st, 1.0, 2, false, h);
        }
        println!("pinned for 20 s: aav {:.5}, 1.3*max = {:.5}", st.angular_velocity, m * 1.3);
        assert!((st.angular_velocity - m * 1.3).abs() < 1e-3, "the flywheel ceiling is 1.3x max, got {:.5}", st.angular_velocity);

        // THE IDLE FLOOR: `max(aav, 0.05 * max)`, which is a CLAMP and not a decay.
        //
        // The first version of this ran a zero-torque engine for 20 s and demanded it fall to the
        // floor; it sat exactly where it started, because `aav += dt * 0 / moment` is `aav` and
        // the engine's only in-gear decay is the clamp. That was an expectation error and it is
        // recognisable as one because the "wrong" answer (0.5 * max, unchanged) is the fixture's
        // own starting value. A flywheel with no torque on it does not slow down; the floor is
        // what stops the engine ever reading below idle. So the floor is tested by starting
        // BELOW it.
        let mut idle = warthog_drivetrain();
        for g in idle.gears.iter_mut() {
            g.loaded_torque = Some(crate::scene::MapFn::Const(0.0));
            g.cruising_torque = Some(crate::scene::MapFn::Const(0.0));
        }
        let mut st = EngineState { gear: 2, angular_velocity: m * 0.01, ..Default::default() };
        idle.update(&mut st, 1.0, 2, false, h);
        println!("started at 0.01*max with no torque: aav {:.5}, 0.05*max = {:.5}", st.angular_velocity, m * 0.05);
        assert!(
            (st.angular_velocity - m * 0.05).abs() < 1e-5,
            "the idle floor must lift the flywheel to 0.05x max in one tick, got {:.5}",
            st.angular_velocity
        );
        // ...and a flywheel ABOVE the floor with no torque must not be dragged down to it, which
        // is the half that makes the assertion above mean "clamp" rather than "decay".
        let mut st = EngineState { gear: 2, angular_velocity: m * 0.5, ..Default::default() };
        for _ in 0..600 {
            idle.update(&mut st, 1.0, 2, false, h);
        }
        assert!(
            (st.angular_velocity - m * 0.5).abs() < 1e-4,
            "with no torque the flywheel must hold its speed, got {:.5} from {:.5}",
            st.angular_velocity,
            m * 0.5
        );

        // THE TORQUE CURVE IS WHAT SETS THE TOP SPEED. A curve that falls to zero at 0.8 of the
        // engine's maximum must settle the flywheel there, and nowhere near the 1.3x clamp --
        // which is the property that makes the authored curve, rather than a ceiling, the thing
        // that decides how fast a vehicle goes.
        let mut taper = warthog_drivetrain();
        for g in taper.gears.iter_mut() {
            // 4000 at u = 0, through zero at u = 0.8, negative above it.
            g.loaded_torque = Some(crate::scene::MapFn::Identity { lo: 4000.0, hi: -1000.0 });
            g.cruising_torque = Some(crate::scene::MapFn::Identity { lo: 4000.0, hi: -1000.0 });
        }
        let mut st = EngineState { gear: 2, angular_velocity: m * 0.1, transition: 1.0, ..Default::default() };
        for _ in 0..900 {
            taper.update(&mut st, 1.0, 2, false, h);
        }
        let zero_at = 4000.0 / (4000.0 + 1000.0); // where Identity{4000,-1000} crosses zero
        println!(
            "a curve crossing zero at u = {zero_at:.3} settled the flywheel at u = {:.4}",
            st.angular_velocity / m
        );
        assert!(
            (st.angular_velocity / m - zero_at).abs() < 1e-3,
            "the flywheel must settle where its own torque curve crosses zero (u = {zero_at:.3}), got u = {:.4}",
            st.angular_velocity / m
        );

        // NEUTRAL DECAY, and it must not follow the frame rate.
        let decayed = |fps: f32| -> f32 {
            let mut st = EngineState { gear: GEAR_NEUTRAL, angular_velocity: 9.0, ..Default::default() };
            for _ in 0..fps as usize {
                dt.update(&mut st, 0.0, GEAR_NEUTRAL, false, 1.0 / fps);
            }
            st.angular_velocity
        };
        let (a, b, c) = (decayed(30.0), decayed(60.0), decayed(120.0));
        // Independently: 0.95 per tick for 30 ticks.
        let want = 9.0 * 0.95f32.powi(30);
        println!("one second of neutral: 30 Hz {a:.5} | 60 Hz {b:.5} | 120 Hz {c:.5}, 0.95^30 says {want:.5}");
        assert!((a - want).abs() < 1e-3, "one second in neutral must be 0.95^30, got {a:.5} against {want:.5}");
        assert!((a - b).abs() < 1e-3 && (b - c).abs() < 1e-3, "the decay must not follow the frame rate");
    }

    /// #vehicle-drivetrain  **A Warthog BUILDS speed through its gears.**
    ///
    /// The whole point of the drivetrain, and the thing a lerp-to-a-constant cannot do. Three
    /// properties, none of them a fitted number:
    ///
    /// 1. from rest at full throttle the hull's speed must RISE MONOTONICALLY and keep rising for
    ///    more than a second — a first-order approach to a constant is also monotone, so this
    ///    alone proves little and is only the precondition for the next two;
    /// 2. **it must pass through every forward gear in order**, 1 then 2 then 3, which no
    ///    speed-target model has at all;
    /// 3. it must settle **below the drivetrain's no-load ceiling** and above the first gear's
    ///    reach, i.e. between `ratio_1 * max_aav` and `ratio_top * max_aav` — a band derived from
    ///    the gearbox and not from the integrator.
    #[test]
    fn a_warthog_builds_speed_through_its_gears_instead_of_lerping_to_a_constant() {
        let d = warthog_model();
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let mut st = settle(&d, &flat, 0.3, 2.0);
        st.vel = Vec3::ZERO;
        let go = DriveInput { throttle: 1.0, slide: 0.0, want_yaw: 0.0, brake: false };
        let mut gears_seen: Vec<i8> = Vec::new();
        let mut speeds: Vec<f32> = Vec::new();
        for i in 0..600 {
            st = d.step(st, go, 1.0 / 60.0, &flat);
            if gears_seen.last() != Some(&st.engine.gear) {
                gears_seen.push(st.engine.gear);
            }
            if i % 30 == 0 {
                speeds.push(st.vel.dot(st.forward()));
            }
        }
        let dtr = d.drivetrain.as_ref().unwrap();
        println!("gears in order: {gears_seen:?}");
        println!("speed each half-second: {:?}", speeds.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
        // 2. every forward gear, in order.
        let forward: Vec<i8> = gears_seen.iter().copied().filter(|&g| g >= 1).collect();
        assert_eq!(forward, vec![1, 2, 3], "a Warthog must climb through gears 1, 2 and 3 in order, saw {gears_seen:?}");
        // 1. and it must have been BUILDING, not snapping: still gaining after a second.
        assert!(speeds[2] > speeds[1] + 0.2, "the speed must still be building after 1 s: {:?}", speeds);
        assert!(speeds[1] > speeds[0] + 0.2, "...and from the very start too");
        // 3. the band, from the gearbox.
        let v = st.vel.dot(st.forward());
        let lo = dtr.ratio(1) * dtr.max_angular_velocity;
        let hi = dtr.speed_ceiling() * 1.3;
        println!("settled at {v:.3} wu/s, between first gear's {lo:.3} and the overspeed ceiling {hi:.3}");
        assert!(v > lo, "a Warthog must get past first gear's reach of {lo:.3} wu/s, got {v:.3}");
        assert!(v <= hi + 1e-3, "...and must not exceed the drivetrain's overspeed ceiling {hi:.3}, got {v:.3}");
    }

    /// #vehicle-drivetrain  **The wheels and the hull are separate, and the difference is slip.**
    ///
    /// `wheel_travel` integrates the wheels' surface speed; `odometer` integrates the hull's
    /// resolved motion. Held still — no ground, so no traction at all — the wheels must still
    /// turn and the hull must not advance. That is what makes wheelspin visible and it is the
    /// reason the two are separate fields rather than one overloaded one.
    #[test]
    fn the_wheels_turn_even_when_the_hull_does_not() {
        let d = warthog_model();
        // In the air: the drivetrain has no grip, so the traction limit is zero.
        let nothing = |_p: Vec3| None;
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 50.0), ..Default::default() };
        for _ in 0..120 {
            st = d.step(st, DriveInput { throttle: 1.0, slide: 0.0, want_yaw: 0.0, brake: false }, 1.0 / 60.0, &nothing);
        }
        println!("airborne, full throttle for 2 s: wheel travel {:.3} wu, forward odometer {:.4} wu", st.wheel_travel, st.odometer);
        assert!(st.wheel_travel > 1.0, "the wheels must spin with no load, got {:.3} wu", st.wheel_travel);
        assert!(st.odometer.abs() < 0.05, "...and the hull must not be driven by them, got {:.4} wu", st.odometer);
        assert!(!st.grounded);
    }

    // -----------------------------------------------------------------------------------------
    // CONTACT GEOMETRY
    // -----------------------------------------------------------------------------------------

    /// #vehicle-suspension  The derived quantities of the contact set, each computed here a second
    /// way.
    ///
    /// The centroid is the weighted mean; the inertia is the weighted mean square offset. **The
    /// centroid is the reference the torques are taken about and that is why a parked hull is
    /// level**, so the gate also asserts the property that makes it the right reference: the
    /// weighted offsets sum to zero.
    #[test]
    fn the_contact_set_derives_its_centroid_and_inertia_from_the_authored_masses() {
        let c = warthog_contacts();
        let total: f32 = c.points.iter().map(|p| p.weight).sum();
        let want_cx: f32 = c.points.iter().map(|p| p.local.x * p.weight / total).sum();
        println!("centroid x {:+.6}, arithmetic {:+.6}", c.centroid.x, want_cx);
        assert!((c.centroid.x - want_cx).abs() < 1e-6);
        // The weighted offsets sum to zero -- which is exactly "no static pitch or roll torque".
        let sx: f32 = c.points.iter().map(|p| (p.local.x - c.centroid.x) * p.weight / total).sum();
        let sy: f32 = c.points.iter().map(|p| (p.local.y - c.centroid.y) * p.weight / total).sum();
        println!("weighted offset sums: x {sx:+.7} y {sy:+.7}");
        assert!(sx.abs() < 1e-5 && sy.abs() < 1e-5, "the centroid must balance the static load");
        let want_ip: f32 = c.points.iter().map(|p| p.weight / total * (p.local.x - c.centroid.x).powi(2)).sum();
        assert!((c.pitch_inertia - want_ip).abs() < 1e-6, "pitch inertia {} against {want_ip}", c.pitch_inertia);
        // And the refusals: fewer than three points is not a plane, and a massless set is not a
        // load distribution. Neither may be substituted for.
        let one = ContactPoint { local: Vec3::ZERO, radius: 0.1, rest_height: 0.0, weight: 1.0, steer: SteerRole::Fixed };
        assert!(ContactSolver::new(vec![one; 2]).is_none(), "two points do not make a plane");
        assert!(ContactSolver::new(vec![ContactPoint { weight: 0.0, ..one }; 4]).is_none(), "no mass, no load solve");
        assert!(ContactSolver::new(vec![one; MAX_CONTACT_POINTS + 1]).is_none(), "past the engine's own maximum");
    }

    /// #vehicle-suspension  The friction-point flags decide which wheels steer, which is the
    /// engine's own reading (`sub_140E69D40` picks `+sin`, `-sin` or the hull's own orientation).
    #[test]
    fn the_friction_point_flags_say_which_wheels_steer() {
        assert_eq!(SteerRole::from_flags(0x04), SteerRole::Steered);
        assert_eq!(SteerRole::from_flags(0x08), SteerRole::Inverted);
        assert_eq!(SteerRole::from_flags(0x00), SteerRole::Fixed);
        // Both set: the engine tests bit 2 FIRST, so forward steering wins.
        assert_eq!(SteerRole::from_flags(0x0C), SteerRole::Steered);
        assert_eq!(SteerRole::Steered.sign(), 1.0);
        assert_eq!(SteerRole::Inverted.sign(), -1.0);
        assert_eq!(SteerRole::Fixed.sign(), 0.0);
    }

    /// #vehicle-suspension  A hover vehicle authors no wheel spring and must not be given one.
    ///
    /// Measured: every hover vehicle in `forge_halo` authors `ground depth = 0` and `ground damp
    /// factor = 0`. So the solve must still report the contacts and the plane — terrain
    /// conformance is geometry and costs nothing — and must produce exactly zero spring force.
    #[test]
    fn a_vehicle_with_no_authored_spring_gets_no_spring() {
        let c = warthog_contacts();
        let hover = GroundSpring { depth: 0.0, damp: 0.0, max_normal_force: 3.0, slope_cos: [1.0, 0.999], iterations: 1 };
        assert!(!hover.authored(), "depth 0 is not an authored spring");
        let slope = 0.25f32;
        let ramp = move |p: Vec3| Some((p.x * slope, Vec3::new(-slope, 0.0, 1.0).normalize()));
        // **The hull has to be LYING ON the ramp, or only the downhill wheels touch and the plane
        // rule is never exercised** — the first version of this sat a level hull at z = 0.2 on a
        // 0.25 slope, which puts the two uphill wheels 0.11 wu in the air, and the gate went red
        // on its own fixture. With the hull pitched to the ramp every point has the SAME
        // compression, because `x * cos(p) * tan(p) - x * sin(p)` is identically zero: the
        // compression is `pos.x * tan(p) + radius - pos.z` whatever `x` is. So `z = 0.22` with
        // radius 0.25 puts all four wheels 0.03 wu into their springs at once.
        let pitch = slope.atan();
        let lying = glam::Quat::from_rotation_y(-pitch);
        let s = c.solve(&hover, 4.1712594, lying, Vec3::new(0.0, 0.0, 0.22), Vec3::ZERO, (0.0, 0.0), &ramp);
        println!("hover solve: {} touching, normal {:?}, spring accel {}", s.touching_count, s.normal, s.spring_accel_z);
        assert_eq!(s.spring_accel_z, 0.0, "an unauthored spring must produce no force");
        assert_eq!(s.alpha_pitch, 0.0);
        assert_eq!(s.alpha_roll, 0.0);
        assert_eq!(s.touching_count, 4, "...but all four contacts must still be found");
        // ...and the plane with them, so a hover vehicle still conforms.
        let (p, _r) = attitude_from_normal(s.normal, 0.0);
        println!("hover plane pitch {:+.4} rad against atan({slope}) = {:+.4}", p, slope.atan());
        assert!((p - slope.atan()).abs() < 1e-3, "a hover vehicle must still get the ground plane");
    }

    /// #vehicle-suspension  **A contact too steep to stand on gives no support**, which is the
    /// engine's own `ground maximum slope 1` test read from the consumption side.
    #[test]
    fn a_face_steeper_than_the_authored_slope_limit_carries_nothing() {
        let c = warthog_contacts();
        let spring = warthog_spring(); // slope_cos[1] = 0.643 = 50 degrees
        let g = 4.1712594f32;
        // A hull LYING ON a face of a given angle, which is the only way all four of its wheels
        // can be on a steep one: see the note in the hover gate for why every wheel then has the
        // same compression.
        let on_face = |deg: f32| -> ContactSolve {
            let rad = deg.to_radians();
            let slope = rad.tan();
            let face = move |p: Vec3| Some((p.x * slope, Vec3::new(-slope, 0.0, 1.0).normalize()));
            c.solve(
                &spring,
                g,
                glam::Quat::from_rotation_y(-rad),
                Vec3::new(0.0, 0.0, 0.22),
                Vec3::ZERO,
                (0.0, 0.0),
                &face,
            )
        };
        // 40 degrees: inside the limit (cos 40 = 0.766 > the authored 0.643), so it carries.
        let s1 = on_face(40.0);
        // 60 degrees: outside it (cos 60 = 0.5 < 0.643).
        let s2 = on_face(60.0);
        println!(
            "40 deg face: normal.z {:.4} -> spring {:.4} | 60 deg face: normal.z {:.4} -> spring {:.4}",
            s1.normal.z, s1.spring_accel_z, s2.normal.z, s2.spring_accel_z
        );
        assert!(s1.spring_accel_z > 0.0, "a 40 degree face is inside the authored 50 and must carry the hull");
        assert_eq!(s2.spring_accel_z, 0.0, "a 60 degree face is outside it and must carry nothing");
        // ...and the contacts are still FOUND on the steep face, so the refusal is about traction
        // and not about the ray.
        assert_eq!(s2.touching_count, 4, "the steep face must still be detected under all four wheels");
        assert_eq!(s1.touching_count, 4);
        // And the boundary is the AUTHORED one and not some other angle: a face at 49 degrees
        // carries and one at 51 does not, because the tag says 50.
        let inside_edge = on_face(49.0);
        let outside_edge = on_face(51.0);
        println!(
            "49 deg -> spring {:.4} | 51 deg -> spring {:.4}  (authored limit acos(0.643) = {:.2} deg)",
            inside_edge.spring_accel_z,
            outside_edge.spring_accel_z,
            spring.slope_cos[1].acos().to_degrees()
        );
        assert!(inside_edge.spring_accel_z > 0.0, "49 degrees is inside an authored 50");
        assert_eq!(outside_edge.spring_accel_z, 0.0, "51 degrees is outside it");
    }

    /// #gravity  **A vehicle is on the engine's authored world constant**, and the authored
    /// `gravity scale` is the only thing that scales it.
    ///
    /// See [`crate::reach_physics::vehicle_fall_acceleration`] for why: the wheel normal force in
    /// `havok_vehicle_physics.cpp` reads the first float of the engine's own `"physics constants"`
    /// block, which is 4.1712594. The second half of this gate is the one that catches a second
    /// writer: a model whose `gravity_scale` is 0.5 must fall at half the rate, and an earlier
    /// version of this code stored gravity in two places so that setting one did nothing.
    #[test]
    fn a_vehicle_falls_at_the_engines_own_authored_gravity() {
        let d = warthog_model();
        let nothing = |_p: Vec3| None;
        let fall = |scale: f32| -> f32 {
            let m = DriveModel { gravity_scale: scale, ..d.clone() };
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 100.0), ..Default::default() };
            for _ in 0..60 {
                st = m.step(st, DriveInput::default(), 1.0 / 60.0, &nothing);
            }
            -st.vel.z
        };
        let g = crate::reach_physics::vehicle_fall_acceleration();
        println!("one second of free fall: {:.5} wu/s against g {g:.5}", fall(1.0));
        assert!((fall(1.0) - g).abs() < 1e-3, "a vehicle must fall at {g}, got {}", fall(1.0));
        assert!((fall(0.5) - g * 0.5).abs() < 1e-3, "an authored 0.5 `gravity scale` must halve it, got {}", fall(0.5));
        assert!((fall(2.0) - g * 2.0).abs() < 1e-3, "and 2.0 must double it, got {}", fall(2.0));
        // It must NOT be the number the biped lane used to share: 10.4576 has no engine basis.
        assert!(
            (g - crate::reach_physics::MEASURED_FALL_ACCELERATION).abs() > 1.0,
            "the vehicle gravity must not have drifted onto the fitted constant"
        );
    }

    /// #vehicle-suspension  REPRO for *"bro the warthog just flew into the air and the map
    /// glitched out"*. Print-only; the gate that fails on it is
    /// [`a_vehicle_cannot_gain_energy_from_its_own_suspension`].
    #[test]
    fn probe_what_the_suspension_does_with_a_hostile_ground_query() {
        let d = warthog_model();
        let rest = d.contacts.as_ref().unwrap().static_compression(0, &d.spring);
        println!("travel {:.4} wu, resting compression {:.4} wu, force cap {:.2}", d.spring.depth, rest, d.spring.max_normal_force);

        // CASE A: the ground query reports a surface ABOVE the wheel. This is what a downward ray
        // cast from `wheel + 2z` returns when it hits the RIDDEN VEHICLE'S OWN collision hull, and
        // `raycast_objects_excluding(origin, dir, &[])` excludes nothing.
        for above in [0.5f32, 1.25, 2.0] {
            let hostile = move |p: Vec3| Some((p.z + above, Vec3::Z));
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.22), ..Default::default() };
            let mut max_v = 0.0f32;
            for i in 0..600 {
                st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &hostile);
                max_v = max_v.max(st.vel.z);
                if i % 120 == 0 || i == 599 {
                    println!(
                        "  A(+{above}) tick {i:3}: z {:12.3}  vz {:12.3}  c0 {:.4}  touch {}  pitch {:+.3} rate {:+.3}",
                        st.pos.z, st.vel.z, st.suspension[0], st.touching[0], st.pitch, st.pitch_rate
                    );
                }
            }
            println!("  A(+{above}) after 10 s: z {:.3}, peak vz {:.3} wu/s", st.pos.z, max_v);
        }

        // CASE B: a landing. The damper term is NOT inside the force cap, so a hull arriving
        // downward sees `a = cap - v_normal * damp` with no bound on the second half.
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        for drop_from in [1.0f32, 6.0, 30.0] {
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, drop_from), ..Default::default() };
            let mut peak_up = 0.0f32;
            let mut impact = 0.0f32;
            for _ in 0..600 {
                let before = st.vel.z;
                st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &flat);
                if st.touching[0] && before < impact {
                    impact = before;
                }
                peak_up = peak_up.max(st.vel.z);
            }
            println!(
                "  B(drop {drop_from}): impact vz {impact:.3}, peak rebound vz {peak_up:.3} (restitution {:.3}), settled z {:.4}",
                if impact < 0.0 { peak_up / -impact } else { 0.0 },
                st.pos.z
            );
        }

        // CASE C: one wheel buried in a Forge block's side face -- a step under the front axle.
        let step = |p: Vec3| Some((if p.x > 0.0 { 1.0f32 } else { 0.0 }, Vec3::Z));
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.22), ..Default::default() };
        let mut max_v = 0.0f32;
        for i in 0..60 {
            st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &step);
            max_v = max_v.max(st.vel.z);
            if i < 3 || i == 59 {
                println!("  C tick {i:2}: z {:10.3}  vz {:10.3}  pitch {:+.3}  rate {:+.3}", st.pos.z, st.vel.z, st.pitch, st.pitch_rate);
            }
        }
        println!("  C after 1 s: z {:.3}, peak vz {:.3} wu/s", st.pos.z, max_v);
    }

    /// #vehicle-suspension  **A VEHICLE MAY NOT GAIN ENERGY FROM ITS OWN SUSPENSION.**
    ///
    /// *"Bro the warthog just flew into the air and the map glitched out."* Every other gate in
    /// this module asserts a steady state or a relationship, and **this regression walked straight
    /// between them**: the hull's resting compression was right, the ramp pitch was right, the
    /// drivetrain was right, and the thing nobody had written down was that the simulation must
    /// not blow up. So this one asserts the thing that must NOT happen, under inputs chosen to be
    /// hostile rather than typical.
    ///
    /// The bound is stated and derived, not fitted. With the normal force capped at
    /// `maximum normal force contribution` per contact, the most upward acceleration the springs
    /// can produce is `contacts * cap`, so against gravity the hull's net upward acceleration
    /// cannot exceed `contacts * cap - g` -- 4.63 wu/s^2 on a Warthog. It therefore cannot rise
    /// faster than that for as long as it is in contact, and a hull parked on the flat must stay
    /// within a suspension travel of its rest height for ever.
    ///
    /// Four grounds. Two of them are what the real per-wheel ray actually returned before it was
    /// fixed, so they are not hypothetical:
    ///
    /// * **flat** -- the control;
    /// * **a surface ABOVE the wheel**, which is what a ray cast from `wheel + 2z` reports when it
    ///   hits the ridden vehicle's OWN collision hull. This walked the hull upward at 75 wu/s with
    ///   a vertical velocity of 0.85 -- it was being teleported by the bump-stop correction, not
    ///   accelerated;
    /// * **a step under the front axle**, a Forge block's side face. This fed the hull's own
    ///   angular rate back through an uncapped one-sided damper and reached **99,206 wu/s in one
    ///   second**;
    /// * **a sawtooth**, which puts the wheels on alternating faces every frame.
    #[test]
    fn a_vehicle_cannot_gain_energy_from_its_own_suspension() {
        let d = warthog_model();
        let c = d.contacts.as_ref().unwrap();
        // Derived from the authored numbers, without the integrator: the springs' ceiling.
        let g = crate::reach_physics::vehicle_fall_acceleration() * d.gravity_scale;
        let max_net_up = c.points.len() as f32 * d.spring.max_normal_force - g;
        // ...and the fastest the bump-stop correction may add on top of it.
        let budget = max_net_up + MAX_PENETRATION_PUSH_OUT;
        println!(
            "{} contacts x cap {:.2} - g {:.3} = {max_net_up:.3} wu/s/s of net lift; \
             push-out {MAX_PENETRATION_PUSH_OUT:.1} wu/s; budget {budget:.3}",
            c.points.len(),
            d.spring.max_normal_force,
            g
        );

        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        // The two grounds the real ray produced, and one worse.
        let own_hull = |p: Vec3| Some((p.z + 1.25f32, Vec3::Z));
        let step = |p: Vec3| Some((if p.x > 0.0 { 1.0f32 } else { 0.0 }, Vec3::Z));
        let sawtooth = |p: Vec3| Some(((p.x * 7.0).sin() * 0.4, Vec3::Z));
        // `satisfiable` marks whether a ground function describes a surface a hull can actually
        // come to rest on. `own_hull` does NOT: it claims solid ground 1.25 wu above the wheel at
        // every height, so no non-penetration solver can ever satisfy it and any correct one keeps
        // pushing up for ever. For that one the gate asserts what a correct solver must do with an
        // impossible input -- stay BOUNDED and degrade to a slow push-out rather than launch --
        // and the input itself is forbidden upstream by
        // [`the_ground_query_can_never_report_a_surface_above_the_wheel`], which is where it
        // belongs.
        let cases: [(&str, &dyn Fn(Vec3) -> Option<(f32, Vec3)>, bool); 4] = [
            ("flat", &flat, true),
            ("a surface 1.25 wu ABOVE the wheel", &own_hull, false),
            ("a step under the front axle", &step, true),
            ("a sawtooth", &sawtooth, true),
        ];

        for (name, ground, satisfiable) in cases {
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.30), ..Default::default() };
            let start_z = st.pos.z;
            let mut peak_speed = 0.0f32;
            let mut peak_z = start_z;
            // Ten seconds, which is long enough for any of these to diverge: the step case reached
            // 99,206 wu/s inside ONE second.
            for _ in 0..600 {
                st = d.step(st, DriveInput { throttle: 0.0, slide: 0.0, want_yaw: 0.0, brake: true }, 1.0 / 60.0, &ground);
                peak_speed = peak_speed.max(st.vel.length());
                peak_z = peak_z.max(st.pos.z);
            }
            println!(
                "  {name:34}  final z {:9.4}  rise {:8.4}  peak |v| {:9.4}  pitch {:+.3} rate {:+.3}",
                st.pos.z,
                peak_z - start_z,
                peak_speed,
                st.pitch,
                st.pitch_rate
            );
            assert!(st.pos.z.is_finite() && st.vel.is_finite(), "{name}: diverged to z {} v {:?}", st.pos.z, st.vel);
            // The hull may NOT be travelling faster than its own springs and its push-out can
            // account for over one second of continuous lift.
            assert!(
                peak_speed <= budget,
                "{name}: the hull reached {peak_speed:.4} wu/s, past the {budget:.3} its springs and \
                 push-out can produce -- energy is being created"
            );
            // ...and on a ground it CAN rest on, it may not climb. Each of those is bounded above
            // by 1.0 wu of terrain, so a ceiling of that plus two travels is generous and still a
            // thousand times tighter than the 255 wu this used to reach in one second.
            if satisfiable {
                assert!(
                    peak_z - start_z <= 1.0 + 2.0 * d.spring.depth,
                    "{name}: the hull climbed {:.4} wu off ground that rises at most 1.0",
                    peak_z - start_z
                );
            } else {
                // The unsatisfiable one: it must climb at the PUSH-OUT rate and no faster, which
                // is a slow extraction and not flight. Ten seconds at 2 wu/s is 20 wu of travel
                // plus the springs' own contribution; what matters is that it is linear in time
                // rather than the 8e26 the double-counted velocity produced.
                let rise = peak_z - start_z;
                assert!(
                    rise <= 10.0 * budget,
                    "{name}: climbed {rise:.4} wu in 10 s, faster than the {:.3} wu/s the push-out                      and the springs together allow -- that is not an extraction, it is a launch",
                    budget
                );
            }
            // The attitude rate must be bounded too: it is an INPUT to the damper through
            // `rate * d`, and that is the loop that diverged.
            assert!(
                st.pitch_rate.abs() <= MAX_ATTITUDE_RATE + 1e-3 && st.roll_rate.abs() <= MAX_ATTITUDE_RATE + 1e-3,
                "{name}: attitude rate {:+.3}/{:+.3} past the stated bound {MAX_ATTITUDE_RATE}",
                st.pitch_rate,
                st.roll_rate
            );
        }

        // And the force law itself, directly: no `v_normal` may push a contact past the cap. This
        // is the assertion that the previous code failed -- the cap was on the spring half only,
        // so a hull arriving downward at 50 wu/s saw 2.2 + 67.5 per wheel.
        for v in [0.0f32, -1.0, -10.0, -50.0, -500.0] {
            let s = c.solve(
                &d.spring,
                g,
                glam::Quat::IDENTITY,
                Vec3::new(0.0, 0.0, 0.22),
                Vec3::new(0.0, 0.0, v),
                (0.0, 0.0),
                &flat,
            );
            let per_contact = s.spring_accel_z / c.points.len() as f32;
            println!("  arriving at vz {v:7.1}: per-contact accel {per_contact:9.4} against a cap of {:.2}", d.spring.max_normal_force);
            assert!(
                per_contact <= d.spring.max_normal_force + 1e-3,
                "a contact produced {per_contact:.4} wu/s/s at vz {v}, past its authored cap of {:.2}",
                d.spring.max_normal_force
            );
        }
    }

    /// #vehicle-suspension  **THE GROUND QUERY MAY NOT REPORT A SURFACE ABOVE THE WHEEL**, and
    /// it may not see the vehicle it belongs to.
    ///
    /// This is the upstream half of the *"warthog just flew into the air"* regression, and it is
    /// the input that made an unsatisfiable problem reachable. Two defects, both in
    /// [`crate::play_vehicle::step`]'s own closure and neither visible to any model-level gate,
    /// because every one of those supplies a synthetic ground:
    ///
    /// 1. the ray was cast from `wheel + 2z`, so it could return a surface up to 2 wu ABOVE the
    ///    wheel it was testing;
    /// 2. it went through `raycast_solid`, which hardcodes an empty exclude list — and a vehicle's
    ///    own collision hull is solid to a ray cast from inside it. The ride hides the placement
    ///    from the RENDER set, which is deliberately not the pick set a ray walks, so the Warthog's
    ///    own roof was the "ground" under its own wheels.
    ///
    /// Together: `h = wheel.z + 1.0`-ish, every wheel, every frame, at every height, because the
    /// surface moved with the vehicle. The gate reads the source for both, because the closure is
    /// a local in a function that needs a live scene and a cache to call — so the properties are
    /// asserted the only way they can be without one, and asserted loudly.
    #[test]
    fn the_ground_query_can_never_report_a_surface_above_the_wheel() {
        let src = std::fs::read_to_string(format!("{}/src/play_vehicle.rs", env!("CARGO_MANIFEST_DIR")))
            .expect("play_vehicle.rs");
        let body = src
            .split("pub fn step(")
            .nth(1)
            .expect("play_vehicle::step")
            .split("\n/// ")
            .next()
            .expect("the end of step");

        // 1. The ray must start AT the contact point. A contact cannot be above a wheel's own
        //    centre, so starting there makes "ground above the wheel" structurally impossible and
        //    bounds the compression at the wheel radius whatever the geometry does.
        assert!(
            !body.contains("p + Vec3::Z"),
            "the per-wheel ground ray starts ABOVE the contact point again, so it can report a \
             surface above the wheel and the compression is no longer bounded by the radius"
        );
        assert!(
            body.contains("let from = p;"),
            "the per-wheel ground ray must be cast from the contact point itself"
        );

        // 2. It must exclude the ridden placement, and the exclusion must be the ride's own datum
        //    rather than an empty list.
        assert!(
            body.contains("raycast_solid_excluding"),
            "the per-wheel ground ray must use the EXCLUDING raycast, or it sees the vehicle it \
             belongs to"
        );
        assert!(
            body.contains("riding.hidden_datum"),
            "...and what it excludes must be the ride's own placement"
        );
        assert!(
            !body.contains("scene.raycast_solid("),
            "a plain `raycast_solid` excludes nothing and is how the vehicle saw its own hull"
        );

        // And the property itself, on the model side: a contact at or below the wheel centre can
        // never produce a compression larger than the wheel's radius plus its rest height, so the
        // spring cannot be driven past its own cap by geometry.
        let c = warthog_contacts();
        let r = c.points[0].radius;
        let at_centre = move |p: Vec3| Some((p.z, Vec3::Z));
        let s = c.solve(
            &warthog_spring(),
            crate::reach_physics::vehicle_fall_acceleration(),
            glam::Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            (0.0, 0.0),
            &at_centre,
        );
        println!("a contact exactly at the wheel centre: compression {:.4} against a radius of {r:.4}", s.compression[0]);
        assert!(
            s.compression[0] <= r + 1e-4,
            "a contact at the wheel centre gave a compression of {:.4}, past the radius {r:.4}",
            s.compression[0]
        );
    }

    /// #vehicle-ride  **THE TRANSFORM OVERRIDE IS NOT WIRED, and that has to be checked rather
    /// than remembered.**
    ///
    /// `scene::object_world` replaced six separate `object_matrix(o) * scale` sites, and the claim
    /// that it is "bit-for-bit the old behaviour" rests entirely on `obj_transform` being empty on
    /// every frame. That was true when it was written and is exactly the kind of thing a later
    /// change makes false silently -- a populated override moves every object those six sites
    /// draw, pick, light and ray-test, which would look like the whole map glitching.
    ///
    /// So this reads the source: nothing may CALL `set_object_transforms`. When the ride is moved
    /// onto the override lane this gate fails, which is the point -- it has to be a decision with
    /// the animation question answered, not a line that slips in.
    #[test]
    fn the_transform_override_has_no_caller_on_the_shipped_path() {
        let mut callers = Vec::new();
        for f in ["main.rs", "play_mode.rs", "play_vehicle.rs", "headless.rs", "scene.rs", "objscene.rs", "h4_app.rs"] {
            let path = format!("{}/src/{f}", env!("CARGO_MANIFEST_DIR"));
            let Ok(src) = std::fs::read_to_string(&path) else { continue };
            for (i, line) in src.lines().enumerate() {
                let l = line.trim_start();
                // A definition or a forwarding impl is not a call; `scene.set_object_transforms(..)`
                // on a receiver is.
                if l.contains("set_object_transforms")
                    && !l.starts_with("fn ")
                    && !l.starts_with("pub fn ")
                    && !l.starts_with("///")
                    && !l.starts_with("//")
                    && !l.contains("SceneController::set_object_transforms")
                {
                    callers.push(format!("{f}:{}: {l}", i + 1));
                }
            }
        }
        assert!(
            callers.is_empty(),
            "`set_object_transforms` now has a caller, so `obj_transform` can be populated and the \
             six sites behind `scene::object_world` are no longer the authored pose. That is a \
             deliberate change and it needs the driven-animation question answered first -- see the \
             note on `SceneController::obj_transform`. Callers: {callers:?}"
        );
    }

    /// `seconds_to_ticks` is the engine's own `ceil(s * 30)`, and the clutch length follows from
    /// it. Both are quoted constants, so this gate exists to catch them being quietly changed.
    #[test]
    fn the_tick_conversion_is_the_engines_own_ceiling() {
        assert_eq!(seconds_to_ticks(0.0), 0);
        assert_eq!(seconds_to_ticks(1.0 / 30.0), 1);
        assert_eq!(seconds_to_ticks(0.05), 2, "ceil(1.5) = 2 -- and that is `clutch_shift_ticks`");
        assert_eq!(seconds_to_ticks(0.85), 26, "ceil(25.5) = 26");
        assert_eq!(seconds_to_ticks(0.25), 8, "ceil(7.5) = 8");
        assert_eq!(seconds_to_ticks(0.45), 14, "ceil(13.5) = 14");
        assert_eq!(CLUTCH_SHIFT_TICKS as u32, seconds_to_ticks(0.05));
    }
}
