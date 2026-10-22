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

/// The game's tick rate: **60 Hz** (`game_time_globals` +6 = 60, `seconds_per_tick` +8 = 1/60,
/// read from `haloreach_debug.dll`; the MCC TOOL build `reach_tag_test.exe` the first version of
/// this module was read from runs at 30, which is where its `30` came from). The whole simulation
/// runs every tick and the vehicle update has no every-other-tick gate (`sub_1805FD970` ->
/// `sub_18055D020` -> the jeep tick `sub_1805D58B0`), with `dt = seconds_per_tick`
/// (`sub_180644B10`).
pub const TICK_RATE: f32 = 60.0;

/// `tick_rate / 30` (`sub_180644BD0`): the factor the engine's authored-in-30ths conversion is
/// scaled by.
pub const TICK_MULTIPLIER: u32 = 2;

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

/// `sub_1806449A0`, the conversion the gearbox and clutch use: `ceil(s * 30) * tick_multiplier`.
/// Authored seconds are rounded UP to a 30 Hz grid and then scaled to the real tick, so 0.85 s is
/// 26 * 2 = 52 ticks (0.867 s) at 60 Hz.
pub fn seconds_to_ticks(s: f32) -> u32 {
    if !(s > 0.0) {
        return 0;
    }
    (s * 30.0).ceil() as u32 * TICK_MULTIPLIER
}

/// `sub_180644A60`, the conversion the standstill damper uses: `round(s * tick_rate)`.
pub fn seconds_to_ticks_rounded(s: f32) -> u32 {
    (s * TICK_RATE + 0.5) as u32
}

/// `clutch_shift_ticks` = `max(seconds_to_ticks(0.05), tick_multiplier)` = **4 ticks** at 60 Hz
/// (1/15 s, the same time as 2 ticks at 30 Hz), named by the engine's own assert
/// `IN_RANGE_INCLUSIVE(engine->clutch_in_ticks, 0, clutch_shift_ticks-1)` (`vehicle_engine.cpp:335`;
/// the clutch compares against `max(sub_1806449A0(0.05), sub_180644BD0())`).
pub const CLUTCH_SHIFT_TICKS: u8 = 4;

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
            // Neutral: decay. The engine's 0.95 is per TICK, so it is raised to dt*60 here
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
    ///
    /// **CORRECTION (#jeep-law):** the engine does not cap the SPRING with this. `sub_1406CC8C0`
    /// stores the spring's normal force uncapped and uses the cap only to clamp the load a
    /// viscous-regime FRICTION force may draw (`+-cap * mass`) -- see [`WheelGround`]. The cap on
    /// the spring below is kept because this path (a jeep with no `phmo` body, the tank, the
    /// hover vehicles) is the kinematic fallback and the clamp is what keeps it from diverging.
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

// ---------------------------------------------------------------------------------------------
// THE JEEP'S RIGID BODY: the engine's own wheel law, `havok_vehicle_physics.cpp`
// ---------------------------------------------------------------------------------------------
//
// #jeep-law  Everything above this line models a wheeled vehicle as a point that is steered, with
// a spring under it. **The engine does not.** A `human_jeep` is a free rigid body (mass, centre of
// mass and inertia tensor from its `phmo`) and the ONLY things that act on it are, per 60 Hz tick:
//
// 1. the wheel forces of `sub_1406CC8C0` — one normal force and one friction force per `friction
//    point`, applied at the wheel hub and turned into a torque about the centre of mass;
// 2. `sub_1406CF0F0`'s integration of those into linear and angular velocity, with the
//    stationary-vehicle damper;
// 3. Havok, which integrates gravity and the velocity.
//
// Nothing in the Warthog's turn is a kinematic model: the car turns because the front wheels' friction
// force points sideways of the hull's travel, and it slides out of a corner when that force
// saturates. This section is that law, function by function, with the engine's own constants.
//
// ```text
//   sub_140E69D40  jeep tick: builds the per-wheel orientation (steered wheels get the hull basis
//                  rotated by the steering angle: quaternion (0,0,sin(d/2),cos(d/2)) = a rotation
//                  of the FULL angle d), hands the wheels their surface speed, calls ...
//   sub_1406CF0F0  ... the integrator: iteration_count x { wheel forces -> dv, dw }, then the
//                  stationary damper `vel.xy *= (1-a)`, `w.z *= (1-a)`
//   sub_1406CC8C0  the wheel forces (7331 bytes; the friction-point half is quoted below)
//   sub_140E60130  the engine's load for the NEXT tick: the summed longitudinal friction force
// ```
//
// PROVEN = read from the decompilation. INFERRED = said so at the site.

/// The collision tolerance of the engine's Havok world (`sub_18009A06C`, see `reach-havok-world-config`):
/// the wheel's closest-points query (`sub_14034F740`, an `hkpSimpleShapePhantom` with a sphere of the
/// friction point's own `radius`) returns a contact while the sphere's surface is within it of the
/// ground. **PROVEN value, INFERRED use** — the phantom's own tolerance is the world's by
/// construction, but the field was not traced into the query.
pub const WHEEL_CONTACT_TOLERANCE: f32 = 0.0328;

/// #jeep-bounce  A multiplier on the spring's damper term. **INFERRED FROM THE RECORDINGS, and against the
/// literal decompilation**: `sub_1406CC8C0` (tool and debug builds identically) computes
/// `N = mass * (g * compression / depth - v_normal * ground_damp_factor)` with no scale, which with the
/// tags' numbers is a damping ratio of 0.23 (Warthog) / 0.25 (Mongoose) and, on a landing, a rebound of
/// 0.5-0.98 of the dip. The saved films show a median rebound/dip of 0.27 (Warthog, 10 landings), 0.20
/// (Mongoose, 1) -- IQR 0.03..0.51 -- which the model reaches at about 1.5 (0.3) to 2 (0.17) and not at 1
/// (0.8). 2 is the factor between the 30 Hz tick the tags were authored at and the 60 Hz tick the game now
/// runs, which is the one natural explanation (a coefficient written per tick and not rescaled); no code was found
/// that does it. **Set to 1.0 to get the literal law.** Gated against the recordings by
/// `the_landing_rebound_matches_the_recordings` and against the law's own damping ratio by
/// `a_small_drop_rings_down_at_the_damping_ratio_the_tags_imply`.
pub const GROUND_DAMP_SCALE: f32 = 2.0;

/// How far below a hub the ground may be before the wheel's contact plane is not even looked for, wu
/// (beyond the wheel radius). See [`WheelLaw::contact`].
pub const CONTACT_REACH: f32 = 1.0;

/// The floor on a wheel's spring compression (`sub_1406CC8C0`: `fmaxf(radius - dist, 0.001)`).
pub const MIN_WHEEL_COMPRESSION: f32 = 0.001;

/// The literal `-0.0329999998` in `sub_1406CC8C0`'s friction expression (identical in the debug
/// build). **A fixed gain, not a tick length**: it is applied to a FORCE and the integrator then
/// multiplies by `dt`, so the friction acceleration per second does not depend on the tick rate.
pub const FRICTION_TICK_GAIN: f32 = 0.033;

/// The `0.899999976` inside `sqrt(mu * 0.9)` in both friction regimes.
pub const FRICTION_MU_FACTOR: f32 = 0.9;

/// `sub_140E60D40` / `sub_140E60A00`: the stationary damper starts after `ticks(0.45 s)` of standing
/// still, and ramps to full over `0.8 s` after that.
pub const PARK_DELAY_SECONDS: f32 = 0.45;
pub const PARK_RAMP_SECONDS: f32 = 0.8;
/// "Standing still": `|v|^2 <= 3.0625`, i.e. 1.75 wu/s (`sub_140E60D40`, `sub_140E60A00`).
pub const PARK_SPEED_SQUARED: f32 = 3.0625;
/// `sub_1406CF0F0` caps the damper's own acceleration: `sqrt(72900)` = 270 wu/s^2 on the
/// horizontal velocity and `sqrt(324)` = 18 rad/s^2 on the yaw rate.
pub const PARK_LINEAR_ACCEL_CAP: f32 = 270.0;
pub const PARK_ANGULAR_ACCEL_CAP: f32 = 18.0;

/// `sub_140E60B00`: a reverse command against the direction of travel brakes for this long
/// (`game_ticks_to_seconds(counter) < 0.5`), and only above 1 wu/s (`sub_140E60B90`).
pub const BRAKE_COMMAND_SECONDS: f32 = 0.5;
pub const BRAKE_MIN_SPEED: f32 = 1.0;

/// What a `friction_point_definition` flag word says (the tag's own `friction_point_definition_flags`
/// enum: gets damage from region, powered, front turning, rear turning, AI force rear turning,
/// attached to e-brake, can be destroyed).
pub const FP_POWERED: u32 = 0x02;
pub const FP_FRONT_TURNING: u32 = 0x04;
pub const FP_REAR_TURNING: u32 = 0x08;
pub const FP_AI_FORCE_REAR_TURNING: u32 = 0x10;
pub const FP_ATTACHED_TO_EBRAKE: u32 = 0x20;

/// One wheel as the engine reads it: the `friction_point_definition_block` plus its hub marker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelDef {
    /// The hub, model space.
    pub local: Vec3,
    pub radius: f32,
    /// `fraction of total mass`, RAW. The engine multiplies the friction by it directly
    /// (`sqrt(mu*0.9) * def[2]`) — it is a gain, not a fraction (Warthog 6.5).
    pub share: f32,
    /// Flag bit 1: this wheel is driven. A driven wheel's friction opposes its slip against the
    /// engine's surface speed; a free wheel's opposes only its sideways slip.
    pub powered: bool,
    /// How the steering angle reaches this wheel.
    pub steer: SteerRole,
    /// Flag bit 4: steers inverted — but only when the vehicle is AI-driven
    /// (`sub_140E69D40`: the bit is tested only `if (driver.actor_index != -1)`).
    pub ai_rear_turning: bool,
    /// Flag bit 5: this wheel locks with the hand brake and takes the `e-brake` friction values.
    pub ebrake_wheel: bool,
    /// `moving friction velocity diff`: the slip speed below which the wheel is in its viscous
    /// regime and above which it is in its Coulomb regime.
    pub moving_velocity_diff: f32,
    /// The three `e-brake` values, used instead of the vehicle's and the above while the hand brake
    /// holds this wheel.
    pub ebrake_friction: f32,
    pub ebrake_moving_friction: f32,
    pub ebrake_moving_velocity_diff: f32,
}

/// `havok_vehicle_physics_struct`'s ground fields, the way `sub_1406CC8C0` reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelGround {
    /// `ground friction` (`physics[1]`): the static-regime coefficient.
    pub friction: f32,
    /// `ground depth` (`physics[2]`): the spring's scale — a wheel compressed by this carries one
    /// full gravity. PROVEN: the engine's `extension` is the wheel `radius` (`inst + 0x9C =
    /// def[3]`, `sub_1406CE570`), so `ground_depth * extension / radius` is exactly `ground_depth`.
    pub depth: f32,
    /// `ground damp factor` (`physics[3]`).
    pub damp: f32,
    /// `ground moving friction` (`physics[4]`): the kinetic-regime coefficient.
    pub moving_friction: f32,
    /// `ground maximum slope 0/1` (`physics[5]`, `[6]`) as cosines: friction falls linearly from full
    /// at `[0]` to none at `[1]` (`sub_1406CC8C0`: `(cos0 - n.z) / (cos0 - cos1)`).
    pub slope_cos: [f32; 2],
    /// `maximum normal force contribution` (`physics[7]`, default 3 applied): the cap on the load a
    /// STATIC-regime friction force may use. **It does not cap the spring** — the old reading of this
    /// field was wrong; `sub_1406CC8C0` stores the normal force uncapped and only clamps the
    /// friction's load to `+-cap * mass`.
    pub max_normal_force: f32,
}

/// The vehicle's rigid body: what `sub_1406CE570` fetches from Havok every tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    /// `inst + 0x74`, the Havok mass. Needed because the wheel forces are `mass * (...)` and the
    /// engine's load is a FORCE.
    pub mass: f32,
    /// Centre of mass, model space.
    pub com: Vec3,
    /// The inverse inertia tensor in model axes (what `sub_140443EF0` hands the integrator).
    pub inv_inertia: glam::Mat3,
    /// `phmo` linear / angular damping, applied by `reach_havok::HavokBody::step_psi`
    /// (`v *= max(0, 1 - dt * d)`).
    pub lin_damping: f32,
    pub ang_damping: f32,
}

impl RigidBody {
    /// `None` when the tensor does not invert: a body with no inertia cannot be driven and the
    /// caller must say so rather than substitute one.
    pub fn new(mass: f32, com: Vec3, inertia: glam::Mat3, lin_damping: f32, ang_damping: f32) -> Option<RigidBody> {
        if !(mass > 1e-3) || inertia.determinant().abs() < 1e-9 {
            return None;
        }
        let inv_inertia = inertia.inverse();
        if !inv_inertia.is_finite() {
            return None;
        }
        Some(RigidBody { mass, com, inv_inertia, lin_damping, ang_damping })
    }
}

/// What the jeep tick hands the wheels this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelCommand {
    /// The steering angle `a1+0xE8`, radians, left positive.
    pub steer_angle: f32,
    /// `gear_ratio * angular_velocity` (0 under the brake), wu/s — the powered wheels' surface speed.
    pub surface_speed: f32,
    /// `sub_140E60B00`: every wheel locked (the reverse-against-travel brake).
    pub brake_lock: bool,
    /// Control bit 15 (`sub_140E69B30`): the hand brake. Locks the `attached to e-brake` wheels.
    pub handbrake: bool,
    /// Whether a PLAYER drives. `AI force rear turning` applies only to an AI driver.
    pub ai_driven: bool,
}

/// One tick's result of `sub_1406CC8C0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WheelForces {
    /// Summed force on the body, world, in force units (mass * wu/s^2).
    pub force: Vec3,
    /// Summed torque about the centre of mass, world.
    pub torque: Vec3,
    /// `inst + 0x78`: the summed longitudinal friction force on the powered wheels, signed so that a
    /// force RESISTING the wheel's spin is negative for the flywheel — see [`Drivetrain::update`].
    pub ground_torque: f32,
    /// The function's return value: more than half the wheels are in the static regime AND the hull
    /// stands above `ground maximum slope 1`. Gates the stationary damper.
    pub majority_static: bool,
    /// Per wheel: spring compression (wu), whether it touched, and whether it GRIPS (touching and
    /// not sliding — the engine's `physics_mask` bit is the negation).
    pub compression: [f32; MAX_CONTACT_POINTS],
    pub touching: [bool; MAX_CONTACT_POINTS],
    pub gripping: [bool; MAX_CONTACT_POINTS],
    pub count: usize,
    /// The plane through the first three contacts (the tail of `sub_140E69D40`), else the hull's up.
    pub normal: Vec3,
}

/// The contact plane under a wheel: from [`WheelLaw::contact`].
#[derive(Clone, Copy, Debug)]
struct WheelContact {
    /// Signed distance from the hub to the plane, along the normal.
    distance: f32,
    normal: Vec3,
    /// A point on the plane (for the contact-plane normal of the hull).
    point: Vec3,
}

/// The steering law: `sub_140E62270` + `sub_141405EF0`, with the jeep block's own numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteerLaw {
    /// `overdampen cusp angle`, radians (the cache holds it already converted: 0.0305 on the
    /// Warthog, which the tag editor shows as degrees).
    pub cusp: f32,
    pub exponent: f32,
    /// `maximum left turn` (positive) and `maximum right turn` (negative), radians.
    pub max_left: f32,
    pub max_right: f32,
    /// `turn rate`, rad/s: the slew of the wheel angle.
    pub turn_rate: f32,
}

impl SteerLaw {
    /// `sub_140E62270`'s shaping of the facing error: the angle is clamped to +-pi, replaced by
    /// `cusp * (|angle| / cusp) ^ exponent` with its sign, and clamped to +-pi again.
    ///
    /// **Everything below `cusp` is damped and everything above it is amplified**: with the
    /// Warthog's 0.0305 rad and exponent 1.5, 3 degrees of error asks for 4.5 degrees of lock and 10
    /// degrees asks for 21.
    pub fn shape(&self, angle: f32) -> f32 {
        let pi = std::f32::consts::PI;
        let a = angle.clamp(-pi, pi);
        if !(self.cusp > 0.0) {
            return a;
        }
        let sign = if a < 0.0 { -1.0 } else { 1.0 };
        ((a / self.cusp).abs().powf(self.exponent) * (self.cusp * sign)).clamp(-pi, pi)
    }

    /// `sub_141405EF0` (non-wrapping): move `state` toward `target` by `rate * dt`, hold it inside
    /// `[max_right, max_left]`, and snap to the target when the step would cross it.
    pub fn slew(&self, state: f32, target: f32, dt: f32) -> f32 {
        let diff = target - state;
        if diff == 0.0 {
            return state;
        }
        let dir = if diff <= 0.0 { -1.0 } else { 1.0 };
        let mut s = dir * self.turn_rate * dt + state;
        if s > self.max_left {
            s = self.max_left;
        } else if s < self.max_right {
            s = self.max_right;
        }
        let after = target - s;
        let after_dir = if after == 0.0 { 0.0 } else if after <= 0.0 { -1.0 } else { 1.0 };
        if dir != after_dir {
            return target;
        }
        s
    }

    /// The facing error (radians from the hull's forward) at which the shaped request just reaches
    /// full lock — what a stick pushed all the way over has to ask for to give full lock, so that
    /// the same law serves a stick and a look vector. **The mapping from a stick to a facing error
    /// is the player-control layer's and was not decompiled** (INFERRED, see the module note); this
    /// is the one choice that makes full stick = full lock without inventing an angle.
    pub fn saturation_angle(&self) -> f32 {
        if !(self.cusp > 0.0) || !(self.exponent > 0.0) {
            return std::f32::consts::PI;
        }
        let lock = self.max_left.max(-self.max_right).max(1e-3);
        (self.cusp * (lock / self.cusp).powf(1.0 / self.exponent)).min(std::f32::consts::PI)
    }
}

/// The whole wheel law of one vehicle.
#[derive(Clone, Debug, PartialEq)]
pub struct WheelLaw {
    pub wheels: Vec<WheelDef>,
    pub ground: WheelGround,
    pub steer: SteerLaw,
    /// `None` until a caller supplies the vehicle's `phmo` body; the law does not run without it.
    pub body: Option<RigidBody>,
    /// `iteration count`.
    pub iterations: u32,
}

/// Where the ground under a point is: height and an advisory normal.
pub type GroundFn<'a> = &'a dyn Fn(Vec3) -> Option<(f32, Vec3)>;

impl WheelLaw {
    /// The contact plane under a wheel hub, or `None` when the sphere is not within the
    /// tolerance of any ground.
    ///
    /// The engine casts a sphere of the wheel's radius at the hub and takes the closest point's
    /// plane (`sub_14034F740`). The ground callback answers "the surface under this point", so the
    /// query starts at the TOP of the wheel sphere (`hub + radius`): nothing above that can touch
    /// the sphere, and a hub that has sunk into the surface still finds it. The plane normal comes
    /// from four neighbouring heights (central differences over half a radius) because the
    /// callback's own normal only exists for the level BSP and a Forge ramp is not that; it falls
    /// back to the callback's normal across a cliff or a missing sample.
    fn contact(&self, w: &WheelDef, hub: Vec3, ground: GroundFn) -> Option<WheelContact> {
        let top = hub + Vec3::Z * w.radius;
        let (h, n0) = ground(top)?;
        // Out of reach: the ground is more than a wheel and a metre below the hub, so no plane through
        // it can touch the sphere (a face steep enough to is a wall, which the height query cannot
        // describe anyway). Skipping the four neighbour samples here is what keeps an AIRBORNE jeep to
        // one ground query per wheel per tick instead of five.
        if hub.z - h > w.radius + CONTACT_REACH {
            return None;
        }
        let e = (w.radius * 0.5).max(0.05);
        let sample = |dx: f32, dy: f32| ground(top + Vec3::new(dx, dy, 0.0)).map(|(hh, _)| hh);
        let normal = match (sample(e, 0.0), sample(-e, 0.0), sample(0.0, e), sample(0.0, -e)) {
            (Some(px), Some(mx), Some(py), Some(my)) if (px - mx).abs() < 8.0 * e && (py - my).abs() < 8.0 * e => {
                Vec3::new(-(px - mx) / (2.0 * e), -(py - my) / (2.0 * e), 1.0).normalize()
            }
            _ => {
                let n = n0.normalize_or_zero();
                if n.length_squared() < 0.5 { Vec3::Z } else if n.z < 0.0 { -n } else { n }
            }
        };
        let point = Vec3::new(hub.x, hub.y, h);
        let distance = (hub - point).dot(normal);
        (distance - w.radius <= WHEEL_CONTACT_TOLERANCE).then_some(WheelContact { distance, normal, point })
    }

    /// `sub_1406CC8C0`, the friction-point half: the force and torque the wheels put on the body
    /// this tick.
    ///
    /// `g` is the acceleration of gravity the vehicle is on (the spring's reference),
    /// `pos`/`orient` the model origin and rotation, `vel`/`ang` the centre-of-mass velocity and the
    /// angular velocity (world). Pure.
    #[allow(clippy::too_many_arguments)]
    pub fn forces(
        &self,
        g: f32,
        pos: Vec3,
        orient: glam::Quat,
        vel: Vec3,
        ang: Vec3,
        cmd: &WheelCommand,
        ground: GroundFn,
        iteration_fraction: f32,
    ) -> Option<WheelForces> {
        let body = self.body.as_ref()?;
        let rot = glam::Mat3::from_quat(orient);
        let (hull_fwd, hull_left, hull_up) = (rot.x_axis, rot.y_axis, rot.z_axis);
        let com = pos + rot * body.com;
        let cap = self.ground.max_normal_force * body.mass;
        let mut out = WheelForces {
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            ground_torque: 0.0,
            majority_static: false,
            compression: [0.0; MAX_CONTACT_POINTS],
            touching: [false; MAX_CONTACT_POINTS],
            gripping: [false; MAX_CONTACT_POINTS],
            count: self.wheels.len().min(MAX_CONTACT_POINTS),
            normal: hull_up,
        };
        let mut considered = 0usize;
        let mut statics = 0usize;
        let mut contact_pts: Vec<Vec3> = Vec::new();
        for (i, w) in self.wheels.iter().enumerate().take(MAX_CONTACT_POINTS) {
            considered += 1;
            let hub = pos + rot * w.local;
            let r = hub - com;
            let vp = vel + ang.cross(r);
            let Some(c) = self.contact(w, hub, ground) else { continue };
            out.touching[i] = true;
            if contact_pts.len() < 3 {
                contact_pts.push(c.point);
            }
            // The wheel's rolling axis: the hull's forward, rotated about the hull's up by the
            // steering angle. A rotation of the FULL angle (quaternion half-angle built from d*0.5).
            let yaw = match w.steer {
                SteerRole::Steered => cmd.steer_angle,
                SteerRole::Inverted => -cmd.steer_angle,
                SteerRole::Fixed if w.ai_rear_turning && cmd.ai_driven => -cmd.steer_angle,
                SteerRole::Fixed => 0.0,
            };
            let axis = hull_fwd * yaw.cos() + hull_left * yaw.sin();
            let n = c.normal;
            // --- the spring: N = mass * (g * compression / depth - v_normal * damp) -------------
            let comp = (w.radius - c.distance).max(MIN_WHEEL_COMPRESSION).min(w.radius);
            out.compression[i] = comp;
            let n_force = if self.ground.depth > 1e-4 {
                (g * comp / self.ground.depth - vp.dot(n) * self.ground.damp * GROUND_DAMP_SCALE) * body.mass
            } else {
                0.0
            };
            // A wheel on the far side of the hull (the hull is upside down against this plane)
            // gives no support (`sub_1406CC8C0`: `if (n.up + 0.1 < 0) scale = 0`).
            let support = if n.dot(hull_up) + 0.1 < 0.0 { 0.0 } else { 1.0 };
            let mut f = n * (n_force * support);
            // --- the friction -------------------------------------------------------------------
            let proj = axis - n * axis.dot(n);
            let plen = proj.length();
            let mut friction = Vec3::ZERO;
            let mut sliding = false;
            if plen >= 1e-4 {
                let lon = proj / plen;
                let vt = vp - n * vp.dot(n);
                let locked = cmd.brake_lock || (w.ebrake_wheel && cmd.handbrake);
                // The slip: a driven wheel against its surface speed, a free one only sideways.
                let slip = if w.powered { vt - lon * cmd.surface_speed } else { vt - lon * vt.dot(lon) };
                let (slip_vec, slip_speed) = if locked { (vt, vt.length()) } else { (slip, slip.length()) };
                let ebrake_now = w.ebrake_wheel && cmd.handbrake;
                let mu_k = if ebrake_now { w.ebrake_moving_friction } else { self.ground.moving_friction };
                let mu_s = if ebrake_now { w.ebrake_friction } else { self.ground.friction };
                let vel_diff = if ebrake_now { w.ebrake_moving_velocity_diff } else { w.moving_velocity_diff };
                let [c0, c1] = self.ground.slope_cos;
                let slope = if (c0 - c1) > 1e-4 {
                    1.0 - ((c0 - n.z) / (c0 - c1)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if slip_speed <= vel_diff {
                    // VISCOUS: linear in the slip, loaded by the (capped) normal force.
                    let load = n_force.max(-cap).min(cap);
                    let k = (mu_s * FRICTION_MU_FACTOR).sqrt() * w.share;
                    friction = slip_vec * (k * load * slope * -FRICTION_TICK_GAIN);
                    statics += 1;
                } else {
                    // COULOMB: constant in the slip, loaded by the uncapped normal force.
                    let k = (mu_k * FRICTION_MU_FACTOR).sqrt() * w.share;
                    let dir = slip_vec.normalize_or_zero();
                    friction = dir * (k * n_force * slope * -FRICTION_TICK_GAIN);
                    sliding = true;
                }
                if w.powered {
                    // The engine's load: the friction force projected on the wheel axis.
                    let along = friction.dot(axis) * axis;
                    let mag = along.length() * iteration_fraction;
                    if friction.dot(axis) <= 0.0 {
                        out.ground_torque += mag;
                    } else {
                        out.ground_torque -= mag;
                    }
                }
            }
            f += friction;
            out.gripping[i] = !sliding;
            out.force += f;
            out.torque += r.cross(f);
        }
        out.majority_static = considered > 0 && statics > considered / 2 && hull_up.z > self.ground.slope_cos[1];
        if contact_pts.len() == 3 {
            let nrm = (contact_pts[2] - contact_pts[0]).cross(contact_pts[1] - contact_pts[0]);
            if nrm.length() > 1e-4 {
                let nrm = nrm.normalize();
                out.normal = if nrm.z < 0.0 { -nrm } else { nrm };
            }
        }
        Some(out)
    }

    /// `sub_1406CF0F0`: advance the body's velocities by the wheel forces over one tick, with the
    /// stationary damper, `iteration_count` times.
    ///
    /// Returns the new `(vel, ang)` plus the last iteration's forces (for the animation and the
    /// engine load). Positions are NOT advanced here — Havok does that — except between
    /// iterations, where the engine moves its own copy of the pose so the next wheel solve sees it.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_tick(
        &self,
        g: f32,
        mut pos: Vec3,
        mut orient: glam::Quat,
        mut vel: Vec3,
        mut ang: Vec3,
        park: f32,
        cmd: &WheelCommand,
        tick_dt: f32,
        ground: GroundFn,
    ) -> Option<(Vec3, Vec3, WheelForces)> {
        let body = self.body.as_ref()?;
        let n = self.iterations.max(1);
        let dt = tick_dt / n as f32;
        let mut park = park;
        let mut last = None;
        let mut ground_torque = 0.0;
        for it in 0..n {
            let wf = self.forces(g, pos, orient, vel, ang, cmd, ground, 1.0 / n as f32)?;
            ground_torque += wf.ground_torque;
            if !wf.majority_static {
                // `sub_1406CF0F0`: `if (CC8C0() == 0) { park = 0 }` -- and it stays zero.
                park = 0.0;
            }
            let rot = glam::Mat3::from_quat(orient);
            let inv_world = rot * body.inv_inertia * rot.transpose();
            vel += wf.force / body.mass * dt;
            ang += inv_world * wf.torque * dt;
            if park > 0.0 {
                let k = -(park / dt);
                let mut a_lin = Vec3::new(vel.x * k, vel.y * k, 0.0);
                if a_lin.length_squared() > PARK_LINEAR_ACCEL_CAP * PARK_LINEAR_ACCEL_CAP {
                    a_lin *= PARK_LINEAR_ACCEL_CAP / a_lin.length();
                }
                let a_full = ang * k;
                let mut a_yaw = a_full.z;
                if a_full.length_squared() > PARK_ANGULAR_ACCEL_CAP * PARK_ANGULAR_ACCEL_CAP {
                    a_yaw *= PARK_ANGULAR_ACCEL_CAP / a_full.length();
                }
                vel += a_lin * dt;
                ang.z += a_yaw * dt;
            }
            if n > 1 && it + 1 < n {
                let (p, q) = advance_pose(pos, orient, body, vel, ang, dt);
                pos = p;
                orient = q;
            }
            last = Some(wf);
        }
        let mut wf = last?;
        wf.ground_torque = ground_torque;
        Some((vel, ang, wf))
    }
}

/// The signed angle from the hull's forward to the unit's facing, about the hull's up
/// (`sub_140E62270`: `atan2((up x forward) . facing, forward . facing)`), left positive. `facing_yaw`
/// is the facing vector's heading (world radians, horizontal). Zero when the facing is along the up
/// axis or opposite-degenerate.
pub fn facing_error(up: Vec3, forward: Vec3, facing_yaw: f32) -> f32 {
    let a = Vec3::new(facing_yaw.cos(), facing_yaw.sin(), 0.0);
    let c = up.cross(forward).dot(a);
    let d = forward.dot(a);
    if c == 0.0 && d == 0.0 { 0.0 } else { c.atan2(d) }
}

/// Move a rigid body over `dt`: the centre of mass by `vel`, the orientation by `ang` (world),
/// keeping the model origin consistent with the centre of mass. Havok's drift step.
pub fn advance_pose(pos: Vec3, orient: glam::Quat, body: &RigidBody, vel: Vec3, ang: Vec3, dt: f32) -> (Vec3, glam::Quat) {
    let com = pos + orient * body.com + vel * dt;
    let w = ang.length();
    let q = if w * dt > 1e-9 {
        (glam::Quat::from_axis_angle(ang / w, w * dt) * orient).normalize()
    } else {
        orient
    };
    (com - q * body.com, q)
}

/// The stationary damper's strength `a3` (`sub_140E60A00`): zero until the vehicle has stood still
/// for more than `round(0.45 s * 60)` = 27 ticks, then a linear ramp over 0.8 s.
pub fn park_factor(idle_ticks: u8, speed_squared: f32) -> f32 {
    if u32::from(idle_ticks) <= seconds_to_ticks_rounded(PARK_DELAY_SECONDS) || speed_squared >= PARK_SPEED_SQUARED {
        return 0.0;
    }
    ((f32::from(idle_ticks) / TICK_RATE - PARK_DELAY_SECONDS) / PARK_RAMP_SECONDS).clamp(0.0, 1.0)
}

/// `sub_140E60D40`: the standstill counter. It counts ticks of "no throttle (or braking), at least
/// one wheel gripping, and slower than 1.75 wu/s" and resets on anything else.
pub fn idle_counter(prev: u8, throttle_abs: f32, braking: bool, any_gripping: bool, speed_squared: f32) -> u8 {
    if (throttle_abs <= 0.001 || braking) && any_gripping && speed_squared <= PARK_SPEED_SQUARED {
        prev.saturating_add(1).min(0xFE)
    } else {
        0
    }
}

/// `sub_140E60B90`: the reverse-against-travel brake counter. `stick_forward` is the movement
/// stick's forward axis; `forward_dot_velocity` the cosine between the hull's forward and its
/// velocity (the sign is what matters). Counts while the driver pushes against the direction of
/// travel above [`BRAKE_MIN_SPEED`].
pub fn brake_counter(prev: u8, stick_forward: f32, speed: f32, forward_dot_velocity: f32, hold_flag: bool) -> u8 {
    if hold_flag {
        return prev.saturating_add(1).min(0xFE);
    }
    if stick_forward == 0.0 || speed <= BRAKE_MIN_SPEED {
        return 0;
    }
    let opposing = if stick_forward > 0.0 { forward_dot_velocity < 0.0 } else { forward_dot_velocity > 0.0 };
    if opposing {
        prev.saturating_add(1).min(0xFE)
    } else {
        0
    }
}

/// `sub_140E60B00`: is the brake on. The first half second of an opposing command; after that only
/// the persistent control flag.
pub fn brake_active(counter: u8, hold_flag: bool) -> bool {
    if counter == 0 {
        return false;
    }
    f32::from(counter) / TICK_RATE < BRAKE_COMMAND_SECONDS || hold_flag
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
    pub(super) fn warthog_drivetrain() -> Drivetrain {
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
    pub(super) fn warthog_model() -> DriveModel {
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
            wheel_law: None,
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
    /// * a shift takes `max(ceil(0.05*30) * 2, 2) = 4` ticks of clutch (1/15 s at 60 Hz), during
    ///   which the throttle the engine sees is zero.
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

        // --- THE CLUTCH: four 60 Hz ticks, throttle cut ------------------------------------
        // `clutch_shift_ticks = max(ceil(0.05 * 30) * 2, 2) = 4` ticks of 1/60 s -- the same 1/15 s
        // the 30 Hz tool build gets from 2 ticks. The shift is requested on tick 1; the clutch goes
        // in over ticks 1..=4; the new gear is engaged on tick 5.
        let h = 1.0 / 60.0;
        let mut st = EngineState { gear: 1, angular_velocity: m * 0.95, transition: 1.0, ..Default::default() };
        dt.update(&mut st, 1.0, 2, false, h);
        assert_eq!(st.gear, 1, "tick 1 of the shift: still in the old gear");
        assert_eq!(st.clutch_in_ticks, 1, "...with the clutch going in");
        assert_eq!(st.desired_gear, 2, "...and the desired gear recorded");
        // **The throttle cut lands on tick TWO, not tick one**, and that is the engine's own
        // ordering: `sub_140E601B0` reads `clutch_in_ticks` at the TOP of the function, before
        // the shift logic that sets it. So the first tick of a shift still sees the real throttle.
        assert_eq!(st.transition, 1.0, "tick 1 still sees the real throttle, so the blend holds");
        for tick in 2..=4u8 {
            dt.update(&mut st, 1.0, 2, false, h);
            assert_eq!(st.clutch_in_ticks, tick, "tick {tick}: the clutch is still going in");
            assert_eq!(st.gear, 1, "tick {tick}: not engaged yet");
        }
        assert_eq!(st.clutch_in_ticks, CLUTCH_SHIFT_TICKS, "tick 4: the clutch is fully in");
        assert!(st.transition < 1.0, "...and the throttle has been cut, so the loaded blend falls");
        dt.update(&mut st, 1.0, 2, false, h);
        assert_eq!(st.gear, 2, "tick 5: the shift completes");
        assert_eq!(st.clutch_in_ticks, 0, "...and the clutch comes back out");
        assert_eq!(st.desired_gear, GEAR_NONE, "...with no shift pending");
        assert_eq!(st.upshift_ticks, 0, "...and both shift timers reset");
        assert_eq!(st.downshift_ticks, 0);
        // The same wall time as the 30 Hz build: 4 ticks of 1/60 = 2 ticks of 1/30 = 0.0667 s.
        assert!((CLUTCH_SHIFT_TICKS as f32 * h - 2.0 / 30.0).abs() < 1e-6);
    }

    /// #vehicle-drivetrain  **The flywheel is integrated, clamped and frame-rate independent.**
    ///
    /// * `aav += dt * (torque + ratio*ground_torque) / engine_moment` — one tick's worth is
    ///   arithmetic on the fixture: 4000 / 2000 * dt.
    /// * it is floored at `0.05 * max` and ceilinged at `1.3 * max`;
    /// * in neutral it decays 0.95 per TICK, which must be the same after one second whatever the
    ///   frame rate — the trap `play_mode::clock` records, and the reason the decay is raised to
    ///   `dt * 30` rather than applied once a frame (`dt * 60`).
    #[test]
    fn the_flywheel_integrates_the_authored_torque_and_clamps_it() {
        let dt = warthog_drivetrain();
        let m = dt.max_angular_velocity;
        // ONE TICK of the integration, from a gear that is already engaged so no clutch runs.
        let h = 1.0 / 60.0;
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
        for _ in 0..1800 {
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
        // Independently: 0.95 per tick (the literal is not scaled by the tick rate) for the 60 ticks of a second.
        let want = 9.0 * 0.95f32.powi(60);
        println!("one second of neutral: 30 Hz {a:.5} | 60 Hz {b:.5} | 120 Hz {c:.5}, 0.95^60 says {want:.5}");
        assert!((a - want).abs() < 1e-3, "one second in neutral must be 0.95^60, got {a:.5} against {want:.5}");
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
        // `sub_1806449A0` = ceil(s * 30) * (tick_rate / 30), tick_rate = 60.
        assert_eq!(seconds_to_ticks(0.0), 0);
        assert_eq!(seconds_to_ticks(1.0 / 30.0), 2, "one 30ths-step is two 60 Hz ticks");
        assert_eq!(seconds_to_ticks(0.05), 4, "ceil(1.5) * 2 -- and that is `clutch_shift_ticks`");
        assert_eq!(seconds_to_ticks(0.85), 52, "ceil(25.5) * 2 = 26 * 2");
        assert_eq!(seconds_to_ticks(0.25), 16, "ceil(7.5) * 2");
        assert_eq!(seconds_to_ticks(0.45), 28, "ceil(13.5) * 2: the engine's idle-reset time");
        assert_eq!(CLUTCH_SHIFT_TICKS as u32, seconds_to_ticks(0.05));
        // `sub_180644A60` = round(s * 60): the standstill damper's delay is NOT on the 30ths grid.
        assert_eq!(seconds_to_ticks_rounded(0.45), 27);
        assert_eq!(seconds_to_ticks_rounded(0.5), 30);
    }
}

#[cfg(test)]
mod jeep_gates {
    //! #jeep-law  Gates for the engine's rigid-body wheel law.
    //!
    //! Two kinds. The ones without `#[ignore]` run on a FIXTURE built from the real Warthog's
    //! tag numbers (measured out of `forge_halo`, written down below) so they run in every suite;
    //! the `#[ignore]`d ones read the real Warthog and Mongoose out of the cache — the tags, the
    //! wheel markers, the real torque curves, the `phmo` body — and need the native library:
    //! `cargo test -p hms-app --profile quick jeep_gates -- --ignored --nocapture --test-threads=1`
    //!
    //! **Every expectation is derived from the tag numbers by physics that is not the code under
    //! test** (the static balance of a spring-borne body, the ballistic arc, the algebra of the
    //! steering curve), or is a RELATIONSHIP (doubling this doubles that). None is a fitted output.

    use super::*;
    use crate::reach_vehicle::{DriveInput, DriveModel, DriveState};

    // -----------------------------------------------------------------------------------------
    // FIXTURE: the Warthog's own measured numbers
    // -----------------------------------------------------------------------------------------

    const FRONT_X: f32 = 0.6342681;
    const REAR_X: f32 = -0.6657315;
    const TRACK_Y: f32 = 0.39;
    const HUB_Z: f32 = 0.19877;
    const RADIUS: f32 = 0.25;

    /// The Warthog's `friction_point` blocks (`forge_halo`): flags 0x46 front / 0x72 rear, share
    /// 6.5 / 6.333, moving friction velocity diff 7.8 / 1.0, rear e-brake 1.55 / 8.25 / 3.0.
    fn fixture_law() -> WheelLaw {
        let wheel = |x: f32, y: f32, front: bool| WheelDef {
            local: Vec3::new(x, y, HUB_Z),
            radius: RADIUS,
            share: if front { 6.5 } else { 6.333 },
            powered: true,
            steer: if front { SteerRole::Steered } else { SteerRole::Fixed },
            ai_rear_turning: !front,
            ebrake_wheel: !front,
            moving_velocity_diff: if front { 7.8 } else { 1.0 },
            ebrake_friction: if front { 0.0 } else { 1.55 },
            ebrake_moving_friction: if front { 0.0 } else { 8.25 },
            ebrake_moving_velocity_diff: if front { 0.0 } else { 3.0 },
        };
        WheelLaw {
            wheels: vec![
                wheel(FRONT_X, TRACK_Y, true),
                wheel(FRONT_X, -TRACK_Y, true),
                wheel(REAR_X, TRACK_Y, false),
                wheel(REAR_X, -TRACK_Y, false),
            ],
            ground: WheelGround {
                friction: 3.37,
                depth: 0.125,
                damp: 1.35,
                moving_friction: 27.5,
                slope_cos: [0.766, 0.643],
                max_normal_force: 2.2,
            },
            steer: SteerLaw { cusp: 0.030543262, exponent: 1.5, max_left: 1.0471976, max_right: -1.0471976, turn_rate: 9.424778 },
            body: RigidBody::new(
                3300.0,
                Vec3::new(0.041846935, 0.0, 0.32126704),
                glam::Mat3::from_diagonal(Vec3::new(353.50116, 1291.3615, 1465.7584)),
                0.0,
                0.335,
            ),
            iterations: 1,
        }
    }

    fn fixture() -> DriveModel {
        let mut m = super::gates::warthog_model();
        m.wheel_law = Some(fixture_law());
        m.gravity_scale = 1.0;
        m
    }

    fn flat(_p: Vec3) -> Option<(f32, Vec3)> {
        Some((0.0, Vec3::Z))
    }

    /// A plane rising along +x at `deg` degrees, with its exact normal.
    fn incline(deg: f32) -> impl Fn(Vec3) -> Option<(f32, Vec3)> {
        let t = deg.to_radians().tan();
        move |p: Vec3| Some((p.x * t, Vec3::new(-t, 0.0, 1.0).normalize()))
    }

    fn run(
        d: &DriveModel,
        mut st: DriveState,
        input: DriveInput,
        secs: f32,
        fps: f32,
        ground: &dyn Fn(Vec3) -> Option<(f32, Vec3)>,
        mut each: impl FnMut(f32, &DriveState),
    ) -> DriveState {
        let n = (secs * fps).round() as usize;
        for i in 0..n {
            st = d.step(st, input, 1.0 / fps, ground);
            each((i + 1) as f32 / fps, &st);
        }
        st
    }

    /// Where the model origin sits when the hubs rest `radius - depth/4` above the flat ground.
    fn rest_z() -> f32 {
        RADIUS - 0.125 / 4.0 - HUB_Z
    }

    fn at(z: f32) -> DriveState {
        DriveState { pos: Vec3::new(0.0, 0.0, z), ..Default::default() }
    }

    // -----------------------------------------------------------------------------------------
    // THE WEIGHT
    // -----------------------------------------------------------------------------------------

    /// **A parked jeep is carried by its four springs at the compression the tag implies.**
    ///
    /// Derived without the solver. The body is in equilibrium, so the four normal forces sum to its
    /// weight (`g * compression / depth` per wheel, so the four compressions sum to `ground
    /// depth`) and their moment about the centre of mass is zero (so the load splits front/rear by
    /// where the centre of mass sits between the axles: `u_front = (x_com - x_rear) / (x_front -
    /// x_rear)`). Both come from the tag's numbers and the wheel and centre-of-mass positions.
    ///
    /// Starts 0.18 wu above its rest so it must fall and land (liveness), and averages the last two
    /// seconds.
    #[test]
    fn a_parked_jeep_is_carried_by_its_springs_at_the_compression_the_tag_implies() {
        let d = fixture();
        let law = d.wheel_law.as_ref().unwrap();
        let body = law.body.as_ref().unwrap();
        let (mut sum, mut front, mut n) = (0.0f32, 0.0f32, 0u32);
        let start = rest_z() + 0.18;
        let st = run(&d, at(start), DriveInput::default(), 6.0, 60.0, &flat, |t, s| {
            if t > 4.0 {
                sum += s.suspension[..4].iter().sum::<f32>();
                front += s.suspension[0] + s.suspension[1];
                n += 1;
            }
        });
        let (sum, front) = (sum / n as f32, front / n as f32);
        let u_front = (body.com.x - REAR_X) / (FRONT_X - REAR_X);
        let want_sum = law.ground.depth;
        let want_front = law.ground.depth * u_front;
        println!("sum {sum:.5} (want {want_sum:.5}) front pair {front:.5} (want {want_front:.5}) z {:.4}", st.pos.z);
        assert!(st.pos.z < start - 0.1, "the jeep must have FALLEN to its springs: z {:.4} from {start:.4}", st.pos.z);
        assert!(st.touching[..4].iter().all(|&t| t), "all four wheels must be down");
        assert!((sum / want_sum - 1.0).abs() < 0.02, "the springs carry the weight: sum of compressions {sum:.5}, the tag implies {want_sum:.5}");
        assert!((front / want_front - 1.0).abs() < 0.03, "the load splits by the centre of mass: front {front:.5} vs {want_front:.5}");
        assert!(st.roll.abs() < 1e-3, "a symmetric jeep parks level, roll {:+.5}", st.roll);
    }

    // -----------------------------------------------------------------------------------------
    // GRAVITY AND THE JUMP
    // -----------------------------------------------------------------------------------------

    /// **A jump goes as high and stays up as long as the gravity in force says.**
    ///
    /// A ballistic arc has no free parameters: launched at `vz`, the apex is `vz^2 / 2g` above the
    /// launch (a little less under Havok's 60 Hz semi-implicit step, which the expectation
    /// reproduces). The expectation is that arc for the engine's `WORLD_GRAVITY` times the authored
    /// `gravity scale` — and the RELATIONSHIP that doubling the scale halves the apex. Horizontal velocity
    /// must not change in the air (nothing acts on a jeep with no wheel down), and neither may the
    /// attitude.
    #[test]
    fn a_jump_follows_the_arc_the_gravity_in_force_gives() {
        let g = crate::reach_physics::vehicle_fall_acceleration();
        let launch = |scale: f32| {
            let mut d = fixture();
            d.gravity_scale = scale;
            let mut st = at(3.0);
            st.vel = Vec3::new(5.0, 0.0, 3.0);
            let (mut apex, mut touch) = (st.pos.z, None);
            let mut vx_min = f32::MAX;
            let mut vx_max = f32::MIN;
            let mut tilt = 0.0f32;
            run(&d, st, DriveInput::default(), 3.0, 60.0, &flat, |t, s| {
                if touch.is_none() {
                    apex = apex.max(s.pos.z);
                    if s.touching[..4].iter().any(|&x| x) {
                        touch = Some(t);
                    } else {
                        vx_min = vx_min.min(s.vel.x);
                        vx_max = vx_max.max(s.vel.x);
                        tilt = tilt.max(s.basis().angle_between(glam::Quat::IDENTITY));
                    }
                }
            });
            (apex - 3.0, touch, vx_min, vx_max, tilt)
        };
        let (rise, touch, vx_min, vx_max, tilt) = launch(1.0);
        // The arc as Havok steps it: once per 60 Hz tick, semi-implicit (gravity changes the velocity,
        // then the body moves with it). A tiny loop over the ARC, not over the code under test; the
        // continuous formulas are checked against it as well.
        let dt = 1.0 / 60.0;
        let (mut v, mut z, mut apex) = (3.0f32, 0.0f32, 0.0f32);
        for _ in 0..60 {
            v -= g * dt;
            z += v * dt;
            apex = apex.max(z);
        }
        let want_rise = apex;
        let cont_rise = 3.0f32 * 3.0 / (2.0 * g);
        // A wheel is "down" at the first tick that starts with its sphere within the tolerance.
        let thr = RADIUS + 0.0328_f32 - HUB_Z;
        let (mut v, mut z, mut k) = (3.0f32, 3.0f32, 0u32);
        while z > thr {
            v -= g * dt;
            z += v * dt;
            k += 1;
        }
        let want_t = k as f32 * dt;
        let t = touch.expect("it must come back down");
        println!("apex {rise:.4} (discrete arc {want_rise:.4}, continuous {cont_rise:.4}), first wheel down at {t:.3} s (want {want_t:.3}), vx {vx_min:.4}..{vx_max:.4}, tilt {tilt:.5}");
        assert!((rise / want_rise - 1.0).abs() < 0.01, "apex {rise:.4} against {want_rise:.4}: gravity in force is not {g}");
        assert!((rise / cont_rise - 1.0).abs() < 0.06, "and it must be the same arc as the continuous one, {cont_rise:.4}");
        assert!((t - want_t).abs() < 0.06, "airtime to first contact {t:.3} against {want_t:.3}");
        assert!(vx_max - vx_min < 1e-3 && (vx_min - 5.0).abs() < 1e-3, "nothing may act on a jeep in the air: vx {vx_min}..{vx_max}");
        assert!(tilt < 1e-3, "a jeep with no angular velocity must not tumble in the air: {tilt}");
        // THE RELATIONSHIP: twice the gravity, half the height.
        let (rise2, ..) = launch(2.0);
        println!("gravity scale 2: apex {rise2:.4}");
        assert!((rise2 / rise - 0.5).abs() < 0.03, "doubling `gravity scale` must halve the apex: {rise2:.4} vs {rise:.4}");
    }

    /// **The body's own damping bleeds a spin in the air.** Havok multiplies the angular velocity by
    /// `max(0, 1 - dt * damping)` once per 60 Hz tick (`reach_havok`); the Warthog's `phmo` authors
    /// 0.335 and no linear damping. After one second of free flight a 3 rad/s spin must read
    /// `3 * (1 - 0.335/60)^60`, and the linear velocity must not have changed at all.
    #[test]
    fn a_spinning_jeep_in_the_air_loses_spin_to_the_bodys_own_damping() {
        let d = fixture();
        let mut st = at(200.0);
        st.ang_vel = Vec3::new(0.0, 0.0, 3.0);
        st.vel = Vec3::new(4.0, 0.0, 0.0);
        let end = run(&d, st, DriveInput::default(), 1.0, 60.0, &flat, |_, _| {});
        let want = 3.0 * (1.0f32 - 0.335 / 60.0).powi(60);
        println!("spin after 1 s: {:.4} (want {want:.4}), vx {:.5}", end.ang_vel.z, end.vel.x);
        assert!((end.ang_vel.z / want - 1.0).abs() < 0.005, "angular damping: {} vs {want}", end.ang_vel.z);
        assert!((end.vel.x - 4.0).abs() < 1e-4, "no linear damping authored, none applied: {}", end.vel.x);
    }

    // -----------------------------------------------------------------------------------------
    // THE STEERING
    // -----------------------------------------------------------------------------------------

    /// **The steering curve, evaluated by hand.** The goldens were computed outside the program
    /// with the Warthog's `cusp` 0.0305433 rad and exponent 1.5: the shaped request is
    /// `cusp * (|a|/cusp)^1.5` with the sign of `a`, clamped to pi; the wheel angle slews toward it
    /// at `turn rate` * one tick and snaps when a step would cross.
    #[test]
    fn the_steering_curve_is_the_engines() {
        let s = fixture_law().steer;
        for (a, want) in [(0.01f32, 0.005721927), (0.030543262, 0.030543262), (0.1, 0.18094321), (0.2, 0.5117847), (-0.2, -0.5117847), (3.0, std::f32::consts::PI)] {
            let got = s.shape(a);
            assert!((got - want).abs() < 2e-5, "shape({a}) = {got}, the curve says {want}");
        }
        // Slew: 0.3141593 rad per tick; four ticks to full lock, and the last one snaps.
        let mut w = 0.0f32;
        let mut ticks = 0;
        let mut seen = Vec::new();
        while w < s.max_left && ticks < 20 {
            w = s.slew(w, 5.0, 1.0 / 30.0);
            seen.push(w);
            ticks += 1;
        }
        println!("slew to an unreachable request: {seen:?}");
        assert_eq!(ticks, 4, "9.4248 rad/s is 0.31416 rad per tick: full lock (1.0472) on the 4th");
        assert!((seen[0] - 0.31415927).abs() < 1e-5 && (seen[2] - 0.9424778).abs() < 1e-5);
        assert_eq!(seen[3], s.max_left, "held at the authored maximum, not past it");
        // and a reachable request is hit exactly rather than overshot
        assert_eq!(s.slew(0.0, 0.1, 1.0 / 30.0), 0.1);
        assert_eq!(s.slew(0.5, -0.9, 1.0 / 30.0), 0.5 - 0.31415927);
    }

    /// **The jeep steers toward its FACING, by the curve: lock = max * (error / saturation)^exponent.**
    /// The facing is `DriveInput::want_yaw` (the engine's `unit +0x1E4` facing vector); a facing
    /// error of `s * saturation` (the error at which the shaped request just reaches full lock)
    /// must give a wheel angle of `max * s^1.5` -- algebra on the tag's cusp and exponent, not a
    /// fit. The lateral STICK (`slide`) must steer nothing, and a parked jeep never turns however
    /// hard over the wheels are.
    ///
    /// The reported defect, as a property: *"I turn and the whole warthog pivots on a dime"*.
    #[test]
    fn a_jeep_steers_toward_its_facing_by_the_overdampen_curve_and_a_parked_jeep_never_turns() {
        let d = fixture();
        let law = d.wheel_law.as_ref().unwrap();
        let st = run(&d, at(rest_z()), DriveInput::default(), 3.0, 60.0, &flat, |_, _| {});
        let sat = law.steer.saturation_angle();
        for s in [0.25f32, 0.5, 1.0] {
            let end = run(&d, st, DriveInput { want_yaw: st.yaw + s * sat, ..Default::default() }, 1.0, 60.0, &flat, |_, _| {});
            let want = law.steer.max_left * s.powf(law.steer.exponent);
            println!("facing error {:.4} rad: lock {:+.4} rad, curve says {want:+.4}, yaw change {:+.6}", s * sat, end.steer_angle, end.yaw - st.yaw);
            assert!((end.steer_angle - want).abs() < 2e-3, "facing error {s} x saturation: lock {} vs {want}", end.steer_angle);
            assert!((end.yaw - st.yaw).abs() < 1e-4, "a parked jeep with the wheels hard over turned {:+.6} rad", end.yaw - st.yaw);
            // the other side is the mirror image
            let right = run(&d, st, DriveInput { want_yaw: st.yaw - s * sat, ..Default::default() }, 1.0, 60.0, &flat, |_, _| {});
            assert!((right.steer_angle + want).abs() < 2e-3, "right lock {} vs {}", right.steer_angle, -want);
        }
        // THE LATERAL STICK DOES NOT STEER: A/D with the facing on the hull leave the wheels straight.
        let keys = run(&d, st, DriveInput { slide: 1.0, want_yaw: st.yaw, ..Default::default() }, 1.0, 60.0, &flat, |_, _| {});
        assert!(keys.steer_angle.abs() < 1e-6 && (keys.yaw - st.yaw).abs() < 1e-6, "the left stick steered a jeep: {}", keys.steer_angle);
        // LIVENESS of "no turning": the same lock with the engine driving DOES turn it.
        let moving = run(&d, st, DriveInput { throttle: 1.0, want_yaw: st.yaw + 1.5, ..Default::default() }, 2.0, 60.0, &flat, |_, _| {});
        assert!((moving.yaw - st.yaw).abs() > 0.5, "driven at full lock it must turn, got {:+.4} rad", moving.yaw - st.yaw);
        // The facing error is the signed angle about the hull's up, the short way round.
        assert!((facing_error(Vec3::Z, Vec3::X, 3.0) - 3.0).abs() < 1e-5);
        assert!((facing_error(Vec3::Z, Vec3::X, -3.0) + 3.0).abs() < 1e-5);
        // Hull pointing -x, facing heading 3.0 rad = (-0.99, +0.14): left of -x is -y, so that is 0.1416 to the RIGHT.
        assert!((facing_error(Vec3::Z, -Vec3::X, 3.0) + (std::f32::consts::PI - 3.0)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------------------------
    // INDEPENDENCE OF THE FRAME RATE
    // -----------------------------------------------------------------------------------------

    /// **The result does not depend on the display.** The engine's logic runs on a 30 Hz clock;
    /// the frame only decides how often the body is looked at. Driven for three seconds at 30,
    /// 60 and 144 fps the jeep must be in the same place.
    #[test]
    fn the_jeep_is_in_the_same_place_at_any_frame_rate() {
        let d = fixture();
        let go = |fps: f32| {
            let st = run(&d, at(rest_z()), DriveInput::default(), 1.0, fps, &flat, |_, _| {});
            run(&d, st, DriveInput { throttle: 1.0, slide: 0.3, ..Default::default() }, 3.0, fps, &flat, |_, _| {})
        };
        let a = go(30.0);
        let b = go(60.0);
        let c = go(144.0);
        println!("30 fps x {:.4} y {:.4} | 60 fps x {:.4} y {:.4} | 144 fps x {:.4} y {:.4}", a.pos.x, a.pos.y, b.pos.x, b.pos.y, c.pos.x, c.pos.y);
        assert!(b.pos.x > 3.0, "it must have driven somewhere");
        for (name, o) in [("30", &a), ("144", &c)] {
            assert!((o.pos - b.pos).length() < 0.02 * b.pos.length(), "{name} fps ended {:?} but 60 fps {:?}", o.pos, b.pos);
        }
    }

    /// **A wheel is down while its sphere is within the Havok world's collision tolerance of the
    /// ground (0.0328 wu), and not beyond.** The hubs are lifted so the sphere's surface is 0.02
    /// above the ground (down) and 0.05 above it (free), and the springs' compression there is the
    /// floor of 0.001 -- the sphere is touching but not pressing.
    #[test]
    fn a_wheel_is_down_within_the_collision_tolerance_and_free_beyond_it() {
        let law = fixture_law();
        let g = crate::reach_physics::vehicle_fall_acceleration();
        let cmd = WheelCommand { steer_angle: 0.0, surface_speed: 0.0, brake_lock: false, handbrake: false, ai_driven: false };
        let at_gap = |gap: f32| {
            let pos = Vec3::new(0.0, 0.0, RADIUS + gap - HUB_Z);
            law.forces(g, pos, glam::Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &cmd, &flat, 1.0).unwrap()
        };
        let (near, far, pressed) = (at_gap(0.02), at_gap(0.05), at_gap(-0.03));
        println!("gap 0.02: touching {:?} comp {:.4}; gap 0.05: touching {:?}; pressed 0.03: comp {:.4}", &near.touching[..4], near.compression[0], &far.touching[..4], pressed.compression[0]);
        assert!(near.touching[..4].iter().all(|&t| t), "0.02 wu off the ground is inside the 0.0328 tolerance");
        assert!((near.compression[0] - 0.001).abs() < 1e-6, "touching but not pressing: the compression floor, got {}", near.compression[0]);
        assert!(far.touching[..4].iter().all(|&t| !t) && far.force == Vec3::ZERO, "0.05 wu off the ground is beyond it: no contact, no force");
        assert!((pressed.compression[0] - 0.03).abs() < 1e-4, "pressed 0.03 into the ground: compression {}", pressed.compression[0]);
    }

    /// **No state is fatal.** Thrown at 25 wu/s with a violent tumble, from height, a jeep must stay
    /// finite, must never put a wheel hub through the ground, and must come to rest -- the class of
    /// report is *"the warthog just flew into the air and the map glitched out"*. Run at three
    /// frame rates, because the failure mode that matters is numerical.
    #[test]
    fn a_thrown_and_tumbling_jeep_stays_finite_above_the_ground_and_comes_to_rest() {
        let d = fixture();
        let law = d.wheel_law.as_ref().unwrap();
        for fps in [30.0f32, 60.0, 144.0] {
            let mut st = at(4.0);
            st.vel = Vec3::new(20.0, -10.0, 15.0);
            st.ang_vel = Vec3::new(5.0, 3.0, 8.0);
            let (mut lowest, mut finite, mut fastest) = (f32::MAX, true, 0.0f32);
            let end = run(&d, st, DriveInput::default(), 40.0, fps, &flat, |_, s| {
                let q = s.basis();
                finite &= s.pos.is_finite() && s.vel.is_finite() && s.ang_vel.is_finite() && q.is_finite();
                fastest = fastest.max(s.vel.length());
                for w in &law.wheels {
                    lowest = lowest.min((s.pos + q * w.local).z);
                }
            });
            println!("{fps} fps: finite {finite}, lowest hub {lowest:+.4}, fastest {fastest:.1}, end speed {:.4} spin {:.4} at {:?}", end.vel.length(), end.ang_vel.length(), end.pos);
            assert!(finite, "{fps} fps: a state went non-finite");
            assert!(lowest > -0.02, "{fps} fps: a wheel hub went {lowest:+.4} through the ground");
            assert!(fastest < 60.0, "{fps} fps: it gained energy: {fastest:.1} wu/s");
            assert!(end.vel.length() < 0.5 && end.ang_vel.length() < 0.5, "{fps} fps: it must come to rest, not keep moving");
        }
    }

    /// **A wheel that the hull is upside down against gives no support.** `sub_1406CC8C0` zeroes the
    /// normal force when `contact normal . hull up + 0.1 < 0`. The same hubs, 0.02 above the
    /// ground, with the hull upright and then rolled 180 degrees (hubs mirrored about the origin
    /// plane): upright pushes up; inverted pushes nothing.
    #[test]
    fn a_wheel_the_hull_is_upside_down_against_gives_no_support() {
        let law = fixture_law();
        let g = crate::reach_physics::vehicle_fall_acceleration();
        let cmd = WheelCommand { steer_angle: 0.0, surface_speed: 0.0, brake_lock: false, handbrake: false, ai_driven: false };
        let up = law.forces(g, Vec3::new(0.0, 0.0, RADIUS - HUB_Z - 0.03), glam::Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &cmd, &flat, 1.0).unwrap();
        // Rolled 180 about x the hub height (+0.1988) becomes -0.1988: lift the origin to match.
        let flipped = law
            .forces(g, Vec3::new(0.0, 0.0, RADIUS + HUB_Z - 0.03), glam::Quat::from_rotation_x(std::f32::consts::PI), Vec3::ZERO, Vec3::ZERO, &cmd, &flat, 1.0)
            .unwrap();
        println!("upright: all down {:?}, fz {:.1}; inverted: all down {:?}, fz {:.1}", up.touching[..4].iter().all(|&t| t), up.force.z, flipped.touching[..4].iter().all(|&t| t), flipped.force.z);
        assert!(up.force.z > 1000.0, "an upright jeep on its springs is pushed up: {}", up.force.z);
        assert!(flipped.touching[..4].iter().all(|&t| t), "the inverted hubs must be touching, or this proves nothing");
        assert!(flipped.force.z.abs() < 1.0, "an inverted hull gets no support from its wheels: {}", flipped.force.z);
    }

    /// **The tick counters, evaluated by hand from the decompilation.** The stationary damper
    /// (`sub_140E60A00`): nothing until the counter EXCEEDS `round(0.45 * 60)` = 27, then
    /// `(counter / 60 - 0.45) / 0.8`, capped at 1, and never above 1.75 wu/s. The standstill counter
    /// (`sub_140E60D40`): counts only with no throttle (or braking), a wheel gripping and under
    /// 1.75 wu/s. The reverse-against-travel brake (`sub_140E60B90` / `sub_140E60B00`): counts above
    /// 1 wu/s while the stick opposes the travel, brakes while `counter / 60 < 0.5` (29 ticks yes, 30
    /// no) or while the hold flag is set.
    #[test]
    fn the_tick_counters_follow_the_decompilation() {
        // (tick rate 60: the damper delay is round(0.45 * 60) = 27 ticks, the ramp is by seconds)
        assert_eq!(park_factor(27, 0.0), 0.0, "27 ticks is not MORE than round(0.45 * 60)");
        assert!((park_factor(28, 0.0) - 0.020833334).abs() < 1e-6, "(28/60 - 0.45) / 0.8");
        assert!((park_factor(48, 0.0) - 0.4375).abs() < 1e-6, "(48/60 - 0.45) / 0.8");
        assert_eq!(park_factor(120, 0.0), 1.0, "capped at 1");
        assert_eq!(park_factor(120, 3.0625), 0.0, "and never while it is still moving at 1.75 wu/s");
        assert!(park_factor(120, 3.0).abs() > 0.99);
        // standstill counter
        assert_eq!(idle_counter(5, 0.0, false, true, 1.0), 6, "no throttle, gripping, slow: counts");
        assert_eq!(idle_counter(5, 0.5, false, true, 1.0), 0, "throttle resets it");
        assert_eq!(idle_counter(5, 0.5, true, true, 1.0), 6, "braking counts even with the stick down");
        assert_eq!(idle_counter(5, 0.0, false, false, 1.0), 0, "no wheel gripping resets it");
        assert_eq!(idle_counter(5, 0.0, false, true, 4.0), 0, "too fast resets it");
        assert_eq!(idle_counter(0xFE, 0.0, false, true, 0.0), 0xFE, "saturates");
        // brake counter: brakes while counter / 60 < 0.5, i.e. 29 ticks yes, 30 no
        let mut c = 0u8;
        for _ in 0..29 {
            c = brake_counter(c, -1.0, 8.0, 0.9, false);
        }
        assert_eq!(c, 29);
        assert!(brake_active(c, false), "29 ticks = 0.483 s < 0.5 s: still braking");
        assert!(!brake_active(brake_counter(c, -1.0, 8.0, 0.9, false), false), "30 ticks = 0.5 s: released");
        assert!(brake_active(30, true), "the hold flag keeps it on");
        assert_eq!(brake_counter(14, -1.0, 8.0, -0.9, false), 0, "reversing WITH the travel is not a brake");
        assert_eq!(brake_counter(14, 1.0, 8.0, 0.9, false), 0, "driving forward is not a brake");
        assert_eq!(brake_counter(14, -1.0, 0.9, 0.9, false), 0, "and not below 1 wu/s");
        assert_eq!(brake_counter(14, 0.0, 8.0, 0.9, false), 0, "no stick, no brake");
        assert!(!brake_active(0, true), "a zero counter is never braking");
    }

    // -----------------------------------------------------------------------------------------
    // SLOPES
    // -----------------------------------------------------------------------------------------

    /// **A slope below the friction limit holds a jeep; one past it does not, and it slides at
    /// `g sin(theta)`.**
    ///
    /// The tag authors the limits: friction is full above `ground maximum slope 0` (cos 0.766,
    /// 40 deg) and zero below `ground maximum slope 1` (cos 0.643, 50 deg). At 30 deg a standing
    /// jeep needs `g sin 30 = 2.09` of the 4.4 wu/s^2 the tyres can give and must stay put; at 55
    /// deg the tyres give nothing, nothing but the spring acts along the slope, and the body
    /// accelerates down it at exactly `g sin 55 = 3.42`. Both expectations are derived from the
    /// tag's angles and gravity; neither is read back from the solver.
    #[test]
    fn a_slope_below_the_friction_limit_holds_and_one_past_it_slides_at_g_sin_theta() {
        let d = fixture();
        let g = crate::reach_physics::vehicle_fall_acceleration();
        // 30 degrees: held.
        let ground = incline(30.0);
        let start = DriveState { pos: Vec3::new(0.0, 0.0, 0.6 + 0.8 * 30f32.to_radians().tan()), ..Default::default() };
        let held = run(&d, start, DriveInput::default(), 3.0, 60.0, &ground, |_, _| {});
        let mut drift = 0.0f32;
        let end = run(&d, held, DriveInput::default(), 3.0, 60.0, &ground, |_, s| drift = drift.max((s.pos - held.pos).length()));
        println!("30 deg: moved {drift:.4} wu in 3 s after settling, pitch {:+.3} rad", end.pitch);
        assert!(end.touching[..4].iter().all(|&t| t), "all four wheels must be on the slope");
        assert!((end.pitch - 30f32.to_radians()).abs() < 0.03, "it must be lying ON the slope, pitch {:+.3}", end.pitch);
        // (Displacement, not tick-end speed: a body at rest reads +-g*dt/2 of kick-and-drift sawtooth.)
        assert!(drift < 0.05, "30 degrees is inside the friction limit: it crept {drift:.3} wu");
        // 55 degrees: past the second limit, sliding.
        let ground = incline(55.0);
        let start = DriveState { pos: Vec3::new(0.0, 0.0, 0.6 + 0.8 * 55f32.to_radians().tan()), ..Default::default() };
        let (mut v1, mut v2) = (None, None);
        run(&d, start, DriveInput::default(), 3.2, 60.0, &ground, |t, s| {
            let along = s.vel.dot(Vec3::new(55f32.to_radians().cos(), 0.0, 55f32.to_radians().sin()));
            if (t - 1.5).abs() < 0.009 {
                v1 = Some(along);
            }
            if (t - 2.5).abs() < 0.009 {
                v2 = Some(along);
            }
        });
        let (v1, v2) = (v1.unwrap(), v2.unwrap());
        let a = (v2 - v1) / 1.0;
        let want = -g * 55f32.to_radians().sin();
        println!("55 deg: along-slope speed {v1:.3} -> {v2:.3}, acceleration {a:+.3}, g sin(theta) {want:+.3}");
        assert!((a / want - 1.0).abs() < 0.10, "past the friction limit it must slide at g sin(theta) = {want:.3}, got {a:.3}");
    }
    // -----------------------------------------------------------------------------------------
    // THE REAL VEHICLES (need the cache)
    // -----------------------------------------------------------------------------------------

    use std::sync::OnceLock;

    /// The real Warthog and Mongoose, built once exactly as a ride builds them.
    fn real() -> Option<&'static (DriveModel, DriveModel)> {
        static CELL: OnceLock<Option<(DriveModel, DriveModel)>> = OnceLock::new();
        CELL.get_or_init(|| {
            let (_g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
            let build = |tail: &str| -> Option<DriveModel> {
                let tag = native
                    .tags_by_group(cache, "vehi")
                    .into_iter()
                    .find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(tail)))?;
                let mut index = crate::play_vehicle::VehicleIndex::default();
                index.entry(&native, cache, tag)?.drive.clone().ok()
            };
            Some((build("warthog\\warthog")?, build("mongoose\\mongoose")?))
        })
        .as_ref()
    }

    fn settle_flat(d: &DriveModel) -> DriveState {
        let r = d.wheel_law.as_ref().unwrap().wheels[0].radius;
        run(d, DriveState { pos: Vec3::new(0.0, 0.0, r + 0.02), ..Default::default() }, DriveInput::default(), 3.0, 60.0, &flat, |_, _| {})
    }

    /// The best average acceleration over any half second of a standing start (straight, full
    /// throttle, first three seconds).
    fn best_launch_acceleration(d: &DriveModel) -> f32 {
        let mut vx = vec![(0.0f32, 0.0f32)];
        run(d, settle_flat(d), DriveInput { throttle: 1.0, ..Default::default() }, 3.0, 60.0, &flat, |t, s| vx.push((t, s.vel.x)));
        vx.iter()
            .filter(|(t, _)| *t >= 0.5)
            .map(|&(t, v)| {
                let (_, v0) = *vx.iter().min_by(|a, b| (a.0 - (t - 0.5)).abs().partial_cmp(&(b.0 - (t - 0.5)).abs()).unwrap()).unwrap();
                (v - v0) / 0.5
            })
            .fold(0.0f32, f32::max)
    }

    /// **Full wheelspin gives exactly the Coulomb force of the tag.** All four wheels commanded to
    /// a surface speed far past every `moving friction velocity diff`, the hull at rest at its
    /// parking height: the summed forward force per unit mass is `0.033 * sqrt(0.9 * moving
    /// friction) * sum(share) * g * compression / depth`, computed here from the tag's numbers and
    /// the pose (every wheel is `radius - hub height` compressed). It is the engine's
    /// `friction * normal force` with the constants quoted — and tripling the surface speed must
    /// change nothing, because the Coulomb regime does not depend on the slip.
    #[test]
    fn full_wheelspin_gives_the_coulomb_force_of_the_tag_and_no_more() {
        let law = fixture_law();
        let body = law.body.unwrap();
        let g = crate::reach_physics::vehicle_fall_acceleration();
        let pos = Vec3::new(0.0, 0.0, rest_z());
        let go = |surface: f32| {
            let cmd = WheelCommand { steer_angle: 0.0, surface_speed: surface, brake_lock: false, handbrake: false, ai_driven: false };
            law.forces(g, pos, glam::Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &cmd, &flat, 1.0).unwrap().force.x / body.mass
        };
        let compression = RADIUS - (pos.z + HUB_Z);
        let share_sum: f32 = law.wheels.iter().map(|w| w.share).sum();
        let want = 0.033_f32
            * (0.9_f32 * law.ground.moving_friction).sqrt()
            * share_sum
            * g
            * compression
            / law.ground.depth;
        let (a, b) = (go(30.0), go(90.0));
        // THE ROOT: the same wheelspin on half the moving friction.
        let mut slippery = law.clone();
        slippery.ground.moving_friction *= 0.5;
        let cmd = WheelCommand { steer_angle: 0.0, surface_speed: 30.0, brake_lock: false, handbrake: false, ai_driven: false };
        let half = slippery.forces(g, pos, glam::Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO, &cmd, &flat, 1.0).unwrap().force.x / body.mass;
        println!("wheelspin: {a:.4} wu/s^2 (tag says {want:.4}), at triple the surface speed {b:.4}; on half the friction x{:.4}", half / a);
        assert!((half / a - 0.5f32.sqrt()).abs() < 1e-3, "traction goes as the root of the friction: x{:.4}", half / a);
        assert!((a / want - 1.0).abs() < 0.01, "full wheelspin accelerates at {a:.4}, the tag's friction gives {want:.4}");
        assert!((b / a - 1.0).abs() < 1e-4, "the Coulomb regime must not depend on the slip: {a} vs {b}");

        // THE ENGINE'S LOAD (`sub_140E60130`): the longitudinal tyre force on the driven wheels, in
        // force units. Wheelspin from rest PULLS the flywheel down (the ground resists the driving
        // wheels): minus the whole force. A hull coasting faster than the locked wheels PUSHES the
        // flywheel up: plus the whole force. Same magnitude both ways, opposite sign.
        let drive = |surface: f32, vx: f32| {
            let cmd = WheelCommand { steer_angle: 0.0, surface_speed: surface, brake_lock: false, handbrake: false, ai_driven: false };
            law.forces(g, pos, glam::Quat::IDENTITY, Vec3::new(vx, 0.0, 0.0), Vec3::ZERO, &cmd, &flat, 1.0).unwrap().ground_torque
        };
        let (spin, overrun) = (drive(30.0, 0.0), drive(0.0, 30.0));
        println!("engine load: wheelspin {spin:.1}, overrun {overrun:.1} (mass * wheelspin accel = {:.1})", body.mass * want);
        assert!((spin / (-body.mass * want) - 1.0).abs() < 0.01, "wheelspin must load the engine by the whole tyre force: {spin}");
        assert!((overrun / (body.mass * want) - 1.0).abs() < 0.01, "overrun must drive the engine by the whole tyre force: {overrun}");
    }

    /// **A standing start is traction-limited.** Halving BOTH friction coefficients cuts every tyre
    /// force by at most `sqrt(1/2)` at a given slip (the pure-function gate above shows the root
    /// exactly), and a launch the engine could out-pull the tyres on would not notice at all. The
    /// launch slows by between that factor (the slip can only grow to compensate, never shrink)
    /// and 0.97 (an engine-limited launch would read 1.0).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_standing_start_is_traction_limited() {
        let Some((wh, mg)) = real() else { return };
        for (name, d) in [("Warthog", wh), ("Mongoose", mg)] {
            let mut slippery = d.clone();
            let law = slippery.wheel_law.as_mut().unwrap();
            law.ground.friction *= 0.5;
            law.ground.moving_friction *= 0.5;
            let (a, b) = (best_launch_acceleration(d), best_launch_acceleration(&slippery));
            println!("{name}: launch {a:.3} wu/s^2, on half the friction {b:.3} (x{:.4})", b / a);
            assert!(a > 3.0, "{name} must actually launch: {a:.3}");
            assert!(b / a > 0.5f32.sqrt() - 0.02 && b / a < 0.97, "{name}: halving the friction scaled the launch by {:.4}", b / a);
        }
    }

    /// **The top speed is the drivetrain's ceiling** — `max angular velocity * the top gear's
    /// ratio` from the tag (Warthog 9.5 * 1.2 = 11.40, Mongoose 8.0 * 1.1 = 8.80) — reached and
    /// not passed, in top gear.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_top_speed_is_the_drivetrain_ceiling_and_it_is_reached() {
        let Some((wh, mg)) = real() else { return };
        for (name, d, want) in [("Warthog", wh, 11.4f32), ("Mongoose", mg, 8.8)] {
            let ceiling = d.drivetrain.as_ref().unwrap().speed_ceiling();
            assert!((ceiling - want).abs() < 1e-3, "the tag's ceiling for the {name} is {want}, the model says {ceiling}");
            let mut top = 0.0f32;
            let mut last_gear = 0;
            let end = run(d, settle_flat(d), DriveInput { throttle: 1.0, ..Default::default() }, 30.0, 60.0, &flat, |_, s| {
                top = top.max(s.vel.length());
                last_gear = s.engine.gear;
            });
            println!("{name}: ceiling {ceiling:.3}, final {:.3}, max {top:.3}, gear {last_gear}", end.vel.length());
            assert!((end.vel.length() / ceiling - 1.0).abs() < 0.005, "{name} ends at {:.3}, not at {ceiling:.3}", end.vel.length());
            assert!(top <= ceiling * 1.02, "{name} passed its ceiling: {top:.3}");
            assert_eq!(last_gear as usize, d.drivetrain.as_ref().unwrap().gears.len() - 1, "{name} must be in its top gear");
        }
    }

    /// **Reverse against travel brakes, then reverses.** Cruising at the ceiling and pushing the
    /// stick back: for the first half second every wheel is locked at the tyres' Coulomb limit
    /// (`sub_140E60B00`), so the deceleration is the traction cap; afterwards the engine selects
    /// reverse and the car backs away to `gear-0 ratio * max angular velocity` (-0.5 * 9.5).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn reverse_against_travel_brakes_at_the_tyre_limit_and_then_reverses() {
        let Some((wh, _)) = real() else { return };
        let d = wh;
        let cruise = run(d, settle_flat(d), DriveInput { throttle: 1.0, ..Default::default() }, 12.0, 60.0, &flat, |_, _| {});
        let law = d.wheel_law.as_ref().unwrap();
        let g = crate::reach_physics::vehicle_fall_acceleration();
        // Every wheel locked and sliding at far more than any velocity diff: the Coulomb force of the
        // static load. Weight transfer in a stop moves it by a few per cent, hence the tolerance.
        let cap = 0.033_f32 * (0.9_f32 * law.ground.moving_friction).sqrt() * law.wheels.iter().map(|w| w.share).sum::<f32>() * g / law.wheels.len() as f32;
        let (mut v_a, mut v_b) = (0.0, 0.0);
        let mut min_vx = 0.0f32;
        let mut gear0_at = None;
        let end = run(d, cruise, DriveInput { throttle: -1.0, ..Default::default() }, 9.0, 60.0, &flat, |t, s| {
            if (t - 0.10).abs() < 0.009 {
                v_a = s.vel.x;
            }
            if (t - 0.45).abs() < 0.009 {
                v_b = s.vel.x;
            }
            min_vx = min_vx.min(s.vel.x);
            if s.engine.gear == 0 && gear0_at.is_none() {
                gear0_at = Some(t);
            }
        });
        let decel = (v_a - v_b) / 0.35;
        let reverse_top = -0.5 * 9.5;
        println!("braking {decel:.3} wu/s^2 (tyre cap {cap:.3}), reverse engaged at {gear0_at:?}, backs away to {:.3} (want {reverse_top})", end.vel.x);
        assert!(cruise.vel.x > 11.0, "it must be at speed to begin with: {}", cruise.vel.x);
        assert!(decel <= cap * 1.12 && decel >= cap * 0.90, "braking {decel:.3} against the tyre limit {cap:.3}");
        assert!(gear0_at.is_some_and(|t| t >= 0.5), "the engine must hold off reverse for the brake's half second: {gear0_at:?}");
        assert!((end.vel.x / reverse_top - 1.0).abs() < 0.02, "backs away to {:.3}, not {reverse_top}", end.vel.x);
        assert!(min_vx > reverse_top * 1.02, "it must not overshoot the reverse ceiling: {min_vx}");
    }

    /// **A hard landing is caught by the springs and the hull never goes through the ground.**
    /// Dropped from 4 and 9 wu the springs bottom out (compression reaches the wheel radius, which
    /// is what makes this a real landing and not a gentle one), no wheel hub ends up below the
    /// surface, and the jeep comes to rest at the height it parks at.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_hard_landing_bottoms_the_springs_but_never_goes_through_the_ground() {
        let Some((wh, _)) = real() else { return };
        let d = wh;
        let law = d.wheel_law.as_ref().unwrap();
        let rest = settle_flat(d);
        for drop in [4.0f32, 9.0] {
            let mut lowest_hub = f32::MAX;
            let mut peak = 0.0f32;
            let end = run(d, DriveState { pos: Vec3::new(0.0, 0.0, rest.pos.z + drop), ..Default::default() }, DriveInput::default(), 8.0, 60.0, &flat, |_, s| {
                let q = s.basis();
                for w in &law.wheels {
                    lowest_hub = lowest_hub.min((s.pos + q * w.local).z);
                }
                peak = peak.max(s.suspension[..4].iter().cloned().fold(0.0, f32::max));
            });
            println!("drop {drop}: lowest hub {lowest_hub:+.4}, peak compression {peak:.3}, rests at {:.4} (parks at {:.4})", end.pos.z, rest.pos.z);
            assert!(peak >= law.wheels[0].radius * 0.99, "the {drop} wu drop must bottom the springs: peak {peak:.3}");
            assert!(lowest_hub > -0.01, "a wheel hub went through the ground: {lowest_hub:+.4}");
            assert!((end.pos.z - rest.pos.z).abs() < 3e-3, "it must settle back to {:.4}, got {:.4}", rest.pos.z, end.pos.z);
            assert!(end.roll.abs() < 0.01 && end.pitch.abs() < 0.02, "and settle level");
        }
    }

    /// **The Mongoose is a jeep too, and steers the right way.** A facing held 0.05 saturation
    /// angles to the left of the hull (about a degree of lock) turns it left, to the right turns it
    /// right, at top speed, and it stays upright. (Gentle: its centre of mass sits 72% of the way
    /// to the front axle, so past ~4 degrees of lock at 8.8 wu/s the engine's own tyre law spins
    /// it out.)
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_mongoose_drives_and_steers_the_right_way() {
        let Some((_, mg)) = real() else { return };
        assert!(mg.rigid(), "the Mongoose must be on the rigid-body law");
        let sat = mg.wheel_law.as_ref().unwrap().steer.saturation_angle();
        let cruise = run(mg, settle_flat(mg), DriveInput { throttle: 1.0, ..Default::default() }, 6.0, 60.0, &flat, |_, _| {});
        for (err, sign) in [(0.05 * sat, 1.0f32), (-0.05 * sat, -1.0)] {
            let (mut max_roll, mut turned, mut last) = (0.0f32, 0.0f32, cruise.yaw);
            let mut st = cruise;
            for _ in 0..180 {
                st = mg.step(st, DriveInput { throttle: 1.0, want_yaw: st.yaw + err, ..Default::default() }, 1.0 / 60.0, &flat);
                max_roll = max_roll.max(st.roll.abs());
                turned += crate::reach_vehicle::wrap_angle(st.yaw - last);
                last = st.yaw;
            }
            println!("Mongoose facing error {err:+.4}: yaw change {turned:+.3} rad, steer {:+.3}, max roll {max_roll:.3}", st.steer_angle);
            assert!(turned * sign > 0.2, "a facing {err:+.3} off the hull must turn it that way: {turned:+.3}");
            assert!(max_roll < 0.15, "and not roll it over: {max_roll:.3}");
        }
    }

    // -----------------------------------------------------------------------------------------
    // THE TYRE (pure: no map, no cache)
    // -----------------------------------------------------------------------------------------

    /// The rear axle only, every `moving friction velocity diff` set to 3, at the pose where the
    /// hubs rest on flat ground. Returns the total sideways force when the hull slides sideways at
    /// `v` with the wheels held still.
    fn rear_axle_sideways_force(handbrake: bool, v: f32) -> f32 {
        let mut law = fixture_law();
        law.wheels.retain(|w| w.steer != SteerRole::Steered);
        for w in &mut law.wheels {
            w.moving_velocity_diff = 3.0;
        }
        let g = crate::reach_physics::vehicle_fall_acceleration();
        let cmd = WheelCommand { steer_angle: 0.0, surface_speed: 0.0, brake_lock: false, handbrake, ai_driven: false };
        let wf = law
            .forces(g, Vec3::new(0.0, 0.0, rest_z()), glam::Quat::IDENTITY, Vec3::new(0.0, v, 0.0), Vec3::ZERO, &cmd, &flat, 1.0)
            .unwrap();
        wf.force.y
    }

    /// **The tyre has two regimes and the tag draws the line between them.** Below `moving friction
    /// velocity diff` the force is VISCOUS (linear in the slip: double the slip, double the force);
    /// above it it is COULOMB (independent of the slip). And the hand brake swaps the rear wheels
    /// onto their own `e-brake moving friction` (8.25 against 27.5), so their sideways grip falls by
    /// `sqrt(8.25 / 27.5)`. Relationships on the production `forces`, with the load identical in
    /// every call (the hull is not moving vertically).
    #[test]
    fn the_tyre_is_viscous_below_the_velocity_diff_coulomb_above_and_the_handbrake_loosens_the_rear() {
        let (f1, f2) = (rear_axle_sideways_force(false, 1.0), rear_axle_sideways_force(false, 2.0));
        let (f4, f8) = (rear_axle_sideways_force(false, 4.0), rear_axle_sideways_force(false, 8.0));
        println!("viscous: {f1:.1} -> {f2:.1} (x{:.3}); Coulomb: {f4:.1} -> {f8:.1} (x{:.3})", f2 / f1, f8 / f4);
        assert!(f1 < 0.0 && f4 < 0.0, "the friction must OPPOSE the slide");
        assert!((f2 / f1 - 2.0).abs() < 1e-3, "below the velocity diff the force doubles with the slip: x{:.4}", f2 / f1);
        assert!((f8 / f4 - 1.0).abs() < 1e-3, "above it the force does not change with the slip: x{:.4}", f8 / f4);
        let hb = rear_axle_sideways_force(true, 8.0);
        let want = (8.25f32 / 27.5).sqrt();
        println!("hand brake: {hb:.1} against {f8:.1}: x{:.4} (sqrt(8.25/27.5) = {want:.4})", hb / f8);
        assert!((hb / f8 - want).abs() < 1e-3, "the hand brake must loosen the rear by sqrt(mu_ebrake/mu): x{:.4} vs {want:.4}", hb / f8);
    }
    /// **A jeep that has rolled over stays rolled over; one that has only leant rights itself.**
    /// The engine has no self-righting in the jeep tick (`sub_140E69D40` applies wheel forces and
    /// nothing else), so tipped 60 degrees the springs (which push along the ground normal at the
    /// hubs, below the centre of mass) bring it level, and tipped 170 degrees nothing turns it
    /// back. The hull's own Havok collision is NOT modelled: a rolled jeep rests on its wheel hubs
    /// (the ground backstop) rather than on its roof -- reported, not hidden.
    ///
    /// (The support cut-off is gated separately, below.)
    #[test]
    fn a_jeep_that_leans_rights_itself_and_one_that_rolls_over_stays_over() {
        let d = fixture();
        let rolled = |deg: f32| {
            let mut st = at(1.0);
            st.orient = Some(glam::Quat::from_rotation_x(deg.to_radians()));
            let end = run(&d, st, DriveInput::default(), 8.0, 60.0, &flat, |_, _| {});
            (end.basis() * Vec3::Z).z
        };
        let (leant, flipped) = (rolled(60.0), rolled(170.0));
        println!("tipped 60 deg -> up.z {leant:+.3} after 8 s; tipped 170 deg -> up.z {flipped:+.3}");
        assert!(leant > 0.99, "a jeep leant 60 degrees must come back level: up.z {leant}");
        assert!(flipped < -0.9, "a rolled-over jeep has nothing to right it: up.z {flipped}");
    }

    /// Least-squares quadratic `y = c0 + c1 t + c2 t^2` over `(t, y)`; returns `(c0, c1, c2, rms)`.
    fn quad_fit(pts: &[(f64, f64)]) -> Option<(f64, f64, f64, f64)> {
        let n = pts.len() as f64;
        if pts.len() < 4 { return None; }
        let (mut s1, mut s2, mut s3, mut s4, mut y0, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for &(t, y) in pts {
            let t2 = t * t;
            s1 += t; s2 += t2; s3 += t2 * t; s4 += t2 * t2;
            y0 += y; y1 += y * t; y2 += y * t2;
        }
        // normal equations 3x3 (Cramer)
        let m = [[n, s1, s2], [s1, s2, s3], [s2, s3, s4]];
        let det = |a: [[f64; 3]; 3]| a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        let d = det(m);
        if d.abs() < 1e-12 { return None; }
        let rhs = [y0, y1, y2];
        let mut c = [0.0; 3];
        for k in 0..3 {
            let mut mk = m;
            for r in 0..3 { mk[r][k] = rhs[r]; }
            c[k] = det(mk) / d;
        }
        let rms = (pts.iter().map(|&(t, y)| (y - (c[0] + c[1] * t + c[2] * t * t)).powi(2)).sum::<f64>() / n).sqrt();
        Some((c[0], c[1], c[2], rms))
    }

    /// One free-flight arc found in a recording: `(film file name, net id, def_wire, a_z, a_horizontal, T, rms_z, vz at start)`.
    #[derive(Clone, Debug)]
    struct Arc {
        film: String,
        net: u32,
        def: Option<u32>,
        az: f64,
        ah: f64,
        t: f64,
        rms: f64,
        vz0: f64,
        z: Vec<(f64, f64)>,
    }

    /// Find ballistic windows in vehicle position tracks. A window is a run of CONSECUTIVE recorded
    /// samples (no gap over `max_gap` ticks) over which a quadratic fits z within `rms_z` AND x, y are
    /// straight lines (speed change under `dh` wu/s over the window, residual under `rms_h`) AND the
    /// vehicle is moving horizontally. **No rule looks at the fitted vertical acceleration.** `min_t`
    /// is the window length; `min_dv` demands the fitted vz to change by at least that much over the
    /// window so the curvature is measurable at all (a selection on how much the arc bends, not on
    /// the value it bends to).
    /// The arcs in ONE position track (`(tick, position)` samples), by the rules of [`film_arcs`].
    /// Returns `(a_z, a_horizontal, T, rms_z, vz at start, samples)` per arc.
    #[allow(clippy::type_complexity)]
    fn track_arcs(pts: &[(u32, [f32; 3])], min_t: f64, rms_z: f64, dh: f64, rms_h: f64, min_dv: f64, min_h: f64) -> (usize, Vec<(f64, f64, f64, f64, f64, Vec<(f64, f64)>)>) {
        let mut rejected = 0usize;
        let mut arcs = vec![];
        let mut i = 0usize;
        while i + 8 < pts.len() {
            let mut best: Option<(usize, f64, f64, f64, f64, f64)> = None;
            let mut j = i + 7;
            while j < pts.len() {
                if pts[j].0 - pts[j - 1].0 > 6 { break; }
                let t0 = pts[i].0 as f64 / 60.0;
                let col = |c: usize| -> Vec<(f64, f64)> { pts[i..=j].iter().map(|&(k, q)| (k as f64 / 60.0 - t0, q[c] as f64)).collect() };
                let (zs, xs, ys) = (col(2), col(0), col(1));
                let (Some(fz), Some(fx), Some(fy)) = (quad_fit(&zs), quad_fit(&xs), quad_fit(&ys)) else { j += 1; continue };
                let tt = zs.last().unwrap().0;
                let hspeed = (fx.1.powi(2) + fy.1.powi(2)).sqrt();
                let h_ok = hspeed >= min_h && (2.0 * fx.2 * tt).abs() < dh && (2.0 * fy.2 * tt).abs() < dh && fx.3 < rms_h && fy.3 < rms_h;
                let ok = h_ok && fz.3 < rms_z;
                if ok && tt >= min_t {
                    let ah = (4.0 * (fx.2 * fx.2 + fy.2 * fy.2)).sqrt();
                    best = Some((j, 2.0 * fz.2, ah, tt, fz.3, fz.1));
                } else if !ok && tt >= min_t {
                    break;
                }
                j += 1;
            }
            match best {
                Some((end, az, ah, tt, rms, vz0)) if (az * tt).abs() >= min_dv => {
                    let z: Vec<(f64, f64)> = pts[i..=end].iter().map(|&(k, q)| (k as f64 / 60.0 - pts[i].0 as f64 / 60.0, q[2] as f64)).collect();
                    arcs.push((az, ah, tt, rms, vz0, z));
                    i = end + 1;
                }
                Some((end, ..)) => { rejected += 1; i = end + 1; }
                None => i += 3,
            }
        }
        (rejected, arcs)
    }

    fn film_arcs(min_t: f64, rms_z: f64, dh: f64, rms_h: f64, min_dv: f64, min_h: f64) -> (usize, usize, Vec<Arc>) {
        use crate::{film, film_decode};
        let films = film::find_finished_films();
        let (mut tracks, mut rejected) = (0usize, 0usize);
        let mut out = vec![];
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            let name = p.file_name().map(|n| n.to_string_lossy().chars().take(20).collect::<String>()).unwrap_or_default();
            for t in idx.tracks.iter().filter(|t| t.entity_type == 15 && t.positions.len() > 30) {
                tracks += 1;
                let (rej, arcs) = track_arcs(&t.positions, min_t, rms_z, dh, rms_h, min_dv, min_h);
                rejected += rej;
                for (az, ah, tt, rms, vz0, z) in arcs {
                    out.push(Arc { film: name.clone(), net: t.net_id, def: t.def_wire, az, ah, t: tt, rms, vz0, z });
                }
            }
        }
        (tracks, rejected, out)
    }

    /// Least-squares polynomial of `deg` over `(t, y)` (Gaussian elimination on the normal equations).
    fn poly_fit(pts: &[(f64, f64)], deg: usize) -> Option<Vec<f64>> {
        let m = deg + 1;
        let mut a = vec![vec![0.0f64; m + 1]; m];
        for &(t, y) in pts {
            let mut pw = vec![1.0; 2 * m];
            for k in 1..2 * m { pw[k] = pw[k - 1] * t; }
            for r in 0..m {
                for c in 0..m { a[r][c] += pw[r + c]; }
                a[r][m] += y * pw[r];
            }
        }
        for col in 0..m {
            let piv = (col..m).max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())?;
            if a[piv][col].abs() < 1e-12 { return None; }
            a.swap(col, piv);
            for r in 0..m {
                if r != col {
                    let f = a[r][col] / a[col][col];
                    for c in col..=m { a[r][c] -= f * a[col][c]; }
                }
            }
        }
        Some((0..m).map(|r| a[r][m] / a[r][r]).collect())
    }

    fn summarize(label: &str, mut v: Vec<f64>) {
        if v.is_empty() { println!("{label}: n 0"); return; }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        let pos = v.iter().filter(|&&x| x > 0.0).count();
        println!("{label}: n {n}, p10 {:.2} p25 {:.2} median {:.3} p75 {:.2} p90 {:.2}, mean {mean:.3}, a>0 (impossible in flight) {pos}", v[n / 10], v[n / 4], v[n / 2], v[n * 3 / 4], v[n * 9 / 10]);
    }

    #[test]
    #[ignore = "print-only; reads the saved films on this machine"]
    fn probe_film_free_flight_arcs() {
        for (label, min_t, rms_z, dh, rms_h, min_dv, min_h) in [
            ("strict", 0.6, 0.02, 0.4, 0.03, 1.5, 2.0),
            ("long arcs only", 1.0, 0.03, 0.6, 0.04, 3.0, 2.0),
            ("BIG jumps, any horizontal speed", 0.8, 0.04, 1.0, 0.06, 4.0, 0.0),
            ("BIGGER jumps", 1.0, 0.05, 1.5, 0.08, 6.0, 0.0),
        ] {
            let (tracks, rejected, arcs) = film_arcs(min_t, rms_z, dh, rms_h, min_dv, min_h);
            println!("-- {label}: {tracks} vehicle tracks, {} arcs, {rejected} too-flat windows skipped", arcs.len());
            summarize("   all vehicles az", arcs.iter().map(|a| a.az).collect());
            {
                let mut v: Vec<&Arc> = arcs.iter().collect();
                v.sort_by(|a, b| b.t.partial_cmp(&a.t).unwrap());
                let wsum: f64 = v.iter().map(|a| a.t.powi(3)).sum();
                let wmean = v.iter().map(|a| a.az * a.t.powi(3)).sum::<f64>() / wsum.max(1e-9);
                println!("   T^3-weighted mean az {wmean:.3}; longest arcs (T, az, rms, def, film, net):");
                for a in v.iter().take(8) { println!("      T {:.2} az {:.2} rms {:.3} vz0 {:.1} def {:?} {} {}", a.t, a.az, a.rms, a.vz0, a.def, a.film, a.net); }
            }
            println!("   vz at window start: {:?}", arcs.iter().map(|a| (a.vz0 * 10.0).round() / 10.0).collect::<Vec<_>>());
            summarize("   horizontal accel", arcs.iter().map(|a| a.ah).collect());
            let names = def_leafs(&arcs.iter().filter_map(|a| a.def).collect::<Vec<_>>());
            let mut defs: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
            for a in &arcs { defs.entry(a.def.and_then(|d| names.get(&d).cloned()).unwrap_or_else(|| format!("{:?}", a.def))).or_default().push(a.az); }
            for (d, v) in defs { if v.len() >= 3 { summarize(&format!("   {d}"), v); } }
            for a in arcs.iter().filter(|a| a.def.and_then(|d| names.get(&d)).is_some_and(|n| n == "warthog" || n == "mongoose")) {
                println!("      {} T {:.2} az {:+.2} vz0 {:+.1} rms {:.3} {}", a.def.and_then(|d| names.get(&d).cloned()).unwrap_or_default(), a.t, a.az, a.vz0, a.rms, a.film);
            }
            // Is there a velocity-dependent term? Cubic fit per arc: a(t) = 2 c2 + 6 c3 t, vz(t) = c1 + 2 c2 t + 3 c3 t^2.
            // Regress a_mid on vz_mid across arcs; linear drag a = -g - k vz gives slope -k.
            let (mut xs, mut ys) = (vec![], vec![]);
            for a in &arcs {
                if let Some(c) = poly_fit(&a.z, 3) {
                    let tm = a.t / 2.0;
                    xs.push(c[1] + 2.0 * c[2] * tm + 3.0 * c[3] * tm * tm);
                    ys.push(2.0 * c[2] + 6.0 * c[3] * tm);
                }
            }
            if let Some(c) = poly_fit(&xs.iter().cloned().zip(ys.iter().cloned()).collect::<Vec<_>>(), 1) {
                println!("   a_mid = {:.3} + {:.3} * vz_mid  over {} arcs (vz_mid range {:.1}..{:.1}); linear drag k would be {:.3}/s", c[0], c[1], xs.len(), xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max), -c[1]);
            }
        }
    }

    /// Landings in the recordings: the tick where a vehicle's vertical velocity goes from a fall (<= -4 wu/s)
    /// to nearly level within 6 ticks, on a track that then stays continuous for 1.2 s with the
    /// horizontal heading nearly straight. Returns `(film, net, fall speed, dz series every 0.05 s
    /// from landing minus the mean level of the last 0.3 s)`.
    /// Landings in one position track: see [`film_landings`]. `(index, fall speed, dz series)`.
    fn track_landings(pts: &[(u32, [f32; 3])]) -> Vec<(u32, f64, Vec<f64>)> {
        let mut out = vec![];
        if pts.len() < 120 { return out; }
        let v = |a: usize| ((pts[a + 1].1[2] - pts[a].1[2]) as f64) * 60.0 / (pts[a + 1].0 - pts[a].0).max(1) as f64;
        let mut w = 12usize;
        while w + 100 < pts.len() {
            let cont = (0..72).all(|k| pts[w + k + 1].0 - pts[w + k].0 <= 2);
            if cont && v(w - 4) < -4.0 && v(w + 4) > -1.5 {
                let fall = -v(w - 4);
                let at = |k: usize| pts.get(w + k).map(|q| q.1[2] as f64);
                let level = (60..72).filter_map(at).sum::<f64>() / 12.0;
                let series: Vec<f64> = (0..60).step_by(3).filter_map(|k| at(k).map(|z| z - level)).collect();
                let h0 = (pts[w].1[1] - pts[w - 3].1[1]).atan2(pts[w].1[0] - pts[w - 3].1[0]);
                let h1 = (pts[w + 60].1[1] - pts[w + 57].1[1]).atan2(pts[w + 60].1[0] - pts[w + 57].1[0]);
                let hs = ((pts[w + 60].1[0] - pts[w].1[0]).powi(2) + (pts[w + 60].1[1] - pts[w].1[1]).powi(2)).sqrt();
                if (h1 - h0).abs() < 0.8 && hs > 0.5 {
                    out.push((pts[w].0, fall, series));
                }
                w += 90;
            } else { w += 1; }
        }
        out
    }

    /// Landings in the recordings, per vehicle name. `(vehicle, film, net, fall speed, dz series every 0.05 s
    /// from the landing minus the mean level of the last 0.3 s)`.
    fn film_landings() -> Vec<(String, String, u32, f64, Vec<f64>)> {
        use crate::{film, film_decode};
        let mut out = vec![];
        let mut found: Vec<(u32, String, u32, f64, Vec<f64>)> = vec![];
        for p in &film::find_finished_films() {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            let name = p.file_name().map(|n| n.to_string_lossy().chars().take(14).collect::<String>()).unwrap_or_default();
            for t in idx.tracks.iter().filter(|t| t.entity_type == 15 && t.positions.len() > 200) {
                for (_, fall, series) in track_landings(&t.positions) {
                    found.push((t.def_wire.unwrap_or(0), name.clone(), t.net_id, fall, series));
                }
            }
        }
        let names = def_leafs(&found.iter().map(|f| f.0).collect::<Vec<_>>());
        for (d, f, n, fall, s) in found {
            out.push((names.get(&d).cloned().unwrap_or_else(|| format!("def{d}")), f, n, fall, s));
        }
        out
    }

    #[test]
    #[ignore = "print-only; reads the saved films"]
    fn probe_film_landings() {
        let l = film_landings();
        println!("{} clean landings", l.len());
        for (v, f, n, fall, s) in l.iter().filter(|l| l.0 == "warthog" || l.0 == "mongoose" || l.0 == "pickup" || l.0 == "scorpion") {
            println!("{v} film {f} net {n} fell {fall:.1}: {}", s.iter().map(|z| format!("{z:+.2}")).collect::<Vec<_>>().join(" "));
        }
    }

    /// `def_wire` -> tag leaf name for the vehicles of the `forge_halo` cache, the way Theater resolves
    /// a creation message's definition (`film_action::RefTable`).
    fn def_leafs(wires: &[u32]) -> std::collections::HashMap<u32, String> {
        let mut out = std::collections::HashMap::new();
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return out };
        let Some(c) = crate::mapcat::enumerate().into_iter().find(|c| {
            c.game == crate::mapcat::Game::Reach && c.path.file_stem().is_some_and(|s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
        }) else { return out };
        let Ok(bytes) = std::fs::read(&c.path) else { return out };
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in crate::film_action::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) { tags.push((t, (*g).to_string())); }
        }
        let Some(table) = crate::film_action::RefTable::for_map(&bytes, &tags) else { return out };
        for &w in wires {
            if let Some(tag) = table.resolve(w) {
                if let Some(n) = native.tag_name(cache, tag) { out.insert(w, n.rsplit('\\').next().unwrap_or(&n).to_string()); }
            }
        }
        out
    }

    /// THE METHOD ON THE MODEL: the same arc finder, run on a simulated Warthog's own 60 Hz position track
    /// over ramps of several sizes, whose true free-fall acceleration is exactly `-g`. What it recovers
    /// is the finder's bias; the recordings must be read against that, not against `g`.
    #[test]
    #[ignore = "opens the native library"]
    fn probe_arc_finder_on_the_model() {
        let Some((wh, mg)) = real() else { return };
        let g = crate::reach_physics::vehicle_fall_acceleration() as f64;
        for (name, d) in [("Warthog", wh), ("Mongoose", mg)] {
            let mut all = vec![];
            for (deg, speed_frames) in [(8.0f32, 8.0f32), (12.0, 10.0), (15.0, 8.0), (20.0, 9.0), (25.0, 10.0), (10.0, 6.0)] {
                let t = deg.to_radians().tan();
                let ramp = move |p: Vec3| -> Option<(f32, Vec3)> { if p.x < 30.0 { Some((0.0, Vec3::Z)) } else if p.x < 34.0 { Some(((p.x - 30.0) * t, Vec3::new(-t, 0.0, 1.0).normalize())) } else { Some((0.0, Vec3::Z)) } };
                let mut st = settle_flat(d);
                st.pos.x = 0.0;
                let mut pts: Vec<(u32, [f32; 3])> = vec![];
                for k in 0..1500u32 {
                    let thr = if st.vel.x > speed_frames { 0.0 } else { 1.0 };
                    st = d.step(st, DriveInput { throttle: thr, ..Default::default() }, 1.0 / 60.0, &ramp);
                    pts.push((k, [st.pos.x, st.pos.y, st.pos.z]));
                }
                let (_, arcs) = track_arcs(&pts, 0.6, 0.02, 0.4, 0.03, 1.5, 2.0);
                for a in &arcs { all.push(a.0); }
                println!("{name} ramp {deg} deg: {} arcs {:?}", arcs.len(), arcs.iter().map(|a| ((a.0 * 100.0).round() / 100.0, (a.2 * 100.0).round() / 100.0)).collect::<Vec<_>>());
            }
            summarize(&format!("{name} SIM (true -{g:.3})"), all);
        }
    }

    #[test]
    #[ignore = "print-only"]
    fn probe_takeoff() {
        let Some((wh0, mg0)) = real() else { return };
        for (name, d0, kdamp) in [("warthog", wh0, 1.0f32), ("warthog", wh0, 2.0), ("warthog", wh0, 3.0), ("mongoose", mg0, 1.0), ("mongoose", mg0, 2.0)] {
        let mut wh_owned = d0.clone();
        wh_owned.wheel_law.as_mut().unwrap().ground.damp *= kdamp;
        let wh = &wh_owned;
        println!("-- {name} damper x{kdamp}");
        for deg in [10.0f32, 15.0, 20.0, 30.0] {
            let t = deg.to_radians().tan();
            // a ramp that ends in a cliff (ground drops away): takeoff is clean
            let ramp = move |p: Vec3| -> Option<(f32, Vec3)> { if p.x < 30.0 { Some((0.0, Vec3::Z)) } else if p.x < 34.0 { Some(((p.x - 30.0) * t, Vec3::new(-t, 0.0, 1.0).normalize())) } else { None } };
            let mut st = settle_flat(wh);
            st.pos.x = 0.0;
            let mut last_touch: Option<(f32, DriveState)> = None;
            let mut first_air: Option<(f32, DriveState)> = None;
            let mut apex = f32::MIN;
            for k in 0..1500u32 {
                st = wh.step(st, DriveInput { throttle: 1.0, ..Default::default() }, 1.0 / 60.0, &ramp);
                let touching = st.touching[..4].iter().any(|&b| b);
                if touching { last_touch = Some((k as f32 / 60.0, st)); }
                else if last_touch.is_some() && first_air.is_none() { first_air = Some((k as f32 / 60.0, st)); }
                if first_air.is_some() { apex = apex.max(st.pos.z); }
                if first_air.is_some() && st.pos.z < -20.0 { break; }
            }
            let (tl, l) = last_touch.unwrap();
            let (ta, a) = first_air.unwrap();
            let hs = a.vel.truncate().length();
            let ballistic_vz = hs * deg.to_radians().tan();
            println!("ramp {deg}: last wheel on ramp t {tl:.2} x {:.2} v ({:.2},{:.2}); first airborne t {ta:.2} v ({:.2},{:.2}) hspeed {hs:.2}, pitch {:+.1} deg; v*tan(ramp) = {ballistic_vz:.2}; apex above takeoff z {:.2}", l.pos.x, l.vel.x, l.vel.z, a.vel.x, a.vel.z, a.pitch.to_degrees(), apex - a.pos.z);
        }
        }
    }

    // -----------------------------------------------------------------------------------------
    // END TO END: the real ride loop at real frame times
    // -----------------------------------------------------------------------------------------

    /// A real Warthog ridden through `PlayMode::tick` (the call the window makes) on the harness floor,
    /// thrown into the air at `(z0, vel)`, stepped at `fps`. Returns `(t, pos, vel, any wheel down)` per frame.
    fn ride_thrown(fps: f32, z0: f32, vel: Vec3, secs: f32) -> Option<Vec<(f32, Vec3, Vec3, bool)>> {
        use crate::play_mode::ride_harness::Ride;
        let (_g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
        let mut r = Ride::with_vehicle(native.clone(), cache, "warthog\\warthog")?;
        let tag = r.objects[0].primary_tag;
        r.force_seat(tag, 0).ok()?;
        {
            let rd = r.pm.riding.as_mut()?;
            rd.drive.pos = Vec3::new(0.0, 0.0, z0);
            rd.drive.vel = vel;
        }
        let input = r.input();
        let mut out = vec![];
        for i in 0..(secs * fps).round() as usize {
            r.tick_dt(&input, 1.0 / fps);
            let d = &r.pm.riding.as_ref()?.drive;
            out.push(((i + 1) as f32 / fps, d.pos, d.vel, d.touching[..4].iter().any(|&b| b)));
        }
        Some(out)
    }

    /// Height and velocity of a body in free flight after `t` seconds under Havok's step as the engine runs
    /// it: once per 1/60 s tick, gravity first, then the move (so the position at a time inside a tick
    /// is the tick's start plus the fraction of that tick's displacement). Independent arithmetic.
    fn free_flight(z0: f64, vz0: f64, g: f64, t: f64) -> (f64, f64) {
        let dt = 1.0 / 60.0;
        let n = (t / dt + 1e-9).floor();
        let f = t / dt - n;
        let (mut z, mut v) = (z0, vz0);
        for _ in 0..n as usize {
            v -= g * dt;
            z += v * dt;
        }
        let v_next = v - g * dt;
        (z + f * dt * v_next, v_next)
    }

    /// **THE END-TO-END FALL.** A real Warthog dropped from rest through the real ride loop at 30, 60 and 144
    /// fps must be at the same height as the engine's own step puts it -- Havok once per 1/60 s tick,
    /// gravity then move -- at 1 s and 2 s, and the three frame rates must agree. This measures what the
    /// player sees, not `jeep_tick`: the tick accumulator, the drift spread across frames and the
    /// production wrapper are all in it. A tick that ran at half rate, a gravity applied per frame, or a
    /// drift with a different `dt` than the velocity change would each move it by tens of per cent.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_ridden_warthog_falls_at_the_engines_gravity_at_every_frame_rate() {
        let g = crate::reach_physics::vehicle_fall_acceleration() as f64;
        let mut at2: Vec<f32> = vec![];
        for fps in [30.0f32, 60.0, 144.0] {
            let Some(run) = ride_thrown(fps, 40.0, Vec3::ZERO, 2.0) else { return };
            for t in [1.0f32, 2.0] {
                let (_, pos, vel, down) = *run.iter().min_by(|a, b| (a.0 - t).abs().partial_cmp(&(b.0 - t).abs()).unwrap()).unwrap();
                let (z, v) = free_flight(40.0, 0.0, g, t as f64);
                println!("{fps} fps t {t}: z {:.4} (engine step {z:.4}, continuous {:.4}), vz {:.4} (engine {v:.4})", pos.z, 40.0 - 0.5 * g * (t as f64).powi(2), vel.z);
                assert!(!down, "{fps} fps: a wheel is 'touching' 30 wu up -- the ground query is holding the Warthog in the air");
                assert!((pos.z as f64 - z).abs() < 0.01, "{fps} fps t={t}: height {:.4} against the engine's step {z:.4}", pos.z);
                // (the velocity is the one the LAST tick set, so it may be one tick ahead of or behind the
                // continuous reading at an exact tick boundary: g * dt = 0.0695)
                assert!((vel.z as f64 - v).abs() < g / 60.0 * 1.1, "{fps} fps t={t}: vz {:.4} against {v:.4}", vel.z);
                if t == 2.0 { at2.push(pos.z); }
            }
        }
        assert!(at2.iter().all(|z| (z - at2[1]).abs() < 0.01), "frame rates disagree: {at2:?}");
    }

    /// **THE END-TO-END LAUNCH.** Thrown up at 3 wu/s and forward at 5, through the real ride loop at three
    /// frame rates: the apex is the engine step's apex, the horizontal speed does not change, and the
    /// first wheel touches down when the engine's step says it reaches the contact tolerance.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_ridden_warthogs_jump_has_the_apex_and_airtime_the_gravity_gives() {
        let g = crate::reach_physics::vehicle_fall_acceleration() as f64;
        let dt = 1.0 / 60.0;
        let (mut v, mut z, mut apex) = (3.0f64, 3.0f64, 3.0f64);
        let rest_origin = 0.25f64 + 0.0328 - 0.19877; // origin height at which a wheel first reads as down
        let (mut k, mut landed) = (0u32, None);
        for _ in 0..600 {
            v -= g * dt;
            z += v * dt;
            apex = apex.max(z);
            k += 1;
            if z <= rest_origin && landed.is_none() { landed = Some(k as f64 * dt); }
        }
        let want_t = landed.unwrap();
        for fps in [30.0f32, 60.0, 144.0] {
            let Some(run) = ride_thrown(fps, 3.0, Vec3::new(5.0, 0.0, 3.0), 3.0) else { return };
            let mut max_z = f32::MIN;
            let mut down_at = None;
            for &(t, pos, vel, down) in &run {
                if down_at.is_none() {
                    max_z = max_z.max(pos.z);
                    if down { down_at = Some(t); }
                    else { assert!((vel.x - 5.0).abs() < 0.02, "{fps} fps: forward speed changed in the air: {}", vel.x); }
                }
            }
            let t = down_at.expect("it must come down");
            println!("{fps} fps: apex {:.4} (engine step {apex:.4}), first wheel down {t:.3} s (engine {want_t:.3})", max_z);
            assert!((max_z as f64 - apex).abs() < 0.02, "{fps} fps: apex {max_z:.4} against {apex:.4}");
            assert!((t as f64 - want_t).abs() < 0.07, "{fps} fps: airtime {t:.3} against {want_t:.3}");
        }
    }

    // -----------------------------------------------------------------------------------------
    // THE RECORDINGS
    // -----------------------------------------------------------------------------------------

    /// **THE RECORDED WARTHOGS FALL AS HMS DOES.** Free-flight arcs found in the saved films (vertical
    /// quadratic, horizontal straight, window of 0.6 s or more; no rule looks at the fitted value) and the
    /// SAME finder run on a simulated Warthog's own 60 Hz track over ramps whose true fall is `g`.
    /// The finder reads low on both (springs and contact at the edges of its windows: it recovers about
    /// -3.75 for a true -4.171 on the model), so the recording is compared with the model through the
    /// finder, not with `g`. They must agree, and the count of arcs must be real.
    #[test]
    #[ignore = "opens the native library and reads the saved films on this machine"]
    fn the_recorded_warthogs_fall_through_the_same_finder_as_the_model_does() {
        let Some((wh, _)) = real() else { return };
        if crate::film::find_finished_films().is_empty() {
            eprintln!("SKIPPED: no saved films on this machine");
            return;
        }
        let model_median = |d: &DriveModel| -> (usize, f64) {
            let mut all = vec![];
            for (deg, v) in [(8.0f32, 8.0f32), (12.0, 10.0), (15.0, 8.0), (20.0, 9.0), (25.0, 10.0), (10.0, 6.0)] {
                let t = deg.to_radians().tan();
                let ramp = move |p: Vec3| -> Option<(f32, Vec3)> {
                    if p.x < 30.0 { Some((0.0, Vec3::Z)) } else if p.x < 34.0 { Some(((p.x - 30.0) * t, Vec3::new(-t, 0.0, 1.0).normalize())) } else { Some((0.0, Vec3::Z)) }
                };
                let mut st = settle_flat(d);
                st.pos.x = 0.0;
                let mut pts: Vec<(u32, [f32; 3])> = vec![];
                for k in 0..1500u32 {
                    let thr = if st.vel.x > v { 0.0 } else { 1.0 };
                    st = d.step(st, DriveInput { throttle: thr, ..Default::default() }, 1.0 / 60.0, &ramp);
                    pts.push((k, [st.pos.x, st.pos.y, st.pos.z]));
                }
                all.extend(track_arcs(&pts, 0.6, 0.02, 0.4, 0.03, 1.5, 2.0).1.into_iter().map(|a| a.0));
            }
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (all.len(), all[all.len() / 2])
        };
        let (n_model, med_model) = model_median(wh);
        let (_, _, arcs) = film_arcs(0.6, 0.02, 0.4, 0.03, 1.5, 2.0);
        let names = def_leafs(&arcs.iter().filter_map(|a| a.def).collect::<Vec<_>>());
        let mut hog: Vec<f64> = arcs.iter().filter(|a| a.def.and_then(|d| names.get(&d)).is_some_and(|n| n == "warthog")).map(|a| a.az).collect();
        hog.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_film = hog[hog.len() / 2];
        let ghost: Vec<f64> = arcs.iter().filter(|a| a.def.and_then(|d| names.get(&d)).is_some_and(|n| n == "ghost")).map(|a| a.az).collect();
        println!("Warthog: {} recorded arcs, median {med_film:.3} (p25 {:.2} p75 {:.2}); the model through the same finder: {n_model} arcs, median {med_model:.3}; true g -{:.3}; ghosts {} arcs",
            hog.len(), hog[hog.len() / 4], hog[hog.len() * 3 / 4], crate::reach_physics::vehicle_fall_acceleration(), ghost.len());
        assert!(hog.len() >= 8 && n_model >= 6, "too few arcs to say anything: {} recorded, {n_model} simulated", hog.len());
        assert!((med_film - med_model).abs() < 0.45, "recorded Warthogs fall at {med_film:.3}, the model at {med_model:.3} through the same finder");
    }

    /// Power spectrum of the high-passed vertical motion of recorded vehicles that are driving (speed above
    /// 3 wu/s, continuous 6 s segments), averaged over segments, in 0.25 Hz bins. Print-only: where does a
    /// Warthog's body bounce on its springs?
    #[test]
    #[ignore = "print-only; reads the saved films"]
    fn probe_film_vertical_spectrum() {
        use crate::{film, film_decode};
        let mut acc: std::collections::BTreeMap<String, (Vec<f64>, usize)> = Default::default();
        let films = film::find_finished_films();
        let mut all: Vec<(u32, String, Vec<(u32, [f32; 3])>)> = vec![];
        for p in &films {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for t in idx.tracks.iter().filter(|t| t.entity_type == 15 && t.positions.len() > 400) {
                all.push((t.def_wire.unwrap_or(0), String::new(), t.positions.clone()));
            }
        }
        let names = def_leafs(&all.iter().map(|a| a.0).collect::<Vec<_>>());
        let n = 360usize; // 6 s
        for (def, _, pts) in &all {
            let name = names.get(def).cloned().unwrap_or_default();
            if !matches!(name.as_str(), "warthog" | "mongoose" | "ghost" | "pickup" | "scorpion") { continue; }
            // segments of 6 s by TICK span, gaps of at most 3 ticks, resampled to every tick
            let mut i = 0;
            while i + 10 < pts.len() {
                let t0 = pts[i].0;
                let mut j = i;
                while j + 1 < pts.len() && pts[j + 1].0 - pts[j].0 <= 3 && pts[j + 1].0 - t0 <= n as u32 { j += 1; }
                if pts[j].0 - t0 >= n as u32 - 3 {
                    let seg = &pts[i..=j];
                    let sp = ((seg[seg.len() - 1].1[0] - seg[0].1[0]).powi(2) + (seg[seg.len() - 1].1[1] - seg[0].1[1]).powi(2)).sqrt() / 6.0;
                    if sp > 3.0 {
                        let mut z = vec![0.0f64; n];
                        let mut q = 0usize;
                        for (k, zk) in z.iter_mut().enumerate() {
                            let tk = t0 + k as u32;
                            while q + 1 < seg.len() - 1 && seg[q + 1].0 <= tk { q += 1; }
                            let (a, b) = (&seg[q], &seg[(q + 1).min(seg.len() - 1)]);
                            let f = if b.0 > a.0 { ((tk.saturating_sub(a.0)) as f64 / (b.0 - a.0) as f64).clamp(0.0, 1.0) } else { 0.0 };
                            *zk = a.1[2] as f64 + (b.1[2] - a.1[2]) as f64 * f;
                        }
                        let w = 45usize;
                        let hp: Vec<f64> = (w..n - w).map(|k| z[k] - z[k - w..=k + w].iter().sum::<f64>() / (2 * w + 1) as f64).collect();
                        let m = hp.len();
                        let e = acc.entry(name.clone()).or_insert((vec![0.0; 24], 0));
                        for b in 0..24 {
                            let fq = 0.25 * (b as f64 + 0.5);
                            let (mut re, mut im) = (0.0, 0.0);
                            for (k, v) in hp.iter().enumerate() {
                                let ph = 2.0 * std::f64::consts::PI * fq * k as f64 / 60.0;
                                re += v * ph.cos();
                                im += v * ph.sin();
                            }
                            e.0[b] += (re * re + im * im) / (m * m) as f64;
                        }
                        e.1 += 1;
                    }
                    i = j + 1;
                } else { i += 20; }
            }
        }
        for (name, (p, cnt)) in acc {
            let tot: f64 = p.iter().sum();
            println!("{name}: {cnt} segments; spectrum (share per 0.25 Hz bin from 0.125 Hz): {}", p.iter().map(|v| format!("{:.0}", 100.0 * v / tot)).collect::<Vec<_>>().join(" "));
        }
    }

    /// A landing's rebound, from a position track: after the landing instant `w` (vertical velocity from a fall
    /// of at least `min_fall` wu/s to above -0.5 within 6 ticks) the lowest point `zt` of the next 0.42 s, the
    /// settled level `zs` (mean of ticks 66..78) and the highest point `zp` of the 0.67 s after the trough.
    /// `dip = zs - zt`, `rebound = zp - zs`. Returns `(fall speed, dip, rebound)` when the dip is measurable.
    fn track_rebounds(pts: &[(u32, [f32; 3])], min_fall: f64) -> Vec<(f64, f64, f64)> {
        let mut out = vec![];
        if pts.len() < 130 { return out; }
        let v = |a: usize| ((pts[a + 1].1[2] - pts[a].1[2]) as f64) * 60.0 / (pts[a + 1].0 - pts[a].0).max(1) as f64;
        let mut w = 12usize;
        while w + 110 < pts.len() {
            let cont = (0..80).all(|k| pts[w + k + 1].0 - pts[w + k].0 <= 2);
            if cont && v(w - 4) < -min_fall && v(w + 4) > -0.5 {
                let z = |k: usize| pts[w + k].1[2] as f64;
                let (ti, zt) = (0..25).map(|k| (k, z(k))).fold((0, f64::MAX), |a, b| if b.1 < a.1 { b } else { a });
                let zs = (66..78).map(z).sum::<f64>() / 12.0;
                let zp = (ti..(ti + 40).min(80)).map(z).fold(f64::MIN, f64::max);
                let hs = ((pts[w + 60].1[0] - pts[w].1[0]).powi(2) + (pts[w + 60].1[1] - pts[w].1[1]).powi(2)).sqrt();
                if zs - zt > 0.05 && hs > 0.5 {
                    out.push((-v(w - 4), zs - zt, zp - zs));
                }
                w += 90;
            } else { w += 1; }
        }
        out
    }

    #[test]
    #[ignore = "print-only; reads the saved films"]
    fn probe_film_rebounds() {
        use crate::{film, film_decode};
        let mut found: Vec<(u32, String, (f64, f64, f64))> = vec![];
        for p in &film::find_finished_films() {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for t in idx.tracks.iter().filter(|t| t.entity_type == 15 && t.positions.len() > 200) {
                for r in track_rebounds(&t.positions, 1.5) { found.push((t.def_wire.unwrap_or(0), format!("{}", t.net_id), r)); }
            }
        }
        let names = def_leafs(&found.iter().map(|f| f.0).collect::<Vec<_>>());
        let mut by: std::collections::BTreeMap<String, Vec<(f64, f64, f64)>> = Default::default();
        for (d, _, r) in found { by.entry(names.get(&d).cloned().unwrap_or_default()).or_default().push(r); }
        for (n, v) in &by {
            let mut ratios: Vec<f64> = v.iter().filter(|r| r.1 < 0.5).map(|r| r.2 / r.1).collect();
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !ratios.is_empty() { println!("{n}: dips<0.5 wu: {} landings, rebound/dip median {:.2} (p25 {:.2} p75 {:.2})", ratios.len(), ratios[ratios.len() / 2], ratios[ratios.len() / 4], ratios[ratios.len() * 3 / 4]); }
            println!("{n}: {} landings; (fall, dip, rebound, rebound/dip): {}", v.len(), v.iter().map(|r| format!("({:.1},{:.2},{:+.2},{:.2})", r.0, r.1, r.2, r.2 / r.1)).collect::<Vec<_>>().join(" "));
        }
    }

    #[test]
    #[ignore = "print-only"]
    fn probe_sim_rebounds() {
        let Some((wh, mg)) = real() else { return };
        for (name, d) in [("warthog", wh), ("mongoose", mg)] {
            for fall in [2.5f32, 3.5, 5.0, 8.0, 14.0, 20.0] {
                let h = fall * fall / (2.0 * 4.1712594);
                let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.0199 + h), ..Default::default() };
                st.vel = Vec3::new(3.0, 0.0, 0.0);
                let mut pts = vec![];
                for k in 0..400u32 {
                    st = d.step(st, DriveInput::default(), 1.0 / 60.0, &flat);
                    pts.push((k, [st.pos.x, st.pos.y, st.pos.z]));
                }
                let r = track_rebounds(&pts, 2.0);
                println!("{name} SIM fall {fall}: {:?}", r.iter().map(|r| format!("(fall {:.1} dip {:.2} rebound {:+.2} ratio {:.2})", r.0, r.1, r.2, r.2 / r.1)).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    #[ignore = "print-only"]
    fn probe_sim_rebound_vs_damper() {
        let Some((wh, mg)) = real() else { return };
        for (name, d0) in [("warthog", wh), ("mongoose", mg)] {
            for k in [1.0f32, 1.25, 1.5, 1.75, 2.0, 3.0] {
                let mut d = d0.clone();
                d.wheel_law.as_mut().unwrap().ground.damp *= k;
                let mut rows = vec![];
                for fall in [2.0f32, 3.0, 4.0, 5.0, 6.0, 8.0] {
                    let h = fall * fall / (2.0 * 4.1712594);
                    let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.0199 + h), ..Default::default() };
                    st.vel = Vec3::new(3.0, 0.0, 0.0);
                    let mut pts = vec![];
                    for kk in 0..400u32 {
                        st = d.step(st, DriveInput::default(), 1.0 / 60.0, &flat);
                        pts.push((kk, [st.pos.x, st.pos.y, st.pos.z]));
                    }
                    for r in track_rebounds(&pts, 1.0) { rows.push(format!("({:.1}: dip {:.2} reb {:+.2} = {:.2})", r.0, r.1, r.2, r.2 / r.1)); }
                }
                println!("{name} damper x{k}: {}", rows.join(" "));
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // THE BOUNCE
    // -----------------------------------------------------------------------------------------

    /// The first undershoot, as a fraction of the starting offset, of a vehicle released `offset` wu above its
    /// parked height and left to ring down (small enough that no spring bottoms or lets go).
    fn ring_down_undershoot(d: &DriveModel, offset: f32) -> f32 {
        let rest = settle_flat(d);
        let mut st = rest;
        st.pos.z += offset;
        let mut lowest = f32::MAX;
        for _ in 0..40 {
            st = d.step(st, DriveInput::default(), 1.0 / 60.0, &flat);
            lowest = lowest.min(st.pos.z - rest.pos.z);
        }
        -lowest / offset
    }

    /// **A small drop rings down at the damping ratio the tags imply.** Heave of a body on four equal springs
    /// is a damped oscillator: `omega0^2 = sum(g / depth)`, `2 zeta omega0 = sum(damp) * scale`, and the first
    /// undershoot after a release is `exp(-pi zeta / sqrt(1 - zeta^2))` of the offset. Computed here from the
    /// TAG NUMBERS (depth, damp, wheel count) and the documented damper scale, not from the solver. The
    /// release is 0.015 wu, inside the linear range (every wheel stays loaded).
    #[test]
    fn a_small_drop_rings_down_at_the_damping_ratio_the_tags_imply() {
        let d = fixture();
        let g = crate::reach_physics::vehicle_fall_acceleration();
        let law = d.wheel_law.as_ref().unwrap();
        let n = law.wheels.len() as f32;
        let w0 = (n * g / law.ground.depth).sqrt();
        let zeta = |scale: f32| n * law.ground.damp * scale / (2.0 * w0);
        let overshoot = |z: f32| (-std::f32::consts::PI * z / (1.0 - z * z).sqrt()).exp();
        let got = ring_down_undershoot(&d, 0.015);
        let want = overshoot(zeta(2.0));
        println!("zeta with the documented x2 damper: {:.3}, undershoot {got:.3} (law {want:.3}); the literal law would be zeta {:.3}, undershoot {:.3}", zeta(2.0), zeta(1.0), overshoot(zeta(1.0)));
        assert!(got > 0.02, "it must actually ring (liveness): {got}");
        assert!((got - want).abs() < 0.06, "undershoot {got:.3} against the damping ratio the tags imply, {want:.3}");
    }

    /// Rebound/dip ratios of simulated landings (falls of 2-8 wu/s onto flat ground), found with the SAME
    /// detector the recordings go through ([`track_rebounds`]).
    fn sim_rebound_ratios(d: &DriveModel) -> Vec<(f64, f64, f64)> {
        let mut out = vec![];
        for fall in [2.0f32, 3.0, 4.0, 5.0, 6.0, 8.0] {
            let h = fall * fall / (2.0 * 4.1712594);
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.0199 + h), ..Default::default() };
            st.vel = Vec3::new(3.0, 0.0, 0.0);
            let mut pts = vec![];
            for k in 0..400u32 {
                st = d.step(st, DriveInput::default(), 1.0 / 60.0, &flat);
                pts.push((k, [st.pos.x, st.pos.y, st.pos.z]));
            }
            out.extend(track_rebounds(&pts, 1.0));
        }
        out
    }

    /// **LANDINGS REBOUND AS LITTLE AS THE RECORDED ONES.** Landings found in the saved films (a fall of at
    /// least 1.5 wu/s ending within 6 ticks, a dip under half a wu, the track continuous for 1.3 s after) give a
    /// rebound/dip ratio; the SAME detector on simulated landings must give a median inside the recordings'
    /// interquartile range, for the Warthog and the Mongoose. Liveness: enough recorded landings and enough
    /// simulated ones to have a median at all. (The literal engine law gives a median of 0.8 and is red.)
    #[test]
    #[ignore = "opens the native library and reads the saved films on this machine"]
    fn the_landing_rebound_matches_the_recordings() {
        use crate::{film, film_decode};
        let Some((wh, mg)) = real() else { return };
        if film::find_finished_films().is_empty() {
            eprintln!("SKIPPED: no saved films on this machine");
            return;
        }
        let mut found: Vec<(u32, f64)> = vec![];
        for p in &film::find_finished_films() {
            let Ok(f) = film::open(p) else { continue };
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for t in idx.tracks.iter().filter(|t| t.entity_type == 15 && t.positions.len() > 200) {
                for r in track_rebounds(&t.positions, 1.5) {
                    if r.1 < 0.5 { found.push((t.def_wire.unwrap_or(0), r.2 / r.1)); }
                }
            }
        }
        let names = def_leafs(&found.iter().map(|f| f.0).collect::<Vec<_>>());
        let mut hog: Vec<f64> = found.iter().filter(|f| names.get(&f.0).is_some_and(|n| n == "warthog")).map(|f| f.1).collect();
        hog.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (q1, med, q3) = (hog[hog.len() / 4], hog[hog.len() / 2], hog[hog.len() * 3 / 4]);
        println!("recorded Warthog landings: {} , rebound/dip median {med:.2} IQR {q1:.2}..{q3:.2}", hog.len());
        assert!(hog.len() >= 8, "too few recorded landings: {}", hog.len());
        for (name, d) in [("Warthog", wh), ("Mongoose", mg)] {
            let mut r: Vec<f64> = sim_rebound_ratios(d).iter().map(|r| r.2 / r.1).collect();
            r.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("{name}: {} simulated landings, median {:.2}: {:?}", r.len(), r[r.len() / 2], r.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
            assert!(r.len() >= 4, "{name}: too few simulated landings to say anything");
            assert!(r[r.len() / 2] >= q1 && r[r.len() / 2] <= q3, "{name}: simulated rebound/dip {:.2} outside the recordings' {q1:.2}..{q3:.2}", r[r.len() / 2]);
        }
    }

    /// **A drop settles.** Dropped from 1 wu the Warthog and the Mongoose are back within 2 cm of their parked
    /// height 1.2 s after the first touch -- the interval in which the recorded landings are read as settled --
    /// and are not moving.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_dropped_vehicle_is_settled_within_a_second_and_a_fifth() {
        let Some((wh, mg)) = real() else { return };
        for (name, d) in [("Warthog", wh), ("Mongoose", mg)] {
            let rest = settle_flat(d);
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, rest.pos.z + 1.0), ..Default::default() };
            let mut touched = None;
            let mut dev_at = None;
            for k in 0..400u32 {
                st = d.step(st, DriveInput::default(), 1.0 / 60.0, &flat);
                let t = k as f32 / 60.0;
                if touched.is_none() && st.touching[..4].iter().any(|&b| b) { touched = Some(t); }
                if let Some(t0) = touched {
                    if (t - t0 - 1.2).abs() < 0.009 { dev_at = Some(((st.pos.z - rest.pos.z).abs(), st.vel.z.abs())); }
                }
            }
            let (dev, v) = dev_at.expect("it must land");
            println!("{name}: 1.2 s after touching, {dev:.4} wu off its parked height, vz {v:.3}");
            assert!(dev < 0.02 && v < 0.1, "{name} is still bouncing 1.2 s after landing: {dev:.3} wu, vz {v:.3}");
        }
    }

    /// **A ramp launches no faster than its geometry.** Driving up a ramp the vertical speed at takeoff cannot
    /// honestly exceed `horizontal speed * tan(ramp)` by much -- that is the velocity of a point following the
    /// slope -- and a spring that rebounds adds to it (the literal law gave 1.23x and 1.27x on 20 and 30 degree
    /// ramps). Between 0.6x (a hull that does not follow) and 1.1x. Both vehicles, 15-30 degrees.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_ramp_launches_no_faster_than_its_geometry() {
        let Some((wh, mg)) = real() else { return };
        for (name, d) in [("Warthog", wh), ("Mongoose", mg)] {
            for deg in [15.0f32, 20.0, 30.0] {
                let t = deg.to_radians().tan();
                let ramp = move |p: Vec3| -> Option<(f32, Vec3)> {
                    if p.x < 30.0 { Some((0.0, Vec3::Z)) } else if p.x < 34.0 { Some(((p.x - 30.0) * t, Vec3::new(-t, 0.0, 1.0).normalize())) } else { None }
                };
                let mut st = settle_flat(d);
                st.pos.x = 0.0;
                let mut took_off = None;
                let mut was_down = false;
                for _ in 0..1500 {
                    st = d.step(st, DriveInput { throttle: 1.0, ..Default::default() }, 1.0 / 60.0, &ramp);
                    let down = st.touching[..4].iter().any(|&b| b);
                    if was_down && !down && took_off.is_none() { took_off = Some(st); }
                    was_down |= down;
                    if took_off.is_some() { break; }
                }
                let a = took_off.expect("it must leave the ramp");
                let hs = a.vel.truncate().length();
                let ratio = a.vel.z / (hs * t);
                println!("{name} {deg} deg ramp: vz {:.2} at hspeed {hs:.2}, geometry {:.2}: x{ratio:.2}", a.vel.z, hs * t);
                assert!(ratio > 0.6 && ratio < 1.1, "{name} {deg} deg: takeoff vz is x{ratio:.2} of v*tan(ramp)");
            }
        }
    }
}
