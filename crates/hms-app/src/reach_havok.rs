//! #havok  The layer UNDER Reach's vehicle code: the Havok rigid body that every vehicle (and every
//! crate, dropped weapon and ragdoll) is, with the engine's own numbers.
//!
//! Everything here is pure -- no HMS state, no globals -- so the vehicle code, the settle and the
//! ragdoll can call it and a gate can drive it. Every claim is labelled **PROVEN** (decompiled from
//! `reach_tag_test.exe` / `haloreach.dll` with the function named) or **INFERRED** (Havok semantics
//! the dumps do not lift: the solver is not in `~/sdk-dump`, only class stubs are).
//!
//! # What a Reach vehicle is, underneath (the answer to "what drives the hull")
//!
//! **PROVEN.** A vehicle is NOT driven by Havok forces. `havok_vehicle_physics.cpp` is the engine's
//! own integrator and it runs on a *copy* of the body's state:
//!
//! ```text
//!   tick, per vehicle  (vehicle_type_*.cpp tick, e.g. jeep sub_140E69D40)
//!     sub_1406CE570   read the Havok body: pose (+0x10 matrix, +0x68 position), mass (+0x74),
//!                     linear / angular velocity (+0x8C / +0xA4) minus the object-frame velocity
//!                     (+0x98 / +0xB0, get_state_scale = the velocity of whatever it rides on)
//!     sub_1406CF0F0   repeat `havok.iteration count` times (vehi +0x5B4, clamped >= 1, sub_1406D51E0)
//!                     with h = tick_length / n:
//!                       sub_1406CC8C0 -> force F, torque T from friction points + anti-gravity
//!                                        points + "magic force/torque"
//!                       a = F / instance->mass ;  alpha = I^-1 T          (mass asserted > 0)
//!                       v += a h ;  w += alpha h ;  [optional extra damping when a3 > 0]
//!                       position / orientation advanced by v h, w h  (the COPY, for the next sub-step)
//!                     then v += object-frame velocity.
//!     sub_1406CEB20   push the result to Havok with SET-VELOCITY: `setLinearVelocity` (motion
//!                     vtable +0x80) and `setAngularVelocity` (+0x88) on the rigid body, via
//!                     havok_component sub_14044B9C0 / sub_14044B7C0.  NOT setTransform, NOT
//!                     applyForce, NOT applyImpulse.
//! ```
//!
//! **There is no gravity in that code.** No function in `havok_vehicle_physics.cpp` reads the
//! `"physics constants"` block; gravity is applied by Havok's integrator (below) to the body's
//! velocity, scaled by the body's `gravityFactor` (hkpMotion +0x132). The wheels' springs
//! (`friction points`, `ground depth`) and the hover pads hold the hull up against it.
//!
//! # What Havok then does with it, per PSI (one game tick, [`TICK_HZ`])
//!
//! **PROVEN**, `sub_141106780` (hkRigidMotionUtilApplyForcesAndStep; caller `sub_1410F27B0`
//! passes the world's `gravity * deltaTime` vector at world+0x2C0). For motion types dynamic /
//! sphere / box inertia (1, 2, 3):
//!
//! ```text
//!   v  += gravityFactor * gravity * dt          gravityFactor = the half at motion+0x132
//!   v  *= max(0, 1 - dt * linearDamping)        motionState +0xA4   (phmo rigid body +0x30)
//!   w  *= max(0, 1 - dt * angularDamping)       motionState +0xA8   (phmo rigid body +0x34)
//!   NaN / |component| >= 1e6  -> v = w = 0
//!   |v| > maxLinear  -> v scaled to maxLinear   (hkUFloat8 table, see [`ufloat8_decode`])
//!   com += dt * v
//!   q    = normalize(dq * q)                    dq = (h, 1 - .823 s - .1305 s^2 - .0444 s^3),
//!                                               h = w dt / 2, s = (2|h|/pi)^2 clamped to
//!                                               min(0.9, dt * maxAngular)^2
//! ```
//!
//! Character motion (7) takes the damping but NOT the gravity; keyframed (4) takes neither and is
//! only advanced; fixed (5) is skipped. [`HavokBody::step_psi`] is exactly this.
//!
//! # The seam for the vehicle code
//!
//! ```text
//!   // once, when the hull is created
//!   let mut hull = HavokBody::new(MotionType::BoxInertia, mass, inv_inertia_local, com, q)
//!       .with_damping(phmo_lin_damping, phmo_ang_damping);        // phmo rigid body +0x30 / +0x34
//!   hull.set_gravity_factor(vehicle_body_gravity_factor(1.0, antigravity_scale,
//!       vehicle_gravity_scale_postprocessed(vehi_havok_gravity_scale_raw)));
//!
//!   // every tick (units: world units, seconds, radians; dt = TICK_SECONDS = 1/60)
//!   hull.set_velocity(v_after_vehicle_law, w_after_vehicle_law);   // sub_1406CEB20
//!   hull.step_psi(TICK_SECONDS, world_gravity());                  // gravity, damping, clamp, move
//!   let contacts: Vec<ContactPoint> = /* hull shapes vs world at hull.com / hull.q */;
//!   solve_contacts(&mut hull, &contacts, TICK_SECONDS, &SolverParams::ENGINE);
//!   // next tick: the vehicle law reads hull.v / hull.w back (sub_1406CE570)
//! ```
//!
//! # Contacts (what a hull does against a mesh, another vehicle, a biped)
//!
//! **PROVEN** (`sub_140452520`, the contact-added listener; `sub_14034DC60`; `sub_14044C7E0`):
//! every contact carries `friction = sqrt(fA * fB)` and `restitution = sqrt(rA * rB)`, where each
//! side's pair is the SHAPE's own: a `phmo` primitive's `friction` (+0x0C) / `restitution` (+0x10),
//! or, for the world BSP, the global material's physics-properties record
//! (`matg materials[i]` +0x24 / +0x28). The friction is stored as an hkUFloat8 (quantised UP to
//! the table, [`contact_friction_stored`]) and the restitution as `(int)(r * 128)` in a byte
//! ([`contact_restitution_stored`]). The Havok material of the BODY itself (`hkpEntity` +0xD4 /
//! +0xD8) is `friction 1.0, restitution 0.0` for every phmo body (`sub_140439DF0`).
//!
//! **PROVEN** world settings ([`SOLVER_TAU`] .. [`CONTACT_RESTING_VELOCITY`]) from `sub_18009A06C`.
//! `contactRestingVelocity` is `FLT_MAX`. **INFERRED** (Havok semantics, not lifted): restitution is
//! applied only to contacts closing faster than that velocity, so with the engine's value a Reach
//! rigid body does not bounce at all. [`SolverParams::ENGINE`] encodes that and a gate pins it.
//!
//! # UNRESOLVED (named, and who it blocks)
//!
//! * the Havok contact solver itself (soft-contact constraint equations, how `tau`/`damp` enter):
//!   [`solve_contacts`] is a sequential-impulse stand-in using the engine's iteration count and
//!   `tau` as a position-correction fraction -- INFERRED, not a port. Blocks nothing today.
//! * `jeep` "gravity adjust" (`vehi +0x5B4-region` jeep block +0x54) -- see
//!   [`vehicle_body_gravity_factor`]: `sub_140E61E80` only changes the factor when the vehicle's up
//!   vector opposes world up (it is upside down) and more than half its wheels are in contact; the
//!   blend constant is `object+0x64` (meaning unresolved). Blocks: jeep agent, only for flipped
//!   vehicles.
//! * `object +0x3A0` (the `antigravity_scale` below): the engine multiplies the vehicle's gravity
//!   scale in only when it is non-zero; what zeroes it (EMP / destroyed hover?) is unresolved.
//! * the TOI / continuous-simulation pass and `havok_collision_damage_update` (impulse -> damage).
//! * quality-type RESPONSE tables (`hkpCollisionQualityInfo`, built in the world ctor): present in
//!   `sub_1410CDD70` (`1.19e-7 / 0.017 / 0.025` thresholds) but not ported here.

use glam::{Mat3, Quat, Vec3};

// ------------------------------------------------------------------------------------------
// Tick
// ------------------------------------------------------------------------------------------

/// The engine's tick rate, Hz.
///
/// **PROVEN for the tool build**: `game_options_get` (`sub_1400B0E20`) sets the tick rate to
/// `(int)(sub_1407C4FB0() + 0.5) / dword_141E05934` where `sub_1407C4FB0()` is the constant
/// `60.0` and the divisor is `1`; `game_time_globals->tick_rate` (`+6`) feeds
/// `seconds_per_tick = 1.0 / tick_rate` (`+8`, `sub_1400BDCC0`, read back by the vehicle and
/// biped code as the tick length).
///
/// **PROVEN for the shipped game too** (`haloreach.dll`, and `haloreach_debug.dll`, which keeps the
/// asserts): `sub_18005C338` (debug `sub_180645530`) initialises `game_time_globals` (TLS+0xA0)
/// with `tick_rate = game_options.tick_rate` (+6) and `seconds_per_tick = 1.0 / tick_rate` (+8);
/// the default options builder `sub_180057B6C` stores `0x3C` = 60 there (debug build: `60.0 /
/// divisor 1`); `Theater_SetPlaybackRate` is the only other writer (it rescales the rate for slow
/// motion). The world is stepped once per game tick with `dt = seconds_per_tick`
/// (`Game_RunMainFrameTick` -> `RenderUIFrame` -> `UIContext_PresentFrame(0, dt)` -> simulation
/// vtable +0x18); `Game_TryRunFrameTicks` calls the frame tick `n` times with no parity gate.
/// The vehicle integrator (`sub_18098B430` in the debug build) uses `tick_length / iterations`;
/// the ground character mode (`sub_180975DA0`) applies gravity as `tick_length * g` and scales its
/// accelerations by `1 / tick_rate`; its only `1/30` is a literal inside a branch that runs once per
/// second (`ticks % tick_rate == 0`).
///
/// `physics_settle::TICK_HZ` now agrees (60).
pub const TICK_HZ: f32 = 60.0;
/// One tick, seconds: the `dt` of every Havok step and of the vehicle sub-steps' parent.
pub const TICK_SECONDS: f32 = 1.0 / TICK_HZ;

// ------------------------------------------------------------------------------------------
// World cinfo (haloreach.dll sub_18009A06C / sub_1808A6DA0 / sub_1808A6F80)
// ------------------------------------------------------------------------------------------

/// `hkpWorldCinfo::solverTau` -- `0x3E99999A`, set by the solver preset `sub_1808A6F80` (SDK 0.6).
pub const SOLVER_TAU: f32 = 0.3;
/// `hkpWorldCinfo::solverDamp` -- `0x3F666666` (SDK 1.0).
pub const SOLVER_DAMP: f32 = 0.9;
/// `hkpWorldCinfo::solverIterations` -- 8 (SDK 4).
pub const SOLVER_ITERATIONS: u32 = 8;
/// `hkpWorldCinfo::solverMicrosteps` -- 1 (the ctor default; the preset does not touch it).
pub const SOLVER_MICROSTEPS: u32 = 1;
/// `hkpWorldCinfo::collisionTolerance`, wu. **Not scaled by the tick rate**: the game scales it
/// only when `byte_184E2FE60` is non-zero, and that byte has exactly ONE reference in the whole
/// dll -- the read in `sub_18009A06C` -- so it is never written and stays at its BSS zero.
pub const COLLISION_TOLERANCE: f32 = 0.0328;
/// `hkpWorldCinfo::expectedMaxLinearVelocity`, wu/s (SDK 200).
pub const EXPECTED_MAX_LINEAR_VELOCITY: f32 = 60.0;
/// `hkpWorldCinfo::contactRestingVelocity` -- `FLT_MAX` (SDK 1.0). See the module note.
pub const CONTACT_RESTING_VELOCITY: f32 = f32::MAX;
/// `hkpWorldCinfo::iterativeLinearCastEarlyOutDistance`, wu (SDK 0.01).
pub const ITERATIVE_LINEAR_CAST_EARLY_OUT: f32 = 0.00328;
/// `hkpWorldCinfo::expectedMinPsiDeltaTime`: `game_time_globals + 8`, the length of one tick.
pub const EXPECTED_MIN_PSI_DELTA_TIME: f32 = TICK_SECONDS;
/// `hkpWorldCinfo::deactivationReferenceDistance` at tick rate `tps`: the game computes
/// `0.1 * (1 / (tps / 30))` with INTEGER division (`sub_18009A06C`), which is 0.05 at 60 Hz.
pub fn deactivation_reference_distance(tps: u32) -> f32 {
    let k = (tps / 30).max(1);
    0.1 * (1.0 / k as f32)
}
/// The cinfo-ctor default for a rigid body's `maxLinearVelocity` AND `maxAngularVelocity`
/// (`sub_1410D7CB0`: `0x43480000`). The engine never overrides it for a phmo body
/// (`sub_14043B2C0` leaves `+0xB8/+0xBC` alone).
pub const DEFAULT_MAX_VELOCITY: f32 = 200.0;
/// The floor under every body's `allowedPenetrationDepth` (`sub_14043B2C0`:
/// `fmaxf(scale * k * minDimension, globals[+0x2C])`, and that global is `0.01`).
pub const MIN_ALLOWED_PENETRATION_DEPTH: f32 = 0.01;

/// The world gravity vector Havok integrates with: `(0, 0, -g)` where `g` is the engine's
/// `"physics constants"` block value ([`crate::reach_physics::WORLD_GRAVITY`]). Havok's own world
/// gravity is this (`sub_18009A06C` builds it from `*(float *)physics_constants`).
pub fn world_gravity() -> Vec3 {
    Vec3::new(0.0, 0.0, -crate::reach_physics::WORLD_GRAVITY)
}

// ------------------------------------------------------------------------------------------
// hkUFloat8 and the half-floats
// ------------------------------------------------------------------------------------------

/// The hkUFloat8 decode table, 256 entries, monotonic from 0 to 1,000,002. **PROVEN**: the 1 KiB
/// at `reach_tag_test.exe` VA `0x141A9C4C0` (`flt_141A9C4C0`), read by the integrator for the
/// per-body velocity caps and written by the encoder `sub_14103C270`. Stored as bit patterns so a
/// reader cannot round a literal.
#[rustfmt::skip]
const UFLOAT8_BITS: [u32; 256] = [
    0x00000000, 0x3C23D70A, 0x3CA3D70A, 0x3CF5C28F, 0x3D23D70A, 0x3D4CCCCD, 0x3D75C28F, 0x3D8F5C29,
    0x3D99999A, 0x3DA3D70A, 0x3DAE147B, 0x3DBA5E35, 0x3DC8B439, 0x3DD4FDF4, 0x3DE353F8, 0x3DF3B646,
    0x3E020C4A, 0x3E0B4396, 0x3E147AE1, 0x3E1EB852, 0x3E29FBE7, 0x3E353F7D, 0x3E428F5C, 0x3E4FDF3B,
    0x3E5E353F, 0x3E6C8B44, 0x3E7CED91, 0x3E872B02, 0x3E90624E, 0x3E9A9FBE, 0x3EA56042, 0x3EB0A3D7,
    0x3EBC6A7F, 0x3EC9BA5E, 0x3ED78D50, 0x3EE66666, 0x3EF645A2, 0x3F039581, 0x3F0C8B44, 0x3F1645A2,
    0x3F208312, 0x3F2B851F, 0x3F374BC7, 0x3F43D70A, 0x3F516873, 0x3F5FBE77, 0x3F6F1AA0, 0x3F7F7CEE,
    0x3F889375, 0x3F91EB85, 0x3F9BE76D, 0x3FA6A7F0, 0x3FB20C4A, 0x3FBE5604, 0x3FCB645A, 0x3FD95810,
    0x3FE851EC, 0x3FF83127, 0x40049BA6, 0x400DC28F, 0x40177CEE, 0x4021DB23, 0x402CFDF4, 0x4038E560,
    0x40459168, 0x405322D1, 0x4061A9FC, 0x407126E9, 0x4080DD2F, 0x4089BA5E, 0x40932B02, 0x409D47AE,
    0x40A81893, 0x40B39DB2, 0x40BFF7CF, 0x40CD26E9, 0x40DB3B64, 0x40EA4DD3, 0x40FA6666, 0x4105CAC1,
    0x410EFDF4, 0x4118D0E5, 0x41235C29, 0x412E8F5C, 0x413A8F5C, 0x41475C29, 0x4154F5C3, 0x4163AE14,
    0x41733333, 0x41820000, 0x418AF5C3, 0x41947AE1, 0x419EA3D7, 0x41A9999A, 0x41B53333, 0x41C1AE14,
    0x41CEF5C3, 0x41DD3333, 0x41EC6666, 0x41FC8F5C, 0x4206F5C3, 0x42103D71, 0x421A28F6, 0x4224B852,
    0x42300A3D, 0x423C28F6, 0x4249147B, 0x4256E148, 0x4265A3D7, 0x42756666, 0x428323D7, 0x428C23D7,
    0x4295C28F, 0x42A00F5C, 0x42AB0A3D, 0x42B6CCCD, 0x42C35C29, 0x42D0CCCD, 0x42DF3333, 0x42EE6666,
    0x42FECCCD, 0x43083333, 0x43118000, 0x431B8000, 0x43263333, 0x4331999A, 0x433DCCCD, 0x434ACCCD,
    0x4358CCCD, 0x4367B333, 0x4377999A, 0x43844CCD, 0x438D599A, 0x4397199A, 0x43A17333, 0x43AC8CCD,
    0x43B86666, 0x43C50CCD, 0x43D2999A, 0x43E10CCD, 0x43F08CCD, 0x44008666, 0x4409599A, 0x4412CCCD,
    0x441CE000, 0x4427A666, 0x44332666, 0x443F799A, 0x444CA000, 0x445AACCD, 0x4469B333, 0x4479C000,
    0x44858000, 0x448EA000, 0x44986000, 0x44A2E000, 0x44AE2000, 0x44BA0000, 0x44C6C000, 0x44D48000,
    0x44E30000, 0x44F2A000, 0x4501A000, 0x450A9000, 0x45141000, 0x451E4000, 0x45292000, 0x4534C000,
    0x45412000, 0x454E7000, 0x455CA000, 0x456BC000, 0x457BF000, 0x4586A000, 0x458FE000, 0x4599C000,
    0x45A45000, 0x45AF9800, 0x45BBA800, 0x45C89000, 0x45D65000, 0x45E50800, 0x45F4C800, 0x4602CC00,
    0x460BC800, 0x46156000, 0x461FA400, 0x462A9C00, 0x46365400, 0x4642D800, 0x46503C00, 0x465E8800,
    0x466DD000, 0x467E2800, 0x4687CE00, 0x46912200, 0x469B1A00, 0x46A5C200, 0x46B12400, 0x46BD4E00,
    0x46CA5000, 0x46D83400, 0x46E70E00, 0x46F6EE00, 0x4703F200, 0x470D0200, 0x4716B100, 0x47210B00,
    0x472C1B00, 0x4737ED00, 0x47448F00, 0x47520F00, 0x47607D00, 0x476FE800, 0x47803180, 0x4788FF80,
    0x47926880, 0x479C7700, 0x47A73600, 0x47B2B280, 0x47BEF880, 0x47CC1680, 0x47DA1B00, 0x47E91600,
    0x47F91900, 0x48051A80, 0x480E3F00, 0x48180400, 0x48227500, 0x482D9DC0, 0x48398A80, 0x48464900,
    0x4853E7C0, 0x486275C0, 0x487203C0, 0x488151C0, 0x488A33A0, 0x4893B1C0, 0x489DD6C0, 0x48A8AE20,
    0x48B44440, 0x48C0A600, 0x48CDE180, 0x48DC05A0, 0x48EB2280, 0x48FB4900, 0x490645C0, 0x490F7ED0,
    0x49195A00, 0x4923E280, 0x492F2430, 0x493B2BE0, 0x49480710, 0x4955C450, 0x49647320, 0x49742420,
];

/// hkUFloat8 -> f32.
pub fn ufloat8_decode(code: u8) -> f32 {
    f32::from_bits(UFLOAT8_BITS[code as usize])
}

/// f32 -> hkUFloat8, exactly as the engine's encoder `sub_14103C270`: values below 0.01 (and all
/// negatives) become 0; otherwise the SMALLEST table entry that is `>=` the value, clamped to the
/// last. **PROVEN** by emulating the encoder's seven-step pointer bisection against `bisect_left`
/// over the table, all 256 entries +-1 ulp and 20,000 random patterns: 0 differences.
pub fn ufloat8_encode(v: f32) -> u8 {
    let bits = v.to_bits() as i32;
    if bits < 0x3C23_D70A {
        return 0;
    }
    let idx = UFLOAT8_BITS.partition_point(|&t| (t as i32) < bits);
    idx.min(255) as u8
}

/// The value a velocity cap actually has once the engine has stored it: `decode(encode(v))`.
/// `DEFAULT_MAX_VELOCITY` (200) becomes 202.8.
pub fn quantised_max_velocity(v: f32) -> f32 {
    ufloat8_decode(ufloat8_encode(v))
}

/// The body's gravity factor is an `hkHalf`, and in this build that is the **top 16 bits of the
/// f32 (truncated, not rounded)**: `sub_14044B880` stores `bits(f) >> 16` at motion+0x132 and the
/// integrator reads it back as `(int16)word << 16`. 1.0 is `0x3F80`.
pub fn gravity_factor_bits(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// The inverse of [`gravity_factor_bits`].
pub fn gravity_factor_from_bits(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ------------------------------------------------------------------------------------------
// Gravity scale
// ------------------------------------------------------------------------------------------

/// `vehi` `havok_vehicle_physics_struct` `gravity scale` (`vehi +0x5AC`) after the tag
/// postprocess `sub_1406D51E0`: **a value <= 0 becomes 1.0; a value in (0, 0.001] becomes 0.0**;
/// anything else is kept. So an authored 0 means "unscaled", and an authored 0.0005 means "no
/// gravity".
pub fn vehicle_gravity_scale_postprocessed(raw: f32) -> f32 {
    if raw > 0.0 {
        if raw <= 0.001 { 0.0 } else { raw }
    } else {
        1.0
    }
}

/// The gravity factor a vehicle's Havok body gets (`sub_140D9D3D0` computes it when the object
/// connects to the map; `sub_140DDAE20` is the vehicle's contribution):
///
/// * `bsp_gravity_scale` -- the structure BSP's `custom gravity scale` the object is in (`sub_14037A190`,
///   1.0 in all 31 rows of 7 maps measured; 1.0 when the object has no BSP);
/// * `antigravity_scale` -- the vehicle object's float at `+0x3A0` (the same value the jeep tick
///   copies into every anti-gravity point). The vehicle multiplies its `gravity scale` in ONLY
///   when this is non-zero; otherwise the factor is left at the BSP's;
/// * `post_scale` -- [`vehicle_gravity_scale_postprocessed`].
///
/// Applied at ALL times, grounded or airborne: it scales the Havok gravity term, and the wheels /
/// hover pads then hold the hull up against it. Not ported: the per-type step that follows
/// (`sub_140E06D20`; the jeep's `gravity adjust`, see the module note), and the game engine's
/// virtual `+0x1F8` (megalo traits).
pub fn vehicle_body_gravity_factor(bsp_gravity_scale: f32, antigravity_scale: f32, post_scale: f32) -> f32 {
    if antigravity_scale != 0.0 {
        bsp_gravity_scale * post_scale
    } else {
        bsp_gravity_scale
    }
}

/// The mass the vehicle code divides by: the Havok body's mass, floored at 1.0 (`sub_1404442E0` =
/// `fmaxf(mass, 1.0)`).
pub fn vehicle_instance_mass(havok_mass: f32) -> f32 {
    havok_mass.max(1.0)
}

// ------------------------------------------------------------------------------------------
// Materials
// ------------------------------------------------------------------------------------------

/// Two contacting shapes' friction combined: the geometric mean (`sqrtf(a * b)`,
/// `sub_140452520`; also `sub_1410F7FA0`). Commutative; 0 if either is 0.
pub fn combined_friction(a: f32, b: f32) -> f32 {
    (a * b).sqrt()
}

/// Two contacting shapes' restitution combined: the geometric mean too.
pub fn combined_restitution(a: f32, b: f32) -> f32 {
    (a * b).sqrt()
}

/// The friction the contact point actually carries: [`combined_friction`] stored as an hkUFloat8
/// (`sub_14044C560` / `sub_14044C7E0` call the encoder), i.e. quantised UP to the table.
pub fn contact_friction_stored(combined: f32) -> f32 {
    ufloat8_decode(ufloat8_encode(combined))
}

/// The restitution the contact point actually carries: `(int)(r * 128)` in a byte (the
/// `cvttss2si` / `mov [r14+0x11], al` at the end of `sub_14044C7E0`), read back as `byte / 128`.
pub fn contact_restitution_stored(combined: f32) -> f32 {
    ((combined * 128.0) as i32 as u8) as f32 / 128.0
}

/// The contact material for a pair of shapes, ready for [`ContactPoint`]: combined, then stored.
pub fn contact_material(friction_a: f32, restitution_a: f32, friction_b: f32, restitution_b: f32) -> (f32, f32) {
    (
        contact_friction_stored(combined_friction(friction_a, friction_b)),
        contact_restitution_stored(combined_restitution(restitution_a, restitution_b)),
    )
}

/// Which friction the engine writes into a contact (`sub_14044C7E0`): the contact datum's own
/// value, except 0 when the datum is flagged (byte `+0x4A` bit `0x40`) or when the OTHER body is a
/// vehicle whose `vehi +0x4CC` bit 0 is set and which is occupied (`unit +0x38C != NONE` or
/// `+0x1C4` bit 1). The predicates' names are UNRESOLVED; the arithmetic is not.
pub fn contact_friction_written(datum_friction: f32, datum_flag_zero_friction: bool, other_vehicle_zeroes_friction: bool) -> f32 {
    if datum_flag_zero_friction || other_vehicle_zeroes_friction {
        0.0
    } else {
        datum_friction
    }
}

/// The friction and restitution of one `matg` `materials[]` element (424 bytes): its physics
/// properties record (`mphp`, +0x20) as the engine's runtime material record exposes it --
/// `+0x24` friction, `+0x28` restitution (`sub_14034DC60` loading `sub_1403EB4A0(material)`).
/// **INFERRED** that the runtime record is the tag element verbatim; a real-tag gate checks the
/// values read sane across every material.
pub fn global_material_friction_restitution(element: &[u8]) -> Option<(f32, f32)> {
    let f = |o: usize| -> Option<f32> {
        let b = element.get(o..o + 4)?;
        Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    Some((f(GLOBAL_MATERIAL_FRICTION)?, f(GLOBAL_MATERIAL_RESTITUTION)?))
}
/// `matg` `materials` block offset and element size (tagdefs `matg`).
pub const MATG_MATERIALS: usize = 0x48C;
pub const MATG_MATERIAL_SIZE: usize = 424;
pub const GLOBAL_MATERIAL_FRICTION: usize = 0x24;
pub const GLOBAL_MATERIAL_RESTITUTION: usize = 0x28;

// ------------------------------------------------------------------------------------------
// Body creation rules (havok_component.cpp sub_14043B2C0 / sub_140439DF0)
// ------------------------------------------------------------------------------------------

/// `hkpMotion::MotionType`, the numbering the engine passes (`sub_14043B2C0`, `sub_1410D7DE0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionType {
    Dynamic = 1,
    SphereInertia = 2,
    BoxInertia = 3,
    Keyframed = 4,
    Fixed = 5,
    ThinBoxInertia = 6,
    Character = 7,
}

/// `hkpCollidable` quality type (the byte at body+0x4A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityType {
    Invalid = -1,
    Fixed = 0,
    Keyframed = 1,
    Debris = 2,
    DebrisSimpleToi = 3,
    Moving = 4,
    Critical = 5,
    Bullet = 6,
    User = 7,
    Character = 8,
    KeyframedReporting = 9,
}

/// The object type numbers `sub_14043B2C0` branches on (the byte at `object header +4`).
pub mod object_type {
    pub const BIPED: u8 = 0;
    pub const VEHICLE: u8 = 1;
    pub const WEAPON: u8 = 2;
    pub const EQUIPMENT: u8 = 3;
    pub const SCENERY: u8 = 6;
}

/// The quality type a body is created with. **PROVEN** from `sub_14043B2C0` + the body ctor
/// `sub_1410D7DE0`:
///
/// * keyframed / fixed bodies pass -1 and the ctor resolves it to `Keyframed` / `Fixed`;
/// * every other body starts at **Critical (5)**; a biped (object type 0) whose definition flag
///   (`+0x138` bit 2) is clear becomes **Character (8)**; weapons, equipment and vehicles stay
///   Critical; any other type on collision layer `0x12` / `0x13` becomes **Debris (2)**;
/// * an object definition with flag bit 25 (`+0xDC & 0x2000000`) is forced to Debris.
///
/// So a vehicle hull is `Critical`, the most conservative continuous-collision quality.
pub fn body_quality_type(
    motion: MotionType,
    object_type: u8,
    layer: u32,
    definition_flag_bit25: bool,
    biped_definition_flag_bit2: bool,
) -> QualityType {
    use object_type::*;
    match motion {
        MotionType::Fixed => return QualityType::Fixed,
        MotionType::Keyframed => return QualityType::Keyframed,
        _ => {}
    }
    let mut q = QualityType::Critical;
    match object_type {
        BIPED => {
            if !biped_definition_flag_bit2 {
                q = QualityType::Character;
            }
        }
        VEHICLE | WEAPON | EQUIPMENT => {}
        _ => {
            if layer == 0x12 || layer == 0x13 {
                q = QualityType::Debris;
            }
        }
    }
    if definition_flag_bit25 {
        q = QualityType::Debris;
    }
    q
}

/// The body's `allowedPenetrationDepth` (`sub_14043B2C0`):
/// `max(scale * k * minimum_dimension, 0.01)` with `k = 0.4`, `0.2` for an unflagged biped, and
/// for a vehicle whose `vehi +0x4CC` bit 10 is set the product is `scale * 0.025` (a fixed
/// depth, no dimension). `minimum_dimension` is the smallest dimension of the body's shapes.
pub fn allowed_penetration_depth(
    scale: f32,
    minimum_dimension: f32,
    object_type: u8,
    biped_definition_flag_bit2: bool,
    vehicle_flag_bit10: bool,
) -> f32 {
    let mut d = minimum_dimension * 0.4;
    if object_type == object_type::BIPED && !biped_definition_flag_bit2 {
        d = minimum_dimension * 0.2;
    } else if object_type == object_type::VEHICLE && vehicle_flag_bit10 {
        d = 0.025;
    }
    (scale * d).max(MIN_ALLOWED_PENETRATION_DEPTH)
}

// ------------------------------------------------------------------------------------------
// The rigid body and its per-PSI integration
// ------------------------------------------------------------------------------------------

/// A Havok rigid body's dynamic state and the per-body constants the integrator reads.
#[derive(Clone, Debug)]
pub struct HavokBody {
    pub motion: MotionType,
    /// Centre of mass, world.
    pub com: Vec3,
    pub q: Quat,
    pub v: Vec3,
    pub w: Vec3,
    pub inv_mass: f32,
    /// Inverse inertia tensor in the body frame.
    pub inv_inertia_local: Mat3,
    /// `motionState.linearDamping` / `angularDamping` (1/s), from the phmo rigid body `+0x30/+0x34`.
    pub lin_damping: f32,
    pub ang_damping: f32,
    max_lin_code: u8,
    max_ang_code: u8,
    gravity_factor_bits: u16,
}

impl HavokBody {
    /// A body at rest with the cinfo-ctor defaults: no damping, [`DEFAULT_MAX_VELOCITY`] caps,
    /// gravity factor 1.0.
    pub fn new(motion: MotionType, mass: f32, inv_inertia_local: Mat3, com: Vec3, q: Quat) -> Self {
        HavokBody {
            motion,
            com,
            q: q.normalize(),
            v: Vec3::ZERO,
            w: Vec3::ZERO,
            inv_mass: if mass > 0.0 { 1.0 / mass } else { 0.0 },
            inv_inertia_local,
            lin_damping: 0.0,
            ang_damping: 0.0,
            max_lin_code: ufloat8_encode(DEFAULT_MAX_VELOCITY),
            max_ang_code: ufloat8_encode(DEFAULT_MAX_VELOCITY),
            gravity_factor_bits: gravity_factor_bits(1.0),
        }
    }

    pub fn with_damping(mut self, linear: f32, angular: f32) -> Self {
        self.lin_damping = linear;
        self.ang_damping = angular;
        self
    }

    /// Set both velocity caps; they are stored as hkUFloat8, so the value in force is the table
    /// entry at or above the one asked for.
    pub fn with_max_velocities(mut self, linear: f32, angular: f32) -> Self {
        self.max_lin_code = ufloat8_encode(linear);
        self.max_ang_code = ufloat8_encode(angular);
        self
    }

    /// The linear velocity cap in force, wu/s.
    pub fn max_linear_velocity(&self) -> f32 {
        ufloat8_decode(self.max_lin_code)
    }
    /// The angular velocity cap in force, rad/s (see the module note for the per-step angle limit
    /// that usually binds first).
    pub fn max_angular_velocity(&self) -> f32 {
        ufloat8_decode(self.max_ang_code)
    }

    /// Set the gravity factor; stored truncated to a half ([`gravity_factor_bits`]).
    pub fn set_gravity_factor(&mut self, f: f32) {
        self.gravity_factor_bits = gravity_factor_bits(f);
    }
    /// The gravity factor in force (the truncated value).
    pub fn gravity_factor(&self) -> f32 {
        gravity_factor_from_bits(self.gravity_factor_bits)
    }

    /// What the vehicle code does to the body each tick: replace both velocities
    /// (`setLinearVelocity` / `setAngularVelocity`, `sub_1406CEB20`).
    pub fn set_velocity(&mut self, v: Vec3, w: Vec3) {
        self.v = v;
        self.w = w;
    }

    /// Inverse inertia in the world frame.
    pub fn inv_inertia_world(&self) -> Mat3 {
        let r = Mat3::from_quat(self.q);
        r * self.inv_inertia_local * r.transpose()
    }

    /// Apply an impulse `j` at world point `p`.
    pub fn apply_impulse_at(&mut self, p: Vec3, j: Vec3) {
        self.v += j * self.inv_mass;
        self.w += self.inv_inertia_world() * (p - self.com).cross(j);
    }

    /// Velocity of the body's material point at world position `p`.
    pub fn point_velocity(&self, p: Vec3) -> Vec3 {
        self.v + self.w.cross(p - self.com)
    }

    /// One Havok PSI of the integrator (`sub_141106780`), in the engine's order. See the module
    /// note for the equations. `gravity` is the world gravity vector ([`world_gravity`]).
    pub fn step_psi(&mut self, dt: f32, gravity: Vec3) {
        match self.motion {
            MotionType::Fixed => return,
            MotionType::Keyframed => {}
            MotionType::Character => self.damp(dt),
            _ => {
                self.v += gravity * dt * self.gravity_factor();
                self.damp(dt);
            }
        }
        let bad = |x: Vec3| !(x.x.abs() < 1.0e6 && x.y.abs() < 1.0e6 && x.z.abs() < 1.0e6);
        if bad(self.v) || bad(self.w) {
            self.v = Vec3::ZERO;
            self.w = Vec3::ZERO;
        }
        let cap = self.max_linear_velocity();
        let speed2 = self.v.length_squared();
        if speed2 > cap * cap {
            self.v *= cap / speed2.sqrt();
        }
        self.com += self.v * dt;

        let mut h = self.w * (dt * 0.5);
        let step_cap = (dt * self.max_angular_velocity()).min(0.9);
        let mut s = h.length_squared() * 0.405_284_703;
        if step_cap * step_cap < s {
            let k = step_cap / s.sqrt();
            self.w *= k;
            h *= k;
            s = step_cap * step_cap;
        }
        let wq = 1.0 - s * 0.822_947_979 - s * s * 0.130_529_001 - s * 0.044_408_001 * s * s;
        let dq = Quat::from_xyzw(h.x, h.y, h.z, wq);
        let q = (dq * self.q).normalize();
        if q.is_finite() {
            self.q = q;
        }
    }

    fn damp(&mut self, dt: f32) {
        self.v *= (1.0 - dt * self.lin_damping).max(0.0);
        self.w *= (1.0 - dt * self.ang_damping).max(0.0);
    }
}

// ------------------------------------------------------------------------------------------
// Contacts
// ------------------------------------------------------------------------------------------

/// One contact between the body and something else.
#[derive(Clone, Copy, Debug)]
pub struct ContactPoint {
    /// World position of the contact.
    pub point: Vec3,
    /// Unit normal pointing from the other surface INTO the body.
    pub normal: Vec3,
    /// Signed gap along the normal: negative = penetration, positive = a speculative contact
    /// (Havok generates contacts out to [`COLLISION_TOLERANCE`]) that the body may close at
    /// `separation / dt` and no faster. A body that crossed a surface inside one PSI should be
    /// rewound to the time of impact first ([`time_of_impact_fraction`]) and its contact taken
    /// there with `separation == 0`: that is what the engine's continuous simulation does, and
    /// it is what lets a fast impact keep its full closing velocity for the restitution test.
    pub separation: f32,
    /// Already-combined, already-stored contact material: see [`contact_material`].
    pub friction: f32,
    pub restitution: f32,
}

/// The solver's settings.
#[derive(Clone, Copy, Debug)]
pub struct SolverParams {
    /// Fraction of a penetration removed per step ([`SOLVER_TAU`]). INFERRED use.
    pub tau: f32,
    pub iterations: u32,
    /// Restitution is applied only to contacts closing faster than this.
    pub resting_velocity: f32,
}

impl SolverParams {
    /// The engine's world settings.
    pub const ENGINE: SolverParams = SolverParams {
        tau: SOLVER_TAU,
        iterations: SOLVER_ITERATIONS,
        resting_velocity: CONTACT_RESTING_VELOCITY,
    };
}

/// Resolve `contacts` against `body` by sequential impulses: a non-negative normal impulse that
/// removes the closing velocity (plus restitution when the contact closed faster than
/// `resting_velocity`, plus a `tau` fraction of any penetration), then Coulomb friction within
/// `mu * normal impulse`. **INFERRED**: the engine's soft-contact solver is not lifted; this is the
/// behaviour it must have at rest and at impact, not a port of its equations. It never moves
/// the position -- Havok's solver changes velocities and the next [`HavokBody::step_psi`] moves.
pub fn solve_contacts(body: &mut HavokBody, contacts: &[ContactPoint], dt: f32, p: &SolverParams) {
    if body.inv_mass <= 0.0 || contacts.is_empty() || dt <= 0.0 {
        return;
    }
    struct Row {
        c: ContactPoint,
        r: Vec3,
        t1: Vec3,
        t2: Vec3,
        target: f32,
        acc_n: f32,
        acc_t: (f32, f32),
    }
    let mut rows: Vec<Row> = contacts
        .iter()
        .map(|c| {
            let n = c.normal.normalize_or_zero();
            let r = c.point - body.com;
            let vn0 = body.point_velocity(c.point).dot(n);
            let closing = -vn0;
            let bias = if c.separation < 0.0 {
                p.tau * (-c.separation) / dt
            } else {
                -c.separation / dt
            };
            // an impact: already touching, or reaching the surface within this step
            let reaches = c.separation <= 0.0 || closing * dt >= c.separation;
            let bounce = if reaches && closing > p.resting_velocity { c.restitution * closing } else { 0.0 };
            let (t1, t2) = tangent_basis(n);
            Row { c: ContactPoint { normal: n, ..*c }, r, t1, t2, target: bias.max(bounce), acc_n: 0.0, acc_t: (0.0, 0.0) }
        })
        .collect();
    for _ in 0..p.iterations.max(1) {
        for row in rows.iter_mut() {
            let n = row.c.normal;
            let u = body.point_velocity(row.c.point);
            // normal row
            let kn = body.inv_mass + n.dot((body.inv_inertia_world() * row.r.cross(n)).cross(row.r));
            if kn > 0.0 {
                let lam = (row.target - u.dot(n)) / kn;
                let new = (row.acc_n + lam).max(0.0);
                let d = new - row.acc_n;
                row.acc_n = new;
                body.apply_impulse_at(row.c.point, n * d);
            }
            // friction rows
            let u = body.point_velocity(row.c.point);
            let mut acc = row.acc_t;
            for (k, t) in [(0usize, row.t1), (1usize, row.t2)] {
                let kt = body.inv_mass + t.dot((body.inv_inertia_world() * row.r.cross(t)).cross(row.r));
                if kt <= 0.0 {
                    continue;
                }
                let lam = -u.dot(t) / kt;
                if k == 0 { acc.0 += lam } else { acc.1 += lam }
                let lim = row.c.friction * row.acc_n;
                let mag = (acc.0 * acc.0 + acc.1 * acc.1).sqrt();
                if mag > lim && mag > 0.0 {
                    let s = lim / mag;
                    acc.0 *= s;
                    acc.1 *= s;
                }
                let prev = if k == 0 { row.acc_t.0 } else { row.acc_t.1 };
                let now = if k == 0 { acc.0 } else { acc.1 };
                body.apply_impulse_at(row.c.point, t * (now - prev));
                if k == 0 { row.acc_t.0 = acc.0 } else { row.acc_t.1 = acc.1 }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// The anti-gravity (hover) point law
// ------------------------------------------------------------------------------------------

/// One `anti_gravity_point_definition_block` element as the force law needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoverLaw {
    /// `strength`: the point's share of the vehicle's weight at full compression.
    pub strength: f32,
    /// `height`: how far below the point the pad reaches, past its radius.
    pub height: f32,
    pub radius: f32,
    /// `compression damping` / `extension damping`: used while the point approaches / leaves the
    /// surface.
    pub compression_damping: f32,
    pub extension_damping: f32,
    /// The two thresholds on the HULL's up-vector z that scale the lift: 0 at or below the lower,
    /// 1 at or above the upper, linear between (an upside-down hull gets no lift). The block
    /// stores the upper one first.
    pub normal_k_hi: f32,
    pub normal_k_lo: f32,
}

/// The lift scale from the hull's tilt (`up_z` = the z of its up vector), exactly the piecewise
/// ramp of `sub_1406CC8C0`: both orderings of the two authored thresholds are handled as the
/// decompile does.
pub fn hover_tilt_scale(k_hi: f32, k_lo: f32, up_z: f32) -> f32 {
    let (a, b, t) = (k_hi, k_lo, up_z);
    if (a - b).abs() < 0.001 {
        return if b < t { 1.0 } else { 0.0 };
    }
    if a <= b {
        if a >= t { 1.0 } else if t < b { (b - t) / (b - a) } else { 0.0 }
    } else if b < t {
        if t >= a { 1.0 } else { (t - b) / (a - b) }
    } else {
        0.0
    }
}

/// The ACCELERATION one hover point applies along the surface normal (wu/s^2), from
/// `sub_1406CC8C0` (PROVEN): `strength * tilt_scale * (g * c^2 - damping * v_n)`, where `c` is the
/// compression fraction `1 - (distance - radius) / height` (1 at or inside the radius, 0 at full
/// reach) and `v_n` is the point's velocity along the normal (the compression damping while it
/// approaches, the extension damping while it leaves). The engine multiplies by the body's mass
/// to get a force and divides it back out; the acceleration is what is returned. `distance` is from
/// the point to the surface along the ray; no surface within `height + radius` -> no force.
pub fn hover_acceleration(law: &HoverLaw, g: f32, distance: Option<f32>, normal_speed: f32, up_z: f32) -> f32 {
    let Some(d) = distance else { return 0.0 };
    if d > law.height + law.radius {
        return 0.0;
    }
    let beyond = d - law.radius;
    let c = if beyond > 0.0 && law.height > 0.0 { 1.0 - beyond / law.height } else { 1.0 };
    let damping = if normal_speed < 0.0 { law.compression_damping } else { law.extension_damping };
    law.strength * hover_tilt_scale(law.normal_k_hi, law.normal_k_lo, up_z) * (g * c * c - damping * normal_speed)
}

/// The authored braking an unpiloted `alien_scout` hover vehicle applies (`sub_140E840D0` run with
/// zero input).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IdleBrake {
    /// `speed deceleration` (forward), `slide deceleration` (sideways), wu/s^2.
    pub speed_decel: f32,
    pub slide_decel: f32,
    /// `drag coeficient` and `constant deceleration`.
    pub drag: f32,
    pub constant_decel: f32,
}

/// The horizontal acceleration `(forward, left)` an empty hover vehicle's velocity controller
/// applies: with no input the target velocity is zero, so the wanted acceleration is
/// `-v / tick` (PROVEN shape; the divisor is `1 / (a scalar the decompile calls
/// object_get_orientation)` and is INFERRED to be the tick length), clamped in magnitude to the
/// authored deceleration blended by direction: `speed_decel * f + slide_decel * (1 - f)` with
/// `f = |pi/2 - |atan2(left, forward)|| / (pi/2)` (PROVEN). `(v_fwd, v_left)` are the velocity
/// along the hull's own forward and left axes.
pub fn hover_idle_brake(b: &IdleBrake, v_fwd: f32, v_left: f32, tick: f32) -> (f32, f32) {
    if v_fwd == 0.0 && v_left == 0.0 || tick <= 0.0 {
        return (0.0, 0.0);
    }
    let (wf, wl) = (-v_fwd / tick, -v_left / tick);
    let half_pi = std::f32::consts::FRAC_PI_2;
    let f = ((half_pi - wl.atan2(wf).abs()) / half_pi).abs().min(1.0);
    let cap = b.speed_decel * f + b.slide_decel * (1.0 - f);
    let m = (wf * wf + wl * wl).sqrt();
    if m > cap { (wf * cap / m, wl * cap / m) } else { (wf, wl) }
}

/// The drag acceleration magnitude along `-v` for speed `speed`: `(speed * drag)^2` plus the
/// constant deceleration, which is limited to what the drag leaves of it and to what stops the
/// vehicle in one tick (`sub_140E840D0`; PROVEN structure, the `speed / tick` limit is the
/// INFERRED reading of `sub_1400BD300`).
pub fn hover_idle_drag(b: &IdleBrake, speed: f32, tick: f32) -> f32 {
    let quad = (speed * b.drag) * (speed * b.drag);
    quad + (b.constant_decel - quad).min(speed / tick.max(1e-6)).max(0.0)
}

/// Where a hull of total lift `sum_strength` (the sum of its points' strengths, upright) comes to
/// rest: the compression fraction solves `sum_strength * c^2 = gravity_factor` hence
/// `c = sqrt(gravity_factor / sum_strength)`; the pad then stands `radius + height * (1 - c)` above
/// the surface. `None` when the strengths cannot hold the hull up (`c > 1`).
pub fn hover_rest_clearance(sum_strength: f32, height: f32, radius: f32, gravity_factor: f32) -> Option<f32> {
    if sum_strength <= 0.0 || gravity_factor <= 0.0 {
        return None;
    }
    let c = (gravity_factor / sum_strength).sqrt();
    (c <= 1.0).then(|| radius + height * (1.0 - c))
}

/// One contact between two bodies of a slice (or one body and the static world).
#[derive(Clone, Copy, Debug)]
pub struct PairContact {
    pub a: usize,
    /// `None` = the static world.
    pub b: Option<usize>,
    pub point: Vec3,
    /// Unit normal from `b` INTO `a`.
    pub normal: Vec3,
    /// Signed gap (negative = penetration), as in [`ContactPoint`].
    pub separation: f32,
    pub friction: f32,
    pub restitution: f32,
}

/// The engine's contact mass clamp (`havok_utilities.cpp:365`, `sub_1807FFED0` in the debug
/// build; no caller is lifted): given the two bodies' masses it returns the factors
/// `(mass_scale_a, mass_scale_b)`. **PROVEN** formula, with `r = mA / mB`: both stay 1 when either
/// mass is under 0.001 or `r <= 0.001`; `scale_a = 16 r` when `r < 1/16`; `scale_a = 16 / r` when
/// `r > 16`; `scale_b` is always 1.
///
/// **INFERRED** reading: the factors belong to a mass-changer modifier atom, which in Havok
/// multiplies the INVERSE mass of its body. [`mass_multipliers`] turns the clamp into the one
/// property every reading shares -- the effective mass ratio of a contact never exceeds 16:1 --
/// by lowering the heavy body's mass, applied symmetrically.
pub fn mass_scale_factors(mass_a: f32, mass_b: f32) -> (f32, f32) {
    if mass_a < 0.001 || mass_b < 0.001 {
        return (1.0, 1.0);
    }
    let r = mass_a / mass_b;
    if r <= 0.001 {
        (1.0, 1.0)
    } else if r < 0.0625 {
        (16.0 * r, 1.0)
    } else if r > 16.0 {
        (16.0 / r, 1.0)
    } else {
        (1.0, 1.0)
    }
}

/// MASS multipliers `(a, b)` for one contact: the heavier body is lowered so the pair's effective
/// mass ratio is at most 16:1. See [`mass_scale_factors`] for what is proven and what is not.
pub fn mass_multipliers(mass_a: f32, mass_b: f32) -> (f32, f32) {
    if mass_a < 0.001 || mass_b < 0.001 {
        return (1.0, 1.0);
    }
    let r = mass_a / mass_b;
    if r > 16.0 {
        (16.0 / r, 1.0)
    } else if r < 1.0 / 16.0 {
        (1.0, 16.0 * r)
    } else {
        (1.0, 1.0)
    }
}

/// [`solve_contacts`] for contacts BETWEEN bodies of one slice (and against the static world):
/// the same rows -- a non-negative normal impulse with the speculative / penetration / restitution
/// target, then two-tangent Coulomb friction -- but applied to both bodies, with the contact mass
/// clamp of [`mass_multipliers`]. A body with `inv_mass == 0` is immovable. A caller that wants a
/// KINEMATIC pusher (the player, a driven vehicle) gives it its real mass and restores its
/// velocity after the solve: the impulse it exerted stays in the body it pushed.
pub fn solve_pair_contacts(bodies: &mut [HavokBody], contacts: &[PairContact], dt: f32, p: &SolverParams) {
    if contacts.is_empty() || dt <= 0.0 {
        return;
    }
    struct Row {
        c: PairContact,
        ra: Vec3,
        rb: Vec3,
        t1: Vec3,
        t2: Vec3,
        target: f32,
        sa: f32,
        sb: f32,
        acc_n: f32,
        acc_t: (f32, f32),
    }
    let iw: Vec<Mat3> = bodies.iter().map(|b| b.inv_inertia_world()).collect();
    let mut rows: Vec<Row> = contacts
        .iter()
        .map(|c| {
            let n = c.normal.normalize_or_zero();
            let ra = c.point - bodies[c.a].com;
            let (rb, ub) = match c.b {
                Some(b) => (c.point - bodies[b].com, bodies[b].point_velocity(c.point)),
                None => (Vec3::ZERO, Vec3::ZERO),
            };
            let (sa, sb) = match c.b {
                Some(b) if bodies[c.a].inv_mass > 0.0 && bodies[b].inv_mass > 0.0 => {
                    mass_multipliers(1.0 / bodies[c.a].inv_mass, 1.0 / bodies[b].inv_mass)
                }
                _ => (1.0, 1.0),
            };
            let closing = -((bodies[c.a].point_velocity(c.point) - ub).dot(n));
            let bias = if c.separation < 0.0 {
                p.tau * (-c.separation) / dt
            } else {
                -c.separation / dt
            };
            let reaches = c.separation <= 0.0 || closing * dt >= c.separation;
            let bounce = if reaches && closing > p.resting_velocity { c.restitution * closing } else { 0.0 };
            let (t1, t2) = tangent_basis(n);
            Row { c: PairContact { normal: n, ..*c }, ra, rb, t1, t2, target: bias.max(bounce), sa, sb, acc_n: 0.0, acc_t: (0.0, 0.0) }
        })
        .collect();

    // effective inverse mass of the pair along `d`, and the impulse application, both with the
    // mass multipliers folded in
    fn k(bodies: &[HavokBody], iw: &[Mat3], r: &Row, d: Vec3) -> f32 {
        let mut k = 0.0;
        let a = &bodies[r.c.a];
        if a.inv_mass > 0.0 {
            k += a.inv_mass / r.sa + d.dot((iw[r.c.a] * r.ra.cross(d)).cross(r.ra)) / r.sa;
        }
        if let Some(bi) = r.c.b {
            let b = &bodies[bi];
            if b.inv_mass > 0.0 {
                k += b.inv_mass / r.sb + d.dot((iw[bi] * r.rb.cross(d)).cross(r.rb)) / r.sb;
            }
        }
        k
    }
    fn apply(bodies: &mut [HavokBody], iw: &[Mat3], r: &Row, imp: Vec3) {
        let a = &mut bodies[r.c.a];
        if a.inv_mass > 0.0 {
            a.v += imp * (a.inv_mass / r.sa);
            a.w += iw[r.c.a] * r.ra.cross(imp) / r.sa;
        }
        if let Some(bi) = r.c.b {
            let b = &mut bodies[bi];
            if b.inv_mass > 0.0 {
                b.v -= imp * (b.inv_mass / r.sb);
                b.w -= iw[bi] * r.rb.cross(imp) / r.sb;
            }
        }
    }
    fn rel_vel(bodies: &[HavokBody], r: &Row) -> Vec3 {
        let ua = bodies[r.c.a].point_velocity(r.c.point);
        let ub = r.c.b.map_or(Vec3::ZERO, |b| bodies[b].point_velocity(r.c.point));
        ua - ub
    }

    let nrows = rows.len();
    for pass in 0..p.iterations.max(1) {
        for step in 0..nrows {
            // alternate the sweep direction: a one-way Gauss-Seidel sweep walks a settled stack
            // sideways (every row sees the others' impulses in the same order each time)
            let ri = if pass % 2 == 0 { step } else { nrows - 1 - step };
            let n = rows[ri].c.normal;
            let kn = k(bodies, &iw, &rows[ri], n);
            if kn > 0.0 {
                let u = rel_vel(bodies, &rows[ri]);
                let lam = (rows[ri].target - u.dot(n)) / kn;
                let new = (rows[ri].acc_n + lam).max(0.0);
                let d = new - rows[ri].acc_n;
                rows[ri].acc_n = new;
                apply(bodies, &iw, &rows[ri], n * d);
            }
            let (t1, t2) = (rows[ri].t1, rows[ri].t2);
            let mut acc = rows[ri].acc_t;
            for (kx, t) in [(0usize, t1), (1usize, t2)] {
                let kt = k(bodies, &iw, &rows[ri], t);
                if kt <= 0.0 {
                    continue;
                }
                let u = rel_vel(bodies, &rows[ri]);
                let lam = -u.dot(t) / kt;
                let before = acc;
                if kx == 0 { acc.0 += lam } else { acc.1 += lam }
                let lim = rows[ri].c.friction * rows[ri].acc_n;
                let mag = (acc.0 * acc.0 + acc.1 * acc.1).sqrt();
                if mag > lim && mag > 0.0 {
                    let sc = lim / mag;
                    acc.0 *= sc;
                    acc.1 *= sc;
                }
                // BOTH tangents' accumulated impulses may have changed (the cone clamp scales
                // the pair), so both deltas are applied
                apply(bodies, &iw, &rows[ri], t1 * (acc.0 - before.0) + t2 * (acc.1 - before.1));
            }
            rows[ri].acc_t = acc;
        }
    }
}

/// Where in a PSI a body that moved its gap along a contact normal from `gap_before >= 0` to
/// `gap_after < 0` first touched: the fraction `t` in `[0, 1]` of the step at which the gap was 0,
/// assuming straight-line motion. `None` when it did not cross. The continuous simulation's
/// time-of-impact pass, reduced to the one number a caller needs to rewind with.
pub fn time_of_impact_fraction(gap_before: f32, gap_after: f32) -> Option<f32> {
    if gap_before >= 0.0 && gap_after < 0.0 {
        Some((gap_before / (gap_before - gap_after)).clamp(0.0, 1.0))
    } else {
        None
    }
}

fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let t1 = n.cross(a).normalize_or_zero();
    (t1, n.cross(t1))
}

// ------------------------------------------------------------------------------------------
// Gates
// ------------------------------------------------------------------------------------------

#[cfg(test)]
mod gates {
    use super::*;

    const G: f32 = crate::reach_physics::WORLD_GRAVITY;

    fn point_mass(com: Vec3) -> HavokBody {
        HavokBody::new(MotionType::Dynamic, 10.0, Mat3::ZERO, com, Quat::IDENTITY)
    }

    /// The world constants against the IMMEDIATES in the decompiled cinfo setup, written as raw
    /// bit patterns so a rounded literal in the source cannot satisfy them.
    #[test]
    fn world_settings_are_the_decompiled_immediates() {
        assert_eq!(SOLVER_TAU.to_bits(), 0x3E99_999A, "sub_1808A6F80 a1[34]");
        assert_eq!(SOLVER_DAMP.to_bits(), 0x3F66_6666, "sub_1808A6F80 a1[35]");
        assert_eq!(SOLVER_ITERATIONS, 8, "sub_1808A6F80 a1[36]");
        assert_eq!(SOLVER_MICROSTEPS, 1, "sub_1808A6DA0 a1[37]");
        assert_eq!(COLLISION_TOLERANCE.to_bits(), 0x3D06_594B, "sub_18009A06C 0.0328000002");
        assert_eq!(EXPECTED_MAX_LINEAR_VELOCITY.to_bits(), 0x4270_0000, "sub_18009A06C 60.0");
        assert_eq!(CONTACT_RESTING_VELOCITY.to_bits(), 0x7F7F_FFFF, "sub_18009A06C a1[9] FLT_MAX");
        assert_eq!(ITERATIVE_LINEAR_CAST_EARLY_OUT.to_bits(), 0x3B56_F545, "sub_18009A06C 0.00328");
        assert_eq!(DEFAULT_MAX_VELOCITY.to_bits(), 0x4348_0000, "sub_1410D7CB0 a1[46]/a1[47]");
        assert_eq!(TICK_HZ, 60.0);
        assert_eq!(deactivation_reference_distance(60), 0.05, "0.1 * 1/(60/30)");
        assert_eq!(deactivation_reference_distance(30), 0.1);
        assert_eq!(world_gravity().z, -4.171_259_4);
    }

    /// Structure of the hkUFloat8 table plus anchors read straight off the exe.
    #[test]
    fn the_ufloat8_table_has_the_engines_shape() {
        assert!((1..256).all(|i| ufloat8_decode(i as u8) > ufloat8_decode(i as u8 - 1)), "strictly increasing");
        assert_eq!(ufloat8_decode(0), 0.0);
        assert_eq!(ufloat8_decode(1), 0.01);
        assert_eq!(ufloat8_decode(8), f32::from_bits(0x3D99_999A)); // 0.075
        assert_eq!(ufloat8_decode(127), 202.8);
        assert_eq!(ufloat8_decode(128), 216.8);
        assert_eq!(ufloat8_decode(255), 1_000_002.0);
        // the encoder's own thresholds: 0x3C23D70A (0.01) and 0x4358CCCD (216.8) split the search
        assert_eq!(ufloat8_encode(0.009), 0);
        assert_eq!(ufloat8_encode(-5.0), 0);
        assert_eq!(ufloat8_encode(0.01), 1);
        assert_eq!(ufloat8_encode(200.0), 127, "ceil: 200 -> the 202.8 entry");
        assert_eq!(quantised_max_velocity(DEFAULT_MAX_VELOCITY), 202.8);
        assert_eq!(ufloat8_encode(1.0e9), 255);
        // ceil, entry by entry: an entry encodes to itself and one ulp above encodes to the next
        for i in 1..255u8 {
            let t = ufloat8_decode(i);
            assert_eq!(ufloat8_encode(t), i);
            assert_eq!(ufloat8_encode(f32::from_bits(t.to_bits() + 1)), i + 1);
        }
    }

    /// The table in the source against the bytes of the real tool exe, when it is on this machine.
    #[test]
    #[ignore = "reads ~/repos/reverseme/bins/reach_tag_test.exe"]
    fn the_ufloat8_table_matches_the_exe() {
        let home = std::env::var("HOME").unwrap_or_default();
        let path = format!("{home}/repos/reverseme/bins/reach_tag_test.exe");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("SKIPPED: no {path}");
            return;
        };
        let at = 0x1A9_B8C0usize;
        for i in 0..256 {
            let o = at + i * 4;
            let bits = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            assert_eq!(bits, UFLOAT8_BITS[i], "entry {i}");
        }
        println!("ufloat8 table: 256/256 entries match the exe");
    }

    /// Free fall: after N ticks from rest the integrator (velocity first, then position with the
    /// NEW velocity) has fallen `g dt^2 N(N+1)/2`, a closed form stated independently of the loop.
    #[test]
    fn free_fall_follows_the_symplectic_series_and_the_gravity_factor_scales_it() {
        let n = 60u32;
        let series = (n * (n + 1)) as f32 / 2.0;
        let mut b = point_mass(Vec3::new(0.0, 0.0, 100.0));
        for _ in 0..n {
            b.step_psi(TICK_SECONDS, world_gravity());
        }
        let want = 100.0 - G * TICK_SECONDS * TICK_SECONDS * series;
        assert!((b.com.z - want).abs() < 2e-4, "fell to {} want {want}", b.com.z);
        assert!((b.v.z + G * TICK_SECONDS * n as f32).abs() < 1e-4, "v = {}", b.v.z);
        // half gravity falls half as far
        let mut h = point_mass(Vec3::new(0.0, 0.0, 100.0));
        h.set_gravity_factor(0.5);
        for _ in 0..n {
            h.step_psi(TICK_SECONDS, world_gravity());
        }
        assert!(((100.0 - h.com.z) / (100.0 - b.com.z) - 0.5).abs() < 1e-5, "ratio {}", (100.0 - h.com.z) / (100.0 - b.com.z));
        // zero gravity does not move it
        let mut z = point_mass(Vec3::new(0.0, 0.0, 100.0));
        z.set_gravity_factor(0.0);
        for _ in 0..n {
            z.step_psi(TICK_SECONDS, world_gravity());
        }
        assert_eq!(z.com.z, 100.0);
    }

    #[test]
    fn the_gravity_factor_is_a_truncated_half() {
        assert_eq!(gravity_factor_bits(1.0), 0x3F80);
        assert_eq!(gravity_factor_bits(0.5), 0x3F00);
        assert_eq!(gravity_factor_bits(0.3), 0x3E99, "0x3E99999A truncated, NOT rounded to 0x3E9A");
        let mut b = point_mass(Vec3::ZERO);
        b.set_gravity_factor(0.3);
        assert_eq!(b.gravity_factor(), 0.298_828_125);
    }

    /// Damping is `v *= max(0, 1 - dt d)` -- with LITERAL expectations, including the clamp that
    /// the implicit `v / (1 + d h)` form (which `ragdoll.rs` uses) does not have.
    #[test]
    fn damping_is_one_minus_dt_d_clamped_at_zero() {
        let mut b = point_mass(Vec3::ZERO).with_damping(0.6, 0.0);
        b.set_gravity_factor(0.0);
        b.v = Vec3::new(10.0, 0.0, 0.0);
        b.step_psi(1.0 / 60.0, world_gravity());
        assert!((b.v.x - 9.9).abs() < 1e-5, "10 * (1 - 0.6/60) = 9.9, got {}", b.v.x);
        for _ in 0..59 {
            b.step_psi(1.0 / 60.0, world_gravity());
        }
        // (0.99)^60 = 0.547157...
        assert!((b.v.x - 5.4716).abs() < 2e-3, "after one second {}", b.v.x);
        // d * dt > 1: the velocity is zeroed, never reversed
        let mut stiff = point_mass(Vec3::ZERO).with_damping(100.0, 100.0);
        stiff.v = Vec3::new(5.0, 0.0, 0.0);
        stiff.w = Vec3::new(0.0, 0.0, 5.0);
        stiff.step_psi(1.0 / 60.0, Vec3::ZERO);
        assert_eq!(stiff.v, Vec3::ZERO);
        assert_eq!(stiff.w, Vec3::ZERO);
        // and the engine's form really differs from the implicit one at this strength
        let implicit = 5.0 / (1.0 + 100.0 / 60.0);
        assert!(implicit > 1.0, "implicit damping would leave {implicit}");
    }

    #[test]
    fn character_motion_takes_damping_but_not_gravity_and_fixed_does_not_move() {
        let mut c = HavokBody::new(MotionType::Character, 80.0, Mat3::ZERO, Vec3::ZERO, Quat::IDENTITY).with_damping(1.0, 0.0);
        c.v = Vec3::new(3.0, 0.0, 0.0);
        c.step_psi(TICK_SECONDS, world_gravity());
        assert_eq!(c.v.z, 0.0, "no gravity on a character-motion body");
        assert!(c.v.x < 3.0 && c.v.x > 2.9);
        let mut f = HavokBody::new(MotionType::Fixed, 0.0, Mat3::ZERO, Vec3::ONE, Quat::IDENTITY);
        f.v = Vec3::X;
        f.step_psi(TICK_SECONDS, world_gravity());
        assert_eq!(f.com, Vec3::ONE);
    }

    /// The velocity cap is the hkUFloat8 entry: 200 asked for, 202.8 in force.
    #[test]
    fn the_velocity_cap_is_the_quantised_table_entry() {
        let mut b = point_mass(Vec3::ZERO);
        b.set_gravity_factor(0.0);
        b.v = Vec3::new(500.0, 0.0, 0.0);
        b.step_psi(TICK_SECONDS, world_gravity());
        assert!((b.v.x - 202.8).abs() < 1e-3, "capped to {}", b.v.x);
        let mut slow = point_mass(Vec3::ZERO).with_max_velocities(10.0, 10.0);
        slow.set_gravity_factor(0.0);
        slow.v = Vec3::new(0.0, 100.0, 0.0);
        slow.step_psi(TICK_SECONDS, world_gravity());
        assert!((slow.v.y - quantised_max_velocity(10.0)).abs() < 1e-4 && slow.v.y >= 10.0 && slow.v.y < 11.0, "got {}", slow.v.y);
    }

    /// Rotation: a modest spin turns by w*dt (the polynomial is within 0.1% of the exact angle);
    /// an absurd spin is limited to 0.9*pi per step.
    #[test]
    fn rotation_advances_by_w_dt_and_is_capped_per_step() {
        let mut b = point_mass(Vec3::ZERO);
        b.set_gravity_factor(0.0);
        b.w = Vec3::new(0.0, 0.0, 100.0);
        b.step_psi(TICK_SECONDS, world_gravity());
        let (_axis, ang) = b.q.to_axis_angle();
        let want = 100.0 * TICK_SECONDS;
        assert!((ang - want).abs() < 0.002, "turned {ang} want {want}");
        let mut fast = point_mass(Vec3::ZERO);
        fast.set_gravity_factor(0.0);
        fast.w = Vec3::new(0.0, 0.0, 1000.0);
        fast.step_psi(TICK_SECONDS, world_gravity());
        let (_a, ang) = fast.q.to_axis_angle();
        let lim = 0.9 * std::f32::consts::PI;
        assert!((ang - lim).abs() < 0.01, "turned {ang}, the engine's cap is 0.9 pi = {lim}");
        assert!((fast.w.z - lim / TICK_SECONDS).abs() < 0.5, "w stored back as {}", fast.w.z);
    }

    #[test]
    fn friction_and_restitution_combine_as_the_geometric_mean() {
        assert_eq!(combined_friction(0.25, 1.0), 0.5);
        assert_eq!(combined_friction(1.0, 0.25), combined_friction(0.25, 1.0), "commutative");
        assert_eq!(combined_friction(0.0, 7.0), 0.0);
        assert_eq!(combined_restitution(0.64, 0.25), 0.4);
        // stored forms: friction quantised UP to the table (0.0748 -> 0.075), restitution
        // truncated to 1/128 (0.4 -> 51/128 = 0.3984375)
        let (f, e) = contact_material(0.07, 0.64, 0.08, 0.25);
        assert_eq!(f, f32::from_bits(0x3D99_999A), "sqrt(0.07*0.08) = 0.0748 -> 0.075");
        assert_eq!(e, 51.0 / 128.0);
        // truncation, not rounding: sqrt(0.7 * 0.5) = 0.5916 -> 75.7/128 -> 75, not 76
        assert_eq!(contact_material(1.0, 0.7, 1.0, 0.5).1, 75.0 / 128.0);
        // the arithmetic mean of the pair would have been 0.445 (-> 56/128): not what the engine does
        assert_ne!(contact_restitution_stored((0.64 + 0.25) * 0.5), e);
        assert_eq!(contact_friction_written(0.5, true, false), 0.0);
        assert_eq!(contact_friction_written(0.5, false, true), 0.0);
        assert_eq!(contact_friction_written(0.5, false, false), 0.5);
    }

    #[test]
    fn the_vehicle_gravity_scale_postprocess_has_the_three_cases() {
        assert_eq!(vehicle_gravity_scale_postprocessed(0.0), 1.0, "0 means 1");
        assert_eq!(vehicle_gravity_scale_postprocessed(-2.0), 1.0);
        assert_eq!(vehicle_gravity_scale_postprocessed(0.0005), 0.0, "a tiny positive means none");
        assert_eq!(vehicle_gravity_scale_postprocessed(0.5), 0.5);
        assert_eq!(vehicle_gravity_scale_postprocessed(1.5), 1.5);
        // only when the vehicle's +0x3A0 float is non-zero is its scale multiplied in
        assert_eq!(vehicle_body_gravity_factor(1.0, 1.0, 0.5), 0.5);
        assert_eq!(vehicle_body_gravity_factor(1.0, 0.0, 0.5), 1.0);
        assert_eq!(vehicle_body_gravity_factor(2.0, 1.0, 0.5), 1.0);
        assert_eq!(vehicle_instance_mass(0.2), 1.0);
        assert_eq!(vehicle_instance_mass(900.0), 900.0);
    }

    #[test]
    fn quality_types_follow_the_creation_rules() {
        use object_type::*;
        let q = |m, t, layer, f25, bf| body_quality_type(m, t, layer, f25, bf);
        assert_eq!(q(MotionType::BoxInertia, VEHICLE, 0xF, false, false), QualityType::Critical, "a vehicle hull");
        assert_eq!(q(MotionType::Character, BIPED, 1, false, false), QualityType::Character);
        assert_eq!(q(MotionType::Character, BIPED, 1, false, true), QualityType::Critical, "biped flag bit 2 set");
        assert_eq!(q(MotionType::Dynamic, WEAPON, 0x12, false, false), QualityType::Critical, "weapons ignore the layer rule");
        assert_eq!(q(MotionType::Dynamic, 10, 0x12, false, false), QualityType::Debris, "a crate on layer 0x12");
        assert_eq!(q(MotionType::Dynamic, 10, 0x13, false, false), QualityType::Debris);
        assert_eq!(q(MotionType::Dynamic, 10, 0x01, false, false), QualityType::Critical);
        assert_eq!(q(MotionType::Dynamic, VEHICLE, 0xF, true, false), QualityType::Debris, "flag bit 25 forces debris");
        assert_eq!(q(MotionType::Keyframed, VEHICLE, 0xF, true, false), QualityType::Keyframed);
        assert_eq!(q(MotionType::Fixed, 10, 0x12, false, false), QualityType::Fixed);
        // allowed penetration: 0.4 / 0.2 / fixed 0.025 for flagged vehicles, floor 0.01
        assert!((allowed_penetration_depth(1.0, 1.0, SCENERY, false, false) - 0.4).abs() < 1e-6);
        assert!((allowed_penetration_depth(1.0, 1.0, BIPED, false, false) - 0.2).abs() < 1e-6);
        assert!((allowed_penetration_depth(2.0, 99.0, VEHICLE, false, true) - 0.05).abs() < 1e-6);
        assert_eq!(allowed_penetration_depth(1.0, 0.001, SCENERY, false, false), MIN_ALLOWED_PENETRATION_DEPTH);
    }

    #[test]
    fn the_time_of_impact_fraction_is_where_the_gap_closes() {
        assert_eq!(time_of_impact_fraction(0.1, -0.1), Some(0.5));
        assert_eq!(time_of_impact_fraction(0.3, -0.1), Some(0.75));
        assert_eq!(time_of_impact_fraction(0.0, -0.2), Some(0.0), "already touching: impact at the start");
        assert_eq!(time_of_impact_fraction(-0.1, -0.2), None, "already inside: not a new crossing");
        assert_eq!(time_of_impact_fraction(0.2, 0.1), None, "did not reach");
    }

    /// The seam: the vehicle law REPLACES the hull's velocity each tick (it does not add to it),
    /// and Havok then adds gravity on top.
    #[test]
    fn the_vehicle_law_replaces_the_velocity_and_havok_adds_gravity() {
        let mut hull = HavokBody::new(MotionType::BoxInertia, 1000.0, Mat3::IDENTITY * 1e-3, Vec3::new(0.0, 0.0, 10.0), Quat::IDENTITY);
        hull.set_gravity_factor(vehicle_body_gravity_factor(1.0, 1.0, vehicle_gravity_scale_postprocessed(0.5)));
        hull.v = Vec3::new(50.0, 0.0, 0.0); // whatever the last tick left
        hull.set_velocity(Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO);
        hull.step_psi(TICK_SECONDS, world_gravity());
        assert!((hull.v.x - 4.0).abs() < 1e-6, "vx = {}", hull.v.x);
        assert!((hull.v.z + 0.5 * G * TICK_SECONDS).abs() < 1e-5, "half gravity for one tick, got vz {}", hull.v.z);
    }

    // ---- contacts -------------------------------------------------------------------------

    /// A unit-mass sphere (r = 0.5) on the plane z = 0, one contact at its lowest point.
    fn sphere(z: f32) -> HavokBody {
        let i = 0.4 * 10.0 * 0.25; // solid sphere 2/5 m r^2
        HavokBody::new(MotionType::SphereInertia, 10.0, Mat3::from_diagonal(Vec3::splat(1.0 / i)), Vec3::new(0.0, 0.0, z), Quat::IDENTITY)
    }
    fn gap(b: &HavokBody) -> f32 {
        b.com.z - 0.5
    }
    fn plane_contact(b: &HavokBody, mu: f32, e: f32) -> ContactPoint {
        ContactPoint { point: Vec3::new(b.com.x, b.com.y, 0.0), normal: Vec3::Z, separation: gap(b), friction: mu, restitution: e }
    }
    /// One PSI the way the engine orders it: integrate, and if the body went through the plane
    /// rewind to the time of impact, resolve there, and spend the rest of the step on the new
    /// velocity. Returns true when the body was in contact this tick.
    fn tick(b: &mut HavokBody, mu: f32, e: f32, p: &SolverParams) -> bool {
        let (g0, p0) = (gap(b), b.com);
        b.step_psi(TICK_SECONDS, world_gravity());
        let g1 = gap(b);
        if let Some(f) = time_of_impact_fraction(g0, g1) {
            b.com = p0 + (b.com - p0) * f;
            solve_contacts(b, &[plane_contact(b, mu, e)], TICK_SECONDS, p);
            b.com += b.v * (1.0 - f) * TICK_SECONDS;
            true
        } else if g1 <= COLLISION_TOLERANCE {
            solve_contacts(b, &[plane_contact(b, mu, e)], TICK_SECONDS, p);
            true
        } else {
            false
        }
    }

    /// A body at rest on a plane stays on it: no sinking, no drift, no pumping.
    #[test]
    fn a_resting_body_does_not_drift() {
        let mut b = sphere(0.5);
        let mut in_contact = 0;
        for _ in 0..600 {
            in_contact += tick(&mut b, 0.8, 0.0, &SolverParams::ENGINE) as u32;
        }
        assert_eq!(in_contact, 600, "the body must be in contact every tick of its rest");
        assert!((b.com.z - 0.5).abs() < 1e-4, "rest height {}", b.com.z);
        assert!(b.v.length() < 0.01, "residual speed {}", b.v.length());
        assert!(b.com.x.abs() < 1e-5 && b.com.y.abs() < 1e-5, "drifted to {:?}", b.com);
    }

    /// With a FINITE resting velocity, a dropped sphere rebounds to `e^2 * h` -- the analytic
    /// energy statement, within the 60 Hz discretisation.
    #[test]
    fn restitution_rebounds_to_e_squared_h_when_it_is_allowed_to_apply() {
        let p = SolverParams { resting_velocity: 1.0, ..SolverParams::ENGINE };
        let e = 0.5;
        let h0 = 2.0;
        let mut b = sphere(0.5 + h0);
        let mut touched = false;
        let mut apex = 0.5f32;
        for _ in 0..600 {
            touched |= tick(&mut b, 0.0, e, &p);
            if touched && b.v.z > 0.0 {
                apex = apex.max(b.com.z);
            }
        }
        assert!(touched, "the sphere never reached the plane");
        let height = apex - 0.5;
        let want = e * e * h0;
        assert!((height - want).abs() < 0.08 * want, "rebounded {height} want {want}");
        // and a different restitution gives the different, squared height (not a linear one)
        let mut c = sphere(0.5 + h0);
        let (mut touched, mut apex2) = (false, 0.5f32);
        for _ in 0..600 {
            touched |= tick(&mut c, 0.0, 0.8, &p);
            if touched && c.v.z > 0.0 {
                apex2 = apex2.max(c.com.z);
            }
        }
        assert!(((apex2 - 0.5) - 0.64 * h0).abs() < 0.08 * 0.64 * h0, "e=0.8 rebounded {} want {}", apex2 - 0.5, 0.64 * h0);
    }

    /// The ENGINE's resting velocity is FLT_MAX: the same drop does not rebound. (INFERRED
    /// consequence -- see the module note.)
    #[test]
    fn with_the_engines_resting_velocity_nothing_bounces() {
        let mut b = sphere(2.5);
        let mut max_up = 0.0f32;
        let mut touched = false;
        for _ in 0..600 {
            touched |= tick(&mut b, 0.0, 0.9, &SolverParams::ENGINE);
            if touched {
                max_up = max_up.max(b.v.z);
            }
        }
        assert!(touched);
        // a 0.9 restitution would send it back up at ~0.9 * 4.4 wu/s
        assert!(max_up < 0.05, "rebounded upward at {max_up} wu/s with restitution 0.9");
        assert!((b.com.z - 0.5).abs() < 1e-3);
    }

    /// Coulomb friction decelerates a sliding block at `mu g`: v(t) = v0 - mu g t.
    #[test]
    fn sliding_friction_decelerates_at_mu_g() {
        let mut b = HavokBody::new(MotionType::BoxInertia, 10.0, Mat3::ZERO, Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY);
        b.v = Vec3::new(3.0, 0.0, 0.0);
        let mu = 0.5;
        let n = 12u32; // 0.2 s
        for _ in 0..n {
            tick(&mut b, mu, 0.0, &SolverParams::ENGINE);
        }
        let want = 3.0 - mu * G * (n as f32 * TICK_SECONDS);
        assert!((b.v.x - want).abs() < 0.03, "v = {} want {want}", b.v.x);
        // friction never reverses the slide: run long enough to stop
        for _ in 0..200 {
            tick(&mut b, mu, 0.0, &SolverParams::ENGINE);
        }
        assert!(b.v.x.abs() < 0.02 && b.v.x > -0.02, "ended at {}", b.v.x);
        // and a frictionless contact keeps sliding
        let mut ice = HavokBody::new(MotionType::BoxInertia, 10.0, Mat3::ZERO, Vec3::new(0.0, 0.0, 0.5), Quat::IDENTITY);
        ice.v = Vec3::new(3.0, 0.0, 0.0);
        for _ in 0..n {
            tick(&mut ice, 0.0, 0.0, &SolverParams::ENGINE);
        }
        assert!((ice.v.x - 3.0).abs() < 1e-3, "ice lost speed: {}", ice.v.x);
    }

    /// The pair solver: two equal bodies colliding inelastically share their momentum (common
    /// velocity = the average), a body landing on an immovable one stops, and a 100:1 pair is
    /// clamped to the engine's 16:1 effective ratio.
    #[test]
    fn the_pair_solver_conserves_momentum_and_clamps_the_mass_ratio() {
        let mk = |m: f32, z: f32, vz: f32| {
            let mut b = HavokBody::new(MotionType::Dynamic, m, Mat3::ZERO, Vec3::new(0.0, 0.0, z), Quat::IDENTITY);
            b.v = Vec3::new(0.0, 0.0, vz);
            b
        };
        let c = |a: usize, b: usize| PairContact { a, b: Some(b), point: Vec3::new(0.0, 0.0, 0.5), normal: -Vec3::Z, separation: 0.0, friction: 0.0, restitution: 0.0 };
        let p = SolverParams::ENGINE;
        // equal masses: lower A at rest, upper B falling at -1
        let mut bodies = vec![mk(10.0, 0.0, 0.0), mk(10.0, 1.0, -1.0)];
        solve_pair_contacts(&mut bodies, &[c(0, 1)], TICK_SECONDS, &p);
        assert!((bodies[0].v.z + 0.5).abs() < 1e-4 && (bodies[1].v.z + 0.5).abs() < 1e-4, "common velocity {} {}", bodies[0].v.z, bodies[1].v.z);
        assert!((10.0 * bodies[0].v.z + 10.0 * bodies[1].v.z + 10.0).abs() < 1e-3, "momentum not conserved");
        // landing on an immovable body
        let mut fixed = mk(10.0, 0.0, 0.0);
        fixed.inv_mass = 0.0;
        let mut bodies = vec![fixed, mk(10.0, 1.0, -1.0)];
        solve_pair_contacts(&mut bodies, &[c(0, 1)], TICK_SECONDS, &p);
        assert!(bodies[1].v.z.abs() < 1e-4, "did not stop: {}", bodies[1].v.z);
        assert_eq!(bodies[0].v, Vec3::ZERO);
        // 1000 kg hits 10 kg: clamped to 16:1, so the light body ends up faster than it would
        // against the full mass and the heavy one slows more than 1000:10 would allow
        assert_eq!(mass_multipliers(1000.0, 10.0), (0.16, 1.0), "the heavy body is lowered to 16x the light one");
        assert_eq!(mass_multipliers(10.0, 1000.0), (1.0, 0.16));
        assert_eq!(mass_multipliers(100.0, 10.0), (1.0, 1.0), "10:1 is within the clamp");
        let mut bodies = vec![mk(1000.0, 0.0, 1.0), mk(10.0, 1.0, 0.0)];
        // heavy A moving up into light B above it: contact normal from B into A is -z
        solve_pair_contacts(&mut bodies, &[c(0, 1)], TICK_SECONDS, &p);
        let heavy_loss = 1.0 - bodies[0].v.z;
        // unclamped, 1000:10 would slow the heavy body by 1/101 = 0.0099; clamped 160:10 by 1/17
        assert!((heavy_loss - 1.0 / 17.0).abs() < 1e-3, "heavy body lost {heavy_loss} of its speed");
    }

    /// The hover law: at the analytic rest clearance the net acceleration is zero; closer it
    /// pushes up, farther it lets the hull fall; beyond reach it does nothing; the damping resists
    /// the approach; an upside-down hull gets no lift. Literal numbers, three points of strength
    /// 0.6: sum 1.8, c = 1/sqrt(1.8) = 0.7454, clearance 0.2 + 1.0 * 0.2546 = 0.4546.
    #[test]
    fn the_hover_law_balances_gravity_at_the_analytic_clearance() {
        let law = HoverLaw { strength: 0.6, height: 1.0, radius: 0.2, compression_damping: 1.0, extension_damping: 1.0, normal_k_hi: 0.8, normal_k_lo: 0.2 };
        let clearance = hover_rest_clearance(1.8, 1.0, 0.2, 1.0).unwrap();
        assert!((clearance - 0.4546).abs() < 1e-3, "clearance {clearance}");
        // three such points hold gravity (g) up exactly there, at rest
        let a = 3.0 * hover_acceleration(&law, G, Some(clearance), 0.0, 1.0);
        assert!((a - G).abs() < 1e-2 * G, "lift {a} against g = {G}: not balanced");
        // lower pushes up harder, higher lets it fall
        assert!(3.0 * hover_acceleration(&law, G, Some(0.3), 0.0, 1.0) > G);
        assert!(3.0 * hover_acceleration(&law, G, Some(0.7), 0.0, 1.0) < G);
        // out of reach: nothing
        assert_eq!(hover_acceleration(&law, G, Some(1.3), 0.0, 1.0), 0.0);
        assert_eq!(hover_acceleration(&law, G, None, 0.0, 1.0), 0.0);
        // approaching the surface (v_n < 0) is resisted, leaving it is damped the other way
        // (one point's lift at equilibrium is a third of g, so compare the three-point total with g)
        assert!(3.0 * hover_acceleration(&law, G, Some(clearance), -1.0, 1.0) > G + 1.0);
        assert!(3.0 * hover_acceleration(&law, G, Some(clearance), 1.0, 1.0) < G - 1.0);
        // tilt ramp: upright full, upside down none, halfway between the thresholds half
        assert_eq!(hover_tilt_scale(0.8, 0.2, 1.0), 1.0);
        assert_eq!(hover_tilt_scale(0.8, 0.2, -1.0), 0.0);
        assert!((hover_tilt_scale(0.8, 0.2, 0.5) - 0.5).abs() < 1e-6);
        // a weak hull cannot hover; a lighter world (gravity factor 0.5) hovers on less
        assert_eq!(hover_rest_clearance(0.9, 1.0, 0.2, 1.0), None);
        assert!(hover_rest_clearance(0.9, 1.0, 0.2, 0.5).is_some());
    }

    #[test]
    fn an_empty_hover_vehicle_brakes_toward_rest_at_its_authored_deceleration() {
        let b = IdleBrake { speed_decel: 6.0, slide_decel: 3.0, drag: 0.0, constant_decel: 0.0 };
        let t = 1.0 / 60.0;
        // forward at 5 wu/s: stopped by the forward deceleration, backwards
        let (af, al) = hover_idle_brake(&b, 5.0, 0.0, t);
        assert!((af + 6.0).abs() < 1e-4 && al.abs() < 1e-4, "{af} {al}");
        // sideways at 5 wu/s: the slide deceleration
        let (af, al) = hover_idle_brake(&b, 0.0, 5.0, t);
        assert!(af.abs() < 1e-4 && (al + 3.0).abs() < 1e-4, "{af} {al}");
        // slower than the cap allows: exactly stopped in one tick
        let (af, _) = hover_idle_brake(&b, 0.01, 0.0, t);
        assert!((af * t + 0.01).abs() < 1e-6);
        // at rest nothing; drag grows with the square of speed and the constant part is capped
        assert_eq!(hover_idle_brake(&b, 0.0, 0.0, t), (0.0, 0.0));
        let d = IdleBrake { drag: 0.1, constant_decel: 1.0, ..b };
        assert!((hover_idle_drag(&d, 2.0, t) - (0.04 + 0.96)).abs() < 1e-5);
        // the constant part tops the total up to the authored 1.0; near rest it is limited to what stops the hull in a tick
        assert!((hover_idle_drag(&d, 0.5, t) - 1.0).abs() < 1e-5);
        assert!((hover_idle_drag(&d, 0.001, t) - 0.06).abs() < 1e-4);
    }

    // ---- real tags ------------------------------------------------------------------------

    fn rd_f32(b: &[u8], o: usize) -> f32 {
        f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }
    fn rd_u32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    /// The `matg` materials' physics-properties record, read through the offsets the tool's own
    /// postprocess (`game_globals_postprocess_materials`) names: `+0x24` friction, `+0x28`
    /// restitution, `+0x2C` density, validated for ROOT materials (no parent name at `+0x04`), whose zero/negative values it replaces with 0.5 / 0.5 / 1000. So after that fix-up
    /// every root must read all three > 0, and the well-known ones must be where a player expects:
    /// liquid slippery, default high friction, rubber ball bouncy.
    ///
    /// ALSO FOUND, and printed: child materials author 0 / 0 (e.g. `tough_terrain_dirt`), and no
    /// code in the tool inherits them from their parent (`sub_1403EB4A0` returns the element
    /// verbatim). Whether the game resolves a BSP triangle to a root before the lookup is an open
    /// question for the live engine, not something this gate can assert.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_root_materials_read_as_friction_coefficients() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("reach_havok gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let matg = native.tags_by_group(cache, "matg");
        assert!(!matg.is_empty(), "no matg tag in forge_halo");
        let blk = native.tag_meta(cache, matg[0], MATG_MATERIALS as u32, 12).expect("matg materials block");
        let (count, ptr) = (rd_u32(&blk, 0) as usize, rd_u32(&blk, 4));
        assert!(count > 20 && count <= 256, "materials count {count}");
        let bytes = native.tag_ptr_bytes(cache, ptr, (count * MATG_MATERIAL_SIZE) as u32).expect("materials bytes");
        let mut roots = std::collections::BTreeMap::new();
        let (mut children, mut zero_children) = (0, 0);
        for i in 0..count {
            let e = &bytes[i * MATG_MATERIAL_SIZE..(i + 1) * MATG_MATERIAL_SIZE];
            let name = native.string_id(cache, rd_u32(e, 0)).unwrap_or_default();
            let (f, r) = global_material_friction_restitution(e).unwrap();
            let density = rd_f32(e, 0x2C);
            let parent_sid = rd_u32(e, 4);
            let parent_name = if parent_sid == 0 || parent_sid == u32::MAX { String::new() } else { native.string_id(cache, parent_sid).unwrap_or_default() };
            if i < 8 {
                println!("   [{i}] {name:<24} parent sid {parent_sid:#010X} = '{parent_name}'  +8 = {:#06X}", u16::from_le_bytes([e[8], e[9]]));
            }
            if parent_name.is_empty() {
                println!("   ROOT {name:<22} friction {f:>6.3}  restitution {r:>6.3}  density {density:>8.1}");
                roots.insert(name, (f, r, density));
            } else {
                children += 1;
                if f == 0.0 && r == 0.0 {
                    zero_children += 1;
                }
            }
        }
        println!("{} root materials, {children} children ({zero_children} author 0/0)", roots.len());
        assert!(roots.len() >= 2, "found {} roots", roots.len());
        for (n, (f, r, d)) in &roots {
            assert!(*f > 0.0 && *f <= 4.0 && *r > 0.0 && *r <= 2.0 && *d > 0.0 && *d < 1.0e5, "root {n}: f {f} r {r} d {d}");
        }
        assert!(zero_children > 0 && zero_children < children, "children inherit-by-zero is the authored pattern");
        // the materials named in the engine's sources, by value (children author their overrides)
        let all: std::collections::BTreeMap<String, (f32, f32)> = (0..count)
            .map(|i| {
                let e = &bytes[i * MATG_MATERIAL_SIZE..(i + 1) * MATG_MATERIAL_SIZE];
                (native.string_id(cache, rd_u32(e, 0)).unwrap_or_default(), global_material_friction_restitution(e).unwrap())
            })
            .collect();
        assert_eq!(all["default_material"], (0.9, 0.2), "the root every other material descends from");
        assert!(all["liquid"].0 < all["default_material"].0, "liquid must be slipperier than the default");
        assert!(all["hard"].1 > all["soft"].1, "hard materials bounce more than soft");
        assert_eq!(all["tough_inorganic_rubber_hum_ball"].1, 1.0, "the rubber ball is perfectly elastic");
    }

    /// Every `vehi`'s authored gravity scale, after the engine's postprocess.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_authored_vehicle_gravity_scales() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("reach_havok gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let vehis = native.tags_by_group(cache, "vehi");
        assert!(vehis.len() > 10, "found {} vehi", vehis.len());
        let mut non_default = 0;
        let mut by_name = std::collections::BTreeMap::new();
        for &v in &vehis {
            let name = native.tag_name(cache, v).unwrap_or_default();
            let m = native.tag_meta(cache, v, 0x5AC, 4).expect("vehi +0x5AC");
            let raw = rd_f32(&m, 0);
            let eff = vehicle_gravity_scale_postprocessed(raw);
            let adjust = crate::reach_vehicle::VehiclePhysics::read(native, cache, v)
                .and_then(|p| p.family.gravity_adjust())
                .map_or(String::from("-"), |a| format!("{a:.3}"));
            println!("   {name:<60} raw {raw:>8.4} -> {eff:>8.4}   type-block gravity adjust {adjust}");
            assert!(raw.is_finite() && raw >= 0.0 && raw < 10.0, "{name}: raw gravity scale {raw}");
            if (eff - 1.0).abs() > 1e-6 {
                non_default += 1;
            }
            by_name.insert(name, (raw, eff));
        }
        // the authored values a player can feel: the Banshee, Falcon and the space fighters are
        // light; the Pelican authors 0, which the engine's postprocess reads as 1
        let find = |leaf: &str| {
            by_name
                .iter()
                .find(|(n, _)| n.ends_with(leaf))
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("no vehi named *{leaf}"))
        };
        assert_eq!(find("human\\warthog\\warthog").0, 1.0);
        assert_eq!(find("covenant\\banshee\\banshee").1, 0.75);
        assert_eq!(find("human\\falcon\\falcon").1, 0.95);
        assert_eq!(find("human\\sabre\\sabre").1, 0.3);
        assert_eq!(find("covenant\\seraph\\seraph").1, 0.3);
        assert_eq!(find("human\\pelican\\pelican"), (0.0, 1.0), "authored 0 means unscaled");
        println!("{} vehicles, {non_default} author a non-unit gravity scale", vehis.len());
    }
}
