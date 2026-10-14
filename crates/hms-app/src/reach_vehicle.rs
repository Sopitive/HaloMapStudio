//! #vehicle-phys  Reach's `vehi` VEHICLE PHYSICS blocks, read from the tag.
//!
//! This module is the tag half of driving. It reads what a vehicle authors and it says plainly
//! what a simple integrator can and cannot do with it; it does not fill a gap with a plausible
//! number.
//!
//! # `vehi`'s own fields are 380 bytes, not the 180 `tagdefs` prints
//!
//! **This is the inline-array trap from `re-notes/reach_runtime_installed_field_sets.md`, and it
//! cost the first version of this module a completely dead gate.** `vehi`'s root field set has no
//! static list, so `tagdefs` falls back to the field array the compiler laid out immediately
//! after the block definition (`blockDef + 0x28`) — and for `vehi` that array is **only the
//! fifteen physics-type blocks**. So `tagdefs` prints `vehicle(180)` and starts the type blocks
//! at `+0x4CC`, and reading block counts there finds **zero populated blocks on all 43 vehicles**
//! in `forge_halo`. That is exactly what the note warns: *"Do not assume the inline array is the
//! complete field list."*
//!
//! The real current array is found by the note's fourth signal — the current field array opens
//! with a `custom` field named `$$$ GROUP $$$`, the legacy one with an `explanation` of the same
//! name. For `vehi`:
//!
//! ```text
//!   .data +0x2141140   custom      "$$$ VEHICLE $$$"   25 records   380 bytes   <- current
//!   .data +0x216B1A0   explanation "$$$ VEHICLE $$$"   54 records   320 bytes   <- legacy
//! ```
//!
//! and the engine's own installer confirms both sizes and which is current
//! (`reverseme func reach_tag_test.exe init_vehicle_definition_field_set`):
//!
//! ```text
//!   declare_legacy_field_set(.., &unk_14216B1A0, "vehicle_definition_fields_v1", 0x140, 0, -1, sub_140E5EE90)
//!   declare_legacy_field_set(.., &unk_142141140, "vehicle_definition_fields_v2", 0x17C, 0, -1, 0)
//! ```
//!
//! `0x140 = 320` and **`0x17C = 380`**, and the legacy one carries an upgrade function while the
//! current one passes NULL.
//!
//! and the current one is:
//!
//! ```text
//!   +0x000 long flags  flags -> vehicle_flags (13 of them; the legacy array's enum has 22)
//!   +0x004 struct      physics types          -> vehicle_physics_types_struct, 180 B
//!   +0x0B8 struct      havok vehicle physics  -> havok_vehicle_physics_struct,  84 B
//!   +0x10C block       tricks                 +0x118 player training vehicle type
//!   +0x119 vehicle size              +0x11C/+0x120 min/max flipping angular velocity
//!   +0x124 crouch transition time    +0x12C/+0x130 seat entrance/exit acceleration scale
//!   +0x134 blur speed                +0x138 flip message (string id)
//!   +0x13C suspension sound          +0x14C special effect
//!   +0x15C driver boost damage       +0x16C rider boost damage        end 0x17C = 380
//! ```
//!
//! So the tag is `object(360) + unit(868) + vehicle(380) = 1608 = 0x648`, the fifteen type blocks
//! begin at **`0x4CC + 4 = 0x4D0`** (everything `tagdefs` printed is four bytes early), and the
//! havok block sits at **`0x584`**. [`gates::the_vehicle_physics_blocks_are_one_per_vehicle`]
//! discriminates the two by the only test that can: at `+0x4CC` every vehicle populates no block,
//! at `+0x4D0` every vehicle populates exactly one.
//!
//! # Fifteen mutually exclusive type blocks
//!
//! ```text
//!   +0x4D0 human_tank    112   +0x524 mantis         172   +0x560 boat           96
//!   +0x4DC human_jeep     88   +0x530 vtol           208   +0x56C space-fighter 196
//!   +0x4E8 human_plane    76   +0x53C chopper        124   +0x578 revenant      168*
//!   +0x4F4 wolverine      96   +0x548 guardian        48
//!   +0x500 alien_scout   116   +0x554 jackal-glider  368
//!   +0x50C alien_fighter 100   +0x518 turret          20
//! ```
//!
//! `*` `revenant_block` is one of the two field sets `tagdefs` re-sums to a different size than
//! the compiler states (168 stated, 173+ summed, with unaligned `real`s at `+0x099`/`+0x09D`),
//! so **only its leading `human_tank_struct` and the four aligned speed pairs at
//! `+0x070..+0x08C` are read here**; the fields past `+0x098` are left alone rather than read at
//! offsets that do not add up.
//!
//! # The havok block is where the WHEELS and the per-vehicle GRAVITY are
//!
//! `havok_vehicle_physics_struct`, 84 bytes at `+0x584`, and it is the half of the authoring that
//! matters most to anything that touches the ground:
//!
//! ```text
//!   +0x04 ground friction        +0x08 ground depth       +0x0C ground damp factor
//!   +0x10 ground moving friction +0x14/+0x18 ground maximum slope 0 / 1 [degrees 0-90]
//!   +0x1C maximum normal force contribution ("0 defaults to 3")
//!   +0x20 anti_gravity_bank_lift +0x24 steering_bank_reaction_scale
//!   +0x28 GRAVITY SCALE          "value of 0 defaults to 1. .5 is half gravity"
//!   +0x30 iteration count        +0x34 suspension count
//!   +0x38 anti gravity points [max 16]  (72 B each: marker name, strength, height, damping,
//!                                        normal k0/k1, radius, per-damage-state error)
//!   +0x44 friction points      [max 16]  (52 B each: marker name, FRACTION OF TOTAL MASS,
//!                                        radius, damaged radius, e-brake frictions, the
//!                                        region whose destruction blows the tyre off)
//!   +0x50 boat physics (52 B)
//! ```
//!
//! **`gravity scale` at `+0x28` is the authored per-vehicle gravity**, and it is a different
//! quantity from the type block's `gravity adjust` ("the fraction by which we scale gravity that
//! is NOT along the ground plane") and from `obje +0x20 vertical acceleration scale`. All three
//! are read and reported separately rather than multiplied together on the strength of their
//! names.
//!
//! **The friction points are the wheels.** A Warthog's four are named markers with a radius, a
//! `fraction of total mass`, a damaged radius "when the tire is blown off", and the model region
//! whose destruction does it — which is the same per-wheel damage the measured 75-vitality wheel
//! sections in [`crate::reach_damage`] describe from the other end.
//!
//! # What a GROUND vehicle actually authors — and the surprise
//!
//! **`human_jeep` (Warthog, Mongoose, the civilian trucks) authors no top speed.** There is no
//! `maximum forward speed` field in the block at all. What it authors is:
//!
//! ```text
//!   +0x000 steering control   overdampen cusp angle [degrees], overdampen exponent
//!   +0x008 turning control    maximum left turn, maximum right turn (negative), turn rate
//!   +0x014 engine (48 B)      engine moment, engine max angular velocity,
//!                             gears[16] (72 B each), gear shift sound, load/cruise sounds
//!   +0x044 boat engine        (empty on a land vehicle)
//!   +0x050 wheel circumferance
//!   +0x054 gravity adjust     "0-1 fraction by which we scale gravity that is not along the
//!                              ground plane"
//! ```
//!
//! and each **gear** (72 bytes) carries two `torque_curve_struct`s — a *loaded* and a *cruising*
//! torque curve, each a `c_function_definition` blob — plus `gear ratio`, `min time to upshift`,
//! `engine up-shift scale`, `min time to downshift`, `engine down-shift scale`. So a Warthog's
//! speed is an EMERGENT property of a Havok drivetrain: engine torque through a gearbox to
//! wheels of a given circumference, against the wheel/suspension/friction data in the `phmo`.
//! [`DriveLimits::from_engine`] computes the kinematic ceiling those three numbers imply and
//! says so is a ceiling, not a simulation.
//!
//! **`alien_scout` (Ghost) is the opposite**: it authors the velocities directly —
//! `maximum forward speed`, `maximum reverse speed`, `speed acceleration`, `speed deceleration`,
//! `maximum left/right slide`, `slide acceleration/deceleration`, `slide speed at top speed`,
//! `drag coeficient`, `constant deceleration`, `torque scale`. That is a model a simple
//! integrator reproduces as authored, because the authoring IS a first-order velocity model.
//!
//! **`human_tank` (Scorpion, Wraith)** authors neither: it is differential (left/right track)
//! steering — `forward arc`, `maximum left/right differential`, `differential
//! acceleration/deceleration` and the same engine-and-gears — plus `flip window` and
//! `pegged fraction`.
//!
//! # What is NOT implemented, stated once
//!
//! * **No Havok.** No wheel contacts, no suspension, no tyre friction circle, no per-wheel load
//!   transfer, so a Warthog here does not lean, does not lose traction, and does not climb a kerb
//!   the way the real one does.
//! * **The torque curves are READ and EXPOSED, not integrated.** [`Gear::loaded_torque`] hands
//!   back the decoded [`crate::scene::MapFn`] so a caller can plot or evaluate it, and
//!   [`VehiclePhysics::describe`] prints the gear table. Turning torque into wheel force needs
//!   the `phmo`'s wheel masses and radii and the friction model, none of which is read here.
//! * **No gear shifting.** `min time to upshift` / `engine up-shift scale` are read and printed;
//!   nothing acts on them.
//! * **The flyers are not implemented.** `human_plane`, `alien_fighter`, `vtol`,
//!   `jackal-glider`, `space-fighter` read and report their authored fields and
//!   [`VehiclePhysics::drivable`] returns false for them, because inventing flight dynamics is
//!   exactly the thing not to do.
//!
//! # Gravity
//!
//! Nothing here holds a gravity constant. A falling vehicle integrates
//! [`crate::reach_physics::fall_acceleration`], and the two per-vehicle modifiers that ARE
//! authored are read: `obje +0x20 vertical acceleration scale`
//! ([`crate::reach_physics::acceleration_scales`] — Scorpion 0.100, Wraith 0.250, Ghost 1.500)
//! and the type block's own `gravity adjust`, whose field documentation is "0-1 fraction by which
//! we scale gravity that is not along the ground plane".

use hms_native::{CacheHandle, NativeDll};

/// #vehicle-drivetrain #vehicle-suspension  The engine's own drivetrain and ground-contact law,
/// disassembled from `reach_tag_test.exe`. A child module so that the crate root does not have to
/// change to carry it.
#[path = "reach_vehicle_drive.rs"]
pub mod drive;

// ---------------------------------------------------------------------------------------------
// THE LAYOUT
// ---------------------------------------------------------------------------------------------

/// `vehi`'s own fields begin here: `object(360) + unit(868)`.
pub const VEHI_OWN_FIELDS: usize = 0x4CC;
/// The current field array's own total, so the tag is `0x4CC + 0x17C = 0x648 = 1608` bytes.
pub const VEHI_OWN_SIZE: usize = 380;
/// `vehicle_flags`, 13 of them in the current array (the legacy array's enum has 22 — one of the
/// signals that separated the two).
pub const VEHI_FLAGS: usize = 0x4CC;
/// `vehicle_physics_types_struct`: the fifteen `[max 1]` type blocks, 180 bytes.
pub const VEHI_PHYSICS_TYPES: usize = 0x4D0;
/// `havok_vehicle_physics_struct`, 84 bytes.
pub const VEHI_HAVOK: usize = 0x584;
pub const VEHI_TRICKS: usize = 0x5D8;
pub const VEHI_PLAYER_TRAINING_TYPE: usize = 0x5E4;
pub const VEHI_VEHICLE_SIZE: usize = 0x5E5;
pub const VEHI_MIN_FLIPPING_ANGULAR_VELOCITY: usize = 0x5E8;
pub const VEHI_MAX_FLIPPING_ANGULAR_VELOCITY: usize = 0x5EC;
pub const VEHI_CROUCH_TRANSITION_TIME: usize = 0x5F0;
pub const VEHI_SEAT_ENTRANCE_ACCEL_SCALE: usize = 0x5F8;
pub const VEHI_SEAT_EXIT_ACCEL_SCALE: usize = 0x5FC;
pub const VEHI_BLUR_SPEED: usize = 0x600;
pub const VEHI_FLIP_MESSAGE: usize = 0x604;

/// `(block offset in the tag, the editor's own name, element size, is it a ground family this
/// module drives)`, in tag order. Offsets are `VEHI_PHYSICS_TYPES + 12 * index`.
pub const TYPE_BLOCKS: [(usize, &str, usize, bool); 15] = [
    (0x4D0, "human_tank", 112, true),
    (0x4DC, "human_jeep", 88, true),
    (0x4E8, "human_plane", 76, false),
    (0x4F4, "wolverine", 96, true),
    (0x500, "alien_scout", 116, true),
    (0x50C, "alien_fighter", 100, false),
    (0x518, "turret", 20, false),
    (0x524, "mantis", 172, false),
    (0x530, "vtol", 208, false),
    (0x53C, "chopper", 124, true),
    (0x548, "guardian", 48, false),
    (0x554, "jackal-glider", 368, false),
    (0x560, "boat", 96, false),
    (0x56C, "space-fighter", 196, false),
    (0x578, "revenant", 168, true),
];

/// `havok_vehicle_physics_struct` field offsets, relative to [`VEHI_HAVOK`].
pub const HAVOK_SIZE: usize = 84;
pub const HV_FLAGS: usize = 0x00;
pub const HV_GROUND_FRICTION: usize = 0x04;
pub const HV_GROUND_DEPTH: usize = 0x08;
pub const HV_GROUND_DAMP_FACTOR: usize = 0x0C;
pub const HV_GROUND_MOVING_FRICTION: usize = 0x10;
pub const HV_GROUND_MAX_SLOPE_0: usize = 0x14;
pub const HV_GROUND_MAX_SLOPE_1: usize = 0x18;
pub const HV_MAX_NORMAL_FORCE: usize = 0x1C;
pub const HV_ANTIGRAVITY_BANK_LIFT: usize = 0x20;
pub const HV_STEERING_BANK_REACTION_SCALE: usize = 0x24;
/// **The authored per-vehicle gravity scale.** "value of 0 defaults to 1. .5 is half gravity".
pub const HV_GRAVITY_SCALE: usize = 0x28;
pub const HV_UPDATE_PERIOD_TICKS: usize = 0x2C;
pub const HV_REMOTE_UPDATE_PERIOD_TICKS: usize = 0x2E;
pub const HV_ITERATION_COUNT: usize = 0x30;
pub const HV_SUSPENSION_COUNT: usize = 0x34;
pub const HV_ANTIGRAVITY_POINTS: usize = 0x38;
pub const HV_FRICTION_POINTS: usize = 0x44;
pub const HV_BOAT_PHYSICS: usize = 0x50;

/// `anti_gravity_point_definition_block`, 72 bytes.
pub const AGP_SIZE: usize = 72;
pub const AGP_MARKER_NAME: usize = 0x00;
pub const AGP_FLAGS: usize = 0x04;
pub const AGP_DAMPING: usize = 0x08;
pub const AGP_STRENGTH: usize = 0x0C;
pub const AGP_HEIGHT: usize = 0x10;
pub const AGP_COMPRESSION_DAMPING: usize = 0x14;
pub const AGP_EXTENSION_DAMPING: usize = 0x18;
pub const AGP_NORMAL_K1: usize = 0x1C;
pub const AGP_NORMAL_K0: usize = 0x20;
pub const AGP_RADIUS: usize = 0x24;

/// `friction_point_definition_block`, 52 bytes. **These are the wheels.**
pub const FP_SIZE: usize = 52;
pub const FP_MARKER_NAME: usize = 0x00;
pub const FP_FLAGS: usize = 0x04;
pub const FP_FRACTION_OF_TOTAL_MASS: usize = 0x08;
pub const FP_RADIUS: usize = 0x0C;
pub const FP_DAMAGED_RADIUS: usize = 0x10;
pub const FP_MOVING_FRICTION_VELOCITY_DIFF: usize = 0x14;
pub const FP_EBRAKE_MOVING_FRICTION: usize = 0x18;
pub const FP_EBRAKE_FRICTION: usize = 0x1C;
pub const FP_EBRAKE_MOVING_FRICTION_VEL_DIFF: usize = 0x20;
pub const FP_MODEL_STATE_DESTROYED: usize = 0x2A;
pub const FP_REGION_NAME: usize = 0x2C;

/// `global_vehicle_engine_struct`, 48 bytes.
pub const ENGINE_SIZE: usize = 48;
pub const ENG_MOMENT: usize = 0x00;
pub const ENG_MAX_ANGULAR_VELOCITY: usize = 0x04;
pub const ENG_GEARS: usize = 0x08;
pub const ENG_GEAR_SHIFT_SOUND: usize = 0x14;
pub const ENG_LOAD_AND_CRUISE: usize = 0x24;

/// `gear_block`, 72 bytes.
pub const GEAR_SIZE: usize = 72;
/// `loaded torque curve` -> `torque_curve_struct` +0x00 -> `scalar_function_named_struct`.
pub const GEAR_LOADED_TORQUE: usize = 0x00;
pub const GEAR_CRUISING_TORQUE: usize = 0x18;
pub const GEAR_MIN_TIME_TO_UPSHIFT: usize = 0x30;
pub const GEAR_UPSHIFT_SCALE: usize = 0x34;
pub const GEAR_RATIO: usize = 0x38;
pub const GEAR_MIN_TIME_TO_DOWNSHIFT: usize = 0x3C;
pub const GEAR_DOWNSHIFT_SCALE: usize = 0x40;

/// A `c_function_definition` / `mapping_function` is 20 bytes, and in the CACHE its `data`
/// payload is `{u32 size at +0x00, ..., u32 address at +0x0C}` — the same pair
/// `crate::scene`'s object-function reader uses at `object_function_block +0x18`.
pub const FUNCTION_SIZE: usize = 20;
pub const FUNCTION_DATA_SIZE: usize = 0x00;
pub const FUNCTION_DATA_ADDRESS: usize = 0x0C;

/// `vehicle_steering_control_struct`, 8 bytes.
pub const STEER_OVERDAMPEN_CUSP_ANGLE: usize = 0x00;
pub const STEER_OVERDAMPEN_EXPONENT: usize = 0x04;
/// `vehicle_turning_control_struct`, 12 bytes.
pub const TURN_MAX_LEFT: usize = 0x00;
pub const TURN_MAX_RIGHT: usize = 0x04;
pub const TURN_RATE: usize = 0x08;

// `human_jeep_block`, 88 bytes.
pub const JEEP_STEERING: usize = 0x00;
pub const JEEP_TURNING: usize = 0x08;
pub const JEEP_ENGINE: usize = 0x14;
pub const JEEP_BOAT_ENGINE: usize = 0x44;
pub const JEEP_WHEEL_CIRCUMFERANCE: usize = 0x50;
pub const JEEP_GRAVITY_ADJUST: usize = 0x54;

// `human_tank_block`, 112 bytes (and the leading 112 bytes of `revenant_block`).
pub const TANK_FORWARD_ARC: usize = 0x00;
pub const TANK_PERPENDICULAR_FORWARD_ARC: usize = 0x04;
pub const TANK_FLIP_WINDOW: usize = 0x08;
pub const TANK_PEGGED_FRACTION: usize = 0x0C;
pub const TANK_FORWARD_TURN_SCALE: usize = 0x10;
pub const TANK_REVERSE_TURN_SCALE: usize = 0x14;
pub const TANK_MAX_LEFT_DIFFERENTIAL: usize = 0x18;
pub const TANK_MAX_RIGHT_DIFFERENTIAL: usize = 0x1C;
pub const TANK_DIFFERENTIAL_ACCELERATION: usize = 0x20;
pub const TANK_DIFFERENTIAL_DECELERATION: usize = 0x24;
pub const TANK_ENGINE: usize = 0x38;
pub const TANK_WHEEL_CIRCUMFERANCE: usize = 0x68;
pub const TANK_GRAVITY_ADJUST: usize = 0x6C;

// `revenant_block`: the aligned speed fields after the embedded `human_tank_struct`.
pub const REV_MAX_FORWARD_SPEED: usize = 0x70;
pub const REV_MAX_REVERSE_SPEED: usize = 0x74;
pub const REV_SPEED_ACCELERATION: usize = 0x78;
pub const REV_SPEED_DECELERATION: usize = 0x7C;
pub const REV_MAX_LEFT_SLIDE: usize = 0x80;
pub const REV_MAX_RIGHT_SLIDE: usize = 0x84;
pub const REV_SLIDE_ACCELERATION: usize = 0x88;
pub const REV_SLIDE_DECELERATION: usize = 0x8C;

// `alien_scout_block`, 116 bytes.
pub const SCOUT_STEERING: usize = 0x00;
pub const SCOUT_MAX_FORWARD_SPEED: usize = 0x08;
pub const SCOUT_MAX_REVERSE_SPEED: usize = 0x0C;
pub const SCOUT_SPEED_ACCELERATION: usize = 0x10;
pub const SCOUT_SPEED_DECELERATION: usize = 0x14;
pub const SCOUT_MAX_LEFT_SLIDE: usize = 0x18;
pub const SCOUT_MAX_RIGHT_SLIDE: usize = 0x1C;
pub const SCOUT_SLIDE_ACCELERATION: usize = 0x20;
pub const SCOUT_SLIDE_DECELERATION: usize = 0x24;
pub const SCOUT_SLIDE_SPEED_AT_TOP_SPEED: usize = 0x28;
pub const SCOUT_SPECIFIC_TYPE: usize = 0x2C;
pub const SCOUT_FLAGS: usize = 0x2D;
pub const SCOUT_DRAG_COEFFICIENT: usize = 0x30;
pub const SCOUT_CONSTANT_DECELERATION: usize = 0x34;
pub const SCOUT_TORQUE_SCALE: usize = 0x38;

// ---------------------------------------------------------------------------------------------
// WHAT IS READ
// ---------------------------------------------------------------------------------------------

fn rd_u32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() { return 0 }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    let v = f32::from_bits(rd_u32(b, o));
    if v.is_finite() { v } else { 0.0 }
}

/// A `vehicle_steering_control_struct`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Steering {
    /// `overdampen cusp angle [degrees]`.
    pub overdampen_cusp_angle: f32,
    pub overdampen_exponent: f32,
}

/// A `vehicle_turning_control_struct`. `max_right` is authored NEGATIVE (the field's own name is
/// "maximum right turn (negative)").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Turning {
    pub max_left: f32,
    pub max_right: f32,
    pub turn_rate: f32,
}

/// One `gear_block`. The torque curves are kept as decoded functions, not evaluated.
#[derive(Clone, Debug)]
pub struct Gear {
    pub index: usize,
    pub ratio: f32,
    pub min_time_to_upshift: f32,
    pub upshift_scale: f32,
    pub min_time_to_downshift: f32,
    pub downshift_scale: f32,
    /// `loaded torque curve`, decoded. `None` when the gear authors no curve.
    pub loaded_torque: Option<crate::scene::MapFn>,
    /// `cruising torque curve`, decoded.
    pub cruising_torque: Option<crate::scene::MapFn>,
}

/// A `global_vehicle_engine_struct`.
#[derive(Clone, Debug, Default)]
pub struct Engine {
    /// "higher moments make engine spin up slower".
    pub moment: f32,
    /// `engine max angular velocity`.
    pub max_angular_velocity: f32,
    pub gears: Vec<Gear>,
}

/// The kinematic ceiling an engine-and-gearbox implies, and the honest caveat that goes with it.
///
/// `engine max angular velocity` x the TOP gear's `gear ratio`.
///
/// # It is NOT times the wheel circumference, and it used to be
///
/// **Measured** (`reach_tag_test.exe`): `vehicle_engine.cpp`'s `sub_140E600E0` returns exactly
/// `gear_ratio * engine_angular_velocity`, and the jeep's own tick (`sub_140E69D40`) does
///
/// ```text
///   phase += that * dt;  phase = fmodf(phase, *((float *)jeep_block + 20));   // +0x50
/// ```
///
/// — it adds it to a rolling PHASE and wraps that phase at `wheel circumferance`. So the product
/// is already a distance rate in wu/s, and the circumference is the wrap length of the wheel
/// animation and nothing else. Multiplying by it as well made a Warthog's ceiling 14.24 wu/s
/// where the drivetrain's real ceiling is **11.40**; a Mongoose 13.20 where it is **8.80**.
///
/// The corrected numbers also line up with the one other authored speed-like quantity, which the
/// old ones did not: the driver seat's `max speed reference` is 9.0 on a Warthog and 7.0 on a
/// Mongoose, so the ratio of drivetrain ceiling to seat reference is **1.267 and 1.257** — the
/// same number on two vehicles with different gearboxes, different wheel sizes and a 3x
/// difference in engine moment.
///
/// # Which gear is top is a MEASUREMENT, and the obvious reading is backwards
///
/// In a real gearbox a ratio is engine-revs-per-wheel-rev and it *decreases* as you shift up.
/// Reach's ratios *increase* with the gear index — the Warthog authors `-0.50, 0.75, 1.00, 1.20`
/// — so the first version of this code took the SMALLEST positive ratio as top gear, and that was
/// wrong. **The shift times settle it**, because they are authored asymmetrically:
///
/// ```text
///   Warthog  gear 0  ratio -0.50   up 0.00s   down 0.00s    <- reverse
///            gear 1  ratio  0.75   up 0.85s   down 0.00s    <- CANNOT downshift: it is lowest
///            gear 2  ratio  1.00   up 0.85s   down 0.25s
///            gear 3  ratio  1.20   up 0.00s   down 0.25s    <- CANNOT upshift: it is highest
/// ```
///
/// A gear with no downshift time is the bottom of the box and a gear with no upshift time is the
/// top, so the index order IS low-to-high and the ratio is a wheel-speed MULTIPLIER, not a
/// reduction. Top gear is therefore the **largest positive ratio**.
/// [`gates::the_gear_order_is_proved_by_the_shift_times`] asserts that pattern over every geared
/// vehicle in the cache, so the reading is checked rather than assumed.
///
/// # It is a CEILING, not a top speed
///
/// It is the speed at which the drivetrain can no longer turn the wheels faster: no load, no
/// drag, no slip, full grip. Measured, it overestimates: the Warthog's ceiling is 14.2 wu/s while
/// the Ghost — which authors its speed DIRECTLY — states 5.0 wu/s, and a Reach Warthog is not
/// nearly three times a Ghost. So this number is reported, and a caller must not use it as a top
/// speed. What the real figure is cannot be had without the Havok drivetrain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DriveLimits {
    /// `max_angular_velocity * top gear ratio`, wu/s. **A ceiling.**
    pub top_speed_ceiling: f32,
    /// The gear index that ceiling came from: the largest positive `gear ratio`.
    pub top_gear: usize,
    pub top_gear_ratio: f32,
    pub wheel_circumferance: f32,
}

impl DriveLimits {
    /// `None` when the engine authors no gear with a positive ratio, no engine speed, or no wheel
    /// circumference — in which case there is no speed to state and the caller must not invent
    /// one.
    pub fn from_engine(engine: &Engine, wheel_circumferance: f32) -> Option<DriveLimits> {
        if !(wheel_circumferance > 0.0) || !(engine.max_angular_velocity > 0.0) {
            return None;
        }
        let (top_gear, ratio) = engine
            .gears
            .iter()
            .filter(|g| g.ratio > 0.0)
            .map(|g| (g.index, g.ratio))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        Some(DriveLimits {
            // #vehicle-drivetrain  **NOT times the circumference.** See the type's own note: the
            // engine's `sub_140E600E0` returns `gear_ratio * angular_velocity` and the jeep adds
            // `that * dt` to a phase it wraps at `wheel circumferance`, so the product is already
            // a distance rate in wu/s and the circumference is only the wrap length.
            top_speed_ceiling: engine.max_angular_velocity * ratio,
            top_gear,
            top_gear_ratio: ratio,
            wheel_circumferance,
        })
    }
}

/// The authored first-order velocity model an `alien_scout` (and the Revenant's own tail) IS.
/// Every field is the tag's; nothing is derived.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VelocityModel {
    pub max_forward_speed: f32,
    pub max_reverse_speed: f32,
    pub speed_acceleration: f32,
    pub speed_deceleration: f32,
    pub max_left_slide: f32,
    pub max_right_slide: f32,
    pub slide_acceleration: f32,
    pub slide_deceleration: f32,
    /// `slide speed at top speed [wu/s]` — "the slide speeds are interpolated down to this value,
    /// reaching it when the vehicle is moving at its top speed". Zero on a block that does not
    /// author it (the Revenant's tail has no such field).
    pub slide_speed_at_top_speed: f32,
    pub drag_coefficient: f32,
    pub constant_deceleration: f32,
    /// `torque scale`, "0 defaults to 1".
    pub torque_scale: f32,
}

impl VelocityModel {
    /// The slide ceiling at a given forward speed, by the field's own documentation: the slide
    /// speeds interpolate down to `slide_speed_at_top_speed` as speed reaches
    /// `max_forward_speed`. With no authored value, there is no interpolation.
    pub fn slide_ceiling(&self, speed: f32) -> (f32, f32) {
        if !(self.slide_speed_at_top_speed > 0.0) || !(self.max_forward_speed > 0.0) {
            return (self.max_left_slide, self.max_right_slide);
        }
        let t = (speed.abs() / self.max_forward_speed).clamp(0.0, 1.0);
        let lerp = |a: f32| a + (self.slide_speed_at_top_speed - a) * t;
        (lerp(self.max_left_slide), lerp(self.max_right_slide))
    }
}

/// The differential-steer authoring of a `human_tank`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Differential {
    /// `forward arc`, radians — "outside of this arc the vehicle reverse direciton".
    pub forward_arc: f32,
    pub perpendicular_forward_arc: f32,
    pub flip_window: f32,
    pub pegged_fraction: f32,
    pub forward_turn_scale: f32,
    pub reverse_turn_scale: f32,
    pub max_left_differential: f32,
    pub max_right_differential: f32,
    pub differential_acceleration: f32,
    pub differential_deceleration: f32,
}

/// One `friction_point_definition_block` element — **a wheel**.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrictionPoint {
    pub index: usize,
    /// The model marker the wheel sits on.
    pub marker_name: String,
    pub flags: u32,
    /// `fraction of total mass` — "(0.0-1.0) fraction of total vehicle mass".
    pub fraction_of_total_mass: f32,
    pub radius: f32,
    /// `damaged radius` — "radius when the tire is blown off".
    pub damaged_radius: f32,
    pub moving_friction_velocity_diff: f32,
    pub ebrake_friction: f32,
    pub ebrake_moving_friction: f32,
    /// `model state destroyed` and the `region name` whose destruction does it — the same
    /// per-wheel damage [`crate::reach_damage`] measures from the vitality side.
    pub model_state_destroyed: i16,
    pub region_name: String,
}

/// One `anti_gravity_point_definition_block` element — a hover point (Ghost, Banshee, Wraith).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AntiGravityPoint {
    pub index: usize,
    pub marker_name: String,
    pub flags: u32,
    pub damping: u32,
    pub strength: f32,
    pub height: f32,
    pub compression_damping: f32,
    pub extension_damping: f32,
    pub normal_k0: f32,
    pub normal_k1: f32,
    pub radius: f32,
}

/// `havok_vehicle_physics_struct`: the ground contact authoring, and the per-vehicle gravity.
#[derive(Clone, Debug, Default)]
pub struct Havok {
    pub flags: u32,
    pub ground_friction: f32,
    pub ground_depth: f32,
    pub ground_damp_factor: f32,
    pub ground_moving_friction: f32,
    /// `ground maximum slope 0` / `1`. **The editor labels these "degrees 0-90" and the tag
    /// stores COSINES**, which is a measurement, not a reading of the label: the second field's
    /// own documentation says it must be "greater than slope 0", and the Warthog authors
    /// `0.8, 0.6` — decreasing as numbers, increasing as angles (36.9 deg then 53.1 deg). The
    /// Scorpion authors `0.5, 0.3` (60 deg then 72.5 deg) and the hover vehicles author `1.0,
    /// 1.0` (zero, because they have no wheels on the ground). It is the same encoding
    /// [`crate::reach_physics::GROUND_SLOPE_LIMIT_COS`] records for the biped: `cosf(0.61086524)
    /// = 0.819` for 35 degrees.
    pub ground_max_slope_cos: [f32; 2],
    /// `maximum normal force contribution`, "0 defaults to 3".
    pub maximum_normal_force: f32,
    pub antigravity_bank_lift: f32,
    pub steering_bank_reaction_scale: f32,
    /// `gravity scale`, authored. **"value of 0 defaults to 1. .5 is half gravity"**, so a read
    /// of 0.0 means UNSCALED and must not be taken as zero gravity — [`Havok::gravity_scale`]
    /// applies that rule in one place.
    pub gravity_scale_raw: f32,
    pub iteration_count: u32,
    pub suspension_count: u32,
    pub friction_points: Vec<FrictionPoint>,
    pub antigravity_points: Vec<AntiGravityPoint>,
}

impl Havok {
    /// The effective gravity scale, applying the field's own documented default.
    ///
    /// The same `0 means default` convention Bungie complained about on the sibling field
    /// `vehi/type-boat[]/gravity scale` — *"0 means 0. How hard is it to type 1?"* — a complaint
    /// that only makes sense because everywhere else in this tag set 0 is the default. The same
    /// reasoning [`crate::reach_physics`] records for `bipd` `gravity scale`.
    pub fn gravity_scale(&self) -> f32 {
        if self.gravity_scale_raw > 0.0 { self.gravity_scale_raw } else { 1.0 }
    }
    /// The two ground slope limits in DEGREES, from the stored cosines. `(0, 0)` on a hover
    /// vehicle, which authors `1.0, 1.0`.
    pub fn ground_max_slope_degrees(&self) -> [f32; 2] {
        self.ground_max_slope_cos.map(|c| c.clamp(-1.0, 1.0).acos().to_degrees())
    }
    /// `maximum normal force contribution` with its documented default applied.
    pub fn maximum_normal_force_effective(&self) -> f32 {
        if self.maximum_normal_force > 0.0 { self.maximum_normal_force } else { 3.0 }
    }
    /// The wheels' total `fraction of total mass`. Printed rather than asserted to be 1: whether
    /// Bungie made it sum is a measurement, not a requirement.
    pub fn wheel_mass_fraction_total(&self) -> f32 {
        self.friction_points.iter().map(|f| f.fraction_of_total_mass).sum()
    }
}

/// Which family, and its numbers.
#[derive(Clone, Debug)]
pub enum Family {
    /// Warthog, Mongoose, the civilian trucks, the Wolverine, the Chopper.
    Jeep { steering: Steering, turning: Turning, engine: Engine, limits: Option<DriveLimits>, gravity_adjust: f32 },
    /// Ghost.
    Scout { steering: Steering, model: VelocityModel },
    /// Scorpion, Wraith.
    Tank { diff: Differential, engine: Engine, limits: Option<DriveLimits>, gravity_adjust: f32 },
    /// Revenant: a `human_tank_struct` followed by its own velocity fields.
    Revenant { diff: Differential, engine: Engine, limits: Option<DriveLimits>, gravity_adjust: f32, model: VelocityModel },
    /// A family this module reads the presence of but does not drive. **No invented dynamics.**
    Unsupported { kind: &'static str, offset: usize, size: usize },
}

impl Family {
    pub fn kind(&self) -> &'static str {
        match self {
            Family::Jeep { .. } => "human_jeep",
            Family::Scout { .. } => "alien_scout",
            Family::Tank { .. } => "human_tank",
            Family::Revenant { .. } => "revenant",
            Family::Unsupported { kind, .. } => kind,
        }
    }
    /// The authored `gravity adjust` of the families that have one.
    pub fn gravity_adjust(&self) -> Option<f32> {
        match self {
            Family::Jeep { gravity_adjust, .. } | Family::Tank { gravity_adjust, .. } | Family::Revenant { gravity_adjust, .. } => {
                Some(*gravity_adjust)
            }
            _ => None,
        }
    }
    /// The authored first-order velocity model, where the family IS one.
    pub fn velocity_model(&self) -> Option<&VelocityModel> {
        match self {
            Family::Scout { model, .. } | Family::Revenant { model, .. } => Some(model),
            _ => None,
        }
    }
}

/// One vehicle's physics authoring.
#[derive(Clone, Debug)]
pub struct VehiclePhysics {
    pub vehi_tag: u32,
    pub family: Family,
    /// `havok vehicle physics`: ground friction, the wheels, and the authored `gravity scale`.
    pub havok: Havok,
    /// `vehicle_flags`.
    pub flags: u32,
    /// `minimum` / `maximum flipping angular velocity`.
    pub flipping_angular_velocity: [f32; 2],
    /// `seat entrance acceleration scale` / `seat exit accelersation scale` (sic) — "how much do
    /// we scale the force the vehicle applies down on the seat when he enters/exits".
    pub seat_entrance_exit_accel_scale: [f32; 2],
    /// `flip message`, a `string_id` (resolves through [`crate::reach_strings`]).
    pub flip_message: String,
    /// The friction points' MARKER POSITIONS in model space, in `havok.friction_points` order.
    /// Empty until a caller that has the render model fills it in with
    /// [`VehiclePhysics::set_wheel_positions`] — the `vehi` tag names the markers but does not
    /// hold their positions, so this module cannot resolve them on its own and does not pretend
    /// to.
    pub wheel_positions: Vec<glam::Vec3>,
    /// The ANTI-GRAVITY points' marker positions in model space, in `havok.antigravity_points`
    /// order. Filled the same way and by the same caller.
    pub antigravity_positions: Vec<glam::Vec3>,
    /// The family's own `wheel circumferance`, wu — the wrap length of the rolling phase. Zero
    /// for a family that authors none.
    pub jeep_wheel_circumference: f32,
    /// `obje +0x1C/+0x20/+0x24` — `(horizontal, vertical, angular) acceleration scale`, read
    /// through [`crate::reach_physics::acceleration_scales`] so there is one reader.
    pub acceleration_scales: (f32, f32, f32),
}

impl VehiclePhysics {
    /// Can this module drive it? False for every flyer and for the families whose authoring this
    /// module does not model — and false is the whole point: the caller refuses rather than
    /// substituting dynamics.
    pub fn drivable(&self) -> bool {
        !matches!(self.family, Family::Unsupported { .. })
    }

    /// Read a `vehi`'s physics. `None` when the tag cannot be read to the end of its own fields.
    pub fn read(native: &NativeDll, cache: CacheHandle, vehi: u32) -> Option<VehiclePhysics> {
        let own = native.tag_meta(cache, vehi, VEHI_OWN_FIELDS as u32, VEHI_OWN_SIZE as u32)?;
        let populated: Vec<(usize, &'static str, usize, bool, u32)> = TYPE_BLOCKS
            .iter()
            .filter_map(|&(off, name, size, ground)| {
                let rel = off - VEHI_OWN_FIELDS;
                let count = rd_u32(&own, rel);
                let ptr = rd_u32(&own, rel + 4);
                (count == 1 && ptr != 0).then_some((off, name, size, ground, ptr))
            })
            .collect();
        // Exactly one, by construction: every block is `[max 1]` and they are the fifteen
        // mutually exclusive physics families. More than one is a layout error, and the gate
        // asserts it rather than this reader silently taking the first.
        let &(off, name, size, _ground, ptr) = populated.first()?;
        let b = native.tag_ptr_bytes(cache, ptr, size as u32)?;
        let family = match name {
            "human_jeep" => {
                let engine = read_engine(native, cache, &b, JEEP_ENGINE);
                let wheel = rd_f32(&b, JEEP_WHEEL_CIRCUMFERANCE);
                Family::Jeep {
                    steering: read_steering(&b, JEEP_STEERING),
                    turning: read_turning(&b, JEEP_TURNING),
                    limits: DriveLimits::from_engine(&engine, wheel),
                    engine,
                    gravity_adjust: rd_f32(&b, JEEP_GRAVITY_ADJUST),
                }
            }
            "alien_scout" => Family::Scout {
                steering: read_steering(&b, SCOUT_STEERING),
                model: VelocityModel {
                    max_forward_speed: rd_f32(&b, SCOUT_MAX_FORWARD_SPEED),
                    max_reverse_speed: rd_f32(&b, SCOUT_MAX_REVERSE_SPEED),
                    speed_acceleration: rd_f32(&b, SCOUT_SPEED_ACCELERATION),
                    speed_deceleration: rd_f32(&b, SCOUT_SPEED_DECELERATION),
                    max_left_slide: rd_f32(&b, SCOUT_MAX_LEFT_SLIDE),
                    max_right_slide: rd_f32(&b, SCOUT_MAX_RIGHT_SLIDE),
                    slide_acceleration: rd_f32(&b, SCOUT_SLIDE_ACCELERATION),
                    slide_deceleration: rd_f32(&b, SCOUT_SLIDE_DECELERATION),
                    slide_speed_at_top_speed: rd_f32(&b, SCOUT_SLIDE_SPEED_AT_TOP_SPEED),
                    drag_coefficient: rd_f32(&b, SCOUT_DRAG_COEFFICIENT),
                    constant_deceleration: rd_f32(&b, SCOUT_CONSTANT_DECELERATION),
                    torque_scale: rd_f32(&b, SCOUT_TORQUE_SCALE),
                },
            },
            "human_tank" => {
                let engine = read_engine(native, cache, &b, TANK_ENGINE);
                let wheel = rd_f32(&b, TANK_WHEEL_CIRCUMFERANCE);
                Family::Tank {
                    diff: read_differential(&b, 0),
                    limits: DriveLimits::from_engine(&engine, wheel),
                    engine,
                    gravity_adjust: rd_f32(&b, TANK_GRAVITY_ADJUST),
                }
            }
            "revenant" => {
                let engine = read_engine(native, cache, &b, TANK_ENGINE);
                let wheel = rd_f32(&b, TANK_WHEEL_CIRCUMFERANCE);
                Family::Revenant {
                    diff: read_differential(&b, 0),
                    limits: DriveLimits::from_engine(&engine, wheel),
                    engine,
                    gravity_adjust: rd_f32(&b, TANK_GRAVITY_ADJUST),
                    model: VelocityModel {
                        max_forward_speed: rd_f32(&b, REV_MAX_FORWARD_SPEED),
                        max_reverse_speed: rd_f32(&b, REV_MAX_REVERSE_SPEED),
                        speed_acceleration: rd_f32(&b, REV_SPEED_ACCELERATION),
                        speed_deceleration: rd_f32(&b, REV_SPEED_DECELERATION),
                        max_left_slide: rd_f32(&b, REV_MAX_LEFT_SLIDE),
                        max_right_slide: rd_f32(&b, REV_MAX_RIGHT_SLIDE),
                        slide_acceleration: rd_f32(&b, REV_SLIDE_ACCELERATION),
                        slide_deceleration: rd_f32(&b, REV_SLIDE_DECELERATION),
                        // NOT authored in `revenant_block`, and not substituted.
                        slide_speed_at_top_speed: 0.0,
                        drag_coefficient: 0.0,
                        constant_deceleration: 0.0,
                        torque_scale: 0.0,
                    },
                }
            }
            other => Family::Unsupported { kind: leak_kind(other), offset: off, size },
        };
        let h = VEHI_HAVOK - VEHI_OWN_FIELDS;
        Some(VehiclePhysics {
            vehi_tag: vehi,
            family,
            havok: read_havok(native, cache, &own, h),
            flags: rd_u32(&own, VEHI_FLAGS - VEHI_OWN_FIELDS),
            flipping_angular_velocity: [
                rd_f32(&own, VEHI_MIN_FLIPPING_ANGULAR_VELOCITY - VEHI_OWN_FIELDS),
                rd_f32(&own, VEHI_MAX_FLIPPING_ANGULAR_VELOCITY - VEHI_OWN_FIELDS),
            ],
            seat_entrance_exit_accel_scale: [
                rd_f32(&own, VEHI_SEAT_ENTRANCE_ACCEL_SCALE - VEHI_OWN_FIELDS),
                rd_f32(&own, VEHI_SEAT_EXIT_ACCEL_SCALE - VEHI_OWN_FIELDS),
            ],
            wheel_positions: Vec::new(),
            antigravity_positions: Vec::new(),
            jeep_wheel_circumference: match name {
                "human_jeep" => rd_f32(&b, JEEP_WHEEL_CIRCUMFERANCE),
                "human_tank" | "revenant" => rd_f32(&b, TANK_WHEEL_CIRCUMFERANCE),
                _ => 0.0,
            },
            flip_message: {
                let v = rd_u32(&own, VEHI_FLIP_MESSAGE - VEHI_OWN_FIELDS);
                if v == 0 || v == u32::MAX { String::new() } else { native.string_id(cache, v).unwrap_or_default() }
            },
            acceleration_scales: crate::reach_physics::acceleration_scales(native, cache, vehi).unwrap_or((0.0, 0.0, 0.0)),
        })
    }

    /// Supply the friction-point marker positions in model space, resolved by a caller that has
    /// the render model (`reach_actor::ReachModel` + a bind pose). Order must match
    /// `havok.friction_points`; a marker the model does not carry contributes nothing, so a
    /// caller passes what it found and this module measures what it can.
    pub fn set_wheel_positions(&mut self, positions: Vec<glam::Vec3>) {
        self.wheel_positions = positions;
    }

    /// The same for the anti-gravity point markers.
    pub fn set_antigravity_positions(&mut self, positions: Vec<glam::Vec3>) {
        self.antigravity_positions = positions;
    }

    /// #vehicle-suspension  The vehicle's GROUND CONTACT POINTS, resolved: the friction points
    /// where it has them, otherwise the anti-gravity points.
    ///
    /// **Friction points come first and are never mixed with hover points.** A vehicle in
    /// `forge_halo` populates one or the other, and the two have different laws: a wheel's
    /// stand-off is its own `radius` and a hover point's is its authored `antigrav height`.
    ///
    /// A marker the render model did not carry contributes nothing, so this returns only what was
    /// actually resolved and the caller finds out by the count being short.
    pub fn contact_points(&self) -> Vec<drive::ContactPoint> {
        // The lengths must MATCH: a short vector would zip wheel 3's marker against wheel 2's
        // radius. Refusing is what makes `DriveModel::for_vehicle` print the note instead.
        if !self.havok.friction_points.is_empty() && self.wheel_positions.len() == self.havok.friction_points.len() {
            return self
                .havok
                .friction_points
                .iter()
                .zip(self.wheel_positions.iter())
                .map(|(f, &local)| drive::ContactPoint {
                    local,
                    radius: f.radius,
                    rest_height: 0.0,
                    // Measured: this field is NOT a 0..1 fraction (Warthog 6.5 per wheel,
                    // Scorpion 40.0), so it is carried raw and normalised where it is used. A
                    // point authoring zero would vanish from the load solve, so it falls back to
                    // an equal share rather than to nothing.
                    weight: if f.fraction_of_total_mass > 1e-6 { f.fraction_of_total_mass } else { 1.0 },
                    steer: drive::SteerRole::from_flags(f.flags),
                })
                .collect();
        }
        if self.antigravity_positions.len() != self.havok.antigravity_points.len() {
            return Vec::new();
        }
        self.havok
            .antigravity_points
            .iter()
            .zip(self.antigravity_positions.iter())
            .map(|(a, &local)| drive::ContactPoint {
                local,
                radius: a.radius,
                rest_height: a.height,
                weight: if a.strength > 1e-6 { a.strength } else { 1.0 },
                steer: drive::SteerRole::Fixed,
            })
            .collect()
    }

    /// Which type blocks a vehicle populates. Used by the gate to assert there is exactly one.
    pub fn populated_blocks(native: &NativeDll, cache: CacheHandle, vehi: u32) -> Vec<&'static str> {
        let Some(own) = native.tag_meta(cache, vehi, VEHI_OWN_FIELDS as u32, VEHI_OWN_SIZE as u32) else {
            return Vec::new();
        };
        TYPE_BLOCKS
            .iter()
            .filter_map(|&(off, name, _, _)| {
                let rel = off - VEHI_OWN_FIELDS;
                (rd_u32(&own, rel) == 1 && rd_u32(&own, rel + 4) != 0).then_some(name)
            })
            .collect()
    }

    /// Lines for a diagnostic: every authored number, so the numbers can be checked by eye
    /// against a game somebody has played thousands of hours of.
    pub fn describe(&self) -> Vec<String> {
        let (h, v, a) = self.acceleration_scales;
        let mut out = vec![format!(
            "  family {}  accel scales h={h:.3} v={v:.3} ang={a:.3}{}",
            self.family.kind(),
            match self.family.gravity_adjust() {
                Some(g) => format!("  gravity adjust {g:.3}"),
                None => String::new(),
            }
        )];
        match &self.family {
            Family::Jeep { steering, turning, engine, limits, .. } => {
                out.push(format!(
                    "  steering: overdampen cusp {:.2} exp {:.2} | turning: max L {:.3} R {:.3} rate {:.3}",
                    steering.overdampen_cusp_angle, steering.overdampen_exponent, turning.max_left, turning.max_right, turning.turn_rate
                ));
                out.extend(describe_engine(engine, *limits));
            }
            Family::Scout { steering, model } => {
                out.push(format!(
                    "  steering: overdampen cusp {:.2} exp {:.2}",
                    steering.overdampen_cusp_angle, steering.overdampen_exponent
                ));
                out.push(describe_model(model));
            }
            Family::Tank { diff, engine, limits, .. } => {
                out.push(describe_diff(diff));
                out.extend(describe_engine(engine, *limits));
            }
            Family::Revenant { diff, engine, limits, model, .. } => {
                out.push(describe_diff(diff));
                out.extend(describe_engine(engine, *limits));
                out.push(describe_model(model));
            }
            Family::Unsupported { kind, offset, size } => {
                out.push(format!("  NOT DRIVEN: '{kind}' at +{offset:#X} ({size} bytes) -- no dynamics invented"));
            }
        }
        let hv = &self.havok;
        out.push(format!(
            "  havok: gravity scale {:.3} (raw {:.3}), ground friction {:.3} / moving {:.3}, depth {:.3}, \
             damp {:.3}, max slope cos {:.3}/{:.3} = {:.1}/{:.1} deg, max normal force {:.2}, bank lift {:.3}, \
             steering bank {:.3}, iterations {}, suspensions {}",
            hv.gravity_scale(), hv.gravity_scale_raw, hv.ground_friction, hv.ground_moving_friction,
            hv.ground_depth, hv.ground_damp_factor, hv.ground_max_slope_cos[0], hv.ground_max_slope_cos[1],
            hv.ground_max_slope_degrees()[0], hv.ground_max_slope_degrees()[1],
            hv.maximum_normal_force_effective(), hv.antigravity_bank_lift, hv.steering_bank_reaction_scale,
            hv.iteration_count, hv.suspension_count
        ));
        if !hv.friction_points.is_empty() {
            out.push(format!(
                "  {} wheel(s), mass fractions sum to {:.3}:",
                hv.friction_points.len(),
                hv.wheel_mass_fraction_total()
            ));
            for w in &hv.friction_points {
                out.push(format!(
                    "    wheel {} '{}' r {:.3} (damaged {:.3}) mass {:.3} e-brake {:.2}/{:.2} region '{}' -> state {}",
                    w.index, w.marker_name, w.radius, w.damaged_radius, w.fraction_of_total_mass,
                    w.ebrake_friction, w.ebrake_moving_friction, w.region_name, w.model_state_destroyed
                ));
            }
        }
        if !hv.antigravity_points.is_empty() {
            out.push(format!("  {} anti-gravity point(s):", hv.antigravity_points.len()));
            for a in &hv.antigravity_points {
                out.push(format!(
                    "    agp {} '{}' strength {:.3} height {:.3} r {:.3} damping {:.3}/{:.3} k0 {:.3} k1 {:.3}",
                    a.index, a.marker_name, a.strength, a.height, a.radius,
                    a.compression_damping, a.extension_damping, a.normal_k0, a.normal_k1
                ));
            }
        }
        out.push(format!(
            "  flipping angular velocity {:.2}..{:.2}, seat entrance/exit accel scale {:.2}/{:.2}, flip message '{}'",
            self.flipping_angular_velocity[0], self.flipping_angular_velocity[1],
            self.seat_entrance_exit_accel_scale[0], self.seat_entrance_exit_accel_scale[1], self.flip_message
        ));
        out
    }
}

fn leak_kind(name: &str) -> &'static str {
    TYPE_BLOCKS.iter().find(|t| t.1 == name).map(|t| t.1).unwrap_or("unknown")
}

fn read_steering(b: &[u8], at: usize) -> Steering {
    Steering {
        overdampen_cusp_angle: rd_f32(b, at + STEER_OVERDAMPEN_CUSP_ANGLE),
        overdampen_exponent: rd_f32(b, at + STEER_OVERDAMPEN_EXPONENT),
    }
}

fn read_turning(b: &[u8], at: usize) -> Turning {
    Turning {
        max_left: rd_f32(b, at + TURN_MAX_LEFT),
        max_right: rd_f32(b, at + TURN_MAX_RIGHT),
        turn_rate: rd_f32(b, at + TURN_RATE),
    }
}

fn read_differential(b: &[u8], at: usize) -> Differential {
    Differential {
        forward_arc: rd_f32(b, at + TANK_FORWARD_ARC),
        perpendicular_forward_arc: rd_f32(b, at + TANK_PERPENDICULAR_FORWARD_ARC),
        flip_window: rd_f32(b, at + TANK_FLIP_WINDOW),
        pegged_fraction: rd_f32(b, at + TANK_PEGGED_FRACTION),
        forward_turn_scale: rd_f32(b, at + TANK_FORWARD_TURN_SCALE),
        reverse_turn_scale: rd_f32(b, at + TANK_REVERSE_TURN_SCALE),
        max_left_differential: rd_f32(b, at + TANK_MAX_LEFT_DIFFERENTIAL),
        max_right_differential: rd_f32(b, at + TANK_MAX_RIGHT_DIFFERENTIAL),
        differential_acceleration: rd_f32(b, at + TANK_DIFFERENTIAL_ACCELERATION),
        differential_deceleration: rd_f32(b, at + TANK_DIFFERENTIAL_DECELERATION),
    }
}

/// Decode a `c_function_definition` at `b[at..]` through [`crate::scene::MapFn`], the decoder
/// `scene` already uses for every other Guerilla mapping function in the game.
fn read_function(native: &NativeDll, cache: CacheHandle, b: &[u8], at: usize) -> Option<crate::scene::MapFn> {
    let size = rd_u32(b, at + FUNCTION_DATA_SIZE);
    let addr = rd_u32(b, at + FUNCTION_DATA_ADDRESS);
    if addr == 0 || size < 0x0C || size > 0x400 {
        return None;
    }
    native.tag_ptr_bytes(cache, addr, size).and_then(|d| crate::scene::MapFn::decode(&d))
}

fn read_engine(native: &NativeDll, cache: CacheHandle, b: &[u8], at: usize) -> Engine {
    let mut e = Engine {
        moment: rd_f32(b, at + ENG_MOMENT),
        max_angular_velocity: rd_f32(b, at + ENG_MAX_ANGULAR_VELOCITY),
        gears: Vec::new(),
    };
    let count = rd_u32(b, at + ENG_GEARS);
    let ptr = rd_u32(b, at + ENG_GEARS + 4);
    if count == 0 || count > 16 || ptr == 0 {
        return e;
    }
    let Some(gb) = native.tag_ptr_bytes(cache, ptr, count * GEAR_SIZE as u32) else { return e };
    for i in 0..count as usize {
        let g = &gb[i * GEAR_SIZE..(i + 1) * GEAR_SIZE];
        e.gears.push(Gear {
            index: i,
            ratio: rd_f32(g, GEAR_RATIO),
            min_time_to_upshift: rd_f32(g, GEAR_MIN_TIME_TO_UPSHIFT),
            upshift_scale: rd_f32(g, GEAR_UPSHIFT_SCALE),
            min_time_to_downshift: rd_f32(g, GEAR_MIN_TIME_TO_DOWNSHIFT),
            downshift_scale: rd_f32(g, GEAR_DOWNSHIFT_SCALE),
            loaded_torque: read_function(native, cache, g, GEAR_LOADED_TORQUE),
            cruising_torque: read_function(native, cache, g, GEAR_CRUISING_TORQUE),
        });
    }
    e
}

fn read_havok(native: &NativeDll, cache: CacheHandle, own: &[u8], at: usize) -> Havok {
    let sid = |v: u32| if v == 0 || v == u32::MAX { String::new() } else { native.string_id(cache, v).unwrap_or_default() };
    let mut hv = Havok {
        flags: rd_u32(own, at + HV_FLAGS),
        ground_friction: rd_f32(own, at + HV_GROUND_FRICTION),
        ground_depth: rd_f32(own, at + HV_GROUND_DEPTH),
        ground_damp_factor: rd_f32(own, at + HV_GROUND_DAMP_FACTOR),
        ground_moving_friction: rd_f32(own, at + HV_GROUND_MOVING_FRICTION),
        ground_max_slope_cos: [rd_f32(own, at + HV_GROUND_MAX_SLOPE_0), rd_f32(own, at + HV_GROUND_MAX_SLOPE_1)],
        maximum_normal_force: rd_f32(own, at + HV_MAX_NORMAL_FORCE),
        antigravity_bank_lift: rd_f32(own, at + HV_ANTIGRAVITY_BANK_LIFT),
        steering_bank_reaction_scale: rd_f32(own, at + HV_STEERING_BANK_REACTION_SCALE),
        gravity_scale_raw: rd_f32(own, at + HV_GRAVITY_SCALE),
        iteration_count: rd_u32(own, at + HV_ITERATION_COUNT),
        suspension_count: rd_u32(own, at + HV_SUSPENSION_COUNT),
        friction_points: Vec::new(),
        antigravity_points: Vec::new(),
    };
    let fc = rd_u32(own, at + HV_FRICTION_POINTS);
    let fp = rd_u32(own, at + HV_FRICTION_POINTS + 4);
    if fc > 0 && fc <= 16 && fp != 0 {
        if let Some(b) = native.tag_ptr_bytes(cache, fp, fc * FP_SIZE as u32) {
            for i in 0..fc as usize {
                let e = &b[i * FP_SIZE..(i + 1) * FP_SIZE];
                hv.friction_points.push(FrictionPoint {
                    index: i,
                    marker_name: sid(rd_u32(e, FP_MARKER_NAME)),
                    flags: rd_u32(e, FP_FLAGS),
                    fraction_of_total_mass: rd_f32(e, FP_FRACTION_OF_TOTAL_MASS),
                    radius: rd_f32(e, FP_RADIUS),
                    damaged_radius: rd_f32(e, FP_DAMAGED_RADIUS),
                    moving_friction_velocity_diff: rd_f32(e, FP_MOVING_FRICTION_VELOCITY_DIFF),
                    ebrake_friction: rd_f32(e, FP_EBRAKE_FRICTION),
                    ebrake_moving_friction: rd_f32(e, FP_EBRAKE_MOVING_FRICTION),
                    model_state_destroyed: {
                        let o = FP_MODEL_STATE_DESTROYED;
                        if o + 2 <= e.len() { i16::from_le_bytes([e[o], e[o + 1]]) } else { 0 }
                    },
                    region_name: sid(rd_u32(e, FP_REGION_NAME)),
                });
            }
        }
    }
    let ac = rd_u32(own, at + HV_ANTIGRAVITY_POINTS);
    let ap = rd_u32(own, at + HV_ANTIGRAVITY_POINTS + 4);
    if ac > 0 && ac <= 16 && ap != 0 {
        if let Some(b) = native.tag_ptr_bytes(cache, ap, ac * AGP_SIZE as u32) {
            for i in 0..ac as usize {
                let e = &b[i * AGP_SIZE..(i + 1) * AGP_SIZE];
                hv.antigravity_points.push(AntiGravityPoint {
                    index: i,
                    marker_name: sid(rd_u32(e, AGP_MARKER_NAME)),
                    flags: rd_u32(e, AGP_FLAGS),
                    damping: rd_u32(e, AGP_DAMPING),
                    strength: rd_f32(e, AGP_STRENGTH),
                    height: rd_f32(e, AGP_HEIGHT),
                    compression_damping: rd_f32(e, AGP_COMPRESSION_DAMPING),
                    extension_damping: rd_f32(e, AGP_EXTENSION_DAMPING),
                    normal_k0: rd_f32(e, AGP_NORMAL_K0),
                    normal_k1: rd_f32(e, AGP_NORMAL_K1),
                    radius: rd_f32(e, AGP_RADIUS),
                });
            }
        }
    }
    hv
}

fn describe_engine(e: &Engine, limits: Option<DriveLimits>) -> Vec<String> {
    let mut out = vec![format!(
        "  engine: moment {:.3}, max angular velocity {:.3}, {} gear(s){}",
        e.moment,
        e.max_angular_velocity,
        e.gears.len(),
        match limits {
            Some(l) => format!(
                "; wheel circumferance {:.3} -> kinematic top-speed CEILING {:.2} wu/s (top gear {} ratio {:.3})",
                l.wheel_circumferance, l.top_speed_ceiling, l.top_gear, l.top_gear_ratio
            ),
            None => "; NO top-speed ceiling computable (no positive gear ratio or no wheel circumferance)".to_string(),
        }
    )];
    for g in &e.gears {
        out.push(format!(
            "    gear {}: ratio {:.4}, up {:.2}s x{:.2}, down {:.2}s x{:.2}, torque curves loaded={} cruising={}",
            g.index,
            g.ratio,
            g.min_time_to_upshift,
            g.upshift_scale,
            g.min_time_to_downshift,
            g.downshift_scale,
            g.loaded_torque.is_some(),
            g.cruising_torque.is_some()
        ));
    }
    out
}

fn describe_model(m: &VelocityModel) -> String {
    format!(
        "  velocity: fwd {:.2} rev {:.2} wu/s, accel {:.2} decel {:.2}, slide L {:.2} R {:.2} \
         (accel {:.2} decel {:.2}, at top speed {:.2}), drag {:.3}, constant decel {:.3}, torque scale {:.2}",
        m.max_forward_speed,
        m.max_reverse_speed,
        m.speed_acceleration,
        m.speed_deceleration,
        m.max_left_slide,
        m.max_right_slide,
        m.slide_acceleration,
        m.slide_deceleration,
        m.slide_speed_at_top_speed,
        m.drag_coefficient,
        m.constant_deceleration,
        m.torque_scale
    )
}

fn describe_diff(d: &Differential) -> String {
    format!(
        "  differential: forward arc {:.4} (perp {:.4}), flip window {:.2}, pegged {:.2}, \
         turn scale fwd {:.2} rev {:.2}, max diff L {:.2} R {:.2}, diff accel {:.2} decel {:.2}",
        d.forward_arc,
        d.perpendicular_forward_arc,
        d.flip_window,
        d.pegged_fraction,
        d.forward_turn_scale,
        d.reverse_turn_scale,
        d.max_left_differential,
        d.max_right_differential,
        d.differential_acceleration,
        d.differential_deceleration
    )
}

// ---------------------------------------------------------------------------------------------
// #vehicle-drive  THE MODEL: the simplest thing the authored parameters genuinely support
// ---------------------------------------------------------------------------------------------

/// Where a ground vehicle's forward SPEED SCALE came from, carried with the number so a caption
/// can say it rather than a reader having to trust it.
///
/// # The three candidates that are NOT a top speed, measured and rejected
///
/// `human_jeep` authors no `maximum forward speed`, so the obvious thing is to hunt for one.
/// Three fields look like they might be it and none is:
///
/// * **`vehi +0x600 blur speed`** — Warthog `0.550`, Scorpion `0.550`, Mongoose `2.400`,
///   **Ghost `0.000`**. A vehicle with an authored 5.0 wu/s top speed authoring zero settles it:
///   this is a blur amount, not a speed.
/// * **`unit +0x33C object velocity maximum`** — reads `0.000` on every vehicle in the cache
///   (the field's own documentation says "0 defaults to 5.0", and it is a *screen-blend* debug
///   value in any case).
/// * **the engine-and-gearbox kinematic ceiling** — `9.5 x 1.2 x 1.249 = 14.2 wu/s` for a
///   Warthog, against the Ghost's directly authored `5.0`. A Reach Warthog is not nearly three
///   times a Ghost, so the no-load figure overestimates and [`DriveLimits`] states it as a
///   ceiling only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeedSource {
    /// The type block's own `maximum forward speed` / `maximum reverse speed`. The `alien_scout`
    /// family (Ghost, Wraith, Revenant) authors these outright, so nothing is chosen.
    Authored,
    /// The driver seat's `max speed reference`.
    ///
    /// **This is a CHOICE and it is the only one in this module.** The field is documented as the
    /// engine's 0..1 normalisation reference for the speed-dependent turn rates — *"the max speed
    /// reference is what the code uses to generate a clamped speed from 0..1"* — **not** as a top
    /// speed. It is used here because it is the only authored per-vehicle speed-like quantity the
    /// `human_jeep` and `human_tank` families have, and because the numbers it gives (Warthog
    /// 9.0, Mongoose 7.0, against the Ghost's authored 5.0) are in the right register where the
    /// kinematic ceiling's 14.2 is not. A caller must print this provenance.
    SeatMaxSpeedReference,
}

impl SpeedSource {
    pub fn label(self) -> &'static str {
        match self {
            SpeedSource::Authored => "the type block's own `maximum forward speed`",
            SpeedSource::SeatMaxSpeedReference => {
                "the driver seat's `max speed reference` -- the engine's 0..1 normalisation \
                 reference, NOT an authored top speed"
            }
        }
    }
}

/// The authored speed-dependent turn rate, straight out of the DRIVER SEAT's own
/// `speed dependant turn rates` group, with the block's own documentation as the law:
///
/// > when the unit velocity is 0, the yaw/pitch rates are the left values.
/// > at \[max speed reference\], the yaw/pitch rates are the right values.
/// > the max speed reference is what the code uses to generate a clamped speed from 0..1.
/// > the exponent controls how midrange speeds are interpreted.
///
/// **The rates are in DEGREES per second**, which is a measurement rather than a reading of the
/// label: the field's type is `real bounds` (not `angle bounds`, which would be radians) and the
/// authored magnitudes are degrees — Ghost `[50.0, 16.67]`, Warthog `[70.0, 50.0]`, Scorpion
/// `[30.0, 30.0]`. A radians reading would turn a Ghost at 50 rad/s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TurnRate {
    /// `yaw rate bounds`, degrees per second: at rest, and at `max speed reference`.
    pub yaw_deg: [f32; 2],
    /// `pitch rate bounds`, degrees per second.
    pub pitch_deg: [f32; 2],
    /// `max speed reference`, wu/s.
    pub max_speed_reference: f32,
    /// `speed exponent`.
    pub exponent: f32,
}

impl TurnRate {
    /// Read it off a seat.
    pub fn from_seat(seat: &crate::reach_seats::Seat) -> TurnRate {
        TurnRate {
            yaw_deg: seat.yaw_rate_bounds,
            pitch_deg: seat.pitch_rate_bounds,
            max_speed_reference: seat.max_speed_reference,
            exponent: seat.speed_exponent,
        }
    }
    /// The 0..1 interpolant the engine's own description defines.
    pub fn t(&self, speed: f32) -> f32 {
        if !(self.max_speed_reference > 0.0) {
            return 0.0;
        }
        let x = (speed.abs() / self.max_speed_reference).clamp(0.0, 1.0);
        if self.exponent > 0.0 { x.powf(self.exponent) } else { x }
    }
    /// Yaw rate at this speed, RADIANS per second (the authored degrees, converted once here so
    /// no caller has to remember).
    pub fn yaw_rate(&self, speed: f32) -> f32 {
        let t = self.t(speed);
        (self.yaw_deg[0] + (self.yaw_deg[1] - self.yaw_deg[0]) * t).to_radians()
    }
    /// Pitch rate at this speed, radians per second.
    pub fn pitch_rate(&self, speed: f32) -> f32 {
        let t = self.t(speed);
        (self.pitch_deg[0] + (self.pitch_deg[1] - self.pitch_deg[0]) * t).to_radians()
    }
}

/// What the driver is asking for this frame. All three are `-1..1` or a flag; nothing here is a
/// speed, so a caller never has to know a vehicle's numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DriveInput {
    /// Forward / reverse, `-1..1`.
    pub throttle: f32,
    /// Strafe, `-1..1` (left positive, as `+Y` is left). Only the hover families slide.
    pub slide: f32,
    /// Where the driver wants to face, world radians. The hover families turn toward it; the
    /// wheeled families steer toward it.
    pub want_yaw: f32,
    pub brake: bool,
}

/// A ground vehicle's integrated state: everything the model owns and nothing it does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriveState {
    pub pos: glam::Vec3,
    /// World velocity, wu/s.
    pub vel: glam::Vec3,
    /// Facing, world radians.
    pub yaw: f32,
    /// #vehicle-suspension  Nose-up pitch, radians. Part of the hull's basis:
    /// `Rz(yaw) * Ry(-pitch) * Rx(roll)`.
    ///
    /// **This is what "a Warthog drives up a hill perfectly level" was missing.** It is not a
    /// cosmetic tilt: it comes out of the per-wheel spring torques, so it is also the lean under
    /// braking and the nose-up on acceleration.
    pub pitch: f32,
    /// #vehicle-suspension  Left-side-up roll, radians.
    pub roll: f32,
    /// Angular rate of [`DriveState::pitch`], rad/s.
    pub pitch_rate: f32,
    /// Angular rate of [`DriveState::roll`], rad/s.
    pub roll_rate: f32,
    /// #vehicle-suspension  Per contact point, the spring compression in wu, in
    /// `havok.friction_points` order (or `antigravity_points` order for a hover vehicle).
    ///
    /// **The quantity the `suspension:*` overlays want**, and the reason they were previously
    /// declined: nothing computed it.
    pub suspension: [f32; drive::MAX_CONTACT_POINTS],
    /// Per contact point, whether it touched the ground last step.
    pub touching: [bool; drive::MAX_CONTACT_POINTS],
    /// How many contact points are valid in the two arrays above.
    pub contact_count: u8,
    /// #vehicle-suspension  The ground normal the hull is conforming to: the plane through the
    /// first three touching contacts, which is the engine's own rule.
    pub ground_normal: glam::Vec3,
    /// #vehicle-drivetrain  The engine: flywheel speed, gear, clutch and the two shift timers.
    pub engine: drive::EngineState,
    /// #vehicle-drivetrain  Signed distance the WHEELS have rolled, wu — `gear ratio * engine
    /// angular velocity` integrated, which is what the engine itself wraps at `wheel
    /// circumferance`. Differs from [`DriveState::odometer`] exactly when the wheels slip.
    pub wheel_travel: f32,
    /// The wheeled families' current steering angle, radians (0 for a hover vehicle).
    pub steer_angle: f32,
    /// Whether the last step found ground under the vehicle.
    pub grounded: bool,
    /// #vehicle-anim  Signed distance travelled along the vehicle's own forward, world units.
    ///
    /// **This is the wheel phase's only input.** A rolling wheel turns once per `2*pi*r`, so the
    /// animation side needs distance and not speed — integrating speed in the renderer instead
    /// would make the wheels' rate follow the frame rate. Signed, so reversing spins them
    /// backwards; monotone only while the vehicle is going one way, which is what a wrapping
    /// phase wants. See [`crate::reach_vehicle_anim`].
    pub odometer: f32,
    /// #vehicle-anim  Seconds the ride has been stepped — the clock the vehicle's own `any:idle`
    /// base clip runs on (a Ghost's hover bob, a Warthog's engine shake).
    ///
    /// **Accumulated `dt`, not a frame count**, for the reason `play_mode::clock` gives: Reach's
    /// clips are authored at 30 fps against the wall clock, so driving one off the frame counter
    /// plays it at half speed on a 30 Hz display and at double on a 120 Hz one.
    pub time: f32,
}

impl Default for DriveState {
    /// Written out rather than derived because two fields have a meaningful non-zero rest value:
    /// the ground normal is UP, and the engine starts in neutral with no shift pending.
    fn default() -> DriveState {
        DriveState {
            pos: glam::Vec3::ZERO,
            vel: glam::Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            pitch_rate: 0.0,
            roll_rate: 0.0,
            suspension: [0.0; drive::MAX_CONTACT_POINTS],
            touching: [false; drive::MAX_CONTACT_POINTS],
            contact_count: 0,
            ground_normal: glam::Vec3::Z,
            engine: drive::EngineState::default(),
            wheel_travel: 0.0,
            steer_angle: 0.0,
            grounded: false,
            odometer: 0.0,
            time: 0.0,
        }
    }
}

impl DriveState {
    /// #vehicle-suspension  The hull's rotation: `Rz(yaw) * Ry(-pitch) * Rx(roll)`.
    ///
    /// One function so that the renderer, the seat camera, the wheel solve and the gates cannot
    /// disagree about which way the hull is facing.
    pub fn basis(&self) -> glam::Quat {
        glam::Quat::from_rotation_z(self.yaw)
            * glam::Quat::from_rotation_y(-self.pitch)
            * glam::Quat::from_rotation_x(self.roll)
    }
    /// The hull's forward axis in the world, including pitch and roll — so a Warthog on a slope
    /// accelerates ALONG the slope instead of into it.
    pub fn forward(&self) -> glam::Vec3 {
        self.basis() * glam::Vec3::X
    }
    /// The hull's left axis in the world.
    pub fn left(&self) -> glam::Vec3 {
        self.basis() * glam::Vec3::Y
    }
}

/// The drive model, built once per vehicle from its tag.
///
/// # What this model DOES
///
/// * forward and reverse speed targets approached at the authored `speed acceleration` /
///   `speed deceleration`;
/// * sliding for the hover families, with the authored `slide speed at top speed`
///   interpolation applied as the field documents it;
/// * turning at the DRIVER SEAT's authored speed-dependent `yaw rate bounds`, through
///   `max speed reference` and `speed exponent`;
/// * for the wheeled families, a steering ANGLE that slews toward the driver's intent at the
///   authored `turn rate` and is bounded by `maximum left turn` / `maximum right turn`, turned
///   into a yaw rate by a bicycle model whose wheelbase is **measured from the vehicle's own
///   friction-point markers** rather than guessed;
/// * gravity from [`crate::reach_physics::vehicle_fall_acceleration`] — the engine's authored
///   world constant, which was MEASURED in the vehicle's own physics path — scaled by the
///   authored havok `gravity scale`;
/// * `drag coeficient` as a quadratic term and `constant deceleration` as a constant one.
///
/// * #vehicle-suspension  **TERRAIN CONFORMANCE AND SUSPENSION.** One ground ray per `friction
///   point`, so each wheel's own contact height is known; the hull's pitch and roll come from the
///   plane through the first three touching contacts, which is the engine's own rule
///   (`vehicle_type_human_jeep.cpp`, `sub_140E69D40`). Each wheel then carries a spring-damper of
///   the engine's own form — `g * compression / ground depth`, damped by `ground damp factor`,
///   capped at `maximum normal force contribution` — whose torques about the WEIGHTED CENTROID of
///   the contact points give the pitch and roll rates. So the hull squats under load, leans in a
///   corner, dives under braking, and lands on its springs; and the per-wheel compression is a
///   real number, which is what the `suspension:*` overlays were waiting for.
/// * #vehicle-drivetrain  **THE DRIVETRAIN**, for every family that authors an engine. Gear
///   selection, the two shift timers, the 2-tick clutch, the loaded/cruising torque blend and the
///   flywheel integration are the engine's, function by function — see [`drive`]. A Warthog
///   builds speed through its gears instead of lerping to a constant.
///
/// # What it does NOT do, stated once and not hidden
///
/// * **No Havok rigid body.** The hull is not a free body: it has no yaw inertia, cannot be
///   flipped by a collision, and its pitch and roll are driven only by the contact springs. There
///   is no tyre friction CIRCLE either — longitudinal traction is bounded by `ground friction`
///   and lateral grip is not bounded at all, so a Warthog will not slide out of a corner.
/// * **The ground torque is derived, not read.** The engine reads the Havok wheels' summed
///   reaction (`sub_140E60130`); this derives it from the no-slip identity instead
///   ([`drive::Drivetrain::ground_torque_for_resistance`]). The forward path — torque curve,
///   gearbox, clutch, flywheel — is the engine's; the loop closure is not.
/// * **No anti-gravity SPRING**, though the points are now used for the geometry. Measured
///   (`probe_how_many_contact_points_each_vehicle_authors`): every hover vehicle authors `ground
///   depth = 0` and `ground damp factor = 0`, so the WHEEL spring is correctly inert for them —
///   but they do author anti-gravity points (Ghost 3, Wraith 6, Revenant 6, Falcon 3), and those
///   markers are used as the contact set, so a Ghost conforms to the terrain through its own
///   hover points. What is NOT modelled is the hover spring itself: `antigrav strength`,
///   `normal k0`/`k1` and the two damping fields are read, reported and not integrated, so a
///   Ghost is held at its authored `antigrav height` above the plane rather than bobbing on it.
/// * **The flyers are not modelled at all** — [`DriveModel::for_vehicle`] returns `Err` with the
///   family name for them.
/// * **`ground moving friction` and `ground maximum slope 0` are still not consumed.** `ground
///   friction` bounds the traction, `ground depth` and `ground damp factor` are the spring, and
///   `ground maximum slope 1` is the traction cut-off (all measured) — but the engine's use of
///   the *moving* friction and of the first slope limit has not been read, so they are reported
///   and nothing is invented for them.
/// #vehicle-law  HOW A FAMILY'S HEADING CHANGES. One law per family, dispatched on the authored
/// physics type — never a default.
///
/// Reported: *"I turn and the whole warthog pivots around on a dime, the game does NOT do that.
/// Warthogs/mongooses have the jeep engine physics and that is what they use."* The kinematics
/// below were already right; what was wrong was that every family was given the same COMMAND —
/// `want_yaw`, the camera's absolute heading — so the hull chased the camera and the steer angle
/// pinned to full lock until it caught up. **A heading servo wrapped around a correct kinematic
/// model looks on screen exactly like a broken kinematic model.** That is why this is an enum: the
/// command and the law have to travel together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TurnLaw {
    /// A STEERED-WHEEL vehicle (`human_jeep`): the lateral stick positions the road wheels and
    /// the heading follows only because the wheels are rolling.
    ///
    /// `omega = v * tan(delta) / L`, with `L` the wheelbase measured from the vehicle's own
    /// friction points. **At `v = 0`, `omega = 0`** — no amount of stick turns a parked Warthog,
    /// which is most of what the dime-pivot complaint was about. It also makes reversing with the
    /// stick held turn the vehicle the other way, for free, because `v` is signed.
    Bicycle {
        /// wu, from the friction points.
        wheelbase: f32,
    },
    /// A TRACKED vehicle (`human_tank`): the stick sets a track-speed DIFFERENCE, so it can turn
    /// about its own axis — but through the tracks, not through a yaw rate.
    ///
    /// `omega = dv / track`, with `dv` the authored `maximum left/right differential` and `track`
    /// the lateral spread of the friction points.
    ///
    /// **DERIVED FROM THE FIELD NAMES AND THE MEASURED TRACK, NOT DISASSEMBLED.** `maximum left
    /// differential` is read as a track-speed difference in wu/s because that is what a
    /// differential is and what its units must be for the division to make sense; nothing in
    /// `haloreach.dll` was read to confirm it.
    Differential {
        /// wu, from the friction points.
        track: f32,
    },
    /// A HOVER vehicle (`alien_scout`): the hull turns toward where the driver is looking, at the
    /// driver seat's own authored `yaw rate bounds`. A Ghost really does turn its hull directly,
    /// so the servo belongs here — and only here.
    HeadingServo,
}

impl TurnLaw {
    pub fn label(&self) -> String {
        match self {
            TurnLaw::Bicycle { wheelbase } => format!("bicycle, wheelbase {wheelbase:.4} wu (omega = v*tan(d)/L)"),
            TurnLaw::Differential { track } => format!("differential, track {track:.4} wu (omega = dv/track)"),
            TurnLaw::HeadingServo => "heading servo (the hull turns toward the look)".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DriveModel {
    pub kind: &'static str,
    /// Forward and reverse speed, wu/s, with where they came from.
    pub max_forward_speed: f32,
    pub max_reverse_speed: f32,
    pub speed_source: SpeedSource,
    pub speed_acceleration: f32,
    pub speed_deceleration: f32,
    /// The hover families only; zero elsewhere.
    pub slide: VelocityModel,
    /// The wheeled families' steering authoring; `None` for a hover vehicle, which turns at the
    /// seat's own rate instead.
    pub turning: Option<Turning>,
    /// Measured from the friction points: the longest front-to-back separation, wu. `None` when
    /// the vehicle authors fewer than two.
    pub wheelbase: Option<f32>,
    /// #vehicle-law  Which law turns this vehicle, and the measured quantity it needs.
    pub law: TurnLaw,
    /// #vehicle-law  Anything the law had to fall back on, in the vehicle's own terms. Surfaced
    /// rather than swallowed: a jeep with no measurable wheelbase must SAY it is not being turned
    /// by the bicycle model, because that is how a Warthog came to be turned like a Ghost.
    pub law_note: Option<String>,
    /// The driver seat's authored turn-rate law.
    pub turn: TurnRate,
    /// #gravity  Authored havok `gravity scale`, default applied. **The only scale on gravity in
    /// this module**, multiplied into [`crate::reach_physics::vehicle_fall_acceleration`] once
    /// per sub-step and handed to the contact solve — so setting it really does change the fall.
    pub gravity_scale: f32,
    /// How far the hull rides above the ground: for a hover vehicle the largest authored
    /// `antigrav height`, for a wheeled one the largest wheel `radius`. Both authored.
    ///
    /// **Only a fallback now.** When [`DriveModel::contacts`] resolved, the ride height comes out
    /// of the per-point solve instead and this number is not used for the pose.
    pub ride_height: f32,
    /// #vehicle-drivetrain  The engine, gearbox and clutch. `None` for a family that authors no
    /// engine (every hover vehicle), and then the speed comes from the authored velocity model
    /// instead — which is what those families have and the geared ones do not.
    pub drivetrain: Option<drive::Drivetrain>,
    /// #vehicle-suspension  The contact points, with the load centroid and the two inertia radii
    /// derived from them. `None` when the vehicle resolved fewer than three markers, and then
    /// `law_note` says the hull is not conforming to anything.
    pub contacts: Option<drive::ContactSolver>,
    /// #vehicle-suspension  The authored ground spring.
    pub spring: drive::GroundSpring,
    /// `havok ground friction`. The traction coefficient the hull's speed chases the wheels'
    /// surface speed at: Warthog 3.370, Mongoose 3.350, Scorpion 0.950. Authored.
    pub ground_friction: f32,
}

impl DriveModel {
    /// Build the model for a vehicle, or refuse with the reason.
    ///
    /// `driver` is the vehicle's driver seat (for the authored turn-rate law). `Err` when the
    /// family is one this module does not drive, or when no speed can be had at all.
    pub fn for_vehicle(p: &VehiclePhysics, driver: Option<&crate::reach_seats::Seat>) -> Result<DriveModel, String> {
        let turn = driver.map(TurnRate::from_seat).unwrap_or_default();
        let hv = &p.havok;
        let wheelbase = wheelbase_from(p);
        let kind = p.family.kind();
        let (max_forward_speed, max_reverse_speed, speed_source, accel, decel, slide, turning) = match &p.family {
            Family::Scout { model, .. } | Family::Revenant { model, .. } => (
                model.max_forward_speed,
                model.max_reverse_speed,
                SpeedSource::Authored,
                model.speed_acceleration,
                model.speed_deceleration,
                *model,
                None,
            ),
            Family::Jeep { turning, .. } => {
                let v = turn.max_speed_reference;
                if !(v > 0.0) {
                    return Err(format!(
                        "{kind} authors no forward speed and its driver seat authors no `max speed \
                         reference` either, so there is no authored speed to drive it at"
                    ));
                }
                // A wheeled vehicle authors no acceleration either. The ACCELERATION is therefore
                // the one quantity this model has to produce, and it is produced from two
                // authored numbers rather than typed: reaching the speed scale in the time the
                // bottom gear is allowed to hold before it upshifts.
                let t = first_upshift_time(&p.family).unwrap_or(1.0);
                (v, v * 0.5, SpeedSource::SeatMaxSpeedReference, v / t.max(0.1), v / t.max(0.1), VelocityModel::default(), Some(*turning))
            }
            Family::Tank { diff, .. } => {
                let v = turn.max_speed_reference;
                if !(v > 0.0) {
                    return Err(format!("{kind} authors no forward speed and no `max speed reference`"));
                }
                let t = first_upshift_time(&p.family).unwrap_or(1.0);
                (
                    v,
                    v * 0.5,
                    SpeedSource::SeatMaxSpeedReference,
                    v / t.max(0.1),
                    v / t.max(0.1),
                    VelocityModel::default(),
                    // A tank steers by its tracks, not by a steering angle: its differential
                    // limits stand in for the turn bounds and the seat's rate law does the rest.
                    Some(Turning {
                        max_left: diff.max_left_differential,
                        max_right: -diff.max_right_differential,
                        turn_rate: diff.differential_acceleration,
                    }),
                )
            }
            // #vehicle-law  A TURRET IS NOT A GAP. 18 of the 36 shipped `vehi` tags are this
            // type — every Shade, every Falcon gun, the machinegun, and all four of the
            // Warthog's own gun positions — and a turret is SUPPOSED not to move. Saying "not
            // modelled" about it would read as a missing feature when it is the correct
            // behaviour, so it says what the rider actually gets instead.
            Family::Unsupported { kind: "turret", .. } => {
                return Err(
                    "a turret does not drive: it is bolted down, and that is the engine's own \
                     behaviour rather than a gap. The rider gets the seat's own authored camera \
                     and the turret's `vehicle:aiming` grid follows where they look."
                        .to_string(),
                )
            }
            // #vehicle-law  A FLYER, and the refusal names what is missing rather than just
            // saying no. Each of these authors a block this module has located and sized but
            // whose fields are not decoded, so the honest statement is "the numbers exist and
            // nobody has read them yet" -- which is a different thing from "this cannot be done".
            Family::Unsupported { kind, offset, size } => {
                let needed = match *kind {
                    "vtol" => Some("the Falcon's own `vtol_block` (208 bytes): its lift, thrust and attitude authoring"),
                    "alien_fighter" => Some("the Banshee's own `alien_fighter_block` (100 bytes)"),
                    "human_plane" => Some("the Pelican's and Phantom's `human_plane_block` (76 bytes)"),
                    "space-fighter" => Some("the Sabre's and Seraph's `space_fighter_block` (196 bytes)"),
                    _ => None,
                };
                return Err(match needed {
                    Some(what) => format!(
                        "'{kind}' FLIES and this build does not fly it. Nothing is substituted: it \
                         sits where you left it rather than being driven by a ground law that would \
                         be wrong. What is missing is {what}, at vehi +{offset:#X} ({size} bytes), \
                         which is located and sized but not decoded. It authors {} friction point(s) \
                         and {} anti-gravity point(s)",
                        hv.friction_points.len(),
                        hv.antigravity_points.len()
                    ),
                    None => format!(
                        "'{kind}' is not a ground family this module drives -- its dynamics are not \
                         modelled and nothing is substituted for them (vehi +{offset:#X}, \
                         {size} bytes)"
                    ),
                });
            }
        };
        // #vehicle-law  ONE LAW PER FAMILY, dispatched on the authored physics type, with every
        // fallback recorded rather than silently taken.
        let (law, law_note) = match &p.family {
            Family::Jeep { .. } => match wheelbase {
                Some(wb) => (TurnLaw::Bicycle { wheelbase: wb }, None),
                None => (
                    TurnLaw::HeadingServo,
                    Some(format!(
                        "{kind} authors {} friction point(s), so no wheelbase could be measured and \
                         this vehicle is being turned by the HOVER law -- it will pivot like a Ghost",
                        hv.friction_points.len()
                    )),
                ),
            },
            Family::Tank { .. } => match track_from(p) {
                Some(tr) => (TurnLaw::Differential { track: tr }, None),
                None => (
                    TurnLaw::HeadingServo,
                    Some(format!(
                        "{kind} authors {} friction point(s), so no track width could be measured \
                         and this vehicle is being turned by the HOVER law",
                        hv.friction_points.len()
                    )),
                ),
            },
            Family::Scout { .. } | Family::Revenant { .. } => (TurnLaw::HeadingServo, None),
            Family::Unsupported { .. } => (TurnLaw::HeadingServo, None),
        };
        // Ride height: the largest authored antigrav height, else the largest wheel radius. Only
        // the fallback for a vehicle whose markers did not resolve.
        let ride_height = hv
            .antigravity_points
            .iter()
            .map(|a| a.height)
            .chain(hv.friction_points.iter().map(|f| f.radius))
            .fold(0.0f32, f32::max);
        // #vehicle-drivetrain  The engine and its gearbox, where the family has one.
        let drivetrain = match &p.family {
            Family::Jeep { engine, .. } => drive::Drivetrain::new(engine, p.jeep_wheel_circumference),
            Family::Tank { engine, .. } | Family::Revenant { engine, .. } => {
                drive::Drivetrain::new(engine, p.jeep_wheel_circumference)
            }
            _ => None,
        };
        // #vehicle-suspension  The contact points, from the markers the caller resolved.
        let contacts = drive::ContactSolver::new(p.contact_points());
        let mut law_note = law_note;
        if contacts.is_none() && law_note.is_none() {
            law_note = Some(format!(
                "{kind} resolved {} contact marker(s) of {} friction + {} anti-gravity point(s), \
                 so there is no plane to sit on: this hull is NOT conforming to the ground and is \
                 held at a flat {ride_height:.3} wu instead",
                p.wheel_positions.len(),
                hv.friction_points.len(),
                hv.antigravity_points.len()
            ));
        }
        let spring = drive::GroundSpring {
            depth: hv.ground_depth,
            damp: hv.ground_damp_factor,
            max_normal_force: hv.maximum_normal_force_effective(),
            slope_cos: hv.ground_max_slope_cos,
            iterations: hv.iteration_count.max(1),
        };
        Ok(DriveModel {
            kind,
            max_forward_speed,
            max_reverse_speed,
            speed_source,
            speed_acceleration: accel,
            speed_deceleration: decel,
            slide,
            turning,
            wheelbase,
            law,
            law_note,
            turn,
            gravity_scale: hv.gravity_scale(),
            ride_height,
            drivetrain,
            contacts,
            spring,
            ground_friction: hv.ground_friction,
        })
    }

    /// One step. `ground` answers "what is the ground height and normal under this point", which
    /// is the caller's business (the level grid plus the Forge objects, the same world the
    /// character controller uses). `None` from it means no ground, and the vehicle falls.
    ///
    /// It is called ONCE PER CONTACT POINT, not once for the hull — that is what makes a Warthog
    /// sit on the ground it is on rather than level on top of a hill.
    ///
    /// Returns the new state. Pure: no interior mutability, no globals, nothing read but its
    /// arguments — so a gate can drive a Ghost for ten seconds and assert where it got to.
    pub fn step(
        &self,
        st: DriveState,
        input: DriveInput,
        dt: f32,
        ground: &dyn Fn(glam::Vec3) -> Option<(f32, glam::Vec3)>,
    ) -> DriveState {
        use glam::Vec3;
        let dt = dt.clamp(0.0, 0.1);
        // #vehicle-suspension  SUB-STEPPING. The engine does this itself: `sub_1406CF0F0` runs
        // its whole body `iteration count` times with `dt / n`. Every vehicle in `forge_halo`
        // authors 1, so the authored count alone would not sub-step at all — and a spring as
        // stiff as `gravity / ground depth` (33 /s^2 on a Warthog) is not stable at a 100 ms
        // frame. So the authored count is honoured as a FLOOR and the step is additionally cut
        // to 1/120 s, which is a numerical choice and changes no law: halving the sub-step again
        // must not move where the vehicle ends up, and a gate asserts that.
        let want = (dt * 120.0).ceil() as u32;
        let n = self.spring.iterations.max(want).max(1).min(32);
        let h = dt / n as f32;
        let mut st = st;
        for _ in 0..n {
            st = self.substep(st, input, h, ground);
        }
        st
    }

    /// One sub-step of [`DriveModel::step`]. Everything below is per `h`, never per frame.
    fn substep(
        &self,
        st: DriveState,
        input: DriveInput,
        dt: f32,
        ground: &dyn Fn(glam::Vec3) -> Option<(f32, glam::Vec3)>,
    ) -> DriveState {
        use glam::Vec3;
        let mut st = st;
        // Kept so the odometer can be the motion that was actually RESOLVED rather than the
        // velocity that was asked for: every branch below writes `st.pos` after clamping it to
        // the ground, and a vehicle held against a slope must not clock up distance its wheels
        // never travelled.
        let prev_pos = st.pos;
        let fwd_flat = Vec3::new(st.yaw.cos(), st.yaw.sin(), 0.0);

        // --- TURNING ------------------------------------------------------------------------
        let planar = Vec3::new(st.vel.x, st.vel.y, 0.0);
        let speed = planar.length();
        let signed_speed = planar.dot(fwd_flat);
        let rate = self.turn.yaw_rate(speed);
        // #vehicle-law  THE COMMAND AND THE LAW TRAVEL TOGETHER. See [`TurnLaw`].
        //
        // **The lateral stick steers; the camera does not.** `DriveInput::slide` carries the
        // left/right movement axis, and it means nothing to a wheeled or tracked vehicle —
        // `maximum left/right slide` is authored only by the hover families, and the lateral
        // basis vector a few lines below is zeroed for everything that does not author it. So
        // `slide` IS the steer command for the families that steer, and `want_yaw` is used only
        // by the family that really does turn its hull toward the look.
        //
        // This is the fix for *"I turn and the whole warthog pivots around on a dime"*: the steer
        // angle used to slew toward `want_yaw - yaw`, the camera's own heading, so looking left
        // pinned the wheels at full lock until the hull had come round to face the camera. The
        // kinematics were never the problem. The command was.
        let steer_cmd = input.slide.clamp(-1.0, 1.0);
        match (self.law, self.turning) {
            // A STEERED-WHEEL vehicle. The stick positions the road wheels between the authored
            // maxima, the angle slews there at the authored `turn rate`, and the heading follows
            // only from rolling: at zero speed the yaw rate is exactly zero.
            (TurnLaw::Bicycle { wheelbase }, Some(t)) => {
                let target = if steer_cmd >= 0.0 { steer_cmd * t.max_left } else { -steer_cmd * t.max_right };
                let slew = t.turn_rate.abs().max(1e-3) * dt;
                st.steer_angle += (target - st.steer_angle).clamp(-slew, slew);
                let yaw_rate = signed_speed * st.steer_angle.tan() / wheelbase.max(1e-3);
                st.yaw = wrap_angle(st.yaw + yaw_rate * dt);
            }
            // A TRACKED vehicle. The stick sets a track-speed DIFFERENCE, so it turns about its
            // own axis — and it may do so standing still, which is the whole point of tracks and
            // the one place a near-dime pivot is correct. Derived from the field names and the
            // measured track, not disassembled.
            (TurnLaw::Differential { track }, Some(t)) => {
                let target = if steer_cmd >= 0.0 { steer_cmd * t.max_left } else { -steer_cmd * t.max_right };
                let slew = t.turn_rate.abs().max(1e-3) * dt;
                st.steer_angle += (target - st.steer_angle).clamp(-slew, slew);
                let yaw_rate = st.steer_angle / track.max(1e-3);
                st.yaw = wrap_angle(st.yaw + yaw_rate * dt);
            }
            // A HOVER vehicle turns toward where the driver is looking, at the seat's authored
            // rate — and so does anything whose law fell back, which `law_note` says out loud.
            _ => {
                st.steer_angle = 0.0;
                let want = wrap_angle(input.want_yaw - st.yaw);
                st.yaw = wrap_angle(st.yaw + want.clamp(-rate * dt, rate * dt));
            }
        }

        // --- THE GROUND, PER CONTACT POINT ---------------------------------------------------
        //
        // #vehicle-suspension  Solved BEFORE the speed, because the hull's attitude decides which
        // way "forward" points and the slope decides how much of gravity the drivetrain is
        // fighting.
        // #gravity  ONE writer. [`crate::reach_physics::vehicle_fall_acceleration`] times the
        // authored havok `gravity scale`, computed here and handed to everything that needs it.
        let g = crate::reach_physics::vehicle_fall_acceleration() * self.gravity_scale;
        let solve = match &self.contacts {
            Some(c) => c.solve(&self.spring, g, st.basis(), st.pos, st.vel, (st.pitch_rate, st.roll_rate), ground),
            None => drive::ContactSolve::default(),
        };
        st.contact_count = solve.count.min(drive::MAX_CONTACT_POINTS) as u8;
        st.suspension = solve.compression;
        st.touching = solve.touching;
        st.ground_normal = solve.normal;
        st.grounded = solve.touching_count > 0;

        // The hull's basis NOW includes pitch and roll, so forward really does follow the slope.
        let fwd = st.forward();
        let left = if self.slide.max_left_slide > 0.0 { st.left() } else { Vec3::ZERO };

        // --- SPEED --------------------------------------------------------------------------
        let throttle = input.throttle.clamp(-1.0, 1.0);
        let cur_fwd = st.vel.dot(fwd);
        // How much the world is resisting, along the hull's own forward: the component of gravity
        // down the slope, plus the family's authored drag terms. All authored; the slope is
        // measured from the contact plane.
        let slope_resist = if st.grounded { g * fwd.z } else { 0.0 };
        let drag_resist = self.slide.drag_coefficient * cur_fwd * cur_fwd * cur_fwd.signum() + self.slide.constant_deceleration * cur_fwd.signum();
        let (new_fwd, wheel_rate) = match &self.drivetrain {
            // #vehicle-drivetrain  A GEARED VEHICLE: the engine makes the speed.
            Some(dt_train) => {
                let has_grip = st.grounded;
                let ebrake = input.brake;
                // The ground torque is the load, from the no-slip identity — see
                // [`drive::Drivetrain::ground_torque_for_resistance`].
                st.engine.ground_torque =
                    dt_train.ground_torque_for_resistance(st.engine.gear, slope_resist + drag_resist);
                let requested = dt_train.requested_gear(&st.engine, throttle, ebrake, true);
                dt_train.update(&mut st.engine, throttle, requested, ebrake, dt);
                let surface = dt_train.wheel_surface_speed(&st.engine);
                // The hull follows the wheels through TRACTION, which is what makes a Warthog
                // build speed instead of snapping to one. The traction limit is authored:
                // `ground friction` times gravity. With no grip the wheels spin and the hull
                // coasts.
                let limit = if has_grip { (g * self.ground_friction).max(0.1) } else { 0.0 };
                let braked = if ebrake { 0.0 } else { surface };
                let v = approach(cur_fwd, braked, limit * dt);
                (v, surface)
            }
            // #vehicle-law  A HOVER VEHICLE: the family authors its speeds outright and there is
            // no drivetrain to derive them from, so the authored first-order model stands.
            None => {
                let target_fwd =
                    if throttle >= 0.0 { throttle * self.max_forward_speed } else { throttle * self.max_reverse_speed };
                let a = if target_fwd.abs() >= cur_fwd.abs() && !input.brake {
                    self.speed_acceleration
                } else {
                    self.speed_deceleration
                };
                let v = approach(cur_fwd, if input.brake { 0.0 } else { target_fwd }, a * dt);
                // Drag, as the two field NAMES read: `drag coeficient` quadratic and `constant
                // deceleration` constant. **The FORM, not the engine's function.**
                let v = v - (drag_resist + slope_resist) * dt;
                (v, 0.0)
            }
        };

        let (sl, sr) = self.slide.slide_ceiling(cur_fwd);
        let s = input.slide.clamp(-1.0, 1.0);
        let target_slide = if s >= 0.0 { s * sl } else { s * sr };
        let cur_slide = st.vel.dot(left);
        // Both `cur_*` are the OLD velocity's components in the hull's own axes, and both are
        // needed below to take the old contribution back out.
        let sa = if target_slide.abs() >= cur_slide.abs() { self.slide.slide_acceleration } else { self.slide.slide_deceleration };
        let new_slide = approach(cur_slide, target_slide, sa.max(0.0) * dt);

        // #vehicle-suspension  **REPLACE the forward and lateral components; do not ADD them to
        // the whole old vector.** This line used to read
        //
        //     let mut vel = fwd * new_fwd + left * new_slide + Vec3::Z * st.vel.z;
        //
        // which is only correct while `fwd` is FLAT. It was, until terrain conformance gave the
        // hull a pitch — and then `fwd` has a Z component, `cur_fwd = vel.dot(fwd)` picks up
        // `vel.z * fwd.z`, and putting `fwd * new_fwd` back on top of a vertical lane that still
        // holds all of `vel.z` counts that part twice:
        //
        //     vel.z' = fwd.z * (fwd.z * vel.z) + vel.z = vel.z * (1 + fwd.z^2)
        //
        // **A geometric gain of `1 + fwd.z^2` per sub-step.** At a 4.8 degree pitch that is
        // 1.007 per sub-step, 2.33x per second, and it compounds as the pitch grows: measured, a
        // hull reached 1.9e10 wu in six seconds and 8e26 wu/s in ten, with its wheels off the
        // ground and its springs reading zero the whole time. It was not the suspension pushing
        // it; it was this algebra. **That is the *"warthog just flew into the air"* report**, and
        // the latent error predates the suspension — conformance only made `fwd.z` non-zero and
        // turned it on.
        //
        // The fix is an exact decomposition: take out whatever the old velocity had along `fwd`
        // and `left`, keep the residual, and put the new components in. With no change asked for
        // it is the identity, which is the property the gate asserts.
        // The decomposition is in the HULL's own two axes, and `side` is taken whether or not the
        // family authors a slide: for a wheeled vehicle `left` is zero, so its lateral component is
        // DROPPED rather than carried, which is the tyre's grip and is what the old flat algebra
        // did by construction. Carrying it instead let a Warthog keep sideways momentum through a
        // corner, which grew its speed through a turn and is a change nobody asked for.
        let side = st.left();
        let vertical = st.vel - fwd * cur_fwd - side * st.vel.dot(side);
        let mut vel = vertical + fwd * new_fwd + left * new_slide;
        let _ = cur_slide;

        // --- GRAVITY, THE SPRINGS AND THE ATTITUDE -------------------------------------------
        //
        // #gravity  [`crate::reach_physics::vehicle_fall_acceleration`] — the engine's authored
        // world constant, measured in the vehicle's OWN physics path (the wheel normal force
        // reads the `"physics constants"` block's first float), scaled by the authored havok
        // `gravity scale`.
        if self.contacts.is_some() && self.spring.authored() {
            // The springs hold the hull up; gravity pulls it down. At rest the two cancel at a
            // compression of `load share * ground depth` per point, which is a number a gate
            // computes from the tag without this code.
            vel.z += (solve.spring_accel_z - g) * dt;
            // The hull's attitude follows the same per-point forces, taken about the WEIGHTED
            // centroid of the contact points so that a parked vehicle is exactly level.
            st.pitch_rate += solve.alpha_pitch * dt;
            st.roll_rate += solve.alpha_roll * dt;
            // Airborne, the hull keeps its attitude and does not spin: it is not being modelled
            // as a free rigid body, so the rates are bled off rather than integrated forever.
            if solve.touching_count == 0 {
                let bleed = (1.0 - 2.0 * dt).clamp(0.0, 1.0);
                st.pitch_rate *= bleed;
                st.roll_rate *= bleed;
            }
            // The RATES are bounded as well as the angles. Clamping only the angle left the rate
            // free to grow, and the rate is an input to the damper through `rate * d` -- which is
            // the loop that diverged. See [`drive::MAX_ATTITUDE_RATE`].
            let r = drive::MAX_ATTITUDE_RATE;
            st.pitch_rate = st.pitch_rate.clamp(-r, r);
            st.roll_rate = st.roll_rate.clamp(-r, r);
            st.pitch = (st.pitch + st.pitch_rate * dt).clamp(-1.4, 1.4);
            st.roll = (st.roll + st.roll_rate * dt).clamp(-1.4, 1.4);
            let mut pos = st.pos + vel * dt;
            // The bump stop: no point may be compressed past the authored travel. Lifting the
            // hull here is a geometric constraint, not a force, and it is what keeps a vehicle
            // out of the terrain when the spring is saturated.
            // THE BUMP STOP, as a RATE-LIMITED correction and not a teleport. See
            // [`drive::MAX_PENETRATION_PUSH_OUT`]: adding the whole penetration every sub-step is
            // what launched a Warthog, and resolving an overlap must not create velocity either.
            if solve.penetration > 0.0 {
                pos.z += solve.penetration.min(drive::MAX_PENETRATION_PUSH_OUT * dt);
                if vel.z < 0.0 {
                    vel.z = 0.0;
                }
            }
            st.pos = pos;
        } else if self.contacts.is_some() {
            // A HOVER vehicle: it authors no wheel spring, so it is held at the plane through its
            // own contact points rather than sprung. It still CONFORMS: pitch and roll come from
            // the measured ground normal.
            let (p, r) = drive::attitude_from_normal(solve.normal, st.yaw);
            let k = (6.0 * dt).clamp(0.0, 1.0);
            st.pitch += (p - st.pitch) * k;
            st.roll += (r - st.roll) * k;
            st.pitch_rate = 0.0;
            st.roll_rate = 0.0;
            vel.z -= g * dt;
            let mut pos = st.pos + vel * dt;
            if solve.penetration > 0.0 {
                pos.z += solve.penetration.min(drive::MAX_PENETRATION_PUSH_OUT * dt);
                if vel.z < 0.0 {
                    vel.z = 0.0;
                }
            }
            st.pos = pos;
        } else {
            // NO CONTACT GEOMETRY AT ALL: the vehicle resolved fewer than three markers, which
            // `law_note` says out loud. It is held at a flat `ride_height`, exactly as before —
            // and it stays level, which is the behaviour being reported as wrong and the reason
            // the note has to be printed.
            vel.z -= g * dt;
            let mut pos = st.pos + vel * dt;
            st.grounded = false;
            if let Some((h, _n)) = ground(pos) {
                let rest = h + self.ride_height;
                if pos.z <= rest {
                    pos.z = rest;
                    if vel.z < 0.0 {
                        vel.z = 0.0;
                    }
                    st.grounded = true;
                }
            }
            st.pos = pos;
        }
        // #vehicle-anim  THE ODOMETER, from the motion that was actually resolved rather than
        // from the velocity that was asked for: `pos` is already clamped to the ground, so a
        // vehicle held still against a slope does not clock up distance its wheels never
        // travelled. The FORWARD component only -- a hover vehicle sliding sideways does not
        // roll, and neither does one falling.
        let moved = st.pos - prev_pos;
        st.odometer += Vec3::new(moved.x, moved.y, 0.0).dot(fwd_flat);
        // #vehicle-drivetrain  And the WHEELS' own travel, which is the engine's rolling phase:
        // `gear ratio * engine angular velocity` integrated. It differs from the odometer
        // exactly when the wheels slip, which is what makes wheelspin visible.
        st.wheel_travel += wheel_rate * dt;
        st.time += dt;
        st.vel = vel;
        st
    }

    /// Lines for the caption: every number and where it came from.
    pub fn describe(&self) -> Vec<String> {
        let mut out = vec![
            format!(
                "  {} drive: fwd {:.2} rev {:.2} wu/s from {}",
                self.kind, self.max_forward_speed, self.max_reverse_speed, self.speed_source.label()
            ),
            format!(
                "  accel {:.2} decel {:.2} wu/s/s, turn {:.1}..{:.1} deg/s over 0..{:.2} wu/s ^{:.2}, \
                 gravity scale {:.2}, ride height {:.3}{}",
                self.speed_acceleration,
                self.speed_deceleration,
                self.turn.yaw_deg[0],
                self.turn.yaw_deg[1],
                self.turn.max_speed_reference,
                self.turn.exponent,
                self.gravity_scale,
                self.ride_height,
                match (self.turning, self.wheelbase) {
                    (Some(t), Some(wb)) => format!(
                        ", steering +-{:.3}/{:.3} rad at {:.2} rad/s over a {:.3} wu wheelbase",
                        t.max_left, t.max_right, t.turn_rate, wb
                    ),
                    (Some(t), None) => format!(
                        ", steering +-{:.3}/{:.3} rad at {:.2} rad/s but NO measurable wheelbase",
                        t.max_left, t.max_right, t.turn_rate
                    ),
                    (None, _) => ", hover: turns toward the driver's look".to_string(),
                }
            ),
            // #vehicle-law  WHICH LAW, said out loud. A vehicle turned by the wrong law is what
            // the dime-pivot report was, and a caption that names the law is how the next one
            // gets diagnosed from the screenshot.
            format!("  turn law: {}", self.law.label()),
        ];
        // #vehicle-drivetrain  Which of the two speed laws is running, and its numbers.
        match &self.drivetrain {
            Some(d) => out.push(format!(
                "  drivetrain: moment {:.0}, max engine {:.2}, {} gear(s), top ratio {:.3} -> ceiling {:.2} wu/s;                  traction limit {:.2} wu/s/s (ground friction {:.3} x g)",
                d.moment,
                d.max_angular_velocity,
                d.gears.len(),
                d.gears.iter().map(|g| g.ratio).fold(0.0f32, f32::max),
                d.speed_ceiling(),
                crate::reach_physics::vehicle_fall_acceleration() * self.gravity_scale * self.ground_friction,
                self.ground_friction
            )),
            None => out.push(format!(
                "  no drivetrain: this family authors no engine, so the speed is its own authored                  {:.2}/{:.2} wu/s first-order model",
                self.max_forward_speed, self.max_reverse_speed
            )),
        }
        // #vehicle-suspension  The contact solve, with the resting compression the tag implies --
        // the number a reader can check against the vehicle by eye.
        match &self.contacts {
            Some(c) => {
                out.push(format!(
                    "  suspension: {} contact point(s), travel {:.4} wu (ground depth), damp {:.3},                      force cap {:.2} wu/s/s, traction above cos {:.3} ({:.1} deg)",
                    c.points.len(),
                    self.spring.depth,
                    self.spring.damp,
                    self.spring.max_normal_force,
                    self.spring.slope_cos[1],
                    self.spring.slope_cos[1].clamp(-1.0, 1.0).acos().to_degrees()
                ));
                out.push(format!(
                    "  load centroid ({:+.3},{:+.3},{:+.3}) wu, pitch inertia {:.4} roll {:.4} wu^2;                      resting compression {}",
                    c.centroid.x,
                    c.centroid.y,
                    c.centroid.z,
                    c.pitch_inertia,
                    c.roll_inertia,
                    (0..c.points.len())
                        .map(|i| format!("{:.4}", c.static_compression(i, &self.spring)))
                        .collect::<Vec<_>>()
                        .join("/")
                ));
            }
            None => out.push(
                "  NO suspension and NO terrain conformance: fewer than three contact markers                  resolved, so this hull stays level"
                    .to_string(),
            ),
        }
        if let Some(n) = &self.law_note {
            out.push(format!("  !! {n}"));
        }
        out
    }

    /// #vehicle-law  The turning circle at full lock, wu — `L / tan(delta_max)`.
    ///
    /// `None` for anything not turned by the bicycle model. Exposed because it is the one number
    /// a reader can check against the game by eye, and because a gate can compute it from the
    /// friction points independently of the integrator.
    pub fn turning_radius(&self) -> Option<f32> {
        let TurnLaw::Bicycle { wheelbase } = self.law else { return None };
        let t = self.turning?;
        let d = t.max_left.abs().max(t.max_right.abs());
        (d > 1e-3).then(|| wheelbase / d.tan())
    }
}

fn approach(cur: f32, target: f32, step: f32) -> f32 {
    if (target - cur).abs() <= step { target } else { cur + (target - cur).signum() * step }
}

pub(crate) fn wrap_angle(a: f32) -> f32 {
    let t = std::f32::consts::TAU;
    (a + std::f32::consts::PI).rem_euclid(t) - std::f32::consts::PI
}

/// The bottom gear's `min time to upshift`, which is how long the engine is allowed to hold the
/// lowest gear. Used as the time constant for a wheeled vehicle's acceleration, because the
/// family authors no acceleration of its own and this is an authored time rather than a typed one.
fn first_upshift_time(f: &Family) -> Option<f32> {
    let e = match f {
        Family::Jeep { engine, .. } | Family::Tank { engine, .. } | Family::Revenant { engine, .. } => engine,
        _ => return None,
    };
    e.gears
        .iter()
        .filter(|g| g.ratio > 0.0 && g.min_time_to_upshift > 0.0)
        .map(|g| g.min_time_to_upshift)
        .fold(None, |acc: Option<f32>, t| Some(acc.map_or(t, |a| a.min(t))))
}

/// The wheelbase, MEASURED from the vehicle's own friction-point markers: the largest
/// front-to-back separation among them. `None` with fewer than two, in which case the caller must
/// not invent one.
///
/// The marker POSITIONS are not in the `vehi` tag — they are in the render model — so this takes
/// them from the caller through [`VehiclePhysics::set_wheel_positions`], and returns `None` until
/// it has been told.
/// #vehicle-law  The TRACK: the lateral spread of the friction points, wu. The quantity a
/// tracked vehicle's differential steering divides by.
///
/// Measured the same way as the wheelbase and from the same markers (the Scorpion's twelve give
/// 1.7400 wu), so neither number is typed in.
fn track_from(p: &VehiclePhysics) -> Option<f32> {
    let ys: Vec<f32> = p.wheel_positions.iter().map(|v| v.y).collect();
    if ys.len() < 2 {
        return None;
    }
    let lo = ys.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let tr = hi - lo;
    (tr > 1e-3).then_some(tr)
}

fn wheelbase_from(p: &VehiclePhysics) -> Option<f32> {
    let xs: Vec<f32> = p.wheel_positions.iter().map(|v| v.x).collect();
    if xs.len() < 2 {
        return None;
    }
    let lo = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let wb = hi - lo;
    (wb > 1e-3).then_some(wb)
}

#[cfg(test)]
mod gates {
    //! #vehicle-phys  `cargo test -p hms-app --profile quick the_vehicle_physics -- --ignored --nocapture`

    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// A flat, unauthored ground spring: what a vehicle with no `ground depth` has, and what a
    /// fixture that is testing something other than the suspension must be given so the
    /// suspension cannot contribute to the number under test.
    fn no_spring() -> drive::GroundSpring {
        drive::GroundSpring {
            depth: 0.0,
            damp: 0.0,
            max_normal_force: 3.0,
            slope_cos: [0.0, 0.0],
            iterations: 1,
        }
    }

    fn gear(index: usize, ratio: f32, up: f32, down: f32) -> Gear {
        Gear {
            index,
            ratio,
            min_time_to_upshift: up,
            upshift_scale: 0.0,
            min_time_to_downshift: down,
            downshift_scale: 0.0,
            loaded_torque: None,
            cruising_torque: None,
        }
    }

    /// **THE FAMILY GATE.** Every `vehi` populates **exactly one** of the fifteen type blocks,
    /// and the one it populates is the family its name says.
    ///
    /// This is the assertion that cannot be satisfied by zeros: zeros populate NO block, and this
    /// demands exactly one on every vehicle in the cache. It is also the one that catches a wrong
    /// `VEHI_OWN_FIELDS` — a shifted window reads block counts out of the unit's tail and either
    /// finds none or finds several.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_vehicle_physics_blocks_are_one_per_vehicle() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let vehis = native.tags_by_group(cache, "vehi");
        assert!(vehis.len() > 10, "forge_halo should carry many vehi tags, found {}", vehis.len());
        let mut per_family: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut wrong: Vec<String> = Vec::new();
        for &v in &vehis {
            let blocks = VehiclePhysics::populated_blocks(native, cache, v);
            let name = native.tag_name(cache, v).unwrap_or_default();
            match blocks.len() {
                1 => *per_family.entry(blocks[0]).or_default() += 1,
                n => wrong.push(format!("{name}: {n} populated type blocks {blocks:?}")),
            }
        }
        println!("{} vehi tags by physics family:", vehis.len());
        for (f, n) in &per_family {
            println!("   {f:<16} {n}");
        }
        for w in &wrong {
            println!("   WRONG: {w}");
        }
        assert!(wrong.is_empty(), "{} vehi tags do not populate exactly one type block", wrong.len());
        assert!(per_family.len() >= 4, "only {} distinct families found -- expected the Forge spread", per_family.len());
    }

    /// **THE AUTHORED-VALUE GATE.** The numbers must read as numbers somebody tuned, per family,
    /// and the four well-known vehicles must come out where a player would expect.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_vehicle_physics_reads_as_physics() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let vehis = native.tags_by_group(cache, "vehi");
        let mut read = 0usize;
        let mut drivable = 0usize;
        let mut with_gears = 0usize;
        let mut with_curves = 0usize;
        for &v in &vehis {
            let Some(p) = VehiclePhysics::read(native, cache, v) else { continue };
            read += 1;
            if p.drivable() {
                drivable += 1;
            }
            let name = native.tag_name(cache, v).unwrap_or_default();
            println!("\n{name}");
            for l in p.describe() {
                println!("{l}");
            }
            match &p.family {
                Family::Jeep { engine, .. } | Family::Tank { engine, .. } | Family::Revenant { engine, .. } => {
                    if !engine.gears.is_empty() {
                        with_gears += 1;
                    }
                    if engine.gears.iter().any(|g| g.loaded_torque.is_some()) {
                        with_curves += 1;
                    }
                }
                Family::Scout { model, .. } => {
                    assert!(
                        model.max_forward_speed > 0.0 && model.max_forward_speed < 100.0,
                        "{name}: alien_scout max forward speed {} is not a speed",
                        model.max_forward_speed
                    );
                    assert!(
                        model.speed_acceleration > 0.0,
                        "{name}: alien_scout authors no speed acceleration"
                    );
                }
                Family::Unsupported { .. } => {}
            }
        }
        println!("\n{read} vehi tags read, {drivable} in a family this module drives, {with_gears} with gears, {with_curves} with decoded torque curves");
        assert!(read > 10, "only {read} vehi tags read their physics block");
        assert!(with_gears > 0, "no vehicle authored a gear -- the engine struct offset is wrong");
        assert!(
            with_curves > 0,
            "no gear's torque curve decoded -- the c_function_definition data offsets are wrong"
        );
    }

    /// **THE GEAR-ORDER GATE.** The thing a gearbox reading gets wrong is which end is top, and
    /// the shift times decide it: a gear with **no downshift time** is the bottom of the box and a
    /// gear with **no upshift time** is the top. Over every geared vehicle in the cache, that
    /// must agree with the ratio ordering — bottom gear has the smallest positive ratio, top gear
    /// the largest.
    ///
    /// This is the assertion that would have caught the first version of
    /// [`DriveLimits::from_engine`], which took the smallest ratio as top gear. **An ordering
    /// assertion alone would not have**: it is only discriminating because it cross-checks two
    /// independent authored quantities against each other.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_gear_order_is_proved_by_the_shift_times() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut tested = 0usize;
        let mut agreed = 0usize;
        let mut witnesses = 0usize;
        for v in native.tags_by_group(cache, "vehi") {
            let Some(p) = VehiclePhysics::read(native, cache, v) else { continue };
            let engine = match &p.family {
                Family::Jeep { engine, .. } | Family::Tank { engine, .. } | Family::Revenant { engine, .. } => engine,
                _ => continue,
            };
            let fwd: Vec<&Gear> = engine.gears.iter().filter(|g| g.ratio > 0.0).collect();
            if fwd.len() < 2 {
                continue;
            }
            let name = native.tag_name(cache, v).unwrap_or_default();
            // the gear that cannot downshift, and the gear that cannot upshift
            let bottom: Vec<usize> = fwd.iter().filter(|g| g.min_time_to_downshift == 0.0).map(|g| g.index).collect();
            let top: Vec<usize> = fwd.iter().filter(|g| g.min_time_to_upshift == 0.0).map(|g| g.index).collect();
            let smallest = fwd.iter().min_by(|a, b| a.ratio.total_cmp(&b.ratio)).unwrap().index;
            let largest = fwd.iter().max_by(|a, b| a.ratio.total_cmp(&b.ratio)).unwrap().index;
            // The claim is an IMPLICATION: *where* a gear with no upshift time exists it is the
            // largest ratio, and *where* a gear with no downshift time exists it is the smallest.
            // Two vehicles satisfy it vacuously -- `bed_long` and `bed_small`, the truck's TRAILER
            // attachments, whose two forward gears both carry an upshift time, so no gear is top.
            // They are never driven (they are attached children of `truck_cab_large`). A vacuous
            // pass proves nothing, so the vehicles that WITNESS the rule are counted separately
            // and the gate requires enough of them.
            let implication = bottom.iter().all(|&i| i == smallest) && top.iter().all(|&i| i == largest);
            let witnessed = !bottom.is_empty() && !top.is_empty();
            tested += 1;
            if witnessed {
                witnesses += 1;
            }
            if implication {
                agreed += 1;
            }
            println!(
                "  {name}: {} forward gear(s); cannot downshift {bottom:?} (smallest ratio {smallest}), \
                 cannot upshift {top:?} (largest ratio {largest}){}",
                fwd.len(),
                if !implication {
                    "   <-- CONTRADICTS"
                } else if !witnessed {
                    "   (vacuous: no gear is top -- a trailer, never driven)"
                } else {
                    ""
                }
            );
        }
        println!(
            "{agreed} of {tested} geared vehicles are consistent with 'largest ratio = top gear', \
             {witnesses} of them WITNESS it (a gear with no upshift time and a gear with no downshift time)"
        );
        assert!(tested >= 2, "only {tested} vehicles have two or more forward gears -- the order was never tested");
        assert_eq!(agreed, tested, "{} geared vehicle(s) CONTRADICT the gear order", tested - agreed);
        assert!(
            witnesses >= 4,
            "only {witnesses} vehicles witness the rule non-vacuously -- a gate satisfied by \
             vehicles with no top gear has not tested anything"
        );
    }

    /// **THE SLOPE-ENCODING GATE.** `ground maximum slope 0/1` are labelled "degrees 0-90" and
    /// stored as COSINES. Both readings cannot be right, and the field's own documentation that
    /// slope 1 must be "greater than slope 0" is what separates them: as cosines the pair is
    /// monotonic the right way on every wheeled vehicle in the cache, and as degrees it is
    /// monotonic the WRONG way on all of them.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_ground_slope_limits_are_cosines_not_degrees() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut wheeled = 0usize;
        let mut cos_ok = 0usize;
        let mut deg_ok = 0usize;
        let mut in_unit_range = 0usize;
        for v in native.tags_by_group(cache, "vehi") {
            let Some(p) = VehiclePhysics::read(native, cache, v) else { continue };
            // only a vehicle with wheels uses a ground slope at all
            if p.havok.friction_points.is_empty() {
                continue;
            }
            let [a, b] = p.havok.ground_max_slope_cos;
            if !(a > 0.0 && b > 0.0) {
                continue;
            }
            wheeled += 1;
            if a <= 1.0 && b <= 1.0 {
                in_unit_range += 1;
            }
            // as COSINES, slope 1 greater in ANGLE means a smaller cosine
            if b < a {
                cos_ok += 1;
            }
            // as DEGREES, slope 1 greater means a bigger number
            if b > a {
                deg_ok += 1;
            }
            let d = p.havok.ground_max_slope_degrees();
            println!(
                "  {}: slope cos {a:.3}/{b:.3} = {:.1}/{:.1} deg",
                native.tag_name(cache, v).unwrap_or_default(),
                d[0],
                d[1]
            );
        }
        println!("{wheeled} wheeled vehicles: {cos_ok} consistent as cosines, {deg_ok} as degrees, {in_unit_range} inside [0,1]");
        assert!(wheeled >= 3, "only {wheeled} wheeled vehicles authored a slope pair -- not enough to decide");
        assert_eq!(in_unit_range, wheeled, "a value above 1.0 cannot be a cosine");
        assert_eq!(cos_ok, wheeled, "{} vehicles are NOT monotonic as cosines", wheeled - cos_ok);
        assert_eq!(deg_ok, 0, "{deg_ok} vehicles are monotonic as degrees too -- this gate does not discriminate");
    }

    /// **THE DRIVE GATE.** Every ground vehicle in the cache must build a drive model, every
    /// flyer must REFUSE with its family named, and a Ghost driven from rest must reach its own
    /// authored top speed in about the time its own authored acceleration says.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_drive_model_builds_for_the_ground_families_and_refuses_for_the_flyers() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_vehicle gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut built = 0usize;
        let mut refused = 0usize;
        let mut authored = 0usize;
        let mut reference = 0usize;
        let mut ghost: Option<DriveModel> = None;
        let mut scorpion_refusal: Option<String> = None;
        for v in native.tags_by_group(cache, "vehi") {
            let Some(mut p) = VehiclePhysics::read(native, cache, v) else { continue };
            let name = native.tag_name(cache, v).unwrap_or_default();
            // Resolve the wheel marker positions the model cannot know on its own.
            let mode = native.resolve_mode(cache, v);
            if let Some(m) = crate::reach_actor::ReachModel::read(native, cache, mode) {
                let pose = crate::reach_actor::Pose::bind(&m.skel);
                let pos: Vec<glam::Vec3> = p
                    .havok
                    .friction_points
                    .iter()
                    .filter_map(|f| m.marker_group(native, cache, &f.marker_name))
                    .filter_map(|g| g.markers.first().copied())
                    .map(|mk| pose.marker_model(&mk).w_axis.truncate())
                    .collect();
                p.set_wheel_positions(pos);
            }
            let seats = crate::reach_seats::UnitSeats::read(native, cache, v);
            let driver = seats.as_ref().and_then(|u| u.seats.iter().find(|s| s.is_driver()).or_else(|| u.seats.first()));
            match DriveModel::for_vehicle(&p, driver) {
                Ok(d) => {
                    built += 1;
                    match d.speed_source {
                        SpeedSource::Authored => authored += 1,
                        SpeedSource::SeatMaxSpeedReference => reference += 1,
                    }
                    println!("\n{name}");
                    for l in d.describe() {
                        println!("{l}");
                    }
                    assert!(
                        d.max_forward_speed > 0.0 && d.max_forward_speed < 40.0,
                        "{name}: forward speed {} is not a speed",
                        d.max_forward_speed
                    );
                    assert!(d.speed_acceleration > 0.0, "{name}: no acceleration");
                    assert!(d.gravity_scale > 0.0, "{name}: gravity scale {} -- the 0-means-1 default did not apply", d.gravity_scale);
                    if name.ends_with("ghost") {
                        ghost = Some(d);
                    }
                }
                Err(why) => {
                    refused += 1;
                    if name.ends_with("scorpion") {
                        scorpion_refusal = Some(why.clone());
                    }
                    println!("\n{name}\n  REFUSED: {why}");
                }
            }
        }
        println!("\n{built} vehicles drive ({authored} from an authored speed, {reference} from the \
                  seat's `max speed reference`), {refused} refused");
        // Eight of 43 in `forge_halo`: the three hover vehicles from their own authored speeds,
        // five of the wheeled ones from the reference. The Scorpion, the big truck and the van
        // refuse because their driver seats author `max speed reference` 0.00 and nothing else
        // in the tag gives a speed -- which is the correct outcome, not a gap to fill.
        assert_eq!(authored, 3, "the Ghost, the Wraith and the Revenant author their own speeds");
        assert!(reference >= 4, "only {reference} wheeled vehicles had a `max speed reference` to drive at");
        assert!(built >= 7, "only {built} vehicles built a drive model");
        assert!(refused >= 20, "only {refused} refused -- the flyers and turrets must refuse, not get invented dynamics");
        assert!(
            scorpion_refusal.as_deref().is_some_and(|r| r.contains("max speed reference")),
            "the Scorpion must refuse naming what it lacks, got {scorpion_refusal:?}"
        );

        // The Ghost, integrated. Its own tag says 5.00 wu/s at 6.00 wu/s/s, so from rest on flat
        // ground it must be at its top speed in a little over a second, and must NOT exceed it.
        let d = ghost.expect("forge_halo carries the Ghost");
        assert_eq!(d.speed_source, SpeedSource::Authored, "the Ghost authors its own speed");
        let flat = |_p: glam::Vec3| Some((0.0f32, glam::Vec3::Z));
        let mut st = DriveState { pos: glam::Vec3::ZERO, ..Default::default() };
        let input = DriveInput { throttle: 1.0, want_yaw: 0.0, ..Default::default() };
        let mut t = 0.0f32;
        let mut top = 0.0f32;
        while t < 6.0 {
            st = d.step(st, input, 1.0 / 60.0, &flat);
            t += 1.0 / 60.0;
            top = top.max(glam::Vec3::new(st.vel.x, st.vel.y, 0.0).length());
        }
        println!("the Ghost after {t:.1}s of full throttle on flat ground: {top:.3} wu/s (authored {:.2})", d.max_forward_speed);
        assert!(
            top <= d.max_forward_speed + 1e-3,
            "the Ghost exceeded its own authored top speed: {top} > {}",
            d.max_forward_speed
        );
        // Drag takes a little off the authored ceiling: at 5 wu/s the authored 0.008 coefficient
        // costs 0.2 wu/s/s, so the steady state sits just under. Within 5% is the claim.
        assert!(
            top >= d.max_forward_speed * 0.95,
            "the Ghost only reached {top} of its authored {}",
            d.max_forward_speed
        );
        // And it must get there in about v/a seconds, not in ten.
        let mut st = DriveState::default();
        let mut t = 0.0f32;
        while glam::Vec3::new(st.vel.x, st.vel.y, 0.0).length() < d.max_forward_speed * 0.9 && t < 10.0 {
            st = d.step(st, input, 1.0 / 60.0, &flat);
            t += 1.0 / 60.0;
        }
        let expect = d.max_forward_speed * 0.9 / d.speed_acceleration;
        println!("   reached 90% in {t:.2}s; v*0.9/a = {expect:.2}s");
        assert!(
            t < expect * 2.0 && t > expect * 0.5,
            "the Ghost reached 90% of its speed in {t:.2}s but its authored acceleration says about {expect:.2}s"
        );
    }

    /// **THE TURN-LAW GATE, on RELATIONSHIPS and not on fitted numbers.** No cache: every number
    /// here is either authored into the fixture or derived from it by the law itself.
    ///
    /// Reported: *"I turn and the whole warthog pivots around on a dime, the game does NOT do
    /// that."* The three properties below are what "the game does not do that" means, stated so
    /// they can fail:
    ///
    /// 1. **At zero speed, full stick, the heading must not change AT ALL.** This is the one the
    ///    user actually saw. It is a property of `omega = v*tan(d)/L`, and the old code broke it
    ///    not through the kinematics but through the COMMAND — so the gate drives the production
    ///    `step` with a full lateral stick and asserts the yaw is bit-for-bit unchanged.
    /// 2. **Doubling the speed at a fixed steer angle doubles the yaw rate.** Linearity in `v`,
    ///    which no heading servo has.
    /// 3. **The turning circle at full lock is `L / tan(d_max)`**, computed here from the
    ///    fixture's own `L` and `d_max` and compared against the radius the integrator actually
    ///    traces — two independent routes to one number.
    ///
    /// And the camera must not steer: the same inputs with `want_yaw` swung right round must give
    /// the identical heading, which is the regression that bug deserves.
    #[test]
    fn a_wheeled_vehicle_turns_only_because_it_is_rolling() {
        use glam::Vec3;
        // The Warthog's own authored numbers: +-60 deg of lock, a 1.3000 wu wheelbase measured
        // from its four friction points, and a turn rate fast enough to reach lock quickly.
        let lock = 60.0f32.to_radians();
        let d = DriveModel {
            kind: "human_jeep",
            max_forward_speed: 8.0,
            max_reverse_speed: 4.0,
            speed_source: SpeedSource::SeatMaxSpeedReference,
            speed_acceleration: 100.0,
            speed_deceleration: 100.0,
            slide: VelocityModel::default(),
            turning: Some(Turning { max_left: lock, max_right: -lock, turn_rate: 9.4248 }),
            wheelbase: Some(1.3),
            law: TurnLaw::Bicycle { wheelbase: 1.3 },
            law_note: None,
            turn: TurnRate { yaw_deg: [50.0, 16.67], pitch_deg: [0.0, 0.0], max_speed_reference: 8.0, exponent: 2.0 },
            gravity_scale: 1.0,
            ride_height: 0.25,
            // No contact geometry and no drivetrain: these fixtures are about the TURN LAW, and
            // this is the "fewer than three markers" state `law_note` reports -- the hull is held
            // at a flat ride height so nothing the suspension does can contribute to the number
            // under test.
            drivetrain: None,
            contacts: None,
            spring: no_spring(),
            ground_friction: 0.0,
        };
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let at_rest = DriveState { pos: Vec3::new(0.0, 0.0, 0.25), yaw: 0.0, ..Default::default() };

        // --- 1. PARKED, FULL STICK, FOR TWO SECONDS: not one radian of yaw -----------------
        let mut st = at_rest;
        for _ in 0..120 {
            st = d.step(st, DriveInput { throttle: 0.0, slide: 1.0, want_yaw: 3.0, brake: true }, 1.0 / 60.0, &flat);
        }
        println!("parked, full stick for 2 s: yaw {:+.6} rad, steer angle {:+.4} rad", st.yaw, st.steer_angle);
        assert_eq!(st.yaw, 0.0, "a PARKED wheeled vehicle must not turn at all, got {:+.6} rad", st.yaw);
        // ...and yet the wheels must be AT LOCK, so the gate is not passing because the stick
        // was ignored. This is what makes the first assertion mean something.
        assert!(
            (st.steer_angle - lock).abs() < 1e-3,
            "the wheels must still be turned to full lock while parked, got {:+.4} of {:+.4} rad",
            st.steer_angle,
            lock
        );

        // --- 2. THE YAW RATE IS LINEAR IN SPEED --------------------------------------------
        //
        // One tick from a known speed and a known steer angle, so nothing else can contribute.
        //
        // **The stick has to COMMAND the angle it starts at.** The first version of this seeded
        // `steer_angle` and then commanded `slide: 0.0`, so the angle slewed a third of the way
        // back to centre before the yaw was taken from it, and the gate's own arithmetic missed
        // by a factor of 0.53 — a fixture error, not a bug, and recognisable as one precisely
        // because the wrong number was derivable from the fixture (`tan(0.349 - 9.4248/60)`
        // instead of `tan(0.349)`). Commanding `20/60` of full lock makes the target equal the
        // seed, so the slew is a no-op and the tick is a clean reading of `v*tan(d)/L`.
        let hold = 20.0f32.to_radians();
        let one_tick = |speed: f32| -> f32 {
            let st = DriveState {
                pos: Vec3::new(0.0, 0.0, 0.25),
                vel: Vec3::new(speed, 0.0, 0.0),
                yaw: 0.0,
                steer_angle: hold,
                grounded: true,
                ..Default::default()
            };
            // **The throttle has to SUSTAIN the speed too.** `throttle: 0.0` asked for a target
            // of zero, and once `step` began sub-stepping (it has to: the suspension spring is
            // not stable at a 100 ms frame) the fixture's own 100 wu/s/s deceleration took a
            // third of the speed away INSIDE the tick, so the second sub-step integrated yaw at a
            // different speed than the first and the ratio came out at 2.26. A fixture error, and
            // recognisable as one because the wrong number is derivable from the fixture. Asking
            // for exactly the speed it already has makes `approach` a no-op and the tick a clean
            // reading of `v*tan(d)/L`.
            let out = d.step(
                st,
                DriveInput { throttle: speed / 8.0, slide: hold / lock, want_yaw: 0.0, brake: false },
                1.0 / 60.0,
                &flat,
            );
            assert!(
                (out.vel.length() - speed).abs() < 1e-4,
                "the fixture must hold its speed, got {} of {speed}",
                out.vel.length()
            );
            // The slew must have been a no-op, or the reading below is of a different angle.
            assert!((out.steer_angle - hold).abs() < 1e-5, "the fixture must hold its steer angle, got {:+.5}", out.steer_angle);
            out.yaw
        };
        let (y2, y4) = (one_tick(2.0), one_tick(4.0));
        println!("one tick at 2 wu/s: {y2:+.6} rad | at 4 wu/s: {y4:+.6} rad  (ratio {:.4})", y4 / y2);
        assert!(y2 > 1e-5, "a rolling vehicle must turn, got {y2:+.6}");
        assert!(
            (y4 / y2 - 2.0).abs() < 0.02,
            "doubling the speed must double the yaw rate; 2 wu/s gave {y2:+.6} and 4 wu/s gave {y4:+.6} (ratio {:.4})",
            y4 / y2
        );
        // And the arithmetic itself, independently: omega = v*tan(d)/L.
        let want = 2.0 * 20.0f32.to_radians().tan() / 1.3 / 60.0;
        assert!((y2 - want).abs() < 1e-5, "one tick at 2 wu/s should be {want:+.6} rad, got {y2:+.6}");

        // --- 3. THE TURNING CIRCLE, two independent ways -----------------------------------
        let stated = d.turning_radius().expect("a bicycle model has a turning radius");
        let arithmetic = 1.3 / lock.tan();
        println!("turning radius: stated {stated:.4} wu, L/tan(lock) = {arithmetic:.4} wu");
        assert!((stated - arithmetic).abs() < 1e-4);
        // Now drive a full circle at lock and measure the radius the integrator traces. The hull
        // yaws through 2*pi and the path's radius is the distance travelled over that angle.
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.25), yaw: 0.0, grounded: true, ..Default::default() };
        let mut turned = 0.0f32;
        let mut dist = 0.0f32;
        let mut prev = st;
        for _ in 0..4000 {
            st = d.step(st, DriveInput { throttle: 1.0, slide: 1.0, want_yaw: 0.0, brake: false }, 1.0 / 60.0, &flat);
            turned += crate::reach_vehicle::wrap_angle(st.yaw - prev.yaw).abs();
            dist += (st.pos - prev.pos).truncate().length();
            prev = st;
            if turned >= std::f32::consts::TAU {
                break;
            }
        }
        let traced = dist / turned.max(1e-6);
        println!("traced a {:.1} deg arc over {dist:.3} wu -> radius {traced:.4} wu", turned.to_degrees());
        assert!(turned >= std::f32::consts::TAU - 0.1, "the vehicle must complete a circle, got {:.1} deg", turned.to_degrees());
        // The slew to lock happens over the first fraction of a second, so the traced radius is
        // slightly larger than the steady-state one; 15% is loose enough for that and far tighter
        // than any other law would give (a heading servo traces a radius of nearly zero).
        assert!(
            (traced - arithmetic).abs() / arithmetic < 0.15,
            "the traced radius {traced:.4} wu is not the authored L/tan(lock) = {arithmetic:.4} wu"
        );

        // --- 4. THE CAMERA MUST NOT STEER --------------------------------------------------
        let drive = |want_yaw: f32| -> f32 {
            let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 0.25), yaw: 0.0, grounded: true, ..Default::default() };
            for _ in 0..90 {
                st = d.step(st, DriveInput { throttle: 1.0, slide: 0.0, want_yaw, brake: false }, 1.0 / 60.0, &flat);
            }
            st.yaw
        };
        let (ahead, behind) = (drive(0.0), drive(std::f32::consts::PI));
        println!("driving straight with the camera ahead: {ahead:+.6} rad | with it swung 180 deg: {behind:+.6} rad");
        assert_eq!(
            ahead, behind,
            "the CAMERA must not steer a wheeled vehicle -- looking 180 deg away changed the heading from {ahead:+.6} to {behind:+.6}"
        );
    }

    /// **THE TANK GATE.** A tracked vehicle turns through its tracks, so equal track speeds must
    /// give exactly no yaw, and it may rotate on the spot — which is the one place a dime pivot
    /// is correct.
    ///
    /// The law is DERIVED FROM THE FIELD NAMES AND THE MEASURED TRACK, not disassembled, and this
    /// gate asserts the relationships that follow from that reading rather than a fitted number.
    #[test]
    fn a_tracked_vehicle_turns_through_its_tracks() {
        use glam::Vec3;
        // The Scorpion's measured track is 1.7400 wu (twelve friction points).
        let track = 1.74f32;
        let diff = 2.0f32; // wu/s of track-speed difference at full stick
        let d = DriveModel {
            kind: "human_tank",
            max_forward_speed: 4.0,
            max_reverse_speed: 2.0,
            speed_source: SpeedSource::SeatMaxSpeedReference,
            speed_acceleration: 100.0,
            speed_deceleration: 100.0,
            slide: VelocityModel::default(),
            turning: Some(Turning { max_left: diff, max_right: -diff, turn_rate: 8.0 }),
            wheelbase: Some(2.8316),
            law: TurnLaw::Differential { track },
            law_note: None,
            turn: TurnRate { yaw_deg: [30.0, 15.0], pitch_deg: [0.0, 0.0], max_speed_reference: 4.0, exponent: 2.0 },
            gravity_scale: 1.0,
            ride_height: 0.375,
            drivetrain: None,
            contacts: None,
            spring: no_spring(),
            ground_friction: 0.0,
        };
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let rest = DriveState { pos: Vec3::new(0.0, 0.0, 0.375), yaw: 0.0, grounded: true, ..Default::default() };

        // Equal track speeds — no stick — and the heading must not move however far it drives.
        let mut st = rest;
        for _ in 0..180 {
            st = d.step(st, DriveInput { throttle: 1.0, slide: 0.0, want_yaw: 2.0, brake: false }, 1.0 / 60.0, &flat);
        }
        println!("tracks equal, driven 3 s: yaw {:+.6} rad, travelled {:.3} wu", st.yaw, st.pos.truncate().length());
        assert_eq!(st.yaw, 0.0, "equal track speeds must give no yaw at all, got {:+.6}", st.yaw);
        assert!(st.pos.truncate().length() > 1.0, "...and it must still have driven, or the assertion above is vacuous");

        // Opposite and equal — full stick, no throttle — and it must rotate where it stands.
        let mut st = rest;
        for _ in 0..180 {
            st = d.step(st, DriveInput { throttle: 0.0, slide: 1.0, want_yaw: 0.0, brake: false }, 1.0 / 60.0, &flat);
        }
        let moved = st.pos.truncate().length();
        println!("full differential, 3 s: yaw {:+.3} rad ({:.1} deg), moved {moved:.4} wu", st.yaw, st.yaw.to_degrees());
        assert!(st.yaw.abs() > 1.0, "a tank must turn on the spot, got {:+.3} rad", st.yaw);
        assert!(moved < 0.05, "...and it must not translate while doing it, got {moved:.4} wu");

        // The rate is the differential over the measured TRACK: a wider tank turns slower for the
        // same differential. Two models, one changed quantity, so the relationship is the subject.
        let wide = DriveModel { law: TurnLaw::Differential { track: track * 2.0 }, ..d.clone() };
        let spin = |m: &DriveModel| -> f32 {
            let mut st = DriveState {
                pos: Vec3::new(0.0, 0.0, 0.375),
                yaw: 0.0,
                steer_angle: diff,
                grounded: true,
                ..Default::default()
            };
            st = m.step(st, DriveInput { throttle: 0.0, slide: 1.0, want_yaw: 0.0, brake: false }, 1.0 / 60.0, &flat);
            st.yaw
        };
        let (narrow_rate, wide_rate) = (spin(&d), spin(&wide));
        println!("one tick: track {track:.3} -> {narrow_rate:+.6} rad | track {:.3} -> {wide_rate:+.6} rad", track * 2.0);
        assert!(narrow_rate > 1e-5);
        assert!(
            (narrow_rate / wide_rate - 2.0).abs() < 0.02,
            "doubling the track must halve the yaw rate; got {narrow_rate:+.6} and {wide_rate:+.6}"
        );
    }

    /// The integrator's own invariants, with no map: a vehicle must not exceed its authored
    /// speed, must come to rest under braking, must fall when there is no ground, and must turn
    /// at the authored rate and no faster.
    #[test]
    fn the_drive_integrator_respects_every_authored_limit() {
        use glam::Vec3;
        let d = DriveModel {
            kind: "alien_scout",
            max_forward_speed: 5.0,
            max_reverse_speed: 2.5,
            speed_source: SpeedSource::Authored,
            speed_acceleration: 6.0,
            speed_deceleration: 8.0,
            slide: VelocityModel {
                max_forward_speed: 5.0,
                max_left_slide: 4.0,
                max_right_slide: 4.0,
                slide_acceleration: 4.0,
                slide_deceleration: 3.0,
                slide_speed_at_top_speed: 1.8,
                ..Default::default()
            },
            turning: None,
            wheelbase: None,
            law: TurnLaw::HeadingServo,
            law_note: None,
            // 50 deg/s at rest falling to 16.67 at 8.5 wu/s, exponent 2 -- the Ghost's own.
            turn: TurnRate { yaw_deg: [50.0, 16.67], pitch_deg: [0.0, 0.0], max_speed_reference: 8.5, exponent: 2.0 },
            gravity_scale: 1.0,
            ride_height: 0.35,
            drivetrain: None,
            contacts: None,
            spring: no_spring(),
            ground_friction: 0.0,
        };
        let flat = |_p: Vec3| Some((0.0f32, Vec3::Z));
        let none = |_p: Vec3| None;

        // The authored turn rate at rest is 50 deg/s and the model must not beat it. `step`
        // clamps `dt` to 0.1 s (a stalled frame must not teleport a vehicle), so one second is
        // sixty frames, not one call with dt = 1.
        let mut st = DriveState::default();
        let input = DriveInput { want_yaw: std::f32::consts::PI, ..Default::default() };
        for _ in 0..60 {
            st = d.step(st, input, 1.0 / 60.0, &flat);
        }
        let turned = st.yaw.abs().to_degrees();
        assert!(turned <= 50.0 + 1e-3, "turned {turned:.2} deg in one second against an authored 50");
        assert!(turned >= 49.0, "turned only {turned:.2} deg -- the authored rate is not being used");

        // Forward is capped by the authored forward speed.
        let mut st = DriveState::default();
        let go = DriveInput { throttle: 1.0, ..Default::default() };
        for _ in 0..600 {
            st = d.step(st, go, 1.0 / 60.0, &flat);
        }
        let v = Vec3::new(st.vel.x, st.vel.y, 0.0).length();
        assert!(v <= 5.0 + 1e-3, "full throttle settled at {v}, authored 5.0");
        assert!(v > 4.8, "full throttle only reached {v} of an authored 5.0");

        // Reverse is capped by its own authored number, not by the forward one.
        let mut st = DriveState::default();
        let back = DriveInput { throttle: -1.0, ..Default::default() };
        for _ in 0..600 {
            st = d.step(st, back, 1.0 / 60.0, &flat);
        }
        let v = Vec3::new(st.vel.x, st.vel.y, 0.0).length();
        assert!(v <= 2.5 + 1e-3 && v > 2.3, "reverse settled at {v}, authored 2.5");

        // Braking brings it to rest.
        let brake = DriveInput { brake: true, ..Default::default() };
        for _ in 0..600 {
            st = d.step(st, brake, 1.0 / 60.0, &flat);
        }
        assert!(Vec3::new(st.vel.x, st.vel.y, 0.0).length() < 1e-2, "braking left {:?}", st.vel);

        // With no ground it falls, at the MEASURED gravity times the authored scale -- and the
        // scale is honoured, which is what the second case proves.
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 10.0), ..Default::default() };
        for _ in 0..60 {
            st = d.step(st, DriveInput::default(), 1.0 / 60.0, &none);
        }
        // #gravity  A VEHICLE is on the engine's authored world constant, which was measured in
        // the vehicle's OWN physics path -- see
        // [`crate::reach_physics::vehicle_fall_acceleration`]. It is deliberately not
        // `rigid_body_fall_acceleration`, which this gate used to read and which has no engine
        // basis at all.
        let g = crate::reach_physics::vehicle_fall_acceleration();
        assert!((st.vel.z + g).abs() < 1e-3, "one second of free fall gave vz {} against g {g}", st.vel.z);
        assert!(!st.grounded);
        let half = DriveModel { gravity_scale: 0.5, ..d.clone() };
        let mut st2 = DriveState { pos: Vec3::new(0.0, 0.0, 10.0), ..Default::default() };
        for _ in 0..60 {
            st2 = half.step(st2, DriveInput::default(), 1.0 / 60.0, &none);
        }
        assert!((st2.vel.z + g * 0.5).abs() < 1e-3, "an authored 0.5 gravity scale must halve the fall");

        // On the ground it rests at the authored ride height and stays there.
        let mut st = DriveState { pos: Vec3::new(0.0, 0.0, 5.0), ..Default::default() };
        for _ in 0..600 {
            st = d.step(st, DriveInput::default(), 1.0 / 60.0, &flat);
        }
        assert!((st.pos.z - 0.35).abs() < 1e-3, "settled at {} against an authored ride height of 0.35", st.pos.z);
        assert!(st.grounded);

        // The slide ceiling must fall off with speed exactly as the field documents.
        let mut st = DriveState::default();
        let slide = DriveInput { slide: 1.0, ..Default::default() };
        for _ in 0..600 {
            st = d.step(st, slide, 1.0 / 60.0, &flat);
        }
        let s = st.vel.dot(Vec3::new(-st.yaw.sin(), st.yaw.cos(), 0.0));
        assert!(s > 3.5 && s <= 4.0 + 1e-3, "sliding from rest settled at {s} against an authored 4.0");
    }

    /// A refusal must be a refusal: a family this module does not drive gets `Err` with its name,
    /// not a default model.
    #[test]
    fn an_unsupported_family_refuses_rather_than_getting_invented_dynamics() {
        let p = VehiclePhysics {
            vehi_tag: 1,
            family: Family::Unsupported { kind: "alien_fighter", offset: 0x50C, size: 100 },
            havok: Havok::default(),
            flags: 0,
            flipping_angular_velocity: [0.0, 0.0],
            seat_entrance_exit_accel_scale: [0.0, 0.0],
            flip_message: String::new(),
            wheel_positions: Vec::new(),
            antigravity_positions: Vec::new(),
            jeep_wheel_circumference: 0.0,
            acceleration_scales: (1.0, 1.0, 1.0),
        };
        let e = DriveModel::for_vehicle(&p, None).expect_err("a flyer must refuse");
        assert!(e.contains("alien_fighter"), "the refusal must name the family, got {e:?}");
        // #vehicle-law  ...and it must say what is MISSING, so a player who sits in a Banshee and
        // finds it does not fly reads why on screen instead of filing it as a mystery. The user's
        // report was *"we need the right physics for ALL the vehicles"*; a refusal that does not
        // name its own gap reads as a bug rather than as an unfinished lane.
        assert!(e.contains("FLIES"), "a flyer's refusal must say it flies, got {e:?}");
        assert!(e.contains("0x50C"), "...and name the block it would need, got {e:?}");
        assert!(e.contains("Banshee"), "...and the vehicle a player would be sitting in, got {e:?}");
        // and a jeep with no authored speed anywhere refuses too, rather than driving at zero
        let p2 = VehiclePhysics {
            family: Family::Jeep {
                steering: Steering::default(),
                turning: Turning::default(),
                engine: Engine::default(),
                limits: None,
                gravity_adjust: 0.0,
            },
            ..p
        };
        let e2 = DriveModel::for_vehicle(&p2, None).expect_err("no speed anywhere must refuse");
        assert!(e2.contains("max speed reference"), "got {e2:?}");
    }

    /// The authored turn-rate law, against the Ghost's own numbers and the block's own words.
    #[test]
    fn the_turn_rate_follows_the_blocks_own_description() {
        let t = TurnRate { yaw_deg: [50.0, 16.67], pitch_deg: [20.0, 10.0], max_speed_reference: 8.5, exponent: 2.0 };
        // "when the unit velocity is 0, the yaw/pitch rates are the LEFT values"
        assert!((t.yaw_rate(0.0).to_degrees() - 50.0).abs() < 1e-3);
        assert!((t.pitch_rate(0.0).to_degrees() - 20.0).abs() < 1e-3);
        // "at [max speed reference], the yaw/pitch rates are the RIGHT values"
        assert!((t.yaw_rate(8.5).to_degrees() - 16.67).abs() < 1e-3);
        // "the max speed reference is what the code uses to generate a clamped speed from 0..1"
        assert!((t.yaw_rate(1000.0).to_degrees() - 16.67).abs() < 1e-3, "clamped past the reference");
        // "the exponent controls how midrange speeds are interpreted" -- with exponent 2 the
        // halfway SPEED is a quarter of the way along the rate, not halfway
        let mid = t.yaw_rate(4.25).to_degrees();
        let quarter = 50.0 + (16.67 - 50.0) * 0.25;
        assert!((mid - quarter).abs() < 1e-2, "midrange gave {mid}, exponent 2 says {quarter}");
        // an unauthored reference must not divide by zero, and an unauthored exponent is linear
        let z = TurnRate { yaw_deg: [30.0, 30.0], max_speed_reference: 0.0, exponent: 0.0, ..Default::default() };
        assert!((z.yaw_rate(5.0).to_degrees() - 30.0).abs() < 1e-3);
        let lin = TurnRate { yaw_deg: [10.0, 20.0], max_speed_reference: 10.0, exponent: 0.0, ..Default::default() };
        assert!((lin.yaw_rate(5.0).to_degrees() - 15.0).abs() < 1e-3, "exponent 0 must be linear");
    }

    /// #vehicle-suspension  **HOW MANY GROUND CONTACT POINTS DOES EACH VEHICLE ACTUALLY AUTHOR?**
    ///
    /// This exists because a prior brief recorded *"ghost 3, wraith 6, revenant 6, falcon 3"*
    /// anti-gravity points and the first run of the physics reader found **zero** on the Ghost. One
    /// of the two is wrong and the difference decides whether a hover vehicle can be made to
    /// conform to terrain at all, so it is measured here rather than argued about.
    ///
    /// The offset is not in doubt: the engine's own postprocess (`reach_tag_test.exe`
    /// `sub_1406D51E0`) walks the anti-gravity block at `havok + 0x38` with a stride of `0x48`
    /// and the friction block at `havok + 0x44` with a stride of `0x34`, which is exactly what
    /// [`HV_ANTIGRAVITY_POINTS`] / [`AGP_SIZE`] and [`HV_FRICTION_POINTS`] / [`FP_SIZE`] say.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_how_many_contact_points_each_vehicle_authors() {
        let Some((_g, native, cache)) = open() else { return };
        let mut any = false;
        let mut hover_with_points = 0usize;
        let mut hover_without = Vec::new();
        let native = &*native;
        for t in native.tags_by_group(cache, "vehi") {
            let Some(p) = VehiclePhysics::read(native, cache, t) else { continue };
            let path = native.tag_name(cache, t).unwrap_or_default();
            let short = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
            let (f, a) = (p.havok.friction_points.len(), p.havok.antigravity_points.len());
            println!(
                "{short:28} {:14} friction {f:2} antigrav {a:2}  depth {:.3} damp {:.3} slope1 cos {:.3}",
                p.family.kind(),
                p.havok.ground_depth,
                p.havok.ground_damp_factor,
                p.havok.ground_max_slope_cos[1]
            );
            any = true;
            if f == 0 && p.drivable() {
                if a > 0 { hover_with_points += 1 } else { hover_without.push(short) }
            }
        }
        assert!(any, "no vehicles read at all -- the fixture is empty and this probe proves nothing");
        println!(
            "drivable vehicles with NO friction points: {} do author anti-gravity points, {} author neither: {:?}",
            hover_with_points,
            hover_without.len(),
            hover_without
        );
    }

    /// The derived ceiling is arithmetic over authored values, so it is testable with no map at
    /// all — and the refusals are tested too, because a `None` that never happens is a `None`
    /// nobody has checked.
    #[test]
    fn the_top_speed_ceiling_is_engine_times_top_gear() {
        let e = Engine {
            moment: 2000.0,
            max_angular_velocity: 9.5,
            gears: vec![gear(0, -0.5, 0.0, 0.0), gear(1, 0.75, 0.85, 0.0), gear(2, 1.0, 0.85, 0.25), gear(3, 1.2, 0.0, 0.25)],
        };
        let l = DriveLimits::from_engine(&e, 1.249).expect("the Warthog's own gearbox");
        // top gear = the LARGEST positive ratio (gear 3), not the smallest and not the last index
        // by accident -- gear 3 happens to be both here, so the next case separates them.
        assert_eq!(l.top_gear, 3);
        // #vehicle-drivetrain  `max angular velocity * top ratio` and NOT times the
        // circumference -- this gate used to assert the circumference factor, and it was wrong.
        // The three-way reading that settled it: `sub_140E600E0` returns `ratio * aav`, the jeep
        // wraps `that * dt` at the circumference, and the corrected ceiling is 1.27x the driver
        // seat's `max speed reference` on BOTH the Warthog and the Mongoose where the old one was
        // 1.58x and 1.89x.
        assert!((l.top_speed_ceiling - 9.5 * 1.2).abs() < 1e-3, "got {}", l.top_speed_ceiling);
        assert!(
            (l.top_speed_ceiling - 11.4).abs() < 1e-3,
            "the Warthog's drivetrain ceiling is 11.40 wu/s, got {}",
            l.top_speed_ceiling
        );
        // the largest ratio NOT at the last index: the rule must still pick it
        let e2 = Engine { moment: 1.0, max_angular_velocity: 10.0, gears: vec![gear(0, 1.5, 0.0, 0.1), gear(1, 0.8, 0.1, 0.0)] };
        let l2 = DriveLimits::from_engine(&e2, 1.0).unwrap();
        assert_eq!(l2.top_gear, 0, "the largest ratio wins wherever it sits");
        // and the refusals
        // The circumference no longer MULTIPLIES the ceiling, but a vehicle that authors none
        // authors no rolling phase either, so it is still the signal that this block is not a
        // wheeled drivetrain and the refusal stands.
        assert!(DriveLimits::from_engine(&e, 0.0).is_none(), "no wheel circumference -> no ceiling");
        assert!(
            DriveLimits::from_engine(&Engine { moment: 1.0, max_angular_velocity: 0.0, gears: e.gears.clone() }, 2.0).is_none(),
            "no engine speed -> no ceiling"
        );
        assert!(
            DriveLimits::from_engine(&Engine { moment: 1.0, max_angular_velocity: 30.0, gears: vec![gear(0, 0.0, 0.0, 0.0), gear(1, -1.0, 0.0, 0.0)] }, 2.0).is_none(),
            "no positive gear ratio -> no ceiling, not a division by zero"
        );
    }

    /// The slide ceiling must follow the field's own documentation, and must do nothing when the
    /// value is not authored.
    #[test]
    fn the_slide_ceiling_interpolates_to_the_authored_top_speed_value() {
        let m = VelocityModel {
            max_forward_speed: 10.0,
            max_left_slide: 4.0,
            max_right_slide: 4.0,
            slide_speed_at_top_speed: 1.0,
            ..Default::default()
        };
        assert_eq!(m.slide_ceiling(0.0), (4.0, 4.0), "at rest the full slide is available");
        assert_eq!(m.slide_ceiling(10.0), (1.0, 1.0), "at top speed it is the authored value");
        let (l, _) = m.slide_ceiling(5.0);
        assert!((l - 2.5).abs() < 1e-5, "halfway is halfway, got {l}");
        assert_eq!(m.slide_ceiling(1e6), (1.0, 1.0), "clamped past top speed");
        // not authored -> no interpolation at all
        let m2 = VelocityModel { max_forward_speed: 10.0, max_left_slide: 4.0, max_right_slide: 3.0, ..Default::default() };
        assert_eq!(m2.slide_ceiling(10.0), (4.0, 3.0));
    }
}
