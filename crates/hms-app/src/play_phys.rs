//! #play  The PLAYER MOVEMENT CONSTANTS, read out of the map's own tags.
//!
//! Test mode walks a Spartan or an Elite around the map the user is building. How fast that
//! walk is, how high the jump goes, how tall the capsule is and how steep a slope it can climb
//! are **authored facts in the cache**, not choices — and a user who has played thousands of
//! hours of Reach would notice an invented walk speed immediately. So every number here comes
//! from a tag field, with the offsets taken from Blam's own reflection data
//! (`reverseme tagdefs reach_tag_test.exe`, the dump at
//! `.reverseme/tagdefs/reach_tag_test.cbdd8448.txt`, which re-sums exactly against 1556 of 1611
//! compiled-in element sizes).
//!
//! # Where each number lives
//!
//! **Speeds and accelerations** are per SPECIES, in `matg` (globals) `@player information`
//! ([`PLAYER_INFO`]), a 244-byte block with up to three elements:
//!
//! | offset | field |
//! |--------|-------|
//! | +0x00  | walking speed [wu/s] |
//! | +0x04  | run forward [wu/s] |
//! | +0x08  | run backward [wu/s] |
//! | +0x0C  | run sideways [wu/s] |
//! | +0x10  | run acceleration [wu/s²] |
//! | +0x14  | sneak forward [wu/s] (= crouched) |
//! | +0x18  | sneak backward [wu/s] |
//! | +0x1C  | sneak sideways [wu/s] |
//! | +0x20  | sneak acceleration [wu/s²] |
//! | +0x24  | airborne acceleration [wu/s²] |
//!
//! The offset of the block itself is CROSS-CHECKED against code this project already verified:
//! the dump puts `@player representation` at `matg +0x474`, which is exactly where
//! `SceneController::first_person_models` has always read the first-person arms from, and
//! `@player information` is the 12-byte block immediately before it at `+0x468`.
//!
//! **The body** is per BIPED, in `bipd` (fields from +0x4CC — the `$$$ BIPED $$$` segment):
//! jump velocity, the four camera (eye) heights, the crouch transition time, the character
//! physics struct (standing height, crouching height, radius, mass) and inside that the ground
//! physics struct (maximum slope angle, gravity scale, airborne acceleration scale). A Spartan
//! and an Elite differ, and because these are read per tag they differ correctly.
//!
//! **Gravity** is NOT in the tags. It is already measured in this tree off the engine's own
//! recordings — [`crate::film_play::CORPSE_GRAVITY`], 10.4576 wu/s², the slope of vz against
//! time over 1,447 unmistakably-unsupported falls on 12 films. (The 3.2147 in
//! `physics_settle` is Earth gravity converted through a world-unit guess and is NOT what this
//! engine uses; nothing here reads it.)
//!
//! **Angles are stored in RADIANS in the cache** even where the editor labels them
//! "[degrees per second]" — Blam's `angle` field type converts on import. [`PlayerPhysics`]
//! keeps them in radians and says so in the field names.
//!
//! # What is deliberately absent
//!
//! There is **no step-up height**, because Reach does not author one: the character proxy is a
//! Havok capsule and stepping over a small ledge is emergent from its bottom hemisphere (whose
//! radius IS authored, above) plus the slope-limit rejection. See `crate::play_ctl`, which is
//! why that module has no step constant either.

use hms_native::{CacheHandle, NativeDll};

// ---------------------------------------------------------------------------------------------
// tag offsets
// ---------------------------------------------------------------------------------------------

/// `matg` `@player information` block header (count, pointer). 244-byte elements, max 3.
pub const PLAYER_INFO: usize = 0x468;
/// One `@player information` element, bytes (`sizeof(s_game_globals_player_information)`).
pub const PLAYER_INFO_ELEM: usize = 244;
/// `matg` `@player representation` — the block that names, per playable species, the
/// first-person hands, the first-person body and **the third-person unit**, i.e. the player's own
/// biped. 108-byte elements, max 3.
///
/// Also the CHECK that [`PLAYER_INFO`] is where the reflection data says, because this one is
/// independently verified elsewhere in HMS (`SceneController::first_person_models`).
pub const PLAYER_REPRESENTATION: usize = 0x474;
/// One `@player representation` element, bytes.
pub const PLAYER_REPRESENTATION_ELEM: usize = 108;
/// `first person hands` / `first person body` / `third person unit` tag references within it.
pub const REP_HANDS: usize = 0x14;
pub const REP_BODY: usize = 0x24;
pub const REP_THIRD_PERSON_UNIT: usize = 0x34;

/// `@player information` SPRINT fields (the block's own `-- sprinting` section).
pub const PI_SPRINT_SECONDS_TO_START: usize = 0xCC;
pub const PI_SPRINT_SECONDS_TO_FULL: usize = 0xD0;
pub const PI_SPRINT_FULL_SPEED_MULTIPLIER: usize = 0xD8;
pub const PI_SPRINT_MIN_VELOCITY: usize = 0xE8;
/// `@player representation` `player information` — a BLOCK INDEX saying which
/// `@player information` row this species uses. The engine's own pairing, so nothing here has to
/// assume the two blocks are in the same order.
pub const REP_PLAYER_INFO_INDEX: usize = 0x68;

// NOTE: there is deliberately no terminal-velocity field here. An earlier version read one at
// `@player information +0x50` and it came out as **0.8 wu/s**, which would mean a Spartan dies
// falling slower than it walks -- because +0x50 is `first person skip fraction [[0,1]]`, and the
// 0.8 was that fraction. The real `terminal velocity` belongs to another block. Caught by dumping
// the values off a real map (`--dump-play-phys`) and reading them, which is exactly why that verb
// exists: a wrong offset does not fail, it returns a plausible-looking number.

/// `bipd` jump velocity [wu/s].
pub const BIPD_JUMP_VELOCITY: usize = 0x4F0;
/// `bipd` moving turning speed — radians/s in the cache ("[degrees per second]" in the editor).
pub const BIPD_TURNING_SPEED: usize = 0x4CC;
/// `bipd` stationary turning threshold, radians. **This field's whole purpose is to let a
/// standing player look around without their feet moving**: below it the body does not turn at
/// all, and past it it turns to catch the aim up.
pub const BIPD_STATIONARY_TURN_THRESHOLD: usize = 0x4D4;
/// `bipd` standing / running / crouching / crouch-walking camera (eye) heights [wu].
pub const BIPD_CAMERA_HEIGHTS: usize = 0x524;
/// `bipd` crouch transition time [s].
pub const BIPD_CROUCH_TRANSITION: usize = 0x534;
/// #crouch-rate  `bipd` `camera height velocity function` — a `mapping_function` (variable-size
/// tag data, which is why the reflection dump gives it size 0). This is the SHAPE the engine
/// moves the eye with; a non-empty one means the move is not linear in time.
pub const BIPD_CAMERA_HEIGHT_FUNCTION: usize = 0x538;
/// #crouch-rate  `bipd` `camera heights` — per weapon class `standing`/`crouching` heights
/// (12-byte `s_biped_camera_height`, max 16), which OVERRIDE the four scalars above.
pub const BIPD_CAMERA_HEIGHTS_BLOCK: usize = 0x54C;
/// #crouch-rate  `bipd` `runtime crouch transition velocity` — derived at tag post-process, so
/// reading it says how the engine itself turns the transition TIME into a rate.
pub const BIPD_RUNTIME_CROUCH_VELOCITY: usize = 0x580;
/// #crouch-rate  `bipd` `runtime_camera_height_velocity` — the same for the EYE, separately.
pub const BIPD_RUNTIME_CAMERA_HEIGHT_VELOCITY: usize = 0x584;
/// `bipd` `physics` — `character_physics_struct`, 184 bytes.
pub const BIPD_PHYSICS: usize = 0x5E8;
/// `character_physics_struct` `ground physics` — `character_physics_ground_struct`, 68 bytes.
pub const PHYSICS_GROUND: usize = 0x44;

// ---------------------------------------------------------------------------------------------
// the constants
// ---------------------------------------------------------------------------------------------

/// Every movement constant test mode needs, read from one `bipd` and one `matg`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerPhysics {
    // --- matg @player information (wu/s, wu/s²) ---
    pub walk_speed: f32,
    pub run_forward: f32,
    pub run_backward: f32,
    pub run_sideways: f32,
    pub run_acceleration: f32,
    pub sneak_forward: f32,
    pub sneak_backward: f32,
    pub sneak_sideways: f32,
    pub sneak_acceleration: f32,
    pub airborne_acceleration: f32,
    /// Sprint, authored: how long the stick must be pegged before a sprint starts, how long the
    /// sprint takes to reach top speed, and **how much faster top speed is** — so sprint is a
    /// read value and not a chosen one.
    pub sprint_seconds_to_start: f32,
    pub sprint_seconds_to_full: f32,
    pub sprint_full_speed_multiplier: f32,
    /// Below this the engine does not consider the player to be sprinting at all, wu/s.
    pub sprint_min_velocity: f32,
    // --- bipd ---
    /// Jump impulse, wu/s, straight up (`bipd` +0x4F0).
    pub jump_velocity: f32,
    /// Eye height standing / running / crouching / crouch-walking, wu above the feet.
    pub eye_standing: f32,
    pub eye_running: f32,
    pub eye_crouching: f32,
    pub eye_crouch_walking: f32,
    /// How long the eye takes to travel between the standing and crouching heights, s.
    pub crouch_transition: f32,
    /// #crouch-rate  `bipd` +0x580 `runtime crouch transition velocity` and +0x584
    /// `runtime_camera_height_velocity`, as the CACHE holds them. Both are post-process
    /// derivations, so what they read says how the engine turns the authored TIME into a rate --
    /// and whether it does so at all. Read rather than assumed; see
    /// [`PlayerPhysics::crouch_rate`].
    pub runtime_crouch_velocity: f32,
    pub runtime_camera_height_velocity: f32,
    /// #crouch-rate  Does the biped author a `camera height velocity function` (+0x538)? A
    /// non-empty mapping function would mean the eye's travel is SHAPED rather than linear.
    pub camera_height_function_authored: bool,
    /// #crouch-rate  That function at u = 0, 0.25, 0.5, 0.75, 1 (NaN = none authored).
    pub camera_height_fn_samples: [f32; 5],
    /// #crouch-rate  The smallest and largest `standing - crouching` SPAN over the
    /// per-weapon-class rows, wu.
    pub camera_height_span: (f32, f32),
    /// #crouch-rate  Rows in the per-weapon-class `camera heights` block (+0x54C).
    pub camera_height_rows: usize,
    /// Capsule height standing / crouching, wu, and its radius.
    pub height_standing: f32,
    pub height_crouching: f32,
    pub radius: f32,
    pub mass: f32,
    /// The steepest ground that can be stood on and walked up, RADIANS from horizontal.
    pub max_slope: f32,
    /// Per-biped scale on gravity and on airborne acceleration.
    pub gravity_scale: f32,
    pub airborne_acceleration_scale: f32,
    /// Body turning speed, radians/s (what the third-person model's yaw chases).
    pub turning_speed: f32,
    /// How far the body's facing may lag the aim before it turns at all, radians. Authored.
    pub stationary_turn_threshold: f32,
    /// Which `@player information` row the speeds came from.
    pub species_row: usize,
}

/// #play  One `@player representation` row: which biped this species' player IS, and the two
/// first-person models it sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerRepresentation {
    /// Row index, which is also the pairing with `@player information`.
    pub row: usize,
    /// `first person hands` render model, or `0xFFFF_FFFF`.
    pub hands: u32,
    /// `first person body` render model, or `0xFFFF_FFFF`.
    pub body: u32,
    /// `third person unit` — the player's own `bipd`, or `0xFFFF_FFFF`.
    pub unit: u32,
    /// Which `@player information` row this species' speeds come from — the block index the row
    /// itself states (`+0x68`), falling back to its own index.
    pub info_row: usize,
}

/// A Reach tag reference inside tag meta -> tag index, or `0xFFFF_FFFF`. The same decode
/// `SceneController::first_person_models` uses on this very block.
fn tagref(b: &[u8], off: usize) -> u32 {
    if off + 16 > b.len() {
        return 0xFFFF_FFFF;
    }
    let d = u32::from_le_bytes([b[off + 0x0C], b[off + 0x0D], b[off + 0x0E], b[off + 0x0F]]);
    if d == 0xFFFF_FFFF {
        d
    } else {
        d & 0xFFFF
    }
}

// ---------------------------------------------------------------------------------------------
// #play  The AUTHORED speed row, public — for every caller that needs it, not just test mode
// ---------------------------------------------------------------------------------------------

/// #play  One playable species' authored movement row: `matg @player information`, paired to a
/// species by `@player representation` rather than by index or by a tag-path guess.
///
/// Exposed because test mode is not the only thing that needs these numbers. Theater picks a
/// body's locomotion clip from thresholds fitted to the film corpus, and those thresholds were
/// fitted to a histogram whose dominant mode was read as "the ordinary run" when it is in fact
/// **the sprint** — so a Spartan running flat out at its authored 2.2 wu/s falls below a 2.4
/// walk/run split and is drawn walking, and a sprint cut of 4.6 sits above the authored sprint
/// top speed of 3.57 and can never fire. One reader, so there is never a second copy of these
/// values to disagree with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayerSpeeds {
    /// Which `@player information` row this is — the index the representation row STATES
    /// ([`REP_PLAYER_INFO_INDEX`]), not its own position.
    pub row: usize,
    /// The biped this species' player IS (`third person unit`), or `0xFFFF_FFFF`.
    pub biped: u32,
    /// From that biped's own tag path ([`crate::film_anim::is_elite`]), never from the row index.
    pub elite: bool,
    pub walk: f32,
    pub run_forward: f32,
    pub run_backward: f32,
    pub run_sideways: f32,
    pub run_acceleration: f32,
    /// Crouched — the tag calls it `sneak`.
    pub sneak_forward: f32,
    pub sneak_backward: f32,
    pub sneak_sideways: f32,
    pub sneak_acceleration: f32,
    pub airborne_acceleration: f32,
    /// `full speed multiplier`: how much faster the forward run is at full sprint.
    pub sprint_multiplier: f32,
    pub sprint_seconds_to_start: f32,
    pub sprint_seconds_to_full: f32,
    /// Below this the engine does not consider the player to be sprinting at all, wu/s.
    pub sprint_min_velocity: f32,
}

impl PlayerSpeeds {
    /// Forward top speed at full sprint, wu/s. **This is the number a film histogram's dominant
    /// mode actually is**, which is why it is worth having.
    pub fn sprint_top_speed(&self) -> f32 {
        self.run_forward * self.sprint_multiplier.max(1.0)
    }

    /// A walk/run split for this species, wu/s.
    ///
    /// **Derived, not measured**: halfway between the authored walk and the authored run. There is
    /// no authored field for "where the animation changes", and nothing here knows where between
    /// two real speeds the switch belongs — only that a split *above the run itself* cannot be
    /// right. A caller with corpus evidence for the real switch should use it instead; this exists
    /// so that two callers with no such evidence at least derive the same thing.
    pub fn walk_run_split(&self) -> f32 {
        (self.walk + self.run_forward) * 0.5
    }

    /// The speed past which this species must be sprinting rather than running, wu/s. Derived the
    /// same way and with the same caveat: midway between the authored run cap and the authored
    /// sprint top speed.
    pub fn sprint_cut(&self) -> f32 {
        (self.run_forward + self.sprint_top_speed()) * 0.5
    }

    /// Are these values shaped like a player's? A read at a wrong offset does not fail, it
    /// returns a plausible-looking number — so a row that is not shaped right never reaches a
    /// caller at all.
    fn looks_authored(&self) -> bool {
        let fin = [
            self.walk,
            self.run_forward,
            self.run_backward,
            self.run_sideways,
            self.run_acceleration,
            self.sneak_forward,
            self.airborne_acceleration,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0);
        fin && self.walk < 20.0
            && self.run_forward > self.walk
            && self.run_forward < 40.0
            && self.run_acceleration < 1000.0
            && self.sprint_multiplier.is_finite()
            && self.sprint_multiplier >= 0.0
            && self.sprint_multiplier < 10.0
    }

    /// One line, for a caption or a log.
    pub fn describe(&self) -> String {
        format!(
            "{} (row {}, bipd {:#x}): walk {:.3} | run fwd/back/side {:.3}/{:.3}/{:.3} accel {:.2} \
             | crouch {:.3} | air accel {:.2} | sprint x{:.3} -> {:.3} wu/s \
             | derived cuts: walk/run {:.3}, run/sprint {:.3}",
            if self.elite { "Elite" } else { "Spartan" },
            self.row,
            self.biped,
            self.walk,
            self.run_forward,
            self.run_backward,
            self.run_sideways,
            self.run_acceleration,
            self.sneak_forward,
            self.airborne_acceleration,
            self.sprint_multiplier,
            self.sprint_top_speed(),
            self.walk_run_split(),
            self.sprint_cut(),
        )
    }
}

/// #play  Every playable species' authored movement row, in `@player representation` order.
///
/// **Empty** when the cache has no `matg`, states no rows, or the values do not look authored —
/// never a fabricated row, so a caller can fall back to its own constants on an empty result and
/// say in words that it did.
pub fn player_speeds(native: &NativeDll, cache: CacheHandle) -> Vec<PlayerSpeeds> {
    let Some(matg) = native.tags_by_group(cache, "matg").into_iter().next() else { return Vec::new() };
    let rows = PlayerPhysics::player_info_rows(native, cache, matg);
    if rows.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for rep in PlayerPhysics::player_representations(native, cache) {
        if rep.unit == 0xFFFF_FFFF || rep.unit == 0 {
            continue;
        }
        let Some(r) = rows.get(rep.info_row.min(rows.len() - 1)) else { continue };
        let path = native.tag_name(cache, rep.unit).unwrap_or_default();
        if path.is_empty() {
            continue;
        }
        let s = PlayerSpeeds {
            row: rep.info_row,
            biped: rep.unit,
            elite: crate::film_anim::is_elite(&path),
            walk: r[0],
            run_forward: r[1],
            run_backward: r[2],
            run_sideways: r[3],
            run_acceleration: r[4],
            sneak_forward: r[5],
            sneak_backward: r[6],
            sneak_sideways: r[7],
            sneak_acceleration: r[8],
            airborne_acceleration: r[9],
            sprint_seconds_to_start: r[10],
            sprint_seconds_to_full: r[11],
            sprint_multiplier: r[12],
            sprint_min_velocity: r[13],
        };
        if s.looks_authored() {
            out.push(s);
        }
    }
    out
}

/// #play  The authored row for one species, by the same authored pairing. `None` = not resolvable,
/// which a caller must treat as "use your own fallback and say so".
pub fn player_speeds_for(native: &NativeDll, cache: CacheHandle, elite: bool) -> Option<PlayerSpeeds> {
    let all = player_speeds(native, cache);
    all.iter().find(|s| s.elite == elite).copied().or_else(|| all.first().copied())
}

/// #gravity  Gravity for the PLAYER'S BODY, wu/s². **ONE name, ONE place.**
///
/// [`crate::reach_physics::biped_fall_acceleration`] — the engine's own authored world constant,
/// **4.1712594 wu/s²**, read out of the `"physics constants"` initialiser in `haloreach.dll` and
/// confirmed independently by the script function `physics_set_gravity`, which multiplies its
/// argument by the identical bit pattern. It is the only engine-sourced gravity that exists.
///
/// # This used to be a film fit, and the fit was wrong
///
/// It was 10.4576, the slope of vertical speed against time over falls in recordings. That fit
/// **clamped its samples at `g < 20.0`**, which threw away the entire real population and
/// reported the median of the clipped tail; re-run with the clamp widened the same selection
/// gives 31.38 at a spread of one part in a thousand. So the number was an artefact, the "2.507x
/// gap to the authored constant" that was the stated reason for preferring it was that artefact
/// divided by this constant, and the apex table that argued for it inherited both.
///
/// The user — who has played thousands of hours of Reach — reported the jump as far too low five
/// times, most plainly as *"our biped hardly makes it off the ground"*. The arithmetic agrees:
///
/// | gravity | Spartan apex (v = 2.300) | airtime |
/// |---------|--------------------------|---------|
/// | the old film fit, 10.4576 | 0.253 wu = 0.77 m (39% of standing height) | 0.44 s |
/// | **this, the engine's 4.1712594** | 0.634 wu = 1.93 m (98%) | 1.10 s |
///
/// # What is still open
///
/// **The biped's integration law is not measured.** Whether a character proxy is integrated at
/// the world constant at all, and whether the engine applies the jump impulse before or after a
/// tick's gravity (which costs a frame of velocity either way), are questions for the physics
/// campaign. This is the best engine-sourced answer available, labelled as such.
///
/// The RAGDOLLS deliberately did not move with it:
/// [`crate::reach_physics::rigid_body_fall_acceleration`] is a separate function with a separate
/// name, so changing how the player jumps cannot silently change how every corpse in Theater
/// falls. A Havok rigid body and a character proxy are different systems.
///
/// Nothing has to be re-tuned if this changes again: every gate asserts the RELATIONSHIP
/// (`apex = v² / 2g`, a jump returns to the z it left, doubling the authored velocity quadruples
/// the apex) rather than a number, so only the expected apex moves.
pub const GRAVITY: f32 = crate::reach_physics::biped_fall_acceleration();

/// #gravity  **False since the fit behind it was shown to be a clamp artefact.** Kept, and kept
/// named for the question it answers, so that anything printing "this number is a measurement"
/// stops saying so: [`GRAVITY`] is now the engine's own authored constant.
pub const GRAVITY_IS_A_FILM_FIT: bool = false;

impl PlayerPhysics {
    /// Read the constants for one biped out of the open cache. `Err` names what was missing —
    /// test mode refuses to start rather than walk at an invented speed.
    pub fn read(native: &NativeDll, cache: CacheHandle, bipd_tag: u32, row: usize) -> Result<PlayerPhysics, String> {
        let matg = native
            .tags_by_group(cache, "matg")
            .into_iter()
            .next()
            .ok_or_else(|| "this cache has no globals ('matg') tag: the player movement speeds are authored there".to_string())?;
        let rows = Self::player_info_rows(native, cache, matg);
        if rows.is_empty() {
            return Err(format!(
                "globals ('matg' {matg:#x}) has no '@player information' rows at +{PLAYER_INFO:#x}: \
                 the walk / run / sneak speeds are authored there and nothing here will invent them"
            ));
        }
        // WHICH ROW is the caller's, because the caller got it from `@player representation` —
        // the engine's own per-species pairing — rather than from a guess here. Clamped, and the
        // chosen row is reported in the caption, so a wrong pick is visible rather than silent.
        let species_row = row.min(rows.len() - 1);
        let r = &rows[species_row];
        let b = Self::biped_fields(native, cache, bipd_tag)
            .ok_or_else(|| format!("biped {bipd_tag:#x}: could not read its physics fields (bipd +{BIPD_PHYSICS:#x})"))?;
        let p = PlayerPhysics {
            walk_speed: r[0],
            run_forward: r[1],
            run_backward: r[2],
            run_sideways: r[3],
            run_acceleration: r[4],
            sneak_forward: r[5],
            sneak_backward: r[6],
            sneak_sideways: r[7],
            sneak_acceleration: r[8],
            airborne_acceleration: r[9],
            sprint_seconds_to_start: r[10],
            sprint_seconds_to_full: r[11],
            sprint_full_speed_multiplier: r[12],
            sprint_min_velocity: r[13],
            jump_velocity: b.jump_velocity,
            eye_standing: b.eye[0],
            eye_running: b.eye[1],
            eye_crouching: b.eye[2],
            eye_crouch_walking: b.eye[3],
            crouch_transition: b.crouch_transition,
            runtime_crouch_velocity: b.runtime_crouch_velocity,
            runtime_camera_height_velocity: b.runtime_camera_height_velocity,
            camera_height_function_authored: !b.camera_height_function.is_empty(),
            camera_height_fn_samples: {
                // #crouch-rate  The SHAPE of the authored function, sampled at five points. A
                // mapping function that is the identity means the eye moves linearly; anything
                // else means the engine's travel is eased and the duration is not the whole story.
                let mut out = [f32::NAN; 5];
                if let Some(f) = crate::scene::MapFn::decode(&b.camera_height_function) {
                    for (i, o) in out.iter_mut().enumerate() {
                        *o = f.eval(i as f32 * 0.25);
                    }
                }
                out
            },
            camera_height_rows: b.camera_height_rows.len(),
            camera_height_span: b
                .camera_height_rows
                .iter()
                .map(|r| r.1 - r.2)
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v))),

            height_standing: b.height_standing,
            height_crouching: b.height_crouching,
            radius: b.radius,
            mass: b.mass,
            max_slope: b.max_slope,
            gravity_scale: b.gravity_scale,
            airborne_acceleration_scale: b.airborne_acceleration_scale,
            turning_speed: b.turning_speed,
            stationary_turn_threshold: b.stationary_turn_threshold,
            species_row,
        };
        p.validate()?;
        Ok(p)
    }

    /// Are these values *shaped* like a player's? A read at the wrong offset does not fail, it
    /// returns garbage — so the numbers that would make test mode nonsense are rejected here and
    /// the message says which one, rather than the user walking at 4000 wu/s.
    pub fn validate(&self) -> Result<(), String> {
        let check = |name: &str, v: f32, lo: f32, hi: f32| -> Result<(), String> {
            if !v.is_finite() || v < lo || v > hi {
                Err(format!("{name} = {v} is outside [{lo}, {hi}]: the tag read did not land on a player movement field"))
            } else {
                Ok(())
            }
        };
        check("walking speed", self.walk_speed, 0.1, 20.0)?;
        check("run forward", self.run_forward, 0.1, 40.0)?;
        check("run acceleration", self.run_acceleration, 0.1, 1000.0)?;
        check("jump velocity", self.jump_velocity, 0.1, 40.0)?;
        check("standing eye height", self.eye_standing, 0.05, 5.0)?;
        check("crouching eye height", self.eye_crouching, 0.05, 5.0)?;
        check("capsule standing height", self.height_standing, 0.1, 6.0)?;
        check("capsule crouching height", self.height_crouching, 0.1, 6.0)?;
        check("capsule radius", self.radius, 0.02, 2.0)?;
        check("maximum slope", self.max_slope, 0.05, std::f32::consts::FRAC_PI_2)?;
        check("gravity scale", self.gravity_scale, 0.0, 10.0)?;
        Ok(())
    }

    /// Gravity this biped falls at, wu/s² — the engine's measured constant times the tag's own
    /// per-biped scale.
    ///
    /// # The double-count this avoids
    ///
    /// [`GRAVITY`] was fitted from **falling bodies in recordings**, which are bipeds whose own
    /// `character_physics_ground_struct` gravity scale was in force at the time. So the measured
    /// figure already has one scale baked in, and multiplying it by the tag's scale again would
    /// count that scale twice — while using it unscaled would assume every biped falls alike.
    ///
    /// Both readings agree **exactly when the scale is 1.0**, which is why the scale is read,
    /// printed by [`describe`](Self::describe) and restated by [`gravity_note`](Self::gravity_note)
    /// rather than quietly applied. Where it is 1.0 the question is moot; where it is not, the
    /// note says so out loud and the number used is `GRAVITY × scale` with the measurement's own
    /// (equally scaled) body as its reference.
    pub fn gravity(&self) -> f32 {
        GRAVITY * self.gravity_scale_or_unity()
    }

    /// The tag's gravity scale, or 1.0 when the field is absent / zero (a zero would mean a
    /// biped that does not fall at all, which no player biped is).
    pub fn gravity_scale_or_unity(&self) -> f32 {
        if self.gravity_scale > 0.0 {
            self.gravity_scale
        } else {
            1.0
        }
    }

    /// Which gravity case this biped is in, in one line, for the caption and the live gate.
    pub fn gravity_note(&self) -> String {
        let s = self.gravity_scale_or_unity();
        if (s - 1.0).abs() < 1e-4 {
            format!(
                "gravity scale {:.4} = UNSCALED, so this biped falls at the measured {GRAVITY:.4} wu/s^2 \
                 and nothing is scaled twice. A reading of exactly 0.0 meaning 'default' is the engine's \
                 own convention, not a guess about the offset: Bungie's comment on a sibling field reads \
                 \"0 means 0. How hard is it to type 1?\" -- which only needed writing because everywhere \
                 else a 0 IS read as the default",
                self.gravity_scale
            )
        } else {
            format!(
                "gravity scale {:.4} is NOT 1.0: this biped falls at {:.4} wu/s^2 = measured {GRAVITY:.4} x {:.4}. \
                 The measurement was fitted from falling bodies that carried their own scale, so this is \
                 stated rather than silently applied",
                self.gravity_scale,
                self.gravity(),
                s
            )
        }
    }

    /// Air control, wu/s² — the globals' airborne acceleration times the biped's own scale.
    pub fn air_acceleration(&self) -> f32 {
        self.airborne_acceleration * if self.airborne_acceleration_scale > 0.0 { self.airborne_acceleration_scale } else { 1.0 }
    }

    /// Top speed for a stick direction in the body's own frame, wu/s: `fwd` +1 is straight
    /// ahead, `right` +1 is a strafe. Reach authors three different caps (forward, backward,
    /// sideways), so a backpedal is slower than a charge, and crouching swaps the whole set for
    /// the `sneak` one.
    ///
    /// The directional caps are composed as an ELLIPSE, which is what makes a diagonal neither
    /// faster than a straight run (the bug a plain max() gives) nor slower than a strafe.
    pub fn top_speed(&self, fwd: f32, right: f32, crouched: bool, walking: bool) -> f32 {
        self.top_speed_with(fwd, right, crouched, walking, false)
    }

    /// As [`top_speed`](Self::top_speed), with the authored SPRINT multiplier applied.
    ///
    /// Sprint is the globals' own `full speed multiplier` on the forward run and nothing else:
    /// Reach cannot strafe or shoot while sprinting, so it only scales the straight-ahead cap.
    pub fn top_speed_with(&self, fwd: f32, right: f32, crouched: bool, walking: bool, sprinting: bool) -> f32 {
        let base = self.top_speed_base(fwd, right, crouched, walking);
        if sprinting && !crouched && !walking && fwd > 0.0 && self.sprint_full_speed_multiplier > 1.0 {
            base * self.sprint_full_speed_multiplier
        } else {
            base
        }
    }

    fn top_speed_base(&self, fwd: f32, right: f32, crouched: bool, walking: bool) -> f32 {
        let (f_pos, f_neg, side) = if crouched {
            (self.sneak_forward, self.sneak_backward, self.sneak_sideways)
        } else if walking {
            // One authored walk speed, no direction split.
            (self.walk_speed, self.walk_speed, self.walk_speed)
        } else {
            (self.run_forward, self.run_backward, self.run_sideways)
        };
        let len = (fwd * fwd + right * right).sqrt();
        if len < 1e-6 {
            return 0.0;
        }
        let (uf, ur) = (fwd / len, right / len);
        let cap_f = if uf >= 0.0 { f_pos } else { f_neg };
        let (a, b) = (cap_f.max(1e-4), side.max(1e-4));
        // 1 / sqrt((uf/a)^2 + (ur/b)^2) — the radius of the ellipse along the stick direction.
        1.0 / ((uf / a).powi(2) + (ur / b).powi(2)).sqrt()
    }

    /// #play-rate  **The AUTHORED natural speed of one locomotion state, wu/s** — the speed at
    /// which that state's clip plays at its own authored 30 fps.
    ///
    /// This is the number a locomotion cycle's RATE has to be built from, and getting it from
    /// somewhere else is the whole of "it's very slidy, especially crouching and strafing".
    ///
    /// A Reach locomotion clip is authored IN PLACE (the root node translates by exactly
    /// (0,0,0) over all eleven of them — `film_anim::probes::probe_clip_root_motion`), so the
    /// stride it covers is not in the clip's geometry: it is `speed × frames / 30`, the ground
    /// the body covers while the clip runs once at its authored rate. Driving the phase from
    /// `distance / stride` then makes the feet match the ground at any speed, and plays the clip
    /// at exactly 30 fps at this one.
    ///
    /// **Why the authored row and not the film corpus.** `film_anim::gait_speed_and_frames`
    /// carries speeds MEASURED from recordings, and Theater is right to use them — a recorded
    /// body is going whatever speed it was going. But they are medians of a gait BUCKET, and the
    /// buckets are wide: `cuts`' own doc records the correction that the corpus histogram's
    /// dominant mode is **sprinting players**, not running ones. The run-forward median (3.52)
    /// is within 1.5% of this species' authored SPRINT top speed (2.200 × 1.623 = 3.571), and
    /// the gait had no stance at all — a crouching body got the standing clip's speed.
    ///
    /// **The caps are the controller's own.** Every answer below is `top_speed_with` for the
    /// stick direction that state means, so the speed the rate assumes and the speed the body is
    /// actually held to cannot drift apart: one writer, consulted twice.
    ///
    /// `None` for a state with no ground travel to drive it — idle, a turn on the spot, an
    /// airborne arc, a seat — which the caller runs on the clock instead, and for a cache that
    /// authors 0 for that direction (a real possibility the caller must not divide by).
    pub fn gait_speed(&self, stance: crate::film_anim::Stance, gait: crate::film_anim::Gait) -> Option<f32> {
        use crate::film_anim::{Gait, Stance};
        // A sprint is one clip in one direction, at the authored multiplier.
        if stance == Stance::Sprint {
            let v = self.top_speed_with(1.0, 0.0, false, false, true);
            return (v > 1e-3).then_some(v);
        }
        if matches!(stance, Stance::Flight | Stance::Seated | Stance::Dead) {
            return None;
        }
        let crouched = stance == Stance::Crouch;
        // The `walk` set is the authored `walking speed`, which has no direction split — and
        // while CROUCHED there is no walk set at all, only `sneak`, so a slow crouch-walk is
        // still the sneak speed. (That is the tag's shape, not a simplification: see
        // `top_speed_base`.)
        let v = match gait {
            Gait::Idle | Gait::TurnLeft | Gait::TurnRight => return None,
            Gait::RunFront => self.top_speed(1.0, 0.0, crouched, false),
            Gait::WalkFront => self.top_speed(1.0, 0.0, crouched, !crouched),
            Gait::Back => self.top_speed(-1.0, 0.0, crouched, false),
            Gait::BackSlow => self.top_speed(-1.0, 0.0, crouched, !crouched),
            Gait::Left => self.top_speed(0.0, -1.0, crouched, false),
            Gait::Right => self.top_speed(0.0, 1.0, crouched, false),
        };
        (v > 1e-3).then_some(v)
    }

    /// #crouch-rate  **How long a change of clip into or out of the crouch cross-fades for**,
    /// seconds -- the biped's own authored `crouch transition time` (`bipd +0x534`) when the
    /// stance actually crosses the crouch boundary, and the generic
    /// [`crate::reach_actor::BLEND_SECONDS`] otherwise.
    ///
    /// A free function of the two stances rather than an expression inside `play_mode::pose`,
    /// so a gate can assert the number without driving a biped, a cache or a GPU.
    ///
    /// Confirmed to be a TIME IN SECONDS, not a rate and not something else wearing the name,
    /// by the two values the engine derives from it at tag post-process and leaves in the cache:
    /// `runtime crouch transition velocity` (+0x580) = `1 / time` and
    /// `runtime_camera_height_velocity` (+0x584) = `(standing eye - crouching eye) / time`. On
    /// `forge_halo` those read 3.3333 and 0.6000 for the Spartan (time 0.30, eyes 0.600/0.420)
    /// and 5.0000 and 0.9000 for the Elite (0.20, 0.690/0.510) -- four numbers, four exact
    /// matches. Gated in `play_mode::probes::probe_the_authored_crouch_rate`.
    pub fn crouch_fade_seconds(&self, from: crate::film_anim::Stance, to: crate::film_anim::Stance) -> f32 {
        use crate::film_anim::Stance;
        let crossing = (to == Stance::Crouch) != (from == Stance::Crouch);
        if crossing && self.crouch_transition > 1e-3 {
            self.crouch_transition
        } else {
            crate::reach_actor::BLEND_SECONDS
        }
    }

    /// Ground acceleration, wu/s².
    pub fn acceleration(&self, crouched: bool) -> f32 {
        if crouched { self.sneak_acceleration } else { self.run_acceleration }
    }

    /// Eye height for a stance, wu above the feet.
    pub fn eye_height(&self, crouched: bool, moving: bool) -> f32 {
        match (crouched, moving) {
            (true, true) => self.eye_crouch_walking,
            (true, false) => self.eye_crouching,
            (false, true) => self.eye_running,
            (false, false) => self.eye_standing,
        }
    }

    /// Capsule height for a stance, wu.
    pub fn capsule_height(&self, crouched: bool) -> f32 {
        if crouched { self.height_crouching } else { self.height_standing }
    }

    /// The values, for the caption and the gates. Every line names its tag field.
    pub fn describe(&self) -> String {
        format!(
            "movement constants, all READ FROM THE TAGS (no invented values)\n\
             matg @player information[{}]  walk {:.3} | run fwd/back/side {:.3}/{:.3}/{:.3} | accel {:.2} \
             | sneak fwd/back/side {:.3}/{:.3}/{:.3} | sneak accel {:.2} | airborne accel {:.2}  (wu/s, wu/s^2)\n\
             sprint x{:.3} -> forward top speed {:.3} wu/s (authored `full speed multiplier`; starts after {:.2} s pegged, full after {:.2} s)\n\
             bipd  jump velocity {:.3} wu/s | eye stand/run/crouch/crouch-walk {:.3}/{:.3}/{:.3}/{:.3} wu \
             | crouch transition {:.2} s\n\
             bipd  #crouch-rate runtime crouch velocity +0x580 = {:.4} | runtime camera height velocity +0x584 = {:.4} \
             | camera height velocity function authored = {} f(0,.25,.5,.75,1) = {:?} \
             | per-weapon-class camera height rows = {} span {:.3}..{:.3} wu\n\
             bipd  physics: capsule {:.3} wu tall ({:.3} crouched), radius {:.3} wu, mass {:.1} \
             | max slope {:.1} deg | gravity scale {:.2} | airborne accel scale {:.2} | turn {:.0} deg/s (stationary threshold {:.0} deg)\n\
             {}\n\
             jump apex from the jump velocity and that gravity alone: {:.3} wu",
            self.species_row,
            self.walk_speed, self.run_forward, self.run_backward, self.run_sideways, self.run_acceleration,
            self.sneak_forward, self.sneak_backward, self.sneak_sideways, self.sneak_acceleration,
            self.airborne_acceleration,
            self.sprint_full_speed_multiplier,
            self.run_forward * self.sprint_full_speed_multiplier.max(1.0),
            self.sprint_seconds_to_start,
            self.sprint_seconds_to_full,
            self.jump_velocity, self.eye_standing, self.eye_running, self.eye_crouching, self.eye_crouch_walking,
            self.crouch_transition,
            self.runtime_crouch_velocity, self.runtime_camera_height_velocity,
            self.camera_height_function_authored, self.camera_height_fn_samples, self.camera_height_rows, self.camera_height_span.0, self.camera_height_span.1,
            self.height_standing, self.height_crouching, self.radius, self.mass,
            self.max_slope.to_degrees(), self.gravity_scale, self.airborne_acceleration_scale,
            self.turning_speed.to_degrees(),
            self.stationary_turn_threshold.to_degrees(),
            self.gravity_note(),
            self.jump_velocity * self.jump_velocity / (2.0 * self.gravity().max(1e-4)),
        )
    }

    /// Every `@player information` row in the cache, as its ten leading reals — for the dump
    /// verb, so WHICH row is the Elite's is measured rather than assumed.
    pub fn player_info_rows(native: &NativeDll, cache: CacheHandle, matg: u32) -> Vec<[f32; 14]> {
        let raw = crate::reach_actor::tag_block(native, cache, matg, PLAYER_INFO, PLAYER_INFO_ELEM, 3);
        raw.chunks_exact(PLAYER_INFO_ELEM)
            .map(|e| {
                let mut v = [0f32; 14];
                // The ten leading ground-movement reals are contiguous from +0x00...
                for (i, s) in v.iter_mut().take(10).enumerate() {
                    *s = rd_f32(e, i * 4);
                }
                // ...and the sprint section and the terminal velocity are not, so they are read
                // by their own offsets rather than by stride.
                v[10] = rd_f32(e, PI_SPRINT_SECONDS_TO_START);
                v[11] = rd_f32(e, PI_SPRINT_SECONDS_TO_FULL);
                v[12] = rd_f32(e, PI_SPRINT_FULL_SPEED_MULTIPLIER);
                v[13] = rd_f32(e, PI_SPRINT_MIN_VELOCITY);
                v
            })
            .collect()
    }

    /// #play  The `@player representation` rows: per playable species, the biped the player
    /// actually IS plus its first-person models.
    ///
    /// This is how the species is resolved rather than by matching tag-path substrings: the
    /// engine states, per row, which `bipd` that row's player is. The row index is also the
    /// pairing with [`PLAYER_INFO`], so the speeds and the body come from the same row.
    pub fn player_representations(native: &NativeDll, cache: CacheHandle) -> Vec<PlayerRepresentation> {
        let Some(matg) = native.tags_by_group(cache, "matg").into_iter().next() else { return Vec::new() };
        let raw = crate::reach_actor::tag_block(
            native,
            cache,
            matg,
            PLAYER_REPRESENTATION,
            PLAYER_REPRESENTATION_ELEM,
            3,
        );
        raw.chunks_exact(PLAYER_REPRESENTATION_ELEM)
            .enumerate()
            .map(|(row, e)| {
                // The row STATES which `@player information` it uses. Negative / absent means
                // "the same index", which is what the two blocks happen to be on the maps
                // measured — but the stated value wins wherever it is present.
                let idx = i32::from_le_bytes([
                    e.get(REP_PLAYER_INFO_INDEX).copied().unwrap_or(0),
                    e.get(REP_PLAYER_INFO_INDEX + 1).copied().unwrap_or(0),
                    e.get(REP_PLAYER_INFO_INDEX + 2).copied().unwrap_or(0),
                    e.get(REP_PLAYER_INFO_INDEX + 3).copied().unwrap_or(0),
                ]);
                PlayerRepresentation {
                    row,
                    hands: tagref(e, REP_HANDS),
                    body: tagref(e, REP_BODY),
                    unit: tagref(e, REP_THIRD_PERSON_UNIT),
                    info_row: if idx >= 0 { idx as usize } else { row },
                }
            })
            .collect()
    }

    /// Every row, printed — including the `@player representation` cross-check that says the
    /// block offset is right.
    pub fn describe_rows(native: &NativeDll, cache: CacheHandle) -> String {
        let Some(matg) = native.tags_by_group(cache, "matg").into_iter().next() else {
            return "no 'matg' in this cache".into();
        };
        let rows = Self::player_info_rows(native, cache, matg);
        let reps = native
            .tag_meta(cache, matg, PLAYER_REPRESENTATION as u32, 4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .unwrap_or(-1);
        let mut s = format!(
            "matg {matg:#x}: {} '@player information' row(s) at +{PLAYER_INFO:#x}  \
             (cross-check: '@player representation' at +{PLAYER_REPRESENTATION:#x} has {reps} row(s) — \
             the arms block HMS already reads, so the offsets are in step)\n",
            rows.len()
        );
        for (i, r) in rows.iter().enumerate() {
            s += &format!(
                "  [{i}] walk {:.3} | run {:.3}/{:.3}/{:.3} accel {:.2} | sneak {:.3}/{:.3}/{:.3} accel {:.2} | air accel {:.2} \
                 | sprint x{:.3} (start {:.2}s, to full {:.2}s, min {:.2} wu/s)\n",
                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[12], r[10], r[11], r[13]
            );
        }
        // #crouch-rate  The per-weapon-class `camera heights` block of every biped in the map,
        // because the SPAN those rows give is what turns the authored
        // `runtime_camera_height_velocity` (wu/s) into a DURATION.
        for (tag, path) in crate::reach_actor::ActorScene::list_bipeds(native, cache) {
            let Some(b) = Self::biped_fields(native, cache, tag) else { continue };
            if b.camera_height_rows.is_empty() { continue; }
            s += &format!("  bipd {tag:#x} {path}  camera heights (+0x54C), {} row(s):\n", b.camera_height_rows.len());
            for (c, st, cr) in &b.camera_height_rows {
                s += &format!(
                    "    {c:<12} standing {st:.4} crouching {cr:.4} span {:.4} wu -> {:.3} s at {:.4} wu/s\n",
                    st - cr,
                    if b.runtime_camera_height_velocity > 1e-6 { (st - cr) / b.runtime_camera_height_velocity } else { f32::NAN },
                    b.runtime_camera_height_velocity
                );
            }
        }
        s
    }

    fn biped_fields(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<BipedFields> {
        // One read covering the whole `$$$ BIPED $$$` segment through the ground physics struct.
        let end = BIPD_PHYSICS + PHYSICS_GROUND + 0x44;
        let m = native.tag_meta(cache, tag, 0, end as u32)?;
        if m.len() < end {
            return None;
        }
        let g = BIPD_PHYSICS + PHYSICS_GROUND;
        Some(BipedFields {
            turning_speed: rd_f32(&m, BIPD_TURNING_SPEED),
            stationary_turn_threshold: rd_f32(&m, BIPD_STATIONARY_TURN_THRESHOLD),
            jump_velocity: rd_f32(&m, BIPD_JUMP_VELOCITY),
            eye: [
                rd_f32(&m, BIPD_CAMERA_HEIGHTS),
                rd_f32(&m, BIPD_CAMERA_HEIGHTS + 4),
                rd_f32(&m, BIPD_CAMERA_HEIGHTS + 8),
                rd_f32(&m, BIPD_CAMERA_HEIGHTS + 12),
            ],
            crouch_transition: rd_f32(&m, BIPD_CROUCH_TRANSITION),
            runtime_crouch_velocity: rd_f32(&m, BIPD_RUNTIME_CROUCH_VELOCITY),
            runtime_camera_height_velocity: rd_f32(&m, BIPD_RUNTIME_CAMERA_HEIGHT_VELOCITY),
            camera_height_function: {
                // #crouch-rate  The 20-byte `c_function_definition` carries a `tag_data`: its
                // SIZE sits 4 bytes in and its ADDRESS 16 bytes in, the same two offsets
                // `scene::object_function_rest_value` reads a `mapping_function` through
                // (element +0x18 / +0x24 of a function struct based at +0x14).
                let o = BIPD_CAMERA_HEIGHT_FUNCTION;
                let rd = |x: usize| if x + 4 <= m.len() {
                    u32::from_le_bytes([m[x], m[x + 1], m[x + 2], m[x + 3]])
                } else { 0 };
                let (size, addr) = (rd(o + 4), rd(o + 0x10));
                if (0x0C..=0x400).contains(&size) && addr != 0 {
                    native.tag_ptr_bytes(cache, addr, size).unwrap_or_default()
                } else {
                    Vec::new()
                }
            },
            camera_height_rows: {
                // #crouch-rate  The per-weapon-class `camera heights` block (+0x54C), 12-byte
                // `s_biped_camera_height` = {weapon class string id, standing [wu], crouching
                // [wu]}. Read because the SPAN the eye travels decides the DURATION: the engine
                // moves it at a constant wu/s (+0x584), not at a constant fraction per second.
                let o = BIPD_CAMERA_HEIGHTS_BLOCK;
                let (cnt, ptr) = if o + 8 <= m.len() {
                    (
                        i32::from_le_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]]),
                        u32::from_le_bytes([m[o + 4], m[o + 5], m[o + 6], m[o + 7]]),
                    )
                } else { (0, 0) };
                let mut rows = Vec::new();
                if (1..=16).contains(&cnt) && ptr != 0 {
                    if let Some(b) = native.tag_ptr_bytes(cache, ptr, cnt as u32 * 12) {
                        for i in 0..cnt as usize {
                            let e = &b[i * 12..i * 12 + 12];
                            let sid = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                            rows.push((
                                native.string_id(cache, sid).unwrap_or_else(|| format!("sid:{sid:#X}")),
                                rd_f32(e, 4),
                                rd_f32(e, 8),
                            ));
                        }
                    }
                }
                rows
            },
            height_standing: rd_f32(&m, BIPD_PHYSICS + 0x04),
            height_crouching: rd_f32(&m, BIPD_PHYSICS + 0x08),
            radius: rd_f32(&m, BIPD_PHYSICS + 0x0C),
            mass: rd_f32(&m, BIPD_PHYSICS + 0x10),
            max_slope: rd_f32(&m, g),
            gravity_scale: rd_f32(&m, g + 0x3C),
            airborne_acceleration_scale: rd_f32(&m, g + 0x40),
        })
    }
}

struct BipedFields {
    /// #crouch-rate  see [`BIPD_RUNTIME_CROUCH_VELOCITY`] / [`BIPD_RUNTIME_CAMERA_HEIGHT_VELOCITY`].
    runtime_crouch_velocity: f32,
    runtime_camera_height_velocity: f32,
    /// #crouch-rate  the 20 raw bytes of `camera height velocity function` and the row count of
    /// the per-weapon-class `camera heights` block. Print-only, for the measurement.
    camera_height_function: Vec<u8>,
    /// #crouch-rate  `(weapon class, standing [wu], crouching [wu])` per row.
    camera_height_rows: Vec<(String, f32, f32)>,
    turning_speed: f32,
    stationary_turn_threshold: f32,
    jump_velocity: f32,
    eye: [f32; 4],
    crouch_transition: f32,
    height_standing: f32,
    height_crouching: f32,
    radius: f32,
    mass: f32,
    max_slope: f32,
    gravity_scale: f32,
    airborne_acceleration_scale: f32,
}

fn rd_f32(b: &[u8], off: usize) -> f32 {
    if off + 4 > b.len() {
        return 0.0;
    }
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// #play  A stand-in with the shape of Reach's own authored values, for the controller's tests.
///
/// **Not a claim about the real numbers** — the real ones come from the cache and the live gate
/// prints them; this exists so the pure controller can be tested without a map, a window or a GPU.
#[cfg(test)]
pub mod tests_support {
    use super::PlayerPhysics;
    /// A stand-in with the shape of Reach's own authored values, for the controller's tests.
    pub fn stand_in() -> PlayerPhysics {
        PlayerPhysics {
            walk_speed: 2.3,
            run_forward: 4.1,
            run_backward: 3.0,
            run_sideways: 3.5,
            run_acceleration: 20.0,
            sneak_forward: 1.4,
            sneak_backward: 1.2,
            sneak_sideways: 1.2,
            sneak_acceleration: 10.0,
            airborne_acceleration: 8.0,
            sprint_seconds_to_start: 0.2,
            sprint_seconds_to_full: 0.5,
            sprint_full_speed_multiplier: 1.8,
            sprint_min_velocity: 1.0,
            jump_velocity: 3.0,
            eye_standing: 0.62,
            eye_running: 0.60,
            eye_crouching: 0.44,
            eye_crouch_walking: 0.42,
            crouch_transition: 0.2,
            runtime_crouch_velocity: 0.0,
            runtime_camera_height_velocity: 0.0,
            camera_height_function_authored: false,
            camera_height_fn_samples: [f32::NAN; 5],
            camera_height_span: (0.0, 0.0),
            camera_height_rows: 0,
            height_standing: 0.70,
            height_crouching: 0.50,
            radius: 0.21,
            mass: 90.0,
            max_slope: 45f32.to_radians(),
            gravity_scale: 1.0,
            airborne_acceleration_scale: 1.0,
            turning_speed: 2.0,
            stationary_turn_threshold: 0.6,
            species_row: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tests_support::stand_in;

    /// The forge_halo SPARTAN row, exactly as the cache reads it (see
    /// `play_mode::probes::probe_locomotion_rate_against_the_authored_speed`). Separate from
    /// [`stand_in`], whose numbers are deliberately NOT Reach's so that a gate asserting a real
    /// authored value cannot be satisfied by the stand-in.
    fn spartan_row() -> PlayerPhysics {
        PlayerPhysics {
            walk_speed: 0.512,
            run_forward: 2.200,
            run_backward: 1.980,
            run_sideways: 1.980,
            run_acceleration: 8.50,
            sneak_forward: 0.900,
            sneak_backward: 0.810,
            sneak_sideways: 0.810,
            sneak_acceleration: 6.10,
            sprint_full_speed_multiplier: 1.623,
            height_standing: 0.650,
            ..stand_in()
        }
    }

    /// #crouch-rate  **THE CROUCH RATE GATE.** The fade into and out of the crouch must be the
    /// biped's OWN authored `crouch transition time`, and every other change of clip must be the
    /// generic one.
    ///
    /// The expectations are literals taken from the cache dump (`--dump-play-phys` on
    /// `forge_halo`), not from the code under test: the Spartan's `bipd +0x534` reads **0.30 s**
    /// and the Elite's **0.20 s**, while [`crate::reach_actor::BLEND_SECONDS`] is 0.18. The
    /// 0.30 is deliberately NOT 0.18 and NOT the Elite's 0.20, so no single constant satisfies
    /// the whole table.
    ///
    /// Sabotage (one assertion each): returning `BLEND_SECONDS` unconditionally fails line 1;
    /// returning `crouch_transition` unconditionally fails line 3; dropping the `!=` that makes
    /// the test a CROSSING (so that merely BEING crouched counts) fails line 4.
    #[test]
    fn the_crouch_fade_is_the_authored_transition_time() {
        use crate::film_anim::Stance;
        let sp = PlayerPhysics { crouch_transition: 0.30, ..spartan_row() };
        let el = PlayerPhysics { crouch_transition: 0.20, ..spartan_row() };
        // 1. standing -> crouching is the authored time.
        assert_eq!(sp.crouch_fade_seconds(Stance::Combat, Stance::Crouch), 0.30);
        // 2. ...and so is coming back up, and the two species differ.
        assert_eq!(sp.crouch_fade_seconds(Stance::Crouch, Stance::Combat), 0.30);
        assert_eq!(el.crouch_fade_seconds(Stance::Combat, Stance::Crouch), 0.20);
        // 3. a change that is not a crouch crossing is the generic fade.
        assert_eq!(
            sp.crouch_fade_seconds(Stance::Combat, Stance::Sprint),
            crate::reach_actor::BLEND_SECONDS
        );
        // 4. STAYING crouched is not a crossing -- every frame of a crouch-walk would otherwise
        //    re-arm a 0.30 s fade.
        assert_eq!(
            sp.crouch_fade_seconds(Stance::Crouch, Stance::Crouch),
            crate::reach_actor::BLEND_SECONDS
        );
        // 5. a biped that authors no transition time falls back rather than snapping.
        let none = PlayerPhysics { crouch_transition: 0.0, ..spartan_row() };
        assert_eq!(
            none.crouch_fade_seconds(Stance::Combat, Stance::Crouch),
            crate::reach_actor::BLEND_SECONDS
        );
    }

    /// #play-gate  **THE SLIDING GATE.** The locomotion cycle's rate must come from the AUTHORED
    /// speed of the state the body is in — species, stance AND direction — because the stride a
    /// clip covers is `speed x frames / 30` and nothing else makes the feet match the ground.
    ///
    /// Every expectation here is a LITERAL worked out from the authored row, not a re-run of the
    /// code under test:
    ///
    /// * walk forward: `0.512 x 24 / 30 = 0.4096` wu per cycle, so `cycles_per_wu = 1 / 0.4096
    ///   = 2.44140625`;
    /// * crouch forward: `0.900 x 26 / 30 = 0.780` wu per cycle;
    /// * sprint: `2.200 x 1.623 = 3.5706` wu/s.
    ///
    /// And the three things that must NOT hold, each of which was true before this round:
    ///
    /// 1. the crouch answer must not be the standing one (`gait_speed_and_frames` takes no
    ///    stance at all, so a crouching body was given the standing clip's speed);
    /// 2. the strafe answer must not be the forward one (Reach authors `run sideways`
    ///    separately);
    /// 3. the rate must not be the film-corpus one, whose run-forward median (3.52 wu/s) is this
    ///    species' authored SPRINT top speed to within 1.4%.
    #[test]
    fn the_locomotion_rate_follows_the_authored_speed_for_the_stance_and_direction() {
        use crate::film_anim::{cycles_per_wu_at_speed, cycles_per_wu_for_clip, Gait, Stance, CLIP_FPS};
        let p = spartan_row();
        // --- the authored speed per state, as literals off the row -------------------------
        //
        // Compared with a tolerance rather than for equality: a SIDEWAYS cap comes back through
        // `top_speed_base`'s directional ellipse, `1 / sqrt((uf/a)^2 + (ur/b)^2)`, which for a
        // pure strafe is `b` only up to f32 rounding (1.9800001 against 1.98). That is the
        // fixture's own arithmetic, not a defect -- the tolerance is 1e-5, four orders of
        // magnitude tighter than the 0.57 wu/s gap between any two authored speeds in the row,
        // so it cannot let a wrong field through.
        let got = |st: Stance, g: Gait| p.gait_speed(st, g);
        let is = |st: Stance, g: Gait, want: f32| {
            let v = got(st, g).unwrap_or_else(|| panic!("{st:?}/{g:?} has no authored speed"));
            assert!((v - want).abs() < 1e-5, "{st:?}/{g:?} = {v}, want the authored {want}");
        };
        is(Stance::Combat, Gait::WalkFront, 0.512);
        is(Stance::Combat, Gait::RunFront, 2.200);
        is(Stance::Combat, Gait::BackSlow, 0.512);
        is(Stance::Combat, Gait::Back, 1.980);
        is(Stance::Combat, Gait::Left, 1.980);
        is(Stance::Combat, Gait::Right, 1.980);
        is(Stance::Crouch, Gait::RunFront, 0.900);
        is(Stance::Crouch, Gait::Back, 0.810);
        is(Stance::Crouch, Gait::Left, 0.810);
        is(Stance::Crouch, Gait::Right, 0.810);
        // 2.200 x 1.623; compared with a tolerance because it is a product of two f32s.
        let sprint = got(Stance::Sprint, Gait::RunFront).expect("a sprint has a speed");
        assert!((sprint - 3.5706).abs() < 1e-3, "sprint speed {sprint}, want 2.200 x 1.623 = 3.5706");
        // --- and NOTHING for a state with no ground travel to drive it ---------------------
        for g in [Gait::Idle, Gait::TurnLeft, Gait::TurnRight] {
            assert_eq!(got(Stance::Combat, g), None, "{g:?} has no stride to drive a cycle");
        }
        for st in [Stance::Flight, Stance::Seated, Stance::Dead] {
            assert_eq!(got(st, Gait::RunFront), None, "{st:?} is not on foot");
        }
        // --- WHAT MUST NOT HOLD (1): the crouch answer is not the standing one --------------
        assert_ne!(
            got(Stance::Crouch, Gait::RunFront),
            got(Stance::Combat, Gait::RunFront),
            "crouching and standing were given the same speed -- the stance is being ignored again"
        );
        // (2): a strafe is not a forward run.
        let (side, fwd) = (got(Stance::Combat, Gait::Left).unwrap(), got(Stance::Combat, Gait::RunFront).unwrap());
        assert!(
            (side - fwd).abs() > 0.1,
            "a strafe ({side}) and a forward run ({fwd}) were given the same speed -- the direction \
             is being ignored"
        );
        // --- the rate itself, against hand-worked literals ---------------------------------
        let cpw = cycles_per_wu_at_speed(0.512, 24.0).expect("a walk has a rate");
        assert!((cpw - 2.441_406_25).abs() < 1e-5, "cycles/wu {cpw}, want 30/(0.512 x 24) = 2.44140625");
        assert!((1.0 / cpw - 0.4096).abs() < 1e-5, "stride {} wu, want 0.512 x 24 / 30 = 0.4096", 1.0 / cpw);
        let crouch = cycles_per_wu_at_speed(0.900, 26.0).expect("a crouch walk has a rate");
        assert!((1.0 / crouch - 0.780).abs() < 1e-4, "crouch stride {} wu, want 0.900 x 26 / 30 = 0.780", 1.0 / crouch);
        // Degenerate inputs define no stride and must say so rather than divide.
        assert!(cycles_per_wu_at_speed(0.0, 24.0).is_none());
        assert!(cycles_per_wu_at_speed(0.512, 0.0).is_none());
        // --- (3): the film-corpus law is a DIFFERENT law, by the measured factor -----------
        //
        // `cycles_per_wu_for_clip` keeps the corpus medians for Theater, where they are right.
        // On the same 22-frame run clip it implies a stride of 3.52 x 22 / 30 = 2.581 wu against
        // this row's 2.200 x 22 / 30 = 1.613 -- a ratio of 1.600, which is the factor the feet
        // were sliding by.
        let film_run = cycles_per_wu_for_clip(Stance::Combat, Gait::RunFront, 22.0).unwrap();
        let auth_run = cycles_per_wu_at_speed(2.200, 22.0).unwrap();
        let ratio = auth_run / film_run;
        assert!((ratio - 1.600).abs() < 0.01, "the two laws differ by {ratio:.3}x, measured 1.600x");
        // --- the INDEPENDENT discriminator: which law is anatomically possible -------------
        //
        // A stride is a physical length and the body it belongs to is 0.650 wu tall. A running
        // human covers roughly 1.7 body heights per cycle (two steps); nothing bipedal covers
        // four. The authored law puts the run cycle at 1.613 / 0.650 = 2.48 body heights, which
        // is a long but possible Spartan stride; the corpus law puts it at 2.581 / 0.650 = 3.97,
        // a 7.9 metre bound per cycle, which is a long jump and not a run. This is the one
        // assertion here that does not come from either formula -- it comes from the body.
        let h = p.height_standing;
        let auth_heights = 1.0 / auth_run / h;
        let film_heights = 1.0 / film_run / h;
        assert!(
            auth_heights < 3.0,
            "the authored law implies a {auth_heights:.2} body-height run stride, which is not a run"
        );
        assert!(
            film_heights > 3.0,
            "the corpus law's {film_heights:.2} body-height stride is no longer the implausible one, \
             so this gate no longer discriminates between the two laws"
        );
        // --- and the whole point, stated as the thing it is for ----------------------------
        //
        // A body travelling at the authored speed for its state plays that state's clip at the
        // authored 30 fps. Checked against the LITERAL 30.0, for every state, at the real clip
        // lengths measured off forge_halo.
        for (st, g, n, label) in [
            (Stance::Combat, Gait::WalkFront, 24.0, "walk forward"),
            (Stance::Combat, Gait::RunFront, 22.0, "run forward"),
            (Stance::Combat, Gait::Left, 19.0, "strafe left"),
            (Stance::Crouch, Gait::RunFront, 26.0, "crouch forward"),
            (Stance::Crouch, Gait::Left, 24.0, "crouch left"),
            (Stance::Sprint, Gait::RunFront, 17.0, "sprint"),
        ] {
            let v = p.gait_speed(st, g).unwrap();
            let fps = v * cycles_per_wu_at_speed(v, n).unwrap() * n;
            assert!(
                (fps - CLIP_FPS).abs() < 1e-2,
                "{label}: at the authored {v:.3} wu/s the {n:.0}-frame clip plays at {fps:.1} fps, not 30"
            );
        }
    }

    /// #play-gate  THE DERIVED CUTS LAND BETWEEN THE AUTHORED SPEEDS THEY SEPARATE.
    ///
    /// The bug this exists to prevent is the one Theater has: a walk/run split ABOVE the run
    /// itself (2.4 against an authored run of 2.2, so a flat-out run is drawn walking) and a
    /// sprint cut above the sprint top speed (4.6 against 3.57, so a sprint can never fire). The
    /// numbers below are the ones measured off `forge_halo`.
    #[test]
    fn the_derived_gait_cuts_lie_between_the_authored_speeds() {
        let s = PlayerSpeeds {
            row: 0,
            biped: 0x3083,
            elite: false,
            walk: 0.512,
            run_forward: 2.200,
            run_backward: 1.980,
            run_sideways: 1.980,
            run_acceleration: 8.5,
            sneak_forward: 0.900,
            sneak_backward: 0.810,
            sneak_sideways: 0.810,
            sneak_acceleration: 6.1,
            airborne_acceleration: 1.05,
            sprint_multiplier: 1.623,
            sprint_seconds_to_start: 0.0,
            sprint_seconds_to_full: 1.0,
            sprint_min_velocity: 0.0,
        };
        assert!(s.looks_authored(), "the measured Forge World row must pass its own sanity check");
        let top = s.sprint_top_speed();
        assert!((top - 3.5706).abs() < 1e-3, "sprint top speed {top}");
        let split = s.walk_run_split();
        assert!(split > s.walk && split < s.run_forward, "walk/run split {split} is not between {} and {}", s.walk, s.run_forward);
        let cut = s.sprint_cut();
        assert!(cut > s.run_forward && cut < top, "sprint cut {cut} is not between {} and {top}", s.run_forward);
        // And the point of all of it: Theater's gait cuts must lie INSIDE the authored range.
        //
        // **This assertion used to be the opposite**, and that is the whole reason it is here.
        // When it was written, `cuts::WALK` was 2.4 — ABOVE the Spartan's authored flat-out run
        // of 2.200, so every genuine full-speed run was drawn walking — and `cuts::SPRINT` was
        // 4.6, ABOVE the authored sprint top speed of 3.571, so a sprint could never be drawn at
        // all. The gate asserted that those two facts held, precisely so that it would FAIL the
        // day they stopped holding rather than quietly agree with whatever the constants became.
        // They have since been corrected against these authored numbers, so it now asserts the
        // corrected relationship. If it fails again, the cuts have drifted back outside the
        // speeds the tags actually author.
        assert!(
            crate::film_anim::cuts::WALK < s.run_forward,
            "the walk/run split ({}) is at or above the authored run forward ({}), so a flat-out run is drawn walking",
            crate::film_anim::cuts::WALK,
            s.run_forward
        );
        assert!(
            crate::film_anim::cuts::SPRINT < top,
            "the sprint cut ({}) is at or above the authored sprint top speed ({top}), so a sprint can never be drawn",
            crate::film_anim::cuts::SPRINT
        );
        assert!(
            crate::film_anim::cuts::WALK > s.walk,
            "the walk/run split ({}) is below the authored walking speed ({})",
            crate::film_anim::cuts::WALK,
            s.walk
        );
    }

    /// #play-gate  A row that is not shaped like a player's never reaches a caller — which is
    /// what keeps a wrong offset from being reported as a speed.
    #[test]
    fn a_row_that_is_not_shaped_like_a_player_is_refused() {
        let good = PlayerSpeeds {
            row: 0, biped: 1, elite: false,
            walk: 0.5, run_forward: 2.2, run_backward: 2.0, run_sideways: 2.0, run_acceleration: 8.5,
            sneak_forward: 0.9, sneak_backward: 0.8, sneak_sideways: 0.8, sneak_acceleration: 6.1,
            airborne_acceleration: 1.05, sprint_multiplier: 1.6,
            sprint_seconds_to_start: 0.0, sprint_seconds_to_full: 1.0, sprint_min_velocity: 0.0,
        };
        assert!(good.looks_authored());
        // A float decoded from unrelated bytes.
        let mut bad = good;
        bad.run_forward = 4.6e18;
        assert!(!bad.looks_authored());
        // A "run" slower than the walk is not a player row either.
        let mut slow = good;
        slow.run_forward = 0.3;
        assert!(!slow.looks_authored());
    }

    #[test]
    fn the_offsets_are_the_reflection_datas_own() {
        // The two blocks are adjacent in `matg`, and the SECOND one is independently verified
        // (scene.rs reads the first-person arms from it). A 12-byte cache block header is what
        // separates them, so this is the whole argument that PLAYER_INFO is in the right place.
        assert_eq!(PLAYER_INFO + 12, PLAYER_REPRESENTATION);
        // The ground physics struct sits 0x44 into the 184-byte character physics struct.
        assert!(PHYSICS_GROUND + 0x44 <= 184);
    }

    /// #gravity  The player falls at the ENGINE'S OWN AUTHORED constant, and at none of the
    /// wrong answers this codebase has held or could be "tidied" onto.
    ///
    /// This gate used to assert the opposite — that [`GRAVITY`] must NOT be
    /// `reach_physics::WORLD_GRAVITY` — on the strength of a film fit that was later shown to be
    /// an artefact of a clamp in its own regression (see [`GRAVITY`]'s note and
    /// `reach_physics`). It is inverted deliberately, and the thing it now guards is the one that
    /// matters: the player's constant and the RAGDOLLS' are separate, so changing one cannot move
    /// the other.
    ///
    /// The assertions name the constants directly rather than going through
    /// `physics_settle::GRAVITY_WU_S2`, which is what this test used to compare against. That
    /// comparison was true only because of what that alias currently points at — repoint it and
    /// the assertion silently stops meaning what its name promises.
    #[test]
    fn gravity_is_the_engines_authored_constant_and_the_ragdolls_did_not_move_with_it() {
        use crate::reach_physics as rp;
        // What it IS.
        assert_eq!(GRAVITY, rp::biped_fall_acceleration());
        assert_eq!(GRAVITY, rp::WORLD_GRAVITY);
        assert!(!GRAVITY_IS_A_FILM_FIT, "the player's gravity is read from the engine, not fitted");

        // What it is NOT, part one: Earth, in either of the two forms this codebase held it --
        // the correct conversion and the digit-transposed typo. This is the guard the test's
        // name has always promised.
        assert!((GRAVITY - rp::EARTH_GRAVITY_WU).abs() > 0.9, "test mode is falling at Earth gravity");
        assert!((GRAVITY - 3.2147f32).abs() > 0.9, "test mode is falling at the transposed Earth typo");

        // What it is NOT, part two: the OLD FILM FIT -- and the ragdolls, which still use it.
        // That separation is the whole point of the split, so it is asserted here as well as in
        // `reach_physics`: this is the file someone edits when they want to change the jump.
        assert!(
            (GRAVITY - rp::MEASURED_FALL_ACCELERATION).abs() > 1.0,
            "the player has been pointed back at the 10.4576 film fit, whose regression clamped \
             its samples at g < 20 and reported the median of the clipped tail. See GRAVITY's note."
        );
        assert_ne!(
            GRAVITY,
            crate::film_play::CORPSE_GRAVITY,
            "the player's gravity and the ragdolls' have become the same constant again -- a change \
             to the jump can now move every corpse in Theater"
        );
        // And the apex this produces, as a RELATIONSHIP rather than a number: the closed form,
        // and the square law. Both hold for any constant, which is the point -- they are what
        // lets the constant move without re-tuning anything.
        let v = 2.300f32; // the Spartan's authored bipd +0x4F0
        let apex = rp::jump_apex(v, GRAVITY);
        assert!((apex - v * v / (2.0 * GRAVITY)).abs() < 1e-6);
        assert!((rp::jump_apex(2.0 * v, GRAVITY) - 4.0 * apex).abs() < 1e-4);
    }

    /// The directional caps compose as an ellipse: a diagonal is between the two straight caps,
    /// never past the faster one. A plain `max()` would let a diagonal run outrun a straight run.
    #[test]
    fn a_diagonal_is_between_the_two_straight_caps() {
        let p = stand_in();
        let f = p.top_speed(1.0, 0.0, false, false);
        let s = p.top_speed(0.0, 1.0, false, false);
        let d = p.top_speed(1.0, 1.0, false, false);
        assert!((f - p.run_forward).abs() < 1e-4, "forward cap is the authored one: {f}");
        assert!((s - p.run_sideways).abs() < 1e-4, "sideways cap is the authored one: {s}");
        assert!(d < f.max(s) + 1e-4 && d > f.min(s) - 1e-4, "diagonal {d} not between {f} and {s}");
        // Backward is its own, slower cap.
        assert!((p.top_speed(-1.0, 0.0, false, false) - p.run_backward).abs() < 1e-4);
        // Crouching swaps the whole set.
        assert!((p.top_speed(1.0, 0.0, true, false) - p.sneak_forward).abs() < 1e-4);
    }

    #[test]
    fn validate_rejects_a_read_at_the_wrong_offset() {
        let mut p = stand_in();
        assert!(p.validate().is_ok());
        // What a garbage read looks like: a float decoded from unrelated bytes.
        p.run_forward = 4.6e18;
        assert!(p.validate().is_err(), "a nonsense run speed must be refused, not walked at");
        let mut q = stand_in();
        q.radius = 0.0;
        assert!(q.validate().is_err(), "a zero capsule radius must be refused");
    }

    /// #gravity  The jump apex is a derived number, not an authored one, and this gate asserts
    /// the RELATIONSHIPS it must obey — not its value.
    ///
    /// It used to assert a BAND (`0.3 < apex < 0.6` wu) worked out for the old 10.4576 fall
    /// constant, so it went red the moment the constant moved even though nothing about the
    /// arithmetic was wrong. **That is a gate whose yardstick manufactures the fault**: with the
    /// stand-in's 3.0 wu/s jump the apex under the engine's authored constant is
    /// `9 / (2 x 4.1712594)` = 1.0788 wu, which is exactly what it reported — a number derivable
    /// from the fixture, which means a fixture error and not a defect.
    ///
    /// So the assertions are now the three laws that hold for ANY constant, which is what lets
    /// the constant move without re-tuning a thing. The constant's identity is pinned once, in
    /// `gravity_is_the_engines_authored_constant_and_the_ragdolls_did_not_move_with_it`, and the
    /// real authored values are checked against a real cache in
    /// `play_mode::map_gates::the_jump_impulse_is_read_from_each_biped_and_reaches_the_body`.
    #[test]
    fn the_apex_obeys_the_closed_form_whatever_the_gravity_constant_is() {
        let p = stand_in();
        let g = p.gravity();
        let apex = crate::reach_physics::jump_apex(p.jump_velocity, g);
        // 1. the closed form, written out independently of the function under test
        assert!((apex - p.jump_velocity * p.jump_velocity / (2.0 * g)).abs() < 1e-6, "apex {apex} wu");
        // 2. it is quadratic in the velocity
        assert!(
            (crate::reach_physics::jump_apex(2.0 * p.jump_velocity, g) - 4.0 * apex).abs() < 1e-4,
            "doubling the jump velocity must quadruple the apex"
        );
        // 3. ...and inverse in the gravity
        assert!(
            (crate::reach_physics::jump_apex(p.jump_velocity, g * 0.5) - 2.0 * apex).abs() < 1e-4,
            "halving gravity must double the apex"
        );
        // And it is a jump at all: positive, finite, and zero gravity is refused rather than
        // dividing. (A band on the VALUE is deliberately absent -- see the note above.)
        assert!(apex > 0.0 && apex.is_finite());
        assert_eq!(crate::reach_physics::jump_apex(p.jump_velocity, 0.0), f32::INFINITY);
    }
}
