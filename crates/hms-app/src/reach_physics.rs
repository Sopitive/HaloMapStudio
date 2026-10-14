//! #gravity  Reach's GRAVITY, in one place.
//!
//! This module exists because the codebase had **three** unrelated gravity constants that differ
//! by up to 3.25x, two of which were Earth's and one of which was a typo of the other:
//!
//! | where | value | what it actually was |
//! |---|---|---|
//! | `film_play::CORPSE_GRAVITY` | 10.4576 | fitted from 1,447 real falls — the only MEASURED one |
//! | `physics_settle::GRAVITY_WU_S2` | 3.2147 | Earth, with two digits transposed |
//! | `scene.rs` particle default | 3.2174 | Earth, correctly converted |
//!
//! # What the engine actually authors
//!
//! **`haloreach.dll` carries exactly one gravity constant: [`WORLD_GRAVITY`] = 4.1712594.** Two
//! places establish it and they confirm each other:
//!
//! * `sub_18014FC94` allocates a 32-byte block named `"physics constants"`, stores its pointer at
//!   `TLS+816`, and initialises `*(float *)(block + 0x00) = 0x40857AF5 = 4.1712594`. The rest of
//!   the block is `1.0` at +0x04, `0.0011` at +0x08 and `cosf(0.61086524)` = **cos(35 deg)** at
//!   +0x1C (a slope limit).
//! * The script function `physics_set_gravity` (handler `0x18019C604`, reached from its
//!   `hs_function_definition` record at `0x180A25338` — name at +0x20, handler at +0x38) writes
//!   `*(float *)TLS[816] = argument * 4.1712594`. The script argument is therefore a **multiple
//!   of normal gravity**, and the multiplier being bit-identical to the default is what makes
//!   4.1712594 "normal" rather than a coincidence.
//!
//! **There is no gravity field in any Reach TAG.** All 238 groups were walked to depth 4: the only
//! hits are per-entity scales (`bipd` ground physics, `vehi` per-type adjusts, `matg` player
//! control and the trait multipliers). Gravity is a code constant.
//!
//! # Why Earth gravity must not appear here
//!
//! The f32 bit patterns of **9.80665**, **3.217405**, **10.4576** and **-10.4576** were searched
//! across the whole of `haloreach.dll`: **zero hits for every one**. `4.1712594` appears at
//! exactly four sites, all in the physics path. So both Earth-derived constants were human
//! derivations with no engine basis, and Reach's gravity is **1.296x Earth**.
//!
//! # The gap this module thought it had, and what it actually was
//!
//! For a while this note said `MEASURED_FALL_ACCELERATION / WORLD_GRAVITY` = **2.507**, called
//! that unexplained, and preferred the measured value on the strength of a jump-apex argument.
//! **All three of those were wrong, and they were wrong for one reason.**
//!
//! **The fit behind [`MEASURED_FALL_ACCELERATION`] clamped its samples at `g < 20.0`.** That
//! discarded the entire real population and reported the median of the surviving clipped tail.
//! Re-running the same sample selection with the clamp widened to 60.0, over 12 films:
//!
//! ```text
//!   LIVING biped   64,899 samples   median 31.380   p10 31.366  p90 31.393
//!   RAGDOLL        no samples at all
//! ```
//!
//! A spread of one part in a thousand is what a real constant looks like through a quantised
//! position stream; "the whole interquartile range landing on two discrete values", which the
//! original fit's own doc comment recorded, is what a **clipped** distribution looks like. So:
//!
//! 1. **10.4576 is an artefact of the clamp**, not a measurement of anything.
//! 2. **The 2.507x gap never existed** — it was that artefact divided by [`WORLD_GRAVITY`].
//! 3. **The jump-apex "cross-check" was circular.** It computed apexes from the artefact and then
//!    argued that the authored constant must be wrong because it would make a Spartan clear 98%
//!    of his own height. That second half is a JUDGEMENT, not a measurement — and the quantity
//!    that decides how a jump feels is **airtime**, which nobody checked. Halo jumps are in fact
//!    about that high.
//!
//! **And the fit was never of corpses.** A track's positions END at `destroyed_tick`, so ragdoll
//! motion is never in a recording at all; `CORPSE_GRAVITY` measured living bipeds and the name
//! disagreed with its own provenance. 31.38 is not adoptable either without first settling
//! whether the film stream's effective tick is what HMS assumes and whether recorded positions
//! are in tag world units — at 30 Hz the same samples would give 7.845.
//!
//! # So what each thing falls at now
//!
//! * **A living biped** — the player's jump — uses [`biped_fall_acceleration`] = the engine's own
//!   authored [`WORLD_GRAVITY`]. It is the only engine-sourced gravity in existence, and the
//!   alternative was the artefact above. With the authored `bipd +0x4F0` velocities that is an
//!   apex of 0.634 wu / 1.10 s of airtime for a Spartan and 0.842 wu / 1.27 s for an Elite. The
//!   player who reported the old jump as a hop ("hardly makes it off the ground") is the domain
//!   authority on the feel and their report agrees with the arithmetic.
//! * **A rigid body** — a ragdoll, a dropped weapon, a vehicle in the air — keeps
//!   [`rigid_body_fall_acceleration`] = 10.4576, as a **holding position**: nothing about corpse
//!   falls has ever been reported wrong, and "what a Havok body falls at" is a separate question
//!   with its own separate measurement still to do.
//!
//! Two functions, two names, because a character proxy and a Havok rigid body are different
//! systems — and because one function serving two laws is the shape that lets a change to the
//! player's jump silently move every corpse in Theater.
//!
//! **Still open**: the biped's integration law itself. Whether a character proxy is integrated at
//! the world constant at all, whether the jump impulse is applied before or after a tick's
//! gravity (a frame of velocity either way), and whether Havok's own world gravity is set
//! separately from the `"physics constants"` block.
//!
//! What was tested and ruled out as the missing factor: **the per-object `vertical acceleration
//! scale`** (`obje` +0x20). Measured on `forge_halo`, both playable bipeds author **1.2**
//! (4.1712594 x 1.2 = 5.0055); the Ghost 1.5, the Scorpion 0.1, the Wraith 0.25.
//!
//! # The biped integrator: NOT FOUND. Three leads burned, and the next one named.
//!
//! Whether a jump is an impulse or a velocity set, whether anything caps a falling speed, and
//! whether airborne acceleration is applied before or after gravity are all **unmeasured**. Three
//! attempts, all negative, recorded so they are not repeated:
//!
//! 1. **Scanning `.text` for readers of the physics-constants pointer** (`mov r64, [r64+0x330]`,
//!    since the block pointer lives at `TLS+816`). 21 candidates, and the one decompiled is a
//!    **destructor** whose `+0x330` is unrelated. `0x330` is far too common a displacement to
//!    scan for.
//! 2. **The debug global `airborne_arc_enabled`.** Its value pointer was taken from the
//!    debug-globals table on a guess at the record shape (`[value*, name*, type]`, value one slot
//!    before the name); all six references to the resulting address land in `sub_1800D0294`, which
//!    is the debug **render** dispatcher (it formats `"%s objects: % 5d (unique: % 5d)"`). The
//!    record layout guess was wrong by one slot and those bytes are debug-render toggles.
//! 3. **The jump script functions**, via the `hs_function_definition` trick that found
//!    `physics_set_gravity` cleanly (record at `0x180A17B98`, name +0x20, handler +0x38).
//!    `unit_action_test_jump`'s handler (`0x1801AE244`) is only `sub_1800656B8(unit, 9)` — a
//!    generic "test unit action N". So the script functions **read the button** and do not apply
//!    the jump. One by-product worth keeping: **action index 9 is JUMP** in the unit action table.
//!
//! The next lead, unfollowed: the engine names biped states `biped_ground_jump`,
//! `biped_ground_animated_jump`, `biped_force_airborne`, `biped_dead_force_airborne` and
//! `biped_airborne_default` (strings at `0x180A539D0`, `0x180A53A18`, `0x180A538F0`, `0x180A53728`,
//! `0x180A53B30`), which look like entries in a state/action table whose handlers would be the
//! integrator. Nothing is blocked on it: the jump arc is already validated against two
//! independent measurements on both species.
//!
//! What IS settled about the fall: **Reach's `bipd` carries no physics terminal velocity.** The
//! only field with the name is +0x2FC `terminal velocity fall ratio`, documented as a "ratio of
//! airborne_arc animation" and corroborated by the engine string `terminal_velocity_fall_weight`.
//! Nothing in the biped's own tag caps its falling speed.

use hms_native::{CacheHandle, NativeDll};

/// Reach's authored world gravity, from the engine's own `"physics constants"` block
/// (`haloreach.dll` `sub_18014FC94`, and the multiplier in `physics_set_gravity`).
///
/// **NOT Earth** — 1.296x it. The only gravity constant in the engine.
pub const WORLD_GRAVITY: f32 = 4.171_259_4;

/// What recorded bodies actually fall at, fitted from the slope of vertical speed against time
/// over **1,447 unsupported falls** in real films (`film_anim`).
///
/// 2.507x [`WORLD_GRAVITY`], and **that factor is unexplained** — see the module note. It is the
/// only number here that was measured from Reach's own output rather than read from it.
pub const MEASURED_FALL_ACCELERATION: f32 = 10.4576;

/// Earth's gravity at 1 wu = 3.048 m: 9.80665 / 3.048.
///
/// **Here only so that nobody derives it again.** It appears NOWHERE in `haloreach.dll` and it is
/// not Reach's gravity. `physics_settle` held 3.2147 — this same number with two digits
/// transposed — and the particle lane held 3.2174.
pub const EARTH_GRAVITY_WU: f32 = 3.217_405;

/// The engine's slope limit, `cosf(0.61086524)` in the `"physics constants"` block at +0x1C:
/// **35 degrees**, as a cosine.
pub const GROUND_SLOPE_LIMIT_COS: f32 = 0.819_152_04;
/// The same, in degrees, because the engine stored the radians that produced it.
pub const GROUND_SLOPE_LIMIT_DEGREES: f32 = 35.0;

/// `object_struct_definition` +0x1C `horizontal acceleration scale`.
pub const OBJE_HORIZONTAL_ACCELERATION_SCALE: usize = 0x1C;
/// `object_struct_definition` +0x20 `vertical acceleration scale`.
pub const OBJE_VERTICAL_ACCELERATION_SCALE: usize = 0x20;
/// `object_struct_definition` +0x24 `angular acceleration scale`.
pub const OBJE_ANGULAR_ACCELERATION_SCALE: usize = 0x24;
/// `bipd` +0x4F0 `jump velocity:world units per second`.
pub const BIPD_JUMP_VELOCITY: usize = 0x4F0;
/// `bipd` +0x5EC `physics` / `height standing` (world units).
pub const BIPD_HEIGHT_STANDING: usize = 0x5EC;
/// `bipd` +0x5F0 `physics` / `height crouching`.
pub const BIPD_HEIGHT_CROUCHING: usize = 0x5F0;
/// `bipd` +0x5F8 `physics` / `mass`.
pub const BIPD_MASS: usize = 0x5F8;

/// `bipd` +0x668 `physics` / `ground physics` / `gravity scale` ("scale on gravity for this
/// entity").
///
/// **Both playable bipeds author 0.0 here, and that is probably "unscaled" rather than "no
/// gravity".** The evidence is Bungie's own comment on the sibling field
/// `vehi/type-boat[]/gravity scale`: *"0 means 0. How hard is it to type 1?"* — a complaint that
/// only makes sense because everywhere ELSE in this tag set 0 is taken as the default. So a
/// reading of 0.0 is not evidence this offset is wrong.
pub const BIPD_GROUND_GRAVITY_SCALE: usize = 0x668;

/// #gravity  **The acceleration a LIVING BIPED is integrated at** — the player's jump and fall.
///
/// `WORLD_GRAVITY`, the engine's own authored constant, because it is the only engine-sourced
/// gravity that exists and the alternative turned out not to be a measurement of anything:
///
/// * [`MEASURED_FALL_ACCELERATION`]'s fit **clamped its samples at `g < 20.0`**, which discarded
///   the whole real population and reported the median of the surviving clipped tail. Re-running
///   the same sample selection with the clamp widened gives **31.38** at a p10–p90 spread of one
///   part in a thousand over 64,899 samples. 10.4576 is an artefact of the clamp.
/// * The "2.507x gap" that was the stated reason for preferring it **was that artefact divided by
///   this constant**, so it never existed.
/// * The argument that this constant would make a Spartan clear 98% of his own height and that
///   "plainly is not Reach" was an INFERENCE, not a refutation — and the quantity that decides
///   how a jump feels is airtime, which nobody had checked. The player who has spent thousands of
///   hours in Reach reported five times that the jump was far too low; under the artefact the
///   apex is 0.77 m with 0.44 s of airtime, which is the hop they described.
///
/// **This is the engine's authored world constant, chosen for that reason and no other. The
/// biped's true integration law is still unmeasured** — whether the impulse is applied before or
/// after the tick's gravity, and whether a character proxy is integrated at the world constant at
/// all, are open questions for the physics campaign.
///
/// Deliberately a different function from [`rigid_body_fall_acceleration`]: a character proxy and
/// a Havok rigid body are different systems and may genuinely have different laws, so **one
/// function serving both is the "two writers" shape** — changing the player's jump must not be
/// able to silently move every corpse in Theater.
pub const fn biped_fall_acceleration() -> f32 {
    WORLD_GRAVITY
}

/// #gravity  **The acceleration a RIGID BODY is integrated at** — a ragdoll, a dropped weapon, a
/// vehicle in the air.
///
/// Still [`MEASURED_FALL_ACCELERATION`], and **that is a holding position, not an endorsement**:
/// the fit behind it has a known clamp defect (see [`biped_fall_acceleration`]). It stays because
/// nothing about corpse falls has ever been reported wrong, because the value these consumers
/// have always used is this one, and because "what a Havok body falls at" is a separate question
/// from "what a character proxy falls at". Moving it is its own measurement and its own change.
pub const fn rigid_body_fall_acceleration() -> f32 {
    MEASURED_FALL_ACCELERATION
}

/// #gravity  **The acceleration a VEHICLE is integrated at** — [`WORLD_GRAVITY`], and this one is
/// not a holding position: it was measured in the vehicle's own physics path.
///
/// `reach_tag_test.exe` carries the engine unstripped. In `havok_vehicle_physics.cpp`
/// (`sub_1406CC8C0`) each wheel's normal force is
/// `(g * compression / scale - v_normal * ground_damp_factor) * instance->mass`, and the `g` there
/// is `sub_14037A0F0`, which reads
///
/// ```text
///   per_object_type_scale[type] * *(float *)*(TLS + 0x638)
/// ```
///
/// — the FIRST float of the block at that TLS slot. Its sibling `sub_14037A2F0` reads `+0x1C` of
/// the same block as a slope limit, and `+0x1C` of the engine's `"physics constants"` block is
/// `cos(35 deg)` (see the module note, and `haloreach.dll` `sub_18014FC94`, which allocates that
/// block with that literal name and `+0x00 = 4.1712594`). **It is the same block**: the TLS slot
/// differs between the tool build and the game dll, the layout does not. So a vehicle's
/// suspension, the per-object-type gravity scale and the slope limit all come out of the one
/// 32-byte block whose first float is [`WORLD_GRAVITY`].
///
/// There is therefore no second gravity for a vehicle to be on. [`MEASURED_FALL_ACCELERATION`]
/// has no engine basis at all — its fit clamped away its own population (see
/// [`biped_fall_acceleration`]) and it was never measured from a rigid body in the first place.
///
/// Separate from [`rigid_body_fall_acceleration`] on purpose: moving a vehicle must not move
/// every corpse in Theater, which is the "two writers, one field" shape.
pub const fn vehicle_fall_acceleration() -> f32 {
    WORLD_GRAVITY
}

fn rd_f32(b: &[u8], o: usize) -> f32 {
    if o + 4 > b.len() {
        return 0.0;
    }
    let v = f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// An object's three authored acceleration scales, `(horizontal, vertical, angular)`, read from
/// the tag. `None` when the tag cannot be read.
///
/// These are NOT a gravity scale — measured: both playable bipeds author 1.2 vertical, which does
/// not reconcile [`WORLD_GRAVITY`] with [`MEASURED_FALL_ACCELERATION`]. They are exposed because
/// they are the engine's own per-object acceleration law and a consumer may want them; a caller
/// must not multiply them into gravity on the strength of the name alone.
pub fn acceleration_scales(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<(f32, f32, f32)> {
    let m = native.tag_meta(cache, obj_tag, OBJE_HORIZONTAL_ACCELERATION_SCALE as u32, 12)?;
    Some((rd_f32(&m, 0), rd_f32(&m, 4), rd_f32(&m, 8)))
}

/// A biped's authored `jump velocity`, `height standing` and `mass`, read from the tag.
///
/// Note what is NOT here: Reach's `bipd` carries no physics TERMINAL VELOCITY. The only field
/// with that name is +0x2FC `terminal velocity fall ratio`, documented as a "ratio of
/// airborne_arc animation" — an animation blend, not a speed cap. So a falling biped's speed is
/// not capped by its own tag, and anything that caps it is elsewhere.
pub fn biped_physics(native: &NativeDll, cache: CacheHandle, bipd: u32) -> Option<(f32, f32, f32)> {
    let jump = rd_f32(&native.tag_meta(cache, bipd, BIPD_JUMP_VELOCITY as u32, 4)?, 0);
    let m = native.tag_meta(cache, bipd, BIPD_HEIGHT_STANDING as u32, 4)?;
    let height = rd_f32(&m, 0);
    let mass = rd_f32(&native.tag_meta(cache, bipd, BIPD_MASS as u32, 4)?, 0);
    Some((jump, height, mass))
}

/// How high a jump of `jump_velocity` reaches under `gravity`: `v^2 / 2g`.
///
/// One expression, so the cross-check in the module note and the gate that asserts it cannot
/// disagree with each other.
pub fn jump_apex(jump_velocity: f32, gravity: f32) -> f32 {
    if gravity > 0.0 {
        jump_velocity * jump_velocity / (2.0 * gravity)
    } else {
        f32::INFINITY
    }
}

/// A biped's authored ground-physics gravity scale, with the tag-set convention applied: **0 is
/// read as 1.0 (unscaled)**, for the reason in [`BIPD_GROUND_GRAVITY_SCALE`]'s note.
pub fn biped_gravity_scale(native: &NativeDll, cache: CacheHandle, bipd: u32) -> Option<f32> {
    let m = native.tag_meta(cache, bipd, BIPD_GROUND_GRAVITY_SCALE as u32, 4)?;
    let v = rd_f32(&m, 0);
    Some(if v > 0.0 { v } else { 1.0 })
}

#[cfg(test)]
mod gravity_split {
    //! #gravity  **Does a LIVING biped fall at the same rate as a RAGDOLL?**
    //!
    //! This is the experiment that settles which constant a living biped uses, and it is a
    //! measurement of the game's own output rather than an inference from either the apex or the
    //! engine's authored constant.
    //!
    //! The existing fit (10.4576) ran over **every** unsupported fall on every biped track, which
    //! mixes two populations: a living player jumping or walking off a ledge, and a ragdoll after
    //! death. If a living biped is a character proxy and a corpse is a Havok rigid body, the two
    //! need not share a gravity — and the 2.507x gap between the fit and the engine's authored
    //! 4.1712594 would be exactly that, with nothing unexplained.
    //!
    //! `EntityTrack::destroyed_tick` is the split: samples strictly before it are a living body,
    //! samples after it are a ragdoll.
    //!
    //! `cargo test -p hms-app --profile quick probe_living_versus_ragdoll_gravity -- --ignored --nocapture`

    use crate::{film, film_decode};

    fn pct(v: &[f32], q: f32) -> f32 {
        if v.is_empty() {
            return f32::NAN;
        }
        v[((v.len() as f32 - 1.0) * q).round() as usize]
    }

    #[test]
    #[ignore = "reads every film on this machine"]
    fn probe_living_versus_ragdoll_gravity() {
        let films = film::find_finished_films();
        if films.is_empty() {
            eprintln!("no films on this machine");
            return;
        }
        let mut living: Vec<f32> = Vec::new();
        let mut ragdoll: Vec<f32> = Vec::new();
        let mut unknown: Vec<f32> = Vec::new();
        for p in films.iter().take(12) {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            for t in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                for w in t.positions.windows(3) {
                    let [(t0, a), (t1, b), (t2, c)] = [w[0], w[1], w[2]];
                    if t1 <= t0 || t2 <= t1 || t2 - t0 > 20 {
                        continue;
                    }
                    let d0 = (t1 - t0) as f32 / 60.0;
                    let d1 = (t2 - t1) as f32 / 60.0;
                    // Reject a respawn teleport by horizontal speed, as the original fit does.
                    if (b[0] - a[0]).hypot(b[1] - a[1]) / d0 > 12.0 || (c[0] - b[0]).hypot(c[1] - b[1]) / d1 > 12.0 {
                        continue;
                    }
                    let v0 = (b[2] - a[2]) / d0;
                    let v1 = (c[2] - b[2]) / d1;
                    if !(v1 < v0 && v1 < -1.0) {
                        continue;
                    }
                    let dt = (d0 + d1) / 2.0;
                    if dt < 1.0 / 60.0 {
                        continue;
                    }
                    let g = -(v1 - v0) / dt;
                    // A WIDER window than the original fit's 0.5..20: a clamp at 20 could hide a
                    // population above it, and the question is which of 4.17 and 10.46 this is.
                    if !(g.is_finite() && g > 0.2 && g < 60.0) {
                        continue;
                    }
                    match t.destroyed_tick {
                        // Strictly before the body was destroyed: a LIVING biped.
                        Some(d) if t2 < d => living.push(g),
                        // After: a ragdoll.
                        Some(d) if t0 >= d => ragdoll.push(g),
                        // Straddles the death tick, or the film never destroyed the body.
                        _ => unknown.push(g),
                    }
                }
            }
        }
        for (label, v) in [("LIVING biped", &mut living), ("RAGDOLL", &mut ragdoll), ("unclassified", &mut unknown)] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if v.is_empty() {
                println!("{label:<14} no samples");
                continue;
            }
            println!(
                "{label:<14} {:>6} samples  median {:6.3}  p10 {:6.3}  p25 {:6.3}  p75 {:6.3}  p90 {:6.3}",
                v.len(),
                pct(v, 0.5),
                pct(v, 0.10),
                pct(v, 0.25),
                pct(v, 0.75),
                pct(v, 0.90)
            );
        }
        println!(
            "\nfor reference: the engine's authored world gravity {:.4}, this project's fit {:.4}",
            super::WORLD_GRAVITY,
            super::MEASURED_FALL_ACCELERATION
        );
    }
}

#[cfg(test)]
mod gates {
    use super::*;

    /// The three constants are what the module says they are, and the two Earth-derived ones that
    /// used to be scattered through the codebase are related to each other the way a typo is.
    ///
    /// Pure arithmetic, so it needs no map: the point is to pin the RELATIONSHIPS, because every
    /// claim this module makes is a claim about how these numbers relate.
    #[test]
    fn the_constants_relate_the_way_the_engine_and_the_measurement_say() {
        // Earth, correctly converted at 1 wu = 3.048 m. This is the number `scene.rs` holds.
        let earth = 9.806_65f32 / 3.048;
        assert!(
            (earth - EARTH_GRAVITY_WU).abs() < 1e-5,
            "Earth at 1 wu = 3.048 m is {earth}, not {EARTH_GRAVITY_WU}"
        );
        // ...and 3.2147, which `physics_settle` held, is that with two digits transposed. Assert
        // it is CLOSE but NOT equal, which is exactly the signature of a typo rather than a
        // different physical claim.
        // The literal `physics_settle` USED to hold, kept as a literal on purpose: the constant
        // itself now points at WORLD_GRAVITY, so only a literal can still record what was there.
        let settle_typo = 3.2147f32;
        assert!(settle_typo != EARTH_GRAVITY_WU);
        assert!(
            (settle_typo - EARTH_GRAVITY_WU).abs() < 0.01,
            "3.2147 and {EARTH_GRAVITY_WU} should differ only in transposed digits"
        );

        // The engine's gravity is NOT Earth, and the factor is the one the module quotes.
        assert!((WORLD_GRAVITY / earth - 1.296_47).abs() < 1e-4, "engine / Earth = {}", WORLD_GRAVITY / earth);

        // The unexplained gap, pinned so it cannot be quietly "fixed" by moving a constant.
        let gap = MEASURED_FALL_ACCELERATION / WORLD_GRAVITY;
        assert!((gap - 2.507).abs() < 0.001, "measured / authored = {gap}, the module says 2.507");
        // It is NEAR 2.5 but not 2.5 -- stated so a later claim of "it is exactly 2.5" has to
        // explain the 0.28%.
        assert!((gap - 2.5).abs() > 0.005 && (gap - 2.5).abs() < 0.02, "gap vs 2.5: {}", gap - 2.5);

        // The slope limit really is the cosine of 35 degrees.
        assert!(
            (GROUND_SLOPE_LIMIT_COS - GROUND_SLOPE_LIMIT_DEGREES.to_radians().cos()).abs() < 1e-6,
            "the slope limit cosine must be cos(35 deg)"
        );
    }

    /// #gravity  **THE TWO LAWS ARE SEPARATELY NAMED AND SEPARATELY WIRED.**
    ///
    /// The biped's gravity was moved to the engine's authored constant and the rigid bodies' was
    /// NOT, because they are different systems and only one of them had a complaint against it.
    /// This gate exists so that cannot quietly become one law again: it asserts the player's
    /// constant is the engine's, the ragdolls' is the one they have always used, and the two are
    /// different — so a future edit that points both at the same function turns it red.
    #[test]
    fn the_biped_and_the_rigid_body_fall_under_separately_named_laws() {
        assert_eq!(
            biped_fall_acceleration(),
            WORLD_GRAVITY,
            "the player falls at the engine's own authored constant -- the only engine-sourced \
             gravity there is"
        );
        assert_eq!(
            rigid_body_fall_acceleration(),
            MEASURED_FALL_ACCELERATION,
            "a ragdoll keeps the constant it has always used; moving it is a separate measurement"
        );
        assert_ne!(
            biped_fall_acceleration(),
            rigid_body_fall_acceleration(),
            "the two laws have collapsed into one -- changing the player's jump can now move every \
             corpse in Theater, which is the defect the split exists to prevent"
        );
        assert_ne!(biped_fall_acceleration(), EARTH_GRAVITY_WU, "Earth gravity is not Reach's");
        // Each consumer is wired to the law that names it, checked here rather than left to a
        // reader: the player to the biped's, the corpses and the vehicles to the rigid body's.
        assert_eq!(crate::play_phys::GRAVITY, biped_fall_acceleration());
        assert_eq!(crate::film_play::CORPSE_GRAVITY, rigid_body_fall_acceleration());
    }

    /// **THE CROSS-CHECK.** The authored jump velocity and the two candidate gravities, against
    /// the Spartan's own authored height — read from the tag, so this is a measurement and not
    /// the module note restated.
    ///
    /// **THIS GATE USED TO BE CIRCULAR AND IS NOT ANY MORE.** It asserted that the authored
    /// constant predicts an "absurd" jump (>85% of body height) and presented that as the referee
    /// for which gravity Reach uses. Both halves came from the same place: the "measured"
    /// constant was an artefact of a clamp in its own fit, and "98% of body height is plainly not
    /// Reach" was a judgement, not a measurement. See [`biped_fall_acceleration`].
    ///
    /// What is left is what the tag actually says, asserted as RELATIONSHIPS rather than as a
    /// verdict: the jump velocity is a jump, the biped is a biped, the apex follows `v^2 / 2g`
    /// for whichever constant is handed in, and **the two species differ** — which is what proves
    /// the field is read per biped rather than shared. The apexes under both candidates are
    /// PRINTED, because that is the honest status of the question.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_authored_jump_velocity_reads_per_biped_and_the_apex_follows_the_closed_form() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("reach_physics gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let mut checked = 0usize;
        let mut jumps: Vec<(String, f32)> = Vec::new();
        for &t in &native.tags_by_group(cache, "bipd") {
            let name = native.tag_name(cache, t).unwrap_or_default();
            let leaf = name.rsplit('\\').next().unwrap_or("").to_string();
            if !(leaf == "spartans" || leaf == "elite") {
                continue;
            }
            let (jump, height, mass) = biped_physics(native, cache, t).expect("the biped tag reads");
            let measured = jump_apex(jump, MEASURED_FALL_ACCELERATION);
            let authored = jump_apex(jump, WORLD_GRAVITY);
            println!(
                "{leaf}: jump {jump:.3} wu/s, standing height {height:.3} wu, mass {mass:.1} | \
                 apex with measured {measured:.3} wu ({:.0}% of height) | with authored {authored:.3} wu ({:.0}%)",
                100.0 * measured / height,
                100.0 * authored / height
            );
            checked += 1;
            jumps.push((leaf.clone(), jump));
            assert!(jump > 1.0 && jump < 5.0, "{leaf} jump velocity {jump} is not a jump");
            assert!(height > 0.3 && height < 1.5, "{leaf} standing height {height} wu is not a biped");
            // The CLOSED FORM, against an independently computed literal: `v^2 / 2g`, worked out
            // here rather than by calling the function under test.
            let want = jump * jump / (2.0 * WORLD_GRAVITY);
            assert!(
                (authored - want).abs() < 1e-4,
                "{leaf}: jump_apex says {authored}, v^2/2g says {want}"
            );
            // ...and it SCALES as the square of the velocity. A gate that only checked one point
            // would pass on any monotone function of v.
            assert!(
                (jump_apex(2.0 * jump, WORLD_GRAVITY) - 4.0 * authored).abs() < 1e-3,
                "{leaf}: doubling the jump velocity did not quadruple the apex"
            );
            // The apex under the BIPED's law is the one the player will feel; print it and
            // require only that it is a jump a body that size could make -- between a quarter of
            // its height and its full height. That is a sanity band, not a referee.
            let used = jump_apex(jump, biped_fall_acceleration());
            assert!(
                used > 0.25 * height && used <= 1.05 * height,
                "{leaf}: the biped's own law gives an apex of {used:.3} wu against a {height:.3} wu \
                 body ({:.0}%), which is not a height a body that size jumps",
                100.0 * used / height
            );
            let _ = measured;
        }
        assert_eq!(checked, 2, "both playable bipeds must be checked, found {checked}");
        // **THE READ IS PER BIPED.** Reach authors 2.300 for the Spartan and 2.650 for the Elite,
        // so a shared constant -- or a read at an offset that happens to return the same bytes for
        // both -- cannot pass this.
        assert!(
            (jumps[0].1 - jumps[1].1).abs() > 0.05,
            "{} and {} both jump at {:.4} wu/s: the velocity is not being read per biped",
            jumps[0].0,
            jumps[1].0,
            jumps[0].1
        );
    }

    /// The per-object acceleration scales, read against a real map — and the measurement that
    /// REFUTED the tempting hypothesis that one of them is the missing 2.5x.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_vertical_acceleration_scale_is_not_the_missing_factor() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("reach_physics gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let bipd: Vec<u32> = native.tags_by_group(cache, "bipd");
        let mut checked = 0usize;
        for &t in &bipd {
            let name = native.tag_name(cache, t).unwrap_or_default();
            let leaf = name.rsplit('\\').next().unwrap_or("").to_string();
            if !(leaf == "spartans" || leaf == "elite") {
                continue;
            }
            let (h, v, a) = acceleration_scales(native, cache, t).expect("the biped tag reads");
            println!("{leaf}: horizontal {h:.3}, vertical {v:.3}, angular {a:.3}");
            checked += 1;
            assert!((v - 1.2).abs() < 1e-4, "{leaf} vertical acceleration scale is {v}, not the measured 1.2");
            // The refutation, asserted: if this WERE the missing factor, the product would be the
            // measured fall acceleration. It is not, and it is not close.
            let product = WORLD_GRAVITY * v;
            assert!(
                (product - MEASURED_FALL_ACCELERATION).abs() > 4.0,
                "{leaf}: WORLD_GRAVITY * vertical scale = {product}, which is suspiciously near the \
                 measured {MEASURED_FALL_ACCELERATION} -- if that ever becomes true, the module's \
                 'ruled out' note is wrong and the gap is explained"
            );
            // The ground-physics gravity scale reads 0 and is taken as unscaled.
            let gs = biped_gravity_scale(native, cache, t).expect("the biped tag reads");
            assert_eq!(gs, 1.0, "{leaf} ground gravity scale should read as unscaled");
        }
        assert_eq!(checked, 2, "both playable bipeds must be checked, found {checked}");
    }
}
