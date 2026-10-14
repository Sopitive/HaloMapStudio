//! #projectile  Reach `weapon` BARRELS and `projectile` tag fields.
//!
//! What a shot IS, in the engine's own words: a `weap` carries up to two `barrels`
//! (`k_weapon_barrel_count`), each of which names the `proj` it launches, the marker it launches
//! it from, and a `barrel_firing_effect_block` whose `firing effect^` is the muzzle `effe`. The
//! `proj` in turn carries the flight law (initial / final velocity, air gravity scale, the
//! detonation countdown, the maximum range) and the three effects a projectile plays: the
//! airborne and ground DETONATION effects and the IMPACT effects.
//!
//! # Where these offsets come from
//!
//! `reach_tag_test.exe`'s own Blam reflection data, read out of the PE. The chain is the one
//! `reverseme tagdefs` walks — `tag_group` +0x20 -> `tag_block_definition` +0x20 -> field set,
//! field set +0x30 fields / +0x38 element size — and the field records are 0x20 bytes
//! `{i32 type, i32 pad, char* name, void* definition, void* _}` with the type indexing the
//! `char*[63]` table at RVA 0x1E09930.
//!
//! **`weapon_barrels` and `projectile` are the two field sets `reverseme tagdefs` prints as
//! `?`**, because their field-set `fields` pointer is NULL in the image and installed by a
//! dynamic initializer (`weapon_barrels`' is copied into place by the thunk at 0x140030370).
//! Their field arrays are in `.data` all the same and were recovered by content, then the
//! element size each one implies was **measured against a real cache** rather than assumed —
//! see [`probes::probe_barrel_stride_and_projectile_layout`], which is the gate.
//!
//! The marker trap from `hms-fx-marker-fallback` applies here: a barrel whose
//! `optional barrel marker name` is 0, or names a marker the weapon's model does not have, gets
//! **no** silent substitution of the weapon origin. [`Barrel::marker_sid`] is reported as-is and
//! the caller must decide, loudly.

use hms_native::{CacheHandle, NativeDll};

use crate::reach_actor::{block_at, tag_block};

// ---------------------------------------------------------------------------------------------
// weap
// ---------------------------------------------------------------------------------------------

/// `weapon_struct_definition` +0x204 within the tag; `barrels` is the block at flat +0x408.
pub const WEAP_BARRELS: usize = 0x408;
/// `k_weapon_barrel_count`.
pub const WEAP_BARREL_MAX: i32 = 2;

/// `weapon_barrels` element size. **Measured** — see the module note and the stride gate.
pub const BARREL_SIZE: usize = 388;

// Offsets inside one `weapon_barrels` element.
pub const BARREL_FLAGS: usize = 0x00;
/// `rounds per second` [initial, final]. Measured: the DMR authors 3.0, the assault rifle 10.0,
/// the needler 8..12 — and the single-shot weapons (magnum / sniper / shotgun / rocket) all
/// author the 30.0 cap and let `fire recovery time` set the real rate.
pub const BARREL_ROUNDS_PER_SECOND: usize = 0x04; // real bounds
/// `optional barrel marker name`. **Authored by only 7 of the 56 barrels on `forge_halo`**
/// (`primary_trigger` / `secondary_trigger` / `b_mount`); the rest leave it 0 and the firing
/// EFFECT names its own markers instead — see [`effect_location_names`].
pub const BARREL_MARKER_NAME: usize = 0x40;
pub const BARREL_PROJECTILE: usize = 0x104; // tag reference -> proj
pub const BARREL_FIRING_EFFECTS: usize = 0x178; // block -> barrel_firing_effect_block

/// `barrel_firing_effect_block` element size: 2 + 2 + 13 tag references.
pub const FIRING_EFFECT_SIZE: usize = 4 + 13 * 16;
/// `firing effect^` — "this effect is used when the weapon is loaded and fired normally".
pub const FIRING_EFFECT_EFFECT: usize = 0x04;
/// `firing damage`, a `jpt!`.
pub const FIRING_EFFECT_DAMAGE: usize = 0x44;
pub const FIRING_EFFECT_MAX: i32 = 4; // k_weapon_barrel_effect_count

/// The second candidate layout the reflection data offers for `weapon_barrels`, kept so the gate
/// can state what it ruled out. See the module note.
pub const BARREL_SIZE_ALT: usize = 304;

// ---------------------------------------------------------------------------------------------
// proj
// ---------------------------------------------------------------------------------------------

/// `object_struct_definition` is 360 bytes, so the projectile's own fields start at +0x168.
pub const PROJ_BASE: usize = 0x168;

pub const PROJ_FLAGS: usize = 0x168;
pub const PROJ_DETONATION_TIMER_STARTS: usize = 0x16c; // short enum
pub const PROJ_COLLISION_RADIUS: usize = 0x170;
pub const PROJ_ARMING_TIME: usize = 0x174;
pub const PROJ_DANGER_RADIUS: usize = 0x178;
pub const PROJ_TIMER: usize = 0x184; // real bounds, "detonation countdown (zero is untimed)"
pub const PROJ_MINIMUM_VELOCITY: usize = 0x18c;
pub const PROJ_MAXIMUM_RANGE: usize = 0x190;
pub const PROJ_DETONATION_STARTED: usize = 0x1a8; // effe
pub const PROJ_DETONATION_AIRBORNE: usize = 0x1b8; // effe
pub const PROJ_DETONATION_GROUND: usize = 0x1c8; // effe
pub const PROJ_DETONATION_DAMAGE: usize = 0x1d8; // jpt!
pub const PROJ_IMPACT_EFFECT: usize = 0x264; // effe
pub const PROJ_OBJECT_IMPACT_EFFECT: usize = 0x274; // effe
pub const PROJ_IMPACT_DAMAGE: usize = 0x284; // jpt!
pub const PROJ_AIR_GRAVITY_SCALE: usize = 0x2b8;
pub const PROJ_WATER_GRAVITY_SCALE: usize = 0x2c4;
pub const PROJ_INITIAL_VELOCITY: usize = 0x2d0;
pub const PROJ_FINAL_VELOCITY: usize = 0x2d4;
pub const PROJ_ACCELERATION_RANGE: usize = 0x2fc; // real bounds
/// The whole `proj` tag, object part included.
pub const PROJ_SIZE: usize = 0x360;

/// The LEGACY (Halo-3-era) projectile layout the same reflection data still carries, kept so the
/// gate can state what it ruled out: `initial velocity` would be at +0x1D0 and the tag 0x258
/// bytes long.
pub const PROJ_SIZE_LEGACY: usize = 0x258;
pub const PROJ_INITIAL_VELOCITY_LEGACY: usize = 0x1d0;

// ---------------------------------------------------------------------------------------------
// readers
// ---------------------------------------------------------------------------------------------

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

fn rd_u32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn rd_i16(b: &[u8], o: usize) -> i16 {
    if o + 2 > b.len() {
        return 0;
    }
    i16::from_le_bytes([b[o], b[o + 1]])
}

/// A Reach tag reference is 16 bytes with the datum at +0x0C; the tag index is its low half.
pub fn tag_ref(b: &[u8], o: usize) -> Option<u32> {
    let d = rd_u32(b, o + 0x0C);
    (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
}

/// One `weapon_barrels` element, with only the fields a shot needs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Barrel {
    /// `optional barrel marker name`. **0 means the weapon authored none** — say so rather than
    /// substituting the weapon's origin (`hms-fx-marker-fallback`).
    pub marker_sid: u32,
    /// The `proj` this barrel launches, if any (a melee-only or beam barrel has none).
    pub projectile: Option<u32>,
    /// `rounds per second` [initial, final].
    pub rounds_per_second: [f32; 2],
    /// `firing effect^` of `firing effects[0]`: the muzzle `effe`.
    pub firing_effect: Option<u32>,
    /// `firing damage` of `firing effects[0]`: a `jpt!`.
    pub firing_damage: Option<u32>,
    /// How many `firing effects` rows the barrel authored (>1 means the engine picks among them
    /// by shot count; this reader takes row 0 and the caller is told).
    pub firing_effect_rows: usize,
}

/// Every barrel of one `weap`.
pub fn weapon_barrels(native: &NativeDll, cache: CacheHandle, weap: u32) -> Vec<Barrel> {
    let b = tag_block(native, cache, weap, WEAP_BARRELS, BARREL_SIZE, WEAP_BARREL_MAX);
    let n = b.len() / BARREL_SIZE;
    (0..n)
        .map(|i| {
            let e = &b[i * BARREL_SIZE..(i + 1) * BARREL_SIZE];
            let fx = block_at(native, cache, e, BARREL_FIRING_EFFECTS, FIRING_EFFECT_SIZE, FIRING_EFFECT_MAX);
            let rows = fx.len() / FIRING_EFFECT_SIZE;
            Barrel {
                marker_sid: rd_u32(e, BARREL_MARKER_NAME),
                projectile: tag_ref(e, BARREL_PROJECTILE),
                rounds_per_second: [rd_f32(e, BARREL_ROUNDS_PER_SECOND), rd_f32(e, BARREL_ROUNDS_PER_SECOND + 4)],
                firing_effect: (rows > 0).then(|| tag_ref(&fx, FIRING_EFFECT_EFFECT)).flatten(),
                firing_damage: (rows > 0).then(|| tag_ref(&fx, FIRING_EFFECT_DAMAGE)).flatten(),
                firing_effect_rows: rows,
            }
        })
        .collect()
}

/// The flight and detonation law of one `proj`, straight off the tag.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProjectileDef {
    pub proj: u32,
    /// `initial velocity:world units per second`.
    pub initial_velocity: f32,
    /// `final velocity:world units per second`.
    pub final_velocity: f32,
    /// `acceleration range:world units` — over this distance the speed goes initial -> final.
    pub acceleration_range: [f32; 2],
    /// `air gravity scale` — the fraction of gravity that applies in air. 0 = flat flight.
    pub air_gravity_scale: f32,
    /// `timer:seconds` [min, max]; zero is untimed.
    pub timer: [f32; 2],
    /// `minimum velocity` — detonates when slowed below this.
    pub minimum_velocity: f32,
    /// `maximum range:world units` — detonates after travelling this far. 0 = unlimited.
    pub maximum_range: f32,
    pub collision_radius: f32,
    pub arming_time: f32,
    /// `detonation effect (airborne)`.
    pub detonation_airborne: Option<u32>,
    /// `detonation effect (ground)`.
    pub detonation_ground: Option<u32>,
    /// `impact effect` — world geometry.
    pub impact_effect: Option<u32>,
    /// `object impact effect`.
    pub object_impact_effect: Option<u32>,
}

impl ProjectileDef {
    /// Speed at `travelled` world units from the muzzle, by the engine's own two-point ramp.
    ///
    /// `acceleration range` is the window over which the authored speed goes from
    /// `initial velocity` to `final velocity`; outside it the speed is clamped to the endpoint.
    /// A projectile with a zero-width range is a constant-speed projectile, which is almost all
    /// of them.
    pub fn speed_at(&self, travelled: f32) -> f32 {
        let (a, b) = (self.acceleration_range[0], self.acceleration_range[1]);
        if !(b > a) {
            return self.initial_velocity;
        }
        let t = ((travelled - a) / (b - a)).clamp(0.0, 1.0);
        self.initial_velocity + (self.final_velocity - self.initial_velocity) * t
    }
    /// The detonation effect for a projectile that ran out of life in the air vs on contact.
    pub fn detonation_effect(&self, airborne: bool) -> Option<u32> {
        if airborne {
            self.detonation_airborne.or(self.detonation_ground)
        } else {
            self.detonation_ground.or(self.detonation_airborne)
        }
    }
}

/// Read one `proj`. `None` when the tag cannot be read to its full authored length, which is the
/// same test the stride gate uses.
pub fn projectile_def(native: &NativeDll, cache: CacheHandle, proj: u32) -> Option<ProjectileDef> {
    // One read of the whole tag: `tag_meta` returns None unless the WHOLE length is readable, so
    // asking for `PROJ_SIZE` is simultaneously the read and the layout check.
    let b = native.tag_meta(cache, proj, 0, PROJ_SIZE as u32)?;
    Some(ProjectileDef {
        proj,
        initial_velocity: rd_f32(&b, PROJ_INITIAL_VELOCITY),
        final_velocity: rd_f32(&b, PROJ_FINAL_VELOCITY),
        acceleration_range: [rd_f32(&b, PROJ_ACCELERATION_RANGE), rd_f32(&b, PROJ_ACCELERATION_RANGE + 4)],
        air_gravity_scale: rd_f32(&b, PROJ_AIR_GRAVITY_SCALE),
        timer: [rd_f32(&b, PROJ_TIMER), rd_f32(&b, PROJ_TIMER + 4)],
        minimum_velocity: rd_f32(&b, PROJ_MINIMUM_VELOCITY),
        maximum_range: rd_f32(&b, PROJ_MAXIMUM_RANGE),
        collision_radius: rd_f32(&b, PROJ_COLLISION_RADIUS),
        arming_time: rd_f32(&b, PROJ_ARMING_TIME),
        detonation_airborne: tag_ref(&b, PROJ_DETONATION_AIRBORNE),
        detonation_ground: tag_ref(&b, PROJ_DETONATION_GROUND),
        impact_effect: tag_ref(&b, PROJ_IMPACT_EFFECT),
        object_impact_effect: tag_ref(&b, PROJ_OBJECT_IMPACT_EFFECT),
    })
}


/// The MARKER NAMES an `effe` names in its own `effect_locations` block (`effe` +0x20, **8-byte**
/// elements with the marker string id first — the size `hms-fx-marker-fallback`'s sibling bug got
/// wrong as 12). This is how a firing effect says where it goes: the weapon's barrel almost never
/// authors `optional barrel marker name`, and the effect does.
pub fn effect_location_names(native: &NativeDll, cache: CacheHandle, effe: u32) -> Vec<String> {
    let b = tag_block(native, cache, effe, EFFE_LOCATIONS, EFFECT_LOCATION_SIZE, 64);
    (0..b.len() / EFFECT_LOCATION_SIZE)
        .filter_map(|i| {
            let sid = rd_u32(&b, i * EFFECT_LOCATION_SIZE);
            (sid != 0).then(|| native.string_id(cache, sid)).flatten().filter(|s| !s.is_empty())
        })
        .collect()
}

/// `effect_definition` +0x20 `effect_locations`.
pub const EFFE_LOCATIONS: usize = 0x20;
/// One `effect_locations_block` element. **8 bytes**, not 12 — the bug that had been in the
/// particle decode (see the actor FX lane's note).
pub const EFFECT_LOCATION_SIZE: usize = 8;

/// #projectile  An object's ATTACHMENTS that are effects: `obje` +0x104
/// `global_object_attachment_block` (32-byte elements, max 24) -> `(effe tag, marker string id,
/// primary scale function string id)`.
///
/// The same block `SceneController::object_effects` reads; this is the native-only form, so the
/// film driver can build its table without a scene in hand.
pub fn object_attachment_effects(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Vec<(u32, u32, u32)> {
    let b = tag_block(native, cache, obj_tag, OBJE_ATTACHMENTS, OBJE_ATTACHMENT_SIZE, 24);
    (0..b.len() / OBJE_ATTACHMENT_SIZE)
        .filter_map(|i| {
            let e = &b[i * OBJE_ATTACHMENT_SIZE..(i + 1) * OBJE_ATTACHMENT_SIZE];
            // The attachment's `type` is a tag reference to any of several groups (`effe`,
            // `ligh`, `lens`); only the `effe` ones are effects. The class fourcc is stored
            // big-endian, exactly as `SceneController::object_effects` tests it, and the primary
            // SCALE function sid is at +0x18 (not +0x14).
            if rd_u32(e, 0) != u32::from_be_bytes(*b"effe") { return None; }
            let effe = tag_ref(e, 0x00)?;
            Some((effe, rd_u32(e, 0x10), rd_u32(e, 0x18)))
        })
        .collect()
}

/// `obje` +0x104 `attachments`.
pub const OBJE_ATTACHMENTS: usize = 0x104;
/// One `global_object_attachment_block` element.
pub const OBJE_ATTACHMENT_SIZE: usize = 32;

// ---------------------------------------------------------------------------------------------
// probes
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod probes {
    use super::*;

    /// The ONE shared Reach cache the map-backed gates use (`reach_actor::tests::open_reach_for`),
    /// not a second copy of the map-opening dance.
    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// **The stride gate.** The reflection data offers two candidate element sizes for
    /// `weapon_barrels` and two candidate layouts for `projectile`, because both field sets'
    /// pointers are installed at runtime. This decides between them against a real cache and
    /// prints the evidence, so the constants above are measured rather than chosen.
    ///
    /// The test that discriminates: with the right stride, barrel 0's `projectile` tag reference
    /// resolves to a tag whose class really is `proj`, for every weapon that fires one.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_barrel_stride_and_projectile_layout() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_projectile: forge_halo cache not available");
            return;
        };
        let native = &*native;
        let weaps = native.tags_by_group(cache, "weap");
        println!("== weapon_barrels stride candidates over {} weap tags ==", weaps.len());
        for &stride in &[252usize, 304, 312, 364] {
            let mut blocks = 0usize;
            let mut proj_ok = 0usize;
            let mut proj_bad = 0usize;
            let mut fx_ok = 0usize;
            let mut fx_bad = 0usize;
            for &w in &weaps {
                let b = tag_block(native, cache, w, WEAP_BARRELS, stride, WEAP_BARREL_MAX);
                if b.is_empty() {
                    continue;
                }
                blocks += 1;
                for i in 0..b.len() / stride {
                    let e = &b[i * stride..(i + 1) * stride];
                    match tag_ref(e, BARREL_PROJECTILE) {
                        Some(t) => {
                            if native.tag_class(cache, t).as_deref() == Some("proj") {
                                proj_ok += 1
                            } else {
                                proj_bad += 1
                            }
                        }
                        None => {}
                    }
                    let fx = block_at(native, cache, e, BARREL_FIRING_EFFECTS, FIRING_EFFECT_SIZE, FIRING_EFFECT_MAX);
                    if !fx.is_empty() {
                        match tag_ref(&fx, FIRING_EFFECT_EFFECT) {
                            Some(t) => {
                                if native.tag_class(cache, t).as_deref() == Some("effe") {
                                    fx_ok += 1
                                } else {
                                    fx_bad += 1
                                }
                            }
                            None => {}
                        }
                    }
                }
            }
            println!(
                "  stride {stride:>4}: {blocks:>3} weapons with a barrels block | projectile -> proj {proj_ok:>3} ok / {proj_bad:>3} WRONG CLASS | firing effect -> effe {fx_ok:>3} ok / {fx_bad:>3} WRONG CLASS"
            );
        }

        println!("\n== proj tag length ==");
        let projs = native.tags_by_group(cache, "proj");
        let mut reach = 0usize;
        let mut legacy_only = 0usize;
        for &p in &projs {
            if native.tag_meta(cache, p, 0, PROJ_SIZE as u32).is_some() {
                reach += 1;
            } else if native.tag_meta(cache, p, 0, PROJ_SIZE_LEGACY as u32).is_some() {
                legacy_only += 1;
            }
        }
        println!(
            "  {} proj tags: {reach} readable to {:#x} (the Reach layout), {legacy_only} only to {:#x} (the legacy one)",
            projs.len(),
            PROJ_SIZE,
            PROJ_SIZE_LEGACY
        );

        println!("\n== what each layout says about the same tags ==");
        println!(
            "{:<58} {:>10} {:>10} {:>8} {:>9} {:>9} {:>9}",
            "proj", "v0(Reach)", "v0(legacy)", "vfinal", "gravityS", "timer.lo", "maxrange"
        );
        for &p in projs.iter().take(40) {
            let name = native.tag_name(cache, p).unwrap_or_default();
            let Some(b) = native.tag_meta(cache, p, 0, PROJ_SIZE as u32) else { continue };
            println!(
                "{:<58} {:>10.2} {:>10.2} {:>8.2} {:>9.3} {:>9.2} {:>9.1}",
                name.rsplit('\\').next().unwrap_or(&name),
                rd_f32(&b, PROJ_INITIAL_VELOCITY),
                rd_f32(&b, PROJ_INITIAL_VELOCITY_LEGACY),
                rd_f32(&b, PROJ_FINAL_VELOCITY),
                rd_f32(&b, PROJ_AIR_GRAVITY_SCALE),
                rd_f32(&b, PROJ_TIMER),
                rd_f32(&b, PROJ_MAXIMUM_RANGE),
            );
        }
    }

    /// Every weapon's barrels, with the effects and the projectile named — the human-readable
    /// form of what the two features consume.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_weapon_barrels() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_projectile: forge_halo cache not available");
            return;
        };
        let native = &*native;
        let mut no_marker = 0usize;
        let mut no_fx = 0usize;
        let mut rows = 0usize;
        for &w in &native.tags_by_group(cache, "weap") {
            let bs = weapon_barrels(native, cache, w);
            if bs.is_empty() {
                continue;
            }
            let wname = native.tag_name(cache, w).unwrap_or_default();
            println!("{}", wname);
            for (i, b) in bs.iter().enumerate() {
                rows += 1;
                if b.marker_sid == 0 {
                    no_marker += 1;
                }
                if b.firing_effect.is_none() {
                    no_fx += 1;
                }
                let marker = if b.marker_sid == 0 {
                    "<NO MARKER AUTHORED>".to_string()
                } else {
                    native.string_id(cache, b.marker_sid).unwrap_or_else(|| format!("sid {:#x}?", b.marker_sid))
                };
                let nm = |t: Option<u32>| {
                    t.and_then(|t| native.tag_name(cache, t))
                        .map(|p| p.rsplit('\\').next().unwrap_or(&p).to_string())
                        .unwrap_or_else(|| "-".into())
                };
                println!(
                    "   barrel {i}: marker '{marker}' rps {:.1}..{:.1} | proj {} | fx {} ({} row(s)) | dmg {}",
                    b.rounds_per_second[0],
                    b.rounds_per_second[1],
                    nm(b.projectile),
                    nm(b.firing_effect),
                    b.firing_effect_rows,
                    nm(b.firing_damage),
                );
            }
        }
        println!("\n{rows} barrels: {no_marker} with no authored marker, {no_fx} with no firing effect");
    }
}

#[cfg(test)]
mod film_probes {
    //! #projectile  What a recording actually says about projectiles.
    //!
    //! The premise handed to this work was "projectiles are not replicated as film entities, so
    //! they must be simulated". That is only half true and this measures which half: the engine
    //! entity table has a `projectile` type (16) and the event channel has SIX projectile events
    //! (`projectile_attached` 4, `projectile_detonate` 5, `projectile_impact_effect` 6,
    //! `projectile_object_impact_effect` 7, `request_projectile_attach` 43,
    //! `projectile_supercombine_request` 45). Run this before building anything on top.
    //!
    //! `cargo test -p hms-app --profile quick probe_what_the_recording_says_about_projectiles -- --ignored --nocapture`

    use crate::{film, film_action, film_decode};
    use std::collections::BTreeMap;

    #[test]
    #[ignore = "reads every film on this machine"]
    fn probe_what_the_recording_says_about_projectiles() {
        let films = film::find_finished_films();
        if films.is_empty() {
            eprintln!("reach_projectile: no films on this machine");
            return;
        }
        const WATCH: [(u8, &str); 9] = [
            (0, "damage_aftermath"),
            (4, "proj_attached"),
            (5, "proj_detonate"),
            (6, "proj_impact_fx"),
            (7, "proj_obj_impact"),
            (25, "req_wpn_fire"),
            (26, "action_wpn_fire"),
            (43, "req_proj_attach"),
            (45, "supercombine"),
        ];
        println!(
            "{:<46} {:>6} {:>6} {:>7} {}",
            "film", "secs", "proj16", "mover16", WATCH.iter().map(|(_, n)| format!("{:>16}", *n)).collect::<Vec<_>>().join("")
        );
        // Payload-shape census for the two events with no object reference, so their parameter
        // blocks can be decoded later instead of guessed at.
        let mut bits: BTreeMap<(u8, u16), u64> = BTreeMap::new();
        let mut refs: BTreeMap<(u8, u8), u64> = BTreeMap::new();
        let mut totals = [0u64; 9];
        let mut proj_tracks = 0u64;
        for p in films.iter() {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() {
                continue;
            }
            let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            let mut n = [0u64; 9];
            for ev in &idx.events {
                if let Some(k) = WATCH.iter().position(|(x, _)| *x == ev.kind) {
                    n[k] += 1;
                    totals[k] += 1;
                }
            }
            // The raw payload shapes, which `build_index` has already folded away.
            let (params, _) = film_decode::solve_params(&f.bytes, &f.data);
            for rec in f.data.records.iter() {
                let Ok(t) = film_decode::decode_record(rec.body(&f.bytes), &params) else { continue };
                for (i, c) in t.commands.iter().enumerate() {
                    if i < t.commands_a || c.kind != 1 {
                        continue;
                    }
                    let Some(ev) = film_action::decode_event(&c.payload) else { continue };
                    if matches!(ev.kind, 4 | 5 | 6 | 7 | 26 | 43 | 45) {
                        *bits.entry((ev.kind, ev.param_bits as u16)).or_default() += 1;
                        *refs.entry((ev.kind, ev.ref_count as u8)).or_default() += 1;
                    }
                }
            }
            let p16: Vec<_> = idx.tracks.iter().filter(|t| t.entity_type == 16).collect();
            let movers = p16.iter().filter(|t| t.positions.len() > 1).count();
            proj_tracks += p16.len() as u64;
            if n.iter().all(|&x| x == 0) && p16.is_empty() {
                continue;
            }
            let lab = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            println!(
                "{:<46} {:>6} {:>6} {:>7} {}",
                &lab[..lab.len().min(45)],
                f.content.seconds,
                p16.len(),
                movers,
                n.iter().map(|v| format!("{v:>16}")).collect::<Vec<_>>().join("")
            );
        }
        println!("\ntotals: {} projectile ENTITY tracks", proj_tracks);
        for (k, (_, nm)) in WATCH.iter().enumerate() {
            println!("  {nm:<18} {}", totals[k]);
        }
        println!("\n== payload shapes (kind, parameter bits) -> count ==");
        for ((k, b), c) in bits.iter() {
            println!("  {:<2} {:<32} {b:>4} bits  x{c}", k, film_action::event_name(*k), );
        }
        println!("\n== (kind, object reference count) -> count ==");
        for ((k, r), c) in refs.iter() {
            println!("  {:<2} {:<32} {r} ref(s)  x{c}", k, film_action::event_name(*k));
        }
    }

    /// The recorded projectile tracks of ONE film, in detail: what they were spawned from, how
    /// long they lived, and how far they flew. The input the draw lane and the simulation gate
    /// both need.
    ///
    /// `HMS_FILM=<path> cargo test -p hms-app --profile quick probe_recorded_projectile_tracks -- --ignored --nocapture`
    #[test]
    #[ignore = "reads one film"]
    fn probe_recorded_projectile_tracks() {
        let path = match std::env::var("HMS_FILM") {
            Ok(v) => std::path::PathBuf::from(v),
            Err(_) => {
                // The longest film on the machine, which is the one the corpus talks about.
                match film::find_finished_films().into_iter().next() {
                    Some(p) => p,
                    None => return,
                }
            }
        };
        let Ok(f) = film::open(&path) else {
            eprintln!("reach_projectile: cannot open {:?}", path.file_name());
            return;
        };
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        println!(
            "film {:?}  {} s  {} tracks",
            path.file_name().unwrap_or_default(),
            f.content.seconds,
            idx.tracks.len()
        );
        let mut by_def: BTreeMap<Option<u32>, (usize, usize)> = BTreeMap::new();
        let mut rows = 0usize;
        for t in idx.tracks.iter().filter(|t| t.entity_type == 16) {
            let e = by_def.entry(t.def_wire).or_default();
            e.0 += 1;
            if t.positions.len() > 1 {
                e.1 += 1;
            }
            if rows < 40 && t.positions.len() > 1 {
                rows += 1;
                let first = t.positions.first().unwrap();
                let last = t.positions.last().unwrap();
                let d = ((last.1[0] - first.1[0]).powi(2) + (last.1[1] - first.1[1]).powi(2) + (last.1[2] - first.1[2]).powi(2)).sqrt();
                let dt = (last.0 - first.0) as f32 / 60.0;
                println!(
                    "  #{:#x} def {:?} ticks {}..{} ({} samples) {:.2} wu over {:.3} s = {:.1} wu/s",
                    t.net_id,
                    t.def_wire,
                    t.first_tick,
                    t.last_tick,
                    t.positions.len(),
                    d,
                    dt,
                    if dt > 0.0 { d / dt } else { 0.0 }
                );
            }
        }
        println!("\nper definition wire value: (tracks, of which moving)");
        for (w, (n, m)) in by_def.iter() {
            println!("  {w:?}  {n} / {m}");
        }
    }
}

#[cfg(test)]
mod locate {
    //! #projectile  WHERE the barrels block really is, found in the cache rather than assumed.
    //!
    //! The stride gate found a block at the reflection data's `+0x408` for 51 of 63 weapons and
    //! not one `projectile` tag reference inside it at any candidate stride, so one of the two
    //! numbers is wrong. This locates both from the bytes: a Reach tag reference stores its class
    //! fourcc BIG-ENDIAN at +0x00, so a reference to a `proj` reads as the four bytes `jorp` and
    //! an `effe` as `effe` reversed. Scanning the pointed-to block data for them says exactly
    //! where the field is, and the distance between two consecutive hits says the stride.

    use super::*;

    /// The barrel element's own field offsets and its STRIDE, measured from the bytes.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn locate_the_barrel_fields() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let mut stride_votes: std::collections::BTreeMap<usize, usize> = Default::default();
        let mut proj_off_votes: std::collections::BTreeMap<usize, usize> = Default::default();
        let mut fx_off_votes: std::collections::BTreeMap<usize, usize> = Default::default();
        let mut marker_off_votes: std::collections::BTreeMap<usize, usize> = Default::default();
        let mut multi = Vec::new();
        for &w in &native.tags_by_group(cache, "weap") {
            let Some(h) = native.tag_meta(cache, w, WEAP_BARRELS as u32, 8) else { continue };
            let n = i32::from_le_bytes([h[0], h[1], h[2], h[3]]);
            let ptr = rd_u32(&h, 4);
            if n <= 0 || n > 2 || ptr == 0 || ptr == u32::MAX { continue; }
            let want = n as usize * 1024;
            let Some(b) = native.tag_ptr_bytes(cache, ptr, want as u32) else { continue };
            let name = native.tag_name(cache, w).unwrap_or_default();
            let leaf = name.rsplit('\\').next().unwrap_or("").to_string();
            // Every `proj` tag reference in the block data, in order.
            let projs: Vec<usize> = (0..b.len().saturating_sub(16))
                .filter(|&i| b[i..i + 4] == *b"jorp" && native.tag_class(cache, rd_u32(&b, i + 0x0C) & 0xFFFF).as_deref() == Some("proj"))
                .collect();
            for &o in &projs { *proj_off_votes.entry(o).or_default() += 1; }
            if n == 2 && projs.len() >= 2 {
                let st = projs[1] - projs[0];
                *stride_votes.entry(st).or_default() += 1;
                multi.push(format!("  {leaf}: 2 barrels, proj refs at {:?} -> stride {st}", projs));
            }
            // `firing effects`: a (count, ptr) whose data holds an `effe` reference at +0x04.
            for off in (0..512).step_by(4) {
                if off + 8 > b.len() { break; }
                let c = i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]]);
                let p2 = rd_u32(&b, off + 4);
                if c <= 0 || c > FIRING_EFFECT_MAX || p2 == 0 || p2 == u32::MAX { continue; }
                let Some(fb) = native.tag_ptr_bytes(cache, p2, 0x10) else { continue };
                if fb[4..8] == *b"effe" && native.tag_class(cache, rd_u32(&fb, 4 + 0x0C) & 0xFFFF).as_deref() == Some("effe") {
                    *fx_off_votes.entry(off).or_default() += 1;
                }
            }
            // `optional barrel marker name`: a u32 that the cache resolves to a non-empty string.
            for off in (0..512).step_by(4) {
                if off + 4 > b.len() { break; }
                let v = rd_u32(&b, off);
                if v == 0 || v == u32::MAX { continue; }
                if let Some(s) = native.string_id(cache, v) {
                    if !s.is_empty() && s.contains("barrel") || s == "primary" {
                        *marker_off_votes.entry(off).or_default() += 1;
                    }
                }
            }
        }
        println!("== barrel STRIDE, from weapons with two barrels ==");
        for l in &multi { println!("{l}"); }
        println!("  votes: {stride_votes:?}");
        println!("== `projectile` offsets seen (offset -> weapons) ==\n  {proj_off_votes:?}");
        println!("== `firing effects` block descriptor offsets ==\n  {fx_off_votes:?}");
        println!("== marker-name string-id offsets (names containing 'barrel') ==\n  {marker_off_votes:?}");
    }

    /// One barrel element, every field interpreted every plausible way, so the remaining offsets
    /// can be read off the authored values instead of guessed.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn dump_one_barrel() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        const STRIDE: usize = 388;
        for leaf in ["assault_rifle", "shotgun", "rocket_launcher", "plasma_pistol"] {
            let Some(w) = native.tags_by_group(cache, "weap").into_iter().find(|&t| {
                native.tag_name(cache, t).map_or(false, |n| n.ends_with(leaf))
            }) else { continue };
            let Some(h) = native.tag_meta(cache, w, WEAP_BARRELS as u32, 8) else { continue };
            let n = i32::from_le_bytes([h[0], h[1], h[2], h[3]]).max(0) as usize;
            let ptr = rd_u32(&h, 4);
            let Some(b) = native.tag_ptr_bytes(cache, ptr, (n * STRIDE) as u32) else { continue };
            println!("\n===== {leaf}: {n} barrel(s), {} bytes =====", b.len());
            let e = &b[..STRIDE.min(b.len())];
            for off in (0..e.len()).step_by(4) {
                let u = rd_u32(e, off);
                let f = rd_f32(e, off);
                let s16 = (rd_i16(e, off), rd_i16(e, off + 2));
                let mut notes: Vec<String> = Vec::new();
                if u != 0 && u != u32::MAX {
                    if let Some(st) = native.string_id(cache, u) {
                        if !st.is_empty() { notes.push(format!("sid '{st}'")); }
                    }
                }
                if off + 16 <= e.len() {
                    let fourcc: String = e[off..off + 4].iter().rev().map(|&c| c as char).collect();
                    if fourcc.chars().all(|c| c.is_ascii_graphic()) {
                        let d = rd_u32(e, off + 0x0C);
                        if d != 0 && d != u32::MAX {
                            if let Some(cl) = native.tag_class(cache, d & 0xFFFF) {
                                if cl == fourcc {
                                    notes.push(format!(
                                        "TAGREF {fourcc} -> {}",
                                        native.tag_name(cache, d & 0xFFFF).map(|p| p.rsplit('\\').next().unwrap_or("").to_string()).unwrap_or_default()
                                    ));
                                }
                            }
                        }
                    }
                }
                if off + 8 <= e.len() {
                    let c = i32::from_le_bytes([e[off], e[off + 1], e[off + 2], e[off + 3]]);
                    let p2 = rd_u32(e, off + 4);
                    if c > 0 && c <= 8 && p2 > 0x1000 && p2 != u32::MAX {
                        if let Some(sub) = native.tag_ptr_bytes(cache, p2, 0x20) {
                            let fc: String = sub[4..8].iter().rev().map(|&c| c as char).collect();
                            notes.push(format!("BLOCK? count {c} ptr {p2:#x} sub[4..8]='{fc}' sub[0..4]={:?}", &sub[..4]));
                        }
                    }
                }
                println!("  +{off:#05x} u={u:#010x} f={f:>12.4} i16={:?} {}", s16, notes.join(" | "));
            }
        }
    }

    /// WHERE a firing effect is located, and what the barrel element really holds.
    ///
    /// The `hms-fx-marker-fallback` question: a firing effect must be placed on a MARKER, and
    /// this says which one — the barrel's own `optional barrel marker name` (authored on only a
    /// handful of weapons), or the markers the firing `effe` itself names in its
    /// `effect_locations` block. It prints both beside the weapon model's actual marker list, so
    /// a missing marker is visible rather than silently replaced by the weapon origin.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn locate_the_muzzle() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let mut rows = 0usize;
        let mut barrel_marker = 0usize;
        let mut effe_named = 0usize;
        let mut effe_marker_missing = 0usize;
        let mut no_effe = 0usize;
        for &w in &native.tags_by_group(cache, "weap") {
            let bs = weapon_barrels(native, cache, w);
            if bs.is_empty() { continue; }
            let wname = native.tag_name(cache, w).unwrap_or_default();
            let leaf = wname.rsplit('\\').next().unwrap_or("").to_string();
            let mode = native.resolve_mode(cache, w);
            let markers: Vec<String> = crate::reach_actor::ReachModel::read(native, cache, mode)
                .map(|m| m.marker_names(native, cache))
                .unwrap_or_default();
            println!("\n{leaf}  model {mode:#x} markers {markers:?}");
            for (i, b) in bs.iter().enumerate() {
                rows += 1;
                let bm = if b.marker_sid == 0 {
                    "<none authored>".to_string()
                } else {
                    barrel_marker += 1;
                    native.string_id(cache, b.marker_sid).unwrap_or_else(|| format!("sid {:#x}", b.marker_sid))
                };
                let (ename, locs) = match b.firing_effect {
                    None => { no_effe += 1; ("-".to_string(), Vec::new()) }
                    Some(e) => {
                        let name = native.tag_name(cache, e).unwrap_or_default();
                        (name.rsplit('\\').next().unwrap_or("").to_string(), effect_location_names(native, cache, e))
                    }
                };
                if !locs.is_empty() {
                    effe_named += 1;
                    for l in &locs {
                        if !markers.iter().any(|m| m == l) { effe_marker_missing += 1; }
                    }
                }
                println!(
                    "  barrel {i}: rps {:.1}..{:.1} | barrel marker {bm} | proj {:?} | effe '{ename}' locations {locs:?}",
                    b.rounds_per_second[0], b.rounds_per_second[1],
                    b.projectile.and_then(|t| native.tag_name(cache, t)).map(|p| p.rsplit('\\').next().unwrap_or("").to_string()),
                );
            }
        }
        println!(
            "\n{rows} barrels: {barrel_marker} author a barrel marker, {no_effe} have no firing effect, \
             {effe_named} have a firing effect that names at least one location, \
             {effe_marker_missing} named locations are NOT markers of the weapon's own model"
        );
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn locate_the_barrels_block() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let weaps = native.tags_by_group(cache, "weap");
        // Three weapons with very different barrels: one bullet hose, one shotgun (8 projectiles
        // per shot), one grenade launcher.
        let pick: Vec<u32> = weaps
            .iter()
            .copied()
            .filter(|&t| {
                let n = native.tag_name(cache, t).unwrap_or_default();
                n.ends_with("assault_rifle") || n.ends_with("shotgun") || n.ends_with("rocket_launcher") || n.ends_with("sniper_rifle")
            })
            .collect();
        println!("== every block descriptor in the weapon tag, and what its data contains ==");
        for &w in &pick {
            let name = native.tag_name(cache, w).unwrap_or_default();
            println!("\n{name}");
            let Some(meta) = native.tag_meta(cache, w, 0, 0x4B8) else {
                println!("  cannot read 0x4B8 bytes of meta");
                continue;
            };
            for off in (0x3C0..0x4B0).step_by(4) {
                let n = i32::from_le_bytes([meta[off], meta[off + 1], meta[off + 2], meta[off + 3]]);
                let ptr = rd_u32(&meta, off + 4);
                if n <= 0 || n > 8 || ptr == 0 || ptr == u32::MAX {
                    continue;
                }
                // Try a generous slab and look for the fourcc fingerprints.
                let Some(b) = native.tag_ptr_bytes(cache, ptr, 4096) else {
                    println!("  +{off:#05x}: count {n} ptr {ptr:#x} -- unreadable");
                    continue;
                };
                let find = |tag: &[u8; 4]| -> Vec<usize> {
                    let rev = [tag[3], tag[2], tag[1], tag[0]];
                    (0..b.len().saturating_sub(4)).filter(|&i| b[i..i + 4] == rev).collect()
                };
                let projs = find(b"proj");
                let effes = find(b"effe");
                let jpts = find(b"jpt!");
                if projs.is_empty() && effes.is_empty() && jpts.is_empty() {
                    println!("  +{off:#05x}: count {n} ptr {ptr:#x} -- no proj/effe/jpt! reference in the first 4 KiB");
                    continue;
                }
                println!(
                    "  +{off:#05x}: count {n} ptr {ptr:#x} | proj refs at {:?} | effe at {:?} | jpt! at {:?}",
                    &projs[..projs.len().min(6)],
                    &effes[..effes.len().min(8)],
                    &jpts[..jpts.len().min(8)]
                );
                for &pi in projs.iter().take(3) {
                    let d = rd_u32(&b, pi + 0x0C);
                    let t = d & 0xFFFF;
                    println!(
                        "        proj@{pi}: datum {d:#x} -> {:?} class {:?}",
                        native.tag_name(cache, t).map(|s| s.rsplit('\\').next().unwrap_or("").to_string()),
                        native.tag_class(cache, t)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod proj_probes {
    //! #projectile  What each `proj` in the map plays: its attachments (the travelling trail),
    //! its detonation effects and its impact effects — and whether the markers they name are
    //! markers the projectile's own model has.
    //!
    //! `cargo test -p hms-app --profile quick dump_projectile_effects -- --ignored --nocapture`

    use super::*;

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn dump_projectile_effects() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let nm = |t: Option<u32>| {
            t.and_then(|t| native.tag_name(cache, t))
                .map(|p| p.rsplit('\\').next().unwrap_or("").to_string())
                .unwrap_or_else(|| "-".into())
        };
        let mut with_attach = 0usize;
        let mut with_det = 0usize;
        let mut with_impact = 0usize;
        let projs = native.tags_by_group(cache, "proj");
        for &p in &projs {
            let Some(d) = projectile_def(native, cache, p) else {
                println!("{:#x} UNREADABLE to {PROJ_SIZE:#x}", p);
                continue;
            };
            let atts = object_attachment_effects(native, cache, p);
            let mode = native.resolve_mode(cache, p);
            let markers: Vec<String> = crate::reach_actor::ReachModel::read(native, cache, mode)
                .map(|m| m.marker_names(native, cache))
                .unwrap_or_default();
            if !atts.is_empty() { with_attach += 1; }
            if d.detonation_ground.is_some() || d.detonation_airborne.is_some() { with_det += 1; }
            if d.impact_effect.is_some() { with_impact += 1; }
            println!(
                "\n{} (mode {mode:#x}, markers {:?})\n  v0 {:.1} -> {:.1} over {:?} | gravityS {:.3} | timer {:?} | maxrange {:.1} | radius {:.2}",
                nm(Some(p)), markers,
                d.initial_velocity, d.final_velocity, d.acceleration_range, d.air_gravity_scale, d.timer, d.maximum_range, d.collision_radius
            );
            println!(
                "  detonation ground {} / airborne {} | impact {} / object impact {}",
                nm(d.detonation_ground), nm(d.detonation_airborne), nm(d.impact_effect), nm(d.object_impact_effect)
            );
            for (e, marker, scale) in &atts {
                println!(
                    "  attachment effe {} marker '{}' scale '{}' | effect locations {:?}",
                    nm(Some(*e)),
                    native.string_id(cache, *marker).unwrap_or_default(),
                    native.string_id(cache, *scale).unwrap_or_default(),
                    effect_location_names(native, cache, *e),
                );
            }
        }
        println!(
            "\n{} proj tags: {with_attach} with an attachment effect, {with_det} with a detonation effect, {with_impact} with an impact effect",
            projs.len()
        );
    }
}

#[cfg(test)]
mod gates {
    //! #projectile  THE LAYOUT GATES. Both `weapon_barrels` and `projectile` are field sets whose
    //! field array is installed at runtime, so neither of their layouts could be read out of
    //! `reach_tag_test.exe`'s reflection data the way the other 1,597 were. The element size and
    //! the field offsets above were MEASURED against a real cache, and these are the measurements.
    //!
    //! Each gate also states what it ruled out, and proves it can fail: the same resolution is run
    //! at every other candidate stride and at the legacy projectile offset, and every one of them
    //! must come out wrong. A gate that only checks the answer it was given cannot tell a correct
    //! constant from a lucky one.
    //!
    //! `cargo test -p hms-app --profile quick the_barrel -- --ignored --nocapture`

    use super::*;

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    /// What a cache's barrels look like when read at `stride`, through the production reader's own
    /// field offsets.
    ///
    /// Returns `(first-barrel projectiles ok, wrong class, SECOND-barrel projectiles ok, wrong
    /// class, firing effects ok, wrong class)`.
    ///
    /// **A layout gate that only reads element 0 cannot test a stride at all** -- the second
    /// barrel is the only thing that discriminates it, and that is a measured correction to the
    /// first draft of this gate, which passed at the wrong stride.** `projectile` sits at +0x104, which
    /// is inside every candidate element, so barrel 0 reads the same bytes at 304 as at 388 and
    /// resolves for 47 weapons either way — the first draft asserted on that number and had to be
    /// rewritten. Barrel 1 starts at `stride`, so a wrong stride lands it on the wrong bytes;
    /// `forge_halo` has five two-barrel weapons to say so.
    fn score(native: &NativeDll, cache: CacheHandle, stride: usize) -> (usize, usize, usize, usize, usize, usize) {
        let mut n = [0usize; 6];
        for &w in &native.tags_by_group(cache, "weap") {
            let b = tag_block(native, cache, w, WEAP_BARRELS, stride, WEAP_BARREL_MAX);
            for i in 0..b.len() / stride {
                let e = &b[i * stride..(i + 1) * stride];
                let base = if i == 0 { 0 } else { 2 };
                if let Some(t) = tag_ref(e, BARREL_PROJECTILE) {
                    if native.tag_class(cache, t).as_deref() == Some("proj") { n[base] += 1 } else { n[base + 1] += 1 }
                }
                let fx = block_at(native, cache, e, BARREL_FIRING_EFFECTS, FIRING_EFFECT_SIZE, FIRING_EFFECT_MAX);
                if let Some(t) = (!fx.is_empty()).then(|| tag_ref(&fx, FIRING_EFFECT_EFFECT)).flatten() {
                    if native.tag_class(cache, t).as_deref() == Some("effe") { n[4] += 1 } else { n[5] += 1 }
                }
            }
        }
        (n[0], n[1], n[2], n[3], n[4], n[5])
    }

    /// **THE BARREL GATE.** At the measured stride every barrel's `projectile` reference resolves
    /// to a `proj` and every `firing effect^` to an `effe`; at every other candidate stride the
    /// same read finds neither. That second half is the proof the gate can fail — the four
    /// rejected strides are the two layouts the reflection data offers (252 and 312) and those
    /// two plus the block definition's own 52-byte inline extension (304 and 364), i.e. every
    /// answer the static data could have given.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_barrel_stride_and_its_projectile_reference_are_the_measured_ones() {
        let Some((_g, native, cache)) = open() else {
            eprintln!("reach_projectile gates SKIPPED: no forge_halo cache");
            return;
        };
        let native = &*native;
        let (ok, bad, ok2, bad2, fx, fxb) = score(native, cache, BARREL_SIZE);
        println!("stride {BARREL_SIZE}: barrel0 proj {ok} ok / {bad} wrong; barrel1 proj {ok2} ok / {bad2} wrong; firing effect {fx} ok / {fxb} wrong");
        assert!(ok >= 40, "only {ok} first barrels resolve a projectile at stride {BARREL_SIZE}; forge_halo authors 47");
        assert_eq!(bad, 0, "{bad} first-barrel 'projectile' references resolve to something that is not a proj");
        assert!(ok2 >= 5, "only {ok2} SECOND barrels resolve a projectile; forge_halo has 5 two-barrel weapons");
        assert_eq!(bad2, 0, "{bad2} second-barrel 'projectile' references resolve to something that is not a proj");
        assert!(fx >= 40, "only {fx} barrels resolve a firing effect at stride {BARREL_SIZE}");
        assert_eq!(fxb, 0, "{fxb} 'firing effect' references resolve to something that is not an effe");

        // ...and nothing else works. The four rejected candidates are the two layouts the
        // reflection data offers (252 and 312) and those two plus the block definition's own
        // 52-byte inline extension (304 and 364) -- every answer the static data could have
        // given. Each must fail on BOTH of the two things the stride actually decides: where the
        // second element begins, and where the `firing effects` descriptor at +0x178 lands.
        for &alt in &[252usize, 304, 312, 364] {
            let (a0, _, a2, _, afx, _) = score(native, cache, alt);
            println!("  rejected stride {alt}: barrel0 proj {a0} ok; barrel1 proj {a2} ok; firing effect {afx} ok");
            assert_eq!(
                a2, 0,
                "stride {alt} also resolves {a2} SECOND-barrel projectiles -- the gate does not discriminate \
                 and the measured {BARREL_SIZE} is not established"
            );
            assert_eq!(
                afx, 0,
                "stride {alt} also resolves {afx} firing effects -- the gate does not discriminate"
            );
        }
    }

    /// **THE PROJECTILE LAYOUT GATE.** Four shipped weapons whose ballistics are a matter of
    /// public record, read through the production decoder; and the legacy offset the same
    /// reflection data also offers, which must read zero on every one of them.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_projectile_flight_law_is_read_at_the_reach_offsets() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let projs = native.tags_by_group(cache, "proj");
        assert!(!projs.is_empty(), "forge_halo has proj tags");
        let find = |leaf: &str| -> Option<u32> {
            projs.iter().copied().find(|&t| {
                native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf))
            })
        };
        // Every proj must read to its full authored length -- which is simultaneously the test
        // that the tag really is 0x360 bytes and not the legacy 0x258.
        for &p in &projs {
            assert!(
                projectile_def(native, cache, p).is_some(),
                "proj {:#x} '{}' is not readable to {PROJ_SIZE:#x}",
                p,
                native.tag_name(cache, p).unwrap_or_default()
            );
        }
        // The authored numbers. A drift of even one field would move all of these.
        let cases: [(&str, f32, f32, f32); 4] = [
            // leaf, initial velocity, final velocity, air gravity scale
            ("sniper_rifle_bullet", 6000.0, 6000.0, 0.0),
            ("dmr_bullet", 3000.0, 3000.0, 0.0),
            ("rocket_launcher_rocket", 8.0, 16.0, 0.02),
            ("needler_shard", 11.0, 11.0, 0.0),
        ];
        for (leaf, v0, v1, g) in cases {
            let Some(p) = find(leaf) else { continue };
            let d = projectile_def(native, cache, p).expect("readable");
            assert!((d.initial_velocity - v0).abs() < 0.01, "{leaf} initial velocity {} != {v0}", d.initial_velocity);
            assert!((d.final_velocity - v1).abs() < 0.01, "{leaf} final velocity {} != {v1}", d.final_velocity);
            assert!((d.air_gravity_scale - g).abs() < 1e-4, "{leaf} air gravity scale {} != {g}", d.air_gravity_scale);
            // The legacy layout would have put `initial velocity` 0x100 earlier. If it reads
            // anything there the two layouts are not distinguishable and the gate is empty.
            let b = native.tag_meta(cache, p, 0, PROJ_SIZE as u32).expect("readable");
            let legacy = f32::from_le_bytes([
                b[PROJ_INITIAL_VELOCITY_LEGACY],
                b[PROJ_INITIAL_VELOCITY_LEGACY + 1],
                b[PROJ_INITIAL_VELOCITY_LEGACY + 2],
                b[PROJ_INITIAL_VELOCITY_LEGACY + 3],
            ]);
            assert!(legacy.abs() < 1e-6, "{leaf} also reads {legacy} at the LEGACY offset; the two layouts are not distinguished");
        }
        // A grenade is launched by the thrower, not by a barrel, so its own initial velocity is
        // zero -- and it is the gravity scale and the fuse that make it a grenade. Both are read
        // from fields 0x100 apart from the ones above, so this is an independent check of the
        // same layout.
        if let Some(p) = find("frag_grenade") {
            let d = projectile_def(native, cache, p).expect("readable");
            assert!((d.air_gravity_scale - 0.9).abs() < 1e-4, "frag gravity scale {}", d.air_gravity_scale);
            assert!(d.timer[0] > 0.5 && d.timer[1] < 1.0, "frag fuse {:?} is not a fuse", d.timer);
            assert!(d.detonation_ground.is_some(), "a frag grenade authors a ground detonation effect");
            assert_eq!(
                native.tag_class(cache, d.detonation_ground.unwrap()).as_deref(),
                Some("effe"),
                "the frag's detonation reference is an effe"
            );
        }
    }

    /// **THE MUZZLE GATE.** A firing effect has to be placed on a marker, and the question
    /// `hms-fx-marker-fallback` exists for is which one. The answer measured here: the barrel's
    /// own `optional barrel marker name` is authored by a small minority of barrels, and the
    /// firing EFFECT names its own locations — so the effect, not the barrel, carries the muzzle,
    /// and `primary_trigger` is a real marker on the weapon's own model.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_firing_effect_names_its_own_muzzle_marker_on_the_weapons_own_model() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let mut checked = 0usize;
        let mut barrel_markers = 0usize;
        for &w in &native.tags_by_group(cache, "weap") {
            let bs = weapon_barrels(native, cache, w);
            let Some(b) = bs.first() else { continue };
            if b.marker_sid != 0 {
                barrel_markers += 1;
            }
            let Some(effe) = b.firing_effect else { continue };
            let locs = effect_location_names(native, cache, effe);
            if locs.is_empty() {
                continue;
            }
            let mode = native.resolve_mode(cache, w);
            let markers = match crate::reach_actor::ReachModel::read(native, cache, mode) {
                Some(m) => m.marker_names(native, cache),
                None => continue,
            };
            if markers.is_empty() {
                continue;
            }
            checked += 1;
            let name = native.tag_name(cache, w).unwrap_or_default();
            assert!(
                locs.iter().any(|l| markers.contains(l)),
                "{name}: its firing effect names {locs:?} and its model has {markers:?} -- not one of the \
                 effect's locations is a marker of the gun, so the flash has nowhere real to go"
            );
            assert!(
                markers.iter().any(|m| m == "primary_trigger"),
                "{name}: the weapon model has no `primary_trigger` marker; markers {markers:?}"
            );
        }
        assert!(checked >= 20, "only {checked} weapons were checkable");
        println!("{checked} weapons checked; {barrel_markers} barrels author their own marker name");
        assert!(
            barrel_markers < checked,
            "every barrel authors `optional barrel marker name` after all -- then the marker should \
             come from the barrel and not from the effect, and this lane is built on the wrong field"
        );
    }
}
