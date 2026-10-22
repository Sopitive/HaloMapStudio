//! The TAG side of Reach's audio: `$#!+` and `ugh!` read out of a cache.
//!
//! `hms-sound` owns the FMOD banks and can turn a *subsound id* into audio.
//! This module is the other half — the thing that says *which* subsound a
//! weapon, an ambience emitter or a film event should play, and at what gain.
//!
//! ```text
//!   weap / ssce / bipd ... --(tag reference)--> snd!  or  lsnd
//!   snd! in a cache  =  $#!+  cache_file_sound                        (52 B)
//!         +0x02 sound class            which of the 20+ classes
//!         +0x03 pitch range count
//!         +0x06 first pitch range index
//!         +0x0E playback index         -> ugh!.playbacks   (THE GAIN LAW)
//!         +0x10 scale index            -> ugh!.scales
//!         +0x30 fmod bank suffix       string id: which .fsb
//!      -> ugh!  sound_cache_file_gestalt                             (224 B)
//!         +0x10 playbacks    84 B  sound_playback_parameters_struct
//!         +0x1C scales       20 B
//!         +0x28 import names  4 B  string ids
//!         +0x4C pitch ranges 12 B
//!         +0x58 permutations 24 B  +0x14 = fmod bank subsound id hash
//! ```
//!
//! Offsets come from `reach_tag_test.exe`'s own reflection data
//! (`reverseme tagdefs reach_tag_test.exe`), not from guesswork; the struct
//! sizes below are the compiled-in element sizes and every reader checks the
//! block stride against them. `re-notes/reach_fmod_sound.md` §3 records the
//! measurement that proved the subsound-id join.
//!
//! **The gain law is the engine's, not ours.** `sound_playback_parameters_struct`
//! carries the attenuation distances, the gain base and variance, the cone
//! angles and the cone gain; `Playback::gain_db` applies them. Nothing here
//! invents a `1/d` falloff.

use hms_native::{CacheHandle, NativeDll};

use crate::reach_actor::{block_at, tag_block};

// ---------------------------------------------------------------------------------------------
// `$#!+` — cache_file_sound_struct_definition, 52 bytes
// ---------------------------------------------------------------------------------------------

pub const SND_SIZE: usize = 52;
pub const SND_FLAGS: usize = 0x00;
pub const SND_CLASS: usize = 0x02;
pub const SND_PITCH_RANGE_COUNT: usize = 0x03;
pub const SND_CODEC_INDEX: usize = 0x04;
pub const SND_FIRST_PITCH_RANGE: usize = 0x06;
pub const SND_SUB_PRIORITY: usize = 0x0C;
pub const SND_PLAYBACK_INDEX: usize = 0x0E;
pub const SND_SCALE_INDEX: usize = 0x10;
pub const SND_PROMOTION_INDEX: usize = 0x12;
pub const SND_MAXIMUM_PLAY_TIME_MS: usize = 0x18;
pub const SND_FMOD_BANK_SUFFIX: usize = 0x30;

/// `sound_definition_flags` bit 1: "always spatialize — always play as 3d
/// sound, even in first person".
pub const SND_FLAG_ALWAYS_SPATIALIZE: u16 = 1 << 1;
/// bit 2: "never obstruct".
pub const SND_FLAG_NEVER_OBSTRUCT: u16 = 1 << 2;
/// bit 12: "play permutations in order".
pub const SND_FLAG_PERMUTATIONS_IN_ORDER: u16 = 1 << 12;

/// `sound_class_enum`, all **71** values in the order `reach_tag_test.exe`
/// lists them. The index is what `$#!+ +0x02` holds, and it is also the row a
/// sound takes in the `sncl` sound-class table.
pub const SOUND_CLASS_NAMES: [&str; 71] = [
    "projectile_impact",             // 0
    "projectile_detonation",         // 1
    "projectile_flyby",              // 2
    "projectile_detonation_lod",     // 3
    "weapon_fire",                   // 4
    "weapon_ready",                  // 5
    "weapon_reload",                 // 6
    "weapon_empty",                  // 7
    "weapon_charge",                 // 8
    "weapon_overheat",               // 9
    "weapon_idle",                   // 10
    "weapon_melee",                  // 11
    "weapon_animation",              // 12
    "object_impacts",                // 13
    "particle_impacts",              // 14
    "weapon_fire_lod",               // 15
    "water_transitions",             // 16
    "lowpass_effects",               // 17
    "unit_footsteps",                // 18
    "unit_dialog",                   // 19
    "unit_animation",                // 20
    "unit_unused!",                  // 21
    "vehicle_collision",             // 22
    "vehicle_engine",                // 23
    "vehicle_animation",             // 24
    "vehicle_engine_lod",            // 25
    "device_door",                   // 26
    "device_unused0!",               // 27
    "device_machinery",              // 28
    "device_stationary",             // 29
    "device_unused1!",               // 30
    "device_unused2!",               // 31
    "music",                         // 32
    "ambient_nature",                // 33
    "ambient_machinery",             // 34
    "ambient_stationary",            // 35
    "huge_ass",                      // 36
    "object_looping",                // 37
    "cinematic_music",               // 38
    "unknown_unused0!",              // 39
    "unknown_unused1!",              // 40
    "ambient_flock",                 // 41
    "no_pad",                        // 42
    "no_pad_stationary",             // 43
    "equipment_effect",              // 44
    "mission_dialog",                // 45
    "cinematic_dialog",              // 46
    "scripted_cinematic_foley",      // 47
    "game_event",                    // 48
    "ui",                            // 49
    "test",                          // 50
    "multiplayer_dialog",            // 51
    "ambient_nature_details",        // 52
    "ambient_machinery_details",     // 53
    "inside_surround_tail",          // 54
    "outside_surround_tail",         // 55
    "vehicle_detonation",            // 56
    "ambient_detonation",            // 57
    "first_person_inside",           // 58
    "first_person_outside",          // 59
    "first_person_anywhere",         // 60
    "space_projectile_detonation",   // 61
    "space_projectile_flyby",        // 62
    "space_vehicle_engine",          // 63
    "space_weapon_fire",             // 64
    "player_voice_team",             // 65
    "player_voice_proxy",            // 66
    "projectile_impact_postpone",    // 67
    "unit_footsteps_postpone",       // 68
    "weapon_ready_third_person",     // 69
    "ui_music",                      // 70
];

/// The classes whose sounds are not spatialised at all: music, UI and the
/// first-person bed. Named rather than guessed — these are the enum's own
/// names, and the `sncl` table's `suppress spatialization` flags agree.
pub fn class_is_unspatialized(c: u8) -> bool {
    matches!(
        sound_class_name(c),
        "music" | "cinematic_music" | "ui_music" | "ui" | "mission_dialog" | "cinematic_dialog"
            | "multiplayer_dialog" | "player_voice_team" | "player_voice_proxy"
            | "first_person_inside" | "first_person_outside" | "first_person_anywhere"
    )
}

/// The ambience classes an acoustics palette's background sound falls in.
pub fn class_is_ambience(c: u8) -> bool {
    sound_class_name(c).starts_with("ambient_")
        || matches!(sound_class_name(c), "inside_surround_tail" | "outside_surround_tail" | "huge_ass")
}

pub fn sound_class_name(c: u8) -> &'static str {
    SOUND_CLASS_NAMES.get(c as usize).copied().unwrap_or("?")
}

/// One `$#!+` tag, flattened.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CacheSound {
    pub tag: u32,
    pub flags: u16,
    pub class: u8,
    pub pitch_range_count: u8,
    pub first_pitch_range: u16,
    pub codec_index: u16,
    pub sub_priority: i16,
    pub playback_index: u16,
    pub scale_index: u16,
    pub promotion_index: u8,
    pub maximum_play_time_ms: u32,
    /// `fmod bank suffix` as a raw string id. Resolve it to pick the bank;
    /// an unset suffix means the world bank, `sfx`.
    pub bank_suffix_sid: u32,
}

/// Every sound tag in a cache.
///
/// The group code on disk is **`snd!`**; `$#!+` is only how a tag editor spells
/// `cache_file_sound`, the flattened 52-byte struct a cache holds in a `snd!`
/// tag's place. Both spellings are tried so a cache that uses either is read,
/// and the one that answers is the one used.
pub fn sound_tags(native: &NativeDll, cache: CacheHandle) -> Vec<u32> {
    let a = native.tags_by_group(cache, "snd!");
    if !a.is_empty() {
        return a;
    }
    native.tags_by_group(cache, "$#!+")
}

impl CacheSound {
    pub fn read(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<CacheSound> {
        if tag == 0 || tag == 0xFFFF_FFFF {
            return None;
        }
        let b = native.tag_meta(cache, tag, 0, SND_SIZE as u32)?;
        if b.len() < SND_SIZE {
            return None;
        }
        let u16a = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32a = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(CacheSound {
            tag,
            flags: u16a(SND_FLAGS),
            class: b[SND_CLASS],
            pitch_range_count: b[SND_PITCH_RANGE_COUNT],
            first_pitch_range: u16a(SND_FIRST_PITCH_RANGE),
            codec_index: u16a(SND_CODEC_INDEX),
            sub_priority: u16a(SND_SUB_PRIORITY) as i16,
            playback_index: u16a(SND_PLAYBACK_INDEX),
            scale_index: u16a(SND_SCALE_INDEX),
            promotion_index: b[SND_PROMOTION_INDEX],
            maximum_play_time_ms: u32a(SND_MAXIMUM_PLAY_TIME_MS),
            bank_suffix_sid: u32a(SND_FMOD_BANK_SUFFIX),
        })
    }

    pub fn class_name(&self) -> &'static str {
        sound_class_name(self.class)
    }

    pub fn always_spatialize(&self) -> bool {
        self.flags & SND_FLAG_ALWAYS_SPATIALIZE != 0
    }
}

// ---------------------------------------------------------------------------------------------
// `sound_distance_parameters_struct` — 32 bytes, all world units but the last
// ---------------------------------------------------------------------------------------------

pub const DIST_SIZE: usize = 32;

/// The engine's own attenuation distances, straight out of the tag.
///
/// The field comments are the tag definition's own, verbatim — they are the
/// engine's documentation of what each distance means:
///
/// ```text
///  dont_obstruct  "don't obstruct below this distance"
///  dont_play      "don't play below this distance"
///  attack         "start playing at full volume at this distance"
///  minimum        "start attenuating at this distance"
///  sustain_begin  "set attenuation to sustain db at this distance"
///  sustain_end    "continue attenuating to silence at this distance"
///  maximum        "the distance beyond which this sound is no longer audible"
///  sustain_db     "the amount of attenuation between sustain begin and end"
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Distances {
    pub dont_obstruct: f32,
    pub dont_play: f32,
    pub attack: f32,
    pub minimum: f32,
    pub sustain_begin: f32,
    pub sustain_end: f32,
    pub maximum: f32,
    pub sustain_db: f32,
}

impl Distances {
    pub fn parse(b: &[u8]) -> Option<Distances> {
        if b.len() < DIST_SIZE {
            return None;
        }
        let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(Distances {
            dont_obstruct: f(0x00),
            dont_play: f(0x04),
            attack: f(0x08),
            minimum: f(0x0C),
            sustain_begin: f(0x10),
            sustain_end: f(0x14),
            maximum: f(0x18),
            sustain_db: f(0x1C),
        })
    }

    /// Do these distances describe a usable curve?
    ///
    /// A gestalt row that is all zeroes (or has the knots that ARE set out of order) is
    /// not a curve, and silently treating it as one is how a fabricated
    /// falloff gets in by the back door. A caller that gets `false` must play
    /// the sound unattenuated or not at all, by its own rule.
    pub fn usable(&self) -> bool {
        let k = [
            self.dont_play,
            self.attack,
            self.minimum,
            self.sustain_begin,
            self.sustain_end,
            self.maximum,
        ];
        if !k.iter().all(|v| v.is_finite() && *v >= 0.0) || !self.sustain_db.is_finite() {
            return false;
        }
        if self.maximum <= 0.0 {
            return false;
        }
        // A knot at distance 0 is not a knot (see `curve`: `push` skips it), so a row that leaves
        // `sustain begin` / `sustain end` at zero is still the curve `(minimum, 1) -> (maximum, 0)`
        // and must not be called unusable for it: it is the SHAPE OF EVERY VEHICLE ENGINE LOOP
        // (`minimum 7, maximum 40`), which therefore played at full volume from any distance.
        // Only the knots that exist have to be in order.
        let set: Vec<f32> = k.into_iter().filter(|v| *v > 0.0).collect();
        set.windows(2).all(|w| w[1] >= w[0])
    }
}

// ---------------------------------------------------------------------------------------------
// `sound_playback_parameters_struct` — 84 bytes, one row per `snd!`
// ---------------------------------------------------------------------------------------------

pub const PLAYBACK_SIZE: usize = 84;
pub const PB_SKIP_FRACTION: usize = 0x04;
pub const PB_MAX_BEND_PER_SECOND: usize = 0x08;
pub const PB_DISTANCES: usize = 0x0C;
pub const PB_GAIN_BASE_DB: usize = 0x2C;
pub const PB_GAIN_VARIANCE_DB: usize = 0x30;
pub const PB_RANDOM_PITCH_BOUNDS: usize = 0x34;
pub const PB_INNER_CONE_DEG: usize = 0x38;
pub const PB_OUTER_CONE_DEG: usize = 0x3C;
pub const PB_OUTER_CONE_GAIN_DB: usize = 0x40;
pub const PB_OVERRIDE_FLAGS: usize = 0x44;
pub const PB_AZIMUTH: usize = 0x48;
pub const PB_POSITIONAL_GAIN_DB: usize = 0x4C;
pub const PB_FIRST_PERSON_GAIN_DB: usize = 0x50;

/// The engine's gain ceiling. `haloreach_debug.dll`'s own assert reads
/// `gain_db <= k_maximum_sound_gain_db`, and the constant it compares against
/// is the float `0.0` at VA 0x1813BECA0 — so **no sound in Reach is ever
/// played above unity gain**, and the sibling clamp pins the floor at -64 dB.
/// Recovered from `sub_1809730B0` / `sub_180973360`.
pub const MAX_GAIN_DB: f32 = 0.0;
/// The engine's audibility floor, from the same pair of clamps. A sound whose
/// summed dB is at or below this is not started at all.
pub const MIN_GAIN_DB: f32 = -64.0;

/// `sub_180973340`: the engine's dB to linear conversion, which is
/// `expf(db / 20.0f * 2.30258512f)` — i.e. exactly `10^(db/20)`.
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

/// `sub_180973360`: linear gain to dB, clamped the way the engine clamps it.
/// A gain of zero or less becomes the floor rather than `-inf`.
pub fn linear_to_db(gain: f32) -> f32 {
    if !(gain > 0.0) {
        return MIN_GAIN_DB;
    }
    (20.0 * gain.clamp(0.0, 1.0).log10()).clamp(MIN_GAIN_DB, MAX_GAIN_DB)
}

/// One `ugh!.playbacks` row: everything the engine needs to place one sound.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Playback {
    pub skip_fraction: f32,
    pub max_bend_per_second_cents: f32,
    pub dist: Distances,
    pub gain_base_db: f32,
    pub gain_variance_db: f32,
    pub random_pitch_cents: (i16, i16),
    pub inner_cone_deg: f32,
    pub outer_cone_deg: f32,
    pub outer_cone_gain_db: f32,
    pub override_flags: u32,
    pub azimuth_deg: f32,
    pub positional_gain_db: f32,
    pub first_person_gain_db: f32,
}

impl Playback {
    pub fn parse(b: &[u8]) -> Option<Playback> {
        if b.len() < PLAYBACK_SIZE {
            return None;
        }
        let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let i16a = |o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
        Some(Playback {
            skip_fraction: f(PB_SKIP_FRACTION),
            max_bend_per_second_cents: f(PB_MAX_BEND_PER_SECOND),
            dist: Distances::parse(&b[PB_DISTANCES..PB_DISTANCES + DIST_SIZE])?,
            gain_base_db: f(PB_GAIN_BASE_DB),
            gain_variance_db: f(PB_GAIN_VARIANCE_DB),
            random_pitch_cents: (i16a(PB_RANDOM_PITCH_BOUNDS), i16a(PB_RANDOM_PITCH_BOUNDS + 2)),
            // `angle` fields are stored in RADIANS in a cache even though the
            // editor labels them degrees, which is the convention every other
            // Reach angle field in this codebase follows.
            inner_cone_deg: f(PB_INNER_CONE_DEG).to_degrees(),
            outer_cone_deg: f(PB_OUTER_CONE_DEG).to_degrees(),
            outer_cone_gain_db: f(PB_OUTER_CONE_GAIN_DB),
            override_flags: u32::from_le_bytes([
                b[PB_OVERRIDE_FLAGS],
                b[PB_OVERRIDE_FLAGS + 1],
                b[PB_OVERRIDE_FLAGS + 2],
                b[PB_OVERRIDE_FLAGS + 3],
            ]),
            azimuth_deg: f(PB_AZIMUTH).to_degrees(),
            positional_gain_db: f(PB_POSITIONAL_GAIN_DB),
            first_person_gain_db: f(PB_FIRST_PERSON_GAIN_DB),
        })
    }

    /// Does this row look like real authored data rather than a misread?
    ///
    /// Deliberately NOT a check that the distances describe a curve: a row
    /// whose distances are all zero is a legitimately non-spatial sound, and
    /// over a quarter of `forge_halo`'s 391 playbacks are exactly that (the
    /// `maximum distance` 25th percentile is 0.000). What pins the offsets is
    /// that **`gain base` lands inside the engine's own clamp range** — the
    /// engine refuses anything outside [-64, 0] dB, and every row measured is
    /// inside it, which random bytes would not be.
    pub fn plausible(&self) -> bool {
        self.skip_fraction.is_finite()
            && (0.0..=1.0).contains(&self.skip_fraction)
            && self.gain_base_db.is_finite()
            && (MIN_GAIN_DB..=MAX_GAIN_DB).contains(&self.gain_base_db)
            && self.gain_variance_db.is_finite()
            && (-120.0..=0.0).contains(&self.gain_variance_db)
            && self.dist.dont_play.is_finite()
            && self.dist.maximum.is_finite()
            && self.dist.maximum >= 0.0
    }

    /// Is there a distance curve to apply at all? A row of zeroes means the
    /// sound is not attenuated by distance, which is a real engine state.
    pub fn spatialized(&self) -> bool {
        self.dist.usable()
    }
}

// ---------------------------------------------------------------------------------------------
// `ugh!` — sound_cache_file_gestalt_struct_definition, 224 bytes of block headers
// ---------------------------------------------------------------------------------------------

pub const UGH_CONTENT_TYPE: usize = 0x00;
pub const UGH_CODECS: usize = 0x04;
pub const UGH_PLAYBACKS: usize = 0x10;
pub const UGH_SCALES: usize = 0x1C;
pub const UGH_IMPORT_NAMES: usize = 0x28;
pub const UGH_PITCH_RANGE_DISTANCES: usize = 0x34;
pub const UGH_PITCH_RANGE_PARAMETERS: usize = 0x40;
pub const UGH_PITCH_RANGES: usize = 0x4C;
pub const UGH_PERMUTATIONS: usize = 0x58;

pub const SCALE_SIZE: usize = 20;
pub const PITCH_PARAMS_SIZE: usize = 16;
pub const PITCH_RANGE_SIZE: usize = 12;
pub const PERMUTATION_SIZE: usize = 24;
pub const IMPORT_NAME_SIZE: usize = 4;

/// `sound_gestalt_pitch_ranges_block` field offsets.
pub const PR_NAME: usize = 0x00;
pub const PR_PARAMETERS: usize = 0x02;
pub const PR_ENCODED_PERMUTATION_DATA: usize = 0x04;
pub const PR_FIRST_RUNTIME_FLAG: usize = 0x06;
pub const PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS: usize = 0x08;

/// `sound_gestalt_permutations_block` field offsets.
pub const PERM_NAME: usize = 0x00;
pub const PERM_ENCODED_SKIP_FRACTION: usize = 0x02;
pub const PERM_SAMPLE_COUNT: usize = 0x04;
pub const PERM_FIRST_CHUNK: usize = 0x08;
pub const PERM_CHUNK_COUNT: usize = 0x0C;
pub const PERM_ENCODED_GAIN_DB: usize = 0x0E;
pub const PERM_INFO_INDEX: usize = 0x0F;
pub const PERM_FMOD_SUBSOUND_HASH: usize = 0x14;

/// One permutation: a single take of a sound.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Permutation {
    pub index: usize,
    /// Index into `ugh!.import names`, which holds string ids.
    pub name_index: u16,
    pub sample_count: u32,
    /// A small authored trim, in whole dB. Measured over all 9,497
    /// permutations of `forge_halo.map`: zero for 9,412 of them, the rest
    /// between -12 and -2. A plain signed dB integer, not a scaled encoding.
    pub gain_db: i8,
    /// The join key into an `.fsb.info` sidecar: equals its `+0x00 subsound
    /// id`, and the record index is the bank's sample index.
    pub fmod_subsound_id: u32,
}

/// One pitch range: a set of interchangeable permutations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PitchRange {
    pub index: usize,
    pub name_index: u16,
    pub parameters_index: u16,
    pub first_permutation: usize,
    pub permutation_count: usize,
}

/// `sound_gestalt_pitch_range_parameters_block`, 16 bytes (all cents). Debug: natural pitch
/// `sub_1809B4B40` reads `+0x00`; the bend bounds `sub_1809B4DA0` are `+0x04/+0x06`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PitchParams {
    pub natural_pitch: i16,
    pub bend_bounds: (i16, i16),
    pub max_gain_bounds: (i16, i16),
    pub playback_bounds: (i16, i16),
}

/// A whole cache's `ugh!`, read once.
#[derive(Clone, Debug, Default)]
pub struct Gestalt {
    pub tag: u32,
    pub playbacks: Vec<Playback>,
    /// `sound_gestalt_scale_block`: gain modifier dB (lower, upper), pitch
    /// modifier cents, skip fraction modifier.
    pub scale_gain_db: Vec<(f32, f32)>,
    /// `sound_gestalt_scale_block +0x08 pitch modifier [cents]` (lower, upper): the span a
    /// looping sound's SCALE (0..1) sweeps. Debug `0x1806538B0`:
    /// `lerp(lower, upper, scale) + pitch_offset`.
    pub scale_pitch_cents: Vec<(i16, i16)>,
    /// `ugh!.pitch range parameters` (16 B), indexed by `PitchRange::parameters_index`.
    pub pitch_params: Vec<PitchParams>,
    pub pitch_ranges: Vec<PitchRange>,
    pub permutations: Vec<Permutation>,
    /// `ugh!.import names`: one string id per entry.
    pub import_names: Vec<u32>,
    /// Rows the pitch-range walk had to reject, for the probes to report.
    pub rejected_pitch_ranges: usize,
}

/// A block count no real gestalt approaches; a bound on a misread header.
pub(crate) const MAX_BLOCK: i32 = 1 << 20;

impl Gestalt {
    /// Find and read the one `ugh!` in a cache.
    pub fn read(native: &NativeDll, cache: CacheHandle) -> Option<Gestalt> {
        let tag = *native.tags_by_group(cache, "ugh!").first()?;
        Gestalt::read_tag(native, cache, tag)
    }

    pub fn read_tag(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<Gestalt> {
        let pb = tag_block(native, cache, tag, UGH_PLAYBACKS, PLAYBACK_SIZE, MAX_BLOCK);
        let sc = tag_block(native, cache, tag, UGH_SCALES, SCALE_SIZE, MAX_BLOCK);
        let nm = tag_block(native, cache, tag, UGH_IMPORT_NAMES, IMPORT_NAME_SIZE, MAX_BLOCK);
        let pp = tag_block(native, cache, tag, UGH_PITCH_RANGE_PARAMETERS, PITCH_PARAMS_SIZE, MAX_BLOCK);
        let pr = tag_block(native, cache, tag, UGH_PITCH_RANGES, PITCH_RANGE_SIZE, MAX_BLOCK);
        let pm = tag_block(native, cache, tag, UGH_PERMUTATIONS, PERMUTATION_SIZE, MAX_BLOCK);
        if pm.is_empty() || pr.is_empty() {
            return None;
        }

        let playbacks: Vec<Playback> = pb
            .chunks_exact(PLAYBACK_SIZE)
            .map(|c| Playback::parse(c).unwrap_or_default())
            .collect();
        let scale_gain_db: Vec<(f32, f32)> = sc
            .chunks_exact(SCALE_SIZE)
            .map(|c| {
                let f = |o: usize| f32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
                (f(0x00), f(0x04))
            })
            .collect();
        let scale_pitch_cents: Vec<(i16, i16)> = sc
            .chunks_exact(SCALE_SIZE)
            .map(|c| (i16::from_le_bytes([c[0x08], c[0x09]]), i16::from_le_bytes([c[0x0A], c[0x0B]])))
            .collect();
        let pitch_params: Vec<PitchParams> = pp
            .chunks_exact(PITCH_PARAMS_SIZE)
            .map(|c| {
                let h = |o: usize| i16::from_le_bytes([c[o], c[o + 1]]);
                PitchParams {
                    natural_pitch: h(0x00),
                    bend_bounds: (h(0x04), h(0x06)),
                    max_gain_bounds: (h(0x08), h(0x0A)),
                    playback_bounds: (h(0x0C), h(0x0E)),
                }
            })
            .collect();
        let import_names: Vec<u32> = nm
            .chunks_exact(IMPORT_NAME_SIZE)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let permutations: Vec<Permutation> = pm
            .chunks_exact(PERMUTATION_SIZE)
            .enumerate()
            .map(|(index, c)| Permutation {
                index,
                name_index: u16::from_le_bytes([c[PERM_NAME], c[PERM_NAME + 1]]),
                sample_count: u32::from_le_bytes([
                    c[PERM_SAMPLE_COUNT],
                    c[PERM_SAMPLE_COUNT + 1],
                    c[PERM_SAMPLE_COUNT + 2],
                    c[PERM_SAMPLE_COUNT + 3],
                ]),
                gain_db: c[PERM_ENCODED_GAIN_DB] as i8,
                fmod_subsound_id: u32::from_le_bytes([
                    c[PERM_FMOD_SUBSOUND_HASH],
                    c[PERM_FMOD_SUBSOUND_HASH + 1],
                    c[PERM_FMOD_SUBSOUND_HASH + 2],
                    c[PERM_FMOD_SUBSOUND_HASH + 3],
                ]),
            })
            .collect();

        let total = permutations.len();
        let mut rejected = 0usize;
        let pitch_ranges: Vec<PitchRange> = pr
            .chunks_exact(PITCH_RANGE_SIZE)
            .enumerate()
            .map(|(index, c)| {
                let enc = u32::from_le_bytes([
                    c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS],
                    c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS + 1],
                    c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS + 2],
                    c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS + 3],
                ]);
                let (first, count) = decode_first_permutation_and_counts(enc);
                // A count of zero is VALID: it names no permutations. m45 has
                // six such rows, and rejecting them reported a decode failure
                // where there was none.
                let ok = first <= total && first + count <= total;
                if !ok {
                    rejected += 1;
                }
                PitchRange {
                    index,
                    name_index: u16::from_le_bytes([c[PR_NAME], c[PR_NAME + 1]]),
                    parameters_index: u16::from_le_bytes([c[PR_PARAMETERS], c[PR_PARAMETERS + 1]]),
                    first_permutation: if ok { first } else { 0 },
                    permutation_count: if ok { count } else { 0 },
                }
            })
            .collect();

        Some(Gestalt {
            tag,
            playbacks,
            scale_gain_db,
            scale_pitch_cents,
            pitch_params,
            pitch_ranges,
            permutations,
            import_names,
            rejected_pitch_ranges: rejected,
        })
    }

    /// Every permutation one `$#!+` can play, across all its pitch ranges.
    pub fn permutations_of(&self, snd: &CacheSound) -> Vec<&Permutation> {
        let mut out = Vec::new();
        let a = snd.first_pitch_range as usize;
        let b = a + snd.pitch_range_count as usize;
        for pr in self.pitch_ranges.get(a..b.min(self.pitch_ranges.len())).unwrap_or(&[]) {
            let s = pr.first_permutation;
            let e = s + pr.permutation_count;
            if let Some(p) = self.permutations.get(s..e) {
                out.extend(p.iter());
            }
        }
        out
    }

    /// The gain law row for a `$#!+`, if it has one.
    pub fn playback_of(&self, snd: &CacheSound) -> Option<&Playback> {
        self.playbacks.get(snd.playback_index as usize)
    }

    /// `ugh!.scales[i]`'s authored pitch modifier, in cents (lower, upper).
    pub fn scale_pitch_of(&self, snd: &CacheSound) -> (i16, i16) {
        self.scale_pitch_cents.get(snd.scale_index as usize).copied().unwrap_or((0, 0))
    }

    /// The pitch parameters of one pitch range.
    pub fn params_of(&self, pr: &PitchRange) -> Option<&PitchParams> {
        self.pitch_params.get(pr.parameters_index as usize)
    }

    /// `ugh!.scales[i]`'s authored gain modifier, in dB (lower, upper).
    pub fn scale_gain_of(&self, snd: &CacheSound) -> (f32, f32) {
        self.scale_gain_db
            .get(snd.scale_index as usize)
            .copied()
            .unwrap_or((0.0, 0.0))
    }
}

/// Decode `sound_gestalt_pitch_ranges_block +0x08 encoded first permutation
/// and counts`.
///
/// **The layout is measured, not assumed.** `probes::probe_pitch_range_encoding`
/// is the derivation: it tries every (shift, width) split of the u32 against
/// the one property the field must have — that the pitch ranges TILE the
/// permutation block, in order, with no gap and no overlap — and only one
/// split satisfies it on a real map. The split it found is the one below; the
/// probe is also a gate, so a wrong constant here goes red.
///
/// Returns `(first permutation index, permutation count)`.
///
/// Measured split: **first permutation = a SIGN-EXTENDED 20-bit field in
/// bits[0..20], count = bits[20..26]**. The engine's own form is
/// `(int32)(raw << 12) >> 12` for the first index, so a negative value means
/// "no permutations".
///
/// The widths were pinned by brute-forcing every (shift, width) split of the
/// u32 against the one property the field cannot fake — that the pitch ranges
/// **tile** the permutation block in order, with no gap and no overlap — over
/// every Reach cache on the machine at once. One map alone cannot settle it:
/// `forge_halo`'s 9,497 permutations fit a 14-bit `first` and never set bits
/// 14..19, `cex_ff_halo`'s 26,024 do not, and `m45` needs a count wider than
/// six bits. `probes::probe_pitch_range_encoding` is that derivation, and it
/// is also the gate — it asserts this function agrees with the split it finds,
/// on every word of every map.
pub fn decode_first_permutation_and_counts(enc: u32) -> (usize, usize) {
    let first = ((enc << 12) as i32) >> 12;
    if first < 0 {
        return (0, 0);
    }
    (first as usize, ((enc >> 20) & 0x3F) as usize)
}

// ---------------------------------------------------------------------------------------------
// THE ENGINE'S DISTANCE ATTENUATION — ported, not invented
// ---------------------------------------------------------------------------------------------
//
// Recovered from `haloreach.dll` (release) and `haloreach_debug.dll` (which
// carries the assert strings and source paths, so the identification is
// exact). Addresses, both builds at base 0x180000000:
//
// ```text
//   build the curve (fetch + merge + knots)   debug 0x1806583A0  release 0x18002DC54
//   push the six knots                        (inlined)          release 0x18002DB58
//   EVALUATE the curve                        debug 0x1809B64B0  release 0x1800CB9D0
//   class + playback parameter fetch          debug 0x1809B0340  release 0x1800B002C
//   merge non-zero fields 1..7                debug 0x1809B0000  release 0x1800AFFAC
//   cone * distance                           debug 0x180657F90  release 0x18002DD80
//   dB clamp to [-64, 0]                      debug 0x1809730B0  release 0x1800AF910
//   the speaker-position table                debug 0x1813CCF70  release 0x1809D8558
//   PAN                                       debug 0x1809B6920  release 0x1800CBA78
// ```
//
// There is **no `powf`, no `exp`, no `log` and no lookup table** anywhere in
// the distance law, and no dB-domain interpolation: every segment is linear in
// RECIPROCAL distance, in linear gain.

/// The runtime 32-byte distance struct, built in four layers.
///
/// `sound_source_get_distance_parameters` (debug `0x1809B0270`) copies the
/// **sound class's** parameters wholesale as the base, then merges the
/// playback's over them, then the pitch range's, then the source's — and the
/// merge (debug `0x1809B0000`) is **field by field, only where the incoming
/// value is non-zero**, over fields 1..7 only:
///
/// ```c
/// for (i = 1; i <= 7; ++i)
///     if (isnan(src[i]) || src[i] != 0.0f) dst[i] = src[i];
/// ```
///
/// That is why a zero in a playback row is not a bug and not "unattenuated":
/// it means **inherit the class's value for that field**. Over a quarter of
/// `forge_halo`'s 391 playbacks have `maximum distance` zero and every one of
/// them still attenuates, through its class.
///
/// **Field 0, `don't obstruct distance`, is deliberately excluded** — the loop
/// starts at 1 — so it is always the class's and can never be overridden. It
/// also has no reader anywhere in the engine, so it is ignored here.
pub fn merge_distances(base: Distances, over: Distances) -> Distances {
    let mut out = base;
    let pick = |dst: f32, src: f32| if src.is_nan() || src != 0.0 { src } else { dst };
    out.dont_play = pick(out.dont_play, over.dont_play);
    out.attack = pick(out.attack, over.attack);
    out.minimum = pick(out.minimum, over.minimum);
    out.sustain_begin = pick(out.sustain_begin, over.sustain_begin);
    out.sustain_end = pick(out.sustain_end, over.sustain_end);
    out.maximum = pick(out.maximum, over.maximum);
    out.sustain_db = pick(out.sustain_db, over.sustain_db);
    out
}

/// `static_stack<{float distance; float value;}, 6>` — the engine's own
/// container for the attenuation curve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Curve {
    pub knots: [(f32, f32); 6],
    pub count: usize,
}

impl Distances {
    /// Build the engine's attenuation curve (debug `0x1806583A0`).
    ///
    /// ```c
    /// S = 10^(sustain_db / 20);
    /// push(don't play,     0.0);   // push(d, v): if (d > 0) knots[count++] = {d, v}
    /// push(attack,         1.0);
    /// push(minimum,        1.0);
    /// push(sustain begin,  S  );
    /// push(sustain end,    S  );
    /// push(maximum,        0.0);
    /// ```
    ///
    /// **A knot whose distance is <= 0 is skipped entirely.** That single rule
    /// is the whole mechanism by which "0 means unauthored", and it produces
    /// the common shapes: with `don't play` and `attack` both zero the first
    /// knot is `(minimum, 1.0)`, so everything closer than `minimum` is flat
    /// at full volume; with `sustain begin`/`sustain end` also zero the curve
    /// is just `(minimum, 1) -> (maximum, 0)`.
    pub fn curve(&self) -> Curve {
        let s = db_to_linear(self.sustain_db);
        let mut c = Curve::default();
        let mut push = |d: f32, v: f32| {
            if d > 0.0 && c.count < c.knots.len() {
                c.knots[c.count] = (d, v);
                c.count += 1;
            }
        };
        push(self.dont_play, 0.0);
        push(self.attack, 1.0);
        push(self.minimum, 1.0);
        push(self.sustain_begin, s);
        push(self.sustain_end, s);
        push(self.maximum, 0.0);
        c
    }

    /// Linear distance gain at `d` world units. Shorthand for
    /// `self.curve().eval(d)`.
    pub fn gain_at(&self, d: f32) -> f32 {
        self.curve().eval(d)
    }
}

/// The engine's coincident-knot epsilon, `9.99999975e-05f`
/// (release `.rdata` 0x180A8A9B8).
const KNOT_EPSILON: f32 = 9.99999975e-05;

impl Curve {
    /// Evaluate the curve (debug `0x1809B64B0`, release `0x1800CB9D0`).
    ///
    /// ```c
    /// v2 = count ? knot[count-1].value : 1.0f;
    /// if (count <= 0) return v2;                 // NO KNOTS -> 1.0, audible everywhere
    /// for (i = 0;; ++i) {
    ///     if (knot[i].distance > d) break;       // first knot strictly beyond d
    ///     if (i + 1 >= count) return v2;         // past the last knot -> last value
    /// }
    /// if (i == 0) return knot[0].value;          // below the first knot
    /// dhi = knot[i].distance;   ghi = knot[i].value;
    /// dlo = knot[i-1].distance; glo = knot[i-1].value;
    /// if (fabsf(dhi - dlo) < 1e-4f) return ghi;
    /// return ghi + ((dhi - d) * dlo / ((dhi - dlo) * d)) * (glo - ghi);
    /// ```
    ///
    /// That weight is algebraically exactly
    /// `t = (1/d - 1/dhi) / (1/dlo - 1/dhi)`, so **every segment is linear in
    /// reciprocal distance** and the interpolation happens in linear gain.
    pub fn eval(&self, d: f32) -> f32 {
        let last = if self.count > 0 { self.knots[self.count - 1].1 } else { 1.0 };
        if self.count == 0 {
            return last;
        }
        let mut i = 0usize;
        loop {
            if self.knots[i].0 > d {
                break;
            }
            if i + 1 >= self.count {
                return last;
            }
            i += 1;
        }
        if i == 0 {
            return self.knots[0].1;
        }
        let (dhi, ghi) = self.knots[i];
        let (dlo, glo) = self.knots[i - 1];
        let w = dhi - dlo;
        if w.abs() < KNOT_EPSILON {
            return ghi;
        }
        // Guard the engine's own division: d is a measured distance and the
        // loop above means d >= dlo > 0, but a NaN position must not poison
        // the mixer.
        if !(d > 0.0) || !d.is_finite() {
            return ghi;
        }
        ghi + ((dhi - d) * dlo / (w * d)) * (glo - ghi)
    }
}

/// `pi / 65536` and `pi / 131072`, the engine's 16-bit angle scale for a
/// source's cone override (release `.rdata` 0x180A8A9A4).
pub const ANGLE16_SCALE: f32 = std::f32::consts::PI / 65536.0;

/// The directional cone factor (debug `0x180657F90`).
///
/// ```c
/// a    = acosf(dot(source_forward, -normalize(local_position)));
/// t    = clamp((a - inner) / (outer - inner), 0, 1);
/// cone = 1.0f + t * (10^(outer_cone_gain_db / 20) - 1.0f);
/// ```
///
/// Interpolated in LINEAR gain, not dB. `inner`/`outer` are radians.
///
/// In Reach's shipped data `outer cone gain` is **0.0 dB in all 16,248
/// playbacks of all 49 maps** and the override-location flags that would let a
/// source supply its own are never set, so this returns 1.0 for everything the
/// game actually plays. It is here because it is exact and cheap, not because
/// it is load-bearing.
pub fn cone_gain(inner_rad: f32, outer_rad: f32, outer_gain_db: f32, angle_rad: f32) -> f32 {
    if !(angle_rad.is_finite() && inner_rad.is_finite() && outer_rad.is_finite()) {
        return 1.0;
    }
    let span = outer_rad - inner_rad;
    let t = if span.abs() < KNOT_EPSILON {
        if angle_rad >= inner_rad {
            1.0
        } else {
            0.0
        }
    } else {
        ((angle_rad - inner_rad) / span).clamp(0.0, 1.0)
    };
    1.0 + t * (db_to_linear(outer_gain_db) - 1.0)
}

// ---------------------------------------------------------------------------------------------
// THE PAN LAW — Reach's own `sound_math.cpp`, ported
// ---------------------------------------------------------------------------------------------

/// `k_speaker_positions` — debug `0x1813CCF70`, release `0x1809D8558`,
/// five `real_point3d` and **byte-identical between the two builds**. In
/// Halo's basis: x forward, y left, z up. `0.7071068` is `0x3F3504F3`.
///
/// Only the first four are used: the shipped pan function's loop bound is
/// `table + 0x34`, four iterations, and in the debug build the five-speaker
/// wrapper has zero callers. **The centre speaker is not computed.**
pub const SPEAKER_POSITIONS: [[f32; 3]; 4] = [
    [0.707_106_8, 0.707_106_8, 0.0],   // front left
    [0.707_106_8, -0.707_106_8, 0.0],  // front right
    [-0.707_106_8, 0.707_106_8, 0.0],  // rear left
    [-0.707_106_8, -0.707_106_8, 0.0], // rear right
];

/// Reach's speaker pan (debug `0x1809B6920`, release `0x1800CBA78`).
///
/// ```c
/// for each speaker s:
///     q = (P - s) * 0.5;
///     out[i] = 1.0f - clamp(q.x*q.x + q.y*q.y + q.z*q.z, 0, 1);
/// ```
///
/// i.e. `gain_i = 1 - clamp(|P_hat - s_i|^2 / 4, 0, 1)` with `P_hat` the unit
/// direction to the source **in listener space** (x forward, y left, z up).
///
/// It is **constant-sum, not constant-power**: because `|P_hat| = 1`,
/// `sum |s_i|^2 = 4` and `sum s_i = 0`, the four gains sum to exactly 2.0 for
/// every direction. Dead ahead gives front 0.85355 / rear 0.14645; a source
/// exactly at the front-left speaker gives 1.0 / 0.5 / 0.5 / 0.0.
///
/// The debug build also has a non-linear branch
/// (`1 - (1 - expf(powf(2, |P-s|) * -0.2))`) but it is dead: its only caller
/// passes the linear flag, and the optimiser deleted it from the shipped
/// build.
///
/// **Honest caveat.** This is Reach's own `sound_math.cpp` pan law, and in the
/// shipped binary its one call site is the looping-sound audibility estimate;
/// MCC's per-voice hardware panning goes through an X3DAudio matrix in the
/// XAudio2 layer. So this is engine-exact arithmetic from the engine's own
/// table, but not literally the matrix the MCC voice uses.
pub fn speaker_pan(dir_listener_space: [f32; 3]) -> [f32; 4] {
    let p = dir_listener_space;
    let len2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    // The engine skips the normalise when the source is on top of the
    // listener; a zero direction then pans to the table's own centre.
    let p = if len2 > 1.0e-12 && len2.is_finite() {
        let inv = len2.sqrt().recip();
        [p[0] * inv, p[1] * inv, p[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    };
    let mut out = [0.0f32; 4];
    for (i, s) in SPEAKER_POSITIONS.iter().enumerate() {
        let q = [
            (p[0] - s[0]) * 0.5,
            (p[1] - s[1]) * 0.5,
            (p[2] - s[2]) * 0.5,
        ];
        let m = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
        out[i] = 1.0 - m.clamp(0.0, 1.0);
    }
    out
}

/// The four speaker gains folded to stereo: `L = FL + RL`, `R = FR + RR`,
/// halved so a source dead ahead comes out at unity (the four gains sum to
/// exactly 2.0 in every direction).
pub fn stereo_pan(dir_listener_space: [f32; 3]) -> [f32; 2] {
    let g = speaker_pan(dir_listener_space);
    [(g[0] + g[2]) * 0.5, (g[1] + g[3]) * 0.5]
}

/// A world position in LISTENER space (debug `0x18075DC90`).
///
/// ```c
/// local = ( dot(p - pos, forward), dot(p - pos, left), dot(p - pos, up) ) / scale
/// ```
///
/// so the result is `(x = forward, y = left, z = up)` — which is the basis
/// `speaker_pan` expects.
pub fn to_listener_space(
    world: [f32; 3],
    listener_pos: [f32; 3],
    forward: [f32; 3],
    left: [f32; 3],
    up: [f32; 3],
) -> [f32; 3] {
    let d = [
        world[0] - listener_pos[0],
        world[1] - listener_pos[1],
        world[2] - listener_pos[2],
    ];
    let dot = |a: [f32; 3]| a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
    [dot(forward), dot(left), dot(up)]
}

// ---------------------------------------------------------------------------------------------
// `sncl` + `snmm` — the sound-class table and the one global authored gain
// ---------------------------------------------------------------------------------------------

/// `matg +0x3D4 sound globals` (108 B): the block that names every mix tag.
pub const MATG_SOUND_GLOBALS: usize = 0x3D4;
pub const SOUND_GLOBALS_SIZE: usize = 108;
pub const SG_SOUND_CLASSES: usize = 0x00; // -> sncl
pub const SG_SOUND_EFFECTS: usize = 0x10;
pub const SG_SOUND_MIX_MASTERING: usize = 0x20; // -> snmm
pub const SG_SOUND_MIX: usize = 0x30; // -> snmx

/// `sncl` holds one block of `sound_class_block_struct`, 212 bytes each,
/// indexed by `sound_class_enum` — so 71 rows.
pub const SNCL_CLASSES: usize = 0x00;
pub const SOUND_CLASS_SIZE: usize = 212;
pub const SC_PRIORITY: usize = 0x10;
pub const SC_AIR_REVERB_GAIN_DB: usize = 0x18;
pub const SC_OVERRIDE_SPEAKER_GAIN_DB: usize = 0x38;
pub const SC_DISTANCES: usize = 0x3C;
/// `gain bounds:dB` — a `real bounds`, so (lower, upper).
pub const SC_GAIN_BOUNDS_DB: usize = 0x5C;

/// `snmm +0x08 gains` is a `sound_mix_channel_gains_struct`: +0x00 a char
/// "enabled", then master / left / right / center / lfe / back gains in dB.
pub const SNMM_GAINS: usize = 0x08;
pub const SNMM_GAINS_ENABLED: usize = SNMM_GAINS + 0x00;
pub const SNMM_MASTER_GAIN_DB: usize = SNMM_GAINS + 0x04;

/// One row of the `sncl` table.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SoundClass {
    /// `gain bounds:dB` (lower, upper). 65 of Reach's 71 classes have
    /// lower == upper, so in practice this term is deterministic.
    pub gain_bounds_db: (f32, f32),
    pub priority: i16,
    pub air_reverb_gain_db: f32,
    pub override_speaker_gain_db: f32,
    /// The class's DEFAULT attenuation distances.
    pub dist: Distances,
}

/// Reach's whole authored gain chain, read once per cache.
///
/// Every term here is tag data. The sum, and the clamp, are the engine's:
/// `sub_1806531A0` (`sound_manager.cpp:2739`) adds the distance attenuation in
/// dB, the source gain, the class ducking, the class-and-scale gain bounds and
/// the playback's `gain base`, then starts the sound only if the total is
/// above -64 dB; `sub_1809730B0` is the clamp and `sub_180973340` is
/// `10^(db/20)`.
#[derive(Clone, Debug, Default)]
pub struct GainLaw {
    /// `snmm` `gains.master gain:dB`, zero when the block is disabled. In
    /// every Reach map measured this is **-2 dB** and byte-identical.
    pub master_db: f32,
    pub master_enabled: bool,
    /// 71 rows, indexed by `sound_class_enum`.
    pub classes: Vec<SoundClass>,
    pub sncl_tag: u32,
    pub snmm_tag: u32,
}

impl GainLaw {
    /// Read it out of `matg`'s sound globals.
    pub fn read(native: &NativeDll, cache: CacheHandle) -> GainLaw {
        let mut law = GainLaw::default();
        let Some(matg) = native.tags_by_group(cache, "matg").first().copied() else { return law };
        let sg = tag_block(native, cache, matg, MATG_SOUND_GLOBALS, SOUND_GLOBALS_SIZE, 4);
        if sg.len() < SOUND_GLOBALS_SIZE {
            return law;
        }
        if let Some(sncl) = tag_ref_at(&sg, SG_SOUND_CLASSES) {
            law.sncl_tag = sncl;
            let rows = tag_block(native, cache, sncl, SNCL_CLASSES, SOUND_CLASS_SIZE, 256);
            law.classes = rows
                .chunks_exact(SOUND_CLASS_SIZE)
                .map(|c| {
                    let f = |o: usize| f32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
                    SoundClass {
                        gain_bounds_db: (f(SC_GAIN_BOUNDS_DB), f(SC_GAIN_BOUNDS_DB + 4)),
                        priority: i16::from_le_bytes([c[SC_PRIORITY], c[SC_PRIORITY + 1]]),
                        air_reverb_gain_db: f(SC_AIR_REVERB_GAIN_DB),
                        override_speaker_gain_db: f(SC_OVERRIDE_SPEAKER_GAIN_DB),
                        dist: Distances::parse(&c[SC_DISTANCES..SC_DISTANCES + DIST_SIZE])
                            .unwrap_or_default(),
                    }
                })
                .collect();
        }
        if let Some(snmm) = tag_ref_at(&sg, SG_SOUND_MIX_MASTERING) {
            law.snmm_tag = snmm;
            if let Some(b) = native.tag_meta(cache, snmm, 0, (SNMM_MASTER_GAIN_DB + 4) as u32) {
                if b.len() > SNMM_MASTER_GAIN_DB + 3 {
                    law.master_enabled = b[SNMM_GAINS_ENABLED] != 0;
                    let g = f32::from_le_bytes([
                        b[SNMM_MASTER_GAIN_DB],
                        b[SNMM_MASTER_GAIN_DB + 1],
                        b[SNMM_MASTER_GAIN_DB + 2],
                        b[SNMM_MASTER_GAIN_DB + 3],
                    ]);
                    if law.master_enabled && g.is_finite() && (-64.0..=0.0).contains(&g) {
                        law.master_db = g;
                    }
                }
            }
        }
        law
    }

    pub fn class(&self, c: u8) -> Option<&SoundClass> {
        self.classes.get(c as usize)
    }

    /// The engine's per-voice gain sum, in dB (debug `0x1806531A0`,
    /// `sound_manager.cpp:2739`, with the later stage from `0x180659D70`).
    ///
    /// ```c
    /// total_db = clamp( to_db(cone * distance_gain)      // itself floored at -64 dB
    ///                 + source.gain_db                   // per-instance; 0 here
    ///                 + sound_class_ducking_db           // transient; 0 here
    ///                 + interp_db(_power, class.lo + scale.lo,
    ///                                     class.hi + scale.hi, random01)
    ///                 + playback.gain_base_db
    ///                 + permutation.gain_db              // (int8), 1 dB per LSB
    ///                 , -64, 0 )
    ///          + snmm.master_gain_db;                    // OUTSIDE the clamp
    /// audible <=> total_db > -64
    /// ```
    ///
    /// Two subtleties that are easy to get wrong and are both load-bearing:
    /// the class-and-scale term is a **random draw in `_power` mode** between
    /// the two bounds, not the lower bound; and the `snmm` master gain is a
    /// later mastering stage and so sits **outside** the clamp.
    ///
    /// This sum is also the whole per-sound normalisation. Reach's banks
    /// decode well above full scale — 79.3% of `sfx.fsb`'s samples peak over
    /// 1.0, median +9.2 dBFS, max +41.2 — and it is these authored dB that
    /// bring each one down. Measured over `forge_halo`'s 2,353 sound tags the
    /// pre-master sum has median **-20 dB** and mean **-20.14**, which is
    /// exactly the shape the bank's peak spread needs.
    ///
    /// `distance_gain` is `None` for a sound that is not distance-attenuated.
    /// `random01` selects within the class's gain bounds; pass 0.0 for the
    /// lower bound, which is exact for the 65 of 71 classes that author
    /// `lower == upper`. **A Theater frame must be a pure function of the
    /// tick**, so a caller that wants variation must derive `random01`
    /// deterministically from the tick and the source, never from an RNG.
    pub fn voice_gain_db(
        &self,
        snd: &CacheSound,
        playback: Option<&Playback>,
        scale_gain_db: (f32, f32),
        permutation_gain_db: i8,
        distance_gain: Option<f32>,
        random01: f32,
    ) -> f32 {
        self.voice_gain_db_extra(snd, playback, scale_gain_db, permutation_gain_db, distance_gain, random01, 0.0)
    }

    /// [`GainLaw::voice_gain_db`] with the SOURCE's own gain term, which the engine sums INSIDE
    /// the clamp (`sub_180653820`: `source.gain_db + class ducking + interp_db(...)`).
    ///
    /// A looping sound's track gives it one: `looping_sound_manager.cpp` writes
    /// `track.gain_db + layer fade + pitch-range crossfade` into `source + 0xCC` every frame
    /// (debug `0x1809BA830`), and `sub_180659D70` adds that to the gain it hands the mixer.
    /// `random01` is the SCALE for a looping sound: `sub_180653820`'s last argument is
    /// `source + 0x24`, the loop's scale, not a random draw.
    #[allow(clippy::too_many_arguments)]
    pub fn voice_gain_db_extra(
        &self,
        snd: &CacheSound,
        playback: Option<&Playback>,
        scale_gain_db: (f32, f32),
        permutation_gain_db: i8,
        distance_gain: Option<f32>,
        random01: f32,
        source_db: f32,
    ) -> f32 {
        let dist_db = match distance_gain {
            Some(g) => linear_to_db(g),
            None => 0.0,
        };
        let (clo, chi) = self.class(snd.class).map(|c| c.gain_bounds_db).unwrap_or((0.0, 0.0));
        let draw = interp_db_power(clo + scale_gain_db.0, chi + scale_gain_db.1, random01);
        let base = playback.map(|p| p.gain_base_db).unwrap_or(0.0);
        let total = dist_db + source_db + draw + base + permutation_gain_db as f32;
        let clamped = if total.is_finite() { total.clamp(MIN_GAIN_DB, MAX_GAIN_DB) } else { MIN_GAIN_DB };
        clamped + self.master_db
    }

    /// The authored gain with no distance attenuation — the pre-master sum,
    /// which is the number the measurements above quote.
    pub fn authored_gain_db(
        &self,
        snd: &CacheSound,
        playback: Option<&Playback>,
        scale_gain_db: (f32, f32),
        permutation_gain_db: i8,
    ) -> f32 {
        self.voice_gain_db(snd, playback, scale_gain_db, permutation_gain_db, None, 0.0)
            - self.master_db
    }

    /// Linear gain for one sound, ready to multiply a decoded sample by.
    pub fn voice_gain(
        &self,
        snd: &CacheSound,
        playback: Option<&Playback>,
        scale_gain_db: (f32, f32),
        permutation_gain_db: i8,
        distance_gain: Option<f32>,
        random01: f32,
    ) -> f32 {
        db_to_linear(self.voice_gain_db(
            snd,
            playback,
            scale_gain_db,
            permutation_gain_db,
            distance_gain,
            random01,
        ))
    }

    /// Would the engine start this sound at all? It refuses anything at or
    /// below the -64 dB floor.
    pub fn audible(
        &self,
        snd: &CacheSound,
        playback: Option<&Playback>,
        scale_gain_db: (f32, f32),
        permutation_gain_db: i8,
        distance_gain: Option<f32>,
    ) -> bool {
        self.voice_gain_db(snd, playback, scale_gain_db, permutation_gain_db, distance_gain, 0.0)
            - self.master_db
            > MIN_GAIN_DB
    }

    /// The merged distance parameters for one sound: the class's as the base,
    /// the playback's merged over them field by field where non-zero.
    ///
    /// `ugh!.pitch range distances`, the third layer, is near-dead in Reach —
    /// `forge_halo` has 2 elements against 2,460 pitch ranges and there are
    /// only 6 distinct rows in the whole game — so it is not applied here, and
    /// the fourth layer (a per-instance source override) does not exist
    /// outside the running game.
    pub fn distances_for(&self, snd: &CacheSound, playback: Option<&Playback>) -> Distances {
        let base = self.class(snd.class).map(|c| c.dist).unwrap_or_default();
        match playback {
            Some(p) => merge_distances(base, p.dist),
            None => base,
        }
    }
}

/// `interp_db` in `_power` mode (debug `0x180973110`, release `0x1800AF94C`,
/// mode 1 of five).
///
/// "Power" means the interpolation happens in **linear gain**, not in dB:
/// `g = lerp(10^(a/20), 10^(b/20), t); db = to_db(g)`. The other modes are
/// 0 `_linear` (lerp in dB), 2/3 `_inverse_power` (a square-root reshaping of
/// `t`) and 4 `_ease_in_out` (`3t^2 - 2t^3`, then as mode 1).
pub fn interp_db_power(a_db: f32, b_db: f32, t: f32) -> f32 {
    if (a_db - b_db).abs() < 1e-6 {
        return a_db;
    }
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let g = db_to_linear(a_db) + (db_to_linear(b_db) - db_to_linear(a_db)) * t;
    linear_to_db(g)
}

/// `interp_db` with its mode (debug `0x180973110`, all five arms). `t` is clamped to `0..1` first
/// (`0x180973410` does that for the callers that use it).
///
/// 0 `_linear`: lerp in dB. 1 `_power`: lerp in linear gain. 2 `_inverse_power`: linear gain
/// again, with `t` reshaped by `sqrt` when the end is louder than the start (and by
/// `1 - sqrt(1 - t)` when it is quieter). 3 is mode 2 with the two shapes swapped. 4
/// `_ease_in_out`: `3t^2 - 2t^3` as the weight, in linear gain.
///
/// The looping-sound manager crossfades overlapping pitch ranges with mode 2 from -64 dB to 0 dB
/// (`looping_sound_manager.cpp`, debug `0x1809BA830`).
pub fn interp_db(mode: u8, a_db: f32, b_db: f32, t: f32) -> f32 {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    if mode == 0 {
        return a_db + (b_db - a_db) * t;
    }
    let (ga, gb) = (db_to_linear(a_db), db_to_linear(b_db));
    let w = match mode {
        2 => if gb > ga { t.sqrt() } else { 1.0 - (1.0 - t).sqrt() },
        3 => if gb > ga { 1.0 - (1.0 - t).sqrt() } else { t.sqrt() },
        4 => (t * t * 3.0 - 2.0 * t * t * t).clamp(0.0, 1.0),
        _ => t,
    };
    linear_to_db(ga + (gb - ga) * w)
}

// ---------------------------------------------------------------------------------------------
// `lsnd` — sound_looping: the ambience and music beds
// ---------------------------------------------------------------------------------------------

pub const LSND_SIZE: usize = 56;
pub const LSND_FLAGS: usize = 0x00;
pub const LSND_RUNTIME_DISTANCE_BOUNDS: usize = 0x0C;
pub const LSND_MAX_FLYBY_RANGE: usize = 0x14;
pub const LSND_RUNTIME_SOUND_CLASS: usize = 0x18;
pub const LSND_MASTER_MARKER_TRACK: usize = 0x1C;
pub const LSND_TRACKS: usize = 0x20;
pub const LSND_DETAIL_SOUNDS: usize = 0x2C;

/// `looping_sound_track_block`, 176 bytes.
pub const TRACK_SIZE: usize = 176;
pub const TRACK_NAME: usize = 0x00;
pub const TRACK_FLAGS: usize = 0x04;
pub const TRACK_GAIN_DB: usize = 0x08;
pub const TRACK_IN: usize = 0x0C;
pub const TRACK_LOOP: usize = 0x1C;
pub const TRACK_OUT: usize = 0x2C;
pub const TRACK_ALT_LOOP: usize = 0x3C;
pub const TRACK_ALT_OUT: usize = 0x4C;
pub const TRACK_FADE_IN_SECONDS: usize = 0x7C;
pub const TRACK_FADE_IN_MODE: usize = 0x80;
pub const TRACK_FADE_OUT_SECONDS: usize = 0x84;
pub const TRACK_FADE_OUT_MODE: usize = 0x88;
pub const TRACK_ALT_CROSSFADE_SECONDS: usize = 0x8C;
pub const TRACK_ALT_CROSSFADE_MODE: usize = 0x90;

/// `looping_sound_flags` bit 1: "not a loop — this is a collection of
/// permutations strung together that should play once then stop".
pub const LSND_FLAG_NOT_A_LOOP: u32 = 1 << 1;

/// One track of a looping sound.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoopTrack {
    pub name_sid: u32,
    pub gain_db: f32,
    /// The `snd!` that plays once on entry, if any.
    pub intro: Option<u32>,
    /// The `snd!` that loops. **This is the ambience bed.**
    pub loop_snd: Option<u32>,
    pub outro: Option<u32>,
    /// The `snd!` that loops while the owner is LOADED (`sound_looping_set_alternate` true).
    /// The vehicle engine flips it from its `load and cruise sound` block: cruise = `loop`,
    /// loaded = `alt loop`.
    pub alt_loop: Option<u32>,
    pub alt_out: Option<u32>,
    pub fade_in_seconds: f32,
    pub fade_out_seconds: f32,
    pub alt_crossfade_seconds: f32,
    /// `looping_sound_fade_modes`, which are `interp_db`'s modes (see [`interp_db`]).
    pub fade_in_mode: u8,
    pub fade_out_mode: u8,
    pub alt_crossfade_mode: u8,
}

/// An `lsnd`, flattened.
#[derive(Clone, Debug, Default)]
pub struct LoopingSound {
    pub tag: u32,
    pub flags: u32,
    pub runtime_class: u16,
    pub tracks: Vec<LoopTrack>,
}

impl LoopingSound {
    pub fn read(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<LoopingSound> {
        if tag == 0 || tag == 0xFFFF_FFFF {
            return None;
        }
        let h = native.tag_meta(cache, tag, 0, LSND_SIZE as u32)?;
        if h.len() < LSND_SIZE {
            return None;
        }
        let tb = block_at(native, cache, &h, LSND_TRACKS, TRACK_SIZE, 64);
        let tracks = tb
            .chunks_exact(TRACK_SIZE)
            .map(|c| {
                let f = |o: usize| f32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
                LoopTrack {
                    name_sid: u32::from_le_bytes([c[TRACK_NAME], c[TRACK_NAME + 1], c[TRACK_NAME + 2], c[TRACK_NAME + 3]]),
                    gain_db: f(TRACK_GAIN_DB),
                    intro: tag_ref_at(c, TRACK_IN),
                    loop_snd: tag_ref_at(c, TRACK_LOOP),
                    outro: tag_ref_at(c, TRACK_OUT),
                    alt_loop: tag_ref_at(c, TRACK_ALT_LOOP),
                    alt_out: tag_ref_at(c, TRACK_ALT_OUT),
                    fade_in_seconds: f(TRACK_FADE_IN_SECONDS),
                    fade_out_seconds: f(TRACK_FADE_OUT_SECONDS),
                    alt_crossfade_seconds: f(TRACK_ALT_CROSSFADE_SECONDS),
                    fade_in_mode: c[TRACK_FADE_IN_MODE],
                    fade_out_mode: c[TRACK_FADE_OUT_MODE],
                    alt_crossfade_mode: c[TRACK_ALT_CROSSFADE_MODE],
                }
            })
            .collect();
        Some(LoopingSound {
            tag,
            flags: u32::from_le_bytes([h[0], h[1], h[2], h[3]]),
            runtime_class: u16::from_le_bytes([h[LSND_RUNTIME_SOUND_CLASS], h[LSND_RUNTIME_SOUND_CLASS + 1]]),
            tracks,
        })
    }

    pub fn is_loop(&self) -> bool {
        self.flags & LSND_FLAG_NOT_A_LOOP == 0
    }
}

// ---------------------------------------------------------------------------------------------
// The scenario's ACOUSTICS palette — Reach's map ambience
// ---------------------------------------------------------------------------------------------

/// `scnr +0x528 acoustics palette{background sound palette}`, 120 bytes an
/// entry. The BSP's own cluster block picks a row with its
/// `structure_bsp_cluster_block_struct +0x1C short block index acoustics`.
pub const SCNR_ACOUSTICS_PALETTE: usize = 0x528;
pub const ACOUSTICS_SIZE: usize = 120;
pub const AC_NAME: usize = 0x00;
/// `reverb` (`scenario_acoustics_environment_definition`, 28 B).
pub const AC_REVERB: usize = 0x04;
pub const AC_REVERB_ENVIRONMENT: usize = AC_REVERB + 0x00;
pub const AC_REVERB_CUTOFF: usize = AC_REVERB + 0x14;
/// `ambience` (`scenario_acoustics_ambience_definition`, 88 B).
pub const AC_AMBIENCE: usize = 0x20;
pub const AC_BACKGROUND_SOUND: usize = AC_AMBIENCE + 0x00;
pub const AC_WEATHER_SOUND: usize = AC_AMBIENCE + 0x10;
pub const AC_ENTRY_SOUND: usize = AC_AMBIENCE + 0x20;
pub const AC_EXIT_SOUND: usize = AC_AMBIENCE + 0x30;
pub const AC_AMBIENCE_CUTOFF: usize = AC_AMBIENCE + 0x40;

/// One acoustics-palette row: an area's reverb and its ambience bed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Acoustics {
    pub index: usize,
    pub name_sid: u32,
    pub name: String,
    pub sound_environment: Option<u32>,
    /// The `lsnd` that plays continuously while the listener is in this area.
    pub background_sound: Option<u32>,
    pub weather_sound: Option<u32>,
    pub entry_sound: Option<u32>,
    pub exit_sound: Option<u32>,
    pub ambience_cutoff: f32,
}

/// `sbsp +0x9C clusters`, and `sbsp +0xF0` its OWN acoustics palette.
///
/// A BSP carries both the cluster table and, usually, the palette its clusters
/// index — so the exterior/interior pair a map authors lives here rather than
/// only in the scenario. Each `structure_bsp_cluster_block_struct` is 140
/// bytes and begins with its own bounds:
///
/// ```text
///   +0x00  real bounds  bounds x
///   +0x08  real bounds  bounds y
///   +0x10  real bounds  bounds z
///   +0x18  char  scenario sky index
///   +0x19  char  atmosphere index
///   +0x1A  char  camera fx index
///   +0x1B  char  weather index
///   +0x1C  short block index  acoustics      <-- the row for this cluster
///   +0x1E  short  acoustics sound cluster index
/// ```
pub const SBSP_CLUSTERS: usize = 0x9C;
pub const SBSP_ACOUSTICS_PALETTE: usize = 0xF0;
pub const CLUSTER_SIZE: usize = 140;
pub const CL_BOUNDS_X: usize = 0x00;
pub const CL_ACOUSTICS: usize = 0x1C;

/// One cluster's box and the acoustics row it selects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcousticCluster {
    pub index: usize,
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// Index into the acoustics palette, or `None` when the cluster names no
    /// row (`-1`).
    pub acoustics: Option<usize>,
}

impl AcousticCluster {
    pub fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }

    /// Volume of the box, so the SMALLEST containing cluster wins — an
    /// interior room is inside the exterior cluster's box, and the engine's
    /// own cluster graph is a partition, so the tightest fit is the right
    /// answer when boxes overlap.
    pub fn volume(&self) -> f32 {
        (0..3).map(|i| (self.max[i] - self.min[i]).max(0.0)).product()
    }

    /// Distance from `p` to the box, zero inside.
    pub fn distance(&self, p: [f32; 3]) -> f32 {
        let mut d = 0.0f32;
        for i in 0..3 {
            let v = (self.min[i] - p[i]).max(0.0).max(p[i] - self.max[i]);
            d += v * v;
        }
        d.sqrt()
    }
}

/// Every cluster of one BSP, with its acoustics row.
pub fn bsp_acoustic_clusters(
    native: &NativeDll,
    cache: CacheHandle,
    sbsp: u32,
) -> Vec<AcousticCluster> {
    let b = tag_block(native, cache, sbsp, SBSP_CLUSTERS, CLUSTER_SIZE, 8192);
    b.chunks_exact(CLUSTER_SIZE)
        .enumerate()
        .filter_map(|(index, c)| {
            let f = |o: usize| f32::from_le_bytes([c[o], c[o + 1], c[o + 2], c[o + 3]]);
            let min = [f(CL_BOUNDS_X), f(CL_BOUNDS_X + 8), f(CL_BOUNDS_X + 16)];
            let max = [f(CL_BOUNDS_X + 4), f(CL_BOUNDS_X + 12), f(CL_BOUNDS_X + 20)];
            if !min.iter().chain(max.iter()).all(|v| v.is_finite()) {
                return None;
            }
            if (0..3).any(|i| max[i] < min[i]) {
                return None;
            }
            let a = i16::from_le_bytes([c[CL_ACOUSTICS], c[CL_ACOUSTICS + 1]]);
            Some(AcousticCluster {
                index,
                min,
                max,
                acoustics: (a >= 0).then_some(a as usize),
            })
        })
        .collect()
}

/// A BSP's own acoustics palette, the same 120-byte rows the scenario holds.
pub fn bsp_acoustics_palette(native: &NativeDll, cache: CacheHandle, sbsp: u32) -> Vec<Acoustics> {
    read_acoustics(native, cache, sbsp, SBSP_ACOUSTICS_PALETTE)
}

/// Which acoustics row the listener is in.
///
/// The engine binds a sound environment and a background bed to the
/// listener's CLUSTER (`structure_bsp_cluster_block_struct +0x1C`), which is
/// why a map authors an `ext`/`int` pair rather than one bed: standing inside
/// a structure must not play the outdoor one. Clusters overlap, so the
/// smallest box containing the point wins, and when the point is outside every
/// box the nearest cluster does — a listener just outside the hull is still in
/// the area it is next to.
pub fn acoustics_at(clusters: &[AcousticCluster], pos: [f32; 3]) -> Option<usize> {
    let mut inside: Option<&AcousticCluster> = None;
    for c in clusters {
        if c.contains(pos) && inside.map_or(true, |b| c.volume() < b.volume()) {
            inside = Some(c);
        }
    }
    if let Some(c) = inside {
        return c.acoustics;
    }
    clusters
        .iter()
        .min_by(|a, b| a.distance(pos).partial_cmp(&b.distance(pos)).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|c| c.acoustics)
}

/// Read a scenario's whole acoustics palette.
pub fn acoustics_palette(native: &NativeDll, cache: CacheHandle, scnr: u32) -> Vec<Acoustics> {
    read_acoustics(native, cache, scnr, SCNR_ACOUSTICS_PALETTE)
}

/// The one palette reader, shared by the scenario's copy and the BSP's.
fn read_acoustics(native: &NativeDll, cache: CacheHandle, tag: u32, off: usize) -> Vec<Acoustics> {
    let b = tag_block(native, cache, tag, off, ACOUSTICS_SIZE, 1024);
    b.chunks_exact(ACOUSTICS_SIZE)
        .enumerate()
        .map(|(index, c)| {
            let sid = u32::from_le_bytes([c[AC_NAME], c[AC_NAME + 1], c[AC_NAME + 2], c[AC_NAME + 3]]);
            Acoustics {
                index,
                name_sid: sid,
                name: native.string_id(cache, sid).unwrap_or_default(),
                sound_environment: tag_ref_at(c, AC_REVERB_ENVIRONMENT),
                background_sound: tag_ref_at(c, AC_BACKGROUND_SOUND),
                weather_sound: tag_ref_at(c, AC_WEATHER_SOUND),
                entry_sound: tag_ref_at(c, AC_ENTRY_SOUND),
                exit_sound: tag_ref_at(c, AC_EXIT_SOUND),
                ambience_cutoff: f32::from_le_bytes([
                    c[AC_AMBIENCE_CUTOFF],
                    c[AC_AMBIENCE_CUTOFF + 1],
                    c[AC_AMBIENCE_CUTOFF + 2],
                    c[AC_AMBIENCE_CUTOFF + 3],
                ]),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Walking an object tag for its sounds
// ---------------------------------------------------------------------------------------------

/// A tag reference in a Reach cache is 16 bytes; the tag index is the last
/// u32 and `0xFFFFFFFF` is null.
pub const TAG_REF_SIZE: usize = 16;

/// Read the tag index out of a 16-byte tag reference at `off` in `b`.
///
/// **The u32 at +0x0C is an MCC datum handle, `salt << 16 | index`, not a tag
/// index** — the low half is the index. Reading it unmasked yields a handle
/// that no `tag_meta` call can resolve, which looks exactly like a missing
/// tag; it cost this module a round of "the `sncl` table has zero rows".
/// `reach_projectile::tag_ref` and `reach_actor` mask it the same way.
pub fn tag_ref_at(b: &[u8], off: usize) -> Option<u32> {
    let s = b.get(off..off + TAG_REF_SIZE)?;
    let d = u32::from_le_bytes([s[12], s[13], s[14], s[15]]);
    if d == 0xFFFF_FFFF || d == 0 {
        None
    } else {
        Some(d & 0xFFFF)
    }
}

// ---------------------------------------------------------------------------------------------
// `effe` — the effect that a firing barrel plays, and the sounds inside it
// ---------------------------------------------------------------------------------------------

/// `effect_block_struct` is 96 bytes.
pub const EFFE_LOCATIONS: usize = 0x20;
pub const EFFE_EVENTS: usize = 0x2C;
pub const EFFE_LOOPING_SOUNDS: usize = 0x38;

pub const EFFECT_EVENT_SIZE: usize = 64;
pub const EVENT_PARTS: usize = 0x1C;

pub const EFFECT_PART_SIZE: usize = 100;
/// `runtime base group tag!` — the 4CC of the part's BASE group, which is how
/// a sound part is told from a particle part without a tag-class lookup.
pub const PART_BASE_GROUP: usize = 0x0C;
pub const PART_TYPE: usize = 0x14;

/// `effect_looping_sound_block_struct`, 20 bytes: a tag reference to an
/// `lsnd`, then the location and bind-scale indices.
pub const EFFECT_LOOPING_SOUND_SIZE: usize = 20;

/// How deep to follow `effe` -> `effe` parts. Nested effects do occur; a
/// visited set plus this bound is what stops a cycle in hostile data.
const EFFECT_RECURSION: u32 = 3;

/// Every `snd!` and `lsnd` one effect can play, by following its events'
/// parts and its looping-sound block.
///
/// Returns `(tag, is_looping)` — `lsnd` tags come back with `true`.
pub fn effect_sounds(native: &NativeDll, cache: CacheHandle, effe: u32) -> Vec<(u32, bool)> {
    let mut out = Vec::new();
    let mut seen = Vec::new();
    effect_sounds_into(native, cache, effe, EFFECT_RECURSION, &mut seen, &mut out);
    out
}

fn effect_sounds_into(
    native: &NativeDll,
    cache: CacheHandle,
    effe: u32,
    depth: u32,
    seen: &mut Vec<u32>,
    out: &mut Vec<(u32, bool)>,
) {
    if effe == 0 || effe == 0xFFFF_FFFF || seen.contains(&effe) || out.len() > 64 {
        return;
    }
    seen.push(effe);
    // The looping-sound block names an `lsnd` directly.
    let ls = tag_block(native, cache, effe, EFFE_LOOPING_SOUNDS, EFFECT_LOOPING_SOUND_SIZE, 32);
    for c in ls.chunks_exact(EFFECT_LOOPING_SOUND_SIZE) {
        if let Some(t) = tag_ref_at(c, 0) {
            if !out.iter().any(|(o, _)| *o == t) {
                out.push((t, true));
            }
        }
    }
    let evs = tag_block(native, cache, effe, EFFE_EVENTS, EFFECT_EVENT_SIZE, 64);
    for ev in evs.chunks_exact(EFFECT_EVENT_SIZE) {
        let parts = block_at(native, cache, ev, EVENT_PARTS, EFFECT_PART_SIZE, 64);
        for pt in parts.chunks_exact(EFFECT_PART_SIZE) {
            let Some(t) = tag_ref_at(pt, PART_TYPE) else { continue };
            // Prefer the part's own recorded base group over a tag-class
            // lookup: it is what the engine switches on, and it is free.
            let base = u32::from_be_bytes([
                pt[PART_BASE_GROUP],
                pt[PART_BASE_GROUP + 1],
                pt[PART_BASE_GROUP + 2],
                pt[PART_BASE_GROUP + 3],
            ]);
            let group = match &base.to_be_bytes() {
                b"snd!" => "snd!".to_string(),
                b"lsnd" => "lsnd".to_string(),
                b"effe" => "effe".to_string(),
                // The recorded 4CC can be byte-swapped depending on how the
                // cache was built, so fall back to asking the cache.
                _ => native.tag_class(cache, t).unwrap_or_default(),
            };
            match group.as_str() {
                "snd!" => {
                    if !out.iter().any(|(o, _)| *o == t) {
                        out.push((t, false));
                    }
                }
                "lsnd" => {
                    if !out.iter().any(|(o, _)| *o == t) {
                        out.push((t, true));
                    }
                }
                "effe" if depth > 0 => {
                    effect_sounds_into(native, cache, t, depth - 1, seen, out)
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// `weap` -> the firing sound
// ---------------------------------------------------------------------------------------------

/// `weapon_barrels`, at weapon-relative +0x204 = absolute +0x408. Element
/// stride **388 (0x184)**, recovered from `reach_tag_test.exe`'s runtime-built
/// `weapon_barrel_definition_fields` array and confirmed in the cache three
/// ways (the `proj` reference recurring at +0x104 and +0x288, the block ending
/// exactly at the tag meta, and the field sizes summing to 0x184).
pub const WEAP_BARRELS_ABS: usize = 0x408;
pub const BARREL_SIZE: usize = 388;
/// `k_weapon_barrel_count`.
pub const BARREL_MAX: i32 = 2;
pub const BARREL_MARKER_NAME: usize = 0x040;
pub const BARREL_PROJECTILE: usize = 0x104;
pub const BARREL_FIRING_EFFECTS: usize = 0x178;

/// `barrel_firing_effect_block`: stride **196 (0xC4)** — a shot-count bound
/// pair then **twelve** tag references (firing / misfire / empty / optional
/// secondary, each in an effect, a damage and a rider-damage flavour).
pub const FIRING_EFFECT_SIZE: usize = 196;
pub const FIRING_EFFECT_MAX: i32 = 4; // k_weapon_barrel_effect_count
pub const FE_FIRING_EFFECT: usize = 0x004;
pub const FE_MISFIRE_EFFECT: usize = 0x014;
pub const FE_EMPTY_EFFECT: usize = 0x024;
pub const FE_SECONDARY_FIRING_EFFECT: usize = 0x034;
pub const FE_FIRING_DAMAGE: usize = 0x044;

/// `object_struct_definition +0x104 attachments`, 32 bytes an element: a tag
/// reference (which may be an `lsnd`), the marker it hangs on, and the object
/// functions that scale it.
pub const OBJE_ATTACHMENTS: usize = 0x104;
pub const ATTACHMENT_SIZE: usize = 32;
pub const ATTACH_TYPE: usize = 0x00;
pub const ATTACH_MARKER: usize = 0x10;
pub const ATTACH_PRIMARY_SCALE: usize = 0x18;

/// Where a weapon's firing sound was found. Reported rather than hidden,
/// because the two routes behave differently: a barrel's firing effect is a
/// one-shot per round, an attachment is a continuous loop gated by an object
/// function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireSource {
    /// `weap` -> barrel -> firing effect (`effe`) -> part -> `snd!`.
    BarrelEffect,
    /// `weap` -> attachments -> `lsnd` -> track `loop` -> `snd!`. This is how
    /// the continuous weapons work (focus rifle, beams, turret loops, horns).
    Attachment,
}

/// What a weapon plays when it fires.
#[derive(Clone, Debug)]
pub struct WeaponFire {
    pub weap: u32,
    pub barrel: usize,
    pub source: FireSource,
    /// The `snd!` whose sound class is `weapon_fire` (4) or
    /// `space_weapon_fire` (64) — the shot itself, not the distant LOD copy
    /// (15) nor the interior/exterior tails (58/59).
    pub snd: u32,
    pub class: u8,
    /// Every `snd!` the chain reached, with its class, for diagnostics.
    pub all: Vec<(u32, u8)>,
}

/// Sound classes that are the shot itself.
pub const CLASS_WEAPON_FIRE: u8 = 4;
pub const CLASS_SPACE_WEAPON_FIRE: u8 = 64;

/// Resolve a weapon's firing sound.
///
/// Measured over 273 barrels on 8 Reach maps: **273 of 273 reach an FSB asset
/// path**, 243 through the barrel's firing effect and 30 through an `lsnd`
/// attachment. The 12 `weap` tags with no barrel at all (energy sword, gravity
/// hammer, flag, skull, ...) are melee and objective items and correctly have
/// no firing chain.
pub fn weapon_fire_sound(native: &NativeDll, cache: CacheHandle, weap: u32) -> Vec<WeaponFire> {
    let mut out = Vec::new();
    let barrels = tag_block(native, cache, weap, WEAP_BARRELS_ABS, BARREL_SIZE, BARREL_MAX);
    for (bi, b) in barrels.chunks_exact(BARREL_SIZE).enumerate() {
        let fx = block_at(native, cache, b, BARREL_FIRING_EFFECTS, FIRING_EFFECT_SIZE, FIRING_EFFECT_MAX);
        let mut found: Vec<(u32, u8)> = Vec::new();
        for fe in fx.chunks_exact(FIRING_EFFECT_SIZE) {
            let Some(effe) = tag_ref_at(fe, FE_FIRING_EFFECT) else { continue };
            for (t, looping) in effect_sounds(native, cache, effe) {
                collect_classes(native, cache, t, looping, &mut found);
            }
        }
        if let Some(pick) = pick_fire(&found) {
            out.push(WeaponFire { weap, barrel: bi, source: FireSource::BarrelEffect, snd: pick.0, class: pick.1, all: found });
        } else if !found.is_empty() {
            // The chain reached sounds but none is the shot class; report the
            // first rather than silently dropping the weapon.
            let f0 = found[0];
            out.push(WeaponFire { weap, barrel: bi, source: FireSource::BarrelEffect, snd: f0.0, class: f0.1, all: found });
        }
    }
    if out.is_empty() {
        // The continuous weapons: the firing sound hangs off the object as an
        // attachment, gated by an object function, not off a barrel.
        let mut found: Vec<(u32, u8)> = Vec::new();
        for (t, looping) in object_sound_attachments(native, cache, weap) {
            collect_classes(native, cache, t, looping, &mut found);
        }
        if let Some(pick) = pick_fire(&found) {
            out.push(WeaponFire { weap, barrel: 0, source: FireSource::Attachment, snd: pick.0, class: pick.1, all: found });
        } else if let Some(f0) = found.first().copied() {
            out.push(WeaponFire { weap, barrel: 0, source: FireSource::Attachment, snd: f0.0, class: f0.1, all: found });
        }
    }
    out
}

/// Prefer the shot's own class; a firing effect typically yields four sounds
/// and picking the first gives the distant LOD copy or a tail.
fn pick_fire(found: &[(u32, u8)]) -> Option<(u32, u8)> {
    found
        .iter()
        .find(|(_, c)| *c == CLASS_WEAPON_FIRE || *c == CLASS_SPACE_WEAPON_FIRE)
        .copied()
}

/// Add a `snd!` (or every `snd!` inside an `lsnd`) to `found`, with its class.
fn collect_classes(
    native: &NativeDll,
    cache: CacheHandle,
    tag: u32,
    looping: bool,
    found: &mut Vec<(u32, u8)>,
) {
    if looping {
        let Some(l) = LoopingSound::read(native, cache, tag) else { return };
        for tr in &l.tracks {
            for t in [tr.loop_snd, tr.intro].into_iter().flatten() {
                if let Some(s) = CacheSound::read(native, cache, t) {
                    if !found.iter().any(|(o, _)| *o == t) {
                        found.push((t, s.class));
                    }
                }
            }
        }
    } else if let Some(s) = CacheSound::read(native, cache, tag) {
        if !found.iter().any(|(o, _)| *o == tag) {
            found.push((tag, s.class));
        }
    }
}

/// Every `snd!` or `lsnd` attached to an object, as `(tag, is_looping)`.
pub fn object_sound_attachments(native: &NativeDll, cache: CacheHandle, obj: u32) -> Vec<(u32, bool)> {
    let b = tag_block(native, cache, obj, OBJE_ATTACHMENTS, ATTACHMENT_SIZE, 64);
    let mut out = Vec::new();
    for c in b.chunks_exact(ATTACHMENT_SIZE) {
        let Some(t) = tag_ref_at(c, ATTACH_TYPE) else { continue };
        match native.tag_class(cache, t).unwrap_or_default().as_str() {
            "snd!" => out.push((t, false)),
            "lsnd" => out.push((t, true)),
            _ => {}
        }
    }
    out
}

/// `weapon_struct_definition +0x0B4 clang effect` — "effect that is played in
/// the air between two melee weapons". Absolute: `516 + 0xB4`.
pub const WEAP_CLANG_EFFECT: usize = WEAPON_BASE + 0x0B4;

/// `weap +0x1B4 player interface` (`weapon_interface_struct`, 44 B). Its
/// `first person` block has one element per species — index 0 Spartan,
/// index 1 Elite — and each names the first-person model and the animation
/// graph that poses it.
pub const WEAP_PLAYER_INTERFACE: usize = WEAPON_BASE + 0x1B4;
pub const WI_FIRST_PERSON: usize = 0x00;
pub const FIRST_PERSON_SIZE: usize = 32;
pub const FP_MODEL: usize = 0x00;
pub const FP_ANIMATIONS: usize = 0x10;

/// `model_animation_graph_block_struct +0x054 sound references`
/// (`animation_graph_sound_reference_block`, 20 B): a tag reference to a
/// `snd!` or `lsnd`, then the key-event flags.
///
/// This is where a weapon's RELOAD, MELEE SWING, READY and IDLE sounds live.
/// They are not on the weapon and they are not in the magazine block — the
/// magazine's four reload/chamber effect references are null on every shipped
/// weapon — they are authored on the first-person animation graph, and each
/// `snd!`'s own sound class says which is which.
pub const JMAD_SOUND_REFERENCES: usize = 0x054;
pub const JMAD_SOUND_REF_SIZE: usize = 20;

/// Sound classes of the animation-authored weapon sounds.
pub const CLASS_WEAPON_READY: u8 = 5;
pub const CLASS_WEAPON_RELOAD: u8 = 6;
pub const CLASS_WEAPON_EMPTY: u8 = 7;
pub const CLASS_WEAPON_IDLE: u8 = 10;
pub const CLASS_WEAPON_MELEE: u8 = 11;

/// Every `snd!` on a weapon's first-person animation graph, with its class.
///
/// `species` is 0 for a Spartan and 1 for an Elite, matching the
/// `first person` block's own element order.
pub fn weapon_animation_sounds(
    native: &NativeDll,
    cache: CacheHandle,
    weap: u32,
    species: usize,
) -> Vec<(u32, u8)> {
    let Some(iface) = native.tag_meta(cache, weap, WEAP_PLAYER_INTERFACE as u32, 44) else {
        return Vec::new();
    };
    let fp = block_at(native, cache, &iface, WI_FIRST_PERSON, FIRST_PERSON_SIZE, 8);
    let rows = fp.len() / FIRST_PERSON_SIZE;
    if rows == 0 {
        return Vec::new();
    }
    // A weapon with only one row uses it for both species.
    let row = species.min(rows - 1);
    let Some(jmad) = tag_ref_at(&fp[row * FIRST_PERSON_SIZE..], FP_ANIMATIONS) else {
        return Vec::new();
    };
    let refs = tag_block(native, cache, jmad, JMAD_SOUND_REFERENCES, JMAD_SOUND_REF_SIZE, 256);
    let mut out = Vec::new();
    for c in refs.chunks_exact(JMAD_SOUND_REF_SIZE) {
        let Some(t) = tag_ref_at(c, 0) else { continue };
        match native.tag_class(cache, t).unwrap_or_default().as_str() {
            "snd!" => {
                if let Some(sn) = CacheSound::read(native, cache, t) {
                    out.push((t, sn.class));
                }
            }
            "lsnd" => {
                if let Some(l) = LoopingSound::read(native, cache, t) {
                    for tr in &l.tracks {
                        if let Some(sd) = tr.loop_snd.or(tr.intro) {
                            if let Some(sn) = CacheSound::read(native, cache, sd) {
                                out.push((sd, sn.class));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The weapon's MELEE SWING sound: the first-person animation graph's `snd!`
/// whose class is `weapon_melee` (11).
pub fn weapon_melee_sound(
    native: &NativeDll,
    cache: CacheHandle,
    weap: u32,
    species: usize,
) -> Option<u32> {
    weapon_animation_sounds(native, cache, weap, species)
        .into_iter()
        .find(|(_, c)| *c == CLASS_WEAPON_MELEE)
        .map(|(t, _)| t)
}

/// The weapon's RELOAD sound, from the same graph (class `weapon_reload`, 6).
pub fn weapon_reload_sound(
    native: &NativeDll,
    cache: CacheHandle,
    weap: u32,
    species: usize,
) -> Option<u32> {
    weapon_animation_sounds(native, cache, weap, species)
        .into_iter()
        .find(|(_, c)| *c == CLASS_WEAPON_RELOAD)
        .map(|(t, _)| t)
}

/// The CLANG: `weap +0x2B8 clang effect`, the effect the engine plays in the
/// air when two melee weapons meet. Walks the effect to its sounds.
/// Returns EVERY candidate, in the effect's own order, not just the first.
///
/// Taking the first was wrong and the gate caught it: of 450 weapons across 8
/// maps, 230 name a clang effect but only **62** of those effects' first sound
/// has a permutation in the banks — the rest lead with a looping sound whose
/// `loop` reference is null, or with a `snd!` the map carries no samples for.
/// The caller picks the first candidate that actually resolves, which is the
/// same thing `Clip::resolve` does for a single sound.
pub fn weapon_clang_sounds(native: &NativeDll, cache: CacheHandle, weap: u32) -> Vec<u32> {
    let Some(b) = native.tag_meta(cache, weap, WEAP_CLANG_EFFECT as u32, TAG_REF_SIZE as u32) else {
        return Vec::new();
    };
    let Some(effe) = tag_ref_at(&b, 0) else { return Vec::new() };
    let mut out = Vec::new();
    for (t, looping) in effect_sounds(native, cache, effe) {
        if looping {
            if let Some(l) = LoopingSound::read(native, cache, t) {
                for tr in &l.tracks {
                    out.extend(tr.loop_snd.or(tr.intro));
                }
            }
        } else {
            out.push(t);
        }
    }
    out
}

/// The EMPTY CLICK: the barrel firing effect's `empty effect`, which is a
/// direct `snd!` on most weapons and an `effe` on the rest.
pub fn weapon_empty_sound(native: &NativeDll, cache: CacheHandle, weap: u32) -> Option<u32> {
    let barrels = tag_block(native, cache, weap, WEAP_BARRELS_ABS, BARREL_SIZE, BARREL_MAX);
    for b in barrels.chunks_exact(BARREL_SIZE) {
        let fx = block_at(native, cache, b, BARREL_FIRING_EFFECTS, FIRING_EFFECT_SIZE, FIRING_EFFECT_MAX);
        for fe in fx.chunks_exact(FIRING_EFFECT_SIZE) {
            let Some(t) = tag_ref_at(fe, FE_EMPTY_EFFECT) else { continue };
            match native.tag_class(cache, t).unwrap_or_default().as_str() {
                "snd!" => return Some(t),
                "effe" => {
                    for (s, looping) in effect_sounds(native, cache, t) {
                        if !looping {
                            return Some(s);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// FOOTSTEPS — the animation's own sound events
// ---------------------------------------------------------------------------------------------
//
// A footstep in Reach is not on a timer. The animation carries the event: a
// clip's `sound events` block names, per event, WHICH of the graph's sound
// references to play and on WHICH FRAME, so the left/right alternation and the
// cadence are the animator's and they stay right at any playback speed.
//
// ```text
//   jmad +0x094 animations          animation_pool_block_struct, 60 B
//        +0x30  shared animation data  -> shared_model_animation_block, 212 B
//               +0x02  short frame count
//               +0x44  sound events       -> see SOUND_EVENT_SIZE
//   jmad +0x054 sound references     animation_graph_sound_reference_block, 20 B
//               +0x00  tag reference  snd! / lsnd
//               +0x10  word flags     key_event_flags_enum
// ```

/// `shared_model_animation_block` offsets.
pub const SHARED_FRAME_COUNT: usize = 0x02;
pub const SHARED_SOUND_EVENTS: usize = 0x44;

/// `animation_sound_event_block` element size.
///
/// **Measured, because two sources disagreed.** The reflection data gives
/// `animation_sound_event_block` as **8** bytes
/// (`+0x00 short sound, +0x02 short frame, +0x04 string id marker name`) while
/// a sibling investigation reported the 12-byte
/// `animation_sound_event_block_extended`. `probes3::probe_animation_sound_event_stride`
/// decides it against the only oracle that matters — every event's `sound`
/// index must be inside the graph's sound-reference block and its `frame`
/// inside the clip's own frame count — over every animation of every biped
/// graph on the machine, and reports both candidates' scores.
pub const SOUND_EVENT_SIZE: usize = 8;
pub const SOUND_EVENT_SIZE_EXTENDED: usize = 12;
pub const SE_SOUND: usize = 0x00;
pub const SE_FRAME: usize = 0x02;
pub const SE_MARKER: usize = 0x04;

/// `unit_footsteps` and its postponed variant.
pub const CLASS_UNIT_FOOTSTEPS: u8 = 18;
pub const CLASS_UNIT_FOOTSTEPS_POSTPONE: u8 = 68;

/// One animation sound event: play graph sound `sound` on `frame`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimSoundEvent {
    /// Index into the graph's `sound references` block.
    pub sound: u16,
    /// Frame within the clip, at 30 fps.
    pub frame: u16,
    /// The marker the engine plays it from (a foot node, for a footstep).
    pub marker_sid: u32,
}

/// Every `snd!`/`lsnd` on an animation graph, with each one's sound class.
///
/// The same join `weapon_animation_sounds` uses, lifted out so a biped's graph
/// and a weapon's first-person graph share one implementation.
pub fn graph_sound_refs(native: &NativeDll, cache: CacheHandle, jmad: u32) -> Vec<(u32, u8)> {
    let refs = tag_block(native, cache, jmad, JMAD_SOUND_REFERENCES, JMAD_SOUND_REF_SIZE, 1024);
    let mut out = Vec::with_capacity(refs.len() / JMAD_SOUND_REF_SIZE);
    for c in refs.chunks_exact(JMAD_SOUND_REF_SIZE) {
        // An entry with no tag still occupies its slot: the INDEX is what an
        // event refers to, so a hole must stay a hole or every later event
        // plays the wrong sound.
        let Some(t) = tag_ref_at(c, 0) else {
            out.push((0u32, u8::MAX));
            continue;
        };
        match native.tag_class(cache, t).unwrap_or_default().as_str() {
            "snd!" => {
                let class = CacheSound::read(native, cache, t).map(|s| s.class).unwrap_or(u8::MAX);
                out.push((t, class));
            }
            "lsnd" => {
                let sd = LoopingSound::read(native, cache, t)
                    .and_then(|l| l.tracks.first().and_then(|tr| tr.loop_snd.or(tr.intro)));
                match sd {
                    Some(sd) => {
                        let class =
                            CacheSound::read(native, cache, sd).map(|s| s.class).unwrap_or(u8::MAX);
                        out.push((sd, class));
                    }
                    None => out.push((0, u8::MAX)),
                }
            }
            _ => out.push((0, u8::MAX)),
        }
    }
    out
}

/// The `sound events` of one animation, read with `stride`.
pub fn animation_sound_events_with(
    native: &NativeDll,
    cache: CacheHandle,
    jmad: u32,
    anim_index: usize,
    stride: usize,
) -> Vec<AnimSoundEvent> {
    // The two layouts differ in WHERE the fields are, not only in stride:
    //   8 B  `animation_sound_event_block`:          sound +0x00, frame +0x02, marker +0x04
    //  12 B  `animation_sound_event_block_extended`: frame event +0x00, sound +0x04,
    //                                                frame offset +0x06, marker +0x08
    let (o_sound, o_frame, o_marker) = if stride == SOUND_EVENT_SIZE_EXTENDED {
        (0x04usize, 0x06usize, 0x08usize)
    } else {
        (SE_SOUND, SE_FRAME, SE_MARKER)
    };
    let anims = tag_block(
        native,
        cache,
        jmad,
        crate::reach_jmad::OFF_ANIMATIONS,
        crate::reach_jmad::ANIMATION_ELEM,
        1 << 16,
    );
    let Some(a) = anims
        .chunks_exact(crate::reach_jmad::ANIMATION_ELEM)
        .nth(anim_index)
    else {
        return Vec::new();
    };
    let shared = block_at(
        native,
        cache,
        a,
        crate::reach_jmad::ANIM_SHARED_BLOCK,
        crate::reach_jmad::SHARED_ELEM,
        4,
    );
    if shared.len() < crate::reach_jmad::SHARED_ELEM {
        return Vec::new();
    }
    let ev = block_at(native, cache, &shared, SHARED_SOUND_EVENTS, stride, 256);
    ev.chunks_exact(stride)
        .map(|c| AnimSoundEvent {
            sound: u16::from_le_bytes([c[o_sound], c[o_sound + 1]]),
            frame: u16::from_le_bytes([c[o_frame], c[o_frame + 1]]),
            marker_sid: u32::from_le_bytes([
                c[o_marker],
                c[o_marker + 1],
                c[o_marker + 2],
                c[o_marker + 3],
            ]),
        })
        .collect()
}

/// The `sound events` of one animation, at the measured stride.
pub fn animation_sound_events(
    native: &NativeDll,
    cache: CacheHandle,
    jmad: u32,
    anim_index: usize,
) -> Vec<AnimSoundEvent> {
    animation_sound_events_with(native, cache, jmad, anim_index, SOUND_EVENT_SIZE)
}

/// The declared frame count of one animation.
pub fn animation_frame_count(
    native: &NativeDll,
    cache: CacheHandle,
    jmad: u32,
    anim_index: usize,
) -> Option<u16> {
    let anims = tag_block(
        native,
        cache,
        jmad,
        crate::reach_jmad::OFF_ANIMATIONS,
        crate::reach_jmad::ANIMATION_ELEM,
        1 << 16,
    );
    let a = anims
        .chunks_exact(crate::reach_jmad::ANIMATION_ELEM)
        .nth(anim_index)?;
    let shared = block_at(
        native,
        cache,
        a,
        crate::reach_jmad::ANIM_SHARED_BLOCK,
        crate::reach_jmad::SHARED_ELEM,
        4,
    );
    if shared.len() < crate::reach_jmad::SHARED_ELEM {
        return None;
    }
    Some(u16::from_le_bytes([
        shared[SHARED_FRAME_COUNT],
        shared[SHARED_FRAME_COUNT + 1],
    ]))
}

// ---------------------------------------------------------------------------------------------
// FOOTSTEPS, part 1: the TRIGGER — the animation's own frame events
// ---------------------------------------------------------------------------------------------
//
// A footstep is not on a timer and it is not a sound event. The animation says
// which foot lands and when, in `shared_model_animation_block +0x38 frame
// events` — 4-byte rows, `{short enum type, short frame}` — and
// `frame_event_type_new` names type **4 `left_foot`** and **5 `right_foot`**
// (with 8 `both-feet shuffle`). That is the engine's own left/right
// alternation and its own cadence, so it stays right at any playback speed.
//
// The SOUND is not in the animation at all: it is in the biped's material
// table, keyed by the surface the foot landed on. See part 2.

pub const SHARED_FRAME_EVENTS: usize = 0x38;
pub const FRAME_EVENT_SIZE: usize = 4;
pub const FE_TYPE: usize = 0x00;
pub const FE_FRAME: usize = 0x02;

/// `frame_event_type_new` values that are a foot landing.
pub const FRAME_EVENT_LEFT_FOOT: u16 = 4;
pub const FRAME_EVENT_RIGHT_FOOT: u16 = 5;
pub const FRAME_EVENT_BOTH_FEET: u16 = 8;

/// Which foot, and on which frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FootEvent {
    /// 4 = left, 5 = right, 8 = both (a shuffle).
    pub kind: u16,
    pub frame: u16,
}

impl FootEvent {
    pub fn is_left(&self) -> bool {
        self.kind == FRAME_EVENT_LEFT_FOOT
    }
    pub fn marker_hint(&self) -> &'static str {
        match self.kind {
            FRAME_EVENT_LEFT_FOOT => "left",
            FRAME_EVENT_RIGHT_FOOT => "right",
            _ => "both",
        }
    }
}

/// The foot-landing frame events of one animation, in frame order.
pub fn animation_foot_events(
    native: &NativeDll,
    cache: CacheHandle,
    jmad: u32,
    anim_index: usize,
) -> Vec<FootEvent> {
    let anims = tag_block(
        native,
        cache,
        jmad,
        crate::reach_jmad::OFF_ANIMATIONS,
        crate::reach_jmad::ANIMATION_ELEM,
        1 << 16,
    );
    let Some(a) = anims
        .chunks_exact(crate::reach_jmad::ANIMATION_ELEM)
        .nth(anim_index)
    else {
        return Vec::new();
    };
    let shared = block_at(
        native,
        cache,
        a,
        crate::reach_jmad::ANIM_SHARED_BLOCK,
        crate::reach_jmad::SHARED_ELEM,
        4,
    );
    if shared.len() < crate::reach_jmad::SHARED_ELEM {
        return Vec::new();
    }
    let ev = block_at(native, cache, &shared, SHARED_FRAME_EVENTS, FRAME_EVENT_SIZE, 512);
    let mut out: Vec<FootEvent> = ev
        .chunks_exact(FRAME_EVENT_SIZE)
        .map(|c| FootEvent {
            kind: u16::from_le_bytes([c[FE_TYPE], c[FE_TYPE + 1]]),
            frame: u16::from_le_bytes([c[FE_FRAME], c[FE_FRAME + 1]]),
        })
        .filter(|e| {
            matches!(
                e.kind,
                FRAME_EVENT_LEFT_FOOT | FRAME_EVENT_RIGHT_FOOT | FRAME_EVENT_BOTH_FEET
            )
        })
        .collect();
    out.sort_by_key(|e| e.frame);
    out
}

// ---------------------------------------------------------------------------------------------
// FOOTSTEPS, part 2: the SURFACE — render material to global material name
// ---------------------------------------------------------------------------------------------
//
// The biped's footstep sounds are keyed by a GLOBAL MATERIAL NAME (`hard`,
// `brittle`, `energy`, `hard_metal_solid`, ...), which is `matg +0x48C
// materials[].name`. A BSP surface reaches that name through its own tables:
//
// ```text
//   sbsp +0x0A8  materials            44 B  +0x00 tag reference render method
//   sbsp +0x024  collision materials  24 B  +0x00 tag reference render method
//                                           +0x10 short runtime global material index
//   matg +0x48C  materials           424 B  +0x00 string id name
// ```
//
// so a render material and a collision material that name the SAME shader are
// the same surface, and the collision row carries the global index. Nothing
// here reads a tag path or guesses from a name.

pub const SBSP_MATERIALS: usize = 0x0A8;
pub const SBSP_MATERIAL_SIZE: usize = 44;
pub const SBSP_COLLISION_MATERIALS: usize = 0x024;
pub const COLLISION_MATERIAL_SIZE: usize = 24;
pub const CM_RENDER_METHOD: usize = 0x00;
pub const CM_GLOBAL_MATERIAL_INDEX: usize = 0x10;
pub const MATG_MATERIALS: usize = 0x48C;
pub const MATG_MATERIAL_SIZE: usize = 424;

/// `matg`'s global material names, indexed by global material index.
pub fn global_material_names(native: &NativeDll, cache: CacheHandle) -> Vec<String> {
    let Some(matg) = native.tags_by_group(cache, "matg").first().copied() else {
        return Vec::new();
    };
    let b = tag_block(native, cache, matg, MATG_MATERIALS, MATG_MATERIAL_SIZE, 4096);
    b.chunks_exact(MATG_MATERIAL_SIZE)
        .map(|c| {
            let sid = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            native.string_id(cache, sid).unwrap_or_default()
        })
        .collect()
}

/// `shader tag -> global material name` for one BSP.
pub fn bsp_surface_materials(
    native: &NativeDll,
    cache: CacheHandle,
    sbsp: u32,
    global_names: &[String],
) -> std::collections::HashMap<u32, String> {
    let mut out = std::collections::HashMap::new();
    let b = tag_block(
        native,
        cache,
        sbsp,
        SBSP_COLLISION_MATERIALS,
        COLLISION_MATERIAL_SIZE,
        4096,
    );
    for c in b.chunks_exact(COLLISION_MATERIAL_SIZE) {
        let Some(shader) = tag_ref_at(c, CM_RENDER_METHOD) else { continue };
        let gi = i16::from_le_bytes([c[CM_GLOBAL_MATERIAL_INDEX], c[CM_GLOBAL_MATERIAL_INDEX + 1]]);
        if gi < 0 {
            continue;
        }
        if let Some(n) = global_names.get(gi as usize) {
            if !n.is_empty() {
                out.insert(shader, n.clone());
            }
        }
    }
    out
}

/// A BSP render material's shader tag, by render-material index.
pub fn bsp_render_material_shader(
    native: &NativeDll,
    cache: CacheHandle,
    sbsp: u32,
    index: usize,
) -> Option<u32> {
    let b = tag_block(native, cache, sbsp, SBSP_MATERIALS, SBSP_MATERIAL_SIZE, 8192);
    let c = b.chunks_exact(SBSP_MATERIAL_SIZE).nth(index)?;
    tag_ref_at(c, 0)
}

// ---------------------------------------------------------------------------------------------
// FOOTSTEPS, part 3: the SOUND — the biped's own material table
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// `scmb` — sound_combiner: several sounds played together as one
// ---------------------------------------------------------------------------------------------

/// `sound_combiner_block_struct +0x00 entries`, 24-byte rows: a tag reference
/// then a delay bounds in seconds.
///
/// This class matters because `obje +0xD0 melee sound` — "the sound that is
/// made when I am meleed" — is an **`scmb`** on a biped, not a `snd!`. A
/// resolver that only knows `snd!` and `lsnd` finds nothing there, which is
/// exactly how "melee a player" ends up silent.
pub const SCMB_ENTRIES: usize = 0x00;
pub const SCMB_ENTRY_SIZE: usize = 24;
pub const SCMB_ENTRY_DELAY: usize = 0x10;

/// Every `snd!` a sound combiner plays, with each one's delay in seconds.
pub fn combiner_sounds(native: &NativeDll, cache: CacheHandle, scmb: u32) -> Vec<(u32, f32)> {
    let b = tag_block(native, cache, scmb, SCMB_ENTRIES, SCMB_ENTRY_SIZE, 64);
    let mut out = Vec::new();
    for c in b.chunks_exact(SCMB_ENTRY_SIZE) {
        let Some(t) = tag_ref_at(c, 0) else { continue };
        let delay = f32::from_le_bytes([
            c[SCMB_ENTRY_DELAY],
            c[SCMB_ENTRY_DELAY + 1],
            c[SCMB_ENTRY_DELAY + 2],
            c[SCMB_ENTRY_DELAY + 3],
        ]);
        let delay = if delay.is_finite() && delay >= 0.0 { delay } else { 0.0 };
        match native.tag_class(cache, t).unwrap_or_default().as_str() {
            "snd!" => out.push((t, delay)),
            "lsnd" => {
                if let Some(l) = LoopingSound::read(native, cache, t) {
                    if let Some(sd) = l.tracks.first().and_then(|tr| tr.intro.or(tr.loop_snd)) {
                        out.push((sd, delay));
                    }
                }
            }
            "scmb" => {
                // Nested combiners are legal; one level is enough and bounds
                // a cycle in hostile data.
                for (sd, d2) in combiner_sounds_inner(native, cache, t) {
                    out.push((sd, delay + d2));
                }
            }
            _ => {}
        }
    }
    out
}

fn combiner_sounds_inner(native: &NativeDll, cache: CacheHandle, scmb: u32) -> Vec<(u32, f32)> {
    let b = tag_block(native, cache, scmb, SCMB_ENTRIES, SCMB_ENTRY_SIZE, 64);
    b.chunks_exact(SCMB_ENTRY_SIZE)
        .filter_map(|c| {
            let t = tag_ref_at(c, 0)?;
            (native.tag_class(cache, t).unwrap_or_default() == "snd!").then_some((t, 0.0))
        })
        .collect()
}

/// Resolve ANY sound-bearing tag class to a playable `snd!`.
///
/// The one place that knows which classes can stand in for a sound, so a new
/// caller cannot quietly omit one. `scmb` is why this exists.
pub fn resolve_to_snd(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<u32> {
    match native.tag_class(cache, tag).unwrap_or_default().as_str() {
        "snd!" => Some(tag),
        "lsnd" => LoopingSound::read(native, cache, tag)
            .and_then(|l| l.tracks.first().and_then(|tr| tr.intro.or(tr.loop_snd))),
        "scmb" => combiner_sounds(native, cache, tag).first().map(|(t, _)| *t),
        "effe" => effect_sounds(native, cache, tag)
            .into_iter()
            .find(|(_, looping)| !*looping)
            .map(|(t, _)| t),
        _ => None,
    }
}

/// The sound an object makes **when it is meleed** (`obje +0xD0`).
///
/// The engine's own words: "this is the sound that is made when I am meleed.
/// This overrides the sweetener sound of my material." On a Spartan and an
/// Elite it is `sound\materials\soft\organic_flesh\soft_organic_flesh`, and it
/// is an **`scmb`** — which is why `resolve_to_snd` has to know that class.
/// The Forge prop set authors its own, one per prop, so meleeing a pallet
/// sounds like a pallet.
pub fn melee_victim_sound(native: &NativeDll, cache: CacheHandle, obj: u32) -> Option<u32> {
    let t = crate::reach_damage::melee_sound(native, cache, obj)?;
    resolve_to_snd(native, cache, t)
}

/// The footstep `snd!` a biped makes on one surface.
///
/// `obje +0xC0 material effects` names a `foot` tag whose rows are keyed by a
/// global material name; `reach_damage::material_responses` reads them (one
/// reader, not two). A Spartan authors **424** such sounds, e.g.
/// `hard -> run/concrete_h3/chief` and `brittle -> walk/glass/chief`.
///
/// A row whose tag is absent is AUTHORED, not a decode error — a Spartan's
/// `default_material` row names no sound at all — so an unmatched or empty row
/// means silence rather than a substituted sound. Likewise a row may carry an
/// `effe` rather than a `snd!`, because the same row shape serves both the
/// `sounds` and the `effects` block, so the class is checked.
pub fn footstep_sound(
    native: &NativeDll,
    cache: CacheHandle,
    biped: u32,
    material: &str,
) -> Option<u32> {
    let (sounds, _effects) = crate::reach_damage::material_responses(native, cache, biped)?;
    let row = sounds
        .iter()
        .find(|r| r.material.eq_ignore_ascii_case(material))?;
    for t in [row.tag, row.secondary].into_iter().flatten() {
        if let Some(sd) = resolve_to_snd(native, cache, t) {
            return Some(sd);
        }
    }
    None
}

/// `object_struct_definition +0x0D0`: "melee sound — this is the sound that
/// is made when I am used to melee". Present on every object.
pub const OBJE_MELEE_SOUND: usize = 0x0D0;

/// Weapon fields, ABSOLUTE in a `weap` tag. `weap` composes as
/// object(360) + item(156) + weapon(692) = 1208, so a weapon-relative
/// `+0xN` is at `516 + N`.
pub const WEAPON_BASE: usize = 516;
pub const WEAP_PICKUP_SOUND: usize = WEAPON_BASE + 0x170;
pub const WEAP_ZOOM_IN_SOUND: usize = WEAPON_BASE + 0x180;
pub const WEAP_ZOOM_OUT_SOUND: usize = WEAPON_BASE + 0x190;
pub const WEAP_BARRELS: usize = WEAPON_BASE + 0x204;
pub const WEAP_SIZE: usize = 1208;

#[cfg(test)]
mod tests {
    use super::*;

    fn playback_bytes(gain_base: f32, maximum: f32) -> Vec<u8> {
        let mut b = vec![0u8; PLAYBACK_SIZE];
        b[PB_SKIP_FRACTION..PB_SKIP_FRACTION + 4].copy_from_slice(&0.0f32.to_le_bytes());
        let d = PB_DISTANCES;
        // dont_obstruct, dont_play, attack, minimum, sustain_begin, sustain_end, maximum
        for (i, v) in [0.0f32, 0.0, 0.5, 2.0, 10.0, 20.0, maximum].iter().enumerate() {
            b[d + i * 4..d + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        b[d + 0x1C..d + 0x20].copy_from_slice(&(-6.0f32).to_le_bytes());
        b[PB_GAIN_BASE_DB..PB_GAIN_BASE_DB + 4].copy_from_slice(&gain_base.to_le_bytes());
        b
    }

    #[test]
    fn a_playback_row_parses_and_the_distances_come_out_in_order() {
        let b = playback_bytes(-3.0, 40.0);
        let p = Playback::parse(&b).unwrap();
        assert_eq!(p.gain_base_db, -3.0);
        assert_eq!(p.dist.attack, 0.5);
        assert_eq!(p.dist.minimum, 2.0);
        assert_eq!(p.dist.sustain_begin, 10.0);
        assert_eq!(p.dist.sustain_end, 20.0);
        assert_eq!(p.dist.maximum, 40.0);
        assert_eq!(p.dist.sustain_db, -6.0);
        assert!(p.dist.usable());
        assert!(p.plausible());
    }

    #[test]
    fn a_zeroed_or_out_of_order_distance_row_is_not_treated_as_a_curve() {
        assert!(!Distances::default().usable(), "all zeroes is not a curve");
        let mut d = Distances { maximum: 10.0, minimum: 5.0, sustain_begin: 1.0, ..Default::default() };
        assert!(!d.usable(), "knots out of order");
        d.sustain_begin = 6.0;
        d.sustain_end = 8.0;
        d.attack = 1.0;
        assert!(d.usable());
        d.maximum = f32::NAN;
        assert!(!d.usable());
    }

    /// Every vehicle engine loop authors `minimum 7 / maximum 40` and nothing else, which is the
    /// two-knot curve `(7, 1) -> (40, 0)`: unset knots are not knots (`curve` skips them), so such
    /// a row is usable and must attenuate. It used to be refused, and those sounds played at full
    /// volume from any distance.
    #[test]
    fn a_row_that_leaves_the_sustain_knots_unset_is_still_a_curve() {
        let d = Distances { minimum: 7.0, maximum: 40.0, ..Default::default() };
        assert!(d.usable());
        assert_eq!(d.curve().count, 2);
        assert_eq!(d.gain_at(3.0), 1.0, "flat inside the minimum distance");
        assert!(d.gain_at(20.0) < 0.5 && d.gain_at(20.0) > 0.0, "falling between");
        assert_eq!(d.gain_at(45.0), 0.0, "silent past the maximum");
    }

    #[test]
    fn a_truncated_struct_is_refused_rather_than_read_past() {
        assert!(Playback::parse(&[0u8; PLAYBACK_SIZE - 1]).is_none());
        assert!(Distances::parse(&[0u8; DIST_SIZE - 1]).is_none());
        assert!(tag_ref_at(&[0u8; TAG_REF_SIZE - 1], 0).is_none());
    }

    #[test]
    fn the_engine_clamps_gain_at_unity_and_minus_sixty_four() {
        // sub_1809730B0 / sub_180973360: 0 dB ceiling, -64 dB floor.
        assert_eq!(linear_to_db(1.0), 0.0);
        assert_eq!(linear_to_db(4.0), 0.0, "nothing plays above unity");
        assert_eq!(linear_to_db(0.0), MIN_GAIN_DB);
        assert_eq!(linear_to_db(-1.0), MIN_GAIN_DB);
        assert_eq!(linear_to_db(1e-9), MIN_GAIN_DB, "below the floor is the floor");
        assert!((linear_to_db(0.5) - -6.0206).abs() < 1e-3);
        // sub_180973340: 10^(db/20), not 2^(db/6).
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear(-6.0206) - 0.5).abs() < 1e-4);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-6);
        assert!(db_to_linear(MIN_GAIN_DB) < 1e-3);
    }

    #[test]
    fn a_null_tag_reference_is_none_not_zero() {
        let mut b = vec![0u8; TAG_REF_SIZE];
        b[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(tag_ref_at(&b, 0).is_none());
        b[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert!(tag_ref_at(&b, 0).is_none());
        b[12..16].copy_from_slice(&0x1234u32.to_le_bytes());
        assert_eq!(tag_ref_at(&b, 0), Some(0x1234));
    }

    #[test]
    fn the_sound_class_table_is_the_enum_the_tag_definition_lists() {
        assert_eq!(sound_class_name(4), "weapon_fire");
        assert_eq!(sound_class_name(6), "weapon_reload");
        assert_eq!(sound_class_name(18), "unit_footsteps");
        assert_eq!(sound_class_name(200), "?");
    }
}

// ---------------------------------------------------------------------------------------------
// Probes — run against the installed game, and the derivation of the one
// encoded field in this module.
// ---------------------------------------------------------------------------------------------
/// Shared test scaffolding: one Reach cache opened once for the whole test
/// binary, and the list of Reach maps on this machine.
#[cfg(test)]
pub mod probes_support {
    use super::*;

    /// Serialises every probe that opens a cache or the audio device.
    ///
    /// The native parser keeps GLOBAL state (`set_shared_cache_dirs` is
    /// process-wide -- see `re-notes/native_parser_global_state.md`) and
    /// there is one audio device, so two of these running at once is a race
    /// and the failures look like decode bugs. `cargo test` runs the binary's
    /// tests in parallel by default, so a probe that passes alone and fails in
    /// the suite is not a flake to be retried: it is this.
    pub static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    static OWN_CACHE: std::sync::OnceLock<
        Option<(std::sync::Arc<hms_native::NativeDll>, hms_native::CacheHandle, std::path::PathBuf)>,
    > = std::sync::OnceLock::new();

    /// Open one Reach cache and keep it for the whole test binary. Returns
    /// `None`, with a printed reason, when the game is not installed.
    pub fn open_reach(stem: &str) -> Option<(std::sync::Arc<NativeDll>, CacheHandle, std::path::PathBuf)> {
        OWN_CACHE
            .get_or_init(|| {
                let mut roots: Vec<std::path::PathBuf> = Vec::new();
                if let Ok(exe) = std::env::current_exe() {
                    let mut d = exe.parent().map(|p| p.to_path_buf());
                    for _ in 0..3 {
                        if let Some(p) = &d {
                            roots.push(p.clone());
                            d = p.parent().map(|q| q.to_path_buf());
                        }
                    }
                }
                let dll = roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file())?;
                let native = std::sync::Arc::new(NativeDll::load(&dll).ok()?);
                let cands = crate::mapcat::enumerate();
                let map = cands.iter().find(|c| {
                    c.game == crate::mapcat::Game::Reach
                        && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(stem))
                })?;
                if let Some(dir) = map.path.parent() {
                    native.set_shared_cache_dirs(&[dir.to_path_buf()]);
                }
                let cache = native.open_cache(&map.path.to_string_lossy()).ok()?;
                Some((native, cache, map.path.clone()))
            })
            .clone()
    }

    /// Every Reach map on this machine, for the wider sweeps.
    pub fn reach_maps() -> Vec<std::path::PathBuf> {
        crate::mapcat::enumerate()
            .into_iter()
            .filter(|c| c.game == crate::mapcat::Game::Reach)
            .map(|c| c.path)
            .collect()
    }
}

#[cfg(test)]
mod probes {
    use super::probes_support::*;
    use super::*;

    /// THE DERIVATION of `encoded first permutation and counts`.
    ///
    /// The field has one property that no wrong split can fake: within a
    /// gestalt the pitch ranges must **TILE** the permutation block — the
    /// first starts at 0, each one's `first + count` is the next one's
    /// `first`, and the last ends at `permutations.len()`. Over a hundred
    /// thousand pitch ranges that is a hundred thousand simultaneous
    /// equations, so a wrong bit split cannot satisfy it.
    ///
    /// **It is derived over every Reach cache on the machine at once**,
    /// because no single map settles the widths: `forge_halo` (9,497
    /// permutations) fits a 14-bit `first` and never sets bits 14..19,
    /// `cex_ff_halo` (26,024) does not, and `m45` needs a count wider than
    /// six bits. Deriving on one map and shipping the result is exactly how a
    /// constant that works on the test map and fails in production gets in.
    ///
    /// This is a derivation AND a gate: it ends by asserting that the
    /// production `decode_first_permutation_and_counts` agrees with a
    /// surviving split on every word of every map.
    #[test]
    fn probe_pitch_range_encoding() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let maps = reach_maps();
        let Some((native, _c, _p)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        if maps.is_empty() {
            eprintln!("reach_sound: no Reach maps on this machine, skipping");
            return;
        }
        // Collect (map name, encoded words, permutation count) for every map.
        let mut corpus: Vec<(String, Vec<u32>, usize)> = Vec::new();
        for m in &maps {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            let tag = native.tags_by_group(cache, "ugh!").first().copied();
            if let Some(tag) = tag {
                let pr = tag_block(&native, cache, tag, UGH_PITCH_RANGES, PITCH_RANGE_SIZE, MAX_BLOCK);
                let pm = tag_block(&native, cache, tag, UGH_PERMUTATIONS, PERMUTATION_SIZE, MAX_BLOCK);
                if !pr.is_empty() && !pm.is_empty() {
                    let enc: Vec<u32> = pr
                        .chunks_exact(PITCH_RANGE_SIZE)
                        .map(|c| {
                            u32::from_le_bytes([
                                c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS],
                                c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS + 1],
                                c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS + 2],
                                c[PR_ENCODED_FIRST_PERMUTATION_AND_COUNTS + 3],
                            ])
                        })
                        .collect();
                    corpus.push((
                        m.file_stem().unwrap().to_string_lossy().into_owned(),
                        enc,
                        pm.len() / PERMUTATION_SIZE,
                    ));
                }
            }
            native.close_cache(cache);
        }
        assert!(!corpus.is_empty(), "no gestalt read from any map");
        let words: usize = corpus.iter().map(|(_, e, _)| e.len()).sum();
        let perms: usize = corpus.iter().map(|(_, _, n)| *n).sum();
        eprintln!("{} gestalts, {words} pitch ranges, {perms} permutations", corpus.len());
        let union = corpus.iter().flat_map(|(_, e, _)| e.iter()).fold(0u32, |a, v| a | *v);
        eprintln!(
            "union of every encoded word: 0x{union:08X}  (bits never set: {})",
            (0..32).filter(|b| union & (1u32 << b) == 0).map(|b| b.to_string()).collect::<Vec<_>>().join(",")
        );

        // Does one split tile EVERY map?
        // A row whose first index is NEGATIVE (the engine sign-extends it) or
        // whose count is zero names no permutations, so it contributes
        // nothing to the tiling rather than breaking it.
        let tile_one = |fs: u32, fw: u32, cs: u32, cw: u32, enc: &[u32], nperm: usize| -> Result<(), usize> {
            let fmask = (1u64 << fw) - 1;
            let cmask = (1u64 << cw) - 1;
            let sign = 1u64 << (fw - 1);
            let mut cursor = 0usize;
            for (i, e) in enc.iter().enumerate() {
                let raw = (*e as u64 >> fs) & fmask;
                let count = ((*e as u64 >> cs) & cmask) as usize;
                if raw & sign != 0 || count == 0 {
                    continue;
                }
                if raw as usize != cursor {
                    return Err(i);
                }
                cursor += count;
            }
            if cursor == nperm {
                Ok(())
            } else {
                Err(enc.len())
            }
        };
        let tiles = |fs: u32, fw: u32, cs: u32, cw: u32| -> bool {
            corpus.iter().all(|(_, enc, n)| tile_one(fs, fw, cs, cw, enc, *n).is_ok())
        };
        let mut winners: Vec<(u32, u32, u32, u32)> = Vec::new();
        for fs in 0..24u32 {
            for fw in 8..=24u32 {
                if fs + fw > 32 {
                    continue;
                }
                for cs in 0..32u32 {
                    for cw in 4..=16u32 {
                        if cs + cw > 32 || (cs < fs + fw && cs + cw > fs) {
                            continue; // overlapping fields
                        }
                        if tiles(fs, fw, cs, cw) {
                            winners.push((fs, fw, cs, cw));
                        }
                    }
                }
            }
        }
        if winners.is_empty() {
            // Say WHICH gestalt the best candidate fails on, rather than just
            // that nothing worked: the whole point of the sweep is to name the
            // map that pins a width.
            let mut best = (0usize, (0u32, 0u32, 0u32, 0u32));
            for fs in 0..24u32 {
                for fw in 8..=24u32 {
                    if fs + fw > 32 { continue; }
                    for cs in 0..32u32 {
                        for cw in 4..=16u32 {
                            if cs + cw > 32 || (cs < fs + fw && cs + cw > fs) { continue; }
                            let n = corpus
                                .iter()
                                .filter(|(_, e, np)| tile_one(fs, fw, cs, cw, e, *np).is_ok())
                                .count();
                            if n > best.0 {
                                best = (n, (fs, fw, cs, cw));
                            }
                        }
                    }
                }
            }
            let (fs, fw, cs, cw) = best.1;
            eprintln!(
                "BEST split first=bits[{fs}..{}] count=bits[{cs}..{}] tiles {} of {} gestalts; failures:",
                fs + fw,
                cs + cw,
                best.0,
                corpus.len()
            );
            for (name, enc, np) in &corpus {
                if let Err(i) = tile_one(fs, fw, cs, cw, enc, *np) {
                    let w = enc.get(i).copied().unwrap_or(0);
                    eprintln!("  {name:<28} fails at pitch range {i} of {} (word 0x{w:08X}), {np} permutations", enc.len());
                }
            }
            panic!("no bit split tiles the permutation block on all {} gestalts", corpus.len());
        }
        eprintln!(
            "{} splits tile every gestalt; narrowest first = bits[{}..{}], count = bits[{}..{}]",
            winners.len(),
            winners[0].0,
            winners[0].0 + winners[0].1,
            winners[0].2,
            winners[0].2 + winners[0].3
        );
        // Every surviving split must agree value-for-value, or the widths are
        // not pinned and the constant is not safe to ship.
        for (name, enc, _) in &corpus {
            for e in enc {
                let mut seen: Option<(usize, usize)> = None;
                for (fs, fw, cs, cw) in &winners {
                    let v = (
                        ((*e as u64 >> fs) & ((1u64 << fw) - 1)) as usize,
                        ((*e as u64 >> cs) & ((1u64 << cw) - 1)) as usize,
                    );
                    match seen {
                        None => seen = Some(v),
                        Some(prev) => assert_eq!(prev, v, "{name}: splits disagree on 0x{e:08X}"),
                    }
                }
                // THE GATE: the shipped function must be one of them.
                assert_eq!(
                    decode_first_permutation_and_counts(*e),
                    seen.unwrap(),
                    "{name}: decode_first_permutation_and_counts disagrees with the measured \
                     split on 0x{e:08X}"
                );
            }
        }

        // And the production reader must tile, with nothing rejected, on every map.
        let mut ok_maps = 0usize;
        for m in &maps {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            if let Some(g) = Gestalt::read(&native, cache) {
                assert_eq!(
                    g.rejected_pitch_ranges,
                    0,
                    "{}: the production walk rejected rows",
                    m.file_stem().unwrap().to_string_lossy()
                );
                let mut cursor = 0usize;
                for pr in &g.pitch_ranges {
                    if pr.permutation_count == 0 {
                        continue; // names no permutations
                    }
                    assert_eq!(
                        pr.first_permutation,
                        cursor,
                        "{}: pitch range {} does not tile",
                        m.file_stem().unwrap().to_string_lossy(),
                        pr.index
                    );
                    cursor += pr.permutation_count;
                }
                assert_eq!(
                    cursor,
                    g.permutations.len(),
                    "{}: the tiling does not cover the block",
                    m.file_stem().unwrap().to_string_lossy()
                );
                ok_maps += 1;
            }
            native.close_cache(cache);
        }
        eprintln!("the production decode tiles exactly on {ok_maps} Reach gestalts");
        assert!(ok_maps >= 5, "only {ok_maps} gestalts checked");
    }

    /// The production gestalt reader, against every Reach map on the machine
    /// and against the real FMOD banks. This replaces the raw-image scan in
    /// `hms_sound::scan_cache_for_subsound_ids` as the proof of the join.
    #[test]
    fn probe_gestalt_joins_every_map_to_the_banks() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let maps = reach_maps();
        if maps.is_empty() {
            eprintln!("reach_sound: no Reach maps on this machine, skipping");
            return;
        }
        let Some((native, _c, any)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let Some(dir) = hms_sound::SoundBank::bank_dir_for_reach(&any) else {
            eprintln!("reach_sound: no fmod/pc beside {}", any.display());
            return;
        };
        let sfx = hms_sound::SoundBank::open(&dir, "sfx").expect("open sfx");
        let eng = hms_sound::SoundBank::open(&dir, "english").ok();
        let mut maps_done = 0usize;
        let mut tot_snd = 0usize;
        let mut tot_perm = 0usize;
        let mut tot_hit = 0usize;
        let mut tot_frames_agree = 0usize;
        let mut tot_playback = 0usize;
        let mut tot_playback_ok = 0usize;
        for m in maps.iter().take(40) {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            let Some(g) = Gestalt::read(&native, cache) else {
                native.close_cache(cache);
                continue;
            };
            let snds = sound_tags(&native, cache);
            let mut hit = 0usize;
            let mut perms = 0usize;
            let mut frames_agree = 0usize;
            for t in &snds {
                let Some(s) = CacheSound::read(&native, cache, *t) else { continue };
                if g.playback_of(&s).is_some() {
                    tot_playback += 1;
                    if g.playback_of(&s).unwrap().plausible() {
                        tot_playback_ok += 1;
                    }
                }
                for p in g.permutations_of(&s) {
                    perms += 1;
                    let rec = sfx
                        .index_of_subsound(p.fmod_subsound_id)
                        .and_then(|i| sfx.info.as_ref().unwrap().get(i))
                        .or_else(|| {
                            eng.as_ref().and_then(|e| {
                                e.index_of_subsound(p.fmod_subsound_id)
                                    .and_then(|i| e.info.as_ref().unwrap().get(i))
                            })
                        });
                    if let Some(r) = rec {
                        hit += 1;
                        if let Some(f) = r.frames() {
                            if (p.sample_count as i64 - f as i64).abs() <= 1 {
                                frames_agree += 1;
                            }
                        }
                    }
                }
            }
            eprintln!(
                "{:<28} {:>4} snd!  {:>5} perms  {:>5} resolved ({:5.1}%)  {:>5} frame-agree  {} rejected PRs",
                m.file_stem().unwrap().to_string_lossy(),
                snds.len(),
                perms,
                hit,
                100.0 * hit as f64 / perms.max(1) as f64,
                frames_agree,
                g.rejected_pitch_ranges
            );
            assert_eq!(g.rejected_pitch_ranges, 0, "{}: rejected pitch ranges", m.display());
            tot_snd += snds.len();
            tot_perm += perms;
            tot_hit += hit;
            tot_frames_agree += frames_agree;
            maps_done += 1;
            native.close_cache(cache);
        }
        eprintln!(
            "TOTAL {maps_done} maps: {tot_snd} $#!+, {tot_perm} permutations, {tot_hit} resolved \
             ({:.2}%), {tot_frames_agree} frame-agree ({:.2}%), playbacks {tot_playback_ok}/{tot_playback} plausible",
            100.0 * tot_hit as f64 / tot_perm.max(1) as f64,
            100.0 * tot_frames_agree as f64 / tot_perm.max(1) as f64,
        );
        assert!(maps_done > 0, "no map opened");
        assert!(tot_perm > 1000, "only {tot_perm} permutations walked");
        // The join is either right or it is noise: a wrong stride or a wrong
        // offset resolves essentially nothing.
        let hit_rate = tot_hit as f64 / tot_perm as f64;
        assert!(hit_rate > 0.98, "only {:.2}% of permutations resolved", 100.0 * hit_rate);
        let agree = tot_frames_agree as f64 / tot_perm as f64;
        assert!(
            agree > 0.95,
            "only {:.2}% of permutations' sample counts matched their bank sample",
            100.0 * agree
        );
        // And the gain law must be real data, not zeroes.
        assert!(tot_playback > 100, "only {tot_playback} playback rows");
        assert!(
            tot_playback_ok as f64 / tot_playback as f64 > 0.9,
            "only {tot_playback_ok} of {tot_playback} playback rows are plausible"
        );
    }

    /// What the gain law actually says, so the numbers are on the record.
    #[test]
    fn probe_gain_law_census() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let Some((native, cache, path)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let Some(g) = Gestalt::read(&native, cache) else {
            eprintln!("reach_sound: {} has no gestalt", path.display());
            return;
        };
        eprintln!(
            "{}: {} playbacks, {} scales, {} pitch ranges, {} permutations, {} import names",
            path.file_stem().unwrap().to_string_lossy(),
            g.playbacks.len(),
            g.scale_gain_db.len(),
            g.pitch_ranges.len(),
            g.permutations.len(),
            g.import_names.len()
        );
        let pct = |v: &mut Vec<f32>, q: f64| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[(((v.len() - 1) as f64) * q) as usize]
        };
        let mut maxd: Vec<f32> = g.playbacks.iter().map(|p| p.dist.maximum).collect();
        let mut mind: Vec<f32> = g.playbacks.iter().map(|p| p.dist.minimum).collect();
        let mut base: Vec<f32> = g.playbacks.iter().map(|p| p.gain_base_db).collect();
        let mut var: Vec<f32> = g.playbacks.iter().map(|p| p.gain_variance_db).collect();
        let mut sus: Vec<f32> = g.playbacks.iter().map(|p| p.dist.sustain_db).collect();
        for (name, v) in [
            ("minimum distance  (wu)", &mut mind),
            ("maximum distance  (wu)", &mut maxd),
            ("gain base      (dB)", &mut base),
            ("gain variance  (dB)", &mut var),
            ("sustain db     (dB)", &mut sus),
        ] {
            if v.is_empty() {
                continue;
            }
            eprintln!(
                "  {name:<22} min {:>9.3}  p25 {:>9.3}  median {:>9.3}  p75 {:>9.3}  max {:>9.3}",
                pct(v, 0.0),
                pct(v, 0.25),
                pct(v, 0.5),
                pct(v, 0.75),
                pct(v, 1.0)
            );
        }
        let mut scales: Vec<f32> = g.scale_gain_db.iter().map(|(a, _)| *a).collect();
        if !scales.is_empty() {
            eprintln!(
                "  scale gain lower (dB)  min {:>9.3}  median {:>9.3}  max {:>9.3}   ({} rows)",
                pct(&mut scales, 0.0),
                pct(&mut scales, 0.5),
                pct(&mut scales, 1.0),
                scales.len()
            );
        }
        let mut gains: std::collections::BTreeMap<i8, usize> = Default::default();
        for p in &g.permutations {
            *gains.entry(p.gain_db).or_default() += 1;
        }
        eprintln!("  permutation encoded gain dB histogram: {gains:?}");
        // The gain law has to be DATA, not a block of zeroes: if the playback
        // offsets were wrong these would all read the same.
        let distinct: std::collections::BTreeSet<u32> =
            g.playbacks.iter().map(|p| p.dist.maximum.to_bits()).collect();
        assert!(
            distinct.len() > 5,
            "only {} distinct maximum distances across {} playbacks — the offsets are probably wrong",
            distinct.len(),
            g.playbacks.len()
        );
        // And the class spread must be real: a map has weapon fire, footsteps
        // and dialog, not one class.
        let snds = sound_tags(&native, cache);
        let mut classes: std::collections::BTreeMap<&str, usize> = Default::default();
        for t in &snds {
            if let Some(s) = CacheSound::read(&native, cache, *t) {
                *classes.entry(s.class_name()).or_default() += 1;
            }
        }
        eprintln!("  snd! sound class histogram ({} tags): {classes:?}", snds.len());
        assert!(classes.len() > 3, "only {} sound classes: {classes:?}", classes.len());
    }
}

#[cfg(test)]
mod probes2 {
    use super::*;
    use super::probes_support::*;

    /// THE WEAPON GATE: every weapon on every map must reach a bank sample,
    /// and the sample's asset path is printed so it can be read.
    #[test]
    fn probe_every_weapon_reaches_a_bank_sample() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let Some((native, _c, any)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let Some(dir) = hms_sound::SoundBank::bank_dir_for_reach(&any) else {
            eprintln!("reach_sound: no fmod/pc beside {}", any.display());
            return;
        };
        // EVERY bank beside the module, not just `sfx` and `english`: a
        // `snd!` names its bank with a string id and a sound whose samples
        // are in one of the other five looks identical to a broken chain.
        let mut banks: Vec<(String, hms_sound::SoundBank)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) != Some("fsb") {
                    continue;
                }
                let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
                if let Ok(b) = hms_sound::SoundBank::open(&dir, &stem) {
                    banks.push((stem, b));
                }
            }
        }
        banks.sort_by(|a, b| a.0.cmp(&b.0));
        eprintln!(
            "banks: {}",
            banks
                .iter()
                .map(|(n, b)| format!("{n} ({})", b.bank.samples.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let sfx = &banks
            .iter()
            .find(|(n, _)| n == "sfx")
            .expect("the sfx bank")
            .1;
        let path_of = |id: u32| -> Option<String> {
            banks.iter().find_map(|(n, b)| {
                b.index_of_subsound(id)
                    .and_then(|i| b.asset_path(i).map(|s| format!("{n}:{s}")))
            })
        };

        let mut tot_barrels = 0usize;
        let mut tot_resolved = 0usize;
        let mut tot_sampled = 0usize;
        let mut by_source = [0usize; 2];
        let mut maps = 0usize;
        for m in reach_maps().iter().take(8) {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            let Some(g) = Gestalt::read(&native, cache) else {
                native.close_cache(cache);
                continue;
            };
            let law = GainLaw::read(&native, cache);
            let weaps = native.tags_by_group(cache, "weap");
            let mut rows: Vec<String> = Vec::new();
            let mut armed = 0usize;
            let mut resolved = 0usize;
            let mut sampled = 0usize;
            for w in &weaps {
                let fires = weapon_fire_sound(&native, cache, *w);
                let barrels = tag_block(&native, cache, *w, WEAP_BARRELS_ABS, BARREL_SIZE, BARREL_MAX);
                let nb = barrels.len() / BARREL_SIZE;
                if nb == 0 && fires.is_empty() {
                    continue; // a melee or objective item: correctly no chain
                }
                armed += nb.max(1);
                for f in &fires {
                    resolved += 1;
                    by_source[if f.source == FireSource::BarrelEffect { 0 } else { 1 }] += 1;
                    let snd = CacheSound::read(&native, cache, f.snd).unwrap_or_default();
                    let perms = g.permutations_of(&snd);
                    let paths: Vec<String> = perms.iter().filter_map(|p| path_of(p.fmod_subsound_id)).collect();
                    if !paths.is_empty() {
                        sampled += 1;
                    }
                    let gain = law.authored_gain_db(&snd, g.playback_of(&snd), g.scale_gain_of(&snd), perms.first().map(|p| p.gain_db).unwrap_or(0));
                    rows.push(format!(
                        "    {:<34} b{} {:<13} {:<22} {:>3} perms  gain {:>6.1} dB  {}",
                        native.tag_name(cache, *w).unwrap_or_default().rsplit('\\').next().unwrap_or("?").to_string(),
                        f.barrel,
                        if f.source == FireSource::BarrelEffect { "barrel-effect" } else { "lsnd-attach" },
                        sound_class_name(f.class),
                        perms.len(),
                        gain,
                        paths.first().map(|p| p.rsplit('\\').next().unwrap_or("").to_string()).unwrap_or_else(|| "NO SAMPLE".into())
                    ));
                }
            }
            eprintln!(
                "{}: {} weap, {armed} barrels, {resolved} firing chains, {sampled} reached a sample",
                m.file_stem().unwrap().to_string_lossy(),
                weaps.len()
            );
            if maps == 0 {
                for r in rows.iter().take(60) {
                    eprintln!("{r}");
                }
            }
            tot_barrels += armed;
            tot_resolved += resolved;
            tot_sampled += sampled;
            maps += 1;
            native.close_cache(cache);
        }
        eprintln!(
            "TOTAL {maps} maps: {tot_barrels} barrels, {tot_resolved} firing chains resolved, \
             {tot_sampled} reached a bank sample ({} barrel-effect, {} lsnd-attachment)",
            by_source[0], by_source[1]
        );

        // The OTHER weapon sounds the user asked for, measured the same way:
        // every weapon of every map, through to a real bank sample.
        let mut tally = [[0usize; 2]; 4]; // [melee, clang, reload, empty] x [resolved, sampled]
        let mut shown_clang = 0usize;
        let mut weapons = 0usize;
        for m in reach_maps().iter().take(8) {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            let Some(g) = Gestalt::read(&native, cache) else {
                native.close_cache(cache);
                continue;
            };
            for w in native.tags_by_group(cache, "weap") {
                weapons += 1;
                let got: [Vec<u32>; 4] = [
                    weapon_melee_sound(&native, cache, w, 0).into_iter().collect(),
                    weapon_clang_sounds(&native, cache, w),
                    weapon_reload_sound(&native, cache, w, 0).into_iter().collect(),
                    weapon_empty_sound(&native, cache, w).into_iter().collect(),
                ];
                for (k, cands) in got.iter().enumerate() {
                    if cands.is_empty() {
                        continue;
                    }
                    tally[k][0] += 1;
                    let any = cands.iter().any(|t| {
                        CacheSound::read(&native, cache, *t).is_some_and(|snd| {
                            g.permutations_of(&snd)
                                .iter()
                                .any(|p| path_of(p.fmod_subsound_id).is_some())
                        })
                    });
                    if any {
                        tally[k][1] += 1;
                    } else if k == 1 && shown_clang < 2 {
                        shown_clang += 1;
                        let eff = native
                            .tag_meta(cache, w, WEAP_CLANG_EFFECT as u32, TAG_REF_SIZE as u32)
                            .and_then(|b| tag_ref_at(&b, 0));
                        eprintln!(
                            "    clang MISS {} -> effe {:?} ({:?}); candidates {:?}",
                            native.tag_name(cache, w).unwrap_or_default().rsplit('\\').next().unwrap_or("?"),
                            eff.and_then(|e| native.tag_name(cache, e)).map(|n| n.rsplit('\\').next().unwrap_or("?").to_string()),
                            eff,
                            cands
                                .iter()
                                .map(|t| (
                                    native.tag_class(cache, *t).unwrap_or_default(),
                                    native.tag_name(cache, *t).unwrap_or_default().rsplit('\\').next().unwrap_or("?").to_string(),
                                    CacheSound::read(&native, cache, *t)
                                        .map(|sn| g.permutations_of(&sn).len())
                                        .unwrap_or(999)
                                ))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
            native.close_cache(cache);
        }
        for (k, name) in ["melee swing", "clang", "reload", "empty click"].iter().enumerate() {
            eprintln!(
                "  {name:<12} {:>4} of {weapons} weapons name one, {:>4} reach a bank sample",
                tally[k][0], tally[k][1]
            );
        }
        // The melee swing, the reload and the dry-fire click must be
        // all-or-nothing: a weapon that names one must reach a sample, and a
        // partial rate would mean an offset is wrong.
        for k in [0usize, 2, 3] {
            let name = ["melee swing", "clang", "reload", "empty click"][k];
            assert_eq!(
                tally[k][0], tally[k][1],
                "{name}: {} weapons name one but only {} reach a bank sample",
                tally[k][0], tally[k][1]
            );
        }
        assert!(tally[0][1] > 20, "only {} melee swings resolved", tally[0][1]);
        assert!(tally[2][1] > 20, "only {} reload sounds resolved", tally[2][1]);
        assert!(tally[3][1] > 20, "only {} empty clicks resolved", tally[3][1]);

        // THE CLANG IS DIFFERENT, and the difference is in the shipped data,
        // not in the chain. 230 of 450 weapons name a clang effect, but 168
        // of those name `fx\generic_melee_clash`, whose `snd!` has five
        // permutations and **not one of their subsound ids appears in ANY of
        // the seven banks** (checked above against every `.fsb` beside the
        // module, not just `sfx` and `english`). So the generic clang cannot
        // be played at all; the 62 that do resolve are the dedicated melee
        // weapons. This is part of the same 0.1% residue the gestalt join
        // leaves (425,933 of 426,350 permutations resolve across 40 maps).
        assert!(tally[1][0] > 200, "only {} weapons name a clang effect", tally[1][0]);
        assert!(
            tally[1][1] >= 60,
            "only {} clang effects reach a bank sample; 62 did when this was measured",
            tally[1][1]
        );
        assert!(maps > 0, "no map opened");
        assert!(tot_barrels > 50, "only {tot_barrels} barrels");
        // Every chain that resolves must reach an actual sample. A wrong
        // stride or offset resolves nothing, so this is the bite.
        assert_eq!(
            tot_sampled, tot_resolved,
            "{} of {tot_resolved} firing chains reached no bank sample",
            tot_resolved - tot_sampled
        );
        assert!(
            tot_resolved as f64 / tot_barrels as f64 > 0.9,
            "only {tot_resolved} of {tot_barrels} barrels resolved"
        );
    }

    /// Map AMBIENCE: the scenario's acoustics palette must name real looping
    /// sounds that reach real bank samples.
    #[test]
    fn probe_map_ambience_reaches_a_bank_sample() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let Some((native, _c, any)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let Some(dir) = hms_sound::SoundBank::bank_dir_for_reach(&any) else {
            return;
        };
        let sfx = hms_sound::SoundBank::open(&dir, "sfx").expect("open sfx");
        let mut maps = 0usize;
        let mut entries = 0usize;
        let mut with_bg = 0usize;
        let mut reached = 0usize;
        for m in reach_maps().iter().take(12) {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            let Some(scnr) = native.find_scenario(cache) else {
                native.close_cache(cache);
                continue;
            };
            let Some(g) = Gestalt::read(&native, cache) else {
                native.close_cache(cache);
                continue;
            };
            let law = GainLaw::read(&native, cache);
            let pal = acoustics_palette(&native, cache, scnr);
            let mut shown = 0usize;
            for a in &pal {
                entries += 1;
                let Some(l) = a.background_sound.and_then(|t| LoopingSound::read(&native, cache, t)) else {
                    continue;
                };
                with_bg += 1;
                let mut got = 0usize;
                let mut first = String::new();
                for tr in &l.tracks {
                    let Some(snd) = tr.loop_snd.and_then(|t| CacheSound::read(&native, cache, t)) else {
                        continue;
                    };
                    for p in g.permutations_of(&snd) {
                        if let Some(i) = sfx.index_of_subsound(p.fmod_subsound_id) {
                            got += 1;
                            if first.is_empty() {
                                first = sfx.asset_path(i).unwrap_or("?").to_string();
                            }
                        }
                    }
                }
                if got > 0 {
                    reached += 1;
                }
                if maps == 0 && shown < 14 {
                    let snd0 = l.tracks.first().and_then(|t| t.loop_snd).and_then(|t| CacheSound::read(&native, cache, t));
                    let gain = snd0
                        .as_ref()
                        .map(|s| law.authored_gain_db(s, g.playback_of(s), g.scale_gain_of(s), 0))
                        .unwrap_or(0.0);
                    eprintln!(
                        "    [{:>2}] {:<26} {} tracks, {got} samples, class {:<18} gain {:>6.1} dB  {}",
                        a.index,
                        if a.name.is_empty() { "(unnamed)".into() } else { a.name.clone() },
                        l.tracks.len(),
                        snd0.as_ref().map(|s| s.class_name()).unwrap_or("?"),
                        gain,
                        first.rsplit('\\').next().unwrap_or("")
                    );
                    shown += 1;
                }
            }
            eprintln!(
                "{}: {} acoustics entries, {} with a background sound",
                m.file_stem().unwrap().to_string_lossy(),
                pal.len(),
                pal.iter().filter(|a| a.background_sound.is_some()).count()
            );
            maps += 1;
            native.close_cache(cache);
        }
        eprintln!("TOTAL {maps} maps: {entries} acoustics entries, {with_bg} resolved to an lsnd, {reached} reached a bank sample");
        assert!(maps > 0, "no map opened");
        assert!(entries > 10, "only {entries} acoustics entries across {maps} maps");
        assert!(with_bg > 5, "only {with_bg} background sounds resolved");
        assert_eq!(reached, with_bg, "{} background sounds reached no bank sample", with_bg - reached);
    }

    /// The gain chain, with the numbers, and the gate that it is real data.
    #[test]
    fn probe_authored_gain_chain() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let Some((native, cache, path)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let law = GainLaw::read(&native, cache);
        let Some(g) = Gestalt::read(&native, cache) else { return };
        eprintln!(
            "{}: sncl {} classes (tag {:#x}), snmm master {:+.1} dB (enabled {}, tag {:#x})",
            path.file_stem().unwrap().to_string_lossy(),
            law.classes.len(),
            law.sncl_tag,
            law.master_db,
            law.master_enabled,
            law.snmm_tag
        );
        assert_eq!(law.classes.len(), 71, "sncl must hold one row per sound_class_enum value");
        assert!(law.master_enabled, "snmm channel gains should be enabled");
        assert!(
            (law.master_db - -2.0).abs() < 1e-6,
            "snmm master gain measured -2.0 dB in every Reach map, read {:+}",
            law.master_db
        );
        for (i, c) in law.classes.iter().enumerate() {
            assert!(
                (MIN_GAIN_DB..=MAX_GAIN_DB).contains(&c.gain_bounds_db.0)
                    && (MIN_GAIN_DB..=MAX_GAIN_DB).contains(&c.gain_bounds_db.1),
                "class {i} ({}) gain bounds {:?} outside the engine's own clamp range",
                sound_class_name(i as u8),
                c.gain_bounds_db
            );
        }
        // The named rows, measured, byte-identical in all 49 Reach caches.
        for (ci, want_lo, want_dist_max) in [
            (CLASS_WEAPON_FIRE, -6.0f32, 50.0f32),
            (6u8, -9.0, 15.0),   // weapon_reload
            (18, -9.0, 10.0),    // unit_footsteps
            (19, -5.0, 45.0),    // unit_dialog
            (32, 0.0, 5.0),      // music
            (33, -6.0, 5.0),     // ambient_nature
        ] {
            let c = law.class(ci).expect("class row");
            assert!(
                (c.gain_bounds_db.0 - want_lo).abs() < 1e-3,
                "{} gain bounds lower {:+}, measured {want_lo:+}",
                sound_class_name(ci),
                c.gain_bounds_db.0
            );
            assert!(
                (c.dist.maximum - want_dist_max).abs() < 1e-3,
                "{} maximum distance {} wu, measured {want_dist_max}",
                sound_class_name(ci),
                c.dist.maximum
            );
        }
        // THE CROSS-CHECK. `playback.gain base + class.gain bounds.lower` was
        // measured independently, from the raw cache bytes by a different
        // route, as median -20 dB / mean -20.14 over forge_halo's sound tags.
        // Reproduce it HERE THROUGH THE PRODUCTION FUNCTION — scale forced
        // off, no permutation trim, no distance, t = 0 so the draw is the
        // lower bound, and the master gain subtracted back off because it
        // sits outside the clamp. Nothing is re-derived: a change to
        // `voice_gain_db` moves this number.
        let mut pair: Vec<f32> = Vec::new();
        let mut full: Vec<f32> = Vec::new();
        for t in sound_tags(&native, cache) {
            let Some(s) = CacheSound::read(&native, cache, t) else { continue };
            let pb = g.playback_of(&s);
            pair.push(law.voice_gain_db(&s, pb, (0.0, 0.0), 0, None, 0.0) - law.master_db);
            let perms = g.permutations_of(&s);
            full.push(law.voice_gain_db(
                &s,
                pb,
                g.scale_gain_of(&s),
                perms.first().map(|p| p.gain_db).unwrap_or(0),
                None,
                0.0,
            ));
        }
        let stats = |v: &mut Vec<f32>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |f: f64| v[(((v.len() - 1) as f64) * f) as usize];
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (q(0.0), q(0.25), q(0.5), q(0.75), q(1.0), mean)
        };
        let (a0, a25, a50, a75, a100, amean) = stats(&mut pair);
        eprintln!(
            "  playback + class.lower over {} sounds: min {a0:+.1}  p25 {a25:+.1}  median {a50:+.1}  \
             p75 {a75:+.1}  max {a100:+.1}  mean {amean:+.2} dB",
            pair.len()
        );
        let (b0, b25, b50, b75, b100, bmean) = stats(&mut full);
        eprintln!(
            "  full chain + scale + perm + master:   min {b0:+.1}  p25 {b25:+.1}  median {b50:+.1}  \
             p75 {b75:+.1}  max {b100:+.1}  mean {bmean:+.2} dB"
        );
        assert!(
            (a50 - -20.0).abs() <= 0.5,
            "median playback+class {a50:+.1} dB; measured independently as -20.0"
        );
        assert!(
            (amean - -20.14).abs() <= 0.5,
            "mean playback+class {amean:+.2} dB; measured independently as -20.14"
        );
        assert!(b100 <= MAX_GAIN_DB + 1e-6, "a sound was louder than unity: {b100:+}");
        assert!(b0 >= MIN_GAIN_DB + law.master_db - 1e-3, "a sound fell below the engine's floor");
    }
}

#[cfg(test)]
mod law_tests {
    use super::*;

    /// The three worked rows, transcribed from the engine's own curve builder
    /// and evaluator and checked against `forge_halo`'s real playback data.
    #[test]
    fn the_distance_curve_matches_the_engine_on_real_forge_halo_rows() {
        // playbacks[2] = (0,0,0, 5,15,50,70, -6)
        let d = Distances {
            dont_obstruct: 0.0, dont_play: 0.0, attack: 0.0,
            minimum: 5.0, sustain_begin: 15.0, sustain_end: 50.0, maximum: 70.0,
            sustain_db: -6.0,
        };
        let c = d.curve();
        assert_eq!(c.count, 4, "don't play and attack are zero, so they are skipped");
        assert!((c.knots[0].0 - 5.0).abs() < 1e-6 && (c.knots[0].1 - 1.0).abs() < 1e-6);
        assert!((c.knots[1].1 - 0.501_187).abs() < 1e-4, "S = 10^(-6/20) = {}", c.knots[1].1);
        for (dist, want) in [
            (0.0f32, 1.0f32), (5.0, 1.0), (8.0, 0.7194), (10.0, 0.6259),
            (15.0, 0.5012), (30.0, 0.5012), (50.0, 0.5012), (70.0, 0.0), (200.0, 0.0),
        ] {
            let g = d.gain_at(dist);
            assert!((g - want).abs() < 1e-3, "d = {dist}: {g}, engine says {want}");
        }

        // playbacks[0] = (0.2,0,0, 1,0,0, 6, 0)
        let d0 = Distances {
            dont_obstruct: 0.2, dont_play: 0.0, attack: 0.0,
            minimum: 1.0, sustain_begin: 0.0, sustain_end: 0.0, maximum: 6.0,
            sustain_db: 0.0,
        };
        assert_eq!(d0.curve().count, 2, "only minimum and maximum are authored");
        for (dist, want) in [(0.5f32, 1.0f32), (1.0, 1.0), (2.0, 0.4), (3.0, 0.2), (5.0, 0.04), (6.0, 0.0), (9.0, 0.0)] {
            let g = d0.gain_at(dist);
            assert!((g - want).abs() < 1e-3, "d = {dist}: {g}, engine says {want}");
        }

        // playbacks[5] = (0.2,0,0, 8,25,50,150,-6)
        let d5 = Distances {
            dont_obstruct: 0.2, dont_play: 0.0, attack: 0.0,
            minimum: 8.0, sustain_begin: 25.0, sustain_end: 50.0, maximum: 150.0,
            sustain_db: -6.0,
        };
        for (dist, want) in [
            (8.0f32, 1.0f32), (10.0, 0.8533), (15.0, 0.6577), (20.0, 0.5599),
            (30.0, 0.5012), (50.0, 0.5012), (70.0, 0.2864), (100.0, 0.1253), (150.0, 0.0),
        ] {
            let g = d5.gain_at(dist);
            assert!((g - want).abs() < 1e-3, "d = {dist}: {g}, engine says {want}");
        }
    }

    #[test]
    fn a_curve_with_no_authored_knots_is_audible_everywhere_at_unity() {
        // 177 of forge_halo's 391 playback rows are in exactly this state.
        let d = Distances::default();
        assert_eq!(d.curve().count, 0);
        for dist in [0.0f32, 1.0, 100.0, 1.0e6] {
            assert_eq!(d.gain_at(dist), 1.0, "d = {dist}");
        }
    }

    #[test]
    fn the_segments_are_linear_in_reciprocal_distance_not_in_distance() {
        // minimum 1 -> maximum 6 with no plateau. The engine's weight is
        // (1/d - 1/6) / (1/1 - 1/6), so the midpoint in DISTANCE (3.5) must
        // NOT be the midpoint in gain.
        let d = Distances { minimum: 1.0, maximum: 6.0, ..Default::default() };
        // t = (6 - 3.5) * 1 / ((6 - 1) * 3.5) = 2.5 / 17.5 = 0.142857
        let mid = d.gain_at(3.5);
        assert!((mid - 0.142_857).abs() < 1e-3, "{mid}");
        assert!(mid < 0.4, "a distance-linear ramp would give 0.5 here, got {mid}");
        // The midpoint in RECIPROCAL distance is 1/d = (1 + 1/6)/2 -> d = 1.714
        let g = d.gain_at(12.0 / 7.0);
        assert!((g - 0.5).abs() < 1e-3, "{g}");
    }

    #[test]
    fn a_knot_at_zero_distance_is_skipped_not_clamped() {
        let d = Distances { dont_play: 0.0, attack: 0.0, minimum: 4.0, maximum: 10.0, ..Default::default() };
        let c = d.curve();
        assert_eq!(c.count, 2);
        assert_eq!(c.knots[0], (4.0, 1.0), "the first knot must be minimum, not don't play");
        // Everything closer than `minimum` is therefore flat at full volume.
        assert_eq!(d.gain_at(0.0), 1.0);
        assert_eq!(d.gain_at(3.9), 1.0);
    }

    #[test]
    fn a_zero_field_in_a_playback_inherits_the_class_value() {
        // weapon_fire's class row, measured: (0,0,0, 8,15,25,50, -6).
        let class = Distances {
            minimum: 8.0, sustain_begin: 15.0, sustain_end: 25.0, maximum: 50.0, sustain_db: -6.0,
            ..Default::default()
        };
        // A playback that only overrides `maximum`.
        let pb = Distances { maximum: 120.0, ..Default::default() };
        let m = merge_distances(class, pb);
        assert_eq!(m.maximum, 120.0, "the non-zero field wins");
        assert_eq!(m.minimum, 8.0, "a zero field inherits");
        assert_eq!(m.sustain_begin, 15.0);
        assert_eq!(m.sustain_db, -6.0);
        // An all-zero playback changes nothing at all.
        assert_eq!(merge_distances(class, Distances::default()), class);
        // And field 0 is excluded from the merge entirely.
        let base = Distances { dont_obstruct: 7.0, maximum: 1.0, ..Default::default() };
        let over = Distances { dont_obstruct: 99.0, ..Default::default() };
        assert_eq!(merge_distances(base, over).dont_obstruct, 7.0);
    }

    #[test]
    fn the_acoustics_row_follows_the_listener_into_the_smallest_cluster() {
        // An interior room sits INSIDE the exterior cluster's box, so the
        // tightest containing box has to win or a listener indoors gets the
        // outdoor bed -- which is the bug this selection exists to fix.
        let outside = AcousticCluster {
            index: 0,
            min: [-100.0, -100.0, -50.0],
            max: [100.0, 100.0, 50.0],
            acoustics: Some(0),
        };
        let room = AcousticCluster {
            index: 1,
            min: [10.0, 10.0, 0.0],
            max: [20.0, 20.0, 6.0],
            acoustics: Some(1),
        };
        let cs = [outside, room];
        assert_eq!(acoustics_at(&cs, [0.0, 0.0, 0.0]), Some(0), "open ground");
        assert_eq!(acoustics_at(&cs, [15.0, 15.0, 3.0]), Some(1), "inside the room");
        assert_eq!(acoustics_at(&cs, [10.0, 10.0, 0.0]), Some(1), "on the room's boundary");
        assert_eq!(acoustics_at(&cs, [21.0, 15.0, 3.0]), Some(0), "just outside the room");
        // Outside every box: the NEAREST cluster, not nothing -- a listener
        // beyond the hull is still in the area it is next to.
        assert_eq!(acoustics_at(&cs, [500.0, 0.0, 0.0]), Some(0));
        assert_eq!(acoustics_at(&cs, [15.0, 15.0, 1000.0]), Some(0));
        // A cluster that names no row is skipped rather than silencing the bed.
        let unnamed = AcousticCluster { acoustics: None, ..room };
        assert_eq!(acoustics_at(&[outside, unnamed], [15.0, 15.0, 3.0]), None);
        assert_eq!(acoustics_at(&[], [0.0, 0.0, 0.0]), None, "no clusters at all");
        // Degenerate boxes must not divide by zero or win on volume.
        let flat = AcousticCluster {
            index: 2,
            min: [0.0, 0.0, 0.0],
            max: [0.0, 0.0, 0.0],
            acoustics: Some(7),
        };
        assert_eq!(flat.volume(), 0.0);
        assert!(acoustics_at(&[outside, flat], [0.0, 0.0, 0.0]).is_some());
    }

    #[test]
    fn the_pan_law_is_constant_sum_and_has_no_centre_channel() {
        // Dead ahead in listener space is +x.
        let g = speaker_pan([1.0, 0.0, 0.0]);
        assert!((g[0] - 0.853_553).abs() < 1e-4, "FL {}", g[0]);
        assert!((g[1] - 0.853_553).abs() < 1e-4, "FR {}", g[1]);
        assert!((g[2] - 0.146_447).abs() < 1e-4, "RL {}", g[2]);
        assert!((g[3] - 0.146_447).abs() < 1e-4, "RR {}", g[3]);
        // Exactly at the front-left speaker.
        let fl = speaker_pan(SPEAKER_POSITIONS[0]);
        assert!((fl[0] - 1.0).abs() < 1e-5, "{fl:?}");
        assert!((fl[1] - 0.5).abs() < 1e-4 && (fl[2] - 0.5).abs() < 1e-4, "{fl:?}");
        assert!(fl[3].abs() < 1e-5, "{fl:?}");
        // Constant SUM = 2.0 in every direction — not constant power.
        for dir in [
            [1.0f32, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.6, 0.8, 0.0], [0.3, -0.4, 0.866], [-0.5, 0.5, 0.707],
        ] {
            let g = speaker_pan(dir);
            let sum: f32 = g.iter().sum();
            assert!((sum - 2.0).abs() < 1e-4, "dir {dir:?} sums to {sum}");
            assert!(g.iter().all(|v| (0.0..=1.0).contains(v)), "dir {dir:?} -> {g:?}");
        }
        assert_eq!(SPEAKER_POSITIONS.len(), 4, "the shipped pan function iterates four speakers");
    }

    #[test]
    fn the_stereo_fold_is_unity_dead_ahead_and_hard_at_the_sides() {
        let f = stereo_pan([1.0, 0.0, 0.0]);
        assert!((f[0] - 0.5).abs() < 1e-4 && (f[1] - 0.5).abs() < 1e-4, "{f:?}");
        // Hard left in Halo's basis is +y.
        let l = stereo_pan([0.0, 1.0, 0.0]);
        assert!(l[0] > l[1], "+y must be the LEFT side: {l:?}");
        let r = stereo_pan([0.0, -1.0, 0.0]);
        assert!(r[1] > r[0], "-y must be the RIGHT side: {r:?}");
        // Mirror symmetry.
        assert!((l[0] - r[1]).abs() < 1e-5 && (l[1] - r[0]).abs() < 1e-5);
        // A source on top of the listener does not produce NaN or silence.
        let z = stereo_pan([0.0, 0.0, 0.0]);
        assert!(z.iter().all(|v| v.is_finite()), "{z:?}");
    }

    #[test]
    fn listener_space_puts_forward_in_x_and_left_in_y() {
        let p = to_listener_space([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(p, [10.0, 0.0, 0.0]);
        let p = to_listener_space([0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(p, [0.0, 3.0, 0.0], "+y world with +y left is 3 units to the LEFT");
        // Facing -x: a source at +x world is now BEHIND.
        let p = to_listener_space([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(p[0], -5.0);
    }

    #[test]
    fn the_power_interpolation_happens_in_linear_gain_not_in_db() {
        // -12 dB to 0 dB. In LINEAR the midpoint is (0.2512 + 1)/2 = 0.6256,
        // which is -4.07 dB, NOT the dB midpoint of -6.
        let m = interp_db_power(-12.0, 0.0, 0.5);
        assert!((m - -4.07).abs() < 0.05, "{m}");
        assert!((interp_db_power(-12.0, 0.0, 0.0) - -12.0).abs() < 1e-3);
        assert!(interp_db_power(-12.0, 0.0, 1.0).abs() < 1e-3);
        // Equal bounds: exact, and that is the case for 65 of the 71 classes.
        assert_eq!(interp_db_power(-6.0, -6.0, 0.37), -6.0);
        // Hostile t.
        assert!(interp_db_power(-6.0, 0.0, f32::NAN).is_finite());
    }

    #[test]
    fn the_cone_is_unity_when_the_outer_gain_is_zero_db() {
        // Reach's shipped data: outer cone gain is 0 dB in all 16,248
        // playbacks, so the cone must not attenuate anything.
        for a in [0.0f32, 0.5, 1.5, 3.0] {
            assert!((cone_gain(0.2, 1.0, 0.0, a) - 1.0).abs() < 1e-6, "angle {a}");
        }
        // And when it IS authored, it lerps in linear gain.
        assert!((cone_gain(0.0, 1.0, -6.0206, 1.0) - 0.5).abs() < 1e-3);
        assert!((cone_gain(0.0, 1.0, -6.0206, 0.0) - 1.0).abs() < 1e-3);
        assert!((cone_gain(0.0, 1.0, -6.0206, 0.5) - 0.75).abs() < 1e-3);
        assert!(cone_gain(f32::NAN, 1.0, -6.0, 0.5).is_finite());
    }
}

#[cfg(test)]
mod probes3 {
    use super::probes_support::*;
    use super::*;

    /// THE DERIVATION of the animation sound-event stride, and the footstep
    /// census.
    ///
    /// Two sources disagreed — the reflection data says
    /// `animation_sound_event_block` is 8 bytes, a sibling investigation
    /// reported the 12-byte `_extended` form — so neither is taken on trust.
    /// The oracle is the one every event must satisfy: its `sound` index must
    /// be inside the graph's own `sound references` block and its `frame`
    /// inside the clip's declared frame count. A wrong stride makes the two
    /// fields read adjacent halves of the wrong words and the score collapses.
    #[test]
    fn probe_animation_sound_event_stride_and_footsteps() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let Some((native, cache, path)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let bipds = native.tags_by_group(cache, "bipd");
        if bipds.is_empty() {
            eprintln!("reach_sound: {} has no bipeds", path.display());
            return;
        }
        let Some(g) = Gestalt::read(&native, cache) else { return };

        // Every distinct biped graph on the map.
        let mut graphs: Vec<(String, u32)> = Vec::new();
        for b in &bipds {
            if let Some(j) = crate::reach_actor::resolve_jmad(&native, cache, *b) {
                if !graphs.iter().any(|(_, t)| *t == j) {
                    let n = native
                        .tag_name(cache, *b)
                        .unwrap_or_default()
                        .rsplit('\\')
                        .next()
                        .unwrap_or("?")
                        .to_string();
                    graphs.push((n, j));
                }
            }
        }
        eprintln!("{} bipeds, {} distinct animation graphs", bipds.len(), graphs.len());
        assert!(!graphs.is_empty(), "no biped resolved an animation graph");

        // Score both candidate strides over every animation of every graph.
        let mut score = [(0usize, 0usize); 2]; // [8, 12] -> (valid, total)
        let mut footsteps = 0usize;
        let mut footstep_clips = 0usize;
        let mut anims_total = 0usize;
        let mut anims_with_events = 0usize;
        let mut examples: Vec<String> = Vec::new();
        let mut bare_markers: std::collections::BTreeMap<String, usize> = Default::default();
        for (bname, j) in graphs.iter().take(6) {
            let refs = graph_sound_refs(&native, cache, *j);
            let nrefs = refs.len();
            let anims = tag_block(
                &native,
                cache,
                *j,
                crate::reach_jmad::OFF_ANIMATIONS,
                crate::reach_jmad::ANIMATION_ELEM,
                1 << 16,
            );
            let n = anims.len() / crate::reach_jmad::ANIMATION_ELEM;
            let mut classes: std::collections::BTreeMap<&str, usize> = Default::default();
            for (t, c) in &refs {
                if *t != 0 {
                    *classes.entry(sound_class_name(*c)).or_default() += 1;
                }
            }
            eprintln!(
                "  {bname:<22} graph {j:#x}: {n} animations, {nrefs} sound references {classes:?}"
            );
            if bname.contains("spartan") {
                let mut leaves: std::collections::BTreeMap<String, usize> = Default::default();
                for (t, c) in &refs {
                    if *t == 0 {
                        *leaves.entry("(empty slot)".into()).or_default() += 1;
                        continue;
                    }
                    let n = native.tag_name(cache, *t).unwrap_or_default();
                    // Group by the folder under data\sound\, which is what
                    // says whether a sound is a footstep.
                    let key = n
                        .split('\\')
                        .take(3)
                        .collect::<Vec<_>>()
                        .join("\\");
                    *leaves.entry(format!("{key} [{}]", sound_class_name(*c))).or_default() += 1;
                }
                for (k, v) in &leaves {
                    eprintln!("      {v:>4} x {k}");
                }
            }
            for ai in 0..n {
                anims_total += 1;
                let Some(fc) = animation_frame_count(&native, cache, *j, ai) else { continue };
                let mut saw = false;
                for (k, stride) in [SOUND_EVENT_SIZE, SOUND_EVENT_SIZE_EXTENDED].iter().enumerate() {
                    let evs = animation_sound_events_with(&native, cache, *j, ai, *stride);
                    for e in &evs {
                        score[k].1 += 1;
                        // 0xFFFF is a `short block index` of -1: the event
                        // names NO sound and is a bare marker. Counting it as
                        // out of range is what held both strides at 93%.
                        if (e.sound == 0xFFFF || (e.sound as usize) < nrefs) && e.frame <= fc {
                            score[k].0 += 1;
                        }
                    }
                    if k == 0 {
                        for e in &evs {
                            if e.sound == 0xFFFF {
                                let m = native.string_id(cache, e.marker_sid).unwrap_or_default();
                                *bare_markers.entry(m).or_default() += 1;
                            }
                        }
                    }
                    if k == 0 && !evs.is_empty() {
                        saw = true;
                        // Footsteps, at the stride this module ships.
                        let mut steps = 0usize;
                        for e in &evs {
                            let Some((t, c)) = refs.get(e.sound as usize).copied() else { continue };
                            if t == 0 {
                                continue;
                            }
                            if c == CLASS_UNIT_FOOTSTEPS || c == CLASS_UNIT_FOOTSTEPS_POSTPONE {
                                steps += 1;
                                footsteps += 1;
                                if examples.len() < 8 {
                                    let snd = CacheSound::read(&native, cache, t);
                                    let perms =
                                        snd.as_ref().map(|s| g.permutations_of(s).len()).unwrap_or(0);
                                    examples.push(format!(
                                        "    {bname} anim {ai}: frame {}/{} marker {:?} -> {} ({perms} takes)",
                                        e.frame,
                                        fc,
                                        native.string_id(cache, e.marker_sid).unwrap_or_default(),
                                        native
                                            .tag_name(cache, t)
                                            .unwrap_or_default()
                                            .rsplit('\\')
                                            .next()
                                            .unwrap_or("?")
                                    ));
                                }
                            }
                        }
                        if steps > 0 {
                            footstep_clips += 1;
                        }
                    }
                }
                if saw {
                    anims_with_events += 1;
                }
            }
        }
        eprintln!(
            "sound-event stride scores: 8 B -> {}/{} valid ({:.1}%), 12 B -> {}/{} valid ({:.1}%)",
            score[0].0,
            score[0].1,
            100.0 * score[0].0 as f64 / score[0].1.max(1) as f64,
            score[1].0,
            score[1].1,
            100.0 * score[1].0 as f64 / score[1].1.max(1) as f64
        );
        eprintln!(
            "{anims_with_events} of {anims_total} animations carry sound events; \
{footsteps} footstep events across {footstep_clips} clips"
        );
        for e in &examples {
            eprintln!("{e}");
        }
        let mut bm: Vec<(&String, &usize)> = bare_markers.iter().collect();
        bm.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!("markers of events that name NO sound (top 12):");
        for (m, n) in bm.iter().take(12) {
            eprintln!("    {n:>5} x {m:?}");
        }
        // The biped's own MATERIAL table, which is where a footstep's sound
        // actually lives: keyed by the surface, not by the animation.
        for b in bipds.iter().take(2) {
            if let Some((sounds, effects)) = crate::reach_damage::material_responses(&native, cache, *b) {
                eprintln!(
                    "  {} material responses: {} sounds, {} effects",
                    native.tag_name(cache, *b).unwrap_or_default().rsplit('\\').next().unwrap_or("?"),
                    sounds.len(),
                    effects.len()
                );
                for r in sounds.iter().take(6) {
                    eprintln!(
                        "      {:<28} -> {:?}",
                        r.material,
                        r.tag.and_then(|t| native.tag_name(cache, t)).map(|n| n
                            .rsplit('\\')
                            .take(3)
                            .collect::<Vec<_>>()
                            .join("/"))
                    );
                }
            }
        }

        assert!(score[0].1 > 0, "no animation on any biped graph carries a sound event");
        // NOT SETTLED, and recorded as not settled. 8 B scores 93.0% of events
        // in range and 12 B 94.2%, which is not a margin anything should be
        // built on. It does not matter: **footsteps do not use this lane.**
        // The graph's sound references on a Spartan are 112 `unit_animation`
        // sounds under `sound\characters\masterchief` and
        // `sound\characters\spartan` with NOT ONE `unit_footsteps` among
        // them, and no event names a null sound. The footstep lane is the
        // FRAME EVENT block (type 4/5) and the sound comes from the material
        // table — see `probe_footstep_chain`. Nothing shipped reads
        // `animation_sound_events`, so leaving this ambiguous costs nothing;
        // anything that comes to depend on it must settle the stride first.
        let r8 = score[0].0 as f64 / score[0].1.max(1) as f64;
        let r12 = score[1].0 as f64 / score[1].1.max(1) as f64;
        eprintln!(
            "the sound-event stride is NOT settled by this oracle (8 B {:.1}%, 12 B {:.1}%); \
nothing shipped reads that lane",
            100.0 * r8,
            100.0 * r12
        );
        assert_eq!(footsteps, 0, "a footstep appeared in the sound-event lane; re-read the model");
    }
}

#[cfg(test)]
mod probes4 {
    use super::probes_support::*;
    use super::*;

    /// THE FOOTSTEP GATE: all three parts, end to end, on real maps.
    ///
    /// 1. the TRIGGER — `left_foot` / `right_foot` frame events on the biped's
    ///    locomotion clips,
    /// 2. the SURFACE — a BSP shader resolved to a global material name
    ///    through the BSP's own collision-material table,
    /// 3. the SOUND — the biped's `foot` material rows for that name, through
    ///    to a real bank sample.
    ///
    /// Written the way the damage agent's gate on the same block is written: a
    /// RATIO on how many rows name a material (a partly-empty row is authored,
    /// not a decode error) plus a STRICT class check on every tag that IS
    /// present, which is the half a wrong stride cannot fake.
    #[test]
    fn probe_footstep_chain() {
        let _serial = super::probes_support::PROBE_LOCK.lock();
        let Some((native, _c, any)) = open_reach("forge_halo") else {
            eprintln!("reach_sound: Reach not installed, skipping");
            return;
        };
        let Some(dir) = hms_sound::SoundBank::bank_dir_for_reach(&any) else { return };
        let sfx = hms_sound::SoundBank::open(&dir, "sfx").expect("open sfx");

        let mut maps = 0usize;
        let mut tot_clips = 0usize;
        let mut tot_steps = 0usize;
        let mut tot_rows = 0usize;
        let mut tot_named = 0usize;
        let mut tot_tagged = 0usize;
        let mut tot_bad_class = 0usize;
        let mut tot_sampled = 0usize;
        let mut tot_surfaces = 0usize;
        let mut tot_surface_hits = 0usize;
        for m in reach_maps().iter().take(4) {
            if let Some(d) = m.parent() {
                native.set_shared_cache_dirs(&[d.to_path_buf()]);
            }
            let Ok(cache) = native.open_cache(&m.to_string_lossy()) else { continue };
            let Some(g) = Gestalt::read(&native, cache) else {
                native.close_cache(cache);
                continue;
            };
            let name = m.file_stem().unwrap().to_string_lossy().into_owned();

            // --- part 2: the surface table ---
            let globals = global_material_names(&native, cache);
            let mut surfaces: std::collections::HashMap<u32, String> = Default::default();
            if let Some(scnr) = native.find_scenario(cache) {
                for sb in native.enumerate_sbsps(cache, scnr, 16) {
                    surfaces.extend(bsp_surface_materials(&native, cache, sb, &globals));
                }
            }
            if maps == 0 {
                let mut rows = 0usize;
                let mut with_shader = 0usize;
                let mut with_global = 0usize;
                if let Some(scnr) = native.find_scenario(cache) {
                    for sb in native.enumerate_sbsps(cache, scnr, 16) {
                        let b = tag_block(
                            &native,
                            cache,
                            sb,
                            SBSP_COLLISION_MATERIALS,
                            COLLISION_MATERIAL_SIZE,
                            4096,
                        );
                        for c in b.chunks_exact(COLLISION_MATERIAL_SIZE) {
                            rows += 1;
                            if tag_ref_at(c, CM_RENDER_METHOD).is_some() {
                                with_shader += 1;
                            }
                            let gi = i16::from_le_bytes([
                                c[CM_GLOBAL_MATERIAL_INDEX],
                                c[CM_GLOBAL_MATERIAL_INDEX + 1],
                            ]);
                            if gi >= 0 {
                                with_global += 1;
                            }
                        }
                    }
                }
                eprintln!(
                    "    SURFACE DIAGNOSIS: {} global material names in matg; {rows} collision-material rows, {with_shader} name a shader, {with_global} carry a non-negative global material index"
                , globals.len());
            }
            let mut by_mat: std::collections::BTreeMap<&str, usize> = Default::default();
            for v in surfaces.values() {
                *by_mat.entry(v.as_str()).or_default() += 1;
            }

            // --- parts 1 and 3, per biped ---
            let bipds = native.tags_by_group(cache, "bipd");
            let mut clips = 0usize;
            let mut steps = 0usize;
            let mut rows = 0usize;
            let mut named = 0usize;
            let mut tagged = 0usize;
            let mut bad_class = 0usize;
            let mut sampled = 0usize;
            let mut surface_hits = 0usize;
            for b in &bipds {
                // part 1
                if let Some(j) = crate::reach_actor::resolve_jmad(&native, cache, *b) {
                    let anims = tag_block(
                        &native,
                        cache,
                        j,
                        crate::reach_jmad::OFF_ANIMATIONS,
                        crate::reach_jmad::ANIMATION_ELEM,
                        1 << 16,
                    );
                    let n = anims.len() / crate::reach_jmad::ANIMATION_ELEM;
                    for ai in 0..n {
                        let fe = animation_foot_events(&native, cache, j, ai);
                        if !fe.is_empty() {
                            clips += 1;
                            steps += fe.len();
                        }
                    }
                }
                // part 3
                let Some((sounds, _fx)) = crate::reach_damage::material_responses(&native, cache, *b)
                else {
                    continue;
                };
                for r in &sounds {
                    rows += 1;
                    if !r.material.is_empty() {
                        named += 1;
                    }
                    for t in [r.tag, r.secondary].into_iter().flatten() {
                        tagged += 1;
                        // THE STRICT HALF: every tag present must be one of
                        // the three classes this row shape can carry.
                        let cls = native.tag_class(cache, t).unwrap_or_default();
                        if !matches!(cls.as_str(), "snd!" | "lsnd" | "effe") {
                            bad_class += 1;
                        }
                    }
                    if !r.material.is_empty() {
                        if let Some(sd) = footstep_sound(&native, cache, *b, &r.material) {
                            if let Some(sn) = CacheSound::read(&native, cache, sd) {
                                if g
                                    .permutations_of(&sn)
                                    .iter()
                                    .any(|p| sfx.index_of_subsound(p.fmod_subsound_id).is_some())
                                {
                                    sampled += 1;
                                }
                            }
                        }
                    }
                }
                // How many of the map's actual SURFACES this biped has a
                // footstep for -- the number that decides whether walking
                // around makes a noise.
                for mat in by_mat.keys() {
                    if footstep_sound(&native, cache, *b, mat).is_some() {
                        surface_hits += 1;
                    }
                }
            }
            eprintln!(
                "{name:<22} {} bipeds | {clips} clips with foot events, {steps} steps | \
{rows} material rows, {named} name a surface, {tagged} tags ({bad_class} wrong class), {sampled} reach a sample | \
{} distinct map surfaces, {surface_hits} have a footstep",
                bipds.len(),
                by_mat.len()
            );
            if maps == 0 {
                let mut top: Vec<(&&str, &usize)> = by_mat.iter().collect();
                top.sort_by(|a, b| b.1.cmp(a.1));
                eprintln!(
                    "    map surfaces (top 8): {:?}",
                    top.iter().take(8).map(|(k, v)| (**k, **v)).collect::<Vec<_>>()
                );
                if let Some(b0) = bipds.first() {
                    if let Some(j) = crate::reach_actor::resolve_jmad(&native, cache, *b0) {
                        for ai in 0..400 {
                            let fe = animation_foot_events(&native, cache, j, ai);
                            if fe.len() >= 2 {
                                eprintln!(
                                    "    example clip {ai}: {:?}",
                                    fe.iter().map(|e| (e.marker_hint(), e.frame)).collect::<Vec<_>>()
                                );
                                break;
                            }
                        }
                    }
                }
            }
            tot_clips += clips;
            tot_steps += steps;
            tot_rows += rows;
            tot_named += named;
            tot_tagged += tagged;
            tot_bad_class += bad_class;
            tot_sampled += sampled;
            tot_surfaces += by_mat.len();
            tot_surface_hits += surface_hits;
            maps += 1;
            native.close_cache(cache);
        }
        eprintln!(
            "TOTAL {maps} maps: {tot_clips} clips with foot events ({tot_steps} steps), \
{tot_rows} material rows ({tot_named} named), {tot_tagged} tags, {tot_bad_class} wrong class, \
{tot_sampled} reach a sample; {tot_surfaces} map surfaces, {tot_surface_hits} covered"
        );
        assert!(maps > 0, "no map opened");

        // PART 1 must find the engine's own footstep timing.
        assert!(tot_clips > 50, "only {tot_clips} clips carry a foot frame event");
        assert!(tot_steps > 100, "only {tot_steps} foot events in all");

        // PART 2, THE SURFACE, IS THE ONE MISSING INPUT, and the diagnosis
        // above says exactly why rather than leaving it a mystery: `matg`
        // yields all **201** global material names, but a cache's `sbsp`
        // gives **0** rows at `+0x24 collision materials`. That offset is the
        // TAG layout; in a cache the collision BSP lives in a paged resource,
        // so the shader-to-global-material table is not in the tag meta at
        // all. Recovering it is a resource read, not an offset fix.
        //
        // This is asserted as a KNOWN ZERO rather than skipped, so that the
        // day someone lands the resource read, this gate turns red and says
        // so instead of silently continuing to believe the surface is
        // unavailable.
        assert_eq!(
            tot_surface_hits, 0,
            "a map surface now resolves to a footstep -- the collision-material table has become \
             readable, so wire the Theater footstep lane to it and update this gate"
        );


        // PART 3: the ratio on names, and the STRICT class check.
        assert!(tot_rows > 500, "only {tot_rows} material rows");
        assert!(
            tot_named as f64 / tot_rows as f64 > 0.9,
            "only {tot_named} of {tot_rows} material rows name a surface"
        );
        assert_eq!(
            tot_bad_class, 0,
            "{tot_bad_class} of {tot_tagged} material-row tags are not snd!/lsnd/effe -- \
             the stride or the offset is wrong"
        );
        assert!(tot_sampled > 200, "only {tot_sampled} footstep rows reach a bank sample");

    }
}
