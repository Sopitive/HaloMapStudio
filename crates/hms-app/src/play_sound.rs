//! #play-sound  TEST MODE'S SOUND: what a driven vehicle (and the rest of a play session) sounds like.
//!
//! Test mode used to be silent -- *"There are also no sounds at all, no vehicle engine revving
//! sounds or anything"*. Theater had the whole sound stack (banks, Vorbis, the engine's gain and
//! distance laws) and test mode wired none of it. This module is the missing wiring, and nothing
//! in it chooses a number: every curve is the tag's, every law is [`crate::reach_sound`]'s.
//!
//! # How the engine makes a vehicle's engine noise (PROVEN, `reach_tag_test.exe` + `haloreach_debug.dll`)
//!
//! There is no "engine sound" field on a `vehi`. It is a **chain of tag data**, each link read here:
//!
//! 1. **`obje +0x104 attachments`** lists `lsnd` tags, each with a `marker` and a **primary
//!    scale** -- the NAME of an object function (`sound_engine_power`, `engine_audio`, `boost`).
//! 2. **`obje +0xEC functions`** defines that function: an `import` name (the engine's built-in
//!    input -- `engine_power`, `engine_rpm`), a mapping curve, and `turn off with`
//!    (`driver_seat_occupied`). Evaluated by `sub_140D47160`; the value is the sound's SCALE.
//!    Active when `> 0` or flagged *always active*, and not while its `turn off with` is off.
//! 3. **The jeep's `engine_power` / `engine_rpm`** (`sub_140E69630`, cases `0x21D` / `0x20B`) are
//!    `flywheel angular velocity / engine max angular velocity * ramp(ticks since the last shift)`,
//!    the ramp (`sub_140E60060`) dipping to 0.5 mid-way through the 0.45 s after a shift -- the
//!    rev-drop you hear on a gear change. The drivetrain is [`crate::reach_vehicle::drive`].
//! 4. **The `lsnd` track** gives the snd!s: `loop` while cruising, `alt loop` while LOADED. The
//!    jeep's engine block lists `load and cruise sound` entries (a function name -- `abs_throttle`
//!    -- and an attachment index); each frame `sub_140E601B0` calls `sub_1401EFB60` with
//!    `function > 0.5`, which is `sound_looping_set_alternate`.
//! 5. **The `snd!`** turns the scale into sound (`sub_1806538B0`, `sub_180659D70`,
//!    `looping_sound_manager.cpp`): pitch in cents is `lerp(scale.pitch lower, upper, scale)`;
//!    gain is `interp_db(class + scale gain bounds, scale)` + `gain base` + the track's gain; and a
//!    sound flagged *all pitch ranges* (`0x2000` -- every Warthog loop) plays EVERY pitch range
//!    whose `bend bounds` contain the pitch, crossfading the edges with `interp_db` mode 2 from
//!    -64 dB. Each range's playback rate is `2^((pitch - natural pitch) / 1200)`. The scale
//!    itself is slewed at the playback's `max bend per second` (0.5 for the Warthog: two seconds
//!    from idle to redline).
//! 6. **Distance and pan** are `reach_sound`'s own, from the marker's world position.
//!
//! The gear change also fires the engine's `gear shift sound` once (`sub_1401F5BC0`). On the
//! player's own vehicle the engine mixes it first-person rather than 3D, so it plays flat.
//!
//! # What is NOT the engine's, and is labelled where it is used
//!
//! * `engine_power` for every family that is not a wheeled jeep (Ghost, Banshee, Wraith,
//!   Scorpion, ...): their input is a spring-damped throttle/speed blend in code this module has
//!   not ported (`sub_140E83C10`). [`Inputs::stand_in`] gives it a documented stand-in.
//! * Built-in inputs other than the ones above resolve to `None` and are REPORTED
//!   ([`PlaySound::unresolved`]); the source stays silent rather than being guessed.
//! * Doppler (the engine has one, `sub_180659D70`, but a driver is at rest relative to the
//!   vehicle), occlusion, the `in`/`out` one-shots of a track, and the detail sounds.

#[path = "play_sound_world.rs"]
mod world;
pub use world::{BodyClip, FootFrame};
use world::{BipedSounds, BodyState, BodyStats, GroundNames, Scenery};

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use hms_native::{CacheHandle, NativeDll};
use hms_sound::VoiceId;

use crate::reach_actor::{block_at, tag_block};
use crate::reach_sound as rs;
use crate::reach_vehicle::drive::{seconds_to_ticks, TICK_RATE};
use crate::scene::MapFn;
use crate::sound::{Clip, LoopSnd, SoundEngine, WeaponSound};

/// The listener, as `SoundEngine::tick` takes it: `(position, forward, left, up)`, Halo's basis.
pub type Listener = ([f32; 3], [f32; 3], [f32; 3], [f32; 3]);

// ---------------------------------------------------------------------------------------------
// OBJECT FUNCTIONS
// ---------------------------------------------------------------------------------------------

pub const OBJE_FUNCTIONS: usize = 0xEC;
pub const FUNCTION_SIZE: usize = 0x40;
pub const FN_FLAGS: usize = 0x00;
pub const FN_IMPORT: usize = 0x04;
pub const FN_EXPORT: usize = 0x08;
pub const FN_OFF_WITH: usize = 0x0C;
pub const FN_MIN_VALUE: usize = 0x14;
/// The embedded `mapping_function`: `{u32 size @0x18, ..., u32 address @0x24}` in a cache.
pub const FN_MAPPING_SIZE: usize = 0x18;
pub const FN_MAPPING_ADDRESS: usize = 0x24;
pub const FN_SCALE_BY: usize = 0x2C;
/// `object_function_flags` bit 0 `invert` ("result of function is one minus actual result") and
/// bit 2 `always active` ("function does not deactivate when at or below lower bound").
pub const FN_FLAG_INVERT: u32 = 1;
pub const FN_FLAG_ALWAYS_ACTIVE: u32 = 4;

/// One entry of a `vehi`'s `functions` block.
#[derive(Clone, Debug)]
pub struct ObjFn {
    pub flags: u32,
    pub import: String,
    pub export: String,
    pub off_with: String,
    pub scale_by: String,
    pub min_value: f32,
    pub map: Option<MapFn>,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).unwrap_or(0)
}
fn rd_f32(b: &[u8], o: usize) -> f32 {
    f32::from_bits(rd_u32(b, o))
}

fn sid_name(native: &NativeDll, cache: CacheHandle, sid: u32) -> String {
    if sid == 0 || sid == u32::MAX {
        return String::new();
    }
    native.string_id(cache, sid).unwrap_or_default()
}

impl ObjFn {
    pub fn read_all(native: &NativeDll, cache: CacheHandle, obj: u32) -> Vec<ObjFn> {
        let b = tag_block(native, cache, obj, OBJE_FUNCTIONS, FUNCTION_SIZE, 256);
        b.chunks_exact(FUNCTION_SIZE)
            .map(|e| {
                let (size, addr) = (rd_u32(e, FN_MAPPING_SIZE), rd_u32(e, FN_MAPPING_ADDRESS));
                let map = if (0x0c..=0x400).contains(&size) && addr != 0 {
                    native.tag_ptr_bytes(cache, addr, size).and_then(|b| MapFn::decode(&b))
                } else {
                    None
                };
                ObjFn {
                    flags: rd_u32(e, FN_FLAGS),
                    import: sid_name(native, cache, rd_u32(e, FN_IMPORT)),
                    export: sid_name(native, cache, rd_u32(e, FN_EXPORT)),
                    off_with: sid_name(native, cache, rd_u32(e, FN_OFF_WITH)),
                    scale_by: sid_name(native, cache, rd_u32(e, FN_SCALE_BY)),
                    min_value: rd_f32(e, FN_MIN_VALUE),
                    map,
                }
            })
            .collect()
    }
}

/// The built-in inputs a vehicle's functions can name, for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Inputs {
    pub engine_power: f32,
    pub engine_rpm: f32,
    /// `abs_throttle`: `|throttle|` of the driver's control.
    pub abs_throttle: f32,
    /// The driver's seat is occupied (by the player, in test mode).
    pub occupied: bool,
}

impl Inputs {
    /// **A jeep's `engine_power` / `engine_rpm`, as the engine computes them** (PROVEN,
    /// `sub_140E69630` cases `0x20B` and `0x21D`): the flywheel's fraction of its maximum times
    /// [`shift_ramp`].
    pub fn jeep(omega_over_max: f32, idle_ticks: u8, abs_throttle: f32) -> Inputs {
        let v = omega_over_max * shift_ramp(idle_ticks);
        Inputs { engine_power: v, engine_rpm: v, abs_throttle, occupied: true }
    }

    /// **STAND-IN, not the engine's** (UNRESOLVED): the `engine_power` of every family that is
    /// not a wheeled jeep. `sub_140E83C10` smooths it with a spring this module has not ported;
    /// its TARGET, read off the decompile, is `(0.5 * speed_fraction + 0.5) * (0.7 * |throttle|
    /// + 0.3 * (1 - boost))` (constants `0.5`, `0.3` from the call in `sub_140E83E50`). Used
    /// with that target, unsmoothed, and `boost = 0`.
    pub fn stand_in(speed_fraction: f32, abs_throttle: f32) -> Inputs {
        let target = (0.5 * speed_fraction.clamp(0.0, 1.0) + 0.5) * (0.7 * abs_throttle.clamp(0.0, 1.0) + 0.3);
        Inputs { engine_power: target, engine_rpm: target, abs_throttle, occupied: true }
    }
}

/// `sub_140E60060`: the factor the engine's output is multiplied by in the 0.45 s after a shift.
///
/// 1 at the shift, falling linearly to 0.5 at the midpoint and rising back to 1 at 0.45 s.
/// `ticks` is the engine's own `idle_ticks` counter (`engine + 0x0E`), which a completed shift
/// resets to 0 and which saturates at 254.
pub fn shift_ramp(ticks: u8) -> f32 {
    let limit = seconds_to_ticks(0.45);
    if u32::from(ticks) >= limit {
        return 1.0;
    }
    let t = (f32::from(ticks) / TICK_RATE) / 0.45;
    let v = if t >= 0.5 { (t - 0.5) * 2.0 } else { 1.0 - t * 2.0 };
    v * 0.5 + 0.5
}

/// A function's value and whether it is active.
type Value = (f32, bool);

/// Evaluate a name: one of the tag's own exported functions, else a built-in input.
/// `None` is an input this module does not have, reported by the caller.
fn value_of(fns: &[ObjFn], name: &str, inp: &Inputs, depth: u8, unresolved: &mut BTreeSet<String>) -> Option<Value> {
    if name.is_empty() {
        return None;
    }
    if depth < 8 {
        if let Some(f) = fns.iter().find(|f| f.export == name) {
            return eval_fn(fns, f, inp, depth + 1, unresolved);
        }
    }
    let v = match name {
        "engine_power" => inp.engine_power,
        "engine_rpm" => inp.engine_rpm,
        "abs_throttle" => inp.abs_throttle,
        // `one` / `zero` are the engine's own constant inputs (`sub_140D47160`: `0x221` -> 1.0,
        // `0x222` -> 0.0).
        "one" => 1.0,
        "zero" => 0.0,
        "driver_seat_occupied" | "driver_seat_power" => f32::from(inp.occupied),
        // Boost is not wired in test mode, so its sounds stay silent: a stated 0, not a guess.
        "boost" => 0.0,
        other => {
            unresolved.insert(other.to_string());
            return None;
        }
    };
    Some((v, v > 0.0))
}

fn eval_fn(fns: &[ObjFn], f: &ObjFn, inp: &Inputs, depth: u8, unresolved: &mut BTreeSet<String>) -> Option<Value> {
    // "turn off with: if the specified function is off, so is this function".
    if !f.off_with.is_empty() {
        let (_, on) = value_of(fns, &f.off_with, inp, depth, unresolved)?;
        if !on {
            return Some((0.0, false));
        }
    }
    let (x, _) = value_of(fns, &f.import, inp, depth, unresolved)?;
    let mut v = match &f.map {
        Some(m) => m.eval(x),
        None => x,
    };
    if !f.scale_by.is_empty() {
        v *= value_of(fns, &f.scale_by, inp, depth, unresolved)?.0;
    }
    if f.flags & FN_FLAG_INVERT != 0 {
        v = 1.0 - v;
    }
    let mut active = f.flags & FN_FLAG_ALWAYS_ACTIVE != 0 || v > 0.0;
    if f.min_value > 0.0 && v <= f.min_value && f.flags & FN_FLAG_ALWAYS_ACTIVE == 0 {
        active = false;
    }
    Some((v, active))
}

// ---------------------------------------------------------------------------------------------
// A VEHICLE'S SOUND SOURCES
// ---------------------------------------------------------------------------------------------

/// `obje +0x104 attachments`, 32 bytes an element.
const ATTACHMENT_MAX: i32 = 24;

/// One `lsnd` track, with its two `snd!`s resolved.
#[derive(Clone, Debug)]
pub struct TrackDef {
    pub gain_db: f32,
    pub lp: Option<LoopSnd>,
    pub alt: Option<LoopSnd>,
    pub fade_in: f32,
    pub fade_out: f32,
    pub fade_in_mode: u8,
    pub fade_out_mode: u8,
    pub alt_crossfade: f32,
    pub alt_mode: u8,
    /// The track's `in` and `out` one-shots: the engine STARTING and STOPPING.
    pub intro: Option<Clip>,
    pub outro: Option<Clip>,
}

/// One `lsnd` attachment of a vehicle.
#[derive(Clone, Debug)]
pub struct LoopSource {
    /// Index in the `attachments` block (what `load and cruise` entries name).
    pub attachment: usize,
    pub marker: String,
    /// The primary scale: an object function's NAME.
    pub scale_name: String,
    pub lsnd_name: String,
    pub tracks: Vec<TrackDef>,
}

/// Everything read once per `vehi` tag.
#[derive(Clone, Debug, Default)]
pub struct VehicleSounds {
    pub fns: Vec<ObjFn>,
    pub sources: Vec<LoopSource>,
    /// `engine.load and cruise sound`: `(function name, attachment index)`.
    pub load_cruise: Vec<(String, usize)>,
    /// `engine.gear shift sound`, as a one-shot.
    pub shift: Option<Clip>,
}

/// `global_vehicle_engine_struct` offsets of the two sound fields, and where each family keeps
/// its engine (`reach_vehicle`'s own `JEEP_ENGINE` / `TANK_ENGINE`).
const ENG_GEAR_SHIFT_SOUND: usize = 0x14;
const ENG_LOAD_AND_CRUISE: usize = 0x24;
const LOAD_CRUISE_SIZE: usize = 8;

/// Read a vehicle's engine block's sound fields: `(gear shift snd, load and cruise entries)`.
pub fn engine_sound_refs(native: &NativeDll, cache: CacheHandle, vehi: u32) -> (Option<u32>, Vec<(String, usize)>) {
    use crate::reach_vehicle as rv;
    let Some(own) = native.tag_meta(cache, vehi, rv::VEHI_OWN_FIELDS as u32, rv::VEHI_OWN_SIZE as u32) else {
        return (None, Vec::new());
    };
    for &(off, name, size, _) in rv::TYPE_BLOCKS.iter() {
        let rel = off - rv::VEHI_OWN_FIELDS;
        if rd_u32(&own, rel) != 1 || rd_u32(&own, rel + 4) == 0 {
            continue;
        }
        let engine_at = match name {
            "human_jeep" => rv::JEEP_ENGINE,
            "human_tank" | "revenant" => rv::TANK_ENGINE,
            _ => return (None, Vec::new()),
        };
        let Some(b) = native.tag_ptr_bytes(cache, rd_u32(&own, rel + 4), size as u32) else { break };
        let shift = rs::tag_ref_at(&b, engine_at + ENG_GEAR_SHIFT_SOUND);
        let lc = block_at(native, cache, &b, engine_at + ENG_LOAD_AND_CRUISE, LOAD_CRUISE_SIZE, 2)
            .chunks_exact(LOAD_CRUISE_SIZE)
            .filter_map(|c| {
                let idx = i32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                (idx >= 0).then(|| (sid_name(native, cache, rd_u32(c, 0)), idx as usize))
            })
            .collect();
        return (shift, lc);
    }
    (None, Vec::new())
}

impl VehicleSounds {
    /// Resolve a vehicle's sound sources. Needs the engine's tag tables up (`begin_frame`).
    pub fn read(eng: &SoundEngine, native: &NativeDll, cache: CacheHandle, vehi: u32) -> VehicleSounds {
        let fns = ObjFn::read_all(native, cache, vehi);
        let mut sources = Vec::new();
        let at = tag_block(native, cache, vehi, rs::OBJE_ATTACHMENTS, rs::ATTACHMENT_SIZE, ATTACHMENT_MAX);
        for (i, c) in at.chunks_exact(rs::ATTACHMENT_SIZE).enumerate() {
            let Some(t) = rs::tag_ref_at(c, rs::ATTACH_TYPE) else { continue };
            if native.tag_class(cache, t).as_deref() != Some("lsnd") {
                continue;
            }
            let Some(l) = rs::LoopingSound::read(native, cache, t) else { continue };
            let resolve = |s: Option<u32>| s.and_then(|s| eng.resolve_loop(native, cache, s));
            let tracks = l
                .tracks
                .iter()
                .map(|tr| TrackDef {
                    gain_db: tr.gain_db,
                    lp: resolve(tr.loop_snd),
                    alt: resolve(tr.alt_loop),
                    fade_in: tr.fade_in_seconds,
                    fade_out: tr.fade_out_seconds,
                    fade_in_mode: tr.fade_in_mode,
                    fade_out_mode: tr.fade_out_mode,
                    alt_crossfade: tr.alt_crossfade_seconds,
                    alt_mode: tr.alt_crossfade_mode,
                    intro: tr.intro.and_then(|t| eng.resolve_one_shot(native, cache, t)),
                    outro: tr.outro.and_then(|t| eng.resolve_one_shot(native, cache, t)),
                })
                .filter(|t| t.lp.is_some() || t.alt.is_some())
                .collect::<Vec<_>>();
            if tracks.is_empty() {
                continue;
            }
            sources.push(LoopSource {
                attachment: i,
                marker: sid_name(native, cache, rd_u32(c, rs::ATTACH_MARKER)),
                scale_name: sid_name(native, cache, rd_u32(c, rs::ATTACH_PRIMARY_SCALE)),
                lsnd_name: native.tag_name(cache, t).unwrap_or_default(),
                tracks,
            });
        }
        let (shift_tag, load_cruise) = engine_sound_refs(native, cache, vehi);
        let shift = shift_tag.and_then(|t| eng.resolve_one_shot(native, cache, t));
        VehicleSounds { fns, sources, load_cruise, shift }
    }

    /// The scale (and whether the source is active) for a source, and whether it is LOADED.
    pub fn scale_of(&self, src: &LoopSource, inp: &Inputs, unresolved: &mut BTreeSet<String>) -> Option<Value> {
        value_of(&self.fns, &src.scale_name, inp, 0, unresolved)
    }

    /// `sound_looping_set_alternate`: the engine's `load and cruise` function is `> 0.5`.
    pub fn loaded(&self, src: &LoopSource, inp: &Inputs, unresolved: &mut BTreeSet<String>) -> bool {
        self.load_cruise
            .iter()
            .filter(|(_, a)| *a == src.attachment)
            .any(|(name, _)| value_of(&self.fns, name, inp, 0, unresolved).map_or(false, |(v, _)| v > 0.5))
    }
}

// ---------------------------------------------------------------------------------------------
// THE RUNTIME
// ---------------------------------------------------------------------------------------------

type VoiceKey = (u8, u8, bool, u8);

/// How long a single-range loop takes to hand over from one pitch range to the next
/// (`sub_180658760(1, 0xB, 0.75, new, old)` in `looping_sound_manager.cpp`, debug `0x1809BA830`).
const RANGE_HANDOVER_SECONDS: f32 = 0.75;

/// Every other voice that starts mid-ride (a range the pitch has just reached, the loaded loop on
/// a throttle press) ramps in over this long: it starts at the beginning of its sample, and the
/// crossfade curve has already lifted it a few dB by the frame it is created.
const ENTRY_RAMP_SECONDS: f32 = 0.08;

/// Time constant of the first-order lag standing in for the engine's spring on a non-jeep
/// family's `engine_power`. A stand-in, not a measurement.
const STAND_IN_TAU: f32 = 0.2;

struct Vox {
    id: VoiceId,
    rate: u32,
    /// The gain and rate last handed to the mixer (before any handover fade), so a voice that
    /// leaves its range can be faded rather than cut.
    gain: [f32; 2],
    mult: f64,
    /// Seconds of fade-IN left (a range taking over from another), and of fade-OUT left (a range
    /// that has been handed over from). `None` = steady.
    born: Option<(f32, f32)>,
    dying: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrackRt {
    scale: f32,
    seeded: bool,
    /// 0..1 fade level: rises over `fade in duration` while active, falls over `fade out`.
    fade: f32,
    /// 0..1: how far the alternate (loaded) loop has taken over.
    alt_mix: f32,
    cur_range: [Option<usize>; 2],
    /// The source was active last frame (its `in` has played and its `out` has not).
    on: bool,
}

/// What one frame of a ride gives the sound.
pub struct RideFrame {
    pub placement_tag: u32,
    /// The player is in the vehicle's DRIVER seat and the seat entry has finished.
    pub driver: bool,
    pub world: Mat4,
    /// Each sound source's marker (first marker of the group), in the vehicle's model space.
    pub markers: std::sync::Arc<HashMap<String, Vec3>>,
    /// The jeep drivetrain, when the vehicle has one: `(omega / omega_max, idle_ticks, gear)`.
    pub drivetrain: Option<(f32, u8, i8)>,
    pub speed: f32,
    pub speed_reference: f32,
    pub throttle: f32,
}

/// What one frame of test mode gives the sound.
pub struct Frame<'a> {
    pub native: &'a NativeDll,
    pub cache: CacheHandle,
    pub map_path: &'a Path,
    pub dt: f32,
    pub listener: Listener,
    pub ride: Option<RideFrame>,
    /// The player's own weapon: `weap` tag and `PlayMode::shots`.
    pub weapon: u32,
    pub shots: u32,
    pub melee: bool,
    /// `PlayHud::reloading.is_some()` and `PlayHud::dry_pulls`.
    pub reloading: bool,
    pub dry_pulls: u32,
    /// The vehicle the player is standing at the door of (the seat prompt is up): its samples are
    /// decoded now, so the engine is heard the moment the player sits instead of seconds later.
    pub offered: Option<u32>,
    /// The player's body, when it is on foot or boarding.
    pub foot: Option<FootFrame>,
}

#[derive(Default)]
struct RideState {
    placement: u32,
    voices: HashMap<VoiceKey, Vox>,
    tracks: HashMap<(u8, u8), TrackRt>,
    last_gear: Option<i8>,
    /// Where the vehicle is (or was left), and its markers in model space, kept so a source can
    /// fade out in place after the driver has gone.
    world: Mat4,
    markers: std::sync::Arc<HashMap<String, Vec3>>,
    /// The smoothed stand-in `engine_power` of a non-jeep family.
    standin: f32,
    standin_seeded: bool,
}

/// Counters and the last frame's decisions, for the caption and for the gates.
#[derive(Clone, Debug, Default)]
pub struct PlaySoundStats {
    pub voices_started: u64,
    pub voices_stopped: u64,
    /// `(key, gain at start, scale)` of the most recent voice start, for the gates.
    pub last_start: Option<(u8, u8, bool, u8, f32)>,
    /// Gear changes that had a shift sound to play, and the ones that actually started a voice
    /// (the tag can author the sound below the engine's -64 dB floor, which is silence).
    pub shifts_triggered: u64,
    /// Engine start (`in`) and stop (`out`) one-shots that started a voice.
    pub starts_played: u64,
    pub stops_played: u64,
    pub shifts_played: u64,
    pub shots_played: u64,
    pub melees_played: u64,
    pub reloads_played: u64,
    pub dry_played: u64,
    /// `(source index, scale, pitch cents)` of the last frame, one per active track.
    pub last: Vec<(usize, f32, f32)>,
}

/// Test mode's audio state. Owned by `App`, next to the [`SoundEngine`] it drives.
#[derive(Default)]
pub struct PlaySound {
    key: Option<(u64, PathBuf)>,
    vehicles: HashMap<u32, VehicleSounds>,
    ride: Option<RideState>,
    last_shots: Option<u32>,
    last_melee: bool,
    last_reloading: bool,
    last_dry: Option<u32>,
    warmed: u32,
    win_t: f32,
    win_started: u64,
    starts_per_s: f32,
    /// Leave the map's ambience bed out. A gate that measures one voice's period cannot have a
    /// second looping bed in the mix.
    mute_ambience: bool,
    pub stats: PlaySoundStats,
    marker_cache: HashMap<u32, std::sync::Arc<HashMap<String, Vec3>>>,
    bipeds: HashMap<u32, Option<BipedSounds>>,
    ground: Option<GroundNames>,
    body: BodyState,
    pub body_stats: BodyStats,
    pub audit: Audit,
    scenery: Scenery,
    /// One-second windows over the output counters, for the caption.
    diag_t: f32,
    diag_prev: hms_sound::OutDiag,
    pub rms_1s: f32,
    /// Last second's RMS per voice category (other, ambience, engine, body, weapon).
    pub cat_rms_1s: [f32; 5],
    pub clipped_1s: u64,
    pub limited_1s: u64,
    last_weapon: u32,
    pub pickups_played: u64,
    /// Built-in function inputs a source asked for that this module does not have.
    pub unresolved: BTreeSet<String>,
}

/// The gameplay events that must make a sound. Every one is counted when it HAPPENS
/// (`expected`) and when a voice actually started (`started`); the difference is in `dropped`,
/// by reason. Always on: the status line prints it, so "it hardly plays" is never a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cat {
    Step,
    Foley,
    Jump,
    Land,
    Fire,
    Melee,
    Reload,
    Pickup,
    EngineStart,
    EngineStop,
    Shift,
    Dry,
}
pub const CAT_N: usize = 12;
pub const CAT_NAMES: [&str; CAT_N] = ["step", "foley", "jump", "land", "fire", "melee", "reload", "pickup", "engine start", "engine stop", "shift", "dry pull"];

#[derive(Clone, Copy, Debug, Default)]
pub struct AuditRow {
    pub expected: u64,
    pub started: u64,
    pub dropped: [u64; crate::sound::REFUSAL_N],
}

/// The rows of the event audit.
#[derive(Clone, Copy, Debug, Default)]
pub struct Audit {
    pub rows: [AuditRow; CAT_N],
}

impl Audit {
    /// Events that were audible (a voice started) over events that were not silent BY THE TAG.
    pub fn delivered_over_expected(&self, cat: Cat) -> (u64, u64) {
        let r = &self.rows[cat as usize];
        (r.started, r.expected - r.dropped[crate::sound::Refusal::NoClip as usize])
    }
}

/// The playback rate for a requested pitch on a range recorded at `natural`: cents are 1200 per
/// octave (`sub_1809B1870` returns `pitch - natural pitch`).
pub fn pitch_ratio(pitch: f32, natural: f32) -> f64 {
    2f64.powf(f64::from(pitch - natural) / 1200.0)
}

impl PlaySound {
    /// Stop everything this module started. Called when test mode ends.
    /// Record one gameplay event: it happened, and whether a voice started (else why not).
    pub fn note(&mut self, cat: Cat, ok: bool, eng: &SoundEngine) {
        let r = &mut self.audit.rows[cat as usize];
        r.expected += 1;
        if ok {
            r.started += 1;
        } else {
            r.dropped[eng.last_drop() as usize] += 1;
        }
    }

    pub fn stop(&mut self, eng: &mut SoundEngine) {
        if let Some(r) = self.ride.take() {
            for v in r.voices.values() {
                eng.voice_stop(v.id);
            }
        }
        self.scenery = Scenery::default();
        eng.stop_all();
        self.last_shots = None;
    }

    /// Forget everything resolved for another map. The engine has already stopped its voices and
    /// cleared its tag tables when the map changed; these are the tables built on top of them.
    pub fn sync_map(&mut self, cache: CacheHandle, map_path: &Path) {
        let key = (cache.0, map_path.to_path_buf());
        if self.key.as_ref() != Some(&key) {
            self.vehicles.clear();
            self.marker_cache.clear();
            self.bipeds.clear();
            self.ground = None;
            self.body = BodyState::default();
            self.scenery = Scenery::default();
            self.ride = None;
            self.last_shots = None;
            self.last_dry = None;
            self.warmed = 0;
            self.key = Some(key);
        }
    }

    /// One frame. Cheap when there is nothing to do; never blocks on the banks.
    pub fn update(&mut self, eng: &mut SoundEngine, f: &Frame) {
        // Voice creations per second, over one-second windows: a steady engine starts ~none.
        self.win_t += f.dt;
        if self.win_t >= 1.0 {
            self.starts_per_s = (self.stats.voices_started - self.win_started) as f32 / self.win_t;
            self.win_t = 0.0;
            self.win_started = self.stats.voices_started;
        }
        if !eng.begin_frame(f.native, f.cache, f.map_path) {
            return;
        }
        self.sync_map(f.cache, f.map_path);
        if !self.mute_ambience {
            eng.ambience_frame(f.native, f.cache, f.listener.0);
        }
        if let Some(t) = f.offered {
            self.warm_vehicle(eng, f, t);
        }
        eng.cat_hint = 4;
        self.weapon_events(eng, f);
        eng.cat_hint = 3;
        if let Some(ft) = f.foot.as_ref() {
            self.body_frame(eng, f, ft);
        }
        eng.cat_hint = 1;
        if !self.mute_ambience {
            self.scenery_frame(eng, f);
        }
        eng.cat_hint = 2;
        self.ride_frame(eng, f);
        eng.cat_hint = 0;
        self.diag_window(eng, f.dt);
    }

    /// Once a second: the output's RMS, and how many frames clipped / were limited.
    fn diag_window(&mut self, eng: &SoundEngine, dt: f32) {
        self.diag_t += dt;
        if self.diag_t < 1.0 {
            return;
        }
        if let Some(d) = eng.out_diag() {
            let df = d.mixed_frames.saturating_sub(self.diag_prev.mixed_frames);
            if df > 0 {
                let ms = (d.sumsq_nano.saturating_sub(self.diag_prev.sumsq_nano)) as f64 / 1.0e9 / df as f64;
                self.rms_1s = ms.sqrt() as f32;
            }
            for c in 0..self.cat_rms_1s.len() {
                let ss = d.cat_sumsq_nano[c].saturating_sub(self.diag_prev.cat_sumsq_nano[c]) as f64 / 1.0e9;
                self.cat_rms_1s[c] = if df > 0 { (ss / df as f64).sqrt() as f32 } else { 0.0 };
            }
            self.clipped_1s = d.clipped.saturating_sub(self.diag_prev.clipped);
            self.limited_1s = d.limited.saturating_sub(self.diag_prev.limited);
            self.diag_prev = d;
        }
        self.diag_t = 0.0;
    }

    /// Resolve a vehicle's sound sources and ask for every sample they can play. Decoding runs on
    /// the worker, so this costs a message per sample and the engine is ready by the time it is
    /// needed: a Warthog's idle loop and its redline loop are different samples.
    fn warm_vehicle(&mut self, eng: &mut SoundEngine, f: &Frame, tag: u32) {
        if self.vehicles.contains_key(&tag) {
            return;
        }
        let vs = VehicleSounds::read(eng, f.native, f.cache, tag);
        let mut asked = 0;
        for ls in vs.sources.iter().flat_map(|s| s.tracks.iter()).flat_map(|t| t.lp.iter().chain(t.alt.iter())) {
            for r in &ls.ranges {
                for &s in r.samples.iter().take(4) {
                    if asked < 96 {
                        let _ = eng.sample(s);
                        asked += 1;
                    }
                }
            }
        }
        for c in vs.sources.iter().flat_map(|s| s.tracks.iter()).flat_map(|t| t.intro.iter().chain(t.outro.iter())).chain(vs.shift.iter()) {
            for &s in c.samples.iter().take(16) {
                let _ = eng.sample(s);
            }
        }
        self.vehicles.insert(tag, vs);
    }

    fn weapon_events(&mut self, eng: &mut SoundEngine, f: &Frame) {
        // The player's own gun sounds from where the listener is, a hand's length ahead.
        let (lp, fwd, ..) = f.listener;
        let at = [lp[0] + fwd[0] * 0.4, lp[1] + fwd[1] * 0.4, lp[2] + fwd[2] * 0.4];
        let shots = f.shots;
        if self.warmed != f.weapon && f.weapon != 0 {
            eng.warm_weapon(f.native, f.cache, f.weapon);
            self.warmed = f.weapon;
        }
        if let Some(prev) = self.last_shots {
            // One sound per shot, however many the frame covered (a long frame can hold two or
            // three at 15 shots a second); more than a few in one frame is a hitch, not a burst.
            for k in 0..shots.saturating_sub(prev).min(8) {
                if f.weapon != 0 {
                    let ok = eng.play_weapon_sound(f.native, f.cache, f.weapon, WeaponSound::Fire, at, f.listener, shots.wrapping_add(k));
                    self.note(Cat::Fire, ok, eng);
                    if ok {
                        self.stats.shots_played += 1;
                    }
                }
            }
        }
        self.last_shots = Some(shots);
        // A different weapon in the hands: its `pickup sound` (weap +0x170).
        if f.weapon != self.last_weapon {
            if self.last_weapon != 0 && f.weapon != 0 {
                if let Some(t) = f.native.tag_meta(f.cache, f.weapon, rs::WEAP_PICKUP_SOUND as u32, 16).and_then(|m| rs::tag_ref_at(&m, 0)) {
                    if let Some(c) = rs::resolve_to_snd(f.native, f.cache, t).and_then(|s| eng.resolve_one_shot(f.native, f.cache, s)) {
                        for &s in c.samples.iter().take(8) {
                            let _ = eng.sample(s);
                        }
                        let ok = eng.play_clip_at(&c, at, f.listener, shots);
                        self.note(Cat::Pickup, ok, eng);
                        if ok {
                            self.pickups_played += 1;
                        }
                    }
                }
            }
            self.last_weapon = f.weapon;
        }
        if f.melee && !self.last_melee && f.weapon != 0 {
            let ok = eng.play_weapon_sound(f.native, f.cache, f.weapon, WeaponSound::Melee, at, f.listener, shots);
            self.note(Cat::Melee, ok, eng);
            if ok {
                self.stats.melees_played += 1;
            }
        }
        self.last_melee = f.melee;
        // The magazine's reload starting (the key or an empty pull both start one), and an empty
        // pull itself: `play_hud` counts both, so this reads them rather than re-deriving them.
        if f.reloading && !self.last_reloading && f.weapon != 0 {
            let ok = eng.play_weapon_sound(f.native, f.cache, f.weapon, WeaponSound::Reload, at, f.listener, shots);
            self.note(Cat::Reload, ok, eng);
            if ok {
                self.stats.reloads_played += 1;
            }
        }
        self.last_reloading = f.reloading;
        if let Some(prev) = self.last_dry.filter(|p| f.dry_pulls > *p) {
            let _ = prev;
            if f.weapon != 0 {
                let ok = eng.play_weapon_sound(f.native, f.cache, f.weapon, WeaponSound::Empty, at, f.listener, shots);
                self.note(Cat::Dry, ok, eng);
                if ok {
                    self.stats.dry_played += 1;
                }
            }
        }
        self.last_dry = Some(f.dry_pulls);
    }

    fn ride_frame(&mut self, eng: &mut SoundEngine, f: &Frame) {
        // Which vehicle, where, and with what inputs. Driving: the frame's. Not driving (got out,
        // or never was the driver): the engine's `driver_seat_occupied` is off, so every source
        // goes inactive and fades out where the vehicle was left.
        let (placement, mut inputs, gear, standin) = match f.ride.as_ref() {
            Some(r) if r.driver => match r.drivetrain {
                Some((frac, idle_ticks, gear)) => (r.placement_tag, Inputs::jeep(frac, idle_ticks, r.throttle.abs()), Some(gear), false),
                None => {
                    let sf = if r.speed_reference > 1e-3 { r.speed.abs() / r.speed_reference } else { 0.0 };
                    (r.placement_tag, Inputs::stand_in(sf, r.throttle.abs()), None, true)
                }
            },
            Some(r) => (r.placement_tag, Inputs::default(), None, false),
            None => match &self.ride {
                Some(st) => (st.placement, Inputs::default(), None, false),
                None => return,
            },
        };
        self.warm_vehicle(eng, f, placement);
        let vs = &self.vehicles[&placement];
        if vs.sources.is_empty() {
            return;
        }
        let st = self.ride.get_or_insert_with(RideState::default);
        if st.placement != placement {
            for v in st.voices.values() {
                eng.voice_stop(v.id);
            }
            *st = RideState { placement, world: Mat4::IDENTITY, ..Default::default() };
        }
        if let Some(r) = f.ride.as_ref() {
            st.world = r.world;
            st.markers = r.markers.clone();
        }

        if standin {
            // The engine smooths this input with a spring (`sub_140E83C10`); its constants are
            // unresolved, so a first-order lag stands in for it (see `Inputs::stand_in`).
            let k = 1.0 - (-f.dt / STAND_IN_TAU).exp();
            st.standin = if st.standin_seeded { st.standin + (inputs.engine_power - st.standin) * k } else { inputs.engine_power };
            st.standin_seeded = true;
            inputs.engine_power = st.standin;
            inputs.engine_rpm = st.standin;
        }

        // The gear shift: once, at the vehicle, on the frame the gear changes.
        if let Some(gear) = gear {
            if let (Some(prev), Some(clip)) = (st.last_gear, vs.shift.as_ref()) {
                if prev != gear {
                    self.stats.shifts_triggered += 1;
                    // The player's own vehicle: the engine mixes it first-person, not 3D.
                    let ok = eng.play_clip_flat(clip, gear as u32);
                    // (field-wise, not `self.note`: `st` still borrows `self.ride`)
                    let r = &mut self.audit.rows[Cat::Shift as usize];
                    r.expected += 1;
                    if ok {
                        r.started += 1;
                        self.stats.shifts_played += 1;
                    } else {
                        r.dropped[eng.last_drop() as usize] += 1;
                    }
                }
            }
            st.last_gear = Some(gear);
        }

        self.stats.last.clear();
        let mut wanted: HashMap<VoiceKey, ()> = HashMap::new();
        let (lpos, fwd, left, up) = f.listener;
        let mut any_fade = false;
        for (si, src) in vs.sources.iter().enumerate() {
            let (target, active) = vs.scale_of(src, &inputs, &mut self.unresolved).unwrap_or((0.0, false));
            let loaded = vs.loaded(src, &inputs, &mut self.unresolved);
            // The source's position: its marker on the hull, else the hull's origin.
            let pos = match st.markers.get(&src.marker).copied() {
                Some(m) => st.world.transform_point3(m),
                None => st.world.w_axis.truncate(),
            };
            let dir = rs::to_listener_space(pos.to_array(), lpos, fwd, left, up);
            let dist = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let pan = rs::stereo_pan(dir);

            for (ti, tr) in src.tracks.iter().enumerate() {
                let rt = st.tracks.entry((si as u8, ti as u8)).or_default();
                // The scale chases its target at the playback's `max bend per second`
                // (`looping_sound_manager.cpp`, debug `0x1809BA830`): a rate on the 0..1 scale.
                let bend = tr.lp.as_ref().or(tr.alt.as_ref()).and_then(|l| l.playback).map_or(0.0, |p| p.max_bend_per_second_cents);
                if !rt.seeded || bend <= 0.0 {
                    rt.scale = target;
                    rt.seeded = true;
                } else {
                    let step = bend * f.dt;
                    rt.scale += (target - rt.scale).clamp(-step, step);
                }
                // Fade in while active, out otherwise.
                let secs = if active { tr.fade_in } else { tr.fade_out };
                let rate = if secs > 1e-3 { f.dt / secs } else { 1.0 };
                rt.fade = if active { (rt.fade + rate).min(1.0) } else { (rt.fade - rate).max(0.0) };
                let fade_db = rs::interp_db(if active { tr.fade_in_mode } else { tr.fade_out_mode }.min(4), rs::MIN_GAIN_DB, 0.0, rt.fade);
                // Cruise <-> loaded crossfade over `alt crossfade duration`.
                let two = tr.lp.is_some() && tr.alt.is_some();
                if two {
                    let want = if loaded { 1.0 } else { 0.0 };
                    let r = if tr.alt_crossfade > 1e-3 { f.dt / tr.alt_crossfade } else { 1.0 };
                    rt.alt_mix += (want - rt.alt_mix).clamp(-r, r);
                }
                // The engine STARTING and STOPPING: the track's `in` one-shot when the source
                // turns on (the driver sits), its `out` when it turns off (he leaves).
                if active != rt.on {
                    rt.on = active;
                    let clip = if active { tr.intro.as_ref() } else { tr.outro.as_ref() };
                    if let Some(c) = clip {
                        // The driver's own engine starting is mixed first-person (the engine routes a
                        // sound on the player's own vehicle that way, `sub_1401F5BC0`), not 3D: heard
                        // from the seat the hull's origin is inside the clip's `don't play` distance
                        // and the start was refused (audit: engine start 0/2, out of range).
                        let own = matches!(f.ride.as_ref(), Some(r) if r.driver);
                        let ok = if own && active {
                            eng.play_clip_flat(c, (si * 8 + ti) as u32)
                        } else {
                            eng.play_clip_at(c, pos.to_array(), f.listener, (si * 8 + ti) as u32)
                        };
                        let r = &mut self.audit.rows[if active { Cat::EngineStart } else { Cat::EngineStop } as usize];
                        r.expected += 1;
                        if ok {
                            r.started += 1;
                            if active { self.stats.starts_played += 1 } else { self.stats.stops_played += 1 }
                        } else {
                            r.dropped[eng.last_drop() as usize] += 1;
                        }
                    }
                }
                if rt.fade <= 0.0 {
                    continue;
                }
                any_fade = true;
                let layers: Vec<(bool, &LoopSnd, f32)> = match (tr.lp.as_ref(), tr.alt.as_ref()) {
                    (Some(a), Some(b)) => vec![(false, a, 1.0 - rt.alt_mix), (true, b, rt.alt_mix)],
                    (Some(a), None) => vec![(false, a, 1.0)],
                    (None, Some(b)) => vec![(true, b, 1.0)],
                    (None, None) => Vec::new(),
                };
                for (alt, ls, w) in layers {
                    if w <= 1e-3 {
                        continue;
                    }
                    let weight_db = if two { rs::interp_db(tr.alt_mode.min(4), rs::MIN_GAIN_DB, 0.0, w) } else { 0.0 };
                    let pitch = ls.pitch_at(rt.scale);
                    self.stats.last.push((si, rt.scale, pitch));
                    let ranges = ls.ranges_at(pitch, rt.cur_range[usize::from(alt)]);
                    rt.cur_range[usize::from(alt)] = ranges.first().map(|r| r.0);
                    for (ri, range_db) in ranges {
                        let r = &ls.ranges[ri];
                        let perm = 0; // the voice steps through the takes itself; the trim below is the first take's
                        if r.samples.is_empty() {
                            continue;
                        }
                        let src_db = tr.gain_db + fade_db + weight_db + range_db;
                        let db = eng.loop_gain_db(ls, rt.scale, r.perm_gain_db.get(perm).copied().unwrap_or(0), src_db);
                        let mut lin = rs::db_to_linear(db);
                        if ls.spatial {
                            lin *= if dist > ls.dist.maximum || dist < ls.dist.dont_play { 0.0 } else { ls.dist.gain_at(dist) };
                        }
                        let gain = [lin * pan[0], lin * pan[1]];
                        let rate = pitch_ratio(pitch, r.natural_pitch);
                        let key: VoiceKey = (si as u8, ti as u8, alt, ri as u8);
                        wanted.insert(key, ());
                        match st.voices.get_mut(&key) {
                            Some(v) => {
                                v.gain = gain;
                                v.mult = rate;
                                v.dying = None;
                            }
                            None => {
                                // A second range of a single-range loop taking over from the one
                                // that held the pitch: fade in as the other fades out.
                                let handover = !ls.all_ranges_at_once()
                                    && st.voices.keys().any(|k| (k.0, k.1, k.2) == (key.0, key.1, key.2));
                                self.stats.last_start = Some((key.0, key.1, key.2, key.3, gain[0].max(gain[1])));
                                if let Some((id, src_rate)) = eng.voice_start(&r.samples, ls.cs.flags & rs::SND_FLAG_PERMUTATIONS_IN_ORDER != 0, [0.0, 0.0], rate) {
                                    st.voices.insert(
                                        key,
                                        Vox { id, rate: src_rate, gain, mult: rate, born: Some(if handover { (RANGE_HANDOVER_SECONDS, RANGE_HANDOVER_SECONDS) } else { (ENTRY_RAMP_SECONDS, ENTRY_RAMP_SECONDS) }), dying: None },
                                    );
                                    self.stats.voices_started += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Voices whose range left the pitch, or whose source went quiet. A range that was crossfaded
        // out by its own `bend bounds` is already at -64 dB and is stopped; a single-range loop's
        // outgoing range fades over the handover time instead of being cut (a click).
        let dt = f.dt;
        st.voices.retain(|k, v| {
            if !wanted.contains_key(k) {
                let slow = v.gain[0].max(v.gain[1]) > 0.01 && st.tracks.get(&(k.0, k.1)).map_or(false, |t| t.fade > 0.0);
                let left = v.dying.unwrap_or(if slow { RANGE_HANDOVER_SECONDS } else { 0.0 });
                let left = left - dt;
                if left <= 0.0 {
                    eng.voice_stop(v.id);
                    self.stats.voices_stopped += 1;
                    return false;
                }
                v.dying = Some(left);
            }
            // Fade the handover in or out and aim the voice.
            let mut g = 1.0;
            if let Some((left, total)) = v.born.as_mut() {
                *left -= dt;
                g *= (1.0 - *left / *total).clamp(0.0, 1.0);
                if *left <= 0.0 {
                    v.born = None;
                }
            }
            if let Some(d) = v.dying {
                g *= (d / RANGE_HANDOVER_SECONDS).clamp(0.0, 1.0);
            }
            eng.voice_set(v.id, v.rate, [v.gain[0] * g, v.gain[1] * g], v.mult);
            true
        });
        // Once nobody is driving and every fade is spent, the ride's audio state is done.
        if !any_fade && f.ride.as_ref().map_or(true, |r| !r.driver) {
            self.ride = None;
        }
    }

    /// One line of numbers for the status caption, readable without setting anything: the
    /// device's rate and buffer, the callback's worst gap and mix time, underruns, how many
    /// voices the ride creates per second (a steady engine: ~0), and decode stalls.
    pub fn caption(&self, eng: &SoundEngine) -> String {
        let st = eng.status();
        let dev = match eng.out_diag() {
            Some(d) if d.callbacks > 0 => format!(
                "device {} Hz, buffer {} fr, gap max {:.1} ms, mix max {:.2} ms, underruns {}, late {}",
                d.rate, d.buffer_frames, d.max_gap_ms, d.max_fill_ms, d.underruns, d.late
            ),
            _ => "no output device running".to_string(),
        };
        let b = &self.body_stats;
        format!(
            "sound: {dev}\nsound: out rms {:.4} (last s), clip {} frames, limiter {} frames | voices: ambient {} (bed {} + scenery {}), engine {}, one-shots live {} | \
             steps {} ({} silent) foley {} jumps {} lands {} on '{}', pickups {}, engine start/stop {}/{}, shots {} | starts {:.1}/s, decode stalls {}",
            self.rms_1s,
            self.clipped_1s,
            self.limited_1s,
            st.ambience_playing + self.scenery.live_voices(),
            st.ambience_playing,
            self.scenery.live_voices(),
            self.ride_voices(),
            st.voices.saturating_sub(st.ambience_playing),
            b.steps,
            b.steps_silent,
            b.foley,
            b.jumps,
            b.lands,
            b.last_material,
            self.pickups_played,
            self.stats.starts_played,
            self.stats.stops_played,
            self.stats.shots_played,
            self.starts_per_s,
            st.decode_stalls,
        ) + &self.audit_lines(eng)
    }

    /// The event audit and the drop counters, for the status line: per gameplay event, voices
    /// started over events seen, then every refusal by reason, then the decoder's backlog.
    pub fn audit_lines(&self, eng: &SoundEngine) -> String {
        let st = eng.status();
        let db = |v: f32| 20.0 * v.max(1e-6).log10();
        let mut ev = String::new();
        for (i, name) in CAT_NAMES.iter().enumerate() {
            let r = &self.audit.rows[i];
            if r.expected > 0 {
                ev.push_str(&format!(" {name} {}/{}", r.started, r.expected));
            }
        }
        let mut dr = String::new();
        for (i, name) in crate::sound::REFUSAL_NAMES.iter().enumerate().skip(1) {
            if st.drops[i] > 0 {
                dr.push_str(&format!(" {name} {}", st.drops[i]));
            }
        }
        format!(
            "\nsound: level last s by category (dBFS): ambience {:.0} engine {:.0} body {:.0} weapon {:.0} other {:.0}\nsound: events heard/seen:{}\nsound: drops:{} | loading {} samples, {} decoded (avg {:.1} ms, slowest {:.0} ms), urgent waits {} ({:.0} ms)",
            db(self.cat_rms_1s[1]),
            db(self.cat_rms_1s[2]),
            db(self.cat_rms_1s[3]),
            db(self.cat_rms_1s[4]),
            db(self.cat_rms_1s[0]),
            if ev.is_empty() { " none yet".to_string() } else { ev },
            if dr.is_empty() { " none".to_string() } else { dr },
            st.loading,
            st.decoded,
            st.decode_ms_total / st.decodes.max(1) as f32,
            st.decode_ms_max,
            st.urgent_waits,
            st.urgent_wait_ms,
        )
    }

    /// Live positional scenery voices, for the gates.
    pub fn scenery_voices(&self) -> usize {
        self.scenery.live_voices()
    }

    /// Live voices of the ride, for the gates.
    pub fn ride_voices(&self) -> usize {
        self.ride.as_ref().map_or(0, |r| r.voices.len())
    }
}

/// What the host (`App::update_play`) knows that test mode's own state does not.
pub struct Host<'a> {
    pub native: &'a NativeDll,
    pub cache: CacheHandle,
    pub map_path: &'a Path,
    pub dt: f32,
    pub listener: Listener,
    pub reloading: bool,
    pub dry_pulls: u32,
    /// The level triangle's render-material index straight below the player's feet.
    pub ground_material: Option<i32>,
}

/// What `PlayMode` hands over: its own state, read off in one place.
pub struct PlayerState<'a> {
    pub weapon: u32,
    pub shots: u32,
    pub melee: bool,
    pub riding: Option<&'a crate::play_vehicle::Riding>,
    /// The movement stick's forward axis, which is the driver's throttle.
    pub stick_forward: f32,
    /// The `vehi` tag of the seat on offer, if the prompt is up.
    pub offered: Option<u32>,
    pub body: BodyClip,
    pub input: crate::play_ctl::MoveInput,
    pub pos: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
    pub jumped: bool,
    pub landed: bool,
    pub biped: u32,
}

/// Build this frame's [`Frame`] from test mode's state and run [`PlaySound::update`].
///
/// `vehicles` is test mode's per-tag cache (`PlayMode::vehicles`), which is where a vehicle's
/// render-model markers and physics live.
pub fn drive(
    ps: &mut PlaySound,
    eng: &mut SoundEngine,
    host: &Host,
    player: &PlayerState,
    vehicles: &mut crate::play_vehicle::VehicleIndex,
) {
    use crate::reach_vehicle::Family;
    let (native, cache) = (host.native, host.cache);
    ps.sync_map(cache, host.map_path);
    // The DRIVER of the ridden placement itself: a gunner on a Warthog's turret does not drive it.
    let driving = player.riding.filter(|r| {
        r.phase == crate::play_vehicle::RidePhase::Seated
            && r.vehi_tag == r.placement_tag
            && vehicles
                .entry(native, cache, r.vehi_tag)
                .and_then(|e| e.seats.seats.get(r.seat))
                .map_or(false, |s| s.is_driver())
    });
    // Markers are resolved once per vehicle tag and shared.
    if let Some(r) = player.riding {
        if !ps.marker_cache.contains_key(&r.placement_tag) {
            let mut m = HashMap::new();
            if let Some(entry) = vehicles.entry(native, cache, r.placement_tag) {
                for name in marker_names(native, cache, r.placement_tag) {
                    if let Some(f) = entry.marker_frames(native, cache, &name).first() {
                        m.insert(name, f.w_axis.truncate());
                    }
                }
            }
            ps.marker_cache.insert(r.placement_tag, std::sync::Arc::new(m));
        }
    }
    let frame_ride = player.riding.and_then(|r| {
        let entry = vehicles.entry(native, cache, r.placement_tag)?;
        let (omega_max, speed_ref) = match entry.phys.as_ref().map(|p| &p.family) {
            Some(Family::Jeep { engine, limits, .. }) => (Some(engine.max_angular_velocity), limits.as_ref().map_or(0.0, |l| l.top_speed_ceiling)),
            Some(Family::Tank { engine, limits, .. }) => (Some(engine.max_angular_velocity), limits.as_ref().map_or(0.0, |l| l.top_speed_ceiling)),
            Some(f) => (None, f.velocity_model().map_or(0.0, |m| m.max_forward_speed)),
            None => (None, 0.0),
        };
        // The jeep family is the one whose input is PROVEN; the tank shares the engine struct but
        // not the function handler, so it takes the stand-in.
        let jeep = matches!(entry.phys.as_ref().map(|p| &p.family), Some(Family::Jeep { .. }));
        let drivetrain = match (jeep, omega_max) {
            (true, Some(m)) if m > 1e-3 => Some((r.drive.engine.angular_velocity / m, r.drive.engine.idle_ticks, r.drive.engine.gear)),
            _ => None,
        };
        Some(RideFrame {
            placement_tag: r.placement_tag,
            driver: driving.is_some(),
            world: r.world(),
            markers: ps.marker_cache.get(&r.placement_tag).cloned().unwrap_or_default(),
            drivetrain,
            speed: r.drive.vel.length(),
            speed_reference: speed_ref,
            throttle: player.stick_forward,
        })
    });
    let frame = Frame {
        native,
        cache,
        map_path: host.map_path,
        dt: host.dt,
        listener: host.listener,
        ride: frame_ride,
        weapon: player.weapon,
        shots: player.shots,
        melee: player.melee,
        reloading: host.reloading,
        dry_pulls: host.dry_pulls,
        offered: player.offered,
        foot: Some(FootFrame {
            body: player.body,
            input: player.input,
            pos: player.pos,
            vel: player.vel,
            grounded: player.grounded,
            jumped: player.jumped,
            landed: player.landed,
            biped: player.biped,
            riding: player.riding.is_some() && player.riding.map_or(false, |r| r.phase == crate::play_vehicle::RidePhase::Seated),
            ground_material: host.ground_material,
        }),
    };
    ps.update(eng, &frame);
}

/// Marker names a vehicle's sound sources hang on, so the host resolves only those.
fn marker_names(native: &NativeDll, cache: CacheHandle, vehi: u32) -> Vec<String> {
    let at = tag_block(native, cache, vehi, rs::OBJE_ATTACHMENTS, rs::ATTACHMENT_SIZE, ATTACHMENT_MAX);
    at.chunks_exact(rs::ATTACHMENT_SIZE)
        .map(|c| sid_name(native, cache, rd_u32(c, rs::ATTACH_MARKER)))
        .filter(|n| !n.is_empty())
        .collect()
}

#[cfg(test)]
mod probe {
    use super::*;

    /// `cargo test -p hms-app --release probe_vehicle_sound_sources -- --ignored --nocapture`
    #[test]
    #[ignore = "probe: prints every vehicle's sound sources"]
    fn probe_vehicle_sound_sources() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        for v in native.tags_by_group(cache, "vehi") {
            let fns = ObjFn::read_all(native, cache, v);
            let (shift, lc) = engine_sound_refs(native, cache, v);
            println!("== {}  fns={} shift={:?} load/cruise={:?}", native.tag_name(cache, v).unwrap_or_default(), fns.len(), shift, lc);
        }
    }
}

/// Gates. The pure ones run in the normal suite; the ones that read the real `forge_halo` cache
/// are `#[ignore]` like every other native-opening gate and run with
/// `cargo test --release -p hms-app play_sound::gates -- --ignored --test-threads=1`.
///
/// **Nothing here opens an audio device.** The engine is [`SoundEngine::offline`]: the real
/// mixer, rendered into a buffer.
#[cfg(test)]
mod gates {
    use super::*;
    use crate::play_ctl::MoveInput;
    use crate::play_mode::ride_harness::{Ride, DT};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const RATE: u32 = 48_000;
    const FRAMES_PER_TICK: usize = 800; // RATE / 60

    // ---- pure gates -------------------------------------------------------------------------

    /// `sub_140E60060`, as decompiled: 1.0 at the shift, 0.5 at the middle of the 0.45 s, back to
    /// 1.0 at the end and from `seconds_to_ticks(0.45)` on.
    #[test]
    fn the_shift_ramp_dips_to_half_and_returns() {
        assert!((shift_ramp(0) - 1.0).abs() < 1e-6);
        let limit = seconds_to_ticks(0.45) as u8;
        assert_eq!(limit, 28, "0.45 s is 28 ticks at the game's 60 Hz");
        assert_eq!(shift_ramp(limit), 1.0);
        assert_eq!(shift_ramp(254), 1.0);
        // The minimum is near the middle and is 0.5.
        let min = (0..limit).map(|t| shift_ramp(t)).fold(f32::MAX, f32::min);
        assert!(min >= 0.5 && min < 0.52, "minimum {min}");
        let argmin = (0..limit).min_by(|a, b| shift_ramp(*a).partial_cmp(&shift_ramp(*b)).unwrap()).unwrap();
        assert!((12..=14).contains(&argmin), "the dip is mid-ramp, not at {argmin}");
        // Monotone down then up.
        for t in 1..=argmin {
            assert!(shift_ramp(t) <= shift_ramp(t - 1) + 1e-6, "falling at {t}");
        }
        for t in argmin + 1..limit {
            assert!(shift_ramp(t) >= shift_ramp(t - 1) - 1e-6, "rising at {t}");
        }
    }

    fn func(import: &str, export: &str, off_with: &str, flags: u32) -> ObjFn {
        ObjFn {
            flags,
            import: import.into(),
            export: export.into(),
            off_with: off_with.into(),
            scale_by: String::new(),
            min_value: 0.0,
            map: None,
        }
    }

    /// The function rules `sub_140D47160` implements: `turn off with`, `always active`, and an
    /// unknown input is `None` (reported), never a guess.
    #[test]
    fn object_functions_obey_turn_off_with_and_always_active() {
        let fns = vec![func("engine_power", "snd_power", "driver_seat_occupied", FN_FLAG_ALWAYS_ACTIVE)];
        let mut un = BTreeSet::new();
        let on = Inputs { engine_power: 0.0, occupied: true, ..Default::default() };
        // Driven, engine at zero: ALWAYS ACTIVE keeps it on at value 0.
        assert_eq!(value_of(&fns, "snd_power", &on, 0, &mut un), Some((0.0, true)));
        // Nobody in the seat: the function is off whatever the engine does.
        let off = Inputs { engine_power: 0.8, occupied: false, ..Default::default() };
        assert_eq!(value_of(&fns, "snd_power", &off, 0, &mut un), Some((0.0, false)));
        // Without ALWAYS ACTIVE a zero value is inactive.
        let plain = vec![func("engine_power", "p", "", 0)];
        assert_eq!(value_of(&plain, "p", &on, 0, &mut un).map(|v| v.1), Some(false));
        let loud = Inputs { engine_power: 0.4, occupied: true, ..Default::default() };
        assert_eq!(value_of(&plain, "p", &loud, 0, &mut un), Some((0.4, true)));
        assert!(un.is_empty());
        // An input this module does not have: None, and it is recorded.
        let odd = vec![func("flux_capacitor", "q", "", 0)];
        assert_eq!(value_of(&odd, "q", &loud, 0, &mut un), None);
        assert!(un.contains("flux_capacitor"));
        // `invert` is one minus the result.
        let inv = vec![func("engine_power", "i", "", FN_FLAG_INVERT)];
        let (v, _) = value_of(&inv, "i", &loud, 0, &mut un).unwrap();
        assert!((v - 0.6).abs() < 1e-6);
    }

    fn hand_snd(flags: u16, ranges: &[((f32, f32), (f32, f32))]) -> LoopSnd {
        use crate::reach_sound::{CacheSound, Distances};
        LoopSnd {
            snd: 0,
            cs: CacheSound { flags, ..Default::default() },
            ranges: ranges
                .iter()
                .map(|(bend, full)| crate::sound::LoopRange { samples: vec![0], perm_gain_db: vec![0], natural_pitch: 0.0, bend: *bend, full: *full })
                .collect(),
            scale_gain_db: (0.0, 0.0),
            scale_pitch: (0.0, 1000.0),
            playback: None,
            dist: Distances::default(),
            spatial: false,
        }
    }

    /// An all-ranges sound plays every range whose bend bounds contain the pitch, full volume
    /// inside the full-gain bounds and `interp_db(2, -64, 0, t)` between the bend and full edges.
    #[test]
    fn overlapping_pitch_ranges_crossfade_edge_to_full() {
        let s = hand_snd(0x2000, &[((0.0, 500.0), (0.0, 300.0)), ((200.0, 800.0), (400.0, 800.0))]);
        // 100: only range 0, inside its full zone.
        assert_eq!(s.ranges_at(100.0, None), vec![(0, 0.0)]);
        // 350: both play. Range 0 is above its full zone, a quarter of the way back toward its
        // bend bound; range 1 is below its full zone, three quarters of the way in.
        let r = s.ranges_at(350.0, None);
        assert_eq!(r.len(), 2);
        let want0 = crate::reach_sound::interp_db(2, -64.0, 0.0, (350.0 - 500.0) / (300.0 - 500.0));
        let want1 = crate::reach_sound::interp_db(2, -64.0, 0.0, (350.0 - 200.0) / (400.0 - 200.0));
        assert!((r[0].1 - want0).abs() < 1e-4 && (r[1].1 - want1).abs() < 1e-4, "{r:?} vs {want0} {want1}");
        // Both are quieter than full volume, and neither is silent.
        assert!(r.iter().all(|(_, db)| *db < 0.0 && *db > -64.0));
        // Past every bend bound: nothing plays.
        assert!(s.ranges_at(900.0, None).is_empty());
        // A sound without the flag plays ONE range, the one that holds the pitch.
        let single = hand_snd(0, &[((0.0, 500.0), (0.0, 300.0)), ((200.0, 800.0), (400.0, 800.0))]);
        assert_eq!(single.ranges_at(600.0, None).len(), 1);
        assert_eq!(single.ranges_at(600.0, None)[0].0, 1);
        // ...and keeps its current range while the pitch is inside its MAX-GAIN bounds
        // (`sub_1809B0080`), here 100..300 for range 0.
        assert_eq!(single.ranges_at(250.0, Some(0))[0].0, 0);
        assert_eq!(single.ranges_at(450.0, Some(1))[0].0, 1);
        // Outside every max-gain bound (350) the nearest BEND wins, a pitch inside a bend with a
        // negative distance: range 0's bend starts further below 350 than range 1's does.
        assert_eq!(single.ranges_at(350.0, Some(1))[0].0, 0);
    }

    // The audit (signal analysis of the real assets + the event audit) lives in its own file.
    include!("play_sound_audit.rs");

    // ---- native gates (the real forge_halo cache, the real PlayMode, an OFFLINE mixer) -------

    struct Rig {
        _g: std::sync::MutexGuard<'static, ()>,
        native: Arc<NativeDll>,
        cache: CacheHandle,
        map: PathBuf,
        eng: SoundEngine,
        ps: PlaySound,
        /// What the HUD would tell the host this frame (the audit's scripted session sets them).
        reloading: bool,
        dry_pulls: u32,
    }

    fn rig() -> Option<Rig> {
        let (g, native, cache) = crate::reach_actor::tests::open_reach_for("forge_halo")?;
        let map = crate::mapcat::enumerate()
            .into_iter()
            .find(|c| c.game == crate::mapcat::Game::Reach && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo")))?
            .path;
        let mut eng = SoundEngine::offline(RATE);
        let deadline = Instant::now() + Duration::from_secs(240);
        while !eng.begin_frame(&native, cache, &map) {
            assert!(Instant::now() < deadline, "the sound banks never opened: {:?}", eng.status());
            std::thread::sleep(Duration::from_millis(20));
        }
        // The period/level gates measure the ENGINE; the ambience bed gets its own gate.
        let ps = PlaySound { mute_ambience: true, ..Default::default() };
        Some(Rig { _g: g, native, cache, map, eng, ps, reloading: false, dry_pulls: 0 })
    }

    fn listener_of(pos: Vec3, yaw: f32) -> Listener {
        let (sy, cy) = yaw.sin_cos();
        let fwd = Vec3::new(cy, sy, 0.0);
        let left = Vec3::new(-sy, cy, 0.0);
        (pos.to_array(), fwd.to_array(), left.to_array(), Vec3::Z.to_array())
    }

    /// A seated driver in `vehicle` (matched by path suffix). The vehicle sits at the origin
    /// facing +X; the listener is the driver's camera.
    fn seated(rig: &Rig, suffix: &str) -> Option<(Ride, u32)> {
        let mut r = Ride::with_vehicle(rig.native.clone(), rig.cache, suffix)?;
        let tag = r.objects[0].primary_tag;
        r.force_seat(tag, 0).ok()?;
        Some((r, tag))
    }

    /// One 60 Hz frame of the PRODUCTION path: the ride ticks, test mode hands its state to
    /// `drive_sound`, and the mixer renders the 1/60 s that frame covers.
    fn frame(rig: &mut Rig, r: &mut Ride, input: &MoveInput, listener: Option<Listener>) -> Vec<f32> {
        r.tick(input);
        step_audio(rig, r, listener, false)
    }

    fn step_audio(rig: &mut Rig, r: &mut Ride, listener: Option<Listener>, no_render: bool) -> Vec<f32> {
        let (pos, yaw, _pitch) = r.pm.camera();
        let host = Host {
            native: &rig.native,
            cache: rig.cache,
            map_path: &rig.map,
            dt: DT,
            listener: listener.unwrap_or_else(|| listener_of(pos, yaw)),
            reloading: rig.reloading,
            dry_pulls: rig.dry_pulls,
            ground_material: None,
        };
        r.pm.drive_sound(&mut rig.ps, &mut rig.eng, &host);
        if no_render { Vec::new() } else { rig.eng.render(FRAMES_PER_TICK) }
    }

    fn omega_max(rig: &Rig, tag: u32) -> f32 {
        match crate::reach_vehicle::VehiclePhysics::read(&rig.native, rig.cache, tag).map(|p| p.family) {
            Some(crate::reach_vehicle::Family::Jeep { engine, .. }) => engine.max_angular_velocity,
            _ => panic!("not a jeep"),
        }
    }

    /// Hold the flywheel at `frac` of its maximum (the gate sets the engine state; the audio
    /// path is what is under test) and run `seconds`, returning the LAST `keep` seconds of audio.
    fn hold_rpm(rig: &mut Rig, r: &mut Ride, frac: f32, seconds: f32, keep: f32, listener: Option<Listener>) -> Vec<f32> {
        let max = omega_max(rig, r.objects[0].primary_tag);
        // A listener at the driver's seat, fixed: without ticking the ride the seat camera would
        // still be blending in from where the player stood.
        let listener = listener.or_else(|| Some(listener_of(Vec3::new(-1.5, 0.0, 1.5), 0.0)));
        let n = (seconds / DT).round() as usize;
        let keep_frames = (keep * 60.0) as usize * FRAMES_PER_TICK;
        let mut out: Vec<f32> = Vec::new();
        for _ in 0..n {
            // No `r.tick`: the physics would wobble the hull and the camera and put that into the
            // level; the audio path reads the state held here.
            if let Some(rd) = r.pm.riding.as_mut() {
                rd.drive.engine.angular_velocity = frac * max;
                rd.drive.engine.idle_ticks = 254;
                // Hold the hull still: a flywheel forced to a speed pulls the vehicle along, and
                // a moving source changes the level and the pan.
                rd.drive.vel = Vec3::ZERO;
                rd.drive.ang_vel = Vec3::ZERO;
            }
            let a = step_audio(rig, r, listener, false);
            out.extend_from_slice(&a);
            if out.len() > keep_frames * 2 {
                out.drain(..out.len() - keep_frames * 2);
            }
        }
        let want = keep_frames * 2;
        out[out.len().saturating_sub(want)..].to_vec()
    }

    fn mono(x: &[f32]) -> Vec<f32> {
        x.chunks_exact(2).map(|c| c[0] + c[1]).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    /// The period, in output frames, at which `x` repeats, searched within `guess * (1 +- tol)`.
    /// Returns `(period, normalised correlation at that lag)`. A decimated search, refined at
    /// full rate: an exactly looping voice correlates at 1.0 at its loop period.
    fn period_near(x: &[f32], guess: f64, tol: f64) -> (f64, f32) {
        let corr = |a: &[f32], b: &[f32]| {
            let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
            for (p, q) in a.iter().zip(b) {
                ab += f64::from(*p) * f64::from(*q);
                aa += f64::from(*p) * f64::from(*p);
                bb += f64::from(*q) * f64::from(*q);
            }
            (ab / (aa * bb).sqrt().max(1e-30)) as f32
        };
        let d = 8usize;
        let y: Vec<f32> = x.chunks_exact(d).map(|c| c.iter().sum::<f32>() / d as f32).collect();
        let (lo, hi) = (((guess * (1.0 - tol)) / d as f64) as usize, ((guess * (1.0 + tol)) / d as f64) as usize + 1);
        let m = y.len().saturating_sub(hi + 1).min(20_000);
        let mut best = (lo, f32::MIN);
        for lag in lo..=hi {
            let c = corr(&y[..m], &y[lag..lag + m]);
            if c > best.1 {
                best = (lag, c);
            }
        }
        let c0 = best.0 * d;
        let m = x.len().saturating_sub(c0 + 2 * d + 1).min(60_000);
        let mut fine = (c0, f32::MIN);
        for lag in c0.saturating_sub(2 * d)..=c0 + 2 * d {
            let c = corr(&x[..m], &x[lag..lag + m]);
            if c > fine.1 {
                fine = (lag, c);
            }
        }
        (fine.0 as f64, fine.1)
    }

    fn warthog(rig: &Rig) -> u32 {
        rig.native
            .tags_by_group(rig.cache, "vehi")
            .into_iter()
            .find(|&t| rig.native.tag_name(rig.cache, t).as_deref() == Some("objects\\vehicles\\human\\warthog\\warthog"))
            .expect("forge_halo carries the Warthog")
    }

    /// Keep one real source and strip its alternate loop, so the output is ONE looping voice per
    /// pitch range and its period can be measured. Everything else is the production path.
    fn isolate_first_source(rig: &mut Rig, tag: u32) {
        let vs = rig.ps.vehicles.get_mut(&tag).expect("the vehicle's sounds were resolved by the first frame");
        vs.sources.retain(|s| s.lsnd_name.ends_with("prototype_load_left"));
        assert_eq!(vs.sources.len(), 1, "the Warthog's left engine source");
        for t in &mut vs.sources[0].tracks {
            t.alt = None;
        }
        vs.load_cruise.clear();
    }

    /// Run until the first frames have resolved the vehicle (the banks decode on a worker).
    fn prime(rig: &mut Rig, r: &mut Ride) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let _ = frame(rig, r, &MoveInput::default(), None);
            if rig.ps.ride_voices() > 0 {
                break;
            }
            assert!(Instant::now() < deadline, "no engine voice ever started: unresolved {:?}", rig.ps.unresolved);
            std::thread::sleep(Duration::from_millis(15));
        }
        // ...and every sample the vehicle can play, so a gate that sweeps the RPM in a fraction
        // of a second is not racing the decoder (in the app the prompt warms them).
        let tag = r.objects[0].primary_tag;
        let all: Vec<usize> = rig.ps.vehicles[&tag]
            .sources
            .iter()
            .flat_map(|s| s.tracks.iter())
            .flat_map(|t| t.lp.iter().chain(t.alt.iter()))
            .flat_map(|l| l.ranges.iter().flat_map(|r| r.samples.iter().copied()))
            .collect();
        let deadline = Instant::now() + Duration::from_secs(120);
        for s in all {
            while rig.eng.sample(s).is_none() {
                assert!(Instant::now() < deadline, "sample {s} never decoded");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    /// **THE PITCH GATE.** The Warthog's engine pitch follows the TAG's curve: the playback rate
    /// of a held-RPM engine, measured as the loop period of the mixer's OUTPUT, equals
    /// `2^((pitch - natural pitch) / 1200)` with `pitch = lerp(scale.pitch lower, upper, scale)`
    /// and `scale` the tag's own mapping of the engine input.
    ///
    /// The measurement is independent of the implementation: a looping voice repeats every
    /// `N / rate` output frames, whatever produced `rate`.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_warthog_engine_pitch_follows_the_tags_curve() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let tag = warthog(&rig);
        let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("a seated Warthog");
        prime(&mut rig, &mut r);
        isolate_first_source(&mut rig, tag);
        // The tag's own numbers, read straight out of what the production reader resolved.
        let vs = rig.ps.vehicles[&tag].clone();
        let src = &vs.sources[0];
        let ls = src.tracks[0].lp.as_ref().expect("a loop");
        assert!(ls.all_ranges_at_once(), "the Warthog loop plays all its pitch ranges");
        let f = vs.fns.iter().find(|f| f.export == src.scale_name).expect("the engine function");
        let map = f.map.as_ref().expect("the function's mapping curve");
        // Find rpm fractions whose pitch sits inside exactly ONE pitch range, in two different
        // places, so the voice is a single looping sample at a known rate.
        let mut singles: Vec<(f32, f32, usize)> = Vec::new(); // (frac, pitch, range)
        for k in 5..=100 {
            let frac = k as f32 / 100.0;
            let pitch = ls.pitch_at(map.eval(frac));
            let rg = ls.ranges_at(pitch, None);
            if rg.len() == 1 && rg[0].1 == 0.0 {
                singles.push((frac, pitch, rg[0].0));
            }
        }
        // Three operating points: two in the SAME pitch range at different pitches (the cents
        // lerp), and one in a DIFFERENT range (the natural pitch it is measured against).
        let mut best: Option<(usize, (f32, f32, usize), (f32, f32, usize))> = None;
        for ri in 0..ls.ranges.len() {
            let mine: Vec<_> = singles.iter().filter(|s| s.2 == ri).copied().collect();
            if let (Some(lo), Some(hi)) = (mine.first().copied(), mine.last().copied()) {
                if best.map_or(true, |b| (hi.1 - lo.1) > (b.2 .1 - b.1 .1)) {
                    best = Some((ri, lo, hi));
                }
            }
        }
        let (pair_range, lo, hi) = best.expect("a single-range operating span");
        let other = singles.iter().find(|s| s.2 != pair_range).copied().expect("a second range");
        let points = vec![lo, hi, other];
        println!("operating points (rpm fraction, pitch cents, range): {points:?}");
        assert!(hi.1 - lo.1 > 100.0, "the same-range pair must differ in pitch: {lo:?} {hi:?}");
        let mut measured = Vec::new();
        for (frac, pitch, ri) in points.iter().copied() {
            let range = &ls.ranges[ri];
            let sample = range.samples[0];
            // The loop as the mixer plays it: the seamless variant (a crossfade shortens it).
            let pcm = loop {
                if let Some(p) = rig.eng.loop_variant(sample) { break p }
                std::thread::sleep(Duration::from_millis(10));
            };
            let ratio = f64::from(2f32.powf((pitch - range.natural_pitch) / 1200.0));
            let want_period = pcm.frames() as f64 * (f64::from(RATE) / f64::from(pcm.sample_rate)) / ratio;
            // Long enough to hold two and a half periods after the scale has settled.
            let keep = (want_period / f64::from(RATE) * 2.6 + 0.5) as f32;
            let audio = hold_rpm(&mut rig, &mut r, frac, 7.0 + keep, keep, None);
            let y = mono(&audio);
            let (period, c) = period_near(&y, want_period, 0.30);
            let got_ratio = pcm.frames() as f64 / period;
            println!("frac {frac:.2} pitch {pitch:.1} range {ri}: want ratio {ratio:.4} (period {want_period:.1}), measured period {period} corr {c:.4} -> ratio {got_ratio:.4}");
            // 0.9, not 1: the true period is fractional (e.g. 418853.8 frames) and the search lag
            // is a whole frame, so the 8-10 kHz content of the loop is out of phase by a fraction
            // of a sample. A voice that is not a loop at that lag correlates at ~0.
            assert!(c > 0.9, "the output is not a clean loop at that period (corr {c})");
            assert!((got_ratio / ratio - 1.0).abs() < 1.0e-4, "playback ratio {got_ratio} vs the tag's {ratio} at frac {frac}");
            measured.push((pitch, got_ratio));
        }
        // Same range, higher pitch: plays faster, by exactly the cents between them.
        let want = 2f64.powf(f64::from(measured[1].0 - measured[0].0) / 1200.0);
        assert!((measured[1].1 / measured[0].1 / want - 1.0).abs() < 2.0e-4, "{measured:?}");
    }

    /// **REVVING WITH THE ENGINE, end to end.** The real Warthog is driven flat out through the
    /// real `PlayMode` and drivetrain, and the engine sound follows the flywheel: its pitch
    /// tracks the RPM the drivetrain reports, it dips and recovers across gear changes, the shift
    /// sound plays, and the mix never goes silent.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_engine_sound_revs_with_the_drivetrain_while_driving() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let tag = warthog(&rig);
        let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("a seated Warthog");
        prime(&mut rig, &mut r);
        let wmax = omega_max(&rig, tag);
        let go = MoveInput { fwd: 1.0, yaw: 0.0, ..Default::default() };
        let mut rec: Vec<(f32, f32)> = Vec::new(); // (rpm fraction, requested pitch of the left engine source)
        let mut win = Vec::new();
        let mut gears = std::collections::BTreeSet::new();
        for _ in 0..(30.0 / DT) as usize {
            let a = frame(&mut rig, &mut r, &go, None);
            win.push(rms(&a));
            if win.len() % 30 == 1 {
                let rd = r.pm.riding.as_ref().unwrap();
                println!("  t {:5.2}s  gear {:2}  rpm {:.2}  speed {:5.2}  pitch {:6.0}  rms {:.4}", win.len() as f32 * DT, rd.drive.engine.gear, rd.drive.engine.angular_velocity / wmax, rd.drive.vel.length(), rig.ps.stats.last.first().map_or(0.0, |x| x.2), rms(&a));
            }
            if let Some(rd) = r.pm.riding.as_ref() {
                gears.insert(rd.drive.engine.gear);
                if let Some(p) = rig.ps.stats.last.iter().find(|x| x.0 == 0) {
                    rec.push((rd.drive.engine.angular_velocity / wmax, p.2));
                }
            }
        }
        let speed = r.pm.riding.as_ref().map_or(0.0, |rd| rd.drive.vel.length());
        let (om_min, om_max) = rec.iter().fold((f32::MAX, f32::MIN), |m, x| (m.0.min(x.0), m.1.max(x.0)));
        let (p_min, p_max) = rec.iter().fold((f32::MAX, f32::MIN), |m, x| (m.0.min(x.1), m.1.max(x.1)));
        println!("drove to {speed:.1} wu/s; gears {gears:?}; rpm {om_min:.2}..{om_max:.2}; pitch {p_min:.0}..{p_max:.0} cents; shifts played {}", rig.ps.stats.shifts_played);
        assert!(speed > 4.0, "the Warthog must actually be driven: {speed}");
        assert!(gears.len() >= 3, "it must go through gears: {gears:?}");
        assert!(rig.ps.stats.shifts_triggered >= 2, "each gear change fires the shift sound: {}", rig.ps.stats.shifts_triggered);
        assert!(om_max - om_min > 0.3 && p_max - p_min > 300.0, "the engine must rev: rpm {om_min}..{om_max}, pitch {p_min}..{p_max}");
        // Pitch follows the RPM (it lags by the 0.5/s scale slew, so this is a strong, not a
        // perfect, correlation).
        let n = rec.len() as f32;
        let (mx, my) = (rec.iter().map(|x| x.0).sum::<f32>() / n, rec.iter().map(|x| x.1).sum::<f32>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
        for (x, y) in &rec {
            sxy += (x - mx) * (y - my);
            sxx += (x - mx).powi(2);
            syy += (y - my).powi(2);
        }
        let corr = sxy / (sxx * syy).sqrt().max(1e-9);
        println!("correlation of pitch with the flywheel: {corr:.3}");
        assert!(corr > 0.7, "pitch must follow the drivetrain's RPM, correlation {corr}");
        // (The first half second is the 0.2 s fade-in.)
        let quiet: Vec<usize> = win.iter().enumerate().skip(30).filter(|(_, w)| **w <= 1e-4).map(|(i, _)| i).collect();
        assert!(quiet.is_empty(), "the engine must never fall silent while driving: {} silent frames, first at {:?} (rpm/pitch there {:?}), rms around it {:?}", quiet.len(), quiet.first(), quiet.first().and_then(|i| rec.get(*i)), quiet.first().map(|i| &win[i.saturating_sub(3)..(*i + 3).min(win.len())]));
    }

    /// **LOOPING.** Every pitch-range sample of the Warthog's engine loops without a click: the
    /// step from its last frame to its first is no larger than the largest step inside it (a cut
    /// loop jumps by about the signal's amplitude, an order of magnitude more). And the voice
    /// really does loop in the mixer: it is still sounding long after one pass of the sample.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_engine_loops_have_no_click_and_really_loop() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let tag = warthog(&rig);
        let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("a seated Warthog");
        prime(&mut rig, &mut r);
        let vs = rig.ps.vehicles[&tag].clone();
        let mut checked = 0;
        for src in &vs.sources {
            for tr in &src.tracks {
                for ls in tr.lp.iter().chain(tr.alt.iter()) {
                    for rg in &ls.ranges {
                        let Some(&idx) = rg.samples.first() else { continue };
                        // What the mixer loops: the seamless variant.
                        let pcm = loop {
                            if let Some(p) = rig.eng.loop_variant(idx) { break p }
                            std::thread::sleep(Duration::from_millis(10));
                        };
                        let n = pcm.frames();
                        let ch = pcm.channels as usize;
                        for c in 0..ch.min(2) {
                            let at = |i: usize| pcm.samples[i * ch + c];
                            let seam = (at(0) - at(n - 1)).abs();
                            let biggest = (1..n).map(|i| (at(i) - at(i - 1)).abs()).fold(0.0f32, f32::max);
                            assert!(seam <= biggest.max(1e-6), "{}: loop seam {seam} exceeds the largest interior step {biggest} (channel {c})", src.lsnd_name);
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 8, "only {checked} engine loop samples were checked");
        // And the mixer loops them: isolate one source at a held RPM and compare the first and
        // the last second of a stretch several loop lengths long.
        isolate_first_source(&mut rig, tag);
        let audio = mono(&hold_rpm(&mut rig, &mut r, 0.30, 12.0, 12.0, None));
        let sec = RATE as usize;
        let (head, tail) = (rms(&audio[4 * sec..5 * sec]), rms(&audio[audio.len() - sec..]));
        assert!(head > 1e-4 && tail > 0.5 * head, "the loop died: rms {head} -> {tail}");
    }

    /// **START AND STOP.** Nothing sounds before the player is in the driver's seat; getting in
    /// starts the engine; getting out fades it and stops every voice.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_engine_starts_when_the_driver_sits_and_stops_when_he_leaves() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let mut r = Ride::with_vehicle(rig.native.clone(), rig.cache, "\\warthog\\warthog").expect("a Warthog");
        let tag = r.objects[0].primary_tag;
        // On foot: no engine voice.
        let mut peak_before = 0.0f32;
        for _ in 0..90 {
            let a = frame(&mut rig, &mut r, &MoveInput::default(), None);
            peak_before = a.iter().fold(peak_before, |m, v| m.max(v.abs()));
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(rig.ps.ride_voices(), 0, "no engine while the player is on foot");
        // Sit down.
        r.force_seat(tag, 0).expect("seat");
        prime(&mut rig, &mut r);
        let mut peak_seated = 0.0f32;
        for _ in 0..120 {
            let a = frame(&mut rig, &mut r, &MoveInput::default(), None);
            peak_seated = a.iter().fold(peak_seated, |m, v| m.max(v.abs()));
        }
        assert!(rig.ps.ride_voices() >= 2, "the engine voices must be live while seated: {}", rig.ps.ride_voices());
        assert!(peak_seated > 10.0 * peak_before.max(1e-5), "seated {peak_seated} vs on foot {peak_before}");
        let voices_seated = rig.eng.mixer_voices();
        // Get out: through the production exit.
        let (native, cache) = (r.native.clone(), r.cache);
        let msg = r.pm.exit_vehicle(&native, cache);
        println!("exit: {msg}");
        let mut last_peak = f32::MAX;
        for _ in 0..(4.0 / DT) as usize {
            let a = frame(&mut rig, &mut r, &MoveInput::default(), None);
            last_peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        }
        assert_eq!(rig.ps.ride_voices(), 0, "every engine voice must be stopped after the exit fade");
        assert!(rig.eng.mixer_voices() < voices_seated, "the mixer still holds the engine");
        assert!(last_peak < 1e-4, "the engine is still audible after getting out: peak {last_peak}");
    }

    /// **POSITION.** The engine is heard from where the vehicle is: a vehicle to the listener's
    /// left sounds from the left and one to the right from the right, it fades with distance by
    /// the tag's own curve, and it is gone beyond the tag's `maximum distance`.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_engine_is_heard_from_where_the_vehicle_is() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let tag = warthog(&rig);
        let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("a seated Warthog");
        prime(&mut rig, &mut r);
        let w_max = omega_max(&rig, tag);
        let at = |rig: &mut Rig, r: &mut Ride, pos: Vec3| -> Vec<f32> {
            let l = listener_of(Vec3::ZERO, 0.0); // the listener: origin, facing +X, left = +Y
            let mut out = Vec::new();
            // 15 s, and the last 12 s measured: a level is an average over more than one pass of
            // the 9-10 s engine loop, whose own loudness moves.
            for i in 0..900 {
                if let Some(rd) = r.pm.riding.as_mut() {
                    rd.drive.pos = pos;
                    rd.drive.vel = Vec3::ZERO;
                    rd.drive.ang_vel = Vec3::ZERO;
                    rd.drive.engine.angular_velocity = 0.4 * w_max;
                    rd.drive.engine.idle_ticks = 254;
                }
                let a = step_audio(rig, r, Some(l), false);
                // The last second, so a level is an average and not one 16 ms slice of a waveform.
                if i >= 180 {
                    out.extend_from_slice(&a);
                }
            }
            out
        };
        let lr = |a: &[f32]| {
            let (mut l, mut rr) = (0.0f32, 0.0f32);
            for c in a.chunks_exact(2) {
                l += c[0] * c[0];
                rr += c[1] * c[1];
            }
            (l.sqrt(), rr.sqrt())
        };
        let left = lr(&at(&mut rig, &mut r, Vec3::new(0.0, 8.0, 0.0)));
        let right = lr(&at(&mut rig, &mut r, Vec3::new(0.0, -8.0, 0.0)));
        println!("vehicle on the left: L {:.4} R {:.4};  on the right: L {:.4} R {:.4}", left.0, left.1, right.0, right.1);
        assert!(left.0 > 1.5 * left.1, "a vehicle on the left must sound from the left: {left:?}");
        assert!(right.1 > 1.5 * right.0, "a vehicle on the right must sound from the right: {right:?}");
        // Distance: ahead of the listener, so pan does not confound the level. ONE source (the
        // LOD loop is a far sound with its own curve and would add to the far reading).
        isolate_first_source(&mut rig, tag);
        let marker = rig.ps.marker_cache[&tag]
            .get(&rig.ps.vehicles[&tag].sources[0].marker)
            .copied()
            .expect("the source's marker resolved");
        let near = rms(&mono(&at(&mut rig, &mut r, Vec3::new(10.0, 0.0, 0.0))));
        let far = rms(&mono(&at(&mut rig, &mut r, Vec3::new(30.0, 0.0, 0.0))));
        let d = rig.ps.vehicles[&tag].sources[0].tracks[0].lp.as_ref().expect("a loop").dist;
        // Where the marker really was, in each case: the hull's own basis times the marker.
        let hull = |pos: Vec3| Mat4::from_rotation_translation(r.pm.riding.as_ref().unwrap().drive.basis(), pos).transform_point3(marker).length();
        let (dn, df) = (hull(Vec3::new(10.0, 0.0, 0.0)), hull(Vec3::new(30.0, 0.0, 0.0)));
        let want = f64::from(d.gain_at(df) / d.gain_at(dn));
        println!("near {near:.5} (marker {dn:.2} wu) far {far:.5} (marker {df:.2} wu): measured ratio {:.3}, the tag's distance curve says {want:.3} (min {} max {} wu)", far / near, d.minimum, d.maximum);
        assert!(near > 1e-4 && far < near, "{near} {far}");
        assert!((f64::from(far / near) / want - 1.0).abs() < 0.10, "level ratio {} vs the tag's curve {want}", far / near);
        let gone = rms(&mono(&at(&mut rig, &mut r, Vec3::new(d.maximum + 20.0, 0.0, 0.0))));
        assert!(gone < 1e-4 * 0.2 + 1e-6 || gone < near * 0.02, "beyond the maximum distance it must be silent, got {gone} (near {near})");
    }

    /// **THE SHIFT.** A gear change plays the engine's `gear shift sound` once, at the vehicle.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_gear_change_plays_the_shift_sound_once() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let tag = warthog(&rig);
        let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("a seated Warthog");
        prime(&mut rig, &mut r);
        let clip = rig.ps.vehicles[&tag].shift.clone().expect("the Warthog authors a gear shift sound");
        println!("shift clip: class {} spatial {} dist {:?} gain {:.4} samples {:?}", clip.class, clip.spatial, clip.dist, clip.gain, clip.samples);
        {
            let law = rs::GainLaw::read(&rig.native, rig.cache);
            let g = rs::Gestalt::read(&rig.native, rig.cache).unwrap();
            let cs = rs::CacheSound::read(&rig.native, rig.cache, clip.snd).unwrap();
            println!("  class {} ({}) bounds {:?}; playback {:?}; scale gain {:?}; perm gains {:?}; master {}",
                cs.class, cs.class_name(), law.class(cs.class).map(|c| c.gain_bounds_db), g.playback_of(&cs).map(|p| (p.gain_base_db, p.gain_variance_db)), g.scale_gain_of(&cs), clip.perm_gain_db, law.master_db);
        }
        // Let the shift sample decode, then change gear and watch for it.
        let deadline = Instant::now() + Duration::from_secs(30);
        while rig.eng.sample(clip.samples[0]).is_none() {
            assert!(Instant::now() < deadline, "the shift sample never decoded");
            std::thread::sleep(Duration::from_millis(10));
        }
        let set_gear = |r: &mut Ride, g: i8| {
            r.tick(&MoveInput::default());
            if let Some(rd) = r.pm.riding.as_mut() {
                rd.drive.engine.gear = g;
                rd.drive.engine.angular_velocity = 4.0;
                rd.drive.engine.idle_ticks = 254;
            }
        };
        for _ in 0..30 {
            set_gear(&mut r, 1);
            step_audio(&mut rig, &mut r, None, false);
        }
        let audible = clip.gain > rs::db_to_linear(rs::MIN_GAIN_DB);
        let (before, played, voices) = (rig.ps.stats.shifts_triggered, rig.ps.stats.shifts_played, rig.eng.mixer_voices());
        set_gear(&mut r, 2);
        step_audio(&mut rig, &mut r, None, false);
        assert_eq!(rig.ps.stats.shifts_triggered, before + 1, "one shift for one gear change");
        // Whether it SOUNDS is the tag's: the Warthog's `warthog_shift` is authored at
        // `gain base` -64 dB, which the engine's own audibility rule (`total_db > -64`) refuses.
        assert_eq!(rig.ps.stats.shifts_played, played + u64::from(audible), "audible = {audible}");
        assert_eq!(rig.eng.mixer_voices(), voices + usize::from(audible));
        for _ in 0..30 {
            set_gear(&mut r, 2);
            step_audio(&mut rig, &mut r, None, false);
        }
        assert_eq!(rig.ps.stats.shifts_triggered, before + 1, "holding the gear changes nothing");
    }

    /// **LIVENESS.** Every vehicle the cache carries that has an engine source is driven by its
    /// OWN tag chain with a driver in the seat, and the count of those that make sound is
    /// asserted (a gate that exercised nothing would pass).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn every_vehicle_with_an_engine_source_is_audible_to_its_driver() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let vehis = rig.native.tags_by_group(rig.cache, "vehi");
        let (mut with_sources, mut audible, mut silent): (usize, usize, Vec<String>) = (0, 0, Vec::new());
        for tag in vehis {
            let name = rig.native.tag_name(rig.cache, tag).unwrap_or_default();
            let vs = VehicleSounds::read(&rig.eng, &rig.native, rig.cache, tag);
            if vs.sources.is_empty() {
                continue;
            }
            with_sources += 1;
            rig.eng.stop_all();
            rig.ps = PlaySound { mute_ambience: true, ..Default::default() };
            let mut peak = 0.0f32;
            let deadline = Instant::now() + Duration::from_secs(20);
            let mut frames = 0;
            // 4 s, but wait (bounded) for the worker to decode the first samples.
            while frames < 240 || (peak == 0.0 && Instant::now() < deadline) {
                let fr = Frame {
                    native: &rig.native,
                    cache: rig.cache,
                    map_path: &rig.map,
                    dt: DT,
                    listener: listener_of(Vec3::new(-2.0, 0.0, 1.0), 0.0),
                    ride: Some(RideFrame {
                        placement_tag: tag,
                        driver: true,
                        world: Mat4::IDENTITY,
                        markers: Default::default(),
                        drivetrain: None,
                        speed: 2.0,
                        speed_reference: 5.0,
                        throttle: 1.0,
                    }),
                    weapon: 0,
                    shots: 0,
                    melee: false,
                    reloading: false,
                    dry_pulls: 0,
                    offered: None,
                    foot: None,
                };
                rig.ps.update(&mut rig.eng, &fr);
                let a = rig.eng.render(FRAMES_PER_TICK);
                peak = a.iter().fold(peak, |m, v| m.max(v.abs()));
                frames += 1;
                if peak == 0.0 {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            if peak > 1e-4 {
                audible += 1;
            } else {
                silent.push(format!("{name} (sources {}, unresolved {:?})", vs.sources.len(), rig.ps.unresolved));
            }
            println!("{name:<70} sources {:>2}  peak {peak:.4}  shift gain {:?}", vs.sources.len(), vs.shift.as_ref().map(|c| c.gain));
        }
        println!("{with_sources} vehicles with engine sources, {audible} audible; silent: {silent:#?}");
        assert!(with_sources >= 12, "only {with_sources} vehicles carry engine sources");
        assert!(audible >= 8, "only {audible} of {with_sources} are audible to their driver");
    }

    /// **THE PLAYER'S OWN WEAPON.** A shot, a swing, a reload and a dry pull each play the
    /// weapon's own authored sound.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_players_weapon_makes_its_own_sounds() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let weap = rig
            .native
            .tags_by_group(rig.cache, "weap")
            .into_iter()
            .find(|&t| rig.native.tag_name(rig.cache, t).map_or(false, |n| n.ends_with("assault_rifle\\assault_rifle")))
            .expect("the assault rifle");
        let run = |rig: &mut Rig, shots: u32, melee: bool, reloading: bool, dry: u32| -> f32 {
            let fr = Frame {
                native: &rig.native,
                cache: rig.cache,
                map_path: &rig.map,
                dt: DT,
                listener: listener_of(Vec3::ZERO, 0.0),
                ride: None,
                weapon: weap,
                shots,
                melee,
                reloading,
                dry_pulls: dry,
                offered: None,
                    foot: None,
            };
            rig.ps.update(&mut rig.eng, &fr);
            rig.eng.render(FRAMES_PER_TICK * 12).iter().fold(0.0f32, |m, v| m.max(v.abs()))
        };
        // Warm: wait for the four samples to decode.
        for _ in 0..400 {
            run(&mut rig, 0, false, false, 0);
            std::thread::sleep(Duration::from_millis(10));
        }
        let quiet = run(&mut rig, 0, false, false, 0);
        let fire = run(&mut rig, 1, false, false, 0);
        let swing = run(&mut rig, 1, true, false, 0);
        let reload = run(&mut rig, 1, true, true, 0);
        let dry = run(&mut rig, 1, true, true, 1);
        println!("quiet {quiet:.4}  fire {fire:.4}  melee {swing:.4}  reload {reload:.4}  dry {dry:.4}  stats {:?}", rig.ps.stats);
        assert_eq!(rig.ps.stats.shots_played, 1, "one shot");
        assert_eq!(rig.ps.stats.melees_played, 1, "one swing");
        assert_eq!(rig.ps.stats.reloads_played, 1, "one reload");
        assert_eq!(rig.ps.stats.dry_played, 1, "one dry pull");
        assert!(fire > 10.0 * quiet.max(1e-5) && swing > 10.0 * quiet.max(1e-5), "fire {fire} swing {swing} quiet {quiet}");
    }

    /// **AMBIENCE.** On foot, with nothing else going on, test mode plays the map's own
    /// acoustics-palette bed: audible, and not distance-attenuated (it is the room tone).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn test_mode_plays_the_maps_ambience_bed() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        rig.ps.mute_ambience = false;
        let mut peak = 0.0f32;
        let deadline = Instant::now() + Duration::from_secs(240);
        while peak < 1e-3 {
            assert!(Instant::now() < deadline, "the ambience bed never started: {:?}", rig.eng.status());
            let fr = Frame {
                native: &rig.native,
                cache: rig.cache,
                map_path: &rig.map,
                dt: DT,
                listener: listener_of(Vec3::ZERO, 0.0),
                ride: None,
                weapon: 0,
                shots: 0,
                melee: false,
                reloading: false,
                dry_pulls: 0,
                offered: None,
                    foot: None,
            };
            rig.ps.update(&mut rig.eng, &fr);
            peak = rig.eng.render(FRAMES_PER_TICK).iter().fold(0.0f32, |m, v| m.max(v.abs()));
            std::thread::sleep(Duration::from_millis(10));
        }
        println!("ambience peak {peak:.4}; status {:?}", rig.eng.status());
        assert!(rig.eng.status().ambience_playing >= 1);
        // And leaving test mode stops it.
        rig.ps.stop(&mut rig.eng);
        // A stop is a 5 ms fade, not a cut (a cut is a click): let the first tick carry it.
        let _ = rig.eng.render(FRAMES_PER_TICK);
        let after = rig.eng.render(FRAMES_PER_TICK * 4).iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(after < 1e-6, "stop() must silence everything: {after}");
    }

    // ---- signal-integrity: what the output callback would produce over a scripted ride ------

    /// Discontinuity census of a mono signal: samples whose first difference is far above the
    /// local scale (a click, a restart, a jump of the playhead).
    fn spikes(y: &[f32]) -> Vec<usize> {
        let d: Vec<f32> = y.windows(2).map(|w| w[1] - w[0]).collect();
        let mut out = Vec::new();
        let win = 256usize;
        for i in win..d.len().saturating_sub(win) {
            if i % 1 != 0 { continue }
            let (mut e, mut n) = (0.0f32, 0.0f32);
            for k in (i - win..i + win).step_by(4) {
                if k != i { e += d[k] * d[k]; n += 1.0; }
            }
            let sc = (e / n).sqrt().max(1e-6);
            if d[i].abs() > 8.0 * sc && d[i].abs() > 0.003 {
                out.push(i);
            }
        }
        out
    }

    /// A scripted ride at a given control-frame cadence. Returns the mono mix the output callback
    /// would have produced, the voice-start count, and per-phase notes.
    fn scripted_ride(rig: &mut Rig, r: &mut Ride, frame_ticks: &[usize], seconds: f32) -> (Vec<f32>, u64, Vec<(f32, u64, u64, Option<(u8, u8, bool, u8, f32)>)>) {
        let mut out: Vec<f32> = Vec::new();
        let mut events = Vec::new();
        let mut t = 0.0f32;
        let mut i = 0usize;
        let started0 = rig.ps.stats.voices_started;
        while t < seconds {
            let n = frame_ticks[i % frame_ticks.len()];
            i += 1;
            // Script: idle, full throttle, coast, reverse/brake.
            let fwd = if t < 3.0 { 0.0 } else if t < 15.0 { 1.0 } else if t < 19.0 { 0.0 } else { -1.0 };
            for _ in 0..n {
                r.tick(&MoveInput { fwd, ..Default::default() });
            }
            let dt = n as f32 * DT;
            let (s0, p0) = (rig.ps.stats.voices_started, rig.ps.stats.voices_stopped);
            let (pos, yaw, _) = r.pm.camera();
            let host = Host { native: &rig.native, cache: rig.cache, map_path: &rig.map, dt, listener: listener_of(pos, yaw), reloading: false, dry_pulls: 0, ground_material: None };
            r.pm.drive_sound(&mut rig.ps, &mut rig.eng, &host);
            if rig.ps.stats.voices_started != s0 || rig.ps.stats.voices_stopped != p0 {
                events.push((t, rig.ps.stats.voices_started - s0, rig.ps.stats.voices_stopped - p0, rig.ps.stats.last_start));
            }
            // The audio thread keeps running for the whole frame, in callback-sized pieces.
            let frames = (dt * RATE as f32) as usize;
            let mut left = frames;
            while left > 0 {
                let c = left.min(512);
                out.extend(mono(&rig.eng.render(c)));
                left -= c;
            }
            t += dt;
        }
        (out, rig.ps.stats.voices_started - started0, events)
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_a_scripted_ride_for_clicks() {
        let Some(mut rig) = rig() else { return };
        for cadence in [vec![1usize], vec![2], vec![3], vec![1, 1, 1, 4, 1, 1, 3]] {
            rig.eng.stop_all();
            rig.ps = PlaySound { mute_ambience: true, ..Default::default() };
            let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("seat");
            prime(&mut rig, &mut r);
            let (y, starts, events) = scripted_ride(&mut rig, &mut r, &cadence, 24.0);
            let sp = spikes(&y);
            let secs = y.len() as f32 / RATE as f32;
            let zeros = y.windows(64).filter(|w| w.iter().all(|v| *v == 0.0)).count();
            for e in &events { println!("    t {:6.2}: +{} -{} {:?}", e.0, e.1, e.2, e.3); }
            println!("cadence {cadence:?}: {secs:.1} s, voice starts {starts}, spikes {} (first {:?}), zero-runs {zeros}", sp.len(), &sp[..sp.len().min(12)].iter().map(|i| *i as f32 / RATE as f32).collect::<Vec<_>>());
        }
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_how_long_the_loops_really_are() {
        let Some(mut rig) = rig() else { return };
        rig.ps.mute_ambience = false;
        for _ in 0..200 {
            let fr = Frame { native: &rig.native, cache: rig.cache, map_path: &rig.map, dt: DT, listener: listener_of(Vec3::ZERO, 0.0), ride: None, weapon: 0, shots: 0, melee: false, reloading: false, dry_pulls: 0, offered: None, foot: None };
            rig.ps.update(&mut rig.eng, &fr);
            let _ = rig.eng.render(800);
            std::thread::sleep(Duration::from_millis(5));
        }
        for (n, samples) in rig.eng.ambience_beds() {
            println!("ambience bed: {n} permutations (bank samples {samples:?})");
            for s in samples {
                let p = loop { if let Some(p) = rig.eng.sample(s) { break p } std::thread::sleep(Duration::from_millis(5)) };
                println!("   sample {s}: {} frames = {:.2} s, {} ch, {} Hz", p.frames(), p.frames() as f32 / p.sample_rate as f32, p.channels, p.sample_rate);
            }
        }
        let g = rs::Gestalt::read(&rig.native, rig.cache).unwrap();
        let tag = warthog(&rig);
        let vs = VehicleSounds::read(&rig.eng, &rig.native, rig.cache, tag);
        for src in &vs.sources {
            for tr in &src.tracks {
                for ls in tr.lp.iter().chain(tr.alt.iter()) {
                    for r in &ls.ranges {
                        for &s in &r.samples {
                            let p = loop { if let Some(p) = rig.eng.sample(s) { break p } std::thread::sleep(Duration::from_millis(5)) };
                            let want = g.permutations.iter().find(|q| rig.eng.sample_subsound(s) == Some(q.fmod_subsound_id)).map(|q| q.sample_count);
                            println!("{} range sample {s}: decoded {} frames, declared {:?}", src.lsnd_name, p.frames(), want);
                        }
                    }
                }
            }
        }
    }

    /// Run `seconds` of a steady engine (flywheel held at `frac`, hull still) with the game thread
    /// delivering a control frame every `frame_ticks[i]` ticks, and the audio thread producing the
    /// whole time in 512-frame callbacks. Returns the mono mix and the voice starts after
    /// `settle` seconds.
    fn steady_engine(rig: &mut Rig, r: &mut Ride, frac: f32, frame_ticks: &[usize], seconds: f32, settle: f32) -> (Vec<f32>, u64) {
        let max = omega_max(rig, r.objects[0].primary_tag);
        let (mut out, mut t, mut i) = (Vec::new(), 0.0f32, 0usize);
        let mut started_at_settle = None;
        while t < seconds {
            let n = frame_ticks[i % frame_ticks.len()];
            i += 1;
            if let Some(rd) = r.pm.riding.as_mut() {
                rd.drive.engine.angular_velocity = frac * max;
                rd.drive.engine.idle_ticks = 254;
                rd.drive.vel = Vec3::ZERO;
                rd.drive.ang_vel = Vec3::ZERO;
            }
            let dt = n as f32 * DT;
            let host = Host { native: &rig.native, cache: rig.cache, map_path: &rig.map, dt, listener: listener_of(Vec3::new(-1.5, 0.0, 1.5), 0.0), reloading: false, dry_pulls: 0, ground_material: None };
            r.pm.drive_sound(&mut rig.ps, &mut rig.eng, &host);
            let mut left = (dt * RATE as f32) as usize;
            while left > 0 {
                let c = left.min(512);
                out.extend(mono(&rig.eng.render(c)));
                left -= c;
            }
            t += dt;
            if started_at_settle.is_none() && t >= settle {
                started_at_settle = Some(rig.ps.stats.voices_started);
            }
        }
        let after = rig.ps.stats.voices_started - started_at_settle.unwrap_or(0);
        (out, after)
    }

    /// **NO RECORD-SKIP IN THE ENGINE.** A steady engine creates no voices (a voice re-created
    /// restarts its sample), has no click, and no dropout, whatever the game's frame time -
    /// including 67 ms frames and a stall in the middle. Measured on the mixer's output.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_steady_engine_never_restarts_clicks_or_drops_out_at_any_frame_time() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        let (mut r, _) = seated(&rig, "\\warthog\\warthog").expect("a seated Warthog");
        prime(&mut rig, &mut r);
        let mut worst = 0usize;
        for cadence in [vec![1usize], vec![2], vec![4], vec![1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1]] {
            for frac in [0.30f32, 0.65] {
                let (y, starts) = steady_engine(&mut rig, &mut r, frac, &cadence, 14.0, 4.0);
                let y = &y[4 * RATE as usize..];
                let sp = spikes(y);
                let zeros = y.windows(64).filter(|w| w.iter().all(|v| *v == 0.0)).count();
                println!("cadence {cadence:?} rpm {frac}: voice starts after settle {starts}, spikes {}, zero-runs {zeros}", sp.len());
                assert_eq!(starts, 0, "a steady engine must not create voices (cadence {cadence:?}, rpm {frac})");
                assert!(sp.is_empty(), "clicks at {:?} s (cadence {cadence:?}, rpm {frac})", sp.iter().take(5).map(|i| *i as f32 / RATE as f32).collect::<Vec<_>>());
                assert_eq!(zeros, 0, "dropouts (cadence {cadence:?}, rpm {frac})");
                worst += sp.len();
            }
        }
        assert_eq!(worst, 0);
    }

    /// **THE BED IS NOT A SHORT LOOP.** forge_halo's ambience is six takes of 3.2-6.3 s. Looping
    /// one of them is a 3-6 second chunk repeating for ever; the output must not correlate with
    /// itself at any single take's length.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_ambience_bed_does_not_repeat_one_short_take() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        rig.ps.mute_ambience = false;
        let mut y: Vec<f32> = Vec::new();
        let mut t = 0usize;
        // Wait for the bed to start, then record 60 s of it.
        let deadline = Instant::now() + Duration::from_secs(240);
        loop {
            let fr = Frame { native: &rig.native, cache: rig.cache, map_path: &rig.map, dt: DT, listener: listener_of(Vec3::ZERO, 0.0), ride: None, weapon: 0, shots: 0, melee: false, reloading: false, dry_pulls: 0, offered: None, foot: None };
            rig.ps.update(&mut rig.eng, &fr);
            let a = mono(&rig.eng.render(FRAMES_PER_TICK));
            if y.is_empty() && a.iter().all(|v| *v == 0.0) {
                assert!(Instant::now() < deadline, "the bed never started");
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            y.extend(a);
            t += 1;
            if t >= 60 * 60 {
                break;
            }
        }
        let beds = rig.eng.ambience_beds();
        assert!(!beds.is_empty() && beds[0].0 >= 2, "the bed must have several takes: {beds:?}");
        let mut lens = Vec::new();
        for s in &beds[0].1 {
            lens.push(rig.eng.sample(*s).expect("decoded").frames());
        }
        let skip = RATE as usize; // past the start
        for len in lens {
            let (_, c) = period_near(&y[skip..skip + 12 * RATE as usize], len as f64, 0.0015);
            println!("take of {len} frames: self-correlation of the bed at that lag {c:.3}");
            assert!(c < 0.5, "the bed repeats a {len}-frame take (correlation {c})");
        }
    }

    /// **ON FOOT, NEVER SILENT.** The owner heard nothing until he shot or drove. A player on foot
    /// standing, walking, running and sprinting must always have the map's ambience under him, and
    /// his feet and foley must be heard at the rate the walk cycle authors (`foot` frame events of
    /// the playing `jmad` clip; the sound is the Spartan's `foot` material effect for the gait).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_player_on_foot_is_never_silent_and_has_footsteps() {
        let Some(mut rig) = rig() else { eprintln!("play_sound gates SKIPPED: no forge_halo cache"); return };
        rig.ps.mute_ambience = false;
        let mut r = Ride::with_vehicle(rig.native.clone(), rig.cache, "\\warthog\\warthog").expect("a ride harness");
        let still = MoveInput::default();
        // Warm: the banks and the first samples decode on a worker.
        // Until the bed is audible (decoding is on a worker thread: on a loaded machine it takes
        // as long as it takes, so wait on the condition and not on a fixed number of frames)...
        let deadline = Instant::now() + Duration::from_secs(240);
        loop {
            let a = frame(&mut rig, &mut r, &still, None);
            if a.iter().any(|v| v.abs() > 1e-5) {
                break;
            }
            assert!(Instant::now() < deadline, "the ambience never became audible: {:?}", rig.eng.status());
            std::thread::sleep(Duration::from_millis(8));
        }
        // ...then walk until the first footstep is heard, which means the foot samples decoded.
        let fwd0 = MoveInput { fwd: 1.0, ..Default::default() };
        while rig.ps.body_stats.steps == 0 {
            frame(&mut rig, &mut r, &fwd0, None);
            assert!(Instant::now() < deadline, "no footstep was ever played: {:?}", rig.ps.body_stats);
            std::thread::sleep(Duration::from_millis(8));
        }
        for _ in 0..120 {
            frame(&mut rig, &mut r, &still, None);
        }
        let mut phase = |rig: &mut Rig, r: &mut Ride, label: &str, inp: MoveInput, secs: f32| -> (f32, f32, u64, u64) {
            let (s0, f0) = (rig.ps.body_stats.steps, rig.ps.body_stats.foley);
            let mut a_all: Vec<f32> = Vec::new();
            for _ in 0..(secs / DT) as usize {
                let a = frame(rig, r, &inp, None);
                a_all.extend_from_slice(&a);
                std::thread::sleep(Duration::from_millis(1));
            }
            let (steps, foley) = (rig.ps.body_stats.steps - s0, rig.ps.body_stats.foley - f0);
            let (rm, pk) = (rms(&a_all), a_all.iter().fold(0.0f32, |m, v| m.max(v.abs())));
            println!("{label:>8}: {secs:.0} s  rms {rm:.4}  peak {pk:.3}  steps {steps} ({:.2}/s)  foley {foley}  anim {:?}", steps as f32 / secs, r.pm.body_clip_state.anim);
            (rm, pk, steps, foley)
        };
        let (rm_stand, _, st_stand, _) = phase(&mut rig, &mut r, "stand", still, 4.0);
        let fwd = MoveInput { fwd: 1.0, ..Default::default() };
        let (rm_walk, _, st_run, _) = phase(&mut rig, &mut r, "run", fwd, 6.0);
        let (_, _, st_slow, _) = phase(&mut rig, &mut r, "walk", MoveInput { fwd: 1.0, walk: true, ..Default::default() }, 6.0);
        let (_, _, st_sprint, fo_sprint) = phase(&mut rig, &mut r, "sprint", MoveInput { fwd: 1.0, sprint: true, ..Default::default() }, 6.0);
        let (_, _, _, _) = phase(&mut rig, &mut r, "jump", MoveInput { jump: true, ..Default::default() }, 0.2);
        let (_, _, _, _) = phase(&mut rig, &mut r, "settle", still, 2.0);
        println!("body {:?}  scenery voices {}", rig.ps.body_stats, rig.ps.scenery_voices());
        assert!(rm_stand > 1e-4, "standing still on foot is SILENT: rms {rm_stand}");
        assert!(rm_walk > 1e-4, "moving on foot is SILENT: rms {rm_walk}");
        assert_eq!(st_stand, 0, "footsteps while standing still");
        assert!(st_run >= 6, "running for 6 s made {st_run} footsteps");
        assert!(st_slow >= 4, "walking for 6 s made {st_slow} footsteps");
        assert!(st_sprint >= 6, "sprinting for 6 s made {st_sprint} footsteps");
        assert!(fo_sprint > 0, "sprinting makes no foley");
    }

    /// The detectors themselves: a click in a noisy signal is found, plain noise and a smooth
    /// sweep are not.
    #[test]
    fn the_click_detector_finds_a_click_and_ignores_noise() {
        let mut seed = 12345u32;
        let mut noise = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02
        };
        let clean: Vec<f32> = (0..20_000).map(|i| (i as f32 * 0.05).sin() * 0.05 + noise()).collect();
        assert!(spikes(&clean).is_empty(), "noise and a sine are not clicks");
        let mut clicked = clean.clone();
        for v in clicked[10_000..].iter_mut() {
            *v += 0.08; // a step: the playhead jumped
        }
        let found = spikes(&clicked);
        assert!(found.iter().any(|i| (*i as i64 - 10_000).abs() <= 2), "{found:?}");
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_where_the_ambience_comes_from() {
        let Some(rig) = rig() else { return };
        let (native, cache) = (&*rig.native, rig.cache);
        let scnr = native.find_scenario(cache).unwrap();
        let bsps = native.enumerate_sbsps(cache, scnr, 16);
        let mut pal = Vec::new();
        let mut clusters = Vec::new();
        for sb in &bsps {
            let p = rs::bsp_acoustics_palette(native, cache, *sb);
            println!("bsp {sb:#x}: bsp palette {} rows", p.len());
            if p.len() > pal.len() { pal = p; }
            clusters.extend(rs::bsp_acoustic_clusters(native, cache, *sb));
        }
        let scpal = rs::acoustics_palette(native, cache, scnr);
        println!("scenario palette {} rows; chosen {} rows; {} clusters", scpal.len(), pal.len(), clusters.len());
        for a in pal.iter().chain(scpal.iter()) {
            let nm = |t: Option<u32>| t.map(|t| native.tag_name(cache, t).unwrap_or_default()).unwrap_or("-".into());
            println!("  row {} '{}': bg {} | weather {} | entry {} | exit {} | cutoff {}", a.index, a.name, nm(a.background_sound), nm(a.weather_sound), nm(a.entry_sound), nm(a.exit_sound), a.ambience_cutoff);
        }
        let mut hist: std::collections::BTreeMap<Option<usize>, usize> = Default::default();
        for c in &clusters { *hist.entry(c.acoustics).or_default() += 1; }
        println!("cluster rows histogram: {hist:?}");
        let mut at: std::collections::BTreeMap<Option<usize>, usize> = Default::default();
        let objs = native.scnr_objects(cache, scnr);
        for o in objs.iter().take(2000) { *at.entry(rs::acoustics_at(&clusters, o.pos)).or_default() += 1; }
        println!("rows at {} object positions: {at:?}", objs.len().min(2000));
        // sound scenery
        let h = native.tag_meta(cache, scnr, 0x1C8, 8).unwrap();
        println!("scenario sound scenery placements: {}", rd_u32(&h, 0));
        let pb = tag_block(native, cache, scnr, 0x1D4, 16, 256);
        for c in pb.chunks_exact(16) {
            if let Some(t) = rs::tag_ref_at(c, 0) {
                let at = rs::object_sound_attachments(native, cache, t);
                println!("  ssce palette {} attachments {:?}", native.tag_name(cache, t).unwrap_or_default(), at.iter().map(|(t, l)| (native.tag_name(cache, *t).unwrap_or_default(), *l)).collect::<Vec<_>>());
            }
        }
        // objects with sound attachments among all placements
        let mut with = 0;
        let mut seen = std::collections::BTreeSet::new();
        for o in &objs {
            if seen.insert(o.obj_tag) {
                let a = rs::object_sound_attachments(native, cache, o.obj_tag);
                if !a.is_empty() { with += 1; println!("  obj {} has {:?}", native.tag_name(cache, o.obj_tag).unwrap_or_default(), a.iter().map(|(t,_)| native.tag_name(cache,*t).unwrap_or_default()).collect::<Vec<_>>()); }
            }
        }
        println!("{with} distinct placed object tags carry sound attachments");
    }

    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_sources_for_coverage() {
        let Some(mut rig) = rig() else { return };
        let (native, cache) = (rig.native.clone(), rig.cache);
        let native = &*native;
        // A: the ambience bed's gain chain.
        let scnr = native.find_scenario(cache).unwrap();
        let pal = rs::acoustics_palette(native, cache, scnr);
        let law = rs::GainLaw::read(native, cache);
        let g = rs::Gestalt::read(native, cache).unwrap();
        for a in pal.iter().take(2) {
            let Some(l) = a.background_sound.and_then(|t| rs::LoopingSound::read(native, cache, t)) else { continue };
            for tr in &l.tracks {
                for (lab, t) in [("loop", tr.loop_snd), ("in", tr.intro)] {
                    let Some(t) = t else { continue };
                    let cs = rs::CacheSound::read(native, cache, t).unwrap();
                    let pb = g.playback_of(&cs);
                    println!("bed '{}' track gain {} {lab}: class {} bounds {:?} base {:?} scale {:?} flags {:04x} perms {} dist {:?}",
                        a.name, tr.gain_db, cs.class_name(), law.class(cs.class).map(|c| c.gain_bounds_db), pb.map(|p| p.gain_base_db), g.scale_gain_of(&cs), cs.flags, g.permutations_of(&cs).len(), law.distances_for(&cs, pb));
                    println!("    voice_gain at scale 0/0.5/1 (dB): {:.1} {:.1} {:.1}", law.voice_gain_db(&cs, pb, g.scale_gain_of(&cs), 0, None, 0.0), law.voice_gain_db(&cs, pb, g.scale_gain_of(&cs), 0, None, 0.5), law.voice_gain_db(&cs, pb, g.scale_gain_of(&cs), 0, None, 1.0));
                }
            }
        }
        // B: sound scenery placements.
        let h = native.tag_meta(cache, scnr, 0x1C8, 8).unwrap();
        let (cnt, ptr) = (rd_u32(&h, 0), rd_u32(&h, 4));
        let raw = native.tag_ptr_bytes(cache, ptr, cnt * 0xA0).unwrap_or_default();
        for e in 0..3usize {
            println!("ssce raw words (first 0xA0 bytes; elem stride unknown), elem-guess start {}:", e * 0x60);
        }
        for row in 0..(raw.len() / 16).min(60) {
            let b = &raw[row * 16..row * 16 + 16];
            let w: Vec<u32> = (0..4).map(|k| rd_u32(b, k * 4)).collect();
            let f: Vec<f32> = w.iter().map(|v| f32::from_bits(*v)).collect();
            println!("  +{:03x}: {:08x} {:08x} {:08x} {:08x}   {:.2} {:.2} {:.2} {:.2}", row * 16, w[0], w[1], w[2], w[3], f[0], f[1], f[2], f[3]);
        }
        // C: the Spartan's footstep material effects.
        let mut r = Ride::with_vehicle(rig.native.clone(), cache, "\\warthog\\warthog").expect("ride");
        let biped = r.pm.biped_tag;
        println!("biped {}", native.tag_name(cache, biped).unwrap_or_default());
        let m = native.tag_meta(cache, biped, 0xC0, 16).unwrap();
        let foot = rd_u32(&m, 0x0C) & 0xFFFF;
        println!("foot tag {}", native.tag_name(cache, foot).unwrap_or_default());
        let outer = tag_block(native, cache, foot, 0x00, 36, 64);
        for (i, e) in outer.chunks_exact(36).enumerate() {
            let rows = block_at(native, cache, e, 0x0C, 44, 512);
            let names: Vec<String> = rows.chunks_exact(44).take(4).map(|row| {
                let t = rd_u32(row, 0x0C); let t = if t != u32::MAX && t != 0 { Some(t & 0xFFFF) } else { None };
                format!("{}->{}", sid_name(native, cache, rd_u32(row, 0x20)), t.map(|t| native.tag_name(cache, t).unwrap_or_default()).unwrap_or("-".into()))
            }).collect();
            println!("  effect[{i}]: {} sound rows; {:?}", rows.len() / 44, names);
        }
        // D: the biped graph's locomotion events.
        let jmad = crate::reach_actor::resolve_jmad(native, cache, biped).unwrap();
        let graph = crate::reach_actor::AnimGraph::read(native, cache, jmad).unwrap();
        let refs = rs::graph_sound_refs(native, cache, jmad);
        println!("graph sound refs: {}", refs.len());
        let mut shown = 0;
        for (i, a) in graph.anims.iter().enumerate() {
            let n = a.name.to_lowercase();
            if !(n.contains("rifle:move_front") || n.contains("rifle:sprint") || n.contains("rifle:run") || n.contains("jump") || n.contains("land") || n.contains("crouch:move_front") || n.contains("rifle:walk")) { continue }
            let fe = rs::animation_foot_events(native, cache, jmad, i);
            let se = rs::animation_sound_events(native, cache, jmad, i);
            if fe.is_empty() && se.is_empty() { continue }
            shown += 1;
            if shown > 40 { break }
            println!("  anim {i} {} frames {:?}: foot {:?}; sounds {:?}", a.name, rs::animation_frame_count(native, cache, jmad, i), fe.iter().map(|e| (e.kind, e.frame)).collect::<Vec<_>>(),
                se.iter().map(|e| (refs.get(e.sound as usize).map(|r| native.tag_name(cache, r.0).unwrap_or_default()), e.frame, sid_name(native, cache, e.marker_sid))).collect::<Vec<_>>());
        }
        let _ = (&mut r, &mut rig);
    }
}
