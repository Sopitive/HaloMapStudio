//! #reach-sound  The app's audio engine: the thing that actually makes a
//! noise.
//!
//! Three modules meet here:
//!
//! * `hms_sound` owns the FMOD banks, the Vorbis decoder and the output
//!   device. It knows nothing about Halo tags.
//! * `reach_sound` owns the tag side — `snd!` / `ugh!` / `sncl` / `snmm` /
//!   `lsnd` / the scenario's acoustics palette — and Reach's own gain,
//!   distance and pan laws.
//! * this module joins them to the Theater clock and the camera -- and, through
//!   [`SoundEngine::begin_frame`] and the voice API, to test mode (`play_sound`), which has no
//!   clock to be a pure function of and keeps voices alive across frames.
//!
//! # The rules this is built around
//!
//! **HMS must start and run on a machine with no working audio device, and
//! must never block startup on one.** So nothing here touches hardware until
//! a sound is actually wanted, the device open cannot fail upward
//! (`Output::open_or_silent`), and a failure is logged once and leaves a
//! silent output that accepts sounds and drops them. There is no
//! environment variable and no cargo feature: audio is either working or
//! quietly absent, exactly as if the volume were down.
//!
//! **Nothing expensive happens on the render thread.** Reading the Vorbis
//! setup tables out of `haloreach.dll` is an ~80 MB scan and decoding a
//! sample is milliseconds, so both happen on one worker thread that owns the
//! banks. The render thread only ever posts a request and looks in a map of
//! what has come back. A sound whose sample is not decoded yet is **skipped**,
//! not waited for; the second time that weapon fires it is already there.
//!
//! **A Theater frame is a pure function of the tick.** The one exception
//! audio cannot avoid is that a sound, once started, keeps playing — so
//! scrubbing, seeking, pausing and changing speed all stop every voice rather
//! than leaving a stale gunshot ringing. Event edges are detected between the
//! previous tick and this one and are suppressed whenever the clock did not
//! advance smoothly forward.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use hms_native::{CacheHandle, NativeDll};
use hms_sound::{Output, Pcm, VoiceId};

use crate::film_play::{ActorDriver, Playback, TICK_HZ};
use crate::reach_sound::{self as rs, CacheSound, Distances, GainLaw, Gestalt, Playback as PlaybackParams};

/// Which bank a sound lives in. Reach has `sfx` for the world and one bank per
/// language for dialogue; `snd!`'s `fmod bank suffix` string id picks between
/// them, and every sound this module plays is a world sound.
pub const WORLD_BANK: &str = "sfx";

/// How long after a `action_weapon_fire` event a shot may still be started.
/// Two ticks at 60 Hz: enough that a frame drop does not swallow the shot,
/// short enough that a scrub cannot replay a burst.
const FIRE_WINDOW_TICKS: u32 = 2;

/// The largest forward clock step that still counts as smooth playback. Above
/// this the clock has been scrubbed or seeked and no event edge fires.
const SMOOTH_STEP_TICKS: f32 = TICK_HZ * 0.5;

/// Decoded samples held at once. A Reach one-shot is a second or two of
/// 48 kHz stereo, so a few hundred is tens of megabytes; this is a bound on
/// that, not a performance tuning knob.
const MAX_CACHED_SAMPLES: usize = 8192;

/// ...and the same bound in bytes (decoded `f32`): the loops a vehicle engine plays are long.
const MAX_CACHED_BYTES: usize = 768 * 1024 * 1024;

// ---------------------------------------------------------------------------------------------
// The worker: owns the banks, the setup table and the decode cache
// ---------------------------------------------------------------------------------------------

enum Job {
    /// Open the banks and read the Vorbis setup tables out of the module.
    Open { haloreach_dir: PathBuf },
    /// Decode one sample of one bank. `urgent`: a voice is waiting for it now, so it goes ahead of
    /// the warm-up queue (a Warthog's 96 engine samples must not hold up a footstep).
    Decode { bank: &'static str, sample: usize, urgent: bool },
    Stop,
}

enum Done {
    /// `Ok` carries the bank's sample count; `Err` the reason audio is off.
    Opened(Result<usize, String>),
    Decoded { bank: &'static str, sample: usize, pcm: Option<Arc<Pcm>>, ms: f32 },
}

/// Everything the worker needs. Lives entirely on the worker thread.
struct Worker {
    banks: HashMap<&'static str, hms_sound::SoundBank>,
    setups: Option<hms_sound::SetupTable>,
}

impl Worker {
    fn run(rx: Receiver<Job>, tx: Sender<Done>) {
        let mut w = Worker { banks: HashMap::new(), setups: None };
        let mut pending: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
        loop {
            if pending.is_empty() {
                match rx.recv() {
                    Ok(j) => pending.push_back(j),
                    Err(_) => break,
                }
            }
            while let Ok(j) = rx.try_recv() {
                pending.push_back(j);
            }
            // Urgent decodes first, otherwise in the order asked.
            let at = pending.iter().position(|j| matches!(j, Job::Decode { urgent: true, .. })).unwrap_or(0);
            let Some(job) = pending.remove(at) else { continue };
            match job {
                Job::Stop => break,
                Job::Open { haloreach_dir } => {
                    let r = w.open(&haloreach_dir);
                    if tx.send(Done::Opened(r)).is_err() {
                        break;
                    }
                }
                Job::Decode { bank, sample, .. } => {
                    let t0 = std::time::Instant::now();
                    let pcm = w.decode(bank, sample);
                    let ms = t0.elapsed().as_secs_f32() * 1000.0;
                    if tx.send(Done::Decoded { bank, sample, pcm, ms }).is_err() {
                        break;
                    }
                }
            }
        }
    }

    fn open(&mut self, haloreach: &Path) -> Result<usize, String> {
        let dir = hms_sound::SoundBank::bank_dir_for_reach(haloreach)
            .ok_or_else(|| format!("no fmod/pc folder above {}", haloreach.display()))?;
        let bank = hms_sound::SoundBank::open(&dir, WORLD_BANK)
            .map_err(|e| format!("{WORLD_BANK}.fsb: {e:#}"))?;
        let n = bank.bank.samples.len();
        // The setup headers are in the game module two directories above the
        // banks. HMS ships no codebook data of its own.
        let module = dir
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("haloreach.dll"))
            .ok_or_else(|| "cannot locate haloreach.dll above the banks".to_string())?;
        let setups = hms_sound::SetupTable::from_module(&module)
            .map_err(|e| format!("{}: {e:#}", module.display()))?;
        self.banks.insert(WORLD_BANK, bank);
        self.setups = Some(setups);
        Ok(n)
    }

    fn decode(&self, bank: &'static str, sample: usize) -> Option<Arc<Pcm>> {
        let b = self.banks.get(bank)?;
        let s = self.setups.as_ref()?;
        match b.decode_sample_playback(sample, s) {
            Ok(p) => Some(Arc::new(p)),
            Err(e) => {
                log::warn!("hms-sound: sample {sample} of {bank}: {e:#}");
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// What a sound tag resolves to, once
// ---------------------------------------------------------------------------------------------

/// A `snd!` resolved down to what is needed to play it: its samples, its gain
/// and its distance curve.
#[derive(Clone, Debug)]
pub struct Clip {
    pub snd: u32,
    pub class: u8,
    /// Bank sample indices, one per permutation, in order.
    pub samples: Vec<usize>,
    /// Each permutation's own authored dB trim.
    pub perm_gain_db: Vec<i8>,
    /// The engine's summed authored gain, linear, with no distance applied.
    pub gain: f32,
    /// The merged class-then-playback distance parameters.
    pub dist: Distances,
    /// Does this sound attenuate with distance at all?
    pub spatial: bool,
    /// `snd!` flag "play permutations in order" (a looping bed steps through its takes).
    pub in_order: bool,
}

impl Clip {
    /// Resolve one `snd!`. `None` when it names no permutation that is in the
    /// world bank.
    pub fn resolve(
        native: &NativeDll,
        cache: CacheHandle,
        gestalt: &Gestalt,
        law: &GainLaw,
        bank: &hms_sound::SoundBank,
        snd_tag: u32,
    ) -> Option<Clip> {
        let snd = CacheSound::read(native, cache, snd_tag)?;
        let pb: Option<&PlaybackParams> = gestalt.playback_of(&snd);
        let mut samples = Vec::new();
        let mut perm_gain_db = Vec::new();
        for p in gestalt.permutations_of(&snd) {
            if let Some(i) = bank.index_of_subsound(p.fmod_subsound_id) {
                samples.push(i);
                perm_gain_db.push(p.gain_db);
            }
        }
        if samples.is_empty() {
            return None;
        }
        let dist = law.distances_for(&snd, pb);
        // t = 0.5: the MIDPOINT of the engine's `_power` draw between the
        // class's gain bounds. 65 of the 71 classes author lower == upper so
        // this is exact for them; where they differ, a Theater frame has to
        // be a pure function of the tick, and the midpoint is the draw's
        // expected value rather than an arbitrary end of it.
        let gain = law.voice_gain(&snd, pb, gestalt.scale_gain_of(&snd), 0, None, 0.5);
        Some(Clip {
            snd: snd_tag,
            class: snd.class,
            samples,
            perm_gain_db,
            gain,
            dist,
            spatial: dist.usable(),
            in_order: snd.flags & rs::SND_FLAG_PERMUTATIONS_IN_ORDER != 0,
        })
    }

    /// Which permutation to use for an event. Deterministic in the tick and
    /// the source, so a re-scrub of the same frame picks the same take.
    pub fn permutation_for(&self, tick: u32, source: u32) -> usize {
        if self.samples.len() <= 1 {
            return 0;
        }
        // A cheap integer hash. Deterministic is the requirement; uniform
        // enough that a burst does not repeat one take is the nicety.
        let mut h = tick
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(source.wrapping_mul(0x85EB_CA6B));
        h ^= h >> 15;
        h as usize % self.samples.len()
    }

    /// The linear gain and L/R pan for this clip heard from a listener.
    ///
    /// `dir` is the source position in LISTENER space (x forward, y left,
    /// z up) and `distance` its length in world units. Both the curve and the
    /// pan are the engine's own; see `reach_sound`.
    pub fn gains_at(&self, dir: [f32; 3], distance: f32, permutation: usize) -> [f32; 2] {
        let trim = rs::db_to_linear(
            self.perm_gain_db.get(permutation).copied().unwrap_or(0) as f32,
        );
        let d = if self.spatial { self.dist.gain_at(distance) } else { 1.0 };
        let g = self.gain * trim * d;
        let pan = rs::stereo_pan(dir);
        [g * pan[0], g * pan[1]]
    }

    /// The gain for a non-positional sound (an ambience bed, music, UI).
    pub fn gain_unspatialized(&self, permutation: usize) -> [f32; 2] {
        let trim = rs::db_to_linear(
            self.perm_gain_db.get(permutation).copied().unwrap_or(0) as f32,
        );
        let g = self.gain * trim;
        [g, g]
    }
}

// ---------------------------------------------------------------------------------------------
// A LOOPING `snd!` — what a vehicle engine's `lsnd` track plays
// ---------------------------------------------------------------------------------------------

/// `sound_definition_flags` bit 13 as a cache holds it (`cache_file_sound +0x00 & 0x2000`): the
/// looping-sound manager plays EVERY pitch range whose `bend bounds` contain the current pitch,
/// at once, crossfading the edges (`looping_sound_manager.cpp`, debug `0x1809BA830`).
pub const SND_FLAG_ALL_PITCH_RANGES: u16 = 0x2000;

/// One pitch range of a looping `snd!`.
#[derive(Clone, Debug)]
pub struct LoopRange {
    /// Bank sample per permutation, in order.
    pub samples: Vec<usize>,
    pub perm_gain_db: Vec<i8>,
    /// `natural pitch`, cents: the pitch the samples sound at when played as recorded.
    pub natural_pitch: f32,
    /// `bend bounds`: the pitches this range is allowed to play at.
    pub bend: (f32, f32),
    /// `max gain pitch bounds`: the pitches it plays at full volume.
    pub full: (f32, f32),
}

/// A `snd!` resolved for a LOOP: its pitch ranges, and the authored scale response.
#[derive(Clone, Debug)]
pub struct LoopSnd {
    pub snd: u32,
    pub cs: CacheSound,
    pub ranges: Vec<LoopRange>,
    /// `scale.gain modifier` (lower, upper), dB.
    pub scale_gain_db: (f32, f32),
    /// `scale.pitch modifier` (lower, upper), cents: the span the loop's 0..1 scale sweeps.
    pub scale_pitch: (f32, f32),
    pub playback: Option<PlaybackParams>,
    pub dist: Distances,
    pub spatial: bool,
}

impl LoopSnd {
    pub fn all_ranges_at_once(&self) -> bool {
        self.cs.flags & SND_FLAG_ALL_PITCH_RANGES != 0
    }

    /// The requested pitch in cents (`sub_1806538B0`): `lerp(lower, upper, scale) + offset`, where
    /// the offset is the playback's `random pitch bounds` (the engine draws one per source at
    /// start; its centre is used so the result is a pure function of the scale).
    pub fn pitch_at(&self, scale: f32) -> f32 {
        let offset = self
            .playback
            .map(|p| (p.random_pitch_cents.0 as f32 + p.random_pitch_cents.1 as f32) * 0.5)
            .unwrap_or(0.0);
        self.scale_pitch.0 + (self.scale_pitch.1 - self.scale_pitch.0) * scale + offset
    }

    /// The `(range, crossfade dB)` pairs playing at `pitch`.
    ///
    /// **All-ranges sounds** (`SND_FLAG_ALL_PITCH_RANGES`) play every range whose bend bounds
    /// contain the pitch. Inside `max gain pitch bounds` that is 0 dB; outside it the gain is
    /// `interp_db(mode 2, -64, 0, t)` with `t` running from 0 at the bend bound to 1 at the full
    /// bound (`sub_1804D2700`), so a range fades in over the span between the two.
    ///
    /// **Other sounds** play one range: the one the engine's `sub_1809B0080` picks, which keeps
    /// `current` while the pitch is still inside its bend bounds and otherwise takes the range
    /// whose bounds contain the pitch, else the nearest.
    pub fn ranges_at(&self, pitch: f32, current: Option<usize>) -> Vec<(usize, f32)> {
        let usable = |r: &LoopRange| !r.samples.is_empty();
        let inside = |r: &LoopRange| pitch >= r.bend.0 && pitch <= r.bend.1;
        if self.all_ranges_at_once() {
            return self
                .ranges
                .iter()
                .enumerate()
                .filter(|(_, r)| usable(r) && inside(r))
                .map(|(i, r)| {
                    let db = if pitch >= r.full.0 && pitch <= r.full.1 {
                        0.0
                    } else {
                        let (edge, full, rising) = if pitch < r.full.0 {
                            (r.bend.0, r.full.0, true)
                        } else {
                            (r.bend.1, r.full.1, false)
                        };
                        let span = full - edge;
                        // `sub_1804D2700`: a degenerate span is 1 on the audible side, else 0.
                        let t = if span.abs() < 1.0e-4 {
                            if (edge > pitch) != rising { 1.0 } else { 0.0 }
                        } else {
                            ((pitch - edge) / span).clamp(0.0, 1.0)
                        };
                        rs::interp_db(2, rs::MIN_GAIN_DB, 0.0, t)
                    };
                    (i, db)
                })
                .collect();
        }
        // `sub_1809B0080` (decompiled): keep the CURRENT range while the pitch is inside its
        // MAX-GAIN bounds (`+0x08`); else the first range whose max-gain bounds hold the pitch;
        // else the range NEAREST by its BEND bounds (`+0x04`): the distance is `lo - pitch` while
        // the pitch is at or below the bend's upper bound (so a pitch INSIDE a bend wins with a
        // negative distance) and `pitch - hi` above it.
        let in_full = |r: &LoopRange| pitch >= r.full.0 && pitch <= r.full.1;
        if let Some(c) = current.filter(|c| self.ranges.get(*c).map_or(false, |r| usable(r) && in_full(r))) {
            return vec![(c, 0.0)];
        }
        let hit = self.ranges.iter().position(|r| usable(r) && in_full(r));
        let pick = hit.or_else(|| {
            let d = |r: &LoopRange| if pitch <= r.bend.1 { r.bend.0 - pitch } else { pitch - r.bend.1 };
            self.ranges
                .iter()
                .enumerate()
                .filter(|(_, r)| usable(r))
                .fold(None::<(usize, f32)>, |best, (i, r)| match best {
                    Some((_, bd)) if bd <= d(r) => best,
                    _ => Some((i, d(r))),
                })
                .map(|(i, _)| i)
        });
        pick.map(|i| vec![(i, 0.0)]).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------

/// Where audio is up to, for the UI and the diagnostics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoundStatus {
    pub device: String,
    /// False when there is no device, or opening one failed.
    pub running: bool,
    pub voices: usize,
    pub bank_samples: usize,
    pub decoded: usize,
    pub ambience_playing: usize,
    /// Which acoustics row the listener's cluster selected.
    pub acoustics_row: usize,
    pub shots_started: u64,
    pub melees_started: u64,
    pub reloads_started: u64,
    pub clangs_started: u64,
    pub melee_hits_started: u64,
    pub footsteps_started: u64,
    pub shots_skipped_not_decoded: u64,
    /// Times a voice could not start because a take was still decoding.
    pub decode_stalls: u64,
    /// Why a `action_weapon_fire` event did NOT become a sound, by reason.
    /// Counters rather than a log line, so the answer is always in hand and
    /// nothing has to be switched on to see it. Indices are
    /// [`FIRE_BAIL_NAMES`].
    pub fire_bail: [u64; 7],
    /// Why a sound that SHOULD have played did not, by [`Refusal`] (index = `Refusal as usize`).
    pub drops: [u64; REFUSAL_N],
    /// Decodes a voice had to wait for, and the milliseconds spent waiting.
    pub urgent_waits: u64,
    pub urgent_wait_ms: f32,
    /// Decodes finished, and the milliseconds they took (total and the slowest one).
    pub decodes: u64,
    pub decode_ms_total: f32,
    pub decode_ms_max: f32,
    /// Samples asked for and not yet decoded (the "sound loading" count).
    pub loading: usize,
    /// Why audio is off, when it is. Shown once, not every frame.
    pub why: String,
}

/// What `play_positioned` did, so the caller can count it by reason.
enum PlayResult {
    Started,
    OutOfRange,
    BelowFloor,
    NotDecoded,
    NoSample,
}

/// Why a sound that should have played did not. Always counted (`SoundStatus::drops`), never
/// gated, so the answer to "why was that silent" is on the status line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Refusal {
    #[default]
    None = 0,
    /// The tag chain authors no sound for this event (silence by the tag).
    NoClip = 1,
    /// The sample was still decoding and did not arrive within the wait.
    NotDecoded = 2,
    /// Outside the sound's own distance window.
    OutOfRange = 3,
    /// Below the engine's -64 dB floor.
    BelowFloor = 4,
    /// No permutation / the mixer refused the voice (its cap).
    NoVoice = 5,
}
pub const REFUSAL_N: usize = 6;
pub const REFUSAL_NAMES: [&str; REFUSAL_N] = ["none", "no clip", "not decoded", "out of range", "below floor", "no voice"];

/// How long a one-shot waits for its sample to decode before it is dropped, milliseconds. A
/// footstep or a shot that arrives 30 ms late is heard; one that is skipped is not.
pub const URGENT_WAIT_MS: u64 = 250;

/// Which sound of a weapon. Each resolves through a different tag chain, so
/// they are cached separately.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WeaponSound {
    /// The shot: barrel -> firing effect -> `snd!` of class `weapon_fire`.
    Fire,
    /// The melee swing: the first-person animation graph's `weapon_melee`.
    Melee,
    /// The clang two melee weapons make: `weap +0x2B8 clang effect`.
    Clang,
    /// The reload: the animation graph's `weapon_reload`.
    Reload,
    /// The dry-fire click: the barrel firing effect's `empty effect`.
    Empty,
}

/// What each slot of [`SoundStatus::fire_bail`] counts.
///
/// Plain counters rather than a log line, so the answer to "why is that shot
/// silent?" is always in hand and nothing has to be switched on to see it.
pub const FIRE_BAIL_NAMES: [&str; 7] = [
    "no track for the firing biped",
    "neither the weapon lane nor the attachment graph names a weapon",
    "the weapon index is outside the cache's weap list",
    "the weapon resolves to no firing sound",
    "the firing biped has no recorded position at that tick",
    "outside the sound's own distance window",
    "below the engine's -64 dB floor",
];

/// An ambience bed that is currently playing.
struct AmbienceVoice {
    acoustics: usize,
    snd: u32,
    voice: Option<VoiceId>,
    clip: Clip,
    permutation: usize,
}

/// The one audio engine. Owned by `App`, and by the scripted host.
pub struct SoundEngine {
    /// `None` until the first sound is wanted.
    out: Option<Output>,
    worker: Option<(Sender<Job>, Receiver<Done>, std::thread::JoinHandle<()>)>,
    /// Set once the worker reports the banks open.
    bank_samples: usize,
    opening: bool,
    /// Decoded samples, and the requests already in flight.
    pcm: HashMap<usize, Arc<Pcm>>,
    /// Least-recently-used stamp per decoded sample, for eviction.
    pcm_used: HashMap<usize, u64>,
    use_clock: u64,
    inflight: HashSet<usize>,
    /// Samples whose decode FAILED: never asked for again (a failing sample must not cost a wait
    /// per frame).
    failed: HashSet<usize>,
    /// Seamless variants of single-sample loops (see `hms_sound::decode::loop_crossfade`).
    loops: HashMap<usize, Arc<Pcm>>,
    /// The reason the last refused sound was refused (see [`SoundEngine::last_drop`]).
    last_drop: Refusal,
    /// Reporting category the next voices started belong to (`hms_sound::mix::CATEGORIES`).
    pub cat_hint: u8,
    /// Which map the tag-side tables belong to.
    map_key: Option<(u64, PathBuf)>,
    gestalt: Option<Gestalt>,
    law: GainLaw,
    /// A sample-index view of the world bank, for `Clip::resolve`. The bank
    /// itself lives on the worker thread, so this is a copy of just the
    /// subsound-id to sample-index map.
    subsound: HashMap<u32, usize>,
    /// `(weap tag, what)` -> the clip, resolved once per map. `what` is one
    /// of [`WeaponSound`]; a firing sound and a melee swing are two different
    /// entries for the same weapon.
    fire_of: HashMap<(u32, WeaponSound), Option<Clip>>,
    /// Track indices of every `weap` entity in the film, found once. The
    /// weapon a biped is firing is the one PARENTED to it, which is how the
    /// engine itself carries the relationship — a biped's own `weapons` lane
    /// is almost always empty (measured: 10 of 112 firing bipeds on a real
    /// film, 13 entries in all).
    weapon_tracks: Vec<usize>,
    weapon_tracks_built: bool,
    /// `(biped tag, lowercased surface name)` -> its footstep clip.
    footstep_of: HashMap<(u32, String), Option<Clip>>,
    /// `bipd`/`scen`/... tag -> its `obje +0xD0 melee sound`, resolved once.
    /// On a Spartan this is an `scmb`, not a `snd!`.
    melee_victim_of: HashMap<u32, Option<Clip>>,
    /// The acoustics palette, resolved once per map: (palette row, track clip).
    acoustics: Vec<(usize, Clip)>,
    /// Every BSP cluster's box and the acoustics row it selects, read once per
    /// map. This is how the exterior/interior pair a map authors gets chosen
    /// by WHERE THE LISTENER IS rather than by taking row 0.
    clusters: Vec<rs::AcousticCluster>,
    /// The row currently playing, so a move between areas switches the bed.
    current_acoustics: Option<usize>,
    playing_ambience: Vec<AmbienceVoice>,
    acoustics_read: bool,
    /// One-shot voices, so a seek can stop them.
    oneshots: Vec<VoiceId>,
    /// User volume, 0..=1. Persisted.
    volume: f32,
    pub muted: bool,
    /// The Theater clock the last tick ran at, for edge detection.
    last_tick: Option<f32>,
    status: SoundStatus,
    logged_failure: bool,
}

impl std::fmt::Debug for SoundEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoundEngine")
            .field("status", &self.status)
            .field("volume", &self.volume)
            .field("muted", &self.muted)
            .finish()
    }
}

impl Default for SoundEngine {
    fn default() -> Self {
        SoundEngine {
            out: None,
            worker: None,
            bank_samples: 0,
            opening: false,
            pcm: HashMap::new(),
            pcm_used: HashMap::new(),
            use_clock: 0,
            inflight: HashSet::new(),
            failed: HashSet::new(),
            loops: HashMap::new(),
            last_drop: Refusal::None,
            cat_hint: 0,
            map_key: None,
            gestalt: None,
            law: GainLaw::default(),
            subsound: HashMap::new(),
            fire_of: HashMap::new(),
            weapon_tracks: Vec::new(),
            weapon_tracks_built: false,
            footstep_of: HashMap::new(),
            melee_victim_of: HashMap::new(),
            acoustics: Vec::new(),
            clusters: Vec::new(),
            current_acoustics: None,
            playing_ambience: Vec::new(),
            acoustics_read: false,
            oneshots: Vec::new(),
            volume: load_volume_setting(),
            muted: false,
            last_tick: None,
            status: SoundStatus::default(),
            logged_failure: false,
        }
    }
}

impl SoundEngine {
    pub fn status(&self) -> &SoundStatus {
        &self.status
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Set the user volume, 0..=1, and persist it.
    pub fn set_volume(&mut self, v: f32) {
        // NaN is the only value that needs special handling: it has no position on the scale, so
        // the safe reading is silence. `clamp` already does the right thing with the infinities --
        // `+inf` clamps to 1.0 and `-inf` to 0.0 -- and the earlier `is_finite` test lumped them
        // in with NaN, which turned "infinitely loud" into SILENCE. That is the opposite of what
        // the value says, and the one case here a user could actually reach (a settings file
        // holding `inf`) would have muted the app with no way to see why.
        let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        if (v - self.volume).abs() > 1.0e-4 {
            self.volume = v;
            save_volume_setting(v);
        }
        self.push_master();
    }

    pub fn set_muted(&mut self, m: bool) {
        self.muted = m;
        self.push_master();
    }

    fn master(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            // The user slider is a plain linear multiplier on top of the
            // engine's own authored gain. It is NOT a second attenuation law:
            // Reach's dB chain has already set the relative loudness.
            self.volume
        }
    }

    fn push_master(&self) {
        if let Some(o) = &self.out {
            o.set_master(self.master());
        }
    }

    /// Stop everything and forget what was playing. Called on a seek, a
    /// pause, a film change and a map change.
    pub fn stop_all(&mut self) {
        if let Some(o) = &self.out {
            o.stop_all();
            o.reap();
        }
        self.playing_ambience.clear();
        self.current_acoustics = None;
        self.oneshots.clear();
        self.status.voices = 0;
        self.status.ambience_playing = 0;
    }

    /// Bring up the device and the worker. Idempotent, non-blocking, and
    /// never fails upward: a machine with no sound card ends up with a silent
    /// output that accepts sounds and drops them.
    fn ensure_open(&mut self, haloreach_dir: &Path) {
        if self.out.is_none() {
            let o = Output::open_or_silent(false);
            match o.state() {
                hms_sound::State::Running(name, rate, ch) => {
                    log::info!("hms-sound: output {name} at {rate} Hz, {ch} channels");
                    self.status.device = format!("{name} · {rate} Hz · {ch} ch");
                    self.status.running = true;
                }
                // No device, or the open failed. Say so ONCE and carry on
                // silently: a machine with no sound card must be
                // indistinguishable from one with the volume down.
                other => {
                    let why = match other {
                        hms_sound::State::Off(w) => w,
                        _ => "no output device".to_string(),
                    };
                    self.status.running = false;
                    self.status.why = why.clone();
                    if !self.logged_failure {
                        log::warn!("hms-sound: no audio output ({why}); films will be silent");
                        self.logged_failure = true;
                    }
                }
            }
            o.set_master(self.master());
            self.out = Some(o);
        }
        if self.worker.is_none() {
            let (jtx, jrx) = std::sync::mpsc::channel();
            let (dtx, drx) = std::sync::mpsc::channel();
            let h = std::thread::Builder::new()
                .name("hms-sound".into())
                .spawn(move || Worker::run(jrx, dtx));
            match h {
                Ok(h) => {
                    let _ = jtx.send(Job::Open { haloreach_dir: haloreach_dir.to_path_buf() });
                    self.opening = true;
                    self.worker = Some((jtx, drx, h));
                }
                Err(e) => {
                    if !self.logged_failure {
                        log::warn!("hms-sound: cannot start the audio worker: {e}");
                        self.status.why = format!("{e}");
                        self.logged_failure = true;
                    }
                }
            }
        }
    }

    /// Drain the worker's replies. Cheap; called every frame.
    fn pump_worker(&mut self) {
        let Some((_, rx, _)) = &self.worker else { return };
        while let Ok(d) = rx.try_recv() {
            match d {
                Done::Opened(Ok(n)) => {
                    self.bank_samples = n;
                    self.opening = false;
                    self.status.bank_samples = n;
                    log::info!("hms-sound: {WORLD_BANK}.fsb open, {n} samples");
                }
                Done::Opened(Err(why)) => {
                    self.opening = false;
                    if !self.logged_failure {
                        log::warn!("hms-sound: {why}");
                        self.logged_failure = true;
                    }
                    self.status.why = why;
                }
                Done::Decoded { bank: _, sample, pcm, ms } => {
                    self.inflight.remove(&sample);
                    if pcm.is_none() {
                        self.failed.insert(sample);
                    }
                    self.status.decodes += 1;
                    self.status.decode_ms_total += ms;
                    self.status.decode_ms_max = self.status.decode_ms_max.max(ms);
                    if let Some(p) = pcm {
                        self.use_clock += 1;
                        self.pcm_used.insert(sample, self.use_clock);
                        self.pcm.insert(sample, p);
                        // Bounded by count and by decoded bytes (a vehicle's engine loops run to
                        // several seconds of 48 kHz stereo each). The LEAST RECENTLY USED sample
                        // goes: an arbitrary one used to, which could be the weapon sample that
                        // had just arrived, so a shot was re-decoded and skipped again and again.
                        while (self.pcm.len() > MAX_CACHED_SAMPLES || self.pcm.values().map(|p| p.samples.len() * 4).sum::<usize>() > MAX_CACHED_BYTES) && self.pcm.len() > 1 {
                            let victim = self.pcm.keys().filter(|k| **k != sample).min_by_key(|k| self.pcm_used.get(*k).copied().unwrap_or(0)).copied();
                            match victim {
                                Some(k) => {
                                    self.pcm.remove(&k);
                                    self.pcm_used.remove(&k);
                                }
                                None => break,
                            }
                        }
                    }
                }
            }
        }
        self.status.decoded = self.pcm.len();
        self.status.loading = self.inflight.len();
    }

    /// Ask for a sample, and say whether it is ready now.
    fn want(&mut self, sample: usize) -> Option<Arc<Pcm>> {
        if let Some(p) = self.pcm.get(&sample) {
            self.use_clock += 1;
            self.pcm_used.insert(sample, self.use_clock);
            return Some(p.clone());
        }
        if self.bank_samples == 0 || self.inflight.contains(&sample) || self.failed.contains(&sample) {
            return None;
        }
        if let Some((tx, _, _)) = &self.worker {
            if tx.send(Job::Decode { bank: WORLD_BANK, sample, urgent: false }).is_ok() {
                self.inflight.insert(sample);
            }
        }
        None
    }

    /// [`SoundEngine::want`] for a voice that is starting NOW: the decode jumps the warm-up queue
    /// and the caller waits up to [`URGENT_WAIT_MS`] for it.
    fn want_now(&mut self, sample: usize) -> Option<Arc<Pcm>> {
        if let Some(p) = self.want(sample) {
            return Some(p);
        }
        if self.bank_samples == 0 || self.failed.contains(&sample) {
            return None;
        }
        // Re-ask as urgent (the warm-up request may be queued behind a hundred others).
        if let Some((tx, _, _)) = &self.worker {
            let _ = tx.send(Job::Decode { bank: WORLD_BANK, sample, urgent: true });
            self.inflight.insert(sample);
        }
        let t0 = std::time::Instant::now();
        let limit = std::time::Duration::from_millis(URGENT_WAIT_MS);
        while t0.elapsed() < limit {
            self.pump_worker();
            if let Some(p) = self.pcm.get(&sample) {
                let p = p.clone();
                self.status.urgent_waits += 1;
                self.status.urgent_wait_ms += t0.elapsed().as_secs_f32() * 1000.0;
                return Some(p);
            }
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
        self.status.urgent_waits += 1;
        self.status.urgent_wait_ms += t0.elapsed().as_secs_f32() * 1000.0;
        None
    }

    /// Tag a started voice with a reporting category.
    fn tag_voice(&self, id: VoiceId, cat: u8) {
        if cat != 0 {
            if let Some(o) = &self.out {
                o.handle().set_cat(id, cat);
            }
        }
    }

    /// The seamless variant of a single-sample loop, built once per sample.
    pub fn loop_variant(&mut self, sample: usize) -> Option<Arc<Pcm>> {
        if let Some(p) = self.loops.get(&sample) {
            return Some(p.clone());
        }
        let pcm = self.want_now(sample)?;
        let v = Arc::new(hms_sound::decode::loop_crossfade(&pcm));
        self.loops.insert(sample, v.clone());
        Some(v)
    }

    /// Why the last sound this engine refused was refused ([`Refusal::None`] if it never refused).
    pub fn last_drop(&self) -> Refusal {
        self.last_drop
    }

    pub(crate) fn refuse(&mut self, why: Refusal) {
        self.last_drop = why;
        self.status.drops[why as usize] += 1;
    }

    /// Read the map's tag side once. Returns false when the map has no sound
    /// gestalt (a non-Reach cache, or one with the audio stripped).
    fn ensure_map(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        map_path: &Path,
    ) -> bool {
        let key = (cache.0, map_path.to_path_buf());
        if self.map_key.as_ref() == Some(&key) {
            return self.gestalt.is_some();
        }
        self.stop_all();
        self.pcm.clear();
        self.pcm_used.clear();
        self.failed.clear();
        self.loops.clear();
        self.inflight.clear();
        self.fire_of.clear();
        self.weapon_tracks.clear();
        self.weapon_tracks_built = false;
        self.footstep_of.clear();
        self.melee_victim_of.clear();
        self.acoustics.clear();
        self.acoustics_read = false;
        self.clusters.clear();
        self.current_acoustics = None;
        self.subsound.clear();
        self.map_key = Some(key);
        self.gestalt = Gestalt::read(native, cache);
        self.law = GainLaw::read(native, cache);
        self.gestalt.is_some()
    }

    /// The subsound-id to sample-index map, built once from the worker's bank.
    ///
    /// The bank lives on the worker thread, so the main thread cannot ask it
    /// directly; instead `Clip::resolve` is given a tiny shim that only knows
    /// how to answer `index_of_subsound`.
    fn ensure_subsound_map(&mut self, dir: &Path) {
        if !self.subsound.is_empty() || self.bank_samples == 0 {
            return;
        }
        // The sidecar is a flat array of 280-byte records; reading it is a
        // 4.5 MB file read, done once per session.
        let Some(bd) = hms_sound::SoundBank::bank_dir_for_reach(dir) else { return };
        let p = bd.join(format!("{WORLD_BANK}.fsb.info"));
        let Ok(info) = hms_sound::BankInfo::open(&p) else { return };
        self.subsound = info
            .records
            .iter()
            .map(|r| (r.subsound_id, r.index))
            .collect();
    }

    fn resolve_clip(
        &self,
        native: &NativeDll,
        cache: CacheHandle,
        snd_tag: u32,
    ) -> Option<Clip> {
        let g = self.gestalt.as_ref()?;
        let snd = CacheSound::read(native, cache, snd_tag)?;
        let pb = g.playback_of(&snd);
        let mut samples = Vec::new();
        let mut perm_gain_db = Vec::new();
        for p in g.permutations_of(&snd) {
            if let Some(i) = self.subsound.get(&p.fmod_subsound_id).copied() {
                samples.push(i);
                perm_gain_db.push(p.gain_db);
            }
        }
        if samples.is_empty() {
            return None;
        }
        let dist = self.law.distances_for(&snd, pb);
        let gain = self
            .law
            .voice_gain(&snd, pb, g.scale_gain_of(&snd), 0, None, 0.5);
        Some(Clip {
            snd: snd_tag,
            class: snd.class,
            samples,
            perm_gain_db,
            gain,
            dist,
            spatial: dist.usable(),
            in_order: snd.flags & rs::SND_FLAG_PERMUTATIONS_IN_ORDER != 0,
        })
    }

    /// Drive one Theater frame.
    ///
    /// `listener` is `(position, forward, left, up)` in world space — Halo's
    /// basis, which is what `reach_sound::to_listener_space` expects.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        native: &NativeDll,
        cache: CacheHandle,
        map_path: &Path,
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
    ) {
        self.ensure_open(map_path);
        self.pump_worker();
        if !self.ensure_map(native, cache, map_path) {
            return;
        }
        self.ensure_subsound_map(map_path);
        if self.bank_samples == 0 || self.subsound.is_empty() {
            return; // still opening; nothing to do and nothing to wait for
        }

        let tick = pb.tick;
        let prev = self.last_tick;
        self.last_tick = Some(tick);
        // A seek, a rewind, a pause or a speed change: stop everything rather
        // than leave a stale shot ringing or replay a burst.
        let smooth = match prev {
            Some(p) => pb.playing && tick >= p && tick - p <= SMOOTH_STEP_TICKS,
            None => false,
        };
        if !smooth {
            self.stop_all();
            if !pb.playing {
                return;
            }
        }

        self.drive_ambience(native, cache, listener.0);
        if let Some(p) = prev.filter(|_| smooth) {
            self.drive_events(pb, driver, native, cache, p, tick, listener);
        }

        if let Some(o) = &self.out {
            o.reap();
            self.status.voices = self.playing_ambience.len() + self.oneshots.len();
        }
        self.status.ambience_playing = self.playing_ambience.len();
    }

    /// Start the map's ambience bed, and keep it running.
    ///
    /// Reach's map ambience is the scenario's **acoustics palette**: each
    /// entry's `ambience.background sound` is an `lsnd` whose tracks' `loop`
    /// references are the `snd!` that plays continuously while the listener is
    /// in that area. Entry 0 is the one played here; selecting per area needs
    /// the BSP cluster's `acoustics` index, which this module does not have
    /// yet, and picking the nearest-sounding entry by name would be a
    /// heuristic rather than the engine's rule.
    fn drive_ambience(&mut self, native: &NativeDll, cache: CacheHandle, listener: [f32; 3]) {
        if !self.acoustics_read {
            // Once per map: a map with NO ambience used to repeat this whole scan (and grow
            // `clusters`) on every frame.
            self.acoustics_read = true;
            let Some(scnr) = native.find_scenario(cache) else { return };
            // A BSP carries its own acoustics palette AND the cluster table
            // that indexes it, so prefer the BSP's; the scenario's copy is the
            // fallback for a map that leaves the BSP's empty.
            let bsps = native.enumerate_sbsps(cache, scnr, 16);
            let mut pal: Vec<rs::Acoustics> = Vec::new();
            for sb in &bsps {
                let p = rs::bsp_acoustics_palette(native, cache, *sb);
                if p.len() > pal.len() {
                    pal = p;
                }
                self.clusters.extend(rs::bsp_acoustic_clusters(native, cache, *sb));
            }
            if pal.is_empty() {
                pal = rs::acoustics_palette(native, cache, scnr);
            }
            for a in &pal {
                let Some(lsnd) = a.background_sound else { continue };
                let Some(l) = rs::LoopingSound::read(native, cache, lsnd) else { continue };
                for tr in &l.tracks {
                    let Some(snd) = tr.loop_snd else { continue };
                    if let Some(mut c) = self.resolve_clip(native, cache, snd) {
                        // The track's own `gain:dB` is part of the authored
                        // chain for a looping sound.
                        c.gain *= rs::db_to_linear(tr.gain_db);
                        self.acoustics.push((a.index, c));
                    }
                }
            }
            if self.acoustics.is_empty() {
                return;
            }
            log::info!(
                "hms-sound: {} acoustics rows, {} ambience tracks, {} clusters",
                pal.len(),
                self.acoustics.len(),
                self.clusters.len()
            );
        }
        // WHICH ROW. The engine binds the bed to the listener's cluster, which
        // is the whole reason a map authors an `ext`/`int` pair; playing row 0
        // unconditionally gives the outdoor bed to someone standing inside a
        // structure. Without a cluster table there is nothing to select on, so
        // fall back to the lowest row the palette resolved rather than
        // guessing from a name.
        let want_row = rs::acoustics_at(&self.clusters, listener)
            .or_else(|| self.acoustics.iter().map(|(i, _)| *i).min());
        let Some(want_row) = want_row else { return };
        if self.current_acoustics != Some(want_row) {
            // Switch areas: drop the old bed before starting the new one, so
            // two beds never play at once.
            for v in self.playing_ambience.drain(..) {
                if let (Some(o), Some(id)) = (&self.out, v.voice) {
                    o.stop(id);
                }
            }
            self.status.acoustics_row = want_row;
            self.current_acoustics = Some(want_row);
        }
        let want: Vec<(usize, Clip)> = self
            .acoustics
            .iter()
            .filter(|(i, _)| *i == want_row)
            .cloned()
            .collect();
        for (idx, clip) in want {
            if self
                .playing_ambience
                .iter()
                .any(|v| v.acoustics == idx && v.snd == clip.snd)
            {
                continue;
            }
            // ALL of the bed's takes, as a playlist: a bed is authored as several permutations
            // (forge_halo's is six, 3.2 to 6.3 s each) and the engine moves on to another when
            // one ends. Looping a single take is a 3-6 second loop repeating for ever -- a record
            // skipping. Every take is requested; the bed starts once they are all decoded.
            let perm = 0;
            let wanted: Vec<Option<Arc<Pcm>>> = clip.samples.iter().take(16).map(|&s| self.want(s)).collect();
            if wanted.is_empty() || wanted.iter().any(|p| p.is_none()) {
                continue;
            }
            let pcms: Vec<Arc<Pcm>> = wanted.into_iter().flatten().collect();
            let g = clip.gain_unspatialized(perm);
            let voice = self.out.as_ref().and_then(|o| o.play_playlist(pcms, clip.in_order, g, 1.0));
            if let Some(v) = voice {
                self.tag_voice(v, 1);
            }
            self.playing_ambience.push(AmbienceVoice {
                acoustics: idx,
                snd: clip.snd,
                voice,
                clip,
                permutation: perm,
            });
        }
    }

    /// The `weap` tag of the weapon parented to `net` at `tick`.
    ///
    /// `BaseAttachment::parent` is the parent object's net id, so this is a
    /// scan of the film's weapon tracks for one whose parent is this biped.
    /// The candidate list is built once per film from the object table's own
    /// resolved identities — nothing here re-derives what a `weap` is.
    fn attached_weapon(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        net: u32,
        tick: u32,
    ) -> Option<u32> {
        let table = driver.objects.as_ref()?;
        if !self.weapon_tracks_built {
            self.weapon_tracks_built = true;
            for (i, tr) in pb.index.tracks.iter().enumerate() {
                if tr.attachments.is_empty() {
                    continue;
                }
                if table.ident(tr.net_id).map(|o| o.group == "weap").unwrap_or(false) {
                    self.weapon_tracks.push(i);
                }
            }
            log::debug!(
                "hms-sound: {} weapon tracks carry an attachment",
                self.weapon_tracks.len()
            );
        }
        for &i in &self.weapon_tracks {
            let tr = pb.index.tracks.get(i)?;
            if tr.attached_at(tick).map(|a| a.parent) == Some(net) {
                if let Some(o) = table.ident(tr.net_id) {
                    if o.obj_tag != 0 && o.obj_tag != 0xFFFF_FFFF {
                        return Some(o.obj_tag);
                    }
                }
            }
        }
        None
    }

    /// Play every event between two ticks that has a sound.
    ///
    /// ONE walk of the decoded event channel, which is in tick order, so a
    /// `partition_point` gives exactly the events in `(from, to]`. Dispatch is
    /// by event kind, and every kind goes through the same positioning and
    /// gain path — a melee swing cannot end up louder than a gunshot because
    /// of a second copy of the law.
    #[allow(clippy::too_many_arguments)]
    fn drive_events(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        native: &NativeDll,
        cache: CacheHandle,
        from: f32,
        to: f32,
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
    ) {
        use crate::film_action as fa;
        let a = from.max(0.0) as u32;
        let b = to.max(0.0) as u32;
        if b < a {
            return;
        }
        let lo = pb.index.events.partition_point(|e| e.tick <= a);
        let hi = pb.index.events.partition_point(|e| e.tick <= b);
        let Some(slice) = pb.index.events.get(lo..hi) else { return };
        // Copy out what is needed so the borrow of `pb.index` ends before the
        // `&mut self` calls below.
        let work: Vec<(u8, u32, Option<u32>, Option<u32>)> =
            slice.iter().map(|e| (e.kind, e.tick, e.subject, e.other)).collect();
        for (kind, tick, subject, other) in work {
            if b.saturating_sub(tick) > FIRE_WINDOW_TICKS {
                continue;
            }
            let what = match kind {
                fa::EV_WEAPON_FIRE => WeaponSound::Fire,
                fa::EV_MELEE_INITIATE => WeaponSound::Melee,
                fa::EV_RELOAD => WeaponSound::Reload,
                fa::EV_MELEE_CLANG => {
                    self.clang(pb, driver, native, cache, tick, listener);
                    continue;
                }
                // A melee that CONNECTS: the thing that was hit makes its own
                // sound, and `obje +0xD0` is the engine's own field for it --
                // "the sound that is made when I am meleed". The victim is
                // reference 1 of the event.
                fa::EV_MELEE_DAMAGE => {
                    if let Some(v) = other {
                        self.melee_victim(pb, driver, native, cache, v, tick, listener);
                    }
                    continue;
                }
                _ => continue,
            };
            let Some(net) = subject else { continue };
            self.fire_one(pb, driver, native, cache, net, tick, what, listener);
        }
    }

    /// The sound the thing that was MELEED makes (`obje +0xD0 melee sound`).
    ///
    /// On a Spartan and an Elite this is
    /// `sound\materials\soft\organic_flesh\soft_organic_flesh`, and it is an
    /// **`scmb`** (a sound combiner) rather than a `snd!` — a resolver that
    /// only knows `snd!` and `lsnd` finds nothing and melee goes silent, which
    /// is why `reach_sound::resolve_to_snd` is the single place that knows
    /// which classes can stand in for a sound. The Forge prop set authors one
    /// per prop, so meleeing a pallet sounds like a pallet.
    fn melee_victim(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        native: &NativeDll,
        cache: CacheHandle,
        victim: u32,
        tick: u32,
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
    ) {
        // What IS the victim? The object table's resolved identity, or the
        // biped definition the driver resolved for a body.
        let obj = driver
            .objects
            .as_ref()
            .and_then(|t| t.ident(victim))
            .map(|o| o.obj_tag)
            // No second route: the object table IS the resolved-identity
            // table, and re-resolving a `def_wire` here would be a second
            // copy of a rule that already has a condition (the nine-group
            // reference table and its salts).
            ;
        let Some(obj) = obj else { return };
        if !self.melee_victim_of.contains_key(&obj) {
            let clip = rs::melee_victim_sound(native, cache, obj)
                .and_then(|t| self.resolve_clip(native, cache, t));
            self.melee_victim_of.insert(obj, clip);
        }
        let Some(Some(clip)) = self.melee_victim_of.get(&obj).cloned() else { return };
        let Some(pos) = driver
            .track_of
            .get(&victim)
            .and_then(|i| pb.index.tracks.get(*i))
            .and_then(|t| t.sample(tick))
        else {
            return;
        };
        if let PlayResult::Started = self.play_positioned(&clip, pos, listener, tick, victim) {
            self.status.melee_hits_started += 1;
        }
    }

    /// `biped_melee_clang` — two melee weapons meeting in the air.
    ///
    /// The event carries **no object reference** (measured: 1,446 of them in
    /// the corpus, 0 refs, 79 parameter bits that are not decoded), so there
    /// is no position for it. It is therefore played UNSPATIALIZED, which is
    /// stated rather than hidden: a clang is a notable, close-quarters sound
    /// and placing it at a guessed position would be worse than placing it
    /// nowhere. The sound is `weap +0x2B8 clang effect`, taken from the
    /// camera subject's weapon when there is one.
    fn clang(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        native: &NativeDll,
        cache: CacheHandle,
        tick: u32,
        _listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
    ) {
        let subject = crate::film_play::camera_subject(pb, tick)
            .map(|(net, _slot)| net)
            .or_else(|| driver.melee_of.keys().next().copied());
        let weap = subject.and_then(|net| self.weapon_of(pb, driver, net, tick));
        let Some(weap) = weap else { return };
        let Some(clip) = self.weapon_clip(native, cache, weap, WeaponSound::Clang) else { return };
        let perm = clip.permutation_for(tick, 0);
        let Some(sample) = clip.samples.get(perm).copied() else { return };
        let Some(pcm) = self.want(sample) else { return };
        let g = clip.gain_unspatialized(perm);
        if let Some(v) = self.out.as_ref().and_then(|o| o.play(pcm, g, 1.0, false)) {
            self.oneshots.push(v);
            self.status.clangs_started += 1;
        }
    }

    /// The `weap` tag a biped holds at `tick`, by either route.
    fn weapon_of(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        net: u32,
        tick: u32,
    ) -> Option<u32> {
        let ti = driver.track_of.get(&net).copied()?;
        let lane = pb
            .index
            .tracks
            .get(ti)
            .and_then(|t| t.weapon_at(tick))
            .and_then(|wi| driver.weap_tags.get(wi as usize).copied());
        lane.or_else(|| self.attached_weapon(pb, driver, net, tick))
    }

    #[allow(clippy::too_many_arguments)]
    fn fire_one(
        &mut self,
        pb: &Playback,
        driver: &ActorDriver,
        native: &NativeDll,
        cache: CacheHandle,
        net: u32,
        tick: u32,
        what: WeaponSound,
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
    ) {
        // Which weapon? The firing biped's track names a `weap` group index
        // at this tick; `ActorDriver::weap_tags` turns that into a cache tag.
        let Some(&ti) = driver.track_of.get(&net) else {
            self.status.fire_bail[0] += 1;
            return;
        };
        let Some(track) = pb.index.tracks.get(ti) else {
            self.status.fire_bail[0] += 1;
            return;
        };
        // WHICH WEAPON. Two routes, in the order of how much they are worth.
        //
        // The biped's own `weapons` lane is what the renderer uses to put a
        // gun in the hand, but it is sparse: measured on a real 1,022-shot
        // film, **10 of 112 firing bipeds have any entry at all** (13 entries
        // between them) and none has a spawn weapon, so it answers almost no
        // shot. The weapon's own ATTACHMENT to the biped is the relationship
        // the engine actually carries, and it is the decode this project has
        // already proven 53 of 53.
        let weap = track
            .weapon_at(tick)
            .and_then(|wi| driver.weap_tags.get(wi as usize).copied())
            .or_else(|| self.attached_weapon(pb, driver, net, tick));
        let Some(weap) = weap else {
            self.status.fire_bail[1] += 1;
            return;
        };

        let Some(clip) = self.weapon_clip(native, cache, weap, what) else {
            self.status.fire_bail[3] += 1;
            return;
        };

        let Some(pos) = track.sample(tick) else {
            self.status.fire_bail[4] += 1;
            return;
        };
        match self.play_positioned(&clip, pos, listener, tick, net) {
            PlayResult::Started => match what {
                WeaponSound::Melee => self.status.melees_started += 1,
                WeaponSound::Reload => self.status.reloads_started += 1,
                _ => self.status.shots_started += 1,
            },
            PlayResult::OutOfRange => self.status.fire_bail[5] += 1,
            PlayResult::BelowFloor => self.status.fire_bail[6] += 1,
            PlayResult::NotDecoded => self.status.shots_skipped_not_decoded += 1,
            PlayResult::NoSample => {}
        }
    }

    // -----------------------------------------------------------------------------------------
    // TEST MODE's side of the engine (`play_sound`): no Theater clock, voices that outlive a frame
    // -----------------------------------------------------------------------------------------

    /// Bring the device, the worker and this map's tag tables up, and say whether sounds can be
    /// started this frame. Non-blocking: `false` while the banks are still opening, and the
    /// caller simply tries again next frame.
    pub fn begin_frame(&mut self, native: &NativeDll, cache: CacheHandle, map_path: &Path) -> bool {
        self.ensure_open(map_path);
        self.pump_worker();
        if !self.ensure_map(native, cache, map_path) {
            return false;
        }
        self.ensure_subsound_map(map_path);
        if let Some(o) = &self.out {
            o.reap();
        }
        self.bank_samples != 0 && !self.subsound.is_empty()
    }

    /// The map's ambience bed for a listener at `pos`, started and kept running (Theater drives
    /// the same code from its clock; test mode drives it from its own frame).
    pub fn ambience_frame(&mut self, native: &NativeDll, cache: CacheHandle, pos: [f32; 3]) {
        self.drive_ambience(native, cache, pos);
        self.status.ambience_playing = self.playing_ambience.len();
    }

    /// **The offline seam.** An engine whose output is the mixer with no device behind it: sounds
    /// are accepted, mixed by the REAL [`hms_sound::mix::Mixer`], and handed back by
    /// [`SoundEngine::render`] as samples. Nothing here can reach a speaker, which is what lets a
    /// test hold a Warthog at idle, mid-rev and redline and measure what comes out.
    #[cfg(test)]
    pub fn offline(rate: u32) -> SoundEngine {
        let mut e = SoundEngine::default();
        e.out = Some(Output::silent(rate, "offline render"));
        e.volume = 1.0;
        e.push_master();
        e
    }

    /// Mix `frames` stereo frames of whatever is playing, interleaved. Offline engines only.
    #[cfg(test)]
    pub fn render(&mut self, frames: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; frames * 2];
        if let Some(o) = self.out.as_mut() {
            o.pump(&mut buf, 2);
        }
        buf
    }

    /// Start a decoded sample directly (gain, pitch ratio) on the offline mixer. Tests only.
    #[cfg(test)]
    pub fn play_pcm_for_test(&mut self, pcm: Arc<Pcm>, gain: [f32; 2], pitch: f64) -> bool {
        self.out.as_ref().and_then(|o| o.play(pcm, gain, pitch, false)).is_some()
    }

    /// The weapon's fire clip as the engine resolves it. Tests only.
    #[cfg(test)]
    pub fn weapon_fire_clip_for_test(&mut self, native: &NativeDll, cache: CacheHandle, weap: u32) -> Option<Clip> {
        self.weapon_clip(native, cache, weap, WeaponSound::Fire)
    }

    /// Has every permutation of the weapon's fire sound decoded? Tests only.
    #[cfg(test)]
    pub fn sample_ready_for_weapon(&mut self, native: &NativeDll, cache: CacheHandle, weap: u32) -> bool {
        self.pump_worker();
        match self.weapon_clip(native, cache, weap, WeaponSound::Fire) {
            Some(c) => c.samples.iter().take(16).all(|s| self.pcm.contains_key(s)),
            None => false,
        }
    }

    /// The subsound id of a bank sample index (reverse of the join map). Tests only.
    #[cfg(test)]
    pub fn sample_subsound(&self, sample: usize) -> Option<u32> {
        self.subsound.iter().find(|(_, v)| **v == sample).map(|(k, _)| *k)
    }

    /// The ambience beds currently chosen: `(permutation count, bank sample per permutation)`.
    #[cfg(test)]
    pub fn ambience_beds(&self) -> Vec<(usize, Vec<usize>)> {
        self.playing_ambience.iter().map(|v| (v.clip.samples.len(), v.clip.samples.clone())).collect()
    }

    /// Voices the mixer is holding right now. Offline engines only.
    #[cfg(test)]
    pub fn mixer_voices(&self) -> usize {
        self.out.as_ref().and_then(|o| o.voice_count()).unwrap_or(0)
    }

    /// Resolve a `snd!` for a loop (pitch ranges and the authored scale response).
    pub fn resolve_loop(&self, native: &NativeDll, cache: CacheHandle, snd_tag: u32) -> Option<LoopSnd> {
        let g = self.gestalt.as_ref()?;
        let cs = CacheSound::read(native, cache, snd_tag)?;
        let pb = g.playback_of(&cs).copied();
        let first = cs.first_pitch_range as usize;
        let mut ranges = Vec::new();
        for pr in g.pitch_ranges.iter().skip(first).take(cs.pitch_range_count as usize) {
            let (a, b) = (pr.first_permutation, pr.first_permutation + pr.permutation_count);
            let mut samples = Vec::new();
            let mut perm_gain_db = Vec::new();
            for p in g.permutations.get(a..b).unwrap_or(&[]) {
                if let Some(i) = self.subsound.get(&p.fmod_subsound_id).copied() {
                    samples.push(i);
                    perm_gain_db.push(p.gain_db);
                }
            }
            let pp = g.params_of(pr).copied().unwrap_or_default();
            ranges.push(LoopRange {
                samples,
                perm_gain_db,
                natural_pitch: pp.natural_pitch as f32,
                bend: (pp.bend_bounds.0 as f32, pp.bend_bounds.1 as f32),
                full: (pp.max_gain_bounds.0 as f32, pp.max_gain_bounds.1 as f32),
            });
        }
        if ranges.iter().all(|r| r.samples.is_empty()) {
            return None;
        }
        let dist = self.law.distances_for(&cs, pb.as_ref());
        let sp = g.scale_pitch_of(&cs);
        Some(LoopSnd {
            snd: snd_tag,
            ranges,
            scale_gain_db: g.scale_gain_of(&cs),
            scale_pitch: (sp.0 as f32, sp.1 as f32),
            playback: pb,
            spatial: dist.usable(),
            dist,
            cs,
        })
    }

    /// The engine's gain for one voice of a loop, dB, BEFORE distance and pan: the class and
    /// scale gain bounds interpolated by the loop's SCALE, the playback `gain base`, the
    /// permutation trim, the track's own gain and crossfades (`source_db`), clamped to
    /// `[-64, 0]`, then the `snmm` master gain. See [`GainLaw::voice_gain_db_extra`].
    pub fn loop_gain_db(&self, ls: &LoopSnd, scale: f32, perm_gain_db: i8, source_db: f32) -> f32 {
        self.law.voice_gain_db_extra(&ls.cs, ls.playback.as_ref(), ls.scale_gain_db, perm_gain_db, None, scale, source_db)
    }

    /// A decoded sample, or `None` after asking for it (non-blocking).
    pub fn sample(&mut self, index: usize) -> Option<Arc<Pcm>> {
        self.pump_worker();
        self.want(index)
    }

    /// Start a looping voice that will be re-aimed and re-pitched every frame, over one or
    /// several takes (permutations): with several, it moves on to another each time one ends.
    /// `None` while any take is still decoding (counted in `SoundStatus::decode_stalls`), or when
    /// audio is off.
    pub fn voice_start(&mut self, samples: &[usize], in_order: bool, gain: [f32; 2], pitch: f64) -> Option<(VoiceId, u32)> {
        // A loop that is starting NOW (a driver sitting down, a vehicle coming into earshot) waits
        // for its takes, once, for as long as a one-shot would: a loop that never starts is an
        // engine that is silent (the audit's Warthog at redline).
        // A single-sample loop plays its SEAMLESS variant (a decoded loop ticks at its wrap).
        let single = samples.len() == 1;
        let got: Vec<Option<Arc<Pcm>>> = samples.iter().take(16).map(|&s| if single { self.loop_variant(s) } else { self.want_now(s) }).collect();
        if got.is_empty() || got.iter().any(|p| p.is_none()) {
            self.status.decode_stalls += 1;
            return None;
        }
        let pcms: Vec<Arc<Pcm>> = got.into_iter().flatten().collect();
        let rate = pcms[0].sample_rate;
        let id = self.out.as_ref()?.play_playlist(pcms, in_order, gain, pitch)?;
        self.tag_voice(id, self.cat_hint);
        Some((id, rate))
    }

    pub fn voice_set(&self, id: VoiceId, source_rate: u32, gain: [f32; 2], pitch: f64) {
        if let Some(o) = &self.out {
            o.set_gain(id, gain);
            o.set_pitch(id, source_rate, pitch);
        }
    }

    pub fn voice_stop(&self, id: VoiceId) {
        if let Some(o) = &self.out {
            o.stop(id);
        }
    }

    /// Resolve one of a weapon's sounds and play it at a world position.
    ///
    /// The same chain Theater uses for another player's gun (`weap` -> `snd!`), so the player's
    /// own shot, swing and reload cannot sound different from the ones in a film. Returns false
    /// when the weapon authors no such sound, the sample is still decoding, or the listener is
    /// outside the sound's own distance window.
    #[allow(clippy::too_many_arguments)]
    pub fn play_weapon_sound(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        weap: u32,
        what: WeaponSound,
        pos: [f32; 3],
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
        seed: u32,
    ) -> bool {
        let Some(clip) = self.weapon_clip(native, cache, weap, what) else {
            self.refuse(Refusal::NoClip);
            return false;
        };
        let started = matches!(self.play_positioned(&clip, pos, listener, seed, weap), PlayResult::Started);
        if started {
            match what {
                WeaponSound::Melee => self.status.melees_started += 1,
                WeaponSound::Reload => self.status.reloads_started += 1,
                WeaponSound::Clang | WeaponSound::Empty => {}
                WeaponSound::Fire => self.status.shots_started += 1,
            }
        }
        started
    }

    /// The output callback's own numbers, when there is a device.
    pub fn out_diag(&self) -> Option<hms_sound::OutDiag> {
        self.out.as_ref().map(|o| o.diag())
    }

    /// Ask the worker to decode a weapon's sounds ahead of the first shot, so the FIRST trigger
    /// pull is heard instead of skipped (a sample that is not decoded yet is skipped, never
    /// waited for).
    pub fn warm_weapon(&mut self, native: &NativeDll, cache: CacheHandle, weap: u32) {
        for what in [WeaponSound::Fire, WeaponSound::Melee, WeaponSound::Reload, WeaponSound::Empty] {
            if let Some(clip) = self.weapon_clip(native, cache, weap, what) {
                // EVERY permutation: a shot picks one by hash, and one that is not decoded yet is
                // skipped, so warming only the first would leave most shots silent.
                for &s in clip.samples.iter().take(16) {
                    let _ = self.want(s);
                }
            }
        }
    }

    /// Resolve a `snd!` to a one-shot [`Clip`] (cached by the caller).
    pub fn resolve_one_shot(&self, native: &NativeDll, cache: CacheHandle, snd_tag: u32) -> Option<Clip> {
        self.resolve_clip(native, cache, snd_tag)
    }

    /// Play a resolved one-shot at a world position (the single positioned path).
    pub fn play_clip_at(
        &mut self,
        clip: &Clip,
        pos: [f32; 3],
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
        seed: u32,
    ) -> bool {
        matches!(self.play_positioned(clip, pos, listener, seed, 0), PlayResult::Started)
    }

    /// Play a resolved one-shot with no position: the clip's authored gain and nothing else.
    /// The engine routes a sound on the PLAYER'S OWN unit or vehicle through its first-person
    /// mix (`sub_1401F5BC0`) rather than a 3D position, so the driver's gear shift is heard
    /// at the same level wherever the camera is.
    pub fn play_clip_flat(&mut self, clip: &Clip, seed: u32) -> bool {
        let perm = clip.permutation_for(seed, clip.snd);
        let Some(&sample) = clip.samples.get(perm) else {
            self.refuse(Refusal::NoClip);
            return false;
        };
        let g = clip.gain_unspatialized(perm);
        // Below the engine's -64 dB floor it would not be started at all.
        if g[0].max(g[1]) <= rs::db_to_linear(rs::MIN_GAIN_DB) {
            self.refuse(Refusal::BelowFloor);
            return false;
        }
        let Some(pcm) = self.want_now(sample) else {
            self.refuse(Refusal::NotDecoded);
            return false;
        };
        match self.out.as_ref().and_then(|o| o.play(pcm, g, 1.0, false)) {
            Some(v) => {
                self.tag_voice(v, self.cat_hint);
                self.oneshots.push(v);
                if self.oneshots.len() > 128 {
                    self.oneshots.remove(0);
                }
                true
            }
            None => {
                self.refuse(Refusal::NoVoice);
                false
            }
        }
    }

    /// Play one FOOTSTEP: the sound `biped` makes on `material`, at `pos`.
    ///
    /// **This is the whole footstep feature's entry point and it is not keyed
    /// on `Playback`** — it takes a tag, a surface name, a world position and
    /// a listener basis, so a walking player in test mode can call it with the
    /// material its own capsule sweep already knows, and a Theater body can
    /// call it the day the collision-material table is readable.
    ///
    /// `material` is a global material NAME as `matg +0x48C materials[].name`
    /// spells it (`hard`, `brittle`, `energy`, `hard_metal_solid`, ...). An
    /// unknown name, or a surface the biped authors no row for, is **silence**
    /// — a Spartan's own `default_material` row names no sound, so there is no
    /// authored fallback to substitute and inventing one would be worse than
    /// nothing.
    ///
    /// Returns true when a voice started.
    pub fn play_footstep(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        biped: u32,
        material: &str,
        pos: [f32; 3],
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
        seed: u32,
    ) -> bool {
        let key = (biped, material.to_ascii_lowercase());
        if !self.footstep_of.contains_key(&key) {
            let clip = rs::footstep_sound(native, cache, biped, material)
                .and_then(|t| self.resolve_clip(native, cache, t));
            self.footstep_of.insert(key.clone(), clip);
        }
        let Some(Some(clip)) = self.footstep_of.get(&key).cloned() else { return false };
        let started = matches!(
            self.play_positioned(&clip, pos, listener, seed, biped),
            PlayResult::Started
        );
        if started {
            self.status.footsteps_started += 1;
        }
        started
    }

    /// The foot-landing frames of one animation, for a caller that drives the
    /// clip itself (test mode). `(which foot, frame)` in frame order.
    pub fn foot_events(
        native: &NativeDll,
        cache: CacheHandle,
        jmad: u32,
        anim_index: usize,
    ) -> Vec<rs::FootEvent> {
        rs::animation_foot_events(native, cache, jmad, anim_index)
    }

    /// Resolve and cache one of a weapon's sounds.
    fn weapon_clip(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        weap: u32,
        what: WeaponSound,
    ) -> Option<Clip> {
        let key = (weap, what);
        if !self.fire_of.contains_key(&key) {
            // Species 0 (Spartan) for the animation-authored sounds: the film
            // does carry the species, but a weapon with one first-person row
            // uses it for both and the Elite row differs only in the take.
            let snd = match what {
                WeaponSound::Fire => rs::weapon_fire_sound(native, cache, weap).first().map(|f| f.snd),
                WeaponSound::Melee => rs::weapon_melee_sound(native, cache, weap, 0),
                // Every candidate, first that resolves wins -- most clang
                // effects lead with a sound the map carries no samples for.
                WeaponSound::Clang => rs::weapon_clang_sounds(native, cache, weap)
                    .into_iter()
                    .find(|t| self.resolve_clip(native, cache, *t).is_some()),
                WeaponSound::Reload => rs::weapon_reload_sound(native, cache, weap, 0),
                WeaponSound::Empty => rs::weapon_empty_sound(native, cache, weap),
            };
            let clip = snd.and_then(|t| self.resolve_clip(native, cache, t));
            self.fire_of.insert(key, clip);
        }
        self.fire_of.get(&key).cloned().flatten()
    }

    /// Start one positioned one-shot. The single place a world position
    /// becomes a voice, so the distance law, the pan and the floor test cannot
    /// differ between a gunshot and a melee swing.
    fn play_positioned(
        &mut self,
        clip: &Clip,
        pos: [f32; 3],
        listener: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]),
        tick: u32,
        source: u32,
    ) -> PlayResult {
        let (lpos, fwd, left, up) = listener;
        let dir = rs::to_listener_space(pos, lpos, fwd, left, up);
        let distance = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        // The engine refuses a sound outside its own distance window.
        if clip.spatial && (distance > clip.dist.maximum || distance < clip.dist.dont_play) {
            self.refuse(Refusal::OutOfRange);
            return PlayResult::OutOfRange;
        }
        let perm = clip.permutation_for(tick, source);
        let g = clip.gains_at(dir, distance, perm);
        // Below the engine's own floor it would not be started at all.
        if g[0].max(g[1]) <= rs::db_to_linear(rs::MIN_GAIN_DB) {
            self.refuse(Refusal::BelowFloor);
            return PlayResult::BelowFloor;
        }
        let Some(sample) = clip.samples.get(perm).copied() else {
            self.refuse(Refusal::NoClip);
            return PlayResult::NoSample;
        };
        let Some(pcm) = self.want_now(sample) else {
            // Still not decoded after the urgent wait: counted, never silent about it.
            self.refuse(Refusal::NotDecoded);
            return PlayResult::NotDecoded;
        };
        if let Some(v) = self.out.as_ref().and_then(|o| o.play(pcm, g, 1.0, false)) {
            self.tag_voice(v, self.cat_hint);
            self.oneshots.push(v);
            if self.oneshots.len() > 128 {
                self.oneshots.remove(0);
            }
            PlayResult::Started
        } else {
            self.refuse(Refusal::NoVoice);
            PlayResult::NoSample
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The one volume control
// ---------------------------------------------------------------------------------------------

/// The master volume row. **One implementation**, called from both the
/// Settings window and the Theater side panel, so the two can never disagree
/// about what the slider means or drift apart.
///
/// It is a plain control: a mute box, a 0..100% slider and a line saying what
/// the device is doing. No hover-then-click, nothing that needs the viewport
/// focused, and it reads correctly before any sound has ever played — a fresh
/// engine has opened no device and says so.
pub fn volume_ui(engine: &mut SoundEngine, ui: &mut eframe::egui::Ui) {
    use eframe::egui;
    ui.horizontal(|ui| {
        let mut muted = engine.muted;
        if ui
            .checkbox(&mut muted, "Mute")
            .on_hover_text("Silence film audio without forgetting the volume.")
            .changed()
        {
            engine.set_muted(muted);
        }
        let mut v = engine.volume() * 100.0;
        let r = ui
            .add_enabled(
                !engine.muted,
                egui::Slider::new(&mut v, 0.0..=100.0).suffix("%").text("Volume"),
            )
            .on_hover_text(
                "Film audio: the map's ambience bed and weapon fire, at the levels Reach's own \
                 sound tags author. Saved between sessions.",
            );
        if r.changed() {
            engine.set_volume(v / 100.0);
        }
    });
    let st = engine.status();
    let line = if !st.why.is_empty() {
        format!("No audio output — {}", st.why)
    } else if st.running {
        let mut l = st.device.clone();
        if st.bank_samples > 0 {
            l.push_str(&format!(" · {} bank samples", st.bank_samples));
        }
        if st.ambience_playing > 0 {
            l.push_str(&format!(" · {} ambience", st.ambience_playing));
        }
        if st.shots_started > 0 {
            l.push_str(&format!(" · {} shots", st.shots_started));
        }
        l
    } else {
        "Audio starts with the first sound in a film.".to_string()
    };
    ui.add(egui::Label::new(egui::RichText::new(line).small().weak()).wrap());
}

// ---------------------------------------------------------------------------------------------
// The persisted volume
// ---------------------------------------------------------------------------------------------

/// The settings file, beside the first-person FOV and the keybinds.
const VOLUME_SETTING_FILE: &str = "sound_volume.txt";
/// Default volume. Reach's own authored gain has already set the relative
/// loudness of every sound, so this is simply "all of it".
pub const DEFAULT_VOLUME: f32 = 1.0;

pub fn load_volume_setting() -> f32 {
    crate::project::settings_dir()
        .map(|d| load_volume_from_dir(&d))
        .unwrap_or(DEFAULT_VOLUME)
}

/// Dir-injectable so a test does not depend on the user's settings folder.
pub fn load_volume_from_dir(dir: &Path) -> f32 {
    std::fs::read_to_string(dir.join(VOLUME_SETTING_FILE))
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_VOLUME)
}

pub fn save_volume_setting(v: f32) {
    if let Some(d) = crate::project::settings_dir() {
        save_volume_to_dir(&d, v);
    }
}

pub fn save_volume_to_dir(dir: &Path, v: f32) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(VOLUME_SETTING_FILE), format!("{v}"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(gain: f32, dist: Distances, samples: usize) -> Clip {
        Clip {
            snd: 1,
            class: rs::CLASS_WEAPON_FIRE,
            samples: (0..samples).collect(),
            perm_gain_db: vec![0; samples],
            gain,
            dist,
            spatial: dist.usable(),
            in_order: false,
        }
    }

    #[test]
    fn a_clip_applies_the_engines_curve_and_pan_and_nothing_else() {
        // weapon_fire's own class row, measured: (0,0,0, 8,15,25,50, -6).
        let d = Distances {
            minimum: 8.0, sustain_begin: 15.0, sustain_end: 25.0, maximum: 50.0,
            sustain_db: -6.0, ..Default::default()
        };
        let c = clip(0.5, d, 3);
        // Dead ahead at 1 wu: inside `minimum`, so the distance gain is 1.0
        // and the stereo fold is 0.5 either side.
        let g = c.gains_at([1.0, 0.0, 0.0], 1.0, 0);
        assert!((g[0] - 0.25).abs() < 1e-4 && (g[1] - 0.25).abs() < 1e-4, "{g:?}");
        // Hard left must be louder on the left.
        let l = c.gains_at([0.0, 1.0, 0.0], 1.0, 0);
        assert!(l[0] > l[1] * 2.0, "{l:?}");
        // Past `maximum` the curve is zero.
        let far = c.gains_at([1.0, 0.0, 0.0], 60.0, 0);
        assert_eq!(far, [0.0, 0.0]);
        // On the plateau it is the sustain gain, 10^(-6/20) = 0.5012.
        let plat = c.gains_at([1.0, 0.0, 0.0], 20.0, 0);
        assert!((plat[0] - 0.5 * 0.5012 * 0.5).abs() < 1e-3, "{plat:?}");
    }

    #[test]
    fn an_unspatialized_clip_is_the_same_on_both_sides_at_any_distance() {
        let c = clip(0.25, Distances::default(), 1);
        assert!(!c.spatial);
        let g = c.gain_unspatialized(0);
        assert_eq!(g, [0.25, 0.25]);
        // And even through the positional path, a zero curve is unity.
        let p = c.gains_at([1.0, 0.0, 0.0], 500.0, 0);
        assert!((p[0] - 0.125).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn the_permutation_choice_is_deterministic_in_the_tick_and_the_source() {
        let c = clip(1.0, Distances::default(), 5);
        for (tick, net) in [(0u32, 0u32), (100, 7), (4096, 123_456)] {
            let a = c.permutation_for(tick, net);
            let b = c.permutation_for(tick, net);
            assert_eq!(a, b, "the same frame must pick the same take");
            assert!(a < 5);
        }
        // Different ticks should not all collapse to one take.
        let picks: std::collections::BTreeSet<usize> =
            (0..64u32).map(|t| c.permutation_for(t, 3)).collect();
        assert!(picks.len() >= 3, "only {} distinct takes over 64 ticks", picks.len());
        // One permutation is always index 0.
        assert_eq!(clip(1.0, Distances::default(), 1).permutation_for(99, 9), 0);
    }

    #[test]
    fn the_permutation_trim_is_applied_in_db() {
        let mut c = clip(1.0, Distances::default(), 2);
        c.perm_gain_db = vec![0, -6];
        let a = c.gain_unspatialized(0);
        let b = c.gain_unspatialized(1);
        assert!((a[0] - 1.0).abs() < 1e-5);
        assert!((b[0] - 0.5012).abs() < 1e-3, "{b:?}");
    }

    #[test]
    fn the_volume_setting_round_trips_and_rejects_rubbish() {
        let dir = std::env::temp_dir().join(format!("hms-sound-vol-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(load_volume_from_dir(&dir), DEFAULT_VOLUME, "a missing file is the default");
        save_volume_to_dir(&dir, 0.35);
        assert!((load_volume_from_dir(&dir) - 0.35).abs() < 1e-6);
        for bad in ["", "   ", "nonsense", "-1", "2.5", "NaN", "inf"] {
            std::fs::write(dir.join(VOLUME_SETTING_FILE), bad).unwrap();
            assert_eq!(load_volume_from_dir(&dir), DEFAULT_VOLUME, "{bad:?} should not be accepted");
        }
        save_volume_to_dir(&dir, 1.0);
        assert_eq!(load_volume_from_dir(&dir), 1.0);
        save_volume_to_dir(&dir, 0.0);
        assert_eq!(load_volume_from_dir(&dir), 0.0, "silence is a legitimate setting");
        let f = dir.join(VOLUME_SETTING_FILE);
        let _ = std::fs::remove_file(f);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn a_fresh_engine_has_touched_no_hardware() {
        // The whole startup contract: constructing the engine must not open a
        // device, must not read a bank and must not fail.
        let e = SoundEngine::default();
        assert!(e.out.is_none(), "no device may be opened before a sound is wanted");
        assert!(e.worker.is_none(), "no worker thread before a sound is wanted");
        assert_eq!(e.bank_samples, 0);
        assert!(!e.status().running);
        assert_eq!(e.status().voices, 0);
        assert!((e.master() - e.volume()).abs() < 1e-6);
    }

    #[test]
    fn muting_overrides_the_volume_without_losing_it() {
        let mut e = SoundEngine::default();
        e.volume = 0.6;
        assert!((e.master() - 0.6).abs() < 1e-6);
        e.set_muted(true);
        assert_eq!(e.master(), 0.0);
        assert!((e.volume() - 0.6).abs() < 1e-6, "mute must not forget the volume");
        e.set_muted(false);
        assert!((e.master() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn a_hostile_volume_is_clamped_rather_than_stored() {
        let mut e = SoundEngine::default();
        for (v, want) in [(f32::NAN, 0.0f32), (f32::INFINITY, 1.0), (-5.0, 0.0), (99.0, 1.0)] {
            e.volume = 0.5;
            e.set_volume(v);
            assert_eq!(e.volume(), want, "set_volume({v}) stored {}", e.volume());
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The production-path gate
// ---------------------------------------------------------------------------------------------
#[cfg(test)]
mod probes {
    use super::*;
    use crate::film;
    use crate::film_decode;

    /// Open the Reach cache a film was recorded on, the way the app does.
    fn open_cache_for(
        map_stem: &str,
    ) -> Option<(std::sync::Arc<NativeDll>, CacheHandle, PathBuf)> {
        let mut roots: Vec<PathBuf> = Vec::new();
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
        let m = cands.iter().find(|c| {
            c.game == crate::mapcat::Game::Reach
                && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(map_stem))
        })?;
        if let Some(d) = m.path.parent() {
            native.set_shared_cache_dirs(&[d.to_path_buf()]);
        }
        let cache = native.open_cache(&m.path.to_string_lossy()).ok()?;
        Some((native, cache, m.path.clone()))
    }

    /// THE GATE: a real film, played through `SoundEngine::tick` — the same
    /// function `App::film_tick` calls — must start real voices.
    ///
    /// **It runs MUTED.** `set_muted(true)` is session-only (it is not the
    /// persisted volume, so it cannot leave the user's app silent) and it
    /// pins the mixer's master gain to 0.0, so the real device opens, the real
    /// callback is fed and nothing comes out of the speakers. The numbers this
    /// prints are the proof; a human listening is not needed and is not
    /// wanted.
    ///
    /// Nothing here builds its own inputs: the `Playback`, the `ActorDriver`
    /// and the camera pose all come from the production code, and the listener
    /// basis is derived the same way `film_tick` derives it.
    #[test]
    fn probe_a_real_film_starts_real_voices() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock();
        let films = film::find_finished_films();
        if films.is_empty() {
            eprintln!("sound: no films on this machine, skipping");
            return;
        }
        let Some((native, cache, map_path)) = open_cache_for("forge_halo") else {
            eprintln!("sound: Reach not installed, skipping");
            return;
        };
        if hms_sound::SoundBank::bank_dir_for_reach(&map_path).is_none() {
            eprintln!("sound: no fmod/pc beside the map, skipping");
            return;
        }

        // SWEEP THE CORPUS. The share of shots whose weapon the recording
        // names varies enormously between clips and is zero on about half of
        // them (`film_decode::WEAPREF_BITS` records 0%, 4%, 0%, 1%, 1%, 4%,
        // 0%, 3%, 0%, 0%, 5%, 8% over twelve films, measured independently),
        // so a single film proves nothing either way. Each one is scored
        // AFTER the spawn-weapon pass, because a unit only reports its weapon
        // when it CHANGES and the spawn weapon is what covers the rest.
        let names = crate::film_variant::VariantNames::build(&native, cache);
        let weap_tags_all = native.tags_by_group(cache, "weap");
        let mut best: Option<(PathBuf, film_decode::FilmIndex, usize, usize)> = None;
        let mut rows: Vec<String> = Vec::new();
        for p in films.iter().take(24) {
            let Ok(f) = film::open(p) else { continue };
            if f.data.records.is_empty() || f.header.map_variant.is_empty() {
                continue;
            }
            let mut idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.entity_messages == 0 {
                continue;
            }
            let spawn = f.header.gametype.clone().map(|gv| {
                crate::film_variant::apply_spawn_weapons(
                    &mut idx.tracks,
                    crate::film_play::BIPED_TYPE,
                    &gv,
                    &names,
                    &weap_tags_all,
                )
            });
            let net_of: HashMap<u32, usize> =
                idx.tracks.iter().enumerate().map(|(i, t)| (t.net_id, i)).collect();
            let fires: Vec<(u32, Option<u32>)> = idx
                .events
                .iter()
                .filter(|e| e.kind == crate::film_action::EV_WEAPON_FIRE)
                .map(|e| (e.tick, e.subject))
                .collect();
            let named = fires
                .iter()
                .filter(|(tick, subject)| {
                    subject
                        .and_then(|n| net_of.get(&n))
                        .and_then(|i| idx.tracks.get(*i))
                        .and_then(|t| t.weapon_at(*tick))
                        .is_some()
                })
                .count();
            rows.push(format!(
                "    {:<44} {:>6} shots, {:>6} name a weapon ({:>5.1}%)  spawn: {}",
                p.file_stem().unwrap().to_string_lossy(),
                fires.len(),
                named,
                100.0 * named as f64 / fires.len().max(1) as f64,
                match &spawn {
                    Some(a) if a.bodies > 0 => format!("{} bodies, {}", a.bodies, a.tag_name),
                    Some(a) => a.why.to_string(),
                    None => "no game variant".into(),
                }
            ));
            if best.as_ref().map_or(true, |b| named > b.2) {
                best = Some((p.clone(), idx, named, fires.len()));
            }
        }
        eprintln!("sound: weapon-fire coverage per film, AFTER the spawn-weapon pass:");
        for r in &rows {
            eprintln!("{r}");
        }
        let total_named: usize = 0;
        let _ = total_named;
        let Some((chosen_path, idx, named, total_fire)) = best else {
            eprintln!("sound: no usable film, skipping");
            return;
        };
        eprintln!(
            "sound: using {} -- {named} of {total_fire} shots name a weapon ({:.1}%)",
            chosen_path.file_stem().unwrap().to_string_lossy(),
            100.0 * named as f64 / total_fire.max(1) as f64
        );
        if named == 0 {
            eprintln!(
                "sound: NO film on this machine names a weapon at any fire event; the shot gate cannot run and that is a property of the recordings, not of the code"
            );
        }

        let mut pb = Playback::new(idx);
        let mut driver = ActorDriver::default();
        let mut actors = crate::reach_actor::ActorScene::default();
        driver.resolve_bipeds(&pb, &native, cache, Some(&map_path));
        // One production `sync` BEFORE measuring anything: it is what fills
        // `track_of`, `fire_of`, `weap_tags` and the object table, and
        // `App::film_tick` calls it every frame before the audio. Measuring
        // coverage against a driver that had not synced reported 0 of 1,707
        // while the film in fact names 300.
        driver.sync(&pb, &mut actors, &native, cache);
        eprintln!(
            "sound: after sync: {} tracks indexed, {} fire tracks, objects {}",
            driver.track_of.len(),
            driver.fire_of.len(),
            if driver.objects.is_some() { "present" } else { "absent" }
        );
        let mut engine = SoundEngine::default();
        // MUTED: the device really opens and the callback really runs, at zero
        // gain. Mute is not the persisted volume, so this cannot change the
        // user's setting.
        engine.set_muted(true);

        // Seek to the BUSIEST stretch of weapon fire, not the first event.
        // The first version of this probe started at the earliest fire tick
        // and swept 1,200 ticks, which on an eleven-minute film contained
        // exactly two events — a sweep that measured almost nothing while
        // looking like it had run.
        const SWEEP: u32 = 1200;
        // Only the events whose weapon the FILM ACTUALLY NAMES can become a
        // sound, so choose the window among those. This is not narrowing the
        // gate to make it pass: the coverage fraction is measured and printed
        // below, and it is a fact about the recording, not about this module.
        let mut fire_ticks: Vec<u32> = pb
            .index
            .events
            .iter()
            .filter(|e| e.kind == crate::film_action::EV_WEAPON_FIRE)
            .filter(|e| {
                let Some(net) = e.subject else { return false };
                let Some(&ti) = driver.track_of.get(&net) else { return false };
                let Some(tr) = pb.index.tracks.get(ti) else { return false };
                tr.weapon_at(e.tick).is_some()
            })
            .map(|e| e.tick)
            .collect();
        fire_ticks.sort_unstable();
        eprintln!(
            "sound: {} of {total_fire} action_weapon_fire events name a weapon the film carries ({:.1}%); driver.objects = {}",
            fire_ticks.len(),
            100.0 * fire_ticks.len() as f64 / total_fire.max(1) as f64,
            if driver.objects.is_some() { "present" } else { "ABSENT" }
        );
        if fire_ticks.is_empty() {
            eprintln!("sound: this film names no weapon at any fire event; skipping the shot gate");
            return;
        }
        let mut best = (0usize, 0u32);
        for (i, &t) in fire_ticks.iter().enumerate() {
            let j = fire_ticks.partition_point(|x| *x <= t + SWEEP);
            if j - i > best.0 {
                best = (j - i, t);
            }
        }
        let (in_window, start) = best;
        eprintln!(
            "sound: densest {SWEEP}-tick window holds {in_window} of those, from tick {start}"
        );
        // PUT THE LISTENER WHERE A VIEWER PUTS IT. The default camera is
        // `Free`, which in a probe sits at the origin — tens of world units
        // from the action, so the engine's own distance window (weapon_fire's
        // class row stops at 50 wu) correctly rejects every shot. A viewer
        // watching a film follows a player, so follow one: the same
        // `FilmCamera::FollowOwner` a click in the Theater roster sets.
        let watch = driver
            .fire_of
            .iter()
            .max_by_key(|(_, v)| v.len())
            .and_then(|(net, _)| driver.track_of.get(net).copied())
            .and_then(|i| pb.index.tracks.get(i))
            .and_then(|t| t.owner_raw);
        match watch {
            Some(o) => {
                pb.camera = crate::film_play::FilmCamera::FollowOwner(o);
                eprintln!("sound: following owner {o} (the most prolific shooter)");
            }
            None => eprintln!("sound: no owner to follow; the listener stays where the camera solver puts it"),
        }
        pb.playing = true;
        pb.seek_tick(start.saturating_sub(4) as f32);

        // Walk forward one tick at a time, exactly as the clock would, and
        // give the worker time to come up and decode. The listener basis is
        // derived the way `film_tick` derives it from `App::camera`.
        let mut ticks = 0usize;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while ticks < SWEEP as usize + 8 && std::time::Instant::now() < deadline {
            // THE PRODUCTION PASS. `App::film_tick` calls `sync` before the
            // camera and before the audio, and it is `sync` that folds the
            // event channel into `ActorDriver::fire_of` and fills
            // `weap_tags`. Skipping it left `fire_of` empty and the first
            // version of this probe measured zero shots while reporting
            // everything else healthy -- the exact "built its own inputs"
            // failure this project keeps getting burned by.
            driver.sync(&pb, &mut actors, &native, cache);
            let pose = crate::film_play::solve_camera_at(&pb, driver.eye_height(&pb));
            let (pos, yaw, pitch) = match &pose {
                Some(p) => (p.pos, p.yaw, p.pitch),
                None => ([0.0, 0.0, 0.0], 0.0, 0.0),
            };
            let cam = hms_render::Camera {
                pos: glam::Vec3::from(pos),
                yaw,
                pitch,
                fov_y: 1.0,
                near: 0.1,
                far: 1000.0,
            };
            let fwd = cam.forward();
            let right = cam.right();
            let left = -right;
            let up = right.cross(fwd).normalize_or_zero();
            engine.tick(
                &pb,
                &driver,
                &native,
                cache,
                &map_path,
                (cam.pos.into(), fwd.into(), left.into(), up.into()),
            );
            pb.advance(1.0 / crate::film_play::TICK_HZ);
            ticks += 1;
            // The worker decodes off-thread; let it.
            if ticks % 30 == 0 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }

        // DIAGNOSTIC: what does the weapon lane actually hold for the bipeds
        // that fire? `weapon_at` returning None for every one of them is a
        // finding about the film format, not about this module, so measure it
        // rather than guessing.
        {
            let mut have_lane = 0usize;
            let mut have_spawn = 0usize;
            let mut lane_len = 0usize;
            let mut resolved_at_some_tick = 0usize;
            let mut n = 0usize;
            for (net, ft) in driver.fire_of.iter() {
                let Some(&ti) = driver.track_of.get(net) else { continue };
                let Some(tr) = pb.index.tracks.get(ti) else { continue };
                n += 1;
                if !tr.weapons.is_empty() {
                    have_lane += 1;
                    lane_len += tr.weapons.len();
                }
                if tr.spawn_weapon.is_some() {
                    have_spawn += 1;
                }
                if ft.iter().any(|t| tr.weapon_at(*t).is_some()) {
                    resolved_at_some_tick += 1;
                }
            }
            eprintln!(
                "sound: of {n} firing bipeds: {have_lane} have a weapon lane ({lane_len} entries in all), {have_spawn} have a spawn weapon, {resolved_at_some_tick} resolve weapon_at() at one of their own fire ticks"
            );
        }
        eprintln!(
            "sound: driver has {} fire tracks ({} fire events in all), {} weap tags",
            driver.fire_of.len(),
            driver.fire_of.values().map(|v| v.len()).sum::<usize>(),
            driver.weap_tags.len()
        );
        {
            use crate::film_action as fa;
            let mut by_kind: std::collections::BTreeMap<&str, usize> = Default::default();
            for e in pb.index.events.iter().filter(|e| e.tick >= start && e.tick <= start + SWEEP) {
                if matches!(
                    e.kind,
                    fa::EV_WEAPON_FIRE | fa::EV_MELEE_INITIATE | fa::EV_RELOAD | fa::EV_MELEE_CLANG
                ) {
                    *by_kind.entry(fa::event_name(e.kind)).or_default() += 1;
                }
            }
            eprintln!("sound: events with a sound in the swept window: {by_kind:?}");
        }
        let st = engine.status().clone();
        for (i, n) in FIRE_BAIL_NAMES.iter().enumerate() {
            if st.fire_bail[i] > 0 {
                eprintln!("sound:   {:>6} shots dropped: {n}", st.fire_bail[i]);
            }
        }
        eprintln!(
            "sound: {ticks} ticks | device {:?} running={} | bank {} samples | {} decoded | ambience {} on row {} | shots {} | melees {} | reloads {} | clangs {} | shots skipped (not yet decoded) {} | why={:?}",
            st.device, st.running, st.bank_samples, st.decoded, st.ambience_playing,
            st.acoustics_row, st.shots_started, st.melees_started, st.reloads_started,
            st.clangs_started, st.shots_skipped_not_decoded, st.why
        );
        if !st.running {
            eprintln!("sound: no audio device on this machine; the tag and decode side still ran");
        }

        // The bank must have opened and samples must have decoded: that is the
        // worker path working end to end.
        assert!(st.bank_samples > 10_000, "the sfx bank did not open ({} samples)", st.bank_samples);
        assert!(st.decoded > 0, "no sample was decoded in {ticks} ticks");
        // The map's ambience bed must be playing.
        assert!(st.ambience_playing > 0, "no ambience voice started");
        // And real shots must have been started, not merely skipped.
        assert!(
            st.shots_started > 0,
            "no shot started ({} were skipped for want of a decoded sample)",
            st.shots_skipped_not_decoded
        );
        // Melee and reload are only asserted when the swept window carries
        // them: a clip in which nobody reloads is not a bug.
        // Melee and reload are asserted only when the window carries SEVERAL:
        // a single event can legitimately be a weapon with no such sound (a
        // plasma weapon has no reload, the sword no melee swing of its own),
        // and a gate that cannot distinguish "silent because the tag says so"
        // from "silent because the code is broken" is worse than no gate. The
        // tag-side resolution rate is measured separately, over every weapon
        // of every map, in `reach_sound::probes2`.
        let count = |k: u8| {
            pb.index
                .events
                .iter()
                .filter(|e| e.kind == k && e.tick >= start && e.tick <= start + SWEEP)
                .count()
        };
        let (nr, nm) = (count(crate::film_action::EV_RELOAD), count(crate::film_action::EV_MELEE_INITIATE));
        if nr >= 3 {
            assert!(st.reloads_started > 0, "{nr} reload events in the window and none played");
        }
        if nm >= 3 {
            assert!(st.melees_started > 0, "{nm} melee events in the window and none played");
        }
        eprintln!("sound: window had {nr} reloads and {nm} melee swings");
        // Muted throughout: this is the assertion that says nothing was audible.
        assert!(engine.muted, "the probe must stay muted");
        assert_eq!(engine.master(), 0.0, "the master gain must be zero while muted");
    }

    /// A seek must silence everything rather than leave a shot ringing, and a
    /// paused film must start nothing.
    #[test]
    fn probe_a_seek_stops_every_voice() {
        let _serial = crate::reach_sound::probes_support::PROBE_LOCK.lock();
        let Some((native, cache, map_path)) = open_cache_for("forge_halo") else {
            eprintln!("sound: Reach not installed, skipping");
            return;
        };
        let films = film::find_finished_films();
        let Some(p) = films.first() else {
            eprintln!("sound: no films, skipping");
            return;
        };
        let Ok(f) = film::open(p) else { return };
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            return;
        }
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        let mut pb = Playback::new(idx);
        let driver = ActorDriver::default();
        let mut engine = SoundEngine::default();
        engine.set_muted(true);
        let listener = ([0.0f32; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);

        // Run forward until the ambience is up.
        pb.playing = true;
        for _ in 0..600 {
            engine.tick(&pb, &driver, &native, cache, &map_path, listener);
            pb.advance(1.0 / crate::film_play::TICK_HZ);
            if engine.status().ambience_playing > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        if engine.status().ambience_playing == 0 {
            eprintln!("sound: ambience never started (no device or no bank); skipping the seek check");
            return;
        }

        // A big backwards seek is not smooth playback: everything must stop.
        pb.seek_tick(0.0);
        engine.tick(&pb, &driver, &native, cache, &map_path, listener);
        // Pausing must start nothing new.
        pb.playing = false;
        engine.tick(&pb, &driver, &native, cache, &map_path, listener);
        assert_eq!(
            engine.status().ambience_playing,
            0,
            "a paused film must hold no ambience voice"
        );
        assert!(engine.muted);
    }
}
