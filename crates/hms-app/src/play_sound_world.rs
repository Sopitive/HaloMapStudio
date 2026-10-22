//! #play-sound  The WORLD half of test mode's sound: what the player's own body makes, and what
//! the map makes around them.
//!
//! * **The body.** Reach does not decide footsteps in code. The biped's animation graph carries
//!   them: every locomotion clip has *frame events* (`left foot` / `right foot` at the frames the
//!   foot lands) and *sound events* (the armour and cloth foley, `spartan_move_short`, the
//!   sprint breath). A foot event is turned into a sound by the biped's `foot` tag
//!   (`material_effects`, `obje +0xC0`): 22 effect TYPES, each a table of `material name -> snd!`.
//!   The material is the SURFACE under the foot. Jump and land are further effect types.
//! * **The map.** The scenario's *sound scenery* (`scnr +0x1C8`, 140-byte placements) are objects
//!   whose tag carries `lsnd` attachments: the waterfalls and the lapping water of Forge World.
//!   They are positional loops with the snd!'s own distance curve, and they are what you hear
//!   standing near them. (The room-tone BED is the acoustics palette, in `sound.rs`.)
//!
//! # PROVEN (this round, forge_halo)
//! * Spartan `foot` = `fx\material_effects\characters\spartan`: effect[0] sounds
//!   `...\walk`, [1] `...\run`, [3] `...\shuffle`, [4] `...\jump`, [5] `...\land`, [6]
//!   `...\dash`, [8] `...\land_hard`; rows are per material (`hard`, `brittle`, `energy`, ...).
//! * `combat:rifle:move_front_fast` (22 frames): feet at 5 (right) and 16 (left), foley at 0 and
//!   11. `sprint:rifle:move_front_fast` (17 frames): feet at 7 and 16, foley
//!   `allow_player_spartan_move_short_*` at 2 and 11.
//! * `scnr +0x1C8` sound scenery: stride 140, `palette index` i16 at +0, position at +8..+0x14;
//!   the palette (`+0x1D4`, 16 B) names `ssce` tags whose attachments are the loops.
//!
//! # INFERRED (labelled where used)
//! * Which effect TYPE a given gait fires: walk 0, run 1, crouch-move 3, sprint 6, jump 4, soft
//!   land 5, hard land 8 -- read from the authored sound paths (`walk`, `run`, `shuffle`, `dash`,
//!   `jump`, `land`, `land_hard`), not from engine code.
//! * A Forge object underfoot (no level triangle below) is treated as `hard`.

use std::collections::HashMap;

use glam::Vec3;
use hms_native::{CacheHandle, NativeDll};

use super::*;
use crate::play_ctl::MoveInput;
use crate::reach_actor::{block_at, tag_block};

/// The body's locomotion clip and the frame it was posed at this tick (30 fps clip frames).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyClip {
    pub anim: Option<usize>,
    pub frame: f32,
    pub frames: f32,
}

/// What one frame of the player's body gives the sound.
#[derive(Clone, Copy, Debug)]
pub struct FootFrame {
    pub body: BodyClip,
    pub input: MoveInput,
    pub pos: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
    pub jumped: bool,
    pub landed: bool,
    pub biped: u32,
    pub riding: bool,
    /// The level triangle's render-material index straight below the feet.
    pub ground_material: Option<i32>,
}

/// `foot` effect TYPES (INFERRED from the authored paths, see the module note).
pub mod effect {
    pub const WALK: usize = 0;
    pub const RUN: usize = 1;
    pub const SHUFFLE: usize = 3;
    pub const JUMP: usize = 4;
    pub const LAND: usize = 5;
    pub const DASH: usize = 6;
    pub const LAND_HARD: usize = 8;
}

/// Which effect type a foot-down fires, from what the player is doing.
pub fn foot_effect(input: &MoveInput, crouched: bool) -> usize {
    if crouched {
        effect::SHUFFLE
    } else if input.sprint {
        effect::DASH
    } else if input.walk {
        effect::WALK
    } else {
        effect::RUN
    }
}

#[derive(Clone, Debug)]
struct FootRow {
    material: String,
    tag: Option<u32>,
    secondary: Option<u32>,
}

#[derive(Clone, Debug, Default)]
struct AnimEvents {
    frames: f32,
    /// `(kind, frame)`: kind 4 left, 5 right, 8 both.
    feet: Vec<(u16, f32)>,
    /// `(sound reference index, frame)`.
    sounds: Vec<(u16, f32)>,
}

/// Everything read once per biped tag.
pub struct BipedSounds {
    jmad: u32,
    refs: Vec<(u32, u8)>,
    foot: Vec<Vec<FootRow>>,
    events: HashMap<usize, AnimEvents>,
    clips: HashMap<u32, Option<Clip>>,
    /// `bipd +0x514 / +0x518`: below the first a landing makes no reaction, above the second it is
    /// a hard one (world units per second).
    land_soft: f32,
    land_hard: f32,
}

const BIPD_MIN_SOFT_LANDING_VELOCITY: usize = 0x514;
const BIPD_MIN_HARD_LANDING_VELOCITY: usize = 0x518;

impl BipedSounds {
    pub fn read(native: &NativeDll, cache: CacheHandle, biped: u32) -> Option<BipedSounds> {
        use crate::reach_damage as rd;
        let jmad = crate::reach_actor::resolve_jmad(native, cache, biped)?;
        let refs = rs::graph_sound_refs(native, cache, jmad);
        let foot_tag = {
            let m = native.tag_meta(cache, biped, rd::OBJE_MATERIAL_EFFECTS as u32, 16)?;
            let d = rd_u32(&m, 0x0C);
            (d != u32::MAX && d != 0).then_some(d & 0xFFFF)?
        };
        let outer = tag_block(native, cache, foot_tag, rd::FOOT_EFFECTS, rd::MATERIAL_EFFECT_SIZE, 64);
        let mut foot = Vec::new();
        for e in outer.chunks_exact(rd::MATERIAL_EFFECT_SIZE) {
            let rows = block_at(native, cache, e, rd::MATERIAL_EFFECT_SOUNDS, rd::MATERIAL_ROW_SIZE, 512);
            let tr = |row: &[u8], o: usize| -> Option<u32> {
                let d = rd_u32(row, o + 0x0C);
                (d != u32::MAX && d != 0).then_some(d & 0xFFFF)
            };
            foot.push(
                rows.chunks_exact(rd::MATERIAL_ROW_SIZE)
                    .map(|row| FootRow {
                        material: sid_name(native, cache, rd_u32(row, rd::MATERIAL_ROW_NAME)),
                        tag: tr(row, rd::MATERIAL_ROW_TAG),
                        secondary: tr(row, rd::MATERIAL_ROW_SECONDARY),
                    })
                    .collect(),
            );
        }
        let h = native.tag_meta(cache, biped, BIPD_MIN_SOFT_LANDING_VELOCITY as u32, 12)?;
        Some(BipedSounds {
            jmad,
            refs,
            foot,
            events: HashMap::new(),
            clips: HashMap::new(),
            land_soft: rd_f32(&h, 0),
            land_hard: rd_f32(&h, BIPD_MIN_HARD_LANDING_VELOCITY - BIPD_MIN_SOFT_LANDING_VELOCITY),
        })
    }

    fn events(&mut self, native: &NativeDll, cache: CacheHandle, anim: usize) -> &AnimEvents {
        let jmad = self.jmad;
        self.events.entry(anim).or_insert_with(|| {
            let frames = rs::animation_frame_count(native, cache, jmad, anim).unwrap_or(0) as f32;
            AnimEvents {
                frames,
                feet: rs::animation_foot_events(native, cache, jmad, anim).iter().map(|e| (e.kind, f32::from(e.frame))).collect(),
                sounds: rs::animation_sound_events(native, cache, jmad, anim)
                    .iter()
                    // A negative frame (65534 = -2) is "at the start of the clip".
                    .map(|e| (e.sound, if (e.frame as i16) < 0 { 0.0 } else { f32::from(e.frame) }))
                    .collect(),
            }
        })
    }

    fn clip(&mut self, eng: &SoundEngine, native: &NativeDll, cache: CacheHandle, snd: u32) -> Option<Clip> {
        self.clips
            .entry(snd)
            .or_insert_with(|| {
                let t = rs::resolve_to_snd(native, cache, snd)?;
                eng.resolve_one_shot(native, cache, t)
            })
            .clone()
    }

    /// The row a material plays for an effect type: its own, else `default_material`.
    fn row(&self, effect: usize, material: &str) -> Option<&FootRow> {
        let rows = self.foot.get(effect)?;
        rows.iter()
            .find(|r| r.material.eq_ignore_ascii_case(material) && (r.tag.is_some() || r.secondary.is_some()))
            .or_else(|| rows.iter().find(|r| r.material.eq_ignore_ascii_case("default_material") && (r.tag.is_some() || r.secondary.is_some())))
    }

    /// Ask for every sample the common effect types play on `material`, ahead of the first step.
    pub fn warm_material(&mut self, eng: &mut SoundEngine, native: &NativeDll, cache: CacheHandle, material: &str) {
        for e in [effect::WALK, effect::RUN, effect::SHUFFLE, effect::JUMP, effect::LAND, effect::DASH, effect::LAND_HARD] {
            let Some(tag) = self.row(e, material).and_then(|r| r.tag.or(r.secondary)) else { continue };
            if let Some(c) = self.clip(eng, native, cache, tag) {
                for &s in c.samples.iter().take(8) {
                    let _ = eng.sample(s);
                }
            }
        }
    }
}

/// The body's running state between frames.
#[derive(Default)]
pub struct BodyState {
    last_anim: Option<usize>,
    last_frame: f32,
    air_vz: f32,
    was_grounded: bool,
    seed: u32,
    warmed: Option<String>,
}

/// What the body made, for the caption and the gates.
#[derive(Clone, Debug, Default)]
pub struct BodyStats {
    pub steps: u64,
    pub foley: u64,
    pub jumps: u64,
    pub lands: u64,
    /// Foot events that found no sound for the material (silence by the tag).
    pub steps_silent: u64,
    pub last_material: String,
}

/// Render material -> global material NAME for this map: `ground_material_index` is a BSP render
/// material index; its shader names a collision material whose global material index is the
/// `matg` row (see `reach_sound::bsp_surface_materials`).
pub struct GroundNames {
    sbsps: Vec<u32>,
    names: Vec<String>,
    by_shader: Vec<std::collections::HashMap<u32, String>>,
    cache: HashMap<i32, Option<String>>,
}

impl GroundNames {
    pub fn read(native: &NativeDll, cache: CacheHandle) -> GroundNames {
        let names = rs::global_material_names(native, cache);
        let sbsps = native.find_scenario(cache).map(|s| native.enumerate_sbsps(cache, s, 16)).unwrap_or_default();
        let by_shader = sbsps.iter().map(|sb| rs::bsp_surface_materials(native, cache, *sb, &names)).collect();
        GroundNames { sbsps, names, by_shader, cache: HashMap::new() }
    }

    /// The material NAME for a render-material index. With several BSPs the index is ambiguous, so
    /// the first BSP whose shader for that index has a collision material wins (INFERRED).
    pub fn name(&mut self, native: &NativeDll, cache: CacheHandle, idx: i32) -> Option<String> {
        if let Some(n) = self.cache.get(&idx) {
            return n.clone();
        }
        let mut found = None;
        for (i, sb) in self.sbsps.iter().enumerate() {
            if let Some(shader) = rs::bsp_render_material_shader(native, cache, *sb, idx as usize) {
                if let Some(n) = self.by_shader[i].get(&shader) {
                    found = Some(n.clone());
                    break;
                }
            }
        }
        let _ = &self.names;
        self.cache.insert(idx, found.clone());
        found
    }
}

impl PlaySound {
    /// The body's sounds for one frame: foot-downs and foley from the clip's events, jump, land.
    pub(super) fn body_frame(&mut self, eng: &mut SoundEngine, f: &Frame, foot: &FootFrame) {
        if foot.biped == 0 {
            return;
        }
        if !self.bipeds.contains_key(&foot.biped) {
            let b = BipedSounds::read(f.native, f.cache, foot.biped);
            self.bipeds.insert(foot.biped, b);
        }
        // Take the tables out so `self` stays free for the stats.
        let Some(mut bs) = self.bipeds.get_mut(&foot.biped).and_then(|b| b.take()) else { return };
        if self.ground.is_none() {
            self.ground = Some(GroundNames::read(f.native, f.cache));
        }
        let name: String = foot
            .ground_material
            .and_then(|i| self.ground.as_mut().and_then(|g| g.name(f.native, f.cache, i)))
            // A Forge object (or anything the level raycast does not see) underfoot.
            .unwrap_or_else(|| "hard".to_string());
        self.body_stats.last_material = name.clone();
        if self.body.warmed.as_deref() != Some(name.as_str()) {
            bs.warm_material(eng, f.native, f.cache, &name);
            self.body.warmed = Some(name.clone());
        }

        let listener = f.listener;
        let feet_at = (foot.pos + Vec3::new(0.0, 0.0, 0.08)).to_array();
        self.body.seed = self.body.seed.wrapping_add(1);
        let seed = self.body.seed;

        // --- airborne bookkeeping (vertical speed just before touching down) ---
        if !foot.grounded {
            self.body.air_vz = self.body.air_vz.min(foot.vel.z);
        }

        // --- the clip's events between the last frame and this one ---
        if let (Some(anim), false) = (foot.body.anim, foot.body.frames <= 0.0) {
            let (prev, cur) = if self.body.last_anim == Some(anim) {
                (self.body.last_frame, foot.body.frame)
            } else {
                // A new clip: a one-shot starts near frame 0 and fires from its beginning; a
                // locomotion cycle entered mid-stride must not replay the events behind it.
                (if foot.body.frame <= 2.0 { -1.0 } else { foot.body.frame }, foot.body.frame)
            };
            let n = foot.body.frames;
            let ev = bs.events(f.native, f.cache, anim).clone();
            let in_window = |fr: f32| -> bool {
                if cur >= prev {
                    fr > prev && fr <= cur
                } else if prev - cur > n * 0.5 {
                    // The cycle wrapped.
                    fr > prev || fr <= cur
                } else {
                    // Moving backward (a ping-pong idle): nothing fires.
                    false
                }
            };
            let crouched = foot.input.crouch;
            for (kind, fr) in ev.feet.iter().copied() {
                if !in_window(fr) || !foot.grounded || foot.riding {
                    continue;
                }
                let _ = kind;
                let eff = foot_effect(&foot.input, crouched);
                let ok = self.play_effect(eng, f, &mut bs, eff, &name, feet_at, listener, seed);
                self.note(super::Cat::Step, ok, eng);
                if ok {
                    self.body_stats.steps += 1;
                } else {
                    self.body_stats.steps_silent += 1;
                }
            }
            for (r, fr) in ev.sounds.iter().copied() {
                if !in_window(fr) {
                    continue;
                }
                let Some(&(snd, _)) = bs.refs.get(r as usize) else { continue };
                if snd == 0 {
                    continue;
                }
                if let Some(c) = bs.clip(eng, f.native, f.cache, snd) {
                    for &s in c.samples.iter().take(8) {
                        let _ = eng.sample(s);
                    }
                    let ok = eng.play_clip_at(&c, feet_at, listener, seed.wrapping_add(u32::from(r)));
                    self.note(super::Cat::Foley, ok, eng);
                    if ok {
                        self.body_stats.foley += 1;
                    }
                }
            }
            self.body.last_anim = Some(anim);
            self.body.last_frame = cur;
        } else {
            self.body.last_anim = None;
        }

        // --- jump and landing ---
        if foot.jumped && !foot.riding {
            let ok = self.play_effect(eng, f, &mut bs, effect::JUMP, &name, feet_at, listener, seed);
            self.note(super::Cat::Jump, ok, eng);
            if ok {
                self.body_stats.jumps += 1;
            }
        }
        if foot.landed && !foot.riding {
            let v = self.body.air_vz.abs();
            if v >= bs.land_soft {
                let eff = if v >= bs.land_hard { effect::LAND_HARD } else { effect::LAND };
                let ok = self.play_effect(eng, f, &mut bs, eff, &name, feet_at, listener, seed);
                self.note(super::Cat::Land, ok, eng);
                if ok {
                    self.body_stats.lands += 1;
                }
            }
        }
        if foot.grounded {
            self.body.air_vz = 0.0;
        }
        self.body.was_grounded = foot.grounded;
        if let Some(slot) = self.bipeds.get_mut(&foot.biped) {
            *slot = Some(bs);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn play_effect(
        &mut self,
        eng: &mut SoundEngine,
        f: &Frame,
        bs: &mut BipedSounds,
        effect: usize,
        material: &str,
        at: [f32; 3],
        listener: Listener,
        seed: u32,
    ) -> bool {
        let Some(tag) = bs.row(effect, material).and_then(|r| r.tag.or(r.secondary)) else {
            eng.refuse(crate::sound::Refusal::NoClip);
            return false;
        };
        let Some(c) = bs.clip(eng, f.native, f.cache, tag) else {
            eng.refuse(crate::sound::Refusal::NoClip);
            return false;
        };
        eng.play_clip_at(&c, at, listener, seed)
    }
}

// ---------------------------------------------------------------------------------------------
// SOUND SCENERY: the map's positional loops
// ---------------------------------------------------------------------------------------------

pub const SCNR_SOUND_SCENERY: usize = 0x1C8;
pub const SCNR_SOUND_SCENERY_PALETTE: usize = 0x1D4;
pub const SOUND_SCENERY_STRIDE: usize = 140;
const SS_POSITION: usize = 0x08;

pub struct SceneryPlacement {
    pub pos: Vec3,
    pub source: usize,
}

/// A map's sound scenery: the placements and, per palette entry, the loops its tag attaches.
#[derive(Default)]
pub struct Scenery {
    pub placements: Vec<SceneryPlacement>,
    pub sources: Vec<VehicleSounds>,
    pub names: Vec<String>,
    pub(crate) voices: HashMap<(u16, u8, u8, u8), Vox>,
    ready: bool,
}

/// At most this many placements sound at once; the nearest win.
const SCENERY_ACTIVE: usize = 10;
/// A placement whose loops are all farther than this beyond their maximum distance is skipped.
const SCENERY_REACH_MARGIN: f32 = 2.0;
/// Fade-in of a scenery voice as the listener walks into its range, seconds.
const SCENERY_RAMP_SECONDS: f32 = 0.4;

impl Scenery {
    pub fn read(eng: &SoundEngine, native: &NativeDll, cache: CacheHandle) -> Scenery {
        let mut s = Scenery { ready: true, ..Default::default() };
        let Some(scnr) = native.find_scenario(cache) else { return s };
        let pal = tag_block(native, cache, scnr, SCNR_SOUND_SCENERY_PALETTE, 16, 256);
        for c in pal.chunks_exact(16) {
            let t = rs::tag_ref_at(c, 0);
            s.names.push(t.and_then(|t| native.tag_name(cache, t)).unwrap_or_default());
            s.sources.push(t.map(|t| VehicleSounds::read(eng, native, cache, t)).unwrap_or_default());
        }
        let raw = tag_block(native, cache, scnr, SCNR_SOUND_SCENERY, SOUND_SCENERY_STRIDE, 512);
        for e in raw.chunks_exact(SOUND_SCENERY_STRIDE) {
            let idx = i16::from_le_bytes([e[0], e[1]]);
            if idx < 0 || idx as usize >= s.sources.len() || s.sources[idx as usize].sources.is_empty() {
                continue;
            }
            let pos = Vec3::new(rd_f32(e, SS_POSITION), rd_f32(e, SS_POSITION + 4), rd_f32(e, SS_POSITION + 8));
            if pos.is_finite() {
                s.placements.push(SceneryPlacement { pos, source: idx as usize });
            }
        }
        s
    }

    pub fn live_voices(&self) -> usize {
        self.voices.len()
    }

    /// Ask for every sample of every loop (they are long) so a walk past a waterfall is not
    /// silent for its first seconds.
    fn warm(&self, eng: &mut SoundEngine) {
        for (n, vs) in self.sources.iter().enumerate() {
            let _ = n;
            for ls in vs.sources.iter().flat_map(|s| s.tracks.iter()).flat_map(|t| t.lp.iter().chain(t.alt.iter())) {
                for r in &ls.ranges {
                    for &s in r.samples.iter().take(2) {
                        let _ = eng.sample(s);
                    }
                }
            }
        }
    }
}

impl PlaySound {
    pub(super) fn scenery_frame(&mut self, eng: &mut SoundEngine, f: &Frame) {
        if !self.scenery.ready {
            self.scenery = Scenery::read(eng, f.native, f.cache);
            self.scenery.warm(eng);
        }
        let (lpos, fwd, left, up) = f.listener;
        let lp = Vec3::from(lpos);
        // The nearest few placements.
        let mut order: Vec<(f32, usize)> = self.scenery.placements.iter().enumerate().map(|(i, p)| ((p.pos - lp).length(), i)).collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut wanted: HashMap<(u16, u8, u8, u8), ()> = HashMap::new();
        for &(dist, pi) in order.iter().take(SCENERY_ACTIVE) {
            let pl = &self.scenery.placements[pi];
            let pos = pl.pos;
            let vs = &self.scenery.sources[pl.source];
            let dir = rs::to_listener_space(pos.to_array(), lpos, fwd, left, up);
            let pan = rs::stereo_pan(dir);
            for (si, src) in vs.sources.iter().enumerate() {
                for (ti, tr) in src.tracks.iter().enumerate() {
                    let Some(ls) = tr.lp.as_ref() else { continue };
                    let reach = if ls.spatial { ls.dist.maximum } else { 80.0 };
                    if dist > reach + SCENERY_REACH_MARGIN {
                        continue;
                    }
                    let scale = 1.0f32;
                    let pitch = ls.pitch_at(scale);
                    for (ri, range_db) in ls.ranges_at(pitch, None) {
                        let r = &ls.ranges[ri];
                        if r.samples.is_empty() {
                            continue;
                        }
                        let db = eng.loop_gain_db(ls, scale, r.perm_gain_db.first().copied().unwrap_or(0), tr.gain_db + range_db);
                        let mut lin = rs::db_to_linear(db);
                        if ls.spatial {
                            lin *= if dist > ls.dist.maximum || dist < ls.dist.dont_play { 0.0 } else { ls.dist.gain_at(dist) };
                        }
                        let gain = [lin * pan[0], lin * pan[1]];
                        let rate = pitch_ratio(pitch, r.natural_pitch);
                        let key = (pi as u16, (si * 4 + ti) as u8, 0u8, ri as u8);
                        wanted.insert(key, ());
                        match self.scenery.voices.get_mut(&key) {
                            Some(v) => {
                                v.gain = gain;
                                v.mult = rate;
                            }
                            None => {
                                if let Some((id, src_rate)) = eng.voice_start(&r.samples, ls.cs.flags & rs::SND_FLAG_PERMUTATIONS_IN_ORDER != 0, [0.0, 0.0], rate) {
                                    self.scenery.voices.insert(key, Vox { id, rate: src_rate, gain, mult: rate, born: Some((SCENERY_RAMP_SECONDS, SCENERY_RAMP_SECONDS)), dying: None });
                                    self.stats.voices_started += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let dt = f.dt;
        self.scenery.voices.retain(|k, v| {
            if !wanted.contains_key(k) {
                eng.voice_stop(v.id);
                return false;
            }
            let mut g = 1.0;
            if let Some((left, total)) = v.born.as_mut() {
                *left -= dt;
                g = (1.0 - *left / *total).clamp(0.0, 1.0);
                if *left <= 0.0 {
                    v.born = None;
                }
            }
            eng.voice_set(v.id, v.rate, [v.gain[0] * g, v.gain[1] * g], v.mult);
            true
        });
    }
}
