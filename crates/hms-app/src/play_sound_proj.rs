//! #projectile-sound  A projectile's own LOOPING sounds: the `lsnd` attachments of its `proj` tag.
//!
//! A rocket hisses, a needle hums, a grenade ticks, a flak shell whines. None of that is in the
//! particle or contrail effects: it is an ordinary `obje` attachment (`obje +0x104`, 32-byte rows:
//! tag reference at +0x00 of class `lsnd`, marker string id +0x10, primary-scale function +0x18),
//! the very block a vehicle's engine loops come from. So this module reads a projectile's
//! attachments with the vehicle reader ([`crate::play_sound::VehicleSounds::read`]) and plays each
//! track the way the ride does: pitch range by the track's scale, `fade in` while the source is
//! on, the distance curve and pan relative to the listener. What is new is only the owner: a
//! voice belongs to ONE projectile, is aimed at that projectile's position every frame (so it
//! follows the shot), and stops the frame the projectile ends (its `out` one-shot, when the track
//! authors one, plays where it ended).
//!
//! Which projectiles carry one (forge_halo, `probe_projectile_loops`): see the gate.
//!
//! # Not modelled
//! The attachment's `marker` (the rocket's `exhaust`) is not looked up: the voice sits on the
//! projectile's origin, which for a projectile-sized model is centimetres off. Scale functions
//! other than "none authored" and the tag's own exported functions are not available (a
//! projectile has no engine input) and stay silent, reported in [`ProjectileLoops::unresolved`].

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use glam::Vec3;
use hms_native::{CacheHandle, NativeDll};
use hms_sound::VoiceId;

use crate::play_sound::{pitch_ratio, Inputs, Listener, VehicleSounds};
use crate::reach_sound as rs;
use crate::sound::SoundEngine;

/// A projectile as the sound sees it this frame.
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    /// Unique per projectile for its life (`play_projectile::Live::id`).
    pub id: u32,
    /// Its `proj` tag.
    pub proj: u32,
    pub pos: Vec3,
    /// #vehicle-weapons  1.0 while the fire button holds the trigger of the weapon this is (a
    /// horn, a chain gun's spin); 0 for a projectile.
    pub trigger: f32,
}

struct Voice {
    id: VoiceId,
    rate: u32,
    gain: [f32; 2],
    mult: f64,
}

#[derive(Default)]
struct Rt {
    voices: HashMap<(u8, u8, bool, u8), Voice>,
    /// 0..1 fade level per `(source, track)`.
    fade: HashMap<(u8, u8), f32>,
    on: HashMap<(u8, u8), bool>,
    last: Vec3,
    proj: u32,
}

/// Every projectile's looping sounds, owned by [`crate::play_sound::PlaySound`].
#[derive(Default)]
pub struct ProjectileLoops {
    defs: HashMap<u32, Arc<VehicleSounds>>,
    live: HashMap<u32, Rt>,
    /// Loop voices created / stopped, and `out` one-shots played, since the session began.
    pub started: u64,
    pub stopped: u64,
    pub outros: u64,
    /// Scale-function names a source asked for that a projectile cannot answer.
    pub unresolved: BTreeSet<String>,
}

impl ProjectileLoops {
    /// How many loop voices are playing now.
    pub fn voices(&self) -> usize {
        self.live.values().map(|r| r.voices.len()).sum()
    }

    /// Projectiles with at least one voice.
    pub fn sounding(&self) -> usize {
        self.live.values().filter(|r| !r.voices.is_empty()).count()
    }

    /// How many looping sources the `proj` tag authors (resolved, with a playable track).
    pub fn sources_of(&mut self, eng: &SoundEngine, native: &NativeDll, cache: CacheHandle, proj: u32) -> usize {
        self.def(eng, native, cache, proj).sources.len()
    }

    fn def(&mut self, eng: &SoundEngine, native: &NativeDll, cache: CacheHandle, proj: u32) -> Arc<VehicleSounds> {
        self.defs.entry(proj).or_insert_with(|| Arc::new(VehicleSounds::read(eng, native, cache, proj))).clone()
    }

    /// Stop every voice (leaving test mode).
    pub fn stop_all(&mut self, eng: &mut SoundEngine) {
        for (_, rt) in self.live.drain() {
            for (_, v) in rt.voices {
                eng.voice_stop(v.id);
                self.stopped += 1;
            }
        }
    }

    /// One frame: `shots` are the projectiles in flight NOW. A projectile that is no longer listed
    /// has ended: its voices stop and its `out` clips play where it was last seen.
    pub fn update(&mut self, eng: &mut SoundEngine, native: &NativeDll, cache: CacheHandle, shots: &[Shot], listener: Listener, dt: f32) {
        let (lpos, fwd, left, up) = listener;
        let alive: BTreeSet<u32> = shots.iter().map(|s| s.id).collect();
        // ended
        let gone: Vec<u32> = self.live.keys().copied().filter(|id| !alive.contains(id)).collect();
        for id in gone {
            let Some(rt) = self.live.remove(&id) else { continue };
            for (_, v) in rt.voices {
                eng.voice_stop(v.id);
                self.stopped += 1;
            }
            let def = self.def(eng, native, cache, rt.proj);
            for (si, src) in def.sources.iter().enumerate() {
                for (ti, tr) in src.tracks.iter().enumerate() {
                    if rt.on.get(&(si as u8, ti as u8)).copied().unwrap_or(false) {
                        if let Some(c) = tr.outro.as_ref() {
                            if eng.play_clip_at(c, rt.last.to_array(), listener, id.wrapping_mul(8).wrapping_add((si * 8 + ti) as u32)) {
                                self.outros += 1;
                            }
                        }
                    }
                }
            }
        }
        for s in shots {
            let def = self.def(eng, native, cache, s.proj);
            if def.sources.is_empty() {
                continue;
            }
            let rt = self.live.entry(s.id).or_default();
            rt.proj = s.proj;
            rt.last = s.pos;
            let dir = rs::to_listener_space(s.pos.to_array(), lpos, fwd, left, up);
            let dist = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            let pan = rs::stereo_pan(dir);
            for (si, src) in def.sources.iter().enumerate() {
                // No scale function authored = the attachment is unconditional (as for a placed object).
                let (scale, active) = if src.scale_name.is_empty() {
                    (1.0, true)
                } else {
                    def.scale_of(src, &Inputs { trigger: s.trigger, ..Default::default() }, &mut self.unresolved).unwrap_or((0.0, false))
                };
                for (ti, tr) in src.tracks.iter().enumerate() {
                    let tk = (si as u8, ti as u8);
                    let secs = if active { tr.fade_in } else { tr.fade_out };
                    let step = if secs > 1.0e-3 { dt / secs } else { 1.0 };
                    let fade = rt.fade.entry(tk).or_insert(0.0);
                    *fade = if active { (*fade + step).min(1.0) } else { (*fade - step).max(0.0) };
                    let fade = *fade;
                    if active && !rt.on.get(&tk).copied().unwrap_or(false) {
                        if let Some(c) = tr.intro.as_ref() {
                            eng.play_clip_at(c, s.pos.to_array(), listener, s.id.wrapping_mul(8).wrapping_add((si * 8 + ti) as u32));
                        }
                    }
                    rt.on.insert(tk, active);
                    if fade <= 0.0 {
                        continue;
                    }
                    let fade_db = rs::interp_db(if active { tr.fade_in_mode } else { tr.fade_out_mode }.min(4), rs::MIN_GAIN_DB, 0.0, fade);
                    for (alt, ls) in [(false, tr.lp.as_ref()), (true, tr.alt.as_ref())] {
                        let Some(ls) = ls else { continue };
                        // a projectile never "loads": the alternate loop only plays when it is the only one
                        if alt && tr.lp.is_some() {
                            continue;
                        }
                        let pitch = ls.pitch_at(scale);
                        let ranges = ls.ranges_at(pitch, None);
                        for (ri, range_db) in ranges {
                            let r = &ls.ranges[ri];
                            if r.samples.is_empty() {
                                continue;
                            }
                            let db = eng.loop_gain_db(ls, scale, r.perm_gain_db.first().copied().unwrap_or(0), tr.gain_db + fade_db + range_db);
                            let mut lin = rs::db_to_linear(db);
                            if ls.spatial {
                                lin *= if dist > ls.dist.maximum || dist < ls.dist.dont_play { 0.0 } else { ls.dist.gain_at(dist) };
                            }
                            let gain = [lin * pan[0], lin * pan[1]];
                            let mult = pitch_ratio(pitch, r.natural_pitch);
                            let key = (si as u8, ti as u8, alt, ri as u8);
                            match rt.voices.get_mut(&key) {
                                Some(v) => {
                                    v.gain = gain;
                                    v.mult = mult;
                                    eng.voice_set(v.id, v.rate, gain, mult);
                                }
                                None => {
                                    let in_order = ls.cs.flags & rs::SND_FLAG_PERMUTATIONS_IN_ORDER != 0;
                                    if let Some((id, rate)) = eng.voice_start(&r.samples, in_order, gain, mult) {
                                        rt.voices.insert(key, Voice { id, rate, gain, mult });
                                        self.started += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Map-backed (`forge_halo`) gates on an OFFLINE mixer: no output device is ever opened.
    //! `cargo test --release -p hms-app play_sound_proj -- --ignored --nocapture --test-threads=1 --skip sound::probes`
    use super::*;
    use std::time::{Duration, Instant};

    const RATE: u32 = 48_000;
    const DT: f32 = 1.0 / 60.0;
    const FRAMES: usize = (RATE / 60) as usize;

    struct Rig {
        _g: std::sync::MutexGuard<'static, ()>,
        native: Arc<NativeDll>,
        cache: CacheHandle,
        eng: SoundEngine,
        loops: ProjectileLoops,
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
        Some(Rig { _g: g, native, cache, eng, loops: ProjectileLoops::default() })
    }

    fn proj_named(rig: &Rig, leaf: &str) -> u32 {
        rig.native
            .tags_by_group(rig.cache, "proj")
            .into_iter()
            .find(|&t| rig.native.tag_name(rig.cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf)))
            .unwrap_or_else(|| panic!("no projectile {leaf}"))
    }

    /// Listener at the origin facing +X with +Y to its left.
    fn listener() -> Listener {
        ([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    }

    /// One 60 Hz frame with `shots` in flight; returns the stereo audio of that frame.
    fn frame(rig: &mut Rig, shots: &[Shot]) -> Vec<f32> {
        rig.eng.cat_hint = 4;
        rig.loops.update(&mut rig.eng, &rig.native, rig.cache, shots, listener(), DT);
        rig.eng.render(FRAMES)
    }

    fn lr(a: &[f32]) -> (f32, f32) {
        let (mut l, mut r) = (0.0f32, 0.0f32);
        for c in a.chunks_exact(2) {
            l += c[0] * c[0];
            r += c[1] * c[1];
        }
        (l.sqrt(), r.sqrt())
    }

    /// Which projectiles of `forge_halo` author a looping sound (the `lsnd` attachments of the `proj`
    /// tags), read through the same reader the player's sound uses. LITERAL: a projectile that gains or
    /// loses one is news.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_projectiles_that_carry_a_loop_are_the_authored_ones() {
        let Some(mut rig) = rig() else { return };
        let mut with: Vec<String> = Vec::new();
        let mut total = 0;
        for p in rig.native.tags_by_group(rig.cache, "proj") {
            total += 1;
            let n = rig.loops.sources_of(&rig.eng, &rig.native, rig.cache, p);
            if n > 0 {
                with.push(rig.native.tag_name(rig.cache, p).unwrap_or_default().rsplit('\\').next().unwrap_or("").to_string());
            }
        }
        with.sort();
        eprintln!("{} of {total} projectiles carry a loop: {with:?}", with.len());
        assert!(total >= 40, "liveness: {total} projectiles examined");
        for must in ["rocket_launcher_rocket", "needler_shard", "grenade_launcher_grenade", "plasma_grenade", "frag_grenade", "flak_bolt", "plasma_launcher_grenade"] {
            assert!(with.iter().any(|w| w == must), "{must} should carry a looping sound; the ones that do: {with:?}");
        }
        for not in ["assault_rifle_bullet", "dmr_bullet", "plasma_rifle_bolt", "sniper_rifle_bullet"] {
            assert!(!with.iter().any(|w| w == not), "{not} authors no looping sound but one was found");
        }
    }

    /// **THE LOOP LIVES WITH THE ROCKET.** It starts when the projectile appears, is HEARD from where
    /// the projectile is (left of the listener while it is on the left, right once it has crossed over),
    /// follows it as it moves, and is gone the frame the projectile ends.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_rockets_loop_starts_follows_and_stops_with_it() {
        let Some(mut rig) = rig() else { return };
        let proj = proj_named(&rig, "rocket_launcher_rocket");
        assert!(rig.loops.sources_of(&rig.eng, &rig.native, rig.cache, proj) >= 1);
        let at = |y: f32| [Shot { id: 7, proj, pos: Vec3::new(12.0, y, 0.0), trigger: 0.0 }];
        // before it exists: silence, no voice
        let quiet = lr(&frame(&mut rig, &[]));
        assert_eq!(rig.loops.voices(), 0);
        // spawned on the LEFT: a voice within a second (the takes may still be decoding on the first frames)
        let mut first = None;
        let mut left = Vec::new();
        for i in 0..240 {
            let a = frame(&mut rig, &at(8.0));
            if first.is_none() && rig.loops.voices() > 0 {
                first = Some(i);
            }
            if i >= 150 {
                left.extend(a);
            }
        }
        let first = first.expect("the rocket's loop never started");
        assert!(first < 180, "the loop started {first} frames after the rocket appeared");
        assert_eq!(rig.loops.sounding(), 1);
        let l = lr(&left);
        // moves to the RIGHT: the same voice follows (no second voice started), the balance flips
        let started = rig.loops.started;
        let mut right = Vec::new();
        for i in 0..120 {
            let a = frame(&mut rig, &at(-8.0));
            if i >= 60 {
                right.extend(a);
            }
        }
        let r = lr(&right);
        eprintln!("quiet {quiet:?}; on the left L {:.4} R {:.4}; on the right L {:.4} R {:.4}", l.0, l.1, r.0, r.1);
        assert!(l.0 > 1.5 * l.1 && l.0 > 1.0e-4, "heard from the left while the rocket is on the left: {l:?}");
        assert!(r.1 > 1.5 * r.0 && r.1 > 1.0e-4, "heard from the right once the rocket is on the right: {r:?}");
        assert_eq!(rig.loops.started, started, "following the rocket must not restart the loop");
        // the rocket ends: the voice is stopped on that frame and the mix goes quiet
        let before = rig.loops.stopped;
        frame(&mut rig, &[]);
        assert_eq!(rig.loops.voices(), 0, "the loop must stop with the projectile");
        assert!(rig.loops.stopped > before);
        let _ = frame(&mut rig, &[]);
        let after = lr(&frame(&mut rig, &[]));
        assert!(after.0 + after.1 < 0.1 * (l.0 + l.1), "silence after impact: {after:?} vs {l:?}");
    }

    /// Past the sound's maximum distance it is silent, and a projectile that carries no loop never makes one.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_far_loop_is_silent_and_a_bullet_has_none() {
        let Some(mut rig) = rig() else { return };
        let rocket = proj_named(&rig, "rocket_launcher_rocket");
        let bullet = proj_named(&rig, "assault_rifle_bullet");
        let mut near = Vec::new();
        let mut far = Vec::new();
        for i in 0..200 {
            let a = frame(&mut rig, &[Shot { id: 1, proj: rocket, pos: Vec3::new(10.0, 0.0, 0.0), trigger: 0.0 }, Shot { id: 2, proj: bullet, pos: Vec3::new(10.0, 0.0, 0.0), trigger: 0.0 }]);
            if i >= 120 {
                near.extend(a);
            }
        }
        assert_eq!(rig.loops.sounding(), 1, "only the rocket sounds, the bullet has no loop");
        for i in 0..200 {
            let a = frame(&mut rig, &[Shot { id: 1, proj: rocket, pos: Vec3::new(5000.0, 0.0, 0.0), trigger: 0.0 }]);
            if i >= 120 {
                far.extend(a);
            }
        }
        let (n, f) = (lr(&near), lr(&far));
        eprintln!("near {n:?} far {f:?}");
        assert!(n.0 + n.1 > 1.0e-4, "liveness: the near rocket is audible");
        assert!(f.0 + f.1 < 0.02 * (n.0 + n.1), "a rocket 5000 wu away must be inaudible: {f:?} vs {n:?}");
    }

    /// #vehicle-weapons  A horn sounds while its trigger is held and stops when it is released: the
    /// `warthog_horn` weapon's `lsnd` attachment is scaled by `primary_firing`.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn a_horn_sounds_only_while_the_trigger_is_held() {
        let Some(mut rig) = rig() else { return };
        let horn = rig
            .native
            .tags_by_group(rig.cache, "weap")
            .into_iter()
            .find(|&t| rig.native.tag_name(rig.cache, t).map_or(false, |n| n.rsplit('\\').next() == Some("warthog_horn")))
            .expect("no warthog_horn weapon");
        assert!(rig.loops.sources_of(&rig.eng, &rig.native, rig.cache, horn) >= 1, "the horn weapon authors a loop");
        let at = |trigger: f32| [Shot { id: 0x8000_0001, proj: horn, pos: Vec3::new(2.0, 0.0, 0.0), trigger }];
        let mut idle = Vec::new();
        for i in 0..200 {
            let a = frame(&mut rig, &at(0.0));
            if i >= 120 {
                idle.extend(a);
            }
        }
        assert_eq!(rig.loops.voices(), 0, "a horn nobody presses makes no voice");
        let mut held = Vec::new();
        for i in 0..240 {
            let a = frame(&mut rig, &at(1.0));
            if i >= 150 {
                held.extend(a);
            }
        }
        assert!(rig.loops.voices() > 0, "pressing the horn starts its loop");
        let mut after = Vec::new();
        for i in 0..240 {
            let a = frame(&mut rig, &at(0.0));
            if i >= 200 {
                after.extend(a);
            }
        }
        let (i, h, a) = (lr(&idle), lr(&held), lr(&after));
        eprintln!("idle {i:?} held {h:?} released {a:?}");
        assert!(h.0 + h.1 > 1.0e-3, "liveness: the held horn is audible: {h:?}");
        assert!(i.0 + i.1 < 0.02 * (h.0 + h.1), "silent before: {i:?}");
        assert!(a.0 + a.1 < 0.05 * (h.0 + h.1), "silent again after release: {a:?}");
    }
}
