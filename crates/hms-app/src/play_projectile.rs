//! #projectile-sim  Test mode's PROJECTILES: every shot a weapon fires is a `proj` object that
//! flies, collides and ends, and is DRAWN through the very lane Theater draws a recorded one on.
//!
//! # What was missing
//!
//! `play_fire` decides WHEN a round leaves a barrel and `PlayMode::fire` casts one aim ray for the
//! damage seam -- so the gun clicked, the sound played, and nothing was ever launched. The owner:
//! *"we can't see the projectiles we shoot at all"*. The flight law lives in the `proj` tag and in
//! `projectiles.cpp`; neither was ported.
//!
//! # The flight law (`reach_tag_test.exe`, `omaha\items\projectiles.cpp`; all PROVEN)
//!
//! One step is `projectile_translate_with_initial_test_against_object_type` (`DD8350`), run once
//! per 60 Hz tick:
//!
//! 1. **Speed profile.** When both `initial velocity` and `final velocity` are non-zero, the speed
//!    along the projectile's forward follows `target = v0 + (vf - v0) * f` where
//!    `f = clamp((ideal_distance - lo) * inverse, 0, 1)`; `ideal_distance` integrates `target * dt`
//!    and `[lo, hi]` / `inverse` are `acceleration range` and the cache's baked
//!    `1 / (hi - lo)` (`+0x304`). The forward component is moved to the new target and clamped at
//!    `vf` (unless `allow deceleration below final velocity`).
//! 2. **Gravity.** `g = 4.1712594 * air gravity scale` (`DD5E50`; 4.1712594 is the engine's
//!    "physics constants" gravity, see `reach-engine-physics-gravity`). The stored velocity loses
//!    `g * dt` on z; the DISPLACEMENT uses the mean of the old and the profile-adjusted velocity
//!    with half a gravity step on z -- the exact integral of constant acceleration.
//! 3. **Range.** `maximum range` truncates the displacement and detonates there.
//! 4. **Timer.** `timer` (seconds -> ticks, a random point in [lo, hi]) advances `1/ticks` per tick
//!    (`DD9CC0`) and detonates at 1.0; `detonation timer starts` gates when it begins counting.
//! 5. **Contact.** The sweep ends at the first world or object surface; the projectile then ends
//!    (`action = detonate`, `reason = impact`) unless the material response bounces it.
//!
//! What a shot LOOKS like is not decided here at all: [`Field::effects`] hands the flight to
//! [`crate::film_projectile::trail_effects`] and [`crate::film_projectile::burst_effect`], the same
//! two producers `ActorDriver::film_effects` uses, and [`Field::sync_actors`] places the `proj`'s
//! render model (when it has one) on an `ActorScene` object actor exactly as Theater places a
//! recorded one. A bullet has no render model; it is drawn by its attached effect alone.
//!
//! # Not modelled (named, not hidden)
//!
//! Guided/homing projectiles (needler, Falcon missile): they need a lock-on target and test mode has
//! no target acquisition, so they fly straight. Per-material responses: the sweep returns no surface
//! material, so a hit takes the projectile's DEFAULT response. Damage: out of scope here, the damage
//! seam is `play_damage`.

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use hms_native::{CacheHandle, NativeDll};

use crate::film_projectile::{FilmEffect, ProjectileFx};
use crate::reach_projectile::{Conical, ProjectileExtras};

/// Engine ticks per second -- the rate `projectiles.cpp` integrates at.
pub const TICK_HZ: f32 = 60.0;
pub const TICK: f32 = 1.0 / TICK_HZ;

/// The engine's world gravity, wu/s^2 (`physics constants` block, `+0x00 = 0x40857AF5`).
pub const WORLD_GRAVITY: f32 = 4.171_259_4;

/// How far the crosshair ray is cast to find the point a projectile converges on, wu. Past it the
/// projectile flies parallel to the aim.
pub const CONVERGE_RANGE: f32 = 1000.0;
/// Hard cap on simultaneous projectiles, so a held fire button on a pellet weapon can never grow
/// the lane without bound.
pub const MAX_LIVE: usize = 128;
/// Model actors one `proj` may have in flight at once (an actor is a GPU mesh).
pub const MAX_ACTORS_PER_PROJ: usize = 8;
/// How long an impact / detonation burst keeps its source installed, seconds. A one-shot's
/// particles live and die inside its source, so the source has to outlast the effect.
pub const BURST_HOLD: f32 = 4.0;
/// How long a muzzle flash keeps its source installed, seconds.
pub const FLASH_HOLD: f32 = 1.5;
/// How long a dead projectile's trail source stays installed, seconds, so particles already
/// emitted can live out instead of vanishing the tick the projectile ends.
pub const TRAIL_LINGER: f32 = 0.6;

/// A projectile definition resolved once: the shared drawing summary plus the flight extras.
#[derive(Clone, Debug)]
pub struct Law {
    pub proj: u32,
    pub fx: ProjectileFx,
    pub extra: ProjectileExtras,
    /// The surface-material `default_material` effect of an impact (`foot` effect-type 0xF): what a
    /// bullet that authors no `impact effect` of its own plays when it strikes.
    pub material_impact: Option<u32>,
}

impl Law {
    pub fn read(native: &NativeDll, cache: CacheHandle, proj: u32) -> Option<Law> {
        let extra = crate::reach_projectile::projectile_extras(native, cache, proj)?;
        let fx = ProjectileFx::read(native, cache, proj, |p| {
            crate::reach_projectile::object_attachment_effects(native, cache, p).into_iter().map(|(e, _, _)| e).collect()
        })?;
        let material_impact = crate::reach_projectile::material_effect(native, cache, proj, crate::reach_projectile::MATERIAL_EFFECT_IMPACT, "");
        Some(Law { proj, fx, extra, material_impact })
    }

    /// Does this projectile draw anything at all? A `proj` with no render model and no attached
    /// effect is invisible in flight (its ending may still play an effect).
    pub fn visible_in_flight(&self) -> bool {
        self.has_model() || !self.fx.attached.is_empty()
    }

    pub fn has_model(&self) -> bool {
        self.fx.mode != 0 && self.fx.mode != 0xFFFF_FFFF
    }
}

/// The engine's `real_random` generator: `x = x * 0x19660D + 0x3C6EF35F`, value `(x >> 16) / 65535`.
#[derive(Clone, Copy, Debug)]
pub struct Rng(pub u32);

impl Rng {
    pub fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(0x0019_660D).wrapping_add(0x3C6E_F35F);
        (self.0 >> 16) as f32 * 1.525_902_2e-5
    }
}

/// What the world answered to a sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub at: Vec3,
    /// Unit surface normal facing the projectile.
    pub normal: Vec3,
    /// A placed object rather than level geometry.
    pub object: bool,
}

/// The collision world a projectile sweeps. [`SceneWorld`] is the production one; a gate supplies a
/// plane and a wall.
pub trait World {
    /// The first surface on the segment `a -> b`.
    fn sweep(&self, a: Vec3, b: Vec3) -> Option<Hit>;
}

/// The production world: the scene's BSP and the placed objects.
pub struct SceneWorld<'a>(pub &'a dyn crate::objscene::ObjectScene);

impl World for SceneWorld<'_> {
    fn sweep(&self, a: Vec3, b: Vec3) -> Option<Hit> {
        let d = b - a;
        let len = d.length();
        if len < 1.0e-6 {
            return None;
        }
        let dir = d / len;
        let mut best: Option<Hit> = None;
        if let Some((p, n)) = self.0.raycast_scene_n(a, dir) {
            if (p - a).length() <= len {
                best = Some(Hit { at: p, normal: n.normalize_or_zero(), object: false });
            }
        }
        if let Some(p) = self.0.raycast_objects_excluding(a, dir, &[]) {
            let t = (p - a).length();
            if t <= len && best.map_or(true, |h| t < (h.at - a).length()) {
                best = Some(Hit { at: p, normal: -dir, object: true });
            }
        }
        // A normal that faces away from the projectile is the back of a triangle: flip it.
        best.map(|mut h| {
            if h.normal.dot(dir) > 0.0 {
                h.normal = -h.normal;
            }
            h
        })
    }
}

/// How a projectile ended, for the effect it plays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum End {
    /// Struck a surface.
    Impact(Hit),
    /// Ran out of life in the air (timer, range, minimum velocity).
    Airborne(Vec3),
}

/// One projectile in flight.
#[derive(Clone, Debug)]
pub struct Live {
    pub id: u32,
    pub law: Arc<Law>,
    pub pos: Vec3,
    pub vel: Vec3,
    /// The projectile's forward (`object.forward`); the speed profile acts along it.
    pub fwd: Vec3,
    /// `object.up`: the roll of the model about its forward. Born as `perpendicular(forward)`
    /// (`weapon_barrel_create_projectiles` builds the placement's up with `sub_14009B390`, NOT
    /// world +Z) and carried along the flight by [`orient`].
    pub up: Vec3,
    /// `projectile + 0x228`: the integral of the profile's target speed.
    ideal: f32,
    /// `projectile + 0x224`: last tick's profile target.
    last_target: f32,
    /// `projectile + 0x208`: distance flown.
    pub travelled: f32,
    pub ticks: u32,
    /// Ticks the detonation timer runs for (0 = untimed) and how far along it is (0..1).
    timer_ticks: f32,
    timer_progress: f32,
    timer_running: bool,
    bounced: bool,
    /// Scene time it was launched and the samples of its flight, for the moving trail source.
    pub born: f32,
    pub path: Vec<(f32, Vec3)>,
    /// A tracer round draws its attached effects; the others fly unseen (`DE4290` clears the
    /// runtime tracer bit of a non-tracer round).
    pub tracer: bool,
    pub ended: Option<(End, f32)>,
}

impl Live {
    /// `projectile_new`: a projectile at `origin` flying along `dir` at its `initial velocity`.
    pub fn new(id: u32, law: &Arc<Law>, origin: Vec3, dir: Vec3, tracer: bool, now: f32, timer_ticks: f32) -> Live {
        let def = &law.fx.def;
        Live {
            id,
            law: law.clone(),
            pos: origin,
            vel: dir * def.initial_velocity,
            fwd: dir,
            up: {
                let u = perpendicular(dir).normalize_or_zero();
                if u == Vec3::ZERO { Vec3::Z } else { u }
            },
            ideal: 0.0,
            last_target: def.initial_velocity,
            travelled: 0.0,
            ticks: 0,
            timer_ticks,
            timer_progress: 0.0,
            timer_running: law.extra.timer_starts == 0,
            bounced: false,
            born: now,
            path: vec![(now, origin)],
            tracer,
            ended: None,
        }
    }
}

/// The directions one round's `proj` is launched along: the aim itself, or `conical spread`
/// pellets around it (see [`conical_directions`]).
pub fn pellet_dirs(law: &Law, aim: Vec3, rng: &mut Rng) -> Vec<Vec3> {
    let aim = aim.normalize_or_zero();
    match law.extra.conical {
        Some(c) if c.pellets() > 0 => conical_directions(aim, &c, rng),
        _ => vec![aim],
    }
}

/// `timer` is a random point in [lo, hi] seconds (`projectile-detonation-time`) turned into ticks;
/// 0 means untimed.
pub fn timer_ticks(rng: &mut Rng, t: &[f32; 2]) -> f32 {
    if t[0] <= 0.0 && t[1] <= 0.0 {
        return 0.0;
    }
    let v = if t[0] != t[1] { t[0] + rng.next() * (t[1] - t[0]) } else { t[0] };
    v * TICK_HZ
}

/// An ending effect to keep installed for [`BURST_HOLD`] seconds.
#[derive(Clone, Debug)]
pub struct Burst {
    pub id: u32,
    pub kind: u64,
    pub effe: u32,
    pub at: Vec3,
    /// The surface normal the projectile struck (world up when it ended in the air).
    pub normal: Vec3,
    pub born: f32,
    pub label: String,
}

/// One shot's launch parameters, shared by every pellet.
#[derive(Clone, Copy, Debug)]
pub struct Launch {
    pub origin: Vec3,
    pub dir: Vec3,
    pub tracer: bool,
}

/// The pellet directions of `conical` around `aim`, by the engine's stratified-jitter rule
/// (`sub_1401B5C40` / `sub_1401B6270`). One direction per pellet.
pub fn conical_directions(aim: Vec3, c: &Conical, rng: &mut Rng) -> Vec<Vec3> {
    let (na, nr) = (c.azimuth_strata.max(0) as usize, c.radial_strata.max(0) as usize);
    let max = c.max_angle.clamp(0.0, std::f32::consts::FRAC_PI_2);
    let strat = |rng: &mut Rng, i: usize, n: usize| -> f32 {
        if n == 0 { 0.0 } else { (i as f32 + rng.next()) / n as f32 }
    };
    // `sub_14009B390` builds a perpendicular to the aim.
    let right = {
        let r = aim.cross(Vec3::Z);
        if r.length_squared() < 1.0e-8 { Vec3::X } else { r.normalize() }
    };
    let up = right.cross(aim);
    let mut out = Vec::with_capacity(na * nr);
    for a in 0..na {
        for r in 0..nr {
            let theta = strat(rng, r, nr).powf(c.exponent) * max;
            let az = strat(rng, a, na) * std::f32::consts::TAU - std::f32::consts::PI;
            let radial = right * az.cos() + up * az.sin();
            out.push((aim * theta.cos() + radial * theta.sin()).normalize_or_zero());
        }
    }
    out
}

/// One pellet's whole flight, simulated once: its position after each tick and how it ended.
#[derive(Clone, Debug)]
pub struct Pellet {
    /// `path[i]` = the position `i` ticks after launch (`path[0]` is the muzzle).
    pub path: Vec<Vec3>,
    /// `orient[i]` = the `(forward, up)` the model has `i` ticks after launch.
    pub orient: Vec<(Vec3, Vec3)>,
    /// How it ended, `path.len() - 1` ticks after launch; `None` = still flying when the
    /// simulation was cut off.
    pub end: Option<End>,
}

/// A whole shot, flown ahead of time from a launch tick -- a **pure function** of its inputs, so a
/// Theater scrub back and forth lands on the same bullet every time.
#[derive(Clone, Debug)]
pub struct Flight {
    pub law: Arc<Law>,
    pub tracer: bool,
    pub pellets: Vec<Pellet>,
}

impl Flight {
    /// Fly one round from `origin` along `aim` for at most `max_ticks`. `seed` fixes the pellet
    /// pattern and the fuse, so the same shot is the same shot.
    pub fn simulate(law: &Arc<Law>, origin: Vec3, aim: Vec3, tracer: bool, seed: u32, max_ticks: u32, world: &dyn World) -> Flight {
        let mut rng = Rng(seed ^ 0x5EED_0001);
        let dirs = pellet_dirs(law, aim, &mut rng);
        let pellets = dirs
            .into_iter()
            .take(MAX_LIVE)
            .map(|d| {
                let timer = timer_ticks(&mut rng, &law.fx.def.timer);
                let mut p = Live::new(0, law, origin, d, tracer, 0.0, timer);
                let mut path = vec![origin];
                let mut orient = vec![(p.fwd, p.up)];
                let mut end = None;
                for _ in 0..max_ticks {
                    let e = step(&mut p, TICK, world);
                    path.push(p.pos);
                    orient.push((p.fwd, p.up));
                    if let Some(e) = e {
                        end = Some(e);
                        break;
                    }
                }
                Pellet { path, orient, end }
            })
            .collect();
        Flight { law: law.clone(), tracer, pellets }
    }

    /// Ticks until the last pellet ended (or the cut-off), so a caller knows when the shot is over.
    pub fn ticks(&self) -> usize {
        self.pellets.iter().map(|p| p.path.len().saturating_sub(1)).max().unwrap_or(0)
    }
}

/// Every projectile in test mode, plus the endings still playing.
#[derive(Default, Debug)]
pub struct Field {
    pub live: Vec<Live>,
    pub bursts: Vec<Burst>,
    laws: HashMap<u32, Arc<Law>>,
    next_id: u32,
    acc: f32,
    rng: Option<Rng>,
    /// Model actors per `proj`, as `(actor id, in use this frame)`.
    pool: HashMap<u32, Vec<(u32, bool)>>,
    /// Rounds launched / ended, for the caption and the gates.
    pub launched: u64,
    pub ended: u64,
    pub impacts: u64,
    /// Park position for an unused model actor.
    park: Vec3,
    /// `rounds between tracers` per `(weapon, barrel)` and the engine's per-barrel counter.
    tracer_every: HashMap<(u32, usize), i16>,
    tracer_count: HashMap<(u32, usize), u16>,
    muzzle: HashMap<(u32, usize), (Option<u32>, Vec<u32>)>,
    /// Muzzle flashes still playing: `(firing effect, gun model, gun world, scene time fired)`.
    flashes: Vec<(u32, u32, Mat4, f32)>,
}

impl Field {
    pub fn new() -> Field {
        Field { park: Vec3::new(0.0, 0.0, -9000.0), ..Default::default() }
    }

    fn rng(&mut self) -> &mut Rng {
        self.rng.get_or_insert(Rng(0x5EED_0001))
    }

    /// The resolved law of `proj`, read once.
    pub fn law(&mut self, native: &NativeDll, cache: CacheHandle, proj: u32) -> Option<Arc<Law>> {
        if let Some(l) = self.laws.get(&proj) {
            return Some(l.clone());
        }
        let l = Arc::new(Law::read(native, cache, proj)?);
        self.laws.insert(proj, l.clone());
        Some(l)
    }

    /// Install an already-built law (gates, or a caller that read it elsewhere).
    pub fn insert_law(&mut self, l: Law) -> Arc<Law> {
        let a = Arc::new(l);
        self.laws.insert(a.proj, a.clone());
        a
    }

    /// Launch the projectiles of one round: `aims` is one direction per projectile the barrel fires
    /// (`FireEvent::Shot`), each of which becomes `conical` pellets when the `proj` authors that.
    pub fn launch(&mut self, law: &Arc<Law>, aims: &[Launch], now: f32) {
        for l in aims {
            let mut rng = *self.rng();
            let dirs = pellet_dirs(law, l.dir, &mut rng);
            for d in dirs {
                if self.live.len() >= MAX_LIVE {
                    *self.rng() = rng;
                    return;
                }
                self.next_id += 1;
                let timer = timer_ticks(&mut rng, &law.fx.def.timer);
                self.live.push(Live::new(self.next_id, law, l.origin, d, l.tracer, now, timer));
                self.launched += 1;
            }
            *self.rng() = rng;
        }
    }

    /// The firing effect of `(weapon, barrel)` and the marker ids it names -- where its muzzle is.
    /// Read once per barrel.
    pub fn muzzle_info(&mut self, native: &NativeDll, cache: CacheHandle, weapon: u32, barrel: usize) -> (Option<u32>, Vec<u32>) {
        self.muzzle
            .entry((weapon, barrel))
            .or_insert_with(|| {
                let fx = crate::reach_projectile::weapon_barrels(native, cache, weapon).get(barrel).and_then(|b| b.firing_effect);
                let sids = fx.map(|e| crate::reach_projectile::effect_location_sids(native, cache, e)).unwrap_or_default();
                (fx, sids)
            })
            .clone()
    }

    /// Is the next round of `(weapon, barrel)` a tracer? The engine's rule (`DE4290`): a per-barrel
    /// counter is incremented for each round and the round is a tracer once the counter has
    /// reached `rounds between tracers`, which also resets it -- so `N` means `N` plain rounds
    /// between tracers and 0 means every round is one.
    pub fn next_tracer(&mut self, native: &NativeDll, cache: CacheHandle, weapon: u32, barrel: usize) -> bool {
        let every = *self.tracer_every.entry((weapon, barrel)).or_insert_with(|| {
            crate::reach_projectile::weapon_barrels(native, cache, weapon)
                .get(barrel)
                .map_or(0, |b| b.rounds_between_tracers)
        });
        self.count_tracer((weapon, barrel), every)
    }

    /// The counter half of [`Field::next_tracer`], separate so a gate can drive it without a map.
    pub fn count_tracer(&mut self, key: (u32, usize), every: i16) -> bool {
        let c = self.tracer_count.entry(key).or_insert(0);
        let was = *c;
        *c = c.wrapping_add(1);
        if was as i32 >= every as i32 {
            *c = 0;
            true
        } else {
            false
        }
    }

    /// Advance by `dt` real seconds in whole 60 Hz ticks. `now` is the scene time at the END of the
    /// frame; each tick is stamped at its own moment so a trail reads the same at any frame rate.
    pub fn advance(&mut self, dt: f32, now: f32, world: &dyn World) {
        self.acc += dt.clamp(0.0, 0.25);
        let n = (self.acc / TICK).floor() as u32;
        self.acc -= n as f32 * TICK;
        for k in 0..n {
            let t = now - (n - 1 - k) as f32 * TICK;
            self.tick(t, world);
        }
        self.retire(now);
    }

    /// One 60 Hz tick of every projectile.
    pub fn tick(&mut self, t: f32, world: &dyn World) {
        let mut ending: Vec<(usize, End)> = Vec::new();
        for (i, p) in self.live.iter_mut().enumerate() {
            if p.ended.is_some() {
                continue;
            }
            if let Some(e) = step(p, TICK, world) {
                ending.push((i, e));
            }
            p.ticks += 1;
            // A frame that runs several ticks stamps the first of them before the shot was born (the
            // shot is launched at the END of the frame); the flight's clock never runs backwards.
            let t = t.max(p.path.last().map_or(t, |l| l.0));
            p.path.push((t, p.pos));
        }
        for (i, e) in ending {
            let (id, law) = (self.live[i].id, self.live[i].law.clone());
            self.live[i].ended = Some((e, t));
            self.ended += 1;
            if matches!(e, End::Impact(_)) {
                self.impacts += 1;
            }
            self.push_ending(id, &law, e, t);
        }
    }

    /// Show `flight` as it is `age` ticks after it was launched at scene time `launched` (seconds),
    /// replacing nothing: the caller clears the field once per frame ([`Field::clear_frame`]) and
    /// shows every shot it wants drawn. Returns whether the shot was visible at that age.
    ///
    /// This is how Theater draws a simulated shot, and it is a PURE FUNCTION of `age`: nothing is
    /// integrated frame to frame. It builds the same [`Live`] / [`Burst`] values the test-mode
    /// integrator does, so [`Field::effects`] and [`Field::sync_actors`] draw both identically.
    pub fn show_flight(&mut self, id: u32, flight: &Flight, launched: f32, age: i64) -> bool {
        if age < 0 {
            return false;
        }
        let age = age as usize;
        let mut shown = false;
        for (k, pe) in flight.pellets.iter().enumerate() {
            let last = pe.path.len() - 1;
            let i = age.min(last);
            let dead = pe.end.is_some() && age >= last;
            let since_end = age.saturating_sub(last) as f32 * TICK;
            if dead && since_end > TRAIL_LINGER.max(BURST_HOLD) {
                continue;
            }
            let pid = id.wrapping_mul(131).wrapping_add(k as u32);
            let t_of = |j: usize| launched + j as f32 * TICK;
            let pos = pe.path[i];
            let prev = pe.path[i.saturating_sub(1)];
            let mut l = Live::new(pid, &flight.law, pe.path[0], Vec3::X, flight.tracer, launched, 0.0);
            l.pos = pos;
            l.vel = if (pos - prev).length_squared() > 1.0e-12 { (pos - prev) / TICK } else { flight.law.fx.def.initial_velocity * Vec3::X };
            (l.fwd, l.up) = pe.orient[i];
            l.ticks = i as u32;
            l.path = (0..=i).map(|j| (t_of(j), pe.path[j])).collect();
            if dead {
                l.ended = Some((pe.end.unwrap(), t_of(last)));
                if since_end <= BURST_HOLD {
                    self.push_ending(pid, &flight.law, pe.end.unwrap(), t_of(last));
                }
            }
            // a trail lingers only a moment; a burst runs on in `bursts`
            if !dead || since_end <= TRAIL_LINGER {
                self.live.push(l);
            }
            shown = true;
        }
        shown
    }

    /// The effects a projectile's ending plays, as [`Burst`]s: the impact effect for a surface, and
    /// the detonation effect (airborne or ground by how it ended).
    fn push_ending(&mut self, id: u32, law: &Law, e: End, t: f32) {
        let (at, grounded, normal) = match e {
            End::Impact(h) => (h.at, true, h.normal),
            End::Airborne(p) => (p, false, Vec3::Z),
        };
        let name = &law.fx.name;
        if let End::Impact(h) = e {
            let authored = if h.object { law.fx.def.object_impact_effect } else { law.fx.def.impact_effect };
            let eff = authored.or(law.material_impact);
            if let Some(effe) = eff {
                self.bursts.push(Burst { id, kind: crate::film_projectile::SEED_IMPACT, effe, at, normal, born: t, label: format!("impact #{id:#x} {name}") });
            }
        }
        if let Some(effe) = law.fx.def.detonation_effect(!grounded) {
            self.bursts.push(Burst { id, kind: crate::film_projectile::SEED_DETONATION, effe, at, normal, born: t, label: format!("detonation #{id:#x} {name}") });
        }
    }

    /// Empty the field's lanes, keeping its pooled actors, so a caller that re-shows its shots
    /// every frame ([`Field::show_flight`]) starts clean.
    pub fn clear_frame(&mut self) {
        self.live.clear();
        self.bursts.clear();
    }

    /// Drop what has finished lingering.
    fn retire(&mut self, now: f32) {
        self.live.retain(|p| p.ended.map_or(true, |(_, t)| now - t <= TRAIL_LINGER));
        self.bursts.retain(|b| now - b.born <= BURST_HOLD);
        self.flashes.retain(|f| now - f.3 <= FLASH_HOLD);
    }

    /// Play the weapon's firing effect (the muzzle flash) from `gun` as a one-shot born at `now`.
    pub fn flash(&mut self, effe: u32, mode: u32, gun: Mat4, now: f32) {
        self.flashes.push((effe, mode, gun, now));
    }

    /// Every effect the lane should be playing: the trail of each tracer in flight (and just
    /// ended), and the impact / detonation bursts still running.
    pub fn effects(&self) -> Vec<FilmEffect> {
        let mut out = Vec::new();
        for p in &self.live {
            if !p.tracer {
                continue;
            }
            let path = Arc::new(p.path.clone());
            out.extend(crate::film_projectile::trail_effects(p.id, &p.law.fx, world_of(p), p.born, path));
        }
        for b in &self.bursts {
            out.push(crate::film_projectile::burst_effect(b.kind, b.id, b.effe, b.at, Some(b.normal), b.born, b.label.clone()));
        }
        for (i, &(effe, mode, gun, born)) in self.flashes.iter().enumerate() {
            // the same producer shape Theater's muzzle flash uses: the gun's own model, so the
            // effect's `primary_trigger` / `muzzle_flash` locations land on the gun
            out.push(crate::film_projectile::FilmEffect {
                seed: crate::film_projectile::FILM_FX_SEED ^ 0x01 ^ ((i as u64) << 24) ^ ((born.to_bits() as u64) << 4),
                effe,
                mode,
                world: gun,
                birth: born,
                path: None,
                label: format!("muzzle flash {i}"),
            });
        }
        out
    }

    /// The light volumes of the projectiles in flight: the same producer Theater uses
    /// ([`crate::film_projectile::trail_beams`]). A tracer's volumes only, like its trail.
    pub fn beams(&self) -> Vec<crate::film_projectile::FilmBeam> {
        let mut out = Vec::new();
        for p in &self.live {
            if p.tracer {
                out.extend(crate::film_projectile::trail_beams(&p.law.fx, world_of(p)));
            }
        }
        out
    }

    /// The contrails (tracers) of the projectiles in flight, laid on each flight so far.
    pub fn contrails(&self) -> Vec<crate::film_projectile::FilmContrail> {
        let mut out = Vec::new();
        for p in &self.live {
            if p.tracer {
                out.extend(crate::film_projectile::trail_contrails(&p.law.fx, world_of(p), p.born, Arc::new(p.path.clone())));
            }
        }
        out
    }

    /// Place the render model of each projectile that has one, on object actors drawn through the
    /// same actor lane Theater uses. Unused pooled actors are parked.
    pub fn sync_actors(&mut self, native: &NativeDll, cache: CacheHandle, actors: &mut crate::reach_actor::ActorScene) {
        for v in self.pool.values_mut() {
            for e in v.iter_mut() {
                e.1 = false;
            }
        }
        let mut want: Vec<(u32, Mat4)> = Vec::new();
        for p in &self.live {
            if p.ended.is_some() || !p.law.has_model() {
                continue;
            }
            want.push((p.law.proj, world_of(p)));
        }
        for (proj, m) in want {
            let pool = self.pool.entry(proj).or_default();
            let slot = match pool.iter().position(|e| !e.1) {
                Some(i) => Some(i),
                None if pool.len() < MAX_ACTORS_PER_PROJ => match actors.add_object(native, cache, proj, self.park, 0.0) {
                    Ok(id) => {
                        pool.push((id, false));
                        Some(pool.len() - 1)
                    }
                    Err(_) => None,
                },
                None => None,
            };
            if let Some(i) = slot {
                pool[i].1 = true;
                if let Some(a) = actors.get_mut(pool[i].0) {
                    a.pos = m.w_axis.truncate();
                    a.world_override = Some(m);
                }
            }
        }
        for v in self.pool.values() {
            for &(id, used) in v {
                if !used {
                    if let Some(a) = actors.get_mut(id) {
                        a.pos = self.park;
                        a.world_override = None;
                    }
                }
            }
        }
    }

    /// The actor ids of every pooled projectile model (for a gate that hides them).
    pub fn pool_ids(&self) -> Vec<u32> {
        self.pool.values().flatten().map(|e| e.0).collect()
    }

    /// Leave test mode: forget every projectile and take the pooled model actors out of the scene.
    pub fn release(&mut self, actors: &mut crate::reach_actor::ActorScene) {
        let ids: Vec<u32> = self.pool.values().flatten().map(|e| e.0).collect();
        if !ids.is_empty() {
            actors.actors.retain(|a| !ids.contains(&a.id));
            actors.rebuild = true;
        }
        self.live.clear();
        self.bursts.clear();
        self.flashes.clear();
        self.pool.clear();
        self.acc = 0.0;
    }

    pub fn caption(&self) -> String {
        format!(
            "projectiles: {} in flight, {} launched, {} ended ({} on a surface), {} effect(s) playing",
            self.live.iter().filter(|p| p.ended.is_none()).count(),
            self.launched,
            self.ended,
            self.impacts,
            self.bursts.len()
        )
    }
}

/// Where a shot leaves the gun, and the gun itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Muzzle {
    /// The weapon's `primary_trigger` in the world.
    pub at: Vec3,
    /// The weapon's model -> world matrix this frame, which a muzzle flash is played in.
    pub gun: Mat4,
    /// The weapon's render model (the flash's marker lookups resolve against it).
    pub mode: u32,
}

/// Where a shot leaves the gun: the weapon's `primary_trigger` (the marker its firing effect names)
/// in the world, from the first-person weapon actor's current pose. `None` when the weapon, its
/// model or the marker is not there, and the caller then fires from the eye.
pub fn muzzle_world(
    sc: &crate::scene::SceneController,
    actors: &crate::reach_actor::ActorScene,
    fp: &[u32],
    muzzle_sids: &[u32],
) -> Option<Muzzle> {
    for &id in fp {
        let Some(a) = actors.get(id) else { continue };
        let Some(w) = a.weapon.as_ref() else { continue };
        for &sid in muzzle_sids {
            if let Some((p, _)) = sc.model_marker(w.mode_tag, sid) {
                let gun = a.world() * w.world(Mat4::IDENTITY, &a.pose);
                let at = gun.transform_point3(p);
                if at.is_finite() {
                    return Some(Muzzle { at, gun, mode: w.mode_tag });
                }
            }
        }
    }
    None
}

/// The projectile's world frame: forward along its velocity (the engine orients a projectile along
/// its flight), up toward +Z.
pub fn world_of(p: &Live) -> Mat4 {
    crate::film_objective::object_basis(p.pos, Some((p.fwd.to_array(), p.up.to_array()))).unwrap_or(Mat4::from_translation(p.pos))
}

/// `sub_14009B390`: a vector perpendicular to `f`, built from its two SMALLEST components (the
/// engine's `perpendicular3d`). It is the `up` a projectile is created with, so a rocket's fins
/// start at a roll that depends on which way it was aimed, not at world up.
pub fn perpendicular(f: Vec3) -> Vec3 {
    let (ax, ay, az) = (f.x.abs(), f.y.abs(), f.z.abs());
    if ay < ax || az < ax {
        if az < ay { Vec3::new(f.y, -f.x, 0.0) } else { Vec3::new(-f.z, 0.0, f.x) }
    } else {
        Vec3::new(0.0, f.z, -f.y)
    }
}

/// Re-orient a projectile after a tick of flight (`projectile_translate...` `DD8350`, the block
/// after the sweep): with `oriented along velocity` (proj flag bit 0) the forward becomes the new
/// velocity direction and the up is the OLD up with its forward component removed, so the roll is
/// carried and not reset; without the flag the orientation stays as launched. (The engine also rolls
/// the up about the forward at the object's angular velocity; that is zero for a projectile that
/// has not been knocked, so no spin is modelled.)
pub fn orient(p: &mut Live) {
    if p.law.extra.flags & crate::reach_projectile::PROJ_FLAG_ORIENTED_ALONG_VELOCITY == 0 {
        return;
    }
    let v = p.vel;
    if v.length_squared() < 1.0e-12 {
        return;
    }
    let f = v.normalize();
    let mut u = p.up - f * p.up.dot(f);
    if u.length_squared() < 1.0e-8 {
        u = perpendicular(f);
    }
    p.fwd = f;
    p.up = u.normalize_or_zero();
}

/// One projectile, one tick. `Some` when it ends.
pub fn step(p: &mut Live, dt: f32, world: &dyn World) -> Option<End> {
    let law = p.law.clone();
    let def = &law.fx.def;
    let ex = &law.extra;

    // ---- 1. the speed profile -------------------------------------------------------------
    let mut adjusted = p.vel;
    if def.initial_velocity != 0.0 && def.final_velocity != 0.0 {
        let (v0, vf) = (def.initial_velocity, def.final_velocity);
        let lo = def.acceleration_range[0];
        let inv = if ex.acceleration_inverse != 0.0 {
            ex.acceleration_inverse
        } else if def.acceleration_range[1] > lo {
            1.0 / (def.acceleration_range[1] - lo)
        } else {
            0.0
        };
        let f = ((p.ideal - lo) * inv).clamp(0.0, 1.0);
        let target = v0 + (vf - v0) * f;
        let speed = p.vel.length();
        let along = p.vel.dot(p.fwd);
        if ex.flags & crate::reach_projectile::PROJ_FLAG_ALLOW_DECELERATION_BELOW_FINAL == 0 || target <= speed {
            let mut want = target - p.last_target + along;
            // `v47 <= v46`: a decelerating profile floors at the final velocity, an accelerating
            // one caps at it.
            want = if vf <= v0 { want.max(vf) } else { want.min(vf) };
            adjusted = p.vel + p.fwd * (want - along);
            p.last_target = target;
            p.ideal += target * dt;
        }
    }

    // ---- 2. gravity and the displacement --------------------------------------------------
    let g = WORLD_GRAVITY * def.air_gravity_scale;
    let mut disp_vel = (adjusted + p.vel) * 0.5;
    disp_vel.z -= g * 0.5 * dt;
    let mut new_vel = adjusted;
    new_vel.z -= g * dt;
    let mut disp = disp_vel * dt;

    // ---- 3. the maximum range -------------------------------------------------------------
    let mut ends_here = false;
    let len = disp.length();
    if def.maximum_range > 0.0 && p.travelled + len > def.maximum_range {
        let keep = if len > 0.0 { ((def.maximum_range - p.travelled) / len).max(0.0) } else { 0.0 };
        disp *= keep;
        ends_here = true;
    }

    // ---- 5. contact ------------------------------------------------------------------------
    let from = p.pos;
    let to = from + disp;
    if let Some(h) = world.sweep(from, to) {
        p.pos = h.at + h.normal * 0.001;
        p.travelled += (h.at - from).length();
        return Some(End::Impact(h));
    }
    p.pos = to;
    p.travelled += disp.length();
    p.vel = new_vel;
    orient(p);
    if ends_here {
        return Some(End::Airborne(p.pos));
    }

    // ---- 4. the timer ----------------------------------------------------------------------
    if p.timer_running && p.timer_ticks >= 1.0 {
        p.timer_progress += 1.0 / p.timer_ticks;
        if p.timer_progress >= 1.0 {
            return Some(End::Airborne(p.pos));
        }
    }
    // `minimum velocity`: detonates when slowed below it.
    if def.minimum_velocity > 0.0 && p.vel.length() < def.minimum_velocity {
        return Some(End::Airborne(p.pos));
    }
    None
}

#[cfg(test)]
mod tests {
    //! The flight law, gated against numbers worked out BY HAND from the tag values (the literals
    //! below), never against the code's own formula: a gate that recomputes `0.5 * g * t * t`
    //! from `WORLD_GRAVITY` passes for any gravity.
    use super::*;
    use crate::reach_projectile::ProjectileDef;

    fn law(def: ProjectileDef) -> Law {
        Law {
            proj: 1,
            fx: ProjectileFx { def, mode: 0, attached: vec![], beams: vec![], contrails: vec![], detonation: None, name: "synthetic".into() },
            extra: ProjectileExtras::default(),
            material_impact: None,
        }
    }

    struct Empty;
    impl World for Empty {
        fn sweep(&self, _: Vec3, _: Vec3) -> Option<Hit> {
            None
        }
    }

    /// A wall at x = `x` facing -x.
    struct Wall(f32);
    impl World for Wall {
        fn sweep(&self, a: Vec3, b: Vec3) -> Option<Hit> {
            if a.x < self.0 && b.x >= self.0 {
                let t = (self.0 - a.x) / (b.x - a.x);
                return Some(Hit { at: a + (b - a) * t, normal: Vec3::NEG_X, object: false });
            }
            None
        }
    }

    /// Fly one projectile launched along +x from the origin for up to `max` ticks; returns the
    /// field after the flight so a gate can read where and how it ended.
    fn fly(l: Law, max: u32, world: &dyn World) -> (Field, u32) {
        let mut f = Field::new();
        let l = f.insert_law(l);
        f.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
        let mut n = 0;
        for k in 0..max {
            f.tick(k as f32 * TICK, world);
            n = k + 1;
            if f.live[0].ended.is_some() {
                break;
            }
        }
        (f, n)
    }

    fn def() -> ProjectileDef {
        ProjectileDef { initial_velocity: 100.0, final_velocity: 100.0, ..Default::default() }
    }

    /// 100 wu/s for one second with no gravity, no profile: exactly 100 wu on. The literal is the
    /// definition of a speed.
    #[test]
    fn a_constant_speed_projectile_covers_speed_times_time() {
        let (f, _) = fly(law(def()), 60, &Empty);
        let p = &f.live[0];
        assert_eq!(p.ticks, 60, "liveness: the projectile was stepped 60 times");
        assert!((p.pos.x - 100.0).abs() < 1.0e-3, "x after 1 s at 100 wu/s is {}", p.pos.x);
        assert!(p.pos.y.abs() < 1.0e-6 && p.pos.z.abs() < 1.0e-6, "flat flight stayed flat: {:?}", p.pos);
    }

    /// Gravity scale 1.0 from rest-vertical: z after one second is -g/2 = -2.0856297 (g = the
    /// engine's 4.1712594). 0.5 * 4.1712594 worked by hand; the constant is not read from the code.
    #[test]
    fn gravity_is_the_engines_constant_times_the_authored_scale() {
        let (f, _) = fly(law(ProjectileDef { air_gravity_scale: 1.0, ..def() }), 60, &Empty);
        assert!((f.live[0].pos.z + 2.085_629_7).abs() < 1.0e-3, "z after 1 s of full gravity is {}", f.live[0].pos.z);
        // and the scale scales it: 0.02 (the rocket's) is fifty times less
        let (f, _) = fly(law(ProjectileDef { air_gravity_scale: 0.02, ..def() }), 60, &Empty);
        assert!((f.live[0].pos.z + 0.041_712_594).abs() < 1.0e-4, "rocket-scale drop is {}", f.live[0].pos.z);
    }

    /// The rocket's real numbers: 8 -> 16 wu/s over a 0..30 wu acceleration range. The speed obeys
    /// ds/dt = s * (8/30) so s(t) = 8 e^(8t/30) and the 30 wu mark (speed 16) is reached at
    /// t = (30/8) ln 2 = 2.599 s = 155.9 ticks. A constant 8 wu/s would take 225 ticks and a
    /// constant 16 wu/s would take 112.
    #[test]
    fn the_speed_profile_accelerates_over_the_acceleration_range() {
        let mut l = law(ProjectileDef { initial_velocity: 8.0, final_velocity: 16.0, acceleration_range: [0.0, 30.0], ..Default::default() });
        l.extra.acceleration_inverse = 1.0 / 30.0;
        let mut f = Field::new();
        let l = f.insert_law(l);
        f.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
        let mut ticks = 0u32;
        while f.live[0].pos.x < 30.0 && ticks < 400 {
            f.tick(ticks as f32 * TICK, &Empty);
            ticks += 1;
        }
        let f2 = f;
        assert!((150..=162).contains(&ticks), "30 wu took {ticks} ticks; the profile predicts 156 (225 = no acceleration, 112 = no start-up)");
        let speed = f2.live[0].vel.length();
        assert!(speed > 15.0 && speed <= 16.0 + 1.0e-3, "speed at 30 wu is {speed}, expected the final 16");
    }

    /// `maximum range` truncates the flight at exactly that distance and ends it in the air.
    #[test]
    fn the_maximum_range_ends_the_flight_at_that_distance() {
        let (f, n) = fly(law(ProjectileDef { maximum_range: 50.0, ..def() }), 600, &Empty);
        let p = &f.live[0];
        assert!(matches!(p.ended, Some((End::Airborne(_), _))), "ended as {:?}", p.ended);
        assert!((p.pos.x - 50.0).abs() < 1.0e-3, "stopped at x = {}", p.pos.x);
        assert_eq!(n, 30, "50 wu at 100 wu/s is half a second = 30 ticks");
    }

    /// `timer` 1.0 s = 60 ticks.
    #[test]
    fn the_detonation_timer_counts_seconds_in_ticks() {
        let (f, n) = fly(law(ProjectileDef { timer: [1.0, 1.0], ..def() }), 600, &Empty);
        assert!(matches!(f.live[0].ended, Some((End::Airborne(_), _))), "ended as {:?}", f.live[0].ended);
        assert!((59..=61).contains(&n), "a 1 s fuse ended after {n} ticks");
        // and with no fuse and no range it does not end by itself
        let (f, n) = fly(law(def()), 120, &Empty);
        assert!(f.live[0].ended.is_none() && n == 120, "an untimed projectile must keep flying");
    }

    /// A wall at x = 10: the projectile ends ON it, and plays the `impact effect` there (when the
    /// tag authors one) and nothing when it does not.
    #[test]
    fn a_surface_ends_the_flight_and_plays_the_impact_effect_on_it() {
        let mut d = def();
        d.impact_effect = Some(7);
        let (f, _) = fly(law(d), 60, &Wall(10.0));
        let p = &f.live[0];
        let Some((End::Impact(h), _)) = p.ended else { panic!("did not impact: {:?}", p.ended) };
        assert!((h.at.x - 10.0).abs() < 1.0e-3, "struck at x = {}", h.at.x);
        assert_eq!(f.impacts, 1);
        assert_eq!(f.bursts.len(), 1, "one impact effect expected");
        assert_eq!(f.bursts[0].effe, 7);
        assert!((f.bursts[0].at.x - 10.0).abs() < 1.0e-3, "the effect is on the wall: {:?}", f.bursts[0].at);
        // no authored impact effect, no invented one
        let (f, _) = fly(law(def()), 60, &Wall(10.0));
        assert!(f.live[0].ended.is_some() && f.bursts.is_empty(), "a projectile with no effects plays none");
    }

    /// A detonation effect is played where it ended, airborne vs ground by how it ended.
    #[test]
    fn a_detonation_plays_the_variant_for_how_it_ended() {
        let d = ProjectileDef { detonation_airborne: Some(11), detonation_ground: Some(22), timer: [0.25, 0.25], ..def() };
        let (f, _) = fly(law(d), 600, &Empty);
        assert_eq!(f.bursts.iter().map(|b| b.effe).collect::<Vec<_>>(), vec![11], "a fuse in the air is the airborne detonation");
        let (f, _) = fly(law(d), 600, &Wall(5.0));
        assert_eq!(f.bursts.iter().map(|b| b.effe).collect::<Vec<_>>(), vec![22], "a hit on a surface is the ground detonation");
    }

    /// `rounds between tracers`: N plain rounds, then a tracer; 0 = every round.
    #[test]
    fn rounds_between_tracers_counts_plain_rounds() {
        let mut f = Field::new();
        let three: Vec<bool> = (0..9).map(|_| f.count_tracer((1, 0), 3)).collect();
        assert_eq!(three, [false, false, false, true, false, false, false, true, false], "3 plain rounds between tracers");
        let zero: Vec<bool> = (0..4).map(|_| f.count_tracer((2, 0), 0)).collect();
        assert_eq!(zero, [true; 4], "0 means every round is a tracer");
    }

    /// Pellets: `azimuth x radial` strata give that many directions, every one inside the cone and
    /// not all the same, and the same seed gives the same spread.
    #[test]
    fn a_conical_spread_launches_its_pellet_count_inside_the_cone() {
        let c = Conical { azimuth_strata: 4, radial_strata: 3, exponent: 1.0, max_angle: 0.1 };
        assert_eq!(c.pellets(), 12);
        let aim = Vec3::new(1.0, 0.0, 0.0);
        let d = conical_directions(aim, &c, &mut Rng(1));
        assert_eq!(d.len(), 12);
        let mut distinct = 0;
        for v in &d {
            let ang = v.dot(aim).clamp(-1.0, 1.0).acos();
            assert!(ang <= 0.1 + 1.0e-4, "a pellet left the cone: {ang}");
            assert!((v.length() - 1.0).abs() < 1.0e-4);
            distinct += usize::from((*v - aim).length() > 1.0e-5);
        }
        assert!(distinct >= 10, "the pellets are all on the aim: {distinct} of 12 deviate");
        assert_eq!(conical_directions(aim, &c, &mut Rng(1)), d, "same seed, same pattern");
        // the outer ring of the highest radial stratum reaches toward the edge of the cone
        let widest = d.iter().map(|v| v.dot(aim).clamp(-1.0, 1.0).acos()).fold(0.0f32, f32::max);
        assert!(widest > 0.05, "the spread never leaves the inner half of the cone: {widest}");
    }

    /// A launch with `conical` pellets creates one projectile per pellet; without it, one.
    #[test]
    fn launch_expands_a_conical_projectile_into_pellets() {
        let mut f = Field::new();
        let mut l = law(def());
        l.extra.conical = Some(Conical { azimuth_strata: 3, radial_strata: 2, exponent: 1.0, max_angle: 0.05 });
        let l = f.insert_law(l);
        f.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
        assert_eq!(f.live.len(), 6);
        let mut g = Field::new();
        let l = g.insert_law(law(def()));
        g.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
        assert_eq!(g.live.len(), 1);
    }

    /// The tick count over a second is the same at 30, 60 and 144 frames per second.
    #[test]
    fn the_flight_does_not_depend_on_the_frame_rate() {
        let at = |fps: f32| {
            let mut f = Field::new();
            let l = f.insert_law(law(ProjectileDef { air_gravity_scale: 1.0, ..def() }));
            f.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
            let n = fps as u32;
            for k in 0..n {
                f.advance(1.0 / fps, (k + 1) as f32 / fps, &Empty);
            }
            (f.live[0].pos, f.live[0].ticks)
        };
        for fps in [30.0, 60.0, 144.0] {
            let (p, ticks) = at(fps);
            assert!((59..=60).contains(&ticks), "liveness: {ticks} ticks in one second at {fps} fps");
            // the position is a function of the TICK COUNT alone: 100 wu/s * ticks / 60
            assert!((p.x - ticks as f32 * 100.0 / 60.0).abs() < 1.0e-3, "x {} after {ticks} ticks at {fps} fps", p.x);
        }
    }

    /// A tracer draws its trail through the SHARED producer; a plain round draws none.
    #[test]
    fn a_tracer_draws_its_trail_through_the_shared_producer() {
        let mut d = def();
        d.timer = [1.0, 1.0];
        let mut l = law(d);
        l.fx.attached = vec![0x42];
        let mut f = Field::new();
        let l = f.insert_law(l);
        f.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }, Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: false }], 1.5);
        f.tick(1.5, &Empty);
        let fx = f.effects();
        assert_eq!(fx.len(), 1, "only the tracer round has a trail; got {}", fx.len());
        let want = crate::film_projectile::trail_effects(f.live[0].id, &l.fx, world_of(&f.live[0]), 1.5, Arc::new(f.live[0].path.clone()));
        assert_eq!(fx, want, "test mode and Theater must build a trail through the same function");
        assert_eq!(fx[0].effe, 0x42);
        assert!(fx[0].path.as_ref().map_or(false, |p| p.len() >= 2), "a moving source needs its flight so far");
    }

    /// The Theater integrator (`Flight::simulate`, used for the rounds a film never replicates) and
    /// test mode's per-tick `Field` must be ONE law: fly the same shot through both and compare
    /// every tick. A gravity-and-profile projectile is used so a difference in any term shows.
    #[test]
    fn theaters_flights_and_test_modes_integrator_are_one_law() {
        let mut d = ProjectileDef { initial_velocity: 8.0, final_velocity: 16.0, acceleration_range: [0.0, 30.0], air_gravity_scale: 0.9, ..Default::default() };
        d.timer = [0.0, 0.0];
        let mut l = law(d);
        l.extra.acceleration_inverse = 1.0 / 30.0;
        let mut f = Field::new();
        let l = f.insert_law(l);
        let dir = Vec3::new(1.0, 0.3, 0.4).normalize();
        f.launch(&l, &[Launch { origin: Vec3::new(1.0, 2.0, 3.0), dir, tracer: true }], 0.0);
        let fl = Flight::simulate(&l, Vec3::new(1.0, 2.0, 3.0), dir, true, 7, 120, &Empty);
        assert_eq!(fl.pellets.len(), 1);
        assert_eq!(fl.pellets[0].path.len(), 121, "liveness: the flight ran its 120 ticks");
        for k in 0..120usize {
            f.tick(k as f32 * TICK, &Empty);
            let (a, b) = (f.live[0].pos, fl.pellets[0].path[k + 1]);
            assert!((a - b).length() < 1.0e-4, "tick {}: the integrator is at {a:?}, the flight at {b:?}", k + 1);
        }
        let mut flat = law(ProjectileDef { air_gravity_scale: 0.0, ..d });
        flat.extra.acceleration_inverse = 1.0 / 30.0;
        let flat = Arc::new(flat);
        let nf = Flight::simulate(&flat, Vec3::new(1.0, 2.0, 3.0), dir, true, 7, 120, &Empty);
        // gravity 0.9 * 4.1712594 over 2 s = 0.5 * 3.754 * 4 = 7.5 wu of fall, by hand
        let fell = nf.pellets[0].path[120].z - fl.pellets[0].path[120].z;
        assert!((fell - 7.508).abs() < 0.2, "two seconds of 0.9 gravity should drop it 7.51 wu, it dropped {fell}");
    }

    /// A flight shown at an age is a pure function of that age: scrub forward, back, forward again
    /// and the same bullet is in the same place. Empty before it is fired and long after it ends.
    #[test]
    fn a_shown_flight_is_a_pure_function_of_its_age() {
        let mut d = def();
        d.maximum_range = 20.0; // ends at tick 12
        d.detonation_airborne = Some(5);
        let mut l = law(d);
        l.fx.attached = vec![1];
        let mut f = Field::new();
        let l = f.insert_law(l);
        let fl = Flight::simulate(&l, Vec3::ZERO, Vec3::X, true, 1, 100, &Empty);
        let shown = |f: &mut Field, age: i64| {
            f.clear_frame();
            let vis = f.show_flight(77, &fl, 2.0, age);
            (vis, f.live.iter().map(|p| p.pos).collect::<Vec<_>>(), f.effects().len())
        };
        let a1 = shown(&mut f, 5);
        let _ = shown(&mut f, 40);
        let a2 = shown(&mut f, 5);
        assert!(a1.0 && !a1.1.is_empty(), "liveness: the shot is visible 5 ticks after firing");
        assert_eq!(a1, a2, "scrubbing away and back changed the picture");
        assert!((a1.1[0].x - 5.0 * 100.0 / 60.0).abs() < 1.0e-3, "5 ticks at 100 wu/s is 8.33 wu, got {}", a1.1[0].x);
        assert!(!shown(&mut f, -1).0, "a shot that has not been fired is not drawn");
        let at_end = shown(&mut f, 12 + 6);
        assert!(at_end.0 && f.bursts.iter().any(|b| b.effe == 5), "the airborne detonation plays after the range ends the flight");
        assert!(!shown(&mut f, 12 + 60 * 5).0, "long after the ending nothing is left");
    }

    /// With no `impact effect` of its own a projectile plays its material's default row; its own wins
    /// when authored; an object hit uses `object impact effect` before the material.
    #[test]
    fn an_impact_without_an_authored_effect_plays_the_materials_default() {
        let mut l = law(def());
        l.material_impact = Some(9);
        let (f, _) = fly(l.clone(), 60, &Wall(10.0));
        assert_eq!(f.bursts.iter().map(|b| b.effe).collect::<Vec<_>>(), vec![9], "the material default plays");
        let mut own = l.clone();
        own.fx.def.impact_effect = Some(7);
        let (f, _) = fly(own, 60, &Wall(10.0));
        assert_eq!(f.bursts.iter().map(|b| b.effe).collect::<Vec<_>>(), vec![7], "an authored impact effect wins");
        let (f, _) = fly(law(def()), 60, &Wall(10.0));
        assert!(f.bursts.is_empty(), "no material default, no authored effect: nothing is invented");
    }

    /// The muzzle flash is a one-shot on the GUN's model, born when fired, gone after its hold.
    #[test]
    fn a_muzzle_flash_is_a_one_shot_on_the_guns_model() {
        let mut f = Field::new();
        let gun = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        f.flash(0xAB, 0x77, gun, 4.0);
        let fx = f.effects();
        assert_eq!(fx.len(), 1);
        assert_eq!((fx[0].effe, fx[0].mode, fx[0].world, fx[0].birth), (0xAB, 0x77, gun, 4.0));
        assert!(fx[0].path.is_none(), "a flash is not a moving source");
        f.retire(4.0 + FLASH_HOLD + 0.1);
        assert!(f.effects().is_empty(), "the flash is dropped once it has played");
    }

    /// Light volumes and contrails come from the same producers Theater uses, only for tracers.
    #[test]
    fn beams_and_contrails_come_from_the_shared_producers_for_tracers_only() {
        let mut l = law(def());
        l.fx.beams = vec![(10, 20)];
        l.fx.contrails = vec![(30, 40)];
        let mut f = Field::new();
        let l = f.insert_law(l);
        f.launch(&l, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }, Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: false }], 1.0);
        f.tick(1.0, &Empty);
        let b = f.beams();
        let c = f.contrails();
        assert_eq!(b.len(), 1, "only the tracer has a light volume");
        assert_eq!(c.len(), 1, "only the tracer has a contrail");
        assert_eq!(b, crate::film_projectile::trail_beams(&l.fx, world_of(&f.live[0])), "Theater and test mode build a beam through one function");
        assert_eq!((c[0].cntl, c[0].marker_sid), (30, 40));
        assert!(c[0].path.len() >= 2 && c[0].birth == 1.0, "a contrail is laid on the flight so far");
    }

    // ----- wiring: the call sites that put the lanes on screen -------------------------------------
    //
    // A projectile that is simulated, decoded and gated but never HANDED to the renderer is exactly
    // the failure this module exists to end, so the call sites are gated too. The needles are
    // literals of the production files, and these tests live in THIS file, so they cannot find
    // themselves.

    fn between<'a>(src: &'a str, start: &str, end: &str) -> &'a str {
        let a = src.find(start).unwrap_or_else(|| panic!("`{start}` must exist; if it was renamed, update this gate"));
        let rest = &src[a..];
        let b = rest[1..].find(end).map(|i| i + 1).unwrap_or(rest.len());
        &rest[..b]
    }

    /// The window: a Shot launches projectiles, the field is advanced each frame, and what it holds
    /// is handed to the actor pool, the particle lane, the light-volume lane and the contrail lane.
    #[test]
    fn the_window_launches_advances_and_draws_every_projectile() {
        let main = include_str!("main.rs");
        let up = between(main, "    fn update_play(&mut self, ctx: &egui::Context, response: &egui::Response) {", "\n    fn ");
        assert!(up.contains("pm.launch_projectiles(scene, native, cache, barrel, &projectiles, muzzle, now)"), "a Shot must put its projectiles in the air");
        assert!(up.contains("pm.projectiles.advance(dt, now, &play_projectile::SceneWorld(scene))"), "the field must be flown every frame");
        let ua = between(main, "    fn update_actors(&mut self) {", "\n    fn object_identity");
        for needle in [
            "pm.projectiles.sync_actors(",
            "want_fx.extend(pm.projectiles.effects())",
            "want_beams.extend(pm.projectiles.beams())",
            "want_trails.extend(pm.projectiles.contrails())",
            "film_projectile::install(sc, &want_fx)",
            "film_projectile::install_beams(sc, &want_beams)",
            "film_projectile::install_contrails(sc, want_trails)",
            "self.film_driver.synth_shots(",
        ] {
            assert!(ua.contains(needle), "update_actors lost `{needle}`");
        }
        let leave = between(main, "    fn leave_play(&mut self) -> String {", "\n    ///");
        assert!(leave.contains("pm.projectiles.release("), "leaving test mode must take the projectile actors out of the scene");
        let render = main.find("build_film_beam_meshes(time,").expect("the render path must build the light-volume ribbons with the scene clock");
        assert!(main[render..(render + 400).min(main.len())].contains("set_beam_meshes(bm)"), "and hand them to the renderer");
        assert!(main.contains("self.renderer.set_beam_meshes(Vec::new())"), "the lane is dropped when particles are switched off");
    }

    /// The scripted host (what a headless capture and the Theater gates run) installs the same lanes.
    #[test]
    fn the_script_host_draws_the_same_lanes() {
        let sh = include_str!("script_host.rs");
        for needle in [
            "self.film_driver.synth_shots(&pbc, Some(&world), &mut self.actors, sc.native_dll(), cache)",
            "crate::film_projectile::install_beams(sc, &self.film_driver.beams)",
            "crate::film_projectile::install_contrails(sc, self.film_driver.contrails.clone())",
            "sc.build_film_beam_meshes(anim_t,",
            "self.renderer.set_beam_meshes(bm)",
        ] {
            assert!(sh.contains(needle), "script_host lost `{needle}`");
        }
    }

    /// Theater builds a recorded projectile's trail, light volume and contrail through the SAME three
    /// producers test mode uses, and the classification that once threw projectiles away stays fixed.
    #[test]
    fn theater_uses_the_shared_producers_and_keeps_projectiles() {
        let fp = include_str!("film_play.rs");
        let fe = between(fp, "    pub fn film_effects(", "\n    /// #projectile-sim  Fly and show");
        for needle in ["film_projectile::trail_effects(", "film_projectile::trail_beams(", "film_projectile::trail_contrails(", "self.shots.effects()", "self.shots.beams()", "self.shots.contrails()"] {
            assert!(fe.contains(needle), "film_effects lost `{needle}`");
        }
        let fo = include_str!("film_objective.rs");
        assert!(fo.contains("\"proj\" => Draw::Projectile,"), "a recorded projectile must classify as Draw::Projectile (the catch-all once discarded them)");
    }

    /// `sub_14009B390` by hand: the vector is built from the two smallest components. Literals worked
    /// out from the decompile (`if |y|<|x| or |z|<|x| { if |z|<|y| (y,-x,0) else (-z,0,x) } else (0,z,-y)`).
    #[test]
    fn perpendicular_is_the_engines_routine() {
        let near = |a: Vec3, b: Vec3| (a - b).length() < 1.0e-6;
        assert!(near(perpendicular(Vec3::X), Vec3::Z), "forward +x -> up +z");
        assert!(near(perpendicular(Vec3::Y), Vec3::NEG_Z), "forward +y -> up -z (the engine rolls it over)");
        assert!(near(perpendicular(Vec3::Z), Vec3::Y), "forward +z -> up +y");
        assert!(near(perpendicular(Vec3::new(0.6, 0.8, 0.0)), Vec3::new(0.8, -0.6, 0.0)));
        assert!(near(perpendicular(Vec3::new(0.6, 0.0, 0.8)), Vec3::new(-0.8, 0.0, 0.6)));
        for f in [Vec3::new(0.3, -0.5, 0.8), Vec3::new(-0.9, 0.1, 0.2), Vec3::new(0.1, 0.1, -0.99)] {
            let f = f.normalize();
            assert!(perpendicular(f).dot(f).abs() < 1.0e-5, "perpendicular to {f:?}");
        }
    }

    /// A projectile that is `oriented along velocity` turns with its velocity and carries its roll; one
    /// that is not keeps its launch orientation however it falls.
    #[test]
    fn the_model_follows_the_velocity_only_when_the_tag_says_so() {
        let dir = Vec3::new(0.6, 0.8, 0.0);
        let run = |flags: u32| {
            let mut l = law(ProjectileDef { initial_velocity: 12.0, final_velocity: 12.0, air_gravity_scale: 1.0, ..Default::default() });
            l.extra.flags = flags;
            let mut f = Field::new();
            let l = f.insert_law(l);
            f.launch(&l, &[Launch { origin: Vec3::ZERO, dir, tracer: true }], 0.0);
            let (f0, u0) = (f.live[0].fwd, f.live[0].up);
            for k in 0..90 {
                f.tick(k as f32 * TICK, &Empty);
            }
            (f0, u0, f.live[0].fwd, f.live[0].up, f.live[0].vel)
        };
        let (f0, u0, f1, u1, v) = run(crate::reach_projectile::PROJ_FLAG_ORIENTED_ALONG_VELOCITY);
        assert!((u0 - Vec3::new(0.8, -0.6, 0.0)).length() < 1.0e-5, "born with the perpendicular up, got {u0:?}");
        assert!((f1 - v.normalize()).length() < 1.0e-5, "forward must be the velocity direction");
        assert!(f1.dot(f0) < 0.97, "after 1.5 s of fall the nose has dropped: {f1:?} vs {f0:?}");
        assert!(f1.dot(u1).abs() < 1.0e-4 && (u1.length() - 1.0).abs() < 1.0e-4, "up stays a unit vector perpendicular to forward");
        assert!(u1.dot(u0) > 0.8, "the roll is carried, not reset to world up: up {u1:?} from {u0:?}");
        let (f0, _, f1, _, _) = run(0);
        assert_eq!(f0, f1, "without the flag the orientation stays as launched");
    }

    /// An impact / detonation effect plays AT the contact point, with its forward along the surface normal;
    /// an effect that ended in the air plays pointing up.
    #[test]
    fn an_ending_effect_plays_at_the_contact_facing_the_surface() {
        let mut d = def();
        d.impact_effect = Some(7);
        d.detonation_ground = Some(8);
        let (f, _) = fly(law(d), 60, &Wall(10.0));
        assert_eq!(f.bursts.len(), 2, "an impact effect and a detonation");
        for b in &f.bursts {
            assert!((b.at.x - 10.0).abs() < 1.0e-3, "{}: at {:?}", b.label, b.at);
            assert_eq!(b.normal, Vec3::NEG_X, "{}: the wall faces -x", b.label);
        }
        let fx = f.effects();
        assert_eq!(fx.len(), 2);
        for e in &fx {
            assert!((e.world.x_axis.truncate() - Vec3::NEG_X).length() < 1.0e-5, "the effect's forward is the surface normal: {:?}", e.world);
            assert!((e.world.w_axis.x - 10.0).abs() < 1.0e-3, "and it is placed on the wall");
        }
        let mut air = def();
        air.timer = [0.2, 0.2];
        air.detonation_airborne = Some(9);
        let (f, _) = fly(law(air), 60, &Empty);
        assert_eq!(f.bursts.len(), 1);
        assert_eq!(f.bursts[0].normal, Vec3::Z, "an air burst points up");
    }

    /// REAL TAGS: every family below ends on a wall with its AUTHORED effect, at the contact, facing the
    /// surface -- bullets and bolts with their material impact, explosives with their detonation.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn each_projectile_family_plays_its_authored_effect_at_the_contact() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        let native = &*native;
        let find = |leaf: &str| native.tags_by_group(cache, "proj").into_iter().find(|&t| native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf))).unwrap_or_else(|| panic!("no {leaf}"));
        // (projectile, plays a detonation, plays a surface impact)
        let families: [(&str, bool, bool); 7] = [
            ("rocket_launcher_rocket", true, true),
            ("concussion_rifle_grenade", true, true),
            ("flak_bolt", true, true),
            ("assault_rifle_bullet", false, true),
            ("sniper_rifle_bullet", false, true),
            ("plasma_rifle_bolt", false, true),
            ("needler_shard", false, true),
        ];
        let mut checked = 0;
        for (leaf, det, imp) in families {
            let proj = find(leaf);
            let law = Arc::new(Law::read(native, cache, proj).expect("readable"));
            let mut f = Field::new();
            f.launch(&law, &[Launch { origin: Vec3::ZERO, dir: Vec3::X, tracer: true }], 0.0);
            for k in 0..3000u32 {
                f.tick(k as f32 * TICK, &Wall(6.0));
                if f.live[0].ended.is_some() {
                    break;
                }
            }
            assert!(matches!(f.live[0].ended, Some((End::Impact(_), _))), "{leaf}: did not end on the wall: {:?}", f.live[0].ended);
            let want_det = law.fx.def.detonation_effect(false);
            let want_imp = law.fx.def.impact_effect.or(law.material_impact);
            assert_eq!(det, want_det.is_some(), "{leaf}: expectation table is stale (detonation {want_det:?})");
            assert_eq!(imp, want_imp.is_some(), "{leaf}: expectation table is stale (impact {want_imp:?})");
            let got: Vec<u32> = f.bursts.iter().map(|b| b.effe).collect();
            let mut want: Vec<u32> = want_imp.into_iter().chain(want_det).collect();
            want.sort_unstable();
            let mut g = got.clone();
            g.sort_unstable();
            assert_eq!(g, want, "{leaf}: played {got:?}");
            assert!(!f.bursts.is_empty(), "{leaf}: liveness -- a hit played nothing");
            for b in &f.bursts {
                assert!((b.at.x - 6.0).abs() < 0.01 && b.normal == Vec3::NEG_X, "{leaf}: {} at {:?} facing {:?}", b.label, b.at, b.normal);
            }
            checked += 1;
        }
        assert_eq!(checked, 7);
    }
}
