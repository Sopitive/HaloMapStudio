//! #film-objects  The OBJECTIVE objects and the loose weapons a saved film carries.
//!
//! The report that started this was "I am looking at the castle wars clip, and I see no flag at
//! all". The flag is in the recording; it was never drawn. `Playback::wanted` keeps only
//! `entity_type == BIPED_TYPE`, and vehicles escape that only because they were given a path of
//! their own ([`crate::film_vehicle`]). Everything else — the flag, its base, the hill beacons,
//! every dropped weapon — was dropped one step before the scene.
//!
//! So this module answers three separate questions, each with its own measurement:
//!
//! 1. **What is in the recording?** Every creation message opens with a 14-bit network tag
//!    reference naming the object DEFINITION ([`crate::film_action::creation_def_wire`]), and
//!    resolving it for every track rather than only the bipeds' names the whole cast. On the
//!    Castle Wars clip that is 1,562 non-biped tracks over 123 distinct definitions. See
//!    [`ObjectTable::build`] and the inventory in `film_action::objective_inventory`.
//! 2. **Which of them deserve drawing?** 1,562 tracks is not a licence to draw everything, and
//!    the measured `moves()` share decides it rather than taste. [`Draw`] is that decision in
//!    one place, with the numbers.
//! 3. **Where does one go, and how is it turned?** A `weap` is an ordinary object, so it is
//!    placed exactly as a vehicle is: [`crate::film_vehicle::vehicle_basis`] over the recorded
//!    `(forward, up)` pair, slerped between samples, falling back to a yaw-only placement when
//!    the recording has not oriented it yet. Nothing here invents a position.
//!
//! Everything is a pure function of the tick, so scrubbing backwards reproduces the frame.
//!
//! # What the recording DOES and DOES NOT carry, measured
//!
//! Both from [`probes::probe_objective_motion_and_carry`] and
//! [`probes::probe_flag_track_lifetimes`] on the Castle Wars clip.
//!
//! **Position: dense, and for a carried flag it does not stop.** 1,112 of this clip's 1,612
//! non-biped, non-vehicle tracks report a position at all (the ~500 that do not are the `eqip`
//! loadout tokens), giving 59,849 position samples. The flag exists as a CHAIN of 33 short
//! tracks: each is ended by a kind-4 destroy and the next begins 0 to 283 ticks later, at most
//! **two alive at once** — one per team. And each track keeps reporting its position for its
//! whole life while a player carries it: on 17 of the 31 real tracks **every single recorded
//! position** is within 1.5 wu of a biped. **So the carriage is RECORDED, not derived** — see
//! [`CARRY_RADIUS_WU`] — and the objective is drawn where the film puts it rather than being
//! attached to a hand marker. Recorded beats derived, and here recorded is simply available.
//!
//! **Orientation: dense for everything EXCEPT the objective.** 51,978 of 59,849 position samples
//! have a basis beside them (86.8%), and 96.6% of those bases are more than 5 degrees off level —
//! this stream is full of tumbling debris, so a yaw-only placement would be wrong nearly always.
//! Per definition: `pickup_garbage_tire` 99.9% as many bases as positions, `hubcap` 101.8%,
//! `cowl_right` 107.1%, `destroyed_gastank` 141.8% (turned without moving), `energy_sword` 86.1%,
//! `gravity_hammer` 93.0% — but **`flag` gets exactly 2 orientation samples per track.** The
//! premise that a flag reports a full basis as often as a vehicle does is false on this clip: a
//! drawn flag's basis is its creation orientation, slerped between its two samples and then held.
//! That is stated rather than papered over, and the placement code needs no special case for it —
//! two samples is still a recorded basis, and a track with none falls back to the yaw.

use glam::Mat4;

/// How far a tracked object may sit from the Forge spawn point its own definition is placed on
/// before the stream has to draw it — the same rule, and the same constant, the vehicle lane
/// uses ([`crate::film_anim::VehicleTable::SPAWN_COINCIDENT_WU`]).
///
/// The film's embedded map variant is drawn in full, weapons included. A weapon that nobody ever
/// picked up matches its own spawn point to a hundredth of a world unit, so drawing the stream's
/// copy as well puts two identical models in the same place and they z-fight. While it sits
/// there the Forge scene's copy is the right thing to show; the moment it moves, the stream's is.
pub const SPAWN_COINCIDENT_WU: f32 = crate::film_anim::VehicleTable::SPAWN_COINCIDENT_WU;

/// What the draw filter does with one recorded object, once its definition has resolved.
///
/// ## The measurement this decision is made from
///
/// Reproduced on the Castle Wars clip (`*53C14E9D*`, 8:58) with
/// `film_action::objective_inventory::print_non_biped_inventory`, which resolves every track's
/// own 14-bit definition reference. **1,562 non-biped tracks over 123 distinct definitions**
/// (plus 50 whose reference did not resolve, all on entity types 1/3/4/5/8/9/10, none of which
/// is an object type). By resolved group:
///
/// | group | tracks | ever moved | verdict |
/// |---|---|---|---|
/// | `weap` | 447 | **378 (84.6%)** | [`Objective`](Draw::Objective) / [`LooseWeapon`](Draw::LooseWeapon) |
/// | `bloc` | 630 | 95 (15.1%) | [`AlreadyInScene`](Draw::AlreadyInScene), or [`Debris`](Draw::Debris) where it moves |
/// | `eqip` | 392 | **0** | [`NoWorldPresence`](Draw::NoWorldPresence) |
/// | `scen` | 22 | 0 | [`AlreadyInScene`](Draw::AlreadyInScene) |
/// | `ctrl` | 13 | 0 | [`AlreadyInScene`](Draw::AlreadyInScene) |
/// | `vehi` | 55 | 33 | its own lane ([`crate::film_vehicle`]) |
/// | `effe` / `jmad` / `snd!` | 3 | 0 | [`NoWorldPresence`](Draw::NoWorldPresence) — not objects at all |
///
/// and the rows that matter, by definition:
///
/// | definition | tracks / of which ever moved |
/// |---|---|
/// | **`weap flag`** | **33 / 28** |
/// | `weap energy_sword` | 407 / 345 |
/// | `weap gravity_hammer` | 7 / 5 |
/// | `bloc pickup_garbage_tire` | 24 / 24 |
/// | `bloc wheel` | 18 / 18 |
/// | `bloc pickup_garbage_hubcap` | 17 / 17 |
/// | `bloc mp_flag_base` | 2 / 0 |
/// | `bloc mp_hill_beacon` | 5 / 0 |
/// | `eqip sprint` | 390 / 0 |
/// | `ctrl health_station` | 13 / 0 |
///
/// **So the split is the move share, and the move share is bimodal.** The `weap` group is 84.6%
/// motion; `eqip`, `scen` and `ctrl` are 0.0% — not one of their 427 tracks ever reports a second
/// position, which is exactly what an object the Forge scene already draws, or a token with no
/// world presence, looks like. `bloc` is the one group with both halves in it: 535 of its 630
/// tracks never move (Forge pieces, which the scene draws) and **95 do** — and those 95 are all
/// physics debris, every one of them either a thrown garbage pickup (`pickup_garbage_tire` 24/24,
/// `pickup_garbage_hubcap` 17/17, `pickup_garbage_bumper` 6/6) or a part knocked off a vehicle
/// (`wheel` 18/18, `hubcap` 13/11, `cowl_left`/`cowl_right`/`handlebar`/`fender_*`/`hull_back`).
/// That is why [`Draw::Debris`] exists: the group cannot be judged, only the track can.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draw {
    /// An OBJECTIVE object: the flag, the ball, the bomb, the skull, the carried cores. Drawn
    /// from the stream, and the only kind that gets a waypoint.
    Objective,
    /// A loose or dropped sandbox weapon. Drawn from the stream — unless it never moved and is
    /// still sitting on its own Forge spawn point, which the scene already draws.
    LooseWeapon,
    /// A `bloc` the recording actually MOVES: physics debris — a garbage tire a player threw, a
    /// wheel knocked off a Warthog. Drawn from the stream, with the same spawn-coincident skip.
    Debris,
    /// #projectile  A PROJECTILE the recording replicated: a grenade, a rocket, a missile, a
    /// needle, a Falcon or turret bullet. Drawn from the stream, on the path the recording states.
    ///
    /// These used to fall into [`Draw::AlreadyInScene`] through the catch-all arm, whose reason
    /// ("the film's own map variant already draws this") is simply untrue of a projectile: a
    /// grenade is not a Forge placement and nothing else was drawing it. Measured on the corpus
    /// in `reach_projectile::film_probes` — `proj` tracks are entity type 16 and the clips carry
    /// hundreds of them.
    Projectile,
    /// Not drawn: the film's own embedded map variant already draws this object, at the authored
    /// placement the recording never moves it off. A `bloc mp_flag_base`, a `ctrl
    /// health_station`, a `scen kill_volume`, a Forge platform. Drawing it again would double
    /// every one of them up and z-fight — and the kill volumes would become visible boxes.
    AlreadyInScene,
    /// Not drawn: no world presence. `eqip sprint` is an armour ability GRANTED to a player by
    /// their loadout, not a pickup lying on the floor — 390 of them exist on the Castle Wars clip
    /// and **none** ever reports a second position. The three `effe` / `jmad` / `snd!` references
    /// land here too: those groups are not objects, so there is nothing to place.
    NoWorldPresence,
}

impl Draw {
    /// Is this kind drawn from the stream at all?
    pub fn drawn(self) -> bool {
        matches!(self, Draw::Objective | Draw::LooseWeapon | Draw::Debris | Draw::Projectile)
    }
    /// Index into [`ObjectTable::by_draw`].
    pub fn bucket(self) -> usize {
        match self {
            Draw::Objective => 0,
            Draw::LooseWeapon => 1,
            Draw::Debris => 2,
            Draw::AlreadyInScene => 3,
            Draw::NoWorldPresence => 4,
            Draw::Projectile => 5,
        }
    }
    /// The one-word name for the caption and the script host.
    pub fn label(self) -> &'static str {
        match self {
            Draw::Objective => "objective",
            Draw::LooseWeapon => "weapon",
            Draw::Debris => "debris",
            Draw::AlreadyInScene => "in-scene",
            Draw::NoWorldPresence => "token",
            Draw::Projectile => "projectile",
        }
    }
}

/// The `weap` definitions that are OBJECTIVE objects rather than sandbox weapons, by tag leaf.
///
/// Reach carries every one of these in the `weap` group, because the engine makes a carried
/// objective a weapon: that is how a player holds it, how it is dropped on death and why it has
/// its own chud reticule. The list is the shipped multiplayer objective set.
///
/// It is a NAME list because the alternative does not separate them. `weap` +0x3AC `weapon class`
/// (see [`crate::reach_actor::weapon_class`], pinned in
/// `re-notes/reach_saved_film_layout.md` #wclass) puts `flag` on its own but lumps
/// `skull` / `assault_bomb` / `dlc_covenant_bomb` in with class `ball`, and puts
/// `cov_power_module` in class `turret` alongside the real `machinegun_turret` and
/// `plasma_turret`. So the class corroborates this list — every leaf here reads `flag`, `ball` or
/// `turret` and no sandbox weapon reads `flag` or `ball` — but it cannot replace it.
pub const OBJECTIVE_LEAVES: &[&str] = &[
    "flag",
    "ball",
    "bomb",
    "skull",
    "assault_bomb",
    "dlc_covenant_bomb",
    "cov_power_module",
    "unsc_data_core",
];

/// Which objective leaf, if any, this tag path names. Case-insensitive on the leaf only, so
/// `objects\multi\flag\flag` and `objects\weapons\multiplayer\flag\flag` both resolve.
pub fn objective_leaf(tag_path: &str) -> Option<&'static str> {
    let leaf = tag_path.rsplit('\\').next().unwrap_or(tag_path);
    OBJECTIVE_LEAVES.iter().copied().find(|o| leaf.eq_ignore_ascii_case(o))
}

/// Classify one resolved definition. `group` is the 4-char tag group, `tag_path` the full path,
/// and `moves` is [`crate::film_decode::EntityTrack::moves`] — whether the recording ever moved
/// this particular track more than half a world unit.
///
/// Group-driven rather than entity-type driven, deliberately: the diagnosis that found the flag
/// resolved every track's own definition, and the resolved group is the thing that says what an
/// object IS. (The entity type says how its message is shaped; `weap` is type 17 and `eqip`
/// type 12 on the measured clip, but `bloc` and `scen` share type 13.) See [`Draw`] for the
/// measured share behind every arm.
pub fn classify(group: &str, tag_path: &str, moves: bool) -> Draw {
    match group {
        "weap" => {
            if objective_leaf(tag_path).is_some() {
                Draw::Objective
            } else {
                Draw::LooseWeapon
            }
        }
        // #projectile  A projectile the recording replicated. NOT in the scene already, and not
        // a token: it has a position every tick it is alive.
        "proj" => Draw::Projectile,
        // An armour ability token. See `Draw::NoWorldPresence` for the 392/0 measurement; a
        // genuinely dropped ability would be a world pickup, and the corpus has none, so none is
        // drawn rather than guessing which of the 392 might be one.
        "eqip" => Draw::NoWorldPresence,
        // Not objects at all: the reference landed in another span of the nine-group network tag
        // table (3 tracks on the measured clip, all on non-object entity types).
        "effe" | "jmad" | "snd!" | "lsnd" | "jpt!" => Draw::NoWorldPresence,
        // The one group that has to be judged per TRACK rather than per definition: 535 of the
        // clip's 630 `bloc` tracks are Forge pieces the scene draws and never move, and 95 are
        // physics debris that does.
        "bloc" if moves => Draw::Debris,
        // Everything else the recording names is a placed Forge object the scene already draws.
        _ => Draw::AlreadyInScene,
    }
}

/// One recorded object, identified.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectIdent {
    pub net_id: u32,
    /// The engine entity type its messages carry, kept for the diagnostic line.
    pub entity_type: u8,
    /// The resolved definition's cache tag id — what [`crate::reach_actor::ActorScene::add_object`]
    /// takes.
    pub obj_tag: u32,
    /// The resolved definition's 4-char group.
    pub group: String,
    /// The definition's tag leaf (`flag`, `energy_sword`), for the UI and the gates.
    pub name: String,
    pub draw: Draw,
    /// `weap` +0x3AC `weapon class`, when the tag carries one. Corroborates [`Draw::Objective`]
    /// and drives nothing.
    pub class: Option<String>,
    /// The nearest Forge spawn point of this same definition in the film's own map variant, and
    /// how far the object's FIRST recorded position is from it. `None` when the variant places no
    /// copy of this definition at all — which is the flag's case in every objective gametype,
    /// because the gametype spawns it rather than Forge placing it.
    pub spawn: Option<([f32; 3], f32)>,
}

impl ObjectIdent {
    /// Is this object still sitting on the Forge spawn point the scene already draws it on?
    ///
    /// `false` whenever the variant places no copy of this definition at all, which is the flag's
    /// case in every objective gametype: the gametype spawns it, Forge does not place it, so
    /// there is nothing for the stream's copy to double up with and it is always drawn.
    pub fn on_its_spawn(&self, pos: [f32; 3]) -> bool {
        match self.spawn {
            Some((sp, _)) => {
                let d = (pos[0] - sp[0]).powi(2) + (pos[1] - sp[1]).powi(2) + (pos[2] - sp[2]).powi(2);
                d.sqrt() <= SPAWN_COINCIDENT_WU
            }
            None => false,
        }
    }
}

/// Every non-biped, non-vehicle object a film recorded, identified through its own definition
/// reference, with the ones worth drawing marked.
#[derive(Clone, Debug, Default)]
pub struct ObjectTable {
    /// By net id. Only entries whose definition RESOLVED are here.
    pub objects: std::collections::HashMap<u32, ObjectIdent>,
    /// Tracks whose 14-bit definition reference did not resolve to a tag in the open cache.
    pub unresolved: usize,
    /// Tracks skipped because they are a biped or a vehicle — those have their own lanes.
    pub other_lane: usize,
    /// Counts by verdict, indexed by [`Draw::bucket`]: objective, loose weapon, debris, already
    /// in the scene, no world presence.
    pub by_draw: [usize; 6],
}

impl ObjectTable {
    /// Build it. `resolve_def` turns a track's 14-bit definition wire value into
    /// `(tag id, group, tag path)` — the caller supplies it because it needs the loaded cache's
    /// tag table AND the `.map` file's per-tag datum salts (see [`crate::film_action::RefTable`]).
    /// `spawns` is every present object in the film's own map variant as `(resolved tag id,
    /// position)`, so a never-touched weapon or a static Forge block can be left to the scene.
    pub fn build(
        idx: &crate::film_decode::FilmIndex,
        mut resolve_def: impl FnMut(u32) -> Option<(u32, String, String)>,
        mut weapon_class: impl FnMut(u32) -> Option<String>,
        spawns: &[(u32, [f32; 3])],
    ) -> ObjectTable {
        let mut t = ObjectTable::default();
        for tr in &idx.tracks {
            if tr.entity_type == crate::film_play::BIPED_TYPE || tr.entity_type == crate::film_anim::VEHICLE_TYPE {
                t.other_lane += 1;
                continue;
            }
            let Some((obj_tag, group, path)) = tr.def_wire.and_then(&mut resolve_def) else {
                t.unresolved += 1;
                continue;
            };
            let draw = classify(&group, &path, tr.moves());
            t.by_draw[draw.bucket()] += 1;
            if !draw.drawn() {
                continue;
            }
            // The nearest placed copy of THIS SAME definition, so a dropped magnum is never
            // matched against some other weapon's spawn point.
            let spawn = tr.positions.first().and_then(|&(_, first)| {
                spawns
                    .iter()
                    .filter(|(tag, _)| *tag == obj_tag)
                    .map(|(_, p)| {
                        let d = (p[0] - first[0]).powi(2) + (p[1] - first[1]).powi(2) + (p[2] - first[2]).powi(2);
                        (*p, d.sqrt())
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            t.objects.insert(
                tr.net_id,
                ObjectIdent {
                    net_id: tr.net_id,
                    entity_type: tr.entity_type,
                    obj_tag,
                    group,
                    name: leaf,
                    draw,
                    class: weapon_class(obj_tag),
                    spawn,
                },
            );
        }
        t
    }

    /// Objects alive at `tick`, with the recorded position and the recorded basis.
    ///
    /// A **pure function of the tick**: both samples interpolate between the bracketing keys and
    /// neither carries state, so scrubbing to the same moment gives the same answer.
    pub fn alive_at<'a>(
        &'a self,
        idx: &'a crate::film_decode::FilmIndex,
        tick: u32,
    ) -> Vec<(&'a ObjectIdent, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        let mut out = Vec::new();
        for tr in &idx.tracks {
            let Some(o) = self.objects.get(&tr.net_id) else { continue };
            if !tr.alive_at(tick) {
                continue;
            }
            let Some(p) = tr.sample(tick) else { continue };
            out.push((o, p, tr.orientation_at(tick)));
        }
        out
    }

    /// Objects alive at `tick` that the STREAM has to draw: everything except an object still
    /// lying on the spawn point the Forge scene is already drawing it on.
    ///
    /// Ordered **objectives first, then by net id** — not by the hash map's walk order, which
    /// differs run to run. That matters twice: it makes the list a pure function of the tick, and
    /// it is what [`Self::to_draw_capped`] truncates, so the objective can never be the thing a
    /// cap drops.
    pub fn to_draw_at<'a>(
        &'a self,
        idx: &'a crate::film_decode::FilmIndex,
        tick: u32,
    ) -> Vec<(&'a ObjectIdent, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        let mut v: Vec<_> =
            self.alive_at(idx, tick).into_iter().filter(|(o, pos, _)| !o.on_its_spawn(*pos)).collect();
        v.sort_by_key(|(o, _, _)| (o.draw != Draw::Objective, o.net_id));
        v
    }

    /// As [`Self::to_draw_at`], truncated to `cap` objects.
    ///
    /// An actor is a GPU mesh build, so the pool is finite: a long clip accumulates far more
    /// recorded weapons than are ever alive at one moment (447 `weap` tracks over nine minutes on
    /// the Castle Wars clip), and a cap is what stops a pathological moment from costing the
    /// frame. Because the order is objectives-first, the flag survives every cap.
    pub fn to_draw_capped<'a>(
        &'a self,
        idx: &'a crate::film_decode::FilmIndex,
        tick: u32,
        cap: usize,
    ) -> Vec<(&'a ObjectIdent, [f32; 3], Option<([f32; 3], [f32; 3])>)> {
        let mut v = self.to_draw_at(idx, tick);
        v.truncate(cap);
        v
    }

    /// How many drawable objects are parked on their own spawn point at `tick`, so the caption
    /// can account for the difference between [`Self::alive_at`] and [`Self::to_draw_at`].
    pub fn parked_on_spawn_at(&self, idx: &crate::film_decode::FilmIndex, tick: u32) -> usize {
        self.alive_at(idx, tick).iter().filter(|(o, pos, _)| o.on_its_spawn(*pos)).count()
    }

    /// The objective objects only, for the waypoint pass.
    /// The resolved identity of one net id, if the table has it.
    ///
    /// #reach-sound needs this to answer "which `weap` is attached to this
    /// biped right now?" — the audio engine walks the attachment graph and
    /// then asks the table what each attached object IS.
    pub fn ident(&self, net_id: u32) -> Option<&ObjectIdent> {
        self.objects.get(&net_id)
    }

    pub fn objectives(&self) -> impl Iterator<Item = &ObjectIdent> {
        self.objects.values().filter(|o| o.draw == Draw::Objective)
    }
}

/// The placement matrix for one recorded object: the engine's `(forward, up)` basis when the
/// recording carries one, otherwise `None` so the caller keeps a yaw-only placement rather than
/// inventing a level one.
///
/// This is deliberately [`crate::film_vehicle::vehicle_basis`] and not a second implementation:
/// the rule is `Matrix_BuildOrientation`, the same one `scene::object_matrix` places every Forge
/// object with, and a flag is placed by it exactly as a Warthog is.
pub fn object_basis(pos: glam::Vec3, orient: Option<([f32; 3], [f32; 3])>) -> Option<Mat4> {
    crate::film_vehicle::vehicle_basis(pos, orient)
}

// ---------------------------------------------------------------------------------------------
// the carrier
// ---------------------------------------------------------------------------------------------

/// How close a biped's recorded position has to be to an objective's for the objective to be
/// taken as CARRIED by it, world units.
///
/// **Measured, not guessed** ([`probes::probe_flag_track_lifetimes`] on the Castle Wars clip).
/// The two populations are cleanly separated and 1.5 wu is where the gap is:
///
/// * of the 31 flag tracks that live past the opening tick, **17 have EVERY ONE of their recorded
///   positions inside 1.5 wu of a biped** (76 of 76, 80 of 80, 67 of 67, …) with a closest
///   approach of 0.10–0.58 wu — a flag being carried for the whole of that track's life;
/// * the tracks that were dropped and left lying come no closer than **6.62 and 7.83 wu**.
///
/// It is NOT tighter than that, and the reason is the two anchors: a biped's recorded position is
/// its FEET and the flag is held at hand height out in front, so a carried flag sits about
/// 0.5–1.5 wu from its carrier's recorded origin. An earlier 0.5 wu constant caught only 13 of
/// those 76 positions and would have reported the flag as dropped while a player ran it home.
pub const CARRY_RADIUS_WU: f32 = 1.5;

/// How an objective's carrier was established — **stated by the recording, or derived by HMS**.
///
/// The distinction is reported rather than hidden, because the two are not equally good:
/// `biped_pickup` (list B kind 1 event type 9) names the biped outright, but the Castle Wars clip
/// carries only **6** of them against 33 flag tracks — and no `weapon_pickup`, `weapon_drop`,
/// `weapon_put_away` or `weapon_trade` event at all — so most carriage is never stated as an
/// event. **What IS always recorded is the flag's own position** (see the module docs), so the
/// DRAWN placement never depends on this; only the waypoint's colour and the "carried by" line do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carry {
    /// A `biped_pickup` / `weapon_pickup` event named this biped and this object.
    Stated,
    /// No event; the objective's own recorded position is inside [`CARRY_RADIUS_WU`] of this
    /// biped's recorded position at this tick.
    ///
    /// **It can be wrong, and in one measured way it is.** A player standing on his own flag's
    /// stand is inside the radius too, so a flag that has not moved all game gets reported as
    /// carried by whoever walks up to it — seen on the Castle Wars clip at 416.7 s. The cost is
    /// bounded: nothing about where the flag is DRAWN depends on this, only the waypoint's colour
    /// and the "carried by" line, and both say `derived` rather than claiming the recording said
    /// it. Narrowing the radius is not the fix — the measurement that set it is in
    /// [`CARRY_RADIUS_WU`] — and a stated `biped_drop` would be, if a film carried one.
    Derived,
}

/// Who is carrying `net_id` at `tick`, and how that was established.
///
/// `bipeds` is `(net id, position)` for every biped drawn this tick — the caller already has the
/// frame, so this takes it rather than walking the index again.
///
/// **The DRAW position is never derived from this.** The objective is drawn at its own recorded
/// position, because the recording keeps reporting it while it is carried (see
/// [`probes::probe_objective_motion_and_carry`]); the carrier is resolved only to colour the
/// waypoint and to name who has it. Recorded beats derived, and the recorded position is right
/// there.
pub fn carrier_at(
    idx: &crate::film_decode::FilmIndex,
    net_id: u32,
    tick: u32,
    pos: [f32; 3],
    bipeds: &[(u32, [f32; 3])],
) -> Option<(u32, Carry)> {
    // The recording's own say first: the last pickup of THIS object at or before now.
    let stated = idx
        .events
        .iter()
        .filter(|e| e.tick <= tick)
        .filter(|e| matches!(e.kind, crate::film_action::EV_PICKUP | EV_WEAPON_PICKUP))
        .filter(|e| e.other == Some(net_id) || e.subject == Some(net_id))
        .max_by_key(|e| e.tick)
        .and_then(|e| {
            // One reference is the unit and the other the item; the unit is the one that is a
            // biped this tick.
            [e.subject, e.other]
                .into_iter()
                .flatten()
                .find(|n| *n != net_id && bipeds.iter().any(|(b, _)| b == n))
        });
    if let Some(b) = stated {
        // Only believe a stated pickup while the two are actually together: the recording states
        // the pickup but not every drop, so a stale one would leave the flag's waypoint on a
        // player who no longer has it.
        if let Some(&(_, bp)) = bipeds.iter().find(|(n, _)| *n == b) {
            if dist(pos, bp) <= CARRY_RADIUS_WU {
                return Some((b, Carry::Stated));
            }
        }
    }
    bipeds
        .iter()
        .map(|&(n, p)| (n, dist(pos, p)))
        .filter(|&(_, d)| d <= CARRY_RADIUS_WU)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(n, _)| (n, Carry::Derived))
}

/// `weapon_pickup`, list B kind 1 event type 32. Not in
/// [`crate::film_action`]'s named constants yet, and that file belongs to another lane, so the
/// value is taken from the event table in `re-notes/reach_saved_film_layout.md` (#film-action).
pub const EV_WEAPON_PICKUP: u8 = 32;

fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

// ---------------------------------------------------------------------------------------------
// the waypoint
// ---------------------------------------------------------------------------------------------

/// How far above an objective object's recorded origin its waypoint floats, world units.
///
/// **HMS's own constant, not the engine's, and that is stated rather than implied.** A player's
/// floating name is anchored on a decoded engine chain — the `head` marker (`string_id` 0xC2)
/// plus the literal 0.075 lift out of `haloreach.dll+0x2CB76C`. An OBJECTIVE waypoint is not
/// made by that producer: in Reach it is a Megalo script action that inserts a chud navpoint for
/// an object, and the anchor that path resolves has not been read out of the binary. So this is a
/// height chosen to clear the flag's own model rather than an engine value, and the waypoint is
/// labelled derived in the UI for that reason.
pub const OBJECTIVE_WAYPOINT_UP_WU: f32 = 0.55;

/// One objective waypoint: what, where and which authored colour.
///
/// This is the DATA only. Painting it is `film_ui::paint_name_tags`' job, through
/// [`waypoint_name_tags`] — the two must not grow separate layout rules, which is why this type
/// carries no screen geometry at all.
#[derive(Clone, Debug, PartialEq)]
pub struct Waypoint {
    /// The objective object's net id.
    pub net_id: u32,
    /// The label: the objective's own name, capitalised (`Flag`, `Ball`, `Skull`).
    pub text: String,
    /// World position of the anchor — the recorded position lifted by
    /// [`OBJECTIVE_WAYPOINT_UP_WU`].
    pub pos: [f32; 3],
    /// The chud colour index [`crate::reach_chud::color_from_index`] takes: 0 `navpt normal`,
    /// 2 `navpt neutral`, 3 `navpt enemy`.
    pub color_index: i32,
    /// The biped carrying it, and whether the recording stated that or HMS derived it.
    pub carrier: Option<(u32, Carry)>,
}

/// The label for an objective leaf: `flag` -> `Flag`, `cov_power_module` -> `Cov Power Module`.
pub fn waypoint_label(leaf: &str) -> String {
    leaf.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The waypoint for every objective object drawn at `tick`.
///
/// ## The colour, and exactly how far the recording supports it
///
/// Reach's objective waypoints are chud navpoints, the same mechanism as the floating player
/// name, so the palette and the index rule are the ones already decoded in `film_ui`:
/// `chud_color_from_index` (`haloreach.dll+0x2E36B8`) over the `chgd` skin's 32 authored colours,
/// with 0 `navpt normal` #2494FF, 2 `navpt neutral` #5E5E5E and 3 `navpt enemy` #D83232.
///
/// **Which index an objective gets is NOT in the recording, and HMS does not pretend otherwise.**
/// A film states neither the gametype's waypoint colour nor the flag's own team — the flag is one
/// tag for both sides, and the team lives in the `mp_flag_base` the gametype was pointed at, not
/// in the stream. What the recording does state is who is holding it, so:
///
/// * **carried** — the carrier's team against the viewer's, through the engine's own
///   `navpoint_color_index` (`+0x14844C`): same team 0, different team 3, unknown on either side
///   2. That is the engine's comparison applied to the one team the recording gives.
/// * **on the ground** — `navpt neutral`. Nothing in the film says whose flag it is, and
///   `navpt neutral` is the engine's own answer for an unknown team.
///
/// `viewer_team` and `teams_enabled` are the same two values `film_ui::resolve_name_colors` takes,
/// and `team_of` maps a biped's net id to its player's team (0xFF unknown) — supplied by the
/// caller because the owner->slot->team bridge lives in the playback, not here.
pub fn waypoints(
    table: &ObjectTable,
    idx: &crate::film_decode::FilmIndex,
    tick: u32,
    bipeds: &[(u32, [f32; 3])],
    viewer_team: u8,
    teams_enabled: bool,
    team_of: &dyn Fn(u32) -> u8,
) -> Vec<Waypoint> {
    let mut out = Vec::new();
    for (o, pos, _) in table.to_draw_at(idx, tick) {
        if o.draw != Draw::Objective {
            continue;
        }
        let carrier = carrier_at(idx, o.net_id, tick, pos, bipeds);
        let color_index = match carrier {
            // Teams off puts every player on their own side, so a carried objective can never be
            // an ally's: the engine's comparison with two different teams is index 3.
            Some((b, _)) if teams_enabled => {
                crate::reach_chud::navpoint_color_index(viewer_team, team_of(b))
            }
            Some(_) => crate::reach_chud::navpoint_color_index(0, 1),
            None => crate::reach_chud::navpoint_color_index(crate::reach_chud::TEAM_NONE, crate::reach_chud::TEAM_NONE),
        };
        out.push(Waypoint {
            net_id: o.net_id,
            text: waypoint_label(&o.name),
            pos: [pos[0], pos[1], pos[2] + OBJECTIVE_WAYPOINT_UP_WU],
            color_index,
            carrier,
        });
    }
    // Stable order so two frames at the same tick produce the same list (a HashMap walk does
    // not): by net id.
    out.sort_by_key(|w| w.net_id);
    out
}

/// The waypoints as the name-tag records `film_ui::paint_name_tags` already paints.
///
/// An objective waypoint is a chud navpoint and so is a player's floating name, so they are drawn
/// by one painter with one set of authored rules rather than two. `on_marker` is left FALSE,
/// because the anchor is [`OBJECTIVE_WAYPOINT_UP_WU`] — HMS's own lift — and that flag is exactly
/// how the UI reports an anchor it did not resolve from the engine's own marker.
///
/// `show_icon` is false: the chevron sprite belongs to the ally-waypoint producer
/// (`+0x2CB76C`), which makes a record only for a teammate, and an objective is not one.
pub fn waypoint_name_tags(ws: &[Waypoint]) -> Vec<crate::film_ui::NameTag> {
    ws.iter()
        .map(|w| crate::film_ui::NameTag {
            text: w.text.clone(),
            pos: w.pos,
            named: true,
            inferred_from: None,
            team: crate::reach_chud::TEAM_NONE,
            net_id: w.net_id,
            feet: [w.pos[0], w.pos[1], w.pos[2] - OBJECTIVE_WAYPOINT_UP_WU],
            on_marker: false,
            slot: None,
            show_icon: false,
            color_index: w.color_index,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// THE GATE ON THE CLASSIFICATION. Every arm of [`classify`] is a decision about a measured
    /// group, so each one is asserted — including the two negatives, which are the ones a
    /// careless widening would break: an `eqip` token must stay off the screen, and so must a
    /// `bloc` the Forge scene already draws.
    #[test]
    fn classify_matches_the_measured_inventory() {
        // Rows of the Castle Wars inventory, by (group, tag path, did the track move).
        let rows: &[(&str, &str, bool, Draw)] = &[
            ("weap", "objects\\weapons\\multiplayer\\flag\\flag", true, Draw::Objective),
            // A flag the recording never moved is still the objective, not scenery.
            ("weap", "objects\\weapons\\multiplayer\\flag\\flag", false, Draw::Objective),
            ("weap", "objects\\weapons\\melee\\energy_sword\\energy_sword", true, Draw::LooseWeapon),
            ("weap", "objects\\weapons\\melee\\gravity_hammer\\gravity_hammer", true, Draw::LooseWeapon),
            ("bloc", "objects\\multi\\models\\mp_flag_base\\mp_flag_base", false, Draw::AlreadyInScene),
            ("bloc", "objects\\multi\\models\\mp_hill_beacon\\mp_hill_beacon", false, Draw::AlreadyInScene),
            // The 95 moving `bloc` tracks: debris, drawn.
            ("bloc", "objects\\levels\\multi\\pickup_garbage_tire\\pickup_garbage_tire", true, Draw::Debris),
            ("bloc", "objects\\vehicles\\warthog\\damage\\wheel\\wheel", true, Draw::Debris),
            // ...and the 535 that do not: the Forge pieces, left to the scene.
            ("bloc", "objects\\multi\\forge_hal\\ff_plat_4x4\\ff_plat_4x4", false, Draw::AlreadyInScene),
            ("eqip", "objects\\equipment\\sprint\\sprint", false, Draw::NoWorldPresence),
            ("eqip", "objects\\equipment\\ammo_box\\ammo_box", false, Draw::NoWorldPresence),
            ("scen", "objects\\levels\\multi\\kill_volume\\kill_volume", false, Draw::AlreadyInScene),
            ("ctrl", "objects\\levels\\shared\\device_controls\\health_station\\health_station", false, Draw::AlreadyInScene),
            ("effe", "effects\\spawn_detonation_hack", false, Draw::NoWorldPresence),
            ("jmad", "objects\\vehicles\\warthog\\warthog_rocket", false, Draw::NoWorldPresence),
            ("snd!", "sound\\evs_ground_push_down_headstab_victim", false, Draw::NoWorldPresence),
        ];
        for (g, p, m, want) in rows {
            assert_eq!(classify(g, p, *m), *want, "{g} {p} moves={m}");
        }
        // And the whole objective set, which is what the user's report was about.
        for leaf in OBJECTIVE_LEAVES {
            let path = format!("objects\\multi\\{leaf}\\{leaf}");
            assert_eq!(classify("weap", &path, true), Draw::Objective, "{leaf}");
        }
        assert!(Draw::Objective.drawn());
        assert!(Draw::LooseWeapon.drawn());
        assert!(Draw::Debris.drawn());
        assert!(!Draw::AlreadyInScene.drawn());
        assert!(!Draw::NoWorldPresence.drawn());
        // The buckets are distinct, or `by_draw` would silently merge two verdicts.
        let mut b: Vec<usize> = [Draw::Objective, Draw::LooseWeapon, Draw::Debris, Draw::AlreadyInScene, Draw::NoWorldPresence]
            .iter()
            .map(|d| d.bucket())
            .collect();
        b.sort();
        assert_eq!(b, vec![0, 1, 2, 3, 4]);
    }

    /// A flag is not matched by a substring of some other weapon's path, and a weapon whose leaf
    /// merely CONTAINS an objective name is not promoted. `flagship` would be a false positive
    /// for a `contains` test; the leaf is compared whole.
    #[test]
    fn objective_leaf_is_the_whole_leaf() {
        assert_eq!(objective_leaf("objects\\multi\\flag\\flag"), Some("flag"));
        assert_eq!(objective_leaf("objects\\multi\\FLAG\\FLAG"), Some("flag"));
        assert_eq!(objective_leaf("objects\\weapons\\rifle\\flagship"), None);
        assert_eq!(objective_leaf("objects\\weapons\\rifle\\dmr\\dmr"), None);
        // `flag` deeper in the PATH but not the leaf is not an objective: that is
        // `mp_flag_base`'s shape, and it must stay a scene object.
        assert_eq!(objective_leaf("objects\\multi\\models\\mp_flag_base\\mp_flag_base"), None);
    }

    /// THE GATE ON THE SPAWN SKIP, and on the structure the skip reads.
    ///
    /// The whole point of the skip is that a never-touched weapon is left to the Forge scene; the
    /// failure it prevents is two identical models in one place. It must bite in exactly one
    /// direction, so this asserts BOTH: a weapon on its spawn is skipped, and one that has moved
    /// off it is not — and that an object with no placed copy at all (the flag) is never skipped,
    /// which is what keeps the flag on screen.
    #[test]
    fn spawn_coincident_skip_bites_both_ways() {
        let placed = |d: f32| ObjectIdent {
            net_id: 1,
            entity_type: 17,
            obj_tag: 7,
            group: "weap".into(),
            name: "dmr".into(),
            draw: Draw::LooseWeapon,
            class: Some("rifle".into()),
            spawn: Some(([0.0, 0.0, 0.0], d)),
        };
        assert!(placed(0.0).on_its_spawn([0.0, 0.0, 0.0]), "exactly on its spawn");
        assert!(placed(0.0).on_its_spawn([0.0, 0.0, SPAWN_COINCIDENT_WU - 0.01]), "inside the radius");
        assert!(!placed(0.0).on_its_spawn([0.0, 0.0, SPAWN_COINCIDENT_WU + 0.01]), "outside the radius");
        // The flag: the gametype spawns it, the variant places no copy, so there is nothing to
        // double up with and it is ALWAYS drawn.
        let flag = ObjectIdent { spawn: None, name: "flag".into(), draw: Draw::Objective, ..placed(0.0) };
        assert!(!flag.on_its_spawn([0.0, 0.0, 0.0]), "an objective with no placed copy is never skipped");
    }

    /// THE GATE ON THE PLACEMENT. A recorded basis has to reach the drawn matrix intact —
    /// including a ROLL, which is the part a yaw-only placement silently throws away. The
    /// structure asserted is the engine's column order (`col0 = forward`, `col1 = up x forward`,
    /// `col2 = up`), so a transposed or reordered basis fails rather than merely looking wrong.
    #[test]
    fn object_basis_keeps_the_recorded_roll() {
        let pos = glam::Vec3::new(1.0, 2.0, 3.0);
        // A flag lying on its side: forward +X, up -Y. 90 degrees of roll.
        let m = object_basis(pos, Some(([1.0, 0.0, 0.0], [0.0, -1.0, 0.0]))).expect("a basis was recorded");
        assert!((m.w_axis.truncate() - pos).length() < 1e-5, "the position survives");
        assert!((m.x_axis.truncate() - glam::Vec3::X).length() < 1e-5, "col0 is forward");
        assert!((m.z_axis.truncate() - glam::Vec3::new(0.0, -1.0, 0.0)).length() < 1e-5, "col2 is up");
        assert!(
            crate::film_vehicle::tilt_degrees(m.z_axis.truncate().to_array()) > 89.0,
            "90 degrees off level is reported as such, not flattened"
        );
        // No recorded orientation: no matrix, so the caller keeps its yaw placement rather than
        // being handed an identity that would face every flag +X.
        assert!(object_basis(pos, None).is_none());
    }

    /// THE GATE ON DETERMINISM. Every placement must be a pure function of the tick, so a scrub
    /// backwards reproduces the frame exactly. Built on a synthetic track with sparse position
    /// AND orientation keys, because the interpolation is where a stateful shortcut would hide.
    #[test]
    fn placement_is_a_pure_function_of_the_tick() {
        let mut tr = crate::film_decode::EntityTrack {
            net_id: 9,
            entity_type: 17,
            first_tick: 0,
            last_tick: 600,
            moves_cached: Some(true),
            ..Default::default()
        };
        for k in 0..20u32 {
            let t = k * 30;
            tr.positions.push((t, [k as f32, 0.5 * k as f32, 1.0]));
        }
        for k in 0..7u32 {
            let t = k * 90;
            let a = k as f32 * 0.4;
            tr.orientations.push((t, [a.cos(), a.sin(), 0.0], [0.0, 0.0, 1.0]));
        }
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.ticks = 601;
        idx.tracks.push(tr);
        let mut table = ObjectTable::default();
        table.objects.insert(
            9,
            ObjectIdent {
                net_id: 9,
                entity_type: 17,
                obj_tag: 3,
                group: "weap".into(),
                name: "flag".into(),
                draw: Draw::Objective,
                class: Some("flag".into()),
                spawn: None,
            },
        );
        let at = |t: u32| -> (Vec<f32>, Vec<f32>) {
            let v = table.to_draw_at(&idx, t);
            let p: Vec<f32> = v.iter().flat_map(|(_, p, _)| p.to_vec()).collect();
            let m: Vec<f32> = v
                .iter()
                .flat_map(|(_, p, o)| {
                    object_basis(glam::Vec3::from(*p), *o)
                        .map(|m| m.to_cols_array().to_vec())
                        .unwrap_or_default()
                })
                .collect();
            (p, m)
        };
        // Forwards, then backwards over the same ticks: bit-identical both ways.
        let ticks: Vec<u32> = (0..=600).step_by(7).collect();
        let fwd: Vec<_> = ticks.iter().map(|&t| at(t)).collect();
        let back: Vec<_> = ticks.iter().rev().map(|&t| at(t)).collect();
        for (i, b) in back.iter().enumerate() {
            let f = &fwd[ticks.len() - 1 - i];
            assert_eq!(f.0, b.0, "position at tick {}", ticks[ticks.len() - 1 - i]);
            assert_eq!(f.1, b.1, "basis at tick {}", ticks[ticks.len() - 1 - i]);
        }
        // And it really did produce something: a gate that passes on an empty list is no gate.
        assert_eq!(fwd[1].0.len(), 3, "a position was produced");
        assert_eq!(fwd[1].1.len(), 16, "a basis was produced");
    }

    /// THE GATE ON THE FLAG'S TRACK CHAIN — the structure the recording actually has.
    ///
    /// A flag is not one long-lived entity: [`probes::probe_flag_track_lifetimes`] measures it as
    /// a CHAIN of 33 short tracks on the Castle Wars clip, each ended by a kind-4 destroy with
    /// the next created 0 to 283 ticks later, **at most two alive at once** (one per team). So the
    /// lifetime that matters is `destroyed_tick`, not `last_tick`, and the two are far apart: on
    /// the measured chain a track's last position report comes up to 3,209 ticks before its
    /// destroy (`0xefff02d0`: last 24083, destroyed 27292). Honouring `last_tick` instead would
    /// make the flag vanish for most of its life; ignoring the destroy would leave the old flag
    /// standing while its replacement appeared, so two flags would be drawn where one exists.
    ///
    /// This builds that exact shape and asserts the count drawn at every tick.
    #[test]
    fn a_flag_chain_draws_exactly_one_flag_at_a_time() {
        let mk = |net: u32, first: u32, last: u32, destroy: u32| crate::film_decode::EntityTrack {
            net_id: net,
            entity_type: 17,
            first_tick: first,
            last_tick: last,
            destroyed_tick: Some(destroy),
            moves_cached: Some(true),
            positions: vec![(first, [net as f32, 0.0, 0.0]), (last, [net as f32 + 1.0, 0.0, 0.0])],
            ..Default::default()
        };
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.ticks = 400;
        // The real shape: a long dead tail between the last report and the destroy, then a gap
        // before the replacement is created.
        idx.tracks = vec![mk(1, 0, 60, 200), mk(2, 220, 280, 300)];
        let mut table = ObjectTable::default();
        for net in [1u32, 2] {
            table.objects.insert(
                net,
                ObjectIdent {
                    net_id: net,
                    entity_type: 17,
                    obj_tag: 3,
                    group: "weap".into(),
                    name: "flag".into(),
                    draw: Draw::Objective,
                    class: Some("flag".into()),
                    spawn: None,
                },
            );
        }
        let n_at = |t: u32| table.to_draw_at(&idx, t).len();
        // Inside the first flag's life -- including the dead tail AFTER its last report, which
        // is where a `last_tick` lifetime would have lost it.
        for t in [0u32, 30, 60, 61, 120, 199, 200] {
            assert_eq!(n_at(t), 1, "one flag at tick {t}");
        }
        // The gap between the destroy and the replacement: no flag exists.
        for t in [201u32, 210, 219] {
            assert_eq!(n_at(t), 0, "no flag at tick {t}");
        }
        // The replacement, alone.
        for t in [220u32, 250, 299, 300] {
            let v = table.to_draw_at(&idx, t);
            assert_eq!(v.len(), 1, "one flag at tick {t}");
            assert_eq!(v[0].0.net_id, 2, "the replacement, not the destroyed one");
        }
        assert_eq!(n_at(301), 0, "after the second destroy, nothing");
    }

    /// THE GATE ON THE WAYPOINT COLOUR. The three outcomes are the engine's own comparison, and
    /// each is a different claim about the recording, so each is asserted — plus the honest
    /// default for a flag nobody is holding.
    #[test]
    fn waypoint_colour_follows_the_carrier_and_nothing_else() {
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.ticks = 100;
        let mut tr = crate::film_decode::EntityTrack {
            net_id: 9,
            entity_type: 17,
            first_tick: 0,
            last_tick: 100,
            moves_cached: Some(true),
            ..Default::default()
        };
        tr.positions.push((0, [10.0, 0.0, 0.0]));
        idx.tracks.push(tr);
        let mut table = ObjectTable::default();
        table.objects.insert(
            9,
            ObjectIdent {
                net_id: 9,
                entity_type: 17,
                obj_tag: 3,
                group: "weap".into(),
                name: "flag".into(),
                draw: Draw::Objective,
                class: Some("flag".into()),
                spawn: None,
            },
        );
        // Nobody near it: neutral, and no carrier claimed.
        let w = waypoints(&table, &idx, 0, &[], 0, true, &|_| 0);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].text, "Flag");
        assert_eq!(w[0].carrier, None);
        assert_eq!(w[0].color_index, 2, "an unheld flag is navpt neutral");
        assert!((w[0].pos[2] - OBJECTIVE_WAYPOINT_UP_WU).abs() < 1e-6, "lifted off the object origin");
        // A player standing a long way off is not the carrier.
        let far = [(1u32, [12.0f32, 0.0, 0.0])];
        assert_eq!(waypoints(&table, &idx, 0, &far, 0, true, &|_| 0)[0].carrier, None);
        // Holding it: the viewer's own team -> navpt normal.
        let held = [(1u32, [10.0f32, 0.0, 0.1])];
        let w = waypoints(&table, &idx, 0, &held, 0, true, &|_| 0);
        assert_eq!(w[0].carrier, Some((1, Carry::Derived)));
        assert_eq!(w[0].color_index, 0, "a teammate carrying it is navpt normal");
        // The other team -> navpt enemy.
        assert_eq!(waypoints(&table, &idx, 0, &held, 0, true, &|_| 1)[0].color_index, 3);
        // An unknown team on either side -> navpt neutral, the engine's own answer.
        assert_eq!(
            waypoints(&table, &idx, 0, &held, crate::reach_chud::TEAM_NONE, true, &|_| 0)[0].color_index,
            2
        );
        // Teams off: Reach gives every player their own team, so a carried objective is enemy.
        assert_eq!(waypoints(&table, &idx, 0, &held, 0, false, &|_| 0)[0].color_index, 3);
    }

    /// A STATED pickup beats the proximity test, and a stale one does not outlive the two being
    /// together. The second half is the one that bites: without it the waypoint would stay on a
    /// player who scored and respawned across the map.
    #[test]
    fn a_stated_pickup_wins_but_does_not_go_stale() {
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.ticks = 100;
        let mut tr = crate::film_decode::EntityTrack {
            net_id: 9,
            entity_type: 17,
            first_tick: 0,
            last_tick: 100,
            moves_cached: Some(true),
            ..Default::default()
        };
        tr.positions.push((0, [0.0, 0.0, 0.0]));
        idx.tracks.push(tr);
        idx.events.push(crate::film_action::ActionEvent {
            tick: 5,
            kind: crate::film_action::EV_PICKUP,
            subject: Some(1),
            other: Some(9),
            ..Default::default()
        });
        // Two bipeds both inside the radius; the stated one is NOT the nearest.
        let bipeds = [(1u32, [0.0f32, 0.0, 0.3]), (2u32, [0.0, 0.0, 0.05])];
        assert_eq!(
            carrier_at(&idx, 9, 10, [0.0, 0.0, 0.0], &bipeds),
            Some((1, Carry::Stated)),
            "the recording's own say wins over the nearer body"
        );
        // Before the event, proximity only.
        assert_eq!(carrier_at(&idx, 9, 4, [0.0, 0.0, 0.0], &bipeds), Some((2, Carry::Derived)));
        // The stated carrier has walked away: the claim lapses rather than following him.
        let gone = [(1u32, [20.0f32, 0.0, 0.0])];
        assert_eq!(carrier_at(&idx, 9, 50, [0.0, 0.0, 0.0], &gone), None);
    }

    /// The label is the leaf made readable, and the name-tag bridge carries every field the
    /// painter reads — including the two that must stay FALSE, because `on_marker` is how the UI
    /// reports an anchor HMS chose rather than decoded and `show_icon` would draw an ally
    /// chevron over an objective.
    #[test]
    fn waypoint_name_tags_carry_the_right_record() {
        assert_eq!(waypoint_label("flag"), "Flag");
        assert_eq!(waypoint_label("cov_power_module"), "Cov Power Module");
        assert_eq!(waypoint_label("unsc_data_core"), "Unsc Data Core");
        let w = Waypoint {
            net_id: 0x1234,
            text: "Flag".into(),
            pos: [1.0, 2.0, 3.0],
            color_index: 3,
            carrier: Some((7, Carry::Stated)),
        };
        let t = waypoint_name_tags(std::slice::from_ref(&w));
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].text, "Flag");
        assert_eq!(t[0].pos, [1.0, 2.0, 3.0]);
        assert_eq!(t[0].color_index, 3);
        assert_eq!(t[0].net_id, 0x1234);
        assert!(!t[0].on_marker, "the anchor is HMS's own lift and must say so");
        assert!(!t[0].show_icon, "the ally chevron belongs to the player-name producer");
        assert_eq!(t[0].slot, None);
        assert!((t[0].feet[2] - (3.0 - OBJECTIVE_WAYPOINT_UP_WU)).abs() < 1e-6);
    }

    /// The build walk routes every track into exactly one bucket, and the three counters add up
    /// to the whole index. A silent miscount here is how a whole group would stop being drawn
    /// with nothing to show for it.
    #[test]
    fn build_accounts_for_every_track() {
        let mk = |net: u32, ty: u8, wire: Option<u32>| crate::film_decode::EntityTrack {
            net_id: net,
            entity_type: ty,
            first_tick: 0,
            last_tick: 10,
            def_wire: wire,
            moves_cached: Some(true),
            positions: vec![(0, [0.0, 0.0, 0.0])],
            ..Default::default()
        };
        let mut idx = crate::film_decode::FilmIndex::default();
        idx.tracks = vec![
            mk(1, crate::film_play::BIPED_TYPE, Some(1)),
            mk(2, crate::film_anim::VEHICLE_TYPE, Some(2)),
            mk(3, 17, Some(3)),   // weap flag
            mk(4, 17, Some(4)),   // weap dmr
            mk(5, 13, Some(5)),   // bloc
            mk(6, 12, Some(6)),   // eqip
            mk(7, 17, None),      // no definition on the wire
            mk(8, 17, Some(999)), // a wire value that does not resolve
        ];
        let table = ObjectTable::build(
            &idx,
            |w| match w {
                3 => Some((103, "weap".into(), "objects\\multi\\flag\\flag".into())),
                4 => Some((104, "weap".into(), "objects\\weapons\\rifle\\dmr\\dmr".into())),
                5 => Some((105, "bloc".into(), "objects\\multi\\models\\mp_flag_base\\mp_flag_base".into())),
                6 => Some((106, "eqip".into(), "objects\\equipment\\sprint\\sprint".into())),
                _ => None,
            },
            |_| None,
            &[],
        );
        assert_eq!(table.other_lane, 2, "the biped and the vehicle have their own lanes");
        assert_eq!(table.unresolved, 2, "one track with no wire value, one that does not resolve");
        // `mk` marks every track as moving, so the `bloc` is debris: objective, weapon, debris,
        // nothing in-scene, one token.
        assert_eq!(table.by_draw, [1, 1, 1, 0, 1, 0]);
        assert_eq!(table.objects.len(), 3, "the three drawable ones are kept");
        assert_eq!(table.objectives().count(), 1);
        assert_eq!(table.objects[&3].name, "flag");
        assert_eq!(table.objects[&3].draw, Draw::Objective);
        assert_eq!(table.objects[&4].draw, Draw::LooseWeapon);
        assert_eq!(table.objects[&5].draw, Draw::Debris);
        // And the buckets really are the whole index.
        assert_eq!(
            table.other_lane + table.unresolved + table.by_draw.iter().sum::<usize>(),
            idx.tracks.len()
        );
        // THE CAP PUTS THE OBJECTIVE FIRST. A cap of one must keep the flag, not whichever
        // net id a hash map happened to walk first.
        let one = table.to_draw_capped(&idx, 5, 1);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].0.draw, Draw::Objective, "a cap never drops the objective");
        let all = table.to_draw_at(&idx, 5);
        assert_eq!(all.len(), 3);
        assert_eq!(
            all.iter().map(|(o, _, _)| o.net_id).collect::<Vec<_>>(),
            vec![3, 4, 5],
            "objectives first, then by net id"
        );
    }
}

#[cfg(test)]
mod probes {
    //! #film-objects  What a real recording says about its objective objects, measured.
    //!
    //! `#[ignore]`d: these open films (and one opens a map), which is minutes of work and
    //! gigabytes of RSS — far too heavy for the ordinary suite. Run one as
    //! `cargo test -p hms-app --profile quick -j 8 probe_objective_motion_and_carry -- --ignored --nocapture`.

    use crate::film;
    use crate::film_decode::{self, FilmIndex};

    fn name(p: &std::path::Path) -> String {
        p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    }

    fn indexed(p: &std::path::Path) -> Option<(film::Film, FilmIndex)> {
        let f = film::open(p).ok()?;
        if f.data.records.is_empty() || f.header.map_variant.is_empty() {
            return None;
        }
        let idx = film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
        (idx.entity_messages > 0).then_some((f, idx))
    }

    /// **Is a `weap` track's ROTATION in the recording, and does the recording keep reporting an
    /// objective's position while a player carries it?**
    ///
    /// Both questions decide a design choice, so both are measured rather than assumed:
    ///
    /// * if the orientation density is high, a flag is placed with the full recorded basis, like
    ///   a vehicle, instead of a compass heading;
    /// * if the position keeps arriving while the flag is held, the carriage is RECORDED and the
    ///   flag is drawn where the film puts it — no derived attachment to a hand marker.
    ///
    /// The carry test is the distance from the objective to the nearest biped at the same tick:
    /// a carried flag tracks its carrier, a dropped one does not.
    #[test]
    #[ignore]
    fn probe_objective_motion_and_carry() {
        let only = std::env::var("HMS_FILM").ok();
        let films: Vec<std::path::PathBuf> = match &only {
            Some(p) => vec![std::path::PathBuf::from(p)],
            None => film::find_finished_films(),
        };
        if films.is_empty() {
            eprintln!("film_objective: no films on this machine");
            return;
        }
        println!(
            "{:<42} {:>6} {:>8} {:>8} {:>6} {:>7} {:>6}",
            "film", "tracks", "pos", "orient", "o/pos", "moved", ">5deg"
        );
        let (mut t_pos, mut t_ori, mut t_tracks, mut t_moved, mut t_tilt) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for p in films.iter() {
            let Some((_f, idx)) = indexed(p) else { continue };
            // Every non-biped, non-vehicle track that reports a position at all: the superset of
            // what the draw filter can ever want, without needing a map to resolve names.
            let ts: Vec<&film_decode::EntityTrack> = idx
                .tracks
                .iter()
                .filter(|t| {
                    t.entity_type != crate::film_play::BIPED_TYPE
                        && t.entity_type != crate::film_anim::VEHICLE_TYPE
                        && !t.positions.is_empty()
                })
                .collect();
            if ts.is_empty() {
                continue;
            }
            let (mut fp, mut fo, mut moved, mut tilt) = (0usize, 0usize, 0usize, 0usize);
            for t in &ts {
                fp += t.positions.len();
                fo += t.orientations.len();
                if t.moves() {
                    moved += 1;
                }
                for &(_, _f, u) in &t.orientations {
                    if crate::film_vehicle::tilt_degrees(u) > 5.0 {
                        tilt += 1;
                    }
                }
            }
            println!(
                "{:<42} {:>6} {:>8} {:>8} {:>6.2} {:>7} {:>6}",
                name(p),
                ts.len(),
                fp,
                fo,
                if fp > 0 { fo as f32 / fp as f32 } else { 0.0 },
                moved,
                tilt
            );
            t_pos += fp;
            t_ori += fo;
            t_tracks += ts.len();
            t_moved += moved;
            t_tilt += tilt;
        }
        println!(
            "\nTOTAL {t_tracks} non-biped/non-vehicle track(s) with a position: {t_pos} positions, \
             {t_ori} orientations ({:.1}% as many), {t_moved} moved ({:.1}%), {t_tilt} bases over 5 deg off level",
            100.0 * t_ori as f64 / t_pos.max(1) as f64,
            100.0 * t_moved as f64 / t_tracks.max(1) as f64,
        );

        // --- the carry question, on the one film named ---------------------------------------
        let Some(p) = films.first() else { return };
        let Some((_f, idx)) = indexed(p) else { return };
        let bipeds: Vec<&film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == crate::film_play::BIPED_TYPE).collect();
        // Candidates: the moving non-biped tracks with the most position reports, which on an
        // objective clip is where the flag is.
        let mut cands: Vec<&film_decode::EntityTrack> = idx
            .tracks
            .iter()
            .filter(|t| {
                t.entity_type != crate::film_play::BIPED_TYPE
                    && t.entity_type != crate::film_anim::VEHICLE_TYPE
                    && t.moves()
            })
            .collect();
        cands.sort_by_key(|t| std::cmp::Reverse(t.positions.len()));
        cands.truncate(12);
        println!(
            "\n=== carriage: how close the nearest biped is, per recorded position ===\n\
             {:<12} {:>4} {:>7} {:>7} {:>8} {:>8} {:>8}",
            "net id", "type", "pos", "orient", "<0.5wu", "<1.5wu", "span(wu)"
        );
        for t in &cands {
            let (mut near, mut mid) = (0usize, 0usize);
            for &(tick, p) in &t.positions {
                let mut best = f32::INFINITY;
                for b in &bipeds {
                    if !b.alive_at(tick) {
                        continue;
                    }
                    if let Some(bp) = b.sample(tick) {
                        let d = ((p[0] - bp[0]).powi(2) + (p[1] - bp[1]).powi(2) + (p[2] - bp[2]).powi(2)).sqrt();
                        if d < best {
                            best = d;
                        }
                    }
                }
                if best <= super::CARRY_RADIUS_WU {
                    near += 1;
                }
                if best <= 1.5 {
                    mid += 1;
                }
            }
            println!(
                "{:#010x} {:>4} {:>7} {:>7} {:>7} {:>7} {:>8.1}",
                t.net_id,
                t.entity_type,
                t.positions.len(),
                t.orientations.len(),
                near,
                mid,
                t.path_length()
            );
        }
        // The pickup events the recording DOES state, which is the other half of the answer.
        let mut by_kind: std::collections::BTreeMap<u8, usize> = Default::default();
        for e in &idx.events {
            if matches!(e.kind, crate::film_action::EV_PICKUP | super::EV_WEAPON_PICKUP | 33 | 34 | 35) {
                *by_kind.entry(e.kind).or_default() += 1;
            }
        }
        println!("\npickup / drop / trade events stated by this film:");
        for (k, n) in &by_kind {
            println!("  {:<24} {n}", crate::film_action::event_name(*k));
        }
        if by_kind.is_empty() {
            println!("  none at all");
        }
    }

    /// The production [`super::ObjectTable::build`] run on a REAL film with the REAL resolver —
    /// the network tag-reference table plus the scenario sandbox palette — so the classification,
    /// the spawn-coincidence skip and the per-tick draw list are exercised end to end against
    /// recorded data rather than against a fixture.
    ///
    /// This is the one check that a GPU is not needed for and that nothing in the unit tests can
    /// give: the unit tests gate the RULES, and this gates the rules applied to the clip the user
    /// reported. It prints the verdict histogram, the drawn set at a spread of ticks, and asserts
    /// the two facts the report turns on — the flag resolves, and it is drawn.
    #[test]
    #[ignore]
    fn probe_real_table_over_a_film() {
        let Ok(film_path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM to the film to inspect");
            return;
        };
        let stem = std::env::var("HMS_MAP_STEM").unwrap_or_else(|_| "forge_halo".to_string());
        let Some((native, cache)) = crate::film_action::objective_inventory::open_reach(&stem) else {
            eprintln!("no Reach cache named {stem} on this machine");
            return;
        };
        let map = crate::mapcat::enumerate()
            .into_iter()
            .find(|c| {
                c.game == crate::mapcat::Game::Reach
                    && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(&stem))
            })
            .expect("the map that just opened is still listed");
        let bytes = std::fs::read(&map.path).expect("map reads");
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in crate::film_action::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) {
                tags.push((t, (*g).to_string()));
            }
        }
        let refs = crate::film_action::RefTable::for_map(&bytes, &tags).expect("ref table builds");
        let Some((_f, idx)) = indexed(std::path::Path::new(&film_path)) else {
            eprintln!("that film does not decode");
            return;
        };
        // The film's own map variant, resolved through the sandbox palette exactly as
        // `ActorDriver::resolve_objects` does it.
        let palette = native.find_scenario(cache).map(|s| native.scnr_forge_palette_full(cache, s)).unwrap_or_default();
        let mut order: Vec<(u32, u32)> = Vec::new();
        {
            let mut seen = std::collections::HashSet::new();
            for e in &palette {
                let k = (e.palette_index, e.entry_within);
                if seen.insert(k) {
                    order.push(k);
                }
            }
        }
        let spawns: Vec<(u32, [f32; 3])> = idx
            .variant_objects
            .iter()
            .filter_map(|&(folder, item, pos)| {
                let &(pi, ew) = order.get(folder as usize)?;
                let e = palette
                    .iter()
                    .find(|p| p.palette_index == pi && p.entry_within == ew && p.variant_within == item as u32)
                    .or_else(|| palette.iter().find(|p| p.palette_index == pi && p.entry_within == ew))?;
                Some((e.tag_short, pos))
            })
            .collect();
        let mut classes: std::collections::HashMap<u32, Option<String>> = Default::default();
        let table = super::ObjectTable::build(
            &idx,
            |w| {
                let tag = refs.resolve(w)?;
                Some((tag, native.tag_class(cache, tag)?, native.tag_name(cache, tag)?))
            },
            |tag| {
                classes
                    .entry(tag)
                    .or_insert_with(|| crate::reach_actor::weapon_class(&native, cache, tag))
                    .clone()
            },
            &spawns,
        );
        println!(
            "\n=== {} ===\n  {} variant placement(s) resolved; verdicts: {} objective, {} weapon, {} debris, \
             {} already in the scene, {} token, {} unresolved, {} in another lane",
            name(std::path::Path::new(&film_path)),
            spawns.len(),
            table.by_draw[0],
            table.by_draw[1],
            table.by_draw[2],
            table.by_draw[3],
            table.by_draw[4],
            table.unresolved,
            table.other_lane
        );
        let mut per: std::collections::BTreeMap<(&str, String), usize> = Default::default();
        for o in table.objects.values() {
            *per.entry((o.draw.label(), format!("{} [{}]", o.name, o.class.clone().unwrap_or_default()))).or_default() +=
                1;
        }
        for ((k, n), c) in &per {
            println!("  {k:<9} {c:>4} x {n}");
        }
        println!("\n  drawn at a spread of ticks (objectives first):");
        let mut ever_flag = 0usize;
        for t in (0..idx.ticks).step_by((idx.ticks / 12).max(1) as usize) {
            let v = table.to_draw_capped(&idx, t, 48);
            let obj: Vec<String> = v
                .iter()
                .filter(|(o, _, _)| o.draw == super::Draw::Objective)
                .map(|(o, p, or)| {
                    format!(
                        "{}@({:.1},{:.1},{:.1}){}",
                        o.name,
                        p[0],
                        p[1],
                        p[2],
                        if or.is_some() { "" } else { " no-basis" }
                    )
                })
                .collect();
            ever_flag += obj.len();
            println!(
                "    tick {t:>6}: {:>3} want drawing, {:>2} after the cap, {:>2} on their spawn  {}",
                table.to_draw_at(&idx, t).len(),
                v.len(),
                table.parked_on_spawn_at(&idx, t),
                obj.join(" ")
            );
        }
        // The two facts the user's report turns on.
        assert!(
            table.objects.values().any(|o| o.name == "flag"),
            "the flag's definition resolves on this clip"
        );
        assert!(ever_flag > 0, "and the flag is in the drawn set at some sampled tick");
    }

    /// **What happens to the FLAG's own track while a player carries it?**
    ///
    /// This is the question that decides whether the carrier attachment is recorded or derived,
    /// and it cannot be answered without the map, because it needs each track's definition
    /// resolved — so this probe opens one.
    ///
    /// Per `weap flag` track it prints the life (`first..last`, whether a kind-4 destroy ended
    /// it), how many positions and orientations it carried, how close the nearest biped ever got,
    /// and the gap to the NEXT flag track. If the flag is attached on pickup and re-created on
    /// drop, the tracks tile the clip with no overlap and each one is short; if the recording
    /// keeps reporting a carried flag, one track lasts and tracks a body closely.
    #[test]
    #[ignore]
    fn probe_flag_track_lifetimes() {
        let Ok(film_path) = std::env::var("HMS_FILM") else {
            eprintln!("set HMS_FILM to the film to inspect");
            return;
        };
        let stem = std::env::var("HMS_MAP_STEM").unwrap_or_else(|_| "forge_halo".to_string());
        let Some((native, cache)) = crate::film_action::objective_inventory::open_reach(&stem) else {
            eprintln!("no Reach cache named {stem} on this machine");
            return;
        };
        let map = crate::mapcat::enumerate()
            .into_iter()
            .find(|c| {
                c.game == crate::mapcat::Game::Reach
                    && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(&stem))
            })
            .expect("the map that just opened is still listed");
        let bytes = std::fs::read(&map.path).expect("map reads");
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in crate::film_action::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) {
                tags.push((t, (*g).to_string()));
            }
        }
        let refs = crate::film_action::RefTable::for_map(&bytes, &tags).expect("ref table builds");
        let Some((_f, idx)) = indexed(std::path::Path::new(&film_path)) else {
            eprintln!("that film does not decode");
            return;
        };
        let bipeds: Vec<&film_decode::EntityTrack> =
            idx.tracks.iter().filter(|t| t.entity_type == crate::film_play::BIPED_TYPE).collect();

        // Group every drawable track by its resolved leaf, then report the objective ones in
        // full and the rest in summary.
        let mut per_leaf: std::collections::BTreeMap<String, Vec<&film_decode::EntityTrack>> = Default::default();
        for tr in &idx.tracks {
            if tr.entity_type == crate::film_play::BIPED_TYPE || tr.entity_type == crate::film_anim::VEHICLE_TYPE {
                continue;
            }
            let Some((tag, group, path)) = tr.def_wire.and_then(|w| {
                let tag = refs.resolve(w)?;
                Some((tag, native.tag_class(cache, tag)?, native.tag_name(cache, tag)?))
            }) else {
                continue;
            };
            let _ = tag;
            if !super::classify(&group, &path, tr.moves()).drawn() {
                continue;
            }
            let leaf = path.rsplit('\\').next().unwrap_or(&path).to_string();
            per_leaf.entry(leaf).or_default().push(tr);
        }

        for (leaf, mut trs) in per_leaf {
            trs.sort_by_key(|t| t.first_tick);
            let objective = super::OBJECTIVE_LEAVES.contains(&leaf.as_str());
            println!(
                "\n=== {leaf} ({} track(s), {} moved) ===",
                trs.len(),
                trs.iter().filter(|t| t.moves()).count()
            );
            if !objective {
                let (p, o) = (
                    trs.iter().map(|t| t.positions.len()).sum::<usize>(),
                    trs.iter().map(|t| t.orientations.len()).sum::<usize>(),
                );
                println!(
                    "  {p} positions, {o} orientations ({:.1}%), median life {} ticks",
                    100.0 * o as f64 / p.max(1) as f64,
                    {
                        let mut l: Vec<u32> = trs.iter().map(|t| t.last_tick - t.first_tick).collect();
                        l.sort();
                        l.get(l.len() / 2).copied().unwrap_or(0)
                    }
                );
                continue;
            }
            println!(
                "  {:<12} {:>7} {:>7} {:>7} {:>5} {:>6} {:>7} {:>7} {:>7} {:>8}",
                "net id", "first", "last", "destroy", "pos", "orient", "<0.5wu", "<1.5wu", "minD", "span"
            );
            for (i, t) in trs.iter().enumerate() {
                let (mut near, mut mid, mut mind) = (0usize, 0usize, f32::INFINITY);
                for &(tick, p) in &t.positions {
                    let mut best = f32::INFINITY;
                    for b in &bipeds {
                        if !b.alive_at(tick) {
                            continue;
                        }
                        if let Some(bp) = b.sample(tick) {
                            let d = ((p[0] - bp[0]).powi(2) + (p[1] - bp[1]).powi(2) + (p[2] - bp[2]).powi(2)).sqrt();
                            if d < best {
                                best = d;
                            }
                        }
                    }
                    if best <= super::CARRY_RADIUS_WU {
                        near += 1;
                    }
                    if best <= 1.5 {
                        mid += 1;
                    }
                    mind = mind.min(best);
                }
                println!(
                    "  {:#010x} {:>7} {:>7} {:>7} {:>5} {:>6} {:>7} {:>7} {:>7.2} {:>8.1}{}",
                    t.net_id,
                    t.first_tick,
                    t.last_tick,
                    t.destroyed_tick.map(|d| d.to_string()).unwrap_or_else(|| "-".into()),
                    t.positions.len(),
                    t.orientations.len(),
                    near,
                    mid,
                    mind,
                    t.path_length(),
                    match trs.get(i + 1) {
                        Some(n) => format!(
                            "   next starts {:+} ticks after this one's end",
                            n.first_tick as i64 - t.destroyed_tick.unwrap_or(t.last_tick) as i64
                        ),
                        None => String::new(),
                    }
                );
            }
            // Do the tracks TILE the clip (one flag at a time) or OVERLAP (two flags at once)?
            let mut most = 0usize;
            for tick in (0..idx.ticks).step_by(30) {
                let n = trs.iter().filter(|t| t.alive_at(tick) && t.sample(tick).is_some()).count();
                most = most.max(n);
            }
            println!("  at most {most} of these exist at the same moment (sampled every 30 ticks)");
        }
    }
}
