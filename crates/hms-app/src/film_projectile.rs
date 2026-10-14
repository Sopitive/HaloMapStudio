//! #projectile  A Theater clip's PROJECTILES, its weapon FIRING EFFECTS and its EXPLOSIONS.
//!
//! # What the recording actually states, measured before any of this was built
//!
//! The premise this work started from was "projectiles are not replicated as film entities, so
//! they must be simulated". Half of that is wrong, and the half that is wrong is the useful half:
//!
//! * The engine's entity table has a **`projectile` type (16)** and Reach films are full of them —
//!   grenades, rockets, missiles, needles, Falcon and turret bullets, each with a position every
//!   tick it is alive. HMS was dropping all of them, because `film_objective::classify`'s
//!   catch-all arm put `proj` in `Draw::AlreadyInScene` ("the map variant already draws this"),
//!   which is simply untrue of a grenade. They are now [`Draw::Projectile`] and drawn on the path
//!   the recording states — no simulation, no invented constant.
//! * The event channel carries `projectile_detonate` (5), `projectile_impact_effect` (6),
//!   `projectile_object_impact_effect` (7) and `projectile_attached` (4) as well as
//!   `action_weapon_fire` (26).
//!
//! What it does NOT state is a bullet. A rifle round is predicted on each client and never
//! replicated, so those have to be simulated — and the honest limit on simulating them is the
//! AIM: a Reach film records one player's look direction and zero-fills everybody else's (see
//! `film_play::ActorDriver::live_aim`). A shot whose direction the recording does not state is
//! **not drawn**, and the caption says how many were dropped, rather than aiming it along the
//! body yaw and calling that a simulation.
//!
//! # The three lanes
//!
//! 1. **Firing effects.** `weap` +0x408 `barrels` -> `firing effects[0].firing effect^` is an
//!    `effe`; it names its own markers (`primary_trigger`, `muzzle_flash`, `primary_ejection`)
//!    in its `effect_locations` block, which is how it finds the muzzle — the barrel's own
//!    `optional barrel marker name` is authored on only 7 of `forge_halo`'s 56 barrels.
//!    Played as a ONE-SHOT at the `action_weapon_fire` tick on the weapon in the body's hands.
//! 2. **Projectile attached effects.** A `proj` is an `obje`, so its travelling effect is an
//!    ordinary object attachment (`obje` +0x104) and goes through the machinery the armour-effect
//!    lane already uses. It is a MOVING source: the particles it emits must be left in the air
//!    where they were born, not dragged along with the rocket, which is what
//!    `scene::ParticleSource::path` is for.
//! 3. **Detonations.** `proj` `detonation effect (ground)` / `(airborne)` at the end of the
//!    projectile's recorded path, as a one-shot.
//!
//! Every marker is resolved strictly. A named marker the model does not have is COUNTED and
//! skipped, never replaced by the object origin — `hms-fx-marker-fallback` is the bug that rule
//! exists for.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::reach_projectile::ProjectileDef;

/// The engine entity type of a projectile (`a1[1 + type]`'s slot 1 returns `projectile`).
pub const PROJECTILE_TYPE: u8 = 16;

/// Scene-time seconds per film tick. The particle lane's scene clock IS
/// `Playback::anim_seconds()` = `tick / TICK_HZ`, so an effect born at film tick `T` is born at
/// scene time `T * TICK_SECONDS` and nothing has to be reconciled.
pub const TICK_SECONDS: f32 = 1.0 / crate::film_play::TICK_HZ;

/// The `effect_locations` name that means "the object's own origin": the model's ROOT NODE,
/// whose frame in model space is the identity. Not a marker, so a marker lookup can never find
/// it — and it is what every bullet's and every grenade's travelling effect names.
pub const ROOT_LOCATION: &str = "root";

/// Seed space for the film effect lane, so its sources can never collide with the object,
/// `effect_scenery` or actor lanes (`0x51F7…`, `0x9E37…`, `0xA7F0…`).
pub const FILM_FX_SEED: u64 = 0xF1_12_0000_0000_0000;

/// One effect the film lane wants played this frame.
///
/// Deliberately plain data with no `SceneController` in it: `ActorDriver::sync` runs with the
/// scene borrowed immutably (it is also what answers the corpse floor casts), so it decides WHAT
/// to play and [`install`] — called next to the actor lane, with `&mut SceneController` — resolves
/// the `effe` and installs the sources. The same split the actor particle lane uses.
#[derive(Clone, Debug, PartialEq)]
pub struct FilmEffect {
    /// Distinguishes this instance for the life of its clock. Must be stable across frames for
    /// one playing instance and different for a different one.
    pub seed: u64,
    /// The `effe` to play.
    pub effe: u32,
    /// The render model whose markers the effect's locations are resolved against (the weapon's
    /// model for a muzzle flash, the projectile's for a trail), or 0 for "no model": then only
    /// systems that name NO marker are played.
    pub mode: u32,
    /// The frame the effect is played in.
    pub world: Mat4,
    /// Scene time the effect started.
    pub birth: f32,
    /// A moving source's `(scene seconds, world position)` samples — see
    /// `scene::ParticleSource::path`.
    pub path: Option<std::sync::Arc<Vec<(f32, Vec3)>>>,
    /// For the diagnostics.
    pub label: String,
}

/// What [`install`] did, for the caption and the gates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FxReport {
    /// Effects asked for.
    pub requested: usize,
    /// Sources installed (an effect with no playable system installs nothing).
    pub installed: usize,
    /// Particle systems installed across them.
    pub systems: usize,
    /// Systems dropped because the effect named a marker the model does not have. **Counted
    /// rather than substituted** (`hms-fx-marker-fallback`).
    pub marker_missing: usize,
    /// Systems dropped because they are first-person-only (`camera_mode == 1`).
    pub first_person_only: usize,
    /// Effects whose `effe` decoded to no particle system at all.
    pub no_systems: usize,
}

impl FxReport {
    pub fn line(&self) -> String {
        format!(
            "{} effect(s) -> {} source(s), {} system(s){}{}{}",
            self.requested,
            self.installed,
            self.systems,
            if self.marker_missing > 0 { format!("; {} system(s) NAME A MARKER THE MODEL LACKS (skipped)", self.marker_missing) } else { String::new() },
            if self.first_person_only > 0 { format!("; {} first-person-only", self.first_person_only) } else { String::new() },
            if self.no_systems > 0 { format!("; {} effect(s) decode to no particle system", self.no_systems) } else { String::new() },
        )
    }
}

/// Resolve every requested effect against the cache and install the film particle lane.
///
/// The marker rule is the engine's and is the same one `reach_actor::resolve_actor_effects`
/// applies: a system is located at the marker ITS OWN `effect_locations` entry names; a system
/// that names nothing sits on the source's origin (the engine's rule for an unlocated system);
/// and a system naming a marker this model does not have **creates nothing**.
pub fn install(sc: &mut crate::scene::SceneController, want: &[FilmEffect]) -> FxReport {
    let mut rep = FxReport { requested: want.len(), ..Default::default() };
    let mut out: Vec<crate::scene::ParticleSource> = Vec::new();
    for fx in want {
        let systems = sc.effect_particle_systems(fx.effe);
        if systems.is_empty() {
            rep.no_systems += 1;
            continue;
        }
        let mut kept = Vec::new();
        let mut locs = Vec::new();
        for sy in systems.iter() {
            // A world-placed effect plays the systems authored for either view (0) and for third
            // person (2), never the first-person-only ones -- the same split the actor lane makes.
            if sy.camera_mode == 1 {
                rep.first_person_only += 1;
                continue;
            }
            let loc = if sy.loc_sid == 0 {
                // No marker named: the engine puts the system on the object's own origin.
                (Vec3::ZERO, Quat::IDENTITY)
            } else if sc.string_of(sy.loc_sid) == ROOT_LOCATION {
                // `root` is the model's ROOT NODE, not a marker. Its frame in model space is the
                // identity by construction, so this is a RESOLUTION and not a fallback -- which
                // matters, because it is the location every bullet's and every grenade's
                // travelling effect names. Measured on `forge_halo`: `assault_rifle_bullet`,
                // `sniper_rifle_bullet`, `falcon_machinegun_bullet`, `needler_shard` and
                // `plasma_grenade` all author exactly `["root"]`, and the first three carry no
                // render model at all, so there is no marker it could ever have meant.
                (Vec3::ZERO, Quat::IDENTITY)
            } else if fx.mode == 0 || fx.mode == 0xFFFF_FFFF {
                // The effect names a marker and this definition has no model to find it on.
                // Counted and skipped, never swapped for the origin -- that substitution is
                // exactly the `hms-fx-marker-fallback` bug.
                rep.marker_missing += 1;
                continue;
            } else {
                match sc.model_marker(fx.mode, sy.loc_sid) {
                    Some(l) => l,
                    None => {
                        rep.marker_missing += 1;
                        continue;
                    }
                }
            };
            kept.push(sy.clone());
            locs.push(loc);
        }
        if kept.is_empty() {
            continue;
        }
        rep.systems += kept.len();
        rep.installed += 1;
        out.push(crate::scene::ParticleSource {
            seed: fx.seed,
            world: fx.world,
            effe: fx.effe,
            label: fx.label.clone(),
            systems: std::sync::Arc::new(kept),
            locs,
            birth: Some(fx.birth),
            path: fx.path.clone(),
        });
    }
    sc.set_film_particle_sources(out);
    rep
}

// ---------------------------------------------------------------------------------------------
// Per-film tables
// ---------------------------------------------------------------------------------------------

/// What one `weap` launches and plays, resolved once per film.
#[derive(Clone, Debug, Default)]
pub struct WeaponShot {
    /// `firing effects[0].firing effect^` of barrel 0.
    pub firing_effect: Option<u32>,
    /// The weapon's own render model — what the firing effect's markers resolve against.
    pub mode: u32,
    /// The `proj` barrel 0 launches, and its flight law.
    pub projectile: Option<(u32, ProjectileDef)>,
    /// `rounds per second` of barrel 0 (initial), for the diagnostics.
    pub rounds_per_second: f32,
}

/// What one `proj` plays, resolved once per film.
#[derive(Clone, Debug, Default)]
pub struct ProjectileFx {
    pub def: ProjectileDef,
    /// The projectile's own render model (its trail's markers resolve against this).
    pub mode: u32,
    /// Its travelling attachments: `obje` +0x104 effects that are LOOPING, i.e. persistent for
    /// the life of the projectile.
    pub attached: Vec<u32>,
    /// `detonation effect (ground)` falling back to `(airborne)` — what an impact plays.
    pub detonation: Option<u32>,
    /// The tag leaf, for the diagnostics.
    pub name: String,
}

/// Everything about one film's weapons and projectiles, resolved once when the map is open.
#[derive(Clone, Debug, Default)]
pub struct ProjectileTable {
    /// `weap` tag -> what it launches.
    pub by_weapon: HashMap<u32, WeaponShot>,
    /// `proj` tag -> what it plays.
    pub by_projectile: HashMap<u32, ProjectileFx>,
    /// Weapons with a barrels block but no firing effect at all.
    pub no_firing_effect: usize,
    /// Weapons whose barrel 0 names no projectile (a melee-only or beam barrel).
    pub no_projectile: usize,
}

impl ProjectileTable {
    /// Build from the cache. `weaps` is the map's `weap` list (the film driver already has it).
    pub fn build(
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        weaps: &[u32],
        mut object_effects: impl FnMut(u32) -> Vec<u32>,
    ) -> ProjectileTable {
        let mut t = ProjectileTable::default();
        for &w in weaps {
            let barrels = crate::reach_projectile::weapon_barrels(native, cache, w);
            let Some(b) = barrels.first() else { continue };
            if b.firing_effect.is_none() {
                t.no_firing_effect += 1;
            }
            if b.projectile.is_none() {
                t.no_projectile += 1;
            }
            let proj = b.projectile.and_then(|p| {
                crate::reach_projectile::projectile_def(native, cache, p).map(|d| (p, d))
            });
            if let Some((p, d)) = proj {
                t.by_projectile.entry(p).or_insert_with(|| {
                    let name = native.tag_name(cache, p).unwrap_or_default();
                    ProjectileFx {
                        def: d,
                        mode: native.resolve_mode(cache, p),
                        attached: object_effects(p),
                        detonation: d.detonation_effect(false),
                        name: name.rsplit('\\').next().unwrap_or("").to_string(),
                    }
                });
            }
            t.by_weapon.insert(
                w,
                WeaponShot {
                    firing_effect: b.firing_effect,
                    mode: native.resolve_mode(cache, w),
                    projectile: proj,
                    rounds_per_second: b.rounds_per_second[0],
                },
            );
        }
        t
    }

    /// Add a `proj` the WEAPON list did not reach — a projectile the recording replicated whose
    /// launcher is not in the map's `weap` list (a vehicle turret's, a detonation-spawned one).
    pub fn learn_projectile(
        &mut self,
        native: &hms_native::NativeDll,
        cache: hms_native::CacheHandle,
        proj: u32,
        mut object_effects: impl FnMut(u32) -> Vec<u32>,
    ) {
        if self.by_projectile.contains_key(&proj) {
            return;
        }
        let Some(d) = crate::reach_projectile::projectile_def(native, cache, proj) else { return };
        let name = native.tag_name(cache, proj).unwrap_or_default();
        self.by_projectile.insert(
            proj,
            ProjectileFx {
                def: d,
                mode: native.resolve_mode(cache, proj),
                attached: object_effects(proj),
                detonation: d.detonation_effect(false),
                name: name.rsplit('\\').next().unwrap_or("").to_string(),
            },
        );
    }

    pub fn status(&self) -> String {
        let mut kinds: Vec<&str> = self.by_projectile.values().map(|p| p.name.as_str()).collect();
        kinds.sort_unstable();
        format!(
            "{} weapon(s) with a barrel, {} projectile definition(s); {} barrel(s) with no firing effect, {} with no projectile",
            self.by_weapon.len(),
            self.by_projectile.len(),
            self.no_firing_effect,
            self.no_projectile,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// The recorded flight of one projectile
// ---------------------------------------------------------------------------------------------

/// The recorded path of one projectile, in SCENE SECONDS, ready to hand to a moving particle
/// source.
///
/// Samples come straight from the entity track; nothing is extrapolated. A projectile whose track
/// carries one sample has a one-sample path, and the source then sits still — which is what the
/// recording says.
pub fn recorded_path(tr: &crate::film_decode::EntityTrack) -> std::sync::Arc<Vec<(f32, Vec3)>> {
    std::sync::Arc::new(
        tr.positions.iter().map(|&(t, p)| (t as f32 * TICK_SECONDS, Vec3::from(p))).collect(),
    )
}

/// The world frame a projectile's attached effect is played in at `ftick`: its recorded position,
/// and its recorded basis when it has one.
pub fn projectile_world(tr: &crate::film_decode::EntityTrack, ftick: f32) -> Option<Mat4> {
    let pos = crate::film_vehicle::position_at(tr, ftick)?;
    let orient = crate::film_vehicle::orientation_at(tr, ftick);
    Some(match orient {
        Some((f, u)) => {
            let fwd = Vec3::from(f);
            let up = Vec3::from(u);
            if fwd.length_squared() > 1e-6 && up.length_squared() > 1e-6 {
                let x = fwd.normalize();
                let z = up.normalize();
                let y = z.cross(x).normalize();
                Mat4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), pos.extend(1.0))
            } else {
                Mat4::from_translation(pos)
            }
        }
        None => Mat4::from_translation(pos),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The path sampler must be a pure function of the time, must not extrapolate, and must agree
    /// with the one the particle emitter uses — they are the SAME function, called through
    /// `scene::SceneController::path_at`, so this gate consumes the production code rather than a
    /// copy of it.
    #[test]
    fn a_recorded_path_is_sampled_not_extrapolated() {
        let p: Vec<(f32, Vec3)> = vec![
            (1.0, Vec3::new(0.0, 0.0, 0.0)),
            (2.0, Vec3::new(10.0, 0.0, 0.0)),
            (3.0, Vec3::new(10.0, 10.0, 0.0)),
        ];
        let at = |t: f32| crate::scene::SceneController::path_at(&p, t);
        // Clamped before the first and after the last sample: a trail particle born before the
        // projectile existed must not be placed somewhere the projectile never was.
        assert_eq!(at(-100.0), p[0].1);
        assert_eq!(at(0.5), p[0].1);
        assert_eq!(at(100.0), p[2].1);
        // Exactly on a sample.
        assert_eq!(at(1.0), p[0].1);
        assert_eq!(at(2.0), p[1].1);
        assert_eq!(at(3.0), p[2].1);
        // Linear between.
        assert!((at(1.5) - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
        assert!((at(2.25) - Vec3::new(10.0, 2.5, 0.0)).length() < 1e-5);
        // Deterministic.
        assert_eq!(at(1.37), at(1.37));
    }

    /// A one-sample path (a projectile that reported one position) must not panic and must sit
    /// still — the shape a dropped grenade's track has.
    #[test]
    fn a_single_sample_path_sits_still() {
        let p = vec![(5.0, Vec3::new(1.0, 2.0, 3.0))];
        for t in [-1.0f32, 0.0, 5.0, 500.0] {
            assert_eq!(crate::scene::SceneController::path_at(&p, t), p[0].1);
        }
        assert_eq!(crate::scene::SceneController::path_at(&[], 1.0), Vec3::ZERO);
    }

    /// The projectile's own basis, when the recording states one, must be the one the trail is
    /// played in — and a projectile that reports no orientation must still be placed.
    #[test]
    fn a_projectile_world_uses_the_recorded_basis_when_there_is_one() {
        let mut tr = crate::film_decode::EntityTrack {
            net_id: 1,
            entity_type: PROJECTILE_TYPE,
            first_tick: 0,
            last_tick: 10,
            positions: vec![(0, [0.0, 0.0, 0.0]), (10, [10.0, 0.0, 0.0])],
            ..Default::default()
        };
        let w = projectile_world(&tr, 5.0).expect("placed from the recorded position");
        assert!((w.w_axis.truncate() - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4, "{:?}", w.w_axis);
        // No orientation: an identity basis, not a panic and not a dropped placement.
        assert!((w.x_axis.truncate() - Vec3::X).length() < 1e-5);
        // With an orientation, the basis is the recorded one.
        tr.orientations = vec![(0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), (10, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])];
        let w = projectile_world(&tr, 5.0).expect("placed");
        assert!((w.x_axis.truncate() - Vec3::Y).length() < 1e-4, "forward should be +Y: {:?}", w.x_axis);
        assert!((w.z_axis.truncate() - Vec3::Z).length() < 1e-4, "up should be +Z: {:?}", w.z_axis);
    }
}

#[cfg(test)]
mod end_to_end {
    //! #projectile  **THE END-TO-END GATE.** A one-shot clock and a sampled path can both be
    //! perfect while nothing reaches the screen, so this drives the two production entry points
    //! the app itself calls — [`crate::film_play::ActorDriver::film_effects`] (which decides WHAT
    //! to play, from the pose) and [`super::install`] (which resolves the `effe` and installs the
    //! lane) — against the real `forge_halo` cache, and then asks the SceneController what it is
    //! actually holding.
    //!
    //! Everything DERIVED is derived by production code: the weapon/projectile table by
    //! `ProjectileTable::build`, the recorded-object table by `ObjectTable::build`, the firing
    //! ticks by `ActorDriver::action_tables`, the muzzle marker by
    //! `SceneController::model_marker`. What the gate supplies is the RECORDING — the tracks and
    //! the events — which is the one thing a fixture legitimately is.
    //!
    //! `cargo test -p hms-app --profile quick the_film_lane -- --ignored --nocapture`

    use super::*;
    use crate::film_decode::{EntityTrack, FilmIndex};
    use crate::film_play::{ActorDriver, Playback};

    const BIPED: u8 = 11;
    /// The body that fires, and the tick it fires on.
    const SHOOTER: u32 = 0x100;
    const FIRE_TICK: u32 = 90;
    /// The grenade's net id and the ticks it exists for.
    const GRENADE: u32 = 0x200;
    const GRENADE_FIRST: u32 = 80;
    const GRENADE_LAST: u32 = 120;

    fn track(net_id: u32, ty: u8, wire: u32, pts: &[(u32, [f32; 3])]) -> EntityTrack {
        EntityTrack {
            net_id,
            entity_type: ty,
            first_tick: pts.first().map(|p| p.0).unwrap_or(0),
            last_tick: pts.last().map(|p| p.0).unwrap_or(0),
            positions: pts.to_vec(),
            def_wire: Some(wire),
            moves_cached: Some(pts.len() > 1),
            ..Default::default()
        }
    }

    /// The recording: one body that fires at tick 90, and one grenade that flies from tick 80 to
    /// tick 120 along +X and then stops existing.
    fn index() -> FilmIndex {
        let mut idx = FilmIndex::default();
        idx.ticks = 300;
        idx.tracks = vec![
            track(SHOOTER, BIPED, 1, &[(0, [0.0, 0.0, 0.0]), (299, [0.0, 0.0, 0.0])]),
            track(
                GRENADE,
                PROJECTILE_TYPE,
                2,
                &[
                    (GRENADE_FIRST, [1.0, 0.0, 1.0]),
                    (100, [6.0, 0.0, 1.0]),
                    (GRENADE_LAST, [11.0, 0.0, 1.0]),
                ],
            ),
        ];
        idx.events = vec![crate::film_action::ActionEvent {
            tick: FIRE_TICK,
            kind: crate::film_action::EV_WEAPON_FIRE,
            subject: Some(SHOOTER),
            ..Default::default()
        }];
        idx
    }

    fn tag_named(native: &hms_native::NativeDll, cache: hms_native::CacheHandle, group: &str, leaf: &str) -> Option<u32> {
        native.tags_by_group(cache, group).into_iter().find(|&t| {
            native.tag_name(cache, t).map_or(false, |n| n.rsplit('\\').next() == Some(leaf))
        })
    }

    /// The whole lane, end to end: what the driver asks for, and what the scene ends up holding.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn the_film_lane_plays_a_muzzle_flash_a_trail_and_a_detonation() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else {
            eprintln!("film_projectile end-to-end SKIPPED: no forge_halo cache");
            return;
        };
        let mut sc = crate::scene::SceneController::new_shared(native.clone());
        sc.attach_cache(cache);
        let native = &*native;

        let assault_rifle = tag_named(native, cache, "weap", "assault_rifle").expect("forge_halo has an assault rifle");
        let frag = tag_named(native, cache, "proj", "frag_grenade").expect("forge_halo has a frag grenade");

        // ---- the driver's inputs, every derived one through production code ----------------
        let idx = index();
        let pb = Playback::new(idx);
        let mut d = ActorDriver::default();
        d.track_of = pb.index.tracks.iter().enumerate().map(|(i, t)| (t.net_id, i)).collect();
        let (_melee, fire) = ActorDriver::action_tables(&pb.index);
        d.fire_of = fire;
        assert_eq!(d.fire_of.get(&SHOOTER).map(|v| v.as_slice()), Some(&[FIRE_TICK][..]), "the production fold found the shot");

        let weaps = native.tags_by_group(cache, "weap");
        d.projectiles = Some(ProjectileTable::build(native, cache, &weaps, |p| {
            crate::reach_projectile::object_attachment_effects(native, cache, p).into_iter().map(|(e, _, _)| e).collect()
        }));
        {
            // The grenade is thrown by a unit, not fired from a barrel, so the weapon walk never
            // reaches it -- the same `learn_projectile` call `sync` makes.
            let t = d.projectiles.as_mut().unwrap();
            t.learn_projectile(native, cache, frag, |q| {
                crate::reach_projectile::object_attachment_effects(native, cache, q).into_iter().map(|(e, _, _)| e).collect()
            });
        }
        d.objects = Some(crate::film_objective::ObjectTable::build(
            &pb.index,
            |wire| match wire {
                2 => Some((frag, "proj".to_string(), native.tag_name(cache, frag)?)),
                _ => None,
            },
            |t| crate::reach_actor::weapon_class(native, cache, t),
            &[],
        ));
        let ot = d.objects.as_ref().unwrap();
        assert_eq!(ot.by_draw[5], 1, "the recorded grenade must be classified Draw::Projectile; buckets {:?}", ot.by_draw);

        // ---- the posed body, holding the rifle ---------------------------------------------
        let mut actors = crate::reach_actor::ActorScene::default();
        let slot = actors
            .add(native, cache, "objects\\characters\\spartans\\spartans", glam::Vec3::ZERO, 0.0)
            .expect("the Spartan biped is addable");
        {
            let a = actors.actors.last_mut().expect("one actor");
            a.pose = crate::reach_actor::Pose::bind(&a.def.model.skel);
            a.anchor = a.world();
        }
        actors.set_weapon_tag(native, cache, slot, Some(assault_rifle)).expect("the rifle goes in the hand");
        d.body_actor.insert(SHOOTER, slot);
        let (wmode, wworld) = {
            let a = actors.actors.iter().find(|a| a.id == slot).expect("the actor");
            let w = a.weapon.as_ref().expect("a weapon in the hand");
            (w.mode_tag, a.world() * w.world(glam::Mat4::IDENTITY, &a.pose))
        };

        // ---- THE PRODUCTION PRODUCER, at a tick 10 ticks after the shot ---------------------
        let mut pb = pb;
        pb.seek_tick((FIRE_TICK + 10) as f32);
        let want = d.film_effects(&pb, &actors, native, cache);
        println!(
            "film_effects at tick {}: {} request(s) -- {} muzzle, {} trail, {} detonation (unarmed {}, no effect {}, model-less {})",
            pb.tick, want.len(), d.fx_muzzle, d.fx_trail, d.fx_detonation, d.fx_unarmed, d.fx_no_effect, d.fx_modelless
        );
        for f in &want {
            println!("   {} effe {:#x} birth {:.3}s mode {:#x} path {:?}", f.label, f.effe, f.birth, f.mode, f.path.as_ref().map(|p| p.len()));
        }
        assert_eq!(d.fx_unarmed, 0, "the body is holding the rifle, so no shot may be dropped as unarmed");
        assert_eq!(d.fx_no_effect, 0, "the assault rifle authors a firing effect");

        // 1. THE MUZZLE FLASH.
        let barrel = crate::reach_projectile::weapon_barrels(native, cache, assault_rifle);
        let want_effe = barrel[0].firing_effect.expect("the rifle's barrel authors a firing effect");
        let muzzle = want.iter().find(|f| f.label.starts_with("muzzle")).expect("a muzzle flash was requested");
        assert_eq!(muzzle.effe, want_effe, "the flash must be the weapon's own firing effect");
        assert_eq!(muzzle.mode, wmode, "its markers must resolve against the WEAPON's render model");
        assert!(
            (muzzle.birth - FIRE_TICK as f32 / crate::film_play::TICK_HZ).abs() < 1e-6,
            "the flash is born at the firing tick, not at the playhead: birth {} vs {}",
            muzzle.birth,
            FIRE_TICK as f32 / crate::film_play::TICK_HZ
        );
        // The gun's frame, not the body's. A rifle held by a Spartan standing at the origin is
        // about half a body height up and a hand's reach out -- so this is both "not the origin"
        // and "not somewhere absurd".
        let mp = muzzle.world.w_axis.truncate();
        assert!((mp - wworld.w_axis.truncate()).length() < 1e-4, "the flash is not on the gun: {mp:?}");
        assert!(mp.length() > 0.05, "the flash is at the body origin, so it is not on the gun at all: {mp:?}");
        assert!(mp.length() < 1.0, "the flash is {:.2} wu from a Spartan standing at the origin", mp.length());
        assert!(muzzle.path.is_none(), "a muzzle flash does not move");

        // 2. THE TRAIL, on the recorded flight.
        let trail = want.iter().find(|f| f.label.starts_with("trail")).expect("a travelling effect was requested");
        let path = trail.path.as_deref().expect("the trail is a MOVING source");
        assert_eq!(path.len(), 3, "the path is the recorded samples, nothing added: {path:?}");
        assert!((path[0].0 - GRENADE_FIRST as f32 / crate::film_play::TICK_HZ).abs() < 1e-6, "path starts at the track's first tick");
        assert!((path[2].1 - glam::Vec3::new(11.0, 0.0, 1.0)).length() < 1e-4, "path ends where the track does");
        // It is at the recorded position for THIS tick, interpolated -- tick 100 of a 80..120
        // flight is 6.0 wu out, and tick 100 is where the playhead is.
        assert!(
            (trail.world.w_axis.truncate() - glam::Vec3::new(6.0, 0.0, 1.0)).length() < 0.2,
            "the trail source is at {:?}, not at the grenade's recorded position for tick {}",
            trail.world.w_axis.truncate(),
            pb.tick
        );

        // 3. THE DETONATION, at the end of the flight, once the flight is over.
        assert!(
            !want.iter().any(|f| f.label.starts_with("detonation")),
            "the grenade has not detonated yet at tick {} -- its track runs to {GRENADE_LAST}",
            pb.tick
        );
        pb.seek_tick((GRENADE_LAST + 5) as f32);
        let later = d.film_effects(&pb, &actors, native, cache);
        let det = later.iter().find(|f| f.label.starts_with("detonation")).expect("the grenade detonates at the end of its track");
        let fx = d.projectiles.as_ref().unwrap().by_projectile.get(&frag).expect("the frag was learned");
        assert_eq!(det.effe, fx.detonation.expect("the frag authors a detonation effect"), "the explosion is the proj's own effect");
        assert!(
            (det.world.w_axis.truncate() - glam::Vec3::new(11.0, 0.0, 1.0)).length() < 1e-4,
            "the explosion must be at the last position the track reports, got {:?}",
            det.world.w_axis.truncate()
        );
        assert!((det.birth - GRENADE_LAST as f32 / crate::film_play::TICK_HZ).abs() < 1e-6, "it is born when the track ends");

        // ---- THE PRODUCTION INSTALLER, and what the scene ends up holding -------------------
        let rep = install(&mut sc, &want);
        println!("install: {}", rep.line());
        assert_eq!(rep.requested, want.len());
        assert!(rep.installed >= 2, "the muzzle flash and the trail must both install: {}", rep.line());
        assert!(rep.systems > 0, "no particle system was installed at all: {}", rep.line());
        assert_eq!(sc.film_particle_source_count(), rep.installed, "the scene is holding what install reported");

        // The muzzle flash's markers must be the WEAPON's own, resolved by the production
        // resolver -- consumed, not re-derived, so this cannot disagree with the lane.
        let systems = sc.effect_particle_systems(want_effe);
        let mut matched = 0usize;
        for sy in systems.iter().filter(|s| s.camera_mode != 1 && s.loc_sid != 0) {
            let name = sc.string_of(sy.loc_sid);
            if name == ROOT_LOCATION {
                continue;
            }
            if let Some(m) = sc.model_marker(wmode, sy.loc_sid) {
                matched += 1;
                let at = muzzle.world.transform_point3(m.0);
                assert!(at.is_finite(), "marker '{name}' put the flash at {at:?}");
                println!("   location '{name}' -> ({:.3},{:.3},{:.3})", at.x, at.y, at.z);
            }
        }
        assert!(matched > 0, "not one of the firing effect's locations is a marker of the rifle's model");

        // THE FALLBACK RULE, which is the `hms-fx-marker-fallback` bug: the SAME effect with no
        // model to resolve against must install NOTHING for a system that names a marker, and
        // must COUNT it -- never quietly move it to the origin.
        let orphan = vec![FilmEffect { mode: 0, ..muzzle.clone() }];
        let rep2 = install(&mut sc, &orphan);
        println!("install with no model: {}", rep2.line());
        assert!(rep2.marker_missing > 0, "a named marker with no model to find it on must be COUNTED: {}", rep2.line());
        let named = systems.iter().filter(|s| s.camera_mode != 1 && s.loc_sid != 0 && sc.string_of(s.loc_sid) != ROOT_LOCATION).count();
        assert_eq!(
            rep2.marker_missing, named,
            "every one of the {named} marker-named system(s) must be dropped, not just some: {}",
            rep2.line()
        );
    }
}
