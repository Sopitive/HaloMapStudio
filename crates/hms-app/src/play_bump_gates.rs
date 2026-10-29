//! #bump-gate  Touching a parked vehicle or a loose prop must not drop it through the floor, and
//! the player must not walk through it.
//!
//! Reported: *"If I so much as bump into any object it just falls through all the other Forge
//! objects. So I go to get in a vehicle and it falls down (the vehicle I bump). Sometimes I even
//! walk straight through the vehicle."*
//!
//! The gates here walk a Spartan into a parked Mongoose / Warthog and loose crates through the REAL
//! `PlayMode::tick` (`Ride::tick_in`): the rigid-body world (`PropSession::tick` -> `PropWorld`), the
//! biped controller against its gathered `PlayWorld`, the parked-vehicle path, all against a REAL
//! `SceneController` (the map's engine collision) with PLACED Forge blocks (`ff_plat_*`) as the
//! support, resting on the block or on the terrain. Nothing between the input and the object's pose
//! is re-implemented here.
#![cfg(test)]

use glam::{Quat, Vec3};
use hms_ipc::ObjectInfo;

use crate::objscene::ObjectScene;
use crate::play_ctl::MoveInput;
use crate::play_ledge_gates::{ground_at, level_seeds, place_platforms, spread, MapScene};
use crate::play_mode::ride_harness::Ride;
use crate::scene::SceneController;

/// How the player walks into it.
#[derive(Clone, Copy, Debug)]
pub enum Gait {
    Walk,
    Run,
    Sprint,
    Jump,
}

/// What the object rests on. `kind` is the END of the block's tag path.
#[derive(Clone, Copy, Debug)]
pub enum Support {
    /// The terrain, at a flat spot.
    Terrain,
    /// A 3 x 3 grid of `ff_plat_4x4`, placement Normal (movable props) or Fixed (terrain).
    Platform { fixed: bool, kind: &'static str },
}

pub struct Bump {
    /// The object at the origin of the scenario (a vehicle tag suffix).
    pub vehicle: &'static str,
    pub support: Support,
    pub gait: Gait,
    /// Direction the player comes from, radians.
    pub from: f32,
    /// A loose crate in the player's path, 3.5 wu before the vehicle.
    pub crate_in_path: bool,
    /// How far into the support the vehicle is placed (0 = dropped from 0.6 above).
    pub sunk: f32,
}

/// What one scenario recorded.
#[derive(Default, Debug)]
pub struct Outcome {
    pub ticks: usize,
    /// Rest height of the vehicle origin after the 5 s settle.
    pub rest_z: f32,
    /// The vehicle's lowest origin height while the player bumped it, relative to rest.
    pub worst_drop: f32,
    /// How far the vehicle was moved by the bumps, wu.
    pub moved: f32,
    /// Deepest the player's axis got inside the vehicle's hull, wu (<= 0 = never inside).
    pub deepest_inside: f32,
    /// The player touched the vehicle (liveness): closest approach of the capsule axis to the hull.
    pub closest: f32,
    pub crate_worst_drop: f32,
    /// The controller's capsule radius, wu.
    pub radius: f32,
    /// The vehicle's rest height above the support top it was placed on (origin - top).
    pub rest_above_top: f32,
    /// The scenario had a crate in the player's path / the player was not walking slowly.
    pub crate_in_path: bool,
    pub fast: bool,
    pub notes: Vec<String>,
}

fn pose_of(r: &Ride, datum: u32) -> Option<(Vec3, Quat)> {
    r.pm.props.world().and_then(|w| w.body_pose(datum))
}

fn object_at(base: &ObjectInfo, origin: Vec3, rot: Quat) -> ObjectInfo {
    let mut o = *base;
    o.pos = origin.to_array();
    o.fwd = (rot * Vec3::X).to_array();
    o.up = (rot * Vec3::Z).to_array();
    o
}

/// Signed distance from `p` to the vehicle's hull shapes at pose (`origin`, `rot`): the largest
/// plane distance of a convex hull minus its radius, so negative = inside that hull. The least
/// over the hulls (the support spheres are not hull).
fn hull_sd(body: &crate::physics_settle::PhysBody, origin: Vec3, rot: Quat, p: Vec3) -> f32 {
    use crate::physics_settle::ShapeGeom;
    let local = rot.inverse() * (p - origin);
    let mut best = f32::MAX;
    for s in body.shapes.iter().filter(|s| !s.support) {
        if let ShapeGeom::Hull { planes, r, .. } = &s.geom {
            let sd = planes.iter().map(|(n, d)| n.dot(local) - d).fold(f32::MIN, f32::max) - r;
            best = best.min(sd);
        }
    }
    best
}

thread_local! {
    /// The placement byte the `fixed` platforms of [`run_bump_only`] get: 0x40 (Fixed); a gate sets 0xC0 (Phased).
    static FIXED_PLATFORM_PLACEMENT: std::cell::Cell<u8> = const { std::cell::Cell::new(0x40) };
}

pub fn run_bump(stem: &str, cases: &[Bump], settle_s: f32, bump_s: f32, verbose: bool) -> Vec<(String, Outcome)> {
    run_bump_only(stem, cases, settle_s, bump_s, verbose, &|_| true)
}

/// [`run_bump`] for the cases `keep` accepts; every case keeps the spot its INDEX in `cases` gives it.
pub fn run_bump_only(stem: &str, cases: &[Bump], settle_s: f32, bump_s: f32, verbose: bool, keep: &dyn Fn(usize) -> bool) -> Vec<(String, Outcome)> {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(stem) else {
        eprintln!("bump gates SKIPPED: no {stem} cache");
        return Vec::new();
    };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let seeds = spread(level_seeds(&native, cache), 4000);
    let level = MapScene::new(&sc, Vec::new());
    // a loose prop with normal physics: the first crate / barrel / box this map has whose phmo moves
    let probe = |tag: u32| ObjectInfo { datum: 0x7777, type_sig: 0, sig0: 0, sig1: 0, pos: [0.0; 3], health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: tag, variant_name_sid: 0 };
    let mut loose: Vec<(u32, String)> = native.tags_by_group(cache, "bloc").into_iter().filter_map(|t| native.tag_name(cache, t).map(|n| (t, n))).filter(|(_, n)| n.contains("crate") || n.contains("barrel") || n.contains("box")).collect();
    loose.sort_by_key(|(_, n)| (!n.contains("crate"), n.len()));
    let crate_tag = loose
        .iter()
        .find(|(t, _)| sc.physics_body(&probe(*t)).is_some_and(|b| !b.fixed && b.mass > 10.0 && b.mass < 3000.0))
        .map(|(t, n)| {
            eprintln!("loose prop: {n}");
            *t
        })
        .expect("a loose prop with normal physics");
    let mut results = Vec::new();
    for (ci, case) in cases.iter().enumerate() {
        if !keep(ci) {
            continue;
        }
        // --- the support ---
        let (mut blocks, centre, top) = match case.support {
            Support::Terrain => {
                let Some(spot) = seeds.iter().find(|s| {
                    // flat 14 x 14 around it
                    (-2..=2).all(|i| (-2..=2).all(|j| ground_at(&level, s.x + i as f32 * 3.0, s.y + j as f32 * 3.0, s.z).is_some_and(|z| (z - s.z).abs() < 0.1)))
                }) else { continue };
                (Vec::new(), Vec3::new(spot.x, spot.y, 0.0), ground_at(&level, spot.x, spot.y, spot.z).unwrap_or(spot.z))
            }
            Support::Platform { kind, .. } => {
                // enough blocks to cover about 12 x 12 wu
                let probe = place_platforms(&sc, &native, cache, &seeds[ci.min(seeds.len() - 1)..], &[kind], 1);
                let Some(first) = probe.first().copied() else { continue };
                let t0 = sc.physics_world_tris(&first);
                let (mut lo0, mut hi0) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for p in t0.iter().flatten() {
                    lo0 = lo0.min(*p);
                    hi0 = hi0.max(*p);
                }
                let n = ((12.0 / (hi0.x - lo0.x).max(0.5)).ceil() as i32).clamp(1, 12);
                let b = place_platforms(&sc, &native, cache, &seeds[ci.min(seeds.len() - 1)..], &[kind], n);
                let b: Vec<ObjectInfo> = b.into_iter().take((n * n) as usize).collect();
                if b.len() < (n * n) as usize {
                    continue;
                }
                let tris = MapScene::new(&sc, b.clone());
                let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for o in &b {
                    for p in tris.tris[&o.datum].iter().flatten() {
                        lo = lo.min(*p);
                        hi = hi.max(*p);
                    }
                }
                (b, Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, 0.0), hi.z)
            }
        };
        // loose crate: its lowest point on the support
        let crate_base = ObjectInfo { datum: 0x7777, type_sig: 0, sig0: 0, sig1: 0, pos: [0.0; 3], health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: crate_tag, variant_name_sid: 0 };
        let ctris = sc.physics_world_tris(&crate_base);
        let clo = ctris.iter().flatten().fold(Vec3::splat(f32::MAX), |a, p| a.min(*p));
        let dir = Vec3::new(case.from.cos(), case.from.sin(), 0.0);
        let mut extras = blocks.clone();
        let mut crate_o = crate_base;
        if case.crate_in_path {
            let at = centre + dir * 3.0;
            crate_o.pos = [at.x, at.y, top - clo.z + 0.02];
            extras.push(crate_o);
        }
        let vpos = Vec3::new(centre.x, centre.y, top + 0.6 - case.sunk);
        let mut r = match Ride::with_vehicle_placed(native.clone(), cache, case.vehicle, vpos, 0.3, &extras) {
            Some(r) => r,
            None => continue,
        };
        let scene = MapScene::new(&sc, r.objects.clone());
        // the first frame of `with_vehicle` ran against the flat test floor: forget that world
        r.pm.props.leave(&mut r.actors);
        r.pm.world.invalidate();
        if let Support::Platform { fixed: true, .. } = case.support {
            for b in &blocks {
                r.pm.props.placement.insert(b.datum, FIXED_PLATFORM_PLACEMENT.with(|c| c.get()));
            }
        }
        blocks.clear();
        let vdatum = 0x2468u32;
        let vbase = r.objects[0];
        let vbody = sc.physics_body(&vbase).expect("vehicle body");
        let feet = centre + dir * 4.8;
        let yaw = (-dir.y).atan2(-dir.x);
        let start_feet = Vec3::new(feet.x, feet.y, top);
        // the player waits far from it while it settles
        r.pm.st = crate::play_ctl::PlayerState::standing_at(Vec3::new(centre.x - dir.x * 40.0, centre.y - dir.y * 40.0, ground_at(&level, centre.x - dir.x * 40.0, centre.y - dir.y * 40.0, top).unwrap_or(top)), yaw);
        let idle = MoveInput { yaw, ..Default::default() };
        let mut out = Outcome::default();
        // one frame builds the rigid-body world; the vehicle is then dropped from 0.6 wu up so it
        // rests where its wheels put it instead of where this harness placed it
        r.tick_in(&idle, &scene);
        let objs = r.objects.clone();
        let (nat, ch) = (r.native.clone(), r.cache);
        assert!(r.pm.props.wake_body(vdatum, &nat, ch, &scene, &objs), "the vehicle is a simulated body");
        for _ in 0..(settle_s * 60.0) as usize {
            r.tick_in(&idle, &scene);
        }
        let Some((rest_o, _)) = pose_of(&r, vdatum).or_else(|| Some((Vec3::from(vbase.pos), Quat::IDENTITY))) else { continue };
        out.rest_z = rest_o.z;
        out.rest_above_top = rest_o.z - top;
        let crate_rest = pose_of(&r, 0x7777).map(|p| p.0.z);
        // --- the bump ---
        r.pm.st = crate::play_ctl::PlayerState::standing_at(start_feet, yaw);
        r.pm.world.invalidate();
        let mut input = MoveInput { fwd: 1.0, yaw, ..Default::default() };
        match case.gait {
            Gait::Walk => input.walk = true,
            Gait::Sprint => input.sprint = true,
            _ => {}
        }
        out.worst_drop = 0.0;
        out.radius = r.pm.phys.radius;
        out.crate_in_path = case.crate_in_path;
        out.fast = !matches!(case.gait, Gait::Walk);
        out.deepest_inside = f32::MIN;
        out.closest = f32::MAX;
        let n = (bump_s * 60.0) as usize;
        for k in 0..n {
            input.jump = matches!(case.gait, Gait::Jump) && k % 40 == 0;
            r.tick_in(&input, &scene);
            out.ticks += 1;
            let (vo, vq) = pose_of(&r, vdatum).unwrap_or((rest_o, Quat::IDENTITY));
            out.worst_drop = out.worst_drop.min(vo.z - rest_o.z);
            out.moved = out.moved.max((vo - rest_o).truncate().length());
            if let Some(cz) = crate_rest {
                if let Some((co, _)) = pose_of(&r, 0x7777) {
                    out.crate_worst_drop = out.crate_worst_drop.min(co.z - cz);
                }
            }
            // the player's axis against the vehicle's hull at its CURRENT pose
            let h = r.pm.st.height(&r.pm.phys);
            let f = r.pm.st.pos;
            for t in [0.3f32, 0.5, 0.8] {
                let p = f + Vec3::Z * (h * t);
                let sd = hull_sd(&vbody, vo, vq, p);
                out.deepest_inside = out.deepest_inside.max(-sd);
                out.closest = out.closest.min(sd);
            }
            if verbose && k % 30 == 0 {
                eprintln!("  t{k} biped ({:.2},{:.2},{:.2}) vehicle ({:.2},{:.2},{:.2}) crate {:?} {}", f.x, f.y, f.z, vo.x, vo.y, vo.z, pose_of(&r, 0x7777).map(|c| c.0), r.pm.props.note);
                if k == 0 { eprintln!("  world: {}", r.pm.world.describe()); }
            }
        }
        results.push((format!("{} {:?} {:?} from {:.2} crate {} sunk {:.1} top {:.2}", case.vehicle, case.support, case.gait, case.from, case.crate_in_path, case.sunk, top), out));
    }
    results
}

/// The scenarios: both vehicles, three supports, four gaits, four approach directions, with a
/// loose crate in the player's path for half of them.
pub fn matrix() -> Vec<Bump> {
    let mut v = Vec::new();
    for vehicle in ["mongoose", "warthog"] {
        for support in [Support::Terrain, Support::Platform { fixed: false, kind: "ff_plat_4x4" }, Support::Platform { fixed: true, kind: "ff_plat_4x4" }] {
            for gait in [Gait::Walk, Gait::Run, Gait::Sprint, Gait::Jump] {
                for (k, from) in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_4, std::f32::consts::PI].into_iter().enumerate() {
                    v.push(Bump { vehicle, support, gait, from, crate_in_path: k % 2 == 1, sunk: 0.0 });
                }
            }
        }
    }
    v
}

/// Thin, low and small platforms, where a vertex-against-triangle contact has the least to hold on to
/// and a heavy vehicle stands on light blocks.
pub fn thin_matrix() -> Vec<Bump> {
    let mut v = Vec::new();
    for vehicle in ["mongoose", "warthog"] {
        for kind in ["ff_plat_4x4_short", "ff_plat_5x5_flat", "ff_plat_2x2_flat", "ff_plat_1x1", "ff_plat_1x2", "ff_plat_2x2"] {
            for fixed in [false, true] {
                for gait in [Gait::Walk, Gait::Run] {
                    for (k, from) in [0.0f32, std::f32::consts::FRAC_PI_2].into_iter().enumerate() {
                        v.push(Bump { vehicle, support: Support::Platform { fixed, kind }, gait, from, crate_in_path: k == 1, sunk: 0.0 });
                    }
                }
            }
        }
    }
    v
}

/// A vehicle still on its platform: it was not pushed far enough to have gone over an edge, so any
/// drop it records is a fall THROUGH something.
const STILL_ON_SUPPORT: f32 = 3.5;
/// The most a body may sink into what it stands on while it is bumped, wu: the engine rests a
/// CRITICAL pair at 0.2 of `allowedPenetrationDepth` (`03-continuous-toi.md`), 0.4 * 0.7 for these
/// hulls, with the position push's slack.
const MAX_SINK: f32 = 0.1;

/// Check a batch of outcomes; `Err` lists every violation.
pub fn judge(res: &[(String, Outcome)], expect_runs: usize) -> Result<(), String> {
    let mut bad = Vec::new();
    if res.len() < expect_runs {
        bad.push(format!("liveness: only {} of the expected {expect_runs} scenarios ran", res.len()));
    }
    let reached = res.iter().filter(|(_, o)| o.closest < o.radius + 0.4).count();
    if reached * 3 < res.len() {
        bad.push(format!("liveness: the player only reached the vehicle in {reached} of {} runs", res.len()));
    }
    // the bump PUSHES: a player running into a loose vehicle moves it (the biped is a dynamic body)
    let fast: Vec<_> = res.iter().filter(|(_, o)| o.fast && !o.crate_in_path && o.closest < o.radius + 0.4).collect();
    let pushed = fast.iter().filter(|(_, o)| o.moved > 0.1).count();
    if fast.len() >= 4 && pushed * 5 < fast.len() * 4 {
        bad.push(format!("the player moved the vehicle in only {pushed} of {} running bumps", fast.len()));
    }
    for (name, o) in res {
        // (B) the player is stopped by it: the capsule's axis stays outside the hull by about the
        // capsule's radius (less its slop), never inside
        if o.closest < o.radius - 0.1 {
            bad.push(format!("{name}: the player's axis got to {:.2} of the hull (radius {:.2}): walked into it", o.closest, o.radius));
        }
        // (A) it came to rest ON the support (its origin is a few cm above the top, never below it),
        // and stays on what it was standing on
        if o.rest_above_top < -MAX_SINK {
            bad.push(format!("{name}: the vehicle settled {:.2} wu below the top of its support", -o.rest_above_top));
        }
        if o.moved < STILL_ON_SUPPORT && o.worst_drop < -MAX_SINK {
            bad.push(format!("{name}: the vehicle dropped {:.2} wu through its support", o.worst_drop));
        }
        if o.crate_worst_drop < -MAX_SINK {
            bad.push(format!("{name}: the crate dropped {:.2} wu through its support", o.crate_worst_drop));
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(bad.join("\n")) }
}

#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn walking_into_a_parked_vehicle_stops_the_player_and_never_drops_the_vehicle() {
    let cases = matrix();
    let res = run_bump("forge_halo", &cases, 6.0, 10.0, false);
    for (name, o) in &res {
        eprintln!("BUMP {name}: drop {:.2} crate {:.2} moved {:.2} closest {:.2}", o.worst_drop, o.crate_worst_drop, o.moved, o.closest);
    }
    if let Err(e) = judge(&res, cases.len() - 4) {
        panic!("{e}");
    }
}

#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn bumping_a_vehicle_on_thin_or_small_blocks_never_drops_it() {
    let cases = thin_matrix();
    let res = run_bump("forge_halo", &cases, 6.0, 8.0, false);
    for (name, o) in &res {
        eprintln!("BUMP {name}: drop {:.2} crate {:.2} moved {:.2} closest {:.2}", o.worst_drop, o.crate_worst_drop, o.moved, o.closest);
    }
    if let Err(e) = judge(&res, cases.len() - 6) {
        panic!("{e}");
    }
}

/// Every placed object of the map's own scenario, as the editor lists it.
fn scenario_infos(sc: &SceneController) -> Vec<ObjectInfo> {
    sc.scenario_objects()
        .iter()
        .enumerate()
        .filter(|(_, o)| o.placement_flags & 0x40 == 0)
        .map(|(i, o)| ObjectInfo { datum: 0x100 + i as u32, type_sig: 0, sig0: 0, sig1: 0, pos: o.pos, health: 1.0, shield: 1.0, mode_tag: o.mode_tag, fwd: o.fwd, up: o.up, attached: [0; 8], primary_tag: o.obj_tag, variant_name_sid: 0 })
        .collect()
}

/// The most a woken object may end up below where the map placed it, wu: a desk or a rock slope is
/// a few wu; falling THROUGH the floor is the whole height of the map.
const MAX_FALL: f32 = 4.0;

/// Wake every normal-physics object the map places (what the first bump does to one of them) and let
/// it settle for `secs`; none may end up through the floor.
fn wake_all(stem: &str, secs: f32) {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(stem) else {
        eprintln!("bump gates SKIPPED: no {stem} cache");
        return;
    };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let objs = scenario_infos(&sc);
    let scene = MapScene::new(&sc, objs.clone());
    let mut props = crate::play_world::PropSession::default();
    let mut actors = crate::reach_actor::ActorScene::default();
    props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    let mut woken = Vec::new();
    for o in &objs {
        if props.wake_body(o.datum, &native, cache, &scene, &objs) {
            woken.push(*o);
        }
    }
    for _ in 0..(secs * 60.0) as usize {
        props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    }
    let w = props.world().expect("world");
    let mut fell = 0;
    for o in &woken {
        let Some((p, _)) = w.body_pose(o.datum) else { continue };
        let drop = o.pos[2] - p.z;
        if drop > 0.25 {
            fell += 1;
            if fell <= 25 {
                let from = Vec3::new(o.pos[0], o.pos[1], o.pos[2] + 0.3);
                let with_objects = scene.raycast_solid_excluding(from, -Vec3::Z, 200.0, &[o.datum]).map(|d| from.z - d);
                let level_only = sc.raycast_solid_excluding(from, -Vec3::Z, 200.0, &[]).map(|d| from.z - d);
                // the object whose hull holds the support, if the objects hold it
                let under = objs.iter().filter(|u| u.datum != o.datum).find(|u| scene.tris.get(&u.datum).is_some_and(|t| crate::play_world::ray_down(t, from, 200.0).is_some_and(|(d, _)| with_objects.is_some_and(|z| ((from.z - d) - z).abs() < 0.02)))).map(|u| format!("{} [{:?}] mode {:?}", native.tag_name(cache, u.primary_tag).unwrap_or_default(), native.tag_class(cache, u.primary_tag), props.world().and_then(|w| w.kind_of(u.datum))));
                eprintln!("      support: level only {level_only:?}, with objects {with_objects:?}, object under: {under:?}");
                eprintln!("FELL[{stem}] {} class {:?} from z {:.2} to {:.2} (drop {:.2}) at ({:.1},{:.1})", native.tag_name(cache, o.primary_tag).unwrap_or_default(), native.tag_class(cache, o.primary_tag), o.pos[2], p.z, drop, o.pos[0], o.pos[1]);
            }
        }
    }
    eprintln!("WAKEALL[{stem}] {} objects placed, {} simulated, {} fell more than 0.25 wu", objs.len(), woken.len(), fell);
    let through: Vec<String> = woken
        .iter()
        .filter_map(|o| {
            let p = w.body_pose(o.datum)?.0;
            (o.pos[2] - p.z > MAX_FALL || !p.is_finite()).then(|| format!("{} fell {:.1} wu through the floor at ({:.1},{:.1},{:.1})", native.tag_name(cache, o.primary_tag).unwrap_or_default(), o.pos[2] - p.z, o.pos[0], o.pos[1], o.pos[2]))
        })
        .collect();
    assert!(woken.len() >= 30, "liveness: only {} objects of {stem} are simulated bodies", woken.len());
    assert!(through.is_empty(), "{stem}: {} woken objects fell through the floor:\n{}", through.len(), through.join("\n"));
}

#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn woken_props_stay_on_the_floor_island() { wake_all("35_island", 8.0); }
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn woken_props_stay_on_the_floor_sword_base() { wake_all("20_sword_slayer", 8.0); }
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn woken_props_stay_on_the_floor_ivory_tower() { wake_all("52_ivory_tower", 8.0); }
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn woken_props_stay_on_the_floor_settlement() { wake_all("30_settlement", 8.0); }


/// One-off class survey: every Forge-world piece the map offers, placed alone with a weapon on its
/// top, the weapon woken. Returns `(piece name, family, weapon drop, tris of the piece)` per piece.
fn piece_survey(stem: &str, name_filter: &dyn Fn(&str) -> bool, piece_mode: u8, secs: f32, orient: u8, pick_scale: f32, piece_scale: f32) -> Vec<(String, f32, usize, bool)> {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(stem) else { return Vec::new() };
    piece_survey_in(&native, cache, name_filter, piece_mode, secs, orient, pick_scale, piece_scale)
}

/// [`piece_survey`] on a map the caller already holds open (the native cache lock is not re-entrant).
#[allow(clippy::too_many_arguments)]
fn piece_survey_in(native: &std::sync::Arc<hms_native::NativeDll>, cache: hms_native::CacheHandle, name_filter: &dyn Fn(&str) -> bool, piece_mode: u8, secs: f32, orient: u8, pick_scale: f32, piece_scale: f32) -> Vec<(String, f32, usize, bool)> {
    piece_survey_loose(native, cache, name_filter, piece_mode, secs, orient, pick_scale, piece_scale, "assault_rifle\\assault_rifle")
}

/// [`piece_survey_in`] with the loose body being the first non-fixed `weap` / `eqip` / `bloc` / `scen` whose name ends in `loose` (a rifle,
/// the soccer ball, ...).
#[allow(clippy::too_many_arguments)]
fn piece_survey_loose(native: &std::sync::Arc<hms_native::NativeDll>, cache: hms_native::CacheHandle, name_filter: &dyn Fn(&str) -> bool, piece_mode: u8, secs: f32, orient: u8, pick_scale: f32, piece_scale: f32, loose: &str) -> Vec<(String, f32, usize, bool)> {
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let weapon = ["weap", "eqip", "bloc", "scen"]
        .iter()
        .flat_map(|g| native.tags_by_group(cache, g))
        .find(|t| native.tag_name(cache, *t).is_some_and(|n| n.ends_with(loose)))
        .expect("the loose body");
    let mk = |datum: u32, tag: u32, pos: Vec3| ObjectInfo { datum, type_sig: 0, sig0: 0, sig1: 0, pos: pos.to_array(), health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: tag, variant_name_sid: 0 };
    let wlo = sc.physics_world_tris(&mk(0, weapon, Vec3::ZERO)).iter().flatten().fold(Vec3::splat(f32::MAX), |a, p| a.min(*p));
    let mut pieces: Vec<(u32, String)> = Vec::new();
    for g in ["bloc", "scen", "mach", "ctrl"] {
        for t in native.tags_by_group(cache, g) {
            if let Some(n) = native.tag_name(cache, t) {
                // a piece the biped passes through by design (respawn zones, one-team doors) is not one
                // a weapon is expected to rest on either
                if name_filter(&n) && crate::reach_collision::collides_with_biped(&native, cache, t) {
                    pieces.push((t, n));
                }
            }
        }
    }
    let mut objs: Vec<ObjectInfo> = Vec::new();
    let mut weapons: Vec<(u32, usize, f32)> = Vec::new(); // datum, piece index, start z
    let mut names: Vec<(String, usize)> = Vec::new();
    let mut bounds: std::collections::HashMap<usize, (Vec3, Vec3, f32)> = std::collections::HashMap::new();
    let mut piece_tris: std::collections::HashMap<usize, Vec<[Vec3; 3]>> = std::collections::HashMap::new();
    if piece_scale != 1.0 {
        sc.set_object_scales((0..pieces.len()).map(|i| (0x1000 + i as u32 * 8, piece_scale)).collect());
    }
    for (i, (tag, name)) in pieces.iter().enumerate() {
        let at = Vec3::new(5000.0 + (i % 20) as f32 * 60.0, 5000.0 + (i / 20) as f32 * 60.0, 500.0);
        let mut piece = mk(0x1000 + i as u32 * 8, *tag, at);
        match orient {
            1 => {
                let (sn, cs) = 0.6f32.sin_cos();
                piece.fwd = [cs, sn, 0.0];
            }
            2 => piece.up = [0.0, 1.0, 0.0],
            3 => {
                // pitched 25 degrees about y
                let (sn, cs) = 0.4363f32.sin_cos();
                piece.fwd = [cs, 0.0, -sn];
                piece.up = [sn, 0.0, cs];
            }
            _ => {}
        }
        let tris = sc.physics_world_tris(&piece);
        if tris.len() < 4 {
            continue;
        }
        // the highest up-facing spot over a few probes across the piece's top
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in tris.iter().flatten() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let mut best: Option<Vec3> = None;
        for (fx, fy) in [(0.5f32, 0.5f32), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7)] {
            let o = Vec3::new(lo.x + (hi.x - lo.x) * fx, lo.y + (hi.y - lo.y) * fy, hi.z + 1.0);
            if let Some((d, n)) = crate::play_world::ray_down(&tris, o, hi.z - lo.z + 2.0) {
                if n.z > 0.95 && best.map_or(true, |b| o.z - d > b.z) {
                    best = Some(Vec3::new(o.x, o.y, o.z - d));
                }
            }
        }
        let Some(top) = best else { continue };
        bounds.insert(i, (lo - Vec3::new(0.0, 0.0, 0.0) , hi, top.z));
        piece_tris.insert(i, tris.clone());
        objs.push(piece);
        // resting on it, a few cm into it, the way the editor drops one (origin ON the surface, so the
        // body's own underside is below it), and a hand's width into it
        for (k, dz) in [0.03f32, -0.03, wlo.z, wlo.z - 0.1].into_iter().enumerate() {
            let wd = 0x1000 + i as u32 * 8 + 1 + k as u32;
            objs.push(mk(wd, weapon, Vec3::new(top.x, top.y, top.z - wlo.z + dz)));
            weapons.push((wd, i, top.z - wlo.z + dz));
        }
        names.push((name.clone(), tris.len()));
    }
    let mut scene = MapScene::new(&sc, objs.clone());
    scene.pick_scale = pick_scale;
    let mut props = crate::play_world::PropSession::default();
    for o in &objs {
        if o.primary_tag != weapon {
            props.placement.insert(o.datum, piece_mode);
        }
    }
    let mut actors = crate::reach_actor::ActorScene::default();
    props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    for (d, _, _) in &weapons {
        props.wake_body(*d, &native, cache, &scene, &objs);
    }
    for _ in 0..(secs * 60.0) as usize {
        props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    }
    let w = props.world().expect("world");
    let mut out = Vec::new();
    for (wd, i, z0) in weapons {
        let pose = w.body_pose(wd).map(|p| p.0);
        let (lo, hi, top_z) = bounds[&i];
        // THROUGH the piece = ended up under its top while still inside its footprint (a weapon that
        // slid or tipped off the edge of a narrow top is not through anything)
        // ...and the piece must be OVER it: a long weapon lying across a chamfered corner or an open end tips off and falls beside
        // the piece, inside its bounding box but under nothing; that is a fall, not a drop through
        let covered = |p: Vec3| crate::play_world::ray_down(&piece_tris[&i], Vec3::new(p.x, p.y, hi.z + 1.0), hi.z - lo.z + 2.0).is_some_and(|(d, _)| (hi.z + 1.0 - d) > p.z + 0.5);
        let through = pose.is_some_and(|p| p.z < top_z - 0.3 && p.x > lo.x + 0.05 && p.x < hi.x - 0.05 && p.y > lo.y + 0.05 && p.y < hi.y - 0.05 && covered(p));
        let drop = if through { z0 - pose.unwrap().z } else { 0.0 };
        if let Some(p) = pose {
            if p.x > lo.x + 0.05 && p.x < hi.x - 0.05 && p.y > lo.y + 0.05 && p.y < hi.y - 0.05 && p.z < top_z {
                eprintln!("BURIED {} variant {}: weapon centre {:.3} wu BELOW the top at rest (placed at {:.3})", pieces[i].1, (wd % 8) - 1, top_z - p.z, z0 - top_z);
            }
        }
        let piece_kind = w.kind_of(0x1000 + i as u32 * 8).unwrap_or("none");
        out.push((format!("{} [{}] variant {}", pieces[i].1, piece_kind, (wd % 8) - 1), drop, names.iter().find(|(n, _)| *n == pieces[i].1).map_or(0, |x| x.1), wd % 8 == 1));
    }
    out
}

/// How far inside the footprint a weapon may sink below a piece's top before it counts as THROUGH it:
/// a detailed piece (an arch, a tower, a rock) has slopes a weapon slides down; the failure is the
/// whole height of the map.
const THROUGH_PIECE: f32 = 5.0;

#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn a_weapon_on_any_forge_piece_is_held_by_it() {
    let filt = |n: &str| n.contains("levels\\forge") || n.contains("ff_");
    let mut fell: Vec<String> = Vec::new();
    let mut total = 0;
    // upright (so the piece's box is its footprint); then boxes the picture got wrong (0.4 size). Turned
    // or tilted pieces are not judged: their box is bigger than their footprint, and a weapon that
    // tipped off a thin top lands inside it without having gone through anything.
    for (orient, pick_scale) in [(0u8, 1.0f32), (0, 0.4)] {
        let res = piece_survey("forge_halo", &filt, 0x40, 6.0, orient, pick_scale, 1.0);
        total += res.len();
        fell.extend(res.iter().filter(|(_, d, _, _)| *d > THROUGH_PIECE).map(|(n, d, t, above)| format!("orient {orient} pick x{pick_scale}: {n} ({t} tris) weapon {} dropped {d:.0} wu", if *above { "above" } else { "inside" })));
    }
    assert!(total >= 800, "liveness: only {total} weapon drops ran");
    assert!(fell.is_empty(), "{} weapons fell through Forge pieces:\n{}", fell.len(), fell.join("\n"));
}

/// **The Coliseum pieces, from the palette.** `wall_coliseum` and `window_coliseum` are variants of the
/// `structure_doors` palette entry; the object under them is `ff_arena_wall` / `ff_arena_window`. Read from
/// the map's own palette: each is solid for a biped (so it is in the prop world's support set), its shape
/// carries NO authored collision group (0xFF: the rigid body's default, so no mask can exclude a loose
/// prop from it), and a weapon placed on it, a few cm in, with its origin on the surface and a hand's
/// width in, stays on it at the editor's scale and at 2.5x and 0.5x.
///
/// SABOTAGE: `shape_contacts_sided(.., true, ..)` -> `false` in `PropWorld::substep` -> `ff_arena_wall`
/// lets the deepest weapon through (12 wu).
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn the_coliseum_pieces_the_palette_offers_hold_a_weapon_at_any_scale() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    let pieces: Vec<_> = sc.forge_palette_full().into_iter().filter(|e| e.variant_name == "wall_coliseum" || e.variant_name == "window_coliseum").collect();
    assert_eq!(pieces.len(), 2, "liveness: the palette offers the two Coliseum variants");
    drop(sc);
    for e in &pieces {
        assert!(crate::reach_collision::collides_with_biped(&native, cache, e.tag_short), "{} must be solid", e.variant_name);
        let groups: Vec<u8> = crate::reach_collision::shape_filters(&native, cache, e.tag_short).iter().map(|s| s.group).collect();
        assert!(!groups.is_empty() && groups.iter().all(|g| *g == crate::reach_collision::NO_GROUP_OVERRIDE), "{}: a shape names a collision group {groups:?}", e.variant_name);
    }
    let names: Vec<String> = pieces.iter().map(|e| native.tag_name(cache, e.tag_short).unwrap_or_default()).collect();
    let filt = |n: &str| names.iter().any(|m| m == n);
    let mut fell: Vec<String> = Vec::new();
    for scale in [1.0f32, 2.5, 0.5] {
        let res = piece_survey_in(&native, cache, &filt, 0x40, 6.0, 0, 1.0, scale);
        assert!(res.len() >= 8, "liveness: {} weapon drops at scale {scale}", res.len());
        fell.extend(res.iter().filter(|(_, d, _, _)| *d > THROUGH_PIECE).map(|(n, d, _, above)| format!("scale {scale}: {n} weapon {} dropped {d:.0} wu", if *above { "above" } else { "inside" })));
    }
    assert!(fell.is_empty(), "weapons fell through the Coliseum pieces:\n{}", fell.join("\n"));
}

/// **Every solid Forge piece the biped collides with is also there for a loose prop.** The player and a driven vehicle
/// collide with `PlayWorld::regather`'s world: the level soup plus EVERY nearby solid object's `physics_world_tris`. A loose
/// prop's world is the level plus the objects the prop world decoded as terrain; an object whose `physics_body` is `None`, whose
/// body has no (positive, finite) mass, or no shapes, was marked `Kind::Inert` and left out, so a prop fell through a wall the
/// player stood on. For every `bloc` / `scen` / `mach` / `ctrl` tag of `forge_halo` that is solid to a biped (`collides_with_biped`)
/// and carries physics triangles (the forge structure pieces, the Coliseum `ff_arena_wall` / `ff_arena_window` among them), placed
/// with the retail default placement (Normal) and woken through a prop touching it:
///   * it must not end `Inert` (it is a body or terrain);
///   * if terrain, the prop world's statics hold exactly the triangles `regather` would give for it (counted in a box around it).
/// The survey prints, per class, what the old rule did (no body / zero mass / no shapes / fixed / body) and the tags in each.
///
/// SABOTAGE: `inert_or_terrain` -> `Kind::Inert` (the pre-fix behaviour) in `PropWorld::materialise` -> red for every piece whose
/// body is missing or massless.
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn every_solid_forge_piece_is_terrain_or_a_body_to_a_loose_prop() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let mut tags: Vec<(u32, String)> = Vec::new();
    for g in ["bloc", "scen", "mach", "ctrl", "weap", "eqip", "vehi", "crea"] {
        for t in native.tags_by_group(cache, g) {
            if let Some(n) = native.tag_name(cache, t) {
                if crate::reach_collision::collides_with_biped(&native, cache, t) {
                    tags.push((t, n));
                }
            }
        }
    }
    let mk = |datum: u32, tag: u32, pos: Vec3| ObjectInfo { datum, type_sig: 0, sig0: 0, sig1: 0, pos: pos.to_array(), health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: tag, variant_name_sid: 0 };
    // the pre-fix rule, as a classification of what the real decode returns
    let mut classes: std::collections::BTreeMap<&str, Vec<String>> = std::collections::BTreeMap::new();
    let mut bad: Vec<String> = Vec::new();
    let mut inert_with_tris = 0;
    let mut checked = 0;
    for (i, (tag, name)) in tags.iter().enumerate() {
        let at = Vec3::new(5000.0 + (i % 20) as f32 * 60.0, 5000.0 + (i / 20) as f32 * 60.0, 500.0);
        let piece = mk(0x1000 + i as u32 * 8, *tag, at);
        let tris = sc.physics_world_tris(&piece);
        let old = match sc.physics_body(&piece) {
            None => "inert: no body",
            Some(b) if b.fixed => "fixed body (terrain)",
            Some(b) if !(b.mass.is_finite() && b.mass > 0.0) => "inert: zero or non-finite mass",
            Some(b) if b.shapes.is_empty() => "inert: no shapes",
            Some(_) => "dynamic body",
        };
        classes.entry(old).or_default().push(format!("{name} ({} tris)", tris.len()));
        if tris.is_empty() {
            continue;
        }
        let scene = MapScene::new(&sc, vec![piece]);
        let mut props = crate::play_world::PropSession::default();
        let mut actors = crate::reach_actor::ActorScene::default();
        let objs = vec![piece];
        props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
        props.wake_body(piece.datum, &native, cache, &scene, &objs);
        props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
        let w = props.world().expect("world");
        let kind = w.kind_of(piece.datum).unwrap_or("none");
        checked += 1;
        if old.starts_with("inert") {
            inert_with_tris += 1;
        }
        if kind == "Inert" {
            bad.push(format!("{name}: {old}, ends Inert with {} triangles the biped collides with", tris.len()));
        }
        if kind == "FixedLoaded" {
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for p in tris.iter().flatten() {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            let want = tris.iter().filter(|t| crate::physics_settle::tri_overlaps(t, lo - Vec3::splat(0.01), hi + Vec3::splat(0.01))).count();
            let (_, got) = w.diag_statics(lo - Vec3::splat(0.01), hi + Vec3::splat(0.01));
            if got != want {
                bad.push(format!("{name}: {old}; regather gives {want} triangles, the prop world's statics hold {got}"));
            }
        }
    }
    for (k, v) in &classes {
        eprintln!("PIECES {k}: {}", v.len());
        if k.starts_with("inert") {
            for n in v {
                eprintln!("   {n}");
            }
        }
    }
    eprintln!("PIECES checked {checked} with triangles; {inert_with_tris} of them were Inert before the fix");
    assert!(checked >= 150, "liveness: only {checked} solid objects with triangles");
    assert!(bad.is_empty(), "{} solid pieces are not there for a loose prop:\n{}", bad.len(), bad.join("\n"));
}

/// **A solid fixed piece is reachable by a body that is near it, with the triangles `regather` gives.** All 193 solid `fixed body`
/// pieces of `forge_halo` (the structure pieces among them: the Coliseum wall and window) are placed far apart, each with the
/// Forge default placement (Normal) and a weapon resting just above its highest up-facing spot, the pick-table box shrunk to
/// 0.4 (the app's box comes from the render mesh and can be wrong). The weapon is woken and the world runs 30 ticks. Each piece
/// must then be `FixedLoaded`, and the prop world's static triangles inside the piece's box must be exactly the piece's
/// `physics_world_tris` that `PlayWorld::regather` hands the biped (same triangles; the sidedness is the triangles' own winding,
/// shared by both).
///
/// SABOTAGE: `seed_world` using the pick box only (drop the union with the physics hull) -> red for every piece the shrunk box
/// no longer covers.
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn a_body_near_a_solid_piece_reaches_exactly_the_triangles_the_biped_collides_with() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let weapon = native
        .tags_by_group(cache, "weap")
        .into_iter()
        .find(|t| native.tag_name(cache, *t).is_some_and(|n| n.ends_with("assault_rifle\\assault_rifle")))
        .expect("assault rifle");
    let mk = |datum: u32, tag: u32, pos: Vec3| ObjectInfo { datum, type_sig: 0, sig0: 0, sig1: 0, pos: pos.to_array(), health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: tag, variant_name_sid: 0 };
    let wlo = sc.physics_world_tris(&mk(0, weapon, Vec3::ZERO)).iter().flatten().fold(Vec3::splat(f32::MAX), |a, p| a.min(*p));
    let mut objs: Vec<ObjectInfo> = Vec::new();
    let mut probes: Vec<(usize, u32)> = Vec::new(); // (piece index in `objs`, weapon datum)
    let mut idx = 0usize;
    for g in ["bloc", "scen", "mach", "ctrl"] {
        for t in native.tags_by_group(cache, g) {
            if !crate::reach_collision::collides_with_biped(&native, cache, t) {
                continue;
            }
            let at = Vec3::new(5000.0 + (idx % 20) as f32 * 60.0, 5000.0 + (idx / 20) as f32 * 60.0, 500.0);
            let piece = mk(0x1000 + idx as u32 * 4, t, at);
            idx += 1;
            if !sc.physics_body(&piece).is_some_and(|b| b.fixed) {
                continue;
            }
            let tris = sc.physics_world_tris(&piece);
            if tris.len() < 4 {
                continue;
            }
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for p in tris.iter().flatten() {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            let mut best: Option<Vec3> = None;
            for (fx, fy) in [(0.5f32, 0.5f32), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7)] {
                let o = Vec3::new(lo.x + (hi.x - lo.x) * fx, lo.y + (hi.y - lo.y) * fy, hi.z + 1.0);
                if let Some((d, n)) = crate::play_world::ray_down(&tris, o, hi.z - lo.z + 2.0) {
                    if n.z > 0.95 && best.map_or(true, |b| o.z - d > b.z) {
                        best = Some(Vec3::new(o.x, o.y, o.z - d));
                    }
                }
            }
            let Some(top) = best else { continue };
            probes.push((objs.len(), piece.datum + 1));
            objs.push(piece);
            objs.push(mk(piece.datum + 1, weapon, Vec3::new(top.x, top.y, top.z - wlo.z + 0.05)));
        }
    }
    let mut scene = MapScene::new(&sc, objs.clone());
    scene.pick_scale = 0.4;
    let mut props = crate::play_world::PropSession::default();
    let mut actors = crate::reach_actor::ActorScene::default();
    props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    for (_, wd) in &probes {
        props.wake_body(*wd, &native, cache, &scene, &objs);
    }
    for _ in 0..30 {
        props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    }
    let w = props.world().expect("world");
    let mut bad: Vec<String> = Vec::new();
    for (pi, _) in &probes {
        let piece = &objs[*pi];
        let name = native.tag_name(cache, piece.primary_tag).unwrap_or_default();
        let tris = sc.physics_world_tris(piece);
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in tris.iter().flatten() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let kind = w.kind_of(piece.datum).unwrap_or("none");
        if kind != "FixedLoaded" {
            bad.push(format!("{name}: {kind}, not terrain, with a weapon resting on it"));
            continue;
        }
        let (a, b) = (lo - Vec3::splat(0.01), hi + Vec3::splat(0.01));
        let want = tris.iter().filter(|t| crate::physics_settle::tri_overlaps(t, a, b)).count();
        let (_, got) = w.diag_statics(a, b);
        if got != want {
            bad.push(format!("{name}: the biped collides with {want} triangles of it, the prop world holds {got}"));
        }
    }
    eprintln!("SAMEPIECES {} fixed pieces compared", probes.len());
    assert!(probes.len() >= 100, "liveness: only {} fixed pieces compared", probes.len());
    assert!(bad.is_empty(), "{} pieces differ between the biped's world and the prop world:\n{}", bad.len(), bad.join("\n"));
}

/// **The bump trace sees what a real scene does.** On a real map, with a player standing by a woken object and a parked-set entry the
/// prop world never saw, the trace writes its startup line, one line per touched or woken object (tag, class, role, slot kind,
/// ownership, drawer), the follow lines for the first three with both terrain heights, and calls the object the prop world was
/// never offered "NOT IN THE PROP WORLD". Bounded to 200 lines.
///
/// SABOTAGE: `ownership` ignoring `kind` -> red (no "NOT IN THE PROP WORLD" line).
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn the_bump_trace_reports_ownership_and_follows_the_first_bodies_on_a_real_map() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("20_sword_slayer") else { return };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let objs = scenario_infos(&sc);
    let scene = MapScene::new(&sc, objs.clone());
    let mut props = crate::play_world::PropSession::default();
    let mut actors = crate::reach_actor::ActorScene::default();
    props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
    let mut woken = Vec::new();
    for o in &objs {
        if props.wake_body(o.datum, &native, cache, &scene, &objs) {
            woken.push(*o);
        }
    }
    assert!(woken.len() >= 10, "liveness: only {} woken", woken.len());
    // an object the prop world was never offered: the first one whose tag it skips
    let skipped = objs.iter().find(|o| props.world().and_then(|w| w.kind_of(o.datum)).is_none()).copied().expect("an object the prop world skips");
    let mut trace = crate::bump_trace::BumpTrace::default();
    let stand = Vec3::from(skipped.pos);
    // the parked set holds the skipped object as a vehicle the ride path steps
    let parked = [(skipped.datum, Vec3::from(skipped.pos))];
    for _ in 0..120 {
        props.tick(1.0 / 60.0, &native, cache, &scene, &objs, None, &[], &mut actors);
        let events = props.take_events();
        trace.observe(crate::bump_trace::Cx { props: &props, objects: &objs, scene: &scene, native: &native, cache, soup: &[], parked: &parked, riding: None, actors: &actors, player: (stand, 0.3, 1.5), events });
    }
    let lines = &trace.captured;
    assert!(lines.len() <= crate::bump_trace::MAX_LINES);
    assert!(lines[0].contains("bump_trace active"), "{}", lines[0]);
    assert!(lines.iter().any(|l| l.contains("NOT IN THE PROP WORLD") && l.contains("RIDE PATH")), "no ownership line for the skipped parked object:\n{}", lines.join("\n"));
    assert!(lines.iter().any(|l| l.contains("in the prop world as Live") && l.contains("role simulated")), "no line for an awake body");
    assert!(lines.iter().filter(|l| l.contains(" follow +")).count() >= 3, "no follow lines");
    assert!(lines.iter().any(|l| l.contains("terrain under it: player world")), "no terrain heights");
    for l in lines.iter().take(12) {
        eprintln!("TRACE {l}");
    }
}

/// **Pieces authored `Phased` hold a weapon, the soccer ball and a Mongoose.** The Forge menu stores Normal 0x00, Fixed 0x40 and Phased 0xC0 in the
/// placement byte (`c_forge_menu_object_properties` ctor, the three `add_value` calls), and a published Coliseum variant has EVERY wall, platform and
/// brace at 0xC0. The player and a driven vehicle collide with them (`PlayWorld::regather` has no placement-mode filter, and the engine's
/// `get_desired_sandbox_physics_mode_for_object` gives a Phased object outside the editor the same mode as a Fixed one: frozen, collidable). The prop
/// world made them `Inert`, so a weapon, the ball and an unmanned vehicle fell through. Every `ff_` piece, placed Phased, with a weapon, with the
/// soccer ball, with the Coliseum pieces among them, never lets the loose body through.
///
/// SABOTAGE: `PropWorld::add` not mapping Phased to Fixed -> red (hundreds of drops through every piece).
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn phased_forge_pieces_hold_a_weapon_and_the_ball() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
    let filt = |n: &str| n.contains("levels\\forge") || n.contains("ff_");
    let mut fell: Vec<String> = Vec::new();
    let mut total = 0;
    let balls: Vec<String> = ["weap", "eqip", "bloc", "scen"].iter().flat_map(|g| native.tags_by_group(cache, g)).filter_map(|t| native.tag_name(cache, t)).filter(|n| n.contains("ball")).collect();
    eprintln!("BALLS {balls:?}");
    for loose in ["assault_rifle\\assault_rifle", "soccer_ball\\soccer_ball"] {
        if !balls.iter().any(|b| b.ends_with(loose)) && loose.contains("ball") {
            continue;
        }
        let res = piece_survey_loose(&native, cache, &filt, 0xC0, 6.0, 0, 1.0, 1.0, loose);
        total += res.len();
        fell.extend(res.iter().filter(|(_, d, _, _)| *d > THROUGH_PIECE).map(|(n, d, t, above)| format!("{loose}: {n} ({t} tris) {} dropped {d:.0} wu", if *above { "above" } else { "inside" })));
    }
    assert!(total >= 600, "liveness: only {total} drops ran");
    assert!(fell.is_empty(), "{} loose bodies fell through Phased Forge pieces:\n{}", fell.len(), fell.join("\n"));
}

/// **The Mongoose and the Warthog stay on platforms authored Phased.** The same bump matrix as `walking_into_a_parked_vehicle_...` (the vehicle
/// settles on a block platform, the player walks into it, a crate sits in the way for half the cases) with the platforms' placement byte at 0xC0
/// instead of 0x40: the vehicle and the crate never drop through, the player is stopped by the hull.
///
/// SABOTAGE: `PropWorld::add` not mapping Phased to Fixed -> red.
#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn vehicles_and_crates_stay_on_phased_platforms() {
    FIXED_PLATFORM_PLACEMENT.with(|c| c.set(0xC0));
    let cases: Vec<Bump> = matrix().into_iter().filter(|c| matches!(c.support, Support::Platform { fixed: true, .. })).collect();
    assert!(cases.len() >= 16, "liveness: {} phased-platform cases", cases.len());
    let res = run_bump("forge_halo", &cases, 6.0, 8.0, false);
    FIXED_PLATFORM_PLACEMENT.with(|c| c.set(0x40));
    assert!(res.len() >= 12, "liveness: only {} ran", res.len());
    if let Err(e) = judge(&res, cases.len().saturating_sub(4)) {
        panic!("{e}");
    }
}
