//! #ledge-gate  A driven vehicle must not move AGAINST its own velocity, nor jump, when it drives
//! off the edge of something.
//!
//! Reported: *"a vehicle teleports backward when it is about to drive off the edge of basically
//! anything"*. The gates here drive through [`Ride::tick_in`], i.e. the real `PlayMode::tick` the
//! window calls every frame -- `drive_and_seat` -> `play_vehicle::step` (the memoised wheel ray, the
//! `DriveModel` sub-steps) -> `PropSession::collide_driven` (the 6 wu hull gather, `sweep_hull`) ->
//! `props.tick` -- against a REAL `SceneController` (the map's engine collision soup, the same
//! `raycast_solid_excluding` the wheels cast) plus real placed Forge blocks. Nothing between the
//! input and the ride's position is re-implemented here.
//!
//! What a block is to the WHEELS differs from the app in one place: the app ray-tests the block's
//! pick mesh (render triangles, or the hull for a hidden block), this scene ray-tests its physics
//! triangles -- the same ones the hull collides with -- because the pick table needs the GPU load.
//!
//! The per-tick test is the owner's complaint written down: the position moved against the
//! horizontal velocity the vehicle HAD at the start of the tick by more than [`BACK_TOL`], or moved
//! further than the speed accounts for by more than [`JUMP_TOL`].
#![cfg(test)]

use std::collections::HashMap;

use glam::{Quat, Vec3};
use hms_ipc::ObjectInfo;

use crate::objscene::ObjectScene;
use crate::play_ctl::MoveInput;
use crate::play_mode::ride_harness::{Ride, DT};
use crate::scene::SceneController;

/// Backward motion (wu in one 60 Hz tick) that is not a collision response a human could see: a
/// push-out of a few millimetres is the hull resting on an edge; 5 cm in one tick (3 wu/s) is not.
pub const BACK_TOL: f32 = 0.05;
/// Distance beyond what the larger of the tick's two speeds explains, wu.
pub const JUMP_TOL: f32 = 0.15;

/// A real `SceneController` plus placed Forge blocks. Everything delegates to the controller except
/// the rays and the object collision, which see the blocks.
pub struct MapScene<'a> {
    pub sc: &'a SceneController,
    pub blocks: Vec<ObjectInfo>,
    /// World physics triangles per block datum, posed once.
    pub tris: HashMap<u32, Vec<[Vec3; 3]>>,
}

impl<'a> MapScene<'a> {
    pub fn new(sc: &'a SceneController, blocks: Vec<ObjectInfo>) -> Self {
        let tris = blocks.iter().map(|o| (o.datum, sc.physics_world_tris(o))).collect();
        MapScene { sc, blocks, tris }
    }
}

impl ObjectScene for MapScene<'_> {
    fn has_cache(&self) -> bool { true }
    fn invalidate(&mut self) {}
    fn rebuild_pending(&self) -> bool { false }
    fn maybe_rebuild(
        &mut self,
        _o: &[ObjectInfo],
        _c: &HashMap<u32, (u8, u8)>,
        _m: &hms_render::MeshRenderer,
        _d: &eframe::wgpu::Device,
        _q: &eframe::wgpu::Queue,
    ) -> Option<crate::objscene::RebuiltLanes> { None }
    fn set_object_scales(&mut self, _s: HashMap<u32, f32>) {}
    fn set_object_casters(&mut self, _c: std::collections::HashSet<u32>) {}
    fn pick(&self, _o: Vec3, _d: Vec3) -> Option<u32> { None }
    fn aabb_of(&self, _d: u32) -> Option<([f32; 3], [f32; 3])> { None }
    fn object_obb(&self, _d: u32) -> Option<crate::construct::Obb> { None }
    fn selection_wireframe(&self, _d: u32) -> Option<Vec<[f32; 3]>> { None }
    fn translate_pick(&mut self, _d: u32, _v: Vec3) {}
    fn rotate_pick(&mut self, _d: u32, _q: Quat, _p: Vec3) {}
    fn raycast_scene(&self, o: Vec3, d: Vec3) -> Option<Vec3> { self.sc.raycast_scene(o, d) }
    fn raycast_scene_n(&self, o: Vec3, d: Vec3) -> Option<(Vec3, Vec3)> { self.sc.raycast_scene_n(o, d) }
    fn raycast_objects_excluding(&self, _o: Vec3, _d: Vec3, _x: &[u32]) -> Option<Vec3> { None }
    fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])> { None }
    fn resolve_object_mode(&self, t: u32) -> u32 { self.sc.resolve_object_mode(t) }
    fn tag_name_of(&self, t: u32) -> String { self.sc.tag_name_of(t) }
    fn raw_tag_class(&self, t: u32) -> Option<String> { self.sc.raw_tag_class(t) }
    fn bsp_flags_report(&self) -> String { String::new() }
    fn blocker_overlay(&self) -> (&[([f32; 3], [f32; 4])], &[([f32; 3], [f32; 3], [f32; 3])]) { (&[], &[]) }
    fn collision_world_edges(&self, _o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
    fn physics_world_edges(&self, _o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
    fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> { None }
    fn physics_body(&self, o: &ObjectInfo) -> Option<crate::physics_settle::PhysBody> { self.sc.physics_body(o) }
    fn physics_world_tris(&self, o: &ObjectInfo) -> Vec<[Vec3; 3]> { self.sc.physics_world_tris(o) }
    fn physics_world_shells(&self, o: &ObjectInfo) -> Vec<Vec<[Vec3; 3]>> { self.sc.physics_world_shells(o) }
    fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) { self.sc.static_tris_in_aabb(mn, mx, out) }
    fn raycast_solid(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> { self.raycast_solid_excluding(origin, dir, max, &[]) }
    fn raycast_solid_excluding(&self, origin: Vec3, dir: Vec3, max: f32, exclude: &[u32]) -> Option<f32> {
        let level = self.sc.raycast_solid_excluding(origin, dir, max, exclude);
        if dir.z > -0.99 {
            return level;
        }
        let mut best = level;
        for o in self.blocks.iter().filter(|o| !exclude.contains(&o.datum)) {
            if let Some((d, _)) = crate::play_world::ray_down(&self.tris[&o.datum], origin, max) {
                if best.map_or(true, |b| d < b) {
                    best = Some(d);
                }
            }
        }
        best
    }
}

/// What one drive recorded.
#[derive(Default, Debug)]
pub struct Drive {
    pub ticks: usize,
    pub backward: usize,
    pub worst_back: f32,
    pub jumps: usize,
    pub worst_jump: f32,
    /// The origin passed the edge.
    pub reached: bool,
    /// ... and then fell at least 80% of the drop.
    pub dropped: bool,
    pub report: Vec<String>,
    /// per tick: position, velocity, the hull note after the tick
    pub track: Vec<(Vec3, Vec3, String)>,
    /// first flagged tick
    pub first_flag: Option<usize>,
}

/// Ground height under (x, y) as the wheels see it: the production ray, 15 wu from `z0 + 3`.
fn ground_at(scene: &dyn ObjectScene, x: f32, y: f32, z0: f32) -> Option<f32> {
    scene.raycast_solid_excluding(Vec3::new(x, y, z0 + 3.0), -Vec3::Z, 15.0, &[]).map(|d| z0 + 3.0 - d)
}

/// A place to drive off: `(start xy, ground z, yaw, distance start -> edge, drop)`.
pub type Ledge = (glam::Vec2, f32, f32, f32, f32);

/// March 8 headings from each seed over level ground (within 15 cm) until the ground falls by more
/// than `min_drop`, after at least `run` wu of level run; the ledge starts `lead` wu before the edge.
pub fn find_ledges(scene: &dyn ObjectScene, seeds: &[Vec3], min_drop: f32, run: f32, lead: f32, want: usize) -> Vec<Ledge> {
    let mut out = Vec::new();
    'seed: for c in seeds {
        if out.len() >= want {
            break;
        }
        let Some(z0) = ground_at(scene, c.x, c.y, c.z) else { continue };
        for k in 0..8 {
            let yaw = k as f32 * std::f32::consts::FRAC_PI_4;
            let dir = glam::Vec2::new(yaw.cos(), yaw.sin());
            for i in 1..80 {
                let d = i as f32 * 0.5;
                let p = glam::Vec2::new(c.x, c.y) + dir * d;
                match ground_at(scene, p.x, p.y, z0) {
                    Some(z) if (z - z0).abs() < 0.15 => {}
                    Some(z) if z < z0 - min_drop && d >= run => {
                        out.push((glam::Vec2::new(c.x, c.y) + dir * (d - lead), z0, yaw, lead, (z0 - z).min(6.0)));
                        continue 'seed;
                    }
                    _ => break,
                }
            }
        }
    }
    out
}

/// Drive `vehicle` (a tag-name suffix) at a ledge through the real `PlayMode::tick`.
#[allow(clippy::too_many_arguments)]
pub fn drive_ledge(
    native: &std::sync::Arc<hms_native::NativeDll>,
    cache: hms_native::CacheHandle,
    scene: &dyn ObjectScene,
    blocks: &[ObjectInfo],
    vehicle: &str,
    ledge: Ledge,
    speed: f32,
    held: bool,
) -> Option<Drive> {
    let (sxy, z0, yaw, edge_dist, drop) = ledge;
    let start = Vec3::new(sxy.x, sxy.y, z0 + 0.6);
    let mut r = Ride::with_vehicle_placed(native.clone(), cache, vehicle, start, yaw, blocks)?;
    let tag = r.objects[0].primary_tag;
    r.force_seat(tag, 0).ok()?;
    let idle = MoveInput { yaw, ..Default::default() };
    for _ in 0..180 {
        r.tick_in(&idle, scene);
    }
    let dir = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
    let begin = r.pm.riding.as_ref()?.drive.pos;
    let z_rest = begin.z;
    r.pm.riding.as_mut()?.drive.vel = dir * speed;
    let go = MoveInput { fwd: if held { 1.0 } else { 0.0 }, yaw, ..Default::default() };
    let mut out = Drive::default();
    for k in 0..(9.0 / DT) as usize {
        let (p0, v0) = {
            let d = &r.pm.riding.as_ref()?.drive;
            (d.pos, d.vel)
        };
        r.tick_in(&go, scene);
        let (p1, v1) = {
            let d = &r.pm.riding.as_ref()?.drive;
            (d.pos, d.vel)
        };
        out.ticks += 1;
        out.track.push((p1, v1, r.pm.props.note.clone()));
        let step = p1 - p0;
        let vh = Vec3::new(v0.x, v0.y, 0.0);
        let along = step.dot(vh.normalize_or_zero());
        let mut flag = |what: &str, amount: f32, out: &mut Drive| {
            if out.first_flag.is_none() {
                out.first_flag = Some(k);
            }
            if out.report.len() < 3 {
                out.report.push(format!("tick {k}: {what} {amount:.3}: pos {p0:?} -> {p1:?}, v {v0:?} -> {v1:?}"));
            }
        };
        if vh.length() > 1.0 && along < -BACK_TOL {
            out.backward += 1;
            out.worst_back = out.worst_back.min(along);
            flag("moved against its velocity by", along, &mut out);
        }
        let allowed = v0.length().max(v1.length()) * DT + JUMP_TOL;
        if step.length() > allowed {
            out.jumps += 1;
            out.worst_jump = out.worst_jump.max(step.length() - allowed);
            flag("moved further than its speed explains by", step.length() - allowed, &mut out);
        }
        let travelled = (p1 - begin).dot(dir);
        out.reached |= travelled > edge_dist;
        if out.reached && p1.z < z_rest - 0.8 * drop && travelled > edge_dist + 0.5 {
            out.dropped = true;
        }
        if p1.z < z_rest - drop - 0.3 {
            break;
        }
    }
    Some(out)
}

/// Real flat spots of the level: centres of big up-facing collision triangles.
fn level_seeds(native: &hms_native::NativeDll, cache: hms_native::CacheHandle) -> Vec<Vec3> {
    let mut seeds = Vec::new();
    for &t in &native.tags_by_group(cache, "sbsp") {
        if let Some(b) = crate::collision_bsp::CollisionBsp::for_sbsp(native, cache, t) {
            for (tr, _) in b.solid_triangles() {
                let n = (tr[1] - tr[0]).cross(tr[2] - tr[0]);
                if n.z > 0.99 * n.length() && 0.5 * n.length() > 4.0 {
                    seeds.push((tr[0] + tr[1] + tr[2]) / 3.0);
                }
            }
        }
    }
    seeds
}

/// Deterministic shuffle so the ledges are spread over the map instead of the first BSP's corner.
fn spread(mut v: Vec<Vec3>, n: usize) -> Vec<Vec3> {
    let mut seed = 0x1234_ABCDu32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let len = v.len();
    for i in (1..len).rev() {
        v.swap(i, rnd() as usize % (i + 1));
    }
    v.truncate(n);
    v
}

/// Real Forge blocks big enough to drive onto (a flat top of at least 6 x 6 wu, 0.5 wu or more
/// above the ground), placed on flat ground. Returns the objects.
fn place_blocks(sc: &SceneController, native: &hms_native::NativeDll, cache: hms_native::CacheHandle, spots: &[Vec3], want: usize) -> Vec<ObjectInfo> {
    let mut out: Vec<ObjectInfo> = Vec::new();
    let mut spot = spots.iter();
    for &tag in native.tags_by_group(cache, "bloc").iter() {
        if out.len() >= want {
            break;
        }
        let name = native.tag_name(cache, tag).unwrap_or_default();
        if !name.contains("levels\\forge\\ff_") {
            continue;
        }
        let Some(at) = spot.next() else { break };
        let mk = |pos: Vec3| ObjectInfo {
            datum: 0x7000 + out.len() as u32,
            type_sig: 0,
            sig0: 0,
            sig1: 0,
            pos: pos.to_array(),
            health: 1.0,
            shield: 1.0,
            mode_tag: 0,
            fwd: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
            attached: [0; 8],
            primary_tag: tag,
            variant_name_sid: 0,
        };
        let probe = mk(*at);
        let tris = sc.physics_world_tris(&probe);
        if tris.is_empty() {
            continue;
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in tris.iter().flatten() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        // a flat top at least 6 x 6 wu and 0.5 wu high
        if hi.x - lo.x < 6.0 || hi.y - lo.y < 6.0 || hi.z - lo.z < 0.5 {
            continue;
        }
        // stand it on the ground at the spot: lowest point on the floor
        eprintln!("block {name}: {:.1} x {:.1} x {:.1}", hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
        out.push(mk(*at + Vec3::Z * (at.z - lo.z)));
    }
    out
}

/// One map's whole gate. `stem` only names the map in the report.
pub fn run_map(stem: &str, vehicles: &[&str], speeds: &[f32], min_ledges: usize, min_block_ledges: usize) {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for(stem) else {
        eprintln!("ledge gates SKIPPED: no {stem} cache");
        return;
    };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let seeds = spread(level_seeds(&native, cache), 400);

    let level = MapScene::new(&sc, Vec::new());
    let level_ledges = find_ledges(&level, &seeds, 1.0, 9.0, 6.0, 14);
    // blocks: place on the flattest seeds, find their edges from seeds ON the block tops
    let blocks = place_blocks(&sc, &native, cache, &seeds[..seeds.len().min(60)], 6);
    let with_blocks = MapScene::new(&sc, blocks.clone());
    let mut tops = Vec::new();
    for o in &blocks {
        for t in with_blocks.tris[&o.datum].iter() {
            let n = (t[1] - t[0]).cross(t[2] - t[0]);
            if n.z > 0.99 * n.length() && 0.5 * n.length() > 1.0 {
                tops.push((t[0] + t[1] + t[2]) / 3.0);
            }
        }
    }
    let block_ledges = find_ledges(&with_blocks, &tops, 0.4, 6.0, 4.0, 14);
    eprintln!("{stem}: {} level ledges, {} blocks placed, {} block-top ledges", level_ledges.len(), blocks.len(), block_ledges.len());

    let (mut runs, mut reached, mut backward, mut jumps, mut incomplete) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut shown = 0;
    let mut worst = (0.0f32, 0.0f32);
    for (label, scene, ledges, objs) in [("level", &level as &dyn ObjectScene, &level_ledges, &[][..]), ("block", &with_blocks as &dyn ObjectScene, &block_ledges, &blocks[..])] {
        for v in vehicles {
            for &l in ledges.iter() {
                for &speed in speeds {
                    for held in [false, true] {
                        let Some(d) = drive_ledge(&native, cache, scene, objs, v, l, speed, held) else { continue };
                        runs += 1;
                        reached += d.reached as usize;
                        backward += d.backward;
                        jumps += d.jumps;
                        worst = (worst.0.min(d.worst_back), worst.1.max(d.worst_jump));
                        let inc = d.reached && !d.dropped;
                        incomplete += inc as usize;
                        if (d.backward > 0 || d.jumps > 0 || inc) && shown < 12 {
                            shown += 1;
                            eprintln!("LEDGE[{label}] {v} yaw {:.2} drop {:.1} speed {speed} held {held}: backward {} (worst {:.3}) jumps {} reached {} dropped {}", l.2, l.4, d.backward, d.worst_back, d.jumps, d.reached, d.dropped);
                            for line in &d.report {
                                eprintln!("     {line}");
                            }
                            if let Some(k) = d.first_flag.filter(|_| shown <= 3) {
                                let mut last = String::new();
                                for j in k.saturating_sub(8)..(k + 4).min(d.track.len()) {
                                    let (p, v, n) = &d.track[j];
                                    let note = if *n != last { n.as_str() } else { "" };
                                    last = n.clone();
                                    eprintln!("        t{j} pos ({:.3},{:.3},{:.3}) v ({:.2},{:.2},{:.2}) {note}", p.x, p.y, p.z, v.x, v.y, v.z);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("LEDGE TOTAL [{stem}] {runs} drives, {reached} crossed the edge: backward ticks {backward} (worst {:.3}), jump ticks {jumps} (worst {:.3}), crossed but did not drop {incomplete}", worst.0, worst.1);
    assert!(level_ledges.len() >= min_ledges && block_ledges.len() >= min_block_ledges, "liveness: {} level ledges (want {min_ledges}), {} block ledges (want {min_block_ledges})", level_ledges.len(), block_ledges.len());
    assert!(runs >= 40 && reached >= 20, "liveness: {runs} drives, {reached} crossed the edge");
    assert_eq!((backward, jumps), (0, 0), "a driven vehicle moved against its velocity or jumped at an edge");
}

#[test]
#[ignore = "opens the native library; run alone with --ignored --nocapture"]
fn driving_off_edges_in_forge_world_never_moves_backward_or_jumps() {
    run_map("forge_halo", &["mongoose", "warthog"], &[4.0, 9.0], 8, 4);
}

#[test]
#[ignore = "TRACE probe"]
fn probe_triangles_at_the_site() {
    let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
    let mut sc = SceneController::new_shared(native.clone());
    sc.attach_cache(cache);
    sc.load_collision_for_test();
    let mut t = Vec::new();
    sc.static_tris_in_aabb(Vec3::new(67.5, 205.5, 42.0), Vec3::new(71.5, 209.0, 47.0), &mut t);
    for tr in &t {
        let n = (tr[1] - tr[0]).cross(tr[2] - tr[0]);
        eprintln!("TRI n ({:.2},{:.2},{:.2}) area {:.2} {:?}", n.normalize().x, n.normalize().y, n.normalize().z, 0.5 * n.length(), tr.map(|p| (p * 100.0).round() / 100.0));
    }
}
