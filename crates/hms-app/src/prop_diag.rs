//! #bump-class  `play propdiag [fixed]` -- a PRINT-ONLY report of how the rigid-body prop world sees
//! the map the headless host has loaded, built from the host's REAL scene (the real pick table and
//! the real object hulls), not from a stand-in. It never changes anything: it builds its own
//! [`PropSession`], wakes every loose object the map or variant places, steps it for a few seconds
//! and prints, for each that ended up lower than it started, what held (and did not hold) it.
//! No window, no audio: the headless host owns neither.

use crate::play_world::PropSession;
use crate::scene::SceneController;
use glam::Vec3;
use hms_ipc::ObjectInfo;

/// The report. `fixed` places every Forge structural piece (`levels\forge\ff_*`) as Fixed physics,
/// otherwise Normal; loose objects are always Normal.
pub fn run(scene: &SceneController, objects: &[ObjectInfo], fixed: bool, secs: f32) -> String {
    use crate::objscene::ObjectScene;
    let Some(cache) = scene.cache_handle() else { return "no map".into() };
    let native = scene.native_dll();
    let name = |o: &ObjectInfo| native.tag_name(cache, o.primary_tag).unwrap_or_default();
    let class = |o: &ObjectInfo| native.tag_class(cache, o.primary_tag).unwrap_or_default();
    let is_piece = |n: &str| n.contains("levels\\forge\\ff_");
    let mut out = String::new();
    let mut props = PropSession::default();
    for o in objects {
        if fixed && is_piece(&name(o)) {
            props.placement.insert(o.datum, 0x40);
        }
    }
    let mut actors = crate::reach_actor::ActorScene::default();
    let sc: &dyn ObjectScene = scene;
    props.tick(1.0 / 60.0, native, cache, sc, objects, None, &[], &mut actors);
    // what the prop world made of every object
    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    let mut no_pick = 0usize;
    for o in objects {
        let k = props.world().and_then(|w| w.kind_of(o.datum)).unwrap_or("not offered");
        *kinds.entry(format!("{} {}", class(o), k)).or_default() += 1;
        if sc.aabb_of(o.datum).is_none() {
            no_pick += 1;
        }
    }
    out += &format!("propdiag: {} objects, {} without a pick-table box; placement of pieces: {}\n", objects.len(), no_pick, if fixed { "Fixed" } else { "Normal" });
    for (k, n) in &kinds {
        out += &format!("   {k}: {n}\n");
    }
    // loose = something a player bumps: weapons, equipment, vehicles, crates and props that are not Forge pieces
    let loose: Vec<ObjectInfo> = objects
        .iter()
        .filter(|o| !is_piece(&name(o)) && matches!(class(o).as_str(), "weap" | "eqip" | "vehi" | "bloc" | "crat") && sc.physics_body(o).is_some_and(|b| !b.fixed && b.mass > 0.0))
        .copied()
        .collect();
    let start: std::collections::HashMap<u32, f32> = loose.iter().map(|o| (o.datum, o.pos[2])).collect();
    let mut woken = 0;
    for o in &loose {
        woken += props.wake_body(o.datum, native, cache, sc, objects) as usize;
    }
    for _ in 0..(secs * 60.0) as usize {
        props.tick(1.0 / 60.0, native, cache, sc, objects, None, &[], &mut actors);
    }
    out += &format!("   {} loose objects, {} woken, {secs} s simulated\n", loose.len(), woken);
    let Some(w) = props.world() else { return out };
    let mut fell = 0;
    for o in &loose {
        let Some((p, _)) = w.body_pose(o.datum) else { continue };
        let drop = start[&o.datum] - p.z;
        if drop <= 0.5 {
            continue;
        }
        fell += 1;
        if fell > 30 {
            continue;
        }
        let at = Vec3::from(o.pos);
        let (mn, mx) = (at - Vec3::splat(5.0), at + Vec3::splat(5.0));
        let (all_static, near_static) = w.diag_statics(mn, mx);
        let mut level = Vec::new();
        sc.static_tris_in_aabb(mn, mx, &mut level);
        let from = at + Vec3::Z * 0.3;
        let with_obj = sc.raycast_solid_excluding(from, -Vec3::Z, 100.0, &[o.datum]).map(|d| from.z - d);
        let level_only = {
            let mut best: Option<f32> = None;
            let tris = level.clone();
            if let Some((d, _)) = crate::play_world::ray_down(&tris, from, 100.0) {
                best = Some(from.z - d);
            }
            best
        };
        // the pieces whose pick box or physics reach within 5 wu of it
        let mut near = String::new();
        for u in objects.iter().filter(|u| u.datum != o.datum && (Vec3::from(u.pos) - at).length() < 12.0) {
            let n = name(u);
            if !is_piece(&n) {
                continue;
            }
            let pick = sc.aabb_of(u.datum).map(|(a, b)| format!("{:.1},{:.1},{:.1}..{:.1},{:.1},{:.1}", a[0], a[1], a[2], b[0], b[1], b[2]));
            let tris = sc.physics_world_tris(u);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for p in tris.iter().flatten() {
                lo = lo.min(*p);
                hi = hi.max(*p);
            }
            let under = at.x >= lo.x - 0.5 && at.x <= hi.x + 0.5 && at.y >= lo.y - 0.5 && at.y <= hi.y + 0.5 && at.z >= lo.z - 0.5;
            near += &format!(
                "      piece {} kind {:?} pick {:?} hull {} tris {:.1},{:.1},{:.1}..{:.1},{:.1},{:.1}{}\n",
                n.rsplit('\\').next().unwrap_or(&n),
                w.kind_of(u.datum),
                pick,
                tris.len(),
                lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
                if under { "  <- under/at the object" } else { "" }
            );
        }
        out += &format!(
            "   FELL {} [{}] {:.2} wu from ({:.1},{:.1},{:.1}); level tris within 5 wu {}, fixed hull tris loaded {} (within 5 wu {}); ray down: level only {:?}, with objects {:?}\n{}",
            name(o), class(o), drop, o.pos[0], o.pos[1], o.pos[2], level.len(), all_static, near_static, level_only, with_obj, near
        );
    }
    out += &format!("   {fell} loose objects fell more than 0.5 wu\n");
    out
}
