//! Where the camera lands after a Halo 4 / H2A map load - this game's CANDIDATES for the shared
//! spawn-camera rule (`crate::spawn_camera`), gathered over the pure-Rust reader:
//!
//!   * the variant's player spawns (`sp_initial_spawn` = type 16 with "initial" in the palette
//!     name -> `VariantInitialSpawn`, any other type 16 -> `VariantSpawn`, with their team) and
//!     its Forge loadout camera (type 28 -> `VariantCamera`, full facing);
//!   * the scenario's spawn placements (`initial_spawn_point` / `respawn_point` /
//!     `respawn_point_invisible` scen objects, classified by tag path like Reach);
//!   * the scenario's `player starting locations` (scnr +0x31C, Halo 4 layout only: on the MP
//!     maps the block is a single editor start that is not always in the play space - Forge
//!     Island's is out on the sea, Longbow's is under the terrain, measured 2026-09-17 - which
//!     is why the shared rule ranks it after the placements);
//!   * the scenario's own `mp_cinematic_camera` placement (the shipped map's loadout camera);
//!   * volumes: the playable clusters' bounds (the headless / GUI frame bounds) when the caller
//!     has them, else the decoded scene bounds. Halo 4's structure_design soft ceilings are not
//!     decoded yet, so that tier is empty here.
//!
//! The rule picks spawn > starting location > camera > volume centre pulled to the ground, and
//! stands the pose off the BSP / objects (`ObjectScene::standoff` over the triangle soup) with the
//! small load-time clearance. See `docs/halo4_scenario_layout.md` for the tag layouts.

use glam::Vec3;

use super::cache::{ByteRead, H4Cache};
use super::edit_scene::H4ObjectScene;
use super::mvar::{H4PaletteEntry, H4PlacedObject, H4Variant};
use super::objects::{euler_to_basis, H4Placement};
use crate::objscene::ObjectScene;
use crate::spawn_camera::{Candidate, Kind, Pose, Volumes};

/// scnr block: player starting locations (checked on all 50 shipped caches, 177 elements:
/// stride from the block gap on every map with >1 element, finite positions, |yaw| <= pi,
/// pitch 0 everywhere). Element 0x24 B: f32x3 position @0, i32 @0xC (-1 or a small index,
/// meaning unknown), i32 @0x10 (-1 / 0, meaning unknown), f32 facing yaw @0x14 (radians), f32
/// facing pitch @0x18, i16 x4 @0x1C (0 / 0-1 / -1 / 0 on every element - editor / team / bsp
/// fields as in Reach, not confirmed). `player starting profile` is the block before it (+0x310).
pub const OFF_SCNR_PLAYER_STARTING_LOCATIONS: usize = 0x31C;
pub const PLAYER_STARTING_LOCATION_ELEM: usize = 0x24;
/// Eye height above a spawn marker (the shared value).
pub const EYE_HEIGHT: f32 = crate::spawn_camera::EYE_HEIGHT;
/// Load-time stand-off clearance (the shared value).
pub const STANDOFF_WU: f32 = crate::spawn_camera::STANDOFF_WU;
/// Variant object types (mvar.rs / docs/halo4_mvar_layout.md section 4).
const TYPE_PLAYER_SPAWN: u8 = 16;
const TYPE_LOADOUT_CAMERA: u8 = 28;

/// The load-time camera (the shared `Pose`: position / yaw / pitch, its source, the raw position).
pub type SpawnCam = Pose;

#[derive(Clone, Copy, Debug)]
pub struct StartingLocation {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub raw_i16: [i16; 4],
}

/// The scnr `player starting locations` block (empty on most MP maps - their spawns are scen
/// placements; 1 on Forge Island / Valhalla, 8-14 on Firefight and campaign maps). Halo 4 layout
/// only: H2A ("groundhog") shares the reader but its scnr differs, and reading these offsets
/// there yields garbage (a NaN position that once rendered the whole map black).
pub fn load_starting_locations(c: &H4Cache) -> Vec<StartingLocation> {
    let d = c.data();
    let mut out = Vec::new();
    if c.engine != super::cache::Engine::Halo4 { return out; }
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let Some((n, o)) = c.block(sm + OFF_SCNR_PLAYER_STARTING_LOCATIONS) else { return out };
    for i in 0..n.min(256) {
        let e = o + i * PLAYER_STARTING_LOCATION_ELEM;
        if e + PLAYER_STARTING_LOCATION_ELEM > d.len() { break; }
        let pos = [d.f32_at(e), d.f32_at(e + 4), d.f32_at(e + 8)];
        let yaw = d.f32_at(e + 0x14);
        let pitch = d.f32_at(e + 0x18);
        if !pos.iter().all(|v| v.is_finite() && v.abs() < 5000.0) || !yaw.is_finite() || !pitch.is_finite() { continue; }
        out.push(StartingLocation { pos, yaw, pitch, raw_i16: [d.i16_at(e + 0x1C), d.i16_at(e + 0x1E), d.i16_at(e + 0x20), d.i16_at(e + 0x22)] });
    }
    out
}

/// The variant's candidates: its loadout camera(s), initial spawns and any player spawn (with
/// the variant team so the shared rule can prefer red / blue / neutral).
pub fn variant_candidates(v: &H4Variant, palette: &[H4PaletteEntry]) -> Vec<Candidate> {
    let name_of = |o: &H4PlacedObject| o.quota.and_then(|q| palette.get(q as usize)).map(|e| e.name.to_ascii_lowercase()).unwrap_or_default();
    let mut out = Vec::new();
    for o in &v.objects {
        let n = name_of(o);
        let kind = if o.object_type == TYPE_LOADOUT_CAMERA || n.contains("loadout_camera") {
            Kind::VariantCamera
        } else if o.object_type == TYPE_PLAYER_SPAWN {
            if n.contains("initial") { Kind::VariantInitialSpawn } else { Kind::VariantSpawn }
        } else {
            continue;
        };
        out.push(Candidate::new(kind, Vec3::from(o.pos), Vec3::from(o.fwd)).team(o.team));
    }
    out
}

/// The scenario's candidates: spawn placements by tag path, the starting locations, the
/// cinematic camera(s).
pub fn scenario_candidates(cache: &H4Cache, placements: &[H4Placement]) -> Vec<Candidate> {
    let mut out = Vec::new();
    for p in placements {
        let Some(kind) = crate::spawn_camera::classify_tag_path(cache.tag_name(p.palette_tag)) else { continue };
        let fwd = p.basis.map(|b| b.0).unwrap_or_else(|| euler_to_basis(p.rot).0);
        out.push(Candidate::new(kind, Vec3::from(p.pos), fwd));
    }
    for s in load_starting_locations(cache) {
        let fwd = Vec3::new(s.yaw.cos(), s.yaw.sin(), 0.0);
        out.push(Candidate::new(Kind::StartingLocation, Vec3::from(s.pos), fwd).pitch(s.pitch));
    }
    out
}

/// The spawn camera for a loaded editor scene: the shared rule over the variant's + the
/// scenario's candidates, `playable` = the playable clusters' bounds when the caller has them
/// (the GUI / headless frame bounds), stood off the map geometry / objects by `STANDOFF_WU`.
pub fn spawn_camera_framed(es: &H4ObjectScene, playable: Option<(Vec3, Vec3)>) -> Option<SpawnCam> {
    let mut cands = es.variant().map(|(_, v)| variant_candidates(v, es.palette())).unwrap_or_default();
    cands.extend(scenario_candidates(es.cache(), es.placements()));
    let vols = Volumes {
        playable_bounds: playable.filter(|(a, b)| a.is_finite() && b.is_finite()),
        scene_bounds: es.scene_bounds().map(|(a, b)| (Vec3::from(a), Vec3::from(b))),
        ..Default::default()
    };
    // A camera OBJECT's own marker model surrounds its position, so a stand-off against the
    // objects would only push the camera off itself: cameras clear the BSP alone; spawn poses
    // (eye height above a marker that may sit inside a Forge block) clear BSP + objects.
    let bsp_only = |o: Vec3, d: Vec3, m: f32| es.raycast_scene(o, d).map(|p| (p - o).length()).filter(|t| *t <= m);
    let solid = |o: Vec3, d: Vec3, m: f32| es.raycast_solid(o, d, m);
    if es.soup_len() == 0 {
        return crate::spawn_camera::choose(&cands, &vols, None);
    }
    let picked_is_camera = crate::spawn_camera::pick(&cands).map_or(false, |c| c.kind.is_camera());
    if picked_is_camera {
        crate::spawn_camera::choose(&cands, &vols, Some(&bsp_only))
    } else {
        crate::spawn_camera::choose(&cands, &vols, Some(&solid))
    }
}

/// `spawn_camera_framed` without playable bounds (the teleport action / trait path).
pub fn spawn_camera(es: &H4ObjectScene) -> Option<SpawnCam> { spawn_camera_framed(es, None) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::h4::cache::maps_dir;
    use crate::h4::mvar::{load_forge_palette, parse_h4_variant, variant_dirs};
    use crate::h4::objects::load_placements;
    use crate::spawn_camera::{choose, pick, Source};

    fn open(name: &str) -> Option<Arc<H4Cache>> {
        let maps = maps_dir()?;
        let map = maps.join(format!("{name}.map"));
        if !map.is_file() { eprintln!("skip: {} missing", map.display()); return None; }
        Some(Arc::new(H4Cache::open(&map).expect("open")))
    }

    #[test]
    fn starting_locations_layout_on_shipped_maps() {
        // (map, count) pinned from the python survey of all 50 caches; Ravine has none.
        for (name, n) in [("ca_forge_ravine", 0usize), ("dlc_forge_island", 1), ("z11_valhalla", 1), ("ff81_courtyard", 8)] {
            let Some(c) = open(name) else { return };
            let s = load_starting_locations(&c);
            assert_eq!(s.len(), n, "{name}: starting locations");
            for l in &s {
                assert!(l.yaw.abs() <= std::f32::consts::PI + 1e-3 && l.pitch == 0.0, "{name}: {l:?}");
                assert_eq!(l.raw_i16[2], -1, "{name}: i16 @0x20 is -1 on every shipped element");
            }
        }
        if let Some(c) = open("z11_valhalla") {
            let s = load_starting_locations(&c);
            assert!((Vec3::from(s[0].pos) - Vec3::new(78.64, 92.77, 4.66)).length() < 0.05 && (s[0].yaw - 0.174).abs() < 0.01, "{:?}", s[0]);
        }
    }

    #[test]
    fn scenario_pose_ravine_stands_at_an_initial_spawn() {
        let Some(c) = open("ca_forge_ravine") else { return };
        let placements = load_placements(&c);
        let cands = scenario_candidates(&c, &placements);
        let p = choose(&cands, &Volumes::default(), None).expect("Ravine has spawn placements");
        assert_eq!(p.source, Source::Spawn(Kind::InitialSpawn));
        // the three initial_spawn_point placements sit at y 48.24, z 1.10 facing -Y (yaw -pi/2)
        assert!((p.pos.y - 48.25).abs() < 0.1 && (p.pos.z - (1.10 + EYE_HEIGHT)).abs() < 0.05, "{:?}", p.pos);
        assert!((p.yaw + std::f32::consts::FRAC_PI_2).abs() < 0.05 && p.pitch == 0.0, "yaw {} pitch {}", p.yaw, p.pitch);
    }

    #[test]
    fn variant_pose_prefers_initial_spawns_over_the_loadout_camera() {
        let Some(c) = open("ca_forge_ravine") else { return };
        let Some(vp) = variant_dirs().into_iter().map(|d| d.join("ca_forge_ravine_settler.mvar")).find(|p| p.is_file()) else { return };
        let palette = load_forge_palette(&c);
        let mut v = parse_h4_variant(&vp).expect("variant");
        let cands = variant_candidates(&v, &palette);
        let chosen = pick(&cands).expect("Settler has spawns");
        assert_eq!(chosen.kind, Kind::VariantInitialSpawn, "spawn points rank above the loadout camera");
        assert_eq!(chosen.team, 0, "red team first");
        // without any spawn: the loadout camera (when the variant has one)
        v.objects.retain(|o| o.object_type != TYPE_PLAYER_SPAWN);
        let cands = variant_candidates(&v, &palette);
        if v.objects.iter().any(|o| o.object_type == TYPE_LOADOUT_CAMERA) {
            let p = choose(&cands, &Volumes::default(), None).expect("camera");
            assert_eq!(p.source, Source::Spawn(Kind::VariantCamera));
            assert!(v.objects.iter().filter(|o| o.object_type == TYPE_LOADOUT_CAMERA).any(|o| Vec3::from(o.pos) == p.pos), "camera pose is a camera object: {:?}", p.pos);
        } else {
            assert!(cands.is_empty());
        }
    }

    #[test]
    fn h2a_maps_resolve_a_scenario_spawn() {
        // #h2a: the placement stride fix (records are stride + 4) made the respawn points readable
        let Some(maps) = crate::h2a::maps_dir() else { return };
        for name in ["ca_ascension", "ca_lockout"] {
            let map = maps.join(format!("{name}.map"));
            if !map.is_file() { eprintln!("skip: {} missing", map.display()); continue; }
            let c = Arc::new(H4Cache::open(&map).expect("open"));
            let placements = load_placements(&c);
            let cands = scenario_candidates(&c, &placements);
            let p = choose(&cands, &Volumes::default(), None).unwrap_or_else(|| panic!("{name}: no spawn candidate among {} placements", placements.len()));
            assert!(matches!(p.source, Source::Spawn(k) if k.is_spawn()), "{name}: {:?}", p.source);
            assert!(p.pos.is_finite() && p.pos.abs().max_element() < 2000.0, "{name}: {:?}", p.pos);
        }
    }
}
