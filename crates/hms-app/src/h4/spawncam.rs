//! Where the camera lands after a Halo 4 map load - the Reach `spawn_camera_pose` /
//! `variant_spawn_camera` / script-host "loadout camera" chain over the Halo 4 reader, so a
//! freshly loaded map (with or without a variant) opens INSIDE the playable area instead of at the
//! 3/4 overview `render::build_camera` frames from the geometry bounds.
//!
//! Priority (docs/halo4_scenario_layout.md):
//!   1. the variant's Forge loadout camera (`sp_loadout_camera`, object type 28): its position
//!      and full facing (the camera the game shows during loadout selection);
//!   2. the variant's initial spawn (`sp_initial_spawn`, type 16 with "initial" in the palette
//!      name), red team -> blue -> neutral -> any, the one nearest the variant's centroid, standing
//!      at eye height (0.62 wu, the Reach value) looking level along the spawn's forward;
//!   3. any variant player spawn (type 16), same pose;
//!   4. the scenario's spawn placements (`initial_spawn_point` > `respawn_point` >
//!      `respawn_point_invisible` scen objects), same standing pose, nearest the map's spawn
//!      centroid;
//!   5. the scenario's `player starting locations` (scnr +0x31C): position + facing yaw/pitch.
//!      AFTER the placements on purpose: on the MP maps the block is a single editor start that
//!      is not always in the play space (Forge Island's is out on the sea, Longbow's is under
//!      the terrain - measured 2026-09-17), while the spawn placements always are;
//!   6. the scenario's own `mp_cinematic_camera` placement (the shipped map's loadout camera);
//!   7. none - the callers keep the bounds-framed camera.
//! The chosen position is then pushed clear of the BSP / objects by the shared stand-off solver
//! (`ObjectScene::standoff`, the Reach `standoff` over the Halo 4 triangle soup) with a SMALL
//! clearance (`STANDOFF_WU`): enough to leave a wall or floor the marker sits in, not enough to
//! lift a standing pose off the ground.

use glam::Vec3;

use super::cache::{ByteRead, H4Cache};
use super::edit_scene::H4ObjectScene;
use super::mvar::{H4PaletteEntry, H4PlacedObject, H4Variant};
use super::objects::{euler_to_basis, H4Placement};
use crate::objscene::ObjectScene;

/// scnr block: player starting locations (checked on all 50 shipped caches, 177 elements:
/// stride from the block gap on every map with >1 element, finite positions, |yaw| <= pi,
/// pitch 0 everywhere). Element 0x24 B: f32x3 position @0, i32 @0xC (-1 or a small index,
/// meaning unknown), i32 @0x10 (-1 / 0, meaning unknown), f32 facing yaw @0x14 (radians), f32
/// facing pitch @0x18, i16 x4 @0x1C (0 / 0-1 / -1 / 0 on every element - editor / team / bsp
/// fields as in Reach, not confirmed). `player starting profile` is the block before it (+0x310).
pub const OFF_SCNR_PLAYER_STARTING_LOCATIONS: usize = 0x31C;
pub const PLAYER_STARTING_LOCATION_ELEM: usize = 0x24;
/// Eye height above a spawn marker (Reach `variant_spawn_camera`: player ~0.7 wu tall).
pub const EYE_HEIGHT: f32 = 0.62;
/// Load-time stand-off clearance (wu): clears a wall / floor the marker sits in without lifting
/// the standing pose (the interactive 2 wu would hoist every spawn camera to head-and-a-half).
pub const STANDOFF_WU: f32 = 0.35;
/// Variant object types (mvar.rs / docs/halo4_mvar_layout.md section 4).
const TYPE_PLAYER_SPAWN: u8 = 16;
const TYPE_LOADOUT_CAMERA: u8 = 28;
/// Variant team order for the pick: red, blue, neutral (8), then anything.
const TEAM_ORDER: [i8; 3] = [0, 1, 8];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpawnCamSource {
    VariantCamera,
    VariantInitialSpawn,
    VariantSpawn,
    ScenarioStart,
    ScenarioSpawn,
    ScenarioCamera,
}

impl SpawnCamSource {
    pub fn label(self) -> &'static str {
        match self {
            SpawnCamSource::VariantCamera => "variant loadout camera",
            SpawnCamSource::VariantInitialSpawn => "variant initial spawn",
            SpawnCamSource::VariantSpawn => "variant spawn",
            SpawnCamSource::ScenarioStart => "scenario player starting location",
            SpawnCamSource::ScenarioSpawn => "scenario spawn placement",
            SpawnCamSource::ScenarioCamera => "scenario cinematic camera",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpawnCam {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub source: SpawnCamSource,
    /// Position BEFORE the stand-off push (diagnostics).
    pub raw_pos: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct StartingLocation {
    pub pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub raw_i16: [i16; 4],
}

/// The scnr `player starting locations` block (empty on most MP maps - their spawns are scen
/// placements; 1 on Forge Island / Valhalla, 8-14 on Firefight and campaign maps).
pub fn load_starting_locations(c: &H4Cache) -> Vec<StartingLocation> {
    let d = c.data();
    let mut out = Vec::new();
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

fn yaw_of(f: Vec3) -> Option<f32> { (f.x.hypot(f.y) > 0.1).then(|| f.y.atan2(f.x)) }

/// Standing pose at a spawn marker: eye height up, level, looking along the marker's forward
/// (or at `fallback_look` when the marker faces nowhere useful).
fn stand_at(spawn: Vec3, fwd: Vec3, fallback_look: Vec3) -> (Vec3, f32, f32) {
    let eye = spawn + Vec3::Z * EYE_HEIGHT;
    let yaw = yaw_of(fwd).or_else(|| yaw_of(fallback_look - eye)).unwrap_or(0.0);
    (eye, yaw, 0.0)
}

/// Full-facing pose at a camera object.
fn camera_at(pos: Vec3, fwd: Vec3) -> (Vec3, f32, f32) {
    let f = fwd.normalize_or_zero();
    let yaw = yaw_of(f).unwrap_or(0.0);
    (pos, yaw, f.z.clamp(-1.0, 1.0).asin())
}

fn centroid(points: impl Iterator<Item = Vec3>) -> Option<Vec3> {
    let (mut sum, mut n) = (Vec3::ZERO, 0usize);
    for p in points { sum += p; n += 1; }
    (n > 0).then(|| sum / n as f32)
}

/// Of `cands` (pos, fwd, team): the first team in `TEAM_ORDER` that has one (else any), nearest
/// `centre` within that team.
fn pick_by_team(cands: &[(Vec3, Vec3, i8)], centre: Vec3) -> Option<(Vec3, Vec3)> {
    let nearest = |it: &mut dyn Iterator<Item = &(Vec3, Vec3, i8)>| it.min_by(|a, b| (a.0 - centre).length_squared().total_cmp(&(b.0 - centre).length_squared())).map(|c| (c.0, c.1));
    for team in TEAM_ORDER {
        if let Some(c) = nearest(&mut cands.iter().filter(|c| c.2 == team)) { return Some(c); }
    }
    nearest(&mut cands.iter())
}

/// Tiers 1-3: the variant's loadout camera / initial spawn / any spawn (pos, yaw, pitch, source).
pub fn variant_pose(v: &H4Variant, palette: &[H4PaletteEntry]) -> Option<(Vec3, f32, f32, SpawnCamSource)> {
    let name_of = |o: &H4PlacedObject| o.quota.and_then(|q| palette.get(q as usize)).map(|e| e.name.to_ascii_lowercase()).unwrap_or_default();
    let centre = centroid(v.objects.iter().map(|o| Vec3::from(o.pos))).unwrap_or(Vec3::ZERO);
    let cand = |o: &H4PlacedObject| (Vec3::from(o.pos), Vec3::from(o.fwd), o.team);
    // 1. the Forge loadout camera (type 28; the palette name is the rename-tolerant backup)
    let cameras: Vec<_> = v.objects.iter().filter(|o| o.object_type == TYPE_LOADOUT_CAMERA || name_of(o).contains("loadout_camera")).map(cand).collect();
    if let Some((p, f)) = pick_by_team(&cameras, centre) {
        let (pos, yaw, pitch) = camera_at(p, f);
        return Some((pos, yaw, pitch, SpawnCamSource::VariantCamera));
    }
    // 2. initial spawns, 3. any player spawn
    let spawns: Vec<_> = v.objects.iter().filter(|o| o.object_type == TYPE_PLAYER_SPAWN).collect();
    let initial: Vec<_> = spawns.iter().filter(|o| { let n = name_of(o); n.contains("initial") }).map(|o| cand(o)).collect();
    if let Some((p, f)) = pick_by_team(&initial, centre) {
        let (pos, yaw, pitch) = stand_at(p, f, centre);
        return Some((pos, yaw, pitch, SpawnCamSource::VariantInitialSpawn));
    }
    let any: Vec<_> = spawns.iter().map(|o| cand(o)).collect();
    if let Some((p, f)) = pick_by_team(&any, centre) {
        let (pos, yaw, pitch) = stand_at(p, f, centre);
        return Some((pos, yaw, pitch, SpawnCamSource::VariantSpawn));
    }
    None
}

/// Tiers 4-6: the scenario's starting locations / spawn placements / cinematic camera.
pub fn scenario_pose(cache: &H4Cache, placements: &[H4Placement]) -> Option<(Vec3, f32, f32, SpawnCamSource)> {
    // 4. spawn placements by kind, nearest the spawn centroid
    let tagged = |suffix: &str| -> Vec<(Vec3, Vec3)> {
        placements.iter()
            .filter(|p| cache.tag_name(p.palette_tag).ends_with(suffix))
            .map(|p| (Vec3::from(p.pos), p.basis.map(|b| b.0).unwrap_or_else(|| euler_to_basis(p.rot).0)))
            .collect()
    };
    let all_spawns = [tagged("initial_spawn_point"), tagged("\\respawn_point"), tagged("respawn_point_invisible")];
    let centre = centroid(all_spawns.iter().flatten().map(|c| c.0)).unwrap_or(Vec3::ZERO);
    for kind in &all_spawns {
        if let Some((p, f)) = kind.iter().min_by(|a, b| (a.0 - centre).length_squared().total_cmp(&(b.0 - centre).length_squared())) {
            let (pos, yaw, pitch) = stand_at(*p, *f, centre);
            return Some((pos, yaw, pitch, SpawnCamSource::ScenarioSpawn));
        }
    }
    // 5. player starting locations (first one - the block is 0-1 elements on every MP map, an
    //    editor start that can sit outside the play space, hence after the placements)
    if let Some(s) = load_starting_locations(cache).first() {
        return Some((Vec3::from(s.pos) + Vec3::Z * EYE_HEIGHT, s.yaw, s.pitch, SpawnCamSource::ScenarioStart));
    }
    // 6. the shipped map's own loadout camera (the non-fallback one first)
    for suffix in ["\\mp_cinematic_camera", "mp_cinematic_fallback_camera"] {
        if let Some((p, f)) = tagged(suffix).first() {
            let (pos, yaw, pitch) = camera_at(*p, *f);
            return Some((pos, yaw, pitch, SpawnCamSource::ScenarioCamera));
        }
    }
    None
}

/// The spawn camera for a loaded editor scene, stood off from the map geometry / objects by
/// `STANDOFF_WU` (the scene's `ObjectScene::standoff`; a scene without a soup leaves it as is).
pub fn spawn_camera(es: &H4ObjectScene) -> Option<SpawnCam> {
    let picked = es.variant().and_then(|(_, v)| variant_pose(v, es.palette()))
        .or_else(|| scenario_pose(es.cache(), es.placements()))?;
    let (raw_pos, yaw, pitch, source) = picked;
    // A camera OBJECT's own marker model surrounds its position, so a stand-off against the
    // objects would only push the camera off itself: cameras clear the BSP alone; spawn poses
    // (eye height above a marker that may sit inside a Forge block) clear BSP + objects.
    let pos = if es.soup_len() == 0 {
        raw_pos
    } else if matches!(source, SpawnCamSource::VariantCamera | SpawnCamSource::ScenarioCamera) {
        crate::scene::standoff_with(&|o, d, m| es.raycast_scene(o, d).map(|p| (p - o).length()).filter(|t| *t <= m), raw_pos, STANDOFF_WU)
    } else {
        es.standoff(raw_pos, STANDOFF_WU)
    };
    Some(SpawnCam { pos, yaw, pitch, source, raw_pos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::h4::cache::maps_dir;
    use crate::h4::mvar::{load_forge_palette, parse_h4_variant, variant_dirs};
    use crate::h4::objects::load_placements;

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
        let (pos, yaw, pitch, src) = scenario_pose(&c, &placements).expect("Ravine has spawn placements");
        assert_eq!(src, SpawnCamSource::ScenarioSpawn);
        // the three initial_spawn_point placements sit at y 48.24, z 1.10 facing -Y (yaw -pi/2)
        assert!((pos.y - 48.25).abs() < 0.1 && (pos.z - (1.10 + EYE_HEIGHT)).abs() < 0.05, "{pos:?}");
        assert!((yaw + std::f32::consts::FRAC_PI_2).abs() < 0.05 && pitch == 0.0, "yaw {yaw} pitch {pitch}");
    }

    #[test]
    fn variant_pose_prefers_the_loadout_camera_then_initial_spawns() {
        let Some(c) = open("ca_forge_ravine") else { return };
        let Some(vp) = variant_dirs().into_iter().map(|d| d.join("ca_forge_ravine_settler.mvar")).find(|p| p.is_file()) else { return };
        let palette = load_forge_palette(&c);
        let mut v = parse_h4_variant(&vp).expect("variant");
        let has_cam = v.objects.iter().any(|o| o.object_type == TYPE_LOADOUT_CAMERA);
        let (pos, _, _, src) = variant_pose(&v, &palette).expect("Settler has spawns");
        assert_eq!(src, if has_cam { SpawnCamSource::VariantCamera } else { SpawnCamSource::VariantInitialSpawn });
        if has_cam {
            let cam = v.objects.iter().find(|o| o.object_type == TYPE_LOADOUT_CAMERA).unwrap();
            assert!(v.objects.iter().filter(|o| o.object_type == TYPE_LOADOUT_CAMERA).any(|o| Vec3::from(o.pos) == pos), "camera pose is a camera object: {pos:?} vs {:?}", cam.pos);
        }
        // without cameras: an INITIAL spawn of the first team present, at eye height
        v.objects.retain(|o| o.object_type != TYPE_LOADOUT_CAMERA);
        let (pos, _, pitch, src) = variant_pose(&v, &palette).expect("spawns");
        assert_eq!(src, SpawnCamSource::VariantInitialSpawn);
        assert_eq!(pitch, 0.0);
        let initial: Vec<_> = v.objects.iter().filter(|o| o.object_type == TYPE_PLAYER_SPAWN && palette[o.quota.unwrap() as usize].name.contains("initial")).collect();
        let want_team = TEAM_ORDER.iter().copied().find(|t| initial.iter().any(|o| o.team == *t));
        let chosen = initial.iter().find(|o| (Vec3::from(o.pos) + Vec3::Z * EYE_HEIGHT - pos).length() < 1e-3).expect("pose is at an initial spawn");
        if let Some(t) = want_team { assert_eq!(chosen.team, t, "team order red/blue/neutral"); }
    }
}
