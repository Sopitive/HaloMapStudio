//! Where the camera lands on a map load / the "teleport to spawn" action - ONE rule for every
//! game (Reach, Halo 4, H2A, Halo 3), so the camera always opens INSIDE the playable map.
//!
//! Each game gathers its own candidates (the scenario's spawn placements, the variant's spawns,
//! the scnr `player starting locations` block, camera objects) and its playable volumes (soft
//! ceilings, playable BSP bounds, scene bounds) and hands them to [`choose`], which applies the
//! shared selection rule:
//!
//!   1. a SPAWN POINT, by kind: the variant's initial spawn > any variant spawn > the scenario's
//!      `initial_spawn_point` > `respawn_point` > `respawn_point_invisible`. Within a kind the
//!      first team in red / blue / neutral / any order that has one, and within that team the
//!      marker nearest the centroid of every spawn (a deterministic "middle of the map" pick).
//!      Standing pose: eye height [`EYE_HEIGHT`] above the marker, level, facing the marker's
//!      stored yaw (else the spawn centroid);
//!   2. the scnr `player starting locations` block (position + stored facing yaw / pitch; the
//!      first element - campaign / Firefight insertion points, the editor start on MP maps);
//!   3. a camera object (the variant's Forge loadout camera / the scenario's `mp_cinematic_camera`
//!      / `spawning_camera`): its own position and full facing;
//!   4. no spawn-like source at all: a point INSIDE the soft-ceiling volume - the centre of the
//!      soft-kill ceilings' box (any kind when there is no kill ceiling), pulled to the ground by
//!      a downward ray from the top of the volume, eye height above the floor;
//!   5. else the playable-bounds centre (playable BSP world bounds / playable clusters), pulled to
//!      the ground the same way; else the scene bounds;
//!   6. else `None` - the callers keep their overview framing.
//!
//! Every pose is then stood off the map geometry + placed objects by the shared solver
//! (`scene::standoff_with` over the caller's solid-ray primitive) with the SMALL load-time
//! clearance [`STANDOFF_WU`]: enough to leave a wall / floor a marker sits in, not enough to
//! hoist a standing pose. Non-finite results are rejected (a NaN camera renders black).
//!
//! Wiring: Reach `SceneController::spawn_camera` (scene.rs), Halo 4 / H2A `h4::spawncam`
//! (variant + scenario candidates over the pure-Rust reader), Halo 3 `h3/gui.rs` (see the
//! `classify_tag_path` helper for the respawn_point placements).

use glam::Vec3;

/// Eye height above a spawn marker (player ~0.7 wu tall; the Reach `variant_spawn_camera` value).
pub const EYE_HEIGHT: f32 = 0.62;
/// Load-time stand-off clearance (wu). The interactive 2 wu would lift every standing pose to
/// head-and-a-half height; 0.35 clears a floor / wall the marker is embedded in.
pub const STANDOFF_WU: f32 = 0.35;
/// Team order for the pick: red, blue, neutral (8), then anything.
const TEAM_ORDER: [i8; 3] = [0, 1, 8];

/// What a candidate IS, in selection order (lower rank wins).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// A map-variant player spawn whose palette name says "initial".
    VariantInitialSpawn,
    /// Any other map-variant player spawn.
    VariantSpawn,
    /// Scenario `initial_spawn_point` placement.
    InitialSpawn,
    /// Scenario `respawn_point` placement.
    RespawnPoint,
    /// Scenario `respawn_point_invisible` placement.
    RespawnPointInvisible,
    /// scnr `player starting locations` element (stored yaw + pitch).
    StartingLocation,
    /// The variant's Forge loadout camera object (full facing).
    VariantCamera,
    /// The scenario's `mp_cinematic_camera` / `spawning_camera` placement (full facing).
    ScenarioCamera,
}

impl Kind {
    pub fn is_camera(self) -> bool { matches!(self, Kind::VariantCamera | Kind::ScenarioCamera) }
    pub fn is_spawn(self) -> bool { !self.is_camera() && self != Kind::StartingLocation }
    pub fn label(self) -> &'static str {
        match self {
            Kind::VariantInitialSpawn => "variant initial spawn",
            Kind::VariantSpawn => "variant spawn",
            Kind::InitialSpawn => "scenario initial_spawn_point",
            Kind::RespawnPoint => "scenario respawn_point",
            Kind::RespawnPointInvisible => "scenario respawn_point_invisible",
            Kind::StartingLocation => "scenario player starting location",
            Kind::VariantCamera => "variant loadout camera",
            Kind::ScenarioCamera => "scenario camera",
        }
    }
}

/// One spawn-like source. `fwd` may be zero (no facing known); `team` -1 = unknown / none;
/// `pitch` is only honoured for starting locations (spawns stand level, cameras use `fwd.z`).
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub kind: Kind,
    pub pos: Vec3,
    pub fwd: Vec3,
    pub team: i8,
    pub pitch: f32,
}

impl Candidate {
    pub fn new(kind: Kind, pos: Vec3, fwd: Vec3) -> Self { Candidate { kind, pos, fwd, team: -1, pitch: 0.0 } }
    pub fn team(mut self, team: i8) -> Self { self.team = team; self }
    pub fn pitch(mut self, pitch: f32) -> Self { self.pitch = pitch; self }
}

/// The playable volumes a map knows about, all optional.
#[derive(Clone, Default, Debug)]
pub struct Volumes {
    /// Soft-KILL ceilings (the map floor / play-space walls) as world-space triangles.
    pub kill_ceilings: Vec<[Vec3; 3]>,
    /// Every other soft ceiling (acceleration / slip) - used only without kill ceilings.
    pub other_ceilings: Vec<[Vec3; 3]>,
    /// Playable bounds (playable BSP world bounds / playable clusters).
    pub playable_bounds: Option<(Vec3, Vec3)>,
    /// The whole decoded scene's AABB (last resort).
    pub scene_bounds: Option<(Vec3, Vec3)>,
}

/// Which rule produced the pose (for the load log / HMS_SPAWNCAM diagnostics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Spawn(Kind),
    SoftCeiling,
    PlayableBounds,
    SceneBounds,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Spawn(k) => k.label(),
            Source::SoftCeiling => "soft ceiling volume centre",
            Source::PlayableBounds => "playable bounds centre",
            Source::SceneBounds => "scene bounds centre",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub source: Source,
    /// Position BEFORE the stand-off push (diagnostics).
    pub raw_pos: Vec3,
}

impl Pose {
    pub fn tuple(&self) -> (Vec3, f32, f32) { (self.pos, self.yaw, self.pitch) }
    /// One line for the load log.
    pub fn describe(&self) -> String {
        format!("source={} pos=({:.2},{:.2},{:.2}) yaw={:.1} pitch={:.1} stand-off moved {:.2} wu",
            self.source.label(), self.pos.x, self.pos.y, self.pos.z, self.yaw.to_degrees(), self.pitch.to_degrees(), (self.pos - self.raw_pos).length())
    }
}

/// A solid-ray primitive: `(origin, dir, max) -> hit distance` over the map geometry + objects.
pub type Raycast<'a> = &'a dyn Fn(Vec3, Vec3, f32) -> Option<f32>;

/// Classify an object tag path (any game) as a spawn-like placement, by its leaf name.
/// `objects\multi\spawning\respawn_point` -> `RespawnPoint`; respawn ZONES are volumes, never
/// a camera target, so they classify as None.
pub fn classify_tag_path(tag_path: &str) -> Option<Kind> {
    if tag_path.is_empty() { return None; }
    let p = tag_path.to_ascii_lowercase().replace('/', "\\");
    let leaf = p.rsplit('\\').next().unwrap_or(p.as_str());
    let leaf = leaf.split('.').next().unwrap_or(leaf);
    Some(match leaf {
        "initial_spawn_point" => Kind::InitialSpawn,
        "respawn_point" => Kind::RespawnPoint,
        "respawn_point_invisible" => Kind::RespawnPointInvisible,
        "mp_cinematic_camera" | "mp_cinematic_fallback_camera" | "spawning_camera" => Kind::ScenarioCamera,
        _ if leaf.starts_with("initial_spawn") && !leaf.contains("zone") => Kind::InitialSpawn,
        _ if leaf.starts_with("respawn_point") && !leaf.contains("zone") => Kind::RespawnPoint,
        _ => return None,
    })
}

fn yaw_of(f: Vec3) -> Option<f32> { (f.x.hypot(f.y) > 0.1).then(|| f.y.atan2(f.x)) }

fn centroid(points: impl Iterator<Item = Vec3>) -> Option<Vec3> {
    let (mut sum, mut n) = (Vec3::ZERO, 0usize);
    for p in points { sum += p; n += 1; }
    (n > 0).then(|| sum / n as f32)
}

/// Of `cands` (all one kind): the first team in `TEAM_ORDER` that has one (else any team),
/// then the one nearest `centre` (stable: the first on a tie).
fn pick_by_team<'a>(cands: &[&'a Candidate], centre: Vec3) -> Option<&'a Candidate> {
    let nearest = |it: &mut dyn Iterator<Item = &&'a Candidate>| {
        it.min_by(|a, b| (a.pos - centre).length_squared().total_cmp(&(b.pos - centre).length_squared())).copied()
    };
    for team in TEAM_ORDER {
        if let Some(c) = nearest(&mut cands.iter().filter(|c| c.team == team)) { return Some(c); }
    }
    nearest(&mut cands.iter())
}

/// The chosen candidate (deterministic), before any stand-off. Public for the callers' tests.
pub fn pick(cands: &[Candidate]) -> Option<&Candidate> {
    let finite: Vec<&Candidate> = cands.iter().filter(|c| c.pos.is_finite() && c.pos.abs().max_element() < 1.0e5).collect();
    let centre = centroid(finite.iter().filter(|c| c.kind.is_spawn()).map(|c| c.pos))
        .or_else(|| centroid(finite.iter().map(|c| c.pos)))?;
    let mut kinds: Vec<Kind> = finite.iter().map(|c| c.kind).collect();
    kinds.sort();
    kinds.dedup();
    for k in kinds {
        let of_kind: Vec<&Candidate> = finite.iter().copied().filter(|c| c.kind == k).collect();
        if let Some(c) = pick_by_team(&of_kind, centre) { return Some(c); }
    }
    None
}

/// The pose at a candidate: standing at eye height facing the stored yaw (spawns), the stored
/// yaw + pitch (starting locations) or the object's full facing (cameras). `look_to` = where to
/// face when the marker has no usable forward.
fn pose_at(c: &Candidate, look_to: Vec3) -> (Vec3, f32, f32) {
    if c.kind.is_camera() {
        let f = c.fwd.normalize_or_zero();
        let yaw = yaw_of(f).or_else(|| yaw_of(look_to - c.pos)).unwrap_or(0.0);
        return (c.pos, yaw, f.z.clamp(-1.0, 1.0).asin());
    }
    let eye = c.pos + Vec3::Z * EYE_HEIGHT;
    let yaw = yaw_of(c.fwd).or_else(|| yaw_of(look_to - eye)).unwrap_or(0.0);
    let pitch = if c.kind == Kind::StartingLocation && c.pitch.is_finite() { c.pitch.clamp(-1.5, 1.5) } else { 0.0 };
    (eye, yaw, pitch)
}

fn tri_bounds(tris: &[[Vec3; 3]]) -> Option<(Vec3, Vec3)> {
    let (mut mn, mut mx, mut n) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY), 0usize);
    for t in tris {
        for v in t {
            if !v.is_finite() || v.abs().max_element() > 1.0e5 { continue; }
            mn = mn.min(*v);
            mx = mx.max(*v);
            n += 1;
        }
    }
    (n > 0 && (mx - mn).max_element() > 0.0).then_some((mn, mx))
}

/// The centre of a playable box pulled to the ground: a ray straight down from the box's top
/// at the centre's x/y lands the eye `EYE_HEIGHT` above the floor; without a hit (no raycast /
/// open air) the box centre itself.
fn grounded_centre(mn: Vec3, mx: Vec3, raycast: Option<Raycast>) -> Vec3 {
    let c = (mn + mx) * 0.5;
    let Some(rc) = raycast else { return c };
    let top = Vec3::new(c.x, c.y, mx.z - 0.05);
    let span = (mx.z - mn.z).max(1.0) + 1.0;
    match rc(top, Vec3::NEG_Z, span) {
        // a floor within the box (a hit right at the ceiling / above the box's floor line is fine too)
        Some(t) if t.is_finite() && top.z - t >= mn.z - 1.0 => Vec3::new(c.x, c.y, top.z - t + EYE_HEIGHT),
        _ => c,
    }
}

/// Apply the shared selection rule. `raycast` = the game's solid-ray primitive for the ground
/// pull + stand-off (None = no geometry yet: poses are used raw).
pub fn choose(cands: &[Candidate], vols: &Volumes, raycast: Option<Raycast>) -> Option<Pose> {
    let (raw_pos, yaw, pitch, source) = if let Some(c) = pick(cands) {
        let centre = centroid(cands.iter().filter(|x| x.kind.is_spawn() && x.pos.is_finite()).map(|x| x.pos))
            .or_else(|| vols.playable_bounds.or(vols.scene_bounds).map(|(a, b)| (a + b) * 0.5))
            .unwrap_or(c.pos);
        let (p, y, pi) = pose_at(c, centre);
        (p, y, pi, Source::Spawn(c.kind))
    } else if let Some((mn, mx)) = tri_bounds(&vols.kill_ceilings).or_else(|| tri_bounds(&vols.other_ceilings)) {
        let p = grounded_centre(mn, mx, raycast);
        (p, 0.0, 0.0, Source::SoftCeiling)
    } else if let Some((mn, mx)) = vols.playable_bounds.filter(|(a, b)| a.is_finite() && b.is_finite() && (*b - *a).max_element() > 0.0) {
        let p = grounded_centre(mn, mx, raycast);
        (p, 0.0, 0.0, Source::PlayableBounds)
    } else if let Some((mn, mx)) = vols.scene_bounds.filter(|(a, b)| a.is_finite() && b.is_finite() && (*b - *a).max_element() > 0.0) {
        let p = grounded_centre(mn, mx, raycast);
        (p, 0.0, 0.0, Source::SceneBounds)
    } else {
        return None;
    };
    if !raw_pos.is_finite() || !yaw.is_finite() || !pitch.is_finite() { return None; }
    let pos = match raycast {
        Some(rc) => crate::scene::standoff_with(rc, raw_pos, STANDOFF_WU),
        None => raw_pos,
    };
    if !pos.is_finite() { return None; }
    Some(Pose { pos, yaw, pitch, source, raw_pos })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(kind: Kind, x: f32, team: i8) -> Candidate { Candidate::new(kind, Vec3::new(x, 0.0, 1.0), Vec3::X).team(team) }

    #[test]
    fn classify_by_leaf_name() {
        assert_eq!(classify_tag_path("objects\\multi\\spawning\\respawn_point"), Some(Kind::RespawnPoint));
        assert_eq!(classify_tag_path("objects/multi/spawning/initial_spawn_point.scenery"), Some(Kind::InitialSpawn));
        assert_eq!(classify_tag_path("objects\\multi\\spawning\\respawn_point_invisible"), Some(Kind::RespawnPointInvisible));
        assert_eq!(classify_tag_path("objects\\multi\\spawning\\respawn_zone"), None);
        assert_eq!(classify_tag_path("objects\\multi\\spawning\\respawn_zone_weak"), None);
        assert_eq!(classify_tag_path("objects\\multi\\generic\\mp_cinematic_camera"), Some(Kind::ScenarioCamera));
        assert_eq!(classify_tag_path("objects\\multi\\spawning\\spawning_camera"), Some(Kind::ScenarioCamera));
        assert_eq!(classify_tag_path(""), None);
        assert_eq!(classify_tag_path("levels\\multi\\forge_halo\\forge_halo"), None);
    }

    #[test]
    fn spawn_kind_then_team_then_nearest_centroid() {
        let cands = [
            c(Kind::RespawnPoint, -10.0, 1),
            c(Kind::RespawnPoint, 2.0, 0),   // red, near the centroid -> wins its kind
            c(Kind::RespawnPoint, 9.0, 0),
            c(Kind::StartingLocation, 0.0, -1),
            c(Kind::ScenarioCamera, 0.0, -1),
        ];
        let p = pick(&cands).unwrap();
        assert_eq!(p.kind, Kind::RespawnPoint);
        assert_eq!(p.pos.x, 2.0);
        // an initial spawn (any team) beats every respawn point
        let mut with_initial = cands.to_vec();
        with_initial.push(c(Kind::InitialSpawn, 40.0, 8));
        assert_eq!(pick(&with_initial).unwrap().kind, Kind::InitialSpawn);
        // a variant spawn beats the scenario's
        with_initial.push(c(Kind::VariantSpawn, 41.0, -1));
        assert_eq!(pick(&with_initial).unwrap().kind, Kind::VariantSpawn);
        // starting locations only when no spawn exists; cameras after them
        assert_eq!(pick(&cands[3..]).unwrap().kind, Kind::StartingLocation);
        assert_eq!(pick(&cands[4..]).unwrap().kind, Kind::ScenarioCamera);
        // non-finite candidates are ignored
        let nan = [Candidate::new(Kind::InitialSpawn, Vec3::NAN, Vec3::X), cands[0]];
        assert_eq!(pick(&nan).unwrap().kind, Kind::RespawnPoint);
    }

    #[test]
    fn poses_stand_at_eye_height_facing_the_stored_yaw() {
        let cands = [Candidate::new(Kind::RespawnPoint, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0))];
        let p = choose(&cands, &Volumes::default(), None).unwrap();
        assert_eq!(p.source, Source::Spawn(Kind::RespawnPoint));
        assert!((p.pos - Vec3::new(1.0, 2.0, 3.0 + EYE_HEIGHT)).length() < 1e-5);
        assert!((p.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-5 && p.pitch == 0.0);
        // a camera keeps its own height and pitch
        let cam = [Candidate::new(Kind::VariantCamera, Vec3::new(0.0, 0.0, 10.0), Vec3::new(1.0, 0.0, -1.0).normalize())];
        let p = choose(&cam, &Volumes::default(), None).unwrap();
        assert_eq!(p.pos.z, 10.0);
        assert!((p.pitch + std::f32::consts::FRAC_PI_4).abs() < 1e-4);
        // a starting location keeps its stored pitch
        let sl = [Candidate::new(Kind::StartingLocation, Vec3::ZERO, Vec3::X).pitch(-0.3)];
        assert!((choose(&sl, &Volumes::default(), None).unwrap().pitch + 0.3).abs() < 1e-6);
    }

    #[test]
    fn no_spawn_falls_to_the_soft_ceiling_then_bounds_then_none() {
        // a kill floor at z = -5 under a 20x20 box: the centre is pulled to the floor at z 0
        let floor_hit = |o: Vec3, d: Vec3, max: f32| -> Option<f32> {
            if d.z < -0.9 && o.z > 0.0 { let t = o.z; (t <= max).then_some(t) } else { None }
        };
        let vols = Volumes {
            kill_ceilings: vec![[Vec3::new(-10.0, -10.0, -5.0), Vec3::new(10.0, -10.0, -5.0), Vec3::new(10.0, 10.0, 12.0)]],
            ..Default::default()
        };
        let p = choose(&[], &vols, Some(&floor_hit)).unwrap();
        assert_eq!(p.source, Source::SoftCeiling);
        assert!(p.pos.x.abs() < 1e-4 && p.pos.y.abs() < 1e-4, "{:?}", p.pos);
        assert!((p.pos.z - EYE_HEIGHT).abs() < 0.1, "grounded: {:?}", p.pos);
        // playable bounds without a raycast: the box centre
        let vols = Volumes { playable_bounds: Some((Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 6.0, 8.0))), ..Default::default() };
        let p = choose(&[], &vols, None).unwrap();
        assert_eq!(p.source, Source::PlayableBounds);
        assert_eq!(p.pos, Vec3::new(2.0, 3.0, 4.0));
        assert!(choose(&[], &Volumes::default(), None).is_none());
    }
}
