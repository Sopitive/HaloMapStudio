//! The "Hard floor (world bounds)" switch -- the total volume a
//! player or object can occupy before the engine's invisible wall, plus the "Playable BSP bounds"
//! box as a second, smaller reference.
//!
//! WHAT STOPS YOU (RE'd from reach_tag_test.exe, imagebase 0x140000000):
//!   * `havok_world_new` (sub_1401471F0, omaha\physics\havok.cpp) sizes the Havok BROADPHASE
//!     AABB by iterating EVERY active structure BSP (`global_structure_bsp_first_active_index_get`
//!     / next), taking each sbsp's physics `mopp bounds min/max` (sbsp+0x41C / +0x428, inside the
//!     global_structure_physics_struct at sbsp+0x40C; field table 0x141EE0F40) into a running
//!     min/max, then padding every axis by 64 wu (`v47 - 64 ... v52 + 64`) before building the
//!     `hkpWorldCinfo` (+ a broad-phase border listener). Nothing can be simulated outside that
//!     box: a character proxy stops at the border as at a wall; a rigid body that exits is deleted
//!     ("objects: rigid body has exited broadphase at (...). The associated object has been
//!     deleted."). The non-playable / shared sky BSPs ARE active BSPs, so they count -- which is
//!     why the box reaches through Forge World's second (sky) BSP.
//!   * Below the box there is no engine surface at all: a biped in ground physics mode falling
//!     faster than the game-globals terminal velocity dies (sub_140E177D0, character_physics.cpp:
//!     fell-to-death damage effect 41; flagged objects erased "reached terminal velocity outside
//!     world").
//!   * The playable BSPs' `world bounds x/y/z` (sbsp+0xF0) are the smaller box the
//!     network position encoding clamps objects to; kept here as the optional second overlay.
//!
//! The mopp bounds sit within ~0.1 wu of each BSP's world bounds, so the world box = union of
//! every structure BSP's world bounds +-64 wu (mopp bounds when readable, world bounds otherwise).
//! Forge World: z -75.2 .. 1265.1 -- the floor is the island BSP's mopp z-min (-11.2) - 64, ABOVE
//! the soft-ceiling floor at -147, so with the soft ceiling off you fall to -75 and stop on the
//! broadphase border: the "hard floor" the user hits.
//!
//! Overlay: full AABB edges (x-ray lane, magenta) + a floor grid at the box's z-min (x-ray) + a
//! translucent floor quad (zone lane). OFF by default, persisted (`hard_floor.txt`), script
//! `hardfloor on|off|get`, batch env `HMS_HARD_FLOOR=1`. The playable box: violet, `playable_bounds.txt`,
//! `playablebounds on|off|get`, `HMS_PLAYABLE_BOUNDS=1`.

use crate::scene::StructureBspInfo;
use glam::Vec3;

const SETTING_FILE: &str = "hard_floor.txt";
const PLAYABLE_SETTING_FILE: &str = "playable_bounds.txt";

/// Havok broadphase padding around the union of the mopp bounds (havok_world_new: +-64.0).
pub const BROADPHASE_PAD: f32 = 64.0;

/// Magenta (world box): distinct from the soft-ceiling reds / oranges / yellows, the trigger boxes
/// and the playable box.
pub const COLOR: [f32; 3] = [0.95, 0.25, 0.95];
/// Violet (playable BSP bounds).
pub const PLAYABLE_COLOR: [f32; 3] = [0.55, 0.35, 1.0];

fn load_flag(dir: &std::path::Path, file: &str) -> bool {
    std::fs::read_to_string(dir.join(file)).ok().and_then(|s| crate::map_spawns::parse_flag(&s)).unwrap_or(false)
}

fn save_flag(dir: &std::path::Path, file: &str, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(file), if on { "1" } else { "0" });
}

/// The persisted "Hard floor (world bounds)" toggle. DEFAULT HIDDEN.
pub fn load_setting() -> bool {
    crate::project::settings_dir().map(|d| load_from_dir(&d)).unwrap_or(false)
}
pub fn load_from_dir(dir: &std::path::Path) -> bool { load_flag(dir, SETTING_FILE) }
pub fn save_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() { save_to_dir(&dir, on); }
}
pub fn save_to_dir(dir: &std::path::Path, on: bool) { save_flag(dir, SETTING_FILE, on) }

/// The persisted "Playable BSP bounds" toggle. DEFAULT HIDDEN.
pub fn load_playable_setting() -> bool {
    crate::project::settings_dir().map(|d| load_flag(&d, PLAYABLE_SETTING_FILE)).unwrap_or(false)
}
pub fn save_playable_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() { save_flag(&dir, PLAYABLE_SETTING_FILE, on); }
}

/// Headless / batch: `HMS_HARD_FLOOR=1|on` draws the world box; `HMS_PLAYABLE_BOUNDS=1` the playable box.
pub fn env_show_hard_floor() -> bool {
    std::env::var("HMS_HARD_FLOOR").ok().and_then(|v| crate::map_spawns::parse_flag(&v)).unwrap_or(false)
}
pub fn env_show_playable_bounds() -> bool {
    std::env::var("HMS_PLAYABLE_BOUNDS").ok().and_then(|v| crate::map_spawns::parse_flag(&v)).unwrap_or(false)
}

/// The engine's world box: the Havok broadphase AABB.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldBox {
    pub min: Vec3,
    pub max: Vec3,
    /// How many structure BSPs contributed.
    pub bsps: usize,
    /// "mopp bounds" (the engine's own source) or "world bounds" (fallback when unreadable).
    pub source: &'static str,
}

/// The union of every structure BSP's physics mopp bounds (`mopp` = (bsp index, min, max) from
/// `SceneController::structure_bsp_mopp_bounds`), padded by 64 wu -- exactly havok_world_new.
/// Falls back to the union of the sbsp world bounds when no mopp bounds were readable.
pub fn world_box(bsps: &[StructureBspInfo], mopp: &[(usize, Vec3, Vec3)]) -> Option<WorldBox> {
    let (mut mn, mut mx, mut n) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY), 0usize);
    let mut source = "mopp bounds";
    for (_, a, b) in mopp {
        mn = mn.min(*a);
        mx = mx.max(*b);
        n += 1;
    }
    if n == 0 {
        source = "world bounds";
        for b in bsps.iter().filter(|b| b.has_bounds()) {
            mn = mn.min(b.min);
            mx = mx.max(b.max);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some(WorldBox { min: mn - Vec3::splat(BROADPHASE_PAD), max: mx + Vec3::splat(BROADPHASE_PAD), bsps: n, source })
}

/// The BSPs the playable box is drawn for: playable (reference flag bit 10 clear) with bounds.
pub fn floor_bsps(bsps: &[StructureBspInfo]) -> Vec<&StructureBspInfo> {
    bsps.iter().filter(|b| !b.non_playable() && b.has_bounds()).collect()
}

/// The lowest floor height among the playable BSPs (None without bounds).
pub fn floor_z(bsps: &[StructureBspInfo]) -> Option<f32> {
    floor_bsps(bsps).iter().map(|b| b.min.z).min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

/// One AABB as overlay geometry: translucent floor quad (zone lane, RGBA) + x-ray lines (the 12
/// edges bright, floor diagonals + grid dimmed). `grid` = grid lines per floor axis.
#[allow(clippy::type_complexity)]
pub fn box_geometry(mn: Vec3, mx: Vec3, rgb: [f32; 3], grid: usize, tris: &mut Vec<([f32; 3], [f32; 4])>, segs: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>) {
    let dim = [rgb[0] * 0.75, rgb[1] * 0.75, rgb[2] * 0.75];
    let c = |x: f32, y: f32, z: f32| [x, y, z];
    let (a, bq, cq, d) = (c(mn.x, mn.y, mn.z), c(mx.x, mn.y, mn.z), c(mx.x, mx.y, mn.z), c(mn.x, mx.y, mn.z));
    let rgba = [rgb[0], rgb[1], rgb[2], 0.30];
    for v in [a, bq, cq, a, cq, d] {
        tris.push((v, rgba));
    }
    let corner = |i: usize| c(if i & 1 != 0 { mx.x } else { mn.x }, if i & 2 != 0 { mx.y } else { mn.y }, if i & 4 != 0 { mx.z } else { mn.z });
    for (i, j) in [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (3, 7), (2, 6)] {
        segs.push((corner(i), corner(j), rgb));
    }
    segs.push((a, cq, dim));
    segs.push((bq, d, dim));
    for k in 1..grid {
        let t = k as f32 / grid as f32;
        let x = mn.x + (mx.x - mn.x) * t;
        let y = mn.y + (mx.y - mn.y) * t;
        segs.push((c(x, mn.y, mn.z), c(x, mx.y, mn.z), dim));
        segs.push((c(mn.x, y, mn.z), c(mx.x, y, mn.z), dim));
    }
}

/// The world box overlay (magenta).
#[allow(clippy::type_complexity)]
pub fn world_geometry(wb: &WorldBox, grid: usize) -> (Vec<([f32; 3], [f32; 4])>, Vec<([f32; 3], [f32; 3], [f32; 3])>) {
    let (mut tris, mut segs) = (Vec::new(), Vec::new());
    box_geometry(wb.min, wb.max, COLOR, grid, &mut tris, &mut segs);
    (tris, segs)
}

/// The playable-BSP boxes overlay (violet): one box + floor per playable BSP with bounds.
#[allow(clippy::type_complexity)]
pub fn overlay_geometry(bsps: &[StructureBspInfo], grid: usize) -> (Vec<([f32; 3], [f32; 4])>, Vec<([f32; 3], [f32; 3], [f32; 3])>) {
    let (mut tris, mut segs) = (Vec::new(), Vec::new());
    for b in floor_bsps(bsps) {
        box_geometry(b.min, b.max, PLAYABLE_COLOR, grid, &mut tris, &mut segs);
    }
    (tris, segs)
}

/// One-line status for the script `hardfloor get` / the View menu tooltip.
pub fn status_line(show: bool, wb: Option<&WorldBox>) -> String {
    match wb {
        Some(w) => format!(
            "hard floor: {} (world box x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}; floor z {:.1}; {} BSP(s), {} +-{:.0})",
            if show { "shown" } else { "hidden" },
            w.min.x, w.max.x, w.min.y, w.max.y, w.min.z, w.max.z, w.min.z, w.bsps, w.source, BROADPHASE_PAD
        ),
        None => format!("hard floor: {} (no structure BSP bounds)", if show { "shown" } else { "hidden" }),
    }
}

/// One-line status for `playablebounds get`.
pub fn playable_status_line(show: bool, bsps: &[StructureBspInfo]) -> String {
    let z = floor_z(bsps).map(|z| format!("floor z {z:.1}")).unwrap_or_else(|| String::from("no bounds"));
    format!("playable bounds: {} ({} playable BSP(s) of {}; {z})", if show { "shown" } else { "hidden" }, floor_bsps(bsps).len(), bsps.len())
}

/// Per-BSP listing (`hardfloor get` / the headless log): name, playable?, world bounds, mopp bounds.
pub fn listing(bsps: &[StructureBspInfo], mopp: &[(usize, Vec3, Vec3)]) -> Vec<String> {
    bsps.iter()
        .map(|b| {
            let leaf = b.name.rsplit('\\').next().unwrap_or(b.name.as_str());
            let what = if b.non_playable() { "not playable" } else { "playable" };
            let m = mopp.iter().find(|(i, _, _)| *i == b.index)
                .map(|(_, a, c)| format!("mopp z {:.1}..{:.1}", a.z, c.z))
                .unwrap_or_else(|| String::from("no mopp bounds"));
            format!(
                "  [{}] {:<28} {:<12} world z {:.1}..{:.1}  x {:.1}..{:.1}  y {:.1}..{:.1}  {m}",
                b.index, leaf, what, b.min.z, b.max.z, b.min.x, b.max.x, b.min.y, b.max.y
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bsp(index: usize, flags: u16, min: [f32; 3], max: [f32; 3]) -> StructureBspInfo {
        StructureBspInfo { index, tag: index as u32, name: format!("levels\\multi\\m\\bsp{index}"), flags, min: Vec3::from(min), max: Vec3::from(max) }
    }

    #[test]
    fn settings_round_trip_and_default_hidden() {
        let dir = std::env::temp_dir().join(format!("hms_hard_floor_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!load_from_dir(&dir));
        save_to_dir(&dir, true);
        assert!(load_from_dir(&dir));
        assert!(!load_flag(&dir, PLAYABLE_SETTING_FILE));
        save_flag(&dir, PLAYABLE_SETTING_FILE, true);
        assert!(load_flag(&dir, PLAYABLE_SETTING_FILE));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn world_box_is_the_padded_union_of_every_bsp_mopp_bounds() {
        // Forge World: playable island + non-playable shared sky BSP; the sky BSP COUNTS.
        let bsps = [
            bsp(0, 0x0200, [-203.6, -360.8, -25.0], [277.7, 324.9, 117.8]),
            bsp(1, 0x0e00, [-4783.4, -5289.7, -2.8], [4243.4, 3737.1, 1201.0]),
        ];
        let mopp = [
            (0usize, Vec3::new(-193.6, -350.1, -11.2), Vec3::new(272.5, 297.5, 117.9)),
            (1usize, Vec3::new(-4783.5, -5289.8, -2.9), Vec3::new(4243.5, 3737.2, 1201.1)),
        ];
        let w = world_box(&bsps, &mopp).unwrap();
        assert_eq!(w.source, "mopp bounds");
        assert_eq!(w.bsps, 2);
        assert!((w.min.z - (-11.2 - 64.0)).abs() < 1e-3 && (w.max.z - (1201.1 + 64.0)).abs() < 1e-3, "{w:?}");
        assert!((w.min.x - (-4783.5 - 64.0)).abs() < 1e-3 && (w.max.y - (3737.2 + 64.0)).abs() < 1e-3, "{w:?}");
        // No mopp bounds readable -> world bounds union, still padded, still includes the sky BSP.
        let f = world_box(&bsps, &[]).unwrap();
        assert_eq!(f.source, "world bounds");
        assert!((f.min.z - (-25.0 - 64.0)).abs() < 1e-3 && (f.max.z - (1201.0 + 64.0)).abs() < 1e-3, "{f:?}");
        assert_eq!(world_box(&[], &[]), None);
        let (tris, segs) = world_geometry(&w, 4);
        assert_eq!(tris.len(), 6);
        assert!(tris.iter().all(|(p, c)| (p[2] - w.min.z).abs() < 1e-3 && c[0] > 0.9 && c[2] > 0.9));
        assert_eq!(segs.len(), 12 + 2 + 6);
        let s = status_line(true, Some(&w));
        assert!(s.contains("shown") && s.contains("floor z -75.2") && s.contains("2 BSP(s), mopp bounds +-64"), "{s}");
        assert!(status_line(false, None).contains("no structure BSP bounds"));
    }

    #[test]
    fn playable_box_uses_playable_bsps_only() {
        let bsps = [
            bsp(0, 0x0200, [-203.6, -360.8, -25.0], [277.7, 324.9, 117.8]),
            bsp(1, 0x0e00, [-4783.4, -5289.7, -2.8], [4243.4, 3737.1, 1201.0]),
            bsp(2, 0, [0.0; 3], [0.0; 3]),
        ];
        assert_eq!(floor_bsps(&bsps).len(), 1);
        assert_eq!(floor_z(&bsps), Some(-25.0));
        let (tris, segs) = overlay_geometry(&bsps, 4);
        assert_eq!(tris.len(), 6);
        assert_eq!(segs.iter().filter(|(_, _, c)| *c == PLAYABLE_COLOR).count(), 12);
        let s = playable_status_line(true, &bsps);
        assert!(s.contains("1 playable BSP(s) of 3") && s.contains("floor z -25.0"), "{s}");
        let l = listing(&bsps, &[(0, Vec3::new(0.0, 0.0, -11.2), Vec3::new(1.0, 1.0, 117.9))]);
        assert!(l[0].contains("playable") && l[0].contains("mopp z -11.2..117.9"), "{}", l[0]);
        assert!(l[1].contains("not playable") && l[1].contains("no mopp bounds"), "{}", l[1]);
    }
}
