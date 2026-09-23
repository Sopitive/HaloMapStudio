//! The "Show hidden-block physics hulls" switch.
//!
//! A HIDDEN BLOCK is a forge-placed object whose render model draws nothing in-game (the
//! engine's `objects\multi\nut_blockers\*` invisible walls, and any ported equivalent): the
//! block is defined purely by its coll/phmo hull. HMS overlays that hull (orange translucent
//! volume + edge outline) so the user can SEE and EDIT the blocks a variant placed -- shipped
//! variants such as Anchor 9's default seal the map with dozens of `wall_xl` blockers along the
//! BSP. Shown BY DEFAULT (that is how the blocks are found); the switch is persisted in the
//! settings dir like `map_spawns.txt` for users who want the game's look.
//!
//! The overlay is ONLY ever built for variant-placed / user-placed objects (datum 0xD / 0xF,
//! `scene.rs` build_object_meshes): scenario (scnr) placements and BSP geometry never get a
//! hull, whatever this switch says. The switch only affects the hidden-block overlay lane
//! (`SceneController::blocker_overlay`): the selection wireframe, boundary zones, trigger volumes
//! and the collision/physics hull of the SELECTED object (their own View switches) are untouched,
//! and the blocks stay pickable either way.

const SETTING_FILE: &str = "physics_outlines.txt";

/// The persisted GUI toggle (settings dir). DEFAULT SHOWN: absent / unreadable / garbage = true.
pub fn load_setting() -> bool {
    match crate::project::settings_dir() {
        Some(dir) => load_from_dir(&dir),
        None => true,
    }
}

/// Read the toggle from an explicit settings directory (default true when absent).
pub fn load_from_dir(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join(SETTING_FILE))
        .ok()
        .and_then(|s| crate::map_spawns::parse_flag(&s))
        .unwrap_or(true)
}

pub fn save_setting(on: bool) {
    if let Some(dir) = crate::project::settings_dir() {
        save_to_dir(&dir, on);
    }
}

/// Write the toggle into an explicit settings directory.
pub fn save_to_dir(dir: &std::path::Path, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(SETTING_FILE), if on { "1" } else { "0" });
}

/// Headless / batch: `HMS_PHYSICS_OUTLINES=0|off` hides the hidden-block hulls; unset or anything
/// else = shown (the same default the GUI has, so a headless render matches a fresh install).
pub fn env_show_physics_outlines() -> bool {
    std::env::var("HMS_PHYSICS_OUTLINES").ok().and_then(|v| crate::map_spawns::parse_flag(&v)).unwrap_or(true)
}

/// One-line status for the script `outlines get` / the View menu tooltip. `tris` is the number of
/// hull triangles the scene currently holds for the overlay (0 = no hidden blocks on this map).
pub fn status_line(show: bool, tris: usize) -> String {
    if show {
        format!("hidden-block hulls: shown ({tris} hull triangle(s))")
    } else {
        format!("hidden-block hulls: hidden ({tris} hull triangle(s) not drawn)")
    }
}

/// The box-select test. An object is inside the axis-aligned world box when its
/// world AABB (a hidden block's physics hull) intersects it; an object with no decoded bounds yet
/// falls back to its centre point. Pure so the marquee rule is unit-testable.
pub fn box_overlaps(bmin: [f32; 3], bmax: [f32; 3], center: [f32; 3], aabb: Option<([f32; 3], [f32; 3])>) -> bool {
    match aabb {
        Some((mn, mx)) => (0..3).all(|i| mn[i] <= bmax[i] && mx[i] >= bmin[i]),
        None => (0..3).all(|i| center[i] >= bmin[i] && center[i] <= bmax[i]),
    }
}

/// The selection after an undo/redo/reload may name objects that no longer exist
/// (undo of a placement, a variant swap). Keep only the datums `present` still knows, and re-derive
/// the primary datum: kept when it survives, else the last surviving member, else none. Pure so
/// the invalidation rule is unit-testable.
pub fn prune_selection(selected: &[u32], primary: Option<u32>, present: impl Fn(u32) -> bool) -> (Vec<u32>, Option<u32>) {
    let kept: Vec<u32> = selected.iter().copied().filter(|&d| present(d)).collect();
    let primary = match primary {
        Some(p) if kept.contains(&p) => Some(p),
        _ => kept.last().copied(),
    };
    (kept, primary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_round_trip_and_default_shown() {
        let dir = std::env::temp_dir().join(format!("hms_physics_outlines_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Absent file = SHOWN (the shipped default: hidden blocks must be visible to be edited).
        assert!(load_from_dir(&dir));
        save_to_dir(&dir, false);
        assert!(!load_from_dir(&dir));
        assert_eq!(std::fs::read_to_string(dir.join(SETTING_FILE)).unwrap(), "0");
        save_to_dir(&dir, true);
        assert!(load_from_dir(&dir));
        // Garbage = the default (shown), not hidden.
        std::fs::write(dir.join(SETTING_FILE), "maybe").unwrap();
        assert!(load_from_dir(&dir));
        std::fs::write(dir.join(SETTING_FILE), " off\n").unwrap();
        assert!(!load_from_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_line_reads_state() {
        assert!(status_line(true, 252).starts_with("hidden-block hulls: shown (252"));
        assert!(status_line(false, 0).starts_with("hidden-block hulls: hidden (0"));
    }

    #[test]
    fn box_overlaps_uses_bounds_not_centre() {
        let b = ([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
        // A hidden block whose CENTRE is outside the box but whose hull reaches in: taken.
        assert!(box_overlaps(b.0, b.1, [6.0, 2.0, 2.0], Some(([3.5, 1.0, 1.0], [8.5, 3.0, 3.0]))));
        // A 0.1 wu nub at that centre does not reach the box: a centre-only test would differ.
        assert!(!box_overlaps(b.0, b.1, [6.0, 2.0, 2.0], Some(([5.95, 1.95, 2.0], [6.05, 2.05, 2.0]))));
        // Fully inside / touching the face / fully outside.
        assert!(box_overlaps(b.0, b.1, [2.0, 2.0, 2.0], Some(([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]))));
        assert!(box_overlaps(b.0, b.1, [5.0, 2.0, 2.0], Some(([4.0, 1.0, 1.0], [6.0, 3.0, 3.0]))));
        assert!(!box_overlaps(b.0, b.1, [9.0, 9.0, 9.0], Some(([8.0, 8.0, 8.0], [10.0, 10.0, 10.0]))));
        // No bounds yet: centre test.
        assert!(box_overlaps(b.0, b.1, [1.0, 1.0, 1.0], None));
        assert!(!box_overlaps(b.0, b.1, [5.0, 1.0, 1.0], None));
    }

    #[test]
    fn prune_selection_drops_missing_and_rederives_primary() {
        let present = |d: u32| d != 0xD000_0002;
        // Primary survives: kept as is.
        let (s, p) = prune_selection(&[0xD000_0001, 0xD000_0002, 0xD000_0003], Some(0xD000_0001), present);
        assert_eq!(s, vec![0xD000_0001, 0xD000_0003]);
        assert_eq!(p, Some(0xD000_0001));
        // Primary was the removed object: falls back to the last survivor.
        let (s, p) = prune_selection(&[0xD000_0001, 0xD000_0002], Some(0xD000_0002), present);
        assert_eq!(s, vec![0xD000_0001]);
        assert_eq!(p, Some(0xD000_0001));
        // Nothing survives: fully cleared (the highlight buffer must be wiped by the caller).
        let (s, p) = prune_selection(&[0xD000_0002], Some(0xD000_0002), present);
        assert!(s.is_empty());
        assert_eq!(p, None);
        // Empty selection stays empty; a stale primary with no set is dropped too.
        let (s, p) = prune_selection(&[], Some(0xD000_0009), |_| true);
        assert!(s.is_empty());
        assert_eq!(p, None);
    }
}
