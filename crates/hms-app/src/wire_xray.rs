//! #wire-visible  The "Selection wireframe through objects" switch (View menu).
//!
//! The selection wireframe is depth-tested: an object behind a wall, a floor, or another Forge
//! piece shows only the edges that happen to be unoccluded, which makes a buried block hard to
//! find and hard to nudge. With this switch ON the wireframe lane swaps to the depth-ALWAYS
//! pipeline (the one the gizmo and the soft-ceiling overlay already use), so the whole outline of
//! the selected object reads through anything in front of it. OFF (the default, and the game-like
//! look) it behaves exactly as before.
//!
//! The switch only moves the SELECTION lane (`SceneRenderer::set_highlight_xray`). The
//! hidden-block hulls, the trigger volumes, the boundary zones and the Construct guides ride the
//! overlay lane and are not affected; the soft-ceiling / hard-floor overlays already draw through
//! geometry and have their own switches.
//!
//! Persisted in the settings dir like `map_spawns.txt` / `physics_outlines.txt`; script verb
//! `wirexray on|off|get`; `HMS_WIRE_XRAY=1` for headless verification.

const SETTING_FILE: &str = "wire_xray.txt";

/// The persisted GUI toggle (settings dir). DEFAULT OFF: absent / unreadable / garbage = false.
pub fn load_setting() -> bool {
    match crate::project::settings_dir() {
        Some(dir) => load_from_dir(&dir),
        None => false,
    }
}

/// Read the toggle from an explicit settings directory (default false when absent).
pub fn load_from_dir(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join(SETTING_FILE))
        .ok()
        .and_then(|s| crate::map_spawns::parse_flag(&s))
        .unwrap_or(false)
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

/// Headless / batch: `HMS_WIRE_XRAY=1|on` draws the selection wireframe through objects; unset or
/// anything else = off (the same default a fresh install has).
pub fn env_wire_xray() -> bool {
    std::env::var("HMS_WIRE_XRAY").ok().and_then(|v| crate::map_spawns::parse_flag(&v)).unwrap_or(false)
}

/// One-line status for the script `wirexray get` and the View-menu tooltip. `lines` is the number
/// of wireframe vertices the highlight lane currently holds (0 = nothing selected).
pub fn status_line(on: bool, lines: usize) -> String {
    let n = lines / 2;
    if on {
        format!("selection wireframe: draws THROUGH objects ({n} edge(s) selected)")
    } else {
        format!("selection wireframe: depth-tested, hidden behind objects ({n} edge(s) selected)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_round_trip_and_default_off() {
        let dir = std::env::temp_dir().join(format!("hms_wire_xray_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Absent file = OFF (the shipped default: the wireframe stays depth-tested).
        assert!(!load_from_dir(&dir));
        save_to_dir(&dir, true);
        assert!(load_from_dir(&dir));
        assert_eq!(std::fs::read_to_string(dir.join(SETTING_FILE)).unwrap(), "1");
        save_to_dir(&dir, false);
        assert!(!load_from_dir(&dir));
        // Garbage = the default (off), not on.
        std::fs::write(dir.join(SETTING_FILE), "maybe").unwrap();
        assert!(!load_from_dir(&dir));
        std::fs::write(dir.join(SETTING_FILE), " on\n").unwrap();
        assert!(load_from_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_line_reads_state() {
        assert!(status_line(true, 24).contains("THROUGH objects (12 edge"));
        assert!(status_line(false, 0).contains("depth-tested"));
    }
}
