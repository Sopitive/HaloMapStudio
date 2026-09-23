//! The "Show map spawn points" switch.
//!
//! A scenario (scnr) ships its own spawn-family placements -- initial spawn points, respawn
//! points, respawn zones (weak/anti), the loadout camera. In the editor they are visual noise on
//! top of the variant's OWN spawns (which the user places, edits and saves), so they are HIDDEN by
//! default. The switch only affects placements that come from the scenario: variant (.mvar) and
//! locally-placed objects are never touched, and hidden scenario objects are dropped from the
//! RENDERED set only -- the editor's bookkeeping (`scenario_objects`, save, picking of the
//! variant's objects) is unchanged; the scene's pick list is rebuilt from the rendered set so a
//! hidden marker can never be clicked.
//!
//! Classification is by the placement's obje TAG PATH (`objects\multi\spawning\...`), which is
//! stable across every Reach map (measured on forge_halo, cex_prisoner, 20_sword_slayer,
//! 45_launch_station -- see `is_map_spawn_name`).

const SETTING_FILE: &str = "map_spawns.txt";

/// True when an obje tag path names a spawn-family marker: anything under the engine's
/// `objects\multi\spawning\` folder (initial_spawn_point, respawn_point, respawn_zone,
/// respawn_zone_weak, respawn_zone_anti, spawning_camera / the gametype-specific *_spawn variants),
/// plus the bare marker names in case a modded map moved them. Matching is case-insensitive and
/// separator-agnostic; an empty / unknown path is never a spawn.
pub fn is_map_spawn_name(tag_path: &str) -> bool {
    if tag_path.is_empty() {
        return false;
    }
    let p = tag_path.to_ascii_lowercase().replace('/', "\\");
    if p.contains("\\multi\\spawning\\") || p.contains("\\spawning\\") {
        return true;
    }
    let leaf = p.rsplit('\\').next().unwrap_or(p.as_str());
    leaf.starts_with("initial_spawn")
        || leaf.starts_with("respawn_point")
        || leaf.contains("respawn_zone")
        || leaf == "mp_spawn_point"
        || leaf == "spawning_camera"
        // Halo 4's loadout camera lives under objects\multi\generic\
        || leaf.starts_with("mp_cinematic")
}

/// The DISTINCT obj tags among `tags` whose tag path is a spawn-family marker. One name lookup per
/// distinct tag (the native lookup is per call; a scenario has hundreds of placements of a few tags).
pub fn spawn_obj_tags(tags: impl IntoIterator<Item = u32>, name_of: impl Fn(u32) -> String) -> std::collections::HashSet<u32> {
    let mut seen = std::collections::HashSet::new();
    let mut out = std::collections::HashSet::new();
    for t in tags {
        if !seen.insert(t) {
            continue;
        }
        if is_map_spawn_name(&name_of(t)) {
            out.insert(t);
        }
    }
    out
}

/// Parse a persisted / scripted flag value: 1|on|true|yes -> true, 0|off|false|no -> false.
pub fn parse_flag(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" | "show" => Some(true),
        "0" | "off" | "false" | "no" | "hide" => Some(false),
        _ => None,
    }
}

/// The persisted GUI toggle (settings dir). DEFAULT HIDDEN: absent / unreadable / garbage = false.
pub fn load_setting() -> bool {
    match crate::project::settings_dir() {
        Some(dir) => load_from_dir(&dir),
        None => false,
    }
}

/// Read the toggle from an explicit settings directory (default false when absent).
pub fn load_from_dir(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join(SETTING_FILE)).ok().and_then(|s| parse_flag(&s)).unwrap_or(false)
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

/// Headless / batch: `HMS_MAP_SPAWNS=1|on` shows the scenario's spawn markers; unset or anything
/// else = hidden (the same default the GUI has, so a headless render matches a fresh install).
pub fn env_show_map_spawns() -> bool {
    std::env::var("HMS_MAP_SPAWNS").ok().and_then(|v| parse_flag(&v)).unwrap_or(false)
}

/// One-line status for the script `mapspawns get` / the View menu tooltip.
pub fn status_line(show: bool, hidden_count: usize) -> String {
    if show {
        format!("map spawns: shown ({hidden_count} scenario spawn marker(s))")
    } else {
        format!("map spawns: hidden ({hidden_count} scenario spawn marker(s) not drawn)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_matches_the_spawning_family_only() {
        for p in [
            "objects\\multi\\spawning\\initial_spawn_point",
            "objects\\multi\\spawning\\respawn_point",
            "objects\\multi\\spawning\\respawn_zone",
            "objects\\multi\\spawning\\respawn_zone_weak",
            "objects\\multi\\spawning\\respawn_zone_anti",
            "objects\\multi\\spawning\\spawning_camera",
            "objects/multi/spawning/initial_spawn_point",
            "OBJECTS\\MULTI\\SPAWNING\\RESPAWN_POINT",
            "objects\\multi\\spawning\\ctf_initial_spawn_point",
            "respawn_zone_anti",
            "initial_spawn_point",
            "objects\\multi\\spawning\\weak_anti_respawn_zone",
            "objects\\multi\\spawning\\respawn_point_invisible",
            "objects\\multi\\generic\\mp_cinematic_camera", // Halo 4 loadout camera
            "objects\\multi\\generic\\mp_cinematic_fallback_camera",
        ] {
            assert!(is_map_spawn_name(p), "{p} should be a map spawn");
        }
        for p in [
            "",
            "objects\\levels\\multi\\forge_halo\\rock_spire",
            "objects\\multi\\models\\mp_flag_base\\mp_flag_base",
            "objects\\multi\\koth\\obj_hill_marker",
            "objects\\multi\\spawning_kill\\spawning_kill", // kill/safe zones are boundaries, not spawns
            "objects\\vehicles\\warthog\\warthog",
            "objects\\weapons\\rifle\\assault_rifle\\assault_rifle",
        ] {
            assert!(!is_map_spawn_name(p), "{p} should NOT be a map spawn");
        }
    }

    #[test]
    fn setting_round_trip_and_default_hidden() {
        let dir = std::env::temp_dir().join(format!("hms_map_spawns_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Absent file = hidden (the shipped default).
        assert!(!load_from_dir(&dir));
        save_to_dir(&dir, true);
        assert!(load_from_dir(&dir));
        assert_eq!(std::fs::read_to_string(dir.join(SETTING_FILE)).unwrap(), "1");
        save_to_dir(&dir, false);
        assert!(!load_from_dir(&dir));
        // Garbage = hidden, not shown.
        std::fs::write(dir.join(SETTING_FILE), "maybe").unwrap();
        assert!(!load_from_dir(&dir));
        std::fs::write(dir.join(SETTING_FILE), " on\n").unwrap();
        assert!(load_from_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn flag_parse() {
        assert_eq!(parse_flag("1"), Some(true));
        assert_eq!(parse_flag("OFF"), Some(false));
        assert_eq!(parse_flag("show"), Some(true));
        assert_eq!(parse_flag("hide"), Some(false));
        assert_eq!(parse_flag("x"), None);
    }
}
