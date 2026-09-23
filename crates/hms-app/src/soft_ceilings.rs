//! The "Soft ceilings" switch -- see the map floor.
//!
//! A Reach level's playable volume is bounded by SOFT CEILINGS: invisible triangle soups baked
//! into each BSP's structure_design (sddt) tag. Three kinds (engine `soft_ceiling_type_enum`):
//!   * soft kill    -- the "map floor" under Forge World / the walls of the play space: a player
//!                     who crosses it dies (after the "return to battlefield" grace);
//!   * acceleration -- pushes bipeds / vehicles back (the sky ceiling);
//!   * slip surface -- can't be stood on (steep rock tops).
//! The engine never draws them, so a Forger has no way to know where the floor is until a piece
//! falls through it. This overlay draws every soft ceiling like the trigger volumes: a translucent
//! fill (red = soft kill, orange = acceleration, yellow = slip) plus its edges, so the floor plane
//! reads from above. OFF by default (the overlay hides the map), persisted in the settings dir like
//! `map_spawns.txt`; the script `softceilings on|off|get` and the batch env `HMS_SOFT_CEILINGS=1`
//! drive the same switch. Layout evidence: `native/MccMapStudioDLL/SoftCeilingWalker.cpp`.

use hms_native::SoftCeiling;

const SETTING_FILE: &str = "soft_ceilings.txt";

/// The persisted GUI toggle (settings dir). DEFAULT HIDDEN: absent / unreadable / garbage = false.
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

/// Headless / batch: `HMS_SOFT_CEILINGS=1|on` draws the soft ceilings; unset or anything else =
/// hidden (the same default the GUI has, so a headless render matches a fresh install).
pub fn env_show_soft_ceilings() -> bool {
    std::env::var("HMS_SOFT_CEILINGS").ok().and_then(|v| crate::map_spawns::parse_flag(&v)).unwrap_or(false)
}

/// Line colour per soft-ceiling kind: soft kill = red, acceleration = orange, slip = yellow.
pub fn kind_color(kind: u32) -> [f32; 3] {
    match kind {
        1 => [0.95, 0.18, 0.18],
        0 => [1.0, 0.55, 0.12],
        2 => [0.98, 0.9, 0.2],
        _ => [0.8, 0.4, 0.8],
    }
}

/// Fill alpha per kind: the kill floor is what people look for, so it is the most opaque.
fn kind_alpha(kind: u32) -> f32 {
    match kind {
        1 => 0.32,
        0 => 0.22,
        _ => 0.26,
    }
}

/// The overlay geometry for a set of soft ceilings: translucent fill triangles (zone lane, RGBA,
/// depth-tested -- tints the walls you look at from inside the play space) + line segments for
/// the X-RAY lane (RGB, drawn with the depth test off, so the floor under the water / terrain
/// reads from above like a cage). Edges that belong to ONE triangle only (the open outline of
/// the soup) are drawn in the full kind colour; shared interior edges are dimmed so the
/// tessellation does not bury the outline but the planes still read from any angle.
#[allow(clippy::type_complexity)]
pub fn overlay_geometry(ceilings: &[SoftCeiling]) -> (Vec<([f32; 3], [f32; 4])>, Vec<([f32; 3], [f32; 3], [f32; 3])>) {
    let mut tris = Vec::new();
    let mut segs = Vec::new();
    for c in ceilings {
        let rgb = kind_color(c.kind);
        let a = kind_alpha(c.kind);
        let dim = [rgb[0] * 0.55, rgb[1] * 0.55, rgb[2] * 0.55];
        let mut edges: std::collections::HashMap<([i32; 3], [i32; 3]), (u32, [f32; 3], [f32; 3])> =
            std::collections::HashMap::new();
        for t in &c.tris {
            for v in t {
                tris.push((*v, [rgb[0], rgb[1], rgb[2], a]));
            }
            for (i, j) in [(0usize, 1usize), (1, 2), (2, 0)] {
                let (ka, kb) = (quant(t[i]), quant(t[j]));
                let key = if ka <= kb { (ka, kb) } else { (kb, ka) };
                let e = edges.entry(key).or_insert((0, t[i], t[j]));
                e.0 += 1;
            }
        }
        for (_, (n, p0, p1)) in edges {
            segs.push((p0, p1, if n == 1 { rgb } else { dim }));
        }
    }
    (tris, segs)
}

fn quant(p: [f32; 3]) -> [i32; 3] {
    [(p[0] * 256.0).round() as i32, (p[1] * 256.0).round() as i32, (p[2] * 256.0).round() as i32]
}

/// One-line status for the script `softceilings get` / the View menu tooltip.
pub fn status_line(show: bool, ceilings: &[SoftCeiling]) -> String {
    let mut kinds = [0usize; 3];
    let mut tris = 0usize;
    for c in ceilings {
        if (c.kind as usize) < 3 {
            kinds[c.kind as usize] += 1;
        }
        tris += c.tris.len();
    }
    format!(
        "soft ceilings: {} ({} ceiling(s): {} soft kill, {} acceleration, {} slip surface; {} triangle(s))",
        if show { "shown" } else { "hidden" },
        ceilings.len(),
        kinds[1],
        kinds[0],
        kinds[2],
        tris
    )
}

/// The per-ceiling listing (`softceilings get` / the headless log): name, kind, triangle count,
/// and the Z range so the kill FLOOR is identifiable by its height.
pub fn listing(ceilings: &[SoftCeiling]) -> Vec<String> {
    ceilings
        .iter()
        .map(|c| {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for t in &c.tris {
                for v in t {
                    lo = lo.min(v[2]);
                    hi = hi.max(v[2]);
                }
            }
            let z = if c.tris.is_empty() { String::from("no triangles") } else { format!("z {lo:.1}..{hi:.1}") };
            let flags = if c.flags == 0 { String::new() } else { format!(" flags=0x{:x}", c.flags) };
            let name = if c.name.is_empty() { "(unnamed)" } else { c.name.as_str() };
            format!("  {:<28} {:<13} {:>5} tris  {z}{flags}", name, c.kind_name(), c.tris.len())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(kind: u32) -> SoftCeiling {
        SoftCeiling {
            name: "floor".into(),
            kind,
            flags: 0,
            tris: vec![
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
        }
    }

    #[test]
    fn setting_round_trip_and_default_hidden() {
        let dir = std::env::temp_dir().join(format!("hms_soft_ceilings_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!load_from_dir(&dir));
        save_to_dir(&dir, true);
        assert!(load_from_dir(&dir));
        save_to_dir(&dir, false);
        assert!(!load_from_dir(&dir));
        std::fs::write(dir.join(SETTING_FILE), "maybe").unwrap();
        assert!(!load_from_dir(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn geometry_outline_vs_interior_edges() {
        let (tris, segs) = overlay_geometry(&[quad(1)]);
        assert_eq!(tris.len(), 6);
        assert!(tris.iter().all(|(_, c)| c[0] > 0.9 && c[3] > 0.3)); // red kill fill
        // A quad has 4 outline edges + 1 shared diagonal.
        assert_eq!(segs.len(), 5);
        let bright = segs.iter().filter(|(_, _, c)| c[0] > 0.9).count();
        let dim = segs.iter().filter(|(_, _, c)| c[0] < 0.6).count();
        assert_eq!((bright, dim), (4, 1));
    }

    #[test]
    fn kinds_have_distinct_colours_and_status_counts() {
        assert_ne!(kind_color(0), kind_color(1));
        assert_ne!(kind_color(1), kind_color(2));
        let cs = [quad(1), quad(0), quad(2), quad(1)];
        let s = status_line(true, &cs);
        assert!(s.contains("shown") && s.contains("4 ceiling(s)") && s.contains("2 soft kill") && s.contains("1 acceleration") && s.contains("1 slip surface") && s.contains("8 triangle(s)"), "{s}");
        assert!(status_line(false, &[]).starts_with("soft ceilings: hidden (0 ceiling(s)"));
        let l = listing(&cs);
        assert_eq!(l.len(), 4);
        assert!(l[0].contains("soft kill") && l[0].contains("2 tris") && l[0].contains("z 0.0..0.0"), "{}", l[0]);
    }
}
