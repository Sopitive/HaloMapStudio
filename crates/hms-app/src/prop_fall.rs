//! #prop-fall  An ALWAYS-ON, PRINT-ONLY diagnostic for loose bodies that go where they should not.
//!
//! Reported (and not reproduced by any stand-in): bump a weapon, a ball or a vehicle resting on a
//! Forge wall or floor and it "phases through like it isn't there". This watches the real
//! rigid-body world ([`PropWorld`]) as test mode steps it and, when a loose body
//!
//! * has its centre of mass BEHIND a surface of the static world (the first surface straight above
//!   it is seen from its back: it is inside the floor or the wall), or
//! * moves more than a metre in one frame that its own speed does not explain,
//!
//! writes ONE line to `prop_fall.log` in the settings folder
//! (`%LOCALAPPDATA%\HaloMapStudio` on Windows, `~/.local/share/HaloMapStudio` on Linux): which
//! object, its pose before and after, its velocity, whether it sleeps, how many static triangles
//! support it, the nearest loaded static hulls with their collision groups, the contact counts of
//! the step and which code path moved it. At most [`MAX_LINES`] lines per session, one per body per
//! 5 seconds. It changes nothing.

use crate::play_world::{PropWorld, StepStats};
use crate::physics_settle::StaticWorld;
use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};
use std::collections::HashMap;
use std::io::Write;

/// Lines a session may write.
pub const MAX_LINES: usize = 20;
/// Frames between two lines about the same body (5 s at 60 Hz).
const QUIET_FRAMES: u64 = 300;
/// The inside-a-surface test runs on every this-many frames.
const CHECK_EVERY: u64 = 6;

/// The watcher's state.
#[derive(Default)]
pub struct FallWatch {
    announced: bool,
    written: usize,
    frame: u64,
    last: HashMap<u32, Vec3>,
    quiet_until: HashMap<u32, u64>,
    /// `matg`'s collision filter, read once.
    filter: Option<Option<crate::reach_collision::CollisionFilter>>,
}

fn log_path() -> Option<std::path::PathBuf> {
    let dir = crate::project::settings_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("prop_fall.log"))
}

fn write_line(line: &str) {
    // the unit and map-backed gates drive this same code; they must not write the user's log
    if cfg!(test) {
        return;
    }
    if let Some(p) = log_path() {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn clock() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// The first surface straight above `from` within `reach`: its distance and whether the ray meets it
/// from BEHIND (the point is inside the solid that triangle bounds).
pub fn first_surface_above(tris: &[[Vec3; 3]], from: Vec3, reach: f32) -> Option<(f32, bool)> {
    let mut best: Option<(f32, bool)> = None;
    for t in tris {
        let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
        let n = e1.cross(e2);
        if n.z.abs() < 1e-9 {
            continue;
        }
        let h = Vec3::Z.cross(e2);
        let a = e1.dot(h);
        if a.abs() < 1e-12 {
            continue;
        }
        let f = 1.0 / a;
        let s = from - t[0];
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = f * Vec3::Z.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let d = f * e2.dot(q);
        if d > 1e-4 && d < reach && best.map_or(true, |b| d < b.0) {
            best = Some((d, n.z > 0.0));
        }
    }
    best
}

impl FallWatch {
    /// One call per stepped frame, after the world has moved.
    #[allow(clippy::too_many_arguments)]
    pub fn check(&mut self, world: &PropWorld, level: &dyn StaticWorld, objects: &[ObjectInfo], native: &NativeDll, cache: CacheHandle, dt: f32, pusher: bool, stats: &StepStats) {
        if !self.announced {
            self.announced = true;
            write_line(&format!("[{}] prop_fall diagnostic active: watching loose bodies for falls through surfaces (max {MAX_LINES} lines per session)", clock()));
        }
        self.frame += 1;
        let bodies = world.diag_bodies();
        let scan = self.frame % CHECK_EVERY == 0;
        for b in &bodies {
            let before = self.last.insert(b.datum, b.origin);
            if self.written >= MAX_LINES || self.quiet_until.get(&b.datum).is_some_and(|q| *q > self.frame) {
                continue;
            }
            let mut why = String::new();
            if let Some(p) = before {
                let d = (b.origin - p).length();
                if d > 1.0 && d > 2.0 * b.v.length() * dt + 0.5 {
                    why = format!("moved {d:.2} wu in one frame (its speed explains {:.2})", b.v.length() * dt);
                }
            }
            let (mn, mx) = (b.aabb.0 - Vec3::splat(6.0), b.aabb.1 + Vec3::splat(6.0));
            let mut tris: Vec<[Vec3; 3]> = Vec::new();
            if scan || !why.is_empty() {
                level.gather(mn, mx, &mut tris);
                world.diag_static_tris(mn, mx, &mut tris);
            }
            let level_n = tris.len();
            if why.is_empty() && scan {
                if let Some((h, true)) = first_surface_above(&tris, b.com, 3.0) {
                    why = format!("centre of mass is inside a surface: the first one above it is {h:.2} wu up and seen from behind");
                }
            }
            if why.is_empty() {
                continue;
            }
            self.quiet_until.insert(b.datum, self.frame + QUIET_FRAMES);
            self.written += 1;
            let info = objects.iter().find(|o| o.datum == b.datum);
            let name = info.map(|o| native.tag_name(cache, o.primary_tag).unwrap_or_default()).unwrap_or_default();
            let class = info.and_then(|o| native.tag_class(cache, o.primary_tag)).unwrap_or_default();
            let filter = self.filter.get_or_insert_with(|| crate::reach_collision::matg_filter(native, cache)).clone();
            let near: Vec<String> = world
                .diag_fixed_near(mn, mx)
                .into_iter()
                .take(6)
                .map(|d| {
                    let tag = objects.iter().find(|o| o.datum == d).map(|o| o.primary_tag);
                    let nm = tag.and_then(|t| native.tag_name(cache, t)).unwrap_or_default();
                    let groups: Vec<String> = tag
                        .map(|t| crate::reach_collision::shape_filters(native, cache, t))
                        .unwrap_or_default()
                        .iter()
                        .map(|s| match (&filter, s.group) {
                            (_, 0xFF) => "none(default)".to_string(),
                            (Some(f), g) => format!("{g}:{:#010x}", f.groups.get(g as usize).copied().unwrap_or(0)),
                            (None, g) => g.to_string(),
                        })
                        .collect();
                    format!("{}[groups {}]", nm.rsplit('\\').next().unwrap_or(&nm), groups.join(","))
                })
                .collect();
            let mut near_tris = Vec::new();
            world.diag_static_tris(mn, mx, &mut near_tris);
            write_line(&format!(
                "[{}] frame {} object {:#x} '{}' ({}{}): {why}; pose before {:?} after {:?}; v {:?} w {:?}; asleep false; static tris in support set within 6 wu: {} fixed + {} level (hulls loaded in total {}); nearest fixed hulls: {}; contacts this step: {} static, {} pair; moved by: PropWorld::substep via PropSession::tick{}",
                clock(),
                self.frame,
                b.datum,
                name,
                class,
                if b.vehicle { ", vehicle" } else { "" },
                before.map(|v| v.to_array()),
                b.origin.to_array(),
                b.v.to_array(),
                b.w.to_array(),
                near_tris.len(),
                level_n.saturating_sub(near_tris.len()),
                world.diag_statics(mn, mx).0,
                if near.is_empty() { "none".to_string() } else { near.join("; ") },
                stats.static_contacts,
                stats.pair_contacts,
                if pusher { " (player capsule is a pusher this frame)" } else { "" },
            ));
        }
        // forget bodies that fell asleep: their next move is a new rest pose
        if self.last.len() > 4 * bodies.len() + 64 {
            let live: std::collections::HashSet<u32> = bodies.iter().map(|b| b.datum).collect();
            self.last.retain(|d, _| live.contains(d));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor() -> Vec<[Vec3; 3]> {
        let (a, b, c, d) = (Vec3::new(-5.0, -5.0, 0.0), Vec3::new(5.0, -5.0, 0.0), Vec3::new(5.0, 5.0, 0.0), Vec3::new(-5.0, 5.0, 0.0));
        vec![[a, b, c], [a, c, d]]
    }

    /// A point UNDER an upward-facing floor sees it from behind; one above it sees nothing; one
    /// under a slab (the floor wound the other way) sees its front.
    #[test]
    fn a_point_below_a_floor_is_inside_it_and_one_above_is_not() {
        let f = floor();
        assert_eq!(first_surface_above(&f, Vec3::new(0.0, 0.0, -0.4), 3.0).map(|(d, back)| ((d * 100.0).round(), back)), Some((40.0, true)));
        assert_eq!(first_surface_above(&f, Vec3::new(0.0, 0.0, 0.4), 3.0), None);
        let ceiling: Vec<[Vec3; 3]> = f.iter().map(|t| [t[0], t[2], t[1]]).collect();
        assert_eq!(first_surface_above(&ceiling, Vec3::new(0.0, 0.0, -0.4), 3.0).map(|(_, back)| back), Some(false));
        assert_eq!(first_surface_above(&f, Vec3::new(9.0, 0.0, -0.4), 3.0), None, "off the floor's edge");
    }
}
