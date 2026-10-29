//! #bump-trace  An ALWAYS-ON, PRINT-ONLY trace that makes "which code moves the thing that falls" decidable.
//!
//! `prop_fall.log` only watches bodies the rigid-body world ([`crate::play_world::PropWorld`]) owns and is awake; an object some
//! OTHER path moves (the ride path's parked vehicles, a static placement) is invisible to it. This writes `bump_trace.log` in the
//! same settings folder, at most [`MAX_LINES`] lines a session, and changes nothing:
//!
//! 1. once per object the player touches or that wakes / is displaced / is claimed / becomes a mover: datum, tag, class, the
//!    `seed_world` role and WHY an object is not a candidate, the slot kind, whether the prop world owns it, `owns_vehicle`, its
//!    pose and which module draws it this frame;
//! 2. for the first [`FOLLOWED`] such objects, their pose every 6 ticks for 90 ticks after the first touch, with the nearest
//!    terrain height under them from BOTH worlds (the player's `PlayWorld::regather` soup and the prop world's level + statics);
//! 3. any frame where the DRAWN pose differs from the prop world's pose by more than [`DRAWN_TOLERANCE`].

use crate::play_world::{seed_role, PropSession, PropWorld, SeedRole};
use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};
use std::collections::{HashMap, HashSet};
use std::io::Write;

/// Lines a session may write.
pub const MAX_LINES: usize = 200;
/// Objects whose pose is followed after the first touch.
pub const FOLLOWED: usize = 3;
/// Drawn pose vs prop-world pose, wu.
pub const DRAWN_TOLERANCE: f32 = 0.05;
const FOLLOW_TICKS: u64 = 90;
const FOLLOW_EVERY: u64 = 6;
/// Ticks between two drawn-pose lines about one object.
const DRAWN_QUIET: u64 = 120;

/// Everything the trace reads, borrowed from the play mode for one frame.
pub struct Cx<'a> {
    pub props: &'a PropSession,
    pub objects: &'a [ObjectInfo],
    pub scene: &'a dyn crate::objscene::ObjectScene,
    pub native: &'a NativeDll,
    pub cache: CacheHandle,
    /// The triangles the player collides with (`PlayWorld::soup`).
    pub soup: &'a [[Vec3; 3]],
    /// The ride path's parked vehicles: datum and pose.
    pub parked: &'a [(u32, Vec3)],
    pub riding: Option<u32>,
    pub actors: &'a crate::reach_actor::ActorScene,
    /// The player's feet, capsule radius and height.
    pub player: (Vec3, f32, f32),
    /// Events the prop session logged since the last frame.
    pub events: Vec<(u32, &'static str)>,
}

struct Follow {
    datum: u32,
    start: u64,
    last: Option<Vec3>,
}

#[derive(Default)]
pub struct BumpTrace {
    announced: bool,
    written: usize,
    tick: u64,
    seen: HashSet<u32>,
    follow: Vec<Follow>,
    followed: usize,
    drawn_quiet: HashMap<u32, u64>,
    names: HashMap<u32, (String, String)>,
    /// What the unit gates read instead of the user's log.
    pub captured: Vec<String>,
}

fn log_path() -> Option<std::path::PathBuf> {
    let dir = crate::project::settings_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("bump_trace.log"))
}

fn clock() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Who owns an object, as a sentence. Pure, so the gate can drive it without a map.
pub fn ownership(kind: Option<&'static str>, role: SeedRole, owns_vehicle: bool, parked: bool, riding: bool) -> String {
    let who = match kind {
        None => "NOT IN THE PROP WORLD (never offered to it)".to_string(),
        Some("Inert") => "in the prop world as Inert (NEVER collides with loose bodies)".to_string(),
        Some(k) => format!("in the prop world as {k}"),
    };
    let ride = if riding {
        ", RIDDEN (ride path)"
    } else if parked && owns_vehicle {
        ", parked set AND owned by the prop world"
    } else if parked {
        ", parked on the RIDE PATH (jeep-law suspension, not the prop world)"
    } else {
        ""
    };
    let r = match role {
        SeedRole::Simulated => "role simulated".to_string(),
        SeedRole::Terrain => "role terrain".to_string(),
        SeedRole::Skipped(why) => format!("role skipped: {why}"),
    };
    format!("{who}{ride}; {r}")
}

impl BumpTrace {
    fn emit(&mut self, line: String) {
        if self.written >= MAX_LINES {
            return;
        }
        self.written += 1;
        // the unit and map-backed gates drive this same code; they must not write the user's log
        if cfg!(test) {
            self.captured.push(line);
            return;
        }
        if let Some(p) = log_path() {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                let _ = writeln!(f, "{line}");
            }
        }
    }

    fn names_of(&mut self, cx: &Cx, tag: u32) -> (String, String) {
        self.names.entry(tag).or_insert_with(|| (cx.native.tag_name(cx.cache, tag).unwrap_or_default(), cx.native.tag_class(cx.cache, tag).unwrap_or_default())).clone()
    }

    /// The object's pose as the world sees it now, and its speed if the prop world knows it.
    fn pose_of(cx: &Cx, w: Option<&PropWorld>, o: &ObjectInfo) -> (Vec3, &'static str, f32) {
        if let Some((_, p)) = cx.parked.iter().find(|(d, _)| *d == o.datum) {
            return (*p, "ride path (parked drive state)", 0.0);
        }
        if let Some(w) = w {
            if let (Some((p, _)), Some((v, _))) = (w.body_pose(o.datum), w.body_velocity(o.datum)) {
                if w.kind_of(o.datum) == Some("Live") {
                    return (p, "prop world body", v.length());
                }
            }
        }
        (Vec3::from(o.pos), "placement (static)", 0.0)
    }

    /// Which module draws it this frame.
    fn drawn_by(cx: &Cx, w: Option<&PropWorld>, datum: u32) -> &'static str {
        if cx.riding == Some(datum) {
            return "ride path (ridden actors)";
        }
        if cx.parked.iter().any(|(d, _)| *d == datum) {
            return "ride path (parked actors)";
        }
        let hidden = w.is_some_and(|w| w.hidden_datums().contains(&datum));
        if hidden {
            return if cx.props.has_actor(datum) { "prop actor" } else { "HIDDEN BUT NOTHING DRAWS IT" };
        }
        "static scene (the placement)"
    }

    fn describe(&mut self, cx: &Cx, o: &ObjectInfo, why: &str) {
        let (name, class) = self.names_of(cx, o.primary_tag);
        let w = cx.props.world();
        let role = seed_role(cx.native, cx.cache, o.primary_tag, cx.props.all_solid);
        let kind = w.and_then(|w| w.kind_of(o.datum));
        let owns_vehicle = cx.props.owns_vehicle(o.datum);
        let parked = cx.parked.iter().any(|(d, _)| *d == o.datum);
        let (pose, pose_src, speed) = Self::pose_of(cx, w, o);
        let mode = w.and_then(|w| w.mode_of(o.datum)).map_or("-".to_string(), |m| if m == crate::play_world::PhysicsMode::Phased { "Phased (held as Fixed terrain for loose bodies)".to_string() } else { format!("{m:?}") });
        let asleep = w.and_then(|w| w.is_asleep(o.datum)).map_or("-".to_string(), |a| a.to_string());
        let extra = if cx.props.disabled { "; PropSession.disabled" } else { "" };
        let line = format!(
            "[{}] t{} {why}: datum {:#x} '{name}' class {class}; {}; placement mode {mode}; slot asleep {asleep}; owns_vehicle {owns_vehicle}; pose {:.2},{:.2},{:.2} (from {pose_src}) speed {speed:.2}; drawn by: {}{extra}",
            clock(),
            self.tick,
            o.datum,
            ownership(kind, role, owns_vehicle, parked, cx.riding == Some(o.datum)),
            pose.x,
            pose.y,
            pose.z,
            Self::drawn_by(cx, w, o.datum),
        );
        self.emit(line);
        if self.followed < FOLLOWED {
            self.followed += 1;
            self.follow.push(Follow { datum: o.datum, start: self.tick, last: None });
        }
    }

    /// One call per stepped frame, after the props and the vehicles have moved.
    pub fn observe(&mut self, mut cx: Cx) {
        if !self.announced {
            self.announced = true;
            self.emit(format!("[{}] bump_trace active: logging objects the player touches or that wake / are displaced / claimed, which code owns and draws them, and their pose for 90 ticks (max {MAX_LINES} lines per session)", clock()));
        }
        self.tick += 1;
        if self.written >= MAX_LINES {
            return;
        }
        let w = cx.props.world();
        // 1. the triggers
        let mut hot: Vec<(u32, String)> = Vec::new();
        for (d, what) in std::mem::take(&mut cx.events) {
            hot.push((d, what.to_string()));
        }
        if let Some(w) = w {
            for b in w.diag_bodies() {
                hot.push((b.datum, "awake in the prop world".into()));
            }
            for d in w.hidden_datums() {
                hot.push((d, "displaced from its placement (hidden)".into()));
            }
        }
        // the player touching an object: its box against the capsule's, grown by the push margin
        let (feet, radius, height) = cx.player;
        let (pmn, pmx) = (feet - Vec3::new(radius + 0.3, radius + 0.3, 0.3), feet + Vec3::new(radius + 0.3, radius + 0.3, height + 0.3));
        for o in cx.objects {
            if self.seen.contains(&o.datum) || (Vec3::from(o.pos) - feet).length() > 12.0 {
                continue;
            }
            let Some((mn, mx)) = cx.scene.aabb_of(o.datum) else { continue };
            if (0..3).all(|k| mn[k] <= pmx[k] && mx[k] >= pmn[k]) {
                // a wall or a floor the player stands on is not news: only what the prop world would NOT treat as plain terrain
                if matches!(seed_role(cx.native, cx.cache, o.primary_tag, cx.props.all_solid), SeedRole::Terrain) {
                    continue;
                }
                hot.push((o.datum, "touched by the player".into()));
            }
        }
        for (d, why) in hot {
            if self.seen.contains(&d) || self.written >= MAX_LINES {
                continue;
            }
            let Some(o) = cx.objects.iter().find(|o| o.datum == d) else { continue };
            self.seen.insert(d);
            self.describe(&cx, o, &why);
        }
        // 2. the followed objects
        let mut lines: Vec<String> = Vec::new();
        for f in self.follow.iter_mut() {
            let age = self.tick - f.start;
            if age > FOLLOW_TICKS || age % FOLLOW_EVERY != 0 {
                continue;
            }
            let Some(o) = cx.objects.iter().find(|o| o.datum == f.datum) else { continue };
            let (pose, src, v) = Self::pose_of(&cx, w, o);
            let speed = if v > 0.0 { v } else { f.last.map_or(0.0, |l| (pose - l).length() * 60.0 / FOLLOW_EVERY as f32) };
            f.last = Some(pose);
            let from = pose + Vec3::Z * 0.5;
            let under_player = crate::play_world::ray_down(cx.soup, from, 60.0).map(|(d, _)| from.z - d);
            let mut tris: Vec<[Vec3; 3]> = Vec::new();
            let (mn, mx) = (pose - Vec3::new(0.7, 0.7, 60.0), pose + Vec3::new(0.7, 0.7, 1.0));
            cx.scene.static_tris_in_aabb(mn, mx, &mut tris);
            if let Some(w) = w {
                w.diag_static_tris(mn, mx, &mut tris);
            }
            let under_props = crate::play_world::ray_down(&tris, from, 60.0).map(|(d, _)| from.z - d);
            let fmt = |h: Option<f32>| h.map_or("none".to_string(), |h| format!("{h:.2}"));
            lines.push(format!(
                "[{}] t{} follow +{age} datum {:#x}: z {:.3} speed {speed:.2} (pose from {src}); terrain under it: player world {}, prop world (level+statics) {}{}",
                clock(),
                self.tick,
                f.datum,
                pose.z,
                fmt(under_player),
                fmt(under_props),
                match (under_player, under_props) {
                    (Some(a), Some(b)) if pose.z < a.min(b) - 0.2 => "  <-- BELOW BOTH",
                    (Some(a), _) if pose.z < a - 0.2 => "  <-- below the player's world",
                    (_, Some(b)) if pose.z < b - 0.2 => "  <-- below the prop world's",
                    _ => "",
                }
            ));
        }
        for l in lines {
            self.emit(l);
        }
        // 3. a stale second pose source
        if let Some(w) = w {
            let mut stale: Vec<String> = Vec::new();
            let mut datums: Vec<u32> = w.diag_bodies().iter().map(|b| b.datum).collect();
            datums.extend(w.hidden_datums());
            datums.sort_unstable();
            datums.dedup();
            for d in datums {
                if self.drawn_quiet.get(&d).is_some_and(|q| *q > self.tick) {
                    continue;
                }
                let Some((want, _)) = w.body_pose(d) else { continue };
                let drawn = if let Some((_, p)) = cx.parked.iter().find(|(x, _)| *x == d) {
                    Some(*p)
                } else if let Some(p) = cx.props.actor_origin(d, cx.actors) {
                    Some(p)
                } else if w.hidden_datums().contains(&d) {
                    None
                } else {
                    cx.objects.iter().find(|o| o.datum == d).map(|o| Vec3::from(o.pos))
                };
                let Some(drawn) = drawn else { continue };
                if (drawn - want).length() > DRAWN_TOLERANCE {
                    self.drawn_quiet.insert(d, self.tick + DRAWN_QUIET);
                    stale.push(format!("[{}] t{} DRAWN POSE DIFFERS: datum {d:#x} drawn at {:.2},{:.2},{:.2} by {}, prop world has it at {:.2},{:.2},{:.2} ({:.2} wu apart)", clock(), self.tick, drawn.x, drawn.y, drawn.z, Self::drawn_by(&cx, Some(w), d), want.x, want.y, want.z, (drawn - want).length()));
                }
            }
            for l in stale {
                self.emit(l);
            }
        }
    }
}

#[cfg(test)]
mod gates {
    use super::*;

    /// **The trace says so when the prop world does not own what is moving.** An object that was never offered to the prop world
    /// (`kind` None: a skipped class, a non-solid tag) is reported as NOT IN THE PROP WORLD with the reason; a parked vehicle the
    /// ride path steps is reported as parked on the ride path and not owned; an owned body says which kind it is.
    ///
    /// SABOTAGE: `ownership` ignoring `kind` (always "in the prop world") -> red.
    #[test]
    fn the_trace_marks_a_body_the_prop_world_does_not_own() {
        let s = ownership(None, SeedRole::Skipped("class bipd/dgrp"), false, false, false);
        assert!(s.contains("NOT IN THE PROP WORLD") && s.contains("class bipd/dgrp"), "{s}");
        let s = ownership(None, SeedRole::Simulated, false, true, false);
        assert!(s.contains("NOT IN THE PROP WORLD") && s.contains("RIDE PATH") && s.contains("not the prop world"), "{s}");
        let s = ownership(Some("Live"), SeedRole::Simulated, true, true, false);
        assert!(s.contains("in the prop world as Live") && s.contains("owned by the prop world") && !s.contains("NOT IN"), "{s}");
        let s = ownership(Some("Inert"), SeedRole::Simulated, false, false, true);
        assert!(s.contains("Inert") && s.contains("RIDDEN"), "{s}");
    }

    /// The log is bounded and does not write the user's file under test: 500 emits keep 200 lines, all captured, none on disk.
    #[test]
    fn the_trace_is_rate_limited() {
        let mut t = BumpTrace::default();
        for k in 0..500 {
            t.emit(format!("line {k}"));
        }
        assert_eq!(t.captured.len(), MAX_LINES);
        assert_eq!(MAX_LINES, 200);
    }
}
