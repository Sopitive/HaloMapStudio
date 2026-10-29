//! #cannon-preview  The editor's man-cannon trajectory preview: select a cannon, see where a
//! Spartan dropped into it would land.
//!
//! ## One law, not two
//!
//! The path is NOT an analytic parabola. It is test mode's own tick, run forward on a copy of the
//! player: [`Launchers::begin_tick`] + [`Launchers::tick_player`] (the phantom law and the latch,
//! the same call `PlayMode::run_phantoms` makes), then [`PlayerState::step`] against a
//! [`PlayWorld`] gathered from the editor's own scene and object table (the same call
//! `PlayMode::tick` makes), one engine tick (1/60 s) at a time. So the barrel, the latch, the
//! biped's air control, gravity and every wall, floor and Forge piece the player would hit are the
//! ones the engine-faithful test mode uses; there is no second collision path and no second copy
//! of the arithmetic to drift.
//!
//! ## What is covered
//!
//! Any placed object whose `phmo` carries a pushing phantom type is a "launcher"
//! ([`crate::play_launch::decode_tag`]): the three Forge man cannons, the vehicle cannon, the
//! grav lift and the Covenant dropship. The preview drops a Spartan into the launcher's first
//! LATCHING volume that acts on a player and reports one of four verdicts ([`Verdict`]):
//! it lands somewhere (the man cannons), it comes back to the launcher (vehicle cannon, grav
//! lift: thrown straight up / lifted, then dropped back in, so there is no landing spot to
//! show), it never leaves the volume, or it never touches down. Vehicles and props are not
//! previewed: they are stepped by the drive law / the rigid-body world, which this module does
//! not run. Forge-scaled cannons are not scaled (test mode does not scale them either).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use glam::Vec3;
use hms_ipc::ObjectInfo;
use hms_native::{CacheHandle, NativeDll};

use crate::objscene::ObjectScene;
use crate::play_ctl::{MoveInput, PlayerState};
use crate::play_launch::{Body, Launchers, ObjType, WorldVolume, TICK_SECONDS};
use crate::play_mode::{PlayWorld, Species};
use crate::play_phys::PlayerPhysics;

/// How long a body is followed before the preview gives up on a landing.
pub const MAX_SECONDS: f32 = 20.0;
/// A phantom has to have acted this many ticks in a row before its letting go counts as the
/// launch (the same bar the test-mode gates use), so a graze of the volume is not a launch.
const MIN_PUSHED_TICKS: usize = 10;
/// Frames of nothing in the catch volume before the preview says the body never launched.
const NEVER_ENTERED_TICKS: usize = 300;
/// How far from every volume's footprint a touchdown still counts as "back on the launcher".
const RETURN_MARGIN: f32 = 1.0;
/// The biped starts this far above the catch volume's top, like test mode's own dropped start.
const DROP_HEIGHT: f32 = 0.4;
/// At most this many selected launchers are simulated at once.
pub const MAX_LAUNCHERS: usize = 8;
/// Never recompute more often than this while the selection is being dragged.
const MIN_INTERVAL: Duration = Duration::from_millis(120);
/// Metres per world unit (the game's 1 wu = 3.048 m), for the label.
pub const METRES_PER_WU: f32 = 3.048;

const SETTING_FILE: &str = "man_cannon_trajectory.txt";
const LANDS_RGB: [f32; 3] = [1.0, 0.55, 0.08];
const OTHER_RGB: [f32; 3] = [0.75, 0.75, 0.25];

/// The persisted View toggle. DEFAULT OFF: an absent, unreadable or garbage file means off.
pub fn load_setting() -> bool {
    crate::project::settings_dir().map_or(false, |d| load_from_dir(&d))
}

pub fn load_from_dir(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join(SETTING_FILE)).ok().and_then(|s| crate::map_spawns::parse_flag(&s)).unwrap_or(false)
}

pub fn save_setting(on: bool) {
    if let Some(d) = crate::project::settings_dir() {
        save_to_dir(&d, on);
    }
}

pub fn save_to_dir(dir: &std::path::Path, on: bool) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(SETTING_FILE), if on { "1" } else { "0" });
}

/// What became of a body dropped into a launcher.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Launched and touched down somewhere else: [`Trajectory::landing`] is set.
    Lands,
    /// Launched, and came back down on the launcher itself (it is thrown again): the vehicle
    /// cannon and the grav lift.
    Returns,
    /// Never launched: the catch volume was never entered, or the body never left the volumes.
    NeverLaunched,
    /// Launched but did not touch down within [`MAX_SECONDS`], or fell out of the map.
    NoLanding(String),
}

/// One launcher's predicted flight of a Spartan.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub datum: u32,
    pub verdict: Verdict,
    /// The launcher's own origin.
    pub origin: Vec3,
    /// Feet position, one per engine tick, from the tick the catch volume first acted to the end.
    pub path: Vec<Vec3>,
    /// Index into `path` of the first tick after the phantom let go.
    pub exit: Option<usize>,
    pub landing: Option<Vec3>,
    pub apex: Vec3,
    /// Seconds between the phantom letting go and touching down.
    pub airtime: f32,
    /// Seconds between the catch volume first acting and touching down.
    pub total_time: f32,
    /// The landing point lies outside the scene's bounds.
    pub outside_bounds: bool,
}

impl Trajectory {
    /// Horizontal distance from the launcher to the landing, wu.
    pub fn horizontal(&self) -> Option<f32> {
        self.landing.map(|l| Vec3::new(l.x - self.origin.x, l.y - self.origin.y, 0.0).length())
    }

    /// Apex height above the launcher's origin, wu.
    pub fn apex_height(&self) -> f32 {
        self.apex.z - self.origin.z
    }

    /// The label drawn in the viewport.
    pub fn label(&self) -> Vec<String> {
        match (&self.verdict, self.horizontal()) {
            (Verdict::Lands, Some(h)) => {
                let mut v = vec![
                    format!("lands {h:.1} wu away (~{:.0} m)", h * METRES_PER_WU),
                    format!("apex +{:.1} wu · airtime {:.1} s", self.apex_height(), self.airtime),
                ];
                if self.outside_bounds {
                    v.push("outside the map bounds".into());
                }
                v
            }
            (Verdict::Returns, _) => vec![
                "comes back down on the launcher".into(),
                format!("(thrown {:+.1} wu, then re-caught: no landing spot)", self.apex_height()),
            ],
            (Verdict::NeverLaunched, _) => vec!["never leaves the launcher".into(), "(blocked, or nothing here pushes a player)".into()],
            (Verdict::NoLanding(why), _) => vec![format!("no landing: {why}")],
            (Verdict::Lands, None) => vec!["landing unknown".into()],
        }
    }

    /// The viewport geometry: the arc, an entry diamond, an apex tick and the landing marker.
    pub fn segments(&self) -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
        let rgb = if self.verdict == Verdict::Lands { LANDS_RGB } else { OTHER_RGB };
        let mut out: Vec<([f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(self.path.len() + 40);
        for w in self.path.windows(2) {
            out.push((w[0].to_array(), w[1].to_array(), rgb));
        }
        let cross = |out: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>, c: Vec3, r: f32| {
            out.push(((c - Vec3::X * r).to_array(), (c + Vec3::X * r).to_array(), rgb));
            out.push(((c - Vec3::Y * r).to_array(), (c + Vec3::Y * r).to_array(), rgb));
        };
        if let Some(&p) = self.path.first() {
            cross(&mut out, p, 0.25);
        }
        cross(&mut out, self.apex, 0.2);
        if let Some(l) = self.landing {
            // a ring, a cross and a pole: readable from any camera angle, on any floor
            let n = 24;
            for k in 0..n {
                let (a0, a1) = (k as f32 / n as f32 * std::f32::consts::TAU, (k + 1) as f32 / n as f32 * std::f32::consts::TAU);
                let ring = |a: f32, r: f32| l + Vec3::new(a.cos() * r, a.sin() * r, 0.03);
                out.push((ring(a0, 0.6).to_array(), ring(a1, 0.6).to_array(), rgb));
                out.push((ring(a0, 0.3).to_array(), ring(a1, 0.3).to_array(), rgb));
            }
            cross(&mut out, l + Vec3::Z * 0.03, 0.9);
            out.push((l.to_array(), (l + Vec3::Z * 2.5).to_array(), rgb));
        }
        out
    }

    /// Where the label hangs: above the landing, or above the apex when there is none.
    pub fn label_anchor(&self) -> Vec3 {
        self.landing.map_or(self.apex, |l| l + Vec3::Z * 2.6)
    }
}

/// The first LATCHING volume of the launcher that acts on a player-controlled biped: the one a
/// body has to be inside for the cannon to take hold of it.
pub fn catch_volume(vols: &[WorldVolume]) -> Option<&WorldVolume> {
    let probe = Body { kind: ObjType::Biped, player: true, dead: false, grounded: false, aabb: (Vec3::ZERO, Vec3::ONE), centre: Vec3::splat(0.5), velocity: Vec3::ZERO };
    vols.iter().find(|v| !v.ty.keeps_only() && v.ty.affects(&probe))
}

/// Where the preview's biped starts: over the catch volume's centre, above its top.
pub fn start_feet(catch: &WorldVolume) -> Vec3 {
    let (lo, hi) = catch.aabb;
    Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, hi.z + DROP_HEIGHT)
}

/// Run one Spartan, dropped into `cannon`, forward from the engine's tick. `objects` must be the
/// whole object table (every other launcher acts, every Forge piece is solid); `None` when the
/// object launches nothing that acts on a player.
#[allow(clippy::too_many_arguments)]
pub fn predict(
    launch: &mut Launchers,
    phys: &PlayerPhysics,
    native: &NativeDll,
    cache: CacheHandle,
    scene: &dyn ObjectScene,
    objects: &[ObjectInfo],
    cannon: &ObjectInfo,
) -> Option<Trajectory> {
    launch.refresh(native, cache, objects);
    let vols = launch.volumes_of(native, cache, cannon);
    let start = start_feet(catch_volume(&vols)?);
    let (foot_lo, foot_hi) = vols.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), v| (a.min(v.aabb.0), b.max(v.aabb.1)));
    let origin = Vec3::from(cannon.pos);
    let g = crate::reach_physics::WORLD_GRAVITY;
    let bounds = scene.scene_bounds();

    let mut world = PlayWorld::default();
    let mut st = PlayerState::standing_at(start, 0.0);
    let idle = MoveInput::default();
    let (mut path, mut pushed, mut exit, mut landing) = (Vec::<Vec3>::new(), 0usize, None, None);
    let mut entered_at: Option<usize> = None;
    let mut apex = start;
    let mut verdict = Verdict::NeverLaunched;
    let on_launcher = |p: Vec3| p.x > foot_lo.x - RETURN_MARGIN && p.x < foot_hi.x + RETURN_MARGIN && p.y > foot_lo.y - RETURN_MARGIN && p.y < foot_hi.y + RETURN_MARGIN;
    let mut recaught = false;
    let ticks = (MAX_SECONDS / TICK_SECONDS) as usize;
    let mut last = 0;
    for t in 0..ticks {
        last = t;
        launch.begin_tick();
        launch.tick_player(&mut st, phys, g);
        if st.in_phantom && entered_at.is_none() {
            entered_at = Some(t);
            path.push(st.pos);
            apex = st.pos;
        }
        if !world.covers(st.pos) {
            world.regather(st.pos, scene, objects, native, cache);
        }
        st.step(&idle, phys, &world, TICK_SECONDS);
        if entered_at.is_some() {
            path.push(st.pos);
            if st.pos.z > apex.z {
                apex = st.pos;
            }
        }
        if st.in_phantom {
            pushed += 1;
        } else if exit.is_none() && pushed >= MIN_PUSHED_TICKS {
            exit = Some(path.len() - 1);
        }
        // thrown, came back down into the launcher's own volumes and was caught again
        if exit.is_some() && st.in_phantom && on_launcher(st.pos) {
            recaught = true;
            break;
        }
        if exit.is_some() && st.grounded {
            landing = Some(st.pos);
            break;
        }
        if let Some((mn, _)) = bounds {
            if st.pos.z < mn[2] - 50.0 && exit.is_some() {
                verdict = Verdict::NoLanding("falls out of the map".into());
                break;
            }
        }
        if entered_at.is_none() && t >= NEVER_ENTERED_TICKS {
            break;
        }
    }
    if recaught {
        verdict = Verdict::Returns;
    } else if let Some(l) = landing {
        verdict = if on_launcher(l) { Verdict::Returns } else { Verdict::Lands };
    } else if exit.is_some() && verdict == Verdict::NeverLaunched {
        verdict = Verdict::NoLanding(format!("still airborne after {MAX_SECONDS:.0} s"));
    }
    let airtime = exit.map_or(0.0, |e| (path.len() - 1 - e) as f32 * TICK_SECONDS);
    let total_time = entered_at.map_or(0.0, |e| (last + 1 - e) as f32 * TICK_SECONDS);
    let outside_bounds = match (landing, bounds) {
        (Some(l), Some((mn, mx))) => (0..3).any(|k| l[k] < mn[k] || l[k] > mx[k]),
        _ => false,
    };
    Some(Trajectory { datum: cannon.datum, verdict, origin, path, exit, landing, apex, airtime, total_time, outside_bounds })
}

/// The preview's state in the app: the setting, the decode caches, the last results.
#[derive(Default)]
pub struct State {
    pub on: bool,
    /// Rate-limit recomputes (the live app, while a drag changes the scene every frame). The
    /// headless host leaves it off so a screenshot always shows the current selection.
    pub throttle: bool,
    launch: Launchers,
    phys: Option<PlayerPhysics>,
    phys_error: Option<String>,
    kinds: HashMap<u32, bool>,
    sig: u64,
    last_run: Option<Instant>,
    pub results: Vec<Trajectory>,
    /// More launchers were selected than [`MAX_LAUNCHERS`].
    pub truncated: bool,
}

/// FNV-1a over the bit patterns of a pose.
fn mix(h: &mut u64, bits: u32) {
    *h = (*h ^ bits as u64).wrapping_mul(0x0100_0000_01b3);
}

/// A signature of everything a prediction depends on among the placed objects: any piece that
/// moves, turns, appears or disappears changes it.
pub fn objects_signature(objects: &[ObjectInfo]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix(&mut h, objects.len() as u32);
    for o in objects {
        mix(&mut h, o.datum);
        for v in o.pos.iter().chain(&o.fwd).chain(&o.up) {
            mix(&mut h, v.to_bits());
        }
    }
    h
}

impl State {
    /// The live app's state: the persisted setting, recomputes rate-limited.
    pub fn for_app() -> State {
        State { on: load_setting(), throttle: true, ..Default::default() }
    }

    /// Is `tag` a launcher (cached).
    pub fn is_launcher(&mut self, native: &NativeDll, cache: CacheHandle, tag: u32) -> bool {
        let launch = &mut self.launch;
        *self.kinds.entry(tag).or_insert_with(|| launch.is_launcher(native, cache, tag))
    }

    /// Recompute for `cannons` (already filtered to launchers) when `sig` changed and the last
    /// run is old enough. Returns true when the results changed. `retry` is set when a change
    /// was held back by the rate limit, so the caller schedules another look.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        sig: u64,
        native: &NativeDll,
        cache: CacheHandle,
        scene: &dyn ObjectScene,
        objects: &[ObjectInfo],
        cannons: &[ObjectInfo],
        retry: &mut bool,
    ) -> bool {
        if sig == self.sig {
            return false;
        }
        if self.throttle && self.last_run.is_some_and(|t| t.elapsed() < MIN_INTERVAL) {
            *retry = true;
            return false;
        }
        self.sig = sig;
        self.last_run = Some(Instant::now());
        if self.phys.is_none() && self.phys_error.is_none() {
            let bipeds = crate::reach_actor::ActorScene::list_bipeds(native, cache);
            match crate::play_mode::player_physics_for(native, cache, &bipeds, Species::Spartan) {
                Ok((_, p)) => self.phys = Some(p),
                Err(e) => self.phys_error = Some(e),
            }
        }
        self.truncated = cannons.len() > MAX_LAUNCHERS;
        self.results.clear();
        if let Some(phys) = self.phys.as_ref() {
            for c in cannons.iter().take(MAX_LAUNCHERS) {
                if let Some(t) = predict(&mut self.launch, phys, native, cache, scene, objects, c) {
                    self.results.push(t);
                }
            }
        }
        true
    }

    /// The one entry the app and the headless host both call each frame: pick the launchers out
    /// of `selected` (datums) in `objects`, and recompute when anything they depend on changed.
    /// Clears the results when nothing launcher-like is selected. True = results changed.
    pub fn sync(&mut self, sc: &crate::scene::SceneController, objects: &[ObjectInfo], selected: &[u32], retry: &mut bool) -> bool {
        let Some(cache) = sc.cache_handle() else { return self.clear() };
        let native = sc.native_dll();
        let cannons: Vec<ObjectInfo> = selected
            .iter()
            .filter_map(|d| objects.iter().find(|o| o.datum == *d))
            .filter(|o| self.is_launcher(native, cache, o.primary_tag))
            .copied()
            .collect();
        if cannons.is_empty() {
            return self.clear();
        }
        let mut sig = objects_signature(objects);
        for c in &cannons {
            mix(&mut sig, c.datum);
        }
        self.update(sig, native, cache, sc as &dyn ObjectScene, objects, &cannons, retry)
    }

    pub fn clear(&mut self) -> bool {
        let had = !self.results.is_empty();
        self.results.clear();
        self.sig = 0;
        had
    }

    /// Why nothing can be drawn, when the player's physics could not be read.
    pub fn error(&self) -> Option<&str> {
        self.phys_error.as_deref()
    }

    /// Every segment of every result, for the overlay lane.
    pub fn segments(&self) -> Vec<([f32; 3], [f32; 3], [f32; 3])> {
        self.results.iter().flat_map(|t| t.segments()).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// the app side
// ---------------------------------------------------------------------------------------------

use eframe::egui;

impl crate::App {
    /// The preview works on a Reach map only: Halo 3 / Halo 4 / Halo 2 Anniversary editors run on
    /// other scenes and other tag formats whose phantom volumes HMS does not decode.
    pub(crate) fn cannon_preview_supported(&self) -> bool {
        !self.h3_active && !self.h4_active && self.scene_ctl.as_ref().is_some_and(|s| s.cache_handle().is_some())
    }

    /// The object table as the editor shows it right now: `last_objects` with every editable
    /// object at its live pose (a gizmo drag moves `mvar_objects` before the scene rebuilds).
    fn cannon_object_table(&self) -> Vec<ObjectInfo> {
        let live: HashMap<u32, &ObjectInfo> = self.mvar_objects.iter().chain(self.local_objects.iter()).map(|o| (o.datum, o)).collect();
        self.last_objects
            .iter()
            .map(|o| {
                let mut o = *o;
                if let Some(l) = live.get(&o.datum) {
                    (o.pos, o.fwd, o.up) = (l.pos, l.fwd, l.up);
                }
                o
            })
            .collect()
    }

    /// Once per frame: keep the preview in step with the selection, the cannon's pose and every
    /// other piece. True = the overlay lanes need rebuilding.
    pub(crate) fn cannon_preview_poll(&mut self, ctx: &egui::Context) -> bool {
        if !self.cannon_preview.on || !self.cannon_preview_supported() {
            return self.cannon_preview.clear();
        }
        let mut want = self.selected_set.clone();
        if let Some(d) = self.selected_datum.filter(|d| !want.contains(d)) {
            want.push(d);
        }
        let objects = self.cannon_object_table();
        let Some(sc) = self.scene_ctl.as_ref() else { return false };
        let mut retry = false;
        let changed = self.cannon_preview.sync(sc, &objects, &want, &mut retry);
        if retry {
            ctx.request_repaint_after(MIN_INTERVAL);
        }
        changed
    }

    /// Push the arcs into the overlay lane (called from `rebuild_overlays`).
    pub(crate) fn push_man_cannon_preview(&self, lane: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>) {
        if self.cannon_preview.on {
            lane.extend(self.cannon_preview.segments());
        }
    }

    /// The text labels, hung on each landing point.
    pub(crate) fn paint_man_cannon_labels(&self, ui: &egui::Ui, rect: egui::Rect) {
        if !self.cannon_preview.on {
            return;
        }
        let painter = ui.painter_at(rect);
        for t in &self.cannon_preview.results {
            let Some((sx, sy)) = self.world_to_screen(t.label_anchor(), rect) else { continue };
            let col = if t.verdict == Verdict::Lands { egui::Color32::from_rgb(255, 170, 60) } else { egui::Color32::from_rgb(220, 220, 120) };
            let lines = t.label();
            let font = egui::FontId::proportional(12.5);
            let line_h = 15.0;
            let width = lines.iter().map(|l| painter.layout_no_wrap(l.clone(), font.clone(), col).size().x).fold(0.0f32, f32::max);
            let top = sy - line_h * lines.len() as f32 - 6.0;
            let bg = egui::Rect::from_min_size(egui::pos2(sx - width * 0.5 - 5.0, top - 2.0), egui::vec2(width + 10.0, line_h * lines.len() as f32 + 4.0));
            painter.rect_filled(bg, 4.0, egui::Color32::from_black_alpha(170));
            for (i, l) in lines.iter().enumerate() {
                painter.text(egui::pos2(sx, top + line_h * i as f32), egui::Align2::CENTER_TOP, l, font.clone(), col);
            }
        }
    }

    /// One line for the status tooltip and the Object panel: what is covered and what is shown.
    pub(crate) fn cannon_preview_status(&self) -> String {
        if !self.cannon_preview_supported() {
            return "Man cannon trajectory: Reach maps only (Halo 3 / Halo 4 / Halo 2 Anniversary launchers are not decoded)".into();
        }
        if let Some(e) = self.cannon_preview.error() {
            return format!("Man cannon trajectory unavailable: {e}");
        }
        format!(
            "Man cannon trajectory: {} -- {} launcher(s) selected{}",
            if self.cannon_preview.on { "on" } else { "off" },
            self.cannon_preview.results.len(),
            if self.cannon_preview.truncated { format!(" (first {MAX_LAUNCHERS} drawn)") } else { String::new() }
        )
    }

    /// The View-menu checkbox (persisted). Reach only; elsewhere a greyed line says why.
    pub(crate) fn cannon_preview_menu(&mut self, ui: &mut egui::Ui) {
        if !self.cannon_preview_supported() {
            ui.add_enabled(false, egui::Checkbox::new(&mut false, "Man cannon trajectory")).on_disabled_hover_text(self.cannon_preview_status());
            return;
        }
        let status = self.cannon_preview_status();
        if ui
            .checkbox(&mut self.cannon_preview.on, "Man cannon trajectory")
            .on_hover_text(format!("Select a man cannon (or grav lift / vehicle cannon) and draw where a Spartan dropped into it would fly and land: the arc, a marker on the landing spot, and its distance, apex and airtime. It runs test mode's own physics, so Forge pieces in the way count. Works on Reach maps. Off by default, remembered.\n{status}"))
            .changed()
        {
            save_setting(self.cannon_preview.on);
            self.overlays_dirty = true;
        }
    }

    /// The Object-panel row for a selected launcher: the same toggle, right where the cannon's
    /// properties are, with the verdict beside it. Nothing for any other object.
    pub(crate) fn cannon_preview_rows(&mut self, ui: &mut egui::Ui, datum: u32) {
        if !self.cannon_preview_supported() {
            return;
        }
        let Some(tag) = self.last_objects.iter().find(|o| o.datum == datum).map(|o| o.primary_tag) else { return };
        let Some(sc) = self.scene_ctl.as_ref() else { return };
        let (Some(cache), native) = (sc.cache_handle(), sc.native_dll()) else { return };
        if !self.cannon_preview.is_launcher(native, cache, tag) {
            return;
        }
        ui.separator();
        ui.strong("launch trajectory");
        if ui.checkbox(&mut self.cannon_preview.on, "Show where a player lands").on_hover_text("Draws the arc a Spartan dropped into this cannon flies, with its landing spot, distance, apex and airtime. Same switch as View > Man cannon trajectory; remembered.").changed() {
            save_setting(self.cannon_preview.on);
            self.overlays_dirty = true;
        }
        if self.cannon_preview.on {
            match self.cannon_preview.results.iter().find(|t| t.datum == datum) {
                Some(t) => {
                    for l in t.label() {
                        ui.small(l);
                    }
                    ui.small("Spartan on foot, test-mode physics, Forge scale not applied. Vehicles and props are not previewed.");
                }
                None => {
                    ui.small(self.cannon_preview.error().unwrap_or("computing..."));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::play_ledge_gates::MapScene;
    use crate::play_launch::tests::{placed, tag_ending, with_site};
    use crate::play_mode::ride_harness::Ride;

    const NORMAL: &str = "ff_man_cannon_forge\\ff_man_cannon_forge";
    const HEAVY: &str = "ff_man_cannon_forge_heavy\\ff_man_cannon_forge_heavy";
    const LIGHT: &str = "ff_man_cannon_forge_light\\ff_man_cannon_forge_light";
    const VEHICLE: &str = "ff_veh_man_cannon\\ff_veh_man_cannon";
    const LIFT: &str = "ff_grav_lift\\ff_grav_lift";
    /// How far down the barrel the wall stands: past the apex, on the way down, where the body
    /// is still about 4 wu up (so the wall is stacked three high).
    const WALL_AT: f32 = 11.0;
    const WALL: &str = "ff_wall_2x4\\ff_wall_2x4";

    fn seg(a: [f32; 3], b: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
        (a, b, [0.0; 3])
    }

    fn fake(verdict: Verdict, landing: Option<Vec3>) -> Trajectory {
        Trajectory {
            datum: 7,
            verdict,
            origin: Vec3::ZERO,
            path: vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(-1.0, 0.0, 2.0), Vec3::new(-2.0, 0.0, 2.5), Vec3::new(-3.0, 0.0, 0.0)],
            exit: Some(1),
            landing,
            apex: Vec3::new(-2.0, 0.0, 2.5),
            airtime: 1.5,
            total_time: 2.0,
            outside_bounds: false,
        }
    }

    // ---- no map needed ---------------------------------------------------------------------

    #[test]
    fn the_setting_is_off_until_turned_on_and_remembered() {
        let dir = std::env::temp_dir().join(format!("hms_cannon_setting_{}", std::process::id()));
        assert!(!load_from_dir(&dir), "a missing file is OFF");
        save_to_dir(&dir, true);
        assert!(load_from_dir(&dir), "saved on, read back on");
        save_to_dir(&dir, false);
        assert!(!load_from_dir(&dir));
        let _ = std::fs::write(dir.join(SETTING_FILE), "garbage");
        assert!(!load_from_dir(&dir), "garbage is OFF");
    }

    #[test]
    fn the_overlay_geometry_has_an_arc_a_landing_marker_and_a_pole() {
        let t = fake(Verdict::Lands, Some(Vec3::new(-3.0, 0.0, 0.0)));
        let s = t.segments();
        // 3 arc segments + entry cross (2) + apex cross (2) + 24 x 2 ring + landing cross (2) + pole (1)
        assert_eq!(s.len(), 3 + 2 + 2 + 48 + 2 + 1);
        assert_eq!(s[0].0, [0.0, 0.0, 1.0], "the arc starts where the body is caught");
        assert!(s.iter().any(|x| x.1 == [-3.0, 0.0, 2.5] || x.0 == [-3.0, 0.0, 2.5]), "the pole rises from the landing point");
        assert!(s.iter().all(|x| x.2 == LANDS_RGB), "a landing is drawn in the landing colour");
        let r = fake(Verdict::Returns, Some(Vec3::new(-0.2, 0.0, 0.5))).segments();
        assert!(r.iter().all(|x| x.2 == OTHER_RGB), "a launcher that re-catches its body is drawn differently");
        let none = fake(Verdict::NeverLaunched, None);
        assert_eq!(none.segments().len(), 3 + 2 + 2, "no landing marker without a landing");
        let _ = seg([0.0; 3], [0.0; 3]);
    }

    #[test]
    fn the_label_reports_distance_apex_and_airtime() {
        let t = fake(Verdict::Lands, Some(Vec3::new(-3.0, 4.0, 0.0)));
        let l = t.label();
        assert!(l[0].starts_with("lands 5.0 wu away"), "{l:?}");
        assert!(l[1].contains("apex +2.5 wu") && l[1].contains("airtime 1.5 s"), "{l:?}");
        let mut o = t.clone();
        o.outside_bounds = true;
        assert!(o.label().iter().any(|x| x.contains("outside the map bounds")));
    }

    #[test]
    fn the_object_signature_follows_any_piece_that_moves() {
        let mut v = vec![crate::play_launch::tests::placed(1, 10, Vec3::ZERO, 0.0), crate::play_launch::tests::placed(2, 11, Vec3::X, 0.0)];
        let a = objects_signature(&v);
        assert_eq!(a, objects_signature(&v), "stable");
        v[1].pos[0] += 0.001;
        assert_ne!(a, objects_signature(&v), "another piece moving changes it");
        v[1].pos[0] -= 0.001;
        v[0].fwd = [0.0, 1.0, 0.0];
        assert_ne!(a, objects_signature(&v), "a turn changes it");
        v.pop();
        assert_ne!(a, objects_signature(&v), "a piece removed changes it");
    }

    // ---- through the production tick (needs the map) ------------------------------------------

    /// Predicted by the preview (through `State::update`, the entry the app uses).
    fn preview(native: &std::sync::Arc<NativeDll>, cache: CacheHandle, sc: &crate::scene::SceneController, objects: &[ObjectInfo], cannon: &ObjectInfo) -> Option<Trajectory> {
        let scene = MapScene::new(sc, objects.to_vec());
        let mut st = State::default();
        let mut retry = false;
        assert!(st.update(1, native, cache, &scene, objects, std::slice::from_ref(cannon), &mut retry));
        assert_eq!(st.error(), None);
        st.results.pop()
    }

    /// The same biped, started where the preview starts it, run through the real
    /// `PlayMode::tick` (the Ride harness). Returns where it first touched down after the
    /// phantom let go, and the tick that took.
    fn simulate(native: &std::sync::Arc<NativeDll>, cache: CacheHandle, sc: &crate::scene::SceneController, objects: &[ObjectInfo], cannon: &ObjectInfo, start: Vec3) -> Option<(Vec3, usize)> {
        let scene = MapScene::new(sc, objects.to_vec());
        let mut r = Ride::with_vehicle(native.clone(), cache, "warthog")?;
        r.objects = objects.to_vec();
        r.pm.st = PlayerState::standing_at(start, 0.0);
        let idle = MoveInput::default();
        let (mut run, mut exited) = (0usize, None);
        for t in 0..(MAX_SECONDS * 60.0) as usize {
            r.tick_in(&idle, &scene);
            if r.pm.st.in_phantom {
                run += 1;
            } else {
                if run >= MIN_PUSHED_TICKS && exited.is_none() {
                    exited = Some(t);
                }
                run = 0;
            }
            if exited.is_some() && r.pm.st.grounded {
                return Some((r.pm.st.pos, t));
            }
        }
        let _ = cannon;
        None
    }

    /// Predicted vs simulated landing for the three Forge man cannons at several facings.
    /// Tolerance 0.02 wu (6 cm): the preview runs the SAME tick on the SAME world, and measured
    /// agreement is bit-identical (error 0.0000 on all 12 rows), so 0.02 is two orders above the
    /// float noise and far below the 0.1+ wu any change of the air law, the latch, the start
    /// state or a collision source produces. Anything beyond it means the paths have drifted.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture --test-threads=1"]
    fn the_preview_lands_where_test_mode_lands() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let mut rows = 0;
            for (name, suffix) in [("normal", NORMAL), ("heavy", HEAVY), ("light", LIGHT)] {
                let tag = tag_ending(native, cache, "bloc", suffix).expect("the cannon");
                for k in 0..4 {
                    let yaw = k as f32 * std::f32::consts::FRAC_PI_2 + 0.3;
                    let cannon = placed(tag, 0x7100, site, yaw);
                    let t = preview(native, cache, sc, &[cannon], &cannon).expect("a trajectory");
                    assert_eq!(t.verdict, Verdict::Lands, "{name} yaw {yaw}: {:?}", t.verdict);
                    let landing = t.landing.expect("a landing");
                    let vols = Launchers::default().volumes_of(native, cache, &cannon);
                    let start = start_feet(catch_volume(&vols).expect("catch volume"));
                    let (sim, tick) = simulate(native, cache, sc, &[cannon], &cannon, start).expect("test mode lands too");
                    let err = landing.distance(sim);
                    eprintln!("CANNON {name:6} yaw {yaw:.2}: predicted {:?} dist {:.2} apex +{:.2} air {:.2}s | simulated {:?} (tick {tick}) | error {err:.4} wu", landing - site, t.horizontal().unwrap(), t.apex_height(), t.airtime, sim - site);
                    assert!(err < 0.02, "{name} yaw {yaw}: predicted {landing:?} vs simulated {sim:?}");
                    assert!(t.horizontal().unwrap() > 5.0, "{name}: landed only {:.1} wu away", t.horizontal().unwrap());
                    rows += 1;
                }
            }
            assert_eq!(rows, 12, "liveness: 3 cannons x 4 facings were compared");
        }) else { return };
    }

    /// Turning the cannon turns the arc: the landing point swings round the cannon by the same
    /// angle, at the same distance.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture --test-threads=1"]
    fn rotating_the_cannon_rotates_the_arc() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let tag = tag_ending(native, cache, "bloc", NORMAL).expect("the cannon");
            let land = |yaw: f32| {
                let c = placed(tag, 0x7100, site, yaw);
                preview(native, cache, sc, &[c], &c).and_then(|t| t.landing).expect("lands")
            };
            let a = land(0.0) - site;
            let b = land(std::f32::consts::FRAC_PI_2) - site;
            let ang = |v: Vec3| v.y.atan2(v.x).to_degrees();
            let mut turned = ang(b) - ang(a);
            if turned < -180.0 { turned += 360.0 }
            eprintln!("yaw 0 -> {a:?}, yaw 90 -> {b:?}, turned {turned:.2} deg");
            assert!((turned - 90.0).abs() < 2.0, "the landing swung {turned} deg for a 90 deg turn");
            assert!((a.truncate().length() - b.truncate().length()).abs() < 0.4, "the distance changed with the facing");
            assert!(a.x < -10.0, "yaw 0 throws toward -X: {a:?}");
        }) else { return };
    }

    /// A Forge piece in the way shortens the arc: the biped hits it (or lands on it) instead of
    /// flying over the open ground behind it.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture --test-threads=1"]
    fn a_wall_in_the_path_shortens_the_arc() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let tag = tag_ending(native, cache, "bloc", NORMAL).expect("the cannon");
            let cannon = placed(tag, 0x7100, site, 0.0);
            let free = preview(native, cache, sc, &[cannon], &cannon).expect("free flight");
            let free_d = free.horizontal().expect("lands");
            // a Forge wall down the barrel, across the line of flight
            let wall_tag = native.tags_by_group(cache, "bloc").into_iter().find(|&t| native.tag_name(cache, t).is_some_and(|n| n.ends_with(WALL))).expect("a wall block in forge_halo");
            // turned so its long side lies across the line of flight (extent along Y > along X)
            let across = |yaw: f32| {
                let w = placed(wall_tag, 0x7101, site + Vec3::new(-WALL_AT, 0.0, 0.0), yaw);
                let (lo, hi) = sc.physics_world_tris(&w).iter().flatten().fold((Vec3::splat(1e9), Vec3::splat(-1e9)), |(a, b), p| (a.min(*p), b.max(*p)));
                (w, (hi.y - lo.y) - (hi.x - lo.x), hi.z - lo.z)
            };
            let (w0, a0, _) = across(0.0);
            let (w1, a1, h) = across(std::f32::consts::FRAC_PI_2);
            let base = if a0 > a1 { w0 } else { w1 };
            let stack: Vec<ObjectInfo> = (0..3).map(|k| { let mut w = base; w.datum += k; w.pos[2] += h * k as f32; w }).collect();
            let wall = base;
            eprintln!("wall extent: across-Y minus along-X {:.2} / {:.2}, height {h:.2}", a0, a1);
            assert!(a0.max(a1) > 1.0 && h > 1.5, "the wall block is a wall");
            eprintln!("wall solid to a biped: {}, wall at {:?}", crate::reach_collision::collides_with_biped(native, cache, wall_tag), wall.pos);
            let mut objects = vec![cannon];
            objects.extend(stack);
            let blocked = preview(native, cache, sc, &objects, &cannon).expect("blocked flight");
            let d = blocked.horizontal().expect("it lands on or before the wall");
            eprintln!("free {free_d:.2} wu, with a wall at {WALL_AT} wu: {d:.2} wu ({:?})", blocked.verdict);
            assert!(free_d > 12.0, "the open flight lands {free_d}");
            assert!(d < free_d - 3.0, "the wall did not shorten the arc: {d} vs {free_d}");
            assert!(d < WALL_AT + 0.5, "the blocked flight should end at the wall, landed {d} wu out");
        }) else { return };
    }

    /// Print-only: which bloc tags look like walls.
    #[test]
    #[ignore = "opens the native library; print-only"]
    fn list_wall_blocks() {
        let Some((_g, native, cache)) = crate::reach_actor::tests::open_reach_for("forge_halo") else { return };
        for t in native.tags_by_group(cache, "bloc") {
            if let Some(n) = native.tag_name(cache, t) {
                if n.contains("wall") || n.contains("block") {
                    eprintln!("BLOC {n}");
                }
            }
        }
    }

    /// Which launchers the preview covers, by verdict, on the shipped tags.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored --nocapture --test-threads=1"]
    fn coverage_by_launcher() {
        let Some(()) = with_site(|native, cache, sc, site| {
            let mut seen = 0;
            for (name, suffix, want) in [("normal", NORMAL, Some(true)), ("heavy", HEAVY, Some(true)), ("light", LIGHT, Some(true)), ("vehicle cannon", VEHICLE, Some(false)), ("grav lift", LIFT, Some(false))] {
                let tag = tag_ending(native, cache, "bloc", suffix).expect("the launcher");
                let c = placed(tag, 0x7100, site, 0.0);
                let t = preview(native, cache, sc, &[c], &c).expect("every shipped launcher yields a trajectory");
                eprintln!("COVERAGE {name:15} {:?} {:?} exit {:?} landing {:?}", t.verdict, t.label(), t.exit, t.landing.map(|l| l - site));
                assert_eq!(t.verdict == Verdict::Lands, want.unwrap(), "{name}: {:?}", t.verdict);
                if want == Some(false) {
                    assert_eq!(t.verdict, Verdict::Returns, "{name} is thrown up and re-caught: {:?}", t.verdict);
                }
                seen += 1;
            }
            assert_eq!(seen, 5, "liveness");
        }) else { return };
    }
}
