//! #quant-error — the GUI/app side of the save-precision prediction: the viewport GHOST, the
//! Object-panel numbers, the snap-to-representable action, the `precision` script verb and the
//! persisted View toggle.
//!
//! The math lives in [`crate::quant_error`]; this module only answers three questions about the
//! app's current state:
//!
//! 1. **Which quantiser applies?** `App::pose_quant()` — the open variant's game plus the exact
//!    `world-bounds` box the next save will quantise against (the EDITED box when the user widened
//!    it, because that is what the encoder writes with).
//! 2. **What does the user see?** `App::quant_error_rows` draws the per-axis / per-angle deltas
//!    under the rotation fields, and `App::push_quant_ghost` pushes the predicted pose into the
//!    overlay lanes as a wireframe ghost.
//! 3. **How do they get rid of it?** `App::snap_datums_to_representable` moves the objects onto the
//!    nearest pose that survives the round trip exactly (one undo step, never automatic).

use eframe::egui;
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::quant_error::{rank_worst, ErrStats, PoseError, PoseQuant, QuantGame, WorstRow};

/// The ghost's colour: a distinct violet nothing else in the overlay lanes uses (collision is
/// cyan, physics orange, blockers amber, construct green).
pub const GHOST_RGB: [f32; 3] = [0.95, 0.35, 0.95];

/// Line budget for ONE ghost. A tenth of [`crate::objscene::WIRE_MAX_LINES`]: the ghost is a
/// reference outline drawn ON TOP of the object's own wireframe, so it does not need every edge,
/// and a Mantis-sized mesh (84 000 edges) must not double the overlay upload every time the piece
/// is nudged. Over budget it is thinned EVENLY, never truncated.
pub const GHOST_MAX_LINES: usize = crate::objscene::WIRE_MAX_LINES / 10;

/// Settings file for the View toggle + its exaggeration factor.
const SETTING_FILE: &str = "quant_ghost.txt";

/// The persisted View toggle and exaggeration factor. DEFAULT OFF at ×1 — an absent, unreadable or
/// garbage file means "off", exactly like a fresh install.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GhostSetting {
    pub on: bool,
    /// How much the predicted drift is amplified when drawn. 1 = true scale.
    pub exaggerate: f32,
}

impl Default for GhostSetting {
    fn default() -> Self {
        GhostSetting { on: false, exaggerate: 1.0 }
    }
}

/// The largest exaggeration offered. ×500 turns a 0.002 wu drift into a full world unit, which is
/// about as far as a ghost can be pushed before it stops reading as "the same object".
pub const EXAGGERATE_MAX: f32 = 500.0;

pub fn load_setting() -> GhostSetting {
    match crate::project::settings_dir() {
        Some(dir) => load_from_dir(&dir),
        None => GhostSetting::default(),
    }
}

/// Read the toggle from an explicit settings directory. Format: `<flag> [factor]` on one line, so
/// the file stays as readable by hand as the other one-flag settings files.
pub fn load_from_dir(dir: &std::path::Path) -> GhostSetting {
    let Ok(s) = std::fs::read_to_string(dir.join(SETTING_FILE)) else { return GhostSetting::default() };
    let mut it = s.split_whitespace();
    let on = it.next().and_then(crate::map_spawns::parse_flag).unwrap_or(false);
    let exaggerate = it.next().and_then(|t| t.parse::<f32>().ok()).filter(|v| v.is_finite() && *v >= 1.0 && *v <= EXAGGERATE_MAX).unwrap_or(1.0);
    GhostSetting { on, exaggerate }
}

pub fn save_setting(s: GhostSetting) {
    if let Some(dir) = crate::project::settings_dir() {
        save_to_dir(&dir, s);
    }
}

pub fn save_to_dir(dir: &std::path::Path, s: GhostSetting) {
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(SETTING_FILE), format!("{} {}", if s.on { 1 } else { 0 }, s.exaggerate));
}

/// Headless / batch seeding: `HMS_QUANT_GHOST=1` (optionally `HMS_QUANT_GHOST=1x40` for ×40).
pub fn env_setting() -> Option<GhostSetting> {
    let v = std::env::var("HMS_QUANT_GHOST").ok()?;
    let (flag, fac) = match v.split_once('x') {
        Some((a, b)) => (a.to_string(), b.parse::<f32>().ok()),
        None => (v, None),
    };
    let on = crate::map_spawns::parse_flag(&flag)?;
    Some(GhostSetting { on, exaggerate: fac.filter(|f| f.is_finite() && *f >= 1.0 && *f <= EXAGGERATE_MAX).unwrap_or(1.0) })
}

// ---------------------------------------------------------------------------------------------
// pose math for the ghost
// ---------------------------------------------------------------------------------------------

/// An orthonormal basis from a forward/up pair, the same Gram-Schmidt choice `euler.rs` and
/// `quant_error` make (forward primary).
fn ortho(fwd: [f32; 3], up: [f32; 3]) -> Mat3 {
    let x = {
        let v = Vec3::from(fwd).normalize_or_zero();
        if v.length_squared() < 0.5 { Vec3::X } else { v }
    };
    let u = Vec3::from(up);
    let mut z = u - x * u.dot(x);
    if z.length_squared() < 1e-12 {
        let r = if x.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        z = r - x * r.dot(x);
    }
    let z = z.normalize_or_zero();
    Mat3::from_cols(x, z.cross(x), z)
}

/// The ghost pose to DRAW: the predicted pose when `k == 1`, otherwise the live→predicted delta
/// scaled by `k` (position linearly, rotation by scaling the relative rotation's angle). Scaling
/// the angle rather than slerping keeps the ghost a rigid body however large `k` gets.
pub fn exaggerated_pose(
    live_pos: [f32; 3], live_fwd: [f32; 3], live_up: [f32; 3],
    saved_pos: [f32; 3], saved_fwd: [f32; 3], saved_up: [f32; 3],
    k: f32,
) -> ([f32; 3], Quat) {
    let k = if k.is_finite() { k.clamp(1.0, EXAGGERATE_MAX) } else { 1.0 };
    let lp = Vec3::from(live_pos);
    let pos = lp + (Vec3::from(saved_pos) - lp) * k;
    let a = Quat::from_mat3(&ortho(live_fwd, live_up)).normalize();
    let b = Quat::from_mat3(&ortho(saved_fwd, saved_up)).normalize();
    let mut rel = (a.conjugate() * b).normalize();
    if rel.w < 0.0 { rel = -rel; } // shortest arc, so the scaled angle is the small one
    let (axis, ang) = rel.to_axis_angle();
    let rel_k = if ang.abs() < 1e-9 { Quat::IDENTITY } else { Quat::from_axis_angle(axis, ang * k) };
    (pos.to_array(), (a * rel_k).normalize())
}

/// The one transform a ghost needs: take the object's CURRENT world geometry (its wireframe, its
/// oriented box) to where the saved pose puts it, with the drift scaled by `k`. Shared by the
/// interactive overlay and the batch host's, so the two draw the same ghost.
pub fn ghost_transform(
    live_pos: [f32; 3], live_fwd: [f32; 3], live_up: [f32; 3],
    saved_pos: [f32; 3], saved_fwd: [f32; 3], saved_up: [f32; 3],
    k: f32,
) -> Mat4 {
    let (gpos, grot) = exaggerated_pose(live_pos, live_fwd, live_up, saved_pos, saved_fwd, saved_up, k);
    ghost_relocate(live_pos, live_fwd, live_up, gpos, grot)
}

/// The rigid transform that carries the object's CURRENT world geometry onto the ghost pose.
/// Derived from orthonormal bases, so an object with a stretched (non-unit) forward/up is moved
/// and turned without its stretch being scaled away.
fn ghost_relocate(live_pos: [f32; 3], live_fwd: [f32; 3], live_up: [f32; 3], ghost_pos: [f32; 3], ghost_rot: Quat) -> Mat4 {
    let a = Quat::from_mat3(&ortho(live_fwd, live_up)).normalize();
    let r = ghost_rot * a.conjugate();
    Mat4::from_translation(Vec3::from(ghost_pos)) * Mat4::from_quat(r) * Mat4::from_translation(-Vec3::from(live_pos))
}

impl crate::App {
    // -----------------------------------------------------------------------------------------
    // the active quantiser
    // -----------------------------------------------------------------------------------------

    /// The quantiser the NEXT SAVE will use, or None when no variant is open (every save path
    /// refuses without one, so there is no error to predict). The box is the one the encoder
    /// writes with: `global_edits.bounds` when the user widened / shrank it, else the file's.
    pub(crate) fn pose_quant(&self) -> Option<PoseQuant> {
        if self.h3_active {
            let b = self.h3_scene.as_ref()?.bounds()?;
            return Some(PoseQuant::new(QuantGame::Halo3, b));
        }
        if self.h4_active {
            // h4_bounds() already prefers the edited box over the file's.
            return Some(PoseQuant::new(QuantGame::Halo4, self.h4_bounds()?));
        }
        let g = self.variant_globals.as_ref()?;
        let b = if self.global_edits.bounds != g.bounds { self.global_edits.bounds } else { g.bounds };
        Some(PoseQuant::new(QuantGame::Reach, b))
    }

    /// A human name for the open variant's format, so the panel can say "Halo 2 Anniversary"
    /// rather than lumping it under Halo 4 (the two share one codec, not one name).
    pub(crate) fn pose_quant_game_name(&self) -> &'static str {
        if self.h3_active { return "Halo 3"; }
        if self.h4_active {
            let h2a = self.h4_scene.as_ref()
                .and_then(|s| s.variant())
                .map_or(false, |(_, v)| v.chunk_version == crate::h4::mvar::H2A_MVAR_VERSION);
            return if h2a { "Halo 2 Anniversary" } else { "Halo 4" };
        }
        "Halo: Reach"
    }

    /// The live editor pose of a variant object (position + forward + up), or None when the datum
    /// is not a saveable placed object.
    fn live_pose(&self, datum: u32) -> Option<([f32; 3], [f32; 3], [f32; 3])> {
        self.mvar_objects.iter().chain(self.local_objects.iter())
            .find(|o| o.datum == datum)
            .map(|o| (o.pos, o.fwd, o.up))
    }

    /// Predict what the next save costs this object. None = no variant open, or not a placed
    /// object (scenario geometry and imported models are never written to a `.mvar`).
    pub(crate) fn predict_pose_error(&self, datum: u32) -> Option<PoseError> {
        let q = self.pose_quant()?;
        let (pos, fwd, up) = self.live_pose(datum)?;
        Some(q.predict(pos, fwd, up))
    }

    /// Every placed object's predicted error, with a display name — the input to the worst-offender
    /// list and the variant summary.
    pub(crate) fn predict_all_pose_errors(&self) -> Vec<WorstRow> {
        let Some(q) = self.pose_quant() else { return Vec::new() };
        self.mvar_objects.iter().chain(self.local_objects.iter())
            .map(|o| {
                let name = self.mvar_meta.get(&o.datum).map(|m| m.name.clone()).unwrap_or_default();
                WorstRow { datum: o.datum, name, err: q.predict(o.pos, o.fwd, o.up) }
            })
            .collect()
    }

    // -----------------------------------------------------------------------------------------
    // the viewport ghost
    // -----------------------------------------------------------------------------------------

    /// A cheap signature of everything the ghost's geometry depends on, so `rebuild_overlays` is
    /// only re-run when the ghost would actually change (the overlay lanes are shared and the
    /// rebuild also walks collision hulls).
    fn quant_ghost_signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.quant_ghost.on.hash(&mut h);
        self.quant_ghost.exaggerate.to_bits().hash(&mut h);
        if let Some(q) = self.pose_quant() {
            for v in q.bounds { v.to_bits().hash(&mut h); }
            q.axis_bits.hash(&mut h);
        }
        for d in self.quant_ghost_targets() {
            d.hash(&mut h);
            if let Some((p, f, u)) = self.live_pose(d) {
                for v in p.iter().chain(f.iter()).chain(u.iter()) { v.to_bits().hash(&mut h); }
            }
        }
        h.finish()
    }

    /// Poll for a ghost change once per frame; true = the overlay lanes need rebuilding. Keeps the
    /// ghost live under a gizmo drag without a per-frame overlay rebuild.
    pub(crate) fn quant_ghost_poll(&mut self) -> bool {
        let sig = self.quant_ghost_signature();
        if sig == self.quant_ghost_sig { return false; }
        self.quant_ghost_sig = sig;
        true
    }

    /// The objects the ghost is drawn for: the selection (the whole multi-selection, so a snapped
    /// group can be checked at once).
    fn quant_ghost_targets(&self) -> Vec<u32> {
        if !self.quant_ghost.on { return Vec::new(); }
        let mut v: Vec<u32> = self.selected_set.clone();
        if let Some(d) = self.selected_datum {
            if !v.contains(&d) { v.push(d); }
        }
        v.retain(|d| self.mvar_meta.contains_key(d));
        v
    }

    /// Push the selected objects' predicted pose into the overlay lanes as a wireframe ghost.
    /// Called from `rebuild_overlays`; `xray` is the depth-ALWAYS line lane, which is what makes a
    /// true-scale ghost (a few thousandths of a unit from the real object) visible at all instead
    /// of z-fighting inside its own surface.
    pub(crate) fn push_quant_ghost(&self, xray: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>) {
        let Some(q) = self.pose_quant() else { return };
        let k = self.quant_ghost.exaggerate;
        for d in self.quant_ghost_targets() {
            let Some((pos, fwd, up)) = self.live_pose(d) else { continue };
            let e = q.predict(pos, fwd, up);
            // Nothing is lost: no ghost (a ghost exactly on the object is noise, and its absence
            // is itself the answer -- this pose survives the save).
            if e.is_exact() { continue; }
            let xf = ghost_transform(pos, fwd, up, e.saved_pos, e.saved_fwd, e.saved_up, k);
            // The object's own mesh edges, re-posed. Falls back to its oriented box when the mesh
            // is not decoded yet (a budgeted rebuild still working, or a model HMS cannot read).
            match self.objscene().and_then(|s| s.selection_wireframe(d)) {
                Some(mut w) => {
                    // The ghost is a REFERENCE outline, not the primary selection wireframe, so it
                    // gets a tenth of that lane's budget -- thinned evenly (`fit_wire_budget`), so a
                    // dense object still shows a ghost everywhere rather than losing its tail meshes.
                    crate::objscene::fit_wire_budget(&mut w, GHOST_MAX_LINES);
                    for seg in w.chunks_exact(2) {
                        let a = xf.transform_point3(Vec3::from(seg[0]));
                        let b = xf.transform_point3(Vec3::from(seg[1]));
                        xray.push((a.to_array(), b.to_array(), GHOST_RGB));
                    }
                }
                None => {
                    // The mesh is not decoded (a budgeted rebuild still running, or a model HMS
                    // cannot read): the object's oriented box is still an honest ghost.
                    let Some(obb) = self.objscene().and_then(|s| s.object_obb(d)) else { continue };
                    for (a, b) in obb.edge_segments() {
                        xray.push((xf.transform_point3(a).to_array(), xf.transform_point3(b).to_array(), GHOST_RGB));
                    }
                }
            }
        }
    }

    /// The status-bar notice. An EXAGGERATED ghost must never be on screen without saying so, so
    /// this returns `Some` whenever the ghost is on, and shouts when the factor is not ×1.
    pub(crate) fn quant_ghost_banner(&self) -> Option<String> {
        if !self.quant_ghost.on { return None; }
        let n = self.quant_ghost_targets().len();
        if self.quant_ghost.exaggerate > 1.0 {
            Some(format!("save-precision ghost: EXAGGERATED ×{:.0} (not the real drift) — {n} object(s)", self.quant_ghost.exaggerate))
        } else {
            Some(format!("save-precision ghost: true scale — {n} object(s)"))
        }
    }

    /// One line for `precision ghost get` and the View-menu tooltip.
    pub(crate) fn quant_ghost_status(&self) -> String {
        let q = self.pose_quant();
        let box_txt = match q {
            Some(q) => format!(
                "{}: position grid {:.6} / {:.6} / {:.6} wu ({} / {} / {} bits), worst rotation {:.4}°",
                self.pose_quant_game_name(),
                q.pos_step()[0], q.pos_step()[1], q.pos_step()[2],
                q.axis_bits[0], q.axis_bits[1], q.axis_bits[2], q.rot_worst_deg()
            ),
            None => "no variant open (nothing to predict)".to_string(),
        };
        format!(
            "save-precision ghost: {}{} — {}",
            if self.quant_ghost.on { "on" } else { "off" },
            if self.quant_ghost.on && self.quant_ghost.exaggerate > 1.0 { format!(", EXAGGERATED x{:.0}", self.quant_ghost.exaggerate) } else { String::new() },
            box_txt
        )
    }

    // -----------------------------------------------------------------------------------------
    // snap to representable
    // -----------------------------------------------------------------------------------------

    /// Move these objects onto the nearest pose that survives the save round trip EXACTLY, so the
    /// predicted error becomes zero. One undo step; a no-op when every object is already
    /// representable. Never automatic — only a button press or the `precision snap` verb.
    pub(crate) fn snap_datums_to_representable(&mut self, datums: &[u32]) -> Result<String, String> {
        let q = self.pose_quant().ok_or("no variant is open, so there is no save quantiser to snap to")?;
        let mut work: Vec<(u32, [f32; 3], [f32; 3], [f32; 3])> = Vec::new();
        for &d in datums {
            let Some((pos, fwd, up)) = self.live_pose(d) else { continue };
            let e = q.predict(pos, fwd, up);
            if e.is_exact() { continue; }
            work.push((d, e.saved_pos, e.saved_fwd, e.saved_up));
        }
        if work.is_empty() {
            return Ok(format!("snap: {} object(s) already survive the save exactly — nothing to do", datums.len()));
        }
        self.push_edit_undo();
        self.prop_snapshotted_for = None;
        let n = work.len();
        let (mut moved, mut turned) = (0.0f32, 0.0f32);
        for (d, p, f, u) in work {
            let Some((p0, f0, u0)) = self.live_pose(d) else { continue };
            moved = moved.max((Vec3::from(p) - Vec3::from(p0)).length());
            turned = turned.max(crate::quant_error::basis_angle_deg(f0, u0, f, u));
            self.set_movable_pose(d, Vec3::from(p), Vec3::from(f), Vec3::from(u));
            // Keep the cached pick boxes truthful so the very next command (or a second snap)
            // sees the new pose rather than the old one -- the same upkeep `snapto` does.
            let q_rel = Quat::from_mat3(&ortho(f, u)).normalize() * Quat::from_mat3(&ortho(f0, u0)).normalize().conjugate();
            if let Some(sc) = self.objscene_mut() {
                sc.rotate_pick(d, q_rel, Vec3::from(p0));
                sc.translate_pick(d, Vec3::from(p) - Vec3::from(p0));
            }
        }
        if let Some(s) = self.objscene_mut() { s.invalidate(); }
        self.highlight_dirty = true;
        self.overlays_dirty = true;
        Ok(format!("snap: {n} object(s) moved onto the nearest representable pose (up to {moved:.6} wu / {turned:.5}°); predicted save error is now zero"))
    }

    /// The selection (primary + multi-select), for the panel button and `precision snap`.
    pub(crate) fn quant_selection(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.selected_set.clone();
        if let Some(d) = self.selected_datum {
            if !v.contains(&d) { v.push(d); }
        }
        v.retain(|d| self.mvar_meta.contains_key(d));
        v
    }

    // -----------------------------------------------------------------------------------------
    // the Object panel section
    // -----------------------------------------------------------------------------------------

    /// "Save precision" rows for the selected object: what the file will actually hold, the delta
    /// per axis, the rotation delta split into roll/pitch/yaw, a warning when the pose sits in the
    /// worst part of a quantisation cell, and the snap button. Updates live as the object moves.
    pub(crate) fn quant_error_rows(&mut self, ui: &mut egui::Ui, datum: u32) {
        let Some(q) = self.pose_quant() else { return };
        let Some(e) = self.predict_pose_error(datum) else { return };
        let step = q.pos_step();
        let sel = self.quant_selection();
        // What the rotation error MOVES: a fraction of a degree is abstract, but "the far corner
        // of this piece lands 4 cm out" is not. The lever arm is the object's own oriented box.
        let lever = self.objscene().and_then(|s| s.object_obb(datum))
            // the distance from the object's centre to its farthest corner
            .map(|o| (o.hx + o.hy + o.hz).length())
            .filter(|r| r.is_finite() && *r > 1e-3);
        egui::CollapsingHeader::new("save precision (what the file will hold)")
            .default_open(e.quantises_badly())
            .show(ui, |ui| {
                ui.small(format!(
                    "{} quantises positions onto a {:.6} × {:.6} × {:.6} wu grid over the variant's world bounds ({} / {} / {} bits per axis), and rotations to no better than {:.4}°.",
                    self.pose_quant_game_name(), step[0], step[1], step[2], q.axis_bits[0], q.axis_bits[1], q.axis_bits[2], q.rot_worst_deg()
                ));
                egui::Grid::new("quant-err").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                    ui.label("position error");
                    ui.vertical(|ui| {
                        ui.monospace(format!("X {:+.6}   Y {:+.6}   Z {:+.6}  wu", e.pos_delta[0], e.pos_delta[1], e.pos_delta[2]));
                        ui.small(format!("total {:.6} wu ({:.0}% of the worst this grid can cost)", e.pos_total, e.pos_frac * 100.0));
                    });
                    ui.end_row();
                    ui.label("rotation error");
                    ui.vertical(|ui| {
                        ui.monospace(format!("{:.5}°", e.rot_deg));
                        ui.small(format!(
                            "roll {:+.5}°   pitch {:+.5}°   yaw {:+.5}°",
                            e.euler_delta_deg[0], e.euler_delta_deg[1], e.euler_delta_deg[2]
                        ));
                        // The angle in the units people care about: how far it slides the piece's
                        // own far corner. A tenth of a degree on a long wall is centimetres.
                        if let Some(r) = lever {
                            ui.small(format!("= {:.5} wu at this piece's far corner ({r:.2} wu out)", r * e.rot_deg.to_radians()));
                        }
                    });
                    ui.end_row();
                    ui.label("stored position");
                    ui.monospace(format!("{:.5}  {:.5}  {:.5}", e.saved_pos[0], e.saved_pos[1], e.saved_pos[2]));
                    ui.end_row();
                });
                if e.is_exact() {
                    ui.colored_label(egui::Color32::from_rgb(120, 200, 120), "This pose survives the save exactly — nothing is lost.");
                } else if e.quantises_badly() {
                    ui.colored_label(
                        egui::Color32::from_rgb(235, 180, 90),
                        format!(
                            "This pose is paying almost the MOST this format can cost it — it sits between two representable values. A nudge of under {:.6} wu, or snapping it, makes the save free.",
                            step.iter().cloned().fold(0.0f32, f32::max) * 0.5
                        ),
                    );
                }
                ui.horizontal(|ui| {
                    let label = if sel.len() > 1 { format!("Snap {} selected to representable", sel.len()) } else { "Snap to representable".to_string() };
                    if ui.button(label)
                        .on_hover_text("Move the object(s) onto the nearest pose the file can store exactly, so the save costs nothing. One undo step; never applied automatically.")
                        .clicked()
                    {
                        match self.snap_datums_to_representable(&sel) {
                            Ok(s) | Err(s) => self.status = s,
                        }
                    }
                    let mut g = self.quant_ghost;
                    if ui.checkbox(&mut g.on, "ghost").on_hover_text("View ▸ Overlays ▸ Save-precision ghost: draw the predicted pose as a violet wireframe.").changed() {
                        self.quant_ghost = g;
                        save_setting(g);
                        self.overlays_dirty = true;
                    }
                });
                if self.quant_ghost.on && self.quant_ghost.exaggerate > 1.0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 120, 230),
                        format!("The violet ghost is EXAGGERATED ×{:.0} — it is not where the object really lands.", self.quant_ghost.exaggerate),
                    );
                }
                if sel.len() > 1 {
                    let mut st = ErrStats::default();
                    for d in &sel {
                        if let Some(x) = self.predict_pose_error(*d) { st.add(&x); }
                    }
                    ui.small(format!(
                        "selection: {} object(s), {} exactly representable, {} quantising badly; worst {:.6} wu / {:.5}°",
                        st.count, st.exact, st.bad, st.pos_max, st.rot_max
                    ));
                }
            });
    }

    // -----------------------------------------------------------------------------------------
    // the `precision` script verb
    // -----------------------------------------------------------------------------------------

    /// `precision [get] | precision <datum> | precision worst [n] | precision snap [<datum>|all]
    /// | precision ghost [on|off|get] [factor]`
    pub(crate) fn precision_cmd(&mut self, sub: &str, args: &[String]) -> Result<String, String> {
        match sub {
            "" | "get" | "summary" | "variant" => {
                let q = self.pose_quant().ok_or("no variant is open, so there is no save quantiser")?;
                let rows = self.predict_all_pose_errors();
                let mut st = ErrStats::default();
                for r in &rows { st.add(&r.err); }
                let step = q.pos_step();
                Ok(format!(
                    "{} save precision\n  world-bounds x {:.2}..{:.2}  y {:.2}..{:.2}  z {:.2}..{:.2}\n  position {} / {} / {} bits -> grid {:.6} / {:.6} / {:.6} wu (worst error = half of each)\n  rotation worst {:.4}°\n  {} object(s): {} exactly representable, {} quantising badly\n  position error mean {:.6} wu, max {:.6} wu\n  rotation error mean {:.5}°, max {:.5}°",
                    self.pose_quant_game_name(),
                    q.bounds[0], q.bounds[1], q.bounds[2], q.bounds[3], q.bounds[4], q.bounds[5],
                    q.axis_bits[0], q.axis_bits[1], q.axis_bits[2], step[0], step[1], step[2],
                    q.rot_worst_deg(), st.count, st.exact, st.bad,
                    st.pos_mean(), st.pos_max, st.rot_mean(), st.rot_max
                ))
            }
            "worst" => {
                let n = args.first().and_then(|s| s.parse::<usize>().ok()).unwrap_or(10).clamp(1, 200);
                let rows = self.predict_all_pose_errors();
                if rows.is_empty() { return Err("no variant is open (or it has no placed objects)".into()); }
                let (by_pos, by_rot) = rank_worst(rows, n);
                let mut s = format!("worst {n} by POSITION error:\n");
                for r in &by_pos {
                    s.push_str(&format!("  0x{:08X}  {:.6} wu  ({:+.6} {:+.6} {:+.6})  {}\n", r.datum, r.err.pos_total, r.err.pos_delta[0], r.err.pos_delta[1], r.err.pos_delta[2], r.name));
                }
                s.push_str(&format!("worst {n} by ROTATION error:\n"));
                for r in &by_rot {
                    s.push_str(&format!("  0x{:08X}  {:.5}°  (roll {:+.4} pitch {:+.4} yaw {:+.4})  {}\n", r.datum, r.err.rot_deg, r.err.euler_delta_deg[0], r.err.euler_delta_deg[1], r.err.euler_delta_deg[2], r.name));
                }
                Ok(s)
            }
            "snap" => {
                let targets: Vec<u32> = match args.first().map(|s| s.to_lowercase()).as_deref() {
                    None | Some("selection") | Some("sel") => {
                        let v = self.quant_selection();
                        if v.is_empty() { return Err("precision snap: nothing is selected (use `precision snap all`, or a datum)".into()); }
                        v
                    }
                    Some("all") => self.mvar_objects.iter().chain(self.local_objects.iter()).map(|o| o.datum).collect(),
                    Some(_) => {
                        let d = crate::script::parse_datum(&args[0])?;
                        vec![d]
                    }
                };
                self.snap_datums_to_representable(&targets)
            }
            "ghost" => {
                let mut g = self.quant_ghost;
                let mut touched = false;
                for a in args {
                    let al = a.to_lowercase();
                    if al == "get" { continue; }
                    if let Some(on) = crate::map_spawns::parse_flag(&al) {
                        g.on = on;
                        touched = true;
                    } else if let Some(f) = al.trim_start_matches('x').parse::<f32>().ok().filter(|v| v.is_finite()) {
                        g.exaggerate = f.clamp(1.0, EXAGGERATE_MAX);
                        touched = true;
                    } else {
                        return Err(format!("precision ghost: '{a}' is not on|off|get or an exaggeration factor (1..{EXAGGERATE_MAX:.0})"));
                    }
                }
                if touched {
                    self.quant_ghost = g;
                    save_setting(g);
                    self.overlays_dirty = true;
                }
                Ok(self.quant_ghost_status())
            }
            other => {
                let d = crate::script::parse_datum(other)?;
                let q = self.pose_quant().ok_or("no variant is open, so there is no save quantiser")?;
                let (pos, fwd, up) = self.live_pose(d).ok_or_else(|| format!("0x{d:08X} is not a placed variant object"))?;
                let e = q.predict(pos, fwd, up);
                let step = q.pos_step();
                // the rotation error in world units: how far it slides the piece's own far corner
                let corner = self.objscene().and_then(|s| s.object_obb(d))
                    .map(|o| (o.hx + o.hy + o.hz).length())
                    .filter(|r| r.is_finite() && *r > 1e-3)
                    .map(|r| format!("\n  rotation moves this piece's far corner ({r:.2} wu out) by {:.6} wu", r * e.rot_deg.to_radians()))
                    .unwrap_or_default();
                Ok(format!(
                    "0x{d:08X} ({})\n  live    {:.6} {:.6} {:.6}\n  stored  {:.6} {:.6} {:.6}\n  {}\n  grid {:.6} / {:.6} / {:.6} wu; worst-case use {:.0}% of position, {:.0}% of rotation{}{}",
                    self.pose_quant_game_name(),
                    pos[0], pos[1], pos[2], e.saved_pos[0], e.saved_pos[1], e.saved_pos[2],
                    e.summary(), step[0], step[1], step[2], e.pos_frac * 100.0, e.rot_frac * 100.0,
                    if e.is_exact() { "  — EXACT, nothing is lost" } else if e.quantises_badly() { "  — QUANTISES BADLY, nudge or snap it" } else { "" },
                    corner
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setting_round_trip_and_default_off() {
        let dir = std::env::temp_dir().join(format!("hms_quant_ghost_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(load_from_dir(&dir), GhostSetting { on: false, exaggerate: 1.0 }, "absent file = off at true scale");
        save_to_dir(&dir, GhostSetting { on: true, exaggerate: 40.0 });
        assert_eq!(load_from_dir(&dir), GhostSetting { on: true, exaggerate: 40.0 });
        save_to_dir(&dir, GhostSetting { on: false, exaggerate: 1.0 });
        assert!(!load_from_dir(&dir).on);
        // garbage, and an out-of-range factor, both fall back to the shipped default
        std::fs::write(dir.join(SETTING_FILE), "maybe 3").unwrap();
        assert_eq!(load_from_dir(&dir), GhostSetting { on: false, exaggerate: 3.0 });
        std::fs::write(dir.join(SETTING_FILE), "1 99999").unwrap();
        assert_eq!(load_from_dir(&dir), GhostSetting { on: true, exaggerate: 1.0 });
        std::fs::write(dir.join(SETTING_FILE), "1").unwrap();
        assert_eq!(load_from_dir(&dir), GhostSetting { on: true, exaggerate: 1.0 });
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// At ×1 the ghost IS the predicted pose; at ×k the position delta and the rotation angle both
    /// scale by k, and the ghost stays a rigid body (its basis stays orthonormal).
    #[test]
    fn exaggeration_scales_both_deltas() {
        let (lp, lf, lu) = ([1.0f32, 2.0, 3.0], [1.0f32, 0.0, 0.0], [0.0f32, 0.0, 1.0]);
        let (sp, sf, su) = ([1.02f32, 2.0, 3.0], [0.99985f32, 0.01745, 0.0], [0.0f32, 0.0, 1.0]);
        let (p1, r1) = exaggerated_pose(lp, lf, lu, sp, sf, su, 1.0);
        assert!((p1[0] - sp[0]).abs() < 1e-6, "x1 must be the predicted position");
        let base = crate::quant_error::basis_angle_deg(lf, lu, sf, su);
        let ang1 = crate::quant_error::basis_angle_deg(lf, lu, (r1 * Vec3::X).to_array(), (r1 * Vec3::Z).to_array());
        assert!((ang1 - base).abs() < 1e-3, "x1 must be the predicted rotation: {ang1} vs {base}");
        let (p10, r10) = exaggerated_pose(lp, lf, lu, sp, sf, su, 10.0);
        assert!((p10[0] - (lp[0] + 10.0 * (sp[0] - lp[0]))).abs() < 1e-5, "position delta must scale: {p10:?}");
        let ang10 = crate::quant_error::basis_angle_deg(lf, lu, (r10 * Vec3::X).to_array(), (r10 * Vec3::Z).to_array());
        assert!((ang10 - base * 10.0).abs() < 1e-2, "rotation delta must scale: {ang10} vs {}", base * 10.0);
        assert!((r10.length() - 1.0).abs() < 1e-5, "the ghost rotation must stay a unit quaternion");
    }

    /// An exactly-representable pose produces NO ghost transform change, whatever the factor —
    /// so a clean object never shows a misleading exaggerated ghost.
    #[test]
    fn an_exact_pose_has_no_ghost_offset() {
        let (p, f, u) = ([5.0f32, -6.0, 7.0], [0.0f32, 1.0, 0.0], [0.0f32, 0.0, 1.0]);
        let (gp, gr) = exaggerated_pose(p, f, u, p, f, u, 250.0);
        assert_eq!(gp, p);
        let xf = ghost_relocate(p, f, u, gp, gr);
        let probe = Vec3::new(1.0, 2.0, 3.0);
        assert!((xf.transform_point3(probe) - probe).length() < 1e-4, "an exact pose must not move the ghost");
    }

    /// The relocate transform is RIGID: it moves the object's geometry without scaling it, even
    /// when the live basis is stretched (a sheared scenario piece).
    #[test]
    fn relocate_is_rigid_even_for_a_stretched_basis() {
        let (p, f, u) = ([0.0f32, 0.0, 0.0], [2.0f32, 0.0, 0.0], [0.0f32, 0.0, 3.0]);
        let (gp, gr) = exaggerated_pose(p, f, u, [0.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 1.0);
        let xf = ghost_relocate(p, f, u, gp, gr);
        let (a, b) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 4.0, -2.0));
        let d0 = (b - a).length();
        let d1 = (xf.transform_point3(b) - xf.transform_point3(a)).length();
        assert!((d1 - d0).abs() < 1e-4, "ghost scaled the object: {d0} -> {d1}");
    }

    #[test]
    fn env_setting_parses_flag_and_factor() {
        assert_eq!(env_setting(), None, "unset = no override");
        std::env::set_var("HMS_QUANT_GHOST", "1");
        assert_eq!(env_setting(), Some(GhostSetting { on: true, exaggerate: 1.0 }));
        std::env::set_var("HMS_QUANT_GHOST", "on x50");
        assert_eq!(env_setting().map(|g| g.on), Some(true));
        std::env::set_var("HMS_QUANT_GHOST", "1x50");
        assert_eq!(env_setting(), Some(GhostSetting { on: true, exaggerate: 50.0 }));
        std::env::set_var("HMS_QUANT_GHOST", "0");
        assert_eq!(env_setting(), Some(GhostSetting { on: false, exaggerate: 1.0 }));
        std::env::remove_var("HMS_QUANT_GHOST");
    }
}
