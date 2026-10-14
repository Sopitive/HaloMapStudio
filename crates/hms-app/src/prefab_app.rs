//! PREFABS in the app (#prefab): the [`crate::prefab::PrefabScene`] adapter over `App` for all
//! four games, the Tools ▸ Prefabs panel, and the `prefab …` script verb's entry point.
//!
//! The adapter is the ONLY place that knows which game is open: it reads the open canvas's Forge
//! palette into the game-neutral [`PalRow`] list, reads a placed object into a [`PrefabObject`],
//! and creates one from a palette row. Everything above it - capture, the file format, the
//! anchor, the transform, the equivalence routing, the capacity check, the report - lives in
//! `prefab.rs` and never mentions a title.
//!
//! WHY the adapter goes through `App::spawn_palette_object`: that is the one call every game's
//! placement already uses (the palette drag-drop, the `place` verb, the Line / Fill tools), so a
//! prefab object is created exactly like a hand-placed one - correct record, correct slot,
//! correct save path - and prefabs cannot drift from the rest of the editor.

use std::collections::HashMap;

use eframe::egui;
use glam::Vec3;

use crate::mapcat::Game;
use crate::prefab::{self, Capacity, PalRow, PlaceOptions, PrefabObject, PrefabScene, Report, Xform};
use crate::xgame::table::Confidence;

/// Reach boundary quanta (11 bits over 0..200 wu) -> world units; the inverse of
/// [`crate::wu_to_bval`], which is what the editor already edits boundaries with.
fn reach_bval_to_wu(v: u16) -> f32 {
    (v as f32 / 2047.0) * 200.0
}

/// How many boundary values a shape uses (none / sphere 1 / cylinder 3 / box 4).
fn shape_count(shape: u8) -> usize {
    [0usize, 1, 3, 4][(shape as usize).min(3)]
}

// =================================================================================================
// the per-game palette, as the resolver sees it
// =================================================================================================

impl crate::App {
    /// The open canvas's Forge palette as game-neutral rows, INDEX-PARALLEL to
    /// `App::static_palette` (so a row's `index` is what `spawn_palette_object` takes).
    ///
    /// `max_allowed` / `price` come from the palette itself on Halo 4, H2A and Halo 3. Reach's
    /// Forge palette carries neither offline - the scenario has no price or per-entry limit field
    /// the editor can read - so Reach rows report 0 (unlimited / unpriced) and the only capacity
    /// gate on Reach is the variant's 651 object slots. The report says so rather than printing a
    /// number nobody can trust.
    fn prefab_palette_rows(&self) -> Vec<PalRow> {
        let names = &self.static_palette;
        if self.h4_active {
            let Some(pal) = self.h4_edit.pal.as_ref() else { return Vec::new() };
            return self
                .h4_edit
                .items
                .iter()
                .enumerate()
                .map(|(i, it)| {
                    let v = pal.resolve(Some(it.quota), Some(it.variant));
                    let cat_index = pal.entries.get(it.quota as usize).map(|e| e.category as u32).unwrap_or(0);
                    PalRow {
                        index: i,
                        key: v.map(|(_, v)| crate::xgame::object_key(&v.tag_path, &v.name, v.variant_sid)).unwrap_or_default(),
                        display: names.get(i).map(|(_, n)| n.clone()).unwrap_or_else(|| it.display.clone()),
                        tag: v.map(|(_, v)| v.tag_path.clone()).unwrap_or_default(),
                        variant: v.map(|(_, v)| v.name.clone()).unwrap_or_default(),
                        variant_sid: v.map(|(_, v)| v.variant_sid).unwrap_or(0),
                        spot: [cat_index, it.quota as u32, it.variant as u32],
                        price: it.price as f64,
                        max_allowed: it.max,
                    }
                })
                .collect();
        }
        if self.h3_active {
            let Some(es) = self.h3_scene.as_ref() else { return Vec::new() };
            let cache = es.cache();
            return self
                .h3_edit
                .items
                .iter()
                .enumerate()
                .map(|(i, it)| {
                    let tag = cache.tag_name(it.tag).to_string();
                    PalRow {
                        index: i,
                        key: crate::xgame::object_key(&tag, "", 0),
                        display: names.get(i).map(|(_, n)| n.clone()).unwrap_or_else(|| it.name.clone()),
                        tag,
                        variant: String::new(), // Halo 3 palette entries have no variants
                        variant_sid: 0,
                        spot: [it.block as u32, it.entry as u32, 0],
                        price: it.price as f64,
                        max_allowed: it.max_allowed,
                    }
                })
                .collect();
        }
        // Reach: rebuild the SAME dedup `forge_palette_static` does, so the row index matches
        // `static_palette` exactly, and recover the flat quota index (`folder`) the records use.
        let Some(sc) = self.scene_ctl.as_ref() else { return Vec::new() };
        let full = sc.forge_palette_full();
        if full.is_empty() {
            return Vec::new();
        }
        let types = sc.forge_type_order(&full);
        let mut seen: std::collections::HashSet<(u32, u32, u32)> = Default::default();
        let mut out = Vec::with_capacity(names.len());
        for e in &full {
            if e.tag_short == 0 || e.tag_short == 0xFFFF_FFFF {
                continue;
            }
            if !seen.insert((e.palette_index, e.entry_within, e.variant_within)) {
                continue;
            }
            let i = out.len();
            let tag = sc.tag_name_of(e.tag_short);
            let quota = types.iter().position(|&(p, w)| p == e.palette_index && w == e.entry_within).unwrap_or(0) as u32;
            out.push(PalRow {
                index: i,
                key: crate::xgame::object_key(&tag, &e.variant_name, e.variant_name_sid),
                display: names.get(i).map(|(_, n)| n.clone()).unwrap_or_else(|| e.name.clone()),
                tag,
                variant: e.variant_name.clone(),
                variant_sid: e.variant_name_sid,
                spot: [e.palette_index, quota, e.variant_within],
                price: 0.0,
                max_allowed: 0,
            });
        }
        out
    }

    /// Palette row index of a PLACED object, per game. This is the identity the capture writes
    /// down, so it has to be the same row the editor would place.
    fn prefab_pal_index_of(&self, datum: u32) -> Option<usize> {
        if self.h4_active {
            let r = self.h4_edit.src.get(&datum)?;
            let (q, v) = (r.quota?, r.variant.unwrap_or(0));
            return self.h4_edit.items.iter().position(|it| it.quota == q && it.variant == v);
        }
        if self.h3_active {
            if let Some(&row) = self.h3_edit.placed_rows.get(&datum) {
                return Some(row);
            }
            // a loaded object: its record's quota indexes the variant's quota table, whose
            // `definition` is the palette row's own `(block + 1) << 16 | entry`
            let r = self.h3_edit.src.get(&datum)?;
            let def = self.h3_variant_quota_definition(r.quota)?;
            return self.h3_edit.items.iter().position(|it| it.definition == def);
        }
        // Reach: the object TAG plus the model-variant sid pick exactly one palette row (every
        // palette variant has its own `obje` tag, and the sid separates a rocket Warthog from a
        // gauss one). Deliberately not the record's own `folder`/`item`: those are this variant's
        // quota coordinates, and what a prefab has to remember is the OBJECT. Placement then
        // derives fresh quota coordinates through the palette row, exactly as a hand-placed
        // object does, so a prefab copy is indistinguishable from one dropped from the palette.
        let o = self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == datum)?;
        self.static_palette
            .iter()
            .enumerate()
            .position(|(i, (t, _))| *t == o.primary_tag && self.static_palette_variant.get(i).copied().unwrap_or(0) == o.variant_name_sid)
            .or_else(|| self.static_palette.iter().position(|(t, _)| *t == o.primary_tag))
    }

    /// The `definition` of quota index `q` in the open Halo 3 variant (its quota table is the
    /// bridge between a record and a palette row).
    fn h3_variant_quota_definition(&self, q: i32) -> Option<u32> {
        let path = self.current_variant_path.as_ref().or(self.new_variant_template.as_ref())?;
        let v = crate::h3::mvar::parse_h3_variant(path).ok()?;
        v.quotas.get(usize::try_from(q).ok()?).map(|x| x.definition)
    }

    /// Boundary values of a placed object in WORLD UNITS - the one spelling all four titles agree
    /// on (Reach quantises 11 bits over 0..200, Halo 3 16 bits over 0..60, Halo 4 1/256 wu).
    fn prefab_boundary_wu(&self, m: &crate::ObjMeta) -> [f32; 4] {
        let n = shape_count(m.boundary_shape);
        let mut out = [0.0f32; 4];
        for k in 0..n {
            out[k] = if let Some(h4) = m.h4.as_deref() {
                crate::h4::edit::shape_raw_to_wu(h4.shape_values[k])
            } else if self.h3_active {
                crate::h3::mvar::dequantize_real(m.boundary[k] as u32, 0.0, crate::h3::mvar::H3_BOUNDARY_MAX, 16, false)
            } else {
                reach_bval_to_wu(m.boundary[k])
            };
        }
        out
    }

    /// Write world-unit boundary values back into a placed object's meta, in the open title's own
    /// quantisation (the inverse of [`Self::prefab_boundary_wu`]).
    fn prefab_set_boundary(&mut self, datum: u32, shape: u8, wu: [f32; 4]) {
        let h3 = self.h3_active;
        let Some(m) = self.mvar_meta.get_mut(&datum) else { return };
        m.boundary_shape = shape;
        let n = shape_count(shape);
        let mut q = [0u16; 4];
        for k in 0..n {
            q[k] = if m.h4.is_some() {
                crate::h4::edit::shape_wu_to_raw(wu[k])
            } else if h3 {
                crate::h3::mvar::quantize_real(wu[k].clamp(0.0, crate::h3::mvar::H3_BOUNDARY_MAX), 0.0, crate::h3::mvar::H3_BOUNDARY_MAX, 16, false, false).min(0xFFFF) as u16
            } else {
                crate::wu_to_bval(wu[k])
            };
        }
        m.boundary = q;
        if let Some(h4) = m.h4.as_deref_mut() {
            h4.shape_values = q;
        }
    }
}

// =================================================================================================
// PrefabScene for App
// =================================================================================================

impl PrefabScene for crate::App {
    fn pf_target(&self) -> Option<(Game, String)> {
        if self.objscene().map_or(true, |s| !s.has_cache()) {
            return None;
        }
        let mp = std::path::Path::new(&self.map_path);
        let stem = mp.file_stem().map(|s| s.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        if stem.is_empty() {
            return None;
        }
        // The game comes from the map CATALOG entry for the loaded cache (that is the only place
        // that tells Halo 4 and H2A apart - they share a container); the active-editor flags are
        // the fallback for a cache that is not in the catalog.
        let game = self
            .map_candidates
            .iter()
            .find(|c| c.path == mp)
            .or_else(|| self.selected_map.and_then(|i| self.map_candidates.get(i)))
            .map(|c| c.game)
            .unwrap_or(if self.h3_active {
                Game::Halo3
            } else if self.h4_active {
                Game::Halo4
            } else {
                Game::Reach
            });
        Some((game, stem))
    }

    fn pf_palette(&self) -> Vec<PalRow> {
        self.prefab_palette_rows()
    }

    fn pf_capacity(&self) -> Capacity {
        let pal = self.prefab_palette_rows();
        // Every record the variant holds occupies a slot, including the ones HMS could not
        // resolve - otherwise a full variant reads as having room and the save then silently
        // drops objects.
        let unresolved = if self.h4_active {
            self.h4_edit.unresolved.len()
        } else if self.h3_active {
            self.h3_edit.kept.len()
        } else {
            self.mvar_unresolved.len()
        };
        let slots_max = if self.h3_active { crate::h3::mvar::H3_SLOTS } else { crate::h4::mvar::H4_SLOTS };
        let mut per_quota: HashMap<u32, u32> = HashMap::new();
        let mut spent = 0.0f64;
        for o in self.mvar_objects.iter().chain(self.local_objects.iter()) {
            if let Some(row) = self.prefab_pal_index_of(o.datum).and_then(|i| pal.get(i)) {
                *per_quota.entry(row.spot[1]).or_default() += 1;
                spent += row.price;
            }
        }
        // Reach's palette has no offline prices, so its budget is not a number worth asserting
        // on; Halo 4 / H2A price every entry and the variant carries the cap.
        // Only the Halo 4 family prices its palette entries offline (Reach has no price field HMS
        // can read; the Halo 3 variant budget is not carried on `variant_globals`).
        let priced = self.h4_active;
        let budget_max = self.variant_globals.as_ref().map(|g| g.budget_max as f64).unwrap_or(0.0);
        Capacity { slots_used: self.mvar_objects.len() + self.local_objects.len() + unresolved, slots_max, budget_spent: spent, budget_max, budget_priced: priced, per_quota }
    }

    fn pf_selection(&self) -> Vec<u32> {
        self.selected_set.clone()
    }

    fn pf_set_selection(&mut self, datums: Vec<u32>) {
        self.selected_datum = datums.last().copied();
        self.selected_set = datums;
        self.apply_selection_highlight();
    }

    fn pf_read(&self, datum: u32) -> Option<(Option<usize>, PrefabObject)> {
        let o = self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == datum)?;
        let m = self.mvar_meta.get(&datum)?;
        let team_u8 = self.mvar_colors.get(&datum).map(|(t, _)| *t).unwrap_or(m.team);
        let mut labels: Vec<String> = Vec::new();
        if !m.label.is_empty() {
            labels.push(m.label.clone());
        }
        let h4 = m.h4.as_deref();
        if let Some(h) = h4 {
            for l in h.labels_extra.iter().flatten() {
                if let Some(s) = self.mvar_labels.get(*l as usize) {
                    if !s.is_empty() && !labels.iter().any(|x| x.eq_ignore_ascii_case(s)) {
                        labels.push(s.clone());
                    }
                }
            }
        }
        let po = PrefabObject {
            key: String::new(), // filled by `capture` from the palette row
            display: m.name.clone(),
            tag: self.objscene().map(|s| s.tag_name_of(o.primary_tag)).unwrap_or_default(),
            variant: String::new(),
            variant_sid: o.variant_name_sid,
            spot: None,
            pos: o.pos,
            fwd: o.fwd,
            up: o.up,
            team: if team_u8 == 0xFF { -1 } else { team_u8.min(8) as i16 },
            color: (m.color >= 0).then(|| m.color as u8),
            spawn_time: m.respawn,
            spawn_sequence: m.spawn_seq,
            user_data: h4.map(|h| h.user_data as i32).unwrap_or(0),
            placement: m.placement,
            placement_hi: h4.map(|h| h.placement_hi).unwrap_or(0),
            shape: m.boundary_shape,
            shape_wu: self.prefab_boundary_wu(m),
            labels,
            tele_channel: m.tele_channel,
            tele_passability: m.tele_passability,
            weapon_clips: m.weapon_clips,
            object_type: m.cached_type,
            scale_q: h4.map(|h| h.scale_q).filter(|q| *q != crate::h4::mvar::SCALE_Q_DEFAULT),
            locked: h4.map(|h| h.locked).unwrap_or(false),
            trait_zone: h4.map(|h| h.trait_zone).unwrap_or(0),
            scaled: m.flags.scaled,
            shadow: m.flags.shadow,
            damage: m.flags.damage,
        };
        Some((self.prefab_pal_index_of(datum), po))
    }

    fn pf_place(&mut self, pal: usize, o: &PrefabObject) -> Result<u32, String> {
        // One object, created the SAME way the palette drag-drop and `place` create one.
        let (datum, _) = self.spawn_palette_object(pal, Vec3::from(o.pos))?;
        self.set_movable_pose(datum, Vec3::from(o.pos), Vec3::from(o.fwd), Vec3::from(o.up));
        let team_u8: u8 = if o.team < 0 { 0xFF } else { o.team.min(8) as u8 };
        let labels = self.mvar_labels.clone();
        let label_idx = |s: &str| labels.iter().position(|k| k.eq_ignore_ascii_case(s)).map(|i| i as u16).unwrap_or(0xFFFF);
        if let Some(m) = self.mvar_meta.get_mut(&datum) {
            m.team = team_u8;
            m.color = o.color.map(|c| c as i32).unwrap_or(-1);
            m.respawn = o.spawn_time;
            m.spawn_seq = o.spawn_sequence;
            m.placement = o.placement;
            m.weapon_clips = o.weapon_clips;
            m.tele_channel = o.tele_channel;
            m.tele_passability = o.tele_passability;
            m.label = o.labels.first().cloned().unwrap_or_default();
            m.label_idx = o.labels.first().map(|s| label_idx(s)).unwrap_or(0xFFFF);
            m.flags.scaled = o.scaled;
            m.flags.shadow = o.shadow;
            m.flags.damage = o.damage;
            if let Some(h4) = m.h4.as_deref_mut() {
                h4.user_data = o.user_data.clamp(-128, 127) as i8;
                h4.placement_hi = o.placement_hi;
                h4.locked = o.locked;
                h4.trait_zone = o.trait_zone.min(3);
                if let Some(q) = o.scale_q {
                    h4.scale_q = q.min(63);
                }
                for k in 0..3 {
                    h4.labels_extra[k] = o.labels.get(k + 1).map(|s| label_idx(s)).filter(|i| *i != 0xFFFF).map(|i| i as u8);
                }
            }
        }
        self.prefab_set_boundary(datum, o.shape, o.shape_wu);
        self.mvar_colors.insert(datum, (team_u8, o.color.unwrap_or(0xFF)));
        Ok(datum)
    }

    fn pf_remove(&mut self, datums: &[u32]) {
        if datums.is_empty() {
            return;
        }
        // The editor's own "drop these spawned copies" primitive (`remove_copy`), so a rollback
        // leaves exactly the state a cancelled duplicate would.
        self.remove_copy(datums);
        self.selected_set.retain(|d| !datums.contains(d));
        if self.selected_datum.map_or(false, |d| datums.contains(&d)) {
            self.selected_datum = self.selected_set.last().copied();
        }
    }

    fn pf_labels(&self) -> Vec<String> {
        self.mvar_labels.clone()
    }

    fn pf_push_undo(&mut self) {
        if self.prefab_ui.script_undo {
            return; // the script runner owns the step (see `PrefabUi::script_undo`)
        }
        self.push_edit_undo();
    }

    fn pf_drop_point(&self) -> [f32; 3] {
        // Where `place … at camera` puts a single object: the map surface the camera is looking
        // at, else 12 wu ahead.
        let fwd = self.camera.forward();
        self.objscene()
            .and_then(|s| s.raycast_scene(self.camera.pos, fwd))
            .unwrap_or(self.camera.pos + fwd * 12.0)
            .to_array()
    }

    fn pf_invalidate(&mut self) {
        if let Some(s) = self.objscene_mut() {
            s.invalidate();
        }
        self.overlays_dirty = true;
        self.highlight_dirty = true;
    }
}

// =================================================================================================
// the panel
// =================================================================================================

/// What Tools ▸ Prefabs remembers between frames.
pub struct PrefabUi {
    pub open: bool,
    /// The list, re-read when the folder changes under us (import / save / delete).
    pub entries: Vec<prefab::Entry>,
    pub listed: bool,
    pub selected: Option<usize>,
    /// Name field for Save selection / Rename.
    pub name: String,
    pub filter: String,
    /// Placement controls.
    pub at_camera: bool,
    pub origin: [f32; 3],
    pub yaw_deg: f32,
    pub mirror: Option<usize>,
    pub allow_cross_game: bool,
    pub enforce_quotas: bool,
    pub enforce_budget: bool,
    pub min_conf: Confidence,
    /// The chosen prefab, read once when the selection changes (the panel paints every frame, so
    /// re-reading and re-parsing the file per frame would be pure waste).
    pub loaded: Option<(std::path::PathBuf, Result<(prefab::Prefab, Vec<String>), String>)>,
    /// The last dry run (shown before anything is created) and the last real placement.
    pub preview: Option<Report>,
    pub last: Option<Report>,
    pub show_all_rows: bool,
    pub status: String,
    /// Set when the user asked the file browser for a prefab to import.
    pub want_import: bool,
    /// Suppress the placement's own undo snapshot while a SCRIPT drives it: `App::run_script`
    /// already wraps a whole run in one snapshot, so pushing a second identical one here would
    /// make the first Ctrl+Z appear to do nothing. The panel's own Place button does not set it.
    pub script_undo: bool,
    /// Rename state: the entry being renamed.
    pub renaming: Option<usize>,
}

impl Default for PrefabUi {
    fn default() -> Self {
        PrefabUi {
            open: false,
            entries: Vec::new(),
            listed: false,
            selected: None,
            name: String::new(),
            filter: String::new(),
            loaded: None,
            at_camera: true,
            origin: [0.0; 3],
            yaw_deg: 0.0,
            mirror: None,
            allow_cross_game: true,
            enforce_quotas: true,
            enforce_budget: true,
            min_conf: Confidence::Approximate,
            preview: None,
            last: None,
            show_all_rows: false,
            status: String::new(),
            want_import: false,
            script_undo: false,
            renaming: None,
        }
    }
}

impl PrefabUi {
    pub fn refresh(&mut self) {
        self.entries = prefab::list();
        self.listed = true;
        if self.selected.map_or(false, |i| i >= self.entries.len()) {
            self.selected = None;
        }
        self.loaded = None;
    }
    /// The chosen prefab, read from disk only when the chosen PATH changes.
    fn chosen(&mut self) -> Option<&Result<(prefab::Prefab, Vec<String>), String>> {
        let e = self.selected.and_then(|i| self.entries.get(i))?;
        if e.problem.is_some() {
            return None;
        }
        let path = e.path.clone();
        if self.loaded.as_ref().map_or(true, |(p, _)| *p != path) {
            self.loaded = Some((path.clone(), prefab::load_file(&path)));
        }
        self.loaded.as_ref().map(|(_, r)| r)
    }
    fn options(&self, drop: [f32; 3]) -> PlaceOptions {
        PlaceOptions {
            xform: Xform { origin: if self.at_camera { drop } else { self.origin }, yaw_deg: self.yaw_deg, mirror: self.mirror },
            min_confidence: self.min_conf,
            enforce_quotas: self.enforce_quotas,
            enforce_budget: self.enforce_budget,
            allow_cross_game: self.allow_cross_game,
            dry_run: false,
        }
    }
}

impl crate::App {
    /// The `prefab …` script verb (shared by the script panel, the TCP server and the MCP bridge).
    pub(crate) fn prefab_command(&mut self, args: &str) -> Result<String, String> {
        self.prefab_ui.script_undo = true;
        let r = prefab::command(self, args);
        self.prefab_ui.script_undo = false;
        self.prefab_ui.listed = false; // the folder may have changed
        if r.is_ok() {
            self.prefab_ui.last = None;
            self.prefab_ui.preview = None;
        }
        r
    }

    /// Tools ▸ Prefabs. Save the selection, browse what is saved, preview it, place it.
    pub(crate) fn prefab_panel_ui(&mut self, ctx: &egui::Context) {
        if !self.prefab_ui.open {
            return;
        }
        if !self.prefab_ui.listed {
            self.prefab_ui.refresh();
        }
        let drop = self.pf_drop_point();
        let target = self.pf_target();
        let sel_n = self.selected_set.len();
        let mut open = self.prefab_ui.open;
        // Actions are collected and run AFTER the window closure, so the panel never holds a
        // borrow of `self` while an editor verb mutates it.
        let mut do_save = false;
        let mut do_place = false;
        let mut do_preview = false;
        let mut do_delete: Option<String> = None;
        let mut do_rename: Option<(String, String)> = None;
        let mut do_export: Option<String> = None;
        let mut do_import = false;
        let mut refresh = false;

        egui::Window::new("Prefabs")
            .open(&mut open)
            .default_width(560.0)
            .vscroll(false)
            .show(ctx, |ui| {
                let st = &mut self.prefab_ui;
                ui.label(
                    egui::RichText::new(
                        "A prefab is a saved group of Forge objects - a tower, a ramp set, a whole base - you can drop into any variant again and again. \
                         Each one is a single shareable file.",
                    )
                    .weak(),
                );
                ui.separator();

                // --- save the selection -------------------------------------------------------
                ui.horizontal(|ui| {
                    ui.strong("Save selection");
                    ui.add(egui::TextEdit::singleline(&mut st.name).hint_text("prefab name").desired_width(180.0));
                    let can = sel_n > 0 && !st.name.trim().is_empty() && target.is_some();
                    let b = ui.add_enabled(can, egui::Button::new(format!("Save {sel_n} selected")));
                    if b.on_hover_text(
                        "Captures the selected objects with their poses relative to the selection's own footprint centre and floor, \
                         plus team, colour, spawn order and time, labels, boundary shape and size and the HMS scale / shadow / damage flags.",
                    )
                    .clicked()
                    {
                        do_save = true;
                    }
                    if sel_n == 0 {
                        ui.label(egui::RichText::new("select some objects first").weak());
                    }
                });
                ui.separator();

                // --- the list -----------------------------------------------------------------
                ui.horizontal(|ui| {
                    ui.strong(format!("Saved prefabs ({})", st.entries.len()));
                    ui.add(egui::TextEdit::singleline(&mut st.filter).hint_text("filter").desired_width(120.0));
                    if ui.button("Refresh").clicked() {
                        refresh = true;
                    }
                    if ui.button("Import file…").on_hover_text("Open a prefab someone shared and copy it into your prefabs folder").clicked() {
                        do_import = true;
                    }
                    if let Some(d) = prefab::prefabs_dir() {
                        ui.label(egui::RichText::new(d.display().to_string()).weak()).on_hover_text("Prefabs live here - one file each, safe to copy or send");
                    }
                });
                let f = st.filter.to_lowercase();
                egui::ScrollArea::vertical().max_height(180.0).id_salt("prefab-list").show(ui, |ui| {
                    egui::Grid::new("prefab-grid").num_columns(5).striped(true).spacing([8.0, 2.0]).show(ui, |ui| {
                        for i in 0..st.entries.len() {
                            if !f.is_empty() && !st.entries[i].name.to_lowercase().contains(&f) {
                                continue;
                            }
                            let e = st.entries[i].clone();
                            let chosen = st.selected == Some(i);
                            if ui.selectable_label(chosen, &e.name).clicked() {
                                st.selected = Some(i);
                                st.preview = None;
                                st.name = e.name.clone();
                            }
                            match &e.problem {
                                Some(why) => {
                                    ui.colored_label(egui::Color32::from_rgb(230, 120, 120), "unreadable").on_hover_text(why.clone());
                                    ui.label("");
                                    ui.label("");
                                }
                                None => {
                                    ui.label(format!("{} object(s)", e.objects));
                                    let g = crate::xgame::game_from_str(&e.game).map(|g| g.display_name().to_string()).unwrap_or_else(|| e.game.clone());
                                    ui.label(g);
                                    ui.label(egui::RichText::new(e.canvas.clone()).weak());
                                }
                            }
                            ui.horizontal(|ui| {
                                if ui.small_button("Rename").clicked() {
                                    st.renaming = Some(i);
                                    st.name = e.name.clone();
                                }
                                if ui.small_button("Export…").on_hover_text("Write a copy you can hand to someone").clicked() {
                                    do_export = Some(e.name.clone());
                                }
                                if ui.small_button("Delete").clicked() {
                                    do_delete = Some(e.name.clone());
                                }
                            });
                            ui.end_row();
                        }
                    });
                });
                if let Some(i) = st.renaming {
                    if let Some(e) = st.entries.get(i).cloned() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Rename '{}' to", e.name));
                            ui.add(egui::TextEdit::singleline(&mut st.name).desired_width(180.0));
                            if ui.button("Rename").clicked() {
                                do_rename = Some((e.name.clone(), st.name.clone()));
                            }
                            if ui.button("Cancel").clicked() {
                                st.renaming = None;
                            }
                        });
                    } else {
                        st.renaming = None;
                    }
                }
                ui.separator();

                // --- what the chosen prefab contains ------------------------------------------
                match st.chosen().cloned() {
                    None => {
                        ui.label(egui::RichText::new("Pick a prefab above to preview and place it.").weak());
                    }
                    Some(read) => {
                        match read {
                            Err(why) => {
                                ui.colored_label(egui::Color32::from_rgb(230, 120, 120), format!("cannot read this prefab: {why}"));
                            }
                            Ok((pf, warn)) => {
                                ui.strong(pf.summary());
                                if !pf.notes.is_empty() {
                                    ui.label(egui::RichText::new(&pf.notes).weak());
                                }
                                for w in &warn {
                                    ui.colored_label(egui::Color32::from_rgb(230, 190, 80), format!("! {w}"));
                                }
                                egui::CollapsingHeader::new(format!("Contents ({} kinds)", pf.contents().len())).default_open(false).show(ui, |ui| {
                                    for (k, n) in pf.contents() {
                                        ui.label(format!("{n:>4} x {k}"));
                                    }
                                });
                                ui.separator();

                                // --- placement controls ------------------------------------
                                let cross = target.as_ref().map_or(false, |(g, _)| pf.game_of() != Some(*g));
                                ui.horizontal(|ui| {
                                    ui.strong("Place");
                                    ui.checkbox(&mut st.at_camera, "at the camera").on_hover_text("Drop it on the surface the camera is looking at; uncheck to type a position");
                                    if st.at_camera {
                                        ui.label(egui::RichText::new(format!("({:.1}, {:.1}, {:.1})", drop[0], drop[1], drop[2])).weak());
                                    } else {
                                        for k in 0..3 {
                                            ui.add(egui::DragValue::new(&mut st.origin[k]).speed(0.25).prefix(["x ", "y ", "z "][k]));
                                        }
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.add(egui::DragValue::new(&mut st.yaw_deg).speed(1.0).range(-360.0..=360.0).suffix(" deg yaw"));
                                    for (lbl, ax) in [("no mirror", None), ("mirror X", Some(0usize)), ("mirror Y", Some(1)), ("mirror Z", Some(2))] {
                                        if ui.selectable_label(st.mirror == ax, lbl).clicked() {
                                            st.mirror = ax;
                                        }
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.checkbox(&mut st.enforce_quotas, "respect per-object limits");
                                    ui.checkbox(&mut st.enforce_budget, "respect the budget");
                                });
                                if cross {
                                    ui.separator();
                                    ui.colored_label(
                                        egui::Color32::from_rgb(230, 190, 80),
                                        format!(
                                            "CROSS-GAME: this prefab was built in {} and you are on {}.",
                                            crate::xgame::game_from_str(&pf.game).map(|g| g.display_name().to_string()).unwrap_or_else(|| pf.game.clone()),
                                            target.as_ref().map(|(g, _)| g.display_name()).unwrap_or("?")
                                        ),
                                    );
                                    ui.label(
                                        egui::RichText::new(
                                            "Every piece goes through the cross-game equivalence resolver. Preview it first: the report says, per object, \
                                             whether the match was exact, close or only APPROXIMATE, and anything with no acceptable match is left out rather than \
                                             replaced with the wrong piece.",
                                        )
                                        .weak(),
                                    );
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut st.allow_cross_game, "allow cross-game placement");
                                        ui.label("accept at best:");
                                        for c in [Confidence::Exact, Confidence::Close, Confidence::Approximate] {
                                            if ui.selectable_label(st.min_conf == c, c.name()).clicked() {
                                                st.min_conf = c;
                                            }
                                        }
                                    });
                                }
                                ui.separator();
                                ui.horizontal(|ui| {
                                    if ui.add_enabled(target.is_some(), egui::Button::new("Preview")).on_hover_text("Resolve and report WITHOUT creating anything").clicked() {
                                        do_preview = true;
                                    }
                                    if ui.add_enabled(target.is_some(), egui::Button::new("Place")).on_hover_text("One undo step; the prefab is selected afterwards so you can nudge it").clicked() {
                                        do_place = true;
                                    }
                                    if target.is_none() {
                                        ui.label(egui::RichText::new("open a map first").weak());
                                    }
                                });
                            }
                        }
                    }
                }

                // --- the report ---------------------------------------------------------------
                if let Some(rep) = st.preview.as_ref().or(st.last.as_ref()) {
                    ui.separator();
                    let bad = rep.refused.is_some();
                    let col = if bad {
                        egui::Color32::from_rgb(230, 120, 120)
                    } else if rep.approximate() > 0 || rep.unmatched() > 0 {
                        egui::Color32::from_rgb(230, 190, 80)
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.colored_label(col, rep.summary());
                    for w in &rep.warnings {
                        ui.colored_label(egui::Color32::from_rgb(230, 190, 80), format!("! {w}"));
                    }
                    let interesting = rep.objects.iter().filter(|o| !o.kind.is_solid()).count();
                    egui::CollapsingHeader::new(format!("Per-object report ({} object(s), {interesting} worth checking)", rep.objects.len()))
                        .default_open(interesting > 0)
                        .show(ui, |ui| {
                            ui.checkbox(&mut st.show_all_rows, "show every object (not just the ones worth checking)");
                            let all = st.show_all_rows;
                            egui::ScrollArea::vertical().max_height(220.0).id_salt("prefab-report").show(ui, |ui| {
                                for o in &rep.objects {
                                    if !all && o.kind.is_solid() {
                                        continue;
                                    }
                                    let c = match o.kind {
                                        prefab::MatchKind::Unmatched => egui::Color32::from_rgb(230, 120, 120),
                                        prefab::MatchKind::Approximate => egui::Color32::from_rgb(230, 190, 80),
                                        _ => ui.visuals().text_color(),
                                    };
                                    ui.colored_label(
                                        c,
                                        format!(
                                            "[{:>3}] {:<12} {} -> {}   {}{}",
                                            o.index,
                                            o.kind.name(),
                                            o.src,
                                            if o.dst.is_empty() { "-" } else { &o.dst },
                                            o.via,
                                            if o.note.is_empty() { String::new() } else { format!("  ({})", o.note) }
                                        ),
                                    );
                                }
                            });
                        });
                }
                if !st.status.is_empty() {
                    ui.separator();
                    ui.label(&st.status);
                }
            });
        self.prefab_ui.open = open;

        // ---- the collected actions ----------------------------------------------------------
        if refresh {
            self.prefab_ui.refresh();
        }
        if do_import {
            self.prefab_ui.want_import = true;
            self.open_prefab_browser();
        }
        if do_save {
            let name = self.prefab_ui.name.clone();
            match prefab::capture(self, &name).and_then(|pf| prefab::store(&pf).map(|p| (pf, p))) {
                Ok((pf, p)) => {
                    self.prefab_ui.status = format!("Saved '{}' ({} objects) to {}", pf.name, pf.objects.len(), p.display());
                    self.status = self.prefab_ui.status.clone();
                    self.prefab_ui.refresh();
                    self.prefab_ui.selected = self.prefab_ui.entries.iter().position(|e| e.name == pf.name);
                }
                Err(e) => {
                    self.prefab_ui.status = format!("Save failed: {e}");
                }
            }
        }
        if let Some(name) = do_delete {
            self.prefab_ui.status = match prefab::delete(&name) {
                Ok(_) => format!("Deleted '{name}'"),
                Err(e) => format!("Delete failed: {e}"),
            };
            self.prefab_ui.selected = None;
            self.prefab_ui.refresh();
        }
        if let Some((old, new)) = do_rename {
            self.prefab_ui.status = match prefab::rename(&old, &new) {
                Ok(p) => format!("Renamed to {}", p.display()),
                Err(e) => format!("Rename failed: {e}"),
            };
            self.prefab_ui.renaming = None;
            self.prefab_ui.refresh();
        }
        if let Some(name) = do_export {
            // The stored file IS the share format, so "export" is only ever a copy to somewhere a
            // user can find and attach: their home folder, else the app's data folder.
            let dst = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(std::path::PathBuf::from)
                .or_else(crate::project::settings_dir)
                .map(|d| d.join(format!("{}.{}", prefab::clean_name(&name), prefab::EXT)));
            self.prefab_ui.status = match (prefab::load_named(&name), dst) {
                (Ok((pf, _)), Some(p)) => match prefab::save_file(&pf, &p) {
                    Ok(p) => format!("Wrote {} - this single file is the whole prefab; send it to anyone", p.display()),
                    Err(e) => format!("Export failed: {e}"),
                },
                (Err(e), _) => format!("Export failed: {e}"),
                (_, None) => "Export failed: no settings folder".into(),
            };
        }
        if do_preview || do_place {
            let chosen = self.prefab_ui.selected.and_then(|i| self.prefab_ui.entries.get(i).cloned());
            if let Some(e) = chosen {
                match prefab::load_file(&e.path) {
                    Ok((pf, _)) => {
                        let mut opt = self.prefab_ui.options(drop);
                        opt.dry_run = do_preview;
                        let rep = prefab::place(self, &pf, &opt);
                        self.status = rep.summary().lines().next().unwrap_or_default().to_string();
                        if do_preview {
                            self.prefab_ui.preview = Some(rep);
                        } else {
                            self.prefab_ui.preview = None;
                            self.prefab_ui.last = Some(rep);
                        }
                    }
                    Err(why) => self.prefab_ui.status = format!("cannot read this prefab: {why}"),
                }
            }
        }
    }

    /// A prefab file the browser handed back: validate it, copy it into the prefabs folder and
    /// select it. The file is untrusted, so nothing is placed - the user sees what it holds first.
    pub(crate) fn prefab_import_file(&mut self, path: std::path::PathBuf) {
        self.prefab_ui.want_import = false;
        match prefab::import_from(&path) {
            Ok((out, pf, warn)) => {
                self.prefab_ui.open = true;
                self.prefab_ui.refresh();
                self.prefab_ui.selected = self.prefab_ui.entries.iter().position(|e| e.name == pf.name);
                let mut s = format!("Imported '{}' ({} objects, {}) to {}", pf.name, pf.objects.len(), pf.game, out.display());
                for w in warn {
                    s.push_str(&format!("  !{w}"));
                }
                self.prefab_ui.status = s.clone();
                self.status = s;
            }
            Err(e) => {
                self.prefab_ui.open = true;
                self.prefab_ui.status = format!("Not a usable prefab: {e}");
                self.status = self.prefab_ui.status.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// The boundary conversions are the one piece of per-game arithmetic the adapter owns, and a
    /// prefab's sizes are stored in world units, so a round trip has to come back.
    #[test]
    fn boundary_world_units_round_trip() {
        for wu in [0.0f32, 0.5, 1.0, 12.5, 40.0, 60.0] {
            let q = crate::wu_to_bval(wu);
            assert!((super::reach_bval_to_wu(q) - wu).abs() < 0.11, "reach {wu} -> {q} -> {}", super::reach_bval_to_wu(q));
            let r = crate::h4::edit::shape_wu_to_raw(wu);
            assert!((crate::h4::edit::shape_raw_to_wu(r) - wu).abs() < 0.01, "h4 {wu}");
            let h = crate::h3::mvar::quantize_real(wu, 0.0, crate::h3::mvar::H3_BOUNDARY_MAX, 16, false, false).min(0xFFFF);
            assert!((crate::h3::mvar::dequantize_real(h, 0.0, crate::h3::mvar::H3_BOUNDARY_MAX, 16, false) - wu).abs() < 0.01, "h3 {wu}");
        }
        assert_eq!(super::shape_count(0), 0);
        assert_eq!(super::shape_count(1), 1);
        assert_eq!(super::shape_count(2), 3);
        assert_eq!(super::shape_count(3), 4);
        assert_eq!(super::shape_count(200), 4, "a nonsense shape must not index out of range");
    }
}
