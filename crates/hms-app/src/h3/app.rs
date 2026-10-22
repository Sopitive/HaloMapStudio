//! The `App` side of the Halo 3 editor (the h4_app.rs counterpart, kept under h3/): the editor
//! scene install, the palette panel rows, the variant's objects as editor objects, placing a
//! palette row, the duplicate hook and File > Save / Save As. #h3-edit
//!
//! The shared editor state (`mvar_objects` / `local_objects` / `mvar_meta` / `mvar_colors` /
//! `static_palette*` / `current_variant_path`) is filled exactly like the Halo 4 path fills it,
//! so every verb, panel and undo step works unchanged; the Halo 3-only state is `H3EditState`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use eframe::egui;
use hms_ipc::ObjectInfo;

use super::edit::{build_h3_save_list, H3EditorSet, H3SaveReport, h3_tag, variant_to_editor, H3LiveMeta, ScnEntry, StorageRole, H3_MARKER_TAG, MP_SYMMETRY_MASK, NEW_SLOT};
use super::scenario::{is_scenario_datum, SLOT_SCENARIO};
use super::edit_scene::{H3EditorAssets, H3ObjectScene};
use super::mvar::{parse_h3_variant, save_h3_variant, H3HeaderEdits, H3Variant, H3VariantObject};
use super::palette::{item_of_quota, palette, H3PaletteItem};
use crate::{App, BspWarnAction, ObjMeta};

/// What `h3_build_save` hands back (see there).
pub(crate) struct H3Built {
    pub src: std::path::PathBuf,
    /// The source variant as read from disk.
    pub source: H3Variant,
    /// The variant to encode.
    pub v: H3Variant,
    pub list: Vec<H3VariantObject>,
    pub report: H3SaveReport,
}

/// The save check's findings, shown by `h3_save_window_ui`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H3SaveGate {
    /// Datums outside the variant bounds ("Save anyway" CLAMPS them).
    pub out_of_bounds: Vec<u32>,
    /// (palette name, objects placed, the quota's `maximum_allowed`).
    pub over_quota: Vec<(String, u32, u8)>,
    /// (spent, maximum).
    pub over_budget: Option<(f32, f32)>,
    /// Objects past the 640 slots, and objects with no palette row: both dropped by the save.
    pub no_slot: usize,
    pub unsaveable: usize,
}

impl H3SaveGate {
    pub fn is_clean(&self) -> bool {
        self.out_of_bounds.is_empty() && self.over_quota.is_empty() && self.over_budget.is_none() && self.no_slot == 0 && self.unsaveable == 0
    }

    pub(crate) fn from_built(b: &H3Built, items: &[H3PaletteItem]) -> Self {
        let name_of = |qi: usize| -> String {
            b.v.quotas.get(qi).and_then(|q| item_of_quota(items, q)).and_then(|i| items.get(i)).map(|it| it.name.clone())
                .unwrap_or_else(|| format!("quota #{qi}"))
        };
        H3SaveGate {
            out_of_bounds: b.report.out_of_bounds.clone(),
            over_quota: b.report.over_quota.iter().map(|&(qi, n, allowed)| (name_of(qi), n, allowed)).collect(),
            over_budget: b.report.over_budget,
            no_slot: b.report.over_slot_cap,
            unsaveable: b.report.unsaveable,
        }
    }
}

/// The save-time fields an editor object contributes to its record: team (from the colour map),
/// spawn time, boundary, and the two role-dependent ones (`shared_storage` lives in whichever
/// panel field names its role for this object type; the symmetry bits in `placement`).
pub fn live_meta_of(om: &ObjMeta, team: u8, palette_item: Option<usize>) -> H3LiveMeta {
    let shared = match StorageRole::of(om.cached_type) {
        StorageRole::SpareClips => Some(om.weapon_clips),
        StorageRole::TeleporterChannel => Some(om.tele_channel),
        StorageRole::SpawnOrder => Some(om.spawn_seq.clamp(0, 255) as u8),
        StorageRole::None => None,
    };
    H3LiveMeta {
        team, spawn_time: om.respawn, boundary_shape: om.boundary_shape, boundary_q: om.boundary, palette_item,
        shared_storage: shared, symmetry: Some(om.placement & MP_SYMMETRY_MASK),
    }
}

#[derive(Default)]
pub struct H3EditState {
    /// The palette rows, parallel to `App.static_palette`.
    pub items: Vec<H3PaletteItem>,
    /// datum -> the source `.mvar` record (loaded objects; a placed row gets one on its first save).
    pub src: HashMap<u32, H3VariantObject>,
    /// Records a save carries verbatim (scenario slots the editor does not own, unresolved placements).
    pub kept: Vec<H3VariantObject>,
    /// The base map's own objects (scenario.rs): every managed scenario slot, live or deleted.
    pub scn: Vec<ScnEntry>,
    /// The load left the base map's own variant slots out of the static scenery, so the editor
    /// must hold them (set by `h3_install_editor_scene`).
    pub scenario_owned: bool,
    /// Palette row of every NEW placement (datum -> row), until it is saved.
    pub placed_rows: HashMap<u32, usize>,
    pub last_save_note: RefCell<String>,
    /// The save check holding a Save / Save As until the window decides.
    pub save_pending: Option<(BspWarnAction, H3SaveGate)>,
    /// Set for the one re-entry "Save anyway" makes, so the gate lets it through.
    pub save_skip_once: bool,
}

/// `ObjMeta` of a Halo 3 record: the shared fields under their Reach names (folder = quota,
/// respawn = spawn time, cached_type = multiplayer object type, boundary = the raw 16-bit quanta).
pub fn obj_meta_from_h3(m: &super::edit::H3ObjMeta, pos: [f32; 3]) -> ObjMeta {
    let role = StorageRole::of(m.object_type);
    ObjMeta {
        name: m.name.clone(),
        folder: m.quota.clamp(0, 0xFFFE) as u16,
        item: 0,
        pos,
        team: m.team,
        color: -1,
        cached_type: m.object_type,
        spawn_seq: if role == StorageRole::SpawnOrder { m.shared_storage as i32 } else { 0 },
        respawn: m.spawn_time,
        label_idx: 0xFFFF,
        label: String::new(),
        placement: m.mp_flags,
        boundary_shape: m.boundary_shape,
        boundary: m.boundary_q,
        // the record's shared byte lands in the field that names its role for this object type
        weapon_clips: if role == StorageRole::SpareClips { m.shared_storage } else { 0 },
        tele_channel: if role == StorageRole::TeleporterChannel { m.shared_storage } else { 0 },
        tele_passability: 0,
        location_name: 0xFFFF,
        spawn_rel: -1,
        slot: if m.slot == NEW_SLOT { 0xFFFF } else { m.slot.min(0xFFFE) as u16 },
        flags: Default::default(),
        h4: None,
    }
}

impl App {
    /// The load worker's assets -> the editor scene + the palette panel + the variant's objects.
    /// Returns the placed count.
    pub(crate) fn h3_install_editor_scene(&mut self, assets: H3EditorAssets) -> usize {
        let (canvas, owned) = (assets.canvas.clone(), assets.scenario_owned);
        let es = H3ObjectScene::new(assets, &self.render_state.device, &self.render_state.queue);
        self.h3_scene = Some(Box::new(es));
        self.h3_edit = H3EditState { scenario_owned: owned, ..Default::default() };
        self.h3_fill_palette();
        self.local_objects.clear();
        self.mvar_objects.clear();
        self.mvar_colors.clear();
        self.mvar_meta.clear();
        self.mvar_unresolved.clear();
        self.mvar_labels.clear();
        self.next_local_datum = 0xF000_0000;
        let n = self.h3_render_variant_objects();
        if self.h3_scene.as_ref().and_then(|s| s.variant()).is_none() {
            self.current_variant_path = None;
            self.loaded_map_id = self.selected_map.and_then(|i| self.map_candidates.get(i)).and_then(|c| c.map_id)
                .or_else(|| crate::mapcat::read_map_id(Path::new(&self.map_path)));
            // #h3-scenario No variant open: start from a blank canvas, so the base map's own
            // objects are editor objects (selectable, deletable) and what is drawn is what a save
            // writes. Placing used to adopt a shipped variant silently while the scenery stayed a
            // static, unselectable backdrop.
            if let Some((tpl, v)) = canvas {
                if let Err(e) = self.h3_adopt_variant(&tpl, v, true) { self.spawn_status = format!("Blank canvas: {e}"); }
            }
        }
        n
    }

    /// The map's sandbox palette -> `static_palette*` rows ("Category · name") + `h3_edit.items`.
    pub(crate) fn h3_fill_palette(&mut self) {
        let Some(es) = self.h3_scene.as_ref() else { return };
        let items = palette(es.cache());
        let mut static_pal: Vec<(u32, String)> = Vec::with_capacity(items.len());
        let mut cat: Vec<u32> = Vec::with_capacity(items.len());
        let mut catname: Vec<String> = Vec::with_capacity(items.len());
        for it in &items {
            static_pal.push((h3_tag(it.tag), format!("{} \u{b7} {}", it.category, it.name)));
            cat.push(it.block as u32);
            catname.push(it.category.to_string());
        }
        log::info!("h3 forge palette: {} rows over {} categories", items.len(), super::palette::CATEGORY_NAMES.len());
        self.static_palette = static_pal;
        self.static_palette_cat = cat;
        self.static_palette_catname = catname;
        self.static_palette_variant = vec![0; items.len()];
        self.selected_static_pal = if items.is_empty() { None } else { Some(0) };
        self.h3_edit.items = items;
    }

    /// Palette row whose name contains `needle` (lower-case substring), for `place <name>`.
    pub(crate) fn h3_palette_index_by_raw(&self, needle: &str) -> Option<usize> {
        if !self.h3_active { return None; }
        self.h3_edit.items.iter().position(|it| it.name.to_lowercase().contains(needle))
    }

    /// A loaded set of records -> the shared editor state (`mvar_objects` / `mvar_colors` /
    /// `mvar_meta`) and the Halo 3 state beside it. Replaces whatever was there.
    fn h3_apply_set(&mut self, set: H3EditorSet) {
        let meta: HashMap<u32, ObjMeta> = set.meta.iter().map(|(&d, m)| {
            let pos = set.objects.iter().find(|o| o.datum == d).map(|o| o.pos).or_else(|| set.src.get(&d).map(|r| r.pos)).unwrap_or_default();
            (d, obj_meta_from_h3(m, pos))
        }).collect();
        self.mvar_objects = set.objects;
        self.mvar_colors = set.colors;
        self.mvar_meta = meta;
        self.mvar_unresolved.clear();
        self.h3_edit.src = set.src;
        self.h3_edit.kept = set.kept;
        self.h3_edit.scn = set.scn;
    }

    /// The open variant's records -> editor objects (the Halo 3 `render_variant_objects`).
    pub(crate) fn h3_render_variant_objects(&mut self) -> usize {
        let Some((path, v)) = self.h3_scene.as_ref().and_then(|s| s.variant()).map(|(p, v)| (p.clone(), v.clone())) else { return 0 };
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        self.variant_title = v.header.name.clone();
        self.variant_description = v.header.description.clone();
        self.variant_author = v.header.author.clone();
        self.variant_editor = String::new();
        self.variant_header_dirty = false;
        self.local_objects.clear();
        self.clear_selection_state();
        let t0 = std::time::Instant::now();
        let Some(es) = self.h3_scene.as_ref() else { return 0 };
        let set = variant_to_editor(es.cache(), &self.h3_edit.items, &v);
        let placed = set.objects.iter().filter(|o| !is_scenario_datum(o.datum)).count();
        let (marker_only, unresolved) = (set.marker_only, set.unresolved);
        let (own, deleted) = (set.objects.len() - placed, set.scn.len() - (set.objects.len() - placed));
        self.h3_apply_set(set);
        self.h3_edit.placed_rows.clear();
        self.apply_pending_obj_flags();
        self.current_variant_path = Some(path.clone());
        self.new_variant_template = None;
        self.loaded_map_id = Some(v.map_id);
        self.variant_retry = None;
        crate::mapcat::save_last_variant(&path.to_string_lossy());
        self.note_recent_variant(&path); // File ▸ Open recent (main.rs / recent.rs)
        let own_note = if self.h3_edit.scn.is_empty() { String::new() } else {
            format!(" plus the map's own {own} objects ({deleted} deleted by this variant), all editable")
        };
        self.spawn_status = format!("{name}: {placed} Halo 3 forge objects{own_note} ({marker_only} marker-only, {unresolved} unresolved - kept as read) in {} ms.",
            t0.elapsed().as_millis());
        log::info!("{}", self.spawn_status);
        if placed > 0 { self.rebuild_overlays(); }
        placed
    }

    /// Place palette row `i` at `pos`: an editor object now, its `.mvar` record (palette.rs
    /// `instantiate`, quota appended when the table lacks it) at save time.
    pub(crate) fn h3_place_palette_item(&mut self, i: usize, pos: glam::Vec3) -> Result<(u32, String), String> {
        let it = self.h3_edit.items.get(i).cloned().ok_or_else(|| format!("palette index {i} out of range (0..{})", self.h3_edit.items.len()))?;
        if self.h3_scene.is_none() { return Err("no Halo 3 map is active".into()); }
        // #h3-place A placement has to have a variant to SAVE into (the Halo 3 encoder writes
        // into a copy of a real file). Opening a Halo 3 MAP leaves none open, and this used to
        // refuse every placement outright - viewport click, Line / Fill and palette
        // drag-and-drop alike - so a Halo 3 map could not be built on at all unless one of its
        // variants was opened first, and a drag that was refused showed nothing but a status
        // line. Adopt one of the map's own variants as a blank canvas instead (what File > New
        // variant does), so the first drop just works.
        if self.current_variant_path.is_none() && self.new_variant_template.is_none() {
            let tpl = self.h3_find_template().ok_or(
                "no Halo 3 .mvar for this map was found to save into - open one of its variants (File > Open .mvar) first")?;
            self.h3_adopt_template(&tpl, false)?;
        }
        self.h3_placement_limits(&it)?;
        let row_name = self.static_palette.get(i).map(|(_, n)| n.clone()).unwrap_or_else(|| it.name.clone());
        let datum = self.next_dup_datum;
        self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
        let mode_tag = it.mode_tag.map(h3_tag).unwrap_or(H3_MARKER_TAG);
        self.mvar_objects.push(ObjectInfo {
            datum, type_sig: 0, sig0: 0, sig1: 0, pos: pos.to_array(), health: 1.0, shield: 1.0,
            mode_tag, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: h3_tag(it.tag), variant_name_sid: 0,
        });
        let m = super::edit::new_placement_meta(&it);
        let om = obj_meta_from_h3(&m, pos.to_array());
        self.mvar_colors.insert(datum, (om.team, 0xFF));
        self.mvar_meta.insert(datum, om);
        self.h3_edit.placed_rows.insert(datum, i);
        if let Some(s) = self.objscene_mut() { s.invalidate(); }
        Ok((datum, row_name))
    }

    /// #h3-place A shipped `.mvar` of the LOADED map to start a new Halo 3 variant from. The
    /// Halo 3 encoder writes into a COPY of a real variant file, so a new variant needs one of
    /// this map's own variants as its canvas: it carries the Forge bounds, the budget and the
    /// scenario-owned slots. Candidates come from the halo3 `map_variants` /
    /// `hopper_map_variants` folders (plus the last-opened variant's folder); the smallest file
    /// wins, since any of them is the same canvas.
    pub(crate) fn h3_find_template(&self) -> Option<std::path::PathBuf> {
        let dir = self.current_variant_path.as_ref().and_then(|p| p.parent());
        super::scenario::find_template(self.loaded_map_id?, dir)
    }

    /// #h3-place Adopt `tpl` as the variant a save writes into: its bounds, budget and quota table
    /// become this session's canvas, while its Forge-PLACED objects are dropped (this is a blank
    /// canvas, not an open variant). `fresh` also clears whatever is in the editor, which is File >
    /// New variant; otherwise objects already placed stay put.
    pub(crate) fn h3_adopt_template(&mut self, tpl: &Path, fresh: bool) -> Result<(), String> {
        let v = parse_h3_variant(tpl).map_err(|e| format!("cannot read the template {}: {e}", tpl.display()))?;
        self.h3_adopt_variant(tpl, v, fresh)
    }

    /// `h3_adopt_template` over an already-parsed template. #h3-scenario The base map's own
    /// objects come with the canvas as EDITOR objects, as the scenario made them (nothing deleted
    /// and nothing moved, whatever the template's author did): selectable, deletable, movable, and
    /// written back by a save exactly as edited.
    pub(crate) fn h3_adopt_variant(&mut self, tpl: &Path, v: H3Variant, fresh: bool) -> Result<(), String> {
        if fresh {
            self.mvar_objects.clear();
            self.local_objects.clear();
            self.clear_selection_state();
            self.mvar_colors.clear();
            self.mvar_meta.clear();
            self.mvar_unresolved.clear();
            self.h3_edit.src.clear();
            self.h3_edit.placed_rows.clear();
            self.h3_edit.scn.clear();
        }
        // a new variant is NOT the canvas file: give it its own header, so a save cannot ship
        // under the canvas's name and author. (Adopting only ever happens with no variant open,
        // so there is no user-typed header here to overwrite.)
        self.variant_title = "New variant".into();
        self.variant_description = String::new();
        if self.variant_author.trim().is_empty() { self.variant_author = "HMS".into(); }
        self.variant_editor = self.variant_author.clone();
        self.variant_header_dirty = true;
        // the canvas's scenario-owned slots: editor objects when the load left them out of the
        // static scenery (and this map's variants register them), else they ride through a save
        // exactly as a loaded variant's do
        let canvas = super::scenario::canvas_of(&v);
        let own = match self.h3_scene.as_ref() {
            Some(es) if self.h3_edit.scenario_owned && self.h3_edit.scn.is_empty() => Some(variant_to_editor(es.cache(), &self.h3_edit.items, &canvas)),
            _ => None,
        };
        match own {
            Some(set) => {
                let (n, kept) = (set.objects.len(), set.kept.clone());
                let base = std::mem::take(&mut self.h3_edit.src);
                let (objs, colors, metas) = (std::mem::take(&mut self.mvar_objects), std::mem::take(&mut self.mvar_colors), std::mem::take(&mut self.mvar_meta));
                self.h3_apply_set(set);
                // objects the session already holds (a non-fresh adopt) stay beside the map's own
                self.mvar_objects.extend(objs);
                self.mvar_colors.extend(colors);
                self.mvar_meta.extend(metas);
                self.h3_edit.src.extend(base);
                self.h3_edit.kept = kept;
                log::info!("h3 canvas: {n} base-map objects are editor objects");
            }
            None if !self.h3_edit.scenario_owned || self.h3_edit.scn.is_empty() => {
                self.h3_edit.kept = v.objects.iter().filter(|o| !o.placed).cloned().collect();
            }
            None => {}
        }
        self.loaded_map_id = Some(v.map_id);
        self.current_variant_path = None;
        self.new_variant_template = Some(tpl.to_path_buf());
        // the scene's open variant supplies the bounds / budget / spawn camera the save gates
        // read (the Halo 4 `h4_new_variant` does the same)
        if let Some(s) = self.h3_scene.as_mut() { s.set_variant(Some((tpl.to_path_buf(), v))); }
        if let Some(s) = self.objscene_mut() { s.invalidate(); }
        self.rebuild_overlays();
        let (alive, deleted) = self.h3_scenario_counts();
        if alive + deleted > 0 {
            self.spawn_status = format!("Blank Halo 3 canvas: the map's own {alive} objects (vehicles, weapons, spawn points, scenery...) are in the editor - select, move or delete them like any other. {} Forge slots free.",
                self.h3_slot_budget().map_or(0, |(used, cap)| cap.saturating_sub(used)));
        }
        Ok(())
    }

    /// #h3-place File > New variant on a Halo 3 map: a blank canvas over one of the map's own
    /// shipped variants, ready to place into and Save As.
    pub(crate) fn h3_new_variant(&mut self) {
        let Some(tpl) = self.h3_find_template() else {
            self.spawn_status = match self.loaded_map_id {
                Some(id) => format!("New variant: no Halo 3 .mvar for map id {id} was found to use as a canvas (open any variant of this map once, or put one in halo3\\map_variants)."),
                None => "New variant: load a Halo 3 base map first.".into(),
            };
            return;
        };
        match self.h3_adopt_template(&tpl, true) {
            Ok(()) => self.spawn_status = format!("New Halo 3 variant (canvas {}): drag palette objects into the viewport, then File > Save As into halo3\\map_variants.",
                tpl.file_name().unwrap_or_default().to_string_lossy()),
            Err(e) => self.spawn_status = format!("New variant: {e}"),
        }
    }

    /// A duplicate's own record / row: a copy of its template's with a fresh slot (no-op unless
    /// a Halo 3 map is active). Called by the dispatch in `spawn_copy_of`. A copy of one of the
    /// base map's own objects is a NEW Forge object: it takes the placed form of the original (its
    /// scenario pose and multiplayer properties), Forge flags and a slot past the scenario block;
    /// a base-map object with no quota (scenery the palette cannot spawn) has nothing to be copied
    /// as, so its copy stays an unsaveable editor object and the save check says so.
    pub(crate) fn h3_copy_source_record(&mut self, from: u32, to: u32) {
        if !self.h3_active { return; }
        let from_scn = self.h3_edit.scn.iter().find(|e| e.datum == from).cloned();
        if let Some(e) = from_scn {
            if e.rec.quota < 0 { return; }
            let mut r = if e.rec.placed { e.rec.clone() } else { e.promote.clone() };
            r.slot = NEW_SLOT;
            r.parent = None;
            r.flags = if e.rec.flags & 0x80 != 0 { super::palette::PLACED_FLAGS } else { super::palette::PLACED_FLAGS & !0x80 };
            self.h3_edit.src.insert(to, r);
            if let Some(m) = self.mvar_meta.get_mut(&to) { m.slot = 0xFFFF; m.spawn_rel = -1; }
        } else if let Some(mut r) = self.h3_edit.src.get(&from).cloned() {
            r.slot = NEW_SLOT;
            r.parent = None;
            self.h3_edit.src.insert(to, r);
            if let Some(m) = self.mvar_meta.get_mut(&to) { m.slot = 0xFFFF; m.spawn_rel = -1; }
        } else if let Some(&row) = self.h3_edit.placed_rows.get(&from) {
            self.h3_edit.placed_rows.insert(to, row);
        }
    }

    /// Properties-panel rows for a Halo 3 object: its multiplayer object type (read-only; the
    /// palette row dictates it) and the one shared byte under the name its type gives it.
    pub(crate) fn h3_rows_type_spawn(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        ui.label("object type");
        ui.label(format!("{} ({})", h3_type_name(m.cached_type), m.cached_type))
            .on_hover_text("Multiplayer object type - dictated by the palette row's object (read-only)");
        ui.end_row();
        let role = StorageRole::of(m.cached_type);
        if role == StorageRole::None { return; }
        ui.label(role.label());
        match role {
            StorageRole::SpareClips => {
                ui.add(egui::DragValue::new(&mut m.weapon_clips).range(0..=255))
                    .on_hover_text("The Forge menu's 'spare clips': how many spare magazines the weapon spawns with");
            }
            StorageRole::TeleporterChannel => {
                ui.add(egui::DragValue::new(&mut m.tele_channel).range(0..=255))
                    .on_hover_text("The Forge menu's 'teleporter channel': a sender / 2-way links to the receivers on the SAME channel");
            }
            StorageRole::SpawnOrder => {
                ui.add(egui::DragValue::new(&mut m.spawn_seq).range(0..=255))
                    .on_hover_text("The Forge menu's 'spawn order': hill / territory / flag-spawn sequence number (255 = none, as the stock maps store it for plain spawn points)");
            }
            StorageRole::None => {}
        }
        ui.end_row();
    }

    /// The Forge menu's `symmetry` property: which gametype symmetry the object appears in. The two
    /// bits are the pair a gametype needs one object of each for (halo3.dll sub_1800BC2BC); "both"
    /// (`0x0C`) is what a fresh drop gets.
    pub(crate) fn h3_row_symmetry(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        ui.label("symmetry").on_hover_text("Which gametype symmetry the object is used in. Bit 0x04 is called symmetric and 0x08 asymmetric (the order the Forge menu lists them -- the engine only needs one object of each, so the naming is inferred). 'both' is what a fresh drop gets.");
        let mut sym = m.placement & MP_SYMMETRY_MASK;
        let txt = match sym { 0x0C => "both", 0x04 => "symmetric", 0x08 => "asymmetric", _ => "never" };
        egui::ComboBox::from_id_salt("h3-sym").selected_text(txt).show_ui(ui, |ui| {
            ui.selectable_value(&mut sym, 0u8, "never");
            ui.selectable_value(&mut sym, 0x04u8, "symmetric");
            ui.selectable_value(&mut sym, 0x08u8, "asymmetric");
            ui.selectable_value(&mut sym, 0x0Cu8, "both");
        });
        m.placement = (m.placement & !MP_SYMMETRY_MASK) | sym;
        ui.end_row();
    }

    /// Boundary rows for a Halo 3 object: the shape and its sizes in world units (the record keeps
    /// 16-bit quanta over [0, 60] wu, so a size snaps to the nearest 0.001 wu).
    pub(crate) fn h3_rows_shape(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        ui.label("boundary");
        let shapes = ["none", "sphere", "cylinder", "box"];
        egui::ComboBox::from_id_salt("h3-shape").selected_text(shapes[(m.boundary_shape as usize).min(3)]).show_ui(ui, |ui| {
            for (i, s) in shapes.iter().enumerate() { ui.selectable_value(&mut m.boundary_shape, i as u8, *s); }
        });
        ui.end_row();
        let names: &[&str] = match m.boundary_shape { 1 => &["radius"], 2 => &["radius", "top", "bottom"], 3 => &["width", "length", "top", "bottom"], _ => &[] };
        if names.is_empty() { return; }
        ui.label("size (wu)");
        ui.horizontal(|ui| {
            for (k, n) in names.iter().enumerate() {
                let mut wu = super::edit::boundary_wu(m.boundary[k]);
                if ui.add(egui::DragValue::new(&mut wu).range(0.0..=60.0).speed(0.05).max_decimals(3).prefix(format!("{n} "))).changed() {
                    m.boundary[k] = super::edit::boundary_q_of(wu);
                }
            }
        });
        ui.end_row();
    }

    /// (spent, maximum) of the open Halo 3 variant as the editor stands: the price of every Forge
    /// object (loaded ones by their record's quota, new placements by their palette row) plus the
    /// kept placed records, against the variant's maximum budget. The same rule as the save check.
    /// The base map's own objects are free.
    pub(crate) fn h3_live_budget(&self) -> Option<(f32, f32)> {
        if !self.h3_active { return None; }
        let (_, v) = self.h3_scene.as_ref()?.variant()?;
        let price = |q: i32| usize::try_from(q).ok().and_then(|i| v.quotas.get(i)).map_or(0.0, |q| q.price);
        let mut spent = 0.0f32;
        for o in self.mvar_objects.iter().chain(self.local_objects.iter()).filter(|o| !is_scenario_datum(o.datum)) {
            spent += match self.h3_edit.src.get(&o.datum) {
                Some(r) => price(r.quota),
                None => self.h3_edit.placed_rows.get(&o.datum).and_then(|&row| self.h3_edit.items.get(row)).map_or(0.0, |it| it.price),
            };
        }
        spent += self.h3_edit.kept.iter().filter(|r| r.placed && r.flags & SLOT_SCENARIO == 0).map(|r| price(r.quota)).sum::<f32>();
        Some((spent, v.maximum_budget))
    }

    /// (used, capacity) of the slots a Forge placement can take on this Halo 3 map: the variant
    /// holds 640 and the scenario's own objects own the first `num_scenario_objects` of them (0 to
    /// 511; 191 to 511 on the maps that have any), so the Forge-placeable capacity is what is
    /// left. A new object takes the first free slot AT OR PAST that block (halo3.dll
    /// sub_1800BBBD0), so deleting a base-map object frees none. `used` counts the editor's Forge
    /// objects plus the kept records that sit past the scenario block (placed records HMS could not
    /// resolve still occupy a slot); the base map's own objects are not in it.
    pub(crate) fn h3_slot_budget(&self) -> Option<(usize, usize)> {
        if !self.h3_active { return None; }
        let scenario = self.h3_scene.as_ref().and_then(|s| s.variant()).map(|(_, v)| v.num_scenario_objects as usize)?;
        Some(super::edit::forge_slots_used(scenario, self.mvar_objects.iter().chain(self.local_objects.iter()).map(|o| o.datum), &self.h3_edit.kept))
    }

    /// (alive, deleted) base-map objects the editor holds: alive ones are editor objects, a deleted
    /// one is only a record the save writes back flagged.
    pub(crate) fn h3_scenario_counts(&self) -> (usize, usize) {
        let alive = self.mvar_objects.iter().chain(self.local_objects.iter()).filter(|o| is_scenario_datum(o.datum)).count();
        (alive, self.h3_edit.scn.len().saturating_sub(alive))
    }

    /// Forge's own refusals at placement (edit.rs `placement_refusal`): no free object slot, the
    /// palette row's `maximum_allowed` reached, the budget spent. The message names which limit.
    pub(crate) fn h3_placement_limits(&self, it: &H3PaletteItem) -> Result<(), String> {
        let (_, v) = self.h3_scene.as_ref().and_then(|s| s.variant()).ok_or("no Halo 3 variant is open")?;
        let (used, _) = self.h3_slot_budget().ok_or("no Halo 3 variant is open")?;
        // every live object of this palette row, loaded (by its record's quota) or new (by its row)
        let def_of = |o: &ObjectInfo| -> Option<u32> {
            match self.h3_edit.src.get(&o.datum) {
                Some(r) => usize::try_from(r.quota).ok().and_then(|q| v.quotas.get(q)).map(|q| q.definition),
                None => self.h3_edit.placed_rows.get(&o.datum).and_then(|&row| self.h3_edit.items.get(row)).map(|r| r.definition),
            }
        };
        let same_row = self.mvar_objects.iter().chain(self.local_objects.iter()).filter(|o| def_of(o) == Some(it.definition)).count();
        let (spent, max) = self.h3_live_budget().unwrap_or((0.0, 0.0));
        match super::edit::placement_refusal(it, v.num_scenario_objects as usize, used, same_row, spent, max) {
            Some(msg) => Err(msg),
            None => Ok(()),
        }
    }

    /// Everything a Halo 3 save writes, built WITHOUT touching the disk: the source file, the
    /// variant to encode (quota table grown, the quota counts and budget re-accounted), the record
    /// list and what the save could not write. The save gate and the save itself both read it.
    pub(crate) fn h3_build_save(&self) -> Result<H3Built, String> {
        let src = self.current_variant_path.clone().or_else(|| self.new_variant_template.clone()).ok_or("no Halo 3 variant is open")?;
        if !super::mvar::is_h3_variant(&src) { return Err(format!("{} is not a Halo 3 variant", src.display())); }
        let v = parse_h3_variant(&src).map_err(|e| format!("cannot re-read the source variant: {e}"))?;
        let meta: HashMap<u32, H3LiveMeta> = self.mvar_objects.iter().chain(self.local_objects.iter()).filter_map(|o| {
            let m = self.mvar_meta.get(&o.datum)?;
            let team = self.mvar_colors.get(&o.datum).map(|(t, _)| *t).unwrap_or(m.team);
            Some((o.datum, live_meta_of(m, team, self.h3_edit.placed_rows.get(&o.datum).copied())))
        }).collect();
        // a blank canvas: the template's Forge objects were dropped, so its stored counts and
        // budget describe objects that are gone and are recounted rather than adjusted
        let fresh = self.current_variant_path.is_none() && self.new_variant_template.is_some();
        let (v2, list, report) = build_h3_save_list(&v, &self.h3_edit.items, &self.h3_edit.src, &self.mvar_objects, &self.local_objects, &self.h3_edit.kept, &self.h3_edit.scn, &meta, fresh);
        Ok(H3Built { src, source: v, v: v2, list, report })
    }

    /// File > Save / Save As for a Halo 3 variant: the record list from the editor state
    /// (edit.rs `build_h3_save_list`) -> mvar.rs `save_h3_variant` into a copy of the source file,
    /// the header strings from the Variant Info panel when edited. Returns (written, not written).
    pub(crate) fn h3_save_variant_to(&self, dst: &Path) -> Result<(usize, usize), String> {
        let H3Built { src, source: v, v: v2, list, report } = self.h3_build_save()?;
        let edits = H3HeaderEdits {
            name: (self.variant_title != v.header.name).then(|| self.variant_title.clone()),
            description: (self.variant_description != v.header.description).then(|| self.variant_description.clone()),
            author: (self.variant_author != v.header.author).then(|| self.variant_author.clone()),
        };
        let edited = edits.name.is_some() || edits.description.is_some() || edits.author.is_some();
        save_h3_variant(&src, dst, &v2, &list, edited.then_some(&edits)).map_err(|e| e.to_string())?;
        let mut note = String::new();
        if !report.out_of_bounds.is_empty() { note.push_str(&format!(" {} object(s) sat outside the variant bounds and were CLAMPED to the edge.", report.out_of_bounds.len())); }
        if report.over_slot_cap > 0 { note.push_str(&format!(" {} object(s) had no free slot (room for {} Forge objects) and were NOT written.", report.over_slot_cap, super::mvar::H3_SLOTS - v.num_scenario_objects as usize)); }
        if report.unsaveable > 0 { note.push_str(&format!(" {} object(s) had no palette row and were NOT written.", report.unsaveable)); }
        if report.quotas_added > 0 { note.push_str(&format!(" {} quota(s) added to the table.", report.quotas_added)); }
        if report.scenario_deleted > 0 { note.push_str(&format!(" {} of the map's own object(s) deleted.", report.scenario_deleted)); }
        if report.scenario_moved > 0 { note.push_str(&format!(" {} of the map's own object(s) moved or edited.", report.scenario_moved)); }
        if !report.over_quota.is_empty() { note.push_str(&format!(" {} quota(s) are over their maximum.", report.over_quota.len())); }
        if let Some((spent, max)) = report.over_budget { note.push_str(&format!(" Over budget ({spent:.0} / {max:.0}).")); }
        if dst.parent().and_then(|d| d.file_name()).map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("hopper_map_variants")) {
            note.push_str(" Saved into hopper_map_variants: MCC's Custom Games list reads map_variants, so it will not appear there.");
        }
        *self.h3_edit.last_save_note.borrow_mut() = note.clone();
        let written = list.iter().filter(|r| r.placed).count();
        log::info!("h3 mvar: saved {written} placed objects ({} records) -> {}{}{}", list.len(), dst.display(), if edited { " (header edited)" } else { "" }, note);
        Ok((written, report.over_slot_cap + report.unsaveable))
    }

    /// The Halo 3 save check: what a save is about to do that the user should hear about FIRST
    /// (objects past the bounds get clamped, past the slot cap or without a palette row get
    /// dropped, a quota over its maximum, a budget overspent). `true` = the save is held and the
    /// window decides. Mirrors `h4_save_gate`.
    pub(crate) fn h3_save_gate(&mut self, action: BspWarnAction) -> bool {
        if !self.h3_active { return false; }
        if self.h3_edit.save_skip_once {
            self.h3_edit.save_skip_once = false;
            return false;
        }
        match self.h3_build_save() {
            Ok(b) => {
                let gate = H3SaveGate::from_built(&b, &self.h3_edit.items);
                if gate.is_clean() { false } else { self.h3_edit.save_pending = Some((action, gate)); true }
            }
            Err(e) => { self.spawn_status = format!("Save: {e}"); true }
        }
    }

    /// The Halo 3 save-check window: double-click an out-of-bounds row to select the object and fly
    /// to it; "Save anyway" proceeds (clamping / dropping as listed), "Cancel" drops the save.
    pub(crate) fn h3_save_window_ui(&mut self, ctx: &egui::Context) {
        let Some((action, gate)) = self.h3_edit.save_pending.clone() else { return };
        let bounds = self.h3_scene.as_ref().and_then(|s| s.variant()).map(|(_, v)| v.bounds);
        // rows follow the live positions: an object moved back inside drops off the list
        let oob: Vec<u32> = gate.out_of_bounds.iter().copied().filter(|d| {
            match (bounds, self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == *d)) {
                (Some(b), Some(o)) => o.pos[0] < b[0] || o.pos[0] > b[1] || o.pos[1] < b[2] || o.pos[1] > b[3] || o.pos[2] < b[4] || o.pos[2] > b[5],
                _ => false,
            }
        }).collect();
        let (mut save_anyway, mut cancel, mut win_open) = (false, false, true);
        let cap = self.h3_slot_budget().map_or(super::mvar::H3_SLOTS, |(_, cap)| cap);
        let mut goto: Option<u32> = None;
        let orange = egui::Color32::from_rgb(255, 160, 60);
        let red = egui::Color32::from_rgb(255, 90, 70);
        egui::Window::new("Halo 3 save check")
            .open(&mut win_open)
            .default_size([560.0, 320.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.screen_rect().center())
            .collapsible(false)
            .resizable(true)
            .show(ctx, |ui| {
                ui.separator();
                if !oob.is_empty() {
                    ui.label(egui::RichText::new(format!("{} object(s) are outside the variant bounds.", oob.len())).color(orange).strong());
                    if let Some(b) = bounds {
                        ui.small(format!("Bounds x [{:.1}, {:.1}]  y [{:.1}, {:.1}]  z [{:.1}, {:.1}]. A position outside them cannot be encoded: saving CLAMPS these to the nearest edge. Double-click a row to select the object and fly to it.", b[0], b[1], b[2], b[3], b[4], b[5]));
                    }
                    let row_h = ui.spacing().interact_size.y;
                    egui::ScrollArea::vertical().id_salt("h3-oob").max_height(150.0).auto_shrink([false, false]).show_rows(ui, row_h, oob.len(), |ui, range| {
                        for k in range {
                            let d = oob[k];
                            let name = self.mvar_meta.get(&d).map(|m| m.name.clone()).unwrap_or_default();
                            let pos = self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == d).map(|o| o.pos).unwrap_or_default();
                            let text = format!("{}  ({:.1}, {:.1}, {:.1})", crate::prettify_stringid(&name), pos[0], pos[1], pos[2]);
                            let r = ui.add_sized([ui.available_width(), row_h], egui::SelectableLabel::new(self.selected_set.contains(&d), text));
                            if r.double_clicked() { goto = Some(d); }
                        }
                    });
                }
                for (name, n, allowed) in &gate.over_quota {
                    ui.label(egui::RichText::new(format!("{name}: {n} placed, the palette allows {allowed}.")).color(orange))
                        .on_hover_text("Forge refuses a placement past this limit; 328 of the 333 shipped variants respect it.");
                }
                if let Some((spent, max)) = gate.over_budget {
                    ui.label(egui::RichText::new(format!("Over budget: {spent:.0} spent of {max:.0}.")).color(orange))
                        .on_hover_text("Forge stops you placing past the budget; 322 of the 333 shipped variants stay within it (the rest load fine, the game just stores the cap).");
                }
                if gate.no_slot > 0 {
                    ui.label(egui::RichText::new(format!("{} object(s) have no free slot and will NOT be written: this map variant has room for {} Forge objects ({} slots minus the map's own {}).", gate.no_slot, cap, super::mvar::H3_SLOTS, super::mvar::H3_SLOTS - cap)).color(red));
                }
                if gate.unsaveable > 0 {
                    ui.label(egui::RichText::new(format!("{} object(s) have no palette row and will NOT be written.", gate.unsaveable)).color(red));
                }
                ui.separator();
                ui.horizontal(|ui| {
                    let verb = match action { BspWarnAction::Save => "Save anyway", BspWarnAction::SaveAs => "Save As anyway" };
                    if ui.button(verb).clicked() { save_anyway = true; }
                    if ui.button("Cancel").clicked() { cancel = true; }
                });
            });
        if let Some(d) = goto {
            self.selected_set = vec![d];
            self.selected_datum = Some(d);
            self.obj_list_anchor = Some(d);
            self.obj_list_last_sel = Some(d);
            self.apply_selection_highlight();
            self.frame_selected();
            self.h3_edit.save_pending = None;
            return;
        }
        if save_anyway {
            self.h3_edit.save_pending = None;
            self.h3_edit.save_skip_once = true;
            match action {
                BspWarnAction::Save => self.save_current_variant(),
                BspWarnAction::SaveAs => self.save_variant_as(),
            }
            self.h3_edit.save_skip_once = false;
            return;
        }
        if cancel || !win_open {
            self.h3_edit.save_pending = None;
            self.status = "Save cancelled.".into();
        }
    }

    /// After a save the placed rows have records on disk: re-read the file so the editor's
    /// source records (slots included) match it, keeping the live objects' datums.
    pub(crate) fn h3_resync_after_save(&mut self, path: &Path) {
        let Ok(v) = parse_h3_variant(path) else { return };
        let Some(es) = self.h3_scene.as_mut() else { return };
        es.set_variant(Some((path.to_path_buf(), v.clone())));
        // match live objects to records by slot (loaded) or by position (new placements); the base
        // map's own objects follow their slot: the file's record (deleted ones included) is theirs now
        let scn_slots: std::collections::HashSet<usize> = self.h3_edit.scn.iter().map(|e| e.rec.slot).collect();
        let mut by_slot: HashMap<usize, &H3VariantObject> = v.objects.iter().filter(|o| o.placed && !scn_slots.contains(&o.slot)).map(|o| (o.slot, o)).collect();
        let mut kept: Vec<H3VariantObject> = v.objects.iter().filter(|o| !o.placed && !scn_slots.contains(&o.slot)).cloned().collect();
        let mut src: HashMap<u32, H3VariantObject> = HashMap::new();
        for e in self.h3_edit.scn.iter_mut() {
            if let Some(r) = v.objects.iter().find(|o| o.slot == e.rec.slot) { e.rec = r.clone(); }
            if let Some(m) = self.mvar_meta.get_mut(&e.datum) { m.slot = e.rec.slot.min(0xFFFE) as u16; }
            if self.mvar_objects.iter().chain(self.local_objects.iter()).any(|o| o.datum == e.datum) { src.insert(e.datum, e.rec.clone()); }
        }
        let datums: Vec<u32> = self.mvar_objects.iter().chain(self.local_objects.iter()).map(|o| o.datum).filter(|d| !is_scenario_datum(*d)).collect();
        for d in &datums {
            let slot = self.h3_edit.src.get(d).map(|r| r.slot);
            let rec = match slot.and_then(|s| by_slot.remove(&s)) {
                Some(r) => Some(r.clone()),
                None => {
                    let pos = self.mvar_objects.iter().chain(self.local_objects.iter()).find(|o| o.datum == *d).map(|o| o.pos).unwrap_or_default();
                    let hit = by_slot.iter().min_by(|(_, a), (_, b)| dist2(a.pos, pos).partial_cmp(&dist2(b.pos, pos)).unwrap_or(std::cmp::Ordering::Equal)).map(|(s, _)| *s);
                    hit.filter(|s| dist2(by_slot[s].pos, pos) < 0.01).and_then(|s| by_slot.remove(&s)).cloned()
                }
            };
            if let Some(r) = rec {
                if let Some(m) = self.mvar_meta.get_mut(d) { m.slot = r.slot.min(0xFFFE) as u16; m.folder = r.quota.clamp(0, 0xFFFE) as u16; }
                src.insert(*d, r);
                self.h3_edit.placed_rows.remove(d);
            }
        }
        kept.extend(by_slot.into_values().cloned());
        self.h3_edit.src = src;
        self.h3_edit.kept = kept;
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 { (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum() }

/// The multiplayer object type's name (the tag enum; confirmed against the palette names).
pub fn h3_type_name(t: u8) -> &'static str {
    const N: [&str; 24] = ["ordinary", "weapon", "grenade", "projectile", "powerup", "equipment", "light land vehicle", "heavy land vehicle",
        "flying vehicle", "teleporter (2-way)", "teleporter sender", "teleporter receiver", "player spawn", "respawn zone", "oddball spawn",
        "flag / bomb spawn", "hold destination", "flag return / goal area", "hill", "infection haven", "territory", "VIP boundary",
        "VIP destination", "juggernaut destination"];
    N.get(t as usize).copied().unwrap_or("unknown")
}

/// Halo 3 records carry NO size field (`shared_storage` is spare clips / a channel / a spawn order,
/// nothing else), so scaling a selection can move its objects but never resize them. The note the
/// scale commit appends on a Halo 3 map instead of the Reach / Halo 4 "sizes land within" report.
pub fn scale_unsupported_note(h3_active: bool) -> Option<&'static str> {
    h3_active.then_some(" -- Halo 3 has no per-object size, so the objects were moved and not resized")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h3::mvar::H3Quota;

    /// Each object type sends the panel field its role names to the record's shared byte, and a
    /// type with no role sends nothing (its byte rides through as read).
    #[test]
    fn each_object_type_edits_the_field_its_role_names() {
        for t in 0u8..24 {
            let om = ObjMeta { cached_type: t, weapon_clips: 11, tele_channel: 22, spawn_seq: 33, placement: 0x0C, ..Default::default() };
            let live = live_meta_of(&om, 8, None);
            let want = match t { 1 | 2 => Some(11), 9..=11 => Some(22), 12..=23 => Some(33), _ => None };
            assert_eq!(live.shared_storage, want, "object type {t}");
            assert_eq!(live.symmetry, Some(0x0C), "object type {t}");
        }
        // the panel's spawn order is a signed int: out of byte range clamps instead of wrapping
        let big = ObjMeta { cached_type: 18, spawn_seq: 300, placement: 0x04, ..Default::default() };
        assert_eq!(live_meta_of(&big, 8, None).shared_storage, Some(255));
        let neg = ObjMeta { cached_type: 18, spawn_seq: -4, placement: 0x04, ..Default::default() };
        assert_eq!(live_meta_of(&neg, 8, None).shared_storage, Some(0));
        // only the symmetry bits leave the placement byte
        let sym = ObjMeta { cached_type: 0, placement: 0xFF, ..Default::default() };
        assert_eq!(live_meta_of(&sym, 8, None).symmetry, Some(0x0C));
    }

    /// The record's bytes land in the same fields `live_meta_of` reads them back from: load ->
    /// panel -> save is the identity for every type, so an untouched object is never rewritten.
    #[test]
    fn loading_then_saving_a_record_is_the_identity_for_every_role() {
        for t in [0u8, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22, 23] {
            let m = crate::h3::edit::H3ObjMeta { name: "x".into(), quota: 3, team: 8, object_type: t, spawn_time: 30, boundary_shape: 2,
                boundary_q: [1, 2, 3, 4], slot: 5, shared_storage: 7, mp_flags: 0x8C };
            let om = obj_meta_from_h3(&m, [0.0; 3]);
            let live = live_meta_of(&om, m.team, None);
            match StorageRole::of(t) {
                StorageRole::None => assert_eq!(live.shared_storage, None, "type {t}"),
                _ => assert_eq!(live.shared_storage, Some(7), "type {t}: the shared byte did not survive load -> save"),
            }
            assert_eq!(live.symmetry, Some(0x8C & MP_SYMMETRY_MASK));
            assert_eq!((live.boundary_shape, live.boundary_q, live.spawn_time), (2, [1, 2, 3, 4], 30));
        }
    }

    /// Scaling a selection cannot resize anything on Halo 3 (the record has no size field) and
    /// says so; on every other game the Reach / Halo 4 report runs as before.
    #[test]
    fn scaling_on_halo_3_says_it_only_moves_things() {
        let n = scale_unsupported_note(true).expect("Halo 3 must explain itself");
        assert!(n.contains("moved") && n.contains("not resized"), "{n}");
        assert_eq!(scale_unsupported_note(false), None);
    }

    /// The save check lists the palette NAME of an over-limit quota (and "quota #n" for one with no
    /// palette row), and is clean exactly when nothing is wrong.
    #[test]
    fn the_save_gate_names_its_findings() {
        let item = H3PaletteItem { index: 0, block: 1, entry: 4, definition: (1 + 1) << 16 | 4, tag: 0, mode_tag: None, name: "battle_rifle".into(),
            category: "Weapons", price: 10.0, max_allowed: 8, mp: None };
        let mut v = H3Variant::default();
        v.quotas = vec![H3Quota { definition: item.definition, maximum_allowed: 8, ..Default::default() }, H3Quota { definition: 0x0009_0001, ..Default::default() }];
        let mk = |report: H3SaveReport| H3Built { src: std::path::PathBuf::new(), source: H3Variant::default(), v: v.clone(), list: vec![], report };
        assert!(H3SaveGate::from_built(&mk(H3SaveReport::default()), &[item.clone()]).is_clean());
        let g = H3SaveGate::from_built(&mk(H3SaveReport { over_quota: vec![(0, 9, 8), (1, 3, 2)], ..Default::default() }), &[item.clone()]);
        assert_eq!(g.over_quota, vec![("battle_rifle".to_string(), 9, 8), ("quota #1".to_string(), 3, 2)]);
        assert!(!g.is_clean());
        for r in [
            H3SaveReport { out_of_bounds: vec![1], ..Default::default() },
            H3SaveReport { over_budget: Some((9.0, 5.0)), ..Default::default() },
            H3SaveReport { over_slot_cap: 1, ..Default::default() },
            H3SaveReport { unsaveable: 1, ..Default::default() },
        ] {
            assert!(!H3SaveGate::from_built(&mk(r.clone()), &[item.clone()]).is_clean(), "{r:?} must hold the save");
        }
    }
}
