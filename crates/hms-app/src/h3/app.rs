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

use hms_ipc::ObjectInfo;

use super::edit::{build_h3_save_list, h3_tag, variant_to_editor, H3LiveMeta, H3_MARKER_TAG, NEW_SLOT};
use super::edit_scene::{H3EditorAssets, H3ObjectScene};
use super::mvar::{parse_h3_variant, save_h3_variant, H3HeaderEdits, H3VariantObject};
use super::palette::{palette, H3PaletteItem};
use crate::{App, ObjMeta};

#[derive(Default)]
pub struct H3EditState {
    /// The palette rows, parallel to `App.static_palette`.
    pub items: Vec<H3PaletteItem>,
    /// datum -> the source `.mvar` record (loaded objects; a placed row gets one on its first save).
    pub src: HashMap<u32, H3VariantObject>,
    /// Records a save carries verbatim (scenario-owned slots, unresolved placements).
    pub kept: Vec<H3VariantObject>,
    /// Palette row of every NEW placement (datum -> row), until it is saved.
    pub placed_rows: HashMap<u32, usize>,
    pub last_save_note: RefCell<String>,
}

/// `ObjMeta` of a Halo 3 record: the shared fields under their Reach names (folder = quota,
/// respawn = spawn time, cached_type = multiplayer object type, boundary = the raw 16-bit quanta).
pub fn obj_meta_from_h3(m: &super::edit::H3ObjMeta, pos: [f32; 3]) -> ObjMeta {
    ObjMeta {
        name: m.name.clone(),
        folder: m.quota.clamp(0, 0xFFFE) as u16,
        item: 0,
        pos,
        team: m.team,
        color: -1,
        cached_type: m.object_type,
        spawn_seq: 0,
        respawn: m.spawn_time,
        label_idx: 0xFFFF,
        label: String::new(),
        placement: 0,
        boundary_shape: m.boundary_shape,
        boundary: m.boundary_q,
        weapon_clips: 0,
        tele_channel: 0,
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
        let es = H3ObjectScene::new(assets, &self.render_state.device, &self.render_state.queue);
        self.h3_scene = Some(Box::new(es));
        self.h3_edit = H3EditState::default();
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
        let placed = set.objects.len();
        let meta: HashMap<u32, ObjMeta> = set.meta.iter().map(|(&d, m)| (d, obj_meta_from_h3(m, set.src.get(&d).map(|r| r.pos).unwrap_or_default()))).collect();
        self.mvar_objects = set.objects;
        self.mvar_colors = set.colors;
        self.mvar_meta = meta;
        self.mvar_unresolved.clear();
        self.h3_edit.src = set.src;
        self.h3_edit.kept = set.kept;
        self.h3_edit.placed_rows.clear();
        self.apply_pending_obj_flags();
        self.current_variant_path = Some(path.clone());
        self.new_variant_template = None;
        self.loaded_map_id = Some(v.map_id);
        self.variant_retry = None;
        crate::mapcat::save_last_variant(&path.to_string_lossy());
        self.note_recent_variant(&path); // File ▸ Open recent (main.rs / recent.rs)
        self.spawn_status = format!("{name}: {placed} Halo 3 forge objects ({} marker-only, {} unresolved - kept as read) in {} ms.",
            set.marker_only, set.unresolved, t0.elapsed().as_millis());
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
        let row_name = self.static_palette.get(i).map(|(_, n)| n.clone()).unwrap_or_else(|| it.name.clone());
        let datum = self.next_dup_datum;
        self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
        let mode_tag = it.mode_tag.map(h3_tag).unwrap_or(H3_MARKER_TAG);
        self.mvar_objects.push(ObjectInfo {
            datum, type_sig: 0, sig0: 0, sig1: 0, pos: pos.to_array(), health: 1.0, shield: 1.0,
            mode_tag, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: h3_tag(it.tag), variant_name_sid: 0,
        });
        let mp = it.mp.unwrap_or_default();
        let m = super::edit::H3ObjMeta { name: it.name.clone(), quota: -1, team: super::palette::TEAM_NEUTRAL, object_type: mp.object_type,
            spawn_time: mp.spawn_time.clamp(0, 255) as u8, boundary_shape: mp.boundary_shape, boundary_q: [0; 4], slot: NEW_SLOT };
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
        let map_id = self.loaded_map_id?;
        let mut dirs = crate::mapcat::h3_variant_dirs();
        if let Some(d) = self.current_variant_path.as_ref().and_then(|p| p.parent()) { dirs.push(d.to_path_buf()); }
        let mut cands: Vec<(u64, std::path::PathBuf)> = Vec::new();
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map_or(true, |x| !x.eq_ignore_ascii_case("mvar")) { continue; }
                // the cheap read: the content header alone says which map a variant targets
                if super::mvar::read_h3_map_id(&p) != Some(map_id) { continue; }
                cands.push((e.metadata().map(|m| m.len()).unwrap_or(u64::MAX), p));
            }
        }
        cands.sort();
        cands.into_iter().map(|(_, p)| p).find(|p| parse_h3_variant(p).is_ok())
    }

    /// #h3-place Adopt `tpl` as the variant a save writes into: its bounds, budget and
    /// scenario-owned slots become this session's canvas, while its Forge-PLACED objects are
    /// dropped (this is a blank canvas, not an open variant). `fresh` also clears whatever is
    /// in the editor, which is File > New variant; otherwise objects already placed stay put.
    pub(crate) fn h3_adopt_template(&mut self, tpl: &Path, fresh: bool) -> Result<(), String> {
        let v = parse_h3_variant(tpl).map_err(|e| format!("cannot read the template {}: {e}", tpl.display()))?;
        if fresh {
            self.mvar_objects.clear();
            self.local_objects.clear();
            self.clear_selection_state();
            self.mvar_colors.clear();
            self.mvar_meta.clear();
            self.mvar_unresolved.clear();
            self.h3_edit.src.clear();
            self.h3_edit.placed_rows.clear();
        }
        // a new variant is NOT the canvas file: give it its own header, so a save cannot ship
        // under the canvas's name and author. (Adopting only ever happens with no variant open,
        // so there is no user-typed header here to overwrite.)
        self.variant_title = "New variant".into();
        self.variant_description = String::new();
        if self.variant_author.trim().is_empty() { self.variant_author = "HMS".into(); }
        self.variant_editor = self.variant_author.clone();
        self.variant_header_dirty = true;
        // the canvas's scenario-owned slots ride through a save exactly as a loaded variant's do
        self.h3_edit.kept = v.objects.iter().filter(|o| !o.placed).cloned().collect();
        self.loaded_map_id = Some(v.map_id);
        self.current_variant_path = None;
        self.new_variant_template = Some(tpl.to_path_buf());
        // the scene's open variant supplies the bounds / budget / spawn camera the save gates
        // read (the Halo 4 `h4_new_variant` does the same)
        if let Some(s) = self.h3_scene.as_mut() { s.set_variant(Some((tpl.to_path_buf(), v))); }
        self.rebuild_overlays();
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
    /// a Halo 3 map is active). Called by the dispatch in `spawn_copy_of`.
    pub(crate) fn h3_copy_source_record(&mut self, from: u32, to: u32) {
        if !self.h3_active { return; }
        if let Some(mut r) = self.h3_edit.src.get(&from).cloned() {
            r.slot = NEW_SLOT;
            r.parent = None;
            self.h3_edit.src.insert(to, r);
            if let Some(m) = self.mvar_meta.get_mut(&to) { m.slot = 0xFFFF; m.spawn_rel = -1; }
        } else if let Some(&row) = self.h3_edit.placed_rows.get(&from) {
            self.h3_edit.placed_rows.insert(to, row);
        }
    }

    /// File > Save / Save As for a Halo 3 variant: the record list from the editor state
    /// (edit.rs `build_h3_save_list`) -> mvar.rs `save_h3_variant` into a copy of the source file,
    /// the header strings from the Variant Info panel when edited. Returns (written, not written).
    pub(crate) fn h3_save_variant_to(&self, dst: &Path) -> Result<(usize, usize), String> {
        let src = self.current_variant_path.clone().or_else(|| self.new_variant_template.clone()).ok_or("no Halo 3 variant is open")?;
        if !super::mvar::is_h3_variant(&src) { return Err(format!("{} is not a Halo 3 variant", src.display())); }
        let v = parse_h3_variant(&src).map_err(|e| format!("cannot re-read the source variant: {e}"))?;
        let meta: HashMap<u32, H3LiveMeta> = self.mvar_objects.iter().chain(self.local_objects.iter()).filter_map(|o| {
            let m = self.mvar_meta.get(&o.datum)?;
            let team = self.mvar_colors.get(&o.datum).map(|(t, _)| *t).unwrap_or(m.team);
            Some((o.datum, H3LiveMeta { team, spawn_time: m.respawn, boundary_shape: m.boundary_shape, boundary_q: m.boundary, palette_item: self.h3_edit.placed_rows.get(&o.datum).copied() }))
        }).collect();
        let (v2, list, report) = build_h3_save_list(&v, &self.h3_edit.items, &self.h3_edit.src, &self.mvar_objects, &self.local_objects, &self.h3_edit.kept, &meta);
        let edits = H3HeaderEdits {
            name: (self.variant_title != v.header.name).then(|| self.variant_title.clone()),
            description: (self.variant_description != v.header.description).then(|| self.variant_description.clone()),
            author: (self.variant_author != v.header.author).then(|| self.variant_author.clone()),
        };
        let edited = edits.name.is_some() || edits.description.is_some() || edits.author.is_some();
        save_h3_variant(&src, dst, &v2, &list, edited.then_some(&edits)).map_err(|e| e.to_string())?;
        let mut note = String::new();
        if !report.out_of_bounds.is_empty() { note.push_str(&format!(" {} object(s) sat outside the variant bounds and were CLAMPED to the edge.", report.out_of_bounds.len())); }
        if report.over_slot_cap > 0 { note.push_str(&format!(" {} object(s) had no free slot (640 max) and were NOT written.", report.over_slot_cap)); }
        if report.unsaveable > 0 { note.push_str(&format!(" {} object(s) had no palette row and were NOT written.", report.unsaveable)); }
        if report.quotas_added > 0 { note.push_str(&format!(" {} quota(s) added to the table.", report.quotas_added)); }
        *self.h3_edit.last_save_note.borrow_mut() = note.clone();
        let written = list.iter().filter(|r| r.placed).count();
        log::info!("h3 mvar: saved {written} placed objects ({} records) -> {}{}{}", list.len(), dst.display(), if edited { " (header edited)" } else { "" }, note);
        Ok((written, report.over_slot_cap + report.unsaveable))
    }

    /// After a save the placed rows have records on disk: re-read the file so the editor's
    /// source records (slots included) match it, keeping the live objects' datums.
    pub(crate) fn h3_resync_after_save(&mut self, path: &Path) {
        let Ok(v) = parse_h3_variant(path) else { return };
        let Some(es) = self.h3_scene.as_mut() else { return };
        es.set_variant(Some((path.to_path_buf(), v.clone())));
        // match live objects to records by slot (loaded) or by position (new placements)
        let mut by_slot: HashMap<usize, &H3VariantObject> = v.objects.iter().filter(|o| o.placed).map(|o| (o.slot, o)).collect();
        let mut kept: Vec<H3VariantObject> = v.objects.iter().filter(|o| !o.placed).cloned().collect();
        let mut src: HashMap<u32, H3VariantObject> = HashMap::new();
        let datums: Vec<u32> = self.mvar_objects.iter().chain(self.local_objects.iter()).map(|o| o.datum).collect();
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
