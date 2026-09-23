//! Halo 4 Forge editing in the GUI: the `App` methods that turn a loaded Halo 4 map + `.mvar`
//! into the same editable object set the Reach editor works on.
//!
//! What lives here (main.rs only dispatches):
//!   * `h4_install_editor_scene`   - the load worker's assets -> `H4ObjectScene` + palette + objects
//!   * `h4_render_variant_objects` - the variant's records -> `ObjectInfo` + `ObjMeta` (`h4: Some`)
//!                                   + colours + source records, header strings, labels, camera
//!   * the palette adapter         - `static_palette*` rows named "<category> . <entry>[ . <variant>]"
//!                                   (localized through `ui\strings\forge`) + `H4PaletteItem` per row so
//!                                   `place` / drag-drop can `instantiate` by INDEX (an obje tag may
//!                                   sit in several quotas, so a tag alone is not an identity)
//!   * `h4_place_palette_item`     - a placed object gets its full `.mvar` record at once (so it saves)
//!   * `h4_save_variant_to` + gates- `build_h4_save_list` -> bounds / slot / quota / budget / type
//!                                   gates -> `h4::mvar::save_objects`
//!   * the Object window rows, `set` / `get` field names, the objects-list marker, the scale
//!
//! Scale: the Halo 4 record carries an object scale - the 6-bit real at datum +0x2C
//! (`H4Fields::scale_q`, see `h4::mvar::H4PlacedObject::scale_q` for the halo4.dll evidence:
//! `c_map_variant::create_object` -> placement data +0x58 -> `object_new` -> obj +0xA0/+0xA4).
//! `set <datum> scale <f>` / the Object window row write that field (quantised to the engine's
//! 64 steps over [0, 10]; 1.0 = q 7 = the shipped default, which the engine itself dequantises
//! to 1.048), and the file carries it. The consumer side is traced too (docs 11b): obj +0xA0 ->
//! node matrix element 0 -> the render-state packer `sub_18033E02C` -> GPU, plus physics shapes
//! / bounding sphere, no gate; H2A's `groundhog.dll` has the identical chain. Two in-game tests
//! and the shipped data show MCC does NOT draw it (docs 11c), so it lives under *raw* for
//! round-tripping. The H4 Editing Kit's `halo4_tag_test.exe` (debug strings) names the field
//! `variant-object-scale` (read_quantized_real 0..10, 6 bits), the lock `locked-from-forge-editing`
//! and the "spawn sequence" / "user data" pair `user-data1` / `user-data2`; its `object_new`
//! asserts `data->scale>0.0f` on the placement copy; no gate or global in any of the three MCC
//! builds (docs 11d). The raw row and `get` use the engine name.
//! Field names otherwise follow the game's own Forge menu: "Spawn sequence" (also "Spawn Order" in
//! the object summary), "Respawn time" / "spawn time", "user data", "Game type label", "Physics".

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use eframe::egui;
use hms_ipc::ObjectInfo;

use crate::h4::edit::{
    build_h4_save_list_core, h4_tag, palette_items, palette_object_type, shape_wu_to_raw, variant_to_editor, H4GateInputs, H4ObjMeta,
    H4PaletteItem, NO_LABEL,
};
use crate::h4::edit_scene::H4ObjectScene;
use crate::h4::mvar::{H4PlacedObject, H4TypeData, H4_SLOTS, NEW_SLOT};
use crate::h4::palette::{ForgePalette, H4Pose};
use crate::{App, BspWarnAction, ObjMeta};

/// The save gate's findings, shown by `h4_save_window_ui`.
#[derive(Clone, Debug, Default)]
pub struct H4SaveGate {
    /// Datums outside the variant bounds (Halo 4 has no bsp escape; "Save anyway" CLAMPS them).
    pub out_of_bounds: Vec<u32>,
    /// Datums whose record type differs from the palette entry's MP type (gate 3, warning).
    pub type_mismatch: Vec<u32>,
    /// Objects past the 651 slots (dropped by the save).
    pub no_slot: usize,
    /// Objects with no resolvable palette entry (dropped by the save).
    pub unsaveable: usize,
}

impl H4SaveGate {
    pub fn is_clean(&self) -> bool {
        self.out_of_bounds.is_empty() && self.type_mismatch.is_empty() && self.no_slot == 0 && self.unsaveable == 0
    }
}

/// Everything the Halo 4 editor keeps beside the shared object lists - one `App` field so main.rs
/// gains a single line.
#[derive(Default)]
pub struct H4EditState {
    /// The rich palette (categories / entries / variants / localized names / MP defaults).
    pub pal: Option<ForgePalette>,
    /// Parallel to `App.static_palette` while a Halo 4 map is active (`h4::edit::palette_items`).
    pub items: Vec<H4PaletteItem>,
    /// Source record per datum (loaded objects: the decoded slot; placed objects: the record
    /// `instantiate` built). A save starts from it so every unmodelled bit survives.
    pub src: HashMap<u32, H4PlacedObject>,
    /// Records HMS cannot display (quota / variant outside the palette, null tag); carried through
    /// a save verbatim - they still occupy a slot.
    pub unresolved: Vec<H4PlacedObject>,
    /// The held GUI save + its gate findings (the save window decides).
    pub save_pending: Option<(BspWarnAction, H4SaveGate)>,
    /// "Save anyway" re-issues the save with the gate skipped once (out-of-bounds objects clamped).
    pub save_skip_once: bool,
    /// What the last save could not write (status suffix), like the Reach `save_failure_note`
    /// (a `Cell`-style slot: the save runs on `&self` like the Reach one).
    pub last_save_note: std::cell::RefCell<String>,
}

/// Multiplayer object type names (`H4PlacedObject::object_type`, 6 bits; docs/halo4_mvar_layout.md).
pub fn h4_type_name(t: u8) -> &'static str {
    match t {
        0 => "ordinary",
        1 => "weapon",
        2 => "grenade",
        3 => "projectile",
        4 => "powerup",
        5 => "equipment",
        6 => "ammo pack",
        7 => "light land vehicle",
        8 => "heavy land vehicle",
        9 => "flying vehicle",
        10 => "turret",
        11 => "device",
        12 => "dominion pad",
        13 => "teleporter sender",
        14 => "teleporter receiver",
        15 => "teleporter 2-way",
        16 => "player spawn",
        17 => "respawn zone",
        18 => "hold spawn / anti zone",
        19 => "hold spawn objective",
        20 => "named location",
        21 => "capture point",
        22 => "flag",
        23 => "bomb",
        24 => "ball",
        25 => "hill",
        26 => "safe area",
        27 => "kill area",
        28 => "loadout camera",
        29 => "juggernaut",
        30 => "territory",
        31 => "trait zone",
        32 => "initial ordnance",
        33 => "random ordnance",
        34 => "objective ordnance",
        35 => "unknown 35",
        _ => "?",
    }
}

/// Shape value slot names per shape (`H4Shape::value_count`): sphere radius; cylinder
/// radius / top / bottom; box width / length / top / bottom. Raw units are 1/256 wu (consistent
/// with the shipped data; not confirmed in-game).
pub fn h4_shape_slot_names(shape: u8) -> &'static [&'static str] {
    match shape {
        1 => &["radius"],
        2 => &["radius", "top", "bottom"],
        3 => &["width", "length", "top", "bottom"],
        _ => &[],
    }
}

/// wu -> raw shape value (1/256 wu, clamped to 16 bits).
pub fn h4_wu_to_shape(wu: f32) -> u16 { shape_wu_to_raw(wu) }

/// True when a path is under a `hopper_map_variants` folder. That folder feeds matchmaking
/// only - MCC's Custom Games list comes from `halo4\map_variants` and the account's
/// `LocalFiles\<xuid>\Halo4\Map` - so a save there is refused.
fn is_hopper_variant_dir(dir: &Path) -> bool {
    dir.components().any(|c| c.as_os_str().to_string_lossy().eq_ignore_ascii_case("hopper_map_variants"))
}

/// The GUI `ObjMeta` of a Halo 4 editor record (`h4::edit::H4ObjMeta`): the
/// shared fields under their Reach names + the H4 block. `name` drops the trailing ':' of a
/// single-variant "entry:" so the objects list / `get` read cleanly.
pub fn obj_meta_from_h4(m: &H4ObjMeta) -> ObjMeta {
    ObjMeta {
        name: m.name.trim_end_matches(':').to_string(),
        folder: m.quota as u16,
        item: m.variant.unwrap_or(0),
        pos: [0.0; 3],
        team: m.team as u8,
        color: m.color.map(|c| c as i32).unwrap_or(-1),
        cached_type: m.object_type,
        spawn_seq: m.spawn_sequence as i32,
        respawn: m.spawn_time,
        label_idx: m.label_idx,
        label: m.label.clone(),
        placement: m.placement,
        boundary_shape: m.boundary_shape,
        boundary: m.h4.shape_values,
        weapon_clips: m.weapon_clips,
        tele_channel: m.tele_channel,
        tele_passability: m.tele_passability,
        location_name: m.location_name,
        spawn_rel: m.parent as i32,
        slot: m.slot,
        flags: Default::default(),
        h4: Some(Box::new(m.h4.clone())),
    }
}

/// `ObjMeta` straight from a record (a placed / duplicated object): `H4ObjMeta::from_record` +
/// `obj_meta_from_h4`, with the record's position.
pub fn h4_meta_from_record(rec: &H4PlacedObject, name: &str, labels: &[String]) -> ObjMeta {
    let mut m = obj_meta_from_h4(&H4ObjMeta::from_record(rec, name, labels));
    m.pos = rec.pos;
    m
}

impl App {
    // -----------------------------------------------------------------------------------------
    // load
    // -----------------------------------------------------------------------------------------

    /// Install the editor scene from the load worker's assets, publish the map's Forge palette
    /// to the palette panel and turn the loaded variant's records into editor objects. Returns
    /// the number of variant objects placed (0 without a variant).
    pub(crate) fn h4_install_editor_scene(&mut self, assets: crate::h4::scene::H4EditorAssets) -> usize {
        let es = H4ObjectScene::new(assets, &self.render_state.device, &self.render_state.queue);
        self.h4_scene = Some(Box::new(es));
        self.h4_edit = H4EditState::default();
        self.h4_fill_palette();
        // a fresh map drops the previous placed set (the Reach load does the same)
        self.local_objects.clear();
        self.mvar_objects.clear();
        self.mvar_colors.clear();
        self.mvar_meta.clear();
        self.mvar_unresolved.clear();
        self.next_local_datum = 0xF000_0000;
        let n = self.h4_render_variant_objects();
        if self.h4_scene.as_ref().and_then(|s| s.variant()).is_none() {
            // no variant rode along: the map alone (the last-opened variant may still be queued)
            self.current_variant_path = None;
            self.loaded_map_id = self.selected_map.and_then(|i| self.map_candidates.get(i)).and_then(|c| c.map_id)
                .or_else(|| crate::mapcat::read_map_id(Path::new(&self.map_path)));
        }
        n
    }

    /// The Forge palette of the active Halo 4 map -> `static_palette*` + `h4_edit.items`
    /// Every entry variant is one row; a null tag is skipped; a variant without
    /// a render model still gets a row (it places as the marker cube and saves normally).
    pub(crate) fn h4_fill_palette(&mut self) {
        let Some(es) = self.h4_scene.as_ref() else { return };
        let pal = crate::h4::palette::palette(es.cache());
        let items = palette_items(&pal);
        let mut static_pal: Vec<(u32, String)> = Vec::with_capacity(items.len());
        let mut cat: Vec<u32> = Vec::with_capacity(items.len());
        let mut catname: Vec<String> = Vec::with_capacity(items.len());
        for it in &items {
            // a hidden (superforge) category's objects can never be grabbed in the game: the
            // Forge highlight predicate (halo4.dll sub_1800BD8C8) rejects any object whose quota
            // resolves to a palette category with flag bit 0 - say so on the row
            let cat_disp = if it.hidden { format!("{} (superforge: FIXED in-game, not grabbable)", it.palette_display) } else { it.palette_display.clone() };
            let display = if !it.variant_display.is_empty() && it.variant_display != it.display {
                format!("{} \u{b7} {} \u{b7} {}", cat_disp, it.display, it.variant_display)
            } else {
                format!("{} \u{b7} {}", cat_disp, it.display)
            };
            static_pal.push((it.obje_tag.map(h4_tag).unwrap_or(0), display));
            cat.push(pal.entries.get(it.quota as usize).map(|e| e.category as u32).unwrap_or(0));
            catname.push(cat_disp);
        }
        log::info!("h4 forge palette: {} placeable rows over {} entries / {} categories ({})", items.len(), pal.entries.len(), pal.categories.len(),
            if pal.localized { "localized" } else { "raw string ids" });
        self.static_palette = static_pal;
        self.static_palette_cat = cat;
        self.static_palette_catname = catname;
        self.static_palette_variant = vec![0; items.len()];
        self.selected_static_pal = if items.is_empty() { None } else { Some(0) };
        self.h4_edit.items = items;
        self.h4_edit.pal = Some(pal);
    }

    /// Palette row whose RAW string ids match `needle` (lower-case substring: `bb_1x1`,
    /// `sp_respawn_point`, `ff_spawning`), for `place <name>` when the localized row text misses.
    pub(crate) fn h4_palette_index_by_raw(&self, needle: &str) -> Option<usize> {
        if !self.h4_active { return None; }
        self.h4_edit.items.iter().position(|it| {
            it.name.to_lowercase().contains(needle) || (!it.variant_name.is_empty() && it.variant_name.to_lowercase().contains(needle))
        })
    }

    /// Localized name of a (quota, variant) pair for the Object window (read-only).
    pub(crate) fn h4_quota_name(&self, quota: u16, variant: u8) -> String {
        let Some(pal) = self.h4_edit.pal.as_ref() else { return String::new() };
        if quota > 255 { return String::new(); }
        match pal.resolve(Some(quota as u8), Some(variant)) {
            Some((e, v)) => if v.sid != 0 { format!("{} / {}", e.display, v.display) } else { e.display.clone() },
            None => match pal.entry(quota as u8) {
                Some(e) => format!("{} / variant {variant} (out of range)", e.display),
                None => "quota out of range".to_string(),
            },
        }
    }

    /// The loaded Halo 4 variant's records -> editor objects (the Halo 4 `render_variant_objects`):
    /// header strings, `ObjectInfo` + `ObjMeta` (`h4: Some`) + colours per record,
    /// the source records for the save, the unresolved list, labels, `current_variant_path`,
    /// `loaded_map_id`, the last-variant memo and the camera at a spawn.
    pub(crate) fn h4_render_variant_objects(&mut self) -> usize {
        let Some((path, v)) = self.h4_scene.as_ref().and_then(|s| s.variant()).map(|(p, v)| (p.clone(), v.clone())) else { return 0 };
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        self.variant_title = v.title_key.clone();
        self.variant_description = v.description_key.clone();
        self.variant_author = v.author.clone();
        self.variant_editor = v.editor.clone();
        self.variant_header_dirty = false;
        self.seed_variant_globals(v.globals());
        self.local_objects.clear();
        self.clear_selection_state();
        let t0 = std::time::Instant::now();
        let Some(es) = self.h4_scene.as_ref() else { return 0 };
        let set = variant_to_editor(es.cache(), es.palette(), &v);
        let n_marker = set.stats.skipped_no_model;
        let out = set.objects;
        let colors = set.colors;
        let meta: HashMap<u32, ObjMeta> = set.meta.iter().map(|(&d, m)| {
            let mut om = obj_meta_from_h4(m);
            om.pos = set.src.get(&d).map(|r| r.pos).unwrap_or_default();
            (d, om)
        }).collect();
        // source records of the RESOLVED datums only (the unresolved ones ride in `unresolved`)
        let src: HashMap<u32, H4PlacedObject> = set.src.into_iter().filter(|(d, _)| meta.contains_key(d)).collect();
        let unresolved = set.unresolved;
        let unresolved_idx: std::collections::HashSet<usize> = set.unresolved_datums.iter().map(|d| (d - crate::h4::edit::LOADED_DATUM_BASE) as usize).collect();
        let placed = out.len();
        let n_unresolved = unresolved.len();
        if placed == 0 && v.objects.len() > 0 { log::warn!("h4 variant '{name}': 0 of {} records resolved ({:?})", v.objects.len(), set.stats); }
        self.mvar_objects = out;
        self.mvar_colors = colors;
        self.mvar_meta = meta;
        self.mvar_unresolved = unresolved_idx;
        self.h4_edit.src = src;
        self.h4_edit.unresolved = unresolved;
        self.mvar_labels = v.labels.clone();
        self.apply_pending_obj_flags(); // the per-object flags + pseudo-scale from the project file
        self.current_variant_path = Some(path.clone());
        self.new_variant_template = None;
        self.loaded_map_id = Some(v.map_id);
        self.variant_retry = None;
        crate::mapcat::save_last_variant(&path.to_string_lossy());
        // (camera: gui.rs `h4_drive_load` lands it at the variant's loadout camera / initial
        // spawn through spawncam.rs right after this returns)
        let ms = t0.elapsed().as_millis();
        self.spawn_status = format!(
            "{name}: {placed} Halo 4 forge objects ({n_marker} marker-only, {n_unresolved} unresolved - they still occupy slots) in {ms} ms."
        );
        log::info!("{}", self.spawn_status);
        if placed > 0 { self.rebuild_overlays(); }
        placed
    }

    // -----------------------------------------------------------------------------------------
    // place
    // -----------------------------------------------------------------------------------------

    /// Place palette row `i` at `pos` as a fully-editable, SAVEABLE object: `instantiate` builds
    /// the `.mvar` record (MP defaults: placement / spawn time / shape / type data), which becomes
    /// the datum's source record; the shared `ObjMeta` fields mirror it.
    pub(crate) fn h4_place_palette_item(&mut self, i: usize, pos: glam::Vec3) -> Result<(u32, String), String> {
        let it = self.h4_edit.items.get(i).cloned().ok_or_else(|| format!("palette index {i} out of range (0..{})", self.h4_edit.items.len()))?;
        let (pal, es) = match (self.h4_edit.pal.as_ref(), self.h4_scene.as_ref()) {
            (Some(p), Some(s)) => (p, s),
            _ => return Err("no Halo 4 map is active".into()),
        };
        let inst = crate::h4::palette::instantiate(es.cache(), pal, it.quota, it.variant, H4Pose::at(pos.into()))
            .ok_or_else(|| format!("'{}' does not instantiate (quota {}.{})", it.display, it.quota, it.variant))?;
        let row_name = self.static_palette.get(i).map(|(_, n)| n.clone()).unwrap_or_else(|| it.display.clone());
        let mode_tag = es.mode_tag_of_obje(inst.tag);
        if mode_tag == 0 { return Err(format!("'{}' has no tag in this cache", it.display)); }
        let datum = self.next_dup_datum;
        self.next_dup_datum = self.next_dup_datum.wrapping_add(1);
        let rec = inst.record.clone();
        let name = if it.variant_name.is_empty() { it.name.clone() } else { format!("{}:{}", it.name, it.variant_name) };
        self.mvar_objects.push(ObjectInfo {
            datum, type_sig: 0, sig0: 0, sig1: 0, pos: rec.pos, health: 1.0, shield: 1.0,
            mode_tag, fwd: rec.fwd, up: rec.up, attached: [0; 8], primary_tag: h4_tag(inst.tag), variant_name_sid: 0,
        });
        let mut m = h4_meta_from_record(&rec, &name, &self.mvar_labels);
        m.slot = NEW_SLOT;
        self.mvar_colors.insert(datum, (m.team, if m.color < 0 { 0xFF } else { m.color as u8 }));
        self.mvar_meta.insert(datum, m);
        self.h4_edit.src.insert(datum, rec);
        if let Some(s) = self.objscene_mut() { s.invalidate(); }
        Ok((datum, row_name))
    }

    /// A duplicated object's source record = a copy of its template's (the Reach `spawn_copy_of`
    /// keeps the meta; for Halo 4 the record must follow too so the copy saves). Called by the
    /// dispatch in `spawn_copy_of` with (template datum, new datum).
    pub(crate) fn h4_copy_source_record(&mut self, from: u32, to: u32) {
        if !self.h4_active { return; }
        if let Some(mut r) = self.h4_edit.src.get(&from).cloned() {
            r.slot = NEW_SLOT;
            r.parent = -1;
            self.h4_edit.src.insert(to, r);
            if let Some(m) = self.mvar_meta.get_mut(&to) { m.slot = NEW_SLOT; m.spawn_rel = -1; }
        }
    }

    // -----------------------------------------------------------------------------------------
    // save
    // -----------------------------------------------------------------------------------------

    /// Bounds of the open Halo 4 variant (xmin xmax ymin ymax zmin zmax): the box a save
    /// quantises against - the open variant's, or the user's edited one (the gate and the clamp
    /// must agree with what the encoder writes).
    fn h4_bounds(&self) -> Option<[f32; 6]> {
        if let Some(g) = self.variant_globals.as_ref() { if self.global_edits.bounds != g.bounds { return Some(self.global_edits.bounds); } }
        self.h4_scene.as_ref().and_then(|s| s.bounds())
    }

    /// The object list a Halo 4 save writes + the gate findings, from the CURRENT editor state
    /// (pure over the lists; `h4::edit::build_h4_save_list` does the record work).
    fn h4_build_save(&self) -> Result<(Vec<H4PlacedObject>, H4SaveGate), String> {
        let es = self.h4_scene.as_ref().ok_or("no Halo 4 map is active")?;
        let bounds = self.h4_bounds().ok_or("no Halo 4 variant is open")?;
        let pal = self.h4_edit.pal.as_ref().ok_or("no Halo 4 palette")?;
        let type_of = |q: u8, v: Option<u8>| palette_object_type(pal, q, v);
        let h4meta: HashMap<u32, H4ObjMeta> = self.mvar_objects.iter().chain(self.local_objects.iter())
            .filter_map(|o| self.mvar_meta.get(&o.datum).and_then(H4ObjMeta::from_obj_meta).map(|m| (o.datum, m)))
            .collect();
        let (mut list, report) = build_h4_save_list_core(
            &self.h4_edit.src,
            &self.mvar_objects,
            &self.local_objects,
            &self.h4_edit.unresolved,
            &h4meta,
            &self.mvar_colors,
            &H4GateInputs { palette: es.palette(), bounds, palette_type: Some(&type_of) },
        );
        let gate = H4SaveGate {
            out_of_bounds: report.out_of_bounds.clone(),
            type_mismatch: report.type_mismatches.iter().map(|(d, _, _)| *d).collect(),
            no_slot: report.over_slot_cap,
            unsaveable: report.unsaveable,
        };
        // slot cap (651): the unresolved records come last, so a full variant drops those first
        list.truncate(H4_SLOTS);
        Ok((list, gate))
    }

    /// Clamp `pos` into the variant bounds (what "Save anyway" does to an out-of-bounds object -
    /// Halo 4 cannot encode a position outside them).
    fn h4_clamp_into_bounds(list: &mut [H4PlacedObject], b: [f32; 6]) -> usize {
        let mut n = 0;
        for o in list.iter_mut() {
            if crate::h4::edit::is_out_of_bounds(o.pos, b) { o.pos = crate::h4::edit::clamp_pos(o.pos, b); o.in_bounds = true; n += 1; }
        }
        n
    }

    /// File > Save / Save As for a Halo 4 variant (the rewritten file loads in MCC):
    /// `build_h4_save_list` -> gates (positions clamped into the bounds, the 651-slot
    /// cap) -> `h4::mvar::save_objects` with the header edits. Refuses a destination inside the
    /// matchmaking folder. Returns (objects written, objects not written).
    pub(crate) fn h4_save_variant_to(&self, dst: &Path) -> Result<(usize, usize), String> {
        let src = self.current_variant_path.clone().or_else(|| self.new_variant_template.clone()).ok_or("no Halo 4 variant is open")?;
        if !crate::h4::mvar::is_h4_variant(&src) { return Err(format!("{} is not a Halo 4 variant", src.display())); }
        // MCC lists custom Halo 4 variants from `halo4\map_variants\*.mvar`
        // (MCC-Win64-Shipping.exe sub_14043CE28 -> sub_1404430A0("halo4\\map_variants", "*.mvar"))
        // and from the signed-in account's `LocalFiles\<xuid>\Halo4\Map`; `hopper_map_variants`
        // feeds matchmaking only, so a file saved there never shows in Custom Games - refuse it.
        if let Some(dir) = dst.parent() {
            if is_hopper_variant_dir(dir) {
                return Err("hopper_map_variants is the matchmaking folder - the game never lists a file saved there. Save into halo4\\map_variants (Custom Games list) or LocalFiles\\<xuid>\\Halo4\\Map instead".into());
            }
        }
        let (mut list, gate) = self.h4_build_save()?;
        let clamped = match self.h4_bounds() { Some(b) => Self::h4_clamp_into_bounds(&mut list, b), None => 0 };
        let es = self.h4_scene.as_ref().ok_or("no Halo 4 map is active")?;
        let v = crate::h4::mvar::parse_h4_variant(&src).map_err(|e| format!("cannot re-read the source variant: {e}"))?;
        let spent = crate::h4::mvar::budget_spent(&list, es.palette());
        // the editable globals, diffed against the SOURCE file's values
        let (category, budget_max, bounds, quotas) = self.global_edits.diff(&v.globals());
        // Keep the file self-consistent: `maximum_budget` covers what the objects cost (the edited
        // maximum when the user set one, else the source file's). Neither field limits anything in
        // game - the loader recomputes the spend and restores the map's own sandbox budget - so an
        // over-budget variant is written, not refused.
        let want_max = crate::h4::edit::budget_max_for(budget_max.unwrap_or(v.budget_max), spent);
        let budget_max = (want_max != v.budget_max).then_some(want_max);
        let mut edits = crate::h4::mvar::H4HeaderEdits {
            title: (self.variant_title != v.title_key).then(|| self.variant_title.clone()),
            description: (self.variant_description != v.description_key).then(|| self.variant_description.clone()),
            author: (self.variant_author != v.author).then(|| self.variant_author.clone()),
            editor: (self.variant_editor != v.editor).then(|| self.variant_editor.clone()),
            labels: (self.mvar_labels != v.labels).then(|| self.mvar_labels.clone()),
            budget_spent: (spent != v.budget_spent).then_some(spent),
            category,
            budget_max,
            bounds,
            quotas,
            ..Default::default()
        };
        // stamp the CreatedBy / ModifiedBy blocks the way the game does: a NEW
        // variant (no file of its own yet) gets created = now, modified = cleared; a re-save of
        // an existing file keeps created and gets modified = now. The unique id stays the
        // template's - that is what the game itself writes (see H4HeaderEdits::stamp_new). An
        // unedited re-save of an opened file stays byte-identical.
        if self.current_variant_path.is_none() {
            edits.stamp_new(0);
        } else if !edits.is_empty() || dst != src {
            edits.stamp_modified(0);
        }
        let edits = (!edits.is_empty()).then_some(edits);
        let slots = crate::h4::mvar::save_objects(&src, dst, &list, edits.as_ref()).map_err(|e| e.to_string())?;
        let mut note = String::new();
        if clamped > 0 { note.push_str(&format!(" {clamped} object(s) were outside the variant bounds and were CLAMPED to the edge.")); }
        if gate.no_slot > 0 { note.push_str(&format!(" {} object(s) had no free slot (651 max) and were NOT written.", gate.no_slot)); }
        if gate.unsaveable > 0 { note.push_str(&format!(" {} object(s) had no palette entry and were NOT written.", gate.unsaveable)); }
        *self.h4_edit.last_save_note.borrow_mut() = note.clone();
        log::info!("h4 mvar: saved {} objects -> {}{}{}", slots.len(), dst.display(), if edits.is_some() { " (header edited)" } else { "" }, note);
        Ok((slots.len(), gate.no_slot + gate.unsaveable))
    }

    /// The Halo 4 save gate (the `bsp_warn_gate` analogue): before a GUI save,
    /// hold it and open the window when the list has findings. Returns true when the caller must
    /// NOT save now. Halo 4 saves are Save As only while unverified in-game: a plain Save is
    /// redirected to Save As.
    pub(crate) fn h4_save_gate(&mut self, action: BspWarnAction) -> bool {
        if !self.h4_active { return false; }
        if self.h4_edit.save_skip_once {
            self.h4_edit.save_skip_once = false;
            return false;
        }
        match self.h4_build_save() {
            Ok((_, gate)) if gate.is_clean() => false,
            Ok((_, gate)) => { self.h4_edit.save_pending = Some((action, gate)); true }
            Err(e) => { self.spawn_status = format!("Save: {e}"); true }
        }
    }

    /// The Halo 4 save-time window: out-of-bounds objects (double-click = select + fly there),
    /// over-quota entries, budget, type mismatches, dropped objects; "Save As anyway" clamps the
    /// out-of-bounds objects and proceeds, "Cancel" drops the save.
    pub(crate) fn h4_save_window_ui(&mut self, ctx: &egui::Context) {
        let Some((action, gate)) = self.h4_edit.save_pending.clone() else { return };
        let bounds = self.h4_bounds();
        // out-of-bounds rows follow the live positions (an object moved back inside drops off)
        let oob: Vec<u32> = gate.out_of_bounds.iter().copied().filter(|d| {
            match (bounds, self.mvar_objects.iter().find(|o| o.datum == *d)) {
                (Some(b), Some(o)) => o.pos[0] < b[0] || o.pos[0] > b[1] || o.pos[1] < b[2] || o.pos[1] > b[3] || o.pos[2] < b[4] || o.pos[2] > b[5],
                _ => false,
            }
        }).collect();
        let mut save_anyway = false;
        let mut cancel = false;
        let mut goto: Option<u32> = None;
        let mut win_open = true;
        let orange = egui::Color32::from_rgb(255, 160, 60);
        let red = egui::Color32::from_rgb(255, 90, 70);
        egui::Window::new("Halo 4 save check")
            .open(&mut win_open)
            .default_size([600.0, 360.0])
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.screen_rect().center())
            .collapsible(false)
            .resizable(true)
            .show(ctx, |ui| {
                ui.separator();
                if !oob.is_empty() {
                    ui.label(egui::RichText::new(format!("{} object(s) are outside the variant bounds.", oob.len())).color(orange).strong());
                    if let Some(b) = bounds {
                        ui.small(format!("Bounds x [{:.1}, {:.1}]  y [{:.1}, {:.1}]  z [{:.1}, {:.1}]. Halo 4 cannot encode a position outside them: 'Save As anyway' CLAMPS these to the nearest edge. Double-click a row to select the object and fly to it.", b[0], b[1], b[2], b[3], b[4], b[5]));
                    }
                    let row_h = ui.spacing().interact_size.y;
                    egui::ScrollArea::vertical().id_salt("h4-oob").max_height(160.0).auto_shrink([false, false]).show_rows(ui, row_h, oob.len(), |ui, range| {
                        for k in range {
                            let d = oob[k];
                            let (name, pos) = match (self.mvar_meta.get(&d), self.mvar_objects.iter().find(|o| o.datum == d)) {
                                (Some(m), Some(o)) => (m.name.clone(), o.pos),
                                (_, Some(o)) => (String::new(), o.pos),
                                _ => (String::new(), [0.0; 3]),
                            };
                            let slot = self.mvar_meta.get(&d).map(|m| if m.slot == NEW_SLOT { "new".to_string() } else { format!("#{}", m.slot) }).unwrap_or_default();
                            let text = format!("{}  {slot}  ({:.1}, {:.1}, {:.1})", crate::prettify_stringid(&name), pos[0], pos[1], pos[2]);
                            let sel = self.selected_set.contains(&d);
                            let r = ui.add_sized([ui.available_width(), row_h], egui::SelectableLabel::new(sel, text));
                            if r.double_clicked() { goto = Some(d); }
                        }
                    });
                }
                if !gate.type_mismatch.is_empty() {
                    ui.label(egui::RichText::new(format!("{} object(s) store a type that differs from their palette entry's type (stale record; written as-is).", gate.type_mismatch.len())).color(orange));
                }
                if gate.no_slot > 0 {
                    ui.label(egui::RichText::new(format!("{} object(s) have no free slot (a variant holds 651) and will NOT be written.", gate.no_slot)).color(red));
                }
                if gate.unsaveable > 0 {
                    ui.label(egui::RichText::new(format!("{} object(s) have no palette entry and will NOT be written.", gate.unsaveable)).color(red));
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
            self.h4_edit.save_pending = None;
            return;
        }
        if save_anyway {
            self.h4_edit.save_pending = None;
            self.h4_edit.save_skip_once = true;
            match action {
                BspWarnAction::Save => self.save_current_variant(),
                BspWarnAction::SaveAs => self.save_variant_as(),
            }
            self.h4_edit.save_skip_once = false;
            return;
        }
        if cancel || !win_open {
            self.h4_edit.save_pending = None;
            self.status = "Save cancelled.".into();
        }
    }

    /// File > New for a Halo 4 map: the fewest-object shipped variant of the same map id is the
    /// template (its header / bounds / budget / trait sets are copied by the encoder); the object
    /// set starts empty. Save As writes the new file.
    pub(crate) fn h4_new_variant(&mut self) {
        let Some(map_id) = self.loaded_map_id else {
            self.spawn_status = "New variant: load a Halo 4 base map first.".into();
            return;
        };
        let mut best: Option<(usize, PathBuf)> = None;
        let mut dirs = crate::h4::mvar::variant_dirs();
        if let Some(d) = self.current_variant_path.as_ref().and_then(|p| p.parent()) { dirs.push(d.to_path_buf()); }
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map_or(true, |x| !x.eq_ignore_ascii_case("mvar")) { continue; }
                let Some(sm) = crate::h4::mvar::read_h4_summary(&p) else { continue };
                if sm.map_id != map_id { continue; }
                if best.as_ref().map_or(true, |(n, _)| sm.objects < *n) { best = Some((sm.objects, p)); }
            }
        }
        let Some((_, tpl)) = best else {
            self.spawn_status = format!("New variant: no Halo 4 .mvar for map id {map_id} found to use as a template.");
            return;
        };
        let Ok(v) = crate::h4::mvar::parse_h4_variant(&tpl) else {
            self.spawn_status = format!("New variant: cannot read the template {}", tpl.display());
            return;
        };
        self.mvar_objects.clear();
        self.local_objects.clear();
        self.clear_selection_state();
        self.mvar_colors.clear();
        self.mvar_meta.clear();
        self.mvar_unresolved.clear();
        self.h4_edit.src.clear();
        self.h4_edit.unresolved.clear();
        self.mvar_labels = v.labels.clone();
        self.seed_variant_globals(v.globals()); // the template's globals
        self.variant_title = "New variant".into();
        self.variant_description = String::new();
        if self.variant_author.trim().is_empty() { self.variant_author = "HMS".into(); }
        self.variant_editor = self.variant_author.clone();
        self.variant_header_dirty = true;
        self.current_variant_path = None;
        self.new_variant_template = Some(tpl.clone());
        // the scene's open variant supplies the bounds / budget / spawn camera
        // that the save gates read; a new variant inherits the template's, otherwise Save As
        // fails with "no Halo 4 variant is open" when the map was loaded without a variant.
        if let Some(sc) = self.h4_scene.as_mut() { sc.set_variant(Some((tpl.clone(), v))); }
        self.rebuild_overlays();
        self.spawn_status = format!("New Halo 4 variant (template {}): place objects, then Save into halo4\\map_variants (the folder MCC's Custom Games list scans).", tpl.file_name().unwrap_or_default().to_string_lossy());
    }

    // -----------------------------------------------------------------------------------------
    // script `set` / `get`
    // -----------------------------------------------------------------------------------------

    /// `set <datum> <field> <value>` for a Halo 4 object. Returns Ok(true) when the field was a
    /// Halo 4 one, Ok(false) to let the shared Reach names run, Err on a bad value. Names mirror
    /// the headless host: spawnseq / spawnorder, spawntime, label2..4, traitzone, userdata,
    /// shape / radius|width|length|top|bottom (wu, x256), b0..b3 raw, placement_hi, locked,
    /// scale (the record's `variant-object-scale` field, 0..10 quantised to 64 steps; `scale_q`
    /// sets the raw quantum; `real6` / `real6q` are the documented old aliases. MCC stores the
    /// field but does not draw it - the gametype rule is the only drawn scale).
    pub(crate) fn h4_set_field(m: &mut ObjMeta, field: &str, value: &str, labels: &mut Vec<String>) -> Result<bool, String> {
        let Some(h) = m.h4.as_deref_mut() else { return Ok(false) };
        let num = |what: &str| -> Result<f32, String> { value.trim().parse::<f32>().map_err(|_| format!("{what} must be a number")) };
        // a label given by NAME that the table lacks is appended
        let mut label_of = |v: &str| -> Result<Option<u8>, String> {
            let t = v.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("none") || t == "-1" { return Ok(None); }
            if let Ok(i) = t.parse::<usize>() { return if i < labels.len() { Ok(Some(i as u8)) } else { Err(format!("label index {i} past the table ({} labels)", labels.len())) }; }
            if let Some(i) = labels.iter().position(|l| l.eq_ignore_ascii_case(t)) { return Ok(Some(i as u8)); }
            if labels.len() >= 255 { return Err(format!("no label '{t}' in the variant and the table is full ({} labels)", labels.len())); }
            labels.push(t.to_string());
            Ok(Some((labels.len() - 1) as u8))
        };
        match field {
            // the Forge menu's "spawn sequence" (datum +0x43, Megalo user_data): the
            // game edits -100..100; the byte is signed, so 0..255 input wraps like the game reads it
            "spawnseq" | "spawn_seq" | "spawnsequence" | "spawn_sequence" | "spawnorder" | "spawn_order" => {
                let v: i32 = value.trim().parse().map_err(|_| "spawn sequence must be -128..255")?;
                if !(-128..=255).contains(&v) { return Err("spawn sequence must be -128..255 (-100..100 in the Forge menu)".into()); }
                m.spawn_seq = (v as u8) as i8 as i32;
            }
            "spawntime" | "spawn_time" | "respawn" => m.respawn = value.trim().parse().map_err(|_| "spawn time must be 0..255")?,
            "label" | "label1" => {
                let l = label_of(value)?;
                m.label_idx = l.map(|i| i as u16).unwrap_or(NO_LABEL);
                m.label = l.and_then(|i| labels.get(i as usize).cloned()).unwrap_or_default();
            }
            "label2" | "label3" | "label4" => {
                let k = (field.as_bytes()[5] - b'2') as usize;
                h.labels_extra[k] = label_of(value)?;
            }
            "traitzone" | "trait_zone" => {
                let v: u8 = value.trim().parse().map_err(|_| "trait zone must be 0..3")?;
                if v > 3 { return Err("trait zone must be 0..3".into()); }
                h.trait_zone = v;
            }
            "userdata" | "user_data" => h.user_data = value.trim().parse().map_err(|_| "user data must be -128..127")?,
            "placement_hi" => {
                let v: u8 = value.trim().parse().map_err(|_| "placement_hi must be 0..3")?;
                if v > 3 { return Err("placement_hi must be 0..3".into()); }
                h.placement_hi = v;
            }
            "locked" => h.locked = crate::forge_scale::parse_flag_value(value)?.unwrap_or(false),
            "shape" | "boundary" => {
                m.boundary_shape = match value.trim().to_lowercase().as_str() {
                    "none" | "0" => 0, "sphere" | "1" => 1, "cylinder" | "2" => 2, "box" | "3" => 3,
                    _ => return Err("shape must be none/sphere/cylinder/box".into()),
                };
                h.shape_values = m.boundary;
            }
            "radius" | "width" => { m.boundary[0] = h4_wu_to_shape(num("radius/width")?); h.shape_values = m.boundary; }
            "length" => { m.boundary[1] = h4_wu_to_shape(num("length")?); h.shape_values = m.boundary; }
            "top" => {
                let k = if m.boundary_shape == 2 { 1 } else { 2 };
                m.boundary[k] = h4_wu_to_shape(num("top")?); h.shape_values = m.boundary;
            }
            "bottom" => {
                let k = if m.boundary_shape == 2 { 2 } else { 3 };
                m.boundary[k] = h4_wu_to_shape(num("bottom")?); h.shape_values = m.boundary;
            }
            "b0" | "b1" | "b2" | "b3" => {
                let k = (field.as_bytes()[1] - b'0') as usize;
                m.boundary[k] = value.trim().parse().map_err(|_| format!("{field} must be 0..65535"))?;
                h.shape_values = m.boundary;
            }
            // the record's scale field (6-bit real over [0, 10], exact endpoints); written to
            // the file, not drawn by MCC
            "scale" | "real6" | "variant_object_scale" => {
                let s = num("scale")?;
                if !(0.0..=10.0).contains(&s) { return Err("scale must be 0..10 (the record stores 64 steps over that range)".into()); }
                h.set_scale(s);
            }
            "scale_q" | "real6q" => {
                let q: u8 = value.trim().parse().map_err(|_| "scale_q must be 0..63")?;
                if q > 63 { return Err("scale_q must be 0..63".into()); }
                h.scale_q = q;
            }
            "clips" | "weaponclips" | "weapon_clips" => m.weapon_clips = value.trim().parse().map_err(|_| "clips must be 0..255")?,
            "channel" | "telechannel" => m.tele_channel = value.trim().parse().map_err(|_| "channel must be 0..31")?,
            "passability" => m.tele_passability = value.trim().parse().map_err(|_| "passability must be 0..31")?,
            // Reach-only names that make no sense here
            "cachedtype" | "cached_type" => return Err("the Halo 4 object type is dictated by the palette entry (read-only)".into()),
            // `scaled on|off|default` is the shared SCALED pseudo-flag (gametype rule)
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The Halo 4 block of `get <datum>`.
    pub(crate) fn h4_dump_lines(&self, m: &ObjMeta) -> String {
        let Some(h) = m.h4.as_deref() else { return String::new() };
        let team_i = if m.team == 0xFF { -1 } else { m.team as i32 };
        let shapes = ["none", "sphere", "cylinder", "box"];
        let label_name = |l: Option<u8>| l.map(|i| self.mvar_labels.get(i as usize).cloned().unwrap_or_else(|| format!("#{i}"))).unwrap_or_else(|| "none".into());
        let labels = format!("'{}' (#{}), {}, {}, {}", m.label, m.label_idx as i32, label_name(h.labels_extra[0]), label_name(h.labels_extra[1]), label_name(h.labels_extra[2]));
        let nvals = [0usize, 1, 3, 4][(m.boundary_shape as usize).min(3)];
        let vals: Vec<String> = (0..nvals).map(|k| format!("{}={} ({:.2} wu)", h4_shape_slot_names(m.boundary_shape)[k], m.boundary[k], m.boundary[k] as f32 / 256.0)).collect();
        let type_data = match &h.ordnance {
            Some(t) => format!("{t:?}"),
            None => match h.object_type {
                1 | 12 => format!("byte={}", m.weapon_clips),
                20 => format!("index={}", m.location_name as i32),
                31 => format!("trait_zone={}", h.trait_zone),
                _ => format!("pair=({}, {})", m.tele_channel, m.tele_passability),
            },
        };
        format!(
            "  h4: quota={} variant={} ({})\n  h4: team={} color={} type={} ({}) spawn_sequence={} spawn_time={}\n  h4: labels={}\n  h4: placement=0x{:02X} [{}] placement_hi={} parent={} slot={}\n  h4: shape={} [{}]\n  h4: {}\n  h4: variant_object_scale={:.3} (q {}{}; the engine's `variant-object-scale`: stored in the .mvar and copied into the object by MCC, NOT drawn - docs 11c/11d) user_data={} locked={} slot_flags={}\n  h4: gametype_scale={} (scale label + spawn sequence, SCALED {}) -> drawn x{:.3}\n",
            m.folder, m.item, self.h4_quota_name(m.folder, m.item),
            crate::forge_team_name(team_i), if m.color < 0 { "inherit".to_string() } else { crate::forge_team_name(m.color) },
            h.object_type, h4_type_name(h.object_type), m.spawn_seq, m.respawn,
            labels,
            m.placement, crate::placement_summary(m.placement), h.placement_hi, m.spawn_rel, if m.slot == NEW_SLOT { "new".to_string() } else { m.slot.to_string() },
            shapes[(m.boundary_shape as usize).min(3)], vals.join(", "),
            type_data,
            h.scale(), h.scale_q, if h.is_scaled() { "" } else { ", default" }, h.user_data, h.locked, h.slot_flags,
            if m.scaled_on() { format!("x{:.3}", crate::forge_scale::spawn_seq_to_scale(m.spawn_seq, self.sc_convention, crate::forge_scale::team_from_u8(m.team))) } else { "off".to_string() },
            if m.scaled_on() { "on" } else { "off" }, m.scale(self.sc_convention),
        )
    }

    // -----------------------------------------------------------------------------------------
    // Object window rows (called INSIDE the "mvar-edit" grid; each helper ends its rows)
    // -----------------------------------------------------------------------------------------

    /// Rows after team / colour: object type (read-only), spawn sequence, spawn time.
    pub(crate) fn h4_rows_type_spawn(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        let Some(h) = m.h4.as_deref() else { return };
        ui.label("quota name");
        ui.add(egui::Label::new(egui::RichText::new(self.h4_quota_name(m.folder, m.item)).small()).truncate())
            .on_hover_text("The palette entry / variant the quota + variant indices name (read-only; edit the indices above)");
        ui.end_row();
        ui.label("object type");
        ui.label(format!("{} ({})", h4_type_name(h.object_type), h.object_type))
            .on_hover_text("Multiplayer object type - dictated by the palette entry's object (read-only)");
        ui.end_row();
        // the Forge menu's "spawn sequence" (datum +0x43 = Megalo object.user_data);
        // like Reach, a Forge gametype with the `scale` label rule reads it as a size - shown live.
        ui.label("spawn sequence");
        ui.horizontal(|ui| {
            // signed byte: a '-' typed anywhere is the sign (crate::numfield)
            ui.add(egui::DragValue::new(&mut m.spawn_seq).range(-128..=127).custom_parser(crate::numfield::parse_signed))
                .on_hover_text("The Forge menu's 'spawn sequence' (\"This controls in what order the item will spawn.\"; -100..100 in game; datum +0x43, Megalo object.user_data): KOTH hill order, extraction site number, ... With the `scale` label a scaling Forge gametype reads it as the object's size (the SCALED flag below, same conventions as Reach).");
            if m.scaled_on() {
                let txt = format!("-> x{:.3}", crate::forge_scale::spawn_seq_to_scale(m.spawn_seq, self.sc_convention, crate::forge_scale::team_from_u8(m.team)));
                if self.obj_globals.scaled { ui.small(txt).on_hover_text("The size the scaling gametype rule gives this spawn sequence (active convention); the only scale MCC draws - the record's own `variant-object-scale` field (under raw) is stored but never drawn, docs 11c/11d"); }
                else { ui.small(egui::RichText::new(txt).weak()).on_hover_text("Scaled objects is OFF in Settings - rendered at x1"); }
            }
        });
        ui.end_row();
        ui.label("spawn time (s)");
        ui.add(egui::DragValue::new(&mut m.respawn).range(0..=255))
            .on_hover_text("Respawn time in seconds (the Forge menu's 'spawn time' / 'Respawn Time'), 0..255");
        ui.end_row();
    }

    /// Rows after the label 0 combo: labels 2..4 over the variant's label table.
    pub(crate) fn h4_rows_labels(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        let Some(h) = m.h4.as_deref_mut() else { return };
        for k in 0..3 {
            ui.label(format!("label {}", k + 2));
            let cur = match h.labels_extra[k] {
                None => "none".to_string(),
                Some(i) => match self.mvar_labels.get(i as usize) {
                    Some(n) if !n.is_empty() => format!("{n}  (#{i})"),
                    _ => format!("#{i}"),
                },
            };
            egui::ComboBox::from_id_salt(("h4-label", k))
                .selected_text(cur)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(h.labels_extra[k].is_none(), "none").clicked() { h.labels_extra[k] = None; }
                    for (i, name) in self.mvar_labels.iter().enumerate() {
                        let disp = if name.is_empty() { format!("#{i}") } else { format!("{name}  (#{i})") };
                        if ui.selectable_label(h.labels_extra[k] == Some(i as u8), disp).clicked() { h.labels_extra[k] = Some(i as u8); }
                    }
                });
            ui.end_row();
        }
    }

    /// Row after the placement editor: the two raw Halo 4 placement bits (8 / 9; their meaning
    /// is not known).
    pub(crate) fn h4_rows_placement_hi(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        let Some(h) = m.h4.as_deref_mut() else { return };
        ui.label("placement 8-9");
        ui.horizontal(|ui| {
            for bit in 0..2u8 {
                let mut on = h.placement_hi & (1 << bit) != 0;
                if ui.checkbox(&mut on, format!("bit {} (unverified)", 8 + bit)).on_hover_text("Placement bits new in Halo 4; their meaning is not known - written back as set").changed() {
                    if on { h.placement_hi |= 1 << bit; } else { h.placement_hi &= !(1 << bit); }
                }
            }
        });
        ui.end_row();
    }

    /// Shape rows: combo none / sphere / cylinder / box + raw 16-bit values with the wu hint.
    pub(crate) fn h4_rows_shape(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        let Some(h) = m.h4.as_deref_mut() else { return };
        ui.label("shape");
        let shapes = ["none", "sphere", "cylinder", "box"];
        egui::ComboBox::from_id_salt("h4-shape")
            .selected_text(shapes[(m.boundary_shape as usize).min(3)])
            .show_ui(ui, |ui| {
                for (i, s) in shapes.iter().enumerate() { ui.selectable_value(&mut m.boundary_shape, i as u8, *s); }
            });
        ui.end_row();
        let names = h4_shape_slot_names(m.boundary_shape);
        for (k, n) in names.iter().enumerate() {
            ui.label(format!("  {n}"));
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut m.boundary[k]).range(0..=65535).speed(8.0));
                ui.small(format!("= {:.2} wu", m.boundary[k] as f32 / 256.0)).on_hover_text("raw value / 256 (the 1/256 wu scale is consistent with the shipped data; not confirmed in-game)");
            });
            ui.end_row();
        }
        h.shape_values = m.boundary;
    }

    /// Type extras: weapon clips (1) / pad byte (12), teleporter channel + passability (13-15),
    /// named-location index (20), trait zone (31), ordnance tables (32/33/34), else the raw pair.
    pub(crate) fn h4_rows_type_extras(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        let Some(h) = m.h4.as_deref_mut() else { return };
        match h.object_type {
            1 => { ui.label("spare clips"); ui.add(egui::DragValue::new(&mut m.weapon_clips)); ui.end_row(); }
            12 => { ui.label("pad byte"); ui.add(egui::DragValue::new(&mut m.weapon_clips)).on_hover_text("dominion pad byte (255 on shipped pads)"); ui.end_row(); }
            13 | 14 | 15 => {
                ui.label("channel");
                ui.add(egui::DragValue::new(&mut m.tele_channel).range(0..=31)).on_hover_text("teleporter channel (5 bits)");
                ui.end_row();
                ui.label("passability");
                ui.add(egui::DragValue::new(&mut m.tele_passability).range(0..=31)).on_hover_text("teleporter passability (5 bits, raw; bit meaning assumed as Reach, not confirmed for Halo 4)");
                ui.end_row();
            }
            20 => {
                ui.label("location index");
                let mut ln = if m.location_name == 0xFFFF { -1 } else { m.location_name as i32 };
                ui.add(egui::DragValue::new(&mut ln).range(-1..=254));
                m.location_name = if ln < 0 { 0xFFFF } else { ln as u16 };
                ui.end_row();
            }
            31 => { ui.label("trait zone"); ui.add(egui::DragValue::new(&mut h.trait_zone).range(0..=3)).on_hover_text("trait-set index 0..3 (the four trait sets at the end of the variant)"); ui.end_row(); }
            32 | 33 | 34 => {
                let mut t = h.ordnance.clone().unwrap_or_else(|| H4TypeData::default_for(h.object_type));
                match &mut t {
                    H4TypeData::InitialOrdnance { team, weapon, timing } => {
                        ui.label("ordnance team"); ui.add(egui::DragValue::new(team).range(-1..=8)); ui.end_row();
                        ui.label("weapon index"); ui.add(egui::DragValue::new(weapon)); ui.end_row();
                        ui.label("drop timing"); ui.add(egui::DragValue::new(timing)); ui.end_row();
                    }
                    H4TypeData::RandomOrdnance(w) => {
                        ui.label("weapons (255 = none)");
                        ui.horizontal_wrapped(|ui| { for x in w.iter_mut() { ui.add(egui::DragValue::new(x)); } });
                        ui.end_row();
                    }
                    H4TypeData::ObjectiveOrdnance { lead, weapons } => {
                        ui.label("lead byte"); ui.add(egui::DragValue::new(lead)); ui.end_row();
                        ui.label("weapons (255 = none)");
                        ui.horizontal_wrapped(|ui| { for x in weapons.iter_mut() { ui.add(egui::DragValue::new(x)); } });
                        ui.end_row();
                    }
                    _ => {}
                }
                h.ordnance = Some(t);
            }
            35 => {}
            _ => {
                ui.label("type pair");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut m.tele_channel).range(0..=31));
                    ui.add(egui::DragValue::new(&mut m.tele_passability).range(0..=31));
                }).response.on_hover_text("the two 5-bit type values every 'other' type stores (raw)");
                ui.end_row();
            }
        }
    }

    /// User data, lock, and the collapsed raw fields (the record's own scale field lives under
    /// raw: MCC stores it but does not draw it, docs 11c).
    pub(crate) fn h4_rows_scale_raw(&self, ui: &mut egui::Ui, m: &mut ObjMeta) {
        let Some(h) = m.h4.as_deref_mut() else { return };
        ui.label("user data");
        ui.add(egui::DragValue::new(&mut h.user_data).custom_parser(crate::numfield::parse_signed)).on_hover_text("Forge 'user data': signed byte stored next to the spawn sequence (object multiplayer block +37; the engine's bitstream calls the pair user-data1 = spawn sequence, user-data2 = this, docs 11d); a free value for gametype scripts, 0 on almost every shipped object");
        ui.end_row();
        ui.label("locked");
        ui.checkbox(&mut h.locked, "").on_hover_text("Forge object lock (datum +0x33, the engine's `locked-from-forge-editing`): a locked object cannot be grabbed in Forge (the engine's setter drops a held object when it is locked); 30 shipped objects carry it");
        ui.end_row();
        ui.label("raw");
        egui::CollapsingHeader::new("raw").id_salt("h4-raw").default_open(false).show(ui, |ui| {
            egui::Grid::new("h4-raw-grid").num_columns(2).spacing([8.0, 2.0]).show(ui, |ui| {
                ui.label("variant-object-scale"); // the engine's own name (H4EK debug strings, docs 11d)
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut h.scale_q).range(0..=63));
                    ui.small(egui::RichText::new(format!("q -> x{:.3}{}", h.scale(), if h.is_scaled() { "" } else { " (default)" })).weak());
                    if h.is_scaled() && ui.small_button("default").clicked() { h.scale_q = crate::h4::mvar::SCALE_Q_DEFAULT; }
                }).response.on_hover_text("The record's own 6-bit scale field (the engine's `variant-object-scale`, 0..10 in 64 steps; q 7 = the shipped default, read as x1.048). MCC stores it and copies it into the spawned object but never draws it (docs/halo4_mvar_layout.md 11c/11d), so it is kept here only so the file round-trips. To scale a Halo 4 object use the scale-label gametype rule (spawn sequence row above).");
                ui.end_row();
                ui.label("slot flags");
                ui.add(egui::DragValue::new(&mut h.slot_flags).range(0..=3)).on_hover_text("2-bit slot flags (1 on every shipped object)");
                ui.end_row();
            });
        });
        ui.end_row();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::mvar::H4Shape;

    #[test]
    fn set_fields_cover_the_h4_names() {
        let mut m = h4_meta_from_record(&H4PlacedObject::new(3, None, 17, [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), "x", &[]);
        let mut labels = vec!["grif_spawn".to_string(), "grif_goal".to_string()];
        assert_eq!(App::h4_set_field(&mut m, "spawnorder", "7", &mut labels), Ok(true));
        assert_eq!(m.spawn_seq, 7);
        assert!(App::h4_set_field(&mut m, "spawnorder", "300", &mut labels).is_err());
        assert_eq!(App::h4_set_field(&mut m, "spawnseq", "-100", &mut labels), Ok(true), "signed like the Forge menu");
        assert_eq!(m.spawn_seq, -100);
        assert_eq!(App::h4_set_field(&mut m, "spawnseq", "255", &mut labels), Ok(true), "a raw byte wraps to the signed value");
        assert_eq!(m.spawn_seq, -1);
        // an unknown label name is appended to the table
        assert_eq!(App::h4_set_field(&mut m, "label", "scale", &mut labels), Ok(true));
        assert_eq!((m.label_idx, m.label.as_str(), labels.len()), (2, "scale", 3));
        assert!(m.scaled_on(), "the scale label turns the SCALED rule on by default");
        m.spawn_seq = 20;
        assert!((m.scale(crate::forge_scale::ScaleConvention::X330) - 1.7).abs() < 0.02, "X330 seq 20 -> x1.70 replaces the record scale");
        assert!((m.scale(crate::forge_scale::ScaleConvention::X47) - 3.0).abs() < 0.02, "X47 seq 20 -> x3.0");
        m.flags.scaled = Some(false);
        assert!((m.scale(crate::forge_scale::ScaleConvention::X330) - 1.0).abs() < 1e-6, "SCALED off -> x1.0: the record's scale field is not what MCC draws");
        m.h4.as_deref_mut().unwrap().scale_q = 14;
        assert!((m.scale(crate::forge_scale::ScaleConvention::X330) - 1.0).abs() < 1e-6, "a non-default scale field (x2.18 stored) still draws at x1.0, like the shipped Vortex shields in MCC");
        m.h4.as_deref_mut().unwrap().scale_q = crate::h4::mvar::SCALE_Q_DEFAULT;
        assert_eq!(App::h4_set_field(&mut m, "label2", "grif_goal", &mut labels), Ok(true));
        assert_eq!(m.h4.as_ref().unwrap().labels_extra[0], Some(1));
        assert_eq!(App::h4_set_field(&mut m, "label", "grif_spawn", &mut labels), Ok(true));
        assert_eq!((m.label_idx, m.label.as_str()), (0, "grif_spawn"));
        assert_eq!(App::h4_set_field(&mut m, "shape", "box", &mut labels), Ok(true));
        assert_eq!(App::h4_set_field(&mut m, "width", "3", &mut labels), Ok(true));
        assert_eq!(App::h4_set_field(&mut m, "top", "1.5", &mut labels), Ok(true));
        assert_eq!(m.boundary, [768, 0, 384, 0]);
        assert_eq!(m.h4.as_ref().unwrap().shape_values, m.boundary);
        // `scale` writes the record's 6-bit scale quantum
        assert_eq!(App::h4_set_field(&mut m, "scale", "2", &mut labels), Ok(true));
        assert_eq!(m.h4.as_ref().unwrap().scale_q, 13);
        assert!(m.h4.as_ref().unwrap().is_scaled());
        assert_eq!(App::h4_set_field(&mut m, "scale", "1", &mut labels), Ok(true));
        assert_eq!(m.h4.as_ref().unwrap().scale_q, 7, "quantising 1.0 gives the shipped q 7");
        assert!(!m.h4.as_ref().unwrap().is_scaled());
        assert!(App::h4_set_field(&mut m, "scale", "11", &mut labels).is_err());
        assert_eq!(App::h4_set_field(&mut m, "spawnseq", "20", &mut labels), Ok(true), "Reach's name is accepted too");
        assert_eq!(m.spawn_seq, 20);
        assert_eq!(App::h4_set_field(&mut m, "scale_q", "14", &mut labels), Ok(true));
        assert!((m.h4.as_ref().unwrap().scale() - 2.1774).abs() < 1e-3);
        assert_eq!(App::h4_set_field(&mut m, "team", "red", &mut labels), Ok(false), "shared names fall through");
        // a Reach object never takes an H4 name
        let mut r = ObjMeta::default();
        assert_eq!(App::h4_set_field(&mut r, "spawnorder", "1", &mut labels), Ok(false));
    }

    #[test]
    fn meta_from_record_maps_the_shared_fields() {
        let mut rec = H4PlacedObject::new(48, Some(2), 16, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        rec.team = 1;
        rec.color = Some(3);
        rec.labels = [Some(1), None, Some(0), None];
        rec.placement = 0x2CC;
        rec.shape = H4Shape::Cylinder;
        rec.shape_values = [768, 512, 512, 0];
        rec.parent = 5;
        rec.slot = 9;
        let m = h4_meta_from_record(&rec, "sp_initial_spawn:", &["a".into(), "b".into()]);
        assert_eq!(m.name, "sp_initial_spawn", "the single-variant trailing ':' is dropped");
        assert_eq!(m.pos, [1.0, 2.0, 3.0]);
        assert_eq!((m.folder, m.item, m.team, m.color, m.cached_type), (48, 2, 1, 3, 16));
        assert_eq!((m.label_idx, m.label.as_str()), (1, "b"));
        assert_eq!(m.placement, 0xCC);
        let h = m.h4.as_ref().unwrap();
        assert_eq!(h.placement_hi, 2);
        assert_eq!(h.labels_extra, [None, Some(0), None]);
        assert_eq!((m.boundary_shape, m.boundary, h.shape_values), (2, [768, 512, 512, 0], [768, 512, 512, 0]));
        assert_eq!((m.spawn_rel, m.slot), (5, 9));
        assert_eq!(m.boundary_shape, 2, "cylinder");
    }

    #[test]
    fn clamp_moves_only_outside_objects() {
        let b = [-10.0, 10.0, -20.0, 20.0, 0.0, 5.0];
        let mut list = vec![
            H4PlacedObject::new(0, None, 0, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            H4PlacedObject::new(0, None, 0, [50.0, -30.0, 9.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        assert_eq!(App::h4_clamp_into_bounds(&mut list, b), 1);
        assert_eq!(list[0].pos, [0.0, 0.0, 1.0]);
        assert_eq!(list[1].pos, [10.0, -20.0, 5.0]);
    }

    /// Only the matchmaking folder is refused; `halo4\map_variants` is where
    /// MCC lists custom variants from, so it MUST be allowed.
    #[test]
    fn only_hopper_dir_is_refused() {
        assert!(is_hopper_variant_dir(Path::new("/x/halo4/hopper_map_variants")));
        assert!(!is_hopper_variant_dir(Path::new("/x/halo4/map_variants")));
        assert!(!is_hopper_variant_dir(Path::new("/tmp/somewhere/else")));
    }
}
