//! Halo 4 editor records: the per-object fields the Reach `ObjMeta` has no slot for, the
//! record <-> editor conversion, and the save list with its gates.
//!
//! `ObjMeta` (main.rs) keeps the SHARED fields under their Reach names (folder/item = quota /
//! variant, team, color, spawn_seq = spawn_sequence, respawn = spawn_time, label_idx = labels[0],
//! placement = low byte, boundary_shape / boundary = shape / values, weapon_clips / tele_* /
//! location_name = type_data, spawn_rel = parent, slot). Everything Halo 4-only lives here in
//! `H4Fields`, boxed behind `ObjMeta.h4` (None for a Reach object). See
//! docs/halo4_mvar_layout.md (the record layout a save writes).
//!
//! The record <-> editor conversion and the save list are pure functions shared by the GUI
//! (`h4_app.rs`) and the headless host (script_host.rs):
//!   * `H4Fields::from_record` / `apply_to` - the Halo 4-only block, both directions;
//!   * `H4ObjMeta` - the SHARED per-object seed (what `ObjMeta` / the host's `HMeta` copy their
//!     fields from), `from_record` / `apply_to` for every editable field;
//!   * `variant_to_editor(cache, palette, variant) -> H4EditorSet` - the Reach
//!     `render_variant_objects` for Halo 4: `ObjectInfo`s (datum `0xD000_0000 + i`, tags encoded,
//!     marker cube for model-less entries), colours, metas, source records, unresolved records;
//!   * `build_h4_save_list` - the Reach `build_save_list` line by line (source record when present
//!     else `instantiate`, pos always copied, orientation re-encoded only when turned, colours,
//!     meta fields, parent as a slot key, unresolved appended verbatim) + `H4SaveReport`;
//!   * the gates as pure functions: bounds (`out_of_bounds` / `clamp_into_bounds`), the 651-slot
//!     cap, and the type check (record type == palette type).

use std::collections::HashMap;

use hms_ipc::ObjectInfo;

use super::cache::H4Cache;
use super::mvar::{H4PaletteEntry, H4PlacedObject, H4Shape, H4TypeData, H4Variant, H4VariantStats, H4_SLOTS, NEW_SLOT};
use super::objects::model_of;
use super::palette::{ForgePalette, H4Pose};

/// Halo 4-only per-object fields. Values are stored RAW (as decoded from the
/// `.mvar`); the UI converts (shape values / 256 = wu, scale dequantised) on display only.
#[derive(Clone, Debug, PartialEq)]
pub struct H4Fields {
    /// Gametype labels 1..3 (label 0 rides in `ObjMeta.label_idx`).
    pub labels_extra: [Option<u8>; 3],
    /// Placement bits 8-9 (new in Halo 4; mapped into the object's multiplayer flags, no engine
    /// reader found, so their meaning is not known); the low byte is `ObjMeta.placement`.
    pub placement_hi: u8,
    /// Forge "user data" (signed byte next to the spawn sequence; 0 on almost every shipped
    /// object; a free byte for gametype scripts).
    pub user_data: i8,
    /// Forge object LOCK (datum +0x33): a locked object cannot be grabbed in Forge.
    pub locked: bool,
    /// OBJECT SCALE quantum (datum +0x2C, 6 bits over [0, 10]; q 7 = the 1.0
    /// default, dequantised 1.048 by the engine). The value the game spawns the object with -
    /// see `H4PlacedObject::scale_q`. Edit through [`Self::scale`] / [`Self::set_scale`].
    pub scale_q: u8,
    /// 2-bit slot flags (1 on every shipped object).
    pub slot_flags: u8,
    /// Multiplayer object type (6 bits) - dictated by the palette entry, effectively read-only.
    pub object_type: u8,
    /// Type 31: trait-set index (0..=3).
    pub trait_zone: u8,
    /// Types 32/33/34: the ordnance tables (None for every other type).
    pub ordnance: Option<H4TypeData>,
    /// Raw 16-bit shape values (1/256 wu, see `SHAPE_UNITS_PER_WU`) - mirrors `ObjMeta.boundary` but
    /// keeps the full 16 bits (the Reach array is 11-bit quantised on save).
    pub shape_values: [u16; 4],
}

impl Default for H4Fields {
    fn default() -> Self {
        H4Fields {
            labels_extra: [None; 3],
            placement_hi: 0,
            user_data: 0,
            locked: false,
            scale_q: super::mvar::SCALE_Q_DEFAULT,
            slot_flags: 1,
            object_type: 0,
            trait_zone: 0,
            ordnance: None,
            shape_values: [0; 4],
        }
    }
}

/// Mass edit for the Halo 4 block: copy across every field the user actually CHANGED between
/// `before` and `after` onto `target` (per-field diff, the same policy as
/// `ObjMeta::apply_changed_fields`). `object_type` is identity (dictated by the palette entry)
/// and is never mass-copied. A Reach `before`/`after` (None) leaves `target` untouched; an H4
/// `after` applied to a Reach `target` (None) stays None (a Reach object has no H4 block).
pub fn merge_h4_changed(target: Option<Box<H4Fields>>, before: Option<&H4Fields>, after: Option<&H4Fields>) -> Option<Box<H4Fields>> {
    let Some(mut t) = target else { return None };
    let (Some(b), Some(a)) = (before, after) else { return Some(t) };
    for k in 0..3 {
        if a.labels_extra[k] != b.labels_extra[k] { t.labels_extra[k] = a.labels_extra[k]; }
    }
    if a.placement_hi != b.placement_hi { t.placement_hi = a.placement_hi; }
    if a.user_data != b.user_data { t.user_data = a.user_data; }
    if a.locked != b.locked { t.locked = a.locked; }
    if a.scale_q != b.scale_q { t.scale_q = a.scale_q; }
    if a.slot_flags != b.slot_flags { t.slot_flags = a.slot_flags; }
    if a.trait_zone != b.trait_zone { t.trait_zone = a.trait_zone; }
    if a.ordnance != b.ordnance { t.ordnance = a.ordnance.clone(); }
    if a.shape_values != b.shape_values { t.shape_values = a.shape_values; }
    Some(t)
}

/// The scale a Halo 4 object shows in the game. The record's own
/// 6-bit scale (datum +0x2C) is NOT it: MCC stores it, decodes it and copies it into the
/// object, but the drawn size never follows it (docs/halo4_mvar_layout.md section 11c: the
/// shipped Exile Scorpion is stored at x0.565, Monolith's paired man cannons at x0.73 / x0.89,
/// every plain object at x1.048, and all of them are drawn at their normal size; two in-game
/// tests of rewritten values showed nothing). So the only scale the game shows is the one a
/// Forge gametype script applies with the Reach-style rule (`scale` label + spawn sequence ->
/// `object.set_scale`, Megalo action 0x4A = int / 100) - the HMS `SCALED` pseudo-flag
/// (`scaled_on`: override, else "has the scale label") decides, exactly as for Reach, and the
/// active convention maps the spawn sequence (the user's own Forge gametypes implement X330
/// with the RED cosmic seed and X47 - see section 12). Everything else renders at 1.0.
pub fn h4_effective_scale(scaled_on: bool, spawn_seq: i32, team: u8, conv: crate::forge_scale::ScaleConvention) -> f32 {
    if scaled_on {
        let t = crate::forge_scale::team_from_u8(team);
        let max = crate::forge_scale::object_max_scale(team).max(conv.max_scale());
        crate::forge_scale::spawn_seq_to_scale(spawn_seq, conv, t).clamp(0.01, max)
    } else {
        1.0
    }
}

impl H4Fields {
    /// The record's scale field dequantised the engine's way (the shipped default q 7 is
    /// 1.048). Stored and copied into the object by MCC but NOT what the game draws (see
    /// `h4_effective_scale`) - a raw field, kept for round-tripping.
    pub fn scale(&self) -> f32 { super::mvar::h4_dequantize_scale(self.scale_q) }
    /// Quantise `s` into the record (clamped into [0, 10]; 1.0 -> q 7).
    pub fn set_scale(&mut self, s: f32) { self.scale_q = super::mvar::h4_quantize_scale(s); }
    /// True when the scale quantum differs from the shipped default (a raw-field fact only:
    /// it does not make the object render scaled).
    pub fn is_scaled(&self) -> bool { self.scale_q != super::mvar::SCALE_Q_DEFAULT }

    /// The Halo 4-only fields of a decoded record (the shared ones go to `ObjMeta`, see the
    /// module doc).
    pub fn from_record(o: &super::mvar::H4PlacedObject) -> Self {
        use super::mvar::H4TypeData as T;
        H4Fields {
            labels_extra: [o.labels[1], o.labels[2], o.labels[3]],
            placement_hi: ((o.placement >> 8) & 0x3) as u8,
            user_data: o.user_data,
            locked: o.locked,
            scale_q: o.scale_q,
            slot_flags: o.flags,
            object_type: o.object_type,
            trait_zone: match o.type_data { T::TraitZone(z) => z, _ => 0 },
            ordnance: match &o.type_data {
                T::InitialOrdnance { .. } | T::RandomOrdnance(_) | T::ObjectiveOrdnance { .. } => Some(o.type_data.clone()),
                _ => None,
            },
            shape_values: o.shape_values,
        }
    }

    /// The inverse of `from_record`: write every Halo 4-only field back into `rec` (the scale
    /// quantum included). The type-specific tail is kept when it already fits the object type;
    /// the trait zone / ordnance tables replace it for types 31 / 32-34, and a tail that does
    /// not fit the type is reset to that type's default.
    pub fn apply_to(&self, rec: &mut H4PlacedObject) {
        rec.labels[1] = self.labels_extra[0];
        rec.labels[2] = self.labels_extra[1];
        rec.labels[3] = self.labels_extra[2];
        rec.placement = (rec.placement & 0x00FF) | (((self.placement_hi & 3) as u16) << 8);
        rec.user_data = self.user_data;
        rec.locked = self.locked;
        rec.scale_q = self.scale_q & 0x3F;
        rec.flags = self.slot_flags & 3;
        rec.object_type = self.object_type & 0x3F;
        rec.shape_values = self.shape_values;
        match rec.object_type {
            31 => rec.type_data = H4TypeData::TraitZone(self.trait_zone & 0x1F),
            32 | 33 | 34 => {
                if let Some(o) = &self.ordnance {
                    if type_data_fits(o, rec.object_type) { rec.type_data = o.clone(); }
                }
            }
            _ => {}
        }
        if !type_data_fits(&rec.type_data, rec.object_type) { rec.type_data = H4TypeData::default_for(rec.object_type); }
    }
}

/// The type-data variant the encoder accepts for an object type (mirror of the private
/// `H4TypeData::matches` in mvar.rs; `write_slot` refuses anything else).
pub fn type_data_fits(td: &H4TypeData, object_type: u8) -> bool {
    match object_type {
        1 | 12 => matches!(td, H4TypeData::Byte(_)),
        20 => matches!(td, H4TypeData::Index(_)),
        31 => matches!(td, H4TypeData::TraitZone(_)),
        32 => matches!(td, H4TypeData::InitialOrdnance { .. }),
        33 => matches!(td, H4TypeData::RandomOrdnance(_)),
        34 => matches!(td, H4TypeData::ObjectiveOrdnance { .. }),
        35 => matches!(td, H4TypeData::Empty),
        _ => matches!(td, H4TypeData::Pair(..)),
    }
}

/// Halo 4 tag ids are cache indices (0..~22k) and index 0 is a valid tag, but the editor
/// treats `0` / `0xFFFF_FFFF` as "no model": encode them with this flag.
pub const H4_TAG_FLAG: u32 = 0x4800_0000;

/// Cache tag index -> editor tag id.
#[inline]
pub fn h4_tag(idx: usize) -> u32 { H4_TAG_FLAG | (idx as u32 & 0xFFFF) }

/// Editor tag id -> cache tag index (None for the "no model" sentinels / a non-H4 id).
#[inline]
pub fn h4_tag_index(tag: u32) -> Option<usize> {
    (tag & 0xFFFF_0000 == H4_TAG_FLAG).then_some((tag & 0xFFFF) as usize)
}

// ---------------------------------------------------------------------------------------------
// editor records
// ---------------------------------------------------------------------------------------------

/// Datum of the i-th record of a loaded variant (the Reach scheme).
pub const LOADED_DATUM_BASE: u32 = 0xD000_0000;
/// First datum of a NEW / duplicated object (`next_dup_datum`).
pub const NEW_DATUM_BASE: u32 = 0xD800_0000;
/// "No label" in the shared `label_idx` field (the Reach convention).
pub const NO_LABEL: u16 = 0xFFFF;
/// "No named location" in the shared `location_name` field.
pub const NO_LOCATION: u16 = 0xFFFF;
/// Shape values are stored in 1/256 world units (consistent with every shipped file; not
/// confirmed in-game).
pub const SHAPE_UNITS_PER_WU: f32 = 256.0;

/// World units -> raw 16-bit shape value (saturating).
pub fn shape_wu_to_raw(wu: f32) -> u16 { (wu.max(0.0) * SHAPE_UNITS_PER_WU).round().min(65535.0) as u16 }
/// Raw 16-bit shape value -> world units.
pub fn shape_raw_to_wu(raw: u16) -> f32 { raw as f32 / SHAPE_UNITS_PER_WU }

/// The SHARED per-object editor record of a Halo 4 object: every editable field of an
/// `H4PlacedObject` under the Reach `ObjMeta` names, plus the Halo 4-only
/// block in `h4`. The GUI copies these into `ObjMeta` (folder/item = quota/variant, spawn_seq =
/// spawn_sequence, respawn = spawn_time, label_idx = label0, boundary_shape/boundary = shape/
/// `h4.shape_values`, weapon_clips / tele_* / location_name = the type tail, spawn_rel =
/// parent) and back at save time; the headless host keeps it as is.
#[derive(Clone, Debug, PartialEq)]
pub struct H4ObjMeta {
    /// "entry:variant" palette name (empty when unresolved).
    pub name: String,
    pub quota: u8,
    /// Variant index inside the quota entry (None = the entry's variant 0).
    pub variant: Option<u8>,
    /// -1 none, 0..7 team colours, 8 neutral (`ObjMeta.team` = `as u8`, 0xFF = none).
    pub team: i8,
    /// Object colour override (None = team / default; `ObjMeta.color` = -1).
    pub color: Option<u8>,
    /// `ObjMeta.cached_type` (read-only, dictated by the palette entry).
    pub object_type: u8,
    /// `ObjMeta.spawn_seq` (the Forge menu's "spawn sequence", signed -100..100 in game).
    pub spawn_sequence: i8,
    /// `ObjMeta.respawn` (seconds).
    pub spawn_time: u8,
    /// Gametype label 0 (`ObjMeta.label_idx`, `NO_LABEL` = none); labels 1..3 are `h4.labels_extra`.
    pub label_idx: u16,
    /// Resolved label 0 text ("" when none).
    pub label: String,
    /// Placement flags, LOW byte (`ObjMeta.placement`); bits 8-9 are `h4.placement_hi`.
    pub placement: u8,
    /// `ObjMeta.boundary_shape` (0 none, 1 sphere, 2 cylinder, 3 box); values in `h4.shape_values`.
    pub boundary_shape: u8,
    /// Types 1 / 12 (`H4TypeData::Byte`): spare clips / pad byte.
    pub weapon_clips: u8,
    /// Every other type (`H4TypeData::Pair`): teleporter channel + passability on 13-15.
    pub tele_channel: u8,
    pub tele_passability: u8,
    /// Type 20 (`H4TypeData::Index`), `NO_LOCATION` = -1.
    pub location_name: u16,
    /// Parent SLOT KEY (`ObjMeta.spawn_rel`, -1 = none). Names the parent's `slot`; the encoder
    /// remaps it to wherever that object lands and orphans a child whose parent is gone.
    pub parent: i16,
    /// This object's own slot (identity for the encoder), `NEW_SLOT` for a not-yet-saved add.
    pub slot: u16,
    /// The Halo 4-only block (`ObjMeta.h4`).
    pub h4: H4Fields,
}

impl H4ObjMeta {
    /// Every editable field of a decoded record (`name` = "entry:variant", `label` resolved
    /// through the variant's label table).
    pub fn from_record(o: &H4PlacedObject, name: &str, labels: &[String]) -> Self {
        let (weapon_clips, tele_channel, tele_passability, location_name) = match o.type_data {
            H4TypeData::Byte(b) => (b, 0, 0, NO_LOCATION),
            H4TypeData::Pair(a, b) => (0, a, b, NO_LOCATION),
            H4TypeData::Index(i) => (0, 0, 0, if i < 0 { NO_LOCATION } else { i as u16 }),
            _ => (0, 0, 0, NO_LOCATION),
        };
        H4ObjMeta {
            name: name.to_string(),
            quota: o.quota.unwrap_or(0),
            variant: o.variant,
            team: o.team,
            color: o.color,
            object_type: o.object_type,
            spawn_sequence: o.spawn_sequence,
            spawn_time: o.spawn_time,
            label_idx: o.labels[0].map_or(NO_LABEL, |l| l as u16),
            label: o.labels[0].and_then(|l| labels.get(l as usize).cloned()).unwrap_or_default(),
            placement: (o.placement & 0xFF) as u8,
            boundary_shape: o.shape as u8,
            weapon_clips,
            tele_channel,
            tele_passability,
            location_name,
            parent: o.parent,
            slot: o.slot,
            h4: H4Fields::from_record(o),
        }
    }

    /// The GUI's `ObjMeta` (Reach names) -> the shared record; None for a Reach
    /// object (`h4: None`) or one without a palette entry (`folder == 0xFFFF`). `boundary` is the
    /// edited shape array (the GUI mirrors it into `h4.shape_values`).
    pub fn from_obj_meta(m: &crate::ObjMeta) -> Option<Self> {
        let h4 = m.h4.as_deref()?;
        if m.folder == 0xFFFF { return None; }
        let mut h = h4.clone();
        h.shape_values = m.boundary;
        Some(H4ObjMeta {
            name: m.name.clone(),
            quota: m.folder.min(255) as u8,
            variant: (m.item != 0 && m.item != 0xFF).then_some(m.item),
            team: if m.team == 0xFF { -1 } else { (m.team as i8).clamp(-1, 8) },
            color: (m.color >= 0).then_some((m.color & 7) as u8),
            object_type: h4.object_type,
            spawn_sequence: m.spawn_seq.clamp(-128, 127) as i8,
            spawn_time: m.respawn,
            label_idx: m.label_idx,
            label: m.label.clone(),
            placement: m.placement,
            boundary_shape: m.boundary_shape,
            weapon_clips: m.weapon_clips,
            tele_channel: m.tele_channel,
            tele_passability: m.tele_passability,
            location_name: m.location_name,
            parent: m.spawn_rel.clamp(-1, i16::MAX as i32) as i16,
            slot: m.slot,
            h4: h,
        })
    }

    /// Write every editable field into `rec` (the properties panel owns these). Position,
    /// orientation and `slot` are NOT touched here: `build_h4_save_list` copies the pose from the
    /// live object and the slot is the source record's identity.
    pub fn apply_to(&self, rec: &mut H4PlacedObject) {
        rec.quota = Some(self.quota);
        // the record ALWAYS stores an explicit index (the game's own convention: a NONE index
        // makes the engine spawn the next palette entry, see `H4PlacedObject::new`); an
        // unchanged Some(0) stays as decoded, a decoded NONE (older HMS saves) is repaired
        let want = self.variant.unwrap_or(0);
        if rec.variant != Some(want) { rec.variant = Some(want); }
        rec.team = self.team.clamp(-1, 8);
        rec.color = self.color.map(|c| c & 7);
        rec.spawn_sequence = self.spawn_sequence;
        rec.spawn_time = self.spawn_time;
        rec.labels[0] = (self.label_idx != NO_LABEL).then_some(self.label_idx.min(255) as u8);
        rec.placement = (rec.placement & 0x0300) | self.placement as u16;
        rec.shape = match self.boundary_shape { 1 => H4Shape::Sphere, 2 => H4Shape::Cylinder, 3 => H4Shape::Box, _ => H4Shape::None };
        rec.parent = self.parent;
        // the type tail: the shared fields for the Byte / Pair / Index forms, then the H4 block
        // (trait zone / ordnance) - `H4Fields::apply_to` resets a tail that does not fit the type
        rec.type_data = match H4TypeData::default_for(self.object_type) {
            H4TypeData::Byte(_) => H4TypeData::Byte(self.weapon_clips),
            H4TypeData::Pair(..) => H4TypeData::Pair(self.tele_channel & 0x1F, self.tele_passability & 0x1F),
            H4TypeData::Index(_) => H4TypeData::Index(if self.location_name == NO_LOCATION { -1 } else { self.location_name as i16 }),
            _ => rec.type_data.clone(),
        };
        self.h4.apply_to(rec);
    }
}

/// One flat Forge palette item: what the palette panel / `place <name>` list from. This is the
/// editor's row view of a (`ForgeEntry`, `ForgeVariant`) pair from `palette.rs` - localized
/// names, tags and price flattened for display and index-addressed placement; the palette
/// module keeps the structured scnr data and builds the records (`instantiate`).
#[derive(Clone, Debug)]
pub struct H4PaletteItem {
    /// Index in the flat list.
    pub index: usize,
    pub quota: u8,
    pub variant: u8,
    /// Raw palette (category) sid name (`ff_weapons_human`) and its localized display.
    pub palette: String,
    pub palette_display: String,
    /// Raw entry name (`sp_respawn_point`) / localized display.
    pub name: String,
    pub display: String,
    /// Raw variant name ("" on single-variant entries) / localized display.
    pub variant_name: String,
    pub variant_display: String,
    pub obje_tag: Option<usize>,
    /// `mode` render model tag (None = marker without a model, still placeable).
    pub mode_tag: Option<usize>,
    /// Editor mode tag id: `h4_tag(mode)` or `H4_MARKER_TAG` when the item has no model.
    pub editor_mode_tag: u32,
    pub object_type: u8,
    pub price: i32,
    /// `Maximum Allowed` (<= 0 = unlimited).
    pub max: i32,
    /// Superforge-only palette (not in the normal Forge menu).
    pub hidden: bool,
}

/// The base map's Forge palette flattened in scnr order (quota = position of the entry), one
/// item per (entry, variant); items whose tag is null are skipped.
pub fn palette_items(pal: &ForgePalette) -> Vec<H4PaletteItem> {
    let mut out = Vec::new();
    for e in &pal.entries {
        let cat = pal.categories.get(e.category);
        for v in &e.variants {
            if v.tag.is_none() { continue; }
            out.push(H4PaletteItem {
                index: out.len(),
                quota: e.quota,
                variant: v.index as u8,
                palette: cat.map(|c| c.name.clone()).unwrap_or_default(),
                palette_display: cat.map(|c| c.display.clone()).unwrap_or_default(),
                name: e.name.clone(),
                display: e.display.clone(),
                variant_name: v.name.clone(),
                variant_display: if v.sid != 0 { v.display.clone() } else { String::new() },
                obje_tag: v.tag,
                mode_tag: v.mode,
                editor_mode_tag: v.mode.map_or(super::edit_scene::H4_MARKER_TAG, h4_tag),
                object_type: v.object_type(),
                price: e.price,
                max: e.max_allowed,
                hidden: cat.map_or(false, |c| c.hidden),
            });
        }
    }
    out
}

/// A fully-formed record for a NEW object of palette entry `quota` / `variant` at a pose
/// (`palette::instantiate` - the engine's Forge defaults: placement 0x0C / 0xCC, team neutral,
/// the MP block's spawn time + boundary, spare clips). None when the entry does not resolve.
pub fn instantiate_record(c: &H4Cache, pal: &ForgePalette, quota: u8, variant: Option<u8>, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> Option<H4PlacedObject> {
    super::palette::instantiate(c, pal, quota, variant.unwrap_or(0), H4Pose { pos, fwd, up }).map(|i| i.record)
}

/// Editor mode tag of an obje tag: `h4_tag(mode)`, the marker cube when the object has no render
/// model (still placeable / pickable), or 0 when the obje tag itself is unknown
/// (`H4ObjectScene::mode_tag_of_obje` without the scene).
pub fn mode_tag_of_obje(c: &H4Cache, obje: usize) -> u32 {
    if c.tag_meta(obje).is_none() { return 0; }
    match model_of(c, obje) {
        Some(mode) => h4_tag(mode),
        None => super::edit_scene::H4_MARKER_TAG,
    }
}

/// Resolve (quota, variant) through the mvar palette -> (obje tag, editor mode tag, "entry:variant").
/// None when the quota / variant is out of range or the entry's tag is null.
pub fn resolve_palette_item(c: &H4Cache, palette: &[H4PaletteEntry], quota: u8, variant: Option<u8>) -> Option<(usize, u32, String)> {
    let entry = palette.get(quota as usize)?;
    let pv = entry.variants.get(variant.unwrap_or(0) as usize)?;
    let tag = pv.tag?;
    let mode_tag = mode_tag_of_obje(c, tag);
    if mode_tag == 0 { return None; }
    Some((tag, mode_tag, format!("{}:{}", entry.name, pv.name)))
}

/// The header fields a loaded variant seeds the editor with.
#[derive(Clone, Debug, Default)]
pub struct H4HeaderSeed {
    /// Displayed strings (`$key` resolved) and the stored keys (what a save compares against).
    pub title: String,
    pub description: String,
    pub title_key: String,
    pub description_key: String,
    pub author: String,
    pub editor: String,
    pub map_id: u32,
    /// World bounds xmin xmax ymin ymax zmin zmax (the out-of-bounds gate).
    pub bounds: [f32; 6],
    pub budget_max: u32,
    pub budget_spent: u32,
}

/// Everything `variant_to_editor` produces: the Reach `render_variant_objects` outputs for a
/// Halo 4 variant.
#[derive(Clone, Debug, Default)]
pub struct H4EditorSet {
    /// One per RESOLVED record, datum `LOADED_DATUM_BASE + record index`, tags encoded
    /// (`primary_tag = h4_tag(obje)`, `mode_tag = h4_tag(mode)` or the marker cube).
    pub objects: Vec<ObjectInfo>,
    /// datum -> (team as u8, colour or 0xFF) for the render / team tint.
    pub colors: HashMap<u32, (u8, u8)>,
    /// datum -> shared editor record (+ the H4 block in `.h4`).
    pub meta: HashMap<u32, H4ObjMeta>,
    /// datum -> SOURCE record of every record (resolved or not): a save starts from these.
    pub src: HashMap<u32, H4PlacedObject>,
    /// Records HMS cannot display (no quota / out of the palette / null tag): they occupy a slot
    /// and are carried through a save verbatim.
    pub unresolved: Vec<H4PlacedObject>,
    pub unresolved_datums: Vec<u32>,
    pub labels: Vec<String>,
    pub header: H4HeaderSeed,
    pub stats: H4VariantStats,
}

/// A parsed variant -> editor objects / colours / metas / source records (the Reach
/// `render_variant_objects` for Halo 4; the counters mirror `mvar::variant_placements`).
pub fn variant_to_editor(c: &H4Cache, palette: &[H4PaletteEntry], v: &H4Variant) -> H4EditorSet {
    let mut set = H4EditorSet {
        labels: v.labels.clone(),
        header: H4HeaderSeed {
            title: v.title.clone(), description: v.description.clone(), title_key: v.title_key.clone(), description_key: v.description_key.clone(),
            author: v.author.clone(), editor: v.editor.clone(), map_id: v.map_id, bounds: v.bounds, budget_max: v.budget_max, budget_spent: v.budget_spent,
        },
        stats: H4VariantStats { objects: v.objects.len(), ..Default::default() },
        ..Default::default()
    };
    for (i, o) in v.objects.iter().enumerate() {
        let datum = LOADED_DATUM_BASE.wrapping_add(i as u32);
        set.src.insert(datum, o.clone());
        let resolved = match o.quota {
            None => { set.stats.skipped_no_quota += 1; None }
            Some(q) => match palette.get(q as usize) {
                None => { set.stats.skipped_quota_range += 1; None }
                Some(entry) => match entry.variants.get(o.variant.unwrap_or(0) as usize) {
                    None => { set.stats.skipped_variant_range += 1; None }
                    Some(pv) => match pv.tag {
                        None => { set.stats.skipped_null_tag += 1; None }
                        Some(tag) => {
                            let mode_tag = mode_tag_of_obje(c, tag);
                            if mode_tag == 0 { set.stats.skipped_null_tag += 1; None }
                            else {
                                if mode_tag == super::edit_scene::H4_MARKER_TAG { set.stats.skipped_no_model += 1; }
                                Some((tag, mode_tag, format!("{}:{}", entry.name, pv.name)))
                            }
                        }
                    }
                }
            }
        };
        let Some((obje, mode_tag, name)) = resolved else {
            set.unresolved.push(o.clone());
            set.unresolved_datums.push(datum);
            continue;
        };
        set.colors.insert(datum, (o.team as u8, o.color.unwrap_or(0xFF)));
        set.meta.insert(datum, H4ObjMeta::from_record(o, &name, &v.labels));
        set.objects.push(ObjectInfo {
            datum, type_sig: 0, sig0: 0, sig1: 0, pos: o.pos, health: 1.0, shield: 1.0,
            mode_tag, fwd: o.fwd, up: o.up, attached: [0; 8], primary_tag: h4_tag(obje), variant_name_sid: 0,
        });
        set.stats.placed += 1;
    }
    set
}

/// What `build_h4_save_list` found while building the list (the gates). Nothing here
/// stops the save by itself: the caller decides (GUI window / script WARNING or refusal).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H4SaveReport {
    /// Live objects with neither a source record nor a resolvable palette entry (dropped).
    pub unsaveable: usize,
    /// Datums whose position lies outside the variant bounds (Halo 4 has no bsp escape; the
    /// encoder CLAMPS them into the bounds, so warn - `clamp_into_bounds` makes the editor agree).
    pub out_of_bounds: Vec<u32>,
    /// Sum of the palette prices of the list (what the header's `budget_spent` becomes). Not a
    /// gate: the loader recomputes the spend and restores the maximum from the scenario's sandbox,
    /// so neither value limits what the game will spawn.
    pub budget_spent: u32,
    /// Objects past the 651 slots (the encoder refuses the list; the caller truncates or aborts).
    pub over_slot_cap: usize,
    /// (datum, record type, palette type) for every live object whose `object_type` disagrees
    /// with its palette entry (the Reach "wrong type" gate). A WARNING, not a refusal: shipped
    /// files carry such records too (2v2_abandon: 7 dropped weapons / grenades stored as type 0).
    pub type_mismatches: Vec<(u32, u8, u8)>,
}

impl H4SaveReport {
    /// The header edit a save should carry so `budget_spent` matches the list.
    pub fn budget_edit(&self) -> Option<u32> { Some(self.budget_spent) }
    /// Human-readable gate summary ("" when everything passed).
    pub fn warnings(&self) -> String {
        let mut s = String::new();
        if self.unsaveable > 0 { s.push_str(&format!("WARNING {} object(s) had no palette entry and were dropped\n", self.unsaveable)); }
        if !self.out_of_bounds.is_empty() {
            s.push_str(&format!("WARNING {} object(s) outside the variant bounds (Halo 4 has no bsp escape; clamped into the bounds on save): {}\n",
                self.out_of_bounds.len(), self.out_of_bounds.iter().map(|d| format!("0x{d:08X}")).collect::<Vec<_>>().join(" ")));
        }
        if self.over_slot_cap > 0 { s.push_str(&format!("ERROR {} object(s) past the {} slots\n", self.over_slot_cap, H4_SLOTS)); }
        for (d, have, want) in &self.type_mismatches { s.push_str(&format!("WARNING 0x{d:08X}: record type {have} but its palette entry is type {want}\n")); }
        s
    }
    /// A save must not proceed (the encoder would refuse the result).
    pub fn blocks_save(&self) -> bool { self.over_slot_cap > 0 }
}

/// True when `pos` lies outside the variant bounds (xmin xmax ymin ymax zmin zmax).
pub fn is_out_of_bounds(pos: [f32; 3], bounds: [f32; 6]) -> bool {
    (0..3).any(|i| pos[i] < bounds[2 * i] || pos[i] > bounds[2 * i + 1])
}

/// `pos` clamped into the variant bounds (what the encoder's quantiser does).
pub fn clamp_pos(pos: [f32; 3], bounds: [f32; 6]) -> [f32; 3] {
    let mut p = pos;
    for i in 0..3 { p[i] = p[i].clamp(bounds[2 * i].min(bounds[2 * i + 1]), bounds[2 * i + 1].max(bounds[2 * i])); }
    p
}

/// Datums of the live objects outside the bounds (gate 1).
pub fn out_of_bounds(live: &[ObjectInfo], bounds: [f32; 6]) -> Vec<u32> {
    live.iter().filter(|o| is_out_of_bounds(o.pos, bounds)).map(|o| o.datum).collect()
}

/// "Save anyway": clamp every out-of-bounds live object into the bounds so the editor shows
/// what the file will hold. Returns the datums moved.
pub fn clamp_into_bounds(live: &mut [ObjectInfo], bounds: [f32; 6]) -> Vec<u32> {
    let mut moved = Vec::new();
    for o in live.iter_mut() {
        if is_out_of_bounds(o.pos, bounds) { o.pos = clamp_pos(o.pos, bounds); moved.push(o.datum); }
    }
    moved
}

/// Gate 2: objects past the 651 slots.
pub fn over_slot_cap(n: usize) -> usize { n.saturating_sub(H4_SLOTS) }

/// The list's budget (sum of palette prices) = `mvar::budget_spent`. Written into the header for
/// the format's sake; the game recomputes it, so it gates nothing.
pub fn budget_of(list: &[H4PlacedObject], palette: &[H4PaletteEntry]) -> u32 { super::mvar::budget_spent(list, palette) }

/// The `maximum_budget` a save writes: what the caller asked for (the variant's own maximum, or an
/// edited one), raised to cover what the objects cost so the file stays self-consistent. Purely
/// cosmetic in game - the loader recomputes the spend and restores the map's own sandbox budget -
/// so a variant that costs more than its stored maximum is written, never refused.
pub fn budget_max_for(want: u32, spent: u32) -> u32 { want.max(spent) }

/// Gate 3: the palette's object type for a (quota, variant) pair (None when unresolved).
pub fn palette_object_type(pal: &ForgePalette, quota: u8, variant: Option<u8>) -> Option<u8> {
    pal.resolve(Some(quota), variant).map(|(_, v)| v.object_type())
}

/// The gate inputs of `build_h4_save_list_core`: the mvar palette (the prices the spend is summed
/// from), the variant bounds, and an optional (quota, variant) -> MP object type lookup for the
/// type gate (`palette_object_type` over a `ForgePalette`).
pub struct H4GateInputs<'a> {
    pub palette: &'a [H4PaletteEntry],
    pub bounds: [f32; 6],
    pub palette_type: Option<&'a dyn Fn(u8, Option<u8>) -> Option<u8>>,
}

/// Build the EXACT object list a Halo 4 save writes - the Reach `build_save_list`
/// (main.rs) line by line, as a pure function over the game-neutral `H4ObjMeta` records:
/// * walk `live` then `extra` IN ORDER; each object starts from its OWN source record (`src`, by
///   DATUM, never by slot arithmetic) so every bit HMS does not model survives; a new /
///   duplicated object without one is built from its meta (`H4PlacedObject::new` + every field
///   of the meta, which a `place` seeded from `palette::instantiate` and a `dup` cloned), so it
///   is unsaveable only when it has no meta at all;
/// * `pos` is always the live object's; the orientation is re-encoded (`set_orientation`) only
///   when the object was actually TURNED, so an untouched one keeps its bits verbatim;
/// * team / colour from `colors`, every other editable field from `meta` (`H4ObjMeta::apply_to`);
/// * `parent` stays a slot KEY (the encoder remaps it and orphans a child whose parent left);
/// * `unresolved` records are appended verbatim (they keep their slots);
/// * the gates are evaluated on the finished list into the report (nothing is clamped or
///   truncated here - the caller decides).
pub fn build_h4_save_list_core(
    src: &HashMap<u32, H4PlacedObject>,
    live: &[ObjectInfo],
    extra: &[ObjectInfo],
    unresolved: &[H4PlacedObject],
    meta: &HashMap<u32, H4ObjMeta>,
    colors: &HashMap<u32, (u8, u8)>,
    gates: &H4GateInputs,
) -> (Vec<H4PlacedObject>, H4SaveReport) {
    let mut list: Vec<H4PlacedObject> = Vec::with_capacity(live.len() + extra.len() + unresolved.len());
    let mut report = H4SaveReport::default();
    for o in live.iter().chain(extra.iter()) {
        let mut rec = match src.get(&o.datum) {
            Some(t) => t.clone(),
            None => {
                let Some(m) = meta.get(&o.datum) else { report.unsaveable += 1; continue };
                let mut r = H4PlacedObject::new(m.quota, m.variant, m.object_type, o.pos, o.fwd, o.up);
                r.slot = NEW_SLOT;
                // a placement the user never touched has no `colors` row; its meta
                // carries the defaults (team neutral), so those are what get saved
                m.apply_to(&mut r);
                r
            }
        };
        rec.pos = o.pos;
        // only re-quantise the orientation when the object was actually turned
        let turned = {
            let d = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs());
            d(o.fwd, rec.fwd) > 1e-6 || d(o.up, rec.up) > 1e-6
        };
        if turned { rec.set_orientation(o.fwd, o.up); }
        if let Some(m) = meta.get(&o.datum) {
            // the properties panel owns these (team / colour below win when a `colors` row exists)
            m.apply_to(&mut rec);
        }
        if let Some(&(team, color)) = colors.get(&o.datum) {
            rec.team = if team == 0xFF { -1 } else { (team as i8).clamp(-1, 8) };
            rec.color = (color != 0xFF).then_some(color & 7);
        }
        if is_out_of_bounds(rec.pos, gates.bounds) { report.out_of_bounds.push(o.datum); }
        if let (Some(q), Some(f)) = (rec.quota, gates.palette_type) {
            if let Some(want) = f(q, rec.variant) {
                if want != rec.object_type { report.type_mismatches.push((o.datum, rec.object_type, want)); }
            }
        }
        list.push(rec);
    }
    for u in unresolved { list.push(u.clone()); }
    report.budget_spent = budget_of(&list, gates.palette);
    report.over_slot_cap = over_slot_cap(list.len());
    (list, report)
}

/// The form over the GUI's `ObjMeta` (`h4: Some`): converts the metas of
/// the live / extra datums with `H4ObjMeta::from_obj_meta` and runs `build_h4_save_list_core`
/// with the bounds + palette gates (budget maximum / type lookup are the GUI's, see `H4GateInputs`).
pub fn build_h4_save_list(
    src: &HashMap<u32, H4PlacedObject>,
    live: &[ObjectInfo],
    extra: &[ObjectInfo],
    unresolved: &[H4PlacedObject],
    meta: &HashMap<u32, crate::ObjMeta>,
    colors: &HashMap<u32, (u8, u8)>,
    palette: &[H4PaletteEntry],
    bounds: [f32; 6],
) -> (Vec<H4PlacedObject>, H4SaveReport) {
    let h4meta: HashMap<u32, H4ObjMeta> = live.iter().chain(extra.iter())
        .filter_map(|o| meta.get(&o.datum).and_then(H4ObjMeta::from_obj_meta).map(|m| (o.datum, m)))
        .collect();
    build_h4_save_list_core(src, live, extra, unresolved, &h4meta, colors, &H4GateInputs { palette, bounds, palette_type: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_copies_only_changed_fields() {
        let base = H4Fields { user_data: 3, trait_zone: 2, ..Default::default() };
        let before = H4Fields { user_data: 9, ..Default::default() };
        let mut after = H4Fields { user_data: 9, locked: true, labels_extra: [Some(2), None, None], ..Default::default() };
        after.set_scale(2.0); // q 13
        let out = merge_h4_changed(Some(Box::new(base.clone())), Some(&before), Some(&after)).unwrap();
        assert_eq!(out.user_data, 3, "unchanged field keeps the target's value");
        assert_eq!(out.trait_zone, 2);
        assert!(out.locked);
        assert_eq!(out.labels_extra, [Some(2), None, None]);
        assert_eq!(out.scale_q, 13);
        assert!((out.scale() - 2.0161).abs() < 1e-3 && out.is_scaled(), "scale round-trips through the engine's 6-bit quantum");
        assert_eq!(out.object_type, base.object_type);
        // a Reach target stays Reach
        assert!(merge_h4_changed(None, Some(&before), Some(&after)).is_none());
        // a Reach edit leaves an H4 target untouched
        assert_eq!(*merge_h4_changed(Some(Box::new(base.clone())), None, None).unwrap(), base);
    }

    #[test]
    fn tag_encoding_round_trips_and_rejects_reach_ids() {
        assert_eq!(h4_tag_index(h4_tag(0)), Some(0));
        assert_eq!(h4_tag_index(h4_tag(0x2A7F)), Some(0x2A7F));
        assert_ne!(h4_tag(0), 0);
        assert_eq!(h4_tag_index(0), None);
        assert_eq!(h4_tag_index(0xFFFF_FFFF), None);
        assert_eq!(h4_tag_index(0x213E), None);
    }

    /// The round trip of every field through the editor record: record -> H4ObjMeta -> a fresh
    /// default record -> equals the original (pose / slot aside). Pure, no MCC files needed.
    #[test]
    fn obj_meta_round_trips_every_field() {
        let mut o = H4PlacedObject::new(48, Some(2), 1, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        o.slot = 17; o.flags = 3; o.parent = 40; o.scale_q = 21; o.locked = true; o.shape = H4Shape::Box;
        o.shape_values = [256, 512, 768, 1024]; o.placement = 0x2CC; o.team = 3; o.spawn_time = 45; o.color = Some(5);
        o.spawn_sequence = -100; o.user_data = -7; o.labels = [Some(1), Some(4), None, Some(9)]; o.type_data = H4TypeData::Byte(9);
        let m = H4ObjMeta::from_record(&o, "sp:x", &["a".into(), "b".into()]);
        assert_eq!((m.label_idx, m.label.as_str(), m.weapon_clips, m.placement, m.h4.placement_hi, m.h4.labels_extra), (1, "b", 9, 0xCC, 2, [Some(4), None, Some(9)]));
        let mut back = H4PlacedObject::new(0, None, 1, o.pos, o.fwd, o.up);
        m.apply_to(&mut back);
        back.slot = o.slot; back.bit = o.bit;
        let same = |a: &H4PlacedObject, b: &H4PlacedObject| {
            a.flags == b.flags && a.quota == b.quota && a.variant == b.variant && a.parent == b.parent && a.scale_q == b.scale_q && a.locked == b.locked
                && a.shape == b.shape && a.shape_values == b.shape_values && a.object_type == b.object_type && a.placement == b.placement && a.team == b.team
                && a.spawn_time == b.spawn_time && a.color == b.color && a.spawn_sequence == b.spawn_sequence && a.user_data == b.user_data && a.labels == b.labels && a.type_data == b.type_data
        };
        assert!(same(&o, &back), "round trip changed a field:\n{o:?}\n{back:?}");
        // the type tails: pair / index / trait zone / ordnance through the H4 block
        for (ty, td) in [(13u8, H4TypeData::Pair(3, 7)), (20, H4TypeData::Index(5)), (31, H4TypeData::TraitZone(2)),
            (32, H4TypeData::InitialOrdnance { team: 2, weapon: 9, timing: 300 }), (33, H4TypeData::RandomOrdnance([1, 2, 3, 4, 5, 6, 7, 255])),
            (34, H4TypeData::ObjectiveOrdnance { lead: 1, weapons: [9; 8] }), (35, H4TypeData::Empty), (0, H4TypeData::Pair(0, 0))] {
            let mut r = H4PlacedObject::new(1, None, ty, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
            r.type_data = td.clone();
            let m = H4ObjMeta::from_record(&r, "", &[]);
            let mut back = H4PlacedObject::new(1, None, ty, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
            m.apply_to(&mut back);
            assert_eq!(back.type_data, td, "type {ty} tail");
            assert!(type_data_fits(&back.type_data, ty));
        }
        // a tail that does not fit the type is reset, never written as garbage
        let mut r = H4PlacedObject::new(1, None, 31, [0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        r.type_data = H4TypeData::Byte(4);
        H4Fields { object_type: 31, ..Default::default() }.apply_to(&mut r);
        assert_eq!(r.type_data, H4TypeData::TraitZone(0));
    }

    #[test]
    fn bounds_gate_reports_and_clamps() {
        let b = [-10.0, 10.0, -20.0, 20.0, 0.0, 5.0];
        assert!(!is_out_of_bounds([0.0, 0.0, 1.0], b));
        assert!(is_out_of_bounds([0.0, 0.0, 2000.0], b));
        assert_eq!(clamp_pos([11.0, -25.0, 2000.0], b), [10.0, -20.0, 5.0]);
        let mk = |d: u32, p: [f32; 3]| ObjectInfo { datum: d, type_sig: 0, sig0: 0, sig1: 0, pos: p, health: 1.0, shield: 1.0, mode_tag: 1, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: 1, variant_name_sid: 0 };
        let mut live = vec![mk(1, [0.0, 0.0, 1.0]), mk(2, [0.0, 0.0, 2000.0])];
        assert_eq!(out_of_bounds(&live, b), vec![2]);
        assert_eq!(clamp_into_bounds(&mut live, b), vec![2]);
        assert_eq!(live[1].pos, [0.0, 0.0, 5.0]);
        assert!(out_of_bounds(&live, b).is_empty());
        assert_eq!(over_slot_cap(651), 0);
        assert_eq!(over_slot_cap(653), 2);
        assert_eq!(shape_wu_to_raw(1.5), 384);
        assert_eq!(shape_raw_to_wu(512), 2.0);
    }
}

/// Save stress suite over the SHIPPED variants (the Reach `save_stress` shape, main.rs):
/// decode -> `variant_to_editor` -> `build_h4_save_list` -> `save_objects` must be a byte-exact
/// fixed point with no edits, and after every kind of edit the reparsed file must hold EVERY
/// field of EVERY object as intended (type asserted, not just count / position). Skips without
/// the MCC Halo 4 files.
#[cfg(test)]
mod save_stress {
    use super::*;
    use crate::h4::cache::maps_dir;
    use crate::h4::mvar::{all_variant_files, load_forge_palette, parse_h4_variant, rebuild_h4_file, save_objects, H4HeaderEdits};
    use std::collections::{BTreeMap, HashMap};
    use std::path::{Path, PathBuf};

    /// map id -> cache path from every .mapinfo next to the maps.
    fn map_paths() -> HashMap<u32, PathBuf> {
        let mut out = HashMap::new();
        let Some(dir) = maps_dir() else { return out };
        for e in std::fs::read_dir(dir.join("info")).into_iter().flatten().flatten() {
            let p = e.path();
            if p.extension().map_or(false, |x| x == "mapinfo") {
                let map = dir.join(p.file_stem().unwrap()).with_extension("map");
                if let Some(id) = crate::mapcat::read_map_id(&map) { if map.is_file() { out.insert(id, map); } }
            }
        }
        out
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hms-h4-stress-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    /// An editor session over a variant file: the exact state the GUI / host keep.
    struct Session {
        path: PathBuf,
        cache: std::sync::Arc<H4Cache>,
        pal: ForgePalette,
        mvar_pal: Vec<H4PaletteEntry>,
        variant: H4Variant,
        set: H4EditorSet,
        live: Vec<ObjectInfo>,
        next_add: u32,
        /// datum -> (quota, variant, object type) every live object must be on disk AS.
        want_type: HashMap<u32, (u8, Option<u8>, u8)>,
    }

    impl Session {
        fn open(path: &Path, cache: std::sync::Arc<H4Cache>) -> Session {
            let pal = crate::h4::palette::palette(&cache);
            let mvar_pal = load_forge_palette(&cache);
            let variant = parse_h4_variant(path).expect("parse");
            let set = variant_to_editor(&cache, &mvar_pal, &variant);
            let live = set.objects.clone();
            let want_type = live.iter().map(|o| { let m = &set.meta[&o.datum]; (o.datum, (m.quota, m.variant, m.object_type)) }).collect();
            Session { path: path.to_path_buf(), cache, pal, mvar_pal, variant, set, live, next_add: NEW_DATUM_BASE, want_type }
        }

        fn build(&self) -> (Vec<H4PlacedObject>, H4SaveReport) {
            let pal = &self.pal;
            let ty = |q: u8, v: Option<u8>| palette_object_type(pal, q, v);
            build_h4_save_list_core(&self.set.src, &self.live, &[], &self.set.unresolved, &self.set.meta, &self.set.colors,
                &H4GateInputs { palette: &self.mvar_pal, bounds: self.variant.bounds, palette_type: Some(&ty) })
        }

        /// Save to `self.path` (a scratch copy), returning the list written + its final slots.
        fn save(&self) -> (Vec<H4PlacedObject>, Vec<u16>, H4SaveReport) {
            let (list, report) = self.build();
            assert_eq!(report.unsaveable, 0, "an object lost its palette entry");
            assert!(!report.blocks_save(), "gate: {}", report.warnings());
            let edits = H4HeaderEdits { budget_spent: report.budget_edit(), ..Default::default() };
            let slots = save_objects(&self.path, &self.path, &list, Some(&edits)).expect("save failed");
            (list, slots, report)
        }

        fn reopen(&mut self) {
            let fresh = Session::open(&self.path, self.cache.clone());
            self.variant = fresh.variant; self.set = fresh.set; self.live = fresh.live; self.want_type = fresh.want_type;
        }

        fn delete(&mut self, datum: u32) {
            self.live.retain(|o| o.datum != datum);
            self.set.meta.remove(&datum);
            self.set.colors.remove(&datum);
            self.want_type.remove(&datum);
        }

        /// A new object from the palette (like `place`): the record is instantiated at SAVE time,
        /// the editor only keeps the meta + colours + the live pose.
        fn add(&mut self, quota: u8, variant: Option<u8>, pos: [f32; 3]) -> u32 {
            let d = self.next_add;
            self.next_add += 1;
            let rec = instantiate_record(&self.cache, &self.pal, quota, variant, pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("palette entry");
            let (obje, mode_tag, name) = resolve_palette_item(&self.cache, &self.mvar_pal, quota, variant).expect("resolves");
            self.live.push(ObjectInfo { datum: d, type_sig: 0, sig0: 0, sig1: 0, pos, health: 1.0, shield: 1.0, mode_tag, fwd: rec.fwd, up: rec.up, attached: [0; 8], primary_tag: h4_tag(obje), variant_name_sid: 0 });
            self.set.meta.insert(d, H4ObjMeta::from_record(&rec, &name, &self.set.labels));
            self.want_type.insert(d, (quota, variant, rec.object_type));
            d
        }

        /// Duplicate a live object (like `dup`): meta cloned, slot = new, colours cloned.
        fn dup(&mut self, datum: u32, off: [f32; 3]) -> u32 {
            let d = self.next_add;
            self.next_add += 1;
            let mut o = self.live.iter().find(|o| o.datum == datum).cloned().expect("live");
            o.datum = d;
            for k in 0..3 { o.pos[k] += off[k]; }
            self.live.push(o);
            let mut m = self.set.meta[&datum].clone();
            m.slot = NEW_SLOT;
            self.set.meta.insert(d, m);
            if let Some(c) = self.set.colors.get(&datum).copied() { self.set.colors.insert(d, c); }
            self.want_type.insert(d, self.want_type[&datum]);
            d
        }

        fn live_mut(&mut self, datum: u32) -> &mut ObjectInfo { self.live.iter_mut().find(|o| o.datum == datum).expect("live") }

        /// ★ Every live object must be ON DISK as the thing it actually IS (multiset of
        /// (position, quota, variant, type) - a stack can share a position).
        fn assert_types_on_disk(&self, what: &str) {
            let disk = parse_h4_variant(&self.path).expect("reparse").objects;
            let key = |p: [f32; 3]| ((p[0] * 10.0).round() as i64, (p[1] * 10.0).round() as i64, (p[2] * 10.0).round() as i64);
            let mut want: HashMap<((i64, i64, i64), (u8, Option<u8>, u8)), i32> = Default::default();
            // the file always stores an EXPLICIT variant index (None -> 0)
            for o in &self.live { let (q, v, t) = self.want_type[&o.datum]; *want.entry((key(o.pos), (q, Some(v.unwrap_or(0)), t))).or_default() += 1; }
            let mut got: HashMap<((i64, i64, i64), (u8, Option<u8>, u8)), i32> = Default::default();
            for d in &disk { assert!(d.variant.is_some(), "{what}: slot {} stores NO variant index", d.slot); *got.entry((key(d.pos), (d.quota.unwrap_or(0), d.variant, d.object_type))).or_default() += 1; }
            let mut wrong = 0;
            let mut first = String::new();
            for (k, n) in &want {
                let have = got.get(k).copied().unwrap_or(0);
                if have < *n {
                    wrong += *n - have;
                    if first.is_empty() {
                        let at: Vec<_> = disk.iter().filter(|d| key(d.pos) == k.0).map(|d| (d.quota, d.variant, d.object_type)).collect();
                        first = format!("expected {:?} at {:?}; the file has {:?} there", k.1, k.0, at);
                    }
                }
            }
            assert_eq!(wrong, 0, "{what}: {wrong} of the editor's objects are missing or written as the WRONG TYPE - {first}");
            assert_eq!(disk.len(), self.live.len() + self.set.unresolved.len(), "{what}: object count on disk");
        }
    }

    /// Every field of two records equal (position within the quantiser's step, basis exact -
    /// the list already holds decoded bases).
    fn assert_same(a: &H4PlacedObject, e: &H4PlacedObject, v: &H4Variant, what: &str) {
        assert_eq!(a.flags, e.flags, "{what}: slot flags");
        assert_eq!(a.quota, e.quota, "{what}: quota");
        assert_eq!(a.variant, e.variant, "{what}: variant");
        assert_eq!(a.in_bounds, e.in_bounds, "{what}: in_bounds");
        let q = e.quantized_pos(v);
        for k in 0..3 { assert!((a.pos[k] - q[k]).abs() < 1e-3, "{what}: pos[{k}] {} vs quantised {}", a.pos[k], q[k]); }
        assert_eq!((a.up_is_global, a.up_quant, a.forward_angle_q), (e.up_is_global, e.up_quant, e.forward_angle_q), "{what}: orientation bits");
        for k in 0..3 { assert!((a.fwd[k] - e.fwd[k]).abs() < 1e-5 && (a.up[k] - e.up[k]).abs() < 1e-5, "{what}: basis"); }
        assert_eq!(a.scale_q, e.scale_q, "{what}: scale");
        assert_eq!(a.locked, e.locked, "{what}: locked");
        assert_eq!(a.shape, e.shape, "{what}: shape");
        assert_eq!(&a.shape_values[..a.shape.value_count()], &e.shape_values[..e.shape.value_count()], "{what}: shape values");
        assert_eq!(a.object_type, e.object_type, "{what}: object type");
        assert_eq!(a.placement, e.placement, "{what}: placement");
        assert_eq!(a.team, e.team, "{what}: team");
        assert_eq!(a.spawn_time, e.spawn_time, "{what}: spawn time");
        assert_eq!(a.color, e.color, "{what}: colour");
        assert_eq!(a.spawn_sequence, e.spawn_sequence, "{what}: spawn sequence");
        assert_eq!(a.user_data, e.user_data, "{what}: user data");
        assert_eq!(a.labels, e.labels, "{what}: labels");
        assert_eq!(a.type_data, e.type_data, "{what}: type data");
    }

    /// Up to `n` shipped variants on distinct maps whose cache is installed: (variant, cache).
    fn samples(n: usize) -> Vec<(PathBuf, PathBuf)> {
        let maps = map_paths();
        let mut seen: BTreeMap<u32, PathBuf> = BTreeMap::new();
        for f in all_variant_files() {
            let Some(id) = crate::h4::mvar::read_h4_map_id(&f) else { continue };
            if seen.contains_key(&id) { continue; }
            if let Some(m) = maps.get(&id) { seen.insert(id, f.clone()); let _ = m; }
            if seen.len() >= n { break; }
        }
        seen.into_iter().map(|(id, f)| (f, maps[&id].clone())).collect()
    }

    fn settler() -> Option<(PathBuf, std::sync::Arc<H4Cache>)> {
        let map = maps_dir()?.join("ca_forge_ravine.map");
        if !map.is_file() { eprintln!("skip: {} missing", map.display()); return None; }
        let vp = all_variant_files().into_iter().find(|p| p.file_name().map_or(false, |n| n == "ca_forge_ravine_settler.mvar"))?;
        Some((vp, std::sync::Arc::new(H4Cache::open(&map).expect("open"))))
    }

    /// Gate 4a: five shipped variants (distinct maps) through the editor records and back with
    /// no edits are BYTE-IDENTICAL, and the gates are silent on shipped content.
    #[test]
    fn shipped_variants_round_trip_byte_identical_through_the_editor() {
        let samples = samples(5);
        if samples.is_empty() { eprintln!("skip: no halo4 variants / maps"); return; }
        for (vp, map) in &samples {
            let cache = std::sync::Arc::new(H4Cache::open(map).expect("open"));
            let s = Session::open(vp, cache);
            assert_eq!(s.live.len() + s.set.unresolved.len(), s.variant.objects.len(), "{}: every record is live or unresolved", vp.display());
            let (list, report) = s.build();
            assert_eq!(report.unsaveable, 0);
            assert!(report.out_of_bounds.is_empty(), "{}: shipped objects out of bounds {:?}", vp.display(), report.out_of_bounds);
            // shipped files DO carry a few records whose type disagrees with the palette (2v2_abandon:
            // 7 dropped weapons / grenades stored as type 0) - the gate is a WARNING, never a refusal
            if !report.type_mismatches.is_empty() { eprintln!("{}: {} shipped record(s) typed unlike their palette entry: {:?}", vp.display(), report.type_mismatches.len(), report.type_mismatches); }
            assert_eq!(report.budget_spent, s.variant.budget_spent, "{}: budget", vp.display());
            assert_eq!(report.over_slot_cap, 0, "{}", vp.display());
            let data = std::fs::read(vp).unwrap();
            let edits = H4HeaderEdits { budget_spent: report.budget_edit(), ..Default::default() };
            let (re, _) = rebuild_h4_file(&data, &list, Some(&edits)).expect("encode");
            assert!(re == data, "{}: re-encode through the editor differs (first byte {:?})", vp.display(), data.iter().zip(&re).position(|(a, b)| a != b));
            eprintln!("{}: {} objects ({} unresolved, {} markers) byte-identical, budget {}", vp.file_name().unwrap().to_string_lossy(), s.live.len(), s.set.unresolved.len(), s.set.stats.skipped_no_model, report.budget_spent);
        }
    }

    /// Gate 4b: every editor verb on Settler - add (a modelled entry AND a model-less spawn
    /// zone), delete (including a PARENT), move, rotate, team / colour, four labels, shape +
    /// values, spawn order / time, placement bits, user data, dup - then save, reparse and
    /// assert EVERY field of EVERY object; a second save is byte-identical; reopen + no-edit
    /// save is byte-identical too.
    #[test]
    fn edits_survive_save_reparse_and_resave_is_a_fixed_point() {
        let Some((vp, cache)) = settler() else { return };
        let work = scratch("settler-edit.mvar");
        std::fs::copy(&vp, &work).unwrap();
        let mut s = Session::open(&work, cache);
        assert_eq!(s.live.len(), 388);
        let pal = s.pal.clone();
        let spawn = pal.find("sp_respawn_point").expect("sp_respawn_point in the Ravine palette");
        // a marker-family entry (respawn zone, type 17 with a boundary): every Ravine palette
        // entry has a render model (0 marker-cube items), so the model-less path is only covered
        // by the edit_scene marker tests; this one proves a zone record survives with its shape
        let zone = pal.find("sp_respawn_zone").expect("sp_respawn_zone in the Ravine palette");
        // --- parent: make object A a child of B, delete B later (a shipped parent if one exists)
        let (a_datum, b_datum) = {
            let shipped = s.live.iter().filter(|o| s.set.meta[&o.datum].parent >= 0).find_map(|o| {
                let p = s.set.meta[&o.datum].parent as u16;
                s.live.iter().find(|q| s.set.src[&q.datum].slot == p).map(|q| (o.datum, q.datum))
            });
            match shipped {
                Some(ab) => { eprintln!("settler: using a shipped parent link {:#x} -> {:#x}", ab.0, ab.1); ab }
                None => { let (a, b) = (s.live[5].datum, s.live[6].datum); let bslot = s.set.src[&b].slot as i16; s.set.meta.get_mut(&a).unwrap().parent = bslot; (a, b) }
            }
        };
        // pick distinct victims for the other edits, clear of the parent pair
        let pick = |s: &Session, from: usize, taken: &[u32]| s.live.iter().skip(from).map(|o| o.datum).find(|d| !taken.contains(d)).unwrap();
        let kept_child = pick(&s, 10, &[a_datum, b_datum]); // stays a child of a kept parent
        let kept_parent = pick(&s, 11, &[a_datum, b_datum, kept_child]);
        let mut taken = vec![a_datum, b_datum, kept_child, kept_parent];
        s.set.meta.get_mut(&kept_child).unwrap().parent = s.set.src[&kept_parent].slot as i16;
        // --- add: a respawn point and a model-less zone (survives even though it has no mesh)
        let placed = s.add(spawn.0, (spawn.1 > 0).then_some(spawn.1), [-25.0, -38.0, 1.0]);
        let zone_d = s.add(zone.0, (zone.1 > 0).then_some(zone.1), [-22.0, -38.0, 1.0]);
        // --- move + rotate the placed object, rotate a shipped one about X (tilted up)
        s.live_mut(placed).pos = [-20.0, -38.0, 1.0];
        let q = glam::Quat::from_rotation_z(45f32.to_radians());
        { let o = s.live_mut(placed); o.fwd = (q * glam::Vec3::from(o.fwd)).into(); o.up = (q * glam::Vec3::from(o.up)).into(); }
        let tilted = pick(&s, 20, &taken); taken.push(tilted);
        let qx = glam::Quat::from_rotation_x(30f32.to_radians());
        let tilted_up: [f32; 3] = (qx * glam::Vec3::from(s.set.src[&tilted].up)).into();
        { let o = s.live_mut(tilted); o.fwd = (qx * glam::Vec3::from(o.fwd)).into(); o.up = tilted_up; }
        let moved = pick(&s, 30, &taken); taken.push(moved);
        { let o = s.live_mut(moved); o.pos[0] += 3.0; o.pos[2] += 1.0; }
        // --- team / colour through `colors` (the way the GUI edits them)
        let recol = pick(&s, 40, &taken); taken.push(recol);
        s.set.colors.insert(recol, (2, 5));
        let uncol = pick(&s, 41, &taken); taken.push(uncol);
        s.set.colors.insert(uncol, (0xFF, 0xFF)); // team none, colour none
        // --- meta edits: labels x4, shape + values, spawn sequence / time, placement, user data, scale, locked
        let edited = pick(&s, 50, &taken); taken.push(edited);
        {
            let m = s.set.meta.get_mut(&edited).unwrap();
            m.label_idx = 0; m.h4.labels_extra = [Some(1), None, Some(2)];
            m.boundary_shape = 3; m.h4.shape_values = [shape_wu_to_raw(2.0), shape_wu_to_raw(3.5), shape_wu_to_raw(1.0), shape_wu_to_raw(0.5)];
            m.spawn_sequence = 77; m.spawn_time = 90; m.placement = 0xCC | 0x10; m.h4.placement_hi = 1; m.h4.user_data = -3; m.h4.set_scale(4.75); m.h4.locked = true; // q 30
        }
        let cyl = pick(&s, 51, &taken); taken.push(cyl);
        { let m = s.set.meta.get_mut(&cyl).unwrap(); m.boundary_shape = 2; m.h4.shape_values = [512, 256, 128, 0]; }
        // type tails on real objects of those types when Settler has them
        let weapon = s.live.iter().find(|o| s.set.meta[&o.datum].object_type == 1 && !taken.contains(&o.datum)).map(|o| o.datum);
        if let Some(w) = weapon { s.set.meta.get_mut(&w).unwrap().weapon_clips = 4; taken.push(w); }
        let tele = s.live.iter().find(|o| matches!(s.set.meta[&o.datum].object_type, 13 | 14 | 15) && !taken.contains(&o.datum)).map(|o| o.datum);
        if let Some(t) = tele { let m = s.set.meta.get_mut(&t).unwrap(); m.tele_channel = 9; m.tele_passability = 2; taken.push(t); }
        // --- dup the placed object, then delete parent B and a shipped object
        let dupd = s.dup(placed, [2.0, 0.0, 0.0]);
        s.delete(b_datum);
        let gone = pick(&s, 100, &taken);
        s.delete(gone);
        assert_eq!(s.live.len(), 388 + 2 + 1 - 2);

        // --- save, reparse, assert every field of every object
        let (list, slots, report) = s.save();
        assert!(report.warnings().is_empty(), "{}", report.warnings());
        let disk = parse_h4_variant(&work).expect("reparse");
        assert_eq!(disk.objects.len(), list.len());
        let by_slot: HashMap<u16, &H4PlacedObject> = disk.objects.iter().map(|o| (o.slot, o)).collect();
        for (i, e) in list.iter().enumerate() {
            let a = by_slot.get(&slots[i]).unwrap_or_else(|| panic!("object {i} landed nowhere (slot {})", slots[i]));
            assert_same(a, e, &disk, &format!("object {i} slot {}", slots[i]));
        }
        s.assert_types_on_disk("after edits");
        // the intended state, spelled out (not just "equals the list")
        let live_idx = |d: u32| s.live.iter().position(|o| o.datum == d).unwrap();
        let on_disk = |d: u32| by_slot[&slots[live_idx(d)]];
        let p = on_disk(placed);
        assert_eq!((p.quota, p.object_type), (Some(spawn.0), 16), "placed respawn point type");
        for k in 0..3 { assert!((p.pos[k] - [-20.0, -38.0, 1.0][k]).abs() < 0.01, "placed pos within quantisation"); }
        let yaw = p.fwd[1].atan2(p.fwd[0]);
        assert!((yaw - 45f32.to_radians()).abs() < 0.02, "placed yaw {yaw}");
        assert!((p.up[2] - 1.0).abs() < 1e-4 && p.up_is_global);
        let z = on_disk(zone_d);
        assert_eq!((z.quota, z.variant, z.object_type), (Some(zone.0), Some(zone.1), 17), "respawn zone kept its entry + type (explicit variant index, #h4-grab-ingame)");
        assert!(z.shape != H4Shape::None && z.shape_values[0] > 0, "the zone's default boundary from the MP block survives: {:?} {:?}", z.shape, z.shape_values);
        let d = on_disk(dupd);
        assert_eq!((d.quota, d.object_type), (Some(spawn.0), 16));
        assert!((d.pos[0] - -18.0).abs() < 0.01 && (d.fwd[1].atan2(d.fwd[0]) - 45f32.to_radians()).abs() < 0.02, "dup keeps the pose + offset");
        let t = on_disk(tilted);
        assert!(!t.up_is_global && (0..3).all(|k| (t.up[k] - tilted_up[k]).abs() < 0.02), "tilted up vector {:?} vs {:?}", t.up, tilted_up);
        let m = on_disk(moved);
        let src = &s.set.src[&moved];
        assert!((m.pos[0] - (src.pos[0] + 3.0)).abs() < 0.01 && (m.pos[2] - (src.pos[2] + 1.0)).abs() < 0.01);
        assert_eq!((m.up_is_global, m.up_quant, m.forward_angle_q), (src.up_is_global, src.up_quant, src.forward_angle_q), "a move keeps the orientation bits verbatim");
        assert_eq!((on_disk(recol).team, on_disk(recol).color), (2, Some(5)));
        assert_eq!((on_disk(uncol).team, on_disk(uncol).color), (-1, None));
        let e = on_disk(edited);
        assert_eq!(e.labels, [Some(0), Some(1), None, Some(2)]);
        assert_eq!((e.shape, e.shape_values), (H4Shape::Box, [512, 896, 256, 128]));
        assert_eq!((e.spawn_sequence, e.spawn_time, e.placement, e.user_data, e.scale_q, e.locked), (77, 90, 0x1DC, -3, 30, true));
        assert!((e.scale() - 4.758).abs() < 1e-3, "#h4-scale the saved scale reads back through the engine's dequantiser");
        let c = on_disk(cyl);
        assert_eq!((c.shape, &c.shape_values[..3]), (H4Shape::Cylinder, &[512u16, 256, 128][..]));
        if let Some(w) = weapon { assert_eq!(on_disk(w).type_data, H4TypeData::Byte(4)); }
        if let Some(t) = tele { assert_eq!(on_disk(t).type_data, H4TypeData::Pair(9, 2)); }
        assert_eq!(on_disk(a_datum).parent, -1, "the child of a DELETED parent is orphaned");
        assert_eq!(on_disk(kept_child).parent, on_disk(kept_parent).slot as i16, "a kept parent link follows the parent's slot");
        assert!(by_slot.values().all(|o| o.parent < 0 || (o.parent as usize) >= H4_SLOTS || by_slot.contains_key(&(o.parent as u16))), "no dangling parent");
        let n_placed_type = disk.objects.iter().filter(|o| o.quota == Some(spawn.0)).count();
        assert_eq!(n_placed_type, s.variant.objects.iter().filter(|o| o.quota == Some(spawn.0)).count() + 2, "two more respawn points than shipped");
        assert_eq!(disk.budget_spent, report.budget_spent);
        assert_eq!(disk.quotas[spawn.0 as usize].placed as usize, n_placed_type, "quota table recomputed");

        // --- a second save from the same editor state is byte-identical
        let bytes1 = std::fs::read(&work).unwrap();
        s.save();
        assert_eq!(std::fs::read(&work).unwrap(), bytes1, "second save from the same state differs");
        // --- reopen, save with no edits: byte-identical; every object still as intended
        s.reopen();
        assert_eq!(s.live.len(), 389);
        s.save();
        assert_eq!(std::fs::read(&work).unwrap(), bytes1, "reopen + no-edit save differs");
        s.assert_types_on_disk("after reopen");
        // the ORIGINAL is untouched
        assert_eq!(parse_h4_variant(&vp).unwrap().objects.len(), 388);
    }

    /// Gate 1 + 3 + 2: an object placed far above the map is REPORTED and kept (the encoder
    /// clamps it; `clamp_into_bounds` = "save anyway" makes the editor agree), a record whose
    /// type disagrees with its palette entry is caught, and the slot / quota / budget caps fire.
    #[test]
    fn gates_report_out_of_bounds_wrong_type_and_caps() {
        let Some((vp, cache)) = settler() else { return };
        let work = scratch("settler-gates.mvar");
        std::fs::copy(&vp, &work).unwrap();
        let mut s = Session::open(&work, cache);
        let spawn = s.pal.find("sp_respawn_point").unwrap();
        let high = s.add(spawn.0, (spawn.1 > 0).then_some(spawn.1), [0.0, 0.0, 2000.0]);
        let (list, report) = s.build();
        assert_eq!(report.out_of_bounds, vec![high]);
        assert_eq!(list.last().unwrap().pos, [0.0, 0.0, 2000.0], "the list is NOT clamped by the builder");
        assert!(report.warnings().contains("outside the variant bounds"));
        assert!(!report.blocks_save());
        // save anyway: the file holds the clamped position and the object is not lost
        let (_, slots, _) = s.save();
        let disk = parse_h4_variant(&work).unwrap();
        let on = disk.objects.iter().find(|o| o.slot == *slots.last().unwrap()).expect("kept");
        assert!(on.pos[2] <= s.variant.bounds[5] && on.pos[2] > s.variant.bounds[5] - 1.0, "clamped to the top {:?}", on.pos);
        assert_eq!(on.quota, Some(spawn.0));
        let moved = clamp_into_bounds(&mut s.live, s.variant.bounds);
        assert_eq!(moved, vec![high]);
        assert!(!is_out_of_bounds(s.live.last().unwrap().pos, s.variant.bounds));
        s.delete(high);
        // type gate: a source record claiming the wrong type for its palette entry
        let victim = s.live[3].datum;
        let want = s.set.src[&victim].object_type;
        s.set.src.get_mut(&victim).unwrap().object_type = if want == 0 { 16 } else { 0 };
        s.set.meta.get_mut(&victim).unwrap().h4.object_type = if want == 0 { 16 } else { 0 };
        let (_, report) = s.build();
        assert_eq!(report.type_mismatches.len(), 1);
        assert_eq!((report.type_mismatches[0].0, report.type_mismatches[0].2), (victim, want));
        s.set.src.get_mut(&victim).unwrap().object_type = want;
        s.set.meta.get_mut(&victim).unwrap().h4.object_type = want;
        assert!(s.build().1.type_mismatches.is_empty());
        // a palette entry placed past its "Maximum Allowed" is NOT a gate: the game spawns the
        // extras anyway, so the report only ever carries the price sum for the header field
        let limited = s.pal.entries.iter().find(|e| e.max_allowed > 0 && e.max_allowed < 8 && e.variants.iter().any(|v| v.tag.is_some())).map(|e| e.quota);
        if let Some(q) = limited {
            let max = s.pal.entry(q).unwrap().max_allowed as usize;
            let already = s.live.iter().filter(|o| s.set.meta[&o.datum].quota == q).count();
            let mut added = Vec::new();
            for k in 0..(max + 1).saturating_sub(already) { added.push(s.add(q, None, [k as f32, 0.0, 1.0])); }
            let (_, report) = s.build();
            assert!(!report.blocks_save() && report.warnings().is_empty(), "{:?}", report);
            for d in added { s.delete(d); }
        } else { eprintln!("no limited palette entry on Ravine - the quota maximum is not exercised"); }
        // the budget is likewise not a gate: a list costing far more than the variant's stored
        // maximum saves, and the maximum written to the file is raised to cover the spend
        s.variant.budget_max = 1;
        let (_, report) = s.build();
        assert!(report.budget_spent > 1 && !report.blocks_save() && report.warnings().is_empty(), "{:?}", report);
        assert_eq!(budget_max_for(1, report.budget_spent), report.budget_spent);
        assert_eq!(budget_max_for(report.budget_spent + 500, report.budget_spent), report.budget_spent + 500);
        // slot cap: fill past 651 with the cheapest entry - this one DOES block (the encoder refuses)
        let cheapest = s.pal.entries.iter().filter(|e| e.variants.iter().any(|v| v.tag.is_some())).min_by_key(|e| e.price).unwrap().quota;
        let n0 = s.live.len();
        for k in 0..(H4_SLOTS + 3 - n0) { s.add(cheapest, None, [k as f32 * 0.1, 1.0, 1.0]); }
        let (list, report) = s.build();
        assert_eq!(list.len(), H4_SLOTS + 3);
        assert_eq!(report.over_slot_cap, 3);
        assert!(report.blocks_save());
        assert!(save_objects(&work, &scratch("overflow.mvar"), &list, None).is_err(), "the encoder refuses 654 objects");
    }
}
