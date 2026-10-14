//! Halo 3 editor records: the map variant's placed objects <-> editor objects, and the save
//! list. #h3-edit
//!
//! The shared editor surface (`hms_ipc::ObjectInfo` + main.rs `ObjMeta`) carries the pose and
//! the few shared fields (team, spawn time, type, boundary); everything else of a Halo 3 record
//! (`mvar::H3VariantObject`: slot flags, engine flags, shared storage, parent, the raw quanta)
//! stays in the datum's SOURCE record (`App.h3_edit.src`) and is copied back on save, so an
//! untouched object re-encodes byte-identical and an edited one changes only what was edited.
//!
//!   * `variant_to_editor` - the loaded variant's placed objects -> `ObjectInfo`s (datum
//!     `0xD000_0000 + record index`, tags encoded with `H3_TAG_FLAG`, the marker cube for a
//!     model-less entry), per-datum `H3ObjMeta` seeds, the source records, and the records that
//!     ride through a save verbatim (the scenario-owned slots without a placement, and any placed
//!     record whose quota does not resolve);
//!   * `build_h3_save_list` - the Reach `build_save_list` for Halo 3: per live object its source
//!     record (or a fresh `palette::instantiate` for a new placement) with the pose re-quantised
//!     ONLY when it moved / turned, team + spawn time from the editor, a slot for every new
//!     record (first free slot past the scenario-owned ones, 640 max), the kept records
//!     appended; the report says what was left out.

use std::collections::HashMap;

use glam::Vec3;
use hms_ipc::ObjectInfo;

use super::cache::H3Cache;
use super::mvar::{quota_tag, H3Variant, H3VariantObject, H3_SLOTS};
use super::objects::model_of;
use super::palette::{instantiate, item_of_quota, H3PaletteItem};

/// Editor tag encoding: a Halo 3 cache tag index is a valid 0, so `ObjectInfo.mode_tag` /
/// `primary_tag` carry `H3_TAG_FLAG | index` (0 stays "none"). Distinct from the Halo 4 flag
/// (0x4800_0000) so a stale value can never be mistaken for the other game's.
pub const H3_TAG_FLAG: u32 = 0x3300_0000;
#[inline] pub fn h3_tag(idx: usize) -> u32 { H3_TAG_FLAG | (idx as u32 & 0xFFFF) }
#[inline] pub fn h3_tag_index(tag: u32) -> Option<usize> { (tag & 0xFFFF_0000 == H3_TAG_FLAG).then_some((tag & 0xFFFF) as usize) }
/// The marker cube's mode tag (an object with no render model still places and saves).
pub const H3_MARKER_TAG: u32 = H3_TAG_FLAG | 0xFFFF;
/// Datum of the variant's record `i`.
pub const LOADED_DATUM_BASE: u32 = 0xD000_0000;
/// Slot of a record not yet saved.
pub const NEW_SLOT: usize = usize::MAX;

/// The shared per-object seed of a Halo 3 record (what `ObjMeta` copies its fields from).
#[derive(Clone, Debug, PartialEq)]
pub struct H3ObjMeta {
    pub name: String,
    pub quota: i32,
    pub team: u8,
    pub object_type: u8,
    pub spawn_time: u8,
    pub boundary_shape: u8,
    /// Raw 16-bit shape quanta (mvar.rs: /65535 * 60 wu).
    pub boundary_q: [u16; 4],
    pub slot: usize,
}

impl H3ObjMeta {
    pub fn from_record(o: &H3VariantObject, name: &str) -> Self {
        H3ObjMeta { name: name.to_string(), quota: o.quota, team: o.team, object_type: o.object_type, spawn_time: o.spawn_time,
            boundary_shape: o.boundary_shape, boundary_q: o.boundary_q, slot: o.slot }
    }
}

/// The loaded variant as editor state.
#[derive(Default)]
pub struct H3EditorSet {
    pub objects: Vec<ObjectInfo>,
    /// datum -> (team, colour 0xFF = none): Halo 3 has no per-object colour.
    pub colors: HashMap<u32, (u8, u8)>,
    pub meta: HashMap<u32, H3ObjMeta>,
    pub src: HashMap<u32, H3VariantObject>,
    /// Records that ride through a save verbatim: scenario-owned slots without a placement, and
    /// placed records whose quota resolves to no tag.
    pub kept: Vec<H3VariantObject>,
    pub unresolved: usize,
    pub marker_only: usize,
}

/// The variant's records -> editor objects (see the module doc).
pub fn variant_to_editor(c: &H3Cache, items: &[H3PaletteItem], v: &H3Variant) -> H3EditorSet {
    let mut set = H3EditorSet::default();
    for (i, o) in v.objects.iter().enumerate() {
        if !o.placed { set.kept.push(o.clone()); continue; }
        let Some(tag) = v.quota_of(o).and_then(|q| quota_tag(c, q)) else { set.unresolved += 1; set.kept.push(o.clone()); continue };
        let datum = LOADED_DATUM_BASE.wrapping_add(i as u32);
        let mode = model_of(c, tag);
        if mode.is_none() { set.marker_only += 1; }
        let name = v.quota_of(o).and_then(|q| item_of_quota(items, q)).map(|k| items[k].name.clone())
            .unwrap_or_else(|| c.tag_name(tag).rsplit('\\').next().unwrap_or("").to_string());
        set.objects.push(ObjectInfo {
            datum, type_sig: 0, sig0: 0, sig1: 0, pos: o.pos, health: 1.0, shield: 1.0,
            mode_tag: mode.map(h3_tag).unwrap_or(H3_MARKER_TAG), fwd: o.fwd, up: o.up, attached: [0; 8],
            primary_tag: h3_tag(tag), variant_name_sid: 0,
        });
        set.colors.insert(datum, (o.team, 0xFF));
        set.meta.insert(datum, H3ObjMeta::from_record(o, &name));
        set.src.insert(datum, o.clone());
    }
    set
}

/// What a save could not write.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H3SaveReport {
    /// Live objects with no source record and no palette row (never written).
    pub unsaveable: usize,
    /// Objects past the 640-slot cap (never written).
    pub over_slot_cap: usize,
    /// Objects outside the variant bounds (written clamped: the quantiser clamps).
    pub out_of_bounds: Vec<u32>,
    /// Quotas appended for rows the source file's table lacked.
    pub quotas_added: usize,
}

/// Per-live-object editor fields the save copies onto the record.
#[derive(Clone, Copy, Debug)]
pub struct H3LiveMeta {
    pub team: u8,
    pub spawn_time: u8,
    pub boundary_shape: u8,
    pub boundary_q: [u16; 4],
    /// Palette row of a NEW placement (None for a loaded record).
    pub palette_item: Option<usize>,
}

/// Build the record list a save writes (see the module doc). `v` is the SOURCE variant (its
/// quota table may grow: the returned variant carries the table to write).
pub fn build_h3_save_list(
    v: &H3Variant,
    items: &[H3PaletteItem],
    src: &HashMap<u32, H3VariantObject>,
    live: &[ObjectInfo],
    extra: &[ObjectInfo],
    kept: &[H3VariantObject],
    meta: &HashMap<u32, H3LiveMeta>,
) -> (H3Variant, Vec<H3VariantObject>, H3SaveReport) {
    let mut out_v = v.clone();
    let mut report = H3SaveReport::default();
    let bounds = v.bounds;
    let mut list: Vec<H3VariantObject> = Vec::new();
    let mut used: Vec<bool> = vec![false; H3_SLOTS];
    for k in kept { if k.slot < H3_SLOTS { used[k.slot] = true; } }
    for o in live.iter().chain(extra.iter()) {
        let m = meta.get(&o.datum);
        let mut rec = match src.get(&o.datum) {
            Some(r) => r.clone(),
            None => {
                // a new placement: its palette row's quota (appended when the table lacks it)
                let Some(it) = m.and_then(|m| m.palette_item).and_then(|i| items.get(i)) else { report.unsaveable += 1; continue };
                let qi = match out_v.quotas.iter().position(|q| q.definition == it.definition) {
                    Some(q) => q as i32,
                    None => {
                        if out_v.quotas.len() >= super::mvar::H3_QUOTA_SLOTS { report.unsaveable += 1; continue; }
                        out_v.quotas.push(it.quota());
                        report.quotas_added += 1;
                        (out_v.quotas.len() - 1) as i32
                    }
                };
                instantiate(it, qi, NEW_SLOT, o.pos, bounds)
            }
        };
        // the pose: re-quantised only when the editor moved / turned it
        let (pos, fwd, up) = (o.pos, Vec3::from(o.fwd), Vec3::from(o.up));
        let moved = (0..3).any(|k| (rec.pos[k] - pos[k]).abs() > 1e-6) || (Vec3::from(rec.fwd) - fwd).length() > 1e-6 || (Vec3::from(rec.up) - up).length() > 1e-6;
        if moved {
            if (0..3).any(|k| pos[k] < bounds[2 * k] || pos[k] > bounds[2 * k + 1]) { report.out_of_bounds.push(o.datum); }
            rec.set_pose(pos, fwd, up, bounds);
        }
        if let Some(m) = m {
            rec.team = m.team;
            rec.spawn_time = m.spawn_time;
            if m.boundary_shape != rec.boundary_shape || m.boundary_q != rec.boundary_q {
                rec.boundary_shape = m.boundary_shape;
                rec.boundary_q = m.boundary_q;
                for k in 0..4 { rec.boundary[k] = super::mvar::dequantize_real(m.boundary_q[k] as u32, 0.0, super::mvar::H3_BOUNDARY_MAX, 16, false); }
            }
        }
        if rec.slot < H3_SLOTS && !used[rec.slot] { used[rec.slot] = true; } else { rec.slot = NEW_SLOT; }
        list.push(rec);
    }
    // slots for the new records: the first free ones past the scenario-owned block
    let first = v.num_scenario_objects as usize;
    let mut next = first;
    for rec in list.iter_mut().filter(|r| r.slot == NEW_SLOT) {
        while next < H3_SLOTS && used[next] { next += 1; }
        if next >= H3_SLOTS { report.over_slot_cap += 1; continue; }
        rec.slot = next;
        used[next] = true;
    }
    list.retain(|r| r.slot != NEW_SLOT);
    list.extend(kept.iter().cloned());
    list.sort_by_key(|r| r.slot);
    (out_v, list, report)
}

#[cfg(test)]
mod tests {
    use super::super::cache::{maps_dir, H3Cache};
    use super::super::mvar::{parse_h3_variant, shipped_variants};
    use super::*;

    /// MLG Guardian -> editor objects -> a no-edit save list: every placed record comes back
    /// EXACTLY (the editor round trip is the identity), so the file re-encodes byte-identical
    /// through the editor path too; then one object moved through the editor lists changes only
    /// that record's pose.
    #[test]
    fn editor_round_trip_is_identity_and_moves_one() {
        let Some(c) = maps_dir().map(|d| d.join("guardian.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let Some(f) = shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { return };
        let v = parse_h3_variant(&f).unwrap();
        let items = super::super::palette::palette(&c);
        let set = variant_to_editor(&c, &items, &v);
        assert_eq!(set.unresolved, 0);
        assert_eq!(set.objects.len() + set.kept.len(), v.objects.len());
        assert!(set.objects.len() > 30);
        let meta: HashMap<u32, H3LiveMeta> = set.meta.iter().map(|(&d, m)| (d, H3LiveMeta { team: m.team, spawn_time: m.spawn_time, boundary_shape: m.boundary_shape, boundary_q: m.boundary_q, palette_item: None })).collect();
        let (v2, list, report) = build_h3_save_list(&v, &items, &set.src, &set.objects, &[], &set.kept, &meta);
        assert_eq!(report, H3SaveReport::default());
        assert_eq!(list, v.objects, "no-edit save list is the file's record list");
        let data = crate::mvar::read_expanded(&f).unwrap();
        assert_eq!(super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap(), data);
        // move one
        let mut objs = set.objects.clone();
        objs[0].pos[2] += 1.0;
        let (_, list2, _) = build_h3_save_list(&v, &items, &set.src, &objs, &[], &set.kept, &meta);
        let d0 = objs[0].datum;
        let slot0 = set.src[&d0].slot;
        let changed: Vec<usize> = list2.iter().zip(v.objects.iter()).filter(|(a, b)| a != b).map(|(a, _)| a.slot).collect();
        assert_eq!(changed, vec![slot0]);
        let m = list2.iter().find(|r| r.slot == slot0).unwrap();
        assert!((m.pos[2] - (set.src[&d0].pos[2] + 1.0)).abs() < 2e-3, "{} vs {}", m.pos[2], set.src[&d0].pos[2] + 1.0);
        // a new placement of the first weapon row appends nothing when its quota exists
        let w = items.iter().position(|it| it.category == "Weapons").unwrap();
        let extra = [ObjectInfo { datum: 0xF000_0001, type_sig: 0, sig0: 0, sig1: 0, pos: [1.0, -10.0, 17.0], health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: h3_tag(items[w].tag), variant_name_sid: 0 }];
        let mut meta2 = meta.clone();
        meta2.insert(0xF000_0001, H3LiveMeta { team: 8, spawn_time: 30, boundary_shape: 0, boundary_q: [0; 4], palette_item: Some(w) });
        let (v3, list3, report3) = build_h3_save_list(&v, &items, &set.src, &set.objects, &extra, &set.kept, &meta2);
        assert_eq!(list3.len(), v.objects.len() + 1);
        let added = list3.iter().find(|r| r.flags == super::super::palette::PLACED_FLAGS && !v.objects.iter().any(|o| o.slot == r.slot)).unwrap();
        assert!(added.slot >= v.num_scenario_objects as usize && added.slot < H3_SLOTS);
        assert_eq!(v3.quotas[added.quota as usize].definition, items[w].definition);
        assert_eq!(report3.unsaveable, 0);
        eprintln!("editor round trip: {} objects, {} kept; moved slot {slot0}; new weapon in slot {} quota {} ({} quotas added)", set.objects.len(), set.kept.len(), added.slot, added.quota, report3.quotas_added);
    }

    /// #h3-place The BLANK CANVAS a bare Halo 3 map now gets: one of the map's shipped variants
    /// is adopted for its bounds and its scenario-owned slots only (`kept`), with no source
    /// records at all, so the save list is those slots plus the fresh placements -- and the file
    /// that comes out re-reads as exactly those objects, where they were put.
    #[test]
    fn a_blank_canvas_saves_only_the_new_placements() {
        let Some(c) = maps_dir().map(|d| d.join("guardian.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let Some(f) = shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { return };
        let v = parse_h3_variant(&f).unwrap();
        let items = super::super::palette::palette(&c);
        // what `h3_adopt_template` keeps: the slots the canvas carries WITHOUT a placement
        let kept: Vec<H3VariantObject> = v.objects.iter().filter(|o| !o.placed).cloned().collect();
        assert!(!kept.is_empty(), "the canvas has scenario-owned slots to carry");
        let row = items.iter().position(|it| it.category == "Weapons").unwrap();
        let at = [2.5f32, -3.5, 1.5];
        let d = 0xD800_0000u32;
        let live = [ObjectInfo { datum: d, type_sig: 0, sig0: 0, sig1: 0, pos: at, health: 1.0, shield: 1.0, mode_tag: 0,
            fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: h3_tag(items[row].tag), variant_name_sid: 0 }];
        let mut meta: HashMap<u32, H3LiveMeta> = HashMap::new();
        meta.insert(d, H3LiveMeta { team: super::super::palette::TEAM_NEUTRAL, spawn_time: 30, boundary_shape: 0, boundary_q: [0; 4], palette_item: Some(row) });
        let (v2, list, report) = build_h3_save_list(&v, &items, &HashMap::new(), &live, &[], &kept, &meta);
        assert_eq!(report.unsaveable, 0);
        assert_eq!(report.over_slot_cap, 0);
        assert!(report.out_of_bounds.is_empty());
        assert_eq!(list.len(), kept.len() + 1, "the canvas's own Forge objects are NOT carried over");
        assert_eq!(list.iter().filter(|r| r.placed).count(), 1);
        // and the written file reads back as that one object, where it was put
        let data = crate::mvar::read_expanded(&f).unwrap();
        let bytes = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
        let (body, _, _) = super::super::mvar::blf_h3_mvar_chunk(&bytes).unwrap();
        let back = super::super::mvar::parse_h3_payload(body).unwrap();
        assert_eq!(back.placed().count(), 1);
        let o = back.placed().next().unwrap();
        for k in 0..3 { assert!((o.pos[k] - at[k]).abs() < 5e-3, "axis {k}: {} vs {}", o.pos[k], at[k]); }
        assert_eq!(back.quotas[o.quota as usize].definition, items[row].definition);
        eprintln!("blank canvas: {} kept slots + 1 placement -> {} records, {} bytes", kept.len(), list.len(), bytes.len());
    }
}
