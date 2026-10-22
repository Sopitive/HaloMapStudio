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
use super::scenario::{is_editable, scenario_datum, scenario_layout, SLOT_DEAD_MASK, SLOT_DELETED, SLOT_SCENARIO};

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
    /// Spare clips / teleporter channel / spawn order, by object type (see `StorageRole`).
    pub shared_storage: u8,
    /// The record's multiplayer flags byte (symmetry is bits 2-3; the rest rides through a save).
    pub mp_flags: u8,
}

impl H3ObjMeta {
    pub fn from_record(o: &H3VariantObject, name: &str) -> Self {
        H3ObjMeta { name: name.to_string(), quota: o.quota, team: o.team, object_type: o.object_type, spawn_time: o.spawn_time,
            boundary_shape: o.boundary_shape, boundary_q: o.boundary_q, slot: o.slot, shared_storage: o.shared_storage, mp_flags: o.mp_flags }
    }
}

/// What a record's `shared_storage` byte means, which depends on the multiplayer object type.
/// The names are the Forge menu's own property keys (halo3.dll .rdata 0x77B008..0x77B188:
/// `spare_clips`, `teleporter_channel`, `spawn_order`), and the object-type numbering is the tag
/// enum, checked against the palette names (1 weapon, 2 grenade, 9-11 teleporters, 12+ spawns
/// and objectives). A type with no role stores 0 and has nothing to edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageRole { None, SpareClips, TeleporterChannel, SpawnOrder }

impl StorageRole {
    pub fn of(object_type: u8) -> Self {
        match object_type {
            1 | 2 => StorageRole::SpareClips,
            9..=11 => StorageRole::TeleporterChannel,
            12..=23 => StorageRole::SpawnOrder,
            _ => StorageRole::None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            StorageRole::None => "", StorageRole::SpareClips => "spare clips",
            StorageRole::TeleporterChannel => "teleporter channel", StorageRole::SpawnOrder => "spawn order",
        }
    }
}

/// The editor metadata of a NEW placement of `it`: the palette row's multiplayer defaults, in the
/// same quanta `palette::instantiate` writes into the record. The boundary quanta matter: an
/// editor metadata that disagrees with the record on them OVERWRITES the record's boundary at
/// save time, and a zero here saved every fresh respawn zone / hill / kill zone at size zero.
pub fn new_placement_meta(it: &H3PaletteItem) -> H3ObjMeta {
    let mp = it.mp.unwrap_or_default();
    H3ObjMeta {
        name: it.name.clone(), quota: -1, team: super::palette::TEAM_NEUTRAL, object_type: mp.object_type,
        spawn_time: mp.spawn_time.clamp(0, 255) as u8, boundary_shape: mp.boundary_shape,
        boundary_q: super::mvar::quantize_boundary(mp.boundary),
        slot: NEW_SLOT, shared_storage: 0, mp_flags: super::palette::PLACED_MP_FLAGS,
    }
}

/// The Forge cost of a record list: the quota price of every slot that carries a Forge placement
/// (scenario-owned slots are free, moved or not). The same rule the budget meter and the save
/// check read.
pub fn forge_cost(quotas: &[super::mvar::H3Quota], records: &[H3VariantObject]) -> f32 {
    records.iter().filter(|r| r.placed && r.flags & SLOT_SCENARIO == 0)
        .filter_map(|r| usize::try_from(r.quota).ok().and_then(|q| quotas.get(q)))
        .map(|q| q.price).sum()
}

/// (used, capacity) of the slots Forge placements can take: the variant holds `H3_SLOTS` and the
/// scenario owns the first `scenario` of them. `used` is the editor's objects plus the kept
/// records that sit past the scenario block (placed records HMS could not resolve still occupy a
/// slot).
pub fn slot_budget(scenario: usize, editor_objects: usize, kept: &[H3VariantObject]) -> (usize, usize) {
    (editor_objects + kept.iter().filter(|r| r.slot >= scenario).count(), H3_SLOTS.saturating_sub(scenario))
}

/// `slot_budget` for the editor's live objects (their datums): the base map's own objects keep the
/// scenario block's slots whether or not they still exist, so only the Forge ones count as used.
pub fn forge_slots_used(scenario: usize, live_datums: impl Iterator<Item = u32>, kept: &[H3VariantObject]) -> (usize, usize) {
    slot_budget(scenario, live_datums.filter(|&d| !super::scenario::is_scenario_datum(d)).count(), kept)
}

/// A record's 16-bit boundary quantum in world units (mvar.rs: `q / 65535 * 60`).
pub fn boundary_wu(q: u16) -> f32 { super::mvar::dequantize_real(q as u32, 0.0, super::mvar::H3_BOUNDARY_MAX, 16, false) }
/// World units -> the quantum a record stores (clamped to [0, 60] wu).
pub fn boundary_q_of(wu: f32) -> u16 { super::mvar::quantize_boundary([wu; 4])[0] }

/// The two symmetry bits of the multiplayer flags (`0x04` | `0x08`; both set = "both"). The engine
/// treats the pair as two roles a gametype needs one object of each for (sub_1800BC2BC), so which
/// bit is which is a naming question only; HMS calls `0x04` symmetric and `0x08` asymmetric, by the
/// order the Forge menu lists them.
pub const MP_SYMMETRY_MASK: u8 = 0x0C;

/// A scenario-owned slot the editor holds (scenario.rs): the base map's own object. Its editor
/// object is live while it exists; one that is no longer live was DELETED, and the save writes its
/// record back flagged deleted (halo3.dll sub_1800BADF0).
#[derive(Clone, Debug)]
pub struct ScnEntry {
    pub datum: u32,
    /// The slot's record as the file (or blank canvas) has it. Unplaced = it stands where the
    /// scenario put it.
    pub rec: H3VariantObject,
    /// The scenario's own pose (position, forward, up): where an unplaced record stands.
    pub rest: ([f32; 3], [f32; 3], [f32; 3]),
    /// The placed record an edit turns it into (`ScenarioSlot::promoted`).
    pub promote: H3VariantObject,
}

/// The loaded variant as editor state.
#[derive(Default)]
pub struct H3EditorSet {
    pub objects: Vec<ObjectInfo>,
    /// datum -> (team, colour 0xFF = none): Halo 3 has no per-object colour.
    pub colors: HashMap<u32, (u8, u8)>,
    pub meta: HashMap<u32, H3ObjMeta>,
    pub src: HashMap<u32, H3VariantObject>,
    /// Records that ride through a save verbatim: scenario slots the editor does not own and placed
    /// records whose quota resolves to no tag.
    pub kept: Vec<H3VariantObject>,
    /// The scenario-owned slots (also in `objects` / `src` unless deleted).
    pub scn: Vec<ScnEntry>,
    pub unresolved: usize,
    pub marker_only: usize,
}

/// The variant's records -> editor objects (see the module doc). When the variant's scenario
/// slots are editable (scenario.rs `is_editable`) every managed one becomes an editor object at the
/// scenario's pose, or at its record's when the slot carries a placement; a slot flagged deleted
/// is kept as a `ScnEntry` with no object.
pub fn variant_to_editor(c: &H3Cache, items: &[H3PaletteItem], v: &H3Variant) -> H3EditorSet {
    let mut set = H3EditorSet::default();
    let layout = scenario_layout(c);
    let editable = is_editable(v, &layout);
    for (i, o) in v.objects.iter().enumerate() {
        if editable {
            match layout.get(o.slot).filter(|s| s.managed && s.palette_tag.is_some()) {
                Some(s) => {
                    let tag = s.palette_tag.unwrap();
                    let datum = scenario_datum(o.slot);
                    let rest = s.pose();
                    let promote = s.promoted(c, &super::scenario::pristine(o), v.bounds);
                    let dead = o.flags & SLOT_DELETED != 0;
                    set.scn.push(ScnEntry { datum, rec: o.clone(), rest, promote: promote.clone() });
                    if dead { continue; }
                    let mode = model_of(c, tag);
                    if mode.is_none() { set.marker_only += 1; }
                    let name = v.quota_of(o).and_then(|q| item_of_quota(items, q)).map(|k| items[k].name.clone())
                        .unwrap_or_else(|| c.tag_name(tag).rsplit('\\').next().unwrap_or("").to_string());
                    let (pos, fwd, up) = if o.placed { (o.pos, o.fwd, o.up) } else { rest };
                    set.objects.push(ObjectInfo {
                        datum, type_sig: 0, sig0: 0, sig1: 0, pos, health: 1.0, shield: 1.0,
                        mode_tag: mode.map(h3_tag).unwrap_or(H3_MARKER_TAG), fwd, up, attached: [0; 8],
                        primary_tag: h3_tag(tag), variant_name_sid: 0,
                    });
                    // an unplaced slot has no multiplayer properties of its own: the panel shows
                    // what an edit would write (its promotion), a placed one shows its record
                    let shown = if o.placed { o } else { &promote };
                    let mut m = H3ObjMeta::from_record(shown, &name);
                    m.quota = o.quota;
                    m.slot = o.slot;
                    set.colors.insert(datum, (shown.team, 0xFF));
                    set.meta.insert(datum, m);
                    set.src.insert(datum, o.clone());
                    continue;
                }
                None if o.slot < v.num_scenario_objects as usize => { set.kept.push(o.clone()); continue; }
                None => {}
            }
        }
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
    /// Quotas holding more objects than their `maximum_allowed` (quota index, count, allowed).
    /// 328 of the 333 shipped variants respect it, so it is a real Forge rule.
    pub over_quota: Vec<(usize, u32, u8)>,
    /// (spent, maximum) when the written objects cost more than the variant's budget (0 of the
    /// 333 shipped variants do; a maximum of 0 means no budget and is never reported).
    pub over_budget: Option<(f32, f32)>,
    /// Scenario-owned objects this save writes as deleted / as placed elsewhere (or edited).
    pub scenario_deleted: usize,
    pub scenario_moved: usize,
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
    /// The editor's `shared_storage` byte; None leaves the record's (a type with no editable role).
    pub shared_storage: Option<u8>,
    /// The editor's symmetry bits (`MP_SYMMETRY_MASK`); None leaves the record's flags alone. Only
    /// those two bits are ever taken, so the rest of the byte rides through a save as read.
    pub symmetry: Option<u8>,
}

/// What Forge refuses a placement for, in the order the engine checks (halo3.dll sub_1800BBBD0: a
/// free slot at or past the scenario block, the quota's `maximum_allowed`; the Forge menu: the
/// budget). `forge_used` counts the Forge objects in the variant, `same_row` the live objects of
/// the palette row (base-map ones included: the engine counts by quota), `spent` / `max_budget`
/// the budget meter. None = the placement is allowed. The message names the limit and its numbers.
pub fn placement_refusal(it: &H3PaletteItem, nso: usize, forge_used: usize, same_row: usize, spent: f32, max_budget: f32) -> Option<String> {
    let cap = H3_SLOTS.saturating_sub(nso);
    if forge_used >= cap {
        return Some(if nso == 0 {
            format!("Object limit: a Halo 3 map variant holds {H3_SLOTS} objects and all {cap} slots are in use.")
        } else {
            format!("Object limit: a Halo 3 map variant holds {H3_SLOTS} object slots and this map's own {nso} objects keep the first {nso} of them (deleting one does not free its slot), leaving {cap} for Forge objects; all {cap} are in use.")
        });
    }
    if it.max_allowed > 0 && same_row >= it.max_allowed as usize {
        return Some(format!("'{}': Forge allows at most {} of these on this map and {} are placed.", it.name, it.max_allowed, same_row));
    }
    if max_budget > 0.0 && it.price > 0.0 && spent + it.price > max_budget {
        return Some(format!("Over budget: '{}' costs {:.0} and only {:.0} of the {:.0} budget is left.", it.name, it.price, (max_budget - spent).max(0.0), max_budget));
    }
    None
}

/// The editor's per-object fields onto a record: team, spawn time, the role-dependent shared byte,
/// the symmetry bits and the boundary (a changed size is re-expressed through its quanta).
fn apply_live_meta(rec: &mut H3VariantObject, m: &H3LiveMeta) {
    rec.team = m.team;
    rec.spawn_time = m.spawn_time;
    if let Some(ss) = m.shared_storage { rec.shared_storage = ss; }
    if let Some(sym) = m.symmetry { rec.mp_flags = (rec.mp_flags & !MP_SYMMETRY_MASK) | (sym & MP_SYMMETRY_MASK); }
    if m.boundary_shape != rec.boundary_shape || m.boundary_q != rec.boundary_q {
        rec.boundary_shape = m.boundary_shape;
        rec.boundary_q = m.boundary_q;
        for k in 0..4 { rec.boundary[k] = super::mvar::dequantize_real(m.boundary_q[k] as u32, 0.0, super::mvar::H3_BOUNDARY_MAX, 16, false); }
    }
}

/// Does this record count toward its quota's `placed_on_map`? Every present slot does, except a
/// base-map object that was deleted (sub_1800BA3F0 skips scenario slots with `flags & 0x60`).
pub fn counts_toward_quota(r: &H3VariantObject) -> bool { r.flags & SLOT_SCENARIO == 0 || r.flags & SLOT_DEAD_MASK == 0 }

/// Build the record list a save writes (see the module doc). `v` is the SOURCE variant (its
/// quota table may grow: the returned variant carries the table to write).
pub fn build_h3_save_list(
    v: &H3Variant,
    items: &[H3PaletteItem],
    src: &HashMap<u32, H3VariantObject>,
    live: &[ObjectInfo],
    extra: &[ObjectInfo],
    kept: &[H3VariantObject],
    scn: &[ScnEntry],
    meta: &HashMap<u32, H3LiveMeta>,
    fresh: bool,
) -> (H3Variant, Vec<H3VariantObject>, H3SaveReport) {
    let mut out_v = v.clone();
    let mut report = H3SaveReport::default();
    let bounds = v.bounds;
    let mut list: Vec<H3VariantObject> = Vec::new();
    let mut used: Vec<bool> = vec![false; H3_SLOTS];
    for k in kept { if k.slot < H3_SLOTS { used[k.slot] = true; } }
    for e in scn { if e.rec.slot < H3_SLOTS { used[e.rec.slot] = true; } }
    let scn_by: HashMap<u32, &ScnEntry> = scn.iter().map(|e| (e.datum, e)).collect();
    let mut live_scn: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for o in live.iter().chain(extra.iter()) {
        let m = meta.get(&o.datum);
        // the base map's own object: its slot is fixed; it is written as it was unless the editor
        // moved or edited it, and then as a record that carries its own placement
        if let Some(e) = scn_by.get(&o.datum) {
            live_scn.insert(o.datum);
            let (rpos, rfwd, rup) = if e.rec.placed { (e.rec.pos, e.rec.fwd, e.rec.up) } else { e.rest };
            let mut rec = if e.rec.placed { e.rec.clone() } else { e.promote.clone() };
            let (pos, fwd, up) = (o.pos, Vec3::from(o.fwd), Vec3::from(o.up));
            let moved = (0..3).any(|k| (rpos[k] - pos[k]).abs() > 1e-6) || (Vec3::from(rfwd) - fwd).length() > 1e-6 || (Vec3::from(rup) - up).length() > 1e-6;
            if moved {
                if (0..3).any(|k| pos[k] < bounds[2 * k] || pos[k] > bounds[2 * k + 1]) { report.out_of_bounds.push(o.datum); }
                rec.set_pose(pos, fwd, up, bounds);
            }
            if let Some(m) = m { apply_live_meta(&mut rec, m); }
            let out = if !e.rec.placed && !moved && rec == e.promote { e.rec.clone() } else { rec };
            if out != e.rec { report.scenario_moved += 1; }
            list.push(out);
            continue;
        }
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
        if let Some(m) = m { apply_live_meta(&mut rec, m); }
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
    // a base-map object the editor no longer has was deleted: its record stays, flagged
    // (sub_1800BADF0 sets 0x20 on a scenario-origin datum and leaves the rest as it was)
    for e in scn.iter().filter(|e| !live_scn.contains(&e.datum)) {
        let mut r = e.rec.clone();
        r.flags |= SLOT_DELETED;
        report.scenario_deleted += 1;
        list.push(r);
    }
    list.extend(kept.iter().cloned());
    list.sort_by_key(|r| r.slot);
    // the quota counts and the budget follow what this save changed (mvar.rs `account_for_save`)
    super::mvar::account_for_save(&mut out_v, &v.objects, &list, fresh);
    let mut per_quota: HashMap<i32, u32> = HashMap::new();
    for r in list.iter().filter(|r| r.quota >= 0 && counts_toward_quota(r)) { *per_quota.entry(r.quota).or_default() += 1; }
    for (i, q) in out_v.quotas.iter().enumerate() {
        let n = *per_quota.get(&(i as i32)).unwrap_or(&0);
        if q.maximum_allowed > 0 && n > q.maximum_allowed as u32 { report.over_quota.push((i, n, q.maximum_allowed)); }
    }
    // judged on what is being WRITTEN, not on the stored figure: Guardian stores 168 against 32 of
    // placed prices, and a stale high base must not raise a false "over budget"
    let cost = forge_cost(&out_v.quotas, &list);
    if out_v.maximum_budget > 0.0 && cost > out_v.maximum_budget { report.over_budget = Some((cost, out_v.maximum_budget)); }
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
        let deleted = set.scn.iter().filter(|e| e.rec.flags & SLOT_DELETED != 0).count();
        assert_eq!(set.objects.len() + set.kept.len() + deleted, v.objects.len());
        assert!(set.objects.len() > 30);
        let meta: HashMap<u32, H3LiveMeta> = set.meta.iter().map(|(&d, m)| (d, H3LiveMeta { team: m.team, spawn_time: m.spawn_time, boundary_shape: m.boundary_shape, boundary_q: m.boundary_q, palette_item: None, shared_storage: None, symmetry: None })).collect();
        let (v2, list, report) = build_h3_save_list(&v, &items, &set.src, &set.objects, &[], &set.kept, &set.scn, &meta, false);
        assert_eq!(report, H3SaveReport { scenario_deleted: deleted, ..Default::default() });
        assert_eq!(list, v.objects, "no-edit save list is the file's record list");
        let data = crate::mvar::read_expanded(&f).unwrap();
        assert_eq!(super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap(), data);
        // move one
        let mut objs = set.objects.clone();
        // a Forge object (the base map's own are moved by the scenario gates)
        let k0 = objs.iter().position(|o| !super::super::scenario::is_scenario_datum(o.datum)).expect("a Forge object");
        objs.swap(0, k0);
        objs[0].pos[2] += 1.0;
        let (_, list2, _) = build_h3_save_list(&v, &items, &set.src, &objs, &[], &set.kept, &set.scn, &meta, false);
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
        meta2.insert(0xF000_0001, H3LiveMeta { team: 8, spawn_time: 30, boundary_shape: 0, boundary_q: [0; 4], palette_item: Some(w), shared_storage: None, symmetry: None });
        let (v3, list3, report3) = build_h3_save_list(&v, &items, &set.src, &set.objects, &extra, &set.kept, &set.scn, &meta2, false);
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
        meta.insert(d, H3LiveMeta { team: super::super::palette::TEAM_NEUTRAL, spawn_time: 30, boundary_shape: 0, boundary_q: [0; 4], palette_item: Some(row), shared_storage: None, symmetry: None });
        let (v2, list, report) = build_h3_save_list(&v, &items, &HashMap::new(), &live, &[], &kept, &[], &meta, true);
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
        // the canvas's stored counts and budget described objects that were dropped: they are
        // recounted, so the file says what it holds (one priced weapon) and not the template's
        assert_eq!(back.spent_budget, items[row].price, "template budget {} survived a blank-canvas save", v.spent_budget);
        let same_quota_kept = kept.iter().filter(|k| k.quota == o.quota).count();
        assert_eq!(back.quotas[o.quota as usize].placed_on_map as usize, 1 + same_quota_kept, "the weapon's quota count was not recounted");
        assert!(v.spent_budget != items[row].price, "the template must carry a different stored budget for this gate to mean anything");
        eprintln!("blank canvas: {} kept slots + 1 placement -> {} records, {} bytes", kept.len(), list.len(), bytes.len());
    }

    // ---- #h3-save: the editor path as production runs it -----------------------------------

    /// Every shipped variant whose base map is installed, with that map's cache and palette.
    /// Returns how many it visited, so a gate can say whether it ran at all.
    fn each_installed_variant(mut f: impl FnMut(&std::path::Path, &H3Cache, &[H3PaletteItem], &H3Variant)) -> usize {
        let cands: Vec<crate::mapcat::MapCandidate> = crate::mapcat::enumerate().into_iter().filter(|c| c.game == crate::mapcat::Game::Halo3).collect();
        let mut opened: HashMap<u32, (H3Cache, Vec<H3PaletteItem>)> = HashMap::new();
        let mut n = 0;
        for file in shipped_variants() {
            let Ok(v) = parse_h3_variant(&file) else { continue };
            let Some(c) = cands.iter().find(|c| c.map_id == Some(v.map_id)) else { continue };
            if !opened.contains_key(&v.map_id) {
                let Ok(cache) = H3Cache::open(&c.path) else { continue };
                let items = super::super::palette::palette(&cache);
                opened.insert(v.map_id, (cache, items));
            }
            let (cache, items) = &opened[&v.map_id];
            f(&file, cache, items, &v);
            n += 1;
        }
        n
    }

    /// The live metadata a save reads, built the way the app builds it: record -> `H3ObjMeta` ->
    /// `ObjMeta` (`obj_meta_from_h3`) -> `edit` (what the properties panel does) -> `live_meta_of`.
    fn live_metas(set: &H3EditorSet, mut edit: impl FnMut(u32, &mut crate::ObjMeta)) -> HashMap<u32, H3LiveMeta> {
        set.meta.iter().map(|(&d, m)| {
            let mut om = super::super::app::obj_meta_from_h3(m, set.src[&d].pos);
            edit(d, &mut om);
            (d, super::super::app::live_meta_of(&om, m.team, None))
        }).collect()
    }

    /// THE gate for "variants can be written": for every shipped variant (333 on a full install),
    /// load it into the editor, change nothing, save through the production metadata path, and
    /// the file comes back byte-identical. This is the check that fails when any field of the
    /// editor's per-object model (shared storage, symmetry, boundary, team, spawn time) or the
    /// quota / budget accounting is not an identity on an untouched object.
    #[test]
    fn every_shipped_variant_survives_the_editor_path_byte_identical() {
        let mut bad: Vec<String> = Vec::new();
        let (mut objects, mut base_map) = (0usize, 0usize);
        let n = each_installed_variant(|file, cache, items, v| {
            let set = variant_to_editor(cache, items, v);
            objects += set.objects.len();
            base_map += set.scn.len();
            let metas = live_metas(&set, |_, _| {});
            let (v2, list, report) = build_h3_save_list(v, items, &set.src, &set.objects, &[], &set.kept, &set.scn, &metas, false);
            let data = crate::mvar::read_expanded(file).unwrap();
            let out = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
            let clean = report.unsaveable == 0 && report.over_slot_cap == 0 && report.out_of_bounds.is_empty() && report.quotas_added == 0;
            if list != v.objects || out != data || !clean {
                bad.push(format!("{}: records {} bytes {} report {:?}", file.file_name().unwrap().to_string_lossy(), list == v.objects, out == data, report));
            }
        });
        eprintln!("EDITOR PATH: {n} variants, {objects} editor objects ({base_map} of them the base map's own slots), {} not byte-identical", bad.len());
        for b in bad.iter().take(8) { eprintln!("  {b}"); }
        assert!(bad.is_empty(), "{} variants change under a no-edit save", bad.len());
        if n > 0 { assert!(objects > 1000, "the sweep ran but saw only {objects} objects"); }
        // the base map's own objects are in this sweep (they were `kept` verbatim before they were editor objects)
        if n > 0 { assert!(base_map > 20_000, "only {base_map} base-map slots went through the editor path"); }
    }

    /// A fresh drop of a SHAPED row (respawn zone, hill, kill zone...) saves the size the tag
    /// authors. The editor's metadata used to carry zeroed boundary quanta, which differ from the
    /// record `instantiate` builds, so the save overwrote the default size with zero.
    #[test]
    fn a_new_shaped_placement_saves_the_size_its_tag_authors() {
        let Some(c) = maps_dir().map(|d| d.join("guardian.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let Some(f) = shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { return };
        let v = parse_h3_variant(&f).unwrap();
        let items = super::super::palette::palette(&c);
        let rows: Vec<usize> = (0..items.len()).filter(|&i| items[i].mp.map_or(false, |m| m.boundary_shape != 0 && m.boundary.iter().any(|&b| b > 0.0))).collect();
        assert!(rows.len() >= 3, "guardian's palette has {} shaped rows with a default size", rows.len());
        let kept: Vec<H3VariantObject> = v.objects.iter().filter(|o| !o.placed).cloned().collect();
        for &row in &rows {
            let it = &items[row];
            let m = new_placement_meta(it);
            let om = super::super::app::obj_meta_from_h3(&m, [0.0; 3]);
            let d = 0xD900_0000u32;
            let mut metas = HashMap::new();
            metas.insert(d, super::super::app::live_meta_of(&om, super::super::palette::TEAM_NEUTRAL, Some(row)));
            let live = [ObjectInfo { datum: d, type_sig: 0, sig0: 0, sig1: 0, pos: [1.0, -10.0, 17.0], health: 1.0, shield: 1.0, mode_tag: 0,
                fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: h3_tag(it.tag), variant_name_sid: 0 }];
            let (v2, list, _) = build_h3_save_list(&v, &items, &HashMap::new(), &live, &[], &kept, &[], &metas, true);
            let rec = list.iter().find(|r| r.placed).unwrap();
            let want = instantiate(it, rec.quota, rec.slot, [1.0, -10.0, 17.0], v.bounds);
            assert_eq!(rec.boundary_shape, want.boundary_shape, "{}", it.name);
            assert_eq!(rec.boundary_q, want.boundary_q, "{}: the saved size is not the authored one", it.name);
            assert_ne!(&rec.boundary_q[..1], &[0u16][..], "{}: a shaped object saved at size zero", it.name);
            // and it survives the real encode -> decode
            let data = crate::mvar::read_expanded(&f).unwrap();
            let bytes = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
            let (body, _, _) = super::super::mvar::blf_h3_mvar_chunk(&bytes).unwrap();
            let back = super::super::mvar::parse_h3_payload(body).unwrap();
            let o = back.placed().next().unwrap();
            assert_eq!(o.boundary_shape, it.mp.unwrap().boundary_shape);
            assert!((o.boundary[0] - it.mp.unwrap().boundary[0]).abs() < 0.01, "{}: {} vs {}", it.name, o.boundary[0], it.mp.unwrap().boundary[0]);
        }
        eprintln!("shaped placements checked: {}", rows.len());
    }

    /// The Forge menu's spare clips / teleporter channel / spawn order and the symmetry pair reach
    /// the file; symmetry touches ONLY its two bits (the rest of the flags byte rides through);
    /// and exactly the edited record changes.
    #[test]
    fn storage_and_symmetry_edits_reach_the_record_and_nothing_else() {
        let (mut seen_clips, mut seen_chan, mut seen_order, mut seen_flagged) = (false, false, false, false);
        let n = each_installed_variant(|_, cache, items, v| {
            let set = variant_to_editor(cache, items, v);
            for (&d, m) in &set.meta {
                let want_role = StorageRole::of(m.object_type);
                let rec = &set.src[&d];
                // a base-map object that stands where the scenario put it has no properties of its own to edit in place (the scenario gates cover its promotion)
                if !rec.placed { continue; }
                let flagged = rec.mp_flags & !MP_SYMMETRY_MASK != 0;
                let pick = match want_role {
                    StorageRole::SpareClips if !seen_clips => { seen_clips = true; true }
                    StorageRole::TeleporterChannel if !seen_chan => { seen_chan = true; true }
                    StorageRole::SpawnOrder if !seen_order => { seen_order = true; true }
                    _ if flagged && !seen_flagged => { seen_flagged = true; true }
                    _ => false,
                };
                if !pick { continue; }
                // the panel edit: a new shared byte in the field that names this type's role, and symmetric-only
                let metas = live_metas(&set, |dd, om| {
                    if dd != d { return; }
                    match want_role {
                        StorageRole::SpareClips => om.weapon_clips = 9,
                        StorageRole::TeleporterChannel => om.tele_channel = 9,
                        StorageRole::SpawnOrder => om.spawn_seq = 9,
                        StorageRole::None => {}
                    }
                    om.placement = (om.placement & !MP_SYMMETRY_MASK) | 0x04;
                });
                let (_, list, _) = build_h3_save_list(v, items, &set.src, &set.objects, &[], &set.kept, &set.scn, &metas, false);
                let changed: Vec<&H3VariantObject> = list.iter().zip(v.objects.iter()).filter(|(a, b)| a != b).map(|(a, _)| a).collect();
                let new = list.iter().find(|r| r.slot == rec.slot).unwrap();
                if want_role != StorageRole::None { assert_eq!(new.shared_storage, 9, "{:?} edit did not reach the record", want_role); }
                else { assert_eq!(new.shared_storage, rec.shared_storage, "a type with no role keeps its byte"); }
                assert_eq!(new.mp_flags, (rec.mp_flags & !MP_SYMMETRY_MASK) | 0x04, "symmetry rewrote bits beyond its two");
                assert!(changed.len() <= 1 && changed.iter().all(|r| r.slot == rec.slot), "{} records changed for a one-object edit", changed.len());
                if want_role == StorageRole::None { continue; }
                let mut want = rec.clone();
                want.shared_storage = 9;
                want.mp_flags = (rec.mp_flags & !MP_SYMMETRY_MASK) | 0x04;
                assert_eq!(*new, want, "something besides storage / symmetry moved");
            }
        });
        eprintln!("STORAGE EDITS: {n} variants; clips {seen_clips} channel {seen_chan} order {seen_order} flagged {seen_flagged}");
        if n > 0 { assert!(seen_clips && seen_chan && seen_order && seen_flagged, "the sweep never met every case"); }
    }

    /// What the save check is told: a quota over its `maximum_allowed`, a spent budget over the
    /// maximum, an object outside the bounds, and objects that do not fit. An untouched variant
    /// reports none of those four.
    #[test]
    fn the_save_report_names_what_would_go_wrong() {
        let Some(c) = maps_dir().map(|d| d.join("guardian.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let Some(f) = shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { return };
        let v = parse_h3_variant(&f).unwrap();
        let items = super::super::palette::palette(&c);
        let set = variant_to_editor(&c, &items, &v);
        let metas = live_metas(&set, |_, _| {});
        // untouched: nothing to report
        let (_, _, quiet) = build_h3_save_list(&v, &items, &set.src, &set.objects, &[], &set.kept, &set.scn, &metas, false);
        assert!(quiet.over_quota.is_empty() && quiet.over_budget.is_none() && quiet.out_of_bounds.is_empty() && quiet.over_slot_cap == 0, "{quiet:?}");

        // N fresh placements of one row, with the row's own palette defaults
        let place = |row: usize, n: usize, at: [f32; 3]| -> (Vec<ObjectInfo>, HashMap<u32, H3LiveMeta>) {
            let it = &items[row];
            let mut live = Vec::new();
            let mut m2 = metas.clone();
            for k in 0..n {
                let d = 0xD900_0000u32 + k as u32;
                let om = super::super::app::obj_meta_from_h3(&new_placement_meta(it), at);
                m2.insert(d, super::super::app::live_meta_of(&om, super::super::palette::TEAM_NEUTRAL, Some(row)));
                live.push(ObjectInfo { datum: d, type_sig: 0, sig0: 0, sig1: 0, pos: at, health: 1.0, shield: 1.0, mode_tag: 0, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
                    attached: [0; 8], primary_tag: h3_tag(it.tag), variant_name_sid: 0 });
            }
            (live, m2)
        };
        let in_bounds = [(v.bounds[0] + v.bounds[1]) * 0.5, (v.bounds[2] + v.bounds[3]) * 0.5, (v.bounds[4] + v.bounds[5]) * 0.5];

        // a quota past its maximum_allowed
        let row = items.iter().position(|it| it.max_allowed > 0 && it.max_allowed < 60).expect("a limited row");
        let allowed = items[row].max_allowed as u32;
        let (live, m2) = place(row, allowed as usize + 1, in_bounds);
        let (v2, _, r) = build_h3_save_list(&v, &items, &set.src, &set.objects, &live, &set.kept, &set.scn, &m2, false);
        let qi = v2.quotas.iter().position(|q| q.definition == items[row].definition).unwrap();
        let hit = r.over_quota.iter().find(|&&(i, _, _)| i == qi).unwrap_or_else(|| panic!("{} placements of a row allowing {allowed} were not reported: {:?}", allowed + 1, r.over_quota));
        assert!(hit.1 > allowed && hit.2 as u32 == allowed, "{hit:?}");
        let (live, m2) = place(row, 1, in_bounds);
        let (_, _, r1) = build_h3_save_list(&v, &items, &set.src, &set.objects, &live, &set.kept, &set.scn, &m2, false);
        assert!(!r1.over_quota.iter().any(|&(i, _, _)| i == qi) || v.quotas.get(qi).map_or(false, |q| q.placed_on_map as u32 >= allowed), "one placement must not trip the limit it fits");

        // a budget overspent: shrink the variant's maximum so a priced placement exceeds it
        let priced = items.iter().position(|it| it.price > 0.0).expect("a priced row");
        let mut tight = v.clone();
        tight.maximum_budget = forge_cost(&v.quotas, &v.objects) + items[priced].price * 0.5;
        let (live, m2) = place(priced, 1, in_bounds);
        let (_, _, rb) = build_h3_save_list(&tight, &items, &set.src, &set.objects, &live, &set.kept, &set.scn, &m2, false);
        let (spent, max) = rb.over_budget.expect("a placement past the budget was not reported");
        assert!(spent > max, "{spent} vs {max}");
        let (_, _, rq) = build_h3_save_list(&tight, &items, &set.src, &set.objects, &[], &set.kept, &set.scn, &metas, false);
        assert!(rq.over_budget.is_none(), "no placement, no overspend");

        // an object outside the bounds
        let far = [v.bounds[1] + 500.0, in_bounds[1], in_bounds[2]];
        let (live, m2) = place(row, 1, far);
        let (_, _, ro) = build_h3_save_list(&v, &items, &set.src, &set.objects, &live, &set.kept, &set.scn, &m2, false);
        assert_eq!(ro.out_of_bounds, vec![0xD900_0000], "{ro:?}");

        // more objects than there are slots
        let used = v.objects.len();
        let (live, m2) = place(row, H3_SLOTS - used + 7, in_bounds);
        let (_, list, rs) = build_h3_save_list(&v, &items, &set.src, &set.objects, &live, &set.kept, &set.scn, &m2, false);
        assert!(rs.over_slot_cap >= 7, "{rs:?}");
        assert!(list.len() <= H3_SLOTS, "{} records past the slot cap", list.len());
    }

    fn blank_record(slot: usize) -> H3VariantObject {
        H3VariantObject { slot, flags: 0x83, quota: 0, parent: None, placed: true, pos: [0.0; 3], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
            pos_q: [0; 3], up_is_global: true, up_q: 0, angle_q: 0, boundary_q: [0; 4], object_type: 0, mp_flags: 0x0C,
            game_engine_flags: 0x3FF, shared_storage: 0, spawn_time: 0, team: 8, boundary_shape: 0, boundary: [0.0; 4] }
    }

    /// The saturation rule the writer relies on is real: no shipped variant STORES a spent budget
    /// above its maximum, though 11 of them hold objects whose prices add up to more (their stored
    /// value is exactly the cap). So an over-budget variant is loadable, and "over budget" is an
    /// advisory the save check raises, not a format error.
    #[test]
    fn no_shipped_variant_stores_more_than_its_budget() {
        let (mut stored_over, mut cost_over) = (Vec::<String>::new(), 0usize);
        let n = each_installed_variant(|file, _, _, v| {
            if v.maximum_budget > 0.0 && v.spent_budget > v.maximum_budget { stored_over.push(file.file_name().unwrap().to_string_lossy().into_owned()); }
            if v.maximum_budget > 0.0 && forge_cost(&v.quotas, &v.objects) > v.maximum_budget { cost_over += 1; }
        });
        eprintln!("BUDGET RULE: {n} variants, {} store more than the cap, {cost_over} hold objects priced past it", stored_over.len());
        assert!(stored_over.is_empty(), "{stored_over:?}");
        if n > 0 { assert!(cost_over < n / 10, "{cost_over} of {n} cost more than their cap: the advisory would fire on stock files"); }
    }

    /// The slot counter: 640 slots, the scenario owns the first `scenario` of them, and a kept
    /// record past that block still takes one.
    #[test]
    fn the_slot_budget_is_what_the_scenario_leaves() {
        let mk = |slot: usize| blank_record(slot);
        // 10 and 300 sit inside the scenario's block (slots 0..331); 331 is the first Forge slot
        let kept = [mk(10), mk(300), mk(331), mk(400)];
        assert_eq!(slot_budget(331, 50, &kept), (52, 640 - 331));
        assert_eq!(slot_budget(331, 0, &[]), (0, 309));
        assert_eq!(slot_budget(700, 0, &[]), (0, 0), "a scenario count past the slot count cannot underflow");
    }
}
