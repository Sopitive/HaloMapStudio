//! The Halo 3 map variant's SCENARIO-OWNED slots: the base map's own vehicles, weapons, equipment,
//! scenery and crates (spawn points, Forge crates, ...) that a map variant carries as its first
//! `num_scenario_objects` object slots. #h3-scenario
//!
//! What halo3.dll does with them (all read out of the `c_map_variant` methods, sub_1800B85F4 ..
//! sub_1800BDDA0; the datum is the 0x54-byte `s_variant_object_datum`, `flags` its first u16):
//!   * Slot = `object_type_start_index[object type]` + the placement's index in the scnr block of
//!     its class. The types a variant registers are 1 vehicle, 2 weapon, 3 equipment, 6 scenery and
//!     10 crate (mask 0x44E in sub_1800BA6A4), in that order, so the start indices are the running
//!     totals of the scnr placement counts: the layout is a pure function of the scenario
//!     (`scenario_layout`), and `num_scenario_objects` is its length (checked on 327 of the 333
//!     shipped variants, all 27 maps; the other 6 are the game's empty `default_*` variants, which
//!     register no slots at all).
//!   * Only an object whose definition has a `Multiplayer Object` block gets a datum (sub_1800BD5AC
//!     returns at once without one): 6162 of 6178 placements across the shipped maps.
//!   * flags: 0x01 used, 0x02 EDITED (the slot carries its own placement: position, axes and the
//!     multiplayer properties; sub_1800B8E9C writes the placement bit for a scenario slot only when
//!     this is set, and always for a slot past the scenario block), 0x08 scenario-origin, 0x20
//!     DELETED (sub_1800BADF0 sets it when a scenario-origin object is deleted; a Forge object's
//!     delete zeroes the datum instead, which is the only way a slot goes absent), 0x40 inactive,
//!     0x80 a sandbox-managed object. 0x20 is cleared (with |0x09) when the gametype needs the object
//!     back (sub_1800BC9A8).
//!   * An ABSENT scenario slot is an object the engine never created (no multiplayer block), NOT a
//!     deleted one. A new Forge object takes the first free slot AT OR PAST `num_scenario_objects`
//!     (sub_1800BBBD0, sub_1800BAAF4): deleting a scenario object frees no Forge slot.
//!   * The quota of a scenario object is the quota whose object IS the placement's palette tag:
//!     the sandbox palette entry when the map's sandbox palette has the tag (kinds 1-7), else the
//!     scenario's own palette entry (kinds 8-12, index = the placement's palette index); -1 when
//!     the variant has none (alive scenery with quota -1 exists in 61 shipped records).
//!
//! The editor shows every managed scenario slot as an editor object (datum `scenario_datum`),
//! drawn from the scnr placement until a record places it elsewhere.

use glam::Vec3;

use super::cache::H3Cache;
use super::mvar::{H3Variant, H3VariantObject};
use super::objects::{PALETTE_ELEM, PLACEMENT_TABLES};
use super::palette::mp_defaults;
use crate::h4::cache::ByteRead;

/// Editor datum of scenario slot `s`: `SCENARIO_DATUM_BASE + s` (a Forge record's is 0xD000_0000 +
/// its record index, a fresh placement's 0xF000_0000 +).
pub const SCENARIO_DATUM_BASE: u32 = 0xD400_0000;
#[inline] pub fn scenario_datum(slot: usize) -> u32 { SCENARIO_DATUM_BASE + slot as u32 }
#[inline] pub fn is_scenario_datum(d: u32) -> bool { d & 0xFFFF_F000 == SCENARIO_DATUM_BASE }
#[inline] pub fn scenario_slot(d: u32) -> usize { (d & 0xFFF) as usize }

/// Slot flag bits of `s_variant_object_datum` (module doc).
pub const SLOT_USED: u16 = 0x01;
pub const SLOT_EDITED: u16 = 0x02;
pub const SLOT_SCENARIO: u16 = 0x08;
pub const SLOT_DELETED: u16 = 0x20;
/// `(flags & 0x60) != 0`: sub_1800BA3F0 does not count the scenario object toward its quota.
pub const SLOT_DEAD_MASK: u16 = 0x60;

/// (multiplayer object type, scnr placement class) in slot order.
pub const SCENARIO_CLASSES: [(u8, [u8; 4]); 5] = [(1, *b"vehi"), (2, *b"weap"), (3, *b"eqip"), (6, *b"scen"), (10, *b"bloc")];

/// Offset of the `Multiplayer Data` block (its `Symmetric Placement` u32) in each class's scnr
/// placement element (Halo3MCC scnr.xml): u32 symmetric @0, u16 game engine flags @4, u16 team @6,
/// i8 spawn order @8, i16 spawn time @0xC.
fn mp_base(class: [u8; 4]) -> usize {
    match &class { b"scen" => 0x80, b"vehi" | b"weap" => 0x74, b"eqip" => 0x58, _ => 0x7C }
}

/// The scenario placement's own multiplayer data.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlacementMp {
    /// 0 ignore, 1 symmetric, 2 asymmetric.
    pub symmetric: u32,
    pub engine_flags: u16,
    /// 0 red .. 7 pink, 8 neutral.
    pub team: u16,
    pub spawn_order: u8,
    pub spawn_time: i16,
}

/// One scnr placement of a variant-registered class.
#[derive(Clone, Debug)]
pub struct ScenarioSlot {
    pub slot: usize,
    pub class: [u8; 4],
    /// Index in its class's placement block.
    pub index: usize,
    pub palette_index: i16,
    /// The placement's object definition (obje) tag.
    pub palette_tag: Option<usize>,
    /// The object name from the scnr names block, or empty.
    pub name: String,
    pub pos: [f32; 3],
    /// yaw, pitch, roll radians as stored.
    pub rot: [f32; 3],
    pub scale: f32,
    pub placement_flags: u32,
    /// The definition has a `Multiplayer Object` block: the engine gives it a datum.
    pub managed: bool,
    pub mp: PlacementMp,
}

impl ScenarioSlot {
    pub fn class_str(&self) -> String { String::from_utf8_lossy(&self.class).into_owned() }

    /// The scenario's own pose of the object: position, forward, up.
    pub fn pose(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (f, u) = crate::h4::objects::euler_to_basis(self.rot);
        (self.pos, f.to_array(), u.to_array())
    }

    /// The record this object would carry if the player placed it elsewhere (`base`: the slot's
    /// current record, which supplies flags / quota / parent): placed, flagged EDITED, with the
    /// multiplayer properties the engine initialises a datum with (sub_1800BD5AC) from the
    /// definition's multiplayer block and this placement's own multiplayer data. The pose is the
    /// scenario's, quantised against `bounds`. INFERRED where the runtime struct is not a copy of
    /// the tag: type, boundary shape and the symmetry bits are exact on every shipped moved record
    /// (`gate`), team / engine flags / spawn time agree on the share the module tests state.
    pub fn promoted(&self, c: &H3Cache, base: &H3VariantObject, bounds: [f32; 6]) -> H3VariantObject {
        let mp = self.palette_tag.and_then(|t| mp_defaults(c, t)).unwrap_or_default();
        let mut o = base.clone();
        o.placed = true;
        o.flags |= SLOT_EDITED;
        o.object_type = mp.object_type;
        // the placement's engine flags, else the definition's, else every engine (a zero is "unset"
        // at both levels: measured on 15 441 shipped moved records)
        o.game_engine_flags = if self.mp.engine_flags != 0 { self.mp.engine_flags } else if mp.engine_flags != 0 { mp.engine_flags } else { 0x3FF };
        // sub_1800BD5AC: both symmetry bits, minus the one the placement's symmetric placement excludes
        o.mp_flags = 0x0C & match self.mp.symmetric { 1 => !0x08u8, 2 => !0x04u8, _ => 0xFF };
        o.shared_storage = if o.object_type >= 12 { self.mp.spawn_order } else { 0 };
        let t = if self.mp.spawn_time != 0 { self.mp.spawn_time } else { mp.spawn_time };
        o.spawn_time = if (0..=0xFE).contains(&t) { t as u8 } else { 0 };
        o.team = self.mp.team.min(8) as u8;
        let (pos, fwd, up) = self.pose();
        o.set_pose(pos, Vec3::from(fwd), Vec3::from(up), bounds);
        o.set_boundary(mp.boundary_shape, mp.boundary);
        o
    }
}

/// Every scnr placement of the five registered classes, in slot order.
#[derive(Clone, Debug, Default)]
pub struct ScenarioLayout {
    pub slots: Vec<ScenarioSlot>,
    /// `object_type_start_index` this layout implies (-1 for a type that is not registered).
    pub type_start: [i16; super::mvar::H3_OBJECT_TYPES],
}

impl ScenarioLayout {
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn get(&self, slot: usize) -> Option<&ScenarioSlot> { self.slots.get(slot) }
}

/// The scenario's slot layout (module doc). Empty when the cache has no scenario.
pub fn scenario_layout(c: &H3Cache) -> ScenarioLayout {
    let mut out = ScenarioLayout { slots: Vec::new(), type_start: [-1; super::mvar::H3_OBJECT_TYPES] };
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let d = c.data();
    let names = c.block(sm + super::objects::OFF_SCNR_OBJECT_NAMES);
    for (otype, class) in SCENARIO_CLASSES {
        let Some(t) = PLACEMENT_TABLES.iter().find(|t| t.class == class) else { continue };
        let (np, po) = c.block(sm + t.placements).unwrap_or((0, 0));
        let pal = c.block(sm + t.palette);
        if np > 0x40000 { continue; }
        out.type_start[otype as usize] = out.slots.len() as i16;
        for i in 0..np {
            let e = po + i * t.stride;
            let pi = d.i16_at(e);
            let palette_tag = match pal {
                Some((npal, palo)) if pi >= 0 && (pi as usize) < npal => c.tag_ref(palo + pi as usize * PALETTE_ELEM).map(|(_, tag)| tag),
                _ => None,
            };
            let ni = d.i16_at(e + 2);
            let name = match names {
                Some((nn, no)) if ni >= 0 && (ni as usize) < nn => c.sid(d.u32_at(no + ni as usize * super::objects::OBJECT_NAME_ELEM)),
                _ => String::new(),
            };
            let m = e + mp_base(class);
            let mut scale = d.f32_at(e + 0x20);
            if !(scale > 0.0) || !scale.is_finite() { scale = 1.0; }
            out.slots.push(ScenarioSlot {
                slot: out.slots.len(), class, index: i, palette_index: pi, palette_tag, name,
                pos: [d.f32_at(e + 8), d.f32_at(e + 12), d.f32_at(e + 16)],
                rot: [d.f32_at(e + 0x14), d.f32_at(e + 0x18), d.f32_at(e + 0x1C)],
                scale, placement_flags: d.u32_at(e + 4),
                managed: palette_tag.map_or(false, |t| editor_owns(c, class, t)),
                mp: PlacementMp { symmetric: d.u32_at(m), engine_flags: d.u16_at(m + 4), team: d.u16_at(m + 6), spawn_order: d.u8_at(m + 8), spawn_time: d.i16_at(m + 0xC) },
            });
        }
    }
    out
}

/// Does the editor hold this scnr placement (so the static scenery must not draw it)? It is one of
/// the five registered classes and its definition has a multiplayer block (the same test
/// `scenario_layout` stamps `managed` with). The GUI build leaves exactly these out when the
/// variant's scenario slots are editable.
pub fn editor_owns(c: &H3Cache, class: [u8; 4], palette_tag: usize) -> bool {
    SCENARIO_CLASSES.iter().any(|&(_, cl)| cl == class) && mp_defaults(c, palette_tag).is_some()
}

/// Can the editor own this variant's scenario slots? The variant registers some, the scnr layout is
/// exactly that many, and the stored start indices are the layout's. False for a game `default_*`
/// variant (it registers none) and for any cache the reader does not understand: those keep their
/// objects as static scenery.
pub fn is_editable(v: &H3Variant, layout: &ScenarioLayout) -> bool {
    v.num_scenario_objects > 0
        && layout.len() == v.num_scenario_objects as usize
        && SCENARIO_CLASSES.iter().all(|&(t, _)| v.type_start[t as usize] == layout.type_start[t as usize])
}

/// A slot of the layout that is an editor object: it has a datum in the engine.
pub fn editor_slots<'a>(v: &'a H3Variant, layout: &'a ScenarioLayout) -> impl Iterator<Item = (&'a H3VariantObject, &'a ScenarioSlot)> + 'a {
    v.objects.iter().filter_map(move |r| layout.get(r.slot).filter(|s| s.managed).map(|s| (r, s)))
}

/// The record `rec` would be without its edits: not deleted, not edited (no placement of its own).
/// An untouched scenario record comes back unchanged.
pub fn pristine(rec: &H3VariantObject) -> H3VariantObject {
    if !rec.placed && rec.flags & SLOT_DELETED == 0 { return rec.clone(); }
    H3VariantObject {
        slot: rec.slot, flags: rec.flags & !(SLOT_EDITED | SLOT_DELETED), quota: rec.quota, parent: rec.parent, placed: false,
        pos: [0.0; 3], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], pos_q: [0; 3], up_is_global: true, up_q: 0, angle_q: 0, boundary_q: [0; 4],
        object_type: 0, mp_flags: 0, game_engine_flags: 0, shared_storage: 0, spawn_time: 0, team: 0, boundary_shape: 0, boundary: [0.0; 4],
    }
}

/// A blank canvas from `template`: the same variant with the scenario slots as the scenario made
/// them (nothing deleted, nothing moved) and none of the template author's own Forge objects.
pub fn canvas_of(template: &H3Variant) -> H3Variant {
    let nso = template.num_scenario_objects as usize;
    let mut v = template.clone();
    v.objects = template.objects.iter().filter(|o| o.slot < nso).map(pristine).collect();
    v
}

/// A shipped `.mvar` of map `map_id` to start from, searching the game's variant folders plus
/// `extra_dir`. A variant that registers the map's scenario slots (`num_scenario_objects > 0`) is
/// the canvas that lets the base map's objects be edited; the game's empty `default_*` variants
/// (an 880-byte file with no slots, the smallest of a map's variants) register none and are only a
/// fallback. Among equals the smallest file wins: any of them is the same canvas.
pub fn find_template(map_id: u32, extra_dir: Option<&std::path::Path>) -> Option<std::path::PathBuf> {
    let mut dirs = crate::mapcat::h3_variant_dirs();
    if let Some(d) = extra_dir { dirs.push(d.to_path_buf()); }
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
    let mut fallback: Option<std::path::PathBuf> = None;
    for (_, p) in cands {
        let Ok(v) = super::mvar::parse_h3_variant(&p) else { continue };
        if v.num_scenario_objects > 0 { return Some(p); }
        fallback.get_or_insert(p);
    }
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::edit::{build_h3_save_list, counts_toward_quota, forge_cost, forge_slots_used, placement_refusal, variant_to_editor, H3LiveMeta, H3_MARKER_TAG};
    use super::super::mvar::{parse_h3_variant, read_h3_map_id, shipped_variants, H3_SLOTS};
    use super::super::palette::{palette, H3PaletteItem};
    use hms_ipc::ObjectInfo;
    use std::collections::{BTreeMap, HashMap};

    /// Every installed Halo 3 map that has shipped variants: (map stem, cache, palette, variants).
    /// Returns how many maps it visited so a gate can say whether it ran at all.
    fn each_map(mut f: impl FnMut(&str, &H3Cache, &[H3PaletteItem], &[(std::path::PathBuf, H3Variant)])) -> usize {
        let cands: Vec<crate::mapcat::MapCandidate> = crate::mapcat::enumerate().into_iter().filter(|c| c.game == crate::mapcat::Game::Halo3).collect();
        let mut by_map: HashMap<u32, Vec<(std::path::PathBuf, H3Variant)>> = HashMap::new();
        for file in shipped_variants() {
            if let Some(id) = read_h3_map_id(&file) { if let Ok(v) = parse_h3_variant(&file) { by_map.entry(id).or_default().push((file, v)); } }
        }
        let mut seen = std::collections::HashSet::new();
        let mut n = 0;
        for c in &cands {
            let Some(id) = c.map_id else { continue };
            let Some(vs) = by_map.get(&id) else { continue };
            if !seen.insert(id) { continue; }
            let Ok(cache) = H3Cache::open(&c.path) else { continue };
            let items = palette(&cache);
            f(&c.stem, &cache, &items, vs);
            n += 1;
        }
        n
    }

    fn live_metas(set: &super::super::edit::H3EditorSet) -> HashMap<u32, H3LiveMeta> {
        set.meta.iter().map(|(&d, m)| {
            let pos = set.objects.iter().find(|o| o.datum == d).map(|o| o.pos).unwrap_or_default();
            let om = super::super::app::obj_meta_from_h3(m, pos);
            (d, super::super::app::live_meta_of(&om, m.team, None))
        }).collect()
    }

    /// Scenario slot = start index + placement index, proven on the data: on every shipped variant
    /// the layout's length is `num_scenario_objects`, its start indices are the stored ones, every
    /// present slot's quota names the very object the placement at that slot uses, presence is the
    /// multiplayer-block rule (but for the handful of exceptions), and the scenario flag 0x08 is
    /// set on exactly the slots below the scenario block.
    #[test]
    fn the_scenario_layout_is_the_slot_layout_of_every_shipped_variant() {
        let (mut records, mut checked_tag, mut tag_bad, mut present_managed, mut present_unmanaged, mut absent_managed, mut absent_unmanaged) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut flag_bad, mut editable_maps, mut static_maps) = (0usize, 0usize, 0usize);
        let mut kinds: BTreeMap<(String, u8), usize> = BTreeMap::new();
        let n = each_map(|stem, c, _, vs| {
            let layout = scenario_layout(c);
            for s in &layout.slots {
                if let Some(mp) = s.palette_tag.and_then(|t| mp_defaults(c, t)) { *kinds.entry((s.class_str(), mp.object_type)).or_default() += 1; }
            }
            for (file, v) in vs {
                let nso = v.num_scenario_objects as usize;
                if nso == 0 { static_maps += 1; assert!(!is_editable(v, &layout), "{stem}: a variant with no scenario slots must not be editable"); continue; }
                assert_eq!(layout.len(), nso, "{stem} {}: scnr placements vs num_scenario_objects", file.display());
                assert!(is_editable(v, &layout), "{stem}: type_start {:?} vs layout {:?}", v.type_start, layout.type_start);
                editable_maps += 1;
                let present: std::collections::HashSet<usize> = v.objects.iter().map(|o| o.slot).collect();
                for s in &layout.slots {
                    match (present.contains(&s.slot), s.managed) {
                        (true, true) => present_managed += 1, (true, false) => present_unmanaged += 1,
                        (false, true) => absent_managed += 1, (false, false) => absent_unmanaged += 1,
                    }
                }
                for o in &v.objects {
                    records += 1;
                    if (o.slot < nso) != (o.flags & SLOT_SCENARIO != 0) { flag_bad += 1; }
                    if o.slot >= nso { continue; }
                    let (Some(s), Some(q)) = (layout.get(o.slot), v.quota_of(o)) else { continue };
                    if let (Some(pt), Some(qt)) = (s.palette_tag, super::super::mvar::quota_tag(c, q)) {
                        checked_tag += 1;
                        if pt != qt { tag_bad += 1; }
                    }
                }
            }
        });
        eprintln!("SCENARIO LAYOUT: {n} maps ({editable_maps} editable variants, {static_maps} with none), {records} records, quota==placement tag on {checked_tag} ({tag_bad} bad); presence: managed present {present_managed} absent {absent_managed}, unmanaged present {present_unmanaged} absent {absent_unmanaged}; scenario-flag disagreements {flag_bad}");
        eprintln!("SCENARIO KINDS (class, multiplayer object type) -> placements over {n} maps: {kinds:?}");
        if n == 0 { return; }
        assert!(n >= 10 && checked_tag > 20_000, "the sweep ran on {n} maps / {checked_tag} quota checks: too thin to prove the slot layout");
        assert_eq!(tag_bad, 0, "a slot's quota names another object than its scnr placement: the slot layout is wrong");
        assert_eq!(flag_bad, 0, "the scenario-origin flag 0x08 does not follow the scenario block");
        // the presence rule (multiplayer block => a datum) holds but for midship-style outliers
        assert!(present_unmanaged + absent_managed <= (present_managed + absent_unmanaged) / 100, "presence rule: {present_unmanaged} unmanaged present, {absent_managed} managed absent");
        assert!(absent_unmanaged > 500 && present_managed > 5_000);
    }

    /// The editor holds the base map's own objects at the scenario's own poses: every managed
    /// scenario slot of every variant is an editor object whose pose is the scnr placement's (read
    /// by objects.rs `load_placements`, an independent loader) or its record's, and a deleted slot
    /// is a record only. Also the static scenery no longer draws what the editor owns: the set the
    /// build leaves out equals the set the editor holds.
    #[test]
    fn scenario_objects_are_editor_objects_at_their_scenario_poses() {
        let (mut live, mut at_scenario_pose, mut at_record_pose, mut deleted, mut hidden_ok) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let n = each_map(|stem, c, items, vs| {
            let placements = super::super::objects::load_placements(c);
            // what build_objects leaves to the editor: (class, tag, position)
            let hidden: Vec<(String, usize, [i64; 3])> = placements.iter()
                .filter(|p| editor_owns(c, p.class, p.palette_tag))
                .map(|p| (p.class_str(), p.palette_tag, p.pos.map(|x| (x * 100.0) as i64))).collect();
            for (_, v) in vs {
                let set = variant_to_editor(c, items, v);
                if v.num_scenario_objects == 0 { assert!(set.scn.is_empty() && set.objects.iter().all(|o| !is_scenario_datum(o.datum)), "{stem}"); continue; }
                let layout = scenario_layout(c);
                let own: Vec<&ObjectInfo> = set.objects.iter().filter(|o| is_scenario_datum(o.datum)).collect();
                live += own.len();
                let n_dead = set.scn.iter().filter(|e| e.rec.flags & SLOT_DELETED != 0).count();
                deleted += n_dead;
                assert_eq!(own.len() + n_dead, set.scn.len(), "{stem}: every scenario entry is live or deleted");
                for o in &own {
                    let e = set.scn.iter().find(|e| e.datum == o.datum).unwrap();
                    let s = layout.get(scenario_slot(o.datum)).unwrap();
                    if e.rec.placed {
                        assert_eq!(o.pos, e.rec.pos, "{stem}: a placed slot stands at its record");
                        at_record_pose += 1;
                    } else {
                        assert_eq!(o.pos, s.pose().0, "{stem} slot {}: an unplaced slot stands where the scenario put it", s.slot);
                        at_scenario_pose += 1;
                    }
                }
                // the build's hidden set (class, tag, pos) == the scenario slots of the layout, as a multiset
                let mut a: Vec<(String, usize, [i64; 3])> = layout.slots.iter().filter(|s| s.managed).map(|s| (s.class_str(), s.palette_tag.unwrap(), s.pos.map(|x| (x * 100.0) as i64))).collect();
                let mut b = hidden.clone();
                a.sort(); b.sort();
                assert_eq!(a, b, "{stem}: the static scenery and the editor disagree about which objects the editor owns");
                hidden_ok += 1;
            }
        });
        eprintln!("SCENARIO EDITOR OBJECTS: {n} maps, {live} live ({at_scenario_pose} at the scenario's pose, {at_record_pose} at a record's), {deleted} deleted, hidden-set check on {hidden_ok} variants");
        if n == 0 { return; }
        assert!(live > 10_000 && at_scenario_pose > 5_000 && at_record_pose > 100 && deleted > 100 && hidden_ok > 100);
        let _ = H3_MARKER_TAG;
    }

    /// Untouched, the base map's own objects re-encode byte-identical (the whole-file sweep is the
    /// editor-path gate in edit.rs; this one adds that the scenario entries ARE in it).
    #[test]
    fn untouched_scenario_objects_are_part_of_the_byte_identical_save() {
        let (mut files, mut entries) = (0usize, 0usize);
        each_map(|stem, c, items, vs| {
            for (file, v) in vs {
                let set = variant_to_editor(c, items, v);
                entries += set.scn.len();
                let metas = live_metas(&set);
                let (v2, list, report) = build_h3_save_list(v, items, &set.src, &set.objects, &[], &set.kept, &set.scn, &metas, false);
                let data = crate::mvar::read_expanded(file).unwrap();
                let out = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
                assert!(list == v.objects && out == data, "{stem} {}: an untouched variant changed (scenario entries {}, report {report:?})", file.display(), set.scn.len());
                assert_eq!((report.scenario_deleted, report.scenario_moved), (set.scn.iter().filter(|e| e.rec.flags & SLOT_DELETED != 0).count(), 0), "{stem}: a no-edit save moved something");
                files += 1;
            }
        });
        eprintln!("UNTOUCHED: {files} variants byte-identical with {entries} scenario entries in the editor state");
        if files > 0 { assert!(entries > 20_000, "only {entries} scenario entries: the sweep did not carry the base map's objects"); }
    }

    /// Deleting a base-map object in the editor writes its record back flagged deleted: exactly
    /// that slot changes (flags |= 0x20), the slot stays present, its quota count falls by one when
    /// it counted, the budget and the Forge slot counter do not move.
    #[test]
    fn deleting_a_scenario_object_flags_its_record_and_changes_nothing_else() {
        let (mut tried, mut quota_counted) = (0usize, 0usize);
        each_map(|stem, c, items, vs| {
            // a few variants per map is enough (the sweep above is the whole-file gate)
            for (file, v) in vs.iter().take(3) {
                if v.num_scenario_objects == 0 { continue; }
                let set = variant_to_editor(c, items, v);
                // delete the first alive base-map object of each kind of record: unplaced, placed, with and without a quota
                let mut picks: Vec<u32> = Vec::new();
                for want in [(false, true), (false, false), (true, true)] {
                    if let Some(e) = set.scn.iter().find(|e| e.rec.flags & SLOT_DELETED == 0 && e.rec.placed == want.0 && (e.rec.quota >= 0) == want.1) { picks.push(e.datum); }
                }
                for d in picks {
                    let live: Vec<ObjectInfo> = set.objects.iter().filter(|o| o.datum != d).cloned().collect();
                    let metas = live_metas(&set);
                    let (v2, list, report) = build_h3_save_list(v, items, &set.src, &live, &[], &set.kept, &set.scn, &metas, false);
                    let rec = &set.src[&d];
                    assert_eq!(report.scenario_deleted, set.scn.iter().filter(|e| e.rec.flags & SLOT_DELETED != 0).count() + 1, "{stem}");
                    // exactly the deleted slot differs, and only in its flags
                    let diff: Vec<(&H3VariantObject, &H3VariantObject)> = list.iter().zip(v.objects.iter()).filter(|(a, b)| a != b).collect();
                    assert_eq!(list.len(), v.objects.len(), "{stem}: the slot must stay present (absent means 'never created', not 'deleted')");
                    assert_eq!(diff.len(), 1, "{stem}: {} records changed", diff.len());
                    assert_eq!(diff[0].0.slot, rec.slot);
                    let mut want = rec.clone();
                    want.flags |= SLOT_DELETED;
                    assert_eq!(*diff[0].0, want, "{stem}: more than the flags changed");
                    // the file re-reads with the flag and the same slot count
                    let data = crate::mvar::read_expanded(file).unwrap();
                    let bytes = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
                    let (body, _, _) = super::super::mvar::blf_h3_mvar_chunk(&bytes).unwrap();
                    let back = super::super::mvar::parse_h3_payload(body).unwrap();
                    assert_eq!(back.objects.iter().find(|o| o.slot == rec.slot).map(|o| o.flags), Some(rec.flags | SLOT_DELETED));
                    assert_eq!(back.num_variant_objects, v.num_variant_objects);
                    // accounting: the quota loses one count (the engine does not count a deleted scenario slot), the budget stays
                    if rec.quota >= 0 {
                        let q = rec.quota as usize;
                        let was = v.quotas[q].placed_on_map;
                        assert_eq!(v2.quotas[q].placed_on_map, was.saturating_sub(1), "{stem}: quota {q} count");
                        quota_counted += 1;
                    }
                    assert_eq!(v2.spent_budget, v.spent_budget, "{stem}: deleting a base-map object must not change the budget");
                    assert!(!counts_toward_quota(&want));
                    // and no Forge slot is freed
                    let before = forge_slots_used(v.num_scenario_objects as usize, set.objects.iter().map(|o| o.datum), &set.kept);
                    let after = forge_slots_used(v.num_scenario_objects as usize, live.iter().map(|o| o.datum), &set.kept);
                    assert_eq!(before, after, "{stem}: deleting a scenario object freed a Forge slot");
                    tried += 1;
                }
            }
        });
        eprintln!("DELETE: {tried} scenario deletions checked, {quota_counted} with a quota count");
        if tried > 0 { assert!(tried > 60 && quota_counted > 20); }
    }

    /// Moving a base-map object gives its slot a placement of its own (EDITED, the position
    /// quantised, the multiplayer properties of `promoted`), changes no other record, and the
    /// written file reads back at the new place. Moving a placed one only changes its pose.
    #[test]
    fn moving_a_scenario_object_gives_its_slot_a_placement_and_round_trips() {
        let (mut unplaced_moved, mut placed_moved, mut team_edit) = (0usize, 0usize, 0usize);
        each_map(|stem, c, items, vs| {
            for (file, v) in vs.iter().take(3) {
                if v.num_scenario_objects == 0 { continue; }
                let set = variant_to_editor(c, items, v);
                let metas = live_metas(&set);
                let data = crate::mvar::read_expanded(file).unwrap();
                for want_placed in [false, true] {
                    let Some(e) = set.scn.iter().find(|e| e.rec.flags & SLOT_DELETED == 0 && e.rec.placed == want_placed) else { continue };
                    let mid = [(v.bounds[0] + v.bounds[1]) * 0.5, (v.bounds[2] + v.bounds[3]) * 0.5, (v.bounds[4] + v.bounds[5]) * 0.5];
                    let target = [mid[0] + 3.0, mid[1] - 2.0, mid[2] + 1.0];
                    let live: Vec<ObjectInfo> = set.objects.iter().cloned().map(|mut o| { if o.datum == e.datum { o.pos = target; } o }).collect();
                    let (v2, list, report) = build_h3_save_list(v, items, &set.src, &live, &[], &set.kept, &set.scn, &metas, false);
                    assert!(report.out_of_bounds.is_empty() && report.unsaveable == 0, "{stem}: {report:?}");
                    let diff: Vec<usize> = list.iter().zip(v.objects.iter()).filter(|(a, b)| a != b).map(|(a, _)| a.slot).collect();
                    assert_eq!(diff, vec![e.rec.slot], "{stem}: a one-object move changed {diff:?}");
                    let r = list.iter().find(|r| r.slot == e.rec.slot).unwrap();
                    assert!(r.placed && r.flags & SLOT_EDITED != 0 && r.flags & SLOT_SCENARIO != 0, "{stem}: moved record {:#x} placed {}", r.flags, r.placed);
                    // the base map's own objects are free: giving one a placement spends nothing
                    assert_eq!(v2.spent_budget, v.spent_budget, "{stem}: moving a base-map object changed the budget");
                    // the position is stored in 16 bits over the variant bounds: within one quantum of the target
                    for k in 0..3 { let q = (v.bounds[2 * k + 1] - v.bounds[2 * k]) / 65535.0; assert!((r.pos[k] - target[k]).abs() <= q + 1e-4, "{stem}: axis {k} {} vs {} (quantum {q})", r.pos[k], target[k]); }
                    if !want_placed { assert_eq!((r.object_type, r.team, r.mp_flags, r.game_engine_flags), (e.promote.object_type, e.promote.team, e.promote.mp_flags, e.promote.game_engine_flags), "{stem}: promotion fields"); }
                    let bytes = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
                    let (body, _, _) = super::super::mvar::blf_h3_mvar_chunk(&bytes).unwrap();
                    let back = super::super::mvar::parse_h3_payload(body).unwrap();
                    let b = back.objects.iter().find(|o| o.slot == e.rec.slot).unwrap();
                    assert_eq!(b, r, "{stem}: the moved record did not survive the file");
                    if want_placed { placed_moved += 1 } else { unplaced_moved += 1 }
                    // no move, but a property edit (team): an unplaced slot still needs a placement to carry it
                    if !want_placed && placed_moved + unplaced_moved < 6 {
                        let mut m2 = metas.clone();
                        m2.get_mut(&e.datum).unwrap().team = if e.promote.team == 0 { 1 } else { 0 };
                        let (_, l2, rep2) = build_h3_save_list(v, items, &set.src, &set.objects, &[], &set.kept, &set.scn, &m2, false);
                        let r2 = l2.iter().find(|r| r.slot == e.rec.slot).unwrap();
                        assert!(r2.placed && r2.team != e.promote.team && rep2.scenario_moved == 1, "{stem}: a property edit did not reach the record");
                        team_edit += 1;
                    }
                }
            }
        });
        eprintln!("MOVE: {unplaced_moved} unplaced + {placed_moved} placed base-map objects moved, {team_edit} property edits");
        if unplaced_moved + placed_moved > 0 { assert!(unplaced_moved > 20 && placed_moved > 5 && team_edit > 0); }
    }

    /// How well `promoted` predicts the multiplayer properties of the base map's objects that
    /// shipped variants DID move: exact on type, near-exact on the symmetry bits and team, the
    /// rest printed. (The runtime struct the engine copies from is not the tag: INFERRED.)
    #[test]
    fn promotion_matches_the_shipped_moved_records() {
        let mut agree: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        let n = each_map(|_, c, items, vs| {
            let layout = scenario_layout(c);
            for (_, v) in vs {
                if !is_editable(v, &layout) { continue; }
                for o in v.objects.iter().filter(|o| o.placed && o.slot < v.num_scenario_objects as usize) {
                    let Some(s) = layout.get(o.slot).filter(|s| s.managed) else { continue };
                    let p = s.promoted(c, &pristine(o), v.bounds);
                    let _ = items;
                    for (k, ok) in [("object type", p.object_type == o.object_type), ("shape", p.boundary_shape == o.boundary_shape),
                        ("symmetry bits", p.mp_flags & 0x0C == o.mp_flags & 0x0C), ("team", p.team == o.team),
                        ("engine flags", p.game_engine_flags == o.game_engine_flags), ("spawn time", p.spawn_time == o.spawn_time)] {
                        let e = agree.entry(k).or_default();
                        e.1 += 1;
                        if ok { e.0 += 1; }
                    }
                }
            }
        });
        for (k, (a, t)) in &agree { eprintln!("PROMOTION {k}: {a}/{t} ({:.1}%)", *a as f64 * 100.0 / (*t).max(1) as f64); }
        if n == 0 { return; }
        let pct = |k: &str| { let (a, t) = agree[k]; a as f64 * 100.0 / t as f64 };
        assert!(agree["object type"].1 > 5_000);
        assert!(pct("object type") > 99.99, "the object type is the definition's: {}%", pct("object type"));
        assert!(pct("engine flags") > 99.0, "engine flags {}%", pct("engine flags"));
        assert!(pct("symmetry bits") > 95.0 && pct("team") > 90.0 && pct("shape") > 95.0, "sym {} team {} shape {}", pct("symmetry bits"), pct("team"), pct("shape"));
    }

    /// File > New variant / a map opened with no variant: the canvas is the base map as the
    /// scenario made it. Whatever the template's author deleted or moved is undone, none of its
    /// Forge objects come along, every managed slot is on screen, and the saved file holds exactly
    /// what is on screen (the map's objects alive and unplaced, nothing deleted).
    #[test]
    fn a_blank_canvas_shows_what_it_saves() {
        let mut maps = 0usize;
        let n = each_map(|stem, c, items, vs| {
            let (path, tpl) = vs.iter().filter(|(_, v)| v.num_scenario_objects > 0).min_by_key(|(p, _)| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX)).unwrap();
            let canvas = canvas_of(tpl);
            assert!(canvas.objects.iter().all(|o| !o.placed && o.flags & SLOT_DELETED == 0 && o.flags & SLOT_EDITED == 0 && o.slot < tpl.num_scenario_objects as usize), "{stem}");
            let set = variant_to_editor(c, items, &canvas);
            let layout = scenario_layout(c);
            let managed = canvas.objects.iter().filter(|o| layout.get(o.slot).map_or(false, |s| s.managed)).count();
            assert_eq!(set.objects.len(), managed, "{stem}: every managed slot of the canvas is an editor object");
            assert!(set.objects.iter().all(|o| is_scenario_datum(o.datum)));
            // saving it as it stands: only the map's own objects, all alive, none placed
            let metas = live_metas(&set);
            let (v2, list, report) = build_h3_save_list(tpl, items, &set.src, &set.objects, &[], &set.kept, &set.scn, &metas, true);
            assert_eq!((report.scenario_deleted, report.scenario_moved, report.unsaveable, report.over_slot_cap), (0, 0, 0, 0), "{stem}");
            assert_eq!(list, canvas.objects, "{stem}: the saved records are the canvas's");
            let data = crate::mvar::read_expanded(path).unwrap();
            let bytes = super::super::mvar::rebuild_h3_file(&data, &v2, &list, None).unwrap();
            let (body, _, _) = super::super::mvar::blf_h3_mvar_chunk(&bytes).unwrap();
            let back = super::super::mvar::parse_h3_payload(body).unwrap();
            assert_eq!(back.placed().count(), 0, "{stem}");
            assert_eq!(back.objects.iter().filter(|o| o.flags & SLOT_DELETED == 0 && layout.get(o.slot).map_or(false, |s| s.managed)).count(), set.objects.len(), "{stem}: saved != shown");
            // the stored counts describe the canvas (every alive scenario object counted), not the template author's
            let alive_q: i64 = back.objects.iter().filter(|o| o.quota >= 0).count() as i64;
            let counted: i64 = back.quotas.iter().map(|q| q.placed_on_map as i64).sum();
            assert!(counted <= alive_q, "{stem}: counts {counted} > alive {alive_q}");
            assert_eq!(back.spent_budget, 0.0, "{stem}: a canvas with no Forge objects spends nothing");
            maps += 1;
        });
        eprintln!("CANVAS: {maps} of {n} maps start as the base map, shown == saved");
        if n > 0 { assert!(maps >= 10); }
    }

    /// Every map gets a canvas that registers the base map's objects. The smallest variant of six
    /// maps is the game's empty `default_*` variant (no slots at all): picking it left the map's
    /// objects as static scenery and 640 "free" slots, which is not the variant the game builds on.
    #[test]
    fn every_map_gets_an_editable_canvas() {
        let (mut picked, mut defaults_skipped) = (0usize, 0usize);
        let n = each_map(|stem, c, _, vs| {
            let id = vs[0].1.map_id;
            let t = find_template(id, None).unwrap_or_else(|| panic!("{stem}: no template"));
            let v = parse_h3_variant(&t).unwrap();
            assert!(v.num_scenario_objects > 0 && is_editable(&v, &scenario_layout(c)), "{stem}: template {} registers {} slots", t.display(), v.num_scenario_objects);
            if vs.iter().any(|(_, w)| w.num_scenario_objects == 0) { defaults_skipped += 1; }
            picked += 1;
        });
        eprintln!("TEMPLATES: {picked} of {n} maps start from a registering variant ({defaults_skipped} of them have an empty default variant that was passed over)");
        if n > 0 { assert!(picked == n && defaults_skipped >= 4, "{picked}/{n}, defaults skipped {defaults_skipped}"); }
    }

    /// The limits as Forge enforces them, per map, through the production refusal rule, with the
    /// owner's flow simulated: a blank canvas of the map, then palette rows placed one after the
    /// other until something refuses. Prints the table the report quotes. The slot limit is exactly
    /// 640 minus the map's own objects (a deleted one frees nothing); a row's own maximum refuses
    /// with its own message, and the base map's own objects count toward it (so the 100 player
    /// spawn points `respawn_point` allows are partly used up by the map's own spawns until the
    /// player deletes them).
    #[test]
    fn placement_limits_per_map() {
        let mut rows = 0usize;
        let n = each_map(|stem, c, items, vs| {
            let layout = scenario_layout(c);
            let (_, tpl) = vs.iter().find(|(_, v)| v.num_scenario_objects > 0).unwrap();
            let nso = tpl.num_scenario_objects as usize;
            let cap = H3_SLOTS - nso;
            let canvas = canvas_of(tpl);
            let editable = is_editable(tpl, &layout);
            let set = variant_to_editor(c, items, &canvas);
            let deleted_in_tpl = tpl.objects.iter().filter(|o| o.slot < nso && o.flags & SLOT_DELETED != 0).count();
            // alive base-map objects per palette row (the quota the engine counts them under)
            let scenario_of = |it: &H3PaletteItem| -> usize {
                set.objects.iter().filter(|o| is_scenario_datum(o.datum))
                    .filter(|o| set.src.get(&o.datum).and_then(|r| canvas.quota_of(r)).map_or(false, |q| q.definition == it.definition)).count()
            };
            // 1. the slot limit, reached with a row that allows enough (or the synthetic count: the engine's check is the count)
            let big = items.iter().max_by_key(|it| if it.max_allowed <= 0 { i32::MAX } else { it.max_allowed });
            let mut slot_msg = String::new();
            if let Some(it) = big {
                assert!(placement_refusal(it, nso, cap - 1, 0, 0.0, 0.0).is_none(), "{stem}: one slot left must place");
                slot_msg = placement_refusal(it, nso, cap, 0, 0.0, 0.0).expect("no slot left must refuse");
                assert!(slot_msg.starts_with("Object limit") && slot_msg.contains(&format!("{cap}")), "{stem}: {slot_msg}");
                if nso > 0 { assert!(slot_msg.contains(&format!("{nso} objects")), "{stem}: {slot_msg}"); }
            }
            // 2. the owner's flow: drop each row in turn, round robin, until every row refuses
            let mut placed_by_row = vec![0usize; items.len()];
            let (mut total, mut first_msg) = (0usize, String::new());
            loop {
                let mut any = false;
                for (i, it) in items.iter().enumerate() {
                    if placement_refusal(it, nso, total, scenario_of(it) + placed_by_row[i], 0.0, 0.0).is_none() { placed_by_row[i] += 1; total += 1; any = true; }
                    else if first_msg.is_empty() { first_msg = placement_refusal(it, nso, total, scenario_of(it) + placed_by_row[i], 0.0, 0.0).unwrap(); }
                }
                if !any || total > 640 { break; }
            }
            // 3. the spawn point row
            let spawn = items.iter().find(|it| it.max_allowed == 100);
            let spawn_note = spawn.map(|it| {
                let own = scenario_of(it);
                let free = 100usize.saturating_sub(own);
                if free > 0 { assert!(placement_refusal(it, nso, 0, own + free - 1, 0.0, 0.0).is_none()); }
                let m = placement_refusal(it, nso, 0, own + free, 0.0, 0.0).expect("at 100");
                assert!(m.contains(&it.name) && m.contains("at most 100"), "{m}");
                format!("{} (max 100): the map's own {own} alive spawn points leave room for {free}", it.name)
            }).unwrap_or_default();
            eprintln!("LIMITS {stem:<12} scenario {nso:>3} (editor objects {}, canvas editable {editable}, deleted by the template {deleted_in_tpl}) | Forge slots {cap:>3} | round-robin placement stops after {total:>3} | first refusal: \"{}\" | {spawn_note} | slot refusal: \"{slot_msg}\"", set.objects.len(), first_msg);
            rows += 1;
        });
        eprintln!("LIMITS: {rows} maps");
        if n > 0 { assert!(rows >= 10); }
        // budget: refused once spent + price passes the maximum
        let it = H3PaletteItem { index: 0, block: 1, entry: 0, definition: 0x0002_0000, tag: 0, mode_tag: None, name: "battle_rifle".into(), category: "Weapons", price: 20.0, max_allowed: 0, mp: None };
        assert!(placement_refusal(&it, 300, 0, 0, 480.0, 500.0).is_none());
        let m = placement_refusal(&it, 300, 0, 0, 490.0, 500.0).unwrap();
        assert!(m.starts_with("Over budget") && m.contains("battle_rifle") && m.contains("500"), "{m}");
        assert!(placement_refusal(&it, 300, 0, 0, 9999.0, 0.0).is_none(), "a maximum of 0 is no budget");
        let _ = forge_cost;
    }


}
