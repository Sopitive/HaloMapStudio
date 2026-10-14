//! Halo 3 `hlmt` model variants and their CHILD OBJECTS (a Warthog's chaingun, a Wraith's
//! turret): variant selection for a placed object, the region / permutation meshes a variant
//! shows, and the children posed at their markers. #h3-attach
//!
//! Layout (Halo3MCC hlmt.xml / mode.xml / the obje plugins, verified on riverworld's Warthog /
//! Mongoose and Sandtrap's vehicles through the tests):
//!   * obje +0x30 `Default Model Variant` sid; +0x34 `Model` (hlmt).
//!   * hlmt +0x64 Variants (0x30): name sid @0, Regions block @0x14 (0x14: region name sid @0,
//!     runtime model region index i8 @4, flags @5, parent variant i16 @6, Permutations block @8
//!     (0x18: permutation name sid @0, runtime model permutation index i8 @4)), Objects block
//!     @0x20 (0x1C: parent marker sid @0, child marker sid @4, child variant name sid @8,
//!     child object tagref @0xC).
//!   * mode +0x30 Nodes (0x60: name @0, parent i16 @4, default translation @0xC, default
//!     rotation ijkw @0x18), +0x3C Marker Groups (0x10: name sid @0, Markers block @4 (0x24:
//!     region i8 @0, permutation i8 @1, node i8 @2, translation @4, rotation ijkw @0x10)).
//!
//! Pose rule (the engine's `object_attach_to_marker`, the same composition the Reach truck-trailer
//! fix and the Halo 4 port use): child_world = parent_world x parent_marker_frame x
//! child_marker_frame^-1, each marker frame = its node chain's default pose x the marker's
//! local translation / rotation; a marker the model lacks is the identity (the child sits at
//! the other marker), and a missing parent marker leaves the child at the parent origin.
//! Nothing here is guessed beyond that rule: the region / permutation names are matched by
//! string id against the render model's own region / permutation names.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use super::cache::H3Cache;
use super::objects::{model_of, H3Model, OFF_OBJE_HLMT};
use crate::h4::cache::ByteRead;

pub const OFF_OBJE_DEFAULT_VARIANT: usize = 0x30;
pub const OFF_HLMT_VARIANTS: usize = 0x64;
pub const HLMT_VARIANT_ELEM: usize = 0x30;
pub const OFF_VARIANT_REGIONS: usize = 0x14;
pub const VARIANT_REGION_ELEM: usize = 0x14;
pub const OFF_VREGION_PERMUTATIONS: usize = 0x8;
pub const VARIANT_PERMUTATION_ELEM: usize = 0x18;
/// #damage hlmt variant permutation +0xC: the DAMAGE STATES block. Halo 3's element is 0x20,
/// not the 0xC Reach and Halo 4 use, because a Halo 3 state also carries a looping-effect
/// tagref at +8, a marker string id at +0x18 and the initial probability at +0x1C
/// (0x8 + 0x10 + 4 + 4 = 0x20, which is what `probe_variant_damage_states` confirms on the
/// shipped models). Halo 3 has NO runtime `i8[5]` state-permutation array - the engine fills
/// that at model load - so the state's permutation is resolved BY NAME against the render
/// model region's permutation names, the same way the Reach and Halo 4 readers ended up doing
/// it after finding their arrays all-zero in shipped caches.
pub const OFF_VPERM_STATES: usize = 0xC;
pub const VARIANT_STATE_ELEM: usize = 0x20;
pub const VSTATE_PERM_NAME: usize = 0x0;
pub const VSTATE_RM_PERM: usize = 0x4;
pub const VSTATE_ENUM: usize = 0x6;
pub const VSTATE_LOOPING_FX: usize = 0x8;
pub const VSTATE_MARKER: usize = 0x18;
pub const VSTATE_PROBABILITY: usize = 0x1C;
/// `model_state_enum`: the five damage states, shared with Reach / Halo 4.
pub const DAMAGE_STATES: [&str; 5] = ["default", "minor", "medium", "major", "destroyed"];

pub const OFF_VARIANT_OBJECTS: usize = 0x20;
pub const VARIANT_OBJECT_ELEM: usize = 0x1C;
pub const OFF_MODE_NODES: usize = 0x30;
pub const NODE_ELEM: usize = 0x60;
pub const OFF_MODE_MARKER_GROUPS: usize = 0x3C;
pub const MARKER_GROUP_ELEM: usize = 0x10;
pub const MARKER_ELEM: usize = 0x24;
pub const MAX_ATTACH_DEPTH: usize = 4;
/// Object classes a child may be (a null / odd ref is skipped).
pub const CHILD_CLASSES: [&[u8; 4]; 10] = [b"scen", b"bloc", b"vehi", b"weap", b"eqip", b"crea", b"mach", b"ctrl", b"bipd", b"gint"];

/// obje -> hlmt.
pub fn hlmt_of(c: &H3Cache, obje: usize) -> Option<usize> { c.tag_ref_of(c.tag_meta(obje)? + OFF_OBJE_HLMT, b"hlmt") }

/// The obje's default model variant sid (0 = none).
pub fn default_variant_sid(c: &H3Cache, obje: usize) -> u32 {
    match c.tag_meta(obje) { Some(om) => match c.data().u32_at(om + OFF_OBJE_DEFAULT_VARIANT) { 0xFFFF_FFFF => 0, v => v }, None => 0 }
}

/// Names (sids) of the hlmt's variants in block order.
pub fn hlmt_variant_sids(c: &H3Cache, hlmt: usize) -> Vec<u32> {
    let Some(hm) = c.tag_meta(hlmt) else { return Vec::new() };
    let Some((n, o)) = c.block(hm + OFF_HLMT_VARIANTS) else { return Vec::new() };
    (0..n.min(256)).map(|i| c.data().u32_at(o + i * HLMT_VARIANT_ELEM)).collect()
}

/// The hlmt variant an object spawns in when `sid` is requested (0 = its default): the named
/// variant, else the default, else variant 0 when the block is non-empty; None = no variants.
pub fn object_variant_index(c: &H3Cache, obje: usize, sid: u32) -> Option<usize> {
    let hlmt = hlmt_of(c, obje)?;
    let sids = hlmt_variant_sids(c, hlmt);
    if sids.is_empty() { return None; }
    let dsid = default_variant_sid(c, obje);
    let want = if sid == 0 { dsid } else { sid };
    if want != 0 { if let Some(i) = sids.iter().position(|&s| s == want) { return Some(i); } }
    if dsid != 0 { if let Some(i) = sids.iter().position(|&s| s == dsid) { return Some(i); } }
    Some(0)
}

/// The render-model meshes an object shows in hlmt variant `variant` (None = default): per model
/// region, the permutation the variant names for it (its first Permutations entry, matched by
/// name; permutation 0 when the variant does not list the region). `model.lod0_meshes` when the
/// object has no variants.
pub fn variant_meshes(c: &H3Cache, obje: usize, model: &H3Model, variant: Option<usize>) -> Vec<usize> {
    variant_state_meshes(c, obje, model, variant, 0)
}

/// #damage One region of an hlmt variant: the region name, the variant's base permutation name
/// ("" = take the render model's first) and, per `model_state_enum` slot, the permutation name
/// the variant shows in that damage state (None = the state is not authored for this region, so
/// it keeps the base pick; `Some("")` = authored with no permutation, i.e. hidden).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H3VariantRegion {
    pub region: String,
    pub base: String,
    pub states: [Option<String>; 5],
}

/// #damage The region table of hlmt variant `variant` (None = variant 0). Empty when the object
/// has no hlmt variants / the index is out of range - the caller then draws `lod0_meshes`.
pub fn variant_region_table(c: &H3Cache, obje: usize, variant: Option<usize>) -> Vec<H3VariantRegion> {
    let d = c.data();
    let mut out: Vec<H3VariantRegion> = Vec::new();
    let Some(hlmt) = hlmt_of(c, obje) else { return out };
    let Some(hm) = c.tag_meta(hlmt) else { return out };
    let Some((nv, vo)) = c.block(hm + OFF_HLMT_VARIANTS) else { return out };
    let vi = variant.unwrap_or(0);
    if vi >= nv { return out };
    let ve = vo + vi * HLMT_VARIANT_ELEM;
    let Some((nr, ro)) = c.block(ve + OFF_VARIANT_REGIONS) else { return out };
    for r in 0..nr.min(1024) {
        let re = ro + r * VARIANT_REGION_ELEM;
        let region = c.sid(d.u32_at(re));
        if out.iter().any(|e| e.region == region) { continue }
        let mut e = H3VariantRegion { region, ..Default::default() };
        if let Some((np, po)) = c.block(re + OFF_VREGION_PERMUTATIONS) {
            if np > 0 {
                // the variant's own pick is its FIRST permutation entry (the engine's
                // `model_get_variant_permutation` without the random-probability roll)
                e.base = c.sid(d.u32_at(po));
                // that permutation's DAMAGE STATES: each names the permutation to swap to.
                // Halo 3 has no runtime i8[5] array, so this block IS the mapping.
                if let Some((ns, so)) = c.block(po + OFF_VPERM_STATES) {
                    for k in 0..ns.min(32) {
                        let se = so + k * VARIANT_STATE_ELEM;
                        let state = d.u16_at(se + VSTATE_ENUM) as usize;
                        if state >= e.states.len() { continue }
                        e.states[state] = Some(c.sid(d.u32_at(se + VSTATE_PERM_NAME)));
                    }
                }
            }
        }
        out.push(e);
    }
    out
}

/// #damage The damage states (1 minor .. 4 destroyed) hlmt variant `variant` authors a
/// permutation for in at least one region, ascending. Empty = the object can only be shown
/// undamaged, so the editor offers no DAMAGE dropdown for it.
pub fn variant_damage_states(c: &H3Cache, obje: usize, variant: Option<usize>) -> Vec<u8> {
    let table = variant_region_table(c, obje, variant);
    (1..=4u8).filter(|&st| table.iter().any(|e| e.states[st as usize].is_some())).collect()
}

/// #damage `variant_meshes` as shown in damage `state` (0 = the regular model): per region the
/// permutation is the state's authored name when the variant has one, else the variant's base
/// pick, else the render model's first - resolved BY NAME against the render model's own
/// region / permutation names (Halo 3 ships no runtime permutation indices).
pub fn variant_state_meshes(c: &H3Cache, obje: usize, model: &H3Model, variant: Option<usize>, state: u8) -> Vec<usize> {
    let fallback = || model.lod0_meshes.clone();
    let table = variant_region_table(c, obje, variant);
    if table.is_empty() { return fallback() }
    let by_region: HashMap<&str, &H3VariantRegion> = table.iter().map(|e| (e.region.as_str(), e)).collect();
    let mut out = Vec::new();
    for region in &model.regions {
        // `None` = this region is not in the variant / the state is unauthored -> the base pick;
        // `Some("")` from a STATE entry = authored with NO permutation, i.e. the region is HIDDEN
        // in that state (riverworld's Warthog hides its `bumper` when destroyed).
        let mut want: Option<&str> = None;
        if let Some(e) = by_region.get(region.name.as_str()) {
            match e.states.get(state as usize) {
                Some(Some(name)) if state > 0 => {
                    if name.is_empty() { continue }
                    want = Some(name.as_str());
                }
                _ => if !e.base.is_empty() { want = Some(e.base.as_str()) },
            }
        }
        let perm = match want {
            Some(name) => region.permutations.iter().find(|p| p.name == name).or_else(|| region.permutations.first()),
            None => region.permutations.first(),
        };
        let Some(p) = perm else { continue };
        for mi in p.mesh_index.max(0)..p.mesh_index.max(0) + p.mesh_count.max(0) {
            let mi = mi as usize;
            if mi < model.meshes.len() && !out.contains(&mi) { out.push(mi); }
        }
    }
    if out.is_empty() { fallback() } else { out }
}

/// One child object of a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H3Attachment {
    pub parent_marker: u32,
    pub child_marker: u32,
    pub child_variant_sid: u32,
    pub child_obje: usize,
}

/// The child objects of hlmt variant `variant` of `obje` (empty without variants / children).
pub fn variant_attachments(c: &H3Cache, obje: usize, variant: Option<usize>) -> Vec<H3Attachment> {
    let d = c.data();
    let mut out = Vec::new();
    let Some(hlmt) = hlmt_of(c, obje) else { return out };
    let Some(hm) = c.tag_meta(hlmt) else { return out };
    let Some((nv, vo)) = c.block(hm + OFF_HLMT_VARIANTS) else { return out };
    let vi = variant.unwrap_or(0);
    if vi >= nv { return out; }
    let Some((no, oo)) = c.block(vo + vi * HLMT_VARIANT_ELEM + OFF_VARIANT_OBJECTS) else { return out };
    for k in 0..no.min(64) {
        let e = oo + k * VARIANT_OBJECT_ELEM;
        let Some((class, child)) = c.tag_ref(e + 0xC) else { continue };
        if !CHILD_CLASSES.iter().any(|k| **k == class) { continue; }
        let sid = |o: usize| match d.u32_at(o) { 0xFFFF_FFFF => 0, v => v };
        out.push(H3Attachment { parent_marker: sid(e), child_marker: sid(e + 4), child_variant_sid: sid(e + 8), child_obje: child });
    }
    out
}

/// MODEL-space frame of marker group `sid` of render model `mode` (node chain default pose x
/// marker local). None when the model has no group of that name.
pub fn marker_frame(c: &H3Cache, mode: usize, sid: u32) -> Option<Mat4> {
    if sid == 0 || sid == 0xFFFF_FFFF { return None; }
    let d = c.data();
    let mm = c.tag_meta(mode)?;
    let (ng, og) = c.block(mm + OFF_MODE_MARKER_GROUPS)?;
    let gi = (0..ng.min(4096)).find(|&i| d.u32_at(og + i * MARKER_GROUP_ELEM) == sid)?;
    let (nm, om) = c.block(og + gi * MARKER_GROUP_ELEM + 4)?;
    if nm == 0 { return None; }
    let f = |o: usize| d.f32_at(o);
    let mpos = Vec3::new(f(om + 4), f(om + 8), f(om + 12));
    let mrot = Quat::from_xyzw(f(om + 0x10), f(om + 0x14), f(om + 0x18), f(om + 0x1C));
    let mrot = if mrot.length_squared() > 1e-8 { mrot.normalize() } else { Quat::IDENTITY };
    let node = d.u8_at(om + 2);
    let (mut npos, mut nrot) = (Vec3::ZERO, Quat::IDENTITY);
    if node != 0xFF {
        if let Some((nn, on)) = c.block(mm + OFF_MODE_NODES) {
            let mut chain = Vec::new();
            let mut cur = node as i32;
            while cur >= 0 && (cur as usize) < nn && chain.len() < 64 {
                chain.push(cur as usize);
                let parent = d.i16_at(on + cur as usize * NODE_ELEM + 4) as i32;
                if parent == cur { break; }
                cur = parent;
            }
            for &ni in chain.iter().rev() {
                let o = on + ni * NODE_ELEM;
                let lp = Vec3::new(f(o + 0xC), f(o + 0x10), f(o + 0x14));
                let lr = Quat::from_xyzw(f(o + 0x18), f(o + 0x1C), f(o + 0x20), f(o + 0x24));
                let lr = if lr.length_squared() > 1e-8 { lr.normalize() } else { Quat::IDENTITY };
                npos += nrot * lp;
                nrot *= lr;
            }
        }
    }
    let pos = npos + nrot * mpos;
    let rot = (nrot * mrot).normalize();
    (pos.is_finite() && rot.is_finite()).then(|| Mat4::from_rotation_translation(rot, pos))
}

/// A resolved child ready to draw.
#[derive(Clone, Copy, Debug)]
pub struct H3AttachedObject {
    pub obje: usize,
    pub mode: usize,
    pub variant: Option<usize>,
    pub world: Mat4,
}

/// Every child of `obje` in variant `variant` at `world`, recursively (depth-capped), posed by
/// the module's rule.
pub fn expand_attachments(c: &H3Cache, obje: usize, variant: Option<usize>, world: Mat4) -> Vec<H3AttachedObject> {
    let mut out = Vec::new();
    expand_into(c, obje, variant, world, 0, &mut out);
    out
}

fn expand_into(c: &H3Cache, obje: usize, variant: Option<usize>, world: Mat4, depth: usize, out: &mut Vec<H3AttachedObject>) {
    if depth >= MAX_ATTACH_DEPTH { return; }
    let Some(parent_mode) = model_of(c, obje) else { return };
    for a in variant_attachments(c, obje, variant) {
        let Some(child_mode) = model_of(c, a.child_obje) else { continue };
        let pm = marker_frame(c, parent_mode, a.parent_marker).unwrap_or(Mat4::IDENTITY);
        let cm = marker_frame(c, child_mode, a.child_marker).unwrap_or(Mat4::IDENTITY);
        let cw = world * pm * cm.inverse();
        let cv = object_variant_index(c, a.child_obje, a.child_variant_sid);
        out.push(H3AttachedObject { obje: a.child_obje, mode: child_mode, variant: cv, world: cw });
        expand_into(c, a.child_obje, cv, cw, depth + 1, out);
    }
}

#[cfg(test)]
mod tests {
    use super::super::cache::{maps_dir, H3Cache};
    use super::super::objects::{load_model, model_of};
    use super::*;

    /// PROBE (print-only, `HMS_H3_PROBE_STATES=1`): dump every hlmt variant region /
    /// permutation / damage-state record of the vehicles on a map, so the 0x20 state stride and
    /// its field offsets can be read off the shipped data. #damage
    #[test]
    fn probe_variant_damage_states() {
        if std::env::var("HMS_H3_PROBE_STATES").is_err() { eprintln!("skip (set HMS_H3_PROBE_STATES=1)"); return }
        let Some(c) = maps_dir().map(|d| d.join("riverworld.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let d = c.data();
        for t in c.find_tags(b"vehi").into_iter().chain(c.find_tags(b"scen")).take(10) {
            let Some(hlmt) = hlmt_of(&c, t) else { continue };
            let Some(hm) = c.tag_meta(hlmt) else { continue };
            let Some((nv, vo)) = c.block(hm + OFF_HLMT_VARIANTS) else { continue };
            eprintln!("{} -> hlmt {} variants {nv}", c.tag_name(t).rsplit('\\').next().unwrap_or(""), c.tag_name(hlmt).rsplit('\\').next().unwrap_or(""));
            for v in 0..nv.min(4) {
                let ve = vo + v * HLMT_VARIANT_ELEM;
                eprintln!("  variant {v} '{}'", c.sid(d.u32_at(ve)));
                let Some((nr, ro)) = c.block(ve + OFF_VARIANT_REGIONS) else { continue };
                for r in 0..nr.min(16) {
                    let re = ro + r * VARIANT_REGION_ELEM;
                    eprintln!("    region '{}' rm_idx {} flags {:#x} parent {}", c.sid(d.u32_at(re)), d.u8_at(re + 4) as i8, d.u8_at(re + 5), d.i16_at(re + 6));
                    let Some((np, po)) = c.block(re + OFF_VREGION_PERMUTATIONS) else { continue };
                    for pi in 0..np.min(8) {
                        let pe = po + pi * VARIANT_PERMUTATION_ELEM;
                        let st = c.block(pe + OFF_VPERM_STATES);
                        eprintln!("      perm '{}' rm_idx {} raw {:02x?} states {:?}", c.sid(d.u32_at(pe)), d.u8_at(pe + 4) as i8,
                            &d[pe..pe + VARIANT_PERMUTATION_ELEM], st.map(|(n, _)| n));
                        if let Some((ns, so)) = st {
                            for k in 0..ns.min(8) {
                                let se = so + k * VARIANT_STATE_ELEM;
                                eprintln!("        state[{k}] name '{}' rm {} enum {} fx {:?} marker '{}' prob {:.3} raw {:02x?}",
                                    c.sid(d.u32_at(se + VSTATE_PERM_NAME)), d.u8_at(se + VSTATE_RM_PERM) as i8, d.u16_at(se + VSTATE_ENUM),
                                    c.tag_ref(se + VSTATE_LOOPING_FX).map(|(cl, i)| (String::from_utf8_lossy(&cl).to_string(), i)),
                                    c.sid(d.u32_at(se + VSTATE_MARKER)), d.f32_at(se + VSTATE_PROBABILITY),
                                    &d[se..se + VARIANT_STATE_ELEM]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// #damage riverworld's Warthog: the concrete damage-state picture read off the shipped
    /// `hlmt`. Pins the Halo 3 layout (variant permutation 0x18 with its STATES block at +0xC,
    /// element 0x20 - 0x14 bigger than Reach's / Halo 4's because of the looping-effect tagref
    /// at +8, the marker string id at +0x18 and the probability at +0x1C) and the by-NAME
    /// state -> permutation resolve Halo 3 needs (it ships no runtime `i8[5]` array).
    #[test]
    fn riverworld_warthog_damage_states() {
        let Some(c) = maps_dir().map(|d| d.join("riverworld.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let hog = c.find_tags(b"vehi").into_iter().find(|&t| c.tag_name(t).ends_with("\\warthog\\warthog")).expect("warthog vehi");
        let v = object_variant_index(&c, hog, 0);
        // all four damage states are authored
        assert_eq!(variant_damage_states(&c, hog, v), vec![1, 2, 3, 4]);

        let table = variant_region_table(&c, hog, v);
        let reg = |n: &str| table.iter().find(|e| e.region == n).unwrap_or_else(|| panic!("no region '{n}' in {:?}", table.iter().map(|e| &e.region).collect::<Vec<_>>()));
        // `hull`: base permutation, then minor->base, medium->medium, major->major, destroyed->destroyed
        let hull = reg("hull");
        assert_eq!(hull.base, "base");
        assert_eq!(hull.states[1].as_deref(), Some("base"));
        assert_eq!(hull.states[2].as_deref(), Some("medium"));
        assert_eq!(hull.states[3].as_deref(), Some("major"));
        assert_eq!(hull.states[4].as_deref(), Some("destroyed"));
        // `f_hull` steps through all four
        let f = reg("f_hull");
        assert_eq!([f.states[1].as_deref(), f.states[2].as_deref(), f.states[3].as_deref(), f.states[4].as_deref()],
            [Some("minor"), Some("medium"), Some("major"), Some("destroyed")]);
        // `bumper` authors only medium, and is HIDDEN when destroyed (an empty permutation name)
        let b = reg("bumper");
        assert_eq!(b.states[1], None);
        assert_eq!(b.states[2].as_deref(), Some("medium"));
        assert_eq!(b.states[3], None);
        assert_eq!(b.states[4].as_deref(), Some(""));

        // ... and the meshes actually differ, with the bumper's gone when destroyed
        let mode = model_of(&c, hog).expect("warthog mode");
        let model = load_model(&c, mode).expect("load warthog model");
        let def = variant_state_meshes(&c, hog, &model, v, 0);
        assert_eq!(def, variant_meshes(&c, hog, &model, v), "state 0 is the default picture");
        assert!(!def.is_empty());
        let bumper = model.regions.iter().find(|r| r.name == "bumper").expect("bumper region");
        let bumper_meshes: Vec<usize> = bumper.permutations.iter()
            .flat_map(|p| (p.mesh_index.max(0)..p.mesh_index.max(0) + p.mesh_count.max(0)).map(|m| m as usize)).collect();
        for st in 1..=4u8 {
            let m = variant_state_meshes(&c, hog, &model, v, st);
            assert!(!m.is_empty(), "state {st} draws nothing");
            assert_ne!(m, def, "state {st} is identical to the default");
            if st == 4 { assert!(!m.iter().any(|i| bumper_meshes.contains(i)), "the bumper must be hidden when destroyed"); }
        }
        // a region the variant leaves alone keeps its base mesh in every state
        let chassis: Vec<usize> = model.regions.iter().filter(|r| !table.iter().any(|e| e.region == r.name))
            .filter_map(|r| r.permutations.first()).map(|p| p.mesh_index.max(0) as usize).collect();
        for st in 0..=4u8 {
            let m = variant_state_meshes(&c, hog, &model, v, st);
            for mi in &chassis { assert!(m.contains(mi) || *mi >= model.meshes.len(), "state {st} dropped untouched mesh {mi}"); }
        }
    }

    /// riverworld (Valhalla): the Warthog resolves a chaingun turret child seated at a marker
    /// above the vehicle's rear (the `hlmt` objects block -> marker frames), the Mongoose has no
    /// children, and every child pose is finite and within a few units of its parent.
    #[test]
    fn riverworld_warthog_turret_and_mongoose() {
        let Some(c) = maps_dir().map(|d| d.join("riverworld.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let vehi = c.find_tags(b"vehi");
        let hog = vehi.iter().copied().find(|&t| c.tag_name(t).ends_with("\\warthog\\warthog")).expect("warthog vehi");
        let goose = vehi.iter().copied().find(|&t| c.tag_name(t).ends_with("\\mongoose\\mongoose")).expect("mongoose vehi");
        let hv = object_variant_index(&c, hog, 0);
        let atts = variant_attachments(&c, hog, hv);
        eprintln!("warthog variant {hv:?} ({:?}) attachments {:?}", hlmt_variant_sids(&c, hlmt_of(&c, hog).unwrap()).iter().map(|s| c.sid(*s)).collect::<Vec<_>>(), atts.iter().map(|a| (c.sid(a.parent_marker), c.sid(a.child_marker), c.tag_name(a.child_obje).rsplit('\\').next().unwrap_or("").to_string())).collect::<Vec<_>>());
        let world = Mat4::from_translation(Vec3::new(10.0, 20.0, 5.0));
        let ex = expand_attachments(&c, hog, hv, world);
        assert!(!ex.is_empty(), "the warthog has a child (turret)");
        for a in &ex {
            let p = a.world.w_axis.truncate();
            assert!(p.is_finite());
            let off = p - Vec3::new(10.0, 20.0, 5.0);
            eprintln!("  child {} at offset {:?}", c.tag_name(a.obje).rsplit('\\').next().unwrap_or(""), off.to_array());
            assert!(off.length() < 6.0, "child {} offset {off:?}", c.tag_name(a.obje));
        }
        // the chaingun sits behind and above the origin
        let turret = ex.iter().find(|a| c.tag_name(a.obje).contains("turret") || c.tag_name(a.obje).contains("chaingun") || c.tag_name(a.obje).contains("gun")).expect("a turret child");
        let off = turret.world.w_axis.truncate() - Vec3::new(10.0, 20.0, 5.0);
        assert!(off.z > 0.2 && off.z < 2.5, "turret height {}", off.z);
        assert!(expand_attachments(&c, goose, object_variant_index(&c, goose, 0), world).is_empty(), "the mongoose has no children");
        // variant meshes of the warthog select one permutation per region
        let model = super::super::objects::load_model(&c, model_of(&c, hog).unwrap()).unwrap();
        let meshes = variant_meshes(&c, hog, &model, hv);
        assert!(!meshes.is_empty());
        eprintln!("warthog regions {} -> {} meshes (lod0 {})", model.regions.len(), meshes.len(), model.lod0_meshes.len());
    }
}
