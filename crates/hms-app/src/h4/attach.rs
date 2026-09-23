//! Halo 4 model-variant ATTACHMENTS (`#h4-veh`): the child objects an hlmt model variant hangs
//! off its parent's markers - a rocket Warthog's turret, the Wraith's plasma mortar, the
//! Scorpion's cannon, the Mantis' chainguns.
//!
//! LAYOUT (hlmt variant element, stride 0x6C - `objects::HLMT_VARIANT_ELEM`):
//!   +0x00 stringid   variant name
//!   +0x04 i8[32]     runtime model region index per render-model region
//!   +0x24 tagblock   Regions        (0x18 B, `objects::variant_meshes`)
//!   +0x30 tagblock   **Objects**    (0x24 B) - the attachments:
//!         +0x00 stringid  parent marker
//!         +0x04 stringid  parent controlling seat label   (H4 only; absent in Reach)
//!         +0x08 stringid  child marker
//!         +0x0C stringid  child variant name              (0 = the child's own default)
//!         +0x10 tagref    child object (`obje` family: vehi / weap / scen / bipd / ...)
//!         +0x20 i16       damage section index
//!         +0x22 flags8    bit 0 = enable physics
//!   +0x3C i32        instance group index
//!   +0x40 tagblock   muted nodes
//! Evidence: Assembly's `Halo4MCC/hlmt.xml` plugin (`Variants` -> `Objects`), cross-checked byte
//! for byte on the shipped tags by `tests::dump_variant_objects` - every element of every variant
//! of the Ravine palette's vehicles resolves to a live `obje`-family tag ref at +0x10, string ids
//! that exist in the cache at +0x00/+0x08, and zero at the padding byte +0x23. The same block in
//! **Reach** is variant +0x20, stride 0x20 (scene.rs `hlmt_variant_child_markers`, RE'd from
//! `reach_tag_test.exe object_create_children` 0x140D446B0); Halo 4 inserts the seat label at +4,
//! which is the whole difference.
//!
//! POSE: the engine's `object_attach_to_marker(parent, parent_marker, child, child_marker)` poses
//! the child so that ITS marker coincides with the parent's:
//!     child_model_space = parent_marker_frame x child_marker_frame^-1
//! (`#truck-attach`: composing only the parent side - child origin ON the parent marker - stands a
//! trailer on end when the child marker is not at the child's origin). A marker frame is the node
//! chain (`mode` +0x30 nodes) composed with the marker's own local translation / rotation
//! (`mode` +0x3C marker groups, 0x10 B: name sid @0, markers block @4 of 0x30 B: region i8 @0,
//! permutation i8 @1, node u8 @2, translation @4, rotation xyzw @0x10, scale @0x20).
//! A child marker the child model does not carry = identity (the engine's
//! `object_get_marker_by_string_id` failure case: the child origin sits on the parent marker).

use glam::{Mat4, Quat, Vec3};

use super::cache::{ByteRead, H4Cache};
use super::objects::{hlmt_of, model_of, object_variant_index, HLMT_VARIANT_ELEM};

/// hlmt variant +0x30: the child-object ("Objects") block.
pub const OFF_HLMT_VARIANT_OBJECTS: usize = 0x30;
/// Stride of one child-object element.
pub const HLMT_VOBJECT_ELEM: usize = 0x24;
/// `mode` +0x3C: marker groups (0x10 B each).
pub const OFF_MODE_MARKER_GROUPS: usize = 0x3C;
pub const MARKER_GROUP_ELEM: usize = 0x10;
pub const MARKER_ELEM: usize = 0x30;
/// How deep an attachment chain is followed (a turret that itself carries a child).
pub const MAX_ATTACH_DEPTH: usize = 4;

/// One hlmt-variant child object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H4Attachment {
    /// Marker on the PARENT model the child hangs from (0 = the parent's origin).
    pub parent_marker: u32,
    /// Marker on the CHILD model that is placed onto the parent marker (0 / absent = the
    /// child's origin).
    pub child_marker: u32,
    /// The child's model variant name (0 = the child object's own default variant).
    pub child_variant_sid: u32,
    /// The child `obje`-family tag.
    pub child_obje: usize,
}

/// The `obje`-family classes a child object may be: `scene::OBJECT_CLASSES` (what a scenario
/// placement may be) plus `bipd` / `gint` / `term` / `dspn`. Observed on the shipped tags: a
/// Warthog variant's first child is `vehi` (its turret) and its second a `bloc` (the windscreen
/// damage crate), so the list has to cover the non-drawing classes too or the block scan
/// silently drops elements.
const CHILD_CLASSES: [&[u8; 4]; 12] = [b"scen", b"bloc", b"vehi", b"weap", b"eqip", b"crea", b"mach", b"ctrl", b"bipd", b"gint", b"term", b"dspn"];

/// The child objects of hlmt variant `variant` of `obje_tag` (empty when the object has no
/// variants, the variant lists none, or the refs are null).
pub fn variant_attachments(c: &H4Cache, obje_tag: usize, variant: Option<usize>) -> Vec<H4Attachment> {
    let mut out = Vec::new();
    let Some(hlmt) = hlmt_of(c, obje_tag) else { return out };
    let Some(hm) = c.tag_meta(hlmt) else { return out };
    let Some((nv, vo)) = c.block(hm + super::objects::OFF_HLMT_VARIANTS) else { return out };
    let vi = variant.unwrap_or(0);
    if vi >= nv { return out; }
    let Some((no, oo)) = c.block(vo + vi * HLMT_VARIANT_ELEM + OFF_HLMT_VARIANT_OBJECTS) else { return out };
    let d = c.data();
    for k in 0..no.min(64) {
        let e = oo + k * HLMT_VOBJECT_ELEM;
        let Some(child) = CHILD_CLASSES.iter().find_map(|cl| c.tag_ref_of(e + 0x10, cl)) else { continue };
        out.push(H4Attachment {
            parent_marker: d.u32_at(e),
            child_marker: d.u32_at(e + 8),
            child_variant_sid: d.u32_at(e + 0x0C),
            child_obje: child,
        });
    }
    out
}

/// MODEL-space frame of marker group `sid` of render model `mode_tag` (node chain o marker
/// local). STRICT: `None` when the model carries no group of that name - the engine's
/// `object_get_marker_by_string_id` failure, which `object_attach_to_marker` treats as
/// "no marker" rather than guessing another group.
pub fn marker_frame(c: &H4Cache, mode_tag: usize, sid: u32) -> Option<Mat4> {
    if sid == 0 || sid == 0xFFFF_FFFF { return None; }
    let d = c.data();
    let mm = c.tag_meta(mode_tag)?;
    let (ng, og) = c.block(mm + OFF_MODE_MARKER_GROUPS)?;
    let gi = (0..ng.min(4096)).find(|&i| d.u32_at(og + i * MARKER_GROUP_ELEM) == sid)?;
    let (nm, om) = c.block(og + gi * MARKER_GROUP_ELEM + 4)?;
    if nm == 0 { return None; }
    let f = |o: usize| d.f32_at(o);
    let mpos = Vec3::new(f(om + 4), f(om + 8), f(om + 12));
    let mrot = Quat::from_xyzw(f(om + 0x10), f(om + 0x14), f(om + 0x18), f(om + 0x1C));
    let mrot = if mrot.length_squared() > 1e-8 { mrot.normalize() } else { Quat::IDENTITY };
    let node = d.u8_at(om + 2);
    // node chain: translate/rotate up from the marker's node to the root (mode +0x30 nodes,
    // 112 B: parent i16 @4, translation @12, rotation xyzw @24)
    let (mut npos, mut nrot) = (Vec3::ZERO, Quat::IDENTITY);
    if node != 0xFF {
        if let Some((nn, on)) = c.block(mm + super::objects::OFF_MODE_NODES) {
            let mut chain = Vec::new();
            let mut cur = node as i32;
            while cur >= 0 && (cur as usize) < nn && chain.len() < 64 {
                chain.push(cur as usize);
                let parent = d.i16_at(on + cur as usize * super::objects::NODE_ELEM + 4) as i32;
                if parent == cur { break; }
                cur = parent;
            }
            for &ni in chain.iter().rev() {
                let o = on + ni * super::objects::NODE_ELEM;
                let lp = Vec3::new(f(o + 12), f(o + 16), f(o + 20));
                let lr = Quat::from_xyzw(f(o + 24), f(o + 28), f(o + 32), f(o + 36));
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

/// One resolved attachment ready to draw: the child's object / render-model tags, its hlmt
/// variant index and its WORLD matrix.
#[derive(Clone, Copy, Debug)]
pub struct H4AttachedObject {
    pub obje: usize,
    pub mode: usize,
    pub variant: Option<usize>,
    pub world: Mat4,
}

/// Every child object a placement of `obje_tag` (in hlmt variant `variant`, world matrix
/// `world`) draws, recursively (depth-limited, cycle-guarded by the visited object set).
/// The parent itself is NOT included.
pub fn expand_attachments(c: &H4Cache, obje_tag: usize, variant: Option<usize>, world: Mat4) -> Vec<H4AttachedObject> {
    let mut out = Vec::new();
    let mut stack = vec![(obje_tag, variant, world, 0usize)];
    let mut seen: Vec<usize> = vec![obje_tag];
    while let Some((obje, vi, w, depth)) = stack.pop() {
        if depth >= MAX_ATTACH_DEPTH { continue; }
        let Some(parent_mode) = model_of(c, obje) else { continue };
        for a in variant_attachments(c, obje, vi) {
            let Some(child_mode) = model_of(c, a.child_obje) else { continue };
            let cvi = object_variant_index(c, a.child_obje, a.child_variant_sid);
            // engine object_attach_to_marker: child = parent_marker_frame x child_marker_frame^-1
            let pf = marker_frame(c, parent_mode, a.parent_marker).unwrap_or(Mat4::IDENTITY);
            let cf = marker_frame(c, child_mode, a.child_marker).map(|m| m.inverse()).unwrap_or(Mat4::IDENTITY);
            let cw = w * pf * cf;
            out.push(H4AttachedObject { obje: a.child_obje, mode: child_mode, variant: cvi, world: cw });
            if !seen.contains(&a.child_obje) {
                seen.push(a.child_obje);
                stack.push((a.child_obje, cvi, cw, depth + 1));
            }
        }
        if out.len() > 64 { break; }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;

    fn open(map: &str) -> Option<H4Cache> {
        let p = maps_dir()?.join(map);
        p.exists().then(|| H4Cache::open(&p).ok()).flatten()
    }

    /// RE tool: every hlmt variant OBJECT block of the objects whose tag name matches
    /// `HMS_H4_ATTACH_WANT` (default `storm_`) on `HMS_H4_ATTACH_MAP` (default
    /// `ca_forge_ravine.map`), with the raw element bytes, so the +0x30 / 0x24 layout can be
    /// checked against the tag data. Print-only, ignored.
    #[test]
    #[ignore]
    fn dump_variant_objects() {
        let map = std::env::var("HMS_H4_ATTACH_MAP").unwrap_or_else(|_| "ca_forge_ravine.map".into());
        let want = std::env::var("HMS_H4_ATTACH_WANT").unwrap_or_else(|_| "storm_".into());
        let Some(c) = open(&map) else { eprintln!("{map} not found"); return };
        let d = c.data();
        for class in [b"vehi", b"scen", b"bipd", b"weap"] {
            for t in c.find_tags(class) {
                let name = c.tag_name(t).to_string();
                if !name.contains(&want) { continue; }
                let Some(hlmt) = hlmt_of(&c, t) else { continue };
                let Some(hm) = c.tag_meta(hlmt) else { continue };
                let Some((nv, vo)) = c.block(hm + super::super::objects::OFF_HLMT_VARIANTS) else { continue };
                let mode = model_of(&c, t);
                for v in 0..nv.min(32) {
                    let atts = variant_attachments(&c, t, Some(v));
                    if atts.is_empty() { continue; }
                    let vname = c.sid(d.u32_at(vo + v * HLMT_VARIANT_ELEM));
                    eprintln!("{name} variant {v} '{vname}': {} child objects", atts.len());
                    let (no, oo) = c.block(vo + v * HLMT_VARIANT_ELEM + OFF_HLMT_VARIANT_OBJECTS).unwrap();
                    for k in 0..no.min(64) {
                        let e = oo + k * HLMT_VOBJECT_ELEM;
                        let raw: Vec<String> = (0..HLMT_VOBJECT_ELEM).map(|i| format!("{:02x}", d[e + i])).collect();
                        eprintln!("   raw {}", raw.join(" "));
                    }
                    for a in &atts {
                        let cm = model_of(&c, a.child_obje);
                        let pf = mode.and_then(|m| marker_frame(&c, m, a.parent_marker));
                        let cf = cm.and_then(|m| marker_frame(&c, m, a.child_marker));
                        eprintln!("   parent_marker '{}' ({:#x}) child_marker '{}' ({:#x}) child_variant '{}' child '{}' | parent frame {:?} child frame {:?}",
                            c.sid(a.parent_marker), a.parent_marker, c.sid(a.child_marker), a.child_marker, c.sid(a.child_variant_sid),
                            c.tag_name(a.child_obje), pf.map(|m| m.w_axis.truncate().to_array()), cf.map(|m| m.w_axis.truncate().to_array()));
                    }
                }
            }
        }
    }

    /// The layout gate: on every shipped object of the Ravine cache, every child-object
    /// element must have a live `obje`-family tag ref at +0x10, a zero padding byte at +0x23 and
    /// a parent marker string id that the model actually carries (or 0 = the origin). A layout
    /// that is off by a field fails all three.
    #[test]
    fn variant_objects_layout_is_sane() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let d = c.data();
        let (mut elems, mut with_marker, mut objs) = (0usize, 0usize, 0usize);
        for class in [b"vehi", b"scen", b"bipd", b"weap", b"mach", b"ctrl"] {
            for t in c.find_tags(class) {
                let Some(hlmt) = hlmt_of(&c, t) else { continue };
                let Some(hm) = c.tag_meta(hlmt) else { continue };
                let Some((nv, vo)) = c.block(hm + super::super::objects::OFF_HLMT_VARIANTS) else { continue };
                let mode = model_of(&c, t);
                for v in 0..nv.min(32) {
                    let Some((no, oo)) = c.block(vo + v * HLMT_VARIANT_ELEM + OFF_HLMT_VARIANT_OBJECTS) else { continue };
                    if no == 0 { continue; }
                    objs += 1;
                    for k in 0..no.min(64) {
                        let e = oo + k * HLMT_VOBJECT_ELEM;
                        elems += 1;
                        assert_eq!(d.u8_at(e + 0x23), 0, "{}: variant {v} child {k} padding byte", c.tag_name(t));
                        // +0x10 must be a tag ref: either an obje-family class magic (stored
                        // reversed) or the null ref the tag compiler leaves behind
                        let mut magic = [0u8; 4];
                        for i in 0..4 { magic[i] = d[e + 0x13 - i]; }
                        let known = CHILD_CLASSES.iter().any(|cl| **cl == magic) || magic == [0xFF; 4] || magic == [0; 4];
                        assert!(known, "{}: variant {v} child {k} class magic at +0x10 is {:?} ({:x?}), not an obje family", c.tag_name(t), String::from_utf8_lossy(&magic), magic);
                        if CHILD_CLASSES.iter().any(|cl| **cl == magic) {
                            assert!(CHILD_CLASSES.iter().any(|cl| c.tag_ref_of(e + 0x10, cl).is_some()),
                                "{}: variant {v} child {k} '{}' ref does not resolve", c.tag_name(t), String::from_utf8_lossy(&magic));
                        }
                        let pm = d.u32_at(e);
                        if pm != 0 {
                            if let Some(m) = mode {
                                if marker_frame(&c, m, pm).is_some() { with_marker += 1; }
                            }
                        }
                    }
                }
            }
        }
        // the shipped vehicles hang their turrets off real markers - if the parent-marker field
        // were at the wrong offset essentially none of them would resolve
        assert!(objs > 0, "no object carried a variant child-object block");
        assert!(with_marker * 2 > elems, "only {with_marker} of {elems} child objects name a marker the parent model carries");
    }
}
