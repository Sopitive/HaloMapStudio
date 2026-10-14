//! Halo 3 (MCC) scenario objects + render models: `scnr` placement tables -> palette `obje`
//! -> `hlmt` -> `mode` render model -> geometry resource + meshes. #h3
//!
//! Layout (Halo3MCC plugins scnr/scen/hlmt/mode, verified on guardian / construct - every
//! placement of every table passes the header sanity test at the pinned stride, every
//! placed palette tag resolves a `mode` whose meshes decode inside the model bounds):
//!   * scnr +0x34 Skies (0x14: `scen` ref @0), +0xA8 Object Names (0x24: sid @0),
//!     placement / palette block pairs per object class - `PLACEMENT_TABLES` (palette element
//!     0x10 = one tag ref). Placement header (shared by every table): i16 palette index @0,
//!     i16 name index @2, u32 placement flags @4, f32x3 position @8, f32x3 rotation @0x14
//!     (yaw / pitch / roll radians as stored), f32 scale @0x20 (0 = 1.0).
//!   * obje (scen/bloc/vehi/weap/eqip/mach/ctrl/bipd/efsc/gint) +0x34 = `hlmt` ref;
//!     hlmt +0x00 = `mode` ref.
//!   * `mode`: +0x0C regions (0x10: sid @0, permutations block @4; permutation 0x18: sid @0,
//!     i16 mesh index @4, i16 mesh count @6), +0x48 materials (0x24: render-method ref @0 -
//!     rmsh/rmsk/... DIRECT, no `mat ` indirection), +0x68 meshes (0x4C - the SAME element as
//!     the Lbsp: parts @0 (0x10), vb[8] @0x18, ib @0x28, vertex type @0x2E, index type @0x30),
//!     +0x74 compression info (0x2C, the Lbsp bounds element; usually ONE model-wide entry),
//!     +0xE0 `[zone] asset datum` = the render-geometry resource (owner `mode`).
//!   * Model vertices use the BSP's compressed rigid layout (stride 36 / 44 +skin: pos int16x3
//!     @0 + uv int16x2 @8 dequantised to the bounds, normal snorm16x3 @0x14) - decoded by
//!     geometry.rs `decode_mesh_generic`, which is why the model path has no decoder of its own.
//! Not confirmed against the game: the euler -> basis convention (the Reach ScenarioObjectWalker
//! port shared with Halo 4, same engine family).

use anyhow::{anyhow, bail, Result};
use glam::{Mat4, Vec3, Vec4};

use super::cache::H3Cache;
use super::geometry::{decode_mesh_generic, parse_geometry_resource, Bounds, DecodedMesh, GeometryResource, MeshDef, Part, BOUNDS_ELEM, MESH_ELEM, PART_ELEM};
use crate::h4::cache::ByteRead;

pub const OFF_SCNR_SKIES: usize = 0x34;
pub const SCNR_SKY_ELEM: usize = 0x14;
pub const OFF_SCNR_OBJECT_NAMES: usize = 0xA8;
pub const OBJECT_NAME_ELEM: usize = 0x24;
pub const PALETTE_ELEM: usize = 0x10;
pub const OFF_OBJE_HLMT: usize = 0x34;
pub const OFF_HLMT_MODE: usize = 0x00;
pub const OFF_MODE_REGIONS: usize = 0x0C;
pub const OFF_MODE_MATERIALS: usize = 0x48;
pub const OFF_MODE_MESHES: usize = 0x68;
pub const OFF_MODE_BOUNDS: usize = 0x74;
pub const OFF_MODE_GEOMETRY_RID: usize = 0xE0;
pub const REGION_ELEM: usize = 0x10;
pub const PERMUTATION_ELEM: usize = 0x18;
pub const MODE_MATERIAL_ELEM: usize = 0x24;

/// One (placements, palette) block pair in the scnr main struct; `stride` = placement element size.
#[derive(Clone, Copy, Debug)]
pub struct PlacementTable {
    pub class: [u8; 4],
    pub placements: usize,
    pub palette: usize,
    pub stride: usize,
}

/// The renderable object classes (sound scenery / terminals / light volumes carry no model).
pub const PLACEMENT_TABLES: [PlacementTable; 10] = [
    PlacementTable { class: *b"scen", placements: 0xB4, palette: 0xC0, stride: 0xB4 },
    PlacementTable { class: *b"bipd", placements: 0xCC, palette: 0xD8, stride: 0x74 },
    PlacementTable { class: *b"vehi", placements: 0xE4, palette: 0xF0, stride: 0xA8 },
    PlacementTable { class: *b"eqip", placements: 0xFC, palette: 0x108, stride: 0x8C },
    PlacementTable { class: *b"weap", placements: 0x114, palette: 0x120, stride: 0xA8 },
    PlacementTable { class: *b"mach", placements: 0x138, palette: 0x144, stride: 0x88 },
    PlacementTable { class: *b"ctrl", placements: 0x168, palette: 0x174, stride: 0x78 },
    PlacementTable { class: *b"gint", placements: 0x198, palette: 0x1A4, stride: 0x84 },
    PlacementTable { class: *b"efsc", placements: 0x1B0, palette: 0x1BC, stride: 0x54 },
    PlacementTable { class: *b"bloc", placements: 0x5BC, palette: 0x5C8, stride: 0xB0 },
];

/// Placement flags bit 3: "never placed" (the object exists for scripts only).
pub const PLACEMENT_NEVER_PLACED: u32 = 1 << 3;

#[derive(Clone, Debug)]
pub struct H3Placement {
    /// Class of the palette tag (scen / bloc / vehi / ...).
    pub class: [u8; 4],
    pub palette_tag: usize,
    /// Object name from the scnr names block (empty when the name index is -1).
    pub name: String,
    pub pos: [f32; 3],
    /// yaw, pitch, roll radians as stored.
    pub rot: [f32; 3],
    /// 1.0 when the stored scale is <= 0 or not finite.
    pub scale: f32,
    pub flags: u32,
}

impl H3Placement {
    pub fn class_str(&self) -> String { String::from_utf8_lossy(&self.class).into_owned() }
    pub fn never_placed(&self) -> bool { self.flags & PLACEMENT_NEVER_PLACED != 0 }
}

/// Every placement of every table, in table order. A placement whose palette index is -1 / out
/// of range or whose palette entry is a null ref is skipped.
pub fn load_placements(c: &H3Cache) -> Vec<H3Placement> {
    let d = c.data();
    let mut out = Vec::new();
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let names = c.block(sm + OFF_SCNR_OBJECT_NAMES);
    for t in PLACEMENT_TABLES.iter() {
        let Some((np, po)) = c.block(sm + t.placements) else { continue };
        let Some((npal, palo)) = c.block(sm + t.palette) else { continue };
        if np > 0x40000 || npal > 4096 { continue; }
        for i in 0..np {
            let e = po + i * t.stride;
            let pal = d.i16_at(e);
            if pal < 0 || pal as usize >= npal { continue; }
            let Some((_, palette_tag)) = c.tag_ref(palo + pal as usize * PALETTE_ELEM) else { continue };
            let class = c.tag_class(palette_tag).unwrap_or(t.class);
            let ni = d.i16_at(e + 2);
            let name = match names {
                Some((nn, no)) if ni >= 0 && (ni as usize) < nn => c.sid(d.u32_at(no + ni as usize * OBJECT_NAME_ELEM)),
                _ => String::new(),
            };
            let f = |o: usize| d.f32_at(e + o);
            let mut scale = f(0x20);
            if !(scale > 0.0) || !scale.is_finite() { scale = 1.0; }
            out.push(H3Placement {
                class,
                palette_tag,
                name,
                pos: [f(8), f(12), f(16)],
                rot: [f(0x14), f(0x18), f(0x1C)],
                scale,
                flags: d.u32_at(e + 4),
            });
        }
    }
    out
}

/// World matrix of a placement: columns [fwd, up x fwd, up, pos] scaled uniformly (local +X =
/// forward, +Z = up), the euler -> basis shared with the Halo 4 / Reach object paths.
pub fn placement_matrix(p: &H3Placement) -> Mat4 {
    let (f, u) = crate::h4::objects::euler_to_basis(p.rot);
    let mut left = u.cross(f);
    if left.length_squared() < 1e-12 { left = Vec3::Y; }
    let s = p.scale;
    Mat4::from_cols(
        Vec4::new(f.x * s, f.y * s, f.z * s, 0.0),
        Vec4::new(left.x * s, left.y * s, left.z * s, 0.0),
        Vec4::new(u.x * s, u.y * s, u.z * s, 0.0),
        Vec4::new(p.pos[0], p.pos[1], p.pos[2], 1.0),
    )
}

/// obje (any object class) -> hlmt (+0x34) -> mode (+0x00).
pub fn model_of(c: &H3Cache, palette_tag: usize) -> Option<usize> {
    let om = c.tag_meta(palette_tag)?;
    let hlmt = c.tag_ref_of(om + OFF_OBJE_HLMT, b"hlmt")?;
    let hm = c.tag_meta(hlmt)?;
    c.tag_ref_of(hm + OFF_HLMT_MODE, b"mode")
}

#[derive(Clone, Debug)]
pub struct H3Permutation {
    pub name: String,
    pub mesh_index: i16,
    pub mesh_count: i16,
}

#[derive(Clone, Debug)]
pub struct H3Region {
    pub name: String,
    pub permutations: Vec<H3Permutation>,
}

pub struct H3Model {
    pub tag: usize,
    pub name: String,
    pub meshes: Vec<MeshDef>,
    /// The compression table (mesh i uses entry min(i, n-1): usually one model-wide element).
    pub bounds: Vec<Bounds>,
    pub geometry: GeometryResource,
    /// Render-method tag per material slot (None for a null ref).
    pub materials: Vec<Option<usize>>,
    pub regions: Vec<H3Region>,
    /// Mesh indices of permutation 0 of every region (the default variant), deduplicated.
    pub lod0_meshes: Vec<usize>,
}

impl H3Model {
    pub fn mesh_bounds(&self, mesh_idx: usize) -> Option<&Bounds> {
        if self.bounds.is_empty() { None } else { self.bounds.get(mesh_idx.min(self.bounds.len() - 1)) }
    }
}

fn read_bounds(d: &[u8], o: usize) -> Bounds {
    let f = |k: usize| d.f32_at(o + k);
    Bounds {
        flags: d.u16_at(o),
        pos_min: [f(4), f(0xC), f(0x14)],
        pos_max: [f(8), f(0x10), f(0x18)],
        uv_min: [f(0x1C), f(0x24)],
        uv_max: [f(0x20), f(0x28)],
    }
}

/// Parse a `mode` tag: meshes, bounds, materials, regions and its geometry resource.
pub fn load_model(c: &H3Cache, mode_tag: usize) -> Result<H3Model> {
    if c.tag_class(mode_tag).as_ref() != Some(b"mode") { bail!("tag {mode_tag} is not a mode"); }
    let d = c.data();
    let mm = c.tag_meta(mode_tag).ok_or_else(|| anyhow!("mode meta"))?;
    let name = c.tag_name(mode_tag).to_string();
    // geometry resource: the datum @0xE0 (owner `mode`), else the mode-owned geometry resource
    // of this tag (a datum this reader cannot key)
    let is_geo = |e: &&super::cache::ResourceEntry| e.is_live() && c.kind_is(e.kind, H3Cache::RES_GEOMETRY) && e.owner == *b"mode";
    let entry = c.resource_by_id(d.u32_at(mm + OFF_MODE_GEOMETRY_RID)).filter(is_geo)
        .or_else(|| c.resources.iter().find(|e| is_geo(e) && e.owner_tag == mode_tag))
        .ok_or_else(|| anyhow!("{name}: no render geometry resource"))?;
    let geometry = parse_geometry_resource(c, entry)?;
    let (nm, om) = c.block(mm + OFF_MODE_MESHES).unwrap_or((0, 0));
    let (nb, ob) = c.block(mm + OFF_MODE_BOUNDS).unwrap_or((0, 0));
    let mut bounds = Vec::with_capacity(nb);
    for i in 0..nb.min(4096) { bounds.push(read_bounds(d, ob + i * BOUNDS_ELEM)); }
    let mut meshes = Vec::with_capacity(nm);
    for i in 0..nm.min(4096) {
        let e = om + i * MESH_ELEM;
        let mut m = MeshDef::default();
        if let Some((pc, pp)) = c.block(e) {
            for k in 0..pc.min(4096) {
                let p = pp + k * PART_ELEM;
                m.parts.push(Part { material: d.i16_at(p), index_start: d.u16_at(p + 4) as u32, index_count: d.u16_at(p + 6) as u32 });
            }
        }
        for k in 0..8 { m.vb[k] = d.i16_at(e + 0x18 + k * 2); }
        m.ib = d.i16_at(e + 0x28);
        m.vertex_type = d.u8_at(e + 0x2E);
        m.index_type = d.u8_at(e + 0x30);
        if nb > 0 { m.bounds = bounds[i.min(nb - 1)]; }
        meshes.push(m);
    }
    let mut materials = Vec::new();
    if let Some((n, o)) = c.block(mm + OFF_MODE_MATERIALS) {
        for i in 0..n.min(4096) { materials.push(c.tag_ref(o + i * MODE_MATERIAL_ELEM).map(|(_, t)| t)); }
    }
    let mut regions = Vec::new();
    let mut lod0_meshes: Vec<usize> = Vec::new();
    if let Some((nr, or)) = c.block(mm + OFF_MODE_REGIONS) {
        for i in 0..nr.min(1024) {
            let e = or + i * REGION_ELEM;
            let mut r = H3Region { name: c.sid(d.u32_at(e)), permutations: Vec::new() };
            if let Some((np, op)) = c.block(e + 4) {
                for k in 0..np.min(4096) {
                    let p = op + k * PERMUTATION_ELEM;
                    r.permutations.push(H3Permutation { name: c.sid(d.u32_at(p)), mesh_index: d.i16_at(p + 4), mesh_count: d.i16_at(p + 6) });
                }
            }
            if let Some(p0) = r.permutations.first() {
                for mi in p0.mesh_index.max(0)..p0.mesh_index.max(0) + p0.mesh_count.max(0) {
                    let mi = mi as usize;
                    if mi < meshes.len() && !lod0_meshes.contains(&mi) { lod0_meshes.push(mi); }
                }
            }
            regions.push(r);
        }
    }
    Ok(H3Model { tag: mode_tag, name, meshes, bounds, geometry, materials, regions, lod0_meshes })
}

/// A model mesh's per-vertex colour stream, when it carries one: a second vertex buffer of
/// kind 14 / stride 12 = f32x3 per vertex (the classic sky domes: guardian's `skydome_dark`
/// 0.25/0.35/0.32 mean, construct's space dome near black). None when absent or all zero.
pub fn model_vertex_colors(c: &H3Cache, model: &H3Model, mesh_idx: usize) -> Option<Vec<[f32; 3]>> {
    let m = model.meshes.get(mesh_idx)?;
    for &vbi in m.vb.iter().skip(1).filter(|&&v| v >= 0) {
        let Ok((info, bytes)) = super::geometry::read_vb_raw_generic(c, &model.geometry, vbi as usize) else { continue };
        if info.kind != 14 || info.stride != 12 { continue; }
        let n = info.count as usize;
        let cols: Vec<[f32; 3]> = (0..n).map(|k| [bytes.f32_at(k * 12), bytes.f32_at(k * 12 + 4), bytes.f32_at(k * 12 + 8)]).collect();
        if cols.iter().all(|c| c.iter().all(|v| v.is_finite())) && cols.iter().any(|c| c.iter().any(|&v| v > 1e-4)) { return Some(cols); }
    }
    None
}

/// Decode one model mesh (model space, dequantised through the model's compression entry).
pub fn decode_model_mesh(c: &H3Cache, model: &H3Model, mesh_idx: usize) -> Result<Option<DecodedMesh>> {
    let m = model.meshes.get(mesh_idx).ok_or_else(|| anyhow!("mesh {mesh_idx} out of range"))?;
    decode_mesh_generic(c, &model.geometry, m, model.mesh_bounds(mesh_idx))
}

#[cfg(test)]
mod tests {
    use super::super::cache::maps_dir;
    use super::*;

    fn open(name: &str) -> Option<H3Cache> {
        let p = maps_dir()?.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H3Cache::open(&p).expect("open"))
    }

    /// guardian: the scenario places objects of several classes, every one resolves a render
    /// model, and the default permutation meshes decode inside the model's compression bounds.
    #[test]
    fn guardian_objects_decode() {
        let Some(c) = open("guardian.map") else { return };
        let placements = load_placements(&c);
        assert!(placements.len() > 20, "{} placements", placements.len());
        let mut classes: Vec<String> = placements.iter().map(|p| p.class_str()).collect();
        classes.sort();
        classes.dedup();
        assert!(classes.iter().any(|k| k == "scen"), "classes {classes:?}");
        assert!(placements.iter().all(|p| p.pos.iter().all(|v| v.is_finite()) && p.rot.iter().all(|v| v.abs() <= 2.0 * std::f32::consts::PI + 1e-3)));
        let (mut models, mut meshes, mut no_model) = (0usize, 0usize, 0usize);
        let mut seen = std::collections::HashSet::new();
        for p in &placements {
            if !seen.insert(p.palette_tag) { continue; }
            let Some(mode) = model_of(&c, p.palette_tag) else { no_model += 1; continue };
            let model = load_model(&c, mode).unwrap_or_else(|e| panic!("{}: {e}", c.tag_name(mode)));
            models += 1;
            for &mi in &model.lod0_meshes {
                let Some(dm) = decode_model_mesh(&c, &model, mi).unwrap() else { continue };
                meshes += 1;
                assert!(dm.indices.iter().all(|&x| (x as usize) < dm.verts.len()), "{} mesh {mi}: index range", model.name);
                if let Some(b) = model.mesh_bounds(mi) {
                    if (b.flags & 1) != 0 && b.pos_max != b.pos_min {
                        for v in &dm.verts { for k in 0..3 { assert!(v.pos[k] >= b.pos_min[k] - 1e-2 && v.pos[k] <= b.pos_max[k] + 1e-2, "{} mesh {mi} pos {:?} outside {:?}..{:?}", model.name, v.pos, b.pos_min, b.pos_max); } }
                    }
                }
                for v in &dm.verts { let n = Vec3::from(v.normal).length(); assert!((n - 1.0).abs() < 0.05 || n < 1e-3, "{} mesh {mi} normal {n}", model.name); }
            }
        }
        eprintln!("guardian objects: {} placements, {} distinct palette tags -> {models} models ({no_model} without), {meshes} meshes decoded; classes {classes:?}", placements.len(), seen.len());
        assert!(models >= 5 && meshes >= 5);
    }

    /// guardian's sky: scnr +0x34 names a `scen` whose model decodes to parts with materials.
    #[test]
    fn guardian_sky_model_decodes() {
        let Some(c) = open("guardian.map") else { return };
        let scnr = c.find_tags(b"scnr")[0];
        let sm = c.tag_meta(scnr).unwrap();
        let (n, so) = c.block(sm + OFF_SCNR_SKIES).expect("skies block");
        assert!(n >= 1);
        let scen = c.tag_ref_of(so, b"scen").expect("sky scen");
        let mode = model_of(&c, scen).expect("sky mode");
        let model = load_model(&c, mode).unwrap();
        assert!(!model.lod0_meshes.is_empty());
        let mut parts = 0;
        for &mi in &model.lod0_meshes {
            let md = &model.meshes[mi];
            let vbs: Vec<String> = md.vb.iter().filter(|&&v| v >= 0).filter_map(|&v| model.geometry.vbs.get(v as usize)).map(|v| format!("k{} s{} n{}", v.kind, v.stride, v.count)).collect();
            let dm = decode_model_mesh(&c, &model, mi).unwrap();
            eprintln!("sky mesh {mi}: vtype {} itype {} vbs [{}] -> {} verts / {} parts", md.vertex_type, md.index_type, vbs.join(", "), dm.as_ref().map_or(0, |d| d.verts.len()), md.parts.len());
            let Some(dm) = dm else { continue };
            for p in &dm.parts {
                let rt = usize::try_from(p.material).ok().and_then(|k| model.materials.get(k).copied().flatten());
                let (mut rmin, mut rmax) = (f32::MAX, f32::MIN);
                for &ix in &dm.indices[p.index_start as usize..(p.index_start + p.index_count) as usize] { let r = Vec3::from(dm.verts[ix as usize].pos).length(); rmin = rmin.min(r); rmax = rmax.max(r); }
                let m = rt.map(|t| super::super::materials::load_material(&c, t));
                if let Some(mm) = &m { if mm.base.is_none() { eprintln!("  floats {:?} vcolors {}", mm.floats, model_vertex_colors(&c, &model, mi).map(|v| v.len()).unwrap_or(0)); } }
                eprintln!("  part mat {} {:?} {} radius {rmin:.1}..{rmax:.1} base {:?} params {:?}", p.material, rt.and_then(|t| c.tag_class(t)).map(|k| String::from_utf8_lossy(&k).into_owned()), rt.map(|t| c.tag_name(t)).unwrap_or(""), m.as_ref().and_then(|m| m.base).map(|b| c.tag_name(b).to_string()), m.as_ref().map(|m| m.params.iter().map(|(n, b)| format!("{n}={}", c.tag_name(*b).rsplit('\\').next().unwrap_or(""))).collect::<Vec<_>>()));
                parts += 1;
            }
        }
        assert!(parts >= 1);
    }

    /// Probe (print-only): riverworld's terrain (`rmtr`) parts - per part the vertex format,
    /// UV range and texcoord transforms, to see whether the ground samples a single texel.
    #[test]
    fn riverworld_terrain_uv_ranges() {
        let Some(c) = open("riverworld.map") else { return };
        let bsp = super::super::geometry::load_bsp(&c, c.find_tags(b"sbsp")[0]).unwrap();
        let mut shown = 0;
        let mut seen = std::collections::HashSet::new();
        for &mi in bsp.clusters.iter().chain(bsp.instances.iter().filter(|i| i.mesh >= 0).map(|i| &i.mesh)) {
            if !seen.insert(mi) { continue; }
            let Ok(Some(dm)) = super::super::geometry::decode_mesh(&c, &bsp, mi as usize) else { continue };
            let md = &bsp.meshes[mi as usize];
            for p in &dm.parts {
                let Some(bm) = usize::try_from(p.material).ok().and_then(|k| bsp.materials.get(k)) else { continue };
                if &bm.class != b"rmtr" { continue; }
                let m = super::super::materials::load_material(&c, bm.render_tag.unwrap());
                let (mut mn, mut mx) = ([f32::MAX; 2], [f32::MIN; 2]);
                for &ix in &dm.indices[p.index_start as usize..(p.index_start + p.index_count) as usize] { let v = &dm.verts[ix as usize]; for k in 0..2 { mn[k] = mn[k].min(v.uv[k]); mx[k] = mx[k].max(v.uv[k]); } }
                let b = &md.bounds;
                eprintln!("mesh {mi} vtype {} part mat '{}' idx {} uv {:?}..{:?} bounds flags {:#x} uv_min {:?} uv_max {:?} pos_min {:?} | blend xf {:?} base0 xf {:?} detail0 xf {:?}",
                    md.vertex_type, m.name.rsplit('\\').next().unwrap_or(""), p.index_count, mn, mx, b.flags, b.uv_min, b.uv_max, b.pos_min, m.xform("blend_map"), m.xform("base_map_m_0"), m.xform("detail_map_m_0"));
                shown += 1;
                if shown > 24 { return; }
            }
        }
    }

    /// Probe (print-only): the bitmap element + resource / segment / page of the object textures
    /// that upload as "empty (0 B)" on riverworld (weapons, vehicles).
    #[test]
    fn riverworld_empty_object_bitmaps() {
        let Some(c) = open("riverworld.map") else { return };
        for t in c.find_tags(b"bitm") {
            let n = c.tag_name(t);
            if !(n.ends_with("battle_rifle") || n.ends_with("mongoose") || n.ends_with("bubble_shield") || n.ends_with("cliff_rock_base")) { continue; }
            let Some(info) = super::super::bitmaps::bitmap_info(&c, t) else { continue };
            let d = c.data();
            let bm = c.tag_meta(t).unwrap();
            // bitmaps block @0x?? - reuse bitmap_info's element: print the raw 0x30 bytes after the pixel fields
            let e = c.resources.iter().find(|e| e.owner_tag == t && e.owner == *b"bitm");
            let seg = e.and_then(|e| c.segments.get(e.loc.max(0) as usize));
            let pages = seg.map(|s| (s.primary_page, s.primary_off, s.secondary_page, s.secondary_off));
            let page_sizes = seg.map(|s| (c.pages.get(s.primary_page.max(0) as usize).map(|p| (p.cache_index, p.decompressed)), c.pages.get(s.secondary_page.max(0) as usize).map(|p| (p.cache_index, p.decompressed))));
            eprintln!("{}: {}x{} fmt {:#x} kind {} levels {} px_off {} px_size {} | res {:?} loc {:?} | seg (pp, poff, sp, soff) {:?} | page (cache_idx, decompressed) {:?}",
                n.rsplit('\\').next().unwrap_or(""), info.w, info.h, info.format, info.kind, info.levels, info.px_off, info.px_size,
                e.map(|e| (e.index, e.datum, e.kind)), e.map(|e| e.loc), pages, page_sizes);
            let _ = (d, bm);
        }
    }

    /// Probe (print-only): the sky meshes' second vertex stream (kind 14, stride 12) decoded
    /// as three f32 - the per-vertex sky colour candidates - min / mean / max per mesh.
    #[test]
    fn guardian_sky_vertex_streams() {
        for map in ["guardian.map", "construct.map"] {
            let Some(c) = open(map) else { continue };
            let scnr = c.find_tags(b"scnr")[0];
            let sm = c.tag_meta(scnr).unwrap();
            let (_, so) = c.block(sm + OFF_SCNR_SKIES).expect("skies block");
            let scen = c.tag_ref_of(so, b"scen").expect("sky scen");
            let model = load_model(&c, model_of(&c, scen).unwrap()).unwrap();
            for &mi in &model.lod0_meshes {
                let md = &model.meshes[mi];
                for &vbi in md.vb.iter().filter(|&&v| v > 0) {
                    let Ok((info, bytes)) = super::super::geometry::read_vb_raw_generic(&c, &model.geometry, vbi as usize) else { continue };
                    if info.stride == 12 {
                        let n = info.count as usize;
                        let (mut mn, mut mx, mut sum) = ([f32::MAX; 3], [f32::MIN; 3], [0f64; 3]);
                        for k in 0..n { for ch in 0..3 { let v = bytes.f32_at(k * 12 + ch * 4); mn[ch] = mn[ch].min(v); mx[ch] = mx[ch].max(v); sum[ch] += v as f64; } }
                        let u8s: Vec<u8> = bytes[..12.min(bytes.len())].to_vec();
                        eprintln!("{map} sky mesh {mi} vb {vbi} kind {} stride 12 n {n}: f32 min {mn:?} max {mx:?} mean [{:.3} {:.3} {:.3}] first bytes {u8s:?}", info.kind, sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64);
                    } else {
                        eprintln!("{map} sky mesh {mi} vb {vbi} kind {} stride {} n {}", info.kind, info.stride, info.count);
                    }
                }
            }
        }
    }
}
