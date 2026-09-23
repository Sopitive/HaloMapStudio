//! Halo 4 BSP render geometry: the `render_geometry_api_resource_definition` resource,
//! the sbsp mesh / bounds / material / cluster blocks and the instance records that live in
//! the `structure_bsp_cache_file_tag_resources` resource. Offsets: docs/halo4_geometry_layout.md.
//!
//! Layout facts (wraparound + ca_forge_ravine cross-checks in the tests):
//!   * the sbsp's own two geometry resources are EMPTY; the lit geometry is the Lbsp's resource
//!     (same tag name as the sbsp), whose definition is
//!     [nvb x 28 B vb interop][nvb x 12 B vb elem][nib x 28 B ib interop][nib x 12 B ib elem][48 B main]
//!   * vb interop {count i32 @0, type i16 @4, stride i16 @6, size i32 @8, page addr (fixup) @20}
//!   * sbsp +0x4A8 = meshes (112 B), +0x4C0 = per-mesh compression bounds (52 B), +0x160 =
//!     materials (44 B, `mat ` ref @0), +0x154 = clusters (128 B, mesh index i16 @0x34)
//!   * vertex type 2 / stride 36: pos f32x3 @0 (normalized to bounds), blend f32 @12, uv u16x2 @16
//!     (normalized), normal snorm16x3 @20, pad @26, tangent snorm16x3 @28 + handedness snorm16 @34;
//!     index buffers are u16 triangle LISTS; cluster meshes carry NO vertex data (vb = ib = -1),
//!     every lit VB belongs to an instance mesh
//!   * instances: 148 B records {scale @0, 3x3 @4, translation @40, mesh i16 @60, world bounds @80}
//! The rotation convention (row vs column) is measured per map by `load_bsp`; the blend scalar
//! @12 is a per-vertex weight / alpha whose exact meaning per shader is not pinned down.

use anyhow::{anyhow, bail, Result};
use glam::{Mat4, Vec3, Vec4};

use super::cache::{ByteRead, H4Cache, ResourceEntry};

pub const OFF_SBSP_MATERIALS: usize = 0x160;
pub const OFF_SBSP_CLUSTERS: usize = 0x154;
pub const OFF_SBSP_MESHES: usize = 0x4A8;
pub const OFF_SBSP_BOUNDS: usize = 0x4C0;
pub const OFF_SBSP_INSTANCE_RID: usize = 0x59C;
pub const OFF_LBSP_GEOMETRY_RID: usize = 0x1D0;
pub const MESH_ELEM: usize = 112;
pub const BOUNDS_ELEM: usize = 52;
pub const MATERIAL_ELEM: usize = 44;
pub const CLUSTER_ELEM: usize = 128;
pub const PART_ELEM: usize = 24;
pub const INSTANCE_ELEM: usize = 148;

#[derive(Clone, Copy, Debug)]
pub struct VbInfo {
    pub count: u32,
    pub kind: i16,
    pub stride: i16,
    pub size: u32,
    pub addr: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct IbInfo {
    pub fmt: i32,
    pub size: u32,
    pub addr: u32,
}

/// A parsed render_geometry_api_resource_definition.
pub struct GeometryResource {
    pub entry: usize,
    pub vbs: Vec<VbInfo>,
    pub ibs: Vec<IbInfo>,
    /// Indices into `vbs` of the buffers wide enough to hold a rigid vertex (stride >= 36), in
    /// order - a FALLBACK reading of a mesh's `vb[0]`, not the primary one.
    ///
    /// A mesh's `vb[0]` is a RAW index into `vbs`: BSP resources list every position buffer first
    /// and append the short-stride lightmap-UV streams (so the two readings coincide), and MODEL
    /// resources interleave them per mesh (`mp_spawn_point`: kinds 2/36, 9/4, 2/36, 9/4 - mesh 1
    /// is vb 2, which the "skip the short-stride entries" reading would resolve to nothing). 13 of
    /// ca_forge_ravine's 689 models interleave this way, so the raw reading is the rule.
    ///
    /// #h2a: `ca_zanzibar_bsp01` is the ONE resource on either engine where the raw reading breaks
    /// - a stray type-4 / stride-4 buffer sits at raw index 1476 inside the position run, so mesh
    /// 1770 resolves to a 4-byte stream and meshes 1771..1871 are each off by one (35 of them then
    /// address vertices past their buffer). `vertex_buffer_fallback` resolves all 36 exactly: each
    /// one's vertex count becomes `max_index + 1`. `decode_mesh_generic` only consults it when the
    /// raw buffer is unusable, so every Halo 4 mesh decodes byte-for-byte as before.
    pub pos_vbs: Vec<usize>,
}

impl GeometryResource {
    /// The buffer a mesh's `vb[0]` names, read RAW (the rule - see `pos_vbs`).
    pub fn vertex_buffer(&self, raw: i16) -> Option<&VbInfo> {
        self.vbs.get(usize::try_from(raw).ok()?)
    }
    /// The fallback reading: `vb[0]` as an index into the position-capable buffers only.
    pub fn vertex_buffer_fallback(&self, logical: i16) -> Option<&VbInfo> {
        self.vbs.get(*self.pos_vbs.get(usize::try_from(logical).ok()?)?)
    }
}

pub fn parse_geometry_resource(c: &H4Cache, e: &ResourceEntry) -> Result<GeometryResource> {
    if !c.kind_is(e.kind, H4Cache::RES_GEOMETRY) { bail!("resource {} is type {}, not render geometry", e.index, e.kind); }
    let dd = c.definition(e);
    let base = (e.def_addr & 0x0FFF_FFFF) as usize;
    if base + 48 > dd.len() { bail!("geometry definition too short ({} B, main @{base:#x})", dd.len()); }
    let nvb = dd.i32_at(base + 0x18).max(0) as usize;
    let nib = dd.i32_at(base + 0x24).max(0) as usize;
    let mut g = GeometryResource { entry: e.index, vbs: Vec::with_capacity(nvb), ibs: Vec::with_capacity(nib), pos_vbs: Vec::new() };
    if nvb > 0 {
        let vbo = (e.fixup_at((base + 0x1C) as u32).ok_or_else(|| anyhow!("vb block fixup"))? & 0x0FFF_FFFF) as usize;
        for i in 0..nvb {
            let io = (e.fixup_at((vbo + i * 12) as u32).ok_or_else(|| anyhow!("vb {i} interop fixup"))? & 0x0FFF_FFFF) as usize;
            g.vbs.push(VbInfo {
                count: dd.i32_at(io).max(0) as u32,
                kind: dd.i16_at(io + 4),
                stride: dd.i16_at(io + 6),
                size: dd.i32_at(io + 8).max(0) as u32,
                addr: e.fixup_at((io + 20) as u32).unwrap_or(0),
            });
        }
    }
    if nib > 0 {
        let ibo = (e.fixup_at((base + 0x28) as u32).ok_or_else(|| anyhow!("ib block fixup"))? & 0x0FFF_FFFF) as usize;
        for i in 0..nib {
            let io = (e.fixup_at((ibo + i * 12) as u32).ok_or_else(|| anyhow!("ib {i} interop fixup"))? & 0x0FFF_FFFF) as usize;
            g.ibs.push(IbInfo { fmt: dd.i32_at(io), size: dd.i32_at(io + 8).max(0) as u32, addr: e.fixup_at((io + 20) as u32).unwrap_or(0) });
        }
    }
    g.pos_vbs = (0..g.vbs.len()).filter(|&i| g.vbs[i].stride >= 36).collect();
    Ok(g)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Part {
    pub material: i16,
    pub index_start: u32,
    pub index_count: u32,
}

/// Per-mesh compression bounds (52-B element: i32 flags @0, then the six position and four uv
/// extents as f32).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds {
    /// i32 @0 of the element. Halo 4: 3 on every BSP / model mesh = "position and uv are
    /// normalized to these extents". #h2a: one H2A cluster mesh (ca_zanzibar mesh 1764) carries
    /// 0 here with all-zero extents and an uncompressed type-1 vertex buffer, so bit 0 =
    /// position compressed, bit 1 = uv compressed (see `is_degenerate`).
    pub flags: i32,
    pub pos_min: [f32; 3],
    pub pos_max: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

impl Bounds {
    /// No usable extents (all-zero element): the vertex buffer's positions / uvs are absolute,
    /// not normalized. Decoding through a zero range would collapse the mesh onto `pos_min`.
    pub fn is_degenerate(&self) -> bool {
        (0..3).all(|k| self.pos_max[k] == self.pos_min[k])
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshDef {
    pub parts: Vec<Part>,
    pub vb: [i16; 8],
    pub ib: [i16; 2],
    pub bounds: Bounds,
    /// u8 @0x34 of the mesh element: 3 = triangle list, 5 = triangle strip (models use both).
    pub index_type: u8,
}

impl MeshDef {
    pub fn has_geometry(&self) -> bool { !self.parts.is_empty() && self.vb[0] >= 0 && (self.ib[0] >= 0 || self.ib[1] >= 0) }
    /// The index buffer slot in use (slot 1 on every wraparound mesh; slot 0 kept as fallback).
    pub fn ib_index(&self) -> i16 { if self.ib[1] >= 0 { self.ib[1] } else { self.ib[0] } }
}

#[derive(Clone, Copy, Debug)]
pub struct H4Vertex {
    pub pos: [f32; 3],
    /// snorm16x3 @20 (unit; 98.7% of triangles face outward, uniform over octants).
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// snorm16x3 @28 + handedness snorm16 @34 (+-1): binormal = cross(normal, tangent) * w.
    pub tangent: [f32; 4],
    /// f32 @12 in 0..1: per-vertex blend weight / vertex alpha (0 on every wraparound vertex,
    /// non-zero on ravine cliff / terrain-blend / foam meshes; `*_vertmask` shaders read it as
    /// the opacity mask, layered terrain as a layer weight).
    pub blend: f32,
}

/// snorm16 -> f32 (clamped so -32768 decodes to -1.0).
#[inline]
pub fn sn16(v: i16) -> f32 { (v as f32 / 32767.0).max(-1.0) }

pub struct DecodedMesh {
    pub verts: Vec<H4Vertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug)]
pub struct Instance {
    pub mesh: i16,
    pub scale: f32,
    pub rot: [f32; 9],
    pub translation: [f32; 3],
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
}

impl Instance {
    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.rot == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] && self.translation == [0.0; 3]
    }
    /// world = scale * (v.x * row0 + v.y * row1 + v.z * row2) + T  (Reach's affine convention).
    pub fn matrix_rows(&self) -> Mat4 {
        let r = &self.rot;
        let s = self.scale;
        Mat4::from_cols(
            Vec4::new(r[0] * s, r[1] * s, r[2] * s, 0.0),
            Vec4::new(r[3] * s, r[4] * s, r[5] * s, 0.0),
            Vec4::new(r[6] * s, r[7] * s, r[8] * s, 0.0),
            Vec4::new(self.translation[0], self.translation[1], self.translation[2], 1.0),
        )
    }
    /// The transposed reading (columns are the axes).
    pub fn matrix_cols(&self) -> Mat4 {
        let r = &self.rot;
        let s = self.scale;
        Mat4::from_cols(
            Vec4::new(r[0] * s, r[3] * s, r[6] * s, 0.0),
            Vec4::new(r[1] * s, r[4] * s, r[7] * s, 0.0),
            Vec4::new(r[2] * s, r[5] * s, r[8] * s, 0.0),
            Vec4::new(self.translation[0], self.translation[1], self.translation[2], 1.0),
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct BspMaterial {
    pub mat_tag: Option<usize>,
    pub name: String,
    /// Texture parameter 0 of the `mat` (the `_diff` map on all 69 wraparound materials).
    pub diffuse_bitm: Option<usize>,
    /// Number of texture parameters (mat +0x1C block -> element sub-block @0).
    pub texture_params: usize,
}

pub struct H4Bsp {
    pub sbsp_tag: usize,
    pub lbsp_tag: Option<usize>,
    pub name: String,
    pub meshes: Vec<MeshDef>,
    pub geometry: Option<GeometryResource>,
    pub clusters: Vec<i16>,
    /// Per-cluster world bounds (cluster element @0: xmin,xmax,ymin,ymax,zmin,zmax), same order as `clusters`.
    pub cluster_bounds: Vec<([f32; 3], [f32; 3])>,
    pub instances: Vec<Instance>,
    pub materials: Vec<BspMaterial>,
    /// (row-convention hits, column-convention hits, non-identity instances tested) from
    /// transforming each mesh's bounds and testing containment in the record's world bounds.
    pub rot_check: (usize, usize, usize),
    /// true when the row (Reach) convention is used for instance matrices.
    pub rows: bool,
}

impl H4Bsp {
    pub fn instance_matrix(&self, i: &Instance) -> Mat4 { if self.rows { i.matrix_rows() } else { i.matrix_cols() } }
}

fn read_bounds(d: &[u8], o: usize) -> Bounds {
    let f = |k: usize| d.f32_at(o + 4 + k * 4);
    Bounds {
        flags: d.i32_at(o),
        pos_min: [f(0), f(2), f(4)],
        pos_max: [f(1), f(3), f(5)],
        uv_min: [f(6), f(8)],
        uv_max: [f(7), f(9)],
    }
}

/// Parse one BSP's mesh table, materials, clusters, instances and locate its geometry resource.
pub fn load_bsp(c: &H4Cache, sbsp_tag: usize) -> Result<H4Bsp> {
    let d = c.data();
    let sb = c.tag_meta(sbsp_tag).ok_or_else(|| anyhow!("sbsp meta"))?;
    let name = c.tag_name(sbsp_tag).to_string();
    // meshes + bounds
    let (nm, om) = c.block(sb + OFF_SBSP_MESHES).unwrap_or((0, 0));
    let (nb, ob) = c.block(sb + OFF_SBSP_BOUNDS).unwrap_or((0, 0));
    let mut meshes = Vec::with_capacity(nm);
    for i in 0..nm {
        let e = om + i * MESH_ELEM;
        let mut m = MeshDef::default();
        if let Some((pc, pp)) = c.block(e) {
            for k in 0..pc.min(4096) {
                let p = pp + k * PART_ELEM;
                m.parts.push(Part { material: d.i16_at(p), index_start: d.i32_at(p + 4).max(0) as u32, index_count: d.i32_at(p + 8).max(0) as u32 });
            }
        }
        for k in 0..8 { m.vb[k] = d.i16_at(e + 0x18 + k * 2); }
        m.ib = [d.i16_at(e + 0x28), d.i16_at(e + 0x2A)];
        m.index_type = d.u8_at(e + 0x34);
        if i < nb { m.bounds = read_bounds(d, ob + i * BOUNDS_ELEM); }
        meshes.push(m);
    }
    // materials
    let mut materials = Vec::new();
    if let Some((n, o)) = c.block(sb + OFF_SBSP_MATERIALS) {
        for i in 0..n {
            let mut bm = BspMaterial::default();
            if let Some(mt) = c.tag_ref_of(o + i * MATERIAL_ELEM, b"mat ") {
                bm.mat_tag = Some(mt);
                bm.name = c.tag_name(mt).to_string();
                if let Some(mm) = c.tag_meta(mt) {
                    if let Some((_, pe)) = c.block(mm + 0x1C) {
                        if let Some((tn, tp)) = c.block(pe) {
                            bm.texture_params = tn;
                            bm.diffuse_bitm = c.tag_ref_of(tp, b"bitm");
                        }
                    }
                }
            }
            materials.push(bm);
        }
    }
    // clusters -> mesh index
    let mut clusters = Vec::new();
    let mut cluster_bounds = Vec::new();
    if let Some((n, o)) = c.block(sb + OFF_SBSP_CLUSTERS) {
        for i in 0..n {
            let e = o + i * CLUSTER_ELEM;
            let mi = d.i16_at(e + 0x34);
            if mi >= 0 && (mi as usize) < meshes.len() {
                clusters.push(mi);
                let f = |k: usize| d.f32_at(e + k * 4);
                cluster_bounds.push(([f(0), f(2), f(4)], [f(1), f(3), f(5)]));
            }
        }
    }
    // geometry resource: the Lbsp with the same tag name
    let lbsp_tag = c.find_tags(b"Lbsp").into_iter().find(|&t| c.tag_name(t) == name);
    let mut geometry = None;
    if let Some(lt) = lbsp_tag {
        if let Some(lm) = c.tag_meta(lt) {
            let is_geo = |e: &&ResourceEntry| c.kind_is(e.kind, H4Cache::RES_GEOMETRY) && e.owner == *b"Lbsp";
            let mut entry = c.resource_by_id(d.u32_at(lm + OFF_LBSP_GEOMETRY_RID)).filter(is_geo);
            if entry.is_none() {
                // fall back: scan the Lbsp main struct for the largest geometry resource it owns
                let mut best: Option<&ResourceEntry> = None;
                for off in (0..0x600).step_by(4) {
                    if let Some(e) = c.resource_by_id(d.u32_at(lm + off)) {
                        if is_geo(&e) && best.map_or(true, |b| e.def_len > b.def_len) { best = Some(e); }
                    }
                }
                entry = best;
            }
            if let Some(e) = entry { geometry = Some(parse_geometry_resource(c, e)?); }
        }
    }
    // instances: sbsp +0x59C -> type-5 resource -> main struct block @+0x30 (page data, 148 B)
    let mut instances = Vec::new();
    if let Some(e) = c.resource_by_id(d.u32_at(sb + OFF_SBSP_INSTANCE_RID)).filter(|e| c.kind_is(e.kind, H4Cache::RES_BSP_INSTANCES)) {
        let dd = c.definition(e);
        let base = (e.def_addr & 0x0FFF_FFFF) as usize;
        let count = dd.i32_at(base + 0x30).max(0) as usize;
        if let Some(addr) = e.fixup_at((base + 0x34) as u32) {
            if count > 0 && count < 1_000_000 {
                let raw = c.stream_bytes(e, addr, count * INSTANCE_ELEM)?;
                for i in 0..count {
                    let r = &raw[i * INSTANCE_ELEM..(i + 1) * INSTANCE_ELEM];
                    let f = |o: usize| r.f32_at(o);
                    instances.push(Instance {
                        mesh: r.i16_at(60),
                        scale: f(0),
                        rot: [f(4), f(8), f(12), f(16), f(20), f(24), f(28), f(32), f(36)],
                        translation: [f(40), f(44), f(48)],
                        world_min: [f(80), f(88), f(96)],
                        world_max: [f(84), f(92), f(100)],
                    });
                }
            }
        }
    }
    // rotation convention check: transformed mesh bounds must fit the record's world bounds
    let mut rot_check = (0usize, 0usize, 0usize);
    for inst in instances.iter().filter(|i| !i.is_identity()) {
        let Some(m) = meshes.get(inst.mesh.max(0) as usize) else { continue };
        if inst.mesh < 0 || !m.has_geometry() { continue; }
        let wmin = Vec3::from(inst.world_min);
        let wmax = Vec3::from(inst.world_max);
        if (wmax - wmin).min_element() < 0.0 || (wmax - wmin).length() < 1e-3 { continue; }
        rot_check.2 += 1;
        let fits = |mat: Mat4| -> bool {
            let (mn, mx) = (Vec3::from(m.bounds.pos_min), Vec3::from(m.bounds.pos_max));
            let tol = (wmax - wmin).length() * 0.02 + 0.05;
            (0..8).all(|k| {
                let p = Vec3::new(if k & 1 == 0 { mn.x } else { mx.x }, if k & 2 == 0 { mn.y } else { mx.y }, if k & 4 == 0 { mn.z } else { mx.z });
                let w = mat.transform_point3(p);
                w.cmpge(wmin - tol).all() && w.cmple(wmax + tol).all()
            })
        };
        if fits(inst.matrix_rows()) { rot_check.0 += 1; }
        if fits(inst.matrix_cols()) { rot_check.1 += 1; }
    }
    let rows = rot_check.0 >= rot_check.1;
    Ok(H4Bsp { sbsp_tag, lbsp_tag, name, meshes, geometry, clusters, cluster_bounds, instances, materials, rot_check, rows })
}

/// Decode one mesh's vertices (world/local space per its bounds) and its u16 triangle list.
pub fn decode_mesh(c: &H4Cache, bsp: &H4Bsp, mesh_idx: usize) -> Result<Option<DecodedMesh>> {
    let m = bsp.meshes.get(mesh_idx).ok_or_else(|| anyhow!("mesh {mesh_idx} out of range"))?;
    if !m.has_geometry() { return Ok(None); }
    let g = bsp.geometry.as_ref().ok_or_else(|| anyhow!("bsp has no geometry resource"))?;
    // Shared decoder (objects.rs): vertex types {2,3,5,18,47,48} with stride >= 36 all carry
    // the same first 36 bytes (checked on 138 BSP meshes / 97k verts + model samples of every
    // type): pos f32x3 @0, blend f32 @12, uv u16x2 @16, normal snorm16x3 @20 (not octahedral -
    // the oct16 reading scored 0.80 vs 0.97 against the geometric normals), pad @26, tangent
    // snorm16x3 @28, handedness snorm16 @34. Index type 5 (strips) is converted to a list.
    super::objects::decode_mesh_generic(c, g, m, m.index_type).map_err(|e| anyhow!("mesh {mesh_idx}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// The numbers measured by the scratch probes on wraparound (docs/halo4_geometry_layout.md).
    #[test]
    fn wraparound_bsp_layout() {
        let Some(c) = open("wraparound.map") else { return };
        let sbsps = c.find_tags(b"sbsp");
        assert_eq!(sbsps.len(), 1);
        let bsp = load_bsp(&c, sbsps[0]).unwrap();
        assert_eq!(bsp.name, "levels\\multi\\wraparound\\wraparound_000");
        assert!(bsp.lbsp_tag.is_some());
        let g = bsp.geometry.as_ref().expect("geometry resource");
        assert_eq!((g.vbs.len(), g.ibs.len()), (1238, 685));
        assert_eq!(g.vbs.iter().filter(|v| v.kind == 2 && v.stride == 36).count(), 685);
        assert_eq!(g.vbs.iter().filter(|v| v.kind == 4 && v.stride == 4).count(), 553);
        // the logical position-buffer list: identity on Halo 4 (every short-stride stream is
        // appended after the 685 position buffers), and as long as the index-buffer list
        assert_eq!(g.pos_vbs.len(), g.ibs.len());
        assert!(g.pos_vbs.iter().enumerate().all(|(i, &r)| i == r), "Halo 4 BSP pos_vbs is the identity");
        assert!(g.vbs.iter().all(|v| v.size == v.count * v.stride as u32));
        assert!(g.ibs.iter().all(|i| i.fmt == 3));
        assert_eq!(bsp.meshes.len(), 1008);
        assert_eq!(bsp.meshes.iter().filter(|m| m.has_geometry()).count(), 685);
        assert_eq!(bsp.materials.len(), 69);
        assert!(bsp.materials.iter().all(|m| m.mat_tag.is_some()));
        assert!(bsp.materials.iter().all(|m| m.diffuse_bitm.is_some()), "every material resolves a diffuse bitmap");
        assert_eq!(bsp.clusters, vec![990, 991, 992, 993, 994, 995, 996, 997]);
        assert_eq!(bsp.instances.len(), 1046);
        assert_eq!(bsp.instances.iter().filter(|i| i.is_identity()).count(), 771);
        assert_eq!(bsp.cluster_bounds.len(), 8);
        assert!(bsp.cluster_bounds.iter().all(|(mn, mx)| (0..3).all(|k| mn[k] <= mx[k])));
        assert!(bsp.instances.iter().all(|i| i.mesh >= 0 && (i.mesh as usize) < 1008));
        // mesh 1: 1104 verts / 2751 indices, a wall spanning x 13..95, z 13..173
        let m1 = decode_mesh(&c, &bsp, 1).unwrap().unwrap();
        assert_eq!((m1.verts.len(), m1.indices.len(), m1.parts.len()), (1104, 2751, 7));
        assert!(m1.indices.iter().all(|&i| (i as usize) < m1.verts.len()));
        let mn = m1.verts.iter().fold([f32::MAX; 3], |a, v| [a[0].min(v.pos[0]), a[1].min(v.pos[1]), a[2].min(v.pos[2])]);
        let mx = m1.verts.iter().fold([f32::MIN; 3], |a, v| [a[0].max(v.pos[0]), a[1].max(v.pos[1]), a[2].max(v.pos[2])]);
        assert!((mn[0] - 13.356).abs() < 0.01 && (mx[0] - 94.906).abs() < 0.01, "{mn:?} {mx:?}");
        assert!((mn[2] - 13.018).abs() < 0.01 && (mx[2] - 173.11).abs() < 0.01, "{mn:?} {mx:?}");
        assert!(m1.verts.iter().all(|v| { let n = v.normal; ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-3 }));
        // triangle-list facing: decoded vertex normals agree with the geometric face normals
        let mut agree = 0usize;
        let mut total = 0usize;
        for t in m1.indices.chunks_exact(3) {
            let p = |i: u32| Vec3::from(m1.verts[i as usize].pos);
            let fnrm = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).normalize_or_zero();
            let vn = (Vec3::from(m1.verts[t[0] as usize].normal) + Vec3::from(m1.verts[t[1] as usize].normal) + Vec3::from(m1.verts[t[2] as usize].normal)).normalize_or_zero();
            total += 1;
            if fnrm.dot(vn) > 0.7 { agree += 1; }
        }
        assert!(agree * 100 / total >= 95, "facing agreement {agree}/{total} (snorm16x3 normals; the octahedral reading scored 83%)");
        // tangent frame: unit, perpendicular to the normal, handedness exactly +-1
        let mut nt = 0.0f32;
        for v in &m1.verts {
            let (n, t) = (Vec3::from(v.normal), Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]));
            assert!((t.length() - 1.0).abs() < 2e-2, "tangent length {}", t.length());
            assert!(v.tangent[3].abs() > 0.99, "handedness {}", v.tangent[3]);
            nt += n.dot(t).abs();
        }
        let mean_nt = nt / (m1.verts.len() as f32);
        assert!(mean_nt < 0.01, "mean |n.t| {mean_nt}");
        // every mesh with geometry decodes and its parts stay inside the index buffer
        for i in 0..bsp.meshes.len() {
            if let Some(dm) = decode_mesh(&c, &bsp, i).unwrap() {
                for p in &dm.parts {
                    assert!((p.index_start + p.index_count) as usize <= dm.indices.len(), "mesh {i} part range");
                    assert!(p.material >= 0 && (p.material as usize) < bsp.materials.len(), "mesh {i} material");
                }
            }
        }
        eprintln!("rot_check row/col/tested = {:?}", bsp.rot_check);
    }

    #[test]
    fn ravine_bsps_load() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let sbsps = c.find_tags(b"sbsp");
        assert_eq!(sbsps.len(), 2);
        for &t in &sbsps {
            let bsp = load_bsp(&c, t).unwrap();
            let g = bsp.geometry.as_ref().expect("geometry");
            assert!(!g.vbs.is_empty() && !g.ibs.is_empty());
            assert!(g.vbs.iter().all(|v| v.size == v.count * v.stride as u32));
            let n_geo = bsp.meshes.iter().filter(|m| m.has_geometry()).count();
            assert!(n_geo > 0);
            assert!(bsp.meshes.iter().filter(|m| m.has_geometry()).all(|m| g.vertex_buffer(m.vb[0]).is_some() && (m.ib_index() as usize) < g.ibs.len()));
            eprintln!("{}: meshes {} (geo {}) vbs {} ibs {} clusters {} instances {} materials {} rot_check {:?}",
                bsp.name, bsp.meshes.len(), n_geo, g.vbs.len(), g.ibs.len(), bsp.clusters.len(), bsp.instances.len(), bsp.materials.len(), bsp.rot_check);
        }
    }
}
