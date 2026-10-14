//! Halo 3 (MCC) BSP render geometry. The lit geometry lives in the `Lbsp` tag (the sbsp's own
//! mesh block is empty, exactly like Halo 4). Layout (Halo3MCC plugins, verified on guardian):
//!   * Lbsp +0x134 Meshes (0x4C), +0x140 Compression bounds (0x2C), +0x10C clusters, +0x118 instances
//!   * sbsp +0xC4 Clusters (0x118, mesh index i16 @0xD8), +0xD0 Materials (0x24, rmsh ref @0),
//!     +0x1C0 Instanced-geometry instances (0x78: scale f32 @0, forward/left/up vec3 @4/@0x10/@0x1C,
//!     position @0x28, definition index i16 @0x34, flags16 @0x36 (bit5 = not for render),
//!     lightmap-texcoord block i16 @0x38, world bounding sphere @0x40 + radius @0x4C, name sid @0x54)
//!   * sbsp +0x38C `[zone] asset datum` -> the `structure_bsp_tag_resources` resource, whose control
//!     root +0x18 is the Instanced-geometry DEFINITIONS block (0xC4, control-data fixup @+0x1C):
//!     checksum @0, local bounding sphere @4 + radius @0x10, collision bsp blocks @0x14..0x74,
//!     mesh index i16 @0xBC, compression index i16 @0xBE. Instance meshes and cluster meshes share
//!     the Lbsp mesh table (guardian: defs 0..211 <-> meshes 0..211, clusters use 212..237).
//!   * mesh (0x4C): parts @0 (0x10: renderMethodIndex i16 @0, indexStart u16 @4, indexCount u16 @6),
//!     vertex-buffer index [8] @0x18, index-buffer index @0x28, vertex type @0x2E, index type @0x30
//!   * compression (0x2C): flags16 @0 (bit0 pos, bit1 uv), pos (min,max) pairs @4/@0xC/@0x14,
//!     uv (min,max) pairs @0x1C/@0x24
//!
//! The vertex/index buffer DESCRIPTORS (count/type/stride/size + a data offset) come from the
//! geometry resource's CONTROL data; the raw bytes from its pageable page (cache.rs). Control
//! layout: main struct at `root`; nvb @root+0x18, vb-list ptr @root+0x1C (fixup); nib @root+0x24,
//! ib-list ptr @root+0x28 (fixup). vb-list / ib-list are nvb / nib 12-byte elements, each with a
//! fixup to a 28-byte interop. VB interop {count i32 @0, type i16 @4, stride i16 @6, size i32 @8,
//! data-offset fixup @+20}; IB interop {fmt i32 @0, size i32 @4 (bytes), data-offset fixup @+16}.
//! The data offset is a byte offset into this resource's pageable slice (VBs first, then IBs).
//!
//! Vertex layout (stride 36 rigid/world, verified on guardian): pos int16x3 @0 (SNORM->pos
//! bounds), uv int16x2 @8 (SNORM->uv bounds), normal snorm16x3 @0x14, tangent snorm16x3 @0x1C +
//! handedness snorm16 @0x22. Index buffers are u16 (type 3 = list, 5 = strip).

use anyhow::{anyhow, bail, Result};
use glam::{Mat4, Vec3, Vec4};

use super::cache::{H3Cache, ResourceEntry};
use crate::h4::cache::ByteRead;

pub const OFF_SBSP_CLUSTERS: usize = 0xC4;
pub const OFF_SBSP_MATERIALS: usize = 0xD0;
pub const OFF_SBSP_INSTANCES: usize = 0x1C0;
pub const OFF_SBSP_TAG_RESOURCES: usize = 0x38C;
pub const OFF_LBSP_MESHES: usize = 0x134;
pub const OFF_LBSP_COMPRESSION: usize = 0x140;
pub const MESH_ELEM: usize = 0x4C;
pub const BOUNDS_ELEM: usize = 0x2C;
pub const PART_ELEM: usize = 0x10;
pub const MATERIAL_ELEM: usize = 0x24;
pub const CLUSTER_ELEM: usize = 0x118;
pub const INSTANCE_ELEM: usize = 0x78;
/// Instance record flags16 @0x36, bit 5.
pub const INSTANCE_FLAG_NOT_FOR_RENDER: u16 = 1 << 5;
/// `structure_bsp_tag_resources` control root: instanced-geometry definitions block (count @+0x18,
/// control-data fixup @+0x1C).
pub const OFF_TAGRES_INSTANCE_DEFS: usize = 0x18;
pub const INSTANCE_DEF_ELEM: usize = 0xC4;
pub const INSTANCE_DEF_MESH: usize = 0xBC;
pub const INSTANCE_DEF_COMPRESSION: usize = 0xBE;

#[derive(Clone, Copy, Debug)]
pub struct VbInfo {
    pub count: u32,
    pub kind: i16,
    pub stride: i16,
    pub size: u32,
    /// Byte offset of this buffer within the resource's pageable slice, or None (no fixup).
    pub off: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct IbInfo {
    pub fmt: i32,
    pub size: u32,
    pub off: Option<u32>,
}

pub struct GeometryResource {
    /// Index into `H3Cache::resources` (so the pageable data can be read).
    pub entry: usize,
    pub vbs: Vec<VbInfo>,
    pub ibs: Vec<IbInfo>,
    /// Indices into `vbs` of the position-capable buffers (stride >= 36), in order. A mesh's
    /// `vb[0]` is normally a RAW index into `vbs`, but some BSPs interleave short-stride
    /// (lightmap-UV) streams inside the position run, so the raw index lands on the wrong buffer;
    /// this is the fallback reading (like the Halo 4 `pos_vbs`).
    pub pos_vbs: Vec<usize>,
}

pub fn parse_geometry_resource(c: &H3Cache, e: &ResourceEntry) -> Result<GeometryResource> {
    if !c.kind_is(e.kind, H3Cache::RES_GEOMETRY) {
        bail!("resource {} is type {}, not render geometry", e.index, e.kind);
    }
    let cd = c.control_data(e);
    let fx = c.resource_fixups(e);
    let root = e.root as usize;
    if root + 0x2C > cd.len() { bail!("geometry control data too short ({} B, root {root})", cd.len()); }
    let val = |loc: u32| fx.get(&loc).map(|(v, _)| *v as usize);
    let nvb = cd.i32_at(root + 0x18).max(0) as usize;
    let nib = cd.i32_at(root + 0x24).max(0) as usize;
    let vblist = val((root + 0x1C) as u32).ok_or_else(|| anyhow!("vb-list fixup"))?;
    let iblist = val((root + 0x28) as u32).ok_or_else(|| anyhow!("ib-list fixup"))?;
    let mut vbs = Vec::with_capacity(nvb);
    for i in 0..nvb {
        let io = match val((vblist + i * 12) as u32) { Some(v) => v, None => { vbs.push(VbInfo { count: 0, kind: 0, stride: 0, size: 0, off: None }); continue } };
        vbs.push(VbInfo {
            count: cd.i32_at(io).max(0) as u32,
            kind: cd.i16_at(io + 4),
            stride: cd.i16_at(io + 6),
            size: cd.i32_at(io + 8).max(0) as u32,
            off: val((io + 20) as u32).map(|v| v as u32),
        });
    }
    let mut ibs = Vec::with_capacity(nib);
    for i in 0..nib {
        let io = match val((iblist + i * 12) as u32) { Some(v) => v, None => { ibs.push(IbInfo { fmt: 0, size: 0, off: None }); continue } };
        ibs.push(IbInfo {
            fmt: cd.i32_at(io),
            size: cd.i32_at(io + 4).max(0) as u32,
            off: val((io + 16) as u32).map(|v| v as u32),
        });
    }
    let pos_vbs = (0..vbs.len()).filter(|&i| vbs[i].stride >= 36 && vbs[i].off.is_some()).collect();
    Ok(GeometryResource { entry: e.index, vbs, ibs, pos_vbs })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Part {
    pub material: i16,
    pub index_start: u32,
    pub index_count: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds {
    pub flags: u16,
    pub pos_min: [f32; 3],
    pub pos_max: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct MeshDef {
    pub parts: Vec<Part>,
    pub vb: [i16; 8],
    pub ib: i16,
    pub bounds: Bounds,
    pub vertex_type: u8,
    pub index_type: u8,
}

impl MeshDef {
    pub fn has_geometry(&self) -> bool { !self.parts.is_empty() && self.vb[0] >= 0 && self.ib >= 0 }
}

#[derive(Clone, Copy, Debug)]
pub struct H3Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone)]
pub struct DecodedMesh {
    pub verts: Vec<H3Vertex>,
    pub indices: Vec<u32>,
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug, Default)]
pub struct BspMaterial {
    /// The render-method tag (rmsh / rmtr / rmd / rmw / rmcs / ...), or None when null.
    pub render_tag: Option<usize>,
    /// Its class fourcc.
    pub class: [u8; 4],
    pub name: String,
}

/// One instanced-geometry definition (from the sbsp's tag-resources control data): the shared
/// mesh + compression bounds every placement of this piece uses.
#[derive(Clone, Copy, Debug, Default)]
pub struct InstanceDef {
    pub mesh: i16,
    pub compression: i16,
    /// Local-space bounding sphere (center xyz, radius) - the convention check transforms it.
    pub sphere: [f32; 4],
}

/// One instanced-geometry placement (sbsp +0x1C0 record) resolved through its definition.
#[derive(Clone, Debug)]
pub struct Instance {
    /// Record index in the sbsp instance block (= the Lbsp +0x118 instance-lighting index).
    pub index: usize,
    pub def: i16,
    /// Lbsp mesh index (from the definition), -1 when the definition is unresolved.
    pub mesh: i16,
    /// Compression-bounds index (from the definition), -1 when unresolved.
    pub compression: i16,
    pub scale: f32,
    /// forward / left / up rows (record @4 / @0x10 / @0x1C).
    pub rot: [f32; 9],
    pub translation: [f32; 3],
    pub flags: u16,
    /// Record @0x38: index into the Lbsp +0x1A0 per-instance lightmap-texcoord blocks (the
    /// instance's own u16x2 lightmap-UV stream), -1 when it has none (per-vertex / probe lit).
    pub lm_texcoord_block: i16,
    /// World-space bounding sphere from the record (center xyz, radius).
    pub world_sphere: [f32; 4],
    pub name: String,
}

impl Instance {
    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.rot == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] && self.translation == [0.0; 3]
    }
    pub fn not_for_render(&self) -> bool { self.flags & INSTANCE_FLAG_NOT_FOR_RENDER != 0 }
    /// Every transform component is finite and the scale is usable.
    pub fn is_finite(&self) -> bool {
        self.scale.is_finite() && self.scale.abs() > 1e-6
            && self.rot.iter().all(|v| v.is_finite()) && self.translation.iter().all(|v| v.is_finite())
    }
    /// world = scale * (v.x * forward + v.y * left + v.z * up) + T (the Reach / Halo 4 row
    /// convention: the three record vectors are the local axes in world space).
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

pub struct H3Bsp {
    pub sbsp_tag: usize,
    pub lbsp_tag: Option<usize>,
    pub name: String,
    pub meshes: Vec<MeshDef>,
    /// The Lbsp compression-bounds table (mesh i defaults to bounds i; an instance definition
    /// may name a different entry through its compression index).
    pub bounds: Vec<Bounds>,
    pub geometry: Option<GeometryResource>,
    pub clusters: Vec<i16>,
    /// The sbsp cluster element index behind each `clusters` entry (= the Lbsp +0x10C cluster
    /// lighting index; clusters without a mesh are not listed, so the two are not parallel).
    pub cluster_ids: Vec<usize>,
    pub cluster_bounds: Vec<([f32; 3], [f32; 3])>,
    pub materials: Vec<BspMaterial>,
    pub instance_defs: Vec<InstanceDef>,
    pub instances: Vec<Instance>,
    /// (row-convention hits, column-convention hits, non-identity instances tested): each
    /// instance's LOCAL definition sphere transformed by the candidate matrix must land on the
    /// record's WORLD sphere.
    pub rot_check: (usize, usize, usize),
    /// true when the row (Reach) convention is used for instance matrices.
    pub rows: bool,
}

impl H3Bsp {
    pub fn instance_matrix(&self, i: &Instance) -> Mat4 { if self.rows { i.matrix_rows() } else { i.matrix_cols() } }
    /// The compression bounds an instance's mesh dequantises with (its definition's compression
    /// index when that names a different table entry than the mesh index).
    pub fn instance_bounds(&self, i: &Instance) -> Option<&Bounds> {
        if i.compression >= 0 && i.compression != i.mesh { self.bounds.get(i.compression as usize) } else { None }
    }
}

#[inline]
fn sn16(v: i16) -> f32 { (v as f32 / 32767.0).max(-1.0) }
/// UNSIGNED normalised 16-bit (0..65535) remapped into [min,max]: Halo 3's compressed position
/// and texcoord format (Gen3 `UInt16N`). Verified on guardian against the instance definitions'
/// own bounding spheres: this mapping puts every mesh centre exactly on its sphere centre
/// (median offset 0.000 r, radius ratio 0.98); the SNORM16 reading was 0.06 r off with ratio
/// 0.83, and every coordinate >= 32768 wrapped negative and folded that vertex to the far side
/// of the bounds - the "exploded shards / streaked trunks" on every instance-heavy map.
#[inline]
fn dq(v: i16, mn: f32, mx: f32) -> f32 { mn + ((v as u16) as f32 / 65535.0) * (mx - mn) }

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

/// Parse a BSP's mesh table, materials, clusters and locate its geometry resource.
pub fn load_bsp(c: &H3Cache, sbsp_tag: usize) -> Result<H3Bsp> {
    let d = c.data();
    let name = c.tag_name(sbsp_tag).to_string();
    // The Lbsp holds the real geometry; find the one with the same tag name.
    let lbsp_tag = c.find_tags(b"Lbsp").into_iter().find(|&t| c.tag_name(t) == name)
        .or_else(|| c.find_tags(b"Lbsp").first().copied());
    // meshes + bounds from the Lbsp
    let mut meshes = Vec::new();
    let mut bounds = Vec::new();
    if let Some(lt) = lbsp_tag {
        if let Some(lb) = c.tag_meta(lt) {
            let (nm, om) = c.block(lb + OFF_LBSP_MESHES).unwrap_or((0, 0));
            let (nb, ob) = c.block(lb + OFF_LBSP_COMPRESSION).unwrap_or((0, 0));
            for i in 0..nb { bounds.push(read_bounds(d, ob + i * BOUNDS_ELEM)); }
            for i in 0..nm {
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
                if i < nb { m.bounds = read_bounds(d, ob + i * BOUNDS_ELEM); }
                meshes.push(m);
            }
        }
    }
    // materials: rmsh render-method tag refs
    let mut materials = Vec::new();
    if let Some(sb) = c.tag_meta(sbsp_tag) {
        if let Some((n, o)) = c.block(sb + OFF_SBSP_MATERIALS) {
            for i in 0..n {
                let mut bm = BspMaterial::default();
                if let Some((cls, mt)) = c.tag_ref(o + i * MATERIAL_ELEM) {
                    bm.render_tag = Some(mt);
                    bm.class = cls;
                    bm.name = c.tag_name(mt).to_string();
                }
                materials.push(bm);
            }
        }
    }
    // clusters -> mesh index (+ world bounds)
    let mut clusters = Vec::new();
    let mut cluster_ids = Vec::new();
    let mut cluster_bounds = Vec::new();
    if let Some(sb) = c.tag_meta(sbsp_tag) {
        if let Some((n, o)) = c.block(sb + OFF_SBSP_CLUSTERS) {
            for i in 0..n {
                let e = o + i * CLUSTER_ELEM;
                let mi = d.i16_at(e + 0xD8);
                if mi >= 0 && (mi as usize) < meshes.len() {
                    clusters.push(mi);
                    cluster_ids.push(i);
                    let f = |k: usize| d.f32_at(e + k);
                    cluster_bounds.push(([f(0), f(8), f(0x10)], [f(4), f(0xC), f(0x14)]));
                }
            }
        }
    }
    // instanced-geometry definitions: sbsp +0x38C datum -> structure_bsp_tag_resources control data
    let mut instance_defs = Vec::new();
    if let Some(sb) = c.tag_meta(sbsp_tag) {
        if let Some(e) = c.resource_by_id(d.u32_at(sb + OFF_SBSP_TAG_RESOURCES)).filter(|e| c.kind_is(e.kind, H3Cache::RES_BSP_TAG)) {
            let cd = c.control_data(e);
            let fx = c.resource_fixups(e);
            let root = e.root as usize;
            let n = cd.i32_at(root + OFF_TAGRES_INSTANCE_DEFS).max(0) as usize;
            // the block pointer is a CONTROL-data fixup (flag 0x20), unlike the raw-page blocks
            if let Some(&(ptr, flag)) = fx.get(&((root + OFF_TAGRES_INSTANCE_DEFS + 4) as u32)) {
                let ptr = ptr as usize;
                if n > 0 && n < 1_000_000 && flag & 0x20 != 0 && ptr + n * INSTANCE_DEF_ELEM <= cd.len() {
                    for i in 0..n {
                        let e0 = ptr + i * INSTANCE_DEF_ELEM;
                        instance_defs.push(InstanceDef {
                            mesh: cd.i16_at(e0 + INSTANCE_DEF_MESH),
                            compression: cd.i16_at(e0 + INSTANCE_DEF_COMPRESSION),
                            sphere: [cd.f32_at(e0 + 4), cd.f32_at(e0 + 8), cd.f32_at(e0 + 0xC), cd.f32_at(e0 + 0x10)],
                        });
                    }
                }
            }
        }
    }
    // instance placements: sbsp +0x1C0 records, resolved through their definition
    let mut instances = Vec::new();
    if let Some((n, o)) = c.tag_meta(sbsp_tag).and_then(|sb| c.block(sb + OFF_SBSP_INSTANCES)) {
        for i in 0..n {
            let e = o + i * INSTANCE_ELEM;
            let f = |k: usize| d.f32_at(e + k);
            let def = d.i16_at(e + 0x34);
            let (mesh, compression) = match usize::try_from(def).ok().and_then(|k| instance_defs.get(k)) {
                Some(dd) => (dd.mesh, dd.compression),
                None => (-1, -1),
            };
            instances.push(Instance {
                index: i,
                def,
                mesh,
                compression,
                scale: f(0),
                rot: [f(4), f(8), f(0xC), f(0x10), f(0x14), f(0x18), f(0x1C), f(0x20), f(0x24)],
                translation: [f(0x28), f(0x2C), f(0x30)],
                flags: d.u16_at(e + 0x36),
                lm_texcoord_block: d.i16_at(e + 0x38),
                world_sphere: [f(0x40), f(0x44), f(0x48), f(0x4C)],
                name: c.sid(d.u32_at(e + 0x54)),
            });
        }
    }
    // rotation convention check: the definition's local sphere, transformed, must land on the
    // record's world sphere (center within a radius-scaled tolerance).
    let mut rot_check = (0usize, 0usize, 0usize);
    for inst in instances.iter().filter(|i| !i.is_identity() && i.is_finite()) {
        let Some(dd) = usize::try_from(inst.def).ok().and_then(|k| instance_defs.get(k)) else { continue };
        let local = Vec3::new(dd.sphere[0], dd.sphere[1], dd.sphere[2]);
        let world = Vec3::new(inst.world_sphere[0], inst.world_sphere[1], inst.world_sphere[2]);
        if !local.is_finite() || !world.is_finite() || dd.sphere[3] <= 0.0 { continue; }
        // a sphere centred at the local origin says nothing about rotation
        if local.length() < 0.05 { continue; }
        rot_check.2 += 1;
        let tol = (dd.sphere[3] * inst.scale.abs()).max(0.1) * 0.1 + 0.05;
        if inst.matrix_rows().transform_point3(local).distance(world) <= tol { rot_check.0 += 1; }
        if inst.matrix_cols().transform_point3(local).distance(world) <= tol { rot_check.1 += 1; }
    }
    let rows = rot_check.0 >= rot_check.1;
    // geometry resource: the Lbsp-owned render-geometry resource
    let mut geometry = None;
    if let Some(lt) = lbsp_tag {
        if let Some(e) = c.resources.iter().find(|e| e.is_live() && e.owner == *b"Lbsp"
            && e.owner_tag == lt && c.kind_is(e.kind, H3Cache::RES_GEOMETRY)) {
            geometry = Some(parse_geometry_resource(c, e)?);
        }
    }
    Ok(H3Bsp { sbsp_tag, lbsp_tag, name, meshes, bounds, geometry, clusters, cluster_ids, cluster_bounds, materials, instance_defs, instances, rot_check, rows })
}

/// Decode one mesh's vertices (dequantised to world/local space via its compression bounds) and
/// its u16 triangle list.
pub fn decode_mesh(c: &H3Cache, bsp: &H3Bsp, mesh_idx: usize) -> Result<Option<DecodedMesh>> {
    decode_mesh_with_bounds(c, bsp, mesh_idx, None)
}

/// `decode_mesh` with an explicit compression-bounds entry (an instance definition whose
/// compression index differs from its mesh index); None = the mesh's own entry.
pub fn decode_mesh_with_bounds(c: &H3Cache, bsp: &H3Bsp, mesh_idx: usize, bounds: Option<&Bounds>) -> Result<Option<DecodedMesh>> {
    let m = bsp.meshes.get(mesh_idx).ok_or_else(|| anyhow!("mesh {mesh_idx} out of range"))?;
    let g = bsp.geometry.as_ref().ok_or_else(|| anyhow!("bsp has no geometry resource"))?;
    decode_mesh_generic(c, g, m, bounds)
}

/// Decode one mesh of ANY Gen3 geometry resource (a BSP's Lbsp resource or a `mode` render
/// model's, objects.rs): the same descriptor / page / vertex-format path. `bounds` overrides the
/// mesh's own compression entry (None = the mesh's own).
pub fn decode_mesh_generic(c: &H3Cache, g: &GeometryResource, m: &MeshDef, bounds: Option<&Bounds>) -> Result<Option<DecodedMesh>> {
    if !m.has_geometry() { return Ok(None); }
    let ib = g.ibs.get(m.ib as usize).ok_or_else(|| anyhow!("ib index {} out of range", m.ib))?;
    let Some(ioff) = ib.off else { return Ok(None) };
    let entry = &c.resources[g.entry];
    let (page, base) = c.read_resource(entry)?;
    let istart = base + ioff as usize;
    let ibytes = page.get(istart..istart + ib.size as usize)
        .ok_or_else(|| anyhow!("index bytes out of page"))?;
    // Raw u16 index buffer. Triangle strips (type 5) are converted to a triangle LIST per PART
    // below (once the vertex count is known), so the parts' index ranges stay valid into the
    // returned list; the clean whole-list here is only for sizing the vertex-buffer pick.
    let strip = m.index_type == 5;
    let raw: Vec<u32> = ibytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as u32).collect();
    let clean = if strip { strip_to_list(&raw) } else { raw.clone() };
    let max_index = clean.iter().copied().max().unwrap_or(0) as u32;

    // Pick the vertex buffer: the raw `vb[0]` reading, falling back to the position-buffer index
    // when the raw buffer is too small for the mesh's indices (interleaved short-stride streams).
    let pick = |vb: &VbInfo| vb.off.is_some() && vb.stride >= 36 && vb.count > max_index;
    let vb = g.vbs.get(m.vb[0] as usize).filter(|v| pick(v))
        .or_else(|| g.pos_vbs.get(m.vb[0] as usize).and_then(|&i| g.vbs.get(i)).filter(|v| pick(v)));
    let Some(vb) = vb else { return Ok(None) };
    let voff = vb.off.unwrap();
    let vstart = base + voff as usize;
    let vbytes = page.get(vstart..vstart + vb.count as usize * vb.stride as usize)
        .ok_or_else(|| anyhow!("vertex bytes out of page"))?;

    let b = bounds.unwrap_or(&m.bounds);
    let stride = vb.stride as usize;
    // Two BSP vertex formats. COMPRESSED (instanced geometry, descriptor type 2, stride 36): a
    // non-degenerate compression-bounds entry; pos int16x3 @0 + uv int16x2 @8 dequantised to the
    // bounds, normal snorm16x3 @0x14. UNCOMPRESSED World (cluster geometry, descriptor type 1,
    // stride 44): no bounds; pos f32x3 @0 (absolute), uv f32x2 @0xC, normal snorm16x3 @0x14... at
    // @0x14 (verified: unit 505/505 @0x14). The bounds being degenerate distinguishes them.
    let compressed = (b.flags & 1) != 0 && b.pos_max != b.pos_min;
    let mut verts = Vec::with_capacity(vb.count as usize);
    for k in 0..vb.count as usize {
        let v = &vbytes[k * stride..k * stride + stride];
        let (pos, uv, normal) = if compressed {
            (
                [dq(v.i16_at(0), b.pos_min[0], b.pos_max[0]),
                 dq(v.i16_at(2), b.pos_min[1], b.pos_max[1]),
                 dq(v.i16_at(4), b.pos_min[2], b.pos_max[2])],
                [dq(v.i16_at(8), b.uv_min[0], b.uv_max[0]),
                 dq(v.i16_at(10), b.uv_min[1], b.uv_max[1])],
                // compressed layout (hexdump-verified on guardian): pos int16x4 @0, uv int16x2
                // @8, NORMAL int16x4 @0xC, tangent int16x4 @0x14, binormal int16x4 @0x1C = 36 B.
                // (@0x14 is the tangent - reading the normal there lit every instance wrong.)
                [sn16(v.i16_at(0xC)), sn16(v.i16_at(0xE)), sn16(v.i16_at(0x10))],
            )
        } else {
            // uncompressed World: pos f32x3 @0, uv f32x2 @0xC, normal snorm16x3 @0x14
            (
                [v.f32_at(0), v.f32_at(4), v.f32_at(8)],
                [v.f32_at(0xC), v.f32_at(0x10)],
                [sn16(v.i16_at(0x14)), sn16(v.i16_at(0x16)), sn16(v.i16_at(0x18))],
            )
        };
        verts.push(H3Vertex { pos, normal, uv });
    }
    let nv = verts.len();
    // Final index list + per-part ranges. LIST meshes keep the raw offsets; STRIP meshes convert
    // each part's sub-strip to a list and rewrite its range (mirrors the Halo 4 objects path), so
    // a caller can draw one part alone by slicing `indices[part.index_start..][..index_count]`.
    // Whole-mesh callers get the same triangles (parts concatenated in order).
    let (indices, parts) = if strip {
        let mut idx = Vec::with_capacity(raw.len() * 3);
        let mut parts = Vec::with_capacity(m.parts.len());
        for p in &m.parts {
            let start = idx.len() as u32;
            let (s, n) = (p.index_start as usize, p.index_count as usize);
            if s + n <= raw.len() && n >= 3 {
                for j in 0..n - 2 {
                    let (mut a, mut b, cc) = (raw[s + j], raw[s + j + 1], raw[s + j + 2]);
                    if a == b || b == cc || a == cc { continue; }
                    if a as usize >= nv || b as usize >= nv || cc as usize >= nv { continue; }
                    if j & 1 == 1 { std::mem::swap(&mut a, &mut b); }
                    idx.extend_from_slice(&[a, b, cc]);
                }
            }
            parts.push(Part { material: p.material, index_start: start, index_count: idx.len() as u32 - start });
        }
        (idx, parts)
    } else {
        (clean, m.parts.clone())
    };
    Ok(Some(DecodedMesh { verts, indices, parts }))
}

/// The raw bytes of one vertex buffer of a BSP's geometry resource (`count * stride` bytes from
/// the pageable page), for the side streams a mesh carries next to its positions: the kind-4
/// stride-4 lightmap-UV stream (u16x2), the kind-6 stride-20 per-vertex lighting stream.
pub fn read_vb_raw(c: &H3Cache, bsp: &H3Bsp, vb_index: usize) -> Result<(VbInfo, Vec<u8>)> {
    let g = bsp.geometry.as_ref().ok_or_else(|| anyhow!("bsp has no geometry resource"))?;
    read_vb_raw_generic(c, g, vb_index)
}

/// `read_vb_raw` over any geometry resource (a render model's side streams, objects.rs).
pub fn read_vb_raw_generic(c: &H3Cache, g: &GeometryResource, vb_index: usize) -> Result<(VbInfo, Vec<u8>)> {
    let vb = *g.vbs.get(vb_index).ok_or_else(|| anyhow!("vb {vb_index} out of range ({})", g.vbs.len()))?;
    let off = vb.off.ok_or_else(|| anyhow!("vb {vb_index} has no data offset"))?;
    if vb.stride <= 0 { bail!("vb {vb_index} stride {}", vb.stride); }
    let (page, base) = c.read_resource(&c.resources[g.entry])?;
    let s = base + off as usize;
    let n = vb.count as usize * vb.stride as usize;
    let bytes = page.get(s..s + n).ok_or_else(|| anyhow!("vb {vb_index} bytes out of page"))?.to_vec();
    Ok((vb, bytes))
}

/// Triangle-strip -> triangle-list (0xFFFF = restart), dropping degenerate triangles.
fn strip_to_list(strip: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut a = 0u32;
    let mut b = 0u32;
    let mut n = 0usize;
    for &i in strip {
        if i == 0xFFFF { n = 0; continue; }
        if n >= 2 {
            let (x, y, z) = if n % 2 == 0 { (a, b, i) } else { (b, a, i) };
            if x != y && y != z && x != z { out.extend_from_slice(&[x, y, z]); }
        }
        a = b; b = i; n += 1;
    }
    out
}

/// Scene-space AABB of every cluster mesh (the identity-placed BSP geometry).
pub fn scene_bounds(_c: &H3Cache, bsp: &H3Bsp) -> Option<(Vec3, Vec3)> {
    let mut mn = Vec3::splat(f32::MAX);
    let mut mx = Vec3::splat(f32::MIN);
    let mut any = false;
    for &(a, b) in &bsp.cluster_bounds {
        mn = mn.min(Vec3::from(a));
        mx = mx.max(Vec3::from(b));
        any = true;
    }
    any.then_some((mn, mx))
}


#[cfg(test)]
fn read_ib_raw_len(_c: &H3Cache, _b: &H3Bsp, _i: usize) -> Option<usize> { None }

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cache::maps_dir;

    fn open(name: &str) -> Option<H3Cache> {
        let p = maps_dir()?.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H3Cache::open(&p).expect("open"))
    }

    #[test]
    fn guardian_geometry_decodes() {
        let Some(c) = open("guardian.map") else { return };
        let sbsps = c.find_tags(b"sbsp");
        assert_eq!(sbsps.len(), 1);
        let bsp = load_bsp(&c, sbsps[0]).unwrap();
        let g = bsp.geometry.as_ref().expect("geometry resource");
        assert_eq!((g.vbs.len(), g.ibs.len()), (727, 230));
        assert_eq!(bsp.meshes.len(), 238);
        assert!(!bsp.clusters.is_empty());
        assert!(!bsp.materials.is_empty());
        let resolved = bsp.materials.iter().filter(|m| m.render_tag.is_some()).count();
        assert!(resolved * 100 / bsp.materials.len() >= 90, "materials resolved {resolved}/{}", bsp.materials.len());
        // BSP materials reference render-method tags directly (no `mat ` indirection); rmsh dominates
        assert!(bsp.materials.iter().any(|m| &m.class == b"rmsh"));
        // every mesh with geometry decodes, indices stay inside the vertex buffer, positions
        // land inside the compression bounds, normals are unit
        let mut decoded = 0;
        for i in 0..bsp.meshes.len() {
            if let Some(dm) = decode_mesh(&c, &bsp, i).unwrap() {
                decoded += 1;
                assert!(dm.indices.iter().all(|&x| (x as usize) < dm.verts.len()), "mesh {i} index range");
                let b = &bsp.meshes[i].bounds;
                let compressed = (b.flags & 1) != 0 && b.pos_max != b.pos_min;
                for v in &dm.verts {
                    // compressed meshes: positions must land inside the compression bounds
                    if compressed {
                        for k in 0..3 {
                            assert!(v.pos[k] >= b.pos_min[k] - 1e-2 && v.pos[k] <= b.pos_max[k] + 1e-2, "mesh {i} pos");
                        }
                    }
                    let n = Vec3::from(v.normal).length();
                    assert!((n - 1.0).abs() < 0.05, "mesh {i} normal {n}");
                }
            }
        }
        assert!(decoded > 200, "decoded {decoded} meshes");
    }

    /// DIAG (print-only): for each instance definition, the decoded mesh's extent about the
    /// definition sphere's centre vs that sphere's radius, with its assigned bounds and with the
    /// best-fitting entry of the whole bounds table. `cargo test ... diag_instance_bounds -- --nocapture`
    #[test]
    #[ignore] // print-only forensics (decodes every mesh several times); run on demand with --ignored
    fn diag_instance_bounds() {
        let Some(c) = open("guardian.map") else { return };
        let bsp = load_bsp(&c, c.find_tags(b"sbsp")[0]).unwrap();
        eprintln!("bounds table {} entries, meshes {}, defs {}", bsp.bounds.len(), bsp.meshes.len(), bsp.instance_defs.len());
        let extent = |dm: &DecodedMesh, sc: [f32; 3]| -> (f32, [f32; 3]) {
            let (mut mn, mut mx, mut r) = ([f32::MAX; 3], [f32::MIN; 3], 0f32);
            for v in &dm.verts { for k in 0..3 { mn[k] = mn[k].min(v.pos[k]); mx[k] = mx[k].max(v.pos[k]); }
                let d = ((v.pos[0]-sc[0]).powi(2) + (v.pos[1]-sc[1]).powi(2) + (v.pos[2]-sc[2]).powi(2)).sqrt(); r = r.max(d); }
            (r, [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]])
        };
        // triangle sanity: median / p90 edge length as a fraction of the mesh's own extent. A
        // healthy mesh is << 1; mis-indexed triangles span the whole mesh (ratios near 1).
        let tri_stats = |dm: &DecodedMesh| -> (f32, f32, usize, usize) {
            let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
            for v in &dm.verts { for k in 0..3 { mn[k] = mn[k].min(v.pos[k]); mx[k] = mx[k].max(v.pos[k]); } }
            let ext = ((mx[0]-mn[0]).powi(2) + (mx[1]-mn[1]).powi(2) + (mx[2]-mn[2]).powi(2)).sqrt().max(1e-6);
            let mut e: Vec<f32> = Vec::new();
            for t in dm.indices.chunks(3) { if t.len() < 3 { break }
                for (a, b) in [(0, 1), (1, 2), (2, 0)] { let (p, q) = (dm.verts[t[a] as usize].pos, dm.verts[t[b] as usize].pos);
                    e.push(((p[0]-q[0]).powi(2) + (p[1]-q[1]).powi(2) + (p[2]-q[2]).powi(2)).sqrt() / ext); } }
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = e.len().max(1);
            (e[n / 2], e[n * 9 / 10], dm.indices.len() / 3, dm.parts.len())
        };
        for (label, mi) in [("instance mesh 0", bsp.instance_defs[0].mesh as usize), ("instance mesh 2", bsp.instance_defs[2].mesh as usize), ("instance mesh 120", bsp.instance_defs[120].mesh as usize), ("cluster mesh", bsp.clusters[0] as usize)] {
            if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi, None) {
                let (med, p90, tris, parts) = tri_stats(&dm);
                let md = &bsp.meshes[mi];
                eprintln!("TRI {label:16} mesh {mi:3}: tris {tris:6} parts {parts:2} edge/extent median {med:.3} p90 {p90:.3} | meshdef {md:?}");
            }
        }
        // matrix sanity: the rotation rows must be unit + mutually orthogonal (det +-1); the
        // definition sphere radius x scale must equal the record's world sphere radius.
        for inst in bsp.instances.iter().filter(|i| i.index < 6 || i.index % 80 == 0) {
            let r = &inst.rot;
            let n = |a: usize| (r[a*3].powi(2) + r[a*3+1].powi(2) + r[a*3+2].powi(2)).sqrt();
            let d = |a: usize, b: usize| r[a*3]*r[b*3] + r[a*3+1]*r[b*3+1] + r[a*3+2]*r[b*3+2];
            let det = r[0]*(r[4]*r[8]-r[5]*r[7]) - r[1]*(r[3]*r[8]-r[5]*r[6]) + r[2]*(r[3]*r[7]-r[4]*r[6]);
            let def = bsp.instance_defs.get(inst.def as usize);
            let m = bsp.instance_matrix(inst);
            let ms = m.to_scale_rotation_translation();
            eprintln!("MAT inst {:4} '{}' scale {:.3} rows |{:.3} {:.3} {:.3}| dots {:.3} {:.3} {:.3} det {:.3} | def sphere r {:.2} x scale = {:.2} vs world r {:.2} | mat4 scale {:?} rows_conv={}",
                inst.index, inst.name, inst.scale, n(0), n(1), n(2), d(0,1), d(0,2), d(1,2), det,
                def.map_or(0.0, |d| d.sphere[3]), def.map_or(0.0, |d| d.sphere[3]) * inst.scale, inst.world_sphere[3], ms.0.to_array().map(|v| (v*1000.0).round()/1000.0), bsp.rows);
        }
        // index sanity: face normal (from positions, per part range) vs the stored vertex
        // normals. Real geometry agrees (~0.9+); mis-indexed triangles are random (~0.5).
        let nrm_agree = |dm: &DecodedMesh| -> (f32, usize) {
            let (mut acc, mut n) = (0f32, 0usize);
            for part in &dm.parts {
                let (s0, e0) = (part.index_start as usize, (part.index_start as usize + part.index_count as usize).min(dm.indices.len()));
                for t in dm.indices[s0..e0].chunks(3) { if t.len() < 3 { break }
                    let p = |i: usize| glam::Vec3::from(dm.verts[t[i] as usize].pos);
                    let fnm = (p(1) - p(0)).cross(p(2) - p(0));
                    if fnm.length() < 1e-9 { continue }
                    let vn = (glam::Vec3::from(dm.verts[t[0] as usize].normal) + glam::Vec3::from(dm.verts[t[1] as usize].normal) + glam::Vec3::from(dm.verts[t[2] as usize].normal)).normalize_or_zero();
                    acc += fnm.normalize().dot(vn).abs(); n += 1;
                } }
            (if n > 0 { acc / n as f32 } else { 0.0 }, n)
        };
        for (label, mi) in [("inst mesh 0 tree", bsp.instance_defs[0].mesh as usize), ("inst mesh 2 ring", bsp.instance_defs[2].mesh as usize), ("inst mesh 120", bsp.instance_defs[120].mesh as usize), ("inst mesh 160", bsp.instance_defs[160].mesh as usize), ("cluster 0", bsp.clusters[0] as usize), ("cluster 5", bsp.clusters[5.min(bsp.clusters.len()-1)] as usize)] {
            if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi, None) {
                let (a, n) = nrm_agree(&dm);
                // the same vertices with the raw index stream re-read as a STRIP, and with the
                // list shifted by 1 / 2 entries (an alignment slip)
                let as_strip = { let mut d = dm.clone(); d.indices = strip_to_list(&dm.indices); d.parts = vec![Part { material: 0, index_start: 0, index_count: d.indices.len() as u32 }]; nrm_agree(&d).0 };
                let shifted = |k: usize| { let mut d = dm.clone(); d.indices = dm.indices[k..].to_vec(); d.parts = vec![Part { material: 0, index_start: 0, index_count: (d.indices.len() / 3 * 3) as u32 }]; nrm_agree(&d).0 };
                eprintln!("NRM {label:18} mesh {mi:3} type {}: list agreement {a:.3} ({n} tris) | as strip {as_strip:.3} | list+1 {:.3} list+2 {:.3} | first idx {:?}", bsp.meshes[mi].index_type, shifted(1), shifted(2), &dm.indices[..12.min(dm.indices.len())]);
            }
        }
        // VB descriptors + vertex type, instance vs cluster; and the agreement if the index
        // buffer were a strip instead of a list.
        for (label, mi) in [("inst mesh 0", bsp.instance_defs[0].mesh as usize), ("inst mesh 2", bsp.instance_defs[2].mesh as usize), ("inst mesh 120", bsp.instance_defs[120].mesh as usize), ("cluster 0", bsp.clusters[0] as usize), ("cluster 5", bsp.clusters[5.min(bsp.clusters.len()-1)] as usize)] {
            let md = &bsp.meshes[mi];
            let vbs: Vec<String> = md.vb.iter().filter(|&&v| v >= 0).map(|&v| match read_vb_raw(&c, &bsp, v as usize) { Ok((info, raw)) => format!("vb{v}:{info:?} bytes {}", raw.len()), Err(e) => format!("vb{v}: ERR {e}") }).collect();
            let ib = match read_ib_raw_len(&c, &bsp, md.ib as usize) { Some(n) => n.to_string(), None => "?".into() };
            eprintln!("VB  {label:14} mesh {mi:3} vtype {} itype {} ib {} (raw idx {ib}) | {}", md.vertex_type, md.index_type, md.ib, vbs.join(" | "));
        }
        // normal-free index sanity: fraction of triangle edges shared by exactly two triangles
        // (a closed/manifold-ish mesh ~0.8-1.0; scrambled indices ~0.0-0.2), max index vs VB count,
        // and a raw hexdump of the first vertices of the compressed stream.
        for (label, mi) in [("inst mesh 0 tree", bsp.instance_defs[0].mesh as usize), ("inst mesh 2 ring", bsp.instance_defs[2].mesh as usize), ("cluster 5", bsp.clusters[5.min(bsp.clusters.len()-1)] as usize)] {
            if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi, None) {
                let mut edges: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
                for t in dm.indices.chunks(3) { if t.len() < 3 { break }
                    for (a, b) in [(0, 1), (1, 2), (2, 0)] { let (x, y) = (t[a].min(t[b]), t[a].max(t[b])); *edges.entry((x, y)).or_default() += 1; } }
                let shared2 = edges.values().filter(|&&n| n == 2).count();
                let maxi = dm.indices.iter().copied().max().unwrap_or(0);
                eprintln!("EDGE {label:16} mesh {mi:3}: edges {} shared-by-2 {:.3} max index {maxi} verts {}", edges.len(), shared2 as f32 / edges.len().max(1) as f32, dm.verts.len());
            }
            let md = &bsp.meshes[mi];
            if let Ok((info, raw)) = read_vb_raw(&c, &bsp, md.vb[0] as usize) {
                let st = info.stride as usize;
                for k in 0..3 { let v = &raw[k*st..k*st+st];
                    let i16s: Vec<i16> = (0..st/2).map(|j| i16::from_le_bytes([v[2*j], v[2*j+1]])).collect();
                    eprintln!("HEX  {label:16} v{k}: i16 {:?}", i16s); }
            }
        }
        // height layout: decoded cluster-vertex z (all clusters, then non-vista ones) vs the
        // instance records' world-sphere z and the instances' transformed local centres.
        let pct = |v: &mut Vec<f32>, q: f32| -> f32 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); if v.is_empty() { f32::NAN } else { v[((v.len() - 1) as f32 * q) as usize] } };
        let (mut cz, mut czs) = (Vec::new(), Vec::new());
        for (k, &mi) in bsp.clusters.iter().enumerate() {
            if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi as usize, None) {
                let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
                for v in &dm.verts { for a in 0..3 { mn[a] = mn[a].min(v.pos[a]); mx[a] = mx[a].max(v.pos[a]); } }
                let diag = ((mx[0]-mn[0]).powi(2) + (mx[1]-mn[1]).powi(2) + (mx[2]-mn[2]).powi(2)).sqrt();
                for v in &dm.verts { cz.push(v.pos[2]); if diag < 200.0 { czs.push(v.pos[2]); } }
                if k < 8 { eprintln!("CLUSTER {k:2} mesh {mi:3}: verts {} diag {diag:7.1} z [{:.1} .. {:.1}] xy [{:.0},{:.0}]..[{:.0},{:.0}]", dm.verts.len(), mn[2], mx[2], mn[0], mn[1], mx[0], mx[1]); }
            }
        }
        let mut iz: Vec<f32> = bsp.instances.iter().map(|i| i.world_sphere[2]).collect();
        let mut tz: Vec<f32> = bsp.instances.iter().map(|i| i.translation[2]).collect();
        eprintln!("HEIGHT clusters(all)  z p05 {:.1} p50 {:.1} p95 {:.1}", pct(&mut cz, 0.05), pct(&mut cz, 0.5), pct(&mut cz, 0.95));
        eprintln!("HEIGHT clusters(<200) z p05 {:.1} p50 {:.1} p95 {:.1}", pct(&mut czs, 0.05), pct(&mut czs, 0.5), pct(&mut czs, 0.95));
        eprintln!("HEIGHT instances      sphere z p05 {:.1} p50 {:.1} p95 {:.1} | translation z p05 {:.1} p50 {:.1} p95 {:.1}", pct(&mut iz, 0.05), pct(&mut iz, 0.5), pct(&mut iz, 0.95), pct(&mut tz, 0.05), pct(&mut tz, 0.5), pct(&mut tz, 0.95));
        // enclosure test: the record's WORLD sphere must contain the transformed mesh. Count, per
        // matrix convention, instances with > 2% of vertices outside (radius x 1.05 + 0.05).
        {
            let mut cache: std::collections::HashMap<usize, DecodedMesh> = std::collections::HashMap::new();
            let conv: [(&str, Box<dyn Fn(&Instance) -> Mat4>); 4] = [
                ("rows", Box::new(|i: &Instance| i.matrix_rows())),
                ("cols", Box::new(|i: &Instance| i.matrix_cols())),
                ("rows-mirrorL", Box::new(|i: &Instance| { let mut j = i.clone(); for k in 3..6 { j.rot[k] = -j.rot[k]; } j.matrix_rows() })),
                ("rows-noscale", Box::new(|i: &Instance| { let mut j = i.clone(); j.scale = 1.0; j.matrix_rows() })),
            ];
            let mut bad = [0usize; 4]; let mut tested = 0usize; let mut worst: Vec<(f32, usize, &str)> = Vec::new();
            for inst in &bsp.instances {
                if inst.mesh < 0 || inst.world_sphere[3] <= 0.0 { continue; }
                let mi = inst.mesh as usize;
                if !cache.contains_key(&mi) { if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi, None) { cache.insert(mi, dm); } else { continue; } }
                let dm = &cache[&mi]; if dm.verts.is_empty() { continue; }
                tested += 1;
                let wc = glam::Vec3::new(inst.world_sphere[0], inst.world_sphere[1], inst.world_sphere[2]);
                let lim = inst.world_sphere[3] * 1.05 + 0.05;
                for (ci, (name, f)) in conv.iter().enumerate() {
                    let m = f(inst);
                    let out = dm.verts.iter().filter(|v| m.transform_point3(glam::Vec3::from(v.pos)).distance(wc) > lim).count();
                    let frac = out as f32 / dm.verts.len() as f32;
                    if frac > 0.02 { bad[ci] += 1; }
                    if ci == 0 && frac > 0.02 { worst.push((frac, inst.index, inst.name.as_str())); }
                }
            }
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("ENCLOSE tested {tested}: instances with >2% verts outside world sphere -> rows {} cols {} rows-mirrorL {} rows-noscale {}", bad[0], bad[1], bad[2], bad[3]);
            for w in worst.iter().take(8) { eprintln!("  rows worst: {:.0}% outside  inst {} '{}'", w.0 * 100.0, w.1, w.2); }
        }
        // UV sanity: decoded uv range per mesh (textures tile in ~[0,1]; a dequant slip gives
        // huge ranges = smeared/streaked texturing) + the bounds entry used.
        for (label, mi) in [("inst mesh 0 tree", bsp.instance_defs[0].mesh as usize), ("inst mesh 2 ring", bsp.instance_defs[2].mesh as usize), ("inst mesh 66 ring11", 66usize), ("cluster 5", bsp.clusters[5.min(bsp.clusters.len()-1)] as usize)] {
            if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi, None) {
                let (mut mn, mut mx) = ([f32::MAX; 2], [f32::MIN; 2]);
                for v in &dm.verts { for a in 0..2 { mn[a] = mn[a].min(v.uv[a]); mx[a] = mx[a].max(v.uv[a]); } }
                let b = &bsp.meshes[mi].bounds;
                eprintln!("UV  {label:18} mesh {mi:3}: uv [{:.2},{:.2}]..[{:.2},{:.2}] | bounds uv_min {:?} uv_max {:?} flags {}", mn[0], mn[1], mx[0], mx[1], b.uv_min, b.uv_max, b.flags);
            }
        }
        // mesh IDENTITY: the definition's local sphere vs the decoded mesh's own bounding sphere
        // (AABB centre + max radius). A right mapping has centres ~coincident and radius ~1; a
        // permuted mapping puts the centre far off. Then the best-matching mesh over the table.
        {
            let mut spheres: Vec<Option<([f32; 3], f32)>> = Vec::new();
            for mi in 0..bsp.meshes.len() {
                spheres.push(decode_mesh_with_bounds(&c, &bsp, mi, None).ok().flatten().filter(|d| !d.verts.is_empty()).map(|dm| {
                    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
                    for v in &dm.verts { for a in 0..3 { mn[a] = mn[a].min(v.pos[a]); mx[a] = mx[a].max(v.pos[a]); } }
                    let cc = [(mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5];
                    let r = dm.verts.iter().map(|v| ((v.pos[0]-cc[0]).powi(2)+(v.pos[1]-cc[1]).powi(2)+(v.pos[2]-cc[2]).powi(2)).sqrt()).fold(0f32, f32::max);
                    (cc, r) }));
            }
            let mut offs: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
            for (k, d) in bsp.instance_defs.iter().enumerate() {
                let dc = [d.sphere[0], d.sphere[1], d.sphere[2]]; let dr = d.sphere[3].max(1e-3);
                let score = |sp: &([f32; 3], f32)| ((sp.0[0]-dc[0]).powi(2)+(sp.0[1]-dc[1]).powi(2)+(sp.0[2]-dc[2]).powi(2)).sqrt() / dr + (sp.1 / dr - 1.0).abs();
                let own = spheres.get(d.mesh as usize).and_then(|s| s.as_ref()).map(score);
                let best = spheres.iter().enumerate().filter_map(|(i, s)| s.as_ref().map(|s| (score(s), i))).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                if let Some((bs, bi)) = best { *offs.entry(bi as i64 - d.mesh as i64).or_default() += 1;
                    if k < 14 || k % 30 == 0 { eprintln!("ID def {k:3} mesh {:3}: own score {:.2} (centre off {:.2}r) | best mesh {bi:3} score {bs:.2}", d.mesh, own.unwrap_or(f32::NAN), own.map_or(f32::NAN, |o| o), ); } }
            }
            let mut ov: Vec<_> = offs.into_iter().collect(); ov.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("ID best-mesh minus def-mesh offsets (count): {:?}", &ov[..ov.len().min(6)]);
        }
        // TABLE PROBE: does the sbsp TAG BODY carry its own instanced-geometry DEFINITIONS block
        // (Halo 3 layout) whose order differs from the resource table the records are resolved
        // against today? Scan the sbsp meta for blocks whose count is the definition count.
        {
            let sbsp = c.find_tags(b"sbsp")[0];
            let sm = c.tag_meta(sbsp).unwrap();
            let ndef = bsp.instance_defs.len();
            let mut hits: Vec<(usize, usize)> = Vec::new();
            for off in (0x40..0x600).step_by(4) {
                if let Some((n, ptr)) = c.block(sm + off) { if n == ndef || n == bsp.instances.len() { hits.push((off, n)); eprintln!("PROBE sbsp +{off:#x}: block count {n} ptr {ptr:#x}"); } }
            }
            // for a definitions-sized block, dump the first two elements as floats/i16 under a few stride guesses
            for &(off, n) in hits.iter().filter(|h| h.1 == ndef) {
                let (_, ptr) = c.block(sm + off).unwrap();
                let d = c.data();
                for stride in [0x28usize, 0x30, 0x38, 0x3C, 0x40, 0x48, 0x50, 0x58] {
                    let e0: Vec<String> = (0..stride/4).map(|k| { let f = d.f32_at(ptr + k*4); let i = d.i16_at(ptr + k*4); if f.is_finite() && f.abs() < 1e5 && f.abs() > 1e-4 { format!("{f:.2}") } else { format!("i{i}") } }).collect();
                    eprintln!("PROBE  +{off:#x} stride {stride:#x} e0: {}", e0.join(" "));
                }
                // the resource table's first def sphere for comparison
                eprintln!("PROBE  resource def0 sphere {:?} mesh {} | def1 {:?} mesh {}", bsp.instance_defs[0].sphere, bsp.instance_defs[0].mesh, bsp.instance_defs[1].sphere, bsp.instance_defs[1].mesh);
            }
            let mut fl: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
            for i in &bsp.instances { *fl.entry(i.flags).or_default() += 1; }
            eprintln!("PROBE instance flags histogram {:?}", fl);
        }
        // TIGHT sphere test: transformed-mesh radius / record world radius, and transformed-mesh
        // centre offset / record radius. Right piece: ratio ~0.7-1.05, offset < 0.3. Wrong
        // (smaller) piece at the right transform: ratio well below 0.7.
        {
            let mut cache: std::collections::HashMap<usize, DecodedMesh> = std::collections::HashMap::new();
            let (mut lo, mut mid, mut hi, mut off_bad) = (0usize, 0usize, 0usize, 0usize);
            for inst in &bsp.instances {
                if inst.mesh < 0 || inst.world_sphere[3] <= 0.0 { continue; }
                let mi = inst.mesh as usize;
                if !cache.contains_key(&mi) { if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, mi, None) { cache.insert(mi, dm); } else { continue; } }
                let dm = &cache[&mi]; if dm.verts.is_empty() { continue; }
                let m = bsp.instance_matrix(inst);
                let pts: Vec<glam::Vec3> = dm.verts.iter().map(|v| m.transform_point3(glam::Vec3::from(v.pos))).collect();
                let (mut mn, mut mx) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
                for p in &pts { mn = mn.min(*p); mx = mx.max(*p); }
                let cc = (mn + mx) * 0.5;
                let r = pts.iter().map(|p| p.distance(cc)).fold(0f32, f32::max);
                let wc = glam::Vec3::new(inst.world_sphere[0], inst.world_sphere[1], inst.world_sphere[2]);
                let ratio = r / inst.world_sphere[3]; let off = cc.distance(wc) / inst.world_sphere[3];
                if ratio < 0.7 { lo += 1 } else if ratio <= 1.05 { mid += 1 } else { hi += 1 }
                if off > 0.3 { off_bad += 1 }
            }
            eprintln!("TIGHT radius-ratio <0.7: {lo}  0.7..1.05: {mid}  >1.05: {hi} | centre offset >0.3r: {off_bad}");
        }
        // DEQUANT hypotheses: decode each instance mesh's raw int16 positions under several
        // mappings and score the mesh centre / radius against the definition's own sphere
        // (median over all 212 definitions). The right mapping has centre offset ~0 and ratio ~1.
        {
            let maps: [(&str, fn(i16, f32, f32) -> f32); 4] = [
                ("signed mid+sn*half [current]", |v, mn, mx| mn + ((v as f32 / 32767.0).max(-1.0) * 0.5 + 0.5) * (mx - mn)),
                ("unsigned u16/65535",          |v, mn, mx| mn + ((v as u16) as f32 / 65535.0) * (mx - mn)),
                ("sn16 * (max-min) (no min)",   |v, mn, mx| (v as f32 / 32767.0) * (mx - mn)),
                ("sn16 * max",                  |v, _mn, mx| (v as f32 / 32767.0) * mx),
            ];
            let mut scores: Vec<(Vec<f32>, Vec<f32>)> = (0..maps.len()).map(|_| (Vec::new(), Vec::new())).collect();
            for d in &bsp.instance_defs {
                let md = &bsp.meshes[d.mesh as usize];
                let Ok((info, raw)) = read_vb_raw(&c, &bsp, md.vb[0] as usize) else { continue };
                if info.stride != 36 || !(md.bounds.flags & 1 != 0) { continue; }
                let b = &md.bounds; let st = info.stride as usize; let n = info.count as usize;
                let dc = glam::Vec3::new(d.sphere[0], d.sphere[1], d.sphere[2]); let dr = d.sphere[3].max(1e-3);
                for (mi, (_, f)) in maps.iter().enumerate() {
                    let pts: Vec<glam::Vec3> = (0..n).map(|k| { let v = &raw[k*st..k*st+st];
                        glam::Vec3::new(f(v.i16_at(0), b.pos_min[0], b.pos_max[0]), f(v.i16_at(2), b.pos_min[1], b.pos_max[1]), f(v.i16_at(4), b.pos_min[2], b.pos_max[2])) }).collect();
                    let (mut mn, mut mx) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
                    for p in &pts { mn = mn.min(*p); mx = mx.max(*p); }
                    let cc = (mn + mx) * 0.5; let r = pts.iter().map(|p| p.distance(cc)).fold(0f32, f32::max);
                    scores[mi].0.push(cc.distance(dc) / dr); scores[mi].1.push(r / dr);
                }
            }
            let med = |v: &mut Vec<f32>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); if v.is_empty() { f32::NAN } else { v[v.len() / 2] } };
            for (mi, (name, _)) in maps.iter().enumerate() {
                let (mut o, mut r) = (scores[mi].0.clone(), scores[mi].1.clone());
                let p90 = { let mut t = o.clone(); t.sort_by(|a, b| a.partial_cmp(b).unwrap()); if t.is_empty() { f32::NAN } else { t[t.len() * 9 / 10] } };
                eprintln!("DEQ {name:30}: centre offset/r median {:.3} p90 {p90:.3} | radius ratio median {:.3}  (n={})", med(&mut o), med(&mut r), scores[mi].0.len());
            }
        }
        for (k, d) in bsp.instance_defs.iter().enumerate().filter(|(k, _)| *k < 12 || *k % 40 == 0) {
            let sc = [d.sphere[0], d.sphere[1], d.sphere[2]];
            let own = decode_mesh_with_bounds(&c, &bsp, d.mesh as usize, None).ok().flatten();
            let Some(own) = own else { eprintln!("def {k} mesh {} : no decode", d.mesh); continue };
            let (r_own, ext_own) = extent(&own, sc);
            // search every bounds entry for the one whose decoded radius best matches the sphere
            let mut best = (f32::MAX, usize::MAX, 0f32);
            for (bi, b) in bsp.bounds.iter().enumerate() {
                if let Ok(Some(dm)) = decode_mesh_with_bounds(&c, &bsp, d.mesh as usize, Some(b)) {
                    let (r, _) = extent(&dm, sc);
                    let err = (r / d.sphere[3].max(1e-3) - 1.0).abs();
                    if err < best.0 { best = (err, bi, r); }
                }
            }
            eprintln!("def {k:3} mesh {:3} comp {:3} sphere r {:6.2} | own-bounds r {:8.2} ext {:?} | best bounds #{} r {:6.2} err {:.3}",
                d.mesh, d.compression, d.sphere[3], r_own, ext_own.map(|e| (e * 10.0).round() / 10.0), best.1, best.2, best.0);
        }
    }

    /// Guardian's instanced geometry: 479 placements over 212 definitions, every definition a
    /// distinct Lbsp mesh below the cluster range, and the row convention proven by the
    /// definition-sphere -> record-sphere check on every non-identity instance.
    #[test]
    fn guardian_instances_decode() {
        let Some(c) = open("guardian.map") else { return };
        let bsp = load_bsp(&c, c.find_tags(b"sbsp")[0]).unwrap();
        assert_eq!((bsp.instances.len(), bsp.instance_defs.len()), (479, 212));
        let mut meshes: Vec<i16> = bsp.instance_defs.iter().map(|d| d.mesh).collect();
        meshes.sort();
        meshes.dedup();
        assert_eq!(meshes.len(), 212, "definitions map to distinct meshes");
        assert!(bsp.instance_defs.iter().all(|d| d.mesh >= 0 && d.compression == d.mesh));
        assert!(bsp.instances.iter().all(|i| i.def >= 0 && i.mesh >= 0 && i.is_finite()));
        // cluster meshes and instance meshes are disjoint ranges of the Lbsp mesh table
        assert!(bsp.clusters.iter().all(|&m| !meshes.contains(&m)));
        assert!(bsp.rows, "rot_check {:?}", bsp.rot_check);
        assert_eq!(bsp.rot_check.0, bsp.rot_check.2, "every tested instance fits the row convention");
        assert!(bsp.rot_check.1 * 2 < bsp.rot_check.2, "the column convention is not a fit");
        // a scaled, rotated instance decodes into local bounds and lands on its record's sphere
        let inst = bsp.instances.iter().find(|i| (i.scale - 1.0).abs() > 0.01).unwrap();
        let dm = decode_mesh_with_bounds(&c, &bsp, inst.mesh as usize, bsp.instance_bounds(inst)).unwrap().unwrap();
        let m = bsp.instance_matrix(inst);
        let ws = Vec3::new(inst.world_sphere[0], inst.world_sphere[1], inst.world_sphere[2]);
        let r = inst.world_sphere[3];
        for v in &dm.verts {
            let w = m.transform_point3(Vec3::from(v.pos));
            assert!(w.distance(ws) <= r * 1.05 + 0.1, "instance {} vertex {:?} outside its world sphere", inst.index, w);
        }
    }

    /// Every shipped halo3 (+ odst) playable cache: every BSP's instances decode - definitions
    /// resolve to real meshes, transforms are finite, and the row convention wins wherever the
    /// sphere check can test it (no BSP where the column reading fits more instances).
    /// Multiplayer caches (type 1) must resolve every instance; campaign caches stream some
    /// BSPs' Lbsp geometry from campaign.map (short mesh tables), so an out-of-range mesh there
    /// is tolerated and counted (the renderer skips it as "mesh without geometry").
    #[test]
    fn every_cache_instances_decode() {
        let caches = super::super::cache::installed_caches();
        if caches.is_empty() { eprintln!("skip: no halo3 maps folder"); return; }
        let (mut bsps, mut instances, mut unresolved) = (0usize, 0usize, 0usize);
        for p in &caches {
            let c = H3Cache::open(p).unwrap();
            if c.map_type == 3 || c.map_type == 4 { continue; }
            let strict = c.map_type == 1;
            for sbsp in c.find_tags(b"sbsp") {
                let bsp = load_bsp(&c, sbsp).unwrap_or_else(|e| panic!("{}: load_bsp: {e}", p.display()));
                if bsp.instances.is_empty() { continue; }
                bsps += 1;
                for i in &bsp.instances {
                    instances += 1;
                    if i.def < 0 || i.mesh < 0 { unresolved += 1; assert!(!strict, "{} '{}' instance {}: unresolved definition {}", p.display(), bsp.name, i.index, i.def); continue; }
                    assert!(i.is_finite(), "{} '{}' instance {}: non-finite transform", p.display(), bsp.name, i.index);
                    if (i.mesh as usize) >= bsp.meshes.len() {
                        unresolved += 1;
                        assert!(!strict, "{} '{}' instance {}: mesh {} out of range", p.display(), bsp.name, i.index, i.mesh);
                    }
                }
                assert!(bsp.rows, "{} '{}': rot_check {:?}", p.display(), bsp.name, bsp.rot_check);
                assert!(bsp.rot_check.2 == 0 || bsp.rot_check.0 * 10 >= bsp.rot_check.2 * 9, "{} '{}': rot_check {:?}", p.display(), bsp.name, bsp.rot_check);
            }
        }
        assert!(bsps >= 3 && instances > 1000, "bsps {bsps} instances {instances}");
        assert!(unresolved * 20 <= instances, "{unresolved} of {instances} instances unresolved");
        eprintln!("h3 instance sweep: {bsps} bsps, {instances} instances ({unresolved} unresolved / streamed-mesh skips on campaign caches)");
    }

    /// Every shipped halo3 (+ odst) playable cache: every BSP's cluster meshes decode with 0
    /// failures (indices in range, positions inside the compression bounds).
    #[test]
    fn every_cache_geometry_decodes() {
        let caches = super::super::cache::installed_caches();
        if caches.is_empty() { eprintln!("skip: no halo3 maps folder"); return; }
        let (mut maps, mut bsps, mut meshes, mut errors) = (0, 0, 0usize, 0usize);
        for p in &caches {
            let c = H3Cache::open(p).unwrap();
            if c.map_type == 3 || c.map_type == 4 { continue; }
            maps += 1;
            // Multiplayer maps (type 1) are self-contained single-BSP caches and must decode with
            // ZERO failures. Campaign maps (type 0) stream some BSP geometry from campaign.map,
            // which is out of scope here, so a resource with no local page is tolerated (skipped).
            let strict = c.map_type == 1;
            for sbsp in c.find_tags(b"sbsp") {
                let bsp = load_bsp(&c, sbsp).unwrap_or_else(|e| panic!("{}: load_bsp: {e}", p.display()));
                if bsp.geometry.is_none() { continue; }
                bsps += 1;
                let mut seen = std::collections::HashSet::new();
                for &mi in &bsp.clusters {
                    if !seen.insert(mi) { continue; }
                    match decode_mesh(&c, &bsp, mi as usize) {
                        Ok(Some(dm)) => {
                            meshes += 1;
                            assert!(dm.indices.iter().all(|&x| (x as usize) < dm.verts.len()),
                                "{} mesh {mi}: index out of range", p.display());
                        }
                        Ok(None) => {}
                        Err(e) => {
                            errors += 1;
                            assert!(!strict, "{} mesh {mi}: {e}", p.display());
                        }
                    }
                }
            }
        }
        assert!(maps >= 3 && bsps >= 3 && meshes > 100, "maps {maps} bsps {bsps} meshes {meshes}");
        eprintln!("h3 geometry sweep: {maps} maps, {bsps} bsps, {meshes} cluster meshes decoded ({errors} campaign streamed-resource skips)");
    }
    /// Audit (print-only, ignored): HMS_H3_AUDIT_MAP=<map> - every BSP of the cache: each cluster
    /// mesh's descriptors and decode result (with the reason when nothing decodes), every part's
    /// material class / template / albedo resolution, the instance skip reasons, and per material
    /// class the triangles that reach a lane. The "no ground on zanzibar" audit. #h3-audit
    #[test]
    #[ignore]
    fn map_decode_audit() {
        use super::super::materials::load_material;
        let map = std::env::var("HMS_H3_AUDIT_MAP").unwrap_or_else(|_| "zanzibar.map".into());
        let Some(c) = open(&map) else { return };
        let short = |s: &str| s.rsplit('\\').next().unwrap_or(s).to_string();
        for sbsp in c.find_tags(b"sbsp") {
            let bsp = load_bsp(&c, sbsp).expect("load_bsp");
            if let Some(g) = bsp.geometry.as_ref() {
                let e = &c.resources[g.entry];
                let seg = c.segments.get(e.loc.max(0) as usize).copied();
                eprintln!("  RES entry {} loc {} segment {:?} pages {:?} / {:?}; fixups (flag histogram) {:?}", e.index, e.loc, seg,
                    seg.and_then(|s| c.pages.get(s.primary_page.max(0) as usize).copied()), seg.and_then(|s| c.pages.get(s.secondary_page.max(0) as usize).copied()),
                    { let mut h = std::collections::BTreeMap::new(); for (_, (_, f)) in c.resource_fixups(e).iter() { *h.entry(*f).or_insert(0usize) += 1; } h });
            }
            if let Some(g) = bsp.geometry.as_ref() {
                // per-descriptor fixup flags: which page each vb / ib offset targets
                let e = &c.resources[g.entry];
                let fx = c.resource_fixups(e);
                let cd = c.control_data(e);
                let root = e.root as usize;
                let val = |loc: u32| fx.get(&loc).copied();
                let vblist = val((root + 0x1C) as u32).map(|v| v.0 as usize).unwrap_or(0);
                let iblist = val((root + 0x28) as u32).map(|v| v.0 as usize).unwrap_or(0);
                let mut vbflags = std::collections::BTreeMap::new();
                let mut ibflags = std::collections::BTreeMap::new();
                for i in 0..g.vbs.len() { if let Some((io, _)) = val((vblist + i * 12) as u32) { if let Some((_, f)) = val(io + 20) { *vbflags.entry((g.vbs[i].kind, f)).or_insert(0usize) += 1; } } }
                for i in 0..g.ibs.len() { if let Some((io, _)) = val((iblist + i * 12) as u32) { if let Some((_, f)) = val(io + 16) { *ibflags.entry(f).or_insert(0usize) += 1; } } }
                eprintln!("  RES vb (kind, page flag) histogram {vbflags:?}; ib page flag histogram {ibflags:?}; control data {} B", cd.len());
                // raw zone entry + the extent of the flag-0x40 region, then the first cluster's ib at
                // base + end40 (+ alignments) - is the flag-0x41 data appended after the required data?
                eprintln!("  RES entry flags {:#x} kind {} ctrl_off {} ctrl_size {}", e.flags, e.kind, e.ctrl_off, e.ctrl_size);
                let mut end40 = 0usize; let mut end41 = 0usize;
                for i in 0..g.vbs.len() { if let Some((io, _)) = val((vblist + i * 12) as u32) { if let Some((o, f)) = val(io + 20) { let e2 = o as usize + g.vbs[i].size as usize; if f == 0x41 { end41 = end41.max(e2) } else { end40 = end40.max(e2) } } } }
                for i in 0..g.ibs.len() { if let Some((io, _)) = val((iblist + i * 12) as u32) { if let Some((o, f)) = val(io + 16) { let e2 = o as usize + g.ibs[i].size as usize; if f == 0x41 { end41 = end41.max(e2) } else { end40 = end40.max(e2) } } } }
                eprintln!("  RES flag-0x40 region end {end40}, flag-0x41 region end {end41}");
                if let Some(&mi) = bsp.clusters.first() {
                    let md = &bsp.meshes[mi as usize];
                    if let (Some(ib), Some(vb)) = (g.ibs.get(md.ib as usize), g.vbs.get(md.vb[0] as usize)) {
                        if let Ok((page, base)) = c.read_resource(e) {
                            for align in [1usize, 16, 4096] {
                                let b = base + (end40 + align - 1) / align * align;
                                let istart = b + ib.off.unwrap_or(0) as usize;
                                let maxi = page.get(istart..istart + ib.size as usize).map(|bb| bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).max().unwrap_or(0));
                                let first: Vec<u16> = page.get(istart..istart + 12).map(|bb| bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect()).unwrap_or_default();
                                eprintln!("  RES first cluster ib at base+end40 (align {align}): max index {maxi:?} vs vb count {} first {first:?}", vb.count);
                            }
                            // brute force: scan the page for the offset where this ib is plausible (all < count, first tri distinct)
                            let n = ib.size as usize;
                            let cnt = vb.count as u16;
                            let mut hits = Vec::new();
                            let mut o = base; while o + n <= page.len() && hits.len() < 5 {
                                let bb = &page[o..o + n];
                                if bb.chunks_exact(2).all(|x| u16::from_le_bytes([x[0], x[1]]) < cnt) { hits.push(o as i64 - base as i64 - ib.off.unwrap_or(0) as i64); }
                                o += 2;
                            }
                            eprintln!("  RES brute-force candidate (page offset - base - ib.off) for the first cluster ib: {hits:?}");
                        }
                    }
                }
                // the play Default Location's sublocation table indices (+0xC / +0xE) and the
                // Streaming Sublocation Tables block (play +0x24: total size @0, sublocations block @4
                // of (memory offset, memory size)) - the optional data's home when no optional page
                {
                    let d = c.data();
                    let play = c.find_tags(b"play")[0];
                    let pm = c.tag_meta(play).unwrap();
                    let (_, so) = c.block(pm + 0x3C).unwrap();
                    let se = so + e.loc as usize * 16;
                    eprintln!("  RES segment raw {:02x?} -> pageable subloc table {} optional subloc table {}", &d[se..se + 16], d.i16_at(se + 0xC), d.i16_at(se + 0xE));
                    if let Some((nt, to)) = c.block(pm + 0x24) {
                        eprintln!("  RES {nt} streaming sublocation tables");
                        for ti in [d.i16_at(se + 0xC), d.i16_at(se + 0xE)] {
                            if ti < 0 || ti as usize >= nt { continue; }
                            let t = to + ti as usize * 16;
                            let subs = c.block(t + 4).map(|(n, o)| (0..n).map(|k| (d.i32_at(o + k * 8), d.i32_at(o + k * 8 + 4))).collect::<Vec<_>>()).unwrap_or_default();
                            eprintln!("    table {ti}: total {} subs {:?}", d.i32_at(t), &subs[..subs.len().min(8)]);
                            if let Some(&mi) = bsp.clusters.first() {
                                let md = &bsp.meshes[mi as usize];
                                if let (Some(ib), Some(vb), Ok((page, base))) = (g.ibs.get(md.ib as usize), g.vbs.get(md.vb[0] as usize), c.read_resource(e)) {
                                    for (k, (mo, _)) in subs.iter().enumerate().take(4) {
                                        for (label, b) in [("page+mo", *mo as usize), ("base+mo", base + *mo as usize)] {
                                            let istart = b + ib.off.unwrap_or(0) as usize;
                                            let maxi = page.get(istart..istart + ib.size as usize).map(|bb| bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).max().unwrap_or(0));
                                            let first: Vec<u16> = page.get(istart..istart + 12).map(|bb| bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect()).unwrap_or_default();
                                            eprintln!("    sub {k} {label}: first cluster ib max index {maxi:?} vs vb count {} first {first:?}", vb.count);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // the page's own streaming sublocation table index (File Location @0x56)
                    let (_, po) = c.block(pm + 0x18).unwrap();
                    let pe = po + 1420 * 88;
                    eprintln!("  RES page 1420 file-location: refcount {} subloc table {}", d.i16_at(pe + 0x54), d.i16_at(pe + 0x56));
                }
                // segments around ours, VB sanity (unit normals, positions inside the cluster bounds),
                // and a page-wide scan for the first cluster's real index buffer (all indices < count,
                // wide index range, face normals agreeing with the stored vertex normals)
                {
                    for k in (e.loc as usize).saturating_sub(3)..(e.loc as usize + 4).min(c.segments.len()) { eprintln!("  RES segment {k}: {:?}", c.segments[k]); }
                    if let Some((&mi, cb)) = bsp.clusters.first().zip(bsp.cluster_bounds.first()) {
                        let md = &bsp.meshes[mi as usize];
                        if let (Some(ib), Some(vb), Ok((page, base))) = (g.ibs.get(md.ib as usize), g.vbs.get(md.vb[0] as usize), c.read_resource(e)) {
                            let vstart = base + vb.off.unwrap_or(0) as usize;
                            let stride = vb.stride as usize;
                            let cnt = vb.count as usize;
                            let mut inside = 0usize; let mut unit = 0usize;
                            let mut vp: Vec<[f32; 3]> = Vec::new(); let mut vn: Vec<[f32; 3]> = Vec::new();
                            for k in 0..cnt {
                                let v = &page[vstart + k * stride..vstart + k * stride + stride];
                                let p = [v.f32_at(0), v.f32_at(4), v.f32_at(8)];
                                let n = [v.i16_at(0x14) as f32 / 32767.0, v.i16_at(0x16) as f32 / 32767.0, v.i16_at(0x18) as f32 / 32767.0];
                                if (0..3).all(|a| p[a] >= cb.0[a] - 1.0 && p[a] <= cb.1[a] + 1.0) { inside += 1; }
                                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                                if (l - 1.0).abs() < 0.05 { unit += 1; }
                                vp.push(p); vn.push(n);
                            }
                            eprintln!("  RES cluster 0 VB sanity: {inside}/{cnt} inside cluster bounds {:?}..{:?}, {unit}/{cnt} unit normals", cb.0, cb.1);
                            let n = ib.size as usize;
                            let agree = |bb: &[u8]| -> (f32, usize) {
                                let idx: Vec<usize> = bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]]) as usize).collect();
                                let (mut acc, mut m) = (0f32, 0usize);
                                for t in idx.chunks_exact(3) {
                                    if t[0] >= cnt || t[1] >= cnt || t[2] >= cnt { return (0.0, 0); }
                                    let p = |i: usize| glam::Vec3::from(vp[i]);
                                    let fnm = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
                                    if fnm.length() < 1e-9 { continue; }
                                    let nn = (glam::Vec3::from(vn[t[0]]) + glam::Vec3::from(vn[t[1]]) + glam::Vec3::from(vn[t[2]])).normalize_or_zero();
                                    acc += fnm.normalize().dot(nn).abs(); m += 1;
                                }
                                (if m > 0 { acc / m as f32 } else { 0.0 }, m)
                            };
                            let mut best: Vec<(f32, usize, i64)> = Vec::new();
                            let mut o = 0usize;
                            while o + n <= page.len() {
                                let bb = &page[o..o + n];
                                let maxi = bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]]) as usize).max().unwrap_or(0);
                                if maxi < cnt && maxi * 10 >= cnt * 7 {
                                    let (a, m) = agree(bb);
                                    if m > 50 && a > 0.9 { best.push((a, m, o as i64 - base as i64 - ib.off.unwrap_or(0) as i64)); if best.len() >= 6 { break; } }
                                }
                                o += 2;
                            }
                            eprintln!("  RES page-wide scan for cluster 0's IB (agreement, tris, page offset - base - ib.off): {best:?}");
                        }
                    }
                }
                // try the first cluster's ib at alternative bases
                if let Some(&mi) = bsp.clusters.first() {
                    let md = &bsp.meshes[mi as usize];
                    if let (Some(ib), Some(vb)) = (g.ibs.get(md.ib as usize), g.vbs.get(md.vb[0] as usize)) {
                        if let Ok((page, base)) = c.read_resource(e) {
                            for (label, b) in [("base", base), ("zero", 0usize)] {
                                let istart = b + ib.off.unwrap_or(0) as usize;
                                let maxi = page.get(istart..istart + ib.size as usize).map(|bb| bb.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).max().unwrap_or(0));
                                eprintln!("  RES first cluster ib at {label}: max index {maxi:?} vs vb count {}", vb.count);
                            }
                        }
                    }
                }
            }
            {
                let n_sec = c.segments.iter().filter(|s| s.secondary_page >= 0).count();
                let mut hist = std::collections::BTreeMap::new();
                for s in &c.segments { *hist.entry((s.primary_page, s.secondary_page)).or_insert(0usize) += 1; }
                eprintln!("  PLAY {} pages, {} segments ({n_sec} with a secondary page); (primary, secondary) page pairs {:?}", c.pages.len(), c.segments.len(), hist.iter().take(12).collect::<Vec<_>>());
                for (i, pg) in c.pages.iter().enumerate().filter(|(_, p)| p.compressed > 1_000_000) { eprintln!("  PAGE {i}: {pg:?}"); }
            }
            eprintln!("== {map} bsp '{}' meshes {} clusters {} (ids {:?}) instances {} defs {} materials {} geometry {}",
                bsp.name, bsp.meshes.len(), bsp.clusters.len(), &bsp.cluster_ids[..bsp.cluster_ids.len().min(8)], bsp.instances.len(), bsp.instance_defs.len(), bsp.materials.len(), bsp.geometry.is_some());
            let Some(g) = bsp.geometry.as_ref() else { continue };
            eprintln!("  geometry: {} vbs, {} ibs; vb kinds/strides: {:?}", g.vbs.len(), g.ibs.len(),
                { let mut h = std::collections::BTreeMap::new(); for v in &g.vbs { *h.entry((v.kind, v.stride)).or_insert(0usize) += 1; } h });
            let mut class_tris: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new(); // class -> (tris, parts)
            let mut mats_seen: std::collections::BTreeMap<i16, usize> = std::collections::BTreeMap::new();
            let mut seen = std::collections::HashSet::new();
            let (mut n_dec, mut n_none, mut n_err) = (0usize, 0usize, 0usize);
            for (ci, &mi) in bsp.clusters.iter().enumerate() {
                if !seen.insert(mi) { continue; }
                let md = &bsp.meshes[mi as usize];
                let vbs: Vec<String> = md.vb.iter().filter(|&&v| v >= 0).map(|&v| g.vbs.get(v as usize).map(|i| format!("vb{v}[k{} s{} n{} off {:?}]", i.kind, i.stride, i.count, i.off)).unwrap_or_else(|| format!("vb{v}[OOR]"))).collect();
                let ib = g.ibs.get(md.ib as usize).map(|i| format!("ib{}[fmt {} size {} off {:?}]", md.ib, i.fmt, i.size, i.off)).unwrap_or_else(|| format!("ib{}[OOR]", md.ib));
                let res = decode_mesh(&c, &bsp, mi as usize);
                let (status, tris) = match &res {
                    Ok(Some(dm)) => { n_dec += 1; (format!("OK verts {} tris {}", dm.verts.len(), dm.indices.len() / 3), dm.indices.len() / 3) }
                    Ok(None) => {
                        n_none += 1;
                        // why: page length vs the descriptor offsets, and the raw max index
                        let mut why = String::new();
                        if let (Some(ib), Some(vb)) = (g.ibs.get(md.ib as usize), g.vbs.get(md.vb[0] as usize)) {
                            if let Ok((page, base)) = c.read_resource(&c.resources[g.entry]) {
                                let istart = base + ib.off.unwrap_or(0) as usize;
                                let maxi = page.get(istart..istart + ib.size as usize).map(|b| b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).max().unwrap_or(0));
                                let first: Vec<u16> = page.get(istart..istart + 16).map(|b| b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect()).unwrap_or_default();
                                why = format!("page {} B base {base} istart {istart} vstart {} max_index {:?} (vb count {}) first idx {:?}", page.len(), base + vb.off.unwrap_or(0) as usize, maxi, vb.count, first);
                            }
                        }
                        (format!("NONE (has_geometry {} parts {} vb0 {} ib {}) {why}", md.has_geometry(), md.parts.len(), md.vb[0], md.ib), 0)
                    }
                    Err(e) => { n_err += 1; (format!("ERR {e}"), 0) }
                };
                let parts: Vec<String> = md.parts.iter().map(|p| {
                    let bm = usize::try_from(p.material).ok().and_then(|k| bsp.materials.get(k));
                    let cls = bm.map(|b| String::from_utf8_lossy(&b.class).into_owned()).unwrap_or("?".into());
                    *mats_seen.entry(p.material).or_insert(0) += 1;
                    let e = class_tris.entry(cls.clone()).or_insert((0, 0));
                    if res.as_ref().map_or(false, |r| r.is_some()) { e.0 += p.index_count as usize / 3; e.1 += 1; }
                    format!("{}:{}({})x{}", p.material, cls, bm.map(|b| short(&b.name)).unwrap_or_default(), p.index_count / 3)
                }).collect();
                eprintln!("  C{ci:3} mesh {mi:3} vt {} it {} bounds flags {:#x} pos {:?}..{:?} | {} {} | {status} | parts {}",
                    md.vertex_type, md.index_type, md.bounds.flags, md.bounds.pos_min, md.bounds.pos_max, vbs.join(" "), ib, parts.join(" "));
                let _ = tris;
            }
            eprintln!("  clusters: {} unique meshes: {n_dec} decoded, {n_none} none, {n_err} errors", seen.len());
            eprintln!("  triangles reaching a lane per material class (decoded cluster meshes): {class_tris:?}");
            // materials
            for (i, bm) in bsp.materials.iter().enumerate() {
                let Some(rt) = bm.render_tag else { eprintln!("  MAT [{i:3}] <null>"); continue };
                let m = load_material(&c, rt);
                let base = m.base.map(|b| { let bi = super::super::bitmaps::bitmap_info(&c, b); format!("{} {}", short(c.tag_name(b)), bi.map(|i| format!("{}x{} fmt {} kind {} px {}", i.w, i.h, i.format, i.kind, i.px_size)).unwrap_or("NOINFO".into())) });
                eprintln!("  MAT [{i:3}] {} ({}) opts {:?} used-by {} parts -> base {} | tex {:?}", short(&m.name), String::from_utf8_lossy(&bm.class), m.template_opts, mats_seen.get(&(i as i16)).copied().unwrap_or(0), base.as_deref().unwrap_or("<none>"),
                    m.params.iter().map(|(n, b)| format!("{n}={}", short(c.tag_name(*b)))).collect::<Vec<_>>());
            }
            // instances
            let mut skips: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
            let mut inst_meshes = std::collections::HashSet::new();
            for inst in &bsp.instances {
                let reason = if inst.def < 0 || inst.mesh < 0 { "unresolved definition" }
                    else if !bsp.meshes.get(inst.mesh as usize).map_or(false, |m| m.has_geometry()) { "mesh without geometry" }
                    else if !inst.is_finite() { "non-finite transform" }
                    else if inst.not_for_render() { "not-for-render flag" }
                    else { inst_meshes.insert(inst.mesh); "drawn" };
                *skips.entry(reason).or_insert(0) += 1;
            }
            let (mut idec, mut inone) = (0usize, 0usize);
            for &mi in &inst_meshes { match decode_mesh(&c, &bsp, mi as usize) { Ok(Some(_)) => idec += 1, _ => { inone += 1; let md = &bsp.meshes[mi as usize]; eprintln!("  INST mesh {mi} no decode: vt {} it {} vb {:?} ib {} bounds {:#x}", md.vertex_type, md.index_type, &md.vb[..4], md.ib, md.bounds.flags); } } }
            eprintln!("  instances: {skips:?}; {} unique drawn meshes: {idec} decode, {inone} none", inst_meshes.len());
        }
    }
    /// #h3-zanzibar regression: zanzibar's Lbsp geometry slice is over 16 MB, so its index
    /// buffers (and 336 lighting streams) sit at fixup addresses >= 0x1000000 whose bit 24 lives
    /// in the fixup word's top byte. With the 29-bit address decode every cluster decodes:
    /// cluster 0 = 296 vertices / 188 triangles, every index in range, and the face normals agree
    /// with the stored vertex normals (0.998 measured; a wrong base reads ~0.5 noise).
    #[test]
    fn zanzibar_fixups_past_16mb_decode() {
        let Some(c) = open("zanzibar.map") else { return };
        let sbsp = c.find_tags(b"sbsp")[0];
        let bsp = load_bsp(&c, sbsp).unwrap();
        let g = bsp.geometry.as_ref().expect("geometry");
        let fx = c.resource_fixups(&c.resources[g.entry]);
        let high = fx.values().filter(|(a, t)| *a >= 0x100_0000 && *t == super::super::cache::FIXUP_TYPE_RAW_PAGE).count();
        assert!(high > 700, "expected the index buffers + lighting streams past 16 MB, got {high} high raw-page fixups");
        assert!(fx.values().all(|(_, t)| *t == super::super::cache::FIXUP_TYPE_RAW_PAGE || *t == super::super::cache::FIXUP_TYPE_CONTROL), "unexpected fixup type");
        let mi = bsp.clusters[0] as usize;
        let dm = decode_mesh(&c, &bsp, mi).unwrap().expect("cluster 0 decodes");
        assert_eq!((dm.verts.len(), dm.indices.len() / 3), (296, 188));
        assert!(dm.indices.iter().all(|&i| (i as usize) < dm.verts.len()));
        let (mut acc, mut n) = (0f32, 0usize);
        for t in dm.indices.chunks_exact(3) {
            let p = |i: usize| glam::Vec3::from(dm.verts[t[i] as usize].pos);
            let fnm = (p(1) - p(0)).cross(p(2) - p(0));
            if fnm.length() < 1e-9 { continue; }
            let vn = (glam::Vec3::from(dm.verts[t[0] as usize].normal) + glam::Vec3::from(dm.verts[t[1] as usize].normal) + glam::Vec3::from(dm.verts[t[2] as usize].normal)).normalize_or_zero();
            acc += fnm.normalize().dot(vn).abs(); n += 1;
        }
        assert!(n > 150 && acc / n as f32 > 0.95, "normal agreement {} over {n}", acc / n as f32);
        // every cluster mesh and every drawn instance mesh decodes
        let mut seen = std::collections::HashSet::new();
        for &mi in &bsp.clusters { if seen.insert(mi) && bsp.meshes[mi as usize].has_geometry() { assert!(decode_mesh(&c, &bsp, mi as usize).unwrap().is_some(), "cluster mesh {mi}"); } }
        for i in bsp.instances.iter().filter(|i| i.mesh >= 0 && bsp.meshes.get(i.mesh as usize).map_or(false, |m| m.has_geometry())) {
            assert!(decode_mesh(&c, &bsp, i.mesh as usize).unwrap().is_some(), "instance mesh {}", i.mesh);
        }
    }

    /// Every installed halo3 multiplayer cache: per map, cluster meshes decoded / total and
    /// instances drawn / total (the "drawn vs total" sweep), asserting every cluster mesh and
    /// every drawable instance mesh of a multiplayer map decodes.
    #[test]
    fn every_cache_draw_counts() {
        let caches = super::super::cache::installed_caches();
        if caches.is_empty() { eprintln!("skip: no halo3 maps folder"); return; }
        for p in &caches {
            let c = H3Cache::open(p).unwrap();
            if c.map_type != 1 { continue; }
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            for sbsp in c.find_tags(b"sbsp") {
                let bsp = load_bsp(&c, sbsp).unwrap();
                if bsp.geometry.is_none() { continue; }
                let mut seen = std::collections::HashSet::new();
                let (mut cl_ok, mut cl_n) = (0usize, 0usize);
                // an empty cluster mesh (no parts / buffers: a sky-only or portal cluster) has nothing to draw
                for &mi in &bsp.clusters { if !seen.insert(mi) || !bsp.meshes[mi as usize].has_geometry() { continue; } cl_n += 1; if decode_mesh(&c, &bsp, mi as usize).unwrap().is_some() { cl_ok += 1; } }
                let (mut in_ok, mut in_geom, mut in_n) = (0usize, 0usize, 0usize);
                let mut cache: std::collections::HashMap<i16, bool> = std::collections::HashMap::new();
                for i in &bsp.instances {
                    in_n += 1;
                    if i.mesh < 0 || !bsp.meshes.get(i.mesh as usize).map_or(false, |m| m.has_geometry()) || i.not_for_render() || !i.is_finite() { continue; }
                    in_geom += 1;
                    let ok = *cache.entry(i.mesh).or_insert_with(|| decode_mesh(&c, &bsp, i.mesh as usize).map(|d| d.is_some()).unwrap_or(false));
                    if ok { in_ok += 1; }
                }
                eprintln!("h3 sweep {name:20} '{}': clusters {cl_ok}/{cl_n} decoded, instances {in_ok}/{in_n} drawn ({} without geometry / not-for-render)", bsp.name.rsplit('\\').next().unwrap_or(""), in_n - in_geom);
                assert_eq!(cl_ok, cl_n, "{name}: cluster meshes");
                assert_eq!(in_ok, in_geom, "{name}: instance meshes");
            }
        }
    }
}




