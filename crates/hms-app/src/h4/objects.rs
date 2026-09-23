//! Halo 4 scenario objects: `scnr` placement tables -> palette tag -> `hlmt` -> `mode`
//! render model -> geometry resource + meshes. Layout: docs/halo4_geometry_layout.md section 11.
//!
//! Layout facts (probed over all 54 shipped Halo 4 caches, re-asserted by the tests below on
//! wraparound / ca_forge_ravine / m10_crash):
//!   * placement/palette block pairs (scnr main struct offsets) and the placement element
//!     STRIDE per table - see `PLACEMENT_TABLES`; palette element = 16 B (one tag ref, may be
//!     null); the placement header is Reach's s_scenario_object_datum header:
//!       i16 palette index @0, i16 name index @2, u32 flags @4, f32x3 position @8,
//!       f32x3 rotation @0x14 (yaw, pitch, roll radians as stored), f32 scale @0x20 (0 = 1.0)
//!     Every element of every table on every map passes the header sanity test at the pinned
//!     stride (palette/name index in range, finite position/rotation, |rotation| <= 2*pi).
//!   * scnr +0x144 = object names, 8 B {sid @0, i16 object type @4, i16 placement index @6}
//!   * obje (scen/bloc/vehi/weap/eqip/mach/ctrl/bipd/efsc) +0x64 = `hlmt` ref; hlmt +0x00 =
//!     `mode` ref (every hlmt on wraparound + m10_crash)
//!   * `mode` main struct = 400 B: +0x0C regions (16 B: sid @0, permutations block @4;
//!     permutation 16 B: sid @0, i16 mesh index @4, i16 mesh count @6), +0x30 nodes (112 B, see
//!     `H4Node`), +0x48 materials (44 B, `mat ` ref @0), +0x68 meshes (112 B, same element as the
//!     sbsp), +0x80 compression bounds (52 B, ONE model-wide element), +0xF8 u32 geometry
//!     resource id (type 1, owner `mode`)
//!   * mesh element bytes past the buffer indices: u16 flags @0x2E, u8 rigid node @0x30 (255 =
//!     skinned), u8 vertex type @0x31, u8 index type @0x34 (3 = triangle LIST, 5 = triangle
//!     STRIP with alternating winding, degenerates included)
//!   * vertices of rigid multi-node models are stored in MODEL space (mongoose wheels / banshee
//!     rudders / phantom doors decode at their node's absolute position) - no node transform
//!     is needed for the default pose
//!   * every model vertex format carries pos f32x3 @0 (normalized to the bounds), 0 @12, uv u16x2
//!     @16, normal snorm16x3 @20, tangent @28 like the BSP's type 2 / 36 B: seen (vb type /
//!     stride) 2/36 rigid, 3/44 skinned (u8x4 node indices @36, u8x4 weights @40), 47/40,
//!     48/44, 5/40, 18/48; type 37/40 differs (word @12 non-zero) and is rejected
//! Not confirmed against the game: the euler -> basis convention (ported from Reach's
//! ScenarioObjectWalker, same engine family), the placement tail (u32 @0x6C looks like a
//! unique id, i16 @0x70 like the origin BSP), the `gint` placement stride (no map has giant
//! placements) and where creature placements live (scnr +0x650 is NOT them), the meaning of
//! the permutation words @8/@12, the skinned-vertex weight layout of types 18/47/48.

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use glam::{Mat4, Vec3, Vec4};

use super::cache::{ByteRead, H4Cache};
use super::geometry::{
    parse_geometry_resource, sn16, Bounds, DecodedMesh, GeometryResource, H4Vertex, MeshDef, Part, VbInfo,
    BOUNDS_ELEM, MESH_ELEM, PART_ELEM,
};

pub const OFF_SCNR_OBJECT_NAMES: usize = 0x144;
pub const OBJECT_NAME_ELEM: usize = 8;
pub const PALETTE_ELEM: usize = 16;
pub const OFF_OBJE_HLMT: usize = 0x64;
pub const OFF_HLMT_MODE: usize = 0x00;
pub const OFF_MODE_REGIONS: usize = 0x0C;
pub const OFF_MODE_NODES: usize = 0x30;
pub const OFF_MODE_MATERIALS: usize = 0x48;
pub const OFF_MODE_MESHES: usize = 0x68;
pub const OFF_MODE_BOUNDS: usize = 0x80;
pub const OFF_MODE_GEOMETRY_RID: usize = 0xF8;
pub const MODE_STRUCT_SIZE: usize = 400;
pub const REGION_ELEM: usize = 16;
pub const PERMUTATION_ELEM: usize = 16;
pub const NODE_ELEM: usize = 112;
pub const MODE_MATERIAL_ELEM: usize = 44;

/// One (placements, palette) block pair in the scnr main struct. `stride` = placement element
/// size, checked on every map that has placements of that class (0 = unknown, table skipped).
#[derive(Clone, Copy, Debug)]
pub struct PlacementTable {
    pub class: [u8; 4],
    pub placements: usize,
    pub palette: usize,
    pub stride: usize,
}

pub const PLACEMENT_TABLES: [PlacementTable; 14] = [
    PlacementTable { class: *b"scen", placements: 0x150, palette: 0x15C, stride: 380 },
    PlacementTable { class: *b"bipd", placements: 0x168, palette: 0x174, stride: 368 },
    PlacementTable { class: *b"vehi", placements: 0x180, palette: 0x18C, stride: 384 },
    PlacementTable { class: *b"eqip", placements: 0x198, palette: 0x1A4, stride: 340 },
    PlacementTable { class: *b"weap", placements: 0x1B0, palette: 0x1BC, stride: 368 },
    PlacementTable { class: *b"mach", placements: 0x1D4, palette: 0x1E0, stride: 388 },
    PlacementTable { class: *b"term", placements: 0x1EC, palette: 0x1F8, stride: 192 },
    PlacementTable { class: *b"ctrl", placements: 0x204, palette: 0x210, stride: 380 },
    PlacementTable { class: *b"dspn", placements: 0x21C, palette: 0x228, stride: 372 },
    PlacementTable { class: *b"ssce", placements: 0x234, palette: 0x240, stride: 208 },
    // giants: palette seen once (n=0 placements everywhere) - stride unknown, skipped
    PlacementTable { class: *b"gint", placements: 0x24C, palette: 0x258, stride: 0 },
    PlacementTable { class: *b"efsc", placements: 0x264, palette: 0x270, stride: 340 },
    PlacementTable { class: *b"spnr", placements: 0x27C, palette: 0x288, stride: 188 },
    PlacementTable { class: *b"bloc", placements: 0x638, palette: 0x644, stride: 376 },
];

#[derive(Clone, Debug)]
pub struct H4Placement {
    /// Class of the palette tag (scen / bloc / vehi / ...).
    pub class: [u8; 4],
    pub palette_tag: usize,
    /// Object name from the scnr names block (empty when the name index is -1).
    pub name: String,
    pub pos: [f32; 3],
    /// yaw, pitch, roll radians as stored.
    pub rot: [f32; 3],
    /// 1.0 when the stored scale is <= 0 or not finite (Reach convention).
    pub scale: f32,
    /// Raw placement flags u32 @4 (semantics unknown).
    pub flags: u32,
    /// Explicit world (forward, up) basis - set by map-variant objects (mvar.rs), whose
    /// orientation is stored as vectors, not eulers; `rot` is ignored when present.
    pub basis: Option<(Vec3, Vec3)>,
    /// The multiplayer (team, colour override) of a MAP-VARIANT placement - what the engine
    /// resolves the object's primary change colour from (`h4::tint::primary_change_color`).
    /// `None` = a scenario placement, which keeps the object's authored change colours. #h4-veh
    pub mp: Option<(i8, Option<u8>)>,
    /// The hlmt model variant a map-variant placement was authored with (palette variant ->
    /// `object_variant_index`); `None` = the object's own default. #h4-veh
    pub model_variant: Option<usize>,
}

impl H4Placement {
    pub fn class_str(&self) -> String { String::from_utf8_lossy(&self.class).into_owned() }
}

/// Every placement of every table with a known stride, in table order. Placements whose
/// palette index is -1 / out of range or whose palette entry is a null ref are skipped.
pub fn load_placements(c: &H4Cache) -> Vec<H4Placement> {
    let d = c.data();
    let mut out = Vec::new();
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let names = c.block(sm + OFF_SCNR_OBJECT_NAMES);
    for t in PLACEMENT_TABLES.iter().filter(|t| t.stride > 0) {
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
            out.push(H4Placement {
                class,
                palette_tag,
                name,
                pos: [f(8), f(12), f(16)],
                rot: [f(0x14), f(0x18), f(0x1C)],
                scale,
                flags: d.u32_at(e + 4),
                basis: None,
                mp: None,
                model_variant: None,
            });
        }
    }
    out
}

/// Euler (yaw about +Z, pitch about +Y, roll about +X, radians) -> forward / up basis.
/// Port of Reach's ScenarioObjectWalker::EulerToBasis (same engine family; not compared
/// against the Halo 4 game - see the module header).
pub fn euler_to_basis(e: [f32; 3]) -> (Vec3, Vec3) {
    let (sy, cy) = e[0].sin_cos();
    let (sp, cp) = e[1].sin_cos();
    let (sr, cr) = e[2].sin_cos();
    let fwd = Vec3::new(cy * cp, sy * cp, -sp);
    let up = Vec3::new(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr);
    (fwd, up)
}

/// World matrix of a placement: columns [fwd, up x fwd, up, pos] scaled uniformly - the same
/// build as scene.rs `object_matrix` for Reach objects (local +X = forward, +Z = up).
pub fn placement_matrix(p: &H4Placement) -> Mat4 {
    // variant objects carry their basis directly
    let (f, u) = p.basis.unwrap_or_else(|| euler_to_basis(p.rot));
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

/// obje (any object class) -> hlmt (+0x64) -> mode (+0x00).
pub fn model_of(c: &H4Cache, palette_tag: usize) -> Option<usize> {
    let om = c.tag_meta(palette_tag)?;
    let hlmt = c.tag_ref_of(om + OFF_OBJE_HLMT, b"hlmt")?;
    let hm = c.tag_meta(hlmt)?;
    c.tag_ref_of(hm + OFF_HLMT_MODE, b"mode")
}

#[derive(Clone, Debug, Default)]
pub struct H4Permutation {
    pub name: String,
    pub mesh_index: i16,
    pub mesh_count: i16,
}

#[derive(Clone, Debug, Default)]
pub struct H4Region {
    pub name: String,
    /// Raw string id of the name (hlmt variant region lists key on it).
    pub sid: u32,
    pub permutations: Vec<H4Permutation>,
}

/// Render-model node (112 B element). Vertices are already in model space; these are kept for
/// markers / animation later.
#[derive(Clone, Debug, Default)]
pub struct H4Node {
    pub name: String,
    pub parent: i16,
    pub first_child: i16,
    pub next_sibling: i16,
    pub translation: [f32; 3],
    /// x, y, z, w
    pub rotation: [f32; 4],
    pub scale: f32,
    /// inverse forward / left / up / position @44 / @56 / @68 / @80
    pub inverse: [[f32; 3]; 4],
}

/// Per-mesh bytes of the 112 B mesh element that the BSP reader does not keep.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshInfo {
    pub flags: u16,
    /// Rigid node index (255 = skinned / none).
    pub node: u8,
    pub vertex_type: u8,
    /// 3 = triangle list, 5 = triangle strip.
    pub index_type: u8,
}

pub struct H4Model {
    pub tag: usize,
    pub name: String,
    pub meshes: Vec<MeshDef>,
    pub mesh_info: Vec<MeshInfo>,
    pub geometry: GeometryResource,
    /// `mat ` tag per material slot (None for a null ref).
    pub materials: Vec<Option<usize>>,
    pub regions: Vec<H4Region>,
    pub nodes: Vec<H4Node>,
    /// Mesh indices of permutation 0 of every region (the default LOD / variant), deduplicated.
    pub lod0_meshes: Vec<usize>,
}

fn read_bounds(d: &[u8], o: usize) -> Bounds {
    let f = |k: usize| d.f32_at(o + 4 + k * 4);
    Bounds { flags: d.i32_at(o), pos_min: [f(0), f(2), f(4)], pos_max: [f(1), f(3), f(5)], uv_min: [f(6), f(8)], uv_max: [f(7), f(9)] }
}

/// Parse a `mode` tag: meshes, bounds, materials, regions, nodes and its geometry resource.
pub fn load_model(c: &H4Cache, mode_tag: usize) -> Result<H4Model> {
    if c.tag_class(mode_tag).as_ref() != Some(b"mode") { bail!("tag {mode_tag} is not a mode"); }
    let d = c.data();
    let mm = c.tag_meta(mode_tag).ok_or_else(|| anyhow!("mode meta"))?;
    let name = c.tag_name(mode_tag).to_string();
    // geometry resource: rid @0xF8, else the largest type-1 resource referenced from the struct
    // resource kinds are compared by NAME (their index differs per map - see cache.rs)
    let is_geo = |e: &&super::cache::ResourceEntry| c.kind_is(e.kind, H4Cache::RES_GEOMETRY) && e.owner == *b"mode";
    let mut entry = c.resource_by_id(d.u32_at(mm + OFF_MODE_GEOMETRY_RID)).filter(is_geo);
    if entry.is_none() {
        for off in (0..MODE_STRUCT_SIZE).step_by(4) {
            if let Some(e) = c.resource_by_id(d.u32_at(mm + off)).filter(is_geo) {
                if entry.map_or(true, |b| e.def_len > b.def_len) { entry = Some(e); }
            }
        }
    }
    let entry = entry.ok_or_else(|| anyhow!("{name}: no render geometry resource"))?;
    let geometry = parse_geometry_resource(c, entry)?;
    // meshes + the single model-wide bounds element
    let (nm, om) = c.block(mm + OFF_MODE_MESHES).unwrap_or((0, 0));
    let (nb, ob) = c.block(mm + OFF_MODE_BOUNDS).unwrap_or((0, 0));
    let mut meshes = Vec::with_capacity(nm);
    let mut mesh_info = Vec::with_capacity(nm);
    for i in 0..nm.min(4096) {
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
        if nb > 0 { m.bounds = read_bounds(d, ob + i.min(nb - 1) * BOUNDS_ELEM); }
        meshes.push(m);
        mesh_info.push(MeshInfo { flags: d.u16_at(e + 0x2E), node: d.u8_at(e + 0x30), vertex_type: d.u8_at(e + 0x31), index_type: d.u8_at(e + 0x34) });
    }
    // materials
    let mut materials = Vec::new();
    if let Some((n, o)) = c.block(mm + OFF_MODE_MATERIALS) {
        for i in 0..n.min(4096) { materials.push(c.tag_ref_of(o + i * MODE_MATERIAL_ELEM, b"mat ")); }
    }
    // regions -> permutations
    let mut regions = Vec::new();
    let mut lod0_meshes: Vec<usize> = Vec::new();
    if let Some((nr, or)) = c.block(mm + OFF_MODE_REGIONS) {
        for i in 0..nr.min(1024) {
            let e = or + i * REGION_ELEM;
            let mut r = H4Region { name: c.sid(d.u32_at(e)), sid: d.u32_at(e), permutations: Vec::new() };
            if let Some((np, op)) = c.block(e + 4) {
                for k in 0..np.min(4096) {
                    let p = op + k * PERMUTATION_ELEM;
                    r.permutations.push(H4Permutation { name: c.sid(d.u32_at(p)), mesh_index: d.i16_at(p + 4), mesh_count: d.i16_at(p + 6) });
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
    // nodes
    let mut nodes = Vec::new();
    if let Some((nn, on)) = c.block(mm + OFF_MODE_NODES) {
        for i in 0..nn.min(4096) {
            let e = on + i * NODE_ELEM;
            let f = |o: usize| d.f32_at(e + o);
            let v3 = |o: usize| [f(o), f(o + 4), f(o + 8)];
            nodes.push(H4Node {
                name: c.sid(d.u32_at(e)),
                parent: d.i16_at(e + 4),
                first_child: d.i16_at(e + 6),
                next_sibling: d.i16_at(e + 8),
                translation: v3(12),
                rotation: [f(24), f(28), f(32), f(36)],
                scale: f(40),
                inverse: [v3(44), v3(56), v3(68), v3(80)],
            });
        }
    }
    Ok(H4Model { tag: mode_tag, name, meshes, mesh_info, geometry, materials, regions, nodes, lod0_meshes })
}

/// Vertex-buffer types whose first 36 bytes match the BSP's rigid layout (checked on samples of
/// each: positions decode inside the model bounds, word @12 zero, bytes 26/27 zero).
/// #h2a: type **1** joins the list. Only one mesh in the whole H2A corpus uses it
/// (`ca_zanzibar` BSP mesh 1764, a cluster mesh) and its bytes satisfy the same tail exactly
/// (blend @12 in 0..1, u16 @26 = 0, unit snorm16x3 normal and tangent, handedness +-1 - all
/// 100 %); its POSITION is the one difference: absolute world space, not normalized, which is
/// why its compression-bounds element is all zeros (`Bounds::is_degenerate`).
pub fn vertex_kind_supported(kind: i16, stride: i16) -> bool {
    stride >= 36 && matches!(kind, 1 | 2 | 3 | 5 | 18 | 47 | 48)
}

/// Decode one mesh of a geometry resource: vertices (pos normalized to `m.bounds`, uv, normal)
/// and a u32 triangle LIST. `index_type` 5 (strips) is converted to a list per part with
/// alternating winding, degenerate triangles dropped, and the parts' index ranges rewritten.
/// Generalises geometry::decode_mesh (which takes an H4Bsp); the caller will fold it back.
pub fn decode_mesh_generic(c: &H4Cache, g: &GeometryResource, m: &MeshDef, index_type: u8) -> Result<Option<DecodedMesh>> {
    if !m.has_geometry() { return Ok(None); }
    let e = c.resources.get(g.entry).ok_or_else(|| anyhow!("geometry entry"))?;
    let ib = g.ibs.get(m.ib_index() as usize).ok_or_else(|| anyhow!("ib {} out of range", m.ib_index()))?;
    let strip = match (ib.fmt, index_type) {
        (3, _) => false,
        (5, _) | (_, 5) => true,
        (f, _) => bail!("unsupported index format {f}"),
    };
    let iraw = c.stream_bytes(e, ib.addr, ib.size as usize)?;
    // m.vb[0] is a RAW index into the resource's vertex buffers. The one exception in the whole
    // corpus is ca_zanzibar's BSP, where a stray stride-4 buffer sits inside the position run and
    // shifts 36 meshes by one; there the raw buffer either is too narrow to hold positions or is
    // too short for this mesh's indices, and the position-capable-only reading is correct. See
    // GeometryResource::pos_vbs. Every other mesh keeps the raw buffer, unchanged. #h2a
    let max_ix = (0..iraw.len() / 2).map(|k| iraw.u16_at(k * 2) as u32).max().unwrap_or(0);
    let raw_vb = g.vertex_buffer(m.vb[0]);
    let usable = |v: &VbInfo| v.stride >= 36 && max_ix < v.count;
    let vb = match raw_vb {
        Some(v) if usable(v) => v,
        other => match g.vertex_buffer_fallback(m.vb[0]).filter(|v| usable(v)) {
            Some(v) => v,
            None => other.ok_or_else(|| anyhow!("vb {} out of range", m.vb[0]))?,
        },
    };
    if !vertex_kind_supported(vb.kind, vb.stride) {
        bail!("unsupported vertex type {} stride {}", vb.kind, vb.stride);
    }
    let vraw = c.stream_bytes(e, vb.addr, vb.size as usize)?;
    let b = &m.bounds;
    // #h2a: an all-zero compression-bounds element means the vertex buffer is UNCOMPRESSED -
    // decoding through a zero range would collapse the whole mesh onto pos_min. Cannot change
    // Halo 4 output: no Halo 4 mesh with geometry has degenerate bounds (its cluster meshes carry
    // no vertex buffer at all), and where the extents are real this is the identical arithmetic.
    let raw_pos = b.is_degenerate();
    let stride = vb.stride as usize;
    let mut verts = Vec::with_capacity(vb.count as usize);
    for k in 0..vb.count as usize {
        let o = k * stride;
        let f = |i: usize| vraw.f32_at(o + i * 4);
        let pos = if raw_pos {
            [f(0), f(1), f(2)]
        } else {
            [
                b.pos_min[0] + f(0) * (b.pos_max[0] - b.pos_min[0]),
                b.pos_min[1] + f(1) * (b.pos_max[1] - b.pos_min[1]),
                b.pos_min[2] + f(2) * (b.pos_max[2] - b.pos_min[2]),
            ]
        };
        let u = vraw.u16_at(o + 16) as f32 / 65535.0;
        let v = vraw.u16_at(o + 18) as f32 / 65535.0;
        let uv = if raw_pos { [u, v] } else { [b.uv_min[0] + u * (b.uv_max[0] - b.uv_min[0]), b.uv_min[1] + v * (b.uv_max[1] - b.uv_min[1])] };
        // same tail as the BSP's type 2 (geometry.rs): normal snorm16x3 @20, tangent snorm16x3
        // @28 + handedness @34, blend f32 @12 (0 on every model format except type 37, rejected)
        let normal = [sn16(vraw.i16_at(o + 20)), sn16(vraw.i16_at(o + 22)), sn16(vraw.i16_at(o + 24))];
        let tangent = [sn16(vraw.i16_at(o + 28)), sn16(vraw.i16_at(o + 30)), sn16(vraw.i16_at(o + 32)), sn16(vraw.i16_at(o + 34))];
        verts.push(H4Vertex { pos, normal, uv, tangent, blend: f(3) });
    }
    let raw_idx: Vec<u32> = (0..iraw.len() / 2).map(|k| iraw.u16_at(k * 2) as u32).collect();
    let nv = verts.len() as u32;
    if !strip {
        return Ok(Some(DecodedMesh { verts, indices: raw_idx, parts: m.parts.clone() }));
    }
    let mut indices = Vec::with_capacity(raw_idx.len() * 3);
    let mut parts = Vec::with_capacity(m.parts.len());
    for p in &m.parts {
        let start = indices.len() as u32;
        let s = p.index_start as usize;
        let n = p.index_count as usize;
        if s + n <= raw_idx.len() && n >= 3 {
            for j in 0..n - 2 {
                let (mut a, mut bb, cc) = (raw_idx[s + j], raw_idx[s + j + 1], raw_idx[s + j + 2]);
                if a == bb || bb == cc || a == cc { continue; }
                if a >= nv || bb >= nv || cc >= nv { continue; }
                if j & 1 == 1 { std::mem::swap(&mut a, &mut bb); }
                indices.extend_from_slice(&[a, bb, cc]);
            }
        }
        parts.push(Part { material: p.material, index_start: start, index_count: indices.len() as u32 - start });
    }
    Ok(Some(DecodedMesh { verts, indices, parts }))
}

/// Decode mesh `mesh_idx` of a loaded model (None when the mesh carries no geometry).
pub fn decode_model_mesh(c: &H4Cache, model: &H4Model, mesh_idx: usize) -> Result<Option<DecodedMesh>> {
    let m = model.meshes.get(mesh_idx).ok_or_else(|| anyhow!("mesh {mesh_idx} out of range"))?;
    let it = model.mesh_info.get(mesh_idx).map(|i| i.index_type).unwrap_or(3);
    decode_mesh_generic(c, &model.geometry, m, it)
}

// ---------------------------------------------------------------------------------------------
// obje / hlmt model variants, default change colours, multiplayer-object defaults
// ---------------------------------------------------------------------------------------------
//
// Engine (halo4.dll, IDA on the shipped build; docs/halo4_forge_palette.md section 3):
//   object_get_variant_index  sub_1805D912C: sid 0 -> obje +0x60 (default model variant sid),
//                             then model_get_variant_index(hlmt datum @ obje +0x70, sid), falling
//                             back to the default sid when the named variant is missing.
//   model_get_variant_index   sub_180279C88: hlmt +0xC4 variants block, 0x6C-byte elements, name
//                             sid @0; sid 0 -> 0 when the block is non-empty, else -1.
// Variant element (Assembly Halo4MCC hlmt.xml, cross-checked on the Ravine palette objects):
//   +0x04 i8[32] runtime model region index per render-model region (-1 = not in this variant),
//   +0x24 regions block, 0x18 B: name @0, rm region i8 @4, parent variant i16 @6, permutations
//   block @8 (0x24 B: name @0, rm permutation i8 @4, probability f32 @8).
// obje +0x148 change colours block (0x18 B): initial permutations block @0 (0x20 B: weight f32
//   @0, lower rgb @4, upper rgb @0x10, variant name sid @0x1C); slot 0 primary, 1 secondary ...
// obje +0x154 multiplayer object block (0xCC B, one element): engine flags u8 @0, type u8 @1,
//   teleporter passability u8 @2, spawn timer type u8 @3, boundary width/radius f32 @4, box
//   length @8, +height @0xC, -height @0x10, shape u8 @0x14, default spawn time i16 @0x18,
//   abandonment time i16 @0x1A, flags u16 @0x1C (bit 1 = phased physics in Forge).

pub const OFF_OBJE_DEFAULT_VARIANT: usize = 0x60;
pub const OFF_OBJE_CHANGE_COLORS: usize = 0x148;
pub const OFF_OBJE_MP_OBJECT: usize = 0x154;
pub const OFF_HLMT_VARIANTS: usize = 0xC4;
pub const HLMT_VARIANT_ELEM: usize = 0x6C;
pub const HLMT_VREGION_ELEM: usize = 0x18;
pub const HLMT_VPERM_ELEM: usize = 0x24;
pub const CHANGE_COLOR_ELEM: usize = 0x18;
pub const CHANGE_COLOR_PERM_ELEM: usize = 0x20;
pub const MP_OBJECT_ELEM: usize = 0xCC;

/// obje -> hlmt (+0x64).
pub fn hlmt_of(c: &H4Cache, obje_tag: usize) -> Option<usize> {
    let om = c.tag_meta(obje_tag)?;
    c.tag_ref_of(om + OFF_OBJE_HLMT, b"hlmt")
}

/// obje +0x60: the object's default model variant string id (0 = unauthored -> hlmt variant 0).
pub fn default_variant_sid(c: &H4Cache, obje_tag: usize) -> u32 {
    match c.tag_meta(obje_tag) {
        Some(om) => match c.data().u32_at(om + OFF_OBJE_DEFAULT_VARIANT) { 0xFFFF_FFFF => 0, v => v },
        None => 0,
    }
}

/// Names (string ids) of the hlmt's model variants in block order.
pub fn hlmt_variant_sids(c: &H4Cache, hlmt_tag: usize) -> Vec<u32> {
    let Some(hm) = c.tag_meta(hlmt_tag) else { return Vec::new() };
    let Some((n, o)) = c.block(hm + OFF_HLMT_VARIANTS) else { return Vec::new() };
    (0..n.min(256)).map(|i| c.data().u32_at(o + i * HLMT_VARIANT_ELEM)).collect()
}

/// Port of `model_get_variant_index` (sub_180279C88): the index of the variant named `sid`;
/// sid 0 -> variant 0 when the block is non-empty; None when missing.
pub fn hlmt_variant_index(c: &H4Cache, hlmt_tag: usize, sid: u32) -> Option<usize> {
    let sids = hlmt_variant_sids(c, hlmt_tag);
    if sids.is_empty() { return None; }
    if sid == 0 { return Some(0); }
    sids.iter().position(|&s| s == sid)
}

/// Port of `object_get_variant_index` (sub_1805D912C): the hlmt variant an object spawns in
/// when `sid` is requested (0 = the object's default). None = the model has no variants.
pub fn object_variant_index(c: &H4Cache, obje_tag: usize, sid: u32) -> Option<usize> {
    let hlmt = hlmt_of(c, obje_tag)?;
    let dsid = default_variant_sid(c, obje_tag);
    let want = if sid == 0 { dsid } else { sid };
    hlmt_variant_index(c, hlmt, want).or_else(|| if want != dsid { hlmt_variant_index(c, hlmt, dsid) } else { None })
}

/// The render-model meshes an object shows in hlmt variant `variant` (None = the engine
/// default): one permutation per region - the variant's first permutation entry for the
/// region (rm permutation -1 = the region is hidden), permutation 0 for regions the variant
/// does not list - the rule ported from the Reach path (scene.rs `resting_section_mask`, same
/// engine family). Falls back to `model.lod0_meshes` when the object has no hlmt variants.
/// The variant/permutation picture is not confirmed against the game for Halo 4 (visually
/// plausible: the Warthog gauss / rocket palette variants show their turrets).
pub fn variant_meshes(c: &H4Cache, obje_tag: usize, model: &H4Model, variant: Option<usize>) -> Vec<usize> {
    let d = c.data();
    let fallback = || model.lod0_meshes.clone();
    let Some(hlmt) = hlmt_of(c, obje_tag) else { return fallback() };
    let Some(hm) = c.tag_meta(hlmt) else { return fallback() };
    let Some((nv, vo)) = c.block(hm + OFF_HLMT_VARIANTS) else { return fallback() };
    let vi = variant.unwrap_or(0);
    if vi >= nv { return fallback(); }
    let ve = vo + vi * HLMT_VARIANT_ELEM;
    // region name -> (rm permutation index or -1 hidden) from the variant's region list
    let mut choice: HashMap<u32, i32> = HashMap::new();
    if let Some((nr, ro)) = c.block(ve + 0x24) {
        for r in 0..nr.min(1024) {
            let re = ro + r * HLMT_VREGION_ELEM;
            let name = d.u32_at(re);
            let perm = match c.block(re + 8) {
                Some((np, po)) if np > 0 => d.u8_at(po) as i8 as i32,
                _ => 0,
            };
            choice.entry(name).or_insert(perm);
        }
    }
    let mut out = Vec::new();
    let mut push_perm = |p: &H4Permutation| {
        for mi in p.mesh_index.max(0)..p.mesh_index.max(0) + p.mesh_count.max(0) {
            let mi = mi as usize;
            if mi < model.meshes.len() && !out.contains(&mi) { out.push(mi); }
        }
    };
    for r in &model.regions {
        let pick = match choice.get(&r.sid) {
            Some(&p) if p < 0 => continue,
            Some(&p) => p as usize,
            None => 0,
        };
        if let Some(p) = r.permutations.get(pick).or_else(|| r.permutations.first()) { push_perm(p); }
    }
    out
}

/// The object's AUTHORED default change colours `[primary, secondary, tertiary, quaternary]`
/// (linear RGB, white when unauthored), the Reach rule ported to the Halo 4 offsets: per slot,
/// the midpoint of the first initial permutation naming the object's variant (`variant_sid`,
/// 0 = the default variant sid), else the first unnamed one, else permutation 0.
/// These are what a SCENARIO placement draws with; a map-variant / Forge placement has its
/// primary overwritten by the team / colour rule (`h4::tint::primary_change_color`). #h4-veh
pub fn default_change_colors(c: &H4Cache, obje_tag: usize, variant_sid: u32) -> [[f32; 3]; 4] {
    let white = [[1.0f32; 3]; 4];
    let d = c.data();
    let Some(om) = c.tag_meta(obje_tag) else { return white };
    let Some((n, o)) = c.block(om + OFF_OBJE_CHANGE_COLORS) else { return white };
    let vsid = if variant_sid == 0 { default_variant_sid(c, obje_tag) } else { variant_sid };
    let mut out = white;
    for slot in 0..n.min(4) {
        let Some((np, po)) = c.block(o + slot * CHANGE_COLOR_ELEM) else { continue };
        let perms: Vec<([f32; 3], u32)> = (0..np.min(64)).filter_map(|k| {
            let e = po + k * CHANGE_COLOR_PERM_ELEM;
            let f = |x: usize| d.f32_at(e + x);
            let lo = [f(4), f(8), f(12)];
            let hi = [f(16), f(20), f(24)];
            let ok = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
            if !lo.iter().chain(hi.iter()).all(|v| ok(*v)) { return None; }
            Some(([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5], d.u32_at(e + 0x1C)))
        }).collect();
        let pick = perms.iter().find(|(_, s)| *s == vsid && vsid != 0)
            .or_else(|| perms.iter().find(|(_, s)| *s == 0))
            .or_else(|| perms.first());
        if let Some((col, _)) = pick { out[slot] = *col; }
    }
    out
}

/// obje +0x154 multiplayer object block: what a Forge placement of this object defaults to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MpObjectDefaults {
    pub engine_flags: u8,
    /// Multiplayer object type (the variant record's 6-bit `object_type`; see mvar.rs).
    pub object_type: u8,
    pub teleporter_passability: u8,
    pub spawn_timer_type: u8,
    /// Boundary width / radius, box length, +height, -height (world units).
    pub boundary: [f32; 4],
    /// 0 none, 1 sphere, 2 cylinder, 3 box.
    pub shape: u8,
    pub spawn_time: i16,
    pub abandonment_time: i16,
    /// bit 0 only visible in editor, bit 1 phased physics in Forge, bit 2 valid initial player
    /// spawn, bit 3 fixed boundary orientation, bit 5 inherit owning team colour.
    pub flags: u16,
}

impl MpObjectDefaults {
    pub fn phased_in_forge(&self) -> bool { self.flags & 2 != 0 }
}

/// The first element of the object's multiplayer object block (None when unauthored).
pub fn mp_object_defaults(c: &H4Cache, obje_tag: usize) -> Option<MpObjectDefaults> {
    let d = c.data();
    let om = c.tag_meta(obje_tag)?;
    let (n, o) = c.block(om + OFF_OBJE_MP_OBJECT)?;
    if n == 0 { return None; }
    Some(MpObjectDefaults {
        engine_flags: d.u8_at(o),
        object_type: d.u8_at(o + 1),
        teleporter_passability: d.u8_at(o + 2),
        spawn_timer_type: d.u8_at(o + 3),
        boundary: [d.f32_at(o + 4), d.f32_at(o + 8), d.f32_at(o + 12), d.f32_at(o + 16)],
        shape: d.u8_at(o + 0x14),
        spawn_time: d.i16_at(o + 0x18),
        abandonment_time: d.i16_at(o + 0x1A),
        flags: d.u16_at(o + 0x1C),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;
    use crate::h4::geometry::load_bsp;
    use std::collections::BTreeMap;

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Union of every BSP's cluster bounds, padded (objects may hang slightly outside).
    fn world_bounds(c: &H4Cache) -> (Vec3, Vec3) {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for t in c.find_tags(b"sbsp") {
            let bsp = load_bsp(c, t).unwrap();
            for (a, b) in &bsp.cluster_bounds { mn = mn.min(Vec3::from(*a)); mx = mx.max(Vec3::from(*b)); }
        }
        let pad = (mx - mn) * 0.1 + 5.0;
        (mn - pad, mx + pad)
    }

    fn histogram(ps: &[H4Placement]) -> BTreeMap<String, usize> {
        let mut h = BTreeMap::new();
        for p in ps { *h.entry(p.class_str()).or_insert(0) += 1; }
        h
    }

    fn print_first(c: &H4Cache, ps: &[H4Placement]) {
        for p in ps.iter().take(5) {
            eprintln!("  {} '{}' {} pos {:?} rot {:?} scale {} flags {:#x}", p.class_str(), p.name, c.tag_name(p.palette_tag), p.pos, p.rot, p.scale, p.flags);
        }
    }

    /// Placements resolve to models that load + decode (>= 80 % of distinct renderable palette tags).
    fn check_models(c: &H4Cache, ps: &[H4Placement]) -> (usize, usize) {
        let mut tags: Vec<usize> = ps.iter().filter(|p| matches!(&p.class, b"scen" | b"bloc" | b"vehi" | b"weap" | b"eqip" | b"mach" | b"ctrl" | b"bipd")).map(|p| p.palette_tag).collect();
        tags.sort();
        tags.dedup();
        let mut ok = 0;
        for &t in &tags {
            let Some(mode) = model_of(c, t) else { eprintln!("  no mode for {}", c.tag_name(t)); continue };
            let model = match load_model(c, mode) { Ok(m) => m, Err(e) => { eprintln!("  load_model {} failed: {e}", c.tag_name(mode)); continue } };
            let mut decoded = false;
            for &mi in &model.lod0_meshes {
                match decode_model_mesh(c, &model, mi) {
                    Ok(Some(dm)) if !dm.verts.is_empty() && !dm.indices.is_empty() => {
                        assert!(dm.indices.iter().all(|&i| (i as usize) < dm.verts.len()), "{}: index range", model.name);
                        assert!(dm.indices.len() % 3 == 0, "{}: triangle list", model.name);
                        let b = &model.meshes[mi].bounds;
                        assert!(dm.verts.iter().all(|v| (0..3).all(|k| v.pos[k] >= b.pos_min[k] - 1e-3 && v.pos[k] <= b.pos_max[k] + 1e-3)), "{}: verts inside bounds", model.name);
                        for p in &dm.parts {
                            assert!((p.index_start + p.index_count) as usize <= dm.indices.len(), "{}: part range", model.name);
                            assert!(p.material < 0 || (p.material as usize) < model.materials.len(), "{}: material index", model.name);
                        }
                        decoded = true;
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("  {} mesh {mi}: {e}", model.name),
                }
            }
            if decoded { ok += 1; } else { eprintln!("  {}: no lod0 mesh decoded (lod0 {:?})", model.name, model.lod0_meshes); }
        }
        (ok, tags.len())
    }

    #[test]
    fn wraparound_placements_and_models() {
        let Some(c) = open("wraparound.map") else { return };
        let ps = load_placements(&c);
        let h = histogram(&ps);
        eprintln!("wraparound placements {} {:?}", ps.len(), h);
        print_first(&c, &ps);
        // scnr blocks: scen 230 / mach 25 / ssce 28 / bloc 29 elements, some with palette -1
        assert_eq!(h["scen"], 230);
        assert_eq!(h["mach"], 25);
        assert_eq!(h["ssce"], 16);
        assert_eq!(h["bloc"], 29);
        assert_eq!(ps.len(), 300);
        assert!(ps.iter().all(|p| c.tag_class(p.palette_tag) == Some(p.class)));
        let (mn, mx) = world_bounds(&c);
        assert!(ps.iter().all(|p| { let v = Vec3::from(p.pos); v.cmpge(mn).all() && v.cmple(mx).all() }), "positions inside {mn:?}..{mx:?}");
        assert!(ps.iter().all(|p| p.rot.iter().all(|r| r.abs() <= std::f32::consts::TAU) && p.scale > 0.0));
        assert!(ps.iter().any(|p| p.name == "lightbeam01"));
        // an identity rotation yields the identity basis
        let id = H4Placement { class: *b"scen", palette_tag: 0, name: String::new(), pos: [1.0, 2.0, 3.0], rot: [0.0; 3], scale: 1.0, flags: 0, basis: None, mp: None, model_variant: None };
        let m = placement_matrix(&id);
        assert!((m.transform_point3(Vec3::X) - Vec3::new(2.0, 2.0, 3.0)).length() < 1e-6);
        let (ok, n) = check_models(&c, &ps);
        eprintln!("wraparound models decoded {ok}/{n}");
        assert!(ok * 100 >= n * 80, "models decoded {ok}/{n}");
    }

    #[test]
    fn ravine_placements_and_models() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        let ps = load_placements(&c);
        let h = histogram(&ps);
        eprintln!("ravine placements {} {:?}", ps.len(), h);
        print_first(&c, &ps);
        assert_eq!(h["scen"], 23);
        assert_eq!(h["ssce"], 58);
        assert_eq!(ps.len(), 81);
        let (mn, mx) = world_bounds(&c);
        assert!(ps.iter().all(|p| { let v = Vec3::from(p.pos); v.cmpge(mn).all() && v.cmple(mx).all() }), "positions inside {mn:?}..{mx:?}");
        let (ok, n) = check_models(&c, &ps);
        eprintln!("ravine models decoded {ok}/{n}");
        assert!(ok * 100 >= n * 80, "models decoded {ok}/{n}");
    }

    /// Campaign: every table class shows up; model pages live in campaign.map (cache index 2,
    /// not opened by cache.rs yet) so only the tag chain is checked here.
    #[test]
    fn m10_placements_resolve() {
        let Some(c) = open("m10_crash.map") else { return };
        let ps = load_placements(&c);
        let h = histogram(&ps);
        eprintln!("m10_crash placements {} {:?}", ps.len(), h);
        print_first(&c, &ps);
        for cls in ["scen", "vehi", "eqip", "weap", "mach", "ctrl", "ssce", "bloc"] { assert!(h.get(cls).copied().unwrap_or(0) > 0, "{cls}"); }
        // scnr blocks: mach 133 elements (9 with palette -1 / null entry), ctrl 18, ssce 419, bloc 373
        assert_eq!(h["mach"], 124);
        assert_eq!(h["ctrl"], 18);
        assert_eq!(h["ssce"], 419);
        assert_eq!(h["bloc"], 373);
        let mut tags: Vec<usize> = ps.iter().filter(|p| &p.class != b"ssce").map(|p| p.palette_tag).collect();
        tags.sort();
        tags.dedup();
        let modes: Vec<usize> = tags.iter().filter_map(|&t| model_of(&c, t)).collect();
        eprintln!("m10_crash palette tags with a mode: {}/{}", modes.len(), tags.len());
        assert!(modes.len() * 100 >= tags.len() * 80);
        // struct + geometry-definition parse (no page reads) works on a campaign cache, whose
        // resource-type indices differ (kinds are matched by name)
        let mut loaded = 0;
        let mut lod0 = 0;
        for &m in &modes {
            match load_model(&c, m) {
                Ok(model) => { loaded += 1; if !model.lod0_meshes.is_empty() && model.lod0_meshes.iter().all(|&i| model.meshes[i].has_geometry()) { lod0 += 1; } }
                Err(e) => eprintln!("  load_model {} failed: {e}", c.tag_name(m)),
            }
        }
        eprintln!("m10_crash models loaded {loaded}/{} with lod0 geometry {lod0}", modes.len());
        assert!(loaded * 100 >= modes.len() * 80 && lod0 * 100 >= modes.len() * 70);
    }

    /// Mode layout numbers measured by the probes: mongoose = 19 nodes / 33 meshes, wheels
    /// decode at their node's absolute position (model space), strips convert to lists.
    #[test]
    fn wraparound_model_layout() {
        let Some(c) = open("wraparound.map") else { return };
        let modes = c.find_tags(b"mode");
        assert_eq!(modes.len(), 215);
        let with_geo = modes.iter().filter(|&&t| load_model(&c, t).is_ok()).count();
        assert_eq!(with_geo, 211);
        let mongoose = modes.iter().copied().find(|&t| c.tag_name(t).ends_with("storm_mongoose\\storm_mongoose")).expect("mongoose");
        let m = load_model(&c, mongoose).unwrap();
        assert_eq!((m.nodes.len(), m.meshes.len()), (19, 33));
        assert!(m.nodes.iter().all(|n| n.parent < m.nodes.len() as i16 && !n.name.is_empty() && (n.scale - 1.0).abs() < 1e-3));
        assert!(m.materials.iter().all(|x| x.is_some()));
        let wheel = m.nodes.iter().position(|n| n.name == "b_wheel_back_left").unwrap();
        let mi = (0..m.meshes.len()).find(|&i| m.mesh_info[i].node as usize == wheel).unwrap();
        let dm = decode_model_mesh(&c, &m, mi).unwrap().unwrap();
        let cen = dm.verts.iter().fold(Vec3::ZERO, |a, v| a + Vec3::from(v.pos)) / dm.verts.len() as f32;
        // absolute node position: chassis (-0.05, 0, 0.30) + wheel (-0.122, 0.109, -0.081) ... measured (-0.386, 0.25, 0.14)
        assert!((cen - Vec3::new(-0.386, 0.276, 0.140)).length() < 0.03, "wheel centroid {cen}");
        // spawn point: strip index buffers, single node at z = 0.5, verts in model space z 0..0.97
        let sp = modes.iter().copied().find(|&t| c.tag_name(t).ends_with("mp_spawn_point\\mp_spawn_point")).expect("spawn point");
        let m = load_model(&c, sp).unwrap();
        assert_eq!(m.mesh_info[0].index_type, 5);
        assert_eq!(m.regions.len(), 2);
        assert_eq!(m.lod0_meshes, vec![0, 1]);
        let dm = decode_model_mesh(&c, &m, 0).unwrap().unwrap();
        assert!(dm.indices.len() % 3 == 0 && dm.indices.len() >= 3 * 300);
        let mut agree = 0usize;
        for t in dm.indices.chunks_exact(3) {
            let p = |i: u32| Vec3::from(dm.verts[i as usize].pos);
            let fnrm = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0])).normalize_or_zero();
            let vn = (Vec3::from(dm.verts[t[0] as usize].normal) + Vec3::from(dm.verts[t[1] as usize].normal) + Vec3::from(dm.verts[t[2] as usize].normal)).normalize_or_zero();
            if fnrm.dot(vn) > 0.5 { agree += 1; }
        }
        let n = dm.indices.len() / 3;
        assert!(agree * 100 / n >= 70, "strip facing agreement {agree}/{n}");
    }
}
