//! Halo 4 / Halo 2 Anniversary collision (`coll`) and physics (`phmo`) models - pure Rust. #h4-phys
//!
//! The editor draws two selected-object overlays (View > Overlays > "Collision (selected)" /
//! "Physics (selected)") and the always-on hidden-block hull. On Reach those come from the native
//! C++ walkers (`native/MccMapStudioDLL/CollisionModelWalker.cpp` / `PhysicsModelWalker.cpp`);
//! Halo 4 has no native parser, so this module reads the same two tags out of the Halo 4 cache.
//!
//! RESOLUTION CHAIN (verified on every `hlmt` of all 50 shipped Halo 4 caches - 16 039 tags,
//! 14 579 with a `coll`, 14 657 with a `phmo`, ZERO class mismatches):
//!     obje +0x64 -> hlmt ; hlmt +0x00 -> mode, **hlmt +0x10 -> coll**, **hlmt +0x30 -> phmo**
//!
//! ## `coll` (collision_model, baseSize 0x58)
//! Halo 4 moved the collision BSP OUT of the tag and into a `collision_model_resource` (zone kind
//! whose type name is exactly that). The tag keeps only the index:
//!   +0x14 materials (4 B)   +0x20 regions (0x10 B: name sid @0, permutations @4)
//!   permutation (0x2C B): name sid @0, **i16 resource BSP offset @4, i16 resource BSP count @6**
//!   +0x44 nodes (0xC B: name sid @0, ... parent @6)   **+0x50 u32 zone asset datum**
//! The resource's definition data is `[N x 0x70 collision BSP][12 B tagblock header]`, the header
//! living at `def_addr & 0x0FFF_FFFF` (count + a fixup pointing at definition offset 0). Each
//! 0x70-byte BSP element is **byte-for-byte the Reach collision BSP**, with its block pointers
//! fixed up into the resource's PRIMARY page stream (fixup nibble 4):
//!   i16 node index @0x00 | bsp3d nodes @0x04 (8 B) | planes @0x1C (0x10) | leaves @0x28 (8)
//!   bsp2d refs @0x34 (4) | bsp2d nodes @0x40 (0x10) | **surfaces @0x4C (0x0C)**
//!   **edges @0x58 (0x0C)** | **vertices @0x64 (0x10)**
//!   surface: plane u16 @0, first edge u16 @2, material i16 @4, breakable set i16 @6,
//!            breakable i16 @8, flags u8 @0x0A (bit1 invisible, bit4 invalid), best-plane vtx @0x0B
//!   edge:    start i16 @0, end i16 @2, forward i16 @4, reverse i16 @6, left surface i16 @8,
//!            right surface i16 @0x0A
//!   vertex:  point f32x3 @0, first edge i16 @0x0C, sink i16 @0x0E
//! EVIDENCE: walking every surface's half-edge loop over all 50 caches visits **2 179 109**
//! surfaces; 10 loops fail to close and **0** produce a polygon that is off its own stored plane
//! by more than 0.02 wu. Per-permutation `(offset, count)` ranges tile the resource's BSP array
//! exactly on all 13 951 resolvable resources (`sum(count) == N`, every range in bounds).
//!
//! ## `phmo` (physics_model, baseSize 0x1A8)
//! Still fully in the tag. Top-level block offsets are Reach's; several ELEMENT SIZES grew:
//!   rigid bodies @0x5C (**0xE0**, Reach 0xD0) | materials @0x68 | spheres @0x74 (0xB0)
//!   multi spheres @0x80 (0xD0) | pills @0x8C (0x70) | boxes @0x98 (0xE0) | triangles @0xA4 (0x90)
//!   polyhedra @0xB0 (**0xA0**, Reach 0xB0) | polyhedron four vectors @0xBC (0x30)
//!   lists @0xE0 (0x90) | list shapes @0xEC (0x20) | MOPPs @0xF8 (0x80) | nodes @0x13C (0xC)
//!   rigid body: i16 node index @0x00, **enum16 shape type @0xB8, i16 shape index @0xBA**
//!               (Reach: 0xA8 / 0xAA)
//!   sphere: radius @0x40, translation @0xA0 | pill: radius @0x40, bottom @0x50, top @0x60
//!   box: half extents @0x50 (rotation i/j/k @0xA0/0xB0/0xC0 and translation @0xD0 are the Reach
//!        fields, but in Halo 4 / H2A they are NOT serialized - see `emit_box`: a box is
//!        axis-aligned at its rigid body's node origin)
//!   triangle: points a/b/c @0x50/0x60/0x70
//!   multi sphere: count @0x40, spheres (xyz + radius w) @0x50 + i*0x10
//!   polyhedron: four-vectors size @0x78, vertex count @0x80, AABB half extents @0x50, centre @0x60
//!   four vectors: 4 vertices SOA - x0..x3 @0x00, y0..y3 @0x10, z0..z3 @0x20
//!   list: child shapes size @0x38 (children are consumed sequentially out of `list shapes`)
//!   list shape / MOPP child: shape type @0x00 / @0x58, shape index @0x02 / @0x5A
//! Shape-type enum (identical to Reach): 0 sphere, 1 pill, 2 box, 3 triangle, 4 polyhedron,
//! 5 multi sphere, 6 phantom (NO geometry - skipped), 0xE list, 0xF MOPP.
//! EVIDENCE: 21 874 rigid bodies over all 50 caches - types {polyhedron 16 597, list 2 340,
//! MOPP 779, pill 1 735, phantom 85, box 314, sphere 24}; every shape index is inside its block
//! except 23 null (-1) bodies on `storm_pelican` and the 85 phantoms (which index the Phantoms
//! block instead). The polyhedron four-vector pool is consumed SEQUENTIALLY in polyhedron order:
//! for all **48 148** polyhedra `four_vectors_size == ceil(vertex_count / 4)`, the sizes sum to
//! the pool length, and the AABB the unpacked vertices produce matches the polyhedron's OWN
//! stored centre / half extents to <= 0.0178 wu (the Havok convex radius, tag default 0.016).
//!
//! ## Node space
//! A collision BSP and a rigid body are in the space of the NODE they name, not model space. The
//! `coll` / `phmo` / `mode` node lists are the same list (identical names at identical indices on
//! every `hlmt` of every cache - 0 mismatches), so the transform comes from `mode` +0x30:
//! rows (inverse forward, inverse left, inverse up) @0x2C/0x38/0x44 and inverse position @0x50
//! give the node->model matrix as **p_model = B * (l - inv_position)**. Checked against
//! accumulating the default translation (@0xC) / rotation (@0x18) down the parent chain: 9 535
//! node/point pairs agree to 1.7e-6. Without it the Warthog's main hull lands rotated 120 deg
//! (its `b_chassis` node is a (-.5,-.5,-.5,.5) quaternion) and every marker model sits 0.5 wu low.
//!
//! #h2a: Assembly's `Halo2AMCC/coll.xml` and `phmo.xml` are byte-identical to the Halo 4 ones, and
//! H2A shares `h4::cache`, so everything here is engine-generic; `tests::h2a_*` parse the H2A tags.

use glam::{Mat4, Vec3};

use super::cache::{ByteRead, H4Cache};
use super::objects::{hlmt_of, model_of, OFF_MODE_NODES, NODE_ELEM};

// ---- hlmt ------------------------------------------------------------------------------------

/// hlmt +0x10: the `coll` reference.
pub const OFF_HLMT_COLL: usize = 0x10;
/// hlmt +0x30: the `phmo` reference.
pub const OFF_HLMT_PHMO: usize = 0x30;

// ---- coll ------------------------------------------------------------------------------------

pub const OFF_COLL_REGIONS: usize = 0x20;
pub const COLL_REGION_ELEM: usize = 0x10;
pub const OFF_COLL_REGION_PERMS: usize = 0x04;
pub const COLL_PERM_ELEM: usize = 0x2C;
pub const OFF_COLL_NODES: usize = 0x44;
pub const COLL_NODE_ELEM: usize = 0xC;
/// coll +0x50: the `collision_model_resource` zone asset datum.
pub const OFF_COLL_RESOURCE: usize = 0x50;
/// The zone resource type name holding the collision BSPs.
pub const RES_COLLISION: &str = "collision_model_resource";

/// One collision BSP element inside the resource definition data.
pub const COLL_BSP_ELEM: usize = 0x70;
const OFF_BSP_SURFACES: usize = 0x4C;
const OFF_BSP_EDGES: usize = 0x58;
const OFF_BSP_VERTICES: usize = 0x64;
const SURFACE_ELEM: usize = 0x0C;
const EDGE_ELEM: usize = 0x0C;
const VERTEX_ELEM: usize = 0x10;
/// Surface flags @0x0A: bit 1 = invisible (skipped, as the Reach walker does), bit 4 = invalid.
const SURF_INVISIBLE: u8 = 0x02;
const SURF_INVALID: u8 = 0x10;
/// A single surface's half-edge loop is bounded (corrupt data guard).
const MAX_LOOP: usize = 256;

// ---- phmo ------------------------------------------------------------------------------------

pub const OFF_PHMO_RIGID_BODIES: usize = 0x5C;
pub const PHMO_RB_ELEM: usize = 0xE0;
const RB_NODE: usize = 0x00;
const RB_SHAPE_TYPE: usize = 0xB8;
const RB_SHAPE_INDEX: usize = 0xBA;

pub const OFF_PHMO_SPHERES: usize = 0x74;
pub const PHMO_SPHERE_ELEM: usize = 0xB0;
pub const OFF_PHMO_MULTI_SPHERES: usize = 0x80;
pub const PHMO_MULTI_SPHERE_ELEM: usize = 0xD0;
pub const OFF_PHMO_PILLS: usize = 0x8C;
pub const PHMO_PILL_ELEM: usize = 0x70;
pub const OFF_PHMO_BOXES: usize = 0x98;
pub const PHMO_BOX_ELEM: usize = 0xE0;
pub const OFF_PHMO_TRIANGLES: usize = 0xA4;
pub const PHMO_TRIANGLE_ELEM: usize = 0x90;
pub const OFF_PHMO_POLYHEDRA: usize = 0xB0;
pub const PHMO_POLYHEDRON_ELEM: usize = 0xA0;
pub const OFF_PHMO_FOUR_VECTORS: usize = 0xBC;
pub const PHMO_FOUR_VECTOR_ELEM: usize = 0x30;
pub const OFF_PHMO_LISTS: usize = 0xE0;
pub const PHMO_LIST_ELEM: usize = 0x90;
pub const OFF_PHMO_LIST_SHAPES: usize = 0xEC;
pub const PHMO_LIST_SHAPE_ELEM: usize = 0x20;
pub const OFF_PHMO_MOPPS: usize = 0xF8;
pub const PHMO_MOPP_ELEM: usize = 0x80;
pub const OFF_PHMO_NODES: usize = 0x13C;
pub const PHMO_NODE_ELEM: usize = 0xC;

const SHAPE_SPHERE: u16 = 0;
const SHAPE_PILL: u16 = 1;
const SHAPE_BOX: u16 = 2;
const SHAPE_TRIANGLE: u16 = 3;
const SHAPE_POLYHEDRON: u16 = 4;
const SHAPE_MULTI_SPHERE: u16 = 5;
const SHAPE_LIST: u16 = 0xE;
const SHAPE_MOPP: u16 = 0xF;

/// How deep a list / MOPP container is followed.
const MAX_SHAPE_DEPTH: usize = 4;
/// Sanity caps (a hull overlay is small; these bound a corrupt pointer).
const MAX_RIGID_BODIES: usize = 4096;
const MAX_POLY_VERTS: usize = 4096;
/// Beyond this a polyhedron falls back to its AABB box instead of the O(n^3) hull scan.
const HULL_BRUTE_LIMIT: usize = 96;
/// Sphere / capsule tessellation (the Reach walker's numbers, so the two overlays match).
const SPHERE_STACKS: usize = 6;
const SPHERE_SLICES: usize = 8;

/// A decoded hull: world-unit model-space triangles.
#[derive(Clone, Debug, Default)]
pub struct HullGeom {
    pub verts: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl HullGeom {
    pub fn is_empty(&self) -> bool { self.verts.len() < 3 || self.indices.len() < 3 }
    pub fn tri_count(&self) -> usize { self.indices.len() / 3 }
    /// Model-space AABB (None when empty).
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for v in &self.verts {
            let v = Vec3::from(*v);
            if !v.is_finite() { continue; }
            mn = mn.min(v);
            mx = mx.max(v);
        }
        (mn.x <= mx.x).then_some((mn, mx))
    }
    /// The largest AABB side (the hidden-block "does this have a volume" test).
    pub fn extent(&self) -> f32 {
        self.bounds().map_or(0.0, |(mn, mx)| (mx - mn).abs().max_element())
    }
    /// The three edges of every triangle, transformed by `m` - the overlay line list.
    pub fn world_edges(&self, m: &Mat4) -> Vec<([f32; 3], [f32; 3])> {
        let n = self.verts.len();
        let mut out = Vec::with_capacity(self.indices.len());
        for t in self.indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if a >= n || b >= n || c >= n { continue; }
            let pa = m.transform_point3(Vec3::from(self.verts[a]));
            let pb = m.transform_point3(Vec3::from(self.verts[b]));
            let pc = m.transform_point3(Vec3::from(self.verts[c]));
            if !(pa.is_finite() && pb.is_finite() && pc.is_finite()) { continue; }
            out.push((pa.to_array(), pb.to_array()));
            out.push((pb.to_array(), pc.to_array()));
            out.push((pc.to_array(), pa.to_array()));
        }
        out
    }
    fn push_tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }
    fn base(&self) -> u32 { self.verts.len() as u32 }
}

// ---- node transforms -------------------------------------------------------------------------

/// `mode` node -> model matrices, index-parallel to the render model's node block (and therefore
/// to the `coll` / `phmo` node blocks). Built from the node's INVERSE absolute frame
/// (`p_model = B * (l - inv_position)`, rows of `B` = inverse forward / left / up).
pub fn model_node_xforms(c: &H4Cache, mode_tag: usize) -> Vec<Mat4> {
    let Some(mm) = c.tag_meta(mode_tag) else { return Vec::new() };
    let Some((n, o)) = c.block(mm + OFF_MODE_NODES) else { return Vec::new() };
    let d = c.data();
    let mut out = Vec::with_capacity(n);
    for i in 0..n.min(1024) {
        let e = o + i * NODE_ELEM;
        let row = |off: usize| Vec3::new(d.f32_at(e + off), d.f32_at(e + off + 4), d.f32_at(e + off + 8));
        let (f, l, u) = (row(0x2C), row(0x38), row(0x44));
        let ip = row(0x50);
        // B has the three inverse basis vectors as ROWS; the translation is -B * inv_position.
        let b = Mat4::from_cols(
            glam::Vec4::new(f.x, l.x, u.x, 0.0),
            glam::Vec4::new(f.y, l.y, u.y, 0.0),
            glam::Vec4::new(f.z, l.z, u.z, 0.0),
            glam::Vec4::W,
        );
        let m = if f.is_finite() && l.is_finite() && u.is_finite() && ip.is_finite() && b.determinant().abs() > 1e-6 {
            Mat4::from_translation(-b.transform_vector3(ip)) * b
        } else {
            Mat4::IDENTITY
        };
        out.push(m);
    }
    out
}

fn node_matrix(nodes: &[Mat4], idx: i16) -> Mat4 {
    if idx < 0 { return Mat4::IDENTITY; }
    nodes.get(idx as usize).copied().unwrap_or(Mat4::IDENTITY)
}

// ---- resolution ------------------------------------------------------------------------------

/// obje -> hlmt -> coll.
pub fn coll_of(c: &H4Cache, obje_tag: usize) -> Option<usize> {
    let hm = c.tag_meta(hlmt_of(c, obje_tag)?)?;
    c.tag_ref_of(hm + OFF_HLMT_COLL, b"coll")
}

/// obje -> hlmt -> phmo.
pub fn phmo_of(c: &H4Cache, obje_tag: usize) -> Option<usize> {
    let hm = c.tag_meta(hlmt_of(c, obje_tag)?)?;
    c.tag_ref_of(hm + OFF_HLMT_PHMO, b"phmo")
}

/// The object's collision hull in MODEL space (`None` = no `coll`, no resource, or no geometry).
pub fn object_collision_hull(c: &H4Cache, obje_tag: usize) -> Option<HullGeom> {
    let coll = coll_of(c, obje_tag)?;
    let nodes = model_of(c, obje_tag).map(|m| model_node_xforms(c, m)).unwrap_or_default();
    let g = coll_geometry(c, coll, &nodes)?;
    (!g.is_empty()).then_some(g)
}

/// The object's physics hull in MODEL space (`None` = no `phmo` or no drawable shape).
pub fn object_physics_hull(c: &H4Cache, obje_tag: usize) -> Option<HullGeom> {
    let phmo = phmo_of(c, obje_tag)?;
    let nodes = model_of(c, obje_tag).map(|m| model_node_xforms(c, m)).unwrap_or_default();
    let g = phmo_geometry(c, phmo, &nodes)?;
    (!g.is_empty()).then_some(g)
}

// ---- coll decode -----------------------------------------------------------------------------

/// Triangulate every collision BSP of a `coll` tag. `nodes` = `model_node_xforms` of the matching
/// render model (empty = treat every BSP as model space).
pub fn coll_geometry(c: &H4Cache, coll_tag: usize, nodes: &[Mat4]) -> Option<HullGeom> {
    if c.tag_class(coll_tag).as_ref() != Some(b"coll") { return None; }
    let cm = c.tag_meta(coll_tag)?;
    let rid = c.data().u32_at(cm + OFF_COLL_RESOURCE);
    let e = c.resource_by_id(rid)?;
    if !c.kind_is(e.kind, RES_COLLISION) { return None; }
    let dd = c.definition(e);
    let base = (e.def_addr & 0x0FFF_FFFF) as usize;
    if base + 12 > dd.len() { return None; }
    let n = dd.i32_at(base);
    if n <= 0 || n > 0x10000 { return None; }
    let n = n as usize;
    let bsp0 = (e.fixup_at((base + 4) as u32)? & 0x0FFF_FFFF) as usize;
    let (page, seg) = c.stream(e, 4).ok()?;
    let sd: &[u8] = &page;
    let mut out = HullGeom::default();
    for b in 0..n {
        let eb = bsp0 + b * COLL_BSP_ELEM;
        if eb + COLL_BSP_ELEM > dd.len() { break; }
        let m = node_matrix(nodes, dd.i16_at(eb));
        // A block inside the resource: count in the definition data, pointer via a page fixup.
        let blk = |off: usize, elem: usize| -> Option<(usize, usize)> {
            let cnt = dd.i32_at(eb + off);
            if cnt <= 0 || cnt > 0x100_0000 { return None; }
            let addr = e.fixup_at((eb + off + 4) as u32)?;
            if addr >> 28 != 4 { return None; }
            let o = seg + (addr & 0x0FFF_FFFF) as usize;
            (o + cnt as usize * elem <= sd.len()).then_some((cnt as usize, o))
        };
        let (Some((nsf, sfo)), Some((ned, edo)), Some((nv, vo))) =
            (blk(OFF_BSP_SURFACES, SURFACE_ELEM), blk(OFF_BSP_EDGES, EDGE_ELEM), blk(OFF_BSP_VERTICES, VERTEX_ELEM))
        else { continue };
        let vbase = out.base();
        for i in 0..nv {
            let p = vo + i * VERTEX_ELEM;
            let v = Vec3::new(sd.f32_at(p), sd.f32_at(p + 4), sd.f32_at(p + 8));
            out.verts.push(m.transform_point3(v).to_array());
        }
        let mut loop_buf = [0u32; MAX_LOOP];
        for s in 0..nsf {
            let p = sfo + s * SURFACE_ELEM;
            let flags = sd.u8_at(p + 0x0A);
            if flags & (SURF_INVISIBLE | SURF_INVALID) != 0 { continue; }
            let first = sd.u16_at(p + 2) as usize;
            if first >= ned { continue; }
            // Half-edge loop: at each edge the surface is either the left or the right one, which
            // picks the vertex to take and the neighbour edge to follow.
            let mut count = 0usize;
            let mut ed = first;
            loop {
                let q = edo + ed * EDGE_ELEM;
                let (sv, ev) = (sd.i16_at(q), sd.i16_at(q + 2));
                let (fwd, rev) = (sd.i16_at(q + 4), sd.i16_at(q + 6));
                let left = sd.i16_at(q + 8);
                let (vtx, next) = if left as usize == s && left >= 0 { (sv, fwd) } else { (ev, rev) };
                if vtx < 0 || vtx as usize >= nv { break; }
                if count < MAX_LOOP { loop_buf[count] = vbase + vtx as u32; count += 1; } else { break; }
                if next < 0 || next as usize >= ned { break; }
                ed = next as usize;
                if ed == first { break; }
            }
            for i in 1..count.saturating_sub(1) {
                out.push_tri(loop_buf[0], loop_buf[i], loop_buf[i + 1]);
            }
        }
    }
    Some(out)
}

// ---- phmo decode -----------------------------------------------------------------------------

/// Cached, validated shape-block bases of one `phmo`.
struct ShapeBlocks {
    sphere: Option<(usize, usize)>,
    multi_sphere: Option<(usize, usize)>,
    pill: Option<(usize, usize)>,
    boxes: Option<(usize, usize)>,
    triangle: Option<(usize, usize)>,
    poly: Option<(usize, usize)>,
    four_vec: Option<(usize, usize)>,
    list: Option<(usize, usize)>,
    list_shape: Option<(usize, usize)>,
    mopp: Option<(usize, usize)>,
    /// Definition-order start of each polyhedron's four-vector run (the pool is consumed
    /// sequentially - see the module header's evidence).
    poly_fv_start: Vec<usize>,
}

/// Triangulate every rigid body of a `phmo` tag. `nodes` = `model_node_xforms` of the matching
/// render model (empty = treat every body as model space).
pub fn phmo_geometry(c: &H4Cache, phmo_tag: usize, nodes: &[Mat4]) -> Option<HullGeom> {
    if c.tag_class(phmo_tag).as_ref() != Some(b"phmo") { return None; }
    let pm = c.tag_meta(phmo_tag)?;
    let d = c.data();
    let blk = |off: usize| c.block(pm + off);
    let mut sb = ShapeBlocks {
        sphere: blk(OFF_PHMO_SPHERES),
        multi_sphere: blk(OFF_PHMO_MULTI_SPHERES),
        pill: blk(OFF_PHMO_PILLS),
        boxes: blk(OFF_PHMO_BOXES),
        triangle: blk(OFF_PHMO_TRIANGLES),
        poly: blk(OFF_PHMO_POLYHEDRA),
        four_vec: blk(OFF_PHMO_FOUR_VECTORS),
        list: blk(OFF_PHMO_LISTS),
        list_shape: blk(OFF_PHMO_LIST_SHAPES),
        mopp: blk(OFF_PHMO_MOPPS),
        poly_fv_start: Vec::new(),
    };
    if let Some((n, o)) = sb.poly {
        let mut cursor = 0usize;
        for i in 0..n {
            sb.poly_fv_start.push(cursor);
            cursor += d.i32_at(o + i * PHMO_POLYHEDRON_ELEM + 0x78).max(0) as usize;
        }
    }
    let (nrb, rbo) = blk(OFF_PHMO_RIGID_BODIES)?;
    let mut out = HullGeom::default();
    for i in 0..nrb.min(MAX_RIGID_BODIES) {
        let e = rbo + i * PHMO_RB_ELEM;
        let m = node_matrix(nodes, d.i16_at(e + RB_NODE));
        emit_shape(c, &sb, d.u16_at(e + RB_SHAPE_TYPE), d.i16_at(e + RB_SHAPE_INDEX), &m, 0, &mut out);
    }
    Some(out)
}

fn emit_shape(c: &H4Cache, sb: &ShapeBlocks, ty: u16, idx: i16, m: &Mat4, depth: usize, out: &mut HullGeom) {
    if idx < 0 || depth > MAX_SHAPE_DEPTH { return; }
    let i = idx as usize;
    let d = c.data();
    match ty {
        SHAPE_SPHERE => {
            let Some((n, o)) = sb.sphere else { return };
            if i >= n { return; }
            let e = o + i * PHMO_SPHERE_ELEM;
            emit_sphere(vec3_at(d, e + 0xA0), d.f32_at(e + 0x40), m, out);
        }
        SHAPE_MULTI_SPHERE => {
            let Some((n, o)) = sb.multi_sphere else { return };
            if i >= n { return; }
            let e = o + i * PHMO_MULTI_SPHERE_ELEM;
            let cnt = d.i32_at(e + 0x40).clamp(0, 8) as usize;
            for k in 0..cnt {
                let s = e + 0x50 + k * 0x10;
                emit_sphere(vec3_at(d, s), d.f32_at(s + 0x0C), m, out);
            }
        }
        SHAPE_PILL => {
            let Some((n, o)) = sb.pill else { return };
            if i >= n { return; }
            let e = o + i * PHMO_PILL_ELEM;
            emit_pill(vec3_at(d, e + 0x50), vec3_at(d, e + 0x60), d.f32_at(e + 0x40), m, out);
        }
        SHAPE_BOX => {
            let Some((n, o)) = sb.boxes else { return };
            if i >= n { return; }
            let e = o + i * PHMO_BOX_ELEM;
            emit_box(d, e, m, out);
        }
        SHAPE_TRIANGLE => {
            let Some((n, o)) = sb.triangle else { return };
            if i >= n { return; }
            let e = o + i * PHMO_TRIANGLE_ELEM;
            let (a, b, cc) = (vec3_at(d, e + 0x50), vec3_at(d, e + 0x60), vec3_at(d, e + 0x70));
            if !(a.is_finite() && b.is_finite() && cc.is_finite()) { return; }
            let base = out.base();
            for p in [a, b, cc] { out.verts.push(m.transform_point3(p).to_array()); }
            out.push_tri(base, base + 1, base + 2);
        }
        SHAPE_POLYHEDRON => {
            let Some((n, o)) = sb.poly else { return };
            if i >= n { return; }
            emit_polyhedron(c, sb, o + i * PHMO_POLYHEDRON_ELEM, i, m, out);
        }
        SHAPE_LIST => {
            let Some((n, o)) = sb.list else { return };
            if i >= n { return; }
            // A list's children are the NEXT `child shapes size` entries of `list shapes`, in
            // list order (the same sequential rule the four-vector pool uses).
            let mut start = 0usize;
            for k in 0..i { start += d.i32_at(o + k * PHMO_LIST_ELEM + 0x38).max(0) as usize; }
            let cnt = d.i32_at(o + i * PHMO_LIST_ELEM + 0x38).max(0) as usize;
            let Some((nls, lso)) = sb.list_shape else { return };
            for k in start..(start + cnt).min(nls) {
                let e = lso + k * PHMO_LIST_SHAPE_ELEM;
                emit_shape(c, sb, d.u16_at(e), d.i16_at(e + 2), m, depth + 1, out);
            }
        }
        SHAPE_MOPP => {
            let Some((n, o)) = sb.mopp else { return };
            if i >= n { return; }
            let e = o + i * PHMO_MOPP_ELEM;
            emit_shape(c, sb, d.u16_at(e + 0x58), d.i16_at(e + 0x5A), m, depth + 1, out);
        }
        // 6 = phantom: a trigger volume with no collision geometry.
        _ => {}
    }
}

fn vec3_at(d: &[u8], o: usize) -> Vec3 { Vec3::new(d.f32_at(o), d.f32_at(o + 4), d.f32_at(o + 8)) }

const BOX_FACES: [[u32; 3]; 12] = [
    [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
    [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5],
];

/// Is this triple a usable rotation frame (unit, mutually orthogonal)?
fn is_frame(i: Vec3, j: Vec3, k: Vec3) -> bool {
    [i, j, k].iter().all(|v| v.is_finite() && (v.length() - 1.0).abs() < 1e-2)
        && i.dot(j).abs() < 1e-2 && i.dot(k).abs() < 1e-2 && j.dot(k).abs() < 1e-2
}

fn emit_box(d: &[u8], e: usize, m: &Mat4, out: &mut HullGeom) {
    let h = vec3_at(d, e + 0x50);
    if !h.is_finite() || h.abs().max_element() > 1.0e5 { return; }
    // In Halo 4 / H2A the serialized Havok transform is runtime-only: over ALL 368 `phmo` boxes of
    // all 60 shipped caches the rotation region holds one of just two byte patterns (zeros plus
    // five 1.0s at +0xBC..+0xCC), never an orthonormal frame, and the translation at +0xD0 is
    // ALWAYS zero - a box is an AXIS-ALIGNED volume at its rigid body's node origin. Reading
    // Reach's basis columns literally scaled every corner by (1,1,1), which turned a 0.37 x 0.97 x
    // 0.97 wu `wall_m` blocker into a 0.97 wu cube. So the frame is used only when it IS one.
    let (ri, rj, rk) = (vec3_at(d, e + 0xA0), vec3_at(d, e + 0xB0), vec3_at(d, e + 0xC0));
    let t = vec3_at(d, e + 0xD0);
    let t = if t.is_finite() { t } else { Vec3::ZERO };
    let (ri, rj, rk) = if is_frame(ri, rj, rk) { (ri, rj, rk) } else { (Vec3::X, Vec3::Y, Vec3::Z) };
    let base = out.base();
    for k in 0..8u32 {
        let s = Vec3::new(
            if k & 4 != 0 { 1.0 } else { -1.0 },
            if k & 2 != 0 { 1.0 } else { -1.0 },
            if k & 1 != 0 { 1.0 } else { -1.0 },
        ) * h;
        let p = ri * s.x + rj * s.y + rk * s.z + t;
        out.verts.push(m.transform_point3(p).to_array());
    }
    for f in BOX_FACES { out.push_tri(base + f[0], base + f[1], base + f[2]); }
}

fn emit_sphere(centre: Vec3, r: f32, m: &Mat4, out: &mut HullGeom) {
    if !centre.is_finite() || !r.is_finite() || r <= 0.0 || r > 1.0e5 { return; }
    let base = out.base();
    for i in 0..=SPHERE_STACKS {
        let phi = i as f32 / SPHERE_STACKS as f32 * std::f32::consts::PI;
        let (sp, cp) = phi.sin_cos();
        for j in 0..=SPHERE_SLICES {
            let th = j as f32 / SPHERE_SLICES as f32 * std::f32::consts::TAU;
            let p = centre + Vec3::new(r * sp * th.cos(), r * sp * th.sin(), r * cp);
            out.verts.push(m.transform_point3(p).to_array());
        }
    }
    let row = (SPHERE_SLICES + 1) as u32;
    for i in 0..SPHERE_STACKS as u32 {
        for j in 0..SPHERE_SLICES as u32 {
            let (a, b) = (base + i * row + j, base + (i + 1) * row + j);
            out.push_tri(a, b, b + 1);
            out.push_tri(a, b + 1, a + 1);
        }
    }
}

fn emit_pill(bottom: Vec3, top: Vec3, r: f32, m: &Mat4, out: &mut HullGeom) {
    if !(bottom.is_finite() && top.is_finite()) || !r.is_finite() || r <= 0.0 || r > 1.0e5 { return; }
    let axis = top - bottom;
    let len = axis.length();
    if !len.is_finite() { return; }
    if len < 1e-5 { emit_sphere(bottom, r, m, out); return; }
    let a = axis / len;
    let seed = if a.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = (seed - a * seed.dot(a)).normalize_or_zero();
    if u == Vec3::ZERO { return; }
    let w = a.cross(u);
    let base = out.base();
    for ring in 0..2 {
        let o = if ring == 1 { top } else { bottom };
        for j in 0..=SPHERE_SLICES {
            let th = j as f32 / SPHERE_SLICES as f32 * std::f32::consts::TAU;
            let (s, cc) = th.sin_cos();
            out.verts.push(m.transform_point3(o + (u * cc + w * s) * r).to_array());
        }
    }
    let row = (SPHERE_SLICES + 1) as u32;
    for j in 0..SPHERE_SLICES as u32 {
        out.push_tri(base + j, base + row + j, base + row + j + 1);
        out.push_tri(base + j, base + row + j + 1, base + j + 1);
    }
    emit_sphere(bottom, r, m, out);
    emit_sphere(top, r, m, out);
}

/// Unpack one polyhedron's SOA four-vectors and triangulate its convex hull.
fn emit_polyhedron(c: &H4Cache, sb: &ShapeBlocks, pe: usize, pi: usize, m: &Mat4, out: &mut HullGeom) {
    let d = c.data();
    let nv = d.i32_at(pe + 0x80);
    if nv <= 0 || nv as usize > MAX_POLY_VERTS { return; }
    let nv = nv as usize;
    let fvs = d.i32_at(pe + 0x78).max(0) as usize;
    let fvs = if fvs > 0 { fvs } else { nv.div_ceil(4) };
    let Some((nfv, fvo)) = sb.four_vec else { return };
    let start = sb.poly_fv_start.get(pi).copied().unwrap_or(0);
    if start + fvs > nfv { return; }
    let mut pts: Vec<Vec3> = Vec::with_capacity(nv);
    'outer: for k in 0..fvs {
        let f = fvo + (start + k) * PHMO_FOUR_VECTOR_ELEM;
        for j in 0..4 {
            if pts.len() >= nv { break 'outer; }
            let p = Vec3::new(d.f32_at(f + j * 4), d.f32_at(f + 0x10 + j * 4), d.f32_at(f + 0x20 + j * 4));
            if p.is_finite() { pts.push(p); } else { pts.push(Vec3::ZERO); }
        }
    }
    hull_triangulate(&pts, m, out);
}

/// Brute-force convex hull: a vertex triple is a face when every other vertex is on one side of
/// its plane. O(n^3) with a tiny n (Halo 4's biggest polyhedron is well under the cap); above the
/// cap the AABB box stands in, which is still a usable overlay.
fn hull_triangulate(pts: &[Vec3], m: &Mat4, out: &mut HullGeom) {
    let n = pts.len();
    if n < 3 { return; }
    if n == 3 {
        let base = out.base();
        for p in pts { out.verts.push(m.transform_point3(*p).to_array()); }
        out.push_tri(base, base + 1, base + 2);
        return;
    }
    if n > HULL_BRUTE_LIMIT {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for p in pts { mn = mn.min(*p); mx = mx.max(*p); }
        let base = out.base();
        for k in 0..8u32 {
            let p = Vec3::new(
                if k & 4 != 0 { mx.x } else { mn.x },
                if k & 2 != 0 { mx.y } else { mn.y },
                if k & 1 != 0 { mx.z } else { mn.z },
            );
            out.verts.push(m.transform_point3(p).to_array());
        }
        for f in BOX_FACES { out.push_tri(base + f[0], base + f[1], base + f[2]); }
        return;
    }
    let base = out.base();
    for p in pts { out.verts.push(m.transform_point3(*p).to_array()); }
    const EPS: f32 = 1e-4;
    for a in 0..n {
        for b in (a + 1)..n {
            for cc in (b + 1)..n {
                let nrm = (pts[b] - pts[a]).cross(pts[cc] - pts[a]);
                let nl = nrm.length();
                if nl < EPS { continue; }
                let nrm = nrm / nl;
                let dd = nrm.dot(pts[a]);
                let (mut pos, mut neg) = (0usize, 0usize);
                for (p, v) in pts.iter().enumerate() {
                    if p == a || p == b || p == cc { continue; }
                    let s = nrm.dot(*v) - dd;
                    if s > EPS { pos += 1; }
                    if s < -EPS { neg += 1; }
                    if pos > 0 && neg > 0 { break; }
                }
                if pos > 0 && neg > 0 { continue; }
                // Wind the face so its normal points away from the interior.
                if pos == 0 { out.push_tri(base + a as u32, base + b as u32, base + cc as u32); }
                else { out.push_tri(base + a as u32, base + cc as u32, base + b as u32); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::{engine_maps_dir, Engine};
    use std::path::PathBuf;

    fn open(engine: Engine, name: &str) -> Option<H4Cache> {
        let dir = engine_maps_dir(engine)?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    fn playable(engine: Engine) -> Vec<PathBuf> {
        let Some(dir) = engine_maps_dir(engine) else { return Vec::new() };
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "map"))
            .filter(|p| !p.to_string_lossy().contains("Copy"))
            .collect();
        v.sort();
        v
    }

    /// Walk every collision surface of a `coll` tag and return (surfaces, worst deviation of a
    /// loop vertex from the surface's own stored plane). The half-edge walk is the decoder's, so
    /// a small number proves the surface / edge / vertex / plane layout on this engine.
    fn planarity_scan(c: &H4Cache, coll_tag: usize) -> (usize, f32) {
        let mut surfaces = 0usize;
        let mut worst = 0.0f32;
        let Some(cm) = c.tag_meta(coll_tag) else { return (0, 0.0) };
        let rid = c.data().u32_at(cm + OFF_COLL_RESOURCE);
        let Some(e) = c.resource_by_id(rid) else { return (0, 0.0) };
        if !c.kind_is(e.kind, RES_COLLISION) { return (0, 0.0); }
        let dd = c.definition(e);
        let base = (e.def_addr & 0x0FFF_FFFF) as usize;
        if base + 12 > dd.len() { return (0, 0.0); }
        let n = dd.i32_at(base).max(0) as usize;
        let Some(a0) = e.fixup_at((base + 4) as u32) else { return (0, 0.0) };
        let bsp0 = (a0 & 0x0FFF_FFFF) as usize;
        let Ok((page, seg)) = c.stream(e, 4) else { return (0, 0.0) };
        let sd: &[u8] = &page;
        for b in 0..n {
            let eb = bsp0 + b * COLL_BSP_ELEM;
            if eb + COLL_BSP_ELEM > dd.len() { break; }
            let blk = |off: usize, elem: usize| -> Option<(usize, usize)> {
                let cnt = dd.i32_at(eb + off);
                if cnt <= 0 { return None; }
                let addr = e.fixup_at((eb + off + 4) as u32)?;
                if addr >> 28 != 4 { return None; }
                let o = seg + (addr & 0x0FFF_FFFF) as usize;
                (o + cnt as usize * elem <= sd.len()).then_some((cnt as usize, o))
            };
            let (Some((nsf, sfo)), Some((ned, edo)), Some((nv, vo)), Some((npl, plo))) =
                (blk(OFF_BSP_SURFACES, SURFACE_ELEM), blk(OFF_BSP_EDGES, EDGE_ELEM),
                 blk(OFF_BSP_VERTICES, VERTEX_ELEM), blk(0x1C, 16)) else { continue };
            for s in 0..nsf {
                let p = sfo + s * SURFACE_ELEM;
                if sd.u8_at(p + 0x0A) & SURF_INVALID != 0 { continue; }
                let pl = sd.u16_at(p) as usize;
                if pl >= npl { continue; }
                let q = plo + pl * 16;
                let nrm = Vec3::new(sd.f32_at(q), sd.f32_at(q + 4), sd.f32_at(q + 8));
                let dist = sd.f32_at(q + 12);
                let first = sd.u16_at(p + 2) as usize;
                if first >= ned { continue; }
                surfaces += 1;
                let mut ed = first;
                let mut count = 0;
                loop {
                    let r = edo + ed * EDGE_ELEM;
                    let (sv, ev) = (sd.i16_at(r), sd.i16_at(r + 2));
                    let (fwd, rev) = (sd.i16_at(r + 4), sd.i16_at(r + 6));
                    let left = sd.i16_at(r + 8);
                    let (vtx, next) = if left as usize == s { (sv, fwd) } else { (ev, rev) };
                    if vtx < 0 || vtx as usize >= nv { break; }
                    let vp = vo + vtx as usize * VERTEX_ELEM;
                    let v = Vec3::new(sd.f32_at(vp), sd.f32_at(vp + 4), sd.f32_at(vp + 8));
                    worst = worst.max((nrm.dot(v) - dist).abs());
                    count += 1;
                    if next < 0 || next as usize >= ned || count > 64 { break; }
                    ed = next as usize;
                    if ed == first { break; }
                }
            }
        }
        (surfaces, worst)
    }

    /// hlmt +0x10 = `coll`, +0x30 = `phmo` on every model tag of every shipped Halo 4 cache.
    #[test]
    fn hlmt_coll_phmo_offsets_hold_on_every_map() {
        let files = playable(Engine::Halo4);
        if files.is_empty() { eprintln!("skip: no halo4 maps"); return; }
        let (mut hlmt, mut coll, mut phmo) = (0usize, 0usize, 0usize);
        for f in &files {
            let Ok(c) = H4Cache::open(f) else { continue };
            if c.map_type == 3 || c.map_type == 4 { continue; }
            for t in c.find_tags(b"hlmt") {
                let Some(m) = c.tag_meta(t) else { continue };
                hlmt += 1;
                // tag_ref_of demands the class, so a hit proves the offset AND the class.
                if c.tag_ref_of(m + OFF_HLMT_COLL, b"coll").is_some() { coll += 1; }
                if c.tag_ref_of(m + OFF_HLMT_PHMO, b"phmo").is_some() { phmo += 1; }
                // a non-null ref at these offsets is NEVER another class
                if let Some((cls, _)) = c.tag_ref(m + OFF_HLMT_COLL) { assert_eq!(&cls, b"coll", "{} hlmt {t}", c.map_name); }
                if let Some((cls, _)) = c.tag_ref(m + OFF_HLMT_PHMO) { assert_eq!(&cls, b"phmo", "{} hlmt {t}", c.map_name); }
            }
        }
        eprintln!("halo4: {hlmt} hlmt, {coll} with coll, {phmo} with phmo");
        assert_eq!((hlmt, coll, phmo), (16039, 14579, 14657), "the shipped Halo 4 build's hlmt census");
    }

    /// Ravine: every `coll` resolves its `collision_model_resource`, the per-permutation
    /// (offset, count) ranges tile the resource's BSP array, and every decoded surface polygon
    /// lies on its own stored plane.
    #[test]
    fn ravine_coll_resource_and_planarity() {
        let Some(c) = open(Engine::Halo4, "ca_forge_ravine.map") else { return };
        let colls = c.find_tags(b"coll");
        assert_eq!(colls.len(), 645);
        let mut decoded = 0;
        let mut tris = 0;
        for &t in &colls {
            let cm = c.tag_meta(t).expect("coll meta");
            let rid = c.data().u32_at(cm + OFF_COLL_RESOURCE);
            let e = c.resource_by_id(rid).expect("collision resource live on Ravine");
            assert!(c.kind_is(e.kind, RES_COLLISION), "coll {t} resource kind");
            let dd = c.definition(e);
            let base = (e.def_addr & 0x0FFF_FFFF) as usize;
            let n = dd.i32_at(base).max(0) as usize;
            // the permutation ranges tile [0, n)
            let mut total = 0usize;
            let (nr, ro) = c.block(cm + OFF_COLL_REGIONS).unwrap_or((0, 0));
            for r in 0..nr {
                let Some((np, po)) = c.block(ro + r * COLL_REGION_ELEM + OFF_COLL_REGION_PERMS) else { continue };
                for p in 0..np {
                    let pe = po + p * COLL_PERM_ELEM;
                    let off = c.data().i16_at(pe + 4);
                    let cnt = c.data().i16_at(pe + 6);
                    assert!(off >= 0 && cnt >= 0 && off as usize + cnt as usize <= n, "coll {t} perm range");
                    total += cnt as usize;
                }
            }
            assert_eq!(total, n, "coll {t} permutation ranges tile the resource");
            if let Some(g) = coll_geometry(&c, t, &[]) {
                if !g.is_empty() { decoded += 1; tris += g.tri_count(); }
            }
        }
        eprintln!("ravine coll: {decoded}/{} decoded, {tris} triangles", colls.len());
        assert_eq!(decoded, 644, "every Ravine coll but one has geometry");
        assert!(tris > 100_000, "collision triangles: {tris}");
    }

    /// The collision half-edge walk is right: every polygon it builds is planar with its surface's
    /// own plane. Run over one Forge map's whole collision set (the sweep that proved the layout
    /// covered all 50 caches; this keeps a fast, deterministic slice of it in CI).
    #[test]
    fn ravine_surfaces_are_planar_with_their_plane() {
        let Some(c) = open(Engine::Halo4, "ca_forge_ravine.map") else { return };
        let mut surfaces = 0usize;
        let mut worst = 0.0f32;
        for t in c.find_tags(b"coll") {
            let (n, w) = planarity_scan(&c, t);
            surfaces += n;
            worst = worst.max(w);
        }
        eprintln!("ravine: {surfaces} surfaces, worst plane deviation {worst:.5} wu");
        assert!(surfaces > 100_000, "surfaces walked: {surfaces}");
        assert!(worst < 0.02, "a decoded surface is off its own plane by {worst} wu");
    }

    /// The Warthog: 23 collision BSPs (one per region permutation), whose node indices name real
    /// nodes, and the node transform puts the main hull back in model space (without it the hull
    /// is rotated 120 degrees - `b_chassis` is a (-.5,-.5,-.5,.5) quaternion).
    #[test]
    fn warthog_collision_is_node_space() {
        let Some(c) = open(Engine::Halo4, "ca_forge_ravine.map") else { return };
        let veh = c.find_tags(b"vehi").into_iter()
            .find(|&t| c.tag_name(t).ends_with("storm_warthog\\storm_warthog"))
            .expect("the Warthog is in Ravine's palette");
        let nodes = model_node_xforms(&c, model_of(&c, veh).expect("mode"));
        assert_eq!(nodes.len(), 53, "warthog node count");
        let raw = coll_geometry(&c, coll_of(&c, veh).expect("coll"), &[]).expect("raw hull");
        let posed = object_collision_hull(&c, veh).expect("posed hull");
        let (rmn, rmx) = raw.bounds().unwrap();
        let (pmn, pmx) = posed.bounds().unwrap();
        eprintln!("warthog coll raw {:?}..{:?} posed {:?}..{:?}", rmn, rmx, pmn, pmx);
        // node space: longest axis is Y; model space: longest axis is X (the vehicle points +X)
        let rext = rmx - rmn;
        let pext = pmx - pmn;
        assert!(rext.y > rext.x * 2.0, "node-space hull is long in Y: {rext:?}");
        assert!(pext.x > pext.y && pext.x > pext.z, "model-space hull is long in X: {pext:?}");
        // and it agrees with the render model's own extents to a fender's width
        assert!((pext.x - 2.07).abs() < 0.4 && (pext.y - 1.02).abs() < 0.3, "warthog model extent {pext:?}");
        assert_eq!(raw.tri_count(), posed.tri_count(), "the transform does not change topology");
    }

    /// The `phmo` shape kinds all tessellate: the Warthog is one polyhedron rigid body, the
    /// Mongoose two, and the Forge hidden blocks are single boxes / one polyhedron.
    #[test]
    fn ravine_physics_shapes() {
        let Some(c) = open(Engine::Halo4, "ca_forge_ravine.map") else { return };
        let by_name = |suffix: &str| -> usize {
            for cls in [b"vehi", b"bloc", b"scen"] {
                if let Some(t) = c.find_tags(cls).into_iter().find(|&t| c.tag_name(t).ends_with(suffix)) { return t; }
            }
            panic!("{suffix} not in ca_forge_ravine");
        };
        // Warthog: 1 rigid body, polyhedron, 20 vertices -> a closed hull
        let wh = by_name("storm_warthog\\storm_warthog");
        let g = object_physics_hull(&c, wh).expect("warthog phmo");
        assert!(g.tri_count() >= 12, "warthog hull tris {}", g.tri_count());
        let (mn, mx) = g.bounds().unwrap();
        let ext = mx - mn;
        eprintln!("warthog phmo extent {ext:?} tris {}", g.tri_count());
        assert!((ext.x - 1.832).abs() < 0.05 && (ext.y - 0.826).abs() < 0.05, "warthog phmo extent {ext:?}");
        // Every nut_blocker (the hidden "superforge" bb_* set) has a phmo volume and NO coll.
        // The FULL x/y/z extent is asserted, not just the longest side: reading Reach's box
        // rotation basis literally (which Halo 4 does not serialize) inflated the short axes into
        // cubes, and only a per-axis check catches that.
        let blockers = [
            ("box_m", [0.968, 0.968, 0.968]), ("box_l", [0.968, 1.968, 1.968]),
            ("box_xxl", [1.968, 3.968, 3.968]), ("box_xxxl", [3.968, 3.968, 3.968]),
            ("wall_s", [0.468, 0.468, 0.668]), ("wall_m", [0.368, 0.968, 0.968]),
            ("wall_l", [0.368, 0.968, 1.968]), ("wall_xl", [0.368, 1.968, 1.968]),
            ("wall_xxl", [0.368, 1.968, 3.968]), ("wall_xxxl", [0.368, 3.968, 3.968]),
        ];
        for (nm, want) in blockers {
            let t = by_name(&format!("nut_blockers\\{nm}\\{nm}"));
            assert!(coll_of(&c, t).is_none(), "{nm} has no collision model in Halo 4");
            let g = object_physics_hull(&c, t).unwrap_or_else(|| panic!("{nm} phmo"));
            let (mn, mx) = g.bounds().unwrap();
            let ext = mx - mn;
            eprintln!("  {nm}: hull extent {ext:?}");
            for k in 0..3 {
                assert!((ext[k] - want[k]).abs() < 0.01, "{nm} hull extent {ext:?} (want {want:?})");
            }
            assert!(g.tri_count() >= 12, "{nm} tris {}", g.tri_count());
        }
    }

    /// Every vehicle, block and a slice of the scenery of every shipped Halo 4 map decodes both
    /// hulls without a panic, with finite geometry and plausible extents.
    #[test]
    fn every_map_vehicles_blocks_and_scenery_decode() {
        let files = playable(Engine::Halo4);
        if files.is_empty() { eprintln!("skip: no halo4 maps"); return; }
        let (mut objs, mut with_coll, mut with_phmo, mut ctris, mut ptris) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for f in &files {
            let Ok(c) = H4Cache::open(f) else { continue };
            if c.map_type == 3 || c.map_type == 4 { continue; }
            let mut tags: Vec<usize> = Vec::new();
            for cls in [b"vehi", b"bloc"] { tags.extend(c.find_tags(cls)); }
            // a deterministic slice of the scenery (every 8th tag) keeps the test quick
            tags.extend(c.find_tags(b"scen").into_iter().step_by(8));
            for t in tags {
                objs += 1;
                if let Some(g) = object_collision_hull(&c, t) {
                    with_coll += 1;
                    ctris += g.tri_count();
                    let (mn, mx) = g.bounds().expect("coll bounds");
                    assert!(mn.is_finite() && mx.is_finite(), "{} {}: coll bounds", c.map_name, c.tag_name(t));
                    assert!((mx - mn).max_element() < 2000.0, "{} {}: coll extent {:?}", c.map_name, c.tag_name(t), mx - mn);
                    assert!(g.indices.iter().all(|&i| (i as usize) < g.verts.len()), "coll index range");
                }
                if let Some(g) = object_physics_hull(&c, t) {
                    with_phmo += 1;
                    ptris += g.tri_count();
                    let (mn, mx) = g.bounds().expect("phmo bounds");
                    assert!(mn.is_finite() && mx.is_finite(), "{} {}: phmo bounds", c.map_name, c.tag_name(t));
                    assert!((mx - mn).max_element() < 2000.0, "{} {}: phmo extent {:?}", c.map_name, c.tag_name(t), mx - mn);
                    assert!(g.indices.iter().all(|&i| (i as usize) < g.verts.len()), "phmo index range");
                }
            }
        }
        eprintln!("halo4: {objs} objects, {with_coll} coll hulls ({ctris} tris), {with_phmo} phmo hulls ({ptris} tris)");
        assert!(with_coll > 3000 && with_phmo > 3000, "hulls decoded: {with_coll} / {with_phmo}");
    }

    /// Polyhedra: the four-vector pool is consumed sequentially, so each polyhedron's unpacked
    /// vertices reproduce its OWN stored AABB (centre @0x60, half extents @0x50). Checked on every
    /// polyhedron of two Halo 4 maps; the deviation is the Havok convex radius (tag default 0.016).
    #[test]
    fn polyhedron_four_vectors_match_stored_aabb() {
        for map in ["ca_forge_ravine.map", "wraparound.map"] {
            let Some(c) = open(Engine::Halo4, map) else { return };
            let d = c.data();
            let mut checked = 0usize;
            let mut worst = 0.0f32;
            for t in c.find_tags(b"phmo") {
                let Some(pm) = c.tag_meta(t) else { continue };
                let Some((n, o)) = c.block(pm + OFF_PHMO_POLYHEDRA) else { continue };
                let Some((nfv, fvo)) = c.block(pm + OFF_PHMO_FOUR_VECTORS) else { continue };
                let mut cursor = 0usize;
                for i in 0..n {
                    let e = o + i * PHMO_POLYHEDRON_ELEM;
                    let nv = d.i32_at(e + 0x80).max(0) as usize;
                    let fvs = d.i32_at(e + 0x78).max(0) as usize;
                    assert_eq!(fvs, nv.div_ceil(4), "{map} phmo {t} poly {i}: four-vector size");
                    if nv == 0 || cursor + fvs > nfv { cursor += fvs; continue; }
                    let mut mn = Vec3::splat(f32::MAX);
                    let mut mx = Vec3::splat(f32::MIN);
                    let mut seen = 0usize;
                    'fv: for k in 0..fvs {
                        let f = fvo + (cursor + k) * PHMO_FOUR_VECTOR_ELEM;
                        for j in 0..4 {
                            if seen >= nv { break 'fv; }
                            let p = Vec3::new(d.f32_at(f + j * 4), d.f32_at(f + 0x10 + j * 4), d.f32_at(f + 0x20 + j * 4));
                            mn = mn.min(p);
                            mx = mx.max(p);
                            seen += 1;
                        }
                    }
                    cursor += fvs;
                    let centre = (mn + mx) * 0.5;
                    let half = (mx - mn) * 0.5;
                    let sc = vec3_at(d, e + 0x60);
                    let sh = vec3_at(d, e + 0x50);
                    worst = worst.max((centre - sc).abs().max_element()).max((half - sh).abs().max_element());
                    checked += 1;
                }
                assert_eq!(cursor, nfv, "{map} phmo {t}: the four-vector pool is exactly consumed");
            }
            eprintln!("{map}: {checked} polyhedra, worst AABB deviation {worst:.4} wu");
            assert!(checked > 100, "polyhedra checked: {checked}");
            assert!(worst < 0.03, "{map}: polyhedron AABB deviation {worst}");
        }
    }

    /// The node transform read from the INVERSE absolute frame equals accumulating the default
    /// translation / rotation down the parent chain (the reading is right, and cheap).
    #[test]
    fn node_xform_matches_default_pose_chain() {
        let Some(c) = open(Engine::Halo4, "ca_forge_ravine.map") else { return };
        let d = c.data();
        let mut checks = 0usize;
        let mut worst = 0.0f32;
        for t in c.find_tags(b"mode").into_iter().step_by(7) {
            let Some(mm) = c.tag_meta(t) else { continue };
            let Some((n, o)) = c.block(mm + OFF_MODE_NODES) else { continue };
            let xf = model_node_xforms(&c, t);
            // absolute = parent_absolute * (translate(default) * rotate(default))
            let mut abs: Vec<Option<Mat4>> = vec![None; n];
            for i in 0..n {
                let mut chain: Vec<usize> = Vec::new();
                let mut k = i as i32;
                while k >= 0 && (k as usize) < n && chain.len() <= n {
                    chain.push(k as usize);
                    let p = d.i16_at(o + k as usize * NODE_ELEM + 4) as i32;
                    if p == k { break; }
                    k = p;
                }
                let mut m = Mat4::IDENTITY;
                for &nd in chain.iter().rev() {
                    let e = o + nd * NODE_ELEM;
                    let tr = vec3_at(d, e + 0x0C);
                    let q = glam::Quat::from_xyzw(d.f32_at(e + 0x18), d.f32_at(e + 0x1C), d.f32_at(e + 0x20), d.f32_at(e + 0x24));
                    m *= Mat4::from_rotation_translation(q.normalize(), tr);
                }
                abs[i] = Some(m);
            }
            for i in 0..n {
                let (Some(a), Some(b)) = (abs[i], xf.get(i).copied()) else { continue };
                for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.3, -0.7, 0.2)] {
                    let e = (a.transform_point3(p) - b.transform_point3(p)).abs().max_element();
                    worst = worst.max(e);
                    checks += 1;
                }
            }
        }
        eprintln!("node transforms: {checks} point checks, worst {worst:.2e}");
        assert!(checks > 1000, "checks: {checks}");
        assert!(worst < 1e-3, "node transform mismatch {worst}");
    }

    /// The HIDDEN-BLOCK rule (`edit_scene::blocker_hull`) agrees with the scenario's own hidden
    /// ("superforge") Forge palette on every Forge map: a palette object is flagged iff the engine
    /// would draw nothing for it AND it has a hull with volume. Nothing here matches on names.
    #[test]
    fn hidden_block_rule_matches_the_hidden_palette() {
        use crate::h4::objects::{model_of, OFF_MODE_BOUNDS};
        use crate::h4::palette::palette;
        // Halo 4 hidden-block cutoffs (the Reach numbers, mirrored in edit_scene.rs).
        const RENDER_MAX: f32 = 0.25;
        const HULL_MIN: f32 = 0.25;
        for map in ["ca_forge_ravine.map", "dlc_forge_island.map", "ca_forge_erosion.map", "ca_forge_bonanza.map", "wraparound.map"] {
            let Some(c) = open(Engine::Halo4, map) else { return };
            let p = palette(&c);
            if p.entries.is_empty() { eprintln!("{map}: no forge palette"); continue; }
            let mut flagged: Vec<String> = Vec::new();
            let mut hidden_cat: Vec<String> = Vec::new();
            for cat in &p.categories {
                for e in &p.entries[cat.first_entry..(cat.first_entry + cat.entry_count).min(p.entries.len())] {
                    for v in &e.variants {
                        let Some(t) = v.tag else { continue };
                        // the render model's VISIBLE extent (mode +0x80 compression bounds, one
                        // model-wide element, stored x0 x1 y0 y1 z0 z1)
                        let mut render_ext = 0.0f32;
                        let mut render_empty = true;
                        if let Some(mode) = model_of(&c, t) {
                            if let Some(mm) = c.tag_meta(mode) {
                                render_empty = c.block(mm + 0x68).is_none();
                                if let Some((_, bo)) = c.block(mm + OFF_MODE_BOUNDS) {
                                    let d = c.data();
                                    for k in 0..3 {
                                        render_ext = render_ext.max((d.f32_at(bo + 4 + k * 8 + 4) - d.f32_at(bo + 4 + k * 8)).abs());
                                    }
                                }
                            }
                        }
                        let hull = object_collision_hull(&c, t).or_else(|| object_physics_hull(&c, t));
                        let is_blocker = (render_empty || render_ext < RENDER_MAX)
                            && hull.as_ref().map_or(false, |h| h.extent() > HULL_MIN);
                        let name = c.tag_name(t).rsplit('\\').next().unwrap_or("").to_string();
                        if is_blocker { flagged.push(name.clone()); }
                        if cat.hidden { hidden_cat.push(name); }
                    }
                }
            }
            flagged.sort();
            flagged.dedup();
            hidden_cat.sort();
            hidden_cat.dedup();
            eprintln!("{map}: rule flags {flagged:?}; hidden palette holds {hidden_cat:?}");
            assert_eq!(flagged, hidden_cat, "{map}: the engine-behaviour rule and the hidden palette disagree");
        }
    }

    /// #h2a The same readers on Halo 2 Anniversary: H2A scenario placements are not decoded yet
    /// (docs/h2a_support_plan.md phase 2), so there is nothing to SELECT on an H2A map - this
    /// parses its `coll` / `phmo` tags directly instead.
    #[test]
    fn h2a_coll_and_phmo_parse() {
        let files = playable(Engine::H2A);
        if files.is_empty() { eprintln!("skip: no groundhog maps"); return; }
        let (mut maps, mut colls, mut phmos, mut ctris, mut ptris) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut worst_plane = 0.0f32;
        let mut surfaces = 0usize;
        for f in &files {
            let Ok(c) = H4Cache::open(f) else { continue };
            if c.map_type == 3 || c.map_type == 4 { continue; }
            assert_eq!(c.engine, Engine::H2A, "{}", f.display());
            maps += 1;
            for t in c.find_tags(b"hlmt") {
                let Some(m) = c.tag_meta(t) else { continue };
                if let Some((cls, _)) = c.tag_ref(m + OFF_HLMT_COLL) { assert_eq!(&cls, b"coll", "H2A hlmt {t}"); }
                if let Some((cls, _)) = c.tag_ref(m + OFF_HLMT_PHMO) { assert_eq!(&cls, b"phmo", "H2A hlmt {t}"); }
            }
            // object hulls through the same chain used on Halo 4
            let mut tags: Vec<usize> = Vec::new();
            for cls in [b"vehi", b"bloc"] { tags.extend(c.find_tags(cls)); }
            tags.extend(c.find_tags(b"scen").into_iter().step_by(8));
            for t in tags {
                if let Some(g) = object_collision_hull(&c, t) {
                    colls += 1;
                    ctris += g.tri_count();
                    let (mn, mx) = g.bounds().expect("bounds");
                    assert!(mn.is_finite() && mx.is_finite() && (mx - mn).max_element() < 2000.0,
                            "{} {}: coll extent {:?}", c.map_name, c.tag_name(t), mx - mn);
                }
                if let Some(g) = object_physics_hull(&c, t) {
                    phmos += 1;
                    ptris += g.tri_count();
                    let (mn, mx) = g.bounds().expect("bounds");
                    assert!(mn.is_finite() && mx.is_finite() && (mx - mn).max_element() < 2000.0,
                            "{} {}: phmo extent {:?}", c.map_name, c.tag_name(t), mx - mn);
                }
            }
            // the same half-edge walk on this engine (a deterministic slice per map)
            for t in c.find_tags(b"coll").into_iter().step_by(5) {
                let (n, w) = planarity_scan(&c, t);
                surfaces += n;
                worst_plane = worst_plane.max(w);
            }
        }
        eprintln!("h2a: {maps} maps, {colls} coll hulls ({ctris} tris), {phmos} phmo hulls ({ptris} tris), {surfaces} surfaces, worst plane {worst_plane:.5}");
        assert!(maps >= 8, "H2A playable caches: {maps}");
        assert!(colls > 100 && phmos > 100, "H2A hulls: {colls} / {phmos}");
        assert!(surfaces > 20_000, "H2A surfaces walked: {surfaces}");
        assert!(worst_plane < 0.02, "H2A collision plane deviation {worst_plane}");
    }
}
