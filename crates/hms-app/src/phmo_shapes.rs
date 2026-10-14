//! `phmo` rigid bodies -> settle shapes, for both games. #settle
//!
//! The physics_model tag layout is Reach's in both games (Assembly `ReachMCC/phmo.xml` and
//! `Halo4MCC/phmo.xml`, cross-checked against `reach_tag_test.exe`'s reflection through ReverseMe
//! `tagdefs phmo`): the same block offsets, the same shape-type enum, the same per-shape
//! `s_physics_model_primitive` header (friction @+0x0C, restitution @+0x10, mass @+0x18) and
//! `hkpConvexShape::m_radius` @+0x40. Only two element sizes and the rigid body's tail differ
//! (`PhmoLayout`). The blocks arrive as plain byte buffers, so the Halo 4 cache reader and the
//! Reach native DLL (`tag_meta` / `tag_ptr_bytes`) both feed the one walker.
//!
//! Rigid body element (both games unless noted):
//!   node i16 @0 | flags u16 @0x18 (bit 1 infinite inertia, bit 3 Havok rigid body, bit 4 explicit
//!   mass, bit 5 absolute centre of mass) | motion type u8 @0x1C (0 sphere .. 4 keyframed, 5 fixed)
//!   inertia tensor scale f32 @0x20 + xyz @0x24 | linear damping @0x30 | angular damping @0x34
//!   centre of mass offset @0x38 | centre of mass xyz (Reach @0x60, H4 @0x70) |
//!   inertia tensor rows x/y/z (Reach @0x70/0x80/0x90, H4 @0x80/0x90/0xA0, 16 B apart) |
//!   shape type u16 / index i16 (Reach @0xA8/0xAA, H4 @0xB8/0xBA) | mass f32 (Reach @0xB0, H4 @0xC0)
//! Shapes: sphere radius @0x40 translation @0xA0 | pill radius @0x40 bottom @0x50 top @0x60 |
//!   box half extents @0x50, radius @0x40, rotation i/j/k @0xA0/B0/C0 + translation @0xD0 (used
//!   only when they form a real frame: Halo 4 does not serialise them, see h4/collision.rs) |
//!   triangle a/b/c @0x50/0x60/0x70 | multi sphere count @0x40, (xyz, r) @0x50 + 16 i |
//!   polyhedron four-vector count @0x78, vertex count @0x80 (SOA pool consumed sequentially) |
//!   list child count @0x38 (children = the next entries of `list shapes`: type @0, index @2) |
//!   MOPP child type @0x58 / index @0x5A. Phantoms (6) carry no geometry.
//!
//! ## Constraints (#ragdoll)
//!
//! **Where `ragdoll constraints` is, and how that was established.** The engine's own tag
//! reflection in `reach_tag_test.exe` lists the phmo's block fields in offset order, and the run
//! that matters reads (field-definition array at +0x1E7C020, stride 0x20, name pointer at +0x18):
//!   `rigid bodies`, `materials`, `spheres`, `pills`, `boxes`, `triangles`, `polyhedra`,
//!   `polyhedron four vectors`, `polyhedron plane equations`, `mass distributions`, `lists`,
//!   `list shapes`, `mopps`, `mopp codes` (a DATA reference, 20 B), **`hinge constraints`**,
//!   **`ragdoll constraints`**, `regions`, `nodes`, `REMOVE ME`, `errors`.
//! That run is used for the ORDER ONLY, not for absolute offsets: it does not list
//! `multi spheres` between `spheres` and `pills`, and the shipped layout must have it (the
//! editor's settle walks `pills` at 0x8C and `mopps` at 0xF8 on real maps and would be a block
//! out otherwise), so the array as read is not the whole phmo field set and counting from its
//! start would be wrong by 12 bytes. With `mopps` at 0xF8 taken as known good plus the engine's
//! own 20-byte `data` size, the order gives
//! `mopp codes` 0x104, **`hinge constraints` 0x118, `ragdoll constraints` 0x124**, `regions`
//! 0x130, `nodes` 0x13C, `REMOVE ME` 0x148 (which is exactly the three nameless 4-byte fields the
//! Assembly plugin shows at 0x148/0x14C/0x150 -- one block reference) and `errors` 0x154. Those
//! two constraint offsets are therefore derived from the engine, not taken on trust.
//!
//! The FOUR other constraint types (`limited hinge` 0x160, `ball and socket` 0x16C,
//! `stiff spring` 0x178, `prismatic` 0x184) are the Assembly plugin's attribution of four of the
//! five block-sized fields that fit between `errors` and the shipped tag's 0x19C size; that
//! build's top-level field array does not reach them, so their BLOCK OFFSETS are not
//! independently confirmed (their element layouts are -- see below). They are read defensively: a
//! constraint whose two frames are not orthonormal bases is dropped, which is what a wrong block
//! offset produces, and `constraint_counts_per_type` reports present versus decoded per type so a
//! real tag answers the question rather than a guess.
//!
//! **Every offset INSIDE a constraint element, and every element size, is the engine's own.** The
//! reflection's struct definitions (same binary; the source path they carry is
//! `c:\mcc\qfe1\reach\shared\engine\source\omaha\physics\physics_model_definitions.cpp`, i.e. the
//! shipped MCC Reach build) state both the field list and the `sizeof`:
//!
//! `constraint_bodies_struct` = `sizeof(s_physics_model_constraint_bodies)` = **0x74**, fields in
//! this order with these types, which sum to exactly 0x74:
//!   `name^` string id 4 B @0x00 | `node a` 2 B @0x04 | `node b` 2 B @0x06 | `a scale` real
//!   @0x08 | `a forward` / `a left` / `a up` real vector 3d @0x0C / 0x18 / 0x24 | `a position`
//!   real point 3d @0x30 | `b scale` @0x3C | `b forward` / `b left` / `b up` @0x40 / 0x4C / 0x58 |
//!   `b position` @0x64 | `edge index` 2 B @0x70 | a 2-byte pad @0x72.
//!
//! `ragdoll_constraints_block` = `sizeof(s_physics_model_ragdoll_constraint)` = **0x94**: that
//! 0x74 struct, a 4-byte pad to 0x78, then seven reals --
//!   `min twist` @0x78 | `max twist` @0x7C | `min cone` @0x80 | `max cone` @0x84 |
//!   `min plane` @0x88 | `max plane` @0x8C | `max friciton torque` @0x90 (the engine's own typo).
//! 0x74 + 4 + 7x4 = 0x94, which is why the stated size is the check on the whole layout.
//!
//! The other stated sizes, each matching the element size this module uses:
//! `sizeof(s_physics_model_ball_and_socket_constraint)` **0x78** (the struct, padded, no tail),
//! `limited_hinge_constraints_block` **0x84**, `stiff_spring_constraints_block` **0x7C**,
//! `prismatic_constraints_block` **0x84**.
//!
//! **The angles are DEGREES** -- see [`angle_range_at`].
//!
//! ## What the shipped bipeds actually hold (`--dump-ragdoll forge_halo.map spartans`)
//!
//! `objects\characters\spartans\spartans`, 82 skeleton nodes: **11 rigid bodies, one per node**
//! (`b_pelvis`, `b_spine1`, `b_head`, `b_l`/`b_r_upperarm`, `_forearm`, `_thigh`, `_calf`), every
//! one motion type 2 with an explicit mass summing to **300.00**, and **10 constraints, all 10
//! decoded**: four `limited hinge` (`$_const_l`/`r_elbow`, `$_const_l`/`r_knee`) and six
//! `ragdoll` (`$_const_head`, `$_const_l`/`r_shoulder`, `$_const_l`/`r_hip`, `$_const_spine1`).
//! `objects\characters\elite\elite`, 67 nodes: the same 11-body, 4-hinge, 6-ragdoll shape around
//! `b_spine2` / `b_neck0`. Both: **every phmo node name equals the `mode` node name of the same
//! index** (82 of 82, 67 of 67), so a constraint's node index really does address the render
//! model's skeleton; and **the bind-pose pivot residual is 0.00000 wu on all twenty joints**,
//! which is the behavioural proof of the two position offsets. Every `max friciton torque` is 0,
//! so a Reach ragdoll has no authored joint friction at all.
//!
//! A ragdoll is the phmo's rigid bodies PLUS the authored joints between them, and those live in
//! six sibling blocks that every constraint type starts with the same 0x78-byte header:
//!   `name` string id @0 | node A index i16 @4 | node B index i16 @6 | A scale f32 @8 |
//!   A forward/left/up @0x0C/0x18/0x24 | A position @0x30 | B scale @0x3C |
//!   B forward/left/up @0x40/0x4C/0x58 | B position @0x64 | edge index i16 @0x70.
//! The two frames are the joint's pivot and basis expressed in EACH body's own node space, which
//! is why a correct read has `node_A_bind * A_position == node_B_bind * B_position` at the bind
//! pose -- the gate `constraints_pivots_agree_at_bind` asserts exactly that on the shipped
//! Spartan, and it is what established these offsets (`ragdoll_dump` prints the residual).
//! The per-type tails:
//!   hinge (0x78, no tail) | limited hinge (0x84): limit friction f32 @0x78, limit angle
//!   (min,max) @0x7C | ragdoll (0x94): twist (min,max) @0x78, cone @0x80, plane @0x88, maximum
//!   friction torque f32 @0x90 | ball and socket (0x78, no tail) | stiff spring (0x7C): spring
//!   length @0x78 | prismatic (0x84): limit angle @0x78, maximum friction force @0x80.
//! Every element size is exactly its own last field's end, which is the first internal check; the
//! `node edges` block (@0x50, 0x1C) carries the same topology again as a graph with per-edge
//! friction and motors and is NOT needed to build the joints, since each constraint element names
//! its own two nodes.

use glam::{Mat3, Mat4, Vec3};

use crate::physics_settle::{ConvexShape, RigidBodyDef, ShapeMaterial};

/// Where the two games differ.
#[derive(Clone, Copy, Debug)]
pub struct PhmoLayout {
    pub rb_elem: usize,
    pub rb_shape_type: usize,
    pub rb_shape_index: usize,
    pub rb_mass: usize,
    pub rb_com: usize,
    pub rb_inertia: usize,
    pub poly_elem: usize,
}

pub const REACH: PhmoLayout = PhmoLayout { rb_elem: 0xD0, rb_shape_type: 0xA8, rb_shape_index: 0xAA, rb_mass: 0xB0, rb_com: 0x60, rb_inertia: 0x70, poly_elem: 0xB0 };
pub const HALO4: PhmoLayout = PhmoLayout { rb_elem: 0xE0, rb_shape_type: 0xB8, rb_shape_index: 0xBA, rb_mass: 0xC0, rb_com: 0x70, rb_inertia: 0x80, poly_elem: 0xA0 };

/// Tag-meta offsets of the blocks the walker reads (count i32 @off, pointer @off+4) and their
/// element sizes where both games agree.
pub const OFF_RIGID_BODIES: usize = 0x5C;
pub const OFF_SPHERES: usize = 0x74;
pub const OFF_MULTI_SPHERES: usize = 0x80;
pub const OFF_PILLS: usize = 0x8C;
pub const OFF_BOXES: usize = 0x98;
pub const OFF_TRIANGLES: usize = 0xA4;
pub const OFF_POLYHEDRA: usize = 0xB0;
pub const OFF_FOUR_VECTORS: usize = 0xBC;
pub const OFF_LISTS: usize = 0xE0;
pub const OFF_LIST_SHAPES: usize = 0xEC;
pub const OFF_MOPPS: usize = 0xF8;
/// #ragdoll Constraint blocks (identical offsets and element sizes in Reach and Halo 4; the
/// Assembly plugins' `baseSize` is 0x19C for Reach and 0x1A8 for Halo 4, both past 0x184+12).
pub const OFF_NODE_EDGES: usize = 0x50;
pub const NODE_EDGE_ELEM: usize = 0x1C;
pub const OFF_PHMO_NODES: usize = 0x13C;
pub const PHMO_NODE_ELEM: usize = 0x0C;
pub const OFF_HINGE_CONSTRAINTS: usize = 0x118;
pub const HINGE_CONSTRAINT_ELEM: usize = 0x78;
pub const OFF_RAGDOLL_CONSTRAINTS: usize = 0x124;
pub const RAGDOLL_CONSTRAINT_ELEM: usize = 0x94;
pub const OFF_LIMITED_HINGE_CONSTRAINTS: usize = 0x160;
pub const LIMITED_HINGE_CONSTRAINT_ELEM: usize = 0x84;
pub const OFF_BALL_SOCKET_CONSTRAINTS: usize = 0x16C;
pub const BALL_SOCKET_CONSTRAINT_ELEM: usize = 0x78;
pub const OFF_STIFF_SPRING_CONSTRAINTS: usize = 0x178;
pub const STIFF_SPRING_CONSTRAINT_ELEM: usize = 0x7C;
pub const OFF_PRISMATIC_CONSTRAINTS: usize = 0x184;
pub const PRISMATIC_CONSTRAINT_ELEM: usize = 0x84;
/// The whole `phmo` meta a reader must map to reach every block above. The last block reference
/// this module reads ends at 0x190, so both games' tag sizes cover it: Reach's `phmo` is 0x19C
/// bytes and Halo 4's 0x1A8 (the Assembly plugins' `baseSize`).
pub const REACH_PHMO_META_SIZE: usize = 0x19C;
pub const H4_PHMO_META_SIZE: usize = 0x1A8;
pub const SPHERE_ELEM: usize = 0xB0;
pub const MULTI_SPHERE_ELEM: usize = 0xD0;
pub const PILL_ELEM: usize = 0x70;
pub const BOX_ELEM: usize = 0xE0;
pub const TRIANGLE_ELEM: usize = 0x90;
pub const FOUR_VECTOR_ELEM: usize = 0x30;
pub const LIST_ELEM: usize = 0x90;
pub const LIST_SHAPE_ELEM: usize = 0x20;
pub const MOPP_ELEM: usize = 0x80;
/// The Reach `mode` node element (Halo 4's is 0x70; the inverse-frame fields sit at the same
/// offsets in both).
pub const REACH_MODE_NODE_ELEM: usize = 0x60;
pub const OFF_MODE_NODES: usize = 0x30;

const SHAPE_SPHERE: u16 = 0;
const SHAPE_PILL: u16 = 1;
const SHAPE_BOX: u16 = 2;
const SHAPE_TRIANGLE: u16 = 3;
const SHAPE_POLYHEDRON: u16 = 4;
const SHAPE_MULTI_SPHERE: u16 = 5;
const SHAPE_LIST: u16 = 0xE;
const SHAPE_MOPP: u16 = 0xF;
const MAX_DEPTH: usize = 4;
const MAX_RIGID_BODIES: usize = 4096;
const MAX_POLY_VERTS: usize = 4096;

/// The raw block bytes of one `phmo` (empty = the block is absent).
#[derive(Default, Clone, Debug)]
pub struct PhmoBlocks {
    pub rigid_bodies: Vec<u8>,
    pub spheres: Vec<u8>,
    pub multi_spheres: Vec<u8>,
    pub pills: Vec<u8>,
    pub boxes: Vec<u8>,
    pub triangles: Vec<u8>,
    pub polyhedra: Vec<u8>,
    pub four_vectors: Vec<u8>,
    pub lists: Vec<u8>,
    pub list_shapes: Vec<u8>,
    pub mopps: Vec<u8>,
    /// #ragdoll The constraint-side blocks, keyed by their meta offset (`OFF_NODE_EDGES`,
    /// `OFF_PHMO_NODES`, `OFF_*_CONSTRAINTS`). Kept in a map rather than as fields so a reader
    /// that only wants collision shapes stays unchanged and pays nothing.
    pub constraints: std::collections::HashMap<usize, Vec<u8>>,
}

/// #ragdoll Every constraint-side block (meta offset, element size), in meta order.
pub const CONSTRAINT_BLOCKS: [(usize, usize); 8] = [
    (OFF_NODE_EDGES, NODE_EDGE_ELEM),
    (OFF_HINGE_CONSTRAINTS, HINGE_CONSTRAINT_ELEM),
    (OFF_RAGDOLL_CONSTRAINTS, RAGDOLL_CONSTRAINT_ELEM),
    (OFF_PHMO_NODES, PHMO_NODE_ELEM),
    (OFF_LIMITED_HINGE_CONSTRAINTS, LIMITED_HINGE_CONSTRAINT_ELEM),
    (OFF_BALL_SOCKET_CONSTRAINTS, BALL_SOCKET_CONSTRAINT_ELEM),
    (OFF_STIFF_SPRING_CONSTRAINTS, STIFF_SPRING_CONSTRAINT_ELEM),
    (OFF_PRISMATIC_CONSTRAINTS, PRISMATIC_CONSTRAINT_ELEM),
];

impl PhmoBlocks {
    /// Fill every block through `read(meta_offset, element_size) -> bytes`.
    pub fn read(l: &PhmoLayout, mut read: impl FnMut(usize, usize) -> Vec<u8>) -> PhmoBlocks {
        PhmoBlocks {
            rigid_bodies: read(OFF_RIGID_BODIES, l.rb_elem),
            spheres: read(OFF_SPHERES, SPHERE_ELEM),
            multi_spheres: read(OFF_MULTI_SPHERES, MULTI_SPHERE_ELEM),
            pills: read(OFF_PILLS, PILL_ELEM),
            boxes: read(OFF_BOXES, BOX_ELEM),
            triangles: read(OFF_TRIANGLES, TRIANGLE_ELEM),
            polyhedra: read(OFF_POLYHEDRA, l.poly_elem),
            four_vectors: read(OFF_FOUR_VECTORS, FOUR_VECTOR_ELEM),
            lists: read(OFF_LISTS, LIST_ELEM),
            list_shapes: read(OFF_LIST_SHAPES, LIST_SHAPE_ELEM),
            mopps: read(OFF_MOPPS, MOPP_ELEM),
            constraints: CONSTRAINT_BLOCKS
                .iter()
                .filter_map(|(off, elem)| {
                    let b = read(*off, *elem);
                    (!b.is_empty()).then_some((*off, b))
                })
                .collect(),
        }
    }
}

// ---- byte helpers (bounds-checked: a short block reads as zero) ------------------------------

fn f32_at(d: &[u8], o: usize) -> f32 { d.get(o..o + 4).map_or(0.0, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])) }
fn i32_at(d: &[u8], o: usize) -> i32 { d.get(o..o + 4).map_or(0, |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])) }
fn u16_at(d: &[u8], o: usize) -> u16 { d.get(o..o + 2).map_or(0, |b| u16::from_le_bytes([b[0], b[1]])) }
fn i16_at(d: &[u8], o: usize) -> i16 { u16_at(d, o) as i16 }
fn u8_at(d: &[u8], o: usize) -> u8 { d.get(o).copied().unwrap_or(0) }
fn vec3_at(d: &[u8], o: usize) -> Vec3 { Vec3::new(f32_at(d, o), f32_at(d, o + 4), f32_at(d, o + 8)) }
fn elem<'a>(d: &'a [u8], i: usize, size: usize) -> Option<&'a [u8]> { d.get(i * size..(i + 1) * size) }

/// The primitive header every shape element starts with.
fn material(e: &[u8]) -> ShapeMaterial {
    ShapeMaterial { friction: f32_at(e, 0x0C), restitution: f32_at(e, 0x10), mass: f32_at(e, 0x18) }
}

fn radius(e: &[u8]) -> f32 { f32_at(e, 0x40) }

/// `mode` node -> model matrices from the node block bytes (`elem` = 0x60 Reach / 0x70 Halo 4):
/// rows (inverse forward / left / up) @0x2C/0x38/0x44 and inverse position @0x50 give
/// `p_model = B * (l - inv_position)` (the h4/collision.rs derivation).
pub fn node_xforms(nodes: &[u8], elem: usize) -> Vec<Mat4> {
    let n = if elem == 0 { 0 } else { (nodes.len() / elem).min(1024) };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = &nodes[i * elem..(i + 1) * elem];
        let (f, l, u, ip) = (vec3_at(e, 0x2C), vec3_at(e, 0x38), vec3_at(e, 0x44), vec3_at(e, 0x50));
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

fn is_frame(i: Vec3, j: Vec3, k: Vec3) -> bool {
    [i, j, k].iter().all(|v| v.is_finite() && (v.length() - 1.0).abs() < 1e-2)
        && i.dot(j).abs() < 1e-2 && i.dot(k).abs() < 1e-2 && j.dot(k).abs() < 1e-2
}

/// Shape-type histogram for the status line.
#[derive(Default, Clone, Debug)]
pub struct ShapeCounts { pub spheres: usize, pub pills: usize, pub boxes: usize, pub triangles: usize, pub polyhedra: usize, pub phantoms: usize }

impl ShapeCounts {
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        let mut push = |n: usize, one: &str, many: &str| if n > 0 { parts.push(format!("{n} {}", if n == 1 { one } else { many })) };
        push(self.polyhedra, "polyhedron", "polyhedra");
        push(self.boxes, "box", "boxes");
        push(self.spheres, "sphere", "spheres");
        push(self.pills, "pill", "pills");
        push(self.triangles, "triangle", "triangles");
        if parts.is_empty() { "no shape".to_string() } else { parts.join(", ") }
    }
}

struct Walker<'a> {
    b: &'a PhmoBlocks,
    l: &'a PhmoLayout,
    /// Definition-order start of each polyhedron's four-vector run.
    poly_fv_start: Vec<usize>,
    counts: ShapeCounts,
}

impl Walker<'_> {
    fn emit(&mut self, ty: u16, idx: i16, m: &Mat4, depth: usize, out: &mut Vec<ConvexShape>) {
        if idx < 0 || depth > MAX_DEPTH { return; }
        let i = idx as usize;
        match ty {
            SHAPE_SPHERE => {
                let Some(e) = elem(&self.b.spheres, i, SPHERE_ELEM) else { return };
                if let Some(s) = ConvexShape::sphere(m.transform_point3(vec3_at(e, 0xA0)), radius(e), material(e)) {
                    out.push(s);
                    self.counts.spheres += 1;
                }
            }
            SHAPE_MULTI_SPHERE => {
                let Some(e) = elem(&self.b.multi_spheres, i, MULTI_SPHERE_ELEM) else { return };
                let cnt = i32_at(e, 0x40).clamp(0, 8) as usize;
                let mat = material(e);
                for k in 0..cnt {
                    let s = 0x50 + k * 0x10;
                    if let Some(sh) = ConvexShape::sphere(m.transform_point3(vec3_at(e, s)), f32_at(e, s + 0x0C), mat) {
                        out.push(sh);
                        self.counts.spheres += 1;
                    }
                }
            }
            SHAPE_PILL => {
                let Some(e) = elem(&self.b.pills, i, PILL_ELEM) else { return };
                if let Some(s) = ConvexShape::capsule(m.transform_point3(vec3_at(e, 0x50)), m.transform_point3(vec3_at(e, 0x60)), radius(e), material(e)) {
                    out.push(s);
                    self.counts.pills += 1;
                }
            }
            SHAPE_BOX => {
                let Some(e) = elem(&self.b.boxes, i, BOX_ELEM) else { return };
                let h = vec3_at(e, 0x50);
                if !h.is_finite() || h.abs().max_element() > 1.0e5 { return; }
                let (ri, rj, rk) = (vec3_at(e, 0xA0), vec3_at(e, 0xB0), vec3_at(e, 0xC0));
                let t = vec3_at(e, 0xD0);
                let t = if t.is_finite() { t } else { Vec3::ZERO };
                let (ri, rj, rk) = if is_frame(ri, rj, rk) { (ri, rj, rk) } else { (Vec3::X, Vec3::Y, Vec3::Z) };
                let mut pts = Vec::with_capacity(8);
                for k in 0..8u32 {
                    let s = Vec3::new(if k & 4 != 0 { 1.0 } else { -1.0 }, if k & 2 != 0 { 1.0 } else { -1.0 }, if k & 1 != 0 { 1.0 } else { -1.0 }) * h.abs();
                    pts.push(m.transform_point3(ri * s.x + rj * s.y + rk * s.z + t));
                }
                if let Some(s) = ConvexShape::hull(&pts, radius(e), material(e)) {
                    out.push(s);
                    self.counts.boxes += 1;
                }
            }
            SHAPE_TRIANGLE => {
                let Some(e) = elem(&self.b.triangles, i, TRIANGLE_ELEM) else { return };
                let pts = [m.transform_point3(vec3_at(e, 0x50)), m.transform_point3(vec3_at(e, 0x60)), m.transform_point3(vec3_at(e, 0x70))];
                if let Some(s) = ConvexShape::hull(&pts, radius(e), material(e)) {
                    out.push(s);
                    self.counts.triangles += 1;
                }
            }
            SHAPE_POLYHEDRON => {
                let Some(e) = elem(&self.b.polyhedra, i, self.l.poly_elem) else { return };
                let nv = i32_at(e, 0x80);
                if nv <= 0 || nv as usize > MAX_POLY_VERTS { return; }
                let nv = nv as usize;
                let fvs = i32_at(e, 0x78).max(0) as usize;
                let fvs = if fvs > 0 { fvs } else { nv.div_ceil(4) };
                let nfv = self.b.four_vectors.len() / FOUR_VECTOR_ELEM;
                let start = self.poly_fv_start.get(i).copied().unwrap_or(0);
                if start + fvs > nfv { return; }
                let mut pts: Vec<Vec3> = Vec::with_capacity(nv);
                'outer: for k in 0..fvs {
                    let f = (start + k) * FOUR_VECTOR_ELEM;
                    let fv = &self.b.four_vectors;
                    for j in 0..4 {
                        if pts.len() >= nv { break 'outer; }
                        let p = Vec3::new(f32_at(fv, f + j * 4), f32_at(fv, f + 0x10 + j * 4), f32_at(fv, f + 0x20 + j * 4));
                        if p.is_finite() { pts.push(m.transform_point3(p)); }
                    }
                }
                if let Some(s) = ConvexShape::hull(&pts, radius(e), material(e)) {
                    out.push(s);
                    self.counts.polyhedra += 1;
                }
            }
            SHAPE_LIST => {
                let Some(_) = elem(&self.b.lists, i, LIST_ELEM) else { return };
                let mut start = 0usize;
                for k in 0..i { start += i32_at(&self.b.lists, k * LIST_ELEM + 0x38).max(0) as usize; }
                let cnt = i32_at(&self.b.lists, i * LIST_ELEM + 0x38).max(0) as usize;
                let nls = self.b.list_shapes.len() / LIST_SHAPE_ELEM;
                for k in start..(start + cnt).min(nls) {
                    let e = k * LIST_SHAPE_ELEM;
                    let (ty, ix) = (u16_at(&self.b.list_shapes, e), i16_at(&self.b.list_shapes, e + 2));
                    self.emit(ty, ix, m, depth + 1, out);
                }
            }
            SHAPE_MOPP => {
                let Some(e) = elem(&self.b.mopps, i, MOPP_ELEM) else { return };
                let (ty, ix) = (u16_at(e, 0x58), i16_at(e, 0x5A));
                self.emit(ty, ix, m, depth + 1, out);
            }
            6 => self.counts.phantoms += 1,
            _ => {}
        }
    }
}

/// Walk every rigid body of a `phmo` into settle shapes (model space via `nodes`, the `mode`
/// node matrices index-parallel to the phmo's rigid-body node indices). Returns the bodies and
/// the shape histogram.
pub fn rigid_bodies(b: &PhmoBlocks, l: &PhmoLayout, nodes: &[Mat4]) -> (Vec<RigidBodyDef>, ShapeCounts) {
    let npoly = b.polyhedra.len() / l.poly_elem.max(1);
    let mut poly_fv_start = Vec::with_capacity(npoly);
    let mut cursor = 0usize;
    for i in 0..npoly {
        poly_fv_start.push(cursor);
        cursor += i32_at(&b.polyhedra, i * l.poly_elem + 0x78).max(0) as usize;
    }
    let mut w = Walker { b, l, poly_fv_start, counts: ShapeCounts::default() };
    let nrb = (b.rigid_bodies.len() / l.rb_elem).min(MAX_RIGID_BODIES);
    let mut out = Vec::with_capacity(nrb);
    for i in 0..nrb {
        let e = &b.rigid_bodies[i * l.rb_elem..(i + 1) * l.rb_elem];
        let node = i16_at(e, 0);
        let m = node_matrix(nodes, node);
        let mut shapes = Vec::new();
        w.emit(u16_at(e, l.rb_shape_type), i16_at(e, l.rb_shape_index), &m, 0, &mut shapes);
        let com = vec3_at(e, l.rb_com);
        let com = (com.is_finite() && com != Vec3::ZERO).then(|| m.transform_point3(com));
        let rows = [vec3_at(e, l.rb_inertia), vec3_at(e, l.rb_inertia + 0x10), vec3_at(e, l.rb_inertia + 0x20)];
        let inertia = (rows.iter().all(|r| r.is_finite()) && rows.iter().any(|r| *r != Vec3::ZERO)).then(|| {
            // rows are in node space: I_model = B I Bᵀ with B the node rotation
            let t = Mat3::from_cols(rows[0], rows[1], rows[2]).transpose();
            let bm = Mat3::from_mat4(m);
            bm * t * bm.transpose()
        });
        let s_all = f32_at(e, 0x20);
        let s_xyz = vec3_at(e, 0x24);
        let inertia_scale = Vec3::new(
            if s_xyz.x > 0.0 { s_xyz.x } else { s_all },
            if s_xyz.y > 0.0 { s_xyz.y } else { s_all },
            if s_xyz.z > 0.0 { s_xyz.z } else { s_all },
        );
        out.push(RigidBodyDef {
            shapes,
            mass: f32_at(e, l.rb_mass),
            com,
            inertia,
            inertia_scale,
            lin_damping: f32_at(e, 0x30),
            ang_damping: f32_at(e, 0x34),
            motion_type: u8_at(e, 0x1C),
            flags: u16_at(e, 0x18),
            node,
        });
    }
    (out, w.counts)
}

// ---- constraints (#ragdoll) ------------------------------------------------------------------

/// The per-type tail of a `phmo` constraint. The angles are radians as the tag stores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstraintKind {
    /// A free hinge about the constraint frame's forward axis (no tail fields).
    Hinge,
    /// A hinge limited to `limit` radians about the frame's forward axis.
    LimitedHinge { friction: f32, limit: (f32, f32) },
    /// Havok's ragdoll constraint: a twist about the frame forward, a cone the forward axis may
    /// leave the parent frame by, and a plane limit on the swing about the frame's up axis.
    Ragdoll { twist: (f32, f32), cone: (f32, f32), plane: (f32, f32), friction_torque: f32 },
    /// Point-to-point: the two pivots coincide, orientation free.
    BallSocket,
    StiffSpring { length: f32 },
    Prismatic { limit: (f32, f32), max_friction: f32 },
}

impl ConstraintKind {
    /// The block this kind lives in (offset, element size) and its `constraint type` enum value
    /// as the `node edges` sub-block spells it (0 hinge .. 5 prismatic, 6 powered chain).
    pub fn block(ty: u16) -> Option<(usize, usize)> {
        Some(match ty {
            0 => (OFF_HINGE_CONSTRAINTS, HINGE_CONSTRAINT_ELEM),
            1 => (OFF_LIMITED_HINGE_CONSTRAINTS, LIMITED_HINGE_CONSTRAINT_ELEM),
            2 => (OFF_RAGDOLL_CONSTRAINTS, RAGDOLL_CONSTRAINT_ELEM),
            3 => (OFF_STIFF_SPRING_CONSTRAINTS, STIFF_SPRING_CONSTRAINT_ELEM),
            4 => (OFF_BALL_SOCKET_CONSTRAINTS, BALL_SOCKET_CONSTRAINT_ELEM),
            5 => (OFF_PRISMATIC_CONSTRAINTS, PRISMATIC_CONSTRAINT_ELEM),
            _ => return None,
        })
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            ConstraintKind::Hinge => "hinge",
            ConstraintKind::LimitedHinge { .. } => "limited hinge",
            ConstraintKind::Ragdoll { .. } => "ragdoll",
            ConstraintKind::BallSocket => "ball and socket",
            ConstraintKind::StiffSpring { .. } => "stiff spring",
            ConstraintKind::Prismatic { .. } => "prismatic",
        }
    }
}

/// One authored joint between two `phmo` nodes, as the tag stores it.
///
/// `pivot_a` / `frame_a` are in NODE A's own space and `pivot_b` / `frame_b` in NODE B's, which is
/// how a correct read is proved: at the bind pose the two pivots are the SAME world point
/// (`bind[a] * pivot_a == bind[b] * pivot_b`).
#[derive(Clone, Copy, Debug)]
pub struct ConstraintDef {
    pub name: u32,
    pub node_a: i16,
    pub node_b: i16,
    pub scale_a: f32,
    /// Columns (forward, left, up) of the constraint frame in node A's space.
    pub frame_a: Mat3,
    pub pivot_a: Vec3,
    pub scale_b: f32,
    pub frame_b: Mat3,
    pub pivot_b: Vec3,
    pub edge: i16,
    pub kind: ConstraintKind,
}

impl ConstraintDef {
    /// Whether both frames really are orthonormal bases (the structural gate on the offsets).
    pub fn frames_are_orthonormal(&self) -> bool {
        [self.frame_a, self.frame_b].iter().all(|m| is_frame(m.x_axis, m.y_axis, m.z_axis))
    }
    /// How far apart the two pivots land at the bind pose, in world units. A correct read of a
    /// shipped phmo gives ~0.
    pub fn pivot_residual(&self, nodes: &[Mat4]) -> f32 {
        let a = node_matrix(nodes, self.node_a).transform_point3(self.pivot_a);
        let b = node_matrix(nodes, self.node_b).transform_point3(self.pivot_b);
        (a - b).length()
    }
}

fn read_frame(e: &[u8], o: usize) -> Mat3 {
    Mat3::from_cols(vec3_at(e, o), vec3_at(e, o + 0x0C), vec3_at(e, o + 0x18))
}

/// An authored angle range, converted to RADIANS. #ragdoll
///
/// **The tag stores DEGREES**, which is read off the shipped bipeds and is not a guess: every one
/// of the twenty authored limits on the Spartan and the Elite is a round multiple of five in
/// degrees and physically sensible as one -- the Spartan's elbow is `-75 .. +15`, its knee
/// `-55 .. +15`, the Elite's forearm `-80 .. 0` and its calf `-20 .. +30`, shoulder cones 45 and
/// 50, hip cones 35 and 25, spine cones 10 and 20. Read as radians instead, those same numbers
/// are 4 to 80 radians, i.e. up to twelve whole turns, so every joint would be unlimited -- which
/// is exactly what happened before this conversion was added: the cone clamped to pi and the
/// ragdoll had no joint limits at all.
fn angle_range_at(e: &[u8], o: usize) -> (f32, f32) {
    (f32_at(e, o).to_radians(), f32_at(e, o + 4).to_radians())
}

/// A plain (unitless) pair of reals.
fn range_at(e: &[u8], o: usize) -> (f32, f32) { (f32_at(e, o), f32_at(e, o + 4)) }

/// Decode one constraint element of type `ty` (0 hinge .. 5 prismatic). `None` for a type with no
/// block, a short element, or a frame that is not orthonormal (which is what a wrong offset gives).
pub fn constraint_at(b: &PhmoBlocks, ty: u16, index: i16) -> Option<ConstraintDef> {
    if index < 0 { return None; }
    let (off, size) = ConstraintKind::block(ty)?;
    let blk = b.constraints.get(&off)?;
    let e = elem(blk, index as usize, size)?;
    let kind = match ty {
        0 => ConstraintKind::Hinge,
        1 => ConstraintKind::LimitedHinge { friction: f32_at(e, 0x78), limit: angle_range_at(e, 0x7C) },
        2 => ConstraintKind::Ragdoll {
            twist: angle_range_at(e, 0x78),
            cone: angle_range_at(e, 0x80),
            plane: angle_range_at(e, 0x88),
            friction_torque: f32_at(e, 0x90),
        },
        3 => ConstraintKind::StiffSpring { length: f32_at(e, 0x78) },
        4 => ConstraintKind::BallSocket,
        5 => ConstraintKind::Prismatic { limit: angle_range_at(e, 0x78), max_friction: f32_at(e, 0x80) },
        _ => return None,
    };
    let c = ConstraintDef {
        name: u32_at(e, 0x00),
        node_a: i16_at(e, 0x04),
        node_b: i16_at(e, 0x06),
        scale_a: f32_at(e, 0x08),
        frame_a: read_frame(e, 0x0C),
        pivot_a: vec3_at(e, 0x30),
        scale_b: f32_at(e, 0x3C),
        frame_b: read_frame(e, 0x40),
        pivot_b: vec3_at(e, 0x64),
        edge: i16_at(e, 0x70),
        kind,
    };
    (c.pivot_a.is_finite() && c.pivot_b.is_finite() && c.frames_are_orthonormal()).then_some(c)
}

/// Every constraint of a `phmo`, in block order per type. Elements whose frames are not
/// orthonormal are dropped (and counted by `constraint_reject_count`).
pub fn constraints(b: &PhmoBlocks) -> Vec<ConstraintDef> {
    let mut out = Vec::new();
    for ty in 0..6u16 {
        let Some((off, size)) = ConstraintKind::block(ty) else { continue };
        let Some(blk) = b.constraints.get(&off) else { continue };
        for i in 0..(blk.len() / size).min(1024) {
            if let Some(c) = constraint_at(b, ty, i as i16) { out.push(c); }
        }
    }
    out
}

/// How many constraint elements were present and how many decoded (a gate: a shipped phmo
/// decodes all of them).
pub fn constraint_counts(b: &PhmoBlocks) -> (usize, usize) {
    let per = constraint_counts_per_type(b);
    (per.iter().map(|(_, p, _)| p).sum(), per.iter().map(|(_, _, o)| o).sum())
}

/// Per constraint type, `(type name, elements present, elements decoded)`.
///
/// This is how the shipped tag settles which of the tail blocks are really what the Assembly
/// plugin calls them: only `hinge` and `ragdoll` are derived from the engine's own field order
/// (see the module docs), and a block that is something else entirely shows up here as present
/// but undecoded, because its "frames" are not orthonormal bases.
pub fn constraint_counts_per_type(b: &PhmoBlocks) -> Vec<(&'static str, usize, usize)> {
    const NAMES: [&str; 6] =
        ["hinge", "limited hinge", "ragdoll", "stiff spring", "ball and socket", "prismatic"];
    let mut out = Vec::new();
    for ty in 0..6u16 {
        let Some((off, size)) = ConstraintKind::block(ty) else { continue };
        let Some(blk) = b.constraints.get(&off) else { continue };
        let n = (blk.len() / size).min(1024);
        if n == 0 { continue; }
        let ok = (0..n).filter(|i| constraint_at(b, ty, *i as i16).is_some()).count();
        out.push((NAMES[ty as usize], n, ok));
    }
    out
}

/// One `node edges` element: the two nodes and the (type, index) of every constraint on that
/// edge. The nested per-edge constraint block is NOT read (its pointer lives inside the element,
/// not at a meta offset) -- the topology is in each constraint element already, so the edges are
/// read only for the node pairing and the diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeEdge {
    pub node_a: i16,
    pub node_b: i16,
    pub constraint_count: usize,
}

pub fn node_edges(b: &PhmoBlocks) -> Vec<NodeEdge> {
    let Some(blk) = b.constraints.get(&OFF_NODE_EDGES) else { return Vec::new() };
    (0..(blk.len() / NODE_EDGE_ELEM).min(1024))
        .filter_map(|i| {
            let e = elem(blk, i, NODE_EDGE_ELEM)?;
            Some(NodeEdge {
                node_a: i16_at(e, 0x04),
                node_b: i16_at(e, 0x06),
                constraint_count: i32_at(e, 0x08).clamp(0, 64) as usize,
            })
        })
        .collect()
}

/// `phmo` node names (the tag's own node list, @0x13C), parallel to the `mode` node list.
pub fn phmo_node_names(b: &PhmoBlocks) -> Vec<u32> {
    let Some(blk) = b.constraints.get(&OFF_PHMO_NODES) else { return Vec::new() };
    (0..(blk.len() / PHMO_NODE_ELEM).min(1024)).map(|i| u32_at(blk, i * PHMO_NODE_ELEM)).collect()
}

// ---- vehicle support points (`vehi` havok vehicle physics) ----------------------------------

/// Where a `vehi` keeps its ground-contact points (Assembly `ReachMCC/vehi.xml`,
/// `Halo4MCC/vehi.xml`): (meta offset, element size) of the friction points (wheels: marker
/// sid @0, radius @0x0C) and the anti-gravity points (hover pads: marker sid @0, antigrav
/// strength @0x0C, height @0x10, pad radius @0x24).
pub const REACH_VEHI_FRICTION: (usize, usize) = (0x5C8, 0x34);
pub const REACH_VEHI_ANTIGRAV: (usize, usize) = (0x5BC, 0x48);
pub const H4_VEHI_FRICTION: (usize, usize) = (0x6C8, 0x44);
pub const H4_VEHI_ANTIGRAV: (usize, usize) = (0x6BC, 0x48);

/// A vehicle's ground-contact point as the tag stores it: the marker it sits on and the
/// distance it keeps from the ground (a tyre's radius, a hover pad's antigrav height).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SupportPoint {
    pub marker: u32,
    pub radius: f32,
    /// An anti-gravity point (hover) rather than a wheel.
    pub hover: bool,
}

/// Parse the friction points (`friction`, elements of `friction_elem` bytes) and the
/// anti-gravity points (`antigrav`, 0x48-byte elements) of a `vehi`.
pub fn vehicle_support_points(friction: &[u8], friction_elem: usize, antigrav: &[u8]) -> Vec<SupportPoint> {
    let mut out = Vec::new();
    if friction_elem > 0 {
        for i in 0..(friction.len() / friction_elem).min(64) {
            let e = &friction[i * friction_elem..(i + 1) * friction_elem];
            let (marker, radius) = (u32_at(e, 0), f32_at(e, 0x0C));
            if marker != 0 && marker != 0xFFFF_FFFF && radius.is_finite() && radius > 0.0 && radius < 5.0 {
                out.push(SupportPoint { marker, radius, hover: false });
            }
        }
    }
    // An anti-gravity point rests on its RADIUS (@0x24), not its antigrav height (@0x10): the
    // height is the powered hover range, and an unoccupied hover vehicle sags onto its pads. The
    // Ravine Settler Ghost (pads r 0.2 / 0.175 at z 0.1) sits 0.08 wu above the ground as shipped;
    // the pad spheres give 0.10, the heights (0.35 / 0.30) would float it 0.17 wu too high.
    for i in 0..(antigrav.len() / 0x48).min(64) {
        let e = &antigrav[i * 0x48..(i + 1) * 0x48];
        let (marker, radius) = (u32_at(e, 0), f32_at(e, 0x24));
        if marker != 0 && marker != 0xFFFF_FFFF && radius.is_finite() && radius > 0.0 && radius < 5.0 {
            out.push(SupportPoint { marker, radius, hover: true });
        }
    }
    out
}

fn u32_at(d: &[u8], o: usize) -> u32 { d.get(o..o + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])) }

/// One line for the status bar: "phmo: 1 rigid body (1 polyhedron), mass 3.2".
pub fn describe(defs: &[RigidBodyDef], counts: &ShapeCounts) -> String {
    let n = defs.len();
    let mass: f32 = defs.iter().map(|d| d.mass.max(0.0)).sum();
    let mut s = format!("phmo: {n} rigid {} ({})", if n == 1 { "body" } else { "bodies" }, counts.describe());
    if mass > 0.0 { s.push_str(&format!(", mass {mass:.2}")); }
    // keyframed (4) / fixed (5) bodies and infinite-inertia (flag bit 1) ones are welded in as
    // geometry; say so, since in-game they would not move on their own
    let pinned = defs.iter().filter(|d| d.motion_type >= 4 || d.flags & 0x2 != 0).count();
    if pinned > 0 { s.push_str(&format!(", {pinned} keyframed/fixed")); }
    if let Some(d) = defs.iter().find(|d| d.node >= 0) { s.push_str(&format!(", node {}", d.node)); }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_f32(d: &mut [u8], o: usize, v: f32) { d[o..o + 4].copy_from_slice(&v.to_le_bytes()); }
    fn put_i32(d: &mut [u8], o: usize, v: i32) { d[o..o + 4].copy_from_slice(&v.to_le_bytes()); }
    fn put_i16(d: &mut [u8], o: usize, v: i16) { d[o..o + 2].copy_from_slice(&v.to_le_bytes()); }

    /// A hand-built Reach-layout phmo: one rigid body -> a list of (a box, a sphere).
    #[test]
    fn walks_a_list_of_a_box_and_a_sphere_with_materials() {
        let mut rb = vec![0u8; REACH.rb_elem];
        put_i16(&mut rb, 0, 0);
        put_i16(&mut rb, REACH.rb_shape_type, SHAPE_LIST as i16);
        put_i16(&mut rb, REACH.rb_shape_index, 0);
        put_f32(&mut rb, REACH.rb_mass, 2.5);
        put_f32(&mut rb, REACH.rb_com, 0.1);
        put_f32(&mut rb, REACH.rb_com + 8, 0.2);
        let mut list = vec![0u8; LIST_ELEM];
        put_i32(&mut list, 0x38, 2);
        let mut ls = vec![0u8; LIST_SHAPE_ELEM * 2];
        put_i16(&mut ls, 0, SHAPE_BOX as i16); put_i16(&mut ls, 2, 0);
        put_i16(&mut ls, LIST_SHAPE_ELEM, SHAPE_SPHERE as i16); put_i16(&mut ls, LIST_SHAPE_ELEM + 2, 0);
        let mut bx = vec![0u8; BOX_ELEM];
        put_f32(&mut bx, 0x0C, 0.7); put_f32(&mut bx, 0x10, 0.3); put_f32(&mut bx, 0x18, 1.5);
        put_f32(&mut bx, 0x40, 0.016);
        put_f32(&mut bx, 0x50, 0.484); put_f32(&mut bx, 0x54, 0.984); put_f32(&mut bx, 0x58, 0.184);
        // translation (a Reach box carries one) + identity rotation
        put_f32(&mut bx, 0xA0, 1.0); put_f32(&mut bx, 0xB4, 1.0); put_f32(&mut bx, 0xC8, 1.0);
        put_f32(&mut bx, 0xD0, 0.0); put_f32(&mut bx, 0xD4, 0.0); put_f32(&mut bx, 0xD8, 0.5);
        let mut sp = vec![0u8; SPHERE_ELEM];
        put_f32(&mut sp, 0x0C, 0.4); put_f32(&mut sp, 0x10, 0.5); put_f32(&mut sp, 0x18, 1.0);
        put_f32(&mut sp, 0x40, 0.25);
        put_f32(&mut sp, 0xA0, 2.0);
        let b = PhmoBlocks { rigid_bodies: rb, lists: list, list_shapes: ls, boxes: bx, spheres: sp, ..Default::default() };
        let (defs, counts) = rigid_bodies(&b, &REACH, &[]);
        assert_eq!(defs.len(), 1);
        assert_eq!((counts.boxes, counts.spheres), (1, 1));
        let d = &defs[0];
        assert_eq!(d.shapes.len(), 2);
        assert!((d.mass - 2.5).abs() < 1e-6);
        assert_eq!(d.com, Some(Vec3::new(0.1, 0.0, 0.2)));
        assert_eq!(d.shapes[0].mat, ShapeMaterial { friction: 0.7, restitution: 0.3, mass: 1.5 });
        let (mn, mx) = d.shapes[0].aabb();
        // core half extents + convex radius, translated 0.5 up
        assert!((mx - Vec3::new(0.5, 1.0, 0.7)).length() < 1e-5 && (mn - Vec3::new(-0.5, -1.0, 0.3)).length() < 1e-5, "{mn:?} {mx:?}");
        assert_eq!(d.shapes[1].mat.friction, 0.4);
        match d.shapes[1].geom { crate::physics_settle::ShapeGeom::Sphere { c, r } => { assert_eq!(c, Vec3::new(2.0, 0.0, 0.0)); assert_eq!(r, 0.25); } _ => panic!() }
        assert_eq!(describe(&defs, &counts), "phmo: 1 rigid body (1 box, 1 sphere), mass 2.50, node 0");
    }

    #[test]
    fn polyhedra_consume_the_four_vector_pool_in_order_and_halo4_offsets_apply() {
        // two polyhedra (H4 element size), a tetrahedron then a 5-vertex pyramid: 1 + 2 four-vectors
        let mut polys = vec![0u8; HALO4.poly_elem * 2];
        put_i32(&mut polys, 0x78, 1); put_i32(&mut polys, 0x80, 4);
        put_i32(&mut polys, HALO4.poly_elem + 0x78, 2); put_i32(&mut polys, HALO4.poly_elem + 0x80, 5);
        let tetra = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let pyr = [Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(-1.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 2.0)];
        let mut fv = vec![0u8; FOUR_VECTOR_ELEM * 3];
        let mut put_pt = |elem: usize, j: usize, p: Vec3| {
            put_f32(&mut fv, elem * FOUR_VECTOR_ELEM + j * 4, p.x);
            put_f32(&mut fv, elem * FOUR_VECTOR_ELEM + 0x10 + j * 4, p.y);
            put_f32(&mut fv, elem * FOUR_VECTOR_ELEM + 0x20 + j * 4, p.z);
        };
        for (j, p) in tetra.iter().enumerate() { put_pt(0, j, *p); }
        for (j, p) in pyr.iter().enumerate() { put_pt(1 + j / 4, j % 4, *p); }
        let mut rbs = vec![0u8; HALO4.rb_elem * 2];
        for i in 0..2 {
            put_i16(&mut rbs, i * HALO4.rb_elem + HALO4.rb_shape_type, SHAPE_POLYHEDRON as i16);
            put_i16(&mut rbs, i * HALO4.rb_elem + HALO4.rb_shape_index, i as i16);
            put_f32(&mut rbs, i * HALO4.rb_elem + HALO4.rb_mass, 1.0);
        }
        let b = PhmoBlocks { rigid_bodies: rbs, polyhedra: polys, four_vectors: fv, ..Default::default() };
        let (defs, counts) = rigid_bodies(&b, &HALO4, &[]);
        assert_eq!(counts.polyhedra, 2);
        let (mn, mx) = defs[1].shapes[0].aabb();
        assert!((mn - Vec3::new(-1.0, -1.0, 0.0)).length() < 1e-5 && (mx - Vec3::new(1.0, 1.0, 2.0)).length() < 1e-5, "{mn:?} {mx:?}");
        if let crate::physics_settle::ShapeGeom::Hull { planes, .. } = &defs[1].shapes[0].geom { assert_eq!(planes.len(), 5, "{planes:?}"); } else { panic!() }
    }

    #[test]
    fn node_frames_pose_the_shapes() {
        // one node: inverse frame = 90 deg about z (forward = +y), inverse position (0,0,-1)
        let mut nodes = vec![0u8; REACH_MODE_NODE_ELEM];
        // inverse forward / left / up rows of B: B maps node -> model; pick B = rot z 90: cols (0,1,0),(-1,0,0),(0,0,1)
        // rows of B: (0,-1,0), (1,0,0), (0,0,1)
        put_f32(&mut nodes, 0x2C + 4, -1.0);
        put_f32(&mut nodes, 0x38, 1.0);
        put_f32(&mut nodes, 0x44 + 8, 1.0);
        put_f32(&mut nodes, 0x50 + 8, -1.0);
        let xf = node_xforms(&nodes, REACH_MODE_NODE_ELEM);
        assert_eq!(xf.len(), 1);
        // p_model = B * (l - inv_position): l = (1,0,0) -> B*(1,0,1) = (0,1,1)
        let p = xf[0].transform_point3(Vec3::X);
        assert!((p - Vec3::new(0.0, 1.0, 1.0)).length() < 1e-5, "{p:?}");
        let mut rb = vec![0u8; REACH.rb_elem];
        put_i16(&mut rb, 0, 0);
        put_i16(&mut rb, REACH.rb_shape_type, SHAPE_SPHERE as i16);
        let mut sp = vec![0u8; SPHERE_ELEM];
        put_f32(&mut sp, 0x40, 0.5);
        put_f32(&mut sp, 0xA0, 1.0);
        let b = PhmoBlocks { rigid_bodies: rb, spheres: sp, ..Default::default() };
        let (defs, _) = rigid_bodies(&b, &REACH, &xf);
        match defs[0].shapes[0].geom { crate::physics_settle::ShapeGeom::Sphere { c, .. } => assert!((c - Vec3::new(0.0, 1.0, 1.0)).length() < 1e-5, "{c:?}"), _ => panic!() }
    }

    fn put_u32(d: &mut [u8], o: usize, v: u32) { d[o..o + 4].copy_from_slice(&v.to_le_bytes()); }

    /// Build one constraint element: a frame on each side, a pivot on each side, and a tail.
    fn constraint_elem(size: usize, node_a: i16, node_b: i16, pa: Vec3, pb: Vec3, fb: Mat3) -> Vec<u8> {
        let mut e = vec![0u8; size];
        put_u32(&mut e, 0x00, 0x1234_5678);
        put_i16(&mut e, 0x04, node_a);
        put_i16(&mut e, 0x06, node_b);
        put_f32(&mut e, 0x08, 1.0);
        // A frame = identity (forward/left/up as the three columns)
        for (k, c) in [Vec3::X, Vec3::Y, Vec3::Z].iter().enumerate() {
            put_f32(&mut e, 0x0C + k * 0x0C, c.x);
            put_f32(&mut e, 0x0C + k * 0x0C + 4, c.y);
            put_f32(&mut e, 0x0C + k * 0x0C + 8, c.z);
        }
        for (k, v) in [pa.x, pa.y, pa.z].iter().enumerate() { put_f32(&mut e, 0x30 + k * 4, *v); }
        put_f32(&mut e, 0x3C, 1.0);
        for (k, c) in [fb.x_axis, fb.y_axis, fb.z_axis].iter().enumerate() {
            put_f32(&mut e, 0x40 + k * 0x0C, c.x);
            put_f32(&mut e, 0x40 + k * 0x0C + 4, c.y);
            put_f32(&mut e, 0x40 + k * 0x0C + 8, c.z);
        }
        for (k, v) in [pb.x, pb.y, pb.z].iter().enumerate() { put_f32(&mut e, 0x64 + k * 4, *v); }
        put_i16(&mut e, 0x70, 0);
        e
    }

    /// #ragdoll The constraint decode: a ragdoll element's node pair, both frames, both pivots and
    /// the twist / cone / plane limits, at the offsets this module establishes.
    #[test]
    fn ragdoll_constraints_decode_their_nodes_frames_and_limits() {
        let mut e = constraint_elem(RAGDOLL_CONSTRAINT_ELEM, 3, 7, Vec3::new(0.0, 0.0, 0.25), Vec3::new(0.0, 0.0, -0.1), Mat3::IDENTITY);
        // authored in DEGREES, exactly as the shipped bipeds do
        put_f32(&mut e, 0x78, -5.0); put_f32(&mut e, 0x7C, 5.0);    // twist
        put_f32(&mut e, 0x80, 0.0);  put_f32(&mut e, 0x84, 35.0);   // cone
        put_f32(&mut e, 0x88, -20.0); put_f32(&mut e, 0x8C, 10.0);  // plane
        put_f32(&mut e, 0x90, 1.75);                                // max friction torque
        let b = PhmoBlocks {
            constraints: [(OFF_RAGDOLL_CONSTRAINTS, e)].into_iter().collect(),
            ..Default::default()
        };
        let cons = constraints(&b);
        assert_eq!(cons.len(), 1, "one constraint decoded");
        assert_eq!(constraint_counts(&b), (1, 1), "every element present decoded");
        let c = cons[0];
        assert_eq!(c.name, 0x1234_5678);
        assert_eq!((c.node_a, c.node_b), (3, 7));
        assert_eq!(c.pivot_a, Vec3::new(0.0, 0.0, 0.25));
        assert_eq!(c.pivot_b, Vec3::new(0.0, 0.0, -0.1));
        assert!(c.frames_are_orthonormal());
        // ...and come back in RADIANS, which is what the solver needs
        match c.kind {
            ConstraintKind::Ragdoll { twist, cone, plane, friction_torque } => {
                assert!((twist.0 + 5f32.to_radians()).abs() < 1e-6 && (twist.1 - 5f32.to_radians()).abs() < 1e-6, "twist {twist:?}");
                assert!(cone.0 == 0.0 && (cone.1 - 35f32.to_radians()).abs() < 1e-6, "cone {cone:?}");
                assert!((plane.0 + 20f32.to_radians()).abs() < 1e-6 && (plane.1 - 10f32.to_radians()).abs() < 1e-6, "plane {plane:?}");
                assert_eq!(friction_torque, 1.75, "a torque is NOT an angle and is not converted");
                // the gate bites: the raw numbers read as radians are absurd
                assert!(cone.1 < 1.0, "35 degrees is 0.61 rad, not 35 rad");
            }
            k => panic!("wrong kind {k:?}"),
        }
        assert_eq!(c.kind.type_name(), "ragdoll");
    }

    /// #ragdoll THE PROOF the offsets are right: `A position` and `B position` are each in their
    /// own node's space, so at the bind pose they must name the SAME world point. A wrong offset
    /// (or the two read in one space) does not satisfy that, which this asserts both ways.
    #[test]
    fn constraint_pivots_agree_at_the_bind_pose_and_a_wrong_space_does_not() {
        // node 3 is 1 wu up, node 7 is 1.35 wu up: a joint 0.25 above node 3 is 0.1 below node 7
        let bind = vec![Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)), Mat4::IDENTITY, Mat4::IDENTITY,
            Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.0, 0.0, 1.35))];
        let e = constraint_elem(RAGDOLL_CONSTRAINT_ELEM, 3, 7, Vec3::new(0.0, 0.0, 0.25), Vec3::new(0.0, 0.0, -0.1), Mat3::IDENTITY);
        let b = PhmoBlocks { constraints: [(OFF_RAGDOLL_CONSTRAINTS, e)].into_iter().collect(), ..Default::default() };
        let c = constraints(&b)[0];
        assert!(c.pivot_residual(&bind) < 1e-5, "both pivots are the same world point: {}", c.pivot_residual(&bind));
        // the same numbers read in ONE space (what a misread looks like) do not agree
        let mut wrong = c;
        wrong.node_b = 3;
        assert!(wrong.pivot_residual(&bind) > 0.3, "a wrong reading does not agree: {}", wrong.pivot_residual(&bind));
    }

    /// #ragdoll A frame that is not orthonormal is what a WRONG offset gives, and it is rejected
    /// rather than simulated.
    #[test]
    fn a_non_orthonormal_frame_is_rejected() {
        let mut e = constraint_elem(RAGDOLL_CONSTRAINT_ELEM, 0, 1, Vec3::ZERO, Vec3::ZERO, Mat3::IDENTITY);
        // shear B's frame: forward and left no longer perpendicular
        put_f32(&mut e, 0x4C, 0.8);
        let b = PhmoBlocks { constraints: [(OFF_RAGDOLL_CONSTRAINTS, e)].into_iter().collect(), ..Default::default() };
        assert!(constraints(&b).is_empty(), "rejected");
        assert_eq!(constraint_counts(&b), (1, 0), "and counted as present-but-undecoded");
    }

    /// #ragdoll Every constraint type lands in its own block at its own element size, and a
    /// limited hinge's tail is read where the tag puts it.
    #[test]
    fn each_constraint_type_has_its_own_block_and_tail() {
        assert_eq!(ConstraintKind::block(0), Some((OFF_HINGE_CONSTRAINTS, 0x78)));
        assert_eq!(ConstraintKind::block(1), Some((OFF_LIMITED_HINGE_CONSTRAINTS, 0x84)));
        assert_eq!(ConstraintKind::block(2), Some((OFF_RAGDOLL_CONSTRAINTS, 0x94)));
        assert_eq!(ConstraintKind::block(3), Some((OFF_STIFF_SPRING_CONSTRAINTS, 0x7C)));
        assert_eq!(ConstraintKind::block(4), Some((OFF_BALL_SOCKET_CONSTRAINTS, 0x78)));
        assert_eq!(ConstraintKind::block(5), Some((OFF_PRISMATIC_CONSTRAINTS, 0x84)));
        assert_eq!(ConstraintKind::block(6), None, "a powered chain is not one of these blocks");
        // every element size is exactly the end of its own last field
        for (ty, size) in [(0u16, 0x78usize), (1, 0x7C + 8), (2, 0x90 + 4), (3, 0x78 + 4), (4, 0x78), (5, 0x80 + 4)] {
            assert_eq!(ConstraintKind::block(ty).unwrap().1, size, "type {ty}");
        }
        // the Spartan's own elbow: -75 .. +15 DEGREES, no friction
        let mut e = constraint_elem(LIMITED_HINGE_CONSTRAINT_ELEM, 2, 4, Vec3::X, Vec3::NEG_X, Mat3::IDENTITY);
        put_f32(&mut e, 0x78, 0.35);
        put_f32(&mut e, 0x7C, -75.0);
        put_f32(&mut e, 0x80, 15.0);
        let b = PhmoBlocks { constraints: [(OFF_LIMITED_HINGE_CONSTRAINTS, e)].into_iter().collect(), ..Default::default() };
        match constraints(&b)[0].kind {
            ConstraintKind::LimitedHinge { friction, limit } => {
                assert_eq!(friction, 0.35, "a friction is not an angle");
                assert!((limit.0 + 75f32.to_radians()).abs() < 1e-6, "limit min {} rad", limit.0);
                assert!((limit.1 - 15f32.to_radians()).abs() < 1e-6, "limit max {} rad", limit.1);
                assert!(limit.0 > -std::f32::consts::TAU, "an elbow limit is inside one turn, not twelve");
            }
            k => panic!("{k:?}"),
        }
    }

    /// #ragdoll A `node edges` element names the same two nodes and counts its constraints.
    #[test]
    fn node_edges_decode_their_node_pair() {
        let mut e = vec![0u8; NODE_EDGE_ELEM * 2];
        put_i16(&mut e, 0x04, 3);
        put_i16(&mut e, 0x06, 7);
        put_i32(&mut e, 0x08, 1);
        put_i16(&mut e, NODE_EDGE_ELEM + 0x04, 7);
        put_i16(&mut e, NODE_EDGE_ELEM + 0x06, 9);
        put_i32(&mut e, NODE_EDGE_ELEM + 0x08, 2);
        let b = PhmoBlocks { constraints: [(OFF_NODE_EDGES, e)].into_iter().collect(), ..Default::default() };
        assert_eq!(
            node_edges(&b),
            vec![
                NodeEdge { node_a: 3, node_b: 7, constraint_count: 1 },
                NodeEdge { node_a: 7, node_b: 9, constraint_count: 2 },
            ]
        );
    }
}
