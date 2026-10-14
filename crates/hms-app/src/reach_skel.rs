//! Reach `render_model` SKELETON + skinning metadata (#reach-skin).
//!
//! Everything here is pure byte -> math: the caller hands over the raw tag blocks (read through
//! the native DLL's `tag_meta` / `tag_ptr_bytes`), so the decode is unit-testable against the
//! engine-written values in a shipped cache.
//!
//! LAYOUT (from the tool build's OWN reflection: `reverseme tagdefs reach_tag_test.exe mode`,
//! tag_group 'mode' render_model version 3, `render_model_block_struct` 612 bytes):
//!
//!   mode +0x0C  regions block            -> `render_model_region` 16 B
//!   mode +0x30  nodes block              -> `render_model_node`   96 B  (`NODE_ELEM`)
//!   mode +0x3C  marker groups block      -> `render_model_marker_group` 16 B
//!   mode +0x48  materials block          -> `geometry_material` 44 B
//!   mode +0x64  `render geometry` struct -> `s_render_geometry` 168 B
//!     +0x04  meshes block               -> `s_mesh` 92 B
//!     +0x4C  per mesh node map block    -> `s_per_mesh_node_map` 12 B (one per mesh;
//!                                          each is itself a block of `node index` BYTES)
//!     +0x94  api resource (the vertex/index pages the native parser decodes)
//!
//!   `render_model_node` (96 B) - the offsets below are the SHIPPED CACHE layout, read off real
//!   bytes (`hms-app --dump-biped <map> spartans` with HMS_ACTOR_HEX=1) and NOT the field order
//!   `reverseme tagdefs` prints: the reflection lists `inverse scale` last (+0x58), while the
//!   cache serialises it FIRST, at +0x28, pushing the three inverse basis vectors and the
//!   inverse position to +0x2C / +0x38 / +0x44 / +0x50. Two independent facts pin it:
//!   `distance from parent` @+0x5C equals |default translation| exactly (Spartan b_pelvis:
//!   0.37990 vs |(-0.00886, 0, 0.37980)| = 0.37990), and the inverse fields satisfy
//!   `inverse_position == -R^T * t` exactly for the same node. (`phmo_shapes::node_xforms`
//!   already used these offsets.)
//!     +0x00 string id name            +0x04 short parent     +0x06 short first child
//!     +0x08 short next sibling        +0x0A pad 2
//!     +0x0C real point 3d  default translation
//!     +0x18 real quaternion default rotation   (i, j, k, w on disk)
//!     +0x28 real        inverse scale
//!     +0x2C real vector 3d inverse forward     +0x38 inverse left   +0x44 inverse up
//!     +0x50 real point 3d  inverse position
//!     +0x5C real        distance from parent
//!   The inverse fields are the MODEL->NODE matrix of the bind pose in Blam's row-basis form:
//!   the stored forward/left/up are the matrix's COLUMNS in column-vector terms, so
//!     model_to_node = [ invF | invL | invU | inverse_position ]
//!   and that is exactly `bind_model[i].inverse()` (`inv_rows` keeps the historical name).
//!   The tests assert the identity on every node of a shipped Spartan and Elite.
//!
//!   `render_model_marker` (48 B, inside a marker group's `markers` block):
//!     +0x00 char region index   +0x01 char permutation index   +0x02 char NODE INDEX
//!     +0x03 byte flags          +0x04 real point 3d translation
//!     +0x10 real quaternion rotation (i, j, k, w)   +0x20 real scale
//!     +0x24 real vector 3d direction
//!
//! SKINNING. A skinned mesh (`s_mesh.vertex type` 0x02 s_skinned_vertex / 0x06 s_flat_skinned)
//! stores its positions in MODEL space (the bind pose), with four bone slots per vertex:
//! `blendindices` UInt8_4 @ +0x24 and `blendweight` UInt8_N4 @ +0x28 of the 44-byte vertex.
//! The indices are NOT render_model node indices - they index that MESH's entry of the
//! `per mesh node map` block, whose bytes are the node indices. So:
//!     bone(vertex slot k) = per_mesh_node_map[mesh][blendindices[k]]
//! and the palette entry the vertex shader needs is
//!     palette[slot k] = posed_model[bone] * inverse_bind[bone]
//! which is the identity for the bind pose.

use glam::{Mat4, Quat, Vec3, Vec4};

/// `mode` tag-meta offsets of the blocks this module reads (count i32 @off, pointer u32 @off+4).
pub const OFF_REGIONS: usize = 0x0C;
pub const OFF_NODES: usize = 0x30;
pub const OFF_MARKER_GROUPS: usize = 0x3C;
pub const OFF_MATERIALS: usize = 0x48;
/// `render geometry` sub-struct base.
pub const OFF_RENDER_GEOMETRY: usize = 0x64;
/// `render geometry` + 0x04 = the meshes block (one element per native "section").
pub const OFF_MESHES: usize = OFF_RENDER_GEOMETRY + 0x04;
/// `render geometry` + 0x4C = the per-mesh node map block.
pub const OFF_PER_MESH_NODE_MAP: usize = OFF_RENDER_GEOMETRY + 0x4C;

pub const NODE_ELEM: usize = 96;
pub const MARKER_GROUP_ELEM: usize = 16;
pub const MARKER_ELEM: usize = 48;
pub const REGION_ELEM: usize = 16;
pub const MESH_ELEM: usize = 92;
pub const PER_MESH_NODE_MAP_ELEM: usize = 12;

/// Reach vertex-format ids whose vertices carry bone data (`s_skinned_vertex` / `s_flat_skinned`).
pub const VFMT_SKINNED: u32 = 0x02;
pub const VFMT_FLAT_SKINNED: u32 = 0x06;
/// True for the two skinned `mesh_vertex_type_definition` values.
pub fn is_skinned_format(fmt: u32) -> bool { fmt == VFMT_SKINNED || fmt == VFMT_FLAT_SKINNED }

#[inline]
fn f32_at(b: &[u8], o: usize) -> f32 {
    if o + 4 > b.len() { return 0.0; }
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn i16_at(b: &[u8], o: usize) -> i16 {
    if o + 2 > b.len() { return -1; }
    i16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() { return 0; }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn vec3_at(b: &[u8], o: usize) -> Vec3 { Vec3::new(f32_at(b, o), f32_at(b, o + 4), f32_at(b, o + 8)) }
/// A `real quaternion` on disk is (i, j, k, w) — glam's own component order.
#[inline]
fn quat_at(b: &[u8], o: usize) -> Quat {
    let q = Quat::from_xyzw(f32_at(b, o), f32_at(b, o + 4), f32_at(b, o + 8), f32_at(b, o + 12));
    if q.length_squared() > 1e-8 { q.normalize() } else { Quat::IDENTITY }
}

/// One `render_model_node`.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: u32,
    pub parent: i16,
    pub first_child: i16,
    pub next_sibling: i16,
    /// `default translation` (+0x0C) — the node's offset in its PARENT's frame.
    pub translation: Vec3,
    /// `default rotation` (+0x18) — the node's orientation in its PARENT's frame.
    pub rotation: Quat,
    /// `inverse forward / left / up` (+0x2C/+0x38/+0x44) = the COLUMNS of the bind pose's
    /// model->node rotation (Blam stores a basis as forward/left/up rows of a row-vector
    /// matrix, which are the columns of the equivalent column-vector matrix).
    pub inv_rows: [Vec3; 3],
    /// `inverse position` (+0x50) = the translation of the model->node matrix.
    pub inv_position: Vec3,
    pub inverse_scale: f32,
    pub distance_from_parent: f32,
}

impl Node {
    /// The node's LOCAL (parent-relative) bind matrix from the authored default T/R.
    pub fn local_bind(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.translation)
    }
    /// The authored model->node matrix: `[ invF | invL | invU | inverse_position ]`.
    pub fn inverse_bind_authored(&self) -> Mat4 {
        let [f, l, u] = self.inv_rows;
        Mat4::from_cols(
            Vec4::new(f.x, f.y, f.z, 0.0),
            Vec4::new(l.x, l.y, l.z, 0.0),
            Vec4::new(u.x, u.y, u.z, 0.0),
            Vec4::new(self.inv_position.x, self.inv_position.y, self.inv_position.z, 1.0),
        )
    }
    /// Is the authored inverse a real orthonormal frame? (A node the tool never solved
    /// serialises zeros / NaNs; those fall back to inverting the chained bind pose.)
    pub fn inverse_is_frame(&self) -> bool {
        let [f, l, u] = self.inv_rows;
        [f, l, u].iter().all(|v| v.is_finite() && (v.length() - 1.0).abs() < 1e-2)
            && self.inv_position.is_finite()
            && f.dot(l).abs() < 1e-2 && f.dot(u).abs() < 1e-2 && l.dot(u).abs() < 1e-2
    }
}

/// A render_model's node hierarchy.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub nodes: Vec<Node>,
    /// #film-perf  The composition order, computed ONCE when the skeleton is decoded.
    ///
    /// [`Skeleton::pose_model`] needs parents before children, and it used to call
    /// [`Skeleton::topo_order`] -- two `Vec` allocations and a comparison SORT of the whole
    /// skeleton -- on every call. `pose_model` runs once per posed body per frame, and a Theater
    /// frame poses up to thirty bodies of ~130 nodes each, so that sort was being redone ~30
    /// times a frame for an answer that cannot change: the order is a pure function of the node
    /// parents, which are fixed the moment the block is decoded.
    ///
    /// Empty means "not precomputed" (a `Default` or hand-built skeleton), and `pose_model`
    /// falls back to computing it, so the two routes cannot disagree.
    pub topo: Vec<usize>,
}

impl Skeleton {
    /// Decode the `nodes` block bytes (`NODE_ELEM` each).
    pub fn from_block(bytes: &[u8]) -> Skeleton {
        let n = bytes.len() / NODE_ELEM;
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let e = &bytes[i * NODE_ELEM..(i + 1) * NODE_ELEM];
            nodes.push(Node {
                name: u32_at(e, 0x00),
                parent: i16_at(e, 0x04),
                first_child: i16_at(e, 0x06),
                next_sibling: i16_at(e, 0x08),
                translation: vec3_at(e, 0x0C),
                rotation: quat_at(e, 0x18),
                inv_rows: [vec3_at(e, 0x2C), vec3_at(e, 0x38), vec3_at(e, 0x44)],
                inv_position: vec3_at(e, 0x50),
                inverse_scale: f32_at(e, 0x28),
                distance_from_parent: f32_at(e, 0x5C),
            });
        }
        let mut sk = Skeleton { nodes, topo: Vec::new() };
        sk.topo = sk.topo_order();
        sk
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Nodes in an order where every parent precedes its children (so one forward sweep can
    /// compose model matrices). Cycles / out-of-range parents are treated as roots.
    pub fn topo_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut depth = vec![u32::MAX; n];
        for i in 0..n {
            // Walk to the root, bounded by n so a cycle cannot hang.
            let mut cur = i as i32;
            let mut d = 0u32;
            let mut steps = 0;
            while steps <= n {
                let p = self.nodes[cur as usize].parent as i32;
                if p < 0 || p as usize >= n || p == cur { break; }
                cur = p;
                d += 1;
                steps += 1;
            }
            depth[i] = d;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (depth[i], i));
        order
    }

    /// The BIND pose in model space: `model[i] = model[parent] * local_bind(i)`.
    pub fn bind_model(&self) -> Vec<Mat4> { self.pose_model(&self.local_binds()) }

    /// Each node's authored local bind matrix.
    pub fn local_binds(&self) -> Vec<Mat4> { self.nodes.iter().map(|n| n.local_bind()).collect() }

    /// Compose per-node LOCAL matrices up the hierarchy into model space.
    pub fn pose_model(&self, locals: &[Mat4]) -> Vec<Mat4> {
        let n = self.nodes.len();
        let mut out = vec![Mat4::IDENTITY; n];
        // #film-perf  The precomputed order (see [`Skeleton::topo`]); computed here only for a
        // skeleton that was not built by `from_block`, so the answer is identical either way.
        let computed;
        let order: &[usize] = if self.topo.len() == n {
            &self.topo
        } else {
            computed = self.topo_order();
            &computed
        };
        for &i in order {
            let l = locals.get(i).copied().unwrap_or(Mat4::IDENTITY);
            let p = self.nodes[i].parent;
            out[i] = if p >= 0 && (p as usize) < n && (p as usize) != i {
                out[p as usize] * l
            } else {
                l
            };
        }
        out
    }

    /// Each node's model->node bind matrix: the AUTHORED `inverse *` fields when they form a
    /// real frame, else the inverse of the chained bind pose.
    pub fn inverse_bind(&self) -> Vec<Mat4> {
        let chained = self.bind_model();
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, nd)| {
                if nd.inverse_is_frame() {
                    nd.inverse_bind_authored()
                } else {
                    chained[i].inverse()
                }
            })
            .collect()
    }

    /// The GPU bone palette for a pose given in MODEL space: `posed[i] * inverse_bind[i]`.
    /// (A skinned Reach section's vertices are already in model space, so the bind pose's
    /// palette is the identity.)
    pub fn skin_palette(&self, posed_model: &[Mat4]) -> Vec<Mat4> {
        let inv = self.inverse_bind();
        (0..self.nodes.len())
            .map(|i| posed_model.get(i).copied().unwrap_or(Mat4::IDENTITY) * inv[i])
            .collect()
    }

    /// Index of the node whose name string-id is `sid`.
    pub fn node_by_name(&self, sid: u32) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == sid)
    }
}

/// One `render_model_marker` instance.
#[derive(Clone, Copy, Debug)]
pub struct Marker {
    pub region: i32,
    pub permutation: i32,
    /// -1 when the authored byte is 0xFF (no node: the marker lives in model space).
    pub node: i32,
    pub flags: u8,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: f32,
    pub direction: Vec3,
}

impl Marker {
    /// The marker's frame in its NODE's space (the authored rotation + translation; `scale` is
    /// a render hint and is NOT folded in).
    pub fn local(&self) -> Mat4 { Mat4::from_rotation_translation(self.rotation, self.translation) }
}

/// One `render_model_marker_group` = a name plus every instance of that marker.
#[derive(Clone, Debug)]
pub struct MarkerGroup {
    pub name: u32,
    pub markers: Vec<Marker>,
}

/// Decode one marker group's `markers` block bytes.
pub fn markers_from_block(bytes: &[u8]) -> Vec<Marker> {
    let n = bytes.len() / MARKER_ELEM;
    (0..n)
        .map(|i| {
            let e = &bytes[i * MARKER_ELEM..(i + 1) * MARKER_ELEM];
            let sx = |b: u8| if b == 0xFF { -1i32 } else { b as i32 };
            Marker {
                region: sx(e[0]),
                permutation: sx(e[1]),
                node: sx(e[2]),
                flags: e[3],
                translation: vec3_at(e, 0x04),
                rotation: quat_at(e, 0x10),
                scale: f32_at(e, 0x20),
                direction: vec3_at(e, 0x24),
            }
        })
        .collect()
}

/// Pick the marker instance the engine would use for a (region, permutation): an exact match
/// first, then one whose region/permutation is the -1 wildcard, then instance 0.
pub fn pick_marker(markers: &[Marker], region: i32, permutation: i32) -> Option<Marker> {
    let m = |want_r: bool, want_p: bool| {
        markers.iter().find(|m| {
            (if want_r { m.region == region } else { m.region < 0 })
                && (if want_p { m.permutation == permutation } else { m.permutation < 0 })
        })
    };
    m(true, true)
        .or_else(|| m(true, false))
        .or_else(|| m(false, false))
        .or_else(|| markers.first())
        .copied()
}

/// Pack one vertex's four (node index, /255 weight) pairs into `MeshVertex::bone`:
/// `[0]` = `i0 | i1<<8 | i2<<16 | i3<<24`, `[1]` = the matching weights. The skinned mesh
/// shader reads exactly these two u32s at shader location 9 (see `SKIN_FN_WGSL`).
pub fn pack_bone_slots(idx: [u8; 4], wt: [u8; 4]) -> [u32; 2] {
    let p = |b: [u8; 4]| u32::from_le_bytes(b);
    [p(idx), p(wt)]
}

/// The inverse of [`pack_bone_slots`] (used by the tests and the CPU-skinning reference).
pub fn unpack_bone_slots(bone: [u32; 2]) -> ([u8; 4], [u8; 4]) {
    (bone[0].to_le_bytes(), bone[1].to_le_bytes())
}

/// CPU reference of the shader's linear-blend skin — the same math as `SKIN_FN_WGSL`. Used by
/// the tests and by the headless `--dump-biped` sanity check, never on the draw path.
pub fn skin_point(palette: &[Mat4], idx: [u8; 4], wt: [u8; 4], p: Vec3) -> Vec3 {
    let sum: f32 = wt.iter().map(|&w| w as f32).sum();
    if sum <= 0.0 {
        return palette.first().copied().unwrap_or(Mat4::IDENTITY).transform_point3(p);
    }
    let mut acc = Vec3::ZERO;
    for k in 0..4 {
        let w = wt[k] as f32 / sum;
        if w <= 0.0 { continue; }
        let m = palette.get(idx[k] as usize).copied().unwrap_or(Mat4::IDENTITY);
        acc += w * m.transform_point3(p);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Spartan's `b_pelvis` node (index 1 of
    /// `objects\characters\spartans\spartans.render_model`), byte for byte as the shipped cache
    /// stores it — captured with `hms-app --dump-biped <map> spartans` under HMS_ACTOR_HEX=1.
    /// This is the engine-written value the whole 96-byte node layout rests on.
    fn spartan_b_pelvis() -> Vec<u8> {
        const DWORDS: [u32; 24] = [
            0x0000_47DC, 0x0007_0000, 0x0000_FFFF, // name sid | parent 0, first child 7 | sibling -1
            0xBC11_3812, 0x2487_8A4B, 0x3EC2_70E9, // default translation
            0x3F00_0000, 0xBF00_0000, 0x3F00_0000, 0x3F00_0000, // default rotation (i,j,k,w)
            0x3F80_0001,                                        // inverse scale
            0x9A9C_B3B5, 0xBF80_0000, 0xB400_0000,              // inverse forward
            0x18CB_3B00, 0x3400_0000, 0xBF80_0000,              // inverse left
            0x3F80_0000, 0x9AB4_F272, 0x1A94_F274,              // inverse up
            0xBEC2_70E9, 0xBC11_3812, 0x99E2_42DB,              // inverse position
            0x3EC2_7E77,                                        // distance from parent
        ];
        DWORDS.iter().flat_map(|d| d.to_le_bytes()).collect()
    }

    #[test]
    fn node_element_is_96_bytes() {
        assert_eq!(NODE_ELEM, 96);
        assert_eq!(spartan_b_pelvis().len(), NODE_ELEM);
    }

    /// Every field of the shipped `b_pelvis` decodes to the authored value, and the two
    /// INDEPENDENT facts that pin the layout hold: `distance from parent` equals the length of
    /// `default translation`, and the authored inverse is the exact inverse of the bind matrix.
    #[test]
    fn spartan_pelvis_node_decodes_and_its_inverse_inverts_the_bind() {
        let sk = Skeleton::from_block(&spartan_b_pelvis());
        assert_eq!(sk.len(), 1);
        let n = &sk.nodes[0];
        assert_eq!(n.name, 0x47DC);
        assert_eq!((n.parent, n.first_child, n.next_sibling), (0, 7, -1));
        assert!((n.translation.x - -0.0088626).abs() < 1e-6, "tx {}", n.translation.x);
        assert!(n.translation.y.abs() < 1e-6, "ty {}", n.translation.y);
        assert!((n.translation.z - 0.3798).abs() < 1e-4, "tz {}", n.translation.z);
        for (got, want) in n.rotation.to_array().iter().zip([0.5, -0.5, 0.5, 0.5]) {
            assert!((got - want).abs() < 1e-6, "rotation {:?}", n.rotation);
        }
        assert!((n.inverse_scale - 1.0).abs() < 1e-5, "inverse scale {}", n.inverse_scale);
        // distance_from_parent == |default translation| (the offset pin).
        assert!(
            (n.distance_from_parent - n.translation.length()).abs() < 1e-5,
            "distance {} vs |t| {}", n.distance_from_parent, n.translation.length()
        );
        // The authored inverse inverts the bind matrix exactly.
        assert!(n.inverse_is_frame(), "the authored inverse is an orthonormal frame");
        let d = (n.inverse_bind_authored() * n.local_bind() - Mat4::IDENTITY)
            .to_cols_array().iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(d < 1e-5, "inverse * bind deviates from the identity by {d}");
        // ... and `inverse_position == -R^T * t`.
        let r = Mat4::from_quat(n.rotation);
        let want = -(r.transpose().transform_vector3(n.translation));
        assert!((n.inv_position - want).length() < 1e-5, "inv_position {:?} vs {:?}", n.inv_position, want);
    }

    #[test]
    fn skinned_format_ids_match_the_engine_enum() {
        assert!(is_skinned_format(0x02) && is_skinned_format(0x06));
        for f in [0x00u32, 0x01, 0x04, 0x05, 0x0F] {
            assert!(!is_skinned_format(f), "format {f:#x} is not skinned");
        }
    }

    #[test]
    fn bone_slots_round_trip_through_the_packed_u32s() {
        for case in [([0u8, 1, 2, 3], [255u8, 0, 0, 0]), ([81, 44, 7, 0], [128, 64, 40, 23]), ([255, 255, 255, 255], [255, 255, 255, 255])] {
            let packed = pack_bone_slots(case.0, case.1);
            assert_eq!(unpack_bone_slots(packed), case);
        }
    }

    #[test]
    fn identity_palette_is_the_bind_pose() {
        let pal = vec![Mat4::IDENTITY; 4];
        let p = Vec3::new(0.3, -1.25, 2.5);
        let got = skin_point(&pal, [0, 1, 2, 3], [100, 80, 50, 25], p);
        assert!((got - p).length() < 1e-5, "{got:?} vs {p:?}");
        // A zero weight set falls back to slot 0 (what the shader does).
        let got0 = skin_point(&pal, [0, 0, 0, 0], [0, 0, 0, 0], p);
        assert!((got0 - p).length() < 1e-5);
    }

    /// A two-bone chain: skinning a point fully weighted to the child must equal transforming it
    /// by the child's palette matrix, and a 50/50 blend must land halfway.
    #[test]
    fn linear_blend_matches_the_palette() {
        let a = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let b = Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0));
        let pal = vec![a, b];
        let p = Vec3::ZERO;
        assert!((skin_point(&pal, [0, 0, 0, 0], [255, 0, 0, 0], p) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((skin_point(&pal, [1, 0, 0, 0], [255, 0, 0, 0], p) - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5);
        let mid = skin_point(&pal, [0, 1, 0, 0], [128, 127, 0, 0], p);
        assert!(mid.length() < 0.01, "50/50 blend should land near the origin, got {mid:?}");
    }

    /// #film-perf  THE GATE on the cached composition order ([`Skeleton::topo`]).
    ///
    /// `pose_model` used to recompute `topo_order()` -- two allocations and a sort of the whole
    /// skeleton -- on every call, once per posed body per frame. It is now stored at decode time.
    /// A cached topology that disagreed with the live skeleton would compose a child before its
    /// parent and silently mis-pose a body, so this asserts three things:
    ///
    /// * what `from_block` stores IS what `topo_order()` computes, for a skeleton whose parents
    ///   are deliberately in the WRONG order (child before parent, and a cycle);
    /// * `pose_model` gives the identical matrices through the cache and through the fallback;
    /// * a skeleton with NO cached order (a `Default`, or one built by hand) still poses
    ///   correctly, which is what makes the two routes impossible to tell apart.
    ///
    /// Sabotage check: reversing the stored `topo` makes the second assertion fail.
    #[test]
    fn the_cached_composition_order_matches_the_computed_one() {
        // Parents chosen so the natural index order is NOT a valid composition order: node 0 is
        // a child of node 3, node 2 is a child of node 1, node 4 names itself (a cycle, which
        // `topo_order` must treat as a root) and node 5 names a parent out of range.
        let parents: [i16; 6] = [3, -1, 1, 1, 4, 99];
        let mut bytes = vec![0u8; parents.len() * NODE_ELEM];
        for (i, p) in parents.iter().enumerate() {
            let e = &mut bytes[i * NODE_ELEM..(i + 1) * NODE_ELEM];
            e[0x04..0x06].copy_from_slice(&p.to_le_bytes());
            e[0x06..0x08].copy_from_slice(&(-1i16).to_le_bytes()); // first_child
            e[0x08..0x0A].copy_from_slice(&(-1i16).to_le_bytes()); // next_sibling
            // a distinct translation per node, so a wrong order shows up as wrong numbers
            let t = [1.0f32 + i as f32, 2.0 + i as f32, 3.0 + i as f32];
            for (k, v) in t.iter().enumerate() {
                e[0x0C + k * 4..0x10 + k * 4].copy_from_slice(&v.to_le_bytes());
            }
            e[0x24..0x28].copy_from_slice(&1.0f32.to_le_bytes()); // rotation w
            e[0x28..0x2C].copy_from_slice(&1.0f32.to_le_bytes()); // inverse scale
        }
        let sk = Skeleton::from_block(&bytes);
        assert_eq!(sk.nodes.len(), 6);
        assert_eq!(sk.topo, sk.topo_order(), "the stored order must BE the computed order");
        // and the order is really not the identity, or this fixture proves nothing
        assert_ne!(sk.topo, (0..6).collect::<Vec<usize>>(), "the fixture must need reordering");

        let locals: Vec<Mat4> = (0..6).map(|i| Mat4::from_translation(Vec3::splat(i as f32 + 1.0))).collect();
        let cached = sk.pose_model(&locals);
        // The SAME skeleton with the cache emptied takes the fallback path inside `pose_model`.
        let mut bare = sk.clone();
        bare.topo = Vec::new();
        let fallback = bare.pose_model(&locals);
        for i in 0..6 {
            assert_eq!(
                cached[i].to_cols_array().map(f32::to_bits),
                fallback[i].to_cols_array().map(f32::to_bits),
                "node {i}: the cached order and the computed order must pose identically"
            );
        }
        // A parent must really precede its child in the result: node 2's model matrix is node 1's
        // times its own local, which is only true if the order put 1 first.
        assert_eq!(
            cached[2].to_cols_array().map(f32::to_bits),
            (cached[1] * locals[2]).to_cols_array().map(f32::to_bits),
            "node 2 must be composed on its parent (node 1)"
        );
        // An empty skeleton must not panic on either route.
        assert!(Skeleton::default().pose_model(&[]).is_empty());
    }

    /// `pick_marker` reproduces the engine's selection rule (first match; -1 = wildcard).
    #[test]
    fn marker_pick_prefers_an_exact_region_permutation_match() {
        let mk = |r: i32, p: i32, x: f32| Marker {
            region: r, permutation: p, node: 0, flags: 0,
            translation: Vec3::new(x, 0.0, 0.0), rotation: Quat::IDENTITY, scale: 0.01, direction: Vec3::X,
        };
        let list = [mk(-1, -1, 9.0), mk(2, 1, 1.0)];
        assert_eq!(pick_marker(&list, 2, 1).unwrap().translation.x, 1.0);
        assert_eq!(pick_marker(&list, 5, 5).unwrap().translation.x, 9.0);
        assert!(pick_marker(&[], 0, 0).is_none());
    }
}
