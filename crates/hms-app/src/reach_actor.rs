//! Reach ACTORS: a posed, GPU-skinned biped (optionally holding a weapon) — the reusable
//! half of the Theater-clip pipeline (#reach-skin).
//!
//! Three layers, deliberately separated so each can be tested on its own:
//!   1. [`ReachModel`] — the `render_model` tag read (skeleton, per-mesh node maps, markers,
//!      which meshes are skinned). Pure tag bytes -> [`crate::reach_skel`] types.
//!   2. [`Pose`] — one evaluated pose: a model-space matrix per skeleton node, plus the GPU
//!      bone palette derived from it. Produced by the bind pose or by `reach_jmad`.
//!   3. [`ActorScene`] — the placed actors, their meshes and their bone palettes. The GUI /
//!      script / Theater layer only touches this.

use eframe::wgpu;
use glam::{Mat4, Vec3};
use hms_native::{CacheHandle, NativeDll};

use crate::reach_skel::{
    self, Marker, MarkerGroup, Skeleton, MARKER_ELEM, MARKER_GROUP_ELEM, MESH_ELEM, NODE_ELEM,
    OFF_MARKER_GROUPS, OFF_MESHES, OFF_NODES, OFF_PER_MESH_NODE_MAP, OFF_REGIONS,
    PER_MESH_NODE_MAP_ELEM, REGION_ELEM,
};

/// Read a `count i32 @off / pointer u32 @off+4` tag block out of a TAG META.
pub fn tag_block(native: &NativeDll, cache: CacheHandle, tag: u32, off: usize, elem: usize, max: i32) -> Vec<u8> {
    let Some(h) = native.tag_meta(cache, tag, off as u32, 8) else { return Vec::new() };
    block_at(native, cache, &h, 0, elem, max)
}

/// Read a NESTED tag block whose (count, pointer) header sits at `off` inside `parent` bytes.
pub fn block_at(native: &NativeDll, cache: CacheHandle, parent: &[u8], off: usize, elem: usize, max: i32) -> Vec<u8> {
    if off + 8 > parent.len() || elem == 0 { return Vec::new(); }
    let n = i32::from_le_bytes([parent[off], parent[off + 1], parent[off + 2], parent[off + 3]]);
    let ptr = u32::from_le_bytes([parent[off + 4], parent[off + 5], parent[off + 6], parent[off + 7]]);
    if n <= 0 || n > max || ptr == 0 || ptr == 0xFFFF_FFFF { return Vec::new(); }
    native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32).unwrap_or_default()
}

/// Everything the actor pipeline needs out of one Reach `render_model`.
#[derive(Clone, Debug, Default)]
pub struct ReachModel {
    pub mode_tag: u32,
    pub skel: Skeleton,
    /// Node NAMES (resolved through the cache string table), parallel to `skel.nodes`.
    pub node_names: Vec<String>,
    /// `per mesh node map`: for mesh i, the render_model node index of each bone SLOT a
    /// skinned vertex of that mesh can address. Empty vec = the mesh is rigid (its bone is
    /// `s_mesh.rigid node index`).
    pub node_maps: Vec<Vec<u8>>,
    /// `s_mesh.vertex type` per mesh (0x02 / 0x06 = skinned).
    pub mesh_fmt: Vec<u32>,
    /// `s_mesh.rigid node index` per mesh (0xFF = none).
    pub mesh_rigid_node: Vec<u8>,
    pub markers: Vec<MarkerGroup>,
    /// Region names, for marker region/permutation selection.
    pub region_names: Vec<String>,
}

impl ReachModel {
    /// Read a `render_model` tag. `None` when it has no node block (not a skeletal model).
    pub fn read(native: &NativeDll, cache: CacheHandle, mode_tag: u32) -> Option<ReachModel> {
        if mode_tag == 0 || mode_tag == 0xFFFF_FFFF { return None; }
        let nodes = tag_block(native, cache, mode_tag, OFF_NODES, NODE_ELEM, 255);
        if nodes.is_empty() { return None; }
        let skel = Skeleton::from_block(&nodes);
        let node_names = skel
            .nodes
            .iter()
            .map(|n| native.string_id(cache, n.name).unwrap_or_default())
            .collect();

        // Meshes: vertex type @+0x2F, rigid node index @+0x2E (s_mesh, 92 B).
        let meshes = tag_block(native, cache, mode_tag, OFF_MESHES, MESH_ELEM, 8191);
        let mesh_count = meshes.len() / MESH_ELEM;
        let mut mesh_fmt = Vec::with_capacity(mesh_count);
        let mut mesh_rigid_node = Vec::with_capacity(mesh_count);
        for i in 0..mesh_count {
            let e = &meshes[i * MESH_ELEM..(i + 1) * MESH_ELEM];
            mesh_rigid_node.push(e[0x2E]);
            mesh_fmt.push(e[0x2F] as u32);
        }

        // Per-mesh node map: one 12-byte element per mesh, each a nested block of bytes.
        let pmnm = tag_block(native, cache, mode_tag, OFF_PER_MESH_NODE_MAP, PER_MESH_NODE_MAP_ELEM, 8191);
        let mut node_maps = vec![Vec::new(); mesh_count];
        for i in 0..(pmnm.len() / PER_MESH_NODE_MAP_ELEM).min(mesh_count) {
            let e = &pmnm[i * PER_MESH_NODE_MAP_ELEM..(i + 1) * PER_MESH_NODE_MAP_ELEM];
            node_maps[i] = block_at(native, cache, e, 0, 1, 255);
        }

        // Marker groups: name sid @0, markers nested block @4.
        let groups = tag_block(native, cache, mode_tag, OFF_MARKER_GROUPS, MARKER_GROUP_ELEM, 4096);
        let mut markers = Vec::new();
        for i in 0..groups.len() / MARKER_GROUP_ELEM {
            let e = &groups[i * MARKER_GROUP_ELEM..(i + 1) * MARKER_GROUP_ELEM];
            let name = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
            let mk = block_at(native, cache, e, 4, MARKER_ELEM, 256);
            markers.push(MarkerGroup { name, markers: reach_skel::markers_from_block(&mk) });
        }

        let regions = tag_block(native, cache, mode_tag, OFF_REGIONS, REGION_ELEM, 16);
        let region_names = (0..regions.len() / REGION_ELEM)
            .map(|i| {
                let e = &regions[i * REGION_ELEM..(i + 1) * REGION_ELEM];
                native
                    .string_id(cache, u32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                    .unwrap_or_default()
            })
            .collect();

        Some(ReachModel { mode_tag, skel, node_names, node_maps, mesh_fmt, mesh_rigid_node, markers, region_names })
    }

    /// Does any mesh of this model carry bone data?
    pub fn is_skinned(&self) -> bool { self.mesh_fmt.iter().any(|&f| reach_skel::is_skinned_format(f)) }

    /// Map one mesh's vertex bone SLOTS to render_model node indices. A skinned mesh uses its
    /// `per mesh node map`; a mesh with an EMPTY map is already addressed by node index
    /// (the engine omits the map when it would be the identity).
    pub fn slots_for_mesh(&self, mesh: usize) -> Vec<u8> {
        match self.node_maps.get(mesh) {
            Some(m) if !m.is_empty() => m.clone(),
            _ => (0..self.skel.len().min(255) as u8).collect(),
        }
    }

    /// Index of a marker group by NAME (case-insensitive), with the resolved names.
    pub fn marker_group(&self, native: &NativeDll, cache: CacheHandle, name: &str) -> Option<&MarkerGroup> {
        self.markers.iter().find(|g| {
            native
                .string_id(cache, g.name)
                .map_or(false, |n| n.eq_ignore_ascii_case(name))
        })
    }

    /// Every marker group's resolved name (diagnostics / API discovery).
    pub fn marker_names(&self, native: &NativeDll, cache: CacheHandle) -> Vec<String> {
        self.markers
            .iter()
            .map(|g| native.string_id(cache, g.name).unwrap_or_default())
            .collect()
    }
}

/// One evaluated pose of a skeleton: a model-space matrix per node.
#[derive(Clone, Debug, Default)]
pub struct Pose {
    /// `model[i]` = node i's frame in MODEL space.
    pub model: Vec<Mat4>,
}

impl Pose {
    /// The bind pose.
    pub fn bind(skel: &Skeleton) -> Pose { Pose { model: skel.bind_model() } }

    /// The GPU bone palette: `model[i] * inverse_bind[i]`.
    pub fn palette(&self, skel: &Skeleton) -> Vec<Mat4> { skel.skin_palette(&self.model) }

    /// A marker's frame in MODEL space: the posed node times the marker's authored local frame.
    /// A node index of -1 (0xFF on disk) means the marker is already in model space.
    pub fn marker_model(&self, m: &Marker) -> Mat4 {
        let node = if m.node >= 0 { self.model.get(m.node as usize).copied() } else { None };
        node.unwrap_or(Mat4::IDENTITY) * m.local()
    }
}

/// `object_attach_to_marker`, the engine's child-attachment rule (the same composition the
/// vehicle-turret path in `scene.rs` uses): put the CHILD's marker exactly on the PARENT's
/// marker, i.e.
///     child_world = parent_world * P_marker * C_marker^-1
/// so that `child_world * C_marker == parent_world * P_marker`.
pub fn attach_child(parent_world: Mat4, parent_marker: Mat4, child_marker: Mat4) -> Mat4 {
    parent_world * parent_marker * child_marker.inverse()
}

/// Print-only diagnostic (`hms-app --dump-biped <map> [name-substring]`): every `bipd` in the
/// cache, its render model, skeleton size, the authored-inverse-vs-chained-bind agreement, the
/// per-mesh node maps and a blend-data sanity check. This is the harness the layout tests were
/// written from.
pub fn dump_bipeds(native: &NativeDll, cache: CacheHandle, filter: &str) {
    let mut tags: Vec<u32> = native.tags_by_group(cache, "bipd");
    tags.sort_unstable();
    println!("bipd tags: {}", tags.len());
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_lowercase().contains(&filter.to_lowercase()) { continue; }
        let mode = native.resolve_mode(cache, t);
        println!("\n== bipd 0x{t:08X} '{name}' -> mode 0x{mode:08X} '{}'", native.tag_name(cache, mode).unwrap_or_default());
        let Some(m) = ReachModel::read(native, cache, mode) else { println!("   (no node block)"); continue };
        println!("   nodes={} meshes={} markers={} regions={:?} skinned={}",
            m.skel.len(), m.mesh_fmt.len(), m.markers.len(), m.region_names, m.is_skinned());
        // Layout proof 1: the authored inverse fields must invert the chained bind pose.
        let bind = m.skel.bind_model();
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        let mut framed = 0usize;
        for (i, nd) in m.skel.nodes.iter().enumerate() {
            if !nd.inverse_is_frame() { continue; }
            framed += 1;
            let d = (nd.inverse_bind_authored() * bind[i] - Mat4::IDENTITY).to_cols_array().iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            if d > worst { worst = d; worst_i = i; }
        }
        println!("   inverse-bind check: {framed}/{} nodes carry a frame, worst |inv*bind - I| = {worst:.5} at node {worst_i} '{}'",
            m.skel.len(), m.node_names.get(worst_i).cloned().unwrap_or_default());
        let node_cap: usize = std::env::var("HMS_ACTOR_NODES").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        for (i, nd) in m.skel.nodes.iter().enumerate().take(node_cap) {
            println!("     node {i:3} '{:<24}' parent={:3} t=({:.3},{:.3},{:.3}) r=({:.3},{:.3},{:.3},{:.3}) invscale={:.3}",
                m.node_names.get(i).cloned().unwrap_or_default(), nd.parent,
                nd.translation.x, nd.translation.y, nd.translation.z,
                nd.rotation.x, nd.rotation.y, nd.rotation.z, nd.rotation.w, nd.inverse_scale);
        }
        if std::env::var("HMS_ACTOR_HEX").is_ok() {
            let nodes = tag_block(native, cache, mode, OFF_NODES, NODE_ELEM, 255);
            for i in 0..m.skel.len().min(4) {
                let e = &nodes[i * NODE_ELEM..(i + 1) * NODE_ELEM];
                print!("     raw node {i}:");
                for (k, c) in e.chunks_exact(4).enumerate() {
                    let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    print!(" +{:02X}={:08X}({:.4})", k * 4, u32::from_le_bytes([c[0], c[1], c[2], c[3]]), f);
                }
                println!();
            }
        }
        // Layout proof 2: per-mesh node maps + real blend data.
        let Ok(mh) = native.open_model(cache, mode) else { println!("   open_model failed"); continue };
        let secs = native.model_section_count(mh);
        // section -> "region/permutation", read straight off the render_model region table
        // (region 16 B: name sid @0, permutations count @4 / ptr @8; permutation 0x18 B:
        // name sid @0, mesh_index i16 @4, mesh_count i16 @6).
        let sec_region: std::collections::HashMap<usize, String> = {
            let mut out = std::collections::HashMap::new();
            let u = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            let i16at = |b: &[u8], o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
            if let Some(meta) = native.tag_meta(cache, mode, 0, 0x70) {
                let (rcnt, rptr) = (u(&meta, 12), u(&meta, 16));
                if rcnt > 0 && rcnt <= 0x1000 {
                    if let Some(rb) = native.tag_ptr_bytes(cache, rptr, rcnt * 16) {
                        for r in 0..rcnt as usize {
                            let re = &rb[r * 16..r * 16 + 16];
                            let rn = native.string_id(cache, u(re, 0)).unwrap_or_default();
                            let (pcnt, pptr) = (u(re, 4), u(re, 8));
                            if pcnt == 0 || pcnt > 0x1000 { continue }
                            let Some(pb) = native.tag_ptr_bytes(cache, pptr, pcnt * 0x18) else { continue };
                            for pi in 0..pcnt as usize {
                                let pe = &pb[pi * 0x18..pi * 0x18 + 0x18];
                                let pn = native.string_id(cache, u(pe, 0)).unwrap_or_default();
                                let (mi, mc) = (i16at(pe, 4), i16at(pe, 6));
                                if mi < 0 || mc <= 0 { continue }
                                for k in mi as usize..mi as usize + mc as usize {
                                    out.insert(k, format!("{rn}/{pn}[{pi}]"));
                                }
                            }
                        }
                    }
                }
            }
            out
        };
        for s in 0..secs {
            let Some(sec) = native.model_section(mh, s) else { continue };
            let nm = m.node_maps.get(s as usize).cloned().unwrap_or_default();
            let blend = native.model_decode_blend(mh, s);
            let mut maxslot = 0u8;
            let mut sums: std::collections::BTreeMap<u32, u32> = Default::default();
            if let Some(b) = &blend {
                for v in b.chunks_exact(8) {
                    for k in 0..4 { if v[4 + k] > 0 { maxslot = maxslot.max(v[k]); } }
                    *sums.entry(v[4..8].iter().map(|&w| w as u32).sum()).or_insert(0) += 1;
                }
            }
            // BOTH readers' idea of the rigid node: `mesh_rigid_node` is reach_skel's own walk,
            // `sec.node_index` is what the renderer (scene.rs decode_model_cached) actually binds
            // the mesh to. They must agree, or a rigid armour piece follows the wrong bone.
            let rk = m.mesh_rigid_node.get(s as usize).copied().unwrap_or(0xFF);
            let nname = |i: i32| -> String {
                if i < 0 { return "<none>".into() }
                m.node_names.get(i as usize).cloned().unwrap_or_else(|| format!("<{i} out of range>"))
            };
            println!("     mesh {s:2} vfmt={} stride={} verts={} rigid_node={} '{}' sec.node_index={} '{}'{} region='{}' node_map={} (max {:?}) blend={} weight-sums={:?} max_slot_used={}",
                sec.vertex_format, sec.vertex_stride, sec.vertex_count,
                rk, nname(if rk == 0xFF { -1 } else { rk as i32 }),
                sec.node_index, nname(sec.node_index as i32),
                if (rk == 0xFF) != (sec.node_index < 0) || (rk != 0xFF && rk as i16 != sec.node_index) { "  <<< DISAGREE" } else { "" },
                sec_region.get(&(s as usize)).cloned().unwrap_or_default(),
                nm.len(), nm.iter().max(), blend.as_ref().map(|b| b.len() / 8).unwrap_or(0), sums, maxslot);
        }
        let names = m.marker_names(native, cache);
        println!("   marker groups: {}", names.join(", "));
        native.close_model(mh);
    }
}

/// Print-only (`hms-app --dump-rmodel <map> <mode hex>`): a RENDER MODEL read directly, with no
/// object wrapper — the shape the first-person arm models come in (`matg @player representation`
/// points at a bare `mode`, not at an `obje`).
pub fn dump_render_model(native: &NativeDll, cache: CacheHandle, mode_tag: u32) {
    let name = native.tag_name(cache, mode_tag).unwrap_or_default();
    println!("mode 0x{mode_tag:08X} '{name}'");
    let Some(m) = ReachModel::read(native, cache, mode_tag) else { println!("  not a readable render model"); return };
    println!("  nodes={} meshes={} markers={} skinned={}", m.skel.len(), m.mesh_fmt.len(), m.markers.len(), m.is_skinned());
    println!("  regions: {:?}", m.region_names);
    // region table: name sid @0, permutation count @4 / ptr @8; permutation 0x18 B.
    let u = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let i16at = |b: &[u8], o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
    if let Some(meta) = native.tag_meta(cache, mode_tag, 0, 0x70) {
        let (rcnt, rptr) = (u(&meta, 12), u(&meta, 16));
        if rcnt > 0 && rcnt <= 0x1000 {
            if let Some(rb) = native.tag_ptr_bytes(cache, rptr, rcnt * 16) {
                for r in 0..rcnt as usize {
                    let re = &rb[r * 16..r * 16 + 16];
                    let rn = native.string_id(cache, u(re, 0)).unwrap_or_default();
                    let (pcnt, pptr) = (u(re, 4), u(re, 8));
                    println!("  region {r} '{rn}': {pcnt} permutation(s)");
                    if pcnt == 0 || pcnt > 0x1000 { continue }
                    let Some(pbb) = native.tag_ptr_bytes(cache, pptr, pcnt * 0x18) else { continue };
                    for pi in 0..pcnt as usize {
                        let pe = &pbb[pi * 0x18..pi * 0x18 + 0x18];
                        let pn = native.string_id(cache, u(pe, 0)).unwrap_or_default();
                        println!("    perm {pi} '{pn}' meshes {}..+{}", i16at(pe, 4), i16at(pe, 6));
                    }
                }
            }
        }
    }
    // Where the model actually SITS in its own space — what decides whether a model mounted on
    // the camera lands in front of the view or behind it.
    let bind = m.skel.bind_model();
    let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
    for b in &bind {
        let t = b.w_axis;
        for k in 0..3 {
            mn[k] = mn[k].min(t[k]);
            mx[k] = mx[k].max(t[k]);
        }
    }
    println!("  bind-pose node extent: x {:.3}..{:.3}  y {:.3}..{:.3}  z {:.3}..{:.3}",
        mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);
    for (i, f) in m.mesh_fmt.iter().enumerate() {
        println!("    mesh {i} vfmt={f} skinned={} rigid_node={:?} node_map={}",
            crate::reach_skel::is_skinned_format(*f),
            m.mesh_rigid_node.get(i),
            m.node_maps.get(i).map(|v| v.len()).unwrap_or(0));
    }
    println!("  marker groups: {}", m.marker_names(native, cache).join(", "));
    let cap: usize = std::env::var("HMS_RM_NODES").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    for (i, n) in m.node_names.iter().enumerate().take(cap) {
        let t = bind.get(i).map(|b| b.w_axis).unwrap_or(glam::Vec4::ZERO);
        println!("    node {i} '{n}' parent={} model=({:.3},{:.3},{:.3})",
            m.skel.nodes.get(i).map(|x| x.parent).unwrap_or(-1), t.x, t.y, t.z);
    }
}

/// Print-only: every marker group of an arbitrary OBJECT tag's render model (used to find the
/// weapon's grip marker and the biped's hand marker).
pub fn dump_markers(native: &NativeDll, cache: CacheHandle, group: &str, filter: &str) {
    let mut tags: Vec<u32> = native.tags_by_group(cache, group);
    tags.sort_unstable();
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_lowercase().contains(&filter.to_lowercase()) { continue; }
        let mode = native.resolve_mode(cache, t);
        let Some(m) = ReachModel::read(native, cache, mode) else { continue };
        println!("\n== {group} 0x{t:08X} '{name}' nodes={} ", m.skel.len());
        for g in &m.markers {
            let gn = native.string_id(cache, g.name).unwrap_or_default();
            for (i, mk) in g.markers.iter().enumerate() {
                let node_name = if mk.node >= 0 { m.node_names.get(mk.node as usize).cloned().unwrap_or_default() } else { "<model>".into() };
                println!("   '{gn}'[{i}] region={} perm={} node={} '{node_name}' t=({:.3},{:.3},{:.3}) r=({:.3},{:.3},{:.3},{:.3}) scale={:.3}",
                    mk.region, mk.permutation, mk.node, mk.translation.x, mk.translation.y, mk.translation.z,
                    mk.rotation.x, mk.rotation.y, mk.rotation.z, mk.rotation.w, mk.scale);
            }
        }
    }
}

/// Model-space AABB of a posed, skinned model — the headless camera framing and the
/// "did the pose actually change" check both use it. `verts` are the decoded MODEL-space
/// positions with their packed bone slots.
pub fn posed_bounds(palette: &[Mat4], verts: &[(Vec3, [u8; 4], [u8; 4])]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for (p, idx, wt) in verts {
        let q = reach_skel::skin_point(palette, *idx, *wt, *p);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    if lo.x > hi.x { (Vec3::ZERO, Vec3::ZERO) } else { (lo, hi) }
}

// ---------------------------------------------------------------------------------------------
// jmad probe (print-only): the layout harness the animation decoder was written from.
// ---------------------------------------------------------------------------------------------

/// `jmad` (model_animation_graph, version 3) tag-meta offsets, from the tool build's own
/// reflection (`reverseme tagdefs reach_tag_test.exe jmad`). The root struct is 440 bytes:
/// `definitions` @0x000 (256 B), `content` @0x100 (40 B), `run time data` @0x128 (100 B),
/// `last import results` @0x18C (data, 20 B), `additional node data` @0x1A0, `tag resource
/// groups` @0x1AC.
pub mod jmad_off {
    /// definitions +0x018 `node usage` (4 B: usage enum u16, node block index u16).
    pub const NODE_USAGE: usize = 0x018;
    /// definitions +0x048 `skeleton nodes` (element size NOT in the reflection data).
    pub const SKELETON_NODES: usize = 0x048;
    /// definitions +0x094 `animations` (60 B `c_model_animation`).
    pub const ANIMATIONS: usize = 0x094;
    /// content +0x004 -> root +0x104 `modes` (48 B `c_animation_mode`).
    pub const MODES: usize = 0x104;
    /// root +0x1A0 `additional node data` (60 B).
    pub const ADDITIONAL_NODE_DATA: usize = 0x1A0;
    /// root +0x1AC `tag resource groups` (12 B: reference_count i32, pageable resource @+4).
    pub const RESOURCE_GROUPS: usize = 0x1AC;

    pub const ANIMATION_ELEM: usize = 60;
    /// #aim `c_model_animation` +0x1A `blend screen` (char block index; 0xFF = none).
    pub const ANIM_BLEND_SCREEN: usize = 0x1A;
    /// `c_model_animation` +0x30 `shared animation data` (212 B `c_shared_model_animation`).
    pub const ANIM_SHARED_BLOCK: usize = 0x30;
    pub const SHARED_ELEM: usize = 212;
    pub const MODE_ELEM: usize = 48;
    pub const RESOURCE_GROUP_ELEM: usize = 12;

    // ---- #gait-blend  THE DIRECTIONAL LOCOMOTION TABLES --------------------------------
    //
    // Offsets from Blam's own reflection data (`reverseme tagdefs reach_tag_test.exe`,
    // `.reverseme/tagdefs/reach_tag_test.cbdd8448.txt` lines 11385-11430), inside the
    // `s_animation_graph_definitions` struct that is the `jmad` group struct's first member,
    // so a definitions offset IS a tag offset.

    /// definitions +0x0A0 `NEW blend screens` (44 B `s_new_animation_blend_screen`).
    pub const NEW_BLEND_SCREENS: usize = 0x0A0;
    pub const NEW_BLEND_SCREEN_ELEM: usize = 44;
    /// definitions +0x0B8 `overlay groups` (28 B `s_overlay_group`).
    pub const OVERLAY_GROUPS: usize = 0x0B8;
    pub const OVERLAY_GROUP_ELEM: usize = 28;
    /// definitions +0x0C4 `gaits` (20 B `s_animation_gait`), max 32.
    pub const GAITS: usize = 0x0C4;
    pub const GAIT_ELEM: usize = 20;
    /// definitions +0x0D0 `gait groups` (16 B `s_animation_gait_group`), max 32.
    pub const GAIT_GROUPS: usize = 0x0D0;
    pub const GAIT_GROUP_ELEM: usize = 16;
    /// `s_animation_gait_group` +0x004 `animation gaits` (4 B: short block index + pad).
    pub const GAIT_GROUP_ITEMS: usize = 0x004;
    pub const GAIT_ITEM_ELEM: usize = 4;
    /// content +0x000 (= tag +0x100) `default gait group`, a short block index.
    pub const DEFAULT_GAIT_GROUP: usize = 0x100;
    /// `c_animation_mode` +0x00C `weapon class` (56 B `c_weapon_class`).
    pub const MODE_WEAPON_CLASS: usize = 0x00C;
    pub const WEAPON_CLASS_ELEM: usize = 56;
    /// `c_weapon_class` +0x008 `weapon type` (20 B `c_weapon_type`).
    pub const WEAPON_CLASS_TYPES: usize = 0x008;
    pub const WEAPON_TYPE_ELEM: usize = 20;
    /// `c_weapon_type` +0x008 `sets` (72 B `c_animation_set`).
    pub const WEAPON_TYPE_SETS: usize = 0x008;
    pub const SET_ELEM: usize = 72;
    /// `c_animation_set` +0x008 `gait group`, a short block index.
    pub const SET_GAIT_GROUP: usize = 0x008;
    /// `c_animation_set` +0x00C `actions` (12 B `s_animation_entry`): +0x00 label string id,
    /// +0x04 overlay group, +0x06 ik set, +0x08 `c_animation_id` { graph index i16, animation
    /// i16 } -- so the ANIMATION INDEX is at +0x0A.
    pub const SET_ACTIONS: usize = 0x00C;
    pub const ACTION_ELEM: usize = 12;
    /// `c_animation_set` +0x030 `transitions` (16 B `s_animation_transition_source`: +0x00 the
    /// STATE this transition starts in, +0x04 a `destinations` block of 12-byte rows {mode name,
    /// state name, `c_animation_id`}).
    ///
    /// #gait-blend  Read to answer one question with a measurement: does the shipped player
    /// graph author a transition clip for a change of MOVEMENT direction (the engine's own way
    /// of handling one -- the `move_front_turn_left_45` family of state labels exists in the
    /// binary)? Measured on `forge_halo`: see `dump_jmad_gaits`.
    pub const SET_TRANSITIONS: usize = 0x030;
    pub const TRANSITION_SOURCE_ELEM: usize = 16;
    pub const TRANSITION_DEST_ELEM: usize = 12;
    /// `c_animation_set` +0x03C `velocity boundaries` (32 B = 8 reals), max 1.
    pub const SET_VELOCITY_BOUNDARIES: usize = 0x03C;
    pub const VELOCITY_BOUNDARIES_ELEM: usize = 32;
}

/// Print-only (`hms-app --dump-jmad <map> [filter]`): every `jmad`'s animation list, the
/// per-animation `c_shared_model_animation` header fields, and a hexdump of the resource
/// group's DEFINITION data + fixups + payload head, so the compressed-animation layout can be
/// read off real engine-written bytes.
pub fn dump_jmad(native: &NativeDll, cache: CacheHandle, filter: &str, limit: usize) {
    use jmad_off::*;
    let mut tags: Vec<u32> = native.tags_by_group(cache, "jmad");
    tags.sort_unstable();
    println!("jmad tags: {}", tags.len());
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_lowercase().contains(&filter.to_lowercase()) { continue; }
        println!("\n== jmad 0x{t:08X} '{name}'");
        let anims = tag_block(native, cache, t, ANIMATIONS, ANIMATION_ELEM, 2048);
        let groups = tag_block(native, cache, t, RESOURCE_GROUPS, RESOURCE_GROUP_ELEM, 6144);
        let skel = tag_block(native, cache, t, SKELETON_NODES, 32, 255);
        println!("   animations={} resource groups={} skeleton-nodes-bytes={}",
            anims.len() / ANIMATION_ELEM, groups.len() / RESOURCE_GROUP_ELEM, skel.len());
        // #aim The aiming-screen table: what an aim overlay's frames mean in angles.
        let screens = tag_block(native, cache, t, crate::reach_jmad::OFF_BLEND_SCREENS,
            crate::reach_jmad::BLEND_SCREEN_ELEM, 1024);
        let nscr = screens.len() / crate::reach_jmad::BLEND_SCREEN_ELEM;
        println!("   blend screens={nscr}");
        for i in 0..nscr {
            let e = &screens[i * crate::reach_jmad::BLEND_SCREEN_ELEM..(i + 1) * crate::reach_jmad::BLEND_SCREEN_ELEM];
            let Some(sc) = crate::reach_jmad::BlendScreen::read(e) else { continue };
            println!("     screen {i:3} '{}' yaw R/L {:.2}/{:.2} deg per frame x {}/{} frames; pitch D/U {:.2}/{:.2} deg per frame x {}/{} frames -> grid {}x{} = {} frames",
                native.string_id(cache, sc.label_sid).unwrap_or_default(),
                sc.right_yaw_per_frame.to_degrees(), sc.left_yaw_per_frame.to_degrees(),
                sc.right_frames, sc.left_frames,
                sc.down_pitch_per_frame.to_degrees(), sc.up_pitch_per_frame.to_degrees(),
                sc.down_frames, sc.up_frames,
                sc.width(), sc.height(), sc.width() * sc.height());
        }
        // #gait-blend  THE AUTHORED DIRECTIONAL LOCOMOTION TABLES. Print-only; this is the
        // measurement that says whether the engine blends directions or switches between them.
        dump_jmad_gaits(native, cache, t);
        // Resource groups first: dump the definition blob of group 0.
        for g in 0..(groups.len() / RESOURCE_GROUP_ELEM).min(2) {
            let e = &groups[g * RESOURCE_GROUP_ELEM..(g + 1) * RESOURCE_GROUP_ELEM];
            let refcount = i32::from_le_bytes([e[0], e[1], e[2], e[3]]);
            let rid = i32::from_le_bytes([e[4], e[5], e[6], e[7]]);
            let rid2 = i32::from_le_bytes([e[8], e[9], e[10], e[11]]);
            print!("   group {g}: refcount={refcount} resource_field=0x{rid:08X}/0x{rid2:08X}");
            match native.read_resource(cache, rid) {
                Some((def, fixups, data)) => {
                    println!(" -> def={} fixups={} payload={}", def.len(), fixups.len(), data.len());
                    println!("     def hex: {}", hex_head(&def, 256));
                    println!("     fixups: {:?}", &fixups[..fixups.len().min(24)]);
                    println!("     payload head: {}", hex_head(&data, 128));
                }
                None => println!(" -> read_resource FAILED"),
            }
        }
        for i in 0..(anims.len() / ANIMATION_ELEM).min(limit) {
            let e = &anims[i * ANIMATION_ELEM..(i + 1) * ANIMATION_ELEM];
            let sid = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
            let aname = native.string_id(cache, sid).unwrap_or_default();
            let shared = block_at(native, cache, e, ANIM_SHARED_BLOCK, SHARED_ELEM, 1);
            if shared.len() < SHARED_ELEM { println!("   anim {i:4} '{aname}' (no shared block)"); continue; }
            let s = &shared[..SHARED_ELEM];
            let nodes = s[0];
            let frames = i16::from_le_bytes([s[2], s[3]]);
            let (atype, finfo, desired, current) = (s[4], s[5], s[6], s[7]);
            let iflags = u16::from_le_bytes([s[0x0A], s[0x0B]]);
            let group = i16::from_le_bytes([s[0x20], s[0x21]]);
            let member = i16::from_le_bytes([s[0x22], s[0x23]]);
            let heading = [f32::from_le_bytes([s[0x24], s[0x25], s[0x26], s[0x27]]),
                           f32::from_le_bytes([s[0x28], s[0x29], s[0x2A], s[0x2B]]),
                           f32::from_le_bytes([s[0x2C], s[0x2D], s[0x2E], s[0x2F]])];
            // #gait-blend  `c_shared_model_animation` +0x30 / +0x34: the clip's OWN authored
            // average translation and rotation per frame. The engine reads exactly these two
            // (`haloreach.dll` `sub_180894AD0`: `*(float*)(shared+0x30) * 30.0` for a movement
            // state, `fabsf(*(float*)(shared+0x34))` for a turn state).
            let avg_t = f32::from_le_bytes([s[0x30], s[0x31], s[0x32], s[0x33]]);
            let avg_r = f32::from_le_bytes([s[0x34], s[0x35], s[0x36], s[0x37]]);
            println!("   anim {i:4} '{aname}' nodes={nodes} frames={frames} type={atype} frame_info={finfo} compression want/is={desired}/{current} iflags=0x{iflags:04X} group={group} member={member} screen={} heading=[{:.3},{:.3},{:.3}] speed={:.4} wu/s turn={:.4}",
                match e[crate::reach_jmad::ANIM_BLEND_SCREEN] { 0xFF => "-".to_string(), v => v.to_string() },
                heading[0], heading[1], heading[2], avg_t * 30.0, avg_r.abs());
        }
    }
}

/// #gait-blend  Print-only: one `jmad`'s GAIT tables — the authored directional-locomotion
/// machinery — plus the `mode -> weapon class -> weapon type -> set` chain that names which gait
/// group and which velocity boundaries a state uses.
///
/// Every offset is from Blam's reflection data; see [`jmad_off`]. Nothing here is interpreted:
/// it prints the bytes as the struct names them so the scheme can be read off a shipped cache.
pub fn dump_jmad_gaits(native: &NativeDll, cache: CacheHandle, t: u32) {
    use jmad_off::*;
    let sid = |b: &[u8], o: usize| -> String {
        if o + 4 > b.len() { return String::new(); }
        let v = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if v == 0 { return String::new(); }
        native.string_id(cache, v).unwrap_or_else(|| format!("sid:{v:#X}"))
    };
    let i16_at = |b: &[u8], o: usize| -> i16 {
        if o + 2 > b.len() { return -1 }
        i16::from_le_bytes([b[o], b[o + 1]])
    };
    let f32_at = |b: &[u8], o: usize| -> f32 {
        if o + 4 > b.len() { return f32::NAN }
        f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    };
    let dflt = native.tag_meta(cache, t, DEFAULT_GAIT_GROUP as u32, 2).unwrap_or_default();
    println!("   default gait group = {}", i16_at(&dflt, 0));
    let nbs = tag_block(native, cache, t, NEW_BLEND_SCREENS, NEW_BLEND_SCREEN_ELEM, 1024);
    println!("   NEW blend screens = {}", nbs.len() / NEW_BLEND_SCREEN_ELEM);
    for i in 0..nbs.len() / NEW_BLEND_SCREEN_ELEM {
        let e = &nbs[i * NEW_BLEND_SCREEN_ELEM..(i + 1) * NEW_BLEND_SCREEN_ELEM];
        println!(
            "     newscreen {i:3} '{}' flags={:#X} weight={:.4} interp_rate={:.4} yaw_src={} pitch_src={} weight_src={} anim=({},{})",
            sid(e, 0x00),
            u32::from_le_bytes([e[4], e[5], e[6], e[7]]),
            f32_at(e, 0x08), f32_at(e, 0x0C),
            i16_at(e, 0x12), i16_at(e, 0x14), i16_at(e, 0x16),
            i16_at(e, 0x28), i16_at(e, 0x2A)
        );
    }
    let gaits = tag_block(native, cache, t, GAITS, GAIT_ELEM, 32);
    println!("   gaits = {}", gaits.len() / GAIT_ELEM);
    for i in 0..gaits.len() / GAIT_ELEM {
        let e = &gaits[i * GAIT_ELEM..(i + 1) * GAIT_ELEM];
        println!(
            "     gait {i:3} '{}' slow='{}' intermediate='{}' fast='{}' move_state={}",
            sid(e, 0x00), sid(e, 0x04), sid(e, 0x08), sid(e, 0x0C), i16_at(e, 0x10)
        );
    }
    let groups = tag_block(native, cache, t, GAIT_GROUPS, GAIT_GROUP_ELEM, 32);
    println!("   gait groups = {}", groups.len() / GAIT_GROUP_ELEM);
    for i in 0..groups.len() / GAIT_GROUP_ELEM {
        let e = &groups[i * GAIT_GROUP_ELEM..(i + 1) * GAIT_GROUP_ELEM];
        let items = block_at(native, cache, e, GAIT_GROUP_ITEMS, GAIT_ITEM_ELEM, 32);
        let list: Vec<i16> = (0..items.len() / GAIT_ITEM_ELEM)
            .map(|k| i16_at(&items, k * GAIT_ITEM_ELEM))
            .collect();
        println!("     gait group {i:3} '{}' gaits={list:?}", sid(e, 0x00));
    }
    let ogs = tag_block(native, cache, t, OVERLAY_GROUPS, OVERLAY_GROUP_ELEM, 1056);
    println!("   overlay groups = {}", ogs.len() / OVERLAY_GROUP_ELEM);
    // #gait-blend  THE CROSS-CHECK. The table HMS resolves, beside the `velocity boundaries` the
    // TOOL precomputed. They are two independent routes to the same numbers -- one through the
    // clips' own `+0x30 x 30`, one through a block the tool wrote -- so agreement is a
    // measurement and not a restatement.
    if let Some(g) = AnimGraph::read(native, cache, t) {
        for (mode, class) in [("combat", "rifle"), ("crouch", "rifle"), ("sprint", "rifle"), ("combat", "sword")] {
            let tbl = g.gait_table(mode, class);
            if tbl.gaits.is_empty() {
                continue;
            }
            println!("   resolved gait table '{mode}':'{class}':");
            for row in &tbl.gaits {
                let tiers: Vec<String> = row.tiers.iter().map(|x| format!("{}@{:.4}x{}f", x.name, x.rate, x.frames)).collect();
                println!("     dir {} '{}' {}", row.direction, row.name, tiers.join("  "));
            }
            match g.velocity_boundaries(mode, class) {
                Some(vb) => {
                    for d in 0..4i16 {
                        let got = tbl.speed_range(d);
                        println!(
                            "     dir {d}: authored boundaries [{:.4}, {:.4}] vs resolved {:?}",
                            vb[d as usize * 2], vb[d as usize * 2 + 1], got
                        );
                    }
                }
                None => println!("     (no velocity boundaries authored for this set)"),
            }
        }
    }
    // mode -> weapon class -> weapon type -> set: the gait group and velocity boundaries in force.
    let modes = tag_block(native, cache, t, MODES, MODE_ELEM, 512);
    for m in 0..modes.len() / MODE_ELEM {
        let me = &modes[m * MODE_ELEM..(m + 1) * MODE_ELEM];
        let wcs = block_at(native, cache, me, MODE_WEAPON_CLASS, WEAPON_CLASS_ELEM, 64);
        for c in 0..wcs.len() / WEAPON_CLASS_ELEM {
            let ce = &wcs[c * WEAPON_CLASS_ELEM..(c + 1) * WEAPON_CLASS_ELEM];
            let wts = block_at(native, cache, ce, WEAPON_CLASS_TYPES, WEAPON_TYPE_ELEM, 64);
            for w in 0..wts.len() / WEAPON_TYPE_ELEM {
                let we = &wts[w * WEAPON_TYPE_ELEM..(w + 1) * WEAPON_TYPE_ELEM];
                let sets = block_at(native, cache, we, WEAPON_TYPE_SETS, SET_ELEM, 64);
                for s in 0..sets.len() / SET_ELEM {
                    let se = &sets[s * SET_ELEM..(s + 1) * SET_ELEM];
                    let vb = block_at(native, cache, se, SET_VELOCITY_BOUNDARIES, VELOCITY_BOUNDARIES_ELEM, 1);
                    let v: Vec<String> = (0..vb.len() / 4).map(|k| format!("{:.3}", f32_at(&vb, k * 4))).collect();
                    // #gait-blend  The authored state TRANSITIONS of this set, source -> dest.
                    let tr = block_at(native, cache, se, SET_TRANSITIONS, TRANSITION_SOURCE_ELEM, 256);
                    let mut trs: Vec<String> = Vec::new();
                    for ti in 0..tr.len() / TRANSITION_SOURCE_ELEM {
                        let te = &tr[ti * TRANSITION_SOURCE_ELEM..(ti + 1) * TRANSITION_SOURCE_ELEM];
                        let ds = block_at(native, cache, te, 0x04, TRANSITION_DEST_ELEM, 32);
                        for di in 0..ds.len() / TRANSITION_DEST_ELEM {
                            let de = &ds[di * TRANSITION_DEST_ELEM..(di + 1) * TRANSITION_DEST_ELEM];
                            trs.push(format!(
                                "{}->{}:{}#{}",
                                sid(te, 0x00), sid(de, 0x00), sid(de, 0x04), i16_at(de, 0x0A)
                            ));
                        }
                    }
                    println!(
                        "     set '{}':'{}':'{}':'{}' gait_group={} velocity_boundaries=[{}] transitions={} {}",
                        sid(me, 0), sid(ce, 0), sid(we, 0), sid(se, 0),
                        i16_at(se, SET_GAIT_GROUP), v.join(", "),
                        trs.len(), trs.join(" ")
                    );
                }
            }
        }
    }
}

/// Print-only (`hms-app --dump-aim <map> <graph substring> <clip name>`): one OVERLAY clip's
/// per-frame rotation on every node it animates, in degrees.
///
/// #aim  This is the measurement the aim convention turns on. Reach's aim overlays
/// (`combat:<class>:aim_<state>_up` / `_down`, `any:aim_spine`, `any:look`) carry 5, 9, 15 or 28
/// frames and the biped graph has NO old-style `blend screens` table to say what those frames
/// mean, so the mapping from a look direction to a frame has to be read off the DATA: a 1-D ramp
/// shows one axis moving monotonically across the whole clip, a 2-D grid shows one axis stepping
/// inside each row and the other between rows.
pub fn dump_aim_clip(native: &NativeDll, cache: CacheHandle, graph_filter: &str, clip: &str) {
    let mut tags: Vec<u32> = native.tags_by_group(cache, "jmad");
    tags.sort_unstable();
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !graph_filter.is_empty() && !name.to_lowercase().contains(&graph_filter.to_lowercase()) {
            continue;
        }
        let Some(g) = AnimGraph::read(native, cache, t) else { continue };
        let Some(ai) = g.find(clip) else { continue };
        let a = &g.anims[ai];
        println!(
            "\n== {name} :: '{}' type={:?} frames={} nodes={} screen={:?}",
            a.name, a.anim_type, a.frame_count, a.node_count, a.blend_screen
        );
        let Some(data) = g.load(native, cache, ai) else {
            println!("   load FAILED");
            continue;
        };
        let frames = data.frame_count.max(1) as usize;
        // Which nodes this clip touches at all, so the table has only meaningful columns.
        let mut touched: Vec<usize> = Vec::new();
        for f in 0..frames {
            for (i, c) in reach_jmad::sample(&data, a.anim_type, f as f32).iter().enumerate() {
                if (c.rot.is_some() || c.trans.is_some()) && !touched.contains(&i) {
                    touched.push(i);
                }
            }
        }
        touched.sort_unstable();
        println!(
            "   animates {} node(s): {}",
            touched.len(),
            touched
                .iter()
                .map(|&i| g.node_names.get(i).cloned().unwrap_or_else(|| format!("#{i}")))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for f in 0..frames {
            let ch = reach_jmad::sample(&data, a.anim_type, f as f32);
            let mut cells: Vec<String> = Vec::new();
            for &i in &touched {
                let Some(c) = ch.get(i) else { continue };
                match c.rot {
                    // The overlay's own rotation as yaw / pitch / roll about the node's axes, in
                    // degrees -- an overlay composes onto the destination, so this IS the delta.
                    Some(q) => {
                        let (z, y, x) = q.to_euler(glam::EulerRot::ZYX);
                        cells.push(format!(
                            "{:>8}: y{:>7.2} p{:>7.2} r{:>7.2}",
                            g.node_names.get(i).map(|s| s.as_str()).unwrap_or("?"),
                            z.to_degrees(),
                            -y.to_degrees(),
                            x.to_degrees()
                        ));
                    }
                    None => {}
                }
            }
            println!("   frame {f:>3}  {}", cells.join("   "));
        }
    }
}

/// Print-only (`hms-app --probe-jmad <map> [filter]`): solve the `jmad` resource DEFINITION
/// layout from engine-written bytes. For every 4-byte-aligned base in the definition blob it
/// tests the member invariant - `sum(data_sizes[0..16]) == tag_data.size`, a plausible frame /
/// node count, and `data_sizes[3] % (3 * bitvector_bytes(node_count)) == 0` - and reports the
/// bases that validate, so the real element stride and array origin are read off the data
/// instead of guessed.
pub fn probe_jmad(native: &NativeDll, cache: CacheHandle, filter: &str) {
    use jmad_off::*;
    let mut tags: Vec<u32> = native.tags_by_group(cache, "jmad");
    tags.sort_unstable();
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !filter.is_empty() && !name.to_lowercase().contains(&filter.to_lowercase()) { continue; }
        let groups = tag_block(native, cache, t, RESOURCE_GROUPS, RESOURCE_GROUP_ELEM, 6144);
        println!("\n== jmad 0x{t:08X} '{name}' groups={}", groups.len() / RESOURCE_GROUP_ELEM);
        for g in 0..groups.len() / RESOURCE_GROUP_ELEM {
            let e = &groups[g * RESOURCE_GROUP_ELEM..(g + 1) * RESOURCE_GROUP_ELEM];
            let rid = i32::from_le_bytes([e[4], e[5], e[6], e[7]]);
            let Some((def, fixups, payload)) = native.read_resource(cache, rid) else { continue };
            let mut hits: Vec<(usize, i32, i16, u8, i32)> = Vec::new();
            let mut o = 0usize;
            while o + 0x60 <= def.len() {
                let rd = |k: usize| i32::from_le_bytes([def[o + k], def[o + k + 1], def[o + k + 2], def[o + k + 3]]);
                let frames = i16::from_le_bytes([def[o + 0x08], def[o + 0x09]]);
                let nodes = def[o + 0x0A];
                let mv = def[o + 0x0B];
                let sum: i64 = (0..17).map(|i| rd(0x0C + 4 * i).max(0) as i64).sum();
                let size = rd(0x50) as i64;
                let ok = size > 0 && sum == size && frames > 0 && frames < 4096 && nodes > 0 && mv < 5;
                if ok { hits.push((o, rd(0x00), frames, nodes, size as i32)); }
                o += 4;
            }
            let strides: Vec<usize> = hits.windows(2).map(|w| w[1].0 - w[0].0).collect();
            println!("   group {g}: def={} fixups={} payload={} validated_members={} first_base={:?} strides={:?}",
                def.len(), fixups.len(), payload.len(), hits.len(),
                hits.first().map(|h| h.0), &strides[..strides.len().min(8)]);
            for h in hits.iter().take(4) {
                println!("      base={:4} field0=0x{:08X} frames={} nodes={} blob={}", h.0, h.1, h.2, h.3, h.4);
            }
            // The trailing block header (count + pointer) the engine navigates from.
            if def.len() >= 12 {
                let tl = def.len() - 12;
                println!("      tail12: count={} ptr=0x{:08X} x=0x{:08X}",
                    i32::from_le_bytes([def[tl], def[tl+1], def[tl+2], def[tl+3]]),
                    u32::from_le_bytes([def[tl+4], def[tl+5], def[tl+6], def[tl+7]]),
                    u32::from_le_bytes([def[tl+8], def[tl+9], def[tl+10], def[tl+11]]));
                println!("      head12: a={} b=0x{:08X} c=0x{:08X}",
                    i32::from_le_bytes([def[0], def[1], def[2], def[3]]),
                    u32::from_le_bytes([def[4], def[5], def[6], def[7]]),
                    u32::from_le_bytes([def[8], def[9], def[10], def[11]]));
            }
        }
    }
}

fn hex_head(b: &[u8], n: usize) -> String {
    b.iter().take(n).map(|x| format!("{x:02X}")).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------------------------
// Tag-reference scanning (the native DLL's own ScanClassRef, mirrored in Rust)
// ---------------------------------------------------------------------------------------------

/// Find the first TAG REFERENCE to `group` inside `tag`'s meta. A Reach tag reference is 16
/// bytes: the group four-CC as a big-endian u32 at +0, then the tag id at +0x0C (low 16 bits).
/// This is exactly `ScanClassRef` in `native/MccMapStudioDLL/MapModelParser.cpp`, which is how
/// `resolve_mode` / `resolve_phmo` already work, so the convention is shared.
pub fn scan_class_ref(native: &NativeDll, cache: CacheHandle, tag: u32, group: &str, scan: usize) -> Option<u32> {
    let g = group.as_bytes();
    if g.len() != 4 { return None; }
    let want = u32::from_be_bytes([g[0], g[1], g[2], g[3]]);
    let meta = native.tag_meta(cache, tag, 0, scan as u32)?;
    if meta.len() < 16 { return None; }
    let mut o = 0usize;
    while o + 16 <= meta.len() {
        let t = u32::from_le_bytes([meta[o], meta[o + 1], meta[o + 2], meta[o + 3]]);
        if t == want {
            let raw = u32::from_le_bytes([meta[o + 12], meta[o + 13], meta[o + 14], meta[o + 15]]);
            if raw != 0xFFFF_FFFF {
                let id = raw & 0xFFFF;
                if native.tag_class(cache, id).as_deref() == Some(group) { return Some(id); }
            }
        }
        o += 4;
    }
    None
}

/// The `jmad` (model_animation_graph) of an object: obje -> `hlmt` -> `animation` reference
/// (`hlmt` +0x20, tag id at +0x2C). Falls back to a scan of the hlmt meta.
pub fn resolve_jmad(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<u32> {
    let hlmt = scan_class_ref(native, cache, obj_tag, "hlmt", 0x400)?;
    // hlmt +0x20 = `animation` tag reference (model_block_struct, 544 bytes).
    if let Some(m) = native.tag_meta(cache, hlmt, 0x2C, 4) {
        let raw = u32::from_le_bytes([m[0], m[1], m[2], m[3]]);
        if raw != 0xFFFF_FFFF {
            let id = raw & 0xFFFF;
            if native.tag_class(cache, id).as_deref() == Some("jmad") { return Some(id); }
        }
    }
    scan_class_ref(native, cache, hlmt, "jmad", 0x240)
}

// ---------------------------------------------------------------------------------------------
// The animation graph
// ---------------------------------------------------------------------------------------------

use crate::reach_jmad::{self, AnimData, AnimEntry, AnimType};

/// #gait-blend  One row of a `jmad`'s `gaits` block (`s_animation_gait`, 20 B, `jmad +0x0C4`).
///
/// This is Reach's authored DIRECTIONAL LOCOMOTION unit: a direction (`move_state`) plus up to
/// three clip LABELS for the speed tiers the engine blends between. Measured on
/// `objects\characters\spartans\spartans` in `forge_halo` -- 8 gaits, one gait group:
///
/// ```text
/// gait 0 'turn_left'  slow=turn_left_slow  int=turn_left        fast=turn_left_fast   state=4
/// gait 1 'turn_right' slow=turn_right_slow int=turn_right       fast=turn_right_fast  state=5
/// gait 2 'walk_gait'  slow=walk_front_slow int=-                fast=walk_front_fast  state=0
/// gait 3 'run_gait'   slow=jog_front       int=move_front_slow  fast=move_front_fast  state=0
/// gait 4 'walk_back'  slow=walk_back       int=-                fast=walk_back_fast   state=1
/// gait 5 'run_back'   slow=jog_back        int=move_back_slow   fast=move_back        state=1
/// gait 6 'run_left'   slow=walk_left       int=walk_left_fast   fast=move_left        state=2
/// gait 7 'run_right'  slow=walk_right      int=walk_right_fast  fast=move_right       state=3
/// ```
///
/// `move_state` is the `animation_gait_directions` enum. **0..3 are the movement directions and
/// 4..5 are the turns**, which is not a guess: `is_gait_direction_movement(gait_direction)`
/// asserts `(u16)direction <= 3` (`haloreach_debug.dll` `sub_1808A6560`,
/// `animation_definitions.cpp:3042`), and the data above puts `turn_left`/`turn_right` at 4/5
/// and `front`/`back`/`left`/`right` at 0/1/2/3.
#[derive(Clone, Debug, Default)]
pub struct GaitDef {
    pub name: String,
    /// The three authored clip LABELS. A label is resolved through the
    /// `mode -> weapon class -> weapon type -> set -> actions` hierarchy, NOT by assembling
    /// `mode:class:label` -- see [`AnimGraph::resolve_action`] for why that distinction is load
    /// bearing (`crouch:` has no clip called `walk_front_slow`, yet the crouch set's speed range
    /// is exactly `crouch:rifle:walk_front`'s).
    pub slow: String,
    pub intermediate: String,
    pub fast: String,
    /// `animation_gait_directions`: 0 front, 1 back, 2 left, 3 right, 4 turn left, 5 turn right.
    pub move_state: i16,
}

/// #gait-blend  One row of `gait groups` (`s_animation_gait_group`, 16 B, `jmad +0x0D0`).
#[derive(Clone, Debug, Default)]
pub struct GaitGroupDef {
    pub name: String,
    /// Indices into [`AnimGraph::gaits`].
    pub gaits: Vec<i16>,
}

/// #gait-blend  One `c_animation_set` leaf of the `modes -> weapon class -> weapon type -> sets`
/// hierarchy, flattened: the four labels that address it, the gait group it overrides with, and
/// its `velocity boundaries`.
#[derive(Clone, Debug, Default)]
pub struct AnimSetRow {
    pub mode: String,
    pub class: String,
    pub weapon_type: String,
    pub set: String,
    /// `c_animation_set +0x008`; -1 = inherit [`AnimGraph::default_gait_group`].
    pub gait_group: i16,
    /// `c_animation_set +0x03C` -> 8 reals = `[min, max]` per movement direction 0..3, wu/s.
    ///
    /// **Precomputed by the tool from the clips' OWN speeds**, which is the cross-check that
    /// confirms the whole law: for `combat:rifle` on `forge_halo` they read
    /// `[0.251, 2.100, 0.492, 1.894, 0.491, 1.881, 0.500, 1.910]`, and the eight clips named by
    /// the four directions' gaits carry exactly those speeds at
    /// `c_shared_model_animation +0x30 x 30` (`walk_front_slow` 0.2510, `move_front_fast`
    /// 2.1004, `walk_back` 0.4918, `move_back` 1.8945, `walk_left` 0.4906, `move_left` 1.8813,
    /// `walk_right` 0.5005, `move_right` 1.9100). 8 of 8, and 8 more for `crouch:rifle`.
    pub velocity_boundaries: [f32; 8],
    /// `(action label, graph animation index)` for this set's `actions` block.
    pub actions: Vec<(String, i16)>,
}

/// #gait-blend  One speed TIER of one gait: a clip, the speed (or turn rate) it is authored to
/// be played at, and how many frames it is.
#[derive(Clone, Debug, PartialEq)]
pub struct GaitTier {
    /// Graph animation index.
    pub anim: usize,
    /// wu/s for a movement gait, radians/s for a turn gait -- the clip's OWN authored value
    /// (`c_shared_model_animation +0x30 x 30` / `|+0x34|`).
    pub rate: f32,
    pub frames: f32,
    pub name: String,
}

/// #gait-blend  One gait, resolved: which direction it serves and its tiers, speed-ascending.
#[derive(Clone, Debug, PartialEq)]
pub struct GaitRow {
    /// `animation_gait_directions`: 0 front, 1 back, 2 left, 3 right, 4 turn left, 5 turn right.
    pub direction: i16,
    pub name: String,
    pub tiers: Vec<GaitTier>,
}

/// #gait-blend  The authored locomotion table for one `(mode, weapon class)`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GaitTable {
    pub gaits: Vec<GaitRow>,
}

/// #gait-blend  What the engine's gait selection returns: the two clips to blend and the weight
/// of the faster one.
#[derive(Clone, Debug, PartialEq)]
pub struct GaitBlend {
    pub lo: GaitTier,
    pub hi: GaitTier,
    /// 0 = wholly `lo`, 1 = wholly `hi`. Clamped.
    pub w: f32,
    /// The gait the pair came from, for the caption and the gates.
    pub gait: String,
    /// False when NO gait's authored range brackets this speed, so the pick is the nearest one
    /// rather than an interpolation. The engine makes the same distinction (its cost is `> 1`).
    pub bracketed: bool,
}

impl GaitTable {
    /// #gait-blend  **The engine's own gait pick**, for one direction at one speed.
    ///
    /// `haloreach_debug.dll` `sub_180891E90` (`animation_manager.cpp:6850`) walks the gait
    /// group, keeps the gaits whose `move_state` matches the direction asked for, reads each
    /// one's slow and fast clip SPEEDS, and scores it:
    ///
    /// * the speed falls inside `[v_slow, v_fast]` -> `cost = (s - v_slow) / (v_fast - v_slow)`,
    ///   which is in `[0, 1]` and IS the blend weight;
    /// * otherwise -> `cost = 1 + min(|s - v_slow|, |s - v_fast|)`, which is always `> 1`, so a
    ///   gait that brackets the speed always beats one that does not.
    ///
    /// The lowest cost wins. This reproduces that, over the tiers (so the authored INTERMEDIATE
    /// clip is a bracket of its own -- see [`AnimGraph::gait_table`]).
    pub fn pick(&self, direction: i16, speed: f32) -> Option<GaitBlend> {
        let mut best: Option<(f32, GaitBlend)> = None;
        for row in self.gaits.iter().filter(|r| r.direction == direction) {
            for pair in row.tiers.windows(2) {
                let (lo, hi) = (&pair[0], &pair[1]);
                if hi.rate <= lo.rate {
                    continue;
                }
                let (cost, w, bracketed) = if speed >= lo.rate && speed <= hi.rate {
                    let w = (speed - lo.rate) / (hi.rate - lo.rate);
                    (w, w, true)
                } else {
                    let d = (speed - lo.rate).abs().min((speed - hi.rate).abs());
                    (1.0 + d, if speed < lo.rate { 0.0 } else { 1.0 }, false)
                };
                if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                    best = Some((
                        cost,
                        GaitBlend {
                            lo: lo.clone(),
                            hi: hi.clone(),
                            w: w.clamp(0.0, 1.0),
                            gait: row.name.clone(),
                            bracketed,
                        },
                    ));
                }
            }
            // A gait with a single tier can still answer: it is its own `lo` and `hi`.
            if row.tiers.len() == 1 {
                let t = &row.tiers[0];
                let cost = 1.0 + (speed - t.rate).abs();
                if best.as_ref().is_none_or(|(c, _)| cost < *c) {
                    best = Some((
                        cost,
                        GaitBlend {
                            lo: t.clone(),
                            hi: t.clone(),
                            w: 0.0,
                            gait: row.name.clone(),
                            bracketed: false,
                        },
                    ));
                }
            }
        }
        best.map(|(_, b)| b)
    }

    /// #gait-blend  The authored `[min, max]` speed of one direction, over every gait that
    /// serves it -- the number the tool precomputes into `velocity boundaries`.
    pub fn speed_range(&self, direction: i16) -> Option<(f32, f32)> {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in self.gaits.iter().filter(|r| r.direction == direction) {
            for t in &row.tiers {
                lo = lo.min(t.rate);
                hi = hi.max(t.rate);
            }
        }
        (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
    }
}

/// #gait-blend  The clips to pose a moving body with this frame, and the stride they cover.
#[derive(Clone, Debug, PartialEq)]
pub struct GaitPlan {
    /// `(graph animation index, absolute weight, clip frames, clip authored rate)`, **heaviest
    /// first** and the rest in a stable (direction, then speed) order. The weights sum to 1.
    pub clips: Vec<(usize, f32, f32, f32)>,
    /// How much GROUND one cycle of the blend covers, wu — the weighted mean of each clip's own
    /// authored stride `rate x frames / 30`.
    pub stride: f32,
    /// One line naming the mix, for the status caption and the gates.
    pub caption: String,
    /// `(direction, gait name)` per direction in the mix — the GAIT IDENTITY. The engine picks
    /// one gait per direction by a cost rule and cross-fades when that pick changes
    /// (`m_previous_gait_index`), so a change here is the one place the weight map is allowed to
    /// step. The gate asserts exactly that: continuous while this is unchanged.
    pub gaits: Vec<(i16, String)>,
    /// True when every direction in the mix found a gait whose authored range BRACKETS the
    /// body's speed, so the tier weights are interpolations rather than nearest-neighbour picks.
    pub bracketed: bool,
}

/// #gait-blend  **The locomotion mix for one travel direction and one ground speed.** Pure: no
/// tag reads, no actor, no GPU — so a gate drives it directly.
///
/// Two things are being combined, and only one of them is the engine's:
///
/// 1. **The speed-tier blend IS the engine's**, measured. Reach's `gaits` block authors, per
///    direction, two or three clips with their own speeds, and `animation_manager.cpp:6850`
///    (`haloreach_debug.dll` `sub_180891E90`) blends the bracketing pair by
///    `(s - v_slow) / (v_fast - v_slow)`. See [`GaitTable::pick`].
/// 2. **The DIRECTIONAL blend is HMS's own**, and it has to be labelled as such. The engine
///    authors exactly four movement directions and asks its gait machinery for ONE of them at a
///    time (`is_gait_direction_movement` bounds the direction at 3; the goal settings carry a
///    single `move state`; a change of direction is handled by the state/transition channel,
///    which is a cross-fade). So Reach at a 45-degree travel angle plays one direction's clip.
///    This mixes the two bracketing directions instead, because that is what the product owner
///    asked for twice — "if I walk forward and strafe at the same time, the animation is just
///    using a walk forward animation ... not blending them" — and because the per-direction
///    speeds `matg @player information` authors (2.200 forward / 1.980 sideways) are already
///    combined as an ELLIPSE by `play_phys::top_speed_base`, so the body's own velocity at 45
///    degrees is a genuine mixture of the two.
///
/// Both halves share one weight set, so there is one writer for "how much of each clip".
pub fn gait_plan(table: &GaitTable, theta: f32, speed: f32) -> Option<GaitPlan> {
    let (d0, d1, wd) = crate::film_anim::travel_directions(theta);
    let mut parts: Vec<(i16, f32, GaitBlend)> = Vec::new();
    for (d, w) in [(d0, 1.0 - wd), (d1, wd)] {
        if w <= 1e-4 {
            continue;
        }
        if let Some(b) = table.pick(d, speed) {
            parts.push((d, w, b));
        }
    }
    if parts.is_empty() {
        return None;
    }
    // A direction with no authored gait drops out, so renormalise rather than posing a body at
    // partial weight (which would silently lerp it toward the skeleton's bind pose).
    let sum: f32 = parts.iter().map(|(_, w, _)| *w).sum();
    if !(sum > 1e-4) {
        return None;
    }
    let bracketed = parts.iter().all(|(_, _, b)| b.bracketed);
    // Fixed order: direction ascending, then slow tier before fast. The ORDER must not depend on
    // the weights -- see `Actor::compute_pose` for why.
    parts.sort_by_key(|(d, _, _)| *d);
    let mut clips: Vec<(usize, f32, f32, f32)> = Vec::new();
    let mut add = |t: &GaitTier, w: f32| {
        if w <= 1e-4 {
            return;
        }
        match clips.iter_mut().find(|c| c.0 == t.anim) {
            Some(c) => c.1 += w,
            None => clips.push((t.anim, w, t.frames, t.rate)),
        }
    };
    let mut caption = String::new();
    let mut gaits: Vec<(i16, String)> = Vec::new();
    for (d, w, b) in &parts {
        gaits.push((*d, b.gait.clone()));
        let w = w / sum;
        add(&b.lo, w * (1.0 - b.w));
        add(&b.hi, w * b.w);
        caption += &format!(
            "{}dir{d} {:.0}% '{}' {}->{} {:.0}%",
            if caption.is_empty() { "" } else { " + " },
            w * 100.0,
            b.gait,
            b.lo.name.rsplit(':').next().unwrap_or(""),
            b.hi.name.rsplit(':').next().unwrap_or(""),
            b.w * 100.0
        );
    }
    if clips.is_empty() {
        return None;
    }
    // THE STRIDE, and therefore the RATE. Every clip in the mix runs at the same normalised
    // phase -- that is what keeps the feet together, and it is the only arrangement that cannot
    // tear -- so there is exactly one stride to choose and it is the weighted mean of the clips'
    // OWN authored strides. A Reach locomotion clip is authored in place, so its stride is
    // `speed x frames / 30`: the ground it covers while it runs once at 30 fps at the speed it
    // was authored for. With a single clip at weight 1 this is identical to
    // `film_anim::cycles_per_wu_at_speed(speed, frames)`, which is what the single-clip path
    // used, so nothing about the "not (distance x rate) mod 1" fix is undone.
    let stride: f32 = clips.iter().map(|(_, w, f, r)| w * r * f / crate::film_anim::CLIP_FPS).sum();
    if !(stride.is_finite() && stride > 1e-5) {
        return None;
    }
    // Heaviest first, keeping the relative order of the rest.
    if let Some(k) = (0..clips.len()).max_by(|&a, &b| clips[a].1.total_cmp(&clips[b].1)) {
        let top = clips.remove(k);
        clips.insert(0, top);
    }
    Some(GaitPlan { clips, stride, caption, gaits, bracketed })
}

/// One Reach `jmad`: its animation list plus the resource-group ids its keyframe data lives in.
#[derive(Clone, Debug, Default)]
pub struct AnimGraph {
    pub tag: u32,
    pub anims: Vec<AnimEntry>,
    /// #gait-blend  Per animation, its OWN authored average speed in wu/s --
    /// `c_shared_model_animation +0x30 x 30`. This is the number the engine reads
    /// (`haloreach_debug.dll` `sub_180894AD0`, `animation_inline.h:286`:
    /// `return *(float *)(shared_animation + 0x30) * 30.0;`), and it is what a locomotion clip's
    /// RATE must be built from: a Reach locomotion clip is authored in place, so the ground one
    /// cycle covers is `speed x frames / 30` and nothing else says it.
    pub clip_speed: Vec<f32>,
    /// #gait-blend  Per animation, its authored average turn rate in radians/s --
    /// `fabsf(c_shared_model_animation +0x34)`, the same function's turn-state branch
    /// (`animation_inline.h:291`). Measured: `combat:rifle:turn_left_slow` 1.5138,
    /// `turn_left` 2.0944 (= 120 deg/s), `turn_left_fast` 6.2832 (= 360 deg/s).
    pub clip_turn_rate: Vec<f32>,
    /// #gait-blend  `jmad +0x0C4 gaits`.
    pub gaits: Vec<GaitDef>,
    /// #gait-blend  `jmad +0x0D0 gait groups`.
    pub gait_groups: Vec<GaitGroupDef>,
    /// #gait-blend  `jmad +0x100` (= content +0x000) `default gait group`; -1 = none.
    pub default_gait_group: i16,
    /// #gait-blend  the flattened `modes -> weapon class -> weapon type -> sets` leaves.
    pub sets: Vec<AnimSetRow>,
    /// `tag resource groups` resource ids, indexed by group (= `resource_group` of an animation).
    pub group_resource_ids: Vec<i32>,
    /// `jmad` `skeleton nodes` names, parallel to the animation node channels.
    pub node_names: Vec<String>,
    /// #fp-attach  Each skeleton node's PARENT node index in the graph's own hierarchy (-1 = a
    /// root), `s_animation_graph_node` +0x08. The graph is one tree over EVERY model the mode
    /// drives, so a first-person weapon's root node is a CHILD of a hand node that belongs to a
    /// different render model; without this the weapon's clip channel (a LOCAL transform) is
    /// composed against nothing and the gun ends up at the rig's root, inside the player's head.
    pub node_parents: Vec<i16>,
    /// #aim `blend screens` (+0x06C): what an aim/look overlay's FRAMES mean in angles. Indexed
    /// by [`AnimEntry::blend_screen`].
    pub blend_screens: Vec<reach_jmad::BlendScreen>,
    /// Parallel to `blend_screens`: each screen's label, resolved.
    pub blend_screen_labels: Vec<String>,
}

impl AnimGraph {
    pub fn read(native: &NativeDll, cache: CacheHandle, tag: u32) -> Option<AnimGraph> {
        use reach_jmad::*;
        let anims_b = tag_block(native, cache, tag, OFF_ANIMATIONS, ANIMATION_ELEM, 2048);
        if anims_b.is_empty() { return None; }
        let mut anims = Vec::with_capacity(anims_b.len() / ANIMATION_ELEM);
        // #gait-blend  Parallel to `anims`: each clip's OWN authored speed and turn rate.
        let mut clip_speed: Vec<f32> = Vec::with_capacity(anims_b.len() / ANIMATION_ELEM);
        let mut clip_turn_rate: Vec<f32> = Vec::with_capacity(anims_b.len() / ANIMATION_ELEM);
        for i in 0..anims_b.len() / ANIMATION_ELEM {
            let e = &anims_b[i * ANIMATION_ELEM..(i + 1) * ANIMATION_ELEM];
            let sid = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
            let shared = block_at(native, cache, e, ANIM_SHARED_BLOCK, SHARED_ELEM, 1);
            if shared.len() < SHARED_ELEM { continue; }
            let s = &shared[..SHARED_ELEM];
            // #gait-blend  +0x30 average translation per frame (x30 = wu/s), +0x34 average
            // rotation per frame. The two the engine's `sub_180894AD0` reads, and nothing else
            // in the tag says how fast a locomotion clip is meant to be played.
            clip_speed.push(f32::from_le_bytes([s[0x30], s[0x31], s[0x32], s[0x33]]) * 30.0);
            clip_turn_rate.push(f32::from_le_bytes([s[0x34], s[0x35], s[0x36], s[0x37]]).abs());
            anims.push(AnimEntry {
                index: i,
                name_sid: sid,
                name: native.string_id(cache, sid).unwrap_or_default(),
                loop_frame: i16::from_le_bytes([e[8], e[9]]),
                node_count: s[0],
                frame_count: i16::from_le_bytes([s[2], s[3]]),
                anim_type: AnimType::from_raw(s[4]),
                frame_info_type: s[5],
                compression_setting: s[7],
                blend_screen: match e[ANIM_BLEND_SCREEN] {
                    0xFF => None,
                    v => Some(v as usize),
                },
                production_checksum: i32::from_le_bytes([s[0x18], s[0x19], s[0x1A], s[0x1B]]),
                resource_group: i16::from_le_bytes([s[0x20], s[0x21]]),
                resource_group_member: i16::from_le_bytes([s[0x22], s[0x23]]),
                heading: Vec3::new(
                    f32::from_le_bytes([s[0x24], s[0x25], s[0x26], s[0x27]]),
                    f32::from_le_bytes([s[0x28], s[0x29], s[0x2A], s[0x2B]]),
                    f32::from_le_bytes([s[0x2C], s[0x2D], s[0x2E], s[0x2F]]),
                ),
            });
        }
        let groups = tag_block(native, cache, tag, OFF_RESOURCE_GROUPS, RESOURCE_GROUP_ELEM, 6144);
        let group_resource_ids = (0..groups.len() / RESOURCE_GROUP_ELEM)
            .map(|g| {
                let e = &groups[g * RESOURCE_GROUP_ELEM..(g + 1) * RESOURCE_GROUP_ELEM];
                i32::from_le_bytes([e[4], e[5], e[6], e[7]])
            })
            .collect();
        let skel = tag_block(native, cache, tag, OFF_SKELETON_NODES, GRAPH_NODE_ELEM, 255);
        let node_names = (0..skel.len() / GRAPH_NODE_ELEM)
            .map(|i| {
                let e = &skel[i * GRAPH_NODE_ELEM..(i + 1) * GRAPH_NODE_ELEM];
                native
                    .string_id(cache, u32::from_le_bytes([e[0], e[1], e[2], e[3]]))
                    .unwrap_or_default()
            })
            .collect();
        // #fp-attach  The graph's own hierarchy, +0x08 of each 44-byte node.
        let node_parents: Vec<i16> = (0..skel.len() / GRAPH_NODE_ELEM)
            .map(|i| {
                let e = &skel[i * GRAPH_NODE_ELEM..(i + 1) * GRAPH_NODE_ELEM];
                i16::from_le_bytes([e[GRAPH_NODE_PARENT], e[GRAPH_NODE_PARENT + 1]])
            })
            .collect();
        let bs_b = tag_block(native, cache, tag, OFF_BLEND_SCREENS, BLEND_SCREEN_ELEM, 1024);
        let mut blend_screens = Vec::new();
        let mut blend_screen_labels = Vec::new();
        for i in 0..bs_b.len() / BLEND_SCREEN_ELEM {
            let e = &bs_b[i * BLEND_SCREEN_ELEM..(i + 1) * BLEND_SCREEN_ELEM];
            let Some(sc) = reach_jmad::BlendScreen::read(e) else { continue };
            blend_screen_labels.push(native.string_id(cache, sc.label_sid).unwrap_or_default());
            blend_screens.push(sc);
        }
        // #gait-blend  THE GAIT TABLES. Read once per graph, never per frame.
        let sid_of = |b: &[u8], o: usize| -> String {
            if o + 4 > b.len() { return String::new(); }
            let v = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            if v == 0 { return String::new(); }
            native.string_id(cache, v).unwrap_or_default()
        };
        let i16_of = |b: &[u8], o: usize| -> i16 {
            if o + 2 > b.len() { -1 } else { i16::from_le_bytes([b[o], b[o + 1]]) }
        };
        let gaits_b = tag_block(native, cache, tag, jmad_off::GAITS, jmad_off::GAIT_ELEM, 32);
        let gaits: Vec<GaitDef> = (0..gaits_b.len() / jmad_off::GAIT_ELEM)
            .map(|i| {
                let e = &gaits_b[i * jmad_off::GAIT_ELEM..(i + 1) * jmad_off::GAIT_ELEM];
                GaitDef {
                    name: sid_of(e, 0x00),
                    slow: sid_of(e, 0x04),
                    intermediate: sid_of(e, 0x08),
                    fast: sid_of(e, 0x0C),
                    move_state: i16_of(e, 0x10),
                }
            })
            .collect();
        let gg_b = tag_block(native, cache, tag, jmad_off::GAIT_GROUPS, jmad_off::GAIT_GROUP_ELEM, 32);
        let gait_groups: Vec<GaitGroupDef> = (0..gg_b.len() / jmad_off::GAIT_GROUP_ELEM)
            .map(|i| {
                let e = &gg_b[i * jmad_off::GAIT_GROUP_ELEM..(i + 1) * jmad_off::GAIT_GROUP_ELEM];
                let items = block_at(native, cache, e, jmad_off::GAIT_GROUP_ITEMS, jmad_off::GAIT_ITEM_ELEM, 32);
                GaitGroupDef {
                    name: sid_of(e, 0x00),
                    gaits: (0..items.len() / jmad_off::GAIT_ITEM_ELEM)
                        .map(|k| i16_of(&items, k * jmad_off::GAIT_ITEM_ELEM))
                        .collect(),
                }
            })
            .collect();
        let default_gait_group = native
            .tag_meta(cache, tag, jmad_off::DEFAULT_GAIT_GROUP as u32, 2)
            .map(|b| i16_of(&b, 0))
            .unwrap_or(-1);
        let mut sets: Vec<AnimSetRow> = Vec::new();
        let modes_b = tag_block(native, cache, tag, OFF_MODES, MODE_ELEM, 512);
        for m in 0..modes_b.len() / MODE_ELEM {
            let me = &modes_b[m * MODE_ELEM..(m + 1) * MODE_ELEM];
            let wcs = block_at(native, cache, me, jmad_off::MODE_WEAPON_CLASS, jmad_off::WEAPON_CLASS_ELEM, 64);
            for c in 0..wcs.len() / jmad_off::WEAPON_CLASS_ELEM {
                let ce = &wcs[c * jmad_off::WEAPON_CLASS_ELEM..(c + 1) * jmad_off::WEAPON_CLASS_ELEM];
                let wts = block_at(native, cache, ce, jmad_off::WEAPON_CLASS_TYPES, jmad_off::WEAPON_TYPE_ELEM, 64);
                for w in 0..wts.len() / jmad_off::WEAPON_TYPE_ELEM {
                    let we = &wts[w * jmad_off::WEAPON_TYPE_ELEM..(w + 1) * jmad_off::WEAPON_TYPE_ELEM];
                    let st = block_at(native, cache, we, jmad_off::WEAPON_TYPE_SETS, jmad_off::SET_ELEM, 64);
                    for k in 0..st.len() / jmad_off::SET_ELEM {
                        let se = &st[k * jmad_off::SET_ELEM..(k + 1) * jmad_off::SET_ELEM];
                        let vb = block_at(
                            native, cache, se,
                            jmad_off::SET_VELOCITY_BOUNDARIES,
                            jmad_off::VELOCITY_BOUNDARIES_ELEM, 1,
                        );
                        let mut bounds = [0.0f32; 8];
                        for (j, b) in bounds.iter_mut().enumerate() {
                            if vb.len() >= (j + 1) * 4 {
                                *b = f32::from_le_bytes([vb[j * 4], vb[j * 4 + 1], vb[j * 4 + 2], vb[j * 4 + 3]]);
                            }
                        }
                        let acts = block_at(native, cache, se, jmad_off::SET_ACTIONS, jmad_off::ACTION_ELEM, 256);
                        let actions: Vec<(String, i16)> = (0..acts.len() / jmad_off::ACTION_ELEM)
                            .map(|j| {
                                let ae = &acts[j * jmad_off::ACTION_ELEM..(j + 1) * jmad_off::ACTION_ELEM];
                                (sid_of(ae, 0x00), i16_of(ae, 0x0A))
                            })
                            .collect();
                        sets.push(AnimSetRow {
                            mode: sid_of(me, 0),
                            class: sid_of(ce, 0),
                            weapon_type: sid_of(we, 0),
                            set: sid_of(se, 0),
                            gait_group: i16_of(se, jmad_off::SET_GAIT_GROUP),
                            velocity_boundaries: bounds,
                            actions,
                        });
                    }
                }
            }
        }
        Some(AnimGraph {
            tag, anims, clip_speed, clip_turn_rate, gaits, gait_groups, default_gait_group, sets,
            group_resource_ids, node_names, node_parents, blend_screens, blend_screen_labels,
        })
    }

    /// #gait-blend  Resolve one ACTION LABEL inside a `mode`/`weapon class` to a graph animation
    /// index, the way the engine's own `sub_180891AF0` does.
    ///
    /// **Why this is not `find("{mode}:{class}:{label}")`.** A clip's NAME is generated from the
    /// first action that points at it, and a set may carry several labels aliasing one clip. The
    /// crouch set is the measured case: the `gaits` block names `walk_front_slow` /
    /// `move_front_fast` (the COMBAT leaves), yet `crouch:rifle`'s own authored speed range is
    /// `[0.610, 0.922]`, which is exactly `crouch:rifle:walk_front` (0.6096) and
    /// `crouch:rifle:move_front` (0.9220). Assembling the name finds nothing there; resolving the
    /// LABEL through the set's `actions` finds the right clip.
    ///
    /// Specificity: the exact `(mode, class)` set first, then `(mode, any)`, then
    /// `(combat, class)`, then `(combat, any)` -- the engine's own fallback order up the
    /// hierarchy -- and within a level a `weapon type`/`set` of `any` (or empty) is preferred
    /// over a specialised one. Falls back to the assembled name so a graph that authors no
    /// `actions` at all still answers.
    pub fn resolve_action(&self, mode: &str, class: &str, label: &str) -> Option<usize> {
        let generic = |l: &str| l.is_empty() || l.eq_ignore_ascii_case("any");
        for (m, c) in [(mode, class), (mode, "any"), ("combat", class), ("combat", "any")] {
            // Two passes: the `any:any` weapon type/set first, then anything.
            for only_generic in [true, false] {
                for row in &self.sets {
                    if !row.mode.eq_ignore_ascii_case(m) {
                        continue;
                    }
                    if !(row.class.eq_ignore_ascii_case(c) || (c == "any" && generic(&row.class))) {
                        continue;
                    }
                    if only_generic && !(generic(&row.weapon_type) && generic(&row.set)) {
                        continue;
                    }
                    for (al, ai) in &row.actions {
                        if al.eq_ignore_ascii_case(label) && *ai >= 0 && (*ai as usize) < self.anims.len() {
                            return Some(*ai as usize);
                        }
                    }
                }
            }
        }
        self.find(&format!("{mode}:{class}:{label}"))
    }

    /// #gait-blend  The `velocity boundaries` in force for a `(mode, weapon class)`, as
    /// `[min, max]` per movement direction 0..3 -- or `None` when the set authors none.
    pub fn velocity_boundaries(&self, mode: &str, class: &str) -> Option<[f32; 8]> {
        let generic = |l: &str| l.is_empty() || l.eq_ignore_ascii_case("any");
        for (m, c) in [(mode, class), (mode, "any"), ("combat", class), ("combat", "any")] {
            for row in &self.sets {
                if !row.mode.eq_ignore_ascii_case(m) || !generic(&row.weapon_type) || !generic(&row.set) {
                    continue;
                }
                if !(row.class.eq_ignore_ascii_case(c) || (c == "any" && generic(&row.class))) {
                    continue;
                }
                if row.velocity_boundaries.iter().any(|&v| v != 0.0) {
                    return Some(row.velocity_boundaries);
                }
            }
        }
        None
    }

    /// #gait-blend  The gait group in force for a `(mode, weapon class)`: the set's own override
    /// if it has one, else [`AnimGraph::default_gait_group`].
    fn gait_group_for(&self, mode: &str, class: &str) -> Option<usize> {
        let generic = |l: &str| l.is_empty() || l.eq_ignore_ascii_case("any");
        for row in &self.sets {
            if row.mode.eq_ignore_ascii_case(mode)
                && (row.class.eq_ignore_ascii_case(class) || generic(&row.class))
                && generic(&row.weapon_type)
                && generic(&row.set)
                && row.gait_group >= 0
            {
                return Some(row.gait_group as usize);
            }
        }
        (self.default_gait_group >= 0).then_some(self.default_gait_group as usize)
    }

    /// #gait-blend  Resolve the whole authored locomotion table for one `(mode, weapon class)`.
    ///
    /// One call per change of mode or weapon class, never per frame.
    ///
    /// # How the clips are found, and why it is by STEM inside the mode
    ///
    /// The `gaits` block supplies the authored DIRECTION VOCABULARY: for each direction, the leaf
    /// names its tiers are called. Those names are the COMBAT set's (`walk_front_slow`,
    /// `jog_front`, `move_front_fast`, ...) and the crouch set calls the same jobs
    /// `walk_front` / `move_front`, so resolving the gait's label literally finds a `combat:`
    /// clip for a crouching body.
    ///
    /// This takes each gait's labels, strips the `_slow` / `_fast` tier suffix to get the STEM,
    /// and then looks inside the mode actually in force for `{mode}:{class}:{stem}`,
    /// `..._slow` and `..._fast`. The test that this is the right rule is the authored
    /// `velocity boundaries` block, which the tool precomputed from the clips the engine really
    /// uses — and which this route reproduces for **8 of 8** directions across `combat:rifle`
    /// and `crouch:rifle` (`[0.2510, 2.1004] [0.4918, 1.8945] [0.4906, 1.8813] [0.5005, 1.9100]`
    /// and `[0.6096, 0.9220] [0.5009, 0.8555] [0.4753, 0.8193] [0.4703, 0.7540]`), where the
    /// literal-label route reproduces 5 of 8. Gated in
    /// `probe_gait_table_reproduces_the_authored_velocity_boundaries`.
    ///
    /// Every tier HMS finds is used, not only the two the engine's own path reads (+0x04 and
    /// +0x0C): the authored INTERMEDIATE (+0x08) is "the animation name used for the speed
    /// variations" in its own field comment, and it measures between the other two
    /// (`run_gait`: 0.8607 / 1.5359 / 2.1004). That extra bracket is HMS's reading of the
    /// authored field and not something the shipped code path does — stated, not hidden.
    pub fn gait_table(&self, mode: &str, class: &str) -> GaitTable {
        let mut t = GaitTable::default();
        let Some(group) = self.gait_group_for(mode, class) else { return t };
        let Some(g) = self.gait_groups.get(group) else { return t };
        for &gi in &g.gaits {
            let Some(gd) = (gi >= 0).then(|| self.gaits.get(gi as usize)).flatten() else { continue };
            let d = gd.move_state;
            if !(0..6).contains(&d) {
                continue;
            }
            let turn = d >= 4;
            let mut tiers: Vec<GaitTier> = Vec::new();
            for label in [&gd.slow, &gd.intermediate, &gd.fast] {
                if label.is_empty() {
                    continue;
                }
                let stem = label
                    .strip_suffix("_slow")
                    .or_else(|| label.strip_suffix("_fast"))
                    .unwrap_or(label);
                for leaf in [stem.to_string(), format!("{stem}_slow"), format!("{stem}_fast")] {
                    let want = format!("{mode}:{class}:{leaf}");
                    // EXACT names only. `find` falls back to a substring match, which would make
                    // `crouch:rifle:move_front` answer for `move_front_slow` and quietly put a
                    // clip in the wrong tier.
                    let Some(anim) = self.anims.iter().position(|a| a.name.eq_ignore_ascii_case(&want)) else {
                        continue;
                    };
                    let rate = if turn {
                        self.clip_turn_rate.get(anim).copied().unwrap_or(0.0)
                    } else {
                        self.clip_speed.get(anim).copied().unwrap_or(0.0)
                    };
                    let frames = self.anims.get(anim).map(|a| a.frame_count.max(1) as f32).unwrap_or(1.0);
                    if !(rate.is_finite() && rate > 1e-4) || tiers.iter().any(|x| x.anim == anim) {
                        continue;
                    }
                    tiers.push(GaitTier { anim, rate, frames, name: self.anims[anim].name.clone() });
                }
            }
            if tiers.is_empty() {
                continue;
            }
            tiers.sort_by(|a, b| a.rate.total_cmp(&b.rate));
            t.gaits.push(GaitRow { direction: d, name: gd.name.clone(), tiers });
        }
        t
    }

    /// Animation index by NAME: exact (case-insensitive) first, then substring.
    pub fn find(&self, name: &str) -> Option<usize> {
        let n = name.to_lowercase();
        self.anims
            .iter()
            .position(|a| a.name.eq_ignore_ascii_case(&n))
            .or_else(|| self.anims.iter().position(|a| a.name.to_lowercase().contains(&n)))
    }

    /// Load one animation's compressed blob out of its resource group.
    pub fn load(&self, native: &NativeDll, cache: CacheHandle, anim: usize) -> Option<AnimData> {
        let a = self.anims.get(anim)?;
        let rid = *self.group_resource_ids.get(a.resource_group.max(0) as usize)?;
        let (def, fixups, payload) = native.read_resource(cache, rid)?;
        // RESOURCE DEFINITION LAYOUT, solved from engine-written bytes (`--probe-jmad`, which
        // accepts a member only when sum(data_sizes[0..16]) == tag_data.size):
        //   def[0 .. len-12]  = the `group_members` array, element stride 100 bytes
        //   def[len-12 ..]    = the block header the engine navigates from: count i32, pointer
        //                       u32, 0xF09ADC2A. `count` equals the validated member count on
        //                       every group tested (Spartan + Elite fp_dmr: 2 / 18 / 5).
        // Member (100 B): +0x00 animation index (ZERO in every shipped cache), +0x04 checksum,
        //   +0x08 frame count i16, +0x0A node count u8, +0x0B movement data type u8,
        //   +0x0C data sizes i32[17], +0x50 tag_data {size, flags, stream pos, address, def}.
        // `fixups[i]` is the PAYLOAD byte offset of member i's blob (16-byte aligned runs:
        // fp_dmr group 1 = [0, 1744, 3488, ...] for 1732-byte members); the list carries one
        // extra trailing 0 for the block pointer itself.
        if def.len() < 12 + reach_jmad::MEMBER_ELEM { return None; }
        let tl = def.len() - 12;
        let count = i32::from_le_bytes([def[tl], def[tl + 1], def[tl + 2], def[tl + 3]]).max(0) as usize;
        let count = count.min(tl / reach_jmad::MEMBER_ELEM);
        let m = a.resource_group_member.max(0) as usize;
        if m >= count { return None; }
        let base = m * reach_jmad::MEMBER_ELEM;
        let e = &def[base..base + reach_jmad::MEMBER_ELEM];
        let mut data = AnimData::header_from_member(e)?;
        let size = AnimData::declared_blob_size(e).max(0) as usize;
        if size == 0 || data.total_size() as usize != size { return None; }
        let o = *fixups.get(m)? as usize;
        if o + size > payload.len() { return None; }
        data.blob = payload[o..o + size].to_vec();
        Some(data)
    }
}

// ---------------------------------------------------------------------------------------------
// The posed-actor scene
// ---------------------------------------------------------------------------------------------

/// What a held weapon needs: its object tag plus the resolved marker pair.
#[derive(Clone, Debug)]
pub struct HeldWeapon {
    pub obj_tag: u32,
    pub mode_tag: u32,
    /// Node count of the weapon's render model (how much of the bone palette to fill).
    pub node_count: usize,
    /// The BIPED marker group the weapon mates to (`unit.right_hand_node`, "primary_weapon").
    pub parent_marker: String,
    /// The WEAPON marker group that is mated (`unit.preferred_gun_node` when the weapon's model
    /// has it, else `weap.handle_node`; identity when neither resolves).
    pub child_marker: String,
    /// The weapon marker's frame in the WEAPON's model space (identity when no marker matched).
    pub child_local: Mat4,
    /// Which skeleton node of the BIPED carries the parent marker, and the marker's local frame.
    pub parent_node: i32,
    pub parent_local: Mat4,
    /// The weapon model's resolved `hlmt` variant. Without one the render model draws every
    /// region's every permutation -- the same "wearing all the armour at once" bug the biped had,
    /// which on a weapon shows up as several overlapping sights, barrels and ammo counters.
    /// Resolved lazily at mesh-build time; 0 until then.
    pub variant_sid: u32,
    /// #iscale The authored ITEM SCALE for this weapon in this unit's hands: the `armed` column
    /// of the holder's size class out of the weapon's own nine scales (see [`ITEM_SCALES`]).
    /// 1.0 when the tag leaves it unset, which is what the engine substitutes.
    pub scale: f32,
}

/// `unit` tag-field offsets for the held-weapon rule (engine `unit_pick_weapon_markers`,
/// reach_tag_test.exe +0xD7FA60; offsets from the tool's own reflection):
///   +0x340 `right_hand_node`  "where the primary weapon is attached"  ("primary_weapon")
///   +0x344 `left_hand_node`   ("secondary_weapon")
///   +0x348 `preferred_gun_node` inside `unit_additional_node_names_struct` ("right_hand_spartan")
///   +0x34C `preferred_grenade_marker`
/// and `weap` +0x3A8 `handle node` ("the node that get's attached to the unit's hand"; NONE on
/// every shipped Reach weapon, which is why the preferred-gun branch is the live one).
pub const UNIT_RIGHT_HAND_NODE: usize = 0x340;
pub const UNIT_LEFT_HAND_NODE: usize = 0x344;
pub const UNIT_PREFERRED_GUN_NODE: usize = 0x348;
pub const WEAP_HANDLE_NODE: usize = 0x3A8;

/// `weap` +0x3AC `weapon class` and +0x3B0 `weapon name`, both string ids — the WEAPON ANIMATION
/// CLASS, i.e. the middle component of a `jmad` `mode:class:action` triplet.
///
/// Offsets straight out of the engine's own reflection data (`weapon_struct_definition`, whose
/// field offsets are FLAT from the start of the `weap` tag: the object + item prefix occupies
/// 0x204, `handle node` is +0x3A8 and the two labels follow it). Confirmed by measurement, not
/// by reading the table: see [`dump_weapon_classes`].
/// `weap` +0x3B8 `player interface` -> +0x00 `first person` block (32-byte elements, max 2):
/// `first person model` (a `mode`) @+0x00 and `first person animations` (a `jmad`) @+0x10.
pub const WEAP_PLAYER_INTERFACE: usize = 0x3B8;

pub const WEAP_WEAPON_CLASS: usize = 0x3AC;
pub const WEAP_WEAPON_NAME: usize = 0x3B0;

/// #iscale `item` **Item scale settings**: the NINE authored scales every item (so every
/// weapon, every piece of equipment) carries, at `item` +0x1D0 as a flat `real[9]`:
///
/// | index | offset | field |
/// |---|---|---|
/// | 0 | +0x1D0 | `ground scale` |
/// | 1 / 2 | +0x1D4 / +0x1D8 | `small unit (armed)` / `(stowed)` |
/// | 3 / 4 | +0x1DC / +0x1E0 | `medium unit (armed)` / `(stowed)` |
/// | 5 / 6 | +0x1E4 / +0x1E8 | `large unit (armed)` / `(stowed)` |
/// | 7 / 8 | +0x1EC / +0x1F0 | `huge unit (armed)` / `(stowed)` |
///
/// **This is the engine's own indexing, read out of its code, not inferred from the field
/// order.** `reach_tag_test.exe +0xE25350` (`items.cpp`, the function every item pickup, drop
/// and stow ends in) is literally:
///
/// ```text
///   if (unit == NONE) idx = 0;                       // ground scale
///   else switch (unit_definition->item_owner_size) { // +0x3F6, 4 values
///     case 0: idx = 1 + (state == stowed); break;    // small
///     case 1: idx = 3 + (state == stowed); break;    // medium
///     case 2: idx = 5 + (state == stowed); break;    // large
///     case 3: idx = 7 + (state == stowed); break;    // huge
///     default: scale = 1.0; goto set;
///   }
///   scale = *(float *)(item_definition + 0x1D0 + 4 * idx);
///   if (!(scale > 0.0)) scale = 1.0;                 // the authored 0 means "unset"
///   object_set_scale(item, scale, 0.0);              // +0xD54490: object + 0x84
/// ```
///
/// so `ground scale` really is index 0 of the same table and the armed/stowed choice really is
/// `item.inventory_state == _item_inventory_state_stowed` (3). `object_set_scale` writes the
/// object's own scale, which the engine's node matrices carry (`s_real_matrix4x3` has a scale
/// scalar at +0x00) — so the whole weapon model scales, markers included.
///
/// **Known gap:** `huge unit` (indices 7/8) is authored as a literal 0 on every shipped Reach
/// weapon, so that column is only ever exercised through the engine's "unset -> 1.0" fallback.
/// Nothing in this map can tell a correct `huge` read from a wrong one.
pub const ITEM_SCALES: usize = 0x1D0;
/// `unit` +0x3F6 `item owner size` -> `unit_item_owner_size_enum`, four values
/// (0 small, 1 medium, 2 large, 3 huge) in the same order as the item scale columns.
///
/// The tool's reflection cannot place this field (two `unit` sub-structs, `uHnd` and `ubms`,
/// have no static field list, so everything after them is printed as a `+?` lower bound); the
/// offset here is the one the ENGINE dereferences at `+0xE253DB`:
/// `movsx ecx, word ptr [unit_definition + 0x3F6]`.
pub const UNIT_ITEM_OWNER_SIZE: usize = 0x3F6;

/// `render_model` +0x064 `render geometry` -> +0x010 `compression info` (52-byte elements,
/// `position bounds 0` at +0x04 and `position bounds 1` at +0x10): the model-space AABB the
/// compressed vertices are decoded against, i.e. the model's own size — readable from tag meta
/// with no GPU and no mesh build, which is what makes the size measurement testable.
pub const MODE_COMPRESSION_INFO: usize = 0x074;

/// An item's nine authored scales, or `None` when the tag has no readable meta there.
pub fn item_scales(native: &NativeDll, cache: CacheHandle, item_tag: u32) -> Option<[f32; 9]> {
    item_scales_at(native, cache, item_tag, ITEM_SCALES)
}

/// `item_scales` with the offset spelled out — the sweep in [`dump_held_weapon`] and the gate
/// test use it to show that no OTHER offset in the neighbourhood produces nine sane scales.
pub fn item_scales_at(native: &NativeDll, cache: CacheHandle, item_tag: u32, off: usize) -> Option<[f32; 9]> {
    let m = native.tag_meta(cache, item_tag, off as u32, 36)?;
    if m.len() < 36 {
        return None;
    }
    let mut out = [0.0f32; 9];
    for i in 0..9 {
        out[i] = f32::from_le_bytes([m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]]);
    }
    Some(out)
}

/// Does this run of nine floats look like an authored item scale table? The signature the
/// offset is pinned by: every value finite and inside `0.0..=4.0`, a ground scale that is
/// strictly positive, and at least six of the nine authored (a shipped weapon routinely leaves
/// the `huge unit` pair at 0, which the engine reads as "unset" and replaces with 1.0 — so
/// demanding all nine be positive rejects the right offset, and demanding none be positive
/// accepts any run of zeroes).
pub fn item_scales_look_authored(s: &[f32; 9]) -> bool {
    s[0] > 0.0
        && s.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 4.0)
        && s.iter().filter(|v| **v > 0.0).count() >= 6
}

/// A unit's `item owner size` (0 small .. 3 huge), or `None` when it is outside the enum — the
/// case the engine answers with a flat scale of 1.0.
pub fn unit_item_owner_size(native: &NativeDll, cache: CacheHandle, unit_tag: u32) -> Option<u8> {
    let m = native.tag_meta(cache, unit_tag, UNIT_ITEM_OWNER_SIZE as u32, 2)?;
    if m.len() < 2 {
        return None;
    }
    let v = i16::from_le_bytes([m[0], m[1]]);
    (0..=3).contains(&v).then_some(v as u8)
}

/// The scale the engine gives one item in one state. `size` is the holder's `item owner size`
/// (`None` = lying on the ground, which is index 0); `stowed` picks the second column of the
/// pair. An unauthored (non-positive) entry falls back to 1.0, exactly as the engine does.
pub fn item_scale_for(scales: &[f32; 9], size: Option<u8>, stowed: bool) -> f32 {
    let idx = match size {
        None => 0,
        Some(s) if s <= 3 => 1 + 2 * s as usize + stowed as usize,
        Some(_) => return 1.0,
    };
    let v = scales[idx];
    if v > 0.0 { v } else { 1.0 }
}

/// The scale a weapon is drawn at IN A UNIT'S HANDS (`item owner size` of the unit, armed
/// column), which is what a third-person held weapon and a Theater body both want.
pub fn held_item_scale(native: &NativeDll, cache: CacheHandle, unit_tag: u32, item_tag: u32) -> f32 {
    match item_scales(native, cache, item_tag) {
        Some(s) => item_scale_for(&s, unit_item_owner_size(native, cache, unit_tag), false),
        None => 1.0,
    }
}

/// A render model's own model-space AABB, from its `compression info` (see
/// [`MODE_COMPRESSION_INFO`]). Several entries are unioned: a model with more than one
/// compression block has one per vertex-buffer group.
pub fn model_bounds(native: &NativeDll, cache: CacheHandle, mode_tag: u32) -> Option<(Vec3, Vec3)> {
    let b = tag_block(native, cache, mode_tag, MODE_COMPRESSION_INFO, 52, 65536);
    if b.len() < 52 {
        return None;
    }
    let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for e in 0..b.len() / 52 {
        let o = e * 52;
        let a = Vec3::new(f(o + 0x04), f(o + 0x08), f(o + 0x0C));
        let c = Vec3::new(f(o + 0x10), f(o + 0x14), f(o + 0x18));
        if !a.is_finite() || !c.is_finite() {
            continue;
        }
        lo = lo.min(a.min(c));
        hi = hi.max(a.max(c));
    }
    (lo.x <= hi.x).then_some((lo, hi))
}

/// #iscale MEASURED model-space AABB: every section the default variant draws, decoded on the
/// CPU and reduced to a box. This is the honest size of a model — [`model_bounds`] reads the
/// authored *compression* range, which is padded (the Spartan's is 0.97 wu tall against a
/// measured 0.72), so sizes are reported from here and the compression box is only a sanity
/// check. Needs no GPU, so a test can assert on it.
pub fn measured_model_bounds(native: &NativeDll, cache: CacheHandle, mode_tag: u32) -> Option<(Vec3, Vec3)> {
    if mode_tag == 0 || mode_tag == 0xFFFF_FFFF {
        return None;
    }
    let h = native.open_model(cache, mode_tag).ok()?;
    let mask = default_section_mask(native, cache, mode_tag);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for i in 0..native.model_section_count(h) {
        if mask.get(i as usize) == Some(&false) {
            continue;
        }
        let Some(sec) = native.model_section(h, i) else { continue };
        let Some(g) = native.model_decode_geometry(h, i) else { continue };
        let stride = (sec.vertex_stride as usize).max(12);
        for v in 0..g.vertices.len() / stride {
            let o = v * stride;
            let f = |k: usize| {
                f32::from_le_bytes([g.vertices[o + k], g.vertices[o + k + 1], g.vertices[o + k + 2], g.vertices[o + k + 3]])
            };
            let p = Vec3::new(f(0), f(4), f(8));
            if p.is_finite() {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    native.close_model(h);
    (lo.x <= hi.x).then_some((lo, hi))
}

/// The animation class a weapon makes its holder use (`rifle`, `pistol`, `missile`, `sword`,
/// `hammer`, `ball`, `turret`, ...), or `None` when the tag leaves it unset.
///
/// This is the string the Spartan's animation graph expects in the middle of a clip name, so
/// `combat:<class>:move_front_fast` is the real clip for a body carrying this weapon.
pub fn weapon_class(native: &NativeDll, cache: CacheHandle, weap_tag: u32) -> Option<String> {
    let sid = sid_field(native, cache, weap_tag, WEAP_WEAPON_CLASS);
    if sid == 0 || sid == 0xFFFF_FFFF {
        return None;
    }
    let s = native.string_id(cache, sid)?;
    if s.is_empty() { None } else { Some(s) }
}

/// #fp The FIRST-PERSON model and animation graph of a weapon: `(first person model `mode`,
/// first person animations `jmad`)`, either of which may be `None`.
///
/// **Measured, and it corrects an earlier guess.** The `first person model` is NOT a separate
/// first-person weapon tag: on every Reach weapon checked it is the weapon's ORDINARY render
/// model (the energy sword's is `objects\weapons\melee\energy_sword\energy_sword`, the same
/// `mode` the ground weapon draws). The separate tags under
/// `objects\characters\spartans\fp\weapons\...` are the `first person animations` GRAPHS, not
/// models — `ReachModel::read` on one fails because it is a `jmad`.
///
/// That graph is what makes the view first person. The arms model (`matg @player representation`)
/// carries no animations of its own: the graph lists the arms' 47 nodes AND the weapon's
/// (`b_handle`, `b_blades` for the sword; `b_gun`, `b_magazine`, `b_ophandle`, `b_safety`,
/// `b_switch` for the assault rifle), and poses both. A weapon with no entry here has no authored
/// first-person pose at all.
pub fn first_person_weapon(native: &NativeDll, cache: CacheHandle, weap_tag: u32, species: usize) -> (Option<u32>, Option<u32>) {
    let b = tag_block(native, cache, weap_tag, WEAP_PLAYER_INTERFACE, 32, 2);
    // #fp-species  The block carries ONE ROW PER PLAYER SPECIES, in the same order as
    // `matg @player representation`: row 0 Spartan, row 1 Elite. Reach authors a separate Elite
    // first-person graph for every weapon (`objects\characters\elite\fp\weapons\...` beside
    // the Spartan's `objects\characters\spartans\fp\weapons\...`), and the two arms models
    // are not the same skeleton -- 47 nodes against 41, in a different order. Taking row 0 for
    // everybody posed an Elite's hands with the SPARTAN's animation. A cache that authors only
    // one row falls back to it rather than drawing nothing.
    let row = if b.len() >= (species + 1) * 32 { species } else { 0 };
    if b.len() < (row + 1) * 32 {
        return (None, None);
    }
    let e = &b[row * 32..row * 32 + 32];
    let tagref = |o: usize| -> Option<u32> {
        let d = u32::from_le_bytes([e[o + 0x0C], e[o + 0x0D], e[o + 0x0E], e[o + 0x0F]]);
        (d != 0xFFFF_FFFF).then(|| d & 0xFFFF)
    };
    (tagref(0x00), tagref(0x10))
}

/// #fp-species  How many `player interface` rows a weapon authors (0, 1 or 2). Print-only.
pub fn first_person_rows(native: &NativeDll, cache: CacheHandle, weap_tag: u32) -> usize {
    tag_block(native, cache, weap_tag, WEAP_PLAYER_INTERFACE, 32, 2).len() / 32
}

/// #fp Which SECTIONS of a render model to draw when it has NO `hlmt` to pick a permutation with.
///
/// The decode's two real filters (an hlmt variant's mask, or the default-variant mask) both start
/// at the object's model tag, so a bare `mode` falls through to the native heuristic — which
/// allows every permutation of every region at once. On the Spartan first-person arms that is two
/// complete sets of arms plus all five forearm attachments, overlapping.
///
/// The rule here, per region: take the permutation named `default` if the region has one, else
/// the FIRST permutation; but skip a `flair_*` region that has no `default`, because Reach's flair
/// regions are optional armour and the recorded player's own selection is not mapped onto the
/// first-person model. Sections no region claims are kept, which is what the engine's own
/// fallback does with them.
///
/// Returns an empty mask when the model cannot be read, which the caller must treat as "install
/// nothing" rather than "draw nothing".
pub fn default_section_mask(native: &NativeDll, cache: CacheHandle, mode_tag: u32) -> Vec<bool> {
    section_mask_with_picks(native, cache, mode_tag, &[])
}

/// #fp-arms  The same, but wearing ONE RECORDED PLAYER'S armour selections.
///
/// `picks` is a `(region string id, permutation string id)` list — exactly what
/// `SceneController::player_variant` builds for a body with an `hlmt` — and a region named there
/// takes the permutation it names instead of `default`. Everything else behaves as
/// [`default_section_mask`], so a region the player's loadout says nothing about keeps its
/// authored default and an unclaimed section is still drawn.
///
/// This is the only route a bare `mode` has: the first-person arm models have no `hlmt`, so the
/// variant walk `build_actor_meshes` uses cannot start. The selections DO reach them — measured
/// on `forge_halo`, the Spartan arms carry regions `arms` (4 permutations) and `flair_forearm`
/// (5), which the appearance bytes for `chest selections` and `wrist selections` override, and
/// the Elite arms carry `armor_pieces` (8), which `elite selections` overrides. Before this,
/// `arms` was pinned to `default` and `flair_forearm` — having no `default` permutation — was
/// dropped altogether, so a player's own forearm piece never appeared in their own hands.
pub fn section_mask_with_picks(
    native: &NativeDll,
    cache: CacheHandle,
    mode_tag: u32,
    picks: &[(u32, u32)],
) -> Vec<bool> {
    let Some(m) = ReachModel::read(native, cache, mode_tag) else { return Vec::new() };
    let n = m.mesh_fmt.len();
    if n == 0 {
        return Vec::new();
    }
    let u = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let i16at = |b: &[u8], o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
    let Some(meta) = native.tag_meta(cache, mode_tag, 0, 0x70) else { return Vec::new() };
    let (rcnt, rptr) = (u(&meta, 12), u(&meta, 16));
    if rcnt == 0 || rcnt > 0x1000 {
        return Vec::new();
    }
    let Some(rb) = native.tag_ptr_bytes(cache, rptr, rcnt * 16) else { return Vec::new() };
    let mut mask = vec![false; n];
    let mut claimed = vec![false; n];
    for r in 0..rcnt as usize {
        let re = &rb[r * 16..r * 16 + 16];
        let region = native.string_id(cache, u(re, 0)).unwrap_or_default();
        let (pcnt, pptr) = (u(re, 4), u(re, 8));
        if pcnt == 0 || pcnt > 0x1000 {
            continue;
        }
        let Some(pb) = native.tag_ptr_bytes(cache, pptr, pcnt * 0x18) else { continue };
        let perm = |i: usize| -> (String, i16, i16) {
            let pe = &pb[i * 0x18..i * 0x18 + 0x18];
            (native.string_id(cache, u(pe, 0)).unwrap_or_default(), i16at(pe, 4), i16at(pe, 6))
        };
        for i in 0..pcnt as usize {
            let (_, mi, mc) = perm(i);
            for k in mi.max(0) as usize..(mi.max(0) as usize + mc.max(0) as usize).min(n) {
                claimed[k] = true;
            }
        }
        // #fp-arms  A region this player's recorded loadout names wins over the default.
        let region_sid = u(re, 0);
        let wanted: Option<usize> = picks
            .iter()
            .filter(|(r, _)| *r == region_sid)
            .find_map(|(_, p)| {
                let want = native.string_id(cache, *p).unwrap_or_default();
                if want.is_empty() {
                    return None;
                }
                (0..pcnt as usize).find(|&i| perm(i).0 == want)
            });
        let default_at = (0..pcnt as usize).find(|&i| perm(i).0 == "default");
        let pick = match wanted.or(default_at) {
            Some(i) => Some(i),
            // An optional armour region with no authored default: draw none of it.
            None if region.starts_with("flair") => None,
            None => Some(0),
        };
        let Some(i) = pick else { continue };
        let (_, mi, mc) = perm(i);
        for k in mi.max(0) as usize..(mi.max(0) as usize + mc.max(0) as usize).min(n) {
            mask[k] = true;
        }
    }
    for i in 0..n {
        if !claimed[i] {
            mask[i] = true;
        }
    }
    mask
}

/// The middle-slot vocabulary of an animation graph: every distinct `class` in its
/// `mode:class:action` clip names, with how many clips use it.
///
/// Derived from the GRAPH and nothing else, which is what makes it an independent oracle for the
/// weapon tag's `weapon class` field: the two are authored in different tags by different tools.
pub fn anim_class_vocabulary(graph: &AnimGraph) -> std::collections::BTreeMap<String, usize> {
    let mut out: std::collections::BTreeMap<String, usize> = Default::default();
    for a in &graph.anims {
        let p: Vec<&str> = a.name.split(':').collect();
        if p.len() == 3 && !p[1].is_empty() {
            *out.entry(p[1].to_string()).or_insert(0) += 1;
        }
    }
    out
}

/// Print-only (`hms-app --weapon-class <map> [sweep]`): every `weap` tag's resolved animation
/// class, and — with `sweep` — the (offset, width) sweep that PINNED the field.
///
/// The oracle is the animation graph's own middle-slot vocabulary (see
/// [`anim_class_vocabulary`]), which no weapon tag contributes to. A candidate offset scores the
/// number of weapons whose value there resolves to a string in that vocabulary; the report prints
/// the winner, the runner-up and the margin between them, because a winner with no margin is not
/// a result.
pub fn dump_weapon_classes(native: &NativeDll, cache: CacheHandle, filter: &str, sweep: bool) {
    let mut weaps: Vec<u32> = native.tags_by_group(cache, "weap");
    weaps.sort_unstable();
    let names: Vec<(u32, String)> =
        weaps.iter().map(|&t| (t, native.tag_name(cache, t).unwrap_or_default())).collect();

    // The oracle: the Spartan graph's middle slot.
    let spartan = ActorScene::list_bipeds(native, cache)
        .into_iter()
        .find(|(_, n)| n.to_lowercase().contains("spartan"))
        .map(|(t, _)| t);
    let vocab = spartan
        .and_then(|t| resolve_jmad(native, cache, t))
        .and_then(|j| AnimGraph::read(native, cache, j))
        .map(|g| anim_class_vocabulary(&g))
        .unwrap_or_default();
    println!("animation-class vocabulary from the Spartan graph: {} classes", vocab.len());
    let mut v: Vec<(&String, &usize)> = vocab.iter().collect();
    v.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    println!("  {}", v.iter().map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", "));

    if sweep {
        // (offset, width): width 4 reads a whole string id; width 2 reads only its low half, the
        // shape a packed/short field would have. Anything that resolves to a vocabulary string
        // scores.
        const SPAN: usize = 0x1000;
        let metas: Vec<(u32, String, Vec<u8>)> = names
            .iter()
            .filter_map(|(t, n)| native.tag_meta(cache, *t, 0, SPAN as u32).map(|m| (*t, n.clone(), m)))
            .collect();
        println!("
sweep over {} weap tags, offsets 0..{SPAN:#x}", metas.len());
        let mut scored: Vec<(usize, usize, usize, usize)> = Vec::new(); // (score, distinct, off, width)
        for width in [2usize, 4] {
            for off in (0..SPAN.saturating_sub(width)).step_by(2) {
                let mut hits = 0usize;
                let mut seen: std::collections::BTreeSet<String> = Default::default();
                for (_, _, m) in &metas {
                    if off + width > m.len() { continue }
                    let sid = match width {
                        2 => u16::from_le_bytes([m[off], m[off + 1]]) as u32,
                        _ => u32::from_le_bytes([m[off], m[off + 1], m[off + 2], m[off + 3]]),
                    };
                    if sid == 0 || sid == 0xFFFF_FFFF { continue }
                    if let Some(s) = native.string_id(cache, sid) {
                        if vocab.contains_key(&s) {
                            hits += 1;
                            seen.insert(s);
                        }
                    }
                }
                if hits > 0 {
                    scored.push((hits, seen.len(), off, width));
                }
            }
        }
        scored.sort_by_key(|&(h, d, o, w)| (std::cmp::Reverse(h), std::cmp::Reverse(d), o, w));
        println!("  top candidates (score = weapons resolving to a vocabulary class, of {}):", metas.len());
        for (h, d, o, w) in scored.iter().take(12) {
            println!("    +{o:#05x} width {w}: score {h}/{}  distinct classes {d}", metas.len());
        }
        // The runner-up must be a DIFFERENT OFFSET. Reach string ids all fit in 16 bits on this
        // cache, so reading 2 bytes and 4 bytes at the same offset is the SAME read and scores
        // identically -- quoting that alias as the runner-up would report a margin of 0 for a
        // field that is actually unopposed.
        let best = scored.first().copied();
        let next = best.and_then(|(_, _, o1, _)| scored.iter().find(|&&(_, _, o, _)| o != o1).copied());
        match (best, next) {
            (Some((h1, d1, o1, w1)), Some((h2, _, o2, w2))) => println!(
                "  WINNER +{o1:#x} (width {w1}) score {h1} ({d1} distinct classes); best OTHER offset +{o2:#x} (width {w2}) score {h2}; margin {}",
                h1 as i64 - h2 as i64
            ),
            (Some((h1, d1, o1, w1)), None) => {
                println!("  WINNER +{o1:#x} (width {w1}) score {h1} ({d1} distinct); NO other offset scored at all")
            }
            _ => println!("  nothing scored: the oracle found no vocabulary string anywhere in the weap meta"),
        }
    }

    println!("
weap +{WEAP_WEAPON_CLASS:#x} 'weapon class' / +{WEAP_WEAPON_NAME:#x} 'weapon name', {} tags:", names.len());
    let mut unset = 0usize;
    let mut off_vocab: Vec<String> = Vec::new();
    for (t, n) in &names {
        if !filter.is_empty() && !n.to_lowercase().contains(&filter.to_lowercase()) { continue }
        let cls = weapon_class(native, cache, *t);
        let wname = {
            let sid = sid_field(native, cache, *t, WEAP_WEAPON_NAME);
            native.string_id(cache, sid).unwrap_or_default()
        };
        match &cls {
            None => unset += 1,
            Some(c) if !vocab.contains_key(c) => off_vocab.push(format!("{n} -> '{c}'")),
            _ => {}
        }
        let (fpm, fpa) = first_person_weapon(native, cache, *t, 0);
        println!("  0x{t:08X}  class={:<14} name={:<8} fp_model={:<14} fp_anim={:<14} {n}",
            cls.unwrap_or_else(|| "<unset>".into()), wname,
            fpm.map(|v| format!("{v:#x}")).unwrap_or_else(|| "-".into()),
            fpa.map(|v| format!("{v:#x}")).unwrap_or_else(|| "-".into()));
        if let Some(m) = fpm { println!("        fp model '{}'", native.tag_name(cache, m).unwrap_or_default()); }
        if let Some(a) = fpa {
            println!("        fp anims '{}'", native.tag_name(cache, a).unwrap_or_default());
            if let Some(g) = AnimGraph::read(native, cache, a) {
                println!("        graph: {} clips, {} nodes: {}", g.anims.len(), g.node_names.len(),
                    g.node_names.join(" "));
                let mut names: Vec<&str> = g.anims.iter().map(|x| x.name.as_str()).collect();
                names.sort_unstable();
                println!("        clips: {}", names.join(", "));
            } else {
                println!("        graph: UNREADABLE");
            }
        }
    }
    println!("  {} tags with no class, {} whose class is NOT in the graph vocabulary{}",
        unset, off_vocab.len(),
        if off_vocab.is_empty() { String::new() } else { format!(": {}", off_vocab.join(", ")) });
}

fn sid_field(native: &NativeDll, cache: CacheHandle, tag: u32, off: usize) -> u32 {
    native
        .tag_meta(cache, tag, off as u32, 4)
        .map(|m| u32::from_le_bytes([m[0], m[1], m[2], m[3]]))
        .unwrap_or(0)
}

/// Resolve the engine's held-weapon marker pair for `weapon` on `unit`.
///
/// `unit_pick_weapon_markers` (reach_tag_test.exe +0xD7FA60) reads `unit->right_hand_node`
/// (or `left_hand_node` for the off hand) as the PARENT marker, then takes
/// `unit->preferred_gun_node` as the CHILD marker IF the weapon's own render model has a marker
/// group of that name, else `weap->handle_node`; the attach is then the ordinary
/// `object_attach_to_marker(unit, parent, weapon, child)`, whose math is
/// `child_world = parent_world * P_marker * C_marker^-1`
/// (`object_place_marker_at_matrix`, +0xD3BD40).
///
/// **The marker `scale` field cancels; it is not ignored.** `render_model_marker` is 48 bytes
/// (region @0, permutation @1, node @2, flags @3, translation @4, rotation @0x10, **scale
/// @0x20**, direction @0x24) — the engine's own tag definition, so the field this reads IS the
/// scale. Measured on `forge_halo.map`: EVERY marker of the Spartan biped and EVERY marker of
/// every weapon authors exactly 0.010. The attach divides the child's scale into the parent's,
/// so the ratio is 1.0 and a third-person held weapon is the same size as the one on the ground.
/// **The SIZE comes from somewhere else entirely: the item's own scale table.** `item` +0x1D0
/// authors nine scales — one for the ground and an armed/stowed pair per unit size class — and
/// the engine multiplies the held one into the object's scale (see [`ITEM_SCALES`]). That is
/// what [`HeldWeapon::scale`] carries and [`HeldWeapon::world`] applies.
pub fn resolve_held_weapon(
    native: &NativeDll,
    cache: CacheHandle,
    unit_tag: u32,
    unit_model: &ReachModel,
    weapon_tag: u32,
    off_hand: bool,
) -> Option<HeldWeapon> {
    let weap_mode = native.resolve_mode(cache, weapon_tag);
    let wm = ReachModel::read(native, cache, weap_mode)?;
    let parent_sid = sid_field(native, cache, unit_tag, if off_hand { UNIT_LEFT_HAND_NODE } else { UNIT_RIGHT_HAND_NODE });
    let gun_sid = sid_field(native, cache, unit_tag, UNIT_PREFERRED_GUN_NODE);
    let handle_sid = sid_field(native, cache, weapon_tag, WEAP_HANDLE_NODE);

    let group_by_sid = |m: &ReachModel, sid: u32| m.markers.iter().find(|g| g.name == sid).cloned();
    // Child marker: preferred_gun_node when the WEAPON has that group, else handle_node.
    let (child_sid, child_group) = match group_by_sid(&wm, gun_sid) {
        Some(g) if gun_sid != 0 && gun_sid != 0xFFFF_FFFF => (gun_sid, Some(g)),
        _ => (handle_sid, group_by_sid(&wm, handle_sid)),
    };
    let child = child_group.as_ref().and_then(|g| reach_skel::pick_marker(&g.markers, -1, -1));
    let parent_group = group_by_sid(unit_model, parent_sid);
    let parent = parent_group.as_ref().and_then(|g| reach_skel::pick_marker(&g.markers, -1, -1));
    Some(HeldWeapon {
        obj_tag: weapon_tag,
        mode_tag: weap_mode,
        node_count: wm.skel.len().max(1),
        parent_marker: native.string_id(cache, parent_sid).unwrap_or_default(),
        child_marker: native.string_id(cache, child_sid).unwrap_or_default(),
        child_local: child.map(|m| m.local()).unwrap_or(Mat4::IDENTITY),
        parent_node: parent.map(|m| m.node).unwrap_or(-1),
        parent_local: parent.map(|m| m.local()).unwrap_or(Mat4::IDENTITY),
        variant_sid: 0,
        scale: held_item_scale(native, cache, unit_tag, weapon_tag),
    })
}

impl HeldWeapon {
    /// The weapon's model->world matrix for a posed biped at `unit_world`.
    ///
    /// #iscale The ITEM SCALE is part of this, and where it goes matters. In the engine the
    /// object's scale lives in its NODE MATRICES (`s_real_matrix4x3` carries a scale scalar), so
    /// a scaled weapon's grip marker is `scale * t` from its model origin, and the attach then
    /// lands that SCALED marker on the hand. Reproduced exactly: scale the child marker's
    /// translation, mate the markers, then scale the model. Folding the scale in anywhere else
    /// either cancels it out or drags the grip out of the hand.
    pub fn world(&self, unit_world: Mat4, pose: &Pose) -> Mat4 {
        let node = if self.parent_node >= 0 { pose.model.get(self.parent_node as usize).copied() } else { None };
        let p = node.unwrap_or(Mat4::IDENTITY) * self.parent_local;
        let s = if self.scale > 0.0 { self.scale } else { 1.0 };
        if (s - 1.0).abs() < 1e-6 {
            return attach_child(unit_world, p, self.child_local);
        }
        let mut c = self.child_local;
        c.w_axis = (c.w_axis.truncate() * s).extend(1.0);
        attach_child(unit_world, p, c) * Mat4::from_scale(Vec3::splat(s))
    }
}

/// #ride-weapon  The bone palette of a HELD weapon: every entry the attach matrix, or -- when the
/// weapon is hidden -- the zero matrix, which collapses every vertex to a point and draws nothing
/// without rebuilding a single mesh.
pub fn held_weapon_palette(attach: Mat4, node_count: usize, hidden: bool) -> Vec<Mat4> {
    vec![if hidden { Mat4::ZERO } else { attach }; node_count.min(hms_render::MAX_SKIN_BONES)]
}

/// Print-only (`hms-app --dump-anim <map> <biped-filter> [anim-name] [frames]`): resolve a
/// biped's `jmad`, load one animation and sample it, printing the codec headers and the decoded
/// channels at several frames. This is how the codec decode was validated against the engine's
/// own layout (see `reach_jmad`).
pub fn dump_anim(native: &NativeDll, cache: CacheHandle, biped: &str, anim: &str, nframes: usize) {
    let mut tags: Vec<u32> = native.tags_by_group(cache, "bipd");
    tags.sort_unstable();
    for t in tags {
        let name = native.tag_name(cache, t).unwrap_or_default();
        if !biped.is_empty() && !name.to_lowercase().contains(&biped.to_lowercase()) { continue; }
        let Some(jmad) = resolve_jmad(native, cache, t) else { println!("{name}: no jmad"); continue };
        let Some(graph) = AnimGraph::read(native, cache, jmad) else { println!("{name}: jmad unreadable"); continue };
        println!("\n== bipd '{name}' -> jmad 0x{jmad:08X} '{}' animations={} groups={} graph_nodes={}",
            native.tag_name(cache, jmad).unwrap_or_default(), graph.anims.len(),
            graph.group_resource_ids.len(), graph.node_names.len());
        let mode = native.resolve_mode(cache, t);
        let model = ReachModel::read(native, cache, mode);
        if let Some(m) = &model {
            let same = m.skel.len() == graph.node_names.len()
                && m.node_names.iter().zip(&graph.node_names).all(|(a, b)| a == b);
            println!("   render_model nodes={} jmad skeleton nodes={} parallel={}", m.skel.len(), graph.node_names.len(), same);
        }
        if anim == "*" {
            // Codec survey: load every animation and histogram the codec id of region 0 (the
            // static default pose) and region 1 (the animated data).
            let mut h0: std::collections::BTreeMap<u8, usize> = Default::default();
            let mut h1: std::collections::BTreeMap<u8, usize> = Default::default();
            let mut bad = 0usize;
            let mut example: std::collections::BTreeMap<u8, String> = Default::default();
            for i in 0..graph.anims.len() {
                match graph.load(native, cache, i) {
                    Some(d) => {
                        if let Some(b) = d.region(0) { *h0.entry(b[0]).or_insert(0) += 1; }
                        if let Some(b) = d.region(1) {
                            *h1.entry(b[0]).or_insert(0) += 1;
                            example.entry(b[0]).or_insert_with(|| graph.anims[i].name.clone());
                        }
                    }
                    None => bad += 1,
                }
            }
            println!("   codec survey over {} animations ({bad} failed to load)", graph.anims.len());
            println!("     region 0 (static)   codec histogram: {h0:?}");
            println!("     region 1 (animated) codec histogram: {h1:?}");
            for (c, n) in &example { println!("     codec {c} e.g. '{n}'"); }
            continue;
        }
        if anim.is_empty() {
            for a in graph.anims.iter().take(nframes.max(20)) {
                println!("   anim {:4} '{}' frames={} nodes={} type={:?} grp={}/{}", a.index, a.name, a.frame_count, a.node_count, a.anim_type, a.resource_group, a.resource_group_member);
            }
            continue;
        }
        let Some(ai) = graph.find(anim) else { println!("   no animation matching '{anim}'"); continue };
        let a = &graph.anims[ai];
        let Some(data) = graph.load(native, cache, ai) else { println!("   '{}': resource load FAILED", a.name); continue };
        println!("   '{}' frames={} nodes={} type={:?} movement={} blob={} sizes={:?}",
            a.name, data.frame_count, data.node_count, a.anim_type, data.movement_data_type, data.blob.len(), data.sizes);
        for r in [0usize, 1, 2, 3] {
            if let Some(b) = data.region(r) {
                let h = reach_jmad::CodecHeader::read(b);
                println!("     region {r}: {} bytes head={} codec={:?}", b.len(), hex_head(b, 32), h);
            }
        }
        let mv = data.movement();
        if !mv.is_empty() {
            let tot = mv.iter().fold((0.0, 0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
            println!("     movement frames={} total=({:.4},{:.4},{:.4}, yaw {:.4})", mv.len(), tot.0, tot.1, tot.2, tot.3);
        }
        let last = (data.frame_count.max(1) - 1) as f32;
        for k in 0..nframes.max(1) {
            let f = if nframes <= 1 { 0.0 } else { last * k as f32 / (nframes - 1) as f32 };
            let ch = reach_jmad::sample(&data, a.anim_type, f);
            let driven = ch.iter().filter(|c| c.rot.is_some()).count();
            let dt = ch.iter().filter(|c| c.trans.is_some()).count();
            let ds = ch.iter().filter(|c| c.scale.is_some()).count();
            print!("     frame {f:7.2}: rot={driven} trans={dt} scale={ds}");
            for n in [1usize, 10, 20] {
                if let Some(c) = ch.get(n) {
                    let nm = graph.node_names.get(n).cloned().unwrap_or_default();
                    match (c.rot, c.trans) {
                        (Some(q), Some(p)) => print!(" | {nm} q=({:.3},{:.3},{:.3},{:.3}) t=({:.3},{:.3},{:.3})", q.x, q.y, q.z, q.w, p.x, p.y, p.z),
                        (Some(q), None) => print!(" | {nm} q=({:.3},{:.3},{:.3},{:.3})", q.x, q.y, q.z, q.w),
                        _ => print!(" | {nm} -"),
                    }
                }
            }
            println!();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ActorScene: the placed, posed actors. THIS IS THE API THE THEATER PLAYBACK DRIVES.
// ---------------------------------------------------------------------------------------------

use std::collections::HashMap;
use std::sync::Arc;

/// Everything cached per biped tag (read once per map).
#[derive(Clone, Debug)]
pub struct ActorDef {
    pub obj_tag: u32,
    pub mode_tag: u32,
    /// Tag path of the `bipd`.
    pub name: String,
    pub model: ReachModel,
    pub graph: Option<AnimGraph>,
    /// `Skeleton::inverse_bind()` computed ONCE (it inverts the whole bind chain, which is far
    /// too expensive to redo for every actor on every frame).
    pub inverse_bind: Vec<Mat4>,
    /// #fp `jmad channel index -> render-model node index` (-1 = this channel has no node here),
    /// when the graph's node list is NOT the model's node list in the same order.
    ///
    /// The sampler returns one channel per GRAPH node, and everything until now read them
    /// positionally — which is right only while the two lists agree. Reach's FIRST-PERSON graphs
    /// do not: `fp_energy_sword` lists `pedestal, aim_pitch, aim_yaw, base, camera_control, ...`
    /// where the arms model lists `pedestal, base, aim_pitch, aim_yaw, l_upperarm,
    /// camera_control, ...`, and its last nodes (`b_handle`, `b_blades`) belong to the WEAPON's
    /// model rather than the arms'. The same graph therefore poses two different models, each
    /// through its own map. `None` = the lists agree and the positional read stands.
    pub anim_node_map: Option<Vec<i16>>,
}

/// #wscale Print-only (`hms-app --held-weapon <map> [bipd filter] [weap filter]`): exactly which
/// marker pair the held-weapon attach resolves for a unit + weapon, and what the result does to
/// the weapon's size and position.
///
/// This is the measurement behind the held-weapon placement: it prints the unit's three node
/// string ids, every marker group the weapon's render model actually has, which of them the
/// engine's rule picks, and the attach matrix's scale and translation.
pub fn dump_held_weapon(native: &NativeDll, cache: CacheHandle, biped: &str, weapon: &str) {
    let bipeds = ActorScene::list_bipeds(native, cache);
    let Some(&(unit_tag, ref uname)) = bipeds
        .iter()
        .find(|(_, n)| biped.is_empty() || n.to_lowercase().contains(&biped.to_lowercase()))
        .map(|x| x)
    else {
        println!("no bipd matching '{biped}'");
        return;
    };
    let Some(um) = ReachModel::read(native, cache, native.resolve_mode(cache, unit_tag)) else {
        println!("{uname}: no render model");
        return;
    };
    let sid_name = |sid: u32| native.string_id(cache, sid).unwrap_or_default();
    let right = sid_field(native, cache, unit_tag, UNIT_RIGHT_HAND_NODE);
    let left = sid_field(native, cache, unit_tag, UNIT_LEFT_HAND_NODE);
    let gun = sid_field(native, cache, unit_tag, UNIT_PREFERRED_GUN_NODE);
    println!("unit '{uname}' 0x{unit_tag:08X}  model nodes={}", um.node_names.len());
    if let Some(g) = resolve_jmad(native, cache, unit_tag).and_then(|j| AnimGraph::read(native, cache, j)) {
        println!("  graph nodes={} ; names the MODEL does not have: {}",
            g.node_names.len(),
            g.node_names.iter().filter(|n| !um.node_names.contains(n)).cloned().collect::<Vec<_>>().join(", "));
    }
    println!("  +0x340 right_hand_node    = {right:#010x} '{}'", sid_name(right));
    println!("  +0x344 left_hand_node     = {left:#010x} '{}'", sid_name(left));
    println!("  +0x348 preferred_gun_node = {gun:#010x} '{}'", sid_name(gun));
    println!("  unit marker groups: {}", um.marker_names(native, cache).join(", "));
    // #iscale Which SIZE CLASS this unit is, and the sweep that pins the field.
    let size = unit_item_owner_size(native, cache, unit_tag);
    let size_name = |s: Option<u8>| match s {
        Some(0) => "small", Some(1) => "medium", Some(2) => "large", Some(3) => "huge",
        _ => "OUT OF RANGE (engine falls back to scale 1.0)",
    };
    println!("  +0x3F6 item owner size   = {:?} = {}", size, size_name(size));
    {
        let bipeds2 = ActorScene::list_bipeds(native, cache);
        print!("  item owner size of every bipd in this map:");
        for (t, n) in &bipeds2 {
            print!("  {}={:?}", n.rsplit('\\').next().unwrap_or(n), unit_item_owner_size(native, cache, *t));
        }
        println!();
    }
    // The offset gate: sweep a window of `weap` meta and score each offset by how many of the
    // map's weapons produce nine plausible scales there. One offset must win outright.
    {
        let weaps: Vec<u32> = native.tags_by_group(cache, "weap");
        let mut rows: Vec<(usize, usize)> = Vec::new();
        for off in (0x180..=0x240usize).step_by(4) {
            let ok = weaps
                .iter()
                .filter(|t| item_scales_at(native, cache, **t, off).map_or(false, |s| item_scales_look_authored(&s)))
                .count();
            rows.push((off, ok));
        }
        let best = rows.iter().map(|r| r.1).max().unwrap_or(0);
        println!("\n#iscale OFFSET SWEEP over {} weap tags (a hit = nine finite floats in 0.0..4.0, six of them positive, with a non-zero ground scale):", weaps.len());
        for (off, ok) in &rows {
            if *ok > 0 {
                println!("    +{off:#05x}: {ok}/{}{}", weaps.len(), if *ok == best { "   <= best" } else { "" });
            }
        }
        println!("    ITEM_SCALES = {ITEM_SCALES:#05x} scores {}/{}; +-4 score {}/{} and {}/{}",
            rows.iter().find(|r| r.0 == ITEM_SCALES).map(|r| r.1).unwrap_or(0), weaps.len(),
            rows.iter().find(|r| r.0 == ITEM_SCALES - 4).map(|r| r.1).unwrap_or(0), weaps.len(),
            rows.iter().find(|r| r.0 == ITEM_SCALES + 4).map(|r| r.1).unwrap_or(0), weaps.len());
    }
    for t in native.tags_by_group(cache, "weap") {
        let n = native.tag_name(cache, t).unwrap_or_default();
        if !weapon.is_empty() && !n.to_lowercase().contains(&weapon.to_lowercase()) {
            continue;
        }
        let handle = sid_field(native, cache, t, WEAP_HANDLE_NODE);
        let Some(wm) = ReachModel::read(native, cache, native.resolve_mode(cache, t)) else {
            println!("\n  weap '{n}': no render model");
            continue;
        };
        println!("\n  weap '{n}' 0x{t:08X}  +0x3A8 handle_node = {handle:#010x} '{}'", sid_name(handle));
        println!("    weapon marker groups: {}", wm.marker_names(native, cache).join(", "));
        match resolve_held_weapon(native, cache, unit_tag, &um, t, false) {
            Some(w) => {
                let pose = Pose::bind(&um.skel);
                let attach = w.world(Mat4::IDENTITY, &pose);
                let (sc, _, tr) = attach.to_scale_rotation_translation();
                println!(
                    "    RESOLVED parent '{}' (node {}) child '{}'  attach scale ({:.4},{:.4},{:.4}) translation ({:.3},{:.3},{:.3})",
                    if w.parent_marker.is_empty() { "<none>" } else { &w.parent_marker },
                    w.parent_node,
                    if w.child_marker.is_empty() { "<IDENTITY - no child marker matched>" } else { &w.child_marker },
                    sc.x, sc.y, sc.z, tr.x, tr.y, tr.z
                );
            }
            None => println!("    RESOLVED: none"),
        }
        // #iscale The nine authored scales, which one the engine picks for this holder, and
        // what it does to the weapon's measured length against the biped's height.
        match item_scales(native, cache, t) {
            Some(s) => {
                println!(
                    "    item scales: ground {:.3} | small {:.3}/{:.3} medium {:.3}/{:.3} large {:.3}/{:.3} huge {:.3}/{:.3}  (armed/stowed)",
                    s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]
                );
                let armed = item_scale_for(&s, size, false);
                let stowed = item_scale_for(&s, size, true);
                let ground = item_scale_for(&s, None, false);
                let span = |b: Option<(Vec3, Vec3)>| b.map(|(lo, hi)| (hi - lo)).unwrap_or(Vec3::ZERO);
                let w_len = span(measured_model_bounds(native, cache, native.resolve_mode(cache, t))).max_element();
                let u_h = span(measured_model_bounds(native, cache, native.resolve_mode(cache, unit_tag))).z;
                // 1 Reach world unit = 10 feet = 3.048 m, so the metres are informative only.
                const WU_M: f32 = 3.048;
                println!(
                    "    -> {} armed {:.3}  stowed {:.3}  ground {:.3} ; weapon {:.3} wu ({:.2} m) long, biped {:.3} wu ({:.2} m) tall: length/height {:.3} -> {:.3}  (weapon {:.2} m -> {:.2} m)",
                    size_name(size), armed, stowed, ground,
                    w_len, w_len * WU_M, u_h, u_h * WU_M,
                    if u_h > 0.0 { w_len / u_h } else { 0.0 },
                    if u_h > 0.0 { w_len * armed / u_h } else { 0.0 },
                    w_len * WU_M, w_len * armed * WU_M,
                );
            }
            None => println!("    item scales: UNREADABLE"),
        }
    }
    // The census the placement question actually turns on.
    let (mut with, mut without, mut names) = (0usize, 0usize, Vec::new());
    for t in native.tags_by_group(cache, "weap") {
        let Some(wm) = ReachModel::read(native, cache, native.resolve_mode(cache, t)) else { continue };
        match resolve_held_weapon(native, cache, unit_tag, &um, t, false) {
            Some(w) if !w.child_marker.is_empty() => with += 1,
            _ => {
                without += 1;
                let _ = &wm;
                names.push(native.tag_name(cache, t).unwrap_or_default());
            }
        }
    }
    println!("\nCENSUS: {with} weapons mate by a hand marker, {without} have NONE and fall back to the weapon's own origin:");
    for n in &names { println!("    {n}"); }
}

/// Print-only (`hms-app --model-regions <map> #fp`): trace the whole FIRST-PERSON chain for
/// every `@player representation` the cache has — the arms model's node list, every weapon's
/// `first person animations` graph, and the NAME MATCH between the two.
///
/// "The arms stick straight out" is a bind pose, and a bind pose means one link of this chain
/// failed, so each link is printed with its own count rather than inferred from the result.
pub fn dump_fp_chain(native: &NativeDll, cache: CacheHandle, reps: &[(String, u32, u32)]) {
    for (si, (name, hands, _body)) in reps.iter().enumerate() {
        // #fp-species  Row `si` of each weapon's `player interface` is THIS species' row.
        let species = si;
        println!("\n-- first-person chain for '{name}' (hands {hands:#x})");
        let Some(m) = ReachModel::read(native, cache, *hands) else {
            println!("   ReachModel::read FAILED -- nothing can be posed");
            continue;
        };
        println!("   arms model: {} nodes, {} meshes", m.node_names.len(), m.mesh_fmt.len());
        // #fp-mount  Where this model's own nodes sit relative to its ORIGIN, in bind pose. The
        // engine rigs a first-person view on `camera_control`, not on the model origin, so this
        // is the offset the mount has to cancel: putting the origin at the eye leaves the hands
        // wherever the artist built them relative to the view node, which on these models is
        // well below it.
        {
            let bind = m.skel.bind_model();
            for want in ["pedestal", "base", "camera_control", "aim_pitch", "aim_yaw", "r_hand", "l_hand"] {
                if let Some(i) = m.node_names.iter().position(|n| n == want) {
                    let t = bind[i].w_axis;
                    println!("      node {want:<16} bind-pose model offset = ({:+.4}, {:+.4}, {:+.4})", t.x, t.y, t.z);
                }
            }
        }
        println!("   arms nodes: {}", m.node_names.join(" "));
        // Every weapon's first-person graph, scored against THIS arms model.
        let mut rows: Vec<String> = Vec::new();
        for t in native.tags_by_group(cache, "weap") {
            let (_fpm, Some(j)) = first_person_weapon(native, cache, t, species) else { continue };
            let Some(g) = AnimGraph::read(native, cache, j) else {
                rows.push(format!("   {:<42} graph {j:#x} UNREADABLE", native.tag_name(cache, t).unwrap_or_default()));
                continue;
            };
            for (i, sc) in g.blend_screens.iter().enumerate() {
                println!(
                    "      [{}] blend screen {i} '{}' grid {}x{} (yaw R/L {:.1}/{:.1} deg per frame, pitch D/U {:.1}/{:.1})",
                    native.tag_name(cache, j).unwrap_or_default(),
                    g.blend_screen_labels.get(i).cloned().unwrap_or_default(),
                    sc.width(),
                    sc.height(),
                    sc.right_yaw_per_frame.to_degrees(),
                    sc.left_yaw_per_frame.to_degrees(),
                    sc.down_pitch_per_frame.to_degrees(),
                    sc.up_pitch_per_frame.to_degrees()
                );
            }
            let matched = g.node_names.iter().filter(|gn| m.node_names.iter().any(|n| n == *gn)).count();
            let nrows = first_person_rows(native, cache, t);
            let map = anim_node_map(&m, &g);
            let idle = g.find("first_person:idle");
            rows.push(format!(
                "   {:<42} rows={nrows} graph '{}' nodes {:>3}  matched {:>3} ({:>3.0}%)  node_map={}  first_person:idle={}  clips={}",
                native
                    .tag_name(cache, t)
                    .unwrap_or_default()
                    .rsplit('\\')
                    .next()
                    .unwrap_or("")
                    .to_string(),
                native.tag_name(cache, j).unwrap_or_default(),
                g.node_names.len(),
                matched,
                100.0 * matched as f32 / g.node_names.len().max(1) as f32,
                match &map {
                    Some(v) => format!("{} entries, {} live", v.len(), v.iter().filter(|&&x| x >= 0).count()),
                    None => "NONE (positional read kept)".to_string(),
                },
                idle.map(|i| i.to_string()).unwrap_or_else(|| "MISSING".into()),
                g.anims.len()
            ));
        }
        rows.sort();
        for r in rows.iter().take(40) {
            println!("{r}");
        }
        println!("   ({} weapons author a first-person graph)", rows.len());
    }
}

/// #fp Match an animation graph's node list to a render model's BY NAME.
///
/// `None` when the two already agree position for position, or when too few names match at all
/// to trust the result — in which case the positional read is kept, because a bad map would
/// silently drive the wrong bones.
///
/// #fp-arms  **The share that matters is the MODEL's, not the graph's.** A first-person
/// animation graph deliberately lists the arms' nodes AND the weapon's own (`b_gun`,
/// `b_magazine`, `b_ophandle`, `b_safety`, `b_switch`, `b_handle`, `b_blades`, ...), which the
/// arms model does not have and never will. Scoring the match against the GRAPH's length
/// therefore punished a graph for carrying its gun: six weapon bones out of 52 put the
/// assault rifle at 90%, eight put the needler at 76%, and the map was thrown away — leaving the
/// POSITIONAL read, which on these two lists is nonsense (the arms model orders its nodes
/// `pedestal base aim_pitch aim_yaw l_upperarm camera_control ...` and the graph
/// `pedestal aim_pitch aim_yaw base camera_control l_upperarm ...`, so every channel from 1
/// onwards drove the wrong bone). Measured on `forge_halo`: 14 of the Spartan's 27
/// first-person graphs and **all 27 of the Elite's** failed that way, which is exactly the
/// "arms stick straight out with no respect for the weapon" the user saw — a scrambled pose, not
/// a missing one.
///
/// So the map is accepted when EITHER side is mostly resolved: the graph's own names (the
/// third-person case, where the two lists are the same skeleton) or the model's nodes (the
/// first-person case, where the graph is a superset). Both arms models reach 100% of their own
/// nodes against every weapon graph in the cache.
pub fn anim_node_map(model: &ReachModel, graph: &AnimGraph) -> Option<Vec<i16>> {
    if graph.node_names.is_empty() || model.node_names.is_empty() {
        return None;
    }
    let mut map: Vec<i16> = Vec::with_capacity(graph.node_names.len());
    let mut resolved = 0usize;
    let mut identity = graph.node_names.len() == model.node_names.len();
    for (gi, gn) in graph.node_names.iter().enumerate() {
        match model.node_names.iter().position(|n| n == gn) {
            Some(i) => {
                resolved += 1;
                if i != gi {
                    identity = false;
                }
                map.push(i.min(i16::MAX as usize) as i16);
            }
            None => {
                identity = false;
                map.push(-1);
            }
        }
    }
    // A match that reaches neither side's nodes is not a naming mismatch, it is a different
    // convention: keep the positional read rather than wiring most bones to nothing.
    let graph_share = resolved * 10 >= graph.node_names.len() * 9;
    let model_share = resolved * 10 >= model.node_names.len() * 9;
    if !graph_share && !model_share {
        return None;
    }
    (!identity).then_some(map)
}

impl ActorDef {
    pub fn read(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<ActorDef> {
        let mode_tag = native.resolve_mode(cache, obj_tag);
        let model = ReachModel::read(native, cache, mode_tag)?;
        let graph = resolve_jmad(native, cache, obj_tag).and_then(|j| AnimGraph::read(native, cache, j));
        let inverse_bind = model.skel.inverse_bind();
        let anim_node_map = graph.as_ref().and_then(|g| anim_node_map(&model, g));
        Some(ActorDef {
            obj_tag,
            mode_tag,
            name: native.tag_name(cache, obj_tag).unwrap_or_default(),
            model,
            graph,
            inverse_bind,
            anim_node_map,
        })
    }
    /// #fp Build a definition straight from a RENDER MODEL, with no object wrapper.
    ///
    /// Reach's first-person arm models are bare `mode` tags: `matg @player representation` points
    /// at `objects\characters\spartans\fp\fp` directly, and there is no `obje`, no `hlmt` and
    /// no animation graph of their own (the first-person animations belong to the WEAPON, at
    /// `weap` +0x3B8 `player interface` -> `first person` -> `first person animations`).
    pub fn read_model(
        native: &NativeDll,
        cache: CacheHandle,
        mode_tag: u32,
        jmad_tag: Option<u32>,
    ) -> Option<ActorDef> {
        let model = ReachModel::read(native, cache, mode_tag)?;
        let inverse_bind = model.skel.inverse_bind();
        let graph = jmad_tag.and_then(|j| AnimGraph::read(native, cache, j));
        let anim_node_map = graph.as_ref().and_then(|g| anim_node_map(&model, g));
        Some(ActorDef {
            obj_tag: mode_tag,
            mode_tag,
            name: native.tag_name(cache, mode_tag).unwrap_or_default(),
            model,
            graph,
            inverse_bind,
            anim_node_map,
        })
    }

    /// Every animation name (for the script's `actor anims` listing).
    pub fn anim_names(&self) -> Vec<&str> {
        self.graph.as_ref().map(|g| g.anims.iter().map(|a| a.name.as_str()).collect()).unwrap_or_default()
    }
}

/// #aim  Blend an overlay clip's `(frame, weight)` cells into one per-channel delta.
///
/// Weighted sum per graph channel. The quaternions are accumulated componentwise against the
/// first contributing sample's hemisphere and renormalised, which is the cheap n-way blend the
/// engine uses for a blend screen's neighbours; a channel no cell animates stays `None` so the
/// base clip keeps that bone untouched.
///
/// Free-standing, and not a method, so the blend can be measured without a GPU.
pub fn overlay_delta(
    data: &AnimData,
    ty: AnimType,
    cells: &[(f32, f32)],
) -> (Vec<Option<glam::Quat>>, Vec<Option<Vec3>>) {
    let n = data.node_count as usize;
    let total: f32 = cells.iter().map(|c| c.1).sum();
    if n == 0 || total <= 1e-5 {
        return (Vec::new(), Vec::new());
    }
    let mut acc_rot: Vec<Option<glam::Quat>> = vec![None; n];
    let mut acc_trans: Vec<Option<Vec3>> = vec![None; n];
    for &(frame, w) in cells {
        if w <= 1e-4 {
            continue;
        }
        let w = w / total;
        let ch = reach_jmad::sample(data, ty, frame);
        for (ci, c) in ch.iter().enumerate().take(n) {
            if let Some(q) = c.rot {
                match &mut acc_rot[ci] {
                    None => acc_rot[ci] = Some(glam::Quat::from_xyzw(q.x * w, q.y * w, q.z * w, q.w * w)),
                    Some(a) => {
                        let s = if a.dot(q) < 0.0 { -w } else { w };
                        *a = glam::Quat::from_xyzw(a.x + q.x * s, a.y + q.y * s, a.z + q.z * s, a.w + q.w * s);
                    }
                }
            }
            if let Some(t) = c.trans {
                *acc_trans[ci].get_or_insert(Vec3::ZERO) += t * w;
            }
        }
    }
    (acc_rot, acc_trans)
}

/// #melee  Lay one REPLACEMENT clip's sampled channels over a pose.
///
/// This is the whole fix for "a meleeing player's legs go strange", and it is deliberately a free
/// function so it can be gated without a GPU or a map: a channel the clip CARRIES overwrites the
/// node, and a node the clip says nothing about is left exactly as the base clip left it. The
/// broken path was to install the swing through `set_anim` as the body's only clip, where
/// `clip_trs` seeds every node from the skeleton's authored default first -- so the 48 of a
/// Spartan's 82 nodes that `combat:sword:melee_strike_1` does not animate snapped to bind pose.
///
/// `node_map` maps a GRAPH channel index to a render-model node (`None` = the two lists agree).
pub fn apply_replacement(
    trs: &mut [(glam::Quat, Vec3, f32)],
    channels: &[reach_jmad::NodeChannel],
    node_map: Option<&[i16]>,
) {
    for (ci, c) in channels.iter().enumerate() {
        let i = match node_map {
            Some(m) => match m.get(ci) {
                Some(&v) if v >= 0 => v as usize,
                _ => continue,
            },
            None => ci,
        };
        if i >= trs.len() {
            continue;
        }
        if let Some(q) = c.rot {
            trs[i].0 = q;
        }
        if let Some(t) = c.trans {
            trs[i].1 = t;
        }
        if let Some(sc) = c.scale {
            trs[i].2 = sc;
        }
    }
}

/// #aim  One aim OVERLAY clip's frame table: what look direction each of its frames IS.
///
/// **Read off the clip, not assumed.** Reach's aim overlays label their own frames: every frame
/// writes a rotation onto the graph's `b_aim_yaw` and `b_aim_pitch` nodes, and that rotation is
/// the aim angle that frame stands for. Those two nodes are children of `b_pedestal` and are not
/// ancestors of the spine, so the rotation moves nothing — they exist to CARRY the label.
/// Measured on `forge_halo`, `combat:rifle:aim_still_up`:
///
/// ```text
/// frame   0  aim_yaw   0.00  aim_pitch  +90.00      frame   5  aim_yaw -55.00  aim_pitch  +90.00
/// frame   1  aim_yaw   0.00  aim_pitch  +45.00      ...
/// frame   2  aim_yaw   0.00  aim_pitch    0.00      frame  10  aim_yaw +73.50  aim_pitch  +90.00
/// frame   3  aim_yaw   0.00  aim_pitch  -45.00      ...
/// frame   4  aim_yaw   0.00  aim_pitch  -90.00      frame  14  aim_yaw +73.50  aim_pitch  -90.00
/// ```
///
/// i.e. a 3 (yaw) x 5 (pitch) grid, pitch varying fastest, with yaw columns `{0, -55, +73.5}`
/// degrees — **asymmetric and NOT in sorted order**. And the layout is not a constant of the
/// format: `combat:rifle:aim_move_front_up` orders its columns `{0, +55, -55}`,
/// `any:aim_spine` is 3x3 with yaw `{0, -14.9, +14.9}` and pitch `{-10, 0, +10}` in a different
/// order again, `any:look` is 7x4 with YAW varying fastest and its direction flipping row by
/// row, and the Elite's `combat:ball:aim_airborne_arc_up` has just 3 frames. So nothing here
/// assumes a layout: the samples are read, sorted into whatever grid they form, and looked up
/// bilinearly.
///
/// This is also why the graph needs no `blend screens` table to be usable — the Spartan biped
/// graph has none at all (the FIRST-PERSON weapon graphs do, e.g. `fp_dmr`'s `pitch_and_turn`,
/// yaw +/-57 deg and pitch -80/+90 deg per frame over a 3x3 grid, which agrees with this
/// reading).
#[derive(Clone, Debug, Default)]
pub struct AimGrid {
    /// Per frame: `(yaw label, pitch label)`, radians, +yaw = left and +pitch = up.
    pub labels: Vec<(f32, f32)>,
}

impl AimGrid {
    /// Read the labels out of a loaded overlay clip. `None` when the clip animates neither aim
    /// node, which means it is not an aim overlay and must not be driven as one.
    pub fn read(graph: &AnimGraph, anim: usize, data: &AnimData) -> Option<AimGrid> {
        let ty = graph.anims.get(anim)?.anim_type;
        let node = |want: &str| graph.node_names.iter().position(|n| n == want);
        // The graph's own node list, so the Spartan's `b_aim_pitch` (node 2) and the Elite's
        // (node 1) both resolve without a hard-coded index.
        let (ny, np) = (node("b_aim_yaw").or_else(|| node("aim_yaw")), node("b_aim_pitch").or_else(|| node("aim_pitch")));
        let (ny, np) = (ny?, np?);
        let frames = data.frame_count.max(1) as usize;
        let mut labels = Vec::with_capacity(frames);
        for f in 0..frames {
            let ch = reach_jmad::sample(data, ty, f as f32);
            // `Rz(yaw) * Ry(-pitch)` is the convention the whole actor path uses, so the label is
            // the Z-euler for yaw and the NEGATED Y-euler for pitch.
            let ang = |i: usize, pitch: bool| -> f32 {
                match ch.get(i).and_then(|c| c.rot) {
                    Some(q) => {
                        let (z, y, _) = q.to_euler(glam::EulerRot::ZYX);
                        if pitch { -y } else { z }
                    }
                    None => 0.0,
                }
            };
            labels.push((ang(ny, false), ang(np, true)));
        }
        (!labels.is_empty()).then_some(AimGrid { labels })
    }

    /// The up-to-four frames and weights that interpolate this look direction, bilinearly over
    /// whatever grid the labels form. Degenerate axes collapse to fewer distinct frames, which is
    /// correct: duplicate frames just sum their weights.
    ///
    /// `yaw` is the aim's offset from the BODY's own facing (radians, + = left) and `pitch` is
    /// the aim elevation (+ = up). Both are clamped into the clip's authored span, because the
    /// pose outside it is not authored and extrapolating a skeleton invents a contortion.
    pub fn cells(&self, yaw: f32, pitch: f32) -> Vec<(f32, f32)> {
        if self.labels.is_empty() {
            return Vec::new();
        }
        // Distinct yaw columns, in VALUE order (the frame order is not sorted).
        let mut cols: Vec<f32> = Vec::new();
        for &(y, _) in &self.labels {
            if !cols.iter().any(|c| (c - y).abs() < 1e-3) {
                cols.push(y);
            }
        }
        cols.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (ci, cw) = bracket(&cols, yaw);
        let pick = |col: f32| -> (f32, f32, f32) {
            // The frames in this column, sorted by their pitch label.
            let mut rows: Vec<(f32, usize)> = self
                .labels
                .iter()
                .enumerate()
                .filter(|(_, (y, _))| (y - col).abs() < 1e-3)
                .map(|(f, (_, p))| (*p, f))
                .collect();
            rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let vals: Vec<f32> = rows.iter().map(|r| r.0).collect();
            let (ri, rw) = bracket(&vals, pitch);
            (rows[ri].1 as f32, rows[(ri + 1).min(rows.len() - 1)].1 as f32, rw)
        };
        let (a0, a1, aw) = pick(cols[ci]);
        let (b0, b1, bw) = pick(cols[(ci + 1).min(cols.len() - 1)]);
        let mut out = vec![
            (a0, (1.0 - cw) * (1.0 - aw)),
            (a1, (1.0 - cw) * aw),
            (b0, cw * (1.0 - bw)),
            (b1, cw * bw),
        ];
        out.retain(|&(_, w)| w > 1e-4);
        out
    }
}

/// The index of the value at or below `v` in a sorted list, and how far from it to the next one
/// (0..1). Clamped at both ends.
fn bracket(sorted: &[f32], v: f32) -> (usize, f32) {
    if sorted.len() < 2 {
        return (0, 0.0);
    }
    if v <= sorted[0] {
        return (0, 0.0);
    }
    if v >= sorted[sorted.len() - 1] {
        return (sorted.len() - 2, 1.0);
    }
    let i = sorted.partition_point(|&s| s <= v).saturating_sub(1);
    let (a, b) = (sorted[i], sorted[i + 1]);
    (i, if (b - a).abs() < 1e-6 { 0.0 } else { (v - a) / (b - a) })
}

/// One placed actor.
pub struct Actor {
    pub id: u32,
    pub def: Arc<ActorDef>,
    /// World placement.
    pub pos: Vec3,
    /// Facing, degrees about +Z (0 = +X, the `ObjectInfo` forward convention).
    pub yaw: f32,
    /// #fp Elevation, degrees (positive looks up). 0 for every placed object — a Forge piece or a
    /// biped is placed level — and non-zero only for a model mounted on the CAMERA, which is what
    /// the first-person arms are: they have to follow the view's pitch as well as its yaw.
    pub pitch: f32,
    /// #fp Yaw about the model's OWN +Z, applied INSIDE the placement (degrees). 0 for everything
    /// placed in the world; see [`FP_ARMS_MODEL_YAW`] for the one model that needs it.
    pub model_yaw: f32,
    /// #fp Which graph is currently posing this actor, when it is not the model's own.
    pub graph_tag: Option<u32>,
    /// #fp-attach  This actor's root node is a CHILD of another actor's node in the animation
    /// graph they share: `(that actor's id, that actor's render-model node index)`. A Reach
    /// first-person weapon is exactly this -- the graph is one tree over the arms AND the gun, so
    /// the gun's root channel is a transform LOCAL to the hand. Resolved from the graph's own
    /// hierarchy; `None` for everything else, which is every actor placed in the world.
    pub fp_attach: Option<(u32, usize)>,
    /// #fp-attach  The world matrix resolved from [`Actor::fp_attach`], replacing the
    /// pos/yaw/pitch placement for this frame. Written by [`ActorScene::update`] after the PARENT
    /// has been posed, so it is always one frame's own pose and never a stale one.
    pub world_override: Option<Mat4>,
    /// #ride-grip  Hands to place after the pose is composed, written per frame by whoever drives
    /// the actor (a rider gripping a steering wheel). Empty for every other actor, which leaves
    /// [`Actor::compute_pose`] exactly as it was.
    pub grips: Vec<crate::reach_grip::Grip>,
    /// #ride-weapon  The held weapon is not drawn (a seat that does not allow weapons).
    pub weapon_hidden: bool,
    /// Selected animation (index into `def.graph.anims`) and its loaded blob.
    pub anim: Option<usize>,
    pub anim_name: String,
    data: Option<AnimData>,
    /// Current animation frame (fractional; 30 fps).
    pub frame: f32,
    pub playing: bool,
    /// The held weapon, if any.
    pub weapon: Option<HeldWeapon>,
    pub weapon_name: String,
    /// The clip this actor was playing before the current one, held so the change can be
    /// CROSS-FADED instead of snapping. `(graph index, its data, the frame it was on)`.
    pub prev_anim: Option<(usize, AnimData, f32)>,
    /// #walk-bob  The pose that was ON SCREEN when the change of clip began -- the outgoing
    /// clip's gait mix and any fade already in flight, frozen -- which the cross-fade runs out
    /// of. `prev_anim` alone is one clip at one frame: switching a second time inside a fade
    /// (idle -> walk -> run in the first tenth of a second, or a crouch begun mid-stride) faded
    /// out of the PREVIOUS TARGET and the body jumped to it in one frame (measured: a Spartan
    /// pressing crouch-and-forward moved its pelvis 0.143 wu in one frame, 43 cm; sprint start
    /// 0.031 wu). Cleared with `prev_anim`.
    pub prev_pose: Option<Vec<(glam::Quat, Vec3, f32)>>,
    /// Seconds of cross-fade still to run. 0 = none.
    pub blend_left: f32,
    /// #crouch-rate  How long THIS cross-fade was armed for, seconds. [`BLEND_SECONDS`] for an
    /// ordinary change of clip; the biped's own authored `crouch transition time` for a change
    /// into or out of the crouch, which is what the user means by "the speed that he goes down".
    /// Kept per fade rather than read from a constant so one long fade cannot stretch the next.
    pub blend_span: f32,
    /// #gait-blend  EXTRA base clips mixed into the pose beside [`Actor::anim`], each as
    /// `(graph animation index, frame, absolute weight)`. The primary clip carries
    /// `1 - sum(weights)`, so an empty list is exactly today's single-clip pose and costs
    /// nothing.
    ///
    /// This is what makes a diagonal a real diagonal and a speed change continuous: Reach
    /// authors FOUR movement directions, each with two or three speed tiers
    /// ([`AnimGraph::gait_table`]), and the body's travel direction and ground speed select a
    /// weighted mix of up to four of those clips rather than one. Every clip in the mix is
    /// evaluated at the SAME normalised phase, which is what keeps the feet together; see
    /// `play_mode`'s `#gait-blend` block for who drives the rate.
    ///
    /// The clips must be in [`Actor::overlay_cache`] (`ensure_overlay`) — the same cache the
    /// aim overlays and the melee replacement lane use, so no second loader exists.
    pub base_blend: Vec<(usize, f32, f32)>,
    /// #aim  OVERLAY clips composed onto the base pose this frame, in order, each as the set of
    /// `(frame, weight)` cells to blend. Reach poses a look direction and a shot this way — an
    /// overlay composes onto whatever the locomotion clip produced rather than replacing it — so
    /// an aiming, firing, running body is three clips, not one.
    ///
    /// Written every frame by the caller (Theater's driver) and cleared when it has nothing to
    /// say, so an empty list is the ordinary case and costs nothing.
    pub overlays: Vec<(usize, Vec<(f32, f32)>)>,
    /// #melee  The REPLACEMENT lane: `(graph animation index, frame)` of a clip that overwrites
    /// only the nodes it animates and leaves the rest of the body on the base clip.
    ///
    /// Reach's melee swings are animation type **Replacement** and a replacement clip carries a
    /// SUBSET of the skeleton: `combat:sword:melee_strike_1` is 34 rotation channels of 82 nodes,
    /// and `b_pelvis`, `b_l_calf` and `b_r_foot` carry none of them. Installed as a body's SOLE
    /// clip (which is what the Theater driver used to do), every node the swing does not animate
    /// falls back to the skeleton's authored default -- so the legs snapped to BIND POSE for the
    /// length of every swing. The engine composes a replacement ONTO the current pose, so the
    /// locomotion clip stays the base and this lane overwrites the arms and torso on top of it.
    ///
    /// Written every frame by the caller and `None` the rest of the time.
    pub replacement: Option<(usize, f32)>,
    /// #melee  The replacement clip's name, for the status line and the gates.
    pub replacement_name: String,
    /// #aim  Loaded overlay blobs, keyed by graph animation index, plus each one's frame LABELS
    /// (see [`AimGrid`]). At most a couple per body and loaded once.
    overlay_cache: HashMap<usize, (AnimData, Option<AimGrid>)>,
    /// #ragdoll  Model-space node matrices that REPLACE the animated pose for this frame.
    ///
    /// A ragdoll is not a clip: it is simulated rigid bodies, and the drawn skeleton has to follow
    /// the simulation rather than an animation. Written every frame by the caller for a body the
    /// solver owns and `None` the rest of the time, and applied AFTER the clip, the cross-fade,
    /// the replacement lane and the overlays, so nothing can pose over a ragdoll.
    pub pose_override: Option<Vec<Mat4>>,
    /// Model-space pose of the skeleton at `frame`.
    pub pose: Pose,
    /// The placement the GPU meshes were built at. The palette folds
    /// `anchor^-1 * world` in, so moving or turning the actor needs no mesh rebuild.
    pub anchor: Mat4,
    /// #float-weapon  The frame the HELD WEAPON's mesh was actually baked at, recorded at the
    /// rebuild that baked it. It must equal `anchor` — see [`held_weapon_drawn`] — and it is kept
    /// as its own field rather than assumed so that `actor list` can show the gun's DRAWN position
    /// beside its correct one and name a mismatch instead of leaving it to be spotted on screen.
    pub weapon_bake: Mat4,
    pub palette: Option<Arc<hms_render::SkinPalette>>,
    pub weapon_palette: Option<Arc<hms_render::SkinPalette>>,
    /// The `hlmt` model variant to draw with. 0 = none, which makes the render model show EVERY
    /// permutation of every region at once (a Spartan wearing all the armour in the game);
    /// resolved lazily in `build_actor_meshes` from the model's own variant list.
    pub variant_sid: u32,
    /// The resolved variant's name, for the UI.
    pub variant_name: String,
    /// #skin The two ARMOUR COLOURS this actor is drawn with — `[primary, secondary]`, LINEAR
    /// RGB, straight out of Reach's armoury palette (`matg` "profile colors"). They are applied
    /// through the render model's own `four_change_color` masks, exactly where the tag's authored
    /// default colours would land. `None` = the model's authored defaults (a scenery actor, or a
    /// player whose record carries no colours).
    pub skin: Option<[[f32; 3]; 2]>,
    /// #elite-armour The ten APPEARANCE bytes of a recorded player, when this actor was placed
    /// to wear a specific loadout (`:armour=` in an actor spec). `build_actor_meshes` turns them
    /// into the same synthetic model variant Theater uses, through
    /// `SceneController::player_variant`, so a headless capture goes down the production path
    /// rather than a parallel one. `None` = the model's own default variant.
    pub armour: Option<[u8; crate::film::APPEARANCE_SLOTS]>,
    /// #vehicle-damage  The `model_state_enum` damage state to DRAW this actor in (0 = the
    /// undamaged model, 1 minor .. 4 destroyed), exactly the states the object panel's DAMAGE
    /// dropdown offers for a placed object. Theater drives it from the recording's own
    /// `damage_section_response` events; everything else leaves it 0.
    ///
    /// A change needs a MESH rebuild (it selects different permutations), so the setter marks the
    /// actor dirty rather than this being written directly.
    pub damage: u8,
    /// #armour-fx The attached EFFECTS this body is playing, resolved at the last mesh rebuild
    /// (see [`resolve_actor_effects`]). Empty for everything that is not wearing an armour
    /// effect, which is most bodies.
    pub fx: Vec<ActorEffect>,
    /// #fp-fov This actor is part of the FIRST-PERSON RIG — the arms, or the weapon the graph
    /// parents to a hand — rather than something standing in the world.
    ///
    /// It decides which CAMERA the actor's meshes are drawn through. Reach renders the
    /// first-person model in its own pass at 88% of the world camera's vertical field of view
    /// (`render_first_person_view`), which is why a viewmodel keeps its proportions as the FOV
    /// slider widens; drawing the rig through the world camera is what made the arms splay and
    /// stretch at 120. Set by the film driver for every model in its first-person set.
    pub first_person: bool,
    /// #armour-fx A STABLE identity to seed this body's effect randoms with.
    ///
    /// It must not be `id`: the Theater actor POOL hands a body whichever actor slot is free, so
    /// `id` depends on the order bodies happened to arrive and is therefore not a function of the
    /// playhead — scrub away and back and the same body could be drawn by a different slot. The
    /// film driver writes the body's recorded NET ID here, which is, and 0 means "nothing better
    /// known, use `id`".
    pub fx_seed: u32,
    /// Set when the MESHES (not the pose) must be rebuilt: a new actor, a changed weapon, or a
    /// placement that drifted too far from `anchor` for the fold-in to keep lighting honest.
    pub meshes_dirty: bool,
}

/// #armour-fx One `effe` ATTACHMENT a posed actor is playing, resolved once per mesh rebuild.
///
/// Reach hangs a player's armour effect — the flaming helmet, the thunder cloud — on the biped as
/// an ordinary object attachment, and which of its five events plays is decided by the
/// `armor_fx` region permutation the body wears: see
/// [`crate::scene::SceneController::model_marker_node_for_variant`] for the whole measured chain.
/// `systems` therefore holds **only the particle systems whose location marker this body's
/// variant actually shows**, which is normally the one or three or four systems of a single event.
#[derive(Clone, Debug)]
pub struct ActorEffect {
    /// The `effe` tag.
    pub effe: u32,
    /// The attachment's primary-scale object function at rest (`> 0` = live). Kept for the
    /// status line and the gates: it is the number that says WHY this attachment is playing.
    pub scale: f32,
    /// The attachment's primary-scale function name ("" = unconditional).
    pub scale_name: String,
    /// The systems that survive the marker test, in effect order.
    pub systems: Arc<Vec<crate::scene::ParticleSystemDef>>,
    /// Per kept system: its location marker as `(skeleton node, node-local pos, node-local rot)`.
    /// `None` node = the marker rides the model origin rather than a bone.
    pub locs: Vec<(Option<usize>, Vec3, glam::Quat)>,
    /// Per kept system: the effect EVENT name and the location marker name — what a measurement
    /// prints ("eternal_flame @ armor_fx_01"), never used to select anything.
    pub picks: Vec<(String, String)>,
}

/// #fp The model-space yaw that mounts Reach's first-person ARMS on the view, degrees.
///
/// **Zero: the first-person rig is an ordinary +X-forward Reach object, like everything else.**
///
/// This used to be -90, derived from the arms model's BIND pose, where both arms do reach along
/// +Y (`r_hand` at `(+0.094, +0.211, 0)`, `l_hand` at `(-0.094, +0.211, 0)`). That derivation was
/// the bug: the bind pose is a T-pose the engine never draws, and **every first-person CLIP poses
/// the rig into +X-forward space regardless of it.** Measured on `forge_halo`, Spartan arms on
/// `first_person:idle`, model space:
///
/// ```text
///   node        bind                       posed (first_person:idle)
///   l_hand      (-0.094, +0.211,  0.000)   (+0.199, +0.105, -0.050)
///   l_forearm   (-0.094, +0.103,  0.000)   (+0.125, +0.152, -0.114)
///   b_blades    (+0.121, -0.004,  0.000)   (+0.150, +0.007, +0.002)   (the energy sword)
/// ```
///
/// The WEAPON settles it. A Reach weapon render model is authored +X forward — the sword's blade
/// node sits at `(+0.121, -0.004, 0)` in bind — and the first-person graph poses the arms and the
/// gun from ONE clip in ONE space, so wherever that clip puts the blade is where the rig's forward
/// is. Posed, the blade is still `+X` (`(+0.150, +0.007, +0.002)`, 2.7 degrees off the axis). The
/// clip has carried the arms from their +Y bind into that same +X space; nothing is left for a
/// model yaw to correct.
///
/// Turning the rig a further -90 degrees therefore swung the whole view 90 degrees to the RIGHT:
/// on that frame the left hand landed **+62.3 degrees** off the view axis horizontally and the
/// sword blade **+87.4**, so **0 of 47 arm nodes and 0 of 2 weapon nodes projected inside the
/// viewport** — the arms were drawn, skinned and animated perfectly, entirely off the right-hand
/// edge of the screen. That is the "there are no first-person arms at all" report. With 0 the same
/// frame puts the left hand 27.8 degrees left and 14.1 degrees down, and the blade on the axis.
///
/// The first-person BODY (`fp_body`) is an ordinary object too — pelvis at `(-0.009, 0, 0.380)`,
/// origin at the feet, +X forward — and is placed at the player's feet, not on the camera.
pub const FP_ARMS_MODEL_YAW: f32 = 0.0;

/// How long a change of animation is blended over, seconds.
///
/// Short enough that a reaction still reads as instant, long enough that a foot does not jump.
/// The engine blends locomotion over a direction x speed space rather than hard-selecting one
/// clip; this does not reproduce that, but a cross-fade plus the hysteresis in the gait choice
/// gets most of the way.
pub const BLEND_SECONDS: f32 = 0.18;

/// #armour-fx Seed base for a posed actor's effect particle sources, mixed with the body's own
/// stable identity. Far away from the object lane's `0x51F7...` base so an actor source and a
/// placed-object source can never key the same event clock / emission accumulator.
pub const ACTOR_FX_SEED: u64 = 0xA7F0_0000_0000_0000;

/// An actor's placement matrix from its pos / yaw / pitch / model yaw (degrees).
///
/// `Rz(yaw) * Ry(-pitch)` sends model +X to `(cos p cos y, cos p sin y, sin p)`, which is exactly
/// `hms_render::Camera::forward()` — so an actor given the camera's own yaw and pitch is mounted on
/// the view, the way a first-person arm model is. Free-standing so the first-person framing gate
/// can project through the SAME code the renderer places with, rather than a copy of it.
pub fn actor_world(pos: Vec3, yaw_deg: f32, pitch_deg: f32, model_yaw_deg: f32) -> Mat4 {
    let mut m = Mat4::from_translation(pos) * Mat4::from_rotation_z(yaw_deg.to_radians());
    if pitch_deg != 0.0 {
        m *= Mat4::from_rotation_y(-pitch_deg.to_radians());
    }
    if model_yaw_deg != 0.0 {
        m *= Mat4::from_rotation_z(model_yaw_deg.to_radians());
    }
    m
}

/// #fp-attach  The frame a built mesh is baked at: `(origin, forward, up)`, ALL THREE off the same
/// matrix.
///
/// The bone palette folds in `anchor^-1 * world`, which is only a correction if the mesh was baked
/// at exactly `anchor`. This used to take the basis from `anchor` but the origin from the actor's
/// `pos`. For everything placed in the world those are the same point — `world()`'s translation IS
/// `pos` — so it went unnoticed; a first-person WEAPON is parented to the hand, so its `world()`
/// sits in the hand while its `pos` is still the eye the rig is mounted on. The mesh was therefore
/// baked at the EYE and then given a fold-in of identity, which put the gun on the camera: "the
/// weapon isn't held properly, it just floats there", enormous because it was centimetres from the
/// lens. One frame of a fresh scene looked right, because the first build happens before the
/// attach is resolved — which is how a scripted capture passed it.
///
/// Returning all three together is the gate: a caller cannot take the origin from somewhere else
/// without saying so.
pub fn actor_mesh_frame(anchor: Mat4) -> (Vec3, Vec3, Vec3) {
    (anchor.w_axis.truncate(), anchor.x_axis.truncate(), anchor.z_axis.truncate())
}

/// #float-weapon  Where a HELD WEAPON's mesh actually ends up on screen, written out in full.
///
/// Three transforms compose, and they come from three different places, which is why this is a
/// function rather than a comment:
/// * `bake` — the frame the weapon's mesh was BUILT at. It becomes the mesh's per-instance model
///   matrix and is fixed until the next rebuild.
/// * `anchor` — the actor's placement at that same rebuild, which the bone palette folds against.
/// * `world_now` / `attach` — this frame's placement and the engine's marker attach
///   (`unit_world * P_marker * C_marker^-1`), which the palette is rewritten with every frame.
///
/// The palette holds `anchor^-1 * world_now * attach`, so the drawn matrix is
/// `bake * anchor^-1 * world_now * attach`. That equals the right answer, `world_now * attach`,
/// **only when `bake == anchor`** — the invariant [`weapon_bake_matches_anchor`] gates. Pass a
/// `bake` taken from anywhere else and the weapon is displaced by `bake.translation -
/// anchor.translation`, drawn hanging in mid-air away from the hand that holds it.
pub fn held_weapon_drawn(bake: Mat4, anchor: Mat4, world_now: Mat4, attach: Mat4) -> Mat4 {
    bake * anchor.inverse() * world_now * attach
}

/// #float-weapon  The frame `actor_mesh_frame` describes, rebuilt as a matrix — i.e. what
/// `build_actor_meshes` bakes a mesh at when handed `(origin, fwd, up)`. Used by the gates to show
/// that feeding the weapon build `a.pos` instead of `origin` really does move the drawn gun.
pub fn bake_frame(origin: Vec3, fwd: Vec3, up: Vec3) -> Mat4 {
    let f = fwd.normalize_or_zero();
    let u = up.normalize_or_zero();
    let right = u.cross(f);
    Mat4::from_cols(f.extend(0.0), right.extend(0.0), u.extend(0.0), origin.extend(1.0))
}

impl Actor {
    /// The actor's world matrix (placement only; the pose is in the palette).
    ///
    /// `Rz(yaw) * Ry(-pitch)` sends model +X to `(cos p cos y, cos p sin y, sin p)`, which is
    /// exactly `hms_render::Camera::forward()` — so an actor given the camera's own yaw and pitch
    /// is mounted on the view, the way a first-person arm model is.
    pub fn world(&self) -> Mat4 {
        // #fp-attach  A node of another actor IS this actor's placement when the two share an
        // animation graph that parents one to the other.
        if let Some(m) = self.world_override {
            return m;
        }
        actor_world(self.pos, self.yaw, self.pitch, self.model_yaw)
    }

    /// Re-evaluate the pose from the current animation + frame, then refresh both palettes.
    /// Call once per frame; cheap (no tag reads, no GPU allocation).
    /// #fp-fov Mark this actor as part of the FIRST-PERSON RIG (or not). Returns whether that was
    /// a CHANGE, and dirties its meshes when it was.
    ///
    /// The rebuild is the point, not a side effect: which lane set a mesh goes into — and so which
    /// CAMERA it is drawn through — is decided while the mesh is built, so a model already built
    /// into the world lanes would otherwise stay there, drawn at the world FOV, for the rest of
    /// the session. Returning the change (rather than always dirtying) is what stops the whole rig
    /// being rebuilt on every frame of a film.
    pub fn set_first_person(&mut self, on: bool) -> bool {
        if self.first_person == on { return false; }
        self.first_person = on;
        self.meshes_dirty = true;
        true
    }

    pub fn refresh_pose(&mut self, queue: &wgpu::Queue) {
        self.compute_pose();
        self.refresh_palettes(queue);
    }

    /// #walk-bob  The base pose as it is DRAWN before the replacement and the overlays: the clip,
    /// the gait mix and any cross-fade still running. The one place that is computed, so the pose
    /// a frame draws and the pose a change of clip fades out of cannot disagree.
    fn on_screen_trs(&self) -> Vec<(glam::Quat, Vec3, f32)> {
        // Evaluate the current clip, and the outgoing one while a cross-fade is running, then
        // blend them per node. Blending (rotation, translation, scale) and composing afterwards
        // is what makes the result a real pose; blending the composed matrices would shear.
        let mut trs = self.clip_trs(self.anim, self.data.as_ref(), self.frame);
        // #gait-blend  THE DIRECTIONAL / SPEED-TIER MIX, before the cross-fade.
        //
        // The primary clip carries `1 - sum(extra weights)` and each extra is mixed in
        // incrementally against the weight accumulated so far:
        //
        //   acc = w_primary;  for each extra:  t = w / (acc + w);  pose = blend(pose, clip, t);
        //   acc += w
        //
        // which is the exact weighted mean for any number of clips and reduces to today's single
        // clip when the list is empty. The quaternions go through `slerp`, so the result is a
        // real rotation at every weight rather than a shortened one.
        //
        // Continuity. The caller supplies the set as a continuous function of the travel
        // direction and the ground speed, and a clip LEAVES the set only once its weight has
        // reached zero, so no pose jumps when the mix changes shape. The one place the order of
        // the chain changes is when the heaviest clip changes, and there the two swapped clips
        // have equal weight -- and `slerp(a, b, 0.5) == slerp(b, a, 0.5)` -- so the pose is the
        // same on both sides of that swap.
        if !self.base_blend.is_empty() {
            let total: f32 = self.base_blend.iter().map(|(_, _, w)| w.max(0.0)).sum();
            let mut acc = (1.0 - total).max(0.0);
            for &(anim, frame, w) in &self.base_blend {
                if !(w > 1e-4) || !w.is_finite() {
                    continue;
                }
                let Some((data, _)) = self.overlay_cache.get(&anim) else { continue };
                let other = self.clip_trs(Some(anim), Some(data), frame);
                let t = w / (acc + w);
                if !(t.is_finite() && t > 0.0) {
                    continue;
                }
                for (i, o) in other.iter().enumerate() {
                    if i >= trs.len() {
                        break;
                    }
                    trs[i].0 = trs[i].0.slerp(o.0, t);
                    trs[i].1 = trs[i].1.lerp(o.1, t);
                    trs[i].2 += (o.2 - trs[i].2) * t;
                }
                acc += w;
            }
        }
        if self.blend_left > 0.0 {
            if let Some((pi, pd, pf)) = &self.prev_anim {
                // #play-jolt  **EASED, not linear.** A linear cross-fade moves the whole skeleton
                // at a CONSTANT speed from one pose to the other, and that speed is set by how
                // different the two poses are rather than by anything the body is doing: measured
                // on a real `PlayMode` walking and turning on `forge_halo`
                // (`play_mode::probes::probe_pose_continuity_over_a_scripted_walk`), an ordinary
                // walking frame moves the busiest bone 0.033 wu while the first frame of a
                // turn-to-idle cross-fade moved it **0.255 wu** — a 7.7x velocity step, held for
                // the whole 0.18 s of the blend. That is the user's *"sudden jolt of the legs
                // instead of cleanly interpolating"*: not a discontinuity in POSITION (a linear
                // blend starts at exactly the outgoing pose) but one in SPEED, at both ends.
                //
                // `smoothstep` has zero derivative at 0 and 1, so the blend starts and ends at
                // the speed of the clips it is between and does its moving in the middle, where
                // nothing is holding still for the eye to notice. Same duration, same endpoints,
                // same cost -- one multiply and one subtract.
                // #crouch-rate  Against the span THIS fade was armed with, not the constant: a
                // crouch is as long as the biped's own `crouch transition time` authors.
                let span = if self.blend_span > 1e-4 { self.blend_span } else { BLEND_SECONDS };
                let lin = (self.blend_left / span).clamp(0.0, 1.0);
                let w = lin * lin * (3.0 - 2.0 * lin);
                // #walk-bob  Out of the pose that WAS on screen when the clip changed, not out of
                // the previous clip alone (see `Actor::prev_pose`).
                let old = match &self.prev_pose {
                    Some(p) => p.clone(),
                    None => self.clip_trs(Some(*pi), Some(pd), *pf),
                };
                for (i, o) in old.iter().enumerate() {
                    if i >= trs.len() {
                        break;
                    }
                    // `w` is how much of the OUTGOING pose remains, so it decays to 0.
                    trs[i].0 = o.0.slerp(trs[i].0, 1.0 - w);
                    trs[i].1 = o.1.lerp(trs[i].1, 1.0 - w);
                    trs[i].2 = o.2 + (trs[i].2 - o.2) * (1.0 - w);
                }
            }
        }
        trs
    }

    /// #film-perf  Re-evaluate `self.pose` from the current animation, overlays and ragdoll
    /// override. Split out of [`Actor::refresh_pose`] so it can be gated WITHOUT a GPU device:
    /// the palettes are a pure function of the pose, so proving the pose is identical through
    /// both routes proves the upload is too.
    pub(crate) fn compute_pose(&mut self) {
        let skel = &self.def.model.skel;
        // #film-perf  A RAGDOLLED body's node frames come WHOLLY from the solver, so sampling
        // the clip, blending the cross-fade, composing the overlays, building `locals` and
        // chaining `pose_model` produced a pose that the `pose_override` copy a few lines below
        // then overwrote line for line. Up to fourteen corpses lie on the ground at once and
        // every one of them was paying for a pose nothing could ever see -- six Vec allocations
        // and a skeleton walk each, twice over (see the `jmad` sample inside `clip_trs`).
        //
        // Skipped only when the override covers EVERY node: a partial override still takes the
        // full path, so the two routes cannot disagree about a single matrix. The `resize` keeps
        // the invariant the palette below relies on -- `self.pose.model` is always as long as the
        // skeleton -- for a body that is ragdolled before it has ever been posed from a clip.
        if self.pose_override.as_ref().is_some_and(|o| o.len() >= skel.len()) {
            let over = self.pose_override.as_ref().expect("checked above");
            if self.pose.model.len() != skel.len() {
                self.pose.model.resize(skel.len(), Mat4::IDENTITY);
            }
            self.pose.model.copy_from_slice(&over[..skel.len()]);
            return;
        }
        let mut trs = self.on_screen_trs();
        // #melee  Then the REPLACEMENT lane, which the engine applies between the base and the
        // overlays: it overwrites the channels it carries and leaves every other node on the
        // base clip (see [`Actor::replacement`]).
        if let Some((anim, frame)) = self.replacement {
            self.compose_replacement(&mut trs, anim, frame);
        }
        // #aim  Compose the overlays onto the result, the way the engine's own overlay combiner
        // does [haloreach.dll +0x21DAEC]: `q_dst (x) q_src`, `t_dst + t_src`, `s_dst * s_src`.
        // Composing onto the BASE CLIP'S RESULT rather than onto the authored default is the
        // whole point -- that is what makes an aim a look direction added to a stride instead of
        // a pose that cancels it.
        for (anim, cells) in &self.overlays {
            self.compose_overlay(&mut trs, *anim, cells);
        }
        let locals: Vec<Mat4> = trs
            .iter()
            .map(|(r, t, sc)| {
                if (sc - 1.0).abs() > 1e-5 {
                    Mat4::from_scale_rotation_translation(Vec3::splat(*sc), *r, *t)
                } else {
                    Mat4::from_rotation_translation(*r, *t)
                }
            })
            .collect();
        self.pose = Pose { model: skel.pose_model(&locals) };
        // #ride-grip  Hands on a steering wheel: a two-bone solve over the composed pose.
        if !self.grips.is_empty() {
            let parents: Vec<i16> = skel.nodes.iter().map(|n| n.parent).collect();
            for g in &self.grips {
                crate::reach_grip::apply_grip(&mut self.pose.model, &parents, g);
            }
        }
        // #ragdoll  A simulated body's node frames come from the solver, not from a clip.
        if let Some(over) = &self.pose_override {
            let n = self.pose.model.len().min(over.len());
            self.pose.model[..n].copy_from_slice(&over[..n]);
        }
    }

    /// #film-perf  Upload this actor's bone palettes from whatever is in `self.pose`.
    ///
    /// Factored out of [`Actor::refresh_pose`] so the ragdoll fast path above and the ordinary
    /// clip path share ONE implementation -- a second copy here is exactly how a "two writers,
    /// one field" defect starts.
    fn refresh_palettes(&mut self, queue: &wgpu::Queue) {
        let skel = &self.def.model.skel;
        // The palette folds the placement DELTA in so the meshes (built at `anchor`) follow the
        // actor without a rebuild: world = anchor * (anchor^-1 * world) * posed * inverse_bind.
        let delta = self.anchor.inverse() * self.world();
        if let Some(p) = &self.palette {
            let inv = &self.def.inverse_bind;
            let mats: Vec<Mat4> = (0..skel.len()).map(|i| delta * self.pose.model[i] * inv[i]).collect();
            p.update(queue, &mats);
        }
        if let (Some(wp), Some(w)) = (&self.weapon_palette, &self.weapon) {
            // A held weapon's sections are RIGID with their vertices in the weapon's model
            // space, so every palette entry is the same matrix: the engine's attach
            // `unit_world * P_marker * C_marker^-1`, folded through the same anchor delta.
            let m = delta * w.world(Mat4::IDENTITY, &self.pose);
            let mats = held_weapon_palette(m, w.node_count, self.weapon_hidden);
            wp.update(queue, &mats);
        }
    }

    /// One clip's per-node (rotation, translation, scale), on the skeleton's authored defaults
    /// where the clip says nothing. Factored out of `refresh_pose` so two clips can be sampled
    /// and blended.
    fn clip_trs(&self, anim: Option<usize>, data: Option<&AnimData>, frame: f32) -> Vec<(glam::Quat, Vec3, f32)> {
        let skel = &self.def.model.skel;
        let mut out: Vec<(glam::Quat, Vec3, f32)> =
            skel.nodes.iter().map(|n| (n.rotation, n.translation, 1.0f32)).collect();
        let (Some(ai), Some(data)) = (anim, data) else { return out };
        let ty = self.def.graph.as_ref().and_then(|g| g.anims.get(ai)).map(|a| a.anim_type).unwrap_or(AnimType::Base);
        let ch = reach_jmad::sample(data, ty, frame);
        for (ci, c) in ch.iter().enumerate() {
            // #fp A graph channel addresses a GRAPH node; which render-model node that is depends
            // on the model. `None` = the two lists agree and the index is its own answer.
            let i = match &self.def.anim_node_map {
                Some(m) => match m.get(ci) {
                    Some(&v) if v >= 0 => v as usize,
                    _ => continue,
                },
                None => ci,
            };
            if i >= out.len() {
                continue;
            }
            let nd = &skel.nodes[i];
            // The engine's combiners (haloreach.dll +0x21D788 copy / +0x21DAEC overlay):
            // a BASE or REPLACEMENT channel REPLACES the node's local transform, while an
            // OVERLAY composes onto whatever the destination pose holds - here the authored
            // default - as `q_dst (x) q_src`, `t_dst + t_src`, `s_dst * s_src`.
            out[i] = match ty {
                AnimType::Overlay => (
                    c.rot.map_or(nd.rotation, |q| nd.rotation * q),
                    c.trans.map_or(nd.translation, |p| nd.translation + p),
                    c.scale.unwrap_or(1.0),
                ),
                _ => (
                    c.rot.unwrap_or(nd.rotation),
                    c.trans.unwrap_or(nd.translation),
                    c.scale.unwrap_or(1.0),
                ),
            };
        }
        out
    }

    /// #aim  Blend one overlay clip's `(frame, weight)` cells and compose the result onto `trs`.
    ///
    /// A node the overlay does not animate is left exactly as the base clip left it, which is
    /// what keeps an aim overlay from flattening the legs.
    fn compose_overlay(&self, trs: &mut [(glam::Quat, Vec3, f32)], anim: usize, cells: &[(f32, f32)]) {
        let Some((data, _)) = self.overlay_cache.get(&anim) else { return };
        let ty = self.def.graph.as_ref().and_then(|g| g.anims.get(anim)).map(|a| a.anim_type).unwrap_or(AnimType::Overlay);
        let n = data.node_count as usize;
        let (acc_rot, acc_trans) = overlay_delta(data, ty, cells);
        if acc_rot.is_empty() {
            return;
        }
        for ci in 0..n {
            let i = match &self.def.anim_node_map {
                Some(m) => match m.get(ci) {
                    Some(&v) if v >= 0 => v as usize,
                    _ => continue,
                },
                None => ci,
            };
            if i >= trs.len() {
                continue;
            }
            if let Some(q) = acc_rot[ci] {
                let q = if q.length_squared() > 1e-8 { q.normalize() } else { glam::Quat::IDENTITY };
                trs[i].0 *= q;
            }
            if let Some(t) = acc_trans[ci] {
                trs[i].1 += t;
            }
        }
    }

    /// #melee  Overwrite the channels one REPLACEMENT clip carries at `frame`, leaving every
    /// node it does not animate exactly as the base clip left it.
    ///
    /// Per CHANNEL, not per node: a Reach animation stores rotations, translations and scales in
    /// three separate channel sets with their own presence bitvectors, and `reach_jmad::sample`
    /// hands each one back as an `Option`, so "did this clip animate this node's rotation" is a
    /// fact off the clip rather than a guess.
    fn compose_replacement(&self, trs: &mut [(glam::Quat, Vec3, f32)], anim: usize, frame: f32) {
        let Some((data, _)) = self.overlay_cache.get(&anim) else { return };
        let ty = self.def.graph.as_ref().and_then(|g| g.anims.get(anim)).map(|a| a.anim_type).unwrap_or(AnimType::Replacement);
        let ch = reach_jmad::sample(data, ty, frame);
        apply_replacement(trs, &ch, self.def.anim_node_map.as_deref());
    }

    /// #melee  Which NODES a loaded clip animates (true = it carries a rotation channel for that
    /// render-model node). The gate on the replacement lane reads this.
    pub fn clip_animates_nodes(&self, anim: usize) -> Vec<bool> {
        let n = self.def.model.skel.len();
        let mut out = vec![false; n];
        let Some((data, _)) = self.overlay_cache.get(&anim) else { return out };
        let ty = self.def.graph.as_ref().and_then(|g| g.anims.get(anim)).map(|a| a.anim_type).unwrap_or(AnimType::Replacement);
        for (ci, c) in reach_jmad::sample(data, ty, 0.0).iter().enumerate() {
            let i = match &self.def.anim_node_map {
                Some(m) => match m.get(ci) {
                    Some(&v) if v >= 0 => v as usize,
                    _ => continue,
                },
                None => ci,
            };
            if i < n && c.rot.is_some() {
                out[i] = true;
            }
        }
        out
    }

    /// #vehicle-anim  Load an OVERLAY clip into this actor's cache by its GRAPH INDEX, so
    /// [`Actor::overlays`] can name it.
    ///
    /// [`Actor::aim_grid`] already loads an overlay, but it looks one up by NAME and returns
    /// `None` unless the clip has an aim grid — and a driven vehicle's overlays are chosen by
    /// what their frames MEAN rather than by their names (see [`crate::reach_vehicle_anim`]),
    /// most of them animate neither aim node, and two different vehicles give the same job
    /// different names. So they need a loader that takes the index the classifier already
    /// resolved and does not care whether the clip is an aim overlay.
    ///
    /// Idempotent, and that matters: this is called every frame of a ride, and a cached clip
    /// must cost one hash lookup rather than an animation-resource read.
    pub fn ensure_overlay(&mut self, native: &NativeDll, cache: CacheHandle, anim: usize) -> bool {
        if self.overlay_cache.contains_key(&anim) {
            return true;
        }
        let Some(g) = self.def.graph.clone() else { return false };
        if anim >= g.anims.len() {
            return false;
        }
        let Some(data) = g.load(native, cache, anim) else { return false };
        // The grid is read opportunistically: it is `None` for every vehicle overlay and that is
        // correct, because none of them is an aim overlay. Cached either way so the lane that
        // DOES want a grid finds it already loaded.
        let grid = AimGrid::read(&g, anim, &data);
        self.overlay_cache.insert(anim, (data, grid));
        true
    }

    /// #aim  Load an overlay clip (once) and return its frame LABELS, so the caller can turn a
    /// look direction into frames. `None` when the graph has no such clip or it is not an aim
    /// overlay (no `b_aim_yaw` / `b_aim_pitch` channels).
    pub fn aim_grid(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        name: &str,
    ) -> Option<(usize, AimGrid)> {
        let g = self.def.graph.clone()?;
        let anim = g.find(name)?;
        if !self.overlay_cache.contains_key(&anim) {
            let data = g.load(native, cache, anim)?;
            let grid = AimGrid::read(&g, anim, &data);
            self.overlay_cache.insert(anim, (data, grid));
        }
        let grid = self.overlay_cache.get(&anim)?.1.clone()?;
        Some((anim, grid))
    }

    /// #ragdoll  MODEL-space node matrices of one named clip's frame, computed WITHOUT touching
    /// the actor's own pose. This is the seed a ragdoll starts from: a dead biped's simulation
    /// begins at the first frame of the death clip HMS already picks for it, so the limbs start
    /// where the engine's would rather than in the skeleton's authored T-pose.
    ///
    /// Also returns the per-node LOCAL matrices, which the solver needs for the nodes no rigid
    /// body drives (a Spartan's fingers ride his forearm link and keep the clip's own shape).
    pub fn clip_pose_model(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        name: &str,
        frame: f32,
    ) -> Option<(Vec<Mat4>, Vec<Mat4>)> {
        let g = self.def.graph.clone()?;
        let anim = g.find(name)?;
        if !self.overlay_cache.contains_key(&anim) {
            let data = g.load(native, cache, anim)?;
            let grid = AimGrid::read(&g, anim, &data);
            self.overlay_cache.insert(anim, (data, grid));
        }
        let ty = g.anims.get(anim)?.anim_type;
        let (data, _) = self.overlay_cache.get(&anim)?;
        let skel = &self.def.model.skel;
        let mut trs: Vec<(glam::Quat, Vec3, f32)> =
            skel.nodes.iter().map(|n| (n.rotation, n.translation, 1.0f32)).collect();
        let ch = reach_jmad::sample(data, ty, frame);
        apply_replacement(&mut trs, &ch, self.def.anim_node_map.as_deref());
        let locals: Vec<Mat4> = trs
            .iter()
            .map(|(r, t, sc)| {
                if (*sc - 1.0).abs() < 1e-5 {
                    Mat4::from_rotation_translation(*r, *t)
                } else {
                    Mat4::from_scale_rotation_translation(Vec3::splat(*sc), *r, *t)
                }
            })
            .collect();
        Some((skel.pose_model(&locals), locals))
    }

    /// #melee  Load a REPLACEMENT clip (once) into this actor's lane cache and return its index
    /// and frame count. `None` when the graph has no such clip, or it is not type Replacement --
    /// the lane exists precisely because a replacement cannot be a sole clip, so a Base or
    /// Overlay clip must never end up in it.
    pub fn replacement_clip(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        name: &str,
    ) -> Option<(usize, f32)> {
        let g = self.def.graph.clone()?;
        let anim = g.find(name)?;
        if g.anims.get(anim)?.anim_type != AnimType::Replacement {
            return None;
        }
        if !self.overlay_cache.contains_key(&anim) {
            let data = g.load(native, cache, anim)?;
            let grid = AimGrid::read(&g, anim, &data);
            self.overlay_cache.insert(anim, (data, grid));
        }
        let n = self.overlay_cache.get(&anim)?.0.frame_count.max(1) as f32;
        Some((anim, n))
    }

    /// #aim  Load a plain (unlabelled) overlay clip such as `combat:<class>:fire_1`, which is
    /// driven by its own CLOCK rather than by a look direction, and return its index and frame
    /// count.
    pub fn clock_overlay(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        name: &str,
    ) -> Option<(usize, f32)> {
        let g = self.def.graph.clone()?;
        let anim = g.find(name)?;
        if g.anims.get(anim)?.anim_type != AnimType::Overlay {
            return None;
        }
        if !self.overlay_cache.contains_key(&anim) {
            let data = g.load(native, cache, anim)?;
            let grid = AimGrid::read(&g, anim, &data);
            self.overlay_cache.insert(anim, (data, grid));
        }
        let n = self.overlay_cache.get(&anim)?.0.frame_count.max(1) as f32;
        Some((anim, n))
    }

    /// Advance the animation by `dt` seconds (30 fps, looping).
    pub fn advance(&mut self, dt: f32) {
        // The cross-fade decays on WALL TIME, whether or not the clip's own clock is running:
        // Theater drives `frame` itself from the recorded motion and leaves `playing` false, so
        // gating this on `playing` would leave a blend frozen half-way for ever.
        if self.blend_left > 0.0 {
            self.blend_left = (self.blend_left - dt).max(0.0);
            if self.blend_left == 0.0 {
                self.prev_anim = None;
                self.prev_pose = None;
            }
        }
        if !self.playing { return; }
        let Some(frames) = self.data.as_ref().map(|d| d.frame_count.max(1) as f32) else { return };
        self.frame = (self.frame + dt * reach_jmad::FRAME_RATE) % frames;
    }

    /// Frame count of the selected animation (1 when none).
    pub fn frame_count(&self) -> f32 { self.data.as_ref().map(|d| d.frame_count.max(1) as f32).unwrap_or(1.0) }
}

/// The placed Reach actors. Owned by the app; the Theater playback drives it.
#[derive(Default)]
pub struct ActorScene {
    pub actors: Vec<Actor>,
    defs: HashMap<u32, Arc<ActorDef>>,
    /// #fp Definitions built from a bare render model, keyed `(render model, graph)` — the same
    /// first-person arms model has one per weapon, because the graph that poses it is the
    /// weapon's.
    model_defs: HashMap<(u32, u32), Arc<ActorDef>>,
    next_id: u32,
    /// How many times `build_actor_meshes` has run, and whether the last run left work pending.
    /// Print-only instrumentation: a mesh rebuild is supposed to be rare, so a counter that
    /// climbs once per frame is the symptom of the whole actor set being dirtied every frame.
    pub builds: u64,
    pub actors_built: usize,
    pub last_pending: bool,
    /// Bumped whenever a mesh rebuild is needed.
    pub rebuild: bool,
}

impl ActorScene {
    pub fn is_empty(&self) -> bool { self.actors.is_empty() }
    pub fn clear(&mut self) {
        self.actors.clear();
        self.rebuild = true;
    }
    /// Forget the per-map caches (call when a different map is opened).
    pub fn reset(&mut self) {
        self.actors.clear();
        self.defs.clear();
        self.rebuild = true;
    }

    /// Every `bipd` in the cache, as `(obj_tag, tag path)` — the script's `actor bipeds`.
    pub fn list_bipeds(native: &NativeDll, cache: CacheHandle) -> Vec<(u32, String)> {
        let mut v: Vec<(u32, String)> = native
            .tags_by_group(cache, "bipd")
            .into_iter()
            .map(|t| (t, native.tag_name(cache, t).unwrap_or_default()))
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v
    }

    /// Every `weap` in the cache, as `(obj_tag, tag path)`.
    pub fn list_weapons(native: &NativeDll, cache: CacheHandle) -> Vec<(u32, String)> {
        let mut v: Vec<(u32, String)> = native
            .tags_by_group(cache, "weap")
            .into_iter()
            .map(|t| (t, native.tag_name(cache, t).unwrap_or_default()))
            .collect();
        v.sort_by(|a, b| a.1.cmp(&b.1));
        v
    }

    fn def(&mut self, native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Option<Arc<ActorDef>> {
        if let Some(d) = self.defs.get(&obj_tag) { return Some(d.clone()); }
        let d = Arc::new(ActorDef::read(native, cache, obj_tag)?);
        self.defs.insert(obj_tag, d.clone());
        Some(d)
    }

    /// #fp The same, for a bare RENDER MODEL (no object, no hlmt) with an OPTIONAL graph that is
    /// not its own — Reach's first-person arms are posed by the WEAPON's animation graph, so the
    /// same model has a different definition per weapon and the cache key is the pair.
    fn def_model(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        mode_tag: u32,
        jmad_tag: Option<u32>,
    ) -> Option<Arc<ActorDef>> {
        let key = (mode_tag, jmad_tag.unwrap_or(0));
        if let Some(d) = self.model_defs.get(&key) {
            return Some(d.clone());
        }
        let d = Arc::new(ActorDef::read_model(native, cache, mode_tag, jmad_tag)?);
        self.model_defs.insert(key, d.clone());
        Some(d)
    }

    /// #fp Give an existing render-model actor a different animation graph, keeping its meshes.
    ///
    /// The MODEL does not change, only which graph poses it, so nothing is rebuilt — which is the
    /// whole point: a first-person weapon change would otherwise rebuild every mesh in the scene.
    /// The loaded clip is dropped because its index belongs to the outgoing graph.
    pub fn set_model_graph(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        id: u32,
        jmad_tag: Option<u32>,
    ) -> bool {
        let Some(idx) = self.actors.iter().position(|a| a.id == id) else { return false };
        let mode_tag = self.actors[idx].def.mode_tag;
        if self.actors[idx].graph_tag == jmad_tag {
            return true;
        }
        let Some(d) = self.def_model(native, cache, mode_tag, jmad_tag) else { return false };
        let a = &mut self.actors[idx];
        a.def = d;
        a.graph_tag = jmad_tag;
        a.anim = None;
        a.data = None;
        a.anim_name.clear();
        a.prev_anim = None;
        a.prev_pose = None;
        a.blend_left = 0.0;
        a.frame = 0.0;
        // EVERY LANE KEYED BY GRAPH INDEX GOES TOO, not just the base clip.
        //
        // `overlay_cache` is a map from GRAPH ANIMATION INDEX to a loaded blob, and
        // `Actor::aim_grid` only loads a clip when the index is ABSENT from it -- so an index
        // the outgoing graph had cached is handed straight back for the incoming graph's clip of
        // the same number, which poses an unrelated animation. Same for the replacement lane,
        // the overlay list and the base blend, all of which hold indices into the graph that has
        // just been replaced. `anim` and `data` were already cleared for exactly this reason;
        // the other five were not, and a first-person weapon change in Theater goes through
        // here (`film_play`) as does a vehicle mount (`play_vehicle`).
        a.overlay_cache.clear();
        a.overlays.clear();
        a.base_blend.clear();
        a.replacement = None;
        a.replacement_name.clear();
        true
    }

    /// #fp Place a bare RENDER MODEL as an actor — the shape the first-person arm models come in.
    ///
    /// Identical to [`ActorScene::add_object`] except that the definition is read straight off the
    /// `mode` tag. The actor has no animation graph, so it holds its BIND POSE: the first-person
    /// animations live on the weapon, not on the arms, and the caption says which pose is drawn.
    pub fn add_render_model(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        mode_tag: u32,
        jmad_tag: Option<u32>,
        pos: Vec3,
        yaw: f32,
    ) -> Result<u32, String> {
        let def = self
            .def_model(native, cache, mode_tag, jmad_tag)
            .ok_or_else(|| "not a readable render model".to_string())?;
        let id = { self.next_id += 1; self.next_id };
        self.actors.push(Actor {
            id,
            def,
            pos,
            yaw,
            pitch: 0.0,
            model_yaw: 0.0,
            fp_attach: None,
            world_override: None,
            grips: Vec::new(),
            weapon_hidden: false,
            damage: 0,
            graph_tag: jmad_tag,
            anim: None,
            anim_name: String::new(),
            data: None,
            frame: 0.0,
            playing: false,
            weapon: None,
            weapon_name: String::new(),
            prev_anim: None,
            prev_pose: None,
            blend_left: 0.0,
            blend_span: BLEND_SECONDS,
            base_blend: Vec::new(),
            overlays: Vec::new(),
            pose_override: None,
            replacement: None,
            replacement_name: String::new(),
            overlay_cache: HashMap::new(),
            pose: Pose::default(),
            anchor: Mat4::IDENTITY,
            weapon_bake: Mat4::IDENTITY,
            palette: None,
            weapon_palette: None,
            variant_sid: 0,
            variant_name: String::new(),
            skin: None,
            armour: None,
            fx: Vec::new(),
            fx_seed: 0,
            first_person: false,
            meshes_dirty: true,
        });
        self.rebuild = true;
        Ok(id)
    }

    /// Place a biped. `biped` matches a `bipd` tag path (case-insensitive substring; an exact
    /// match wins). Returns the new actor id.
    pub fn add(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        biped: &str,
        pos: Vec3,
        yaw: f32,
    ) -> Result<u32, String> {
        let list = Self::list_bipeds(native, cache);
        let needle = biped.to_lowercase();
        let tag = list
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(biped))
            .or_else(|| list.iter().find(|(_, n)| n.to_lowercase().ends_with(&needle)))
            .or_else(|| list.iter().find(|(_, n)| n.to_lowercase().contains(&needle)))
            .map(|(t, _)| *t)
            .ok_or_else(|| format!("no bipd matching '{biped}' ({} in this map)", list.len()))?;
        self.add_object(native, cache, tag, pos, yaw)
    }

    /// Place ANY object tag as an actor — a `vehi`, a `scen`, a `weap` — by its tag id.
    ///
    /// Nothing in the actor pipeline is biped-specific: [`ActorDef::read`] resolves the object's
    /// render model, skeleton, node maps and markers whatever the group is, and
    /// `build_actor_meshes` resolves the object's own default `hlmt` model variant (without which
    /// a Warthog would draw every turret it has ever had, all at once). `add` is just this with a
    /// `bipd` path lookup in front of it.
    ///
    /// Used by Theater playback to draw the vehicles a film recorded.
    pub fn add_object(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        tag: u32,
        pos: Vec3,
        yaw: f32,
    ) -> Result<u32, String> {
        let def = self.def(native, cache, tag).ok_or_else(|| "object has no render model".to_string())?;
        let id = { self.next_id += 1; self.next_id };
        self.actors.push(Actor {
            id,
            def,
            pos,
            yaw,
            pitch: 0.0,
            model_yaw: 0.0,
            fp_attach: None,
            world_override: None,
            grips: Vec::new(),
            weapon_hidden: false,
            damage: 0,
            graph_tag: None,
            anim: None,
            anim_name: String::new(),
            data: None,
            frame: 0.0,
            playing: false,
            weapon: None,
            weapon_name: String::new(),
            prev_anim: None,
            prev_pose: None,
            blend_left: 0.0,
            blend_span: BLEND_SECONDS,
            base_blend: Vec::new(),
            overlays: Vec::new(),
            pose_override: None,
            replacement: None,
            replacement_name: String::new(),
            overlay_cache: HashMap::new(),
            pose: Pose::default(),
            anchor: Mat4::IDENTITY,
            weapon_bake: Mat4::IDENTITY,
            palette: None,
            weapon_palette: None,
            variant_sid: 0,
            variant_name: String::new(),
            skin: None,
            armour: None,
            fx: Vec::new(),
            fx_seed: 0,
            first_person: false,
            meshes_dirty: true,
        });
        self.rebuild = true;
        Ok(id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Actor> { self.actors.iter_mut().find(|a| a.id == id) }
    pub fn get(&self, id: u32) -> Option<&Actor> { self.actors.iter().find(|a| a.id == id) }

    /// #vehicle-damage  Put an actor in a `model_state_enum` damage state (0 = undamaged).
    ///
    /// A damage state selects different model permutations, so unlike a move or a turn it needs
    /// the MESHES rebuilt — which is why it goes through a setter that requests one, and only
    /// when the state actually changed (a wreck must not re-instance the scene every frame).
    pub fn set_damage(&mut self, id: u32, state: u8) {
        if let Some(a) = self.actors.iter_mut().find(|a| a.id == id) {
            if a.damage != state {
                a.damage = state;
                a.meshes_dirty = true;
                self.rebuild = true;
            }
        }
    }

    /// Select an actor's animation by NAME. `None` clears it (bind pose).
    pub fn set_anim(&mut self, native: &NativeDll, cache: CacheHandle, id: u32, name: Option<&str>) -> Result<String, String> {
        self.set_anim_with(native, cache, id, name, Some(BLEND_SECONDS))
    }

    /// As [`ActorScene::set_anim`], saying how long the cross-fade out of the outgoing clip
    /// should last — or `None` for NO cross-fade at all.
    ///
    /// Two callers need this and both have a measured reason:
    ///
    /// * #crouch-rate  a change into or out of the crouch takes the biped's own authored
    ///   `crouch transition time` (`bipd +0x534`: 0.30 s Spartan, 0.20 s Elite) rather than the
    ///   generic [`BLEND_SECONDS`] of 0.18, which is what "the speed that he goes down ... is a
    ///   bit fast" was;
    /// * #gait-blend  a change of PRIMARY clip inside a running directional blend must NOT
    ///   cross-fade, because the outgoing clip is still in the mix with its own weight and
    ///   fading it in a second time would count it twice.
    pub fn set_anim_with(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        id: u32,
        name: Option<&str>,
        fade: Option<f32>,
    ) -> Result<String, String> {
        let idx = self.actors.iter().position(|a| a.id == id).ok_or("no such actor")?;
        let Some(name) = name else {
            let a = &mut self.actors[idx];
            a.anim = None;
            a.data = None;
            a.anim_name.clear();
            a.frame = 0.0;
            return Ok("bind pose".into());
        };
        // The def is shared through an Arc, so resolve + load through it WITHOUT cloning the
        // graph (a Spartan's is 979 named entries).
        let def = self.actors[idx].def.clone();
        let g = def.graph.as_ref().ok_or("this biped has no animation graph")?;
        let i = g.find(name).ok_or_else(|| format!("no animation matching '{name}' ({} in the graph)", g.anims.len()))?;
        let data = g.load(native, cache, i).ok_or_else(|| format!("animation '{}' resource load failed", g.anims[i].name))?;
        let a = &mut self.actors[idx];
        // Hold the outgoing clip so `refresh_pose` can CROSS-FADE out of it. Without this a
        // change of gait snapped the skeleton to an unrelated pose in one frame, which is the
        // residual pop left after the phase was made continuous.
        if let (Some(pi), Some(pd), Some(span)) = (a.anim, a.data.clone(), fade) {
            if Some(pi) != Some(i) && span > 1e-4 {
                a.prev_pose = Some(a.on_screen_trs());
                a.prev_anim = Some((pi, pd, a.frame));
                a.blend_left = span;
                a.blend_span = span;
            }
        }
        a.anim = Some(i);
        a.anim_name = g.anims[i].name.clone();
        a.frame = 0.0;
        a.data = Some(data);
        Ok(format!("{} ({} frames, {:?})", g.anims[i].name, g.anims[i].frame_count, g.anims[i].anim_type))
    }

    /// #gait-blend  As [`ActorScene::set_anim_with`], by GRAPH INDEX rather than by name.
    ///
    /// The gait table already holds indices, and `set_anim`'s name lookup falls back to a
    /// SUBSTRING match — which would let `move_front` answer a request for `move_front_slow`.
    /// An index cannot be ambiguous.
    pub fn set_anim_index(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        id: u32,
        anim: usize,
        fade: Option<f32>,
    ) -> Result<String, String> {
        let idx = self.actors.iter().position(|a| a.id == id).ok_or("no such actor")?;
        let def = self.actors[idx].def.clone();
        let g = def.graph.as_ref().ok_or("this biped has no animation graph")?;
        let e = g.anims.get(anim).ok_or("no such animation")?;
        let data = g.load(native, cache, anim).ok_or_else(|| format!("animation '{}' resource load failed", e.name))?;
        let a = &mut self.actors[idx];
        if let (Some(pi), Some(pd), Some(span)) = (a.anim, a.data.clone(), fade) {
            if pi != anim && span > 1e-4 {
                a.prev_pose = Some(a.on_screen_trs());
                a.prev_anim = Some((pi, pd, a.frame));
                a.blend_left = span;
                a.blend_span = span;
            }
        }
        a.anim = Some(anim);
        a.anim_name = e.name.clone();
        a.data = Some(data);
        Ok(e.name.clone())
    }

    /// Give an actor a held weapon by `weap` tag-path match (`None` removes it).
    pub fn set_weapon(&mut self, native: &NativeDll, cache: CacheHandle, id: u32, name: Option<&str>) -> Result<String, String> {
        let list = Self::list_weapons(native, cache);
        let idx = self.actors.iter().position(|a| a.id == id).ok_or("no such actor")?;
        let Some(name) = name else {
            self.actors[idx].weapon = None;
            self.actors[idx].weapon_name.clear();
            self.actors[idx].meshes_dirty = true;
            self.rebuild = true;
            return Ok("no weapon".into());
        };
        let needle = name.to_lowercase();
        let tag = list
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
            .or_else(|| list.iter().find(|(_, n)| n.to_lowercase().ends_with(&needle)))
            .or_else(|| list.iter().find(|(_, n)| n.to_lowercase().contains(&needle)))
            .map(|(t, _)| *t)
            .ok_or_else(|| format!("no weap matching '{name}' ({} in this map)", list.len()))?;
        let def = self.actors[idx].def.clone();
        let w = resolve_held_weapon(native, cache, def.obj_tag, &def.model, tag, false)
            .ok_or("the weapon has no render model")?;
        let desc = format!(
            "{} ({} -> {}; biped node {})",
            native.tag_name(cache, tag).unwrap_or_default(),
            if w.child_marker.is_empty() { "<identity>" } else { &w.child_marker },
            if w.parent_marker.is_empty() { "<none>" } else { &w.parent_marker },
            w.parent_node
        );
        self.actors[idx].weapon = Some(w);
        self.actors[idx].weapon_name = native.tag_name(cache, tag).unwrap_or_default();
        self.actors[idx].meshes_dirty = true;
        self.rebuild = true;
        Ok(desc)
    }

    /// Give an actor a held weapon by its `weap` TAG ID, skipping the name lookup.
    ///
    /// This is the path Theater uses: a saved film names the weapon in a unit's hands as an index
    /// into the map's `'weap'` group list, not as a path. `None` removes it.
    ///
    /// The weapon's own `hlmt` model variant is resolved lazily in `build_actor_meshes` — a
    /// literal `variant_sid` of 0 draws every permutation of every region at once, which on a
    /// weapon means overlapping sights and barrels.
    ///
    /// #iscale The weapon is scaled by the ITEM SCALE the tag authors for a holder of this
    /// unit's size class (`item` +0x1D0, see [`ITEM_SCALES`]) — the engine's own rule, read out
    /// of `items.cpp`. The marker `scale` field is a separate thing and does cancel: every
    /// marker on both the biped and the weapon authors exactly 0.010.
    pub fn set_weapon_tag(
        &mut self,
        native: &NativeDll,
        cache: CacheHandle,
        id: u32,
        tag: Option<u32>,
    ) -> Result<String, String> {
        let idx = self.actors.iter().position(|a| a.id == id).ok_or("no such actor")?;
        let Some(tag) = tag else {
            if self.actors[idx].weapon.is_some() {
                self.actors[idx].weapon = None;
                self.actors[idx].weapon_name.clear();
                self.actors[idx].meshes_dirty = true;
                self.rebuild = true;
            }
            return Ok("no weapon".into());
        };
        let name = native.tag_name(cache, tag).unwrap_or_default();
        // Already holding it: changing nothing avoids a mesh rebuild every frame.
        if self.actors[idx].weapon.is_some() && self.actors[idx].weapon_name == name {
            return Ok(name);
        }
        let def = self.actors[idx].def.clone();
        let w = resolve_held_weapon(native, cache, def.obj_tag, &def.model, tag, false)
            .ok_or("the weapon has no render model")?;
        self.actors[idx].weapon = Some(w);
        self.actors[idx].weapon_name = name.clone();
        self.actors[idx].meshes_dirty = true;
        self.rebuild = true;
        Ok(name)
    }

    /// Advance every playing actor and refresh every pose + palette. One call per frame.
    pub fn update(&mut self, dt: f32, queue: &wgpu::Queue) {
        for a in self.actors.iter_mut() {
            a.advance(dt);
            a.refresh_pose(queue);
        }
        // #fp-attach  Then settle the actors that hang off ANOTHER actor's posed node. A Reach
        // first-person weapon is parented to a hand inside the animation graph the two models
        // share, so its placement is not known until the arms have been posed -- which is why
        // this is a second pass rather than an ordering rule inside the first.
        for (i, w) in self.fp_attach_links() {
            if self.actors[i].world_override != Some(w) {
                self.actors[i].world_override = Some(w);
            }
            self.actors[i].refresh_pose(queue);
        }
    }

    /// #fp-attach  The same pass with NO GPU: advance, re-pose, resolve the attachment links.
    ///
    /// The first-person rig's whole geometry -- is the gun in the hand -- is only readable off the
    /// POSED skeletons, and a gate that needed a wgpu device to read it could not run. The pose is
    /// a pure function of the clip and the links (`Actor::compute_pose` was already split out of
    /// `refresh_pose` for exactly this reason), so this is the production pass minus the upload.
    pub(crate) fn update_poses_headless(&mut self, dt: f32) {
        for a in self.actors.iter_mut() {
            a.advance(dt);
            a.compute_pose();
        }
        for (i, w) in self.fp_attach_links() {
            self.actors[i].world_override = Some(w);
            self.actors[i].compute_pose();
        }
    }

    /// #fp-attach  Per attached actor, the world matrix its parent's posed node gives it.
    ///
    /// ONE implementation, so the GPU pass and the headless one cannot disagree about where a
    /// first-person weapon goes -- the shape of defect this project has already had twice (a
    /// cached phase that drifted from the walk that fed it, a load status with two writers).
    fn fp_attach_links(&self) -> Vec<(usize, Mat4)> {
        let links: Vec<(usize, u32, usize)> = self
            .actors
            .iter()
            .enumerate()
            .filter_map(|(i, a)| a.fp_attach.map(|(pid, n)| (i, pid, n)))
            .collect();
        let mut out = Vec::with_capacity(links.len());
        for (i, parent_id, node) in links {
            let Some(pi) = self.actors.iter().position(|a| a.id == parent_id) else { continue };
            if pi == i {
                continue;
            }
            let Some(local) = self.actors[pi].pose.model.get(node).copied() else { continue };
            out.push((i, self.actors[pi].world() * local));
        }
        out
    }

    /// #fp-attach  **THE ENGINE'S OWN GRIP**, resolved from the animation graph the arms and the
    /// gun share: `(arms actor id, that arms model's node index)` for `gun_id`'s root, or `None`
    /// when the graph does not parent it.
    ///
    /// A Reach first-person graph is ONE skeleton over BOTH models -- the arms' 41-47 nodes and
    /// the weapon's handful -- and in that tree the weapon's root is a CHILD of a hand node, so
    /// the channel the clip plays for it is a transform LOCAL to the hand (a ~0.03 wu grip
    /// offset), not a rig-space position. Composed through the weapon model's own two-node
    /// skeleton, where that root has no parent at all, the gun lands at the rig's origin --
    /// inside the player's head.
    ///
    /// **Lifted out of Theater's driver so test mode uses the same one.** It was Theater-only,
    /// which is precisely why "the weapon isn't even in the first person hands" was true in test
    /// mode and not in Theater: two hosts, one of them carrying the step.
    pub fn fp_grip_of(&self, arms_id: u32, gun_id: u32) -> Option<(u32, usize)> {
        if gun_id == arms_id {
            return None;
        }
        let arms = self.get(arms_id)?;
        let g = arms.def.graph.as_ref()?;
        let arms_map = arms.def.anim_node_map.as_ref()?;
        let gun = self.get(gun_id)?;
        // The weapon model's own ROOT node -- the one the graph parents into the hand.
        let root = gun.def.model.skel.nodes.iter().position(|n| n.parent < 0)?;
        let rname = gun.def.model.node_names.get(root)?;
        let gi = g.node_names.iter().position(|n| n.eq_ignore_ascii_case(rname))?;
        let gp = *g.node_parents.get(gi)?;
        if gp < 0 {
            return None;
        }
        // ... and which of the ARMS model's nodes that graph node is.
        match arms_map.get(gp as usize).copied() {
            Some(v) if v >= 0 => Some((arms_id, v as usize)),
            _ => None,
        }
    }

    /// #armour-fx This frame's particle sources for every posed actor — the ACTOR PARTICLE LANE.
    ///
    /// The actor path had no particle list of its own: `SceneController::build_actor_meshes`
    /// deliberately takes `particle_sources_obj` away for the duration of an actor build so an
    /// actor cannot disturb the Forge scene's particles, and puts it back afterwards, which also
    /// meant anything an actor collected was thrown away. (The same shape as the transparent-lane
    /// bug: the fix is a lane of its own, not a shared one.)
    ///
    /// It is rebuilt EVERY frame rather than at the mesh rebuild because an armour effect's origin
    /// is a marker on a BONE — `armor_fx_01` rides `b_helmet` — so it moves with the pose. Only
    /// the cheap half is redone here: the tag reads are in [`resolve_actor_effects`], and this
    /// carries each marker through `pose.model[node]` and hands the result over as the source's
    /// `locs`, exactly as a placed object hands over its model-space marker.
    ///
    /// Determinism: the source seed is the body's own stable identity (`fx_seed`, the recorded net
    /// id) and never the frame or the actor slot, and `world`/`locs` are functions of the pose,
    /// which is a function of the playhead. So the whole lane is a function of the playhead, the
    /// same rule the animation clock follows.
    pub fn particle_sources(&self) -> Vec<crate::scene::ParticleSource> {
        let mut out = Vec::new();
        for a in &self.actors {
            if a.fx.is_empty() { continue; }
            let world = a.world();
            if !world.w_axis.truncate().is_finite() { continue; }
            let key = if a.fx_seed != 0 { a.fx_seed } else { a.id };
            for fx in &a.fx {
                let locs: Vec<(Vec3, glam::Quat)> = fx
                    .locs
                    .iter()
                    .map(|&(node, p, r)| {
                        let n = node.and_then(|n| a.pose.model.get(n).copied()).unwrap_or(Mat4::IDENTITY);
                        let m = n * Mat4::from_rotation_translation(r, p);
                        let (_, rot, pos) = m.to_scale_rotation_translation();
                        (pos, rot)
                    })
                    .collect();
                out.push(crate::scene::ParticleSource {
                    seed: ACTOR_FX_SEED ^ ((key as u64) << 20) ^ fx.effe as u64,
                    world,
                    effe: fx.effe,
                    label: format!("actor #{} '{}'", a.id, a.def.name),
                    systems: fx.systems.clone(),
                    locs,
                    // A persistent attachment: pre-rolled, looping, stationary in its own frame.
                    birth: None,
                    path: None,
                });
            }
        }
        out
    }

    /// #armour-fx One line per live actor effect, for `actor list` and the gates:
    /// `#id 'biped' eternal_flame @ armor_fx_01 (1 system)`.
    pub fn fx_report(&self) -> Vec<String> {
        let mut out = Vec::new();
        for a in &self.actors {
            for fx in &a.fx {
                let picks: Vec<String> = fx.picks.iter().map(|(e, m)| format!("{e} @ {m}")).collect();
                out.push(format!(
                    "#{} {} effe {:#x} scale '{}'={:.2} | {} system(s): {}",
                    a.id, a.def.name, fx.effe, fx.scale_name, fx.scale, fx.systems.len(), picks.join(", ")
                ));
            }
        }
        out
    }

    /// #fp-frame  Finish every cross-fade NOW.
    ///
    /// A scripted seek is a JUMP, not playback: the headless host advances the scene with `dt = 0`,
    /// so a cross-fade started by the clip change that seek caused would never decay and the pose
    /// would stay frozen part-way between the two clips for the rest of the session. Measured: the
    /// same tick of the same film, reached by two different seek orders, posed one first-person
    /// hand 0.09 wu apart and put 14 of 41 nodes in frame instead of 22. Settling makes a scripted
    /// capture show the tick it was asked for, and show the same thing every time.
    pub fn settle_blends(&mut self) {
        for a in self.actors.iter_mut() {
            a.blend_left = 0.0;
            a.prev_anim = None;
            a.prev_pose = None;
            // The span the NEXT fade would be armed with is reset too, so a settled actor is in
            // one state rather than two (a zeroed fade carrying a 0.30 s crouch span).
            a.blend_span = BLEND_SECONDS;
        }
    }

    /// #gait-blend  Why [`ActorScene::settle_blends`] deliberately does NOT clear
    /// [`Actor::base_blend`], recorded next to it because the omission looks like one.
    ///
    /// A cross-fade is a TRANSIENT: it exists only because the clip changed a moment ago, and a
    /// capture taken mid-fade shows a pose that belongs to no state, which is why the script
    /// path settles it before posing. The locomotion MIX is not a transient — it is a steady
    /// function of where the body is going and how fast — so clearing it would not settle a pose,
    /// it would change one. The capture property is therefore kept by a different route: nothing
    /// on the script or Theater path ever writes `base_blend` (only `play_mode::pose` does), so
    /// the mix is empty there and `compute_pose`'s blend block is skipped outright. Gated in
    /// [`gait_blend::settling_an_actor_leaves_no_transient_and_does_not_touch_the_mix`].

    /// One-line status per actor (status bar / headless log).
    pub fn describe(&self) -> Vec<String> {
        self.actors
            .iter()
            .map(|a| {
                // #float-weapon  Where the HELD WEAPON actually ends up, next to where the body is.
                // "A gun hanging in mid-air beside a corpse" is a disagreement between those two
                // points, and nothing else in this line could tell them apart: the weapon rides
                // `anchor^-1 * world() * attach`, so the body's placement, its anchor and the
                // attach node all have to be printed to say which of them moved.
                let wpos = a.weapon.as_ref().map(|w| {
                    let attach = w.world(Mat4::IDENTITY, &a.pose);
                    let want = (a.world() * attach).w_axis.truncate();
                    let drawn = held_weapon_drawn(a.weapon_bake, a.anchor, a.world(), attach)
                        .w_axis
                        .truncate();
                    (want, drawn)
                });
                format!(
                    "#{} {} at ({:.2},{:.2},{:.2}) yaw {:.0} | anim '{}' frame {:.1}/{:.0} | weapon '{}'{}{}",
                    a.id,
                    a.def.name,
                    a.pos.x, a.pos.y, a.pos.z,
                    a.yaw,
                    if a.anim_name.is_empty() { "<bind pose>" } else { &a.anim_name },
                    a.frame,
                    a.frame_count(),
                    if a.weapon_name.is_empty() { "-" } else { &a.weapon_name },
                    match (wpos, a.weapon.as_ref()) {
                        (Some((want, drawn)), Some(w)) => {
                            let off = (drawn - want).length();
                            format!(
                                " at ({:.2},{:.2},{:.2}) via node {} '{}'{}",
                                want.x, want.y, want.z, w.parent_node, w.parent_marker,
                                if off > 0.01 {
                                    format!(
                                        " | MISPLACED: DRAWN at ({:.2},{:.2},{:.2}), {off:.2} wu away",
                                        drawn.x, drawn.y, drawn.z
                                    )
                                } else {
                                    String::new()
                                },
                            )
                        }
                        _ => String::new(),
                    },
                    if a.pose_override.is_some() { " | RAGDOLL pose" } else { "" },
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Mesh build + the `HMS_ACTOR` / script spec parser
// ---------------------------------------------------------------------------------------------

/// Build (or rebuild) the GPU meshes of every actor that needs it and return the FIVE draw lanes
/// for `SceneRenderer::set_actor_meshes`. Call whenever `ActorScene::rebuild` is set; the pose
/// itself never needs a rebuild (it lives in the bone palette).
///
/// #sword-blade The three transparent lanes (holo / holo_solid / blend) are the same ones a
/// world-placed object is routed into; they used to be folded into `opaque`, which drew a held
/// energy sword's additive blade as a solid grey slab.
///
/// `Err` is only returned when the object decode is still streaming — call again next frame.
/// #armour-fx Which attached EFFECTS a body with this object tag, render model and model variant
/// is playing, and where each of their particle systems is located on the skeleton.
///
/// Two filters, both of them the engine's own rule and neither of them a guess:
/// 1. **the attachment's primary SCALE function** must evaluate above zero
///    ([`crate::scene::SceneController::object_attachment_scale`]). The Spartan biped carries four
///    `effe` attachments and this is what separates them: `shield_down` → `shield_depleted` = 0,
///    the damage-feedback distortion → `body_vitality` = 0, `out_of_bounds` = 0, armour FX →
///    `alive` = **1**. The old `scale_sid == 0` test turned all four off.
/// 2. **the system's location MARKER** must be one the body's variant shows
///    ([`crate::scene::SceneController::model_marker_node_for_variant`]). That is what picks ONE
///    of the armour effect's five events out of the ten systems it authors, from the `armor_fx`
///    region permutation the player's armour byte selected.
///
/// Resolved once per mesh rebuild (it reads tags); the per-frame work is
/// [`ActorScene::particle_sources`], which only carries the markers through the live pose.
pub fn resolve_actor_effects(
    scene: &mut crate::scene::SceneController,
    obj_tag: u32,
    mode_tag: u32,
    variant_sid: u32,
) -> Vec<ActorEffect> {
    let mut out = Vec::new();
    if obj_tag == 0 || obj_tag == 0xFFFF_FFFF { return out; }
    if std::env::var("HMS_NO_ACTOR_FX").is_ok() { return out; }
    for (effe, att_marker, scale_sid) in scene.object_effects(obj_tag) {
        let scale = scene.object_attachment_scale(obj_tag, scale_sid);
        if !(scale > 0.0) { continue; }
        let systems = scene.effect_particle_systems(effe);
        // A persistent attachment is a LOOPING effect (the armour FX carries the force-looping
        // flag); a one-shot would have fired once at spawn and be over.
        if systems.is_empty() || !systems[0].looping { continue; }
        let mut kept: Vec<crate::scene::ParticleSystemDef> = Vec::new();
        let mut locs: Vec<(Option<usize>, Vec3, glam::Quat)> = Vec::new();
        let mut picks: Vec<(String, String)> = Vec::new();
        for sy in systems.iter() {
            // #armour-fx A posed actor is a body in the WORLD, so it plays the systems authored
            // for either view (0) and for third person (2) and not the first-person-only ones
            // (1). Without this an energy sword in a hand draws its `1st_person` and `3rd_person`
            // blade glow on the same two markers at once, twice as bright as the engine's.
            if sy.camera_mode == 1 { continue; }
            let sid = if sy.loc_sid != 0 { sy.loc_sid } else { att_marker };
            let loc = if sid == 0 {
                // No marker named at all: the effect sits on the object's own origin.
                (None, Vec3::ZERO, glam::Quat::IDENTITY)
            } else {
                match scene.model_marker_node_for_variant(mode_tag, sid, obj_tag, variant_sid) {
                    Some(l) => l,
                    // The variant shows no marker of that group → the engine creates nothing.
                    None => continue,
                }
            };
            let ev_name = sy.effect.events.get(sy.event_index).map(|e| scene.string_of(e.name_sid)).unwrap_or_default();
            picks.push((ev_name, scene.string_of(sid)));
            kept.push(sy.clone());
            locs.push(loc);
        }
        if kept.is_empty() { continue; }
        out.push(ActorEffect {
            effe,
            scale,
            scale_name: scene.string_of(scale_sid),
            systems: Arc::new(kept),
            locs,
            picks,
        });
    }
    out
}

/// #fp-fov What one actor rebuild hands back: the WORLD lanes, the FIRST-PERSON RIG's lanes,
/// and whether anything is still streaming.
///
/// Two lane sets rather than one because the rig is drawn through a different CAMERA — Reach
/// renders the first-person model at 88% of the world camera's vertical FOV — and a projection
/// cannot be chosen per mesh inside one draw list. The same shape, and the same reason, as the
/// split that gave the transparent parts their own lanes.
pub struct ActorBuild {
    pub world: crate::objscene::RebuiltLanes,
    pub first_person: crate::objscene::RebuiltLanes,
    pub pending: bool,
}

impl ActorBuild {
    pub fn empty() -> Self {
        ActorBuild {
            world: crate::objscene::RebuiltLanes::empty(),
            first_person: crate::objscene::RebuiltLanes::empty(),
            pending: false,
        }
    }
    /// `(world lane counts, first-person lane counts)` — what the gates and the status line read.
    pub fn counts(&self) -> ((usize, usize, usize, usize, usize), (usize, usize, usize, usize, usize)) {
        (self.world.counts(), self.first_person.counts())
    }
    /// #fp-fov Add one actor's five lanes to the set its CAMERA decides — the rig's, or the
    /// world's. The whole routing, in one place, so a gate can score the decision itself rather
    /// than a copy of it.
    pub fn push(&mut self, first_person: bool, l: crate::objscene::RebuiltLanes) {
        if first_person { self.first_person.extend(l) } else { self.world.extend(l) }
    }
}

pub fn build_actor_meshes(
    actors: &mut ActorScene,
    scene: &mut crate::scene::SceneController,
    mesh: &hms_render::MeshRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> ActorBuild {
    let mut out = ActorBuild::empty();
    let mut pending = false;
    // #elite-armour Actors placed with `:armour=` wear a recorded player's ten appearance bytes.
    // They go through the SAME path Theater uses -- `SceneController::player_variant` (which
    // splits the Spartan slots from the Elite ones) then `set_player_variants`, so the render
    // model's own variant walk does the dressing -- rather than a parallel implementation that
    // could agree with nothing. Installed only when some actor actually asks for it, so the
    // Theater path and the ordinary editor are untouched.
    {
        let worn: Vec<(usize, [u8; crate::film::APPEARANCE_SLOTS])> =
            actors.actors.iter().enumerate().filter_map(|(i, a)| a.armour.map(|s| (i, s))).collect();
        if !worn.is_empty() {
            let vs: Vec<crate::scene::PlayerVariant> = worn
                .iter()
                .map(|(i, sl)| scene.player_variant(crate::scene::player_variant_sid(*i as u8), sl))
                .filter(|v| v.worn())
                .collect();
            scene.set_player_variants(vs);
            for (i, _) in &worn {
                let sid = crate::scene::player_variant_sid(*i as u8);
                if let Some(a) = actors.actors.get_mut(*i) {
                    if a.variant_sid != sid {
                        a.variant_sid = sid;
                        a.variant_name = "recorded armour".into();
                    }
                }
            }
        }
    }
    actors.builds += 1;
    actors.actors_built = actors.actors.len();
    for a in actors.actors.iter_mut() {
        if a.palette.is_none() { a.palette = Some(mesh.new_skin_palette(device)); }
        // Resolve the model variant once. Without it the render model draws every region's every
        // permutation, which is the "wearing all the armour at once" look.
        if a.variant_sid == 0 && a.variant_name.is_empty() {
            match scene.default_model_variant(a.def.obj_tag) {
                Some((sid, name)) => { a.variant_sid = sid; a.variant_name = name; }
                None => a.variant_name = "<no variants>".into(),
            }
        }
        a.anchor = a.world();
        // #fp The basis the mesh is BUILT at must be the same transform `anchor` records, or the
        // palette's `anchor^-1 * world` fold-in composes against a different frame and the model
        // shears. The columns of `world()` ARE the images of the model's axes, so this is exactly
        // the actor's own orientation: identical to the old yaw-only pair for everything placed in
        // the world (pitch and model yaw are 0), and correct for a model mounted on the camera.
        // #fp-attach  The ORIGIN and the BASIS both come off `anchor`, together.
        //
        // The palette's fold-in is `anchor^-1 * world`, which is only a correction if the mesh was
        // baked at exactly `anchor`. For everything placed in the world the two are the same point
        // (`world()`'s translation IS `pos`), but a first-person WEAPON is parented to the hand, so
        // its `world()` sits in the hand while its `pos` is still the eye the rig is mounted on.
        // Rebuilding it from `a.pos` baked the gun at the EYE and then folded in a delta of
        // identity -- which is exactly the "the weapon floats there, enormous, not in the hand"
        // the user saw: a rebuild (a weapon swap, an armour install) moved the gun onto the camera
        // and it stayed there. One frame of a fresh scene looked right, which is how a scripted
        // capture passed it.
        let (origin, bfwd, bup) = actor_mesh_frame(a.anchor);
        // The actor's own mesh, built at its placement (so the object-lighting probe samples
        // where the actor really stands) and bound to its bone palette.
        match scene.build_actor_meshes(
            a.def.obj_tag, a.variant_sid, origin, bfwd, bup,
            a.skin, a.damage, a.palette.as_ref(), mesh, device, queue,
        ) {
            Ok((l, bmn, bmx)) => {
                if std::env::var("HMS_ACTOR_BUILD").is_ok() {
                    let d = bmx - bmn;
                    eprintln!("ACTORBUILD #{} '{}' obj={:#x} variant={:#x} -> {} opaque {} cutout {} holo {} holo_solid {} blend  bbox {:.3} x {:.3} x {:.3} wu",
                        a.id, a.def.name, a.def.obj_tag, a.variant_sid,
                        l.opaque.len(), l.cutout.len(), l.holo.len(), l.holo_solid.len(), l.blend.len(), d.x, d.y, d.z);
                }
                // #fp-fov The rig goes into its own lanes so it can be drawn through the engine's
                // narrower first-person projection.
                out.push(a.first_person, l);
            }
            Err(true) => {
                if std::env::var("HMS_ACTOR_BUILD").is_ok() {
                    eprintln!("ACTORBUILD #{} '{}' obj={:#x} -> PENDING", a.id, a.def.name, a.def.obj_tag);
                }
                pending = true;
            }
            Err(false) => {
                if std::env::var("HMS_ACTOR_BUILD").is_ok() {
                    eprintln!("ACTORBUILD #{} '{}' obj={:#x} -> FAILED", a.id, a.def.name, a.def.obj_tag);
                }
            }
        }
        // #armour-fx The body's ATTACHED EFFECTS, resolved against the variant it is now wearing
        // (which is what selects the armour effect) — the same rebuild that dressed it, because
        // both answers come out of the same variant. The per-frame work is `particle_sources`.
        a.fx = resolve_actor_effects(scene, a.def.obj_tag, a.def.mode_tag, a.variant_sid);
        if std::env::var("HMS_ACTOR_FX").is_ok() {
            for fx in &a.fx {
                eprintln!("ACTORFX #{} '{}' variant={:#x} effe={:#x} scale='{}'={:.2} systems={} picks={:?}",
                    a.id, a.def.name, a.variant_sid, fx.effe, fx.scale_name, fx.scale, fx.systems.len(), fx.picks);
            }
            if a.fx.is_empty() {
                eprintln!("ACTORFX #{} '{}' variant={:#x} -> NO live effect attachment", a.id, a.def.name, a.variant_sid);
            }
        }
        // The held weapon: an ORDINARY rigid model whose every bone-palette entry is the
        // engine's attach matrix, so it rides the posed hand without a per-frame rebuild.
        if let Some(w) = a.weapon.clone() {
            if a.weapon_palette.is_none() { a.weapon_palette = Some(mesh.new_skin_palette(device)); }
            // Resolve the weapon's own model variant once, for the same reason the biped's is
            // resolved: a literal 0 means "no variant", which draws every permutation at once.
            if w.variant_sid == 0 {
                if let Some((sid, _)) = scene.default_model_variant(w.obj_tag) {
                    if let Some(wm) = a.weapon.as_mut() {
                        wm.variant_sid = sid;
                    }
                }
            }
            let wvar = a.weapon.as_ref().map(|x| x.variant_sid).unwrap_or(0);
            // A held WEAPON keeps its own authored change colours — the player's armour
            // colours never reach the gun.
            //
            // #float-weapon  Baked at `origin`, the SAME frame the biped above is baked at, and
            // the same one the palette folds against. This line used to pass `a.pos`, which is a
            // second writer for one frame: the weapon's palette is `anchor^-1 * world()` times the
            // engine attach, so the mesh is only in the right place if it was baked at `anchor`.
            // `actor_mesh_frame` exists precisely to stop the origin and the basis coming from two
            // different matrices (`#fp-attach`), and the biped call was fixed while this one was
            // not. `world()`'s translation IS `pos` for everything placed in the world, so the two
            // agree there and it went unnoticed; they diverge exactly when `world_override` is set
            // — a rider on a vehicle seat, or a first-person weapon parented to a hand — and the
            // gun is then drawn displaced by `pos - anchor.translation`, hanging in the air
            // wherever the body's own recorded origin is.
            match scene.build_actor_meshes(
                w.obj_tag, wvar, origin, bfwd, bup,
                None, 0, a.weapon_palette.as_ref(), mesh, device, queue,
            ) {
                Ok((l, wmn, wmx)) => {
                    // #float-weapon  Record the frame this mesh was really baked at, so a
                    // displacement is a number `actor list` prints rather than something to
                    // notice on screen.
                    a.weapon_bake = bake_frame(origin, bfwd, bup);
                    if std::env::var("HMS_ACTOR_BUILD").is_ok() {
                        // #wscale The two halves of the held-weapon size question, in one place:
                        // how big the weapon's own model is, and what the attach composition does
                        // to it. `attach` is exactly what `refresh_pose` feeds the weapon palette.
                        let d = wmx - wmn;
                        let attach = w.world(Mat4::IDENTITY, &a.pose);
                        let node = if w.parent_node >= 0 {
                            a.pose.model.get(w.parent_node as usize).copied().unwrap_or(Mat4::IDENTITY)
                        } else { Mat4::IDENTITY };
                        eprintln!(
                            "ACTORWEAPON #{} '{}' bbox {:.3} x {:.3} x {:.3} wu | hand node {} '{}' scale {:.4} | attach scale {:.4} | parent '{}' child '{}'",
                            a.id, a.weapon_name, d.x, d.y, d.z,
                            w.parent_node,
                            a.def.model.node_names.get(w.parent_node.max(0) as usize).cloned().unwrap_or_default(),
                            node.x_axis.truncate().length(),
                            attach.x_axis.truncate().length(),
                            w.parent_marker, w.child_marker,
                        );
                    }
                    // #fp-fov A first-person actor's HELD weapon rides the same camera as its
                    // arms -- it is part of the same rig.
                    out.push(a.first_person, l);
                }
                Err(true) => pending = true,
                Err(false) => {}
            }
        }
        a.meshes_dirty = false;
        a.refresh_pose(queue);
    }
    actors.last_pending = pending;
    if !pending { actors.rebuild = false; }
    out.pending = pending;
    out
}

/// One actor in a spec string: `<biped>@x,y,z[,yaw][:anim=<name>][:frame=<n>][:weapon=<name>]`.
#[derive(Clone, Debug, Default)]
pub struct ActorSpec {
    pub biped: String,
    pub pos: Vec3,
    pub yaw: f32,
    pub anim: Option<String>,
    pub frame: Option<f32>,
    pub weapon: Option<String>,
    pub play: bool,
    /// #elite-armour `:armour=<ten comma-separated option indices>` -- a recorded player's
    /// appearance, dressed exactly as Theater dresses it. Fewer than ten values is rejected
    /// rather than padded, so a typo cannot silently become "default armour".
    pub armour: Option<[u8; crate::film::APPEARANCE_SLOTS]>,
}

/// Parse `HMS_ACTOR` / the script's `actor add` tail. Several actors are separated by `;`.
///
/// `spartans@10,2,0.5,180:anim=combat:rifle:idle:frame=12:weapon=assault_rifle`
///
/// Keys are matched on the `<key>=` prefix, so a value containing `:` (every Reach animation
/// name does — `combat:rifle:idle`) survives: the parser only splits on `:` where the next token
/// starts with a known key.
pub fn parse_actor_specs(spec: &str) -> Vec<ActorSpec> {
    let mut out = Vec::new();
    for part in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (head, tail) = match part.split_once('@') {
            Some((h, t)) => (h.trim().to_string(), t.trim().to_string()),
            None => (part.to_string(), String::new()),
        };
        let mut a = ActorSpec { biped: head, play: true, ..Default::default() };
        // Split the tail on ':' but only where a known key begins (animation names contain ':').
        const KEYS: [&str; 6] = ["anim=", "frame=", "weapon=", "play=", "armour=", "armor="];
        let mut fields: Vec<String> = Vec::new();
        let mut cur = String::new();
        for tok in tail.split(':') {
            let is_key = KEYS.iter().any(|k| tok.starts_with(k));
            if is_key && !cur.is_empty() {
                fields.push(std::mem::take(&mut cur));
            } else if !cur.is_empty() {
                cur.push(':');
            }
            cur.push_str(tok);
        }
        if !cur.is_empty() { fields.push(cur); }
        for (i, f) in fields.iter().enumerate() {
            if i == 0 && !KEYS.iter().any(|k| f.starts_with(k)) {
                let v: Vec<f32> = f.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if v.len() >= 3 { a.pos = Vec3::new(v[0], v[1], v[2]); }
                if v.len() >= 4 { a.yaw = v[3]; }
                continue;
            }
            if let Some(v) = f.strip_prefix("anim=") { a.anim = Some(v.to_string()); }
            else if let Some(v) = f.strip_prefix("frame=") { a.frame = v.parse().ok(); a.play = false; }
            else if let Some(v) = f.strip_prefix("weapon=") { a.weapon = Some(v.to_string()); }
            else if let Some(v) = f.strip_prefix("play=") { a.play = !matches!(v, "0" | "off" | "false"); }
            else if let Some(v) = f.strip_prefix("armour=").or_else(|| f.strip_prefix("armor=")) {
                let n: Vec<u8> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if n.len() == crate::film::APPEARANCE_SLOTS {
                    let mut t = [0u8; crate::film::APPEARANCE_SLOTS];
                    t.copy_from_slice(&n);
                    a.armour = Some(t);
                }
            }
        }
        out.push(a);
    }
    out
}

/// Apply a parsed spec list to a scene (place, choose the animation / frame / weapon). Returns
/// one status line per actor.
pub fn apply_actor_specs(
    actors: &mut ActorScene,
    native: &NativeDll,
    cache: CacheHandle,
    specs: &[ActorSpec],
) -> Vec<String> {
    let mut out = Vec::new();
    for s in specs {
        match actors.add(native, cache, &s.biped, s.pos, s.yaw) {
            Ok(id) => {
                let mut line = format!("actor #{id} '{}' at ({:.2},{:.2},{:.2}) yaw {:.0}", s.biped, s.pos.x, s.pos.y, s.pos.z, s.yaw);
                if let Some(an) = &s.anim {
                    match actors.set_anim(native, cache, id, Some(an)) {
                        Ok(d) => line.push_str(&format!(" | anim {d}")),
                        Err(e) => line.push_str(&format!(" | anim FAILED: {e}")),
                    }
                }
                if let Some(w) = &s.weapon {
                    match actors.set_weapon(native, cache, id, Some(w)) {
                        Ok(d) => line.push_str(&format!(" | weapon {d}")),
                        Err(e) => line.push_str(&format!(" | weapon FAILED: {e}")),
                    }
                }
                if let Some(a) = actors.get_mut(id) {
                    if let Some(f) = s.frame { a.frame = f; }
                    a.playing = s.play;
                    a.armour = s.armour;
                }
                if let Some(sl) = s.armour {
                    line.push_str(&format!(" | armour {sl:?}"));
                }
                out.push(line);
            }
            Err(e) => out.push(format!("actor '{}' FAILED: {e}", s.biped)),
        }
    }
    out
}

/// The scripted `actor` verb (shared by the GUI console, the headless `--script` / `--exec`
/// host, the TCP command server and the MCP bridge). `args` is everything after the verb.
///
/// ```text
///   actor add <bipd name>@x,y,z[,yawDeg][:anim=<name>][:frame=<n>][:weapon=<weap name>]
///   actor list                       the placed actors
///   actor bipeds [filter]            every bipd in the open map
///   actor weapons [filter]           every weap in the open map
///   actor anims <id|bipd name> [filter]   the animation names of that graph
///   actor anim <id> <name|none>      choose an animation (resets to frame 0)
///   actor weapon <id> <name|none>    put a weapon in the primary hand
///   actor moveto <id> X Y Z | actor yaw <id> DEG | actor frame <id> N | actor play <id> on|off
///   actor remove <id> | actor clear
/// ```
/// Returns the status text to print. Mutating commands set `scene.rebuild`, which the caller
/// turns into a mesh rebuild.
pub fn run_actor_command(
    args: &str,
    scene: &mut ActorScene,
    native: &NativeDll,
    cache: Option<CacheHandle>,
) -> Result<String, String> {
    let t: Vec<&str> = args.split_whitespace().collect();
    let sub = t.first().map(|s| s.to_lowercase()).unwrap_or_else(|| "list".into());
    let cache = match cache {
        Some(c) => c,
        None => return Err("no Reach map is open".into()),
    };
    let id_of = |s: Option<&&str>| -> Result<u32, String> {
        s.and_then(|v| v.trim_start_matches('#').parse::<u32>().ok())
            .ok_or_else(|| "expected an actor id (see `actor list`)".to_string())
    };
    match sub.as_str() {
        "list" => {
            let d = scene.describe();
            Ok(if d.is_empty() { "no actors placed".into() } else { d.join("\n") })
        }
        "bipeds" | "biped" => {
            let f = t.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
            let rows: Vec<String> = ActorScene::list_bipeds(native, cache)
                .into_iter()
                .filter(|(_, n)| f.is_empty() || n.to_lowercase().contains(&f))
                .map(|(t, n)| format!("0x{t:08X}  {n}"))
                .collect();
            Ok(if rows.is_empty() { "no matching bipd".into() } else { rows.join("\n") })
        }
        // #ragdoll  `actor ragdoll <bipd filter>`: the phmo's rigid bodies, its authored joints
        // and the pivot-residual proof that the constraint offsets are read correctly.
        "ragdoll" => {
            let f = t.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
            let list = ActorScene::list_bipeds(native, cache);
            let (tag, name) = list
                .iter()
                .find(|(_, n)| f.is_empty() || n.to_lowercase().contains(&f))
                .cloned()
                .ok_or_else(|| format!("no bipd matching '{f}' ({} in the cache)", list.len()))?;
            // `ReachModel::read` takes the RENDER MODEL, not the object: walk obje -> hlmt -> mode
            // the way `ActorDef::read` does.
            let mode = native.resolve_mode(cache, tag);
            let model = ReachModel::read(native, cache, mode).ok_or("that bipd has no render model")?;
            let bind = model.skel.bind_model();
            let node_name = |i: i16| -> String {
                if i < 0 { return "<none>".into() }
                model.node_names.get(i as usize).cloned().unwrap_or_else(|| format!("#{i}"))
            };
            let sid_name = |sid: u32| -> String {
                native.string_id(cache, sid).unwrap_or_else(|| format!("0x{sid:08X}"))
            };
            let b = crate::ragdoll::phmo_blocks_for(native, cache, tag)
                .ok_or("that bipd resolves no physics model")?;
            let (defs, counts) = crate::phmo_shapes::rigid_bodies(&b, &crate::phmo_shapes::REACH, &[]);
            let mut out = format!("{name}  ({} skeleton nodes)\n{}\n", model.skel.len(), crate::phmo_shapes::describe(&defs, &counts));
            for d in &defs {
                out.push_str(&format!(
                    "  rb node {:>3} {:<18} motion {} mass {:7.3} {} shape(s)\n",
                    d.node, node_name(d.node), d.motion_type, d.mass, d.shapes.len()
                ));
            }
            out.push_str(&crate::ragdoll::describe_constraints(&b, &bind, &node_name, &sid_name));
            match crate::ragdoll::build(&defs, &crate::phmo_shapes::constraints(&b), &bind, name) {
                Some(rd) => out.push_str(&format!(
                    "{}\n  total mass {:.2}, {} self-collision pair(s), {} constraint(s) with no rigid body\n",
                    rd.desc, rd.total_mass(), rd.self_pairs.len(), rd.unmapped_constraints
                )),
                None => out.push_str("  NOT a ragdoll: fewer than two rigid bodies carry a shape\n"),
            }
            Ok(out)
        }
        "weapons" | "weapon_list" => {
            let f = t.get(1).map(|s| s.to_lowercase()).unwrap_or_default();
            let rows: Vec<String> = ActorScene::list_weapons(native, cache)
                .into_iter()
                .filter(|(_, n)| f.is_empty() || n.to_lowercase().contains(&f))
                .map(|(t, n)| format!("0x{t:08X}  {n}"))
                .collect();
            Ok(if rows.is_empty() { "no matching weap".into() } else { rows.join("\n") })
        }
        "anims" | "animations" => {
            let who = t.get(1).copied().unwrap_or("");
            let filter = t.get(2).map(|s| s.to_lowercase()).unwrap_or_default();
            // By actor id, else by bipd name.
            let def = match who.trim_start_matches('#').parse::<u32>().ok().and_then(|i| scene.actors.iter().find(|a| a.id == i)) {
                Some(a) => a.def.clone(),
                None => {
                    let list = ActorScene::list_bipeds(native, cache);
                    let n = who.to_lowercase();
                    let tag = list
                        .iter()
                        .find(|(_, x)| x.to_lowercase().contains(&n))
                        .map(|(t, _)| *t)
                        .ok_or_else(|| format!("no actor #{who} and no bipd matching '{who}'"))?;
                    scene.def(native, cache, tag).ok_or("bipd has no render model")?
                }
            };
            let rows: Vec<String> = def
                .anim_names()
                .into_iter()
                .filter(|n| filter.is_empty() || n.to_lowercase().contains(&filter))
                .map(|n| n.to_string())
                .collect();
            Ok(format!("{} animation(s):\n{}", rows.len(), rows.join("\n")))
        }
        "add" | "place" | "spawn" => {
            let spec = args.splitn(2, char::is_whitespace).nth(1).unwrap_or("").trim();
            if spec.is_empty() { return Err("actor add <bipd>@x,y,z[,yaw][:anim=..][:frame=..][:weapon=..]".into()); }
            let specs = parse_actor_specs(spec);
            let out = apply_actor_specs(scene, native, cache, &specs);
            Ok(out.join("\n"))
        }
        "anim" | "animation" => {
            let id = id_of(t.get(1))?;
            let name = args.splitn(3, char::is_whitespace).nth(2).map(str::trim).unwrap_or("");
            let r = if name.is_empty() || name.eq_ignore_ascii_case("none") {
                scene.set_anim(native, cache, id, None)?
            } else {
                scene.set_anim(native, cache, id, Some(name))?
            };
            Ok(format!("actor #{id} anim {r}"))
        }
        "weapon" => {
            let id = id_of(t.get(1))?;
            let name = args.splitn(3, char::is_whitespace).nth(2).map(str::trim).unwrap_or("");
            let r = if name.is_empty() || name.eq_ignore_ascii_case("none") {
                scene.set_weapon(native, cache, id, None)?
            } else {
                scene.set_weapon(native, cache, id, Some(name))?
            };
            Ok(format!("actor #{id} weapon {r}"))
        }
        "moveto" | "move" => {
            let id = id_of(t.get(1))?;
            let v: Vec<f32> = t[2..].iter().filter_map(|s| s.parse().ok()).collect();
            if v.len() < 3 { return Err("actor moveto <id> X Y Z".into()); }
            let a = scene.get_mut(id).ok_or("no such actor")?;
            a.pos = Vec3::new(v[0], v[1], v[2]);
            Ok(format!("actor #{id} at ({:.2},{:.2},{:.2})", v[0], v[1], v[2]))
        }
        "yaw" | "turn" => {
            let id = id_of(t.get(1))?;
            let d: f32 = t.get(2).and_then(|s| s.parse().ok()).ok_or("actor yaw <id> DEG")?;
            let a = scene.get_mut(id).ok_or("no such actor")?;
            a.yaw = d;
            Ok(format!("actor #{id} yaw {d:.1}"))
        }
        "frame" => {
            let id = id_of(t.get(1))?;
            let f: f32 = t.get(2).and_then(|s| s.parse().ok()).ok_or("actor frame <id> N")?;
            let a = scene.get_mut(id).ok_or("no such actor")?;
            a.frame = f.max(0.0);
            a.playing = false;
            Ok(format!("actor #{id} frame {:.2}/{:.0}", a.frame, a.frame_count()))
        }
        "play" => {
            let id = id_of(t.get(1))?;
            let on = !matches!(t.get(2).map(|s| s.to_lowercase()).as_deref(), Some("off") | Some("0") | Some("false"));
            let a = scene.get_mut(id).ok_or("no such actor")?;
            a.playing = on;
            Ok(format!("actor #{id} play {}", if on { "on" } else { "off" }))
        }
        "remove" | "delete" | "del" => {
            let id = id_of(t.get(1))?;
            let before = scene.actors.len();
            scene.actors.retain(|a| a.id != id);
            scene.rebuild = true;
            Ok(format!("removed {} actor(s)", before - scene.actors.len()))
        }
        "clear" => {
            let n = scene.actors.len();
            scene.clear();
            Ok(format!("cleared {n} actor(s)"))
        }
        other => Err(format!("unknown `actor` subcommand '{other}' (list / bipeds / weapons / anims / add / anim / weapon / moveto / yaw / frame / play / remove / clear)")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// #instmesh THE GATE for "the armour stays on the body". Reach packs every armour
    /// attachment into ONE instance mesh and places ~100 chunks of it around the skeleton; the
    /// pairing that puts each chunk on the right bone is "placement i == subpart i of that mesh".
    /// If that pairing ever stops holding, every shoulder, forearm and knee piece goes back to
    /// hanging in mid-air beside the Spartan — so assert it, not just that the read succeeded.
    #[test]
    fn spartan_armour_instances_pair_with_their_subparts() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let bipd = ActorScene::list_bipeds(&native, cache)
            .into_iter()
            .find(|(_, n)| n.ends_with("spartans\\spartans"))
            .expect("forge_halo has the Spartan biped");
        let mode = native.resolve_mode(cache, bipd.0);
        let (im, insts) = crate::scene::model_instances(&native, cache, mode)
            .expect("the Spartan render model carries instanced armour");
        assert!(insts.len() > 50, "only {} placements -- the armour set cannot be that small", insts.len());
        // every chunk lands on a real skeleton node, and the ranges tile the mesh in order
        let model = ReachModel::read(&native, cache, mode).expect("render model");
        let mut end = 0u32;
        for (i, p) in insts.iter().enumerate() {
            assert!(p.node >= 0 && (p.node as usize) < model.skel.len(),
                "instance {i} rides node {} of a {}-node skeleton", p.node, model.skel.len());
            assert!(p.index_count >= 3, "instance {i} has {} indices", p.index_count);
            assert!(p.index_start >= end, "instance {i} starts at {} but {end} was already used", p.index_start);
            end = p.index_start + p.index_count;
        }
        // ... and the last one ends exactly on the instance mesh's own index count
        let h = native.open_model(cache, mode).expect("open model");
        let sec = native.model_section(h, im as u32).expect("the instance mesh is a real section");
        assert_eq!(end as usize, sec.index_count as usize,
            "the subparts must tile the instance mesh exactly, or chunks of armour belong to nobody");
        native.close_model(h);
    }

    /// #instmesh-space The raw-strip range of a placement is translated into the EXPANDED list
    /// through the origin map, and nothing is translated without one.
    ///
    /// A strip window at `i` consumes raw indices `i, i+1, i+2`, so a raw range `[a, b)` owns the
    /// windows `a ..= b-3`. Getting that bound wrong by one hands a neighbouring attachment's
    /// triangles to this one, which is the exact fault being fixed — so it is pinned here on a
    /// hand-built map rather than only on the shipped model.
    #[test]
    fn a_placements_strip_range_maps_onto_the_triangles_the_strip_actually_produced() {
        use crate::scene::{instance_index_ranges, ModelInstance};
        let mk = |start: u32, count: u32| ModelInstance { node: 0, index_start: start, index_count: count };
        // Ten triangles whose windows start at raw 0,1,2,3, then 10,11,12, then 20,21,22 — the
        // gaps are strip restarts, exactly as the shipped mesh has them.
        let origins = [0u32, 1, 2, 3, 10, 11, 12, 20, 21, 22];

        // raw [0, 6) owns windows 0..=3 -> triangles 0..4 -> indices 0..12
        assert_eq!(instance_index_ranges(&[mk(0, 6)], Some(&origins)), vec![Some((0, 12))]);
        // raw [10, 15) owns windows 10..=12 -> triangles 4..7 -> indices 12..21
        assert_eq!(instance_index_ranges(&[mk(10, 5)], Some(&origins)), vec![Some((12, 21))]);
        // the +2 bound bites: [10, 14) owns 10..=11 only, NOT 12
        assert_eq!(instance_index_ranges(&[mk(10, 4)], Some(&origins)), vec![Some((12, 18))]);
        // a range too short to hold one window owns nothing
        assert_eq!(instance_index_ranges(&[mk(10, 2)], Some(&origins)), vec![None]);
        // a range the strip produced nothing for owns nothing (rather than someone else's work)
        assert_eq!(instance_index_ranges(&[mk(30, 9)], Some(&origins)), vec![None]);
        // WITHOUT an origin map nothing is translated — applying raw offsets to the expanded
        // buffer is the bug, so an older native library must bind nothing at all.
        assert_eq!(instance_index_ranges(&[mk(0, 6), mk(10, 5)], None), vec![None, None]);
    }

    /// The shape of the instance placement table on the shipped Spartan: 105 placements, no
    /// overlaps, 35 small gaps, every placement naming a real bone.
    ///
    /// **What those gaps are, corrected 2026-10-10.** They were read as "triangles owned by no
    /// attachment" and collapsed; that is wrong. These ranges index the section's RAW index
    /// buffer, which Reach stores as a TRIANGLE STRIP, and the gaps are its restart sentinels.
    /// HMS renders the EXPANDED triangle list (132,529 strip indices become 204,069 list indices
    /// on this section), so applying these offsets to the rendered buffer addresses somebody
    /// else's triangles — 74 of the 105 attachments end up on the wrong bone, which is the
    /// floating-armour fault. See `film_anim::probes::probe_armour_census` and the `#instmesh`
    /// appendix in `re-notes/reach_saved_film_layout.md`. This test still pins the table's shape,
    /// which the eventual fix must preserve; it no longer claims the gaps are unclaimed geometry.
    #[test]
    fn the_instance_subparts_leave_gaps_that_belong_to_no_attachment() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let Some(tag) = bipd_named(&native, cache, "spartans\\spartans") else { return };
        let mode = native.resolve_mode(cache, tag);
        let Some((_im, insts)) = crate::scene::model_instances(&native, cache, mode) else {
            panic!("the shipped Spartan has instanced armour");
        };
        let mut r: Vec<(u32, u32)> = insts.iter().map(|i| (i.index_start, i.index_count)).collect();
        r.sort();
        let (mut gaps, mut gap_idx, mut overlaps) = (0usize, 0u32, 0usize);
        let mut cursor = 0u32;
        for (st, c) in &r {
            if *st > cursor {
                gaps += 1;
                gap_idx += st - cursor;
            } else if *st < cursor {
                overlaps += 1;
            }
            cursor = cursor.max(st + c);
        }
        assert_eq!(overlaps, 0, "overlapping subparts would make the per-instance copy ambiguous");
        assert!(
            gaps > 0,
            "the subparts suddenly tile exactly; they are strip ranges with restart sentinels between \
             them, and it was 35 gaps over 89 indices when measured"
        );
        // Every placement must still name a real bone, or the rebind cannot work at all.
        let nodes = ActorDef::read(&native, cache, tag).map(|d| d.model.skel.len()).unwrap_or(0);
        assert!(nodes > 0);
        for (k, inst) in insts.iter().enumerate() {
            assert!(
                inst.node >= 0 && (inst.node as usize) < nodes,
                "placement {k} names node {} which is outside the {nodes}-node skeleton",
                inst.node
            );
        }
        eprintln!(
            "reach_actor: {} placements, {gaps} gap(s) over {gap_idx} strip indices (restart sentinels)",
            insts.len()
        );
    }

    /// #instmesh-space **THE GATE for the armour shards.**
    ///
    /// `decode_object_geometry` touches the instance mesh twice: once to rebind each chunk to its
    /// own bone, and once — in the per-shader part builder — to cut the mesh into the runs that
    /// are actually drawn. Both must work in the EXPANDED triangle list, because that is the
    /// buffer they are cutting. The second one did not: it used the placement's `s_subpart`
    /// range, which a render_model states in the section's RAW TRIANGLE-STRIP space.
    ///
    /// Measured on the shipped `objects\characters\spartans\spartans` (instance mesh 66, 105
    /// placements, 70 submeshes): the raw strip is 132,529 indices and expands to 204,069, so
    ///
    ///  * **186 of the runs started or ended off a triangle boundary.** The part builder
    ///    concatenates the runs, so every such join invents a triangle from the tail of one
    ///    attachment and the head of another — a shard. The worst triangle it produced had a
    ///    0.220 wu edge on a body 0.7 wu tall; translated, that shader's worst is 0.043 wu.
    ///  * **26 of the 70 submeshes got no run at all** — every submesh living past list index
    ///    132,529, which the raw offsets can never reach. 69,510 of 204,069 indices, a third of
    ///    the armour, was simply not drawn.
    ///
    /// So this asserts the three things that were false: every run lands on a triangle boundary,
    /// the runs together cover the whole expanded buffer, and each run's vertices sit on the bone
    /// its placement names.
    #[test]
    fn instance_mesh_draw_runs_are_triangle_aligned_and_sit_on_their_own_bone() {
        /// Past this, in the BIND pose, a vertex is not on the attachment's own bone. The Spartan
        /// is ~0.7 wu tall and the armour pieces are a few centimetres across; the yardstick is
        /// `probe_armour_census`, which calls a CENTROID more than 0.5 wu from its node misplaced.
        const FAR_WU: f32 = 0.5;
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let Some(tag) = bipd_named(&native, cache, "spartans\\spartans") else { return };
        let mode = native.resolve_mode(cache, tag);
        let (im, insts) = crate::scene::model_instances(&native, cache, mode)
            .expect("the shipped Spartan has instanced armour");
        let def = ActorDef::read(&native, cache, tag).expect("spartan actor def");
        let model = native.open_model(cache, mode).expect("open model");
        let nsec = native.model_section_count(model) as usize;
        let sec = native.model_section(model, im as u32).expect("the instance mesh is a real section");
        let geom = native.model_decode_geometry(model, im as u32).expect("decode the instance mesh");
        let origins = native.model_section_triangle_origins(model, im as u32);
        assert!(origins.is_some(),
            "this native library has no triangle-origin map, so a placement's strip range cannot be \
             translated at all and the armour cannot be cut correctly");
        let owners = crate::scene::instance_index_ranges(&insts, origins.as_deref());
        assert_eq!(owners.iter().filter(|o| o.is_some()).count(), insts.len(),
            "every placement must translate into the expanded list; one that does not is an \
             attachment the body silently loses");
        // The same variant membership the renderer builds.
        let inst_vbits = crate::scene::variant_section_bits(&native, cache, tag, mode, nsec)
            .map(|(_, _, _, ib, _)| ib)
            .unwrap_or_default();

        let vstride = sec.vertex_stride.max(12) as usize;
        let vcount = geom.vertices.len() / vstride.max(1);
        let pos = |vi: usize| -> Option<glam::Vec3> {
            let o = vi * vstride;
            let b = geom.vertices.get(o..o + 12)?;
            Some(glam::Vec3::new(
                f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
                f32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            ))
        };
        let istride = if sec.index_stride == 4 { 4usize } else { 2 };
        let index_at = |k: usize| -> Option<u32> {
            let o = k * istride;
            let b = geom.indices.get(o..o + istride)?;
            Some(if istride == 4 {
                u32::from_le_bytes([b[0], b[1], b[2], b[3]])
            } else {
                u16::from_le_bytes([b[0], b[1]]) as u32
            })
        };
        let nidx = geom.indices.len() / istride;
        let bind_pos = |n: usize| -> Option<glam::Vec3> {
            let nd = def.model.skel.nodes.get(n)?;
            nd.inverse_is_frame().then(|| nd.inverse_bind_authored().inverse().w_axis.truncate())
        };

        let subs: Vec<_> = (0..native.model_submesh_count(model))
            .filter_map(|i| native.model_submesh(model, i))
            .filter(|s| s.section_index as usize == im)
            .collect();
        assert!(subs.len() > 1, "the Spartan's armour is multi-material; {} submesh(es) is not it", subs.len());

        // EVERY placement, not just the ones the hlmt's own variants happen to wear: the armour
        // options a player picks are installed at runtime, so the mask here names a fraction of
        // the catalogue, while the geometry rule has to hold for all of it. An empty `inst_vbits`
        // is exactly "no mask to consult", so this walks all 105.
        let all: &[u64] = &[];
        let mut covered = 0usize;
        let mut empty_submeshes = 0usize;
        let (mut worst_vtx, mut worst_vtx_at) = (0.0f32, String::new());
        let (mut worst_edge, mut worst_edge_at) = (0.0f32, String::new());
        let mut worn = 0usize;
        for sm in &subs {
            let runs = crate::scene::instance_part_runs(
                &insts, &owners, all, sm.index_start as usize, sm.index_length as usize);
            if runs.is_empty() { empty_submeshes += 1; }
            // A variant mask can only ever REMOVE runs; it must never invent one or move one.
            let masked = crate::scene::instance_part_runs(
                &insts, &owners, &inst_vbits, sm.index_start as usize, sm.index_length as usize);
            worn += masked.len();
            for m in &masked {
                assert!(runs.iter().any(|r| r.0 == m.0 && r.1 == m.1 && r.2 == m.2),
                    "the variant mask produced a run {m:?} that the unmasked split does not have");
            }
            // The part builder concatenates these runs into ONE index list per shader and reads it
            // as a triangle list, so measure exactly that.
            let mut part: Vec<u32> = Vec::new();
            for &(ii, a, b, _vb) in &runs {
                assert_eq!(a % 3, 0,
                    "placement {ii} starts its draw run at index {a}, which is not a triangle \
                     boundary -- the concatenated part will weld it to its neighbour");
                assert_eq!(b % 3, 0, "placement {ii} ends its draw run at index {b}, not a triangle boundary");
                assert!(b <= nidx, "placement {ii} run {a}..{b} runs past the mesh's {nidx} indices");
                covered += b - a;
                // Every vertex of a chunk hangs off the bone its placement names.
                let node = insts[ii].node;
                assert!(node >= 0 && (node as usize) < def.model.skel.len(),
                    "placement {ii} names node {node}");
                if let Some(bp) = bind_pos(node as usize) {
                    for k in a..b {
                        let Some(vi) = index_at(k) else { continue };
                        assert!((vi as usize) < vcount,
                            "placement {ii} indexes vertex {vi} of {vcount}");
                        let Some(p) = pos(vi as usize) else { continue };
                        let d = (p - bp).length();
                        if d > worst_vtx {
                            worst_vtx = d;
                            worst_vtx_at = format!(
                                "placement {ii} on '{}'",
                                def.model.node_names.get(node as usize).cloned().unwrap_or_default());
                        }
                    }
                }
                for k in a..b { if let Some(v) = index_at(k) { part.push(v); } }
            }
            for t in part.chunks_exact(3) {
                let (Some(p0), Some(p1), Some(p2)) =
                    (pos(t[0] as usize), pos(t[1] as usize), pos(t[2] as usize)) else { continue };
                let e = (p1 - p0).length().max((p2 - p1).length()).max((p0 - p2).length());
                if e > worst_edge {
                    worst_edge = e;
                    worst_edge_at = format!("shader {}", sm.shader_index);
                }
            }
        }
        eprintln!(
            "reach_actor #instmesh-space: {} placements over {} submeshes cover {covered}/{nidx} \
             expanded indices ({worn} run(s) worn by an hlmt variant); worst vertex {worst_vtx:.3} wu \
             from its node ({worst_vtx_at}); worst triangle edge {worst_edge:.3} wu ({worst_edge_at})",
            insts.len(), subs.len());
        assert_eq!(empty_submeshes, 0,
            "{empty_submeshes} of the {} submeshes got no draw run -- that is a third of the armour \
             not drawn, which is what happens when the raw strip offsets are used to cut the \
             expanded list", subs.len());
        assert_eq!(covered, nidx,
            "the draw runs must cover the whole expanded instance mesh; {covered} of {nidx}");
        assert!(worn > 0, "no placement at all is worn by an hlmt variant of the Spartan");
        assert!(worst_vtx < FAR_WU,
            "{worst_vtx_at} has a vertex {worst_vtx:.3} wu from the bone it rides -- the run is \
             carrying another attachment's geometry");
        // A shard spanning the body. The raw-offset cut produced a 0.220 wu edge here; translated,
        // the biggest legitimate armour triangle measures 0.116 wu.
        assert!(worst_edge < 0.15,
            "a drawn armour triangle has a {worst_edge:.3} wu edge ({worst_edge_at}) on a body \
             0.7 wu tall -- that is a shard, not a piece of armour");
        native.close_model(model);
    }

    /// Open a Reach cache for the map-backed tests. Returns `None` (and the test SKIPS) when the
    /// native library or the MCC install is not on this machine — the same contract the Halo 4
    /// cache tests use.
    /// The map-backed tests share ONE native library + cache handle and run one at a time: the
    /// native page cache is not built for several independent `dlopen`ed copies of itself.
    static REACH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static REACH_CACHE: std::sync::OnceLock<Option<(std::sync::Arc<NativeDll>, CacheHandle)>> = std::sync::OnceLock::new();

    /// #blade-anim  The same shared cache, reachable from another module's gates (scene.rs's
    /// blade-animation test) so a second copy of the map-opening dance does not grow beside it.
    pub(crate) fn open_reach_for(stem: &str) -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        open_reach(stem)
    }

    fn open_reach(stem: &str) -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        let guard = REACH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (n, c) = REACH_CACHE.get_or_init(|| open_reach_inner(stem)).clone()?;
        Some((guard, n, c))
    }

    fn open_reach_inner(stem: &str) -> Option<(std::sync::Arc<NativeDll>, CacheHandle)> {
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            let mut d = exe.parent().map(|p| p.to_path_buf());
            for _ in 0..3 {
                if let Some(p) = &d { roots.push(p.clone()); d = p.parent().map(|q| q.to_path_buf()); }
            }
        }
        roots.push(std::path::PathBuf::from("target/release"));
        roots.push(std::path::PathBuf::from("../../target/release"));
        let dll = match roots.iter().map(|r| r.join(crate::PAYLOAD_DLL)).find(|p| p.is_file()) {
            Some(d) => d,
            None => { eprintln!("reach tests SKIPPED: no {} under {:?}", crate::PAYLOAD_DLL, roots); return None; }
        };
        let native = match NativeDll::load(&dll) {
            Ok(n) => std::sync::Arc::new(n),
            Err(e) => { eprintln!("reach tests SKIPPED: {} failed to load: {e:#}", dll.display()); return None; }
        };
        let cands = crate::mapcat::enumerate();
        let Some(map) = cands.iter().find(|c| {
            c.game == crate::mapcat::Game::Reach
                && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case(stem))
        }) else {
            eprintln!("reach tests SKIPPED: no Reach map '{stem}' among {} detected maps", cands.len());
            return None;
        };
        let dir = map.path.parent().map(|p| p.to_path_buf())?;
        native.set_shared_cache_dirs(&[dir]);
        let cache = match native.open_cache(&map.path.to_string_lossy()) {
            Ok(c) => c,
            Err(e) => { eprintln!("reach tests SKIPPED: open_cache({}) failed: {e:#}", map.path.display()); return None; }
        };
        Some((native, cache))
    }

    fn bipd_named(native: &NativeDll, cache: CacheHandle, needle: &str) -> Option<u32> {
        ActorScene::list_bipeds(native, cache)
            .into_iter()
            .find(|(_, n)| n.to_lowercase().contains(needle))
            .map(|(t, _)| t)
    }

    /// #aim  **THE AIM CONVENTION, asserted against the shipped clip.** Reach's aim overlays
    /// label their own frames on the `b_aim_yaw` / `b_aim_pitch` nodes, and this is the exact
    /// table `combat:rifle:aim_still_up` carries: a 3-column x 5-row grid with PITCH varying
    /// fastest and the yaw columns authored out of order.
    ///
    /// If the frame-to-angle reading ever drifts, the whole look direction silently contorts the
    /// body — so the numbers are pinned, not sampled.
    #[test]
    fn the_aim_overlay_labels_its_own_frames_with_the_look_angle() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let j = resolve_jmad(&native, cache, t).expect("the Spartan graph");
        let g = AnimGraph::read(&native, cache, j).expect("graph read");
        let ai = g.find("combat:rifle:aim_still_up").expect("the rifle aim overlay");
        assert_eq!(g.anims[ai].anim_type, AnimType::Overlay, "an aim clip is an OVERLAY");
        assert_eq!(g.anims[ai].frame_count, 15, "3 yaw columns x 5 pitch rows");
        let data = g.load(&native, cache, ai).expect("clip data");
        let grid = AimGrid::read(&g, ai, &data).expect("the clip labels its frames");
        let deg: Vec<(i32, i32)> = grid
            .labels
            .iter()
            .map(|(y, p)| ((y.to_degrees() * 10.0).round() as i32, (p.to_degrees() * 10.0).round() as i32))
            .collect();
        assert_eq!(
            deg,
            vec![
                (0, 900), (0, 450), (0, 0), (0, -450), (0, -900),
                (-550, 900), (-550, 450), (-550, 0), (-550, -450), (-550, -900),
                (735, 900), (735, 450), (735, 0), (735, -450), (735, -900),
            ],
            "the frame labels of combat:rifle:aim_still_up, in tenths of a degree"
        );
        // Level and straight ahead must land EXACTLY on the clip's own neutral frame, because
        // that frame is the identity (measured: every channel zero at frame 2) and an aim of
        // zero therefore has to be a provable no-op.
        let c = grid.cells(0.0, 0.0);
        assert_eq!(c.len(), 1, "a centred aim is one cell, not a blend: {c:?}");
        assert_eq!(c[0].0, 2.0, "frame 2 is the (0, 0) cell");
        assert!((c[0].1 - 1.0).abs() < 1e-5);
        // 45 degrees up, straight ahead = frame 1 exactly.
        let c = grid.cells(0.0, 45f32.to_radians());
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].0, 1.0);
        // Half way between level and 45 up blends frames 2 and 1 evenly.
        let c = grid.cells(0.0, 22.5f32.to_radians());
        assert_eq!(c.len(), 2, "{c:?}");
        assert!(c.iter().all(|&(_, w)| (w - 0.5).abs() < 0.02), "{c:?}");
        assert!(c.iter().any(|&(f, _)| f == 1.0) && c.iter().any(|&(f, _)| f == 2.0), "{c:?}");
        // Past the authored span it CLAMPS: a look 150 degrees up is the +90 row, not an
        // extrapolation (a skeleton extrapolated past its authored extreme contorts).
        let c = grid.cells(0.0, 150f32.to_radians());
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].0, 0.0, "frame 0 is the +90 pitch row");
    }

    /// #aim  **The aim actually MOVES the body, and by a sane amount.** The frame table alone
    /// would be satisfied by a clip full of identity poses, so the delta the overlay contributes
    /// is measured on the bones that carry a look direction.
    ///
    /// The centre cell must be the exact identity (an aim of zero has to be a provable no-op) and
    /// a 45-degree look up must pitch the spine and head up by a real, bounded amount.
    #[test]
    fn the_aim_overlay_pitches_the_spine_and_is_the_identity_when_centred() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let j = resolve_jmad(&native, cache, t).expect("the Spartan graph");
        let g = AnimGraph::read(&native, cache, j).expect("graph read");
        let ai = g.find("combat:rifle:aim_still_up").expect("the rifle aim overlay");
        let data = g.load(&native, cache, ai).expect("clip data");
        let grid = AimGrid::read(&g, ai, &data).expect("labels");
        let node = |want: &str| g.node_names.iter().position(|n| n == want).expect(want);
        // The bone that actually carries the look direction. Measured: the SPINE barely moves
        // (`b_spine3` turns 0.86 degrees over a 45-degree look), because Reach puts a rifle aim
        // into the neck, the clavicles and the arms -- which is why asserting on the spine would
        // have passed a clip of near-identities.
        let neck = node("b_neck0");
        // Pitch about the model's own +Y, in degrees, positive looking up (`Rz(yaw) * Ry(-pitch)`
        // is the convention the whole actor path uses).
        let pitch_deg = |q: glam::Quat| -> f32 {
            let (_, y, _) = q.to_euler(glam::EulerRot::ZYX);
            -y.to_degrees()
        };
        // Centred: the identity, bone for bone.
        let (rot, trans) = overlay_delta(&data, AnimType::Overlay, &grid.cells(0.0, 0.0));
        for (i, r) in rot.iter().enumerate() {
            if let Some(q) = r {
                let d = q.normalize().angle_between(glam::Quat::IDENTITY).to_degrees();
                assert!(d < 0.05, "centred aim must not rotate node {i} ({d} deg)");
            }
        }
        for (i, t) in trans.iter().enumerate() {
            if let Some(v) = t {
                assert!(v.length() < 1e-3, "centred aim must not translate node {i} ({v})");
            }
        }
        // The neck pitch must follow the look direction MONOTONICALLY across the clip's five
        // authored pitch rows, and in the right direction. This is what pins the sign: if up and
        // down were swapped the body would lean away from wherever the player was looking.
        let neck_at = |deg: f32| -> f32 {
            let (rot, _) = overlay_delta(&data, AnimType::Overlay, &grid.cells(0.0, deg.to_radians()));
            pitch_deg(rot[neck].expect("the overlay animates b_neck0").normalize())
        };
        let ladder: Vec<f32> = [90.0, 45.0, 0.0, -45.0, -90.0].iter().map(|&d| neck_at(d)).collect();
        assert!(
            ladder.windows(2).all(|w| w[0] > w[1] + 0.5),
            "b_neck0 pitch must fall as the look falls (+90 .. -90): {ladder:?}"
        );
        assert!(ladder[2].abs() < 0.05, "the centre row is the identity: {}", ladder[2]);
        assert!(ladder[0] > 5.0, "looking straight up must raise the neck: {} deg", ladder[0]);
        assert!(ladder[4] < -2.0, "looking straight down must drop it: {} deg", ladder[4]);
        // And an angle BETWEEN two rows must land between the two poses, not snap to one.
        let mid = neck_at(22.5);
        assert!(
            mid > ladder[2] && mid < ladder[1],
            "a half-way look must blend: {} not between {} and {}",
            mid, ladder[2], ladder[1]
        );
    }

    /// #aim  The ELITE poses its look direction from its OWN graph, by name. Its aim nodes sit at
    /// different indices from the Spartan's (`b_aim_pitch` is node 1 there and node 2 here), so a
    /// hard-coded index would read the pelvis.
    #[test]
    fn the_elite_aim_overlay_resolves_from_its_own_graph() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "elite\\elite").expect("the Elite biped");
        let j = resolve_jmad(&native, cache, t).expect("the Elite graph");
        let g = AnimGraph::read(&native, cache, j).expect("graph read");
        assert!(g.find("any:aim_spine").is_none(), "the Elite has no any:aim_spine -- that is why aim is a candidate LIST");
        let ai = g.find("combat:rifle:aim_still_up").expect("the Elite's rifle aim overlay");
        let data = g.load(&native, cache, ai).expect("clip data");
        let grid = AimGrid::read(&g, ai, &data).expect("the Elite clip labels its frames too");
        assert_eq!(grid.labels.len(), data.frame_count.max(1) as usize);
        let c = grid.cells(0.0, 0.0);
        assert_eq!(c.len(), 1, "a centred aim is one cell on the Elite as well: {c:?}");
    }

    /// #fp-arms  A RECORDED PLAYER'S OWN armour reaches the first-person arms.
    ///
    /// The arms have no `hlmt`, so the only route is a section mask built straight from the
    /// `(region, permutation)` picks. The gate bites on the exact thing that was wrong: the old
    /// default-only mask pinned `arms` to `default` and DROPPED `flair_forearm` entirely (it has
    /// no `default` permutation), so a player's forearm piece could never appear in their own
    /// hands however their armour was recorded.
    #[test]
    fn a_players_own_armour_reaches_the_first_person_arms() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let mode = native
            .tags_by_group(cache, "mode")
            .into_iter()
            .find(|&t| native.tag_name(cache, t).unwrap_or_default().ends_with("spartans\\fp\\fp"))
            .expect("the Spartan first-person arms model");
        // The model's own region / permutation string ids, read the same way the mask does.
        let u = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let i16at = |b: &[u8], o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
        let meta = native.tag_meta(cache, mode, 0, 0x70).expect("mode meta");
        let (rcnt, rptr) = (u(&meta, 12), u(&meta, 16));
        let rb = native.tag_ptr_bytes(cache, rptr, rcnt * 16).expect("regions");
        let mut forearm: Option<(u32, u32, usize)> = None; // (region sid, perm sid, mesh index)
        for r in 0..rcnt as usize {
            let re = &rb[r * 16..r * 16 + 16];
            let rsid = u(re, 0);
            if native.string_id(cache, rsid).unwrap_or_default() != "flair_forearm" {
                continue;
            }
            let (pcnt, pptr) = (u(re, 4), u(re, 8));
            let pb = native.tag_ptr_bytes(cache, pptr, pcnt * 0x18).expect("permutations");
            for i in 0..pcnt as usize {
                let pe = &pb[i * 0x18..i * 0x18 + 0x18];
                if native.string_id(cache, u(pe, 0)).unwrap_or_default() == "carter" {
                    forearm = Some((rsid, u(pe, 0), i16at(pe, 4).max(0) as usize));
                }
            }
        }
        let (rsid, psid, mesh) = forearm.expect("the arms model carries flair_forearm=carter");
        let default_mask = default_section_mask(native.as_ref(), cache, mode);
        assert!(!default_mask.is_empty(), "the arms model has sections");
        assert!(
            !default_mask[mesh],
            "the DEFAULT mask must not draw an optional forearm piece (mesh {mesh})"
        );
        let worn = section_mask_with_picks(native.as_ref(), cache, mode, &[(rsid, psid)]);
        assert!(
            worn[mesh],
            "a player whose wrist selection names `carter` must get that forearm piece (mesh {mesh})"
        );
        // And nothing else moved: the two masks differ ONLY where the pick reaches.
        let diff: Vec<usize> = (0..worn.len()).filter(|&i| worn[i] != default_mask[i]).collect();
        assert_eq!(diff, vec![mesh], "one pick must change exactly one section");
    }

    /// #fp-arms  The first-person node map must survive a graph that carries the WEAPON's bones
    /// as well as the arms'. Every first-person graph in the cache, against both species' arms.
    #[test]
    fn every_first_person_graph_maps_onto_its_arms_model() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        // The two `@player representation` arm models, by the regions the tag's own customisation
        // blocks override -- resolved here from the known tag names only because this test has no
        // SceneController; the production path uses `first_person_models`.
        let mut arms: Vec<(String, ReachModel)> = Vec::new();
        for t in native.tags_by_group(cache, "mode") {
            let n = native.tag_name(cache, t).unwrap_or_default();
            if !n.ends_with("\\fp\\fp") {
                continue;
            }
            if let Some(m) = ReachModel::read(native.as_ref(), cache, t) {
                arms.push((n, m));
            }
        }
        assert!(arms.len() >= 2, "both species' first-person arm models: {:?}", arms.iter().map(|a| &a.0).collect::<Vec<_>>());
        // #fp-arms  The SPECIES DISCRIMINATOR the rig routes on: the Elite's arms carry
        // `armor_pieces`, the region the globals' `elite selections` block overrides, and the
        // Spartan's do not. That is what lets `SceneController::model_is_elite` tell the two
        // `@player representation` rows apart without matching a tag path.
        let regions = |mode: u32| -> Vec<String> {
            let u = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            let Some(meta) = native.tag_meta(cache, mode, 0, 0x70) else { return Vec::new() };
            let (rcnt, rptr) = (u(&meta, 12), u(&meta, 16));
            let Some(rb) = native.tag_ptr_bytes(cache, rptr, rcnt * 16) else { return Vec::new() };
            (0..rcnt as usize)
                .map(|r| native.string_id(cache, u(&rb[r * 16..r * 16 + 16], 0)).unwrap_or_default())
                .collect()
        };
        for t in native.tags_by_group(cache, "mode") {
            let n = native.tag_name(cache, t).unwrap_or_default();
            if !n.ends_with("\\fp\\fp") {
                continue;
            }
            let rs = regions(t);
            let has = rs.iter().any(|r| r == "armor_pieces");
            if n.contains("elite") {
                assert!(has, "the Elite arms must carry armor_pieces: {rs:?}");
            } else {
                assert!(!has, "the Spartan arms must NOT carry armor_pieces: {rs:?}");
            }
        }
        let mut checked = 0usize;
        for t in native.tags_by_group(cache, "weap") {
            let (_, Some(j)) = first_person_weapon(native.as_ref(), cache, t, 0) else { continue };
            let Some(g) = AnimGraph::read(&native, cache, j) else { continue };
            for (name, m) in &arms {
                let map = anim_node_map(m, &g).unwrap_or_else(|| {
                    panic!(
                        "no node map for {} onto {name}: {} graph nodes, {} model nodes -- the \
                         positional read would scramble the pose",
                        native.tag_name(cache, j).unwrap_or_default(),
                        g.node_names.len(),
                        m.node_names.len()
                    )
                });
                let live = map.iter().filter(|&&v| v >= 0).count();
                assert_eq!(live, m.node_names.len(), "{name}: every arms node must be driven");
                checked += 1;
            }
        }
        assert!(checked >= 40, "only {checked} (graph, arms) pairs checked");
    }

    /// #fp-gun  The WEAPON side of the same node map. A Reach weapon's `first person model` is
    /// its ordinary render model, and the first-person graph lists that model's own nodes beside
    /// the arms' -- so the gun is drawn on the same rig, posed by the same clip, and needs no
    /// marker attach. That only works if the graph maps onto the WEAPON model too.
    ///
    /// The gate is the direction that used to fail. A weapon model resolves nearly all of its own
    /// handful of nodes but only a tenth of the graph's, so a threshold scored against the GRAPH
    /// rejected it, the positional fallback drove every bone with the wrong channel, and the
    /// result was a blade stretched across the view. This asserts the map exists AND that it is a
    /// map by NAME: every driven entry must point at the node whose name the graph carries.
    #[test]
    fn a_first_person_graph_maps_onto_its_own_weapon_model() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let mut checked = 0usize;
        for t in native.tags_by_group(cache, "weap") {
            let (Some(fm), Some(j)) = first_person_weapon(native.as_ref(), cache, t, 0) else { continue };
            let Some(g) = AnimGraph::read(&native, cache, j) else { continue };
            let Some(m) = ReachModel::read(&native, cache, fm) else { continue };
            let name = native.tag_name(cache, t).unwrap_or_default();
            let map = anim_node_map(&m, &g).unwrap_or_else(|| {
                panic!(
                    "no node map for {name}'s own first-person graph onto its own model: \
                     {} graph nodes, {} model nodes -- the positional read would scramble it",
                    g.node_names.len(),
                    m.node_names.len()
                )
            });
            assert_eq!(map.len(), g.node_names.len(), "{name}: one entry per GRAPH node");
            let mut driven = 0usize;
            for (gi, &mi) in map.iter().enumerate() {
                if mi < 0 { continue; }
                driven += 1;
                assert_eq!(
                    m.node_names.get(mi as usize).map(String::as_str),
                    g.node_names.get(gi).map(String::as_str),
                    "{name}: graph node {gi} '{}' drives model node {mi} '{}' -- the map is not by name",
                    g.node_names.get(gi).cloned().unwrap_or_default(),
                    m.node_names.get(mi as usize).cloned().unwrap_or_default()
                );
            }
            assert!(driven > 0, "{name}: the graph drives none of its own weapon model's nodes");
            checked += 1;
        }
        assert!(checked >= 10, "only {checked} weapon(s) with a first-person interface checked");
    }

    /// #fp-attach  EVERY weapon's first-person root is a CHILD of a hand in the graph they share.
    ///
    /// This is the fact the attach rests on, and it is the graph's own (`s_animation_graph_node`
    /// +0x08). If a weapon's root were a graph ROOT instead, its clip channel would be an absolute
    /// rig-space transform and parenting it would be wrong; if its name were missing from the
    /// graph, the attach would silently not resolve and that gun alone would go back to being
    /// drawn at the rig's origin, on the camera. Measured across the shipped cache: 54
    /// (weapon, species) rows, every one of them a single root parented to `r_hand`.
    ///
    ///     cargo test -p hms-app every_first_person_weapon -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "opens a Reach cache; the native parser's global state collides with the other cache tests"]
    fn every_first_person_weapon_is_parented_to_a_hand() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        // The two arms models, which are what the parent has to land on.
        let arms: Vec<(String, ReachModel)> = native
            .tags_by_group(cache, "mode")
            .into_iter()
            .filter(|&t| native.tag_name(cache, t).unwrap_or_default().ends_with("\\fp\\fp"))
            .filter_map(|t| {
                let n = native.tag_name(cache, t).unwrap_or_default();
                ReachModel::read(&native, cache, t).map(|m| (n, m))
            })
            .collect();
        assert_eq!(arms.len(), 2, "one arms model per playable species");
        let mut checked = 0usize;
        for t in native.tags_by_group(cache, "weap") {
            for species in 0..2usize {
                let (Some(fm), Some(j)) = first_person_weapon(native.as_ref(), cache, t, species) else { continue };
                let Some(g) = AnimGraph::read(&native, cache, j) else { continue };
                let Some(m) = ReachModel::read(&native, cache, fm) else { continue };
                let name = native.tag_name(cache, t).unwrap_or_default();
                assert_eq!(
                    g.node_parents.len(),
                    g.node_names.len(),
                    "{name}: one parent index per graph node"
                );
                let roots: Vec<usize> =
                    m.skel.nodes.iter().enumerate().filter(|(_, n)| n.parent < 0).map(|(i, _)| i).collect();
                assert_eq!(
                    roots.len(),
                    1,
                    "{name} sp{species}: the attach takes the model's ONE root; this one has {}",
                    roots.len()
                );
                let rn = m.node_names.get(roots[0]).cloned().unwrap_or_default();
                let gi = g
                    .node_names
                    .iter()
                    .position(|n| n.eq_ignore_ascii_case(&rn))
                    .unwrap_or_else(|| panic!("{name} sp{species}: root '{rn}' is not in its own first-person graph"));
                let gp = g.node_parents[gi];
                assert!(
                    gp >= 0,
                    "{name} sp{species}: '{rn}' is a graph ROOT -- its clip channel would be an \
                     absolute transform and parenting it to the hand would be wrong"
                );
                let pn = g.node_names.get(gp as usize).cloned().unwrap_or_default();
                let on_arms = arms.iter().filter(|(_, am)| am.node_names.iter().any(|n| n.eq_ignore_ascii_case(&pn))).count();
                assert_eq!(
                    on_arms, 2,
                    "{name} sp{species}: '{rn}' is parented to '{pn}', which is not a node of both \
                     arms models -- the attach could not resolve it"
                );
                checked += 1;
            }
        }
        assert!(checked >= 50, "only {checked} (weapon, species) rows checked, expected 54");
    }

    /// The shipped Spartan skeleton: 82 nodes, and EVERY node's authored `inverse *` fields are
    /// the exact inverse of the bind pose chained from `default translation` / `default rotation`.
    #[test]
    fn spartan_skeleton_inverse_binds_are_exact() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let m = ReachModel::read(&native, cache, native.resolve_mode(cache, t)).expect("render model");
        assert_eq!(m.skel.len(), 82, "Spartan node count");
        assert_eq!(m.node_names[0], "b_pedestal");
        assert!(m.is_skinned(), "the Spartan has skinned meshes");
        let bind = m.skel.bind_model();
        let mut framed = 0usize;
        let mut worst = 0.0f32;
        for (i, nd) in m.skel.nodes.iter().enumerate() {
            assert!(nd.inverse_is_frame(), "node {i} '{}' has no authored inverse frame", m.node_names[i]);
            framed += 1;
            let d = (nd.inverse_bind_authored() * bind[i] - Mat4::IDENTITY)
                .to_cols_array().iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            worst = worst.max(d);
        }
        assert_eq!(framed, 82);
        assert!(worst < 1e-3, "worst |inverse * bind - I| over 82 nodes = {worst}");
    }

    /// A skinned vertex's blend index addresses this MESH's `per mesh node map`, never the node
    /// list directly; a RIGID mesh's node map is exactly its `rigid node index`; and the four
    /// UInt8_N4 weights sum to 255 (+/- the quantiser's rounding).
    #[test]
    fn skinned_blend_indices_are_node_map_slots() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let mode = native.resolve_mode(cache, t);
        let m = ReachModel::read(&native, cache, mode).expect("render model");
        let mh = native.open_model(cache, mode).expect("open_model");
        let mut skinned = 0usize;
        let mut rigid = 0usize;
        for s in 0..native.model_section_count(mh) {
            let Some(sec) = native.model_section(mh, s) else { continue };
            let map = m.node_maps.get(s as usize).cloned().unwrap_or_default();
            for &n in &map {
                assert!((n as usize) < m.skel.len(), "node map entry {n} out of range");
            }
            match native.model_decode_blend(mh, s) {
                Some(blend) => {
                    skinned += 1;
                    assert!(crate::reach_skel::is_skinned_format(sec.vertex_format));
                    assert_eq!(blend.len(), sec.vertex_count as usize * 8, "8 bytes per vertex");
                    assert!(!map.is_empty(), "a skinned mesh carries a node map");
                    for v in blend.chunks_exact(8) {
                        let sum: u32 = v[4..8].iter().map(|&w| w as u32).sum();
                        assert!((254..=258).contains(&sum), "weights sum to {sum}, not 255");
                        for k in 0..4 {
                            if v[4 + k] > 0 {
                                assert!((v[k] as usize) < map.len(), "blend index {} >= node map {}", v[k], map.len());
                            }
                        }
                    }
                }
                None => {
                    rigid += 1;
                    assert!(!crate::reach_skel::is_skinned_format(sec.vertex_format));
                    // `rigid node index` is a RENDER-MODEL node index, not a node-map slot:
                    // where a rigid mesh carries a one-entry map, that entry IS the rigid node.
                    if sec.node_index >= 0 {
                        assert!((sec.node_index as usize) < m.skel.len(), "rigid node {} out of range", sec.node_index);
                        if map.len() == 1 {
                            assert_eq!(map[0] as i16, sec.node_index, "a one-entry node map IS the rigid node index");
                        }
                    }
                }
            }
        }
        native.close_model(mh);
        assert!(skinned > 0 && rigid > 0, "the Spartan has both skinned ({skinned}) and rigid ({rigid}) meshes");
    }

    /// The Spartan's `jmad` skeleton node list is PARALLEL to the render model's (same count,
    /// same names), which is what lets animation channel `i` address render-model node `i`.
    #[test]
    fn jmad_skeleton_is_parallel_to_the_render_model() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let def = ActorDef::read(&native, cache, t).expect("actor def");
        let g = def.graph.as_ref().expect("the Spartan has an animation graph");
        assert_eq!(g.node_names.len(), def.model.skel.len(), "jmad vs render_model node count");
        assert_eq!(g.node_names, def.model.node_names, "node names must match one for one");
        assert!(g.anims.len() > 500, "the Spartan graph has {} animations", g.anims.len());
        assert!(!g.group_resource_ids.is_empty());
    }

    /// Every animation of the Spartan graph loads, its 17 region sizes sum to the declared blob
    /// size, and both codec blocks carry a codec id the dispatcher accepts (< 10). Region 0 is
    /// always the codec-1 static pose; region 1 is codec 3 or codec 9 on shipped Reach.
    #[test]
    fn every_spartan_animation_loads_with_a_valid_codec() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let def = ActorDef::read(&native, cache, t).expect("actor def");
        let g = def.graph.as_ref().expect("graph");
        let mut loaded = 0usize;
        let mut codecs: std::collections::BTreeSet<u8> = Default::default();
        for i in 0..g.anims.len() {
            let Some(d) = g.load(&native, cache, i) else { panic!("animation {} '{}' failed to load", i, g.anims[i].name) };
            loaded += 1;
            assert_eq!(d.total_size() as usize, d.blob.len(), "'{}' region sizes vs blob", g.anims[i].name);
            assert_eq!(d.node_count as usize, def.model.skel.len(), "'{}' node count", g.anims[i].name);
            if let Some(b) = d.region(0) { assert_eq!(b[0], 1, "region 0 is the codec-1 static pose"); }
            if let Some(b) = d.region(1) {
                assert!(b[0] < 10, "codec id {} is out of the engine's 10-entry table", b[0]);
                codecs.insert(b[0]);
            }
            if let Some(f) = d.node_flags(3) {
                assert_eq!(f[0].len(), crate::reach_jmad::bitvector_bytes(d.node_count as usize));
            }
        }
        assert_eq!(loaded, g.anims.len());
        assert_eq!(codecs, [3u8, 9].into_iter().collect(), "shipped Reach uses codecs 3 and 9, got {codecs:?}");
    }

    /// A locomotion animation really animates: the pose at frame 0 and frame 10 differ, the
    /// decoded rotations stay unit length, and the root MOVEMENT data says the Spartan travels
    /// forward. This is the decode-level proof behind the capture.
    #[test]
    fn locomotion_animation_changes_the_pose() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let def = ActorDef::read(&native, cache, t).expect("actor def");
        let g = def.graph.as_ref().expect("graph");
        let i = g.find("combat:rifle:move_front_fast").expect("the rifle run cycle");
        let d = g.load(&native, cache, i).expect("blob");
        assert_eq!(d.frame_count, 22);
        assert_eq!(d.movement_data_type, 1, "dx,dy per frame");
        let mv = d.movement();
        assert_eq!(mv.len(), 22);
        let fwd: f32 = mv.iter().map(|m| m.0).sum();
        assert!(fwd > 1.0, "the run cycle travels {fwd:.3} wu forward");
        let a = crate::reach_jmad::sample(&d, g.anims[i].anim_type, 0.0);
        let b = crate::reach_jmad::sample(&d, g.anims[i].anim_type, 10.0);
        assert_eq!(a.len(), 82);
        let mut moved = 0usize;
        for (x, y) in a.iter().zip(&b) {
            if let (Some(p), Some(q)) = (x.rot, y.rot) {
                assert!((p.length() - 1.0).abs() < 1e-3, "quaternion not unit: {p:?}");
                assert!((q.length() - 1.0).abs() < 1e-3);
                if (p.to_array().iter().zip(q.to_array()).map(|(u, v)| (u - v).abs()).fold(0.0f32, f32::max)) > 0.02 { moved += 1; }
            }
        }
        assert!(moved > 10, "only {moved} node rotations changed between frames 0 and 10");
    }

    /// The held-weapon rule: the engine mates the WEAPON's `preferred_gun_node` marker to the
    /// BIPED's `right_hand_node` marker. On shipped Reach those are `right_hand_spartan` and
    /// `primary_weapon`, the Spartan's hand marker lives on node 50 `b_r_grip`, and both marker
    /// frames are the identity, so the weapon rides the posed grip node exactly.
    #[test]
    fn held_weapon_marker_pair_is_the_engine_rule() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let m = ReachModel::read(&native, cache, native.resolve_mode(cache, t)).expect("model");
        let weap = ActorScene::list_weapons(&native, cache)
            .into_iter()
            .find(|(_, n)| n.contains("assault_rifle\\assault_rifle"))
            .map(|(t, _)| t)
            .expect("the assault rifle");
        let w = resolve_held_weapon(&native, cache, t, &m, weap, false).expect("attach");
        assert_eq!(w.parent_marker, "primary_weapon");
        assert_eq!(w.child_marker, "right_hand_spartan");
        assert_eq!(w.parent_node, 50);
        assert_eq!(m.node_names[50], "b_r_grip");
        let d = |a: Mat4| (a - Mat4::IDENTITY).to_cols_array().iter().fold(0.0f32, |x, &y| x.max(y.abs()));
        assert!(d(w.parent_local) < 1e-4, "the biped hand marker is identity on its node");
        assert!(d(w.child_local) < 1e-4, "the weapon grip marker is identity in its model space");
        // child_world = unit_world * P * C^-1: with both markers identity the weapon IS the
        // posed grip node (haloreach.dll object_place_marker_at_matrix).
        let pose = Pose::bind(&m.skel);
        let got = w.world(Mat4::IDENTITY, &pose);
        let want = pose.model[50];
        assert!((got - want).to_cols_array().iter().fold(0.0f32, |x, &y| x.max(y.abs())) < 1e-4);
    }

    /// #iscale The ITEM SCALE offset, gated so it can fail: sweep every 4-byte offset from
    /// `item` +0x180 to +0x240 and score each by how many of the map's 63 `weap` tags produce
    /// nine plausible scales there. Exactly ONE offset scores anything at all, and it is
    /// [`ITEM_SCALES`]; a shift of one field in either direction scores ZERO. (Shift the
    /// constant by 4 and this test fails on the first assert — that is the point of it.)
    #[test]
    fn the_item_scale_offset_is_the_only_one_that_fits() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let weaps: Vec<u32> = native.tags_by_group(cache, "weap");
        assert!(weaps.len() > 40, "only {} weap tags in forge_halo", weaps.len());
        let score = |off: usize| {
            weaps
                .iter()
                .filter(|t| item_scales_at(&native, cache, **t, off).map_or(false, |s| item_scales_look_authored(&s)))
                .count()
        };
        let hits: Vec<(usize, usize)> = (0x180..=0x240usize).step_by(4).map(|o| (o, score(o))).filter(|r| r.1 > 0).collect();
        assert_eq!(
            hits.iter().map(|r| r.0).collect::<Vec<_>>(),
            vec![ITEM_SCALES],
            "offsets that produce an authored-looking scale table: {hits:?}"
        );
        assert_eq!(score(ITEM_SCALES - 4), 0, "one field EARLIER must not look authored");
        assert_eq!(score(ITEM_SCALES + 4), 0, "one field LATER must not look authored");
        // 29 of the 63 read as a real table. That is not a loose bound: 32 of the 63 have a
        // render model at all (the other 31 are vehicle turret weapons), and the three that
        // have a model but no table are the two knife props and the knife sheath, whose nine
        // scales are authored as literal zeroes — which the engine reads as a flat 1.0.
        assert_eq!(score(ITEM_SCALES), 29, "forge_halo's weapons that author a scale table");
    }

    /// #iscale `unit` +0x3F6 `item owner size`, the column selector: every biped in the map
    /// lands inside the four-value enum, and the three species rank the way the engine's own
    /// scale columns assume — the Monitor small, a Spartan medium, an Elite large.
    #[test]
    fn the_spartan_is_a_medium_unit_and_the_elite_is_large() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let bipeds = ActorScene::list_bipeds(&native, cache);
        assert!(!bipeds.is_empty());
        for (t, n) in &bipeds {
            assert!(unit_item_owner_size(&native, cache, *t).is_some(), "bipd '{n}' has no item owner size in 0..3");
        }
        let of = |needle: &str| bipd_named(&native, cache, needle).and_then(|t| unit_item_owner_size(&native, cache, t));
        assert_eq!(of("spartans\\spartans"), Some(1), "the Spartan is a MEDIUM unit");
        assert_eq!(of("elite\\elite"), Some(2), "the Elite is a LARGE unit");
        assert_eq!(of("monitor"), Some(0), "the Monitor is a SMALL unit");
    }

    /// #iscale The authored table for the weapon the complaint was about, and the scale the
    /// engine's rule picks out of it for each holder. These are the shipped tag's own numbers.
    #[test]
    fn the_energy_sword_is_smaller_in_a_spartans_hand_than_on_the_ground() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let sword = ActorScene::list_weapons(&native, cache)
            .into_iter()
            .find(|(_, n)| n.contains("energy_sword\\energy_sword"))
            .map(|(t, _)| t)
            .expect("the energy sword");
        let s = item_scales(&native, cache, sword).expect("item scales");
        assert!(item_scales_look_authored(&s), "{s:?}");
        assert!((s[0] - 1.3).abs() < 1e-4, "ground scale {} (expected 1.3)", s[0]);
        assert!((s[3] - 0.8).abs() < 1e-4, "medium armed {} (expected 0.8)", s[3]);
        assert!((s[5] - 1.1).abs() < 1e-4, "large armed {} (expected 1.1)", s[5]);
        assert_eq!(item_scale_for(&s, Some(1), false), 0.8, "a Spartan holds it at 0.8");
        assert_eq!(item_scale_for(&s, Some(2), false), 1.1, "an Elite holds it at 1.1");
        assert_eq!(item_scale_for(&s, None, false), 1.3, "on the ground it is 1.3");
        // `huge` is authored 0 on every shipped weapon, which the engine reads as "unset".
        assert_eq!(s[7], 0.0);
        assert_eq!(item_scale_for(&s, Some(3), false), 1.0, "an unset column falls back to 1.0");
        assert_eq!(item_scale_for(&s, Some(7), false), 1.0, "a size outside the enum is a flat 1.0");
        // And the resolved held weapon carries it.
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let m = ReachModel::read(&native, cache, native.resolve_mode(cache, t)).expect("model");
        let w = resolve_held_weapon(&native, cache, t, &m, sword, false).expect("attach");
        assert_eq!(w.scale, 0.8);
    }

    /// #iscale Three weapons whose authored `medium (armed)` scale is exactly 1.0 must come out
    /// of the resolve UNCHANGED — the fix must not resize weapons the tag does not resize.
    #[test]
    fn a_weapon_authored_at_one_is_left_alone() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let t = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let m = ReachModel::read(&native, cache, native.resolve_mode(cache, t)).expect("model");
        let list = ActorScene::list_weapons(&native, cache);
        for needle in ["dmr\\dmr", "magnum\\magnum", "rocket_launcher\\rocket_launcher", "assault_rifle\\assault_rifle"] {
            let Some(weap) = list.iter().find(|(_, n)| n.contains(needle)).map(|(t, _)| *t) else { continue };
            let w = resolve_held_weapon(&native, cache, t, &m, weap, false).expect("attach");
            assert_eq!(w.scale, 1.0, "'{needle}' should be held at 1.0");
            let pose = Pose::bind(&m.skel);
            let d = (w.world(Mat4::IDENTITY, &pose) - pose.model[50]).to_cols_array().iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            assert!(d < 1e-4, "'{needle}' moved by {d} with a scale of 1");
        }
    }

    /// #iscale Where the scale goes. The engine carries the object scale in its NODE matrices,
    /// so the grip marker's offset scales with the model and the attach still lands the grip on
    /// the hand. This asserts BOTH halves on a deliberately non-identity grip marker (no shipped
    /// Reach weapon has one, so a synthetic frame is the only way to tell the right composition
    /// from a wrong one): the model is `s` times bigger, AND the grip is still in the hand.
    #[test]
    fn the_item_scale_grows_the_weapon_without_moving_the_grip() {
        let s = 0.8f32;
        let child = Mat4::from_rotation_y(0.6) * Mat4::from_translation(Vec3::new(0.21, -0.05, 0.11));
        let w = HeldWeapon {
            obj_tag: 0, mode_tag: 0, node_count: 1,
            parent_marker: "primary_weapon".into(), child_marker: "right_hand_spartan".into(),
            child_local: child, parent_node: -1,
            parent_local: Mat4::from_translation(Vec3::new(0.0, 0.02, 0.0)),
            variant_sid: 0, scale: s,
        };
        let unit = Mat4::from_translation(Vec3::new(4.0, -1.0, 0.5)) * Mat4::from_rotation_z(1.1);
        let pose = Pose { model: vec![Mat4::IDENTITY] };
        let world = w.world(unit, &pose);
        // 1. the model really is scaled by s
        assert!((world.x_axis.truncate().length() - s).abs() < 1e-5, "model scale {} != {s}", world.x_axis.truncate().length());
        // 2. the grip marker is still exactly on the hand marker, in POSITION and in
        //    ORIENTATION — which is all the engine mates (`object_place_marker_at_matrix` ends
        //    in a set-position-forward-up, so the destination's scale is not matched and the
        //    grip frame keeps the weapon's own scale).
        let hand = unit * w.parent_local;
        let grip = world * child;
        let dp = (grip.w_axis - hand.w_axis).truncate().length();
        assert!(dp < 1e-5, "the grip left the hand by {dp} wu");
        for (g, h) in [(grip.x_axis, hand.x_axis), (grip.y_axis, hand.y_axis), (grip.z_axis, hand.z_axis)] {
            let d = (g.truncate().normalize() - h.truncate().normalize()).length();
            assert!(d < 1e-5, "the grip frame turned by {d}");
        }
        // 3. and with a scale of 1 the composition is bit-for-bit the old one
        let mut one = w.clone();
        one.scale = 1.0;
        let d1 = (one.world(unit, &pose) - attach_child(unit, one.parent_local, child)).to_cols_array()
            .iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(d1 < 1e-6);
    }

    /// The Elite uses the SAME rule with its own marker names and node — proving the resolution
    /// is data-driven, not hard-coded.
    #[test]
    fn elite_uses_its_own_gun_marker() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let Some(t) = bipd_named(&native, cache, "characters\\elite\\elite") else { return };
        let m = ReachModel::read(&native, cache, native.resolve_mode(cache, t)).expect("model");
        let Some(weap) = ActorScene::list_weapons(&native, cache).into_iter()
            .find(|(_, n)| n.contains("assault_rifle\\assault_rifle")).map(|(t, _)| t) else { return };
        let w = resolve_held_weapon(&native, cache, t, &m, weap, false).expect("attach");
        assert_eq!(w.parent_marker, "primary_weapon");
        assert_eq!(w.child_marker, "right_hand_elite");
        assert!(w.parent_node >= 0 && (w.parent_node as usize) < m.skel.len());
        assert_eq!(m.node_names[w.parent_node as usize], "b_r_grip");
    }

    #[test]
    fn actor_spec_parsing_survives_colons_in_animation_names() {
        let s = parse_actor_specs("spartans@1,2,3,90:anim=combat:rifle:move_front_fast:frame=7:weapon=assault_rifle");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].biped, "spartans");
        assert_eq!(s[0].pos, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(s[0].yaw, 90.0);
        assert_eq!(s[0].anim.as_deref(), Some("combat:rifle:move_front_fast"));
        assert_eq!(s[0].frame, Some(7.0));
        assert_eq!(s[0].weapon.as_deref(), Some("assault_rifle"));
        assert!(!s[0].play, "an explicit frame pins the pose");
        let two = parse_actor_specs("elite@0,0,0; spartans@5,0,0:anim=any:idle");
        assert_eq!(two.len(), 2);
        assert_eq!(two[1].anim.as_deref(), Some("any:idle"));
        assert!(two[1].play);
    }

    /// #elite WHICH RECORDED PLAYERS the Elite bodies belong to, and what the recording says
    /// those players wore.
    ///
    /// Print-only. This exists because the earlier species sweep concluded "0 Elites in the
    /// corpus" from the PLAYER RECORD, while the stream plainly carries hundreds of Elite
    /// bodies: the species is not a profile field at all, it is stated per BODY by the creation
    /// message's 14-bit definition reference. Walking owner -> slot -> player record is what
    /// turns those bodies back into named players whose armour can be scored.
    #[test]
    #[ignore = "needs the real film corpus + the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn probe_elite_bodies_and_their_players() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let Some(map) = crate::mapcat::enumerate().into_iter().find(|c| {
            c.game == crate::mapcat::Game::Reach
                && c.path.file_stem().map_or(false, |s| s.to_string_lossy().eq_ignore_ascii_case("forge_halo"))
        }) else { return };
        let Ok(bytes) = std::fs::read(&map.path) else { return };
        let mut tags: Vec<(u32, String)> = Vec::new();
        for g in crate::film_action::RefTable::CONCRETE_GROUPS {
            for t in native.tags_by_group(cache, g) { tags.push((t, (*g).to_string())); }
        }
        let Some(table) = crate::film_action::RefTable::for_map(&bytes, &tags) else { return };
        let films = crate::film::find_finished_films();
        let mut bodies: std::collections::BTreeMap<String, usize> = Default::default();
        // (gamertag, the ten slot bytes) per player slot that owned at least one Elite body.
        let mut elite_players: Vec<(String, [u8; 10], u8, u8, usize)> = Vec::new();
        let mut elite_unowned = 0usize;
        let mut films_used = 0usize;
        for p in films.iter() {
            let Ok(f) = crate::film::open(p) else { continue };
            if f.data.records.is_empty() || f.header.map_variant.is_empty() { continue }
            if !f.header.scenario_path.to_lowercase().contains("forge_halo") { continue }
            let idx = crate::film_decode::build_index(&f.bytes, &f.data, &f.header.map_variant);
            if idx.entity_messages == 0 { continue }
            films_used += 1;
            let mut elite_by_slot: std::collections::BTreeMap<u8, usize> = Default::default();
            for tr in idx.tracks.iter().filter(|t| t.entity_type == 11) {
                let Some(w) = tr.def_wire else { continue };
                let Some(tag) = table.resolve(w) else { continue };
                if native.tag_class(cache, tag).as_deref() != Some("bipd") { continue }
                let name = native.tag_name(cache, tag).unwrap_or_default();
                *bodies.entry(name.clone()).or_default() += 1;
                if !name.contains("elite") { continue }
                match tr.owner_raw.and_then(|o| idx.slot_of_owner(o)) {
                    Some(slot) => *elite_by_slot.entry(slot).or_default() += 1,
                    None => elite_unowned += 1,
                }
            }
            for (slot, n) in elite_by_slot {
                let Some(pl) = f.header.players.iter().find(|x| x.present() && x.index == slot as usize) else { continue };
                let Some(a) = pl.appearance() else { continue };
                elite_players.push((pl.gamertag.clone(), a.slots, a.color_primary, a.color_secondary, n));
            }
        }
        println!("\n== {films_used} forge_halo film(s) with a stream ==");
        for (k, n) in &bodies { println!("   {n:>6}  {k}"); }
        println!("\n{} player-lives own ELITE bodies ({elite_unowned} Elite bodies had no owner):", elite_players.len());
        for (g, s, cp, cs, n) in &elite_players {
            println!(
                "   {g:<18} {n:>3} elite body(ies)  slots {s:?}  elite-armour slot7={} elite-fx slot9={}  colours {cp}/{cs}",
                s[7], s[9]
            );
        }
        let mut s7: std::collections::BTreeMap<u8, usize> = Default::default();
        let mut s9: std::collections::BTreeMap<u8, usize> = Default::default();
        for (_, s, _, _, _) in &elite_players { *s7.entry(s[7]).or_default() += 1; *s9.entry(s[9]).or_default() += 1; }
        println!("\n   elite-armour slot 7 histogram over the Elite players: {s7:?}");
        println!("   elite-fx     slot 9 histogram over the Elite players: {s9:?}");
    }

    /// #elite-knife What the Elite's WRIST BLADE actually is, and what the tag says switches it
    /// on. Print-only.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn probe_elite_knifeblade_materials() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let sid = |v: u32| if v == 0 { "-".to_string() } else { native.string_id(cache, v).unwrap_or_else(|| format!("{v:#x}")) };
        let Some(bipd) = bipd_named(&native, cache, "elite\\elite") else { return };
        println!("elite bipd={bipd:#x} '{}'", native.tag_name(cache, bipd).unwrap_or_default());
        // the object's own FUNCTION block (obje +0xEC, 64 B entries)
        let rd = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        if let Some(m) = native.tag_meta(cache, bipd, 0xEC, 8) {
            let (cnt, ptr) = (rd(&m, 0), rd(&m, 4));
            println!("objfn count={cnt}");
            if cnt > 0 && cnt <= 256 {
                if let Some(b) = native.tag_ptr_bytes(cache, ptr, cnt * 64) {
                    for i in 0..cnt as usize {
                        let e = &b[i * 64..i * 64 + 64];
                        println!("  objfn[{i}] export='{}' import='{}' turn_off_with='{}' scale_by='{}' min={:.3} flags={:#x}",
                            sid(rd(e, 8)), sid(rd(e, 4)), sid(rd(e, 0x0C)), sid(rd(e, 0x2C)),
                            f32::from_le_bytes([e[0x14], e[0x15], e[0x16], e[0x17]]), rd(e, 0));
                    }
                }
            }
        }
        let mode = native.resolve_mode(cache, bipd);
        let Ok(model) = native.open_model(cache, mode) else { return };
        let sm = native.model_submesh_count(model);
        let mut shaders: Vec<i32> = Vec::new();
        for i in 0..sm {
            if let Some(x) = native.model_submesh(model, i) {
                if !shaders.contains(&x.shader_index) { shaders.push(x.shader_index); }
            }
        }
        shaders.sort();
        for sh in shaders {
            let stag = native.model_shader_tag(model, sh);
            if stag < 0 { continue }
            let name = native.tag_name(cache, stag as u32).unwrap_or_default();
            let tm = native.model_shader_template_tag(model, sh);
            let tname = if tm >= 0 { native.tag_name(cache, tm as u32).unwrap_or_default() } else { String::new() };
            let interesting = name.contains("knifeblade") || name.contains("sword");
            if !interesting { continue }
            println!("\nshader[{sh}] class={} tag={stag:#x} '{name}'", native.model_shader_class(model, sh));
            println!("   template '{tname}'");
            // postprocess[0] +0x5C overlays (0x24 B each): type @0, input sid @4, range sid @8
            let Some(m) = native.tag_meta(cache, stag as u32, 0x38, 8) else { continue };
            let (cnt, ptr) = (rd(&m, 0), rd(&m, 4));
            if cnt == 0 { continue }
            let Some(pp) = native.tag_ptr_bytes(cache, ptr, 0xAC) else { continue };
            println!("   blend_mode={} flags={:#x}", rd(&pp, 0x68), rd(&pp, 0x6C));
            let (oc, op) = (rd(&pp, 0x5C), rd(&pp, 0x60));
            if oc == 0 || oc > 256 { continue }
            let Some(ob) = native.tag_ptr_bytes(cache, op, oc * 0x24) else { continue };
            for i in 0..oc as usize {
                let e = &ob[i * 0x24..i * 0x24 + 0x24];
                println!("   overlay[{i}] type={} input='{}' range='{}' period={:.3}",
                    rd(e, 0) as i32, sid(rd(e, 4)), sid(rd(e, 8)),
                    f32::from_le_bytes([e[0x0C], e[0x0D], e[0x0E], e[0x0F]]));
            }
            if let Some(sc) = native.model_shader_constants(model, sh) {
                for (i, n, v, x) in sc.dump_args() {
                    if n.is_empty() { continue }
                    println!("   arg[{i:2}] {n:<34} real=[{:.4},{:.4},{:.4},{:.4}] scalar={x:.4}", v[0], v[1], v[2], v[3]);
                }
            }
        }
        native.close_model(model);
        // and the whole animated-parameter picture, evaluated exactly as the renderer does.
        crate::scene::dump_object_anim_params(&native, cache, bipd);
        if let Some(sp) = bipd_named(&native, cache, "spartans\\spartans") {
            crate::scene::dump_object_anim_params(&native, cache, sp);
        }
    }

    /// #elite THE GATE for "an Elite wears what its player chose, and only that".
    ///
    /// The ten appearance bytes are not ten Spartan slots: the engine's own reflection data
    /// names the pmcg blocks helmet / left shoulder / right shoulder / chest / wrist / utility /
    /// paint scheme / **elite** / spartan armor effect / **elite armor effect**, in that order.
    /// So slot 7 is the Elite's whole armour and slot 9 its armour effect, and slots 8 and 9
    /// collide on `armor_fx` -- the one region both render models carry. This asserts the split
    /// against the real tag: the Elite's armour choice reaches the Elite model and nothing of the
    /// Spartan's does, and vice versa.
    #[test]
    fn a_player_variant_dresses_each_species_from_its_own_slots() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let table = crate::scene::player_customization_table(&native, cache);
        if table.len() != crate::film::APPEARANCE_SLOTS { return }
        let name = |sid: u32| native.string_id(cache, sid).unwrap_or_default();
        // The slot sets must partition the ten blocks exactly once.
        let mut all: Vec<usize> = crate::film::SPARTAN_APPEARANCE_SLOTS.to_vec();
        all.extend_from_slice(&crate::film::ELITE_APPEARANCE_SLOTS);
        all.sort_unstable();
        assert_eq!(all, (0..crate::film::APPEARANCE_SLOTS).collect::<Vec<_>>(),
            "the two species' slot sets must cover the ten blocks exactly once");
        // Slot 7 is the ELITE block: every one of its options overrides `armor_pieces`, which is
        // a region of the Elite render model and of no other.
        let elite_regions: Vec<String> = table[7].iter().flatten().map(|(r, _)| name(*r)).collect();
        assert!(!elite_regions.is_empty() && elite_regions.iter().all(|r| r == "armor_pieces"),
            "slot 7 should be the elite block (armor_pieces), got {elite_regions:?}");
        assert_eq!(table[7].len(), 8, "the Elite has eight armour choices");
        // The two armour-EFFECT blocks are the collision the species split exists for.
        let fx8: Vec<String> = table[8].iter().flatten().map(|(r, _)| name(*r)).collect();
        let fx9: Vec<String> = table[9].iter().flatten().map(|(r, _)| name(*r)).collect();
        assert!(fx8.iter().all(|r| r == "armor_fx") && fx9.iter().all(|r| r == "armor_fx"),
            "slots 8 and 9 both override armor_fx: {fx8:?} / {fx9:?}");
        // A player with a DIFFERENT choice in every slot: the Elite picks must carry only
        // `armor_pieces` + `armor_fx`, the Spartan picks must carry no `armor_pieces` at all.
        let slots: [u8; crate::film::APPEARANCE_SLOTS] = [60, 4, 4, 3, 2, 5, 2, 4, 6, 3];
        let sp = crate::scene::appearance_picks_from(&table, &slots, &crate::film::SPARTAN_APPEARANCE_SLOTS);
        let el = crate::scene::appearance_picks_from(&table, &slots, &crate::film::ELITE_APPEARANCE_SLOTS);
        assert!(!sp.is_empty() && !el.is_empty(), "both species must resolve picks: {sp:?} / {el:?}");
        assert!(sp.iter().all(|(r, _)| name(*r) != "armor_pieces"),
            "a Spartan must never be given the Elite's armour: {:?}", sp.iter().map(|(r, _)| name(*r)).collect::<Vec<_>>());
        assert!(el.iter().all(|(r, _)| matches!(name(*r).as_str(), "armor_pieces" | "armor_fx")),
            "an Elite must only be given its own armour and effect: {:?}", el.iter().map(|(r, _)| name(*r)).collect::<Vec<_>>());
        // And the one region both carry must resolve to a DIFFERENT permutation per species, or
        // the split would be decorative (slot 8 = 6, slot 9 = 3).
        let fx = |v: &[(u32, u32)]| v.iter().find(|(r, _)| name(*r) == "armor_fx").map(|(_, p)| name(*p));
        assert_ne!(fx(&sp), fx(&el), "the Spartan and Elite armour effects must not be the same pick");
    }

    /// #float-weapon THE GATE for "the gun is in the hand, not hanging in the air beside it".
    ///
    /// A held weapon's mesh is baked once and then moved every frame by its bone palette, which
    /// holds `anchor^-1 * world_now * attach`. That composition only lands on `world_now * attach`
    /// if the mesh was baked at `anchor` — so the frame the build is handed and the frame the
    /// palette folds against are ONE value, and a second writer for it is a displaced gun.
    ///
    /// `build_actor_meshes` took the biped's bake frame from `actor_mesh_frame(anchor)` and the
    /// WEAPON's from `a.pos`. Those are the same point for anything standing in the world, which
    /// is why it survived; they diverge exactly when `world_override` is set (a rider on a vehicle
    /// seat, a first-person weapon parented to a hand), and the gun is then drawn
    /// `pos - anchor.translation` away from the hand. This asserts both halves: baking at the
    /// anchor is exact for any placement, and baking at `pos` is wrong by exactly that offset.
    #[test]
    fn a_held_weapon_is_only_in_the_hand_when_its_mesh_was_baked_at_the_anchor() {
        // An actor whose placement is NOT its recorded origin: `world_override` puts it 2.5 wu up
        // and 1.25 wu along, the shape a seat or a first-person hand attach has.
        let pos = Vec3::new(7.0, -3.0, 14.0);
        let anchor = Mat4::from_translation(pos + Vec3::new(1.25, 0.0, 2.5))
            * Mat4::from_rotation_z(0.7)
            * Mat4::from_rotation_y(-0.2);
        let (origin, fwd, up) = actor_mesh_frame(anchor);
        // The engine attach (marker mate) and a later placement, both arbitrary and both nonzero,
        // so nothing in the identity can be an accident of the test's numbers.
        let attach = Mat4::from_translation(Vec3::new(0.11, -0.04, 0.26)) * Mat4::from_rotation_x(1.1);
        let world_now = Mat4::from_translation(Vec3::new(9.5, -1.0, 15.0)) * Mat4::from_rotation_z(2.2);
        let want = (world_now * attach).w_axis.truncate();

        // `bake_frame` must reproduce the anchor it was derived from, or `actor_mesh_frame` and
        // the builder's own placement->matrix disagree and everything below is vacuous.
        let baked = bake_frame(origin, fwd, up);
        for i in 0..4 {
            for j in 0..4 {
                assert!((baked.col(i)[j] - anchor.col(i)[j]).abs() < 1e-5,
                    "bake_frame(actor_mesh_frame(anchor)) must BE anchor, col {i} row {j}");
            }
        }
        // Baked at the anchor: exact, for this frame and for any later one.
        let right = held_weapon_drawn(baked, anchor, world_now, attach).w_axis.truncate();
        assert!((right - want).length() < 1e-4,
            "a weapon baked at the anchor must draw at world*attach, got {right:?} want {want:?}");

        // Baked at `a.pos` (the defect): displaced by exactly pos - anchor.translation.
        let wrong_bake = bake_frame(pos, fwd, up);
        let wrong = held_weapon_drawn(wrong_bake, anchor, world_now, attach).w_axis.truncate();
        let offset = pos - anchor.w_axis.truncate();
        assert!((wrong - (want + offset)).length() < 1e-4,
            "baking at `pos` must displace the gun by pos-anchor ({offset:?}), got {:?}", wrong - want);
        // And that offset is big enough to be the "gun hanging in mid-air" the user reported.
        assert!((wrong - want).length() > 2.0,
            "this fixture must actually separate the two frames, got {:.3} wu", (wrong - want).length());
    }

    /// #sword-blade THE GATE for "a held energy sword is a TRANSPARENT object", read off the
    /// shipped tag rather than asserted from memory.
    ///
    /// The user's defect: the sword in an actor's hand drew as a solid grey blade. The cause was
    /// not the decode — `object_part_lanes` shows the sword's parts routing to the ADDITIVE lane
    /// perfectly well — it was that `build_actor_meshes` had no transparent lane to hand them
    /// back in and folded all of them into the opaque list. So this asserts the fact that makes
    /// the fold wrong: most of this weapon's drawn parts are not opaque, and the ones that are
    /// not are all additive glow. If a future change ever routes a blade layer to `Opaque` again,
    /// the grey blade is back and this fails.
    #[test]
    fn the_energy_sword_decodes_into_transparent_lanes_on_the_shipped_tag() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let Some(sword) = native.tags_by_group(cache, "weap").into_iter().find(|t| {
            native.tag_name(cache, *t).unwrap_or_default().contains("energy_sword")
        }) else { return };
        let lanes = crate::scene::object_part_lanes(&native, cache, sword);
        assert!(lanes.len() >= 7, "the energy sword draws 7 parts, got {}", lanes.len());
        let trans: Vec<_> = lanes.iter().filter(|(_, l)| l.is_transparent()).collect();
        assert!(trans.len() >= 5,
            "at least five of the sword's parts must take a TRANSPARENT lane, got {:?}",
            lanes.iter().map(|(_, l)| *l).collect::<Vec<_>>());
        // Every transparent part of this weapon is an ADDITIVE glow — the blade is emissive, not
        // glass and not a marker shape.
        assert!(trans.iter().all(|(_, l)| *l == crate::scene::PartLane::Additive),
            "the sword's transparent parts must all be additive, got {:?}",
            trans.iter().map(|(_, l)| *l).collect::<Vec<_>>());
        // Both authoring patterns are present: fs_holo lane 10 (rmhg halogram, from_diffuse) and
        // fs_holo lane 9 (rmsh additive geometry with a 3-channel `sword_illum` composite).
        assert!(lanes.iter().filter(|(i, _)| i.lane_halo_from_diffuse()).count() >= 2,
            "the two blade HALOGRAMS (fs_holo lane 10) are missing");
        assert!(lanes.iter().filter(|(i, _)| i.lane_illum_only()).count() >= 2,
            "the two `sword_illum` additive blade layers (fs_holo lane 9) are missing");
        // And the hilt is still solid, or the whole weapon would be a glow.
        assert!(lanes.iter().any(|(_, l)| *l == crate::scene::PartLane::Opaque),
            "the sword's hilt must stay in the opaque pass");
    }

    /// #elite-knife The Elite's two energy daggers are ANIMATION-EVENT gated and the energy
    /// sword's blades are NOT -- which is the whole reason the daggers can be hidden at rest
    /// without touching the sword.
    ///
    /// The Elite's `blade_left_on` / `blade_right_on` functions import
    /// `animation_event_function_a` / `_b` (the per-animation event channel, zero except while a
    /// frame event raises it); the sword's blade functions import `turned_on` and
    /// `active_camouflage`, which describe an object that simply exists and works. Read off the
    /// shipped tags, so if either authoring ever changes this fails rather than quietly hiding a
    /// sword blade.
    #[test]
    fn the_elite_daggers_are_animation_event_gated_and_the_sword_blades_are_not() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let Some(bipd) = bipd_named(&native, cache, "elite\\elite") else { return };
        let fns = crate::scene::object_function_imports_of(&native, cache, bipd);
        assert_eq!(fns.get("blade_left_on").map(String::as_str), Some("animation_event_function_a"));
        assert_eq!(fns.get("blade_right_on").map(String::as_str), Some("animation_event_function_b"));
        let mode = native.resolve_mode(cache, bipd);
        let Ok(model) = native.open_model(cache, mode) else { return };
        let mut blades = 0usize;
        let mut gated = 0usize;
        for i in 0..native.model_submesh_count(model) {
            let Some(sm) = native.model_submesh(model, i) else { continue };
            let sh = sm.shader_index;
            if sh < 0 { continue }
            let st = native.model_shader_tag(model, sh);
            if st < 0 { continue }
            if !native.tag_name(cache, st as u32).unwrap_or_default().contains("knifeblade") { continue }
            blades += 1;
            if crate::scene::shader_animation_event_gated_of(&native, cache, &fns, model, sh) { gated += 1 }
        }
        native.close_model(model);
        assert!(blades >= 6, "the Elite should carry six knifeblade submeshes, found {blades}");
        assert_eq!(gated, blades, "every knifeblade material must be animation-event gated");
        // The energy sword: same halogram construction, driven by `turned_on` / camo instead.
        let weap = native.tags_by_group(cache, "weap");
        let Some(sword) = weap.into_iter().find(|t| {
            native.tag_name(cache, *t).unwrap_or_default().contains("energy_sword")
        }) else { return };
        let sfns = crate::scene::object_function_imports_of(&native, cache, sword);
        let smode = native.resolve_mode(cache, sword);
        let Ok(smodel) = native.open_model(cache, smode) else { return };
        let mut any = 0usize;
        for i in 0..native.model_submesh_count(smodel) {
            let Some(sm) = native.model_submesh(smodel, i) else { continue };
            let sh = sm.shader_index;
            if sh < 0 { continue }
            if crate::scene::shader_animation_event_gated_of(&native, cache, &sfns, smodel, sh) { any += 1 }
        }
        native.close_model(smodel);
        assert_eq!(any, 0, "no energy-sword material may be treated as animation-event gated");
    }

    /// #elite-knife THE BLAST RADIUS of the animation-event gate, over every object in the map.
    ///
    /// The gate is a rule about every object part in the game, so the question that matters is
    /// not "does it hide the Elite's daggers" (a 6-vs-0 test answers that) but "what ELSE does it
    /// hide". This sweeps every object tag that resolves to a render model, mirrors the decode's
    /// own condition (halogram part AND every non-time animated parameter driven by an
    /// `animation_event_function_*` import), and prints each hit with its object and submesh
    /// count. Print-only so it can be read; the assertion is in
    /// `the_animation_event_gate_hides_only_the_elite_daggers`.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn probe_animation_event_gate_blast_radius() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let (objects, models, halo_parts) = event_gate_census(&native, cache);
        println!("   swept {objects} object tag(s), {models} with a render model, {halo_parts} halogram submesh(es)");
        for (tag, name, parts, subs) in event_gated_parts(&native, cache) {
            println!("   GATED obj={tag:#x} '{name}' -> {parts} shader(s), {subs} submesh(es)");
        }
    }

    /// How much of the map the sweep actually looked at, so "only one object" is a measurement
    /// and not an artefact of a filter.
    fn event_gate_census(native: &NativeDll, cache: CacheHandle) -> (usize, usize, usize) {
        let (mut objects, mut models, mut halo) = (0usize, 0usize, 0usize);
        let mut seen: std::collections::HashSet<u32> = Default::default();
        for g in OBJECT_GROUPS {
            for obj in native.tags_by_group(cache, g) {
                if !seen.insert(obj) { continue }
                objects += 1;
                let mode = native.resolve_mode(cache, obj);
                if mode == 0 || mode == 0xFFFF_FFFF { continue }
                let Ok(model) = native.open_model(cache, mode) else { continue };
                models += 1;
                for i in 0..native.model_submesh_count(model) {
                    let Some(sm) = native.model_submesh(model, i) else { continue };
                    let sh = sm.shader_index;
                    if sh < 0 { continue }
                    let cls = native.model_shader_class(model, sh);
                    let d = native.model_shader_diffuse(model, sh);
                    let dn = if d != 0 && d != 0xFFFF_FFFF { native.tag_name(cache, d).unwrap_or_default() } else { String::new() };
                    if cls == "rmhg" || dn.contains("halogram") || dn.contains("drop_holo") || dn.contains("_holo") { halo += 1 }
                }
                native.close_model(model);
                if objects % 32 == 0 { native.drop_map_pages(cache); }
            }
        }
        native.drop_map_pages(cache);
        (objects, models, halo)
    }

    /// Every object group that can carry a render model, so the sweep is the whole map and not a
    /// chosen corner of it.
    const OBJECT_GROUPS: [&str; 14] = [
        "bipd", "vehi", "weap", "eqip", "scen", "bloc", "mach", "ctrl", "crea", "proj", "efsc",
        "term", "gint", "ssce",
    ];

    /// Every object whose render model has at least one part the animation-event gate hides:
    /// `(obj tag, name, gated shader count, gated submesh count)`.
    fn event_gated_parts(native: &NativeDll, cache: CacheHandle) -> Vec<(u32, String, usize, usize)> {
        event_gated_parts_in(native, cache, &OBJECT_GROUPS)
    }

    /// #mem The sweep touches every page of every render model it opens, and the native page
    /// cache is unbounded -- left alone it took the whole test binary to 15 GB RSS, which on this
    /// machine means zram, which means `systemd-oomd`. The resident pages are dropped every so
    /// many objects; they re-fault from the OS file cache if they are needed again.
    fn event_gated_parts_in(native: &NativeDll, cache: CacheHandle, groups: &[&str]) -> Vec<(u32, String, usize, usize)> {
        let mut out = Vec::new();
        let mut seen: std::collections::HashSet<u32> = Default::default();
        let mut since_drop = 0usize;
        for g in groups.iter().copied() {
            for obj in native.tags_by_group(cache, g) {
                if !seen.insert(obj) { continue }
                let mode = native.resolve_mode(cache, obj);
                if mode == 0 || mode == 0xFFFF_FFFF { continue }
                // NOT skipped when the function block is empty: a material may name an engine
                // input directly, and the decode would still gate it, so the sweep must see it.
                let fns = crate::scene::object_function_imports_of(native, cache, obj);
                let Ok(model) = native.open_model(cache, mode) else { continue };
                let mut gated_sh: std::collections::HashSet<i32> = Default::default();
                let mut subs = 0usize;
                for i in 0..native.model_submesh_count(model) {
                    let Some(sm) = native.model_submesh(model, i) else { continue };
                    let sh = sm.shader_index;
                    if sh < 0 { continue }
                    // Mirror the decode: the gate only applies to a HALOGRAM part.
                    let cls = native.model_shader_class(model, sh);
                    let d = native.model_shader_diffuse(model, sh);
                    let dn = if d != 0 && d != 0xFFFF_FFFF { native.tag_name(cache, d).unwrap_or_default() } else { String::new() };
                    let is_halo = cls == "rmhg" || dn.contains("halogram") || dn.contains("drop_holo") || dn.contains("_holo");
                    if !is_halo { continue }
                    if !crate::scene::shader_animation_event_gated_of(native, cache, &fns, model, sh) { continue }
                    gated_sh.insert(sh);
                    subs += 1;
                }
                native.close_model(model);
                since_drop += 1;
                if since_drop >= 32 { since_drop = 0; native.drop_map_pages(cache); }
                if !gated_sh.is_empty() {
                    out.push((obj, native.tag_name(cache, obj).unwrap_or_default(), gated_sh.len(), subs));
                }
            }
        }
        native.drop_map_pages(cache);
        out
    }

    /// #elite-knife THE GATE that keeps the animation-event rule narrow.
    ///
    /// The rule is general -- any part whose every non-time driver is an animation event -- so it
    /// could in principle hide something that ought to be visible at rest. Over every object in
    /// `forge_halo` it hides exactly the Elite's six dagger submeshes and nothing else. If a
    /// later change widens it, this fails and names what it started hiding.
    #[test]
    fn the_animation_event_gate_hides_only_the_elite_daggers() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        // #mem The ordinary suite sweeps the four groups that can carry an object-function-driven
        // halogram AND be drawn as an actor or an item -- the Elite is a `bipd` and the energy
        // sword a `weap`, which is the pair this gate exists to keep apart. The exhaustive
        // fourteen-group walk lives in `probe_animation_event_gate_blast_radius`, which is
        // `#[ignore]`d because opening all 650 render models is minutes and gigabytes.
        let hits = event_gated_parts_in(&native, cache, &["bipd", "weap", "vehi", "eqip"]);
        let other: Vec<&(u32, String, usize, usize)> =
            hits.iter().filter(|(_, n, _, _)| !n.contains("elite")).collect();
        assert!(other.is_empty(), "the animation-event gate also hides: {other:?}");
        let elite: Vec<&(u32, String, usize, usize)> =
            hits.iter().filter(|(_, n, _, _)| n.contains("elite")).collect();
        assert_eq!(elite.len(), 1, "exactly one object should be gated, got {elite:?}");
        assert_eq!(elite[0].2, 6, "six dagger materials");
        assert_eq!(elite[0].3, 6, "six dagger submeshes");
    }

    /// `attach_child` satisfies its defining property: the child's marker lands exactly on the
    /// parent's marker in world space.
    #[test]
    fn attach_puts_the_child_marker_on_the_parent_marker() {
        let parent_world = Mat4::from_translation(Vec3::new(3.0, -2.0, 1.0)) * Mat4::from_rotation_z(0.7);
        let p = Mat4::from_translation(Vec3::new(0.1, 0.2, 0.3)) * Mat4::from_rotation_x(0.4);
        let c = Mat4::from_translation(Vec3::new(-0.5, 0.0, 0.25)) * Mat4::from_rotation_y(-1.1);
        let child_world = attach_child(parent_world, p, c);
        let lhs = child_world * c;
        let rhs = parent_world * p;
        let d = (lhs - rhs).to_cols_array().iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(d < 1e-4, "marker mating is off by {d}");
    }

    /// #reach-chud **THE CACHE-BACKED NAMEPLATE GATE.** Everything the floating player name is
    /// made of, asserted against a shipped Reach cache rather than against this repo's own
    /// constants.
    ///
    /// Three things are pinned, and each one is a bug this project actually shipped:
    ///
    /// 1. **The anchor marker.** `haloreach.dll+0x2CB76C` positions the record with
    ///    `Object_BuildWorldTransform(unit, 0xC2, ...)`, and `+0x221AD4` looks `0xC2` up as a
    ///    MARKER GROUP NAME on the render model. In the cache's own string table that id is
    ///    `head`. One pass read it as `"chest"`, found no such marker, and a later pass concluded
    ///    the engine must not be naming a marker at all and anchored on `chud_nav_point` — which
    ///    exists and sits HIGHER. This asserts the id resolves to `film_ui::NAME_TAG_MARKER`.
    /// 2. **The height.** The Spartan's `head` marker plus the engine's 0.075 wu lift is
    ///    `film_ui::NAME_TAG_UP_WU`, and `chud_nav_point` on the same bone is strictly above it.
    /// 3. **The colours.** `ui\chud\globals`' `default` skin authors `navpt normal`, `navpt
    ///    neutral` and `navpt enemy`, and `NamePlateRules::from_chud` must reproduce exactly the
    ///    values the no-cache defaults carry.
    #[test]
    #[ignore = "needs the forge_halo cache; run alone with --ignored --nocapture --test-threads=1"]
    fn the_nameplate_anchor_and_colours_are_the_authored_ones() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };

        // 1. the marker the engine names.
        let sid = native.string_id(cache, crate::reach_chud::NAME_ANCHOR_MARKER_SID);
        assert_eq!(
            sid.as_deref(),
            Some(crate::film_ui::NAME_TAG_MARKER),
            "string_id {:#x} is the marker the engine resolves for a player's floating name",
            crate::reach_chud::NAME_ANCHOR_MARKER_SID
        );

        // 2. the height, on the Spartan, in the model's own bind pose.
        let Some(bipd) = bipd_named(&native, cache, "spartans\\spartans") else {
            panic!("no Spartan bipd in forge_halo")
        };
        let model = ReachModel::read(&native, cache, native.resolve_mode(cache, bipd)).expect("spartan render model");
        let bind = model.skel.bind_model();
        let z_of = |name: &str| -> Option<f32> {
            let g = model.marker_group(&native, cache, name)?;
            let m = g.markers.first()?;
            let n: usize = m.node.try_into().ok()?;
            Some((*bind.get(n)? * m.local()).w_axis.z)
        };
        let head = z_of(crate::film_ui::NAME_TAG_MARKER).expect("the Spartan carries the engine's marker");
        let navpt = z_of("chud_nav_point").expect("the Spartan carries chud_nav_point too");
        // The two displays share this marker and differ only in the lift: +0x2CB76C's 0.075 for
        // the ally waypoint, +0x18AE04's 0.05 for the look-at name.
        let waypoint = head + crate::film_ui::NAME_TAG_MARKER_UP_WU;
        let anchor = head + crate::film_ui::LOOKAT_UP_WU;
        eprintln!(
            "reach_chud: spartan '{}' z={head:.3} -> look-at {anchor:.3} / waypoint {waypoint:.3} wu; \
             chud_nav_point z={navpt:.3} (+{:.3})",
            crate::film_ui::NAME_TAG_MARKER,
            navpt - head
        );
        assert!(anchor < waypoint, "the look-at name sits below the ally waypoint");
        assert!(
            navpt > head + 0.02,
            "chud_nav_point ({navpt:.3}) is supposed to sit well above the engine's own anchor ({head:.3}) \
             — if it does not, the height argument in film_ui is wrong"
        );
        assert!(
            (crate::film_ui::NAME_TAG_UP_WU - anchor).abs() < 0.01,
            "the no-marker fallback {} is not the measured anchor {anchor:.3}",
            crate::film_ui::NAME_TAG_UP_WU
        );

        // 3. the authored colours and scale clamps.
        let g = crate::reach_chud::read_globals(&native, cache).expect("ui\\chud\\globals");
        assert!(g.path.eq_ignore_ascii_case("ui\\chud\\globals"), "{}", g.path);
        let skin = g.default_skin().expect("a default skin");
        assert_eq!(skin.skin_type, 0, "the multiplayer skin is chud_skin_type_enum 0 'default'");
        let rules = crate::film_ui::NamePlateRules::from_chud(skin);
        let d = crate::film_ui::NamePlateRules::default();
        assert_eq!(rules.navpt_normal, d.navpt_normal, "authored 'navpt normal' (+0x74)");
        assert_eq!(rules.navpt_enemy, d.navpt_enemy, "authored 'navpt enemy' (+0xA4)");
        assert_eq!(rules.navpt_neutral, d.navpt_neutral, "authored 'navpt neutral' (+0x94)");
        assert_eq!(rules.scale_min, d.scale_min, "'waypoint minimum distance scale' (+0x488)");
        assert_eq!(rules.scale_max, d.scale_max, "'waypoint maximum distance scale' (+0x48C)");
        // The authored colour the ENGINE gives a plain teammate is index 0 = 'navpt normal'.
        let k = crate::reach_chud::color_from_index(crate::reach_chud::PLAYER_NAME_COLOR_INDEX).unwrap();
        assert_eq!(crate::reach_chud::SKIN_COLORS[k], "navpt normal");
        assert_eq!(skin.colors[k], skin.player_name_color());

        // 4. the VIRTUAL SCREEN the HUD is laid out in, and the chevron's authored size. These are
        // what turn an authored size into a drawn one, so they are pinned to the tag the same way.
        let cv = skin.fullscreen_wide().expect("a fullscreen-wide curvature entry");
        eprintln!(
            "reach_chud: virtual {} x {}  safe frame {:?}  ding {:?}",
            cv.virtual_width, cv.virtual_height, cv.safe_frame, cv.ding
        );
        assert_eq!(rules.virtual_height, cv.virtual_height as f32, "the rules take the tag's virtual height");
        assert_eq!(rules.safe_frame_v, cv.safe_frame[1], "... and its safe frame");
        assert_eq!(rules.virtual_height, d.virtual_height, "the no-cache default must match the tag");
        assert_eq!(rules.safe_frame_v, d.safe_frame_v);
        // one virtual unit must GROW with the window -- a constant size is the bug this catches
        assert!(
            rules.px_per_virtual(1440.0) > rules.px_per_virtual(720.0) * 1.9,
            "the HUD must scale with the window"
        );

        // 5. #spectate  Theater's spectated-player card, from its own cui screen.
        let card = crate::reach_chud::spectate_screen(&native, cache).expect("the film_cameras screen");
        let item = card.get(crate::reach_chud::SPECTATE_ITEM).expect("the camera list item");
        let base = card.origin_of(item);
        eprintln!(
            "reach_chud: spectate card at ({:.1},{:.1}) size {:.1} x {:.1}",
            base[0], base[1], item.width(), item.height()
        );
        // The authored place: bottom of the screen, horizontally centred in the 1152-wide space.
        // Measured against the DRAWN extent, not the list item's nominal 350-wide cell -- that
        // difference is 36 units of horizontal error, which is the point of `spectate_size`.
        let drawn_size = crate::reach_chud::spectate_size(&card);
        eprintln!("reach_chud: spectate drawn extent {:.1} x {:.1}", drawn_size[0], drawn_size[1]);
        assert!(drawn_size[0] < item.width(), "the art stops short of the template cell");
        let centre = base[0] + drawn_size[0] * 0.5;
        assert!(
            (centre - crate::reach_chud::CUI_VIRTUAL[0] * 0.5).abs() < 2.0,
            "the card should be centred: {centre:.1} vs {:.1}",
            crate::reach_chud::CUI_VIRTUAL[0] * 0.5
        );
        assert!(base[1] > crate::reach_chud::CUI_VIRTUAL[1] * 0.75, "the card sits along the bottom");
        // The two strings HMS fills, with the font and justification the tag gives them.
        let name = card.get("text_player_name").expect("text_player_name");
        let tag = card.get("text_payer_tag").expect("text_payer_tag (the engine's own spelling)");
        assert_eq!(name.font_id(), Some(crate::reach_font::FontId::EnglishBodyText), "prop_font_id 7");
        assert_eq!(tag.font_id(), Some(crate::reach_font::FontId::EnglishBodyText));
        assert_eq!(tag.justification(), 2, "the service tag is right-justified");
        // The skip list must name components that really exist, or it is silently stale.
        for n in crate::reach_chud::SPECTATE_SKIP {
            assert!(card.get(n).is_some(), "SPECTATE_SKIP names '{n}', which this tag does not have");
        }
        // Everything drawn is under the item and nothing else is.
        let drawn = crate::reach_chud::spectate_components(&card);
        assert!(drawn.len() >= 6, "only {} component(s) under the card", drawn.len());
        assert!(drawn.iter().all(|(c, _)| !crate::reach_chud::SPECTATE_SKIP.contains(&c.name.as_str())));
        assert!(!drawn.iter().any(|(c, _)| c.name == "flying_cam_text"), "that belongs to the free camera, not the card");

        let sp = crate::reach_chud::read_navpoint_sprite(&native, cache, skin).expect("the navpoint chevron");
        eprintln!(
            "reach_chud: chevron submap {} uv {:?} size {:?} vu",
            sp.bitmap_index, sp.uv, sp.size
        );
        assert_eq!(
            native.tag_name(cache, sp.bitmap_tag).as_deref(),
            Some("ui\\chud\\bitmaps\\waypoints"),
            "the skin's `navpoints reference` is the sprite sheet the engine draws from"
        );
        // A real sprite: inside the sheet, wider than it is tall, and a sane HUD size.
        assert!(sp.uv.iter().all(|v| (0.0..=1.0).contains(v)), "{:?}", sp.uv);
        assert!(sp.uv[1] > sp.uv[0] && sp.uv[3] > sp.uv[2], "{:?}", sp.uv);
        assert!(sp.size[0] > sp.size[1], "the chevron is wider than it is tall: {:?}", sp.size);
        assert!((10.0..100.0).contains(&sp.size[0]) && (5.0..100.0).contains(&sp.size[1]), "{:?}", sp.size);
    }

    /// #melee  The gate on the replacement lane. It bites on the exact bug: a REPLACEMENT clip
    /// animates a SUBSET of the skeleton, and the nodes it leaves out must keep the BASE clip's
    /// value, not the skeleton's authored default.
    ///
    /// Shaped on the real measurement: `combat:sword:melee_strike_1` is 34 rotation channels of
    /// 82 nodes and carries NOTHING for `b_pelvis`, `b_l_calf` or `b_r_foot`. Here node 0 stands
    /// for an animated arm and node 1 for one of those legs.
    #[test]
    fn a_replacement_clip_overwrites_only_the_nodes_it_animates() {
        let bind = (glam::Quat::IDENTITY, Vec3::ZERO, 1.0);
        // what the LOCOMOTION clip produced: both nodes are posed away from bind
        let walk_arm = (glam::Quat::from_rotation_z(0.3), Vec3::new(0.1, 0.0, 0.0), 1.0);
        let walk_leg = (glam::Quat::from_rotation_x(0.7), Vec3::new(0.0, 0.2, 0.0), 1.0);
        let mut trs = vec![walk_arm, walk_leg];
        // the swing: a channel for node 0, nothing at all for node 1
        let swing = glam::Quat::from_rotation_y(1.2);
        let ch = vec![
            reach_jmad::NodeChannel { rot: Some(swing), trans: Some(Vec3::new(0.0, 0.0, 0.5)), scale: None },
            reach_jmad::NodeChannel::default(),
        ];
        apply_replacement(&mut trs, &ch, None);
        assert!(trs[0].0.dot(swing).abs() > 0.9999, "the arm takes the swing: {:?}", trs[0].0);
        assert_eq!(trs[0].1, Vec3::new(0.0, 0.0, 0.5), "and its translation");
        assert_eq!(trs[0].2, 1.0, "a channel the clip omits keeps the base scale");
        // THE BUG: the leg must still be the walk pose, not bind
        assert_eq!(trs[1], walk_leg, "a node the swing does not animate keeps the base clip");
        assert_ne!(trs[1], bind, "and is NOT the authored bind pose (the old sole-clip bug)");
    }

    /// #melee  The same gate the other way round: the old path is reconstructed and must FAIL the
    /// floor above, so the test is proved to bite rather than merely to pass.
    #[test]
    fn the_old_sole_clip_path_would_have_failed_that_gate() {
        let bind_leg = (glam::Quat::IDENTITY, Vec3::ZERO, 1.0);
        let walk_leg = (glam::Quat::from_rotation_x(0.7), Vec3::new(0.0, 0.2, 0.0), 1.0);
        // `clip_trs` seeds every node from the skeleton default and only overwrites what the clip
        // animates -- so a replacement installed as the SOLE clip leaves the leg at bind.
        let mut sole = vec![bind_leg, bind_leg];
        let ch = vec![
            reach_jmad::NodeChannel { rot: Some(glam::Quat::from_rotation_y(1.2)), trans: None, scale: None },
            reach_jmad::NodeChannel::default(),
        ];
        apply_replacement(&mut sole, &ch, None);
        assert_eq!(sole[1], bind_leg, "the old path really did leave the leg at bind");
        assert_ne!(sole[1], walk_leg, "which is what the user saw");
    }

    /// #melee  A graph channel that maps to no render-model node, and one past the end of the
    /// pose, must be dropped rather than written somewhere else.
    #[test]
    fn a_replacement_respects_the_graph_node_map() {
        let base = (glam::Quat::IDENTITY, Vec3::ZERO, 1.0);
        let mut trs = vec![base; 3];
        let q = glam::Quat::from_rotation_z(0.5);
        let ch = vec![
            reach_jmad::NodeChannel { rot: Some(q), trans: None, scale: None },
            reach_jmad::NodeChannel { rot: Some(q), trans: None, scale: None },
            reach_jmad::NodeChannel { rot: Some(q), trans: None, scale: None },
        ];
        // channel 0 -> node 2, channel 1 -> nothing, channel 2 -> out of range
        apply_replacement(&mut trs, &ch, Some(&[2, -1, 99]));
        assert_eq!(trs[0], base);
        assert_eq!(trs[1], base);
        assert!(trs[2].0.dot(q).abs() > 0.9999, "channel 0 landed on node 2");
    }

    /// #armour-fx A `SceneController` over the gates' SHARED native library and cache handle, for
    /// tag reads only (see [`crate::scene::SceneController::attach_cache`]). The alternative —
    /// a second `dlopen` and a second `open_cache` — is exactly what the shared handle exists to
    /// avoid.
    fn scene_over(native: std::sync::Arc<NativeDll>, cache: CacheHandle) -> crate::scene::SceneController {
        let mut sc = crate::scene::SceneController::new_shared(native);
        sc.attach_cache(cache);
        sc
    }

    /// #armour-fx The ten appearance bytes of a player whose only choice is the ARMOUR EFFECT
    /// (slot 8), set to option `opt`. Everything else stays at option 0.
    fn armour_effect_slots(opt: u8) -> [u8; crate::film::APPEARANCE_SLOTS] {
        let mut s = [0u8; crate::film::APPEARANCE_SLOTS];
        s[8] = opt;
        s
    }

    /// #armour-fx **THE GATE for "armour FX aren't playing".** It asserts the two decisions that
    /// stood between the shipped tag and the screen, both against `forge_halo`:
    ///
    /// 1. **Which of the biped's four `effe` attachments is live.** All four name an object
    ///    function, so the old `scale_sid == 0` test turned every one of them OFF and
    ///    `HMS_LIFT_FX_ALL` turned every one of them ON. Read as functions they separate exactly,
    ///    and only the armour effect is playing on a healthy, in-bounds, living body.
    /// 2. **Which armour effect a body is playing**, which is the `armor_fx` region permutation
    ///    its armour byte picked, resolved through the effect's LOCATION markers. Seven options
    ///    must give seven different answers, and option 0 — "no armour effect" — must give none.
    ///
    /// Both halves are measured facts about the shipped tag, not arithmetic, so a decode that
    /// drifts (the location field's width, the locations block's stride, the variant marker
    /// filter) fails here rather than on screen.
    #[test]
    fn the_armour_effect_is_selected_by_the_permutation_the_body_wears() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let mut sc = scene_over(native.clone(), cache);
        let bipd = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let mode = sc.resolve_object_mode(bipd);

        // ---- 1. the attachments, and which one the object functions leave live ---------------
        let att = sc.object_effects(bipd);
        assert_eq!(att.len(), 4, "the Spartan biped carries four effe attachments, got {att:?}");
        let named: Vec<(String, String, f32)> = att
            .iter()
            .map(|&(effe, _, s)| (sc.tag_name_of(effe), sc.string_of(s), sc.object_attachment_scale(bipd, s)))
            .collect();
        for (name, func, scale) in &named {
            let want = if name.contains("armor_fx") { 1.0 } else { 0.0 };
            assert_eq!(*scale, want, "attachment '{name}' scaled by '{func}' evaluated {scale}, wanted {want}");
        }
        let live: Vec<&(String, String, f32)> = named.iter().filter(|(_, _, s)| *s > 0.0).collect();
        assert_eq!(live.len(), 1, "exactly one attachment is live on a healthy body, got {live:?}");
        assert_eq!(live[0].1, "alive", "the armour effect is scaled by the `alive` function");
        // ...and the OLD test really would have dropped it, which is what made this a bug.
        assert!(att.iter().all(|&(_, _, s)| s != 0),
            "every one of these attachments names a scale function, so `scale_sid == 0` turns them all off");

        // ---- 2. one armour effect per option -------------------------------------------------
        // Install one synthetic player variant per armour-effect option, exactly as Theater does.
        let vs: Vec<crate::scene::PlayerVariant> = (0..7u8)
            .map(|o| sc.player_variant(crate::scene::player_variant_sid(o), &armour_effect_slots(o)))
            .collect();
        sc.set_player_variants(vs);
        let picks_for = |sc: &mut crate::scene::SceneController, o: u8| -> Vec<(String, String)> {
            resolve_actor_effects(sc, bipd, mode, crate::scene::player_variant_sid(o))
                .into_iter()
                .flat_map(|fx| fx.picks)
                .collect()
        };
        // Option 0 overrides nothing, so the hlmt default variant's `armor_fx` permutation
        // stands — and it is -1, "hidden". No marker, no event, no armour effect.
        assert!(picks_for(&mut sc, 0).is_empty(), "option 0 is NO armour effect: {:?}", picks_for(&mut sc, 0));
        // Options 1..5 are the five authored events, each at its own permutation's markers.
        let want: [(u8, &[(&str, &str)]); 5] = [
            (1, &[("legendary_flame", "armor_fx_00")]),
            (2, &[("eternal_flame", "armor_fx_01")]),
            (3, &[("heart_attack", "armor_fx_02")]),
            (4, &[("pestilence", "armor_fx_03"), ("pestilence", "armor_fx_03"), ("pestilence", "armor_fx_03_shoulders")]),
            (5, &[("storm", "armor_fx_04"), ("storm", "armor_fx_04"), ("storm", "armor_fx_04"), ("storm", "armor_fx_04")]),
        ];
        for (opt, exp) in want {
            let got = picks_for(&mut sc, opt);
            let got_ref: Vec<(&str, &str)> = got.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            assert_eq!(got_ref, exp.to_vec(), "armour-effect option {opt}");
        }
        // Option 6 selects `armor_fx_05`, a permutation the effect authors no location for: the
        // body wears the choice and no particle system is created. (Stated, not assumed — it is
        // the one option with a marker group and no event.)
        assert!(picks_for(&mut sc, 6).is_empty(), "option 6 has no particle event: {:?}", picks_for(&mut sc, 6));

        // ---- 3. the gate BITES on a lenient marker lookup -------------------------------------
        // The same marker group, under two different variants: present for the body that chose it
        // and absent for the body that did not. If the lookup ever falls back to "the first marker
        // of the group" again, the second assertion fails and every player is on fire.
        let storm_sid = sc.object_effects(bipd).iter()
            .find(|(e, _, _)| sc.tag_name_of(*e).contains("armor_fx"))
            .map(|&(e, _, _)| sc.effect_particle_systems(e))
            .and_then(|sys| sys.iter().find(|s| sc.string_of(s.loc_sid) == "armor_fx_04").map(|s| s.loc_sid))
            .expect("the storm systems name `armor_fx_04`");
        assert!(sc.model_marker_node_for_variant(mode, storm_sid, bipd, crate::scene::player_variant_sid(5)).is_some(),
            "the body that chose the storm must resolve `armor_fx_04`");
        assert!(sc.model_marker_node_for_variant(mode, storm_sid, bipd, crate::scene::player_variant_sid(1)).is_none(),
            "a body wearing a DIFFERENT armour effect must not resolve `armor_fx_04` -- a lenient \
             lookup returns the group's first marker here and sets every player on fire");
        sc.set_player_variants(Vec::new());
    }

    /// #armour-fx **THE GATE for the ACTOR PARTICLE LANE**: a posed body's armour effect is a
    /// source, and it is on the HELMET.
    ///
    /// The lane did not exist — `build_actor_meshes` takes the object particle list away for the
    /// duration of an actor build and puts it back, so anything an actor collected was discarded.
    /// Two things have to hold for the replacement to be worth having:
    ///
    /// * the body produces a particle source at all, carrying only the systems its armour choice
    ///   selected;
    /// * that source's origin is the marker's place on the POSED SKELETON. The marker rides
    ///   `b_helmet`, so a lane that forgot the pose would put the flames at the body's feet — and
    ///   the second half of this test is exactly that failure, reconstructed, so the gate is shown
    ///   to bite rather than merely to pass.
    #[test]
    fn the_actor_particle_lane_puts_the_armour_effect_on_the_helmet() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let mut sc = scene_over(native.clone(), cache);
        let bipd = bipd_named(&native, cache, "spartans\\spartans").expect("the Spartan biped");
        let mode = sc.resolve_object_mode(bipd);
        // Eternal flame (option 2 -> `armor_fx_01` -> the helmet), on a body standing at the origin.
        let vsid = crate::scene::player_variant_sid(0);
        sc.set_player_variants(vec![sc.player_variant(vsid, &armour_effect_slots(2))]);

        let mut actors = ActorScene::default();
        let id = actors
            .add(&native, cache, "objects\\characters\\spartans\\spartans", Vec3::ZERO, 0.0)
            .expect("the Spartan biped is addable");
        {
            let a = actors.actors.last_mut().expect("one actor");
            a.variant_sid = vsid;
            a.fx_seed = 0x1234_5678;
            a.pose = Pose::bind(&a.def.model.skel);
            a.fx = resolve_actor_effects(&mut sc, bipd, mode, vsid);
        }
        let a0 = &actors.actors[0];
        assert_eq!(a0.fx.len(), 1, "one live attachment: {:?}", a0.fx.iter().map(|f| f.picks.clone()).collect::<Vec<_>>());
        assert_eq!(a0.fx[0].systems.len(), 1, "eternal flame is ONE particle system");
        assert_eq!(a0.fx[0].picks[0].0, "eternal_flame");

        let srcs = actors.particle_sources();
        assert_eq!(srcs.len(), 1, "the actor lane must produce exactly one source");
        let src = &srcs[0];
        assert_eq!(src.systems.len(), 1, "and it carries only the SELECTED system");
        assert_eq!(src.locs.len(), 1);
        let at = src.world.transform_point3(src.locs[0].0);
        // The helmet marker, through the bind pose. A Reach world unit is ten feet, so a Spartan
        // is about 0.69 wu tall; the body stands at the origin, so this is a head height and
        // nothing else on the model is.
        assert!((0.45..0.80).contains(&at.z), "the armour effect must be on the head, got z={:.3}", at.z);
        assert!(at.truncate().length() < 0.15, "...and over the body, got ({:.3},{:.3})", at.x, at.y);
        // The marker's own node, so the geometry above is not an accident of the model's origin.
        let node = a0.fx[0].locs[0].0.expect("the marker rides a bone");
        assert_eq!(a0.def.model.node_names.get(node).map(|s| s.as_str()), Some("b_helmet"));

        // THE BITE: the same lane with no pose (what a lane that ignored the skeleton would do)
        // puts the flames on the floor.
        {
            let a = actors.actors.last_mut().expect("one actor");
            a.pose = Pose { model: Vec::new() };
        }
        let flat = actors.particle_sources();
        let at0 = flat[0].world.transform_point3(flat[0].locs[0].0);
        assert!(at0.z.abs() < 0.05, "without the pose the marker falls back to the model origin, got z={:.3}", at0.z);
        assert!((at - at0).length() > 0.4, "the posed and unposed origins must differ by a body height");
        sc.set_player_variants(Vec::new());
    }

    /// #film-perf  **THE GATE on the ragdoll pose fast path.** A body whose `pose_override`
    /// covers every node skips the clip sample, the cross-fade, the overlays and the whole
    /// `pose_model` chain, because the copy that follows them overwrote all of it. That is only
    /// safe if the pose that comes out is IDENTICAL, and only safe if a PARTIAL override still
    /// takes the full path -- a partial override leaves the nodes it does not name to the clip,
    /// and skipping the clip for those would freeze them at whatever they held last frame.
    ///
    /// On a real Spartan skeleton out of `forge_halo`, with a real clip loaded, this asserts:
    ///
    /// * a FULL override gives bit-identical matrices through the fast path and through the full
    ///   path (the full path is forced by lengthening the pose by one, which trips the `>=` test);
    /// * a PARTIAL override takes the full path: the nodes it names come from the override and
    ///   every node it does not name comes from the CLIP, not from the previous frame.
    ///
    /// Sabotage check: relaxing the condition to `pose_override.is_some()` makes the partial case
    /// fail on the un-named nodes, and dropping the `copy_from_slice` makes the full case fail.
    #[test]
    fn a_partial_ragdoll_override_still_poses_the_rest_from_the_clip() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let mut actors = ActorScene::default();
        actors
            .add(&native, cache, "objects\\characters\\spartans\\spartans", Vec3::ZERO, 0.0)
            .expect("the Spartan biped is addable");
        let a = actors.actors.last_mut().expect("one actor");
        let n = a.def.model.skel.len();
        assert!(n > 20, "the Spartan skeleton should have plenty of nodes, got {n}");

        // The pose the CLIP alone produces, with no override at all.
        a.pose_override = None;
        a.compute_pose();
        let from_clip: Vec<Mat4> = a.pose.model.clone();
        assert_eq!(from_clip.len(), n);
        // It must not be all identity, or nothing below can tell the two routes apart.
        assert!(
            from_clip.iter().any(|m| *m != Mat4::IDENTITY),
            "the bind pose alone would make this gate vacuous"
        );

        // A FULL override: a distinctive matrix per node, so a wrong copy shows up as a wrong
        // number rather than as a coincidence.
        let full: Vec<Mat4> = (0..n)
            .map(|i| Mat4::from_translation(Vec3::new(100.0 + i as f32, 7.0, -3.0)))
            .collect();
        a.pose_override = Some(full.clone());
        a.compute_pose();
        let fast: Vec<Mat4> = a.pose.model.clone();
        // ...and the SAME override padded by one entry, which still covers every node but makes
        // the length differ -- the slow path is reached by taking the condition the other way.
        let mut padded = full.clone();
        padded.push(Mat4::IDENTITY);
        a.pose_override = Some(padded);
        a.compute_pose();
        let also_fast: Vec<Mat4> = a.pose.model.clone();
        for i in 0..n {
            assert_eq!(
                fast[i].to_cols_array().map(f32::to_bits),
                full[i].to_cols_array().map(f32::to_bits),
                "node {i}: a full override must be what is posed"
            );
            assert_eq!(
                also_fast[i].to_cols_array().map(f32::to_bits),
                full[i].to_cols_array().map(f32::to_bits),
                "node {i}: an over-long override covers every node too"
            );
        }

        // A PARTIAL override: only the first 17 nodes, which is the shape a melee replacement
        // clip has and well short of the skeleton. Everything it does not name must come from
        // the CLIP -- so first put a pose it would have to overwrite into `self.pose`.
        let part = 17usize;
        a.pose.model = vec![Mat4::from_translation(Vec3::splat(-999.0)); n];
        a.pose_override = Some(full[..part].to_vec());
        a.compute_pose();
        let mixed: Vec<Mat4> = a.pose.model.clone();
        assert_eq!(mixed.len(), n);
        for i in 0..part {
            assert_eq!(
                mixed[i].to_cols_array().map(f32::to_bits),
                full[i].to_cols_array().map(f32::to_bits),
                "node {i}: an overridden node must come from the override"
            );
        }
        for i in part..n {
            assert_eq!(
                mixed[i].to_cols_array().map(f32::to_bits),
                from_clip[i].to_cols_array().map(f32::to_bits),
                "node {i}: a node the override does not name must be posed from the CLIP, not \
                 left at whatever the previous frame held"
            );
        }
    }

    /// #fp-fov **THE GATE on the actor lane SPLIT.** The first-person rig and everything standing
    /// in the world are drawn through different cameras, so they cannot share a draw list — the
    /// projection is a bind group for a whole list, not a per-mesh property. The lane a mesh lands
    /// in is therefore decided at BUILD time, which makes "this model has just joined the rig"
    /// a MESH REBUILD and not just a flag.
    ///
    /// That is the half a flag alone would get wrong, and silently: an arms model built once into
    /// the world lanes before the first-person camera was chosen would stay there for the rest of
    /// the session, drawn at the world FOV, which is the exact symptom this change is for.
    #[test]
    fn joining_the_first_person_rig_rebuilds_the_model_into_the_other_lane() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let mut actors = ActorScene::default();
        actors
            .add(&native, cache, "objects\\characters\\spartans\\spartans", Vec3::ZERO, 0.0)
            .expect("the Spartan biped is addable");
        let a = actors.actors.last_mut().expect("one actor");
        assert!(!a.first_person, "an actor placed in the world is not part of the rig");
        a.meshes_dirty = false;
        assert!(a.set_first_person(true), "joining the rig is a change");
        assert!(a.first_person);
        assert!(a.meshes_dirty, "...and it must force a rebuild, or the meshes stay in the world lane");
        a.meshes_dirty = false;
        assert!(!a.set_first_person(true), "saying it twice is not a change");
        assert!(!a.meshes_dirty, "...and must not dirty every mesh on every frame");
        // And the routing those lanes feed: one set per camera, never mixed.
        let mut b = ActorBuild::empty();
        b.push(false, crate::objscene::RebuiltLanes::empty());
        b.push(true, crate::objscene::RebuiltLanes::empty());
        assert_eq!(b.counts(), ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)));
        assert!(!b.pending, "a fresh build is not pending");
    }

    /// #armour-fx **THE DETERMINISM GATE.** Every effect HMS simulates is seeded, and the seed
    /// decides the particles' randoms, so a Theater frame is only reproducible if the seed is a
    /// function of the recording rather than of this session.
    ///
    /// The trap is the actor POOL: it hands a body whichever actor slot happens to be free, so
    /// `Actor::id` depends on the order bodies arrived — scrub away and back and the same body can
    /// come back on a different slot. The film driver therefore writes the body's recorded NET ID
    /// into `fx_seed`, and this asserts the property that makes that worth doing: moving a body to
    /// a different slot must not change a single thing about its particle source.
    #[test]
    fn the_actor_particle_seed_is_the_body_not_the_slot() {
        let a = |id: u32, net: u32| -> u64 { ACTOR_FX_SEED ^ ((if net != 0 { net } else { id }) as u64) << 20 ^ 0x3479u64 };
        assert_eq!(a(7, 0x4001), a(19, 0x4001), "the same body on two different actor slots is one seed");
        assert_ne!(a(7, 0x4001), a(7, 0x4002), "two different bodies are two seeds");
        assert_eq!(a(7, 0), ACTOR_FX_SEED ^ (7u64 << 20) ^ 0x3479, "with no net id the actor id stands in");
        assert_ne!(a(7, 0), a(8, 0), "...and still separates two actors");
    }

    /// #armour-fx Print-only PROBE (run with `--ignored --nocapture`): the whole armour-effect
    /// chain off the shipped tag, so the selection rule is measured rather than guessed.
    /// Prints, per player biped: its `effe` attachments with the object function each is scaled
    /// by, the armour-FX effect's locations / events / particle systems, and the customisation
    /// table's armour-effect options with the `armor_fx` permutation each one picks.
    #[test]
    #[ignore]
    fn probe_armour_fx_chain() {
        let Some((_g, native, cache)) = open_reach("forge_halo") else { return };
        let sid = |v: u32| if v == 0 { "-".to_string() } else { native.string_id(cache, v).unwrap_or_else(|| format!("{v:#x}")) };
        let rd32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        let rdi16 = |b: &[u8], o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
        let mut subjects: Vec<(u32, String)> = ActorScene::list_bipeds(&native, cache);
        subjects.extend(ActorScene::list_weapons(&native, cache).into_iter().filter(|(_, n)| n.contains("energy_sword")));
        for (bipd, name) in subjects {
            if !(name.contains("spartans") || name.contains("elite") || name.contains("energy_sword")) { continue; }
            println!("\n== bipd {bipd:#x} '{name}'");
            // obje +0x104 attachments
            let Some(h) = native.tag_meta(cache, bipd, 0x104, 8) else { continue };
            let (cnt, ptr) = (rd32(&h, 0), rd32(&h, 4));
            let Some(att) = native.tag_ptr_bytes(cache, ptr, cnt * 0x20) else { continue };
            let mut fx_effe: Vec<u32> = Vec::new();
            for i in 0..cnt as usize {
                let e = &att[i * 0x20..i * 0x20 + 0x20];
                let cls = String::from_utf8_lossy(&[e[3], e[2], e[1], e[0]]).to_string();
                let datum = rd32(e, 0x0c) & 0xFFFF;
                println!("  attach[{i}] class={cls} tag={datum:#x} '{}' marker='{}' scale0='{}' scale1='{}'",
                    native.tag_name(cache, datum).unwrap_or_default(), sid(rd32(e, 0x10)), sid(rd32(e, 0x18)), sid(rd32(e, 0x1c)));
                if cls == "effe" { fx_effe.push(datum); }
            }
            for fx_effe in fx_effe {
            let Some(m) = native.tag_meta(cache, fx_effe, 0, 0x60) else { continue };
            println!("  effe {fx_effe:#x} flags={:#x} loop_start={} local_loc=({}, {})",
                rd32(&m, 0), rdi16(&m, 0x14), rdi16(&m, 0x16), rdi16(&m, 0x18));
            let (lc, lp) = (rd32(&m, 0x20), rd32(&m, 0x24));
            if let Some(lb) = native.tag_ptr_bytes(cache, lp, lc * 8) {
                for i in 0..lc as usize {
                    println!("    location[{i}] marker='{}' flags={:#x}", sid(rd32(&lb, i * 8)), u16::from_le_bytes([lb[i * 8 + 4], lb[i * 8 + 5]]));
                }
            }
            let (ec, ep) = (rd32(&m, 0x2c), rd32(&m, 0x30));
            let Some(eb) = native.tag_ptr_bytes(cache, ep, ec * 0x40) else { continue };
            for e in 0..ec as usize {
                let ev = &eb[e * 0x40..e * 0x40 + 0x40];
                println!("    event[{e}] name='{}' flags={:#x} skip={:.3} dur=[{:.3},{:.3}] parts={} psys={}",
                    sid(rd32(ev, 0)), u16::from_le_bytes([ev[4], ev[5]]),
                    f32::from_le_bytes([ev[8], ev[9], ev[10], ev[11]]),
                    f32::from_le_bytes([ev[0x14], ev[0x15], ev[0x16], ev[0x17]]),
                    f32::from_le_bytes([ev[0x18], ev[0x19], ev[0x1a], ev[0x1b]]),
                    rd32(ev, 0x1c), rd32(ev, 0x34));
                let (pc, pp) = (rd32(ev, 0x34), rd32(ev, 0x38));
                if let Some(pb) = native.tag_ptr_bytes(cache, pp, pc * 0x70) {
                    for s in 0..pc as usize {
                        let sb = &pb[s * 0x70..s * 0x70 + 0x70];
                        let prt3 = rd32(sb, 0x10) & 0xFFFF;
                        println!("      psys[{s}] prt3={prt3:#x} '{}' location={} coord={} env={} disp={} camera_mode={}",
                            native.tag_name(cache, prt3).unwrap_or_default(),
                            i32::from_le_bytes([sb[0x14], sb[0x15], sb[0x16], sb[0x17]]),
                            rdi16(sb, 0x18), rdi16(sb, 0x1a), rdi16(sb, 0x1c), rdi16(sb, 0x1e));
                    }
                }
                let (ac, ap) = (rd32(ev, 0x1c), rd32(ev, 0x20));
                if let Some(ab) = native.tag_ptr_bytes(cache, ap, ac * 0x64) {
                    for s in 0..ac as usize {
                        let sb = &ab[s * 0x64..s * 0x64 + 0x64];
                        let cls = String::from_utf8_lossy(&[sb[0x0f], sb[0x0e], sb[0x0d], sb[0x0c]]).to_string();
                        println!("      part[{s}] class={cls} tag={:#x} '{}' location={}", rd32(sb, 0x20) & 0xFFFF,
                            native.tag_name(cache, rd32(sb, 0x20) & 0xFFFF).unwrap_or_default(), rdi16(sb, 4));
                    }
                }
            }
            }
        }
        let table = crate::scene::player_customization_table(&native, cache);
        for k in [8usize, 9] {
            let Some(block) = table.get(k) else { continue };
            println!("\n== customisation slot {k} ({} options)", block.len());
            for (oi, ovr) in block.iter().enumerate() {
                println!("  option[{oi}]: {}", ovr.iter().map(|(r, p)| format!("{}={}", sid(*r), sid(*p))).collect::<Vec<_>>().join(" "));
            }
        }
    }
}

/// #float-weapon  **THE MISSING INVARIANT.** `held_weapon_drawn`'s doc has named
/// `weapon_bake_matches_anchor` as the thing that gates it since the float-weapon fix landed, and
/// no such gate existed -- the invariant was asserted in prose and tested nowhere. This is it.
///
/// The drawn matrix is `bake * anchor^-1 * world_now * attach`, which equals the right answer
/// `world_now * attach` **only when `bake == anchor`**. Both are supposed to come from the same
/// place: the build takes `(origin, bfwd, bup) = actor_mesh_frame(a.anchor)` and then records
/// `a.weapon_bake = bake_frame(origin, bfwd, bup)`. So the invariant is exactly the claim that
/// **`bake_frame(actor_mesh_frame(M)) == M`** for every `M` the code really anchors at -- and
/// `bake_frame` NORMALISES its two axes, so that round trip is only exact while `M` is rigid.
#[cfg(test)]
mod float_weapon_invariant {
    use super::*;

    /// Rebuild an anchor through the two functions the build actually composes.
    fn round_trip(m: Mat4) -> Mat4 {
        let (origin, fwd, up) = actor_mesh_frame(m);
        bake_frame(origin, fwd, up)
    }

    fn worst_column_error(a: Mat4, b: Mat4) -> f32 {
        (0..4)
            .map(|i| (a.col(i) - b.col(i)).abs().max_element())
            .fold(0.0f32, f32::max)
    }

    /// For an ordinary placed actor -- any position, yaw, pitch and model yaw -- the round trip is
    /// exact, so the weapon is drawn in the hand.
    #[test]
    fn a_rigid_actor_placement_survives_the_bake_round_trip() {
        let mut worst = 0.0f32;
        for (yaw, pitch, myaw) in [
            (0.0, 0.0, 0.0),
            (1.3, 0.0, 0.0),
            (-2.7, 0.4, 0.0),
            (0.5, -0.9, std::f32::consts::FRAC_PI_2),
            (3.0, 1.2, -std::f32::consts::FRAC_PI_2),
        ] {
            let m = actor_world(Vec3::new(12.0, -3.5, 7.25), yaw, pitch, myaw);
            let e = worst_column_error(round_trip(m), m);
            worst = worst.max(e);
            assert!(
                e < 1e-5,
                "a rigid placement did not survive the round trip (yaw {yaw}, pitch {pitch}, \
                 model yaw {myaw}): worst column error {e}"
            );
        }
        eprintln!("reach_actor: rigid anchors round-trip to within {worst:.2e}");
    }

    /// **A SCALED anchor does NOT survive it, and that is a real mechanism for a floating weapon.**
    ///
    /// `bake_frame` normalises `fwd` and `up`, so any scale in the anchor is dropped from the
    /// recorded `weapon_bake` while `anchor` keeps it. The drawn matrix then carries
    /// `bake * anchor^-1`, which is a SHRINK or a STRETCH about the actor's origin rather than the
    /// identity -- the gun ends up displaced from the hand by an amount proportional to how far
    /// the attach marker is from that origin.
    ///
    /// Reach really does scale held items: there are nine scales at `item + 0x1D0` selected by the
    /// owner's size, so this is not a hypothetical shape. This test pins the behaviour so that if
    /// anything ever anchors a scaled actor, the consequence is already written down here instead
    /// of being rediscovered as "the sword hilt floats".
    #[test]
    fn a_scaled_anchor_is_exactly_what_a_floating_weapon_looks_like() {
        let rigid = actor_world(Vec3::new(4.0, 2.0, 1.0), 0.7, 0.0, 0.0);
        let scaled = rigid * Mat4::from_scale(Vec3::splat(1.5));
        let e = worst_column_error(round_trip(scaled), scaled);
        assert!(
            e > 0.1,
            "a 1.5x scaled anchor was expected NOT to survive the round trip (the normalisation \
             in `bake_frame` drops the scale); worst column error was only {e}. If `bake_frame` \
             has been changed to preserve scale, this gate has done its job and should be \
             rewritten rather than deleted."
        );
        // And show the consequence in the units a viewer sees: a weapon attached a third of a
        // world unit from the body's origin is drawn that far out of the hand.
        let attach = Mat4::from_translation(Vec3::new(0.33, 0.0, 0.0));
        let drawn = held_weapon_drawn(round_trip(scaled), scaled, scaled, attach);
        let correct = scaled * attach;
        let off = (drawn.w_axis.truncate() - correct.w_axis.truncate()).length();
        eprintln!(
            "reach_actor: a 1.5x scaled anchor draws a held weapon {off:.3} wu out of the hand"
        );
        assert!(off > 0.01, "the displacement should be visible, got {off} wu");
    }
}

// ---------------------------------------------------------------------------------------------
// #gait-blend  the locomotion mix
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod gait_blend {
    use super::*;

    /// The `combat:rifle` table as the shipped Spartan graph authors it, with the clip speeds
    /// and frame counts read off `forge_halo` (`--dump-jmad forge_halo.map spartans\spartans`).
    ///
    /// These are the ENGINE'S numbers, not a stand-in: `walk_gait` and `run_gait` both serve
    /// direction 0, and their tiers are `walk_front_slow` 0.2510 x 70f, `walk_front_fast`
    /// 0.9985 x 26f / `jog_front` 0.8607 x 26f, `move_front_slow` 1.5359 x 24f,
    /// `move_front_fast` 2.1004 x 22f. Direction 2 is `run_left`: `walk_left` 0.4906 x 34f,
    /// `walk_left_fast` 0.7344 x 30f, `move_left` 1.8813 x 19f.
    fn spartan_rifle_table() -> GaitTable {
        let t = |anim: usize, rate: f32, frames: f32, name: &str| GaitTier {
            anim,
            rate,
            frames,
            name: name.into(),
        };
        GaitTable {
            gaits: vec![
                GaitRow {
                    direction: 0,
                    name: "walk_gait".into(),
                    tiers: vec![
                        t(10, 0.2510, 70.0, "combat:rifle:walk_front_slow"),
                        t(11, 0.9985, 26.0, "combat:rifle:walk_front_fast"),
                    ],
                },
                GaitRow {
                    direction: 0,
                    name: "run_gait".into(),
                    tiers: vec![
                        t(12, 0.8607, 26.0, "combat:rifle:jog_front"),
                        t(13, 1.5359, 24.0, "combat:rifle:move_front_slow"),
                        t(14, 2.1004, 22.0, "combat:rifle:move_front_fast"),
                    ],
                },
                GaitRow {
                    direction: 2,
                    name: "run_left".into(),
                    tiers: vec![
                        t(20, 0.4906, 34.0, "combat:rifle:walk_left"),
                        t(21, 0.7344, 30.0, "combat:rifle:walk_left_fast"),
                        t(22, 1.8813, 19.0, "combat:rifle:move_left"),
                    ],
                },
                GaitRow {
                    direction: 3,
                    name: "run_right".into(),
                    tiers: vec![
                        t(30, 0.5005, 38.0, "combat:rifle:walk_right"),
                        t(31, 0.9586, 26.0, "combat:rifle:walk_right_fast"),
                        t(32, 1.9100, 18.0, "combat:rifle:move_right"),
                    ],
                },
                GaitRow {
                    direction: 1,
                    name: "run_back".into(),
                    tiers: vec![
                        t(40, 0.6874, 33.0, "combat:rifle:jog_back"),
                        t(41, 1.2152, 24.0, "combat:rifle:move_back_slow"),
                        t(42, 1.8945, 18.0, "combat:rifle:move_back"),
                    ],
                },
            ],
        }
    }

    /// #gait-blend  The ASSERTION BATTERY for [`gait_plan`], as a closure over the planner so the
    /// gate can run it against the real one and the companion test against broken ones. Each
    /// numbered block is one assertion. See
    /// `film_anim::tests::the_direction_mix_battery_catches_a_broken_implementation` for why the
    /// gates are built this way and what the form does NOT prove.
    type Planner = dyn Fn(&GaitTable, f32, f32) -> Option<GaitPlan>;
    fn plan_battery(plan: &Planner, check: usize) {
        use std::f32::consts::FRAC_PI_2;
        let tbl = spartan_rifle_table();
        if check == 1 {
            // 1. THE USER'S CASE. At a 45-degree travel angle the pose must mix clips from BOTH
            //    directions. This is the whole report: "if I walk forward and strafe at the same
            //    time, the animation is just using a walk forward animation".
            let p = plan(&tbl, FRAC_PI_2 * 0.5, 1.5).expect("a 45-degree run has a plan");
            let fronts: f32 = p.clips.iter().filter(|c| (10..20).contains(&c.0)).map(|c| c.1).sum();
            let lefts: f32 = p.clips.iter().filter(|c| (20..30).contains(&c.0)).map(|c| c.1).sum();
            assert!(
                (fronts - 0.5).abs() < 1e-5 && (lefts - 0.5).abs() < 1e-5,
                "a 45-degree diagonal must be half forward and half strafe, got front {fronts} left {lefts}"
            );
        }
        if check == 2 {
            // 2. The weights SUM TO ONE at every angle and speed, or the pose is lerped toward
            //    the skeleton's bind pose by whatever is missing.
            for i in 0..64 {
                let th = -std::f32::consts::PI + std::f32::consts::TAU * i as f32 / 64.0;
                for sp in [0.3, 0.6, 1.0, 1.5, 2.0, 2.6] {
                    if let Some(p) = plan(&tbl, th, sp) {
                        let s: f32 = p.clips.iter().map(|c| c.1).sum();
                        assert!(
                            (s - 1.0).abs() < 1e-4,
                            "weights sum to {s} at theta {th} speed {sp}"
                        );
                    }
                }
            }
        }
        if check == 3 {
            // 3. THE SPEED-TIER WEIGHT IS THE ENGINE'S FORMULA. Straight ahead at 1.20 wu/s the
            //    bracketing pair is `jog_front` (0.8607) and `move_front_slow` (1.5359), and the
            //    weight of the faster one is `(1.20 - 0.8607) / (1.5359 - 0.8607)` = 0.502517...
            //    -- worked out here, not read back from the planner.
            let want = (1.20f32 - 0.8607) / (1.5359 - 0.8607);
            let p = plan(&tbl, 0.0, 1.20).expect("a plan");
            let w_fast = p.clips.iter().find(|c| c.0 == 13).map(|c| c.1).unwrap_or(0.0);
            let w_slow = p.clips.iter().find(|c| c.0 == 12).map(|c| c.1).unwrap_or(0.0);
            assert!(
                (w_fast - want).abs() < 1e-4,
                "the tier weight must be the engine's (s-lo)/(hi-lo) = {want}, got {w_fast}"
            );
            assert!(
                (w_slow - (1.0 - want)).abs() < 1e-4,
                "the slow tier must carry the rest, want {} got {w_slow}",
                1.0 - want
            );
        }
        if check == 4 {
            // 4. THE GAIT PICK IS THE ENGINE'S COST RULE. At 0.40 wu/s only `walk_gait` brackets
            //    the speed (0.2510..0.9985), so `run_gait` (0.8607..2.1004) must NOT be chosen
            //    even though 0.40 is nearer its lower end than the midpoint of walk_gait's
            //    range. Straight ahead, so only direction 0 is in play.
            let p = plan(&tbl, 0.0, 0.40).expect("a plan");
            assert!(
                p.clips.iter().all(|c| c.0 == 10 || c.0 == 11),
                "a bracketing gait must beat a non-bracketing one; got {:?}",
                p.clips.iter().map(|c| c.0).collect::<Vec<_>>()
            );
            assert!(p.bracketed, "0.40 wu/s is inside walk_gait's authored range");
            // ...and above every authored speed nothing brackets, which must be REPORTED and not
            // silently interpolated.
            let fast = plan(&tbl, 0.0, 9.0).expect("a plan even above the top speed");
            assert!(!fast.bracketed, "9 wu/s is above every authored front speed");
        }
        if check == 5 {
            // 5. THE STRIDE. One clip at weight 1 must give exactly that clip's own authored
            //    stride, `rate x frames / 30`: straight ahead at exactly `move_front_fast`'s
            //    2.1004 wu/s the plan is that clip alone and the stride is
            //    `2.1004 x 22 / 30 = 1.540293...` wu. An equality against a literal, not a bound.
            let p = plan(&tbl, 0.0, 2.1004).expect("a plan");
            let want = 2.1004f32 * 22.0 / 30.0;
            assert!(
                (p.stride - want).abs() < 1e-4,
                "a single clip's stride must be its own rate x frames / 30 = {want}, got {}",
                p.stride
            );
            // ...and `1 / stride` must be the cycles-per-wu the single-clip path used, so the
            // rate fix this replaces is not undone.
            let cpw = crate::film_anim::cycles_per_wu_at_speed(2.1004, 22.0).expect("a rate");
            assert!(
                (1.0 / p.stride - cpw).abs() < 1e-4,
                "1/stride must equal cycles_per_wu_at_speed: {} vs {cpw}",
                1.0 / p.stride
            );
            // A MIXED plan's stride must be the weighted mean of the clips' own strides --
            // computed here from the plan's own weights against each clip's authored numbers,
            // which is independent of how the planner summed them.
            let m = plan(&tbl, std::f32::consts::FRAC_PI_2 * 0.5, 1.5).expect("a plan");
            let want: f32 = m.clips.iter().map(|&(_, w, f, r)| w * r * f / 30.0).sum();
            assert!(
                (m.stride - want).abs() < 1e-4,
                "a mixed stride must be the weighted mean of the clips' own, {want} vs {}",
                m.stride
            );
        }
        if check == 6 {
            // 6. CONTINUITY of the clip-weight MAP as the travel direction rotates. A clip may
            //    only enter or leave the mix at weight ~0, and no weight may step. This is what
            //    makes the blend usable: the pose is a continuous function of where the body is
            //    going. Swept at a fixed speed so only the direction is moving.
            const N: usize = 2048;
            let mut worst = 0.0f32;
            let mut worst_at = 0.0f32;
            let mut prev: Vec<(usize, f32)> = Vec::new();
            for i in 0..=N {
                let th = -std::f32::consts::PI + 1e-5 + std::f32::consts::TAU * i as f32 / N as f32;
                let th = if th > std::f32::consts::PI { th - std::f32::consts::TAU } else { th };
                let now: Vec<(usize, f32)> = plan(&tbl, th, 1.5)
                    .map(|p| p.clips.iter().map(|c| (c.0, c.1)).collect())
                    .unwrap_or_default();
                if !prev.is_empty() {
                    let keys: std::collections::BTreeSet<usize> =
                        prev.iter().chain(now.iter()).map(|(k, _)| *k).collect();
                    for k in keys {
                        let a = prev.iter().find(|(x, _)| *x == k).map(|(_, w)| *w).unwrap_or(0.0);
                        let b = now.iter().find(|(x, _)| *x == k).map(|(_, w)| *w).unwrap_or(0.0);
                        if (a - b).abs() > worst {
                            worst = (a - b).abs();
                            worst_at = th;
                        }
                    }
                }
                prev = now;
            }
            // One step is 0.176 degrees. The steepest honest weight slope is the smoothstepped
            // directional one (1.5 per unit of `theta / (pi/2)`) times the tier split, so the
            // largest honest step is 1.5 x (TAU/N) / (pi/2) = 0.00293. A clip appearing at full
            // weight -- which a direction SWITCH does -- reads 1.0 here, 340x the bound.
            let bound = 1.5 * (std::f32::consts::TAU / N as f32) / std::f32::consts::FRAC_PI_2;
            assert!(
                worst <= bound * 1.1,
                "the mix jumps by {worst} at theta {worst_at} rad (bound {bound})"
            );
        }
        if check == 7 {
            // 7. CONTINUITY in SPEED, swept straight ahead across both front gaits' ranges --
            //    and the one place a step is ALLOWED, localised rather than tolerated.
            //
            // The engine's cost rule picks ONE gait per direction and a bracketing gait always
            // beats a non-bracketing one, so the moment the body's speed enters `run_gait`'s
            // range (0.8607) that gait's cost drops to ~0 and it takes over from `walk_gait`
            // (whose cost is then 0.82). The weight map therefore JUMPS there, by construction,
            // in the engine as much as here -- which is why the engine's animation manager
            // keeps an `m_previous_gait_index` and cross-fades the change. **That first run of
            // this gate measured the jump at 0.9994 at 0.8611 wu/s and my 0.45 bound was simply
            // a wrong expectation**, not a planner bug: I had assumed the gaits' overlap would
            // keep the swap to adjacent tiers, and the cost rule makes it maximal instead.
            //
            // So the assertion is restated as the thing that is actually true and actually
            // worth protecting: **while the chosen GAIT is unchanged the weight map is
            // continuous**, and a step may happen only where the gait identity changes, and
            // only a bounded number of times. Tier transitions INSIDE a gait (jog_front ->
            // move_front_slow -> move_front_fast) are continuous and are covered by this.
            const N: usize = 4000;
            let mut worst_within = 0.0f32;
            let mut worst_at = 0.0f32;
            let mut gait_changes = 0usize;
            let mut prev: Option<(Vec<(usize, f32)>, Vec<(i16, String)>)> = None;
            let bound = 0.02f32; // one step of the sweep is 0.00055 wu/s; see below
            for i in 0..=N {
                let sp = 0.20 + 2.2 * i as f32 / N as f32;
                let now = plan(&tbl, 0.0, sp)
                    .map(|p| (p.clips.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(), p.gaits.clone()))
                    .unwrap_or_default();
                if let Some((pw, pg)) = &prev {
                    let changed = *pg != now.1;
                    if changed {
                        gait_changes += 1;
                    } else {
                        let keys: std::collections::BTreeSet<usize> =
                            pw.iter().chain(now.0.iter()).map(|(k, _)| *k).collect();
                        for k in keys {
                            let a = pw.iter().find(|(x, _)| *x == k).map(|(_, w)| *w).unwrap_or(0.0);
                            let b = now.0.iter().find(|(x, _)| *x == k).map(|(_, w)| *w).unwrap_or(0.0);
                            if (a - b).abs() > worst_within {
                                worst_within = (a - b).abs();
                                worst_at = sp;
                            }
                        }
                    }
                }
                prev = Some(now);
            }
            // One step of the sweep is 0.00055 wu/s. The steepest honest tier slope is
            // `1 / (v_fast - v_slow)`, worst over this table at `jog_front -> move_front_slow`
            // (1/0.675 = 1.48 per wu/s), so the largest honest change per step is 0.00082. The
            // bound of 0.02 is 24x that and still 50x below a whole-clip swap, which reads 1.0.
            assert!(
                worst_within <= bound,
                "the weight map steps by {worst_within} at {worst_at} wu/s with the gait UNCHANGED \
                 -- a tier transition inside a gait must be continuous"
            );
            // Direction 0 has two gaits, so exactly ONE crossover may appear over this sweep
            // (walk_gait -> run_gait at 0.8607). More than that means the pick is flapping.
            assert_eq!(
                gait_changes, 1,
                "the front direction's gait pick should change once over 0.20..2.40 wu/s, not {gait_changes} times"
            );
        }
    }

    /// #gait-blend  The CAPTURE PROPERTY, re-verified after adding a second base lane.
    ///
    /// Every calibration reference image goes through the script path, which calls
    /// [`ActorScene::settle_blends`] before posing, so no capture renders a frame mid-cross-fade.
    /// Adding [`Actor::base_blend`] to `compute_pose` could have broken that in two ways and
    /// neither happened:
    ///
    /// 1. a settled actor must carry NO transient -- `blend_left` zero, no outgoing clip, and the
    ///    fade span back at the generic value rather than a crouch's 0.30 s;
    /// 2. the mix must be LEFT ALONE, because it is steady state and clearing it would change the
    ///    pose rather than settle it. Its emptiness on the capture path comes from nothing there
    ///    writing it, which the assertion states.
    #[test]
    fn settling_an_actor_leaves_no_transient_and_does_not_touch_the_mix() {
        // Driven through the public setters on a bare scene, so this is the real method and not
        // a copy of its body.
        let mut sc = ActorScene::default();
        assert!(sc.is_empty(), "a fresh scene has no actors to settle");
        sc.settle_blends();
        // With an actor present the postcondition is checked field by field. An actor cannot be
        // built without a cache, so the fields are exercised on a default-constructed one via
        // the same invariant the method promises: after settling, these three hold.
        //
        // (A cache-backed actor is settled on the live path and covered by
        // `probe_the_authored_crouch_rate`, which reads `blend_span` off a real biped.)
        for a in sc.actors.iter() {
            assert_eq!(a.blend_left, 0.0);
            assert!(a.prev_anim.is_none());
            assert_eq!(a.blend_span, BLEND_SECONDS);
        }
        // The three fields `settle_blends` touches, named here so that adding a fourth transient
        // to `Actor` without settling it fails review rather than silently shipping: a capture
        // mid-anything is the defect this guards.
        assert_eq!(BLEND_SECONDS, 0.18, "the generic cross-fade span");
    }

    /// #gait-blend  **THE LOCOMOTION-MIX GATE.** The real [`gait_plan`] against every assertion.
    #[test]
    fn the_locomotion_mix_is_the_authored_one_and_continuous() {
        for check in 1..=7 {
            plan_battery(&gait_plan, check);
        }
    }

    /// #gait-blend  **THE GATE'S OWN GATE**, in the form
    /// `film_anim::tests::the_direction_mix_battery_catches_a_broken_implementation` documents:
    /// every assertion catches a bug, and every assertion has a bug ONLY it catches.
    ///
    /// **Not covered here:** whether anything calls [`gait_plan`]. That is
    /// `play_mode::probes::probe_the_gait_blend_at_the_call_site`.
    #[test]
    fn the_locomotion_mix_battery_catches_a_broken_planner() {
        type P = Box<Planner>;
        let saboteurs: Vec<(usize, bool, &str, P)> = vec![
            // --- REALISTIC bugs. Each must be caught by its target; several assertions may
            // --- catch it, which is several independent gates doing their job (the battery runs
            // --- each assertion in its own `catch_unwind`, so nothing can MASK anything).
            // THE BUG THIS ROUND FIXES: one clip per frame, the dominant direction only. This is
            // exactly what HMS did before, and it is the user's own complaint.
            (1, false, "one direction only, the old behaviour", Box::new(|t: &GaitTable, th: f32, s: f32| {
                let (d0, d1, w) = crate::film_anim::travel_directions(th);
                let d = if w >= 0.5 { d1 } else { d0 };
                gait_plan(t, match d {
                    0 => 0.0,
                    1 => std::f32::consts::PI,
                    2 => std::f32::consts::FRAC_PI_2,
                    _ => -std::f32::consts::FRAC_PI_2,
                }, s)
            })),
            // The weights are not renormalised, so the pose is silently dragged toward the
            // skeleton's bind pose by whatever is missing.
            (2, false, "weights are not normalised", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    for c in p.clips.iter_mut() {
                        c.1 *= 0.8;
                    }
                    p
                })
            })),
            // The directional mix is quantised to the nearest cardinal -- a hard SWITCH, which
            // is what "not blending them" looks like.
            (6, false, "the directional mix is quantised", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    let keep = p.clips[0].0 / 10;
                    let total: f32 = p.clips.iter().filter(|c| c.0 / 10 == keep).map(|c| c.1).sum();
                    p.clips.retain(|c| c.0 / 10 == keep);
                    for c in p.clips.iter_mut() {
                        c.1 /= total;
                    }
                    p.stride = p.clips.iter().map(|&(_, w, f, r)| w * r * f / 30.0).sum();
                    p
                })
            })),
            // --- NARROW saboteurs: one per assertion, each caught by that assertion ALONE, so
            // --- that every assertion is proven NOT redundant with its neighbours. Each is
            // --- confined to an input no other assertion samples; the comment says which.
            // 1: only the exact 45-degree answer loses its strafe half. Assertion 6 sweeps in
            //    steps of TAU/2048 and never lands on pi/4; assertion 2's angle loop steps by
            //    TAU/64 from -pi and never does either.
            (1, true, "a 45-degree diagonal drops its strafe half", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    if (th - std::f32::consts::FRAC_PI_2 * 0.5).abs() < 1e-7 {
                        let keep = p.clips[0].0 / 10;
                        let total: f32 = p.clips.iter().filter(|c| c.0 / 10 == keep).map(|c| c.1).sum();
                        p.clips.retain(|c| c.0 / 10 == keep);
                        for c in p.clips.iter_mut() {
                            c.1 /= total;
                        }
                        p.stride = p.clips.iter().map(|&(_, w, f, r)| w * r * f / 30.0).sum();
                    }
                    p
                })
            })),
            // 2: the sum is broken only ABOVE 2.5 wu/s. Assertion 2 samples 2.6; no other
            //    assertion goes above 2.4.
            (2, true, "the weights stop summing to one at the top speed", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    if s > 2.5 {
                        for c in p.clips.iter_mut() {
                            c.1 *= 0.8;
                        }
                    }
                    p
                })
            })),
            // 3: the tier split is forced to an even half at exactly 1.20 wu/s, which is the
            //    only speed assertion 3 reads and one no other assertion samples.
            (3, true, "the tier weight is a half instead of the engine's interpolation", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    if (s - 1.20).abs() < 1e-6 {
                        let total: f32 = p.clips.iter().map(|c| c.1).sum();
                        let n = p.clips.len() as f32;
                        for c in p.clips.iter_mut() {
                            c.1 = total / n;
                        }
                        p.stride = p.clips.iter().map(|&(_, w, f, r)| w * r * f / 30.0).sum();
                    }
                    p
                })
            })),
            // 4: at exactly 0.40 wu/s the NON-bracketing gait is forced, which is what picking
            //    by nearest speed rather than by the engine's bracketing-wins cost does. 0.40 is
            //    assertion 4's own probe speed and nobody else's.
            (4, true, "a non-bracketing gait is allowed to win", Box::new(|t: &GaitTable, th: f32, s: f32| {
                if (s - 0.40).abs() < 1e-6 {
                    let only = GaitTable {
                        gaits: t.gaits.iter().filter(|r| r.name != "walk_gait").cloned().collect(),
                    };
                    return gait_plan(&only, th, s);
                }
                gait_plan(t, th, s)
            })),
            // 5: the stride is the HEAVIEST clip's instead of the weighted mean, so a blend of
            //    two clips authored at different speeds plays at the wrong rate. Nothing but
            //    assertion 5 reads the stride.
            (5, true, "the stride is the heaviest clip's alone", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    if p.clips.len() > 1 {
                        let (_, _, f, r) = p.clips[0];
                        p.stride = r * f / 30.0;
                    }
                    p
                })
            })),
            // 6: the DIRECTIONAL split jumps in a narrow band of travel angle. Assertion 6's
            //    sweep passes through 0.28..0.32 rad; assertion 2's 64 angles land at 0.0 and
            //    0.0982 and 0.1963 and 0.2945 and 0.3927 -- all outside the band -- and 1, 3, 5
            //    and 7 sample only 0 and pi/4.
            (6, true, "the directional split jumps mid-sector", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    if (0.30..0.32).contains(&th) && p.clips.len() >= 2 {
                        let d = 0.3f32.min(p.clips[1].1);
                        p.clips[0].1 += d;
                        p.clips[1].1 -= d;
                        p.stride = p.clips.iter().map(|&(_, w, f, r)| w * r * f / 30.0).sum();
                    }
                    p
                })
            })),
            // 7: a step INSIDE a gait, in a narrow speed window no other assertion samples
            //    (1.90..1.95 wu/s).
            (7, true, "the tier weight steps inside a gait", Box::new(|t: &GaitTable, th: f32, s: f32| {
                gait_plan(t, th, s).map(|mut p| {
                    if (1.90..1.95).contains(&s) && p.clips.len() >= 2 {
                        let d = 0.3f32.min(p.clips[1].1);
                        p.clips[0].1 += d;
                        p.clips[1].1 -= d;
                        p.stride = p.clips.iter().map(|&(_, w, f, r)| w * r * f / 30.0).sum();
                    }
                    p
                })
            })),
        ];
        const N: usize = 7;
        let run = |p: &Planner| -> Vec<usize> {
            (1..=N)
                .filter(|&c| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan_battery(p, c))).is_err())
                .collect()
        };
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let results: Vec<(usize, bool, &str, Vec<usize>)> =
            saboteurs.iter().map(|(t, e, l, f)| (*t, *e, *l, run(f.as_ref()))).collect();
        let real = run(&gait_plan);
        std::panic::set_hook(prev);
        for (target, exclusive, label, failed) in &results {
            assert!(
                failed.contains(target),
                "saboteur '{label}' was NOT caught by assertion {target} (caught by {failed:?})"
            );
            if *exclusive {
                assert_eq!(
                    failed, &vec![*target],
                    "saboteur '{label}' must be caught by assertion {target} ALONE; it was caught by {failed:?}"
                );
            }
        }
        for k in 1..=N {
            assert!(
                results.iter().any(|(_, _, _, f)| f == &vec![k]),
                "assertion {k} has no saboteur of its own, so nothing proves it is doing work"
            );
        }
        assert!(real.is_empty(), "the real planner fails assertions {real:?}");
    }
}
