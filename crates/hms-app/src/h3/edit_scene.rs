//! `H3ObjectScene`: the `ObjectScene` surface (objscene.rs) over the pure-Rust Halo 3 reader, so
//! the editor verbs (select / gizmo / place / dup / delete / undo / box-select / wireframe) work on
//! a Halo 3 map exactly as on Reach / Halo 4. #h3-edit
//!
//! Models decode through scene.rs `decode_object_geom` (the static object path's decoder) into a
//! per-mode-tag cache; a rebuild groups the live objects by render model, uploads one draw per
//! part carrying every instance matrix through scene.rs `upload_model_parts` (same textures /
//! lanes / flat bake light as the static objects), and records a pick entry (world AABB +
//! transform) per object. Picking = AABB broad phase + triangle refine in model space;
//! `raycast_scene` runs over the BSP triangle soup the load collected (scene.rs
//! `H3EditorCollect`). A model-less object draws as the editor marker cube.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eframe::wgpu;
use glam::{Mat4, Quat, Vec3};
use hms_ipc::ObjectInfo;
use hms_render::{GpuMesh, MeshRenderer, MeshVertex};

use super::cache::H3Cache;
use crate::h4::cache::ByteRead;
use super::edit::{h3_tag_index, H3_MARKER_TAG};
use super::mvar::H3Variant;
use super::scene::{decode_model_geom, upload_model_parts, H3EditorCollect, H3GeomMesh, H3ModelGeom, Lane, TexCtx};
use crate::objscene::{ObjectScene, RebuiltLanes};
use crate::scene::{object_matrix, ray_aabb, raycast_mesh_local, signature, transform_aabb};

const MARKER_HALF: f32 = 0.3;

/// What the load hands the editor scene.
pub struct H3EditorAssets {
    pub cache: Arc<H3Cache>,
    pub variant: Option<(std::path::PathBuf, H3Variant)>,
    pub collect: H3EditorCollect,
    pub world_bounds: (Vec3, Vec3),
}

#[derive(Clone)]
struct PickEntry {
    datum: u32,
    min: Vec3,
    max: Vec3,
    mode_tag: u32,
    xform: Mat4,
    /// #h3-attach an hlmt variant CHILD (child obje, child variant, damage state) drawn under
    /// the parent's datum. #damage
    att: Option<(usize, Option<usize>, u8)>,
}

pub struct H3ObjectScene {
    cache: Arc<H3Cache>,
    variant: Option<(std::path::PathBuf, H3Variant)>,
    models: HashMap<u32, Option<Arc<H3ModelGeom>>>,
    /// Child-object geometry by (child obje, child variant, damage state). #h3-attach #damage
    att_models: HashMap<(usize, Option<usize>, u8), Option<Arc<H3ModelGeom>>>,
    /// #damage Geometry of a PLACED object in a damage display state, keyed by
    /// (obje, mode, hlmt variant, state 1..4). State 0 lives in `models`.
    state_models: HashMap<(usize, usize, Option<usize>, u8), Option<Arc<H3ModelGeom>>>,
    /// #damage Per-datum displayed damage state (the DAMAGE pseudo-flag; only non-default).
    /// A DISPLAY option: stored in the project file, never written to the .mvar.
    obj_damage: HashMap<u32, u8>,
    /// #damage The authored damage states per (obje, hlmt variant). Answering LOADS and parses
    /// the whole render model, and the object panel asks for the selected object's list EVERY
    /// frame, so the answer is memoised behind a `RefCell` (the query takes `&self`).
    dmg_states: std::cell::RefCell<HashMap<(usize, Option<usize>), Vec<u8>>>,
    tex: TexCtx,
    obj_color: [f32; 4],
    picks: Vec<PickEntry>,
    soup: crate::decal_projector::TriSoup,
    obj_scales: HashMap<u32, f32>,
    casters: HashSet<u32>,
    last_signature: u64,
    rebuild_pending: bool,
    world_bounds: (Vec3, Vec3),
    failed: Vec<String>,
}

impl H3ObjectScene {
    pub fn new(assets: H3EditorAssets, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let H3EditorAssets { cache, variant, collect, world_bounds } = assets;
        let tex = collect.tex.unwrap_or_else(|| TexCtx::new(cache.clone(), device, queue));
        let mut models: HashMap<u32, Option<Arc<H3ModelGeom>>> = HashMap::new();
        models.insert(H3_MARKER_TAG, Some(Arc::new(marker_geom())));
        H3ObjectScene { cache, variant, models, att_models: HashMap::new(), state_models: HashMap::new(), tex, obj_color: collect.obj_color, picks: Vec::new(), soup: collect.soup,
            obj_scales: HashMap::new(), casters: HashSet::new(), obj_damage: HashMap::new(), dmg_states: std::cell::RefCell::new(HashMap::new()),
            last_signature: 0, rebuild_pending: false, world_bounds, failed: Vec::new() }
    }
    pub fn cache(&self) -> &Arc<H3Cache> { &self.cache }
    pub fn variant(&self) -> Option<&(std::path::PathBuf, H3Variant)> { self.variant.as_ref() }
    pub fn set_variant(&mut self, v: Option<(std::path::PathBuf, H3Variant)>) { self.variant = v; }
    pub fn bounds(&self) -> Option<[f32; 6]> { self.variant.as_ref().map(|(_, v)| v.bounds) }
    pub fn pick_count(&self) -> usize { self.picks.len() }
    pub fn soup_len(&self) -> usize { self.soup.len() }
    pub fn model_count(&self) -> usize { self.models.values().filter(|m| m.is_some()).count() }

    /// #damage The damage states a placement can be displayed in: those its hlmt variant
    /// authors a permutation for, plus its variant attachments' (each child follows its own
    /// hlmt). Ascending, without 0; empty = the object has only a default state.
    pub fn object_damage_states(&self, o: &ObjectInfo) -> Vec<u8> {
        let Some(ot) = h3_tag_index(o.primary_tag) else { return Vec::new() };
        let pv = super::attach::object_variant_index(&self.cache, ot, 0);
        let mut out = self.variant_damage_states(ot, pv);
        for a in super::attach::expand_attachments(&self.cache, ot, pv, Mat4::IDENTITY) {
            for st in self.variant_damage_states(a.obje, a.variant) {
                if !out.contains(&st) { out.push(st); }
            }
        }
        out.sort_unstable();
        out
    }

    /// #damage `attach::variant_damage_states` through the per-(obje, variant) memo.
    pub fn variant_damage_states(&self, obje: usize, variant: Option<usize>) -> Vec<u8> {
        if let Some(v) = self.dmg_states.borrow().get(&(obje, variant)) { return v.clone(); }
        let v = super::attach::variant_damage_states(&self.cache, obje, variant);
        self.dmg_states.borrow_mut().insert((obje, variant), v.clone());
        v
    }

    /// Decode the models the objects need that are not cached yet (8 ms per frame, at least one).
    fn decode_budgeted(&mut self, objects: &[ObjectInfo]) -> bool {
        let t0 = std::time::Instant::now();
        let mut remaining = false;
        for o in objects {
            if o.mode_tag == 0 || self.models.contains_key(&o.mode_tag) { continue; }
            if t0.elapsed().as_millis() > 8 && self.models.len() > 1 { remaining = true; break; }
            let Some(mode) = h3_tag_index(o.mode_tag) else { self.models.insert(o.mode_tag, None); continue };
            match decode_model_geom(&self.cache, mode) {
                Ok(g) if g.tri_count() > 0 => { self.models.insert(o.mode_tag, Some(Arc::new(g))); }
                Ok(g) => { self.failed.push(format!("{}: no triangles", g.name)); self.models.insert(o.mode_tag, None); }
                Err(e) => { self.failed.push(format!("mode {mode}: {e}")); self.models.insert(o.mode_tag, None); }
            }
        }
        remaining
    }

    fn pick_refine(&self, p: &PickEntry, origin: Vec3, dir: Vec3) -> (Option<f32>, bool) {
        let inv = p.xform.inverse();
        let lo = inv.transform_point3(origin);
        let ld = inv.transform_vector3(dir);
        let geom = match &p.att { Some(k) => self.att_models.get(k), None => self.models.get(&p.mode_tag) };
        match geom {
            Some(Some(g)) if g.tri_count() > 0 => {
                let mut best: Option<f32> = None;
                for m in &g.meshes {
                    if let Some(t) = raycast_mesh_local(&m.verts, &m.indices, lo, ld) {
                        if best.map_or(true, |b| t < b) { best = Some(t); }
                    }
                }
                (best, true)
            }
            _ => (None, false),
        }
    }

    fn nearest_hit(&self, origin: Vec3, dir: Vec3, exclude: &[u32], min_t: f32) -> Option<(f32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for p in &self.picks {
            if exclude.contains(&p.datum) { continue; }
            let Some(aabb_t) = ray_aabb(origin, dir, p.min, p.max) else { continue };
            let (tri_t, has_mesh) = self.pick_refine(p, origin, dir);
            let t = match tri_t { Some(t) => t, None => { if has_mesh { continue; } aabb_t } };
            if t > min_t && best.map_or(true, |(bt, _)| t < bt) { best = Some((t, p.datum)); }
        }
        best
    }

    /// The palette preview: one synthetic placement of `obj_tag` at `at` through the same
    /// rebuild path (the Halo 4 preview's contract: Err(true) = still decoding, Err(false) = nothing).
    pub fn build_preview_meshes(&mut self, obj_tag: u32, _variant_sid: u32, at: Vec3, mesh: &MeshRenderer, device: &wgpu::Device, queue: &wgpu::Queue)
        -> Result<(Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec3, Vec3), bool> {
        let Some(obje) = h3_tag_index(obj_tag) else { return Err(false) };
        let mode = super::objects::model_of(&self.cache, obje).map(super::edit::h3_tag).unwrap_or(H3_MARKER_TAG);
        const PREVIEW_DATUM: u32 = 0xF2F0_0000;
        let objects = [ObjectInfo { datum: PREVIEW_DATUM, type_sig: 0, sig0: 0, sig1: 0, pos: [at.x, at.y, at.z], health: 1.0, shield: 1.0,
            mode_tag: mode, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], attached: [0; 8], primary_tag: obj_tag, variant_name_sid: 0 }];
        let colors: HashMap<u32, (u8, u8)> = HashMap::from([(PREVIEW_DATUM, (super::palette::TEAM_NEUTRAL, 0xFF))]);
        let saved_picks = std::mem::take(&mut self.picks);
        let saved_sig = self.last_signature;
        let saved_pending = self.rebuild_pending;
        self.rebuild_pending = true;
        let mut out = None;
        for _ in 0..8 {
            out = ObjectScene::maybe_rebuild(self, &objects, &colors, mesh, device, queue).or(out);
            if !self.rebuild_pending { break; }
        }
        let complete = !self.rebuild_pending;
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &self.picks { mn = mn.min(p.min); mx = mx.max(p.max); }
        self.picks = saved_picks;
        self.last_signature = saved_sig;
        self.rebuild_pending = saved_pending;
        if !complete { return Err(true); }
        let Some(l) = out else { return Err(false) };
        if mn.x > mx.x || (l.opaque.is_empty() && l.cutout.is_empty() && l.holo.is_empty() && l.holo_solid.is_empty() && l.blend.is_empty()) { return Err(false); }
        Ok((l.opaque, l.cutout, l.holo, l.holo_solid, l.blend, mn, mx))
    }
}

/// scnr +0x24C Player Starting Locations (0x18): position @0, facing yaw / pitch @0xC / @0x10
/// (radians as stored; the plugin's `degree2` displays them in degrees), bsp index i16 @0x14.
pub const OFF_SCNR_STARTING_LOCATIONS: usize = 0x24C;
pub const STARTING_LOCATION_ELEM: usize = 0x18;

/// The spawn-camera candidates of a Halo 3 map (spawn_camera.rs): the scenario's spawn
/// placements classified by tag path (facing = the placement's yaw), the loaded variant's
/// placed spawn objects (initial spawns first, with their team), and the scnr player starting
/// locations (stored yaw / pitch).
pub fn spawn_candidates(c: &H3Cache, variant: Option<&H3Variant>) -> Vec<crate::spawn_camera::Candidate> {
    use crate::spawn_camera::{classify_tag_path, Candidate, Kind};
    let mut out = Vec::new();
    for p in super::objects::load_placements(c) {
        if p.never_placed() { continue; }
        let Some(kind) = classify_tag_path(c.tag_name(p.palette_tag)) else { continue };
        let (fwd, _) = crate::h4::objects::euler_to_basis(p.rot);
        out.push(Candidate::new(kind, Vec3::from(p.pos), fwd));
    }
    if let Some(v) = variant {
        for o in v.placed() {
            let Some(tag) = v.quota_of(o).and_then(|q| super::mvar::quota_tag(c, q)) else { continue };
            let name = c.tag_name(tag).to_ascii_lowercase();
            let Some(base) = classify_tag_path(&name) else { continue };
            if !base.is_spawn() { continue; }
            let kind = if name.contains("initial") { Kind::VariantInitialSpawn } else { Kind::VariantSpawn };
            let team = if o.team < 8 { o.team as i8 } else { 8 };
            out.push(Candidate::new(kind, Vec3::from(o.pos), Vec3::from(o.fwd)).team(team));
        }
    }
    if let Some(sm) = c.find_tags(b"scnr").first().and_then(|&t| c.tag_meta(t)) {
        if let Some((n, o)) = c.block(sm + OFF_SCNR_STARTING_LOCATIONS) {
            let d = c.data();
            for i in 0..n.min(64) {
                let e = o + i * STARTING_LOCATION_ELEM;
                let pos = Vec3::new(d.f32_at(e), d.f32_at(e + 4), d.f32_at(e + 8));
                let (mut yaw, mut pitch) = (d.f32_at(e + 0xC), d.f32_at(e + 0x10));
                if yaw.abs() > 7.0 || pitch.abs() > 7.0 { yaw = yaw.to_radians(); pitch = pitch.to_radians(); }
                if !pos.is_finite() || !yaw.is_finite() || !pitch.is_finite() { continue; }
                out.push(Candidate::new(Kind::StartingLocation, pos, Vec3::new(yaw.cos(), yaw.sin(), 0.0)).pitch(pitch));
            }
        }
    }
    out
}

/// The spawn camera of a Halo 3 map through the shared rule (`spawn_camera::choose`).
pub fn spawn_camera_for(c: &H3Cache, variant: Option<&H3Variant>, playable: Option<(Vec3, Vec3)>, scene_bounds: Option<(Vec3, Vec3)>, raycast: Option<crate::spawn_camera::Raycast>) -> Option<crate::spawn_camera::Pose> {
    let cands = spawn_candidates(c, variant);
    let vols = crate::spawn_camera::Volumes { playable_bounds: playable, scene_bounds, ..Default::default() };
    crate::spawn_camera::choose(&cands, &vols, raycast)
}

impl H3ObjectScene {
    /// The spawn camera over this scene's candidates, stood off the BSP soup + the placed
    /// objects (`raycast_solid`). `playable` = the framed cluster bounds when the load has them.
    pub fn spawn_camera_framed(&self, playable: Option<(Vec3, Vec3)>) -> Option<crate::spawn_camera::Pose> {
        let variant = self.variant.as_ref().map(|(_, v)| v);
        let scene = self.scene_bounds().map(|(a, b)| (Vec3::from(a), Vec3::from(b)));
        let rc = |o: Vec3, d: Vec3, m: f32| ObjectScene::raycast_solid(self, o, d, m);
        let raycast: Option<crate::spawn_camera::Raycast> = if self.soup.len() > 0 { Some(&rc) } else { None };
        spawn_camera_for(&self.cache, variant, playable, scene, raycast)
    }
}

/// The unit holo cube: a `MARKER_HALF` half-extent box standing on the object origin, one part
/// with no material (drawn on the blend lane).
fn marker_geom() -> H3ModelGeom {
    let h = MARKER_HALF;
    let (z0, z1) = (0.0, 2.0 * h);
    let corners = [[-h, -h, z0], [h, -h, z0], [h, h, z0], [-h, h, z0], [-h, -h, z1], [h, -h, z1], [h, h, z1], [-h, h, z1]];
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 3, 2, 1], [0.0, 0.0, -1.0]), ([4, 5, 6, 7], [0.0, 0.0, 1.0]), ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([2, 3, 7, 6], [0.0, 1.0, 0.0]), ([1, 2, 6, 5], [1.0, 0.0, 0.0]), ([3, 0, 4, 7], [-1.0, 0.0, 0.0]),
    ];
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (f, n) in &faces {
        let base = verts.len() as u32;
        for (k, &ci) in f.iter().enumerate() {
            let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][k];
            verts.push(MeshVertex { pos: corners[ci], normal: *n, uv, color: [1.0; 4], ..Default::default() });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    H3ModelGeom {
        name: "editor marker cube".into(),
        materials: vec![None],
        meshes: vec![H3GeomMesh { verts, indices, parts: vec![super::geometry::Part { material: 0, index_start: 0, index_count: 36 }] }],
        min: Vec3::new(-h, -h, z0),
        max: Vec3::new(h, h, z1),
    }
}

impl ObjectScene for H3ObjectScene {
    fn has_cache(&self) -> bool { true }
    fn invalidate(&mut self) { self.last_signature = 0; }
    fn rebuild_pending(&self) -> bool { self.rebuild_pending }

    fn maybe_rebuild(&mut self, objects: &[ObjectInfo], colors: &HashMap<u32, (u8, u8)>, mr: &MeshRenderer, _device: &wgpu::Device, _queue: &wgpu::Queue) -> Option<RebuiltLanes> {
        let scale_sig = self.obj_scales.iter().fold(0u64, |acc, (d, s)| acc ^ (*d as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(s.to_bits() as u64));
        // #damage a displayed damage state swaps the object's permutations -> rebuild.
        let damage_sig = self.obj_damage.iter().fold(0u64, |acc, (d, s)| acc ^ ((*d as u64) << 3 | *s as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
        let sig = signature(objects, colors) ^ scale_sig ^ damage_sig;
        if sig == self.last_signature && !self.rebuild_pending { return None; }
        self.last_signature = sig;
        let n_failed = self.failed.len();
        self.rebuild_pending = self.decode_budgeted(objects);
        for f in &self.failed[n_failed..] { log::warn!("h3 editor model: {f}"); }
        // group by (render model, damage display state), first-seen order. #damage state 0 draws
        // the mode tag's own geometry (`models`); a damaged state draws a (obje, mode, variant,
        // state) entry of `state_models`, so two placements of one vehicle in different states
        // are two draws.
        let mut by_mode: HashMap<(u32, u64), Vec<Mat4>> = HashMap::new();
        let mut order: Vec<(u32, u64)> = Vec::new();
        let mut geom_by_key: HashMap<(u32, u64), Arc<H3ModelGeom>> = HashMap::new();
        let mut att: HashMap<(usize, Option<usize>, u8), (Vec<Mat4>, u32)> = HashMap::new();
        let mut att_order: Vec<(usize, Option<usize>, u8)> = Vec::new();
        self.picks.clear();
        for o in objects {
            if o.mode_tag == 0 || o.mode_tag == 0xFFFF_FFFF { continue; }
            let Some(Some(geom)) = self.models.get(&o.mode_tag).cloned() else { continue };
            let s = self.obj_scales.get(&o.datum).copied().unwrap_or(1.0);
            let m = if (s - 1.0).abs() > 1e-4 { object_matrix(o) * Mat4::from_scale(Vec3::splat(s)) } else { object_matrix(o) };
            let obje = h3_tag_index(o.primary_tag);
            let pv = obje.and_then(|ot| super::attach::object_variant_index(&self.cache, ot, 0));
            // #damage the DAMAGE pseudo-flag: when this object's hlmt variant authors the state,
            // draw its permutations from a model decoded in that state.
            let st = self.obj_damage.get(&o.datum).copied().unwrap_or(0);
            let mut sk = 0u64;
            let mut sgeom: Option<Arc<H3ModelGeom>> = None;
            if st > 0 {
                if let (Some(ot), Some(mi)) = (obje, h3_tag_index(o.mode_tag)) {
                    if self.variant_damage_states(ot, pv).contains(&st) {
                        let cache = &self.cache;
                        let failed = &mut self.failed;
                        let entry = self.state_models.entry((ot, mi, pv, st)).or_insert_with(|| {
                            match super::scene::decode_object_state_geom(cache, ot, pv, st) {
                                Ok(g) if g.tri_count() > 0 => Some(Arc::new(g)),
                                Ok(_) => None,
                                Err(e) => { failed.push(format!("{} (damage state {st}): {e}", cache.tag_name(mi))); None }
                            }
                        });
                        if let Some(g) = entry {
                            sgeom = Some(g.clone());
                            sk = (ot as u64) << 24 | (pv.map_or(0, |v| v as u64 + 1)) << 8 | st as u64;
                        }
                    }
                }
            }
            let key = (o.mode_tag, sk);
            by_mode.entry(key).or_insert_with(|| { order.push(key); Vec::new() }).push(m);
            if let Some(g) = &sgeom { geom_by_key.entry(key).or_insert_with(|| g.clone()); }
            let g: &H3ModelGeom = sgeom.as_deref().unwrap_or(&geom);
            let (mn, mx) = transform_aabb(g.min, g.max, &m);
            self.picks.push(PickEntry { datum: o.datum, min: mn, max: mx, mode_tag: o.mode_tag, xform: m, att: None });
            // #h3-attach the object's hlmt variant children at their markers, picked under the parent
            if let Some(obje) = obje {
                for a in super::attach::expand_attachments(&self.cache, obje, pv, m) {
                    // #damage a child follows the parent's state when its own hlmt authors one
                    let ast = if st > 0 && self.variant_damage_states(a.obje, a.variant).contains(&st) { st } else { 0 };
                    let key = (a.obje, a.variant, ast);
                    att.entry(key).or_insert_with(|| { att_order.push(key); (Vec::new(), o.datum) }).0.push(a.world);
                }
            }
        }
        // decode the children's geometry (cached by (obje, variant)), record their picks
        for key in &att_order {
            if !self.att_models.contains_key(key) {
                let g = super::scene::decode_object_state_geom(&self.cache, key.0, key.1, key.2).ok().filter(|g| g.tri_count() > 0).map(Arc::new);
                self.att_models.insert(*key, g);
            }
            let Some(Some(g)) = self.att_models.get(key) else { continue };
            let (mats, datum) = &att[key];
            for m in mats {
                let (mn, mx) = transform_aabb(g.min, g.max, m);
                self.picks.push(PickEntry { datum: *datum, min: mn, max: mx, mode_tag: 0, xform: *m, att: Some(*key) });
            }
        }
        let mut lanes = RebuiltLanes { opaque: Vec::new(), cutout: Vec::new(), holo: Vec::new(), holo_solid: Vec::new(), blend: Vec::new() };
        let color = self.obj_color;
        for key in order {
            let tag = key.0;
            // #damage a damaged placement draws the state's own decoded geometry
            let Some(geom) = geom_by_key.get(&key).cloned().or_else(|| self.models.get(&tag).cloned().flatten()) else { continue };
            let mats = &by_mode[&key];
            // the marker cube: a translucent placeholder (blend lane, half alpha)
            let color = if tag == H3_MARKER_TAG { [0.9, 0.9, 0.2, 0.5] } else { color };
            let mut out = |lane: Lane, gm: GpuMesh| match lane {
                Lane::Opaque | Lane::Water | Lane::Sky | Lane::Terrain => lanes.opaque.push(gm),
                Lane::AlphaTest => lanes.cutout.push(gm),
                Lane::Blend => lanes.blend.push(gm),
                Lane::Additive => lanes.holo.push(gm),
            };
            if tag == H3_MARKER_TAG {
                let (device, queue) = (_device, _queue);
                for m in &geom.meshes {
                    let verts: Vec<MeshVertex> = m.verts.iter().map(|v| MeshVertex { color, ..*v }).collect();
                    lanes.blend.push(super::scene::upload_part(mr, device, queue, &verts, &m.indices, mats, None));
                }
            } else {
                upload_model_parts(&mut self.tex, mr, &geom, mats, color, &mut out);
            }
        }
        for key in att_order {
            let Some(Some(geom)) = self.att_models.get(&key).cloned() else { continue };
            let (mats, _) = &att[&key];
            let mut out = |lane: Lane, gm: GpuMesh| match lane {
                Lane::Opaque | Lane::Water | Lane::Sky | Lane::Terrain => lanes.opaque.push(gm),
                Lane::AlphaTest => lanes.cutout.push(gm),
                Lane::Blend => lanes.blend.push(gm),
                Lane::Additive => lanes.holo.push(gm),
            };
            upload_model_parts(&mut self.tex, mr, &geom, mats, color, &mut out);
        }
        Some(lanes)
    }

    fn set_object_scales(&mut self, scales: HashMap<u32, f32>) { self.obj_scales = scales; }
    fn set_object_casters(&mut self, casters: HashSet<u32>) { self.casters = casters; }
    /// #damage The DAMAGE display option (project-file state; never written to the .mvar).
    fn set_object_damage(&mut self, states: HashMap<u32, u8>) { self.obj_damage = states; }
    fn damage_states_of(&self, o: &ObjectInfo) -> Vec<u8> { self.object_damage_states(o) }

    fn pick(&self, origin: Vec3, dir: Vec3) -> Option<u32> { self.nearest_hit(origin, dir, &[], f32::NEG_INFINITY).map(|(_, d)| d) }

    /// The union of every entry of the datum (the object + its attachments). #h3-attach
    fn aabb_of(&self, datum: u32) -> Option<([f32; 3], [f32; 3])> {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        let mut any = false;
        for p in self.picks.iter().filter(|p| p.datum == datum) {
            if !(p.min.is_finite() && p.max.is_finite()) { continue; }
            mn = mn.min(p.min); mx = mx.max(p.max); any = true;
        }
        any.then(|| (mn.into(), mx.into()))
    }

    fn object_obb(&self, datum: u32) -> Option<crate::construct::Obb> {
        let p = self.picks.iter().find(|p| p.datum == datum && p.att.is_none()).or_else(|| self.picks.iter().find(|p| p.datum == datum))?;
        let Some(Some(g)) = self.models.get(&p.mode_tag).filter(|_| p.att.is_none()) else { return Some(crate::construct::Obb::from_aabb(p.min, p.max)) };
        if g.tri_count() == 0 { return Some(crate::construct::Obb::from_aabb(p.min, p.max)); }
        let (mut lmn, mut lmx) = (g.min, g.max);
        // grow the local box by every attachment's world box (the turret is inside the gizmo box)
        let inv = p.xform.inverse();
        for a in self.picks.iter().filter(|q| q.datum == datum && q.att.is_some()) {
            if !(a.min.is_finite() && a.max.is_finite()) { continue; }
            for k in 0..8u32 {
                let cnr = Vec3::new(if k & 4 != 0 { a.max.x } else { a.min.x }, if k & 2 != 0 { a.max.y } else { a.min.y }, if k & 1 != 0 { a.max.z } else { a.min.z });
                let l = inv.transform_point3(cnr);
                if l.is_finite() { lmn = lmn.min(l); lmx = lmx.max(l); }
            }
        }
        Some(crate::construct::Obb::from_local(&p.xform, lmn, lmx))
    }

    /// Every entry of the datum: the model's triangle edges plus each attachment's (attachments
    /// first, half the budget reserved for them). #h3-attach
    fn selection_wireframe(&self, datum: u32) -> Option<Vec<[f32; 3]>> {
        const MAX_LINES: usize = 60_000;
        const ATT_LINES: usize = MAX_LINES / 2;
        let mut out: Vec<[f32; 3]> = Vec::new();
        let mut saw = false;
        let mut order: Vec<&PickEntry> = self.picks.iter().filter(|p| p.datum == datum).collect();
        order.sort_by_key(|p| p.att.is_none());
        for p in order {
            saw = true;
            let cap = if p.att.is_some() { ATT_LINES } else { MAX_LINES };
            let geom = match &p.att { Some(k) => self.att_models.get(k), None => self.models.get(&p.mode_tag) };
            let Some(Some(g)) = geom else { continue };
            'meshes: for m in &g.meshes {
                for tri in m.indices.chunks_exact(3) {
                    if out.len() >= cap * 2 { break 'meshes; }
                    let (Some(va), Some(vb), Some(vc)) = (m.verts.get(tri[0] as usize), m.verts.get(tri[1] as usize), m.verts.get(tri[2] as usize)) else { continue };
                    let wa = p.xform.transform_point3(Vec3::from(va.pos)).to_array();
                    let wb = p.xform.transform_point3(Vec3::from(vb.pos)).to_array();
                    let wc = p.xform.transform_point3(Vec3::from(vc.pos)).to_array();
                    out.extend_from_slice(&[wa, wb, wb, wc, wc, wa]);
                }
            }
        }
        if !saw { return None; }
        (!out.is_empty()).then_some(out)
    }

    fn translate_pick(&mut self, datum: u32, delta: Vec3) {
        for p in self.picks.iter_mut().filter(|p| p.datum == datum) {
            p.min += delta;
            p.max += delta;
            p.xform = Mat4::from_translation(delta) * p.xform;
        }
    }

    fn rotate_pick(&mut self, datum: u32, q: Quat, pivot: Vec3) {
        let m = Mat4::from_translation(pivot) * Mat4::from_quat(q) * Mat4::from_translation(-pivot);
        for p in self.picks.iter_mut().filter(|p| p.datum == datum) { p.xform = m * p.xform; }
    }

    fn raycast_scene(&self, origin: Vec3, dir: Vec3) -> Option<Vec3> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        self.soup.raycast(origin, d, 1.0e5).map(|(t, _)| origin + d * t)
    }

    fn raycast_scene_n(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        self.soup.raycast(origin, d, 1.0e5).map(|(t, n)| (origin + d * t, n))
    }

    /// #play  Level geometry for the physics sampler and TEST MODE, out of the same BSP triangle
    /// soup the load already collected for raycasting and decals.
    ///
    /// **This was missing, and the trait's empty default hid it.** `static_tris_in_aabb` is what
    /// test mode collides the player's capsule against, so a scene that does not answer it is a
    /// frictionless void: walking a Halo 3 map would have dropped the player straight through the
    /// floor with nothing to say why. Reach and Halo 4 both override it; Halo 3 had the soup all
    /// along and simply never wired it up. The default has since been removed from the trait so
    /// the next game cannot make the same omission silently.
    fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        self.soup.tris_in_aabb(mn, mx, out);
    }

    fn raycast_objects_excluding(&self, origin: Vec3, dir: Vec3, exclude: &[u32]) -> Option<Vec3> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        self.nearest_hit(origin, d, exclude, 1e-4).map(|(t, _)| origin + d * t)
    }

    fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let (a, b) = self.world_bounds;
        (a.x <= b.x).then(|| (a.into(), b.into()))
    }

    fn resolve_object_mode(&self, obj_tag: u32) -> u32 {
        h3_tag_index(obj_tag).and_then(|t| super::objects::model_of(&self.cache, t)).map(super::edit::h3_tag).unwrap_or(H3_MARKER_TAG)
    }

    fn tag_name_of(&self, tag: u32) -> String {
        if tag == H3_MARKER_TAG { return "editor marker cube".into(); }
        h3_tag_index(tag).map(|t| self.cache.tag_name(t).to_string()).unwrap_or_default()
    }

    fn raw_tag_class(&self, tag: u32) -> Option<String> {
        h3_tag_index(tag).and_then(|t| self.cache.tag_class(t)).map(|c| String::from_utf8_lossy(&c).into_owned())
    }

    fn blocker_overlay(&self) -> (&[([f32; 3], [f32; 4])], &[([f32; 3], [f32; 3], [f32; 3])]) { (&[], &[]) }
    fn collision_world_edges(&self, _o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
    fn physics_world_edges(&self, _o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> { Vec::new() }
    fn pick_entry_counts(&self, datum: u32) -> (usize, usize) {
        let all: Vec<&PickEntry> = self.picks.iter().filter(|p| p.datum == datum).collect();
        (all.len(), all.iter().filter(|p| p.att.is_some()).count())
    }

    fn bsp_flags_report(&self) -> String {
        match &self.variant {
            Some((p, v)) => format!("Halo 3 variant {}: bounds x [{:.1}, {:.1}] y [{:.1}, {:.1}] z [{:.1}, {:.1}], {} of 640 slots, {} quotas, budget {} / {}",
                p.file_name().unwrap_or_default().to_string_lossy(), v.bounds[0], v.bounds[1], v.bounds[2], v.bounds[3], v.bounds[4], v.bounds[5], v.num_variant_objects, v.quotas.len(), v.spent_budget, v.maximum_budget),
            None => "Halo 3 map: no variant open".into(),
        }
    }

    fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> { self.spawn_camera().map(|p| p.tuple()) }
    fn spawn_camera(&self) -> Option<crate::spawn_camera::Pose> { self.spawn_camera_framed(None) }
}
