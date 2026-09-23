//! `H4ObjectScene`: the Halo 4 implementation of the editor's `ObjectScene` surface.
//!
//! The Reach editor talks to `SceneController` for exactly one thing per verb: turn the current
//! `ObjectInfo` list into dynamic-lane `GpuMesh`es + a pick list, and answer pick / bounds /
//! wireframe / raycast queries from that pick list. This type does the same over the pure-Rust
//! Halo 4 reader:
//!
//! * `maybe_rebuild` mirrors `SceneController::maybe_rebuild` (scene.rs): signature of the object
//!   set + scales, an 8 ms/frame budgeted decode of NEW lod0 models (`decode_model_geom`, cached
//!   per mode tag, `rebuild_pending` re-runs next frame), instances grouped per mode tag with
//!   `object_matrix(o) * scale`, uploaded through the SAME `upload_model_parts` the static object
//!   path uses (so an object draws byte-identically whether it is static or editable), lit by the
//!   same `ObjectLightField` (per-instance colour lane). Lanes: Opaque -> opaque, AlphaTest ->
//!   cutout, Blend -> blend, Additive -> holo (the renderer's additive dynamic pass).
//! * `pick` / `raycast_objects_excluding` / `aabb_of` / `object_obb` / `selection_wireframe` /
//!   `translate_pick` / `rotate_pick` are the Reach bodies over the decoded model geometry
//!   (`scene::ray_aabb` / `raycast_mesh_local` / `transform_aabb` shared).
//! * `raycast_scene` / `raycast_scene_n` run over the `TriSoup` `build_meshes` collected (every
//!   drawn BSP triangle, world space), the Reach `bsp_soup` analogue.
//! * Palette entries WITHOUT a render model (marker types: spawn zones, kill areas, ...) get the
//!   unit holo cube `H4_MARKER_TAG` so they still get a `PickEntry` and can be moved.
//!
//! Tag ids: `h4::edit::h4_tag(idx)` (`0x4800_0000 | idx`) because cache index 0 is a valid tag and
//! the editor treats 0 / 0xFFFF_FFFF as "no model". Datums: `0xD000_0000 + i` for loaded variant
//! objects, `0xD8..` for new ones (the Reach scheme, unchanged).
//!
//! `h4/edit.rs::variant_to_editor` owns the full editor record conversion; the
//! `variant_editor_objects` here is the minimal `ObjectInfo` form the render / pick layer needs
//! (the headless screenshot path).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use eframe::wgpu;
use glam::{Mat4, Quat, Vec3};
use hms_ipc::ObjectInfo;
use hms_render::{GpuMesh, MeshRenderer, MeshVertex};

use super::cache::H4Cache;
use super::collision::{self, HullGeom};
use super::edit::{h4_tag, h4_tag_index, mode_tag_of_obje, H4_TAG_FLAG};
use super::geometry::Part;
use super::materials::H4Material;
use super::mvar::{H4PaletteEntry, H4Variant};
use super::objects::H4Placement;
use super::scene::{decode_model_geom, decode_object_geom, upload_model_parts, H4EditorAssets, H4GeomMesh, H4ModelGeom, H4Stats, Lane, NEUTRAL_CC, ObjectLightField, TexCtx};
use crate::objscene::{ObjectScene, RebuiltLanes};
use crate::scene::{object_matrix, ray_aabb, raycast_mesh_local, signature, transform_aabb};

/// The "no render model" marker: a palette entry whose obje tag has no `mode` (spawn zones, kill
/// areas, ...) renders as a translucent unit cube so it is still pickable / movable.
/// The index part (0xFFFF) can never be a real cache tag (Halo 4 caches hold ~22k tags).
pub const H4_MARKER_TAG: u32 = H4_TAG_FLAG | 0xFFFF;

/// Marker cube half-extent (world units) and the height its base sits above the object origin.
const MARKER_HALF: f32 = 0.3;

/// Per-frame decode budget (ms) - the Reach value (scene.rs `FRAME_DECODE_MS`).
const FRAME_DECODE_MS: u128 = 8;

/// #h4-phys The hidden-block cutoff, the Reach numbers: a render model whose largest visible
/// extent is under this draws nothing in game (every `nut_blocker` reads 0.10; the smallest
/// genuinely visible Forge block is ~0.67), ...
const BLOCKER_RENDER_MAX: f32 = 0.25;
/// ... and the hull must have real volume before it is worth drawing.
const BLOCKER_HULL_MIN: f32 = 0.25;
/// The hidden-block overlay colours (identical to Reach's, so the two games look the same).
const BLOCKER_SOLID: [f32; 4] = [1.0, 0.55, 0.15, 0.28];
const BLOCKER_LINE: [f32; 3] = [1.0, 0.62, 0.20];

/// One placed object's cached pick data (the Reach `scene.rs::PickEntry`).
#[derive(Clone)]
struct PickEntry {
    datum: u32,
    min: Vec3,
    max: Vec3,
    mode_tag: u32,
    xform: Mat4,
    /// #h4-phys A HIDDEN BLOCK's physics hull. Set only for the `nut_blocker`-style pieces whose
    /// render model draws nothing: their box / bounds / wireframe / pick refine all read the hull
    /// instead of the 0.1 wu nub, exactly as the Reach `PickEntry::hull` does.
    hull: Option<Arc<HullGeom>>,
    /// #wire-attach An ATTACHMENT entry: the hlmt-variant child object (a Wraith's mortar, a
    /// rocket Warthog's turret) posed on its parent's marker. It carries the PARENT's datum, so a
    /// click on the turret selects the vehicle, and the parent's bounds / oriented box / selection
    /// wireframe include it. `None` = the object's own entry.
    att: Option<(usize, usize, Option<usize>)>,
}

pub struct H4ObjectScene {
    cache: Arc<H4Cache>,
    palette: Vec<H4PaletteEntry>,
    variant: Option<(std::path::PathBuf, H4Variant)>,
    placements: Vec<H4Placement>,
    /// Decoded lod0 model per editor mode tag (None = load / decode failed, never retried).
    models: HashMap<u32, Option<Arc<H4ModelGeom>>>,
    /// #h4-veh Decoded hlmt-variant CHILD object models, keyed by (child obje, child mode, child
    /// hlmt variant) so each turret keeps its own materials / permutations.
    att_models: HashMap<(usize, usize, Option<usize>), Option<Arc<H4ModelGeom>>>,
    tex: TexCtx,
    obj_mats: HashMap<usize, H4Material>,
    stats: H4Stats,
    shading_on: bool,
    picks: Vec<PickEntry>,
    /// #h4-phys `coll` / `phmo` hulls per obje cache tag, decoded on demand (the overlays are
    /// `&self` calls, so the cache is a `RefCell`). `None` = the tag resolves no such model.
    hulls: RefCell<HashMap<usize, (Option<Arc<HullGeom>>, Option<Arc<HullGeom>>)>>,
    /// #h4-phys The always-on hidden-block overlay, rebuilt with the picks.
    blocker_tris: Vec<([f32; 3], [f32; 4])>,
    blocker_lines: Vec<([f32; 3], [f32; 3], [f32; 3])>,
    soup: crate::decal_projector::TriSoup,
    light: ObjectLightField,
    obj_scales: HashMap<u32, f32>,
    casters: HashSet<u32>,
    last_signature: u64,
    rebuild_pending: bool,
    world_bounds: ([f32; 3], [f32; 3]),
    /// Names of models that failed (reported once).
    failed: Vec<String>,
    /// The static scenario casters' AABB (build_meshes) + the BSP's cascade config (sun shadows).
    static_caster_bounds: Option<(Vec3, Vec3)>,
    cascade: Option<hms_render::H4CascadeCfg>,
    /// Direction TO the sun (the object sun-visibility raycasts).
    sun_to: Vec3,
}

impl H4ObjectScene {
    /// Build the editor scene from a finished load. The marker cube is decoded here so a scene
    /// never lacks it; nothing else is decoded until `maybe_rebuild` asks.
    pub fn new(assets: H4EditorAssets, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let H4EditorAssets { cache, palette, variant, collect, placements, world_bounds, static_caster_bounds, cascade, sun_to } = assets;
        let tex = collect.tex.unwrap_or_else(|| TexCtx::new(cache.clone(), device, queue));
        let mut models = HashMap::new();
        models.insert(H4_MARKER_TAG, Some(Arc::new(marker_geom())));
        H4ObjectScene {
            cache, palette, variant, placements, models, tex,
            att_models: HashMap::new(),
            obj_mats: HashMap::new(),
            stats: H4Stats::default(),
            shading_on: super::scene::shading_requested(),
            picks: Vec::new(),
            hulls: RefCell::new(HashMap::new()),
            blocker_tris: Vec::new(),
            blocker_lines: Vec::new(),
            soup: collect.soup,
            light: collect.light,
            obj_scales: HashMap::new(),
            casters: HashSet::new(),
            last_signature: 0,
            rebuild_pending: false,
            world_bounds,
            failed: Vec::new(),
            static_caster_bounds,
            cascade,
            sun_to,
        }
    }

    /// A scene WITHOUT GPU resources (unit tests: pick / bounds / wireframe over decoded models).
    #[cfg(test)]
    pub(crate) fn new_offline(cache: Arc<H4Cache>, palette: Vec<H4PaletteEntry>, variant: Option<(std::path::PathBuf, H4Variant)>) -> Option<Self> {
        // A headless wgpu device (any adapter, the software one is fine) - only `TexCtx::new`
        // wants one; no texture is ever uploaded by the offline paths. None = no adapter at all
        // (the tests skip, like they do without the MCC files).
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::LowPower, compatible_surface: None, force_fallback_adapter: false }))
            .or_else(|| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::LowPower, compatible_surface: None, force_fallback_adapter: true })));
        let Some(adapter) = adapter else { eprintln!("skip: no wgpu adapter"); return None };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("h4-edit-offline"), required_features: wgpu::Features::empty(), required_limits: adapter.limits(), memory_hints: wgpu::MemoryHints::MemoryUsage }, None)).ok()?;
        let assets = H4EditorAssets { cache, palette, variant, collect: super::scene::H4EditorCollect::new(), placements: Vec::new(), world_bounds: ([0.0; 3], [0.0; 3]), static_caster_bounds: None, cascade: None, sun_to: super::scene::placeholder_sun_dir() };
        Some(Self::new(assets, &device, &queue))
    }

    pub fn cache(&self) -> &Arc<H4Cache> { &self.cache }
    pub fn palette(&self) -> &[H4PaletteEntry] { &self.palette }
    pub fn variant(&self) -> Option<&(std::path::PathBuf, H4Variant)> { self.variant.as_ref() }
    /// Swap the open variant on the SAME map (bounds / spawn camera read the new one).
    pub fn set_variant(&mut self, v: Option<(std::path::PathBuf, H4Variant)>) { self.variant = v; }
    pub fn placements(&self) -> &[H4Placement] { &self.placements }
    /// The variant's world bounds (xmin xmax ymin ymax zmin zmax) for the out-of-bounds gate.
    pub fn bounds(&self) -> Option<[f32; 6]> { self.variant.as_ref().map(|(_, v)| v.bounds) }
    pub fn pick_count(&self) -> usize { self.picks.len() }
    /// Print-only lighting diagnostic: objects lit by (surface probe, airprobe, instance probe, mean bake).
    pub fn light_source_tally(&self) -> [u32; 4] { self.light.stats.lock().map(|s| *s).unwrap_or_default() }
    pub fn model_count(&self) -> usize { self.models.values().filter(|m| m.is_some()).count() }
    pub fn soup_len(&self) -> usize { self.soup.len() }
    /// The decoded model of an editor mode tag (after a rebuild asked for it).
    pub fn model_geom(&self, mode_tag: u32) -> Option<&H4ModelGeom> { self.models.get(&mode_tag).and_then(|m| m.as_deref()) }

    /// Editor mode tag of an obje tag (`edit::mode_tag_of_obje` over this scene's cache).
    pub fn mode_tag_of_obje(&self, obje: usize) -> u32 { mode_tag_of_obje(&self.cache, obje) }

    /// Resolve a variant object's (quota, variant) through the palette -> (obje tag, editor
    /// mode tag, "entry:variant" name). None when the quota / variant is out of range or the
    /// entry's tag is null (the record then occupies a slot invisibly, like Reach's unresolved).
    /// Unlike `edit::resolve_palette_item` an obje tag the cache does not know still resolves,
    /// with mode tag 0.
    pub fn resolve_item(&self, quota: u8, variant: Option<u8>) -> Option<(usize, u32, String)> {
        let entry = self.palette.get(quota as usize)?;
        let pv = entry.variants.get(variant.unwrap_or(0) as usize)?;
        let tag = pv.tag?;
        Some((tag, self.mode_tag_of_obje(tag), format!("{}:{}", entry.name, pv.name)))
    }

    /// The loaded variant's objects as editor `ObjectInfo`s (datum `0xD000_0000 + i`, tags
    /// encoded, fwd/up from the record) + the (team, colour) map - the minimal form the render /
    /// pick layer needs. Unresolvable records are skipped (`variant_to_editor` keeps them as
    /// unresolved source records).
    pub fn variant_editor_objects(&self) -> (Vec<ObjectInfo>, HashMap<u32, (u8, u8)>) {
        let mut objects = Vec::new();
        let mut colors = HashMap::new();
        let Some((_, v)) = &self.variant else { return (objects, colors) };
        for (i, o) in v.objects.iter().enumerate() {
            let Some(q) = o.quota else { continue };
            let Some((obje, mode_tag, _)) = self.resolve_item(q, o.variant) else { continue };
            let datum = 0xD000_0000u32 + i as u32;
            objects.push(ObjectInfo {
                datum, type_sig: 0, sig0: 0, sig1: 0, pos: o.pos, health: 1.0, shield: 1.0,
                mode_tag, fwd: o.fwd, up: o.up, attached: [0; 8], primary_tag: h4_tag(obje), variant_name_sid: 0,
            });
            colors.insert(datum, (o.team as u8, o.color.unwrap_or(0xFF)));
        }
        (objects, colors)
    }

    /// The editor mode tag the preview would draw for `obj_tag` - the marker cube for an object
    /// with no render model - or None when there is nothing to show (unknown tag).
    pub fn preview_mode_tag(&self, obj_tag: u32) -> Option<u32> {
        let mode = ObjectScene::resolve_object_mode(self, obj_tag);
        (mode != 0 && mode != 0xFFFF_FFFF).then_some(mode)
    }

    /// PREVIEW PARITY (#h4-preview): build the palette preview's meshes through the SAME
    /// `maybe_rebuild` object path the viewport uses (same `decode_model_geom` cache, same
    /// per-material lanes / `upload_model_parts` routing, same `ObjectLightField` probe lighting
    /// sampled at `at` and the same sun-visibility raycast), for ONE synthetic placement of
    /// `obj_tag` at `at`. The Halo 4 twin of `SceneController::build_preview_meshes` (scene.rs),
    /// same signature shape and same contract, so `App::show_preview` calls one shape on either
    /// game. The scene's per-rebuild state (pick list, signature, pending flag) is snapshotted and
    /// restored, so the viewport never sees the preview object; the model / texture / material
    /// caches are shared on purpose (decode once).
    ///
    /// A palette row with no render model resolves to `H4_MARKER_TAG` and previews as the marker
    /// cube, exactly as it draws in the viewport. `_variant_sid` only keeps the Reach signature:
    /// a Halo 4 palette variant IS its own obje tag, which the caller already resolved.
    ///
    /// Returns `Ok((opaque, cutout, holo, holo_solid, blend, world aabb min, max))`, `Err(true)`
    /// while the budgeted decode still has work left (the caller retries next frame), `Err(false)`
    /// when there is nothing to show (unknown tag / empty decode).
    #[allow(clippy::type_complexity)]
    pub fn build_preview_meshes(
        &mut self,
        obj_tag: u32,
        _variant_sid: u32,
        at: Vec3,
        mesh: &MeshRenderer,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Result<(Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec3, Vec3), bool> {
        let Some(mode) = self.preview_mode_tag(obj_tag) else { return Err(false) };
        const PREVIEW_DATUM: u32 = 0xF2F0_0000;
        let objects = [ObjectInfo {
            datum: PREVIEW_DATUM, type_sig: 0, sig0: 0, sig1: 0,
            pos: [at.x, at.y, at.z], health: 1.0, shield: 1.0,
            mode_tag: mode, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
            attached: [0; 8], primary_tag: obj_tag, variant_name_sid: 0,
        }];
        // A NEW placement's team / colour (Forge neutral, no override) - the Reach preview's rule.
        let colors: HashMap<u32, (u8, u8)> = HashMap::from([(PREVIEW_DATUM, (crate::mvar::TEAM_NEUTRAL, 0xFF))]);
        let saved_picks = std::mem::take(&mut self.picks);
        let saved_sig = self.last_signature;
        let saved_pending = self.rebuild_pending;
        self.rebuild_pending = true; // force the pass even if the signature happened to match
        let mut out = None;
        for _ in 0..8 {
            out = ObjectScene::maybe_rebuild(self, &objects, &colors, mesh, _device, _queue).or(out);
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

    /// Decode NEW models for `objects` within the frame budget (always at least one). Returns
    /// true when some are still left for a later frame.
    fn decode_budgeted(&mut self, objects: &[ObjectInfo]) -> bool {
        let t_budget = std::time::Instant::now();
        let mut remaining = false;
        let mut decoded_any = false;
        let mut seen: HashSet<u32> = HashSet::new();
        for o in objects {
            if o.mode_tag == 0 || o.mode_tag == 0xFFFF_FFFF { continue; }
            if !seen.insert(o.mode_tag) { continue; }
            if self.models.contains_key(&o.mode_tag) { continue; }
            if decoded_any && t_budget.elapsed().as_millis() >= FRAME_DECODE_MS { remaining = true; break; }
            let dec = match h4_tag_index(o.mode_tag) {
                Some(idx) => match decode_model_geom(&self.cache, idx) {
                    Ok(g) => {
                        for f in &g.mesh_fail { self.failed.push(f.clone()); }
                        if g.meshes.is_empty() { self.failed.push(format!("{}: no drawable lod0 mesh", g.name)); None } else { Some(Arc::new(g)) }
                    }
                    Err(e) => { self.failed.push(format!("{}: {e}", self.cache.tag_name(idx))); None }
                },
                None => None,
            };
            self.models.insert(o.mode_tag, dec);
            decoded_any = true;
        }
        remaining
    }

    // ---- #h4-phys collision / physics hulls -------------------------------------------------

    /// The object's (collision, physics) hulls in MODEL space, decoded once per obje tag.
    fn obj_hulls(&self, obje_idx: usize) -> (Option<Arc<HullGeom>>, Option<Arc<HullGeom>>) {
        if let Some(v) = self.hulls.borrow().get(&obje_idx) { return v.clone(); }
        let v = (
            collision::object_collision_hull(&self.cache, obje_idx).map(Arc::new),
            collision::object_physics_hull(&self.cache, obje_idx).map(Arc::new),
        );
        self.hulls.borrow_mut().insert(obje_idx, v.clone());
        v
    }

    /// The placement matrix the overlays use: the object pose times its editor scale (the same
    /// matrix the drawn mesh and the pick entry get, so a scaled block's hull scales with it).
    fn overlay_matrix(&self, o: &ObjectInfo) -> Mat4 {
        let s = self.obj_scales.get(&o.datum).copied().unwrap_or(1.0);
        if (s - 1.0).abs() > 1e-4 { object_matrix(o) * Mat4::from_scale(Vec3::splat(s)) } else { object_matrix(o) }
    }

    /// The HIDDEN-BLOCK rule, ported verbatim from Reach (`scene.rs` build_object_meshes): the
    /// render model draws essentially nothing - empty geometry or a near-zero visible extent (the
    /// engine's invisible-wall placeholder: a ~2-triangle nub, Halo 4's `nut_blockers` all read
    /// exactly 0.10 x 0.10 x 0.01 wu) - AND a collision / physics hull with real volume exists.
    /// This is an ENGINE-BEHAVIOUR test, not a name or category match, so a renamed or ported
    /// blocker still lights up; on the shipped maps it selects exactly the objects the scenario's
    /// hidden ("superforge") Forge palette holds (`palette.rs` flag bit 0 = the `bb_*` set, all ten
    /// of which are `objects\multi\nut_blockers\*`). Collision is preferred over physics because
    /// it is the surface players stand on; Halo 4's blockers carry only a `phmo`.
    fn blocker_hull(&self, obje_idx: usize, render_empty: bool, render_ext: f32) -> Option<Arc<HullGeom>> {
        if !(render_empty || render_ext < BLOCKER_RENDER_MAX) { return None; }
        let (coll, phmo) = self.obj_hulls(obje_idx);
        let hull = coll.or(phmo)?;
        (hull.extent() > BLOCKER_HULL_MIN).then_some(hull)
    }

    /// #h4-phys Register a hidden block: REPLACE the render nub's pick entry (pushed just before,
    /// same datum) with a hull entry, and add the hull to the always-on overlay. One place, so the
    /// real rebuild and the offline pick builder cannot drift apart.
    fn push_blocker(&mut self, h: Arc<HullGeom>, m: Mat4, datum: u32, mode_tag: u32) {
        let wpts: Vec<Vec3> = h.verts.iter().map(|v| m.transform_point3(Vec3::from(*v))).collect();
        let mut hmn = Vec3::splat(f32::MAX);
        let mut hmx = Vec3::splat(f32::MIN);
        for p in &wpts { if p.is_finite() { hmn = hmn.min(*p); hmx = hmx.max(*p); } }
        if hmn.x > hmx.x { return; }
        // With both the nub entry and the hull entry present every `.find(datum)` consumer would
        // read the 0.1 wu nub, so the nub's entry goes.
        if self.picks.last().map(|p| (p.datum, p.att.is_none())) == Some((datum, true)) { self.picks.pop(); }
        self.picks.push(PickEntry { datum, min: hmn, max: hmx, mode_tag, xform: m, hull: Some(h.clone()), att: None });
        for t in h.indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if a >= wpts.len() || b >= wpts.len() || c >= wpts.len() { continue; }
            let (pa, pb, pc) = (wpts[a], wpts[b], wpts[c]);
            self.blocker_tris.push((pa.to_array(), BLOCKER_SOLID));
            self.blocker_tris.push((pb.to_array(), BLOCKER_SOLID));
            self.blocker_tris.push((pc.to_array(), BLOCKER_SOLID));
            self.blocker_lines.push((pa.to_array(), pb.to_array(), BLOCKER_LINE));
            self.blocker_lines.push((pb.to_array(), pc.to_array(), BLOCKER_LINE));
            self.blocker_lines.push((pc.to_array(), pa.to_array(), BLOCKER_LINE));
        }
    }

    /// Ray-vs-triangles refine for one pick entry in its local space: (hit t along the WORLD
    /// ray, whether the entry has triangle geometry to test). The un-normalised local direction
    /// keeps `t` comparable to the world distance (scene.rs `pick_refine`).
    fn pick_refine(&self, p: &PickEntry, origin: Vec3, dir: Vec3) -> (Option<f32>, bool) {
        let inv = p.xform.inverse();
        let lo = inv.transform_point3(origin);
        let ld = inv.transform_vector3(dir);
        // #h4-phys a hidden block refines against its physics hull, never the invisible nub.
        if let Some(h) = &p.hull {
            return (crate::scene::raycast_hull_pts_local(&h.verts, &h.indices, lo, ld), h.indices.len() >= 3);
        }
        // #wire-attach an attachment entry refines against the CHILD model's triangles (it kept
        // the parent's datum, so a hit still selects the vehicle).
        let geom = match &p.att {
            Some(k) => self.att_models.get(k),
            None => self.models.get(&p.mode_tag),
        };
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

    /// Nearest object surface hit (t) along a ray with the Reach pick rules, excluding datums.
    fn nearest_hit(&self, origin: Vec3, dir: Vec3, exclude: &[u32], min_t: f32) -> Option<(f32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for p in &self.picks {
            if exclude.contains(&p.datum) { continue; }
            let Some(aabb_t) = ray_aabb(origin, dir, p.min, p.max) else { continue };
            let (tri_t, has_mesh) = self.pick_refine(p, origin, dir);
            let t = match tri_t {
                Some(t) => t,
                None => { if has_mesh { continue; } aabb_t }
            };
            if t > min_t && best.map_or(true, |(bt, _)| t < bt) { best = Some((t, p.datum)); }
        }
        best
    }
}

/// The unit holo cube: 12 triangles, a `MARKER_HALF` half-extent box whose base sits on
/// the object origin, one part with no material (drawn through `upload_part` on the blend lane).
fn marker_geom() -> H4ModelGeom {
    let h = MARKER_HALF;
    let (z0, z1) = (0.0, 2.0 * h);
    let corners = [
        [-h, -h, z0], [h, -h, z0], [h, h, z0], [-h, h, z0],
        [-h, -h, z1], [h, -h, z1], [h, h, z1], [-h, h, z1],
    ];
    // (face indices, normal)
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0, 3, 2, 1], [0.0, 0.0, -1.0]),
        ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ([2, 3, 7, 6], [0.0, 1.0, 0.0]),
        ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
        ([3, 0, 4, 7], [-1.0, 0.0, 0.0]),
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
    H4ModelGeom {
        name: "editor marker cube".into(),
        materials: vec![None],
        meshes: vec![H4GeomMesh { verts, indices, parts: vec![Part { material: 0, index_start: 0, index_count: 36 }] }],
        min: Vec3::new(-h, -h, z0),
        max: Vec3::new(h, h, z1),
        mesh_fail: Vec::new(),
    }
}

impl ObjectScene for H4ObjectScene {
    fn has_cache(&self) -> bool { true }
    fn invalidate(&mut self) { self.last_signature = 0; }
    fn rebuild_pending(&self) -> bool { self.rebuild_pending }

    fn maybe_rebuild(
        &mut self,
        objects: &[ObjectInfo],
        colors: &HashMap<u32, (u8, u8)>,
        mr: &MeshRenderer,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Option<RebuiltLanes> {
        // Fold the scale / caster maps into the signature (order-independent) - scene.rs.
        let scale_sig = self.obj_scales.iter().fold(0u64, |acc, (d, s)| acc ^ (*d as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(s.to_bits() as u64));
        let caster_sig = self.casters.iter().fold(0u64, |acc, d| acc ^ (*d as u64 ^ 0xA5A5_5A5A).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        let sig = signature(objects, colors) ^ scale_sig ^ caster_sig;
        if sig == self.last_signature && !self.rebuild_pending { return None; }
        self.last_signature = sig;
        let n_failed = self.failed.len();
        self.rebuild_pending = self.decode_budgeted(objects);
        for f in &self.failed[n_failed..] { log::warn!("h4 editor model: {f}"); }

        // Group instances by (mode tag, casts sun shadow) in first-seen order (like the static path)
        // and record the picks. #h4-shadows: EVERY Halo 4 object casts (the engine's forge-lightmap
        // burn renders all registered objects into its shadow map, halo4.dll sub_1803443DC), so the
        // group key is always `true`; the per-object SHADOW override set (`set_object_casters`) is
        // kept in the signature only so a change still triggers a rebuild.
        // #h4-veh the key carries the object's PRIMARY change colour (a per-mesh instance lane, so
        // two placements may only share a draw when it matches - the same rule as the static path).
        let mut by_mode: HashMap<(u32, bool, [i32; 4]), (Vec<Mat4>, Vec<[f32; 4]>, [f32; 4])> = HashMap::new();
        let sun_to = self.sun_to;
        let mut order: Vec<(u32, bool, [i32; 4])> = Vec::new();
        let mut lanes_by_key: HashMap<(u32, bool, [i32; 4]), Vec<[u32; 4]>> = HashMap::new();
        // #h4-veh hlmt-variant CHILD objects of the placed objects: (child obje, child mode, child
        // variant, colour) -> world matrices + the parent datum (so a pick on the turret selects
        // the vehicle).
        let mut att: HashMap<(usize, usize, Option<usize>, [i32; 4]), (Vec<Mat4>, [f32; 4], u32)> = HashMap::new();
        let mut att_order: Vec<(usize, usize, Option<usize>, [i32; 4])> = Vec::new();
        self.picks.clear();
        self.blocker_tris.clear();
        self.blocker_lines.clear();
        for o in objects {
            if o.mode_tag == 0 || o.mode_tag == 0xFFFF_FFFF { continue; }
            let Some(Some(geom)) = self.models.get(&o.mode_tag) else { continue };
            let s = self.obj_scales.get(&o.datum).copied().unwrap_or(1.0);
            let m = if (s - 1.0).abs() > 1e-4 { object_matrix(o) * Mat4::from_scale(Vec3::splat(s)) } else { object_matrix(o) };
            // the placement's team / colour override -> the engine's primary change colour
            let obje = h4_tag_index(o.primary_tag);
            let mp = colors.get(&o.datum).map(|(t, c)| (*t as i8, (*c != 0xFF).then_some(*c)));
            let cc = match obje {
                Some(ot) => super::tint::primary_change_color(&self.cache, ot, o.variant_name_sid, mp.or(Some((crate::mvar::TEAM_NEUTRAL as i8, None)))),
                None => NEUTRAL_CC,
            };
            let key = (o.mode_tag, true, super::tint::color_key(cc));
            let (mats, cols, _) = by_mode.entry(key).or_insert_with(|| { order.push(key); (Vec::new(), Vec::new(), cc) });
            mats.push(m);
            if let Some(ot) = obje {
                let pv = super::objects::object_variant_index(&self.cache, ot, o.variant_name_sid);
                for a in super::attach::expand_attachments(&self.cache, ot, pv, m) {
                    let vsid = super::objects::default_variant_sid(&self.cache, a.obje);
                    let acc = super::tint::primary_change_color(&self.cache, a.obje, vsid, mp);
                    let akey = (a.obje, a.mode, a.variant, super::tint::color_key(acc));
                    att.entry(akey).or_insert_with(|| { att_order.push(akey); (Vec::new(), acc, o.datum) }).0.push(a.world);
                }
            }
            let (mn, mx) = transform_aabb(geom.min, geom.max, &m);
            // #h4-phys the VISIBLE extent of the render model - the hidden-block discriminator.
            let render_ext = (geom.max - geom.min).abs().max_element();
            let render_empty = geom.tri_count() == 0;
            // probe SH ambient + lobes, sun visibility by raycast against the BSP soup
            let vis = super::scene::object_sun_vis(Some(&self.soup), sun_to, mn, mx);
            let (c, l) = self.light.object_lanes_aabb(Some(&self.soup), mn, mx, vis); // surface probe
            cols.push(c);
            lanes_by_key.entry(key).or_insert_with(Vec::new).push(l);
            self.picks.push(PickEntry { datum: o.datum, min: mn, max: mx, mode_tag: o.mode_tag, xform: m, hull: None, att: None });
            // #h4-phys HIDDEN BLOCKS (the "superforge" `bb_*` / `nut_blockers` pieces): overlay the
            // physics hull and make the block pickable BY that hull. Forge-placed objects only
            // (datum 0xD variant / dup, 0xF local) - the Reach restriction, so a scenario object
            // with a big collision volume can never paint the map orange.
            let hull = match (matches!(o.datum >> 28, 0xD | 0xF), obje) {
                (true, Some(ot)) => self.blocker_hull(ot, render_empty, render_ext),
                _ => None,
            };
            if let Some(h) = hull { self.push_blocker(h, m, o.datum, o.mode_tag); }
        }

        let mut lanes = RebuiltLanes { opaque: Vec::new(), cutout: Vec::new(), holo: Vec::new(), holo_solid: Vec::new(), blend: Vec::new() };
        for key in order {
            let (tag, casts, _) = key;
            let Some(Some(geom)) = self.models.get(&tag).cloned() else { continue };
            let (mats, cols, cc) = &by_mode[&key];
            let cc = *cc;
            let shading_on = self.shading_on;
            let mut out = |lane: Lane, mut gm: GpuMesh| {
                gm.casts_shadow = casts;
                match lane {
                    Lane::Opaque => lanes.opaque.push(gm),
                    Lane::AlphaTest => lanes.cutout.push(gm),
                    Lane::Blend => lanes.blend.push(gm),
                    Lane::Additive => lanes.holo.push(gm),
                    Lane::Sky => {}
                }
            };
            if tag == H4_MARKER_TAG {
                // the marker cube: translucent, untextured, on the alpha-blend object lane
                let mesh = &geom.meshes[0];
                let gm = super::scene::upload_part(mr, &self.tex.device, &self.tex.queue, &mesh.verts, &mesh.indices, mats, None, super::materials::H4MatKind::AlphaBlend, 0.35, None);
                out(Lane::Blend, gm);
                continue;
            }
            let lanes = lanes_by_key.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
            upload_model_parts(&mut self.tex, mr, &geom, mats, cols, lanes, cc, &mut self.obj_mats, shading_on, &mut self.stats, &mut out);
        }
        // #h4-veh the child objects: decoded with their OWN obje tag (their materials, hlmt
        // variant and change colours), posed by the parent/child marker rule, drawn into the same
        // lanes. Their AABB joins the parent's pick entry list under the PARENT datum, so clicking
        // a turret selects its vehicle.
        for akey in att_order {
            let (mats, cc, parent) = att[&akey].clone();
            if !self.att_models.contains_key(&(akey.0, akey.1, akey.2)) {
                let dec = match decode_object_geom(&self.cache, akey.0, akey.1, akey.2) {
                    Ok(g) if !g.meshes.is_empty() => Some(Arc::new(g)),
                    Ok(_) => None,
                    Err(e) => { self.failed.push(format!("{}: {e}", self.cache.tag_name(akey.1))); None }
                };
                self.att_models.insert((akey.0, akey.1, akey.2), dec);
            }
            let Some(Some(geom)) = self.att_models.get(&(akey.0, akey.1, akey.2)).cloned() else { continue };
            let mut cols = Vec::with_capacity(mats.len());
            let mut olanes = Vec::with_capacity(mats.len());
            for m in &mats {
                let (mn, mx) = transform_aabb(geom.min, geom.max, m);
                let vis = super::scene::object_sun_vis(Some(&self.soup), sun_to, mn, mx);
                let (c, l) = self.light.object_lanes_aabb(Some(&self.soup), mn, mx, vis);
                cols.push(c);
                olanes.push(l);
                self.picks.push(PickEntry { datum: parent, min: mn, max: mx, mode_tag: 0, xform: *m, hull: None, att: Some((akey.0, akey.1, akey.2)) });
            }
            let shading_on = self.shading_on;
            let mut out = |lane: Lane, mut gm: GpuMesh| {
                gm.casts_shadow = true;
                match lane {
                    Lane::Opaque => lanes.opaque.push(gm),
                    Lane::AlphaTest => lanes.cutout.push(gm),
                    Lane::Blend => lanes.blend.push(gm),
                    Lane::Additive => lanes.holo.push(gm),
                    Lane::Sky => {}
                }
            };
            upload_model_parts(&mut self.tex, mr, &geom, &mats, &cols, &olanes, cc, &mut self.obj_mats, shading_on, &mut self.stats, &mut out);
        }
        Some(lanes)
    }

    fn set_object_scales(&mut self, scales: HashMap<u32, f32>) { self.obj_scales = scales; }
    fn set_object_casters(&mut self, casters: HashSet<u32>) { self.casters = casters; }

    fn pick(&self, origin: Vec3, dir: Vec3) -> Option<u32> {
        self.nearest_hit(origin, dir, &[], f32::NEG_INFINITY).map(|(_, d)| d)
    }

    /// #wire-attach The UNION of every pick entry with this datum - the object's own box plus each
    /// attachment's (a Wraith's mortar, a Mantis' guns), so box-select, frame-selected, the magnet
    /// faces and the array cell size all cover the whole vehicle.
    fn aabb_of(&self, datum: u32) -> Option<([f32; 3], [f32; 3])> {
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        let mut any = false;
        for p in self.picks.iter().filter(|p| p.datum == datum) {
            if !(p.min.is_finite() && p.max.is_finite()) { continue; }
            mn = mn.min(p.min);
            mx = mx.max(p.max);
            any = true;
        }
        any.then(|| (mn.into(), mx.into()))
    }

    fn object_obb(&self, datum: u32) -> Option<crate::construct::Obb> {
        // The object's OWN entry supplies the frame; attachments only grow the box.
        let p = self.picks.iter().find(|p| p.datum == datum && p.att.is_none())
            .or_else(|| self.picks.iter().find(|p| p.datum == datum))?;
        // #h4-phys a hidden block's oriented box is its physics hull, not the nub.
        let local = if let Some(h) = &p.hull {
            h.bounds()
        } else {
            match self.models.get(&p.mode_tag) {
                Some(Some(g)) if g.tri_count() > 0 => Some((g.min, g.max)),
                _ => None,
            }
        };
        let Some((mut lmn, mut lmx)) = local else { return Some(crate::construct::Obb::from_aabb(p.min, p.max)) };
        // #wire-attach grow the LOCAL box by every attachment's world box, mapped back through the
        // parent frame, so a turret is inside the oriented box the gizmo / snap / array read.
        let inv = p.xform.inverse();
        for a in self.picks.iter().filter(|q| q.datum == datum && q.att.is_some()) {
            if !(a.min.is_finite() && a.max.is_finite()) { continue; }
            for k in 0..8u32 {
                let c = Vec3::new(
                    if k & 4 != 0 { a.max.x } else { a.min.x },
                    if k & 2 != 0 { a.max.y } else { a.min.y },
                    if k & 1 != 0 { a.max.z } else { a.min.z },
                );
                let l = inv.transform_point3(c);
                if !l.is_finite() { continue; }
                lmn = lmn.min(l);
                lmx = lmx.max(l);
            }
        }
        Some(crate::construct::Obb::from_local(&p.xform, lmn, lmx))
    }

    /// #wire-attach Every pick entry of this datum contributes its own geometry: the object's
    /// model (or, for a hidden block, its physics hull) PLUS each attachment's child model at the
    /// pose `h4::attach` computed, so the wireframe wraps a Wraith's mortar and a rocket Warthog's
    /// turret. The whole set shares one line budget, so a big turret cannot blow the buffer.
    fn selection_wireframe(&self, datum: u32) -> Option<Vec<[f32; 3]>> {
        const MAX_LINES: usize = 60_000; // 2 verts each
        // #wire-attach ATTACHMENTS FIRST, with half the budget reserved for them: a dense parent
        // mesh would otherwise eat the whole budget and leave its turret out of the outline.
        const ATT_LINES: usize = MAX_LINES / 2;
        let mut out: Vec<[f32; 3]> = Vec::new();
        let mut saw_entry = false;
        let mut order: Vec<&PickEntry> = self.picks.iter().filter(|p| p.datum == datum).collect();
        order.sort_by_key(|p| p.att.is_none()); // attachments first
        for p in order {
            saw_entry = true;
            let cap = if p.att.is_some() { ATT_LINES } else { MAX_LINES };
            if out.len() >= cap * 2 { continue; }
            // #h4-phys a hidden block's wireframe IS its physics hull (the nub is 2 triangles).
            if let Some(h) = &p.hull {
                for (a, b) in h.world_edges(&p.xform) {
                    if out.len() >= cap * 2 { break; }
                    out.push(a);
                    out.push(b);
                }
                continue;
            }
            let geom = match &p.att {
                Some(k) => self.att_models.get(k),
                None => self.models.get(&p.mode_tag),
            };
            let Some(Some(g)) = geom else { continue };
            'meshes: for m in &g.meshes {
                let n = m.indices.len() / 3 * 3;
                let mut i = 0;
                while i < n {
                    if out.len() >= cap * 2 { break 'meshes; }
                    let (a, b, c) = (m.indices[i] as usize, m.indices[i + 1] as usize, m.indices[i + 2] as usize);
                    i += 3;
                    let (Some(va), Some(vb), Some(vc)) = (m.verts.get(a), m.verts.get(b), m.verts.get(c)) else { continue };
                    let wa = p.xform.transform_point3(Vec3::from(va.pos)).to_array();
                    let wb = p.xform.transform_point3(Vec3::from(vb.pos)).to_array();
                    let wc = p.xform.transform_point3(Vec3::from(vc.pos)).to_array();
                    out.extend_from_slice(&[wa, wb, wb, wc, wc, wa]);
                }
            }
        }
        if !saw_entry { return None; }
        (!out.is_empty()).then_some(out)
    }

    /// #wire-attach EVERY entry of the datum moves - the attachments ride with their parent, so
    /// the union AABB stays truthful between rebuilds.
    fn translate_pick(&mut self, datum: u32, delta: Vec3) {
        for p in self.picks.iter_mut().filter(|p| p.datum == datum) {
            p.min += delta;
            p.max += delta;
            p.xform = Mat4::from_translation(delta) * p.xform;
        }
    }

    fn rotate_pick(&mut self, datum: u32, q: Quat, pivot: Vec3) {
        let m = Mat4::from_translation(pivot) * Mat4::from_quat(q) * Mat4::from_translation(-pivot);
        for p in self.picks.iter_mut().filter(|p| p.datum == datum) {
            p.xform = m * p.xform;
        }
    }

    fn raycast_scene(&self, origin: Vec3, dir: Vec3) -> Option<Vec3> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        self.soup.raycast(origin, d, 1.0e5).map(|(t, _n)| origin + d * t)
    }

    fn raycast_scene_n(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        self.soup.raycast(origin, d, 1.0e5).map(|(t, n)| (origin + d * t, n))
    }

    fn raycast_objects_excluding(&self, origin: Vec3, dir: Vec3, exclude: &[u32]) -> Option<Vec3> {
        self.nearest_hit(origin, dir, exclude, 1e-4).map(|(t, _)| origin + dir * t)
    }

    fn h4_shadow_setup(&self) -> Option<(Option<(Vec3, Vec3)>, Option<hms_render::H4CascadeCfg>)> {
        let mut b = self.static_caster_bounds;
        for p in &self.picks {
            if p.mode_tag == H4_MARKER_TAG { continue; }
            if !(p.min.is_finite() && p.max.is_finite()) { continue; }
            b = Some(match b { Some((a, c)) => (a.min(p.min), c.max(p.max)), None => (p.min, p.max) });
        }
        Some((b, self.cascade))
    }

    fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let (mn, mx) = self.world_bounds;
        (mn[0] <= mx[0]).then_some((mn, mx))
    }

    fn resolve_object_mode(&self, obj_tag: u32) -> u32 {
        match h4_tag_index(obj_tag) {
            Some(idx) => self.mode_tag_of_obje(idx),
            None => 0,
        }
    }

    fn tag_name_of(&self, tag: u32) -> String {
        if tag == H4_MARKER_TAG { return "editor marker (no render model)".into(); }
        match h4_tag_index(tag) {
            Some(idx) if self.cache.tag_meta(idx).is_some() => self.cache.tag_name(idx).to_string(),
            _ => String::new(),
        }
    }

    fn raw_tag_class(&self, tag: u32) -> Option<String> {
        let idx = h4_tag_index(tag)?;
        self.cache.tag_class(idx).map(|c| String::from_utf8_lossy(&c).into_owned())
    }

    fn blocker_overlay(&self) -> (&[([f32; 3], [f32; 4])], &[([f32; 3], [f32; 3], [f32; 3])]) {
        (&self.blocker_tris, &self.blocker_lines)
    }

    /// #h4-phys `coll` hull edges of one placed object (empty when the object has no collision
    /// model - Halo 4's Forge blockers, for instance, ship only a `phmo`).
    fn collision_world_edges(&self, o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> {
        let Some(ot) = h4_tag_index(o.primary_tag) else { return Vec::new() };
        match self.obj_hulls(ot).0 {
            Some(h) => h.world_edges(&self.overlay_matrix(o)),
            None => Vec::new(),
        }
    }

    /// #h4-phys `phmo` rigid-body hull edges of one placed object.
    fn physics_world_edges(&self, o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])> {
        let Some(ot) = h4_tag_index(o.primary_tag) else { return Vec::new() };
        match self.obj_hulls(ot).1 {
            Some(h) => h.world_edges(&self.overlay_matrix(o)),
            None => Vec::new(),
        }
    }

    fn pick_entry_counts(&self, datum: u32) -> (usize, usize) {
        let all: Vec<&PickEntry> = self.picks.iter().filter(|p| p.datum == datum).collect();
        (all.len(), all.iter().filter(|p| p.att.is_some()).count())
    }

    fn bsp_flags_report(&self) -> String {
        match self.bounds() {
            Some(b) => format!("H4BOUNDS variant bounds x [{:.1}, {:.1}] y [{:.1}, {:.1}] z [{:.1}, {:.1}] (Halo 4 has no bsp escape: objects outside cannot be encoded)\n", b[0], b[1], b[2], b[3], b[4], b[5]),
            None => "H4BOUNDS no variant loaded\n".to_string(),
        }
    }

    fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)> {
        // the Reach chain over the Halo 4 reader (spawncam.rs): the variant's
        // loadout camera / initial spawn (red, blue, neutral) / any spawn, else the scenario's
        // starting locations / spawn placements / cinematic camera - stood off the geometry.
        super::spawncam::spawn_camera(self).map(|c| (c.pos, c.yaw, c.pitch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;
    use crate::h4::mvar::{load_forge_palette, parse_h4_variant, variant_dirs};

    /// Settler on Ravine (388 objects) as an offline editor scene, or None when MCC is absent.
    fn settler_scene() -> Option<H4ObjectScene> {
        let maps = maps_dir()?;
        let map = maps.join("ca_forge_ravine.map");
        if !map.is_file() { eprintln!("skip: {} missing", map.display()); return None; }
        let vp = variant_dirs().into_iter().map(|d| d.join("ca_forge_ravine_settler.mvar")).find(|p| p.is_file())?;
        let cache = Arc::new(H4Cache::open(&map).expect("open"));
        let palette = load_forge_palette(&cache);
        let v = parse_h4_variant(&vp).expect("variant");
        H4ObjectScene::new_offline(cache, palette, Some((vp, v)))
    }

    /// Decode every model + build the pick list WITHOUT the GPU (the pick / bounds / wireframe
    /// layer is what `maybe_rebuild` fills before it uploads). Mirrors the three entry kinds the
    /// real pass pushes: the object's own box, a hidden block's hull (#h4-phys) and one per
    /// hlmt-variant attachment under the parent's datum (#wire-attach).
    fn build_picks(es: &mut H4ObjectScene, objects: &[ObjectInfo]) {
        while es.decode_budgeted(objects) {}
        es.picks.clear();
        es.blocker_tris.clear();
        es.blocker_lines.clear();
        for o in objects {
            let Some(Some(geom)) = es.models.get(&o.mode_tag) else { continue };
            let m = object_matrix(o);
            let (mn, mx) = transform_aabb(geom.min, geom.max, &m);
            let render_ext = (geom.max - geom.min).abs().max_element();
            let render_empty = geom.tri_count() == 0;
            es.picks.push(PickEntry { datum: o.datum, min: mn, max: mx, mode_tag: o.mode_tag, xform: m, hull: None, att: None });
            let obje = h4_tag_index(o.primary_tag);
            if let (true, Some(ot)) = (matches!(o.datum >> 28, 0xD | 0xF), obje) {
                if let Some(h) = es.blocker_hull(ot, render_empty, render_ext) {
                    es.push_blocker(h, m, o.datum, o.mode_tag);
                }
            }
            let Some(ot) = obje else { continue };
            let pv = crate::h4::objects::object_variant_index(&es.cache, ot, o.variant_name_sid);
            for a in crate::h4::attach::expand_attachments(&es.cache, ot, pv, m) {
                let key = (a.obje, a.mode, a.variant);
                if !es.att_models.contains_key(&key) {
                    let dec = match decode_object_geom(&es.cache, a.obje, a.mode, a.variant) {
                        Ok(g) if !g.meshes.is_empty() => Some(Arc::new(g)),
                        _ => None,
                    };
                    es.att_models.insert(key, dec);
                }
                let Some(Some(g)) = es.att_models.get(&key) else { continue };
                let (amn, amx) = transform_aabb(g.min, g.max, &a.world);
                es.picks.push(PickEntry { datum: o.datum, min: amn, max: amx, mode_tag: 0, xform: a.world, hull: None, att: Some(key) });
            }
        }
    }

    /// #h4-phys A HIDDEN ("superforge") block: its render model is an invisible 0.10 wu nub, so
    /// every selection consumer must read its PHYSICS hull instead - the overlay is built, the box
    /// is the hull's, the wireframe is the hull's, a ray from above hits it, and a marquee that
    /// only clips the hull still catches it.
    #[test]
    fn hidden_block_is_overlaid_and_pickable_by_its_hull() {
        let Some(mut es) = settler_scene() else { return };
        let cache = es.cache.clone();
        // (tag suffix, the hull's longest side in wu - see h4::collision::tests)
        let blockers = [("box_xxxl", 3.968f32), ("wall_m", 0.968), ("wall_s", 0.668)];
        let mut objects: Vec<ObjectInfo> = Vec::new();
        let mut which: Vec<(&str, f32, u32)> = Vec::new();
        for (i, (nm, longest)) in blockers.iter().enumerate() {
            let suffix = format!("nut_blockers\\{nm}\\{nm}");
            let Some(ot) = [b"bloc", b"scen"].iter().find_map(|cls| cache.find_tags(cls).into_iter().find(|&t| cache.tag_name(t).ends_with(&suffix))) else {
                eprintln!("skip: {suffix} missing");
                continue;
            };
            let mode_tag = es.mode_tag_of_obje(ot);
            let datum = 0xD000_0100 + i as u32;
            objects.push(ObjectInfo {
                datum, type_sig: 0, sig0: 0, sig1: 0,
                pos: [i as f32 * 20.0, 0.0, 0.0], health: 1.0, shield: 1.0,
                mode_tag, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
                attached: [0; 8], primary_tag: h4_tag(ot), variant_name_sid: 0,
            });
            which.push((nm, *longest, datum));
        }
        if objects.is_empty() { return; }
        build_picks(&mut es, &objects);
        let (tris, lines) = es.blocker_overlay();
        eprintln!("hidden blocks: {} overlay triangles, {} outline segments", tris.len() / 3, lines.len());
        assert!(tris.len() >= which.len() * 3 * 12, "one hull per blocker in the overlay");
        assert_eq!(tris.len(), lines.len(), "one solid tri per outline triple");
        for (nm, longest, datum) in which {
            let p = es.picks.iter().find(|p| p.datum == datum).expect("pick entry");
            assert!(p.hull.is_some(), "{nm}: the pick entry must carry the hull");
            let (mn, mx) = es.aabb_of(datum).expect("aabb");
            let ext = (Vec3::from(mx) - Vec3::from(mn)).max_element();
            eprintln!("{nm}: hull aabb {mn:?}..{mx:?} (longest {ext:.3} wu, want {longest})");
            // the HULL's extent, not the 0.1 wu render nub
            assert!((ext - longest).abs() < 0.02, "{nm}: box is {ext} wu, want the hull's {longest}");
            // the wireframe is the hull (a nub would be 3 lines)
            let w = es.selection_wireframe(datum).expect("wireframe");
            assert!(w.len() / 2 >= 36, "{nm}: hull wireframe has {} lines", w.len() / 2);
            // the oriented box is the hull's too
            let obb = es.object_obb(datum).expect("obb");
            let he = obb.hx.length().max(obb.hy.length()).max(obb.hz.length());
            assert!((he * 2.0 - longest).abs() < 0.02, "{nm}: obb longest side {} wu", he * 2.0);
            // a ray straight down hits it by the hull. NOT through the dead centre: on an
            // axis-aligned box the centre ray runs exactly along the top face's triangulation
            // diagonal, where the barycentric test is a coin flip.
            let c = (Vec3::from(mn) + Vec3::from(mx)) * 0.5;
            let off = (Vec3::from(mx) - Vec3::from(mn)).min(Vec3::splat(1.0)) * 0.17;
            assert_eq!(es.pick(Vec3::new(c.x + off.x, c.y - off.y, mx[2] + 5.0), Vec3::NEG_Z), Some(datum), "{nm}: click pick");
            // a marquee that only clips the hull's corner still catches it (the nub's box would not)
            let corner_min = [mx[0] - 0.05, mx[1] - 0.05, mx[2] - 0.05];
            let corner_max = [mx[0] + 1.0, mx[1] + 1.0, mx[2] + 1.0];
            assert!(crate::physics_outlines::box_overlaps(corner_min, corner_max, [0.0; 3], Some((mn, mx))), "{nm}: box-select by hull");
        }
    }

    /// #wire-attach A vehicle whose hlmt variant hangs CHILD objects off its markers (the Wraith's
    /// plasma mortar, a rocket Warthog's turret) must have those children inside its selection
    /// geometry: the union AABB, the oriented box and the wireframe. The Mantis is checked too -
    /// its weapons are model REGIONS, not child objects, so it has no attachment entries and its
    /// own model bounds must already cover its guns.
    #[test]
    fn selection_geometry_covers_hlmt_variant_attachments() {
        let Some(mut es) = settler_scene() else { return };
        let cache = es.cache.clone();
        let find = |suffix: &str| -> Option<usize> {
            for cls in [b"vehi", b"bipd", b"bloc", b"scen"] {
                if let Some(t) = cache.find_tags(cls).into_iter().find(|&t| cache.tag_name(t).ends_with(suffix)) { return Some(t); }
            }
            None
        };
        // (tag suffix, does its hlmt variant hang CHILD objects?) - the Mantis is a `bipd` whose
        // weapons are model REGIONS, so it must have none and its own bounds must already cover them.
        let wants = [
            ("storm_wraith\\storm_wraith", true),
            ("storm_mantis\\storm_mantis", false),
            ("storm_warthog\\storm_warthog", true),
        ];
        let mut objects: Vec<ObjectInfo> = Vec::new();
        let mut which: Vec<(&str, bool, u32)> = Vec::new();
        for (i, (suffix, expect_children)) in wants.iter().enumerate() {
            let Some(ot) = find(suffix) else { eprintln!("skip: {suffix} not in ca_forge_ravine"); continue };
            let mode_tag = es.mode_tag_of_obje(ot);
            if mode_tag == 0 { continue; }
            let datum = 0xD000_0010 + i as u32;
            objects.push(ObjectInfo {
                datum, type_sig: 0, sig0: 0, sig1: 0,
                pos: [i as f32 * 40.0, 0.0, 0.0], health: 1.0, shield: 1.0,
                mode_tag, fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
                attached: [0; 8], primary_tag: h4_tag(ot), variant_name_sid: 0,
            });
            which.push((suffix, *expect_children, datum));
        }
        if objects.is_empty() { return; }
        build_picks(&mut es, &objects);
        for (suffix, expect_children, datum) in which {
            // the object's OWN entry = the old (parent-only) answer
            let own = es.picks.iter().find(|p| p.datum == datum && p.att.is_none()).expect("own pick");
            let (own_mn, own_mx) = (own.min, own.max);
            let n_att = es.picks.iter().filter(|p| p.datum == datum && p.att.is_some()).count();
            let (umn, umx) = es.aabb_of(datum).expect("union aabb");
            let (umn, umx) = (Vec3::from(umn), Vec3::from(umx));
            let grew = (own_mn - umn).abs().max_element().max((own_mx - umx).abs().max_element());
            eprintln!(
                "{suffix}: {n_att} attachment pick(s); own aabb {own_mn:?}..{own_mx:?} -> union {umn:?}..{umx:?} (grew {grew:.3} wu)"
            );
            // the union always contains the own box
            assert!(umn.cmple(own_mn).all() && umx.cmpge(own_mx).all(), "{suffix}: union must contain the own box");
            if expect_children {
                assert!(n_att > 0, "{suffix}: expected hlmt-variant child objects");
                assert!(grew > 0.05, "{suffix}: the selection box did not grow to cover the attachment");
                // the oriented box grew as well (it is what snap / array / the gizmo read)
                let obb = es.object_obb(datum).expect("obb");
                let mut lo = Vec3::splat(f32::MAX);
                let mut hi = Vec3::splat(f32::MIN);
                for i in 0..8u8 { let c = obb.corner(i); lo = lo.min(c); hi = hi.max(c); }
                for a in es.picks.iter().filter(|p| p.datum == datum && p.att.is_some()) {
                    let c = (a.min + a.max) * 0.5;
                    assert!(c.cmpge(lo - Vec3::splat(0.01)).all() && c.cmple(hi + Vec3::splat(0.01)).all(),
                            "{suffix}: attachment centre {c:?} outside the oriented box {lo:?}..{hi:?}");
                }
            } else {
                assert_eq!(n_att, 0, "{suffix}: the Mantis' weapons are model regions, not child objects");
            }
            // the wireframe covers every entry of the datum
            let w = es.selection_wireframe(datum).expect("wireframe");
            assert!(!w.is_empty() && w.len() % 2 == 0);
            let mut wmn = Vec3::splat(f32::MAX);
            let mut wmx = Vec3::splat(f32::MIN);
            for v in &w { wmn = wmn.min(Vec3::from(*v)); wmx = wmx.max(Vec3::from(*v)); }
            let truncated = w.len() / 2 >= 60_000; // the shared line budget
            eprintln!("  wireframe {} lines{}, bounds {wmn:?}..{wmx:?}", w.len() / 2, if truncated { " (budget reached)" } else { "" });
            // within a hair of the union box (the wireframe IS the geometry the boxes come from)
            // - unless the line budget cut it short, which is the point of the budget.
            if !truncated {
                assert!((wmn - umn).abs().max_element() < 0.02 && (wmx - umx).abs().max_element() < 0.02,
                        "{suffix}: wireframe bounds {wmn:?}..{wmx:?} vs union {umn:?}..{umx:?}");
            }
        }
    }

    #[test]
    fn settler_388_objects_decode_and_pick_from_above() {
        let Some(mut es) = settler_scene() else { return };
        let (objects, _colors) = es.variant_editor_objects();
        assert_eq!(objects.len(), 388, "every Settler record resolves through the palette");
        assert!(objects.iter().all(|o| o.mode_tag & 0xFFFF_0000 == H4_TAG_FLAG), "tags are encoded");
        build_picks(&mut es, &objects);
        // #wire-attach one entry per OBJECT, plus one per hlmt-variant attachment (the Settler's
        // vehicles carry turrets), all under the parent's datum.
        let own = es.picks.iter().filter(|p| p.att.is_none()).count();
        let atts = es.picks.iter().filter(|p| p.att.is_some()).count();
        eprintln!("settler picks: {own} own + {atts} attachment(s)");
        assert_eq!(own, 388, "one pick entry per object (marker cube for model-less)");
        assert_eq!(es.pick_count(), own + atts);
        let n_marker = objects.iter().filter(|o| o.mode_tag == H4_MARKER_TAG).count();
        eprintln!("settler: {} models decoded, {} marker objects, {} failed", es.model_count(), n_marker, es.failed.len());
        assert!(es.failed.is_empty(), "model failures: {:?}", es.failed);
        // AABB / OBB / wireframe for every object; wireframe non-empty for every object with a model
        for o in &objects {
            let (mn, mx) = es.aabb_of(o.datum).expect("aabb");
            assert!(mn.iter().zip(mx).all(|(a, b)| a <= &b), "aabb ordered for {:#x}", o.datum);
            assert!(es.object_obb(o.datum).is_some());
            let w = es.selection_wireframe(o.datum).expect("wireframe");
            assert!(!w.is_empty() && w.len() % 2 == 0, "wireframe lines for {:#x}", o.datum);
        }
        // rays straight down from above each object's footprint (a 7x7 grid over its AABB) hit the
        // object itself at least once; the box centre alone misses hollow / thin markers (arrows,
        // rings), and an object stacked INSIDE another's box may be shadowed at some samples
        let mut self_hit = 0;
        let mut centre_hit = 0;
        let mut misses = Vec::new();
        for o in &objects {
            let (mn, mx) = es.aabb_of(o.datum).unwrap();
            let c = (Vec3::from(mn) + Vec3::from(mx)) * 0.5;
            let top = mx[2] + 0.5;
            if es.pick(Vec3::new(c.x, c.y, top), Vec3::NEG_Z) == Some(o.datum) { centre_hit += 1; }
            let mut hit = false;
            'grid: for ix in 0..7 {
                for iy in 0..7 {
                    let x = mn[0] + (mx[0] - mn[0]) * (ix as f32 + 0.5) / 7.0;
                    let y = mn[1] + (mx[1] - mn[1]) * (iy as f32 + 0.5) / 7.0;
                    if es.pick(Vec3::new(x, y, top), Vec3::NEG_Z) == Some(o.datum) { hit = true; break 'grid; }
                }
            }
            if hit { self_hit += 1; } else { misses.push((o.datum, es.tag_name_of(o.mode_tag))); }
        }
        eprintln!("settler picks: {} of {} self-hits over the footprint grid ({} at the box centre), misses {:?}", self_hit, objects.len(), centre_hit, misses);
        assert!(self_hit * 20 >= objects.len() * 19, "at least 95% of the objects are picked from above their own footprint: {:?}", misses);
        assert!(centre_hit * 10 >= objects.len() * 7, "most objects are picked at their box centre: {centre_hit}");
        // translate / rotate keep the cached picks truthful
        let d0 = objects[0].datum;
        let before = es.aabb_of(d0).unwrap();
        es.translate_pick(d0, Vec3::new(1.0, 2.0, 3.0));
        let after = es.aabb_of(d0).unwrap();
        for k in 0..3 { assert!((after.0[k] - before.0[k] - [1.0, 2.0, 3.0][k]).abs() < 1e-4); }
        let obb0 = es.object_obb(d0).unwrap();
        es.rotate_pick(d0, Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), obb0.c);
        let obb1 = es.object_obb(d0).unwrap();
        assert!((obb1.c - obb0.c).length() < 1e-3, "rotation about the centre keeps the centre");
        assert!((obb1.hx - Quat::from_rotation_z(std::f32::consts::FRAC_PI_2) * obb0.hx).length() < 1e-3);
    }

    /// #h4-preview: the palette preview's contract on the Halo 4 scene - every palette row
    /// resolves to something drawable (never 0, which would leave the panel empty; a row whose
    /// object has no render model resolves to the marker cube), and an unknown tag is refused
    /// ("nothing to show", no rebuild). The pixels themselves are checked headlessly
    /// (HMS_PREVIEW_SHOT, see the headless page of the docs).
    #[test]
    fn palette_preview_contract() {
        let Some(es) = settler_scene() else { return };
        for e in es.palette() {
            for pv in &e.variants {
                let Some(tag) = pv.tag else { continue };
                assert!(es.preview_mode_tag(h4_tag(tag)).is_some(), "palette row '{}' has no previewable model tag", pv.name);
            }
        }
        // the marker cube IS drawable geometry, so a model-less row previews as a cube
        assert_eq!(es.model_geom(H4_MARKER_TAG).map(|g| g.tri_count()), Some(12));
        assert!(es.preview_mode_tag(h4_tag(0xFFFE)).is_none(), "an unknown tag has nothing to preview");
        assert!(es.preview_mode_tag(0x213E).is_none(), "a Reach tag id has nothing to preview");
    }

    #[test]
    fn marker_cube_is_pickable_and_named() {
        let g = marker_geom();
        assert_eq!(g.tri_count(), 12);
        let hit = raycast_mesh_local(&g.meshes[0].verts, &g.meshes[0].indices, Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z);
        assert!((hit.unwrap() - (5.0 - 2.0 * MARKER_HALF)).abs() < 1e-5, "top face at 2*half above the origin");
        assert!(h4_tag_index(H4_MARKER_TAG).is_some() && H4_MARKER_TAG != 0 && H4_MARKER_TAG != 0xFFFF_FFFF);
    }

    #[test]
    fn resolve_object_mode_marks_model_less_entries() {
        let Some(es) = settler_scene() else { return };
        // a palette entry whose obje has no render model resolves to the marker, not 0
        let mut markers = 0;
        let mut models = 0;
        for e in es.palette() {
            for pv in &e.variants {
                let Some(tag) = pv.tag else { continue };
                match es.resolve_object_mode(h4_tag(tag)) {
                    0 => panic!("known obje {} resolved to 0", pv.name),
                    H4_MARKER_TAG => markers += 1,
                    _ => models += 1,
                }
            }
        }
        eprintln!("ravine palette: {models} entries with a render model, {markers} markers");
        assert!(models > 0);
        assert_eq!(es.resolve_object_mode(0), 0);
        assert_eq!(es.resolve_object_mode(0x213E), 0, "a Reach tag id is not an H4 tag");
        // the pose comes from spawncam.rs: a variant loadout camera, else an initial
        // spawn (red team first) at eye height; either way it sits ON a variant object (no soup
        // offline, so no stand-off push)
        let spawn = es.spawn_camera_pose().expect("Settler has initial spawns");
        let (_, v) = es.variant().unwrap();
        let on_object = v.objects.iter().any(|o| {
            let p = Vec3::from(o.pos);
            (o.object_type == 28 && (p - spawn.0).length() < 1e-3) || (o.object_type == 16 && (p + Vec3::Z * crate::h4::spawncam::EYE_HEIGHT - spawn.0).length() < 1e-3)
        });
        assert!(on_object, "spawn camera sits on a Settler camera / spawn object: {:?}", spawn.0);
    }
}
