//! `ObjectScene`: the game-agnostic "object scene" surface the editor verbs talk to.
//!
//! Every editor verb (select / box-select / wireframe / gizmo / place / dup / delete / undo /
//! magnet / settle / script `pick`) operates on `hms_ipc::ObjectInfo` + `ObjMeta` plus a
//! small set of scene queries. This trait is that set:
//! `SceneController` (Reach) forwards to its own methods verbatim, and
//! `h4::edit_scene::H4ObjectScene` implements the same surface over the pure-Rust Halo 4 reader,
//! so `App` can dispatch through `objscene()` / `objscene_mut()` without the verbs knowing which
//! game is loaded. Reach-only calls (`forge_palette_full`, `object_material`, particles, lights,
//! background pre-decode) stay on `scene_ctl` and are simply skipped for Halo 4.

use std::collections::{HashMap, HashSet};

use eframe::wgpu;
use glam::{Mat4, Quat, Vec3};
use hms_ipc::ObjectInfo;
use hms_render::{GpuMesh, MeshRenderer};

/// The five dynamic renderer lanes a rebuild refreshes (`set_dynamic_*`): the Reach 5-tuple
/// from `SceneController::maybe_rebuild`, named.
pub struct RebuiltLanes {
    pub opaque: Vec<GpuMesh>,
    pub cutout: Vec<GpuMesh>,
    pub holo: Vec<GpuMesh>,
    pub holo_solid: Vec<GpuMesh>,
    pub blend: Vec<GpuMesh>,
}

impl RebuiltLanes {
    pub fn from_tuple(t: (Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>, Vec<GpuMesh>)) -> Self {
        RebuiltLanes { opaque: t.0, cutout: t.1, holo: t.2, holo_solid: t.3, blend: t.4 }
    }
    /// An empty set of lanes (the accumulator the Reach actor build fills, one actor at a time).
    pub fn empty() -> Self {
        RebuiltLanes { opaque: Vec::new(), cutout: Vec::new(), holo: Vec::new(), holo_solid: Vec::new(), blend: Vec::new() }
    }
    /// #sword-blade Append another build's five lanes, lane for lane. Used by the Reach actor
    /// build to accumulate one biped + its held weapon + the next actor without ever moving a
    /// transparent mesh into the opaque list (which is exactly the bug this replaced).
    pub fn extend(&mut self, mut o: RebuiltLanes) {
        self.opaque.append(&mut o.opaque);
        self.cutout.append(&mut o.cutout);
        self.holo.append(&mut o.holo);
        self.holo_solid.append(&mut o.holo_solid);
        self.blend.append(&mut o.blend);
    }
    /// `(opaque, cutout, holo, holo_solid, blend)` mesh counts — what the gates measure.
    pub fn counts(&self) -> (usize, usize, usize, usize, usize) {
        (self.opaque.len(), self.cutout.len(), self.holo.len(), self.holo_solid.len(), self.blend.len())
    }
}

pub trait ObjectScene {
    /// A map is open and objects can be built / picked (the Reach `has_cache()` gate).
    fn has_cache(&self) -> bool;
    /// Force the next `maybe_rebuild` to rebuild (positions / colours changed).
    fn invalidate(&mut self);
    /// True while a budgeted rebuild still has models left to decode on a later frame.
    fn rebuild_pending(&self) -> bool;
    /// Rebuild the dynamic lanes from a fresh object snapshot if it changed; None = no change.
    fn maybe_rebuild(
        &mut self,
        objects: &[ObjectInfo],
        colors: &HashMap<u32, (u8, u8)>,
        mr: &MeshRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<RebuiltLanes>;
    /// Per-datum visual scale multipliers used by the next rebuild (non-unity entries only).
    fn set_object_scales(&mut self, scales: HashMap<u32, f32>);
    /// #vehicle-ride  Per-datum WORLD TRANSFORM overrides for the next rebuild: draw a placement
    /// somewhere other than where the variant puts it, without touching the variant.
    ///
    /// Empty is the default and is bit-for-bit the authored pose, so a backend that has not
    /// implemented it is simply one where a ride does not move the real object — it is not one
    /// that draws it in the wrong place.
    fn set_object_transforms(&mut self, _transforms: HashMap<u32, Mat4>) {}
    /// Datums forced to cast a sun shadow at the next rebuild.
    fn set_object_casters(&mut self, casters: HashSet<u32>);
    /// #damage Per-datum DISPLAYED damage state (the DAMAGE pseudo-flag: 1 minor .. 4 destroyed;
    /// absent = the undamaged default) used by the next rebuild. Changes which render_model
    /// permutations the object draws, nothing else. Default: no game path (ignored).
    fn set_object_damage(&mut self, _states: HashMap<u32, u8>) {}
    /// #damage The damage states this placement's model authors a permutation for (union over
    /// its hlmt variant's regions and its variant attachments), ascending, never 0. Empty =
    /// default only. Default: none known.
    fn damage_states_of(&self, _o: &ObjectInfo) -> Vec<u8> { Vec::new() }

    // --- picking / bounds ---
    /// Nearest object along a world ray (AABB broad-phase + triangle refine) -> its datum.
    fn pick(&self, origin: Vec3, dir: Vec3) -> Option<u32>;
    /// World AABB of a placed object (from the cached pick list).
    fn aabb_of(&self, datum: u32) -> Option<([f32; 3], [f32; 3])>;
    /// Oriented world box of a placed object (local model AABB x placement transform).
    fn object_obb(&self, datum: u32) -> Option<crate::construct::Obb>;
    /// World-space line list of the object's mesh edges (None when the mesh isn't decoded).
    fn selection_wireframe(&self, datum: u32) -> Option<Vec<[f32; 3]>>;
    /// Keep the cached pick entry truthful between rebuilds after a translation.
    fn translate_pick(&mut self, datum: u32, delta: Vec3);
    /// Keep the cached pick entry truthful between rebuilds after a rotation about `pivot`.
    fn rotate_pick(&mut self, datum: u32, q: Quat, pivot: Vec3);
    /// Nearest map-geometry (BSP) hit point along a ray.
    fn raycast_scene(&self, origin: Vec3, dir: Vec3) -> Option<Vec3>;
    /// Same, plus the hit triangle's normal.
    fn raycast_scene_n(&self, origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)>;
    /// Nearest OTHER-object surface hit point along a ray (the moved objects excluded).
    fn raycast_objects_excluding(&self, origin: Vec3, dir: Vec3, exclude: &[u32]) -> Option<Vec3>;
    /// Overall world AABB of the loaded map geometry.
    fn scene_bounds(&self) -> Option<([f32; 3], [f32; 3])>;

    // --- identity / palette ---
    /// Object tag -> render-model tag (0 when unresolved).
    fn resolve_object_mode(&self, obj_tag: u32) -> u32;
    /// Tag path for diagnostics ("" when unknown).
    fn tag_name_of(&self, tag: u32) -> String;
    /// Tag class fourcc as text (None when unknown).
    fn raw_tag_class(&self, tag: u32) -> Option<String>;
    /// Hidden-blocker overlay: (solid tris (pos, rgba), outline segments (a, b, rgb)).
    fn blocker_overlay(&self) -> (&[([f32; 3], [f32; 4])], &[([f32; 3], [f32; 3], [f32; 3])]);
    /// #h4-phys World-space edges of an object's COLLISION hull (`coll`), posed by the placement.
    /// Empty when the object resolves no collision model. Drawn for the selected object only
    /// (View > Overlays > "Collision (selected)"); both games implement it, so the row lights up
    /// on Halo 4 / H2A as well as Reach.
    fn collision_world_edges(&self, o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])>;
    /// #h4-phys World-space edges of an object's PHYSICS hull (`phmo` rigid-body shapes).
    fn physics_world_edges(&self, o: &ObjectInfo) -> Vec<([f32; 3], [f32; 3])>;
    /// #settle The object's settle body: its `phmo` rigid-body shapes in model space at the
    /// object's EDITOR SCALE (the same scale the mesh, the pick entry and the hull overlays use),
    /// welded into one body (`physics_settle::PhysBody`). Objects without a `phmo` fall back to a
    /// convex hull of their `coll` model. None = no physics model at all.
    fn physics_body(&self, _o: &ObjectInfo) -> Option<crate::physics_settle::PhysBody> { None }
    /// #settle World-space triangles of the object's physics hull (else its collision hull), posed
    /// and scaled like the overlays: what OTHER objects settle against.
    fn physics_world_tris(&self, _o: &ObjectInfo) -> Vec<[Vec3; 3]> { Vec::new() }
    /// #collision  The object's CLOSED convex shells (one per box / polyhedron shape, triangles
    /// wound outward, placed and scaled like [`physics_world_tris`](Self::physics_world_tris)).
    /// Empty when the physics model is not made only of such shapes: those objects have no known
    /// inside, so a body overlapping them is only pushed along the nearest face.
    fn physics_world_shells(&self, _o: &ObjectInfo) -> Vec<Vec<[Vec3; 3]>> { Vec::new() }
    /// #settle #play Level geometry triangles overlapping a world box (the raycast soup),
    /// appended.
    ///
    /// **NO DEFAULT, deliberately.** This is what test mode collides the player against, so a
    /// scene that answers nothing is a frictionless void — the player falls through the floor and
    /// nothing in the code says why. It shipped that way for Halo 3, which had the triangle soup
    /// all along and simply never wired it up, and an empty default is what let that happen
    /// quietly. A new game now has to answer the question, even if the answer is `Vec::new()`
    /// with a comment saying so.
    fn static_tris_in_aabb(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>);
    /// #wire-attach Print-only diagnostic: (pick entries this datum owns, how many of them are
    /// ATTACHMENTS). A vehicle whose turret is a child object reports (2, 1); one whose weapons are
    /// model regions reports (1, 0). Lets a script prove the selection really covers the turret.
    fn pick_entry_counts(&self, _datum: u32) -> (usize, usize) { (0, 0) }
    /// The structure-BSP table as text (Reach) / the variant bounds report (Halo 4).
    fn bsp_flags_report(&self) -> String;
    /// A sensible spawn camera (pos, yaw, pitch) for the loaded map, when known.
    fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)>;
    /// The same camera with its SOURCE (`spawn_camera.rs`: which spawn / starting location /
    /// volume rule produced it) for the load log. Default: the pose, labelled as unknown.
    fn spawn_camera(&self) -> Option<crate::spawn_camera::Pose> {
        self.spawn_camera_pose().map(|(pos, yaw, pitch)| crate::spawn_camera::Pose { pos, yaw, pitch, source: crate::spawn_camera::Source::Spawn(crate::spawn_camera::Kind::RespawnPoint), raw_pos: pos })
    }
    /// Halo 4 only: (object caster AABB for the sun shadow fit, the BSP's floating-shadow
    /// cascade) for `SceneRenderer::set_h4_shadow`, re-read after every rebuild. None = Reach scene.
    fn h4_shadow_setup(&self) -> Option<(Option<(Vec3, Vec3)>, Option<hms_render::H4CascadeCfg>)> { None }

    // --- camera clearance (Reach overrides with its own primitive) ---
    /// #play-sound  The BSP render-material index of the level triangle under `origin` (a ray
    /// straight down, at most `max` long) -- what a footstep is keyed on. `None` where the level
    /// geometry is not what is underfoot.
    fn ground_material_index(&self, _origin: Vec3, _max: f32) -> Option<i32> {
        None
    }
    /// Distance to the nearest solid surface along a ray - map geometry AND placed objects -
    /// capped at `max`. Default: the two raycasts above, so the Halo 4 scene gets it for free.
    fn raycast_solid(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> {
        self.raycast_solid_excluding(origin, dir, max, &[])
    }
    /// #vehicle-suspension  `raycast_solid` ignoring some placements -- a RIDDEN vehicle's own
    /// collision hull, which is otherwise solid to the ground ray cast from its own wheels.
    ///
    /// The ride hides the placement from the RENDER set, and that is deliberately not the pick set
    /// a ray walks: the editor's table has to keep seeing the vehicle where the user put it. So the
    /// exclusion is passed in rather than inferred.
    fn raycast_solid_excluding(&self, origin: Vec3, dir: Vec3, max: f32, exclude: &[u32]) -> Option<f32> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        let mut best = self.raycast_scene(origin, d).map(|p| (p - origin).length());
        if let Some(p) = self.raycast_objects_excluding(origin, d, exclude) {
            let t = (p - origin).length();
            if best.map_or(true, |b| t < b) { best = Some(t); }
        }
        best.filter(|t| *t <= max)
    }
    /// Nearest solid surface around `pos` in any direction within `max` (distance, direction,
    /// blocked probe rays); `None` = open space. The Reach `nearest_solid` over `raycast_solid`.
    fn nearest_solid(&self, pos: Vec3, max: f32) -> (Option<f32>, Vec3, usize) {
        crate::scene::nearest_solid_with(&|o, d, m| self.raycast_solid(o, d, m), pos, max)
    }
    /// A position near `pos` with nothing solid within `clearance` wu (unchanged when already
    /// clear or when no escape exists). The Reach `standoff` solver over `raycast_solid`.
    fn standoff(&self, pos: Vec3, clearance: f32) -> Vec3 {
        crate::scene::standoff_with(&|o, d, m| self.raycast_solid(o, d, m), pos, clearance)
    }
}

/// The active object scene by FIELD (not `&mut App`), so a caller can hold `renderer` /
/// `render_state` borrows alongside (tick_scene's rebuild). Halo 4 wins while `h4_active`
/// (a Halo 4 load closes the Reach cache, so the two are never both live).
pub(crate) fn objscene_parts_mut<'a>(
    h4_active: bool,
    h4: &'a mut Option<Box<crate::h4::edit_scene::H4ObjectScene>>,
    reach: &'a mut Option<crate::scene::SceneController>,
    // #h3-edit: the Halo 3 editor scene wins while `h3_active` (its map closes the Reach cache too)
    h3_active: bool,
    h3: &'a mut Option<Box<crate::h3::edit_scene::H3ObjectScene>>,
) -> Option<&'a mut dyn ObjectScene> {
    if h4_active {
        h4.as_deref_mut().map(|s| s as &mut dyn ObjectScene)
    } else if h3_active {
        h3.as_deref_mut().map(|s| s as &mut dyn ObjectScene)
    } else {
        reach.as_mut().map(|s| s as &mut dyn ObjectScene)
    }
}

impl crate::App {
    /// The object scene the editor verbs talk to: the Halo 4 scene while a Halo 4 map is active,
    /// the Halo 3 scene while a Halo 3 map is (#h3-edit), else the Reach `SceneController` (None
    /// before any map is loaded).
    pub(crate) fn objscene(&self) -> Option<&dyn ObjectScene> {
        if self.h4_active {
            self.h4_scene.as_deref().map(|s| s as &dyn ObjectScene)
        } else if self.h3_active {
            self.h3_scene.as_deref().map(|s| s as &dyn ObjectScene)
        } else {
            self.scene_ctl.as_ref().map(|s| s as &dyn ObjectScene)
        }
    }
    pub(crate) fn objscene_mut(&mut self) -> Option<&mut dyn ObjectScene> {
        objscene_parts_mut(self.h4_active, &mut self.h4_scene, &mut self.scene_ctl, self.h3_active, &mut self.h3_scene)
    }
}

// ===================== selection wireframe (#wire-cover) =====================

/// Line budget for ONE selected object's wireframe (2 vertices per line). A safeguard for the
/// highlight upload, never a cut-off: an object with more unique edges than this is thinned
/// EVENLY by [`fit_wire_budget`], so every mesh, part and attachment keeps its outline. The old
/// 60 000-line cap was first-come: a Halo 4 Mantis (28 059 triangles = 84 177 lines at three per
/// triangle) ran out of budget two thirds of the way through its meshes, and its last five
/// meshes - a leg, the guns - had no wireframe at all.
pub const WIRE_MAX_LINES: usize = 150_000;

/// Push every UNIQUE edge of an indexed triangle list as a world-space line (vertex pair). A
/// closed mesh shares nearly every edge between two triangles, so emitting three lines per
/// triangle drew almost every edge twice (identical pixels, double the budget). `pos(i)` is the
/// LOCAL position of vertex `i`, None for an out-of-range index (that edge is skipped).
pub fn push_wire_edges(out: &mut Vec<[f32; 3]>, indices: &[u32], xform: &Mat4, pos: impl Fn(usize) -> Option<Vec3>) {
    let nv = indices.iter().copied().max().map_or(0, |m| m as usize + 1);
    // each vertex is transformed once, on first use
    let mut world: Vec<Option<[f32; 3]>> = vec![None; nv];
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(indices.len());
    for t in indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            if a == b || !seen.insert((a.min(b), a.max(b))) { continue; }
            let mut at = |i: u32| -> Option<[f32; 3]> {
                let slot = &mut world[i as usize];
                if slot.is_none() { *slot = pos(i as usize).map(|p| xform.transform_point3(p).to_array()); }
                *slot
            };
            let (Some(pa), Some(pb)) = (at(a), at(b)) else { continue };
            out.push(pa);
            out.push(pb);
        }
    }
}

/// Thin a line list (vertex pairs) EVENLY to at most `max_lines`: a uniform stride through the
/// whole list, so a dense object stays outlined everywhere rather than losing its tail meshes.
pub fn fit_wire_budget(lines: &mut Vec<[f32; 3]>, max_lines: usize) {
    let n = lines.len() / 2;
    if n <= max_lines || max_lines == 0 { return; }
    let mut kept = Vec::with_capacity(max_lines * 2);
    for k in 0..max_lines {
        let i = (k as u64 * n as u64 / max_lines as u64) as usize;
        kept.push(lines[i * 2]);
        kept.push(lines[i * 2 + 1]);
    }
    *lines = kept;
}

/// `HMS_WIRE_DIAG=1`: print-only per-entry edge counts of the selection wireframe.
pub fn wire_diag() -> bool { std::env::var("HMS_WIRE_DIAG").is_ok() }

#[cfg(test)]
mod wire_tests {
    use super::*;

    /// Two triangles sharing an edge = 5 unique edges, not 6; a degenerate edge is dropped.
    #[test]
    fn unique_edges_and_even_thinning() {
        let verts = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE];
        let mut out = Vec::new();
        push_wire_edges(&mut out, &[0, 1, 2, 2, 1, 3, 3, 3, 0], &Mat4::from_translation(Vec3::Z), |i| verts.get(i).copied());
        assert_eq!(out.len() / 2, 5 + 1, "5 shared-quad edges + the (3,0) edge of the degenerate triangle");
        assert_eq!(out[0], [0.0, 0.0, 1.0], "transformed by the object matrix");
        // out-of-range index: that edge is skipped, the rest stays
        let mut out = Vec::new();
        push_wire_edges(&mut out, &[0, 1, 9], &Mat4::IDENTITY, |i| verts.get(i).copied());
        assert_eq!(out.len() / 2, 1);
        // thinning keeps a uniform stride over the WHOLE list (first and last thirds both survive)
        let mut lines: Vec<[f32; 3]> = (0..300u32).flat_map(|i| [[i as f32, 0.0, 0.0], [i as f32, 1.0, 0.0]]).collect();
        fit_wire_budget(&mut lines, 100);
        assert_eq!(lines.len() / 2, 100);
        assert_eq!(lines[0][0], 0.0);
        assert_eq!(lines[198][0], 297.0);
        let mut small = lines.clone();
        fit_wire_budget(&mut small, 100);
        assert_eq!(small, lines, "under budget = untouched");
    }
}
