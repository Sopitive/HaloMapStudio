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
use glam::{Quat, Vec3};
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
    /// Datums forced to cast a sun shadow at the next rebuild.
    fn set_object_casters(&mut self, casters: HashSet<u32>);

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
    /// #wire-attach Print-only diagnostic: (pick entries this datum owns, how many of them are
    /// ATTACHMENTS). A vehicle whose turret is a child object reports (2, 1); one whose weapons are
    /// model regions reports (1, 0). Lets a script prove the selection really covers the turret.
    fn pick_entry_counts(&self, _datum: u32) -> (usize, usize) { (0, 0) }
    /// The structure-BSP table as text (Reach) / the variant bounds report (Halo 4).
    fn bsp_flags_report(&self) -> String;
    /// A sensible spawn camera (pos, yaw, pitch) for the loaded map, when known.
    fn spawn_camera_pose(&self) -> Option<(Vec3, f32, f32)>;
    /// Halo 4 only: (object caster AABB for the sun shadow fit, the BSP's floating-shadow
    /// cascade) for `SceneRenderer::set_h4_shadow`, re-read after every rebuild. None = Reach scene.
    fn h4_shadow_setup(&self) -> Option<(Option<(Vec3, Vec3)>, Option<hms_render::H4CascadeCfg>)> { None }

    // --- camera clearance (Reach overrides with its own primitive) ---
    /// Distance to the nearest solid surface along a ray - map geometry AND placed objects -
    /// capped at `max`. Default: the two raycasts above, so the Halo 4 scene gets it for free.
    fn raycast_solid(&self, origin: Vec3, dir: Vec3, max: f32) -> Option<f32> {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO { return None; }
        let mut best = self.raycast_scene(origin, d).map(|p| (p - origin).length());
        if let Some(p) = self.raycast_objects_excluding(origin, d, &[]) {
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
) -> Option<&'a mut dyn ObjectScene> {
    if h4_active {
        h4.as_deref_mut().map(|s| s as &mut dyn ObjectScene)
    } else {
        reach.as_mut().map(|s| s as &mut dyn ObjectScene)
    }
}

impl crate::App {
    /// The object scene the editor verbs talk to: the Halo 4 scene while a Halo 4 map is active,
    /// else the Reach `SceneController` (None before any map is loaded).
    pub(crate) fn objscene(&self) -> Option<&dyn ObjectScene> {
        if self.h4_active {
            self.h4_scene.as_deref().map(|s| s as &dyn ObjectScene)
        } else {
            self.scene_ctl.as_ref().map(|s| s as &dyn ObjectScene)
        }
    }
    pub(crate) fn objscene_mut(&mut self) -> Option<&mut dyn ObjectScene> {
        objscene_parts_mut(self.h4_active, &mut self.h4_scene, &mut self.scene_ctl)
    }
}
