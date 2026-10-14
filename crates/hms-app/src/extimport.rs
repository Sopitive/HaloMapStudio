//! Externally imported models (OBJ/GLB) placed in the scene with a live, user-driven transform.
//!
//! The import feature started as a FIXED base mesh: the file was parsed, uploaded with an identity
//! instance matrix, and its triangles were flattened into one world-space soup for the
//! snap-to-surface raycasts — so the model could never be moved (issue #4). This module is the
//! placement half of the fix: each import keeps its triangles in MODEL space plus a pose
//! (position + forward/up basis + uniform scale), and everything downstream reads
//! [`ImportedModel::matrix`] / [`ImportedModel::world_tris`]:
//!
//!   * the renderer gets the matrix as the mesh's single instance transform;
//!   * `import_tris` (snap-to-surface, the fill tool, picking, `physics_settle`) is rebuilt from
//!     `world_tris()`, so a moved model moves what objects land on;
//!   * the project file stores the pose, so a reopened project puts the model back.
//!
//! The pose uses the SAME representation as a placed forge object — a position and a
//! forward/up pair (see [`crate::euler`]) — so the editor's existing modal grab/rotate, its gizmo
//! and its numeric angle fields all drive an import through one pseudo-datum
//! ([`import_datum`]) with no parallel transform code.

use glam::{Mat4, Vec3};
use std::path::PathBuf;

/// Top nibble of an imported model's PSEUDO-datum. Real object datums are 0xD… (variant),
/// 0xE… (scenario) and 0xF… (locally placed), so 0xC… cannot collide with one, and
/// `App::object_datum_present` only validates the three real families.
pub const IMPORT_DATUM_TAG: u32 = 0xC000_0000;

/// The pseudo-datum that selects imported model `idx`.
pub fn import_datum(idx: usize) -> u32 {
    IMPORT_DATUM_TAG | (idx as u32 & 0x0FFF_FFFF)
}

/// The imported-model index a pseudo-datum refers to, or None when it is a real object datum.
pub fn datum_index(datum: u32) -> Option<usize> {
    (datum & 0xF000_0000 == IMPORT_DATUM_TAG).then(|| (datum & 0x0FFF_FFFF) as usize)
}

/// An orthonormal (forward, left, up) basis from a possibly sloppy forward/up pair.
/// Same rule as [`crate::euler`]'s private `basis` (Gram-Schmidt with forward as the primary
/// axis), so the angles the panel shows and the matrix the renderer gets never disagree.
pub fn basis(fwd: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let x = fwd.normalize_or_zero();
    let x = if x.length_squared() < 0.5 { Vec3::X } else { x };
    let mut z = up - x * up.dot(x);
    if z.length_squared() < 1e-12 {
        let r = if x.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        z = r - x * r.dot(x);
    }
    let z = z.normalize_or_zero();
    let y = z.cross(x);
    (x, y, z)
}

/// One imported model file: its model-space geometry plus the pose the user drives.
#[derive(Clone)]
pub struct ImportedModel {
    /// Where it was imported from (re-read on a project load).
    pub path: PathBuf,
    /// File name, for the list rows and the status line.
    pub name: String,
    pub pos: [f32; 3],
    /// Orientation as a forge object stores it: forward (+X at rest) and up (+Z at rest).
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    /// UNIFORM scale. Non-uniform scaling would need a normal matrix in the shader
    /// (`vs_impl` rotates the normal by the model matrix), so it is deliberately one number.
    pub scale: f32,
    /// The parsed triangles in MODEL space (already Y-up → Halo Z-up converted at import).
    /// Never mutated by a transform — `world_tris` applies the pose instead, so repeated moves
    /// cannot accumulate drift.
    pub base_tris: Vec<[Vec3; 3]>,
    /// Model-space bounds of `base_tris` (the selection box + the "drop it in front of the
    /// camera" default placement).
    pub bmin: [f32; 3],
    pub bmax: [f32; 3],
    /// #model-import: what the file yielded — "GLB · 296 parts · 11 materials, 4 textures
    /// (up to 2048x2048)". Shown in the panel row and the import status line so the user can see
    /// at a glance whether the model brought any materials with it. Derived from the file, never
    /// saved, so a project load recomputes it.
    pub summary: String,
}

impl ImportedModel {
    /// A freshly imported model at the identity pose (what the pre-#4 fixed base mesh rendered).
    pub fn at_identity(path: PathBuf, base_tris: Vec<[Vec3; 3]>) -> Self {
        let (bmin, bmax) = tri_bounds(&base_tris);
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        Self { path, name, pos: [0.0; 3], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], scale: 1.0, base_tris, bmin: bmin.into(), bmax: bmax.into(), summary: String::new() }
    }

    /// Model → world matrix: the orthonormal basis columns scaled by `scale`, translated to `pos`.
    /// `scale <= 0` is clamped to a hair above zero so a mistyped 0 cannot collapse the mesh to a
    /// point the user can no longer see or pick.
    pub fn matrix(&self) -> Mat4 {
        let (x, y, z) = basis(Vec3::from(self.fwd), Vec3::from(self.up));
        let s = if self.scale.is_finite() { self.scale.clamp(1e-3, 1.0e4) } else { 1.0 };
        Mat4::from_cols(
            (x * s).extend(0.0),
            (y * s).extend(0.0),
            (z * s).extend(0.0),
            Vec3::from(self.pos).extend(1.0),
        )
    }

    /// The model's triangles in WORLD space — the soup every raycast consumer uses
    /// (snap-to-surface placement, the fill tool, `physics_settle`, viewport picking).
    pub fn world_tris(&self) -> Vec<[Vec3; 3]> {
        let m = self.matrix();
        self.base_tris
            .iter()
            .map(|t| [m.transform_point3(t[0]), m.transform_point3(t[1]), m.transform_point3(t[2])])
            .collect()
    }

    /// The ray `origin + t*dir` (world space) pushed into MODEL space.
    ///
    /// Picking a model must not allocate: a converted BSP is millions of triangles, and building
    /// its world soup on every click would stall the UI. Transforming the RAY instead is two
    /// matrix-vector products, and because the scale is uniform the `t` that comes back from the
    /// model-space test is already the world distance: the hit is
    /// `M(o' + t·d') = M·o' + t·M_linear(d') = origin + t·dir`.
    pub fn ray_to_model(&self, origin: Vec3, dir: Vec3) -> (Vec3, Vec3) {
        let inv = self.matrix().inverse();
        (inv.transform_point3(origin), inv.transform_vector3(dir))
    }

    /// Rotation as (x_roll, y_pitch, z_yaw) degrees — the panel's numeric angle fields.
    pub fn euler_deg(&self) -> [f32; 3] {
        crate::euler::to_euler_deg(self.fwd, self.up)
    }

    /// Set the rotation from (x_roll, y_pitch, z_yaw) degrees.
    pub fn set_euler_deg(&mut self, deg: [f32; 3]) {
        let (f, u) = crate::euler::from_euler_deg(deg, 1.0, 1.0);
        self.fwd = f;
        self.up = u;
    }

    /// World-space centre of the model's bounds (what "frame it" / "where is it" means; the
    /// gizmo pivot is the model ORIGIN, which may sit well away from the geometry).
    pub fn world_center(&self) -> Vec3 {
        self.matrix().transform_point3((Vec3::from(self.bmin) + Vec3::from(self.bmax)) * 0.5)
    }

    /// Longest world-space extent of the bounds — the "is it visible from here" camera distance.
    pub fn world_radius(&self) -> f32 {
        let s = if self.scale.is_finite() { self.scale.abs().max(1e-3) } else { 1.0 };
        (Vec3::from(self.bmax) - Vec3::from(self.bmin)).length() * 0.5 * s
    }

    /// The 12 edges of the model's ORIENTED bounds as line-segment endpoint PAIRS — the
    /// selection highlight (a forge object shows its own wireframe; an import has no model to
    /// walk, so its box is the selection feedback).
    pub fn selection_box_lines(&self) -> Vec<[f32; 3]> {
        let m = self.matrix();
        let (lo, hi) = (Vec3::from(self.bmin), Vec3::from(self.bmax));
        let c = |i: usize| -> [f32; 3] {
            let p = Vec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            m.transform_point3(p).to_array()
        };
        // 4 edges along X, 4 along Y, 4 along Z (corner index bits = x,y,z).
        const EDGES: [(usize, usize); 12] = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        let mut out = Vec::with_capacity(24);
        for (a, b) in EDGES {
            out.push(c(a));
            out.push(c(b));
        }
        out
    }
}

/// Axis-aligned bounds of a triangle soup (ZERO..ZERO for an empty one).
pub fn tri_bounds(tris: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for t in tris {
        for v in t {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_quad() -> Vec<[Vec3; 3]> {
        // A 2x2 plate on z = 0, centred on the origin (the "ground" an object can be dropped on).
        let (a, b, c, d) = (
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        );
        vec![[a, b, c], [a, c, d]]
    }

    /// The pseudo-datum must round-trip and must never look like a real object datum
    /// (0xD variant / 0xE scenario / 0xF locally placed).
    #[test]
    fn import_datums_round_trip_and_do_not_collide() {
        for i in [0usize, 1, 7, 4095] {
            let d = import_datum(i);
            assert_eq!(datum_index(d), Some(i), "datum 0x{d:08X}");
            assert!(!matches!(d >> 28, 0xD | 0xE | 0xF), "import datum 0x{d:08X} collides with a real family");
        }
        for real in [0xD000_0001u32, 0xE000_0000, 0xF000_0003] {
            assert_eq!(datum_index(real), None, "0x{real:08X} must not read as an import");
        }
    }

    /// The default pose must reproduce the pre-#4 behaviour EXACTLY: identity matrix,
    /// world triangles == the parsed model triangles.
    #[test]
    fn identity_pose_is_the_old_fixed_base_mesh() {
        let m = ImportedModel::at_identity(PathBuf::from("/x/plate.obj"), unit_quad());
        assert_eq!(m.name, "plate.obj");
        assert!((m.matrix() - Mat4::IDENTITY).to_cols_array().iter().all(|v| v.abs() < 1e-6), "default pose must be the identity");
        let w = m.world_tris();
        for (a, b) in w.iter().zip(unit_quad().iter()) {
            for k in 0..3 {
                assert!((a[k] - b[k]).length() < 1e-6, "identity moved a vertex: {:?} vs {:?}", a[k], b[k]);
            }
        }
        assert_eq!(m.bmin, [-1.0, -1.0, 0.0]);
        assert_eq!(m.bmax, [1.0, 1.0, 0.0]);
    }

    /// Translation, uniform scale and yaw must all reach the WORLD triangles — this is what
    /// makes snap-to-surface / settle agree with the picture.
    #[test]
    fn pose_reaches_the_world_triangles() {
        let mut m = ImportedModel::at_identity(PathBuf::from("plate.obj"), unit_quad());
        m.pos = [10.0, -4.0, 7.5];
        let w = m.world_tris();
        let (lo, hi) = tri_bounds(&w);
        assert!((lo - Vec3::new(9.0, -5.0, 7.5)).length() < 1e-5, "translated lo {lo:?}");
        assert!((hi - Vec3::new(11.0, -3.0, 7.5)).length() < 1e-5, "translated hi {hi:?}");

        m.scale = 3.0;
        let (lo, hi) = tri_bounds(&m.world_tris());
        assert!((lo - Vec3::new(7.0, -7.0, 7.5)).length() < 1e-5, "scaled lo {lo:?}");
        assert!((hi - Vec3::new(13.0, -1.0, 7.5)).length() < 1e-5, "scaled hi {hi:?}");

        // 90 deg of PITCH (about Y) stands the plate on edge: its z extent becomes +-3, x collapses.
        m.scale = 3.0;
        m.pos = [0.0; 3];
        m.set_euler_deg([0.0, 90.0, 0.0]);
        let (lo, hi) = tri_bounds(&m.world_tris());
        assert!(lo.x.abs() < 1e-4 && hi.x.abs() < 1e-4, "pitched plate should be flat in x, got {lo:?}..{hi:?}");
        assert!((hi.z - 3.0).abs() < 1e-4 && (lo.z + 3.0).abs() < 1e-4, "pitched plate z extent {lo:?}..{hi:?}");
    }

    /// The transform must be re-DERIVED from `base_tris` every time, never accumulated:
    /// moving a model a thousand times and back must land exactly where it started.
    #[test]
    fn repeated_moves_do_not_drift() {
        let mut m = ImportedModel::at_identity(PathBuf::from("plate.obj"), unit_quad());
        let before = m.world_tris();
        for _ in 0..1000 {
            m.pos = [m.pos[0] + 0.1, m.pos[1], m.pos[2]];
        }
        for _ in 0..1000 {
            m.pos = [m.pos[0] - 0.1, m.pos[1], m.pos[2]];
        }
        m.pos = [0.0; 3]; // the user typed 0 back in
        let after = m.world_tris();
        for (a, b) in after.iter().zip(before.iter()) {
            for k in 0..3 {
                assert!((a[k] - b[k]).length() < 1e-6, "drifted: {:?} vs {:?}", a[k], b[k]);
            }
        }
    }

    /// The euler fields must round-trip through the forward/up pair the gizmo rotates.
    #[test]
    fn euler_round_trips() {
        let mut m = ImportedModel::at_identity(PathBuf::from("a.obj"), unit_quad());
        for deg in [[0.0, 0.0, 0.0], [0.0, 0.0, 45.0], [15.0, -30.0, 120.0], [0.0, 0.0, -179.0]] {
            m.set_euler_deg(deg);
            let back = m.euler_deg();
            for k in 0..3 {
                assert!((back[k] - deg[k]).abs() < 0.05, "euler {deg:?} -> {back:?}");
            }
            // and the basis stays orthonormal, so the matrix has no shear/stretch
            let (x, y, z) = basis(Vec3::from(m.fwd), Vec3::from(m.up));
            assert!((x.length() - 1.0).abs() < 1e-5 && (y.length() - 1.0).abs() < 1e-5 && (z.length() - 1.0).abs() < 1e-5);
            assert!(x.dot(y).abs() < 1e-5 && x.dot(z).abs() < 1e-5 && y.dot(z).abs() < 1e-5);
            assert!(x.cross(y).dot(z) > 0.9, "basis must stay right-handed");
        }
    }

    /// A degenerate pose must not produce NaNs or a collapsed mesh (a mistyped 0 scale, a zero
    /// forward vector) — those would silently delete the user's model from the raycast soup.
    #[test]
    fn degenerate_pose_is_clamped() {
        let mut m = ImportedModel::at_identity(PathBuf::from("a.obj"), unit_quad());
        m.scale = 0.0;
        m.fwd = [0.0; 3];
        m.up = [0.0; 3];
        let w = m.world_tris();
        assert!(w.iter().all(|t| t.iter().all(|v| v.is_finite())), "degenerate pose produced NaNs");
        let (lo, hi) = tri_bounds(&w);
        assert!((hi - lo).length() > 0.0, "a 0 scale must not collapse the mesh to a point");
        m.scale = f32::NAN;
        assert!(m.matrix().to_cols_array().iter().all(|v| v.is_finite()), "NaN scale leaked into the matrix");
    }

    /// A ray pushed into model space must give the SAME hit distance as testing it against the
    /// world-space triangles — that equality is what lets picking skip building the world soup.
    #[test]
    fn model_space_ray_gives_the_world_distance() {
        // Möller-Trumbore, the same test `main.rs::raycast_tris` runs.
        fn hit(o: Vec3, d: Vec3, tris: &[[Vec3; 3]]) -> Option<f32> {
            let mut best = f32::MAX;
            for t in tris {
                let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
                let p = d.cross(e2);
                let det = e1.dot(p);
                if det.abs() < 1e-7 { continue; }
                let inv = 1.0 / det;
                let s = o - t[0];
                let u = s.dot(p) * inv;
                if !(0.0..=1.0).contains(&u) { continue; }
                let q = s.cross(e1);
                let v = d.dot(q) * inv;
                if v < 0.0 || u + v > 1.0 { continue; }
                let dist = e2.dot(q) * inv;
                if dist > 1e-4 && dist < best { best = dist; }
            }
            (best < f32::MAX).then_some(best)
        }
        let mut m = ImportedModel::at_identity(PathBuf::from("a.obj"), unit_quad());
        m.pos = [10.0, 20.0, 30.0];
        m.scale = 3.0;
        m.set_euler_deg([0.0, 0.0, 47.0]);
        // straight down onto the plate from 50 above it
        let (o, d) = (Vec3::new(10.0, 20.0, 80.0), Vec3::new(0.0, 0.0, -1.0));
        let world = hit(o, d, &m.world_tris()).expect("the world soup must be hit");
        let (lo, ld) = m.ray_to_model(o, d);
        let local = hit(lo, ld, &m.base_tris).expect("the model-space ray must hit the same plate");
        assert!((world - local).abs() < 1e-3, "model-space t {local} != world t {world}");
        assert!((world - 50.0).abs() < 1e-3, "the plate is 50 wu below the ray origin, got {world}");
    }

    /// The selection box must be 12 segments (24 endpoints) and must follow the pose.
    #[test]
    fn selection_box_follows_the_pose() {
        let mut m = ImportedModel::at_identity(PathBuf::from("a.obj"), unit_quad());
        m.pos = [5.0, 0.0, 0.0];
        m.scale = 2.0;
        let l = m.selection_box_lines();
        assert_eq!(l.len(), 24, "12 edges = 24 endpoints");
        let xs: Vec<f32> = l.iter().map(|p| p[0]).collect();
        assert!(xs.iter().all(|x| (3.0..=7.0).contains(x)), "box did not follow pos+scale: {xs:?}");
    }
}
