//! "Resolve physics": a physics-only rigid-body settle of ONE placed object. #settle
//!
//! The object is its `phmo` rigid-body shapes (spheres / pills / boxes / triangles / convex
//! polyhedra, in model space, at the object's EDITOR SCALE) and nothing else -- the render mesh is
//! never consulted, so a scaled object rests exactly where its scaled physics hull touches the
//! world. The world is the level's triangle soup plus the physics hulls of every other placed
//! object. The body is dropped under gravity and stepped at the engine's 30 Hz tick with a
//! sequential-impulse contact solver (Coulomb friction + restitution from the shape's own tag
//! fields, split-impulse penetration recovery) until it comes to rest, or reports that it never
//! touched anything.
//!
//! What this keeps from the engine (Havok in Halo: Reach / Halo 4):
//!   * the collision geometry itself: the tag's rigid-body shapes, the Havok convex radius
//!     (`hkpConvexShape::m_radius`, tag +0x40) added around boxes / polyhedra exactly as Havok
//!     does (a 0.968 wu `wall_m` core + 2 x 0.016 radius = the 1.0 wu block players touch),
//!     the rigid body's node frame (`mode` node inverse matrices), the tag mass, centre of mass,
//!     inertia tensor scale, linear / angular damping, and every shape's friction / restitution;
//!   * the tick rate (30 Hz), gravity along -Z, sequential impulses (Havok's solver is a PGS /
//!     sequential-impulse solver too), sleeping on a velocity threshold held over several ticks.
//! What it simplifies (stated so nobody mistakes it for a Havok port):
//!   * ONE rigid assembly per object: every non-phantom rigid body of the phmo is welded into a
//!     single body at the model's default pose (a Warthog's turret does not articulate). The
//!     SOLVER itself is no longer limited to that -- [`crate::ragdoll`] is this module's shapes
//!     and contact solver generalised to N bodies joined by the tag's own constraints, and
//!     `settle` is a one-link, no-joint call into it -- it is only this ENTRY POINT, the editor's
//!     "Simulate physics for selection", that welds the object into one body. A vehicle also gets
//!     its `vehi` ground-contact points -- friction points
//!     (tyres) and anti-gravity points (hover pads) -- as spheres at their markers, so it rests
//!     on its wheels like in-game (the Warthog's origin lands 0.05 wu above the ground, its
//!     chassis hull 0.2 wu up); their contacts are force-limited (`SUPPORT_G`) as a stand-in for
//!     the suspension, which is not otherwise simulated;
//!   * contacts are vertex-vs-triangle and triangle-vertex-vs-hull (no edge-edge pairs, capsules
//!     sample five spheres along their axis), the level's friction is taken equal to the object's
//!     (the level's per-material physics properties are not read), the inertia tensor is the tag
//!     one when it is well-formed else the shape geometry's, and the level is the render soup the
//!     editor already raycasts against (the structure BSP's collision BSP is not decoded).
//!   * the settle runs to rest in one call (no real-time animation); it is only ever started by
//!     an explicit user action.

use glam::{Mat3, Quat, Vec3};

/// The engine's physics tick.
pub const TICK_HZ: f32 = 30.0;
/// #gravity  **The ENGINE's own world gravity**, from [`crate::reach_physics::WORLD_GRAVITY`].
///
/// This held **3.2147** — Earth's 9.8 m/s^2 at 1 wu = 3.048 m, with two digits transposed from
/// the 3.2174 the particle lane holds. Earth gravity appears NOWHERE in `haloreach.dll` (the f32
/// patterns of 9.80665 and 3.217405 have zero hits), so it never had a basis; the engine's
/// `"physics constants"` block authors 4.1712594 and that is what a rigid body dropped in Reach's
/// world falls at.
///
/// **This is a behaviour change: props settle about 30% faster.** Only the dynamics depend on it
/// (bounce heights, time to rest) and the rest pose does not, which is the whole point of the
/// feature — but it is a change and it is in the CHANGELOG.
pub const GRAVITY_WU_S2: f32 = crate::reach_physics::WORLD_GRAVITY;

// ------------------------------------------------------------------------------------------
// Shapes / bodies
// ------------------------------------------------------------------------------------------

/// Per-shape Havok primitive fields (tag +0x0C / +0x10 / +0x18 of every shape element).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeMaterial {
    pub friction: f32,
    pub restitution: f32,
    pub mass: f32,
}

impl Default for ShapeMaterial {
    fn default() -> Self { ShapeMaterial { friction: 0.5, restitution: 0.2, mass: 0.0 } }
}

impl ShapeMaterial {
    fn sane(self) -> Self {
        ShapeMaterial {
            friction: if self.friction.is_finite() { self.friction.clamp(0.0, 4.0) } else { 0.5 },
            restitution: if self.restitution.is_finite() { self.restitution.clamp(0.0, 1.0) } else { 0.2 },
            mass: if self.mass.is_finite() && self.mass > 0.0 { self.mass } else { 0.0 },
        }
    }
}

/// One convex collision shape in BODY (model) space.
#[derive(Clone, Debug)]
pub enum ShapeGeom {
    Sphere { c: Vec3, r: f32 },
    Capsule { a: Vec3, b: Vec3, r: f32 },
    /// Convex vertex set, its outward face planes (`n . x <= d` inside) and the Havok convex
    /// radius that pads the vertices outward.
    Hull { pts: Vec<Vec3>, planes: Vec<(Vec3, f32)>, r: f32 },
}

#[derive(Clone, Debug)]
pub struct ConvexShape {
    pub geom: ShapeGeom,
    pub mat: ShapeMaterial,
    /// A vehicle ground-contact sphere (tyre / hover pad): its contacts are force-limited like a
    /// suspension (`SUPPORT_G`) and never bounce. False for every phmo shape.
    pub support: bool,
}

/// Above this many vertices a hull's face planes come from its AABB (the O(n^3) face scan is
/// for the small Forge polyhedra; the vertices are still used for contacts).
const HULL_PLANE_LIMIT: usize = 64;
/// Sanity cap on any shape's extent (a corrupt tag never builds a map-sized body).
const MAX_EXTENT: f32 = 500.0;
/// How far past a triangle's edge (in its plane) a vertex that crossed the plane still counts as
/// that triangle's contact (see `shape_contacts`).
const EDGE_MARGIN: f32 = 0.05;

impl ConvexShape {
    pub fn sphere(c: Vec3, r: f32, mat: ShapeMaterial) -> Option<Self> {
        (c.is_finite() && r.is_finite() && r > 0.0 && r < MAX_EXTENT)
            .then(|| ConvexShape { geom: ShapeGeom::Sphere { c, r }, mat: mat.sane(), support: false })
    }
    pub fn capsule(a: Vec3, b: Vec3, r: f32, mat: ShapeMaterial) -> Option<Self> {
        if !(a.is_finite() && b.is_finite() && r.is_finite()) || r <= 0.0 || r >= MAX_EXTENT { return None; }
        if (b - a).length() < 1e-5 { return Self::sphere(a, r, mat); }
        Some(ConvexShape { geom: ShapeGeom::Capsule { a, b, r }, mat: mat.sane(), support: false })
    }
    /// A convex hull of `pts` padded by the convex radius `r` (>= 0). Fewer than 3 finite points,
    /// or a point cloud larger than `MAX_EXTENT`, is rejected.
    pub fn hull(pts: &[Vec3], r: f32, mat: ShapeMaterial) -> Option<Self> {
        let mut p: Vec<Vec3> = pts.iter().copied().filter(|v| v.is_finite()).collect();
        p.dedup_by(|a, b| (*a - *b).length_squared() < 1e-10);
        if p.len() < 3 { return None; }
        let (mn, mx) = aabb_of_points(&p);
        if (mx - mn).max_element() >= MAX_EXTENT { return None; }
        let r = if r.is_finite() { r.clamp(0.0, 0.25) } else { 0.0 };
        let planes = hull_planes(&p);
        Some(ConvexShape { geom: ShapeGeom::Hull { pts: p, planes, r }, mat: mat.sane(), support: false })
    }
    /// An axis-aligned box of half extents `h` at `c` (a 6-plane hull).
    pub fn aabb_box(c: Vec3, h: Vec3, r: f32, mat: ShapeMaterial) -> Option<Self> {
        let mut pts = Vec::with_capacity(8);
        for k in 0..8u32 {
            pts.push(c + Vec3::new(
                if k & 4 != 0 { h.x } else { -h.x },
                if k & 2 != 0 { h.y } else { -h.y },
                if k & 1 != 0 { h.z } else { -h.z },
            ));
        }
        Self::hull(&pts, r, mat)
    }

    /// The same shape at a uniform scale about the body origin (radii scale too: a scaled Forge
    /// object's physics hull scales with it).
    pub fn scaled(&self, s: f32) -> Self {
        let geom = match &self.geom {
            ShapeGeom::Sphere { c, r } => ShapeGeom::Sphere { c: *c * s, r: *r * s },
            ShapeGeom::Capsule { a, b, r } => ShapeGeom::Capsule { a: *a * s, b: *b * s, r: *r * s },
            ShapeGeom::Hull { pts, planes, r } => ShapeGeom::Hull {
                pts: pts.iter().map(|p| *p * s).collect(),
                planes: planes.iter().map(|(n, d)| (*n, *d * s)).collect(),
                r: *r * s,
            },
        };
        ConvexShape { geom, mat: self.mat, support: self.support }
    }

    /// Body-space AABB including the convex radius.
    pub fn aabb(&self) -> (Vec3, Vec3) {
        match &self.geom {
            ShapeGeom::Sphere { c, r } => (*c - Vec3::splat(*r), *c + Vec3::splat(*r)),
            ShapeGeom::Capsule { a, b, r } => (a.min(*b) - Vec3::splat(*r), a.max(*b) + Vec3::splat(*r)),
            ShapeGeom::Hull { pts, r, .. } => {
                let (mn, mx) = aabb_of_points(pts);
                (mn - Vec3::splat(*r), mx + Vec3::splat(*r))
            }
        }
    }

    pub fn centroid(&self) -> Vec3 {
        match &self.geom {
            ShapeGeom::Sphere { c, .. } => *c,
            ShapeGeom::Capsule { a, b, .. } => (*a + *b) * 0.5,
            ShapeGeom::Hull { pts, .. } => pts.iter().copied().sum::<Vec3>() / pts.len().max(1) as f32,
        }
    }

    /// Rough volume (mass split between shapes when the tag gives none).
    pub fn volume(&self) -> f32 {
        match &self.geom {
            ShapeGeom::Sphere { r, .. } => 4.0 / 3.0 * std::f32::consts::PI * r * r * r,
            ShapeGeom::Capsule { a, b, r } => std::f32::consts::PI * r * r * ((*b - *a).length() + 4.0 / 3.0 * r),
            ShapeGeom::Hull { .. } => {
                let (mn, mx) = self.aabb();
                let e = (mx - mn).max(Vec3::splat(1e-3));
                e.x * e.y * e.z
            }
        }
    }

    /// Inertia tensor per unit mass about the shape's own centroid, body axes (hulls use their
    /// AABB box, capsules a box around the axis: good enough for how a piece tips over).
    pub fn unit_inertia(&self) -> Mat3 {
        match &self.geom {
            ShapeGeom::Sphere { r, .. } => Mat3::from_diagonal(Vec3::splat(0.4 * r * r)),
            _ => {
                let (mn, mx) = self.aabb();
                let e = (mx - mn).max(Vec3::splat(1e-3));
                let (a, b, c) = (e.x * e.x, e.y * e.y, e.z * e.z);
                Mat3::from_diagonal(Vec3::new(b + c, a + c, a + b) / 12.0)
            }
        }
    }
}

fn aabb_of_points(p: &[Vec3]) -> (Vec3, Vec3) {
    let mut mn = Vec3::splat(f32::MAX);
    let mut mx = Vec3::splat(f32::MIN);
    for v in p { mn = mn.min(*v); mx = mx.max(*v); }
    (mn, mx)
}

/// Outward face planes of a convex point set (`n . x <= d` inside). Brute force over vertex
/// triples like the overlay's hull triangulation; a flat / degenerate set (a triangle shape) and a
/// set above `HULL_PLANE_LIMIT` fall back to the AABB's six planes (thickness >= 1e-3).
pub fn hull_planes(pts: &[Vec3]) -> Vec<(Vec3, f32)> {
    let n = pts.len();
    let mut out: Vec<(Vec3, f32)> = Vec::new();
    if n >= 4 && n <= HULL_PLANE_LIMIT {
        const EPS: f32 = 1e-4;
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    let nrm = (pts[b] - pts[a]).cross(pts[c] - pts[a]);
                    let nl = nrm.length();
                    if nl < EPS { continue; }
                    let mut nrm = nrm / nl;
                    let mut d = nrm.dot(pts[a]);
                    let (mut pos, mut neg) = (0usize, 0usize);
                    for (i, v) in pts.iter().enumerate() {
                        if i == a || i == b || i == c { continue; }
                        let s = nrm.dot(*v) - d;
                        if s > EPS { pos += 1; }
                        if s < -EPS { neg += 1; }
                        if pos > 0 && neg > 0 { break; }
                    }
                    if pos > 0 && neg > 0 { continue; }
                    if pos > 0 { nrm = -nrm; d = -d; }
                    if !out.iter().any(|(on, od)| on.dot(nrm) > 0.9999 && (od - d).abs() < 1e-4) {
                        out.push((nrm, d));
                    }
                }
            }
        }
    }
    if out.len() < 4 {
        out.clear();
        let (mn, mx) = aabb_of_points(pts);
        let c = (mn + mx) * 0.5;
        let h = ((mx - mn) * 0.5).max(Vec3::splat(5e-4));
        for (n, hh) in [(Vec3::X, h.x), (Vec3::Y, h.y), (Vec3::Z, h.z)] {
            out.push((n, n.dot(c) + hh));
            out.push((-n, -n.dot(c) + hh));
        }
    }
    out
}

/// One rigid body as the tag stores it (model space), before the bodies are welded.
#[derive(Clone, Debug)]
pub struct RigidBodyDef {
    pub shapes: Vec<ConvexShape>,
    /// Tag mass (0 = compute from the shapes).
    pub mass: f32,
    /// Tag centre of mass (model space), when stored.
    pub com: Option<Vec3>,
    /// Tag inertia tensor rows, when stored.
    pub inertia: Option<Mat3>,
    pub inertia_scale: Vec3,
    pub lin_damping: f32,
    pub ang_damping: f32,
    /// 0 sphere, 1 stabilized sphere, 2 box, 3 stabilized box, 4 keyframed, 5 fixed.
    pub motion_type: u8,
    pub flags: u16,
    pub node: i16,
}

/// The welded physics body the settle simulates (body = model space).
#[derive(Clone, Debug)]
pub struct PhysBody {
    pub shapes: Vec<ConvexShape>,
    pub mass: f32,
    /// Centre of mass, body space.
    pub com: Vec3,
    /// Inertia tensor about `com`, body axes.
    pub inertia: Mat3,
    pub lin_damping: f32,
    pub ang_damping: f32,
    /// Editor scale already applied to the geometry (1.0 = the tag as stored).
    pub scale: f32,
    /// Where the body came from, for the status line ("phmo: 1 rigid body, 1 polyhedron").
    pub desc: String,
}

impl PhysBody {
    /// Weld every rigid body into one. Per body: tag mass (else the shapes' masses, else a
    /// volume-proportional unit density), tag centre of mass when it is finite and inside the
    /// body's box (else the volume-weighted shape centroid), tag inertia when it is a finite
    /// positive-diagonal tensor (else the shape geometry's), all combined with the parallel-axis
    /// theorem about the common centre of mass. `None` when no body has a shape.
    pub fn from_rigid_bodies(defs: &[RigidBodyDef], desc: String) -> Option<PhysBody> {
        struct Part { shapes: Vec<ConvexShape>, m: f32, com: Vec3, i: Mat3 }
        let mut parts: Vec<Part> = Vec::new();
        let (mut lin_d, mut ang_d, mut nd) = (0.0f32, 0.0f32, 0usize);
        for d in defs {
            if d.shapes.is_empty() { continue; }
            let (mn, mx) = d.shapes.iter().map(|s| s.aabb()).fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), (c, e)| (a.min(c), b.max(e)));
            let vols: Vec<f32> = d.shapes.iter().map(|s| s.volume().max(1e-6)).collect();
            let vsum: f32 = vols.iter().sum();
            let shape_mass_sum: f32 = d.shapes.iter().map(|s| s.mat.mass).sum();
            let m = if d.mass.is_finite() && d.mass > 0.0 { d.mass } else if shape_mass_sum > 0.0 { shape_mass_sum } else { vsum };
            // per-shape masses (for the centroid / inertia split)
            let ms: Vec<f32> = if shape_mass_sum > 0.0 {
                d.shapes.iter().map(|s| if s.mat.mass > 0.0 { s.mat.mass } else { 0.0 } ).map(|x| x / shape_mass_sum * m).collect()
            } else {
                vols.iter().map(|v| v / vsum * m).collect()
            };
            let geo_com = d.shapes.iter().zip(&ms).map(|(s, mi)| s.centroid() * *mi).sum::<Vec3>() / m;
            let com = match d.com {
                Some(c) if c.is_finite() && (0..3).all(|k| c[k] >= mn[k] - 1.0 && c[k] <= mx[k] + 1.0) => c,
                _ => geo_com,
            };
            let mut geo_i = Mat3::ZERO;
            for (s, mi) in d.shapes.iter().zip(&ms) {
                geo_i += s.unit_inertia() * *mi + parallel_axis(*mi, s.centroid() - com);
            }
            let scale = Vec3::new(
                if d.inertia_scale.x > 0.0 { d.inertia_scale.x } else { 1.0 },
                if d.inertia_scale.y > 0.0 { d.inertia_scale.y } else { 1.0 },
                if d.inertia_scale.z > 0.0 { d.inertia_scale.z } else { 1.0 },
            );
            // The tag tensor's normalisation is checked against the shape geometry: the Halo 4
            // Warthog stores the FULL tensor (diag (324.5, 1118.9, 1278.9) for a 3322-mass
            // 1.83 x 0.83 x 0.70 wu hull; mass x the box formula gives (326, 1063, 1130)), a tag
            // that stores it per unit mass is smaller by the mass. Either is accepted when it lands
            // within 10x of the geometric estimate; anything else (zeros, a content-tools
            // tensor in other units) falls back to the geometry. The tag's inertia tensor scale
            // (1.395 on that Warthog) is applied on top, as the engine's rigid-body setup does.
            let geo_trace = geo_i.x_axis.x + geo_i.y_axis.y + geo_i.z_axis.z;
            let i = match d.inertia {
                Some(t) if tensor_ok(&t) && geo_trace > 0.0 => {
                    let tag_trace = t.x_axis.x + t.y_axis.y + t.z_axis.z;
                    let ratio = tag_trace / geo_trace;
                    let t = if ratio > 0.1 && ratio < 10.0 {
                        t
                    } else if ratio * m > 0.1 && ratio * m < 10.0 {
                        t * m
                    } else {
                        geo_i
                    };
                    Mat3::from_cols(t.x_axis * scale.x, t.y_axis * scale.y, t.z_axis * scale.z)
                }
                _ => Mat3::from_cols(geo_i.x_axis * scale.x, geo_i.y_axis * scale.y, geo_i.z_axis * scale.z),
            };
            lin_d += d.lin_damping.max(0.0);
            ang_d += d.ang_damping.max(0.0);
            nd += 1;
            parts.push(Part { shapes: d.shapes.clone(), m, com, i });
        }
        if parts.is_empty() { return None; }
        let mass: f32 = parts.iter().map(|p| p.m).sum();
        let com = parts.iter().map(|p| p.com * p.m).sum::<Vec3>() / mass;
        let mut inertia = Mat3::ZERO;
        let mut shapes = Vec::new();
        for p in parts {
            inertia += p.i + parallel_axis(p.m, p.com - com);
            shapes.extend(p.shapes);
        }
        // Symmetrise + floor the diagonal so the inverse always exists.
        let mut inertia = (inertia + inertia.transpose()) * 0.5;
        let floor = mass * 1e-4;
        for k in 0..3 { if inertia.col(k)[k] < floor { let mut c = inertia.col(k); c[k] = floor; *inertia.col_mut(k) = c; } }
        Some(PhysBody {
            shapes,
            mass,
            com,
            inertia,
            lin_damping: if nd > 0 { lin_d / nd as f32 } else { 0.0 },
            ang_damping: if nd > 0 { ang_d / nd as f32 } else { 0.0 },
            scale: 1.0,
            desc,
        })
    }

    /// A single-shape body (the collision-hull fallback for objects without a `phmo`).
    pub fn from_shape(shape: ConvexShape, desc: String) -> PhysBody {
        let d = RigidBodyDef {
            shapes: vec![shape], mass: 0.0, com: None, inertia: None, inertia_scale: Vec3::ONE,
            lin_damping: 0.0, ang_damping: 0.05, motion_type: 2, flags: 0, node: -1,
        };
        PhysBody::from_rigid_bodies(&[d], desc).expect("one shape")
    }

    /// A vehicle's ground-contact points (`vehi` friction points = the tyres, anti-gravity points
    /// = the hover pads) as spheres of their radius / hover height at their marker positions
    /// (model space). In-game a wheeled vehicle rests on those, not on its chassis hull (the
    /// Warthog's hull bottom sits 0.2 wu above its origin; the origin is at ground level on its
    /// wheels). Friction / restitution come from the chassis; mass, centre of mass and inertia
    /// are left as the phmo has them. `wheels` / `hover` count what was added for the status.
    pub fn add_support_spheres(&mut self, points: &[(Vec3, f32, bool)]) {
        let mat = ShapeMaterial { mass: 0.0, restitution: 0.0, ..self.shapes.first().map(|s| s.mat).unwrap_or_default() };
        let (mut wheels, mut hover) = (0usize, 0usize);
        let mut seen: Vec<Vec3> = Vec::new();
        for (c, r, h) in points {
            // a friction point naming a marker group with several instances repeats positions
            if seen.iter().any(|q| (*q - *c).length_squared() < 1e-6) { continue; }
            if let Some(mut s) = ConvexShape::sphere(*c, *r, mat) {
                s.support = true;
                self.shapes.push(s);
                seen.push(*c);
                if *h { hover += 1 } else { wheels += 1 }
            }
        }
        if wheels > 0 { self.desc.push_str(&format!(", {wheels} wheel{}", if wheels == 1 { "" } else { "s" })); }
        if hover > 0 { self.desc.push_str(&format!(", {hover} hover point{}", if hover == 1 { "" } else { "s" })); }
    }

    /// The body at a uniform editor scale about the model origin: geometry and centre of mass
    /// scale linearly, the inertia tensor by s^2 (mass unchanged).
    pub fn scaled(&self, s: f32) -> PhysBody {
        if !(s.is_finite() && s > 0.0) || (s - 1.0).abs() < 1e-6 { return self.clone(); }
        PhysBody {
            shapes: self.shapes.iter().map(|sh| sh.scaled(s)).collect(),
            mass: self.mass,
            com: self.com * s,
            inertia: self.inertia * (s * s),
            lin_damping: self.lin_damping,
            ang_damping: self.ang_damping,
            scale: self.scale * s,
            desc: self.desc.clone(),
        }
    }

    /// Body-space AABB of every shape (convex radius included).
    pub fn aabb(&self) -> (Vec3, Vec3) {
        self.shapes.iter().map(|s| s.aabb()).fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), (c, d)| (a.min(c), b.max(d)))
    }

    /// Largest distance from the centre of mass to any shape point (+ radius).
    pub fn bounding_radius(&self) -> f32 {
        let mut r = 0.0f32;
        for s in &self.shapes {
            match &s.geom {
                ShapeGeom::Sphere { c, r: sr } => r = r.max((*c - self.com).length() + sr),
                ShapeGeom::Capsule { a, b, r: sr } => r = r.max((*a - self.com).length().max((*b - self.com).length()) + sr),
                ShapeGeom::Hull { pts, r: sr, .. } => for p in pts { r = r.max((*p - self.com).length() + sr) },
            }
        }
        r
    }
}

/// The body of an object that has a `coll` but no `phmo`: one convex hull of the collision
/// model's vertices (its AABB box above `HULL_PLANE_LIMIT` vertices), no convex radius. This is
/// the ONE place the settle uses something other than a phmo shape, and the status line says so.
pub fn collision_fallback_body(verts: &[[f32; 3]]) -> Option<PhysBody> {
    let pts: Vec<Vec3> = verts.iter().map(|v| Vec3::from(*v)).filter(|v| v.is_finite()).collect();
    if pts.len() < 3 { return None; }
    let shape = if pts.len() > HULL_PLANE_LIMIT {
        let (mn, mx) = aabb_of_points(&pts);
        ConvexShape::aabb_box((mn + mx) * 0.5, (mx - mn) * 0.5, 0.0, ShapeMaterial::default())?
    } else {
        ConvexShape::hull(&pts, 0.0, ShapeMaterial::default())?
    };
    Some(PhysBody::from_shape(shape, format!("no phmo: coll hull of {} vertices", pts.len())))
}

fn parallel_axis(m: f32, d: Vec3) -> Mat3 {
    let d2 = d.length_squared();
    Mat3::from_diagonal(Vec3::splat(d2 * m)) - Mat3::from_cols(d * d.x, d * d.y, d * d.z) * m
}

fn tensor_ok(t: &Mat3) -> bool {
    let cols = [t.x_axis, t.y_axis, t.z_axis];
    if !cols.iter().all(|c| c.is_finite()) { return false; }
    let diag = Vec3::new(t.x_axis.x, t.y_axis.y, t.z_axis.z);
    if !(diag.min_element() > 1e-6 && diag.max_element() < 1e6) { return false; }
    // diagonally dominant enough to be a real inertia tensor (|off| < diag on every row)
    for k in 0..3 {
        let c = cols[k];
        for j in 0..3 { if j != k && c[j].abs() > diag[k] * 1.5 { return false; } }
    }
    true
}

// ------------------------------------------------------------------------------------------
// The static world
// ------------------------------------------------------------------------------------------

/// Static collision geometry the body settles against: world-space triangles, queried by box.
pub trait StaticWorld {
    /// Append every triangle whose AABB overlaps [mn, mx].
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>);
}

impl StaticWorld for [[Vec3; 3]] {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        for t in self {
            if tri_overlaps(t, mn, mx) { out.push(*t); }
        }
    }
}

impl StaticWorld for Vec<[Vec3; 3]> {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) { self.as_slice().gather(mn, mx, out) }
}

pub fn tri_overlaps(t: &[Vec3; 3], mn: Vec3, mx: Vec3) -> bool {
    let tmn = t[0].min(t[1]).min(t[2]);
    let tmx = t[0].max(t[1]).max(t[2]);
    (0..3).all(|k| tmn[k] <= mx[k] && tmx[k] >= mn[k])
}

// ------------------------------------------------------------------------------------------
// Settle
// ------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SettleParams {
    pub dt: f32,
    pub gravity: f32,
    /// Hard cap on simulated ticks (3000 = 100 s of game time).
    pub max_steps: usize,
    pub solver_iters: usize,
    /// Rest = below these speeds for `sleep_steps` consecutive ticks while touching something.
    pub sleep_lin: f32,
    pub sleep_ang: f32,
    pub sleep_steps: usize,
    /// Falling below this z with nothing hit = "no contact" (the scene floor minus a margin).
    pub floor_z: f32,
    /// Penetration allowed before the split impulse pushes out (wu).
    pub slop: f32,
    /// Approach speed below which a contact does not bounce (wu/s).
    pub restitution_threshold: f32,
}

impl Default for SettleParams {
    fn default() -> Self {
        SettleParams {
            dt: 1.0 / TICK_HZ,
            gravity: GRAVITY_WU_S2,
            max_steps: 3000,
            solver_iters: 16,
            sleep_lin: 0.02,
            sleep_ang: 0.05,
            sleep_steps: 15,
            floor_z: -1.0e4,
            slop: 0.004,
            restitution_threshold: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettleStatus {
    /// Came to rest on something.
    Settled,
    /// Fell past `floor_z` without ever touching anything (the pose is unchanged).
    NoContact,
    /// Still moving at `max_steps` (the pose at the last tick is returned).
    Timeout,
}

#[derive(Clone, Debug)]
pub struct SettleResult {
    /// Final model-origin position and rotation.
    pub pos: Vec3,
    pub rot: Quat,
    pub status: SettleStatus,
    pub steps: usize,
    pub sim_time: f32,
    /// Contacts at the final tick.
    pub contacts: usize,
    /// How far the centre of mass fell (+ = down).
    pub drop: f32,
    /// Angle between the initial and final up axes, degrees.
    pub tilt_deg: f32,
    /// Triangles the body was tested against at the final tick.
    pub tris_tested: usize,
}

impl SettleResult {
    /// One status-bar line.
    pub fn summary(&self) -> String {
        match self.status {
            SettleStatus::Settled => format!(
                "Settled in {} steps ({:.2} s): dropped {:.3} wu, tilted {:.1} deg, {} contact(s) on {} triangle(s)",
                self.steps, self.sim_time, self.drop, self.tilt_deg, self.contacts, self.tris_tested
            ),
            SettleStatus::NoContact => format!(
                "No contact: fell {:.1} wu in {} steps without touching anything (pose unchanged)",
                self.drop, self.steps
            ),
            SettleStatus::Timeout => format!(
                "Still moving after {} steps ({:.1} s): dropped {:.3} wu, tilted {:.1} deg, {} contact(s) - pose at the last step applied",
                self.steps, self.sim_time, self.drop, self.tilt_deg, self.contacts
            ),
        }
    }
}

pub(crate) struct Contact {
    /// World point on the static surface.
    pub(crate) p: Vec3,
    /// World normal from the static surface INTO the body (the push direction).
    pub(crate) n: Vec3,
    pub(crate) depth: f32,
    pub(crate) mu: f32,
    pub(crate) e: f32,
    /// A support-sphere (tyre / hover pad) contact: force-limited to `SUPPORT_G` per point until
    /// it has compressed `SUPPORT_TRAVEL` of its radius, then hard (the bump stop).
    pub(crate) soft: Option<f32>,
}

/// Suspension stand-in for vehicle ground-contact points: each tyre / hover pad can push with at
/// most this many times the whole vehicle's weight while it is within its travel. A Halo vehicle
/// dropped in Forge lands on compliant wheels, not on rigid spheres: without this a Warthog
/// dropped 2.5 wu onto a 9 deg slope pitches at 4 rad/s off its front wheels and lands on its
/// side, which the game never does. The rest pose is unchanged by it (at rest the wheels carry
/// exactly the weight).
pub(crate) const SUPPORT_G: f32 = 1.0;
/// Fraction of a support sphere's radius it may compress before its contact turns hard.
pub(crate) const SUPPORT_TRAVEL: f32 = 0.5;

/// Closest point on triangle abc to p, and whether it is the interior (face) projection.
/// Ericson, Real-Time Collision Detection 5.1.5.
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, bool) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 { return (a, false); }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 { return (b, false); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, false);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 { return (c, false); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, false);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, false);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, true)
}

/// A world-space sphere (centre, radius) against one triangle.
fn sphere_tri_contact(c: Vec3, r: f32, tri: &[Vec3; 3], mat: &ShapeMaterial, soft: Option<f32>, out: &mut Vec<Contact>) {
    let (q, _) = closest_point_on_triangle(c, tri[0], tri[1], tri[2]);
    let delta = c - q;
    let d = delta.length();
    if d >= r { return; }
    let n = if d > 1e-6 {
        delta / d
    } else {
        let tn = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
        if tn == Vec3::ZERO { return; }
        tn
    };
    out.push(Contact { p: q, n, depth: r - d, mu: mat.friction, e: mat.restitution, soft });
}

/// Contacts of one shape (posed by `rot`/`pos` = body->world, body space points) against `tris`.
pub(crate) fn shape_contacts(shape: &ConvexShape, rot: Mat3, pos: Vec3, inv_rot: Mat3, tris: &[[Vec3; 3]], out: &mut Vec<Contact>) {
    let mat = &shape.mat;
    match &shape.geom {
        ShapeGeom::Sphere { c, r } => {
            let cw = pos + rot * *c;
            let soft = shape.support.then_some(*r * SUPPORT_TRAVEL);
            for t in tris { sphere_tri_contact(cw, *r, t, mat, soft, out); }
        }
        ShapeGeom::Capsule { a, b, r } => {
            // five spheres along the axis (the segment-triangle pair is not solved exactly)
            let aw = pos + rot * *a;
            let bw = pos + rot * *b;
            for k in 0..5 {
                let cw = aw.lerp(bw, k as f32 / 4.0);
                for t in tris { sphere_tri_contact(cw, *r, t, mat, None, out); }
            }
        }
        ShapeGeom::Hull { pts, planes, r } => {
            let cen_w = pos + rot * shape.centroid();
            let (lmn, lmx) = shape.aabb();
            let ext = (lmx - lmn).max_element().max(1e-3);
            // (a) hull vertices vs triangles
            let world: Vec<Vec3> = pts.iter().map(|p| pos + rot * *p).collect();
            for t in tris {
                let tn = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
                if tn == Vec3::ZERO { continue; }
                for pw in &world {
                    let (q, face) = closest_point_on_triangle(*pw, t[0], t[1], t[2]);
                    let delta = *pw - q;
                    // the triangle normal on the body's side
                    let n_face = if tn.dot(cen_w - q) >= 0.0 { tn } else { -tn };
                    if face {
                        let s = n_face.dot(delta); // signed distance above the face
                        let depth = *r - s;
                        // a vertex deeper than the hull itself is a triangle we came from
                        // elsewhere (a wall behind the piece): not a contact
                        if depth > 0.0 && depth < ext { out.push(Contact { p: q, n: n_face, depth, mu: mat.friction, e: mat.restitution, soft: None }); }
                    } else {
                        let d = delta.length();
                        if d < 1e-6 { continue; }
                        let n = delta / d;
                        if n.dot(n_face) < 0.0 {
                            // The vertex is on the far side of the plane and its projection lies just
                            // outside the triangle (it crossed the plane at a triangle edge: a mesh
                            // border or a ledge). Push it back through the face, with the contact UNDER
                            // the vertex (its projection onto the plane), never at the edge's closest
                            // point - a lever arm at the edge would spin the body about that edge.
                            // Beyond `EDGE_MARGIN` in the plane the vertex belongs to a neighbouring
                            // triangle, which reports it itself.
                            let s = n_face.dot(delta);
                            let inplane = (delta - n_face * s).length();
                            let depth = *r - s;
                            if depth > 0.0 && depth < ext && s > -ext && inplane < *r + EDGE_MARGIN {
                                out.push(Contact { p: *pw - n_face * s, n: n_face, depth, mu: mat.friction, e: mat.restitution, soft: None });
                            }
                        } else if d < *r {
                            out.push(Contact { p: q, n, depth: *r - d, mu: mat.friction, e: mat.restitution, soft: None });
                        }
                    }
                }
            }
            // (b) triangle vertices inside the (radius-padded) hull
            for t in tris {
                for v in t {
                    let vb = inv_rot * (*v - pos);
                    let mut best = f32::MIN;
                    let mut best_n = Vec3::Z;
                    for (n, d) in planes {
                        let s = n.dot(vb) - d;
                        if s > best { best = s; best_n = *n; }
                    }
                    let depth = *r - best;
                    if depth > 0.0 && depth < ext {
                        let n = rot * best_n;
                        out.push(Contact { p: *v, n, depth, mu: mat.friction, e: mat.restitution, soft: None });
                    }
                }
            }
        }
    }
}

pub(crate) fn integrate_quat(q: Quat, w: Vec3, h: f32) -> Quat {
    let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * q;
    let q2 = Quat::from_xyzw(q.x + 0.5 * h * dq.x, q.y + 0.5 * h * dq.y, q.z + 0.5 * h * dq.z, q.w + 0.5 * h * dq.w);
    let n = q2.normalize();
    if n.is_finite() { n } else { q }
}

/// Drop `body` from (`pos`, `rot`) onto `world` and step it until rest. See the module docs for
/// what is modelled. `pos` is the model origin; `rot` the body->world rotation.
///
/// #ragdoll This is now a ONE-LINK, NO-JOINT call into [`crate::ragdoll::step`] -- the same
/// stepper, the same contact rows, the same integrator a Theater ragdoll uses. The physics the
/// editor's "Simulate physics for selection" button runs and the physics a dead biped runs are
/// one implementation, which is what makes the ragdoll work pay for itself twice. With one link
/// and no joints the stepper reduces to exactly the arithmetic this function used to contain in
/// line (gravity and damping per substep, the body's swept box gathered from the static world,
/// `shape_contacts` per shape, `solver_iters` passes of normal + two-direction Coulomb friction +
/// split impulse, then the integration), in the same order -- which the settle gates below, and
/// the vehicle ones in `main.rs`, hold it to.
pub fn settle(body: &PhysBody, pos: Vec3, rot: Quat, world: &dyn StaticWorld, p: &SettleParams) -> SettleResult {
    let rot = rot.normalize();
    let start_pos = pos;
    let start_up = rot * Vec3::Z;

    let rd = crate::ragdoll::one_link(body.clone());
    let mut st = crate::ragdoll::RagdollState {
        links: vec![crate::ragdoll::LinkState { com: pos + rot * body.com, q: rot, v: Vec3::ZERO, w: Vec3::ZERO }],
        ..Default::default()
    };
    let start_com_z = st.links[0].com.z;
    let sim = crate::ragdoll::SimParams {
        dt: p.dt,
        gravity: p.gravity,
        solver_iters: p.solver_iters,
        slop: p.slop,
        restitution_threshold: p.restitution_threshold,
        max_substeps: 8,
        // one link has nothing to collide with and nothing to be joined to
        self_collision: false,
        sleep_lin: p.sleep_lin,
        sleep_ang: p.sleep_ang,
        sleep_steps: p.sleep_steps as u32,
        ..Default::default()
    };

    let mut steps = 0usize;
    let mut status = SettleStatus::Timeout;
    let mut last = crate::ragdoll::StepStats::default();
    // HMS_SETTLE_TRACE=1: one line per tick on stderr (diagnostics)
    let trace = std::env::var("HMS_SETTLE_TRACE").is_ok();

    while steps < p.max_steps {
        steps += 1;
        last = crate::ragdoll::step(&rd, &mut st, world, &sim);
        let s = st.links[0];
        if trace {
            let up = s.q * Vec3::Z;
            eprintln!("settle step {steps}: com ({:.4},{:.4},{:.4}) v ({:.4},{:.4},{:.4}) w ({:.4},{:.4},{:.4}) up ({:.3},{:.3},{:.3}) contacts {} tris {} nsub {}",
                s.com.x, s.com.y, s.com.z, s.v.x, s.v.y, s.v.z, s.w.x, s.w.y, s.w.z, up.x, up.y, up.z,
                last.contacts, last.tris_tested, last.substeps);
        }
        if s.com.z < p.floor_z && !st.touched {
            status = SettleStatus::NoContact;
            break;
        }
        if st.asleep {
            status = SettleStatus::Settled;
            break;
        }
    }
    let s = st.links[0];
    let q = s.q;
    let r = Mat3::from_quat(q);
    let final_pos = s.com - r * body.com;
    let (pos_out, rot_out) = if status == SettleStatus::NoContact { (start_pos, rot) } else { (final_pos, q) };
    let up = q * Vec3::Z;
    SettleResult {
        pos: pos_out,
        rot: rot_out,
        status,
        steps,
        sim_time: steps as f32 * p.dt,
        contacts: last.contacts,
        drop: start_com_z - s.com.z,
        tilt_deg: start_up.dot(up).clamp(-1.0, 1.0).acos().to_degrees(),
        tris_tested: last.tris_tested,
    }
}

/// The editor's (forward, up) pair of a rotation (Halo: local +X forward, +Z up).
pub fn rot_to_fwd_up(q: Quat) -> (Vec3, Vec3) { (q * Vec3::X, q * Vec3::Z) }

/// A rotation from a possibly non-unit / non-orthogonal (forward, up) pair.
pub fn fwd_up_to_rot(fwd: Vec3, up: Vec3) -> Quat {
    let mut f = fwd.normalize_or_zero();
    if f == Vec3::ZERO { f = Vec3::X; }
    let mut u = up - f * up.dot(f);
    if u.length_squared() < 1e-8 { u = if f.z.abs() < 0.9 { Vec3::Z - f * f.z } else { Vec3::X - f * f.x }; }
    let u = u.normalize();
    let l = u.cross(f);
    Quat::from_mat3(&Mat3::from_cols(f, l, u)).normalize()
}

// ------------------------------------------------------------------------------------------
// Scene driver (both games through the ObjectScene trait)
// ------------------------------------------------------------------------------------------

/// What a scene settle produced: the new pose plus the numbers for the status bar.
#[derive(Clone, Debug)]
pub struct SettleOutcome {
    pub pos: Vec3,
    pub fwd: Vec3,
    pub up: Vec3,
    pub result: SettleResult,
    pub body_desc: String,
    pub scale: f32,
    pub static_objects: usize,
}

impl SettleOutcome {
    pub fn summary(&self) -> String {
        let scale = if (self.scale - 1.0).abs() > 1e-3 { format!(", scale x{:.2}", self.scale) } else { String::new() };
        format!("{} [{}{}, {} static object(s)]", self.result.summary(), self.body_desc, scale, self.static_objects)
    }
    pub fn moved(&self) -> bool { self.result.status != SettleStatus::NoContact }
}

struct SceneWorld<'a> {
    scene: &'a dyn crate::objscene::ObjectScene,
    extra: Vec<[Vec3; 3]>,
}

impl StaticWorld for SceneWorld<'_> {
    fn gather(&self, mn: Vec3, mx: Vec3, out: &mut Vec<[Vec3; 3]>) {
        self.scene.static_tris_in_aabb(mn, mx, out);
        self.extra.as_slice().gather(mn, mx, out);
    }
}

/// How far below the object other objects are still considered as landing candidates.
const CANDIDATE_DROP: f32 = 200.0;

/// Settle object `datum` of `objects` (every drawn object: scenario + variant + placed; the
/// others are static colliders through their physics hulls) against the scene's level geometry
/// and `extra_tris` (imported meshes). Pure with respect to the scene: the caller applies the
/// returned pose.
pub fn settle_in_scene(
    scene: &dyn crate::objscene::ObjectScene,
    objects: &[hms_ipc::ObjectInfo],
    datum: u32,
    extra_tris: &[[Vec3; 3]],
    params: &SettleParams,
) -> Result<SettleOutcome, String> {
    let o = objects.iter().find(|o| o.datum == datum).ok_or_else(|| format!("no object 0x{datum:08X}"))?;
    let body = scene.physics_body(o).ok_or_else(|| "the object has no physics model (no phmo / coll hull)".to_string())?;
    if body.shapes.is_empty() { return Err("the physics model has no collidable shape".into()); }
    let pos = Vec3::from(o.pos);
    let rot = fwd_up_to_rot(Vec3::from(o.fwd), Vec3::from(o.up));
    // the sweep box: the body's world box, extended down to the candidate floor
    let (lmn, lmx) = body.aabb();
    let r = Mat3::from_quat(rot);
    let mut wmn = Vec3::splat(f32::MAX);
    let mut wmx = Vec3::splat(f32::MIN);
    for k in 0..8u32 {
        let c = Vec3::new(if k & 4 != 0 { lmx.x } else { lmn.x }, if k & 2 != 0 { lmx.y } else { lmn.y }, if k & 1 != 0 { lmx.z } else { lmn.z });
        let cw = pos + r * c;
        wmn = wmn.min(cw);
        wmx = wmx.max(cw);
    }
    let brad = body.bounding_radius();
    let smn = Vec3::new(wmn.x - brad, wmn.y - brad, wmn.z - CANDIDATE_DROP);
    let smx = Vec3::new(wmx.x + brad, wmx.y + brad, wmx.z + brad);
    let mut extra: Vec<[Vec3; 3]> = Vec::new();
    let mut n_static = 0usize;
    for other in objects {
        if other.datum == datum { continue; }
        let op = Vec3::from(other.pos);
        // cheap reject on the position (any real object is far smaller than this)
        if op.x < smn.x - 60.0 || op.x > smx.x + 60.0 || op.y < smn.y - 60.0 || op.y > smx.y + 60.0 || op.z < smn.z - 60.0 || op.z > smx.z + 60.0 { continue; }
        let tris = scene.physics_world_tris(other);
        let before = extra.len();
        for t in tris {
            if tri_overlaps(&t, smn, smx) { extra.push(t); }
        }
        if extra.len() > before { n_static += 1; }
    }
    for t in extra_tris {
        if tri_overlaps(t, smn, smx) { extra.push(*t); }
    }
    let mut p = *params;
    if let Some((bmn, _)) = scene.scene_bounds() {
        p.floor_z = p.floor_z.max(bmn[2] - 20.0);
    }
    let world = SceneWorld { scene, extra };
    let result = settle(&body, pos, rot, &world, &p);
    let (fwd, up) = rot_to_fwd_up(result.rot);
    Ok(SettleOutcome {
        pos: result.pos,
        fwd,
        up,
        scale: body.scale,
        body_desc: body.desc.clone(),
        static_objects: n_static,
        result,
    })
}

// ------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(z: f32, half: f32) -> Vec<[Vec3; 3]> {
        let a = Vec3::new(-half, -half, z);
        let b = Vec3::new(half, -half, z);
        let c = Vec3::new(half, half, z);
        let d = Vec3::new(-half, half, z);
        vec![[a, b, c], [a, c, d]]
    }

    fn unit_box(h: Vec3) -> PhysBody {
        let s = ConvexShape::aabb_box(Vec3::ZERO, h, 0.0, ShapeMaterial { friction: 0.6, restitution: 0.1, mass: 1.0 }).unwrap();
        PhysBody::from_shape(s, "test box".into())
    }

    #[test]
    fn hull_planes_of_a_box_are_its_six_faces() {
        let s = ConvexShape::aabb_box(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 1.0, 1.5), 0.0, ShapeMaterial::default()).unwrap();
        let ShapeGeom::Hull { planes, .. } = &s.geom else { panic!() };
        assert_eq!(planes.len(), 6, "{planes:?}");
        // +x face: n=(1,0,0), d = 1.5
        assert!(planes.iter().any(|(n, d)| (n.x - 1.0).abs() < 1e-5 && (d - 1.5).abs() < 1e-4));
        assert!(planes.iter().any(|(n, d)| (n.z + 1.0).abs() < 1e-5 && (d - (-1.5)).abs() < 1e-4));
    }

    #[test]
    fn box_drops_onto_a_plane_and_rests_at_its_half_height() {
        let body = unit_box(Vec3::splat(0.5));
        let world = plane(0.0, 10.0);
        let r = settle(&body, Vec3::new(0.0, 0.0, 3.0), Quat::IDENTITY, &world, &SettleParams::default());
        eprintln!("{}", r.summary());
        assert_eq!(r.status, SettleStatus::Settled);
        assert!((r.pos.z - 0.5).abs() < 0.01, "rest z {}", r.pos.z);
        assert!(Vec3::new(r.pos.x, r.pos.y, 0.0).length() < 0.01, "no lateral drift {:?}", r.pos);
        assert!(r.tilt_deg < 0.5, "stays flat: {}", r.tilt_deg);
    }

    #[test]
    fn scaled_box_rests_higher_by_its_scale() {
        let body = unit_box(Vec3::splat(0.5));
        let big = body.scaled(2.0);
        let world = plane(0.0, 10.0);
        let r1 = settle(&body, Vec3::new(0.0, 0.0, 3.0), Quat::IDENTITY, &world, &SettleParams::default());
        let r2 = settle(&big, Vec3::new(0.0, 0.0, 3.0), Quat::IDENTITY, &world, &SettleParams::default());
        assert_eq!(r1.status, SettleStatus::Settled);
        assert_eq!(r2.status, SettleStatus::Settled);
        assert!((r1.pos.z - 0.5).abs() < 0.01, "x1 rest z {}", r1.pos.z);
        assert!((r2.pos.z - 1.0).abs() < 0.01, "x2 rest z {}", r2.pos.z);
        assert!((big.inertia.x_axis.x / body.inertia.x_axis.x - 4.0).abs() < 1e-3, "inertia scales by s^2");
    }

    #[test]
    fn convex_radius_pads_the_hull() {
        // a 0.484 half-extent core + 0.016 radius = the 0.5 wu Forge block
        let s = ConvexShape::aabb_box(Vec3::ZERO, Vec3::splat(0.484), 0.016, ShapeMaterial::default()).unwrap();
        let body = PhysBody::from_shape(s, "block".into());
        let world = plane(0.0, 10.0);
        let r = settle(&body, Vec3::new(0.0, 0.0, 2.0), Quat::IDENTITY, &world, &SettleParams::default());
        assert_eq!(r.status, SettleStatus::Settled);
        assert!((r.pos.z - 0.5).abs() < 0.01, "rest z {}", r.pos.z);
    }

    #[test]
    fn sphere_and_capsule_rest_at_their_radius() {
        let sp = ConvexShape::sphere(Vec3::ZERO, 0.3, ShapeMaterial::default()).unwrap();
        let body = PhysBody::from_shape(sp, "sphere".into());
        let world = plane(0.0, 10.0);
        let r = settle(&body, Vec3::new(0.0, 0.0, 2.0), Quat::IDENTITY, &world, &SettleParams::default());
        assert_eq!(r.status, SettleStatus::Settled);
        assert!((r.pos.z - 0.3).abs() < 0.01, "sphere rest z {}", r.pos.z);
        let cap = ConvexShape::capsule(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0), 0.2, ShapeMaterial::default()).unwrap();
        let body = PhysBody::from_shape(cap, "capsule".into());
        let r = settle(&body, Vec3::new(0.0, 0.0, 2.0), Quat::IDENTITY, &world, &SettleParams::default());
        assert_eq!(r.status, SettleStatus::Settled);
        assert!((r.pos.z - 0.2).abs() < 0.01, "capsule rest z {}", r.pos.z);
        assert!(r.tilt_deg < 1.0, "capsule stays horizontal: {}", r.tilt_deg);
    }

    #[test]
    fn tilted_box_falls_flat_onto_a_face() {
        let body = unit_box(Vec3::new(0.6, 0.4, 0.2));
        let world = plane(0.0, 10.0);
        // 25 degrees about y: the box lands on a corner and must topple flat
        let q = Quat::from_rotation_y(25f32.to_radians());
        let r = settle(&body, Vec3::new(0.0, 0.0, 1.5), q, &world, &SettleParams::default());
        eprintln!("{}", r.summary());
        assert_eq!(r.status, SettleStatus::Settled);
        let up = r.rot * Vec3::Z;
        assert!(up.z.abs() > 0.999, "a face is down: up {up:?}");
        assert!((r.pos.z - 0.2).abs() < 0.01, "rests on its thin side: z {}", r.pos.z);
    }

    #[test]
    fn body_above_nothing_reports_no_contact_and_keeps_its_pose() {
        let body = unit_box(Vec3::splat(0.5));
        let world: Vec<[Vec3; 3]> = Vec::new();
        let p = SettleParams { floor_z: -50.0, ..Default::default() };
        let r = settle(&body, Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, &world, &p);
        assert_eq!(r.status, SettleStatus::NoContact);
        assert_eq!(r.pos, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn friction_holds_a_box_on_a_gentle_slope_and_ice_lets_it_slide() {
        // a 15 degree ramp: tan = 0.27 -> mu 0.6 holds, mu 0.0 slides off
        let ang = 15f32.to_radians();
        let q = Quat::from_rotation_y(-ang);
        let half = 20.0;
        let mut ramp = plane(0.0, half);
        for t in ramp.iter_mut() { for v in t.iter_mut() { *v = q * *v; } }
        let mk = |mu: f32| {
            let s = ConvexShape::aabb_box(Vec3::ZERO, Vec3::splat(0.5), 0.0, ShapeMaterial { friction: mu, restitution: 0.0, mass: 1.0 }).unwrap();
            PhysBody::from_shape(s, "ramp box".into())
        };
        let start = q * Vec3::new(0.0, 0.0, 0.5) + Vec3::new(0.0, 0.0, 0.5);
        let rough = settle(&mk(0.6), start, q, &ramp, &SettleParams::default());
        eprintln!("rough: {}", rough.summary());
        assert_eq!(rough.status, SettleStatus::Settled);
        assert!((rough.pos - q * Vec3::new(0.0, 0.0, 0.5)).length() < 0.05, "held in place: {:?}", rough.pos);
        let ice = settle(&mk(0.0), start, q, &ramp, &SettleParams { max_steps: 600, ..Default::default() });
        eprintln!("ice: {}", ice.summary());
        assert!(ice.status != SettleStatus::Settled || (ice.pos - start).length() > 1.0, "slid down: {:?}", ice.pos);
    }

    #[test]
    fn rigid_bodies_weld_into_one_with_the_combined_mass_and_com() {
        let a = RigidBodyDef {
            shapes: vec![ConvexShape::sphere(Vec3::new(-1.0, 0.0, 0.0), 0.5, ShapeMaterial::default()).unwrap()],
            mass: 1.0, com: None, inertia: None, inertia_scale: Vec3::ZERO, lin_damping: 0.0, ang_damping: 0.0, motion_type: 2, flags: 0, node: 0,
        };
        let b = RigidBodyDef {
            shapes: vec![ConvexShape::sphere(Vec3::new(1.0, 0.0, 0.0), 0.5, ShapeMaterial::default()).unwrap()],
            mass: 3.0, com: None, inertia: None, inertia_scale: Vec3::ZERO, lin_damping: 0.0, ang_damping: 0.0, motion_type: 2, flags: 0, node: 1,
        };
        let body = PhysBody::from_rigid_bodies(&[a, b], "two".into()).unwrap();
        assert_eq!(body.shapes.len(), 2);
        assert!((body.mass - 4.0).abs() < 1e-6);
        assert!((body.com.x - 0.5).abs() < 1e-5, "com {:?}", body.com);
        // parallel axis: Iyy = sum(0.4 m r^2 + m d^2) = 0.4*0.25*4 + 1*1.5^2 + 3*0.5^2 = 0.4+2.25+0.75
        assert!((body.inertia.y_axis.y - 3.4).abs() < 1e-4, "Iyy {}", body.inertia.y_axis.y);
    }

    #[test]
    fn fwd_up_round_trip() {
        let q = Quat::from_euler(glam::EulerRot::ZYX, 0.7, 0.2, -0.4);
        let (f, u) = rot_to_fwd_up(q);
        let q2 = fwd_up_to_rot(f, u);
        assert!(q.dot(q2).abs() > 0.99999, "{q:?} vs {q2:?}");
        // a sheared / non-unit pair is orthonormalised, forward kept
        let q3 = fwd_up_to_rot(Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 1.0));
        let (f3, u3) = rot_to_fwd_up(q3);
        assert!((f3 - Vec3::X).length() < 1e-5 && (u3 - Vec3::Z).length() < 1e-5);
    }
}
