//! The shapes a body can carry (what a `phmo` primitive becomes: box, sphere, capsule, polyhedron)
//! and the narrow-phase dispatch over pairs of them.
//!
//! Box / polyhedron pairs use the SAT + clipping manifold of [`super::poly`]; everything with a
//! sphere or capsule goes through GJK/EPA ([`super::gjk`]) like the engine's `hkpGsk*` agents.
//! A capsule is tested as three spheres along its axis (its two ends and its middle), which gives
//! the 2-3 point line contact a lying capsule needs.

use super::gjk::{self, PointsCore, SegmentCore, SphereCore};
use super::poly::{collide as collide_poly, Poly, RawContact};
use glam::{Quat, Vec3};

/// A body-local shape.
#[derive(Clone, Debug)]
pub enum LocalShape {
    /// Convex polyhedron (core vertices, face loops) + convex radius.
    Hull { verts: Vec<Vec3>, loops: Vec<Vec<usize>>, radius: f32 },
    Sphere { center: Vec3, radius: f32 },
    Capsule { a: Vec3, b: Vec3, radius: f32 },
}

impl LocalShape {
    /// A box of half extents `half` (the CORE) and convex radius `radius`.
    pub fn cuboid(half: Vec3, radius: f32) -> Self {
        let p = Poly::cuboid(half, Vec3::ZERO, Quat::IDENTITY, radius);
        LocalShape::Hull { verts: p.verts.clone(), loops: p.faces.iter().map(|f| f.v.clone()).collect(), radius }
    }

    pub fn world(&self, pos: Vec3, rot: Quat) -> WShape {
        match self {
            LocalShape::Hull { verts, loops, radius } => WShape::Poly(Poly::hull(verts, loops, pos, rot, *radius)),
            LocalShape::Sphere { center, radius } => WShape::Sphere { c: pos + rot * *center, r: *radius },
            LocalShape::Capsule { a, b, radius } => WShape::Capsule { a: pos + rot * *a, b: pos + rot * *b, r: *radius },
        }
    }
}

/// A shape in world space.
#[derive(Clone, Debug)]
pub enum WShape {
    Poly(Poly),
    Sphere { c: Vec3, r: f32 },
    Capsule { a: Vec3, b: Vec3, r: f32 },
}

impl WShape {
    /// Conservative bounds inflated by `margin`.
    pub fn aabb(&self, margin: f32) -> (Vec3, Vec3) {
        match self {
            WShape::Poly(p) => {
                let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for v in &p.verts {
                    mn = mn.min(*v);
                    mx = mx.max(*v);
                }
                let m = Vec3::splat(margin + p.radius);
                (mn - m, mx + m)
            }
            WShape::Sphere { c, r } => (*c - Vec3::splat(r + margin), *c + Vec3::splat(r + margin)),
            WShape::Capsule { a, b, r } => (a.min(*b) - Vec3::splat(r + margin), a.max(*b) + Vec3::splat(r + margin)),
        }
    }
}

fn spheres_of(s: &WShape) -> Vec<(Vec3, f32, u32)> {
    match s {
        WShape::Sphere { c, r } => vec![(*c, *r, 0)],
        WShape::Capsule { a, b, r } => vec![(*a, *r, 1), ((*a + *b) * 0.5, *r, 2), (*b, *r, 3)],
        WShape::Poly(_) => Vec::new(),
    }
}

fn core_points(p: &Poly) -> PointsCore<'_> {
    PointsCore(&p.verts)
}

/// Contact of one sphere (centre `c`, radius `r`) against a polyhedron (A = sphere, B = poly).
fn sphere_vs_poly(c: Vec3, r: f32, p: &Poly, tolerance: f32, id: u32, out: &mut Vec<RawContact>) -> bool {
    let sc = SphereCore(c);
    let pc = core_points(p);
    let d = gjk::distance(&sc, &pc);
    let pad = r + p.radius;
    if !d.overlap {
        let gap = d.distance - pad;
        if gap > tolerance {
            return false;
        }
        let n = if d.distance > 1.0e-6 {
            (d.pa - d.pb) / d.distance
        } else {
            // centre on a flat triangle: use the face normal on the centre's side
            let f = &p.faces[0];
            if f.n.dot(c - p.center) >= 0.0 {
                f.n
            } else {
                -f.n
            }
        };
        let surf_a = d.pa - n * r;
        let surf_b = d.pb + n * p.radius;
        out.push(RawContact { edge: false, id, pos: (surf_a + surf_b) * 0.5, normal: n, dist: gap });
        true
    } else if let Some((depth, n_epa, pa, pb)) = gjk::penetration(&sc, &pc) {
        let n = -n_epa;
        out.push(RawContact { edge: false, id, pos: (pa + pb) * 0.5, normal: n, dist: -depth - pad });
        true
    } else {
        false
    }
}

/// Narrow phase for any pair; contacts have normals from `b` into `a`.
pub fn collide(a: &WShape, b: &WShape, tolerance: f32, max_points: usize, out: &mut Vec<RawContact>) -> bool {
    match (a, b) {
        (WShape::Poly(pa), WShape::Poly(pb)) => collide_poly(pa, pb, tolerance, max_points, out),
        (WShape::Poly(_), _) => {
            let start = out.len();
            let hit = collide(b, a, tolerance, max_points, out);
            for c in &mut out[start..] {
                c.normal = -c.normal;
            }
            hit
        }
        (_, WShape::Poly(pb)) => {
            let mut hit = false;
            for (c, r, id) in spheres_of(a) {
                hit |= sphere_vs_poly(c, r, pb, tolerance, id, out);
            }
            hit
        }
        _ => {
            // sphere / capsule against sphere / capsule: closest points of the axes
            let seg = |s: &WShape| -> (Vec3, Vec3, f32) {
                match s {
                    WShape::Sphere { c, r } => (*c, *c, *r),
                    WShape::Capsule { a, b, r } => (*a, *b, *r),
                    WShape::Poly(_) => unreachable!(),
                }
            };
            let (a0, a1, ra) = seg(a);
            let (b0, b1, rb) = seg(b);
            let d = gjk::distance(&SegmentCore(a0, a1), &SegmentCore(b0, b1));
            let gap = d.distance - (ra + rb);
            if gap > tolerance {
                return false;
            }
            let n = if d.distance > 1.0e-6 { (d.pa - d.pb) / d.distance } else { Vec3::Z };
            out.push(RawContact { edge: false, id: 0x40000, pos: (d.pa - n * ra + d.pb + n * rb) * 0.5, normal: n, dist: gap });
            true
        }
    }
}

/// Signed gap between two shapes' PADDED surfaces: the core distance minus both radii, or the
/// negative penetration depth of the cores minus the radii when the cores overlap. Returns the
/// gap, the unit normal from `b` into `a`, and a point between the surfaces. This is the
/// `gsk_distance` of the engine's `update_manifold` (`01-collision-detection.md` 3.2.1).
pub fn signed_gap(a: &WShape, b: &WShape) -> Option<(f32, Vec3, Vec3)> {
    use gjk::Convex;
    fn core_radius(s: &WShape) -> (Box<dyn Convex + '_>, f32) {
        match s {
            WShape::Poly(p) => (Box::new(core_points(p)), p.radius),
            WShape::Sphere { c, r } => (Box::new(SphereCore(*c)), *r),
            WShape::Capsule { a, b, r } => (Box::new(SegmentCore(*a, *b)), *r),
        }
    }
    let (ca, ra) = core_radius(a);
    let (cb, rb) = core_radius(b);
    let d = gjk::distance(ca.as_ref(), cb.as_ref());
    let pad = ra + rb;
    if !d.overlap {
        let n = if d.distance > 1.0e-6 { (d.pa - d.pb) / d.distance } else { return None };
        Some((d.distance - pad, n, (d.pa - n * ra + d.pb + n * rb) * 0.5))
    } else {
        let (depth, n_epa, pa, pb) = gjk::penetration(ca.as_ref(), cb.as_ref())?;
        Some((-depth - pad, -n_epa, (pa + pb) * 0.5))
    }
}
