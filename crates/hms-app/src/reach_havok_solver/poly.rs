//! Convex polyhedra (boxes, hulls, triangles) in world space and the separating-axis contact
//! generator for a pair of them.
//!
//! The engine's box-box agent and its convex agents (`sub_180D0AB60` registers all of them) produce
//! a contact manifold from the CORE shapes plus their convex radii (see `05-shapes-mass-inertia.md`:
//! the stored half extents / vertices are the core, the collidable surface is core (+) sphere of
//! `radius`), generating a point whenever the padded surfaces are closer than the world's
//! `collisionTolerance` (0.0328). This module implements that contract with a SAT test (the 3
//! face axes of each box + 9 edge-edge axes for a box pair, generalised to any hull), face clipping
//! for face contacts and a closest-segment pair for edge-edge contacts. The numeric details of the
//! engine's own agents (point reduction, feature ids) are documented in `01-collision-detection.md`
//! and marked there; this file notes where it deviates.

use super::manifold::ContactPoint;
use glam::{Quat, Vec3};

/// One face of a world-space polyhedron: outward unit normal, plane offset (`n . v == d`) and the
/// counter-clockwise (seen from outside) vertex loop.
#[derive(Clone, Debug)]
pub struct Face {
    pub n: Vec3,
    pub d: f32,
    pub v: Vec<usize>,
}

/// A convex polyhedron in world space with its convex radius.
#[derive(Clone, Debug)]
pub struct Poly {
    pub verts: Vec<Vec3>,
    pub faces: Vec<Face>,
    /// Unique edges as vertex index pairs.
    pub edges: Vec<(usize, usize)>,
    pub radius: f32,
    pub center: Vec3,
}

impl Poly {
    fn build(verts: Vec<Vec3>, loops: Vec<Vec<usize>>, radius: f32) -> Poly {
        let center = verts.iter().copied().sum::<Vec3>() / verts.len() as f32;
        let mut faces = Vec::new();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for l in loops {
            if l.len() < 3 {
                continue;
            }
            // Newell's method: robust against a collinear first triple
            let a = verts[l[0]];
            let mut n = Vec3::ZERO;
            for k in 0..l.len() {
                let (p, q) = (verts[l[k]], verts[l[(k + 1) % l.len()]]);
                n += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
            }
            let n_len = n.length();
            if n_len < 1.0e-12 {
                continue;
            }
            let mut n = n / n_len;
            let mut l = l;
            if n.dot(a - center) < 0.0 {
                n = -n;
                l.reverse();
            }
            for k in 0..l.len() {
                let (i, j) = (l[k], l[(k + 1) % l.len()]);
                let e = if i < j { (i, j) } else { (j, i) };
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
            faces.push(Face { n, d: n.dot(a), v: l });
        }
        Poly { verts, faces, edges, radius, center }
    }

    /// An oriented box: half extents (the CORE), centre, rotation and convex radius.
    pub fn cuboid(half: Vec3, pos: Vec3, rot: Quat, radius: f32) -> Poly {
        let mut v = Vec::with_capacity(8);
        for i in 0..8u32 {
            let l = Vec3::new(if i & 1 == 0 { -half.x } else { half.x }, if i & 2 == 0 { -half.y } else { half.y }, if i & 4 == 0 { -half.z } else { half.z });
            v.push(pos + rot * l);
        }
        let loops = vec![vec![0, 2, 6, 4], vec![1, 5, 7, 3], vec![0, 4, 5, 1], vec![2, 3, 7, 6], vec![0, 1, 3, 2], vec![4, 6, 7, 5]];
        Poly::build(v, loops, radius)
    }

    /// A triangle as a flat two-sided polyhedron (what `hkpTriangleShape` is to the convex agents).
    pub fn triangle(a: Vec3, b: Vec3, c: Vec3, radius: f32) -> Poly {
        Poly::build(vec![a, b, c], vec![vec![0, 1, 2], vec![0, 2, 1]], radius)
    }

    /// A convex hull given by its vertices and face loops (any winding), e.g. a `phmo` polyhedron.
    pub fn hull(verts: &[Vec3], loops: &[Vec<usize>], pos: Vec3, rot: Quat, radius: f32) -> Poly {
        Poly::build(verts.iter().map(|v| pos + rot * *v).collect(), loops.to_vec(), radius)
    }

    pub fn support_dist(&self, n: Vec3) -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for v in &self.verts {
            let p = n.dot(*v);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, hi)
    }
}

/// A contact before it becomes a [`ContactPoint`]: normal from B into A, gap between the PADDED
/// surfaces (negative = penetrating).
#[derive(Clone, Copy, Debug)]
pub struct RawContact {
    /// true for an edge-edge feature pair (admitted by the engine at `dist < tolerance` instead of
    /// the penetration threshold, `sub_180E2A180`: `thr = nA + nB == 4 ? qi[+4] : qi[+8]`)
    pub edge: bool,
    pub id: u32,
    pub pos: Vec3,
    pub normal: Vec3,
    pub dist: f32,
}

impl RawContact {
    pub fn into_point(self) -> ContactPoint {
        ContactPoint::new(self.id, self.pos, self.normal, self.dist)
    }
}

/// Sutherland-Hodgman against `n . p <= d`; every vertex carries a tag (the original polygon
/// vertex, or a hash of the edge and plane that produced it) so contact ids survive from tick to tick.
fn clip_polygon(poly: &[(Vec3, u32)], n: Vec3, d: f32, plane: u32) -> Vec<(Vec3, u32)> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let ((p, tp), (q, tq)) = (poly[i], poly[(i + 1) % poly.len()]);
        let (sp, sq) = (n.dot(p) - d, n.dot(q) - d);
        if sp <= 0.0 {
            out.push((p, tp));
        }
        if (sp < 0.0 && sq > 0.0) || (sp > 0.0 && sq < 0.0) {
            let t = sp / (sp - sq);
            out.push((p + (q - p) * t, 0x100 + (tp.min(tq) * 31 + tp.max(tq)) * 7 + plane));
        }
    }
    out
}

fn closest_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let (d1, d2, r) = (q1 - p1, q2 - p2, p1 - p2);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    let (s, t);
    if a <= 1e-12 && e <= 1e-12 {
        return (p1, p2);
    }
    if a <= 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s0 = if denom > 1e-12 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t0 = 0.0;
                s0 = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t0 = 1.0;
                s0 = ((b - c) / a).clamp(0.0, 1.0);
            }
            s = s0;
            t = t0;
        }
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// Keep at most `max` points: the deepest, then the farthest from it, then the two that maximise
/// the covered area (the usual 4-point manifold reduction).
fn reduce(points: &mut Vec<RawContact>, max: usize) {
    if points.len() <= max {
        return;
    }
    let mut keep: Vec<RawContact> = Vec::with_capacity(max);
    let (mut bi, mut bd) = (0, f32::MAX);
    for (i, p) in points.iter().enumerate() {
        if p.dist < bd {
            bd = p.dist;
            bi = i;
        }
    }
    keep.push(points.swap_remove(bi));
    let far = |from: &[RawContact], pts: &[RawContact]| -> usize {
        let mut best = (0, -1.0f32);
        for (i, p) in pts.iter().enumerate() {
            let d = from.iter().map(|k| (k.pos - p.pos).length_squared()).fold(f32::MAX, f32::min);
            if d > best.1 {
                best = (i, d);
            }
        }
        best.0
    };
    let i = far(&keep, points);
    keep.push(points.swap_remove(i));
    while keep.len() < max && !points.is_empty() {
        // maximise the distance from the line / plane spanned so far
        let mut best = (0, -1.0f32);
        for (i, p) in points.iter().enumerate() {
            let a = (p.pos - keep[0].pos).cross(keep[1].pos - keep[0].pos).length();
            let score = if keep.len() == 2 { a } else { keep.iter().map(|k| (k.pos - p.pos).length_squared()).fold(f32::MAX, f32::min) };
            if score > best.1 {
                best = (i, score);
            }
        }
        keep.push(points.swap_remove(best.0));
    }
    *points = keep;
}

/// SAT contact generation between convex polyhedra `a` and `b`. Appends contacts whose padded gap
/// is below `tolerance`; normals point from `b` into `a`. Returns false when no axis lets them
/// touch (separated by more than `tolerance` after padding).
pub fn collide(a: &Poly, b: &Poly, tolerance: f32, max_points: usize, out: &mut Vec<RawContact>) -> bool {
    let pad = a.radius + b.radius;
    let margin = tolerance + pad;
    // face axes: separation = the other body's lowest point above the face plane
    let mut best_face = (f32::MIN, 0usize, true);
    for (owner_is_a, (me, other)) in [(true, (a, b)), (false, (b, a))] {
        for (fi, f) in me.faces.iter().enumerate() {
            let (lo, _) = other.support_dist(f.n);
            let s = lo - f.d;
            if s > margin {
                return false;
            }
            if s > best_face.0 {
                best_face = (s, fi, owner_is_a);
            }
        }
    }
    // edge-edge axes
    let mut best_edge: Option<(f32, usize, usize, Vec3)> = None;
    let ab = b.center - a.center;
    for (ia, ea) in a.edges.iter().enumerate() {
        let da = a.verts[ea.1] - a.verts[ea.0];
        for (ib, eb) in b.edges.iter().enumerate() {
            let db = b.verts[eb.1] - b.verts[eb.0];
            let mut axis = da.cross(db);
            let l = axis.length();
            if l < 1.0e-5 * da.length() * db.length() {
                continue;
            }
            axis /= l;
            if axis.dot(ab) < 0.0 {
                axis = -axis;
            }
            let (_, hi_a) = a.support_dist(axis);
            let (lo_b, _) = b.support_dist(axis);
            let s = lo_b - hi_a;
            if s > margin {
                return false;
            }
            // only a pair of SUPPORT edges is a face of the Minkowski difference; any other pair
            // gives the same number but its closest points are not where the shapes touch
            let eps = 1.0e-4 * (1.0 + hi_a.abs().max(lo_b.abs()));
            let on_a = axis.dot(a.verts[ea.0]) >= hi_a - eps && axis.dot(a.verts[ea.1]) >= hi_a - eps;
            let on_b = axis.dot(b.verts[eb.0]) <= lo_b + eps && axis.dot(b.verts[eb.1]) <= lo_b + eps;
            if !(on_a && on_b) {
                continue;
            }
            if best_edge.map_or(true, |e| s > e.0) {
                best_edge = Some((s, ia, ib, axis));
            }
        }
    }
    let start = out.len();
    let use_edge = best_edge.map_or(false, |e| e.0 > 0.98 * best_face.0 + 0.001);
    if use_edge {
        let (s, ia, ib, axis) = best_edge.unwrap();
        let (ea, eb) = (a.edges[ia], b.edges[ib]);
        let (pa, pb) = closest_segments(a.verts[ea.0], a.verts[ea.1], b.verts[eb.0], b.verts[eb.1]);
        // axis points from a to b; the contact normal points from b into a
        out.push(RawContact { edge: true, id: 0x20000 | ((ia as u32) << 8) | ib as u32, pos: (pa + pb) * 0.5, normal: -axis, dist: s - pad });
    } else {
        let (_, fi, ref_is_a) = best_face;
        let (rp, ip) = if ref_is_a { (a, b) } else { (b, a) };
        let rf = &rp.faces[fi];
        // incident face: the other body's face most anti-parallel to the reference normal
        let mut inc = (0usize, f32::MAX);
        for (k, f) in ip.faces.iter().enumerate() {
            let dot = f.n.dot(rf.n);
            if dot < inc.1 {
                inc = (k, dot);
            }
        }
        let ifc = &ip.faces[inc.0];
        let mut poly: Vec<(Vec3, u32)> = ifc.v.iter().map(|&i| (ip.verts[i], i as u32)).collect();
        // clip against the side planes of the reference face
        let nv = rf.v.len();
        for k in 0..nv {
            let (p, q) = (rp.verts[rf.v[k]], rp.verts[rf.v[(k + 1) % nv]]);
            let side = (q - p).cross(rf.n).normalize(); // outward from the face
            poly = clip_polygon(&poly, side, side.dot(p), k as u32);
            if poly.is_empty() {
                break;
            }
        }
        let n_into_a = if ref_is_a { -rf.n } else { rf.n }; // from b into a
        for (p, tag) in poly.iter() {
            let s = rf.n.dot(*p) - rf.d;
            if s <= margin {
                let surf = *p - rf.n * (s * 0.5);
                out.push(RawContact { edge: false, id: (((ref_is_a as u32) << 31) | ((fi as u32) << 24) | (tag & 0xFF_FFFF)).wrapping_add(1), pos: surf, normal: n_into_a, dist: s - pad });
            }
        }
    }
    let mut mine: Vec<RawContact> = out.drain(start..).collect();
    reduce(&mut mine, max_points);
    out.extend(mine);
    out.len() > start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_boxes_make_a_four_point_face_manifold() {
        let b = Poly::cuboid(Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY, 0.0);
        let a = Poly::cuboid(Vec3::splat(0.5), Vec3::new(0.1, -0.05, 0.99), Quat::IDENTITY, 0.0);
        let mut out = Vec::new();
        assert!(collide(&a, &b, 0.0328, 4, &mut out));
        assert_eq!(out.len(), 4, "{out:?}");
        for c in &out {
            assert!((c.normal - Vec3::Z).length() < 1e-5, "normal from b into a = +z, got {:?}", c.normal);
            assert!((c.dist + 0.01).abs() < 1e-5, "gap -0.01, got {}", c.dist);
        }
    }

    #[test]
    fn crossed_boxes_edge_on_edge_make_one_point() {
        // b: box rotated 45 degrees about x; a: box rotated 45 degrees about y above it: their
        // lowest / highest edges cross at right angles
        let q = |axis: Vec3| Quat::from_axis_angle(axis, std::f32::consts::FRAC_PI_4);
        let h = 0.5f32;
        let b = Poly::cuboid(Vec3::splat(h), Vec3::ZERO, q(Vec3::X), 0.0);
        let edge = h * std::f32::consts::SQRT_2;
        let a = Poly::cuboid(Vec3::splat(h), Vec3::new(0.0, 0.0, 2.0 * edge - 0.01), q(Vec3::Y), 0.0);
        let mut out = Vec::new();
        assert!(collide(&a, &b, 0.0328, 4, &mut out));
        assert_eq!(out.len(), 1, "{out:?}");
        let c = out[0];
        assert!((c.normal - Vec3::Z).length() < 1e-4, "{:?}", c.normal);
        assert!((c.dist + 0.01).abs() < 1e-4, "{}", c.dist);
        assert!(c.pos.x.abs() < 1e-4 && c.pos.y.abs() < 1e-4, "{:?}", c.pos);
    }

    #[test]
    fn radius_pads_the_surfaces_and_tolerance_gates_the_contact() {
        let b = Poly::cuboid(Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY, 0.0164);
        // cores 0.03 apart, radii 0.0164 each side of b only: padded gap = 0.03 - 0.0164
        let a = Poly::cuboid(Vec3::splat(0.5), Vec3::new(0.0, 0.0, 1.03), Quat::IDENTITY, 0.0);
        let mut out = Vec::new();
        assert!(collide(&a, &b, 0.0328, 4, &mut out));
        assert!((out[0].dist - (0.03 - 0.0164)).abs() < 1e-5);
        out.clear();
        let far = Poly::cuboid(Vec3::splat(0.5), Vec3::new(0.0, 0.0, 1.06), Quat::IDENTITY, 0.0);
        assert!(!collide(&far, &b, 0.0328, 4, &mut out), "core gap 0.06 - radius 0.0164 = 0.0436 > tolerance");
        assert!(out.is_empty());
    }
}
