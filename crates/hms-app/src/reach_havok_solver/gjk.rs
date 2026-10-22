//! Closest points between convex cores (GJK) and penetration (EPA) for the shapes the SAT module
//! does not cover: spheres and capsules against anything convex, and the distance queries the
//! time-of-impact search needs.
//!
//! The engine's `hkpGskfAgent` / `hkpPredGskfAgent` / `hkpGskBaseAgent` are GJK based (agent table
//! of `sub_180D0AB60`); this is a standard GJK/EPA over support functions, with the convex radius
//! kept OUT of the core (`radius` is added by the caller) exactly like the engine's padded shapes.

use glam::Vec3;

/// A convex core in world space.
pub trait Convex {
    /// The core's point farthest along `dir`.
    fn support(&self, dir: Vec3) -> Vec3;
    /// A point inside the core (start direction / EPA seed).
    fn center(&self) -> Vec3;
}

pub struct SphereCore(pub Vec3);
impl Convex for SphereCore {
    fn support(&self, _: Vec3) -> Vec3 {
        self.0
    }
    fn center(&self) -> Vec3 {
        self.0
    }
}

/// A segment (capsule axis).
pub struct SegmentCore(pub Vec3, pub Vec3);
impl Convex for SegmentCore {
    fn support(&self, d: Vec3) -> Vec3 {
        if d.dot(self.0) >= d.dot(self.1) {
            self.0
        } else {
            self.1
        }
    }
    fn center(&self) -> Vec3 {
        (self.0 + self.1) * 0.5
    }
}

pub struct PointsCore<'a>(pub &'a [Vec3]);
impl Convex for PointsCore<'_> {
    fn support(&self, d: Vec3) -> Vec3 {
        let mut best = self.0[0];
        let mut bd = d.dot(best);
        for p in &self.0[1..] {
            let s = d.dot(*p);
            if s > bd {
                bd = s;
                best = *p;
            }
        }
        best
    }
    fn center(&self) -> Vec3 {
        self.0.iter().copied().sum::<Vec3>() / self.0.len() as f32
    }
}

#[derive(Clone, Copy, Debug)]
struct SV {
    w: Vec3,
    a: Vec3,
    b: Vec3,
}

fn support(a: &dyn Convex, b: &dyn Convex, d: Vec3) -> SV {
    let pa = a.support(d);
    let pb = b.support(-d);
    SV { w: pa - pb, a: pa, b: pb }
}

/// The result of a distance query.
#[derive(Clone, Copy, Debug)]
pub struct Closest {
    /// Distance between the cores (0 when they overlap).
    pub distance: f32,
    /// Point on `a`, point on `b`.
    pub pa: Vec3,
    pub pb: Vec3,
    pub overlap: bool,
}

/// Closest point of the simplex to the origin, reducing the simplex; returns the barycentric point.
fn nearest_simplex(s: &mut Vec<SV>) -> (Vec3, Vec3, Vec3) {
    match s.len() {
        1 => (s[0].w, s[0].a, s[0].b),
        2 => {
            let (a, b) = (s[0], s[1]);
            let ab = b.w - a.w;
            let t = (-a.w.dot(ab) / ab.length_squared().max(1e-30)).clamp(0.0, 1.0);
            if t <= 0.0 {
                s.truncate(1);
                return (a.w, a.a, a.b);
            }
            if t >= 1.0 {
                *s = vec![b];
                return (b.w, b.a, b.b);
            }
            (a.w + ab * t, a.a + (b.a - a.a) * t, a.b + (b.b - a.b) * t)
        }
        3 => {
            // Ericson, closest point on triangle to origin
            let (a, b, c) = (s[0], s[1], s[2]);
            let (ab, ac) = (b.w - a.w, c.w - a.w);
            let ap = -a.w;
            let (d1, d2) = (ab.dot(ap), ac.dot(ap));
            if d1 <= 0.0 && d2 <= 0.0 {
                *s = vec![a];
                return (a.w, a.a, a.b);
            }
            let bp = -b.w;
            let (d3, d4) = (ab.dot(bp), ac.dot(bp));
            if d3 >= 0.0 && d4 <= d3 {
                *s = vec![b];
                return (b.w, b.a, b.b);
            }
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                let v = d1 / (d1 - d3);
                *s = vec![a, b];
                return (a.w + ab * v, a.a + (b.a - a.a) * v, a.b + (b.b - a.b) * v);
            }
            let cp = -c.w;
            let (d5, d6) = (ab.dot(cp), ac.dot(cp));
            if d6 >= 0.0 && d5 <= d6 {
                *s = vec![c];
                return (c.w, c.a, c.b);
            }
            let vb = d5 * d2 - d1 * d6;
            if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                let w = d2 / (d2 - d6);
                *s = vec![a, c];
                return (a.w + ac * w, a.a + (c.a - a.a) * w, a.b + (c.b - a.b) * w);
            }
            let va = d3 * d6 - d5 * d4;
            if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                *s = vec![b, c];
                return (b.w + (c.w - b.w) * w, b.a + (c.a - b.a) * w, b.b + (c.b - b.b) * w);
            }
            let denom = 1.0 / (va + vb + vc);
            let (v, w) = (vb * denom, vc * denom);
            (a.w + ab * v + ac * w, a.a + (b.a - a.a) * v + (c.a - a.a) * w, a.b + (b.b - a.b) * v + (c.b - a.b) * w)
        }
        _ => {
            // tetrahedron: origin inside if on the inner side of all four faces, else reduce to the
            // nearest face
            let p = [s[0], s[1], s[2], s[3]];
            let faces = [(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0)];
            let mut best: Option<(f32, Vec<SV>)> = None;
            let mut inside = true;
            for (i, j, k, o) in faces {
                let n = (p[j].w - p[i].w).cross(p[k].w - p[i].w);
                let sign_o = n.dot(p[o].w - p[i].w);
                let sign_0 = n.dot(-p[i].w);
                if sign_o * sign_0 < 0.0 {
                    inside = false;
                    let mut sub = vec![p[i], p[j], p[k]];
                    let (q, _, _) = nearest_simplex(&mut sub);
                    let d = q.length_squared();
                    if best.as_ref().map_or(true, |b| d < b.0) {
                        best = Some((d, sub));
                    }
                }
            }
            if inside {
                return (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            }
            *s = best.unwrap().1;
            nearest_simplex(s)
        }
    }
}

/// GJK distance between two convex cores. `overlap` is true when the origin is inside the
/// Minkowski difference (use [`penetration`] then).
pub fn distance(a: &dyn Convex, b: &dyn Convex) -> Closest {
    let mut dir = b.center() - a.center();
    if dir.length_squared() < 1e-12 {
        dir = Vec3::X;
    }
    let mut simplex = vec![support(a, b, dir)];
    let mut closest = simplex[0].w;
    let mut pa = simplex[0].a;
    let mut pb = simplex[0].b;
    for _ in 0..64 {
        let (q, qa, qb) = nearest_simplex(&mut simplex);
        closest = q;
        pa = qa;
        pb = qb;
        if simplex.len() == 4 || closest.length_squared() < 1e-12 {
            return Closest { distance: 0.0, pa, pb, overlap: true };
        }
        let d = -closest;
        let w = support(a, b, d);
        // progress test: the new support must pass the current closest point
        if w.w.dot(d) - closest.dot(d) <= 1e-6 * closest.length().max(1e-3) {
            break;
        }
        if simplex.iter().any(|v| (v.w - w.w).length_squared() < 1e-14) {
            break;
        }
        simplex.push(w);
    }
    Closest { distance: closest.length(), pa, pb, overlap: false }
}

/// EPA: penetration depth and normal (from `b` into `a`... i.e. the direction `a` must move to
/// separate) for overlapping cores. Falls back to the centre direction for degenerate cases.
pub fn penetration(a: &dyn Convex, b: &dyn Convex) -> Option<(f32, Vec3, Vec3, Vec3)> {
    // build an initial tetrahedron from supports along the axes
    let mut verts: Vec<SV> = Vec::new();
    for d in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z, Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, -1.0)] {
        let s = support(a, b, d);
        if !verts.iter().any(|v| (v.w - s.w).length_squared() < 1e-12) {
            verts.push(s);
        }
        if verts.len() == 4 {
            // need non-degenerate tetra
            let vol = (verts[1].w - verts[0].w).cross(verts[2].w - verts[0].w).dot(verts[3].w - verts[0].w);
            if vol.abs() < 1e-9 {
                verts.pop();
            } else {
                break;
            }
        }
    }
    if verts.len() < 4 {
        return None;
    }
    let mut faces: Vec<[usize; 3]> = vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
    // orient faces outward
    let c = (verts[0].w + verts[1].w + verts[2].w + verts[3].w) * 0.25;
    for f in faces.iter_mut() {
        let n = (verts[f[1]].w - verts[f[0]].w).cross(verts[f[2]].w - verts[f[0]].w);
        if n.dot(verts[f[0]].w - c) < 0.0 {
            f.swap(1, 2);
        }
    }
    for _ in 0..64 {
        // closest face to the origin
        let mut best = (f32::MAX, 0usize, Vec3::ZERO);
        for (i, f) in faces.iter().enumerate() {
            let n = (verts[f[1]].w - verts[f[0]].w).cross(verts[f[2]].w - verts[f[0]].w);
            let l = n.length();
            if l < 1e-12 {
                continue;
            }
            let n = n / l;
            let d = n.dot(verts[f[0]].w);
            if d < best.0 {
                best = (d, i, n);
            }
        }
        let (d, fi, n) = best;
        let s = support(a, b, n);
        if s.w.dot(n) - d < 1e-5 {
            let f = faces[fi];
            // contact points: barycentric of the origin's projection
            let p = n * d;
            let (v0, v1, v2) = (verts[f[0]], verts[f[1]], verts[f[2]]);
            let (e0, e1, e2) = (v1.w - v0.w, v2.w - v0.w, p - v0.w);
            let (d00, d01, d11, d20, d21) = (e0.dot(e0), e0.dot(e1), e1.dot(e1), e2.dot(e0), e2.dot(e1));
            let den = d00 * d11 - d01 * d01;
            let (bv, bw) = if den.abs() > 1e-20 { ((d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den) } else { (0.0, 0.0) };
            let bu = 1.0 - bv - bw;
            let pa = v0.a * bu + v1.a * bv + v2.a * bw;
            let pb = v0.b * bu + v1.b * bv + v2.b * bw;
            return Some((d, n, pa, pb));
        }
        // expand: remove the faces the new support point sees, stitch the horizon
        let idx = verts.len();
        verts.push(s);
        let visible: Vec<bool> = faces
            .iter()
            .map(|f| {
                let fn_ = (verts[f[1]].w - verts[f[0]].w).cross(verts[f[2]].w - verts[f[0]].w);
                let l = fn_.length();
                l > 1.0e-12 && fn_.dot(s.w - verts[f[0]].w) > 1.0e-7 * l
            })
            .collect();
        if !visible.iter().any(|v| *v) {
            // numerically on the hull already: take the current best face
            let f = faces[fi];
            let p = n * d;
            let (v0, v1, v2) = (verts[f[0]], verts[f[1]], verts[f[2]]);
            let (e0, e1, e2) = (v1.w - v0.w, v2.w - v0.w, p - v0.w);
            let (d00, d01, d11, d20, d21) = (e0.dot(e0), e0.dot(e1), e1.dot(e1), e2.dot(e0), e2.dot(e1));
            let den = d00 * d11 - d01 * d01;
            let (bv, bw) = if den.abs() > 1e-20 { ((d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den) } else { (0.0, 0.0) };
            let bu = 1.0 - bv - bw;
            return Some((d, n, v0.a * bu + v1.a * bv + v2.a * bw, v0.b * bu + v1.b * bv + v2.b * bw));
        }
        let mut directed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for (f, vis) in faces.iter().zip(&visible) {
            if *vis {
                for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                    directed.insert(e);
                }
            }
        }
        let mut keep: Vec<[usize; 3]> = Vec::with_capacity(faces.len());
        for (f, vis) in faces.iter().zip(&visible) {
            if !*vis {
                keep.push(*f);
            }
        }
        for (f, vis) in faces.iter().zip(&visible) {
            if *vis {
                for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                    if !directed.contains(&(e.1, e.0)) {
                        keep.push([e.0, e.1, idx]);
                    }
                }
            }
        }
        faces = keep;
        if faces.len() > 2048 {
            return None;
        }
    }
    None
}
