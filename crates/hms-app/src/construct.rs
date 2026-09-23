//! CAD-style construction geometry for the Forge editor.
//!
//! An *anchor* is a snappable point of an object's ORIENTED bounding box -- its 8 corners,
//! 12 edge midpoints, 6 face centres, or its centre -- or the centre of a whole selection.
//! A *guide* is a dotted construction line drawn between two anchors. Guides are not map
//! objects: they exist only in the editor and are never written to a .mvar.
//!
//! Every guide contributes SNAP TARGETS: its two endpoints, its midpoint, and its
//! intersection with any other guide. That is what makes "centre this inside that box"
//! exact -- draw a face's two diagonals, and the point where they cross IS the face centre,
//! so anchoring the selection to that intersection centres it with no eyeballing. The same
//! machinery gives distance constraints (place N units along a guide) and the mirror plane
//! a symmetric map like Griffball needs.

use glam::{Mat4, Vec3};

/// Which point of an object's box an anchor refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnchorKind {
    /// Box centre (or, for a multi-object selection, the centre of the combined box).
    Center,
    /// Face centre. 0..6 = -X, +X, -Y, +Y, -Z, +Z.
    Face(u8),
    /// Corner. Index bits: 1 = +X, 2 = +Y, 4 = +Z.
    Corner(u8),
    /// Edge midpoint, 0..12 (see [`BOX_EDGES`]).
    Edge(u8),
    /// A point that is not a box feature: a guide intersection, a guide midpoint, or a
    /// free point picked on the map surface.
    Point,
}

impl AnchorKind {
    pub fn label(&self) -> &'static str {
        match self {
            AnchorKind::Center => "centre",
            AnchorKind::Face(_) => "face",
            AnchorKind::Corner(_) => "corner",
            AnchorKind::Edge(_) => "edge",
            AnchorKind::Point => "point",
        }
    }
    /// Marker colour, so the three anchor families read apart at a glance.
    pub fn color(&self) -> [f32; 3] {
        match self {
            AnchorKind::Center => [1.0, 0.85, 0.2],
            AnchorKind::Face(_) => [0.3, 0.9, 1.0],
            AnchorKind::Corner(_) => [1.0, 0.4, 0.85],
            AnchorKind::Edge(_) => [0.5, 1.0, 0.5],
            AnchorKind::Point => [1.0, 1.0, 1.0],
        }
    }
}

/// A snappable point, optionally tied to the object it was taken from.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Object the anchor came from. `None` = selection centre, guide intersection, or a
    /// free point -- anything not owned by a single object.
    pub datum: Option<u32>,
    pub kind: AnchorKind,
    pub pos: Vec3,
}

impl Anchor {
    pub fn point(pos: Vec3) -> Self {
        Anchor { datum: None, kind: AnchorKind::Point, pos }
    }
}

/// A construction line between two anchors.
#[derive(Clone, Debug)]
pub struct Guide {
    pub a: Anchor,
    pub b: Anchor,
    pub color: [f32; 3],
}

impl Guide {
    pub fn length(&self) -> f32 {
        (self.b.pos - self.a.pos).length()
    }
    pub fn mid(&self) -> Vec3 {
        (self.a.pos + self.b.pos) * 0.5
    }
    /// Unit direction A -> B (zero for a degenerate guide).
    pub fn dir(&self) -> Vec3 {
        (self.b.pos - self.a.pos).normalize_or_zero()
    }
}

/// The 12 edges of a box as corner-index pairs. Corner index bits: 1 = +X, 2 = +Y, 4 = +Z.
const BOX_EDGES: [(u8, u8); 12] = [
    (0, 1), (2, 3), (4, 5), (6, 7), // along X
    (0, 2), (1, 3), (4, 6), (5, 7), // along Y
    (0, 4), (1, 5), (2, 6), (3, 7), // along Z
];

/// An oriented bounding box: centre plus the three half-extent AXES (already scaled, so
/// `c + hx + hy + hz` is a corner).
#[derive(Clone, Copy, Debug)]
pub struct Obb {
    pub c: Vec3,
    pub hx: Vec3,
    pub hy: Vec3,
    pub hz: Vec3,
}

impl Obb {
    /// Build from a placement transform and the model's LOCAL aabb. Using the local box +
    /// transform (rather than a world-axis-aligned box) is what makes a rotated block's
    /// corners its real corners.
    pub fn from_local(xform: &Mat4, lmin: Vec3, lmax: Vec3) -> Obb {
        let lc = (lmin + lmax) * 0.5;
        let he = (lmax - lmin) * 0.5;
        Obb {
            c: xform.transform_point3(lc),
            hx: xform.transform_vector3(Vec3::X * he.x),
            hy: xform.transform_vector3(Vec3::Y * he.y),
            hz: xform.transform_vector3(Vec3::Z * he.z),
        }
    }

    /// Axis-aligned box (used when an object has no render mesh to take local bounds from).
    pub fn from_aabb(min: Vec3, max: Vec3) -> Obb {
        let he = (max - min) * 0.5;
        Obb {
            c: (min + max) * 0.5,
            hx: Vec3::X * he.x,
            hy: Vec3::Y * he.y,
            hz: Vec3::Z * he.z,
        }
    }

    pub fn corner(&self, i: u8) -> Vec3 {
        let s = |bit: u8| if i & bit != 0 { 1.0 } else { -1.0 };
        self.c + self.hx * s(1) + self.hy * s(2) + self.hz * s(4)
    }

    pub fn face_center(&self, i: u8) -> Vec3 {
        match i {
            0 => self.c - self.hx,
            1 => self.c + self.hx,
            2 => self.c - self.hy,
            3 => self.c + self.hy,
            4 => self.c - self.hz,
            _ => self.c + self.hz,
        }
    }

    /// Unit OUTWARD normal of face `i` (same index order as [`Obb::face_center`]:
    /// 0..6 = -X, +X, -Y, +Y, -Z, +Z). Taken from the box's own axes, so a rotated block
    /// reports the face's real world direction -- which is what a face-to-face constraint
    /// has to align against.
    pub fn face_normal(&self, i: u8) -> Vec3 {
        let n = match i {
            0 => -self.hx,
            1 => self.hx,
            2 => -self.hy,
            3 => self.hy,
            4 => -self.hz,
            _ => self.hz,
        };
        n.normalize_or_zero()
    }

    pub fn edge_mid(&self, i: u8) -> Vec3 {
        let (a, b) = BOX_EDGES[i as usize % 12];
        (self.corner(a) + self.corner(b)) * 0.5
    }

    /// Every anchor of this box, filtered by `mode` (see [`AnchorMode`]).
    pub fn anchors(&self, datum: Option<u32>, mode: AnchorMode) -> Vec<Anchor> {
        let mut out = Vec::new();
        let mut push = |kind: AnchorKind, pos: Vec3| out.push(Anchor { datum, kind, pos });
        if mode.wants_center() {
            push(AnchorKind::Center, self.c);
        }
        if mode.wants_faces() {
            for i in 0..6u8 {
                push(AnchorKind::Face(i), self.face_center(i));
            }
        }
        if mode.wants_corners() {
            for i in 0..8u8 {
                push(AnchorKind::Corner(i), self.corner(i));
            }
        }
        if mode.wants_edges() {
            for i in 0..12u8 {
                push(AnchorKind::Edge(i), self.edge_mid(i));
            }
        }
        out
    }

    /// The 12 wireframe edges, for drawing the box a hovered anchor belongs to.
    pub fn edge_segments(&self) -> Vec<(Vec3, Vec3)> {
        BOX_EDGES.iter().map(|&(a, b)| (self.corner(a), self.corner(b))).collect()
    }
}

/// Which anchor families the user wants to snap to. Restricting this is the difference
/// between "somewhere on that block" and "that block's top face, exactly".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnchorMode {
    All,
    Corners,
    Edges,
    Faces,
    Centers,
}

impl AnchorMode {
    fn wants_center(self) -> bool {
        matches!(self, AnchorMode::All | AnchorMode::Centers)
    }
    fn wants_faces(self) -> bool {
        matches!(self, AnchorMode::All | AnchorMode::Faces)
    }
    fn wants_corners(self) -> bool {
        matches!(self, AnchorMode::All | AnchorMode::Corners)
    }
    fn wants_edges(self) -> bool {
        matches!(self, AnchorMode::All | AnchorMode::Edges)
    }
    pub const ALL: [AnchorMode; 5] = [
        AnchorMode::All,
        AnchorMode::Corners,
        AnchorMode::Edges,
        AnchorMode::Faces,
        AnchorMode::Centers,
    ];
    pub fn label(self) -> &'static str {
        match self {
            AnchorMode::All => "All points",
            AnchorMode::Corners => "Corners",
            AnchorMode::Edges => "Edge midpoints",
            AnchorMode::Faces => "Face centres",
            AnchorMode::Centers => "Centres",
        }
    }
}

/// Where a snap target came from (drives its marker colour and the status text).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SnapKind {
    /// A guide endpoint.
    End,
    /// A guide's midpoint -- the "half way along" constraint, free of charge.
    Mid,
    /// Where two guides cross. This is the one that centres things.
    Intersection,
}

#[derive(Clone, Copy, Debug)]
pub struct SnapTarget {
    pub pos: Vec3,
    pub kind: SnapKind,
}

impl SnapKind {
    pub fn color(self) -> [f32; 3] {
        match self {
            SnapKind::End => [0.9, 0.9, 0.95],
            SnapKind::Mid => [0.6, 0.8, 1.0],
            SnapKind::Intersection => [1.0, 0.3, 0.3],
        }
    }
}

/// Closest approach between two segments. Returns the midpoint of the closest pair when the
/// two lines pass within `tol` of each other -- i.e. where they visually "cross" in 3D, which
/// for the diagonals of a planar face is exactly the face centre.
///
/// Nearly-parallel pairs are rejected: they have no well-defined crossing, and accepting one
/// would scatter bogus targets all along two collinear guides.
fn segment_intersection(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3, tol: f32) -> Option<Vec3> {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let (l1, l2) = (d1.length(), d2.length());
    if l1 < 1e-5 || l2 < 1e-5 {
        return None;
    }
    let (u1, u2) = (d1 / l1, d2 / l2);
    let cos = u1.dot(u2).abs();
    if cos > 0.999 {
        return None; // parallel / collinear: no crossing point
    }
    let r = p1 - p2;
    let b = u1.dot(u2);
    let denom = 1.0 - b * b;
    // Standard closest-points-between-two-lines solve, with e = u1.r and f = u2.r:
    //   t1 = (b*f - e) / (1 - b^2)  along line 1,  t2 = (f - b*e) / (1 - b^2)  along line 2.
    let e = u1.dot(r);
    let f = u2.dot(r);
    let t1 = (b * f - e) / denom;
    let t2 = (f - b * e) / denom;
    // Both parameters must land ON the drawn segments (with a small tolerance in units), so a
    // guide only snaps where it actually reaches.
    let slack = tol.max(1e-4);
    if t1 < -slack || t1 > l1 + slack || t2 < -slack || t2 > l2 + slack {
        return None;
    }
    let c1 = p1 + u1 * t1.clamp(0.0, l1);
    let c2 = p2 + u2 * t2.clamp(0.0, l2);
    if (c1 - c2).length() > tol {
        return None;
    }
    Some((c1 + c2) * 0.5)
}

/// Every point the editor will snap to, derived from the guides: endpoints, midpoints, and
/// pairwise intersections. Near-duplicate points are merged so a shared corner does not
/// stack a dozen identical targets.
pub fn snap_targets(guides: &[Guide], tol: f32) -> Vec<SnapTarget> {
    let mut out: Vec<SnapTarget> = Vec::new();
    let push = |pos: Vec3, kind: SnapKind, out: &mut Vec<SnapTarget>| {
        // An intersection outranks an endpoint at the same spot -- it is the more meaningful
        // constraint, and it is what the marker should show.
        if let Some(e) = out.iter_mut().find(|t| (t.pos - pos).length() < tol) {
            if kind == SnapKind::Intersection {
                e.kind = SnapKind::Intersection;
                e.pos = pos;
            }
            return;
        }
        out.push(SnapTarget { pos, kind });
    };
    for g in guides {
        push(g.a.pos, SnapKind::End, &mut out);
        push(g.b.pos, SnapKind::End, &mut out);
        push(g.mid(), SnapKind::Mid, &mut out);
    }
    for i in 0..guides.len() {
        for j in (i + 1)..guides.len() {
            let (g, h) = (&guides[i], &guides[j]);
            if let Some(p) = segment_intersection(g.a.pos, g.b.pos, h.a.pos, h.b.pos, tol) {
                push(p, SnapKind::Intersection, &mut out);
            }
        }
    }
    out
}

/// Append a DOTTED line to an overlay segment list. `dash` is the on-length in world units;
/// the gap matches it. Short guides still get a minimum number of dashes so they read as
/// construction geometry rather than as solid map lines.
pub fn dashed(
    a: Vec3,
    b: Vec3,
    dash: f32,
    color: [f32; 3],
    out: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>,
) {
    let d = b - a;
    let len = d.length();
    if len < 1e-5 {
        return;
    }
    let dir = d / len;
    // Aim for an odd number of cells so the line starts and ends with an "on" dash.
    let cell = dash.max(len / 201.0).max(1e-4);
    let cells = ((len / cell).round() as i32).max(3);
    let step = len / cells as f32;
    let mut i = 0;
    while i < cells {
        let t0 = i as f32 * step;
        let t1 = ((i + 1) as f32 * step).min(len);
        out.push(((a + dir * t0).into(), (a + dir * t1).into(), color));
        i += 2; // skip a cell = the gap
    }
}

/// Append a 3-axis cross marker (a small "+" in 3D) at `p`.
pub fn cross(p: Vec3, r: f32, color: [f32; 3], out: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>) {
    for ax in [Vec3::X, Vec3::Y, Vec3::Z] {
        out.push(((p - ax * r).into(), (p + ax * r).into(), color));
    }
}

/// Append a small axis-aligned diamond (octahedron wireframe) -- used for the hovered
/// anchor so it stands out from the plain crosses of the other candidates.
pub fn diamond(p: Vec3, r: f32, color: [f32; 3], out: &mut Vec<([f32; 3], [f32; 3], [f32; 3])>) {
    let (x, y, z) = (Vec3::X * r, Vec3::Y * r, Vec3::Z * r);
    let pts = [p + z, p + x, p + y, p - x, p - y, p - z];
    let edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (5, 1), (5, 2), (5, 3), (5, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    ];
    for (a, b) in edges {
        out.push((pts[a].into(), pts[b].into(), color));
    }
}

/// Reflect a point across the plane through `origin` with unit normal `n`.
pub fn mirror_point(p: Vec3, origin: Vec3, n: Vec3) -> Vec3 {
    p - n * (2.0 * (p - origin).dot(n))
}

/// Reflect a DIRECTION across a plane with unit normal `n` (no origin term).
pub fn mirror_dir(v: Vec3, n: Vec3) -> Vec3 {
    v - n * (2.0 * v.dot(n))
}

/// How tightly a face-to-face coincident constraint binds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaceMate {
    /// Make the two faces COPLANAR: slide along the target's normal only, so the moved part
    /// keeps its position within the plane. This is the CAD "coincident" most people want --
    /// it makes the faces flush without throwing the object sideways.
    Plane,
    /// Also make the two face CENTRES coincide (coplanar AND centred).
    Centered,
}

/// Translation that makes face A coincident with face B.
///
/// `Plane` projects A's face centre onto B's plane along B's normal -- the faces end up
/// flush, and nothing moves within the plane. `Centered` puts A's face centre exactly on
/// B's, which is coplanar as well but also centres it.
///
/// Returns a delta to add to every object being moved. A degenerate target normal (a
/// zero-extent box axis) falls back to no movement rather than flinging the selection.
fn face_coincident_delta(a_center: Vec3, b_center: Vec3, b_normal: Vec3, mate: FaceMate) -> Vec3 {
    match mate {
        FaceMate::Centered => b_center - a_center,
        FaceMate::Plane => {
            let n = b_normal.normalize_or_zero();
            if n.length_squared() < 0.5 {
                return Vec3::ZERO;
            }
            n * (b_center - a_center).dot(n)
        }
    }
}

/// The full face-to-face MATE — rotation AND translation.
///
/// A real mate turns the part so its face looks straight INTO the target face (normals
/// anti-parallel), then brings them together. Translating alone only works when the two
/// parts already happen to be square to each other, which is why a plain slide "isn't
/// working too well" on anything rotated.
///
/// The rotation pivots about face A's CENTRE, so that face stays put while the part swings
/// round it; then the translation closes the gap (`Plane`) or also centres it (`Centered`).
///
/// Returns `(rotation, delta)`. A part vertex/origin `p` ends at
/// `a_center + rotation * (p - a_center) + delta`.
pub fn face_mate_transform(
    a_center: Vec3,
    a_normal: Vec3,
    b_center: Vec3,
    b_normal: Vec3,
    mate: FaceMate,
    rotate: bool,
) -> (glam::Quat, Vec3) {
    let an = a_normal.normalize_or_zero();
    let bn = b_normal.normalize_or_zero();
    if bn.length_squared() < 0.5 {
        return (glam::Quat::IDENTITY, Vec3::ZERO);
    }
    // face A must end up pointing INTO face B
    let q = if rotate && an.length_squared() > 0.5 {
        glam::Quat::from_rotation_arc(an, -bn)
    } else {
        glam::Quat::IDENTITY
    };
    // A's centre is the pivot, so it does not move under the rotation
    let delta = face_coincident_delta(a_center, b_center, bn, mate);
    (q, delta)
}

/// An orthonormal pair spanning the plane whose normal is `n`. Picks the reference axis
/// least parallel to `n` so the basis never degenerates.
fn basis_from_normal(n: Vec3) -> (Vec3, Vec3) {
    let n = n.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return (Vec3::X, Vec3::Y);
    }
    let r = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = n.cross(r).normalize_or_zero();
    let u = if u.length_squared() < 0.5 { Vec3::Y } else { u };
    (u, n.cross(u).normalize_or_zero())
}

/// A construction CIRCLE as a closed fan of guides, in the plane through `center` with
/// normal `normal`. Every vertex becomes a snap point and every chord can be intersected,
/// so a circle is how you space things evenly around a point (a ring of cover, a round
/// arena) without eyeballing angles. `segments` is clamped to a sane 3..=256.
fn circle_guides(center: Vec3, normal: Vec3, radius: f32, segments: u32, color: [f32; 3]) -> Vec<Guide> {
    let seg = segments.clamp(3, 256);
    let (u, v) = basis_from_normal(normal);
    let r = radius.max(0.0);
    let pt = |k: u32| {
        let t = (k % seg) as f32 / seg as f32 * std::f32::consts::TAU;
        center + u * (r * t.cos()) + v * (r * t.sin())
    };
    (0..seg)
        .map(|k| Guide { a: Anchor::point(pt(k)), b: Anchor::point(pt(k + 1)), color })
        .collect()
}

/// A construction SQUARE (4 guides) centred on `center`, lying in the plane with normal
/// `normal`. `half` is the half-side, so the square spans `2*half`. `rotation` turns it
/// within its own plane (radians) for diagonal layouts.
fn square_guides(center: Vec3, normal: Vec3, half: f32, rotation: f32, color: [f32; 3]) -> Vec<Guide> {
    let (u0, v0) = basis_from_normal(normal);
    let (c, s) = (rotation.cos(), rotation.sin());
    let u = u0 * c + v0 * s;
    let v = v0 * c - u0 * s;
    let h = half.max(0.0);
    let p = [
        center - u * h - v * h,
        center + u * h - v * h,
        center + u * h + v * h,
        center - u * h + v * h,
    ];
    (0..4)
        .map(|i| Guide { a: Anchor::point(p[i]), b: Anchor::point(p[(i + 1) % 4]), color })
        .collect()
}

/// A construction SHAPE the user drew: a circle or a square that lives in the editor as an
/// editable thing, not a one-shot burst of guides. It can be re-centred, resized and deleted
/// after the fact, which is what makes it usable as a layout aid rather than a stamp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShapeKind {
    Circle,
    Square,
}

impl ShapeKind {
    pub fn label(self) -> &'static str {
        match self {
            ShapeKind::Circle => "circle",
            ShapeKind::Square => "square",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)] // PartialEq for the undo-snapshot no-op check
pub struct Shape {
    pub kind: ShapeKind,
    pub center: Vec3,
    /// Plane normal. The shape is drawn in the plane through `center` with this normal.
    pub normal: Vec3,
    /// Circle radius, or the square's half-side.
    pub radius: f32,
    /// In-plane rotation, radians (squares; a circle is rotation-invariant).
    pub rot: f32,
    pub segments: u32,
    pub color: [f32; 3],
}

impl Shape {
    pub fn new(kind: ShapeKind, center: Vec3, normal: Vec3) -> Self {
        let color = match kind {
            ShapeKind::Circle => [1.0, 0.75, 0.3],
            ShapeKind::Square => [0.6, 0.8, 1.0],
        };
        Shape { kind, center, normal, radius: 0.0, rot: 0.0, segments: 32, color }
    }

    /// The outline as guides -- so a shape snaps exactly like any other construction line
    /// (endpoints, midpoints and intersections all become snap targets).
    pub fn guides(&self) -> Vec<Guide> {
        match self.kind {
            ShapeKind::Circle => circle_guides(self.center, self.normal, self.radius, self.segments, self.color),
            ShapeKind::Square => square_guides(self.center, self.normal, self.radius, self.rot, self.color),
        }
    }

    /// `n` points spaced evenly along the outline, each with the OUTWARD in-plane direction
    /// at that point. This is what "fill the edge with this object" walks: a circle spaces by
    /// angle, a square by ARC LENGTH so the copies stay evenly spaced across its corners
    /// rather than bunching four-per-side.
    pub fn perimeter_points(&self, n: u32) -> Vec<(Vec3, Vec3)> {
        let n = n.max(1);
        let (u, v) = basis_from_normal(self.normal);
        match self.kind {
            ShapeKind::Circle => (0..n)
                .map(|i| {
                    let t = i as f32 / n as f32 * std::f32::consts::TAU;
                    let out = (u * t.cos() + v * t.sin()).normalize_or_zero();
                    (self.center + out * self.radius, out)
                })
                .collect(),
            ShapeKind::Square => {
                let (c, s) = (self.rot.cos(), self.rot.sin());
                let ax = u * c + v * s;
                let ay = v * c - u * s;
                let h = self.radius;
                // corners in order, then walk the closed path by arc length
                let p = [
                    self.center - ax * h - ay * h,
                    self.center + ax * h - ay * h,
                    self.center + ax * h + ay * h,
                    self.center - ax * h + ay * h,
                ];
                let side = (2.0 * h).max(1e-6);
                let total = side * 4.0;
                (0..n)
                    .map(|i| {
                        let d = i as f32 / n as f32 * total;
                        let e = ((d / side) as usize).min(3);
                        let f = (d - e as f32 * side) / side;
                        let a = p[e];
                        let b = p[(e + 1) % 4];
                        let pos = a + (b - a) * f;
                        let out = (pos - self.center).normalize_or_zero();
                        (pos, out)
                    })
                    .collect()
            }
        }
    }

    /// Distance from `p` to the outline (edge-select helper, exercised by the tests only).
    #[cfg(test)]
    pub fn dist_to_outline(&self, p: Vec3) -> f32 {
        self.guides()
            .iter()
            .map(|g| dist_point_segment(p, g.a.pos, g.b.pos))
            .fold(f32::INFINITY, f32::min)
    }
}

/// Distance from a point to a finite segment.
#[cfg(test)]
pub fn dist_point_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let l2 = ab.length_squared();
    if l2 < 1e-12 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// Where a ray meets the plane through `p0` with normal `n`. `None` when the ray is parallel
/// to the plane or would hit behind the viewer -- a drag must not snap to a point behind the
/// camera.
pub fn ray_plane(origin: Vec3, dir: Vec3, p0: Vec3, n: Vec3) -> Option<Vec3> {
    let n = n.normalize_or_zero();
    if n.length_squared() < 0.5 {
        return None;
    }
    let denom = dir.dot(n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (p0 - origin).dot(n) / denom;
    (t > 0.0).then(|| origin + dir * t)
}

/// Where one arrayed copy goes.
///
/// `first` is where the ORIGINAL sits (the shape's first perimeter point). For copy `k` at
/// `target`, the copy is the original ROTATED about the shape's normal through the shape
/// centre -- so a ring of cover turns to follow the circle -- and then nudged by whatever the
/// rotation did not already account for. On a circle that nudge is zero (the rotation maps the
/// first point exactly onto the k-th); on a square it corrects the difference, because a
/// square's perimeter points are not related by a pure rotation.
///
/// Returns `(rotation, delta)`: a copy whose original pose is `p` ends at
/// `shape_center + rotation * (p - shape_center) + delta`.
pub fn array_copy_transform(
    shape_center: Vec3,
    normal: Vec3,
    first: Vec3,
    first_out: Vec3,
    target: Vec3,
    target_out: Vec3,
    rotate: bool,
) -> (glam::Quat, Vec3) {
    let n = normal.normalize_or_zero();
    let q = if rotate && n.length_squared() > 0.5 {
        let a = first_out.normalize_or_zero();
        let b = target_out.normalize_or_zero();
        let ang = a.cross(b).dot(n).atan2(a.dot(b));
        glam::Quat::from_axis_angle(n, ang)
    } else {
        glam::Quat::IDENTITY
    };
    let rotated_first = shape_center + q * (first - shape_center);
    (q, target - rotated_first)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_diagonals_cross_at_the_face_centre() {
        // A 4x6x2 box at an offset, rotated 30 degrees about Z: the two diagonals of its +Z
        // face must cross exactly at that face's centre.
        let rot = Mat4::from_rotation_z(0.5236) * Mat4::from_translation(Vec3::new(3.0, -2.0, 1.0));
        let obb = Obb::from_local(&rot, Vec3::new(-2.0, -3.0, -1.0), Vec3::new(2.0, 3.0, 1.0));
        // +Z face corners have bit 4 set: 4,5,6,7. Diagonals are 4-7 and 5-6.
        let hit = segment_intersection(obb.corner(4), obb.corner(7), obb.corner(5), obb.corner(6), 1e-3)
            .expect("diagonals must cross");
        assert!((hit - obb.face_center(5)).length() < 1e-3, "got {hit:?}");
    }

    #[test]
    fn space_diagonals_cross_at_the_box_centre() {
        let obb = Obb::from_aabb(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 4.0, 6.0));
        let hit = segment_intersection(obb.corner(0), obb.corner(7), obb.corner(1), obb.corner(6), 1e-3)
            .expect("space diagonals must cross");
        assert!((hit - obb.c).length() < 1e-3, "got {hit:?} want {:?}", obb.c);
    }

    #[test]
    fn parallel_guides_do_not_produce_an_intersection() {
        let a = (Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0));
        let b = (Vec3::new(0.0, 2.0, 0.0), Vec3::new(10.0, 2.0, 0.0));
        assert!(segment_intersection(a.0, a.1, b.0, b.1, 0.05).is_none());
    }

    #[test]
    fn skew_lines_that_never_meet_are_rejected() {
        // Perpendicular but separated in Z by more than the tolerance.
        let a = (Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
        let b = (Vec3::new(0.0, -5.0, 3.0), Vec3::new(0.0, 5.0, 3.0));
        assert!(segment_intersection(a.0, a.1, b.0, b.1, 0.05).is_none());
    }

    #[test]
    fn crossing_beyond_the_drawn_segment_is_rejected() {
        // The lines would cross at x=20, well past where either segment stops.
        let a = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));
        let b = (Vec3::new(20.0, -5.0, 0.0), Vec3::new(20.0, 5.0, 0.0));
        assert!(segment_intersection(a.0, a.1, b.0, b.1, 0.05).is_none());
    }

    #[test]
    fn snap_targets_merge_shared_endpoints_and_find_the_centre() {
        let c = |p: Vec3| Anchor::point(p);
        let g = vec![
            Guide { a: c(Vec3::new(-1.0, -1.0, 0.0)), b: c(Vec3::new(1.0, 1.0, 0.0)), color: [1.0; 3] },
            Guide { a: c(Vec3::new(-1.0, 1.0, 0.0)), b: c(Vec3::new(1.0, -1.0, 0.0)), color: [1.0; 3] },
        ];
        let t = snap_targets(&g, 0.05);
        let inter: Vec<_> = t.iter().filter(|t| t.kind == SnapKind::Intersection).collect();
        assert_eq!(inter.len(), 1);
        assert!(inter[0].pos.length() < 1e-4);
        // The two midpoints coincide with the crossing, so they must have merged into it.
        assert_eq!(t.iter().filter(|t| t.kind == SnapKind::Mid).count(), 0);
    }

    #[test]
    fn mirror_reflects_position_and_direction() {
        let o = Vec3::new(5.0, 0.0, 0.0);
        let n = Vec3::X;
        assert!((mirror_point(Vec3::new(7.0, 1.0, 2.0), o, n) - Vec3::new(3.0, 1.0, 2.0)).length() < 1e-5);
        assert!((mirror_dir(Vec3::new(1.0, 1.0, 0.0), n) - Vec3::new(-1.0, 1.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn dashed_line_produces_gaps_and_stays_on_the_line() {
        let mut out = Vec::new();
        dashed(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 0.5, [1.0; 3], &mut out);
        assert!(out.len() >= 5, "expected several dashes, got {}", out.len());
        let covered: f32 = out.iter().map(|(a, b, _)| (Vec3::from(*b) - Vec3::from(*a)).length()).sum();
        assert!(covered < 10.0 * 0.75, "dashes should not cover the whole line: {covered}");
        for (a, b, _) in &out {
            assert!(a[1].abs() < 1e-4 && b[1].abs() < 1e-4);
            assert!(a[0] >= -1e-4 && b[0] <= 10.0 + 1e-4);
        }
    }

    // --- #cad-mate: face-to-face coincident -------------------------------------------------

    #[test]
    fn face_normals_follow_box_rotation() {
        // 90 deg about Z: the box's +X face must report world +Y.
        let m = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let o = Obb::from_local(&m, Vec3::splat(-1.0), Vec3::splat(1.0));
        let n = o.face_normal(1);
        assert!((n - Vec3::Y).length() < 1e-5, "rotated +X face normal = {n:?}");
        assert!((o.face_normal(0) + Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn coincident_plane_only_slides_along_the_normal() {
        // A's face is 3 units below B's plane and offset sideways; Plane mate must close the
        // 3-unit gap and NOT touch the sideways offset.
        let a_c = Vec3::new(5.0, 2.0, 0.0);
        let b_c = Vec3::new(0.0, 0.0, 0.0);
        let d = face_coincident_delta(a_c, b_c, Vec3::Z, FaceMate::Plane);
        assert!((d - Vec3::new(0.0, 0.0, 0.0)).length() < 1e-6, "no gap along Z -> no move, got {d:?}");
        let a_c2 = Vec3::new(5.0, 2.0, -3.0);
        let d2 = face_coincident_delta(a_c2, b_c, Vec3::Z, FaceMate::Plane);
        assert!((d2 - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-5, "expected +3 Z, got {d2:?}");
        // the moved face centre now lies IN B's plane
        assert!(((a_c2 + d2) - b_c).dot(Vec3::Z).abs() < 1e-5);
    }

    #[test]
    fn coincident_centered_puts_the_face_centres_together() {
        let a_c = Vec3::new(5.0, 2.0, -3.0);
        let b_c = Vec3::new(-1.0, 4.0, 7.0);
        let d = face_coincident_delta(a_c, b_c, Vec3::Z, FaceMate::Centered);
        assert!(((a_c + d) - b_c).length() < 1e-5);
    }

    #[test]
    fn coincident_is_inert_on_a_degenerate_normal() {
        let d = face_coincident_delta(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, Vec3::ZERO, FaceMate::Plane);
        assert_eq!(d, Vec3::ZERO, "a zero-extent box axis must not fling the selection");
    }

    #[test]
    fn mating_two_real_boxes_makes_the_faces_flush() {
        // Target box sits at the origin; the mover is 10 units up. Mate the mover's -Z face
        // (index 4) to the target's +Z face (index 5): they must end up touching.
        let target = Obb::from_aabb(Vec3::new(-2.0, -2.0, -1.0), Vec3::new(2.0, 2.0, 1.0));
        let mover = Obb::from_aabb(Vec3::new(-1.0, -1.0, 9.0), Vec3::new(1.0, 1.0, 11.0));
        let d = face_coincident_delta(mover.face_center(4), target.face_center(5), target.face_normal(5), FaceMate::Plane);
        let moved_face_z = mover.face_center(4).z + d.z;
        assert!((moved_face_z - target.face_center(5).z).abs() < 1e-5, "faces not flush: {moved_face_z}");
    }

    // --- #cad-shapes: circles and squares ---------------------------------------------------

    #[test]
    fn circle_is_closed_and_on_radius() {
        let c = Vec3::new(3.0, -1.0, 2.0);
        let g = circle_guides(c, Vec3::Z, 5.0, 16, [1.0; 3]);
        assert_eq!(g.len(), 16);
        for seg in &g {
            assert!(((seg.a.pos - c).length() - 5.0).abs() < 1e-4, "vertex off-radius");
            assert!((seg.a.pos.z - c.z).abs() < 1e-5, "vertex left the plane");
        }
        // closed ring: each segment's end is the next one's start, and the last closes it
        for i in 0..g.len() {
            let nxt = &g[(i + 1) % g.len()];
            assert!((g[i].b.pos - nxt.a.pos).length() < 1e-4, "ring broken at {i}");
        }
    }

    #[test]
    fn circle_segments_are_clamped() {
        assert_eq!(circle_guides(Vec3::ZERO, Vec3::Z, 1.0, 0, [1.0; 3]).len(), 3);
        assert_eq!(circle_guides(Vec3::ZERO, Vec3::Z, 1.0, 9999, [1.0; 3]).len(), 256);
    }

    #[test]
    fn circle_respects_a_tilted_plane() {
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        for seg in circle_guides(Vec3::ZERO, n, 2.0, 12, [1.0; 3]) {
            assert!(seg.a.pos.dot(n).abs() < 1e-4, "vertex off the requested plane");
        }
    }

    #[test]
    fn square_has_four_equal_sides_and_closes() {
        let c = Vec3::new(1.0, 2.0, 3.0);
        let g = square_guides(c, Vec3::Z, 2.0, 0.0, [1.0; 3]);
        assert_eq!(g.len(), 4);
        let l0 = g[0].length();
        for seg in &g {
            assert!((seg.length() - l0).abs() < 1e-4, "sides unequal");
            assert!((seg.a.pos.z - c.z).abs() < 1e-5);
        }
        assert!((l0 - 4.0).abs() < 1e-4, "half=2 -> side 4, got {l0}");
        for i in 0..4 {
            assert!((g[i].b.pos - g[(i + 1) % 4].a.pos).length() < 1e-4);
        }
    }

    #[test]
    fn square_rotation_turns_it_in_plane_without_resizing() {
        let a = square_guides(Vec3::ZERO, Vec3::Z, 1.5, 0.0, [1.0; 3]);
        let b = square_guides(Vec3::ZERO, Vec3::Z, 1.5, std::f32::consts::FRAC_PI_4, [1.0; 3]);
        assert!((a[0].length() - b[0].length()).abs() < 1e-4, "rotation changed the size");
        // a 45 deg turn must actually move the corners
        assert!((a[0].a.pos - b[0].a.pos).length() > 0.1, "rotation did nothing");
    }

    #[test]
    fn shape_guides_feed_the_snap_system() {
        // A square's corners must show up as snap targets, which is the whole point of
        // drawing construction shapes.
        let g = square_guides(Vec3::ZERO, Vec3::Z, 1.0, 0.0, [1.0; 3]);
        let t = snap_targets(&g, 0.05);
        assert!(!t.is_empty(), "square produced no snap targets");
        let corner = Vec3::new(1.0, 1.0, 0.0);
        assert!(t.iter().any(|x| (x.pos - corner).length() < 1e-3), "corner is not snappable");
    }

    // --- #cad-shape-tool: shapes as movable, resizable things ------------------------------

    #[test]
    fn shape_outline_matches_the_free_functions() {
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::new(1.0, 2.0, 3.0), Vec3::Z);
        sh.radius = 4.0;
        sh.segments = 16;
        assert_eq!(sh.guides().len(), 16);
        for g in sh.guides() {
            assert!(((g.a.pos - sh.center).length() - 4.0).abs() < 1e-3);
        }
    }

    #[test]
    fn moving_a_shape_moves_its_whole_outline() {
        let mut sh = Shape::new(ShapeKind::Square, Vec3::ZERO, Vec3::Z);
        sh.radius = 2.0;
        let before: Vec<Vec3> = sh.guides().iter().map(|g| g.a.pos).collect();
        let d = Vec3::new(5.0, -3.0, 1.0);
        sh.center += d;
        for (i, g) in sh.guides().iter().enumerate() {
            assert!((g.a.pos - (before[i] + d)).length() < 1e-4, "outline did not follow the centre");
        }
    }

    #[test]
    fn resizing_a_shape_keeps_it_centred() {
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::new(2.0, 2.0, 0.0), Vec3::Z);
        sh.radius = 1.0;
        let c0 = sh.center;
        sh.radius = 7.5;
        for g in sh.guides() {
            assert!(((g.a.pos - c0).length() - 7.5).abs() < 1e-3);
        }
        assert_eq!(sh.center, c0);
    }

    #[test]
    fn outline_distance_is_zero_on_the_edge_and_grows_inward() {
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::ZERO, Vec3::Z);
        sh.radius = 5.0;
        sh.segments = 64;
        let on_edge = Vec3::new(5.0, 0.0, 0.0);
        assert!(sh.dist_to_outline(on_edge) < 0.02, "edge point should be on the outline");
        // the centre is one radius away from the ring
        assert!((sh.dist_to_outline(Vec3::ZERO) - 5.0).abs() < 0.05);
    }

    #[test]
    fn ray_plane_hits_and_rejects_parallel_or_behind() {
        let hit = ray_plane(Vec3::new(0.0, 0.0, 10.0), -Vec3::Z, Vec3::ZERO, Vec3::Z);
        assert!(hit.is_some());
        assert!((hit.unwrap() - Vec3::ZERO).length() < 1e-5);
        // parallel
        assert!(ray_plane(Vec3::new(0.0, 0.0, 10.0), Vec3::X, Vec3::ZERO, Vec3::Z).is_none());
        // plane behind the ray
        assert!(ray_plane(Vec3::new(0.0, 0.0, 10.0), Vec3::Z, Vec3::ZERO, Vec3::Z).is_none());
    }

    #[test]
    fn drag_radius_is_the_in_plane_distance() {
        // Drawing: press at the centre, drag out; the radius is the distance from the centre to
        // where the cursor ray meets the shape's plane.
        let center = Vec3::new(1.0, 1.0, 0.0);
        let p = ray_plane(Vec3::new(4.0, 1.0, 9.0), -Vec3::Z, center, Vec3::Z).unwrap();
        assert!(((p - center).length() - 3.0).abs() < 1e-4, "expected radius 3, got {}", (p - center).length());
    }

    #[test]
    fn point_segment_distance_clamps_to_the_ends() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 0.0, 0.0);
        assert!((dist_point_segment(Vec3::new(5.0, 3.0, 0.0), a, b) - 3.0).abs() < 1e-5);
        // beyond the end -> distance to the endpoint, not the infinite line
        assert!((dist_point_segment(Vec3::new(20.0, 0.0, 0.0), a, b) - 10.0).abs() < 1e-5);
    }

    // --- #cad-array: duplicate along a shape's edge -----------------------------------------

    #[test]
    fn circle_perimeter_points_are_evenly_spaced_on_the_ring() {
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::new(1.0, 2.0, 3.0), Vec3::Z);
        sh.radius = 6.0;
        let pts = sh.perimeter_points(8);
        assert_eq!(pts.len(), 8);
        for (p, out) in &pts {
            assert!(((*p - sh.center).length() - 6.0).abs() < 1e-3, "point off the ring");
            assert!((p.z - sh.center.z).abs() < 1e-4, "point left the plane");
            // outward direction points from the centre to the point
            assert!((*out - (*p - sh.center).normalize()).length() < 1e-3);
        }
        // equal spacing: every consecutive gap is the same
        let gap = (pts[1].0 - pts[0].0).length();
        for i in 0..pts.len() {
            let g = (pts[(i + 1) % pts.len()].0 - pts[i].0).length();
            assert!((g - gap).abs() < 1e-3, "uneven spacing at {i}");
        }
    }

    #[test]
    fn square_perimeter_spaces_by_arc_length_not_per_side() {
        let mut sh = Shape::new(ShapeKind::Square, Vec3::ZERO, Vec3::Z);
        sh.radius = 2.0; // side 4, perimeter 16
        let pts = sh.perimeter_points(8);
        assert_eq!(pts.len(), 8);
        // 8 points over a perimeter of 16 -> one every 2 units along the path
        for i in 0..pts.len() {
            let g = (pts[(i + 1) % pts.len()].0 - pts[i].0).length();
            assert!(g > 0.1, "duplicate point at {i}");
        }
        // every point lies ON the square: max-norm in the plane equals the half-side
        for (p, _) in &pts {
            let m = p.x.abs().max(p.y.abs());
            assert!((m - 2.0).abs() < 1e-3, "point not on the square edge: {p:?}");
        }
    }

    #[test]
    fn perimeter_count_is_respected_and_never_zero() {
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::ZERO, Vec3::Z);
        sh.radius = 1.0;
        assert_eq!(sh.perimeter_points(1).len(), 1);
        assert_eq!(sh.perimeter_points(0).len(), 1, "0 copies must not panic or divide by zero");
        assert_eq!(sh.perimeter_points(33).len(), 33);
    }

    #[test]
    fn perimeter_follows_a_tilted_plane() {
        let n = Vec3::new(0.0, 1.0, 1.0).normalize();
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::ZERO, n);
        sh.radius = 3.0;
        for (p, _) in sh.perimeter_points(12) {
            assert!(p.dot(n).abs() < 1e-3, "array point left the shape plane");
        }
    }

    #[test]
    fn arrayed_copies_land_exactly_on_the_perimeter_points() {
        for kind in [ShapeKind::Circle, ShapeKind::Square] {
            let mut sh = Shape::new(kind, Vec3::new(4.0, -2.0, 1.0), Vec3::Z);
            sh.radius = 6.0;
            let pts = sh.perimeter_points(12);
            let (first, first_out) = pts[0];
            for (k, (target, out)) in pts.iter().enumerate() {
                let (q, d) = array_copy_transform(sh.center, sh.normal, first, first_out, *target, *out, true);
                // a copy of the original (which sits at `first`) must land ON the k-th point
                let landed = sh.center + q * (first - sh.center) + d;
                assert!((landed - *target).length() < 1e-3,
                    "{kind:?} copy {k} landed {landed:?}, wanted {target:?}");
            }
        }
    }

    #[test]
    fn circle_array_needs_no_corrective_nudge() {
        // On a circle the rotation alone maps point 0 onto point k, so delta must be ~0 --
        // that is what keeps a rotated ring perfectly concentric.
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::ZERO, Vec3::Z);
        sh.radius = 5.0;
        let pts = sh.perimeter_points(8);
        for (target, out) in pts.iter().skip(1) {
            let (_, d) = array_copy_transform(sh.center, sh.normal, pts[0].0, pts[0].1, *target, *out, true);
            assert!(d.length() < 1e-3, "circle array should not need a nudge, got {d:?}");
        }
    }

    #[test]
    fn array_without_rotation_keeps_the_original_facing() {
        let mut sh = Shape::new(ShapeKind::Circle, Vec3::ZERO, Vec3::Z);
        sh.radius = 3.0;
        let pts = sh.perimeter_points(4);
        let (q, d) = array_copy_transform(sh.center, sh.normal, pts[0].0, pts[0].1, pts[2].0, pts[2].1, false);
        assert!((q.to_array()[3].abs() - 1.0).abs() < 1e-5, "expected identity rotation");
        // with no rotation the copy is a pure translation onto the target
        assert!(((pts[0].0 + d) - pts[2].0).length() < 1e-4);
    }

    // --- #cad-mate-rot: a mate that also turns the part --------------------------------------

    #[test]
    fn mate_rotates_face_a_to_oppose_face_b() {
        // A faces +X, target faces +Z: after the mate A must face -Z (into B).
        let (q, _) = face_mate_transform(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 0.0, 5.0), Vec3::Z, FaceMate::Plane, true);
        let a_after = q * Vec3::X;
        assert!((a_after - (-Vec3::Z)).length() < 1e-5, "A ended facing {a_after:?}, wanted -Z");
    }

    #[test]
    fn mate_handles_faces_that_already_point_the_same_way() {
        // Both +Z: needs a 180 deg flip, the degenerate case for a shortest-arc rotation.
        let (q, _) = face_mate_transform(Vec3::ZERO, Vec3::Z, Vec3::new(0.0, 0.0, 4.0), Vec3::Z, FaceMate::Plane, true);
        let a_after = q * Vec3::Z;
        assert!((a_after - (-Vec3::Z)).length() < 1e-4, "flip failed, got {a_after:?}");
    }

    #[test]
    fn mate_leaves_already_opposed_faces_alone() {
        let (q, _) = face_mate_transform(Vec3::ZERO, -Vec3::Z, Vec3::new(0.0, 0.0, 4.0), Vec3::Z, FaceMate::Plane, true);
        assert!((q * Vec3::X - Vec3::X).length() < 1e-5, "should be an identity rotation");
    }

    #[test]
    fn mate_pivots_about_face_a_so_that_face_lands_on_the_target_plane() {
        let a_c = Vec3::new(3.0, 0.0, 0.0);
        let b_c = Vec3::new(0.0, 0.0, 10.0);
        let (q, d) = face_mate_transform(a_c, Vec3::X, b_c, Vec3::Z, FaceMate::Plane, true);
        // the pivot IS face A's centre, so it only moves by the translation
        let landed = a_c + q * (a_c - a_c) + d;
        assert!(((landed - b_c).dot(Vec3::Z)).abs() < 1e-4, "face A not on B's plane: {landed:?}");
    }

    #[test]
    fn mate_without_rotation_is_the_old_slide() {
        let (q, d) = face_mate_transform(Vec3::ZERO, Vec3::X, Vec3::new(0.0, 0.0, 5.0), Vec3::Z, FaceMate::Plane, false);
        assert!((q * Vec3::X - Vec3::X).length() < 1e-6);
        assert!((d - Vec3::new(0.0, 0.0, 5.0)).length() < 1e-5);
    }

    #[test]
    fn mate_centered_also_brings_the_centres_together() {
        let a_c = Vec3::new(7.0, -2.0, 1.0);
        let b_c = Vec3::new(0.0, 3.0, 9.0);
        let (_, d) = face_mate_transform(a_c, Vec3::Y, b_c, Vec3::Z, FaceMate::Centered, true);
        assert!(((a_c + d) - b_c).length() < 1e-5);
    }
}
