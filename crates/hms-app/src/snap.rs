//! #snap-array: the PURE geometry behind the Ctrl face magnet, the "clicks into place" edge
//! alignment, and line arrays (spacing / overlap).
//!
//! Everything here is a function of geometry only — no `App`, no egui, no renderer — so the
//! interactive editor, the headless script host and the unit tests all run the same maths.
//! `main.rs` only wires input and status text to it.
//!
//! The model:
//!
//! * A [`Face`] is a FINITE planar face: a plane (centre + outward normal) plus a rectangle in
//!   that plane (`u`/`v` axes with half-extents). Both the six oriented box faces of a piece
//!   ([`obb_faces`]) and the piece's real large planar faces ([`dominant_planes`], e.g. a Forge
//!   ramp's sloped deck and its tall end face) are Faces, so the snap solver treats them alike.
//!   That is what makes an angled piece snap: an AABB has no sloped face to offer.
//! * [`solve`] picks the best (source face, target face) pair: it travels each source face along
//!   its own outward normal (or along the locked axis), measures the gap to each target plane,
//!   and scores by gap, how anti-parallel the normals are, and how much the two faces overlap.
//!   The result is a TRANSLATION only — a move never turns the piece.
//! * [`edge_align`] then slides the piece WITHIN the contact plane so the nearest pair of
//!   parallel face edges (or the two face centres) line up. That is the "it clicks into place"
//!   half: a ramp's high end lands flush with the platform edge and its deck ends level with the
//!   platform top instead of a few centimetres off.
//! * [`line_positions`] / [`axis_positions`] fill a line with copies at a chosen STEP (so the
//!   copies may overlap) or a chosen COUNT over a distance.

use std::collections::HashMap;

use glam::{Quat, Vec3};

use crate::construct::Obb;
use crate::objscene::ObjectScene;

/// Default magnet range (wu): the largest face-to-face gap that still snaps. Matches the old
/// AABB magnet so the feel of a plain Ctrl drag is unchanged.
pub const DEFAULT_RANGE: f32 = 3.0;

/// Hard cap on how many copies one array command may stamp (a mistyped step must not try to
/// fill a map with 100 000 blocks).
pub const MAX_COPIES: usize = 512;

// ===================================================================== faces

/// Where a [`Face`] came from — only for the status line, never for the maths.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaceKind {
    /// One of the six oriented bounding-box faces, 0..6 = -X, +X, -Y, +Y, -Z, +Z in the piece's
    /// OWN axes (so a rotated block reports its real face).
    Box(u8),
    /// A large planar face of the decoded mesh (index into the piece's dominant-plane list).
    Mesh(u8),
    /// Level geometry (BSP) hit by a probe ray — an effectively infinite plane.
    Bsp,
}

/// A finite planar face: the plane (`c`, `n`) plus its rectangle (`u`/`v`, `hu`/`hv`).
#[derive(Clone, Copy, Debug)]
pub struct Face {
    /// Face centre.
    pub c: Vec3,
    /// Unit OUTWARD normal.
    pub n: Vec3,
    /// In-plane orthonormal axes.
    pub u: Vec3,
    pub v: Vec3,
    /// Half-extents along `u` / `v`.
    pub hu: f32,
    pub hv: f32,
    /// Face area (`4·hu·hv`, or the summed triangle area for a mesh plane).
    pub area: f32,
    pub kind: FaceKind,
    /// The object this face belongs to; `u32::MAX` for level geometry / a group box.
    pub datum: u32,
}

/// `u32::MAX`: a face that belongs to no single object (level geometry, or a multi-object
/// selection's group box).
pub const NO_DATUM: u32 = u32::MAX;

impl Face {
    /// Move the face by `d` (a pure translation, so the plane's normal and rectangle are kept).
    pub fn translated(&self, d: Vec3) -> Face {
        Face { c: self.c + d, ..*self }
    }

    /// Half-extent of this face's rectangle measured along an arbitrary in-plane direction `a`
    /// (the rectangle's support function).
    pub fn support(&self, a: Vec3) -> f32 {
        self.u.dot(a).abs() * self.hu + self.v.dot(a).abs() * self.hv
    }

    /// Signed interval `[min, max]` this face spans along `a`, measured from `origin`.
    pub fn span(&self, origin: Vec3, a: Vec3) -> (f32, f32) {
        let mid = (self.c - origin).dot(a);
        let h = self.support(a);
        (mid - h, mid + h)
    }

    /// A short human label for the face ("+Z", "slope", "map surface").
    pub fn label(&self) -> &'static str {
        match self.kind {
            FaceKind::Box(i) => ["-X", "+X", "-Y", "+Y", "-Z", "+Z"][(i as usize).min(5)],
            FaceKind::Bsp => "map surface",
            FaceKind::Mesh(_) => {
                // Name a mesh plane by its inclination, which is what the user sees.
                let z = self.n.z;
                if z > 0.96 {
                    "deck"
                } else if z > 0.25 {
                    "slope"
                } else if z < -0.96 {
                    "underside"
                } else if z < -0.25 {
                    "underslope"
                } else {
                    "end"
                }
            }
        }
    }
}

/// An orthonormal pair spanning the plane whose normal is `n`.
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let r = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = n.cross(r).normalize_or_zero();
    let u = if u.length_squared() < 0.5 { Vec3::Y } else { u };
    (u, n.cross(u).normalize_or_zero())
}

/// The six ORIENTED faces of a box, in the box's own axes. Degenerate faces (a zero-extent
/// axis, e.g. a flat plate's rim) are dropped — they have no area to mate against.
pub fn obb_faces(o: &Obb, datum: u32) -> Vec<Face> {
    let mut out = Vec::with_capacity(6);
    for i in 0..6u8 {
        // The two half-axes that are NOT this face's normal span the face, in right-handed
        // order (so `a x b` is the +axis, which is the fallback normal for a FLAT piece whose
        // own box axis has no length to take a direction from).
        let (a, b) = match i {
            0 | 1 => (o.hy, o.hz),
            2 | 3 => (o.hz, o.hx),
            _ => (o.hx, o.hy),
        };
        let (hu, hv) = (a.length(), b.length());
        if hu * hv < 1e-4 {
            continue;
        }
        let mut n = o.face_normal(i);
        if n.length_squared() < 0.5 {
            let s = if i % 2 == 0 { -1.0 } else { 1.0 };
            n = a.cross(b).normalize_or_zero() * s;
        }
        if n.length_squared() < 0.5 {
            continue;
        }
        out.push(Face {
            c: o.face_center(i),
            n,
            u: a.normalize_or_zero(),
            v: b.normalize_or_zero(),
            hu,
            hv,
            area: 4.0 * hu * hv,
            kind: FaceKind::Box(i),
            datum,
        });
    }
    out
}

// =========================================================== dominant planes

/// Cluster `tris` into the mesh's LARGE planar faces.
///
/// A Forge ramp's sloped deck is thousands of coplanar triangles; its bounding box has no such
/// face, which is exactly why an AABB magnet cannot put a ramp against a platform. Triangles are
/// grouped by (normal, plane offset) and every group holding at least `min_area_frac` of the
/// model's total area becomes a [`Face`] whose rectangle is the group's own extent.
///
/// `tris` may be in any space; the faces come back in that same space.
pub fn dominant_planes(tris: &[[Vec3; 3]], min_area_frac: f32, max_planes: usize) -> Vec<Face> {
    /// One accumulating plane group.
    struct Cluster {
        n: Vec3, // area-weighted normal sum
        d: f32,  // representative plane offset
        area: f32,
        cen: Vec3, // area-weighted centroid sum
        tris: Vec<u32>,
    }
    const COS_TOL: f32 = 0.985; // ~10 deg
    const D_TOL: f32 = 0.08; // wu
    const MAX_CLUSTERS: usize = 2048;

    let mut cl: Vec<Cluster> = Vec::new();
    let mut total = 0.0f32;
    for (ti, t) in tris.iter().enumerate() {
        let e1 = t[1] - t[0];
        let e2 = t[2] - t[0];
        let cr = e1.cross(e2);
        let a2 = cr.length();
        if a2 < 1e-9 {
            continue;
        }
        let area = 0.5 * a2;
        let n = cr / a2;
        let cen = (t[0] + t[1] + t[2]) / 3.0;
        let d = n.dot(cen);
        total += area;
        let mut hit = None;
        for (i, c) in cl.iter().enumerate() {
            let cn = c.n.normalize_or_zero();
            if cn.dot(n) > COS_TOL && (c.d - d).abs() < D_TOL {
                hit = Some(i);
                break;
            }
        }
        match hit {
            Some(i) => {
                cl[i].n += n * area;
                cl[i].area += area;
                cl[i].cen += cen * area;
                cl[i].tris.push(ti as u32);
            }
            None if cl.len() < MAX_CLUSTERS => {
                cl.push(Cluster { n: n * area, d, area, cen: cen * area, tris: vec![ti as u32] });
            }
            None => {}
        }
    }
    if total <= 0.0 {
        return Vec::new();
    }
    cl.retain(|c| c.area >= min_area_frac * total);
    cl.sort_by(|a, b| b.area.total_cmp(&a.area));
    cl.truncate(max_planes);
    cl.iter()
        .enumerate()
        .filter_map(|(idx, c)| {
            let n = c.n.normalize_or_zero();
            if n.length_squared() < 0.5 {
                return None;
            }
            let cen = c.cen / c.area;
            let (u, v) = basis(n);
            // Second pass: the group's own extent within its plane.
            let (mut u0, mut u1, mut v0, mut v1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for &ti in &c.tris {
                for p in tris[ti as usize] {
                    let r = p - cen;
                    let (du, dv) = (r.dot(u), r.dot(v));
                    u0 = u0.min(du);
                    u1 = u1.max(du);
                    v0 = v0.min(dv);
                    v1 = v1.max(dv);
                }
            }
            if !(u1 > u0 && v1 > v0) {
                return None;
            }
            Some(Face {
                // Centre the rectangle on the group, not on its area centroid.
                c: cen + u * (0.5 * (u0 + u1)) + v * (0.5 * (v0 + v1)),
                n,
                u,
                v,
                hu: 0.5 * (u1 - u0),
                hv: 0.5 * (v1 - v0),
                area: c.area,
                kind: FaceKind::Mesh(idx.min(255) as u8),
                datum: NO_DATUM,
            })
        })
        .collect()
}

/// Rebuild triangles from the line-list a wireframe query returns. `selection_wireframe` emits
/// `a,b, b,c, c,a` per triangle, so every six vertices are one triangle.
pub fn tris_from_line_list(lines: &[[f32; 3]]) -> Vec<[Vec3; 3]> {
    lines
        .chunks_exact(6)
        .map(|w| [Vec3::from(w[0]), Vec3::from(w[1]), Vec3::from(w[3])])
        .collect()
}

// ============================================================ model-local cache

/// An object's own orthonormal frame (box centre + unit box axes). Used to store a model's
/// dominant planes POSE-INDEPENDENTLY, so the expensive plane extraction happens once per render
/// model instead of once per frame.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub o: Vec3,
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
}

impl Frame {
    pub fn of_obb(o: &Obb) -> Frame {
        let x = o.hx.normalize_or_zero();
        let y = o.hy.normalize_or_zero();
        let z = o.hz.normalize_or_zero();
        // A zero-extent axis still needs a direction, or the frame is not invertible.
        let x = if x.length_squared() < 0.5 { y.cross(z).normalize_or_zero() } else { x };
        let y = if y.length_squared() < 0.5 { z.cross(x).normalize_or_zero() } else { y };
        let z = if z.length_squared() < 0.5 { x.cross(y).normalize_or_zero() } else { z };
        Frame { o: o.c, x, y, z }
    }
    pub fn to_local(&self, p: Vec3) -> Vec3 {
        let r = p - self.o;
        Vec3::new(r.dot(self.x), r.dot(self.y), r.dot(self.z))
    }
    pub fn dir_to_local(&self, d: Vec3) -> Vec3 {
        Vec3::new(d.dot(self.x), d.dot(self.y), d.dot(self.z))
    }
    pub fn to_world(&self, p: Vec3) -> Vec3 {
        self.o + self.x * p.x + self.y * p.y + self.z * p.z
    }
    pub fn dir_to_world(&self, d: Vec3) -> Vec3 {
        self.x * d.x + self.y * d.y + self.z * d.z
    }
}

/// A [`Face`] stored in a model's own frame, so it survives the piece moving and turning.
#[derive(Clone, Copy, Debug)]
pub struct LocalFace {
    pub c: Vec3,
    pub n: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub hu: f32,
    pub hv: f32,
    pub area: f32,
    pub idx: u8,
}

impl LocalFace {
    pub fn of(f: &Face, fr: &Frame, idx: u8) -> LocalFace {
        LocalFace {
            c: fr.to_local(f.c),
            n: fr.dir_to_local(f.n),
            u: fr.dir_to_local(f.u),
            v: fr.dir_to_local(f.v),
            hu: f.hu,
            hv: f.hv,
            area: f.area,
            idx,
        }
    }
    pub fn to_world(&self, fr: &Frame, datum: u32) -> Face {
        Face {
            c: fr.to_world(self.c),
            n: fr.dir_to_world(self.n).normalize_or_zero(),
            u: fr.dir_to_world(self.u).normalize_or_zero(),
            v: fr.dir_to_world(self.v).normalize_or_zero(),
            hu: self.hu,
            hv: self.hv,
            area: self.area,
            kind: FaceKind::Mesh(self.idx),
            datum,
        }
    }
}

/// Per-render-model dominant-plane cache. Keyed by render-model tag AND the piece's box extents
/// (quantised), so a scaled copy of the same block gets its own entry instead of inheriting
/// planes at the wrong offsets. An empty entry is cached too ("this model has no big flat face"),
/// which is what keeps a drag from re-walking a 20 000-triangle mesh every frame.
#[derive(Default)]
pub struct PlaneCache {
    map: HashMap<u64, Vec<LocalFace>>,
}

impl PlaneCache {
    /// Render-model tag + the piece's box extents quantised to 1/64 wu, packed into one key.
    pub fn key(mode_tag: u32, o: &Obb) -> u64 {
        let q = |v: f32| ((v * 64.0).round().clamp(0.0, 1_048_575.0) as u64) & 0xFFFFF;
        let ext = (q(o.hx.length()) << 40) | (q(o.hy.length()) << 20) | q(o.hz.length());
        // Mixed rather than concatenated: a tag is 32 bits and the extents already fill 60.
        ext ^ (mode_tag as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
    pub fn get(&self, key: u64) -> Option<&Vec<LocalFace>> {
        self.map.get(&key)
    }
    pub fn insert(&mut self, key: u64, v: Vec<LocalFace>) {
        self.map.insert(key, v);
    }
    /// Forget every model (a new map reuses tag ids, so a load must not inherit planes).
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

// ==================================================================== solving

/// Tolerances the snap solver works to.
#[derive(Clone, Copy, Debug)]
pub struct SnapCfg {
    /// Largest face-to-face gap that still snaps (wu).
    pub range: f32,
    /// How far the snap may pull a piece BACK out of a surface it already overlaps (wu).
    pub pull_back: f32,
    /// Required lateral overlap of the two faces, as a fraction of the smaller one.
    pub min_overlap: f32,
    /// Faces must oppose at least this much (`-n_src . n_tgt`).
    pub min_anti: f32,
    /// Edge-alignment tolerance as a fraction of the source face's LARGER dimension (so the
    /// tolerance scales with the piece, and a short-but-wide face like a ramp's end still clicks
    /// vertically by a sensible amount instead of only by its own small height).
    pub lateral_frac: f32,
    /// Lateral shift always permitted, even when the face snap itself moved nothing (wu).
    /// Above this, the lateral shift may never exceed the face snap's own travel.
    pub lateral_floor: f32,
}

impl Default for SnapCfg {
    fn default() -> Self {
        SnapCfg {
            range: DEFAULT_RANGE,
            pull_back: 0.75,
            min_overlap: 0.10,
            min_anti: 0.35,
            lateral_frac: 0.18,
            lateral_floor: 0.35,
        }
    }
}

/// One scored (source face -> target face) pairing.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// Unit travel direction.
    pub dir: Vec3,
    /// Distance along `dir` that brings the faces coplanar (negative = pull back out).
    pub t: f32,
    /// Lower is better: gap, then how squarely the faces oppose, then how much they overlap.
    pub score: f32,
    pub src: Face,
    pub tgt: Face,
}

/// Distance along `dir` that puts `src`'s plane onto `tgt`'s plane. `None` when the travel is
/// (nearly) parallel to the target plane, where no finite move makes them coincide.
pub fn plane_contact(src: &Face, tgt: &Face, dir: Vec3) -> Option<f32> {
    let den = dir.dot(tgt.n);
    if den.abs() < 0.15 {
        return None;
    }
    Some((tgt.c - src.c).dot(tgt.n) / den)
}

/// Lateral overlap of two faces once `src` has been moved by `dir·t`, as a fraction of the
/// SMALLER face (so mating a big platform onto a small block is not penalised for the size
/// difference). Measured in the target face's own axes.
pub fn overlap_fraction(src: &Face, tgt: &Face, shift: Vec3) -> f32 {
    let s = src.translated(shift);
    let mut acc = 1.0f32;
    let mut s_area = 1.0f32;
    let mut t_area = 1.0f32;
    for a in [tgt.u, tgt.v] {
        let (s0, s1) = s.span(tgt.c, a);
        let (t0, t1) = tgt.span(tgt.c, a);
        let o = (s1.min(t1) - s0.max(t0)).max(0.0);
        acc *= o;
        s_area *= s1 - s0;
        t_area *= t1 - t0;
    }
    let den = s_area.min(t_area);
    if den <= 1e-6 {
        return 0.0;
    }
    (acc / den).clamp(0.0, 1.0)
}

/// Score one pairing (lower is better), or `None` when it is not a legal snap.
fn score(src: &Face, tgt: &Face, dir: Vec3, cfg: &SnapCfg) -> Option<Candidate> {
    // The face must point roughly the way we travel: we push a face FORWARD into a surface.
    if dir.dot(src.n) < 0.30 {
        return None;
    }
    let anti = (-src.n.dot(tgt.n)).clamp(0.0, 1.0);
    if anti < cfg.min_anti {
        return None;
    }
    let t = plane_contact(src, tgt, dir)?;
    if t > cfg.range || t < -cfg.pull_back {
        return None;
    }
    let overlap = overlap_fraction(src, tgt, dir * t);
    if overlap < cfg.min_overlap {
        return None;
    }
    // Nearest wins; a squarer mate and a fuller overlap break ties; a DOWNWARD-facing source
    // face gets a small edge so a plain Ctrl drag still feels like dropping a piece on a floor.
    let down = if src.n.z < -0.7 { 0.12 } else { 0.0 };
    let s = t.abs() / cfg.range.max(1e-3) + 0.60 * (1.0 - anti) + 0.30 * (1.0 - overlap) - down;
    Some(Candidate { dir, t, score: s, src: *src, tgt: *tgt })
}

/// Best pairing over every source face x travel direction x target face.
///
/// `dirs` empty = each source face travels along its OWN outward normal (free move: "push this
/// face into whatever is in front of it"). Otherwise every source face is tried against each of
/// the given directions (an axis-locked move passes `[+axis, -axis]`), so the snap can only
/// translate along the axis the user locked.
pub fn best_candidate(srcs: &[Face], tgts: &[Face], dirs: &[Vec3], cfg: &SnapCfg) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for s in srcs {
        let own = [s.n];
        let list: &[Vec3] = if dirs.is_empty() { &own } else { dirs };
        for &d in list {
            let d = d.normalize_or_zero();
            if d.length_squared() < 0.5 {
                continue;
            }
            for t in tgts {
                if let Some(c) = score(s, t, d, cfg) {
                    if best.map_or(true, |b| c.score < b.score) {
                        best = Some(c);
                    }
                }
            }
        }
    }
    best
}

/// What the lateral ("clicks into place") pass did.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeAlign {
    pub shift: Vec3,
    /// Per-axis shift actually applied, in the target face's `u` / `v` axes.
    pub du: f32,
    pub dv: f32,
}

impl EdgeAlign {
    pub fn is_zero(&self) -> bool {
        self.du == 0.0 && self.dv == 0.0
    }
}

/// Smallest shift along one in-plane axis that lines `src` up with `tgt` — either an edge onto
/// an edge (4 pairings) or the two centres. `None` when nothing is within `tol`.
fn axis_align(src: &Face, tgt: &Face, a: Vec3, tol: f32) -> Option<f32> {
    let (s0, s1) = src.span(tgt.c, a);
    let (t0, t1) = tgt.span(tgt.c, a);
    let cands = [
        t0 - s0, // left edges flush
        t1 - s1, // right edges flush
        t0 - s1, // src sits just outside the low edge
        t1 - s0, // ... the high edge
        0.5 * (t0 + t1) - 0.5 * (s0 + s1), // centred
    ];
    cands
        .into_iter()
        // Sub-millimetre "corrections" are noise, not an alignment worth reporting.
        .filter(|d| d.abs() <= tol && d.abs() > 1.0e-3)
        .min_by(|a, b| a.abs().total_cmp(&b.abs()))
}

/// After face contact, slide the piece WITHIN the contact plane so the nearest pair of parallel
/// face edges (or the two centres) line up.
///
/// * `src` is the source face ALREADY moved by the face-contact shift.
/// * Tolerance scales with the piece: `lateral_frac` of the source face's LARGER dimension,
///   clamped to a sane 0.10..1.50 wu, so a small block clicks over centimetres and a big
///   platform over a decimetre or two. The face's larger dimension (not its extent on the axis
///   being corrected) is deliberate: a ramp's end face is wide and short, and "the deck should
///   have landed level" is a correction on its SHORT axis.
/// * The lateral move may never outweigh the face snap: its length is capped at
///   `max(|face travel|, lateral_floor)`. Without the floor a piece that is already touching
///   could never click into line; without the cap a near-miss could slide a piece sideways
///   further than the snap itself moved it.
pub fn edge_align(src: &Face, tgt: &Face, face_travel: f32, cfg: &SnapCfg) -> EdgeAlign {
    let tol = (cfg.lateral_frac * 2.0 * src.hu.max(src.hv)).clamp(0.10, 1.50);
    let mut du = axis_align(src, tgt, tgt.u, tol).unwrap_or(0.0);
    let mut dv = axis_align(src, tgt, tgt.v, tol).unwrap_or(0.0);
    let cap = face_travel.abs().max(cfg.lateral_floor);
    // Over budget: give up the bigger correction first, then both.
    if (du * du + dv * dv).sqrt() > cap {
        if du.abs() >= dv.abs() {
            du = 0.0;
        } else {
            dv = 0.0;
        }
    }
    if (du * du + dv * dv).sqrt() > cap {
        du = 0.0;
        dv = 0.0;
    }
    EdgeAlign { shift: tgt.u * du + tgt.v * dv, du, dv }
}

/// The world axis a direction is closest to ("X" / "Y" / "Z"), for the status line.
pub fn axis_name(d: Vec3) -> &'static str {
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        "X"
    } else if a.y >= a.z {
        "Y"
    } else {
        "Z"
    }
}

/// A solved snap: the translation to apply, and what it mated, for the status line.
#[derive(Clone, Debug)]
pub struct SnapResult {
    /// Total translation = face contact + lateral alignment.
    pub delta: Vec3,
    /// The lateral ("clicks into place") part alone; `delta - lateral` is the face contact.
    pub lateral: Vec3,
    /// Face-to-face gap that was closed (negative = the piece was overlapping and got pulled out).
    pub gap: f32,
    pub src_label: &'static str,
    pub tgt_label: &'static str,
    pub tgt_datum: u32,
    /// World axis the lateral alignment worked on, when it did anything.
    pub edge_axis: Option<&'static str>,
}

impl SnapResult {
    /// "face slope -> 0xD0000012 +Z (gap 0.42), edge aligned X".
    pub fn describe(&self) -> String {
        let who = if self.tgt_datum == NO_DATUM {
            self.tgt_label.to_string()
        } else {
            format!("0x{:08X} {}", self.tgt_datum, self.tgt_label)
        };
        let mut s = format!("face {} to {who} (gap {:+.3} wu)", self.src_label, self.gap);
        if let Some(ax) = self.edge_axis {
            s.push_str(&format!(", edge aligned {ax} by {:.3} wu", self.lateral.length()));
        }
        s
    }
}

/// Finish a chosen candidate: face contact, then the lateral edge alignment.
pub fn finish(c: &Candidate, cfg: &SnapCfg) -> SnapResult {
    let face_shift = c.dir * c.t;
    let moved = c.src.translated(face_shift);
    let ea = edge_align(&moved, &c.tgt, c.t, cfg);
    SnapResult {
        delta: face_shift + ea.shift,
        lateral: ea.shift,
        gap: c.t,
        src_label: c.src.label(),
        tgt_label: c.tgt.label(),
        tgt_datum: c.tgt.datum,
        edge_axis: if ea.is_zero() {
            None
        } else if ea.du.abs() >= ea.dv.abs() {
            Some(axis_name(c.tgt.u))
        } else {
            Some(axis_name(c.tgt.v))
        },
    }
}

/// Solve a snap from ready-made face lists: the whole pipeline (score, pick, edge-align).
pub fn solve_faces(srcs: &[Face], tgts: &[Face], dirs: &[Vec3], cfg: &SnapCfg) -> Option<SnapResult> {
    best_candidate(srcs, tgts, dirs, cfg).map(|c| finish(&c, cfg))
}

// ================================================================= gathering

/// Source faces for a moving set.
///
/// One object offers its oriented box faces PLUS its model's large planar faces (so a ramp's
/// deck and its tall end can mate). A multi-object selection offers only the group box's faces:
/// a group must snap by its OUTER face, never by an inner member's, or the members would be
/// driven into each other.
pub fn source_faces(
    scene: &dyn ObjectScene,
    moving: &[u32],
    mode_tag_of: &dyn Fn(u32) -> u32,
    cache: &mut PlaneCache,
) -> Vec<Face> {
    match moving {
        [] => Vec::new(),
        [d] => {
            let Some(o) = scene.object_obb(*d) else { return Vec::new() };
            let mut out = obb_faces(&o, *d);
            out.extend(mesh_planes(scene, *d, mode_tag_of(*d), cache));
            out
        }
        many => {
            let mut mn = Vec::new();
            let mut mx = Vec::new();
            for &d in many {
                if let Some((a, b)) = scene.aabb_of(d) {
                    mn.push(Vec3::from(a));
                    mx.push(Vec3::from(b));
                }
            }
            if mn.is_empty() {
                return Vec::new();
            }
            let lo = mn.iter().copied().fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let hi = mx.iter().copied().fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            obb_faces(&Obb::from_aabb(lo, hi), NO_DATUM)
        }
    }
}

/// The model's large planar faces in WORLD space, extracted once per render model and cached.
pub fn mesh_planes(scene: &dyn ObjectScene, datum: u32, mode_tag: u32, cache: &mut PlaneCache) -> Vec<Face> {
    let Some(o) = scene.object_obb(datum) else { return Vec::new() };
    let fr = Frame::of_obb(&o);
    let key = PlaneCache::key(mode_tag, &o);
    if cache.get(key).is_none() {
        let tris: Vec<[Vec3; 3]> = scene
            .selection_wireframe(datum)
            .map(|l| tris_from_line_list(&l))
            .unwrap_or_default();
        // Clustered in WORLD space (a handful of faces to convert, rather than every vertex),
        // then stored in the piece's own frame so the entry survives it being moved or turned.
        let lf = dominant_planes(&tris, MIN_PLANE_AREA_FRAC, MAX_MODEL_PLANES)
            .iter()
            .enumerate()
            .map(|(i, f)| LocalFace::of(f, &fr, i.min(255) as u8))
            .collect();
        cache.insert(key, lf);
    }
    cache
        .get(key)
        .map(|v| v.iter().map(|lf| lf.to_world(&fr, datum)).collect())
        .unwrap_or_default()
}

/// A mesh plane must hold at least this fraction of the model's triangle area to count as one of
/// its real faces (a ramp deck is far above it; a bevel or a bolt head is far below).
pub const MIN_PLANE_AREA_FRAC: f32 = 0.06;
/// At most this many planar faces per model (the biggest ones win).
pub const MAX_MODEL_PLANES: usize = 8;

/// Target faces: the oriented box faces of every candidate object except the moving set.
/// Poses of non-moving objects do not change during a drag, so this is gathered ONCE per
/// transform op and only distance-filtered per frame ([`near_targets`]).
pub fn target_faces(scene: &dyn ObjectScene, candidates: &[u32], exclude: &[u32]) -> Vec<Face> {
    let mut out = Vec::new();
    for &d in candidates {
        if exclude.contains(&d) {
            continue;
        }
        if let Some(o) = scene.object_obb(d) {
            out.extend(obb_faces(&o, d));
        }
    }
    out
}

/// Faces whose rectangle could possibly be reached from `center` within `reach`.
pub fn near_targets(all: &[Face], center: Vec3, reach: f32) -> Vec<Face> {
    all.iter()
        .filter(|f| (f.c - center).length() <= reach + f.hu + f.hv)
        .copied()
        .collect()
}

/// Probe the LEVEL geometry (BSP) for target planes: one ray per source face along its travel
/// direction. The hit triangle's plane becomes an effectively infinite target face, which is how
/// a piece snaps flush to terrain and to map walls, not just to other pieces.
pub fn bsp_targets(
    ray: &dyn Fn(Vec3, Vec3) -> Option<(Vec3, Vec3)>,
    srcs: &[Face],
    dirs: &[Vec3],
    cfg: &SnapCfg,
) -> Vec<Face> {
    let mut out = Vec::new();
    for s in srcs {
        let own = [s.n];
        let list: &[Vec3] = if dirs.is_empty() { &own } else { dirs };
        for &d in list {
            let d = d.normalize_or_zero();
            if d.length_squared() < 0.5 || d.dot(s.n) < 0.30 {
                continue;
            }
            // Start a touch behind the face so a piece already resting on a surface still sees it.
            let o = s.c - d * cfg.pull_back;
            let Some((p, n)) = ray(o, d) else { continue };
            if (p - o).dot(d) > cfg.range + cfg.pull_back + 1e-3 {
                continue;
            }
            // The BSP triangle soup is NOT consistently wound (gravity settle flips its normals
            // the same way), so present the hit plane FACING the ray we arrived on. Without this
            // a back-wound floor triangle reads as pointing into the ground and the pair is
            // thrown out for not opposing the moving face.
            let n = n.normalize_or_zero();
            if n.length_squared() < 0.5 {
                continue;
            }
            let n = if n.dot(d) > 0.0 { -n } else { n };
            let (u, v) = basis(n);
            out.push(Face {
                c: p,
                n,
                u,
                v,
                hu: 1.0e4,
                hv: 1.0e4,
                area: f32::MAX,
                kind: FaceKind::Bsp,
                datum: NO_DATUM,
            });
        }
    }
    out
}

/// Full scene-driven snap: gather the moving set's faces, the nearby objects' faces and (with
/// `bsp`) the level-geometry probe planes, then solve. `axis` locks the travel to +/- that
/// direction; `None` lets every source face push along its own normal.
///
/// Used by the scripted `snapto` verb; the interactive drag pre-gathers its faces once per op
/// (see `App::begin_xform`) and calls [`solve_faces`] each frame instead.
#[allow(clippy::too_many_arguments)]
pub fn solve_scene(
    scene: &dyn ObjectScene,
    moving: &[u32],
    candidates: &[u32],
    mode_tag_of: &dyn Fn(u32) -> u32,
    axis: Option<Vec3>,
    bsp: bool,
    cache: &mut PlaneCache,
    cfg: &SnapCfg,
) -> Option<SnapResult> {
    let srcs = source_faces(scene, moving, mode_tag_of, cache);
    if srcs.is_empty() {
        return None;
    }
    let center = srcs.iter().fold(Vec3::ZERO, |a, f| a + f.c) / srcs.len() as f32;
    let reach = srcs.iter().map(|f| (f.c - center).length()).fold(0.0f32, f32::max) + cfg.range;
    let dirs: Vec<Vec3> = match axis {
        Some(a) => vec![a, -a],
        None => Vec::new(),
    };
    let mut tgts = near_targets(&target_faces(scene, candidates, moving), center, reach);
    if bsp {
        tgts.extend(bsp_targets(&|o, d| scene.raycast_scene_n(o, d), &srcs, &dirs, cfg));
    }
    solve_faces(&srcs, &tgts, &dirs, cfg)
}

// ============================================================== line arrays

/// How a line array decides where the copies go.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineSpec {
    /// Exactly `n` copies, the first at the start point and the last at the end point.
    Count(u32),
    /// One copy every `step` world units from the start until the end is passed. A step smaller
    /// than the piece's own extent makes the copies OVERLAP, which is the whole point.
    Step(f32),
}

/// The Construct LINE tool's settings. Kept here next to the fill maths so the panel, the
/// viewport clicks and the scripted `array line` verb can never disagree about what they mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineTool {
    /// true = fill by STEP (copies every `step` wu, overlap allowed); false = fill by COUNT.
    pub use_step: bool,
    pub count: u32,
    pub step: f32,
    /// Turn each copy's heading to follow the line (yaw only, so nothing tips over).
    pub align: bool,
    /// true = ONE click plus `length` along `axis`; false = click the two ends of the line.
    pub one_click: bool,
    pub length: f32,
    /// 0 = X, 1 = Y, 2 = Z, with `negative` flipping it.
    pub axis: usize,
    pub negative: bool,
}

impl Default for LineTool {
    fn default() -> Self {
        LineTool { use_step: false, count: 5, step: 2.0, align: false, one_click: false, length: 20.0, axis: 0, negative: false }
    }
}

impl LineTool {
    pub fn spec(&self) -> LineSpec {
        if self.use_step {
            LineSpec::Step(self.step.max(0.05))
        } else {
            LineSpec::Count(self.count.max(1))
        }
    }
    /// The one-click mode's direction.
    pub fn dir(&self) -> Vec3 {
        let a = [Vec3::X, Vec3::Y, Vec3::Z][self.axis.min(2)];
        if self.negative {
            -a
        } else {
            a
        }
    }
}

/// Copy positions along a segment. Position 0 is always `from`.
pub fn line_positions(from: Vec3, to: Vec3, spec: LineSpec) -> Vec<Vec3> {
    let seg = to - from;
    let len = seg.length();
    match spec {
        LineSpec::Count(n) => {
            let n = (n.max(1) as usize).min(MAX_COPIES);
            if n == 1 {
                return vec![from];
            }
            (0..n).map(|k| from + seg * (k as f32 / (n - 1) as f32)).collect()
        }
        LineSpec::Step(s) => {
            if !(s > 1e-4) || len < 1e-4 {
                return vec![from];
            }
            let d = seg / len;
            let n = ((len / s).floor() as usize + 1).clamp(1, MAX_COPIES);
            (0..n).map(|k| from + d * (s * k as f32)).collect()
        }
    }
}

/// Copy positions along an axis from `origin`: `count` copies, `step` apart.
pub fn axis_positions(origin: Vec3, dir: Vec3, count: u32, step: f32) -> Vec<Vec3> {
    let d = dir.normalize_or_zero();
    let n = (count.max(1) as usize).min(MAX_COPIES);
    if d.length_squared() < 0.5 {
        return vec![origin];
    }
    (0..n).map(|k| origin + d * (step * k as f32)).collect()
}

/// The actual step between consecutive copies of a `count` fill (0 when there is only one).
pub fn count_step(from: Vec3, to: Vec3, count: u32) -> f32 {
    if count <= 1 {
        0.0
    } else {
        (to - from).length() / (count - 1) as f32
    }
}

/// Gap (positive) or overlap (negative) between consecutive copies of a piece `extent` wu long.
pub fn step_gap(step: f32, extent: f32) -> f32 {
    step - extent
}

/// "flush" / "gap 0.50 wu" / "overlap 1.20 wu" — the live readout's spacing half.
pub fn spacing_label(step: f32, extent: f32) -> String {
    let g = step_gap(step, extent);
    if g.abs() < 0.005 {
        "flush".to_string()
    } else if g > 0.0 {
        format!("gap {g:.2} wu")
    } else {
        format!("overlap {:.2} wu", -g)
    }
}

/// YAW-only rotation taking `fwd`'s heading onto `dir`'s.
///
/// "Align each copy to the line" must not tip a Forge piece over, so only the heading turns: a
/// wall stays vertical and a ramp keeps its rise. A vertical line (or a piece with no horizontal
/// heading) leaves the piece alone.
pub fn yaw_to(fwd: Vec3, dir: Vec3) -> Quat {
    let a = Vec3::new(fwd.x, fwd.y, 0.0);
    let b = Vec3::new(dir.x, dir.y, 0.0);
    if a.length_squared() < 1e-8 || b.length_squared() < 1e-8 {
        return Quat::IDENTITY;
    }
    let (a, b) = (a.normalize(), b.normalize());
    Quat::from_rotation_z(b.y.atan2(b.x) - a.y.atan2(a.x))
}

/// One notch of the array-step wheel: `fine` (Alt held) trims by centimetres, otherwise by a
/// quarter unit. Scroll UP grows the step (copies spread out), DOWN shrinks it (they overlap).
pub const STEP_NOTCH_COARSE: f32 = 0.25;
pub const STEP_NOTCH_FINE: f32 = 0.05;

/// Apply `notches` of wheel to an array step adjustment, keeping the resulting step positive.
pub fn adjust_step(adj: f32, base: f32, notches: i32, fine: bool) -> f32 {
    let inc = if fine { STEP_NOTCH_FINE } else { STEP_NOTCH_COARSE };
    let a = adj + inc * notches as f32;
    // The total step (base + adj) must stay meaningfully positive, or the array collapses.
    a.max(0.05 - base)
}

// ===================================================================== tests

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box(c: Vec3, h: Vec3) -> Obb {
        Obb { c, hx: Vec3::X * h.x, hy: Vec3::Y * h.y, hz: Vec3::Z * h.z }
    }

    /// A box offers six faces with outward normals and the right centres.
    #[test]
    fn box_faces_are_oriented_and_outward() {
        let o = unit_box(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 2.0, 0.5));
        let f = obb_faces(&o, 7);
        assert_eq!(f.len(), 6);
        let top = f.iter().find(|f| f.n.z > 0.99).unwrap();
        assert!((top.c - Vec3::new(1.0, 2.0, 3.5)).length() < 1e-5);
        assert!((top.area - 4.0 * 1.0 * 2.0).abs() < 1e-4);
        assert_eq!(top.datum, 7);
        // Every normal points away from the centre.
        for face in &f {
            assert!((face.c - o.c).dot(face.n) > 0.0);
        }
    }

    /// A rotated box reports its REAL face directions (the whole point of oriented faces).
    #[test]
    fn rotated_box_faces_follow_the_rotation() {
        let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let o = Obb { c: Vec3::ZERO, hx: q * (Vec3::X * 2.0), hy: q * Vec3::Y, hz: Vec3::Z };
        let f = obb_faces(&o, 0);
        let plus_x = f.iter().find(|f| f.kind == FaceKind::Box(1)).unwrap();
        assert!((plus_x.n - (q * Vec3::X)).length() < 1e-5);
    }

    /// A flat plate's rim faces have no area and are dropped.
    #[test]
    fn degenerate_faces_are_dropped() {
        let o = unit_box(Vec3::ZERO, Vec3::new(2.0, 2.0, 0.0));
        let f = obb_faces(&o, 0);
        assert_eq!(f.len(), 2, "only the two flat faces survive");
        assert!(f.iter().all(|f| f.n.z.abs() > 0.99));
    }

    /// A wedge (ramp) mesh yields its deck, its floor and its tall end as dominant planes —
    /// exactly the faces an AABB cannot offer.
    #[test]
    fn dominant_planes_find_a_ramp_deck() {
        // A 4 x 2 x 4 wedge rising along +X, wound so every normal points OUT.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(4.0, 0.0, 0.0);
        let c = Vec3::new(4.0, 0.0, 4.0);
        let d = Vec3::new(0.0, 2.0, 0.0);
        let e = Vec3::new(4.0, 2.0, 0.0);
        let g = Vec3::new(4.0, 2.0, 4.0);
        let tris = vec![
            [a, e, b], [a, d, e], // floor, outward -Z
            [b, e, g], [b, g, c], // tall end, outward +X
            [a, c, g], [a, g, d], // sloped deck, outward (-1, 0, 1)/sqrt(2)
            [a, b, c],            // side y = 0, outward -Y
            [d, g, e],            // side y = 2, outward +Y
        ];
        let f = dominant_planes(&tris, 0.05, 8);
        let has = |pred: &dyn Fn(&Face) -> bool| f.iter().any(pred);
        assert!(has(&|f: &Face| f.n.z < -0.99), "floor plane (outward -Z)");
        assert!(has(&|f: &Face| f.n.z > 0.2 && f.n.z < 0.99 && f.n.x < -0.2), "sloped deck");
        assert!(has(&|f: &Face| f.n.x > 0.99), "tall end face");
        // The deck's label reads as a slope, so the status line can say so.
        let deck = f.iter().find(|f| f.n.z > 0.2 && f.n.z < 0.99).unwrap();
        assert_eq!(deck.label(), "slope");
    }

    /// Faces are cached in the piece's own frame and come back in world space unchanged.
    #[test]
    fn local_face_round_trips_through_a_frame() {
        let q = Quat::from_rotation_z(0.7);
        let o = Obb { c: Vec3::new(5.0, -2.0, 1.0), hx: q * (Vec3::X * 2.0), hy: q * Vec3::Y, hz: Vec3::Z * 0.5 };
        let fr = Frame::of_obb(&o);
        let f = obb_faces(&o, 3)[0];
        let lf = LocalFace::of(&f, &fr, 0);
        let back = lf.to_world(&fr, 3);
        assert!((back.c - f.c).length() < 1e-4);
        assert!((back.n - f.n).length() < 1e-4);
    }

    /// The classic case: push a block's +X face onto a wall 0.4 wu away.
    #[test]
    fn face_snap_closes_the_gap_along_the_axis() {
        let block = unit_box(Vec3::ZERO, Vec3::splat(1.0)); // +X face at x = 1
        let wall = unit_box(Vec3::new(3.4, 0.0, 0.0), Vec3::new(1.0, 4.0, 4.0)); // -X face at x = 2.4
        let srcs = obb_faces(&block, 1);
        let tgts = obb_faces(&wall, 2);
        let r = solve_faces(&srcs, &tgts, &[Vec3::X, -Vec3::X], &SnapCfg::default()).expect("a snap");
        assert!((r.gap - 1.4).abs() < 1e-4, "gap {}", r.gap);
        assert!((r.delta.x - 1.4).abs() < 1e-4, "delta {:?}", r.delta);
        assert!(r.delta.y.abs() < 1e-4 && r.delta.z.abs() < 1e-4, "axis-locked move stays on the axis");
        assert_eq!(r.tgt_datum, 2);
        assert_eq!(r.src_label, "+X");
    }

    /// Out of range = no snap at all (the magnet must not reach across the map).
    #[test]
    fn nothing_snaps_beyond_the_range() {
        let block = unit_box(Vec3::ZERO, Vec3::splat(1.0));
        let wall = unit_box(Vec3::new(20.0, 0.0, 0.0), Vec3::new(1.0, 4.0, 4.0));
        let srcs = obb_faces(&block, 1);
        let tgts = obb_faces(&wall, 2);
        assert!(solve_faces(&srcs, &tgts, &[Vec3::X, -Vec3::X], &SnapCfg::default()).is_none());
    }

    /// Faces that do not overlap laterally are not a mate, however close their planes are.
    #[test]
    fn a_face_that_misses_sideways_does_not_snap() {
        let block = unit_box(Vec3::ZERO, Vec3::splat(1.0));
        // Same plane distance, but 20 wu off to the side.
        let wall = unit_box(Vec3::new(2.5, 20.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let srcs = obb_faces(&block, 1);
        let tgts = obb_faces(&wall, 2);
        assert!(solve_faces(&srcs, &tgts, &[Vec3::X, -Vec3::X], &SnapCfg::default()).is_none());
    }

    /// A RAMP dropped near a platform: its tall end face mates with the platform's side, and the
    /// lateral pass lands the deck level with the platform top instead of a few cm off.
    #[test]
    fn ramp_end_mates_platform_side_and_edge_aligns() {
        // Ramp: 4 long, 2 wide, 1 high, deck rising toward +X, sitting with its top at z = 1.
        // Its tall end face: centre (2, 0, 0.5), normal +X, half-extents 1 (y) x 0.5 (z).
        let ramp_end = Face {
            c: Vec3::new(2.0, 0.0, 0.5),
            n: Vec3::X,
            u: Vec3::Y,
            v: Vec3::Z,
            hu: 1.0,
            hv: 0.5,
            area: 4.0,
            kind: FaceKind::Mesh(0),
            datum: 11,
        };
        // Platform: top at z = 0.92 (8 cm below the ramp deck), its -X side face at x = 2.6.
        let plat = Obb {
            c: Vec3::new(5.6, 0.0, -0.08),
            hx: Vec3::X * 3.0,
            hy: Vec3::Y * 3.0,
            hz: Vec3::Z * 1.0,
        };
        let tgts = obb_faces(&plat, 22);
        let r = solve_faces(&[ramp_end], &tgts, &[], &SnapCfg::default()).expect("a snap");
        // Face contact: the ramp end travels +X to x = 2.6.
        assert!((r.delta - r.lateral - Vec3::X * 0.6).length() < 1e-4, "face shift {:?}", r.delta - r.lateral);
        // Lateral: -0.08 in Z, dropping the ramp's deck level with the platform top.
        assert!((r.lateral.z + 0.08).abs() < 1e-4, "lateral {:?}", r.lateral);
        assert_eq!(r.edge_axis, Some("Z"));
        assert_eq!(r.src_label, "end");
    }

    /// A ramp's deck lands level with a platform top from a 0.35 wu miss: the tolerance comes
    /// from the end face's WIDTH (2 wu), not from its 1 wu height.
    #[test]
    fn a_ramp_deck_still_clicks_level_from_a_third_of_its_height() {
        // Ramp 2 x 2 x 1 with its top at z = 1.0; its +X end face is 2 wide, 1 tall.
        let end = Face {
            c: Vec3::new(1.0, 0.0, 0.5),
            n: Vec3::X,
            u: Vec3::Y,
            v: Vec3::Z,
            hu: 1.0,
            hv: 0.5,
            area: 4.0,
            kind: FaceKind::Mesh(0),
            datum: 1,
        };
        // Platform 1 wu thick with its top 0.35 above the ramp's deck; its -X side is 0.6 away.
        let plat = Obb {
            c: Vec3::new(4.6, 0.0, 0.85),
            hx: Vec3::X * 3.0,
            hy: Vec3::Y * 3.0,
            hz: Vec3::Z * 0.5,
        };
        let r = solve_faces(&[end], &obb_faces(&plat, 9), &[], &SnapCfg::default()).expect("a snap");
        assert!((r.delta.x - 0.6).abs() < 1e-4, "delta {:?}", r.delta);
        assert!((r.lateral.z - 0.35).abs() < 1e-4, "lateral {:?}", r.lateral);
        assert_eq!(r.edge_axis, Some("Z"));
    }

    /// The lateral pass never outweighs the face snap: a 1.2 wu misalignment is left alone even
    /// though the piece is big enough for the tolerance to allow it.
    #[test]
    fn lateral_alignment_is_capped_by_the_face_travel() {
        let src = Face {
            c: Vec3::new(2.0, 0.0, 1.2),
            n: Vec3::X,
            u: Vec3::Y,
            v: Vec3::Z,
            hu: 4.0,
            hv: 4.0,
            area: 64.0,
            kind: FaceKind::Box(1),
            datum: 1,
        };
        let tgt = Face {
            c: Vec3::new(2.05, 0.0, 0.0),
            n: -Vec3::X,
            u: Vec3::Y,
            v: Vec3::Z,
            hu: 4.0,
            hv: 4.0,
            area: 64.0,
            kind: FaceKind::Box(0),
            datum: 2,
        };
        let cfg = SnapCfg::default();
        let r = solve_faces(&[src], &[tgt], &[], &cfg).expect("a snap");
        assert!(((r.delta - r.lateral).x - 0.05).abs() < 1e-4);
        // Face travel 0.05 < the 0.35 floor, so the cap is the floor -- and 1.2 wu is over it.
        assert!(r.lateral.length() < 1e-6, "lateral {:?} should have been refused", r.lateral);
        // Inside the floor it DOES click: same pair, 0.3 wu off.
        let near = Face { c: Vec3::new(2.0, 0.0, 0.3), ..src };
        let r2 = solve_faces(&[near], &[tgt], &[], &cfg).expect("a snap");
        assert!((r2.lateral.z + 0.3).abs() < 1e-4, "lateral {:?}", r2.lateral);
    }

    /// A piece already buried a little in a surface is pulled back OUT (negative gap).
    #[test]
    fn overlapping_pieces_are_pushed_apart() {
        let block = unit_box(Vec3::ZERO, Vec3::splat(1.0)); // +X face at x = 1
        let wall = unit_box(Vec3::new(1.7, 0.0, 0.0), Vec3::new(1.0, 4.0, 4.0)); // -X face at 0.7
        let r = solve_faces(&obb_faces(&block, 1), &obb_faces(&wall, 2), &[Vec3::X, -Vec3::X], &SnapCfg::default())
            .expect("a snap");
        assert!(r.gap < 0.0 && (r.gap + 0.3).abs() < 1e-4, "gap {}", r.gap);
    }

    /// Free (no axis lock) snapping picks the NEAREST face pair over all six directions.
    #[test]
    fn free_snap_picks_the_nearest_surface() {
        let block = unit_box(Vec3::new(0.0, 0.0, 5.0), Vec3::splat(1.0));
        let floor = unit_box(Vec3::new(0.0, 0.0, 3.0), Vec3::new(8.0, 8.0, 0.9)); // top at 3.9
        let wall = unit_box(Vec3::new(3.2, 0.0, 5.0), Vec3::new(1.0, 8.0, 8.0)); // -X at 2.2
        let mut tgts = obb_faces(&floor, 2);
        tgts.extend(obb_faces(&wall, 3));
        let r = solve_faces(&obb_faces(&block, 1), &tgts, &[], &SnapCfg::default()).expect("a snap");
        // floor gap = 4.0 - 3.9 = 0.1; wall gap = 2.2 - 1.0 = 1.2 -> the floor wins.
        assert_eq!(r.tgt_datum, 2);
        assert!((r.delta.z + 0.1).abs() < 1e-4, "delta {:?}", r.delta);
    }

    /// A back-wound floor triangle still snaps: the probe plane is flipped to face the ray.
    #[test]
    fn a_back_wound_bsp_triangle_still_snaps() {
        let block = unit_box(Vec3::new(0.0, 0.0, 2.0), Vec3::splat(1.0)); // bottom at z = 1
        let srcs = obb_faces(&block, 1);
        let cfg = SnapCfg::default();
        // Same floor as below, but the soup hands back the normal pointing DOWN.
        let ray = |o: Vec3, d: Vec3| -> Option<(Vec3, Vec3)> {
            (d.z < -0.5).then(|| (Vec3::new(o.x, o.y, 0.35), -Vec3::Z))
        };
        let tgts = bsp_targets(&ray, &srcs, &[], &cfg);
        assert_eq!(tgts.len(), 1);
        assert!(tgts[0].n.z > 0.99, "the probe plane faces the ray");
        let r = solve_faces(&srcs, &tgts, &[], &cfg).expect("a snap");
        assert!((r.delta.z + 0.65).abs() < 1e-4, "delta {:?}", r.delta);
    }

    /// BSP probes become target planes and the piece lands flush on terrain.
    #[test]
    fn bsp_probe_becomes_a_target_plane() {
        let block = unit_box(Vec3::new(0.0, 0.0, 2.0), Vec3::splat(1.0)); // bottom at z = 1
        let srcs = obb_faces(&block, 1);
        let cfg = SnapCfg::default();
        // A flat floor at z = 0.35.
        let ray = |o: Vec3, d: Vec3| -> Option<(Vec3, Vec3)> {
            if d.z < -0.5 {
                Some((Vec3::new(o.x, o.y, 0.35), Vec3::Z))
            } else {
                None
            }
        };
        let tgts = bsp_targets(&ray, &srcs, &[], &cfg);
        assert_eq!(tgts.len(), 1);
        let r = solve_faces(&srcs, &tgts, &[], &cfg).expect("a snap");
        assert!((r.delta.z + 0.65).abs() < 1e-4, "delta {:?}", r.delta);
        assert_eq!(r.tgt_label, "map surface");
        assert_eq!(r.tgt_datum, NO_DATUM);
    }

    // --------------------------------------------------------------- arrays

    #[test]
    fn count_fill_spans_the_line_inclusive() {
        let p = line_positions(Vec3::ZERO, Vec3::X * 10.0, LineSpec::Count(5));
        assert_eq!(p.len(), 5);
        assert!((p[0] - Vec3::ZERO).length() < 1e-6);
        assert!((p[4] - Vec3::X * 10.0).length() < 1e-6);
        assert!((p[1].x - 2.5).abs() < 1e-6);
        assert!((count_step(Vec3::ZERO, Vec3::X * 10.0, 5) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn one_copy_sits_at_the_start() {
        assert_eq!(line_positions(Vec3::ZERO, Vec3::X * 10.0, LineSpec::Count(1)).len(), 1);
        assert_eq!(line_positions(Vec3::ZERO, Vec3::ZERO, LineSpec::Step(1.0)).len(), 1);
    }

    /// A step SMALLER than the piece is legal and is what "overlapping copies" means.
    #[test]
    fn step_fill_allows_overlap_and_counts_correctly() {
        let p = line_positions(Vec3::ZERO, Vec3::X * 10.0, LineSpec::Step(1.5));
        assert_eq!(p.len(), 7, "0, 1.5 .. 9.0");
        assert!((p[1].x - 1.5).abs() < 1e-6);
        // A 2 wu piece stepped 1.5 overlaps by 0.5.
        assert!((step_gap(1.5, 2.0) + 0.5).abs() < 1e-6);
        assert_eq!(spacing_label(1.5, 2.0), "overlap 0.50 wu");
        assert_eq!(spacing_label(2.0, 2.0), "flush");
        assert_eq!(spacing_label(2.5, 2.0), "gap 0.50 wu");
    }

    #[test]
    fn a_fill_can_never_run_away() {
        let p = line_positions(Vec3::ZERO, Vec3::X * 1.0e6, LineSpec::Step(0.001));
        assert_eq!(p.len(), MAX_COPIES);
        assert_eq!(line_positions(Vec3::ZERO, Vec3::X, LineSpec::Count(100_000)).len(), MAX_COPIES);
    }

    #[test]
    fn axis_fill_steps_from_the_origin() {
        let p = axis_positions(Vec3::new(1.0, 1.0, 1.0), -Vec3::Y, 3, 2.0);
        assert_eq!(p.len(), 3);
        assert!((p[2] - Vec3::new(1.0, -3.0, 1.0)).length() < 1e-6);
    }

    /// The wheel adjusts the step and can never drive it to zero or negative.
    #[test]
    fn step_adjustment_stays_positive() {
        assert!((adjust_step(0.0, 2.0, 2, false) - 0.5).abs() < 1e-6);
        assert!((adjust_step(0.0, 2.0, -3, true) + 0.15).abs() < 1e-6);
        // 2.0 base, 40 notches down: clamped so base+adj = 0.05.
        assert!((2.0 + adjust_step(0.0, 2.0, -40, false) - 0.05).abs() < 1e-6);
    }

    /// Aligning to the line turns the heading only — a piece is never tipped over.
    #[test]
    fn yaw_alignment_keeps_pieces_upright() {
        let q = yaw_to(Vec3::X, Vec3::Y);
        assert!((q * Vec3::X - Vec3::Y).length() < 1e-5);
        assert!((q * Vec3::Z - Vec3::Z).length() < 1e-5);
        // A vertical line has no heading -> leave the piece alone.
        assert_eq!(yaw_to(Vec3::X, Vec3::Z), Quat::IDENTITY);
    }
}
