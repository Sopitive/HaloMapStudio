//! A contact manifold as the engine keeps it: one `hkpSimpleContactConstraintAtom` per body pair,
//! its contact points, the per-point solver memory and the friction memory of the atom.
//!
//! # Provenance (all **PROVEN** from `haloreach_debug.dll` unless marked)
//!
//! * `hkpSimpleConstraintContactMgr::addContactPointImpl` `sub_180D77270`: a new point gets a
//!   properties record (`sub_180D8D990`) zeroed except `flags = CONTACT_IS_NEW (1)`; a point at an
//!   odd index whose predecessor has no `CONTACT_USES_SOLVER_PATH2` flag and no max impulse gets
//!   `flags = 3` (so points 1, 3, 5, ... are solved as coupled pairs with their predecessor).
//! * the material `sub_180D78040`: `friction = hkUFloat8(sqrt(fA * fB))` (round UP into the table),
//!   `restitution = (int)(sqrt(rA * rB) * 128)`, `maxImpulse = 0`.
//! * `processContact` `sub_180D776E0`: every tick the agent's current position / normal / distance
//!   overwrite the stored point (matched by contact-point id); the solver memory stays.
//! * record layout (offsets inside the properties record the solver reads/writes):
//!   `+0x00` last total normal impulse, `+0x04` pending separation ("rec4"), `+0x10` friction code,
//!   `+0x11` restitution code, `+0x12` max-impulse code, `+0x13` flags, `+0x18` the penetration
//!   memory ("mem"). Atom: `+0x12` friction axis, `+0x14` mean radius, `+0x18..+0x2C` friction
//!   impulses and slip memory.

use glam::Vec3;

/// The part of the point record the solver owns (see the module note).
#[derive(Clone, Copy, Debug, Default)]
pub struct PointState {
    /// `rec+0x00`: the total normal impulse of the last solve (the friction cap reads it).
    pub impulse: f32,
    /// `rec+0x04`: remaining target separation (restitution seed, then bookkeeping).
    pub pending: f32,
    /// `rec+0x18`: the penetration memory (always negative once solved).
    pub mem: f32,
    /// `CONTACT_IS_NEW`.
    pub is_new: bool,
}

/// One contact point of a manifold.
#[derive(Clone, Copy, Debug)]
pub struct ContactPoint {
    /// Identity across ticks (the agent's contact-point id: a feature id of the shape pair).
    pub id: u32,
    pub pos: Vec3,
    /// Unit normal pointing from body B into body A (`hkContactPoint::separatingNormal.xyz`).
    pub normal: Vec3,
    /// `separatingNormal.w`: the gap along the normal, negative = penetrating. Includes the
    /// shapes' convex radii (a padded shape reports the gap between the padded surfaces).
    pub dist: f32,
    pub st: PointState,
}

impl ContactPoint {
    pub fn new(id: u32, pos: Vec3, normal: Vec3, dist: f32) -> Self {
        ContactPoint { id, pos, normal, dist, st: PointState { is_new: true, ..Default::default() } }
    }
}

/// `hkContactPointMaterial` as the engine mixes it (`sub_180D78040`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactMaterial {
    /// hkUFloat8 code of `sqrt(fA * fB)`.
    pub friction_code: u8,
    /// `(int)(sqrt(rA * rB) * 128)`.
    pub restitution_code: u8,
}

impl ContactMaterial {
    /// Mix two bodies' materials exactly as the engine's contact manager does.
    pub fn mix(friction_a: f32, restitution_a: f32, friction_b: f32, restitution_b: f32) -> Self {
        let f = (friction_a * friction_b).sqrt();
        let r = ((restitution_a * restitution_b).sqrt() * 128.0) as i32;
        ContactMaterial { friction_code: crate::reach_havok::ufloat8_encode(f), restitution_code: r.clamp(0, 255) as u8 }
    }
    /// The friction coefficient in force (the table entry).
    pub fn friction(&self) -> f32 {
        crate::reach_havok::ufloat8_decode(self.friction_code)
    }
    /// The restitution in force.
    pub fn restitution(&self) -> f32 {
        self.restitution_code as f32 * 0.007_812_5
    }
}

/// The atom's own memory (`hkpSimpleContactConstraintAtom` + 0x12 .. 0x2C).
#[derive(Clone, Copy, Debug, Default)]
pub struct AtomState {
    /// `+0x12`: which world axis the friction tangent is built from.
    pub axis: u8,
    /// `+0x14`: mean distance of the points from their centroid (torsional friction radius).
    pub radius: f32,
    /// `info.flags & 4`: the radius must be recomputed.
    pub radius_dirty: bool,
    /// `+0x1C, +0x24, +0x2C`: tangential slip memory of the two friction rows and the torsion row.
    pub slip: [f32; 3],
    /// `+0x18, +0x20, +0x28`: last friction impulses (informational).
    pub impulse: [f32; 3],
}

/// One contact constraint: bodies `a` and `b` (`None` = the fixed world) and their points.
#[derive(Clone, Debug)]
pub struct Manifold {
    pub a: usize,
    pub b: Option<usize>,
    pub material: ContactMaterial,
    pub points: Vec<ContactPoint>,
    pub atom: AtomState,
    /// `hkpMassChangerModifierConstraintAtom` factors `(a, b)` multiplying the INVERSE mass and
    /// inertia of each body for this constraint only (`sub_180D93CF0` case 0x18). `(1, 1)` = none.
    pub mass_scale: (f32, f32),
}

impl Manifold {
    pub fn new(a: usize, b: Option<usize>, material: ContactMaterial) -> Self {
        Manifold { a, b, material, points: Vec::new(), atom: AtomState { radius_dirty: true, ..Default::default() }, mass_scale: (1.0, 1.0) }
    }

    /// `processContact`: replace the geometry of the existing points by the agent's fresh result.
    /// Points are matched by `id`; a point that disappeared is removed, a new id is appended as a
    /// new point (memory zeroed, `CONTACT_IS_NEW`). Returns true when the point set changed
    /// (the atom's radius flag is raised then). This is the plain form (every fresh point is
    /// taken); the engine's admission rules are [`Manifold::update`].
    pub fn sync(&mut self, fresh: &[ContactPoint]) -> bool {
        let mut changed = false;
        let old = std::mem::take(&mut self.points);
        for f in fresh {
            match old.iter().find(|p| p.id == f.id) {
                Some(p) => self.points.push(ContactPoint { id: f.id, pos: f.pos, normal: f.normal, dist: f.dist, st: p.st }),
                None => {
                    changed = true;
                    self.points.push(ContactPoint::new(f.id, f.pos, f.normal, f.dist));
                }
            }
        }
        if self.points.len() != old.len() {
            changed = true;
        }
        if changed {
            self.atom.radius_dirty = true;
        }
        changed
    }

    /// The engine's PRED_GSK admission rules applied to the agent's candidate points
    /// (`sub_180E2A180` / `sub_180E28940` / `sub_180E29410`, `01-collision-detection.md` 3.2):
    ///
    /// * a stored point lives while its candidate is still present with `dist < tolerance`;
    ///   it follows the candidate's geometry;
    /// * a NEW point is admitted only if it already penetrates: `dist < new_threshold`
    ///   (`-0.5 |g| dt^2 = -5.79e-4`; an edge-edge feature uses `tolerance`) and the pair is
    ///   predicted to be at least `predicted` deep at the end of the step
    ///   (`dist + approach <= -1.158e-3`);
    /// * a manifold holds at most 4 points: a fifth candidate replaces the one that leaves the
    ///   widest spread ([`choose_drop`]).
    pub fn update(&mut self, fresh: &[Candidate], adm: &Admission) -> bool {
        let mut changed = false;
        let old = std::mem::take(&mut self.points);
        // 1. persistent points
        for p in old.iter() {
            match fresh.iter().find(|c| c.point.id == p.id) {
                Some(c) if c.point.dist < adm.tolerance => self.points.push(ContactPoint { id: p.id, pos: c.point.pos, normal: c.point.normal, dist: c.point.dist, st: p.st }),
                _ => changed = true,
            }
        }
        // 2. new points, deepest first
        let mut news: Vec<&Candidate> = fresh.iter().filter(|c| !old.iter().any(|p| p.id == c.point.id)).collect();
        news.sort_by(|a, b| a.point.dist.partial_cmp(&b.point.dist).unwrap_or(std::cmp::Ordering::Equal));
        for c in news {
            let thr = if c.edge { adm.tolerance } else { adm.new_threshold };
            if !(c.point.dist < thr) || !(c.edge || c.point.dist + c.approach <= adm.predicted) {
                continue;
            }
            let np = ContactPoint::new(c.point.id, c.point.pos, c.point.normal, c.point.dist);
            if self.points.len() < MAX_POINTS {
                self.points.push(np);
                changed = true;
            } else if let Some(k) = choose_drop(&self.points, &np) {
                self.points[k] = np;
                changed = true;
            }
        }
        if changed {
            self.atom.radius_dirty = true;
        }
        changed
    }
}

/// At most this many points per manifold (`PRED_GSK_AGENT3`: 4 per child pair).
pub const MAX_POINTS: usize = 4;

/// A candidate contact point from the narrow phase, with what the admission rules need.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub point: ContactPoint,
    /// Edge-edge feature pair (admitted at `dist < tolerance`).
    pub edge: bool,
    /// Predicted change of the gap over the step, `n . (dispA - dispB) - angleBound`
    /// (the engine's `v16 = dist - n.relMove - angBound`, `sub_180D2C670`): negative when closing,
    /// and it includes `dt * (|wA| rA + |wB| rB)`, the most the rotation can move a surface point.
    pub approach: f32,
}

/// Thresholds of [`Manifold::update`].
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub tolerance: f32,
    pub new_threshold: f32,
    pub predicted: f32,
}

impl Admission {
    /// The engine's values for gravity magnitude `g` and tick `dt`
    /// (`qualityInfo[q]`, `sub_180D0C1D0`: `+0x08 = -0.5 g dt^2`, `+0x0C = -2 * that`).
    pub fn engine(tolerance: f32, g: f32, dt: f32) -> Self {
        let v1 = 0.5 * g * dt * dt;
        Admission { tolerance, new_threshold: -v1, predicted: -2.0 * v1 }
    }
}

/// `sub_180E298E0` (Choose): which of the four stored points (or the newcomer, `None`) leaves.
/// Keeps the four of five that span the widest quadrilateral; a clearly DEEPER newcomer that was
/// not selected still replaces the point it overlaps most (`dz^2 * 526 > nearest distance^2`,
/// ratio threshold `0.0304617453`).
pub fn choose_drop(existing: &[ContactPoint], new: &ContactPoint) -> Option<usize> {
    debug_assert_eq!(existing.len(), 4);
    let p = |i: usize| if i < 4 { existing[i].pos } else { new.pos };
    let d2 = |i: usize, j: usize| (p(i) - p(j)).length_squared();
    // a: farthest stored point from the newcomer; m: nearest
    let (mut a, mut far, mut m, mut near) = (0usize, -1.0f32, 0usize, f32::MAX);
    for i in 0..4 {
        let d = d2(i, 4);
        if d > far {
            far = d;
            a = i;
        }
        if d < near {
            near = d;
            m = i;
        }
    }
    // b: farthest from a among the other four candidates (the newcomer included)
    let b = (0..5).filter(|&j| j != a).max_by(|&x, &y| d2(x, a).partial_cmp(&d2(y, a)).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    // c: the one forming the largest triangle with a and b
    let c = (0..5)
        .filter(|&j| j != a && j != b)
        .max_by(|&x, &y| {
            let ar = |j: usize| (p(j) - p(b)).cross(p(a) - p(b)).length_squared();
            ar(x).partial_cmp(&ar(y)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    // d: of the remaining two, the one farther from c (the stored point wins a near tie)
    let rest: Vec<usize> = (0..5).filter(|&j| j != a && j != b && j != c).collect();
    let (x, y) = (rest[0], rest[1]);
    // the stored point wins a near tie (5 %); otherwise the one farther from c stays
    let drop = if d2(x, c) * 1.05 >= d2(y, c) && !(y == 4 && d2(y, c) * 1.05 >= d2(x, c)) { y } else { x };
    if drop == 4 {
        // deep-point override
        let dz = new.dist - existing[m].dist;
        if dz < 0.0 && dz * dz * 526.0 > near {
            let mut best = (0.0304617453f32, None);
            for (i, e) in existing.iter().enumerate() {
                let r = ((e.dist - new.dist).powi(2) + 1.0e-9) / (d2(i, 4) + 1.0e-9);
                if r > best.0 {
                    best = (r, Some(i));
                }
            }
            return best.1;
        }
        return None;
    }
    Some(drop)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adm() -> Admission {
        // |g| = 4.1712594, dt = 1/60: v1 = 0.5 g dt^2 = 5.7934e-4
        Admission::engine(0.0328, 4.1712594, 1.0 / 60.0)
    }

    fn cand(id: u32, x: f32, dist: f32, approach: f32, edge: bool) -> Candidate {
        Candidate { point: ContactPoint::new(id, Vec3::new(x, 0.0, 0.0), Vec3::Z, dist), edge, approach }
    }

    /// `qualityInfo[q]+0x08 = -0.5 g dt^2 = -5.79e-4`, `+0x0C = -2x`: derived by hand above.
    #[test]
    fn admission_thresholds_are_the_decoded_ones() {
        let a = adm();
        assert!((a.new_threshold + 5.7934e-4).abs() < 1e-7, "{}", a.new_threshold);
        assert!((a.predicted + 1.15868e-3).abs() < 1e-6, "{}", a.predicted);
    }

    /// A new feature pair needs to penetrate by more than 0.58 mm AND be predicted 1.16 mm deep at
    /// the end of the step; an edge-edge feature only needs to be inside the tolerance; a point that
    /// exists persists out to the tolerance.
    #[test]
    fn new_points_need_penetration_existing_ones_persist() {
        let mut m = Manifold::new(0, None, ContactMaterial { friction_code: 40, restitution_code: 0 });
        // 0.3 mm deep: not enough
        m.update(&[cand(1, 0.0, -0.0003, -0.01, false)], &adm());
        assert!(m.points.is_empty());
        // 0.8 mm deep but the step does not end 1.16 mm deep (not closing): not admitted
        m.update(&[cand(1, 0.0, -0.0008, 0.0, false)], &adm());
        assert!(m.points.is_empty());
        // 0.8 mm deep and closing 1 mm: admitted
        m.update(&[cand(1, 0.0, -0.0008, -0.001, false)], &adm());
        assert_eq!(m.points.len(), 1);
        m.points[0].st.impulse = 3.0;
        // persists at +30 mm (inside 32.8 mm), keeps its memory, follows the geometry
        m.update(&[cand(1, 0.5, 0.030, 0.0, false)], &adm());
        assert_eq!(m.points.len(), 1);
        assert_eq!(m.points[0].st.impulse, 3.0);
        assert!((m.points[0].pos.x - 0.5).abs() < 1e-6);
        // gone at +40 mm
        m.update(&[cand(1, 0.5, 0.040, 0.0, false)], &adm());
        assert!(m.points.is_empty());
        // an edge-edge feature is admitted at +20 mm
        let mut e = Manifold::new(0, None, ContactMaterial { friction_code: 40, restitution_code: 0 });
        e.update(&[cand(2, 0.0, 0.020, 0.0, true)], &adm());
        assert_eq!(e.points.len(), 1);
    }

    /// Four points, a fifth: the widest quadrilateral stays. Corners of a unit square plus a point
    /// near one corner: the near-duplicate loses.
    #[test]
    fn fifth_point_replaces_the_least_spread() {
        let mut m = Manifold::new(0, None, ContactMaterial { friction_code: 40, restitution_code: 0 });
        let sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mk = |id: u32, p: (f32, f32), d: f32| Candidate { point: ContactPoint::new(id, Vec3::new(p.0, p.1, 0.0), Vec3::Z, d), edge: false, approach: -0.01 };
        let four: Vec<_> = sq.iter().enumerate().map(|(i, p)| mk(i as u32, *p, -0.002)).collect();
        m.update(&four, &adm());
        assert_eq!(m.points.len(), 4);
        // a fifth, equally deep, 5 cm from corner 0: discarded
        let mut with_fifth = four.clone();
        with_fifth.push(mk(9, (0.05, 0.05), -0.002));
        m.update(&with_fifth, &adm());
        assert_eq!(m.points.len(), 4);
        assert!(m.points.iter().all(|p| p.id != 9), "the near-duplicate is not kept");
        // a fifth far outside the square (a wider footprint): it replaces one of the four
        let mut wide = four.clone();
        wide.push(mk(10, (3.0, 0.5), -0.002));
        let mut m2 = Manifold::new(0, None, ContactMaterial { friction_code: 40, restitution_code: 0 });
        m2.update(&four, &adm());
        m2.update(&wide, &adm());
        assert_eq!(m2.points.len(), 4);
        assert!(m2.points.iter().any(|p| p.id == 10), "the wider footprint wins a slot");
    }

    /// `choose_drop` on a unit square plus a far point: a = the corner farthest from the newcomer
    /// (0,0), b = the newcomer (3, 0.5), c = (0,1) (largest triangle with a and b: area 1.5), d = (1,0)
    /// (farther from c than (1,1)); the leftover (1,1), index 2, leaves.
    #[test]
    fn choose_drop_numbers() {
        let mk = |id: u32, x: f32, y: f32| ContactPoint::new(id, Vec3::new(x, y, 0.0), Vec3::Z, -0.002);
        let existing = [mk(0, 0.0, 0.0), mk(1, 1.0, 0.0), mk(2, 1.0, 1.0), mk(3, 0.0, 1.0)];
        assert_eq!(choose_drop(&existing, &mk(10, 3.0, 0.5)), Some(2));
        // a newcomer inside the square: it is the one that goes (None) -- same depth, no override
        assert_eq!(choose_drop(&existing, &mk(11, 0.5, 0.5)), None);
    }

    /// Prop (1.0 / 0.0) on the level (0.5 / 0.4): `sqrt(0.5) = 0.7071` rounds UP to table index 42
    /// (0.716, `01` 5.3), restitution `(int)(sqrt(0 * 0.4) * 128) = 0`; prop on prop is index 48
    /// (1.067); a 0.9 / 0.2 pair gives `(int)(0.2 * 128) = 25`.
    #[test]
    fn engine_material_mix() {
        let pl = ContactMaterial::mix(1.0, 0.0, 0.5, 0.4);
        assert_eq!((pl.friction_code, pl.restitution_code), (42, 0));
        assert!((pl.friction() - 0.716).abs() < 1e-3);
        let pp = ContactMaterial::mix(1.0, 0.0, 1.0, 0.0);
        assert_eq!(pp.friction_code, 48);
        let m = ContactMaterial::mix(0.9, 0.2, 0.9, 0.2);
        assert_eq!(m.restitution_code, 25); // (int)(0.2 * 128) = 25
    }
}
