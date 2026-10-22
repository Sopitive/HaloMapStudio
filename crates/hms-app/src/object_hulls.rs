//! #collision  A placed object's STATIC collision hull, with the Havok convex radius put back.
//!
//! A `phmo` box / polyhedron stores its CORE shape; the surface Havok collides with is the core
//! PLUS `hkpConvexShape::m_radius` (tag shape +0x40, 0.0164 wu on every Forge block). The native
//! `phmo_geometry` decode hands out the core only, so a Forge block's hull was 0.0164 wu smaller
//! per side than the block the player sees and the engine collides with: `ff_wall_2x2` 0.167
//! thick instead of 0.200, a capsule sunk 1.6 cm into it, a 3.3 cm crack between butted blocks.
//!
//! The settle code already pads these shapes (`physics_settle::ConvexShape::hull` -> `ShapeGeom::
//! Hull { r }`); this module turns the same padded shapes into world triangles. **The radius is
//! applied exactly once, here**, in [`hull_tris`], which `SceneController::physics_world_tris`
//! (the one source of placed-object static triangles for test mode, the vehicles and the settle)
//! calls before it falls back to the native decode.

use glam::{Mat4, Vec3};
use hms_native::{CacheHandle, NativeDll};

use crate::physics_settle::ShapeGeom;

/// Vertices of the convex polyhedron `planes` (outward normal, offset) describes.
fn plane_vertices(planes: &[(Vec3, f32)]) -> Vec<Vec3> {
    const EPS: f32 = 1e-4;
    let mut verts: Vec<Vec3> = Vec::new();
    for a in 0..planes.len() {
        for b in (a + 1)..planes.len() {
            for c in (b + 1)..planes.len() {
                let (na, nb, nc) = (planes[a].0, planes[b].0, planes[c].0);
                let det = na.dot(nb.cross(nc));
                if det.abs() < 1e-6 {
                    continue;
                }
                let p = (nb.cross(nc) * planes[a].1 + nc.cross(na) * planes[b].1 + na.cross(nb) * planes[c].1) / det;
                if !p.is_finite() || planes.iter().any(|(n, d)| n.normalize_or_zero().dot(p) > d + EPS) {
                    continue;
                }
                if !verts.iter().any(|v| v.distance_squared(p) < 1e-8) {
                    verts.push(p);
                }
            }
        }
    }
    verts
}

/// The convex hull of `pts` as outward-wound triangles (incremental hull; points inside or on
/// the hull are dropped, so the result is a closed 2-manifold whatever the input).
pub fn convex_hull(pts: &[Vec3]) -> Vec<[usize; 3]> {
    for scale in [1e-5f32, 1e-4, 1e-3, 1e-2] {
        let h = convex_hull_eps(pts, scale);
        if is_closed_manifold(&h) {
            return h;
        }
    }
    Vec::new()
}

/// Every directed edge used once and its reverse used once.
fn is_closed_manifold(h: &[[usize; 3]]) -> bool {
    if h.len() < 4 {
        return false;
    }
    let mut m: std::collections::HashMap<(usize, usize), u8> = Default::default();
    for f in h {
        for k in 0..3 {
            *m.entry((f[k], f[(k + 1) % 3])).or_default() += 1;
        }
    }
    m.iter().all(|((a, b), c)| *c == 1 && m.get(&(*b, *a)) == Some(&1))
}

fn convex_hull_eps(pts: &[Vec3], scale: f32) -> Vec<[usize; 3]> {
    let eps = scale * pts.iter().map(|p| p.abs().max_element()).fold(1.0f32, f32::max);
    let plane_of = |f: &[usize; 3]| {
        let n = (pts[f[1]] - pts[f[0]]).cross(pts[f[2]] - pts[f[0]]).normalize_or_zero();
        (n, n.dot(pts[f[0]]))
    };
    // a non-degenerate tetrahedron to start from
    let i0 = 0;
    let Some(i1) = (0..pts.len()).find(|&i| pts[i].distance(pts[i0]) > eps) else { return Vec::new() };
    let Some(i2) = (0..pts.len()).find(|&i| (pts[i] - pts[i0]).cross(pts[i1] - pts[i0]).length() > eps * eps.max(1e-3) * 10.0) else { return Vec::new() };
    let base = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0]).normalize_or_zero();
    let Some(i3) = (0..pts.len()).find(|&i| base.dot(pts[i] - pts[i0]).abs() > eps * 10.0) else { return Vec::new() };
    let mut faces: Vec<[usize; 3]> = vec![[i0, i1, i2], [i0, i2, i3], [i0, i3, i1], [i1, i3, i2]];
    let c = (pts[i0] + pts[i1] + pts[i2] + pts[i3]) * 0.25;
    for f in faces.iter_mut() {
        let (n, d) = plane_of(f);
        if n.dot(c) > d {
            f.swap(1, 2);
        }
    }
    for p in 0..pts.len() {
        if [i0, i1, i2, i3].contains(&p) {
            continue;
        }
        let visible: Vec<bool> = faces.iter().map(|f| { let (n, d) = plane_of(f); n.dot(pts[p]) - d > eps }).collect();
        if !visible.iter().any(|&v| v) {
            continue;
        }
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        for (fi, f) in faces.iter().enumerate().filter(|(i, _)| visible[*i]) {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                let shared = faces.iter().enumerate().any(|(gi, g)| visible[gi] && gi != fi && (0..3).any(|m| g[m] == b && g[(m + 1) % 3] == a));
                if !shared {
                    horizon.push((a, b));
                }
            }
        }
        let mut next: Vec<[usize; 3]> = faces.iter().enumerate().filter(|(i, _)| !visible[*i]).map(|(_, f)| *f).collect();
        next.extend(horizon.into_iter().map(|(a, b)| [a, b, p]));
        faces = next;
    }
    faces
}

/// The closed, outward-wound surface of the convex hull of `pts` padded by `r`: the true
/// Minkowski sum with a sphere, with flat bevels. Every hull triangle moves `r` along its
/// normal, every hull edge gets a quad between its two moved faces and every vertex a polygon
/// of its faces' moved copies, so a box of half extent h becomes one of h + r and an acute
/// ramp corner does NOT grow a spike (offsetting the planes would: r / sin(half angle)).
pub fn padded_hull(pts: &[Vec3], r: f32) -> Vec<[Vec3; 3]> {
    let faces = convex_hull(pts);
    if faces.is_empty() {
        return flat_prism(pts, r);
    }
    let mut out: Vec<[Vec3; 3]> = Vec::new();
    let mut push = |t: [Vec3; 3], toward: Vec3| {
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        if n.length_squared() == 0.0 {
            return;
        }
        out.push(if n.dot(toward) >= 0.0 { t } else { [t[0], t[2], t[1]] });
    };
    let mut edge_face: std::collections::HashMap<(usize, usize), usize> = Default::default();
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            edge_face.insert((f[k], f[(k + 1) % 3]), fi);
        }
    }
    // One normal per PLANE: triangles that are neighbours in the same plane (a cap fan, a
    // rectangle's two halves) get bit-identical normals, so their bevel quads are exactly
    // degenerate and their moved copies exactly coincide -- no sliver surfaces, no float gaps.
    let raw: Vec<Vec3> = faces.iter().map(|f| (pts[f[1]] - pts[f[0]]).cross(pts[f[2]] - pts[f[0]]).normalize_or_zero()).collect();
    let mut rep: Vec<usize> = (0..faces.len()).collect();
    fn find(rep: &mut Vec<usize>, mut i: usize) -> usize {
        while rep[i] != i {
            rep[i] = rep[rep[i]];
            i = rep[i];
        }
        i
    }
    for (fi, f) in faces.iter().enumerate() {
        for k in 0..3 {
            if let Some(&g) = edge_face.get(&(f[(k + 1) % 3], f[k])) {
                if raw[fi].dot(raw[g]) > 1.0 - 1e-6 {
                    let (a, b) = (find(&mut rep, fi), find(&mut rep, g));
                    rep[a.max(b)] = a.min(b);
                }
            }
        }
    }
    let nrm: Vec<Vec3> = (0..faces.len()).map(|i| raw[find(&mut rep, i)]).collect();
    let normal = |f: usize| nrm[f];
    for (fi, f) in faces.iter().enumerate() {
        let n = normal(fi);
        push([pts[f[0]] + n * r, pts[f[1]] + n * r, pts[f[2]] + n * r], n);
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            // each undirected edge once: from the face that holds it as (low, high)
            if a < b {
                if let Some(&g) = edge_face.get(&(b, a)) {
                    let m = normal(g);
                    let q = [pts[a] + n * r, pts[b] + n * r, pts[b] + m * r, pts[a] + m * r];
                    push([q[0], q[1], q[2]], n + m);
                    push([q[0], q[2], q[3]], n + m);
                }
            }
        }
    }
    // vertex polygons: walk the faces around each vertex through the shared edges
    for v in 0..pts.len() {
        let Some(start) = faces.iter().position(|f| f.contains(&v)) else { continue };
        let mut ring: Vec<usize> = Vec::new();
        let mut fi = start;
        loop {
            ring.push(fi);
            let f = &faces[fi];
            let k = f.iter().position(|&x| x == v).unwrap();
            let x = f[(k + 1) % 3];
            let Some(&g) = edge_face.get(&(x, v)) else { break };
            if g == start || ring.len() > faces.len() {
                break;
            }
            fi = g;
        }
        if ring.len() < 3 {
            continue;
        }
        let axis = ring.iter().map(|&f| normal(f)).sum::<Vec3>().normalize_or_zero();
        let p = |f: usize| pts[v] + normal(f) * r;
        for k in 1..ring.len() - 1 {
            push([p(ring[0]), p(ring[k]), p(ring[k + 1])], axis);
        }
    }
    out
}

/// A FLAT polyhedron (a polygon with no thickness: a Forge door panel's collision sheet) padded
/// by `r`: the convex polygon moved `r` either way along its normal, with a quad per side. Empty
/// for anything with no area (a point or a segment).
/// The cross product of the largest triangle among `pts` taken off the first point: the normal
/// of their best plane, its length twice the area (zero for a point or a segment).
fn largest_cross(pts: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 1..pts.len() {
        for j in (i + 1)..pts.len() {
            let c = (pts[i] - pts[0]).cross(pts[j] - pts[0]);
            if c.length_squared() > n.length_squared() {
                n = c;
            }
        }
    }
    n
}

fn flat_prism(pts: &[Vec3], r: f32) -> Vec<[Vec3; 3]> {
    let n = largest_cross(pts);
    if n.length() < 1e-4 {
        return Vec::new();
    }
    let n = n.normalize();
    let u = (pts[1] - pts[0]).normalize_or_zero();
    let u = if u.cross(n).length() < 1e-6 { n.any_orthonormal_vector() } else { (u - n * u.dot(n)).normalize() };
    let w = n.cross(u);
    // 2-D convex hull (monotone chain) of the projected points
    let mut p2: Vec<(f32, f32, usize)> = pts.iter().enumerate().map(|(i, p)| ((*p - pts[0]).dot(u), (*p - pts[0]).dot(w), i)).collect();
    p2.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let cross = |o: &(f32, f32, usize), a: &(f32, f32, usize), b: &(f32, f32, usize)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut hull: Vec<(f32, f32, usize)> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let it: Box<dyn Iterator<Item = &(f32, f32, usize)>> = if pass == 0 { Box::new(p2.iter()) } else { Box::new(p2.iter().rev()) };
        for q in it {
            while hull.len() >= start + 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], q) <= 1e-7 {
                hull.pop();
            }
            hull.push(*q);
        }
        hull.pop();
    }
    if hull.len() < 3 {
        return Vec::new();
    }
    let ring: Vec<Vec3> = hull.iter().map(|h| pts[h.2]).collect();
    let mut out = Vec::new();
    let mut push = |t: [Vec3; 3], toward: Vec3| {
        let c = (t[1] - t[0]).cross(t[2] - t[0]);
        if c.length_squared() > 0.0 {
            out.push(if c.dot(toward) >= 0.0 { t } else { [t[0], t[2], t[1]] });
        }
    };
    let centre = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
    for k in 1..ring.len() - 1 {
        push([ring[0] + n * r, ring[k] + n * r, ring[k + 1] + n * r], n);
        push([ring[0] - n * r, ring[k] - n * r, ring[k + 1] - n * r], -n);
    }
    for k in 0..ring.len() {
        let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
        let side = ((a + b) * 0.5 - centre).normalize_or_zero();
        push([a + n * r, b + n * r, b - n * r], side);
        push([a + n * r, b - n * r, a - n * r], side);
    }
    out
}

/// Triangles of the convex hull `planes` (outward normal, offset) padded by `r` -- see
/// [`padded_hull`].
pub fn hull_triangles(planes: &[(Vec3, f32)], r: f32) -> Vec<[Vec3; 3]> {
    padded_hull(&plane_vertices(planes), r)
}

/// The raw phmo blocks through the native DLL's tag reads.
pub fn read_blocks(native: &NativeDll, cache: CacheHandle, phmo: u32) -> Option<crate::phmo_shapes::PhmoBlocks> {
    let meta = native.tag_meta(cache, phmo, 0, 0x19C)?;
    let rd = |o: usize| u32::from_le_bytes([meta[o], meta[o + 1], meta[o + 2], meta[o + 3]]);
    Some(crate::phmo_shapes::PhmoBlocks::read(&crate::phmo_shapes::REACH, |off, elem| {
        let n = rd(off) as i32;
        let ptr = rd(off + 4);
        if n <= 0 || n > 0x10000 || ptr == 0 || ptr == 0xFFFF_FFFF { return Vec::new(); }
        native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32).unwrap_or_default()
    }))
}

/// The object's `mode` node -> model matrices (rigid bodies are in node space).
pub fn mode_node_xforms(native: &NativeDll, cache: CacheHandle, obj_tag: u32) -> Vec<Mat4> {
    let mode = native.resolve_mode(cache, obj_tag);
    if mode == 0 || mode == 0xFFFF_FFFF { return Vec::new(); }
    let Some(h) = native.tag_meta(cache, mode, crate::phmo_shapes::OFF_MODE_NODES as u32, 12) else { return Vec::new() };
    let n = i32::from_le_bytes([h[0], h[1], h[2], h[3]]);
    let ptr = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
    if n <= 0 || n > 1024 || ptr == 0 || ptr == 0xFFFF_FFFF { return Vec::new(); }
    let elem = crate::phmo_shapes::REACH_MODE_NODE_ELEM;
    match native.tag_ptr_bytes(cache, ptr, (n as usize * elem) as u32) {
        Some(b) => crate::phmo_shapes::node_xforms(&b, elem),
        None => Vec::new(),
    }
}

/// Model-space triangles (unscaled; the caller applies the object's placement matrix) of a
/// physics model made ONLY of boxes and polyhedra, radius included, ONE CLOSED SHELL PER SHAPE
/// (every triangle wound outward). `None` for anything else (spheres, pills, triangle meshes):
/// those keep the native decode and have no known inside.
#[cfg(test)]
pub fn hull_points(blocks: &crate::phmo_shapes::PhmoBlocks, nodes: &[Mat4]) -> Vec<(Vec<Vec3>, f32)> {
    let (defs, _) = crate::phmo_shapes::rigid_bodies(blocks, &crate::phmo_shapes::REACH, nodes);
    defs.iter().flat_map(|d| d.shapes.iter()).filter_map(|sh| if let ShapeGeom::Hull { pts, r, .. } = &sh.geom { Some((pts.clone(), *r)) } else { None }).collect()
}

pub fn hull_shells(blocks: &crate::phmo_shapes::PhmoBlocks, nodes: &[Mat4]) -> Option<Vec<Vec<[Vec3; 3]>>> {
    let (defs, counts) = crate::phmo_shapes::rigid_bodies(blocks, &crate::phmo_shapes::REACH, nodes);
    if counts.spheres + counts.pills + counts.triangles != 0 || counts.boxes + counts.polyhedra == 0 {
        return None;
    }
    let mut out = Vec::new();
    for d in &defs {
        for sh in &d.shapes {
            let ShapeGeom::Hull { pts, planes, r } = &sh.geom else { return None };
            // above the plane limit the settle planes are an AABB stand-in, not the real faces
            if planes.len() < 4 || pts.len() > crate::physics_settle::HULL_PLANE_LIMIT {
                return None;
            }
            let t = padded_hull(pts, *r);
            if !t.is_empty() {
                out.push(t);
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

/// [`hull_shells`], flattened.
pub fn hull_tris(blocks: &crate::phmo_shapes::PhmoBlocks, nodes: &[Mat4]) -> Option<Vec<[Vec3; 3]>> {
    hull_shells(blocks, nodes).map(|s| s.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_planes(h: Vec3) -> Vec<(Vec3, f32)> {
        vec![(Vec3::X, h.x), (Vec3::NEG_X, h.x), (Vec3::Y, h.y), (Vec3::NEG_Y, h.y), (Vec3::Z, h.z), (Vec3::NEG_Z, h.z)]
    }

    fn extent(t: &[[Vec3; 3]]) -> (Vec3, Vec3) {
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in t.iter().flatten() {
            mn = mn.min(*v);
            mx = mx.max(*v);
        }
        (mn, mx)
    }

    /// #collision  **THE RADIUS IS PUT BACK.** A box core of half extents (0.4836, 0.4836, 0.0836)
    /// -- `ff_plat_1x1_flat`'s stored hull -- with the 0.0164 convex radius is the 1.0 x 1.0 x
    /// 0.2 block the player sees; the same box without it is the 0.967 x 0.967 x 0.167 core.
    /// Twelve triangles, wound outward.
    ///
    /// SABOTAGE: drop the `+ r` in `hull_triangles` -> the extent assertion goes red.
    #[test]
    fn a_box_core_plus_the_convex_radius_is_the_authored_block() {
        let t = hull_triangles(&box_planes(Vec3::new(0.4836, 0.4836, 0.0836)), 0.0164);
        assert!(t.len() >= 12);
        let (mn, mx) = extent(&t);
        assert!((mx - mn - Vec3::new(1.0, 1.0, 0.2)).abs().max_element() < 1e-4, "extent {:?}", mx - mn);
        for tri in &t {
            let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
            let c = (tri[0] + tri[1] + tri[2]) / 3.0;
            assert!(n.dot(c) > 0.0, "a face is wound inward");
        }
        let core = hull_triangles(&box_planes(Vec3::new(0.4836, 0.4836, 0.0836)), 0.0);
        let (a, b) = extent(&core);
        assert!((b - a).x < 0.97, "the core alone is the small hull: {:?}", b - a);
    }

    fn open() -> Option<(std::sync::MutexGuard<'static, ()>, std::sync::Arc<NativeDll>, CacheHandle)> {
        crate::reach_actor::tests::open_reach_for("forge_halo")
    }

    fn mesh_extent(g: &(Vec<f32>, Vec<u32>)) -> Option<(Vec3, Vec3)> {
        let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &k in &g.1 {
            let k = k as usize * 3;
            if k + 2 >= g.0.len() { return None; }
            let v = Vec3::new(g.0[k], g.0[k + 1], g.0[k + 2]);
            mn = mn.min(v);
            mx = mx.max(v);
        }
        (!g.1.is_empty()).then_some((mn, mx))
    }

    /// #collision  **EVERY PLACED BLOCK'S HULL IS THE AUTHORED SIZE.** Over all 551 objects of
    /// forge_halo that have both a physics model and a collision model: the share whose hull
    /// extents equal the collision model's within 5 mm per axis, with the native zero-radius
    /// decode (BEFORE) and with `hull_tris` (AFTER). A butted pair of `ff_plat_1x1_flat` at a 1.0
    /// pitch has no crack after, a 3.3 cm one before.
    ///
    /// SABOTAGE: `hull_triangles(planes, 0.0)` in `hull_tris` -> the AFTER share collapses and
    /// the crack returns (red).
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn every_placed_block_hull_is_the_authored_size() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let ext = |t: &[[Vec3; 3]]| extent(t);
        let (mut measured, mut before_ok, mut after_ok, mut replaced) = (0, 0, 0, 0);
        let (mut fg, mut fg_before, mut fg_after) = (0, 0, 0);
        let mut plat: Option<Vec<[Vec3; 3]>> = None;
        let mut worst: Vec<(f32, String)> = Vec::new();
        for group in ["bloc", "scen", "mach", "ctrl", "vehi", "crea"] {
            for &t in &native.tags_by_group(cache, group) {
                let name = native.tag_name(cache, t).unwrap_or_default();
                let ph = native.resolve_phmo(cache, t);
                let co = native.resolve_coll(cache, t);
                if ph == 0 || ph == 0xFFFF_FFFF || co == 0 || co == 0xFFFF_FFFF { continue; }
                let (Some(pg), Some(cg)) = (native.phmo_geometry(cache, ph), native.coll_geometry(cache, co)) else { continue };
                let (Some(pe), Some(ce)) = (mesh_extent(&pg), mesh_extent(&cg)) else { continue };
                let hulls = read_blocks(native, cache, ph).and_then(|b| hull_tris(&b, &mode_node_xforms(native, cache, t)));
                let close = |a: (Vec3, Vec3)| (a.0 - ce.0).abs().max_element() < 0.005 && (a.1 - ce.1).abs().max_element() < 0.005;
                measured += 1;
                let before = close(pe);
                before_ok += before as usize;
                let after = match &hulls {
                    Some(h) => { replaced += 1; close(ext(h)) }
                    None => before,
                };
                after_ok += after as usize;
                if name.contains("levels\\forge") { fg += 1; fg_before += before as usize; fg_after += after as usize; }
                if name.ends_with("ff_plat_1x1_flat") { plat = hulls.clone(); }
                if let (Some(h), false) = (&hulls, after) {
                    let e = ext(h);
                    worst.push(((e.0 - ce.0).abs().max_element().max((e.1 - ce.1).abs().max_element()), format!("{name}  hull {:?}..{:?} coll {:?}..{:?}", e.0, e.1, ce.0, ce.1)));
                }
            }
        }
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("{measured} objects with both models: hull == collision extents (5 mm) BEFORE {before_ok}, AFTER {after_ok}; {replaced} hulls rebuilt with the radius");
        println!("   of those, Forge blocks (objects\\levels\\forge): {fg}, BEFORE {fg_before}, AFTER {fg_after}");
        assert!(fg > 100 && fg_after > fg_before + 60, "Forge blocks {fg}: {fg_before} -> {fg_after}");
        for (d, n) in worst.iter().filter(|w| w.1.contains("levels\\forge")).take(25) { println!("   still off by {d:.3}  {n}"); }
        assert!(measured > 400, "liveness: only {measured} objects measured");
        assert!(replaced > 100, "liveness: only {replaced} hulls rebuilt");
        assert!(after_ok > before_ok + 50, "the radius did not fix the block sizes: {before_ok} -> {after_ok}");
        let plat = plat.expect("ff_plat_1x1_flat has a boxes-only phmo");
        let (a, b) = ext(&plat);
        let shifted: Vec<[Vec3; 3]> = plat.iter().map(|t| t.map(|p| p + Vec3::X)).collect();
        let (c, _) = ext(&shifted);
        println!("ff_plat_1x1_flat hull x [{:.4}, {:.4}]; butted at pitch 1.0 the neighbour starts {:.4}", a.x, b.x, c.x);
        assert!(b.x - c.x > -1e-4, "crack of {:.4} wu between butted blocks", c.x - b.x);
    }

    /// Edge incidence of a shell: (edges used by exactly two triangles, edges used otherwise).
    fn edge_balance(t: &[[Vec3; 3]]) -> (usize, usize) {
        let key = |p: Vec3| (p.x.to_bits() as i64, p.y.to_bits() as i64, p.z.to_bits() as i64);
        let mut m: std::collections::HashMap<((i64, i64, i64), (i64, i64, i64)), i32> = Default::default();
        for tr in t {
            for k in 0..3 {
                let (a, b) = (key(tr[k]), key(tr[(k + 1) % 3]));
                if a == b {
                    continue;
                }
                *m.entry(if a < b { (a, b) } else { (b, a) }).or_default() += 1;
            }
        }
        let good = m.values().filter(|&&c| c == 2).count();
        (good, m.len() - good)
    }

    /// #collision  **THE SHELLS ARE WATERTIGHT.** The inside test (parity) and the nearest-exit
    /// search both assume each Forge shell is a closed surface: every edge shared by exactly two
    /// triangles. Counted over every shell of every Forge block.
    #[test]
    #[ignore = "opens the native library; run alone with --ignored"]
    fn probe_the_forge_shells_are_watertight() {
        let Some((_g, native, cache)) = open() else { return };
        let native = &*native;
        let (mut shells, mut open_shells, mut missing, mut worst) = (0, 0, 0usize, Vec::new());
        for &tag in native.tags_by_group(cache, "bloc").iter() {
            let name = native.tag_name(cache, tag).unwrap_or_default();
            let ph = native.resolve_phmo(cache, tag);
            if !name.contains("levels\\forge") || ph == 0 || ph == 0xFFFF_FFFF {
                continue;
            }
            let Some(blk) = read_blocks(native, cache, ph) else { continue };
            let Some(sh) = hull_shells(&blk, &mode_node_xforms(native, cache, tag)) else { continue };
            for (pts, r) in hull_points(&blk, &mode_node_xforms(native, cache, tag)) {
                if padded_hull(&pts, r).is_empty() && largest_cross(&pts).length() >= 1e-4 {
                    missing += 1;
                    let (mut mn, mut mx) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                    for p in &pts { mn = mn.min(*p); mx = mx.max(*p); }
                    println!("   NOHULL {name}: {} pts, extent {:?}, hull faces at 1e-2: {}", pts.len(), mx - mn, convex_hull_eps(&pts, 1e-2).len());
                }
            }
            for s in &sh {
                shells += 1;
                let (good, bad) = edge_balance(s);
                if bad > 0 {
                    open_shells += 1;
                    worst.push((bad, good, s.len(), name.clone()));
                }
            }
        }
        worst.sort();
        worst.reverse();
        println!("{shells} shells, {open_shells} not watertight, {missing} hull shapes that produced no shell");
        for w in worst.iter().take(10) {
            println!("   {} bad edges / {} good, {} tris  {}", w.0, w.1, w.2, w.3);
        }
        assert!(shells > 100);
        assert_eq!((open_shells, missing), (0, 0));
    }
}
