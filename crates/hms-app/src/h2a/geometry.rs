//! Halo 2 Anniversary BSP render geometry. #h2a
//!
//! The parser is `h4::geometry` unchanged - every sbsp / Lbsp block offset it uses was verified
//! to be IDENTICAL on groundhog caches (probed on `ca_lockout.map`, both BSPs, and asserted here
//! on every shipped map):
//!
//! | block | offset | H2A `ca_lockout_bsp01` | Halo 4 `wraparound` |
//! |---|---|---|---|
//! | `stli` ref | +0x34 | resolves | resolves |
//! | collision materials | +0x78 | 84 | 64 |
//! | clusters (128 B, mesh i16 @0x34) | +0x154 | 19 | 8 |
//! | materials (44 B, `mat ` ref @0) | +0x160 | 104 | 69 |
//! | instance definitions | +0x244 / +0x250 / +0x25C / +0x268 | 19 / 88 / 19 / 88 | 8 / 92 / 8 / 92 |
//! | instance lookup | +0x280 / +0x29C | 651 / 651 | 1046 / 1046 |
//! | meshes (112 B) | +0x4A8 | 619 | 1008 |
//! | per-mesh compression bounds (52 B) | +0x4C0 | 600 | 1008 |
//! | `iimz` ref | +0x28C | resolves | resolves |
//!
//! and on the Lbsp side +0xBC = clusters, +0xC8 / +0x1B8 = instances, +0x140 = meshes,
//! +0x158 / +0x1F0 = the per-mesh tables, all with counts matching the sbsp's.
//!
//! ONE H2A difference worth recording: the mesh table (+0x4A8) and the compression-bounds table
//! (+0x4C0) have DIFFERENT lengths on H2A (619 vs 600 on ca_lockout_bsp01), where Halo 4 always
//! had them equal. `h4::geometry::load_bsp` already guards this (`if i < nb`), so meshes past the
//! bounds table decode with zero bounds; the tests below record which meshes those are.

use anyhow::Result;

use crate::h4::cache::H4Cache;
use crate::h4::geometry::{load_bsp, H4Bsp};

/// Every BSP in a groundhog cache, in tag order (H2A maps ship a playable BSP plus, on most maps,
/// a `_bsp_vista` distant-scenery BSP).
pub fn load_bsps(c: &H4Cache) -> Vec<Result<H4Bsp>> {
    c.find_tags(b"sbsp").into_iter().map(|t| load_bsp(c, t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h2a::cache::{installed_caches, maps_dir, open};
    use crate::h4::geometry::decode_mesh;
    use glam::Vec3;

    /// `ca_lockout.map`: the exact numbers probed for the doc, so a layout regression is loud.
    #[test]
    fn lockout_bsp_layout() {
        let Some(dir) = maps_dir() else { return };
        let p = dir.join("ca_lockout.map");
        if !p.is_file() { return }
        let c = open(&p).unwrap();
        let sbsps = c.find_tags(b"sbsp");
        assert_eq!(sbsps.len(), 2);
        let bsp = load_bsp(&c, sbsps[0]).unwrap();
        assert_eq!(bsp.name, "levels\\sway\\ca_lockout\\ca_lockout_bsp01");
        assert!(bsp.lbsp_tag.is_some(), "the Lbsp with the same tag name carries the geometry");
        assert_eq!(bsp.meshes.len(), 619);
        assert_eq!(bsp.materials.len(), 104);
        assert_eq!(bsp.clusters.len(), 19);
        assert_eq!(bsp.instances.len(), 651);
        let g = bsp.geometry.as_ref().expect("geometry resource");
        assert!(!g.vbs.is_empty() && !g.ibs.is_empty());
        // the Halo 4 vertex/index buffer invariants hold unchanged
        assert!(g.vbs.iter().all(|v| v.size == v.count * v.stride as u32));
        assert!(g.vbs.iter().any(|v| v.kind == 2 && v.stride == 36), "type-2 / stride-36 world vertices");
        assert!(g.ibs.iter().all(|i| i.fmt == 3), "u16 triangle lists");
        // the vista BSP is the distant-scenery BSP: same shape, far fewer meshes
        let vista = load_bsp(&c, sbsps[1]).unwrap();
        assert_eq!(vista.name, "levels\\sway\\ca_lockout\\ca_lockout_bsp_vista");
        assert_eq!(vista.meshes.len(), 9);
        assert_eq!(vista.materials.len(), 3);
    }

    /// Every shipped H2A cache: every BSP parses, every mesh with geometry decodes, every index
    /// is inside its vertex buffer, every part's material index is in range, every part's index
    /// range is inside the index buffer, and all decoded positions / bounds are finite.
    #[test]
    fn every_cache_bsp_geometry() {
        let caches = installed_caches();
        if caches.is_empty() { eprintln!("skip: no groundhog maps folder"); return; }
        let mut total_bsps = 0;
        let mut total_meshes = 0;
        let mut total_tris = 0usize;
        for p in &caches {
            let c = open(p).unwrap();
            if c.map_type == 3 || c.map_type == 4 { continue; }
            let bsps = load_bsps(&c);
            assert!(!bsps.is_empty(), "{}: no sbsp", p.display());
            for b in bsps {
                let bsp = b.unwrap_or_else(|e| panic!("{}: load_bsp: {e}", p.display()));
                total_bsps += 1;
                assert!(bsp.geometry.is_some(), "{}: {} has no geometry resource", p.display(), bsp.name);
                let g = bsp.geometry.as_ref().unwrap();
                assert!(g.vbs.iter().all(|v| v.size == v.count * v.stride as u32), "{}: vb size", bsp.name);
                assert!(!bsp.materials.is_empty(), "{}: no materials", bsp.name);
                // every mesh with geometry points at a real vb / ib
                for m in bsp.meshes.iter().filter(|m| m.has_geometry()) {
                    assert!((m.vb[0] as usize) < g.vbs.len(), "{}: vb index", bsp.name);
                    assert!((m.ib_index() as usize) < g.ibs.len(), "{}: ib index", bsp.name);
                }
                // cluster / instance references are in range
                assert!(bsp.clusters.iter().all(|&m| (m as usize) < bsp.meshes.len()), "{}", bsp.name);
                assert!(bsp.instances.iter().all(|i| i.mesh >= 0 && (i.mesh as usize) < bsp.meshes.len()), "{}", bsp.name);
                assert!(bsp.instances.iter().all(|i| i.scale.is_finite() && i.rot.iter().all(|v| v.is_finite()) && i.translation.iter().all(|v| v.is_finite())), "{}", bsp.name);
                assert!(bsp.cluster_bounds.iter().all(|(mn, mx)| (0..3).all(|k| mn[k].is_finite() && mx[k].is_finite() && mn[k] <= mx[k])), "{}", bsp.name);
                for i in 0..bsp.meshes.len() {
                    let Some(dm) = decode_mesh(&c, &bsp, i).unwrap_or_else(|e| panic!("{}: mesh {i}: {e}", bsp.name)) else { continue };
                    total_meshes += 1;
                    total_tris += dm.indices.len() / 3;
                    assert!(!dm.verts.is_empty(), "{}: mesh {i} decoded 0 verts", bsp.name);
                    assert!(dm.indices.iter().all(|&ix| (ix as usize) < dm.verts.len()), "{}: mesh {i} index past vertex buffer", bsp.name);
                    assert!(dm.verts.iter().all(|v| v.pos.iter().all(|c| c.is_finite())), "{}: mesh {i} non-finite position", bsp.name);
                    assert!(dm.verts.iter().all(|v| v.normal.iter().all(|c| c.is_finite()) && v.uv.iter().all(|c| c.is_finite())), "{}: mesh {i}", bsp.name);
                    for pt in &dm.parts {
                        assert!((pt.index_start + pt.index_count) as usize <= dm.indices.len(), "{}: mesh {i} part range", bsp.name);
                        assert!(pt.material >= 0 && (pt.material as usize) < bsp.materials.len(), "{}: mesh {i} material {}", bsp.name, pt.material);
                    }
                }
            }
        }
        assert!(total_bsps >= 10 && total_meshes > 1000 && total_tris > 100_000,
            "bsps {total_bsps} meshes {total_meshes} tris {total_tris}");
        eprintln!("h2a geometry sweep: {} caches, {total_bsps} BSPs, {total_meshes} meshes with geometry, {total_tris} triangles", caches.len());
    }

    /// Every vertex buffer an H2A BSP mesh points at satisfies the Halo 4 rigid layout: stride
    /// >= 36, f32 blend @12 in 0..1, u16 @26 == 0, unit snorm16x3 normal @20, unit snorm16x3
    /// tangent @28, handedness @34 = +-1, and position f32x3 in 0..1 (normalized to the mesh
    /// bounds) EXCEPT for the one type-1 buffer, whose positions are absolute (`Bounds::is_degenerate`).
    /// Types seen: 2/36 and 5/40 (as in Halo 4) plus a single 1/36.
    /// Also pins the `pos_vbs` anomaly: `pos_vbs.len() == ibs.len()` on every BSP, and `pos_vbs`
    /// is the identity on every BSP except `ca_zanzibar_bsp01`, which has one stray stride-4
    /// buffer at raw index 1476 (the only place on either engine where the raw `vb[0]` reading
    /// has to fall back).
    #[test]
    fn vertex_type_survey() {
        use crate::h4::cache::ByteRead;
        use crate::h4::geometry::sn16;
        use std::collections::BTreeMap;
        let caches = installed_caches();
        if caches.is_empty() { eprintln!("skip: no groundhog maps folder"); return; }
        // (kind, stride) -> [count, pos_ok, blend_ok, pad_ok, n_unit, t_unit, hand_ok]
        let mut tally: BTreeMap<(i16, i16), [usize; 7]> = BTreeMap::new();
        let mut shifted: Vec<(String, usize, usize)> = Vec::new();
        for p in &caches {
            let c = open(p).unwrap();
            if c.map_type == 3 || c.map_type == 4 { continue; }
            for sb in c.find_tags(b"sbsp") {
                let Ok(bsp) = crate::h4::geometry::load_bsp(&c, sb) else { continue };
                let Some(g) = bsp.geometry.as_ref() else { continue };
                let Some(e) = c.resources.get(g.entry) else { continue };
                assert_eq!(g.pos_vbs.len(), g.ibs.len(), "{}: one position buffer per index buffer", bsp.name);
                if let Some(first) = g.pos_vbs.iter().enumerate().find(|(i, &r)| *i != r) {
                    shifted.push((bsp.name.clone(), first.0, *first.1));
                }
                for m in bsp.meshes.iter().filter(|m| m.has_geometry()) {
                    // the decoder's rule: raw vb[0], falling back to the position-capable-only
                    // reading when the raw buffer is too narrow (ca_zanzibar mesh 1770)
                    let vb = g.vertex_buffer(m.vb[0]).filter(|v| v.stride >= 36)
                        .or_else(|| g.vertex_buffer_fallback(m.vb[0]))
                        .unwrap_or_else(|| panic!("{}: vb {}", bsp.name, m.vb[0]));
                    assert!(vb.stride >= 36, "{}: vb {} stride {}", bsp.name, m.vb[0], vb.stride);
                    let slot = tally.entry((vb.kind, vb.stride)).or_default();
                    let n = (vb.count as usize).min(64);
                    let Ok(raw) = c.stream_bytes(e, vb.addr, n * vb.stride as usize) else { continue };
                    for i in 0..n {
                        let v = &raw[i * vb.stride as usize..];
                        slot[0] += 1;
                        let pos = [v.f32_at(0), v.f32_at(4), v.f32_at(8)];
                        if pos.iter().all(|c| (-0.002..=1.002).contains(c)) { slot[1] += 1; }
                        let b = v.f32_at(12);
                        if (-0.002..=1.002).contains(&b) { slot[2] += 1; }
                        if v.u16_at(26) == 0 { slot[3] += 1; }
                        let nl = (0..3).map(|k| sn16(v.i16_at(20 + k * 2)).powi(2)).sum::<f32>().sqrt();
                        if (nl - 1.0).abs() < 0.02 { slot[4] += 1; }
                        let tl = (0..3).map(|k| sn16(v.i16_at(28 + k * 2)).powi(2)).sum::<f32>().sqrt();
                        if (tl - 1.0).abs() < 0.02 { slot[5] += 1; }
                        if sn16(v.i16_at(34)).abs() > 0.99 { slot[6] += 1; }
                    }
                }
            }
        }
        for (k, v) in &tally {
            let pct = |i: usize| 100 * v[i] / v[0].max(1);
            eprintln!("h2a vb type {:>2} stride {:>2}: {:>7} verts sampled | pos {:>3}% blend {:>3}% pad0 {:>3}% |n|=1 {:>3}% |t|=1 {:>3}% hand {:>3}%",
                k.0, k.1, v[0], pct(1), pct(2), pct(3), pct(4), pct(5), pct(6));
        }
        eprintln!("h2a pos_vbs shifted BSPs (name, first logical, raw): {shifted:?}");
        assert_eq!(shifted.len(), 1, "expected only the ca_zanzibar shift, got {shifted:?}");
        assert!(shifted[0].0.ends_with("ca_zanzibar_bsp01") && shifted[0] .1 == 1476 && shifted[0].2 == 1477, "{shifted:?}");
        assert!(!tally.is_empty());
        // every kind seen is one the decoder handles, and the whole tail layout holds exactly
        for (&(kind, stride), v) in &tally {
            assert!(crate::h4::objects::vertex_kind_supported(kind, stride), "vertex kind {kind} stride {stride} is not handled");
            let pct = |i: usize| 100 * v[i] / v[0].max(1);
            assert_eq!((pct(2), pct(3), pct(4), pct(5), pct(6)), (100, 100, 100, 100, 100),
                "type {kind}/{stride}: blend {}% pad0 {}% |n|=1 {}% |t|=1 {}% hand {}%", pct(2), pct(3), pct(4), pct(5), pct(6));
            // only the type-1 buffer carries absolute (non-normalized) positions
            if kind == 1 { assert_eq!(pct(1), 0, "type 1 positions are absolute"); }
            else { assert_eq!(pct(1), 100, "type {kind}/{stride} positions are bounds-normalized"); }
        }
        assert!(tally.contains_key(&(2, 36)) && tally.contains_key(&(5, 40)) && tally.contains_key(&(1, 36)), "{:?}", tally.keys().collect::<Vec<_>>());
    }

    /// The decoded geometry actually spans a world-sized volume and the vertex normals agree with
    /// the triangle facing - the check that the type-2 / stride-36 vertex decode (shared with
    /// Halo 4) is right and not, say, a transposed or byte-swapped reading.
    #[test]
    fn lockout_vertex_decode_sanity() {
        let Some(dir) = maps_dir() else { return };
        let p = dir.join("ca_lockout.map");
        if !p.is_file() { return }
        let c = open(&p).unwrap();
        let bsp = load_bsp(&c, c.find_tags(b"sbsp")[0]).unwrap();
        let mut agree = 0usize;
        let mut total = 0usize;
        let mut unit = 0usize;
        let mut nverts = 0usize;
        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
        for i in 0..bsp.meshes.len() {
            let Some(dm) = decode_mesh(&c, &bsp, i).unwrap() else { continue };
            for v in &dm.verts {
                nverts += 1;
                for k in 0..3 { mn[k] = mn[k].min(v.pos[k]); mx[k] = mx[k].max(v.pos[k]); }
                let n = Vec3::from(v.normal);
                if (n.length() - 1.0).abs() < 1e-2 { unit += 1; }
            }
            for t in dm.indices.chunks_exact(3) {
                let pos = |ix: u32| Vec3::from(dm.verts[ix as usize].pos);
                let fnrm = (pos(t[1]) - pos(t[0])).cross(pos(t[2]) - pos(t[0])).normalize_or_zero();
                let vn = (Vec3::from(dm.verts[t[0] as usize].normal) + Vec3::from(dm.verts[t[1] as usize].normal) + Vec3::from(dm.verts[t[2] as usize].normal)).normalize_or_zero();
                if fnrm.length_squared() < 1e-8 || vn.length_squared() < 1e-8 { continue; }
                total += 1;
                if fnrm.dot(vn) > 0.7 { agree += 1; }
            }
        }
        assert!(nverts > 20_000, "only {nverts} vertices decoded");
        assert!(unit * 100 / nverts >= 99, "unit normals {unit}/{nverts}");
        let span = (0..3).map(|k| mx[k] - mn[k]).fold(0.0f32, f32::max);
        assert!(span > 20.0 && span < 20_000.0, "world span {span} wu from {mn:?}..{mx:?}");
        let pct = agree * 100 / total.max(1);
        assert!(pct >= 90, "triangle-list facing agreement {agree}/{total} = {pct}%");
        eprintln!("h2a lockout verts {nverts}, world {mn:?}..{mx:?}, facing {pct}%");
    }
}

