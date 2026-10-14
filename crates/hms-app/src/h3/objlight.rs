//! Halo 3 object lighting: the per-object light probe and its GPU lanes. #h3-objlight
//!
//! A Halo 3 dynamic object (weapon, vehicle, scenery, Forge piece) has no lightmap of its own:
//! the engine lights it through the `static_sh` entry point of its shader (Halo 3 HLSL
//! `entry_points_fx.hlsl` `static_sh_ps`, protomorph `entry_static_sh.wgsl`) from ONE spherical
//! harmonic probe (`p_lighting_constant_0..3`: DC + the three per-channel L1 vectors) plus the
//! probe's dominant light (`k_ps_dominant_light_direction` / `_intensity`):
//!   diffuse_radiance = ravi_order_2(N, sh)   (spherical_harmonics_fx: (0.886227 sh0 - 1.023328 N.l1) / pi)
//!   analytical light = the dominant light (specular only: the SH already holds its diffuse)
//!   area specular   = sh_glossy(view, N, sh)  (cook_torrance_fx sh_glossy_ct_2 / _3)
//! The probe itself is the structure's lightmap sampled at the object's lighting point: the
//! lightprobe texture array texel of the lit surface under the object (the same DXT5 pair decode
//! the per-pixel BSP shader uses, lightmaps.rs `probe_at`), so an object under a bridge is lit
//! like the floor beneath it and one in the sun like the sunlit ground - the classic Halo 3
//! look. `H3LightField` keeps every lightmapped triangle (cluster meshes + per-pixel-lit
//! instances, world space, with their lightmap UVs) in an XY grid; `sample` drops a vertical ray
//! from the object origin, takes the nearest lit surface below it (else the nearest above, else
//! the map's exposed mean bake as a flat probe) and reads the texel at the barycentric UV, scaled
//! by that BSP's bake gain (the auto-exposure stand-in every lit lane carries).
//!
//! The probe rides the renderer's per-instance object lanes (`H3ObjProbe::lanes`): the dominant
//! direction + intensity in `obj_probe0/1`, the SH DC in the instance colour, the nine L1 floats
//! packed as f16 pairs in `obj_light` (+ the ninth in the colour alpha), and mesh_shade's H3
//! object branch (`fres_rgb.w == 3`) evaluates the laws above per pixel with the material's
//! cook-torrance constants (scene.rs `upload_object_part`).

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;

use super::lightmaps::H3Lightmap;

/// One object's light probe: the four SH vectors (DC, then the raw V1..V3 lightmap vectors) and
/// the dominant light, already scaled by the bake gain (scene units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H3ObjProbe {
    /// The lightmap's SH vectors V0..V3 (V0 = DC; the L1 vector of channel c is (V3, V1, -V2)[c]).
    pub sh: [[f32; 3]; 4],
    /// Dominant light intensity (rgb).
    pub intensity: [f32; 3],
    /// Dominant light direction (unit, toward the light; +Z when there is none).
    pub dir: [f32; 3],
    /// Where the probe came from (log / diagnostics).
    pub source: ProbeSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeSource {
    /// A lightmapped surface below the object (the engine's case).
    Below,
    /// Nothing below: the nearest lit surface above (a hanging object under a ceiling).
    Above,
    /// No lit surface on the vertical: the map's mean bake as a flat probe.
    Fallback,
}

fn lum(c: [f32; 3]) -> f32 { 0.212656 * c[0] + 0.715158 * c[1] + 0.0721856 * c[2] }

impl H3ObjProbe {
    /// A flat probe: constant irradiance `e` (diffuse = e, the same units as a baked vertex colour).
    pub fn flat(e: [f32; 3]) -> Self {
        // ravi_order_2 of a DC-only probe = 0.886227 * V0 / pi
        let k = std::f32::consts::PI / 0.886227;
        H3ObjProbe { sh: [[e[0] * k, e[1] * k, e[2] * k], [0.0; 3], [0.0; 3], [0.0; 3]], intensity: [0.0; 3], dir: [0.0, 0.0, 1.0], source: ProbeSource::Fallback }
    }

    /// From a lightmap texel (`H3Lightmap::probe_at`), scaled by the BSP's bake gain. The
    /// dominant direction is the engine's (lightmaps.rs `shade`): normalize(-lum(V3), -lum(V1), lum(V2)).
    pub fn from_texel(sh: [[f32; 3]; 4], intensity: [f32; 3], gain: f32, source: ProbeSource) -> Self {
        let mut d = Vec3::new(-lum(sh[3]), -lum(sh[1]), lum(sh[2]));
        let l2 = d.length_squared();
        d = if l2 > 1e-12 && l2.is_finite() { d / l2.sqrt() } else { Vec3::Z };
        let s = |v: [f32; 3]| [v[0] * gain, v[1] * gain, v[2] * gain];
        H3ObjProbe { sh: [s(sh[0]), s(sh[1]), s(sh[2]), s(sh[3])], intensity: s(intensity), dir: d.to_array(), source }
    }

    /// The per-channel L1 vector of channel `c` with the dominant light REMOVED and re-added as
    /// an analytical lambertian - the engine's `ravi_order_2_with_dominant_light`
    /// (spherical_harmonics_fx.hlsl:130, the same law lightmaps.rs `shade` evaluates for BSP
    /// texels): `l1 = (V3, V1, -V2) + 0.488603 * dir * I`.
    pub fn l1(&self, c: usize) -> [f32; 3] {
        let d = self.dir;
        [self.sh[3][c] + 0.488603 * d[0] * self.intensity[c],
         self.sh[1][c] + 0.488603 * d[1] * self.intensity[c],
         -self.sh[2][c] + 0.488603 * d[2] * self.intensity[c]]
    }

    /// The DC term with the dominant light removed: `V0 - 0.282095 * I`.
    pub fn sh0(&self) -> [f32; 3] {
        [self.sh[0][0] - 0.282095 * self.intensity[0],
         self.sh[0][1] - 0.282095 * self.intensity[1],
         self.sh[0][2] - 0.282095 * self.intensity[2]]
    }

    /// The engine diffuse at normal `n` - the CPU twin of the shader branch (and of
    /// lightmaps.rs `shade`, which this must match texel for texel):
    /// `max(0, (0.886227 sh0 - 1.023328 N.l1)/pi + 0.281 I max(0, dir.N))`.
    pub fn diffuse(&self, n: Vec3) -> [f32; 3] {
        let ndl = Vec3::from(self.dir).dot(n).max(0.0);
        let sh0 = self.sh0();
        let mut out = [0f32; 3];
        for c in 0..3 {
            let e = 0.886227 * sh0[c] - 1.023328 * n.dot(Vec3::from(self.l1(c)));
            out[c] = (e * std::f32::consts::FRAC_1_PI + 0.281 * self.intensity[c] * ndl).max(0.0);
        }
        out
    }

    /// The PER-INSTANCE renderer lanes of this probe (scene.rs `upload_object_part`):
    ///   instance colour = `[sh0.rgb, l1B.z]`
    ///   obj_probe0      = `[dom_dir.xyz, 1]` (w = 1 flags a valid probe)
    ///   obj_probe1      = `[dom_intensity.rgb, 1]`
    ///   obj_light       = f16 pairs `[l1R.x, l1R.y][l1R.z, l1G.x][l1G.y, l1G.z][l1B.x, l1B.y]`
    /// (the 9 L1 floats do not fit the 12 per-instance f32 slots, so 8 of them ride the f16 lane
    /// and the ninth the colour alpha; f16 holds the lightmap's magnitudes to ~1e-3 relative.)
    pub fn lanes(&self) -> ([f32; 4], [f32; 8], [u32; 4]) {
        let (r, g, b) = (self.l1(0), self.l1(1), self.l1(2));
        let s = self.sh0();
        let p = crate::scene::pack_f16x2;
        (
            [s[0], s[1], s[2], b[2]],
            [self.dir[0], self.dir[1], self.dir[2], 1.0, self.intensity[0], self.intensity[1], self.intensity[2], 1.0],
            [p(r[0], r[1]), p(r[2], g[0]), p(g[1], g[2]), p(b[0], b[1])],
        )
    }
}

/// A lightmapped world-space triangle with its lightmap UVs.
#[derive(Clone, Copy, Debug)]
struct Tri {
    p: [[f32; 3]; 3],
    uv: [[f32; 2]; 3],
    lm: u16,
}

/// Every lightmapped triangle of the map in an XY grid, with the lightmaps they sample.
pub struct H3LightField {
    lms: Vec<Arc<H3Lightmap>>,
    gains: Vec<f32>,
    tris: Vec<Tri>,
    grid: HashMap<(i32, i32), Vec<u32>>,
    cell: f32,
    /// The map's exposed mean bake (scene units): the flat fallback probe.
    pub fallback: [f32; 3],
}

impl H3LightField {
    pub fn new() -> Self { H3LightField { lms: Vec::new(), gains: Vec::new(), tris: Vec::new(), grid: HashMap::new(), cell: 2.0, fallback: [1.0; 3] } }

    /// Register a BSP's lightmap (+ its bake gain); returns the index `add_tris` takes.
    pub fn add_lightmap(&mut self, lm: Arc<H3Lightmap>, gain: f32) -> u16 {
        self.lms.push(lm);
        self.gains.push(gain);
        (self.lms.len() - 1) as u16
    }

    /// Add one lit mesh use: `pos` / `uv` per vertex (model space), `indices` triangles, one
    /// placement matrix at a time (world = m * pos).
    pub fn add_tris(&mut self, lm: u16, pos: &[[f32; 3]], uv: &[[f32; 2]], indices: &[u32], m: &glam::Mat4) {
        for t in indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if a >= pos.len() || b >= pos.len() || c >= pos.len() || a >= uv.len() || b >= uv.len() || c >= uv.len() { continue; }
            let w = |i: usize| m.transform_point3(Vec3::from(pos[i])).to_array();
            let tri = Tri { p: [w(a), w(b), w(c)], uv: [uv[a], uv[b], uv[c]], lm };
            if !tri.p.iter().all(|p| p.iter().all(|v| v.is_finite())) { continue; }
            let idx = self.tris.len() as u32;
            self.tris.push(tri);
            let (x0, x1) = (tri.p.iter().map(|p| p[0]).fold(f32::MAX, f32::min), tri.p.iter().map(|p| p[0]).fold(f32::MIN, f32::max));
            let (y0, y1) = (tri.p.iter().map(|p| p[1]).fold(f32::MAX, f32::min), tri.p.iter().map(|p| p[1]).fold(f32::MIN, f32::max));
            let (cx0, cx1) = ((x0 / self.cell).floor() as i32, (x1 / self.cell).floor() as i32);
            let (cy0, cy1) = ((y0 / self.cell).floor() as i32, (y1 / self.cell).floor() as i32);
            // a map-spanning triangle would flood the grid: cap the footprint (it is still found
            // through the cells it does cover)
            if (cx1 - cx0) as i64 * (cy1 - cy0) as i64 > 4096 { self.tris.pop(); continue; }
            for cx in cx0..=cx1 { for cy in cy0..=cy1 { self.grid.entry((cx, cy)).or_default().push(idx); } }
        }
    }

    pub fn tri_count(&self) -> usize { self.tris.len() }
    pub fn lightmap_count(&self) -> usize { self.lms.len() }

    /// The lit surface on the vertical through `pos`: the nearest below (up to `above_tol` above
    /// the origin, an object whose origin sits at its contact point), else the nearest above.
    fn vertical_hit(&self, pos: Vec3, above_tol: f32) -> Option<(f32, [f32; 2], u16, bool)> {
        let key = ((pos.x / self.cell).floor() as i32, (pos.y / self.cell).floor() as i32);
        let Some(list) = self.grid.get(&key) else { return None };
        let mut below: Option<(f32, [f32; 2], u16)> = None;
        let mut above: Option<(f32, [f32; 2], u16)> = None;
        for &i in list {
            let t = &self.tris[i as usize];
            let (a, b, c) = (t.p[0], t.p[1], t.p[2]);
            // 2D barycentrics in XY
            let d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            if d.abs() < 1e-9 { continue; }
            let l0 = ((b[1] - c[1]) * (pos.x - c[0]) + (c[0] - b[0]) * (pos.y - c[1])) / d;
            let l1 = ((c[1] - a[1]) * (pos.x - c[0]) + (a[0] - c[0]) * (pos.y - c[1])) / d;
            let l2 = 1.0 - l0 - l1;
            let eps = -1e-4;
            if l0 < eps || l1 < eps || l2 < eps { continue; }
            let z = l0 * a[2] + l1 * b[2] + l2 * c[2];
            let uv = [l0 * t.uv[0][0] + l1 * t.uv[1][0] + l2 * t.uv[2][0], l0 * t.uv[0][1] + l1 * t.uv[1][1] + l2 * t.uv[2][1]];
            if z <= pos.z + above_tol {
                if below.map_or(true, |(bz, _, _)| z > bz) { below = Some((z, uv, t.lm)); }
            } else if above.map_or(true, |(az, _, _)| z < az) {
                above = Some((z, uv, t.lm));
            }
        }
        below.map(|(z, uv, lm)| (z, uv, lm, true)).or(above.map(|(z, uv, lm)| (z, uv, lm, false)))
    }

    /// The probe for an object at `pos` (world), always Some (the fallback when nothing lit is
    /// on the vertical). `above_tol` in world units (0.5 = a vehicle whose origin sits half a
    /// unit above its wheels still finds the ground).
    pub fn sample(&self, pos: Vec3) -> H3ObjProbe {
        if let Some((_, uv, lm, below)) = self.vertical_hit(pos, 0.5) {
            let (sh, i) = self.lms[lm as usize].probe_at(uv[0], uv[1]);
            return H3ObjProbe::from_texel(sh, i, self.gains[lm as usize], if below { ProbeSource::Below } else { ProbeSource::Above });
        }
        H3ObjProbe::flat(self.fallback)
    }

    /// Probes for a list of placement matrices (their translations).
    pub fn sample_mats(&self, mats: &[glam::Mat4]) -> Vec<H3ObjProbe> {
        mats.iter().map(|m| self.sample(m.w_axis.truncate())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flat probe is exactly its irradiance at every normal, and the lane packing carries
    /// the engine's `ravi_order_2_with_dominant_light` inputs: colour = [sh0.rgb, l1B.z],
    /// obj_probe = [dir, 1, I, 1], obj_light = the eight remaining L1 halves.
    #[test]
    fn flat_probe_and_lanes() {
        let p = H3ObjProbe::flat([0.3, 0.5, 0.7]);
        for n in [Vec3::Z, -Vec3::Z, Vec3::X, Vec3::new(0.3, -0.4, 0.5).normalize()] {
            let d = p.diffuse(n);
            assert!((d[0] - 0.3).abs() < 1e-4 && (d[1] - 0.5).abs() < 1e-4 && (d[2] - 0.7).abs() < 1e-4, "{d:?}");
        }
        let (col, pr, ol) = p.lanes();
        assert_eq!(ol, [0; 4]);
        assert!((col[0] - 0.3 * std::f32::consts::PI / 0.886227).abs() < 1e-4 && col[3] == 0.0);
        assert_eq!([pr[3], pr[7]], [1.0, 1.0]);
        // A lit texel: the CPU diffuse must equal lightmaps.rs `shade` (the BSP per-pixel law) at
        // the same SH - an object and the floor under it are lit by ONE law.
        let sh = [[1.0, 1.1, 1.2], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.7, -0.8, -0.9]];
        let i = [2.0, 1.8, 1.6];
        let q = H3ObjProbe::from_texel(sh, i, 1.0, ProbeSource::Below);
        for n in [Vec3::Z, Vec3::X, -Vec3::Y, Vec3::new(0.2, 0.3, 0.9).normalize()] {
            let a = q.diffuse(n);
            let b = super::super::lightmaps::shade(&sh, i, n);
            for k in 0..3 { assert!((a[k] - b[k]).abs() < 1e-5, "n {n:?} ch {k}: probe {a:?} vs shade {b:?}"); }
        }
        // the f16 lane holds l1R.xyz, l1G.xy...; the colour alpha the ninth (l1B.z)
        let (col, _, ol) = q.lanes();
        let un = |w: u32| (half::f16::from_bits((w & 0xFFFF) as u16).to_f32(), half::f16::from_bits((w >> 16) as u16).to_f32());
        let (a, b) = un(ol[0]);
        assert!((a - q.l1(0)[0]).abs() < 1e-3 && (b - q.l1(0)[1]).abs() < 1e-3, "{a} {b}");
        assert!((col[3] - q.l1(2)[2]).abs() < 1e-6);
        // the bake gain scales the probe linearly
        let g = H3ObjProbe::from_texel(sh, i, 4.0, ProbeSource::Below);
        let (d1, d4) = (q.diffuse(Vec3::Z), g.diffuse(Vec3::Z));
        for k in 0..3 { assert!((d4[k] - 4.0 * d1[k]).abs() < 1e-4); }
    }

    /// The vertical query: a floor quad at z 0 and a bridge at z 5; an object at z 2 finds the
    /// floor, one at z 6 the bridge, one under the floor the floor from above, one off the map
    /// the fallback.
    #[test]
    fn vertical_hit_picks_the_surface_below() {
        let mut f = H3LightField::new();
        f.fallback = [0.2; 3];
        // a lightmap is needed only for a real sample; the hit test itself is checked here
        let pos = [[-10.0, -10.0, 0.0], [10.0, -10.0, 0.0], [10.0, 10.0, 0.0], [-10.0, 10.0, 0.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        f.add_tris(0, &pos, &uv, &[0, 1, 2, 0, 2, 3], &glam::Mat4::IDENTITY);
        let bridge: Vec<[f32; 3]> = pos.iter().map(|p| [p[0] * 0.2, p[1] * 0.2, 5.0]).collect();
        f.add_tris(0, &bridge, &uv, &[0, 1, 2, 0, 2, 3], &glam::Mat4::IDENTITY);
        assert_eq!(f.tri_count(), 4);
        let h = f.vertical_hit(Vec3::new(0.5, 0.5, 2.0), 0.5).unwrap();
        assert!((h.0 - 0.0).abs() < 1e-5 && h.3, "floor below: {h:?}");
        let h = f.vertical_hit(Vec3::new(0.5, 0.5, 6.0), 0.5).unwrap();
        assert!((h.0 - 5.0).abs() < 1e-5 && h.3, "bridge below: {h:?}");
        let h = f.vertical_hit(Vec3::new(0.5, 0.5, -3.0), 0.5).unwrap();
        assert!((h.0 - 0.0).abs() < 1e-5 && !h.3, "floor above: {h:?}");
        assert!(f.vertical_hit(Vec3::new(50.0, 50.0, 0.0), 0.5).is_none());
        let p = f.sample(Vec3::new(50.0, 50.0, 0.0));
        assert_eq!(p.source, ProbeSource::Fallback);
        assert!((p.diffuse(Vec3::Z)[0] - 0.2).abs() < 1e-4);
    }
}
