//! Halo 3 (MCC) baked lighting: the Lbsp lightprobe texture array + the per-vertex lighting
//! streams, decoded with the ENGINE law read off the shipped DXBC (guardian,
//! `shaders\shader_templates\_2_0_1_...` pixl entry 3 `static_per_pixel` / entry 4
//! `static_per_vertex`, and the `shaders\shader_shared_vertex_shaders` glvs rigid VS of entry 4;
//! disassembled with d3dcompiler_47 `D3DDisassemble`).
//!
//! Lbsp layout (Halo3MCC/Lbsp.xml, verified on guardian):
//!   * +0x8   "compression vectors": 18 x 12 B, of which only the x of every SECOND one (the
//!            plugin's named vectors 0/2/4/6/8, 24 B apart) is nonzero - they are the five per-pair
//!            scale constants the PS reads as cb3[0].xyz / cb3[1].x (SH vectors 0..3) and
//!            cb3[1].y (dominant intensity); guardian 48.2 / 35.2 / 51.1 / 49.7 / 131.2
//!   * +0xE0  lightprobe `bitm` (1024x1024 x 8 slices DXT5 array) - slices 2k, 2k+1 encode SH
//!            vector k ("16f": two DXT5 texels per value)
//!   * +0xF0  dominant-light-intensity `bitm` (2 slices, the same pair encoding)
//!   * +0x100 per-vertex lighting blocks (0x10: vertex-buffer index i32 @0xC -> a kind-6 stride-20
//!            stream: 5 RGBA8 dwords per vertex = SH vectors 0..3 + intensity)
//!   * +0x10C clusters (0x4: lightprobe array index i16, per-vertex block i16)
//!   * +0x118 instances (0x8: lightprobe array index, per-vertex block, probe block)
//!   * +0x1A0 per-instance lightmap texcoords (0x10: vertex-buffer index i16 @0xC -> a kind-4
//!            stride-4 u16x2 stream, unorm16 atlas UV). Cluster meshes carry theirs in `vb[1]`.
//!
//! Per-pixel decode (PS entry 3), per slice pair (s0, s1) with scale C:
//!   V = C * s0.a * s1.a * (2 * (s0.rgb + s1.rgb) - 2)
//! Per-vertex decode (VS entry 4): V = 2 * rgb - 1 per dword, NO scale, alpha unused.
//! Shading (both PS entries, `n` = the surface normal):
//!   lum(c) = 0.212656 r + 0.715158 g + 0.0721856 b
//!   dir    = normalize(-lum(V3), -lum(V1), +lum(V2))        (dominant light direction)
//!   sh0    = V0 - 0.282095 * I;   l1 = (V3, V1, -V2) + 0.488603 * dir * I      (per channel)
//!   E      = 0.886227 * sh0 - 1.02333 * dot(n, l1)
//!   diffuse = E / pi + 0.281 * I * max(0, dot(dir, n))        -> multiplied by the albedo
//! Everything here is CPU-evaluated per VERTEX and handed to the shared renderer as the baked
//! vertex colour (alpha 0 = no analytic sun on top: the bake already holds the sun).

use anyhow::{anyhow, bail, Result};
use glam::Vec3;

use super::cache::H3Cache;
use super::geometry::{read_vb_raw, H3Bsp};
use crate::h4::cache::ByteRead;
use crate::h4::lightmaps::decode_bc3_bgra;

pub const OFF_LBSP_COMPRESSION: usize = 0x8;
pub const COMPRESSION_STRIDE: usize = 24;
pub const OFF_LBSP_LIGHTPROBE_BITMAP: usize = 0xE0;
pub const OFF_LBSP_INTENSITY_BITMAP: usize = 0xF0;
pub const OFF_LBSP_PERVERTEX: usize = 0x100;
pub const OFF_LBSP_CLUSTERS: usize = 0x10C;
pub const OFF_LBSP_INSTANCES: usize = 0x118;
pub const OFF_LBSP_INST_TEXCOORDS: usize = 0x1A0;
pub const PERVERTEX_STRIDE: usize = 20;

/// One decoded lightmap page: the 8 lightprobe + 2 intensity slices as BGRA8 (from the BC3
/// decoder), all `w x h`.
pub struct H3Lightmap {
    pub w: usize,
    pub h: usize,
    pub lp: Vec<Vec<u8>>,
    pub intensity: Vec<Vec<u8>>,
    /// The five pair scales (SH vectors 0..3, intensity).
    pub consts: [f32; 5],
    /// Per sbsp cluster: (lightprobe array index, per-vertex block index).
    pub clusters: Vec<(i16, i16)>,
    /// Per sbsp instance: (lightprobe array index, per-vertex block index, probe block index).
    pub instances: Vec<(i16, i16, i16)>,
    /// Per-vertex block -> geometry-resource vertex-buffer index.
    pub pervertex_vb: Vec<i32>,
    /// Per-instance lightmap-texcoord block -> vertex-buffer index.
    pub inst_texcoord_vb: Vec<i16>,
}

/// How one mesh use is lit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightingMode {
    /// Per-pixel lightmap through a u16x2 UV stream (evaluated at the vertices here).
    PerPixel,
    /// The instance's kind-6 per-vertex stream.
    PerVertex,
    /// Nothing usable (no stream / probe-only): flat fallback.
    Flat,
}

fn decode_slices(c: &H3Cache, tag: usize, expect: usize) -> Result<(usize, usize, Vec<Vec<u8>>)> {
    let (info, bytes) = super::bitmaps::bitmap_pixels_avail(c, tag)?;
    if info.format != 0x10 { bail!("lightmap bitmap format {:#x}, expected DXT5", info.format); }
    let (w, h) = (info.w as usize, info.h as usize);
    let per = w.div_ceil(4) * h.div_ceil(4) * 16;
    if per == 0 { bail!("empty lightmap bitmap"); }
    let avail = bytes.len() / per;
    if avail < expect { bail!("lightmap bitmap holds {avail} slices, expected {expect} ({} B)", bytes.len()); }
    let slices = (0..expect).map(|k| decode_bc3_bgra(&bytes[k * per..(k + 1) * per], w as u32, h as u32)).collect();
    Ok((w, h, slices))
}

/// Load a BSP's lightmap page + lighting index tables from its Lbsp.
pub fn load_lightmap(c: &H3Cache, bsp: &H3Bsp) -> Result<H3Lightmap> {
    let lt = bsp.lbsp_tag.ok_or_else(|| anyhow!("no Lbsp"))?;
    let lb = c.tag_meta(lt).ok_or_else(|| anyhow!("Lbsp has no meta"))?;
    let d = c.data();
    let lp_tag = c.tag_ref(lb + OFF_LBSP_LIGHTPROBE_BITMAP).map(|(_, t)| t).ok_or_else(|| anyhow!("no lightprobe bitmap"))?;
    let in_tag = c.tag_ref(lb + OFF_LBSP_INTENSITY_BITMAP).map(|(_, t)| t).ok_or_else(|| anyhow!("no intensity bitmap"))?;
    let (w, h, lp) = decode_slices(c, lp_tag, 8)?;
    let (iw, ih, intensity) = decode_slices(c, in_tag, 2)?;
    if (iw, ih) != (w, h) { bail!("intensity map {iw}x{ih} != lightprobe {w}x{h}"); }
    let mut consts = [0f32; 5];
    for (k, v) in consts.iter_mut().enumerate() { *v = d.f32_at(lb + OFF_LBSP_COMPRESSION + k * COMPRESSION_STRIDE); }
    let block = |off: usize| c.block(lb + off).unwrap_or((0, 0));
    let (nc, oc) = block(OFF_LBSP_CLUSTERS);
    let clusters = (0..nc).map(|i| (d.i16_at(oc + i * 4), d.i16_at(oc + i * 4 + 2))).collect();
    let (ni, oi) = block(OFF_LBSP_INSTANCES);
    let instances = (0..ni).map(|i| (d.i16_at(oi + i * 8), d.i16_at(oi + i * 8 + 2), d.i16_at(oi + i * 8 + 4))).collect();
    let (npv, opv) = block(OFF_LBSP_PERVERTEX);
    let pervertex_vb = (0..npv).map(|i| d.i32_at(opv + i * 0x10 + 0xC)).collect();
    let (nt, ot) = block(OFF_LBSP_INST_TEXCOORDS);
    let inst_texcoord_vb = (0..nt).map(|i| d.i16_at(ot + i * 0x10 + 0xC)).collect();
    Ok(H3Lightmap { w, h, lp, intensity, consts, clusters, instances, pervertex_vb, inst_texcoord_vb })
}

impl H3Lightmap {
    /// Bilinear fetch of one BGRA8 slice at a unorm UV (D3D half-texel convention, clamped) as
    /// (r, g, b, a) in 0..1.
    fn sample(&self, slice: &[u8], u: f32, v: f32) -> [f32; 4] {
        let fx = (u.clamp(0.0, 1.0) * self.w as f32 - 0.5).max(0.0);
        let fy = (v.clamp(0.0, 1.0) * self.h as f32 - 0.5).max(0.0);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (x0, y0) = (x0.min(self.w - 1), y0.min(self.h - 1));
        let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
        let px = |x: usize, y: usize| { let o = (y * self.w + x) * 4; [slice[o + 2] as f32, slice[o + 1] as f32, slice[o] as f32, slice[o + 3] as f32] };
        let (a, b, c, d) = (px(x0, y0), px(x1, y0), px(x0, y1), px(x1, y1));
        let mut out = [0f32; 4];
        for k in 0..4 {
            out[k] = ((a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty) / 255.0;
        }
        out
    }

    /// One slice pair -> the decoded vector (PS entry 3 law).
    fn pair(&self, s0: &[u8], s1: &[u8], scale: f32, u: f32, v: f32) -> [f32; 3] {
        let a = self.sample(s0, u, v);
        let b = self.sample(s1, u, v);
        let k = scale * a[3] * b[3];
        [k * (2.0 * (a[0] + b[0]) - 2.0), k * (2.0 * (a[1] + b[1]) - 2.0), k * (2.0 * (a[2] + b[2]) - 2.0)]
    }

    /// The four SH vectors + dominant intensity at a lightmap UV.
    pub fn probe_at(&self, u: f32, v: f32) -> ([[f32; 3]; 4], [f32; 3]) {
        let mut sh = [[0f32; 3]; 4];
        for k in 0..4 { sh[k] = self.pair(&self.lp[2 * k], &self.lp[2 * k + 1], self.consts[k], u, v); }
        let i = self.pair(&self.intensity[0], &self.intensity[1], self.consts[4], u, v);
        (sh, i)
    }
}

/// A per-vertex lighting record (5 RGBA8 dwords) -> SH vectors + intensity (VS entry 4 law).
pub fn pervertex_probe(rec: &[u8]) -> ([[f32; 3]; 4], [f32; 3]) {
    let dw = |k: usize| [rec[k * 4] as f32 / 127.5 - 1.0, rec[k * 4 + 1] as f32 / 127.5 - 1.0, rec[k * 4 + 2] as f32 / 127.5 - 1.0];
    ([dw(0), dw(1), dw(2), dw(3)], dw(4))
}

fn lum(c: [f32; 3]) -> f32 { 0.212656 * c[0] + 0.715158 * c[1] + 0.0721856 * c[2] }

/// The engine diffuse factor at normal `n` (multiplied by the albedo downstream).
pub fn shade(sh: &[[f32; 3]; 4], i: [f32; 3], n: Vec3) -> [f32; 3] {
    let mut dir = Vec3::new(-lum(sh[3]), -lum(sh[1]), lum(sh[2]));
    let l2 = dir.length_squared();
    dir = if l2 > 0.0 && l2.is_finite() { dir / l2.sqrt() } else { Vec3::ZERO };
    let ndl = dir.dot(n).max(0.0);
    let mut out = [0f32; 3];
    for ch in 0..3 {
        let sh0 = sh[0][ch] - 0.282095 * i[ch];
        let l1 = Vec3::new(sh[3][ch], sh[1][ch], -sh[2][ch]) + 0.488603 * dir * i[ch];
        let e = 0.886227 * sh0 - 1.02333 * n.dot(l1);
        out[ch] = (e * 0.318310 + 0.281 * i[ch] * ndl).max(0.0);
    }
    out
}

/// A cluster mesh's lightmap UVs (its `vb[1]`, kind 4 stride 4 u16x2 unorm), one per vertex.
pub fn cluster_lightmap_uvs(c: &H3Cache, bsp: &H3Bsp, mesh_idx: usize, n_verts: usize) -> Result<Vec<[f32; 2]>> {
    let m = bsp.meshes.get(mesh_idx).ok_or_else(|| anyhow!("mesh {mesh_idx} out of range"))?;
    if m.vb[1] < 0 { bail!("mesh {mesh_idx} has no lightmap-uv stream"); }
    uv_stream(c, bsp, m.vb[1] as usize, n_verts)
}

/// A u16x2 unorm UV stream, checked against the mesh's vertex count.
pub fn uv_stream(c: &H3Cache, bsp: &H3Bsp, vb_index: usize, n_verts: usize) -> Result<Vec<[f32; 2]>> {
    let (vb, bytes) = read_vb_raw(c, bsp, vb_index)?;
    if vb.stride != 4 { bail!("uv stream vb {vb_index} stride {} (kind {}), expected 4", vb.stride, vb.kind); }
    if vb.count as usize != n_verts { bail!("uv stream vb {vb_index} has {} entries for {n_verts} verts", vb.count); }
    Ok(bytes.chunks_exact(4).map(|b| [u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0, u16::from_le_bytes([b[2], b[3]]) as f32 / 65535.0]).collect())
}

/// A kind-6 per-vertex lighting stream, checked against the mesh's vertex count.
pub fn pervertex_stream(c: &H3Cache, bsp: &H3Bsp, vb_index: usize, n_verts: usize) -> Result<Vec<u8>> {
    let (vb, bytes) = read_vb_raw(c, bsp, vb_index)?;
    if vb.stride as usize != PERVERTEX_STRIDE { bail!("per-vertex stream vb {vb_index} stride {} (kind {}), expected 20", vb.stride, vb.kind); }
    if vb.count as usize != n_verts { bail!("per-vertex stream vb {vb_index} has {} records for {n_verts} verts", vb.count); }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cache::maps_dir;
    use super::super::geometry::{decode_mesh, load_bsp};

    #[test]
    fn guardian_lightmap_decodes() {
        let Some(dir) = maps_dir() else { return };
        let p = dir.join("guardian.map");
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return; }
        let c = H3Cache::open(&p).unwrap();
        let bsp = load_bsp(&c, c.find_tags(b"sbsp")[0]).unwrap();
        let lm = load_lightmap(&c, &bsp).unwrap();
        assert_eq!((lm.w, lm.h, lm.lp.len(), lm.intensity.len()), (1024, 1024, 8, 2));
        assert_eq!(lm.clusters.len(), 26);
        assert_eq!(lm.instances.len(), 479);
        assert!(lm.consts.iter().all(|&k| k > 1.0 && k < 1000.0), "consts {:?}", lm.consts);
        // every cluster mesh has a UV stream of the right length, and its lighting is finite +
        // non-black on average
        let mut sum = [0f64; 3];
        let mut n = 0usize;
        for (k, &mi) in bsp.clusters.iter().enumerate() {
            let dm = decode_mesh(&c, &bsp, mi as usize).unwrap().unwrap();
            let uvs = cluster_lightmap_uvs(&c, &bsp, mi as usize, dm.verts.len()).unwrap();
            assert_eq!(lm.clusters[bsp.cluster_ids[k]].0, 0);
            for (v, uv) in dm.verts.iter().zip(&uvs) {
                let (sh, i) = lm.probe_at(uv[0], uv[1]);
                let d = shade(&sh, i, Vec3::from(v.normal));
                assert!(d.iter().all(|x| x.is_finite() && *x >= 0.0));
                for ch in 0..3 { sum[ch] += d[ch] as f64; }
                n += 1;
            }
        }
        let mean = sum.map(|s| s / n as f64);
        assert!(mean.iter().all(|&m| m > 0.01), "cluster mean diffuse {mean:?}");
        // per-vertex instances: the stream decodes for every lit instance
        let mut pv = 0;
        for inst in &bsp.instances {
            let (_, pvb, _) = lm.instances[inst.index];
            if pvb < 0 { continue; }
            let dm = decode_mesh(&c, &bsp, inst.mesh as usize).unwrap().unwrap();
            let rec = pervertex_stream(&c, &bsp, lm.pervertex_vb[pvb as usize] as usize, dm.verts.len()).unwrap();
            let (sh, i) = pervertex_probe(&rec[..PERVERTEX_STRIDE]);
            let d = shade(&sh, i, Vec3::Z);
            assert!(d.iter().all(|x| x.is_finite()));
            pv += 1;
        }
        assert!(pv > 400, "per-vertex instances {pv}");
    }
}
