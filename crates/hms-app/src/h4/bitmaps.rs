//! Halo 4 bitmaps: `bitm` element (48 B) + the `bitmap_texture_interop_resource` streams.
//!
//! Layout (wraparound, all 2735 bitmap resources): the 92-B resource definition holds three
//! 20-B tag_data entries at 0 / 0x14 / 0x28 {size i32 @0, page address (fixup) @12}; their
//! fixup nibbles map to page streams and mip ranges as
//!   nibble 6 (segment page[2]) = mip 0, nibble 8 (page[1]) = mip 1, nibble 4 (page[0]) = mips 2..N
//! and the three sizes sum to the element's total pixel size (@24). Bitmaps with only the
//! primary stream keep mips from some level L (derived from the size here).
//! Element: w i16 @0, h i16 @2, depth u8 @4, type u8 @6 (0 = 2D, 2 = cube, 3 = array),
//! format i16 @8 (Reach enum: 14 DXT1, 15 DXT3, 16 DXT5, 38 DXN, 11 A8R8G8B8, ...), total @24.
//! Formats 24..36 (the fp16 / BC6H / BC7 family, not identified) are skipped.
//!
//! On 5 000+ bitmaps of wraparound + ca_forge_ravine: the resource definition's
//! interop struct at +0x3C carries the D3D11 texture description - w u16 @0x3C, h @0x3E, depth
//! u8 @0x40, mip count @0x41, type @0x42, bitmap format @0x43 and the **DXGI_FORMAT byte @0x4C**
//! (71 / 72 = BC1_UNORM / _SRGB, 77 / 78 = BC3, 84 = BC5_SNORM, 87 / 91 = B8G8R8A8_UNORM / _SRGB):
//! the per-bitmap sRGB flag the engine samples with. DXN normal maps are BC5_SNORM (the srf_*
//! shaders use .xy raw; sampled endpoint means are 0.000 signed vs 0.03..0.50 unsigned).
//! Cube maps (type 2): 6 faces of BGRA8 / DXT1 / DXT5, MIP-MAJOR (all six faces of mip 0, then
//! mip 1, ...; per-face means match across mips). Some 2D DXN resources carry a
//! second, unused flat-normal chain after the mip tail (element flags byte @5 bit 4); the primary
//! stream is then LARGER than the mip tail and only its prefix is real.

use anyhow::{anyhow, bail, Result};

use super::cache::{ByteRead, H4Cache};

/// `bitm` main struct: usage @0x00 (0 diffuse, 1 specular, 4 detail, 7 cube, 36 normal ...),
/// bitmaps block @0x7C (48-B elements), resources block @0xA8.
pub const OFF_BITM_BITMAPS: usize = 0x7C;
pub const OFF_BITM_RESOURCES: usize = 0xA8;
#[cfg(test)]
pub const BITMAP_ELEM: usize = 48;
/// Resource definition: the D3D11 desc at +0x3C, DXGI format byte at +0x4C.
pub const DEF_DXGI_FORMAT: usize = 0x4C;

/// True for the *_SRGB DXGI formats the engine binds colour maps with.
pub fn dxgi_is_srgb(dxgi: u32) -> bool { matches!(dxgi, 72 | 75 | 78 | 91 | 29 | 99) }

/// The DXGI format the engine creates this bitmap's texture with (resource definition
/// +0x4C), or None when the resource is unreadable.
pub fn bitmap_dxgi(c: &H4Cache, tag: usize) -> Option<u32> {
    let m = c.tag_meta(tag)?;
    let (_, ro) = c.block(m + OFF_BITM_RESOURCES)?;
    let e = c.resource_by_id(c.data().u32_at(ro))?;
    let dd = c.definition(e);
    dd.get(DEF_DXGI_FORMAT).map(|b| *b as u32)
}

#[derive(Clone, Copy, Debug)]
pub struct BitmapInfo {
    pub w: u16,
    pub h: u16,
    pub depth: u8,
    pub kind: u8,
    pub format: i16,
    /// Mip levels INCLUDING the base (element byte @16 + 1). Chains are truncated (a 64x64
    /// A8R8G8B8 ships 5 levels, sprites 1), consistent with the stream sizes.
    pub levels: u32,
    pub total: u32,
}

pub fn bitmap_info(c: &H4Cache, tag: usize) -> Option<BitmapInfo> {
    let m = c.tag_meta(tag)?;
    let (_, o) = c.block(m + OFF_BITM_BITMAPS)?;
    let d = c.data();
    Some(BitmapInfo { w: d.u16_at(o), h: d.u16_at(o + 2), depth: d.u8_at(o + 4), kind: d.u8_at(o + 6), format: d.i16_at(o + 8), levels: d.u8_at(o + 16) as u32 + 1, total: d.u32_at(o + 24) })
}

/// Bytes per 4x4 block for the BCn formats, or bytes per pixel (negative) for uncompressed ones.
fn format_desc(format: i16) -> Option<(u32 /*dxgi*/, usize /*block bytes*/, bool /*compressed*/)> {
    match format {
        14 => Some((71, 8, true)),   // DXT1 -> BC1
        15 => Some((74, 16, true)),  // DXT3 -> BC2
        16 => Some((77, 16, true)),  // DXT5 -> BC3
        38 => Some((84, 16, true)),  // DXN  -> BC5_SNORM (the shaders read .xy signed)
        36 => Some((80, 8, true)),   // DXT5a mono -> BC4 (PROVISIONAL)
        11 => Some((87, 4, false)),  // A8R8G8B8 -> B8G8R8A8
        _ => None,
    }
}

fn level_size(w: u32, h: u32, block: usize, compressed: bool) -> usize {
    if compressed { (w.max(1).div_ceil(4) * h.max(1).div_ceil(4)) as usize * block } else { (w.max(1) * h.max(1)) as usize * block }
}

/// A base-colour texture ready for the renderer.
pub enum Texel {
    /// DDS (DXT10 header) with `mips` levels starting at `w` x `h`.
    Dds { bytes: Vec<u8>, w: u32, h: u32, mips: u32 },
    /// One uncompressed BGRA8 level.
    Bgra { bytes: Vec<u8>, w: u32, h: u32 },
}

/// A colour-grading volume (`bitm` type 1 = 3D, A8R8G8B8, n x n x n; the cfxs
/// +0xF0 16^3 LUTs): mip 0 as BGRA8 texels, x fastest, then y, then z (slice-major), from the
/// stream that holds the mip chain (mip 0 first; every shipped LUT ships 5 levels = 18724 B for
/// n = 16). Returns (bgra, n, dxgi format byte).
pub fn load_volume_lut(c: &H4Cache, tag: usize) -> Result<(Vec<u8>, u32, u32)> {
    let info = bitmap_info(c, tag).ok_or_else(|| anyhow!("no bitmap element"))?;
    if info.kind != 1 { bail!("bitmap type {} (not 3D)", info.kind); }
    if info.format != 11 { bail!("volume LUT format {} (expected A8R8G8B8 = 11)", info.format); }
    let n = info.w as u32;
    if n < 2 || info.h as u32 != n || info.depth as u32 != n { bail!("volume LUT {}x{}x{} is not cubic", info.w, info.h, info.depth); }
    let m = c.tag_meta(tag).ok_or_else(|| anyhow!("meta"))?;
    let (_, ro) = c.block(m + OFF_BITM_RESOURCES).ok_or_else(|| anyhow!("no resources block"))?;
    let e = c.resource_by_id(c.data().u32_at(ro)).ok_or_else(|| anyhow!("resource id does not resolve"))?;
    if !c.kind_is(e.kind, H4Cache::RES_BITMAP) { bail!("resource type {} is not a bitmap", e.kind); }
    let dd = c.definition(e);
    let need = (n * n * n * 4) as usize;
    // the largest stream holds mip 0 first (the whole chain when only one stream exists)
    let mut best: Option<(usize, u32)> = None;
    for k in 0..3 {
        let size = dd.i32_at(k * 20).max(0) as usize;
        if let Some(addr) = e.fixup_at((k * 20 + 12) as u32) {
            if size >= need && best.map_or(true, |b| size > b.0) { best = Some((size, addr)); }
        }
    }
    let (_, addr) = best.ok_or_else(|| anyhow!("no page stream holds the {need} B volume"))?;
    let bytes = c.stream_bytes(e, addr, need)?;
    Ok((bytes, n, dd.get(DEF_DXGI_FORMAT).map(|b| *b as u32).unwrap_or(0)))
}

/// Assemble the largest available mip chain of a 2D bitmap from its page streams.
pub fn load_base_texture(c: &H4Cache, tag: usize) -> Result<Texel> {
    let info = bitmap_info(c, tag).ok_or_else(|| anyhow!("no bitmap element"))?;
    if info.kind != 0 { bail!("bitmap type {} (not 2D)", info.kind); }
    let (dxgi, block, compressed) = format_desc(info.format).ok_or_else(|| anyhow!("unsupported bitmap format {}", info.format))?;
    let m = c.tag_meta(tag).ok_or_else(|| anyhow!("meta"))?;
    let (_, ro) = c.block(m + OFF_BITM_RESOURCES).ok_or_else(|| anyhow!("no resources block"))?;
    let e = c.resource_by_id(c.data().u32_at(ro)).ok_or_else(|| anyhow!("resource id does not resolve"))?;
    if !c.kind_is(e.kind, H4Cache::RES_BITMAP) { bail!("resource type {} is not a bitmap", e.kind); }
    let dd = c.definition(e);
    // the shipped chain (truncated at info.levels, not at 1x1)
    let mut levels: Vec<(u32, u32, usize)> = Vec::new();
    let (mut w, mut h) = (info.w as u32, info.h as u32);
    for _ in 0..info.levels.clamp(1, 16) {
        levels.push((w, h, level_size(w, h, block, compressed)));
        if w <= 1 && h <= 1 { break; }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    // streams: (nibble, size, addr)
    let mut streams: Vec<(u32, usize, u32)> = Vec::new();
    for k in 0..3 {
        let size = dd.i32_at(k * 20).max(0) as usize;
        if let Some(addr) = e.fixup_at((k * 20 + 12) as u32) {
            if size > 0 { streams.push((addr >> 28, size, addr)); }
        }
    }
    let find = |nib: u32| streams.iter().find(|s| s.0 == nib).copied();
    let primary = find(4).ok_or_else(|| anyhow!("no primary stream"))?;
    // which level does the primary chunk start at? (its size == sum of levels L..N)
    let mut start_l = None;
    for l in 0..levels.len() {
        let s: usize = levels[l..].iter().map(|x| x.2).sum();
        if s == primary.1 { start_l = Some(l); break; }
    }
    // interleaved DXN resources pad the primary stream with a second (flat) chain;
    // the real mip tail is the prefix whose size matches the levels below the streamed pair.
    if start_l.is_none() && primary.1 > 0 {
        let l = match (find(6).is_some(), find(8).is_some()) { (true, true) => 2, (true, false) => 1, _ => 0 }.min(levels.len() - 1);
        let s: usize = levels[l..].iter().map(|x| x.2).sum();
        if primary.1 > s { start_l = Some(l); }
    }
    let start_l = start_l.ok_or_else(|| anyhow!("primary stream size {} does not match a mip tail ({}x{} fmt {})", primary.1, info.w, info.h, info.format))?;
    // gather chunks from the top: mip0 (nibble 6) and mip1 (nibble 8) when their pages exist
    let mut chunks: Vec<(usize /*level*/, Vec<u8>)> = Vec::new();
    let mut top = start_l;
    if start_l >= 2 {
        let m0 = find(6).filter(|s| s.1 == levels[0].2).and_then(|s| c.stream_bytes(e, s.2, s.1).ok());
        let m1 = find(8).filter(|s| s.1 == levels.get(1).map_or(0, |x| x.2)).and_then(|s| c.stream_bytes(e, s.2, s.1).ok());
        match (m0, m1) {
            (Some(a), Some(b)) if start_l == 2 => { chunks.push((0, a)); chunks.push((1, b)); top = 0; }
            (_, Some(b)) if start_l == 2 => { chunks.push((1, b)); top = 1; }
            _ => {}
        }
    }
    let tail_len: usize = levels[start_l..].iter().map(|x| x.2).sum();
    let tail = c.stream_bytes(e, primary.2, tail_len.min(primary.1))?;
    chunks.push((start_l, tail));
    let (tw, th) = (levels[top].0, levels[top].1);
    let mips = (levels.len() - top) as u32;
    if !compressed {
        // one level is enough for the base-colour path (the renderer builds its own mips)
        let first = &chunks[0].1;
        let n = level_size(tw, th, block, false);
        return Ok(Texel::Bgra { bytes: first[..n.min(first.len())].to_vec(), w: tw, h: th });
    }
    let mut bytes = dds_header_dxt10(tw, th, mips, dxgi);
    for (_, ch) in chunks { bytes.extend_from_slice(&ch); }
    Ok(Texel::Dds { bytes, w: tw, h: th, mips })
}

/// Decode BC1 (DXT1) blocks to BGRA8 (opaque; the 1-bit alpha is ignored).
pub fn decode_bc1_bgra(data: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (bw, bh) = (w.max(1).div_ceil(4) as usize, h.max(1).div_ceil(4) as usize);
    let mut out = vec![0u8; (w * h * 4) as usize];
    let c565 = |c: u16| -> [f32; 3] { [((c >> 11) & 31) as f32 / 31.0, ((c >> 5) & 63) as f32 / 63.0, (c & 31) as f32 / 31.0] };
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * 8;
            if o + 8 > data.len() { break; }
            let c0 = u16::from_le_bytes([data[o], data[o + 1]]);
            let c1 = u16::from_le_bytes([data[o + 2], data[o + 3]]);
            let (p0, p1) = (c565(c0), c565(c1));
            let mut pal = [p0, p1, [0.0; 3], [0.0; 3]];
            for k in 0..3 {
                if c0 > c1 { pal[2][k] = (2.0 * p0[k] + p1[k]) / 3.0; pal[3][k] = (p0[k] + 2.0 * p1[k]) / 3.0; }
                else { pal[2][k] = 0.5 * (p0[k] + p1[k]); pal[3][k] = 0.0; }
            }
            let bits = u32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]);
            for py in 0..4 {
                for px in 0..4 {
                    let (x, y) = (bx as u32 * 4 + px, by as u32 * 4 + py);
                    if x >= w || y >= h { continue; }
                    let idx = ((bits >> (2 * (py * 4 + px))) & 3) as usize;
                    let p = ((y * w + x) * 4) as usize;
                    out[p] = (pal[idx][2] * 255.0 + 0.5) as u8;
                    out[p + 1] = (pal[idx][1] * 255.0 + 0.5) as u8;
                    out[p + 2] = (pal[idx][0] * 255.0 + 0.5) as u8;
                    out[p + 3] = 255;
                }
            }
        }
    }
    out
}

fn srgb_to_linear(v: f32) -> f32 { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } }

/// The six mip-0 faces of a cube bitmap as LINEAR RGBA f32 (sRGB-decoded when the bitmap's
/// DXGI format is an sRGB one), in D3D11 cube order (+X -X +Y -Y +Z -Z), for
/// `build_hdr_env_cube`. TODO: confirm the face order against an engine capture.
pub fn load_cube_faces(c: &H4Cache, tag: usize) -> Result<Vec<(Vec<f32>, u32, u32)>> {
    let info = bitmap_info(c, tag).ok_or_else(|| anyhow!("no bitmap element"))?;
    if info.kind != 2 { bail!("bitmap type {} (not a cube)", info.kind); }
    let (_, block, compressed) = format_desc(info.format).ok_or_else(|| anyhow!("unsupported cube format {}", info.format))?;
    if !matches!(info.format, 11 | 14 | 16) { bail!("cube format {} not decoded", info.format); }
    let m = c.tag_meta(tag).ok_or_else(|| anyhow!("meta"))?;
    let (_, ro) = c.block(m + OFF_BITM_RESOURCES).ok_or_else(|| anyhow!("no resources block"))?;
    let e = c.resource_by_id(c.data().u32_at(ro)).ok_or_else(|| anyhow!("resource id does not resolve"))?;
    let dd = c.definition(e);
    let srgb = dd.get(DEF_DXGI_FORMAT).map_or(false, |b| dxgi_is_srgb(*b as u32));
    let (w, h) = (info.w as u32, info.h as u32);
    let face_bytes = level_size(w, h, block, compressed);
    // mip 0 of all six faces is the first 6 * face_bytes of the primary stream (mip-major)
    let mut primary = None;
    for k in 0..3 {
        let size = dd.i32_at(k * 20).max(0) as usize;
        if let Some(addr) = e.fixup_at((k * 20 + 12) as u32) {
            if size > 0 && addr >> 28 == 4 { primary = Some((size, addr)); }
        }
    }
    let (size, addr) = primary.ok_or_else(|| anyhow!("no primary stream"))?;
    if size < 6 * face_bytes { bail!("cube stream {size} < 6 x {face_bytes}"); }
    let bytes = c.stream_bytes(e, addr, 6 * face_bytes)?;
    let mut faces = Vec::with_capacity(6);
    for f in 0..6 {
        let raw = &bytes[f * face_bytes..(f + 1) * face_bytes];
        let bgra = match info.format {
            11 => raw.to_vec(),
            14 => decode_bc1_bgra(raw, w, h),
            _ => super::lightmaps::decode_bc3_bgra(raw, w, h),
        };
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for px in bgra.chunks_exact(4) {
            for &ch in &[px[2], px[1], px[0]] {
                let v = ch as f32 / 255.0;
                out.push(if srgb { srgb_to_linear(v) } else { v });
            }
            out.push(1.0);
        }
        faces.push((out, w, h));
    }
    Ok(faces)
}

/// 148-byte DDS header (magic + DDS_HEADER + DXT10) as `upload_texture_dds` expects.
pub fn dds_header_dxt10(w: u32, h: u32, mips: u32, dxgi: u32) -> Vec<u8> {
    let mut v = vec![0u8; 148];
    v[0..4].copy_from_slice(b"DDS ");
    let put = |v: &mut Vec<u8>, o: usize, x: u32| v[o..o + 4].copy_from_slice(&x.to_le_bytes());
    put(&mut v, 4, 124);
    put(&mut v, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000); // caps|height|width|pixelformat|mipmapcount|linearsize
    put(&mut v, 12, h);
    put(&mut v, 16, w);
    put(&mut v, 28, mips);
    put(&mut v, 76, 32);
    put(&mut v, 80, 0x4); // DDPF_FOURCC
    v[84..88].copy_from_slice(b"DX10");
    put(&mut v, 108, 0x1000 | 0x400000 | 0x8); // texture | mipmap | complex
    put(&mut v, 128, dxgi);
    put(&mut v, 132, 3); // D3D10_RESOURCE_DIMENSION_TEXTURE2D
    put(&mut v, 140, 1); // array size
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;

    #[test]
    fn wraparound_bitmaps() {
        let Some(dir) = maps_dir() else { return };
        let p = dir.join("wraparound.map");
        if !p.is_file() { return; }
        let c = H4Cache::open(&p).unwrap();
        // mp_forerunner_tilesheet_03_diff: 512x512 DXT1, 174776 B total = 131072 + 32768 + 10936
        let tag = (0..c.tag_count()).find(|&i| c.tag_name(i).ends_with("mp_forerunner_tilesheet_03_diff")).expect("tag");
        let info = bitmap_info(&c, tag).unwrap();
        assert_eq!((info.w, info.h, info.format, info.levels, info.total), (512, 512, 14, 10, 174776));
        match load_base_texture(&c, tag).unwrap() {
            Texel::Dds { bytes, w, h, mips } => {
                assert_eq!((w, h, mips), (512, 512, 10));
                assert_eq!(bytes.len(), 148 + 174776);
            }
            _ => panic!("expected DDS"),
        }
        // every 2D DXT1/3/5 + A8R8G8B8 bitmap loads (DXN failures are counted separately: some
        // ship a padded primary stream, see `load_base_texture`)
        let mut ok = 0usize;
        let mut fail = 0usize;
        let mut dxn_fail = 0usize;
        let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();
        for t in c.find_tags(b"bitm") {
            let Some(i) = bitmap_info(&c, t) else { continue };
            if i.kind != 0 || format_desc(i.format).is_none() { continue; }
            match load_base_texture(&c, t) {
                Ok(_) => ok += 1,
                Err(_) if i.format == 38 => dxn_fail += 1,
                Err(e) => {
                    fail += 1;
                    let msg = e.to_string();
                    let key = msg.split(" (").next().unwrap_or(&msg).split(" size ").next().unwrap_or(&msg).to_string();
                    *reasons.entry(format!("fmt {} {}", i.format, key)).or_default() += 1;
                }
            }
        }
        eprintln!("bitmaps loaded {ok} failed {fail} (dxn failed {dxn_fail}) reasons {reasons:?}");
        assert!(ok > 1500 && fail * 10 < ok, "loaded {ok} failed {fail}");
    }
}
