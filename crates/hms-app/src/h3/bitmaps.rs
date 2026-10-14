//! Halo 3 (MCC) bitmaps: the `bitm` element + the `bitmap_texture_interop_resource` pixel bytes.
//!
//! Layout (Halo3MCC/bitm.xml, verified on guardian):
//!   bitm main: Sequences @0x54, Bitmaps @0x60 (0x38-B elements), Hardware Textures @0x8C
//!   Bitmaps element (0x38): signature "mtib" @0, W i16 @4, H i16 @6, depth i8 @8, moreFlags @9,
//!     type u16 @0xA (0 2D, 1 3D, 2 cube, 3 array), format u16 @0xC (classic Halo enum, below),
//!     flags16 @0xE (bit1 compressed, bit3 swizzled), mip count i8 @0x14 (levels = value + 1),
//!     curve @0x15, pixels offset i32 @0x18, pixels size i32 @0x1C, hi-res offset/size @0x20/@0x24,
//!     hardware format i32 @0x28
//! The pixel bytes live in the bitm-owned `bitmap_texture_interop_resource` (the same gestalt /
//! DEFLATE page as geometry), sliced at `pixels_offset` for `pixels_size` bytes. MCC-PC data is
//! untiled/linear (no X360 untwiddle).

use anyhow::{anyhow, Result};
use std::sync::Arc;

use super::cache::H3Cache;
use crate::h4::cache::ByteRead;

pub const OFF_BITM_BITMAPS: usize = 0x60;
pub const BITMAP_ELEM: usize = 0x38;

#[derive(Clone, Copy, Debug)]
pub struct BitmapInfo {
    pub w: u16,
    pub h: u16,
    pub depth: u8,
    pub kind: u16,
    pub format: u16,
    /// Levels INCLUDING the base (element mip-count byte + 1).
    pub levels: u32,
    pub px_off: i32,
    pub px_size: u32,
    pub curve: u8,
}

pub fn bitmap_info(c: &H3Cache, tag: usize) -> Option<BitmapInfo> {
    let m = c.tag_meta(tag)?;
    let (_, o) = c.block(m + OFF_BITM_BITMAPS)?;
    let d = c.data();
    Some(BitmapInfo {
        w: d.u16_at(o + 4),
        h: d.u16_at(o + 6),
        depth: d.u8_at(o + 8),
        kind: d.u16_at(o + 0xA),
        format: d.u16_at(o + 0xC),
        levels: d.u8_at(o + 0x14) as u32 + 1,
        px_off: d.i32_at(o + 0x18),
        px_size: d.u32_at(o + 0x1C).max(0),
        curve: d.u8_at(o + 0x15),
    })
}

/// Bytes per 4x4 block for a BCn format, or bytes-per-pixel (negative) for an uncompressed one,
/// plus a short name. None = a format this reader does not model.
pub fn format_desc(format: u16) -> Option<(i32, &'static str)> {
    Some(match format {
        0x0E => (8, "BC1/DXT1"),
        0x0F => (16, "BC2/DXT3"),
        0x10 => (16, "BC3/DXT5"),
        0x1E | 0x1F | 0x23 | 0x24 | 0x25 | 0x26 => (8, "BC4/DXT5a"),
        0x21 | 0x27 => (16, "BC5/DXN"),
        0x0B => (-4, "A8R8G8B8"),
        0x0A => (-4, "X8R8G8B8"),
        0x04 => (-1, "R8"),
        0x00 | 0x01 | 0x02 => (-1, "A8/Y8"),
        0x03 => (-2, "A8Y8"),
        0x06 | 0x08 | 0x09 => (-2, "16bpp"),
        _ => return None,
    })
}

/// True when the colour curve marks an sRGB-encoded colour map (0 = xRGB linear, 1 = sRGB,
/// 2 = gamma2, 3 = linear). Colour maps decode as sRGB; normal/bump maps stay linear.
pub fn is_srgb(curve: u8) -> bool { curve == 1 }

/// Bytes a full mip chain of `levels` occupies for the given format at (w,h).
pub fn chain_bytes(format: u16, w: u16, h: u16, levels: u32) -> Option<usize> {
    let (bpb, _) = format_desc(format)?;
    let mut total = 0usize;
    let (mut mw, mut mh) = (w as usize, h as usize);
    for _ in 0..levels.max(1) {
        total += if bpb > 0 {
            ((mw + 3) / 4).max(1) * ((mh + 3) / 4).max(1) * bpb as usize
        } else {
            mw * mh * (-bpb) as usize
        };
        mw = (mw / 2).max(1);
        mh = (mh / 2).max(1);
    }
    Some(total)
}

/// The bitm-owned `bitmap_texture_interop_resource` for this tag.
fn bitmap_resource<'a>(c: &'a H3Cache, tag: usize) -> Option<&'a super::cache::ResourceEntry> {
    c.resources.iter().find(|e| e.is_live() && e.owner == *b"bitm"
        && e.owner_tag == tag && c.kind_is(e.kind, H3Cache::RES_BITMAP))
}

/// Raw pixel bytes of a bitmap (the full stream slice at `pixels_offset`, `pixels_size` long) plus
/// its info. The bytes are the untiled BCn / uncompressed mip chain, largest-first.
pub fn bitmap_pixels(c: &H3Cache, tag: usize) -> Result<(BitmapInfo, Vec<u8>)> {
    let info = bitmap_info(c, tag).ok_or_else(|| anyhow!("no bitmap element for tag {tag}"))?;
    let e = bitmap_resource(c, tag).ok_or_else(|| anyhow!("no bitmap resource for tag {tag}"))?;
    let (page, base): (Arc<Vec<u8>>, usize) = c.read_resource(e)?;
    let s = base + info.px_off.max(0) as usize;
    let n = info.px_size as usize;
    let bytes = page.get(s..s + n).ok_or_else(|| bail_range(s, n, page.len()))?.to_vec();
    Ok((info, bytes))
}

fn bail_range(s: usize, n: usize, len: usize) -> anyhow::Error {
    anyhow!("pixel range {s}+{n} past page ({len})")
}

/// Like [`bitmap_pixels`] but CLAMPED to the bytes actually present in the page. Many bitmaps store
/// only the base mip LEVEL in their page (page size == mip 0), while `pixels_size` records the full
/// 4/3 mip chain, so the exact slice overruns the page. The returned bytes are the largest-first
/// prefix that fits (always at least mip 0) - enough for the base-colour render path.
pub fn bitmap_pixels_avail(c: &H3Cache, tag: usize) -> Result<(BitmapInfo, Vec<u8>)> {
    let info = bitmap_info(c, tag).ok_or_else(|| anyhow!("no bitmap element for tag {tag}"))?;
    let e = bitmap_resource(c, tag).ok_or_else(|| anyhow!("no bitmap resource for tag {tag}"))?;
    let (page, base): (Arc<Vec<u8>>, usize) = c.read_resource(e)?;
    let s = base + info.px_off.max(0) as usize;
    if s >= page.len() { return Err(anyhow!("pixel offset {s} past page ({})", page.len())); }
    // `pixels_size` 0 (object textures paged from a shared cache: riverworld's weapons /
    // vehicles) = the resource's secondary page IS the pixel data, mip 0 exactly (verified:
    // battle_rifle 512x512 DXT5 -> a 262144 B page).
    let want = if info.px_size > 0 { info.px_size as usize } else { page.len() - s };
    let n = want.min(page.len() - s);
    Ok((info, page[s..s + n].to_vec()))
}

/// The wgpu texture format for a bitmap, or None for one this reader does not model. `srgb`
/// selects the *_SRGB view for colour maps.
pub fn wgpu_format(format: u16, srgb: bool) -> Option<eframe::wgpu::TextureFormat> {
    use eframe::wgpu::TextureFormat as F;
    Some(match format {
        0x0E => if srgb { F::Bc1RgbaUnormSrgb } else { F::Bc1RgbaUnorm },
        0x0F => if srgb { F::Bc2RgbaUnormSrgb } else { F::Bc2RgbaUnorm },
        0x10 => if srgb { F::Bc3RgbaUnormSrgb } else { F::Bc3RgbaUnorm },
        0x1E | 0x1F | 0x23 | 0x24 | 0x25 | 0x26 => F::Bc4RUnorm,
        0x21 | 0x27 => F::Bc5RgUnorm,
        0x0B | 0x0A => if srgb { F::Bgra8UnormSrgb } else { F::Bgra8Unorm },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::cache::maps_dir;

    fn open(name: &str) -> Option<H3Cache> {
        let p = maps_dir()?.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H3Cache::open(&p).expect("open"))
    }

    #[test]
    fn guardian_bitmaps_decode() {
        let Some(c) = open("guardian.map") else { return };
        let bitms = c.find_tags(b"bitm");
        assert_eq!(bitms.len(), 2116);
        // every bitmap has a readable element with a modelled format on the vast majority
        let mut ok = 0;
        let mut checked = 0;
        for &t in bitms.iter() {
            let Some(info) = bitmap_info(&c, t) else { continue };
            if info.w == 0 || info.h == 0 || info.px_size == 0 { continue; }
            if format_desc(info.format).is_none() { continue; }
            checked += 1;
            // read the pixel bytes and confirm the slice length matches the recorded size
            if let Ok((_, px)) = bitmap_pixels(&c, t) {
                assert_eq!(px.len(), info.px_size as usize, "tag {t}");
                ok += 1;
            }
            if checked >= 200 { break; }
        }
        assert!(ok > 150, "decoded {ok}/{checked} bitmap pixel streams");
    }

    /// Probe (print-only): the object textures paged from a SHARED cache (`Page::cache_index`
    /// >= 0) - which file the index names, the page the reader picks, and whether the pixel bytes
    /// are real texels (a zero-filled slice = a wrong page/offset, drawn as a black object).
    #[test]
    fn guardian_object_bitmap_bytes() {
        let Some(c) = open("guardian.map") else { return };
        eprintln!("shared files: {:?}", c.shared_files);
        for t in c.find_tags(b"bitm") {
            let n = c.tag_name(t);
            if !(n.ends_with("gravity_hammer") || n.ends_with("battle_rifle") || n.ends_with("shotgun") || n.ends_with("man_cannon_guardian_iii") || n.contains("gravity_hammer\\")) { continue; }
            let Some(info) = bitmap_info(&c, t) else { continue };
            let e = bitmap_resource(&c, t);
            let seg = e.and_then(|e| c.segments.get(e.loc.max(0) as usize));
            let pages = seg.map(|s| (c.pages.get(s.primary_page.max(0) as usize).copied(), c.pages.get(s.secondary_page.max(0) as usize).copied()));
            match bitmap_pixels_avail(&c, t) {
                Ok((_, px)) => {
                    let zeros = px.iter().filter(|&&b| b == 0).count();
                    eprintln!("{}: {}x{} fmt {:#x} levels {} px_off {} px_size {} -> {} B, {:.1}% zero, head {:02x?} | seg {:?} pages {:?}",
                        n.rsplit('\\').next().unwrap_or(""), info.w, info.h, info.format, info.levels, info.px_off, info.px_size, px.len(), zeros as f64 * 100.0 / px.len().max(1) as f64, &px[..16.min(px.len())], seg, pages);
                }
                Err(err) => eprintln!("{}: {}x{} fmt {:#x} -> ERR {err} | seg {:?} pages {:?}", n.rsplit('\\').next().unwrap_or(""), info.w, info.h, info.format, seg, pages),
            }
        }
    }
}
