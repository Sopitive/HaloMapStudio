//! Halo 2 Anniversary textures. #h2a
//!
//! The decoder is `h4::bitmaps` unchanged: `bitm` bitmaps block at +0x7C (48-B elements: `w u16
//! @0, h u16 @2, depth u8 @4, type u8 @6, format i16 @8, mip count u8 @16`), resources block at
//! +0xA8, and the same `bitmap_texture_interop_resource` definition (per-stream `size i32 @k*20`,
//! page address fixup @`k*20+12`, DXGI format byte @0x4C). The tests here assert that on every
//! shipped groundhog cache and record the format histogram, which is the H2A-vs-Halo-4 delta
//! that matters for the renderer.

use anyhow::Result;

use crate::h4::bitmaps::{bitmap_dxgi, bitmap_info, load_base_texture, BitmapInfo, Texel};
use crate::h4::cache::H4Cache;

/// Texture info for a `bitm` tag (None when the tag has no bitmap element).
pub fn info(c: &H4Cache, tag: usize) -> Option<BitmapInfo> { bitmap_info(c, tag) }

/// The DXGI format the engine creates this bitmap with.
pub fn dxgi(c: &H4Cache, tag: usize) -> Option<u32> { bitmap_dxgi(c, tag) }

/// A 2D bitmap's mip chain, ready for the renderer.
pub fn load(c: &H4Cache, tag: usize) -> Result<Texel> { load_base_texture(c, tag) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h2a::cache::{installed_caches, maps_dir, open};
    use std::collections::BTreeMap;

    /// Every `bitm` on ca_lockout reads a plausible element, and the format histogram is recorded
    /// (formats the decoder does not handle yet are counted, not asserted).
    #[test]
    fn lockout_bitmap_elements() {
        let Some(dir) = maps_dir() else { return };
        let p = dir.join("ca_lockout.map");
        if !p.is_file() { return }
        let c = open(&p).unwrap();
        let bitms = c.find_tags(b"bitm");
        assert_eq!(bitms.len(), 2166);
        let mut fmt: BTreeMap<i16, usize> = BTreeMap::new();
        let mut kinds: BTreeMap<u8, usize> = BTreeMap::new();
        let mut with_element = 0;
        for &t in &bitms {
            let Some(i) = info(&c, t) else { continue };
            with_element += 1;
            *fmt.entry(i.format).or_default() += 1;
            *kinds.entry(i.kind).or_default() += 1;
            assert!(i.w > 0 && i.h > 0 && i.w <= 8192 && i.h <= 8192, "bitm {t} {}x{}", i.w, i.h);
            assert!(i.levels >= 1 && i.levels <= 16, "bitm {t} levels {}", i.levels);
        }
        assert!(with_element * 100 / bitms.len() >= 99, "{with_element}/{} bitmaps carry an element", bitms.len());
        eprintln!("h2a lockout bitmap formats {fmt:?} kinds {kinds:?}");
        // the BSP materials' diffuse maps must actually decode
        let bsp = crate::h4::geometry::load_bsp(&c, c.find_tags(b"sbsp")[0]).unwrap();
        let mut ok = 0;
        let mut fail = 0;
        for m in &bsp.materials {
            let Some(b) = m.diffuse_bitm else { continue };
            match load(&c, b) {
                Ok(Texel::Dds { w, h, mips, bytes }) => { assert!(w > 0 && h > 0 && mips >= 1 && !bytes.is_empty()); ok += 1; }
                Ok(Texel::Bgra { w, h, bytes }) => { assert_eq!(bytes.len(), (w * h * 4) as usize); ok += 1; }
                Err(_) => fail += 1,
            }
        }
        eprintln!("h2a lockout BSP diffuse textures: {ok} decoded, {fail} unsupported");
        assert!(ok > 0 && ok * 100 / (ok + fail).max(1) >= 80, "{ok} ok / {fail} failed");
    }

    /// Sweep every shipped cache: bitmap elements are plausible, and each BSP's material diffuse
    /// maps mostly decode. Records the aggregate format histogram.
    #[test]
    fn every_cache_bitmaps() {
        let caches = installed_caches();
        if caches.is_empty() { eprintln!("skip: no groundhog maps folder"); return; }
        let mut fmt: BTreeMap<i16, usize> = BTreeMap::new();
        let mut ok = 0usize;
        let mut fail = 0usize;
        for p in &caches {
            let c = open(p).unwrap();
            if c.map_type == 3 || c.map_type == 4 { continue; }
            for &t in c.find_tags(b"bitm").iter() {
                let Some(i) = info(&c, t) else { continue };
                *fmt.entry(i.format).or_default() += 1;
                assert!(i.w > 0 && i.h > 0 && i.w <= 8192 && i.h <= 8192, "{}: bitm {t}", p.display());
            }
            for sb in c.find_tags(b"sbsp") {
                let Ok(bsp) = crate::h4::geometry::load_bsp(&c, sb) else { continue };
                for m in bsp.materials.iter().filter_map(|m| m.diffuse_bitm) {
                    if load(&c, m).is_ok() { ok += 1 } else { fail += 1 }
                }
            }
        }
        eprintln!("h2a bitmap sweep: formats {fmt:?}; BSP diffuse {ok} decoded / {fail} unsupported");
        assert!(ok > 500, "only {ok} BSP diffuse textures decoded");
        assert!(ok * 100 / (ok + fail).max(1) >= 80, "{ok} ok / {fail} failed");
    }
}
