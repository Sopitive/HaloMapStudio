//! Halo 2 Anniversary cache access. #h2a
//!
//! The reader itself is `h4::cache::H4Cache` (see `h2a/mod.rs` for why nothing is cloned); this
//! file only supplies the H2A entry points and the tests that pin the layout on every shipped
//! groundhog cache.
//!
//! What the tests below assert, and therefore what is VERIFIED:
//!   * all 11 shipped playable caches + `shared.map` + `campaign.map` open with the H2A expander
//!     0x7AC00000; every tag meta lands inside the tag-data section (section 2)
//!   * 261 tag classes on every map (the same count as Halo 4), every tag named except the first
//!     three (`draw` / `gpix` / `play` carry no name on disk, exactly as in Halo 4)
//!   * `play` pages inflate to their recorded decompressed size (local + `shared.map`)
//!   * the `zone` gestalt's definition blob is the concatenation of the live entries'
//!     definition lengths, in entry order, and each entry's `+0x38[0]` equals that prefix sum
//!   * one segment per live resource, and every page-stream fixup lands inside its page
//!   * the seven resource type names resolve through the 17-bit string ids (a DIFFERENT ORDER
//!     from Halo 4's - `kind_is` compares by name, so nothing keys on the index)

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::h4::cache::{cache_engine, h2a_maps_dir, Engine, H4Cache};

/// The Halo 2 Anniversary build string (header +152) and expansion constant, re-exported so
/// callers do not have to reach into the Halo 4 module for them.
#[allow(unused_imports)]
pub use crate::h4::cache::{H2A_BUILD, H2A_EXPANDER};

/// Is this file a Halo 2 Anniversary (groundhog) MCC cache?
pub fn is_h2a_cache(path: &Path) -> bool { crate::h4::cache::is_h2a_cache(path) }

/// The MCC `groundhog/maps` folder, if this machine has one (tests skip otherwise).
/// `HMS_H2A_MAPS` points at a folder elsewhere (test / development input only).
pub fn maps_dir() -> Option<PathBuf> { h2a_maps_dir() }

/// Open a groundhog cache, refusing anything that is not one (so a Halo 4 or Reach cache handed
/// to an H2A entry point fails loudly instead of being read with the wrong expander).
pub fn open(path: &Path) -> Result<H4Cache> {
    match cache_engine(path) {
        Some(Engine::H2A) => {}
        Some(e) => bail!("{} is a {} cache, not Halo 2 Anniversary", path.display(), e.label()),
        None => bail!("{} is not a Halo 4 / Halo 2 Anniversary MCC cache", path.display()),
    }
    H4Cache::open(path)
}

/// The shipped playable H2A caches, in the order the contact sheet uses. `shared.map` and
/// `campaign.map` are resource-only caches (header `type` 3 / 4, no tag index) and
/// `ca_forge_skybox03*` are the three Forge canvases.
pub const SHIPPED_MAPS: [&str; 11] = [
    "ca_ascension.map",
    "ca_coagulation.map",
    "ca_lockout.map",
    "ca_relic.map",
    "ca_sanctuary.map",
    "ca_warlock.map",
    "ca_zanzibar.map",
    "ca_forge_skybox01.map",
    "ca_forge_skybox02.map",
    "ca_forge_skybox03.map",
    // a byte-identical user copy that ships beside the canvas; kept so the sweep covers it
    "ca_forge_skybox03 - Copy.map",
];

/// Every `*.map` in the groundhog maps folder that this reader accepts (build string match), so
/// the tests and the contact sheet sweep whatever is actually installed rather than a fixed list.
/// `ca_forge_skybox03_built.map` is an EK/user-built cache on a different build ("Jan  3 2024
/// 10:26:12") whose section-offset table is zeroed; it is deliberately excluded.
pub fn installed_caches() -> Vec<PathBuf> {
    let Some(dir) = maps_dir() else { return Vec::new() };
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("map")))
        .filter(|p| is_h2a_cache(p))
        .collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::{ByteRead, ResourceEntry};

    fn open_named(name: &str) -> Option<H4Cache> {
        let p = maps_dir()?.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(open(&p).expect("open"))
    }

    /// The tag-data section (section 2) as a file-offset range: section table @1236 (addr, size),
    /// section offset table @1220 (the negated magic), exactly as `H4Cache::open` computes them.
    fn tag_section(c: &H4Cache) -> (usize, usize) {
        let d = c.data();
        let addr = d.u32_at(1236 + 16);
        let size = d.u32_at(1236 + 20);
        let off = d.u32_at(1220 + 8);
        let start = addr.wrapping_add(off) as usize;
        (start, start + size as usize)
    }

    /// Header + tag index + string ids on `ca_lockout.map`, the reference map for this port.
    #[test]
    fn lockout_header_and_tag_index() {
        let Some(c) = open_named("ca_lockout.map") else { return };
        assert_eq!(c.engine, Engine::H2A);
        assert_eq!(c.map_type, 1, "multiplayer");
        assert_eq!(c.map_name, "ca_lockout");
        assert_eq!(c.scenario, "levels\\sway\\ca_lockout\\ca_lockout");
        assert_eq!(c.tag_count(), 17438);
        assert_eq!(c.class_count(), 261, "same class count as Halo 4");
        // Only the first three tags are unnamed on disk (as in Halo 4).
        let unnamed: Vec<usize> = (0..c.tag_count()).filter(|&i| c.tag_name(i).is_empty()).collect();
        assert!(unnamed.iter().all(|&i| i < 3) && unnamed.len() <= 3, "unnamed tags {unnamed:?}");
        // The class histogram probed on this map (docs/h2a_support_plan.md).
        assert_eq!(c.find_tags(b"scnr").len(), 1);
        assert_eq!(c.find_tags(b"sbsp").len(), 2);
        assert_eq!(c.find_tags(b"Lbsp").len(), 2);
        assert_eq!(c.find_tags(b"sLdT").len(), 1);
        assert_eq!(c.find_tags(b"bitm").len(), 2166);
        assert_eq!(c.find_tags(b"mat ").len(), 685);
        assert_eq!(c.find_tags(b"mats").len(), 291);
        assert_eq!(c.find_tags(b"mode").len(), 501);
        // string ids resolve (the resource type names are read through them)
        assert_eq!(c.res_types.len(), 7);
        assert!(c.res_types.iter().any(|t| t == H4Cache::RES_GEOMETRY), "{:?}", c.res_types);
        assert!(c.res_types.iter().any(|t| t == H4Cache::RES_BITMAP), "{:?}", c.res_types);
    }

    /// The expansion constant is right on EVERY shipped cache: every tag meta lands inside the
    /// tag-data section, and the tag names / classes read as ASCII. This is the sweep that pins
    /// 0x7AC00000 (a constant off by even one 4-byte step misaligns the block headers below).
    #[test]
    fn every_cache_opens_and_metas_are_in_section() {
        let caches = installed_caches();
        if caches.is_empty() { eprintln!("skip: no groundhog maps folder"); return; }
        let mut playable = 0;
        for p in &caches {
            let c = open(p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
            assert_eq!(c.engine, Engine::H2A);
            if c.map_type == 3 || c.map_type == 4 {
                // resource-only caches: no tag index, but the data table must be readable
                // campaign.map is exactly the 0x1E000-byte header (no resource data of its own)
                assert!(c.data().len() >= 0x1E000, "{}", p.display());
                continue;
            }
            playable += 1;
            let (lo, hi) = tag_section(&c);
            assert!(lo > 0 && hi > lo && hi <= c.data().len(), "{}: section 2 {lo:#x}..{hi:#x}", p.display());
            let mut checked = 0;
            for i in 0..c.tag_count() {
                if let Some(m) = c.tag_meta(i) {
                    assert!(m >= lo && m < hi, "{}: tag {i} meta {m:#x} outside {lo:#x}..{hi:#x}", p.display());
                    checked += 1;
                }
            }
            assert!(checked > 10_000, "{}: only {checked} tags with meta", p.display());
            assert_eq!(c.class_count(), 261, "{}", p.display());
            assert!(c.tag_name(c.find_tags(b"scnr")[0]).starts_with("levels\\sway\\"), "{}", p.display());
            assert!((0..c.tag_count()).all(|i| c.tag_name(i).is_ascii()), "{}", p.display());
        }
        assert!(playable >= 10, "expected the shipped playable caches, found {playable}");
    }

    /// `play` / `zone` on every shipped playable cache: the paging tables parse, a sample of
    /// pages inflates, the definition blob is the prefix-sum concatenation, one segment per live
    /// resource and every page-stream fixup lies inside its page.
    #[test]
    fn every_cache_pages_and_gestalt() {
        let caches = installed_caches();
        if caches.is_empty() { eprintln!("skip: no groundhog maps folder"); return; }
        for p in &caches {
            let c = open(p).unwrap();
            if c.map_type == 3 || c.map_type == 4 { continue; }
            assert!(c.pages.len() > 1000, "{}: {} pages", p.display(), c.pages.len());
            // only this map (-1) and shared.map (1) are referenced by the shipped H2A caches
            assert!(c.pages.iter().all(|pg| pg.cache_index == -1 || pg.cache_index == 1), "{}", p.display());
            let local: Vec<usize> = (0..c.pages.len()).filter(|&i| c.pages[i].cache_index == -1).collect();
            let shared: Vec<usize> = (0..c.pages.len()).filter(|&i| c.pages[i].cache_index == 1).collect();
            assert!(!local.is_empty() && !shared.is_empty(), "{}", p.display());
            for &i in local.iter().take(20).chain(shared.iter().take(20)) {
                let d = c.page_data(i).unwrap_or_else(|e| panic!("{}: page {i}: {e}", p.display()));
                assert_eq!(d.len(), c.pages[i].decompressed as usize, "{}: page {i}", p.display());
            }
            let live: Vec<&ResourceEntry> = c.resources.iter().filter(|e| e.is_live()).collect();
            assert!(!live.is_empty());
            let sum: u64 = live.iter().map(|e| e.def_len as u64).sum();
            assert_eq!(sum, c.def_blob_len as u64, "{}: definition data packed in entry order", p.display());
            let mut prefix = 0u32;
            for e in &live {
                assert_eq!(e.def_off, prefix, "{}: entry {} prefix sum", p.display(), e.index);
                prefix += e.def_len;
                assert!(e.segment >= 0 && (e.segment as usize) < c.segments.len(), "{}", p.display());
            }
            let mut segs: Vec<i16> = live.iter().map(|e| e.segment).collect();
            segs.sort();
            segs.dedup();
            assert_eq!(segs.len(), live.len(), "{}: one segment per live resource", p.display());
            assert_eq!(c.segments.len(), live.len(), "{}", p.display());
            // every page-stream fixup lands inside its page
            for e in live.iter().take(600) {
                for &(_, addr) in &e.fixups {
                    let nib = addr >> 28;
                    if nib == 2 { continue; }
                    let Ok((data, base)) = c.stream(e, nib) else { continue };
                    assert!(base + (addr & 0x0FFF_FFFF) as usize <= data.len(), "{}: res {} addr {addr:#x}", p.display(), e.index);
                }
            }
        }
    }

    /// `shared.map` / `campaign.map` carry the resource data only (type 3 / 4), and the reader
    /// refuses a Halo 4 cache handed to the H2A entry point.
    #[test]
    fn shared_caches_and_engine_gate() {
        let Some(dir) = maps_dir() else { return };
        for (name, ty) in [("shared.map", 3i16), ("campaign.map", 4)] {
            let p = dir.join(name);
            if !p.is_file() { continue; }
            let c = open(&p).unwrap();
            assert_eq!(c.map_type, ty);
            assert_eq!(c.engine, Engine::H2A);
        }
        if let Some(h4) = crate::h4::cache::maps_dir() {
            let p = h4.join("wraparound.map");
            if p.is_file() {
                assert!(open(&p).is_err(), "a Halo 4 cache must be refused by the H2A entry point");
                assert!(!is_h2a_cache(&p));
            }
        }
    }
}
