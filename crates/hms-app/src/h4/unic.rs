//! Halo 4 in-cache UI strings (`unic` multilingual_unicode_string_list tags).
//!
//! Halo 4 does not keep the strings inside the `unic` tags: every language's strings of the whole
//! map live in ONE table per language at the end of the cache file, and a `unic` tag only names a
//! contiguous window of it. Checked on all 50 MP / FF / campaign caches that carry a `matg`
//! (17 / 17 languages each, `cargo test --release -p hms-app h4::unic`):
//!
//!   matg +0x08: 17 language packs of 0x50 bytes (English first; enum order in
//!               docs/halo4_forge_palette.md): u32 string count @+0x10, u32 data size @+0x14,
//!               u32 reference table offset @+0x18, u32 string data offset @+0x1C.
//!   those offsets are ADDRESSES inside header section 3 (the locale section: address @1260,
//!               size @1264); file offset = address + section offset mask[3] (@1232), 32-bit
//!               wrapping (m30_cryptum's mask is 0xFF4AA000) - the same `file = address + mask`
//!               convention cache.rs uses for the tag section. Section 3 ends exactly at the end
//!               of the file on every cache, and the English reference table starts at its
//!               address (holds on all 50).
//!   reference table: count x {u32 string id, u32 byte offset into the string data}, offsets
//!               non-decreasing, string data = NUL-terminated UTF-8.
//!   unic +0x2C: 17 x u32 = `u16 first reference index | u16 count << 16` per language, the
//!               tag's window into that language's table; the windows of all 45 `unic` tags of
//!               Ravine tile the English table exactly (0..14597, no gaps).
//!
//! The Forge palette display names come from `ui\strings\forge` (`ff_weapons_human` ->
//! "Weapons, Human", `gad_fusion_coil` -> "Fusion Coil"): every palette / entry / variant name
//! string id of every MP map resolves there (tested).

use std::collections::HashMap;

use super::cache::{ByteRead, H4Cache};

/// Header offsets (Halo 4 = Reach U13 minus 8 bytes, see cache.rs): section offset masks
/// (4 x u32) @1220, section table (4 x {u32 address, u32 size}) @1236; the locale section is 3.
const HDR_SECTION_OFFSETS: usize = 1220;
const HDR_SECTIONS: usize = 1236;
const LOCALE_SECTION: usize = 3;
/// matg language pack stride / start.
pub const MATG_LANG_PACKS: usize = 0x08;
pub const MATG_LANG_PACK_SIZE: usize = 0x50;
pub const LANGUAGE_COUNT: usize = 17;
/// `unic` +0x2C: per-language `{u16 first index, u16 count}`.
pub const OFF_UNIC_LANG_RANGES: usize = 0x2C;
/// Language 0 = English (matg enum: English, Japanese, German, French, Spanish, Mexican Spanish,
/// Italian, Korean, Chinese Traditional, Chinese Simplified, Portuguese, Polish, Russian, Danish,
/// Finnish, Dutch, Norwegian).
pub const LANG_ENGLISH: usize = 0;

/// One language's whole string table of a cache: `entries[i] = (string id, text)` in table
/// order, plus a first-occurrence index by string id (a sid can repeat across `unic` tags:
/// `default`, `none` ... - use `UnicWindow::get` for a specific tag).
#[derive(Debug, Default)]
pub struct LocaleTable {
    pub language: usize,
    pub entries: Vec<(u32, String)>,
    by_sid: HashMap<u32, usize>,
}

/// The locale section as (address, size, file offset of its start): file = address + mask[3]
/// (wrapping u32); None when that range does not lie inside the file.
pub fn locale_section(c: &H4Cache) -> Option<(u32, u32, usize)> {
    let d = c.data();
    let mask = d.u32_at(HDR_SECTION_OFFSETS + LOCALE_SECTION * 4);
    let addr = d.u32_at(HDR_SECTIONS + LOCALE_SECTION * 8);
    let size = d.u32_at(HDR_SECTIONS + LOCALE_SECTION * 8 + 4);
    let start = addr.wrapping_add(mask) as usize;
    if size == 0 || start.checked_add(size as usize)? > d.len() { return None; }
    Some((addr, size, start))
}

/// File offset of a locale-section address (matg language pack offsets), None when outside it.
pub fn locale_file_offset(c: &H4Cache, addr: u32) -> Option<usize> {
    let (base, size, start) = locale_section(c)?;
    let rel = addr.checked_sub(base)?;
    if rel >= size { return None; }
    Some(start + rel as usize)
}

impl LocaleTable {
    /// Parse language `lang` (0 = English) of the cache's `matg`; None when the cache has no
    /// `matg` (shared / campaign resource caches) or the offsets fall outside the file.
    pub fn load(c: &H4Cache, lang: usize) -> Option<LocaleTable> {
        if lang >= LANGUAGE_COUNT { return None; }
        let d = c.data();
        let &matg = c.find_tags(b"matg").first()?;
        let m = c.tag_meta(matg)?;
        let p = m + MATG_LANG_PACKS + lang * MATG_LANG_PACK_SIZE;
        let count = d.u32_at(p + 0x10) as usize;
        let size = d.u32_at(p + 0x14) as usize;
        let refs = locale_file_offset(c, d.u32_at(p + 0x18))?;
        let data = locale_file_offset(c, d.u32_at(p + 0x1C))?;
        if count == 0 || count > 0x10_0000 || refs + count * 8 > d.len() || data + size > d.len() { return None; }
        let mut entries = Vec::with_capacity(count);
        let mut by_sid = HashMap::with_capacity(count);
        for i in 0..count {
            let sid = d.u32_at(refs + i * 8);
            let off = d.u32_at(refs + i * 8 + 4) as usize;
            if off >= size { return None; }
            let s = d.cstr_at(data + off, size - off);
            by_sid.entry(sid).or_insert(i);
            entries.push((sid, s));
        }
        Some(LocaleTable { language: lang, entries, by_sid })
    }

    /// First string carrying this string id anywhere in the language table.
    pub fn get(&self, sid: u32) -> Option<&str> { self.by_sid.get(&sid).map(|&i| self.entries[i].1.as_str()) }

    /// The window of a `unic` tag in this language (`unic` +0x2C + 4 * language).
    pub fn window(&self, c: &H4Cache, unic_tag: usize) -> Option<UnicWindow> {
        if c.tag_class(unic_tag).as_ref() != Some(b"unic") { return None; }
        let m = c.tag_meta(unic_tag)?;
        let v = c.data().u32_at(m + OFF_UNIC_LANG_RANGES + self.language * 4);
        let (first, count) = ((v & 0xFFFF) as usize, (v >> 16) as usize);
        if first + count > self.entries.len() { return None; }
        Some(UnicWindow { first, count })
    }

    /// The window of the `unic` tag with this path (`ui\strings\forge`).
    pub fn window_of(&self, c: &H4Cache, path: &str) -> Option<UnicWindow> {
        let t = c.find_tags(b"unic").into_iter().find(|&t| c.tag_name(t).eq_ignore_ascii_case(path))?;
        self.window(c, t)
    }

    /// Look a string id up inside a window first, then anywhere in the language.
    pub fn get_in(&self, w: Option<UnicWindow>, sid: u32) -> Option<&str> {
        if let Some(w) = w {
            if let Some((_, s)) = self.entries[w.first..w.first + w.count].iter().find(|(k, _)| *k == sid) { return Some(s.as_str()); }
        }
        self.get(sid)
    }
}

/// A `unic` tag's slice of a language table (`entries[first .. first + count]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnicWindow {
    pub first: usize,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;

    /// Every cache with a `matg`: all 17 language tables parse, and the 45 `unic` windows of the
    /// English table tile it without gaps or overlaps (checked on every map that has unic tags).
    #[test]
    fn locale_tables_parse_on_every_cache() {
        let Some(dir) = maps_dir() else { eprintln!("skip: no halo4 maps"); return };
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "map") && !p.to_string_lossy().contains(" - Copy")).collect();
        files.sort();
        let mut checked = 0;
        for p in files {
            let Ok(c) = H4Cache::open(&p) else { continue };
            if c.find_tags(b"matg").is_empty() { continue; }
            let name = c.map_name.clone();
            let (addr, size, start) = locale_section(&c).unwrap_or_else(|| panic!("{name}: no locale section"));
            assert_eq!(start + size as usize, c.data().len(), "{name}: the locale section does not end at EOF");
            // the English reference table is the first thing in the section
            let m = c.tag_meta(c.find_tags(b"matg")[0]).unwrap();
            assert_eq!(c.data().u32_at(m + MATG_LANG_PACKS + 0x18), addr, "{name}: English refs not at the section start");
            for lang in 0..LANGUAGE_COUNT {
                let t = LocaleTable::load(&c, lang).unwrap_or_else(|| panic!("{name}: language {lang} table"));
                assert!(!t.entries.is_empty(), "{name}: language {lang} empty");
            }
            let en = LocaleTable::load(&c, LANG_ENGLISH).unwrap();
            let mut windows: Vec<UnicWindow> = c.find_tags(b"unic").iter().filter_map(|&t| en.window(&c, t)).collect();
            windows.sort_by_key(|w| w.first);
            let mut next = 0;
            for w in &windows {
                assert_eq!(w.first, next, "{name}: unic windows do not tile the English table");
                next += w.count;
            }
            if !windows.is_empty() { assert_eq!(next, en.entries.len(), "{name}: unic windows end before the table does"); }
            checked += 1;
        }
        eprintln!("locale tables ok on {checked} caches");
        assert!(checked > 0);
    }

    #[test]
    fn ravine_forge_strings() {
        let Some(p) = maps_dir().map(|d| d.join("ca_forge_ravine.map")).filter(|p| p.is_file()) else { eprintln!("skip"); return };
        let c = H4Cache::open(&p).unwrap();
        let en = LocaleTable::load(&c, LANG_ENGLISH).expect("english table");
        assert_eq!(en.entries.len(), 14597);
        assert_eq!(en.entries[0].1, "Checkpoint...");
        let w = en.window_of(&c, "ui\\strings\\forge").expect("forge unic");
        assert_eq!((w.first, w.count), (10172, 832));
        let forge: Vec<&str> = en.entries[w.first..w.first + w.count].iter().map(|(_, s)| s.as_str()).collect();
        for want in ["Fusion Coil", "Weapons, Human", "Block, 1X1", "Initial Spawn", "Gadgets (MCC)"] {
            assert!(forge.contains(&want), "'{want}' missing from ui\\strings\\forge");
        }
    }
}
