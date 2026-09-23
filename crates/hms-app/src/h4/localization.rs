//! MCC UI localization tables (`<MCC>/data/ui/Localization/<LANG>_<Title>.bin`).
//!
//! Shipped Halo 4 (and a few Reach) map variants store LOCALIZATION KEYS instead of literal
//! titles / descriptions (`$h4_mvar_settler_name`, `$H4_MPMap_4-3_Title`); the game resolves them
//! through these tables at display time. This module parses the tables and resolves `$key`
//! strings to the English text. Format (checked on all 121 `.bin` files, see
//! docs/mcc_localization_format.md):
//!
//!   0x00  magic `7A 6B CE 90`            0x40  magic again
//!   0x04  u32 file size - 16             0x44  table name, 32 bytes, NUL padded (`EN_Global`)
//!   0x08  u32 file size - 16 (same)      0x68  u32 = 8 + hash table size + string table size
//!   0x28  u32 checksum (algorithm unknown) 0x70  u32 = 8
//!   rest of both headers zero            0x78  u32 hash table size (4 x count)
//!                                        0x7C  u32 string table size
//!   0x80  count x u32 key hash (little-endian), then `count` NUL-terminated UTF-8 strings in
//!         the same order, then zero fill to a multiple of 8 bytes.
//!
//! Key hash = MSB-first CRC-32 (polynomial 0x04C11DB7, init 0xFFFFFFFF, NO final xor) over the
//! ASCII-UPPERCASED key bytes (case-insensitive) - checked against all 9 098 keys of
//! `DBG_BLNK_Global.bin` (whose values are `DBG-BLK::<key>`) and the 6 keys of `DA_Debug.bin`.
//! Every `$` key of the 388 shipped Halo 4 variants (34 fields) and the 338 shipped Reach
//! variants (48 fields) resolves through `EN_Global.bin`; `EN_Halo4.bin` / `EN_HaloReach.bin`
//! carry the in-game strings and are loaded as the next fallbacks, `DBG_BLNK_Global.bin` last.
//!
//! Language: the app has no language setting, so `DEFAULT_LANG` = `EN` (the prefix of the file
//! names: CS CT DA DE DU EN FI FR IT JP KO NO PB PO PR RU SP SU).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Result};

/// `7A 6B CE 90` at 0x00 and 0x40 of every table.
pub const LOC_MAGIC: [u8; 4] = [0x7A, 0x6B, 0xCE, 0x90];
/// Language prefix used until the app grows a language setting.
pub const DEFAULT_LANG: &str = "EN";

/// Localization key hash: MSB-first CRC-32, poly 0x04C11DB7, init 0xFFFFFFFF, no final xor,
/// over the ASCII-uppercased key (so `h4_mvar_settler_name` == `H4_MVAR_SETTLER_NAME`).
pub fn key_hash(key: &str) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for b in key.bytes() {
        let b = b.to_ascii_uppercase();
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 { (crc << 1) ^ 0x04C1_1DB7 } else { crc << 1 };
        }
    }
    crc
}

/// One parsed `.bin` table: key hash -> UTF-8 string.
#[derive(Debug, Default)]
pub struct LocTable {
    /// The 32-byte name field at 0x44 (`EN_Global`).
    pub name: String,
    pub strings: HashMap<u32, String>,
    /// Entry count (equals `strings.len()` unless the file repeats a hash).
    pub count: usize,
}

impl LocTable {
    pub fn parse(data: &[u8]) -> Result<LocTable> {
        let u32_at = |o: usize| -> Result<u32> {
            data.get(o..o + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| anyhow!("table truncated at {o:#x}"))
        };
        if data.len() < 0x80 || data[0..4] != LOC_MAGIC || data[0x40..0x44] != LOC_MAGIC { bail!("not an MCC localization table (magic)"); }
        if u32_at(4)? as usize != data.len() - 16 { bail!("size field {} != file size - 16", u32_at(4)?); }
        let name_raw = &data[0x44..0x64];
        let name = String::from_utf8_lossy(&name_raw[..name_raw.iter().position(|&b| b == 0).unwrap_or(name_raw.len())]).into_owned();
        let table_size = u32_at(0x78)? as usize;
        let string_size = u32_at(0x7C)? as usize;
        if u32_at(0x68)? as usize != table_size + string_size + 8 { bail!("payload length field mismatch"); }
        if table_size % 4 != 0 || 0x80 + table_size + string_size > data.len() { bail!("hash/string table sizes exceed the file"); }
        let count = table_size / 4;
        let hashes = &data[0x80..0x80 + table_size];
        let mut strings = HashMap::with_capacity(count);
        let mut pos = 0x80 + table_size;
        let end = pos + string_size;
        for i in 0..count {
            let h = u32::from_le_bytes([hashes[4 * i], hashes[4 * i + 1], hashes[4 * i + 2], hashes[4 * i + 3]]);
            let nul = data[pos..end].iter().position(|&b| b == 0).ok_or_else(|| anyhow!("string {i} of {count} unterminated"))?;
            strings.insert(h, String::from_utf8_lossy(&data[pos..pos + nul]).into_owned());
            pos += nul + 1;
        }
        Ok(LocTable { name, strings, count })
    }

    pub fn load(path: &Path) -> Result<LocTable> {
        let data = std::fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        LocTable::parse(&data).map_err(|e| anyhow!("{}: {e}", path.display()))
    }

    /// The string for a key (without the `$`), if this table has it.
    pub fn get(&self, key: &str) -> Option<&str> { self.strings.get(&key_hash(key)).map(String::as_str) }
}

/// The tables of one language, searched in order.
#[derive(Debug, Default)]
pub struct Localization {
    pub dir: Option<PathBuf>,
    pub lang: String,
    pub tables: Vec<LocTable>,
}

impl Localization {
    /// Table files searched for a key, in order: the language's Global, Halo4 and HaloReach
    /// tables, then the debug table whose values are `DBG-BLK::<key>` (so a key that no English
    /// table carries still shows its own name).
    pub fn table_files(lang: &str) -> [String; 4] {
        [format!("{lang}_Global.bin"), format!("{lang}_Halo4.bin"), format!("{lang}_HaloReach.bin"), "DBG_BLNK_Global.bin".to_string()]
    }

    /// Load whatever tables exist in `dir` (missing files are skipped, unreadable ones logged).
    pub fn load_dir(dir: &Path, lang: &str) -> Localization {
        let mut tables = Vec::new();
        for f in Self::table_files(lang) {
            let p = dir.join(&f);
            if !p.is_file() { continue; }
            match LocTable::load(&p) {
                Ok(t) => tables.push(t),
                Err(e) => log::warn!("h4-loc: {e:#}"),
            }
        }
        Localization { dir: Some(dir.to_path_buf()), lang: lang.to_string(), tables }
    }

    pub fn is_empty(&self) -> bool { self.tables.is_empty() }

    /// Resolve a key (with or without its leading `$`) through the tables in order.
    pub fn resolve_key(&self, key: &str) -> Option<String> {
        let key = key.strip_prefix('$').unwrap_or(key);
        if key.is_empty() { return None; }
        for t in &self.tables {
            if let Some(s) = t.get(key) {
                // the debug table's value is `DBG-BLK::<key>`: hand back the bare key name
                return Some(s.strip_prefix("DBG-BLK::").unwrap_or(s).to_string());
            }
        }
        None
    }

    /// Display form of a variant header string: `$key` -> its text (the raw `$key` when no table
    /// has it), anything else unchanged.
    pub fn display(&self, text: &str) -> String {
        if text.starts_with('$') { self.resolve_key(text).unwrap_or_else(|| text.to_string()) } else { text.to_string() }
    }
}

/// `<MCC>/data/ui/Localization` of the first detected MCC install that has the language's
/// Global table.
pub fn localization_dir(lang: &str) -> Option<PathBuf> {
    let mut roots = crate::mapcat::mcc_install_roots();
    if let Some(maps) = super::cache::maps_dir() {
        if let Some(root) = maps.parent().and_then(|p| p.parent()) { roots.push(root.to_path_buf()); }
    }
    roots.into_iter().map(|r| r.join("data").join("ui").join("Localization")).find(|d| d.join(format!("{lang}_Global.bin")).is_file())
}

/// The process-wide English tables, loaded on first use (empty when no MCC install is found,
/// in which case `display` returns keys unchanged).
pub fn global() -> &'static Localization {
    static LOC: OnceLock<Localization> = OnceLock::new();
    LOC.get_or_init(|| match localization_dir(DEFAULT_LANG) {
        Some(dir) => {
            let l = Localization::load_dir(&dir, DEFAULT_LANG);
            log::info!("h4-loc: {} tables from {} ({} strings)", l.tables.len(), dir.display(), l.tables.iter().map(|t| t.count).sum::<usize>());
            l
        }
        None => { log::info!("h4-loc: no MCC localization folder found; $keys stay raw"); Localization::default() }
    })
}

/// `$key` -> English text through the global tables; other strings pass through unchanged.
pub fn display(text: &str) -> String { global().display(text) }

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> Option<PathBuf> {
        let d = localization_dir(DEFAULT_LANG);
        if d.is_none() { eprintln!("skip: no MCC localization folder"); }
        d
    }

    /// The hash from single-character keys (`0`..`9` differ by exactly the CRC polynomial) and
    /// the key/value pairs of the debug table: every one of its 9 098 values is `DBG-BLK::<key>`
    /// and hashes to its own slot.
    #[test]
    fn debug_table_hashes_every_key() {
        assert_eq!(key_hash("0") ^ key_hash("1"), 0x04C1_1DB7);
        assert_eq!(key_hash("h4_mvar_settler_name"), 0x4E50_4031);
        assert_eq!(key_hash("h4_mvar_settler_description"), 0x0C49_351F);
        assert_eq!(key_hash("h4_mvar_settler_name"), key_hash("H4_MVAR_SETTLER_NAME"));
        let Some(d) = dir() else { return };
        let t = LocTable::load(&d.join("DBG_BLNK_Global.bin")).unwrap();
        assert_eq!((t.name.as_str(), t.count), ("DBG_BLNK_Global", 9098));
        assert_eq!(t.strings.len(), 9098, "no duplicate hashes");
        for (h, s) in &t.strings {
            let key = s.strip_prefix("DBG-BLK::").unwrap_or_else(|| panic!("{s}"));
            assert_eq!(key_hash(key), *h, "{key}");
        }
        assert_eq!(t.get("h4_mvar_settler_name"), Some("DBG-BLK::h4_mvar_settler_name"));
        // every language table in the folder parses with the same layout
        let mut n = 0;
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.extension().map_or(false, |x| x == "bin") {
                let t = LocTable::load(&p).unwrap_or_else(|e| panic!("{e:#}"));
                assert_eq!(format!("{}.bin", t.name), p.file_name().unwrap().to_string_lossy());
                n += 1;
            }
        }
        assert!(n >= 100, "{n} tables");
    }

    /// The English tables resolve the shipped variant keys to the in-game strings.
    #[test]
    fn english_resolves_variant_keys() {
        let Some(d) = dir() else { return };
        let l = Localization::load_dir(&d, DEFAULT_LANG);
        assert_eq!(l.tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["EN_Global", "EN_Halo4", "EN_HaloReach", "DBG_BLNK_Global"]);
        assert_eq!(l.display("$h4_mvar_settler_name"), "Settler");
        assert_eq!(l.display("$h4_mvar_settler_description"), "This variant of Ravine is a semi-symmetrical map resting atop a verdant cliff side.");
        assert_eq!(l.display("$h4_mvar_grifball_name"), "Grifball Court");
        assert_eq!(l.display("$h4_mvar_relay_name"), "Relay");
        assert_eq!(l.display("$h4_mvar_ascent_name"), "Ascent");
        assert_eq!(l.display("$H4_MPMap_4-3_Title"), "COMPLEX");
        assert_eq!(l.display("$hr_mvar_Hardcore_Battle_Canyon"), "MLG BATTLE CANYON");
        assert_eq!(l.display("$HReach_MPMap_R-21_Title"), "ASYLUM");
        assert!(l.display("$h4_mvar_ascent_description").starts_with("This variant of Erosion"));
        // literal strings and unknown keys pass through
        assert_eq!(l.display("Firebase Sandbox"), "Firebase Sandbox");
        assert_eq!(l.display("$no_such_key_hms_test"), "$no_such_key_hms_test");
        assert_eq!(l.display(""), "");
        // the Halo 4 table is English UTF-8 (a curly apostrophe survives)
        assert!(l.tables[1].strings.values().any(|s| s.contains('\u{2019}')));
        assert!(!global().is_empty());
    }

    /// Every `$` key in every shipped Halo 4 AND Reach variant resolves (34 + 48 fields).
    #[test]
    fn shipped_variant_keys_all_resolve() {
        let Some(d) = dir() else { return };
        let l = Localization::load_dir(&d, DEFAULT_LANG);
        let h4_files = super::super::mvar::all_variant_files();
        if h4_files.is_empty() { eprintln!("skip: no halo4 variants"); return; }
        let (mut keys, mut literal, mut empty, mut unresolved) = (0, 0, 0, Vec::new());
        for f in &h4_files {
            let v = super::super::mvar::parse_h4_variant(f).unwrap();
            for (raw, shown) in [(&v.title_key, &v.title), (&v.description_key, &v.description)] {
                if raw.starts_with('$') {
                    keys += 1;
                    if l.resolve_key(raw).is_none() { unresolved.push(format!("{}: {raw}", f.display())); }
                    assert_ne!(shown, raw, "{}: {raw} not resolved by the decoder", f.display());
                } else if raw.is_empty() { empty += 1; } else { literal += 1; assert_eq!(shown, raw); }
            }
        }
        eprintln!("h4-loc: {} halo4 variants, {keys} $-key fields, {literal} literal, {empty} empty, unresolved {unresolved:?}", h4_files.len());
        assert!(unresolved.is_empty(), "{unresolved:?}");
        assert!(keys >= 34 && literal >= 700, "{keys} keys / {literal} literal");
        // Reach: the hopper variants with $ keys resolve through the same tables
        let mut reach_keys = 0;
        for dir in crate::mapcat::variant_dirs() {
            for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let p = e.path();
                if !p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) { continue; }
                let Some(v) = crate::mvar::parse_variant(&p) else { continue };
                for s in [&v.title, &v.description] {
                    if s.starts_with('$') { reach_keys += 1; assert!(l.resolve_key(s).is_some(), "{}: {s}", p.display()); }
                }
            }
        }
        eprintln!("h4-loc: {reach_keys} reach $-key fields resolved");
    }
}
