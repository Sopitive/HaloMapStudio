//! #reach-font  Halo: Reach's OWN fonts, decoded from the game's font assets.
//!
//! Reach does not keep its fonts in tags. The engine builds the path
//! `<maps>\fonts\font_package[_<language>].bin` (`haloreach.dll` `BuildFontFilmPath` +
//! `font_package`), and the Reach Editing Kit additionally ships every font as a loose
//! `<maps>\fonts\<lang>\<name>.font` next to the `font_settings.txt` it was imported from and the
//! `font_table.txt` that orders them.
//!
//! WHICH FONT IS WHICH is not a guess: `maps\fonts\font_table.txt` lists eleven files in exactly
//! the order of the engine's `global_font_id` enum (the enum's own value names come out of
//! `reach_tag_test.exe`'s tag reflection), and `font_settings.txt` names the typeface each one was
//! built from:
//!
//! | id | `global_font_id`               | file                           | typeface |
//! |----|--------------------------------|--------------------------------|----------|
//! | 0  | `terminal_font`                | `fixedsys-9`                   | Fixedsys |
//! | 1  | `body_text_font`               | `tv_nord_condensed-18`         | TV Nord Condensed |
//! | 2  | `title_font`                   | `tv_nord_condensed_bold-24`    | TV Nord Condensed Bold |
//! | 3  | `super_large_font`             | `tv_nord_condensed_bold-42`    | TV Nord Condensed Bold |
//! | 4  | `large_body_text_font`         | `tv_nord_condensed_bold-30`    | TV Nord Condensed Bold |
//! | 5  | `split_screen_hud_message_font`| `eurostile_lt_demi-13`         | Eurostile LT Pro Demi |
//! | 6  | `full_screen_hud_message_font` | `tv_nord_hud-15`               | TV Nord Condensed |
//! | 7  | `english_body_text_font`       | `tv_nord_condensed-15`         | TV Nord Condensed |
//! | 8  | `hud_number_font`              | `eurostile_lt_demi-18`         | Eurostile LT Pro Demi |
//! | 9  | `subtitle_font`                | `tv_nord_condensed-16`         | TV Nord Condensed |
//! | 10 | `main_menu_font`               | `tv_nord_condensed-17`         | TV Nord Condensed |
//!
//! FORMAT (reverse-engineered from `reach_tag_test.exe`, `omaha\text\font_group.cpp`:
//! `font_header_validate` for the header, `font_group.cpp:238` for kerning, `:651..:753` for the
//! glyph decoder, `:464..:608` for its encoder):
//!
//! ```text
//! font header, 0x168 bytes (magic 0xF0000006 for a loose .font / a font inside a package)
//!   +0x000 u32   magic 0xF0000006
//!   +0x004 char[32] name ("en\tv_nord_hud-15")
//!   +0x024 i16   ascender  (pixels above the baseline)
//!   +0x026 i16   descender (pixels below)
//!   +0x02C i32   kerning pair offset (>= 0x168)
//!   +0x030 i32   kerning pair count (<= 0xFF)
//!   +0x034 u8[256] first kerning pair index, by FIRST character (monotone)
//!   +0x134 i32   character table offset
//!   +0x138 i32   character table length, in codepoints (<= 0x10000)
//!   +0x13C i32   number of characters that are actually present
//!   +0x140 i32   character data offset
//!   +0x144 i32   character data size
//! character table: `length` x { u32 offset, u32 size } indexed BY CODEPOINT; 0xFFFFFFFF = absent.
//!   Both are in units of 8 bytes from the character data offset.
//! character record: { i16 advance, u16 _, u32 packed size, u16 width, u16 height,
//!                     i16 origin x, i16 origin y } then `packed size` bytes of packed pixels.
//!   The baseline sits `origin y` rows down from the top of the bitmap.
//! kerning pair: { u8 second character, i8 delta }, sorted by the second character.
//! ```
//!
//! The packed pixels are a run-length code over 4-bit alpha. Each byte:
//!
//! ```text
//!   0x00          -> the next TWO bytes are one pixel, big-endian: alpha<<12 | value
//!   0x01..=0x3F   -> that many fully transparent pixels
//!   0x40..=0x7F   -> (b & 0x3F) fully opaque pixels
//!   0x80..=0xBF   -> one pixel of class (b-0x80)>>3, then a run given by (b & 7):
//!                    1,2,3 -> 4,3,2 more OPAQUE; 4,5,6,7 -> 5,4,3,2 more TRANSPARENT; 0 -> none
//!   0xC0..=0xFF   -> two pixels, classes (b-0xC0)>>3 and (b & 7)
//! ```
//!
//! A class 0..7 expands to 4-bit alpha as `(c << 1) | (c & 1)` (0,3,4,7,8,11,12,15) — the engine's
//! own expansion, `font_group.cpp:696`. The 12-bit "value" is the colour channel and is 0xFFF
//! (white) in every shipped Reach font, so a glyph is pure coverage and takes its colour from the
//! caller.
//!
//! WHERE THE FILES ARE: MCC ships only `font_package_icon*.bin` for Reach (six fonts named
//! `icon\fixedsys-*`, holding nothing but private-use button-icon codepoints), so the text glyphs
//! are NOT in a stock MCC install — MCC renders the engine's text with its own scalable font
//! instead. The text fonts ARE in the Halo Reach Editing Kit (`HREK\maps\fonts`), which is what
//! `find_font_source` looks for. With no source found the caller falls back to its own font and
//! says so; nothing here is embedded in HMS.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `global_font_id`, the engine's font selector (`reach_tag_test.exe` tag reflection:
/// `global_font_id_enum_definition`, 11 values). `chud` text widgets and `cui_screen` text
/// components both store an index into this.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[allow(dead_code)] // the whole enum is the engine's; HMS uses two of its eleven today
pub enum FontId {
    Terminal = 0,
    BodyText = 1,
    Title = 2,
    SuperLarge = 3,
    LargeBodyText = 4,
    SplitScreenHudMessage = 5,
    FullScreenHudMessage = 6,
    EnglishBodyText = 7,
    HudNumber = 8,
    Subtitle = 9,
    MainMenu = 10,
}

#[allow(dead_code)]
impl FontId {
    /// The enum value's own name, as the tag reflection spells it.
    pub fn enum_name(self) -> &'static str {
        FONT_ENUM_NAMES[self as usize]
    }
    /// The file `maps\fonts\font_table.txt` maps this id to, without the `.font`.
    pub fn file_stem(self) -> &'static str {
        FONT_TABLE_FILES[self as usize]
    }
    /// The typeface `font_settings.txt` built it from.
    pub fn typeface(self) -> &'static str {
        FONT_TYPEFACES[self as usize]
    }
    pub fn from_index(i: i32) -> Option<FontId> {
        Some(match i {
            0 => FontId::Terminal,
            1 => FontId::BodyText,
            2 => FontId::Title,
            3 => FontId::SuperLarge,
            4 => FontId::LargeBodyText,
            5 => FontId::SplitScreenHudMessage,
            6 => FontId::FullScreenHudMessage,
            7 => FontId::EnglishBodyText,
            8 => FontId::HudNumber,
            9 => FontId::Subtitle,
            10 => FontId::MainMenu,
            _ => return None,
        })
    }
}

/// `global_font_id_enum_definition`'s value names, in order.
pub const FONT_ENUM_NAMES: [&str; 11] = [
    "terminal_font",
    "body_text_font",
    "title_font",
    "super_large_font",
    "large_body_text_font",
    "split_screen_hud_message_font",
    "full_screen_hud_message_font",
    "english_body_text_font",
    "hud_number_font",
    "subtitle_font",
    "main_menu_font",
];

/// `maps\fonts\font_table.txt`, verbatim and in order (the `en\` prefix and `.font` stripped).
pub const FONT_TABLE_FILES: [&str; 11] = [
    "fixedsys-9",
    "tv_nord_condensed-18",
    "tv_nord_condensed_bold-24",
    "tv_nord_condensed_bold-42",
    "tv_nord_condensed_bold-30",
    "eurostile_lt_demi-13",
    "tv_nord_hud-15",
    "tv_nord_condensed-15",
    "eurostile_lt_demi-18",
    "tv_nord_condensed-16",
    "tv_nord_condensed-17",
];

/// The typeface each of those was imported from (`maps\fonts\font_settings.txt`).
#[allow(dead_code)] // provenance: what each font file actually IS
pub const FONT_TYPEFACES: [&str; 11] = [
    "Fixedsys",
    "TV Nord Condensed",
    "TV Nord Condensed Bold",
    "TV Nord Condensed Bold",
    "TV Nord Condensed Bold",
    "Eurostile LT Pro Demi",
    "TV Nord Condensed",
    "TV Nord Condensed",
    "Eurostile LT Pro Demi",
    "TV Nord Condensed",
    "TV Nord Condensed",
];

const FONT_MAGIC: u32 = 0xF000_0006;
const PACKAGE_MAGIC: u32 = 0xC000_0004;
/// The header is this long in both the loose files and inside a package.
const HEADER_LEN: usize = 0x168;
/// The highest codepoint this module decodes. Gamertags are ASCII; Latin-1 covers the rest of the
/// authored UI text without carrying 61 000 table entries into an atlas.
const MAX_CODEPOINT: u32 = 0xFF;

fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn rd_i32(b: &[u8], o: usize) -> Option<i32> {
    Some(rd_u32(b, o)? as i32)
}
fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn rd_i16(b: &[u8], o: usize) -> Option<i16> {
    Some(rd_u16(b, o)? as i16)
}

/// One decoded character: its 8-bit coverage bitmap and the metrics that place it.
#[derive(Clone, Debug, Default)]
pub struct Glyph {
    /// How far the pen moves after drawing this character, pixels.
    pub advance: i16,
    pub w: u16,
    pub h: u16,
    /// Bitmap origin relative to the pen, pixels right.
    pub dx: i16,
    /// Rows from the top of the bitmap down to the baseline.
    pub dy: i16,
    /// `w * h` coverage values, 0..=255, row major.
    pub alpha: Vec<u8>,
}

/// One of Reach's fonts, with every codepoint up to [`MAX_CODEPOINT`] decoded.
#[derive(Clone, Debug)]
pub struct ReachFont {
    /// The name the asset carries, e.g. `en\tv_nord_hud-15`.
    pub name: String,
    pub ascender: i16,
    pub descender: i16,
    /// Codepoints present in the file at all (before this module's Latin-1 cut).
    #[allow(dead_code)]
    pub codepoints_in_file: i32,
    /// Characters present in the file at all.
    #[allow(dead_code)]
    pub glyphs_in_file: i32,
    glyphs: HashMap<u32, Glyph>,
    /// `first kerning pair index` by first character.
    kern_first: Vec<u8>,
    /// `{ second character, delta }` pairs, in the file's order.
    kern_pairs: Vec<(u8, i8)>,
}

/// Expand an encoded pixel class 0..7 to 4-bit alpha, `font_group.cpp:696`.
fn expand_class(c: u8) -> u8 {
    ((c << 1) | (c & 1)) & 0xF
}

/// The engine's packed-pixel decoder (`font_group.cpp:651..:753`), returning 8-bit coverage.
///
/// Stops at `npix` pixels or when the stream runs out; a short stream leaves the rest transparent,
/// which is what the engine's own bounds assert allows.
fn unpack_pixels(blob: &[u8], npix: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(npix);
    let mut i = 0usize;
    while i < blob.len() && out.len() < npix {
        let b = blob[i];
        match b >> 6 {
            0 | 1 => {
                let n = (b & 0x3F) as usize;
                if n != 0 || b >= 0x40 {
                    // a run of transparent (b < 0x40) or opaque pixels
                    let a = if b >= 0x40 { 255 } else { 0 };
                    out.extend(std::iter::repeat(a).take(n.min(npix - out.len())));
                    i += 1;
                } else {
                    // 0x00: an escape carrying one pixel's own 16-bit value, big-endian
                    if i + 2 >= blob.len() {
                        break;
                    }
                    let v = ((blob[i + 1] as u16) << 8) | blob[i + 2] as u16;
                    out.push(((v >> 12) as u8) * 17);
                    i += 3;
                }
            }
            2 => {
                let c0 = (b >> 3) & 7;
                out.push(expand_class(c0) * 17);
                let (n, opaque) = match b & 7 {
                    0 => (0usize, false),
                    1 => (4, true),
                    2 => (3, true),
                    3 => (2, true),
                    4 => (5, false),
                    5 => (4, false),
                    6 => (3, false),
                    _ => (2, false),
                };
                let a = if opaque { 255 } else { 0 };
                let room = npix.saturating_sub(out.len());
                out.extend(std::iter::repeat(a).take(n.min(room)));
                i += 1;
            }
            _ => {
                out.push(expand_class((b >> 3) & 7) * 17);
                if out.len() < npix {
                    out.push(expand_class(b & 7) * 17);
                }
                i += 1;
            }
        }
    }
    out.resize(npix, 0);
    out
}

impl ReachFont {
    /// Decode a font whose 0x168-byte header starts at `base` in `data`. All of the header's
    /// offsets are relative to `base`, in both the loose files and inside a package.
    pub fn parse_at(data: &[u8], base: usize) -> Option<ReachFont> {
        let h = data.get(base..)?;
        if rd_u32(h, 0)? != FONT_MAGIC {
            return None;
        }
        let name_end = h.get(4..4 + 32)?.iter().position(|&c| c == 0).unwrap_or(32);
        let name = String::from_utf8_lossy(&h[4..4 + name_end]).to_string();
        let ascender = rd_i16(h, 0x24)?;
        let descender = rd_i16(h, 0x26)?;
        let kern_off = rd_i32(h, 0x2C)?;
        let kern_cnt = rd_i32(h, 0x30)?;
        if kern_off < HEADER_LEN as i32 || !(0..=0xFF).contains(&kern_cnt) {
            return None;
        }
        let kern_first = h.get(0x34..0x34 + 256)?.to_vec();
        let mut kern_pairs = Vec::with_capacity(kern_cnt as usize);
        for i in 0..kern_cnt as usize {
            let o = kern_off as usize + i * 2;
            kern_pairs.push((*h.get(o)?, *h.get(o + 1)? as i8));
        }
        let tbl_off = rd_i32(h, 0x134)?;
        let tbl_len = rd_i32(h, 0x138)?;
        let glyph_count = rd_i32(h, 0x13C)?;
        let data_off = rd_i32(h, 0x140)?;
        let data_size = rd_i32(h, 0x144)?;
        if tbl_off < 0 || tbl_len <= 0 || tbl_len > 0x10000 || data_off < 0 || data_size < 0 {
            return None;
        }
        let mut glyphs = HashMap::new();
        for cp in 0..(tbl_len as u32).min(MAX_CODEPOINT + 1) {
            let e = tbl_off as usize + cp as usize * 8;
            let off = rd_u32(h, e)?;
            let size = rd_u32(h, e + 4)?;
            if off == u32::MAX {
                continue;
            }
            let rec = data_off as usize + off as usize * 8;
            let avail = (size as usize).saturating_mul(8);
            if avail < 16 {
                continue;
            }
            let advance = rd_i16(h, rec)?;
            let packed = rd_u32(h, rec + 4)? as usize;
            let w = rd_u16(h, rec + 8)?;
            let hgt = rd_u16(h, rec + 0x0A)?;
            let dx = rd_i16(h, rec + 0x0C)?;
            let dy = rd_i16(h, rec + 0x0E)?;
            if w == 0 || hgt == 0 || packed > avail - 16 {
                continue;
            }
            let blob = h.get(rec + 16..rec + 16 + packed)?;
            let alpha = unpack_pixels(blob, w as usize * hgt as usize);
            glyphs.insert(cp, Glyph { advance, w, h: hgt, dx, dy, alpha });
        }
        Some(ReachFont {
            name,
            ascender,
            descender,
            codepoints_in_file: tbl_len,
            glyphs_in_file: glyph_count,
            glyphs,
            kern_first,
            kern_pairs,
        })
    }

    /// Decode a loose `<lang>\<name>.font`.
    pub fn parse(data: &[u8]) -> Option<ReachFont> {
        ReachFont::parse_at(data, 0)
    }

    /// Pull one `global_font_id` out of a `font_package[_lang].bin`.
    ///
    /// Package header (`font_group.cpp:866`, `font_package_header_validate`): magic 0xC0000004 at
    /// +0x00, font count at +0x04, then `count` x { u32 offset, u32 size, i16 first character,
    /// i16 character count } at +0x08, and the `global_font_id` -> font index table at +0x30C.
    pub fn from_package(data: &[u8], id: FontId) -> Option<ReachFont> {
        if rd_u32(data, 0)? != PACKAGE_MAGIC {
            return None;
        }
        let count = rd_i32(data, 4)?;
        if count <= 0 || count > 0x40 {
            return None;
        }
        let idx = rd_i32(data, 0x30C + id as usize * 4)?;
        if idx < 0 || idx >= count {
            return None;
        }
        let off = rd_u32(data, 8 + idx as usize * 12)?;
        ReachFont::parse_at(data, off as usize)
    }

    /// Height of a line: ascender plus descender, the engine's own two header fields.
    pub fn line_height(&self) -> f32 {
        (self.ascender as i32 + self.descender as i32) as f32
    }

    pub fn glyph(&self, cp: u32) -> Option<&Glyph> {
        self.glyphs.get(&cp)
    }

    /// The glyph the font itself uses for anything it has no character for (codepoint 0).
    pub fn notdef(&self) -> Option<&Glyph> {
        self.glyphs.get(&0)
    }

    /// The kerning delta between two characters, `font_group.cpp:238`. Zero when either character
    /// is outside the single-byte range the table covers, which is the engine's own behaviour.
    pub fn kerning(&self, a: u32, b: u32) -> i32 {
        if a == 0 || b == 0 || a > 0xFF || b > 0xFF {
            return 0;
        }
        let first = self.kern_first[a as usize] as usize;
        let last = if a >= 0xFF { self.kern_pairs.len() } else { self.kern_first[a as usize + 1] as usize };
        for &(c, d) in self.kern_pairs.get(first..last.min(self.kern_pairs.len())).unwrap_or(&[]) {
            if c as u32 == b {
                return d as i32;
            }
            if c as u32 > b {
                break;
            }
        }
        0
    }

    /// Advance width of a string in font pixels, kerning included.
    pub fn measure(&self, text: &str) -> f32 {
        let mut w = 0i32;
        let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
        for (i, &cp) in cps.iter().enumerate() {
            let Some(g) = self.glyph(cp).or_else(|| self.notdef()) else { continue };
            w += g.advance as i32;
            if let Some(&next) = cps.get(i + 1) {
                w += self.kerning(cp, next);
            }
        }
        w as f32
    }
}

// ---------------------------------------------------------------------------------------------
// where the game's font files are
// ---------------------------------------------------------------------------------------------

/// A place a Reach font can be read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FontSource {
    /// A `maps\fonts\<lang>` folder of loose `.font` files (the Editing Kit layout).
    Loose(PathBuf),
    /// A `font_package[_lang].bin`.
    Package(PathBuf),
}

impl FontSource {
    pub fn display(&self) -> String {
        match self {
            FontSource::Loose(d) => d.display().to_string(),
            FontSource::Package(f) => f.display().to_string(),
        }
    }
    /// Read one `global_font_id` from this source.
    pub fn load(&self, id: FontId) -> Option<ReachFont> {
        match self {
            FontSource::Loose(dir) => {
                let bytes = std::fs::read(dir.join(format!("{}.font", id.file_stem()))).ok()?;
                ReachFont::parse(&bytes)
            }
            FontSource::Package(file) => {
                let bytes = std::fs::read(file).ok()?;
                ReachFont::from_package(&bytes, id)
            }
        }
    }
}

/// Does this `maps\fonts` folder hold Reach's text fonts?
///
/// `en\` loose files first (an Editing Kit), then `font_package.bin`. A stock MCC install has only
/// `font_package_icon*.bin`, which carries no text glyphs, so it is deliberately not accepted.
fn font_source_in(fonts_dir: &Path) -> Option<FontSource> {
    for lang in ["en", "."] {
        let d = if lang == "." { fonts_dir.to_path_buf() } else { fonts_dir.join(lang) };
        if d.join(format!("{}.font", FontId::FullScreenHudMessage.file_stem())).is_file() {
            return Some(FontSource::Loose(d));
        }
    }
    let pkg = fonts_dir.join("font_package.bin");
    if pkg.is_file() {
        return Some(FontSource::Package(pkg));
    }
    None
}

/// Every place a REACH font folder can sit: `<root>\steamapps\common\HREK\maps\fonts` for a
/// root that is a Steam library, and the same under a Windows Steam inside a mounted partition.
///
/// Only ever HREK. The other Editing Kits ship their own game's fonts under identical names, and
/// Halo 4's `font_package.bin` would load without complaint and be the wrong typeface.
fn hrek_font_dirs(root: &Path) -> [PathBuf; 2] {
    [
        root.join("steamapps").join("common").join("HREK").join("maps").join("fonts"),
        root.join("Program Files (x86)")
            .join("Steam")
            .join("steamapps")
            .join("common")
            .join("HREK")
            .join("maps")
            .join("fonts"),
    ]
}

/// Find a folder of Reach's text fonts, or `None` when this machine has none.
///
/// Looked at, in order:
/// 1. `HMS_REACH_FONTS`, a `maps\fonts` folder, for an install in an unusual place;
/// 2. `<the open map's folder>\fonts` — where the engine itself looks;
/// 3. `HREK\maps\fonts` under every Steam library this machine registers, and then under every
///    mounted volume (two levels deep) and the home directory, because the Editing Kit is often
///    installed by the Windows Steam on another partition, which a Linux `libraryfolders.vdf`
///    never lists.
///
/// Every probe is a single `is_file`, so this costs nothing even when it finds nothing; the one
/// caller caches the answer for the life of the process.
pub fn find_font_source(open_map: Option<&Path>) -> Option<FontSource> {
    if let Ok(dir) = std::env::var("HMS_REACH_FONTS") {
        let p = PathBuf::from(dir);
        if let Some(s) = font_source_in(&p) {
            return Some(s);
        }
    }
    if let Some(maps) = open_map.and_then(|m| m.parent()) {
        if let Some(s) = font_source_in(&maps.join("fonts")) {
            return Some(s);
        }
    }
    let mut roots: Vec<PathBuf> = crate::mapcat::steam_library_folders();
    roots.push(PathBuf::from("C:\\"));
    for base in ["/mnt", "/run/media", "/media"] {
        let Ok(rd) = std::fs::read_dir(base) else { continue };
        for e in rd.flatten().take(64) {
            roots.push(e.path());
            let Ok(rd2) = std::fs::read_dir(e.path()) else { continue };
            for e2 in rd2.flatten().take(64) {
                roots.push(e2.path());
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home));
    }
    for root in roots {
        for dir in hrek_font_dirs(&root) {
            if let Some(s) = font_source_in(&dir) {
                return Some(s);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// painting with it
// ---------------------------------------------------------------------------------------------

use eframe::egui;

/// One Reach font uploaded as a single-page coverage atlas, ready to paint with.
pub struct FontAtlas {
    pub font: ReachFont,
    /// Where it was read from, for the UI to say so.
    pub source: String,
    tex: egui::TextureHandle,
    uv: HashMap<u32, egui::Rect>,
}

impl FontAtlas {
    /// Pack every decoded glyph into one row-per-glyph-less, left-to-right strip and upload it.
    pub fn build(ctx: &egui::Context, font: ReachFont, source: String) -> FontAtlas {
        // One strip: glyphs are at most ~50px tall, so a single row keeps the packer trivial and
        // the texture well inside any sane limit.
        let mut cps: Vec<u32> = font.glyphs.keys().copied().collect();
        cps.sort_unstable();
        let pad = 1usize;
        let h = cps.iter().map(|c| font.glyphs[c].h as usize).max().unwrap_or(1) + pad * 2;
        let w: usize = cps.iter().map(|c| font.glyphs[c].w as usize + pad).sum::<usize>() + pad;
        let (w, h) = (w.max(1), h.max(1));
        let mut px = vec![egui::Color32::TRANSPARENT; w * h];
        let mut uv = HashMap::new();
        let mut x = pad;
        for &cp in &cps {
            let g = &font.glyphs[&cp];
            for yy in 0..g.h as usize {
                for xx in 0..g.w as usize {
                    let a = g.alpha[yy * g.w as usize + xx];
                    px[(yy + pad) * w + x + xx] = egui::Color32::from_rgba_unmultiplied(255, 255, 255, a);
                }
            }
            uv.insert(
                cp,
                egui::Rect::from_min_max(
                    egui::pos2(x as f32 / w as f32, pad as f32 / h as f32),
                    egui::pos2((x + g.w as usize) as f32 / w as f32, (pad + g.h as usize) as f32 / h as f32),
                ),
            );
            x += g.w as usize + pad;
        }
        let img = egui::ColorImage { size: [w, h], pixels: px };
        let tex = ctx.load_texture(format!("reach-font-{}", font.name), img, egui::TextureOptions::LINEAR);
        FontAtlas { font, source, tex, uv }
    }

    /// Advance width of `text` at `scale`, in screen points.
    #[allow(dead_code)]
    pub fn measure(&self, text: &str, scale: f32) -> egui::Vec2 {
        egui::vec2(self.font.measure(text) * scale, self.font.line_height() * scale)
    }

    /// Paint `text` with its baseline's left end at `pen`, as one textured mesh.
    ///
    /// `color`'s alpha multiplies the glyph coverage, so a fade is just a translucent colour.
    /// Returns the pen position after the last character.
    pub fn paint(&self, painter: &egui::Painter, pen: egui::Pos2, text: &str, color: egui::Color32, scale: f32) -> egui::Pos2 {
        let mut mesh = egui::Mesh::with_texture(self.tex.id());
        let cps: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let mut x = pen.x;
        for (i, &cp) in cps.iter().enumerate() {
            let key = if self.uv.contains_key(&cp) { cp } else { 0 };
            if let (Some(g), Some(&uv)) = (self.font.glyph(key), self.uv.get(&key)) {
                let min = egui::pos2(x + g.dx as f32 * scale, pen.y - g.dy as f32 * scale);
                let rect = egui::Rect::from_min_size(min, egui::vec2(g.w as f32 * scale, g.h as f32 * scale));
                mesh.add_rect_with_uv(rect, uv, color);
                x += g.advance as f32 * scale;
            }
            if let Some(&next) = cps.get(i + 1) {
                x += self.font.kerning(cp, next) as f32 * scale;
            }
        }
        if !mesh.is_empty() {
            painter.add(egui::Shape::mesh(mesh));
        }
        egui::pos2(x, pen.y)
    }
}

/// The one HUD font HMS paints with, built once per process and shared.
///
/// `full_screen_hud_message_font` (id 6) because that is the font the engine hard-codes for the
/// floating player name: `haloreach.dll+0x2E3BAC` passes a literal `6` as the font argument of the
/// text layout call, where the generic chud text widget passes `chud_widget_text_block+0x7E`.
///
/// Returns `None` on a machine with no Reach font files; every caller must keep a fallback.
/// Takes no map argument on purpose: it is cached on the first call, and a stock MCC `maps\fonts`
/// never holds text glyphs anyway, so the search that matters is the env override and the Editing
/// Kits in the Steam libraries.
pub fn hud_atlas(ctx: &egui::Context) -> Option<std::rc::Rc<FontAtlas>> {
    atlas_for(ctx, FontId::FullScreenHudMessage)
}

/// The font Reach's own `cui_screen` widgets ask for: `prop_font_id` 7,
/// `english_body_text_font` = `tv_nord_condensed-15`. Every text component of the shipped
/// scoreboard uses it.
pub fn ui_atlas(ctx: &egui::Context) -> Option<std::rc::Rc<FontAtlas>> {
    atlas_for(ctx, FontId::EnglishBodyText)
}

/// Any `global_font_id`, built once per process and shared. `None` on a machine with no Reach
/// font files; every caller must keep a fallback.
pub fn atlas_for(ctx: &egui::Context, id: FontId) -> Option<std::rc::Rc<FontAtlas>> {
    use std::cell::RefCell;
    thread_local! {
        static ATLASES: RefCell<HashMap<FontId, Option<std::rc::Rc<FontAtlas>>>> =
            RefCell::new(HashMap::new());
        /// Looked up once: the disk scan is the expensive part, not the decode.
        static SOURCE: RefCell<Option<Option<FontSource>>> = const { RefCell::new(None) };
    }
    let src = SOURCE.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.is_none() {
            *slot = Some(find_font_source(None));
        }
        slot.as_ref().unwrap().clone()
    })?;
    ATLASES.with(|m| {
        if let Some(a) = m.borrow().get(&id) {
            return a.clone();
        }
        let made = src.load(id).map(|f| std::rc::Rc::new(FontAtlas::build(ctx, f, src.display())));
        m.borrow_mut().insert(id, made.clone());
        made
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the smallest legal font: one 2x2 character at codepoint 'A', no kerning.
    fn synth() -> Vec<u8> {
        let mut f = vec![0u8; HEADER_LEN];
        f[0..4].copy_from_slice(&FONT_MAGIC.to_le_bytes());
        f[4..4 + 7].copy_from_slice(b"en\\test");
        f[0x24..0x26].copy_from_slice(&3i16.to_le_bytes()); // ascender
        f[0x26..0x28].copy_from_slice(&1i16.to_le_bytes()); // descender
        f[0x2C..0x30].copy_from_slice(&(HEADER_LEN as i32).to_le_bytes()); // kerning at the header end
        f[0x30..0x34].copy_from_slice(&0i32.to_le_bytes()); // no kerning pairs
        let tbl = HEADER_LEN as i32;
        let n_cp = ('A' as i32) + 1;
        f[0x134..0x138].copy_from_slice(&tbl.to_le_bytes());
        f[0x138..0x13C].copy_from_slice(&n_cp.to_le_bytes());
        f[0x13C..0x140].copy_from_slice(&1i32.to_le_bytes());
        let data = tbl + n_cp * 8;
        f[0x140..0x144].copy_from_slice(&data.to_le_bytes());
        f[0x144..0x148].copy_from_slice(&24i32.to_le_bytes());
        // character table: every codepoint absent but 'A'
        for cp in 0..n_cp {
            let mut entry = [0xFFu8; 8];
            if cp == 'A' as i32 {
                entry[0..4].copy_from_slice(&0u32.to_le_bytes());
                entry[4..8].copy_from_slice(&3u32.to_le_bytes()); // 3 * 8 = 24 bytes
            }
            f.extend_from_slice(&entry);
        }
        // character record: advance 3, 2x2, origin (0, 3), then packed pixels
        let mut rec = Vec::new();
        rec.extend_from_slice(&3i16.to_le_bytes());
        rec.extend_from_slice(&0u16.to_le_bytes());
        rec.extend_from_slice(&3u32.to_le_bytes()); // packed size
        rec.extend_from_slice(&2u16.to_le_bytes());
        rec.extend_from_slice(&2u16.to_le_bytes());
        rec.extend_from_slice(&0i16.to_le_bytes());
        rec.extend_from_slice(&3i16.to_le_bytes());
        // 0x42 = two opaque, 0x01 = one transparent, 0xC7|.. two classes 0 and 7
        rec.extend_from_slice(&[0x42, 0x01, 0xC7]);
        rec.resize(24, 0);
        f.extend_from_slice(&rec);
        f
    }

    #[test]
    fn synthetic_font_round_trips() {
        let f = ReachFont::parse(&synth()).expect("parse");
        assert_eq!(f.name, "en\\test");
        assert_eq!((f.ascender, f.descender), (3, 1));
        assert_eq!(f.line_height(), 4.0);
        let g = f.glyph('A' as u32).expect("A");
        assert_eq!((g.advance, g.w, g.h, g.dx, g.dy), (3, 2, 2, 0, 3));
        // 0x42 -> 255,255 ; 0x01 -> 0 ; 0xC7 -> class 0 then class 7 (the 4th pixel is the last)
        assert_eq!(g.alpha, vec![255, 255, 0, 0]);
        assert_eq!(f.measure("AA"), 6.0);
        assert_eq!(f.kerning('A' as u32, 'A' as u32), 0);
    }

    #[test]
    fn rejects_foreign_bytes() {
        assert!(ReachFont::parse(&[0u8; 0x200]).is_none());
        assert!(ReachFont::parse(&[]).is_none());
    }

    /// The decoder's four byte classes, straight out of `font_group.cpp:651..:753`.
    #[test]
    fn packed_pixel_classes() {
        // a transparent run, an opaque run
        assert_eq!(unpack_pixels(&[0x03], 3), vec![0, 0, 0]);
        assert_eq!(unpack_pixels(&[0x43], 3), vec![255, 255, 255]);
        // 0x80..0xBF: one pixel of the class, then a run. 0x81 -> class 0 then 4 opaque.
        assert_eq!(unpack_pixels(&[0x81], 5), vec![0, 255, 255, 255, 255]);
        // 0xBC -> class 7 then 5 transparent
        assert_eq!(unpack_pixels(&[0xBC], 6), vec![255, 0, 0, 0, 0, 0]);
        // 0xC0..0xFF: two classes. 0xC0|3<<3|5 -> classes 3 and 5 -> alpha 7 and 11 (of 15)
        assert_eq!(unpack_pixels(&[0xC0 | (3 << 3) | 5], 2), vec![7 * 17, 11 * 17]);
        // the escape: one pixel carrying its own value, big-endian
        assert_eq!(unpack_pixels(&[0x00, 0x8F, 0xFF], 1), vec![8 * 17]);
        // every class expands the way the engine expands it
        let a: Vec<u8> = (0..8).map(expand_class).collect();
        assert_eq!(a, vec![0, 3, 4, 7, 8, 11, 12, 15]);
    }

    #[test]
    fn font_table_matches_the_enum() {
        assert_eq!(FONT_TABLE_FILES.len(), FONT_ENUM_NAMES.len());
        assert_eq!(FontId::FullScreenHudMessage.file_stem(), "tv_nord_hud-15");
        assert_eq!(FontId::FullScreenHudMessage.enum_name(), "full_screen_hud_message_font");
        assert_eq!(FontId::from_index(6), Some(FontId::FullScreenHudMessage));
        assert_eq!(FontId::from_index(11), None);
        for i in 0..11 {
            assert_eq!(FontId::from_index(i).unwrap() as i32, i);
        }
    }

    /// Reach's real `tv_nord_hud-15`, when this machine has the Editing Kit. Skipped otherwise, so
    /// the suite does not depend on a game install.
    #[test]
    fn real_reach_hud_font_decodes() {
        let Some(src) = find_font_source(None) else {
            eprintln!("no Reach font source on this machine - skipped");
            return;
        };
        let f = src.load(FontId::FullScreenHudMessage).expect("tv_nord_hud-15");
        assert_eq!((f.ascender, f.descender), (17, 4));
        let a = f.glyph('A' as u32).expect("A");
        assert_eq!((a.advance, a.w, a.h, a.dx, a.dy), (9, 10, 22, 0, 17));
        // the top three rows of 'A' are empty and row 3 is where the apex starts
        assert!(a.alpha[..3 * 10].iter().all(|&v| v == 0));
        assert!(a.alpha[3 * 10..4 * 10].iter().any(|&v| v > 200));
        // kerning is present and negative between the pairs it was authored for
        assert!(f.kerning('T' as u32, '.' as u32) < 0 || f.kerning('Y' as u32, '.' as u32) < 0);
        assert!(f.measure("Sopitive") > 30.0);
    }
}
