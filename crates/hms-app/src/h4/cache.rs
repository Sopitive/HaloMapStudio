//! Halo 4 (MCC) cache reader - pure Rust, no native DLL.
//!
//! Layout facts come from docs/halo4_support_plan.md (section 2) plus the resource-gestalt /
//! paging layout in docs/halo4_geometry_layout.md. Everything here is cross-checked against
//! the shipped caches by the tests at the bottom of this file (they skip when the MCC halo4
//! maps folder is absent).
//!
//! Header (Halo 4 = Reach U13 minus 8 bytes after 0x58):
//!   magic "daeh" @0, version 13 @4, type i16 @0x18, file table 32/36/40/44,
//!   string table 48/52/56/60, namespaces 64/68, build string @152, map name @0xB8,
//!   scenario path @0xD8, virtual base u64 @728, tag index ptr u64 @736,
//!   section offset table @1220, data table (resource data start) @1224, section table @1236.
//! Pointer expansion: file_off = (raw << 2) + expander - tagMagic (Halo 4 0x4FFF0000,
//! Halo 2 Anniversary 0x7AC00000 - see `Engine`; every other offset here is shared by both).
//! Tag block = {i32 count, u32 raw ptr, u32 pad}; tag ref = {fourcc reversed, u32, u32, u32 datum}
//! with the two middle words 0xCDCDCDCD (never test them for zero). String ids: 16 index bits,
//! 8 namespace bits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};

/// The single Halo 4 MCC build on disk today (all 54 caches). Two spaces after "Apr".
pub const H4_BUILD: &str = "Apr  1 2023 17:35:22";
/// Halo 4 pointer-expansion constant (Reach MCC uses 0x50000000).
pub const H4_EXPANDER: i64 = 0x4FFF_0000;
/// The single Halo 2 Anniversary ("groundhog") MCC build on disk today (all 12 playable caches
/// + shared.map + campaign.map). `docs/h2a_support_plan.md`.
pub const H2A_BUILD: &str = "Jun 13 2023 20:21:18";
/// Halo 2 Anniversary pointer-expansion constant. #h2a: derived by intersecting, over all 11
/// shipped playable caches, the window of constants that lands every tag meta inside section 2
/// (the tag-data section) - [0x7AB9D7C0, 0x7AC05E8C], width 0x686CC - and then verified
/// functionally: only the exact value makes `play +0x18` read 3470 pages whose cacheIndex is
/// all in {-1, 1} and makes 100 % of them inflate to their recorded decompressed size.
pub const H2A_EXPANDER: i64 = 0x7AC0_0000;

/// Which MCC engine a cache belongs to. **Halo 2 Anniversary's cache format is Halo 4's**: the
/// header layout, tag index, string ids, `play`/`zone` paging and every sbsp / Lbsp / mat block
/// offset verified so far are identical - the ONE difference is the pointer-expansion constant.
/// So `H4Cache` reads both and this enum supplies the per-engine constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Halo4,
    /// Halo 2 Anniversary, MCC's `groundhog` folder.
    H2A,
}

impl Engine {
    /// The build string at header +152 that identifies this engine.
    pub fn build(self) -> &'static str {
        match self { Engine::Halo4 => H4_BUILD, Engine::H2A => H2A_BUILD }
    }
    /// Pointer expansion: `file_off = (raw << 2) + expander - tag_magic`.
    pub fn expander(self) -> i64 {
        match self { Engine::Halo4 => H4_EXPANDER, Engine::H2A => H2A_EXPANDER }
    }
    /// The MCC install sub-folder holding this engine's `maps/`.
    pub fn folder(self) -> &'static str {
        match self { Engine::Halo4 => "halo4", Engine::H2A => "groundhog" }
    }
    pub fn label(self) -> &'static str {
        match self { Engine::Halo4 => "Halo 4", Engine::H2A => "Halo 2 Anniversary" }
    }
    /// The engine whose build string this is, if any.
    pub fn of_build(build: &str) -> Option<Engine> {
        [Engine::Halo4, Engine::H2A].into_iter().find(|e| e.build() == build)
    }
    /// Env var overriding the maps folder for tests (`HMS_H4_MAPS` / `HMS_H2A_MAPS`).
    fn maps_env(self) -> &'static str {
        match self { Engine::Halo4 => "HMS_H4_MAPS", Engine::H2A => "HMS_H2A_MAPS" }
    }
}
const HDR_BUILD: usize = 152;
const HDR_MAP_NAME: usize = 0xB8;
const HDR_SCENARIO: usize = 0xD8;
const HDR_VBASE: usize = 728;
const HDR_TAG_INDEX: usize = 736;
const HDR_SECTION_OFFSETS: usize = 1220;
const HDR_DATA_TABLE: usize = 1224;
const HDR_SECTIONS: usize = 1236;

/// Little-endian readers over a byte slice. Out-of-range reads return 0 / empty so a corrupt
/// pointer degrades to "nothing there" instead of a panic; callers validate counts/ranges.
pub trait ByteRead {
    fn u8_at(&self, o: usize) -> u8;
    fn i16_at(&self, o: usize) -> i16;
    fn u16_at(&self, o: usize) -> u16;
    fn i32_at(&self, o: usize) -> i32;
    fn u32_at(&self, o: usize) -> u32;
    fn u64_at(&self, o: usize) -> u64;
    fn f32_at(&self, o: usize) -> f32;
    fn cstr_at(&self, o: usize, max: usize) -> String;
}

impl ByteRead for [u8] {
    fn u8_at(&self, o: usize) -> u8 { self.get(o).copied().unwrap_or(0) }
    fn i16_at(&self, o: usize) -> i16 { self.u16_at(o) as i16 }
    fn u16_at(&self, o: usize) -> u16 {
        match self.get(o..o + 2) { Some(b) => u16::from_le_bytes([b[0], b[1]]), None => 0 }
    }
    fn i32_at(&self, o: usize) -> i32 { self.u32_at(o) as i32 }
    fn u32_at(&self, o: usize) -> u32 {
        match self.get(o..o + 4) { Some(b) => u32::from_le_bytes([b[0], b[1], b[2], b[3]]), None => 0 }
    }
    fn u64_at(&self, o: usize) -> u64 {
        match self.get(o..o + 8) {
            Some(b) => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            None => 0,
        }
    }
    fn f32_at(&self, o: usize) -> f32 { f32::from_bits(self.u32_at(o)) }
    fn cstr_at(&self, o: usize, max: usize) -> String {
        let Some(b) = self.get(o..) else { return String::new() };
        let b = &b[..b.len().min(max)];
        let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        String::from_utf8_lossy(&b[..n]).into_owned()
    }
}

/// The MCC engine this file's cache header names (magic + version 13 + the build string at 152),
/// or None when it is not a cache this reader handles (Reach caches keep their build at 160).
pub fn cache_engine(path: &Path) -> Option<Engine> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 256];
    let n = f.read(&mut buf).ok()?;
    if n < 256 || &buf[0..4] != b"daeh" || buf[..].u32_at(4) != 13 { return None; }
    Engine::of_build(&buf[..].cstr_at(HDR_BUILD, 32))
}

/// Is this file a Halo 4 MCC cache? Header magic + version + the known build string at 152.
pub fn is_halo4_cache(path: &Path) -> bool { cache_engine(path) == Some(Engine::Halo4) }

/// Is this file a Halo 2 Anniversary (groundhog) MCC cache? #h2a
pub fn is_h2a_cache(path: &Path) -> bool { cache_engine(path) == Some(Engine::H2A) }

#[derive(Clone, Debug)]
pub struct TagEntry {
    pub class_idx: i16,
    pub raw_meta: u32,
    pub name: String,
}

#[derive(Clone, Copy, Debug)]
pub struct Page {
    /// -1 = this map, 1 = shared.map, 0 = mainmenu.map.
    pub cache_index: i16,
    pub data_offset: u32,
    pub compressed: u32,
    pub decompressed: u32,
}

/// `play` segment (24 B): three streams, each {offset i32 @0, page i16 @12, size index i16
/// @18}. Stream k of a resource is addressed by fixup nibble 4 / 8 / 6 -> page[0] / page[1] /
/// page[2] (on bitmaps every stream ends exactly at its page's end).
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub off: [i32; 3],
    pub page: [i16; 3],
}

/// `zone` resource entry (68 B: owner fourcc @0, datum @12, u16 salt @0x10, u8 kind @0x12, u8
/// flags @0x13, i32 definition length @0x14, i16 segment @0x1A, definition address @0x1C, the
/// fixup block @0x20, the definition-fixup block @0x2C, the offset block @0x38). `def_off` /
/// `def_len` locate this resource's definition data in the gestalt's fixup blob; `fixups` are
/// {offset in definition, address}, address top nibble: 2 = definition data, 4 = primary page,
/// 6 = secondary page, 8 = tertiary page.
#[derive(Clone, Debug)]
pub struct ResourceEntry {
    pub index: usize,
    pub owner: [u8; 4],
    pub datum: u32,
    pub salt: u16,
    pub kind: u8,
    pub def_len: u32,
    pub def_off: u32,
    pub segment: i16,
    pub def_addr: u32,
    pub fixups: Vec<(u32, u32)>,
}

impl ResourceEntry {
    pub fn is_live(&self) -> bool { self.datum != 0xFFFF_FFFF }
    /// Address a fixup wrote at `def_offset` (the pointer slot inside the definition data).
    pub fn fixup_at(&self, def_offset: u32) -> Option<u32> {
        self.fixups.iter().find(|(o, _)| *o == def_offset).map(|(_, a)| *a)
    }
}

struct SharedCache {
    map: memmap2::Mmap,
    data_table: u32,
}

pub struct H4Cache {
    pub path: PathBuf,
    map: memmap2::Mmap,
    /// Halo 4 or Halo 2 Anniversary - the only field that changes how the bytes are read (it
    /// supplies the pointer expander); every offset below is shared. #h2a
    pub engine: Engine,
    pub map_type: i16,
    pub map_name: String,
    pub scenario: String,
    pub header_magic: u32,
    pub tag_magic: i64,
    pub data_table: u32,
    classes: Vec<[u8; 4]>,
    tags: Vec<TagEntry>,
    str_idx: Vec<i32>,
    str_blob: Vec<u8>,
    ns: Vec<i32>,
    pub pages: Vec<Page>,
    pub segments: Vec<Segment>,
    pub resources: Vec<ResourceEntry>,
    /// Zone resource TYPE names (zone +0x04, 32 B, sid @+20), indexed by `ResourceEntry::kind`.
    /// The order differs per map (campaign / firefight caches insert `pca_coefficients_resource_definition`
    /// etc. before the BSP types), so kinds are always compared by NAME (`kind_is`).
    pub res_types: Vec<String>,
    pub def_blob_off: usize,
    pub def_blob_len: usize,
    page_cache: Mutex<HashMap<usize, Arc<Vec<u8>>>>,
    shared: Mutex<HashMap<i16, Arc<SharedCache>>>,
}

impl H4Cache {
    pub fn open(path: &Path) -> Result<H4Cache> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only mapping of a file we never write; the game folder is treated as immutable.
        let map = unsafe { memmap2::Mmap::map(&file) }.context("mmap")?;
        let d: &[u8] = &map;
        if d.len() < 0x1000 || &d[0..4] != b"daeh" || d.u32_at(4) != 13 {
            bail!("not a MCC cache (magic/version)");
        }
        let build = d.cstr_at(HDR_BUILD, 32);
        let engine = Engine::of_build(&build)
            .ok_or_else(|| anyhow!("not a known Halo 4 / Halo 2 Anniversary build: '{build}'"))?;
        let map_type = d.i16_at(0x18);
        let s0a = d.u32_at(HDR_SECTIONS) as u64;
        let s2a = d.u32_at(HDR_SECTIONS + 16) as u64;
        let s0o = d.u32_at(HDR_SECTION_OFFSETS) as u64;
        let s2o = d.u32_at(HDR_SECTION_OFFSETS + 8) as u64;
        let vbase = d.u64_at(HDR_VBASE) as i64;
        let header_magic = (s0a.wrapping_sub((s0a + s0o) & 0xffff_ffff) & 0xffff_ffff) as u32;
        let tag_magic = vbase - (((s2a + s2o) & 0xffff_ffff) as i64);
        let data_table = d.u32_at(HDR_DATA_TABLE);
        let map_name = d.cstr_at(HDR_MAP_NAME, 32);
        let scenario = d.cstr_at(HDR_SCENARIO, 256);
        let mut c = H4Cache {
            path: path.to_path_buf(),
            map,
            engine,
            map_type,
            map_name,
            scenario,
            header_magic,
            tag_magic,
            data_table,
            classes: Vec::new(),
            tags: Vec::new(),
            str_idx: Vec::new(),
            str_blob: Vec::new(),
            ns: Vec::new(),
            pages: Vec::new(),
            segments: Vec::new(),
            resources: Vec::new(),
            res_types: Vec::new(),
            def_blob_off: 0,
            def_blob_len: 0,
            page_cache: Mutex::new(HashMap::new()),
            shared: Mutex::new(HashMap::new()),
        };
        if map_type == 3 || map_type == 4 {
            return Ok(c); // shared.map / campaign.map: resource data only
        }
        c.parse_tag_index()?;
        c.parse_strings()?;
        c.parse_layout()?;
        Ok(c)
    }

    pub fn data(&self) -> &[u8] { &self.map }

    /// Expand a raw tag-data pointer to a file offset (None when it lands outside the file).
    pub fn meta_off(&self, raw: u32) -> Option<usize> {
        let v = ((raw as i64) << 2) + self.engine.expander() - self.tag_magic;
        if v >= 0 && (v as usize) < self.map.len() { Some(v as usize) } else { None }
    }

    fn header_off(&self, ptr: u32) -> usize { ptr.wrapping_sub(self.header_magic) as usize }

    fn parse_tag_index(&mut self) -> Result<()> {
        let d: &[u8] = &self.map;
        let idx = d.u64_at(HDR_TAG_INDEX) as i64 - self.tag_magic;
        if idx < 0 || idx as usize + 32 > d.len() { bail!("tag index pointer out of range"); }
        let idx = idx as usize;
        let n_classes = d.i32_at(idx);
        let class_ptr = d.u64_at(idx + 8) as i64 - self.tag_magic;
        let n_tags = d.i32_at(idx + 16);
        let tag_ptr = d.u64_at(idx + 24) as i64 - self.tag_magic;
        if !(0..100_000).contains(&n_classes) || !(0..10_000_000).contains(&n_tags) || class_ptr < 0 || tag_ptr < 0 {
            bail!("tag index counts/pointers implausible");
        }
        let (co, to) = (class_ptr as usize, tag_ptr as usize);
        if co + n_classes as usize * 16 > d.len() || to + n_tags as usize * 8 > d.len() {
            bail!("tag index tables out of range");
        }
        self.classes = (0..n_classes as usize)
            .map(|i| { let b = &d[co + i * 16..co + i * 16 + 4]; [b[3], b[2], b[1], b[0]] })
            .collect();
        let mut tags: Vec<TagEntry> = (0..n_tags as usize)
            .map(|i| TagEntry { class_idx: d.i16_at(to + i * 8), raw_meta: d.u32_at(to + i * 8 + 4), name: String::new() })
            .collect();
        // Tag names: file table (count/ptr/size/index ptr at 32/36/40/44), header-magic relative.
        let fc = d.i32_at(32).max(0) as usize;
        let fo = self.header_off(d.u32_at(36));
        let fs = d.i32_at(40).max(0) as usize;
        let fio = self.header_off(d.u32_at(44));
        for (i, t) in tags.iter_mut().enumerate().take(fc.min(n_tags as usize)) {
            let off = d.i32_at(fio + i * 4);
            if off >= 0 && (off as usize) < fs {
                t.name = d.cstr_at(fo + off as usize, 512);
            }
        }
        self.tags = tags;
        Ok(())
    }

    fn parse_strings(&mut self) -> Result<()> {
        let d: &[u8] = &self.map;
        let n = d.i32_at(48).max(0) as usize;
        let bo = self.header_off(d.u32_at(52));
        let size = d.i32_at(56).max(0) as usize;
        let io = self.header_off(d.u32_at(60));
        if io + n * 4 > d.len() || bo + size > d.len() { bail!("string table out of range"); }
        self.str_idx = (0..n).map(|i| d.i32_at(io + i * 4)).collect();
        self.str_blob = d[bo..bo + size].to_vec();
        let nsn = d.i32_at(64).max(0) as usize;
        let nso = self.header_off(d.u32_at(68));
        if nso + nsn * 4 > d.len() { bail!("namespace table out of range"); }
        self.ns = (0..nsn).map(|i| d.i32_at(nso + i * 4)).collect();
        Ok(())
    }

    /// `play` pages/segments + `zone` resource entries and the definition-data blob.
    fn parse_layout(&mut self) -> Result<()> {
        let play = self.find_tags(b"play").first().copied().ok_or_else(|| anyhow!("no play tag"))?;
        let pm = self.tag_meta(play).ok_or_else(|| anyhow!("play meta unreadable"))?;
        let (np, po) = self.block(pm + 0x18).ok_or_else(|| anyhow!("play pages block"))?;
        let d: &[u8] = &self.map;
        self.pages = (0..np)
            .map(|i| { let e = po + i * 88; Page { cache_index: d.i16_at(e + 4), data_offset: d.u32_at(e + 8), compressed: d.u32_at(e + 12), decompressed: d.u32_at(e + 16) } })
            .collect();
        let (ns, so) = self.block(pm + 0x30).ok_or_else(|| anyhow!("play segments block"))?;
        self.segments = (0..ns)
            .map(|i| { let e = so + i * 24; Segment {
                off: [d.i32_at(e), d.i32_at(e + 4), d.i32_at(e + 8)],
                page: [d.i16_at(e + 12), d.i16_at(e + 14), d.i16_at(e + 16)] } })
            .collect();
        let zone = self.find_tags(b"zone").first().copied().ok_or_else(|| anyhow!("no zone tag"))?;
        let zm = self.tag_meta(zone).ok_or_else(|| anyhow!("zone meta unreadable"))?;
        self.res_types = match self.block(zm + 0x04) {
            Some((n, o)) if n < 64 => (0..n).map(|i| self.sid(d.u32_at(o + i * 32 + 20))).collect(),
            _ => Vec::new(),
        };
        self.def_blob_len = d.i32_at(zm + 0x154).max(0) as usize;
        self.def_blob_off = self.meta_off(d.u32_at(zm + 0x160)).ok_or_else(|| anyhow!("zone fixup blob pointer"))?;
        if self.def_blob_off + self.def_blob_len > d.len() { bail!("zone fixup blob out of range"); }
        let (ne, eo) = self.block(zm + 0x58).ok_or_else(|| anyhow!("zone resource block"))?;
        let mut res = Vec::with_capacity(ne);
        let mut prefix = 0u32;
        for i in 0..ne {
            let e = eo + i * 68;
            let datum = d.u32_at(e + 12);
            let live = datum != 0xFFFF_FFFF;
            let fixups: Vec<(u32, u32)> = match self.block(e + 0x20) {
                Some((n, o)) if n < 100_000 => (0..n).map(|j| (d.u32_at(o + j * 8), d.u32_at(o + j * 8 + 4))).collect(),
                _ => Vec::new(),
            };
            let b38: Vec<i32> = match self.block(e + 0x38) {
                Some((n, o)) if n <= 3 => (0..n).map(|j| d.i32_at(o + j * 4)).collect(),
                _ => Vec::new(),
            };
            let def_len = d.i32_at(e + 0x14).max(0) as u32;
            // Definition data is packed in entry order; b38[0] carries the explicit offset and the
            // test below asserts both agree on every entry.
            let def_off = b38.first().copied().filter(|v| *v >= 0).map(|v| v as u32).unwrap_or(prefix);
            res.push(ResourceEntry {
                index: i,
                owner: { let b = &d[e..e + 4]; [b[3], b[2], b[1], b[0]] },
                datum,
                salt: d.u16_at(e + 0x10),
                kind: d.u8_at(e + 0x12),
                def_len,
                def_off,
                segment: d.i16_at(e + 0x1A),
                def_addr: d.u32_at(e + 0x1C),
                fixups,
            });
            if live { prefix = prefix.wrapping_add(def_len); }
        }
        self.resources = res;
        Ok(())
    }

    // ---- tags -------------------------------------------------------------------------------

    pub fn tag_count(&self) -> usize { self.tags.len() }
    pub fn class_count(&self) -> usize { self.classes.len() }
    pub fn tag_name(&self, idx: usize) -> &str { self.tags.get(idx).map(|t| t.name.as_str()).unwrap_or("") }
    pub fn tag_class(&self, idx: usize) -> Option<[u8; 4]> {
        let t = self.tags.get(idx)?;
        if t.class_idx < 0 { return None; }
        self.classes.get(t.class_idx as usize).copied()
    }
    pub fn find_tags(&self, class: &[u8; 4]) -> Vec<usize> {
        (0..self.tags.len()).filter(|&i| self.tag_class(i).as_ref() == Some(class)).collect()
    }
    /// File offset of a tag's main struct.
    pub fn tag_meta(&self, idx: usize) -> Option<usize> {
        let t = self.tags.get(idx)?;
        if t.raw_meta == 0 { return None; }
        self.meta_off(t.raw_meta)
    }
    /// Tag block at file offset `off` -> (count, file offset of element 0).
    pub fn block(&self, off: usize) -> Option<(usize, usize)> {
        let d: &[u8] = &self.map;
        let count = d.i32_at(off);
        let ptr = d.u32_at(off + 4);
        if count <= 0 || count > 0x100_0000 || ptr == 0 || ptr == 0xCDCD_CDCD { return None; }
        let o = self.meta_off(ptr)?;
        Some((count as usize, o))
    }
    /// Tag reference at file offset `off` -> (class fourcc, tag index). Tolerates the 0xCD fill in
    /// the middle words; null when the datum is -1.
    pub fn tag_ref(&self, off: usize) -> Option<([u8; 4], usize)> {
        let d: &[u8] = &self.map;
        let datum = d.u32_at(off + 12);
        if datum == 0xFFFF_FFFF || datum == 0xCDCD_CDCD { return None; }
        let idx = (datum & 0xFFFF) as usize;
        if idx >= self.tags.len() { return None; }
        let b = d.get(off..off + 4)?;
        Some(([b[3], b[2], b[1], b[0]], idx))
    }
    /// Like `tag_ref` but also requires the referenced tag to be of class `class`.
    pub fn tag_ref_of(&self, off: usize, class: &[u8; 4]) -> Option<usize> {
        let (_, idx) = self.tag_ref(off)?;
        (self.tag_class(idx).as_ref() == Some(class)).then_some(idx)
    }

    /// Resource kind `kind` (a `ResourceEntry::kind` index) has this zone type name.
    pub fn kind_is(&self, kind: u8, name: &str) -> bool {
        self.res_types.get(kind as usize).map_or(false, |n| n == name)
    }
    pub const RES_BITMAP: &'static str = "bitmap_texture_interop_resource";
    pub const RES_GEOMETRY: &'static str = "render_geometry_api_resource_definition";
    pub const RES_BSP_INSTANCES: &'static str = "structure_bsp_cache_file_tag_resources";

    /// Resolve a string id: **17 index bits** / namespace above. The 16-bit Reach split fails on
    /// the big campaign string tables: m10_crash has 73 290 strings and its zone type name sid
    /// 0x1027B only resolves with >= 17 index bits (`bitmap_texture_interop_resource` on every
    /// map). Namespace-0 ids are pinned by the tests.
    /// TODO: pin the exact split for namespaced ids (17/8 vs more) - no namespaced sample yet.
    pub fn sid(&self, v: u32) -> String {
        let mask = 0x1FFFFu32;
        let low = (v & mask) as i64;
        let mut ns = ((v >> 17) & 0xFF) as usize;
        if self.ns.is_empty() { return String::new(); }
        let arr: Vec<i64> = self.ns.iter().map(|&x| (x as u32 & mask) as i64).collect();
        // namespace table: (min, start) per namespace, as in the Reach resolver
        let mut tbl: HashMap<usize, (i64, i64)> = HashMap::new();
        let mut start = arr[0];
        for (i, &a) in arr.iter().enumerate().skip(1) { tbl.insert(i, (0, start)); start += a; }
        tbl.insert(0, (arr[0], start));
        while !tbl.contains_key(&ns) && ns > 0 { ns -= 1; }
        let (mn, st) = tbl[&ns];
        let i = if low < mn { low } else { low - mn + st };
        if i < 0 || i as usize >= self.str_idx.len() { return String::new(); }
        let o = self.str_idx[i as usize];
        if o < 0 { return String::new(); }
        self.str_blob.cstr_at(o as usize, 1024)
    }

    // ---- resources ---------------------------------------------------------------------------

    /// Zone entry for a resource id (`salt << 16 | index`), or None when null / stale.
    pub fn resource_by_id(&self, rid: u32) -> Option<&ResourceEntry> {
        let e = self.resources.get((rid & 0xFFFF) as usize)?;
        (e.is_live() && (rid >> 16) as u16 == e.salt).then_some(e)
    }
    /// A resource's definition data (the serialized interop structs + fixed-up blocks).
    pub fn definition(&self, e: &ResourceEntry) -> &[u8] {
        let s = self.def_blob_off + e.def_off as usize;
        let n = e.def_len as usize;
        self.map.get(s..s + n).unwrap_or(&[])
    }
    /// Inflated page bytes (cached).
    pub fn page_data(&self, page: usize) -> Result<Arc<Vec<u8>>> {
        if let Some(p) = self.page_cache.lock().unwrap().get(&page) { return Ok(p.clone()); }
        let pg = *self.pages.get(page).ok_or_else(|| anyhow!("page {page} out of range"))?;
        let bytes = match pg.cache_index {
            -1 => read_page(&self.map, self.data_table, &pg)?,
            ci => {
                let sc = self.shared_cache(ci)?;
                read_page(&sc.map, sc.data_table, &pg)?
            }
        };
        let arc = Arc::new(bytes);
        self.page_cache.lock().unwrap().insert(page, arc.clone());
        Ok(arc)
    }
    fn shared_cache(&self, ci: i16) -> Result<Arc<SharedCache>> {
        if let Some(s) = self.shared.lock().unwrap().get(&ci) { return Ok(s.clone()); }
        // cacheIndex 2 = campaign.map (campaign model pages, e.g. m10_crash's `mode` resources)
        let name = match ci { 1 => "shared.map", 0 => "mainmenu.map", 2 => "campaign.map", _ => bail!("unknown shared cache index {ci}") };
        let p = self.path.parent().map(|d| d.join(name)).ok_or_else(|| anyhow!("no parent dir"))?;
        let f = std::fs::File::open(&p).with_context(|| format!("open shared cache {}", p.display()))?;
        // SAFETY: read-only mapping, file treated as immutable.
        let map = unsafe { memmap2::Mmap::map(&f) }.context("mmap shared")?;
        if map.len() < 0x1000 || &map[0..4] != b"daeh" { bail!("{} is not a cache", p.display()); }
        let sc = Arc::new(SharedCache { data_table: map[..].u32_at(HDR_DATA_TABLE), map });
        self.shared.lock().unwrap().insert(ci, sc.clone());
        Ok(sc)
    }
    /// Page bytes + base offset for stream `nibble` (4 primary / 6 secondary / 8 tertiary) of a
    /// resource. The nibble->segment-page mapping is 4->page[0], 8->page[1], 6->page[2].
    pub fn stream(&self, e: &ResourceEntry, nibble: u32) -> Result<(Arc<Vec<u8>>, usize)> {
        let k = match nibble { 4 => 0, 8 => 1, 6 => 2, n => bail!("fixup nibble {n} is not a page stream") };
        let seg = self.segments.get(e.segment.max(0) as usize).ok_or_else(|| anyhow!("segment {} out of range", e.segment))?;
        if e.segment < 0 || seg.page[k] < 0 { bail!("resource {} has no stream {nibble}", e.index); }
        let data = self.page_data(seg.page[k] as usize)?;
        let off = seg.off[k].max(0) as usize;
        if off > data.len() { bail!("segment offset past page end"); }
        Ok((data, off))
    }
    /// Bytes at a fixed-up page address (`nibble << 28 | offset`), `len` long.
    pub fn stream_bytes(&self, e: &ResourceEntry, addr: u32, len: usize) -> Result<Vec<u8>> {
        let (data, base) = self.stream(e, addr >> 28)?;
        let s = base + (addr & 0x0FFF_FFFF) as usize;
        data.get(s..s + len).map(|b| b.to_vec()).ok_or_else(|| anyhow!("stream range {s}+{len} past page ({})", data.len()))
    }
}

fn read_page(file: &[u8], data_table: u32, pg: &Page) -> Result<Vec<u8>> {
    let s = data_table as usize + pg.data_offset as usize;
    let raw = file.get(s..s + pg.compressed as usize).ok_or_else(|| anyhow!("page data out of file"))?;
    if pg.compressed == pg.decompressed { return Ok(raw.to_vec()); }
    let out = miniz_oxide::inflate::decompress_to_vec(raw).map_err(|e| anyhow!("inflate: {e:?}"))?;
    if out.len() != pg.decompressed as usize {
        bail!("page inflated to {} bytes, expected {}", out.len(), pg.decompressed);
    }
    Ok(out)
}

/// This engine's MCC maps folder, if this machine has one (tests skip otherwise).
/// `HMS_H4_MAPS` / `HMS_H2A_MAPS` point at a folder elsewhere (test / development input only).
///
/// Every MCC install root is found the same way the Reach picker finds its maps
/// (`mapcat::mcc_install_roots` - Steam registry -> `libraryfolders.vdf`, plus the Xbox /
/// Windows Store layout); this engine's folder (`halo4` / `groundhog`) sits beside `haloreach`
/// under each root. The Windows default Steam path is a last-resort fallback for the rare
/// install that the registry lookup misses.
pub fn engine_maps_dir(engine: Engine) -> Option<PathBuf> {
    if let Ok(v) = std::env::var(engine.maps_env()) { let p = PathBuf::from(v); if p.is_dir() { return Some(p); } }
    for root in crate::mapcat::mcc_install_roots() {
        let p = root.join(engine.folder()).join("maps");
        if p.is_dir() { return Some(p); }
    }
    let fallback = Path::new("C:\\Program Files (x86)\\Steam")
        .join("steamapps/common/Halo The Master Chief Collection")
        .join(engine.folder())
        .join("maps");
    if fallback.is_dir() { return Some(fallback); }
    None
}

/// The MCC halo4 maps folder, if this machine has one (tests skip otherwise).
pub fn maps_dir() -> Option<PathBuf> { engine_maps_dir(Engine::Halo4) }

/// The MCC groundhog (Halo 2 Anniversary) maps folder, if this machine has one. #h2a
pub fn h2a_maps_dir() -> Option<PathBuf> { engine_maps_dir(Engine::H2A) }

#[cfg(test)]
mod tests {
    use super::*;

    fn open(name: &str) -> Option<H4Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// Appendix-A numbers: ca_forge_ravine has 22275 tags / 261 classes, every tag named, and a
    /// 17-entry forge palette at scnr+0x2C4 whose names resolve through the 16/8 string ids.
    #[test]
    fn ravine_tag_index_and_palette() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        assert_eq!(c.tag_count(), 22275);
        assert_eq!(c.class_count(), 261);
        // tags 0..2 (draw, gpix, play) carry empty names on disk; every other tag is named
        let unnamed: Vec<usize> = (0..c.tag_count()).filter(|&i| c.tag_name(i).is_empty()).collect();
        assert!(unnamed.iter().all(|&i| i < 3) && unnamed.len() <= 3, "unnamed tags {unnamed:?}");
        assert_eq!(c.find_tags(b"scnr").len(), 1);
        assert_eq!(c.find_tags(b"sbsp").len(), 2);
        assert_eq!(c.find_tags(b"bitm").len(), 2888);
        assert_eq!(c.find_tags(b"mat ").len(), 1154);
        let scnr = c.tag_meta(c.find_tags(b"scnr")[0]).unwrap();
        let (n, po) = c.block(scnr + 0x2C4).unwrap();
        assert_eq!(n, 17, "forge palette count");
        let names: Vec<String> = (0..n).map(|i| c.sid(c.data().u32_at(po + i * 0x14))).collect();
        assert!(names.iter().all(|s| !s.is_empty() && s.is_ascii()), "{names:?}");
        assert_eq!(c.map_type, 1);
        assert_eq!(c.scenario, "levels\\multi\\ca_forge_ravine\\ca_forge_ravine");
    }

    /// Paging: 40 local + 30 shared pages inflate to their decompressed size; the segment stride
    /// (24) and the zone entry layout hold (sum of definition lengths == blob, prefix == b38[0],
    /// every segment index unique and in range).
    #[test]
    fn ravine_pages_and_gestalt() {
        let Some(c) = open("ca_forge_ravine.map") else { return };
        assert_eq!(c.pages.len(), 6110);
        let local: Vec<usize> = (0..c.pages.len()).filter(|&i| c.pages[i].cache_index == -1).collect();
        let shared: Vec<usize> = (0..c.pages.len()).filter(|&i| c.pages[i].cache_index == 1).collect();
        assert_eq!((local.len(), shared.len()), (2862, 3248));
        for &i in local.iter().take(40).chain(shared.iter().take(30)) {
            let d = c.page_data(i).unwrap();
            assert_eq!(d.len(), c.pages[i].decompressed as usize, "page {i}");
        }
        let live: Vec<&ResourceEntry> = c.resources.iter().filter(|e| e.is_live()).collect();
        let sum: u64 = live.iter().map(|e| e.def_len as u64).sum();
        assert_eq!(sum, c.def_blob_len as u64, "definition data packed in entry order");
        let mut prefix = 0u32;
        for e in &live {
            assert_eq!(e.def_off, prefix, "entry {} b38[0] == prefix sum", e.index);
            prefix += e.def_len;
            assert!(e.segment >= 0 && (e.segment as usize) < c.segments.len());
            assert!((e.def_addr & 0x0FFF_FFFF) < e.def_len.max(1));
        }
        let mut segs: Vec<i16> = live.iter().map(|e| e.segment).collect();
        segs.sort();
        segs.dedup();
        assert_eq!(segs.len(), live.len(), "one segment per live resource");
        assert_eq!(c.segments.len(), live.len());
    }

    #[test]
    fn wraparound_basics() {
        let Some(c) = open("wraparound.map") else { return };
        assert_eq!(c.tag_count(), 16133);
        assert_eq!(c.pages.len(), 3798);
        assert_eq!(c.segments.len(), 3370);
        assert_eq!(c.resources.len(), 5098);
        assert_eq!(c.def_blob_len, 718008);
        assert_eq!(c.resources.iter().filter(|e| e.is_live()).count(), 3370);
        // every live resource's page streams lie inside their pages
        for e in c.resources.iter().filter(|e| e.is_live()) {
            for &(_, addr) in &e.fixups {
                let nib = addr >> 28;
                if nib == 2 { continue; }
                let (data, base) = match c.stream(e, nib) { Ok(v) => v, Err(_) => continue };
                assert!(base + (addr & 0x0FFF_FFFF) as usize <= data.len(), "res {} addr {addr:#x}", e.index);
            }
        }
    }

    #[test]
    fn shared_cache_headers() {
        let Some(dir) = maps_dir() else { return };
        for (name, ty) in [("shared.map", 3), ("campaign.map", 4)] {
            let c = H4Cache::open(&dir.join(name)).unwrap();
            assert_eq!(c.map_type, ty);
            assert_eq!(c.data_table, 0x1E000);
        }
        assert!(is_halo4_cache(&dir.join("wraparound.map")));
    }
}
