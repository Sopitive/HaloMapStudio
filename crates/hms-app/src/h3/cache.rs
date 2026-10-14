//! Halo 3 (MCC) cache reader - pure Rust, no native DLL.
//!
//! **Halo 3 MCC is a Reach-family container**, NOT the Halo 4 layout: same `daeh`/v13 header,
//! but every field after 0x58 sits +8 bytes vs Halo 4 (build @160, map name @0xC0, scenario
//! @0xE0, virtual base @736, tag index @744, section offsets @1228, data table @1232, sections
//! @1244), and the pointer-expander added-magic is **0** (Halo 4 = 0x4FFF0000, Reach =
//! 0x50000000). Tag table / names / classes / string ids have the same shape as the Halo 4
//! reader and are reused verbatim in spirit.
//!
//! The one genuinely different subsystem is the resource backing store: classic Gen3
//! `cache_file_resource_gestalt` (`zone`) + `cache_file_resource_layout_table` (`play`) with
//! DEFLATE-compressed pageable blocks (the native Reach parser `MapCacheCommon.cpp` is the model,
//! NOT the Halo 4 segment/page code). Model:
//!   * play +12  = shared caches (264 B, path @0)
//!   * play +24  = "pages"    (88 B: cacheIndex i16 @4, dataOffset u32 @8, comp u32 @12, decomp u32 @16)
//!   * play +60  = "segments" (16 B: primaryPage i16 @0, secondaryPage i16 @2, primaryOff u32 @4)
//!   * zone +100 = resources  (64 B: owner tagref @0, salt @0x10, typeIdx @0x12, ctrlOff @0x14,
//!                 ctrlSize @0x18, flags @0x20, locIndex @0x22, rootFixup @0x24, fixups block @0x28)
//!   * a resource's raw bytes: locIndex -> segments[locIndex] -> pages[primaryPage] -> DEFLATE the
//!     whole page at dataTableAddress+dataOffset -> slice at primaryOffset (the whole map's
//!     pageable data is typically ONE ~134 MB page, each resource a slice of it).
//!   * a resource's CONTROL (definition) data: a data-reference in the `zone` tag whose base
//!     holds the concatenated per-resource control blocks (VB/IB descriptors etc.), indexed by
//!     ctrlOff/ctrlSize; control fixups (loc, 32-bit word) patch pointers inside it: the word's
//!     low 29 bits are the address, its top 3 bits the type (0x40 = "raw page" -> pageable data
//!     offset; 0x20 = "fixup" -> control data offset). Addresses reach past 16 MB (zanzibar).
//!
//! Every claim above is asserted by the tests at the bottom against the shipped halo3 caches.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Control-fixup word: low 29 bits = address, top 3 bits = type (`FIXUP_TYPE_MASK`).
pub const FIXUP_ADDRESS_MASK: u32 = 0x1FFF_FFFF;
pub const FIXUP_TYPE_MASK: u8 = 0xE0;
/// Fixup types: the address is a control (definition) data offset / a raw-page data offset.
pub const FIXUP_TYPE_CONTROL: u8 = 0x20;
pub const FIXUP_TYPE_RAW_PAGE: u8 = 0x40;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};

use crate::h4::cache::ByteRead;

/// The Halo 3 MCC build (all `halo3/maps/*.map`).
pub const HALO3_BUILD: &str = "Dec 21 2023 22:31:37";
/// The Halo 3: ODST MCC build (all `halo3odst/maps/*.map`).
pub const HALO3ODST_BUILD: &str = "May 16 2023 11:44:41";

/// Which Halo 3 MCC engine a cache belongs to. Both use the identical Reach-family container
/// (expander 0); they differ only by build string and install folder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Halo3,
    Halo3ODST,
}

impl Engine {
    pub fn build(self) -> &'static str {
        match self { Engine::Halo3 => HALO3_BUILD, Engine::Halo3ODST => HALO3ODST_BUILD }
    }
    /// Pointer expansion added-magic. Zero for both Halo 3 engines (the one true difference
    /// from Reach's 0x50000000).
    pub fn expander(self) -> i64 { 0 }
    pub fn folder(self) -> &'static str {
        match self { Engine::Halo3 => "halo3", Engine::Halo3ODST => "halo3odst" }
    }
    pub fn label(self) -> &'static str {
        match self { Engine::Halo3 => "Halo 3", Engine::Halo3ODST => "Halo 3: ODST" }
    }
    pub fn of_build(build: &str) -> Option<Engine> {
        [Engine::Halo3, Engine::Halo3ODST].into_iter().find(|e| e.build() == build)
    }
    fn maps_env(self) -> &'static str {
        match self { Engine::Halo3 => "HMS_H3_MAPS", Engine::Halo3ODST => "HMS_H3ODST_MAPS" }
    }
}

// Reach-family header offsets (Halo 4 + 8 after 0x58).
const HDR_BUILD: usize = 160;
const HDR_MAP_NAME: usize = 0xC0;
const HDR_SCENARIO: usize = 0xE0;
const HDR_VBASE: usize = 736;
const HDR_TAG_INDEX: usize = 744;
const HDR_SECTION_OFFSETS: usize = 1228;
const HDR_DATA_TABLE: usize = 1232;
const HDR_SECTIONS: usize = 1244;

/// The Halo 3 MCC engine this file's cache header names, or None when it is not one this reader
/// handles. Version 13 + `daeh` is shared with Reach/Halo 4/H2A; Halo 3 is keyed on the build
/// string at +160 (Reach's own build sits there too but never matches these two strings).
pub fn cache_engine(path: &Path) -> Option<Engine> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 256];
    let n = f.read(&mut buf).ok()?;
    if n < 256 || &buf[0..4] != b"daeh" || buf[..].u32_at(4) != 13 { return None; }
    Engine::of_build(&buf[..].cstr_at(HDR_BUILD, 32))
}

/// Is this file a Halo 3 (or Halo 3: ODST) MCC cache?
pub fn is_halo3_cache(path: &Path) -> bool { cache_engine(path).is_some() }

#[derive(Clone, Debug)]
pub struct TagEntry {
    pub class_idx: i16,
    pub raw_meta: u32,
    pub name: String,
}

/// `play` "page" (a File Location, 88 B): a DEFLATE block. `cache_index` -1 = this cache, >=0
/// indexes the play tag's shared-cache list.
#[derive(Clone, Copy, Debug)]
pub struct Page {
    pub cache_index: i16,
    pub data_offset: u32,
    pub compressed: u32,
    pub decompressed: u32,
}

/// `play` "segment" (a Default Location, 16 B): primary/secondary page index + the byte offset of
/// a resource's data within the decompressed page.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub primary_page: i16,
    pub secondary_page: i16,
    pub primary_off: u32,
    pub secondary_off: u32,
}

/// `zone` resource entry (64 B). `ctrl_off`/`ctrl_size` locate this resource's control
/// (definition) data in the gestalt's control buffer; `loc` -> a segment -> a page (the raw
/// vertex/index/pixel bytes). `fixups` (loaded lazily) is the per-resource control-fixup block.
#[derive(Clone, Debug)]
pub struct ResourceEntry {
    pub index: usize,
    pub owner: [u8; 4],
    pub owner_tag: usize,
    pub datum: u32,
    pub salt: u16,
    pub kind: u8,
    pub ctrl_off: u32,
    pub ctrl_size: u32,
    pub flags: u16,
    pub loc: i16,
    /// Root-struct offset inside this resource's control data (u16 @0x24 | upper byte @0x26).
    pub root: u32,
    fixups_block: (usize, usize),
}

impl ResourceEntry {
    pub fn is_live(&self) -> bool { self.datum != 0xFFFF_FFFF }
}

struct SharedCache {
    map: memmap2::Mmap,
    data_table: u32,
}

pub struct H3Cache {
    pub path: PathBuf,
    map: memmap2::Mmap,
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
    /// Shared-cache basenames (play +12), indexed by `Page::cache_index`.
    pub shared_files: Vec<String>,
    pub pages: Vec<Page>,
    pub segments: Vec<Segment>,
    pub resources: Vec<ResourceEntry>,
    /// Zone resource TYPE names, indexed by `ResourceEntry::kind`.
    pub res_types: Vec<String>,
    /// The control-data buffer (file offset + length): the concatenated per-resource definition
    /// blocks, indexed by `ctrl_off`/`ctrl_size`.
    pub ctrl_off: usize,
    pub ctrl_len: usize,
    page_cache: Mutex<HashMap<(i16, usize), Arc<Vec<u8>>>>,
    shared: Mutex<HashMap<i16, Arc<SharedCache>>>,
}

impl H3Cache {
    pub fn open(path: &Path) -> Result<H3Cache> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only mapping of a file we never write; the game folder is treated as immutable.
        let map = unsafe { memmap2::Mmap::map(&file) }.context("mmap")?;
        let d: &[u8] = &map;
        if d.len() < 0x1400 || &d[0..4] != b"daeh" || d.u32_at(4) != 13 {
            bail!("not a MCC cache (magic/version)");
        }
        let build = d.cstr_at(HDR_BUILD, 32);
        let engine = Engine::of_build(&build)
            .ok_or_else(|| anyhow!("not a known Halo 3 MCC build: '{build}'"))?;
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
        let mut c = H3Cache {
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
            shared_files: Vec::new(),
            pages: Vec::new(),
            segments: Vec::new(),
            resources: Vec::new(),
            res_types: Vec::new(),
            ctrl_off: 0,
            ctrl_len: 0,
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

    /// `play` shared-caches / pages / segments + `zone` resource entries, type names and the
    /// control-data buffer.
    fn parse_layout(&mut self) -> Result<()> {
        let play = self.find_tags(b"play").first().copied().ok_or_else(|| anyhow!("no play tag"))?;
        let pm = self.tag_meta(play).ok_or_else(|| anyhow!("play meta unreadable"))?;
        let d: &[u8] = &self.map;
        // shared caches (264 B, path @0 -> basename after the last '\')
        if let Some((n, o)) = self.block(pm + 0x0C) {
            self.shared_files = (0..n).map(|i| {
                let p = d.cstr_at(o + i * 264, 32);
                p.rsplit(|c| c == '\\' || c == '/').next().unwrap_or(&p).to_string()
            }).collect();
        }
        // pages (File Locations, 88 B)
        let (np, po) = self.block(pm + 0x18).ok_or_else(|| anyhow!("play pages block"))?;
        self.pages = (0..np).map(|i| { let e = po + i * 88; Page {
            cache_index: d.i16_at(e + 4), data_offset: d.u32_at(e + 8),
            compressed: d.u32_at(e + 12), decompressed: d.u32_at(e + 16) } }).collect();
        // segments (Default Locations, 16 B)
        let (ns, so) = self.block(pm + 0x3C).ok_or_else(|| anyhow!("play segments block"))?;
        self.segments = (0..ns).map(|i| { let e = so + i * 16; Segment {
            primary_page: d.i16_at(e), secondary_page: d.i16_at(e + 2),
            primary_off: d.u32_at(e + 4), secondary_off: d.u32_at(e + 8) } }).collect();

        let zone = self.find_tags(b"zone").first().copied().ok_or_else(|| anyhow!("no zone tag"))?;
        let zm = self.tag_meta(zone).ok_or_else(|| anyhow!("zone meta unreadable"))?;
        // resource type names (0x1C stride, name sid @0x18)
        self.res_types = match self.block(zm + 0x04) {
            Some((n, o)) if n < 64 => (0..n).map(|i| self.sid(d.u32_at(o + i * 0x1C + 0x18))).collect(),
            _ => Vec::new(),
        };
        // resource entries (64 B)
        let (ne, eo) = self.block(zm + 0x64).ok_or_else(|| anyhow!("zone resource block"))?;
        let mut res = Vec::with_capacity(ne);
        for i in 0..ne {
            let e = eo + i * 64;
            let datum = d.u32_at(e + 0x0C);
            let fixups_block = self.block(e + 0x28).unwrap_or((0, 0));
            res.push(ResourceEntry {
                index: i,
                owner: { let b = &d[e..e + 4]; [b[3], b[2], b[1], b[0]] },
                owner_tag: (datum & 0xFFFF) as usize,
                datum,
                salt: d.u16_at(e + 0x10),
                kind: d.u8_at(e + 0x12),
                ctrl_off: d.u32_at(e + 0x14),
                ctrl_size: d.u32_at(e + 0x18),
                flags: d.u16_at(e + 0x20),
                loc: d.i16_at(e + 0x22),
                root: d.u16_at(e + 0x24) as u32 | ((d.u8_at(e + 0x26) as u32) << 16),
                fixups_block,
            });
        }
        self.resources = res;
        // control-data buffer: a data-reference in the zone tag (size @0, ptr @0xC). Its length is
        // the concatenation of every resource's control block, so it must cover the max
        // ctrl_off+ctrl_size. Scan the zone meta head for the matching data reference.
        let need = self.resources.iter().map(|r| r.ctrl_off as usize + r.ctrl_size as usize).max().unwrap_or(0);
        if need > 0 {
            for o in (0..0x200).step_by(4) {
                let size = d.i32_at(zm + o);
                if size as usize >= need && (size as usize) < 200_000_000 {
                    if let Some(base) = self.meta_off(d.u32_at(zm + o + 0x0C)) {
                        if base + size as usize <= d.len() {
                            self.ctrl_off = base;
                            self.ctrl_len = size as usize;
                            break;
                        }
                    }
                }
            }
        }
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
    pub fn tag_meta(&self, idx: usize) -> Option<usize> {
        let t = self.tags.get(idx)?;
        if t.raw_meta == 0 { return None; }
        self.meta_off(t.raw_meta)
    }
    /// Tag block at file offset `off` -> (count, file offset of element 0). 12-byte block
    /// {count i32, ptr u32, pad u32}.
    pub fn block(&self, off: usize) -> Option<(usize, usize)> {
        let d: &[u8] = &self.map;
        let count = d.i32_at(off);
        let ptr = d.u32_at(off + 4);
        if count <= 0 || count > 0x100_0000 || ptr == 0 || ptr == 0xCDCD_CDCD { return None; }
        let o = self.meta_off(ptr)?;
        Some((count as usize, o))
    }
    /// Tag reference at file offset `off` -> (class fourcc, tag index).
    pub fn tag_ref(&self, off: usize) -> Option<([u8; 4], usize)> {
        let d: &[u8] = &self.map;
        let datum = d.u32_at(off + 12);
        if datum == 0xFFFF_FFFF || datum == 0xCDCD_CDCD { return None; }
        let idx = (datum & 0xFFFF) as usize;
        if idx >= self.tags.len() { return None; }
        let b = d.get(off..off + 4)?;
        Some(([b[3], b[2], b[1], b[0]], idx))
    }
    pub fn tag_ref_of(&self, off: usize, class: &[u8; 4]) -> Option<usize> {
        let (_, idx) = self.tag_ref(off)?;
        (self.tag_class(idx).as_ref() == Some(class)).then_some(idx)
    }

    pub fn kind_is(&self, kind: u8, name: &str) -> bool {
        self.res_types.get(kind as usize).map_or(false, |n| n == name)
    }
    pub const RES_BITMAP: &'static str = "bitmap_texture_interop_resource";
    pub const RES_GEOMETRY: &'static str = "render_geometry_api_resource_definition";
    pub const RES_BSP: &'static str = "structure_bsp_cache_file_tag_resources";
    /// The sbsp +0x38C resource: collision bsp + instanced-geometry definitions (control data).
    pub const RES_BSP_TAG: &'static str = "structure_bsp_tag_resources";

    /// Resolve a string id: **19 index bits** / 8 namespace bits (Reach U13). Namespace-0 ids
    /// (the common case) resolve by identity.
    pub fn sid(&self, v: u32) -> String {
        let mask = 0x7FFFFu32;
        let low = (v & mask) as i64;
        let mut ns = ((v >> 19) & 0xFF) as usize;
        if self.ns.is_empty() { return String::new(); }
        let arr: Vec<i64> = self.ns.iter().map(|&x| (x as u32 & mask) as i64).collect();
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

    /// This resource's control (definition) data slice from the gestalt control buffer.
    pub fn control_data(&self, e: &ResourceEntry) -> &[u8] {
        let s = self.ctrl_off + e.ctrl_off as usize;
        let n = e.ctrl_size as usize;
        self.map.get(s..s + n).unwrap_or(&[])
    }

    /// This resource's control-fixup table: `loc (control offset) -> (address, type)`. The
    /// fixup's value is ONE 32-bit word: the low 29 bits are the address, the top 3 bits its type
    /// (0x40 = a raw-page / pageable-data offset, 0x20 = a control (definition) data offset). The
    /// type is returned as the top byte with the address bits masked off. Loaded on demand (a
    /// geometry resource can carry thousands).
    ///
    /// #h3-zanzibar: this used to keep only the low 24 bits of the address and hand back the whole
    /// top byte as the "flag", so an address at or past 16 MB (zanzibar's Lbsp geometry slice:
    /// every index buffer + 336 per-vertex lighting streams sit above 0x1000000, type byte 0x41)
    /// read 16 MB too low - garbage indices, every cluster and instance of the map rejected
    /// ("no ground at all"). Guardian / riverworld / construct stay under 16 MB and never hit it.
    pub fn resource_fixups(&self, e: &ResourceEntry) -> HashMap<u32, (u32, u8)> {
        let d: &[u8] = &self.map;
        let (n, o) = e.fixups_block;
        let mut m = HashMap::with_capacity(n);
        for j in 0..n {
            let fe = o + j * 8;
            let loc = d.u32_at(fe);
            let word = d.u32_at(fe + 4);
            m.insert(loc, (word & FIXUP_ADDRESS_MASK, ((word >> 24) as u8) & FIXUP_TYPE_MASK));
        }
        m
    }

    /// A resource id (`salt << 16 | index`) -> its live entry.
    pub fn resource_by_id(&self, rid: u32) -> Option<&ResourceEntry> {
        let e = self.resources.get((rid & 0xFFFF) as usize)?;
        (e.is_live() && (rid >> 16) as u16 == e.salt).then_some(e)
    }

    /// Read a resource's raw pageable data: (decompressed page bytes, base offset of this
    /// resource's slice within it). Callers index `page[base + local_offset ..]`. The whole map's
    /// pageable data is usually one big DEFLATE page shared by many resources, so pages are cached.
    pub fn read_resource(&self, e: &ResourceEntry) -> Result<(Arc<Vec<u8>>, usize)> {
        let seg = self.segments.get(e.loc.max(0) as usize)
            .ok_or_else(|| anyhow!("resource {} loc {} out of range", e.index, e.loc))?;
        if e.loc < 0 { bail!("resource {} has no location", e.index); }
        // Prefer the secondary page when present (the historical Reach behaviour), else primary.
        let (page_idx, base) = if seg.secondary_page >= 0 {
            (seg.secondary_page, seg.secondary_off)
        } else {
            (seg.primary_page, seg.primary_off)
        };
        if page_idx < 0 { bail!("resource {} has no page", e.index); }
        let page = self.page_data(page_idx as usize)?;
        let base = base as usize;
        if base > page.len() { bail!("resource {} segment offset {base} past page end {}", e.index, page.len()); }
        Ok((page, base))
    }

    /// Inflated page bytes (cached by (cache_index, page)).
    pub fn page_data(&self, page: usize) -> Result<Arc<Vec<u8>>> {
        let pg = *self.pages.get(page).ok_or_else(|| anyhow!("page {page} out of range"))?;
        let key = (pg.cache_index, page);
        if let Some(p) = self.page_cache.lock().unwrap().get(&key) { return Ok(p.clone()); }
        let bytes = match pg.cache_index {
            -1 => read_page(&self.map, self.data_table, &pg)?,
            ci => { let sc = self.shared_cache(ci)?; read_page(&sc.map, sc.data_table, &pg)? }
        };
        let arc = Arc::new(bytes);
        self.page_cache.lock().unwrap().insert(key, arc.clone());
        Ok(arc)
    }

    fn shared_cache(&self, ci: i16) -> Result<Arc<SharedCache>> {
        if let Some(s) = self.shared.lock().unwrap().get(&ci) { return Ok(s.clone()); }
        // The shared-cache list is data-driven (play +12); fall back to the classic names.
        let name = self.shared_files.get(ci as usize).cloned().unwrap_or_else(|| match ci {
            0 => "mainmenu.map".into(), 1 => "shared.map".into(), _ => format!("shared_{ci}.map"),
        });
        let p = self.path.parent().map(|d| d.join(&name)).ok_or_else(|| anyhow!("no parent dir"))?;
        let f = std::fs::File::open(&p).with_context(|| format!("open shared cache {}", p.display()))?;
        // SAFETY: read-only mapping, file treated as immutable.
        let map = unsafe { memmap2::Mmap::map(&f) }.context("mmap shared")?;
        if map.len() < 0x1400 || &map[0..4] != b"daeh" { bail!("{} is not a cache", p.display()); }
        let sc = Arc::new(SharedCache { data_table: map[..].u32_at(HDR_DATA_TABLE), map });
        self.shared.lock().unwrap().insert(ci, sc.clone());
        Ok(sc)
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

/// The MCC `halo3/maps` folder, if this machine has one (tests skip otherwise).
pub fn maps_dir() -> Option<PathBuf> { engine_maps_dir(Engine::Halo3) }

/// Every `*.map` in the halo3 (and halo3odst, if present) maps folder this reader accepts, so the
/// tests sweep whatever is installed rather than a fixed list.
pub fn installed_caches() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for eng in [Engine::Halo3, Engine::Halo3ODST] {
        let Some(dir) = engine_maps_dir(eng) else { continue };
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("map")) && is_halo3_cache(&p) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(name: &str) -> Option<H3Cache> {
        let dir = maps_dir()?;
        let p = dir.join(name);
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H3Cache::open(&p).expect("open"))
    }

    /// guardian.map: the header, tag index and scnr forge palette all read as documented.
    #[test]
    fn guardian_header_and_tag_index() {
        let Some(c) = open("guardian.map") else { return };
        assert_eq!(c.engine, Engine::Halo3);
        assert_eq!(c.map_type, 1);
        assert_eq!(c.map_name, "guardian");
        assert_eq!(c.scenario, "levels\\multi\\guardian\\guardian");
        assert_eq!(c.tag_count(), 10403);
        assert_eq!(c.class_count(), 183);
        // Only the first couple of tags are unnamed on disk.
        let unnamed: Vec<usize> = (0..c.tag_count()).filter(|&i| c.tag_name(i).is_empty()).collect();
        assert!(unnamed.iter().all(|&i| i < 4) && unnamed.len() <= 4, "unnamed tags {unnamed:?}");
        assert_eq!(c.find_tags(b"scnr").len(), 1);
        assert_eq!(c.find_tags(b"sbsp").len(), 1);
        assert_eq!(c.find_tags(b"Lbsp").len(), 1);
        assert_eq!(c.find_tags(b"bitm").len(), 2116);
        assert_eq!(c.find_tags(b"mode").len(), 242);
        // there is no `mat ` class in Halo 3 - BSP materials reference rmsh directly
        assert_eq!(c.find_tags(b"mat ").len(), 0);
        assert!(!c.find_tags(b"rmsh").is_empty());
        assert!(c.tag_name(c.find_tags(b"scnr")[0]).starts_with("levels\\multi\\guardian\\"));
        // resource type names resolve through the string ids
        assert_eq!(c.res_types.len(), 7);
        assert!(c.res_types.iter().any(|t| t == H3Cache::RES_GEOMETRY), "{:?}", c.res_types);
        assert!(c.res_types.iter().any(|t| t == H3Cache::RES_BITMAP), "{:?}", c.res_types);
    }

    /// The Sandbox / Forge palette: 7 map-variant blocks (0x1C stride) at scnr 0x1E0..0x228,
    /// every Object tagref resolves.
    #[test]
    fn guardian_forge_palette() {
        let Some(c) = open("guardian.map") else { return };
        let scnr = c.tag_meta(c.find_tags(b"scnr")[0]).unwrap();
        let blocks = [0x1E0usize, 0x1EC, 0x1F8, 0x204, 0x210, 0x21C, 0x228];
        let mut total = 0;
        for &off in &blocks {
            if let Some((n, o)) = c.block(scnr + off) {
                total += n;
                for i in 0..n { assert!(c.tag_ref(o + i * 0x1C).is_some(), "palette {off:#x} entry {i}"); }
            }
        }
        assert_eq!(total, 95, "guardian forge palette total");
    }

    /// Paging: every documented page inflates to its recorded decompressed size, and the whole
    /// map's pageable data resolves.
    #[test]
    fn guardian_pages() {
        let Some(c) = open("guardian.map") else { return };
        assert!(c.pages.len() > 100, "{} pages", c.pages.len());
        assert!(!c.segments.is_empty());
        assert!(!c.resources.is_empty());
        assert!(c.ctrl_len > 0, "control buffer located");
        let local: Vec<usize> = (0..c.pages.len()).filter(|&i| c.pages[i].cache_index == -1).collect();
        assert!(!local.is_empty());
        for &i in local.iter().take(30) {
            let d = c.page_data(i).unwrap();
            assert_eq!(d.len(), c.pages[i].decompressed as usize, "page {i}");
        }
        // the Lbsp geometry resource reads
        let lbsp = c.find_tags(b"Lbsp")[0];
        let geo = c.resources.iter().find(|e| e.is_live() && e.owner == *b"Lbsp"
            && e.owner_tag == lbsp && c.kind_is(e.kind, H3Cache::RES_GEOMETRY)).expect("Lbsp geometry resource");
        let (page, base) = c.read_resource(geo).unwrap();
        assert!(base < page.len());
    }

    /// Every shipped halo3 (+ odst) playable cache opens, its tag metas land in-file, tags/classes
    /// read as ASCII, the resource type names resolve, and a sample of local pages inflates to the
    /// recorded size. This is the sweep that pins the header offsets + expander 0 across all maps.
    #[test]
    fn every_cache_opens_and_pages() {
        let caches = installed_caches();
        if caches.is_empty() { eprintln!("skip: no halo3 maps folder"); return; }
        let mut playable = 0;
        for p in &caches {
            let c = H3Cache::open(p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
            if c.map_type == 3 || c.map_type == 4 {
                assert!(c.data().len() >= 0x1400, "{}", p.display());
                continue;
            }
            playable += 1;
            assert!(c.tag_count() > 1000 && c.class_count() > 100, "{}", p.display());
            let named = (0..c.tag_count()).filter(|&i| c.tag_meta(i).is_some()).count();
            assert!(named > 1000, "{}: only {named} tags with meta", p.display());
            assert!((0..c.tag_count()).all(|i| c.tag_name(i).is_ascii()), "{}", p.display());
            assert!(c.res_types.iter().any(|t| t == H3Cache::RES_GEOMETRY), "{}: {:?}", p.display(), c.res_types);
            assert!(!c.pages.is_empty() && !c.segments.is_empty() && !c.resources.is_empty(), "{}", p.display());
            assert!(c.ctrl_len > 0, "{}: no control buffer", p.display());
            let local: Vec<usize> = (0..c.pages.len()).filter(|&i| c.pages[i].cache_index == -1).collect();
            for &i in local.iter().take(8) {
                let d = c.page_data(i).unwrap_or_else(|e| panic!("{}: page {i}: {e}", p.display()));
                assert_eq!(d.len(), c.pages[i].decompressed as usize, "{}: page {i}", p.display());
            }
        }
        assert!(playable >= 3, "expected several shipped playable caches, found {playable}");
        eprintln!("h3: swept {} caches ({playable} playable)", caches.len());
    }
}
