//! Halo 4 (MCC) map variant codec: `.mvar` = BLF (`_blf`, `chdr` v10, `mvar` chunk v50,
//! `_eof`, `_fsm`), the `mvar` chunk carrying a 28 672-byte MSB-first BITSTREAM after its
//! 12-byte chunk header + 20-byte SHA-1 + 4-byte length. Decoder, encoder (byte-identical
//! re-encode of every shipped file), forge-palette resolution and the file-level save.
//!
//! The layout comes from the retail `halo4.dll` decoder (IDA on the shipped DLL):
//!   c_map_variant::decode              sub_1800B3BD8   header, 651 slots, 256 quotas, 4 trait sets
//!   s_variant_object_datum::decode     sub_1800B7C9C   present/flags/quota/variant/pos/axes/parent
//!   mp object properties decode        sub_1800B7410   shape/type/flags/team/spawn time/colour/...
//!   boundary shape decode              sub_1806912B8   2-bit shape + 16-bit values
//!   position axis bit count            sub_18051AE50   ceil(extent / 2*min_step) capped 0x400000
//!   content header                     sub_18006A43C   activity is 2 bits (Reach: 3)
//! and holds on every one of the 388 shipped variants (see the tests + docs/halo4_mvar_layout.md):
//! the object array + quota table end exactly where the four trailing player-trait records start,
//! every quota index falls inside the base map's forge palette, every quota entry's
//! `placed_on_map` equals the number of objects of that quota, and `budget_spent` equals the sum
//! of the placed objects' palette prices. Field NAMES beyond that are inferred from value
//! distributions per palette entry (documented per field); orientation reuses the Reach decode
//! (same 20-bit up + 14-bit forward angle over [-pi, pi], constants read from the DLL).

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use glam::Vec3;

use super::cache::{ByteRead, H4Cache};
use super::objects::{model_of, H4Placement};

/// `mvar` chunk major version this decoder understands (Reach = 31).
pub const H4_MVAR_VERSION: u16 = 50;
/// Bitstream payload size (chunk size 28 712 - 12 header - 20 hash - 4 length - 4).
pub const H4_MVAR_PAYLOAD: usize = 28_676;
/// Bytes the engine's bitstream writer owns (0x7000; the chunk's last 4 bytes are a zero u32).
pub const H4_MVAR_BITSTREAM_BYTES: usize = 28_672;
/// Object slots per variant (same as Reach).
pub const H4_SLOTS: usize = 651;
/// Quota entries iterated by the engine (only `num_quotas` are stored).
pub const H4_QUOTA_SLOTS: usize = 256;

// ---------------------------------------------------------------------------------------------
// bit reader (MSB first, out-of-range bits read as 0 like the Reach reader)
// ---------------------------------------------------------------------------------------------

struct Bits<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }
    fn bits(&mut self, n: u32) -> u64 {
        let mut r = 0u64;
        for _ in 0..n {
            let byte = self.pos >> 3;
            let off = 7 - (self.pos & 7);
            let bit = self.buf.get(byte).map_or(0, |b| (b >> off) & 1);
            r = (r << 1) | bit as u64;
            self.pos += 1;
        }
        r
    }
    fn flag(&mut self) -> bool { self.bits(1) != 0 }
    fn u8(&mut self) -> u8 { self.bits(8) as u8 }
    fn i8(&mut self) -> i8 { self.bits(8) as u8 as i8 }
    fn u16(&mut self) -> u16 { self.bits(16) as u16 }
    fn u32(&mut self) -> u32 { self.bits(32) as u32 }
    fn u64(&mut self) -> u64 { self.bits(64) }
    fn f32(&mut self) -> f32 { f32::from_bits(self.u32()) }
    /// `read_index(max, bits)`: a set flag means "none".
    fn index(&mut self, bits: u32) -> Option<u32> {
        if self.flag() { None } else { Some(self.bits(bits) as u32) }
    }
    fn cstr(&mut self, max: usize) -> String {
        let mut v = Vec::new();
        for _ in 0..max {
            let b = self.u8();
            if b == 0 { break; }
            v.push(b);
        }
        String::from_utf8_lossy(&v).into_owned()
    }
    fn wstr(&mut self, max: usize) -> String {
        let mut v = Vec::new();
        for _ in 0..max {
            let u = self.u16();
            if u == 0 { break; }
            v.push(u);
        }
        String::from_utf16_lossy(&v)
    }
}

// ---------------------------------------------------------------------------------------------
// decoded structures
// ---------------------------------------------------------------------------------------------

/// Boundary shape of an object (props +8, 2 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H4Shape {
    None = 0,
    Sphere = 1,
    Cylinder = 2,
    Box = 3,
}

impl H4Shape {
    fn from_bits(v: u64) -> Self {
        match v { 1 => H4Shape::Sphere, 2 => H4Shape::Cylinder, 3 => H4Shape::Box, _ => H4Shape::None }
    }
    /// 16-bit values stored: sphere radius; cylinder radius/top/bottom; box width/length/top/bottom.
    pub fn value_count(self) -> usize { [0, 1, 3, 4][self as usize] }
}

/// Type-specific tail of the multiplayer properties (the engine switches on the object type).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum H4TypeData {
    /// Types 1 (weapon: spare clips) and 12: one u8.
    Byte(u8),
    /// Type 20: u8 - 1 (Reach's named-location index by analogy; no shipped object uses it).
    Index(i16),
    /// Type 31 (trait zone): 5-bit trait-set index.
    TraitZone(u8),
    /// Every other type: two 5-bit values (teleporter channel + passability on types 13-15).
    Pair(u8, u8),
    /// Type 32 (initial ordnance): team, weapon index, 16-bit drop timing.
    InitialOrdnance { team: i8, weapon: u8, timing: u16 },
    /// Type 33 (random ordnance): 8 weapon indices (255 = none).
    RandomOrdnance([u8; 8]),
    /// Type 34 (objective ordnance): leading byte + 8 weapon indices.
    ObjectiveOrdnance { lead: u8, weapons: [u8; 8] },
    /// Type 35: nothing.
    Empty,
}

/// One present object slot (`s_variant_object_datum`, 76 bytes in the engine).
#[derive(Clone, Debug)]
pub struct H4PlacedObject {
    /// Slot index 0..650 - the object's IDENTITY for the encoder: a decoded
    /// object keeps its slot on save; a NEW object uses `NEW_SLOT` (any free slot) or any unique
    /// value >= 651 so other objects can name it as their `parent`.
    pub slot: u16,
    /// Bit offset of the slot inside the payload (diagnostics).
    pub bit: usize,
    /// 2-bit slot flags (1 on every shipped object).
    pub flags: u8,
    /// Forge palette quota index (flat index over the scnr +0x2C4 palette entries), None = empty.
    pub quota: Option<u8>,
    /// Variant index inside the quota's entry, None = default.
    pub variant: Option<u8>,
    /// Position "in bounds" flag (always set on shipped objects; no bsp escape is read).
    pub in_bounds: bool,
    pub pos: [f32; 3],
    pub up_is_global: bool,
    pub up_quant: u32,
    pub forward_angle_q: u32,
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    /// Parent slot (10 bits - 1; -1 = none). On save this names the parent's `slot` key and is
    /// remapped to wherever that object lands; a parent missing from the list orphans the child.
    pub parent: i16,
    /// OBJECT SCALE: 6-bit quantized real over [0, 10] with exact endpoints (datum +0x2C),
    /// stored raw. In halo4.dll `c_map_variant::create_object`
    /// (sub_1800B5EE8 -> sub_1800B6184) copies datum +0x2C into `s_object_placement_data`
    /// +0x58, which `object_new` (sub_1805D2034, 0x1805D2342..50) writes to the runtime
    /// object's scale (obj +0xA0 current / +0xA4 target - the pair `Object_SetScale`
    /// RVA 0x5D144C drives); `c_map_variant::update_object` (sub_1800B5B00) copies it back.
    /// q = 7 = `write_quantized_real(1.0)` on 99.9 percent of shipped objects and dequantises
    /// to 1.048 (the engine's own asymmetry); shipped non-default values: Vortex / Redoubt
    /// dominion shields q 14 (x2.18), Abandon initial spawns q 9 (x1.37), Landfall crates
    /// q 5 (x0.73), Blood Crash scorpion q 4 (x0.56). Use [`Self::scale`] / [`Self::set_scale`].
    pub scale_q: u8,
    /// Forge LOCK flag (datum +0x33, 1 bit; 30 shipped objects). Identified by code shape:
    /// the per-object getter sub_1800B99F8 feeds the Forge HUD state (sub_180497F3C
    /// +140) and the highlight style (sub_1800C2D78 -> outline mode 2), the setter
    /// sub_1800C0E4C(player, object, on) DROPS the object if the player is holding it, and the
    /// editor toggles it (sub_1800C04B4). Never read by the object create path.
    pub locked: bool,
    pub shape: H4Shape,
    /// Raw 16-bit shape values, 1/256 world units (consistent with every shipped boundary;
    /// TODO: verify the unit against the engine's shape encoder).
    pub shape_values: [u16; 4],
    /// Multiplayer object type (6 bits): 0 ordinary, 1 weapon, 2 grenade, 5 equipment, 7/8/9
    /// light/heavy/flying vehicle, 11 device (dominion terminal), 12 dominion pad, 13/14/15
    /// teleporter sender/receiver/2-way, 16 player spawn, 17 respawn zone, 26 safe area, 27
    /// kill area, 28 loadout camera, 31 trait zone, 32/33/34 initial/random/objective ordnance.
    pub object_type: u8,
    /// 10-bit placement flags: low byte = Reach's placement byte (bits 2/3 asymmetric/symmetric,
    /// bit 4 not-initially-placed, bit 5 unique spawn, bits 6-7 physics 0 normal/1 fixed/3
    /// phased); bits 8-9 are new in Halo 4: the create path maps bit 8 -> placement data +603
    /// flag 0x10 and bit 9 -> +603 flag 0x20 / object multiplayer flag 0x800 (sub_1800B6260,
    /// sub_1800B6574); no reader of either flag found, so their meaning is unknown.
    pub placement: u16,
    /// 4 bits - 1: 0..7 team colours, 8 = neutral, -1 = none (only types 33/34 have none).
    pub team: i8,
    /// Respawn time in seconds (weapons 30/60/120/180 etc.).
    pub spawn_time: u8,
    /// Object colour override (flag + 3 bits), None = team colour / default.
    pub color: Option<u8>,
    /// SPAWN SEQUENCE = the Forge object menu's "spawn sequence" ("This controls in
    /// what order the item will spawn.", string id 0x8058F; the menu constructor
    /// `sub_1803D1C64` binds that item to property block +0x0F = datum +0x43 with the shared
    /// -100..100 range object at `0x1803D1ED6`). Megalo `object.user_data` (= `user_data1`;
    /// number-variable scope 7 in `sub_180208A10` -> object multiplayer block +36, filled by
    /// `update_object_multiplayer_properties` sub_1800B6574). 343's own scripts use it as the
    /// KOTH hill order, extraction site number, dominion base number ("Forge Spawn sequence, AKA
    /// 'user_data'" - H4EK H4_GOAL_FORGE.txt). A custom Forge gametype reads it (with the
    /// `scale` label) as the scale seed, like Reach. Signed: shipped files carry -100 (156) and
    /// -1 (255).
    pub spawn_sequence: i8,
    /// USER DATA = the Forge object menu's "user data" ("Game mode specific data.",
    /// string id 0x80590, property block +0x10 = datum +0x44, same -100..100 range). Megalo
    /// `object.user_data2` (scope 8 -> multiplayer block +37); 343's Ricochet / Goal scripts use
    /// it as the ball-spawn offset selector (-5..3); 0 on all but a handful of shipped objects.
    pub user_data: i8,
    /// Up to four gametype label indices into `H4Variant::labels`.
    pub labels: [Option<u8>; 4],
    pub type_data: H4TypeData,
}

impl H4PlacedObject {
    pub fn physics(&self) -> u8 { ((self.placement >> 6) & 3) as u8 }
    pub fn label_names<'a>(&self, v: &'a H4Variant) -> Vec<&'a str> {
        self.labels.iter().flatten().filter_map(|&i| v.labels.get(i as usize).map(|s| s.as_str())).collect()
    }
}

/// One placeable-object quota entry (3 bytes, same as Reach).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H4Quota {
    pub minimum: u8,
    pub maximum: u8,
    pub placed: u8,
}

/// A decoded Halo 4 map variant.
#[derive(Clone, Debug)]
pub struct H4Variant {
    pub chunk_version: u16,
    /// Content header title / description as DISPLAYED: the header's UTF-16 string, or when that
    /// is a `$key` (17 of the 388 shipped variants), its English text from the MCC localization
    /// tables (`localization::display`; the raw key when no table has it).
    pub title: String,
    pub description: String,
    /// The header strings exactly as stored (`$h4_mvar_settler_name` or the literal title).
    pub title_key: String,
    pub description_key: String,
    pub author: String,
    pub editor: String,
    /// CreatedBy / ModifiedBy (unix timestamp, xuid) of the content header; the
    /// game stamps created = (save time, player xuid) on a NEW variant and modified = (save time,
    /// player xuid) when it re-saves someone else's (see docs/halo4_mvar_layout.md, "listing").
    pub created: (u64, u64),
    pub modified: (u64, u64),
    /// The `author-flags` bit of each history block (1 = the author was online;
    /// the engine asserts it is a single bit).
    pub created_online: bool,
    pub modified_online: bool,
    /// Content header `type` (engine name; raw 4 bits minus 1): 5 = map variant on
    /// every file both games ship. Kept for display; never rewritten.
    pub content_type: i8,
    /// `file-size`: the byte length of the file up to and including `_eof` (the
    /// `_fsm` signature chunk is NOT counted). 29 481 on every shipped file.
    pub file_size: u32,
    /// `uid` / `parent-uid` / `root-uid` / `game-id` (64 bits each). MCC keeps the
    /// template's ids on every Forge save (128 of 388 shipped files carry uid == parent == root);
    /// HMS copies them verbatim.
    pub uid: u64,
    pub parent_uid: u64,
    pub root_uid: u64,
    pub game_id: u64,
    /// `activity` (2 bits in Halo 4, raw value: 1 on every shipped file; 2 = matchmaking, which
    /// adds the 16-bit `hopper-id`), `game-mode` (3 bits: 4 on every shipped file) and
    /// `game-engine-type` (3 bits: 0 on every shipped file; 3 = campaign and 4 = firefight add
    /// their own blocks after the description). Read-only: changing them changes the header
    /// layout, so HMS never rewrites them.
    pub activity: u8,
    pub game_mode: u8,
    pub engine: u8,
    /// Map id from the content header (equals `map_id` on every shipped variant).
    pub header_map_id: u32,
    /// `megalo-category-index` (signed 8 bits; -1 on every shipped file): the
    /// menu category a game-variant file lists under. Editable (spliced by the encoder).
    pub category: i8,
    /// `hopper-id` (16 bits, only present when `activity == 2`). Never seen.
    pub hopper_id: Option<u16>,
    /// Variant body version: 50 (shipped) or 51 (saved by MCC). The encoder
    /// writes the version it read.
    pub version: u8,
    /// The map GUID stored by version 51 files (raw 16 memory bytes; a v50 file
    /// has none - the engine synthesises `{u64 lo = map_index | 0x8888_0000_0000_0000, hi 0}`
    /// from its map-id table sub_180072E34, and MCC saves store exactly that).
    pub map_guid: Option<[u8; 16]>,
    /// `map-variant-checksum` (32 bits; 0xFFFFFFFF on every shipped file) and
    /// `m_scenario_palette_crc` (32 bits): the CRC of the base map's forge palette. The loader
    /// compares the CRC with the loaded scenario's and rebuilds the variant from the scenario
    /// when they differ (H4EK `sub_1402F5A20`). Both copied verbatim.
    pub checksum: u32,
    pub palette_crc: u32,
    pub num_quotas: u16,
    /// Base map id (== `levl` chunk +12 big-endian in `maps/info/<map>.mapinfo`).
    pub map_id: u32,
    /// `built_in` / `m_built_from_xml` (1 bit each): 1 / 1 on 371 of 388 shipped files, 0 / 0 on
    /// the 17 `ca_forge_*` / `grif_*` files and on MCC Forge saves. Copied verbatim.
    pub built_in: bool,
    pub built_from_xml: bool,
    /// `world-bounds` xmin xmax ymin ymax zmin zmax: the box object positions are quantised
    /// against. Editable: the encoder re-quantises every object against the new
    /// box. NOTE the game replaces it with the scenario's world bounds when the variant is loaded
    /// on its own map (`rebuild_from`, H4EK `sub_1402F2880` -> `sub_1402F0DC0`), so it only
    /// governs what the FILE can store, not where the game lets objects live.
    pub bounds: [f32; 6],
    /// `maximum_budget` / `spent_budget`. Editable maximum; the loader also
    /// overwrites it from the scenario's sandbox budget on the matching map. Spent is recomputed
    /// by HMS from the palette prices on save (and by the game on load).
    pub budget_max: u32,
    pub budget_spent: u32,
    pub labels: Vec<String>,
    pub objects: Vec<H4PlacedObject>,
    /// `num_quotas` entries (engine iterates 256, stores only these).
    pub quotas: Vec<H4Quota>,
    /// Bit offsets: first object slot, end of the slot array, end of the quota table.
    pub obj_start_bit: usize,
    pub obj_end_bit: usize,
    pub quota_end_bit: usize,
    /// End of the four trait-set records = the last bit the engine wrote (`length` = ceil(/8)).
    pub end_bit: usize,
    /// Bit spans of the editable header fields (encoder splices new values in).
    pub spans: H4HeaderSpans,
    /// BLF facts filled in by `parse_h4_variant` (zero / false from
    /// `parse_h4_payload`, which only sees the bitstream): the stored chunk `length`, whether the
    /// stored SHA-1 matches it, and whether an `_fsm` chunk is present.
    pub chunk_length: u32,
    pub hash_ok: bool,
    pub has_fsm: bool,
}

impl H4Variant {
    /// The game-neutral global-field view (see `crate::mvar::VariantGlobals`).
    pub fn globals(&self) -> crate::mvar::VariantGlobals {
        crate::mvar::VariantGlobals {
            game: "Halo 4",
            content_type: self.content_type, file_size: self.file_size, uid: self.uid, parent_uid: self.parent_uid, root_uid: self.root_uid, game_id: self.game_id,
            activity: self.activity as i8, game_mode: self.game_mode, engine: self.engine, header_map_id: self.header_map_id, category: self.category,
            created: self.created, modified: self.modified, created_online: self.created_online, modified_online: self.modified_online, hopper_id: self.hopper_id,
            version: self.version, checksum: self.checksum, palette_crc: self.palette_crc, num_quotas: self.num_quotas, map_id: self.map_id,
            built_in: self.built_in, built_from_xml: self.built_from_xml, bounds: self.bounds, budget_max: self.budget_max, budget_spent: self.budget_spent,
            mcc_map_id: self.map_guid, quotas: self.quotas.iter().map(|q| (q.minimum, q.maximum, q.placed)).collect(), end_bit: self.end_bit,
            chunk_length: self.chunk_length, hash_ok: self.hash_ok, has_fsm: self.has_fsm,
        }
    }
}

/// Bit spans `[start, end)` of the header fields the encoder can replace; every
/// other header bit is copied verbatim from the source payload.
#[derive(Clone, Copy, Debug, Default)]
pub struct H4HeaderSpans {
    /// The 128 bits (u64 timestamp + u64 xuid) right before the CreatedBy / ModifiedBy name.
    pub created: (usize, usize),
    pub modified: (usize, usize),
    pub author: (usize, usize),
    pub editor: (usize, usize),
    pub title: (usize, usize),
    pub description: (usize, usize),
    /// The 32-bit `budget_spent` field.
    pub budget_spent: usize,
    /// The 8-bit `megalo-category-index` (right before the CreatedBy block).
    pub category: usize,
    /// The 6 x 32-bit `world-bounds` (followed by `maximum_budget` = +192 and
    /// `spent_budget` = +224).
    pub bounds: usize,
    /// The forge label table (9-bit count .. buffer bytes).
    pub labels: (usize, usize),
}

// ---------------------------------------------------------------------------------------------
// BLF container
// ---------------------------------------------------------------------------------------------

/// Walk the BLF chunks -> (payload bitstream, chunk major, chunk minor) of the `mvar` chunk.
pub fn blf_h4_mvar_chunk(data: &[u8]) -> Option<(&[u8], u16, u16)> {
    let mut pos = 0usize;
    while pos + 12 <= data.len() {
        let magic = &data[pos..pos + 4];
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 12 || pos + size > data.len() { break; }
        if magic == b"mvar" && size >= 36 {
            let major = u16::from_be_bytes([data[pos + 8], data[pos + 9]]);
            let minor = u16::from_be_bytes([data[pos + 10], data[pos + 11]]);
            return Some((&data[pos + 36..pos + size], major, minor));
        }
        pos += size;
    }
    None
}

/// The `mvar` chunk's stored payload length (BE u32 at chunk +32 = bytes the engine hashed).
pub fn mvar_chunk_length(data: &[u8]) -> Option<u32> {
    let mut pos = 0usize;
    while pos + 12 <= data.len() {
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 12 || pos + size > data.len() { return None; }
        if &data[pos..pos + 4] == b"mvar" && size >= 36 { return Some(u32::from_be_bytes([data[pos + 32], data[pos + 33], data[pos + 34], data[pos + 35]])); }
        pos += size;
    }
    None
}

/// `std::fs::read` + `_cmp` expansion (a game-saved variant can carry its `mvar`
/// chunk zlib-wrapped in a `_cmp` chunk, exactly like Reach's Empire.mvar; the Reach expander
/// inflates it in place and restamps `_eof`). Every reader / the writer works on this form, so
/// a `_cmp` source is SAVED in the plain layout (the one every shipped file uses).
pub fn read_h4_expanded(path: &Path) -> std::io::Result<Vec<u8>> { crate::mvar::read_expanded(path) }

/// Is this file a Halo 4 map variant (BLF with an `mvar` chunk of major version 50)?
/// Reads only the chunk headers; a Reach variant (v31) answers false.
pub fn is_h4_variant(path: &Path) -> bool {
    let Ok(data) = read_h4_expanded(path) else { return false };
    matches!(blf_h4_mvar_chunk(&data), Some((_, H4_MVAR_VERSION, _)))
}

/// Base map id of a Halo 4 variant without decoding the objects (None when not a v50 variant).
pub fn read_h4_map_id(path: &Path) -> Option<u32> {
    let data = read_h4_expanded(path).ok()?;
    let (payload, major, _) = blf_h4_mvar_chunk(&data)?;
    if major != H4_MVAR_VERSION { return None; }
    let mut r = Bits::new(payload);
    let (h, _) = read_content_header_spans(&mut r);
    Some(h.map_id)
}

// ---------------------------------------------------------------------------------------------
// bitstream decode
// ---------------------------------------------------------------------------------------------

/// Position axis bit widths: halo4.dll sub_18051AE50 with bit count 21. Differs from Reach's
/// `compute_axis_bits` (ceil instead of +0.9999, cap 0x400000 instead of 0x800000 - the cap is
/// what makes the Ravine / Forge Island bounds decode with 22 bits, not 23).
pub fn h4_axis_bits(bitcount: i32, ext: [f32; 3]) -> [u32; 3] {
    const MIN_UNIT: f32 = 0.008_333_333_8;
    let min_step = if bitcount > 16 { MIN_UNIT / (1i32 << (bitcount - 16)) as f32 } else { (1i32 << (16 - bitcount)) as f32 * MIN_UNIT };
    if min_step < 0.0001 { return [26, 26, 26]; }
    let step2 = min_step * 2.0;
    let mut out = [0u32; 3];
    for i in 0..3 {
        let mut v = (ext[i] / step2).ceil() as i64;
        if v > 0x40_0000 { v = 0x40_0000; }
        if v <= 0 { out[i] = 0; continue; }
        let hb = 63 - (v as u64).leading_zeros() as i64;
        let b = hb + if v & ((1i64 << hb) - 1) != 0 { 1 } else { 0 };
        out[i] = b.min(26) as u32;
    }
    out
}

/// Engine `read_quantized_real(min, max, bits, exact_midpoint, exact_endpoints)`.
pub fn dequantize_real(q: u32, min: f32, max: f32, bits: u32, exact_midpoint: bool, exact_endpoints: bool) -> f32 {
    let mut count = 1i64 << bits;
    if exact_midpoint { count -= 1; }
    let v = if exact_endpoints {
        if q == 0 { min }
        else if q as i64 == count - 1 { max }
        else {
            let step = (max - min) / (count - 2) as f32;
            min + step * ((q as f32 - 1.0) + 0.5)
        }
    } else {
        let step = (max - min) / count as f32;
        min + step * (q as f32 + 0.5)
    };
    if exact_midpoint && 2 * q as i64 == count - 1 { (min + max) * 0.5 } else { v }
}

impl H4PlacedObject {
    /// The record's scale field as the engine decodes it: `scale_q` dequantised the way
    /// `read_quantized_real(0, 10, 6, false, true)` does (q 7 -> 1.048, q 0 -> 0, q 63 -> 10).
    /// The engine copies it into the object but does NOT draw it (docs/halo4_mvar_layout.md
    /// section 11c) - a raw field kept for round-tripping, never a render input.
    pub fn scale(&self) -> f32 { h4_dequantize_scale(self.scale_q) }
    /// Quantise a scale the way the engine's encoder does (`write_quantized_real`,
    /// exact endpoints): 1.0 -> q 7 (the shipped default). Values are clamped into [0, 10].
    pub fn set_scale(&mut self, s: f32) { self.scale_q = h4_quantize_scale(s); }
    /// True when the record's scale is not the shipped default (q 7).
    pub fn is_scaled(&self) -> bool { self.scale_q != SCALE_Q_DEFAULT }
    /// Shape values in world units (1/256 units, see `shape_values`).
    pub fn shape_wu(&self) -> [f32; 4] { self.shape_values.map(|v| v as f32 / 256.0) }
}

/// The decoded content header (sub_18006A43C; H4EK `sub_1402D9FF0` names every
/// field). Activity is 2 bits in Halo 4 (Reach reads 3, which is the one-bit slip that makes the
/// Reach reader mis-decode these files).
#[derive(Clone, Debug, Default)]
struct H4ContentHeader {
    content_type: i8,
    file_size: u32,
    uid: u64,
    parent_uid: u64,
    root_uid: u64,
    game_id: u64,
    activity: u8,
    game_mode: u8,
    engine: u8,
    map_id: u32,
    category: i8,
    author: String,
    editor: String,
    created: (u64, u64),
    modified: (u64, u64),
    created_online: bool,
    modified_online: bool,
    title: String,
    description: String,
    hopper_id: Option<u16>,
}

/// The content header plus the bit spans of the editable fields (encoder input). Field order and
/// widths from H4EK `sub_1402D9FF0` (hold on all 388 shipped files, docs/halo4_mvar_layout.md
/// section 2): type 4, file-size 32, uid / parent-uid / root-uid / game-id 64 each, activity 2,
/// game-mode 3, game-engine-type 3, map-id 32, megalo-category-index 8 (signed), two history
/// blocks {timestamp 64, author-xuid 64, author-gamertag str16, author-flags 1}, name / description
/// (wide, 128), then the type / activity / engine specific tails.
fn read_content_header_spans(r: &mut Bits) -> (H4ContentHeader, H4HeaderSpans) {
    let mut sp = H4HeaderSpans::default();
    let mut h = H4ContentHeader::default();
    h.content_type = r.bits(4) as i8 - 1;
    h.file_size = r.u32();
    h.uid = r.u64();
    h.parent_uid = r.u64();
    h.root_uid = r.u64();
    h.game_id = r.u64();
    h.activity = r.bits(2) as u8;
    h.game_mode = r.bits(3) as u8;
    h.engine = r.bits(3) as u8;
    h.map_id = r.u32();
    sp.category = r.pos;
    h.category = r.bits(8) as u8 as i8;
    let author_block = |r: &mut Bits| { let ts = r.u64(); let xuid = r.u64(); let s = r.pos; let n = r.cstr(16); let e = r.pos; let online = r.flag(); (n, (s, e), (ts, xuid), online) };
    let (author, a_span, created, c_on) = author_block(r);
    let (editor, e_span, modified, m_on) = author_block(r);
    let ts = r.pos;
    let title = r.wstr(128);
    let ds = r.pos;
    let description = r.wstr(128);
    let de = r.pos;
    // type 3 / 4 (film) carry `seconds` (32), type 6 (game variant) `icon-index` (8); a map
    // variant (type 5) has neither. Then `hopper-id` (16) when activity == 2, then the campaign
    // (engine 3: campaign-id 8, difficulty 2, metagame-scoring 2, insertion-point 8, skull-flags
    // 32) or firefight (engine 4: difficulty 2, skull-flags 32) block.
    match h.content_type {
        3 | 4 => { r.bits(32); }
        6 => { r.bits(8); }
        _ => {}
    }
    if h.activity == 2 { h.hopper_id = Some(r.bits(16) as u16); }
    match h.engine {
        3 => { r.bits(8); r.bits(2); r.bits(2); r.bits(8); r.bits(32); }
        4 => { r.bits(2); r.bits(32); }
        _ => {}
    }
    sp.created = (a_span.0 - 128, a_span.0);
    sp.modified = (e_span.0 - 128, e_span.0);
    sp.author = a_span;
    sp.editor = e_span;
    sp.title = (ts, ds);
    sp.description = (ds, de);
    h.author = author;
    h.editor = editor;
    h.created = created;
    h.modified = modified;
    h.created_online = c_on;
    h.modified_online = m_on;
    h.title = title;
    h.description = description;
    (h, sp)
}

/// Decode the `mvar` chunk payload (the bitstream after the 36-byte chunk prefix).
pub fn parse_h4_payload(payload: &[u8], chunk_version: u16) -> Result<H4Variant> {
    if chunk_version != H4_MVAR_VERSION { bail!("mvar chunk v{chunk_version}, expected v{H4_MVAR_VERSION}"); }
    if payload.len() < H4_MVAR_PAYLOAD { bail!("mvar payload {} bytes, expected {}", payload.len(), H4_MVAR_PAYLOAD); }
    let mut r = Bits::new(payload);
    let (hdr, mut spans) = read_content_header_spans(&mut r);
    let H4ContentHeader { title: title_key, description: description_key, author, editor, map_id: header_map_id, activity, game_mode, engine, created, modified, .. } = hdr.clone();
    // `$key` titles / descriptions -> English text (literal strings pass through)
    let title = super::localization::display(&title_key);
    let description = super::localization::display(&description_key);
    let version = r.u8();
    // MCC writes body version 51 for user-saved / file-share variants: the same
    // layout plus a 16-byte map GUID after the labels (sub_1800B3BD8 reads it via
    // sub_1800F64D0 when version >= 51; a v50 file gets it synthesised from the map id and is
    // relabelled 51 in memory, which is why every save from the game comes out as 51)
    if version != 50 && version != 51 { bail!("variant body version {version}, expected 50 or 51"); }
    let checksum = r.u32(); // map-variant-checksum
    let palette_crc = r.u32(); // m_scenario_palette_crc
    let num_quotas = r.bits(9) as u16;
    if num_quotas as usize > H4_QUOTA_SLOTS { bail!("num_quotas {num_quotas} > 256"); }
    let map_id = r.u32();
    let built_in = r.flag();
    let built_from_xml = r.flag();
    spans.bounds = r.pos;
    let mut bounds = [0f32; 6];
    for b in bounds.iter_mut() { *b = r.f32(); }
    let budget_max = r.u32();
    spans.budget_spent = r.pos;
    let budget_spent = r.u32();
    // forge label table: 9-bit count, per string flag + 12-bit offset, 13-bit buffer size,
    // compressed flag (+ 13-bit compressed length, zlib stream after a 4-byte BE length)
    spans.labels.0 = r.pos;
    let nlabels = r.bits(9) as usize;
    let mut offsets: Vec<Option<usize>> = Vec::with_capacity(nlabels);
    for _ in 0..nlabels { offsets.push(if r.flag() { Some(r.bits(12) as usize) } else { None }); }
    let mut labels = Vec::with_capacity(nlabels);
    if nlabels > 0 {
        let buffer_size = r.bits(13) as usize;
        let buf: Vec<u8> = if r.flag() {
            let clen = r.bits(13) as usize;
            let raw: Vec<u8> = (0..clen).map(|_| r.u8()).collect();
            if raw.len() > 4 { miniz_oxide::inflate::decompress_to_vec_zlib(&raw[4..]).unwrap_or_default() } else { Vec::new() }
        } else {
            (0..buffer_size).map(|_| r.u8()).collect()
        };
        for o in offsets {
            labels.push(match o {
                Some(o) if o < buf.len() => buf[o..].cstr_at(0, buf.len() - o),
                _ => String::new(),
            });
        }
    }
    spans.labels.1 = r.pos;
    // version >= 51: the map GUID (16 raw memory bytes: u32 Data1, u16 Data2,
    // u16 Data3, 8 bytes Data4 - sub_1800F5DB0 byte-swaps each part before the MSB-first write,
    // so the stream bytes equal the in-memory GUID); v50 goes straight to the slots
    let map_guid = if version >= 51 { let mut g = [0u8; 16]; for b in g.iter_mut() { *b = r.u8(); } Some(g) } else { None };
    let ext = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]];
    let axis = h4_axis_bits(21, ext);
    let obj_start_bit = r.pos;
    let mut objects = Vec::new();
    for slot in 0..H4_SLOTS {
        let bit = r.pos;
        if !r.flag() { continue; }
        let flags = r.bits(2) as u8;
        let quota = r.index(8).map(|v| v as u8);
        let variant = r.index(5).map(|v| v as u8);
        // position: 1 flag, then the 3 quantized axes (bounds are always supplied, so the
        // engine never reads a bsp index here)
        let in_bounds = r.flag();
        let mut pos = [0f32; 3];
        for i in 0..3 {
            let q = r.bits(axis[i]) as f32;
            let step = ext[i] / (1u64 << axis[i]) as f32;
            pos[i] = bounds[2 * i] + q * step + step * 0.5;
        }
        let up_is_global = r.flag();
        let up_quant = if up_is_global { 0 } else { r.bits(20) as u32 };
        let forward_angle_q = r.bits(14) as u32;
        let (fwd, up) = crate::mvar::decode_orientation(up_is_global, up_quant, forward_angle_q);
        let parent = r.bits(10) as i16 - 1;
        let scale_q = r.bits(6) as u8;
        let locked = r.flag();
        // multiplayer object properties (sub_1800B7410)
        let shape = H4Shape::from_bits(r.bits(2));
        let mut shape_values = [0u16; 4];
        for v in shape_values.iter_mut().take(shape.value_count()) { *v = r.u16(); }
        let object_type = r.bits(6) as u8;
        let placement = r.bits(10) as u16;
        let mut team = -1i8;
        let mut spawn_time = 0u8;
        let mut color = None;
        let mut spawn_sequence = 0i8;
        let mut user_data = 0i8;
        let mut labels4 = [None; 4];
        let type_data = match object_type {
            32 => {
                team = r.bits(4) as i8 - 1;
                let weapon = r.u8();
                let timing = r.u16();
                H4TypeData::InitialOrdnance { team, weapon, timing }
            }
            33 => {
                let mut w = [0u8; 8];
                for x in w.iter_mut() { *x = r.u8(); }
                H4TypeData::RandomOrdnance(w)
            }
            34 => {
                let lead = r.u8();
                let mut w = [0u8; 8];
                for x in w.iter_mut() { *x = r.u8(); }
                H4TypeData::ObjectiveOrdnance { lead, weapons: w }
            }
            35 => H4TypeData::Empty,
            t => {
                team = r.bits(4) as i8 - 1;
                spawn_time = r.u8();
                color = r.index(3).map(|v| v as u8);
                spawn_sequence = r.u8() as i8;
                user_data = r.i8();
                for l in labels4.iter_mut() { *l = r.index(8).map(|v| v as u8); }
                match t {
                    1 | 12 => H4TypeData::Byte(r.u8()),
                    20 => H4TypeData::Index(r.u8() as i16 - 1),
                    31 => H4TypeData::TraitZone(r.bits(5) as u8),
                    _ => { let a = r.bits(5) as u8; let b = r.bits(5) as u8; H4TypeData::Pair(a, b) }
                }
            }
        };
        objects.push(H4PlacedObject {
            slot: slot as u16, bit, flags, quota, variant, in_bounds, pos, up_is_global, up_quant, forward_angle_q, fwd, up,
            parent, scale_q, locked, shape, shape_values, object_type, placement, team, spawn_time, color, spawn_sequence,
            user_data, labels: labels4, type_data,
        });
    }
    let obj_end_bit = r.pos;
    let mut quotas = Vec::with_capacity(num_quotas as usize);
    for _ in 0..num_quotas {
        quotas.push(H4Quota { minimum: r.u8(), maximum: r.u8(), placed: r.u8() });
    }
    let quota_end_bit = r.pos;
    if quota_end_bit > payload.len() * 8 { bail!("variant bitstream overran the payload"); }
    // the four player-trait-set records are not decoded, only measured (sub_18006C408 layout)
    for _ in 0..4 { skip_trait_set(&mut r); }
    let end_bit = r.pos;
    if end_bit > H4_MVAR_BITSTREAM_BYTES * 8 { bail!("trait records overran the 28 672-byte bitstream"); }
    Ok(H4Variant {
        chunk_version, title, description, title_key, description_key, author, editor, created, modified,
        created_online: hdr.created_online, modified_online: hdr.modified_online, content_type: hdr.content_type, file_size: hdr.file_size,
        uid: hdr.uid, parent_uid: hdr.parent_uid, root_uid: hdr.root_uid, game_id: hdr.game_id,
        activity, game_mode, engine, header_map_id, category: hdr.category, hopper_id: hdr.hopper_id, version, map_guid, checksum, palette_crc,
        num_quotas, map_id, built_in, built_from_xml, bounds, budget_max, budget_spent, labels, objects, quotas,
        obj_start_bit, obj_end_bit, quota_end_bit, end_bit, spans,
        chunk_length: 0, hash_ok: false, has_fsm: false,
    })
}

/// What the variant browser shows for a Halo 4 file: resolved title / description,
/// authors, base map id and the placed object count.
#[derive(Clone, Debug)]
pub struct H4VariantSummary {
    pub title: String,
    pub description: String,
    pub author: String,
    pub editor: String,
    pub map_id: u32,
    pub objects: usize,
}

/// Summary of a Halo 4 variant, None for anything else (a Reach file, a non-BLF file).
/// Reads the file once; a v50 chunk is decoded in full (the object count needs the slots).
pub fn read_h4_summary(path: &Path) -> Option<H4VariantSummary> {
    let data = read_h4_expanded(path).ok()?;
    let (payload, major, _) = blf_h4_mvar_chunk(&data)?;
    if major != H4_MVAR_VERSION { return None; }
    let v = parse_h4_payload(payload, major).ok()?;
    Some(H4VariantSummary { title: v.title, description: v.description, author: v.author, editor: v.editor, map_id: v.map_id, objects: v.objects.len() })
}

/// Parse a Halo 4 `.mvar` file.
pub fn parse_h4_variant(path: &Path) -> Result<H4Variant> {
    let data = read_h4_expanded(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let (payload, major, _minor) = blf_h4_mvar_chunk(&data).ok_or_else(|| anyhow!("{}: no mvar chunk", path.display()))?;
    let mut v = parse_h4_payload(payload, major)?;
    // the chunk header facts
    if let Some(len) = mvar_chunk_length(&data) {
        v.chunk_length = len;
        v.hash_ok = detect_hash_kind(&data).is_some();
    }
    v.has_fsm = { let mut pos = 0usize; let mut f = false; while pos + 12 <= data.len() { let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize; if size < 12 || pos + size > data.len() { break; } if &data[pos..pos + 4] == b"_fsm" { f = true; } pos += size; } f };
    // The game never writes a record without a variant index: in MCC such a record spawns the
    // NEXT palette entry's object, or nothing. Re-saving through HMS repairs it (the encoder
    // always writes an explicit index).
    let none = v.objects.iter().filter(|o| o.quota.is_some() && o.variant.is_none()).count();
    if none > 0 {
        log::warn!("{}: {none} of {} objects store NO variant index (MCC spawns the wrong object for those) - Save / Save As from HMS repairs the file", path.display(), v.objects.len());
    }
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// forge palette (scnr +0x2C4) and placement resolution
// ---------------------------------------------------------------------------------------------

pub const OFF_SCNR_FORGE_PALETTE: usize = 0x2C4;
/// Palette element: sid @0, i32 @4, entries block @8 (count @8, ptr @0xC), pad @0x10.
pub const FORGE_PALETTE_ELEM: usize = 0x14;
/// Entry element: sid @0, variants block @4, i32 max @0x10, i32 price @0x14, i32 @0x18.
pub const FORGE_ENTRY_ELEM: usize = 0x1C;
/// Variant element: sid @0, object tag ref @0x04 (fourcc @4, datum @0x10), variant sid @0x14,
/// 4 x 12-byte tail (ZeroHour's `var+0x10 TAG INDEX` + the fourcc bytes at +4).
pub const FORGE_VARIANT_ELEM: usize = 0x48;

#[derive(Clone, Debug)]
pub struct H4PaletteVariant {
    pub name: String,
    pub variant_name: String,
    /// Object tag (obje class: scen / bloc / vehi / weap / ...), None for a null reference.
    pub tag: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct H4PaletteEntry {
    /// Flat quota index (the variant's `quota` field).
    pub quota: u8,
    pub palette: String,
    pub name: String,
    pub max: i32,
    pub price: i32,
    pub variants: Vec<H4PaletteVariant>,
}

/// The base map's forge palette flattened in scnr order: quota index i = entry i.
/// (Every shipped variant's quota indices, `placed_on_map` counts and `budget_spent` agree
/// with this order and these prices.)
pub fn load_forge_palette(c: &H4Cache) -> Vec<H4PaletteEntry> {
    let d = c.data();
    let mut out = Vec::new();
    let Some(&scnr) = c.find_tags(b"scnr").first() else { return out };
    let Some(sm) = c.tag_meta(scnr) else { return out };
    let Some((np, po)) = c.block(sm + OFF_SCNR_FORGE_PALETTE) else { return out };
    for i in 0..np.min(64) {
        let pe = po + i * FORGE_PALETTE_ELEM;
        let palette = c.sid(d.u32_at(pe));
        let Some((ne, eo)) = c.block(pe + 8) else { continue };
        for k in 0..ne.min(256) {
            let e = eo + k * FORGE_ENTRY_ELEM;
            let mut variants = Vec::new();
            if let Some((nv, vo)) = c.block(e + 4) {
                for v in 0..nv.min(32) {
                    let va = vo + v * FORGE_VARIANT_ELEM;
                    variants.push(H4PaletteVariant {
                        name: c.sid(d.u32_at(va)),
                        variant_name: c.sid(d.u32_at(va + 0x14)),
                        tag: c.tag_ref(va + 0x04).map(|(_, t)| t),
                    });
                }
            }
            if out.len() >= 256 { return out; }
            out.push(H4PaletteEntry { quota: out.len() as u8, palette: palette.clone(), name: c.sid(d.u32_at(e)), max: d.i32_at(e + 0x10), price: d.i32_at(e + 0x14), variants });
        }
    }
    out
}

/// Counters from `variant_placements`.
#[derive(Clone, Debug, Default)]
pub struct H4VariantStats {
    pub objects: usize,
    pub placed: usize,
    pub skipped_no_quota: usize,
    pub skipped_quota_range: usize,
    pub skipped_variant_range: usize,
    pub skipped_null_tag: usize,
    pub skipped_no_model: usize,
}

/// Resolve every variant object to a scenario-style placement (palette tag + world basis) the
/// h4 object path can draw. Objects without a drawable model (marker types like spawn zones
/// whose tag has no `mode`) are counted, not placed. `variant` None = the entry's variant 0.
/// #h4-veh The index of the hlmt model variant NAMED `name` on `obje_tag` (the palette
/// variant's model variant, e.g. the Warthog's `rocket`); `None` when the name is empty or absent.
fn hlmt_variant_by_name(c: &H4Cache, obje_tag: usize, name: &str) -> Option<usize> {
    if name.is_empty() { return None; }
    let hlmt = super::objects::hlmt_of(c, obje_tag)?;
    super::objects::hlmt_variant_sids(c, hlmt).iter().position(|&s| c.sid(s) == name)
}

pub fn variant_placements(c: &H4Cache, v: &H4Variant, palette: &[H4PaletteEntry]) -> (Vec<H4Placement>, H4VariantStats) {
    let mut out = Vec::new();
    let mut st = H4VariantStats { objects: v.objects.len(), ..Default::default() };
    for o in &v.objects {
        let Some(q) = o.quota else { st.skipped_no_quota += 1; continue };
        let Some(entry) = palette.get(q as usize) else { st.skipped_quota_range += 1; continue };
        let vi = o.variant.unwrap_or(0) as usize;
        let Some(pv) = entry.variants.get(vi) else { st.skipped_variant_range += 1; continue };
        let Some(tag) = pv.tag else { st.skipped_null_tag += 1; continue };
        if model_of(c, tag).is_none() { st.skipped_no_model += 1; continue; }
        let class = c.tag_class(tag).unwrap_or(*b"scen");
        let fwd = Vec3::from(o.fwd);
        let up = Vec3::from(o.up);
        out.push(H4Placement {
            class,
            palette_tag: tag,
            name: format!("{}:{}", entry.name, pv.name),
            pos: o.pos,
            rot: [0.0; 3],
            scale: 1.0,
            flags: o.placement as u32,
            basis: Some((fwd, up)),
            // #h4-veh a map-variant placement carries the multiplayer properties the engine
            // resolves its primary change colour from, and the palette variant's hlmt variant.
            mp: Some((o.team, o.color)),
            model_variant: hlmt_variant_by_name(c, tag, &pv.variant_name).or_else(|| super::objects::object_variant_index(c, tag, 0)),
        });
        st.placed += 1;
    }
    (out, st)
}

/// Human-readable listing (HMS_MVAR_LIST / --dump-h4-mvar).
pub fn describe(v: &H4Variant, palette: Option<&[H4PaletteEntry]>) -> String {
    let mut s = String::new();
    // resolved text first, the stored `$key` in brackets when the header holds one
    let key_note = |shown: &str, raw: &str| if shown != raw { format!(" [{raw}]") } else { String::new() };
    s.push_str(&format!("h4 mvar chunk v{} body v{}{}: '{}'{} by {} (edited by {}) map id {} quotas {} bounds {:?} budget {}/{} labels {:?}\n  '{}'{}\n  objects {} (slots used {}..{})\n",
        v.chunk_version, v.version, v.map_guid.map(|g| format!(" map guid {}", g.iter().map(|b| format!("{b:02x}")).collect::<String>())).unwrap_or_default(), v.title, key_note(&v.title, &v.title_key), v.author, v.editor, v.map_id, v.num_quotas, v.bounds, v.budget_spent, v.budget_max, v.labels,
        v.description, key_note(&v.description, &v.description_key),
        v.objects.len(), v.objects.first().map_or(0, |o| o.slot), v.objects.last().map_or(0, |o| o.slot)));
    for o in &v.objects {
        let name = match (palette, o.quota) {
            (Some(p), Some(q)) => p.get(q as usize).map(|e| {
                let vn = e.variants.get(o.variant.unwrap_or(0) as usize).map(|x| x.name.as_str()).unwrap_or("?");
                format!("{}/{}", e.name, vn)
            }).unwrap_or_else(|| format!("quota {q} OUT OF RANGE")),
            (_, Some(q)) => format!("quota {q} variant {:?}", o.variant),
            _ => "empty".to_string(),
        };
        // every stored field (flags / in_bounds / parent / scale / lock / colour /
        // user data too) so an HMS-written record can be diffed against a game-written one
        s.push_str(&format!("  [{:3}] {:<44} type {:2} pos ({:8.2},{:8.2},{:8.2}) fwd ({:5.2},{:5.2},{:5.2}) up ({:5.2},{:5.2},{:5.2}) team {:2} phys {} place {:#05x} spawn {:3}s order {:3} shape {:?} {:?} labels {:?} {:?} flags {} inb {} parent {} scale_q {} locked {} color {:?} user {} q {:?}/{:?}\n",
            o.slot, name, o.object_type, o.pos[0], o.pos[1], o.pos[2], o.fwd[0], o.fwd[1], o.fwd[2], o.up[0], o.up[1], o.up[2],
            o.team, o.physics(), o.placement, o.spawn_time, o.spawn_sequence, o.shape, &o.shape_values[..o.shape.value_count()], o.label_names(v), o.type_data,
            o.flags, o.in_bounds as u8, o.parent, o.scale_q, o.locked as u8, o.color, o.user_data, o.quota, o.variant));
    }
    s
}

/// The MCC halo4 variant folders on this machine (tests skip when absent).
pub fn variant_dirs() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(v) = std::env::var("HMS_H4_MVAR_DIRS") {
        for p in v.split(';').filter(|s| !s.is_empty()) { let p = std::path::PathBuf::from(p); if p.is_dir() { out.push(p); } }
    }
    if let Some(maps) = super::cache::maps_dir() {
        if let Some(h4) = maps.parent() {
            for sub in ["map_variants", "hopper_map_variants"] {
                let p = h4.join(sub);
                if p.is_dir() && !out.contains(&p) { out.push(p); }
            }
        }
    }
    out
}

/// Every `.mvar` under the halo4 variant folders (sorted).
pub fn all_variant_files() -> Vec<std::path::PathBuf> {
    let mut v: Vec<std::path::PathBuf> = variant_dirs().iter().flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar"))).collect();
    v.sort();
    v
}

/// Every Halo 4 variant SAVED BY THE GAME on this machine: `MCC/LocalFiles/<xuid>/
/// Halo4/Map/*.mvar` under each MCC profile (Windows users + the Proton prefix, mapcat's roots)
/// plus the `HMS_H4_MVAR_DIRS` folders. Sorted.
pub fn user_variant_files() -> Vec<std::path::PathBuf> {
    let mut out: Vec<std::path::PathBuf> = Vec::new();
    let mut push_dir = |d: &Path| {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) && !out.contains(&p) { out.push(p); }
        }
    };
    for root in crate::mapcat::mcc_localfiles_roots() {
        for e in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            let d = e.path().join("Halo4").join("Map");
            if d.is_dir() { push_dir(&d); }
        }
    }
    if let Ok(v) = std::env::var("HMS_H4_MVAR_DIRS") {
        for p in v.split(';').filter(|s| !s.is_empty()) { let p = std::path::PathBuf::from(p); if p.is_dir() { push_dir(&p); } }
    }
    out.sort();
    out
}

/// `maps/info/<stem>.mapinfo` -> map id (`levl` chunk +12, big-endian; same reader as mapcat).
pub fn map_id_of_cache(map_path: &Path) -> Option<u32> { crate::mapcat::read_map_id(map_path) }

// ---------------------------------------------------------------------------------------------
// map GUID
// ---------------------------------------------------------------------------------------------

/// halo4.dll `sub_180072E34`: map id -> index into the engine's level table (None = unknown).
/// Ported verbatim (the whole table; 51 ids).
pub fn h4_map_index(map_id: u32) -> Option<u32> {
    Some(match map_id {
        10080 => 118, 10085 => 116, 10091 => 121, 10102 => 129, 10200 => 119, 10202 => 122,
        10210 => 114, 10225 => 115, 10226 => 117, 10245 => 126, 10252 => 123, 10255 => 125,
        10256 => 124, 10261 => 120, 11061 => 152, 11071 => 144, 11081 => 145, 11084 => 143,
        11101 => 141, 11111 => 140, 11141 => 142, 11151 => 146, 11161 => 147, 11200 => 149,
        11210 => 151, 11230 => 150, 11240 => 153, 11250 => 154, 11302 => 148, 12000 => 103,
        12010 => 104, 12020 => 105, 12030 => 106, 12040 => 108, 12060 => 107, 12070 => 109,
        12080 => 110, 12090 => 111, 12100 => 112, 13110 => 131, 13120 => 136, 13130 => 134,
        13131 => 132, 13140 => 135, 13160 => 133, 13301 => 128, 13302 => 130, 14100 => 127,
        15000 => 137, 15010 => 138,
        _ => return None,
    })
}

/// The map GUID a version 51 variant of `map_id` carries (`sub_180076294`: `u64 lo = index |
/// 0x8888_0000_0000_0000, u64 hi = 0`, as raw little-endian memory bytes). None when the map
/// id is not in the engine table (the engine then stores its "invalid" constant).
pub fn h4_map_guid(map_id: u32) -> Option<[u8; 16]> {
    let lo = h4_map_index(map_id)? as u64 | 0x8888_0000_0000_0000;
    let mut g = [0u8; 16];
    g[..8].copy_from_slice(&lo.to_le_bytes());
    Some(g)
}

// ---------------------------------------------------------------------------------------------
// encoder
//
// Mirrors the retail halo4.dll ENCODER (IDA on the shipped DLL, image base 0x180000000):
//   c_map_variant::encode              sub_1800B379C   version u8, checksum/752, 9-bit quota count,
//                                                      map id, 2 flags, 6 bounds, budgets, labels,
//                                                      651 x object, quotas, 4 trait sets
//   content header encode              sub_180069304   type+1 (4), file length, 4 x u64, activity 2,
//                                                      game mode 3, engine 3, map id, category 8,
//                                                      2 x author (ts, xuid, str16, flag), title/desc
//                                                      (wide, max 128, terminator only when shorter)
//   s_variant_object_datum::encode     sub_1800B7A68   present = flags&1, flags 2, quota (flag+8),
//                                                      variant (flag+5), position, axes, parent+1
//                                                      (10), real (0..10, 6 bits, exact endpoints),
//                                                      flag, properties
//   position encode                    sub_18051721C / sub_180517034: flag ALWAYS 1 when bounds
//                                                      are supplied (the point is clamped into
//                                                      them), q = trunc((clamp(p)-min) /
//                                                      (ext / 2^bits)) clamped to 2^bits-1,
//                                                      bits from sub_18051AE50 (h4_axis_bits)
//   axes encode                        sub_1800B8B34   up within 1e-4 of +Z -> flag 1; else flag 0
//                                                      + 20-bit cube-face up (sub_1800B9164:
//                                                      q = trunc((c+1)/0.0048076925) clamp 415,
//                                                      value = w + face*174762 + 417*u);
//                                                      forward angle atan2(f.cr, f.ref) over the
//                                                      DEQUANTISED up (sub_1800F5968), 14 bits:
//                                                      q = trunc((a+pi)/0.00038349521) clamp 16383
//   write_quantized_real               sub_1800F5A7C   exact_endpoints: 0 / count-1 at the ends,
//                                                      else trunc((v-min)/((max-min)/(count-2)))+1
//   mp object properties encode        sub_1800B6ED8   the exact mirror of the decoder
//   boundary shape encode              sub_180691208   2 bits + 16-bit values (last = abs())
//   label table encode                 sub_1800B8E3C   13-bit size; zlib (level 9) only when the
//                                                      buffer is >= 128 bytes, else raw bytes
//   quota encode                       sub_1800B1CF0   3 x u8
//   chunk writer                       sub_1802380D4   zero 0x7000 bytes, encode, flush, length =
//                                                      ceil(bits/8) (BE u32 at +32), then
//                                                      SHA-1(LE u32 length || payload[..length])
//                                                      into +12 (plain BCrypt SHA1, NO salt)
//   chunk reader                       sub_180238258   recomputes that SHA-1 but only REJECTS a
//                                                      mismatch when byte_184967B01 is set - a
//                                                      zero-initialised .bss flag no code writes,
//                                                      so retail never enforces it (HMS still
//                                                      writes the correct hash)
// Every one of the 388 shipped variants re-encodes BYTE-IDENTICAL (whole file: BLF
// wrapper, SHA-1, length, payload) from its decoded object list with no edits, and the stored
// length equals ceil(end_bit / 8) with the trait records measured by `skip_trait_set`.
// ---------------------------------------------------------------------------------------------

/// `slot` value of a NEW object: the encoder puts it into the lowest free slot.
pub const NEW_SLOT: u16 = u16::MAX;

/// MSB-first bit writer, the exact mirror of `Bits`.
struct BitWriter {
    buf: Vec<u8>,
    pos: usize,
}

impl BitWriter {
    fn new(bytes: usize) -> Self { Self { buf: vec![0u8; bytes], pos: 0 } }
    fn bits(&mut self, n: u32, v: u64) {
        for i in (0..n).rev() {
            let byte = self.pos >> 3;
            if byte >= self.buf.len() { self.buf.push(0); }
            if (v >> i) & 1 != 0 { self.buf[byte] |= 1 << (7 - (self.pos & 7)); }
            self.pos += 1;
        }
    }
    fn flag(&mut self, v: bool) { self.bits(1, v as u64); }
    fn u8(&mut self, v: u8) { self.bits(8, v as u64); }
    fn u16(&mut self, v: u16) { self.bits(16, v as u64); }
    fn u32(&mut self, v: u32) { self.bits(32, v as u64); }
    fn f32(&mut self, v: f32) { self.u32(v.to_bits()); }
    /// `write_index`: a set flag means "none".
    fn index(&mut self, v: Option<u32>, bits: u32) {
        match v { None => self.flag(true), Some(x) => { self.flag(false); self.bits(bits, x as u64); } }
    }
    /// sub_1800F54BC: bytes up to and including the terminator, never more than `max`.
    fn cstr(&mut self, s: &str, max: usize) {
        let b = s.as_bytes();
        let n = b.len().min(max);
        for &c in &b[..n] { self.u8(c); }
        if n < max { self.u8(0); }
    }
    /// sub_1800F55D4: UTF-16 units up to and including the terminator, never more than `max`.
    fn wstr(&mut self, s: &str, max: usize) {
        let u: Vec<u16> = s.encode_utf16().collect();
        let n = u.len().min(max);
        for &c in &u[..n] { self.u16(c); }
        if n < max { self.u16(0); }
    }
    /// Copy `n` raw bits of `src` starting at bit `from` (bits past the end read as 0).
    fn copy_bits(&mut self, src: &[u8], from: usize, n: usize) {
        for k in 0..n {
            let sp = from + k;
            let bit = src.get(sp >> 3).map_or(0, |b| (b >> (7 - (sp & 7))) & 1);
            self.bits(1, bit as u64);
        }
    }
}

/// Engine position quantiser (sub_180517034 inner loop): clamp into [min, max], then
/// `trunc((v - min) / (ext / 2^bits))` clamped to `2^bits - 1`. All f32 like the engine.
pub fn h4_quantize_pos(v: f32, min: f32, max: f32, bits: u32) -> u32 {
    let mut c = v;
    if c - min < 0.0 { c = min; }
    if c - max >= 0.0 { c = max; }
    let count = (1u32 << bits) as f32;
    let step = (max - min) / count;
    let q = ((c - min) / step) as i64; // cvttss2si: truncation
    q.clamp(0, (1i64 << bits) - 1) as u32
}

/// sub_1800B9164: the engine's 20-bit cube-face unit-vector quantiser (f32, truncating). Face
/// choice: |x| <= |y| -> (|y| > |z| ? y-face : z-face), else (|x| > |z| ? x-face : z-face);
/// positive axis -> face 0/1/2, negative -> 3/4/5; the two other components / |dominant| ->
/// u, w in [-1, 1] -> trunc((c + 1) / (2/416)) clamped to 415; value = w + face*174762 + 417*u.
pub fn h4_quantize_unit_vector(v: [f32; 3]) -> u32 {
    const STEP: f32 = 0.004_807_692_5;
    const APAMC: u32 = 174_762; // dword_180ED80A0 = unit_vector_encoding_constants(20).0
    let (x, y, z) = (v[0], v[1], v[2]);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    let (face, u, w) = if ax <= ay {
        if ay > az { (if y > 0.0 { 1 } else { 4 }, x / ay, z / ay) } else { (if z > 0.0 { 2 } else { 5 }, x / az, y / az) }
    } else if ax > az {
        (if x > 0.0 { 0 } else { 3 }, y / ax, z / ax)
    } else {
        (if z > 0.0 { 2 } else { 5 }, x / az, y / az)
    };
    let q = |c: f32| (((c - -1.0) / STEP) as i32).clamp(0, 415) as u32;
    q(w) + face * APAMC + 417 * q(u)
}

/// sub_1800B8B34 + sub_1800F5968 + sub_1800B9274: engine-exact axes encoding ->
/// (up_is_global, up_quant, forward_angle_q). The yaw is measured in the reference frame the
/// decoder rebuilds from the DEQUANTISED up (so decode(encode(x)) is a fixed point).
pub fn h4_encode_axes(fwd: [f32; 3], up: [f32; 3]) -> (bool, u32, u32) {
    const EPS: f32 = 0.000_099_999_997;
    let (global, up_quant, up_used) = if (up[0] - 0.0).abs() < EPS && (up[1] - 0.0).abs() < EPS && (up[2] - 1.0).abs() < EPS {
        (true, 0u32, [0.0f64, 0.0, 1.0])
    } else {
        let q = h4_quantize_unit_vector(up);
        // the decoder's own f64 dequantisation, so a tie in the tangent choice below
        // (|up.x| == |up.y| happens on symmetric up vectors) resolves exactly as it will
        (false, q, crate::mvar::dequantize_unit_vector3d(q as i64, 20))
    };
    // sub_1800F583C: reference tangent + its cross with up, then atan2(f.cr, f.ref)
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm = |v: [f64; 3]| { let m = dot(v, v).sqrt(); if m > 1e-12 { [v[0] / m, v[1] / m, v[2] / m] } else { v } };
    let refv = norm(if dot(up_used, [0.0, 1.0, 0.0]).abs() <= dot(up_used, [1.0, 0.0, 0.0]).abs() {
        cross([0.0, 1.0, 0.0], up_used)
    } else {
        cross(up_used, [1.0, 0.0, 0.0])
    });
    let crv = norm(cross(up_used, refv));
    let f = [fwd[0] as f64, fwd[1] as f64, fwd[2] as f64];
    let angle = (dot(f, crv) as f32).atan2(dot(f, refv) as f32);
    let fq = (((angle - -std::f32::consts::PI) / 0.000_383_495_21) as i32).clamp(0, 0x3FFF) as u32;
    (global, up_quant, fq)
}

/// The shipped default scale quantum: `write_quantized_real(1.0, 0, 10, 6)` = 7.
pub const SCALE_Q_DEFAULT: u8 = 7;
/// The object scale the engine reads back from quantum `q` (sub_1800F6194 with
/// min 0, max 10, 6 bits, exact endpoints): 0 -> 0, 63 -> 10, else `(q - 0.5) * 10 / 62`.
pub fn h4_dequantize_scale(q: u8) -> f32 { dequantize_real((q & 0x3F) as u32, 0.0, 10.0, 6, false, true) }
/// Quantise an object scale for the record (clamped into [0, 10]).
pub fn h4_quantize_scale(s: f32) -> u8 { h4_quantize_real6(s.clamp(0.0, 10.0)) }

/// Engine `write_quantized_real` (sub_1800F5A7C) for the exact-endpoints form used by the
/// object's 6-bit scale over [0, 10].
pub fn h4_quantize_real6(v: f32) -> u8 {
    let (min, max, count) = (0.0f32, 10.0f32, 1i32 << 6);
    if v == min { return 0; }
    if v == max { return (count - 1) as u8; }
    let q = ((v - min) / ((max - min) / (count - 2) as f32)) as i32 + 1;
    q.clamp(1, count - 2) as u8
}

impl H4TypeData {
    /// The type-data variant the encoder expects for an object type.
    pub fn default_for(object_type: u8) -> H4TypeData {
        match object_type {
            1 | 12 => H4TypeData::Byte(0),
            20 => H4TypeData::Index(-1),
            31 => H4TypeData::TraitZone(0),
            32 => H4TypeData::InitialOrdnance { team: -1, weapon: 0, timing: 0 },
            33 => H4TypeData::RandomOrdnance([255; 8]),
            34 => H4TypeData::ObjectiveOrdnance { lead: 0, weapons: [255; 8] },
            35 => H4TypeData::Empty,
            _ => H4TypeData::Pair(0, 0),
        }
    }
    fn matches(&self, object_type: u8) -> bool {
        matches!((object_type, self),
            (1 | 12, H4TypeData::Byte(_)) | (20, H4TypeData::Index(_)) | (31, H4TypeData::TraitZone(_))
            | (32, H4TypeData::InitialOrdnance { .. }) | (33, H4TypeData::RandomOrdnance(_))
            | (34, H4TypeData::ObjectiveOrdnance { .. }) | (35, H4TypeData::Empty))
            || (!matches!(object_type, 1 | 12 | 20 | 31 | 32 | 33 | 34 | 35) && matches!(self, H4TypeData::Pair(..)))
    }
}

impl H4PlacedObject {
    /// A NEW object for `save_objects`: palette `quota` / `variant`, position, orientation
    /// (engine-quantised at once so `pos`/`fwd`/`up` already hold what the file will decode
    /// to) and the defaults of a player-placed Halo 4 object (flags 1, placement 12 =
    /// symmetric+asymmetric with NORMAL physics - shipped Forge structures use 0xCC = phased,
    /// set `placement` for that; team 8 = neutral like 1123 of 1169 shipped structures; scale
    /// q 7 = 1.0 quantised like 99.9 percent of shipped objects; no shape, no labels).
    ///
    /// `variant` is ALWAYS stored explicitly (`None` -> `Some(0)`): the game
    /// writes an index on every record (0 of 146 647 shipped objects and 0 of the game-saved
    /// user files carry NONE), and its loader has no NONE case - `c_map_variant::copy_and_validate`
    /// (halo4.dll sub_1800B254C) clamps the byte to `min(v, variant_count)` and `create_object`
    /// (sub_1800B5EE8 / sub_1800B6184) indexes the entry's variant block with it UNCHECKED, so a
    /// NONE (0xFF) record spawns the NEXT palette entry's first variant (Magnum -> Assault Rifle,
    /// Scorpion -> Mantis) or a non-object tag (structures), never the object HMS placed.
    pub fn new(quota: u8, variant: Option<u8>, object_type: u8, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3]) -> Self {
        let mut o = H4PlacedObject {
            slot: NEW_SLOT, bit: 0, flags: 1, quota: Some(quota), variant: Some(variant.unwrap_or(0)), in_bounds: true, pos,
            up_is_global: true, up_quant: 0, forward_angle_q: 0, fwd, up, parent: -1, scale_q: SCALE_Q_DEFAULT, locked: false,
            shape: H4Shape::None, shape_values: [0; 4], object_type, placement: 12, team: 8, spawn_time: 0,
            color: None, spawn_sequence: 0, user_data: 0, labels: [None; 4], type_data: H4TypeData::default_for(object_type),
        };
        o.set_orientation(fwd, up);
        o
    }
    /// Re-quantise the orientation the way the engine does and store the DECODED basis, so
    /// what the caller sees equals what the saved file decodes to.
    pub fn set_orientation(&mut self, fwd: [f32; 3], up: [f32; 3]) {
        let (g, uq, fq) = h4_encode_axes(fwd, up);
        self.up_is_global = g;
        self.up_quant = uq;
        self.forward_angle_q = fq;
        let (f, u) = crate::mvar::decode_orientation(g, uq, fq);
        self.fwd = f;
        self.up = u;
    }
    /// The position the saved file will decode to for `pos` under these variant bounds.
    pub fn quantized_pos(&self, v: &H4Variant) -> [f32; 3] {
        let (bounds, ext, axis) = variant_axes(v);
        let mut p = [0f32; 3];
        for i in 0..3 {
            let q = h4_quantize_pos(self.pos[i], bounds[2 * i], bounds[2 * i + 1], axis[i]) as f32;
            let step = ext[i] / (1u64 << axis[i]) as f32;
            p[i] = bounds[2 * i] + q * step + step * 0.5;
        }
        p
    }
}

fn variant_axes(v: &H4Variant) -> ([f32; 6], [f32; 3], [u32; 3]) {
    let ext = [v.bounds[1] - v.bounds[0], v.bounds[3] - v.bounds[2], v.bounds[5] - v.bounds[4]];
    (v.bounds, ext, h4_axis_bits(21, ext))
}

/// Skip one player-trait-set record (sub_18006C408: health, weapons, movement, appearance,
/// sensors). Optional 16-bit reals are flag-prefixed; everything else is fixed width. A default
/// record is 291 bits (13+8+2, 22+2+5+2+24+40, 4+4+2+8, 1+3+2+2+3+50+1+8+64+2, 2+3+14).
fn skip_trait_set(r: &mut Bits) {
    let optional_reals = |r: &mut Bits, n: usize| { for _ in 0..n { if r.flag() { r.bits(16); } } };
    optional_reals(r, 13); r.bits(8); r.bits(2);                                  // health   sub_1800705A8
    optional_reals(r, 22); r.bits(2); r.bits(5); r.bits(2); r.bits(24); r.bits(40); // weapons  sub_18007072C
    optional_reals(r, 4); r.bits(4); r.bits(2); r.bits(8);                         // movement sub_180070968
    optional_reals(r, 1); r.bits(3); r.bits(2); r.bits(2); r.bits(3);              // appearance sub_180070B28
    r.bits(25); r.bits(25); r.bits(1); r.bits(8); r.bits(32); r.bits(32); r.bits(2);
    optional_reals(r, 2); r.bits(3); r.bits(14);                                   // sensors  sub_180070D44
}

/// Write one present object slot - the exact mirror of the slot decode in `parse_h4_payload`
/// and of sub_1800B7A68 / sub_1800B6ED8. `parent` must already be the FINAL slot value.
fn write_slot(w: &mut BitWriter, o: &H4PlacedObject, parent: i16, bounds: [f32; 6], axis: [u32; 3]) -> Result<()> {
    if !o.type_data.matches(o.object_type) { bail!("slot {}: type data {:?} does not fit object type {}", o.slot, o.type_data, o.object_type); }
    w.flag(true);
    w.bits(2, o.flags as u64 & 3);
    w.index(o.quota.map(|q| q as u32), 8);
    // never NONE: the engine indexes the palette entry's variant block with the
    // raw byte (see `H4PlacedObject::new`); an explicit 0 is what the game itself writes
    w.index(Some(o.variant.unwrap_or(0) as u32), 5);
    w.flag(o.in_bounds);
    for i in 0..3 { w.bits(axis[i], h4_quantize_pos(o.pos[i], bounds[2 * i], bounds[2 * i + 1], axis[i]) as u64); }
    w.flag(o.up_is_global);
    if !o.up_is_global { w.bits(20, o.up_quant as u64); }
    w.bits(14, o.forward_angle_q as u64);
    w.bits(10, (parent as i32 + 1) as u64 & 0x3FF);
    w.bits(6, o.scale_q as u64);
    w.flag(o.locked);
    w.bits(2, o.shape as u64);
    for &v in o.shape_values.iter().take(o.shape.value_count()) { w.u16(v); }
    w.bits(6, o.object_type as u64);
    w.bits(10, o.placement as u64);
    match &o.type_data {
        H4TypeData::InitialOrdnance { team, weapon, timing } => { w.bits(4, (*team as i32 + 1) as u64 & 15); w.u8(*weapon); w.u16(*timing); }
        H4TypeData::RandomOrdnance(ws) => { for &x in ws { w.u8(x); } }
        H4TypeData::ObjectiveOrdnance { lead, weapons } => { w.u8(*lead); for &x in weapons { w.u8(x); } }
        H4TypeData::Empty => {}
        td => {
            w.bits(4, (o.team as i32 + 1) as u64 & 15);
            w.u8(o.spawn_time);
            w.index(o.color.map(|c| c as u32), 3);
            w.u8(o.spawn_sequence as u8);
            w.u8(o.user_data as u8);
            for l in &o.labels { w.index(l.map(|x| x as u32), 8); }
            match td {
                H4TypeData::Byte(b) => w.u8(*b),
                H4TypeData::Index(i) => w.u8((*i + 1) as u8),
                H4TypeData::TraitZone(t) => w.bits(5, *t as u64),
                H4TypeData::Pair(a, b) => { w.bits(5, *a as u64); w.bits(5, *b as u64); }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

/// Forge label table, uncompressed (sub_1800B8E3C compresses only buffers >= 128 bytes; the
/// decoder accepts either form, so the plain layout is always valid). Caps: 9-bit count,
/// 12-bit offsets, 13-bit size - labels past a cap are dropped.
fn write_labels(w: &mut BitWriter, labels: &[String]) {
    let mut buf: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    for s in labels {
        if offsets.len() >= 511 || buf.len() >= 4096 || buf.len() + s.len() + 1 > 8191 { break; }
        offsets.push(buf.len());
        buf.extend_from_slice(s.as_bytes());
        buf.push(0);
    }
    w.bits(9, offsets.len() as u64);
    for &o in &offsets { w.flag(true); w.bits(12, o as u64); }
    if offsets.is_empty() { return; }
    w.bits(13, buf.len() as u64);
    w.flag(false);
    for &b in &buf { w.u8(b); }
}

/// New values for the editable header fields. `None` = copy the original bits verbatim (so an
/// untouched field survives byte-for-byte even when a sibling changed).
#[derive(Clone, Debug, Default)]
pub struct H4HeaderEdits {
    /// Content-header title / description (UTF-16, max 128 units; also mirrored into `chdr`).
    pub title: Option<String>,
    pub description: Option<String>,
    /// Author / editor gamertags (max 16 bytes; mirrored into `chdr`).
    pub author: Option<String>,
    pub editor: Option<String>,
    /// Whole forge-label table (see `write_labels`).
    pub labels: Option<Vec<String>>,
    /// `budget_spent`; use `budget_spent(objects, palette)` to recompute it from the palette prices.
    pub budget_spent: Option<u32>,
    /// CreatedBy / ModifiedBy (unix timestamp, xuid), written into the bitstream
    /// AND the `chdr` mirror (+0x3C/+0x44 and +0x60/+0x68). `stamp_new` / `stamp_modified` build
    /// them the way the game does; `None` keeps the source bits.
    pub created: Option<(u64, u64)>,
    pub modified: Option<(u64, u64)>,
    /// `megalo-category-index` (signed 8 bits, -1 = none).
    pub category: Option<i8>,
    /// `maximum_budget`.
    pub budget_max: Option<u32>,
    /// `world-bounds` xmin xmax ymin ymax zmin zmax. Every object is re-quantised
    /// against the new box (positions outside it are clamped to its edge by the quantiser, as
    /// the engine's own writer does); each axis must have max > min.
    pub bounds: Option<[f32; 6]>,
    /// Per-quota-entry `(minimum_count, maximum_count)`; entry i replaces quota
    /// i, entries past the vec (or past `num_quotas`) keep the source values. `placed_on_map` is
    /// always recomputed and `maximum_count` still raised to it.
    pub quotas: Option<Vec<(u8, u8)>>,
}

impl H4HeaderEdits {
    pub fn is_empty(&self) -> bool {
        self.title.is_none() && self.description.is_none() && self.author.is_none() && self.editor.is_none() && self.labels.is_none() && self.budget_spent.is_none()
            && self.created.is_none() && self.modified.is_none()
            && self.category.is_none() && self.budget_max.is_none() && self.bounds.is_none() && self.quotas.is_none()
    }
    /// Stamp a NEW variant the way MCC does when Forge saves a fresh one (as on the game-saved
    /// files in `LocalFiles\<xuid>\HaloReach\Map`): CreatedBy = (now, xuid),
    /// ModifiedBy = (0, 0). The unique id is NOT touched - the game itself keeps the base
    /// variant's id on every derived save (30+ user files share one id and all list), so a copy
    /// of the template's id is exactly what a game-made file carries.
    pub fn stamp_new(&mut self, xuid: u64) {
        self.created = Some((unix_now(), xuid));
        self.modified = Some((0, 0));
    }
    /// Stamp a re-save of an existing variant the way MCC does: CreatedBy kept, ModifiedBy =
    /// (now, xuid).
    pub fn stamp_modified(&mut self, xuid: u64) {
        self.modified = Some((unix_now(), xuid));
    }
}

/// Seconds since the unix epoch (what the content header timestamps count; 0 when the clock is
/// before 1970, which never happens on a real machine).
pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Sum of the palette prices of `objects` (what the engine stores as `budget_spent`; equal on
/// all 388 shipped files). Objects whose quota is outside the palette cost 0.
pub fn budget_spent(objects: &[H4PlacedObject], palette: &[H4PaletteEntry]) -> u32 {
    objects.iter().filter_map(|o| o.quota.and_then(|q| palette.get(q as usize)).map(|e| e.price.max(0) as u32)).sum()
}

/// Result of `encode_h4_payload`.
pub struct H4Encoded {
    /// The 28 672-byte bitstream (zero past `bits`).
    pub payload: Vec<u8>,
    /// Bits written = the engine's `length` numerator (`length = (bits + 7) / 8`).
    pub bits: usize,
    /// Final slot of each input object (index-aligned with the input list).
    pub slots: Vec<u16>,
}

/// Encode a variant payload from `src` (the source payload, header + trait records copied
/// verbatim) and the FULL object list:
/// * an object keeps its `slot` (decoded objects), a `NEW_SLOT` / >= 651 key takes the lowest
///   free slot; duplicate keys or more objects than free slots are errors;
/// * `parent` names the parent's slot key and is remapped to that object's final slot; a parent
///   that is not in the list orphans the child (-1) - except values past the 651 slots, which
///   are kept as found (a couple of Reach files store such values; no meaning is invented);
/// * the quota table's `placed_on_map` is recomputed per quota index and `maximum` raised to
///   it when exceeded (`minimum` untouched); `num_quotas` never changes;
/// * header strings / labels / budget come from `edits`, everything else verbatim.
pub fn encode_h4_payload(src: &[u8], v: &H4Variant, objects: &[H4PlacedObject], edits: Option<&H4HeaderEdits>) -> Result<H4Encoded> {
    if v.version != 50 && v.version != 51 { bail!("variant body version {} (encoder handles 50 and 51)", v.version); }
    let e = edits.cloned().unwrap_or_default();
    // an edited box re-quantises every object (write) while the source is read
    // with its own box (the decoded `pos` floats are the bridge)
    let (bounds, _ext, axis) = match e.bounds {
        Some(b) => {
            if (0..3).any(|i| !(b[2 * i + 1] > b[2 * i]) || !b[2 * i].is_finite() || !b[2 * i + 1].is_finite()) { bail!("world bounds must have max > min on every axis: {b:?}"); }
            let ext = [b[1] - b[0], b[3] - b[2], b[5] - b[4]];
            (b, ext, h4_axis_bits(21, ext))
        }
        None => variant_axes(v),
    };
    let mut w = BitWriter::new(H4_MVAR_BITSTREAM_BYTES);
    // --- header: verbatim except the edited spans ---
    let sp = &v.spans;
    let stamp = |w: &mut BitWriter, span: (usize, usize), v: Option<(u64, u64)>| match v {
        Some((ts, xuid)) => { w.bits(64, ts); w.bits(64, xuid); }
        None => w.copy_bits(src, span.0, span.1 - span.0),
    };
    w.copy_bits(src, 0, sp.category);
    match e.category { Some(c) => w.bits(8, c as u8 as u64), None => w.copy_bits(src, sp.category, 8) }
    w.copy_bits(src, sp.category + 8, sp.created.0 - (sp.category + 8));
    stamp(&mut w, sp.created, e.created);
    match &e.author { Some(s) => w.cstr(s, 16), None => w.copy_bits(src, sp.author.0, sp.author.1 - sp.author.0) }
    w.copy_bits(src, sp.author.1, sp.modified.0 - sp.author.1); // the IsOnline flag
    stamp(&mut w, sp.modified, e.modified);
    match &e.editor { Some(s) => w.cstr(s, 16), None => w.copy_bits(src, sp.editor.0, sp.editor.1 - sp.editor.0) }
    w.copy_bits(src, sp.editor.1, sp.title.0 - sp.editor.1);
    match &e.title { Some(s) => w.wstr(s, 128), None => w.copy_bits(src, sp.title.0, sp.title.1 - sp.title.0) }
    match &e.description { Some(s) => w.wstr(s, 128), None => w.copy_bits(src, sp.description.0, sp.description.1 - sp.description.0) }
    w.copy_bits(src, sp.description.1, sp.bounds - sp.description.1);
    match e.bounds { Some(b) => { for x in b { w.f32(x); } } None => w.copy_bits(src, sp.bounds, 192) }
    match e.budget_max { Some(b) => w.u32(b), None => w.copy_bits(src, sp.bounds + 192, 32) }
    match e.budget_spent { Some(b) => w.u32(b), None => w.copy_bits(src, sp.budget_spent, 32) }
    w.copy_bits(src, sp.budget_spent + 32, sp.labels.0 - (sp.budget_spent + 32));
    match &e.labels { Some(ls) => write_labels(&mut w, ls), None => w.copy_bits(src, sp.labels.0, sp.labels.1 - sp.labels.0) }
    w.copy_bits(src, sp.labels.1, v.obj_start_bit - sp.labels.1);
    // --- slot assignment ---
    let mut by_slot: Vec<Option<usize>> = vec![None; H4_SLOTS];
    let mut key_to_slot: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();
    let mut fresh: Vec<usize> = Vec::new();
    for (i, o) in objects.iter().enumerate() {
        if (o.slot as usize) < H4_SLOTS {
            if by_slot[o.slot as usize].is_some() { bail!("two objects carry slot {}", o.slot); }
            by_slot[o.slot as usize] = Some(i);
            key_to_slot.insert(o.slot, o.slot);
        } else {
            if o.slot != NEW_SLOT && key_to_slot.contains_key(&o.slot) { bail!("two new objects carry key {}", o.slot); }
            fresh.push(i);
        }
    }
    let mut next_free = 0usize;
    for &i in &fresh {
        while next_free < H4_SLOTS && by_slot[next_free].is_some() { next_free += 1; }
        if next_free >= H4_SLOTS { bail!("no free slot for object {} of {} (651 slots)", i, objects.len()); }
        by_slot[next_free] = Some(i);
        if objects[i].slot != NEW_SLOT { key_to_slot.insert(objects[i].slot, next_free as u16); }
        next_free += 1;
    }
    let mut slots = vec![NEW_SLOT; objects.len()];
    let mut placed = vec![0u32; H4_QUOTA_SLOTS];
    for s in 0..H4_SLOTS {
        let Some(i) = by_slot[s] else { w.flag(false); continue };
        let o = &objects[i];
        slots[i] = s as u16;
        let parent = if o.parent < 0 { -1 } else if (o.parent as usize) >= H4_SLOTS && !key_to_slot.contains_key(&(o.parent as u16)) { o.parent } else { key_to_slot.get(&(o.parent as u16)).map_or(-1, |&p| p as i16) };
        write_slot(&mut w, o, parent, bounds, axis)?;
        if let Some(q) = o.quota { placed[q as usize] += 1; }
    }
    // --- quota table: source min / max, recomputed placed ---
    let mut r = Bits::new(src);
    r.pos = v.obj_end_bit;
    for i in 0..v.num_quotas as usize {
        let (mut mn, mut mx, _) = (r.u8(), r.u8(), r.u8());
        if let Some((a, b)) = e.quotas.as_ref().and_then(|q| q.get(i)) { mn = *a; mx = *b; }
        let n = placed[i].min(255) as u8;
        w.u8(mn);
        w.u8(mx.max(n));
        w.u8(n);
    }
    // --- the four trait-set records, verbatim ---
    w.copy_bits(src, v.quota_end_bit, v.end_bit - v.quota_end_bit);
    let bits = w.pos;
    if bits > H4_MVAR_BITSTREAM_BYTES * 8 { bail!("encoded variant is {bits} bits, over the 28 672-byte bitstream"); }
    Ok(H4Encoded { payload: w.buf, bits, slots })
}

/// Plain SHA-1 (FIPS 180-4) - the chunk hash needs no salt, see the section comment.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut wds = [0u32; 80];
        for i in 0..16 { wds[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]); }
        for i in 16..80 { wds[i] = (wds[i - 3] ^ wds[i - 8] ^ wds[i - 14] ^ wds[i - 16]).rotate_left(1); }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wd) in wds.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wd);
            e = d; d = c; c = b.rotate_left(30); b = a; a = t;
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b); h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d); h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for i in 0..5 { out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}

/// The `mvar` chunk hash (sub_1802380D4): SHA-1 over the LITTLE-endian u32 `length` followed by
/// the first `length` payload bytes (the MCC / x64 flavour, `H4HashKind::PlainLe`).
pub fn mvar_chunk_hash(payload: &[u8], length: u32) -> [u8; 20] { mvar_chunk_hash_kind(payload, length, H4HashKind::PlainLe) }

/// Which hash a `mvar` chunk carries. The engine's formula is one and the same
/// (`SHA1(salt? || native u32 length || payload[..length])`); MCC hashes on x64 with NO salt
/// (`sub_180053654`), the Xbox 360 game hashed big-endian with the public 34-byte "shared"
/// BLF salt - 360-origin variants (file-share downloads / migrated saves, author XUIDs
/// `0xE000...`) still carry that hash and MCC's reader never checks it. A save reproduces
/// whichever flavour the source has (byte-identical on both kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H4HashKind {
    /// `SHA1(LE length || payload[..length])` - MCC.
    PlainLe,
    /// `SHA1(SALT_360 || BE length || payload[..length])` - Xbox 360 origin.
    SaltedBe360,
}

/// The Xbox 360 Halo Reach / Halo 4 "shared" BLF salt (public; every 360 disc shipped it and
/// every open-source BLF tool carries it).
pub const SALT_360: [u8; 34] = [
    0xED, 0xD4, 0x30, 0x09, 0x66, 0x6D, 0x5C, 0x4A, 0x5C, 0x36, 0x57, 0xFA, 0xB4, 0x0E, 0x02, 0x2F, 0x53,
    0x5A, 0xC6, 0xC9, 0xEE, 0x47, 0x1F, 0x01, 0xF1, 0xA4, 0x47, 0x56, 0xB7, 0x71, 0x4F, 0x1C, 0x36, 0xEC,
];

pub fn mvar_chunk_hash_kind(payload: &[u8], length: u32, kind: H4HashKind) -> [u8; 20] {
    let mut m = Vec::with_capacity(38 + length as usize);
    match kind {
        H4HashKind::PlainLe => m.extend_from_slice(&length.to_le_bytes()),
        H4HashKind::SaltedBe360 => { m.extend_from_slice(&SALT_360); m.extend_from_slice(&length.to_be_bytes()); }
    }
    m.extend_from_slice(&payload[..(length as usize).min(payload.len())]);
    sha1(&m)
}

/// Which flavour the file's `mvar` chunk carries (None: neither matches - the writer then uses
/// `PlainLe`, the one MCC writes).
pub fn detect_hash_kind(data: &[u8]) -> Option<H4HashKind> {
    let (payload, _, _) = blf_h4_mvar_chunk(data)?;
    let length = mvar_chunk_length(data)?;
    let off = data.windows(4).position(|w| w == b"mvar")? + 12;
    let stored = &data[off..off + 20];
    [H4HashKind::PlainLe, H4HashKind::SaltedBe360].into_iter().find(|k| mvar_chunk_hash_kind(payload, length, *k)[..] == *stored)
}

/// `chdr` (v10) body offsets, relative to the chunk's +12 body, that mirror the bitstream header
/// (holds on all 388 files): author name @0x4C (16 B), editor name @0x70 (16 B), title
/// @0x84 (128 UTF-16LE units), description @0x184 (128 units).
pub const CHDR_AUTHOR: usize = 0x4C;
pub const CHDR_EDITOR: usize = 0x70;
/// CreatedBy timestamp / xuid and ModifiedBy timestamp / xuid (LE u64 each; the
/// 16 bytes right before each name).
pub const CHDR_CREATED: usize = 0x3C;
pub const CHDR_MODIFIED: usize = 0x60;
pub const CHDR_TITLE: usize = 0x84;
pub const CHDR_DESCRIPTION: usize = 0x184;
/// The `chdr` body mirrors the whole content header little-endian: build number
/// u16 @0, map minor version u16 @2, then the 688-byte `s_content_item_metadata`: type i32 @4,
/// file size @8, uid @0xC, parent uid @0x14, root uid @0x1C, game id @0x24, activity i8 @0x2C,
/// game mode @0x2D, engine @0x2E, map id @0x30, megalo category index i8 @0x34, CreatedBy @0x3C
/// {ts, xuid, name[16], flags u8 @0x5C}, ModifiedBy @0x60, title @0x84, description @0x184,
/// then the type / activity / engine tail (film seconds / icon index / hopper id / campaign) at
/// @0x284 (checked on every shipped file of both games against the bitstream copy).
pub const CHDR_MAP_ID: usize = 0x30;
pub const CHDR_CATEGORY: usize = 0x34;

pub(crate) fn chdr_put_str(body: &mut [u8], at: usize, s: &str, max: usize) {
    let b = s.as_bytes();
    let n = b.len().min(max);
    body[at..at + max].fill(0);
    body[at..at + n].copy_from_slice(&b[..n]);
}
pub(crate) fn chdr_put_stamp(body: &mut [u8], at: usize, ts: u64, xuid: u64) {
    body[at..at + 8].copy_from_slice(&ts.to_le_bytes());
    body[at + 8..at + 16].copy_from_slice(&xuid.to_le_bytes());
}
/// The (timestamp, xuid) pair stored at `at` in a `chdr` body (the mirror the tests compare).
pub fn chdr_stamp(body: &[u8], at: usize) -> (u64, u64) {
    let u = |o: usize| u64::from_le_bytes(body[o..o + 8].try_into().unwrap());
    (u(at), u(at + 8))
}
pub(crate) fn chdr_put_wstr(body: &mut [u8], at: usize, s: &str, max: usize) {
    let u: Vec<u16> = s.encode_utf16().take(max).collect();
    body[at..at + 2 * max].fill(0);
    for (i, c) in u.iter().enumerate() { body[at + 2 * i..at + 2 * i + 2].copy_from_slice(&c.to_le_bytes()); }
}

/// Rebuild a whole `.mvar` file from `src_data` (the original file bytes) with a new object list:
/// every chunk but `mvar` (and the mirrored `chdr` strings) is copied verbatim; the `mvar` chunk
/// gets the new payload, `length = ceil(bits / 8)` and its SHA-1. Returns the file bytes and the
/// final slots (index-aligned with `objects`).
pub fn rebuild_h4_file(src_data: &[u8], objects: &[H4PlacedObject], edits: Option<&H4HeaderEdits>) -> Result<(Vec<u8>, Vec<u16>)> {
    let (payload, major, _) = blf_h4_mvar_chunk(src_data).ok_or_else(|| anyhow!("no mvar chunk"))?;
    let v = parse_h4_payload(payload, major)?;
    let enc = encode_h4_payload(payload, &v, objects, edits)?;
    let length = ((enc.bits + 7) / 8) as u32;
    let hash = mvar_chunk_hash_kind(&enc.payload, length, detect_hash_kind(src_data).unwrap_or(H4HashKind::PlainLe));
    let mut out = Vec::with_capacity(src_data.len());
    let mut pos = 0usize;
    while pos + 12 <= src_data.len() {
        let magic = &src_data[pos..pos + 4];
        let size = u32::from_be_bytes([src_data[pos + 4], src_data[pos + 5], src_data[pos + 6], src_data[pos + 7]]) as usize;
        if size < 12 || pos + size > src_data.len() { break; }
        if magic == b"mvar" && size >= 36 {
            out.extend_from_slice(&src_data[pos..pos + 12]);
            out.extend_from_slice(&hash);
            out.extend_from_slice(&length.to_be_bytes());
            out.extend_from_slice(&enc.payload);
            // the chunk's bytes past the 0x7000 bitstream (a zero u32 on every shipped file)
            if H4_MVAR_BITSTREAM_BYTES < size - 36 { out.extend_from_slice(&src_data[pos + 36 + H4_MVAR_BITSTREAM_BYTES..pos + size]); }
        } else if magic == b"chdr" && size >= 12 + CHDR_DESCRIPTION + 256 {
            let mut body = src_data[pos + 12..pos + size].to_vec();
            if let Some(e) = edits {
                if let Some(s) = &e.author { chdr_put_str(&mut body, CHDR_AUTHOR, s, 16); }
                if let Some(s) = &e.editor { chdr_put_str(&mut body, CHDR_EDITOR, s, 16); }
                if let Some((ts, xuid)) = e.created { chdr_put_stamp(&mut body, CHDR_CREATED, ts, xuid); }
                if let Some((ts, xuid)) = e.modified { chdr_put_stamp(&mut body, CHDR_MODIFIED, ts, xuid); }
                if let Some(s) = &e.title { chdr_put_wstr(&mut body, CHDR_TITLE, s, 128); }
                if let Some(s) = &e.description { chdr_put_wstr(&mut body, CHDR_DESCRIPTION, s, 128); }
                if let Some(c) = e.category { body[CHDR_CATEGORY] = c as u8; }
            }
            out.extend_from_slice(&src_data[pos..pos + 12]);
            out.extend_from_slice(&body);
        } else {
            out.extend_from_slice(&src_data[pos..pos + size]);
        }
        pos += size;
    }
    out.extend_from_slice(&src_data[pos..]);
    Ok((out, enc.slots))
}

/// THE save: write `objects` (the full list, see `encode_h4_payload`) into a copy
/// of `src` at `dst`, with optional header edits. `src` may equal `dst`. Returns the final slot
/// of every object (a GUI keeps these as the objects' identities for the next save).
pub fn save_objects(src: &Path, dst: &Path, objects: &[H4PlacedObject], edits: Option<&H4HeaderEdits>) -> Result<Vec<u16>> {
    let data = read_h4_expanded(src).map_err(|e| anyhow!("{}: {e}", src.display()))?;
    let (out, slots) = rebuild_h4_file(&data, objects, edits)?;
    std::fs::write(dst, out).map_err(|e| anyhow!("{}: {e}", dst.display()))?;
    Ok(slots)
}

/// Header-only edit: rewrite `src` -> `dst` with its own objects and the given strings.
pub fn write_header(src: &Path, dst: &Path, edits: &H4HeaderEdits) -> Result<()> {
    let v = parse_h4_variant(src)?;
    save_objects(src, dst, &v.objects, Some(edits)).map(|_| ())
}

/// Round-trip report for `--h4-mvar-roundtrip`: decode, re-encode with no edits, compare with
/// the original bytes. `out` receives the re-encoded file when given.
pub fn roundtrip_report(path: &Path, out: Option<&Path>) -> Result<String> {
    let raw = std::fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let data = crate::mvar::expand_blf_cmp(&raw);
    let cmp_note = if data != raw { format!(" [source `_cmp`-compressed, {} B; compared against its {} B expansion]", raw.len(), data.len()) } else { String::new() };
    let (payload, major, _) = blf_h4_mvar_chunk(&data).ok_or_else(|| anyhow!("{}: no mvar chunk", path.display()))?;
    let v = parse_h4_payload(payload, major)?;
    let (re, _) = rebuild_h4_file(&data, &v.objects, None)?;
    if let Some(o) = out { std::fs::write(o, &re).map_err(|e| anyhow!("{}: {e}", o.display()))?; }
    let first_diff = data.iter().zip(re.iter()).position(|(a, b)| a != b).or_else(|| (data.len() != re.len()).then_some(data.len().min(re.len())));
    let stored_length = mvar_chunk_length(&data).unwrap_or(0);
    Ok(match first_diff {
        None => format!("{}: body v{}, {} objects, {} bits (length {} == stored {}), hash {:?}, re-encode BYTE-IDENTICAL ({} bytes){cmp_note}", path.display(), v.version, v.objects.len(), v.end_bit, (v.end_bit + 7) / 8, stored_length, detect_hash_kind(&data), data.len()),
        Some(i) => format!("{}: {} objects, re-encode DIFFERS at byte {} (original {} bytes, re-encoded {})", path.display(), v.objects.len(), i, data.len(), re.len()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h4::cache::maps_dir;
    use crate::h4::geometry::load_bsp;
    use std::collections::{BTreeMap, HashMap};

    fn open_map(name: &str) -> Option<H4Cache> {
        let p = maps_dir()?.join(format!("{name}.map"));
        if !p.is_file() { eprintln!("skip: {} missing", p.display()); return None; }
        Some(H4Cache::open(&p).expect("open"))
    }

    /// map id -> map stem from every .mapinfo next to the maps.
    fn map_ids() -> HashMap<u32, String> {
        let mut out = HashMap::new();
        let Some(dir) = maps_dir() else { return out };
        for e in std::fs::read_dir(dir.join("info")).into_iter().flatten().flatten() {
            let p = e.path();
            if p.extension().map_or(false, |x| x == "mapinfo") {
                if let Some(id) = crate::mapcat::read_map_id(&dir.join(p.file_stem().unwrap()).with_extension("map")) {
                    out.insert(id, p.file_stem().unwrap().to_string_lossy().into_owned());
                }
            }
        }
        out
    }

    /// Bit offset of the first player-trait record after the quota table: in every shipped file
    /// 78 zero bits precede the record's `11111101 x5` run (the default trait bytes; the one file
    /// with an edited first trait set, h4_impact_squad_monsoon, still starts with them). Located
    /// by pattern search, independent of the object decode - the cross-check for the layout.
    fn trait_anchor(payload: &[u8]) -> Option<usize> {
        let bits: Vec<u8> = payload.iter().flat_map(|b| (0..8).rev().map(move |k| (b >> k) & 1)).collect();
        let pat: Vec<u8> = "11111101".repeat(5).bytes().map(|c| c - b'0').collect();
        (0..bits.len() - pat.len()).find(|&i| bits[i..i + pat.len()] == pat[..])
    }

    /// The game's variant folders are writable and both MCC Forge and HMS save into them, so this
    /// gate has to tell a shipped file from a save made on this machine. The content header's
    /// CreatedBy / ModifiedBy timestamps do it: every one of the 388 shipped variants is stamped
    /// between 2012-11-05 and 2020-11-26 (the last MCC-era Halo 4 content update), and a save
    /// carries the time it was written. The other header fields do not separate them - `built_in`
    /// / `built_from_xml` are 1 / 1 on all 388 and are copied verbatim into a save, the xuid of
    /// both stamps is 0 everywhere, and ~180 of the shipped hopper variants are authored by
    /// community gamertags rather than 343 Industries.
    const SHIPPED_STAMP_CUTOFF: u64 = 1_609_459_200; // 2021-01-01

    fn is_shipped(v: &H4Variant) -> bool {
        v.created.0 <= SHIPPED_STAMP_CUTOFF && v.modified.0 <= SHIPPED_STAMP_CUTOFF
    }

    /// Every variant file found in the game's variant folders decodes, has its object positions
    /// inside the variant bounds, unit orientation vectors, in-range teams and the constant record
    /// head. On the SHIPPED files this also checks the 343-authored invariants: the `(50, 1)` chunk
    /// version, slots + quotas ending exactly 78 bits before the trait records, quota indices
    /// inside the base map's palette, and placed counts + spent budget exact against that palette.
    /// User saves (MCC Forge, HMS) live in the same folders and are held to the decode half only,
    /// since nothing constrains what they may contain; the census line names them so a file can
    /// never be skipped unnoticed.
    #[test]
    fn shipped_variants_decode_exactly() {
        let files = all_variant_files();
        if files.is_empty() { eprintln!("skip: no halo4 variant folders"); return; }
        let ids = map_ids();
        let mut palettes: HashMap<String, Vec<H4PaletteEntry>> = HashMap::new();
        let mut caches: HashMap<String, H4Cache> = HashMap::new();
        let mut anchored = 0usize;
        let mut total_objects = 0usize;
        let mut by_map: BTreeMap<String, usize> = BTreeMap::new();
        let mut n_shipped = 0usize;
        let mut user_anchored = 0usize;
        let mut user_files: Vec<String> = Vec::new();
        for f in &files {
            let data = std::fs::read(f).unwrap();
            let (payload, major, minor) = blf_h4_mvar_chunk(&data).expect("mvar chunk");
            assert_eq!(payload.len(), H4_MVAR_PAYLOAD);
            let v = parse_h4_payload(payload, major).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
            let shipped = is_shipped(&v);
            if shipped {
                n_shipped += 1;
                assert_eq!((major, minor), (50, 1), "{}", f.display());
                assert!(v.num_quotas as usize <= 256 && v.budget_spent <= v.budget_max, "{}", f.display());
            }
            assert_eq!(v.map_id, v.header_map_id);
            assert!(v.num_quotas as usize <= 256, "{}: num_quotas {}", f.display(), v.num_quotas);
            // The base map's palette prices the objects. One cache per map; a user save for a map
            // that is not installed keeps the per-object checks and skips the palette ones.
            let stem = match ids.get(&v.map_id) {
                Some(s) => s.clone(),
                None if !shipped => {
                    user_files.push(format!("{} (map id {}: not installed)", f.file_name().unwrap_or_default().to_string_lossy(), v.map_id));
                    check_objects(f, &v, None);
                    total_objects += v.objects.len();
                    continue;
                }
                None => panic!("{}: map id {} has no .mapinfo", f.display(), v.map_id),
            };
            *by_map.entry(stem.clone()).or_default() += 1;
            if let Some(a) = trait_anchor(payload) {
                if shipped {
                    assert_eq!(a, v.quota_end_bit + 78, "{}: slots+quotas end {} vs trait anchor {}", f.display(), v.quota_end_bit, a);
                    anchored += 1;
                } else {
                    user_anchored += 1;
                }
            }
            if !caches.contains_key(&stem) {
                if let Some(c) = open_map(&stem) { palettes.insert(stem.clone(), load_forge_palette(&c)); caches.insert(stem.clone(), c); }
            }
            let pal = &palettes[&stem];
            let (per_quota, spent) = check_objects(f, &v, Some(pal));
            if shipped {
                assert!(v.num_quotas as usize == 256 || v.num_quotas as usize == pal.len(), "{}: num_quotas {} palette {}", f.display(), v.num_quotas, pal.len());
                for (i, q) in v.quotas.iter().enumerate() { assert_eq!(q.placed as usize, per_quota[i], "{}: quota {i} placed", f.display()); }
                assert_eq!(spent, v.budget_spent as i64, "{}: budget", f.display());
            } else {
                let placed_ok = v.quotas.iter().enumerate().all(|(i, q)| q.placed as usize == per_quota[i]);
                user_files.push(format!("{} on {stem}: {} objects, budget {}/{} (palette prices sum to {spent}), quota placed counts {}",
                    f.file_name().unwrap_or_default().to_string_lossy(), v.objects.len(), v.budget_spent, v.budget_max,
                    if placed_ok { "match" } else { "DIFFER" }));
            }
            total_objects += v.objects.len();
        }
        eprintln!("h4 mvar: {} files ({n_shipped} shipped, {} user), {} objects, {anchored} shipped + {user_anchored} user trait-anchored, per map {:?}",
            files.len(), user_files.len(), total_objects, by_map);
        for u in &user_files { eprintln!("h4 mvar: user save (decode checks only) {u}"); }
        assert_eq!(anchored, n_shipped, "trait anchor found in {anchored} of {n_shipped} shipped files");
        assert!(n_shipped >= 380, "only {n_shipped} of {} files look shipped - has the census rule stopped matching?", files.len());
        assert!(total_objects > 60_000);
    }

    /// The per-object decode checks every variant file must pass, shipped or not: positions inside
    /// the variant bounds, an orthonormal basis, the constant record head, an in-range team, and
    /// (when the base map's palette is available) a quota / variant that exists in it. Returns the
    /// per-quota placement census and the budget the palette prices come to.
    fn check_objects(f: &Path, v: &H4Variant, pal: Option<&Vec<H4PaletteEntry>>) -> (Vec<usize>, i64) {
        let mut per_quota = vec![0usize; 256];
        let mut spent = 0i64;
        for o in &v.objects {
            let q = o.quota.expect("a placed object always carries a quota") as usize;
            per_quota[q] += 1;
            if let Some(pal) = pal {
                assert!(q < pal.len(), "{}: quota {q} outside the {}-entry palette", f.display(), pal.len());
                assert!(o.variant.map_or(true, |vi| (vi as usize) < pal[q].variants.len().max(1)), "{}: variant {:?} of {}", f.display(), o.variant, pal[q].name);
                spent += pal[q].price as i64;
            }
            for i in 0..3 { assert!(o.pos[i] >= v.bounds[2 * i] - 0.01 && o.pos[i] <= v.bounds[2 * i + 1] + 0.01, "{}: slot {} pos {:?} outside {:?}", f.display(), o.slot, o.pos, v.bounds); }
            let (fw, up) = (Vec3::from(o.fwd), Vec3::from(o.up));
            assert!((fw.length() - 1.0).abs() < 1e-3 && (up.length() - 1.0).abs() < 1e-3 && fw.dot(up).abs() < 1e-3, "{}: slot {} basis", f.display(), o.slot);
            assert!(o.in_bounds && o.flags == 1 && o.parent < 651, "{}: slot {} head {:?}/{}/{}", f.display(), o.slot, o.in_bounds, o.flags, o.parent);
            assert!(o.team >= -1 && o.team <= 8, "{}: slot {} team {}", f.display(), o.slot, o.team);
        }
        (per_quota, spent)
    }

    /// Named facts for one shipped variant of each kind.
    #[test]
    fn known_variants() {
        let dirs = variant_dirs();
        let Some(mv) = dirs.iter().find(|d| d.ends_with("map_variants")) else { eprintln!("skip"); return };
        let g = parse_h4_variant(&mv.join("grifballcourt.mvar")).unwrap();
        assert_eq!((g.map_id, g.num_quotas, g.budget_max, g.budget_spent), (10245, 256, 10000, 20));
        assert_eq!(g.labels, vec!["grif_spawn".to_string(), "grif_goal".to_string()]);
        assert_eq!(g.title_key, "$h4_mvar_grifball_name");
        assert_eq!(g.title, "Grifball Court"); // resolved through EN_Global.bin
        assert_eq!(g.description_key, "$h4_mvar_grifball_description");
        assert!(g.description.starts_with("Prepare for awesome."), "{}", g.description);
        assert_eq!(g.objects.len(), 68);
        let o = &g.objects[0];
        assert_eq!((o.quota, o.variant, o.object_type, o.placement, o.team, o.parent, o.scale_q), (Some(48), Some(0), 16, 12, 1, -1, 7));
        assert!((o.pos[0] + 25.15).abs() < 0.05 && (o.pos[1] + 38.42).abs() < 0.05 && (o.pos[2] - 0.82).abs() < 0.05, "{:?}", o.pos);
        assert!(o.up_is_global && o.labels.iter().all(|l| l.is_none()));
        // the two capture plates carry the grif_goal label (index 1)
        let plates: Vec<&H4PlacedObject> = g.objects.iter().filter(|o| o.quota == Some(61)).collect();
        assert_eq!(plates.len(), 3);
        assert!(plates.iter().all(|p| p.labels[0].is_some()) && plates[1].labels[0] == Some(1));
        let s = parse_h4_variant(&mv.join("ca_forge_ravine_settler.mvar")).unwrap();
        assert_eq!((s.map_id, s.objects.len(), s.budget_spent), (10256, 388, 4280));
        assert_eq!(h4_axis_bits(21, [s.bounds[1] - s.bounds[0], s.bounds[3] - s.bounds[2], s.bounds[5] - s.bounds[4]]), [22, 22, 21]);
        assert_eq!(h4_axis_bits(21, [412.0047, 521.9852, 261.4819]), [20, 20, 19]);
        assert!(!is_h4_variant(&std::path::PathBuf::from("/nonexistent.mvar")));
        assert!(is_h4_variant(&mv.join("grifballcourt.mvar")));
        assert_eq!(read_h4_map_id(&mv.join("grifballcourt.mvar")), Some(10245));
    }

    /// Placement resolution on the Ravine canvas: most objects reach a render model and land
    /// inside the map's cluster bounds.
    #[test]
    fn settler_places_on_ravine() {
        let Some(c) = open_map("ca_forge_ravine") else { return };
        let Some(mv) = variant_dirs().into_iter().find(|d| d.ends_with("map_variants")) else { return };
        let v = parse_h4_variant(&mv.join("ca_forge_ravine_settler.mvar")).unwrap();
        assert_eq!(map_id_of_cache(&maps_dir().unwrap().join("ca_forge_ravine.map")), Some(v.map_id));
        let pal = load_forge_palette(&c);
        assert_eq!(pal.len(), 125);
        assert_eq!((pal[48].name.as_str(), pal[49].name.as_str(), pal[61].name.as_str()), ("sp_initial_spawn", "sp_respawn_point", "obj_capture_plate"));
        assert!(pal.iter().all(|e| !e.variants.is_empty()));
        let (placements, st) = variant_placements(&c, &v, &pal);
        eprintln!("settler: {st:?}");
        assert_eq!(st.objects, 388);
        assert!(st.placed * 100 >= st.objects * 90, "{st:?}");
        assert_eq!(st.skipped_quota_range + st.skipped_variant_range + st.skipped_no_quota, 0);
        let mut mn = Vec3::splat(f32::MAX);
        let mut mx = Vec3::splat(f32::MIN);
        for t in c.find_tags(b"sbsp") {
            let bsp = load_bsp(&c, t).unwrap();
            for (a, b) in &bsp.cluster_bounds { mn = mn.min(Vec3::from(*a)); mx = mx.max(Vec3::from(*b)); }
        }
        let pad = (mx - mn) * 0.1 + 5.0;
        assert!(placements.iter().all(|p| { let q = Vec3::from(p.pos); q.cmpge(mn - pad).all() && q.cmple(mx + pad).all() }), "positions inside the BSP bounds {mn:?}..{mx:?}");
        assert!(placements.iter().all(|p| p.basis.is_some()));
    }

    // -----------------------------------------------------------------------------------------
    // encoder gates
    // -----------------------------------------------------------------------------------------

    fn scratch_dir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join("hms-h4-mvar-tests");
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn chdr_body(data: &[u8]) -> &[u8] {
        let mut pos = 0;
        while pos + 12 <= data.len() {
            let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
            if &data[pos..pos + 4] == b"chdr" { return &data[pos + 12..pos + size]; }
            pos += size;
        }
        panic!("no chdr");
    }
    fn chdr_str(b: &[u8], at: usize, n: usize) -> String {
        let s = &b[at..at + n];
        String::from_utf8_lossy(&s[..s.iter().position(|&c| c == 0).unwrap_or(n)]).into_owned()
    }
    fn chdr_wstr(b: &[u8], at: usize, n: usize) -> String {
        let u: Vec<u16> = (0..n).map(|i| u16::from_le_bytes([b[at + 2 * i], b[at + 2 * i + 1]])).collect();
        String::from_utf16_lossy(&u[..u.iter().position(|&c| c == 0).unwrap_or(n)])
    }

    /// The CreatedBy / ModifiedBy stamps: the decoder reads the (timestamp, xuid)
    /// pairs the chdr mirrors; `stamp_new` writes created = (now, xuid) / modified = (0, 0) into the
    /// bitstream AND the chdr (both LE u64 at +0x3C/+0x44 and +0x60/+0x68); `stamp_modified` keeps
    /// created; nothing else moves (the whole file differs from a no-edit rebuild ONLY in those 32
    /// bytes + the chunk SHA-1); and a no-edit rebuild is still byte-identical.
    #[test]
    fn created_modified_stamps_round_trip_and_mirror_into_chdr() {
        let files = all_variant_files();
        if files.is_empty() { eprintln!("skip: no halo4 variant folders"); return; }
        let mut checked = 0;
        for f in files.iter().filter(|f| f.file_name().map_or(false, |n| n.to_string_lossy().starts_with("grif_") || n.to_string_lossy().starts_with("ca_forge_"))) {
            let data = std::fs::read(f).unwrap();
            let (payload, major, _) = blf_h4_mvar_chunk(&data).unwrap();
            let v = parse_h4_payload(payload, major).unwrap();
            let cb = chdr_body(&data);
            assert_eq!(v.created, chdr_stamp(cb, CHDR_CREATED), "{}: created mirror", f.display());
            assert_eq!(v.modified, chdr_stamp(cb, CHDR_MODIFIED), "{}: modified mirror", f.display());
            let (plain, _) = rebuild_h4_file(&data, &v.objects, None).unwrap();
            assert_eq!(plain, data, "{}: no-edit rebuild changed", f.display());
            // stamp_new
            let mut e = H4HeaderEdits::default();
            e.stamp_new(0x0009_01F7_0000_0001);
            let (ts, _) = e.created.unwrap();
            assert!(ts >= 1_750_000_000, "unix_now() is a 2025+ timestamp");
            let (out, _) = rebuild_h4_file(&data, &v.objects, Some(&e)).unwrap();
            let (op, om, _) = blf_h4_mvar_chunk(&out).unwrap();
            let ov = parse_h4_payload(op, om).unwrap();
            assert_eq!(ov.created, (ts, 0x0009_01F7_0000_0001));
            assert_eq!(ov.modified, (0, 0));
            let ob = chdr_body(&out);
            assert_eq!(chdr_stamp(ob, CHDR_CREATED), ov.created);
            assert_eq!(chdr_stamp(ob, CHDR_MODIFIED), (0, 0));
            assert_eq!(ov.author, v.author);
            assert_eq!(ov.editor, v.editor);
            assert_eq!(ov.title_key, v.title_key);
            assert_eq!(ov.objects.len(), v.objects.len());
            assert_eq!(ov.end_bit, v.end_bit, "fixed-width stamps never move a bit");
            // everything outside the two 16-byte chdr stamps, the 32 stamp bytes of the bitstream
            // and the chunk SHA-1 is untouched
            let chdr_off = out.windows(4).position(|w| w == b"chdr").unwrap() + 12;
            let mvar_off = out.windows(4).position(|w| w == b"mvar").unwrap();
            let bs = mvar_off + 36;
            let stamp_bytes: Vec<std::ops::Range<usize>> = vec![
                chdr_off + CHDR_CREATED..chdr_off + CHDR_CREATED + 16,
                chdr_off + CHDR_MODIFIED..chdr_off + CHDR_MODIFIED + 16,
                mvar_off + 12..mvar_off + 32,
                bs + v.spans.created.0 / 8..bs + (v.spans.created.1 + 7) / 8,
                bs + v.spans.modified.0 / 8..bs + (v.spans.modified.1 + 7) / 8,
            ];
            let diffs: Vec<usize> = (0..out.len()).filter(|&i| out[i] != data[i]).collect();
            for i in &diffs {
                assert!(stamp_bytes.iter().any(|r| r.contains(i)), "{}: byte {i:#x} changed outside the stamps", f.display());
            }
            assert_eq!(out.len(), data.len());
            // stamp_modified keeps created
            let mut e2 = H4HeaderEdits::default();
            e2.stamp_modified(7);
            let (out2, _) = rebuild_h4_file(&data, &v.objects, Some(&e2)).unwrap();
            let (op2, om2, _) = blf_h4_mvar_chunk(&out2).unwrap();
            let ov2 = parse_h4_payload(op2, om2).unwrap();
            assert_eq!(ov2.created, v.created);
            assert_eq!(ov2.modified.1, 7);
            assert_eq!(chdr_stamp(chdr_body(&out2), CHDR_MODIFIED), ov2.modified);
            // and a re-save of the stamped file with no edits is a fixed point
            let (again, _) = rebuild_h4_file(&out, &ov.objects, None).unwrap();
            assert_eq!(again, out);
            checked += 1;
        }
        assert!(checked > 0, "no grif_/ca_forge_ files to check");
    }

    /// Tool, not a gate: `HMS_STAMP_IN=<file.mvar> HMS_STAMP_OUT=<out.mvar> cargo test --release
    /// -p hms-app stamp_new_file_from_env -- --ignored` rewrites a Halo 4 variant with its own
    /// objects and a fresh CreatedBy = now / ModifiedBy = 0 stamp (what File > New + Save As
    /// writes), for dropping into `halo4\map_variants` to test the listing.
    #[test]
    #[ignore]
    fn stamp_new_file_from_env() {
        let (Ok(i), Ok(o)) = (std::env::var("HMS_STAMP_IN"), std::env::var("HMS_STAMP_OUT")) else { return };
        let v = parse_h4_variant(Path::new(&i)).unwrap();
        let mut e = H4HeaderEdits::default();
        e.stamp_new(0);
        save_objects(Path::new(&i), Path::new(&o), &v.objects, Some(&e)).unwrap();
        let w = parse_h4_variant(Path::new(&o)).unwrap();
        eprintln!("{o}: {} objects, created {:?}, modified {:?}", w.objects.len(), w.created, w.modified);
    }

    /// THE gate: every shipped file re-encodes BYTE-IDENTICAL (BLF wrapper, chdr, SHA-1, length,
    /// bitstream, _eof, _fsm) from its decoded object list with no edits; the stored chunk length
    /// equals ceil(end_bit / 8) with the trait records measured by `skip_trait_set`; the stored
    /// SHA-1 equals `mvar_chunk_hash`; the `chdr` strings mirror the bitstream header.
    #[test]
    fn shipped_variants_reencode_byte_identical() {
        let files = all_variant_files();
        if files.is_empty() { eprintln!("skip: no halo4 variant folders"); return; }
        let (mut gaps, mut far_parents, mut compressed_labels, mut stale_tail) = (0usize, 0usize, 0usize, 0usize);
        for f in &files {
            let data = std::fs::read(f).unwrap();
            let (payload, major, _) = blf_h4_mvar_chunk(&data).unwrap();
            let v = parse_h4_payload(payload, major).unwrap();
            let stored_len = mvar_chunk_length(&data).unwrap();
            assert_eq!(((v.end_bit + 7) / 8) as u32, stored_len, "{}: trait records end at bit {} but the chunk length is {}", f.display(), v.end_bit, stored_len);
            let hash_off = data.windows(4).position(|w| w == b"mvar").unwrap() + 12;
            assert_eq!(&data[hash_off..hash_off + 20], &mvar_chunk_hash(payload, stored_len)[..], "{}: SHA-1", f.display());
            assert_eq!(detect_hash_kind(&data), Some(H4HashKind::PlainLe));
            // A file most recently saved by something other than this codec (observed: the game
            // itself, mid-session, on a file the user keeps editing in Forge) can leave non-zero
            // bytes past its own declared length - the region shrank since an earlier save and
            // the writer never cleared its old tail. Length and hash are still self-consistent
            // (both asserts above passed), so the file is genuine; it just cannot be byte-
            // identical to a fresh re-encode, which always zero-pads. Counted, not asserted -
            // exactly the "stale" tolerance the Reach sibling test already has.
            let tail_dirty = payload[stored_len as usize..H4_MVAR_BITSTREAM_BYTES].iter().any(|&b| b != 0);
            if tail_dirty {
                stale_tail += 1;
                let (re, slots) = rebuild_h4_file(&data, &v.objects, None).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
                assert_eq!(re.len(), data.len(), "{}: size", f.display());
                assert!(slots.iter().zip(v.objects.iter()).all(|(s, o)| *s == o.slot), "{}: slot assignment", f.display());
                // the re-encode must still decode back to the same objects, even though its bytes
                // differ from the dirty-tailed source
                let (re_payload, re_major, _) = blf_h4_mvar_chunk(&re).unwrap();
                let re_v = parse_h4_payload(re_payload, re_major).unwrap();
                assert_eq!(re_v.objects.len(), v.objects.len(), "{}: re-decoded object count", f.display());
                continue;
            }
            let (re, slots) = rebuild_h4_file(&data, &v.objects, None).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
            assert_eq!(re.len(), data.len(), "{}: size", f.display());
            if let Some(i) = data.iter().zip(re.iter()).position(|(a, b)| a != b) { panic!("{}: re-encode differs at byte {i} (bit {})", f.display(), i * 8); }
            assert!(slots.iter().zip(v.objects.iter()).all(|(s, o)| *s == o.slot));
            // chdr mirror
            let b = chdr_body(&data);
            assert_eq!(chdr_str(b, CHDR_AUTHOR, 16), v.author, "{}: chdr author", f.display());
            assert_eq!(chdr_str(b, CHDR_EDITOR, 16), v.editor, "{}: chdr editor", f.display());
            assert_eq!(chdr_wstr(b, CHDR_TITLE, 128), v.title_key, "{}: chdr title", f.display());
            assert_eq!(chdr_wstr(b, CHDR_DESCRIPTION, 128), v.description_key, "{}: chdr description", f.display());
            // census for the report
            if v.objects.iter().enumerate().any(|(i, o)| o.slot as usize != i) { gaps += 1; }
            if v.objects.iter().any(|o| o.parent >= H4_SLOTS as i16) { far_parents += 1; }
            let mut r = Bits::new(payload);
            r.pos = v.spans.labels.0;
            let n = r.bits(9) as usize;
            if n > 0 { for _ in 0..n { if r.flag() { r.bits(12); } } r.bits(13); if r.flag() { compressed_labels += 1; } }
        }
        eprintln!("h4 mvar encode: {} / {} files byte-identical ({} dirty-tailed, decode-checked only); {} with slot gaps, {} with parents >= 651, {} with a compressed label table",
            files.len() - stale_tail, files.len(), stale_tail, gaps, far_parents, compressed_labels);
    }

    /// The engine's own (data-derived) fixed points: quantise -> decode -> quantise is stable for
    /// positions inside the bounds, for orientations (global and tilted), and for the 6-bit real.
    #[test]
    fn quantisers_are_fixed_points() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f64 / (1u64 << 53) as f64 };
        let bounds = [-512.0f32, 412.0047 - 512.0, -300.0, 221.9852, -100.0, 161.4819];
        let ext = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]];
        let axis = h4_axis_bits(21, ext);
        for _ in 0..20_000 {
            for i in 0..3 {
                let p = bounds[2 * i] + rnd() as f32 * ext[i];
                let q = h4_quantize_pos(p, bounds[2 * i], bounds[2 * i + 1], axis[i]);
                let step = ext[i] / (1u64 << axis[i]) as f32;
                let d = bounds[2 * i] + q as f32 * step + step * 0.5;
                assert_eq!(h4_quantize_pos(d, bounds[2 * i], bounds[2 * i + 1], axis[i]), q, "axis {i} p {p} q {q} decoded {d}");
                assert!((d - p).abs() <= step, "axis {i}: {p} -> {d} (step {step})");
            }
            // clamping: outside the bounds lands on the edge buckets
            assert_eq!(h4_quantize_pos(bounds[0] - 50.0, bounds[0], bounds[1], axis[0]), 0);
            assert_eq!(h4_quantize_pos(bounds[1] + 50.0, bounds[0], bounds[1], axis[0]), (1 << axis[0]) - 1);
            // orientation: random up (tilted) + a perpendicular forward
            let mut up = [rnd() as f32 * 2.0 - 1.0, rnd() as f32 * 2.0 - 1.0, rnd() as f32 * 2.0 - 1.0];
            let m = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
            if m < 1e-3 { continue; }
            up = [up[0] / m, up[1] / m, up[2] / m];
            let t = [rnd() as f32 * 2.0 - 1.0, rnd() as f32 * 2.0 - 1.0, rnd() as f32 * 2.0 - 1.0];
            let d = t[0] * up[0] + t[1] * up[1] + t[2] * up[2];
            let mut fwd = [t[0] - d * up[0], t[1] - d * up[1], t[2] - d * up[2]];
            let fm = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt();
            if fm < 1e-3 { continue; }
            fwd = [fwd[0] / fm, fwd[1] / fm, fwd[2] / fm];
            let e1 = h4_encode_axes(fwd, up);
            let (f2, u2) = crate::mvar::decode_orientation(e1.0, e1.1, e1.2);
            assert_eq!(h4_encode_axes(f2, u2), e1, "tilted fixed point {fwd:?} {up:?}");
            let dotu = u2[0] * up[0] + u2[1] * up[1] + u2[2] * up[2];
            let dotf = f2[0] * fwd[0] + f2[1] * fwd[1] + f2[2] * fwd[2];
            assert!(dotu > 0.9999 && dotf > 0.9999, "orientation error: up {dotu} fwd {dotf}");
            // upright: global-up path
            let yaw = (rnd() * std::f64::consts::TAU - std::f64::consts::PI) as f32;
            let e2 = h4_encode_axes([yaw.cos(), yaw.sin(), 0.0], [0.0, 0.0, 1.0]);
            assert!(e2.0);
            let (f3, u3) = crate::mvar::decode_orientation(e2.0, e2.1, e2.2);
            assert_eq!(h4_encode_axes(f3, u3), e2);
            assert!((f3[0] - yaw.cos()).abs() < 1e-3 && (f3[1] - yaw.sin()).abs() < 1e-3);
        }
        for q in 0..64u32 {
            let v = dequantize_real(q, 0.0, 10.0, 6, false, true);
            assert_eq!(h4_quantize_real6(v) as u32, q, "scale {q} -> {v}");
        }
        assert_eq!(h4_quantize_real6(1.05), 7);
        assert_eq!(h4_quantize_scale(1.0), SCALE_Q_DEFAULT);
        assert!((h4_dequantize_scale(SCALE_Q_DEFAULT) - 1.0484).abs() < 1e-3, "the engine spawns a default object at 1.048");
        assert_eq!((h4_quantize_scale(0.0), h4_quantize_scale(10.0), h4_quantize_scale(-3.0), h4_quantize_scale(99.0)), (0, 63, 0, 63));
        assert_eq!(h4_quantize_scale(2.18), 14, "Vortex dominion shields");
        // the engine's unit-vector table constant equals the Reach formula for 20 bits; its
        // truncation puts a 0 component into bucket 207 ((0 + 1) / 0.0048076925 = 207.99999 in
        // f32), whose dequantised value is -0.0024 - an engine quirk, reproduced on purpose
        assert_eq!(h4_quantize_unit_vector([0.0, 0.0, -1.0]), 5 * 174_762 + 417 * 207 + 207);
        assert_eq!(h4_quantize_unit_vector([1.0, 0.0, 0.0]), 417 * 207 + 207);
    }

    /// The five stress-suite variants (skipping those not installed).
    fn stress_files() -> Vec<std::path::PathBuf> {
        let want = ["grifballcourt.mvar", "ca_forge_ravine_settler.mvar", "island_salem.mvar", "h4_impact_squad_monsoon.mvar", "dom_settler.mvar", "forge_hitchhiker.mvar"];
        let all = all_variant_files();
        let mut out = Vec::new();
        for w in want {
            if let Some(p) = all.iter().find(|p| p.file_name().map_or(false, |n| n == w)) { if !out.contains(p) { out.push(p.clone()); } }
            if out.len() == 5 { break; }
        }
        out
    }

    fn assert_object_eq(a: &H4PlacedObject, b: &H4PlacedObject, what: &str) {
        assert_eq!(a.flags, b.flags, "{what}: flags");
        assert_eq!(a.quota, b.quota, "{what}: quota");
        assert_eq!(a.variant, b.variant, "{what}: variant");
        assert_eq!(a.in_bounds, b.in_bounds, "{what}: in_bounds");
        assert_eq!(a.pos, b.pos, "{what}: pos");
        assert_eq!((a.up_is_global, a.up_quant, a.forward_angle_q), (b.up_is_global, b.up_quant, b.forward_angle_q), "{what}: orientation bits");
        assert_eq!((a.fwd, a.up), (b.fwd, b.up), "{what}: basis");
        assert_eq!(a.parent, b.parent, "{what}: parent");
        assert_eq!((a.scale_q, a.locked), (b.scale_q, b.locked), "{what}: scale/locked");
        assert_eq!((a.shape, a.shape_values), (b.shape, b.shape_values), "{what}: shape");
        assert_eq!((a.object_type, a.placement), (b.object_type, b.placement), "{what}: type/placement");
        assert_eq!((a.team, a.spawn_time, a.color, a.spawn_sequence, a.user_data), (b.team, b.spawn_time, b.color, b.spawn_sequence, b.user_data), "{what}: mp fields");
        assert_eq!(a.labels, b.labels, "{what}: labels");
        assert_eq!(a.type_data, b.type_data, "{what}: type data");
    }

    /// Save stress suite (the Reach gates ported): on five shipped variants delete (including a
    /// parent, which must orphan its children), move, rotate and tilt objects, add new ones
    /// (one parented to an existing object, one to another NEW object), edit every header
    /// string + the labels + the budget, save, reparse and check EVERY field of EVERY object,
    /// the quota table, the trait tail, the chunk length + SHA-1 and the chdr mirror; then save
    /// the reparsed list again with no edits (byte-identical = idempotent) and re-run the same
    /// edit from the source (byte-identical = deterministic).
    #[test]
    fn save_stress_suite() {
        let files = stress_files();
        if files.is_empty() { eprintln!("skip: no halo4 variant folders"); return; }
        let ids = map_ids();
        let dir = scratch_dir();
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let data = std::fs::read(f).unwrap();
            let v = parse_h4_variant(f).unwrap();
            let palette = ids.get(&v.map_id).and_then(|stem| open_map(stem)).map(|c| load_forge_palette(&c));
            let mut list = v.objects.clone();
            let n0 = list.len();
            assert!(n0 >= 20, "{name}: too small for the suite");
            // --- deletes: a parent with children (if any), plus every 7th object ---
            let mut deleted: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
            if let Some(p) = list.iter().find(|o| list.iter().any(|c| c.parent == o.slot as i16)).map(|o| o.slot) { deleted.insert(p); }
            for (i, o) in list.iter().enumerate() { if i % 7 == 3 { deleted.insert(o.slot); } }
            list.retain(|o| !deleted.contains(&o.slot));
            // --- moves (every 5th), rotations (every 3rd), tilts (every 4th), a colour + team ---
            for (i, o) in list.iter_mut().enumerate() {
                if i % 5 == 0 { o.pos = [o.pos[0] + 1.5, o.pos[1] - 2.25, o.pos[2] + 0.75]; }
                if i % 3 == 1 { let a = 0.37 * i as f32; o.set_orientation([a.cos(), a.sin(), 0.0], [0.0, 0.0, 1.0]); }
                if i % 4 == 2 { let a = 0.11 * i as f32; let up = [0.3 * a.sin(), 0.3 * a.cos(), 0.9]; let m = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt(); let up = [up[0] / m, up[1] / m, up[2] / m]; let f = [up[2], 0.0, -up[0]]; o.set_orientation(f, up); }
                // (ordnance drops, types 32..35, store none of these - see write_slot)
                if i % 11 == 5 && o.object_type < 32 { o.color = Some((i % 8) as u8); o.team = (i % 9) as i8; o.spawn_time = 45; o.spawn_sequence = ((i % 250) as u8) as i8; }
            }
            // --- adds: clone the palette identity of existing objects ---
            let proto = list[0].clone();
            let mut a1 = H4PlacedObject::new(proto.quota.unwrap(), proto.variant, proto.object_type, [proto.pos[0] + 3.0, proto.pos[1], proto.pos[2] + 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
            a1.parent = list[1].slot as i16; // parented to an existing object
            let mut a2 = H4PlacedObject::new(proto.quota.unwrap(), proto.variant, proto.object_type, [proto.pos[0] - 3.0, proto.pos[1] + 1.0, proto.pos[2]], [0.7071, 0.0, 0.7071], [-0.7071, 0.0, 0.7071]);
            a2.slot = 700; // a keyed new object other objects can name
            let mut a3 = H4PlacedObject::new(proto.quota.unwrap(), proto.variant, proto.object_type, proto.pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
            a3.parent = 700; // parented to the NEW object
            a3.shape = H4Shape::Box;
            a3.shape_values = [1792, 512, 512, 256];
            a3.labels[0] = Some(0);
            list.push(a1); list.push(a2); list.push(a3);
            let n_expect = list.len();
            // --- header edits ---
            let mut labels = v.labels.clone();
            labels.push("hms_test_label".to_string());
            let edits = H4HeaderEdits {
                title: Some(format!("HMS {name} title")),
                description: Some("Saved by the HMS Halo 4 encoder stress suite.".to_string()),
                author: Some("Sopitive".to_string()),
                editor: Some("hms-stress".to_string()),
                labels: Some(labels.clone()),
                budget_spent: palette.as_ref().map(|p| budget_spent(&list, p)),
                ..Default::default()
            };
            let dst1 = dir.join(format!("{name}.stress1.mvar"));
            let slots = save_objects(f, &dst1, &list, Some(&edits)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(slots.len(), n_expect);
            let out1 = std::fs::read(&dst1).unwrap();
            let s1 = parse_h4_variant(&dst1).unwrap_or_else(|e| panic!("{name}: reparse {e}"));
            // header
            assert_eq!(s1.title_key, edits.title.clone().unwrap());
            assert_eq!(s1.description_key, edits.description.clone().unwrap());
            assert_eq!((s1.author.as_str(), s1.editor.as_str()), ("Sopitive", "hms-stress"));
            assert_eq!(s1.labels, labels);
            if let Some(b) = edits.budget_spent { assert_eq!(s1.budget_spent, b); }
            assert_eq!((s1.map_id, s1.num_quotas, s1.bounds, s1.budget_max), (v.map_id, v.num_quotas, v.bounds, v.budget_max));
            let cb = chdr_body(&out1);
            assert_eq!(chdr_wstr(cb, CHDR_TITLE, 128), s1.title_key, "{name}: chdr title mirror");
            assert_eq!(chdr_wstr(cb, CHDR_DESCRIPTION, 128), s1.description_key);
            assert_eq!(chdr_str(cb, CHDR_AUTHOR, 16), "Sopitive");
            assert_eq!(chdr_str(cb, CHDR_EDITOR, 16), "hms-stress");
            // chunk length + hash + trait tail
            let stored_len = mvar_chunk_length(&out1).unwrap();
            assert_eq!(((s1.end_bit + 7) / 8) as u32, stored_len, "{name}: length");
            let (p1, _, _) = blf_h4_mvar_chunk(&out1).unwrap();
            let hoff = out1.windows(4).position(|w| w == b"mvar").unwrap() + 12;
            assert_eq!(&out1[hoff..hoff + 20], &mvar_chunk_hash(p1, stored_len)[..], "{name}: SHA-1");
            assert_eq!(s1.end_bit - s1.quota_end_bit, v.end_bit - v.quota_end_bit, "{name}: trait records changed size");
            let (src_payload, _, _) = blf_h4_mvar_chunk(&data).unwrap();
            let tail = |p: &[u8], from: usize, n: usize| -> Vec<u8> { let mut r = Bits::new(p); r.pos = from; (0..n).map(|_| r.bits(1) as u8).collect() };
            assert_eq!(tail(p1, s1.quota_end_bit, s1.end_bit - s1.quota_end_bit), tail(src_payload, v.quota_end_bit, v.end_bit - v.quota_end_bit), "{name}: trait bits");
            // every object, every field
            assert_eq!(s1.objects.len(), n_expect, "{name}: object count");
            for (i, o) in list.iter().enumerate() {
                let got = s1.objects.iter().find(|g| g.slot == slots[i]).unwrap_or_else(|| panic!("{name}: object {i} (slot {}) missing", slots[i]));
                let mut want = o.clone();
                want.pos = o.quantized_pos(&v);
                // retained objects keep their slots, so a surviving parent keeps its value; a
                // deleted parent orphans the child; key 700 is the new object's final slot
                want.parent = match o.parent {
                    -1 => -1,
                    700 => slots[list.iter().position(|x| x.slot == 700).unwrap()] as i16,
                    p if deleted.contains(&(p as u16)) => -1,
                    p => p,
                };
                if (o.slot as usize) < H4_SLOTS { assert_eq!(slots[i], o.slot, "{name}: object {i} moved slots"); }
                assert_object_eq(got, &want, &format!("{name}: object {i} slot {}", slots[i]));
            }
            // the deleted objects are gone: their slots are absent unless a NEW object took one
            let new_slots: Vec<u16> = slots[n_expect - 3..].to_vec();
            for d in &deleted { assert!(s1.objects.iter().all(|o| o.slot != *d) || new_slots.contains(d), "{name}: deleted slot {d} still present"); }
            // the new objects landed in the lowest free slots, in list order
            let mut free: Vec<u16> = (0..H4_SLOTS as u16).filter(|s| !list[..n_expect - 3].iter().any(|o| o.slot == *s)).collect();
            free.truncate(3);
            assert_eq!(new_slots, free, "{name}: new slots");
            // quota table
            let mut per_quota = vec![0u8; 256];
            for o in &s1.objects { per_quota[o.quota.unwrap() as usize] += 1; }
            for (i, (q, q0)) in s1.quotas.iter().zip(v.quotas.iter()).enumerate() {
                assert_eq!(q.placed, per_quota[i], "{name}: quota {i} placed");
                assert_eq!(q.minimum, q0.minimum, "{name}: quota {i} minimum");
                assert_eq!(q.maximum, q0.maximum.max(per_quota[i]), "{name}: quota {i} maximum");
            }
            // idempotent: saving the reparsed list with no edits reproduces the file byte for byte
            let dst2 = dir.join(format!("{name}.stress2.mvar"));
            let slots2 = save_objects(&dst1, &dst2, &s1.objects, None).unwrap();
            assert_eq!(std::fs::read(&dst2).unwrap(), out1, "{name}: second save differs");
            assert!(slots2.iter().zip(s1.objects.iter()).all(|(s, o)| *s == o.slot));
            // deterministic: the same edit from the source again
            let dst3 = dir.join(format!("{name}.stress3.mvar"));
            save_objects(f, &dst3, &list, Some(&edits)).unwrap();
            assert_eq!(std::fs::read(&dst3).unwrap(), out1, "{name}: re-running the edit differs");
            // header-only rewrite keeps the objects bit-exact
            let dst4 = dir.join(format!("{name}.header.mvar"));
            write_header(f, &dst4, &H4HeaderEdits { title: Some("t".into()), ..Default::default() }).unwrap();
            let s4 = parse_h4_variant(&dst4).unwrap();
            assert_eq!(s4.title_key, "t");
            assert_eq!(s4.objects.len(), v.objects.len());
            for (a, b) in s4.objects.iter().zip(v.objects.iter()) { assert_object_eq(a, b, &format!("{name}: header-only rewrite")); }
            // error paths: duplicate slot keys, a full variant
            let mut dup = v.objects.clone();
            dup.push(v.objects[0].clone());
            assert!(rebuild_h4_file(&data, &dup, None).is_err(), "{name}: duplicate slot accepted");
            let mut full = v.objects.clone();
            for _ in 0..(H4_SLOTS - v.objects.len() + 1) { full.push(H4PlacedObject::new(0, None, 0, v.objects[0].pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])); }
            assert!(rebuild_h4_file(&data, &full, None).is_err(), "{name}: 652 objects accepted");
            full.pop();
            let (_, fs) = rebuild_h4_file(&data, &full, None).unwrap();
            assert_eq!(fs.iter().copied().collect::<std::collections::BTreeSet<u16>>().len(), H4_SLOTS);
            eprintln!("h4 mvar stress: {name}: {n0} objects -> {} deleted, 3 added, {} saved; {} bits", deleted.len(), n_expect, s1.end_bit);
        }
    }

    /// `--h4-mvar-roundtrip` report on a shipped file.
    #[test]
    fn roundtrip_report_is_identical() {
        let Some(f) = stress_files().into_iter().next() else { return };
        let r = roundtrip_report(&f, Some(&scratch_dir().join("roundtrip.mvar"))).unwrap();
        assert!(r.contains("BYTE-IDENTICAL"), "{r}");
    }

    /// Every variant the GAME saved on this machine (LocalFiles under each MCC
    /// profile + HMS_H4_MVAR_DIRS): body version 51, map GUID == the engine's synthesised value
    /// for the map id, chunk length == ceil(end_bit / 8), byte-identical re-encode; then an
    /// edit (delete / move / rotate / add / header) saved, reparsed field-exact, saved again
    /// byte-identical, STILL version 51 with the same GUID.
    #[test]
    fn game_saved_variants_are_v51_and_roundtrip() {
        let files = user_variant_files();
        if files.is_empty() { eprintln!("skip: no game-saved Halo 4 variants (LocalFiles / HMS_H4_MVAR_DIRS)"); return; }
        let dir = scratch_dir();
        let mut versions: BTreeMap<u8, usize> = BTreeMap::new();
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let raw = std::fs::read(f).unwrap();
            let data = crate::mvar::expand_blf_cmp(&raw);
            let compressed = data != raw;
            let Some((payload, major, _)) = blf_h4_mvar_chunk(&data) else { eprintln!("{name}: no mvar chunk (not Halo 4)"); continue };
            assert!(is_h4_variant(f) && read_h4_summary(f).is_some(), "{name}: browser classification");
            let v = parse_h4_payload(payload, major).unwrap_or_else(|e| panic!("{name}: {e}"));
            *versions.entry(v.version).or_default() += 1;
            if v.version >= 51 {
                assert_eq!(v.map_guid, h4_map_guid(v.map_id), "{name}: map guid vs the engine table for map id {}", v.map_id);
            }
            assert_eq!(((v.end_bit + 7) / 8) as u32, mvar_chunk_length(&data).unwrap(), "{name}: length");
            let kind = detect_hash_kind(&data);
            assert!(kind.is_some(), "{name}: the chunk hash is neither the MCC nor the 360 flavour");
            let (re, _) = rebuild_h4_file(&data, &v.objects, None).unwrap();
            assert_eq!(re, data, "{name}: no-edit re-encode differs");
            // an edit round trip
            let mut list = v.objects.clone();
            let deleted = list.remove(list.len() / 2).slot;
            list[0].pos[0] += 2.5;
            list[1].set_orientation([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]);
            let proto = list[2].clone();
            list.push(H4PlacedObject::new(proto.quota.unwrap(), proto.variant, proto.object_type, [proto.pos[0], proto.pos[1] + 4.0, proto.pos[2]], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]));
            let edits = H4HeaderEdits { title: Some(format!("{} (HMS)", v.title_key)), ..Default::default() };
            let dst = dir.join(format!("{name}.v51edit.mvar"));
            let slots = save_objects(f, &dst, &list, Some(&edits)).unwrap();
            let s1 = parse_h4_variant(&dst).unwrap();
            assert_eq!((s1.version, s1.map_guid, s1.map_id, s1.num_quotas), (v.version, v.map_guid, v.map_id, v.num_quotas), "{name}: header after the edit");
            assert_eq!(s1.title_key, edits.title.clone().unwrap());
            assert_eq!(s1.objects.len(), list.len());
            assert!(s1.objects.iter().all(|o| o.slot != deleted) || slots.contains(&deleted));
            for (i, o) in list.iter().enumerate() {
                let got = s1.objects.iter().find(|g| g.slot == slots[i]).unwrap();
                let mut want = o.clone();
                want.pos = o.quantized_pos(&v);
                if o.parent >= 0 && o.parent as u16 == deleted { want.parent = -1; }
                assert_object_eq(got, &want, &format!("{name}: object {i}"));
            }
            let dst2 = dir.join(format!("{name}.v51edit2.mvar"));
            save_objects(&dst, &dst2, &s1.objects, None).unwrap();
            assert_eq!(std::fs::read(&dst2).unwrap(), std::fs::read(&dst).unwrap(), "{name}: second save differs");
            assert_eq!(detect_hash_kind(&std::fs::read(&dst).unwrap()), kind, "{name}: the edited save must keep the source's hash flavour");
            eprintln!("h4 mvar game-saved: {name}: body v{} map id {} ({} objects{}) hash {:?} round-trips; edit save field-exact", v.version, v.map_id, v.objects.len(), if compressed { ", _cmp-compressed source" } else { "" }, kind);
        }
        eprintln!("h4 mvar game-saved: {} files, versions {:?}", files.len(), versions);
        assert!(versions.keys().all(|k| *k == 50 || *k == 51));
    }

    /// Every record the encoder writes carries an EXPLICIT variant index: a new object of a
    /// single-variant entry (variant 0) and a decoded record whose index is NONE both come
    /// back as `Some(0)`, byte-for-byte what the game writes
    /// (0 of 146 647 shipped objects store NONE). Shipped-file identity is untouched
    /// (`shipped_variants_reencode_byte_identical`).
    #[test]
    fn saved_records_always_carry_a_variant_index() {
        let Some(mv) = variant_dirs().into_iter().find(|d| d.ends_with("map_variants")) else { eprintln!("skip"); return };
        let src = mv.join("ca_forge_ravine_settler.mvar");
        if !src.is_file() { eprintln!("skip: {} missing", src.display()); return }
        let v = parse_h4_variant(&src).unwrap();
        assert!(v.objects.iter().all(|o| o.variant.is_some()), "the shipped file stores an index on every record");
        let mut list = v.objects.clone();
        // a new Magnum (quota 0, the entry's only variant) placed the way the palette does it
        list.push(H4PlacedObject::new(0, None, 1, [10.0, 20.0, 5.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]));
        assert_eq!(list.last().unwrap().variant, Some(0));
        // a record with a NONE index (simulated on a decoded structure)
        let mut old = v.objects[0].clone();
        old.slot = NEW_SLOT;
        old.variant = None;
        list.push(old);
        let dst = scratch_dir().join("settler_variant_index.mvar");
        let slots = save_objects(&src, &dst, &list, None).unwrap();
        let back = parse_h4_variant(&dst).unwrap();
        assert_eq!(back.objects.len(), list.len());
        for (i, o) in list.iter().enumerate() {
            let got = back.objects.iter().find(|g| g.slot == slots[i]).unwrap();
            assert_eq!(got.variant, Some(o.variant.unwrap_or(0)), "slot {}: explicit variant index", slots[i]);
            assert_eq!(got.quota, o.quota);
            assert_eq!((got.flags, got.in_bounds, got.parent, got.scale_q, got.locked), (1, true, -1, SCALE_Q_DEFAULT, o.locked), "slot {}: grab-relevant defaults", slots[i]);
        }
        assert!(back.objects.iter().all(|o| o.variant.is_some()), "no NONE index survives a save");
    }

    // ---- global fields ---------------------------------------------------------------------

    /// Every Halo 4 variant on this machine: the content header's global fields agree with the
    /// little-endian `chdr` mirror (type, file size, the four ids, activity / game mode /
    /// engine, map id, category, both stamps + online flags), the body map id equals the
    /// header's, file-size is the `_eof` end, and the `globals()` view carries the chunk facts.
    #[test]
    fn globals_mirror_the_chdr_on_every_h4_variant() {
        // (shipped files only: a user save folder can hold tool-written files with a garbage
        // chdr - one here stores its file-size byte-swapped)
        let files: Vec<std::path::PathBuf> = all_variant_files();
        if files.is_empty() { eprintln!("skip: no halo4 variant folders"); return; }
        let mut checked = 0;
        for f in &files {
            let Ok(data) = read_h4_expanded(f) else { continue };
            let Ok(v) = parse_h4_variant(f) else { continue };
            let g = v.globals();
            let b = chdr_body(&data);
            let le32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            let le64 = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
            assert_eq!(g.game, "Halo 4");
            assert_eq!(g.content_type as i32, le32(0x04) as i32, "{}: type", f.display());
            assert_eq!(g.content_type, 5);
            assert_eq!(g.file_size, le32(0x08), "{}: file-size", f.display());
            assert_eq!(g.uid, le64(0x0C), "{}: uid", f.display());
            assert_eq!(g.parent_uid, le64(0x14), "{}: parent-uid", f.display());
            assert_eq!(g.root_uid, le64(0x1C), "{}: root-uid", f.display());
            assert_eq!(g.game_id, le64(0x24), "{}: game-id", f.display());
            assert_eq!(g.activity, b[0x2C] as i8, "{}: activity", f.display());
            assert_eq!(g.game_mode, b[0x2D], "{}: game-mode", f.display());
            assert_eq!(g.engine, b[0x2E], "{}: engine", f.display());
            assert_eq!(g.header_map_id, le32(CHDR_MAP_ID), "{}: map-id", f.display());
            assert_eq!(g.category, b[CHDR_CATEGORY] as i8, "{}: category", f.display());
            assert_eq!(v.created, (le64(CHDR_CREATED), le64(CHDR_CREATED + 8)));
            assert_eq!(g.created_online, b[0x5C] & 1 != 0, "{}: created online", f.display());
            assert_eq!(v.modified, (le64(CHDR_MODIFIED), le64(CHDR_MODIFIED + 8)));
            assert_eq!(g.modified_online, b[0x80] & 1 != 0, "{}: modified online", f.display());
            assert_eq!(g.map_id, g.header_map_id);
            assert!(g.hopper_id.is_none());
            let eof = data.windows(4).position(|w| w == b"_eof").unwrap();
            assert_eq!(g.file_size as usize, eof + 17, "{}: file-size = end of _eof", f.display());
            assert_eq!(g.chunk_length, mvar_chunk_length(&data).unwrap());
            assert!(g.hash_ok, "{}: SHA-1", f.display());
            assert_eq!(g.end_bit, v.end_bit);
            assert_eq!(g.quotas.len(), v.quotas.len());
            checked += 1;
        }
        eprintln!("h4 globals: {checked} files mirror their chdr");
        assert!(checked > 0);
    }

    /// Field-exact edits of the newly editable Halo 4 globals (category, maximum budget, quota
    /// minimum / maximum, world bounds) on the shipped Forge canvases: the value reads back
    /// (category in the chdr mirror too), every other global and every object is unchanged,
    /// `placed_on_map` is not taken from the edit and `maximum_count` is still raised to it,
    /// a re-save of the edited file with no edits is a fixed point, a widened box moves no
    /// object by more than a quantisation step, an inverted box is refused.
    #[test]
    fn global_field_edits_are_exact_and_isolated_h4() {
        let files: Vec<std::path::PathBuf> = all_variant_files().into_iter().filter(|f| f.file_name().map_or(false, |n| n.to_string_lossy().starts_with("ca_forge_"))).collect();
        if files.is_empty() { eprintln!("skip: no halo4 variant folders"); return; }
        let mut checked = 0;
        for f in &files {
            let data = read_h4_expanded(f).unwrap();
            let v = parse_h4_variant(f).unwrap();
            if v.objects.is_empty() || v.quotas.is_empty() { continue; }
            let g = v.globals();
            let same_but = |a: &crate::mvar::VariantGlobals, b: &crate::mvar::VariantGlobals, what: &str| {
                let (mut x, mut y) = (a.clone(), b.clone());
                x.category = 0; y.category = 0; x.budget_max = 0; y.budget_max = 0; x.bounds = [0.0; 6]; y.bounds = [0.0; 6];
                x.quotas.clear(); y.quotas.clear(); x.end_bit = 0; y.end_bit = 0; x.chunk_length = 0; y.chunk_length = 0;
                assert_eq!(x, y, "{}: {what}: an untouched global changed", f.display());
            };
            let tmp = std::env::temp_dir().join(format!("hms_h4_globals_{}.mvar", std::process::id()));
            let mut e = H4HeaderEdits::default();
            e.category = Some(3);
            e.budget_max = Some(12_345);
            let mut q: Vec<(u8, u8)> = v.quotas.iter().map(|q| (q.minimum, q.maximum)).collect();
            q[0] = (2, 200);
            let last = q.len() - 1;
            q[last] = (0, 0);
            e.quotas = Some(q);
            let (out, _) = rebuild_h4_file(&data, &v.objects, Some(&e)).unwrap();
            std::fs::write(&tmp, &out).unwrap();
            let w = parse_h4_variant(&tmp).unwrap();
            assert_eq!(w.category, 3);
            assert_eq!(chdr_body(&out)[CHDR_CATEGORY] as i8, 3, "chdr category mirror");
            assert_eq!(w.budget_max, 12_345);
            assert_eq!((w.quotas[0].minimum, w.quotas[0].maximum, w.quotas[0].placed), (2, 200, v.quotas[0].placed));
            assert_eq!(w.quotas[last].maximum, v.quotas[last].placed, "maximum raised to placed");
            for i in 1..last { assert_eq!(w.quotas[i], v.quotas[i]); }
            assert_eq!(w.bounds, v.bounds);
            assert_eq!(w.objects.len(), v.objects.len());
            for (a, b) in w.objects.iter().zip(v.objects.iter()) { let (mut x, mut y) = (a.clone(), b.clone()); x.bit = 0; y.bit = 0; assert_eq!(format!("{x:?}"), format!("{y:?}"), "{}: an object changed", f.display()); }
            assert_eq!(w.labels, v.labels);
            assert_eq!(w.end_bit, v.end_bit);
            assert!(w.hash_ok);
            assert_eq!(((w.end_bit + 7) / 8) as u32, w.chunk_length);
            same_but(&w.globals(), &g, "category/budget/quota");
            let (again, _) = rebuild_h4_file(&out, &w.objects, None).unwrap();
            assert_eq!(again, out, "{}: re-save of the edited file drifted", f.display());
            // bounds: widen 25% per axis
            let mut nb = v.bounds;
            for i in 0..3 { let ext = nb[2 * i + 1] - nb[2 * i]; nb[2 * i] -= 0.25 * ext; nb[2 * i + 1] += 0.25 * ext; }
            let mut eb = H4HeaderEdits::default();
            eb.bounds = Some(nb);
            let (outb, _) = rebuild_h4_file(&data, &v.objects, Some(&eb)).unwrap();
            std::fs::write(&tmp, &outb).unwrap();
            let wb = parse_h4_variant(&tmp).unwrap();
            assert_eq!(wb.bounds, nb);
            assert_eq!(wb.objects.len(), v.objects.len());
            let old_axis = h4_axis_bits(21, [v.bounds[1] - v.bounds[0], v.bounds[3] - v.bounds[2], v.bounds[5] - v.bounds[4]]);
            let new_axis = h4_axis_bits(21, [nb[1] - nb[0], nb[3] - nb[2], nb[5] - nb[4]]);
            for (a, b) in wb.objects.iter().zip(v.objects.iter()) {
                for i in 0..3 {
                    let os = (v.bounds[2 * i + 1] - v.bounds[2 * i]) / (1u32 << old_axis[i]) as f32;
                    let ns = (nb[2 * i + 1] - nb[2 * i]) / (1u32 << new_axis[i]) as f32;
                    assert!((a.pos[i] - b.pos[i]).abs() <= os + ns, "{}: object moved {} on axis {i}", f.display(), (a.pos[i] - b.pos[i]).abs());
                }
                let (mut x, mut y) = (a.clone(), b.clone()); x.bit = 0; y.bit = 0; x.pos = [0.0; 3]; y.pos = [0.0; 3];
                assert_eq!(format!("{x:?}"), format!("{y:?}"), "{}: a non-position field changed with the bounds", f.display());
            }
            assert!(wb.hash_ok);
            same_but(&wb.globals(), &g, "bounds");
            let (againb, _) = rebuild_h4_file(&outb, &wb.objects, None).unwrap();
            assert_eq!(againb, outb, "{}: re-save of the re-boxed file drifted", f.display());
            let mut bad = H4HeaderEdits::default();
            bad.bounds = Some([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
            assert!(rebuild_h4_file(&data, &v.objects, Some(&bad)).is_err());
            let _ = std::fs::remove_file(&tmp);
            checked += 1;
        }
        assert!(checked > 0, "no ca_forge_ canvas exercised the global edits");
    }

    /// The ported map-id table agrees with the shipped variants' map ids (every
    /// shipped map id is in it) and with the one known game-saved GUID.
    #[test]
    fn map_index_table_covers_the_shipped_maps() {
        for f in all_variant_files() {
            let id = read_h4_map_id(&f).unwrap();
            assert!(h4_map_index(id).is_some(), "{}: map id {id} missing from sub_180072E34", f.display());
        }
        assert_eq!(h4_map_index(14100), Some(127));
        assert_eq!(h4_map_guid(14100).map(|g| g[..8].to_vec()), Some(vec![0x7f, 0, 0, 0, 0, 0, 0x88, 0x88]));
        assert_eq!(h4_map_index(1), None);
    }
}

