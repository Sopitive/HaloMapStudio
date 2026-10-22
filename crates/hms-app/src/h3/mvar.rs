//! Halo 3 (MCC) `.mvar` map-variant reader (read-only). #h3
//!
//! The file is a BLF container (big-endian chunk headers: sig[4], size[4] INCLUDING the 12-byte
//! header, major[2], minor[2]): `_blf`, `athr`, `mvar`, `_eof`. The `mvar` chunk (major 12) holds
//! ONE MSB-first bitstream from its byte 12 to its end - the content header immediately followed
//! by the map variant, exactly as `halo3.dll` writes and reads it. Nothing here is inferred from
//! value distributions: every field width below is the shipped DLL's (IDA on halo3.dll, MCC
//! build "Dec 21 2023"):
//!
//!   file -> chunk walk + decode      sub_1800E18F4  finds "mvar", bitstream over [chunk+12, end)
//!   c_map_variant::decode            sub_1800B97C8  (content header first, via sub_18010BCE0)
//!   s_content_item_metadata decode   sub_18010BCE0
//!   c_map_variant::encode            sub_1800B8E9C  (the symmetric writer: absent-slot rule)
//!   dequantize_real                  sub_1800A35F8  lerp(min, max, q / ((1<<bits)-1)), q=0 -> min
//!   read_point3d (position)          sub_18004E8D8  3 x 16 bits over the world bounds
//!   read_axes (forward / up)         sub_18011091C  up: 1 flag or 19-bit unit vector; fwd: 8-bit
//!                                                   angle over [-pi, pi] about `up` from the
//!                                                   engine's perpendicular seed (sub_18010FEAC,
//!                                                   Rodrigues rotate sub_18009FB70)
//!   dequantize_unit_vector3d         sub_1801F7A40  3-bit face + two 8-bit [-1, 1] components
//!   angle range / boundary range     .rdata 0x847700 (-pi) / 0x8471A8 (pi) / 0x847374 (60.0)
//!   quantize_real                    sub_1800A3554  round-half-away((v-min)/(range/steps))
//!   write_point3d                    sub_18004E758  clamp into the bounds, then quantize_real
//!   write_axes                       sub_1801102E8  up within 1e-4 of +Z -> flag; else 19-bit
//!                                                   quantize_unit_vector3d (sub_1801F791C);
//!                                                   angle = atan2(fwd . left0, fwd . fwd0)
//!                                                   (sub_18010FFD8) over the seed basis
//! The encoder below is the writer (sub_1800B8E9C) field for field; every shipped file
//! re-encodes byte-identical from its decoded records (`shipped_h3_variants_reencode_byte_identical`).
//! The rest of the BLF (`_blf`, `athr`, `_eof` with the byte offset of itself, `_fsm`) is copied
//! from the source file with `_eof` re-stamped; the `_fsm` signature block is left as found
//! (the 360-era files carry an all-zero one; MCC ignores it for local files, as for Reach).
//!
//! Content header (s_content_item_metadata, in bitstream order): unique_id 64, name wchar[16]
//! (null-stopped, 16-bit chars), description char[128], author char[16], file_type 5 (-1),
//! author_is_online 1, author_xuid 64, size_bytes 64, date 64, length_seconds 32, campaign_id 32,
//! map_id 32, game_engine_type 4, campaign_difficulty 3 (-1), hopper_id 16, game_id 64.
//!
//! Map variant body: version 8 (13 on every MCC file), checksum 32, number_of_scenario_objects
//! 10, number_of_variant_objects 10 (<= 640), number_of_placeable_object_quotas 9 (<= 256),
//! map_id 32, built_in 1, world bounds 6 x f32 (xmin xmax ymin ymax zmin zmax),
//! game_engine_subtype 4, maximum_budget f32, spent_budget f32; then `number_of_variant_objects`
//! object slots (slot i >= that count is never read); then object_type_start_index[14] as 9 bits
//! each (+1 stored); then `number_of_placeable_object_quotas` quotas (object definition tag
//! reference 32 - see `H3Quota::definition` -, minimum 8, maximum 8, placed_on_map 8,
//! maximum_allowed 8, price f32); a version >= 14 stream (MCC re-saves: `h3_rats_nest_heavy`
//! and friends) ends with the 128-bit MCC map id (sub_180110A30).
//!
//! Object slot: present 1 (0 = empty slot); flags 16; variant_quota_index 32; has_parent 1
//! [+ parent object identifier 64]; has_placement 1 - CLEAR on the scenario's own objects
//! (the first `number_of_scenario_objects` slots, position fixed by the scenario) - [+ position
//! 3 x 16 over the bounds, axes, then the multiplayer properties: object_type 8, flags 8,
//! game_engine_flags 16, shared_storage 8, spawn_time 8, team 8, boundary_shape 8 and the
//! shape's 16-bit sizes over [0, 60]: sphere radius; cylinder radius / top / bottom; box width /
//! length / top / bottom].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use glam::{Mat4, Vec3, Vec4};

use super::cache::H3Cache;
use crate::h4::cache::ByteRead;

/// `mvar` chunk major version of a Halo 3 map variant.
pub const H3_MVAR_CHUNK_VERSION: u16 = 12;
/// Map-variant body version of the 360-era shipped files; an MCC re-save writes 14 (+ the
/// 128-bit MCC map id after the quotas; sub_1800BE1A8 bumps 13 -> 14 on load).
pub const H3_VARIANT_VERSION: u16 = 13;
pub const H3_VARIANT_VERSION_MCC: u16 = 14;
/// Object slots per variant (the DLL's loop bound).
pub const H3_SLOTS: usize = 640;
/// Quota entries per variant.
pub const H3_QUOTA_SLOTS: usize = 256;
/// Object types with a start index (object_type_start_index[14]).
pub const H3_OBJECT_TYPES: usize = 14;
/// Bits per position axis (`read_point3d(16)`).
pub const H3_POSITION_BITS: u32 = 16;
/// Boundary size range the 16-bit shape values span (halo3.dll .rdata 0x847374).
pub const H3_BOUNDARY_MAX: f32 = 60.0;
/// Content file type of a map variant ("usermap") in Halo 3's saved-game enum.
pub const H3_FILE_TYPE_USERMAP: i32 = 10;

// ---------------------------------------------------------------------------------------------
// bit reader (MSB first; out-of-range bits read as 0, and `overrun` records it)
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
    fn u16(&mut self) -> u16 { self.bits(16) as u16 }
    fn u32(&mut self) -> u32 { self.bits(32) as u32 }
    fn u64(&mut self) -> u64 { self.bits(64) }
    fn f32(&mut self) -> f32 { f32::from_bits(self.u32()) }
    fn overrun(&self) -> bool { self.pos > self.buf.len() * 8 }
    /// `read_string_wchar(max)`: 16-bit chars until a null (at most `max` read).
    fn wstr(&mut self, max: usize) -> String {
        let mut v = Vec::new();
        for _ in 0..max {
            let u = self.u16();
            if u == 0 { break; }
            v.push(u);
        }
        String::from_utf16_lossy(&v)
    }
    /// `read_string(max)`: bytes until a null (at most `max` read).
    fn cstr(&mut self, max: usize) -> String {
        let mut v = Vec::new();
        for _ in 0..max {
            let b = self.u8();
            if b == 0 { break; }
            v.push(b);
        }
        String::from_utf8_lossy(&v).into_owned()
    }
}

// ---------------------------------------------------------------------------------------------
// engine quantisation (halo3.dll)
// ---------------------------------------------------------------------------------------------

/// sub_1800A35F8: `q` over `bits` bits -> [min, max]. `exact_midpoint` drops the top step so the
/// range's centre is representable (the angle and unit-vector paths use it).
pub fn dequantize_real(q: u32, min: f32, max: f32, bits: u32, exact_midpoint: bool) -> f32 {
    let mut steps = (1u32 << bits) - 1;
    if exact_midpoint { steps -= steps % 2; }
    if q == 0 { return min; }
    if q >= steps { return max; }
    ((steps - q) as f32 * min + q as f32 * max) / steps as f32
}

/// sub_1801F7A40: a 19-bit unit vector = face (3 bits) + two 8-bit components over [-1, 1]
/// (exact midpoint), normalised.
pub fn dequantize_unit_vector(q: u32) -> Vec3 {
    let a = dequantize_real((q >> 3) & 0xFF, -1.0, 1.0, 8, true);
    let b = dequantize_real((q >> 11) & 0xFF, -1.0, 1.0, 8, true);
    let v = match q & 7 {
        0 => Vec3::new(1.0, a, b),
        1 => Vec3::new(a, 1.0, b),
        2 => Vec3::new(a, b, 1.0),
        3 => Vec3::new(-1.0, a, b),
        4 => Vec3::new(a, -1.0, b),
        5 => Vec3::new(a, b, -1.0),
        _ => Vec3::Z,
    };
    v.normalize_or_zero()
}

/// sub_18010FEAC: the engine's forward seed perpendicular to `up` (the |y| <= |x| split picks
/// the axis to cross against), normalised. The angle in the stream rotates it about `up`.
pub fn perpendicular_seed(up: Vec3) -> Vec3 {
    let v = if up.y.abs() <= up.x.abs() { Vec3::new(up.z, 0.0, -up.x) } else { Vec3::new(0.0, up.z, -up.y) };
    v.normalize_or_zero()
}

/// sub_18009FB70: rotate `v` about the unit axis `axis` by the angle with the given sine / cosine
/// (Rodrigues, right-handed: v' = axis (axis.v)(1-c) + v c + (axis x v) s).
pub fn rotate_about(v: Vec3, axis: Vec3, s: f32, c: f32) -> Vec3 {
    axis * (axis.dot(v) * (1.0 - c)) + v * c + axis.cross(v) * s
}

/// sub_1800A3554: `v` -> `bits` bits over [min, max]: round-half-away-from-zero of
/// (v - min) / (range / steps); `exact_midpoint` as in `dequantize_real`; `clamp_ends` keeps a
/// value strictly inside the range off the end steps (the unit-vector path).
pub fn quantize_real(v: f32, min: f32, max: f32, bits: u32, exact_midpoint: bool, clamp_ends: bool) -> u32 {
    let mut steps = (1u32 << bits) - 1;
    if exact_midpoint { steps -= steps % 2; }
    let x = (v - min) / ((max - min) / steps as f32);
    let half = if x < 0.0 { -0.5 } else { 0.5 };
    let mut q = (half + x) as i32; // cvttss2si: truncation toward zero
    if clamp_ends {
        if q == 0 && v != min { return 1; }
        if q == steps as i32 && v != max { q -= 1; }
    }
    q.max(0) as u32
}

/// sub_18004E758 / sub_1800A366C: one position axis - clamped into [min, max], then
/// `quantize_real(16, plain)`.
pub fn quantize_pos(v: f32, min: f32, max: f32) -> u16 {
    let c = if v <= min { min } else if max <= v { max } else { v };
    quantize_real(c, min, max, H3_POSITION_BITS, false, false).min(0xFFFF) as u16
}

/// sub_1801F791C: the 19-bit unit vector: face = the major axis (+ 3 when negative), the two
/// other components divided by the major's magnitude, 8 bits each over [-1, 1] (exact midpoint).
pub fn quantize_unit_vector(v: Vec3) -> u32 {
    let (ax, ay, az) = (v.x.abs(), v.y.abs(), v.z.abs());
    // (major axis, first = the component packed at bits 11.., second = the one at bits 3..)
    let (face, first, second) = if ax <= ay {
        if ay > az { (1, v.z / ay, v.x / ay) } else { (2, v.y / az, v.x / az) }
    } else if ax <= az { (2, v.y / az, v.x / az) } else { (0, v.z / ax, v.y / ax) };
    let neg = match face { 0 => v.x < 0.0, 1 => v.y < 0.0, _ => v.z < 0.0 };
    let q_first = quantize_real(first, -1.0, 1.0, 8, true, false);
    let q_second = quantize_real(second, -1.0, 1.0, 8, true, false);
    ((q_first << 8 | q_second) << 3) | (face + if neg { 3 } else { 0 })
}

/// sub_18010FFD8: the forward angle about `up` relative to the engine's seed basis
/// (`perpendicular_seed`, left = up x seed): atan2(fwd . left, fwd . seed).
pub fn forward_angle(fwd: Vec3, up: Vec3) -> f32 {
    let seed = perpendicular_seed(up);
    let left = up.cross(seed).normalize_or_zero();
    fwd.dot(left).atan2(fwd.dot(seed))
}

// ---------------------------------------------------------------------------------------------
// decoded structures
// ---------------------------------------------------------------------------------------------

/// The content header (s_content_item_metadata) of a Halo 3 variant.
#[derive(Clone, Debug, Default)]
pub struct H3ContentHeader {
    pub unique_id: u64,
    pub name: String,
    pub description: String,
    pub author: String,
    /// Saved-game file type (10 = usermap).
    pub file_type: i32,
    pub author_is_online: bool,
    pub author_xuid: u64,
    pub size_bytes: u64,
    /// FILETIME-style stamp as stored (seconds since the epoch on every shipped file).
    pub date: u64,
    pub length_seconds: u32,
    pub campaign_id: i32,
    pub map_id: u32,
    pub game_engine_type: u32,
    pub campaign_difficulty: i32,
    pub hopper_id: u16,
    pub game_id: u64,
}

/// One placeable-object quota (s_variant_quota).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct H3Quota {
    /// Object definition tag datum (`salt << 16 | tag index`) in the base map's tag table.
    pub definition: u32,
    pub minimum: u8,
    pub maximum: u8,
    pub placed_on_map: u8,
    pub maximum_allowed: u8,
    pub price: f32,
}

impl H3Quota {
    /// Tag index of the object definition (the low half of the datum).
    pub fn tag_index(&self) -> usize { (self.definition & 0xFFFF) as usize }
}

/// One object slot that is present in the stream (s_variant_object_datum).
#[derive(Clone, Debug, PartialEq)]
pub struct H3VariantObject {
    pub slot: usize,
    pub flags: u16,
    /// Quota index (-1 on the scenario's own objects).
    pub quota: i32,
    /// Parent object identifier when the slot is attached to one.
    pub parent: Option<u64>,
    /// The slot carries a placement (position / axes / multiplayer properties). Clear on the
    /// scenario's own objects, whose placement lives in the scenario.
    pub placed: bool,
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    /// Raw stream quanta behind `pos` / `fwd` / `up` / `boundary` (re-emitted verbatim by the
    /// encoder; `set_pose` / `set_boundary` recompute them): 16-bit position per axis, the
    /// up-is-global flag, the 19-bit up, the 8-bit angle, the 16-bit boundary sizes.
    pub pos_q: [u16; 3],
    pub up_is_global: bool,
    pub up_q: u32,
    pub angle_q: u8,
    pub boundary_q: [u16; 4],
    // multiplayer object properties (s_variant_multiplayer_object_properties_definition)
    pub object_type: u8,
    pub mp_flags: u8,
    pub game_engine_flags: u16,
    /// Spare clips / teleporter channel / spawn order, by object type.
    pub shared_storage: u8,
    pub spawn_time: u8,
    pub team: u8,
    /// 0 none, 1 sphere, 2 cylinder, 3 box.
    pub boundary_shape: u8,
    /// Sphere: radius; cylinder: radius, top, bottom; box: width, length, top, bottom.
    pub boundary: [f32; 4],
}

/// Boundary sizes (wu, clamped to [0, 60]) -> the 16-bit quanta a record stores.
pub fn quantize_boundary(values: [f32; 4]) -> [u16; 4] {
    let mut q = [0u16; 4];
    for k in 0..4 {
        let v = values[k].clamp(0.0, H3_BOUNDARY_MAX);
        q[k] = quantize_real(v, 0.0, H3_BOUNDARY_MAX, 16, false, false).min(0xFFFF) as u16;
    }
    q
}

impl H3VariantObject {
    /// Re-quantise a new pose the way the engine writes it (sub_1801102E8 / sub_18004E758) and
    /// store what the file will decode back, so the editor shows the saved pose.
    pub fn set_pose(&mut self, pos: [f32; 3], fwd: Vec3, up: Vec3, bounds: [f32; 6]) {
        for k in 0..3 {
            self.pos_q[k] = quantize_pos(pos[k], bounds[2 * k], bounds[2 * k + 1]);
            self.pos[k] = dequantize_real(self.pos_q[k] as u32, bounds[2 * k], bounds[2 * k + 1], H3_POSITION_BITS, false);
        }
        let up = up.normalize_or_zero();
        let fwd = fwd.normalize_or_zero();
        let global = (up.x - 0.0).abs() < 0.000099999997 && (up.y - 0.0).abs() < 0.000099999997 && (up.z - 1.0).abs() < 0.000099999997;
        let (dec_up, up_q) = if global { (Vec3::Z, 0) } else { let q = quantize_unit_vector(up); (dequantize_unit_vector(q), q) };
        self.up_is_global = global;
        self.up_q = up_q;
        // the engine measures the angle against the basis of the DECODED up (the writer's
        // sub_1801102E8 dequantises before sub_18010FFD8)
        let angle = forward_angle(fwd, dec_up);
        self.angle_q = quantize_real(angle, -std::f32::consts::PI, std::f32::consts::PI, 8, true, false) as u8;
        let angle = dequantize_real(self.angle_q as u32, -std::f32::consts::PI, std::f32::consts::PI, 8, true);
        let (s, c) = if angle == std::f32::consts::PI || angle == -std::f32::consts::PI { (0.0, -1.0) } else { angle.sin_cos() };
        self.up = dec_up.to_array();
        self.fwd = rotate_about(perpendicular_seed(dec_up), dec_up, s, c).normalize_or_zero().to_array();
    }

    /// Set the boundary sizes (wu, clamped to [0, 60]) through the 16-bit quantiser.
    pub fn set_boundary(&mut self, shape: u8, values: [f32; 4]) {
        self.boundary_shape = shape;
        self.boundary_q = quantize_boundary(values);
        for k in 0..4 { self.boundary[k] = dequantize_real(self.boundary_q[k] as u32, 0.0, H3_BOUNDARY_MAX, 16, false); }
    }

    /// World matrix: columns [fwd, up x fwd, up, pos] (local +X = forward, +Z = up), the same
    /// build as objects.rs `placement_matrix` for scenario placements.
    pub fn matrix(&self) -> Mat4 {
        let f = Vec3::from(self.fwd);
        let u = Vec3::from(self.up);
        let mut left = u.cross(f);
        if left.length_squared() < 1e-12 { left = Vec3::Y; }
        Mat4::from_cols(
            Vec4::new(f.x, f.y, f.z, 0.0),
            Vec4::new(left.x, left.y, left.z, 0.0),
            Vec4::new(u.x, u.y, u.z, 0.0),
            Vec4::new(self.pos[0], self.pos[1], self.pos[2], 1.0),
        )
    }
}

/// A decoded Halo 3 map variant.
#[derive(Clone, Debug, Default)]
pub struct H3Variant {
    pub header: H3ContentHeader,
    pub chunk_major: u16,
    pub chunk_minor: u16,
    pub version: u16,
    pub checksum: u32,
    pub num_scenario_objects: u16,
    pub num_variant_objects: u16,
    pub num_quotas: u16,
    /// The base map (equals the header's on every shipped file).
    pub map_id: u32,
    pub built_in: bool,
    /// xmin, xmax, ymin, ymax, zmin, zmax.
    pub bounds: [f32; 6],
    pub game_engine_subtype: u32,
    pub maximum_budget: f32,
    pub spent_budget: f32,
    /// Every PRESENT slot, in slot order (scenario-owned ones included: `placed` tells them apart).
    pub objects: Vec<H3VariantObject>,
    /// object_type_start_index[14] (-1 = none).
    pub type_start: [i16; H3_OBJECT_TYPES],
    pub quotas: Vec<H3Quota>,
    /// The MCC map id (version >= 14 streams only).
    pub mcc_map_id: Option<[u8; 16]>,
    /// Bit position just past the last field read (the stream's true length).
    pub end_bit: usize,
}

impl H3Variant {
    pub fn title(&self) -> &str { &self.header.name }
    /// The Forge-placed objects: present slots that carry a placement.
    pub fn placed(&self) -> impl Iterator<Item = &H3VariantObject> { self.objects.iter().filter(|o| o.placed) }
    /// Quota of a placed object.
    pub fn quota_of(&self, o: &H3VariantObject) -> Option<&H3Quota> {
        usize::try_from(o.quota).ok().and_then(|q| self.quotas.get(q))
    }
}

/// Rewrite the quota bookkeeping a save is responsible for: each quota's `placed_on_map` and the
/// variant's `spent_budget`. #h3-accounting
///
/// The engine's own bookkeeping is lossy and the shipped files do not agree with themselves
/// (measured over all 333: `placed_on_map` equals the live slot count on 87.3% of the 42 598 quota
/// rows, `spent_budget` equals the sum of the placed prices on 77 of 333 files, and no rule over the
/// slot flags / engine flags / variant engine recovers the rest), so there is no exact formula to
/// reproduce. What a save CAN do is keep the two honest about what it changed:
///
///   * `absolute = false` (an open variant being edited): the stored values move by the DIFFERENCE
///     between the records before and after, per quota, so an untouched file stays byte-identical
///     and an add / delete is reflected by exactly its own price / one count, quirks of the stored
///     value included;
///   * `absolute = true` (a blank canvas adopted from a template, whose Forge objects were dropped):
///     the stored values describe objects that are no longer there, so they are recounted from the
///     records being written.
///
/// A count is every present slot that names a quota (scenario-owned ones too: that rule fits the
/// most rows) except a base-map object that was deleted (halo3.dll sub_1800BA3F0 does not count a
/// scenario slot with `flags & 0x60`); the budget is the price of the slots that carry a Forge
/// placement (the base map's own objects are free). Neither goes
/// below zero, and a count saturates at the field's 255.
pub fn account_for_save(out: &mut H3Variant, before: &[H3VariantObject], after: &[H3VariantObject], absolute: bool) {
    let count = |objs: &[H3VariantObject]| -> HashMap<i32, u32> {
        let mut m: HashMap<i32, u32> = HashMap::new();
        for o in objs.iter().filter(|o| o.quota >= 0 && super::edit::counts_toward_quota(o)) { *m.entry(o.quota).or_default() += 1; }
        m
    };
    let price_of = |q: &[H3Quota], o: &H3VariantObject| -> f32 {
        // the base map's own objects are free, moved or not
        if !o.placed || o.flags & super::scenario::SLOT_SCENARIO != 0 { return 0.0; }
        usize::try_from(o.quota).ok().and_then(|i| q.get(i)).map_or(0.0, |q| q.price)
    };
    let (b, a) = (count(before), count(after));
    for (i, q) in out.quotas.iter_mut().enumerate() {
        let (nb, na) = (*b.get(&(i as i32)).unwrap_or(&0) as i64, *a.get(&(i as i32)).unwrap_or(&0) as i64);
        q.placed_on_map = if absolute { na } else { q.placed_on_map as i64 + (na - nb) }.clamp(0, 255) as u8;
    }
    let spent_before: f32 = before.iter().map(|o| price_of(&out.quotas, o)).sum();
    let spent_after: f32 = after.iter().map(|o| price_of(&out.quotas, o)).sum();
    // the difference is formed first, so "nothing changed" is exactly 0.0 and the stored value
    // is left bit-for-bit (a subtract-then-add could drift on a fractional price)
    out.spent_budget = if absolute { spent_after } else { (out.spent_budget + (spent_after - spent_before)).max(0.0) };
    // the stored figure saturates at the cap: 0 of the 333 shipped variants store more than their
    // maximum, though 11 of them hold objects priced past it (their stored value is exactly the cap)
    if out.maximum_budget > 0.0 { out.spent_budget = out.spent_budget.min(out.maximum_budget); }
}

/// The browser row's summary of a Halo 3 variant.
#[derive(Clone, Debug)]
pub struct H3VariantSummary {
    pub title: String,
    pub description: String,
    pub author: String,
    pub map_id: u32,
    /// Forge-placed objects (slots with a placement).
    pub objects: usize,
}

// ---------------------------------------------------------------------------------------------
// BLF container
// ---------------------------------------------------------------------------------------------

/// The `mvar` chunk of a Halo 3 BLF: (bitstream bytes = chunk body after its 12-byte header,
/// major, minor). Walks the big-endian chunk headers from the file start (sub_1800E18F4 walks the
/// same way, bounding every chunk size by 0xE0A0).
pub fn blf_h3_mvar_chunk(data: &[u8]) -> Option<(&[u8], u16, u16)> {
    let mut pos = 0usize;
    while pos + 12 <= data.len() {
        let magic = &data[pos..pos + 4];
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 12 || pos + size > data.len() { break; }
        if magic == b"mvar" {
            let major = u16::from_be_bytes([data[pos + 8], data[pos + 9]]);
            let minor = u16::from_be_bytes([data[pos + 10], data[pos + 11]]);
            return Some((&data[pos + 12..pos + size], major, minor));
        }
        if magic == b"_eof" { break; }
        pos += size;
    }
    None
}

/// Is this file a Halo 3 map variant (BLF with an `mvar` chunk of major version 12)? Reads the
/// chunk headers only; Reach (v31), Halo 4 (v50/51) and H2A (v52) files answer false.
pub fn is_h3_variant(path: &Path) -> bool {
    let Ok(data) = crate::mvar::read_expanded(path) else { return false };
    matches!(blf_h3_mvar_chunk(&data), Some((_, H3_MVAR_CHUNK_VERSION, _)))
}

/// Base map id of a Halo 3 variant from its content header (None for anything else).
pub fn read_h3_map_id(path: &Path) -> Option<u32> {
    let data = crate::mvar::read_expanded(path).ok()?;
    let (body, major, _) = blf_h3_mvar_chunk(&data)?;
    if major != H3_MVAR_CHUNK_VERSION { return None; }
    let mut r = Bits::new(body);
    Some(parse_content_header(&mut r).map_id)
}

/// Summary of a Halo 3 variant for the browser row, None for anything else. The object count
/// needs the slots, so the whole stream is decoded.
pub fn read_h3_summary(path: &Path) -> Option<H3VariantSummary> {
    let v = parse_h3_variant(path).ok()?;
    Some(H3VariantSummary {
        title: v.header.name.clone(), description: v.header.description.clone(), author: v.header.author.clone(),
        map_id: v.map_id, objects: v.placed().count(),
    })
}

/// Parse a Halo 3 `.mvar` file.
pub fn parse_h3_variant(path: &Path) -> Result<H3Variant> {
    let data = crate::mvar::read_expanded(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let (body, major, minor) = blf_h3_mvar_chunk(&data).ok_or_else(|| anyhow!("{}: no mvar chunk", path.display()))?;
    if major != H3_MVAR_CHUNK_VERSION { bail!("{}: mvar chunk v{major} is not a Halo 3 variant (v{H3_MVAR_CHUNK_VERSION})", path.display()); }
    let mut v = parse_h3_payload(body).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    v.chunk_major = major;
    v.chunk_minor = minor;
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// bitstream decode
// ---------------------------------------------------------------------------------------------

/// sub_18010BCE0.
fn parse_content_header(r: &mut Bits) -> H3ContentHeader {
    let unique_id = r.u64();
    let name = r.wstr(16);
    let description = r.cstr(128);
    let author = r.cstr(16);
    let file_type = r.bits(5) as i32 - 1;
    let author_is_online = r.flag();
    let author_xuid = r.u64();
    let size_bytes = r.u64();
    let date = r.u64();
    let length_seconds = r.u32();
    let campaign_id = r.u32() as i32;
    let map_id = r.u32();
    let game_engine_type = r.bits(4) as u32;
    let campaign_difficulty = r.bits(3) as i32 - 1;
    let hopper_id = r.u16();
    let game_id = r.u64();
    H3ContentHeader { unique_id, name, description, author, file_type, author_is_online, author_xuid, size_bytes, date,
        length_seconds, campaign_id, map_id, game_engine_type, campaign_difficulty, hopper_id, game_id }
}

/// sub_18011091C: (forward, up, up_is_global, up_q, angle_q).
fn read_axes(r: &mut Bits) -> (Vec3, Vec3, bool, u32, u8) {
    let up_is_global = r.flag();
    let (up, up_q) = if up_is_global { (Vec3::Z, 0) } else { let q = r.bits(19) as u32; (dequantize_unit_vector(q), q) };
    let angle_q = r.u8();
    let angle = dequantize_real(angle_q as u32, -std::f32::consts::PI, std::f32::consts::PI, 8, true);
    let seed = perpendicular_seed(up);
    // the engine skips sin/cos at +-pi and rotates with (s, c) = (0, -1) exactly
    let (s, c) = if angle == std::f32::consts::PI || angle == -std::f32::consts::PI { (0.0, -1.0) } else { angle.sin_cos() };
    let fwd = rotate_about(seed, up, s, c).normalize_or_zero();
    (fwd, up, up_is_global, up_q, angle_q)
}

/// Decode the `mvar` chunk body (content header + map variant).
pub fn parse_h3_payload(body: &[u8]) -> Result<H3Variant> {
    let mut r = Bits::new(body);
    let header = parse_content_header(&mut r);
    let version = r.u8() as u16;
    let checksum = r.u32();
    let num_scenario_objects = r.bits(10) as u16;
    let num_variant_objects = r.bits(10) as u16;
    let num_quotas = r.bits(9) as u16;
    if version > H3_VARIANT_VERSION_MCC { bail!("map variant version {version} (this reader knows <= {H3_VARIANT_VERSION_MCC})"); }
    if num_variant_objects as usize > H3_SLOTS || num_quotas as usize > H3_QUOTA_SLOTS {
        bail!("counts out of range: {num_variant_objects} objects / {num_quotas} quotas");
    }
    let map_id = r.u32();
    let built_in = r.flag();
    let mut bounds = [0f32; 6];
    for b in bounds.iter_mut() { *b = r.f32(); }
    let game_engine_subtype = r.bits(4) as u32;
    let maximum_budget = r.f32();
    let spent_budget = r.f32();
    // a variant saved with no objects carries the engine's EMPTY box (+FLT_MAX .. -FLT_MAX);
    // the DLL dequantises against it all the same, and no placed slot reads it here
    if !bounds.iter().all(|v| v.is_finite()) { bail!("world bounds {bounds:?} are not finite"); }
    let mut objects = Vec::new();
    for slot in 0..num_variant_objects as usize {
        if !r.flag() { continue; }
        let flags = r.u16();
        let quota = r.u32() as i32;
        let parent = if r.flag() { Some(r.u64()) } else { None };
        let mut o = H3VariantObject { slot, flags, quota, parent, placed: false, pos: [0.0; 3], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
            pos_q: [0; 3], up_is_global: true, up_q: 0, angle_q: 0, boundary_q: [0; 4], object_type: 0, mp_flags: 0, game_engine_flags: 0,
            shared_storage: 0, spawn_time: 0, team: 0, boundary_shape: 0, boundary: [0.0; 4] };
        if r.flag() {
            o.placed = true;
            for k in 0..3 {
                let q = r.bits(H3_POSITION_BITS) as u32;
                o.pos_q[k] = q as u16;
                o.pos[k] = dequantize_real(q, bounds[2 * k], bounds[2 * k + 1], H3_POSITION_BITS, false);
            }
            let (fwd, up, up_is_global, up_q, angle_q) = read_axes(&mut r);
            o.fwd = fwd.to_array();
            o.up = up.to_array();
            o.up_is_global = up_is_global;
            o.up_q = up_q;
            o.angle_q = angle_q;
            o.object_type = r.u8();
            o.mp_flags = r.u8();
            o.game_engine_flags = r.u16();
            o.shared_storage = r.u8();
            o.spawn_time = r.u8();
            o.team = r.u8();
            o.boundary_shape = r.u8();
            let n = match o.boundary_shape { 1 => 1, 2 => 3, 3 => 4, _ => 0 };
            for k in 0..n {
                let q = r.u16();
                o.boundary_q[k] = q;
                o.boundary[k] = dequantize_real(q as u32, 0.0, H3_BOUNDARY_MAX, 16, false);
            }
        }
        objects.push(o);
        if r.overrun() { bail!("stream overrun inside object slot {slot}"); }
    }
    let mut type_start = [-1i16; H3_OBJECT_TYPES];
    for t in type_start.iter_mut() { *t = r.bits(9) as i16 - 1; }
    let mut quotas = Vec::with_capacity(num_quotas as usize);
    for _ in 0..num_quotas {
        quotas.push(H3Quota { definition: r.u32(), minimum: r.u8(), maximum: r.u8(), placed_on_map: r.u8(), maximum_allowed: r.u8(), price: r.f32() });
    }
    let mcc_map_id = (version >= H3_VARIANT_VERSION_MCC).then(|| { let mut g = [0u8; 16]; for b in g.iter_mut() { *b = r.u8(); } g });
    if r.overrun() { bail!("stream overrun: {} bits read of {}", r.pos, body.len() * 8); }
    Ok(H3Variant { header, chunk_major: H3_MVAR_CHUNK_VERSION, chunk_minor: 0, version, checksum, num_scenario_objects, num_variant_objects,
        num_quotas, map_id, built_in, bounds, game_engine_subtype, maximum_budget, spent_budget, objects, type_start, quotas, mcc_map_id, end_bit: r.pos })
}

// ---------------------------------------------------------------------------------------------
// bitstream encode (sub_1800B8E9C, content header sub_18010BCE0's writer twin)
// ---------------------------------------------------------------------------------------------

struct BitWriter {
    buf: Vec<u8>,
    pos: usize,
}

impl BitWriter {
    fn new() -> Self { Self { buf: Vec::new(), pos: 0 } }
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
    fn u64(&mut self, v: u64) { self.bits(64, v); }
    fn f32(&mut self, v: f32) { self.u32(v.to_bits()); }
    /// `write_string`: bytes up to and including the terminator, never more than `max`.
    fn cstr(&mut self, s: &str, max: usize) {
        let b = s.as_bytes();
        let n = b.len().min(max);
        for &c in &b[..n] { self.u8(c); }
        if n < max { self.u8(0); }
    }
    /// `write_string_wchar`: UTF-16 units up to and including the terminator, never more than `max`.
    fn wstr(&mut self, s: &str, max: usize) {
        let u: Vec<u16> = s.encode_utf16().collect();
        let n = u.len().min(max);
        for &c in &u[..n] { self.u16(c); }
        if n < max { self.u16(0); }
    }
}

/// Editable header strings for a save (None = keep the file's).
#[derive(Clone, Debug, Default)]
pub struct H3HeaderEdits {
    pub name: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
}

/// Encode a variant's content header + body from `objects` (the FULL slot list: every present
/// slot in slot order; a slot index past `number_of_variant_objects` grows the count, capped at
/// 640). The header, budgets, bounds, type starts and quotas are the variant's own. Returns the
/// chunk body (bit-padded to a byte, as the engine's stream is).
pub fn encode_h3_payload(v: &H3Variant, objects: &[H3VariantObject], edits: Option<&H3HeaderEdits>) -> Result<Vec<u8>> {
    let h = &v.header;
    let mut w = BitWriter::new();
    w.u64(h.unique_id);
    w.wstr(edits.and_then(|e| e.name.as_deref()).unwrap_or(&h.name), 16);
    w.cstr(edits.and_then(|e| e.description.as_deref()).unwrap_or(&h.description), 128);
    w.cstr(edits.and_then(|e| e.author.as_deref()).unwrap_or(&h.author), 16);
    w.bits(5, (h.file_type + 1) as u64 & 0x1F);
    w.flag(h.author_is_online);
    w.u64(h.author_xuid);
    w.u64(h.size_bytes);
    w.u64(h.date);
    w.u32(h.length_seconds);
    w.u32(h.campaign_id as u32);
    w.u32(h.map_id);
    w.bits(4, h.game_engine_type as u64 & 0xF);
    w.bits(3, (h.campaign_difficulty + 1) as u64 & 7);
    w.u16(h.hopper_id);
    w.u64(h.game_id);
    // body
    let mut slots: Vec<&H3VariantObject> = objects.iter().collect();
    slots.sort_by_key(|o| o.slot);
    for pair in slots.windows(2) { if pair[0].slot == pair[1].slot { bail!("two objects in slot {}", pair[0].slot); } }
    let nvar = slots.last().map_or(0, |o| o.slot + 1).max(v.num_variant_objects as usize);
    if nvar > H3_SLOTS { bail!("{nvar} object slots (the map variant holds {H3_SLOTS})"); }
    w.u8(v.version as u8);
    w.u32(v.checksum);
    w.bits(10, v.num_scenario_objects as u64);
    w.bits(10, nvar as u64);
    w.bits(9, v.quotas.len() as u64);
    w.u32(v.map_id);
    w.flag(v.built_in);
    for b in v.bounds { w.f32(b); }
    w.bits(4, v.game_engine_subtype as u64 & 0xF);
    w.f32(v.maximum_budget);
    w.f32(v.spent_budget);
    let mut next = slots.iter().peekable();
    for slot in 0..nvar {
        let o = match next.peek() { Some(o) if o.slot == slot => next.next().unwrap(), _ => { w.flag(false); continue } };
        w.flag(true);
        w.u16(o.flags);
        w.u32(o.quota as u32);
        match o.parent { Some(p) => { w.flag(true); w.u64(p); } None => w.flag(false) }
        w.flag(o.placed);
        if !o.placed { continue; }
        for k in 0..3 { w.bits(H3_POSITION_BITS, o.pos_q[k] as u64); }
        w.flag(o.up_is_global);
        if !o.up_is_global { w.bits(19, o.up_q as u64); }
        w.u8(o.angle_q);
        w.u8(o.object_type);
        w.u8(o.mp_flags);
        w.u16(o.game_engine_flags);
        w.u8(o.shared_storage);
        w.u8(o.spawn_time);
        w.u8(o.team);
        w.u8(o.boundary_shape);
        let n = match o.boundary_shape { 1 => 1, 2 => 3, 3 => 4, _ => 0 };
        for k in 0..n { w.u16(o.boundary_q[k]); }
    }
    for t in v.type_start { w.bits(9, (t as i32 + 1) as u64 & 0x1FF); }
    for q in &v.quotas {
        w.u32(q.definition);
        w.u8(q.minimum);
        w.u8(q.maximum);
        w.u8(q.placed_on_map);
        w.u8(q.maximum_allowed);
        w.f32(q.price);
    }
    if v.version >= H3_VARIANT_VERSION_MCC {
        for b in v.mcc_map_id.unwrap_or([0; 16]) { w.u8(b); }
    }
    Ok(w.buf)
}

/// Rebuild a whole `.mvar` from a source file's bytes: the `mvar` chunk carries the new stream,
/// `_eof` is re-stamped with its own byte offset (the "file size" the engine checks), every
/// other chunk (`_blf`, `athr`, `_fsm`) is copied as found.
pub fn rebuild_h3_file(src: &[u8], v: &H3Variant, objects: &[H3VariantObject], edits: Option<&H3HeaderEdits>) -> Result<Vec<u8>> {
    let body = encode_h3_payload(v, objects, edits)?;
    let mut out = Vec::with_capacity(src.len());
    let mut pos = 0usize;
    let mut seen_mvar = false;
    while pos + 12 <= src.len() {
        let magic = &src[pos..pos + 4];
        let size = u32::from_be_bytes([src[pos + 4], src[pos + 5], src[pos + 6], src[pos + 7]]) as usize;
        if size < 12 || pos + size > src.len() { break; }
        if magic == b"mvar" {
            out.extend_from_slice(magic);
            out.extend_from_slice(&((body.len() + 12) as u32).to_be_bytes());
            out.extend_from_slice(&src[pos + 8..pos + 12]);
            out.extend_from_slice(&body);
            seen_mvar = true;
        } else if magic == b"_eof" && size >= 16 {
            let at = out.len() as u32;
            out.extend_from_slice(&src[pos..pos + 12]);
            out.extend_from_slice(&at.to_be_bytes());
            out.extend_from_slice(&src[pos + 16..pos + size]);
        } else {
            out.extend_from_slice(&src[pos..pos + size]);
        }
        pos += size;
    }
    if !seen_mvar { bail!("no mvar chunk in the source file"); }
    out.extend_from_slice(&src[pos..]);
    Ok(out)
}

/// THE save: write `objects` into a copy of `src` at `dst` (`src` may equal `dst`).
pub fn save_h3_variant(src: &Path, dst: &Path, v: &H3Variant, objects: &[H3VariantObject], edits: Option<&H3HeaderEdits>) -> Result<()> {
    let data = crate::mvar::read_expanded(src).map_err(|e| anyhow!("{}: {e}", src.display()))?;
    let out = rebuild_h3_file(&data, v, objects, edits)?;
    std::fs::write(dst, out).map_err(|e| anyhow!("{}: {e}", dst.display()))
}

/// Round-trip report for `--mvar-roundtrip`: decode, re-encode with no edits, compare with the
/// original bytes. `out` receives the re-encoded file when given. The twin of
/// [`crate::h4::mvar::roundtrip_report`], so one verb can gate every title the same way.
pub fn roundtrip_report(path: &Path, out: Option<&Path>) -> Result<String> {
    let raw = std::fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let data = crate::mvar::expand_blf_cmp(&raw);
    let cmp_note = if data != raw { format!(" [source `_cmp`-compressed, {} B]", raw.len()) } else { String::new() };
    let v = parse_h3_variant(path)?;
    let (body, major, _) = blf_h3_mvar_chunk(&data).ok_or_else(|| anyhow!("{}: no mvar chunk", path.display()))?;
    let re = rebuild_h3_file(&data, &v, &v.objects, None)?;
    if let Some(o) = out { std::fs::write(o, &re).map_err(|e| anyhow!("{}: {e}", o.display()))?; }
    let first_diff = data.iter().zip(re.iter()).position(|(a, b)| a != b).or_else(|| (data.len() != re.len()).then_some(data.len().min(re.len())));
    Ok(match first_diff {
        None => format!("{}: Halo 3 chunk v{major}, {} placed / {} slots, {} body bytes, re-encode BYTE-IDENTICAL ({} bytes){cmp_note}",
            path.display(), v.placed().count(), v.objects.len(), body.len(), data.len()),
        Some(i) => format!("{}: Halo 3, {} slots, re-encode DIFFERS at byte {i} (original {} bytes, re-encoded {})",
            path.display(), v.objects.len(), data.len(), re.len()),
    })
}

// ---------------------------------------------------------------------------------------------
// resolution against the base map
// ---------------------------------------------------------------------------------------------

/// The scnr map-variant (sandbox) palette blocks, in quota-kind order 1..=7 (Halo3MCC scnr:
/// vehicle, weapon, equipment, scenery, teleporter, goal, spawner). Element 0x1C: object tagref
/// @0, display name sid @0x10, maximum allowed i32 @0x14, price per instance f32 @0x18.
pub const SANDBOX_PALETTES: [usize; 7] = [0x1E0, 0x1EC, 0x1F8, 0x204, 0x210, 0x21C, 0x228];
pub const SANDBOX_ELEM: usize = 0x1C;
pub const SANDBOX_OFF_MAX: usize = 0x14;
pub const SANDBOX_OFF_PRICE: usize = 0x18;

/// A quota's `definition` is NOT a tag datum: `kind << 16 | index`, where kind 1..=7 indexes
/// `SANDBOX_PALETTES` (proven on every shipped variant: the quota's price and maximum equal the
/// palette element's) and kind 8+ indexes one of the scenario's OWN object palettes - the map's
/// objects Forge may move but not spawn (price -1 / maximum 255 on every one): 8 = scenery
/// (`respawn_point_invisible`, Sandtrap's sandbag walls), 9 = vehicles (the map turrets /
/// security cameras), 10 = weapons (`forge_ball`), 12 = crates (Foundry's `box_l`.. `wall_xxxl`,
/// the only kind that carries placed objects: 462 across the shipped files); 11 (equipment by the
/// same order) never occurs. Pinned by the entry names + placed counts on every map (the
/// `palette_blocks_vs_quota_kinds` probe).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuotaSource {
    /// A sandbox palette block (0-based block index into `SANDBOX_PALETTES`).
    Sandbox(usize),
    /// A scnr object palette (objects.rs `PLACEMENT_TABLES` class).
    Scenario([u8; 4]),
}

impl H3Quota {
    pub fn kind(&self) -> u32 { self.definition >> 16 }
    pub fn index(&self) -> usize { (self.definition & 0xFFFF) as usize }
    pub fn source(&self) -> Option<QuotaSource> {
        match self.kind() {
            1..=7 => Some(QuotaSource::Sandbox(self.kind() as usize - 1)),
            8 => Some(QuotaSource::Scenario(*b"scen")),
            9 => Some(QuotaSource::Scenario(*b"vehi")),
            10 => Some(QuotaSource::Scenario(*b"weap")),
            11 => Some(QuotaSource::Scenario(*b"eqip")),
            12 => Some(QuotaSource::Scenario(*b"bloc")),
            _ => None,
        }
    }
}

/// Tag classes a quota's object definition may carry (every scnr palette class + the ones with a
/// model of their own).
pub const OBJECT_CLASSES: [&[u8; 4]; 13] = [b"scen", b"bloc", b"vehi", b"weap", b"eqip", b"mach", b"ctrl", b"bipd", b"efsc", b"gint", b"crea", b"proj", b"ssce"];

/// The scnr main struct offset of the base map (None without a scenario).
fn scnr_meta(c: &H3Cache) -> Option<usize> { c.tag_meta(*c.find_tags(b"scnr").first()?) }

/// The object definition tag of a quota in the base map (see `QuotaSource`), plus the palette
/// element's (maximum allowed, price) for a sandbox entry.
pub fn quota_entry(c: &H3Cache, q: &H3Quota) -> Option<(usize, Option<(i32, f32)>)> {
    let sm = scnr_meta(c)?;
    let d = c.data();
    match q.source()? {
        QuotaSource::Sandbox(b) => {
            let (n, o) = c.block(sm + SANDBOX_PALETTES[b])?;
            if q.index() >= n { return None; }
            let e = o + q.index() * SANDBOX_ELEM;
            let (_, t) = c.tag_ref(e)?;
            Some((t, Some((d.i32_at(e + SANDBOX_OFF_MAX), d.f32_at(e + SANDBOX_OFF_PRICE)))))
        }
        QuotaSource::Scenario(class) => {
            let t = super::objects::PLACEMENT_TABLES.iter().find(|t| t.class == class)?;
            let (n, o) = c.block(sm + t.palette)?;
            if q.index() >= n { return None; }
            let (_, tag) = c.tag_ref(o + q.index() * super::objects::PALETTE_ELEM)?;
            Some((tag, None))
        }
    }
}

/// The object definition tag of a quota in the base map, when it resolves.
pub fn quota_tag(c: &H3Cache, q: &H3Quota) -> Option<usize> { quota_entry(c, q).map(|(t, _)| t) }

/// The variant's Forge-placed objects resolved to (object definition tag, world matrix, slot),
/// plus the count that resolved to no tag.
pub fn resolve_placed(c: &H3Cache, v: &H3Variant) -> (Vec<(usize, Mat4, usize)>, usize) {
    let mut out = Vec::new();
    let mut unresolved = 0usize;
    for o in v.placed() {
        match v.quota_of(o).and_then(|q| quota_tag(c, q)) {
            Some(t) => out.push((t, o.matrix(), o.slot)),
            None => unresolved += 1,
        }
    }
    (out, unresolved)
}

/// Every Halo 3 `.mvar` under the game's `map_variants` + `hopper_map_variants` folders.
pub fn shipped_variants() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for d in crate::mapcat::h3_variant_dirs() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut files: Vec<std::path::PathBuf> = rd.flatten().map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar"))).collect();
        files.sort();
        out.extend(files);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::cache::{maps_dir, H3Cache};
    use super::*;
    use std::collections::HashMap;

    /// Every shipped Halo 3 variant decodes: usermap content type, a v13 body whose counts sit
    /// inside the DLL's bounds, header and body map ids that agree, the stream ending inside the
    /// chunk's last byte (a wrong field width anywhere desyncs the slots and blows this), every
    /// placed object inside the world bounds with unit / perpendicular axes and a quota index
    /// inside the quota table.
    #[test]
    fn shipped_h3_variants_decode() {
        let files = shipped_variants();
        if files.is_empty() { eprintln!("skip: no Halo 3 variants installed"); return; }
        let mut per_map: HashMap<u32, usize> = HashMap::new();
        let (mut placed_total, mut scenario_total) = (0usize, 0usize);
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            assert!(is_h3_variant(f), "{name}: not classified as Halo 3");
            let data = crate::mvar::read_expanded(f).unwrap();
            let (body, major, _) = blf_h3_mvar_chunk(&data).unwrap();
            assert_eq!(major, H3_MVAR_CHUNK_VERSION);
            let v = parse_h3_variant(f).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(v.header.file_type, H3_FILE_TYPE_USERMAP, "{name}: file type");
            assert!(v.version == H3_VARIANT_VERSION || v.version == H3_VARIANT_VERSION_MCC, "{name}: body version {}", v.version);
            assert_eq!(v.mcc_map_id.is_some(), v.version >= H3_VARIANT_VERSION_MCC, "{name}: mcc map id");
            assert_eq!(v.header.map_id, v.map_id, "{name}: header vs body map id");
            assert!(!v.header.name.is_empty(), "{name}: empty title");
            assert!(v.num_variant_objects as usize <= H3_SLOTS && v.num_quotas as usize <= H3_QUOTA_SLOTS);
            assert!(v.num_scenario_objects <= v.num_variant_objects, "{name}: {} scenario > {} variant objects", v.num_scenario_objects, v.num_variant_objects);
            assert_eq!(v.quotas.len(), v.num_quotas as usize);
            // the encoder pads the last byte only
            let len_bits = body.len() * 8;
            assert!(v.end_bit <= len_bits && v.end_bit + 8 > len_bits, "{name}: stream ends at bit {} of {len_bits}", v.end_bit);
            for o in &v.objects {
                if o.placed {
                    assert!(o.quota >= 0 && (o.quota as usize) < v.quotas.len(), "{name}: slot {} quota {}", o.slot, o.quota);
                    assert!(v.bounds[0] <= v.bounds[1] && v.bounds[2] <= v.bounds[3] && v.bounds[4] <= v.bounds[5], "{name}: placed objects over the empty box");
                    for k in 0..3 { assert!(o.pos[k] >= v.bounds[2 * k] - 1e-3 && o.pos[k] <= v.bounds[2 * k + 1] + 1e-3, "{name}: slot {} pos {:?} outside {:?}", o.slot, o.pos, v.bounds); }
                    let (f, u) = (Vec3::from(o.fwd), Vec3::from(o.up));
                    assert!((f.length() - 1.0).abs() < 1e-3 && (u.length() - 1.0).abs() < 1e-3, "{name}: slot {} axes {f:?} {u:?}", o.slot);
                    assert!(f.dot(u).abs() < 2e-3, "{name}: slot {} fwd.up = {}", o.slot, f.dot(u));
                    placed_total += 1;
                } else {
                    scenario_total += 1;
                }
            }
            *per_map.entry(v.map_id).or_default() += 1;
        }
        eprintln!("{} Halo 3 variants: {placed_total} placed objects, {scenario_total} scenario-owned slots, per map {per_map:?}", files.len());
        assert!(placed_total > 0);
    }

    /// Every variant's map id names an installed Halo 3 map (mapcat reads the id from the
    /// `.mapinfo`); every sandbox quota (kind 1..=7) lands on a palette element whose price and
    /// maximum-allowed equal the quota's own - engine-written numbers checked against the cache,
    /// independent of the bit layout - and every quota of every kind resolves to an object tag,
    /// so every placed object has a model to draw.
    #[test]
    fn shipped_h3_variant_quotas_resolve() {
        let files = shipped_variants();
        if files.is_empty() { eprintln!("skip: no Halo 3 variants installed"); return; }
        let caches: Vec<crate::mapcat::MapCandidate> = crate::mapcat::enumerate().into_iter().filter(|c| c.game == crate::mapcat::Game::Halo3).collect();
        if caches.is_empty() { eprintln!("skip: no Halo 3 maps installed"); return; }
        let mut opened: HashMap<u32, H3Cache> = HashMap::new();
        let (mut quotas, mut sandbox, mut objects, mut unresolved_objects, mut max_edited, mut price_edited) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut kinds: HashMap<u32, usize> = HashMap::new();
        let mut classes: HashMap<String, usize> = HashMap::new();
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let v = parse_h3_variant(f).unwrap();
            let Some(cand) = caches.iter().find(|c| c.map_id == Some(v.map_id)) else { panic!("{name}: map id {} is not an installed Halo 3 map", v.map_id) };
            if !opened.contains_key(&v.map_id) { opened.insert(v.map_id, H3Cache::open(&cand.path).unwrap()); }
            let c = &opened[&v.map_id];
            for (qi, q) in v.quotas.iter().enumerate() {
                let (t, pal) = quota_entry(c, q).unwrap_or_else(|| panic!("{name}: quota {qi} {:#x} (kind {} index {}) does not resolve", q.definition, q.kind(), q.index()));
                let class = c.tag_class(t).unwrap_or(*b"????");
                assert!(OBJECT_CLASSES.iter().any(|k| **k == class), "{name}: quota {qi} -> '{}' class {:?}", c.tag_name(t), String::from_utf8_lossy(&class));
                if let Some((max, price)) = pal {
                    // price / maximum are the palette element's on the 360-era files; a few
                    // tool-edited MLG variants change them (h3_hardcoreFoundry_amplified: warthog
                    // maximum 4 -> 8, warehouse_window price 0 -> 3), so these are counted and
                    // bounded rather than asserted one by one
                    if price != q.price { price_edited += 1; }
                    if max > 0 && max as u8 != q.maximum_allowed { max_edited += 1; }
                    sandbox += 1;
                } else {
                    assert!(q.price < 0.0 && q.maximum_allowed == 255, "{name}: scenario quota {qi} price {} max {}", q.price, q.maximum_allowed);
                }
                *kinds.entry(q.kind()).or_default() += 1;
                *classes.entry(String::from_utf8_lossy(&class).into_owned()).or_default() += 1;
                quotas += 1;
            }
            let (placed, un) = resolve_placed(c, &v);
            objects += placed.len();
            unresolved_objects += un;
        }
        eprintln!("{} variants: {quotas} quotas ({sandbox} sandbox: {price_edited} with an edited price, {max_edited} an edited maximum) by kind {kinds:?}, by class {classes:?}; {objects} placed objects resolved, {unresolved_objects} unresolved", files.len());
        assert!(price_edited * 100 < sandbox, "{price_edited} of {sandbox} sandbox quota prices differ from the palette");
        assert_eq!(unresolved_objects, 0);
    }

    /// THE gate: every shipped Halo 3 variant decodes and re-encodes BYTE-IDENTICAL (chunk
    /// body, `_eof` stamp and the copied chunks) from its decoded records with no edits.
    #[test]
    fn shipped_h3_variants_reencode_byte_identical() {
        let files = shipped_variants();
        if files.is_empty() { eprintln!("skip: no Halo 3 variants installed"); return; }
        for f in &files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            let data = crate::mvar::read_expanded(f).unwrap();
            let v = parse_h3_variant(f).unwrap();
            let (body, _, _) = blf_h3_mvar_chunk(&data).unwrap();
            let enc = encode_h3_payload(&v, &v.objects, None).unwrap();
            assert_eq!(enc.len(), body.len(), "{name}: encoded body length");
            if enc != body {
                let at = enc.iter().zip(body.iter()).position(|(a, b)| a != b).unwrap();
                panic!("{name}: re-encode differs at byte {at} (bit {}): {:02x?} vs {:02x?}", at * 8, &enc[at..(at + 8).min(enc.len())], &body[at..(at + 8).min(body.len())]);
            }
            let re = rebuild_h3_file(&data, &v, &v.objects, None).unwrap();
            assert_eq!(re, data, "{name}: whole-file rebuild differs");
        }
        eprintln!("{} Halo 3 variants re-encode byte-identical", files.len());
    }

    /// An edit round trip through the file: move one placed object and turn it, save to a temp
    /// file, re-parse: the moved object reads back at the quantised pose (position within half
    /// a 16-bit step of the request, axes within the 8-bit angle / 19-bit up resolution), every
    /// other object and every header / quota field is unchanged, and the file re-encodes
    /// byte-identical from its own records again. Also the writer twins: quantise -> dequantise
    /// of every shipped object's floats reproduces its raw quanta EXACTLY (the engine's
    /// rounding, not a nearby one).
    #[test]
    fn edit_round_trip_and_quantisers_reproduce_shipped_quanta() {
        let files = shipped_variants();
        let Some(f) = files.iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { eprintln!("skip"); return };
        let v = parse_h3_variant(f).unwrap();
        // quantiser twins on every shipped object of every file
        let (mut n_pos, mut n_up, mut n_ang, mut n_bnd) = (0usize, 0usize, 0usize, 0usize);
        for g in &files {
            let vv = parse_h3_variant(g).unwrap();
            for o in vv.placed() {
                for k in 0..3 { assert_eq!(quantize_pos(o.pos[k], vv.bounds[2 * k], vv.bounds[2 * k + 1]), o.pos_q[k], "{}: slot {} axis {k}", g.display(), o.slot); n_pos += 1; }
                if !o.up_is_global { assert_eq!(quantize_unit_vector(Vec3::from(o.up)), o.up_q, "{}: slot {} up {:?}", g.display(), o.slot, o.up); n_up += 1; }
                let a = forward_angle(Vec3::from(o.fwd), Vec3::from(o.up));
                let q = quantize_real(a, -std::f32::consts::PI, std::f32::consts::PI, 8, true, false) as u8;
                // an exactly-reversed forward is +-pi by the sign of a zero: quanta 0 and 254
                // decode to the same pose (the reader rotates by (0, -1) for both)
                let half_turn = |q: u8| q == 0 || q == 254;
                assert!(q == o.angle_q || (half_turn(q) && half_turn(o.angle_q)), "{}: slot {} angle {a} -> {q} vs {}", g.display(), o.slot, o.angle_q);
                n_ang += 1;
                for k in 0..4 { if o.boundary_q[k] != 0 { assert_eq!(quantize_real(o.boundary[k], 0.0, H3_BOUNDARY_MAX, 16, false, false), o.boundary_q[k] as u32); n_bnd += 1; } }
            }
        }
        eprintln!("quantiser twins: {n_pos} position axes, {n_up} up vectors, {n_ang} angles, {n_bnd} boundary values reproduced");
        // the edit
        let mut objects = v.objects.clone();
        let target = objects.iter().position(|o| o.placed && o.quota >= 0).unwrap();
        let before = objects[target].clone();
        let want = [before.pos[0] + 1.5, before.pos[1] - 2.25, before.pos[2] + 0.5];
        let up = Vec3::new(0.0, -0.2, 0.98).normalize();
        // the file stores only the angle ABOUT up, so the request is the in-plane forward
        let fwd = { let f = Vec3::new(0.6, 0.8, 0.0); (f - up * f.dot(up)).normalize() };
        objects[target].set_pose(want, fwd, up, v.bounds);
        let dir = std::env::temp_dir().join(format!("hms_h3_roundtrip_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dst = dir.join("moved.mvar");
        save_h3_variant(f, &dst, &v, &objects, None).unwrap();
        let back = parse_h3_variant(&dst).unwrap();
        assert_eq!(back.objects.len(), v.objects.len());
        for (a, b) in back.objects.iter().zip(objects.iter()) { assert_eq!(a, b, "slot {}", b.slot); }
        let m = &back.objects[target];
        for k in 0..3 { assert!((m.pos[k] - want[k]).abs() <= (v.bounds[2 * k + 1] - v.bounds[2 * k]) / 65535.0, "axis {k}: {} vs {}", m.pos[k], want[k]); }
        assert!(Vec3::from(m.up).dot(up) > 0.9999, "up {:?} vs {up:?}", m.up);
        assert!(Vec3::from(m.fwd).dot(fwd.normalize()) > 0.9995, "fwd {:?} vs {fwd:?}", m.fwd);
        assert_ne!(m.pos, before.pos);
        assert_eq!(back.header.name, v.header.name);
        assert_eq!(back.quotas, v.quotas);
        assert_eq!(back.spent_budget, v.spent_budget);
        // the saved file is itself a fixed point of the codec
        let data = std::fs::read(&dst).unwrap();
        assert_eq!(rebuild_h3_file(&data, &back, &back.objects, None).unwrap(), data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The quantisers against the DLL's arithmetic: q = 0 -> min, q = steps -> max, the exact
    /// midpoint on the odd-step paths, and a unit vector that survives the face packing.
    #[test]
    fn quantisers_match_engine() {
        assert_eq!(dequantize_real(0, -1.0, 1.0, 8, true), -1.0);
        assert_eq!(dequantize_real(254, -1.0, 1.0, 8, true), 1.0);
        assert_eq!(dequantize_real(127, -1.0, 1.0, 8, true), 0.0);
        assert_eq!(dequantize_real(65535, 0.0, 60.0, 16, false), 60.0);
        assert!((dequantize_real(32768, -10.0, 10.0, 16, false) - 0.00015).abs() < 1e-3);
        // face 2 (+Z major), components 0 -> (0, 0, 1)
        assert_eq!(dequantize_unit_vector(2 | (127 << 3) | (127 << 11)), Vec3::Z);
        // the perpendicular seed is unit and perpendicular for every face
        for up in [Vec3::Z, Vec3::X, Vec3::Y, -Vec3::Z, Vec3::new(0.3, 0.4, 0.866).normalize()] {
            let s = perpendicular_seed(up);
            assert!((s.length() - 1.0).abs() < 1e-5 && s.dot(up).abs() < 1e-5, "{up:?} -> {s:?}");
            // a full turn comes back to the seed; a quarter turn stays perpendicular to up
            let q = rotate_about(s, up, 1.0, 0.0);
            assert!(q.dot(up).abs() < 1e-5 && q.dot(s).abs() < 1e-5);
            assert!((rotate_about(s, up, 0.0, 1.0) - s).length() < 1e-5);
        }
        // global up + angle 0 -> the engine seed for +Z: (up.z, 0, -up.x) = +X
        let mut b = [0u8; 4];
        b[0] = 0b1000_0000 | (127 >> 1); // flag 1, then angle bits 0111 1111 (127 = 0 rad)
        b[1] = (127 & 1) << 7;
        let (f, u, g, _, a) = read_axes(&mut Bits::new(&b));
        assert!(g && a == 127 && u == Vec3::Z && (f - Vec3::X).length() < 1e-5, "{f:?}");
    }

    /// Probe (print-only): the scnr map-variant palette blocks of guardian / warehouse (count +
    /// first names per block, and any further 0x1C-stride tagref block after the seventh) next to
    /// the quota `definition` kinds of their variants, to pin the kind -> block mapping.
    #[test]
    fn palette_blocks_vs_quota_kinds() {
        for (map, var) in [("guardian.map", "h3_hardcoreGuardian.mvar"), ("warehouse.map", "agla_foundry_012.mvar")] {
            let Some(c) = maps_dir().map(|d| d.join(map)).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { continue };
            let scnr = c.tag_meta(c.find_tags(b"scnr")[0]).unwrap();
            let d = c.data();
            for off in (0x1E0..0x300).step_by(0xC) {
                let Some((n, o)) = c.block(scnr + off) else { continue };
                let names: Vec<String> = (0..n.min(40)).map(|i| c.tag_ref(o + i * 0x1C).map(|(k, t)| format!("{}:{}", String::from_utf8_lossy(&k), c.tag_name(t).rsplit('\\').next().unwrap_or(""))).unwrap_or_else(|| "<null>".into())).collect();
                if names.iter().filter(|s| s.contains(':')).count() * 2 < n { continue; }
                eprintln!("{map} scnr +{off:#x}: {n} entries: {}", names.join(" "));
            }
            let Some(f) = shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy() == var)) else { continue };
            let v = parse_h3_variant(&f).unwrap();
            let mut kinds: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
            for q in &v.quotas { kinds.entry(q.definition >> 16).or_default().push(q.definition & 0xFFFF); }
            eprintln!("{var}: quota kinds {kinds:?}");
            let _ = &c;
        }
        // every scenario-owned quota kind (>= 8) of every variant: the scnr palettes whose
        // count covers the index, with the entry's name, keyed by (map, kind, index)
        let caches: Vec<crate::mapcat::MapCandidate> = crate::mapcat::enumerate().into_iter().filter(|c| c.game == crate::mapcat::Game::Halo3).collect();
        let mut opened: HashMap<u32, H3Cache> = HashMap::new();
        let mut seen = std::collections::BTreeSet::new();
        for f in shipped_variants() {
            let v = parse_h3_variant(&f).unwrap();
            let Some(cand) = caches.iter().find(|c| c.map_id == Some(v.map_id)) else { continue };
            if !opened.contains_key(&v.map_id) { opened.insert(v.map_id, H3Cache::open(&cand.path).unwrap()); }
            let c = &opened[&v.map_id];
            let scnr = c.tag_meta(c.find_tags(b"scnr")[0]).unwrap();
            for q in v.quotas.iter().filter(|q| q.kind() >= 8 && q.kind() != 12) {
                if !seen.insert((v.map_id, q.kind(), q.index())) { continue; }
                let idx = q.index();
                let cands: Vec<String> = super::super::objects::PLACEMENT_TABLES.iter().filter_map(|t| {
                    let (n, o) = c.block(scnr + t.palette)?;
                    if idx >= n { return None; }
                    c.tag_ref(o + idx * super::super::objects::PALETTE_ELEM).map(|(_, tg)| format!("{}[{idx}/{n}]={}", String::from_utf8_lossy(&t.class), c.tag_name(tg).rsplit('\\').next().unwrap_or("")))
                }).collect();
                eprintln!("  map {} kind {} idx {idx} (placed {}): {}", v.map_id, q.kind(), q.placed_on_map, cands.join(" | "));
            }
        }
        {
            let c = &opened[&opened.keys().next().copied().unwrap_or(0)];
            let scnr = c.tag_meta(c.find_tags(b"scnr")[0]).unwrap();
            let v = parse_h3_variant(&shipped_variants()[0]).unwrap();
            // the scenario's own object palettes (objects.rs PLACEMENT_TABLES) as candidates for
            // the kinds past the seven sandbox blocks
            let pal_counts: Vec<String> = super::super::objects::PLACEMENT_TABLES.iter().map(|t| format!("{}@{:#x}:{}", String::from_utf8_lossy(&t.class), t.palette, c.block(scnr + t.palette).map_or(0, |b| b.0))).collect();
            eprintln!("  scenario palettes {}", pal_counts.join(" "));
            for q in v.quotas.iter().filter(|q| q.definition >> 16 >= 8) {
                let (kind, idx) = (q.definition >> 16, (q.definition & 0xFFFF) as usize);
                let cands: Vec<String> = super::super::objects::PLACEMENT_TABLES.iter().filter_map(|t| {
                    let (n, o) = c.block(scnr + t.palette)?;
                    if idx >= n { return None; }
                    c.tag_ref(o + idx * super::super::objects::PALETTE_ELEM).map(|(_, tg)| format!("{}[{idx}]={}", String::from_utf8_lossy(&t.class), c.tag_name(tg).rsplit('\\').next().unwrap_or("")))
                }).collect();
                eprintln!("  kind {kind} idx {idx} (max {} placed {} price {}): {}", q.maximum_allowed, q.placed_on_map, q.price, cands.join(" | "));
            }
        }
    }

    /// Probe (print-only): guardian's shipped `h3_hardcoreGuardian` - the header, budgets, the
    /// per-class object counts and the first placed objects with their resolved tags.
    #[test]
    fn hardcore_guardian_probe() {
        let Some(f) = shipped_variants().into_iter().find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().eq_ignore_ascii_case("h3_hardcoreGuardian.mvar"))) else { return };
        let Some(c) = maps_dir().map(|d| d.join("guardian.map")).filter(|p| p.is_file()).and_then(|p| H3Cache::open(&p).ok()) else { return };
        let v = parse_h3_variant(&f).unwrap();
        eprintln!("'{}' by {} ({}): map {} v{} scenario {} / variant {} objects, {} quotas, budget {} / {}, bounds {:?}, subtype {}",
            v.header.name, v.header.author, v.header.description, v.map_id, v.version, v.num_scenario_objects, v.num_variant_objects, v.num_quotas,
            v.spent_budget, v.maximum_budget, v.bounds, v.game_engine_subtype);
        let mut flags: HashMap<u16, usize> = HashMap::new();
        for o in &v.objects { *flags.entry(o.flags).or_default() += 1; }
        eprintln!("slot flags {flags:?}; type starts {:?}", v.type_start);
        for (i, o) in v.placed().enumerate().take(12) {
            let q = v.quota_of(o).unwrap();
            eprintln!("  slot {} quota {} -> '{}' pos {:?} fwd {:?} up {:?} type {} team {} flags {:#x} mp {:#x} eng {:#x} shape {} {:?}",
                o.slot, o.quota, c.tag_name(q.tag_index()).rsplit('\\').next().unwrap_or(""), o.pos, o.fwd, o.up, o.object_type, o.team, o.flags, o.mp_flags, o.game_engine_flags, o.boundary_shape, &o.boundary[..]);
            let _ = i;
        }
    }

    /// Probe (print-only, #[ignore]): which header / body fields vary across the shipped variants.
    /// Decides what a SAVE has to recompute instead of copying from its template.
    #[test]
    #[ignore]
    fn writer_field_survey() {
        use std::collections::{BTreeMap, BTreeSet};
        let files = shipped_variants();
        eprintln!("SURVEY {} variants", files.len());
        let mut cks: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let mut tstart: BTreeMap<u32, BTreeSet<Vec<i16>>> = BTreeMap::new();
        let mut uid: HashMap<u64, usize> = HashMap::new();
        let (mut size_eq, mut size_zero, mut size_other) = (0, 0, 0);
        let (mut placed_ok, mut placed_bad, mut spent_ok, mut spent_bad) = (0, 0, 0, 0);
        let mut ex_bad: Vec<String> = Vec::new();
        let mut fl: BTreeMap<(u8, u8, u16, u8), usize> = BTreeMap::new();
        let mut ss: BTreeMap<(u8, u8), BTreeMap<u8, usize>> = BTreeMap::new();
        let mut sorted_by_type = (0, 0);
        for f in &files {
            let Ok(v) = parse_h3_variant(f) else { continue };
            let raw = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            cks.entry(v.map_id).or_default().insert(v.checksum);
            tstart.entry(v.map_id).or_default().insert(v.type_start.to_vec());
            *uid.entry(v.header.unique_id).or_default() += 1;
            if v.header.size_bytes == raw { size_eq += 1 } else if v.header.size_bytes == 0 { size_zero += 1 } else { size_other += 1 }
            // quota placed_on_map vs the live count of placed objects per quota
            let mut cnt: HashMap<i32, u32> = HashMap::new();
            let mut spent = 0.0f32;
            for o in v.placed() {
                *cnt.entry(o.quota).or_default() += 1;
                if let Some(q) = v.quota_of(o) { spent += q.price; }
                *fl.entry((o.object_type, o.mp_flags, o.game_engine_flags, o.boundary_shape)).or_default() += 1;
                *ss.entry((o.object_type, o.mp_flags)).or_default().entry(o.shared_storage).or_default() += 1;
            }
            for (i, q) in v.quotas.iter().enumerate() {
                if q.placed_on_map as u32 == *cnt.get(&(i as i32)).unwrap_or(&0) { placed_ok += 1 } else {
                    placed_bad += 1;
                    if ex_bad.len() < 6 { ex_bad.push(format!("{} quota {i}: file {} actual {}", f.file_name().unwrap().to_string_lossy(), q.placed_on_map, cnt.get(&(i as i32)).unwrap_or(&0))); }
                }
            }
            if (v.spent_budget - spent).abs() < 0.01 { spent_ok += 1 } else { spent_bad += 1; if ex_bad.len() < 10 { ex_bad.push(format!("{} spent file {} actual {spent}", f.file_name().unwrap().to_string_lossy(), v.spent_budget)); } }
            // are the Forge-placed slots (past the scenario block) grouped by object type?
            let types: Vec<u8> = v.objects.iter().filter(|o| o.placed && o.slot >= v.num_scenario_objects as usize).map(|o| o.object_type).collect();
            if types.windows(2).all(|w| w[0] <= w[1]) { sorted_by_type.0 += 1 } else { sorted_by_type.1 += 1 }
        }
        for (m, s) in &cks { eprintln!("map {m}: {} distinct checksum(s) {:?}", s.len(), s.iter().take(3).map(|c| format!("{c:#x}")).collect::<Vec<_>>()); }
        let all_ck: BTreeSet<u32> = cks.values().flatten().copied().collect();
        eprintln!("checksum values over everything: {}", all_ck.len());
        for (m, s) in &tstart { eprintln!("map {m}: {} distinct type_start table(s)", s.len()); }
        eprintln!("unique_id: {} distinct over {} files", uid.len(), files.len());
        eprintln!("header size_bytes == file len {size_eq}, == 0 {size_zero}, other {size_other}");
        eprintln!("quota placed_on_map: {placed_ok} agree / {placed_bad} disagree; spent_budget: {spent_ok} agree / {spent_bad} disagree");
        for e in &ex_bad { eprintln!("  {e}"); }
        eprintln!("forge slots sorted by object type: {} files yes / {} no", sorted_by_type.0, sorted_by_type.1);
        eprintln!("(type, mp_flags, engine_flags, shape) -> count, top 25:");
        let mut v: Vec<_> = fl.into_iter().collect(); v.sort_by(|a, b| b.1.cmp(&a.1));
        for (k, n) in v.iter().take(25) { eprintln!("  type {:>2} mp {:#04x} eng {:#06x} shape {} : {n}", k.0, k.1, k.2, k.3); }
        eprintln!("shared_storage by (type, mp_flags):");
        for (k, m) in &ss { eprintln!("  type {:>2} mp {:#04x} : {:?}", k.0, k.1, m.iter().take(12).collect::<Vec<_>>()); }
    }

    /// Probe (print-only, #[ignore]): can quota `placed_on_map` / `spent_budget` be reproduced
    /// exactly from the slots? Counts EVERY present slot with a quota (scenario-owned ones too).
    #[test]
    #[ignore]
    fn budget_rule_survey() {
        use std::collections::BTreeMap;
        let (mut files, mut cnt_all_ok, mut cnt_placed_ok, mut spent_all_ok, mut spent_placed_ok) = (0, 0, 0, 0, 0);
        let (mut rows, mut rows_all_ok, mut rows_placed_ok) = (0usize, 0usize, 0usize);
        let (mut healthy, mut healthy_cnt_all, mut healthy_cnt_placed) = (0usize, 0usize, 0usize);
        let mut ex: Vec<String> = Vec::new();
        let mut max_budget: BTreeMap<u32, usize> = BTreeMap::new();
        for f in shipped_variants() {
            let Ok(v) = parse_h3_variant(&f) else { continue };
            files += 1;
            let mut all: BTreeMap<i32, u32> = BTreeMap::new();
            let mut pl: BTreeMap<i32, u32> = BTreeMap::new();
            let (mut sp_all, mut sp_pl) = (0.0f32, 0.0f32);
            for o in &v.objects {
                if o.quota < 0 { continue; }
                let Some(q) = v.quota_of(o) else { continue };
                *all.entry(o.quota).or_default() += 1;
                sp_all += q.price;
                if o.placed { *pl.entry(o.quota).or_default() += 1; sp_pl += q.price; }
            }
            let (mut a_ok, mut p_ok) = (true, true);
            for (i, q) in v.quotas.iter().enumerate() {
                rows += 1;
                let ia = *all.get(&(i as i32)).unwrap_or(&0) == q.placed_on_map as u32;
                let ip = *pl.get(&(i as i32)).unwrap_or(&0) == q.placed_on_map as u32;
                rows_all_ok += ia as usize; rows_placed_ok += ip as usize;
                a_ok &= ia; p_ok &= ip;
            }
            cnt_all_ok += a_ok as usize; cnt_placed_ok += p_ok as usize;
            let sa = (v.spent_budget - sp_all).abs() < 0.01; let spp = (v.spent_budget - sp_pl).abs() < 0.01;
            if spp { healthy += 1; healthy_cnt_all += a_ok as usize; healthy_cnt_placed += p_ok as usize; }
            spent_all_ok += sa as usize; spent_placed_ok += spp as usize;
            if !sa && ex.len() < 8 { ex.push(format!("{}: file {} all {sp_all} placed {sp_pl} (hopper={})", f.file_name().unwrap().to_string_lossy(), v.spent_budget, f.to_string_lossy().contains("hopper"))); }
            *max_budget.entry(v.maximum_budget as u32).or_default() += 1;
        }
        eprintln!("BUDGET {files} files | every quota row's placed_on_map matches: all-slots {cnt_all_ok}, placed-only {cnt_placed_ok} | rows {rows}: all {rows_all_ok}, placed {rows_placed_ok}");
        eprintln!("BUDGET spent_budget matches: all-slots {spent_all_ok}, placed-only {spent_placed_ok}");
        eprintln!("BUDGET among the {healthy} files whose spent_budget == sum(price of placed): placed_on_map exact for all-slots {healthy_cnt_all}, placed-only {healthy_cnt_placed}");
        for e in &ex { eprintln!("  {e}"); }
        eprintln!("BUDGET maximum_budget values {max_budget:?}");
    }

    /// Probe (print-only, #[ignore]): the quota rows of the first budget-consistent file whose
    /// placed_on_map disagrees with the slot counts.
    #[test]
    #[ignore]
    fn quota_rows_probe() {
        use std::collections::BTreeMap;
        let mut shown = 0;
        for f in shipped_variants() {
            let Ok(v) = parse_h3_variant(&f) else { continue };
            let sp_pl: f32 = v.placed().filter_map(|o| v.quota_of(o)).map(|q| q.price).sum();
            if (v.spent_budget - sp_pl).abs() > 0.01 { continue; }
            let mut all: BTreeMap<i32, u32> = BTreeMap::new();
            let mut pl: BTreeMap<i32, u32> = BTreeMap::new();
            let mut par: BTreeMap<i32, u32> = BTreeMap::new();
            for o in &v.objects {
                if o.quota < 0 { continue; }
                *all.entry(o.quota).or_default() += 1;
                if o.placed { *pl.entry(o.quota).or_default() += 1; }
                if o.parent.is_some() { *par.entry(o.quota).or_default() += 1; }
            }
            let bad: Vec<usize> = (0..v.quotas.len()).filter(|&i| *all.get(&(i as i32)).unwrap_or(&0) != v.quotas[i].placed_on_map as u32).collect();
            if bad.is_empty() { continue; }
            eprintln!("QROWS {} budget {}/{} scen {} var {}", f.file_name().unwrap().to_string_lossy(), v.spent_budget, v.maximum_budget, v.num_scenario_objects, v.num_variant_objects);
            for i in bad.iter().take(14) {
                let q = &v.quotas[*i];
                eprintln!("  quota {i} def {:#x} kind {} min {} max {} allowed {} price {}: file {} | all {} placed {} parented {}", q.definition, q.definition >> 16, q.minimum, q.maximum, q.maximum_allowed, q.price,
                    q.placed_on_map, all.get(&(*i as i32)).unwrap_or(&0), pl.get(&(*i as i32)).unwrap_or(&0), par.get(&(*i as i32)).unwrap_or(&0));
            }
            shown += 1; if shown >= 3 { break; }
        }
    }

    /// Probe (print-only, #[ignore]): does `placed_on_map` count only the objects whose
    /// `game_engine_flags` include the variant's engine? Row agreement per candidate rule.
    #[test]
    #[ignore]
    fn quota_engine_rule_probe() {
        use std::collections::BTreeMap;
        let mut agree: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut subtypes: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for f in shipped_variants() {
            let Ok(v) = parse_h3_variant(&f) else { continue };
            *subtypes.entry((v.game_engine_subtype, v.header.game_engine_type)).or_default() += 1;
            let rules: Vec<(String, Box<dyn Fn(&H3VariantObject) -> bool>)> = vec![
                ("all".into(), Box::new(|_| true)),
                ("placed".into(), Box::new(|o| o.placed)),
                (format!("eng>>subtype"), { let s = v.game_engine_subtype; Box::new(move |o| (o.game_engine_flags >> s) & 1 == 1) }),
                (format!("eng>>hdrtype"), { let s = v.header.game_engine_type; Box::new(move |o| (o.game_engine_flags >> s) & 1 == 1) }),
                ("sym|asym".into(), Box::new(|o| o.mp_flags & 0x0C != 0)),
                ("bit1".into(), Box::new(|o| o.mp_flags & 0x01 != 0)),
                ("!bit7".into(), Box::new(|o| o.mp_flags & 0x80 == 0)),
                ("!bit6".into(), Box::new(|o| o.mp_flags & 0x40 == 0)),
                ("!bit1".into(), Box::new(|o| o.mp_flags & 0x02 == 0)),
                ("placed&!bit7".into(), Box::new(|o| o.placed && o.mp_flags & 0x80 == 0)),
                ("kind>=6 only if eng".into(), { let s = v.game_engine_subtype; Box::new(move |o| (o.game_engine_flags >> s) & 1 == 1 || o.game_engine_flags == 0x3FF) }),
            ];
            for (name, pred) in &rules {
                let mut cnt: BTreeMap<i32, u32> = BTreeMap::new();
                for o in v.objects.iter().filter(|o| o.quota >= 0 && pred(o)) { *cnt.entry(o.quota).or_default() += 1; }
                let e = agree.entry(name.clone()).or_default();
                for (i, q) in v.quotas.iter().enumerate() {
                    e.1 += 1;
                    if *cnt.get(&(i as i32)).unwrap_or(&0) == q.placed_on_map as u32 { e.0 += 1; }
                }
            }
        }
        for (n, (a, t)) in &agree { eprintln!("RULE {n:<22} {a}/{t} rows = {:.2}%", 100.0 * *a as f64 / *t as f64); }
        eprintln!("RULE (subtype, header engine type) -> files {subtypes:?}");
    }

    /// Probe (print-only, #[ignore]): which tag names carry mp_flags 0x04-only / 0x08-only.
    #[test]
    #[ignore]
    fn symmetry_bit_probe() {
        use std::collections::BTreeMap;
        let caches: Vec<crate::mapcat::MapCandidate> = crate::mapcat::enumerate().into_iter().filter(|c| c.game == crate::mapcat::Game::Halo3).collect();
        let mut opened: HashMap<u32, H3Cache> = HashMap::new();
        let mut tally: BTreeMap<(String, u8), usize> = BTreeMap::new();
        for f in shipped_variants() {
            let Ok(v) = parse_h3_variant(&f) else { continue };
            let Some(cand) = caches.iter().find(|c| c.map_id == Some(v.map_id)) else { continue };
            if !opened.contains_key(&v.map_id) { opened.insert(v.map_id, H3Cache::open(&cand.path).unwrap()); }
            let c = &opened[&v.map_id];
            for o in v.placed() {
                if !matches!(o.mp_flags & 0x0C, 0x04 | 0x08) { continue; }
                let Some(q) = v.quota_of(o) else { continue };
                let Some(t) = quota_tag(c, q) else { continue };
                let nm = c.tag_name(t).rsplit('\\').next().unwrap_or("").to_string();
                *tally.entry((nm, o.mp_flags & 0x0C)).or_default() += 1;
            }
        }
        let mut rows: Vec<_> = tally.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        for ((n, b), c) in rows.iter().take(40) { eprintln!("SYM {n:<40} bit {:#04x}: {c}", b); }
    }

    /// Probe (print-only, #[ignore]): how often do shipped variants exceed a quota's limits?
    #[test]
    #[ignore]
    fn quota_limit_probe() {
        use std::collections::BTreeMap;
        let (mut files_over_allowed, mut rows_over_allowed, mut rows_over_max, mut files_over_budget, mut files_over_640) = (0, 0, 0, 0, 0);
        let mut by_kind: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
        let mut n = 0;
        for f in shipped_variants() {
            let Ok(v) = parse_h3_variant(&f) else { continue };
            n += 1;
            let mut cnt: BTreeMap<i32, u32> = BTreeMap::new();
            for o in v.objects.iter().filter(|o| o.quota >= 0) { *cnt.entry(o.quota).or_default() += 1; }
            let mut any = false;
            for (i, q) in v.quotas.iter().enumerate() {
                let c = *cnt.get(&(i as i32)).unwrap_or(&0);
                let e = by_kind.entry(q.definition >> 16).or_default();
                e.1 += 1;
                if q.maximum_allowed > 0 && c > q.maximum_allowed as u32 { rows_over_allowed += 1; any = true; e.0 += 1; }
                if q.maximum > 0 && c > q.maximum as u32 { rows_over_max += 1; }
            }
            files_over_allowed += any as usize;
            if v.spent_budget > v.maximum_budget && v.maximum_budget > 0.0 { files_over_budget += 1; }
            if v.objects.len() > 640 { files_over_640 += 1; }
        }
        eprintln!("QLIM {n} files: {files_over_allowed} exceed some maximum_allowed ({rows_over_allowed} rows), {rows_over_max} rows exceed `maximum`, {files_over_budget} spent>max budget, {files_over_640} >640");
        eprintln!("QLIM by kind (rows over allowed, rows): {by_kind:?}");
    }

    // ---- #h3-accounting: account_for_save --------------------------------------------------

    fn blank(slot: usize, quota: i32, placed: bool) -> H3VariantObject {
        H3VariantObject { slot, flags: 0x83, quota, parent: None, placed, pos: [0.0; 3], fwd: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0],
            pos_q: [0; 3], up_is_global: true, up_q: 0, angle_q: 0, boundary_q: [0; 4], object_type: 0, mp_flags: 0x0C,
            game_engine_flags: 0x3FF, shared_storage: 0, spawn_time: 0, team: 8, boundary_shape: 0, boundary: [0.0; 4] }
    }

    /// A variant whose quota rows are (stored `placed_on_map`, price) and whose stored budget is `spent`.
    fn variant_of(rows: &[(u8, f32)], spent: f32) -> H3Variant {
        let mut v = H3Variant::default();
        v.quotas = rows.iter().map(|&(p, price)| H3Quota { placed_on_map: p, price, ..Default::default() }).collect();
        v.spent_budget = spent;
        v
    }

    /// Nothing changed: the stored values come back bit-for-bit, STALE ones included (the shipped
    /// files disagree with their own slots, and an untouched save must not "fix" them).
    #[test]
    fn accounting_leaves_an_untouched_save_exactly_as_read() {
        let mut v = variant_of(&[(7, 10.0), (0, 25.5)], 123.25);
        let objs = vec![blank(0, 0, true), blank(1, 0, true)];
        account_for_save(&mut v, &objs, &objs, false);
        assert_eq!(v.quotas[0].placed_on_map, 7, "a stale count survives a save that changed nothing");
        assert_eq!(v.quotas[1].placed_on_map, 0);
        assert_eq!(v.spent_budget.to_bits(), 123.25f32.to_bits());
    }

    /// An added object moves its own quota's count by one and the budget by exactly its price.
    #[test]
    fn accounting_charges_an_add_its_own_price_and_count() {
        let mut v = variant_of(&[(7, 10.0), (0, 25.5)], 123.25);
        let before = vec![blank(0, 0, true), blank(1, 0, true)];
        let mut after = before.clone();
        after.push(blank(2, 1, true));
        account_for_save(&mut v, &before, &after, false);
        assert_eq!(v.quotas[1].placed_on_map, 1);
        assert_eq!(v.quotas[0].placed_on_map, 7, "the other quota is untouched");
        assert_eq!(v.spent_budget, 123.25 + 25.5);
    }

    /// A delete refunds its price and count, and a stale value smaller than the refund stops at
    /// zero instead of wrapping the u8 / going negative.
    #[test]
    fn accounting_refunds_a_delete_and_never_underflows() {
        let mut v = variant_of(&[(0, 10.0)], 5.0);
        let before = vec![blank(0, 0, true), blank(1, 0, true)];
        account_for_save(&mut v, &before, &[], false);
        assert_eq!(v.quotas[0].placed_on_map, 0, "0 stored, 2 removed: stops at 0 (a u8 would wrap to 254)");
        assert_eq!(v.spent_budget, 0.0, "5 stored, 20 refunded: stops at 0");
    }

    /// A scenario-owned slot (no Forge placement) counts toward its quota but is free.
    #[test]
    fn accounting_counts_scenario_slots_but_charges_only_forge_placements() {
        let mut v = variant_of(&[(0, 10.0)], 0.0);
        let mut owned = blank(0, 0, false);
        owned.placed = false;
        account_for_save(&mut v, &[], &[owned], false);
        assert_eq!(v.quotas[0].placed_on_map, 1, "counted");
        assert_eq!(v.spent_budget, 0.0, "but free");
    }

    /// A blank canvas recounts: the template's stored numbers describe objects that were dropped.
    #[test]
    fn accounting_recounts_a_blank_canvas_instead_of_adjusting_it() {
        let mut v = variant_of(&[(99, 10.0), (3, 5.0)], 9999.0);
        let after = vec![blank(0, 1, true)];
        account_for_save(&mut v, &[blank(5, 0, true)], &after, true);
        assert_eq!(v.quotas[0].placed_on_map, 0);
        assert_eq!(v.quotas[1].placed_on_map, 1);
        assert_eq!(v.spent_budget, 5.0);
    }

    /// The stored budget never exceeds the cap (the engine saturates it), however much was placed.
    #[test]
    fn accounting_saturates_the_stored_budget_at_the_cap() {
        let mut v = variant_of(&[(0, 100.0)], 0.0);
        v.maximum_budget = 250.0;
        let after: Vec<_> = (0..5).map(|i| blank(i, 0, true)).collect();
        account_for_save(&mut v, &[], &after, false);
        assert_eq!(v.spent_budget, 250.0, "500 of objects against a cap of 250");
        // a variant with no cap (0) is left unclamped
        let mut u = variant_of(&[(0, 100.0)], 0.0);
        account_for_save(&mut u, &[], &after, false);
        assert_eq!(u.spent_budget, 500.0);
    }

    /// The count field is a byte: 300 objects of one quota saturate at 255.
    #[test]
    fn accounting_saturates_a_count_at_the_field_width() {
        let mut v = variant_of(&[(0, 0.0)], 0.0);
        let after: Vec<_> = (0..300).map(|i| blank(i, 0, true)).collect();
        account_for_save(&mut v, &[], &after, false);
        assert_eq!(v.quotas[0].placed_on_map, 255);
    }

    /// Probe (print-only, #[ignore]): which objects does the engine NOT charge for? Candidate
    /// exclusion rules scored by how many files' stored spent_budget they reproduce exactly.
    #[test]
    #[ignore]
    fn price_rule_probe() {
        let files: Vec<H3Variant> = shipped_variants().iter().filter_map(|f| parse_h3_variant(f).ok()).collect();
        let rules: Vec<(&str, Box<dyn Fn(&H3VariantObject) -> bool>)> = vec![
            ("charge all placed", Box::new(|_| false)),
            ("exclude parented", Box::new(|o| o.parent.is_some())),
            ("exclude slot flag 0x01", Box::new(|o| o.flags & 0x01 != 0)),
            ("exclude slot flag 0x02", Box::new(|o| o.flags & 0x02 != 0)),
            ("exclude slot flag 0x04", Box::new(|o| o.flags & 0x04 != 0)),
            ("exclude slot flag 0x08", Box::new(|o| o.flags & 0x08 != 0)),
            ("exclude slot flag 0x10", Box::new(|o| o.flags & 0x10 != 0)),
            ("exclude slot flag 0x20", Box::new(|o| o.flags & 0x20 != 0)),
            ("exclude slot flag 0x80", Box::new(|o| o.flags & 0x80 != 0)),
            ("exclude slot flag 0x100", Box::new(|o| o.flags & 0x100 != 0)),
            ("exclude mp 0x01", Box::new(|o| o.mp_flags & 0x01 != 0)),
            ("exclude mp 0x02", Box::new(|o| o.mp_flags & 0x02 != 0)),
            ("exclude mp 0x40", Box::new(|o| o.mp_flags & 0x40 != 0)),
            ("exclude mp 0x80", Box::new(|o| o.mp_flags & 0x80 != 0)),
            ("exclude type 12 (spawns)", Box::new(|o| o.object_type == 12)),
            ("exclude type >= 12", Box::new(|o| o.object_type >= 12)),
            ("exclude type 0 (ordinary)", Box::new(|o| o.object_type == 0)),
        ];
        for (name, excl) in &rules {
            let (mut exact, mut within, mut n) = (0, 0, 0);
            for v in &files {
                n += 1;
                let cost: f32 = v.objects.iter().filter(|o| o.placed && !excl(o)).filter_map(|o| v.quota_of(o)).map(|q| q.price).sum();
                if (cost - v.spent_budget).abs() < 0.01 { exact += 1; }
                if v.maximum_budget <= 0.0 || cost <= v.maximum_budget { within += 1; }
            }
            eprintln!("PRICE {name:<28} exact {exact:>3}/{n}  within-max {within:>3}/{n}");
        }
        // the budget-over files, in detail
        for v in files.iter().filter(|v| v.maximum_budget > 0.0 && v.objects.iter().filter(|o| o.placed).filter_map(|o| v.quota_of(o)).map(|q| q.price).sum::<f32>() > v.maximum_budget).take(4) {
            let mut by_flags: std::collections::BTreeMap<(u16, u8), (u32, f32)> = std::collections::BTreeMap::new();
            for o in v.objects.iter().filter(|o| o.placed) {
                let e = by_flags.entry((o.flags, o.mp_flags)).or_default();
                e.0 += 1; e.1 += v.quota_of(o).map_or(0.0, |q| q.price);
            }
            eprintln!("PRICE over: stored {} max {} -> {:?}", v.spent_budget, v.maximum_budget, by_flags);
        }
    }

    /// Probe (print-only, #[ignore]): the scenario-owned slot count per map, and how many of those
    /// slots carry a quota (placed objects the scenario itself spawns) -- this is what a Forge
    /// placement has to share the 640 slots with.
    #[test]
    #[ignore]
    fn scenario_slots_per_map_probe() {
        use std::collections::BTreeMap;
        let mut by_map: BTreeMap<u32, (usize, usize, usize)> = BTreeMap::new();
        for f in shipped_variants() {
            let Ok(v) = parse_h3_variant(&f) else { continue };
            let owned = v.objects.iter().filter(|o| !o.placed).count();
            let e = by_map.entry(v.map_id).or_insert((v.num_scenario_objects as usize, owned, 0));
            e.2 += 1;
        }
        for (m, (scen, owned, n)) in &by_map { eprintln!("SCEN map {m}: num_scenario_objects {scen} (present unplaced slots {owned}) -> forge capacity {} ({n} variants)", H3_SLOTS - scen); }
    }
}
