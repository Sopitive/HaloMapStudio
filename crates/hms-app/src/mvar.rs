//! Halo: Reach `.mvar` map-variant codec. Unwraps the BLF container, bit-decodes the content
//! header, the variant body, the 651 forge-object slots and the quota table, and re-encodes the
//! whole file bit-exactly (object list, header strings, stamps, global fields). Layout evidence:
//! reach_tag_test.exe (`sub_14053A5F0` content header, `sub_140216020` body) and
//! docs/reach_mvar_layout.md.

// ─────────────────────────────────────────────────────────────────────────────────────────────
// Forge object flag semantics, from `reach_tag_test_360.exe` (a symbol-bearing Halo Reach
// tag_test build; the engine's `scenario_map_variant.cpp`): the flag-enum member strings the
// engine embeds near the `.mvar`/`scenario_map_variant` bounds-check asserts
// (`c_variant_placement_flags_enum` / `c_variant_object_placement_flags_enum`):
//
//     hide_unless_required        // bit0
//     is_shortcut                 // bit1
//     asymmetric_placement        // bit2
//     symmetric_placement         // bit3
//     not_initially_placed        // bit4
//     unique_spawn                // bit5
//     _map_variant_hard_attachment   // (head `flags` field — see PlacedObject.flags)
//     _map_variant_occupied_slot     // (head `flags` field)
//
// Cross-validated against the 6 distinct `placement` bytes in retail `forge_halo_hemorrhage.mvar`
// (12, 44, 76, 108, 204, 236 — all decode cleanly under this layout: symmetric+asymmetric both set,
// initially-placed, physics in bits 6-7). Note that "placed at start" is bit4 (inverted), not
// bit1, and "gametype gating" is bit0 rather than bit5; both bits are 0 on every shipped variant.
//
// `PlacedObject.placement` (8-bit placement-flags byte):
pub const PLACE_HIDE_UNLESS_REQUIRED: u8 = 0x01; // bit0 — hide unless a gametype/megalo script requires this type
pub const PLACE_IS_SHORTCUT: u8 = 0x02;          // bit1 — object is a navigation "shortcut"
pub const PLACE_ASYMMETRIC: u8 = 0x04;           // bit2 — appears in ASYMMETRIC gametypes
pub const PLACE_SYMMETRIC: u8 = 0x08;            // bit3 — appears in SYMMETRIC gametypes
pub const PLACE_SYMMETRY_MASK: u8 = 0x0C;        // bits2-3 — 0 never, 1 asymmetric-only, 2 symmetric-only, 3 both ("Disregard" in the Forge menu)
pub const PLACE_NOT_AT_START: u8 = 0x10;         // bit4 — INVERTED: CLEAR = "placed at match start" (spawns)
pub const PLACE_UNIQUE_SPAWN: u8 = 0x20;         // bit5 — unique spawn
pub const PLACE_PHYSICS_MASK: u8 = 0xC0;         // bits6-7 — `forge_object_properties_physics`: 0 normal, 1 fixed, 3 phased (2 unused in Reach)
// The most common retail placement (210/431 Hemorrhage objects): both symmetries set (appears
// everywhere), initially placed, Normal physics = 0b0000_1100 = 12.
pub const PLACEMENT_DEFAULT_SPAWNABLE: u8 = PLACE_SYMMETRY_MASK; // 0x0C

/// Best-effort human name for a `PlacedObject.cached_type` (the 5-bit
/// `s_variant_multiplayer_object_properties` object type).
///
/// The value indexes the retail globals-tag `multiplayer_object_type_list` — a data-driven list,
/// not a fixed enum baked into the exe (the engine warns to run
/// `multiplayer-generate-global-object-type-list` after editing it), so the value→name mapping is
/// retail convention. Engine-confirmed are the indices with hardcoded per-type extra fields in the
/// variant (read/written specially): 1=weapon (spare clips), 12/13/14=teleporter (channel + 5-bit
/// passability), 19=location (location-name index).
///
/// Observed value distribution in retail `forge_halo_hemorrhage.mvar` (431 objs):
///   0:141  1:19  2:8  7:12  8:2  9:2  13:2  14:2  15:221  16:4  25:2  26:11  27:5
///
/// | value | name                        | source                               |
/// |------:|-----------------------------|--------------------------------------|
/// |     0 | none / ordinary             | inferred (community/RVT)             |
/// |     1 | weapon                      | CONFIRMED (engine reads spare_clips) |
/// |     2 | grenade                     | inferred                             |
/// |  7..9 | vehicle-class spawns        | inferred                             |
/// |    12 | teleporter 2-way            | CONFIRMED (engine reads channel/pass)|
/// |    13 | teleporter sender           | CONFIRMED (engine reads channel/pass)|
/// |    14 | teleporter receiver         | CONFIRMED (engine reads channel/pass)|
/// |    15 | ordinary forge object       | inferred (dominant retail value)     |
/// |    19 | named location / area       | CONFIRMED (engine reads location_name)|
/// | 25-27 | objective / gadget spawns   | inferred                             |
/// Indices not listed are unverified; treat every `inferred` name as tentative.
pub fn cached_type_name(t: u8) -> &'static str {
    match t {
        0 => "none",
        1 => "weapon",           // CONFIRMED
        2 => "grenade",          // inferred
        12 => "teleporter 2-way",  // CONFIRMED
        13 => "teleporter sender", // CONFIRMED
        14 => "teleporter receiver", // CONFIRMED
        19 => "location",        // CONFIRMED
        _ => "type (see cached_type table)",
    }
}

/// One placed forge object read from a .mvar.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedObject {
    /// `folder` = variant_quota_index: a FLAT positional index into the ordered list of
    /// distinct sandbox-palette object types (the quota array is 1:1 with that palette).
    /// `item` = variant_index: the variant/permutation within that type. Resolved offline
    /// by SceneController::forge_type_order + resolve_forge_model. (NOT a paletteIndex.)
    pub folder: u16,
    pub item: u8,
    pub pos: [f32; 3],
    pub team: u8,
    pub color: i32, // -1 = none
    /// World orientation, decoded from the variant's up-vector + forward-angle.
    /// Defaults to identity (forward=+X, up=+Z) — overwritten by `decode_orientation`.
    pub fwd: [f32; 3],
    pub up: [f32; 3],
    /// Raw rotation fields, captured so orientation can be (re)decoded without a re-parse.
    /// `up_is_global` → up = global up (0,0,1); otherwise `up_quant` is the 20-bit
    /// octahedral-quantized up vector. `forward_angle_q` is the 14-bit yaw over [-π, π].
    pub up_is_global: bool,
    pub up_quant: u32,
    pub forward_angle_q: u32,
    /// spawnSequence (−100..100). Retail = spawn ordering; the Mjolnir "X330" Forge-editor
    /// mod hijacks it to encode a visual scale multiplier (see `forge_scale`).
    pub spawn_seq: i32,
    /// respawn/spawn-time in seconds (0 = default). blf `spawn_time` (8 bits).
    pub respawn: u8,
    /// s_variant_multiplayer_object_properties cached-object-type (5 bits). Indexes the retail
    /// globals `multiplayer_object_type_list`. Engine-CONFIRMED anchors: 1=weapon, 12/13/14=
    /// teleporter 2-way/sender/receiver, 19=location. See `cached_type_name`.
    pub cached_type: u8,
    /// forge label / location-name index into the variant string table (0xFFFF = none).
    pub label_idx: u16,
    /// placement-flags bitfield (8 bits): spawn-at-start, physics, etc.
    pub placement: u8,
    /// boundary shape: 0=none, 1=sphere, 2=cylinder, 3=box.
    pub boundary_shape: u8,
    /// `s_variant_object` flags (2 bits) at the head of each object record. reach_tag_test
    /// (`scenario_map_variant.cpp`) names two bits here: `_map_variant_occupied_slot` and
    /// `_map_variant_hard_attachment` (object is hard-attached to its `spawn_rel` parent). In every
    /// shipped variant this field is 1 for every real object (bit0 set) — the "occupied/used" state
    /// the engine tests via `flags.test(_map_variant_occupied_slot)`. hard_attachment is 0 in retail
    /// data; the exact bit0/bit1 assignment beyond "bit0 = occupied/used" is inferred. New objects
    /// must set this to 1 or the game reads the slot but never spawns the object.
    pub flags: u8,
    /// whether the quantised position was stored in-bounds. When false the record
    /// carried a bsp-index escape (`bsp_index`). Needed for a faithful re-encode.
    pub in_bounds: bool,
    /// out-of-bounds bsp escape index (−1 = none / in-bounds).
    pub bsp_index: i32,
    /// spawn-relative-to: parent object slot index (−1 = none). Parenting.
    pub spawn_rel: i32,
    /// The .mvar SLOT index (0..650) this object was parsed from — the value `spawn_rel` on ANOTHER
    /// object refers to. 0xFFFF for objects not sourced from a slot (freshly-created adds, whose slot
    /// is only assigned when they fill an empty slot at encode time). Used by the parenting UI to map
    /// a chosen parent object → its slot for `spawn_rel`.
    pub slot: u16,
    /// boundary shape values (quantised 11-bit): sphere=[radius], cylinder=[radius,
    /// top, bottom], box=[width, length, top, bottom]. Only `boundary_shape`-many valid.
    pub boundary: [u16; 4],
    /// cached_type 1 (weapon): spare-clips/rounds (8 bits).
    pub weapon_clips: u8,
    /// cached_type 12/13/14 (teleporter): channel + passability (5 bits each).
    pub tele_channel: u8,
    pub tele_passability: u8,
    /// cached_type 19 (location): location-name index (0xFFFF = none).
    pub location_name: u16,
}

/// MSB-first bit reader over the variant bitstream.
struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize, // bit position
}
impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn bits(&mut self, n: u32) -> u64 {
        let mut r = 0u64;
        for _ in 0..n {
            let byte = self.pos >> 3;
            let off = 7 - (self.pos & 7);
            let bit = if byte < self.buf.len() { (self.buf[byte] >> off) & 1 } else { 0 };
            r = (r << 1) | bit as u64;
            self.pos += 1;
        }
        r
    }
    fn bits_signed(&mut self, n: u32) -> i64 {
        let r = self.bits(n);
        if n > 0 && (r & (1 << (n - 1))) != 0 {
            (r as i128 - (1i128 << n)) as i64
        } else {
            r as i64
        }
    }
    fn flag(&mut self) -> bool {
        self.bits(1) != 0
    }
    fn byte(&mut self) -> u8 {
        self.bits(8) as u8
    }
    fn u16_be(&mut self) -> u16 {
        ((self.bits(8) << 8) | self.bits(8)) as u16
    }
    fn u32_le(&mut self) -> u32 {
        (self.bits(8) | (self.bits(8) << 8) | (self.bits(8) << 16) | (self.bits(8) << 24)) as u32
    }
    fn u64_le(&mut self) -> u64 {
        let mut r = 0u64;
        for i in 0..8 {
            r |= self.bits(8) << (i * 8);
        }
        r
    }
    fn f32_be(&mut self) -> f32 {
        let b = [self.byte(), self.byte(), self.byte(), self.byte()];
        f32::from_be_bytes(b)
    }
    fn skip_string_stop(&mut self, max_n: usize) {
        for _ in 0..max_n {
            if self.byte() == 0 {
                break;
            }
        }
    }
    fn skip_widechar_stop(&mut self, max_wc: usize) {
        for _ in 0..max_wc {
            if self.u16_be() == 0 {
                break;
            }
        }
    }
    /// Read a null-terminated ASCII/UTF-8 string of up to `max_n` bytes (mirrors
    /// `skip_string_stop`: stops at a 0 byte — terminator consumed — or after `max_n` bytes).
    fn read_string_stop(&mut self, max_n: usize) -> String {
        let mut bytes = Vec::new();
        for _ in 0..max_n {
            let b = self.byte();
            if b == 0 {
                break;
            }
            bytes.push(b);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
    /// Read a null-terminated UTF-16BE string of up to `max_wc` wide chars (mirrors
    /// `skip_widechar_stop`: stops at a 0x0000 unit — consumed — or after `max_wc` units).
    fn read_widechar_stop(&mut self, max_wc: usize) -> String {
        let mut units = Vec::new();
        for _ in 0..max_wc {
            let u = self.u16_be();
            if u == 0 {
                break;
            }
            units.push(u);
        }
        String::from_utf16_lossy(&units)
    }
    /// Read a `content_author` block: (display Name, bit pos at the START of the Name field, bit
    /// pos at the END of the Name field i.e. before the IsOnline flag) plus the (Timestamp, Xuid)
    /// pair, both 64-bit MSB-first values (the chdr chunk stores the same numbers little-endian).
    /// The IsOnline bit is consumed but not returned. Capturing the Name span lets the encoder
    /// splice a new Name while copying the fixed surrounding bits verbatim.
    fn read_content_author_stamped(&mut self) -> ((String, usize, usize), (u64, u64)) {
        let ts = self.bits(64); // Timestamp
        let xuid = self.bits(64); // Xuid
        let start = self.pos;
        let name = self.read_string_stop(16);
        let end = self.pos;
        self.flag(); // IsOnline
        ((name, start, end), (ts, xuid))
    }
    fn read_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.byte());
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Rotation decode — port of the Blam map-variant `read_axes` chain (Blam-Network/blf
// @ 3778221: dequantize_unit_vector3d + angle_to_axes_internal + dequantize_real). The
// bitstream stores the up vector as a cube-face projection (NOT true octahedral) plus a
// quantized forward (yaw) angle.
//
// The quantized data was written by the ENGINE, which is Z-up Blam (up=+Z, forward=+X,
// left=+Y) — the same frame object_matrix expects (local +X=fwd, +Z=up), so the decode
// happens DIRECTLY in that frame using the engine's world axes and interprets the dequantized
// components as (x,y,z) with no remap. (blf-ts re-labels these as a Y-up {i,j,k} frame;
// applying that relabel here is only correct for the global-up case and scrambles tilted
// up-vectors — ramps and angled forge pieces.)
// ---------------------------------------------------------------------------

const K_PI: f64 = std::f64::consts::PI;
// Engine (Blam Z-up) world axes — matches object_matrix + native scenario objects.
const GLOBAL_UP: [f64; 3] = [0.0, 0.0, 1.0];
const GLOBAL_FORWARD: [f64; 3] = [1.0, 0.0, 0.0];
const GLOBAL_LEFT: [f64; 3] = [0.0, 1.0, 0.0];

/// (actual_per_axis_max_count, quantized_value_count) for a unit-vector bit
/// count. Alternating-bit mask starting at bit `bit_count-2`, then floor(sqrt)-1.
/// (Load-bearing; do NOT simplify — matches the engine's precomputed table.)
fn unit_vector_encoding_constants(bit_count: u32) -> (i64, i64) {
    let mut mask: i64 = 0;
    let mut i: u32 = 0;
    let mut bit = bit_count as i32 - 2;
    while bit >= 0 {
        if i % 2 == 1 {
            mask |= 1i64 << bit;
        }
        bit -= 1;
        i += 1;
    }
    let qvc = (mask as f64).sqrt().floor() as i64 - 1;
    (mask, qvc)
}

fn dequantize_real_f(
    quantized: i64,
    min_v: f64,
    max_v: f64,
    quantized_value_count: i64,
    exact_midpoints: bool,
    exact_endpoints: bool,
) -> f64 {
    let mut value_count = quantized_value_count;
    if exact_midpoints {
        value_count -= 1;
    }
    if exact_endpoints {
        if quantized == 0 {
            return min_v;
        }
        if quantized == value_count - 1 {
            return max_v;
        }
        let step = (max_v - min_v) / (value_count - 2) as f64;
        return min_v + step * ((quantized - 1) as f64 + 0.5);
    }
    let step = (max_v - min_v) / value_count as f64;
    min_v + step * (quantized as f64 + 0.5)
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize3(v: &mut [f64; 3]) {
    let m = dot3(*v, *v).sqrt();
    if m > 1e-6 {
        v[0] /= m;
        v[1] /= m;
        v[2] /= m;
    }
}

/// Decode a cube-face-projected unit vector (blf frame). Falls back to global up
/// on a degenerate bit count.
pub(crate) fn dequantize_unit_vector3d(value: i64, bit_count: u32) -> [f64; 3] { // shared with h4::mvar
    let (apamc, qvc) = unit_vector_encoding_constants(bit_count);
    if apamc <= 0 || qvc <= 0 {
        return GLOBAL_UP;
    }
    let face = (value / apamc).max(0);
    let rem = ((value % apamc) + apamc) % apamc;
    let qu = rem / qvc;
    let qw = rem % qvc;
    let u = dequantize_real_f(qu, -1.0, 1.0, qvc, true, false);
    let w = dequantize_real_f(qw, -1.0, 1.0, qvc, true, false);
    let mut v = match face {
        0 => [1.0, u, w],
        1 => [u, 1.0, w],
        2 => [u, w, 1.0],
        3 => [-1.0, u, w],
        4 => [u, -1.0, w],
        5 => [u, w, -1.0],
        _ => GLOBAL_UP,
    };
    normalize3(&mut v);
    v
}

/// Reconstruct the forward vector (blf frame) from up + a yaw angle (Rodrigues
/// rotation of a deterministic reference tangent about `up`).
fn angle_to_axes_internal(up: [f64; 3], angle: f64) -> [f64; 3] {
    let v10 = dot3(up, GLOBAL_FORWARD).abs();
    let v9 = dot3(up, GLOBAL_LEFT).abs();
    let mut forward = if v10 >= v9 {
        cross3(GLOBAL_LEFT, up)
    } else {
        cross3(up, GLOBAL_FORWARD)
    };
    normalize3(&mut forward);
    let (s, c) = if angle == K_PI || angle == -K_PI {
        (0.0, -1.0)
    } else {
        (angle.sin(), angle.cos())
    };
    let cr = cross3(up, forward);
    let d = dot3(up, forward);
    let omc = 1.0 - c;
    let mut f = [
        forward[0] * c + cr[0] * s + up[0] * d * omc,
        forward[1] * c + cr[1] * s + up[1] * d * omc,
        forward[2] * c + cr[2] * s + up[2] * d * omc,
    ];
    normalize3(&mut f);
    f
}

/// Decode world (forward, up) in engine/HMS space (Z-up) from the captured raw fields.
/// Dequantized components are already engine (x,y,z); no frame remap.
pub fn decode_orientation(up_is_global: bool, up_quant: u32, forward_angle_q: u32) -> ([f32; 3], [f32; 3]) {
    let up = if up_is_global {
        GLOBAL_UP
    } else {
        dequantize_unit_vector3d(up_quant as i64, 20)
    };
    let angle = dequantize_real_f(forward_angle_q as i64, -K_PI, K_PI, 1i64 << 14, false, false);
    let fwd = angle_to_axes_internal(up, angle);
    let f32v = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    (f32v(fwd), f32v(up))
}

/// Inverse of the forward-angle decode for an UPRIGHT (global-Z up) object — quantise a world
/// forward vector's yaw to the 14-bit `forward_angle_q` field. With up=+Z the decode produces
/// fwd=[cos θ, sin θ, 0], so θ = atan2(fwd.y, fwd.x). Objects placed upright (the common case)
/// round-trip; a tilted `up` is not encoded here (caller uses global up), so tilt is dropped.
pub fn quantize_forward_angle(fwd: [f32; 3]) -> u32 {
    let angle = (fwd[1] as f64).atan2(fwd[0] as f64); // [-π, π]
    let count = (1i64 << 14) as f64;
    let step = (K_PI - -K_PI) / count;
    let q = ((angle - -K_PI) / step - 0.5).round();
    q.clamp(0.0, count - 1.0) as u32
}

/// EXACT inverse of `dequantize_unit_vector3d`: quantise a unit vector to the engine's cube-face
/// scheme for `bit_count` bits. The decoder picks the DOMINANT (largest-|·|) axis as the cube face
/// (sign → which of the 6 faces), then stores the OTHER two components divided by |dominant| as two
/// `[-1,1]` reals `u`,`w` (`exact_midpoints` mapping). We invert each step: choose the same face,
/// form the same ratios, and quantise `u`/`w` with the inverse of `dequantize_real_f(_, true, false)`.
/// The packed value is `face*apamc + qu*qvc + qw` (matching `rem = qu*qvc + qw`, `face = value/apamc`).
fn quantize_unit_vector3d(v: [f64; 3], bit_count: u32) -> i64 {
    let (apamc, qvc) = unit_vector_encoding_constants(bit_count);
    if apamc <= 0 || qvc <= 0 {
        return 0;
    }
    let mut n = v;
    normalize3(&mut n);
    let (ax, ay, az) = (n[0].abs(), n[1].abs(), n[2].abs());
    // Dominant axis + sign → face; the remaining two components / |dominant| → (u, w). This mirrors
    // the decoder's face table exactly: face 0/3=±x → (y,z); 1/4=±y → (x,z); 2/5=±z → (x,y).
    let (face, u, w) = if ax >= ay && ax >= az {
        if n[0] >= 0.0 { (0i64, n[1] / ax, n[2] / ax) } else { (3, n[1] / ax, n[2] / ax) }
    } else if ay >= ax && ay >= az {
        if n[1] >= 0.0 { (1, n[0] / ay, n[2] / ay) } else { (4, n[0] / ay, n[2] / ay) }
    } else if n[2] >= 0.0 {
        (2, n[0] / az, n[1] / az)
    } else {
        (5, n[0] / az, n[1] / az)
    };
    // Inverse of dequantize_real_f(_, -1, 1, qvc, exact_midpoints=true, exact_endpoints=false):
    //   value = -1 + step*(q+0.5), step = 2/(qvc-1)  ⇒  q = round((value+1)/step - 0.5).
    let value_count = qvc - 1;
    let step = 2.0 / value_count as f64;
    let q = |val: f64| -> i64 {
        let qi = ((val - -1.0) / step - 0.5).round() as i64;
        qi.clamp(0, value_count - 1)
    };
    face * apamc + q(u) * qvc + q(w)
}

/// EXACT inverse of `decode_orientation`. When `up` ≈ global +Z → the upright fast path
/// `(true, 0, quantize_forward_angle(fwd))`. Otherwise quantise `up` via
/// `quantize_unit_vector3d`, then recover the yaw the same reference frame `angle_to_axes_internal`
/// builds: with `ref`,`cr` = the deterministic tangent + `cross(up, ref)`, the decoder produces
/// `fwd = ref*cos θ + cr*sin θ`, so `θ = atan2(fwd·cr, fwd·ref)`. We build that frame from the
/// DEQUANTISED up (what the decoder will actually use) so the round-trip is tight, then quantise θ
/// over [-π, π] with 14 bits exactly like `quantize_forward_angle`.
pub fn encode_orientation(fwd: [f32; 3], up: [f32; 3]) -> (bool, u32, u32) {
    let mut upn = [up[0] as f64, up[1] as f64, up[2] as f64];
    normalize3(&mut upn);
    if dot3(upn, GLOBAL_UP) > 1.0 - 1e-6 {
        return (true, 0, quantize_forward_angle(fwd));
    }
    let up_quant = quantize_unit_vector3d(upn, 20) as u32;
    // Rebuild the decoder's reference frame from the DEQUANTISED up (angle_to_axes_internal's exact
    // steps) so the recovered yaw reconstructs fwd through decode_orientation.
    let up_used = dequantize_unit_vector3d(up_quant as i64, 20);
    let v10 = dot3(up_used, GLOBAL_FORWARD).abs();
    let v9 = dot3(up_used, GLOBAL_LEFT).abs();
    let mut refv = if v10 >= v9 { cross3(GLOBAL_LEFT, up_used) } else { cross3(up_used, GLOBAL_FORWARD) };
    normalize3(&mut refv);
    let mut crv = cross3(up_used, refv);
    normalize3(&mut crv);
    let fwdf = [fwd[0] as f64, fwd[1] as f64, fwd[2] as f64];
    let angle = dot3(fwdf, crv).atan2(dot3(fwdf, refv));
    let count = (1i64 << 14) as f64;
    let step = (K_PI - -K_PI) / count;
    let fq = ((angle - -K_PI) / step - 0.5).round().clamp(0.0, count - 1.0) as u32;
    (false, up_quant, fq)
}

/// The decoded team byte of the Forge "Neutral" team (raw 4-bit field 9; the engine's
/// `_multiplayer_team_designator_neutral` = 8, after the 8 coloured designators). Every object a
/// player places in the game's Forge starts on this team, so HMS's new placements do too.
/// `TEAM_NONE` (raw 0) is the distinct "none" value; it is only kept when a variant already
/// carries it.
pub const TEAM_NEUTRAL: u8 = 8;
/// "No team" as decoded from the .mvar (raw 4-bit field 0).
pub const TEAM_NONE: u8 = 0xFF;

/// Build a new PlacedObject to ADD to a variant, from a palette (folder,item), world pos, and full
/// orientation (forward + up). team/color follow the same convention as the reader (team
/// `TEAM_NONE` = none, color -1 = none). Boundary/label/type-extras default to none. The FULL
/// rotation (including any tilt where up ≠ global +Z) is encoded via `encode_orientation`.
pub fn new_placed_object(folder: u16, item: u8, pos: [f32; 3], fwd: [f32; 3], up: [f32; 3], team: u8, color: i32) -> PlacedObject {
    let (up_is_global, up_quant, forward_angle_q) = encode_orientation(fwd, up);
    let (f, u) = decode_orientation(up_is_global, up_quant, forward_angle_q);
    PlacedObject {
        folder, item, pos, team, color, fwd: f, up: u,
        up_is_global, up_quant, forward_angle_q,
        // Required for the object to SPAWN IN GAME (not just be readable): every REAL forge
        // object in a shipped variant (Hemorrhage: all 431) has `flags=1` and `placement` with
        // bits 2,3 set (value 12 = spawn-at-start). With flags=0 / placement=0 MCC reads the
        // slot but never spawns the object.
        spawn_seq: 0, respawn: 0, cached_type: 0, label_idx: 0xFFFF, placement: PLACEMENT_DEFAULT_SPAWNABLE,
        boundary_shape: 0, flags: 1, in_bounds: true, bsp_index: -1, spawn_rel: -1, slot: 0xFFFF,
        boundary: [0; 4], weapon_clips: 0, tele_channel: 0, tele_passability: 0, location_name: 0xFFFF,
    }
}

fn highest_bit_set(v: i32) -> i32 {
    31 - (v as u32).leading_zeros() as i32
}

fn compute_axis_bits(bitcount: i32, ext: [f32; 3]) -> [u32; 3] {
    const MIN_UNIT: f32 = 0.00833333333;
    let mut min_step = if bitcount > 16 {
        MIN_UNIT / (1i32 << (bitcount - 16)) as f32
    } else {
        (1i32 << (16 - bitcount)) as f32 * MIN_UNIT
    };
    if min_step < 0.0001 {
        return [26, 26, 26];
    }
    min_step *= 2.0;
    let mut out = [bitcount as u32; 3];
    for i in 0..3 {
        let edx = (ext[i] / min_step + 0.9999).min(0x800000 as f32) as i32;
        if edx <= 0 {
            out[i] = 0;
        } else {
            let ecx = highest_bit_set(edx);
            let eax = (1 << ecx) - 1;
            let r8 = ecx + if (edx & eax) != 0 { 1 } else { 0 };
            out[i] = r8.min(26) as u32;
        }
    }
    out
}

fn skip_content_author(br: &mut BitReader) {
    br.u64_le(); // Timestamp
    br.u64_le(); // Xuid
    br.skip_string_stop(16); // Name
    br.flag(); // IsOnline
}

// c_single_language_string_table<256,4096,12,13,9>::decode (blf, verified):
//   string_count : 9 bits; if 0 -> done.
//   per string   : `exists` bool + (if exists) offset : 12 bits.
//   buffer_size  : 13 bits.
//   compressed   : bool. if set -> compressed_length : 13 bits, then compressed_length BYTES.
//                        else    -> buffer_size BYTES.
// On the COMPRESSED branch exactly `compressed_length` bytes follow (not `buffer_size`);
// reading the wrong count desyncs the whole object array.
/// Read the variant's forge-label string table → Vec<String> indexed by label index.
/// Each label carries an offset into a shared string buffer of null-terminated UTF-8. A
/// compressed buffer is inflated; if that fails the labels come back empty but the bitstream
/// stays in sync (the compressed bytes are still consumed).
fn read_forge_labels(br: &mut BitReader) -> Vec<String> {
    let count = br.bits(9) as usize;
    // HMS_LABELDIAG=1: print-only diagnostic of the label table's layout (count / buffer size /
    // compression) as it is read.
    let diag = std::env::var("HMS_LABELDIAG").is_ok();
    if count == 0 {
        if diag { eprintln!("HMS_LABELDIAG EMPTY count=0"); }
        return Vec::new();
    }
    let mut offsets: Vec<Option<usize>> = Vec::with_capacity(count);
    for _ in 0..count {
        if br.flag() {
            offsets.push(Some(br.bits(12) as usize));
        } else {
            offsets.push(None);
        }
    }
    let buffer_size = br.bits(13) as usize;
    let compressed = br.flag();
    let buf: Vec<u8> = if compressed {
        let compressed_length = br.bits(13) as usize;
        // read_bytes consumes exactly compressed_length bytes, so the stream stays in sync
        // even if the inflate fails.
        let blob = br.read_bytes(compressed_length);
        // Layout: 4-byte BIG-ENDIAN uncompressed size, then a zlib stream (78 da).
        // Verified against retail variants: the prefix always equals `buffer_size`.
        let out = if blob.len() > 4 {
            let want = u32::from_be_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
            match miniz_oxide::inflate::decompress_to_vec_zlib(&blob[4..]) {
                Ok(v) => {
                    if diag && v.len() != want {
                        eprintln!("HMS_LABELDIAG size mismatch: prefix={want} inflated={}", v.len());
                    }
                    v
                }
                Err(e) => {
                    if diag { eprintln!("HMS_LABELDIAG inflate failed: {e:?}"); }
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        if diag {
            eprintln!("HMS_LABELDIAG COMPRESSED count={count} buffer_size={buffer_size} clen={compressed_length} inflated={}", out.len());
        }
        out
    } else {
        if diag {
            eprintln!("HMS_LABELDIAG PLAIN count={count} buffer_size={buffer_size}");
        }
        br.read_bytes(buffer_size)
    };
    let read_str = |off: usize| -> String {
        if off >= buf.len() {
            return String::new();
        }
        let end = buf[off..].iter().position(|&b| b == 0).map(|p| off + p).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[off..end]).into_owned()
    };
    offsets.iter().map(|o| o.map(read_str).unwrap_or_default()).collect()
}

/// The exact inverse of `read_forge_labels`: write the forge-label string table UNCOMPRESSED. The
/// engine reads both compressed and uncompressed forms, so a re-encode always emits the simple
/// uncompressed layout. Field caps (c_single_language_string_table<…,12,13,9>): count is 9-bit
/// (<512), each offset is 12-bit (<4096), buffer_size is 13-bit (<8192). A label that would push
/// any of those over the limit is DROPPED (clean truncation) rather than corrupting the widths. An
/// empty label set writes just `count=0`, matching `read_forge_labels`' early-out.
fn write_forge_labels(w: &mut BitWriter, labels: &[String]) {
    // Build the shared NUL-terminated UTF-8 buffer, recomputing each string's offset as we go and
    // stopping before any field cap is exceeded.
    let mut buf: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    for s in labels {
        if offsets.len() >= 511 {
            break; // count must stay < 512 (9-bit)
        }
        let off = buf.len();
        let need = s.as_bytes().len() + 1; // +1 for the NUL terminator
        if off >= 4096 || off + need > 8191 {
            break; // offset must stay < 4096 (12-bit); buffer_size < 8192 (13-bit)
        }
        offsets.push(off);
        buf.extend_from_slice(s.as_bytes());
        buf.push(0);
    }
    if offsets.is_empty() {
        w.bits(9, 0);
        return;
    }
    w.bits(9, offsets.len() as u64);
    for &off in &offsets {
        w.flag(true); // exists
        w.bits(12, off as u64);
    }
    w.bits(13, buf.len() as u64); // buffer_size
    w.flag(false); // compressed = false
    for &b in &buf {
        w.bits(8, b as u64);
    }
}

/// The `mvar` chunk payload of a BLF file, for the diagnostic tests outside this module.
#[cfg(test)]
pub fn blf_mvar_payload_pub(data: &[u8]) -> Option<Vec<u8>> { blf_mvar_payload(data) }

// Some shipped variants (e.g. Empire.mvar) store the `mvar` chunk INSIDE a `_cmp`
// chunk: payload = u8 compression type (0 = zlib) + u32 BE uncompressed size + zlib stream, and
// the stream inflates to the ordinary `mvar` chunk (header included). Expand that in place so
// every reader/writer below sees the plain layout; saves write the plain layout (the game reads
// both). The `_eof` chunk carries the byte offset of itself, so it is re-stamped after expansion.
pub fn expand_blf_cmp(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    let mut expanded = false;
    while pos + 12 <= data.len() {
        let magic = &data[pos..pos + 4];
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 12 || pos + size > data.len() {
            break;
        }
        if magic == b"_cmp" && size >= 12 + 5 {
            let body = &data[pos + 12..pos + size];
            let ctype = body[0];
            let usize_ = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
            let inflated = if ctype == 0 {
                miniz_oxide::inflate::decompress_to_vec_zlib(&body[5..]).ok()
            } else {
                None
            };
            match inflated {
                Some(v) if v.len() == usize_ && v.len() >= 12 => {
                    out.extend_from_slice(&v);
                    expanded = true;
                }
                _ => {
                    eprintln!("mvar: unsupported _cmp chunk (type {ctype}, {} B -> {usize_} B)", size - 17);
                    out.extend_from_slice(&data[pos..pos + size]);
                }
            }
        } else if magic == b"_eof" && expanded && size >= 16 {
            // Re-stamp the "bytes before _eof" field for the expanded layout.
            out.extend_from_slice(&data[pos..pos + 12]);
            let before = (out.len() - 12) as u32;
            out.extend_from_slice(&before.to_be_bytes());
            out.extend_from_slice(&data[pos + 16..pos + size]);
        } else {
            out.extend_from_slice(&data[pos..pos + size]);
        }
        pos += size;
    }
    if pos < data.len() {
        out.extend_from_slice(&data[pos..]);
    }
    out
}

/// `std::fs::read` + `_cmp` expansion: the form every reader/writer in this module works on.
pub fn read_expanded(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path).map(|d| expand_blf_cmp(&d))
}

fn blf_mvar_payload(data: &[u8]) -> Option<Vec<u8>> {
    let data: std::borrow::Cow<[u8]> = if data.windows(4).take(4096).any(|w| w == b"_cmp") {
        std::borrow::Cow::Owned(expand_blf_cmp(data))
    } else {
        std::borrow::Cow::Borrowed(data)
    };
    let data = &data[..];
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let magic = &data[pos..pos + 4];
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 8 || pos + size > data.len() {
            break;
        }
        if magic == b"mvar" && size >= 36 {
            return Some(data[pos + 36..pos + size].to_vec());
        }
        pos += size;
    }
    None
}

/// A parsed .mvar: the base map it targets + its placed forge objects.
#[derive(Clone, Debug)]
pub struct Variant {
    /// `c_map_variant::m_map_id` — identifies the BASE (canvas) map this variant is
    /// authored on. Must match the loaded map before its objects make sense; otherwise
    /// the correct base map has to be loaded first.
    pub map_id: u32,
    /// `m_number_of_placeable_object_quotas` — the count of distinct object types the
    /// variant's quota array holds (1:1 with the base map's sandbox palette). Should equal
    /// the number of distinct sandbox-palette object types; a mismatch means the offline
    /// palette walk order/count diverges from the engine's quota order.
    pub num_quotas: u32,
    pub objects: Vec<PlacedObject>,
    /// Forge label string table (indexed by an object's `label_idx`). Empty when the variant
    /// has none (or a compressed buffer failed to inflate).
    pub labels: Vec<String>,
    /// Variant display Title (header widechar string). Editable → persisted on save.
    pub title: String,
    /// Variant Description (header widechar string). Editable → persisted on save.
    pub description: String,
    /// CreatedBy display name (header author). Editable → persisted on save.
    pub author: String,
    /// ModifiedBy display name (header editor). Editable → persisted on save.
    pub editor: String,
    /// Every other GLOBAL field of the file (content header + variant body + quota table),
    /// decoded from the engine's own field order (reach_tag_test.exe `sub_14053A5F0` content
    /// header, `sub_140216020` body; docs/reach_mvar_layout.md).
    pub globals: VariantGlobals,
}

/// The global (non-object) fields of a map variant, in one game-neutral shape so
/// the Variant-properties panel and the `variant get/set` script verbs serve Reach and Halo 4
/// alike. Engine field names in the comments; docs/reach_mvar_layout.md and
/// docs/halo4_mvar_layout.md carry the bit layout and the runtime semantics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VariantGlobals {
    /// "Reach" or "Halo 4".
    pub game: &'static str,
    /// Content header `type` (raw 4 bits minus 1): 5 = map variant on every shipped file.
    pub content_type: i8,
    /// `file-size`: bytes up to and including the `_eof` chunk (the `_fsm` chunk is not counted).
    pub file_size: u32,
    /// `uid` / `parent-uid` / `root-uid` / `game-id`. The game keeps the template's ids on every
    /// Forge save (seen on the game-saved files), so HMS copies them verbatim on Save As too.
    pub uid: u64,
    pub parent_uid: u64,
    pub root_uid: u64,
    pub game_id: u64,
    /// `activity` (Reach: 3 bits minus 1, 4 on every shipped file; Halo 4: 2 bits raw, 1 on every
    /// shipped file). 2 = matchmaking adds the 16-bit `hopper-id`. Read-only.
    pub activity: i8,
    /// `game-mode` (3 bits): Reach 3 = multiplayer (1 = campaign / 2 = firefight add extra
    /// header blocks); Halo 4 4 on every shipped file. Read-only.
    pub game_mode: u8,
    /// `game-engine-type` (3 bits): 0 on every shipped file of both games (Halo 4 keys its
    /// campaign / firefight tails on this field). Read-only.
    pub engine: u8,
    /// `map-id` of the content header (equals the body's `map_id` on every file seen).
    pub header_map_id: u32,
    /// `megalo-category-index` (signed 8 bits, -1 on every shipped file). Editable.
    pub category: i8,
    /// CreatedBy / ModifiedBy `timestamp` (unix seconds) + `author-xuid`, and the `author-flags`
    /// bit of each block (1 = online). Stamped by Save (see `HeaderEdits::stamp_new`).
    pub created: (u64, u64),
    pub modified: (u64, u64),
    pub created_online: bool,
    pub modified_online: bool,
    /// `hopper-id` (16 bits), only stored when `activity == 2`.
    pub hopper_id: Option<u16>,
    /// `map-variant-version` (8 bits): Reach 31 (shipped) / 32 (MCC saves, adds `mcc-map-id`);
    /// Halo 4 50 / 51.
    pub version: u8,
    /// `map-variant-checksum` (32 bits, 0xFFFFFFFF on every shipped file) - copied verbatim.
    pub checksum: u32,
    /// `m_scenario_palette_crc` (32 bits): the loader compares it with the loaded scenario's
    /// palette CRC and rebuilds the variant from the scenario on a mismatch. Copied verbatim.
    pub palette_crc: u32,
    /// `number_of_placeable_object_quotas` (9 bits) = the quota table length.
    pub num_quotas: u16,
    /// `map_id` (32 bits): the base map.
    pub map_id: u32,
    /// `built_in` / `m_built_from_xml` (1 bit each). Copied verbatim.
    pub built_in: bool,
    pub built_from_xml: bool,
    /// `world-bounds` xmin xmax ymin ymax zmin zmax (6 x f32): the quantisation box for object
    /// positions. Editable (the encoder re-quantises every object); NOTE the game replaces it
    /// with the scenario's world bounds when the variant is loaded on its map.
    pub bounds: [f32; 6],
    /// `maximum_budget` / `spent_budget` (32 bits each). Maximum editable (also rebuilt from the
    /// scenario's sandbox budget by the game on load); spent is recomputed by the game on load.
    pub budget_max: u32,
    pub budget_spent: u32,
    /// Reach `mcc-map-id` (version >= 32) / Halo 4 map GUID (version >= 51): 16 raw bytes.
    pub mcc_map_id: Option<[u8; 16]>,
    /// The quota table: `(minimum_count, maximum_count, placed_on_map)` per palette entry, in
    /// palette order (entry i = quota index i). Minimum / maximum editable; placed is recomputed
    /// on every save (and by the game on load).
    pub quotas: Vec<(u8, u8, u8)>,
    /// Bit just past the quota table = what the engine wrote (`length` = ceil(/8) for Reach;
    /// Halo 4 adds the four trait-set records).
    pub end_bit: usize,
    /// BLF wrapper facts: chunk `length` stored in the `mvar` chunk header, whether the stored
    /// SHA-1 matches `sha1(LE length || payload[..length])`, and whether an `_fsm` chunk (MCC
    /// file-share signature block) is present.
    pub chunk_length: u32,
    pub hash_ok: bool,
    pub has_fsm: bool,
}

/// The edit set for the editable global fields, seeded from `VariantGlobals` and
/// diffed against it at save time (both games).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlobalEdits {
    pub category: i8,
    pub budget_max: u32,
    pub bounds: [f32; 6],
    /// `(minimum_count, maximum_count)` per quota entry.
    pub quota_minmax: Vec<(u8, u8)>,
}

impl GlobalEdits {
    pub fn from_globals(g: &VariantGlobals) -> Self {
        Self { category: g.category, budget_max: g.budget_max, bounds: g.bounds, quota_minmax: g.quotas.iter().map(|q| (q.0, q.1)).collect() }
    }
    /// The differences against `g` as `Some` fields (None = unchanged / keep the source bits).
    pub fn diff(&self, g: &VariantGlobals) -> (Option<i8>, Option<u32>, Option<[f32; 6]>, Option<Vec<(u8, u8)>>) {
        let base: Vec<(u8, u8)> = g.quotas.iter().map(|q| (q.0, q.1)).collect();
        (
            (self.category != g.category).then_some(self.category),
            (self.budget_max != g.budget_max).then_some(self.budget_max),
            (self.bounds != g.bounds).then_some(self.bounds),
            (self.quota_minmax != base).then(|| self.quota_minmax.clone()),
        )
    }
    pub fn is_dirty(&self, g: &VariantGlobals) -> bool { *self != Self::from_globals(g) }
}

/// Unix seconds -> "YYYY-MM-DD HH:MM UTC" (0 -> "-"), no external crate.
pub fn unix_date(ts: u64) -> String {
    if ts == 0 { return "-".into(); }
    let days = (ts / 86_400) as i64;
    let secs = ts % 86_400;
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", secs / 3600, (secs % 3600) / 60)
}

/// Names for the enum-valued header fields as far as they are known (both games;
/// the value is always printed too).
pub fn activity_name(game: &str, v: i8) -> &'static str {
    // Reach stores the raw 3-bit value minus 1 (4 on every file); Halo 4 the raw 2 bits (1 on
    // every file). Only "2 = matchmaking" is engine-proven (it gates the hopper-id field).
    let _ = game;
    match v { -1 => "none", 2 => "matchmaking", _ => "" }
}
pub fn game_mode_name(game: &str, v: u8) -> &'static str {
    if game == "Halo 4" { return match v { 0 => "none", 4 => "multiplayer", _ => "" }; }
    match v { 0 => "none", 1 => "campaign", 2 => "firefight", 3 => "multiplayer", 4 => "mainmenu", _ => "" }
}
pub fn engine_name(game: &str, v: u8) -> &'static str {
    if game == "Halo 4" { return match v { 0 => "none", 3 => "campaign", 4 => "firefight", _ => "" }; }
    match v { 0 => "none", 1 => "sandbox", 2 => "megalo", 3 => "campaign", 4 => "survival", _ => "" }
}

/// The `variant get [field]` report (both games): every global with its ENGINE
/// field name, the edited value where one is pending, the quota table with palette names.
pub fn globals_report(g: &VariantGlobals, e: &GlobalEdits, names: &[String], field: Option<&str>, title: &str, description: &str, author: &str, editor: &str, labels: &[String]) -> Result<String, String> {
    let pend = |same: bool| if same { "" } else { "  (edited, pending save)" };
    let bounds_line = |b: [f32; 6]| format!("x {:.3}..{:.3}  y {:.3}..{:.3}  z {:.3}..{:.3}", b[0], b[1], b[2], b[3], b[4], b[5]);
    let quota_rows = || -> String {
        let mut out = String::new();
        for (i, q) in g.quotas.iter().enumerate() {
            let (mn, mx) = e.quota_minmax.get(i).copied().unwrap_or((q.0, q.1));
            let flag = if (mn, mx) != (q.0, q.1) { " *" } else { "" };
            out.push_str(&format!("  {i:3}  min {mn:3}  max {mx:3}  placed {:3}  {}{flag}\n", q.2, names.get(i).map(|s| s.as_str()).unwrap_or("")));
        }
        out
    };
    let mut lines: Vec<String> = Vec::new();
    let mut put = |k: &str, v: String| lines.push(format!("{k}: {v}"));
    let want = field.map(|f| f.trim().to_ascii_lowercase());
    let all = want.is_none();
    let is = |k: &str| all || want.as_deref() == Some(k);
    if is("game") { put("game", g.game.to_string()); }
    if is("name") || is("title") { put("name", format!("\"{title}\"")); }
    if is("description") { put("description", format!("\"{description}\"")); }
    if is("author") || is("created") { put("CreatedBy", format!("\"{author}\"  timestamp {} ({})  author-xuid {:016x}  online {}", g.created.0, unix_date(g.created.0), g.created.1, g.created_online)); }
    if is("editor") || is("modified") { put("ModifiedBy", format!("\"{editor}\"  timestamp {} ({})  author-xuid {:016x}  online {}", g.modified.0, unix_date(g.modified.0), g.modified.1, g.modified_online)); }
    if is("type") { put("type", format!("{} ({})", g.content_type, if g.content_type == 5 { "map variant" } else { "?" })); }
    if is("file-size") || is("filesize") { put("file-size", g.file_size.to_string()); }
    if is("uid") { put("uid", format!("{:016x}", g.uid)); }
    if is("parent-uid") { put("parent-uid", format!("{:016x}", g.parent_uid)); }
    if is("root-uid") { put("root-uid", format!("{:016x}", g.root_uid)); }
    if is("game-id") { put("game-id", format!("{:016x}", g.game_id)); }
    if is("activity") { put("activity", format!("{} {}", g.activity, activity_name(g.game, g.activity))); }
    if is("game-mode") || is("gamemode") { put("game-mode", format!("{} {}", g.game_mode, game_mode_name(g.game, g.game_mode))); }
    if is("engine") || is("game-engine-type") { put("game-engine-type", format!("{} {}", g.engine, engine_name(g.game, g.engine))); }
    if is("map-id") || is("mapid") || is("map_id") { put("map-id", format!("{} (header {})", g.map_id, g.header_map_id)); }
    if is("category") || is("megalo-category-index") { put("megalo-category-index", format!("{}{}", e.category, pend(e.category == g.category))); }
    if is("hopper-id") || is("hopper") { put("hopper-id", g.hopper_id.map_or("- (activity != 2)".to_string(), |h| h.to_string())); }
    if is("version") { put("map-variant-version", g.version.to_string()); }
    if is("checksum") { put("map-variant-checksum", format!("{:08x}", g.checksum)); }
    if is("palette-crc") || is("crc") { put("m_scenario_palette_crc", format!("{:08x}", g.palette_crc)); }
    if is("quotas") || is("num-quotas") { put("number_of_placeable_object_quotas", g.num_quotas.to_string()); }
    if is("built-in") || is("builtin") { put("built_in", g.built_in.to_string()); }
    if is("built-from-xml") { put("m_built_from_xml", g.built_from_xml.to_string()); }
    if is("bounds") || is("world-bounds") { put("world-bounds", format!("{}{}", bounds_line(e.bounds), pend(e.bounds == g.bounds))); if e.bounds != g.bounds { put("world-bounds (file)", bounds_line(g.bounds)); } }
    if is("budget") || is("budget-max") || is("maximum_budget") { put("maximum_budget", format!("{}{}", e.budget_max, pend(e.budget_max == g.budget_max))); }
    if is("budget") || is("budget-spent") || is("spent_budget") { put("spent_budget", format!("{} (recomputed on save)", g.budget_spent)); }
    if is("mcc-map-id") || is("guid") { put("mcc-map-id", g.mcc_map_id.map_or("- (version < 32 / 51)".to_string(), |x| x.iter().map(|b| format!("{b:02x}")).collect())); }
    if is("labels") { put("labels", format!("{} {:?}", labels.len(), labels)); }
    if is("chunk") || is("length") { put("mvar chunk", format!("length {} (bits {}), sha1 {}, _fsm {}", g.chunk_length, g.end_bit, if g.hash_ok { "ok" } else { "STALE" }, if g.has_fsm { "present" } else { "absent" })); }
    if is("quota") || is("quota-table") { lines.push(format!("quota table ({} entries; min/max editable, * = edited):\n{}", g.quotas.len(), quota_rows())); }
    if lines.is_empty() { return Err(format!("variant get: unknown field '{}'", field.unwrap_or(""))); }
    Ok(lines.join("\n"))
}

/// `variant set <field> <value>` (both games). Returns (message, header-string
/// edited). Fields: name|title, description, author, editor, category, budget|budget-max,
/// bounds <xmin xmax ymin ymax zmin zmax> | bounds default, quotamin <index|name> <n>,
/// quotamax <index|name> <n>, quota <index|name> <min> <max>.
pub fn apply_global_set(g: &VariantGlobals, e: &mut GlobalEdits, field: &str, value: &str, title: &mut String, description: &mut String, author: &mut String, editor: &mut String) -> Result<(String, bool), String> {
    let f = field.trim().to_ascii_lowercase();
    let v = value.trim();
    let quota_index = |key: &str| -> Result<usize, String> {
        if let Ok(i) = key.parse::<usize>() { if i < g.quotas.len() { return Ok(i); } return Err(format!("quota index {i} out of range (0..{})", g.quotas.len())); }
        Err(format!("quota '{key}': give the index (0..{}) - names are listed by `variant get quota`", g.quotas.len()))
    };
    match f.as_str() {
        "name" | "title" => { *title = v.to_string(); Ok((format!("name = \"{v}\""), true)) }
        "description" | "desc" => { *description = v.to_string(); Ok((format!("description = \"{v}\""), true)) }
        "author" => { *author = v.chars().take(16).collect(); Ok((format!("author = \"{author}\""), true)) }
        "editor" => { *editor = v.chars().take(16).collect(); Ok((format!("editor = \"{editor}\""), true)) }
        "category" | "megalo-category-index" => {
            let c: i32 = v.parse().map_err(|_| "category: an integer -128..127".to_string())?;
            if !(-128..=127).contains(&c) { return Err("category must be -128..127".into()); }
            e.category = c as i8;
            Ok((format!("megalo-category-index = {c}"), true))
        }
        "budget" | "budget-max" | "budget_max" | "maximum_budget" | "max-budget" => {
            let b: u32 = v.parse().map_err(|_| "budget: a non-negative integer".to_string())?;
            e.budget_max = b;
            Ok((format!("maximum_budget = {b} (NOTE the game restores the map's own sandbox budget when the variant loads on its map)"), true))
        }
        "bounds" | "world-bounds" => {
            if v.eq_ignore_ascii_case("default") || v.eq_ignore_ascii_case("reset") || v.eq_ignore_ascii_case("file") {
                e.bounds = g.bounds;
                return Ok(("world-bounds reset to the file's".into(), true));
            }
            let nums: Vec<f32> = v.split_whitespace().map(|t| t.parse::<f32>().map_err(|_| format!("bounds: '{t}' is not a number"))).collect::<Result<_, _>>()?;
            if nums.len() != 6 { return Err("bounds: xmin xmax ymin ymax zmin zmax (6 numbers) or `default`".into()); }
            let b = [nums[0], nums[1], nums[2], nums[3], nums[4], nums[5]];
            if (0..3).any(|i| !(b[2 * i + 1] > b[2 * i]) || !b[2 * i].is_finite()) { return Err("bounds: max must exceed min on every axis".into()); }
            e.bounds = b;
            Ok((format!("world-bounds = x {}..{} y {}..{} z {}..{} (every object is re-quantised on save; the game replaces the box with the map's world bounds when it loads the variant on its own map)", b[0], b[1], b[2], b[3], b[4], b[5]), true))
        }
        "quotamin" | "quota-min" | "quotamax" | "quota-max" | "quota" => {
            let toks: Vec<&str> = v.split_whitespace().collect();
            let key = toks.first().ok_or("quota: <index> <value>")?;
            let i = quota_index(key)?;
            if e.quota_minmax.len() < g.quotas.len() { e.quota_minmax = g.quotas.iter().map(|q| (q.0, q.1)).collect(); }
            let n8 = |t: Option<&&str>| -> Result<u8, String> { let n: u32 = t.ok_or("quota: missing value")?.parse().map_err(|_| "quota: a value 0..255".to_string())?; if n > 255 { return Err("quota: 0..255".into()); } Ok(n as u8) };
            match f.as_str() {
                "quotamin" | "quota-min" => { e.quota_minmax[i].0 = n8(toks.get(1))?; }
                "quotamax" | "quota-max" => { e.quota_minmax[i].1 = n8(toks.get(1))?; }
                _ => { e.quota_minmax[i] = (n8(toks.get(1))?, n8(toks.get(2))?); }
            }
            let (mn, mx) = e.quota_minmax[i];
            Ok((format!("quota {i}: min {mn} max {mx} (placed {} - a maximum below it is raised to it on save)", g.quotas[i].2), true))
        }
        _ => Err(format!("variant set: unknown field '{field}' (name, description, author, editor, category, budget, bounds, quotamin, quotamax, quota)")),
    }
}

/// The Reach `mvar` chunk: the 20-byte SHA-1 at +12 and the big-endian `length` at +32 (bytes
/// actually written by the encoder) sit before the bitstream at +36 - the same layout as
/// Halo 4's. The hash is plain `sha1(LE length || payload[..length])` (holds on every shipped
/// file); reach_tag_test.exe `sub_14053C670` recomputes it but only refuses a mismatch when
/// `byte_143E857C4` is set, a zero-initialised flag no code writes, so a stale hash loads - HMS
/// still writes a correct one.
pub fn reach_chunk_hash(payload: &[u8], length: u32) -> [u8; 20] {
    crate::h4::mvar::mvar_chunk_hash(payload, length)
}

/// Parse a .mvar file → base map id + placed forge objects.
pub fn parse_variant(path: &std::path::Path) -> Option<Variant> {
    let data = read_expanded(path).ok()?;
    let payload = blf_mvar_payload(&data)?;
    let (map_id, num_quotas, _labels, objects) = parse_payload(&payload);
    // Second header walk to capture the editable header strings, the forge-label table with its
    // exact bit span and the global fields (kept separate from parse_payload, whose object-decode
    // tests pin its signature).
    let (hi, quotas, end_bit) = {
        let mut r = BitReader::new(&payload);
        let hi = parse_header(&mut r);
        for _ in 0..651 {
            let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext);
        }
        let mut quotas = Vec::with_capacity(hi.num_quotas as usize);
        for _ in 0..hi.num_quotas as usize {
            if r.pos + 24 > 229408 { break; }
            quotas.push((r.bits(8) as u8, r.bits(8) as u8, r.bits(8) as u8));
        }
        (hi, quotas, r.pos)
    };
    // BLF-level facts (the mvar chunk header + the presence of the signature chunk)
    let (chunk_length, hash_ok, has_fsm) = {
        let mut pos = 0usize;
        let (mut len, mut ok, mut fsm) = (0u32, false, false);
        while pos + 12 <= data.len() {
            let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
            if size < 12 || pos + size > data.len() { break; }
            match &data[pos..pos + 4] {
                b"mvar" if size >= 36 => {
                    len = u32::from_be_bytes([data[pos + 32], data[pos + 33], data[pos + 34], data[pos + 35]]);
                    ok = (len as usize) <= payload.len() && data[pos + 12..pos + 32] == reach_chunk_hash(&payload, len)[..];
                }
                b"_fsm" => fsm = true,
                _ => {}
            }
            pos += size;
        }
        (len, ok, fsm)
    };
    let globals = VariantGlobals {
        game: "Reach",
        content_type: hi.content_type, file_size: hi.file_size, uid: hi.uid, parent_uid: hi.parent_uid, root_uid: hi.root_uid, game_id: hi.game_id,
        activity: hi.activity, game_mode: hi.game_mode, engine: hi.engine, header_map_id: hi.header_map_id, category: hi.category,
        created: hi.created, modified: hi.modified, created_online: hi.created_online, modified_online: hi.modified_online, hopper_id: hi.hopper_id,
        version: hi.version, checksum: hi.checksum, palette_crc: hi.palette_crc, num_quotas: hi.num_quotas as u16, map_id: hi.map_id,
        built_in: hi.built_in, built_from_xml: hi.built_from_xml, bounds: hi.bb, budget_max: hi.budget_max, budget_spent: hi.budget_spent,
        mcc_map_id: hi.mcc_map_id, quotas, end_bit, chunk_length, hash_ok, has_fsm,
    };
    Some(Variant {
        map_id,
        num_quotas,
        objects,
        labels: hi.labels,
        title: hi.title,
        description: hi.description,
        author: hi.author,
        editor: hi.editor,
        globals,
    })
}

fn parse_payload(payload: &[u8]) -> (u32, u32, Vec<String>, Vec<PlacedObject>) {
    let mut br = BitReader::new(payload);
    let _type = br.bits(4);
    let _file_len = br.u32_le();
    for _ in 0..4 {
        br.u64_le(); // Unk08/10/18/20
    }
    let activity = br.bits(3) as i32 - 1;
    let game_mode = br.bits(3);
    let _engine = br.bits(3);
    let _hdr_mapid = br.bits(32);
    let _engine_cat = br.bits(8);
    skip_content_author(&mut br); // CreatedBy
    skip_content_author(&mut br); // ModifiedBy
    br.skip_widechar_stop(128); // Title
    br.skip_widechar_stop(128); // Description
    match _type as i64 - 1 {
        3 | 4 => { br.bits(32); } // film: seconds
        6 => { br.bits(8); } // game variant: icon-index
        _ => {}
    }
    if activity == 2 {
        br.u16_be(); // HopperId
    }
    if game_mode == 1 {
        br.bits(8);
        br.bits(2);
        br.bits(2);
        br.bits(8);
        br.bits(32);
    } else if game_mode == 2 {
        br.bits(2);
        br.bits(32);
    }
    let version = br.bits(8);
    br.bits(64);
    let num_quotas = br.bits(9) as u32; // number-of-placeable-object-quotas
    let map_id = br.bits(32) as u32; // c_map_variant::m_map_id — the base/canvas map
    br.bits(1); // built-in
    br.bits(1); // built-from-xml
    let bb = [
        br.f32_be(), br.f32_be(), // Xmin Xmax
        br.f32_be(), br.f32_be(), // Ymin Ymax
        br.f32_be(), br.f32_be(), // Zmin Zmax
    ];
    let ext = [bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]];
    br.bits(32); // BudgetMax
    br.bits(32); // BudgetSpent
    let labels = read_forge_labels(&mut br);
    if version >= 32 {
        br.bits_signed(32);
        br.bits(16);
        br.bits(16);
        br.bits(64);
    }

    let axis = compute_axis_bits(21, ext);
    let mut out = Vec::new();
    for slot_idx in 0..651u16 {
        if !br.flag() {
            continue; // not present
        }
        let flags = br.bits(2) as u8;
        let folder = if br.flag() { 0xFFFF } else { br.bits(8) as u16 };
        let item = if br.flag() { 0xFF } else { br.bits(5) as u8 };

        // Position (bbox-relative quantised). The in-bounds bool, then (out-of-bounds only) a
        // bsp-index escape, then the 3-axis quantised point.
        let in_bounds = br.flag();
        let mut bsp_index = -1i32;
        if !in_bounds {
            if !br.flag() {
                bsp_index = br.bits(2) as i32;
            }
        }
        let dec = |br: &mut BitReader, i: usize| -> f32 {
            let q = br.bits(axis[i]) as f32;
            let range = [ext[0], ext[1], ext[2]][i];
            let mn = [bb[0], bb[2], bb[4]][i];
            (0.5 + q) * (range / (1u32 << axis[i]) as f32) + mn
        };
        let x = dec(&mut br, 0);
        let y = dec(&mut br, 1);
        let z = dec(&mut br, 2);

        // Orientation: read_axes(forward_bits=14, up_bits=20). 1 bool (up-is-global);
        // if not global, a 20-bit cube-face up vector; then a 14-bit yaw over [-π, π].
        let up_is_global = br.flag();
        let up_quant = if up_is_global { 0 } else { br.bits(20) as u32 };
        let forward_angle_q = br.bits(14) as u32;
        let spawn_rel = br.bits(10) as i32 - 1; // spawn-relative-to: parent slot index (-1 = none)

        // ObjectExtra — boundary shape + its quantised (11-bit) size values.
        let shape = br.bits(2) as u8;
        let nvals = [0usize, 1, 3, 4][shape as usize];
        let mut boundary = [0u16; 4];
        for k in 0..nvals {
            boundary[k] = br.bits(11) as u16;
        }
        let spawn_seq = br.bits_signed(8) as i32;
        let respawn = br.bits(8) as u8;
        let mp_type = br.bits(5);
        let label_idx = if !br.flag() { br.bits(8) as u16 } else { 0xFFFF };
        let placement = br.bits(8) as u8;
        // team is read_integer(4) minus 1 (blf ref scenario_map_variant.ts): 0..7 = the 8 team
        // colours (0=red,1=blue,…), 8 = neutral, raw 0 → -1 → TEAM_NONE.
        let team = {
            let r = br.bits(4) as i32 - 1;
            if r < 0 { TEAM_NONE } else { r as u8 }
        };
        let color = if br.flag() { -1 } else { br.bits(3) as i32 };
        let (mut weapon_clips, mut tele_channel, mut tele_passability, mut location_name) =
            (0u8, 0u8, 0u8, 0xFFFFu16);
        if mp_type == 1 {
            weapon_clips = br.bits(8) as u8;
        } else if (12..=14).contains(&mp_type) {
            tele_channel = br.bits(5) as u8;
            tele_passability = br.bits(5) as u8;
        } else if mp_type == 19 {
            // cached_type 19 (location): location_name_index is read_index(255,8) = 1 bool + (8 bits
            // only when the flag is CLEAR), NOT an unconditional 8 bits (blf ref); a bare 8 would
            // desync the bitstream for every object after a type-19 object.
            if !br.flag() {
                location_name = br.bits(8) as u16;
            }
        }

        if folder != 0xFFFF && item != 0xFF {
            let (fwd, up) = decode_orientation(up_is_global, up_quant, forward_angle_q);
            out.push(PlacedObject {
                folder,
                item,
                pos: [x, y, z],
                team,
                color,
                fwd,
                up,
                up_is_global,
                up_quant,
                forward_angle_q,
                spawn_seq,
                respawn,
                cached_type: mp_type as u8,
                label_idx,
                placement,
                boundary_shape: shape,
                flags,
                in_bounds,
                bsp_index,
                spawn_rel,
                slot: slot_idx,
                boundary,
                weapon_clips,
                tele_channel,
                tele_passability,
                location_name,
            });
        }
    }
    (map_id, num_quotas, labels, out)
}

// ---------------------------------------------------------------------------
// Write path — MSB-first BitWriter. Every slot is decoded into a `PlacedObject` and re-emitted
// field by field; the header is copied verbatim (or spliced), so a load→save reproduces the
// variant bit-for-bit and edits are applied to the decoded fields before encoding.
// ---------------------------------------------------------------------------

struct BitWriter {
    buf: Vec<u8>,
    pos: usize,
}
impl BitWriter {
    fn new() -> Self {
        Self { buf: Vec::new(), pos: 0 }
    }
    fn bits(&mut self, n: u32, value: u64) {
        for i in (0..n).rev() {
            let byte = self.pos >> 3;
            if byte >= self.buf.len() {
                self.buf.push(0);
            }
            let off = 7 - (self.pos & 7);
            if (value >> i) & 1 != 0 {
                self.buf[byte] |= 1 << off;
            }
            self.pos += 1;
        }
    }
    fn flag(&mut self, v: bool) {
        self.bits(1, v as u64);
    }
    fn u16_be(&mut self, v: u16) {
        self.bits(8, (v >> 8) as u64);
        self.bits(8, (v & 0xFF) as u64);
    }
    fn f32_be(&mut self, v: f32) {
        for b in v.to_be_bytes() {
            self.bits(8, b as u64);
        }
    }
    /// Write a null-terminated ASCII/UTF-8 string capped at `max_n` bytes — the exact inverse of
    /// `BitReader::read_string_stop`. When the (truncated) content is shorter than `max_n`, a 0
    /// terminator byte is written; when it fills `max_n` exactly, NO terminator is written (the
    /// reader stops at the cap), matching the on-disk convention.
    fn write_string_stop(&mut self, s: &str, max_n: usize) {
        let bytes = s.as_bytes();
        let n = bytes.len().min(max_n);
        for &b in &bytes[..n] {
            self.bits(8, b as u64);
        }
        if n < max_n {
            self.bits(8, 0); // terminator
        }
    }
    /// Write a null-terminated UTF-16BE string capped at `max_wc` wide chars — the exact inverse of
    /// `BitReader::read_widechar_stop`. Terminator (0x0000) written only when shorter than the cap.
    fn write_widechar_stop(&mut self, s: &str, max_wc: usize) {
        let units: Vec<u16> = s.encode_utf16().collect();
        let n = units.len().min(max_wc);
        for &u in &units[..n] {
            self.u16_be(u);
        }
        if n < max_wc {
            self.u16_be(0); // terminator
        }
    }
    /// Write `n`-bit two's-complement (mirrors BitReader::bits_signed).
    fn bits_signed(&mut self, n: u32, v: i64) {
        let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        self.bits(n, (v as u64) & mask);
    }
    /// Copy `n` raw bits out of `src` starting at bit `src_pos`.
    fn copy_bits(&mut self, src: &[u8], src_pos: usize, n: usize) {
        for k in 0..n {
            let sp = src_pos + k;
            let byte = sp >> 3;
            let bit = if byte < src.len() { (src[byte] >> (7 - (sp & 7))) & 1 } else { 0 };
            self.bits(1, bit as u64);
        }
        let _ = n;
    }
    fn into_padded(mut self, total_bits: usize) -> Vec<u8> {
        let target = (total_bits + 7) >> 3;
        if self.buf.len() < target {
            self.buf.resize(target, 0);
        }
        self.buf
    }
}

/// A single edit to apply to a placed object during save. Fields left `None` keep their
/// original value; `deleted` removes the object (its slot is re-emitted as absent).
#[derive(Clone, Debug, Default)]
pub struct ObjEdit {
    pub pos: Option<[f32; 3]>,
    pub team: Option<u8>,
    pub color: Option<i32>,
    /// New placement-flags byte (physics / symmetry / placed-at-start / flags). See PLACE_* consts.
    pub placement: Option<u8>,
    /// New spawn sequence (−100..100). Retail = spawn ordering.
    pub spawn_seq: Option<i32>,
    /// New respawn/spawn-time in seconds.
    pub respawn: Option<u8>,
    /// New FULL orientation `(fwd, up)`, re-quantised via `encode_orientation` on apply so a
    /// rotated/tilted object persists its up-vector + yaw.
    pub rot: Option<([f32; 3], [f32; 3])>,
    /// New forge label / location-name index into the variant string table (0xFFFF = none).
    pub label_idx: Option<u16>,
    /// New spawn-relative-to PARENT slot index (−1 = none), from the panel's parent picker.
    pub spawn_rel: Option<i32>,
    pub deleted: bool,
}

/// New values for the four editable header strings, applied to the header on save. A `Some` field
/// is re-encoded (its new value written); a `None` field is copied from the ORIGINAL bitstream
/// verbatim — so an untouched field round-trips bit-for-bit even when a sibling field changed (this
/// matters for names that aren't clean UTF-8, which would otherwise be mangled by a String
/// round-trip). `author`=CreatedBy Name (cap 16 bytes), `editor`=ModifiedBy Name (cap 16 bytes),
/// `title`/`description`=UTF-16 header strings (cap 128 wide chars).
#[derive(Clone, Debug, Default)]
pub struct HeaderEdits {
    pub title: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub editor: Option<String>,
    /// New forge-label string table. `Some` → re-encode the WHOLE table from this vec (always written
    /// UNcompressed via `write_forge_labels`); `None` → copy the original table bits verbatim.
    pub labels: Option<Vec<String>>,
    /// CreatedBy / ModifiedBy (unix timestamp, xuid): the 128 bits right before each name.
    /// `stamp_new` / `stamp_modified` fill them the way the game does; `None` copies the source
    /// bits. `rebuild_file` mirrors a stamp into the `chdr` chunk as well (MCC reads the
    /// bitstream, not the chdr, for its listing).
    pub created: Option<(u64, u64)>,
    pub modified: Option<(u64, u64)>,
    /// `megalo-category-index` (signed 8 bits, -1 = none).
    pub category: Option<i8>,
    /// `maximum_budget`.
    pub budget_max: Option<u32>,
    /// `world-bounds` xmin xmax ymin ymax zmin zmax. Every object is re-quantised
    /// against the new box (a position outside it is clamped to the edge, as the engine's own
    /// writer does); each axis must have max > min.
    pub bounds: Option<[f32; 6]>,
    /// Per-quota-entry `(minimum_count, maximum_count)`; entry i replaces quota
    /// i, entries past the vec keep the source values. `placed_on_map` is always recomputed
    /// and `maximum_count` still raised to it.
    pub quotas: Option<Vec<(u8, u8)>>,
}

impl HeaderEdits {
    pub fn is_empty(&self) -> bool {
        self.title.is_none() && self.description.is_none() && self.author.is_none() && self.editor.is_none() && self.labels.is_none()
            && self.created.is_none() && self.modified.is_none()
            && self.category.is_none() && self.budget_max.is_none() && self.bounds.is_none() && self.quotas.is_none()
    }
    /// A NEW variant the way MCC stamps one (as on the game-saved files under
    /// `LocalFiles\<xuid>\HaloReach\Map`): created = (now, xuid), modified = (0, 0). The unique
    /// id is left as the template's: the game keeps the base variant's id on every derived save
    /// (30+ of one player's files share `567b3fca0707e5ad`, Forge World's), so that is what a
    /// game-made file carries and the listing does not key on it.
    pub fn stamp_new(&mut self, xuid: u64) {
        self.created = Some((crate::h4::mvar::unix_now(), xuid));
        self.modified = Some((0, 0));
    }
    /// A re-save of an existing variant the way MCC stamps one: created kept, modified = (now, xuid).
    pub fn stamp_modified(&mut self, xuid: u64) {
        self.modified = Some((crate::h4::mvar::unix_now(), xuid));
    }
}

/// The parsed variant header: geometry needed to (de)quantise object positions, the four editable
/// display strings, and the exact bit spans of each string field so the encoder can splice new
/// strings in while copying every surrounding bit verbatim.
struct HeaderInfo {
    bb: [f32; 6],
    /// Number of placeable-object QUOTA entries stored after the 651-slot array.
    num_quotas: u32,
    ext: [f32; 3],
    axis: [u32; 3],
    /// CreatedBy Name.
    author: String,
    /// ModifiedBy Name.
    editor: String,
    /// (Timestamp, Xuid) of the CreatedBy / ModifiedBy blocks (the 128 bits before each name).
    created: (u64, u64),
    modified: (u64, u64),
    title: String,
    description: String,
    /// bit span [start,end) of the CreatedBy Name field (end = just before IsOnline1).
    name1_start: usize,
    name1_end: usize,
    /// bit span [start,end) of the ModifiedBy Name field (end = just before IsOnline2).
    name2_start: usize,
    name2_end: usize,
    /// bit pos of the first Title wide-char (= name2_end + 1, past the IsOnline2 flag).
    title_start: usize,
    /// bit pos just past the Title terminator = start of the Description string.
    desc_start: usize,
    /// bit pos just past the Description string's terminator (start of the rest of the header).
    desc_end: usize,
    /// The parsed forge-label string table (indexed by an object's `label_idx`).
    labels: Vec<String>,
    /// bit span [start,end) of the WHOLE forge-label-table field (the region `read_forge_labels`
    /// consumes: count + per-string exists/offset + buffer_size + compressed flag + buffer). Lets
    /// the encoder splice a re-encoded label table in while copying every surrounding bit verbatim.
    label_start: usize,
    label_end: usize,
    // The remaining global fields (engine names: reach_tag_test.exe
    // `sub_14053A5F0` content header, `sub_140216020` body) and the bit positions of the
    // editable ones (`category_pos`: the 8-bit megalo-category-index; `bounds_pos`: the first of
    // the six world-bounds floats, followed by maximum_budget at +192 and spent_budget at +224).
    content_type: i8,
    file_size: u32,
    uid: u64,
    parent_uid: u64,
    root_uid: u64,
    game_id: u64,
    activity: i8,
    game_mode: u8,
    engine: u8,
    header_map_id: u32,
    category: i8,
    category_pos: usize,
    created_online: bool,
    modified_online: bool,
    hopper_id: Option<u16>,
    version: u8,
    checksum: u32,
    palette_crc: u32,
    map_id: u32,
    built_in: bool,
    built_from_xml: bool,
    bounds_pos: usize,
    budget_max: u32,
    budget_spent: u32,
    mcc_map_id: Option<[u8; 16]>,
}

/// Consume the variant HEADER (everything before the 651-slot object array) from `br`, leaving
/// `br` positioned at the first object slot. Captures the bbox/extents/axis bit widths AND the four
/// editable header strings with their exact bit spans. Mirrors `parse_payload`'s header reads.
fn parse_header(br: &mut BitReader) -> HeaderInfo {
    // Content header (sub_14053A5F0): type 4, file-size 32, uid / parent-uid / root-uid / game-id
    // 64 each (all MSB-first), activity 3 (minus 1), game-mode 3, game-engine-type 3, map-id 32,
    // megalo-category-index 8 (signed).
    let content_type = br.bits(4) as i8 - 1;
    let file_size = br.bits(32) as u32;
    let uid = br.bits(64);
    let parent_uid = br.bits(64);
    let root_uid = br.bits(64);
    let game_id = br.bits(64);
    let activity = br.bits(3) as i32 - 1;
    let game_mode = br.bits(3);
    let engine = br.bits(3) as u8;
    let header_map_id = br.bits(32) as u32;
    let category_pos = br.pos;
    let category = br.bits(8) as u8 as i8;
    let ((author, name1_start, name1_end), created) = br.read_content_author_stamped(); // CreatedBy
    let created_online = (br.buf[(br.pos - 1) >> 3] >> (7 - ((br.pos - 1) & 7))) & 1 != 0;
    let ((editor, name2_start, name2_end), modified) = br.read_content_author_stamped(); // ModifiedBy
    let modified_online = (br.buf[(br.pos - 1) >> 3] >> (7 - ((br.pos - 1) & 7))) & 1 != 0;
    let title_start = br.pos;
    let title = br.read_widechar_stop(128);
    let desc_start = br.pos;
    let description = br.read_widechar_stop(128);
    let desc_end = br.pos;
    // type-specific tail: film (3 / 4) `seconds` 32, game variant (6) `icon-index` 8 - never on a
    // map variant (5); then `hopper-id` 16 when activity == 2; then the campaign (game-mode 1:
    // campaign-id 8, difficulty-level 2, metagame-scoring 2, insertion-point 8, skull-flags 32)
    // or firefight (game-mode 2: difficulty-level 2, skull-flags 32) block.
    match content_type {
        3 | 4 => { br.bits(32); }
        6 => { br.bits(8); }
        _ => {}
    }
    let hopper_id = if activity == 2 { Some(br.u16_be()) } else { None };
    if game_mode == 1 {
        br.bits(8);
        br.bits(2);
        br.bits(2);
        br.bits(8);
        br.bits(32);
    } else if game_mode == 2 {
        br.bits(2);
        br.bits(32);
    }
    // Body (sub_140216020): map-variant-version 8, map-variant-checksum 32,
    // m_scenario_palette_crc 32, number_of_placeable_object_quotas 9, map_id 32, built_in 1,
    // m_built_from_xml 1, world-bounds 6 x 32, maximum_budget 32, spent_budget 32, the forge
    // label string table, then (version >= 32) mcc-map-id 128.
    let version = br.bits(8) as u8;
    let checksum = br.bits(32) as u32;
    let palette_crc = br.bits(32) as u32;
    let num_quotas = br.bits(9) as u32; // entry count for the table after the slots
    let map_id = br.bits(32) as u32;
    let built_in = br.flag();
    let built_from_xml = br.flag();
    let bounds_pos = br.pos;
    let bb = [br.f32_be(), br.f32_be(), br.f32_be(), br.f32_be(), br.f32_be(), br.f32_be()];
    let ext = [bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]];
    let budget_max = br.bits(32) as u32;
    let budget_spent = br.bits(32) as u32;
    let label_start = br.pos;
    let labels = read_forge_labels(br);
    let label_end = br.pos;
    let mcc_map_id = if version >= 32 {
        let mut g = [0u8; 16];
        for b in g.iter_mut() { *b = br.bits(8) as u8; }
        Some(g)
    } else {
        None
    };
    let axis = compute_axis_bits(21, ext);
    HeaderInfo {
        num_quotas,
        bb, ext, axis, author, editor, created, modified, title, description,
        name1_start, name1_end, name2_start, name2_end, title_start, desc_start, desc_end,
        labels, label_start, label_end,
        content_type, file_size, uid, parent_uid, root_uid, game_id, activity: activity as i8, game_mode: game_mode as u8, engine, header_map_id,
        category, category_pos, created_online, modified_online, hopper_id, version, checksum, palette_crc, map_id, built_in, built_from_xml,
        bounds_pos, budget_max, budget_spent, mcc_map_id,
    }
}

/// Read ONE object slot (present flag + all fields). Returns `None` for an absent slot. Mirrors
/// `parse_payload`'s per-slot reads exactly — the symmetric counterpart of `write_slot`. Note a
/// PRESENT slot may still be "empty" (folder=0xFFFF / item=0xFF); such slots are re-emitted
/// verbatim but never enumerated as real objects.
fn read_slot(br: &mut BitReader, axis: [u32; 3], bb: [f32; 6], ext: [f32; 3]) -> Option<PlacedObject> {
    if !br.flag() {
        return None;
    }
    let flags = br.bits(2) as u8;
    let folder = if br.flag() { 0xFFFF } else { br.bits(8) as u16 };
    let item = if br.flag() { 0xFF } else { br.bits(5) as u8 };
    let in_bounds = br.flag();
    let mut bsp_index = -1i32;
    if !in_bounds && !br.flag() {
        bsp_index = br.bits(2) as i32;
    }
    let dec = |br: &mut BitReader, i: usize| -> f32 {
        let q = br.bits(axis[i]) as f32;
        let range = ext[i];
        let mn = [bb[0], bb[2], bb[4]][i];
        (0.5 + q) * (range / (1u32 << axis[i]) as f32) + mn
    };
    let x = dec(br, 0);
    let y = dec(br, 1);
    let z = dec(br, 2);
    let up_is_global = br.flag();
    let up_quant = if up_is_global { 0 } else { br.bits(20) as u32 };
    let forward_angle_q = br.bits(14) as u32;
    let spawn_rel = br.bits(10) as i32 - 1;
    let shape = br.bits(2) as u8;
    let nvals = [0usize, 1, 3, 4][shape as usize];
    let mut boundary = [0u16; 4];
    for k in 0..nvals {
        boundary[k] = br.bits(11) as u16;
    }
    let spawn_seq = br.bits_signed(8) as i32;
    let respawn = br.bits(8) as u8;
    let mp_type = br.bits(5);
    let label_idx = if !br.flag() { br.bits(8) as u16 } else { 0xFFFF };
    let placement = br.bits(8) as u8;
    let team = {
        let r = br.bits(4) as i32 - 1;
        if r < 0 { TEAM_NONE } else { r as u8 }
    };
    let color = if br.flag() { -1 } else { br.bits(3) as i32 };
    let (mut weapon_clips, mut tele_channel, mut tele_passability, mut location_name) = (0u8, 0u8, 0u8, 0xFFFFu16);
    if mp_type == 1 {
        weapon_clips = br.bits(8) as u8;
    } else if (12..=14).contains(&mp_type) {
        tele_channel = br.bits(5) as u8;
        tele_passability = br.bits(5) as u8;
    } else if mp_type == 19 && !br.flag() {
        location_name = br.bits(8) as u16;
    }
    let (fwd, up) = decode_orientation(up_is_global, up_quant, forward_angle_q);
    Some(PlacedObject {
        folder, item, pos: [x, y, z], team, color, fwd, up, up_is_global, up_quant, forward_angle_q,
        spawn_seq, respawn, cached_type: mp_type as u8, label_idx, placement, boundary_shape: shape,
        flags, in_bounds, bsp_index, spawn_rel, slot: 0xFFFF, boundary, weapon_clips, tele_channel, tele_passability, location_name,
    })
}

/// Write ONE object slot — the symmetric inverse of `read_slot`. `None` writes an absent slot.
fn write_slot(w: &mut BitWriter, slot: Option<&PlacedObject>, axis: [u32; 3], bb: [f32; 6], ext: [f32; 3]) {
    let Some(o) = slot else {
        w.flag(false);
        return;
    };
    w.flag(true);
    w.bits(2, o.flags as u64);
    if o.folder == 0xFFFF {
        w.flag(true);
    } else {
        w.flag(false);
        w.bits(8, o.folder as u64);
    }
    if o.item == 0xFF {
        w.flag(true);
    } else {
        w.flag(false);
        w.bits(5, o.item as u64);
    }
    w.flag(o.in_bounds);
    if !o.in_bounds {
        if o.bsp_index < 0 {
            w.flag(true);
        } else {
            w.flag(false);
            w.bits(2, o.bsp_index as u64);
        }
    }
    // Quantised in f64: `read_slot` decodes in f32 like the engine, and for a box
    // whose extent is ~2^23 steps (Forge World, any widened box) the f32 intermediate
    // `(v - min)` carries an error of up to a third of a step; encoding in f64 from the decoded
    // f32 position keeps decode -> encode a fixed point (the byte-identical gates prove the
    // original files still re-encode to the same bits).
    let enc = |w: &mut BitWriter, i: usize, v: f32| {
        let range = ext[i] as f64;
        let mn = [bb[0], bb[2], bb[4]][i] as f64;
        let scale = range / (1u64 << axis[i]) as f64;
        let maxq = (1i64 << axis[i]) - 1;
        let q = (((v as f64 - mn) / scale) - 0.5).round() as i64;
        w.bits(axis[i], q.clamp(0, maxq) as u64);
    };
    enc(w, 0, o.pos[0]);
    enc(w, 1, o.pos[1]);
    enc(w, 2, o.pos[2]);
    w.flag(o.up_is_global);
    if !o.up_is_global {
        w.bits(20, o.up_quant as u64);
    }
    w.bits(14, o.forward_angle_q as u64);
    w.bits(10, (o.spawn_rel + 1) as u64);
    w.bits(2, o.boundary_shape as u64);
    let nvals = [0usize, 1, 3, 4][o.boundary_shape as usize];
    for k in 0..nvals {
        w.bits(11, o.boundary[k] as u64);
    }
    w.bits_signed(8, o.spawn_seq as i64);
    w.bits(8, o.respawn as u64);
    w.bits(5, o.cached_type as u64);
    if o.label_idx == 0xFFFF {
        w.flag(true);
    } else {
        w.flag(false);
        w.bits(8, o.label_idx as u64);
    }
    w.bits(8, o.placement as u64);
    w.bits(4, if o.team == TEAM_NONE { 0 } else { o.team as u64 + 1 });
    if o.color < 0 {
        w.flag(true);
    } else {
        w.flag(false);
        w.bits(3, o.color as u64);
    }
    let mp = o.cached_type as u64;
    if mp == 1 {
        w.bits(8, o.weapon_clips as u64);
    } else if (12..=14).contains(&mp) {
        w.bits(5, o.tele_channel as u64);
        w.bits(5, o.tele_passability as u64);
    } else if mp == 19 {
        if o.location_name == 0xFFFF {
            w.flag(true);
        } else {
            w.flag(false);
            w.bits(8, o.location_name as u64);
        }
    }
}

/// The variant header, written once for either encoder. Either splices the edited
/// strings in, or copies every header bit verbatim (the bit-exact no-edit fast path).
fn write_header(
    w: &mut BitWriter,
    payload: &[u8],
    hi: &HeaderInfo,
    obj_start: usize,
    header: Option<&HeaderEdits>,
) {
    match header {
        // Header-string edit requested → rebuild the header, splicing the four (possibly new)
        // strings in while copying every surrounding fixed bit verbatim. Order of fields on disk:
        // …ts1 xuid1 [Name1] IsOnline1 ts2 xuid2 [Name2] IsOnline2 [Title] [Description]…
        Some(h) => {
            // The (Timestamp, Xuid) pair is the 128 bits right before each Name;
            // a stamp rewrites exactly those, anything else before the Name stays verbatim.
            let stamp = |w: &mut BitWriter, name_start: usize, v: Option<(u64, u64)>| match v {
                Some((ts, xuid)) => { w.bits(64, ts); w.bits(64, xuid); }
                None => w.copy_bits(payload, name_start - 128, 128),
            };
            // megalo-category-index: the 8 bits right before the CreatedBy block
            w.copy_bits(payload, 0, hi.category_pos);
            match h.category {
                Some(c) => w.bits(8, c as u8 as u64),
                None => w.copy_bits(payload, hi.category_pos, 8),
            }
            w.copy_bits(payload, hi.category_pos + 8, hi.name1_start - 128 - (hi.category_pos + 8));
            stamp(w, hi.name1_start, h.created);
            match &h.author {
                Some(s) => w.write_string_stop(s, 16),
                None => w.copy_bits(payload, hi.name1_start, hi.name1_end - hi.name1_start),
            }
            // IsOnline1 (the single bit between Name1 and ts2) — verbatim; then ts2 + xuid2.
            w.copy_bits(payload, hi.name1_end, hi.name2_start - 128 - hi.name1_end);
            stamp(w, hi.name2_start, h.modified);
            match &h.editor {
                Some(s) => w.write_string_stop(s, 16),
                None => w.copy_bits(payload, hi.name2_start, hi.name2_end - hi.name2_start),
            }
            // IsOnline2 — the single bit between Name2 and Title — verbatim.
            w.copy_bits(payload, hi.name2_end, hi.title_start - hi.name2_end);
            match &h.title {
                Some(s) => w.write_widechar_stop(s, 128),
                None => w.copy_bits(payload, hi.title_start, hi.desc_start - hi.title_start),
            }
            match &h.description {
                Some(s) => w.write_widechar_stop(s, 128),
                None => w.copy_bits(payload, hi.desc_start, hi.desc_end - hi.desc_start),
            }
            // Rest of the header up to the world bounds (HopperId / game-mode block / version /
            // checksum / palette crc / quota count / map id / flags) — verbatim; then
            // world-bounds (6 x f32) and maximum_budget spliced when edited,
            // spent_budget verbatim.
            w.copy_bits(payload, hi.desc_end, hi.bounds_pos - hi.desc_end);
            match h.bounds {
                Some(b) => { for x in b { w.f32_be(x); } }
                None => w.copy_bits(payload, hi.bounds_pos, 192),
            }
            match h.budget_max {
                Some(b) => w.bits(32, b as u64),
                None => w.copy_bits(payload, hi.bounds_pos + 192, 32),
            }
            w.copy_bits(payload, hi.bounds_pos + 224, hi.label_start - (hi.bounds_pos + 224));
            match &h.labels {
                // Re-encode the whole label table from the edited vec (always uncompressed), copying
                // the fixed bits on either side verbatim.
                Some(ls) => {
                    write_forge_labels(w, ls);
                    w.copy_bits(payload, hi.label_end, obj_start - hi.label_end);
                }
                // Label table untouched → copy the rest of the header (including the table) verbatim.
                None => {
                    w.copy_bits(payload, hi.label_start, obj_start - hi.label_start);
                }
            }
        }
        // No header edit → verbatim header (bit-exact round-trip fast path).
        None => {
            w.copy_bits(payload, 0, obj_start);
        }
    }
}

/// The placeable-object QUOTA table + trailing bits, written once for either encoder.
fn write_quota_and_tail(
    w: &mut BitWriter,
    r: &mut BitReader,
    payload: &[u8],
    hi: &HeaderInfo,
    placed: &[u32; 256],
    quota_edits: Option<&[(u8, u8)]>,
) -> usize {
    const TOTAL_BITS: usize = 229408;
    // The block does NOT end with the slot array: the placeable-object quota table follows it and
    // must be carried over (`into_padded` fills the rest of the fixed-size block with zeros).
    // The tail is copied from wherever the reader stopped to wherever the writer is now:
    // adds/deletes change the slot array's length, and the format is a sequential bitstream, so
    // the tail simply shifts.
    // The quota table is `num_quotas` entries of 24 bits = { u8 minimum_count, u8 maximum_count,
    // u8 placed_on_map }. It is POSITIONAL -- entry i describes palette folder index i (there is
    // no tag id in it), which is why folder indices resolve positionally against the scenario
    // palette. On every sample variant byte 2 equals the true number of objects of that folder
    // in the slot array. Rewrite bytes 1-2 from what was actually saved (raising the maximum when
    // the user placed more than the authored cap) and leave the minimum alone.
    for i in 0..hi.num_quotas as usize {
        if r.pos + 24 > TOTAL_BITS || w.pos + 24 > TOTAL_BITS {
            break;
        }
        let (mut b0, mut b1, _b2) = (r.bits(8), r.bits(8), r.bits(8));
        // edited minimum / maximum for this entry (placed is never taken from an edit)
        if let Some(&(mn, mx)) = quota_edits.and_then(|q| q.get(i)) {
            b0 = mn as u64;
            b1 = mx as u64;
        }
        let n = placed.get(i).copied().unwrap_or(0).min(255) as u64;
        w.bits(8, b0);
        w.bits(8, b1.max(n));
        w.bits(8, n);
    }
    // the engine's `length` numerator: the last bit the encoder wrote
    let end_bit = w.pos;
    // Anything after the quota table (trailing padding) is carried over verbatim rather than
    // zeroed.
    let src_rest = TOTAL_BITS.saturating_sub(r.pos);
    let dst_room = TOTAL_BITS.saturating_sub(w.pos);
    w.copy_bits(payload, r.pos, src_rest.min(dst_room));
    end_bit
}

/// The bounds / extents / axis bit widths the encoder WRITES with: the edited
/// world bounds when the header edit carries one, else the source's.
fn write_axes(hi: &HeaderInfo, header: Option<&HeaderEdits>) -> Result<([f32; 6], [f32; 3], [u32; 3]), String> {
    match header.and_then(|h| h.bounds) {
        Some(b) => {
            if (0..3).any(|i| !(b[2 * i + 1] > b[2 * i]) || !b[2 * i].is_finite() || !b[2 * i + 1].is_finite()) {
                return Err(format!("world bounds must have max > min on every axis: {b:?}"));
            }
            let ext = [b[1] - b[0], b[3] - b[2], b[5] - b[4]];
            Ok((b, ext, compute_axis_bits(21, ext)))
        }
        None => Ok((hi.bb, hi.ext, hi.axis)),
    }
}

/// Re-encode the variant payload with per-object `edits` applied. `edits` is keyed by the object's
/// ENUMERATION index (0-based over REAL objects in slot order — the same order `parse_payload`
/// returns). The header is copied verbatim (or spliced, see `write_header`); each of the 651
/// slots is read from the source and re-emitted (with edits / deletion); the block is zero-padded
/// back to its fixed 229408-bit size. Also returns how many of `adds` were actually written:
/// adds can only go into ABSENT slots, and a variant may hold slots that are
/// present-but-not-a-real-object; those are neither counted as objects nor available to fill, so
/// "651 - object count" overstates the free space and the caller must report the real number.
fn encode_payload_counted(
    payload: &[u8],
    edits: &std::collections::HashMap<usize, ObjEdit>,
    adds: &[PlacedObject],
    header: Option<&HeaderEdits>,
) -> (Vec<u8>, usize) {
    const TOTAL_BITS: usize = 229408;
    if payload.len() * 8 < TOTAL_BITS {
        return (payload.to_vec(), 0);
    }
    let mut r = BitReader::new(payload);
    let hi = parse_header(&mut r);
    // the SOURCE is read with its own box; the OUTPUT is quantised against the (possibly
    // edited) box
    let (rbb, rext, raxis) = (hi.bb, hi.ext, hi.axis);
    let Ok((bb, ext, axis)) = write_axes(&hi, header) else { return (payload.to_vec(), 0) };
    let obj_start = r.pos;
    let mut w = BitWriter::new();
    write_header(&mut w, payload, &hi, obj_start, header);
    let mut real_idx = 0usize;
    let mut adds_written = 0usize;
    let mut add_iter = adds.iter();
    // How many objects of each palette folder index end up in the saved slot array. The
    // quota table after the slots stores this per type ("placed on map"), and the engine's Forge
    // budget reads it -- so it must be recomputed, not carried over stale, whenever objects are
    // added or deleted.
    let mut placed = [0u32; 256];
    let mut tally = |o: &PlacedObject| {
        if o.folder != 0xFFFF && o.item != 0xFF && (o.folder as usize) < 256 {
            placed[o.folder as usize] += 1;
        }
    };
    for _ in 0..651 {
        let slot = read_slot(&mut r, raxis, rbb, rext);
        // ABSENT slot → fill it with a NEW object if any remain; else keep absent.
        if slot.is_none() {
            if let Some(a) = add_iter.next() {
                write_slot(&mut w, Some(a), axis, bb, ext);
                tally(a);
                adds_written += 1;
            } else {
                write_slot(&mut w, None, axis, bb, ext);
            }
            continue;
        }
        let is_real = slot.as_ref().map_or(false, |o| o.folder != 0xFFFF && o.item != 0xFF);
        if !is_real {
            write_slot(&mut w, slot.as_ref(), axis, bb, ext);
            continue;
        }
        let edit = edits.get(&real_idx);
        real_idx += 1;
        if edit.map_or(false, |e| e.deleted) {
            // Do NOT hand this freed slot to a pending add. It is free space, but filling it
            // REORDERS the real-slot enumeration: the app addresses objects by
            // `datum = 0xD0000000 + i` where `i` is the i-th REAL slot in order, so slotting a
            // new object ahead of retained originals shifts every later object's index, and the
            // next save writes each object's pos/rotation into a DIFFERENT object's slot (the map
            // comes back scattered and rotated wrong). Adds may only land in slots that were
            // ALREADY absent in the source (handled above), which keeps the enumeration intact.
            write_slot(&mut w, None, axis, bb, ext); // delete → absent slot
            continue;
        }
        let mut o = slot.unwrap();
        if let Some(e) = edit {
            if let Some(p) = e.pos {
                o.pos = p;
            }
            if let Some(t) = e.team {
                o.team = t;
            }
            if let Some(c) = e.color {
                o.color = c;
            }
            if let Some(pl) = e.placement {
                o.placement = pl;
            }
            if let Some(s) = e.spawn_seq {
                o.spawn_seq = s;
            }
            if let Some(r) = e.respawn {
                o.respawn = r;
            }
            if let Some((fwd, up)) = e.rot {
                let (g, uq, fq) = encode_orientation(fwd, up);
                o.up_is_global = g;
                o.up_quant = uq;
                o.forward_angle_q = fq;
                o.fwd = fwd;
                o.up = up;
            }
            if let Some(l) = e.label_idx {
                o.label_idx = l;
            }
            if let Some(sr) = e.spawn_rel {
                o.spawn_rel = sr;
            }
        }
        tally(&o);
        write_slot(&mut w, Some(&o), axis, bb, ext);
    }
    write_quota_and_tail(&mut w, &mut r, payload, &hi, &placed, header.and_then(|h| h.quotas.as_deref()));
    (w.into_padded(TOTAL_BITS), adds_written)
}

/// The stored quota table (min, max, placed_on_map) per folder index, for the diagnostic tests
/// outside this module.
#[cfg(test)]
pub fn quota_table(payload: &[u8]) -> Vec<(u8, u8, u8)> {
    const TOTAL_BITS: usize = 229408;
    let mut r = BitReader::new(payload);
    let hi = parse_header(&mut r);
    for _ in 0..651 {
        let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext);
    }
    let mut out = Vec::new();
    for _ in 0..hi.num_quotas as usize {
        if r.pos + 24 > TOTAL_BITS { break; }
        out.push((r.bits(8) as u8, r.bits(8) as u8, r.bits(8) as u8));
    }
    out
}

/// Bit just past the quota table of a source payload = the engine's `length` numerator
/// (`length = ceil(bits / 8)`, equal to the stored chunk length on every shipped Reach file).
pub fn payload_end_bit(payload: &[u8]) -> usize {
    let mut r = BitReader::new(payload);
    let hi = parse_header(&mut r);
    for _ in 0..651 {
        let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext);
    }
    r.pos + 24 * hi.num_quotas as usize
}

/// Write the variant's object list WHOLESALE — the editor's save. The editor knows the full list
/// of forge objects, so a save is "write this list into slots 0..N, leave the rest absent": order
/// in == order out, the datum enumeration (`0xD0000000 + i`, the i-th real slot) can never drift,
/// and saving twice is the same as saving once. A positional diff against the source (an edit
/// map keyed by slot plus an adds queue, as `encode_payload_counted` still does for the
/// `--mvar-delete` tool) can shift that enumeration and write objects into each other's slots.
///
/// Lossless: `PlacedObject` round-trips a slot bit-for-bit (see
/// `full_rewrite_of_untouched_map_is_bit_identical`), so objects carried over from the source keep
/// every field, including ones HMS does not model or display.
///
/// Returns `(bytes, written, end_bit)`: `written` is `objects.len()` capped at the 651 slots,
/// `end_bit` the bit just past the quota table (the chunk `length` numerator). Errors only on an
/// invalid edited world box.
fn encode_payload_objects_measured(
    payload: &[u8],
    objects: &[PlacedObject],
    header: Option<&HeaderEdits>,
) -> Result<(Vec<u8>, usize, usize), String> {
    const TOTAL_BITS: usize = 229408;
    if payload.len() * 8 < TOTAL_BITS {
        return Ok((payload.to_vec(), 0, 0));
    }
    let mut r = BitReader::new(payload);
    let hi = parse_header(&mut r);
    let (rbb, rext, raxis) = (hi.bb, hi.ext, hi.axis);
    let (bb, ext, axis) = write_axes(&hi, header)?;
    let obj_start = r.pos;
    let mut w = BitWriter::new();
    write_header(&mut w, payload, &hi, obj_start, header);

    let mut placed = [0u32; 256];
    let mut written = 0usize;
    for i in 0..651 {
        // advance the READER through the source slot array so the tail lands in the right place
        let _ = read_slot(&mut r, raxis, rbb, rext);
        match objects.get(i) {
            Some(o) => {
                write_slot(&mut w, Some(o), axis, bb, ext);
                if o.folder != 0xFFFF && o.item != 0xFF && (o.folder as usize) < 256 {
                    placed[o.folder as usize] += 1;
                }
                written += 1;
            }
            None => write_slot(&mut w, None, axis, bb, ext),
        }
    }
    let end_bit = write_quota_and_tail(&mut w, &mut r, payload, &hi, &placed, header.and_then(|h| h.quotas.as_deref()));
    Ok((w.into_padded(TOTAL_BITS), written, end_bit))
}

/// Rebuild a whole Reach `.mvar` from `src_data` with a new object list: every chunk but `mvar`
/// is copied verbatim (the `chdr` mirror gets the edited strings / stamps, as the game's own
/// writer keeps it in step with the bitstream), the `mvar` chunk gets the new payload with
/// `length = ceil(bits / 8)` and its SHA-1 recomputed (the engine never enforces them, but a
/// game-written file always has them right). Returns the file bytes and the number of objects
/// written.
pub fn rebuild_file(src_data: &[u8], objects: &[PlacedObject], header: Option<&HeaderEdits>) -> Result<(Vec<u8>, usize), String> {
    use crate::h4::mvar::{chdr_put_stamp, chdr_put_str, chdr_put_wstr, CHDR_AUTHOR, CHDR_CREATED, CHDR_DESCRIPTION, CHDR_EDITOR, CHDR_MODIFIED, CHDR_TITLE};
    let data = src_data;
    let mut written = 0usize;
    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let magic = &data[pos..pos + 4];
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 8 || pos + size > data.len() {
            break;
        }
        if magic == b"mvar" && size >= 36 {
            let payload = &data[pos + 36..pos + size];
            let (re, n, end_bit) = encode_payload_objects_measured(payload, objects, header)?;
            written = n;
            let new_size = 36 + re.len();
            out.extend_from_slice(&data[pos..pos + 12]);
            let hdr_start = out.len() - 12;
            out[hdr_start + 4..hdr_start + 8].copy_from_slice(&(new_size as u32).to_be_bytes());
            let length = if end_bit > 0 && re.len() * 8 >= 229408 { ((end_bit + 7) / 8) as u32 } else { u32::from_be_bytes([data[pos + 32], data[pos + 33], data[pos + 34], data[pos + 35]]) };
            out.extend_from_slice(&reach_chunk_hash(&re, length));
            out.extend_from_slice(&length.to_be_bytes());
            out.extend_from_slice(&re);
        } else if magic == b"chdr" && size >= 12 + CHDR_DESCRIPTION + 256 {
            let mut body = data[pos + 12..pos + size].to_vec();
            if let Some(h) = header {
                if let Some(s) = &h.author { chdr_put_str(&mut body, CHDR_AUTHOR, s, 16); }
                if let Some(s) = &h.editor { chdr_put_str(&mut body, CHDR_EDITOR, s, 16); }
                if let Some((ts, xuid)) = h.created { chdr_put_stamp(&mut body, CHDR_CREATED, ts, xuid); }
                if let Some((ts, xuid)) = h.modified { chdr_put_stamp(&mut body, CHDR_MODIFIED, ts, xuid); }
                if let Some(s) = &h.title { chdr_put_wstr(&mut body, CHDR_TITLE, s, 128); }
                if let Some(s) = &h.description { chdr_put_wstr(&mut body, CHDR_DESCRIPTION, s, 128); }
                if let Some(c) = h.category { body[0x34] = c as u8; }
            }
            out.extend_from_slice(&data[pos..pos + 12]);
            out.extend_from_slice(&body);
        } else {
            out.extend_from_slice(&data[pos..pos + size]);
        }
        pos += size;
    }
    out.extend_from_slice(&data[pos..]);
    Ok((out, written))
}

/// Write the whole object list to `dst`, preserving every other BLF chunk of `src`.
pub fn save_objects(
    src: &std::path::Path,
    dst: &std::path::Path,
    objects: &[PlacedObject],
    header: Option<&HeaderEdits>,
) -> std::io::Result<usize> {
    let data = read_expanded(src)?;
    let (out, written) = rebuild_file(&data, objects, header).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    std::fs::write(dst, out)?;
    Ok(written)
}

/// Read a .mvar, apply `edits` to its objects, and write the result to `dst` (preserving every
/// other BLF chunk). `edits` keyed by object enumeration index (see `encode_payload_counted`).
pub fn save_with_edits(
    src: &std::path::Path,
    dst: &std::path::Path,
    edits: &std::collections::HashMap<usize, ObjEdit>,
    adds: &[PlacedObject],
    header: Option<&HeaderEdits>,
) -> std::io::Result<usize> {
    // How many of `adds` actually found a free slot (see encode_payload_counted).
    let mut adds_written = 0usize;
    let data = read_expanded(src)?;
    let mut out = Vec::with_capacity(data.len());
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let magic = &data[pos..pos + 4];
        let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        if size < 8 || pos + size > data.len() {
            break;
        }
        if magic == b"mvar" && size >= 36 {
            let payload = &data[pos + 36..pos + size];
            let (re, n_adds) = encode_payload_counted(payload, edits, adds, header);
            adds_written = n_adds;
            let new_size = 36 + re.len();
            out.extend_from_slice(&data[pos..pos + 12]);
            let hdr_start = out.len() - 12;
            out[hdr_start + 4..hdr_start + 8].copy_from_slice(&(new_size as u32).to_be_bytes());
            // chunk length + SHA-1 recomputed for the new payload
            let length = if re.len() * 8 >= 229408 { ((payload_end_bit(&re) + 7) / 8) as u32 } else { u32::from_be_bytes([data[pos + 32], data[pos + 33], data[pos + 34], data[pos + 35]]) };
            out.extend_from_slice(&reach_chunk_hash(&re, length));
            out.extend_from_slice(&length.to_be_bytes());
            out.extend_from_slice(&re);
        } else {
            out.extend_from_slice(&data[pos..pos + size]);
        }
        pos += size;
    }
    std::fs::write(dst, out)?;
    Ok(adds_written)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_payload_counted` without the adds count.
    fn encode_payload(payload: &[u8], edits: &std::collections::HashMap<usize, ObjEdit>, adds: &[PlacedObject], header: Option<&HeaderEdits>) -> Vec<u8> {
        encode_payload_counted(payload, edits, adds, header).0
    }

    /// `encode_payload_objects_measured` without the end bit (an invalid box returns the source).
    fn encode_payload_objects(payload: &[u8], objects: &[PlacedObject], header: Option<&HeaderEdits>) -> (Vec<u8>, usize) {
        encode_payload_objects_measured(payload, objects, header).map(|(p, n, _)| (p, n)).unwrap_or_else(|_| (payload.to_vec(), 0))
    }

    /// How many NEW objects would fit: the number of ABSENT slots (651 minus the present slots).
    fn absent_slot_count(payload: &[u8]) -> usize {
        let mut r = BitReader::new(payload);
        let hi = parse_header(&mut r);
        (0..651).filter(|_| read_slot(&mut r, hi.axis, hi.bb, hi.ext).is_none()).count()
    }

    // Empire.mvar ships its `mvar` chunk zlib-compressed inside a `_cmp` chunk.
    #[test]
    fn compressed_cmp_chunk_expands_to_a_readable_variant() {
        let p = std::path::Path::new(
            "/mnt/games/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach/map_variants/Empire.mvar",
        );
        let Ok(raw) = std::fs::read(p) else { return };
        assert!(raw.windows(4).any(|w| w == b"_cmp"));
        let ex = expand_blf_cmp(&raw);
        assert!(ex.windows(4).any(|w| w == b"mvar") && !ex.windows(4).any(|w| w == b"_cmp"));
        // _eof re-stamped to the byte offset of the _eof chunk.
        let eof = ex.windows(4).position(|w| w == b"_eof").unwrap();
        let before = u32::from_be_bytes([ex[eof + 12], ex[eof + 13], ex[eof + 14], ex[eof + 15]]) as usize;
        assert_eq!(before, eof);
        let v = parse_variant(p).expect("compressed variant parses");
        assert!(v.objects.len() > 10, "objects {}", v.objects.len());
        assert!(!v.title.is_empty());
        // Expanding twice is a no-op; saving from the compressed source works and yields plain chunks.
        assert_eq!(expand_blf_cmp(&ex), ex);
        let dir = std::env::temp_dir().join("hms_mvar_cmp_test");
        std::fs::create_dir_all(&dir).unwrap();
        let dst = dir.join("empire_plain.mvar");
        save_objects(p, &dst, &v.objects, None).unwrap();
        let v2 = parse_variant(&dst).unwrap();
        assert_eq!(v2.objects.len(), v.objects.len());
        assert!(!std::fs::read(&dst).unwrap().windows(4).any(|w| w == b"_cmp"));
    }

    #[test]
    fn bit_roundtrip() {
        // Write a mix of widths/endianness, read them back identically.
        let mut w = BitWriter::new();
        w.bits(4, 0xB);
        w.flag(true);
        for b in 0xDEADBEEFu32.to_le_bytes() { w.bits(8, b as u64); }
        w.f32_be(1.5);
        w.bits(14, 0x1234);
        let bytes = w.into_padded(0);

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.bits(4), 0xB);
        assert!(r.flag());
        assert_eq!(r.u32_le(), 0xDEADBEEF);
        assert_eq!(r.f32_be(), 1.5);
        assert_eq!(r.bits(14), 0x1234);
    }

    #[test]
    fn copy_bits_is_identity() {
        let src = [0b1010_1100u8, 0b0011_0101, 0xFF, 0x00];
        let mut w = BitWriter::new();
        w.copy_bits(&src, 0, 32);
        assert_eq!(w.into_padded(32), src);
    }

    // Real .mvar files to round-trip if present on this machine (skipped in CI/other machines).
    pub(super) fn sample_mvars_probe() -> Vec<std::path::PathBuf> { sample_mvars() }

    fn sample_mvars() -> Vec<std::path::PathBuf> {
        // These gates only mean anything if they actually FIND files: sweep the real variant
        // folders (both the Windows drive as mounted here and the native paths) and take whatever
        // exists; `samples_are_found` fails when the set is empty.
        let mut found: Vec<std::path::PathBuf> = Vec::new();
        // Extra personal sample folders come from HMS_MVAR_SAMPLE_DIRS (";"-separated), never from
        // hard-coded user paths.
        let extra: Vec<String> = std::env::var("HMS_MVAR_SAMPLE_DIRS").map(|v| v.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default();
        let mut dirs: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();
        dirs.extend([
            // The GAME's own saved variants -- the user's real maps. These are exactly the files a
            // save must never damage, so they belong in the gate.
            "/mnt/win/Program Files (x86)/Steam/steamapps/common/Halo The Master Chief Collection/haloreach/map_variants",
            "/mnt/win/Program Files (x86)/Steam/steamapps/common/Halo The Master Chief Collection/haloreach/hopper_map_variants",
            // The install on this box (the user's own maps -- the ones a save bug damages).
            "/mnt/games/haloreach/map_variants",
            "/mnt/games/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach/map_variants",
            "/mnt/games/SteamLibrary/steamapps/common/Halo The Master Chief Collection/haloreach/hopper_map_variants",
            "/mnt/games/XboxGames/Halo- The Master Chief Collection/Content/haloreach/map_variants",
        ]);
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(d) else { continue };
            for e in rd.flatten() {
                let path = e.path();
                if path.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) {
                    found.push(path);
                }
            }
        }
        found.sort();
        // Keep the sweep bounded so the suite stays quick, but cover a spread of files.
        found.truncate(24);
        if !found.is_empty() {
            return found;
        }
        let cands = [
            r"target\release\batch_in2\asylum.mvar",
            r"target\release\batch_in2\thecage.mvar",
            r"C:\Program Files (x86)\Steam\steamapps\common\Halo The Master Chief Collection\haloreach\hopper_map_variants\forge_halo_hemorrhage.mvar",
        ];
        cands.iter().map(std::path::PathBuf::from).filter(|p| p.exists()).collect()
    }

    /// A NEW object added via `new_placed_object` must encode with the spawn-critical
    /// fields the game requires (flags=1, placement bits 2,3) so it actually spawns in-game.
    #[test]
    fn new_object_has_spawn_fields() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            // Place inside the variant's bounds by reusing an existing object's position.
            let pos = before.first().map(|o| o.pos).unwrap_or([0.0, 0.0, 0.0]);
            let add = new_placed_object(1, 0, pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0xFF, -1);
            let re = encode_payload(&payload, &std::collections::HashMap::new(), &[add], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(after.len(), before.len() + 1, "{:?}: add not written", p);
            let obj = after.iter().find(|o| o.folder == 1 && o.item == 0)
                .expect("added object present");
            assert_eq!(obj.flags, 1, "{:?}: new object flags must be 1 (spawn)", p);
            assert_eq!(obj.placement & 0b1100, 0b1100, "{:?}: new object placement must have spawn bits 2,3", p);
        }
    }


    /// Diagnostic: list the sample variants with their canvas map id, title and object count.
    /// `cargo test -p hms-app list_samples -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn list_samples() {
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { eprintln!("{:?}: not an mvar", p.file_name()); continue };
            let (map_id, quotas, _labels, objs) = parse_payload(&payload);
            let v = parse_variant(&p);
            eprintln!("{:60} map_id={:<6} quotas={:<4} objs={:<4} title={:?}",
                p.file_name().unwrap().to_string_lossy(), map_id, quotas, objs.len(),
                v.as_ref().map(|v| v.title.clone()).unwrap_or_default());
        }
    }


    /// A no-edit encode must preserve EVERYTHING after the slot array (the quota table).
    /// `encode_payload` ends with `into_padded`, which fills the rest of the fixed block with
    /// ZEROS -- so if this fails, every save wipes that region.
    #[test]
    fn encode_preserves_the_tail_after_slots() {
        let no_edits = std::collections::HashMap::new();
        let mut checked = 0;
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            if payload.len() * 8 < 229408 { continue }
            let mut r = BitReader::new(&payload);
            let hi = parse_header(&mut r);
            for _ in 0..651 { let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext); }
            let end = r.pos;
            let src_set = (end..229408).filter(|&b| b >> 3 < payload.len() && payload[b >> 3] >> (7 - (b & 7)) & 1 == 1).count();
            if src_set == 0 { continue } // nothing after the slots to lose
            checked += 1;
            let re = encode_payload(&payload, &no_edits, &[], None);
            let out_set = (end..229408).filter(|&b| b >> 3 < re.len() && re[b >> 3] >> (7 - (b & 7)) & 1 == 1).count();
            assert_eq!(
                src_set, out_set,
                "{:?}: {src_set} set bits after the slot array (bit {end}) but {out_set} survived the encode -- the quota table is being wiped",
                p.file_name().unwrap()
            );
        }
        assert!(checked > 0, "no sample had data after the slots; the gate proved nothing");
    }

    /// After an ADD the slot array grows, so the quota table and everything after it
    /// SHIFT. The quota table is deliberately rewritten (see quota_placed_counts_track_the_slots),
    /// so what must be preserved bit-for-bit is the region AFTER it -- and the quota region itself
    /// must never come back zeroed.
    #[test]
    fn tail_survives_adds() {
        let slot_end = |pl: &[u8]| -> (usize, u32) {
            let mut r = BitReader::new(pl);
            let hi = parse_header(&mut r);
            for _ in 0..651 { let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext); }
            (r.pos, hi.num_quotas)
        };
        let bit = |b: &[u8], i: usize| -> u8 {
            if i >> 3 >= b.len() { 0 } else { b[i >> 3] >> (7 - (i & 7)) & 1 }
        };
        let mut checked = 0;
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            if payload.len() * 8 < 229408 { continue }
            let (_, _, _, before) = parse_payload(&payload);
            let Some(first) = before.first().cloned() else { continue };
            let (src_end, nq) = slot_end(&payload);
            let qlen = nq as usize * 24;
            let src_set = (src_end..229408).filter(|&b| bit(&payload, b) == 1).count();
            if src_set == 0 { continue }
            let add = new_placed_object(first.folder, first.item, first.pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0xFF, -1);
            let (re, written) = encode_payload_counted(&payload, &std::collections::HashMap::new(), &[add], None);
            if written == 0 { continue } // variant full
            checked += 1;
            let (out_end, nq2) = slot_end(&re);
            assert_eq!(nq, nq2, "{:?}: quota count changed", p.file_name());
            // The quota table must still hold data.
            let q_set = (out_end..out_end + qlen).filter(|&b| bit(&re, b) == 1).count();
            assert!(q_set > 0, "{:?}: quota table came back all zeros", p.file_name());
            // Everything past the quota table must match the source's corresponding region.
            let n = (229408 - out_end.max(src_end) - qlen).min(4096);
            for k in 0..n {
                assert_eq!(
                    bit(&payload, src_end + qlen + k), bit(&re, out_end + qlen + k),
                    "{:?}: bit {k} past the quota table changed across an add", p.file_name()
                );
            }
        }
        assert!(checked > 0, "gate proved nothing");
    }

    /// The quota table's "placed on map" byte must always equal the number of objects of
    /// that folder actually in the saved slot array -- on a plain re-save, after an ADD, and after
    /// a DELETE. A stale count is what the engine's Forge budget reads.
    #[test]
    fn quota_placed_counts_track_the_slots() {
        let read_quotas = |pl: &[u8]| -> Vec<(u32, u32, u32)> {
            let mut r = BitReader::new(pl);
            let hi = parse_header(&mut r);
            for _ in 0..651 { let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext); }
            (0..hi.num_quotas as usize).map(|_| (r.bits(8) as u32, r.bits(8) as u32, r.bits(8) as u32)).collect()
        };
        let truth = |pl: &[u8]| -> std::collections::BTreeMap<u16, u32> {
            let (_, _, _, objs) = parse_payload(pl);
            let mut m: std::collections::BTreeMap<u16, u32> = Default::default();
            for o in &objs { *m.entry(o.folder).or_default() += 1; }
            m
        };
        let mut checked = 0;
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            if payload.len() * 8 < 229408 { continue }
            let (_, _, _, before) = parse_payload(&payload);
            let Some(first) = before.first().cloned() else { continue };
            checked += 1;

            // (a) plain re-save
            let re = encode_payload(&payload, &std::collections::HashMap::new(), &[], None);
            let (q, t) = (read_quotas(&re), truth(&re));
            for (i, e) in q.iter().enumerate() {
                assert_eq!(e.2, *t.get(&(i as u16)).unwrap_or(&0),
                    "{:?}: quota[{i}] placed={} but the slots hold {}", p.file_name(), e.2, t.get(&(i as u16)).unwrap_or(&0));
            }

            // (b) after an ADD of the first object's type
            let add = new_placed_object(first.folder, first.item, first.pos, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0xFF, -1);
            let (re2, written) = encode_payload_counted(&payload, &std::collections::HashMap::new(), &[add], None);
            if written == 1 {
                let (q2, t2) = (read_quotas(&re2), truth(&re2));
                for (i, e) in q2.iter().enumerate() {
                    assert_eq!(e.2, *t2.get(&(i as u16)).unwrap_or(&0),
                        "{:?}: after add, quota[{i}] placed={} but slots hold {}", p.file_name(), e.2, t2.get(&(i as u16)).unwrap_or(&0));
                }
                let f = first.folder as usize;
                if f < q.len() {
                    assert_eq!(q2[f].2, q[f].2 + 1, "{:?}: adding one object of folder {f} must bump its placed count", p.file_name());
                    assert!(q2[f].1 >= q2[f].2, "{:?}: maximum must not sit below placed", p.file_name());
                }
            }

            // (c) after a DELETE of object 0
            let mut edits = std::collections::HashMap::new();
            edits.insert(0usize, ObjEdit { deleted: true, ..Default::default() });
            let re3 = encode_payload(&payload, &edits, &[], None);
            let (q3, t3) = (read_quotas(&re3), truth(&re3));
            for (i, e) in q3.iter().enumerate() {
                assert_eq!(e.2, *t3.get(&(i as u16)).unwrap_or(&0),
                    "{:?}: after delete, quota[{i}] placed={} but slots hold {}", p.file_name(), e.2, t3.get(&(i as u16)).unwrap_or(&0));
            }
        }
        assert!(checked > 0, "gate proved nothing");
    }


    /// THE fundamental gate: a save with NO edits must reproduce the payload BIT-FOR-BIT. Comparing
    /// parsed objects only proves our writer agrees with our reader; it says nothing about whether
    /// the GAME agrees. Any bit we change that we do not understand can desynchronise the engine's
    /// sequential read and scatter every object after it.
    #[test]
    fn noedit_save_is_bit_identical() {
        let no_edits = std::collections::HashMap::new();
        let mut checked = 0;
        let mut bad: Vec<String> = Vec::new();
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            if payload.len() * 8 < 229408 { continue }
            checked += 1;
            let re = encode_payload(&payload, &no_edits, &[], None);
            if re.len() != payload.len() {
                bad.push(format!("{:?}: length {} -> {}", p.file_name().unwrap(), payload.len(), re.len()));
                continue;
            }
            let diffs: Vec<usize> = (0..229408)
                .filter(|&b| {
                    let (i, m) = (b >> 3, 1u8 << (7 - (b & 7)));
                    (payload[i] & m) != (re[i] & m)
                })
                .collect();
            if !diffs.is_empty() {
                bad.push(format!("{:?}: {} bits differ, first at {} (of 229408)",
                    p.file_name().unwrap(), diffs.len(), diffs[0]));
            }
        }
        assert!(checked > 0, "gate proved nothing");
        assert!(bad.is_empty(), "no-edit save is NOT bit-identical:\n  {}", bad.join("\n  "));
    }


    /// An ObjEdit for EVERY object carrying its live pos + orientation must, on an untouched map,
    /// reproduce the source BIT-FOR-BIT. Anything that changes is a field re-encoded differently
    /// from how the game wrote it -- which the game reads back as a moved or re-oriented object.
    #[test]
    fn full_rewrite_of_untouched_map_is_bit_identical() {
        let mut report: Vec<String> = Vec::new();
        let mut checked = 0;
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            if payload.len() * 8 < 229408 { continue }
            let (_, _, _, before) = parse_payload(&payload);
            if before.is_empty() { continue }
            checked += 1;
            let mut edits = std::collections::HashMap::new();
            for (i, o) in before.iter().enumerate() {
                edits.insert(i, ObjEdit {
                    pos: Some(o.pos), team: Some(o.team), color: Some(o.color),
                    placement: Some(o.placement), spawn_seq: Some(o.spawn_seq),
                    respawn: Some(o.respawn), rot: Some((o.fwd, o.up)),
                    label_idx: Some(o.label_idx), spawn_rel: Some(o.spawn_rel), deleted: false,
                });
            }
            let re = encode_payload(&payload, &edits, &[], None);
            let diffs = (0..229408).filter(|&b| {
                let (i, m) = (b >> 3, 1u8 << (7 - (b & 7)));
                (payload[i] & m) != (re[i] & m)
            }).count();
            if diffs > 0 {
                // How far did objects actually move / rotate?
                let (_, _, _, after) = parse_payload(&re);
                let mut maxd = 0.0f32;
                let mut rot = 0;
                for (a, b) in before.iter().zip(after.iter()) {
                    for k in 0..3 { maxd = maxd.max((a.pos[k] - b.pos[k]).abs()); }
                    if a.up_quant != b.up_quant || a.forward_angle_q != b.forward_angle_q { rot += 1; }
                }
                report.push(format!("{:?}: {diffs} bits differ, max position delta {maxd:.4} wu, {rot} objects re-oriented",
                    p.file_name().unwrap()));
            }
        }
        assert!(checked > 0);
        assert!(report.is_empty(), "full rewrite is not bit-identical:\n  {}", report.join("\n  "));
    }


    /// A variant whose quota table is zeroed (a file damaged by another writer) must be REPAIRED
    /// simply by loading and saving it again -- the counts are recoverable from the slot array.
    #[test]
    fn saving_repairs_a_zeroed_quota_table() {
        let mut checked = 0;
        for p in sample_mvars() {
            let Ok(data) = std::fs::read(&p) else { continue };
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            if payload.len() * 8 < 229408 { continue }
            let (_, _, _, objs) = parse_payload(&payload);
            if objs.is_empty() { continue }
            // Reproduce the damage: zero everything after the slot array.
            let mut r = BitReader::new(&payload);
            let hi = parse_header(&mut r);
            for _ in 0..651 { let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext); }
            let end = r.pos;
            let mut damaged = payload.clone();
            for b in end..229408 {
                if b >> 3 < damaged.len() { damaged[b >> 3] &= !(1u8 << (7 - (b & 7))); }
            }
            let quotas_of = |pl: &[u8]| -> Vec<u32> {
                let mut r = BitReader::new(pl);
                let hi = parse_header(&mut r);
                for _ in 0..651 { let _ = read_slot(&mut r, hi.axis, hi.bb, hi.ext); }
                (0..hi.num_quotas).map(|_| { let _ = r.bits(8); let _ = r.bits(8); r.bits(8) as u32 }).collect()
            };
            assert!(quotas_of(&damaged).iter().all(|&v| v == 0), "damage setup failed");
            checked += 1;
            // Load + save with the current encoder.
            let repaired = encode_payload(&damaged, &std::collections::HashMap::new(), &[], None);
            let got = quotas_of(&repaired);
            let want = quotas_of(&payload);
            assert_eq!(got, want, "{:?}: re-saving did not restore the quota placed counts", p.file_name());
        }
        assert!(checked > 0, "gate proved nothing");
    }


    /// A full edit for every object from its own live values (an unmoved, unrotated map) must
    /// keep every object in its slot with its exact quantised fields.
    #[test]
    fn full_rewrite_is_a_fixed_point() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            let (_, _, _, before) = parse_payload(&payload);
            if before.is_empty() {
                continue;
            }
            // A full edit for every object, from its own live values.
            let mut edits = std::collections::HashMap::new();
            for (i, o) in before.iter().enumerate() {
                edits.insert(i, ObjEdit {
                    pos: Some(o.pos),
                    team: Some(o.team),
                    color: Some(o.color),
                    placement: Some(o.placement),
                    spawn_seq: Some(o.spawn_seq),
                    respawn: Some(o.respawn),
                    rot: Some((o.fwd, o.up)),
                    label_idx: Some(o.label_idx),
                    spawn_rel: Some(o.spawn_rel),
                    deleted: false,
                });
            }
            let re = encode_payload(&payload, &edits, &[], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(before.len(), after.len(), "{:?}: object count changed", p);
            for (i, (a, b)) in before.iter().zip(after.iter()).enumerate() {
                assert_eq!(a.folder, b.folder, "{:?} obj{i}: folder", p);
                assert_eq!(a.item, b.item, "{:?} obj{i}: item", p);
                assert_eq!(a.slot, b.slot, "{:?} obj{i}: SLOT MOVED", p);
                for k in 0..3 {
                    assert!((a.pos[k] - b.pos[k]).abs() < 1e-3,
                        "{:?} obj{i}: position drifted on save: {:?} -> {:?}", p, a.pos, b.pos);
                }
                assert_eq!(a.up_is_global, b.up_is_global, "{:?} obj{i}: up_is_global flipped (fwd={:?} up={:?})", p, a.fwd, a.up);
                assert_eq!(a.up_quant, b.up_quant, "{:?} obj{i}: UP QUANT drifted {} -> {} (up={:?})", p, a.up_quant, b.up_quant, a.up);
                assert_eq!(a.forward_angle_q, b.forward_angle_q, "{:?} obj{i}: YAW drifted {} -> {} (fwd={:?})", p, a.forward_angle_q, b.forward_angle_q, a.fwd);
                assert_eq!(a.cached_type, b.cached_type, "{:?} obj{i}: cached_type", p);
                assert_eq!(a.spawn_rel, b.spawn_rel, "{:?} obj{i}: parent slot ref", p);
                assert_eq!(a.boundary_shape, b.boundary_shape, "{:?} obj{i}: boundary shape", p);
            }
        }
    }

    /// Saving twice must be identical to saving once. A map is "extended" by
    /// repeated load>edit>save cycles, so any per-save drift compounds.
    #[test]
    fn repeated_saves_do_not_drift() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let Some(payload) = blf_mvar_payload(&data) else { continue };
            let full = |pl: &[u8]| -> Vec<u8> {
                let (_, _, _, objs) = parse_payload(pl);
                let mut edits = std::collections::HashMap::new();
                for (i, o) in objs.iter().enumerate() {
                    edits.insert(i, ObjEdit {
                        pos: Some(o.pos), team: Some(o.team), color: Some(o.color),
                        placement: Some(o.placement), spawn_seq: Some(o.spawn_seq),
                        respawn: Some(o.respawn), rot: Some((o.fwd, o.up)),
                        label_idx: Some(o.label_idx), spawn_rel: Some(o.spawn_rel), deleted: false,
                    });
                }
                encode_payload(pl, &edits, &[], None)
            };
            let one = full(&payload);
            let two = full(&one);
            let (_, _, _, a) = parse_payload(&one);
            let (_, _, _, b) = parse_payload(&two);
            assert_eq!(a.len(), b.len(), "{:?}: count changed on 2nd save", p);
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(x.up_quant, y.up_quant, "{:?} obj{i}: rotation drifts every  save", p);
                for k in 0..3 {
                    assert!((x.pos[k] - y.pos[k]).abs() < 1e-4, "{:?} obj{i}: position drifts every save", p);
                }
            }
        }
    }

    #[test]
    fn encode_noedit_preserves_objects() {
        let no_edits = std::collections::HashMap::new();
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).expect("mvar chunk");
            let (_, _, _, before) = parse_payload(&payload);
            let re = encode_payload(&payload, &no_edits, &[], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(before.len(), after.len(), "{:?}: object count changed", p);
            for (a, b) in before.iter().zip(after.iter()) {
                assert_eq!(a.folder, b.folder, "{:?}", p);
                assert_eq!(a.item, b.item, "{:?}", p);
                assert_eq!(a.team, b.team, "{:?}", p);
                assert_eq!(a.color, b.color, "{:?}", p);
                assert_eq!(a.up_quant, b.up_quant, "{:?}", p);
                assert_eq!(a.forward_angle_q, b.forward_angle_q, "{:?}", p);
                assert_eq!(a.cached_type, b.cached_type, "{:?}", p);
                assert_eq!(a.label_idx, b.label_idx, "{:?}", p);
                assert_eq!(a.placement, b.placement, "{:?}", p);
                for k in 0..3 {
                    assert!((a.pos[k] - b.pos[k]).abs() < 0.05, "{:?}: pos drift {} vs {}", p, a.pos[k], b.pos[k]);
                }
            }
            // The payload must also stay the fixed block size.
            assert_eq!(re.len(), payload.len(), "{:?}: payload size changed", p);
        }
    }

    /// A position edit on object 0 must round-trip to (approximately) the requested position, and
    /// leave all OTHER objects untouched.
    #[test]
    fn encode_position_edit_applies() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            if before.is_empty() {
                continue;
            }
            let target = [before[0].pos[0] + 5.0, before[0].pos[1] - 3.0, before[0].pos[2] + 1.0];
            let mut edits = std::collections::HashMap::new();
            edits.insert(0usize, ObjEdit { pos: Some(target), ..Default::default() });
            let re = encode_payload(&payload, &edits, &[], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(before.len(), after.len());
            for k in 0..3 {
                assert!((after[0].pos[k] - target[k]).abs() < 0.05, "{:?}: edited pos wrong", p);
            }
            for i in 1..before.len() {
                for k in 0..3 {
                    assert!((after[i].pos[k] - before[i].pos[k]).abs() < 0.05, "{:?}: obj {} moved unexpectedly", p, i);
                }
            }
        }
    }

    /// A placement/spawn_seq/respawn edit on object 0 must round-trip EXACTLY through
    /// encode→parse, and leave every OTHER object's placement/spawn_seq/respawn untouched.
    #[test]
    fn encode_placement_edit_applies() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            if before.is_empty() {
                continue;
            }
            // Build a distinctly-different placement byte: Physics=Fixed + placed-at-start=false,
            // symmetry=Both, from scratch — guaranteed to differ from the retail value (usually 12).
            let mut target_place = PLACEMENT_DEFAULT_SPAWNABLE; // both symmetries, placed, normal
            // set Physics=Fixed (bits 6-7 = 1)
            target_place = (target_place & !PLACE_PHYSICS_MASK) | (1 << 6);
            // set NOT placed at start
            target_place |= PLACE_NOT_AT_START;
            let target_seq = -37i32;
            let target_respawn = 25u8;
            assert_ne!(target_place, before[0].placement, "{:?}: pick a different placement", p);

            let mut edits = std::collections::HashMap::new();
            edits.insert(0usize, ObjEdit {
                placement: Some(target_place),
                spawn_seq: Some(target_seq),
                respawn: Some(target_respawn),
                ..Default::default()
            });
            let re = encode_payload(&payload, &edits, &[], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(before.len(), after.len(), "{:?}: object count changed", p);

            // Edited object carries the new values EXACTLY.
            assert_eq!(after[0].placement, target_place, "{:?}: placement not applied", p);
            assert_eq!(after[0].spawn_seq, target_seq, "{:?}: spawn_seq not applied", p);
            assert_eq!(after[0].respawn, target_respawn, "{:?}: respawn not applied", p);
            assert_eq!(after[0].placement & PLACE_PHYSICS_MASK, 1 << 6, "{:?}: physics wrong", p);
            assert_ne!(after[0].placement & PLACE_NOT_AT_START, 0, "{:?}: placed_at_start should be false", p);

            // Every OTHER object's placement/spawn_seq/respawn is untouched.
            for i in 1..before.len() {
                assert_eq!(after[i].placement, before[i].placement, "{:?}: obj {} placement drifted", p, i);
                assert_eq!(after[i].spawn_seq, before[i].spawn_seq, "{:?}: obj {} spawn_seq drifted", p, i);
                assert_eq!(after[i].respawn, before[i].respawn, "{:?}: obj {} respawn drifted", p, i);
            }
        }
    }

    /// Deleting object 0 must drop exactly one object and shift the rest down by one.
    #[test]
    fn encode_delete_applies() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            if before.len() < 2 {
                continue;
            }
            let mut edits = std::collections::HashMap::new();
            edits.insert(0usize, ObjEdit { deleted: true, ..Default::default() });
            let re = encode_payload(&payload, &edits, &[], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(after.len(), before.len() - 1, "{:?}: delete count wrong", p);
            assert_eq!(after[0].folder, before[1].folder, "{:?}: wrong object after delete", p);
        }
    }

    /// Adding a new object must increase the count by one and round-trip its folder/item/pos.
    #[test]
    fn encode_add_applies() {
        let no_edits = std::collections::HashMap::new();
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            if before.is_empty() {
                continue;
            }
            // Reuse an existing object's (folder,item) so it resolves; place it at a new position.
            let (f, i) = (before[0].folder, before[0].item);
            let target = [before[0].pos[0] + 2.0, before[0].pos[1] + 2.0, before[0].pos[2] + 2.0];
            let add = new_placed_object(f, i, target, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0xFF, -1);
            let re = encode_payload(&payload, &no_edits, std::slice::from_ref(&add), None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(after.len(), before.len() + 1, "{:?}: add count wrong", p);
            // The added object is the one at our target position with the reused folder/item.
            let found = after.iter().any(|o| {
                o.folder == f && o.item == i && (0..3).all(|k| (o.pos[k] - target[k]).abs() < 0.05)
            });
            assert!(found, "{:?}: added object not found", p);
        }
    }

    /// A NEW placement carries the game's NEUTRAL team (raw 9 -> decoded 8) through encode +
    /// parse, and so do an explicit team (purple = 4), an explicit object colour (the "set
    /// colour" path: colour 4 with team neutral) and the distinct "none" (raw 0).
    #[test]
    fn new_placement_team_neutral_roundtrips() {
        let no_edits = std::collections::HashMap::new();
        let mut checked = 0;
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            if before.is_empty() {
                continue;
            }
            let (f, i) = (before[0].folder, before[0].item);
            let base = before[0].pos;
            let cases: [(u8, i32); 4] = [(TEAM_NEUTRAL, -1), (4, -1), (TEAM_NEUTRAL, 4), (TEAM_NONE, -1)];
            let adds: Vec<PlacedObject> = cases.iter().enumerate().map(|(k, (t, c))| {
                let target = [base[0] + 3.0 + k as f32, base[1] + 3.0, base[2] + 3.0];
                new_placed_object(f, i, target, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], *t, *c)
            }).collect();
            let re = encode_payload(&payload, &no_edits, &adds, None);
            let (_, _, _, after) = parse_payload(&re);
            if after.len() != before.len() + adds.len() {
                continue; // no free slots in this sample: nothing to check here
            }
            for (k, (t, c)) in cases.iter().enumerate() {
                let target = adds[k].pos;
                let o = after.iter().find(|o| {
                    o.folder == f && o.item == i && (0..3).all(|j| (o.pos[j] - target[j]).abs() < 0.05)
                }).unwrap_or_else(|| panic!("{:?}: added object {k} not found", p));
                assert_eq!(o.team, *t, "{:?}: case {k} team byte did not round-trip", p);
                assert_eq!(o.color, *c, "{:?}: case {k} colour did not round-trip", p);
            }
            checked += 1;
        }
        assert!(checked > 0, "no sample variant with free slots was found");
        assert_eq!(TEAM_NEUTRAL, 8);
        assert_eq!(TEAM_NONE, 0xFF);
    }

    // ---- rotation-encode helpers/tests ------------------------------------------------------
    fn dot3f(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
    /// Rodrigues rotation of `v` about unit `axis` by `angle` (f64), returned f32.
    fn rotate_axis_angle(v: [f32; 3], axis: [f64; 3], angle: f64) -> [f32; 3] {
        let mut a = axis;
        normalize3(&mut a);
        let vf = [v[0] as f64, v[1] as f64, v[2] as f64];
        let (s, c) = (angle.sin(), angle.cos());
        let cr = cross3(a, vf);
        let d = dot3f(a, vf);
        let r = [
            vf[0] * c + cr[0] * s + a[0] * d * (1.0 - c),
            vf[1] * c + cr[1] * s + a[1] * d * (1.0 - c),
            vf[2] * c + cr[2] * s + a[2] * d * (1.0 - c),
        ];
        [r[0] as f32, r[1] as f32, r[2] as f32]
    }
    /// Tiny deterministic LCG so the tests need no rand crate.
    struct Lcg(u64);
    impl Lcg {
        fn next_f(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// (1) `dequantize_unit_vector3d(quantize_unit_vector3d(v)) ≈ v` for a grid of directions.
    #[test]
    fn quantize_unit_vector_roundtrip() {
        let mut worst = 0.0f64;
        // A dense grid over the sphere via spherical angles, plus the 6 axis directions.
        let mut dirs: Vec<[f64; 3]> = vec![
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ];
        for i in 0..24 {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / 24.0; // (0, π)
            for j in 0..48 {
                let phi = 2.0 * std::f64::consts::PI * j as f64 / 48.0;
                dirs.push([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]);
            }
        }
        for v in dirs {
            let q = quantize_unit_vector3d(v, 20);
            let d = dequantize_unit_vector3d(q, 20);
            let mut vn = v;
            normalize3(&mut vn);
            let dot = dot3f(vn, d).clamp(-1.0, 1.0);
            worst = worst.max(1.0 - dot);
            assert!(dot > 0.9999, "octa round-trip too lossy for {:?}: dot={} q={}", v, dot, q);
            for k in 0..3 {
                assert!((vn[k] - d[k]).abs() < 0.01, "component drift {:?} -> {:?}", vn, d);
            }
        }
        eprintln!("quantize_unit_vector_roundtrip: worst (1-dot) = {:.2e}", worst);
    }

    /// (2) `decode_orientation(encode_orientation(fwd,up)) ≈ (fwd,up)` for arbitrary tilted pairs
    /// built from random axis-angle rotations of the reference frame (fwd=+X, up=+Z).
    #[test]
    fn encode_orientation_roundtrip() {
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let mut worst_fwd = 0.0f64;
        let mut worst_up = 0.0f64;
        for _ in 0..4000 {
            // random unit axis
            let axis = [rng.next_f() * 2.0 - 1.0, rng.next_f() * 2.0 - 1.0, rng.next_f() * 2.0 - 1.0];
            if dot3f(axis, axis) < 1e-6 { continue; }
            let angle = (rng.next_f() * 2.0 - 1.0) * std::f64::consts::PI;
            let fwd = rotate_axis_angle([1.0, 0.0, 0.0], axis, angle);
            let up = rotate_axis_angle([0.0, 0.0, 1.0], axis, angle);
            let (g, uq, fq) = encode_orientation(fwd, up);
            let (df, du) = decode_orientation(g, uq, fq);
            let f64v = |a: [f32; 3]| [a[0] as f64, a[1] as f64, a[2] as f64];
            let dfwd = dot3f(f64v(fwd), f64v(df)).clamp(-1.0, 1.0);
            let dup = dot3f(f64v(up), f64v(du)).clamp(-1.0, 1.0);
            worst_fwd = worst_fwd.max(1.0 - dfwd);
            worst_up = worst_up.max(1.0 - dup);
            assert!(dup > 0.999, "up round-trip: dot={} up={:?} decoded={:?}", dup, up, du);
            assert!(dfwd > 0.999, "fwd round-trip: dot={} fwd={:?} decoded={:?}", dfwd, fwd, df);
        }
        eprintln!("encode_orientation_roundtrip: worst 1-dot fwd={:.2e} up={:.2e}", worst_fwd, worst_up);
    }

    /// (3) Adding a NEW object with a TILTED up (up=+Y) + a yaw via `new_placed_object` must persist
    /// the full orientation through encode→reparse, and leave placement/flags + existing objects intact.
    #[test]
    fn encode_add_rotation_persists() {
        let p = std::path::PathBuf::from(r"target\release\batch_in2\asylum.mvar");
        if !p.exists() { return; }
        let data = std::fs::read(&p).unwrap();
        let payload = blf_mvar_payload(&data).unwrap();
        let (_, _, _, before) = parse_payload(&payload);
        assert!(!before.is_empty());
        // Distinct target position so the add is uniquely identifiable by (folder,item,pos).
        let (f, i) = (before[0].folder, before[0].item);
        let pos = [before[0].pos[0] + 7.0, before[0].pos[1] + 5.0, before[0].pos[2] + 3.0];
        // up = +Y (a 90° tilt), fwd chosen ⟂ up (a yaw about +Y): fwd = normalize([1,0,1]).
        let up = [0.0, 1.0, 0.0];
        let fwd = { let m = (2.0f32).sqrt(); [1.0 / m, 0.0, 1.0 / m] };
        let add = new_placed_object(f, i, pos, fwd, up, 0xFF, -1);
        // The constructor must NOT collapse to global-up.
        assert!(!add.up_is_global, "tilted up must encode as non-global");
        let re = encode_payload(&payload, &std::collections::HashMap::new(), &[add], None);
        let (_, _, _, after) = parse_payload(&re);
        assert_eq!(after.len(), before.len() + 1, "add not written");
        // Identify the add by its unique (folder,item,pos) — mirrors encode_add_applies.
        let obj = after.iter().find(|o| o.folder == f && o.item == i
            && (0..3).all(|k| (o.pos[k] - pos[k]).abs() < 0.05))
            .expect("tilted added object present");
        let f64v = |a: [f32; 3]| [a[0] as f64, a[1] as f64, a[2] as f64];
        let dup = dot3f(f64v(obj.up), f64v(up)).clamp(-1.0, 1.0);
        let dfwd = dot3f(f64v(obj.fwd), f64v(fwd)).clamp(-1.0, 1.0);
        assert!(dup > 0.999, "added up not persisted: {:?} vs {:?}", obj.up, up);
        assert!(dfwd > 0.999, "added fwd not persisted: {:?} vs {:?}", obj.fwd, fwd);
        // spawn fields intact.
        assert_eq!(obj.flags, 1, "new object flags must be 1");
        assert_eq!(obj.placement & 0b1100, 0b1100, "new object placement spawn bits");
        // Existing objects untouched.
        for (a, b) in before.iter().zip(after.iter()) {
            assert_eq!(a.folder, b.folder);
            assert_eq!(a.item, b.item);
        }
    }

    /// (4) An ObjEdit that changes an existing object's rotation + label_idx must round-trip through
    /// encode→reparse, while OTHER objects keep their orientation/label.
    #[test]
    fn edit_rotation_label_persists() {
        for p in sample_mvars() {
            let data = std::fs::read(&p).unwrap();
            let payload = blf_mvar_payload(&data).unwrap();
            let (_, _, _, before) = parse_payload(&payload);
            if before.len() < 2 { continue; }
            // Tilt object 0: up=+Y, fwd ⟂ up. label_idx = 3 (arbitrary, differs from 0xFFFF).
            let up = [0.0, 1.0, 0.0];
            let fwd = { let m = (2.0f32).sqrt(); [1.0 / m, 0.0, 1.0 / m] };
            let target_label = 3u16;
            assert_ne!(before[0].label_idx, target_label, "pick a different label");
            let mut edits = std::collections::HashMap::new();
            edits.insert(0usize, ObjEdit {
                rot: Some((fwd, up)),
                label_idx: Some(target_label),
                ..Default::default()
            });
            let re = encode_payload(&payload, &edits, &[], None);
            let (_, _, _, after) = parse_payload(&re);
            assert_eq!(before.len(), after.len(), "{:?}: count changed", p);
            let f64v = |a: [f32; 3]| [a[0] as f64, a[1] as f64, a[2] as f64];
            assert!(!after[0].up_is_global, "{:?}: edited up should be non-global", p);
            let dup = dot3f(f64v(after[0].up), f64v(up)).clamp(-1.0, 1.0);
            let dfwd = dot3f(f64v(after[0].fwd), f64v(fwd)).clamp(-1.0, 1.0);
            assert!(dup > 0.999, "{:?}: edited up not persisted: {:?}", p, after[0].up);
            assert!(dfwd > 0.999, "{:?}: edited fwd not persisted: {:?}", p, after[0].fwd);
            assert_eq!(after[0].label_idx, target_label, "{:?}: label not persisted", p);
            // Every OTHER object keeps its rotation fields + label.
            for i in 1..before.len() {
                assert_eq!(after[i].up_is_global, before[i].up_is_global, "{:?}: obj {} up_is_global drift", p, i);
                assert_eq!(after[i].up_quant, before[i].up_quant, "{:?}: obj {} up_quant drift", p, i);
                assert_eq!(after[i].forward_angle_q, before[i].forward_angle_q, "{:?}: obj {} yaw drift", p, i);
                assert_eq!(after[i].label_idx, before[i].label_idx, "{:?}: obj {} label drift", p, i);
            }
        }
    }

    /// Compare the first `nbits` bits of two byte slices (MSB-first, matching the bitstream).
    fn bits_eq(a: &[u8], b: &[u8], nbits: usize) -> bool {
        let full = nbits / 8;
        if a.len() < full || b.len() < full || a[..full] != b[..full] {
            return false;
        }
        let rem = nbits % 8;
        if rem > 0 {
            let mask = 0xFFu8 << (8 - rem);
            if (a[full] & mask) != (b[full] & mask) {
                return false;
            }
        }
        true
    }

    /// Reach CreatedBy / ModifiedBy stamps: the parsed (timestamp, xuid) pairs
    /// equal the `chdr` chunk's little-endian copies (so the byte order is right); `stamp_new`
    /// rewrites exactly the 128 bits before each name (created = now / modified = 0) and nothing
    /// else in the header; `stamp_modified` keeps created; an all-None edit is still verbatim.
    /// Runs on the first shipped Reach hopper variant on this machine (skips without the game).
    #[test]
    fn created_modified_stamps_reach() {
        let Some(p) = crate::mapcat::variant_dirs().iter()
            .filter(|d| d.file_name().map_or(false, |n| n == "hopper_map_variants"))
            .flat_map(|d| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()))
            .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")))
            .min() else { eprintln!("skip: no Reach variant folders"); return };
        let data = read_expanded(&p).unwrap();
        let payload = blf_mvar_payload(&data).unwrap();
        let (hi, obj_start) = { let mut r = BitReader::new(&payload); let hi = parse_header(&mut r); (hi, r.pos) };
        // the chdr chunk (at +0x30, body +12) mirrors the two stamps little-endian at body +0x3C / +0x60
        let chdr = data.windows(4).position(|w| w == b"chdr").unwrap() + 12;
        let le = |o: usize| u64::from_le_bytes(data[chdr + o..chdr + o + 8].try_into().unwrap());
        assert_eq!(hi.created, (le(0x3C), le(0x44)), "{}: created vs chdr", p.display());
        assert_eq!(hi.modified, (le(0x60), le(0x68)), "{}: modified vs chdr", p.display());
        let no_edits = std::collections::HashMap::new();
        let mut e = HeaderEdits::default();
        e.stamp_new(0x0009_01F7_0000_0001);
        let (ts, _) = e.created.unwrap();
        assert!(ts >= 1_750_000_000);
        let out = encode_payload(&payload, &no_edits, &[], Some(&e));
        let (ho, obj_start2) = { let mut r = BitReader::new(&out); let hi = parse_header(&mut r); (hi, r.pos) };
        assert_eq!(obj_start2, obj_start, "fixed-width stamps never move a bit");
        assert_eq!(ho.created, (ts, 0x0009_01F7_0000_0001));
        assert_eq!(ho.modified, (0, 0));
        assert_eq!(ho.author, hi.author);
        assert_eq!(ho.editor, hi.editor);
        assert_eq!(ho.title, hi.title);
        // every header bit outside the two 128-bit stamps is unchanged
        let bit = |b: &[u8], i: usize| (b[i >> 3] >> (7 - (i & 7))) & 1;
        for i in 0..obj_start {
            let in_stamp = (hi.name1_start - 128..hi.name1_start).contains(&i) || (hi.name2_start - 128..hi.name2_start).contains(&i);
            if !in_stamp { assert_eq!(bit(&out, i), bit(&payload, i), "header bit {i} changed outside the stamps"); }
        }
        let mut e2 = HeaderEdits::default();
        e2.stamp_modified(7);
        let out2 = encode_payload(&payload, &no_edits, &[], Some(&e2));
        let h2 = { let mut r = BitReader::new(&out2); parse_header(&mut r) };
        assert_eq!(h2.created, hi.created);
        assert_eq!(h2.modified.1, 7);
        let none = encode_payload(&payload, &no_edits, &[], Some(&HeaderEdits::default()));
        assert!(bits_eq(&none, &payload, obj_start), "all-None header splice is NOT bit-identical");
    }

    /// Header-edit gate: (1) re-encoding with the SAME header strings must reproduce the header
    /// bit-for-bit (splice == verbatim) and leave every object intact; (2) re-encoding with CHANGED
    /// strings must read those strings back AND keep the object array fully intact.
    #[test]
    fn header_string_roundtrip() {
        let p = std::path::PathBuf::from(r"target\release\batch_in2\asylum.mvar");
        if !p.exists() {
            return; // sample not on this machine
        }
        let no_edits = std::collections::HashMap::new();
        let data = std::fs::read(&p).unwrap();
        let payload = blf_mvar_payload(&data).unwrap();
        let (_, _, _, before) = parse_payload(&payload);
        assert!(!before.is_empty(), "asylum should have objects to guard");

        // Original description + the bit offset where the object array begins.
        let (desc, obj_start) = {
            let mut r = BitReader::new(&payload);
            let hi = parse_header(&mut r);
            (hi.description, r.pos)
        };

        // (1a) No-edit save (header=None) → the HEADER region is bit-identical to the original, and
        // every object round-trips (the object array is re-emitted, so it matches semantically /
        // within quantisation tolerance — the same guarantee the object-edit tests assert).
        let re_verbatim = encode_payload(&payload, &no_edits, &[], None);
        assert!(
            bits_eq(&re_verbatim, &payload, obj_start),
            "no-edit re-encode: header region is NOT bit-identical to the original"
        );
        assert_eq!(re_verbatim.len(), payload.len(), "no-edit re-encode changed the payload size");
        {
            let (_, _, _, after) = parse_payload(&re_verbatim);
            assert_eq!(before.len(), after.len(), "no-edit re-encode changed the object count");
            for (a, b) in before.iter().zip(after.iter()) {
                assert_eq!(a.folder, b.folder);
                assert_eq!(a.item, b.item);
                for k in 0..3 {
                    assert!((a.pos[k] - b.pos[k]).abs() < 0.05, "no-edit object pos drifted");
                }
            }
        }

        // (1b) An all-`None` HeaderEdits (every field kept) must ALSO reproduce the header verbatim,
        // even though asylum's author/editor names aren't clean UTF-8 (they must be copied, not
        // re-encoded from a lossy String).
        let all_none = HeaderEdits::default();
        let re_none = encode_payload(&payload, &no_edits, &[], Some(&all_none));
        assert!(
            bits_eq(&re_none, &payload, obj_start),
            "all-None header splice is NOT bit-identical to the original header"
        );
        assert_eq!(re_none, re_verbatim, "all-None splice must equal the verbatim path");

        // (2) Change ONLY Title + Author; leave Description + Editor untouched (None → verbatim).
        // The changed pair must read back, the untouched pair must be preserved, objects intact.
        let changed = HeaderEdits {
            title: Some("Test Title".into()),
            author: Some("sopitive".into()),
            description: None,
            editor: None,
            labels: None,
            ..Default::default()
        };
        let re_ch = encode_payload(&payload, &no_edits, &[], Some(&changed));
        // Original editor (kept as-is; asylum's is not clean UTF-8 so we compare to the parsed value).
        let editor_orig = { let mut r = BitReader::new(&payload); parse_header(&mut r).editor };
        let (t2, d2, a2, e2) = {
            let mut r = BitReader::new(&re_ch);
            let hi = parse_header(&mut r);
            (hi.title, hi.description, hi.author, hi.editor)
        };
        assert_eq!(t2, "Test Title", "changed title did not round-trip");
        assert_eq!(a2, "sopitive", "changed author did not round-trip");
        assert_eq!(d2, desc, "untouched description was not preserved");
        assert_eq!(e2, editor_orig, "untouched editor was not preserved");

        let (_, _, _, after_ch) = parse_payload(&re_ch);
        assert_eq!(before.len(), after_ch.len(), "object count changed after header edit");
        // First AND last object survive with matching folder/item and position.
        for &idx in &[0usize, before.len() - 1] {
            assert_eq!(before[idx].folder, after_ch[idx].folder, "obj {idx} folder drifted");
            assert_eq!(before[idx].item, after_ch[idx].item, "obj {idx} item drifted");
            for k in 0..3 {
                assert!(
                    (before[idx].pos[k] - after_ch[idx].pos[k]).abs() < 0.05,
                    "obj {idx} pos drifted after header edit"
                );
            }
        }
    }

    /// Forge-label gate: (1) re-encoding with `labels=None` must reproduce the header (including the
    /// forge-label table) bit-for-bit — the untouched-table verbatim path; (2) re-encoding with an
    /// APPENDED label must (a) read the new label back, (b) preserve the ORIGINAL labels, and (c)
    /// leave the object array fully intact (count + first/last folder/item/pos).
    #[test]
    fn forge_label_roundtrip() {
        let p = std::path::PathBuf::from(r"target\release\batch_in2\asylum.mvar");
        if !p.exists() {
            return; // sample not on this machine
        }
        let no_edits = std::collections::HashMap::new();
        let data = std::fs::read(&p).unwrap();
        let payload = blf_mvar_payload(&data).unwrap();
        let (_, _, _, before) = parse_payload(&payload);
        assert!(!before.is_empty(), "asylum should have objects to guard");

        // Original labels + the bit offset where the object array begins.
        let (labels_orig, obj_start) = {
            let mut r = BitReader::new(&payload);
            let hi = parse_header(&mut r);
            (hi.labels, r.pos)
        };

        // (1) A HeaderEdits with labels=None (all fields None) must copy the whole header — including
        // the forge-label table — VERBATIM (bit-identical to the original header region).
        let keep = HeaderEdits::default(); // labels: None
        let re_keep = encode_payload(&payload, &no_edits, &[], Some(&keep));
        assert!(
            bits_eq(&re_keep, &payload, obj_start),
            "labels=None re-encode: header region (incl. label table) is NOT bit-identical"
        );
        assert_eq!(re_keep.len(), payload.len(), "labels=None re-encode changed the payload size");

        // (2) Append "TestLabel" to the label table and re-encode.
        let mut new_labels = labels_orig.clone();
        new_labels.push("TestLabel".to_string());
        let edited = HeaderEdits { labels: Some(new_labels.clone()), ..Default::default() };
        let re_add = encode_payload(&payload, &no_edits, &[], Some(&edited));

        // (2a/2b) The label table reads back with the new label AND all originals preserved in order.
        let labels_after = {
            let mut r = BitReader::new(&re_add);
            parse_header(&mut r).labels
        };
        assert_eq!(labels_after.len(), labels_orig.len() + 1, "appended label not written");
        for (i, orig) in labels_orig.iter().enumerate() {
            assert_eq!(&labels_after[i], orig, "original label #{i} not preserved");
        }
        assert_eq!(labels_after.last().unwrap(), "TestLabel", "appended label did not round-trip");

        // (2c) The object array is fully intact — count + first/last folder/item/pos.
        let (_, _, _, after) = parse_payload(&re_add);
        assert_eq!(before.len(), after.len(), "object count changed after label edit");
        for &idx in &[0usize, before.len() - 1] {
            assert_eq!(before[idx].folder, after[idx].folder, "obj {idx} folder drifted");
            assert_eq!(before[idx].item, after[idx].item, "obj {idx} item drifted");
            for k in 0..3 {
                assert!(
                    (before[idx].pos[k] - after[idx].pos[k]).abs() < 0.05,
                    "obj {idx} pos drifted after label edit"
                );
            }
        }
    }

    /// Writing the parsed object list straight back must reproduce the payload
    /// BIT-FOR-BIT. If it does not, the simple save is lossy and every save would nudge the map.
    #[test]
    fn writing_the_parsed_list_unchanged_is_bit_identical() {
        let samples = sample_mvars();
        assert!(!samples.is_empty(), "no sample .mvar found — this gate would be vacuous");
        let mut checked = 0;
        for path in samples.iter().take(12) {
            let Some(v) = parse_variant(path) else { continue };
            let Some(bytes) = std::fs::read(path).ok() else { continue };
            let Some(payload) = blf_mvar_payload(&bytes) else { continue };
            let (out, written) = encode_payload_objects(&payload, &v.objects, None);
            assert_eq!(written, v.objects.len(), "{}: wrote {written} of {} objects", path.display(), v.objects.len());
            assert_eq!(
                out, payload,
                "{}: writing the object list back changed the payload — the simple save is LOSSY",
                path.display()
            );
            checked += 1;
        }
        assert!(checked > 0, "no usable sample variant exercised the gate");
    }

    /// Order in == order out, and saving twice == saving once -- the property that keeps objects
    /// out of each other's slots.
    #[test]
    fn list_save_preserves_order_and_is_a_fixed_point() {
        let samples = sample_mvars();
        assert!(!samples.is_empty(), "no sample .mvar found — this gate would be vacuous");
        let mut checked = 0;
        for path in samples.iter().take(12) {
            let Some(v) = parse_variant(path) else { continue };
            let n = v.objects.len();
            if n < 6 { continue; }
            let Some(bytes) = std::fs::read(path).ok() else { continue };
            let Some(payload) = blf_mvar_payload(&bytes) else { continue };
            // drop a MIDDLE object and append a new one — the shape of a real edit session
            let mut list: Vec<PlacedObject> = v.objects.clone();
            let removed = list.remove(n / 2);
            // NOTE: use an IN-BOUNDS position — slot coordinates are quantised into the variant's
            // bounding box, so an out-of-range value comes back clamped and looks like order drift.
            list.push(new_placed_object(removed.folder, removed.item, removed.pos, removed.fwd, removed.up, 0xFF, -1));
            let (out1, _) = encode_payload_objects(&payload, &list, None);
            let back = parse_payload(&out1).3;
            assert_eq!(back.len(), list.len(), "{}: object count changed", path.display());
            for (i, (a, b)) in back.iter().zip(list.iter()).enumerate() {
                assert_eq!(a.folder, b.folder, "{}: object {i} type changed — order drifted", path.display());
                assert_eq!(a.item, b.item, "{}: object {i} variant changed — order drifted", path.display());
                let d = (a.pos[0] - b.pos[0]).abs().max((a.pos[1] - b.pos[1]).abs()).max((a.pos[2] - b.pos[2]).abs());
                assert!(d < 0.01, "{}: object {i} moved by {d} — order drifted", path.display());
            }
            // saving the result again changes nothing
            let (out2, _) = encode_payload_objects(&out1, &back, None);
            assert_eq!(out2, out1, "{}: second save differs from the first", path.display());
            checked += 1;
        }
        assert!(checked > 0, "no usable sample variant exercised the gate");
    }

    /// Regression gate for the "hodgepodge" corruption: the app addresses objects by `datum = 0xD0000000 + i`, where `i` is the i-th REAL slot in
    /// order. So a save must NEVER reorder the retained originals: if a new object is slotted ahead
    /// of them, every later object's index shifts by one and the NEXT save writes each object's
    /// position/rotation/type into a DIFFERENT object's slot. The map comes back with walls where
    /// spawns should be and objects rotated wrong.
    ///
    /// Deleting a MIDDLE object and adding a new one must leave the surviving prefix byte-identical
    /// and must not insert the add before any survivor.
    #[test]
    fn adds_never_reorder_the_retained_objects() {
        let samples = sample_mvars();
        assert!(!samples.is_empty(), "no sample .mvar found — this gate would be vacuous");
        let mut checked = 0;
        for path in samples.iter().take(8) {
            let Some(v) = parse_variant(path) else { continue };
            let n = v.objects.len();
            if n < 8 { continue; }
            let Some(bytes) = std::fs::read(path).ok() else { continue };
            let Some(payload) = blf_mvar_payload(&bytes) else { continue };
            if absent_slot_count(&payload) == 0 { continue; } // needs somewhere for the add to go
            let k = n / 2; // delete a MIDDLE object
            let mut edits: std::collections::HashMap<usize, ObjEdit> = std::collections::HashMap::new();
            edits.insert(k, ObjEdit { deleted: true, ..Default::default() });
            let src0 = &v.objects[0];
            let adds = vec![new_placed_object(src0.folder, src0.item, [12345.0, 0.0, 0.0], src0.fwd, src0.up, 0xFF, -1)];
            let (out, _) = encode_payload_counted(&payload, &edits, &adds, None);
            let after = parse_payload(&out).3;
            assert_eq!(after.len(), n, "{}: object count changed", path.display());
            // every object BEFORE the deleted one keeps its index and its identity
            for i in 0..k {
                assert_eq!(after[i].folder, v.objects[i].folder, "{}: object {i} changed type — enumeration shifted", path.display());
                assert_eq!(after[i].item, v.objects[i].item, "{}: object {i} changed variant — enumeration shifted", path.display());
                assert_eq!(after[i].pos, v.objects[i].pos, "{}: object {i} moved — enumeration shifted", path.display());
            }
            // the survivors AFTER the hole shift down by exactly one, in order — the add must not
            // have been slotted into the hole
            for i in k..(n - 1) {
                assert_eq!(after[i].pos, v.objects[i + 1].pos,
                    "{}: object {i} is not the expected survivor — an add was slotted into the freed hole",
                    path.display());
            }
            checked += 1;
        }
        assert!(checked > 0, "no usable sample variant exercised the gate");
    }

    // ---- global fields ---------------------------------------------------------------------

    /// EVERY Reach variant on this machine, no truncation: the shipped folders, the loose
    /// `/mnt/games/haloreach/map_variants`, and every MCC account's `HaloReach\Map` save folder.
    /// Files that are not a v31 `mvar` chunk (a foreign / damaged file in a save folder) are
    /// skipped by the callers.
    fn all_reach_mvars() -> Vec<std::path::PathBuf> {
        let mut dirs: Vec<std::path::PathBuf> = crate::mapcat::variant_dirs();
        dirs.push(std::path::PathBuf::from("/mnt/games/haloreach/map_variants"));
        for root in crate::mapcat::mcc_localfiles_roots() {
            for x in std::fs::read_dir(&root).into_iter().flatten().flatten() {
                dirs.push(x.path().join("HaloReach").join("Map"));
            }
        }
        let mut out: Vec<std::path::PathBuf> = Vec::new();
        for d in dirs {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("mvar")) && !out.contains(&p) { out.push(p); }
            }
        }
        out.sort();
        out
    }

    /// Is this a Reach map-variant file HMS handles (a `mvar` chunk of major version 31)?
    fn is_reach_v31(data: &[u8]) -> bool {
        let mut pos = 0usize;
        while pos + 12 <= data.len() {
            let size = u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
            if size < 12 || pos + size > data.len() { return false; }
            if &data[pos..pos + 4] == b"mvar" { return u16::from_be_bytes([data[pos + 8], data[pos + 9]]) == 31 && size == 28712; }
            pos += size;
        }
        false
    }

    /// Every Reach variant on this machine (shipped + game-saved, ~500 files): the global
    /// fields decode consistently with the little-endian `chdr` mirror (type, file size, the
    /// four ids, activity / game mode / engine, map id, category, both stamps), the body map id
    /// equals the header's, the stored chunk `length` equals ceil(end_bit / 8) with the quota
    /// table as the last thing written, the stored SHA-1 is `sha1(LE length || payload[..length])`,
    /// the quota table's `placed_on_map` equals the slot census, and a no-edit `rebuild_file`
    /// (which recomputes length + hash and mirrors the chdr) is byte-identical.
    #[test]
    fn globals_decode_and_chunk_hash_on_every_reach_variant() {
        let files = all_reach_mvars();
        if files.is_empty() { eprintln!("skip: no Reach variant folders"); return; }
        let (mut checked, mut skipped, mut v32, mut fsm, mut odd_size) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut quota_rows, mut quota_odd, mut stale_hdr, mut odd_len, mut gaps, mut cmp_files, mut mirror_odd, mut map_id_odd) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut dirty_tail_count = 0usize;
        for p in &files {
            let data = read_expanded(p).unwrap();
            if !is_reach_v31(&data) { skipped += 1; continue; }
            let Some(v) = parse_variant(p) else { skipped += 1; continue };
            let g = &v.globals;
            if g.version != 31 && g.version != 32 { skipped += 1; continue; }
            let chdr = data.windows(4).position(|w| w == b"chdr").unwrap() + 12;
            let b = &data[chdr..];
            let le32 = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
            let le64 = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
            // (Empire.mvar, the one `_cmp`-compressed file, carries a chdr from another save of
            // the same map - its ids / size disagree with the bitstream; the mirror check is for
            // game-written files)
            let cmp = std::fs::read(p).map_or(false, |raw| raw.windows(4).take(4096).any(|w| w == b"_cmp"));
            if cmp { cmp_files += 1; }
            // (a few tool-written files - TEST.mvar, chainTest.mvar - carry a garbage chdr; the
            // mirror is a census with a floor, not a per-file gate)
            let mirror_bad = std::cell::Cell::new(false);
            let chk = |a: u64, b: u64, _what: &str| { if a != b { mirror_bad.set(true); } };
            chk(g.content_type as i32 as u64, le32(0x04) as i32 as u64, "type");
            // (Empire.mvar ships `_cmp`-compressed: its chdr file-size is the compressed file's)
            if g.file_size != le32(0x08) { odd_size += 1; }
            chk(g.uid, le64(0x0C), "uid");
            chk(g.parent_uid, le64(0x14), "parent-uid");
            chk(g.root_uid, le64(0x1C), "root-uid");
            chk(g.game_id, le64(0x24), "game-id");
            chk(g.activity as u64, b[0x2C] as i8 as u64, "activity");
            chk(g.game_mode as u64, b[0x2D] as u64, "game-mode");
            chk(g.engine as u64, b[0x2E] as u64, "engine");
            chk(g.header_map_id as u64, le32(0x30) as u64, "map-id");
            chk(g.category as u64, b[0x34] as i8 as u64, "category");
            chk(g.created.0, le64(0x3C), "created ts"); chk(g.created.1, le64(0x44), "created xuid");
            chk(g.created_online as u64, (b[0x5C] & 1) as u64, "created online");
            chk(g.modified.0, le64(0x60), "modified ts"); chk(g.modified.1, le64(0x68), "modified xuid");
            chk(g.modified_online as u64, (b[0x80] & 1) as u64, "modified online");
            if mirror_bad.get() { mirror_odd += 1; }
            assert_eq!(g.content_type, 5, "{}: a map variant is content type 5", p.display());
            // (a modded map's variant - cenotaph_snow, map id 0xFFFFFFFE - has a different header map id)
            if g.map_id != g.header_map_id { map_id_odd += 1; }
            assert!(g.hopper_id.is_none() || g.activity == 2);
            // the `_eof` chunk ends at file-size (the `_fsm` signature block is not counted)
            let eof = data.windows(4).position(|w| w == b"_eof").unwrap();
            if g.file_size as usize != eof + 17 { odd_size += 1; } // theCage ships 29484; a few tool-written files store it byte-swapped
            // chunk length + hash
            let payload = blf_mvar_payload(&data).unwrap();
            assert_eq!(g.end_bit, payload_end_bit(&payload));
            // (a file saved by an older writer carries the SOURCE's length + hash - stale; such
            // files are counted, and their rebuild below gets a correct header, so they cannot be
            // byte-identical either)
            // (one user file, 22f4eb5f, stores a length 194 bytes past the quota table with a
            // matching hash and stale quota-like bytes in between - some other writer's buffer;
            // counted, not asserted: the engine reads the whole 0x7000 bitstream regardless)
            // (a file whose length/hash are self-consistent yet whose tail is non-zero was most
            // recently saved by a writer other than this codec on a payload that used to be
            // longer - e.g. the game itself, mid-session, on a file the user keeps editing in
            // Forge (observed on scaleTest.mvar / spookyForest.mvar, both re-touched between two
            // runs of this test). The engine reads the whole 0x7000 bitstream regardless of the
            // declared length, so a dirty tail is harmless, not a corruption; treated as another
            // flavour of "stale" so the byte-identical check below is skipped for it, but the
            // bytes BEFORE the declared end still have to match exactly.
            let dirty_tail = g.hash_ok && ((g.end_bit + 7) / 8) as u32 == g.chunk_length
                && payload[g.chunk_length as usize..].iter().any(|&x| x != 0);
            let stale = !g.hash_ok || ((g.end_bit + 7) / 8) as u32 != g.chunk_length || dirty_tail;
            if !g.hash_ok { stale_hdr += 1; } else if ((g.end_bit + 7) / 8) as u32 != g.chunk_length { odd_len += 1; }
            else if dirty_tail { dirty_tail_count += 1; }
            // quota census
            let mut placed = vec![0u32; g.num_quotas as usize];
            for o in &v.objects { if (o.folder as usize) < placed.len() { placed[o.folder as usize] += 1; } }
            // (headlong_cl_031 ships one entry with placed_on_map = 255 for 2 objects; the game
            // recomputes the column on load, so it is a census, not a gate)
            let mut this_odd = false;
            for (i, q) in g.quotas.iter().enumerate() { quota_rows += 1; if q.2 as u32 != placed[i].min(255) { quota_odd += 1; this_odd = true; } }
            assert_eq!(g.quotas.len(), g.num_quotas as usize);
            // no-edit rebuild (length + hash recomputed, chdr untouched) is byte-identical
            // (a file whose placed column is off gets it REPAIRED by the save, so only its
            // quota bytes and the hash may differ)
            let (re, n) = rebuild_file(&data, &v.objects, None).unwrap();
            assert_eq!(n, v.objects.len());
            assert_eq!(re.len(), data.len());
            // (the list save COMPACTS the slot array - a source with absent slots between
            // objects comes back with the same objects in lower slots, so those files are
            // compared object-by-object instead of byte-by-byte)
            let gapped = v.objects.iter().enumerate().any(|(i, o)| o.slot as usize != i);
            if gapped { gaps += 1; }
            if gapped {
                let tmp = std::env::temp_dir().join(format!("hms_globals_gap_{}.mvar", std::process::id()));
                std::fs::write(&tmp, &re).unwrap();
                let w = parse_variant(&tmp).unwrap();
                let _ = std::fs::remove_file(&tmp);
                assert_eq!(w.objects.len(), v.objects.len());
                for (a, b) in v.objects.iter().zip(w.objects.iter()) {
                    let (mut x, mut y) = (a.clone(), b.clone());
                    x.slot = 0; y.slot = 0;
                    assert_eq!(x, y, "{}: an object changed in the compacting rebuild", p.display());
                }
                let mut gw = w.globals.clone();
                let mut gv = g.clone();
                gw.end_bit = 0; gv.end_bit = 0; gw.chunk_length = 0; gv.chunk_length = 0; gw.hash_ok = true; gv.hash_ok = true;
                gw.quotas.clear(); gv.quotas.clear();
                assert_eq!(gw, gv, "{}: a global changed in the compacting rebuild", p.display());
            } else if !this_odd && !stale {
                if let Some(i) = data.iter().zip(re.iter()).position(|(a, b)| a != b) { panic!("{}: no-edit rebuild differs at byte {i}", p.display()); }
            } else {
                let mvar = data.windows(4).position(|w| w == b"mvar").unwrap();
                let quota_start = mvar + 36 + (g.end_bit - 24 * g.num_quotas as usize) / 8;
                let payload_end = mvar + 36 + (g.end_bit + 7) / 8;
                for i in 0..data.len() {
                    // a dirty tail is a leftover buffer past the declared end, not corruption
                    // (see above) - a mismatch there is expected and never checked
                    if dirty_tail && i >= payload_end { continue; }
                    if data[i] != re[i] && !((mvar + 12..mvar + 36).contains(&i) || (quota_start..payload_end).contains(&i)) {
                        let tmp = std::env::temp_dir().join(format!("hms_globals_dbg_{}.mvar", std::process::id()));
                        std::fs::write(&tmp, &re).unwrap();
                        let w = parse_variant(&tmp).unwrap();
                        let od: Vec<String> = v.objects.iter().zip(w.objects.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(k, (a, b))| format!("#{k}: {a:?} -> {b:?}")).take(1).collect();
                        panic!("{}: repair touched byte {i} outside the quota table / hash / length; bounds {:?}; {od:?}", p.display(), g.bounds);
                    }
                }
            }
            if g.version >= 32 { v32 += 1; assert!(g.mcc_map_id.is_some()); } else { assert!(g.mcc_map_id.is_none()); }
            if g.has_fsm { fsm += 1; }
            checked += 1;
        }
        eprintln!("reach globals: {checked} files checked ({v32} version 32, {fsm} with _fsm, {odd_size} with a file-size that is not the _eof end), {skipped} skipped (not a v31 mvar); {stale_hdr} with a stale chunk hash, {odd_len} with a hashed length past the quota table, {dirty_tail_count} with a self-consistent length/hash but a non-zero tail, {gaps} with slot gaps (compacted on save), {cmp_files} _cmp-compressed, {mirror_odd} whose chdr disagrees with the bitstream header, {map_id_odd} whose header map id differs from the body's; quota placed_on_map: {quota_odd} of {quota_rows} rows differ from the slot census");
        assert!(checked > 0);
        assert!(quota_odd * 100 < quota_rows, "placed_on_map disagrees with the slots on {quota_odd} of {quota_rows} rows");
        assert!(odd_len * 50 <= checked, "{odd_len} of {checked} files store a length that is not the quota-table end");
        assert!(mirror_odd * 20 <= checked, "the chdr mirror disagrees with the bitstream on {mirror_odd} of {checked} files");
        assert!(map_id_odd * 50 <= checked);
    }

    /// Field-exact edits of every editable global (category, maximum budget, quota
    /// minimum / maximum, world bounds): the edited value reads back, EVERY other global and
    /// every object is unchanged, the chdr mirror carries the category, `placed_on_map` is not
    /// taken from the edit, `maximum_count` is still raised to the placed count, and a re-save of
    /// the edited file with no edits is a fixed point. A widened box re-quantises every object
    /// within one step of the old position; a box that would not hold the objects still writes
    /// (clamped, like the engine) but an inverted box is refused.
    #[test]
    fn global_field_edits_are_exact_and_isolated() {
        let files: Vec<_> = all_reach_mvars().into_iter().filter(|p| read_expanded(p).map_or(false, |d| is_reach_v31(&d))).take(12).collect();
        if files.is_empty() { eprintln!("skip: no Reach variant folders"); return; }
        let mut checked = 0;
        for p in &files {
            let data = read_expanded(p).unwrap();
            let Some(v) = parse_variant(p) else { continue };
            if v.objects.is_empty() || v.globals.quotas.is_empty() { continue; }
            let g = &v.globals;
            let same_but = |a: &VariantGlobals, b: &VariantGlobals, what: &str| {
                let mut x = a.clone();
                let mut y = b.clone();
                x.category = 0; y.category = 0;
                x.budget_max = 0; y.budget_max = 0;
                x.bounds = [0.0; 6]; y.bounds = [0.0; 6];
                x.quotas.clear(); y.quotas.clear();
                x.end_bit = 0; y.end_bit = 0;
                x.chunk_length = 0; y.chunk_length = 0;
                assert_eq!(x, y, "{}: {what}: an untouched global changed", p.display());
            };
            // category + budget max + quota min/max in one edit
            let mut e = HeaderEdits::default();
            e.category = Some(3);
            e.budget_max = Some(12_345);
            let mut q: Vec<(u8, u8)> = g.quotas.iter().map(|q| (q.0, q.1)).collect();
            q[0] = (2, 200);
            let last = q.len() - 1;
            q[last] = (0, 0); // maximum below placed -> raised to placed
            e.quotas = Some(q.clone());
            let (out, _) = rebuild_file(&data, &v.objects, Some(&e)).unwrap();
            let tmp = std::env::temp_dir().join(format!("hms_globals_{}.mvar", std::process::id()));
            std::fs::write(&tmp, &out).unwrap();
            let w = parse_variant(&tmp).unwrap();
            assert_eq!(w.globals.category, 3);
            assert_eq!(w.globals.budget_max, 12_345);
            assert_eq!(w.globals.quotas[0].0, 2);
            assert_eq!(w.globals.quotas[0].1, 200);
            assert_eq!(w.globals.quotas[0].2, g.quotas[0].2, "placed is never taken from the edit");
            assert_eq!(w.globals.quotas[last].1, g.quotas[last].2, "maximum raised to placed");
            for i in 1..last { assert_eq!(w.globals.quotas[i], g.quotas[i]); }
            assert_eq!(w.globals.bounds, g.bounds);
            assert_eq!(w.objects, v.objects, "{}: objects changed by a global edit", p.display());
            assert_eq!(w.labels, v.labels);
            assert_eq!((w.title.as_str(), w.description.as_str(), w.author.as_str(), w.editor.as_str()), (v.title.as_str(), v.description.as_str(), v.author.as_str(), v.editor.as_str()));
            assert_eq!(w.globals.end_bit, g.end_bit);
            assert!(w.globals.hash_ok);
            same_but(&w.globals, g, "category/budget/quota");
            let chdr = out.windows(4).position(|x| x == b"chdr").unwrap() + 12;
            assert_eq!(out[chdr + 0x34] as i8, 3, "chdr category mirror");
            // fixed point
            let (again, _) = rebuild_file(&out, &w.objects, None).unwrap();
            assert_eq!(again, out, "{}: re-save of the edited file drifted", p.display());
            // bounds: widen by 25% on every axis
            let mut nb = g.bounds;
            for i in 0..3 {
                let ext = nb[2 * i + 1] - nb[2 * i];
                nb[2 * i] -= 0.25 * ext;
                nb[2 * i + 1] += 0.25 * ext;
            }
            let mut eb = HeaderEdits::default();
            eb.bounds = Some(nb);
            let (outb, _) = rebuild_file(&data, &v.objects, Some(&eb)).unwrap();
            std::fs::write(&tmp, &outb).unwrap();
            let wb = parse_variant(&tmp).unwrap();
            assert_eq!(wb.globals.bounds, nb);
            assert_eq!(wb.objects.len(), v.objects.len());
            let old_axis = compute_axis_bits(21, [g.bounds[1] - g.bounds[0], g.bounds[3] - g.bounds[2], g.bounds[5] - g.bounds[4]]);
            for (a, b) in wb.objects.iter().zip(v.objects.iter()) {
                for i in 0..3 {
                    let old_step = (g.bounds[2 * i + 1] - g.bounds[2 * i]) / (1u32 << old_axis[i]) as f32;
                    let new_step = (nb[2 * i + 1] - nb[2 * i]) / (1u32 << wb_axis(&wb)[i]) as f32;
                    assert!((a.pos[i] - b.pos[i]).abs() <= old_step + new_step, "{}: object moved {} on axis {i} (steps {old_step} / {new_step})", p.display(), (a.pos[i] - b.pos[i]).abs());
                }
                let (mut x, mut y) = (a.clone(), b.clone());
                x.pos = [0.0; 3]; y.pos = [0.0; 3];
                assert_eq!(x, y, "{}: a non-position field changed with the bounds", p.display());
            }
            assert!(wb.globals.hash_ok);
            assert_eq!(((wb.globals.end_bit + 7) / 8) as u32, wb.globals.chunk_length);
            same_but(&wb.globals, g, "bounds");
            let (againb, _) = rebuild_file(&outb, &wb.objects, None).unwrap();
            if againb != outb {
                let mvar = outb.windows(4).position(|x| x == b"mvar").unwrap();
                let diffs: Vec<isize> = (0..outb.len().min(againb.len())).filter(|&i| againb[i] != outb[i]).map(|i| i as isize - (mvar + 36) as isize).filter(|&d| d >= 0).take(12).collect();
                std::fs::write(&tmp, &againb).unwrap();
                let wb2 = parse_variant(&tmp).unwrap();
                let od: Vec<String> = wb.objects.iter().zip(wb2.objects.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(i, (a, b))| format!("#{i}: {:?} {} {:?} -> {:?} {} {:?}", a.pos, a.in_bounds, a.bsp_index, b.pos, b.in_bounds, b.bsp_index)).take(4).collect();
                panic!("{}: re-save of the re-boxed file drifted; payload byte offsets {diffs:?}; end_bit {} vs {} (lengths {} / {}); bounds {:?} axis {:?}; objects: {od:?}", p.display(), wb.globals.end_bit, wb2.globals.end_bit, outb.len(), againb.len(), wb.globals.bounds, wb_axis(&wb));
            }
            // an inverted box is refused
            let mut bad = HeaderEdits::default();
            bad.bounds = Some([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
            assert!(rebuild_file(&data, &v.objects, Some(&bad)).is_err());
            let _ = std::fs::remove_file(&tmp);
            checked += 1;
        }
        assert!(checked > 0, "no sample variant exercised the global edits");
    }

    fn wb_axis(v: &Variant) -> [u32; 3] {
        let b = v.globals.bounds;
        compute_axis_bits(21, [b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    }

    /// `GlobalEdits` seeds from and diffs against `VariantGlobals` field-exactly.
    #[test]
    fn global_edits_diff_is_exact() {
        let mut g = VariantGlobals::default();
        g.category = -1;
        g.budget_max = 10_000;
        g.bounds = [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0];
        g.quotas = vec![(0, 10, 3), (1, 5, 5)];
        let mut e = GlobalEdits::from_globals(&g);
        assert!(!e.is_dirty(&g));
        assert_eq!(e.diff(&g), (None, None, None, None));
        e.category = 2;
        e.quota_minmax[1] = (1, 9);
        assert!(e.is_dirty(&g));
        let d = e.diff(&g);
        assert_eq!(d.0, Some(2));
        assert_eq!(d.1, None);
        assert_eq!(d.2, None);
        assert_eq!(d.3, Some(vec![(0, 10), (1, 9)]));
    }
}

#[cfg(test)]
mod sample_probe {
    #[test]
    fn samples_are_found() {
        let s = super::tests::sample_mvars_probe();
        eprintln!("sample_mvars found {} files", s.len());
        assert!(!s.is_empty(), "round-trip gates would be vacuous: no .mvar samples found");
    }
}
