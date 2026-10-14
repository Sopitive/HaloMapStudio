//! Reach `jmad` (model_animation_graph) DECODE: the animation list and the compressed keyframe
//! codecs (#reach-skin).
//!
//! Every layout here was read off the engine's own code (`haloreach.dll`, image base
//! 0x180000000) and the matching MCC Reach tool build (`reach_tag_test.exe`, the reflection the
//! `tagdefs` dumps come from). The RVAs in the comments are the proof for each claim.
//!
//! # Where the bytes live
//!
//! `jmad +0x1AC` = `tag resource groups`, 12-byte elements `{i32 reference_count, pageable
//! resource @+4}`. Each group's pageable resource resolves to a
//! `c_model_animation_tag_animation_group_resource` (12 B) whose single block is
//! `group_members` -> `c_model_animation_tag_resource_member` (100 B)   [engine +0x213EC4:
//! group stride `4*(base + 3*i)` = 12 B, member stride `4*(base + 25*i)` = 100 B]:
//!
//! ```text
//!   +0x00  i32   animation index          (ZERO in every shipped cache tested)
//!   +0x04  i32   animation checksum       (== c_shared_model_animation.production_checksum)
//!   +0x08  i16   frame count
//!   +0x0A  u8    node count
//!   +0x0B  u8    movement data type       (= frame_info_type_enum)
//!   +0x0C  i32[17] data sizes             (the 17 regions of the blob, in order)
//!   +0x50  tag_data animation data        (size @+0x50, flags, stream pos, ADDRESS @+0x5C, def)
//! ```
//! In a CACHE the member array sits at the START of the resource's definition data and the
//! block header (count i32, pointer u32, 0xF09ADC2A) occupies its LAST 12 bytes - solved from
//! real bytes with `hms-app --probe-jmad`, which only accepts a base where
//! `sum(data_sizes[0..16]) == tag_data.size`. Each member's blob is located by the resource's
//! own fixup list: `fixups[member]` is the byte offset into the decompressed payload.
//!
//! An animation picks its member with `c_shared_model_animation.resource_group` (+0x20) and
//! `resource_group_member` (+0x22).
//!
//! # The blob's 17 regions
//!
//! Region `i` starts at `prefix(i) = sum(sizes[0..i])` and is `sizes[i]` bytes
//! [tool `sub_140181C60`, `k_internal_data_type_count = 17`]. The engine-proven meanings of the
//! first seven differ from the tool's (invisible) reflection field names:
//!
//! ```text
//!   0  STATIC codec block     (applied only for animation type 1 = base)   [engine +0x21A80B]
//!   1  ANIMATED codec block   (what actually plays)                        [engine +0x219D70]
//!   2  static node flags   - 3 bitvectors of B bytes (rotation, translation, scale)
//!   3  animated node flags - 3 bitvectors of B bytes
//!   4  movement (frame info) data, `movement_stride(type)` bytes per frame [engine +0x211C54]
//!   5  pill offset data (real_point3d per frame)
//!   6  uncompressed source codec block (compressor input; never read at runtime)
//!   7..16  blend screen / object-space offsets / ik chains / FIK anchors / event curves
//! ```
//! with `B = 4 * ceil(node_count / 32)` [engine +0x219D70].
//!
//! # Codecs
//!
//! The codec id is BYTE 0 of a codec block, NOT a tag field — the dispatcher at engine
//! +0x219584 indexes `g_codec_descriptions` (`.data +0xBB07E0`, 0x58 stride, 10 entries) with
//! it. (`desired/current compression` in the tag is a COMPRESSOR SELECTOR — "best score",
//! "best compression", "best accuracy", "old codec" — and must not be used as a format id.)
//!
//! ```text
//!   0 no_compression                       (all accessors NULL)
//!   1 uncompressed_static_data             fullframe header, ONE key, quantized rotation
//!   2 uncompressed_animated_data           fullframe header, float4 rotation per frame
//!   3 8byte_quantized_rotation_only        fullframe header, i16x4 rotation per frame
//!   4 byte_keyframe_lightly_quantized      keyframe header, u8 key times
//!   5 word_keyframe_lightly_quantized      keyframe header, u16 key times
//!   6 reverse_byte_keyframe...             ACCESSORS BIT-IDENTICAL to 4 (only the tool-side
//!   7 reverse_word_keyframe...             compressor differs; 7 == 5)
//!   8 blend_screen                         rotation/translation/scale accessors == codec 2
//!   9 curve ('WANG')                        per-channel Catmull/Hermite splines, see below
//! ```
//! A codec survey over all 979 animations of the Reach Spartan graph: region 0 (the static
//! default pose) is ALWAYS codec 1; region 1 (the animated data) is codec 3 (401 animations)
//! or codec 9 (578). Codecs 2/4/5/6/7/8 do not occur in a shipped Reach biped graph.
//!
//! # Codec 9, `_curve_codec`
//!
//! Header, 32 bytes (`s_curve_codec_header`; engine accessors at +0x21834C rotation /
//! +0x218644 translation / +0x21875C scale, writer at tool +0x6F6E50):
//! ```text
//!   +0x00 u8  codec id = 9        +0x01 u8 rotation channels
//!   +0x02 u8  translation chans   +0x03 u8 scale channels
//!   +0x04 f32 mean compression error   (informational; read by nothing)
//!   +0x08 f32 mean compression ratio   (informational)
//!   +0x0C u32 translation offset-table offset, relative to `header + [0x14]`; 0xFF = absent
//!   +0x10 u32 scale offset-table offset, same base;                          0xFF = absent
//!   +0x14 u32 payload base = 32 (every table / channel offset is relative to header + this)
//!   +0x18 u32 total block size
//!   +0x1C u32 'WANG' (0x474E4157) - the format tag the engine asserts on
//! ```
//! `P = header + header[0x14]`; the rotation offset table is `u32[rot_count]` at `P`, the
//! translation one at `P + header[0x0C]`, the scale one at `P + header[0x10]`; channel `i`'s
//! record is at `P + table[i]`. (Records are NOT tightly packed - always go through the table.)
//!
//! Channel record: `+0x00 u8 type id` ('R' / 'T' / 'S'), `+0x02 i16 spline_count` (segments;
//! keys = spline_count + 1), `+0x04 u8 flags` with bit 0 = DENSE (every segment is one frame
//! long, the duration array is omitted and keys are indexed straight by frame).
//! Then, per type:
//! ```text
//!   'R' data @ +0x08:  sparse u8 duration[n], then keys stride 10 (i16 q[3] + u8 lean[4])
//!                      dense                       keys stride  6 (i16 q[3])
//!   'T' +0x08 f32 base.xyz, +0x14 f32 scale, data @ +0x18: sparse stride 9 (i16 v[3]+u8 lean[3])
//!                                                          dense  stride 6 (i16 v[3])
//!   'S' +0x08 f32 base,     +0x0C f32 scale, data @ +0x10: sparse stride 3 (i16 v + u8 lean)
//!                                                          dense  stride 2 (i16 v)
//! ```
//! A rotation key is a 6-byte packed quaternion (engine +0x218220): `a,b,c = i16/32767`,
//! `nz = sqrt(max(0, 1-a^2-b^2))` with the SIGN of `c`, `w = 2*|c| - 1`, `s = sqrt(1-w^2)`,
//! `q = (s*a, s*b, s*nz, w)`.
//! Translation / scale values are `base + scale * (i16/32767)` applied AFTER the spline.
//! A key's `lean` byte packs two 4-bit tangent indices: `m0 = (b >> 4) - 7`, `m1 = (b & 15) - 7`;
//! with `d = p1 - p0` and `u = idx/7`, the Hermite tangent is `d + |u| * (u * 0.3)` - so a lean
//! index of 0 degenerates to exact linear interpolation. The segment is evaluated with the
//! standard cubic Hermite basis `B0 = 2t^3-3t^2+1, B1 = t^3-2t^2+t, B2 = 3t^2-2t^3, B3 = t^3-t^2`.
//! Segment search (engine +0x218198): walk the u8 durations accumulating `cum`, take the first
//! `i` with `cum + duration[i] >= frame`, `t = (frame - cum) / duration[i]`; a frame outside
//! `[0, sum(durations)]` FAILS and the engine then leaves the destination channel untouched.
//!
//! Rotation components are always `i16 / 32767` then renormalised (engine constant
//! 0.000030518509 = 1/32767); translations are always `real_point3d` (3 x f32) and scale a
//! single f32 — Reach has no per-node translation range.
//!
//! # Channel <-> node
//!
//! A codec channel is NOT a node index. Engine +0x219484 builds the map by walking the node
//! bits of the region's PRESENT bitvector and taking the running popcount:
//! `channel(node, kind) = popcount(present[kind] & ((1 << node) - 1))`. A node whose bit is
//! clear keeps whatever the destination pose already holds (so the caller seeds the pose with
//! the render_model's authored defaults).
//!
//! # Timing and combination
//!
//! 30 fps: `duration_seconds = frame_count * 0.033333335` [engine +0x218060], one frame = one
//! tick. `animation_type` 0 none / 1 base / 2 overlay / 3 replacement [tool enum 0x1EAD1B8];
//! a BASE animation at full weight COPIES the sampled channel [engine +0x21D788], an OVERLAY
//! composes `D.q (x) S.q` (Hamilton, destination on the LEFT), `D.t + S.t`, `D.s * S.s`
//! [engine +0x21DAEC].

use glam::{Quat, Vec3};

/// Engine frame rate: `frame_count * 0.033333335` seconds [haloreach.dll +0x218060].
pub const FRAME_RATE: f32 = 30.0;
/// The engine's own 1/32767 rotation dequantiser (the literal in every accessor).
pub const INV_32767: f32 = 0.000030518509;

/// `jmad` tag-meta offsets (see the module docs and `reach_actor::jmad_off`).
pub const OFF_ANIMATIONS: usize = 0x094;
pub const OFF_SKELETON_NODES: usize = 0x048;
pub const OFF_MODES: usize = 0x104;
pub const OFF_RESOURCE_GROUPS: usize = 0x1AC;
/// `jmad` +0x06C `blend screens` -> `animation_blend_screen_block` (28 B). The table every
/// aiming/steering/look OVERLAY indexes to say what its frames MEAN in angles.
pub const OFF_BLEND_SCREENS: usize = 0x06C;
pub const BLEND_SCREEN_ELEM: usize = 28;
/// `c_model_animation` +0x1A `blend screen`, a CHAR block index (0xFF / -1 = none).
pub const ANIM_BLEND_SCREEN: usize = 0x1A;
pub const ANIMATION_ELEM: usize = 60;
pub const ANIM_SHARED_BLOCK: usize = 0x30;
pub const SHARED_ELEM: usize = 212;
pub const MODE_ELEM: usize = 48;
pub const RESOURCE_GROUP_ELEM: usize = 12;
/// `s_animation_graph_node`: 44 bytes, parent index at +0x08 — recovered from the tool's static
/// `animation_graph_node_fields` array (.data +0x1EAC370) and proven by the engine's own stride
/// arithmetic `*(i16*)(base + 4*(addr + 11*node) + 8)` at +0x2199E0 / +0x219618.
pub const GRAPH_NODE_ELEM: usize = 44;
pub const GRAPH_NODE_PARENT: usize = 0x08;
/// `c_model_animation_tag_resource_member` [engine +0x213EC4: stride 4*25].
pub const MEMBER_ELEM: usize = 100;

/// One `animation_blend_screen_block`: the AIMING SCREEN bounds that turn a look direction into
/// a frame of an aim overlay.
///
/// **This is the frame-to-angle convention, read out of the engine's own reflection data** rather
/// than guessed (`reach_tag_test.exe` tagdefs, `model_animation_graph` +0x06C):
///
/// ```text
/// struct animation_blend_screen_block            // 28 bytes
///   +0x000  string id        label
///   +0x004  struct animation_aiming_screen_struct   // 24 bytes
///     +0x000 angle  right yaw per frame
///     +0x004 angle  left yaw per frame
///     +0x008 short  right frame count
///     +0x00A short  left frame count
///     +0x00C angle  down pitch per frame
///     +0x010 angle  up pitch per frame
///     +0x014 short  down pitch frame count
///     +0x016 short  up pitch frame count
/// ```
///
/// The clip's frames are a 2-D GRID: `width = left + 1 + right` columns of yaw, `height = down +
/// 1 + up` rows of pitch, centre cell = the neutral pose, and `*_per_frame` is radians of look
/// angle per grid STEP. Measured on `fp_dmr`'s only screen, `pitch_and_turn`: yaw +/-57 deg x 1
/// frame each side, pitch -80 / +90 deg x 1 each, i.e. a 3x3 grid of 9 frames, which is exactly
/// the frame count of the clips that index it.
///
/// **Decoded and printed for the record; the AIM path does not use it.** The BIPED graphs that
/// carry the third-person aim overlays have no blend screens at all (the Spartan's has zero), and
/// those overlays label their own frames instead -- see `reach_actor::AimGrid`, which is what the
/// look direction is driven from. Only the FIRST-PERSON weapon graphs carry a screen, and the two
/// readings agree.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlendScreen {
    pub label_sid: u32,
    pub right_yaw_per_frame: f32,
    pub left_yaw_per_frame: f32,
    pub right_frames: i16,
    pub left_frames: i16,
    pub down_pitch_per_frame: f32,
    pub up_pitch_per_frame: f32,
    pub down_frames: i16,
    pub up_frames: i16,
}

impl BlendScreen {
    pub fn read(e: &[u8]) -> Option<BlendScreen> {
        if e.len() < BLEND_SCREEN_ELEM {
            return None;
        }
        let f = |o: usize| f32::from_le_bytes([e[o], e[o + 1], e[o + 2], e[o + 3]]);
        let i = |o: usize| i16::from_le_bytes([e[o], e[o + 1]]);
        Some(BlendScreen {
            label_sid: u32::from_le_bytes([e[0], e[1], e[2], e[3]]),
            right_yaw_per_frame: f(0x04),
            left_yaw_per_frame: f(0x08),
            right_frames: i(0x0C),
            left_frames: i(0x0E),
            down_pitch_per_frame: f(0x10),
            up_pitch_per_frame: f(0x14),
            down_frames: i(0x18),
            up_frames: i(0x1A),
        })
    }

    /// Columns of the frame grid (yaw axis).
    pub fn width(&self) -> i32 {
        self.left_frames.max(0) as i32 + 1 + self.right_frames.max(0) as i32
    }
    /// Rows of the frame grid (pitch axis).
    pub fn height(&self) -> i32 {
        self.down_frames.max(0) as i32 + 1 + self.up_frames.max(0) as i32
    }

}

/// `animation_type_enum` [tool enum table 0x1EAD1B8].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnimType { None, Base, Overlay, Replacement }

impl AnimType {
    pub fn from_raw(v: u8) -> AnimType {
        match v { 1 => AnimType::Base, 2 => AnimType::Overlay, 3 => AnimType::Replacement, _ => AnimType::None }
    }
}

/// Bytes per frame of region 4 for each `frame_info_type_enum` value [engine +0x211C54;
/// type 4's stride 24 from the tool's `sub_14048B1A0(this, frames-1, 24)`].
pub fn movement_stride(frame_info_type: u8) -> usize {
    match frame_info_type { 1 => 8, 2 => 12, 3 => 16, 4 => 24, _ => 0 }
}

/// Bitvector size for `node_count` nodes: `4 * ceil(node_count / 32)` [engine +0x219D70].
pub fn bitvector_bytes(node_count: usize) -> usize { 4 * ((node_count + 31) / 32) }

/// One animation's entry in the `jmad` `animations` block + its `shared animation data` header.
#[derive(Clone, Debug)]
pub struct AnimEntry {
    pub index: usize,
    pub name_sid: u32,
    pub name: String,
    pub loop_frame: i16,
    pub node_count: u8,
    pub frame_count: i16,
    pub anim_type: AnimType,
    pub frame_info_type: u8,
    /// `current compression` — a COMPRESSOR SELECTOR, not the storage codec (see module docs).
    pub compression_setting: u8,
    /// `c_model_animation` +0x1A `blend screen`: index into the graph's own `blend screens`
    /// table, or `None` when this clip's frames are a plain timeline. Every aim/look/steer
    /// OVERLAY has one, and it is what says what its frames mean in angles.
    pub blend_screen: Option<usize>,
    pub production_checksum: i32,
    pub resource_group: i16,
    pub resource_group_member: i16,
    pub heading: Vec3,
}

impl AnimEntry {
    pub fn duration_seconds(&self) -> f32 { self.frame_count.max(0) as f32 / FRAME_RATE }
}

/// One resource member: the animation blob plus the 17 region sizes.
#[derive(Clone, Debug)]
pub struct AnimData {
    pub animation_index: i32,
    pub checksum: i32,
    pub frame_count: i16,
    pub node_count: u8,
    pub movement_data_type: u8,
    pub sizes: [i32; 17],
    /// The `animation data` tag_data payload.
    pub blob: Vec<u8>,
}

impl AnimData {
    /// Decode one 100-byte `c_model_animation_tag_resource_member` (without its blob).
    pub fn header_from_member(e: &[u8]) -> Option<AnimData> {
        if e.len() < MEMBER_ELEM { return None; }
        let rd = |o: usize| i32::from_le_bytes([e[o], e[o + 1], e[o + 2], e[o + 3]]);
        let mut sizes = [0i32; 17];
        for i in 0..17 { sizes[i] = rd(0x0C + 4 * i); }
        Some(AnimData {
            animation_index: rd(0x00),
            checksum: rd(0x04),
            frame_count: i16::from_le_bytes([e[0x08], e[0x09]]),
            node_count: e[0x0A],
            movement_data_type: e[0x0B],
            sizes,
            blob: Vec::new(),
        })
    }

    /// `tag_data.size` of the member (region sizes must sum to it — the engine's own sanity check).
    pub fn declared_blob_size(e: &[u8]) -> i32 {
        if e.len() < MEMBER_ELEM { return 0; }
        i32::from_le_bytes([e[0x50], e[0x51], e[0x52], e[0x53]])
    }

    /// `tag_data.address` of the member (the raw cache pointer the fixup resolves).
    pub fn data_address(e: &[u8]) -> u32 {
        if e.len() < MEMBER_ELEM { return 0; }
        u32::from_le_bytes([e[0x5C], e[0x5D], e[0x5E], e[0x5F]])
    }

    pub fn total_size(&self) -> i32 { self.sizes.iter().map(|&s| s.max(0)).sum() }

    /// Byte range of region `i` inside `blob` (`None` when the region is empty / out of range).
    pub fn region(&self, i: usize) -> Option<&[u8]> {
        if i >= 17 { return None; }
        let start: usize = self.sizes[..i].iter().map(|&s| s.max(0) as usize).sum();
        let len = self.sizes[i].max(0) as usize;
        if len == 0 || start + len > self.blob.len() { return None; }
        Some(&self.blob[start..start + len])
    }

    /// `(rotation, translation, scale)` present-bitvectors of a node-flags region (2 = static,
    /// 3 = animated). Each is `bitvector_bytes(node_count)` long, laid out back to back.
    pub fn node_flags(&self, region: usize) -> Option<[&[u8]; 3]> {
        let b = bitvector_bytes(self.node_count as usize);
        let r = self.region(region)?;
        if b == 0 || r.len() < 3 * b { return None; }
        Some([&r[..b], &r[b..2 * b], &r[2 * b..3 * b]])
    }

    /// Per-frame root MOVEMENT deltas from region 4 (empty when the animation has none).
    /// Each entry is `(dx, dy, dz, dyaw)` — the components the frame-info type does not carry
    /// are zero.
    pub fn movement(&self) -> Vec<(f32, f32, f32, f32)> {
        let stride = movement_stride(self.movement_data_type);
        let Some(r) = self.region(4) else { return Vec::new() };
        if stride == 0 { return Vec::new(); }
        let f = |b: &[u8], o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        r.chunks_exact(stride)
            .map(|c| match self.movement_data_type {
                1 => (f(c, 0), f(c, 4), 0.0, 0.0),
                2 => (f(c, 0), f(c, 4), 0.0, f(c, 8)),
                3 => (f(c, 0), f(c, 4), f(c, 8), f(c, 12)),
                // type 4 = dx,dy,dz + an angle-axis rotation; only the yaw-ish magnitude is used here.
                4 => (f(c, 0), f(c, 4), f(c, 8), f(c, 20)),
                _ => (0.0, 0.0, 0.0, 0.0),
            })
            .collect()
    }
}

/// Is bit `n` set in a little-endian u32 bitvector?
fn bit(bv: &[u8], n: usize) -> bool {
    let w = n / 32;
    if w * 4 + 4 > bv.len() { return false; }
    let v = u32::from_le_bytes([bv[w * 4], bv[w * 4 + 1], bv[w * 4 + 2], bv[w * 4 + 3]]);
    v & (1u32 << (n % 32)) != 0
}

/// `channel(node) = popcount(present & ((1 << node) - 1))` [engine +0x219484]. `None` when the
/// node's bit is clear (the channel is absent and the destination pose is kept).
pub fn channel_of(bv: &[u8], node: usize) -> Option<usize> {
    if !bit(bv, node) { return None; }
    let mut c = 0usize;
    for n in 0..node {
        if bit(bv, n) { c += 1; }
    }
    Some(c)
}

/// One node's sampled channels. A `None` field means the animation does not drive it.
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeChannel {
    pub rot: Option<Quat>,
    pub trans: Option<Vec3>,
    pub scale: Option<f32>,
}

#[inline]
fn rd_u32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() { return 0; }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn rd_i16(b: &[u8], o: usize) -> i16 {
    if o + 2 > b.len() { return 0; }
    i16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn rd_f32(b: &[u8], o: usize) -> f32 {
    if o + 4 > b.len() { return 0.0; }
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn rd_vec3(b: &[u8], o: usize) -> Vec3 { Vec3::new(rd_f32(b, o), rd_f32(b, o + 4), rd_f32(b, o + 8)) }

/// A quantized quaternion: four `i16 / 32767`, renormalised (the engine normalises after
/// dequantising — every accessor ends in `v / sqrt(x²+y²+z²+w²)`).
fn quat_quantized(b: &[u8], o: usize) -> Quat {
    let q = Quat::from_xyzw(
        rd_i16(b, o) as f32 * INV_32767,
        rd_i16(b, o + 2) as f32 * INV_32767,
        rd_i16(b, o + 4) as f32 * INV_32767,
        rd_i16(b, o + 6) as f32 * INV_32767,
    );
    norm(q)
}

fn quat_float(b: &[u8], o: usize) -> Quat {
    norm(Quat::from_xyzw(rd_f32(b, o), rd_f32(b, o + 4), rd_f32(b, o + 8), rd_f32(b, o + 12)))
}

fn norm(q: Quat) -> Quat {
    let l2 = q.length_squared();
    if l2.is_finite() && l2 > 1e-12 { q / l2.sqrt() } else { Quat::IDENTITY }
}

/// The engine's keyframe-codec rotation blend: component lerp by the UNNORMALISED key-time
/// weights, then normalise (nlerp, not slerp; the compressor already canonicalised each key's
/// sign so no shortest-arc flip happens inside the codec) [engine +0x24628C / +0x246E40].
fn nlerp_keys(a: Quat, b: Quat, wa: f32, wb: f32) -> Quat {
    let q = Quat::from_xyzw(
        a.x * wa + b.x * wb,
        a.y * wa + b.y * wb,
        a.z * wa + b.z * wb,
        a.w * wa + b.w * wb,
    );
    if q.length_squared().is_finite() && q.length_squared() > 1e-12 { norm(q) } else { b }
}

/// A codec block's channel counts + the class of its header.
#[derive(Clone, Copy, Debug)]
pub struct CodecHeader {
    pub id: u8,
    pub rot_channels: usize,
    pub trans_channels: usize,
    pub scale_channels: usize,
}

impl CodecHeader {
    pub fn read(b: &[u8]) -> Option<CodecHeader> {
        if b.len() < 20 { return None; }
        Some(CodecHeader { id: b[0], rot_channels: b[1] as usize, trans_channels: b[2] as usize, scale_channels: b[3] as usize })
    }
    /// 1/2/3/8 use the 32-byte fullframe header, 4/5/6/7 the 48-byte keyframe header, 9 the
    /// 32-byte curve header.
    pub fn is_keyframe(&self) -> bool { matches!(self.id, 4 | 5 | 6 | 7) }
    pub fn is_fullframe(&self) -> bool { matches!(self.id, 1 | 2 | 3 | 8) }
    /// Keyframe key TIMES are u8 for codec 4/6 and u16 for 5/7 [engine +0x24628C vs +0x246E40].
    pub fn key_time_bytes(&self) -> usize { if matches!(self.id, 5 | 7) { 2 } else { 1 } }
}

/// Sample ONE codec block's channels at `frame` (integer part + fraction) and write the result
/// into `out`, which is indexed by NODE (not channel). Nodes whose present bit is clear are
/// left untouched.
pub fn sample_codec_block(blk: &[u8], flags: [&[u8]; 3], node_count: usize, frame: f32, out: &mut [NodeChannel]) {
    let Some(h) = CodecHeader::read(blk) else { return };
    if h.id == 0 { return; } // `_no_compression_codec`: every accessor is NULL
    let fc_a = frame.floor().max(0.0) as usize;
    let sub = (frame - frame.floor()).clamp(0.0, 1.0);
    for node in 0..node_count.min(out.len()) {
        if let Some(ch) = channel_of(flags[0], node) {
            if ch < h.rot_channels { out[node].rot = codec_rotation(blk, &h, ch, fc_a, sub); }
        }
        if let Some(ch) = channel_of(flags[1], node) {
            if ch < h.trans_channels { out[node].trans = codec_translation(blk, &h, ch, fc_a, sub); }
        }
        if let Some(ch) = channel_of(flags[2], node) {
            if ch < h.scale_channels { out[node].scale = codec_scale(blk, &h, ch, fc_a, sub); }
        }
    }
}

/// `(key_count, first_key)` of one keyframe channel descriptor u32:
/// 12-bit count, 20-bit first-key index [engine +0x24628C].
fn key_desc(blk: &[u8], desc_array_off: usize, ch: usize) -> (usize, usize) {
    let d = rd_u32(blk, desc_array_off + 4 * ch);
    ((d & 0xFFF) as usize, (d >> 12) as usize)
}

/// Key time `k` of a channel (u8 or u16 table).
fn key_time(blk: &[u8], times_off: usize, width: usize, first_key: usize, k: usize) -> f32 {
    let o = times_off + width * (first_key + k);
    match width {
        2 => u16::from_le_bytes([*blk.get(o).unwrap_or(&0), *blk.get(o + 1).unwrap_or(&0)]) as f32,
        _ => *blk.get(o).unwrap_or(&0) as f32,
    }
}

/// Largest `k` with `times[k] <= t` (the engine's binary search, lo=0 / hi=key_count).
fn find_key(blk: &[u8], times_off: usize, width: usize, first_key: usize, count: usize, t: f32) -> usize {
    if count <= 1 { return 0; }
    let (mut lo, mut hi) = (0usize, count);
    while hi > lo + 1 {
        let m = (hi + lo) >> 1;
        if key_time(blk, times_off, width, first_key, m) > t { hi = m; } else { lo = m; }
    }
    lo
}

// ---------------------------------------------------------------------------------------------
// Codec 9: `_curve_codec` ('WANG')
// ---------------------------------------------------------------------------------------------

/// Which offset table a channel lives in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CurveKind { Rot, Trans, Scale }

/// The 'WANG' tag the engine asserts on (`s_curve_codec_header::k_codec_type_id`).
pub const CURVE_MAGIC: u32 = 0x474E_4157;

/// Resolve one curve channel's record: `P + table[ch]`, where `P = blk + blk[0x14]` and the
/// table for `kind` starts at `P`, `P + blk[0x0C]` or `P + blk[0x10]` (0xFF = section absent).
fn curve_channel<'a>(blk: &'a [u8], h: &CodecHeader, kind: CurveKind, ch: usize) -> Option<&'a [u8]> {
    if blk.len() < 32 || rd_u32(blk, 0x1C) != CURVE_MAGIC { return None; }
    let p = rd_u32(blk, 0x14) as usize;
    if p >= blk.len() { return None; }
    let (tab_rel, count) = match kind {
        CurveKind::Rot => (0usize, h.rot_channels),
        CurveKind::Trans => (rd_u32(blk, 0x0C) as usize, h.trans_channels),
        CurveKind::Scale => (rd_u32(blk, 0x10) as usize, h.scale_channels),
    };
    if count == 0 || ch >= count || tab_rel == 0xFF { return None; }
    let tab = p + tab_rel;
    if tab + 4 * ch + 4 > blk.len() { return None; }
    let at = p + rd_u32(blk, tab + 4 * ch) as usize;
    if at + 8 > blk.len() { return None; }
    Some(&blk[at..])
}

/// `(spline_count, dense)` of a channel record whose type id must be `want`.
fn curve_meta(rec: &[u8], want: u8) -> Option<(usize, bool)> {
    if rec.is_empty() || rec[0] != want { return None; }
    let n = rd_i16(rec, 0x02);
    if n <= 0 { return None; }
    Some((n as usize, rec[0x04] & 1 != 0))
}

/// The engine's segment search (+0x218198): the first segment whose cumulative end reaches
/// `frame`, plus the local parameter `t` in [0,1]. `None` = the frame is outside the channel.
fn curve_segment(dur: &[u8], frame: f32) -> Option<(usize, f32)> {
    if frame < 0.0 { return None; }
    let mut cum = 0.0f32;
    for (i, &d) in dur.iter().enumerate() {
        let d = d as f32;
        let end = cum + d;
        if end >= frame {
            return Some((i, if d > 0.0 { ((frame - cum) / d).clamp(0.0, 1.0) } else { 0.0 }));
        }
        cum = end;
    }
    None
}

/// The lean nibbles of a key byte -> the two Hermite tangents of one component
/// (`m = d + |u| * (u * 0.3)`, `u = index / 7`).
fn curve_tangents(lean: u8, p0: f32, p1: f32) -> (f32, f32) {
    let d = p1 - p0;
    let m = |idx: i32| {
        let u = idx as f32 / 7.0;
        d + u.abs() * (u * 0.3)
    };
    let i0 = ((lean >> 4) as i32 - 7).clamp(-7, 7);
    let i1 = ((lean & 0x0F) as i32 - 7).clamp(-7, 7);
    (m(i0), m(i1))
}

/// Cubic Hermite with the engine's basis.
fn hermite(t: f32, p0: f32, m0: f32, p1: f32, m1: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * p0 + (t3 - 2.0 * t2 + t) * m0 + (3.0 * t2 - 2.0 * t3) * p1 + (t3 - t2) * m1
}

/// The curve codec's 6-byte quaternion (engine +0x218220). The 3rd word's SIGN carries the sign
/// of the reconstructed axis z and its MAGNITUDE carries `w = 2|c| - 1`.
fn curve_quat(b: &[u8], o: usize) -> Quat {
    let k = INV_32767;
    let a = rd_i16(b, o) as f32 * k;
    let bb = rd_i16(b, o + 2) as f32 * k;
    let c = (rd_i16(b, o + 4) as f32 * k).clamp(-1.0, 1.0);
    let mut nz = (1.0f32 - a * a - bb * bb).max(0.0).sqrt();
    if c < 0.0 { nz = -nz; }
    let w = 2.0 * c.abs() - 1.0;
    let sn = (1.0f32 - w * w).max(0.0).sqrt();
    Quat::from_xyzw(sn * a, sn * bb, sn * nz, w)
}

/// Sample a curve ROTATION channel at an integer-or-fractional frame.
fn curve_rotation(blk: &[u8], h: &CodecHeader, ch: usize, frame: f32) -> Option<Quat> {
    let rec = curve_channel(blk, h, CurveKind::Rot, ch)?;
    let (n, dense) = curve_meta(rec, b'R')?;
    if dense {
        // Keys are indexed straight by frame; bracketing keys are blended with the same nlerp
        // the engine's sub-frame accessor uses (+0x218428).
        let f0 = (frame.floor().max(0.0) as usize).min(n);
        let f1 = (f0 + 1).min(n);
        if 0x08 + 6 * f1 + 6 > rec.len() { return None; }
        let a = curve_quat(rec, 0x08 + 6 * f0);
        let t = frame - frame.floor();
        if f1 == f0 || t <= 1e-4 { return Some(norm(a)); }
        return Some(nlerp_keys(a, curve_quat(rec, 0x08 + 6 * f1), 1.0 - t, t));
    }
    let dur = rec.get(0x08..0x08 + n)?.to_vec();
    let (k, t) = curve_segment(&dur, frame)?;
    let keys = 0x08 + n;
    let (o0, o1) = (keys + 10 * k, keys + 10 * (k + 1));
    if o1 + 10 > rec.len() { return None; }
    let a0 = curve_quat(rec, o0).to_array();
    let a1 = curve_quat(rec, o1).to_array();
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let (m0, m1) = curve_tangents(rec[o0 + 6 + i], a0[i], a1[i]);
        out[i] = hermite(t, a0[i], m0, a1[i], m1);
    }
    // `get_rotation` normalises unconditionally (+0x21834C).
    Some(norm(Quat::from_xyzw(out[0], out[1], out[2], out[3])))
}

/// Sample a curve TRANSLATION channel. `base + scale * spline(i16/32767)`.
fn curve_translation(blk: &[u8], h: &CodecHeader, ch: usize, frame: f32) -> Option<Vec3> {
    let rec = curve_channel(blk, h, CurveKind::Trans, ch)?;
    let (n, dense) = curve_meta(rec, b'T')?;
    let base = rd_vec3(rec, 0x08);
    let scale = rd_f32(rec, 0x14);
    let q = |o: usize, i: usize| rd_i16(rec, o + 2 * i) as f32 * INV_32767;
    let v = if dense {
        let f = (frame.round().max(0.0) as usize).min(n);
        let o = 0x18 + 6 * f;
        if o + 6 > rec.len() { return None; }
        Vec3::new(q(o, 0), q(o, 1), q(o, 2))
    } else {
        let dur = rec.get(0x18..0x18 + n)?.to_vec();
        let (k, t) = curve_segment(&dur, frame)?;
        let keys = 0x18 + n;
        let (o0, o1) = (keys + 9 * k, keys + 9 * (k + 1));
        if o1 + 9 > rec.len() { return None; }
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            let (p0, p1) = (q(o0, i), q(o1, i));
            let (m0, m1) = curve_tangents(rec[o0 + 6 + i], p0, p1);
            out[i] = hermite(t, p0, m0, p1, m1);
        }
        Vec3::from(out)
    };
    Some(base + scale * v)
}

/// Sample a curve SCALE channel.
fn curve_scale(blk: &[u8], h: &CodecHeader, ch: usize, frame: f32) -> Option<f32> {
    let rec = curve_channel(blk, h, CurveKind::Scale, ch)?;
    let (n, dense) = curve_meta(rec, b'S')?;
    let base = rd_f32(rec, 0x08);
    let scale = rd_f32(rec, 0x0C);
    let v = if dense {
        let f = (frame.round().max(0.0) as usize).min(n);
        let o = 0x10 + 2 * f;
        if o + 2 > rec.len() { return None; }
        rd_i16(rec, o) as f32 * INV_32767
    } else {
        let dur = rec.get(0x10..0x10 + n)?.to_vec();
        let (k, t) = curve_segment(&dur, frame)?;
        let keys = 0x10 + n;
        let (o0, o1) = (keys + 3 * k, keys + 3 * (k + 1));
        if o1 + 3 > rec.len() { return None; }
        let (p0, p1) = (rd_i16(rec, o0) as f32 * INV_32767, rd_i16(rec, o1) as f32 * INV_32767);
        let (m0, m1) = curve_tangents(rec[o0 + 2], p0, p1);
        hermite(t, p0, m0, p1, m1)
    };
    Some(base + scale * v)
}

fn codec_rotation(blk: &[u8], h: &CodecHeader, ch: usize, frame: usize, sub: f32) -> Option<Quat> {
    match h.id {
        // Codec 1: one static key per channel, quantized, inline at +0x20 [engine +0x2454D0].
        1 => Some(quat_quantized(blk, 0x20 + 8 * ch)),
        // Codec 2 / 8: float4 per frame, channel stride at header +0x14 [engine +0x2458E0].
        2 | 8 => {
            let stride = rd_u32(blk, 0x14) as usize;
            let base = ch * stride + 0x20;
            let n = if stride >= 16 { stride / 16 } else { 1 };
            let f0 = frame.min(n.saturating_sub(1));
            let f1 = (f0 + 1).min(n.saturating_sub(1));
            let a = quat_float(blk, base + 16 * f0);
            if f1 == f0 || sub <= 1e-4 { Some(a) } else { Some(nlerp_keys(a, quat_float(blk, base + 16 * f1), 1.0 - sub, sub)) }
        }
        // Codec 3: i16x4 per frame [engine +0x245BB0].
        3 => {
            let stride = rd_u32(blk, 0x14) as usize;
            let base = ch * stride + 0x20;
            let n = if stride >= 8 { stride / 8 } else { 1 };
            let f0 = frame.min(n.saturating_sub(1));
            let f1 = (f0 + 1).min(n.saturating_sub(1));
            let a = quat_quantized(blk, base + 8 * f0);
            if f1 == f0 || sub <= 1e-4 { Some(a) } else { Some(nlerp_keys(a, quat_quantized(blk, base + 8 * f1), 1.0 - sub, sub)) }
        }
        // Keyframe codecs: descriptor array inline at +0x30, times at +0x14, payload at +0x20.
        // Codec 9: a per-channel Hermite spline (see the module docs).
        9 => curve_rotation(blk, h, ch, frame as f32 + sub),
        4 | 5 | 6 | 7 => {
            let (count, first) = key_desc(blk, 0x30, ch);
            if count == 0 { return None; }
            let times = rd_u32(blk, 0x14) as usize;
            let pay = rd_u32(blk, 0x20) as usize;
            let w = h.key_time_bytes();
            let t = frame as f32 + sub;
            let k = find_key(blk, times, w, first, count, t);
            let a = quat_quantized(blk, pay + 8 * (first + k));
            if k + 1 >= count { return Some(a); }
            let (ta, tb) = (key_time(blk, times, w, first, k), key_time(blk, times, w, first, k + 1));
            if (t - ta).abs() <= 1e-4 { return Some(a); }
            let b = quat_quantized(blk, pay + 8 * (first + k + 1));
            Some(nlerp_keys(a, b, tb - t, t - ta))
        }
        _ => None,
    }
}

fn codec_translation(blk: &[u8], h: &CodecHeader, ch: usize, frame: usize, sub: f32) -> Option<Vec3> {
    match h.id {
        // Codec 1: one key per channel at header[+0x0C] + 12*ch [engine +0x2455C8].
        1 => Some(rd_vec3(blk, rd_u32(blk, 0x0C) as usize + 12 * ch)),
        2 | 3 | 8 => {
            let region = rd_u32(blk, 0x0C) as usize;
            let stride = rd_u32(blk, 0x18) as usize;
            let base = region + ch * stride;
            let n = if stride >= 12 { stride / 12 } else { 1 };
            let f0 = frame.min(n.saturating_sub(1));
            let f1 = (f0 + 1).min(n.saturating_sub(1));
            let a = rd_vec3(blk, base + 12 * f0);
            if f1 == f0 || sub <= 1e-4 { Some(a) } else { Some(a + (rd_vec3(blk, base + 12 * f1) - a) * sub) }
        }
        9 => curve_translation(blk, h, ch, frame as f32 + sub),
        4 | 5 | 6 | 7 => {
            let (count, first) = key_desc(blk, rd_u32(blk, 0x0C) as usize, ch);
            if count == 0 { return None; }
            let times = rd_u32(blk, 0x18) as usize;
            let pay = rd_u32(blk, 0x24) as usize;
            let w = h.key_time_bytes();
            let t = frame as f32 + sub;
            let k = find_key(blk, times, w, first, count, t);
            let a = rd_vec3(blk, pay + 12 * (first + k));
            if k + 1 >= count { return Some(a); }
            let (ta, tb) = (key_time(blk, times, w, first, k), key_time(blk, times, w, first, k + 1));
            if tb <= ta { return Some(a); }
            // Translation/scale use a PROPERLY normalised lerp [engine +0x2465FC].
            let f = ((t - ta) / (tb - ta)).clamp(0.0, 1.0);
            let b = rd_vec3(blk, pay + 12 * (first + k + 1));
            Some(a + (b - a) * f)
        }
        _ => None,
    }
}

fn codec_scale(blk: &[u8], h: &CodecHeader, ch: usize, frame: usize, sub: f32) -> Option<f32> {
    match h.id {
        1 => Some(rd_f32(blk, rd_u32(blk, 0x10) as usize + 4 * ch)),
        2 | 3 | 8 => {
            let region = rd_u32(blk, 0x10) as usize;
            let stride = rd_u32(blk, 0x1C) as usize;
            let base = region + ch * stride;
            let n = if stride >= 4 { stride / 4 } else { 1 };
            let f0 = frame.min(n.saturating_sub(1));
            let f1 = (f0 + 1).min(n.saturating_sub(1));
            let a = rd_f32(blk, base + 4 * f0);
            if f1 == f0 || sub <= 1e-4 { Some(a) } else { Some(a + (rd_f32(blk, base + 4 * f1) - a) * sub) }
        }
        9 => curve_scale(blk, h, ch, frame as f32 + sub),
        4 | 5 | 6 | 7 => {
            let (count, first) = key_desc(blk, rd_u32(blk, 0x10) as usize, ch);
            if count == 0 { return None; }
            let times = rd_u32(blk, 0x1C) as usize;
            let pay = rd_u32(blk, 0x28) as usize;
            let w = h.key_time_bytes();
            let t = frame as f32 + sub;
            let k = find_key(blk, times, w, first, count, t);
            let a = rd_f32(blk, pay + 4 * (first + k));
            if k + 1 >= count { return Some(a); }
            let (ta, tb) = (key_time(blk, times, w, first, k), key_time(blk, times, w, first, k + 1));
            if tb <= ta { return Some(a); }
            let f = ((t - ta) / (tb - ta)).clamp(0.0, 1.0);
            Some(a + (rd_f32(blk, pay + 4 * (first + k + 1)) - a) * f)
        }
        _ => None,
    }
}

/// Sample a whole animation at `frame` (fractional frames interpolate):
///   * region 0/2 = the STATIC (default) pose, applied only for a BASE animation
///     [engine +0x219D70 block A, guarded by `anim_type == 1 && sizes[0]`];
///   * region 1/3 = the ANIMATED pose, always applied.
/// Returns one [`NodeChannel`] per node of the animation's own node list.
pub fn sample(data: &AnimData, anim_type: AnimType, frame: f32) -> Vec<NodeChannel> {
    let nc = data.node_count as usize;
    let mut out = vec![NodeChannel::default(); nc];
    if anim_type == AnimType::Base {
        if let (Some(blk), Some(fl)) = (data.region(0), data.node_flags(2)) {
            sample_codec_block(blk, fl, nc, 0.0, &mut out);
        }
    }
    if let (Some(blk), Some(fl)) = (data.region(1), data.node_flags(3)) {
        let last = (data.frame_count.max(1) - 1) as f32;
        sample_codec_block(blk, fl, nc, frame.clamp(0.0, last), &mut out);
    }
    out
}

/// Compose an OVERLAY animation's channels onto a destination pose, exactly as the engine's
/// full-weight overlay combiner does [engine +0x21DAEC]: rotation = `dst (x) src` (Hamilton,
/// destination on the LEFT, no renormalisation), translation `+=`, scale `*=`.
pub fn apply_overlay(dst: &mut [NodeChannel], src: &[NodeChannel], default_rot: &[Quat], default_trans: &[Vec3]) {
    for (i, s) in src.iter().enumerate() {
        if i >= dst.len() { break; }
        if let Some(sq) = s.rot {
            let d = dst[i].rot.or_else(|| default_rot.get(i).copied()).unwrap_or(Quat::IDENTITY);
            dst[i].rot = Some(d * sq);
        }
        if let Some(st) = s.trans {
            let d = dst[i].trans.or_else(|| default_trans.get(i).copied()).unwrap_or(Vec3::ZERO);
            dst[i].trans = Some(d + st);
        }
        if let Some(ss) = s.scale {
            let d = dst[i].scale.unwrap_or(1.0);
            dst[i].scale = Some(d * ss);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(dwords: &[u32]) -> Vec<u8> { dwords.iter().flat_map(|d| d.to_le_bytes()).collect() }

    /// `s_fullframe_codec_header` of the Spartan's `any:aim_spine` (an OVERLAY animation, 9
    /// frames, 3 rotation channels) — the engine-written bytes of region 1, captured with
    /// `hms-app --dump-anim <map> spartans any:aim_spine`. The whole codec-3 layout follows
    /// from them: the strides are exactly `elem_size * frame_count` and the region size is
    /// exactly `header + channels * stride`.
    #[test]
    fn codec3_fullframe_header_is_strides_times_frame_count() {
        let blk = bytes(&[
            0x0000_0303, // id 3 | 3 rot | 0 trans | 0 scale
            0x3828_7A3B, 0x3F08_D3DD,            // the two unread dwords
            0x0000_00F8, 0x0000_00F8,            // translation / scale region offsets (= size: empty)
            0x0000_0048, 0x0000_006C, 0x0000_0024, // rot / trans / scale per-channel strides
        ]);
        let h = CodecHeader::read(&blk).expect("header");
        assert_eq!((h.id, h.rot_channels, h.trans_channels, h.scale_channels), (3, 3, 0, 0));
        assert!(h.is_fullframe() && !h.is_keyframe());
        const FRAMES: u32 = 9;
        assert_eq!(u32::from_le_bytes([blk[0x14], blk[0x15], blk[0x16], blk[0x17]]), 8 * FRAMES, "rotation stride = 8 B/frame");
        assert_eq!(u32::from_le_bytes([blk[0x18], blk[0x19], blk[0x1A], blk[0x1B]]), 12 * FRAMES, "translation stride = 12 B/frame");
        assert_eq!(u32::from_le_bytes([blk[0x1C], blk[0x1D], blk[0x1E], blk[0x1F]]), 4 * FRAMES, "scale stride = 4 B/frame");
        // The shipped region is 248 bytes: 32-byte header + 3 channels * 72.
        assert_eq!(32 + 3 * 8 * FRAMES as usize, 248);
    }

    /// `s_curve_codec_header` of the Spartan's `combat:rifle:idle` (156 frames) — region 1 is
    /// 1320 bytes of codec 9, and the header says so at +0x18.
    #[test]
    fn codec9_curve_header_carries_wang_and_its_own_size() {
        let blk = bytes(&[
            0x0001_2409, // id 9 | 36 rot | 1 trans | 0 scale  (little-endian: 09 24 01 00)
            0x3CC0_FAF3, 0x3C6B_B08B,  // mean error / mean ratio (informational)
            0x0000_04A0,               // translation offset-table offset
            0x0000_00FF,               // scale table: the "absent" sentinel (0 scale channels)
            0x0000_0020,               // payload base = sizeof(header) = 32
            0x0000_0528,               // total block size = 1320
            CURVE_MAGIC,               // 'WANG'
        ]);
        let h = CodecHeader::read(&blk).expect("header");
        assert_eq!((h.id, h.rot_channels, h.trans_channels, h.scale_channels), (9, 36, 1, 0));
        assert_eq!(u32::from_le_bytes([blk[0x1C], blk[0x1D], blk[0x1E], blk[0x1F]]), 0x474E_4157);
        assert_eq!(u32::from_le_bytes([blk[0x14], blk[0x15], blk[0x16], blk[0x17]]), 32, "payload base");
        assert_eq!(u32::from_le_bytes([blk[0x18], blk[0x19], blk[0x1A], blk[0x1B]]), 1320, "total size");
        assert_eq!(u32::from_le_bytes([blk[0x10], blk[0x11], blk[0x12], blk[0x13]]), 0xFF, "absent scale table");
        // A channel of an absent section must never be dereferenced.
        assert!(curve_channel(&blk, &h, CurveKind::Scale, 0).is_none());
    }

    /// `c_model_animation_tag_resource_member` of the Spartan FP DMR's
    /// `first_person:acceleration_screens` (group 0, member 0): 9 frames, 51 nodes, and the
    /// defining invariant — the 17 region sizes sum to `tag_data.size`.
    #[test]
    fn resource_member_regions_sum_to_the_declared_blob_size() {
        let mut m = vec![0u8; MEMBER_ELEM];
        let put = |m: &mut Vec<u8>, o: usize, v: u32| m[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut m, 0x00, 0);            // animation index (always 0 in a cache)
        put(&mut m, 0x04, 0x6211_7C12);  // checksum
        put(&mut m, 0x08, 0x0033_0009);  // frame count 9 | node count 0x33 = 51 | movement 0
        put(&mut m, 0x0C + 4 * 1, 0x2CC);  // animated codec block
        put(&mut m, 0x0C + 4 * 3, 0x18);   // animated node flags = 3 * bitvector_bytes(51) = 24
        put(&mut m, 0x0C + 4 * 5, 0x6C);   // pill offset = 12 * 9 frames
        put(&mut m, 0x0C + 4 * 7, 0x94);   // blend screen data
        put(&mut m, 0x50, 0x3E4);          // tag_data.size = 996
        let d = AnimData::header_from_member(&m).expect("member");
        assert_eq!(d.frame_count, 9);
        assert_eq!(d.node_count, 51);
        assert_eq!(d.movement_data_type, 0);
        assert_eq!(bitvector_bytes(51), 8, "4 * ceil(51/32)");
        assert_eq!(d.sizes[3] as usize, 3 * bitvector_bytes(51));
        assert_eq!(d.sizes[5] as usize, 12 * d.frame_count as usize);
        assert_eq!(d.total_size(), AnimData::declared_blob_size(&m));
        assert_eq!(d.total_size(), 996);
    }

    #[test]
    fn bitvector_size_and_movement_strides_match_the_engine() {
        // B = 4 * ceil(node_count / 32)  [haloreach.dll +0x219D70]
        assert_eq!(bitvector_bytes(1), 4);
        assert_eq!(bitvector_bytes(32), 4);
        assert_eq!(bitvector_bytes(33), 8);
        assert_eq!(bitvector_bytes(51), 8);
        assert_eq!(bitvector_bytes(82), 12);
        // frame_info_type -> bytes per frame  [haloreach.dll +0x211C54]
        assert_eq!([0, 1, 2, 3, 4].map(movement_stride), [0, 8, 12, 16, 24]);
    }

    /// `channel(node) = popcount(present & ((1 << node) - 1))`  [haloreach.dll +0x219484].
    #[test]
    fn channel_index_is_the_running_popcount_of_the_present_bitvector() {
        // nodes 0, 3, 5 and 40 present.
        let bv = bytes(&[0b0010_1001, 1 << 8]);
        assert_eq!(channel_of(&bv, 0), Some(0));
        assert_eq!(channel_of(&bv, 1), None);
        assert_eq!(channel_of(&bv, 3), Some(1));
        assert_eq!(channel_of(&bv, 5), Some(2));
        assert_eq!(channel_of(&bv, 40), Some(3));
        assert_eq!(channel_of(&bv, 41), None);
    }

    /// The curve codec's 6-byte quaternion (engine +0x218220) is unit length, and the identity
    /// quaternion round-trips: `c = 32767` gives `w = 1`, `s = 0`.
    #[test]
    fn curve_quaternion_unpacks_to_a_unit_quaternion() {
        let key = |a: i16, b: i16, c: i16| {
            let mut v = Vec::new();
            v.extend_from_slice(&a.to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
            v.extend_from_slice(&c.to_le_bytes());
            v
        };
        // c = +32767 -> w = 2*1 - 1 = 1, s = 0: the identity quaternion.
        let q = curve_quat(&key(0, 0, 32767), 0);
        assert!((q.w - 1.0).abs() < 1e-4 && q.xyz().length() < 1e-3, "{q:?}");
        // A 90-degree rotation about +X: q = (sin45, 0, 0, cos45). w = 0.7071 -> c = (w+1)/2,
        // s = sqrt(1-w^2) = 0.7071, and x = s*a so a = 1.
        let c = ((0.70710678f32 + 1.0) / 2.0 * 32767.0).round() as i16;
        let q = curve_quat(&key(32767, 0, c), 0);
        assert!((q.length() - 1.0).abs() < 1e-3, "unit length, got {}", q.length());
        assert!((q.w - 0.70710678).abs() < 2e-4, "w {}", q.w);
        assert!((q.x - 0.70710678).abs() < 2e-3, "x {}", q.x);
        // A NEGATIVE 3rd word flips the reconstructed axis z but keeps |w|.
        let qn = curve_quat(&key(0, 0, -c), 0);
        assert!((qn.w - 0.70710678).abs() < 2e-4 && qn.z < 0.0, "{qn:?}");
    }

    /// A lean index of 0 (nibble 7) makes the Hermite tangents exactly the chord, so the
    /// segment degenerates to LINEAR interpolation — the engine's own degenerate case.
    #[test]
    fn zero_lean_hermite_is_linear() {
        let (m0, m1) = curve_tangents(0x77, 0.0, 1.0);
        assert!((m0 - 1.0).abs() < 1e-6 && (m1 - 1.0).abs() < 1e-6, "{m0} {m1}");
        for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert!((hermite(t, 0.0, m0, 1.0, m1) - t).abs() < 1e-5, "t={t}");
        }
        // The lean dequant: index +7 -> m = d + 1*0.3, index -7 -> m = d - 0.3.
        let (hi, lo) = curve_tangents(0xE0, 0.0, 1.0);
        assert!((hi - 1.3).abs() < 1e-5, "m0 {hi}");
        assert!((lo - 0.7).abs() < 1e-5, "m1 {lo}");
    }

    /// Segment search (engine +0x218198): the first segment whose cumulative end reaches the
    /// frame; outside `[0, sum(durations)]` it FAILS and the engine keeps the existing pose.
    #[test]
    fn curve_segment_search_matches_the_engine() {
        let dur = [4u8, 6, 2];
        assert_eq!(curve_segment(&dur, 0.0), Some((0, 0.0)));
        assert_eq!(curve_segment(&dur, 4.0), Some((0, 1.0)));
        assert_eq!(curve_segment(&dur, 7.0), Some((1, 0.5)));
        assert_eq!(curve_segment(&dur, 12.0), Some((2, 1.0)));
        assert_eq!(curve_segment(&dur, 12.5), None);
        assert_eq!(curve_segment(&dur, -1.0), None);
    }

    #[test]
    fn animation_type_enum_and_frame_rate() {
        assert_eq!(AnimType::from_raw(1), AnimType::Base);
        assert_eq!(AnimType::from_raw(2), AnimType::Overlay);
        assert_eq!(AnimType::from_raw(3), AnimType::Replacement);
        assert_eq!(AnimType::from_raw(0), AnimType::None);
        assert_eq!(FRAME_RATE, 30.0);
        // duration = frame_count * 0.033333335  [haloreach.dll +0x218060]
        let a = AnimEntry {
            index: 0, name_sid: 0, name: String::new(), loop_frame: 0, node_count: 1,
            frame_count: 156, anim_type: AnimType::Base, frame_info_type: 0, compression_setting: 2,
            blend_screen: None,
            production_checksum: 0, resource_group: 0, resource_group_member: 0, heading: Vec3::X,
        };
        assert!((a.duration_seconds() - 5.2).abs() < 1e-4, "{}", a.duration_seconds());
    }
}
